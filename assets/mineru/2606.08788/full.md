# MaskAlign: Token-Subset Representation Alignment for Efficient Diffusion Training

Lianyu Pang\*1,2

Huan Yang2

Tianlin Pan\*1,2,3

Kun Gai2

Cheng Da2

Song Guo1

Changqian Yu2

Wenhan Luo†1

1The Hong Kong University of Science and Technology 2Kuaishou Technology

3University of Chinese Academy of Sciences

∗Equal contribution

†Corresponding author

## Abstract

Representation alignment with pretrained vision models has recently shown strong potential for accelerating diffusion transformer training. By aligning intermediate diffusion features with clean-image representations from self-supervised vision encoders, existing methods improve convergence and generation quality. However, such alignment also introduces a non-trivial constraint: diffusion models operate on noisy inputs whose usable information varies across timesteps, while the reference features are extracted from clean images. In this paper, we revisit this mismatch from a token-level perspective. We find that, under full-token representation alignment, tokens with large alignment-gradient norms exhibit a stable spatial preference, suggesting that the alignment objective does not affect all tokens uniformly and may encourage the model to rely on the complete set of clean-image tokens. To address this issue, we propose MaskAlign, a token-subset representation alignment method that applies alignment to randomly sampled token subsets during training. By exposing the model to different token subsets across iterations, MaskAlign reduces the dependence of representation alignment on the complete token set and encourages alignment behavior that is more stable under token-subset perturbations. To mitigate the information loss caused by directly dropping tokens, we further introduce a lightweight pre-mask token mixing block that shares information across tokens before masking. Experiments on ImageNet 256 × 256 show that MaskAlign consistently improves training convergence and generation quality. On SiT-XL/2, MaskAlign reaches the 8.3 FID level about 77× faster than vanilla SiT-XL/2 and the 5.9 FID level about 30× faster than SiT-XL/2 + REPA, measured by the number of training iterations required to reach the same FID level. It also reduces per-step training time by 11.6% relative to REG, while improving FID from 3.4 to 2.8 at 400K iterations and from 2.7 to 2.4 at 1M iterations.

## Introduction

Diffusion models have advanced significantly in recent years [1, 6, 11, 16, 19, 20, 24]. Latent diffusion models (LDMs) [19] utilize a Variational Autoencoder (VAE) [7] to shift the image generation process from the pixel space to the latent space. DiT [17] improves scalability through a transformer-based architecture, and SiT [14] further enhances performance by employing continuous-time stochastic interpolants. Despite these advances, training high-quality image generation models at scale remains prohibitively expensive, requiring enormous computational resources and training time.

Recent studies have utilized pretrained self-supervised vision models to accelerate diffusion training, as their rich visual features can guide the generative model toward better representations. REPA [28] is a representative method in this direction, directly aligning intermediate diffusion features with those of a vision encoder to improve convergence and generation quality. Following this paradigm, subsequent studies have improved representation-based diffusion training through class tokens [27], shared latent feature coupling [8], VAE-level representation alignment [12], and other alignment-based objectives [18, 23, 26].

![](images/069e1b11c3e8d25478438d728eca6b4d34dc0b5e769cb8e27a0b9391f54e0a1b.jpg)  
Figure 1 MaskAlign generates high-quality ImageNet 256 × 256 samples and reaches comparable FID with substantially fewer training iterations, showing faster convergence.

While these methods have proven highly successful at speeding up diffusion training, representation alignment introduces a non-trivial training constraint. Pretrained vision models usually take clean images as input, so their features encode rich visual and semantic information. In contrast, diffusion models operate on noisy inputs, where the usable information varies with the noise level and the model’s intermediate features shift accordingly. This leads to a potential mismatch: the diffusion model is encouraged to match tokens derived from a clean image, even though its own input is noisy and only partially informative.

We inspect this mismatch at the token level by studying the gradient distribution of the alignment loss, as shown in Figure 2. Figure 2a shows that certain spatial positions are more likely to produce top-10% gradient-norm tokens than others, even after averaging over many images. These high-gradient tokens form a stable spatial pattern, suggesting that the alignment objective does not affect all tokens uniformly. Since the alignment loss is applied to all clean-image tokens unconditionally, it may encourage a feature-fitting shortcut that matches clean feature patterns without ensuring their usefulness under noisy denoising conditions.

Building on these observations, we adopt a dropout-like strategy inspired by random feature dropping for preventing co-adaptation [2, 25]: we randomly mask patch tokens during alignment to reduce shortcuts that rely on the complete token set. By averaging the alignment objective over random token subsets, this strategy disrupts stable patterns of concentrated gradients and encourages alignment signals that remain effective across different subsets. However, directly dropping tokens may disrupt fine-grained spatial patterns. We therefore add a lightweight pre-mask mixing block to share information across tokens before masking.

Figures 2c and 2d show that masked training not only reduces the alignment loss, but also narrows the alignment-loss gap between randomly masked and full-token inputs. This indicates that the learned alignment behavior becomes less sensitive to token-subset perturbations. Figure 1 further reports FID over training steps on ImageNet 256 × 256 [3]. MaskAlign reaches the same FID levels with substantially fewer training iterations: it reaches the 8.3 FID level about 77× faster than vanilla SiT-XL/2 and the 5.9 FID level about 30× faster than SiT-XL/2 + REPA. Here, speedup is measured by the number of training iterations required to reach the same FID level. Together with the lower per-step cost introduced by token masking, these results show that MaskAlign improves both convergence and training efficiency.

In summary, our contributions are as follows:

• We analyze the training behavior of representation alignment at the token level. We find that, under full-token representation alignment, gradients are non-uniformly distributed across patch tokens, with high-gradient tokens exhibiting a stable spatial preference.

![](images/8ea104b42332fe73ce9a9a89a02f11013df68c3495b88108f6c8cea7e11ac616.jpg)

<details>
<summary>natural_image</summary>

Pixelated abstract pattern with dark purple and orange squares (no text or symbols)
</details>

(a) Full-token heatmap

![](images/a635202bd1cecab0bc77d51794ad9b16fad58641b975a458cd878597ec6a93df.jpg)

<details>
<summary>natural_image</summary>

Solid dark purple square with no visible text, symbols, or patterns.
</details>

(b) 25% mask heatmap

![](images/7fe949d9e8a44f202bfaf462441c915c3834c268b3cb05a4aacfbeb04e950e10.jpg)

<details>
<summary>line chart</summary>

| Step   | mixer_ratio=0.0 | mixer_ratio=0.25 |
| ------ | --------------- | ---------------- |
| 0      | 1.4             | 1.35             |
| 25000  | 0.9             | 0.85             |
| 50000  | 0.7             | 0.65             |
| 75000  | 0.6             | 0.55             |
| 100000 | 0.55            | 0.5              |
| 125000 | 0.52            | 0.48             |
| 150000 | 0.5             | 0.45             |
| 175000 | 0.48            | 0.43             |
| 200000 | 0.45            | 0.4              |
</details>

(c) Alignment loss

![](images/a8c5a93168f7255b971c923dd23dbacf86d9f04792aaa2b4a06d42a500037043.jpg)

<details>
<summary>line chart</summary>

| Step   | mixer_ratio=0.0 | mixer_ratio=0.25 |
| ------ | --------------- | ---------------- |
| 25000  | -0.002          | 0.000            |
| 50000  | 0.008           | 0.001            |
| 75000  | 0.010           | 0.0015           |
| 100000 | 0.011           | 0.0018           |
| 125000 | 0.0115          | 0.0019           |
| 150000 | 0.012           | 0.002            |
| 175000 | 0.012           | 0.002            |
| 200000 | 0.012           | 0.002            |
</details>

(d) Alignment-loss gap  
Figure 2 Token-level behavior and alignment stability under token masking. (a,b) Heatmaps show the probability that each spatial position appears among the top-10% alignment-gradient tokens, using the same color range [0, 0.8]. For reference, a uniform distribution would correspond to approximately 10% for each position. (a) Full-token alignment exhibits a stable spatial preference. (b) A 25% mask ratio substantially reduces this concentrated pattern. (c) MaskAlign lowers the full-token alignment loss. (d) $L _ { \mathrm { R E P A } } ^ { \mathrm { m a s k } } - L _ { \mathrm { R E P A } } ^ { \mathrm { f u l l } }$

• We propose MaskAlign, a random token masking strategy that applies alignment to randomly sampled token subsets instead of the complete token set. Motivated by dropout’s ability to prevent co-adaptation, MaskAlign discourages feature-fitting shortcuts and encourages alignment signals that remain stable across different token subsets. We further introduce a lightweight pre-mask token mixer to reduce the information loss caused by directly dropping tokens.  
• We validate the effectiveness of MaskAlign on ImageNet 256 × 256. MaskAlign reaches the same FID levels with substantially fewer training iterations, achieving about 77× faster convergence than vanilla SiT-XL/2 at the 8.3 FID level and about 30× faster convergence than SiT-XL/2 + REPA at the 5.9 FID level. It also reduces the full-token alignment loss and improves alignment stability under token-subset perturbations.

## 2 Related Work

Generative Models for Image Generation. Early methods, such as DDPM [6] and DDIM [24], generate images by denoising directly in the pixel space. In contrast, Latent Diffusion Models (LDMs) [19] first use a VAE [7] to map images into a latent space before performing the denoising process, which significantly improves both training and inference efficiency. Early LDMs [1, 16, 19] utilized U-Net as their foundational architecture. Later, the transformer-based DiT [17] architecture was adopted to enhance scalability. Most recently, SiT [14], which incorporates continuous-time stochastic interpolants, has further improved the training efficiency of LDMs. Despite these significant advancements, training large-scale image generation models remains a challenge that requires substantial computational resources.

Efficient Training via Token Masking. Accelerating the training of LDMs has been a major research focus. Token masking provides a viable solution approach. Methods like MDT [4] and MaskDiT [29] reduce the number of input tokens during training. By forcing the model to predict all tokens from a subset of tokens, these methods encourage the model to better learn the contextual relationships within the image. To mitigate the information loss caused by masking, MicroDiT [22] first uses a lightweight mixer to aggregate token information before applying the mask. Furthermore, TREAD [9] observed minimal output variations across intermediate DiT layers and proposed routing a portion of tokens to skip these layers, thereby avoiding the masking-induced information loss. Different from these methods, we do not use masking as a reconstruction task over missing tokens. Instead, we use random masking to construct token subsets for representation alignment, with the prediction and alignment losses computed on the preserved class token and visible patch tokens.

Representation Alignment with External Models. Representation alignment has recently become an active research direction. REPA [28] observed that DiT models also capture image semantics during training. It proposed aligning the intermediate features of DiTs with the output features of a strong pretrained vision model to improve both training efficiency and final generation performance. Building upon this, REG [27] and ReDi [8] improved the alignment strategy, enabling the model to better learn the semantic information from the pretrained vision model. REPA-E [12] employs the REPA loss to train a VAE model, substantially enhancing the overall generation quality. However, HASTE [26] identified a conflict between the two optimization objectives in REPA. Specifically, in the later stages of training, forcing the intermediate DiT features to align with the output features of an external pretrained vision model can degrade the model’s generation performance. To prevent this degradation, they introduced an early stopping mechanism. Different from these works, we study representation alignment from a token-level perspective and show that random token subsets can improve alignment stability during training.

![](images/b4a809b82e01724d5429d95ce68c56a8100db823b1bb47f654a3c5226a2d17dd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Noisy Image"] --> B["SiT Layers"]
  B --> C["Noisy Tokens"]
  C --> D["L_REPA"]
  D --> E["Clean-image features"]
  E --> F["Pretrained Vision Encoder"]
  F --> G["Clean Image"]
    
  H["Full-token alignment"] --> I["Stable spatial preference"]
  J["MaskAlign"] --> K["Clean-image features"]
  K --> L["Shared Random Mask"]
  L --> M["SiT Layers"]
  M --> N["Alignment on token subsets"]
  N --> O["Concentration largely reduced"]
    
    style A fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style J fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style M fill:#f9f,stroke:#333
    style N fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
    
    subgraph a_Observation_Alignment_mismatch["\"a) Observation: Alignment mismatch\""]
        B
        C
        D
        E
        F
        G
        H
        I
        J
        K
        L
        M
        N
        O
    end
    
    subgraph b_Token_level_observation["\"b) Token-level observation\""]
        I
        J
        K
        L
        M
        N
        O
    end
    
    subgraph c_MaskAlign["\"c) MaskAlign\""]
  P["Noisy Tokens"] --> Q["Pre-mask Token Mixing Block"]
  Q --> R["Clean-image features"]
  R --> S["Shared Random Mask"]
  S --> T["SiT Layers"]
  T --> U["Alignment on token subsets"]
  U --> V["Concentration largely reduced"]
    end
    
    style A fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style J fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style M fill:#f9f,stroke:#333
    style N fill:#f9f, stroke:#333
    style O fill:#f9f,stroke:#333
    style P fill:#f9f,stroke:#333
```
</details>

Figure 3 Overview of MaskAlign. a) Representation alignment matches noisy diffusion tokens with clean-image features extracted by a pretrained vision encoder, leading to a potential mismatch across denoising timesteps. b) Full-token alignment exhibits a stable spatial preference, where high-gradient tokens concentrate at specific spatial positions. c) MaskAlign first applies pre-mask token mixing and then uses a shared random mask to compute representation alignment on token subsets while preserving the class token. With a 25% mask ratio, MaskAlign substantially reduces the concentrated spatial pattern.

## 3 Preliminaries

Denoising Diffusion Probabilistic Models (DDPM). As a prominent family of generative models, diffusion models [6, 17, 24] synthesize high-fidelity images through a process of iterative denoising. Under the common noise-prediction parameterization, the training objective minimizes the distance between the injected noise and the network prediction:

$$
\mathcal {L} _ {\text { diffusion }} = \mathbb {E} _ {z, c, \varepsilon , t} \left[ \| \varepsilon - \varepsilon_ {\theta} (z _ {t}, t, c) \| _ {2} ^ {2} \right], \tag {1}
$$

Here, the network $\varepsilon _ { \theta }$ predicts the noise added to the corrupted input $z _ { t } ,$ , conditioned on the timestep ?? and the context vector ??.

Scalable Interpolant Transformers (SiT). Our method follows the SiT framework [14], which is derived from the stochastic interpolant formulation [13]. Let $z _ { * }$ denote a clean image, and let a pretrained VAE encoder $\mathcal { E } _ { z }$ map it into the latent space as $z _ { 0 } \in \mathbb { R } ^ { D _ { z } \times H _ { z } \times W _ { z } }$ . Based on this latent representation, we construct a continuous-time interpolation process defined as:

$$
z _ {t} = \alpha_ {t} z _ {0} + \sigma_ {t} \epsilon_ {z}, \quad \epsilon_ {z} \sim \mathcal {N} (0, I), t \in [ 0, 1 ] \tag {2}
$$

where the coefficients satisfy boundary conditions $\alpha _ { 0 } = \sigma _ { 1 } = 1$ and $\alpha _ { 1 } = \sigma _ { 0 } = 0$ . As ?? increases, $\alpha _ { t }$ decreases while $\sigma _ { t }$ increases accordingly.

The SiT model adopts a Transformer architecture composed of ?? stacked blocks to learn a velocity function $\nu _ { \theta } ( z _ { t } , t )$ . Training is carried out by minimizing the following velocity matching objective:

$$
\mathcal {L} _ {\mathrm{SiT}} = \mathbb {E} _ {z, \epsilon_ {z}, t} \left[ \| v _ {\theta} (z _ {t}, t) - \dot {\alpha} _ {t} z _ {0} - \dot {\sigma} _ {t} \epsilon_ {z} \| _ {2} ^ {2} \right]. \tag {3}
$$

In our implementation, we use a linear parameterization $\alpha _ { t } = 1 - t$ and $\sigma _ { t } = t$ , which results in constant time derivatives $\dot { \alpha } _ { t } = - 1$ and $\dot { \sigma } _ { t } = 1$ , unless stated otherwise.

## 4 Token-level Analysis

## 4.1 Alignment-Gradient Distribution

Representation alignment trains a diffusion model by matching its intermediate features with clean-image representations extracted by a pretrained vision encoder. However, the diffusion model operates on noisy inputs, where the usable information varies with the noise level. At different timesteps, the model may rely on different visual cues, from coarse structures under high noise levels to finer details under lower noise levels. In contrast, the reference features are always extracted from clean images. This creates a potential mismatch between the clean-image reference features and the model’s noisy intermediate features. We therefore inspect this mismatch at the token level by analyzing the gradient distribution of the alignment loss.

We first consider the full-token alignment setting, where all patch tokens are aligned with their corresponding clean-image reference features. Since the class token has no spatial position, our token-level heatmap analysis focuses on patch tokens. Given the hidden state $h _ { i } ^ { [ \ell _ { a } ] }$ at layer $\ell _ { a }$ and the reference feature $r _ { i } .$ , the alignment loss is defined as

$$
\mathcal {L} _ {\mathrm{REPA}} = - \frac {1}{N} \sum_ {i = 1} ^ {N} \operatorname{sim} \left(r _ {i}, h _ {\phi} (h _ {i} ^ {[ \ell_ {a} ]})\right), \tag {4}
$$

where $h _ { \phi } ( \cdot )$ is the alignment projector. We omit the expectation over samples and timesteps for simplicity.

To examine how this objective affects training, we analyze the gradient norms of $\mathcal { L } _ { \mathrm { R E P A } }$ with respect to the hidden states at layer $\ell _ { a }$ . We focus on this layer because it is where the alignment supervision is explicitly injected through the projector.

Let $h _ { i } ^ { [ \ell _ { a } ] }$ denote the hidden state of the ??-th patch token at the alignment layer $\ell _ { a }$ . We compute the alignment-gradient norm for each patch token as

$$
g _ {i} ^ {\text { align }} = \left\| \frac {\partial \mathcal {L} _ {\mathrm{REPA}}}{\partial h _ {i} ^ {[ \ell_ {a} ]}} \right\| _ {2}. \tag {5}
$$

For each image, we select the top-?? patch tokens with the largest gradient norms. We then compute the probability that each spatial position appears in this top-?? set across multiple images.

As shown in Figure 2a, certain spatial positions remain more likely to appear in the top-?? set, even after averaging over many images. The largest spatial probability is about 21× the smallest, suggesting that this preference cannot be explained by minor random fluctuations. This indicates that the alignment-loss gradients are not uniformly distributed: tokens with large gradient norms tend to concentrate at certain spatial positions. Therefore, we seek to reduce the dependence of representation alignment on the complete token set.

## 4.2 Motivation for Token-Subset Alignment

The token-level observation above suggests that full-token alignment may repeatedly reinforce high-gradient tokens at certain spatial positions. Since the reference features are extracted from clean images, the model may learn feature-fitting shortcuts that reduce the alignment loss for the complete token set but do not remain consistently useful under noisy denoising conditions.

Building on this observation, MaskAlign applies random token masking during representation alignment. Motivated by random feature dropping for preventing co-adaptation [2, 25], we randomly sample patch-token subsets during training. As the visible token subsets vary across iterations, shortcuts that rely on the complete token set are less consistently reinforced. The model is therefore encouraged to rely on alignment signals that remain stable across different random token subsets.

## 5 MaskAlign

## 5.1 Framework

The overall framework of MaskAlign is shown in Figure 3. Following REG, MaskAlign prepends a class token with global semantics to the patch tokens. During training, the class token is always preserved, and representation alignment is applied to this token together with a randomly sampled subset of patch tokens. Before masking, we apply lightweight pre-mask token mixing to share information across tokens and mitigate the disruption from dropping patch tokens. The mixed class token and visible patch tokens are then fed into the diffusion transformer. Random token masking is used only during training; at inference time, all tokens are retained.

Pre-mask Token Mixing and Random Masking. Following Sec. 3, let $z _ { * }$ denote a clean image and let $z _ { 0 } = \mathcal { E } _ { z } ( z _ { * } ) \in$ $\mathbb { R } ^ { D _ { z } \times H _ { z } \times W _ { z } }$ be its clean latent. At timestep ??, the noisy latent is constructed as

$$
z _ {t} = \alpha_ {t} z _ {0} + \sigma_ {t} \epsilon_ {z}, \quad \epsilon_ {z} \sim \mathcal {N} (0, I). \tag {6}
$$

$z _ { t }$ $\boldsymbol { x } _ { t } ^ { 0 } = \{ x _ { t . 1 } ^ { 0 } , \boldsymbol { \cdot } \boldsymbol { \cdot } \cdot , x _ { t , N } ^ { 0 } \} \in \mathbb { R } ^ { N \times D }$ is the number of patch tokens and $D$ is the hidden dimension. Following REG, we prepend a class token $c _ { t } ^ { 0 }$ to form $H _ { t } ^ { 0 } = [ c _ { t } ^ { 0 } , x _ { t } ^ { 0 } ] \in \bar { \mathbb { R } } ^ { ( N + 1 ) \times D }$ .

Before random token masking, we apply a lightweight pre-mask token mixing block $M _ { \psi } ( \cdot )$ to share information across tokens:

$$
\bar {H} _ {t} ^ {0} = \left[ \bar {c} _ {t} ^ {0}, \bar {x} _ {t} ^ {0} \right] = M _ {\psi} (H _ {t} ^ {0}, t, y), \tag {7}
$$

where ?? denotes the class condition. This step mitigates the disruption caused by directly dropping patch tokens.

We then sample a binary keep mask $m \in \{ 0 , 1 \} ^ { N }$ over patch tokens, where $m _ { i } = 1$ indicates that the ??-th patch token is visible. Let $S ( m ) = \{ i \mid m _ { i } = 1 \}$ denote the visible patch-token indices, with $N _ { m } = \vert { \cal { S } } ( m ) \vert$ |. The class token is always preserved, while random masking is applied only to patch tokens:

$$
\widetilde {H} _ {t} ^ {0} (m) = \left[ \bar {c} _ {t} ^ {0}, \{\bar {x} _ {t, i} ^ {0} \} _ {i \in S (m)} \right] \in \mathbb {R} ^ {(1 + N _ {m}) \times D}, \tag {8}
$$

where [ · ] denotes sequence concatenation. The masked sequence $\widetilde { H } _ { t } ^ { 0 } ( m )$ is then fed into the following SiT blocks. At layer ℓ, the transformer produces

$$
H _ {t} ^ {[ \ell ]} (m) = \left[ h _ {t, \mathrm{cls}} ^ {[ \ell ]} (m), \{h _ {t, i} ^ {[ \ell ]} (m) \} _ {i \in S (m)} \right]. \tag {9}
$$

Training Losses. After random token masking, the prediction loss is computed on the preserved class token and the visible patch tokens. Let $\boldsymbol { r ^ { * } } = \{ r _ { \mathrm { c l s } } , r _ { 1 } , . . . , r _ { N } \}$ be the reference representation extracted from the clean image by the pretrained vision encoder, where $r _ { \mathrm { c l s } }$ denotes the projected clean class token. Following $\mathbf { R E G }$ , we construct the noisy class token as $c _ { t } ^ { 0 } = \alpha _ { t } r _ { \mathrm { c l s } } + \sigma _ { t } \epsilon _ { \mathrm { c l s } }$ , with target velocity $\nu _ { \mathrm { c l s } } ^ { * } ( t ) = \dot { \alpha } _ { t } r _ { \mathrm { c l s } } + \dot { \sigma } _ { t } \epsilon _ { \mathrm { c l s } }$ . For each visible patch token $i \in S ( m )$ , let $\hat { \nu } _ { i } ( m , t )$ and $\nu _ { i } ^ { * } ( t ) = \dot { \alpha } _ { t } z _ { 0 , i } + \dot { \sigma } _ { t } \epsilon _ { z , i }$ denote the predicted and target velocities. For the class token, let $\hat { \nu } _ { \mathrm { c l s } } ( m , t )$ denote the predicted velocity. We weight the class-token prediction loss by $\beta \colon$

$$
\mathcal {L} _ {\text { pred }} = \mathbb {E} _ {z ^ {*}, \epsilon_ {z}, \epsilon_ {\mathrm{cls}}, t, m} \left[ \frac {1}{N _ {m}} \sum_ {i \in S (m)} \left\| \hat {v} _ {i} (m, t) - v _ {i} ^ {*} (t) \right\| _ {2} ^ {2} + \beta \left\| \hat {v} _ {\mathrm{cls}} (m, t) - v _ {\mathrm{cls}} ^ {*} (t) \right\| _ {2} ^ {2} \right]. \tag {10}
$$

At alignment layer $\ell _ { a }$ , we define the visible alignment index set as ${ \mathcal { A } } ( m ) = \{ \operatorname { c l s } \} \cup S ( m )$ , where the class token is always included. The projector $h _ { \phi } ( \cdot )$ maps hidden states into the reference feature space. For each $a \in \mathcal { A } ( m )$ , let $r _ { a }$ $h _ { t , a } ^ { [ \ell _ { a } ] } ( m )$

$$
\mathcal {L} _ {\text { REPA }} := - \mathbb {E} _ {z _ {*}, \epsilon_ {z}, t, m} \left[ \frac {1}{| \mathcal {A} (m) |} \sum_ {a \in \mathcal {A} (m)} \operatorname{sim} \left(r _ {a}, h _ {\phi} (h _ {t, a} ^ {[ \ell_ {a} ]} (m))\right) \right], \tag {11}
$$

where $| \mathcal { A } ( m ) | = N _ { m } + 1$ . The final training objective is $\mathcal { L } _ { \mathrm { t o t a l } } = \mathcal { L } _ { \mathrm { p r e d } } + \lambda \mathcal { L } _ { \mathrm { R E P A } }$ , where ?? controls the strength of representation alignment.

## 5.2 Measuring Alignment Stability

To assess alignment stability under token-subset perturbations, we compare the full-token and masked-input alignment $\mathcal { L } _ { \mathrm { R E P A } } ^ { \mathrm { f u l l } }$ $\mathcal { L } _ { \mathrm { R E P A } } ^ { \mathrm { m a s k } }$ denote the alignment loss computed using the class token and a randomly sampled subset of patch tokens. We define the alignment-loss gap as

Table 1 FID comparison during training on ImageNet $2 5 6 \times 2 5 6$ without CFG.

<table><tr><td>Method</td><td>#Params</td><td>Iter.</td><td>FID↓</td></tr><tr><td>SiT-B/2</td><td>130M</td><td>400K</td><td>33.0</td></tr><tr><td>REPA</td><td>130M</td><td>400K</td><td>24.4</td></tr><tr><td>REG</td><td>132M</td><td>400K</td><td>15.2</td></tr><tr><td>MaskAlign</td><td>154M</td><td>400K</td><td>14.8</td></tr><tr><td>SiT-XL/2</td><td>675M</td><td>7M</td><td>8.3</td></tr><tr><td>REPA</td><td>675M</td><td>150K</td><td>13.6</td></tr><tr><td>REPA + MaskAlign</td><td>728M</td><td>150K</td><td>10.8</td></tr><tr><td>REPA</td><td>675M</td><td>200K</td><td>11.1</td></tr><tr><td>ReDi</td><td>675M</td><td>200K</td><td>12.5</td></tr><tr><td>REG</td><td>677M</td><td>200K</td><td>5.0</td></tr><tr><td>MaskAlign</td><td>732M</td><td>200K</td><td>4.0</td></tr><tr><td>REPA</td><td>675M</td><td>400K</td><td>7.9</td></tr><tr><td>ReDi</td><td>675M</td><td>400K</td><td>7.5</td></tr><tr><td>REG</td><td>677M</td><td>400K</td><td>3.4</td></tr><tr><td>MaskAlign</td><td>732M</td><td>400K</td><td>2.8</td></tr><tr><td>REPA</td><td>675M</td><td>1M</td><td>6.4</td></tr><tr><td>ReDi</td><td>675M</td><td>1M</td><td>5.1</td></tr><tr><td>REG</td><td>677M</td><td>1M</td><td>2.7</td></tr><tr><td>MaskAlign</td><td>732M</td><td>1M</td><td>2.4</td></tr><tr><td>REG</td><td>677M</td><td>2.4M</td><td>2.2</td></tr><tr><td>MaskAlign</td><td>732M</td><td>2.4M</td><td>2.1</td></tr></table>

Table 2 Comparison with state-of-the-art methods on ImageNet $2 5 6 \times 2 5 6$ with CFG.

<table><tr><td>Model</td><td>Epochs</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td><td>Prec.↑</td><td>Rec.↑</td></tr><tr><td colspan="7">Autoregressive Models</td></tr><tr><td>VAR</td><td>350</td><td>1.80</td><td>-</td><td>365.4</td><td>0.83</td><td>0.57</td></tr><tr><td>MagViTv2</td><td>1080</td><td>1.78</td><td>-</td><td>319.4</td><td>0.83</td><td>0.57</td></tr><tr><td>MAR</td><td>800</td><td>1.55</td><td>-</td><td>303.7</td><td>0.81</td><td>0.62</td></tr><tr><td colspan="7">Latent Diffusion Models</td></tr><tr><td>LDM</td><td>200</td><td>3.60</td><td>-</td><td>247.7</td><td>0.87</td><td>0.48</td></tr><tr><td>U-ViT-H/2</td><td>240</td><td>2.29</td><td>5.68</td><td>263.9</td><td>0.82</td><td>0.57</td></tr><tr><td>DiT-XL/2</td><td>1400</td><td>2.27</td><td>4.60</td><td>278.2</td><td>0.83</td><td>0.57</td></tr><tr><td>MaskDiT</td><td>1600</td><td>2.28</td><td>5.67</td><td>276.6</td><td>0.80</td><td>0.61</td></tr><tr><td>SD-DiT</td><td>480</td><td>3.23</td><td>-</td><td>270.3</td><td>0.82</td><td>0.59</td></tr><tr><td>SiT-XL/2</td><td>1400</td><td>2.06</td><td>4.50</td><td>270.3</td><td>0.82</td><td>0.59</td></tr><tr><td>FasterDiT</td><td>400</td><td>2.03</td><td>4.63</td><td>264.0</td><td>0.81</td><td>0.60</td></tr><tr><td>MDTV2</td><td>1080</td><td>1.58</td><td>4.52</td><td>317.7</td><td>0.79</td><td>0.65</td></tr><tr><td colspan="7">Leveraging Visual Representations</td></tr><tr><td>REG</td><td>80</td><td>1.86</td><td>4.49</td><td>321.4</td><td>0.76</td><td>0.63</td></tr><tr><td>MaskAlign</td><td>80</td><td>1.82</td><td>4.48</td><td>310.0</td><td>0.81</td><td>0.63</td></tr><tr><td>REG</td><td>160</td><td>1.59</td><td>4.36</td><td>304.6</td><td>0.77</td><td>0.65</td></tr><tr><td>MaskAlign</td><td>160</td><td>1.56</td><td>4.37</td><td>304.1</td><td>0.79</td><td>0.65</td></tr><tr><td>ReDi</td><td>800</td><td>1.61</td><td>4.66</td><td>295.1</td><td>0.78</td><td>0.64</td></tr><tr><td>REPA</td><td>800</td><td>1.42</td><td>4.70</td><td>305.7</td><td>0.80</td><td>0.65</td></tr><tr><td>REG</td><td>800</td><td>1.36</td><td>4.25</td><td>299.4</td><td>0.77</td><td>0.66</td></tr><tr><td>MaskAlign</td><td>800</td><td>1.35</td><td>4.31</td><td>312.9</td><td>0.78</td><td>0.67</td></tr></table>

$$
G _ {r} = \mathcal {L} _ {\text { REPA }} ^ {\text { mask }} - \mathcal {L} _ {\text { REPA }} ^ {\text { full }}. \tag {12}
$$

A smaller $G _ { r }$ indicates that the alignment loss is less sensitive to token-subset perturbations, and thus the learned alignment behavior is more stable across random token subsets.

Figures 2c and 2d report the full-token alignment loss and the alignment-loss gap ???? for REG and MaskAlign under a 25% mask ratio. At 200K steps, the gap of MaskAlign is only 13.8% of that of REG, showing that MaskAlign is much less sensitive to token-subset perturbations. In contrast, the larger gap of REG suggests stronger dependence on the complete token set. These results provide evidence that random token masking encourages more stable alignment behavior under token-subset perturbations.

## 6 Experiments

## 6.1 Experimental Setup

Implementation Details. We follow the standard training procedures of SiT and REG. We conduct experiments on ImageNet, where all images are center-cropped and resized to 256 × 256 following the ADM preprocessing protocol. Each image is then encoded into a latent representation ?? using the Stable Diffusion VAE. We adopt SiT-B/2 and SiT-XL/2 as the backbone architecture. For fair comparison, we use a fixed batch size of 256 and adopt the same learning rate and exponential moving average (EMA) settings as REG. More implementation details are provided in the Appendix.

Evaluation Protocol. To evaluate image generation quality from multiple aspects, we report a set of standard quantitative metrics. Specifically, we use Fréchet Inception Distance (FID) [5] to measure sample realism, structural FID (sFID) [15] to evaluate spatial coherence, and Inception Score (IS) [21] to assess class-conditional diversity. We also report precision (Prec.) to measure sample fidelity and recall (Rec.) [10] to evaluate coverage of the target distribution. All metrics are computed using 50K generated images for reliable evaluation. Following REPA, we use the SDE Euler-Maruyama solver with 250 sampling steps. Full details of the evaluation protocol are provided in the Appendix.

Accelerating Training Convergence. Table 1 reports the FID scores of different alignment-based training methods on ImageNet 256 × 256 without classifier-free guidance (CFG). Across different backbones and training budgets, our method consistently achieves the best FID among methods evaluated at the same number of training iterations, showing its effectiveness in accelerating training convergence.

Table 3 Ablation study on token masking and token mixing. All experiments are conducted on ImageNet 256 × 256 using SiT-XL/2 models trained for 600K iterations without CFG.

<table><tr><td>Method</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td></tr><tr><td>MaskAlign</td><td>2.67</td><td>4.79</td><td>198.10</td></tr><tr><td>w/o Mixing</td><td>3.54</td><td>6.65</td><td>194.51</td></tr><tr><td>w/o Masking</td><td>3.20</td><td>4.92</td><td>188.84</td></tr><tr><td>w/o Both</td><td>3.01</td><td>4.88</td><td>193.16</td></tr></table>

Table 4 Computational cost and performance comparison on ImageNet 256 × 256 at 400K training iterations. Time denotes the average training time per iteration in seconds. Both methods use the SiT-XL/2 backbone and the same GPU hardware.

<table><tr><td>Method</td><td>Params</td><td>Time</td><td>Tokens</td><td>FID ↓</td></tr><tr><td>REG</td><td>677M</td><td>0.359</td><td>257</td><td>3.4</td></tr><tr><td>Ours</td><td>732M</td><td>0.317</td><td>193</td><td>2.8</td></tr></table>

Figure 1 further compares the convergence curves of SiT-XL/2, SiT-XL/2 + REPA, and SiT-XL/2 + MaskAlign. To make the speedup comparison explicit, we measure the number of training iterations required to reach the same FID level. MaskAlign reaches the 8.3 FID level about 77× faster than vanilla SiT-XL/2, and reaches the 5.9 FID level about 30× faster than SiT-XL/2 + REPA. This shows that MaskAlign does not merely improve FID at fixed training budgets, but also reaches comparable generation quality with substantially fewer training iterations.

On SiT-B/2, our method improves REG from 15.2 to 14.8 FID at 400K iterations. On the larger SiT-XL/2 backbone, our method also brings consistent gains over REG, reducing FID from 5.0 to 4.0 at 200K iterations, from 3.4 to 2.8 at 400K iterations, and from 2.7 to 2.4 at 1M iterations. At the longer 2.4M training budget, our method further improves the FID from 2.2 to 2.1. These results indicate that MaskAlign remains effective from early training stages to longer training schedules. In addition, our method is not limited to REG. When applied to REPA, our method reduces the FID from 13.6 to 10.8 at 150K iterations, demonstrating that random token-subset alignment can also improve standard representation alignment. More experimental comparisons are provided in the Appendix.

Comparison with SOTA Methods. Table 2 compares MaskAlign with recent generative models on ImageNet 256×256 with classifier-free guidance (CFG). MaskAlign achieves competitive performance while requiring substantially fewer training epochs than many prior diffusion-transformer baselines. At 80 epochs, MaskAlign improves REG from 1.86 to 1.82 FID and increases precision from 0.76 to 0.81, while maintaining the same recall. This model already achieves lower FID than the vanilla SiT-XL/2 trained for 1,400 epochs. At 160 epochs, MaskAlign further improves REG from 1.59 to 1.56 FID and increases precision from 0.77 to 0.79. Under the 800-epoch schedule, MaskAlign reaches 1.35 FID, slightly improving over REG and achieving higher IS and recall. These results indicate that token-subset representation alignment provides consistent gains under both short and long training schedules.

Computational Cost Comparison. Table 4 compares REG and MaskAlign at 400K training iterations using the same SiT-XL/2 backbone and GPU hardware. Although MaskAlign introduces about 8% more parameters, random token masking reduces the number of input tokens from 257 to 193 and lowers the training time per step from 0.359s to 0.317s. This corresponds to a 24.9% reduction in tokens and an 11.6% reduction in time. Together with the faster convergence shown in Figure 1, this indicates that MaskAlign improves training efficiency from two aspects: it reaches the same FID level with fewer iterations and also reduces the per-step training cost. Meanwhile, MaskAlign improves FID from 3.4 to 2.8, demonstrating better sample quality with lower per-step computational cost.

## 6.2 Ablation

Effect of Token Masking and Token Mixing. We ablate the effects of pre-mask token mixing and random token masking by removing each component separately. As shown in Table 3, the full model achieves the best performance across all metrics, indicating that both components are important for MaskAlign. Removing pre-mask token mixing leads to the worst FID and sFID, suggesting that directly applying random masking without first sharing information across tokens can severely disrupt the input token representations. Removing random masking also degrades performance, reducing the method to a token-mixing-only variant that performs worse than the baseline. These results show that token mixing and random masking are complementary: pre-mask token mixing reduces the information loss caused by dropping tokens, while random masking provides the token-subset training signal needed for more stable alignment.

Effect of Mask Ratio. We study the effect of the mask ratio by training models with different ratios for 400K iterations. As shown in Table 5, a moderate mask ratio of 0.25 achieves the best performance, reducing FID from 3.52 without masking to 2.84. Increasing the mask ratio to 0.5 weakens the improvement, while an excessively high mask ratio of 0.75 severely degrades performance. These results suggest that random token masking should provide sufficient token-subset perturbations to regularize alignment, while still preserving enough input information for stable training.

Effect of Mixing Layers. We study the effect of the number of pre-mask token mixing layers. As shown in Table 6, using two mixing layers achieves the best performance, reducing FID to 2.84. With only one mixing layer, the model obtains a higher FID of 3.23, suggesting that insufficient token mixing cannot fully compensate for the information disruption caused by random masking. Increasing the number of mixing layers to three also degrades performance, likely because excessive mixing alters the effective depth of the aligned representation and weakens the alignment supervision.

These results indicate that a lightweight pre-mask token mixing block is sufficient.

Table 5 Ablation study on the mask ratio. All models are trained for 400K iterations without CFG.

<table><tr><td>Mask Ratio</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td></tr><tr><td>0</td><td>3.52</td><td>4.90</td><td>184.13</td></tr><tr><td>0.25 (Ours)</td><td>2.84</td><td>4.85</td><td>194.57</td></tr><tr><td>0.5</td><td>3.15</td><td>5.08</td><td>188.38</td></tr><tr><td>0.75</td><td>5.82</td><td>5.29</td><td>152.28</td></tr></table>

Table 6 Ablation study on the number of pre-mask token mixing layers. All models are trained for 400K iterations without CFG.

<table><tr><td>Mixing Layers</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td></tr><tr><td>1</td><td>3.23</td><td>4.93</td><td>188.49</td></tr><tr><td>2 (Ours)</td><td>2.84</td><td>4.85</td><td>194.57</td></tr><tr><td>3</td><td>3.02</td><td>4.88</td><td>192.54</td></tr></table>

## 7 Conclusion

In this paper, we present MaskAlign, a token-subset representation alignment method for efficient diffusion transformer training. Motivated by the mismatch between noisy diffusion features and clean-image reference representations, we analyze full-token alignment at the token level and observe a stable spatial preference among tokens with large alignment-gradient norms, suggesting that full-token alignment may encourage feature-fitting shortcuts that depend on the complete token set. To address this issue, MaskAlign applies representation alignment to randomly sampled token subsets and uses a lightweight pre-mask token mixing block to reduce the information loss caused by directly dropping tokens. Experiments on ImageNet 256 × 256 show that MaskAlign improves alignment stability under token-subset perturbations, accelerates training convergence, and achieves better generation quality with lower per-step computational cost.

Limitations. Despite these encouraging results, our study is mainly evaluated on ImageNet 256 × 256 with SiT-based backbones and pretrained DINOv2 features, and its generality to higher-resolution generation, text-to-image generation, and other teacher representations remains to be further explored. In addition, MaskAlign depends on design choices such as the mask ratio and the number of pre-mask token mixing layers, where overly aggressive masking or excessive mixing can degrade performance. Future work may investigate adaptive masking strategies and broader model families to better understand the scope and robustness of token-subset representation alignment.

## References

[1] Yogesh Balaji, Seungjun Nah, Xun Huang, Arash Vahdat, Jiaming Song, Qinsheng Zhang, Karsten Kreis, Miika Aittala, Timo Aila, Samuli Laine, Bryan Catanzaro, Tero Karras, and Ming-Yu Liu. ediff-i: Text-to-image diffusion models with an ensemble of expert denoisers. arXiv preprint arXiv:2211.01324, 2022.  
[2] Pierre Baldi and Peter J Sadowski. Understanding dropout. Advances in neural information processing systems, 26, 2013.  
[3] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.  
[4] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Mdtv2: Masked diffusion transformer is a strong image synthesizer. arXiv preprint arXiv:2303.14389, 2023.  
[5] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.  
[6] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. In NeurIPS, 2020.  
[7] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.  
[8] Theodoros Kouzelis, Efstathios Karypidis, Ioannis Kakogeorgiou, Spyros Gidaris, and Nikos Komodakis. Boosting generative image modeling via joint image-feature synthesis. arXiv preprint arXiv:2504.16064, 2025.  
[9] Felix Krause, Timy Phan, Ming Gui, Stefan Andreas Baumann, Vincent Tao Hu, and Björn Ommer. Tread: Token routing for efficient architecture-agnostic diffusion training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15703–15713, 2025.  
[10] Tuomas Kynkäänniemi, Tero Karras, Samuli Laine, Jaakko Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019.  
[11] Black Forest Labs. Flux. https://github.com/black-forest-labs/flux, 2024.  
[12] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning of latent diffusion transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18262–18272, 2025.  
[13] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.  
[14] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024.  
[15] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter W Battaglia. Generating images with sparse representations. arXiv preprint arXiv:2103.03841, 2021.  
[16] Alex Nichol, Prafulla Dhariwal, Aditya Ramesh, Pranav Shyam, Pamela Mishkin, Bob McGrew, Ilya Sutskever, and Mark Chen. Glide: Towards photorealistic image generation and editing with text-guided diffusion models. arXiv preprint arXiv:2112.10741, 2021.  
[17] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023.  
[18] Giorgos Petsangourakis, Christos Sgouropoulos, Bill Psomas, Theodoros Giannakopoulos, Giorgos Sfikas, and Ioannis Kakogeorgiou. Reglue your latents with global and local semantics for entangled diffusion. arXiv preprint arXiv:2512.16636, 2025.  
[19] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.  
[20] Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L Denton, Kamyar Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, et al. Photorealistic text-to-image diffusion models with deep language understanding. In NeurIPS, 2022.  
[21] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016.  
[22] Vikash Sehwag, Xianghao Kong, Jingtao Li, Michael Spranger, and Lingjuan Lyu. Stretching each dollar: Diffusion training from scratch on a micro-budget. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 28596–28608, 2025.  
[23] Jaskirat Singh, Xingjian Leng, Zongze Wu, Liang Zheng, Richard Zhang, Eli Shechtman, and Saining Xie. What matters for representation alignment: Global information or spatial structure? arXiv preprint arXiv:2512.10794, 2025.  
[24] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.  
[25] Stefan Wager, Sida Wang, and Percy S Liang. Dropout training as adaptive regularization. Advances in neural information processing systems, 26, 2013.  
[26] Ziqiao Wang, Wangbo Zhao, Yuhao Zhou, Zekai Li, Zhiyuan Liang, Mingjia Shi, Xuanlei Zhao, Pengfei Zhou, Kaipeng Zhang, Zhangyang Wang, et al. Repa works until it doesn’t: Early-stopped, holistic alignment supercharges diffusion training. arXiv preprint arXiv:2505.16792, 2025.  
[27] Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, et al. Representation entanglement for generation: Training diffusion transformers is much easier than you think. arXiv preprint arXiv:2507.01467, 2025.  
[28] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024.  
[29] Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305, 2023.

## A Experimental Setup

Table 7 summarizes the hyperparameter settings of MaskAlign for SiT-B/2 and SiT-XL/2. Following the experimental protocol of REPA, we train models in the latent space with v-prediction and use the Euler-Maruyama solver with 250 sampling steps for evaluation. Across both model scales, we use DINOv2-B as the pretrained vision encoder, cosine similarity for representation alignment, two pre-mask token mixing layers, and a mask ratio of 25%. The alignment weight is set to $\lambda = 0 . 5 .$ , and the class-token prediction weight is set to $\beta = 0 . 0 3$ . For optimization, we use AdamW with a batch size of 256 and a learning rate of $1 \times 1 0 ^ { - 4 }$ .

Table 7 Hyperparameter settings across different model scales.

<table><tr><td>Backbone</td><td>SiT-B</td><td>SiT-XL</td></tr><tr><td colspan="3">Architecture</td></tr><tr><td>#Params</td><td>154M</td><td>732M</td></tr><tr><td>Input</td><td> $32 \times 32 \times 4$ </td><td> $32 \times 32 \times 4$ </td></tr><tr><td>Layers</td><td>12</td><td>28</td></tr><tr><td>Hidden dim.</td><td>768</td><td>1,152</td></tr><tr><td>Num. heads</td><td>12</td><td>16</td></tr><tr><td colspan="3">MaskAlign settings</td></tr><tr><td> $\beta$ </td><td>0.03</td><td>0.03</td></tr><tr><td> $\lambda$ </td><td>0.5</td><td>0.5</td></tr><tr><td>Alignment depth</td><td>4</td><td>8</td></tr><tr><td>Mixing Layers</td><td>2</td><td>2</td></tr><tr><td>Mask Ratio</td><td>25%</td><td>25%</td></tr><tr><td>sim(·,·)</td><td>cos. sim.</td><td>cos. sim.</td></tr><tr><td>Encoder  $\mathcal{E}_{VF}(I)$ </td><td>DINOv2-B</td><td>DINOv2-B</td></tr><tr><td colspan="3">Optimization</td></tr><tr><td>Batch size</td><td>256</td><td>256</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>lr</td><td>0.0001</td><td>0.0001</td></tr><tr><td> $(\beta_1, \beta_2)$ </td><td>(0.9, 0.999)</td><td>(0.9, 0.999)</td></tr><tr><td colspan="3">Interpolants</td></tr><tr><td> $\alpha_t$ </td><td> $1 - t$ </td><td> $1 - t$ </td></tr><tr><td> $\sigma_t$ </td><td> $t$ </td><td> $t$ </td></tr><tr><td> $w_t$ </td><td> $\sigma_t$ </td><td> $\sigma_t$ </td></tr><tr><td>Training objective</td><td>v-prediction</td><td>v-prediction</td></tr><tr><td>Sampler</td><td>Euler-Maruyama</td><td>Euler-Maruyama</td></tr><tr><td>Sampling steps</td><td>250</td><td>250</td></tr></table>

## B Additional Token-Level Alignment Heatmaps

Table 8 provides additional heatmaps of the token-level alignment-gradient distribution under different timesteps and training iterations. Each heatmap shows the probability that each spatial position appears among the top-10% tokens ranked by alignment-gradient norm. Across different timesteps and checkpoints, the high-gradient tokens exhibit non-uniform spatial patterns, further supporting our observation that full-token representation alignment does not affect all patch tokens uniformly.

## C Additional Results on ImageNet

Table 9 reports additional quantitative results of MaskAlign on ImageNet $2 5 6 \times 2 5 6$ without classifier-free guidance (CFG). We evaluate MaskAlign at different training iterations to provide a more detailed view of its convergence behavior. As training proceeds, MaskAlign consistently improves generation quality, reducing FID from 22.36 at 50K iterations to 2.38 at 1M iterations. Compared with REG trained for 1M iterations, MaskAlign achieves better FID, sFID, and IS at the same training budget, while maintaining comparable precision and recall.

Table 8 Additional alignment-gradient heatmaps across timesteps and training iterations. Rows denote training iterations, and columns denote timesteps. Each heatmap shows the spatial probability of top-10% alignment-gradient tokens, using the same color range [0, 0.8].  
![](images/22ad3146a1c749d19894a5c0c233593e91a6bb5196195d3f79a93a8259940927.jpg)

<details>
<summary>heatmap</summary>

| Temperature | Time | Value |
|-------------|------|-------|
| 100K        | t=0.1 | High |
| 100K        | t=0.3 | Medium-High |
| 100K        | t=0.5 | Low |
| 100K        | t=0.7 | Medium |
| 100K        | t=0.9 | High |
| 500K        | t=0.1 | High |
| 500K        | t=0.3 | Medium |
| 500K        | t=0.5 | Low |
| 500K        | t=0.7 | Medium |
| 500K        | t=0.9 | High |
| 1M          | t=0.1 | High |
| 1M          | t=0.3 | Medium |
| 1M          | t=0.5 | Low |
| 1M          | t=0.7 | Medium |
| 1M          | t=0.9 | High |
</details>

Table 9 Additional quantitative results of MaskAlign on ImageNet 256 × 256 without classifier-free guidance (CFG). We report FID, sFID, IS, precision, and recall across different training iterations.

<table><tr><td>Model</td><td>#Params</td><td>Iter.</td><td>FID↓</td><td>sFID↓</td><td>IS↑</td><td>Prec.↑</td><td>Rec.↑</td></tr><tr><td>SiT-XL/2</td><td>675M</td><td>7M</td><td>8.3</td><td>6.32</td><td>131.7</td><td>0.68</td><td>0.67</td></tr><tr><td>REG</td><td>677M</td><td>1M</td><td>2.7</td><td>4.93</td><td>201.8</td><td>0.76</td><td>0.66</td></tr><tr><td>MaskAlign</td><td>732M</td><td>50K</td><td>22.36</td><td>20.62</td><td>70.24</td><td>0.63</td><td>0.53</td></tr><tr><td>MaskAlign</td><td>732M</td><td>110K</td><td>6.34</td><td>6.47</td><td>145.79</td><td>0.75</td><td>0.58</td></tr><tr><td>MaskAlign</td><td>732M</td><td>200K</td><td>3.98</td><td>5.18</td><td>172.60</td><td>0.77</td><td>0.60</td></tr><tr><td>MaskAlign</td><td>732M</td><td>400K</td><td>2.84</td><td>4.85</td><td>194.57</td><td>0.77</td><td>0.62</td></tr><tr><td>MaskAlign</td><td>732M</td><td>600K</td><td>2.67</td><td>4.78</td><td>198.01</td><td>0.77</td><td>0.64</td></tr><tr><td>MaskAlign</td><td>732M</td><td>1M</td><td>2.38</td><td>4.78</td><td>205.37</td><td>0.76</td><td>0.65</td></tr></table>

## D Broader Impacts

This work aims to improve the efficiency of diffusion transformer training. Its potential positive impacts include reducing the computational cost of training high-quality generative models and making research on diffusion models more accessible. However, more efficient training may also lower the barrier to building image generation systems, which could increase risks such as misleading synthetic content, impersonation, and biases inherited from training data or pretrained vision models. Our work does not introduce a deployed system or a new dataset, but responsible use of trained models should consider safeguards such as data curation, provenance tracking, watermarking, and controlled release when appropriate.

## E Assets and Licenses

We use ImageNet for non-commercial research and educational purposes, following its terms of access, and cite the original ImageNet paper. We use the Stable Diffusion VAE released by Stability AI under the MIT License to encode images into latent representations. We use DINOv2-B as the pretrained vision encoder for representation alignment; DINOv2 code and model weights are released under the Apache License 2.0. Our implementation also builds on the SiT,

REPA, and REG codebases, which are released under the MIT License. We properly credit these prior works through citations and use the corresponding assets only for research purposes and in accordance with their licenses and terms of use.

## F More Visualization Results

We present more visualization results of MaskAlign in Figures $_ { 4 - 8 . }$

![](images/29aa244c5e73831915bb959a50dbd66f15cc90f349e2f1686ffef2400cad1525.jpg)

<details>
<summary>natural_image</summary>

Collage of multiple owls from natural settings, including peck and otter, with no visible text or symbols.
</details>

Figure 4 Generated samples from $\mathrm { S i T } { \mathrm { - } } \mathrm { X L } / 2 + \mathrm { M a s k A l i g n }$ . The class label is “great grey $\mathsf { o w l } ^ { \mathsf { } } ( 2 4 )$ .

![](images/4953e92f7adf3e7cfcb6639291977d0564591a6036ddb978877d088b8d480095.jpg)

<details>
<summary>natural_image</summary>

Collage of multiple Golden retriever puppies in various poses and expressions, including vibrant expressions, animals, and scenic backgrounds (no text or symbols visible)
</details>

Figure 5 Generated samples from $\mathrm { S i T \mathrm { - } X L } / 2 + \mathrm { M a s k A l i g n } .$ . The class label is “golden retriever” (207).

![](images/8727803b40b2053ddc138cdb1c030871daa7441aae7f13478915390185e920fa.jpg)

<details>
<summary>natural_image</summary>

Collage of twelve white wolves in various poses and expressions, including standing, running, and interacting with a snowy landscape (no text or symbols visible)
</details>

Figure 6 Generated samples from SiT-XL/2 + MaskAlign. The class label is “arctic wolf” (270).

![](images/eb589abb025125da06b701389a4ab0f7d29e93b3e61890578140e312c91e14a4.jpg)

<details>
<summary>natural_image</summary>

Collage of ten different animal snub-nosed levers in natural habitat, showing various poses and expressions (no text or symbols)
</details>

Figure 7 Generated samples from SiT-XL/2 + MaskAlign. The class label is “polecat” (358).

![](images/7a91093e9d5485ff963a9e78856f6eb360b5e39966a87aced66b7b0616c9ae3d.jpg)

<details>
<summary>natural_image</summary>

Collage of scenic and architectural landmarks including historic stone buildings, a lake, and a castle with boats (no visible text or symbols)
</details>

Figure 8 Generated samples from SiT-XL/2 + MaskAlign. The class label is “castle” (483).