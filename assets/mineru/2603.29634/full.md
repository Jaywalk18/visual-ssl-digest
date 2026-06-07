# MacTok: Robust Continuous Tokenization for Image Generation

Hengyu Zeng1\* Jiaoyang Ruan1

Xin Gao1\* Junpeng Ma1

Guanghao Li1

Yuxiang Yan1

Haoyu Albert Wang1

Jian Pu1†

1 Fudan University

{hyzeng24, gaoxin23}@m.fudan.edu.cn, {jianpu}@fudan.edu.cn

# Abstract

Continuous image tokenizers enable efficient visual generation, and those based on variational frameworks can learn smooth, structured latent representations through KL regularization. Yet this often leads to posterior collapse when using fewer tokens, where the encoder fails to encode informative features into the compressed latent space. To address this, we introduce MacTok, a Masked Augmenting 1D Continuous Tokenizer that leverages image masking and representation alignment to prevent collapse while learning compact and robust representations. MacTok applies both random masking to regularize latent learning and DINOguided semantic masking to emphasize informative regions in images, forcing the model to encode robust semantics from incomplete visual evidence. Combined with global and local representation alignment, MacTok preserves rich discriminative information in a highly compressed 1D latent space, requiring only 64 or 128 tokens. On ImageNet, Mac-Tok achieves a competitive gFID of 1.44 at 256×256 and a state-of-the-art 1.52 at 512×512 with SiT-XL, while reducing token usage by up to 64×. These results confirm that masking and semantic guidance together prevent posterior collapse and achieve efficient, high-fidelity tokenization.

# 1. Introduction

In recent years, visual generative models have rapidly advanced by modeling data in compressed latent spaces, substantially reducing the cost of training and inference. A crucial component in this paradigm is the image tokenizer, which maps raw images into latent representations that are then used by diffusion [38], flow [36], or autoregressive models [3]. Tokenizers generally fall into two categories: discrete and continuous. Vector-Quantized Variational Auto-Encoders (VQ-VAE) [49] and its variants (e.g., VQ-GAN [13], TiTok [60]) represent the discrete family, discretizing the latent space through a finite codebook with straight-through estimation to ensure stability but at the cost of quantization error. In contrast, Kullback–Leibler Variational Auto-Encoders (KL-VAE) [25] and related methods (e.g., SD-VAE [41], MAR-VAE [34]) define a continuous latent space regularized by a Gaussian prior via KL divergence, yielding smooth representations but prone to posterior collapse under strong compression [5, 7].

![](images/b8db98a025d7ac43cbd307adc3678f91fd534530ea3f3de44d7e9701e7d9d123.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["w/o mask"] --> B["Encoder"]
    B --> C["KL loss"]
    C --> D["Decoder"]
    D --> E["Latent Space Collapsed"]
    F["w/ latent mask"] --> G["Encoder"]
    G --> H["KL loss"]
    H --> I["Decoder"]
    I --> J["Latent Space Collapsed"]
    K["w/ image mask"] --> L["Encoder"]
    L --> M["KL loss"]
    M --> N["Decoder"]
    N --> O["Uncollapsed"]
    style A fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style G fill:#ccf,stroke:#333
    style L fill:#ccf,stroke:#333
    style D fill:#cfc,stroke:#333
    style I fill:#cfc,stroke:#333
    style N fill:#cfc,stroke:#333
    style O fill:#fcc,stroke:#333
```
</details>

Figure 1. Effect of random masking in continuous tokenizers. Left: plain KL-VAE, latent token masking, and image token masking, with only the latter preventing posterior collapse. Right: collapsed latent space shows poor structure, while the uncollapsed one yields well-structured and diverse representations.

Although both discrete and continuous tokenizers have been extensively studied, achieving an optimal balance between compression efficiency and generation quality remains challenging. Recent studies have investigated strategies to enhance the efficiency of VQ-based discrete tokenizers, including two-stage training schemes to obtain more compact token representations [60], as well as the integration of pre-trained visual features [35]. In contrast, KLbased continuous tokenizer remain underexplored, primarily due to the persistent issue of posterior collapse. When trained with strong KL regularization, KL-VAEs tend to excessively constrain the latent distribution, pushing it toward an isotropic Gaussian prior and thereby discarding meaningful semantics [7, 14, 21]. Our experiments with KL-VAEs reveal frequent posterior collapse under strong compression, where latents lose essential information, leading to poor reconstruction and generation. While prior work addresses this through KL weight tuning [14, 21], these methods require delicate hyperparameter tuning and do not fundamentally resolve latent degeneration.

![](images/23a4be843b14c60e86c008fed6354c234404213903a7a535586b4730446a80f3.jpg)

<details>
<summary>natural_image</summary>

Collage of diverse images including parrot, butterfly, hot air balloon, dog, car, airplane, sea lion, wolf, and goldfish (no text or symbols)
</details>

Figure 2. Generation results produced by generative models with MacTok using 64 and 128 tokens on ImageNet at 256×256 and 512×512.

The key to overcoming posterior collapse lies in learning a robust and discriminative latent representation that preserves semantics under strong compression. A wellstructured latent space mitigates collapse by maintaining sufficient mutual information between the input and its encoded representation, preventing overreliance on the prior [39, 66]. To this end, we draw inspiration from masked representation learning [19] and require the tokenizer to reconstruct images from partial inputs. Specifically, we investigate two masking schemes, as shown in Fig. 1: one operates on latent tokens, and the other on image tokens. For latent tokens, previous work suggests that randomly dropping some of them can improve representation robustness [56], and our experiments likewise show that latent token masking temporarily delays posterior collapse compared with a standard KL-VAE. However, this stabilization is short-lived and the model eventually collapses as training continues (details in Appendix A.3). In contrast, masking image tokens yields consistently stable optimization and more robust representations, as it forces both encoder and decoder to infer from incomplete inputs, encouraging the latent space to capture global structural semantics. Building on this observation, we further hypothesize that masking can make the latent space robust and semantically discriminative. Therefore, we propose a DINO-guided semantic masking strategy that selectively occludes the most informative regions identified by computing the similarity between the classification token and each patch token in DINOv2 [37]. Unlike random masking, this targeted strategy enforces the reconstruction of key semantic regions from incomplete observations, implicitly transferring semantic priors from the image space into the latent space. As shown in Fig. 3, semantic masking significantly improves generation quality by producing more semantically rich latent representations.

Furthermore, state-of-the-art tokenizers, both discrete and continuous, often improve reconstruction and generation quality by aligning latent representations with external semantic features [61], such as those extracted by DI-NOv2. However, previous approaches have notable drawbacks: they rely on a fixed token count [57], perform only coarse global alignment [35], or require multiple auxiliary objectives [4]. To address these issues, we propose the global and local representation alignment that aligns each latent token with both holistic and local region features. This approach facilitates more consistent semantic guidance across varying token lengths, leading to a well-structured latent space for improved reconstruction and generation.

![](images/f6a0b3f8cb69140d62359539d51f3bb35e95a7196b9f455c84ee95cc71ac91c8.jpg)

<details>
<summary>line</summary>

| M | gFID (J) - Random | gFID (J) - Random+Semantic | IS (J) - Random | IS (J) - Random+Semantic |
|---|---|---|---|---|
| 0.4 | 15.5 | 14.9 | 78.5 | 84.5 |
| 0.5 | 14.9 | 14.8 | 81.2 | 84.0 |
| 0.6 | 14.8 | 14.7 | 81.8 | 83.5 |
| 0.7 | 14.6 | 14.6 | 82.5 | 83.0 |
| 0.8 | 14.9 | 14.7 | 81.0 | 82.5 |
</details>

Figure 3. Generation performance of MacTok with varying mask ratios sampled up to M as detailed in Sec. 3.2. The orange star corresponds to random and semantic mask with equal probability.

Extensive experiments on ImageNet demonstrate the effectiveness of MacTok in mitigating posterior collapse and enhancing generation quality under strong compression. We observe that an appropriate mask ratio consistently improves generation performance. Combining masking and representation alignment, MacTok achieves a superior trade-off between reconstruction fidelity and generative quality, reaching competitive rFID and state-of-the-art gFID with only 64 and 128 tokens on ImageNet at 256×256 and 512×512. Our contributions are summarized as follows:

• We identify posterior collapse in continuous tokenizers as a form of over-regularization that suppresses semantic information, and show that maintaining information flow through reconstruction from incomplete inputs stabilizes latent learning under strong compression.   
• We introduce MacTok, which jointly regularizes and semantically structures the latent space through both image masking and representation alignment. This prevents posterior collapse while preserving discriminative features.   
• MacTok achieves competitive reconstruction and state-ofthe-art generation results on ImageNet, with a gFID of 1.44 at 256×256 and 1.52 at 512×512 using only 64 and 128 tokens, ensuring high fidelity, strong compression, and stable training.

# 2. Related Work

# 2.1. Image Tokenization

Image tokenization has emerged as a fundamental component of modern generative modeling, converting images from the pixel space into compact latent representations for efficient synthesis. Most approaches build on autoencoders equipped with quantizers or variational posteriors, each exhibiting characteristic limitations. Discrete tokenizers such as VQ-VAE [49] and VQ-GAN [13] learn a codebook of discrete tokens, with the latter employing adversarial training for improved perceptual fidelity. However, they suffer from limited gradient flow and suboptimal codebook utilization due to the straight-through estimator used during training. Subsequent works improve efficiency through better codebook learning strategies. For example, IBQ [44] updates the full codebook via index backpropagation, while MAGVIT-v2 [59] adopts Look-up Free Quantization (LFQ) to bypass explicit codebook management. Continuous tokenizers, in contrast, map images to continuous latent spaces without discrete quantization. MAETok [4] employs a plain autoencoder with auxiliary losses for semantic guidance, while SoftVQ-VAE [5] enhances latent space capacity under high compression by bridging between discrete and continuous encoding via soft codeword aggregation. However, KL-based continuous models remain vulnerable to posterior collapse [5, 7] under strong compression, where the encoder fails to preserve informative features, leading to poor reconstruction and limited generative fidelity. Our work introduces a KL-VAE framework that combines random and semantic masking with global and local representation alignment, effectively mitigating collapse and enhancing both reconstruction fidelity and generative quality.

# 2.2. Image Generation

The integration of tokenization into generative models has driven significant advances in image synthesis, particularly in diffusion models [36, 38, 57] and autoregressive [34, 45, 46] architectures. Diffusion-based approaches excel at generating high-resolution images by iteratively denoising representations in continuous latent spaces [17]. In contrast, autoregressive [45, 46] and masked prediction models [3, 59] often operate in discrete token spaces. Convolutional backbones [42, 48] have gradually been replaced by transformer-based architectures [36, 38], which offer improved scalability and representational capacity for both 2D synthesis and broader spatial modeling [6, 30–33].

# 2.3. Representation Alignment for Generation

More recent works explore incorporating semantic information into tokenization and generation to enhance image synthesis. In diffusion-based models, methods such as [29, 52, 61] align transformer representations with pretrained visual embeddings, improving both training efficiency and generation quality. For discrete tokenizers, VQGAN-LC [68] leverages CLIP [40] features for codebook initialization to boost utilization and perceptual fidelity, while VQ-KD [50] trains tokenizers to reconstruct features from pretrained visual encoders. ImageFolder [35] adopts product quantization to produce spatially aligned semantic and detail tokens, reducing sequence length without compromising quality. In continuous tokenizers, VA-VAE [57] aligns its latent space with vision foundation models to stabilize optimization. TexTok [62] introduces textual information into tokenization to enhance both reconstruction and generation. SoftVQ-VAE [5] employs soft categorical posteriors for feature alignment, and MAETok [4] incorporates auxiliary semantic targets from HOG [8], DINOv2 [37], and Sig-

CLIP [63]. Our work combine global and local representation alignment as latent space regularization, leading to improved reconstruction fidelity and generative performance.

# 3. Method

We present MacTok, a 1D continuous tokenizer that prevents posterior collapse by enforcing information preservation through image masking and feature alignment. As illustrated in Fig. 4, MacTok reconstructs complete images from incomplete visual evidence using two complementary masking strategies, while aligning the latent space with pretrained features to maintain robust and discriminative representations even under strong compression.

# 3.1. Continuous Tokenizer Architecture

As illustrated in Fig. 4, MacTok adopts a Vision Transformer (ViT) as both the encoder E and decoder D [11, 54, 55, 58]. Building on recent one-dimensional tokenizer designs [4, 5, 60] that allow flexible token lengths for image representation, we extend ViT to jointly process image tokens and latent tokens, where the latent tokens serve as compact representations for reconstruction and generation.

The encoder E first partitions an input image $\mathrm { ~ { ~ \bf ~ I ~ } ~ } \in$ $\mathbb { R } ^ { H \times W \times 3 }$ into non-overlapping patches of size $P ,$ producing image tokens $\mathbf { x } \in \mathbb { R } ^ { N \times D }$ , where $\begin{array} { r } { N = \frac { H W } { P ^ { 2 } } } \end{array}$ P 2 is the number of patches and $D$ denotes the embedding dimension. A set of learnable tokens $\mathbf { z } \in \mathbb { R } ^ { L \times D }$ is concatenated with the image tokens, forming the sequence [x; z] ∈ R(N+L)×D, $[ \mathbf { x } ; \mathbf { z } ] \in \mathbb { R } ^ { ( N + L ) \times D }$ which is then passed through the encoder. The encoder outputs corresponding to the latent tokens are taken as the latent representations $\hat { \mathbf { z } } \in \mathbb { R } ^ { L \times Z } \colon \hat { \mathbf { z } } = E ( [ \mathbf { x } ; \mathbf { z } ] )$ , where Z denotes the latent dimensionality. To model a continuous latent space, the latent vector zˆ is treated as a Gaussian random variable parameterized by its mean and variance. A KL divergence regularizes the posterior toward an isotropic Gaussian prior, promoting smoothness in the latent space. However, under strong compression, this constraint may excessively regularize latent representations, leading to posterior collapse [5, 7, 16]. This issue is precisely what MacTok is designed to address.

During decoding, the sampled latent representations zˆ are concatenated with learnable reconstruction tokens h ∈ $\mathbb { R } ^ { N \times Z }$ and passed to the decoder: $\hat { \mathbf { x } } ~ = ~ D ( [ \mathbf { h } ; \hat { \mathbf { z } } ] )$ . Decoder outputs corresponding to h are projected through a linear layer to reconstruct the image ˆI. We use 2D absolute positional embeddings for image tokens to preserve spatial structure and 1D embeddings for latent and reconstruction tokens without explicit spatial coordinates.

# 3.2. Image Masking for Latent Preservation

Posterior collapse occurs when the latent variables fail to capture informative content, causing the decoder to reconstruct images primarily from priors. To address this, Mac-Tok enforces information flow via masked reconstruction, requiring the model to infer missing content from partial inputs and thus maintain robust representations. Two complementary masking strategies, Random and Semantic, are applied with equal probability during training.

Random Masking. At each iteration, a random subset of image patches is replaced with mask tokens before encoding, forcing the latent tokens to infer complete information and reconstruct the missing regions from visible context. The mask ratio m is uniformly sampled from [−0.1, M ] and clipped to [0, M ] (typically M =0.7), which allows the model to occasionally observe unmasked images $( m { = } 0 )$ and prevents excessive reconstruction degradation from persistent masking. This stochastic corruption of images compels the latent variables to encode essential information for reconstruction, thereby preventing posterior collapse (see analysis in Appendix A.2).

Semantic Masking. While random masking enhances robustness, it is agnostic to semantic structure. To inject semantic priors into the latent space, we further employ a guided masking strategy based on DINOv2 features. Given the classification token $\textbf { c } \in \mathbb { R } ^ { D }$ and patch tokens $\mathrm { ~ \bf ~ P ~ } =$ $\{ \mathbf { p } _ { i } \in \mathbb { R } ^ { D } \} _ { i = 1 } ^ { N }$ , we compute the cosine similarity between the classification token and each patch token:

$$
s _ {i} = \frac {\mathbf {c} ^ {\top} \mathbf {p} _ {i}}{\| \mathbf {c} \| \| \mathbf {p} _ {i} \|}, \quad i = 1, \dots , N, \tag {1}
$$

where $s _ { i }$ denotes the semantic relevance of patch i. We then select the top $\lfloor m \times N \rfloor$ patches with the highest similarity scores:

$$
M _ {p} = \operatorname{TopK} (\{s _ {i} \} _ {i = 1} ^ {N}, \lfloor m \times N \rfloor), \tag {2}
$$

where $M _ { p }$ is the index set of masked patches. When semantically important regions are masked, the reconstruction task becomes substantially harder, encouraging the latent tokens to capture object-level structures and global context. This semantic masking implicitly transfers knowledge from the image space to the latent space and yields more discriminative representations.

# 3.3. Local and Global Representation Alignment

We further align MacTok’s latent representations with pretrained DINOv2 features [37] to enforce semantic consistency, as representation alignment has been shown to be an effective auxiliary objective for enhancing visual generation [5, 57, 61]. Unlike previous approaches, we introduce a lightweight global and local alignment that links each latent token to both regional and holistic semantics, improving structural coherence and detail preservation.

Local and Gobal Feature Construction. Let $\hat { \mathbf { z } } = \{ \hat { \mathbf { z } } _ { i } \} _ { i = 1 } ^ { L }$ denote the latent tokens produced by the encoder. To match the spatial granularity of DINOv2 patch features $\{ { \bf p } _ { i } \} _ { i = 1 } ^ { N } ,$ we expand zˆ into a sequence $\mathbf { \tilde { z } } _ { \mathrm { l o c } } ~ \in ~ \mathbb { R } ^ { N \times Z }$ by repeating each latent token r times, where $r = N / L !$ :

![](images/99dd9e09a488a663d695fc81b61ed1af8ec6f2a027422e8b77999d40d0c058f6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Image Masking"] --> B["Random mask"]
    B --> C{Select}
    C --> D["ViT Encoder"]
    D --> E["KL Loss"]
    E --> F["ViT Decoder"]
    F --> G["Representation Alignment"]
    G --> H["Discriminator"]
    G --> I["Perceptual Network"]
    H --> J["GAN Loss"]
    I --> K["Perceptual Loss"]
    L["Similarity map"] --> M["Semantic mask"]
    M --> N["Image Masking"]
    N --> O["Random mask"]
    O --> P{Select}
    P --> Q["ViT Encoder"]
    Q --> R["ViT Decoder"]
    R --> S["Representation Alignment"]
    S --> T["Discriminator"]
    S --> U["Perceptual Network"]
    V["cls token"] --> W["Global alignment"]
    W --> X["MLP1"]
    X --> Y["global token"]
    Y --> Z["Average"]
    Z --> AA["Local alignment"]
    AA --> AB["patch tokens"]
    AB --> AC["Local tokens"]
    AC --> AD["MLP2"]
    AD --> AE["Expand"]
    AE --> AF["Local tokens"]
    AF --> AG["Local alignment"]
    AG --> AH["Global alignment"]
    AH --> AI["MLP1"]
    AI --> AJ["Global alignment"]
    AJ --> AK["Average"]
    AK --> AL["Local alignment"]
    AL --> AM["MLP2"]
    AM --> AN["Expand"]
    AN --> AO["Local tokens"]
    AO --> AP["Local alignment"]
    AP --> AQ["Global alignment"]
    AQ --> AR["Local tokens"]
    AR --> AS["Global alignment"]
    AS --> AT["Local alignment"]
    AT --> AU["Global alignment"]
    AU --> AV["Local alignment"]
    AV --> AW["Global alignment"]
    AW --> AX["Local alignment"]
    AX --> AY["Global alignment"]
    AY --> AZ["Local alignment"]
    AZ --> BA["Global alignment"]
    BA --> BB["Local alignment"]
    BB --> BC["Global alignment"]
    BC --> BD["Local alignment"]
    BD --> BE["Global alignment"]
    BE --> BF["Local alignment"]
    BF --> BG["Global alignment"]
    BG --> BH["Local alignment"]
    BH --> BI["Global alignment"]
    BI --> BJ["Local alignment"]
    BJ --> BK["Global alignment"]
    BK --> BL["Local alignment"]
    BL --> BM["Global alignment"]
    BM --> BN["Local alignment"]
    BN --> BO["Global alignment"]
    BO --> BP["Local alignment"]
    BP --> BQ["Global alignment"]
    BQ --> BR["Local alignment"]
    BR --> BS["Global alignment"]
    BS --> BT["Local alignment"]
    BT --> BU["Global alignment"]
    BU --> BV["Local alignment"]
    BV --> BW["Global alignment"]
    BW --> BX["Local alignment"]
    BX --> BY["Global alignment"]
    BY --> BZ["Local alignment"]
    BZ --> CA["Global alignment"]
    CA --> CB["Local alignment"]
    CB --> CC["Global alignment"]
    CC --> CD["Local alignment"]
    CD --> CE["Global alignment"]
    CE --> CF["Local alignment"]
    CF --> CG["Global alignment"]
    CG --> CH["Local alignment"]
    CH --> CI["Global alignment"]
    CI --> CJ["Local alignment"]
    CJ --> CK["Global alignment"]
```
</details>

Figure 4. Overview of the MacTok framework. Top: Transformer-based encoder and decoder operating on image, latent, and mask tokens. Bottom left: DINO-guided image masking introduces semantic priors. Bottom center: Global and local representation alignment between latent and pretrained visual representations. Bottom right: Discriminator and perceptual networks provide auxiliary supervision.

$$
\tilde {\mathbf {z}} _ {\mathrm{loc}} = \operatorname{Expand} (\hat {\mathbf {z}}, r). \tag {3}
$$

In parallel, we obtain a global latent representation by average pooling:

$$
\tilde {\mathbf {z}} _ {\text {glob}} = \frac {1}{L} \sum_ {i = 1} ^ {L} \hat {\mathbf {z}} _ {i}. \tag {4}
$$

Both local and global features are linearly projected into the DINOv2 feature space using two lightweight MLPs:

$$
\mathbf {o} _ {\text { loc }} = \mathrm{MLP} _ {1} (\tilde {\mathbf {z}} _ {\text { loc }}), \quad \mathbf {o} _ {\text { glob }} = \mathrm{MLP} _ {2} (\tilde {\mathbf {z}} _ {\text { glob }}). \tag {5}
$$

Representation Alignment Loss. We encourage the projected latent features to align with the corresponding DI-NOv2 features using cosine similarity. The local alignment compares $\mathbf { o } _ { \mathrm { l o c } }$ with patch tokens $\{ \mathbf { p } _ { i } \}$ , while the global alignment matches $\mathbf { o } _ { \mathrm { g l o b } }$ with the classification token c:

$$
L _ {\mathrm{RA}} = - \frac {1}{(N + 1)} \left(\sum_ {i = 1} ^ {N} \operatorname{sim} \left(\mathbf {o} _ {\text {loc}, i}, \mathbf {p} _ {i}\right) + \operatorname{sim} \left(\mathbf {o} _ {\text {glob}}, \mathbf {c}\right)\right). \tag {6}
$$

This loss encourages latent tokens to encode semantically coherent information at both patch and image levels, resulting in a well-structured latent space that enhances reconstruction and generation quality under strong compression.

# 3.4. Training Objectives

MacTok is optimized with a composite objective that includes reconstruction, perceptual [12, 23, 27, 65], and adversarial [18] terms, following [5, 13], as well as regularization [25] and representation alignment terms:

$$
L = L _ {\text { recon }} + \lambda_ {1} L _ {\text { percep }} + \lambda_ {2} L _ {\text { adv }} + \lambda_ {3} L _ {\text { KL }} + \lambda_ {4} L _ {\text { RA }}, \tag {7}
$$

where $\lambda _ { 1 } \mathfrak { - } \lambda _ { 4 }$ are weighting coefficients. $L _ { \mathrm { r e c o n } }$ is a pixelwise reconstruction loss, $L _ { \mathrm { p e r c e p } }$ enforces perceptual similarity in a pretrained feature space, $L _ { \mathrm { a d v } }$ encourages realistic image synthesis through adversarial learning, $L _ { \mathrm { K L } }$ regularizes the latent distribution toward a Gaussian prior, and $L _ { \mathrm { R A } }$ is the proposed representation alignment loss.

# 4. Experiments

# 4.1. Experiments Setup

Implementation Details of Our Method. By default, Mac-Tok adopts a ViT-Base backbone for both the encoder and decoder, totaling 176M parameters. We use DINOv2 [37] pretrained features and initialize the encoder with DINOv2 weights to inject richer semantic priors into the latent space, following [5]. DINOv2 features are also used to guide the semantic masking process, promoting more robust latent space as shown in Sec. 3.2. MacTok is trained on ImageNet [9] at 256×256 for 250K iterations and $5 1 2 \times 5 1 2$ for 500K iterations. A frozen DINO-S [2, 37] discriminator is used, with DiffAug [67], consistency regularization [64], and LeCAM [47] as in [5, 46]. During training, we apply random and semantic masking with equal probability, using M of 70%. For decoder fine-tuning, the encoder is frozen and the decoder is trained for 10 epochs without mask. Unless otherwise specified, the image token channel dimension in MacTok is set to 32. The loss weights are set to $\lambda _ { 1 } { = } 1 . 0 $ , $\lambda _ { 2 } { = } 0 . 2 , \lambda _ { 3 } { = } 1 0 ^ { - 6 } .$ , and $\lambda _ { 4 } { = } 0 . 1$ , following common practice. More training details are provided in Appendix B.1.

Table 1. System-level comparison on ImageNet 256×256 conditional generation. “# Params (G)” denotes generator parameters; “Tok. Model” refers to the tokenizer model type; “Token Type” indicates 1D or 2D tokenization; “# Params (T)” denotes tokenizer parameters; and “# Tokens” represents the number of latent tokens. ‡ denotes methods that rely on pretrained vision models. 

<table><tr><td rowspan="2">Method</td><td rowspan="2"># Params (G)</td><td rowspan="2">Tok. Model</td><td rowspan="2">Token Type</td><td rowspan="2"># Params (T)</td><td rowspan="2"># Tokens↓</td><td rowspan="2">Tok. rFID↓</td><td colspan="2">w/o CFG</td><td colspan="2">w/ CFG</td></tr><tr><td>gFID↓</td><td>IS↑</td><td>gFID↓</td><td>IS↑</td></tr><tr><td colspan="11">Auto-regressive</td></tr><tr><td>ViT-VQGAN [58]</td><td>1.7B</td><td>VQ</td><td>2D</td><td>64M</td><td>1024</td><td>1.28</td><td>4.17</td><td>175.1</td><td>-</td><td>-</td></tr><tr><td>RQ-Trans. [28]</td><td>3.8B</td><td>RQ</td><td>2D</td><td>66M</td><td>256</td><td>3.20</td><td>-</td><td>-</td><td>3.80</td><td>323.7</td></tr><tr><td>MaskGIT [3]</td><td>227M</td><td>VQ</td><td>2D</td><td>66M</td><td>256</td><td>2.28</td><td>6.18</td><td>182.1</td><td>-</td><td>-</td></tr><tr><td>LlamaGen-3B [45]</td><td>3.1B</td><td>VQ</td><td>2D</td><td>72M</td><td>576</td><td>2.19</td><td>-</td><td>-</td><td>2.18</td><td>263.3</td></tr><tr><td>WeTok [69]</td><td>1.5B</td><td>VQ</td><td>2D</td><td>400M</td><td>256</td><td>0.60</td><td>-</td><td>-</td><td>2.31</td><td>276.6</td></tr><tr><td>VAR [46]</td><td>2B</td><td>MSRQ</td><td>2D</td><td>109M</td><td>680</td><td>0.90</td><td>-</td><td>-</td><td>1.92</td><td>323.1</td></tr><tr><td>MaskBit [51]</td><td>305M</td><td>LFQ</td><td>2D</td><td>54M</td><td>256</td><td>1.61</td><td>-</td><td>-</td><td>1.52</td><td>328.6</td></tr><tr><td>MAR-H [34]</td><td>943M</td><td>KL</td><td>2D</td><td>66M</td><td>256</td><td>1.22</td><td>2.35</td><td>227.8</td><td>1.55</td><td>303.7</td></tr><tr><td>l-DeTok [56]</td><td>479M</td><td>KL</td><td>2D</td><td>172M</td><td>256</td><td>0.62</td><td>1.86</td><td>238.6</td><td>1.35</td><td>304.1</td></tr><tr><td>TiTok-S-128 [60]</td><td>287M</td><td>VQ</td><td>1D</td><td>72M</td><td>128</td><td>1.61</td><td>-</td><td>-</td><td>1.97</td><td>281.8</td></tr><tr><td>GigaTok‡ [53]</td><td>111M</td><td>VQ</td><td>1D</td><td>622M</td><td>256</td><td>0.51</td><td>-</td><td>-</td><td>3.15</td><td>224.3</td></tr><tr><td>ImageFolder† [35]</td><td>362M</td><td>MSRQ</td><td>1D</td><td>176M</td><td>286</td><td>0.80</td><td>-</td><td>-</td><td>2.60</td><td>295.0</td></tr><tr><td colspan="11">Diffusion-based</td></tr><tr><td>LDM-4 [41]</td><td>400M</td><td></td><td>2D</td><td></td><td></td><td></td><td>10.56</td><td>103.5</td><td>3.60</td><td>247.7</td></tr><tr><td>U-ViT-H/2 [1]</td><td>501M</td><td>KL</td><td>2D</td><td>55M</td><td>4096</td><td>0.27</td><td>-</td><td>-</td><td>2.29</td><td>263.9</td></tr><tr><td>MDTv2-XL/2 [15]</td><td>676M</td><td></td><td>2D</td><td></td><td></td><td></td><td>5.06</td><td>155.6</td><td>1.58</td><td>314.7</td></tr><tr><td>DiT-XL/2 [38]</td><td>675M</td><td></td><td>2D</td><td></td><td></td><td></td><td>9.62</td><td>121.5</td><td>2.27</td><td>278.2</td></tr><tr><td>SiT-XL/2 [36]</td><td></td><td>KL</td><td>2D</td><td>84M</td><td>1024</td><td>0.62</td><td>8.30</td><td>131.7</td><td>2.06</td><td>270.3</td></tr><tr><td>+REPA‡ [61]</td><td>675M</td><td></td><td>2D</td><td></td><td></td><td></td><td>5.90</td><td>157.8</td><td>1.42</td><td>305.7</td></tr><tr><td>LightningDiT‡ [57]</td><td>675M</td><td>KL</td><td>2D</td><td>70M</td><td>256</td><td>0.28</td><td>2.17</td><td>205.6</td><td>1.35</td><td>295.3</td></tr><tr><td>TexTok-256 [62]</td><td>675M</td><td>KL</td><td>1D</td><td>176M</td><td>256</td><td>0.73</td><td>-</td><td>-</td><td>1.46</td><td>303.1</td></tr><tr><td>MAETok‡ [4]</td><td>675M</td><td>AE</td><td>1D</td><td>176M</td><td>128</td><td>0.48</td><td>2.31</td><td>216.5</td><td>1.67</td><td>311.2</td></tr><tr><td>SoftVQ-VAE‡ [5]</td><td>675M</td><td>SoftVQ</td><td>1D</td><td>176M</td><td>64</td><td>0.88</td><td>5.98</td><td>138.0</td><td>1.78</td><td>279.0</td></tr><tr><td colspan="11">Ours</td></tr><tr><td rowspan="2">MacTok+LightningDiT‡</td><td rowspan="2">675M</td><td></td><td></td><td></td><td>64</td><td>0.75</td><td>4.15</td><td>167.8</td><td>1.68</td><td>307.3</td></tr><tr><td>KL</td><td>1D</td><td>176M</td><td>128</td><td>0.43</td><td>3.12</td><td>186.2</td><td>1.50</td><td>299.8</td></tr><tr><td rowspan="2">MacTok+SiT-XL‡</td><td rowspan="2">675M</td><td></td><td></td><td></td><td>64</td><td>0.75</td><td>3.77</td><td>181.6</td><td>1.58</td><td>310.4</td></tr><tr><td></td><td></td><td></td><td>128</td><td>0.43</td><td>2.82</td><td>189.2</td><td>1.44</td><td>302.5</td></tr></table>

Implementation Details of Generative Modeling. For downstream generation, we employ SiT [36] and LightningDiT [57] due to their strength and flexibility in modeling 1D token sequences. SiT uses a patch size of 1 with absolute positional embeddings, while LightningDiT adopts rotary positional embeddings. In the main experiments, LightningDiT-XL is trained for 400K steps and SiT-XL for 4M steps, compared to 4M steps in REPA [61] and 7M steps in the original SiT [36]. For additional experiments, SiT-B is trained for 500K steps. Additional implementation details can are shown in Appendix B.2.

Evaluation. We evaluate reconstruction quality using the reconstruction Frechet Inception Distance (rFID) [ ´ 20], Peak Signal-to-Noise Ratio (PSNR), and Structural Similarity Index Measure (SSIM) on 50K validation images from ImageNet. For generation performance, we report the generation FID (gFID) [20], Inception Score (IS) [43], and Precision and Recall [26] (see Appendix C.3 for details), both with and without classifier-free guidance (CFG) [22], following the ADM [10] evaluation protocol and toolkit.

# 4.2. Main Results

Generation. We evaluate SiT-XL and LightningDiT trained with MacTok using 64 and 128 tokens on ImageNet at 256×256 and 512×512 resolutions, respectively. Their performance is compared against state-of-the-art (SOTA) generative models. Both LightningDiT-XL and SiT-XL trained with MacTok variants show substantial improvements in generation quality, surpassing SiT-XL/2 with 1024 tokens without CFG, and outperforming other tokenizers with CFG under the same token length. At 256×256 resolution, Mac-Tok surpasses SoftVQ-VAE [5] by 2.21 gFID using 64 tokens without CFG and achieves a gFID of 1.44 using 128 tokens with CFG, comparable to the state of the art. While LightningDiT-XL produces slightly lower quality than SiT-XL, it still outperforms other baselines. With CFG applied, SiT-XL with MacTok using 128 tokens achieves a new SOTA of 1.52 gFID and 316.0 IS on the 512 benchmark. Interestingly, MacTok with 64 tokens performs even better than 128 tokens without CFG at 512 resolution, mainly due to the larger decoder used for fair comparison with SoftVQ-VAE. It outperforms SoftVQ-VAE by 0.69 gFID using 64 tokens and surpasses MAETok [4] with CFG using 128 tokens. These results demonstrate that MacTok effectively mitigates posterior collapse in KL-based tokenizers, while maintaining strong generation fidelity. We present representative samples across different resolutions in Fig. 2, with additional visual results provided in Appendix C.5.

Table 2. System-level comparison on ImageNet 512×512 conditional generation. SiT-XL trained with MacTok achieves state-of-the-art generation performance using only 64 and 128 tokens (†: Large decoder for fair comparison; ‡: relies on pretrained vision models). 

<table><tr><td rowspan="2">Method</td><td rowspan="2"># Params (G)</td><td rowspan="2">Tok. Model</td><td rowspan="2">Token Type</td><td rowspan="2"># Params (T)</td><td rowspan="2"># Tokens↓</td><td rowspan="2">Tok. rFID↓</td><td colspan="2">w/o CFG</td><td colspan="2">w/ CFG</td></tr><tr><td>gFID↓</td><td>IS↑</td><td>gFID↓</td><td>IS↑</td></tr><tr><td colspan="11">GAN</td></tr><tr><td>BigGAN [3]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>8.43</td><td>177.9</td></tr><tr><td>StyleGAN-XL [24]</td><td>168M</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.41</td><td>267.7</td></tr><tr><td colspan="11">Auto-regressive</td></tr><tr><td>MaskGIT [3]</td><td>227M</td><td>VQ</td><td>2D</td><td>66M</td><td>1024</td><td>1.97</td><td>7.32</td><td>156.0</td><td>-</td><td>-</td></tr><tr><td>MAGVIT-v2 [59]</td><td>307M</td><td>LFQ</td><td>2D</td><td>116M</td><td>1024</td><td>-</td><td>-</td><td>-</td><td>1.91</td><td>324.3</td></tr><tr><td>MAR-H [34]</td><td>943M</td><td>KL</td><td>2D</td><td>66M</td><td>1024</td><td>-</td><td>2.74</td><td>205.2</td><td>1.73</td><td>279.9</td></tr><tr><td>TiTok-B-128 [60]</td><td>177M</td><td>VQ</td><td>1D</td><td>202M</td><td>128</td><td>1.52</td><td>-</td><td>-</td><td>2.13</td><td>261.2</td></tr><tr><td>TiTok-L-64 [60]</td><td>177M</td><td>VQ</td><td>1D</td><td>644M</td><td>64</td><td>1.77</td><td>-</td><td>-</td><td>2.74</td><td>221.1</td></tr><tr><td colspan="11">Diffusion-based</td></tr><tr><td>ADM [10]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>23.24</td><td>58.1</td><td>3.85</td><td>221.7</td></tr><tr><td>U-ViT-H/4 [1]</td><td>501M</td><td></td><td>2D</td><td></td><td></td><td></td><td>-</td><td>-</td><td>4.05</td><td>263.8</td></tr><tr><td>DiT-XL/2 [38]</td><td>675M</td><td></td><td>2D</td><td></td><td></td><td></td><td>9.62</td><td>121.5</td><td>3.04</td><td>240.8</td></tr><tr><td>SiT-XL/2 [36]</td><td>675M</td><td>KL</td><td>2D</td><td>84M</td><td>4096</td><td>0.62</td><td>-</td><td>-</td><td>2.62</td><td>252.2</td></tr><tr><td>DiT-XL [38]</td><td>675M</td><td></td><td>2D</td><td></td><td></td><td></td><td>9.56</td><td>-</td><td>2.84</td><td>-</td></tr><tr><td>UViT-H [1]</td><td>501M</td><td></td><td>2D</td><td></td><td></td><td></td><td>9.83</td><td>-</td><td>2.53</td><td>-</td></tr><tr><td>UViT-H [1]</td><td>501M</td><td></td><td>2D</td><td></td><td></td><td></td><td>12.26</td><td>-</td><td>2.66</td><td>-</td></tr><tr><td>UViT-2B [1]</td><td>2B</td><td>AE</td><td>2D</td><td>323M</td><td>256</td><td>0.22</td><td>6.50</td><td>-</td><td>2.25</td><td>-</td></tr><tr><td>TexTok-128 [62]</td><td>675M</td><td>KL</td><td>1D</td><td>176M</td><td>128</td><td>0.97</td><td>-</td><td>-</td><td>1.80</td><td>305.4</td></tr><tr><td>MAETok $^{\ddagger}$  [4]</td><td>675M</td><td>AE</td><td>1D</td><td>176M</td><td>128</td><td>0.62</td><td>2.79</td><td>204.3</td><td>1.69</td><td>304.2</td></tr><tr><td>SoftVQ-VAE $^{\ddagger}$  [5]</td><td>675M</td><td>SoftVQ</td><td>1D</td><td>391M</td><td>64</td><td>0.71</td><td>7.96</td><td>133.9</td><td>2.21</td><td>290.5</td></tr><tr><td colspan="11">Ours</td></tr><tr><td rowspan="2">MacTok+SiT-XL $^{\ddagger}$ </td><td rowspan="2">675M</td><td rowspan="2">KL</td><td rowspan="2">1D</td><td>391M $^{\dagger}$ </td><td>64</td><td>0.89</td><td>4.63</td><td>163.7</td><td>1.52</td><td>306.0</td></tr><tr><td>176M</td><td>128</td><td>0.79</td><td>5.12</td><td>156.3</td><td>1.52</td><td>316.0</td></tr></table>

Reconstruction. MacTok also exhibits strong reconstruction performance while using substantially fewer tokens. It achieves rFID scores of 0.75 and 0.43 with 64 and 128 tokens on the 256 benchmark, and 0.89 and 0.79 on the 512 benchmark. These results outperform VQ-based tokenizers that typically require at least 256 tokens [28, 51, 58]. Moreover, MacTok achieves competitive results compared to KL-based tokenizers used in diffusion-based models while requiring up to 64× fewer tokens. The superior performance with such compact representations highlights Mac-Tok’s ability to learn latents rich in semantic information, maintaining fidelity for downstream generative modeling despite the significantly reduced token count. Comprehensive reconstruction samples across varying token numbers, as well as visualization of posterior collapse scenarios, are included in Appendix C.4.

# 4.3. Comparison of Tokenizers

We compare MacTok with several leading continuous tokenizers, including VA-VAE [57], MAETok [4], SoftVQ-VAE [5], SD-VAE [41], MAR-VAE [34], and l-DeTok [56]. For these experiments, SiT-B is trained for 500K steps, and gFID and IS are evaluated on the 256×256 benchmark under optimal CFG settings. MacTok achieves the better balance between reconstruction quality and token efficiency: with 128 tokens, it reaches rFID 0.43, PSNR 25.03 and SSIM 0.806, surpassing MAETok; with only 64 tokens, it still achieves competitive results, rFID 0.75, PSNR 23.10 and SSIM 0.738, outperforming SoftVQ-VAE. For generation, SiT-B trained with MacTok using 128 tokens achieves a gFID of 3.15, exceeding all other continuous tokenizers.

Table 3. Comparison of continuous tokenizers. MacTok attains a better balance between compression and reconstruction quality, while delivering the best generation performance. All generation results are reported with optimal CFG scales. 

<table><tr><td rowspan="2">Tokenizer</td><td rowspan="2">#Tokens↓</td><td colspan="3">Tok.</td><td colspan="2">SiT-B</td></tr><tr><td>rFID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>gFID↓</td><td>IS↑</td></tr><tr><td>VA-VAE</td><td>256</td><td>0.28</td><td>26.30</td><td>0.846</td><td>4.33</td><td>222.1</td></tr><tr><td>MAETok</td><td>128</td><td>0.48</td><td>23.61</td><td>0.763</td><td>4.77</td><td>243.2</td></tr><tr><td>SoftVQ-VAE</td><td>64</td><td>0.88</td><td>22.13</td><td>0.706</td><td>4.09</td><td>256.9</td></tr><tr><td>SD-VAE</td><td>1024</td><td>0.61</td><td>26.04</td><td>0.834</td><td>7.66</td><td>187.5</td></tr><tr><td>MAR-VAE</td><td>256</td><td>0.53</td><td>-</td><td>-</td><td>6.26</td><td>177.5</td></tr><tr><td>l-DeTok</td><td>256</td><td>0.68</td><td>-</td><td>-</td><td>5.13</td><td>207.3</td></tr><tr><td rowspan="2">MacTok</td><td>64</td><td>0.75</td><td>23.10</td><td>0.738</td><td>3.22</td><td>262.8</td></tr><tr><td>128</td><td>0.43</td><td>25.03</td><td>0.806</td><td>3.15</td><td>258.3</td></tr></table>

# 4.4. Latent Space Analysis

We analyze how MacTok avoids posterior collapse and learns a semantically structured latent space.

Latent Space Visualization. Fig. 5 compares three latent spaces: (a) a collapsed KL-VAE baseline, (b) MacTok with masking but without representation alignment, and (c) the full MacTok model. In (a), the KL-VAE exhibits severe posterior collapse, forming an isotropic and uninformative latent distribution that collapses to the prior, which consequently fails to reconstruct meaningful and recognizable images. Compared with (c), the latent space in (b) appears more compact and less dispersed across the feature space, as image-level masking imposes an implicit semantic prior that encourages the model to preserve finer visual details and structural information, thereby providing an explanation for MacTok’s superior reconstruction performance. Finally, (c) incorporates global and local representation alignment, resulting in a more well-structured and discriminative latent space where similar semantic concepts cluster together. More visualizations are provided in Appendix C.1.

![](images/feb1198380788c534e694b81a0189ece570fe7a203e846b75a57479ad5044d81.jpg)

<details>
<summary>heatmap</summary>

| Region | Value |
|--------|-------|
| Central | High |
| Northeast | Medium-High |
| Southeast | Low |
| Southwest | Medium |
| Northwest | Low |
</details>

(a) Collapsed

![](images/aaf722f685528783c792ffd3bbd84fa62a6a914e5ceebcba420e8c1fadbdb125.jpg)

<details>
<summary>pie</summary>

| Category | Value |
|---|---|
| Red | 100 |
| Blue | 85 |
| Green | 70 |
| Orange | 65 |
</details>

(b) MacTok-128 w/o RA.

![](images/d3a9a804e26989d6810e3614a45223bfa9bdbb9a08df8f524ff62ef0facb7408.jpg)

<details>
<summary>scatter</summary>

| Group | X Coordinate | Y Coordinate |
|-------|--------------|--------------|
| Blue  | -1.2         | 0.8          |
| Blue  | 0.5          | 0.3          |
| Blue  | 1.1          | 0.6          |
| Blue  | -0.8         | 0.4          |
| Orange| 0.7          | 0.9          |
| Orange| 1.3          | 0.5          |
| Orange| -0.5         | 0.7          |
| Orange| 0.9          | 0.2          |
| Red   | 0.6          | 0.7          |
| Red   | -0.3         | 0.8          |
| Red   | 0.4          | 0.6          |
| Red   | -0.6         | 0.9          |
</details>

(c) MacTok-128

Figure 5. Visualization of latent space from (a) Collapsed; (b) MacTok-128 trained without representation alignment; (c) MacTok-128   
![](images/5150b3c17604e062c1d1c027f89fff488222dc3e8f8f887fab434af48599689e.jpg)

<details>
<summary>bar_line</summary>

| Model | gFID | Accuracy |
|---|---|---|
| MacTok-128 | 14 | 50 |
| MacTok-64 | 15 | 53 |
| SoftVQ-VAE | 20 | 41 |
| MacTok-128 w/o RA | 37 | 30 |
| MacTok-64 w/o RA | 39 | 30 |
</details>

(a) gFID vs. Accuracy.

![](images/835cf1c138ac232cf73a139896d63374e45c727e16e04f6f284547e3f0ed6c98.jpg)

<details>
<summary>line</summary>

| Training Steps | MacTok-128 w/o RA | REPA | SoftVQ-VAE | MacTok-128 |
| -------------- | ---------------- | ---- | ---------- | ---------- |
| 50k            | 78               | 50   | 58         | 50         |
| 100k           | 60               | 40   | 38         | 30         |
| 200k           | 48               | 35   | 30         | 20         |
| 400k           | 40               | 25   | 22         | 15         |
| 500k           | 35               | 20   | 20         | 12         |
</details>

(b) gFID vs. Training steps.   
Figure 6. Linear probing accuracy (a) of ImageNet-1k val. and generation performance (b) of MacTok with training steps.

Linear Probing and Generation Performance. We evaluate latent space quality by correlating linear probing accuracy, which measures how well latent features linearly separate semantic categories, with generative performance. As shown in Fig. 6a, higher probing accuracy indicates stronger semantic retention and better generation quality. Fig. 6b further shows that MacTok not only surpasses other strong baselines in generation fidelity [5, 61], but also exhibits significantly faster convergence during training.

# 4.5. Ablation Studies

We conduct ablation studies to analyze the effect of key design choices in MacTok. Unless otherwise noted, experiments use MacTok-128 with SiT-B trained for 500K steps. Mask Ratio. As shown in Tab. 4, the gFID initially de-Table 4. Ablation on maximum mask ratio M (w/o Decoder Finetuning). MacTok is evaluated over mask ratios from 0.4 to 0.8 and different DINO-guided semantic masking settings: “dino 100%” denotes full use of DINO-guided semantic masking, while “dino 50%” applies random and semantic masking with equal probability. Generation performance is reported without CFG.

<table><tr><td rowspan="2">M</td><td colspan="3">Tok.</td><td colspan="2">SiT-B</td></tr><tr><td>rFID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>gFID↓</td><td>IS↑</td></tr><tr><td>0.4</td><td>0.49</td><td>24.95</td><td>0.809</td><td>15.49</td><td>78.6</td></tr><tr><td>0.5</td><td>0.54</td><td>25.02</td><td>0.812</td><td>14.87</td><td>81.1</td></tr><tr><td>0.6</td><td>0.57</td><td>24.95</td><td>0.808</td><td>14.79</td><td>81.8</td></tr><tr><td>0.7</td><td>0.56</td><td>24.93</td><td>0.808</td><td>14.59</td><td>82.5</td></tr><tr><td>0.8</td><td>0.59</td><td>24.89</td><td>0.805</td><td>14.92</td><td>80.8</td></tr><tr><td>0.7+dino 100%</td><td>0.66</td><td>24.92</td><td>0.809</td><td>14.84</td><td>81.1</td></tr><tr><td>0.7+dino 50%</td><td>0.57</td><td>24.91</td><td>0.808</td><td>13.95</td><td>84.8</td></tr><tr><td>+Decoder Fine-tuning</td><td>0.43</td><td>25.03</td><td>0.806</td><td>13.90</td><td>85.1</td></tr></table>

Table 5. Ablation of different modules (w/ Decoder Fine-tuning). We report the impact of each module on MacTok’s reconstruction and generation performance with optimal CFG scales. 

<table><tr><td>Setup</td><td colspan="3">Tok</td><td colspan="2">SiT-B</td></tr><tr><td>MacTok</td><td>rFID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>gFID↓</td><td>IS↑</td></tr><tr><td>+ random mask</td><td>0.58</td><td>24.30</td><td>0.779</td><td>6.01</td><td>234.8</td></tr><tr><td>+ local alignment</td><td>0.44</td><td>25.02</td><td>0.806</td><td>3.53</td><td>241.9</td></tr><tr><td>+ semantic mask</td><td>0.43</td><td>24.97</td><td>0.805</td><td>3.32</td><td>249.2</td></tr><tr><td>+ global alignment</td><td>0.43</td><td>25.03</td><td>0.806</td><td>3.15</td><td>258.3</td></tr></table>

creases and then increases as the M grows. A moderate M of 70% achieves the best generation performance, indicating that stronger masking enhances the robustness and information richness of latent representations. Applying random and semantic masking with equal probability further improves generation quality. Although stronger masking slightly reduces reconstruction fidelity, this degradation can be mitigated through decoder fine-tuning (see Appendix C.2 for more details), which restores image quality while preserving the learned semantic structure.

Key Modules. Tab. 5 reports the impact of each module sequentially added to MacTok under decoder fine-tuning and optimal CFG. Random masking mitigates posterior collapse in KL-based tokenizers. Local alignment improves both reconstruction and generation by imposing structured organization in the latent space. DINO-guided semantic masking strengthens semantic robustness and improves gFID and IS. Global alignment further enforces high-level semantic consistency through effective regularization. Combining all modules yields the best overall performance.

# 5. Conclusion

We introduced MacTok, a continuous tokenizer driven by masking, which effectively mitigates posterior collapse and achieves efficient and high-fidelity image tokenization. By combining random and DINO-guided semantic masking, MacTok learns robust and semantically structured latent representations, enabling strong generation and reconstruction with only 64 or 128 tokens. Our findings demonstrate that posterior collapse in continuous tokenizers can be mitigated through masking, and learning a more discriminative latent space is key to advancing generative modeling.

# References

[1] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22669–22679, 2023. 6, 7, 14, 16   
[2] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou, ´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021. 5, 13   
[3] Huiwen Chang, Han Zhang, Lu Jiang, Ce Liu, and William T Freeman. Maskgit: Masked generative image transformer. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11315–11325, 2022. 1, 3, 6, 7, 16   
[4] Hao Chen, Yujin Han, Fangyi Chen, Xiang Li, Yidong Wang, Jindong Wang, Ze Wang, Zicheng Liu, Difan Zou, and Bhiksha Raj. Masked autoencoders are effective tokenizers for diffusion models. In Forty-second International Conference on Machine Learning, 2025. 2, 3, 4, 6, 7, 16   
[5] Hao Chen, Ze Wang, Xiang Li, Ximeng Sun, Fangyi Chen, Jiang Liu, Jindong Wang, Bhiksha Raj, Zicheng Liu, and Emad Barsoum. Softvq-vae: Efficient 1-dimensional continuous tokenizer. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28358–28370, 2025. 1, 3, 4, 5, 6, 7, 8, 13, 14, 16   
[6] Qi Chen, Guanghao Li, Xiangyang Xue, and Jian Pu. Multilio: A lightweight multiple lidar-inertial odometry system. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 13748–13754. IEEE, 2024. 3   
[7] Xi Chen, Diederik P Kingma, Tim Salimans, Yan Duan, Prafulla Dhariwal, John Schulman, Ilya Sutskever, and Pieter Abbeel. Variational lossy autoencoder. arXiv preprint arXiv:1611.02731, 2016. 1, 2, 3, 4   
[8] Navneet Dalal and Bill Triggs. Histograms of oriented gradients for human detection. In 2005 IEEE computer society conference on computer vision and pattern recognition (CVPR’05), pages 886–893. Ieee, 2005. 3   
[9] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 5   
[10] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. Advances in neural information processing systems, 34:8780–8794, 2021. 6, 7, 16   
[11] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 4

[12] Alexey Dosovitskiy and Thomas Brox. Generating images with perceptual similarity metrics based on deep networks. Advances in neural information processing systems, 29, 2016. 5   
[13] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 12873–12883, 2021. 1, 3, 5   
[14] Hao Fu, Chunyuan Li, Xiaodong Liu, Jianfeng Gao, Asli Celikyilmaz, and Lawrence Carin. Cyclical annealing schedule: A simple approach to mitigating kl vanishing. arXiv preprint arXiv:1903.10145, 2019. 2   
[15] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Mdtv2: Masked diffusion transformer is a strong image synthesizer. arXiv preprint arXiv:2303.14389, 2023. 6, 16   
[16] Xin Gao and Jian Pu. Deep incomplete multi-view learning via cyclic permutation of vaes. In The Thirteenth International Conference on Learning Representations, 2025. 4   
[17] Xin Gao, Jiyao Liu, Guanghao Li, Yueming Lyu, Jianxiong Gao, Weichen Yu, Ningsheng Xu, Liang Wang, Caifeng Shan, Ziwei Liu, et al. Good: Training-free guided diffusion sampling for out-of-distribution detection. In The Thirtyninth Annual Conference on Neural Information Processing Systems, 2025. 3   
[18] Ian Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, and Yoshua Bengio. Generative adversarial networks. Communications of the ACM, 63(11):139–144, 2020. 5   
[19] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable ´ vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16000– 16009, 2022. 2   
[20] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017. 6   
[21] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In International conference on learning representations, 2017. 2, 12   
[22] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 6   
[23] Justin Johnson, Alexandre Alahi, and Li Fei-Fei. Perceptual losses for real-time style transfer and super-resolution. In European conference on computer vision, pages 694–711. Springer, 2016. 5   
[24] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019. 7, 16   
[25] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 1, 5, 12

[26] Tuomas Kynka¨anniemi, Tero Karras, Samuli Laine, Jaakko ¨ Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019. 6   
[27] Anders Boesen Lindbo Larsen, Søren Kaae Sønderby, Hugo Larochelle, and Ole Winther. Autoencoding beyond pixels using a learned similarity metric. In International conference on machine learning, pages 1558–1566. PMLR, 2016. 5   
[28] Doyup Lee, Chiheon Kim, Saehoon Kim, Minsu Cho, and Wook-Shin Han. Autoregressive image generation using residual quantization. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 11523–11532, 2022. 6, 7, 16   
[29] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. Repa-e: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025. 3   
[30] Guanghao Li, Yu Cao, Qi Chen, Xin Gao, Yifan Yang, and Jian Pu. Papl-slam: Principal axis-anchored monocular point-line slam. IEEE Robotics and Automation Letters, 2025. 3   
[31] Guanghao Li, Qi Chen, Sijia Hu, Yuxiang Yan, and Jian Pu. Constrained gaussian splatting via implicit tsdf hash grid for dense rgb-d slam. IEEE Transactions on Artificial Intelligence, 2025.   
[32] Guanghao Li, Qi Chen, Yuxiang Yan, and Jian Pu. Ec-slam: Effectively constrained neural rgb-d slam with tsdf hash encoding and joint optimization. Pattern Recognition, 170: 112034, 2026.   
[33] Guanghao Li, Kerui Ren, Linning Xu, Zhewen Zheng, Changjian Jiang, Xin Gao, Bo Dai, Jian Pu, Mulin Yu, and Jiangmiao Pang. Artdeco: Toward high-fidelity on-the-fly reconstruction with hierarchical gaussian structure and feedforward guidance. In The Fourteenth International Conference on Learning Representations, 2026. 3   
[34] Tianhong Li, Yonglong Tian, He Li, Mingyang Deng, and Kaiming He. Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems, 37:56424–56445, 2024. 1, 3, 6, 7, 16   
[35] Xiang Li, Kai Qiu, Hao Chen, Jason Kuen, Jiuxiang Gu, Bhiksha Raj, and Zhe Lin. Imagefolder: Autoregressive image generation with folded tokens. arXiv preprint arXiv:2410.01756, 2024. 1, 2, 3, 6, 16   
[36] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024. 1, 3, 6, 7, 13, 14, 16   
[37] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy ´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 2, 3, 4, 5, 13   
[38] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 4195–4205, 2023. 1, 3, 6, 7, 14, 16

[39] Dong Qian and William K Cheung. Enhancing variational autoencoders with mutual information neural estimation for text generation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 4047–4057, 2019. 2   
[40] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021. 3   
[41] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image ¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022. 1, 6, 7, 14, 16   
[42] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Unet: Convolutional networks for biomedical image segmentation. In International Conference on Medical image computing and computer-assisted intervention, pages 234–241. Springer, 2015. 3   
[43] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. Advances in neural information processing systems, 29, 2016. 6   
[44] Fengyuan Shi, Zhuoyan Luo, Yixiao Ge, Yujiu Yang, Ying Shan, and Limin Wang. Scalable image tokenization with index backpropagation quantization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16037–16046, 2025. 3   
[45] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024. 3, 6, 16   
[46] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems, 37:84839–84865, 2024. 3, 5, 6, 13, 16   
[47] Hung-Yu Tseng, Lu Jiang, Ce Liu, Ming-Hsuan Yang, and Weilong Yang. Regularizing generative adversarial networks under limited data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 7921–7931, 2021. 5, 13   
[48] Aaron Van den Oord, Nal Kalchbrenner, Lasse Espeholt, Oriol Vinyals, Alex Graves, et al. Conditional image generation with pixelcnn decoders. Advances in neural information processing systems, 29, 2016. 3   
[49] Aaron Van Den Oord, Oriol Vinyals, et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017. 1, 3   
[50] Luting Wang, Yang Zhao, Zijian Zhang, Jiashi Feng, Si Liu, and Bingyi Kang. Image understanding makes for a good tokenizer for image generation. Advances in Neural Information Processing Systems, 37:31015–31035, 2024. 3   
[51] Mark Weber, Lijun Yu, Qihang Yu, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. Maskbit:

Embedding-free image generation via bit tokens. arXiv preprint arXiv:2409.16211, 2024. 6, 7, 16   
[52] Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, et al. Representation entanglement for generation: Training diffusion transformers is much easier than you think. arXiv preprint arXiv:2507.01467, 2025. 3   
[53] Tianwei Xiong, Jun Hao Liew, Zilong Huang, Jiashi Feng, and Xihui Liu. Gigatok: Scaling visual tokenizers to 3 billion parameters for autoregressive image generation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18770–18780, 2025. 6, 16   
[54] Yuxiang Yan, Boda Liu, Jianfei Ai, Qinbu Li, Ru Wan, and Jian Pu. Pointssc: A cooperative vehicle-infrastructure point cloud benchmark for semantic scene completion. In 2024 IEEE International Conference on Robotics and Automation (ICRA), pages 17027–17034. IEEE, 2024. 4   
[55] Yuxiang Yan, Zhiyuan Zhou, Xin Gao, Guanghao Li, Shenglin Li, Jiaqi Chen, Qunyan Pu, and Jian Pu. Learning spatial-aware manipulation ordering. arXiv preprint arXiv:2510.25138, 2025. 4   
[56] Jiawei Yang, Tianhong Li, Lijie Fan, Yonglong Tian, and Yue Wang. Latent denoising makes good visual tokenizers. arXiv preprint arXiv:2507.15856, 2025. 2, 6, 7, 16   
[57] Jingfeng Yao, Bin Yang, and Xinggang Wang. Reconstruction vs. generation: Taming optimization dilemma in latent diffusion models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15703–15712, 2025. 2, 3, 4, 6, 7, 13, 16   
[58] Jiahui Yu, Xin Li, Jing Yu Koh, Han Zhang, Ruoming Pang, James Qin, Alexander Ku, Yuanzhong Xu, Jason Baldridge, and Yonghui Wu. Vector-quantized image modeling with improved vqgan. arXiv preprint arXiv:2110.04627, 2021. 4, 6, 7, 16   
[59] Lijun Yu, Jose Lezama, Nitesh B Gundavarapu, Luca Ver- ´ sari, Kihyuk Sohn, David Minnen, Yong Cheng, Vighnesh Birodkar, Agrim Gupta, Xiuye Gu, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023. 3, 7, 16   
[60] Qihang Yu, Mark Weber, Xueqing Deng, Xiaohui Shen, Daniel Cremers, and Liang-Chieh Chen. An image is worth 32 tokens for reconstruction and generation. Advances in Neural Information Processing Systems, 37:128940– 128966, 2024. 1, 4, 6, 7, 16   
[61] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. arXiv preprint arXiv:2410.06940, 2024. 2, 3, 4, 6, 8, 14, 16   
[62] Kaiwen Zha, Lijun Yu, Alireza Fathi, David A Ross, Cordelia Schmid, Dina Katabi, and Xiuye Gu. Languageguided image tokenization for generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 15713–15722, 2025. 3, 6, 7, 16   
[63] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pages 11975–11986, 2023. 4

[64] Han Zhang, Zizhao Zhang, Augustus Odena, and Honglak Lee. Consistency regularization for generative adversarial networks. arXiv preprint arXiv:1910.12027, 2019. 5, 13   
[65] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018. 5   
[66] Shengjia Zhao, Jiaming Song, and Stefano Ermon. Infovae: Information maximizing variational autoencoders. arXiv preprint arXiv:1706.02262, 2017. 2   
[67] Shengyu Zhao, Zhijian Liu, Ji Lin, Jun-Yan Zhu, and Song Han. Differentiable augmentation for data-efficient gan training. Advances in neural information processing systems, 33:7559–7570, 2020. 5, 13   
[68] Lei Zhu, Fangyun Wei, Yanye Lu, and Dong Chen. Scaling the codebook size of vq-gan to 100,000 with a utilization rate of 99%. Advances in Neural Information Processing Systems, 37:12612–12635, 2024. 3   
[69] Shaobin Zhuang, Yiwei Guo, Canmiao Fu, Zhipeng Huang, Zeyue Tian, Fangyikang Wang, Ying Zhang, Chen Li, and Yali Wang. Wetok: Powerful discrete tokenization for high-fidelity visual reconstruction. arXiv preprint arXiv:2508.05599, 2025. 6, 16

# A. Additional Theoretical and Empirical Analysis

# A.1. KL-VAE Formulation

In this section, we provide a detailed description of $\mathrm { K L } -$ VAE [21, 25]. KL-VAE models both the prior and posterior distributions as Gaussians. Specifically, the prior p(z) is defined as an isotropic unit Gaussian $\mathcal { N } ( 0 , \bf { I } )$ . The posterior distribution $q _ { \phi } ( z | x )$ is parameterized by an encoder that predicts the mean $\mu _ { \phi } ( x )$ and variance $\sigma _ { \phi } ^ { 2 } ( x )$ . Using the reparameterization trick, the latent variable z is obtained as

$$
\begin{array}{l} q _ {\phi} (z | x) = \mathcal {N} (z; \mu_ {\phi} (x), \sigma_ {\phi} ^ {2} (x)), \tag {8} \\ z = \mu_ {\phi} (x) + \sigma_ {\phi} (x) \odot \epsilon , \quad \epsilon \sim \mathcal {N} (0, \mathbf {I}). \\ \end{array}
$$

The KL divergence between the posterior and the prior is given by

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{KL}} (q _ {\phi} (z) \| p (z)) \\ = \int q _ {\phi} (z | x) \left(\log q _ {\phi} (z | x) - \log p (z)\right) d z \tag {9} \\ = - \frac {1}{2} \sum_ {i = 1} ^ {D} \left(1 + \log (\sigma_ {i} ^ {2}) - \mu_ {i} ^ {2} - \sigma_ {i} ^ {2}\right), \\ \end{array}
$$

where D denotes the dimensionality of the latent space.The KL term plays a crucial role in the overall training objective, i.e., the Evidence Lower Bound (ELBO). Specifically, it acts as a regularizer that enforces the learned posterior $q _ { \phi } ( z | x )$ to stay close to the prior $p ( z )$ , thereby encouraging smooth and continuous representations.

# A.2. Mitigating Posterior Collapse via Masked Reconstruction

# A.2.1. Corrupted Evidence Lower Bound (ELBO)

Standard VAE training optimizes the Evidence Lower Bound (ELBO):

$$
\mathcal {L} _ {\mathrm{ELBO}} = \mathbb {E} _ {q _ {\phi} (Z | X)} [ \log p _ {\theta} (X | Z) ] - \beta \cdot \mathrm{KL} (q _ {\phi} (Z | X) \| p (Z)), \tag {10}
$$

which balances reconstruction (first term) against regularization of the posterior $q _ { \phi } ( Z | X )$ toward the prior $p ( Z )$ (second term). Under strong compression and large $\beta ,$ this KL penalty can push $q _ { \phi } ( Z | X )$ too close to $p ( Z )$ , causing posterior collapse: $q _ { \phi } ( Z | X ) \approx p ( Z )$ . At this point, $Z$ carries no information about X, and the decoder effectively becomes an unconditional model $p _ { \theta } ( X )$ , leading to poor reconstructions.

MacTok takes a different approach by training on masked images. Let $\tilde { X }$ be the masked image after applying a stochastic masking operation $C _ { m } ( { \tilde { X } } | X )$ with ratio m. The encoder sees only $\tilde { X }$ , but the decoder must still reconstruct the full image X. This gives us the corrupted ELBO:

$$
\begin{array}{l} \mathcal {L} _ {\text { corrupted }} = \mathbb {E} _ {X, \tilde {X} \sim C _ {m} (\cdot | X)} [ \mathbb {E} _ {q _ {\phi} (Z | \tilde {X})} [ - \log p _ {\theta} (X | Z) ] \\ + \beta \cdot \mathrm{KL} (q _ {\phi} (Z | \tilde {X}) \| p (Z)) ]. \tag {11} \\ \end{array}
$$

The key difference is this information asymmetry: the encoder only gets partial information X˜ , while the decoder has to predict everything, including what was masked. This forces $Z$ to actually encode useful information from X˜ —otherwise the decoder has no way to reconstruct the missing parts.

# A.2.2. Why Collapsed Solutions Become Suboptimal

Consider what happens when the posterior collapses: $q _ { \phi } ( Z | \tilde { X } ) = p ( Z )$ . Now $Z$ is independent of both $\tilde { X }$ and $X ,$ so:

$$
\begin{array}{l} \mathbb {E} _ {q _ {\phi} (Z | \tilde {X}) = p (Z)} [ - \log p _ {\theta} (X | Z) ] = \mathbb {E} _ {Z \sim p (Z)} [ - \log p _ {\theta} (X | Z) ] \\ = \mathbb {E} _ {Z \sim p (Z)} [ - \log p _ {\theta} (X) ] \\ = - \log p _ {\theta} (X), \tag {12} \\ \end{array}
$$

where $p _ { \theta } ( X )$ is just the unconditional image distribution.

We can break this down by what’s visible versus what’s masked:

$$
- \log p _ {\theta} (X) = - \log p _ {\theta} (X _ {\text { visible }}) - \log p _ {\theta} (X _ {\text { masked }}). \tag {13}
$$

The problem is the second term: $- \log p _ { \theta } ( X _ { \mathrm { m a s k e d } } )$ . Without any context, the decoder has to guess what’s in the masked regions based purely on dataset statistics—maybe “skies are usually blue” or “grass is usually green.” But this fails for any specific image. As we mask more pixels (higher m), this blind guessing gets worse and $- \log p _ { \theta } ( X )$ shoots up.

Compare this to when Z actually encodes information from $\tilde { X } .$ . Now the decoder can use contextual clues—if it sees grass and trees in the visible parts, it knows this is probably an outdoor scene; if the visible colors are warm, maybe it’s sunset. This capability of recovering latent details from partial or degraded visual cues shares underlying principles with robust image processing pipelines designed for severely suboptimal conditions. This gives much better predictions:

$$
- \log p _ {\theta} (X | Z) = - \log p _ {\theta} (X _ {\text { visible }} | Z) - \log p _ {\theta} (X _ {\text { masked }} | Z), \tag {14}
$$

where − log $p _ { \theta } ( X _ { \mathrm { m a s k e d } } | Z )$ is now significantly smaller because the decoder can make informed guesses based on what $Z$ encoded.

Let’s define the benefit of having an informative $Z$ as:

$$
\Delta \triangleq - \log p _ {\theta} (X) - \mathbb {E} _ {q _ {\phi} (Z | \tilde {X})} [ - \log p _ {\theta} (X | Z) ]. \tag {15}
$$

Larger ∆ means $Z$ is more useful. Now compare total losses:

$$
\text { Loss } _ {\text { collapse }} = - \log p _ {\theta} (X), \tag {16}
$$

$$
\text { Loss } _ {\text { informative }} = \mathbb {E} _ {q _ {\phi} (Z | \tilde {X})} [ - \log p _ {\theta} (X | Z) ] + \beta \cdot \epsilon , \tag {17}
$$

where $\epsilon = \mathrm { K L } ( q _ { \phi } ( Z | \tilde { X } ) | | p ( Z ) ) > 0$ is the KL cost of keeping Z informative. The informative solution wins when:

$$
\Delta > \beta \cdot \epsilon . \tag {18}
$$

So the collapsed solution is suboptimal whenever $\beta < \Delta / \epsilon$ .

Here’s where masking matters: it directly increases $\Delta$ . As we mask more:

• Without context (collapsed case), predicting more masked pixels becomes exponentially harder, pushing − $. \log p _ { \theta } ( X )$ way up.   
• With context from $Z$ (informative case), we can still make reasonable predictions based on visible cues, so $\mathbb { E } [ - \log p _ { \theta } ( X | Z ) ]$ stays relatively controlled.

Higher m widens the gap $\Delta ,$ which means informative posteriors stay optimal for a broader range of $\beta$ (Eq. 18).

Without masking, there’s a loophole: the decoder can just copy local patterns from the input. Even if Z is mostly useless, reconstructions still look okay, so $\Delta$ stays small and collapse becomes competitive. Masking closes this loophole—the decoder has to use $Z$ to fill in the missing parts, which keeps information flowing through the latent space even under strong regularization.

In conclusion, masking prevents collapse through a simple mechanism. First, it makes the reconstruction task harder, so $Z$ needs to be informative. Second, if Z collapses and becomes useless, the decoder is forced to blindly guess large portions of the image, incurring huge losses. Third, by increasing $\Delta .$ , masking ensures that keeping Z informative remains the better strategy across a wide range of $\beta$ values. This is how MacTok maintains meaningful continuous tokens even with aggressive compression and regularization.

# A.3. Visualization of KL Divergence Dynamics

As illustrated in Fig. 7, applying latent token masking postpones posterior collapse compared to the conventional KL-VAE baseline. Nevertheless, this improvement is transient, as the model ultimately converges to a degenerate solution over the course of training. In contrast, masking image tokens yields a markedly steadier optimization process and produces more resilient latent representations. We attribute this behavior to the fact that image masking encourages both the encoder and decoder to reason over incomplete visual inputs, thereby encouraging the latent space to encode more structural and semantic information.

![](images/cc8c5ec1cffaf439688843f427d88cdbe540b44f4f4a721d4beaa0f84b254040.jpg)

<details>
<summary>line</summary>

| Step | w/o mask | w/ latent mask | w/ image mask |
| ---- | -------- | -------------- | ------------- |
| 0k   | 0.004    | 0.008          | 0.006         |
| 20k  | 0.008    | 0.009          | 0.007         |
| 40k  | 0.009    | 0.010          | 0.008         |
| 60k  | 0.010    | 0.010          | 0.008         |
| 80k  | 0.010    | 0.010          | 0.008         |
| 100k | 0.012    | 0.012          | 0.012         |
</details>

Figure 7. Comparison of different masking strategies of the KL loss curve.

# B. Additional Implementation Details

In this section, we present additional implementation details for tokenizer training and downstream generative model training.

# B.1. Implementation Details of MacTok

We train the MacTok tokenizers on ImageNet at resolution of 256×256 for 250K iterations with a batch size of 256 and at 512×512 for 500K iterations with a batch size of 128. Data augmentation includes horizontal flipping and center cropping. We use AdamW optimizer with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } =$ 0.95, a weight decay of $1 \times 1 0 ^ { - 4 }$ . The learning rate follows a cosine annealing schedule, peaking at $1 \times 1 0 ^ { - 4 }$ and preceded by a linear warm-up of 5K and 10K steps for the 256 and 512 resolutions. To improve the stability of adversarial learning, we employ a frozen DINO-S [2, 37] network as the discriminator as in [5, 46] and incorporate the adaptive weighting scheme. Moreover, we enhance discriminator training by introducing DiffAug [67], consistency regularization [64], and LeCAM regularization [47], as used in [5]. The regularization weights for the consistency and LeCAM terms are set to 4.0 and 0.001, respectively. The overall training objective follows common practice with loss weights $\lambda _ { 1 } = 1 . 0 , \lambda _ { 2 } = 0 . 2 , \lambda _ { 3 } = 1 \times 1 0 ^ { - 6 }$ , and $\lambda _ { 4 } = 0 . 1$ .

# B.2. Implementation Details of Generative Models

LightningDiT [57] The training configuration of our LightningDiT models closely follows the original setup. As our model operates on 1D latent tokens, we set the patch size to 1. LightningDiT-XL is trained with a constant learning rate of $2 \times 1 0 ^ { - 4 }$ and a global batch size of 1024. We adopt a cosine noise scheduler and rotary positional embeddings, consistent with the original implementation. In the main paper, we report results of LightningDiT-XL trained for 400K iterations. For conditional generation with classifier-free guidance (CFG), we use a guidance scale of 2.5 for LightningDiT models trained on MacTok with 128 tokens and 2.7 for those trained with 64 tokens. These values are selected via grid search based on gFID and IS metrics computed over 10K generated samples.

SiT [36] We follow the original training configuration of SiT, using a constant learning rate of $1 \times 1 0 ^ { - 4 }$ and a global batch size of 256. A linear learning rate scheduler is adopted, as it demonstrates better empirical performance in our setting. The main results are reported after 4M training iterations. For conditional generation with CFG, we set the guidance scale to 2.3 for SiT models trained on MacTok with 128 tokens and 2.4 for those trained with 64 tokens. Following REPA [61], the guidance interval is set to [0, 0.7] for CFG-based results. The optimal values are determined through grid search by evaluating gFID and IS over 10K generated samples.

![](images/ce45be74483a3f40b143563263a41b2944a71a0a3321d644ed3c91b8e286e096.jpg)

<details>
<summary>heatmap</summary>

| Category | Value |
| -------- | ----- |
| Low      | 100   |
| Medium   | 80    |
| High     | 95    |
</details>

(a) MacTok-64 w/o RA.

![](images/b0970deb0756330f2fd0f707e06c144fb027afaf26535ab8529c462bef113318.jpg)

<details>
<summary>scatter</summary>

| x | y | cluster |
| --- | --- | --- |
| 0.1 | 0.8 | blue |
| 0.2 | 0.75 | blue |
| 0.3 | 0.65 | blue |
| 0.4 | 0.55 | blue |
| 0.5 | 0.45 | blue |
| 0.6 | 0.35 | blue |
| 0.7 | 0.25 | blue |
| 0.8 | 0.15 | blue |
| 0.9 | 0.05 | blue |
| 0.15 | 0.95 | orange |
| 0.25 | 0.85 | orange |
| 0.35 | 0.75 | orange |
| 0.45 | 0.65 | orange |
| 0.55 | 0.55 | orange |
| 0.65 | 0.45 | orange |
| 0.75 | 0.35 | orange |
| 0.85 | 0.25 | orange |
| 0.95 | 0.15 | orange |
| 0.12 | 0.88 | red |
| 0.22 | 0.78 | red |
| 0.32 | 0.68 | red |
| 0.42 | 0.58 | red |
| 0.52 | 0.48 | red |
| 0.62 | 0.38 | red |
| 0.72 | 0.28 | red |
| 0.82 | 0.18 | red |
| 0.92 | 0.08 | red |
| 0.18 | 0.92 | green |
| 0.28 | 0.82 | green |
| 0.38 | 0.72 | green |
| 0.48 | 0.62 | green |
| 0.58 | 0.52 | green |
| 0.68 | 0.42 | green |
| 0.78 | 0.32 | green |
| 0.88 | 0.22 | green |
| 0.98 | 0.12 | green |
| 0.14 | 0.84 | yellow |
| 0.24 | 0.74 | yellow |
| 0.34 | 0.64 | yellow |
| 0.44 | 0.54 | yellow |
| 0.54 | 0.44 | yellow |
| 0.64 | 0.34 | yellow |
| 0.74 | 0.24 | yellow |
| 0.84 | 0.14 | yellow |
| 0.94 | 0.04 | yellow |
| 0.16 | 0.86 | pink |
| 0.26 | 0.76 | pink |
| 0.36 | 0.66 | pink |
| 0.46 | 0.56 | pink |
| 0.56 | 0.46 | pink |
| 0.66 | 0.36 | pink |
| 0.76 | 0.26 | pink |
| 0.86 | 0.16 | pink |
| 0.96 | 0.06 | pink |
| 0.13 | 0.83 | gray |
| 0.23 | 0.73 | gray |
| 0.33 | 0.63 | gray |
| 0.43 | 0.53 | gray |
| 0.53 | 0.43 | gray |
| 0.63 | 0.33 | gray |
| 0.73 | 0.23 | gray |
| 0.83 | 0.13 | gray |
| 0.93 | 0.03 | gray |
| 1.01 | -0.11 | white |
| -1.11, -1.21, -1.31, -1.41, -1.51, -1.61, -1.71, -1.81, -1.91, -2.11, -2.31, -2.51, -2.71, -3.91, -4.11, -4.31, -4.51, -4.71, -5.91, -6.11, -6.31, -6.51, -6.71, -7.91, -91, -111, -131, -151, -171, -191, -211, -231, -251, -271, -391, -411, -431, -451, -471, -591, -611, -711, -731, -751, -771, -891, -991, -1191, -1391, -1691, -1991, -2291, -2591, -2891, -3991, -4291, -4591, -4891, -5991, -6291, -7491, -7791, -8991, -9291, -9591, -9991, -11991, -13991, -16991, -19991, -22991, -25991, -28991, -32991, -35991, -39991, -42991, -45991, -48991, -52991, -55991, -59991, -62991, -65991, -68991, -72991, -75991, -78991, -82991, -85991, -88991, -92991, -95991, -98991, -102991, -122991, -132991, -142991, -152991, -162991, -172991, -182991, -202991, -222991, -242991, -262991, -282991, -302991, -322991, -342991, -362991, -382991, -402991, -422991, -442991, -462991, -482991, -502991, -522991, -542991, -562991, -582991, -602991, -622991, -642991, -662991, -682991, -702991, -722991, -742991, -762991, -782991, -802991, -822991, -842991, -862991, -882991, -902991, -922991, -942991, -962991, -982   \end{array}
</details>

(b) MacTok-64   
Figure 8. Visualization of laten space from (a) MacTok-64 trained without Representation alignment; (b) MacTok-64

# C. Additional Results

In this appendix, we provide supplementary evidence to support the effectiveness of our approach. Specifically, we include further visualizations of the latent token space, more ablation studies, extended quantitative evaluations of generative models trained on MacTok, and additional qualitative examples of reconstructed and generated images. These results complement the main paper by highlighting the structural organization of the latent space, the generative fidelity across different resolutions and token settings.

# C.1. Latent Space Visualization

Fig. 8 illustrates the UMAP projection of the latent representations obtained with 64 tokens. We compare the latent space learned by MacTok-64 with and without Representation alignment (RA). As shown, MacTok-64 with Representation alignment generates more structured and separable embeddings compared to the model trained without alignment. This visualization confirms that MacTok effectively organizes the latent space with fewer tokens, supporting downstream tasks such as linear probing and generative modeling, and showing great promise for broader spatial perception applications that require dense structural consistency.

# C.2. Ablation Study

Decoder Fine-tuning. Tab. 6a reports MacTok’s performance when freezing the encoder and fine-tuning only the decoder without masking. Specifically, the encoder is frozen and the decoder is trained for 10 epochs without mask. This strategy notably improves rFID and slightly enhances gFID, indicating that decoder fine-tuning effectively restores reconstruction quality degraded by high mask ratios while preserving the latent space.

Table 6. Ablation studies of decoder fine-tuning and model size, showing their effects on MacTok’s performance 

<table><tr><td>Tokenizer</td><td>rFID↓</td><td>gFID↓</td></tr><tr><td>MacTok-64</td><td>0.93</td><td>3.28</td></tr><tr><td>+FT</td><td>0.75</td><td>3.22</td></tr><tr><td>MacTok-128</td><td>0.57</td><td>3.19</td></tr><tr><td>+FT</td><td>0.43</td><td>3.15</td></tr></table>

(a) Decoder fine-tuning.

<table><tr><td>Model Size</td><td>#Params</td><td>rFID</td></tr><tr><td>MacTok-S</td><td>45M</td><td>0.78</td></tr><tr><td>MacTok-B</td><td>176M</td><td>0.57</td></tr><tr><td>MacTok-BL</td><td>391M</td><td>0.57</td></tr></table>

(b) MacTok model size.

Model Size. Tab. 6b evaluates MacTok model size on ImageNet at 256×256. MacTok-B significantly outperforms MacTok-S, whereas further scaling does not yield additional gains. Consequently, MacTok-B is adopted as the default. For 512×512 generation with 64 tokens, we use MacTok-BL to ensure fair comparison with SoftVQ-VAE and mitigate reconstruction degradation at higher resolution.

# C.3. Main Results

We present the complete quantitative results, including both precision and recall, for the ImageNet 256×256 and 512×512 benchmarks in Tab. 7 and Tab. 8, respectively. All evaluations are conducted on SiT-XL models trained for 4M steps and LightningDiT-XL models trained for 400K steps. Notably, our models achieve state-of-the-art generative performance at 512×512 resolution and deliver results comparable to leading approaches at 256×256 resolution. Moreover, our models exhibit superior conditional gFID scores even without applying classifier-free guidance (CFG), outperforming SoftVQ-VAE [5] and other vanilla generative baselines [1, 36, 38, 41, 61] that utilize at least 256 or 1024 tokens. We further include results measured across different training durations, as summarized in Tab. 9.

# C.4. Reconstruction Visualization

We present the reconstruction results of MacTok using 64 and 128 latent tokens in Fig. 9 and Fig. 10, respectively. As shown, increasing the number of tokens leads to finer spatial details and improved texture fidelity, demonstrating the scalability of MacTok’s latent representation. In contrast, reconstructions from collapsed baselines (see Fig. 11) fail to recover meaningful visual content, indicating that posterior collapse severely limits the model’s representational capacity. MacTok’s semantically structured latent space effectively preserves both global layout and local semantics, resulting in faithful and perceptually consistent reconstructions even under limited token budgets. These visualizations complement the quantitative evaluation in the main paper and further verify the robustness of our latent modeling strategy.

# C.5. Generation Visualization

More visualizations of LightningDiT-X and SiT-XL trained on MacTok with 64 and 128 tokens are provided here.

Table 7. System-level comparison on ImageNet 256×256 conditional generation. We report both Precision and Recall under classifierfree guidance (CFG) and non-CFG settings. “# Params (G)” denotes generator parameters; “Tok. Model” refers to the tokenizer model type; “Token Type” indicates 1D or 2D tokenization; “# Params (T)” denotes tokenizer parameters; and “# Tokens” represents the number of latent tokens. 

<table><tr><td rowspan="2">Method</td><td rowspan="2"># Params(G)</td><td rowspan="2">Tok. Model</td><td rowspan="2">Token Type</td><td rowspan="2"># Params(T)</td><td rowspan="2">#Tokens↓</td><td rowspan="2">Tok. rFID↓</td><td colspan="4">w/o CFG</td><td colspan="4">w/ CFG</td></tr><tr><td>gFID↓</td><td>IS↑</td><td>Prec↑</td><td>Recall↑</td><td>gFID↓</td><td>IS↑</td><td>Prec↑</td><td>Recall↑</td></tr><tr><td colspan="15">Auto-regressive</td></tr><tr><td>ViT-VQGAN [58]</td><td>1.7B</td><td>VQ</td><td>2D</td><td>64M</td><td>1024</td><td>1.28</td><td>4.17</td><td>175.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RQ-Trans. [28]</td><td>3.8B</td><td>RQ</td><td>2D</td><td>66M</td><td>256</td><td>3.20</td><td>-</td><td>-</td><td>-</td><td>-</td><td>3.80</td><td>323.7</td><td>-</td><td>-</td></tr><tr><td>MaskGIT [3]</td><td>227M</td><td>VQ</td><td>2D</td><td>66M</td><td>256</td><td>2.28</td><td>6.18</td><td>182.1</td><td>0.80</td><td>0.51</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LlamaGen-3B [45]</td><td>3.1B</td><td>VQ</td><td>2D</td><td>72M</td><td>576</td><td>2.19</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.18</td><td>263.3</td><td>0.80</td><td>0.58</td></tr><tr><td>WeTok [69]</td><td>1.5B</td><td>VQ</td><td>2D</td><td>400M</td><td>256</td><td>0.60</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.31</td><td>276.6</td><td>0.84</td><td>0.55</td></tr><tr><td>VAR [46]</td><td>2B</td><td>MSRQ</td><td>2D</td><td>109M</td><td>680</td><td>0.90</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.92</td><td>323.1</td><td>0.75</td><td>0.63</td></tr><tr><td>MaskBit [51]</td><td>305M</td><td>LFQ</td><td>2D</td><td>54M</td><td>256</td><td>1.61</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.52</td><td>328.6</td><td>-</td><td>-</td></tr><tr><td>MAR-H [34]</td><td>943M</td><td>KL</td><td>2D</td><td>66M</td><td>256</td><td>1.22</td><td>2.35</td><td>227.8</td><td>0.79</td><td>0.62</td><td>1.55</td><td>303.7</td><td>0.81</td><td>0.62</td></tr><tr><td>l-DeTok [56]</td><td>479M</td><td>KL</td><td>2D</td><td>172M</td><td>256</td><td>0.62</td><td>1.86</td><td>238.6</td><td>0.82</td><td>0.61</td><td>1.35</td><td>304.1</td><td>0.81</td><td>0.62</td></tr><tr><td>TiTok-S-128 [60]</td><td>287M</td><td>VQ</td><td>1D</td><td>72M</td><td>128</td><td>1.61</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.97</td><td>281.8</td><td>-</td><td>-</td></tr><tr><td>GigaTok [53]</td><td>111M</td><td>VQ</td><td>1D</td><td>622M</td><td>256</td><td>0.51</td><td>-</td><td>-</td><td>-</td><td>-</td><td>3.15</td><td>224.3</td><td>0.82</td><td>0.55</td></tr><tr><td>ImageFolder [35]</td><td>362M</td><td>MSRQ</td><td>1D</td><td>176M</td><td>286</td><td>0.80</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.60</td><td>295.0</td><td>0.75</td><td>0.63</td></tr><tr><td colspan="15">Diffusion-based</td></tr><tr><td>LDM-4 [41]</td><td>400M</td><td></td><td>2D</td><td></td><td></td><td></td><td>10.56</td><td>103.5</td><td>0.71</td><td>0.62</td><td>3.60</td><td>247.7</td><td>0.87</td><td>0.48</td></tr><tr><td>U-ViT-H/2 [1]</td><td>501M</td><td>KL</td><td>2D</td><td>55M</td><td>4096</td><td>0.27</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.29</td><td>263.9</td><td>0.82</td><td>0.57</td></tr><tr><td>MDTv2-XL/2 [15]</td><td>676M</td><td></td><td>2D</td><td></td><td></td><td></td><td>5.06</td><td>155.6</td><td>0.72</td><td>0.66</td><td>1.58</td><td>314.7</td><td>0.79</td><td>0.65</td></tr><tr><td>DiT-XL/2 [38]</td><td>675M</td><td></td><td>2D</td><td></td><td></td><td></td><td>9.62</td><td>121.5</td><td>0.67</td><td>0.67</td><td>2.27</td><td>278.2</td><td>0.79</td><td>0.65</td></tr><tr><td>SiT-XL/2 [36]</td><td></td><td>KL</td><td>2D</td><td>84M</td><td>1024</td><td>0.62</td><td>8.30</td><td>131.7</td><td>0.68</td><td>0.67</td><td>2.06</td><td>270.3</td><td>0.83</td><td>0.53</td></tr><tr><td>+REPA [61]</td><td></td><td></td><td>2D</td><td></td><td></td><td></td><td>5.90</td><td>157.8</td><td>0.70</td><td>0.69</td><td>1.42</td><td>305.7</td><td>0.82</td><td>0.59</td></tr><tr><td>LightningDiT [57]</td><td>675M</td><td>KL</td><td>2D</td><td>70M</td><td>256</td><td>0.28</td><td>2.17</td><td>205.6</td><td>-</td><td>-</td><td>1.35</td><td>295.3</td><td>-</td><td>-</td></tr><tr><td>TexTok-256 [62]</td><td>675M</td><td>KL</td><td>1D</td><td>176M</td><td>256</td><td>0.73</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.46</td><td>303.1</td><td>0.79</td><td>0.64</td></tr><tr><td>MAETok [4]</td><td>675M</td><td>AE</td><td>1D</td><td>176M</td><td>128</td><td>0.48</td><td>2.31</td><td>216.5</td><td>0.78</td><td>0.62</td><td>1.67</td><td>311.2</td><td>0.81</td><td>0.63</td></tr><tr><td>SoftVQ-VAE [5]</td><td>675M</td><td>SoftVQ</td><td>1D</td><td>176M</td><td>64</td><td>0.88</td><td>5.98</td><td>138.0</td><td>0.74</td><td>0.64</td><td>1.78</td><td>279.0</td><td>0.80</td><td>0.63</td></tr><tr><td colspan="15">Ours</td></tr><tr><td>MacTok+LightningDiT</td><td>675M</td><td></td><td></td><td></td><td>64</td><td>0.75</td><td>4.15</td><td>167.8</td><td>0.75</td><td>0.65</td><td>1.68</td><td>307.3</td><td>0.77</td><td>0.66</td></tr><tr><td></td><td></td><td>KL</td><td>1D</td><td>176M</td><td>128</td><td>0.43</td><td>3.12</td><td>186.2</td><td>0.75</td><td>0.66</td><td>1.50</td><td>299.8</td><td>0.78</td><td>0.67</td></tr><tr><td>MacTok+SiT-XL</td><td>675M</td><td></td><td></td><td></td><td>64</td><td>0.75</td><td>3.77</td><td>181.6</td><td>0.77</td><td>0.63</td><td>1.58</td><td>310.4</td><td>0.78</td><td>0.66</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>128</td><td>0.43</td><td>2.82</td><td>189.2</td><td>0.77</td><td>0.64</td><td>1.44</td><td>302.5</td><td>0.79</td><td>0.66</td></tr></table>

Table 8. System-level comparison on ImageNet 512×512 conditional generation. We report both Precision and Recall under classifier-free guidance (CFG) and non-CFG settings. 

<table><tr><td rowspan="2">Method</td><td rowspan="2"># Params(G)</td><td rowspan="2">Tok. Model</td><td rowspan="2">Token Type</td><td rowspan="2"># Params(T)</td><td rowspan="2">#Tokens↓</td><td rowspan="2">Tok. rFID↓</td><td colspan="4">w/o CFG</td><td colspan="4">w/ CFG</td></tr><tr><td>gFID↓</td><td>IS↑</td><td>Prec↑</td><td>Recall↑</td><td>gFID↓</td><td>IS↑</td><td>Prec↑</td><td>Recall↑</td></tr><tr><td colspan="15">GAN</td></tr><tr><td>BigGAN [3]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>8.43</td><td>177.9</td><td>-</td><td>-</td></tr><tr><td>StyleGAN-XL [24]</td><td>168M</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.41</td><td>267.7</td><td>-</td><td>-</td></tr><tr><td colspan="15">Auto-regressive</td></tr><tr><td>MaskGIT [3]</td><td>227M</td><td>VQ</td><td>2D</td><td>66M</td><td>1024</td><td>1.97</td><td>7.32</td><td>156.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MAGVIT-v2 [59]</td><td>307M</td><td>LFQ</td><td>2D</td><td>116M</td><td>1024</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.91</td><td>324.3</td><td>-</td><td>-</td></tr><tr><td>MAR-H [34]</td><td>943M</td><td>KL</td><td>2D</td><td>66M</td><td>1024</td><td>-</td><td>2.74</td><td>205.2</td><td>0.69</td><td>0.59</td><td>1.73</td><td>279.9</td><td>0.77</td><td>0.61</td></tr><tr><td>TiTok-B-128 [60]</td><td>177M</td><td>VQ</td><td>1D</td><td>202M</td><td>128</td><td>1.52</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.13</td><td>261.2</td><td>-</td><td>-</td></tr><tr><td>TiTok-L-64 [60]</td><td>177M</td><td>VQ</td><td>1D</td><td>644M</td><td>64</td><td>1.77</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.74</td><td>221.1</td><td>-</td><td>-</td></tr><tr><td colspan="15">Diffusion-based</td></tr><tr><td>ADM [10]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>23.24</td><td>58.1</td><td>-</td><td>-</td><td>3.85</td><td>221.7</td><td>0.84</td><td>0.53</td></tr><tr><td>U-ViT-H/4 [1]</td><td>501M</td><td></td><td>2D</td><td></td><td></td><td></td><td>-</td><td>-</td><td>-</td><td>-</td><td>4.05</td><td>263.8</td><td>0.84</td><td>0.48</td></tr><tr><td>DiT-XL/2 [38]</td><td>675M</td><td></td><td>2D</td><td></td><td></td><td></td><td>9.62</td><td>121.5</td><td>-</td><td>-</td><td>3.04</td><td>240.8</td><td>0.84</td><td>0.54</td></tr><tr><td>SiT-XL/2 [36]</td><td>675M</td><td>KL</td><td>2D</td><td>84M</td><td>4096</td><td>0.62</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.62</td><td>252.2</td><td>0.84</td><td>0.57</td></tr><tr><td>DiT-XL [38]</td><td>675M</td><td></td><td>2D</td><td></td><td></td><td></td><td>9.56</td><td>-</td><td>-</td><td>-</td><td>2.84</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UViT-H [1]</td><td>501M</td><td></td><td>2D</td><td></td><td></td><td></td><td>9.83</td><td>-</td><td>-</td><td>-</td><td>2.53</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UViT-H</td><td>501M</td><td></td><td>2D</td><td></td><td></td><td></td><td>12.26</td><td>-</td><td>-</td><td>-</td><td>2.66</td><td>-</td><td>-</td><td>-</td></tr><tr><td>UViT-2B [1]</td><td>2B</td><td>AE</td><td>2D</td><td>323M</td><td>256</td><td>0.22</td><td>6.50</td><td>-</td><td>-</td><td>-</td><td>2.25</td><td>-</td><td>-</td><td>-</td></tr><tr><td>TexTok-128 [62]</td><td>675M</td><td>KL</td><td>1D</td><td>176M</td><td>128</td><td>0.97</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.80</td><td>305.4</td><td>0.81</td><td>0.63</td></tr><tr><td>MAETok [4]</td><td>675M</td><td>AE</td><td>1D</td><td>176M</td><td>128</td><td>0.62</td><td>2.79</td><td>204.3</td><td>0.81</td><td>0.62</td><td>1.69</td><td>304.2</td><td>0.82</td><td>0.62</td></tr><tr><td>SoftVQ-VAE [5]</td><td>675M</td><td>SoftVQ</td><td>1D</td><td>391M</td><td>64</td><td>0.71</td><td>7.96</td><td>133.9</td><td>0.73</td><td>0.63</td><td>2.21</td><td>290.5</td><td>0.85</td><td>0.59</td></tr><tr><td colspan="15">Ours</td></tr><tr><td rowspan="2">MacTok+SiT-XL</td><td rowspan="2">675M</td><td rowspan="2">KL</td><td rowspan="2">1D</td><td>391M</td><td>64</td><td>0.89</td><td>4.63</td><td>163.7</td><td>0.80</td><td>0.61</td><td>1.52</td><td>306.0</td><td>0.80</td><td>0.63</td></tr><tr><td>176M</td><td>128</td><td>0.79</td><td>5.12</td><td>156.3</td><td>0.79</td><td>0.61</td><td>1.52</td><td>316.0</td><td>0.80</td><td>0.63</td></tr></table>

Table 9. Generation performance over training of SiT-XL trained on MacTok with 64 and 128 tokens. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Training Iter.</td><td colspan="4">w/o CFG</td><td colspan="4">w/ CFG</td></tr><tr><td>FID</td><td>IS</td><td>Prec.</td><td>Recall</td><td>FID</td><td>IS</td><td>Prec.</td><td>Recall</td></tr><tr><td rowspan="5">MacTok-64</td><td>400K</td><td>7.60</td><td>121.4</td><td>0.72</td><td>0.63</td><td>2.15</td><td>268.2</td><td>0.77</td><td>0.63</td></tr><tr><td>1M</td><td>5.34</td><td>147.7</td><td>0.74</td><td>0.63</td><td>1.73</td><td>290.4</td><td>0.77</td><td>0.65</td></tr><tr><td>2M</td><td>4.58</td><td>159.9</td><td>0.75</td><td>0.63</td><td>1.60</td><td>303.0</td><td>0.78</td><td>0.65</td></tr><tr><td>3M</td><td>3.98</td><td>174.7</td><td>0.76</td><td>0.63</td><td>1.60</td><td>308.2</td><td>0.78</td><td>0.66</td></tr><tr><td>4M</td><td>3.77</td><td>181.6</td><td>0.77</td><td>0.63</td><td>1.58</td><td>310.4</td><td>0.78</td><td>0.66</td></tr><tr><td rowspan="5">MacTok-128</td><td>400K</td><td>6.45</td><td>127.2</td><td>0.73</td><td>0.63</td><td>1.97</td><td>253.2</td><td>0.77</td><td>0.64</td></tr><tr><td>1M</td><td>4.31</td><td>153.6</td><td>0.75</td><td>0.64</td><td>1.60</td><td>271.7</td><td>0.77</td><td>0.65</td></tr><tr><td>2M</td><td>3.69</td><td>168.5</td><td>0.75</td><td>0.65</td><td>1.48</td><td>287.0</td><td>0.78</td><td>0.66</td></tr><tr><td>3M</td><td>3.28</td><td>176.2</td><td>0.76</td><td>0.65</td><td>1.45</td><td>293.1</td><td>0.78</td><td>0.66</td></tr><tr><td>4M</td><td>2.82</td><td>189.2</td><td>0.77</td><td>0.64</td><td>1.44</td><td>302.5</td><td>0.79</td><td>0.66</td></tr></table>

![](images/0a1aecaffa48eb8e601a915c6515dd4693718e38b89513e5f704729fdd39a114.jpg)  
Figure 9. Reconstruction results of MacTok with 64 tokens.

![](images/729108398079f6263920a912ca73dd5c06e288d3aa3f693d091bcb507b160fa0.jpg)

<details>
<summary>natural_image</summary>

Collage of various marine and food items including animals, turtles, mushrooms, yurts, dishes, and food containers (no text or symbols visible)
</details>

Figure 10. Reconstruction results of MacTok with 128 tokens.

![](images/eee51a8d8ee396209dde35ed4cfcf725b7fb9a204cb7bb5c1a8e992a00098d59.jpg)

<details>
<summary>natural_image</summary>

Collage of 3D images showing various scenes including a girl in a white hat, a museum exhibit, a person relaxing, and a motorcycle on the right (no visible text or symbols)
</details>

Figure 11. Reconstruction results of collapsed KL-VAE.

![](images/e12210a5ee5c1272295a0b7e2dc5aa0df7ffd0de265fd5967bdffa9738813f88.jpg)

<details>
<summary>natural_image</summary>

Collage of underwater turtle and sea turtle images showing various types of life, swimming, and fishing (no text or symbols visible)
</details>

Figure 12. Uncurated 256×256 generation results of SiT-XL with MacTok 128 tokens. We use CFG with 4.0. Class label =“loggerhead turtle” (33).

![](images/8fb52d8fa75fda0acdc5109bcd726933927e54531f53e38f2570bcd11bdb595a.jpg)

<details>
<summary>natural_image</summary>

Collage of photos of colorful macromos and parrots in natural habitats, including head, wings, and perched heads (no text or symbols)
</details>

Figure 13. Uncurated 256×256 generation results of SiT-XL with MacTok 128 tokens. We use CFG with 4.0. Class label =“macaw” (88).

![](images/7b239d5b04e92f415e98292c931df412bfecb0df0872e91edeadea0b640b8648.jpg)

<details>
<summary>natural_image</summary>

Collage of various bird and perched birds in natural settings, including head, wings, and perched birds (no text or symbols visible)
</details>

Figure 14. Uncurated 256×256 generation results of SiT-XL with MacTok 128 tokens. We use CFG with 4.0. Class label =“Kakatoe galerita” (89).

![](images/3703755838ccf52a60a5c5b937216d16b13a8874799659a8cb417bd1d837ff20.jpg)

<details>
<summary>natural_image</summary>

Collage of multiple photos of Golden F------ dogs in various poses, expressions, and scenes (no text or symbols)
</details>

Figure 15. Uncurated 256×256 generation results of SiT-XL with MacTok 128 tokens. We use CFG with 4.0. Class label =“golden retriever” (207).

![](images/52391b24e87dd143d775fc2788952a2cdb9a48add9a17a747fdc385e8f1210c7.jpg)

<details>
<summary>natural_image</summary>

Grid of black-and-white animal portraits including a wolf, a wolf, and a wolf with visible facial features (no text or symbols)
</details>

Figure 16. Uncurated 256×256 generation results of SiT-XL with MacTok 128 tokens. We use CFG with 4.0. Class label =“Arctic wolf” (270).

![](images/b373ab7ed3a3d2ceccd115b8b2efae41501ff6ae27f5594c28151a71f49d5b66.jpg)

<details>
<summary>natural_image</summary>

Grid of 24 white Arctic fox photos including animals in natural settings, no text or symbols present.
</details>

Figure 17. Uncurated 256×256 generation results of SiT-XL with MacTok 128 tokens. We use CFG with 4.0. Class label =“Arctic fox” (279).

![](images/a53da738e213fcae2ce98790ea1c86b79b75b0feb778e3f558a307c5900aa0a9.jpg)

<details>
<summary>natural_image</summary>

Grid of photos of various otters and seals, including cats, otters, and seals in natural setting (no text or labels)
</details>

Figure 18. Uncurated 256×256 generation results of SiT-XL with MacTok 128 tokens. We use CFG with 4.0. Class label =“otter” (360).

![](images/87bb6a181e7cc669c759b533c8f4d17fea07fb30dd6fa0c7e9ea0d0f1ea27c06.jpg)

<details>
<summary>natural_image</summary>

Collage of multiple photos of giant pandas in various animals, including eating bamboo, eating bamboo stalks, and eating bamboo sheets (no text or symbols visible)
</details>

Figure 19. Uncurated 256×256 generation results of SiT-XL with MacTok 128 tokens. We use CFG with 4.0. Class label =“panda” (388).

![](images/c5d64c192301a36f78ae4cea77dd7283f4321a24effd55545420e01dc16a04d9.jpg)

<details>
<summary>natural_image</summary>

Collage of red fire trucks and outdoor scenes, including outdoor events and family gatherings (no visible text or symbols)
</details>

Figure 20. Uncurated 256×256 generation results of SiT-XL with MacTok 64 tokens. We use CFG with 4.0. Class label =“fire engine” (555).

![](images/25f0239e398ebcc9e34fc814aee4092e1395c61f966abaaf45c75b189ab803ba.jpg)

<details>
<summary>natural_image</summary>

Collage of space missions including the Space Shuttle, Long March, and several spacecraft with launch or landing designs (no visible text or symbols)
</details>

Figure 21. Uncurated 256×256 generation results of SiT-XL with MacTok 64 tokens. We use CFG with 4.0. Class label =“space shuttle” (812).

![](images/65564ac82aed6fb389215b11652d681977835f17a45ce6d2daad6e78729af065.jpg)

<details>
<summary>natural_image</summary>

Collage of various food and drink items including ice cream, ice cream scoops, and dessert dishes (no visible text or labels)
</details>

Figure 22. Uncurated 256×256 generation results of SiT-XL with MacTok 64 tokens. We use CFG with 4.0. Class label =“ice cream” (928).

![](images/de02c276496ddf9b758097c1087bef06c335bd5087e06ba16af822cc85075685.jpg)

<details>
<summary>natural_image</summary>

Collage of various fast food and snack items including a hamburger, cheese, burgers, fries, and instant noodles (no text or labels visible)
</details>

Figure 23. Uncurated 256×256 generation results of SiT-XL with MacTok 64 tokens. We use CFG with 4.0. Class label =“cheeseburger” (933).

![](images/ed4ce647f29fd4c43b1332cb4a06a5b763bdedb351a0008e6140ccffb3c8dfc0.jpg)

<details>
<summary>natural_image</summary>

Collage of underwater and offshore marine shark images showing various scales, fins, and scenes (no text or symbols)
</details>

Figure 24. Uncurated 256×256 generation results of LightningDiT-XL with MacTok 128 tokens. We use CFG with 3.0. Class label =“white shark” (2).

![](images/ddbba09f8bb5917569e25784f12927136621bbcad4086ebd785b2e9bfacb9b74.jpg)

<details>
<summary>natural_image</summary>

Collage of various types of crab and seafood including a large red crab, a plate of green onions, and various outdoor scenes with no visible text or labels.
</details>

Figure 25. Uncurated 256×256 generation results of LightningDiT-XL with MacTok 128 tokens. We use CFG with 3.0. Class label =“Dungeness crab” (118).

![](images/5c34202291f547598f5054b8317cf016500e913552e6359327a1ecd96c987d2d.jpg)

<details>
<summary>natural_image</summary>

Collage of various dog breeds and activities including pet care, outdoor events, and daily activities (no visible text or symbols)
</details>

Figure 26. Uncurated 256×256 generation results of LightningDiT-XL with MacTok 128 tokens. We use CFG with 3.0. Class label =“Chesapeake Bay retriever” (209).

![](images/7246095abd7ebb1ddaa0016d190452a5d9c0842e6e8e2cb51f0cee9b73b1343f.jpg)

<details>
<summary>natural_image</summary>

Collage of various food and drink items including rolled spreads, vegetables, and instant noodles (no visible text or labels)
</details>

Figure 27. Uncurated 256×256 generation results of LightningDiT-XL with MacTok 128 tokens. We use CFG with 3.0. Class label =“burrito” (965).

![](images/072f8d34ce30db424bb11d7c514e5ddaf6c923a69b6c3b74a91add794deb9b71.jpg)

<details>
<summary>natural_image</summary>

Collage of photos showing various geothermal sites including turrets, girders, and plumes under a blue sky with clouds (no text or symbols visible)
</details>

Figure 28. Uncurated 256×256 generation results of LightningDiT-XL with MacTok 64 tokens. We use CFG with 3.0. Class label $= ^ {  } \operatorname { g e y s e r } ^ { \prime \prime } ( 9 7 4 )$ .

![](images/6017a5ac23451f08f3f04cce7070ddedcc3cc50e39b8e95a3af8478802f50698.jpg)

<details>
<summary>natural_image</summary>

Collage of scenic mountain landscapes with natural peaks, valleys, and forested slopes (no text or symbols visible)
</details>

Figure 29. Uncurated 256×256 generation results of LightningDiT-XL with MacTok 64 tokens. We use CFG with 3.0. Class label =“valley” (979).

![](images/4bf6358c0c0446ce3b04a417175038e62a5ee463689f664b0c036ea1ef298173.jpg)  
Figure 30. Uncurated 512×512 generation results of SiT-XL with MacTok 128 tokens. We use CFG with 4.0. Class label =“castle” (483).

![](images/31d2644c63f50e4b91b6590b0fe5fc6374be80774c81d357d87249949f0a96e5.jpg)

<details>
<summary>natural_image</summary>

Collage of coastal and mountain landscapes featuring cliffs, sea, rocks, and natural settlements (no text or symbols)
</details>

Figure 31. Uncurated 512×512 generation results of SiT-XL with MacTok 128 tokens. We use CFG with 4.0. Class label =“cliff” (972).

![](images/b6cdd7a648e2f5292ca9bdfb4beb4d304d7f8fb944bac3b1a1ad8acda319899f.jpg)

<details>
<summary>natural_image</summary>

Underwater coral and reef scenes with various colorful corals, fish, and marine life (no text or symbols)
</details>

Figure 32. Uncurated 512×512 generation results of SiT-XL with MacTok 64 tokens. We use CFG with 4.0. Class label =“coral reef” (973).

![](images/ed621de5aca9db631c7843d8b4cba619f33627fa1ae145a02d9a27097e79ac00.jpg)  
Figure 33. Uncurated 512×512 generation results of SiT-XL with MacTok 64 tokens. We use CFG with 4.0. Class label =“volcano” (980).