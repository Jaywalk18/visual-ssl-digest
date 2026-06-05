# ChannelTok: Efficient Flexible-Length Vision Tokenization

Sukriti Paul Arpit Bansal Tom Goldstein University of Maryland, College Park {sukriti5, bansal01, tomg}@umd.edu

# Abstract

Leading flexible vision tokenizers achieve SOTA quality at an extreme cost, relying on parameter-heavy backbones and slow, multi-step generative decoders. We depart from this complex, spatial-token paradigm and introduce a simple, lightweight, and fast channel-wise flexible-length tokenizer. Our method treats each latent channel as a visual token, enabling a parameter-efficient CNN-Transformer hybrid backbone. Furthermore, employing a stochastic taildropping paradigm during training naturally forces channels to organize by semantic importance. This allows for flexible compression at inference by simply retaining the first k channels, and naturally enables variable-length autoregressive image generation. We validate our approach through extensive experiments on ImageNet, demonstrating consistent quality across diverse token budgets. The results establish a new quality-efficiency frontier: our model achieves state-of-the-art perceptual quality (rFID 2.92) while being 8.6× faster in decoding and 2.1× smaller (159M params) than the next-best alternative. Our work establishes channel-wise tokenization as a powerful and practical paradigm for efficient visual representation. Project page: https://channeltok.github.io

# 1. Introduction

Vision tokenization serves as a foundational component in discrete generative image models, compressing images from pixel space into compact, semantically rich representations in latent space. While early generative representations for images focused on continuous-valued autoencoders [9] and VAEs [13], the pursuit of more efficient and discrete representations led to vector quantization, first via explicit codebooks in VQ-VAE [24] and VQ-GAN [7], and more recently through lookup-free methods such as FSQ [16], LFQ [29], and BSQ [31].

Efficiency in vision tokenization spans multiple dimensions like compression ratio, reconstruction fidelity, semantic richness, encoding-decoding latency, and downstream task performance. Until recently, however, another critical dimension remained largely unexplored: flexible-length tok-

![](images/87eff21ed9233681b65603af8be3aeb19800d01fa6d3ea32a706f58fe867f887.jpg)  
Figure 1. Quality-efficiency comparison. Reconstruction fidelity (rFID), decoding throughput, and model size across recent tokenizers. Our method achieves state-of-the-art rFID while being the smallest and among the fastest decoders.

enization that adjusts representation length based on visual complexity. This need has become increasingly urgent in the era of large-scale vision models, where compute budgets constrain deployment and the inherent variability in visual data suggests that not all images require equal representational capacity.

In this work, we introduce a simple channel-wise approach for training a flexible-length tokenizer. Our approach is based off a lightweight VQGAN [7] based autoencoder where each channel slice forms a visual token. However unlike standard architectures, our latent features are ordered from low- to high-level. On each step, the encoder maps an image onto the latent space, a random number k of features is chosen, and the decoder must reconstruct the image using only the k lowest-level features. From this process, a coarse-to-fine hierarchy of features naturally emerges in which “low-level” features represent concepts and shapes, and “high-level” features represent details.

![](images/f07099d4f907c1def1588a4a6ce0510a816ffecfe992f15d9551bb096a6e0e8f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Original"] --> B["VQGAN Encoder"]
    B --> C["h'"]
    C --> D["Channel-wise masking"]
    D --> E["Inactive channels C - k"]
    E --> F["Active channels k"]
    F --> G["Stop gradient"]
    G --> H["Channel-wise BSQ (active channels)"]
    H --> I["c = 0"]
    H --> J["c = 1"]
    H --> K["c = 2"]
    H --> L["c = 3"]
    I --> M["Reconstructed VQGAN Decoder"]
    J --> M
    K --> M
    L --> M
```
</details>

Figure 2. Overview of our channel-wise flexible tokenizer. The encoder compresses the input image into a latent representation $\mathbf { z } \in \mathbb { R } ^ { C \times h \times w }$ . During training, we adaptively mask channels by retaining only the first k active channels (shown in teal) while stopping gradients through inactive channels (shown in gray). Each active channel is independently quantized using Binary Spherical Quantization (BSQ). The decoder reconstructs the image from the quantized active channels and the masked inactive channels. At inference, varying k enables flexible compression rates without retraining.

By leveraging the inherent structure along the channel dimension, our architecture avoids the complexity of parameter-heavy, multi-step generative decoders [1, 15]. Fig. 1 shows our method strikes a favorable balance across the quality-efficiency frontier: competitive reconstruction quality (rFID 2.92) while being the smallest (159M parameters) and among the fastest high-fidelity flexible tokenizers. This balance of efficiency and fidelity makes our method ideally suited for deployment scenarios where size, speed, and quality collectively matter, such as edge devices and real-time applications.

Our contributions include: (1) a channel-wise flexible tokenization mechanism that dynamically allocates latent channel capacity based on image complexity; (2) a lightweight VQGAN-based architecture optimized for parameter efficiency and high-throughput encoding–decoding; (3) practical training strategies and regularization for stable learning in the channel-dimension tokenization regime; and (4) a demonstration that channel-wise token ordering transfers to autoregressive generation, where a LlamaGen model trained on our tokens produces coherent images at variable token budgets, suggesting channel-wise tokenization as a promising direction for efficient AR generation.

# Background & Related Work

Recent work has begun to address flexible tokenization through various mechanisms. CAT [21] employs a visionlanguage model to classify images by complexity level and assigns them to one of three fixed compression ratios within a nested VAE architecture, though this relies on external LLM-based complexity prediction of an image’s textual description. Other approaches enforce nested token hierarchies through tail-token dropping [1, 17] and block-wise causal masking [27]. These Matryoshka-inspired [14] methods demonstrate a semantic hierarchy from coarse to fine representations through explicit token ordering. ALIT [5] instead uses recurrence to iteratively adjust token lengths over multiple roll-outs, rather than nesting multiple granularities within a single pass. As a further departure, KARL [6] predicts the appropriate token count in one forward pass guided by Kolmogorov Complexity, framing token length as a proxy for minimal description length. However, it relies on an upsidedown reinforcement learning training paradigm and depends heavily on multiple manual design choices for optimizing token budgets, with heightened reconstruction sensitivity to these manual selections.

While these methods expand the notion of “flexibility” to encompass reconstruction fidelity, downstream performance, and latent compressibility, industry-scale deployment demands additional considerations: parameter efficiency and inference speed. Many existing approaches rely on sophisticated, parameter-heavy architectures with large ViT backbones [1, 15, 23]. Some methods, though achieving compelling reconstruction fidelity, require multi-step decoding (FlexTok’s [1] rectified flow decoder, DOVE’s [15] transformer-based generative decoder), adding computational overhead at inference. Others limit compression to fixed token buckets [5] or train on hundreds of millions of samples [27]. Moreover, these approaches typically require extensive training recipes, including multi-stage training pipelines or prolonged training schedules, to maintain quality in low-token regimes.

Contemporary flexible tokenization research for 2D discrete representations primarily operates along the spatial dimension, typically treating local $h \times w$ patches as tokens. Recent fixed-length tokenization work [28, 32] has marked a conceptual shift by operating in the channel dimension, revealing that latent channels naturally encode a coarse-to-fine hierarchy that emerges without explicit ordering constraints. Unlike spatial approaches that enforce token ordering through architectural constraints and training objectives, channel-based representations exhibit this hierarchy organically. Motivated by this redefinition of “visual words,” we identify the channel dimension as the preferred axis for flexible tokenization, enabling a simpler and more parameter-efficient design.

# 2. Method

We introduce a channel-wise adaptive tokenizer that dynamically selects the number of latent channels to retain for each input. Our approach employs a VQGAN-inspired [7] autoencoder where each latent channel functions as a discrete visual token. An overview appears in Fig. 2.

# 2.1. Preliminaries

Let $\mathbf { x } \in \mathbb { R } ^ { H \times W \times 3 }$ denote an input image. Our tokenizer comprises an encoder $E _ { \theta }$ , quantizer $Q _ { \phi }$ , and decoder $D _ { \psi }$ The encoder compresses x into latent representation ${ \bf z } =$ $E _ { \theta } ( \mathbf { x } ) \in \mathbb { R } ^ { C \times h \times w }$ , where C is the channel dimension and $h ,$ w are spatial dimensions. Unlike spatial tokenization methods treating each spatial location as a token, we operate along the channel dimension where each channel slice $\mathbf { z } _ { c } \in$ $\mathbb { R } ^ { h \times w }$ constitutes a visual token. This design enables a natural coarse-to-fine hierarchy without explicit ordering constraints.

# 2.2. Encoder-Decoder Architecture

Encoder. Building on VQGAN [7], our encoder applies convolutional blocks with residual connections and downsampling layers. Multi-head self-attention blocks at resolutions {32, 16, 8} capture global dependencies. The encoder reduces spatial resolution by factor 16 (from $2 5 6 \times 2 5 6$ to $1 6 \times 1 6 )$ while expanding to $C = 5 1 2$ channels, yielding z ∈ R512×16×16. $\mathbf { z } \in \mathbb { R } ^ { 5 1 2 \times 1 6 \times 1 6 }$

Algorithm 1 Adaptive Channel Masking.   
Require: Latent $z \in R^{C \times h \times w}$ , mask probability $p_{mask}$ , range $[t_{\min}, t_{\max}]$ Ensure: Masked latent $z_{input}$ , mask M, active channels k

1: Sample apply_mask ~ Bernoulli( $p_{mask}$ )

2: if apply_mask = 1 then

3: Sample $t \sim \mathcal{U}(t_{\min}, t_{\max})$ 4: $k \leftarrow \max(1, \min(\lfloor t \cdot C \rfloor, C))$ 5: Initialize $M \in \{0, 1\}^{C \times h \times w}$ with zeros

6: for c = 1 to k do

7: $M_c \leftarrow 1_{h \times w}$ // Set first k channels to 1

8: end for

9: $z_{active} \leftarrow M \odot z$ 10: $z_{inactive} \leftarrow (1 - M) \odot \text{sg}(z)$ // Stop-grad on inactive

11: $z_{input} \leftarrow z_{active} + z_{inactive}$ 12: else

13: $M \leftarrow 1_{C \times h \times w}$ // Use all channels

14: $k \leftarrow C$ 15: $z_{input} \leftarrow z$ 16: end if

17: return $z_{input}$ , M, k

Channel as token. Each of the C channels encodes distinct visual information at varying abstraction levels. We treat the channel dimension as the token dimension, where $\mathbf { z } _ { c }$ for $c \in \{ 1 , \ldots , C \}$ represents the c-th visual token. This formulation differs fundamentally from spatial tokenization: rather than encoding local patches, each channel captures global patterns across the entire spatial extent, with earlier channels encoding coarse structure and later channels refining details.

Decoder. The decoder mirrors the encoder with upsampling blocks and multi-head self-attention at resolutions {8, 16, 32}. It reconstructs the image $\hat { \mathbf { x } } = D _ { \psi } ( \mathbf { q } )$ from quantized representation q with inactive channels set to zero.

# 2.3. Adaptive Channel Masking

To enable flexible tokenization, we introduce a stochastic prefix channel masking mechanism during training. This encourages the model to learn an ordered representation where information importance decreases along the channel dimension, following strategies from FlexTok [1], OneDPiece [17], and ElasticTok [27].

Mask generation and application. For each training sample, we uniformly sample mask retention ratio $t \sim$ $\mathcal { U } ( t _ { \operatorname* { m i n } } , t _ { \operatorname* { m a x } } )$ with $t _ { \mathrm { m i n } } ~ = ~ 0 . 0 0 2$ and $t _ { \mathrm { m a x } } ~ = ~ 1 . 0 .$ . The number of active channels is $k = \lfloor t \cdot C \rfloor$ , clamped to [1, C ].

Table 1. Flexible tokenization results on ImageNet-1K validation set. We report learnable parameters (M = millions), reconstruction metrics (rFID ↓, LPIPS ↓, DreamSim ↓), and system efficiency (encoding/decoding throughput in images/s and latency in ms/image). All methods are evaluated at a 256-token budget; we additionally report a 512-token variant of our method. Best results are shown in bold, second-best are underlined. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Parameters (M)</td><td colspan="3">Reconstruction Quality</td><td colspan="4">System Efficiency</td></tr><tr><td>Encoder</td><td>Decoder</td><td>Total</td><td>rFID ↓</td><td>LPIPS ↓</td><td>DreamSim ↓</td><td>Enc. Tput ↑</td><td>Dec. Tput ↑</td><td>Enc. Lat. ↓</td><td>Dec. Lat. ↓</td></tr><tr><td>OneDPiece [17]</td><td>304</td><td>307</td><td>642</td><td>7.61</td><td>0.180</td><td>0.109</td><td>32.84</td><td>24.77</td><td>30.5</td><td>40.4</td></tr><tr><td>DOVE [15]</td><td>70</td><td>70</td><td>287</td><td>13.82</td><td>0.153</td><td>0.133</td><td>118.70</td><td>68.83</td><td>8.4</td><td>14.5</td></tr><tr><td>ALIT [5]</td><td>202</td><td>206</td><td>431</td><td>9.74</td><td>0.180</td><td>0.11</td><td>18.51</td><td>18.34</td><td>54.01</td><td>54.54</td></tr><tr><td>KARL [6]</td><td>101</td><td>105</td><td>239</td><td>5.74</td><td>0.154</td><td>0.116</td><td>63.54</td><td>54.66</td><td>15.7</td><td>18.3</td></tr><tr><td>FlexTok [1]</td><td>85</td><td>172</td><td>341</td><td>2.97</td><td>0.228</td><td>0.154</td><td>61.33</td><td>4.78</td><td>16.3</td><td>209.3</td></tr><tr><td>Ours (256)</td><td>65</td><td>95</td><td>159</td><td>3.70</td><td>0.169</td><td>0.111</td><td>51.34</td><td>41.39</td><td>19.5</td><td>24.2</td></tr><tr><td>Ours (512)</td><td>65</td><td>95</td><td>159</td><td>2.92</td><td>0.153</td><td>0.096</td><td>51.40</td><td>41.30</td><td>19.5</td><td>24.2</td></tr></table>

![](images/88d86dcd344086e985769a60414291247fdd43e98554e698a2a87bbcd2fc4ce9.jpg)

<details>
<summary>line</summary>

| Token Budget | OneDPiece | DOVE | ALIT | KARL | FlexTok | Ours |
| ------------ | --------- | ---- | ---- | ---- | ------- | ---- |
| 32           | 11        | 21   | 26   | 37   | 4       | 26   |
| 64           | 10        | 18   | 19   | 28   | 3       | 13   |
| 128          | 9         | 16   | 13   | 14   | 3       | 6    |
| 256          | 8         | 15   | 10   | 6    | 3       | 4    |
| 512          | 7         | 14   | 9    | 4    | 3       | 3    |
</details>

(a) rFID vs. Token Budget.

![](images/aa10fc965d137dfe5d991722e9f5706b54650174ac67945ffc789607288e4f95.jpg)

<details>
<summary>line</summary>

| Token Budget | OneDPiece | DOVE | ALIT | KARL | FlexTok | Ours |
| ------------ | --------- | ---- | ---- | ---- | ------- | ---- |
| 32           | 0.125     | 0.065 | 0.115 | 0.125 | 0.14    | 0.11 |
| 64           | 0.11      | 0.06  | 0.105 | 0.11  | 0.125   | 0.095 |
| 128          | 0.09      | 0.06  | 0.095 | 0.08  | 0.095   | 0.08 |
| 256          | 0.075     | 0.06  | 0.075 | 0.06  | 0.075   | 0.07 |
| 512          | 0.07      | 0.06  | 0.07  | 0.06  | 0.07    | 0.065 |
</details>

(b) L1 vs. Token Budget.

![](images/4aca4e7dda7db34131599f944e92157e3f3b38377f94bbff26cc4543d079d937.jpg)

<details>
<summary>line</summary>

| Token Budget | OneDPiece | DOVE  | ALIT  | KARL  | FlexTok | Ours  |
| ------------ | --------- | ----- | ----- | ----- | ------- | ----- |
| 32           | 0.18      | 0.18  | 0.30  | 0.42  | 0.27    | 0.35  |
| 64           | 0.15      | 0.15  | 0.22  | 0.33  | 0.22    | 0.23  |
| 128          | 0.14      | 0.14  | 0.14  | 0.21  | 0.18    | 0.14  |
| 256          | 0.12      | 0.12  | 0.10  | 0.12  | 0.16    | 0.12  |
| 512          | 0.10      | 0.10  | 0.10  | 0.10  | 0.10    | 0.10  |
</details>

(c) DreamSim vs. Token Budget.   
Figure 3. Performance across token budgets. Our method demonstrates consistent quality improvement across reconstruction metrics across token budgets while being computationally efficient.

We construct binary mask $\mathbf { M } \in \{ 0 , 1 \} ^ { C \times h \times w }$ as:

$$
\mathbf {M} _ {c} = \left\{ \begin{array}{l l} 1 & \text { if } c \leq k \\ 0 & \text { otherwise } \end{array} \right. \tag {1}
$$

The mask will be used to stochastically drop the tail of the feature tensor, promoting hierarchical organization where critical information concentrates in early channels, and fine details in later channels.

During training, we randomly choose between applying masking or using the full latent $\mathbf { z } ,$ each with equal probability. For masked channels $( c > k )$ , we apply stop-gradient to prevent encoding information into inactive channels during the backward pass:

$$
\mathbf {z} _ {\text { input }} = \mathbf {M} \odot \mathbf {z} + (1 - \mathbf {M}) \odot \operatorname{sg} (\mathbf {z}), \tag {2}
$$

where sg(·) denotes stop-gradient. This ensures gradients flow only through active channels, encouraging the encoder to prioritize information in early channels that are more likely to be retained. Algorithm 1 formalizes the procedure.

Inference flexibility. At inference, we control compression rate by specifying active channel count $k \in [ 1 , C ]$ . Given target k, we construct mask M deterministically with first k channels active, encode the image to obtain z, and apply the mask. We quantize only the active channels via BSQ, while inactive channels are set to zero. The decoder reconstructs from both the quantized active channels and the zero-masked inactive channels, enabling continuous control over the rate-distortion tradeoff without retraining. The emergent coarse-to-fine structure allows progressive decoding: reconstructions with k = 32 capture global structure, while increasing k progressively refines local details.

# 2.4. Binary Spherical Quantization

We incorporate Binary Spherical Quantization (BSQ) [31] to discretize each active channel independently. Following the analysis in WeTok [32], BSQ provides a lookup-free, parameter-efficient alternative to codebook-based methods that is particularly well-suited for channel-wise tokenization. BSQ projects latent features onto a unit hypersphere and applies binary quantization. We use straight-through estimation [2] for gradients and optimize entropy-based and commitment losses following [31], with $\lambda _ { \mathrm { e n t } } ~ = ~ 0 . 1$ and $\lambda _ { \mathrm { c o m m i t } } = 0 . 2 5$ in our quantization loss

$$
\mathcal {L} _ {\text { quant }} = \lambda_ {\text { ent }} \mathcal {L} _ {\text { ent }} + \lambda_ {\text { commit }} \mathcal {L} _ {\text { commit }}.
$$

# 2.5. Training Objective

Reconstruction and perceptual losses. We supervise the decoder with an $\ell _ { 1 }$ reconstruction loss $\mathcal { L } _ { \mathrm { r e c } } = \| \mathbf { x } - \hat { \mathbf { x } } \| _ { 1 }$ and

![](images/ecf5cf20987eec5a4cc383e5c574b7c7c8678f4ea874ddb15b06f4aab4eb2597.jpg)

<details>
<summary>text_image</summary>

Original
32 tokens
Ours
ALIT
DOVE
FlexTok
KARL
One-D-Piece
64 tokens
128 tokens
256 tokens
</details>

Figure 4. Qualitative comparison across token budgets. We show reconstructions from our model and prior flexible tokenizers at 32–256 tokens, along with the original image. Differences in sharpness, color consistency, and structural preservation can be observed as the token budget increases. Additional results are provided in the Appendix.

a VGG-based perceptual loss

$$
\mathcal {L} _ {\text { perc }} = \sum_ {l} \| \Phi_ {l} (\mathbf {x}) - \Phi_ {l} (\hat {\mathbf {x}}) \| _ {2} ^ {2}, \tag {3}
$$

where l indexes feature layers of the pretrained network Φ. The base objective combines these with the quantization loss:

$$
\mathcal {L} _ {\text { base }} = \lambda_ {\text { rec }} \mathcal {L} _ {\text { rec }} + \lambda_ {\text { perc }} \mathcal {L} _ {\text { perc }} + \mathcal {L} _ {\text { quant }}, \tag {4}
$$

with fixed weights $\lambda _ { \mathrm { r e c } } = 0 . 2 5 \mathrm { a n d } \lambda _ { \mathrm { p e r c } } = 1 . 0 .$

Adversarial training. To improve perceptual fidelity, we employ a PatchGAN discriminator [10] with the standard hinge loss [18]. We train for 15 epochs using only $\mathcal { L } _ { \mathrm { b a s e } }$ as a warm-up, after which we add the adversarial term and optimize

$$
\mathcal {L} = \mathcal {L} _ {\text { base }} + \lambda_ {\text { gan }} \mathcal {L} _ {\text { hinge }}, \quad \lambda_ {\text { gan }} = 0. 1.
$$

This schedule avoids early adversarial instability while boosting perceptual quality in later stages.

# 2.6. Training Considerations

Flexible-length channel-wise tokenization remains largely unexplored compared to spatial tokenization, with limited established training recipes in the literature. Through experimentation, we identify choices that impact training stability and reconstruction fidelity. We adopt $C = 5 1 2$ latent channels to balance capacity with stability. We employ $\ell _ { 1 }$ reconstruction loss with perceptual loss weighting $( \lambda _ { \mathrm { p e r c } } = 1 . 0 $ , $\lambda _ { \mathrm { r e c } } = 0 . 2 5 )$ , where higher perceptual weight enhances reconstruction fidelity; this reversal of traditional weighting aligns with recent findings in visual tokenization that prioritize perceptual quality [1, 29]. We also found $\ell _ { 2 }$ loss, Charbonnier loss, and gram matrix losses [26] produced noticeable blurring in our channel-wise setting, while perceptual weights below 0.1 caused training instabilities.

We introduce adversarial training to mitigate BSQ quantization artifacts, particularly checkerboard patterns and gridlike distortions that emerge from binary quantization. However, we delay GAN loss introduction until epoch 15, as applying it from initialization led to mode collapse in preliminary experiments. This approach allows the reconstruction objective to establish a stable baseline before adversarial refinement. Our stochastic masking $( p _ { \mathrm { m a s k } } = 0 . 5 )$ was more stable than deterministic warmup schemes that maintain full channels for initial epochs before transitioning to adaptive masking. Cosine decay with linear warmup (5K steps to peak LR $1 0 ^ { - 3 }$ , decaying to $5 \times 1 0 ^ { - 5 } )$ becomes crucial when combining reconstruction and adversarial objectives [3, 12]. Finally, we adopt BSQ with an implicit codebook size of $2 ^ { 1 6 }$ over codebook-based vector quantization since our aim was to first establish a training recipe for flexible channel-wise tokenization without additional complexity from codebook commitment and collapse issues [29]. BSQ eliminates lookup operations and codebook memory storage, with recent work showing its relevance in channelwise regimes [32].

![](images/2a07b0e86140ec54550793fa3fd98f9c755fb734ca2a1464885c2f1b6d74295b.jpg)

<details>
<summary>text_image</summary>

Original
t=16
t=32
t=48
t=64
t=96
t=128
t=256
t=512
Flexible
Base
Flexible
Base
</details>

Figure 5. Comparison of reconstructions with and without our channel-wise flexible masking module. The baseline model is identical to the flexible model but trained without masking. For the baseline: semantic coherence appears only at high token budgets, indicating a lack of emergent semantic hierarchy. Our flexible tokenizer clearly shows a semantic coarse-to-fine progression, with meaningful reconstructions even at low and moderate token budgets.

System Specifications. For training, we use 8× NVIDIA H100 80GB GPUs over a period of 48 hours for training over 150 epochs on ImageNet-1K [20]. For our inference, we use 1x NVIDIA A5000 GPU.

# 3. Experiments

# 3.1. Flexible Length Tokenization

Evaluation protocol. We evaluate reconstruction quality and system efficiency on ImageNet-1K [20] validation set (50k images) with standard transforms. We measure reconstruction fidelity and perceptual quality via PSNR, rFID [11], L1 distance, LPIPS [30], SSIM [25], and DreamSim [8]. For system performance, we report encoding/decoding throughput (images/s) and latency (ms/image) on a single A5000 GPU. To ensure a fair and rigorous comparison, all methods were benchmarked with fp16 precision (Note: DOVE required bf16 as it was incompatible with fp16) and a fixed batch size of 128. We isolate model performance to a forward pass by excluding data loading, GPU-CPU transfers, and disk I/O from timing measurements. All baselines were benchmarked under these identical hardware and protocol settings.

![](images/8dc8a870dbf6052988d7ba3511819fdb8f725688008b73ae6663f6f7d7903875.jpg)

<details>
<summary>text_image</summary>

Original
t=8
t=32
t=64
t=128
t=256
t=512
</details>

Figure 6. Semantic organization in early channels. Channel swapping experiments demonstrate hierarchical semantic encoding. Each pair of rows shows two images whose first t channels are progressively swapped. When channels from one image are replaced with another, the image progressively transforms from the source to the target.

Performance at 256 tokens. Tab. 1 demonstrates that our method establishes a new Pareto frontier for flexible tokenization, delivering competitive perceptual quality while being significantly smaller and faster than existing methods. We first evaluate at a 256-token budget to enable fair comparison with key baselines [1, 5, 17] that support this maximum token count. At 256 tokens, our model achieves an rFID of 3.70, closely matching FlexTok’s 2.97. However, our decoder is 8.6× faster (41.4 vs 4.8 images/s) and our total model is 2.1× smaller (159M vs 341M params). This efficiency gain stems from our lightweight architecture that avoids the costly multi-step generative decoder used in FlexTok. In contrast, while DOVE achieves high throughput, its perceptual quality (rFID 13.82) is insufficient for highfidelity reconstruction tasks. A key advantage of our channelwise design is the ability to scale capacity without retraining. At our model’s full 512-channel capacity (row Ours (512)), we achieve state-of-the-art perceptual quality with an rFID of 2.92, surpassing FlexTok’s 2.97. Our method is the only flexible tokenizer that simultaneously achieves best-in-class perceptual quality (rFID 2.92, DreamSim 0.096) and efficiency (smallest model at 159M params, among the fastest decoders), making it ideal for latency-sensitive applications.

![](images/b7db5cbdafd07f975e3e15c5c1fb0b4062882ade5f3ece89a9fcda8d09243828.jpg)

<details>
<summary>text_image</summary>

t=32
t=64
t=128
t=256
t=32
t=64
t=128
t=256
t=32
t=64
t=128
t=256
</details>

Figure 7. Autoregressive image generation across token budgets. Images are generated by sampling from the LlamaGen [22] GPT-L transformer trained on discrete channel tokens. Generation begins from a randomly sampled first token and proceeds autoregressively, with remaining channels zero-filled at truncation. Even at 32 tokens (7.9× speedup), generated samples show coherent global structure, with quality improving progressively as more channels are generated.

Performance across token budgets. Fig. 3 shows reconstruction quality across varying token budgets. Our method demonstrates consistent quality improvement across all metrics, with rFID decreasing from 12.60 at 64 tokens to 2.92 at 512 tokens. At the highest budget, we achieve the best rFID (2.92) and DreamSim (0.096) across all methods, indicating superior semantic preservation and perceptual alignment. FlexTok maintains consistently low rFID (2.83-3.81) across budgets due to its rectified flow decoder [1], which provides perceptual coherence largely independent of token count. KARL and ALIT exhibit steeper rFID degradation at lower budgets but achieve strong pixel-level reconstruction at higher token counts, with KARL reaching competitive L1 distance (0.060) at 256 tokens. The DreamSim metric further validates our approach: scores improve from 0.347 at 32 tokens to 0.096 at 512 tokens, demonstrating that additional channels enable richer feature representations that better capture perceptually-salient content. This trend suggests our channel-wise design naturally scales with capacity, allocating tokens to maximally preserve semantic information. A key advantage of our approach is seamless scaling from 64 to 512 tokens without architectural modifications, enabling flexible quality-efficiency operating points. Visual comparisons across methods and token budgets are provided in Fig. 4.

# 4. Induced Semantic Hierarchy

# 4.1. Prefix Masking Enforces Ordering

Fig. 5 illustrates the effect of our adaptive masking on the ordering of representations. Our baseline model without tail-dropping produces unstructured channel representations where the progression of tokens lacks sufficient semantic content until the last few tokens. Our tail-dropping strategy forces a causal structure along the channel dimension. Therefore, early channels must encode information independently of later channels, as tail channels may be masked during training. This causal constraint forces our method to concentrate important information in early channels through reconstruction from variable-length prefixes, with later channels adding progressive refinement. Additional training details on our baseline model and qualitative examples are in the supplementary material.

![](images/4b5e2169284e298238976345f45e91854ab97be9707cdcc4b1bb95b9f5644559.jpg)

<details>
<summary>line</summary>

| Token Budget | Mask 0.2 | Mask 0.5 | Mask 0.8 |
| ------------ | -------- | -------- | -------- |
| 32           | 49       | 34       | 36       |
| 64           | 36       | 22       | 26       |
| 128          | 29       | 15       | 21       |
| 256          | 26       | 13       | 19       |
</details>

![](images/4e736d1ca9d25146c2b8977805bbb14ec492f59b1e481ac772985fb3238b9ad9.jpg)

<details>
<summary>line</summary>

| Token Budget | Bias High | Bias Low | Bias Normal |
| ------------ | --------- | -------- | ----------- |
| 32           | 41.0      | 28.0     | 34.0        |
| 64           | 25.0      | 20.0     | 22.0        |
| 128          | 17.0      | 15.0     | 15.0        |
| 256          | 13.0      | 12.0     | 12.0        |
</details>

![](images/8ba1adfc2a57e585276607a872585faf91b988d19cbaac64b7d7104be1854af9.jpg)

<details>
<summary>line</summary>

| Token Budget | Tiny (6.3M) | Small (18.16M) | Medium (31.73M) | Large (72.39M) |
| ------------ | ----------- | -------------- | --------------- | -------------- |
| 32           | 57          | 44             | 42              | 34             |
| 64           | 44          | 30             | 28              | 22             |
| 128          | 36          | 22             | 18              | 15             |
| 256          | 34          | 18             | 14              | 12             |
</details>

Figure 8. Architectural ablations. (a) Effect of masking probability $p _ { \mathrm { m a s k } } .$ . (b) Effect of sampling bias on retention ratio t. (c) Effect of model scale. rFID consistently improves with more balanced masking, uniform sampling, and larger model capacity.

# 4.2. Semantic Transferability

We investigate whether the global semantics encoded in early channels transfer between pairs of images. We select visually dissimilar image pairs and swap their first t channels, as shown in Fig. 6. For small swaps $( t < 3 2 )$ , we observe transfer of stylistic elements such as background characteristics, color temperature, and lighting while preserving the primary subject. However, as swap depth increases $( t \geq 6 4 )$ , images undergo more noticeable semantic transformations with foreground subjects beginning to morph. This behavior mirrors the hierarchical refinement in progressive image coding, where early layers establish global structure before fine details emerge.

# 5. Autoregressive Image Generation

A key advantage of channel-wise tokenization is its natural compatibility with autoregressive (AR) generation. In spatial tokenization, each token encodes a local patch, and the generation order is an arbitrary raster scan, where truncating the sequence yields an incomplete spatial grid rather than a usable image. In contrast, our channel-wise tokens are ordered by semantic importance: each successive token refines the entire image, progressing from coarse structure to fine detail. This ordering maps directly onto the autoregressive factorization, making variable-length generation not merely possible but semantically meaningful.

We validate this by first training a tokenizer with a reduced implicit codebook vocabulary $( 2 ^ { 1 4 } )$ and a maximum dimension of 256, for the smaller ImageNet-100 subset. We then train a LlamaGen [22] GPT-L transformer on the discrete token sequences extracted by our tokenizer. The model is trained with standard next-token prediction using crossentropy loss. To reflect the coarse-to-fine hierarchy, we introduce a position-weighted loss that emphasizes early channels:

$$
\mathcal {L} _ {\mathrm{AR}} = \sum_ {c = 0} ^ {C - 1} w _ {c} \cdot \mathcal {L} _ {\mathrm{CE}} (c), \quad w _ {c} = 1 + \alpha \left(1 - \frac {c}{C - 1}\right), \tag {5}
$$

where $\mathcal { L } _ { \mathrm { C E } } ( c )$ is the cross-entropy at channel position c and α controls the weighting strength. With $\alpha = 1$ , weights decrease linearly from $w _ { 0 } = 2 . 0$ for the first channel to $w _ { C - 1 } = 1 . 0$ for the last, biasing the model toward accurately predicting the structurally critical early channels while still learning fine-grained later channels.

Flexible generation budget. At inference, generation can be terminated after any number of tokens $k \leq C ,$ , with remaining channels zero-filled before decoding. Tab. 2 reports FID (calculated with respect to the tokenizer outputs) and generation throughput across token budgets on a single H100 GPU. At 64 tokens, the model achieves FID 9.75 with a 4.1× speedup over full 256-token generation (3.77s → 0.91s per image). The marginal gain from 128 to 256 tokens is negligible (FID $7 . 9 6  7 . 8 5 )$ , consistent with the coarse-to-fine hierarchy: early channels carry the dominant semantic content, and later channels contribute diminishing perceptual improvement. Even at 32 tokens (7.9× speedup), generated samples exhibit coherent global structure (Fig. 7), confirming that the channel ordering learned by the tokenizer transfers effectively to the generative setting.

This quality-speed tradeoff is architecturally infeasible with spatial tokenizers, where partial sequences produce fragmented images. Channel-wise generation uniquely enables a single trained model to serve diverse latency budgets without retraining or architectural modification.

# 6. Ablations

# 6.1. Effects of Sampling and Masking

As described in Sec. 2.3, our architecture introduces two forms of stochasticity: a global masking probability $p _ { \mathrm { m a s k } }$ and a per-sample retention ratio t. We sweep $p _ { \mathrm { m a s k } }$ on ImageNet-100 and measure rFID across token budgets. Low masking $( p _ { \mathrm { m a s k } } = 0 . 2 )$ performs worst as the model overfits to full-context reconstructions, while high masking $( p _ { \mathrm { m a s k } } = 0 . 8 )$ improves robustness but slightly hurts fidelity. We find $p _ { \mathrm { m a s k } } = 0 . 5$ optimal, providing sufficient masking pressure to induce semantic structure without destabilizing reconstructions. We then vary the retention distribution t at fixed $p _ { \mathrm { m a s k } } = 0 . 5 ( \mathrm { F i g . \ 8 b } )$ . Biasing t towards lower or higher values trades off performance between token regimes. Uniform sampling provides the most balanced training signal across all budgets.

Table 2. Token budget vs. speed/quality (tokenizer reference). 

<table><tr><td>Budget</td><td>Speedup</td><td>Time/Im.</td><td>gFID $^{\dagger}$ </td></tr><tr><td>32</td><td>7.93×</td><td>0.475s</td><td>20.71</td></tr><tr><td>64</td><td>4.13×</td><td>0.92s</td><td>9.75</td></tr><tr><td>128</td><td>2.06×</td><td>1.82s</td><td>7.96</td></tr><tr><td>256</td><td>1.0×</td><td>3.78s</td><td>7.85</td></tr></table>

†FID w.r.t. tokenizer reconstruction.

# 6.2. Scaling Parameters

We train four models by varying the parameters of both the encoder and decoder. The trained models clearly exhibit a scaling curve, with larger models consistently improving rFID scores across token budgets.

# 6.3. Effect of Quantizers

To study the effect of different quantizers we also train our tokenizer with FSQ on ImageNet-100 [16] and compare it with BSQ in Tab. 3. We observe that the channelwise tokenization is agnostic to the quantization method.

# 7. Conclusion

We introduce a channel-wise flexible tokenizer that achieves excellent rFID with high image throughput, offering the best trade-off between quality and efficiency. Our key insight is that coarse-to-fine semantic hierarchy emerges naturally when tokenizing in the channel dimension, eliminating the need for complex architectural constraints. Future work includes exploring adaptive channel selection mechanisms for task-specific optimization and investigating training strategies to further improve low-token regime performance.

# Acknowledgments

This work was supported by DOE Office of Science’s ASCR AI for Science initiative, the NSF TRAILS Institute (2229885), and Coefficient Giving, and Longview Philanthropy.

Table 3. Quantizer performance across token budgets. 

<table><tr><td rowspan="2">Budget</td><td colspan="2">FSQ</td><td colspan="2">BSQ</td></tr><tr><td>PSNR</td><td>L1</td><td>PSNR</td><td>L1</td></tr><tr><td>32</td><td>15.37</td><td>0.1166</td><td>15.58</td><td>0.1158</td></tr><tr><td>64</td><td>16.33</td><td>0.1020</td><td>16.42</td><td>0.1031</td></tr><tr><td>128</td><td>17.46</td><td>0.0881</td><td>17.34</td><td>0.0911</td></tr><tr><td>256</td><td>18.42</td><td>0.0775</td><td>18.04</td><td>0.0827</td></tr></table>

PSNR in dB; L1 is mean absolute error.

# References

[1] Roman Bachmann, Jesse Allardice, David Mizrahi, Enrico Fini, Oguzhan Fatih Kar, Elmira Amirloo, Alaaeldin El- ˘ Nouby, Amir Zamir, and Afshin Dehghan. Flextok: Resampling images into 1d token sequences of flexible length. arXiv preprint arXiv:2502.13967, 2025.   
[2] Yoshua Bengio, Nicholas Leonard, and Aaron Courville. Es- ´ timating or propagating gradients through stochastic neurons for conditional computation. 2013.   
[3] Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale GAN training for high fidelity natural image synthesis. In Int. Conf. Learn. Represent., 2019.   
[4] Tadeusz Calinski and Jerzy Harabasz. A dendrite method for ´ cluster analysis. Communications in Statistics-Theory and Methods, 3(1):1–27, 1974.   
[5] Shivam Duggal, Sanghyun Byun, William T Freeman, Antonio Torralba, and Phillip Isola. Adaptive length image tokenization via recurrent allocation. arXiv preprint arXiv:2411.02393, 2024.   
[6] Shivam Duggal, Sanghyun Byun, William T Freeman, Antonio Torralba, and Phillip Isola. Single-pass adaptive image tokenization for minimum program search. arXiv preprint arXiv:2507.07995, 2025.   
[7] Patrick Esser, Robin Rombach, and Bjorn Ommer. Taming ¨ transformers for high-resolution image synthesis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 12873–12883, 2021.   
[8] Stephanie Fu, Netanel Y Ramesh, Vongani H Xie, Yue Luo, Philip HS Torr, Joshua B Tenenbaum, Olga Russakovsky, William T Freeman, and Stephanie Wong. Dreamsim: Learning new dimensions of human visual similarity using synthetic data. In Advances in Neural Information Processing Systems, 2023.   
[9] Geoffrey E Hinton and Ruslan R Salakhutdinov. Reducing the dimensionality of data with neural networks. Science, 313 (5786):504–507, 2006.   
[10] Phillip Isola, Jun-Yan Zhu, Tinghui Zhou, and Alexei A Efros. Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 1125–1134, 2017.   
[11] Sadeep Jayasumana, Srikumar Ramalingam, Andreas Veit, Daniel Glasner, Ayan Chakrabarti, and Sanjiv Kumar. Rethinking fid: Towards a better evaluation metric for image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9307–9315, 2024.

[12] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In IEEE Conf. Comput. Vis. Pattern Recog., pages 4401–4410, 2019.   
[13] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013.   
[14] Aditya Kusupati, Gantavya Bhatt, Aniket Rege, Matthew Wallingford, Aditya Sinha, Vivek Ramanujan, William Howard-Snyder, Kaifeng Chen, Sham M. Kakade, Prateek Jain, and Ali Farhadi. Matryoshka representation learning. In Advances in Neural Information Processing Systems 35 (NeurIPS 2022), 2022.   
[15] Lingjun Mao, Zikang Jin, Haokui Wang, Xiaodan Zhang, and Xin Li. Images are worth variable length of representations. arXiv preprint arXiv:2506.03643, 2025.   
[16] Fabian Mentzer, David Minnen, Eirikur Agustsson, and Michael Tschannen. Finite scalar quantization: Vq-vae made simple. arXiv preprint arXiv:2309.15505, 2023.   
[17] Kazuki Miwa, Go Irie, Yuki Nakashima, and Rin-ichiro Taniguchi. One-d-piece: Image tokenizer meets qualitycontrollable compression. arXiv preprint arXiv:2501.10064, 2025.   
[18] Takeru Miyato, Toshiki Kataoka, Masanori Koyama, and Yuichi Yoshida. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations, 2018.   
[19] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy V. ´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve´ Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and ´ Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2023.   
[20] Olga Russakovsky, Jia Deng, Hao Su, Jonathan Krause, Sanjeev Satheesh, Sean Ma, Zhiheng Huang, Andrej Karpathy, Aditya Khosla, Michael S Bernstein, Alexander C Berg, and Li Fei-Fei. Imagenet large scale visual recognition challenge. International Journal of Computer Vision, 115:211– 252, 2014.   
[21] Junhong Shen, Kushal Tirumala, Michihiro Yasunaga, Ishan Misra, Luke Zettlemoyer, Lili Yu, and Chunting Zhou. Cat: Content-adaptive image tokenization. arXiv preprint arXiv:2501.03120, 2025.   
[22] Peize Sun, Yi Jiang, Shoufa Chen, Shilong Zhang, Bingyue Peng, Ping Luo, and Zehuan Yuan. Autoregressive model beats diffusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525, 2024.   
[23] Keyu Tian, Yi Jiang, Zehuan Yuan, Bingyue Peng, and Liwei Wang. Detailflow: 1d coarse-to-fine autoregressive image generation via next-detail prediction. arXiv preprint arXiv:2505.21473, 2024.   
[24] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In Advances in Neural Information Processing Systems, 2017.   
[25] Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to

structural similarity. IEEE Transactions on Image Processing, 13(4):600–612, 2004.   
[26] Wentao Wu, Libin Huang, Wenyi Xu, Qi Chen, Yue Zhang, and Weiwei Zhou. AToken: Adaptive tokenization for vision transformers. arXiv preprint arXiv:2509.14476, 2024.   
[27] Wilson Yan, Matei Zaharia, Volodymyr Mnih, Pieter Abbeel, Aleksandra Faust, and Hao Liu. Elastictok: Adaptive tokenization for image and video. arXiv preprint arXiv:2410.08368, 2024.   
[28] Jingfeng Yao and Xinggang Wang. Quantize-then-rectify: Efficient vq-vae training. arXiv preprint arXiv:2507.10547, 2025.   
[29] Lijun Yu, Jose Lezama, Nitesh B Gundavarapu, Luca Versari, ´ Kihyuk Sohn, David Minnen, Yong Cheng, Agrim Gupta, Xiuye Gu, Alexander G Hauptmann, et al. Language model beats diffusion–tokenizer is key to visual generation. arXiv preprint arXiv:2310.05737, 2023.   
[30] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 586–595, 2018.   
[31] Yue Zhao, Yuanjun Panda, Zhengzhong Xu, Zhenzhong Wang, Gaurav Kumar, Yu Zhang, Jinshuo Zhou, Yan Chen, Guan Wang, Jiaqi Zhang, et al. Image and video tokenization with binary spherical quantization. arXiv preprint arXiv:2406.07548, 2024.   
[32] Shaobin Zhuang, Yiwei Guo, Canmiao Fu, Zhipeng Huang, Zeyue Tian, Fangyikang Wang, Ying Zhang, Chen Li, and Yali Wang. Wetok: Powerful discrete tokenization for high-fidelity visual reconstruction. arXiv preprint arXiv:2508.05599, 2025.

# ChannelTok: Efficient Flexible-Length Vision Tokenization Supplementary Material

# A. Training and Evaluation Details

# A.1. Training

We provide training specifications to facilitate reproducibility in Tab. 4, Tab. 5, and Tab. 6. All training experiments for flexible and baseline tokenizer models were conducted on 8× NVIDIA H100 80GB GPUs using PyTorch with distributed data parallel (DDP) training. Our baseline model is architecturally identical to our flexible tokenizer model, but devoid of the flexible module (causal masking).

# A.2. Training Stability Notes

Several design choices were critical for stable training:

• Two-stage GAN introduction: Applying adversarial loss from initialization caused mode collapse. Delaying until epoch 15 allows reconstruction objectives to establish a stable baseline.   
• Stochastic masking: Deterministic warmup schemes (using full channels for initial epochs) were less stable than our stochastic approach with $p _ { \mathrm { m a s k } } = 0 . 5$ .   
• Learning rate schedule: Cosine decay with warmup is essential when combining reconstruction and adversarial objectives.   
• Gradient clipping: Max norm clipping at 1.0 prevents gradient explosions during masked training.   
• Loss weighting: Higher perceptual weight $( \lambda _ { \mathrm { p e r c } } > \lambda _ { \mathrm { r e c } } )$ improves rFID; values below 0.1 cause instabilities.

# A.3. Evaluation

We evaluate reconstruction quality and system efficiency on the ImageNet-1K validation set (50,000 images). All images undergo standard preprocessing: resizing to 256 × 256 and normalization with mean and standard deviation of [0.5, 0.5, 0.5] across RGB channels. We save preprocessed original images as lossless PNG files to ensure consistent rFID computation.

Reconstruction fidelity is assessed through complementary metrics that are computed using established libraries: LPIPS with VGG backbone via lpips package1, SSIM via torchmetrics2, and rFID via clean-fid3. For rFID computation, both original and reconstructed images are saved as lossless PNG files following the preprocessing protocol above.

![](images/f4a52b140738257f1f0339ebb8dbaa090619741f0b3098d21ca14b4e016fb72a.jpg)

<details>
<summary>line</summary>

| Token Budget | Top-1 Accuracy | Top-5 Accuracy |
| ------------ | -------------- | -------------- |
| 32           | 27             | 47             |
| 64           | 50             | 73             |
| 128          | 68             | 87             |
| 256          | 74             | 91             |
| 512          | 77             | 93             |
</details>

Figure 9. DINOv2 classification accuracy across token budgets. Higher token budgets preserve more discriminative structure, mirroring the rFID trends.

# B. Downstream Analysis

# B.1. Autoregressive Image Generation (LlamaGen)

To evaluate the efficacy of our flexible-length visual tokens, we train an autoregressive (AR) generation model following the LlamaGen framework. We adopt a GPT-L (Large) architecture with 343M parameters. The model is trained unconditionally or class-conditionally on pre-extracted latents of ImageNet-100.

Flexible-Length AR Training Objective. Unlike standard fixed-length tokenization which treats all spatial tokens equally, our autoregressive training utilizes the full token sequence (e.g., 512 tokens) but applies a temporally weighted Cross-Entropy loss. This weighting scheme assigns higher importance to earlier predictions in the sequence, forcing the AR model to prioritize generating the most critical structural and semantic information first. Because the network is explicitly rewarded for early accuracy, we can simply truncate (or ”chop”) the generation process at inference time at any desired token budget $( t \leq t _ { \operatorname* { m a x } } )$ to achieve the desired compute-quality trade-off, entirely avoiding the need for variable-length padding or masking during training.

Optimization Specifications. We optimize the model using fused AdamW with a peak learning rate of $8 \times 1 0 ^ { - 4 }$ . We utilize a cosine decay learning rate schedule with a 10% linear warmup. For class-conditional models, classifier-free guidance is enabled by dropping the class label with a 10% probability. Detailed hyperparameters are provided in Tab. 7.

Table 4. Model architecture specifications. 

<table><tr><td colspan="2">Encoder (65M)</td><td colspan="2">Decoder (95M)</td><td colspan="2">BSQ Quantizer</td></tr><tr><td>Input resolution</td><td> $256^{2} \times 3$ </td><td>Input resolution</td><td> $16^{2} \times 512$ </td><td>Method</td><td>Binary Spherical</td></tr><tr><td>Latent resolution</td><td> $16^{2} \times 512$ </td><td>Output resolution</td><td> $256^{2} \times 3$ </td><td>Codebook size</td><td> $2^{16}$ </td></tr><tr><td>Downsampling</td><td>16×</td><td>Upsampling</td><td>16×</td><td>Dimension</td><td>16 (4 × 4)</td></tr><tr><td>Channels</td><td>[192,256,384,512,640]</td><td>Channels</td><td>[640,512,384,256,192]</td><td>Channel-wise</td><td>True</td></tr><tr><td>Attention res.</td><td>{32,16,8}</td><td>Attention res.</td><td>{8,16,32}</td><td>Lookup-free</td><td>Yes</td></tr><tr><td>Attention heads</td><td>8 (Flash)</td><td>Attention heads</td><td>8 (Flash)</td><td></td><td></td></tr><tr><td>Residual blocks</td><td>2 per res.</td><td>Residual blocks</td><td>3 per res.</td><td colspan="2">Total: 159M params</td></tr></table>

Table 5. Training configuration and hyperparameters. 

<table><tr><td colspan="2">Optimization</td><td colspan="2">Training Schedule</td><td colspan="2">Data Pipeline</td></tr><tr><td>Optimizer</td><td>AdamW (fused)</td><td>Total epochs</td><td>150</td><td>Dataset</td><td>ImageNet-1K</td></tr><tr><td>Peak LR</td><td> $10^{-3}$ </td><td>Batch/GPU</td><td>32</td><td>Train images</td><td>1.28M</td></tr><tr><td>Discriminator LR</td><td> $10^{-4}$ </td><td>Effective batch</td><td>256</td><td>Val images</td><td>50K</td></tr><tr><td>Weight decay</td><td>0.01</td><td>Num workers</td><td>8</td><td>Data format</td><td>WebDataset</td></tr><tr><td>Gradient clip</td><td>1.0 (max norm)</td><td>Precision</td><td>bfloat16 mixed</td><td>Loader</td><td>DALI (GPU)</td></tr><tr><td>LR schedule</td><td>Cosine + warmup</td><td>DDP</td><td>True</td><td>Train aug.</td><td>Crop + flip</td></tr><tr><td>Warmup steps</td><td>5,000</td><td>Sync BN</td><td>True</td><td>Val aug.</td><td>Resize + crop</td></tr><tr><td>Init LR</td><td> $10^{-6}$ </td><td>Compile</td><td>torch.compile</td><td>Normalize</td><td>[0.5,0.5,0.5]</td></tr><tr><td>Min LR</td><td> $5 \times 10^{-5}$ </td><td>Flash Attn</td><td>Enabled</td><td></td><td></td></tr></table>

System: 8× NVIDIA H100 80GB, 48 hours, PyTorch 2.1+

# B.2. Semantic Feature Preservation

To measure how well our tokenizer retains semantic information, we evaluate reconstructed images using a pretrained DinoV2 [19] backbone on the ImageNet-1K validation set. As shown in Fig. 9, both Top-1 and Top-5 accuracy increase steadily with the reconstruction token budget, indicating that higher budgets preserve more discriminative structure. This trend closely follows the rFID improvements reported earlier, confirming that perceptual fidelity and semantic retention improve hand-in-hand in our model.

# B.3. Semantic Organization in Early Channels

We perform K-means clustering (K = 50) on the first 32 channels of quantized latents from the ImageNet-1K validation set. As shown in Fig. 10, clusters exhibit coherent semantic groupings along two axes: (1) global scene characteristics (marine blues, verdant backgrounds, warm palettes, high-contrast compositions) and (2) object-level semantics (aquatic life, reptiles, birds, canines, produce). These groupings emerge organically without class supervision, confirming early channels encode coarse categorical structure. A Calinski-Harabasz [4] score of 404.7 confirms well-separated centroids across ImageNet’s categories.

# C. Per-Class Token Budget Analysis

We conduct an analysis of token requirements across all 1,000 ImageNet-1K validation classes to understand our tokenizer’s flexible allocation behavior across different classes. For each class, we determine the minimum number of tokens required to achieve a perceptual quality threshold (LPIPS≤ 0.15). Aggregating statistics yields per-class distributions characterized by mean, revealing the relationship between semantic content and representational capacity.

Fig. 11 shows the correlation between visual complexity and token allocation. Geometrically simple classes such as Airship, Parachute, and Nematode consistently achieve the perceptual threshold with mean token budgets of only 56–87 tokens, well below the global per-class mean of 199.6 tokens. Conversely, classes with intricate textures and fine-grained details, including Coral fungus, Toyshop, Rotisserie, and Jinrikisha, require considerably higher budgets of 392–494 tokens on average. Fig. 11 summarizes this gap by plotting the mean token counts for the five simplest and five most complex classes, making the inter-class stratification explicit: token budgets align with visual complexity rather than being uniformly allocated across categories.

Table 6. Loss configuration and channel masking. 

<table><tr><td rowspan="2">Component</td><td colspan="2">Loss Components</td><td colspan="2">Channel Masking</td></tr><tr><td>Weight</td><td>Schedule</td><td>Parameter</td><td>Value</td></tr><tr><td colspan="3">Stage 1: Reconstruction (Epochs 0-14)</td><td>Mask prob.  $p_{mask}$ </td><td>0.5</td></tr><tr><td>L1 reconstruction</td><td> $\lambda_{rec} = 0.25$ </td><td>All epochs</td><td>Min retention  $t_{min}$ </td><td>0.002</td></tr><tr><td>Perceptual (LPIPS)</td><td> $\lambda_{perc} = 1.0$ </td><td>All epochs</td><td>Max retention  $t_{max}$ </td><td>1.0</td></tr><tr><td>BSQ entropy</td><td> $\lambda_{ent} = 0.1$ </td><td>All epochs</td><td>Sampling</td><td> $\mathcal{U}(t_{min}, t_{max})$ </td></tr><tr><td>BSQ commitment</td><td> $\lambda_{commit} = 0.25$ </td><td>All epochs</td><td>Total channels  $C$ </td><td>512</td></tr><tr><td>BSQ diversity  $\gamma$ </td><td>1.0</td><td>All epochs</td><td>Active channels  $k$ </td><td> $\max(1, \lfloor t \cdot C \rfloor)$ </td></tr><tr><td colspan="3">Stage 2: + Adversarial (Epoch 15+)</td><td>Stop-gradient</td><td>True (inactive)</td></tr><tr><td>Generator adversarial</td><td> $\lambda_{gan} = 0.1$ </td><td>Epoch  $\geq 15$ </td><td></td><td></td></tr><tr><td>Discriminator</td><td>PatchGAN (3 layers, 64 ch)</td><td>Epoch  $\geq 15$ </td><td></td><td></td></tr><tr><td>Adversarial loss</td><td>Hinge loss</td><td>Epoch  $\geq 15$ </td><td></td><td></td></tr></table>

Table 7. Autoregressive Generation configuration and hyperparameters. 

<table><tr><td colspan="2">Architecture (GPT-L)</td><td colspan="2">Optimization</td><td colspan="2">Flexible Generation Setup</td></tr><tr><td>Parameters</td><td>343M</td><td>Optimizer</td><td>AdamW (fused)</td><td>Training length</td><td>512 (Full Budget)</td></tr><tr><td>Layers</td><td>12</td><td>Peak LR</td><td> $8 \times 10^{-4}$ </td><td>Inference length</td><td>Dynamic (32–512)</td></tr><tr><td>Attention heads</td><td>12</td><td>Weight decay</td><td>0.05</td><td>Objective</td><td>Weighted Cross-Entropy</td></tr><tr><td>Hidden dimension</td><td>768</td><td>Betas ( $\beta_1$ ,  $\beta_2$ )</td><td>(0.9, 0.95)</td><td>Early token reward</td><td>High</td></tr><tr><td>Vocab size</td><td>16384</td><td>Gradient clip</td><td>1.0 (max norm)</td><td>Resid/FFN Dropout</td><td>0.1</td></tr><tr><td>Max seq. length</td><td>512</td><td>LR schedule</td><td>Cosine</td><td>Class drop prob.</td><td>0.1</td></tr><tr><td>Pos. Encoding</td><td>1D Absolute/RoPE</td><td>Warmup fraction</td><td>0.1 (10%)</td><td>Token dropout</td><td>0.1</td></tr><tr><td>Hardware</td><td>8× H100</td><td></td><td></td><td></td><td></td></tr></table>

# D. Ablations

We conduct ablations (Sec. 6) on ImageNet-100 to analyze key design choices. For scaling trends (Sec. 6.2), we train four model sizes: Tiny (6.3M), Small (18.16M), Medium (31.73M), and Large (72.39M). We perform network-width ablation by progressively increasing channel dimensions of convolutional layers in the encoder-decoder backbone, while keeping latent space (256-D), quantizer (BSQ with 65,536 codes), training schedule, and loss configuration fixed.

Sec. 6.1 investigates sampling bias effects on retention ratio t at fixed $p _ { \mathrm { m a s k } } = 0 . 5$ using three piecewise-uniform distributions:

• bias lower: Samples from lower 25% of $[ t _ { \mathrm { m i n } } , t _ { \mathrm { m a x } } ]$ with probability 0.75, remainder otherwise   
• bias higher: Symmetrically emphasizes upper 25% of $[ t _ { \mathrm { m i n } } , t _ { \mathrm { m a x } } ]$ with probability 0.75   
• uniform: Standard uniform distribution on $[ t _ { \mathrm { m i n } } , t _ { \mathrm { m a x } } ]$

To isolate the gains of the channel-wise paradigm from the quantizer choice, we run our tokenizer with FSQ [16] as a drop-in replacement for BSQ (Sec. 6.3). Both yield similar trends across budgets, confirming the improvements stem from the channel-wise paradigm, not the quantizer.

# E. Performance Across Token Budgets

Complementing our quantitative analysis in Fig. 3, Tab. 8 presents additional reconstruction quality metrics (LPIPS, SSIM, PSNR) across token budgets. Our method shows consistent perceptual quality improvement across token budgets (LPIPS: 0.344 → 0.153, SSIM: 0.363 → 0.556). At higher token counts (512), we achieve competitive fidelity with FlexTok (LPIPS 0.153 vs 0.228) while maintaining 8.6× faster decoding.

# F. Qualitative Analysis

We evaluate reconstruction across diverse visual scenarios: contrasting foreground-background (Fig. 14), varied textures (Fig. 15), scenes with vibrant colors and landscapes (Fig. 16). Our method shows consistent quality improvement with token budget, maintaining competitive fidelity at 64– 128 tokens. Fig. 12 contrasts our flexible tokenizer against a baseline without prefix channel masking. We also present qualitative autoregressive generation results across token budgets in Fig. 13, demonstrating coherent global structure even at 32 tokens.

Clusters (? = 50)   
![](images/89f83642f52371c8c4de40dc117f95b39f811c4c00e1226d5fa5eb260492633e.jpg)

Figure 10. Semantic clustering of early channels. K-means clustering on first 32 channels produces semantically coherent groups organized by scene characteristics (black, marine blue, greenery), hinting that early channels encode meaningful semantic structure. Beyond global scene attributes, clusters also align with object-level semantics, grouping marine life and birds into distinct regions. Crucially, this organisation is never explicitly supervised: it emerges purely from the channel-wise masking objective, suggesting that prioritising early channels during training naturally induces a semantically ordered latent space.   
![](images/36e86889144bfe033ce36721d33348f9dc29c5f31ce289f2e5843b12a92752b1.jpg)

<details>
<summary>natural_image</summary>

Collage of diverse images including superhero, kitchen, food, and space-themed scenes (no text or symbols)
</details>

![](images/7e17979dca1d24e1db4a046af5bcfecfd17e67b12da789e5eee843138cd3f877.jpg)

<details>
<summary>bar</summary>

| Category | Simplest Classes | Most Complex Classes |
| :--- | :--- | :--- |
| Nematode | 56.2 | - |
| jigsaw puzzle | 57.5 | - |
| Airship | 76.2 | - |
| Matchstick | 84.5 | - |
| Parachute | 86.7 | - |
| Butcher shop | - | 392.7 |
| Rotisserie | - | 395.8 |
| Coral fungus | - | 413.0 |
| Jimrikisha | - | 422.0 |
| Toyshop | - | 494.0 |
Overall Mean: 199.6
</details>

Figure 11. Token allocation across ImageNet-1K validation classes. Left: Rows 1–2 show complex classes that require high token counts: Coral fungus (498, 490), Toyshop (494), Rotisserie (491, 459, 442, 432), and Jinrikisha (422), all featuring intricate textures and fine-grained details. Rows 3–4 show visually simple classes that need far fewer tokens: Airship (5, 6, 9, 12), Parachute (9), and Nematode (12, 12, 13), characterised by uniform backgrounds and simple structures. Right: Per-class mean token counts for the five simplest and five most complex classes. Teal bars denote simple classes, red bars complex classes, and the dashed line marks the dataset mean (199.6 tokens). The clear separation between the two groups shows that our tokenizer adapts its budget to visual complexity rather than allocating tokens uniformly across categories. The complexity ordering also aligns with human intuition, as classes that people would judge as visually intricate (dense textures, cluttered scenes) consistently demand more tokens, while perceptually simple classes (plain backgrounds, minimal structure) require far fewer.

Table 8. Reconstruction quality across token budgets on the ImageNet-1K validation set. We report LPIPS ↓, SSIM ↑, and PSNR ↑ at five token budgets. Our method is the only one that scales to 512 tokens, where it achieves LPIPS 0.153 (matching the best score of any competing method) with a 1.8× lighter model (159M vs. 287M parameters). Quality improves monotonically at every budget (LPIPS: 0.344 → 0.153), reflecting the coarse-to-fine channel hierarchy learned during training. 

<table><tr><td rowspan="2">Method</td><td colspan="3">32 Tokens</td><td colspan="3">64 Tokens</td><td colspan="3">128 Tokens</td><td colspan="3">256 Tokens</td><td colspan="3">512 Tokens</td></tr><tr><td>LPIPS ↓</td><td>SSIM ↑</td><td>PSNR ↑</td><td>LPIPS ↓</td><td>SSIM ↑</td><td>PSNR ↑</td><td>LPIPS ↓</td><td>SSIM ↑</td><td>PSNR ↑</td><td>LPIPS ↓</td><td>SSIM ↑</td><td>PSNR ↑</td><td>LPIPS ↓</td><td>SSIM ↑</td><td>PSNR ↑</td></tr><tr><td>OneDPiece</td><td>0.372</td><td>0.379</td><td>15.66</td><td>0.280</td><td>0.409</td><td>16.83</td><td>0.212</td><td>0.450</td><td>18.09</td><td>0.180</td><td>0.472</td><td>18.80</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DOVE</td><td>0.197</td><td>0.596</td><td>20.34</td><td>0.168</td><td>0.619</td><td>21.06</td><td>0.165</td><td>0.623</td><td>21.17</td><td>0.153</td><td>0.633</td><td>21.54</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ALIT</td><td>0.301</td><td>0.350</td><td>15.86</td><td>0.237</td><td>0.394</td><td>17.13</td><td>0.186</td><td>0.443</td><td>18.42</td><td>0.147</td><td>0.477</td><td>19.52</td><td>-</td><td>-</td><td>-</td></tr><tr><td>KARL</td><td>0.401</td><td>0.371</td><td>15.42</td><td>0.335</td><td>0.413</td><td>16.66</td><td>0.237</td><td>0.489</td><td>18.81</td><td>0.154</td><td>0.568</td><td>21.08</td><td>-</td><td>-</td><td>-</td></tr><tr><td>FlexTok</td><td>0.434</td><td>0.327</td><td>14.24</td><td>0.370</td><td>0.361</td><td>15.14</td><td>0.290</td><td>0.443</td><td>17.16</td><td>0.228</td><td>0.523</td><td>19.19</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Ours</td><td>0.344</td><td>0.363</td><td>15.72</td><td>0.263</td><td>0.427</td><td>16.90</td><td>0.201</td><td>0.497</td><td>18.17</td><td>0.169</td><td>0.536</td><td>19.05</td><td>0.153</td><td>0.556</td><td>19.56</td></tr></table>

![](images/48e566a3732bdd58282ac85570dd584c0b1c8d0595c03823235bc0913076b963.jpg)

<details>
<summary>text_image</summary>

Original
32 tokens
48 tokens
64 tokens
96 tokens
112 tokens
128 tokens
256 tokens
512 tokens
Ours
Baseline
Ours
Baseline
</details>

Figure 12. Reconstruction with and without prefix masking. Each image pair shows channel progression across increasing token budgets. The first row is our flexible tokenizer and the second row is the baseline, which is architecturally identical but trained without channel-wise adaptive masking. Without masking, the baseline produces no meaningful reconstruction at low token counts, with recognisable structure emerging only at high budgets. Our flexible tokenizer, by contrast, exhibits a clear coarse-to-fine progression, recovering global semantics early and refining detail as more channels are added.

![](images/dd28a2f44ca927cc310491dde869ad5d17b0771c563ed9695326d2c965ccdb75.jpg)

<details>
<summary>text_image</summary>

t=32
t=64
t=128
t=256
t=32
t=64
t=128
t=256
</details>

Figure 13. Autoregressive generation across token budgets. LlamaGen [22] GPT-L generations across diverse ImageNet-100 categories (birds, insects, annelids, and marine life) using discrete channel tokens with truncated channels zero-filled. Even at 32 tokens, outputs maintain coherent global structure, with fidelity improving progressively at higher budgets. Generation at such low token counts is made possible by our tokenizer’s channel ordering, which concentrates the most semantically meaningful information into the earliest tokens.

![](images/0f1effd5d2ad780bafd119ddc356e3d3b33de92ec9c6b96c98a5c4ac19a47ae2.jpg)

<details>
<summary>text_image</summary>

Original
Ours
ALIT
DOVE
FlexTok
KARL
One-D-Piece
32 tokens
64 tokens
128 tokens
256 tokens
32 tokens
64 tokens
128 tokens
256 tokens
</details>

Figure 14. Qualitative comparison on images with contrasting tones. Top: A jellyfish against a dark background, where our method preserves color fidelity even at lower token budgets. Bottom: A butterfly on a flower, where subtle wing textures and fine details emerge progressively with increasing tokens. Our method maintains perceptual coherence and colour consistency across all budgets.

![](images/eded484d4ab85b6578a0ac2d208e317b90ab4e755da1ad0101f3ef2a58de99b3.jpg)  
Figure 15. Qualitative comparison on images with varied textures. Top: A red mushroom with white spots against a mossy background, where our method preserves fine surface detail and color fidelity even at low token budgets. Bottom: A dark round fruit, where competing methods introduce color artifacts and lose surface sheen at low tokens, while ours maintains perceptual consistency across all budgets.

![](images/f6e55c664af62d22639d6bb9b79a7890802f6f4f95766062cae4f44929253786.jpg)

<details>
<summary>text_image</summary>

Original
Ours
ALIT
DOVE
FlexTok
KARL
One-D-Piece
32 tokens
64 tokens
128 tokens
256 tokens
32 tokens
64 tokens
128 tokens
256 tokens
</details>

Figure 16. Reconstructions on cases with text and vibrant colours. Top: Christmas stocking with text, where legibility remains difficult at low token counts but improves by 128 tokens. Bottom: A geyser eruption scene, where our method recovers landscape structure.