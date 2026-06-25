# MIMFlow: Integrating Masked Image Modeling with Normalizing Flows for End-to-End Image Generation

Yang Chen<sup>1,2</sup>, Xiaowei Xu<sup>2</sup>, Shuai Wang<sup>1</sup>, Xinwen Zhang<sup>1,2</sup>, Qiushi Guo<sup>2</sup>, Tiezheng Ge<sup>2</sup>, and Limin Wang<sup>1,3,⋆</sup>

1 State Key Laboratory for Novel Software Technology, Nanjing University <sup>2</sup> Alibaba Grou p <sup>3</sup> Shanghai AI Lab ⋆ Corresponding author: lmwang@nju.edu.cn Code: https://github.com/MCG-NJU/MIMFlow

Abstract. Normalizing Flows (NFs) are powerful generative models capable of exact density estimation and sampling. However, their strict invertibility often forces the model to exhaust its capacity on low-level pixel details, hindering the capture of high-level semantic structures. While Masked Image Modeling (MIM) has excelled in representation learning, its integration into generative pipelines has remained largely modular and disjointed. In this paper, we propose MIMFlow, a unified end-to-end framework that jointly optimizes latent semantics, pixel reconstruction, and generative flow. By employing a VAE encoder to infer semantic latent from masked images, MIMFlow achieves a principled decoupling of the generative task: the Normalizing Flow focuses on modeling a simplified, low-frequency semantic manifold, while a specialized decoder handles high-frequency synthesis. This design efectively resolves the inherent capacity bottleneck of NFs, allowing the model to prioritize global structural coherence over redundant noise. Empirical results on ImageNet 256×256 show that MIMFlow-L reaches 71.3% linear probing accuracy and an FID of 2.50. Despite using only 128 tokens (50% fewer than standard models), it yields a 32.8% performance gain over similar-scale NF baselines.

Keywords: Image Generation · Normalizing Flow · MIM

## 1 Introduction

Normalizing Flows (NFs) are generative models that map a complex data distribution to a simple prior distribution through a series of invertible and diferentiable transformations [4, 8, 15, 16, 21, 36, 39, 40, 48, 49, 51]. A primary advantage of NFs is their ability to perform both exact density estimation and sampling within a single network. Besides, from the perspective of flow matching, NFs can be viewed as the expansion of an Ordinary Diferential Equation (ODE), allowing for end-to-end optimization of the entire flow process [2]. Recent work, such as SimFlow [49], has further demonstrated that NFs can provide highly expressive priors for Variational Autoencoders (VAEs) by leveraging their likelihood estimation capabilities. However, strict invertibility forces NFs to prioritize low-level pixel details over high-level semantics, exhausting model capacity and hindering generative quality.

![](images/7e8236508f1a74e99ca1e1d8d7ac1c03752f0911b62dcb8518a9550a2e4560a3.jpg)  
Fig. 1: MIM in Diferent Paradigms. (a) Self-Supervised Learning: Employs highratio masking as a self-supervised proxy task for representation learning. (b) Generative Tokenizers: A two-stage approach where the latent space is pre-trained with MIM before training a separate generative model. (c) MIMFlow (Ours): A unified framework that jointly optimizes latent semantics, pixel reconstruction, and generative flow in an endto-end manner.

While NFs struggle to capture global structures, Masked Image Modeling (MIM, Fig. 1a) has established itself as a cornerstone of self-supervised representation learning [17, 43]. Despite its success in discriminative tasks, the role of MIM in generative modeling remains relatively under-explored. Recent literature suggests that integrating discriminative features can significantly bolster generative performance, with some approaches distilling knowledge from pre-trained representation models to guide the generation process [24, 47, 53]. Furthermore, RAE [37, 50] directly employs representation models as VAE encoders. Specifically, they found that MAE lags behind DINO in these generative frameworks. This likely stems from the mismatch between MAE’s high-mask reconstruction and the continuous distribution modeling required for synthesis. Thus, whether semantic-focused MIM can efectively enhance generative models remains an open question.

Recent attempts to incorporate MIM into generative pipelines have primarily focused on refining visual tokenizers. Specifically, works such as MAETok [1] and DeTok [44] have pioneered the use of masking as a denoising or discriminative objective to enhance the robustness of latent structures. Building on this, VTP [45] demonstrated that unifying MIM with contrastive learning can further foster a high-level understanding essential for tokenizer scalability. However, these methods largely treat the tokenizer as a modular, standalone precursor, separating representation learning from the core generative process ( Fig. 1b). In contrast, our method jointly optimizes representation, reconstruction and generation in an end-to-end framework ( Fig. 1c). This design enables a principled decoupling of the generative task: the NF is dedicated to modeling the low-frequency semantic manifold, while a specialized decoder handles high-frequency synthesis. By reducing the burden on NFs to capture pixel-level noise, MIMFlow mitigates a central capacity bottleneck of traditional NFs and ofers an NF-oriented recipe for more semantic latent modeling.

Specifically, our MIMFlow employs a VAE encoder with learnable tokens to extract stable latent representations from masked images. These latents are then optimized through a dual-objective framework: a NF performs exact density estimation, while a decoder focuses on pixel-level reconstruction. This architecture facilitates a strategic division of labor: by mandating the encoder to infer missing spatial context, the resulting latent space is biased toward global structural coherence rather than redundant local noise. Consequently, the NF can model a simpler semantic manifold instead of dense pixel-level correlations. Crucially, linear probing evaluations reveal a substantial increase in classification accuracy within the latent space, validating the enhanced semantic quality of our representations. Finally, on the ImageNet 256×256 benchmark, MIMFlow improves similar-scale NF baselines while using fewer latent tokens.

In summary, our main contributions are as follows:

– MIMFlow Framework: We propose an end-to-end framework that unifies representation and generation, moving beyond the modular limitations of existing tokenizer-based methods.

– Principled Decoupling: We introduce a strategy to decouple semantic modeling from pixel-level synthesis, efectively resolving the inherent capacity bottleneck of NFs.

– Empirical Excellence: We demonstrate that MIMFlow significantly enhances latent semantics and achieves superior generative performance on the ImageNet 256 × 256 benchmark.

## 2 Related Work

Normalizing Flows [7–10, 13, 21, 22, 28, 33] provide a mathematically principled framework for bidirectional mapping between data and latent spaces. Early works like RealNVP [8] and Glow [21] established exact log-likelihood estimation through coupling layers, though they struggled to scale to high-resolution synthesis. Recent advancements [4, 14, 39, 48] have revitalized the field by integrating autoregressive Transformers and latent-space modeling, efectively leveraging NFs for both high-fidelity generation and semantic alignment. Building on this, SimFlow [49] demonstrates that NFs can serve as highly expressive probability estimators to replace restrictive VAE priors, significantly enhancing generative performance. While these models focus on architectural scaling or post-hoc alignment, our work is the first to integrate MIM objectives directly into the end-to-end training of NFs. By unifying self-supervised representation learning with generative flow optimization, we fully exploit the dual potential of NFs for simultaneous robust feature extraction and high-quality synthesis.

Masked Image Modeling has emerged as a dominant paradigm in self-supervised learning, with methods like MAE [17] and SimMIM [43] demonstrating that reconstructing masked inputs forces encoders to learn robust structural features. While initially designed for discriminative tasks, it has also been utilized in generative frameworks. Specifically, recent literature explores distilling knowledge from pre-trained vision foundation models to guide difusion processes [47, 53]. Within the context of visual tokenizers, MAETok [1] and DeTok [44] employ masking as a denoising or discriminative auxiliary task to enhance latent robustness. Unlike methods above, MIMFlow integrates MIM directly into end-to-end generative training, ensuring that the generative flow is conditioned on highly compressed, semantic-rich features.

End-to-End Joint Training. Generative models typically follow a two-stage pipeline—training a VAE for reconstruction and then a generative model on the frozen latent space. However, this can lead to a mismatch where high reconstruction fidelity fails to translate into generative quality. To bridge this gap, REPA-E [24] and SimFlow [49] have pioneered end-to-end joint training, with SimFlow optimizing NFs and VAEs simultaneously from scratch. While our MIMFlow adopts this end-to-end philosophy, it distinguishes itself by incorporating a masked bottleneck to explicitly decouple semantic modeling from texture synthesis. This approach shields the NF from high-frequency noise and ensures the learned latent space is inherently optimized for global structure.

## 3 Method

In this section, we present the architecture and mathematical formulation of MIMFlow, a unified generative framework that integrates MIM with Normalizing Flows. As illustrated in Fig. 2, the framework comprises three core modules. First, a Masked Encoder $\left( E _ { \phi } \right)$ extracts robust latent representations by capturing global structural semantics from masked images. Second, a Latent Normalizing Flow $\left( f _ { \theta } \right)$ performs probabilistic modeling by mapping these representations to a Gaussian prior, enabling exact density estimation as the prior for VAE modeling. Finally, a Generative Decoder $\left( D _ { \psi } \right)$ handles pixel-level reconstruction and texture synthesis. This hierarchical design ensures that the flow model focuses on global structure while the decoder handles local details. In the following, we detail the latent extraction mechanism, the probabilistic modeling framework, and our multi-stage optimization strategy involving auxiliary supervision and adversarial refinement.

## 3.1 Semantic Latent Extraction via Learnable Tokens

A fundamental challenge in integrating MIM with NFs lies in the construction of a stable and expressive latent space. Conventional MIM backbones, such as

![](images/ea65cb7f445432680931ca95bea835c01f069bb61dec3c8963178a09c9b66293.jpg)  
Fig. 2: Structure of MIMFlow. N is the number of image patches, K is the number of learnable latent query tokens, m is the binary mask, and e denotes learnable decoder embeddings.

MAE [17] and SimMIM [43], present inherent dificulties for density estimation. MAE only processes visible patches, resulting in a latent sequence whose length and positional context vary with the random mask pattern, which imposes an intractable burden on the NF to learn a consistent distribution. Conversely, while SimMIM maintains a fixed sequence length by utilizing mask tokens, the information density within each token fluctuates significantly depending on its masking status. Such stochasticity prevents the NF from converging on a stable semantic manifold.

To resolve these issues, we introduce a Learnable Token Bottleneck to extract fixed-dimensional latent representations. We posit that since masked images contain lower information density than full images, they can be more efficiently compressed into a compact latent space. Formally, given an input image x, we partition it into N patches and apply a random masking ratio. The resulting N tokens (including mask tokens) are concatenated with K learnable query tokens, where $K < N$ . These $N + K$ tokens are processed by a bidirectional Transformer encoder $E _ { \phi }$ , enabling the learnable queries to aggregate global semantic information from the partially observed image through self-attention.

After the encoding stage, we extract only the K refined query tokens as our latent representation $\check { \mathbf { z } } \in \mathbb { R } ^ { K \times D }$ . Following the practice of SimFlow [49], we inject additive Gaussian noise with a fixed variance to z during training to facilitate continuous density modeling:

$$
\hat {\mathbf {z}} = \mathbf {z} + \sigma \epsilon , \quad \epsilon \sim \mathcal {N} (0, \mathbf {I}),\tag{1}
$$

where σ is a hyperparameter. This ˆz serves a dual purpose within our framework:

– Generative Modeling: It is treated as a 1D sequence and fed into the NF $f _ { \theta }$ for exact likelihood estimation, and reverse sampling.

– Reconstruction: It is passed to the decoder $D _ { \psi }$ , where it is combined with learnable image embeddings e to reconstruct the original pixel-level signal through cross-modal attention.

This design provides several strategic advantages. First, the fixed-length K tokens provide a stable target for the NF, regardless of the random mask positions. Second, the $K < N$ bottleneck forces the model to discard local pixel redundancies and concentrate on high-level structural semantics. Finally, the flexible masking ratio during training enhances the robustness of the latent space, ensuring that the NF models a manifold that is truly representative of the underlying global data structure.

## 3.2 Probabilistic Modeling and Joint Optimization

In this section, we formulate MIMFlow as a conditional generative model grounded in the Variational Inference (VI) framework. Specifically, we treat the encoder $E _ { \phi }$ and decoder $D _ { \psi }$ as a Variational Autoencoder (VAE) where the standard Gaussian prior is replaced by a high-capacity Normalizing Flow $f _ { \theta }$ . Unlike standard VAEs that take the complete image as input, our encoder is conditioned on the masked image $\tilde { \mathbf { x } } = \mathbf { x } \odot ( 1 - \mathbf { m } )$ , where m denotes the binary mask.

Variational Inference Perspective. Our objective is to maximize the loglikelihood of the data distribution $p ( \mathbf { x } | \mathbf { m } )$ . Following the variational principle, we introduce an approximate posterior $q _ { \phi } ( { \bf z } | \tilde { \bf x } )$ and derive the Evidence Lower Bound (ELBO):

$$
\log p (\mathbf {x} | \mathbf {m}) \geq \mathbb {E} _ {q _ {\phi} (\mathbf {z} | \tilde {\mathbf {x}})} [ \log p _ {\psi} (\mathbf {x} | \mathbf {z}) ] - D _ {\mathrm{KL}} (q _ {\phi} (\mathbf {z} | \tilde {\mathbf {x}}) \| p _ {\theta} (\mathbf {z})) = \mathcal {L} _ {\mathrm{ELBO}},\tag{2}
$$

where $p _ { \psi } ( \mathbf { x } | \mathbf { z } )$ is the reconstruction likelihood and $p _ { \boldsymbol { \theta } } ( \mathbf { z } )$ is the prior distribution modeled by the Normalizing Flow.

In our framework, we define $q _ { \phi } ( { \bf z } | \tilde { \bf x } )$ as a Gaussian distribution with fixed variance $\sigma ^ { 2 }$ centered at the encoder’s output $E _ { \phi } ( \tilde { \mathbf { x } } )$ . This simplifies the KL divergence term to the cross-entropy between the posterior and the flow-based prior (omitting constant entropy terms):

$$
D _ {\mathrm{KL}} (q _ {\phi} (\mathbf {z} | \tilde {\mathbf {x}}) \parallel p _ {\theta} (\mathbf {z})) \propto - \mathbb {E} _ {q _ {\phi} (\mathbf {z} | \tilde {\mathbf {x}})} [ \log p _ {\theta} (\mathbf {z}) ] + C.\tag{3}
$$

Density Estimation via Normalizing Flow. The prior $p _ { \boldsymbol { \theta } } ( \mathbf { z } )$ is parameterized by an invertible transformation $f _ { \theta }$ that maps the complex latent z to a simple base distribution $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ . Using the change-of-variables formula, the exact log-likelihood of the latent z can be computed as:

$$
\log p _ {\theta} (\mathbf {z}) = \log p _ {\epsilon} (f _ {\theta} (\mathbf {z})) + \log \left| \det \frac {\partial f _ {\theta} (\mathbf {z})}{\partial \mathbf {z}} \right|,\tag{4}
$$

where the first term represents the density under the Gaussian prior, and the second term is the log-determinant of the Jacobian, accounting for the volume change induced by the transformation. By optimizing this term, the NF efectively learns to warp the simple Gaussian into a sophisticated semantic manifold that captures the global dependencies of the image.

Joint Training Objective. Combining the reconstruction and the flow-based prior, we arrive at the total loss function for MIMFlow. The reconstruction term $\mathbb { E } _ { q _ { \phi } } \left[ \log p _ { \psi } ( \mathbf { x } | \mathbf { z } ) \right]$ is implemented as a combination of $\ell _ { 2 }$ loss and perceptual loss to ensure both pixel-level fidelity and semantic coherence:

$$
\mathcal {L} _ {\mathrm{rec}} = \| \mathbf {x} - D _ {\psi} (\mathbf {z}) \| _ {2} ^ {2} + \lambda_ {\mathrm{perc}} \mathcal {L} _ {\mathrm{LPIPS}} (\mathbf {x}, D _ {\psi} (\mathbf {z})),\tag{5}
$$

where ${ \mathbf z } \sim q _ { \phi } ( { \mathbf z } | \tilde { \mathbf { x } } )$ . Notably, since z is extracted from the masked image $\tilde { \mathbf { x } } .$ the decoder is forced to perform both denoising and inpainting, which encourages the latent space to prioritize high-level structure over local noise.

The final joint loss is defined as:

$$
\mathcal {L} _ {\text { prob }} = \mathcal {L} _ {\text { rec }} + \beta \mathcal {L} _ {\text { NF }},\tag{6}
$$

where $\mathcal { L } _ { \mathrm { N F } } = - \log p _ { \theta } ( \mathbf { z } )$ is the negative log-likelihood (NLL) provided by the NF. By jointly optimizing ϕ, ψ and θ, MIMFlow ensures that the latent space is simultaneously optimized for representation, reconstruction and density estimation, leading to a more robust and expressive generative model.

## 3.3 Auxiliary Supervision and Adversarial Refinement

To further enrich the latent representations and enhance the perceptual quality of the synthesized images, we incorporate auxiliary semantic supervision and a subsequent adversarial fine-tuning stage.

Auxiliary Feature Prediction. Following the design of generative tokenizers such as MAETok [1], we augment the training objective with an auxiliary discriminative task. In addition to the primary pixel decoder, a lightweight MLPbased auxiliary decoder $D _ { a u x }$ is employed to predict high-level features $\mathbf { F } _ { t a r g e t }$ (e.g., from DINO [27] or CLIP [31]) directly from the latent z. The auxiliary loss is defined as:

$$
\mathcal {L} _ {\mathrm{aux}} = \| D _ {a u x} (\mathbf {z}) - \mathrm{sg} (\mathbf {F} _ {t a r g e t} (\mathbf {x})) \| _ {2} ^ {2}.\tag{7}
$$

By supervising the latent space with pre-trained discriminative priors, this objective encourages the encoder to capture more robust semantic context beyond low-level pixel statistics. During the end-to-end training phase, the total loss is formulated as $\mathcal { L } = \mathcal { L } _ { \mathrm { r e c } } + \beta \mathcal { L } _ { \mathrm { N F } } + \gamma \mathcal { L } _ { \mathrm { a u x } }$ , ensuring that the Normalizing Flow models a latent distribution that is both reconstructive and semantically rich.

Adversarial Fine-tuning. While the joint training phase establishes a structured latent space, the resulting reconstructions often lack the high-frequency details necessary for photorealistic generation. To address this, we perform a targeted fine-tuning of the pixel decoder $D _ { \psi }$ . In this stage, we introduce a patch-based discriminator D and optimize the model using a combination of reconstruction loss and GAN loss:

$$
\mathcal {L} _ {\mathrm{FT}} = \mathcal {L} _ {\mathrm{rec}} + \alpha \mathcal {L} _ {\mathrm{GAN}} (D _ {\psi}, \mathcal {D}).\tag{8}
$$

Crucially, during fine-tuning, the encoder $E _ { \phi }$ continues to receive the masked image x˜ as input. This preserves distributional consistency in the latent space: the decoder continues to observe latents from the same masked posterior family that the NF was trained to model, while sampling draws latents from the corresponding NF-modeled manifold rather than relying on an explicit mask. This allows the decoder to focus on synthesizing sharp textures without disrupting the probabilistic alignment between the generative flow and the latent representations.

## 4 Experiment

## 4.1 Experimental Setup

Datasets and Evaluation Metrics We conduct our experiments on the ImageNet dataset at a 256 × 256 resolution [5]. To comprehensively evaluate the generative performance, we report the Fréchet Inception Distance (FID) [18], Inception Score (IS) [35], Precision, and Recall [23], all calculated using the evaluation suite provided by ADM [6]. For reconstruction quality, we compute the reconstruction FID (rFID) on the ImageNet validation set. Furthermore, to assess the semantic quality of the learned latent space, we perform linear probing on the validation set following REPA’s [47] protocol, reporting the top-1 classification accuracy.

Implementation Details Our framework consists of two primary components: the Normalizing Flow prior and the Transformer-based Autoencoder.

Normalizing Flow Architecture We adopt the STARFlow [14] architecture for density estimation in the latent space. Our model, designated as STARFlow-L, comprises approximately 482M parameters with a hidden dimension of 1024. The backbone consists of seven 2-layer blocks followed by a final 20-layer block.

Table 1: Baseline Construction (last line), evaluated with 10K samples.

<table><tr><td rowspan="2">Configuration</td><td colspan="2">w/o CFG</td><td colspan="2">w/ CFG</td></tr><tr><td>gFID ↓</td><td>gFID ↓</td><td>sFID ↓</td><td></td></tr><tr><td>STARFlow-L</td><td>50.22</td><td>7.79</td><td>20.54</td><td></td></tr><tr><td>- Softplus</td><td>48.24</td><td>7.65</td><td>18.98</td><td></td></tr><tr><td>+ Gated Attn.</td><td>47.44</td><td>7.45</td><td>18.64</td><td></td></tr><tr><td>+ End-to-End</td><td>11.24</td><td>5.99</td><td>-</td><td></td></tr></table>

We iteratively construct our baseline starting from a STARFlow-L model trained on a fixed VAE latent space. The refinement process involves: (1) removing the softplus operation on the scaling factors to improve numerical flexibility; (2) incorporating gated attention [30] mechanisms to enhance feature interaction; and (3) transitioning to full end-to-end (e2e) training with GAN loss, and without auxiliary loss. Relative to SimFlow-L, this baseline changes only the NF backbone, which makes the close FID (3.70 vs. 3.72) an apples-to-apples check. The performance gains from each stage are summarized in Tab. 1. For optimization, we use the AdamW optimizer with a learning rate of $1 \times 1 0 ^ { - 4 }$ and a global batch size of 256. The final model is trained end-to-end for 90 epochs, followed by 2 additional epochs of decoder fine-tuning.

Transformer-based Autoencoder To accommodate the requirements of MIM, we replace the traditional convolutional stages of the SD-VAE<sup>3</sup> [34] with Transformer-based MAE-Tok [1]. Notably, we train the entire framework from scratch without loading any pretrained parameters for the MAETok backbone. Our latent space is configured with 128 tokens, each having a dimensionality of 64, and is integrated with 1D positional encodings. During training, we adopt a random masking strategy to facilitate the MIM objective, with a default mask ratio ranging from 0.4 to 0.6. As shown in Tab. 2, while the ViT-B backbone increases the parameter count, it significantly reduces the computational overhead in terms of FLOPs compared to the convolutional VAE, making it more suitable for high-resolution processing

Table 2: Comparison of Architectural Complexity

<table><tr><td>Model</td><td>Parameters</td><td>FLOPs</td></tr><tr><td>SD-VAE</td><td>83.7M</td><td>446.21G</td></tr><tr><td>MAETok</td><td>172.0M</td><td>71.25G</td></tr></table>

## 4.2 Main Results

We compare MIMFlow-L with state-of-the-art (SOTA) generative models on the ImageNet 256 × 256 benchmark, including Pixel-space models, Latent Autoregressive (AR) models, Latent Difusion Models (LDMs), and existing Latent Normalizing Flows. The quantitative results are summarized in Table 3. Besides, qualitative results are shown in Fig. 3, where MIMFlow-L produces high-fidelity images with consistent global structures.

Superiority within the NF Paradigm. MIMFlow-L demonstrates a significant performance leap over existing Normalizing Flow baselines. Compared to the closely related SimFlow-L, which shares a similar parameter count (∼480M), MIMFlow-L improves the FID from 3.72 to 2.50—a 32.8% reduction. Notably, our model with only 482M parameters outperforms much larger models like FAE-NF-XXL (FID 2.67), which utilizes nearly 3 × the parameters (1.4B) and is built upon the advanced RAE latent space. It also closely approaches the performance of STARFlow-XXL (FID 2.40). This suggests that integrating MIM allows the flow model to learn a more eficient and structured semantic manifold, extracting higher generative value from the same parameter budget.

![](images/a3f7b4b41bef0b0b7292134d528ccd574d4e0d086017b4a7fc241d33240962b2.jpg)  
Fig. 3: Selected Samples on ImageNet 256 × 256 from MIMFlow-L. We use classifierfree guidance equal to 2.0.

Eficiency via Token Compression. A key highlight of MIMFlow is its token eficiency. While most latent models (e.g., DiT, LDM, SimFlow) operate on a 16 × 16 = 256 token grid, and some even require 1024 tokens (STARFlow, FlowBack), MIMFlow-L achieves strong NF results using only 128 tokens for sampling.

1. Reduced Computational Complexity: By halving the sequence length compared to standard VAE-based models, the computational overhead of the Normalizing Flow (which typically scales quadratically or involves deep Transformer blocks) is significantly reduced.

2. Information Density: This 128-token bottleneck supports our hypothesis in Sec. 3.1: because the NF is less exposed to high-frequency noise through MIM, it can represent the global image structure more compactly. This allows MIMFlow to maintain high precision (0.82) while operating at a fraction of the sequence length used by many competitors. Appendix Table 9 further reports the latency benefit of reducing the token count.

Impact of Guidance and Semantic Quality. As shown in Tab. 3, the gap between “w/o guidance” and “w/ guidance” performance is notably narrower for MIMFlow compared to early flow models. Without guidance, MIMFlow-L achieves an FID of 3.64, which is significantly better than the 10.13 reported for SimFlow-XXL. This indicates that the latent space learned through joint MIM and NF optimization is inherently more structured and semantically coherent, requiring less external guidance to produce high-quality samples.

Comparison with Difusion and AR Models. While Latent Difusion Models (like DiT and REPA) currently lead in FID, MIMFlow-L narrows the gap between ODE-based Normalizing Flows and these heavy-duty generative frameworks. Unlike difusion models that require hundreds of denoising steps or complex scheduling, our flow-based approach ofers exact density estimation and a deterministic mapping in a single continuous ODE trajectory. The competitive Precision (0.82) of MIMFlow-L underscores its ability to generate high-fidelity samples that faithfully respect the learned data manifold.

Table 3: System performance comparison on ImageNet 256 × 256 classconditioned generation. Gray rows denote larger-scale models within the latent normalizing flow category, provided for reference.

<table><tr><td rowspan="2">Method</td><td rowspan="2">#Tokens</td><td rowspan="2">#Params</td><td colspan="4">W/o guidance</td><td colspan="4">W/ guidance</td></tr><tr><td>gFID↓</td><td>IS↑</td><td>Prec.↑</td><td>Rec.↑</td><td>gFID↓</td><td>IS↑</td><td>Prec.↑</td><td>Rec.↑</td></tr><tr><td colspan="11">Pixel Space</td></tr><tr><td>ADM [6]</td><td>-</td><td>554M</td><td>10.94</td><td>101.0</td><td>0.69</td><td>0.63</td><td>3.94</td><td>215.8</td><td>0.83</td><td>0.53</td></tr><tr><td>RIN [20]</td><td>-</td><td>410M</td><td>3.42</td><td>182.0</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PixelFlow [3]</td><td>4096</td><td>677M</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.98</td><td>282.1</td><td>0.81</td><td>0.60</td></tr><tr><td>PixNerd [41]</td><td>1024</td><td>700M</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.15</td><td>297.0</td><td>0.79</td><td>0.59</td></tr><tr><td>SiD2 [19]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.38</td><td>-</td><td>-</td><td>-</td></tr><tr><td>TARFlow [48]</td><td>1024</td><td>1.4B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>4.69</td><td>-</td><td>-</td><td>-</td></tr><tr><td>JetFormer [39]</td><td>256</td><td>2.8B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>6.64</td><td>-</td><td>0.69</td><td>0.56</td></tr><tr><td>FARMER [51]</td><td>1024</td><td>1.9B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>3.60</td><td>269.2</td><td>0.81</td><td>0.51</td></tr><tr><td colspan="11">Latent Autoregressive</td></tr><tr><td>VAR [38]</td><td>680</td><td>2.0B</td><td>1.92</td><td>323.1</td><td>0.82</td><td>0.59</td><td>1.73</td><td>350.2</td><td>0.82</td><td>0.60</td></tr><tr><td>MAR [25]</td><td>256</td><td>943M</td><td>2.35</td><td>227.8</td><td>0.79</td><td>0.62</td><td>1.55</td><td>303.7</td><td>0.81</td><td>0.62</td></tr><tr><td>xAR [32]</td><td>-</td><td>1.1B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.24</td><td>301.6</td><td>0.83</td><td>0.64</td></tr><tr><td colspan="11">Latent Diffusion</td></tr><tr><td>DiT [29]</td><td>256</td><td>675M</td><td>9.62</td><td>121.5</td><td>0.67</td><td>0.67</td><td>2.27</td><td>278.2</td><td>0.83</td><td>0.57</td></tr><tr><td>MaskDiT [52]</td><td>-</td><td>675M</td><td>5.69</td><td>177.9</td><td>0.74</td><td>0.60</td><td>2.28</td><td>276.6</td><td>0.80</td><td>0.61</td></tr><tr><td>SiT [26]</td><td>256</td><td>675M</td><td>8.61</td><td>131.7</td><td>0.68</td><td>0.67</td><td>2.06</td><td>270.3</td><td>0.82</td><td>0.59</td></tr><tr><td>MDTv2 [11]</td><td>256</td><td>675M</td><td>-</td><td>-</td><td>-</td><td>-</td><td>1.58</td><td>314.7</td><td>0.79</td><td>0.65</td></tr><tr><td>REPA [47]</td><td>256</td><td>675M</td><td>5.78</td><td>158.3</td><td>0.70</td><td>0.68</td><td>1.29</td><td>306.3</td><td>0.79</td><td>0.64</td></tr><tr><td>VA-VAE [46]</td><td>256</td><td>675M</td><td>2.17</td><td>205.6</td><td>0.77</td><td>0.65</td><td>1.35</td><td>295.3</td><td>0.79</td><td>0.65</td></tr><tr><td>DDT [42]</td><td>256</td><td>675M</td><td>6.27</td><td>154.7</td><td>0.68</td><td>0.69</td><td>1.26</td><td>310.6</td><td>0.79</td><td>0.65</td></tr><tr><td>REPA-E [24]</td><td>256</td><td>675M</td><td>1.69</td><td>219.3</td><td>0.77</td><td>0.67</td><td>1.12</td><td>302.9</td><td>0.79</td><td>0.66</td></tr><tr><td>RAE [50]</td><td>256</td><td>839M</td><td>1.51</td><td>242.9</td><td>0.79</td><td>0.63</td><td>1.13</td><td>262.6</td><td>0.78</td><td>0.67</td></tr><tr><td colspan="11">Latent Normalizing Flows</td></tr><tr><td>FlowBack-XL [4]</td><td>1024</td><td>831M</td><td>-</td><td>-</td><td>-</td><td>-</td><td>4.18</td><td>240.8</td><td>-</td><td>-</td></tr><tr><td>STARFlow-XXL [14]</td><td>1024</td><td>1.4B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.40</td><td>-</td><td>-</td><td>-</td></tr><tr><td>FAE-NF-XXL [12]</td><td>256</td><td>1.4B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>2.67</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SimFlow-XXL [49]</td><td>256</td><td>1.4B</td><td>10.13</td><td>124.7</td><td>0.71</td><td>0.61</td><td>1.91</td><td>284.4</td><td>0.82</td><td>0.60</td></tr><tr><td>SimFlow-L [49]</td><td>256</td><td>475M</td><td>33.53</td><td>-</td><td>-</td><td>-</td><td>3.72</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Baseline (Ours)</td><td>256</td><td>482M</td><td>-</td><td>-</td><td>-</td><td>-</td><td>3.70</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MIMFlow-L (Ours)</td><td>128</td><td>482M</td><td>3.64</td><td>158.6</td><td>0.78</td><td>0.60</td><td>2.50</td><td>233.5</td><td>0.82</td><td>0.57</td></tr></table>

In summary, the results demonstrate that by decoupling semantic modeling from texture synthesis, MIMFlow-L mitigates a capacity bottleneck in traditional NFs and provides an eficient path for improving flow-based image generation.

## 4.3 Ablation Study

To validate the efectiveness of the proposed components in MIMFlow, we conduct a series of ablation experiments on the ImageNet 256 × 256 benchmark. To ensure a fair comparison, all models in this section are trained end-to-end

Table 4: Ablation studies on ImageNet 256×256. Bold indicates the default configuration. Acc. denotes linear probing accuracy on the encoder’s representations before the projection layer. Mix in Tab. 4a represents a 50/50 probability of using None and 0.4–0.6 masking ratios. 10K images are sampled for evaluation.

(a) Masking Strategies

<table><tr><td>Ratio</td><td>rFID↓</td><td>gFID↓</td><td>IS↑</td><td>Prec.↑</td><td>Rec.↑</td><td>Acc.↑</td></tr><tr><td>None</td><td>23.5</td><td>29.0</td><td>65.5</td><td>0.59</td><td>0.56</td><td>56.6</td></tr><tr><td>0.2–0.4</td><td>8.66</td><td>24.47</td><td>60.1</td><td>0.59</td><td>0.57</td><td>54.2</td></tr><tr><td>0.4–0.6</td><td>3.40</td><td>12.82</td><td>88.6</td><td>0.70</td><td>0.66</td><td>71.3</td></tr><tr><td>0.6–0.8</td><td>5.26</td><td>15.92</td><td>79.8</td><td>0.65</td><td>0.65</td><td>65.9</td></tr><tr><td>Mix</td><td>6.58</td><td>26.98</td><td>51.9</td><td>0.58</td><td>0.55</td><td>45.7</td></tr></table>

(b) Auxiliary Loss Targets

<table><tr><td>Target</td><td>rFID↓</td><td>gFID↓</td><td>IS↑</td><td>Prec.↑</td><td>Rec.↑</td></tr><tr><td>DINO</td><td>4.00</td><td>12.89</td><td>91.0</td><td>0.70</td><td>0.66</td></tr><tr><td>D+HOG</td><td colspan="5">Training Collapsed</td></tr><tr><td>D+CLIP</td><td>3.60</td><td>12.46</td><td>92.3</td><td>0.70</td><td>0.66</td></tr><tr><td>C+HOG</td><td>4.18</td><td>20.77</td><td>56.0</td><td>0.63</td><td>0.59</td></tr><tr><td>All</td><td>3.64</td><td>12.82</td><td>88.6</td><td>0.70</td><td>0.66</td></tr></table>

(d) Latent Noise Scale (σ)

(c) Token Number (K)

<table><tr><td>K</td><td>rFID↓</td><td>gFID↓</td><td>IS↑</td><td>Prec.↑</td><td>Rec.↑</td></tr><tr><td>64</td><td>5.61</td><td>12.78</td><td>91.8</td><td>0.71</td><td>0.65</td></tr><tr><td>128</td><td>3.60</td><td>12.46</td><td>92.3</td><td>0.70</td><td>0.66</td></tr><tr><td>192</td><td>24.59</td><td>30.42</td><td>68.0</td><td>0.57</td><td>0.57</td></tr></table>

<table><tr><td>σ</td><td>rFID↓</td><td>gFID↓</td><td>IS↑</td><td>Prec.↑</td><td>Rec.↑</td></tr><tr><td>0.2</td><td>3.93</td><td>14.30</td><td>81.5</td><td>0.66</td><td>0.66</td></tr><tr><td>0.3</td><td>3.40</td><td>12.82</td><td>88.6</td><td>0.70</td><td>0.66</td></tr><tr><td>0.5</td><td>6.95</td><td>17.69</td><td>75.6</td><td>0.65</td><td>0.62</td></tr></table>

for 50 epochs. Unless otherwise specified, the reported generative metrics are calculated using 10K samples (50K in Tab. 3).

The Necessity of Masked Decoupling. Table 4a demonstrates the critical role of the masking strategy under the same end-to-end training setup. Without masking (nomask), the model exhibits the poorest performance (gFID 29.0), and its latent semantic quality (Acc. 56.6%) is significantly lower than that of masked versions. This supports our hypothesis that unmasked training leaves the Normalizing Flow exposed to redundant pixel-level details. A mask ratio of 0.4–0.6 achieves the best balance, significantly enhancing both generative quality (gFID 12.82) and semantic representation (Acc. 71.3%). Interestingly, a mixed strategy or low mask ratios lead to performance degradation, suggesting that a consistent and substantial information bottleneck is important for stabilizing the semantic manifold.

Optimal Bottleneck Size. As shown in Tab. 4c, the number of latent tokens K serves as a physical bottleneck for information compression. While 64 tokens provide reasonable performance, increasing to 128 tokens yields the best reconstruction (rFID 3.60) and generation results. This observation aligns with the information density provided by our masking strategy: since a 256 × 256 image is partitioned into 16 × 16 = 256 patches and approximately 50% of the patches are masked during training, the remaining information can be optimally represented by a compressed latent space of 128 tokens. However, expanding to 192 tokens leads to a sharp performance drop (gFID 30.42). This indicates that an excessively large latent space allows high-frequency noise to leak into the flow model, thereby violating the principled decoupling and hindering the NF’s ability to model global structure.

![](images/29f34100e1ef35e9a61db4a3c12827aa84488f08df13695d75d5feeb5d7020f1.jpg)  
(a) SD-VAE

![](images/1885f779cecc1aa35305aca4c341889ca3da8ed3a327b8ae97d3ff72d2122747.jpg)  
(b) MIMFlow  
Fig. 4: UMAP visualization on ImageNet of the learned latent space from (a) SD-VAE; (b) MIMFlow. Colors indicate diferent classes. MIMFlow presents a more discriminative latent space.

Synergy of Auxiliary Semantic Priors. We investigate various auxiliary supervision signals (DINO, CLIP, HOG) in Tab. 4b. The combination of DINO and CLIP features achieves the superior performance, as they provide complementary structural and semantic guidance. In contrast, incorporating low-level features like HOG either leads to training collapse or suboptimal results. This validates that the MIMFlow latent space is inherently optimized for high-level semantic abstractions rather than local gradient textures.

Impact of Latent Stochasticity. The additive Gaussian noise σ is a crucial trick for NF training. As shown in Tab. 4d, σ = 0.3 is the optimal value. Smaller values (σ = 0.2) fail to suficiently smoothen the manifold, while larger values $( \sigma = 0 . 5 )$ introduce excessive variance that compromises reconstruction fidelity (rFID 6.95), confirming that a precise calibration of latent stochasticity is vital for end-to-end flow modeling.

## 4.4 Analysis of Latent Space and Flow Dynamics

In this section, we further investigate why the proposed MIMFlow framework facilitates more efective generative modeling by analyzing the properties of the learned latent space and the flow transformations.

Semantic Discriminability. A core motivation of MIMFlow is to decouple high-frequency pixel noise from global structural semantics. We visualize the latent space z using UMAP projection in Fig. 4. Compared to the relatively entangled latent space of a standard SD-VAE, the MIMFlow latent space exhibits significantly clearer clustering corresponding to ImageNet categories. This enhanced discriminative power is quantitatively supported by the linear probing results in Tab. 4a: our masked approach achieves a classification accuracy of 71.3%, a substantial improvement over the 56.6% of the unmasked baseline.

![](images/7c4e92a10204312f1e45d31d8152df868a846c8e017fed45a137550858443e59.jpg)

![](images/7e07d1e6921452f0e8e95e06cb98fe093028ec162e2a593462c4e8c75b7bd925.jpg)

![](images/2e5909bfa2d6bd8b308006e326b25567fc739100fa4a59b65d9e688f670a7da0.jpg)  
Fig. 5: Jacobian Spectral Analysis of STARFlow and MIMFlow. The three panels report, from left to right, the empirical distributions of the largest singular value $\sigma _ { \operatorname* { m a x } } ( J )$ , the smallest singular value $\sigma _ { \mathrm { m i n } } ( J )$ , and the log-condition number log $_ { 1 0 } \kappa ( J )$ (with $\kappa ( J ) = \sigma _ { \mathrm { m a x } } ( J ) / \sigma _ { \mathrm { m i n } } ( J ) )$ .

These results confirm that the masking bottleneck efectively forces the latent manifold to prioritize high-level semantic coherence over local redundancies.

Jacobian Spectral Analysis. To assess the numerical stability and bijectivity of the learned Normalizing Flow, we analyze the spectral properties of the Jacobian $J = \partial f _ { \theta } / \partial \mathbf { z }$ for both STARFlow and MIMFlow (Fig. 5). The analysis reveals several key insights:

1. Enhanced Bijectivity: MIMFlow exhibits a larger and more stable minimum singular value $\left( \sigma _ { \mathrm { m i n } } \right)$ , keeping the Jacobian further from singularity and reducing the risk of ill-conditioned mappings.

2. Superior Numerical Stability: Compared to STARFlow, MIMFlow achieves a lower and more concentrated log-condition number. A lower condition number indicates a more well-conditioned transformation that is less sensitive to numerical perturbations.

These spectral properties suggest that by alleviating the burden on the NF to capture pixel-level details, MIMFlow learns a smoother flow field with less extreme spatial warping, which is consistent with more stable optimization and higher generative fidelity.

## 5 Conclusion

In this paper, we presented MIMFlow, a unified end-to-end generative framework that integrates Masked Image Modeling with Normalizing Flows. By introducing a masked bottleneck with learnable tokens, we achieve a principled decoupling of generative tasks: the Normalizing Flow focuses on modeling the lowfrequency semantic manifold, while a specialized decoder handles high-frequency texture synthesis. This design mitigates the capacity bottleneck of traditional NFs, which often spend representational power on redundant pixel-level noise.

Empirical results on ImageNet 256×256 demonstrate that MIMFlow-L improves similar-scale NF baselines, achieving an FID of 2.50. Notably, our model achieves this with only 128 tokens–a 50% reduction compared to standard latent generative models–while maintaining high semantic discriminability, as evidenced by a 71.3% linear probing accuracy. Our analysis of the Jacobian spectrum further suggests that this decoupling leads to better-conditioned flow transformations. Ultimately, MIMFlow provides a new perspective on unifying self-supervised representation learning and probabilistic modeling, paving the way for more eficient and semantically-aware generative systems.

## Acknowledgements

This work is supported by by the Basic Research Program of Jiangsu (No. BK20250009), the Fundamental Research Funds for the Central Universities (No.020214380140), the Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM118), the Collaborative Innovation Center of Novel Software Technology and Industrialization, Alibaba Group through Alibaba Innovative Research Program.

## A Implementation Details

Detailed architectural specifications and training hyperparameters are summarized in Table 5 and Table 6, respectively. Regarding the optimization objectives, the pixel-level MSE loss is computed over the entire image to ensure global reconstruction fidelity, whereas the auxiliary semantic loss is restricted to the masked patches to enforce the model’s ability to infer missing structural context. For the adversarial refinement stage, we adopt the GAN loss formulation and discriminator architecture following RAE [50], with the distinction that our decoder and discriminator are optimized simultaneously from the beginning of the fine-tuning phase. During inference, we use the Classifier-Free Guidance (CFG) strategy designed by TARFlow [48] exclusively within the final deep block. We strictly follow the setup of ADM [6] for data augmentation and evaluation.

## B Spectral Analysis of the Jacobian

To characterize the geometric properties and the stability of the Normalizing Flow (NF) mapping $f : \mathcal { X } \ :  \ : \mathcal { Z } .$ we analyze the spectral properties of the Jacobian matrix $J ( \mathbf { x } ) = \nabla _ { \mathbf { x } } { f } ( \mathbf { x } )$ in Sec. 4.4. Due to the high dimensionality of the data space, explicitly computing or storing the $D \times D$ Jacobian matrix is computationally intractable. Instead, we employ matrix-free iterative methods based on Automatic Diferentiation (AD) primitives to estimate the extreme singular values.

Table 5: Detailed Architecture Configurations of MIMFlow-L. K denotes the number of latent tokens, and D denotes the latent dimensionality.

<table><tr><td>Module</td><td>Hyperparameter</td><td>Value</td></tr><tr><td rowspan="5">Masked Encoder  $E_{\phi}$ </td><td>Backbone</td><td>ViT-B</td></tr><tr><td>Patch Size</td><td>16 × 16</td></tr><tr><td>Layers / Hidden Dim / Heads</td><td>12 / 768 / 12</td></tr><tr><td>Input Resolution</td><td>256 × 256</td></tr><tr><td>Masking Ratio</td><td>0.4 – 0.6</td></tr><tr><td rowspan="4">Latent Space</td><td>Number of Latent Tokens (K)</td><td>128</td></tr><tr><td>Latent Dimension (D)</td><td>64</td></tr><tr><td>Positional Encoding</td><td>1D Learnable</td></tr><tr><td>Latent Noise Scale (σ)</td><td>0.3</td></tr><tr><td rowspan="3">Normalizing Flow  $f_{\theta}$ </td><td>Architecture</td><td>Improved-STARFlow-L</td></tr><tr><td>Blocks / Layers</td><td>8 / 2×7+20</td></tr><tr><td>Hidden Dimension</td><td>1024</td></tr><tr><td rowspan="3">Generative Decoder  $D_{\psi}$ </td><td>Backbone</td><td>ViT-B</td></tr><tr><td>Layers / Hidden Dim / Heads</td><td>12 / 768 / 12</td></tr><tr><td>Output Resolution</td><td>256 × 256</td></tr></table>

## B.1 Estimation of the Maximum Singular Value $\left( \sigma _ { \mathbf { m a x } } \right)$

The spectral norm, or the maximum singular value $\sigma _ { \operatorname* { m a x } } ( J )$ , is computed using the Power Iteration method applied to the symmetric positive semi-definite operator $J ^ { T } J$ . The algorithm avoids explicit matrix construction by leveraging Jacobian-Vector Products (JVP) and Vector-Jacobian Products (VJP):

$$
\mathbf {v} _ {k + 1} = \frac {J ^ {T} (J \mathbf {v} _ {k})}{\| J ^ {T} (J \mathbf {v} _ {k}) \|},\tag{9}
$$

where $J { \bf v } _ { k }$ is computed via a JVP, and the subsequent multiplication by $J ^ { T }$ is performed via a VJP. Upon convergence, the maximum singular value is obtained as $\sigma _ { \operatorname* { m a x } } ( J ) = \| J \mathbf { v } \|$

## B.2 Estimation of the Minimum Singular Value $( \sigma _ { \mathrm { m i n } } )$

To estimate the minimum singular value, we implement the Inexact Shifted Inverse Iteration. We consider the regularized operator $A = J ^ { T } J + \alpha I$ , where $\alpha > 0$ is a small numerical shift (Tikhonov regularization) introduced to ensure strict positive definiteness and numerical stability. In each iteration, we solve the linear system:

$$
(J ^ {T} J + \alpha I) \mathbf {w} = \mathbf {v} _ {k},\tag{10}
$$

for w using the Conjugate Gradient (CG) algorithm. This approach is “matrixfree” as CG only requires the evaluation of the operator-vector product $A \mathbf { v } ,$ which is eficiently computed using the JVP-VJP sequence. The smallest singular value is then recovered from the converged eigenvalue $\lambda _ { \mathrm { m i n } }$ of A:

Table 6: Training Hyperparameters. Phase 1 is the joint optimization of VAE and NF; Phase 2 is the adversarial refinement of the decoder.

<table><tr><td colspan="3">Hyperparameter Phase 1: Joint Training Phase 2: Decoder Fine-tuning</td></tr><tr><td>Total Epochs</td><td>90</td><td>2</td></tr><tr><td>Batch Size</td><td>256</td><td>256</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>EMA update</td><td>0.9999</td><td>0.9999</td></tr><tr><td>Learning Rate</td><td> $1 \times 10^{-4}$ </td><td> $1 \times 10^{-4}$ </td></tr><tr><td>LR Schedule</td><td>Constant</td><td>Constant</td></tr><tr><td>Weight Decay</td><td> $1 \times 10^{-4}$ </td><td> $1 \times 10^{-4}$ </td></tr><tr><td>Adam  $(\beta_1, \beta_2)$ </td><td>(0.9, 0.95)</td><td>(0.9, 0.95)</td></tr><tr><td colspan="3">Loss Weights</td></tr><tr><td>Flow Loss  $(\beta)$ </td><td>1.0</td><td>-</td></tr><tr><td>Reconstruction  $(\ell_2)$ </td><td>1.0</td><td>1.0</td></tr><tr><td>Perceptual (LPIPS)</td><td>1.1</td><td>1.1</td></tr><tr><td>GAN Loss  $(\alpha)$ </td><td>-</td><td>0.05</td></tr><tr><td>Hardware</td><td colspan="2">8 × NVIDIA A100 (80GB)</td></tr></table>

$$
\sigma_ {\mathrm{min}} (J) = \sqrt {\max (\lambda_ {\mathrm{min}} - \alpha , 0)}.\tag{11}
$$

The local conditioning of the NF is subsequently evaluated via the condition number $\kappa ( J ) = \sigma _ { \mathrm { m a x } } / \sigma _ { \mathrm { m i n } }$

## C Extended Experimental Results

Table 7: Progressive ablation on ImageNet 256 × 256 with 10K samples, evaluated without CFG.

<table><tr><td>Configuration</td><td>rFID↓</td><td>gFID↓</td><td>IS↑</td><td>Prec.↑</td><td>Rec.↑</td></tr><tr><td>Baseline (e2e, SD-VAE, 256tok, GAN)</td><td>2.74</td><td>11.24</td><td>86.0</td><td>0.68</td><td>0.63</td></tr><tr><td>+ Learnable Tokens, - GAN Loss</td><td>4.15</td><td>19.52</td><td>63.1</td><td>0.52</td><td>0.66</td></tr><tr><td>+ Aux Loss (DINO+CLIP)</td><td>14.75</td><td>18.71</td><td>91.0</td><td>0.65</td><td>0.62</td></tr><tr><td>+ Masking (0.4–0.6)</td><td>3.81</td><td>10.14</td><td>105.2</td><td>0.64</td><td>0.66</td></tr><tr><td>+ GAN FT (Full MIMFlow)</td><td>1.47</td><td>6.15</td><td>130.0</td><td>0.77</td><td>0.71</td></tr></table>

## C.1 Progressive ablation.

As shown in Tab. 7, this ablation progressively isolates the contribution of each component under a 10K-sample evaluation without CFG. Starting from the endto-end SD-VAE baseline with 256 tokens and GAN training, replacing the latent interface with learnable tokens while removing the GAN loss degrades gFID from 11.24 to 19.52, indicating that learnable tokens alone do not explain the final gain. Adding DINO+CLIP auxiliary supervision improves semantic confidence as reflected by IS, but only marginally changes gFID (19.52→18.71). The decisive improvement appears when the 0.4–0.6 masking bottleneck is introduced, reducing gFID by 46% (18.71→10.14) and surpassing the GAN-trained baseline. This supports our claim that masking provides the primary structural constraint for simplifying the latent distribution modeled by the NF, rather than the gain coming mainly from auxiliary supervision. The final GAN fine-tuning stage further recovers low-level fidelity and texture detail, improving rFID from 3.81 to 1.47 and gFID from 10.14 to 6.15.

Table 8: Ablation of MIM weight

<table><tr><td>MIM</td><td>rFID↓</td><td>gFID↓</td><td>IS↑</td><td>Prec.↑</td><td>Rec.↑</td></tr><tr><td>1</td><td>3.60</td><td>12.46</td><td>92.3</td><td>0.70</td><td>0.66</td></tr><tr><td>5</td><td>4.10</td><td>13.33</td><td>85.6</td><td>0.69</td><td>0.64</td></tr><tr><td>10</td><td>13.73</td><td>26.19</td><td>61.8</td><td>0.46</td><td>0.55</td></tr></table>

Table 9: Eficiency Analysis

<table><tr><td>Token</td><td>Train(ms/iter)</td><td>Inference(s)</td></tr><tr><td>128</td><td>183</td><td>1.60</td></tr><tr><td>256</td><td>352</td><td>3.64</td></tr><tr><td>1024</td><td>1515</td><td>25.3</td></tr></table>

Table 10: Hardware eficiency comparison on ImageNet $2 5 6 \times 2 5 6$ using 8×H800 GPUs. We use no auxiliary supervision in this comparison; the per-GPU batch sizes are 16 for training and 256 for inference.

<table><tr><td>Model</td><td>Tokens</td><td>Params</td><td>Train Mem.</td><td>Train Speed</td><td>Sample / img</td></tr><tr><td>SimFlow-L</td><td>256</td><td>475M</td><td>52.3GB</td><td>2.83 it/s</td><td>0.020s</td></tr><tr><td>MIMFlow-L</td><td>128</td><td>482M</td><td>37.6GB</td><td>3.11 it/s</td><td>0.011s</td></tr></table>

## C.2 Ablation on Masked Reconstruction Weight

In standard Masked Image Modeling (MIM), such as MAE [17], the reconstruction loss is typically computed only on the masked patches. However, for generative modeling, maintaining global pixel-level fidelity is crucial for high-quality synthesis. To balance these objectives, we investigate the impact of the MIM weight, which scales the reconstruction loss of the masked regions relative to the unmasked ones (the latter fixed at a weight of 1).

As shown in Tab. 8, the optimal performance is achieved when the MIM weight is set to 1, efectively treating masked and unmasked regions with equal importance during joint optimization. Increasing the weight to 5 or 10 leads to a noticeable degradation in both reconstruction (rFID) and generation (gFID) metrics. This performance drop suggests that over-weighting the masked regions may bias the model toward local patch-filling at the expense of global structural coherence, thereby distorting the learned semantic manifold.

Linear Probe Accuracy vs Depth under Different Mask Ratios

![](images/4c5f94e4a39cb83e2172a1874d494ca67e743fb9a1b1ad1d9119dd8451d409a9.jpg)  
Fig. 6: Linear Probe Accuracy vs Depth under Diferent Mask Ratios.

## C.3 Eficiency Analysis

A key advantage of our MIMFlow is its high eficiency, achieved through a significantly reduced token budget. While existing methods typically rely on 256 or even 1024 tokens to represent sequences, our approach operates efectively with only 128 tokens. As demonstrated in Table 9, we first isolate the latency efect of token count using the Improved STARFlow backbone. On an H20 GPU, the 128-token setting achieves a training latency of 183ms per iteration (batch size 32) and an inference latency of 1.60s per sample. Compared to the 1024-token setting, it provides an 8.3× speedup in training and a 15.8× acceleration in inference.

We further compare end-to-end L-scale models under the same 8×H800 setting in Tab. 10. Although MIMFlow-L has a slightly larger parameter count than SimFlow-L (482M vs. 475M), its 128-token latent reduces training memory from 52.3GB to 37.6GB, improves throughput from 2.83 to 3.11 iterations per second, and reduces per-image sampling time from 0.020s to 0.011s. This corresponds to a 28% memory reduction, a 10% throughput improvement, and nearly halved sampling time; the lower memory footprint also makes training feasible on 8×A100 GPUs with a total batch size of 256. Since auxiliary supervision is disabled in this comparison, the measured gain mainly reflects the computational benefit of the compact masked latent rather than additional training objectives.

## C.4 Semantic Evolution across Flow Depth

To investigate the feature abstraction capabilities of Normalizing Flows, we conduct linear probing on the intermediate representations at various depths of the

NF. As illustrated in Fig. 6, while the encoder’s latent space exhibits high classification accuracy (facilitated by the MIM objective), this accuracy does not increase—and occasionally plateaus or slightly declines—as the depth of the NF increases.

This observation reveals a fundamental characteristic of Normalizing Flows: while they excel at complex distribution warping via bijective mappings, they lack the inherent ability to perform further hierarchical feature abstraction or semantic distillation. This finding underscores the necessity of our MIMFlow paradigm, which delegates the burden of semantic modeling to the encoder through Masked Image Modeling, allowing the subsequent flow blocks to focus purely on probabilistic density estimation within a well-structured manifold.

## References

1. Chen, H., Han, Y., Chen, F., Li, X., Wang, Y., Wang, J., Wang, Z., Liu, Z., Zou, D., Raj, B.: Masked autoencoders are efective tokenizers for difusion models (2025), https://arxiv.org/abs/2502.03444

2. Chen, R.T.Q., Rubanova, Y., Bettencourt, J., Duvenaud, D.: Neural ordinary differential equations (2019), https://arxiv.org/abs/1806.07366

3. Chen, S., Ge, C., Zhang, S., Sun, P., Luo, P.: Pixelflow: Pixel-space generative models with flow. arXiv preprint arXiv:2504.07963 (2025)

4. Chen, Y., Xu, X., Wang, S., Zhu, C., Wen, R., Li, X., Ge, T., Wang, L.: Flowing backwards: Improving normalizing flows via reverse representation alignment. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 3074– 3082 (2026)

5. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet: A Large-scale Hierarchical Image Database. IEEE Conference on Computer Vision and Pattern Recognition pp. 248–255 (2009)

6. Dhariwal, P., Nichol, A.: Difusion models beat gans on image synthesis. Advances in Neural Information Processing Systems 34, 8780–8794 (2021)

7. Dinh, L., Krueger, D., Bengio, Y.: Nice: Non-linear independent components estimation. arXiv preprint arXiv:1410.8516 (2014)

8. Dinh, L., Sohl-Dickstein, J., Bengio, S.: Density estimation using real nvp. arXiv preprint arXiv:1605.08803 (2016)

9. Draxler, F., Sorrenson, P., Zimmermann, L., Rousselot, A., Köthe, U.: Free-form flows: Make any architecture a normalizing flow. In: International Conference on Artificial Intelligence and Statistics. pp. 2197–2205. PMLR (2024)

10. Draxler, F., Wahl, S., Schnörr, C., Köthe, U.: On the universality of volumepreserving and coupling-based normalizing flows. arXiv preprint arXiv:2402.06578 (2024)

11. Gao, S., Zhou, P., Cheng, M.M., Yan, S.: Masked difusion transformer is a strong image synthesizer. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 23164–23173 (2023)

12. Gao, Y., Chen, C., Chen, T., Gu, J.: One layer is enough: Adapting pretrained visual encoders for image generation (2025), https://arxiv.org/abs/2512.07829

13. Giaquinto, R., Banerjee, A.: Gradient boosted normalizing flows. Advances in Neu ral Information Processing Systems 33, 22104–22117 (2020)

14. Gu, J., Chen, T., Berthelot, D., Zheng, H., Wang, Y., Zhang, R., Dinh, L., Bautista, M.A., Susskind, J., Zhai, S.: Starflow: Scaling latent normalizing flows for highresolution image synthesis. arXiv preprint arXiv:2506.06276 (2025)

15. Gu, J., Chen, T., Shen, Y., Berthelot, D., Zhai, S., Susskind, J.: Normalizing trajectory models (2026), https://arxiv.org/abs/2605.08078

16. Gu, J., Shen, Y., Chen, T., Dinh, L., Wang, Y., Bautista, M.A., Berthelot, D., Susskind, J., Zhai, S.: Starflow-v: End-to-end video generative modeling with autoregressive normalizing flows. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9084–9094 (2026)

17. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., Girshick, R.: Masked autoencoders are scalable vision learners (2021), https://arxiv.org/abs/2111.06377

18. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems 30 (2017)

19. Hoogeboom, E., Mensink, T., Heek, J., Lamerigts, K., Gao, R., Salimans, T.: Simpler difusion (sid2): 1.5 fid on imagenet512 with pixel-space difusion. arXiv preprint arXiv:2410.19324 (2024)

20. Jabri, A., Fleet, D., Chen, T.: Scalable adaptive computation for iterative generation. arXiv preprint arXiv:2212.11972 (2022)

21. Kingma, D.P., Dhariwal, P.: Glow: Generative flow with invertible 1x1 convolutions (2018), https://arxiv.org/abs/1807.03039

22. Kobyzev, I., Prince, S.J., Brubaker, M.A.: Normalizing flows: An introduction and review of current methods. IEEE transactions on pattern analysis and machine intelligence 43(11), 3964–3979 (2020)

23. Kynkäänniemi, T., Karras, T., Laine, S., Lehtinen, J., Aila, T.: Improved precision and recall metric for assessing generative models. Advances in Neural Information Processing Systems 32 (2019)

24. Lee, S.H., Park, S., Kim, G.M.: REPA-E: End-to-end training of latent-difusion models via representation alignment. In: arXiv preprint arXiv:2405.18373 (2024)

25. Li, T., Tian, Y., Li, H., Deng, M., He, K.: Autoregressive image generation without vector quantization. Advances in Neural Information Processing Systems 37, 56424–56445 (2024)

26. Ma, N., Goldstein, M., Albergo, M.S., Bofi, N.M., Vanden-Eijnden, E., Xie, S.: Sit: Exploring flow and difusion-based generative models with scalable interpolant transformers. arXiv preprint arXiv:2401.08740 (2024)

27. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023)

28. Papamakarios, G., Nalisnick, E., Rezende, D.J., Mohamed, S., Lakshminarayanan, B.: Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research 22(57), 1–64 (2021)

29. Peebles, W., Xie, S.: Scalable difusion models with transformers. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 4195–4205 (2023)

30. Qiu, Z., Wang, Z., Zheng, B., Huang, Z., Wen, K., Yang, S., Men, R., Yu, L., Huang, F., Huang, S., Liu, D., Zhou, J., Lin, J.: Gated attention for large language models: Non-linearity, sparsity, and attention-sink-free (2025), https://arxiv. org/abs/2505.06708

31. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020 (2021)

32. Ren, S., Yu, Q., He, J., Shen, X., Yuille, A., Chen, L.C.: Beyond next-token: Nextx prediction for autoregressive visual generation. arXiv preprint arXiv:2502.20388 (2025)

33. Rezende, D., Mohamed, S.: Variational inference with normalizing flows. In: Bach, F., Blei, D. (eds.) Proceedings of the 32nd International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 37, pp. 1530–1538. PMLR, Lille, France (07–09 Jul 2015), https://proceedings.mlr.press/v37/ rezende15.html

34. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022)

35. Salimans, T., Goodfellow, I., Zaremba, W., Cheung, V., Radford, A., Chen, X.: Improved techniques for training gans. Advances in neural information processing systems 29 (2016)

36. Shen, Y., Chen, T., Gao, Y., Zhang, Y., Wang, Y., Ángel Bautista, M., Zhai, S., Susskind, J.M., Gu, J.: Starflow2: Bridging language models and normalizing flows for unified multimodal generation (2026), https://arxiv.org/abs/2605.08029

37. Singh, J., Zheng, B., Wu, Z., Zhang, R., Shechtman, E., Xie, S.: Improved baselines with representation autoencoders (2026), https://arxiv.org/abs/2605.18324

38. Tian, K., Jiang, Y., Yuan, Z., Peng, B., Wang, L.: Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems 37, 84839–84865 (2024)

39. Tschannen, M., Pinto, A.S., Kolesnikov, A.: Jetformer: An autoregressive generative model of raw images and text. arXiv preprint arXiv:2411.19722 (2024)

40. Tu, G., Fu, X., Yu, S., Tang, Y., Kang, H., Qin, L., Zhang, Y., Gu, J.: Latent reasoning with normalizing flows (2026), https://arxiv.org/abs/2606.06447

41. Wang, S., Gao, Z., Zhu, C., Huang, W., Wang, L.: Pixnerd: Pixel neural field difusion (2025), https://arxiv.org/abs/2507.23268

42. Wang, S., Tian, Z., Huang, W., Wang, L.: Ddt: Decoupled difusion transformer. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 40633–40642 (June 2026)

43. Xie, Z., Zhang, Z., Cao, Y., Lin, Y., Bao, J., Yao, Z., Dai, Q., Hu, H.: Simmim: A simple framework for masked image modeling (2022), https://arxiv.org/abs/ 2111.09886

44. Yang, J., Li, T., Fan, L., Tian, Y., Wang, Y.: Latent denoising makes good tokenizers (2026), https://arxiv.org/abs/2507.15856

45. Yao, J., Song, Y., Zhou, Y., Wang, X.: Towards scalable pre-training of visual tokenizers for generation (2025), https://arxiv.org/abs/2512.13687

46. Yao, J., Yang, B., Wang, X.: Reconstruction vs. generation: Taming optimization dilemma in latent difusion models (2025), https://arxiv.org/abs/2501.01423

47. Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., Xie, S.: Representation alignment for generation: Training difusion transformers is easier than you think. arXiv preprint arXiv:2410.06940 (2024)

48. Zhai, S., Zhang, R., Nakkiran, P., Berthelot, D., Gu, J., Zheng, H., Chen, T., Bautista, M.A., Jaitly, N., Susskind, J.: Normalizing flows are capable generative models. arXiv preprint arXiv:2412.06329 (2024)

49. Zhao, Q., Zheng, G., Yang, T., Zhu, R., Leng, X., Gould, S., Zheng, L.: Simflow: Simplified and end-to-end training of latent normalizing flows (2025), https:// arxiv.org/abs/2512.04084

50. Zheng, B., Ma, N., Tong, S., Xie, S.: Difusion transformers with representation autoencoders (2025), https://arxiv.org/abs/2510.11690

51. Zheng, G., Zhao, Q., Yang, T., Xiao, F., Lin, Z., Wu, J., Deng, J., Zhang, Y., Zhu, R.: Farmer: Flow autoregressive transformer over pixels (2025), https://arxiv. org/abs/2510.23588

52. Zheng, H., Nie, W., Vahdat, A., Anandkumar, A.: Fast training of difusion models with masked transformers. In: Transactions on Machine Learning Research (TMLR) (2024)

53. Zheng, Y., Tian, Y., Li, S., Wu, Z., Liu, B., Li, J., Ye, B., Zhou, J.R.: LightningDiT: A vision-foundation-model-aligned VAE for fast and high-quality generation. In: arXiv preprint arXiv:2405.15438 (2024)