# Relighting as a Probe of Visual Priors via Augmented Latent Intrinsics

Xiaoyan Xing 1 Xiao Zhang 2 Sezer Karagolu 1 Theo Gevers 1 Ananad Bhattad 3

![](images/8670021ccbec90edd900c5c82ea70243f69d97f79d83ac5a1448e06ceb505323.jpg)

<details>
<summary>text_image</summary>

Input image
CLIP
MAE
Ours
Target lighting
DINOv2
DINOv3
GT Relit
</details>

Generative relighting with different visual representation features

![](images/97fcf032dbe08b54849cf247bb0013cc7244a476050a55c1dcdf58c93434fda0.jpg)

<details>
<summary>scatterplot</summary>

| Method     | Relighting performance (PSNR) |
| ---------- | ----------------------------- |
| MAE        | 18.0                          |
| CLIP       | 16.4                          |
| DINOv2     | 15.9                          |
| DINOv3     | 16.2                          |
| Ours       | 18.4                          |
</details>

Linear-probe (ImageNet-1K,top-1)  
Figure 1. Stronger Semantic Encoders Can Harm Relighting Performance. Left: Visual comparison on a scene with complex specular materials. The task is to relight the input image (top-left) using the target illumination (bottom-left), which requires moving specular highlights from left to right, as indicated by the chrome sphere. While features from semantic encoders (CLIP, DINO) fail to reproduce realistic highlights, the MAE plausibly moves the highlight but blurs fine details, such as text labels. Our method (top-right), which combines features from RADIO (a pretrained model; distilled from many vision encoders) with latent intrinsics, closely matches the ground truth. Right: Quantitative analysis reveals a trade-off: for most encoders optimized for pure semantics, relighting quality (PSNR) is inversely correlated with recognition performance (ImageNet-1K linear probing as reported in the original papers.). Our approach breaks this trend, achieving high performance on both tasks.

## Abstract

Image-to-image relighting requires representations that separate illumination from scene properties while preserving dense geometry, material, and photometric cues. We use this task as a probe of visual priors: unlike recognition tasks that reward invariance, relighting tests whether visual features retain the information needed for light transfer. Through a controlled generative relighting framework, we find that strong semantic encoders can degrade relighting quality, exposing a semantic–photometric trade-off between abstraction and physical fidelity. We introduce Augmented Latent Intrinsics (ALI), which balances this trade-off by fusing dense, pixel-aligned visual features into a latent-intrinsic relighting model and refining it with self-supervision on unlabeled real image pairs. ALI improves relighting quality, especially on glossy, metallic, and transparent materials, and demonstrates that generative relighting is an effective tool for quantifying what visual encoders encode about the physical world.

## 1. Introduction

Image-based relighting is the task of transferring illumination from one image to another. It has important applications in augmented reality, computational photography, and digital content creation. Yet it remains fundamentally ill-posed: brightness may arise from a light source or a shiny surface, and darkness may indicate either shadow or material absorption. Successful relighting, therefore, requires more than semantic recognition. It demands visual representations that encode how illumination interacts with scene geometry, material properties, and fine-grained appearance.

In this paper, we use generative relighting as a probe of visual representations. While recognition tasks often reward invariance to lighting, texture, and local appearance, relighting requires exactly the information that such invariances tend to suppress. A representation useful for relighting must preserve dense, pixel-aligned photometric structure while still providing enough semantic context to respect object and material boundaries. This makes relighting a controlled testbed for studying what visual encoders encode about the physical world: not merely whether they recognize objects, but whether their features support physically meaningful image transformations.

We focus on image-to-image relighting as a practical setting for this analysis. Unlike classical inverse-graphics pipelines that explicitly estimate geometry, reflectance, and light sources, we test whether learned visual features—augmented with latent intrinsic representations (Zhang et al., 2024a)— are sufficient to support illumination transfer. This direct formulation avoids explicit decomposition into physical components, but places strong demands on the representation. Recent diffusion-based relighting models show impressive generative capacity, yet often struggle with materials involving complex, view-dependent effects such as specularity and transparency (Xing et al., 2025; Zeng et al., 2024b; Liang et al., 2025; He et al., 2025). These failures suggest that generative power alone is insufficient: the conditioning representation must also encode the physical cues needed for light transport.

A natural hypothesis is that large-scale pretrained visual encoders, such as CLIP and DINO, should provide strong priors for resolving these ambiguities. Our findings reveal a more nuanced picture. Features optimized for semantic invariance through contrastive learning or self-distillation can systematically degrade relighting performance, despite their strong recognition ability. In contrast, features from a masked auto-encoder (MAE), trained with a dense pixel reconstruction objective, yield better relighting results even though they are weaker semantic representations (Fig. 1). This exposes a critical mismatch between semantic representation quality and physical usefulness: the inductive biases that make features robust for recognition can discard the fine-grained photometric and spatial information required for relighting.

This motivates our central question of investigation in this paper: what makes a visual representation relightable? We define relightability operationally and objectively as the ability to support three properties: (i) photometric consistency under light transfer, (ii) material fidelity for view- and materialdependent effects, and (iii) illumination robustness while retaining dense per-pixel detail. Through controlled comparisons of different pretrained encoders within the same generative relighting pipeline, we show that relightability is controlled by a semantic–photometric trade-off. High-level semantic features provide object-level context but often lose dense appearance cues; low-level features preserve photometric detail but lack semantic structure. Dense, pixel-aligned encoders such as RADIO (Heinrich et al., 2024b), which distill complementary visual teachers, best balance these requirements.

To build this probe, we introduce Augmented Latent Intrinsics (ALI), a controlled framework for image-to-image relighting that allows us to quantify how different visual priors affect generative light transfer. ALI combines latent intrinsic features with semantic features from a frozen visual encoder through a lightweight fusion adapter, and aligns the resulting representation with a diffusion-based decoder using a progressive training schedule. This design enables systematic analysis of how encoder choice changes relighting behavior while also producing a practical relighting model. A final self-supervision stage improves robustness on in-the-wild images without altering the core representation analysis.

Experiments on the MIIW benchmark show that ALI achieves state-of-the-art performance among open-sourced diffusion-based relighting methods, with especially large gains on glossy and specular materials. More importantly, our analysis demonstrates that relighting can serve as a sensitive probe of physically grounded visual priors: encoders that appear strong under semantic benchmarks may fail to preserve the material, geometric, and photometric cues required for realistic illumination transfer.

In summary, our contributions are:

• We propose generative relighting as a probe of physically grounded visual priors, using relightability to quantify what visual encoders encode about illumination, material, and geometry.  
• We introduce ALI, a controlled image-to-image relighting framework that exposes and balances the semantic– photometric trade-off in pretrained visual representations.  
• We provide a systematic analysis showing that strong semantic encoders can fail at relighting, while dense pixelaligned representations better preserve the physical cues required for light transfer.  
• ALI achieves state-of-the-art results on MIIW among open-sourced diffusion-based methods, improving RMSE by 4.5% and SSIM by 4.9% over LumiNet, with the largest gains on glossy and specular materials.

## 2. Related Work

Generative Relighting. Image-based relighting modifies illumination while preserving scene content, implicitly requiring a model to separate lighting from geometry, material, and semantics. Classical inverse-rendering methods address this ambiguity by estimating or assuming physical factors such as geometry, reflectance, and illumination (Li et al., 2022; 2023; 2021; Zhang et al., 2016; Zhu et al., 2023; Garon et al., 2019; Gardner et al., 2019), but often rely on controlled assumptions and accurate intermediate estimates.

Recent generative methods learn relighting directly, achieving compelling results for objects (Zeng et al., 2024a; Deng et al., 2024; Jin et al., 2024; Zhang et al., 2025; Bharadwaj et al., 2024), portraits (Pandey et al., 2021; He et al., 2024;

Mei et al., 2025), and indoor scenes (Kocsis et al., 2024a; Xing et al., 2024; Choi et al., 2025; Zeng et al., 2024b). However, strong systems such as the concurrent LightLab (Magar et al., 2025), UniRelight (He et al., 2025), and IntrinsicEdit (Lyu et al., 2025) rely heavily on dense supervision, synthetic data, or explicit physical annotations. Unsupervised alternatives exploit priors from pretrained GANs and diffusion models (Bhattad et al., 2024; Xing et al., 2025), but photometric objectives alone remain vulnerable to intrinsic ambiguities, especially for glossy, transparent, or spatially complex materials. We instead treat relighting as a controlled probe for evaluating whether learned visual priors preserve the physical information needed for light transfer.

Physically Grounded Visual Priors. Intrinsic image decomposition (Barrow & Tenenbaum, 1978) provides a natural physical prior by factorizing images into illuminationindependent properties, such as albedo, and illuminationdependent components, such as shading. Prior work has used low-level cues (Baslamisli et al., 2021; Luo et al., 2020; Das et al., 2022; Fan et al., 2018; Chen & Koltun, 2013; Xing et al., 2022), physical assumptions (Barron & Malik, 2014; Grosse et al., 2009), ordinal supervision (Careaga & Aksoy, 2023; 2024; Dille et al., 2024), semantic reasoning (Baslamisli et al., 2018), and generative priors (Bhattad et al., 2023; Du et al., 2023; Kocsis et al., 2024b; Zeng et al., 2024b; Luo et al., 2024; Xi et al., 2024). Self-supervised methods further exploit real image pairs captured under varying illumination (Li & Snavely, 2018; Ma et al., 2018; Janner et al., 2017).

These representations preserve dense photometric structure, but intrinsic decomposition alone does not resolve all relighting ambiguities. In real scenes, distinguishing lighting from material or object identity often requires semantic context. Latent Intrinsics (Zhang et al., 2024a) captures useful photometric structure in latent space without explicit labels, but lacks the semantic grounding needed for complex materials and object boundaries. We therefore use latent intrinsics as one component in a broader probe of physically grounded visual priors.

Semantic Visual Representations. Large-scale visual encoders trained with contrastive, distillation, or reconstruction objectives (He et al., 2020; Chen et al., 2020; Caron et al., 2021; Vincent et al., 2008; Ho et al., 2020; He et al., 2022a; Zhang et al., 2024b;c) encode rich information about object identity, layout, parts, and material categories. Such semantics are useful for relighting, where object boundaries and material identity help constrain illumination transfer.

However, semantic strength does not guarantee physical usefulness. Many high-level encoders are optimized for invariance to lighting, texture, and local appearance, while relighting requires these signals to be preserved. This creates a semantic–photometric trade-off: recognition-oriented features may lose dense light-transfer cues, while photometric features may lack semantic structure. Relighting therefore offers a sensitive probe of visual representations, requiring features that encode both what the scene contains and how its surfaces respond to light.

To our knowledge, prior work has not systematically examined large-scale visual encoders through generative relighting. We address this with ALI, a controlled image-toimage relighting framework for quantifying how visual priors affect relighting behavior. Our analysis shows that dense, pixel-aligned representations such as RADIOv2.5H (Heinrich et al., 2024b), when fused with latent intrinsic features, better balance semantic grounding and photometric fidelity than either prior alone.

## 3. Preliminaries

Our work builds on the concept of latent intrinsics, where lighting-invariant features can be learned from multiillumination image pairs without direct supervision (Zhang et al., 2024a). Given an image $I _ { s } ^ { l }$ of a scene s under lighting $l ,$ an encoder $\scriptstyle { E _ { \theta } }$ disentangles it into a set of hierarchical, lighting-invariant intrinsic features $\{ S _ { s , i } ^ { l } \}$ and a global lighting embedding $L _ { s } ^ { l } .$ . A decoder $D _ { \phi }$ can then relight the scene by combining the intrinsics from one view with the lighting from another:

$$
\widetilde {\boldsymbol {I}} _ {s} ^ {l _ {1} \rightarrow l _ {2}} = \boldsymbol {D} _ {\phi} (\{\boldsymbol {S} _ {s, i} ^ {l _ {1}} \}, \boldsymbol {L} _ {s} ^ {l _ {2}}). \tag {1}
$$

The model is trained from scratch using a combination of reconstruction, intrinsic invariance, and latent space regularization losses. However, because these models are typically trained on a limited set of real-world multi-illumination pairs, their learned representations are constrained to the lighting and material types seen during training. This often leads to failures in accurately disentangling complex materials under diverse, unseen lighting conditions. To address this, we propose to augment the latent intrinsic features with powerful visual priors from foundation models trained on large-scale, diverse image collections. While these priors are effective for high-level tasks, their utility for the fine-grained physical reasoning required in relighting remains unexplored.

## 4. Method

Our goal is to use image-to-image relighting as a controlled probe of physically grounded visual priors. Rather than changing the full relighting pipeline, we keep the backbone, training data, and decoder objective fixed, and vary the pretrained visual encoder used to augment the latent intrinsic representation. This isolates how different visual priors affect relightability, including photometric consistency, material fidelity, and preservation of dense scene structure.

We instantiate this probe with Augmented Latent Intrinsics (ALI), a framework built on LumiNet (Xing et al., 2025). ALI injects frozen visual encoder features into latent intrinsic representations and aligns them with a generative relighting decoder through a three-stage training strategy.

![](images/2997437e5b1b1ed647c81e43f1e272bd3aff92f70879970ecdff784ee8b375a0.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    subgraph Stage_I[Stage I: Visual Priors Augmented Latent Intrinsic]
  A1["Image 1"] --> A2["Visual Encoder"]
  A2 --> A3["ProJθ"]
  A3 --> A4["Intrinsic Encoder"]
  A4 --> A5["Augmented Latent Intrinsic"]
  A5 --> A6["LumiNet Diffusion"]
  A6 --> A7["Output"]
    end

    subgraph Stage_II[Stage II: Aligning Generative Decoder with Augmented Latent Intrinsic]
  B1["Image 1"] --> B2["Visual Encoder"]
  B2 --> B3["ProJθ"]
  B3 --> B4["Intrinsic Encoder"]
  B4 --> B5["Augmented Latent Intrinsic"]
  B5 --> B6["LumiNet Diffusion"]
  B6 --> B7["Output"]
    end

  A1 --> A5
  A2 --> A5
  A3 --> A5
  A4 --> A5
  A5 --> A6
  A6 --> A7
  B1 --> B6
  B2 --> B7
  B3 --> B7
  B4 --> B7
  B5 --> B7
  B6 --> B7
  B7 --> Output
    style Stage_I fill:#f9f,stroke:#333
    style Stage_II fill:#f9f,stroke:#333
```
</details>

![](images/7b30953f819907a4a519f88e654c32532ca38e205c0cbbecf5728cd85caf42d4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["In-the-wild Image"] --> B["Augmented Latent Instrinsics"]
  B --> C["Augmented Latent-intrinsics"]
  C --> D["LumiNet Diffusion"]
  D --> E["Lighting Zoo"]
  E --> F["Augmented Latent Instrinsics"]
  F --> G["LumiNet Diffusion"]
  G --> H["Output"]
  B --> I["Latent-extrinsics"]
  I --> J["Random Images for Extrinsic"]
  J --> K["In-the-wild Image"]
  F --> L["Latent-extrinsics"]
  L --> M["Stage III: Self-Refinement with Mixed Supervision"]
  G --> N["Latent-extrinsics"]
  N --> O["Output"]
    style A fill:#f9f,stroke:#333
    style H fill:#bbf,stroke:#333
```
</details>

Figure 2. Our Three-Stage Training Pipeline for Augmented Latent Intrinsics (ALI). Our method progressively adapts a pretrained visual encoder and fine-tunes a generative decoder for high-fidelity, unsupervised relighting. Stage I: Augmenting Latent Intrinsics. We inject semantic features from a frozen vision encoder into the intrinsics encoder. This creates our semantically-enriched ALI, which better disentangles scene properties from illumination. Stage II: Aligning the Generative Decoder. With the encoder fixed, we fine-tune the LumiNet diffusion decoder to condition on the new $\mathbf { A L I }$ representation, aligning the generator with the scene’s improved physical understanding. Stage III: Self-Refinement. We generate $\mathrm { a \ { } ^ { * } \bar { I } }$ ighting ${ Z _ { 0 0 } } ^ { , , }$ (examples in supplementary) of pseudo-relit images to overcome data scarcity. These synthetic images serve as new inputs, with the original real image as the ground truth. This self-supervision trains the network to ignore artifacts and focus on essential structural properties, improving realism for in-the-wild images.

Stage 1: Augmenting Latent Intrinsics. To improve the disentanglement of latent intrinsic (like albedo) and extrinsic (like lighting), we inject semantic context from a frozen visual encoder $E _ { \mathrm { s e m } } \ \mathrm { ( e . g . , R A D I O v { 2 . } } 5 )$ into the latent intrinsics. First, we extract a hierarchy of feature maps $\{ V _ { s , i } ^ { l } \} _ { i = 1 } ^ { N } = E _ { \mathrm { s e m } } ( I _ { s } ^ { l } )$ $\pmb { I } _ { s } ^ { l } .$ notes the number of selected intermediate layers from each VRM encoder; the exact layer indices and feature dimensions are provided in the supplementary material. These maps are upsampled to the input resolution and concatenated into a pixel-wise hypercolumn descriptor $H _ { s } ^ { l } ( x , y )$ . A learnable projection layer $\operatorname { P r o j } _ { \theta ^ { \prime } }$ then aligns these features with the intrinsic features $S _ { s , i } ^ { l }$ from the relighting encoder $\scriptstyle { E _ { \theta } } :$ :

$$
\boldsymbol {A} _ {s, i} ^ {l} = \operatorname{Proj} _ {\theta^ {\prime}} (\boldsymbol {H} _ {s, i} ^ {l}) + \boldsymbol {S} _ {s, i} ^ {l}. \tag {2}
$$

In this stage, we freeze the visual encoder $E _ { \mathrm { s e m } }$ and all decoders, training only the relighting encoder $\scriptstyle { E _ { \theta } }$ and the projection layers $\operatorname { P r o j } _ { \theta ^ { \prime } }$ . The training objective combines a standard reconstruction loss $\mathcal { L } _ { \mathrm { r e l i g h t } }$ and a hyperspherical regularization loss $\mathcal { L } _ { \mathrm { r e g } }$ applied to both intrinsic and lighting features (Zhang et al., 2024a):

$$
\mathcal {L} _ {\text { reg }} (\boldsymbol {A}) = \left\| R (\boldsymbol {A}) - R (\hat {\boldsymbol {A}}) \right\| _ {2} ^ {2} \tag {3}
$$

$$
R (\boldsymbol {A}) = \log \det \left(\boldsymbol {I} + \frac {d}{n \lambda^ {2}} \boldsymbol {A} ^ {\top} \boldsymbol {A}\right), \tag {4}
$$

where $\hat { A }$ is sampled from a uniform hyperspherical distribution, n is the number of spatial locations, d is the feature dimension, and λ is a regularization temperature parameter. This term encourages features to uniformly spread out in the sphere, aiding feature optimization.

To stabilize training, we replace the original intrinsic invariance loss with an improved version that regularizes the intrinsic features $A _ { s , i } ^ { l _ { n } }$ of a scene s toward their mean across M different lighting conditions:

$$
\mathcal {L} _ {\text { Improved   Intrinsic }} = \sum_ {s, m} \| \boldsymbol {A} _ {s, i} ^ {l _ {m}} - \frac {1}{M} \sum_ {m ^ {\prime}} \boldsymbol {A} _ {s, i} ^ {l _ {m ^ {\prime}}} \| _ {2}. \tag {5}
$$

The total loss for this stage is ř $\begin{array} { r l r } { { \mathcal { L } } _ { \mathrm { S t a g e 1 } } } & { { } = } & { { \mathcal { L } } _ { \mathrm { r e l i g h t } } + } \end{array}$ LImproved Intrinsics $\begin{array} { r } { + \sum _ { i } \mathcal { L } _ { \mathrm { r e g } } ( A _ { s , i } ^ { l } ) + \mathcal { L } _ { \mathrm { r e g } } ( \mathbf { \bar { L } } _ { s } ^ { l } ) } \end{array}$ .

Stage 2: Aligning the Generative Decoder. With visual encoder $\pmb { E } _ { \mathrm { s e m } }$ relighting encoder $E _ { \theta }$ and projection layer $\operatorname { P r o j } _ { \theta ^ { \prime } }$ fixed, we fine-tune LumiNet’s diffusion decoder $D _ { \phi ^ { \prime } }$ to align with our semantically-augmented latent intrinsic representation Al1 . $A _ { s , i } ^ { l _ { 1 } }$ The decoder is trained to predict the noise ϵ added to the target lighting latent $\pmb { L } _ { s } ^ { l _ { 2 } }$ at timestep t, conditioned on the new intrinsics. The optimization uses only the standard denoising score-matching loss from DDPM (Ho et al., 2020):

$$
\mathcal {L} _ {\text { Stage2 }} = \mathbb {E} _ {t, \epsilon} \left[ \| \boldsymbol {D} _ {\phi^ {\prime}} (\alpha_ {t} \boldsymbol {L} _ {s} ^ {l _ {2}} + \beta_ {t} \epsilon , \{\boldsymbol {A} _ {s, i} ^ {l _ {1}} \}, \boldsymbol {L} _ {s} ^ {l _ {2}}) - \epsilon \| _ {2} ^ {2} \right] \tag {6}
$$

Stage 3: Self-Refinement. To address the scarcity of paired real-world data, we refine the decoder using a self-training scheme. Pseudo-relit pairs are generated by transferring illumination between randomly sampled images within a batch. The decoder is fine-tuned on these pseudo-pairs using the same diffusion loss as in Stage II, while periodically mixing in same-image reconstructions (source equals target) to preserve content fidelity. This self-refinement improves realism and robustness without requiring any labeled data.

## 5. Experiments

Experimental Setup. We train ALI on real image pairs from two datasets: the MIT Multi-Illuminant (MIIW) dataset (Murmann et al., 2019), which contains 985 scenes captured under 25 lighting conditions, and the BigTime dataset (Li & Snavely, 2018), which contains 460 scenes under 20–50 natural illumination conditions.

All stages are trained with AdamW (Loshchilov & Hutter, 2019) using a learning rate of $4 \times 1 0 ^ { - 5 }$ . In Stage I, we use a scene-aware batch sampler that groups multiple illuminations of the same scene to enforce intrinsic consistency. In Stage II, this constraint is relaxed and images are sampled across scenes to increase decoder diversity. In Stage III, we perform self-refinement using pseudo-relit data generated from a lighting zoo, randomly sampling 1,000 scenes for this process (Sec. A.2).

At inference time, although ALI is trained on same-scene image pairs, it generalizes to arbitrary unpaired image-toimage relighting. Intrinsic features are extracted from a content image, while lighting embeddings are extracted from a separate illumination image and transferred through the diffusion decoder. Following LumiNet (Xing et al., 2025), we use the bypass decoder (Wang et al., 2024) by default to preserve identity. However, all feature-fusion experiments disable this bypass to isolate the effect of the augmented latent intrinsic representation.

Evaluation Overview. We evaluate ALI in two complementary roles: as an image-to-image relighting model and as a controlled probe of physically grounded visual priors. First, we compare relighting quality against existing methods on MIIW and in-the-wild images. We then use the same framework to analyze how different pretrained visual encoders affect relightability, with emphasis on material-dependent behavior and the semantic–photometric trade-off.

Standard full-reference metrics such as PSNR, SSIM, and RMSE correlate imperfectly with perceived lighting quality and often emphasize low-frequency tone consistency over directional shadows or specular effects (Giroux et al., 2024; Xing et al., 2025). We therefore report them for protocol consistency and controlled ablations, while also relying on qualitative comparisons and human evaluation to assess physical plausibility.

## 5.1. Relighting Performance

Relighting Performance on MIIW. We first evaluate crossscene relighting on the unseen test split of MIIW. We compare against feed-forward methods, including SA-AE (Hu et al., 2020) and Latent-Intrinsic (Zhang et al., 2024a), as well as diffusion-based methods, including RGBØX (Zeng et al., 2024b) and LumiNet (Xing et al., 2025). Following the standard protocol (Zhang et al., 2024a), each trial samples one target image and 12 reference lighting conditions. We report results on both raw outputs and outputs after global color correction to compensate for white-balance discrepancies.

As shown in Tab. 1, ALI achieves state-of-the-art performance among open-sourced diffusion-based methods under both settings, improving SSIM by 4.9% and RMSE by 4.5% over LumiNet. Methods trained exclusively on MIIW, such as SA-AE and Latent-Intrinsic, can achieve stronger pixellevel scores, reflecting the bias of standard metrics toward tone consistency rather than accurate modeling of shadows, specularities, and material-dependent effects. This discrepancy is visible in qualitative comparisons (Fig. 3) and human evaluation (Tab. 3); details of the human study are provided in the appendix.

Qualitatively, LumiNet produces coherent global illumination but often blurs material and geometric details, especially on reflective or transparent surfaces. In contrast, ALI better preserves fine structures and material-specific effects such as metallic highlights, cast shadows, and glass transparency. Several competing methods also use privileged inputs: DiffusionRenderer uses ground-truth light probes, while UniRelight requires albedo and environment maps. ALI achieves competitive or better relighting quality without 3D supervision, inverse-graphics labels, or ground-truth lighting maps.

In-Scene Relighting. We also evaluate in-scene relighting on MIIW (Tab. 2), where each image captured under illumination i is relit to the corresponding pi \` 12q illumination condition. ALI achieves consistent performance and the second-best LPIPS without privileged labels. In contrast, UniRelight, despite using additional supervision, video-based training, and dataset-specific optimization, still exhibits noticeable failures in challenging material regions, as shown in Fig. 3.

![](images/643e141a4f92f1f8793648f05e966813329135a8812529c7abe234f65e4cd925.jpg)  
Figure 3. Qualitative comparison of relighting methods on challenging MIIW test scenes. The task is to relight the Input scene using illumination from the Target lighting image. Compared with competing methods, including approaches that rely on privileged information such as ground-truth light maps, G-buffers, albedo, or environment maps, our method produces more physically plausible results. In the top row, baselines render the metallic toaster with faint or blurry highlights, while our method produces sharper reflections closer to the ground truth. In the second row, baselines miss or distort the shadow of the orange cereal box, and LumiNet also blurs text on the packaging. In the third row, baselines struggle with transparency and caustic-like effects on the bottles, while our method produces a more plausible result. Red and green markers highlight representative failures and successes, respectively. Results for UniRelight were provided by the authors. Best viewed on screen with zoom.

Table 1. MIIW cross-scene evaluation. Following Latent-Intrinsics and LumiNet, we compare source and target images from different scenes and report RMSE and SSIM on raw and colorcorrected outputs. Trained on diverse images without privileged labels, our method performs competitively—especially within its training category. Bold highlights the best in each block (MIIWonly vs. diverse images for training). Note that metrics favor tone consistency over shadow or specular accuracy; qualitative results remain the primary evidence of lighting fidelity (Giroux et al., 2024).

<table><tr><td>Methods</td><td>Labels</td><td>RMSE↓</td><td>SSIM↑</td><td>RMSE↓</td><td>SSIM↑</td></tr><tr><td colspan="6">Trained only on MIIW</td></tr><tr><td>SA-AE (Hu et al., 2020)</td><td>Light</td><td>0.288</td><td>0.484</td><td>0.232</td><td>0.559</td></tr><tr><td>SA-AE (Hu et al., 2020)</td><td>-</td><td>0.443</td><td>0.300</td><td>0.317</td><td>0.431</td></tr><tr><td>S3Net (Yang et al., 2021)</td><td>Depth</td><td>0.512</td><td>0.331</td><td>0.418</td><td>0.374</td></tr><tr><td>S3Net (Yang et al., 2021)</td><td>-</td><td>0.499</td><td>0.336</td><td>0.414</td><td>0.377</td></tr><tr><td>Latent-Intrinsic (Zhang et al., 2024a)</td><td>-</td><td>0.297</td><td>0.473</td><td>0.222</td><td>0.571</td></tr><tr><td colspan="6">Trained on diverse indoor images</td></tr><tr><td>RGB↔X (Zeng et al., 2024b)</td><td>G-Buffer</td><td>0.587</td><td>0.070</td><td>0.427</td><td>0.215</td></tr><tr><td>DiffusionRenderer (Liang et al., 2025)</td><td>Env. map</td><td>0.399</td><td>0.354</td><td>0.341</td><td>0.355</td></tr><tr><td>LumiNet (Xing et al., 2025)</td><td>-</td><td>0.310</td><td>0.440</td><td>0.240</td><td>0.527</td></tr><tr><td>Ours</td><td>-</td><td>0.294</td><td>0.464</td><td>0.231</td><td>0.553</td></tr></table>

## 5.2. Relighting as a Probe of Visual Priors

Which Visual Priors Are Relightable? We next use ALI as a controlled probe to study how different pretrained visual encoders affect relighting. We keep the relighting backbone, training data, and decoder objective fixed, and vary only the frozen encoder used for semantic augmentation. This isolates the contribution of each visual prior to relightability.

Table 4 compares CLIP, DINOv2/3, MAE, and RADIOv2.5. MAE and RADIOv2.5 consistently outperform CLIP and DINO-based features. This suggests that semantic recognition strength alone is not sufficient for relighting. Contrastive and distillation-based encoders often promote invariance to lighting, texture, and local appearance, which are precisely the signals required for light transfer. In contrast, MAE’s pixel-reconstruction objective better preserves fine-grained color, shading, and reflectance cues. RADIOv2.5 further improves this balance by providing dense, pixel-aligned features distilled from complementary teachers. These results support our central finding: relightability depends on balancing semantic context with photometric fidelity.

Table 2. MIIW in-scene evaluation. Results on the multiillumination dataset of (Murmann et al., 2019), where source and target are the same scene captured under different illuminations. We report PSNR, RMSE, LPIPS, and SSIM. Our method, trained on diverse images without privileged labels, achieves competitive results compared to prior approaches. Higher PSNR/SSIM and lower RMSE/LPIPS indicate better performance. † trained only on MIIW; \* numbers reported from UniRelight. Pixel metrics can be sensitive to small misalignments (Giroux et al., 2024).

<table><tr><td>Methods</td><td>Labels</td><td>PSNR↑</td><td>RMSE↓</td><td>LPIPS↓</td><td>SSIM↑</td></tr><tr><td>RGB↔X (Zeng et al., 2024b)</td><td>G-Buffer</td><td>15.674</td><td>0.161</td><td>0.323</td><td>0.500</td></tr><tr><td>DiffusionRenderer (Liang et al., 2025)</td><td>Env. map</td><td>16.810</td><td>0.156</td><td>0.343</td><td>0.612</td></tr><tr><td>UniRelight (He et al., 2025)*</td><td>Albedo, Env. map</td><td>20.760</td><td>-</td><td>0.251</td><td>0.749</td></tr><tr><td>Latent-Intrinsics (Zhang et al., 2024a)†</td><td>-</td><td>21.350</td><td>0.092</td><td>0.157</td><td>0.794</td></tr><tr><td>LumiNet (Xing et al., 2025)</td><td>-</td><td>18.568</td><td>0.123</td><td>0.228</td><td>0.645</td></tr><tr><td>Ours</td><td>-</td><td>18.872</td><td>0.119</td><td>0.213</td><td>0.671</td></tr></table>

This observation is consistent with concurrent work RAE (Zheng et al., 2025), which finds that an MAE-based representation paired with a diffusion decoder outperforms a DINO-based counterpart for image reconstruction under the same decoding architecture. Our results extend this insight to relighting, where preserving photometric structure is not only useful for reconstruction but essential for physically meaningful illumination transfer.

Material-Wise Relightability. To further probe representation quality, we evaluate relighting performance within material-specific regions. We group per-pixel MIIW material labels into five physically motivated categories: Diffuse,

![](images/c0d719e9139f1554888f807fd1155bc41bf7875c2612d24cb00e803c3e20605d.jpg)

<details>
<summary>text_image</summary>

Light 1
Light 2
Light 1
Light 2
Target Lighting
Original Image
IC-Light
Latent-intrinsics
LumiNet
Ours
</details>

Figure 4. Relighting comparison across two real-world images, each shown under two target illuminations (Light 1: lamp-dominated; Light 2: sunlight through windows). IC-Light (Zhang et al., 2025) produces stylized results with exaggerated glow and artifacts that diverge from the targets. Latent-intrinsics (Zhang et al., 2024a) captures some variation but yields low-contrast, flattened illumination with weak directionality. LumiNet (Xing et al., 2025) better matches global tone but remains overly diffuse, often missing dominant light sources and underestimating cast shadows and highlight localization. Ours preserves material detail and transfers both global and directional lighting, producing images that most closely match the targets lighting.

Table 3. User Study. We conduct two pairwise user studies: (a) an in-the-wild comparison among LumiNet, Latent-Intrinsics, and our method; and (b) a stage-wise evaluation across our training pipeline. Users are shown two relit images and asked to select the one with better lighting alignment, identity preservation, and lighting realism. Latent-Intrinsics better preserves identity, but struggles with lighting alignment and realism. Our method achieves stronger lighting realism and alignment, with Stage III further improving perceptual quality. $^ { * } p < 0 . 0 5$  
(a) In-the-wild relighting Ò

<table><tr><td></td><td>Latent-Intrinsics</td><td>LumiNet</td><td>Ours</td></tr><tr><td>Lighting alignment</td><td>0.13*</td><td>0.42</td><td>0.93*</td></tr><tr><td>Identity preservation</td><td>0.68*</td><td>0.05*</td><td>0.63*</td></tr><tr><td>Lighting realism</td><td>0.28*</td><td>0.75*</td><td>0.90*</td></tr></table>

(b) Stage-wise evaluation Ò

<table><tr><td></td><td>LumiNet</td><td>Stage I</td><td>Stage I&amp;II</td><td>All Stages</td></tr><tr><td>Lighting alignment</td><td>0.44</td><td>0.20*</td><td>0.21*</td><td>0.75*</td></tr><tr><td>Identity preservation</td><td>0.05*</td><td>0.45</td><td>0.68*</td><td>0.96*</td></tr><tr><td>Lighting realism</td><td>0.25*</td><td>0.35*</td><td>0.68*</td><td>0.89*</td></tr></table>

Glossy, Specular, Metallic, and Uncertain/Mixed. We then compute SSIM, PSNR, and RMSE within each region.

As shown in Tab. 5, semantic augmentation in Stage I improves performance across all material categories, with the largest gains on non-diffuse materials such as Glossy, Metallic, and Specular. These regions require semantic context to distinguish material appearance from illumination effects, while also requiring dense photometric cues to reproduce highlights and view-dependent effects. Because specular and translucent regions occupy only a small fraction of the semantic masks, region-averaged metrics likely understate the perceptual improvement on the most challenging pixels.

Table 4. Impact of semantic features on relighting. Augmenting latent intrinsics with RADIOv2.5 or MAE significantly improves metrics after Stage II.

<table><tr><td>Feature</td><td>Stage</td><td>RMSE ↓</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td rowspan="2">Latent Intrinsic (Zhang et al., 2024a)</td><td>I</td><td>0.1380</td><td>0.2857</td><td>17.5763</td><td>0.5461</td></tr><tr><td>I&amp;II</td><td>0.1383</td><td>0.2844</td><td>17.5463</td><td>0.5531</td></tr><tr><td rowspan="2">MAE (He et al., 2022b)</td><td>I</td><td>0.2195</td><td>0.4820</td><td>14.2381</td><td>0.4571</td></tr><tr><td>I&amp;II</td><td>0.1286</td><td>0.2554</td><td>17.9861</td><td>0.4852</td></tr><tr><td rowspan="2">DINOv2 (Caron et al., 2021)</td><td>I</td><td>0.2786</td><td>0.3295</td><td>13.2439</td><td>0.4646</td></tr><tr><td>I&amp;II</td><td>0.1686</td><td>0.3253</td><td>15.7945</td><td>0.4815</td></tr><tr><td rowspan="2">DINOv3 (Siméoni et al., 2025)</td><td>I</td><td>0.1794</td><td>0.3588</td><td>15.1375</td><td>0.4824</td></tr><tr><td>I&amp;II</td><td>0.1654</td><td>0.2923</td><td>16.1510</td><td>0.5299</td></tr><tr><td rowspan="2">CLIP (Radford et al., 2021)</td><td>I</td><td>0.2189</td><td>0.4556</td><td>13.9671</td><td>0.3987</td></tr><tr><td>I&amp;II</td><td>0.1627</td><td>0.3153</td><td>16.1333</td><td>0.5039</td></tr><tr><td rowspan="2">RADIOv2.5 (Heinrich et al., 2024a)</td><td>I</td><td>0.1312</td><td>0.2673</td><td>17.9448</td><td>0.5609</td></tr><tr><td>I&amp;II</td><td>0.1260</td><td>0.2440</td><td>18.3426</td><td>0.5958</td></tr></table>

Intrinsic–Extrinsic Disentanglement. ALI learns a representation that separates scene-invariant intrinsic structure from illumination-dependent extrinsic codes. Figures S.2 and 4 show that holding intrinsics fixed while varying the lighting input produces coherent relighting. Moreover, Fig. 6 shows that interpolating the extrinsic code yields smooth and physically plausible lighting transitions within the same scene. This supports the interpretation that ALI improves relightability by augmenting latent intrinsics without collapsing lighting and content information.

## 6. Discussion

This work studies image-to-image relighting as a probe of physically grounded visual priors. Our results reveal a counterintuitive finding: features from strong semantic encoders can degrade, rather than improve, high-fidelity relighting. This suggests that recognition-oriented representations, while powerful for semantic abstraction, often suppress the illumination-sensitive details needed for physical image transformation. Relighting therefore exposes a semantic– photometric trade-off: useful features must preserve dense appearance cues while still providing enough semantic context to resolve material and object-level ambiguities.

Table 5. Performance comparison across different material semantic labels. Background color represents relative improvement within each metric, redder (warmer) shades indicating greater enhancement compared to the baseline LumiNet, which utilizes latent intrinsic features as intrinsic representation. Large gains are observed in complex metallic, glossy or specular surfaces. Note because scene/protocol/metric are fixed for this evaluation, PSNR/SSIM/RMSE serve as internal proxies: the monotonic improvements from Stage I→II→All-Stage confirm better photometric calibration without regressions; perceptual lighting gains are corroborated by the visual comparisons and directional cues.

<table><tr><td rowspan="2">Semantic Label</td><td colspan="3">LumiNet (Xing et al., 2025)</td><td colspan="3">Ours (Stage I Only)</td><td colspan="3">Ours (Stage I&amp;II)</td><td colspan="3">Ours (All Stage)</td></tr><tr><td>SSIM↑</td><td>PSNR↑</td><td>RMSE↓</td><td>SSIM↑</td><td>PSNR↑</td><td>RMSE↓</td><td>SSIM↑</td><td>PSNR↑</td><td>RMSE↓</td><td>SSIM↑</td><td>PSNR↑</td><td>RMSE↓</td></tr><tr><td>Uncertain</td><td>0.6082</td><td>17.8499</td><td>0.1426</td><td>0.6178</td><td>18.0678</td><td>0.1373</td><td>0.6518</td><td>18.6282</td><td>0.1307</td><td>0.6460</td><td>18.4975</td><td>0.1310</td></tr><tr><td>Diffuse</td><td>0.5281</td><td>18.0883</td><td>0.1369</td><td>0.5464</td><td>18.3033</td><td>0.1320</td><td>0.5798</td><td>18.7476</td><td>0.1261</td><td>0.5731</td><td>18.6726</td><td>0.1269</td></tr><tr><td>Glossy</td><td>0.5720</td><td>18.6291</td><td>0.1296</td><td>0.5988</td><td>19.0248</td><td>0.1217</td><td>0.6291</td><td>19.6906</td><td>0.1152</td><td>0.6196</td><td>19.5275</td><td>0.1157</td></tr><tr><td>Metallic</td><td>0.4164</td><td>15.4926</td><td>0.1759</td><td>0.4608</td><td>16.0831</td><td>0.1636</td><td>0.4855</td><td>16.3285</td><td>0.1588</td><td>0.4822</td><td>16.3827</td><td>0.1581</td></tr><tr><td>Specular</td><td>0.3778</td><td>17.0860</td><td>0.1490</td><td>0.4120</td><td>17.6720</td><td>0.1394</td><td>0.4423</td><td>17.9798</td><td>0.1357</td><td>0.4365</td><td>18.0688</td><td>0.1330</td></tr></table>

![](images/9c6a2d947bd9ccfc2a213df7120882d884dce8e348a19f22d7a0a40e6a5e7a6f.jpg)  
(a) Multi-stage ablation on MIIW (Murmann et al., 2019) dataset.

![](images/c7fc70c60376363703266818e3b3e67fda7e581342f9c6ba4d383ffe516bff19.jpg)  
Target lighting

![](images/51650416925fead747897025cc10531e95554c1eccbeb38034262f22b9c12b80.jpg)  
Input image

![](images/603ced7b24e3541696088052d67443bb1af56ecff904d9140866f838a3514fc0.jpg)  
LumiNet

![](images/4f4f7e523e0e8694dbb935dc92dbcc56c2b4b18d85cecf993f6cadc552363022.jpg)

![](images/f41268620b21c0c84f2c44141b46470fac2fa23b1c1517b624ffd6b8ba38cc7d.jpg)  
Stage I & II

![](images/661c802fd44f68548e8edeee2ab4388f7641715633db6e756409aefc6887fb18.jpg)  
Stage I & II & III  
(b) Multi-stage ablation on in-the-wild image (with bypass decoder disabled).  
Figure 5. Multi-stage ablation. Top: Compared to LumiNet, our Stage I improves fine geometry details. Adding Stage II sharpens directional cues and specular effects, while the full pipeline (Stage I&II&III) produces the closest match to ground truth, with accurate shadows, highlights, and material fidelity. This progression illustrates how each stage contributes complementary improvements, consistent with the quantitative gains in Tab. 5. Bottom: LumiNet produces flat illumination with weak lamp cues. Stage I introduces coarse global tone but has color shift, Stage II suppresses these effects, and the full pipeline (Stage I&II&III) yields the most faithful transfer: interior warmth from the lamp is preserved while maintaining the outdoor scene, closely matching the target lighting. This demonstrates that our stage-wise design generalizes to unconstrained real-world images.

Relightability depends on representation bias, not semantic strength alone. Our backbone analysis shows that stronger semantic encoders are not necessarily more relightable. Increasing feature density, as in DINOv2 to DI-NOv3, is insufficient when the underlying objective promotes invariance to texture, lighting, and local appearance. In contrast, MAE-style reconstruction objectives better preserve pixel-aligned structure and photometric detail, which are essential for light transfer. RADIOv2.5 further improves this balance by combining dense spatial alignment with complementary semantic cues from multi-teacher distillation. These results suggest that relightability depends less on a specific backbone family than on the inductive biases imposed by reconstruction, spatial alignment, and feature granularity.

Prior design can substitute for supervision scale. ALI belongs to a broader class of prior-driven generative methods. Rather than relying on large synthetic datasets, dense physical annotations, or explicit inverse-graphics supervision, ALI injects a frozen visual prior into a latent-intrinsic framework. This lightweight fusion improves both material fidelity and relighting robustness, especially for glossy and specular surfaces. In contrast to scale-centric approaches that attempt to learn physical structure from large supervised datasets (Zeng et al., 2024b; Liang et al., 2025; He et al., 2025), our results indicate that careful prior selection and integration can provide a more data-efficient route to physically grounded generation.

![](images/ac707984e6ec0fda5e3ed2f689c2c6446c9eab7b3367f34b36185b76c9ee9d95.jpg)

<details>
<summary>text_image</summary>

Light 1
Interpolations
Light 2
</details>

(a) Image relighting with interpolated lighting code.

![](images/f13ada5c2d3ada222acfad6115d60d95424f97e2e97108820a9ffca2f68ab941.jpg)

<details>
<summary>natural_image</summary>

Collage of interior photos showing various dining and living spaces with wooden furniture and minimalist decor (no visible text or symbols)
</details>

Original lighting  
Random lighting sampling  
(b) Zero-Shot image relighting with unpaired, randomly sampled lighting code.  
Figure 6. Lighting interpolation and diversity. Top: Generated images showing a smooth interpolation between two lighting codes. Note the plausible evolution of directional lighting, including the progressive appearance of sharp specular highlights on the toaster and caustic effects from the bottle. Bottom: In-the-wild relighting results using lighting codes sampled from random, unpaired images. Our method produces a diverse range of distinct illumination effects, plausible altering the glossy reflections on the dining table and the ambient lighting in the living room.

Limitations and future work. ALI improves material-aware relighting but does not solve inverse rendering. It can still shift fine details, conflate surface color with illumination, and hallucinate shadows from learned priors rather than explicit geometry. These limitations are inherent to image-only relighting without 3D structure or physical supervision. Extending this probing framework to related tasks, such as reflectance editing, shadow manipulation, view-consistent generation, or material decomposition, may further clarify which representation properties support physical reasoning in generative models.

## Impact Statement

This work contributes to the understanding of how visual representations support physically grounded image generation, using relighting as a diagnostic task. By showing that stronger semantic encoders can degrade relighting performance, our findings may influence how pretrained visual priors are selected and integrated into generative models for graphics and computational photography applications. Improved relighting models can benefit content creation, augmented reality, and visual effects by enabling more realistic lighting edits without requiring synthetic data or privileged supervision. We do not anticipate direct negative societal impacts from this work; however, as with other image generation technologies, downstream applications should consider ethical use and potential misuse of generated imagery.

## References

Barron, J. T. and Malik, J. Shape, illumination, and reflectance from shading. IEEE transactions on pattern analysis and machine intelligence, 2014.  
Barrow, H. and Tenenbaum, J. Recovering intrinsic scene characteristics from images. In Computer Vision Systems, 1978.  
Baslamisli, A. S., Groenestege, T. T., Das, P., Le, H.-A., Karaoglu, S., and Gevers, T. Joint learning of intrinsic images and semantic segmentation. In ECCV, 2018.  
Baslamisli, A. S., Das, P., Le, H.-A., Karaoglu, S., and Gevers, T. Shadingnet: Image intrinsics by fine-grained shading decomposition. IJCV, 2021.  
Bell, S., Bala, K., and Snavely, N. Intrinsic images in the wild. ACM Trans. on Graphics (SIGGRAPH), 2014.  
Bharadwaj, S., Feng, H., Becherini, G., Abrevaya, V. F., and Black, M. J. Genlit: Reformulating single-image relighting as video generation. arXiv preprint arXiv:2412.11224, 2024.  
Bhattad, A., McKee, D., Hoiem, D., and Forsyth, D. Stylegan knows normal, depth, albedo, and more. In NeurIPS, 2023.  
Bhattad, A., Soole, J., and Forsyth, D. A. Stylitgan: Imagebased relighting via latent control. In CVPR, 2024.  
Careaga, C. and Aksoy, Y. Intrinsic image decomposition via ordinal shading. ACM ToG, 2023.  
Careaga, C. and Aksoy, Y. Colorful diffuse intrinsic image decomposition in the wild. ACM ToG, 2024.  
Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., ´ Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021.  
Chen, Q. and Koltun, V. A simple model for intrinsic image decomposition with depth cues. In ICCV, 2013.  
Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597–1607. PmLR, 2020.  
Choi, J. M., Wang, A., Peers, P., Bhattad, A., and Sengupta, R. Scribblelight: Single image indoor relighting with scribbles. In CVPR, 2025.  
Das, P., Karaoglu, S., and Gevers, T. Pie-net: Photometric invariant edge guided network for intrinsic image decomposition. In CVPR, 2022.  
Deng, K., Omernick, T., Weiss, A., Ramanan, D., Zhu, J.-Y., Zhou, T., and Agrawala, M. Flashtex: Fast relightable mesh texturing with lightcontrolnet. In ECCV, 2024.  
Dille, S., Careaga, C., and Aksoy, Y. Intrinsic single-image hdr reconstruction. In ECCV, 2024.  
Du, X., Kolkin, N., Shakhnarovich, G., and Bhattad, A. Generative models: What do they know? do they know things? let’s find out! arXiv preprint arXiv:2311.17137, 2023.  
Fan, Q., Yang, J., Hua, G., Chen, B., and Wipf, D. Revisiting deep intrinsic image decompositions. 2018.  
Gardner, M.-A., Hold-Geoffroy, Y., Sunkavalli, K., Gagne,´ C., and Lalonde, J.-F. Deep parametric indoor lighting estimation. In Proceedings of the IEEE International Conference on Computer Vision, pp. 7175–7183, 2019.  
Garon, M., Sunkavalli, K., Hadap, S., Carr, N., and Lalonde, J.-F. Fast spatially-varying indoor lighting estimation. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2019.  
Giroux, J., Dastjerdi, M. R. K., Hold-Geoffroy, Y., Vazquez-Corral, J., and Lalonde, J.-F. Towards a perceptual evaluation framework for lighting estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4410–4419, 2024.  
Grosse, R., Johnson, M. K., Adelson, E. H., and Freeman, W. T. Ground truth dataset and baseline evaluations for intrinsic image algorithms. In ICCV, 2009.  
He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 9729–9738, 2020.  
He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, ´ R. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009, 2022a.  
He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, ´ R. Masked autoencoders are scalable vision learners. In CVPR, 2022b.  
He, K., Liang, R., Munkberg, J., Hasselgren, J., Vijaykumar, N., Keller, A., Fidler, S., Gilitschenski, I., Gojcic, Z., and Wang, Z. Unirelight: Learning joint decomposition and synthesis for video relighting. arXiv preprint arXiv:2506.15673, 2025.  
He, M., Clausen, P., Tas¸el, A. L., Ma, L., Pilarski, O., Xian, W., Rikker, L., Yu, X., Burgert, R., Yu, N., et al. Diffrelight: Diffusion-based facial performance relighting. In SIGGRAPH Asia, pp. 1–12, 2024.  
Heinrich, G., Ranzinger, M., Hongxu, Yin, Lu, Y., Kautz, J., Tao, A., Catanzaro, B., and Molchanov, P. Radiov2.5: Improved baselines for agglomerative vision foundation models. In CVPR2025, 2024a.  
Heinrich, G., Ranzinger, M., Lu, Y., Kautz, J., Tao, A., Catanzaro, B., Molchanov, P., et al. Radio amplified: Improved baselines for agglomerative vision foundation models. arXiv preprint arXiv:2412.07679, 2024b.  
Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.  
Hu, Z., Huang, X., Li, Y., and Wang, Q. Sa-ae for any-to-any relighting. In ECCV. Springer, 2020.  
Janner, M., Wu, J., Kulkarni, T. D., Yildirim, I., and Tenenbaum, J. Self-supervised intrinsic image decomposition. In NIPS, 2017.  
Jin, H., Li, Y., Luan, F., Xiangli, Y., Bi, S., Zhang, K., Xu, Z., Sun, J., and Snavely, N. Neural gaffer: Relighting any object via diffusion. In NeurIPS, 2024.  
Kocsis, P., Philip, J., Sunkavalli, K., Nießner, M., and Hold-Geoffroy, Y. Lightit: Illumination modeling and control for diffusion models. arXiv preprint arXiv:2403.10615, 2024a.  
Kocsis, P., Sitzmann, V., and Nießner, M. Intrinsic image diffusion for single-view material estimation. In CVPR, 2024b.  
Li, J., Li, H., and Matsushita, Y. Lighting, reflectance and geometry estimation from 360˝ panoramic stereo, 2021. URL https://arxiv.org/abs/2104.09886.  
Li, Z. and Snavely, N. Learning intrinsic image decomposition from watching the world. CVPR, 2018.  
Li, Z., Shi, J., Bi, S., Zhu, R., Sunkavalli, K., Hasan, M., ˇ Xu, Z., Ramamoorthi, R., and Chandraker, M. Physicallybased editing of indoor scene lighting from a single image, 2022. URL https://arxiv.org/abs/2205. 09343.  
Li, Z., Wang, L., Cheng, M., Pan, C., and Yang, J. Multiview inverse rendering for large-scale real-world indoor scenes, 2023. URL https://arxiv.org/abs/ 2211.10206.  
Liang, R., Gojcic, Z., Ling, H., Munkberg, J., Hasselgren, J., Lin, Z.-H., Gao, J., Keller, A., Vijaykumar, N., Fidler, S., and Wang, Z. Diffusionrenderer: Neural inverse and forward rendering with video diffusion models. In CVPR, June 2025.  
Ling, L., Sheng, Y., Tu, Z., Zhao, W., Xin, C., Wan, K., Yu, L., Guo, Q., Yu, Z., Lu, Y., et al. Dl3dv-10k: A large-scale scene dataset for deep learning-based 3d vision. In CVPR, pp. 22160–22169, 2024.  
Loshchilov, I. and Hutter, F. Decoupled weight decay regularization. In ICLR, 2019.  
Luo, J., Huang, Z., Li, Y., Zhou, X., Zhang, G., and Bao, H. Niid-net: adapting surface normal knowledge for intrinsic image decomposition in indoor scenes. IEEE TVCG, 26 (12):3434–3445, 2020.  
Luo, J., Ceylan, D., Yoon, J. S., Zhao, N., Philip, J., Fruhst ¨ uck, A., Li, W., Richardt, C., and Wang, T. In-¨ trinsicdiffusion: joint intrinsic layers from latent diffusion models. In SIGGRAPH, 2024.  
Lyu, L., Deschaintre, V., Hold-Geoffroy, Y., Hasan, M.,ˇ Yoon, J. S., Leimkuehler, T., Theobalt, C., and Georgiev, ¨ I. Intrinsicedit: Precise generative image manipulation in intrinsic space. ACM Transactions on Graphics, 44(4), 2025.  
Ma, W.-C., Chu, H., Zhou, B., Urtasun, R., and Torralba, A. Single image intrinsic decomposition without a single intrinsic image. In ECCV, 2018.  
Magar, N., Hertz, A., Tabellion, E., Pritch, Y., Rav-Acha, A., Shamir, A., and Hoshen, Y. Lightlab: Controlling light sources in images with diffusion models. arXiv preprint arXiv:2505.09608, 2025.  
Mei, Y., He, M., Ma, L., Philip, J., Xian, W., George, D. M., Yu, X., Dedic, G., Tas¸el, A. L., Yu, N., et al. Lux post facto: Learning portrait performance relighting with conditional video diffusion and a hybrid dataset. In CVPR, 2025.  
Murmann, L., Gharbi, M., Aittala, M., and Durand, F. A multi-illumination dataset of indoor object appearance. In ICCV, Oct 2019.  
Pandey, R., Orts-Escolano, S., LeGendre, C., Haene, C., Bouaziz, S., Rhemann, C., Debevec, P., and Fanello, S. Total relighting: Learning to relight portraits for background replacement. In ACM Transactions on Graphics (Proceedings SIGGRAPH), volume 40, August 2021. doi: 10.1145/3450626.3459872.  
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.  
Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, ´ M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jegou, H., Labatut, P., and Bojanowski, P. DINOv3, 2025. ´ URL https://arxiv.org/abs/2508.10104.  
Vincent, P., Larochelle, H., Bengio, Y., and Manzagol, P.-A. Extracting and composing robust features with denoising autoencoders. In Proceedings of the 25th international conference on Machine learning, pp. 1096–1103, 2008.  
Wang, W., Yang, H., Fu, J., and Liu, J. Zero-reference lowlight enhancement via physical quadruple priors, 2024.  
Xi, C., Sida, P., Dongchen, Y., Yuan, L., Bowen, P., Chengfei, L., and Xiaowei, Z. Intrinsicanything: Learning diffusion priors for inverse rendering under unknown illumination. In ECCV, 2024.  
Xing, X., Qian, Y., Feng, S., Dong, Y., and Matas, J. Point cloud color constancy. In CVPR, 2022.  
Xing, X., Hu, V. T., Metzen, J. H., Groh, K., Karaoglu, S., and Gevers, T. Retinex-diffusion: On controlling illumination conditions in diffusion models via retinex theory. arXiv preprint arXiv:2407.20785, 2024.  
Xing, X., Groh, K., Karagolu, S., Gevers, T., and Bhattad, A. Luminet: Latent intrinsics meets diffusion models for indoor scene relighting. In CVPR, 2025.  
Yang, H.-H., Chen, W.-T., and Kuo, S.-Y. S3net: A single stream structure for depth guided image relighting. In CVPR, 2021.  
Zeng, C., Dong, Y., Peers, P., Kong, Y., Wu, H., and Tong, X. Dilightnet: Fine-grained lighting control for diffusionbased image generation. In SIGGRAPH, 2024a.  
Zeng, Z., Deschaintre, V., Georgiev, I., Hold-Geoffroy, Y., Hu, Y., Luan, F., Yan, L.-Q., and Hasan, M. Rgb-x: Image ˇ decomposition and synthesis using material-and lightingaware diffusion models. In SIGGRAPH, 2024b.  
Zhang, E., Cohen, M. F., and Curless, B. Emptying, refurnishing, and relighting indoor spaces. ACM ToG, 35(6), 2016.  
Zhang, L., Rao, A., and Agrawala, M. Scaling in-the-wild training for diffusion-based illumination harmonization and editing by imposing consistent light transport. In ICLR, 2025.  
Zhang, X., Gao, W., Jain, S., Maire, M., Forsyth, D., and Bhattad, A. Latent intrinsics emerge from training to relight. In NeurIPS, 2024a.  
Zhang, X., Jiang, R., Gao, W., Willett, R., and Maire, M. Residual connections harm generative representation learning. arXiv preprint arXiv:2404.10947, 2024b.  
Zhang, X., Yunis, D., and Maire, M. Deciphering’what’and’where’visual pathways from spectral clustering of layer-distributed neural representations. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4165–4175, 2024c.  
Zheng, B., Ma, N., Tong, S., and Xie, S. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.  
Zhou, T., Tucker, R., Flynn, J., Fyffe, G., and Snavely, N. Stereo magnification: Learning view synthesis using multiplane images. In SIGGRAPH, 2018.  
Zhu, J., Huo, Y., Ye, Q., Luan, F., Li, J., Xi, D., Wang, L., Tang, R., Hua, W., Bao, H., and Wang, R. I2-sdf: Intrinsic indoor scene reconstruction and editing via raytracing in

neural sdfs, 2023. URL https://arxiv.org/abs/ 2303.07634.

## A. Implementation details

## A.1. Training

All experiments are conducted on a node equipped with 8 NVIDIA A6000 Ada 48GB GPUs. The model is trained at a resolution of 512 ˆ 512 with an effective batch size of 64 (including gradient accumulation).

For Stage I, we employ a scene-aware sampler that ensures each batch contains images from the same scene captured under different lighting conditions. For Stage II and Stage III, we use standard random sampling across scenes. Additionally, in Stage III, we include a small proportion (10% of the training set) of identity relighting samples—where the input and target share the same lighting—to explicitly enforce geometry preservation.

Stage I is trained for 4 epochs, while Stage II and Stage III are trained for 2 epochs each. Due to differences in model size and parameter updates, Stage I training takes approximately 8 hours on a single GPU, while each of Stage II and III takes around 10 hours. We use the AdamW optimizer for all training stages.

## A.2. Lighting zoo generation

To enhance the model’s generalization to real-world inputs, we use the Stage II checkpoint to generate a large-scale in-thewild relighting dataset, referred to as the Lighting Zoo. We curate approximately 6,000 diverse images from open-source datasets including IIW (Bell et al., 2014), RealEstate-10K (Zhou et al., 2018), and DL3DV (Ling et al., 2024). Fig. S.1 shows a selection of pseudo-relit pairs; note the plausibility and diversity of the synthesized lighting.

During generation, images are randomly grouped into batches. Each original image serves as the input, while the target lighting condition is randomly sampled from other images within the same batch. For each scene, we generate seven relit versions under different lighting conditions. The process is distributed across multiple NVIDIA A6000 Ada 48GB GPUs for efficiency. In total, generating the Lighting Zoo requires approximately 500 GPU hours.

## A.3. User study

We conduct two user studies: (1) comparing real-world performance of LumiNet, Latent Intrinsics, and our method; and (2) a stepwise assessment of our training pipeline’s phases. Participants view two relit images—from different methods (Study 1) or different pipeline stages (Study 2)—and select the one with superior lighting alignment, identity preservation, and lighting realism. 23 participants completed all comparisons. Study 1 had each participant judge 9 image pairs across the three methods; Study 2 had each judge 12 pairs across four stages (LumiNet, Stage I, Stage I&II, All Stages). All pairs were shown side-by-side at full resolution in randomized order with win/tie/lose selection.

## B. Semantic alignment in relighting

## B.1. Semantic group curation

The MIIW dataset (Murmann et al., 2019) provides detailed semantic annotations, including binary masks for 31 distinct material labels. To efficiently evaluate relighting performance across materials with varying reflectance properties, we regroup these fine-grained labels into five broader categories: Diffuse, Glossy, Specular, Metallic, and Uncertain/Mixed.

To ensure consistency and physical relevance, we utilize advanced reasoning model GPT-O3 with prompt - ”cluster those class label into a higher-level BRDF-style categories.” - to assist in the semantic reasoning and grouping process. The full mapping between original material labels and grouped categories is provided in Table S.1.

## C. More Visual Results

We present additional visual results on the MIIW dataset and in-the-wild images. Fig. S.2 illustrates the same scene relit under different target lighting conditions. Fig. S.3 demonstrates relighting on in-the-wild examples.

Benefiting from the proposed augmented latent-intrinsics feature, our method is capable of relighting a wide variety of scenes while preserving geometry and scene identity. These include: 1) a real indoor image captured by a phone, 2) a painting, and 3) an internet-sourced image.

Table S.1. Mapping of material classes to reflectance clusters. Each material class is assigned a cluster label based on its dominant reflectance properties: Diffuse, Glossy, Specular, Metallic, or Uncertain. The cluster name is shown only at the start of each group for clarity.

<table><tr><td>Cluster</td><td>Class Index</td><td>Class Label</td></tr><tr><td rowspan="20">Diffuse</td><td>3</td><td>Cardboard</td></tr><tr><td>5</td><td>Concrete</td></tr><tr><td>6</td><td>Cork/corkboard</td></tr><tr><td>7</td><td>Dirt</td></tr><tr><td>8</td><td>Fabric/cloth</td></tr><tr><td>9</td><td>Foliage</td></tr><tr><td>10</td><td>Food</td></tr><tr><td>11</td><td>Fur</td></tr><tr><td>14</td><td>Laminate</td></tr><tr><td>16</td><td>Linoleum</td></tr><tr><td>21</td><td>Paper/tissue</td></tr><tr><td>25</td><td>Sponge</td></tr><tr><td>26</td><td>Styrofoam</td></tr><tr><td>29</td><td>Wallpaper</td></tr><tr><td>31</td><td>Wicker</td></tr><tr><td>32</td><td>Wood</td></tr><tr><td>33</td><td>Stone</td></tr><tr><td>34</td><td>Chalkboard/blackboard</td></tr><tr><td>35</td><td>Carpet/rug</td></tr><tr><td>36</td><td>Brick</td></tr><tr><td rowspan="6">Glossy</td><td>4</td><td>Ceramic</td></tr><tr><td>15</td><td>Leather</td></tr><tr><td>23</td><td>Plastic — opaque</td></tr><tr><td>24</td><td>Rubber/latex</td></tr><tr><td>27</td><td>Tile</td></tr><tr><td>30</td><td>Wax</td></tr><tr><td rowspan="3">Specular</td><td>12</td><td>Glass</td></tr><tr><td>18</td><td>Mirror</td></tr><tr><td>22</td><td>Plastic - clear</td></tr><tr><td>Metallic</td><td>17</td><td>Metal</td></tr><tr><td rowspan="12">Uncertain / Mixed</td><td>0</td><td>unassigned</td></tr><tr><td>1</td><td>I can’t tell</td></tr><tr><td>2</td><td>More than one material</td></tr><tr><td>13</td><td>Granite/marble</td></tr><tr><td>19</td><td>Not on list</td></tr><tr><td>20</td><td>Painted</td></tr><tr><td>27</td><td>Tile</td></tr><tr><td>28</td><td>splitshape</td></tr><tr><td>37</td><td>Skin</td></tr><tr><td>38</td><td>Water</td></tr><tr><td>39</td><td>Hair</td></tr><tr><td>40</td><td>no_consensus</td></tr></table>

![](images/4edb754053584db823c3b9f2f1d3a3f8831d48d0edd875584cadb7e039b14db0.jpg)  
Figure S.1. Lighting Zoo. Pseudo-relit image pairs generated by our method for the third-stage refinement.

![](images/ee94043d88dcc1fbce774fd6087dfc94a9a762b064fb6fa7f3b5f7aa800f457e.jpg)  
Figure S.2. Scene relighting comparison on the MIIW (Murmann et al., 2019) dataset. Each row corresponds to a different target lighting condition (left). We compare the relighting outputs of RGB-X, LumiNet, and our method against the ground truth (with bypass decoder disabled).

Input  
![](images/67b6748fa4a20c404791965ace9c004b0184b5bd122998fdf429f717440f3d6c.jpg)  
Relighting 1  
Relighting 2

![](images/a82c2fadbdb72d71b816723ee5141d16ccdc3b41c153f2fd60eb029be46d16e8.jpg)  
Figure S.3. In-the-wild relighting. Our method is capable of relighting a wide variety of scenes while preserving geometry and scene identity. These include: 1) a real indoor image captured by a phone, 2) a painting, and 3) an internet-sourced image. The target lightings are indicated in the bottom left corner of each image.