# MRT: Masked Region Transformer for Layered Image Generation and Editing at Scale

Zhicong Tang†\* Zhao Zhang† Jingye Chen Mohan Zhou Yifan Pu Yuchi Liu Yalong Bai Ethan Smith Yuhui Yuan Canva Research

![](images/313664ff2efc7516f6ecefb9b50a6d2f56f7a20c081c94a7fb88d82c8dda6b4c.jpg)  
Thecentralfocalpointisacharmingtwo-story Englishcountrysidecottagefeaturinga distinctivethatchedroof.Therofisthickand textured,golden-brownincolorwithgetle curves thatdrape over"eyebrow"dormer windowson thesecondfloor.Two tall rectangularbrick chimneysrisefromthe rooflineTeoagewallsreeacmy white stucco..   
<layer>Majesticeaglein flightwithspread wings.,deepindigo blue vintageengraving style. Cross-hatchedfeathers with fluid inklinesand whitehighlights fora glossyfinish.c/layer>

Figure 1. Overview of Masked Region Transformer capabilities. Our framework supports four tasks: (1) multilingual text-to-layers generation, (2) image-to-layers decomposition (including natural images), (3) layer addition, and (4) layer restylization for user-provided layers.

# Abstract

Layered image generation and editing is a fundamental capability that enables layer-wise reuse, editing, and composition of generated visual content, analogous to word-level editing in natural language. Despite its importance, this remains an underexplored area at scale. To address this gap, we present MRT, a 20B-parameter masked region diffusion model tailored for multi-layer transparent image generation and editing, trained on over 10M multilingual design samples spanning diverse aspect ratios and textual prompts. To fully leverage this scale, we make two key technical contributions. First, we unify three complementary tasks—textto-layers, image-to-layers, and layers-to-layers—within a shared masked region diffusion framework, where selec-

tive token masking enables flexible layer-wise generation and editing. Second, to enable overflow layer generation, we introduce an overflow-aware canvas layer that handles boundary inconsistencies and supports semi-transparent background synthesis, enabling complete editable layers extending beyond visible canvas boundaries. Additionally, we apply diffusion distillation to achieve 8-step, realtime multi-layer generation with minimal quality degradation. Extensive experiments demonstrate that our framework substantially outperforms prior state-of-the-art approaches, including various commercial systems, across all three tasks, establishing a new benchmark for multi-layer transparent image generation. Notably, our model significantly outperforms the concurrent Qwen-Image-Layered model in image-to-layers quality according to user-study results, while achieving 10∼100× faster inference and saving a 50%∼90% activation GPU memory consumption during

# 1. Introduction

Text-to-image generation has achieved remarkable quality improvements in recent years through various technological advances, including large-scale diffusion transformers [9, 36, 37], distributed training on billions of highquality text-image pairs [13, 14, 43, 55], rectified flow matching [9, 30] that transforms simple prior distributions into complex data distributions via straight paths, distribution matching distillation [12, 33, 34, 41, 60, 61, 65–67] for accelerated inference, and advanced text encoder architectures [14, 31, 32]. In contrast, generative models for layered image generation [5, 17, 21, 22, 28, 38, 48, 62, 63] remain significantly underdeveloped. This gap primarily stems from two factors: the absence of large-scale, highquality datasets comparable to LAION-5B [42], and limited exploitation of prior knowledge from state-of-the-art opensource text-to-image models. These constraints have hindered systematic exploration of critical research directions in layered image synthesis.

We address this fundamental research gap through a comprehensive study on a high-quality, large-scale multilayer dataset comprising over ≥ 10 million samples—an order of magnitude larger than recent work [38]. Our dataset spans diverse resolutions and aspect ratios, encompassing over 43 million unique layers and over 7 million unique oversized visual elements to support overflow layer generation. We employ GPT-5 mini to generate global captions for all graphic designs. For visual text layers, we utilize ground-truth typography attributes, ensuring comprehensive high-quality annotations. To fully leverage this dataset at scale, we build our multi-layer generative model by implementing the masked region transformer on Qwen-Image [55], the largest open-source text-to-image diffusion model with approximately ∼ 20B parameters.

To advance the efficiency of layered image generation and editing during both training and inference, we introduce the following key technical contributions: First, we propose a unified masked region transformer framework that handles three complementary tasks: text-to-layers, image-to-layers, and layers-to-layers generation and editing. The key innovation lies in our adaptive masking mechanism, which determines whether to initialize each layer from clean latents or noise based on the specific task requirements. Second, our masked region transformer operates directly on the fullsize canvas by treating the background as a special transparent foreground layer and encapsulating overflow layers that extend partially beyond the background region. This architecture ensures that all foreground layers maintain full reusability and can be arbitrarily repositioned on the canvas, which is illustrated in Figure 3 and experimental section. Third, we further propose leveraging distribution matching distillation schema to develop a few-step multi-layer generator with minimal quality degradation.

We conduct thorough ablation experiments to study the effects of different components. We empirically demonstrate that scaling both the model and dataset elevates performance to a new level, and that joint multi-task training further enhances performance while improving the user experience. We show that our image-to-layers task generalizes exceptionally well to various out-of-domain design images and natural images. Our layers-to-layers task readily supports multi-image fusion, seamlessly integrating any given user image into an existing design. We hope our masked region transformer advances the understanding of this fundamentally challenging task at an unprecedented scale.

# 2. Related Work

Layered image generation and editing task follows two paradigms: simultaneous generation (Text2Layer [63], LayerDiff [17], ART [38], PrismLayer [5], Qwen-Image-Layered [59]) and sequential generation (LayerDiffuse [62], COLE [22], OpenCOLE [21], LayerD [45]). Related layout generation and control methods fall into two categories: (1) generating layouts from visual elements [2– 4, 6, 7, 10, 11, 15, 18–20, 23–25, 27, 44, 46, 53, 54, 56, 58], and (2) controlling generation via spatial conditioning [1, 4, 10, 26, 29, 40, 47, 51, 52, 57, 64]. Compared to the most closely related work, ART [38] and Qwen-Image-Layered [59], our masked region transformer unifies three tasks: text-to-layers, image-to-layers, and layers-tolayers generation. We further introduce native support for overflow layers and enable few-step multi-layer generation through distillation.

# 3. Approach

# 3.1. Scaling-up Layered Data and Diffusion Model

Scaled Layered Dataset. The scarcity of large-scale, highquality multi-layer transparent images presents a fundamental challenge for advancing multi-layer generative modeling. Rather than relying on noisy, uncurated internet sources, we construct a curated in-house dataset comprising over 10M multi-layer graphic designs from one of the world’s largest graphic design platforms. All designs are created by professional designers and fully licensed for generative model training. Figure 2 illustrates key dataset statistics, showing that our dataset spans diverse aspect ratios and resolutions while supporting multilingual visual text rendering and bilingual text prompts.

Scaled Region Transformer. To incorporate the generation of overflow layers, we follow ART [38] to perform the denoising diffusion process in a regional manner as follows: First, we represent a multi-layer transparent image as $\{ \mathbf { I } _ { \mathrm { c a n v a s } } , \mathbf { I } _ { \mathrm { b g } } , \{ \mathbf { I } _ { \mathrm { f g } } ^ { i } \} _ { i = 1 } ^ { K } \}$ , where $\mathbf { I } _ { \mathrm { c a n v a s } }$ is the composed image on the full-size canvas, $\mathbf { I } _ { \mathrm { b g } }$ is a semi-transparent RGBA background layer, and $\{ \mathbf { I } _ { \mathrm { f g } } ^ { i } \} _ { i = 1 } ^ { \bar { K } }$ are K RGBA foreground layers. Second, we perform the diffusion process on a merged image that integrates the fully transparent canvas as the base layer and overlays $\mathbf { I } _ { \mathrm { b g } }$ and all ${ \bf I } _ { \mathrm { f g } } ^ { i }$ layers according to a predefined layout. Third, we use the WAN-2.1- VAE [50] encoder to extract the regional cropped representations for all foreground layers, the representation of the background layer, and the representation of the composed full design. Last, we implement an anonymous regional diffusion transformer [38] with 20B parameters following Qwen-Image [55] to perform full attention jointly on these regional foreground layer tokens, background layer tokens, and composed full design image tokens.

![](images/199df62820239b7db58d5d4e7a916bd2b25f63d93095003135404678b0c0cb32.jpg)  
[a)# of layers

![](images/188009ce2ae6b8899b9d9a0115a88b80a6801140ddc3f1fd02d167f0a04d8f66.jpg)  
(b]#of text layers

![](images/772f30bd28e3eb8d1e51916a9bc0d7709c1178284acc00b5159abae1357e0560.jpg)  
[c] Languages

![](images/f5b608cd32ccb3bd57a8d0c3597d80a8735bd55224510cc3e9e20d54c417cf56.jpg)  
[d) Layer Types

![](images/54326d40468ff9ab60c0f283bcee2f0b3e0003fdd11d8024011e705b6b2d5fbe.jpg)  
[e] # of tokens w/o overflow

![](images/b87db439477eeec44889cf18c497d8580906db68e51eb802fec16008664ed1a4.jpg)  
[f] # of tokens w/ overflow

![](images/bd118a8da4e01235b5a32def133a09d6d00ba79d5ce4ea4ab4620f4bc8d176ee.jpg)  
[g] aspect ratio

Figure 2. Illustrating the dataset statistics. Figures (a) and (b) show the distribution of the number of unique layers per design. Figures (c) and (d) show the distribution of different languages in visual text and the distribution of different layer types, respectively. Figures (e) and (f) show the distribution of total visual token counts for all transparent layers before and after supporting overflow layers. Figure (g) shows the distribution of width-to-height aspect ratios.   
![](images/2b850fb924aff35ca3b8bc4e09eba2a6c297c2489d08b4d6306d849287f8108a.jpg)

<details>
<summary>text_image</summary>

What I feel like
you felt like to be?
A HAPPY
Anniversary
Will You Be My Date
on Valentine's Day?
A 30-Day, Lovely Valentine's Day Programist
TORNEO
DOMA
VAQUERA
SINCEMBER 2018
DISCOUNT
75% OFF
Spring
FLASH SALE!
DISCOUNT
75% OFF
Spring
FLASH SALE!
DISCOUNT
75% OFF
</details>

Figure 3. Illustrating the overflowing layers. The first row visualizes the canvas layer with a fully transparent background, exposing pixels beyond the main background region. Rows 2-3 compare multi-layer generation without overflow support (baseline) and with overflow support (ours). Full-size overflow layer generation is essential for maintaining complete editability and reusability, preventing layer content from being truncated at background boundaries.

Overflow Layer Support. Previous work [5, 38] generates foreground layers only within the visible canvas region, producing incomplete elements that extend beyond background boundaries. This limits layer reusability, as shown in the second row of Figure 3. However, we find that over 60% of samples in our training set contain overflow layers, making this a critical practical concern. To address this, we introduce an additional full-size canvas layer that supports generation of complete semi-transparent backgrounds

and overflowing elements. This is feasible since we have access to ground-truth complete layers for all samples in our dataset. This design is essential for practical editing workflows: without it, layers extending beyond the canvas would be cropped and rendered non-editable, severely limiting their usability in downstream compositional tasks. Figure 3 shows representative overflow layer examples from our dataset (first row) and compares layered samples with and without overflow layer support (second and third rows).

# 3.2. Masked Region Transformer

We illustrate how our masked region diffusion transformer framework addresses three challenging multi-layer generation tasks—Text-to-Layers, Image-to-Layers, and Layersto-Layers—in a unified manner in Figure 4. The key insight is to conditionally mask either the global image tokens or the combination of reference tokens and existing layer tokens within the regional diffusion transformer. Masked latents denote clean tokens encoding pre-existing conditions, with noise injection and diffusion supervision applied exclusively to non-masked tokens. We apply full attention between masked clean tokens and noise tokens, enabling the model to adaptively learn their relationships across different tasks. The detailed masking mechanism for each task is described as follows:

Text-to-Layers. The text-to-layers generation task aims to synthesize a multi-layer transparent design from a text prompt c, comprising a canvas layer $\mathbf { I } _ { \mathrm { c a n v a s } }$ , a semitransparent background layer $\mathbf { I } _ { \mathrm { b g } }$ , and K foreground layers $\{ \mathbf { I } _ { \mathrm { f g } } ^ { i } \} _ { i = 1 } ^ { K }$ 1 that compose into Icomposed with overflow support. The canvas layer defines the full design dimensions to accommodate overflowing elements and is fully transparent by construction. Thus we apply diffusion to the concatenation of latents $[ \mathbf { z } _ { \mathrm { c o m p o s e d } } ; \mathbf { z } _ { \mathrm { b g } } ; \{ \mathbf { z } _ { \mathrm { f g } } ^ { i } \} _ { i = 1 } ^ { K } ]$ , excluding the canvas layer, conditioned on shared text embeddings c. Following [38], we include $\mathbf { z } _ { \mathrm { c o m p o s e d } }$ to ensure layer coherence. Since no pre-existing layers exist, we set masked token $\mathbf { z } _ { \mathrm { m a s k } }$ as ∅. See Figure 4 (panel 1) for details.

Let $\mathbf { z } _ { 0 } = [ \mathbf { z } _ { \mathrm { c o m p o s e d } } ; \mathbf { z } _ { \mathrm { b g } } ; \mathbf { z } _ { \mathrm { f g } } ^ { 1 } ; \ldots ; \mathbf { z } _ { \mathrm { f g } } ^ { K } ]$ denote the concatenation of all non-masked clean latents, and $\epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ denote the noise prior. The flow matching framework learns a vector field that transports samples from the noise distribution to the data distribution through a continuous-time interpolation path. At time-step $t \in [ 0 , 1 ]$ ], the interpolated latent is given by:

![](images/5b07525338e3d11e04c7227a4cf7a9e0bc7d679f23b6958fb12ccf8e07b0272e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Text-to-Layers"] --> B["Decoder"]
    C["Image-to-Layers"] --> D["Decoder"]
    E["Layers-to-Layers"] --> F["Decoder"]
    G["Layout Addition"] --> H["Decoder"]
    I["Layer Restylization"] --> J["Decoder"]
    K["Unified Modeling with Masked Region Diffusion Transformer"] --> L["Layout"]
    L --> M["Masked latents z_mask"]
    L --> N["noised latents z_mask"]
    M --> O["Encoder"]
    N --> P["Encoder"]
    Q["Qwen-VL global text prompt"] --> R["Layout Detector"]
    R --> S["image to decompose"]
    T["Layout"] --> U["Masked latents z_mask"]
    T --> V["noised latents z_mask"]
    U --> W["Audiovisual Image"]
    V --> X["Audiovisual Image"]
    Y["Encoder"] --> Z["Layer prompt"]
    AA["Encoder"] --> AB["Layer prompt"]
    AC["Layout"] --> AD["Masked latents z_mask"]
    AC --> AE["noised latents z_mask"]
    AD --> AF["Audiovisual Image"]
    AE --> AG["Audiovisual Image"]
    AH["Qwen-VL layer prompt"] --> AI["Layer prompt"]
    AJ["Qwen-VL"] --> AK["Layer prompt"]
    AL["Qwen-VL"] --> AM["Layer prompt"]
    AN["Qwen-VL"] --> AO["Layer prompt"]
```
</details>

Figure 4. Illustrating the Masked Region Transformer framework. We unify three different tasks including text-to-layers, image-to-layers, and layersto-layers with a shared masked regional diffusion transformer. $L e f t { \mathrm { : } }$ Text-to-Layers directly transforms a stack of noise latents into a set of transparent layers and a composed canvas image (panel #1). We add noise to the latents of all transparent layers during training. Middle: Image-to-Layers aims to decompose a raster image into a set of high-quality transparent layers. We set masked latents to the noise-free global image tokens and apply the diffusion process to layer tokens corresponding to spatial regions defined by either automatic layout detector or manual annotation (panel #2). Right: Layers-to-Layers enables two editing capabilities: (i) generating new layers from layer prompt conditioned on existing ones (panel #3), and (ii) transforming reference images into layers with visual style aligned to the existing composition (panel #4). In both layer addition and layer restylization scenarios, we define masked tokens as the noise-free latent representations of the reference content, existing layers, and global composition. Some text layers are omitted for clarity.

$$
\mathbf {z} _ {t} = (1 - t) \mathbf {z} _ {0} + t \epsilon , \tag {1}
$$

We train the diffusion model $f _ { \theta }$ predicts the flow velocity conditioned on the interpolated latent $\mathbf { z } _ { t } ,$ time-step $t ,$ and text prompt t: $\hat { \mathbf { v } } = f _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { c } )$ . The training objective minimizes the mean squared error between the predicted and ground-truth velocity:

$$
\mathcal {L} _ {\mathrm{flow}} = \mathbb {E} _ {\mathbf {z} _ {0}, \epsilon \sim \mathcal {N} (\mathbf {0}, \mathbf {I}), t} \left[ \| \mathbf {v} _ {t} - f _ {\theta} (\mathbf {z} _ {t}, t, \mathbf {c}) \| ^ {2} \right], (2)
$$

where the ground-truth velocity $\mathbf { v } _ { t }$ along the interpolation path is $( \mathbf { z } _ { 0 } - \epsilon )$ , the expectation is taken over the clean latents $\mathbf { z } _ { 0 } .$ , random noise $\epsilon ,$ and uniformly sampled time-steps t.

Image-to-Layers. The image-to-layers task has emerged as a critical capability in commercial generative systems, with products such as Adobe Firefly’s Layered Image Editing and Lovart’s Edit Elements recently introducing support for this functionality. The image-to-layers task aims to decompose a raster image $\mathbf { I } _ { \mathrm { i n p u t } }$ (or Icomposed) into a multi-layer transparent design comprising a canvas layer Icanvas, a background layer $\mathbf { I } _ { \mathrm { b g } }$ and $K$ foreground layers $\{ \mathbf { I } _ { \mathrm { f g } } ^ { i } \} _ { i = 1 } ^ { K }$ , conditioned on a target layout specifying each layer’s spatial location and an optional text prompt for semantic guidance. This task inherently involves two subtasks: segmentation to identify layer regions with accurate alpha masks and inpainting to complete occluded areas. We either use human annotations or a layout detector to extract the target layout from the input raster image.

The masked clean tokens $\mathbf { z } _ { \mathrm { m a s k } }$ are set to the global composed image representation $\bf { z } _ { \mathrm { c o m p o s e d } } .$ , encoding the conditional image targeted for decomposition. We add noise to the concatenation of the non-masked tokens $\begin{array} { r l } { \mathbf { z } _ { 0 } } & { { } = } \end{array}$ $[ \mathbf { z } _ { \mathrm { b g } } ; \mathbf { z } _ { \mathrm { f g } } ^ { 1 } ; \dots ; \mathbf { z } _ { \mathrm { f g } } ^ { K } ]$ . Through the regional diffusion process, the diffusion model $f _ { \theta }$ is trained to extract all transparent layers conditioned on the given global image and layout. Since requiring users to provide designs with overflow layers is impractical, we instead use the latent encoding of pixels located within the visible canvas. See Figure 4 (panel 2) for details.

We observe that individual layers often exhibit structural ambiguity and can be further decomposed. To address this, we propose layer grouping augmentation, which randomly groups overlapping or adjacent layers during training. This strategy increases structural diversity, improves robustness to ambiguous boundaries, and enhances generalization to out-of-domain images with noisy layouts.

Layers-to-Layers. To enable a flexible, layer-wise interaction experience, we frame the layered image editing task as a layer-to-layer task that covers two key scenarios: (i) layer addition, which generates new coherent layers from text prompts conditioned on existing layers while maintaining spatial and stylistic consistency across the composition;

and (ii) layer restylization, which focuses on transforming any user-provided images or transparent layers into stylistically aligned layers that match the appearance and visual identity of the existing composition.

To model the layers-to-layers task, we retain existing layer latents as masked clean tokens $\mathbf { z } _ { \mathrm { m a s k } }$ and apply diffusion only to: (i) newly added layers conditioned on text prompts, or (ii) designated layers conditioned on visual references for restylization. Given the challenge of constructing training data for these scenarios, we randomly select a subset of layers from each design to serve as conditional existing layers, treating the remaining layers as generation targets. For layer restylization training, we use Image editing model to transfer the style of non-selected layers, creating style-transformed variants as training pairs. See the appendix for details on the dataset construction pipeline.

Formally, in the layer addition task, we aim to synthesize a subset of foreground layers conditioned on the remaining layers and layer-level textual descriptions. We apply diffusion to the latent token sequence $[ \mathbf { z } _ { \mathrm { c o m p o s e d } } ; \mathbf { z } _ { \mathrm { b g } } ; \{ \mathbf { z } _ { \mathrm { f g } } ^ { i } \} _ { i = 1 } ^ { K } ] ,$ where $\mathbf { z } _ { \mathrm { c o m p o s e d } }$ encodes the alpha-composited context formed by the background and all non-target layers. Let $A \subseteq \{ 1 , \dots , K \}$ denote the indices of layers to be generated (an arbitrary subset, not necessarily contiguous). We set the masked clean tokens as $\begin{array} { r l } { \mathbf { z } _ { \mathrm { m a s k } } } & { { } = } \end{array}$ [zcomposed; $\mathbf { z } _ { \mathrm { b g } } \colon \{ \mathbf { z } _ { \mathrm { f g } } ^ { i } \} _ { i \notin A } ]$ , and treat the target slots $\begin{array} { r l } { \mathbf { z } _ { 0 } } & { { } = } \end{array}$ $[ \{ \mathbf { z } _ { \mathrm { f g } } ^ { i } \} _ { i \in A } ]$ as the non-masked tokens to be noised and denoised. The text condition $\mathbf { c } _ { A }$ is derived from a layercaption prompt constructed by concatenating <layer> $c _ { i }$ $< / 1$ ayer> for all $i \in A$ in layer order, where $c _ { i }$ is the caption of layer i. During training, we add noise to $\mathbf { z } _ { 0 }$ and optimize the flow-matching objective conditioned on $\left( \mathbf { z } _ { \mathrm { m a s k } } , \mathbf { c } _ { A } \right)$ ; during inference, we initialize $\mathbf { z } _ { 0 }$ from noise and denoise it under the same conditions, yielding the added layers in their original indices.

In the layer restylization task, we update a user-uploaded layered design by restylizing selected layers under additional appearance conditions while preserving the remaining layers. Given target indices ${ \mathcal { T } } \subseteq \{ 1 , \ldots , K \}$ , we construct $\mathbf { z } _ { \mathrm { c o m p o s e d } }$ by compositing the background with the non-target original layers $\{ \mathbf { z } _ { \mathrm { f g } } ^ { i } \} _ { i \notin \mathbb { Z } }$ , and keep ${ \bf z } _ { \mathrm { m a s k } } =$ $[ \mathbf { z } _ { \mathrm { c o m p o s e d } } ; \mathbf { z } _ { \mathrm { b g } } ; \{ \mathbf { z } _ { \mathrm { f g } } ^ { i } \} _ { i \notin \mathcal { T } } ]$ as masked clean conditions. For each $i \in \mathcal { T } ,$ , we are additionally given a conditional latent $\mathbf { z } _ { \mathrm { c o n d } } ^ { i }$ that specifies the desired appearance of layer i. We append $\{ \mathbf { z } _ { \mathrm { c o n d } } ^ { i } \} _ { i \in \mathcal { I } }$ as extra conditioning tokens and treat them as masked, so they are not prediction targets. To make this role explicit, we add a learnable condition-token embedding to the appended conditional tokens. We further copy the RoPE positional encoding from the corresponding original layer token to its conditional token, ensuring that the two tokens share identical spatial positional cues. Accordingly, we apply diffusion only to the nonmasked original target slots $\mathbf { z } _ { 0 } ~ = ~ [ \{ \mathbf { z } _ { \mathrm { f g } } ^ { i } \} _ { i \in \mathcal { T } } ]$ , conditioned on $[ \mathbf { z } _ { \mathrm { m a s k } } ; \{ \mathbf { z } _ { \mathrm { c o n d } } ^ { i } \} _ { i \in \mathcal { T } } ]$ and a fixed instruction prompt such as Harmonize these layers. During training, noise is added only to $\mathbf { z } _ { 0 }$ and the model is trained to denoise the original target slots under the conditional latents. During inference, we initialize $\mathbf { z } _ { 0 }$ from noise and denoise it under the same conditions, reading the final restylized layers from the original target slots while excluding the appended conditional tokens from the output layer set.

# 3.3. Accelerated Multi-Layer Generator

We adopt the improved distribution matching distillation (DMD) technique [8, 35, 60, 61] to compress our multistep diffusion model (teacher) into a few-step generator (student) while maintaining distributional consistency between the teacher and student models. Let the teacher model $f _ { \boldsymbol { \theta } _ { T } } ( \mathbf { z } _ { t - 1 } | \mathbf { z } _ { t } )$ denote the reverse process of a standard multi-step diffusion model, and let the student model $f _ { { \boldsymbol { \theta } } _ { S } } ( \mathbf { z } _ { t - 1 } | \mathbf { z } _ { t } )$ approximate it using fewer denoising steps. The objective of DMD is to minimize the Kullback–Leibler (KL) divergence between the teacher and student transition distributions:

$$
\mathcal {L} _ {\mathrm{DMD}} = \mathbb {E} _ {\mathbf {z} _ {0} \sim p _ {d a t a}, t \sim \mathcal {U} (1, T)} \left[ D _ {\mathrm{KL}} \big (f _ {\theta_ {T}} (\mathbf {z} _ {t - 1} | \mathbf {z} _ {t}) \parallel f _ {\theta_ {S}} (\mathbf {z} _ {t - 1} | \mathbf {z} _ {t}) \big) \right]. \tag {3}
$$

During inference, the distilled student model performs generation in a reduced number of steps $T _ { S } ~ \ll ~ T _ { T }$ , effectively approximating the teacher’s multi-step trajectory: $\mathbf { z } _ { t - 1 } = f _ { \theta _ { S } } ( \mathbf { z } _ { t } , t )$ , where we set $t = T _ { S } , \dots , 1$ . We show that the distilled model preserves the sample quality of the teacher while substantially reducing the number of sampling steps, resulting in faster and more efficient generation. We also support various techniques, such as CacheDiT and sequence parallelization across multiple GPUs, to further accelerate inference speed.

# 4. Experiment

# 4.1. Implementation Details

We conduct all experiments using Qwen-Image as our base architecture, consisting of 60 layers with a hidden dimension of 3584 and 24 attention heads per layer. We initialize model weights from the open-source pretrained checkpoint available on HuggingFace. Unlike previous approaches [5, 38] that fine-tune only LoRA [16] weights due to resource constraints, we perform full-parameter finetuning with FSDP2 to explore the model’s performance upper bound. This approach is necessary given the significant distribution shift from standard flat image generation and the inherent complexity of multi-layer synthesis.

For ablation experiments, we train on a curated subset of 0.5M layered designs for 4,000 iterations at $5 1 2 \times 5 1 2$ resolution using 8×H200 GPUs with the batch size 16 per GPU and 128 globally. We use the AdamW optimizer with a constant learning rate of $1 \times 1 0 ^ { - 4 }$ . For system-level experiments, we employ two-stage training: ∼70,000 iterations at $5 1 2 \times 5 1 2$ on the full 10M dataset, followed by ∼20,000 iterations at $1 0 2 4 \times 1 0 2 4$ . This progressive strategy allows the model to first establish multi-layer decomposition capabilities before scaling to high resolution. Training uses 64×H200 GPUs with batch size 16 per GPU and 1,024 globally.

# 4.2. Evaluation Protocol

Benchmark. We compare our approach with previous state-of-the-art methods on DESIGN-MULTI-LAYER-BENCH, introduced by ART [38], which is curated from the VistaCreate graphic design platform [49]. However, this evaluation dataset does not include overflow layers. To address this gap, we construct OVERFLOWERFLOW-DESIGN-BENCH to evaluate the model’s ability to generate complete layers from full layouts, which is essential for ensuring overflow layer reusability.

Metrics. We evaluate model performance from multiple perspectives. For merged image quality, we report $\mathrm { P S N R } _ { \mathrm { l a y e r } }$ , $\mathrm { S S I M _ { l a y e r } } ;$ , $\mathrm { P S N R } _ { \mathrm { m e r g e d } }$ , $\mathrm { S S I M } _ { \mathrm { m e r g e d } } .$ , FIDmerged (measuring overall coherence), and FID following [38]. Since our layer is RGBA images with transparency, we only compute on non-transparent pixels as $\mathrm { P S N R } _ { \mathrm { l a y e r } }$ and $\mathrm { S S I M _ { l a y e r } . }$ . For human evaluation, we collect multidimensional user preferences on a subset of DESIGN-MULTI-LAYER-BENCH for the text-to-layers (T2L) task and image-to-layers (I2L) task, reflecting real user experience. The evaluation protocol and interface are described in the supplementary material.

# 4.3. Main Results

# 4.3.1. Text-to-Layers: Comparison with SoTAs

We compare our method with ART [38] on a subset of DESIGN-MULTI-LAYER-BENCH. In our user study illustrated in Fig. 6, participants consistently preferred our results over ART in instruction following, overall aesthetics, and layer quality. These findings indicate stronger alignment between prompts and layered compositions, further illustrated in Fig. 5 by layouts that better preserve spatial intent and stylistic consistency.

Only our method natively supports generating overflow RGBA layers that extend beyond the background boundary on a full-size canvas, preserving editability and reuse; prior systems $( e . g . , \mathrm { A R T } )$ restrict pixels to the background region, leading to cropped or missing content. See Fig. 7 and Fig. 9 for a visual results.

# 4.3.2. Image-to-Layers: Comparison with SoTAs

In a user study comparing the layer decomposition capabilities of the latest work LayerD [45] and commercial systems like RoboNeo and Lovart, participants consistently preferred our method for layer quality, content integrity, and decompose granularity. Since I2L evaluation assumes a layer layout (bounding boxes with Z-order), we evaluate our method with the layout extracted by a z-order-aware detector (details in the supplementary). Qualitative comparisons in Fig. 20 also show that our method produces more complete, reusable RGBA layers with sharper boundaries. We further demonstrate the generalization of our model to natural scenes in Fig. 24.

# 4.3.3. Image-to-Layers: Comparison with con-current Qwen-Image-Layered

Recently, Qwen-Image-Layered [59] has attracted significant interest from the community since its release on Huggingface, due to its strong generalization capability on various design images. We demonstrate the advantages of our approach by conducting rigorous comparisons from three aspects: quality, latency, and memory.

Better Quality. We first construct an out-of-domain test set consisting of 100 creative designs obtained from three sources: images generated by the latest Nano-Banana-Pro (and Ideogram 3.0) image generation model and test images from the official Qwen-Image-Layered repository [39]. We report the quantitative comparison results in Table 1, which shows that our approach achieves significantly higher $\mathrm { S N R } _ { \mathrm { m e r g e d } }$ and $\mathrm { S S I M } _ { \mathrm { m e r g e d } }$ . We calculate the metrics across three groups based on the number of layers, and our MRT consistently performs better across all groups.

Fig. 10, Fig. 11, Fig. 12, Fig. 13, Fig. 14, Fig. 15, Fig. 16, and Fig. 17 provide further qualitative comparison results. We empirically find that our approach performs substantially better when required to decompose flat designs into an increasing number of transparent layers; our approach continues to perform well, while Qwen-Image-Layered struggled to assign meaningful objects to each layer. These visual results not only echo the above findings but also show that significant room for improvement remains, even though our model substantially outperforms Qwen-Image-Layered. We also conduct an apple-to-apple user study on this test set, with results reported in Fig. 8. Our approach achieves win rates of 79.5%, 68.9%, and 82.6% for layer quality, integrity, and granularity, respectively.

Lower Latency. As shown in Fig. 19, due to our regional diffusion transformer architecture, we achieve significant speedup compared to Qwen-Image-Layered, which uses the same number of full-resolution tokens to model each transparent layer regardless of their actual area within the canvas. We achieve similar latency speed-up as the statistics shown in ART [38] and we further applied various advanced cache techniques, model distillation, lower-precision, parallel inference to optimize the latency of our model to within ∼ 3 seconds when running with 4× H100 GPUs and ∼ 6 seconds on a single H100 GPU when required to decompose a single 1K high-resolution image into nearly 20 transparent

![](images/0be5f27bd51650cceac6039909a69353cd662ce8c891b2d4489265521a36d19c.jpg)

<details>
<summary>text_image</summary>

Adam
Noah
YOUR LIT PARTY
Let Stuck Alum
NEW
COLLECTION
50%
OFF
ON ALL
Wlascieway scódě
nosjeding maski na twarz
EXPLORE +
VIRGINIA'S
HIKING TRAILS
TASTY
Donuts
GET SPECIAL
DISCOUNT DOUNT
40% OFF
www.earlygreatsite.com
EXO TRAVEL +
BOOKING
ONLINE
Welcome to science.
class study
with Mrs Juliana Silva
Equestrian
Showcase
showcase your home's speed, and agility in
our virtual competition for exciting prizes.
10% off on un registration
HAPPY CHINESE
NEW YEAR
20XX
May This New Year Is Filled With Happiness Prosperity
and Many Precious Moments With Your Loved One
健康美味的
自助餐
It's refreshing, your
Beauty
Enjoy
</details>

Figure 5. Qualitative results on text-to-layers. See supplementary material for individual layer visualizations.

![](images/f6965c1d6ac811f7d637f6ccb8379065e8e71e4c9a005bf2f42d565f0ad9b42f.jpg)

<details>
<summary>bar_stacked</summary>

| Category | MRT Win (%) | Draw (%) | ART Win (%) |
| :--- | :--- | :--- | :--- |
| Overall | 63.0 | 18.0 | 19.0 |
| Aesthetics | 69.0 | 7.0 | 24.0 |
| Typography | 61.0 | 17.0 | 22.0 |
| Layout | 54.3 | 17.3 | 28.4 |
</details>

Figure 6. User study comparison with previous SOTA approach on text-to-layers task. Our method significantly outperforms ART across multiple aspects.   
![](images/bca609accdebced89dc7a79b22f672110db7a4ff5d716b84a5b9e5737c8677e6.jpg)

<details>
<summary>text_image</summary>

FEUILLES DE TRAVIL
DE RÉDEACTION

COAL ES DU ESTREO
</details>

Figure 7. Qualitative results of layer overflow. Our approach supports generating overflow layers with partially visible pixels extending beyond the background region.   
![](images/cd66a90f34e652c3e64281d51fc46bcbacda9887545b2f4ec580431115d0961d.jpg)

<details>
<summary>bar_stacked</summary>

| Category | MRT Win | Tie | Qwen Win |
| :--- | :--- | :--- | :--- |
| Quality | 79.5 | 12.1 | 8.4 |
| Integrity | 68.9 | 22.5 | 8.5 |
| Granularity | 82.6 | 9.6 | 7.8 |
</details>

Figure 8. User study comparison with Qwen-Image-Layered on image-to-layers task.

layers.

Efficient Memory. Unlike Qwen-Image-Layered, which requires K× more visual tokens to extract K× different layers from a flat image, our approach is significantly more memory efficient, requiring far fewer tokens to decompose an image into many transparent layers. Fig. 19 shows latency vs. number of layers, latency vs. number of tokens, and peak memory consumption vs. number of layers. Our method achieves clear advantages in both inference speed and memory usage; for example, generating more then 20 layers with our MRT results in over 100× acceleration.

Challenges. We identify several remaining key challenges in the image-to-layer decomposition task: (i) limited generalization to photorealistic images, where models strug-

<table><tr><td rowspan="2">Layers</td><td colspan="3">PSNRmerged ↑</td><td colspan="3">SSIMmerged ↑</td></tr><tr><td>[4, 8)</td><td>[8, 16)</td><td>[16, 32)</td><td>[4, 8)</td><td>[8, 16)</td><td>[16, 32)</td></tr><tr><td>MRT (Ours)</td><td>27.3440</td><td>25.9068</td><td>25.7229</td><td>0.9034</td><td>0.8762</td><td>0.8485</td></tr><tr><td>Qwen-Image-Layered</td><td>25.8111</td><td>23.0645</td><td>22.1828</td><td>0.8706</td><td>0.8319</td><td>0.8065</td></tr></table>

Table 1. Comparison with Qwen-Image-Layered on the image-to-layers.

gle to maintain fidelity and realism on diverse real-world scenes; (ii) ambiguity in layer granularity, arising from the ill-posed nature of layer definitions and the absence of clear ground-truth separation; (iii) occluded layer completion, which remains difficult when layered occlusions involve semi-transparent or complex blending; and (iv) background inpainting, where reconstructing plausible unseen regions is challenging under severe occlusion. We visualize representative failure cases in Fig. 18. The principal causes of failures in occluded layer completion are twofold: on the one hand, the layout detector may fail to predict accurate amodal bounding regions for occluded layers; on the other hand, the image-to-layer generation model may not faithfully reconstruct complex occluded pixels due to insufficient contextual cues and data diversity. These limitations highlight avenues for future research.

# 4.3.4. Layers-to-Layers: Layered Editing

To the best of our knowledge, no prior work has studied the task of layered image editing. To establish a comparison for this task, we instantiate a baseline using GPT-Image-1, which supports multi-conditional image inputs and transparent RGBA layer outputs. We report results for our approach on two key tasks, detail how GPT-Image-1 is configured as a competitive baseline, and highlight the distinctive properties of our method.

Layer Addition. Layer Addition aims to insert new layers into an existing design conditioned on layer-wise captions. In this comparison, we simulate the user by providing two target bounding boxes on the template together with the

Theimage features a vibrant Haloween theme with dominant colors ofpurple,orange,and black. Key elements include a younggirlinwitchosumetwoglowingghostscaedpumpkinandalagespidertotaingfivemanbectse girlandpumpkinarecentralandprominent,withtheghostsandspiderprovidingbalance.Thebackgroundshowcasesa cityscape with tallbuildings,adding depth and asenseof scale.Thetexture is glossyand vector-like, with smooth gradientandaartoonshstyle.etext'HAHALWEENisd,uppercaseandenteredatthetop,sing serif font in white and orange,creating a playful and festive mood.The compositionis balanced with elements.

![](images/f31f5bbeb840e0180f77fe0e40a06d1c8969b222b8d359d8bf893315d15f91d6.jpg)

<details>
<summary>text_image</summary>

HAPPY
HALLOWEEN
</details>

![](images/cb18b69fd69a950aca1043c47a834ad7c3c38bb909fba2b5881cf3e7f9e3063c.jpg)

<details>
<summary>natural_image</summary>

Silhouette of a city skyline at sunset with illuminated buildings and a pumpkin in the foreground (no text or symbols)
</details>

![](images/55c22322dccca3298cfc4757d6bce18dbaf11a710e607be61896a5ec142b5ce9.jpg)

<details>
<summary>text_image</summary>

HAPPY
HALOWEEN
</details>

displayingthetextSLANACDADLUZinbolduppercaseseriffont.hebckgroundshowcasesaush,illated coffeeplantationwithglowingyelloworbsamongtheplantsreatingamagicalsereneatmosphere.Thecoffeecupis centrallyplaced,withsteamrising,suggestingwarmthandcomfortTelayoutisbancedwiththecupintheforeground andtheplantationextendingintothedistanceleadingtoafullmooninthesky.Thetextureisamixofvectorand

![](images/f2925a1857de1c4147ba9c3ba6d1c2085bd18d92a83d097c6dc1cd63b3b02ac4.jpg)

<details>
<summary>natural_image</summary>

Illustration of a blue coffee cup with a steaming cup, set against a glowing field at night (no text or symbols)
</details>

![](images/8bcdb8b4d0c0b82a505e0dca80d532417bc372129987e07be0b0d8324abdf9e5.jpg)

<details>
<summary>natural_image</summary>

Night scene of a flower field illuminated by glowing yellow lights, with no visible text or symbols.
</details>

![](images/9652afa6aae5748eb4d70368c37598b056f994564efcd5193dc7b221a8d81602.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing a blue teacup, a bowl, and a dark container with yellow highlights, all on transparent background (no text or symbols)
</details>

headscarfandablackof-shouldertoppairedwithawhiteskit.Thecafésetingincludesawoodentablewitha colorfultableclotandnkwitbuestraTeckgronismurredsigngeadafeelementeating avibrantyetlaedtmospereTeomposiiosanedithewomantrallitionedndheetues predominantlyphotographicwithafocusonnaturallightandsoftshadows.Themoodiscasualandstylish,with novisible

![](images/62739105192a44645a6937f46547d779990511df9d84af98dd4c495c5da54486.jpg)

![](images/c57b467a570c7978eb89a0b0f5ae1c077ba275722720242697475e05332b2702.jpg)

![](images/bfa847468895b1a5347741015276b28e6a0233c09fe2d270689bff4d4c6d502b.jpg)

![](images/32673466c46159a3cbebd0f6928056f0ccb9ed7995ca371ff7d1b8ba3ef89cf1.jpg)

![](images/bc83c9b98d976a1656a036a7b55fa16d0d56570c866967e44a5f06ab0c5a2d9e.jpg)

Theimage features adramatic andintense scene withdominant darkand fiery colors,including deepreds,blacks,and oranges.Centraltothecompositionisamuscularmanseatedonathronemadeofswordssuggestingpowerandonflct Thethrone and the man'sarmor havea metalic texture,contrasting withtheflat,cracked ground scatered with red petals.Inegoepaegasi thetitleRUNOFTHWADSHEARTpomnentlysplayedinackederystylenANAVESERie elegantscripteoealltmeseindamaticimodfespasi

![](images/ad829ddb9678190892e811ab00f582245c6af94d34743f2331fe7555b5ed352e.jpg)

![](images/6a70bf3248db7e74c5a78568e2a8b19ad3c42cde827a8f6f90782f4addf1c4af.jpg)

![](images/ac404161d1b01ae9bb1387ebe7644d8aa65311f3e2059276991557bb258bbe98.jpg)

![](images/1616d3ec1bfac747dcb10b6048a81e8c22c4e2f757e78a58920b5e6cbcc98271.jpg)

<details>
<summary>text_image</summary>

RUN OF THE
WARLORD'S
HEART

ISLANA VESPER
</details>

blueAdidasl layeredoverateniscourtbackgroudwitmultipleennisblinmotinddingasenseoactionteeisa mixofglossyandmatte,withtheshoeappearingmoretexturedanthebackgroundflatTheAdidaslogoisplacedatthe topandthetextaadeindsifisemgdealleoi includesapolaroid-styleimage oftheshoe,ading depthandaplayfulmood.Theoverallthemeis sportyand.

![](images/85feb78e5769b29656046579802817521081f3632abe88a9ef90f4788fa98bf3.jpg)

![](images/119c809bf5b081e5be718bd2fca0b9b15b9dcd1a5a8f5d19fbceea71f9db5660.jpg)

![](images/0312dac8f670fecd6f84372ee5e489e41f0e93e9832276813b72c451222fe66e.jpg)

<details>
<summary>natural_image</summary>

Adidas Barricade shoe with various green and white product photos including a tennis court, ball, and accessories (no readable text or symbols)
</details>

The image features a cat wearing a chef’s hatand apron, cooking in a kitchen with warm,earthy tones dominating the scene.ThecatisthecentralgurepromientlyosiionedinefoegondstngaredotoeTeien isdetailedwitwoodenabnetsandutensileatingacozymelyatmosphere.Thetexuresareealisticadossy withaphotographicqualitythatenhancesthelifeikeappearanceofthecatandthekitchenseting.Thelightingisoft andnatural,castinggentleshadowsandhighlightingthecalsfurandthesteamrisingfromthepan.Thecompoitionis balanedide

![](images/4ee5ee8a4b8d17a68b614ba700e604aea8e3f2950479b47fa0f8410703d39aae.jpg)

![](images/f546ca53d1461e42ae6d31d2f28a4253d23638f1fc6609b782e3d749ed17beeb.jpg)

![](images/ed10d5fb41937266a0083c4a52c0fc226b99b97d5c4ad8d9b51bb67eaef8f500.jpg)

![](images/0d56ec868e27463f2d1bcdb2bc2b36b31699371f0b1c3bd4d54e4a8d3438c398.jpg)

<details>
<summary>natural_image</summary>

Collage of kitchen ingredients including a red frying pan, food bowl, and kitchen utensils (no text or symbols visible)
</details>

Dominantcolorsinludesoftgreensandinks,withthecotage'swhitewalsprovidinganeutralbackdrop.Theottage hastwoprominentchimneysand severalwindowsadornedwithflowerboxes.Thepathleadingto thecotageis genty curved,addingasenseofepthndvitingtevieweinohescene.Thetextureissmoothandpinterlyift wgh centrallyplacedandframedbythesurroundingfoliage,creatinga harmoniousand tranquilatmosphere.Novisibletext

![](images/278c2dbf8a12bee564ff71895a553bf7ba08cde8a52c96f5fa89c973d9dc6d87.jpg)

![](images/aa26783080d97b7a304ca2cc0baefeb8766a6c4895540b49d7e06af090f752db.jpg)

![](images/ee55dd703bdb771d8a5c182493b290707cedae6aff817f201f2779039211c29f.jpg)

![](images/61f1bb0c7017c465c136fe89c8047c8e584f6a08a422400e201ed5a1e28d2a16.jpg)

![](images/3e52bca646f7d671958cd8e13e0a88f19119623bf8d8ae1443aefacbc92a937b.jpg)

Theimagefeaturesavintage-themed posterwithadominantcolorpaleteof warmbrowns,redsandgoldsaccentedbydeep blues.ThecentralfocusisthegrandarchedfacadeofDALLASUNONSTAONwithitsintricatearchitecturaletals andlargeodseriftypographyeposterdesmulpleumanfguresapproximatelyt5dressedinealy900s atirepositionedinfrontofwosteamlocomotivesreatingasenseofhistoricaldepthTebackgroundshowcasesa cloudyskyngtooalgiodimeicalitthsBand aligned centraly,creatinga balanced composition.The texture isa mix of photographic realismand vector-like..

![](images/d381a8b06145d98635f71f0bcf62211a0b4e1b2fa5320caf4b9a46ae2aa41f2f.jpg)

![](images/9b93767c91dc536de97c61baae8aea5eadf3b75a4409badc630001216b9bb8c1.jpg)

![](images/62e72fb7ea56320ef5b389ade564d5167dc9e72d4949e3b090bc8462e7ee26b6.jpg)

![](images/80aa03bce46b8f7aa739f5f198593ad1b0de02d141f701fe43ffeae443e97e19.jpg)

<details>
<summary>text_image</summary>

DALLAS
THIAX SATURS
ION
</details>

The poster featuresa playfuland vibrantdesign with dominant colors of green and blue,accented by yellow and white. It includesthreekeyimagesofapersoninaracingjacket,withthelargestimageprominentlyplacedinthecenter.The backgroundisamix ofagrassytextureandaskywithcouds,creatingawhimsicalandcheerfulmood.Thelayoutis dynamicwithoverlappingimagesandplayfuldoodleslikeheartsandstarsscatteredthroughout.Typographyisoldand sans-serifiteetiaeoaeseiet Thecompositioisbancedwitmiofpotogapcandvetorelementsncuingayelowfowrtod..

![](images/3ce3a6a7d9773ee2967888b698a512914447678530912e6d8b70e54d43d7ede4.jpg)

![](images/b56697e12653e59992c1f6b8ca061faed17230b8fd9701e0c6d26f89d7876079.jpg)

![](images/cb78b34c68d7941fa916145167e018064fbb98096fc81afea44f4edba3bcd335.jpg)

![](images/519bfced5ec3508bf7d3d7af89887f708b80a07cb4cf3ef0fdee628ceab06335.jpg)

<details>
<summary>text_image</summary>

HANNI
little bunny
</details>

hues,creatingawarmandfocusedatmosphere.Theword'EDNG'isdispayedinlarge,bold,whiteuppercaseletersat thetopitteaebeilifevawaeosinLe and'Ps,are floating aroundthelaptop,addinga dynamicand creative touch Thelaptop screen shows a software interfacegispellpaie senseofdepthandlayering.Thebottmleftcornercontainsthetext‘Stillfiguringitout,butmgetingbette..

![](images/86b3326137da12e42bf8bee735a155574ebd3298444767f5897cc983c12aca85.jpg)

![](images/df413613fb2f4e12df06c2d3de321f557196c8bd858d188ab9763e1cb6b3784c.jpg)

![](images/fafbac566cf075438fb259010216c3a65356871ca9264adaba8bc0c1e6c2f9db.jpg)

![](images/ad8d445278e1fc104bc479aab92ba27d35faad213a179a00487cf42fc176e08b.jpg)  
Figure 9. More Text-to-Layers Results.

![](images/d1e6b4507c22eee846b88a27d2884c2aff288dcfe6e5638527d3eea9c2870c4b.jpg)  
Figure 10. Image-to-Layers Results on Designs Generated with Nano-Banana-Pro (1/3): Comparison with Qwen-Image-Layered

![](images/58b87fd0bf05ac144f578cbb0af9a693a866e25ff72ca7259150ca2a6f46c239.jpg)  
Figure 11. Image-to-Layers Results on Designs Generated with Nano-Banana-Pro (2/2): Comparison with Qwen-Image-Layered

![](images/5c5edcc7bfdd89c91df3e0318eb9dafccc82b7f152d3fb975f34be47aa0d13f2.jpg)

<details>
<summary>text_image</summary>

Input Image
WALLOWS
MODEL
HAPPY
YEJI
DAY
0526
Owen-Image-Layered
(4 Layers)
WALLOWS
MODEL
I
YEJ
DAY
BLOS
SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS SOM
BLOS TOMES
DARE OR Drink
MUA 1 TANG 1
CHAI JINRO SOJU
TILL TIPM
77 Ham Right, D. L. HCMC
DOBI DOKI EVERY TUESDAY
DORI
Herbal Sachet
Dior
Dior
Herbal Sachet
Dior
Dior walking plan
小狗 Dog
遛弯计划
ナッシブシリーンク (899)
小狗 遥弯计划 小狗 遥弯计划 小狗 遥弯计划 小狗 遥弯计划 小狗 遥弯计划 小狗 遥弯计划 小狗 遥弯计划 小狗 遥弯计划 小狗 遥弯计划 小狗 遥弯计划 小狗 遥弯计划 小狗 遥弯计划 小狗 遥弯计划 小狗 遥弓形图 (899)
Ours
MODEL WALLOWS
NEW JOHN CHINA GAMES (899)
YEEJI YEEJI DAY (899)
0526 24th (899)
HAPPY
YEEJI YEEJI DAY (899)
DAY (899)
YEJI YEEJI DAY (899)
YEEJI YEEJI DAY (899)
DAY (899)
BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOS SOM BLOSSOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM SOM
</details>

Figure 12. Image-to-Layers Results on Designs Generated with Nano-Banana-Pro (3/3): Comparison with Qwen-Image-Layered

Input Image   
![](images/63bb1dd91bd1db74011a87add6340dac3e47814afce2a90b24383f5f4f7282a8.jpg)

<details>
<summary>natural_image</summary>

Illustration of a coffee cup with a glowing lotus tree in the background, no text or symbols present.
</details>

Layers   
![](images/0d867d712ef490f870a72ccf2558ebabcef9b8397427899a009bcb71965f3fdc.jpg)

<details>
<summary>natural_image</summary>

Night scene of a glowing fruit lantern in an orchard with a wooden base, no visible text or symbols
</details>

![](images/5fbd4ade84c10d059a291fd473b50acf88b501c982a30f0080522aa14a43b8c1.jpg)

<details>
<summary>natural_image</summary>

Illustration of a coffee cup, leafy greens, and a table with a coffee pan, alongside the word 'CIUDAD SALAMÍNA' (no other text or symbols)
</details>

![](images/19f5df892fab8951f30e115c9b0317ae2d97edbd9d280b25f8aff393dce90df8.jpg)

<details>
<summary>text_image</summary>

РОНЕТ
МОЛОН
РОНЕТ СОЛЕНТ
</details>

![](images/effa8cd3a1edacbeaeeeed609d88e056c249d0238481d00659443304210516ea.jpg)

<details>
<summary>text_image</summary>

ИПОЛОН СОЛЕТ
РОНЕ
</details>

![](images/8b8f37ce47ecf6d655eee5a10990fc66fac79baa3f8e5551beb3d7da7ec338b0.jpg)

<details>
<summary>text_image</summary>

Tumor
me tritura
</details>

![](images/b8447c46e1c6577a43270b7122ef4edc82b39e69c38d08c371698d6c3b37574c.jpg)

<details>
<summary>text_image</summary>

me tritura
Tu amoi
</details>

![](images/d2f96ae60b012c3dc7f4a2a6f912e31494e6cde6653514b55c3cbc479ecbc1b8.jpg)

<details>
<summary>natural_image</summary>

Illustration of women relaxing on a sandy beach with industrial factory buildings in the background (no text or symbols)
</details>

![](images/574da1f42597320ecf9f12705e0719e3d5b99844a4004a712d95b49687344275.jpg)

<details>
<summary>natural_image</summary>

Collage of fashion photos showing a coastal industrial city, a woman reading, and various clothing designs (no text or symbols)
</details>

![](images/246dd1c157090203506c592841e08059863c7effd6b10621e3ee4944bf992807.jpg)

<details>
<summary>text_image</summary>

RUIN
OF THE MARLORD'S HEART
ISLANA VESPER
</details>

![](images/ab402881492a844678af4363ff9f2858a75a409b34a5ac290441ceb013301e4d.jpg)

<details>
<summary>text_image</summary>

OF THE WARLORD'S HEART
ISLANA VESPER
CRUIN
</details>

Input Image   
![](images/94edca02c6a029668fdd1aa6435f3f08ce85a0ea5cd97510cbbfb42f31c6f073.jpg)

<details>
<summary>natural_image</summary>

Illustration of a house with a thatched roof, surrounded by blooming flowers and trees (no text or symbols)
</details>

Layers   
![](images/cc8cd6425c02926db565c517716be1b21eb32b1e3108cbea866cbdb7c55c1b45.jpg)

<details>
<summary>natural_image</summary>

Illustration of a rural house with a garden scene, framed by various architectural elements (no text or symbols)
</details>

![](images/a1c3292abde3b9d0fce95cb1d66c0edfecc6c1db3e5621211699d920e920aacf.jpg)

<details>
<summary>text_image</summary>

HOT WHEELS
8/9th
'56 FORCKUP
</details>

![](images/dda866885bfd12a05f36fa3eaf16b39cd6b791403dbe4a4ebaf5299e22f0e47e.jpg)

<details>
<summary>text_image</summary>

56 FORCKUP
† 8/10 †
† †
FOR WHEELS
</details>

![](images/58295bf50830ec02b4b2b98ed623ec1325f7e65672ca9241e2c31d8e7161ef41.jpg)

<details>
<summary>natural_image</summary>

Painting of a banquet scene with a large table, chandelier, and people in historical attire (no text or symbols visible)
</details>

![](images/733115a413172d8164056c0e32a37b2babff45ca15dfe77cc053d2a1c739df18.jpg)

<details>
<summary>natural_image</summary>

Collage of various scenes including a dining table, a man in a suit, and a wine bottle, all on transparent background (no text or symbols)
</details>

![](images/3e95972f8cc6be1229dc82faee3ffe0741fcd4144f6ef8e1c1cd3c17d5fc40b3.jpg)

<details>
<summary>natural_image</summary>

Traditional ink painting of a heron by a lakeside with mountains in the background (no text or symbols)
</details>

![](images/c052396ee2ed3a9f4bbfe7095fce778c6c8b124bcdaed4b5db9c69b89748e4d5.jpg)

<details>
<summary>natural_image</summary>

Traditional Chinese ink painting depicting mountains, a river, a tree, a heron, and water with seals (no readable text or symbols)
</details>

![](images/eeba15aae82989a0c51197bd5357424c6154951fe49c51b6d2d99df10e84a800.jpg)

<details>
<summary>natural_image</summary>

Abstract illustration of birds in blue and green tones with colorful circles (no text or symbols)
</details>

![](images/afadd16a46056f873e7b1e9872ee1d4417e5b5c1ea8985915fecb05909ae8b79.jpg)

<details>
<summary>natural_image</summary>

Colorful abstract illustration of stylized birds and fish in a grid layout (no text or symbols)
</details>

Figure 13. More Image-to-Layers Results on Designs Generated with Ideogram (1/2).

corresponding layer-wise captions. Our model predicts the requested layers in parallel while maintaining cross-layer consistency. For GPT-Image-1, we adopt an iterative generation procedure. We condition on the current composite image, draw red bounding boxes at the insertion locations, and input the corresponding layer-wise caption to GPT-Image-

1, which outputs a transparent RGBA layer. We then insert the generated layer at the specified position and iterate the process for the remaining layers. By generating multiple layers in single pass and conditioning on all layers, our method better captures inter-layer relationships and produces coherent insertions that preserve global composition and style in Fig. 22 and outperforms GPT-Image-1.

Input Image   
![](images/8dc7397b2d18631d5627191d8c50aeae7badd80b450f813346f6af645ee0ed5f.jpg)

<details>
<summary>natural_image</summary>

Cat wearing chef's hat cooking in a kitchen with cooking utensils (no text or symbols visible)
</details>

Layers   
![](images/053d7690d0c48a0be65356116de094e3fd56a6e73e8dea89b17b5f14d6431b0a.jpg)

<details>
<summary>natural_image</summary>

Interior scene of a kitchen with a cat holding a spoon, surrounded by food items including a dog, utensils, and cooking utensils (no visible text or symbols)
</details>

Input Image   
![](images/20ef845424651884ba9b3869fa03cbf798568ad5423ae9bfb91bfd75b7aacabc.jpg)

<details>
<summary>natural_image</summary>

Illustration of a person standing on a rooftop overlooking a city skyline with a glowing 'MAXL' sign (no text or symbols on the figure or background)
</details>

Layers   
![](images/a6fb14ddbaa15a33ea875febc405918c67719c84357b7c76d6f2319a1a6ede93.jpg)

<details>
<summary>natural_image</summary>

Composite image showing a city night scene with a glowing light, a pink tent, and abstract shapes including a figure in motion (no text or symbols)
</details>

![](images/b7745922d824dd74ae62d33cab80eb28114bc8e4af871a19890d5070f44e7b83.jpg)

<details>
<summary>natural_image</summary>

Futuristic illustration of a smiling woman wearing a coffee cup with a landscape background (no text or symbols)
</details>

![](images/7a210088b275f1b5f0eeb9166ac051474a4ef98a2de0d35c775ad821284a8436.jpg)

<details>
<summary>natural_image</summary>

Collage of colorful artistic images including a dragon, moon, face, cup, and landscape (no text or symbols)
</details>

![](images/01f3bf42edfe1f1216180d3acc39c607ac6606226be6c725003f294eba1d7c61.jpg)

<details>
<summary>text_image</summary>

KINGDOM
OF ASH AND
LIGHT.
OWEBNOYSIS OF ORUT LAGIFF
</details>

![](images/50aa664be15312ca729a60f41da67913b26cf61ca88e02b5ff6c091dca1b66ff.jpg)

<details>
<summary>text_image</summary>

KINGDOM
LIGHT.
AND
OF
ASH
OWEBNOYS OF ORU LAGBE
</details>

![](images/df081d941331510f39273903c287e4f420b358b4d0e65e93a5d15b8bbd494362.jpg)

<details>
<summary>natural_image</summary>

Woman sitting at an outdoor café table with a blue headscarf, holding a drink (no visible text or signage)
</details>

![](images/2c25a92143cdc8d42e62dfea4769200796995b51778819f031ebf9513d55ef40.jpg)

<details>
<summary>natural_image</summary>

Collage of outdoor street scenes including a woman in a blue headscarf, a chair with a straw, and a drink in a glass (no visible text or symbols)
</details>

![](images/cd4d684afed4186a1c47b06dd4b4d10da2e181b38ec918603b31c83b95241e27.jpg)

<details>
<summary>natural_image</summary>

Surreal illustration of a glowing orb with a planet inside, surrounded by abstract patterns and glowing elements (no text or symbols)
</details>

![](images/4926a333d67cc22814c5218eec6e594939468df3af66a5d85dcb8b14a7be2dba.jpg)

<details>
<summary>natural_image</summary>

Abstract digital artwork featuring a glowing purple starry background, a human figure with glowing eyes, and colorful abstract patterns on transparent backgrounds (no text or symbols)
</details>

![](images/47bb569fc99d27a262b081e851fcd7b78c0b05e7963c167639b89d26fb48f85a.jpg)

<details>
<summary>text_image</summary>

THE DALLAS
UNION STATION
THE EARLY TIME
THIAX SATTUIRS
</details>

![](images/13d5f147d182ef1bb4ac885598f78e08eb3e49bbaf25b6fab72b2755bb91eb4b.jpg)

<details>
<summary>text_image</summary>

DALLAS
UNION STATION
THE NEW SATELLINES
</details>

![](images/01785cfa9ed135a3efb08e7c98e659a0d2d61b5a3a634debd7456267861b418c.jpg)

<details>
<summary>text_image</summary>

EL PORTAL
11.11 SE ABRE.
11.11
RED LINE
CHILE
</details>

![](images/9e0d8d00d79f0606ab0e8f077bcc18b05248dd8201acee7d5398361e73463985.jpg)

<details>
<summary>text_image</summary>

11.11 EL H'SE ABRE
EL PORTAL
RED LINE
</details>

![](images/650b0e50d72abf649dfb36899403775ac9c77a87a435b5fdf3ff03474923cd13.jpg)

<details>
<summary>natural_image</summary>

Festive birthday cake setup with 13th grade decoration, surrounded by candles and cupcakes (no visible text or symbols)
</details>

![](images/45b48485258a8dce185213e0ee1d5c4c926687e16152d084b42a6b8773f3d281.jpg)

<details>
<summary>natural_image</summary>

Collage of vintage cake and table decorations including candles, pastries, and decorative items (no text or symbols)
</details>

![](images/4228c261de8aa8f3d55927d669c56d13eb8dfba69e054d694b7fd96f427a0b79.jpg)

<details>
<summary>natural_image</summary>

Group of people dining outdoors with chicken and food items, no visible text or symbols
</details>

![](images/3d85822ca588322fbe7243f6a80dccc02e45d0e256e5d786bab827d8a90bc1dc.jpg)

<details>
<summary>natural_image</summary>

Collage of food and kitchen scenes: outdoor dining scene with chicken, table with tableware, and interior kitchen setup (no visible text or symbols)
</details>

Figure 14. More Image-to-Layers Results on Designs Generated with Ideogram (2/2).

Layer Restylization. For restylizing target layers, the user provides assets to be placed on the canvas; we restylize these assets into layers that harmonize with the overall composition. For GPT-Image-1, we provide multi-image inputs:

the merged image of existing layers annotated with a red bounding box to indicate the insertion location, together with the user-specified asset. After predicting one layer, we insert it at the specified position and iterate for the remaining targets. Our method harmonizes all selected layers in a single pass, whereas GPT-Image-1 requires layer-by-layer

![](images/ae9292abc743bc90dbbbb3f0196d7f1b47dcaef676a414da59f4d920733ab325.jpg)  
Figure 15. More Image-to-Layers Results on Designs Generated with Nano-Banana-Pro (1/2).

![](images/79f816226af03a67570ca0cf9859e217ba163ded9fc07ef076272f758e19ae22.jpg)  
Figure 16. More Image-to-Layers Results on Designs Generated with Nano-Banana-Pro (2/2).

![](images/3e07ee68fa8e475821850a9a06f5531503f38bde7bce02a064cff69ace09da46.jpg)  
Figure 17. More Image-to-Layers Results on Qwen-Image-Layered test set.

generation, which increases latency and may propagate inconsistencies across multiple edits. Fig. 22 shows that our

edits better preserve geometry while adapting appearance to the target style.

![](images/91c775de0a1268aef77dee7e6b64ff818c2401e072c5907f7cdaae6efb700915.jpg)

<details>
<summary>text_image</summary>

DARE • Drink
MUA TANG TI
CRAI SHENG 2023
TILL LIPM
FENTY x PUMA XXX
PUMA
XIII Torneio
SUECA
6/12/2025
</details>

Figure 18. Illustrating the Challenges of the Image-to-Layers. We show some representative failure cases when handling occluded layer completion. We find that our model fails to generate the occluded parts due to the regional crop design when the bounding boxes are tightly fit around only the visible pixels. We suspect another key reason is that these test cases differ from our training data distribution, and we leave this challenge to future work.

![](images/c81665dc3b0d358554a016dcb06493db2947570405874417605ad9baa210a4a1.jpg)

<details>
<summary>line</summary>

| # of Layers | MRT (B200) | MRT (H200) | Qwen-Image-Layered (B200) |
| ----------- | ---------- | ---------- | -------------------------- |
| 7.5         | ~0         | ~0         | 108.5×                     |
| 10.0        | ~0         | ~0         | ~150                       |
| 12.5        | ~0         | ~0         | ~250                       |
| 15.0        | ~0         | ~0         | ~400                       |
| 17.5        | ~0         | ~0         | ~500                       |
| 20.0        | ~0         | ~0         | ~550                       |
</details>

[a) Latency Comparison

![](images/f79276d26a7874345a008e39e169a0d2b2eb31f1e6431a70e1fdefb0805be9b1.jpg)

<details>
<summary>scatter</summary>

| # of Tokens (×10³) | Latency (s) - B200 | Latency (s) - H200 |
| ------------------ | ------------------ | ------------------ |
| 9.5                | 3.5                | 6.5                |
| 10.0               | 4.0                | 7.5                |
| 10.5               | 4.5                | 8.5                |
| 11.0               | 5.0                | 9.5                |
| 11.5               | 5.5                | 10.5               |
| 12.0               | 6.0                | 11.5               |
| 12.5               | 6.5                | 12.5               |
| 13.0               | 7.0                | 13.5               |
| 13.5               | 7.5                | 14.5               |
| 14.0               | 8.0                | 15.5               |
| 14.5               | 8.5                | 16.5               |
| 15.0               | 9.0                | 17.5               |
| 15.5               | 9.5                | 18.5               |
| 16.0               | 10.0               | 19.5               |
</details>

[b] MRT Tokens vs. Iference Time

![](images/9c5841aa527e407d66389d2363b709c3556cca9fbcd3e9228519196bd806b9ac.jpg)

<details>
<summary>bar_stacked</summary>

| # of Layers | MRT (H200 I B200) (GB) | Owen-Image-Layered (B200) (GB) | Static Memory (GB) |
|---|---|---|---|
| 4-8 | 56 | 78 | 54 |
| 8-12 | 56 | 89 | 53 |
| 12-16 | 56 | 100 | 53 |
| 16-20 | 56 | 115 | 53 |
* indicates statistical significance. The stacked bars represent the sum of memory consumption for each layer. Error bars are present on the top of the bars.
</details>

[c] Peak Memory Consumption

Figure 19. Inference efficiency comparison between MRT and Qwen-Image-Layered. (a) Latency scaling with number of layers. MRT maintains near-constant latency (∼5s) while Qwen-Image-Layered scales linearly, resulting in up to 108.5× speedup at ∼20 layers. (b) MRT inference time vs. token count on H200 and B200 GPUs, demonstrating linear scaling behavior. (c) Peak GPU memory consumption across varying layer configurations. The shaded region indicates the baseline memory allocated to model weights. MRT reduces memory consumption by $1 0 . 5 \times  2 3 . 6 \times$ , with efficiency gains scaling proportionally with layer numbers. All reported results are conducted over 100 samples on single GPU with identical layer numbers.   
![](images/26207f7ab8b70069ef9056863ed5a7fd0c1e7cdc4a21cdad7b79f0a3e7c32ced.jpg)

<details>
<summary>text_image</summary>

Input
Ours
Lovart
RoboNeo
LayerD
Qwen-Image-Layered
EASTER SALE
50% OFF
SPECIAL OFFER
SHOP NOW
EASTER SALE
50% OFF
EASTER SALE
50% OFF
EASTER SALE
50% OFF
EASTER SALE
50% OFF
EASTER SALE
50% OFF
EASTER SALE
50% OFF
EASTER SALE
50% OFF
EASTER SALE
50% OFF
EASTER SALE
50% OFF
EASTER SALE
50% OFF
EASTER SALE
50% OFF
EASTER SALE
50% OFF
</details>

Figure 20. Image-to-layers comparison. Each panel’s top-left shows the composed image with decomposed layers. Our method outperforms all baselines. Lovart shows poor decomposition quality, RoboNeo exhibits artifacts, LayerD and Qwen-Image-Layered produce overly grouped layers. Top-left: composed image with layers. (Best viewed zoomed in)

# 4.4. Ablation Study and Analaysis

Larger models and dataset improve quality. To demonstrate the importance of model and dataset scaling, we train text-to-layers models using FLUX.1 [dev] (13B) and Qwen-Image (20B) on the same 0.5M-sample dataset. Model scaling alone reduces FID from 17.79 to 16.15. Subsequently scaling the dataset to 10M samples further reduces FID to 15.63 under a limited training budget, with additional gains expected from extended training. These results confirm that both model capacity and dataset scale are essential for highquality generation.

![](images/e3cf54c1f38b69333420d27b68811d1fd3758a1900e8cdd54a8a53ede255fd4e.jpg)

<details>
<summary>bar_stacked</summary>

| Category | MRT Win (%) | Draw (%) | LayerD Win (%) |
| :--- | :--- | :--- | :--- |
| Quality | 60.0 | 37.0 | 5.0 |
| Integrity | 81.0 | 19.0 | 5.0 |
| Granularity | 95.0 | 5.0 | 5.0 |
| Quality | 52.0 | 41.0 | 7.0 |
| Integrity | 41.0 | 57.0 | 7.0 |
| Granularity | 89.0 | 7.0 | 7.0 |
| Quality | 82.5 | 10.0 | 7.5 |
| Integrity | 81.2 | 15.0 | 12.5 |
| Granularity | 75.0 | 12.5 | 12.5 |
</details>

Figure 21. Comparison with SOTA and commercial systems on image-to-layers. We conduct a blind user study where participants select the better result from paired samples. Blind user study shows our method significantly outperforms LayerD and commercial systems (Lovart, Robo-Neo). Participants evaluate the results from three aspects including (i) Quality: semantic correctness and transparency, (ii) Integrity: faithful reconstruction of the input, and (iii) Granularity: appropriate decomposition level—avoiding overly grouped layers. Our approach demonstrates significant advantages across all evaluation dimensions according to user study. 

<table><tr><td>Training Data</td><td> $FID_{merged} \downarrow$ </td><td> $PSNR_{merged} \uparrow$ </td><td> $SSIM_{merged} \uparrow$ </td></tr><tr><td>w/o overflow data</td><td>15.68</td><td>21.81</td><td>0.8543</td></tr><tr><td>w/ overflow data</td><td>16.15</td><td>22.75</td><td>0.8711</td></tr></table>

Table 2. Overflow support on text-to-layers (T2L). 

<table><tr><td>method</td><td>task mix ratio</td><td> $FID_{merged} \downarrow$ </td><td> $PSNR_{merged} \uparrow$ </td><td> $SSIM_{merged} \uparrow$ </td></tr><tr><td>T2L</td><td>100% / 0% / 0%</td><td>16.15</td><td>22.75</td><td>0.8711</td></tr><tr><td>T2L+I2L</td><td>80% / 20% / 0%</td><td>15.68</td><td>23.06</td><td>0.8924</td></tr><tr><td>T2L+I2L+L2L</td><td>70% / 15% / 15%</td><td>17.06</td><td>21.97</td><td>0.8606</td></tr></table>

Table 3. Multiple task training. T2L: text-to-layers. I2L: image-tolayers. L2L: layers-to-layers. 

<table><tr><td>method</td><td> $PSNR_{merged} \uparrow$ </td><td> $SSIM_{merged} \uparrow$ </td><td> $PSNR_{layer} \uparrow$ </td><td> $SSIM_{layer} \uparrow$ </td></tr><tr><td>w/o text condition</td><td>21.27</td><td>0.8697</td><td>26.03</td><td>0.9794</td></tr><tr><td>w/ text condition</td><td>21.65</td><td>0.8805</td><td>27.24</td><td>0.9846</td></tr></table>

Table 4. Text condition on image-to-layers (I2L) task. 

<table><tr><td>method</td><td> $PSNR_{merged} \uparrow$ </td><td> $SSIM_{merged} \uparrow$ </td><td> $PSNR_{layer} \uparrow$ </td><td> $SSIM_{layer} \uparrow$ </td></tr><tr><td>w/o merge aug.</td><td>21.65</td><td>0.8805</td><td>27.24</td><td>0.9846</td></tr><tr><td>w/ merge aug.</td><td>21.97</td><td>0.8864</td><td>26.96</td><td>0.9840</td></tr></table>

Table 5. Layer grouping augmentation on image-to-layers (I2L) task. 

<table><tr><td>method</td><td>Denoise steps</td><td> $FID_{merged} \downarrow$ </td></tr><tr><td>Baseline</td><td>50</td><td>16.02</td></tr><tr><td>+ DMD2 Distillation</td><td>16</td><td>16.21</td></tr><tr><td>+ DMD2 Distillation</td><td>8</td><td>18.58</td></tr></table>

Table 6. Multi-layer generator distillation.

Overflow support w/o performance loss. Table 2 evaluates the impact of overflow-aware generation. Over 60% of designs contain overflow layers while previous works all truncate these elements, severely limiting editability and reusability. Training with overflow data enables complete layer generation with minimal performance cost: our model achieves comparable FID, PSNR, and SSIM scores while uniquely preserving overflow elements.

Multi-task training and performance trade-offs Table 3 shows unified multi-task training with random task sampling. Our framework integrates all three tasks without multi-stage fine-tuning while maintaining comparable performance across configurations, demonstrating minimal degradation from unification. We observe that introducing the layers-to-layers task slightly reduces overall performance, which we attribute to layer-to-layer dataset quality issues—a direction we leave for future work.

Textual conditioning is not essential for image-to-layers. An important question is whether global captions are necessary for image-to-layers decomposition. Table 4 ablates caption conditioning and shows modest but consistent improvements across metrics. This reveals a noteworthy finding: while textual guidance aids boundary disambiguation and provides semantic cues for complex overlapping compositions, it is not essential for our framework.

Layer grouping augmentation improves robustness. Table 5 validates layer grouping augmentation. Since our framework requires layout inputs, a distribution gap exists between precise training layout annotations and noisy test-time layouts from users or detectors. We address this by randomly merging layers during training to increase layout diversity. This strategy yields consistent improvements even on DESIGN-MULTI-LAYER-BENCH with highquality layout annotations, with larger gains expected under noisy layout conditions.

Distilled multi-layer generator brings significant acceleration. By incorporating DMD2 distillation [60, 61], we accelerate our multi-layer generation from 50 to 8 denoising steps, achieving a 6× speedup with minimal performance degradation. FID scores remain comparable in Table 6 and visual quality is largely preserved in Fig. 23, demonstrating the effectiveness of distillation for few-step generation in multi-layer image diffusion models.

Additional ablations. We provide additional ablation studies on caption length, multilingual design generation, and fine-tuning with PrismLayers data in the supplementary material.

# 5. Conclusion

In this paper, we have presented the first systematic study examining the performance frontier of multi-layer transparent image generation at scale. We introduced the Masked Region Transformer, a large-scale diffusion framework that unifies text-to-layers, image-to-layers, and layers-tolayers generation within a shared masked region paradigm. Trained on over 10M multilingual design samples, our 20Bparameter model incorporates key technical innovations: an overflow-aware canvas layer for complete boundary handling, and distribution matching distillation for real-time generation. Together, these contributions enable efficient synthesis of high-fidelity, semi-transparent, fully editable visual layers.

![](images/ee1e5c64c94692427f1681cb719c8b88a6380c35f8fef7ba98a6bf902e67da18.jpg)

<details>
<summary>text_image</summary>

Layer Addition
Existed Layer
SMART TELEVISION
SMART TELEVISION
20% OFF
DISCO-NIGHT
PARTY DANCE
DISCO-NIGHT
Ours
Layer Caption
<layer> Horizontal oval gradient banner (light left → dark right), rounded edges, no text. <layer>
<layer> Bold black sans-serif "20% OFF"—"20%" larger; "OFF" smaller but bold. </layer>
SMART TELEVISION
20% OFF
20% OFF
<layer> Bold dark-blue sans-serif "DicoNight"; D/Nt larger; modern, energetic. </layer>
<layer> Bold black sans-serif "20% OFF"—"20%" larger; "OFF" smaller but bold. </layer>
DISCO-NIGHT
PARTY DANCE
DISCO-NIGHT
Ours
Layer Restylization
Existed Layer
SUMMER FASHION
SUMMER FASHION
JOHN FRUTANG
ADEN
Ours
User Assets
SUMMER FASHION
JOHN FRUTANG
ADEN
ZEAL
ZEAL
Ours
Image-1
Image-1
</details>

Figure 22. Qualitative comparison on layers-to-layers. Layer addition (first two rows) and layer restylization (last two rows). For layer addition, our approch also better follow the layer-wise instructions than GPT-Image-1. For layer resylization, our method also outperforms GPT-Image-1 in terms of layer coherence and style consistency. The layers-to-layers task enables flexible user interaction with the generative model through iterative layer-wise editing.

![](images/a7efa34ad24b0b6ca7e0eda1e73db6bbb126580b75f3832068616f723d2f60f6.jpg)

<details>
<summary>text_image</summary>

baseline 50-NFE
CATTLE MARKET
KEEP YOUR BODY HEALTHY AND GROWTH
distilled 16-NFE
CATTLE MARKET
KEEP YOUR BODY HEALTHY AND GROWTH
distilled 8-NFE
CATTLE MARKET
KEEP YOUR BODY HEALTHY AND GROWTH
PIZZERIA NAME
MAYBE PIZZA?
HAPPY HOURS
3PM-5PM.
PIZZERIA NAME
MAYBE PIZZA?
HAPPY HOURS
3PM-5PM.
PIZZERIA NAME
MAYBE PIZZA?
HAPPY HOURS
3PM-5PM.
</details>

Figure 23. Comparison between baseline and few-step distilled model.

![](images/b00f910b617aa68ca4b39a527de97911a9ac4230793b4c276200f7f3728dcf5e.jpg)

<details>
<summary>natural_image</summary>

Six-panel image grid showing outdoor equipment including headsets, tools, and a device on grass (no text or symbols)
</details>

Figure 24. Qualitative results of image-to-layers on out-of-domain natural images. Despite only trained on poster-style design datasets, our model generalizes to natural scenes.

# References

[1] Omer Bar-Tal, Lior Yariv, Yaron Lipman, and Tali Dekel. MultiDiffusion: Fusing diffusion paths for controlled image generation. In ICML, 2023. 2   
[2] Cameron Braunstein, Hevra Petekkaya, Jan Eric Lenssen, Mariya Toneva, and Eddy Ilg. Slayr: Scene layout generation with rectified flow. arXiv preprint arXiv:2412.05003, 2024. 2   
[3] Shang Chai, Liansheng Zhuang, and Fengying Yan. LayoutDM: Transformer-based diffusion model for layout generation. In CVPR, 2023.   
[4] Jian Chen, Ruiyi Zhang, Yufan Zhou, Jennifer Healey, Jiuxiang Gu, Zhiqiang Xu, and Changyou Chen. TextLap: Customizing language models for text-to-layout planning. In EMNLP Findings, 2024. 2   
[5] Junwen Chen, Heyang Jiang, Yanbin Wang, Keming Wu, Ji Li, Chao Zhang, Keiji Yanai, Dong Chen, and Yuhui Yuan. Prismlayers: Open data for high-quality multilayer transparent image generative models. arXiv preprint arXiv:2505.22523, 2025. 2, 3, 5   
[6] Chin-Yi Cheng, Forrest Huang, Gang Li, and Yang Li. Play: Parametrically conditioned layout generation using latent diffusion. In ICML, 2023. 2   
[7] Yutao Cheng, Zhao Zhang, Maoke Yang, Hui Nie, Chunyuan Li, Xinglong Wu, and Jie Shao. Graphic design with large multimodal model. arXiv:2404.14368, 2024. 2   
[8] Zhuobai Dong, Rui Zhao, Songjie Wu, Junchao Yi, Linjie Li, Zhengyuan Yang, Lijuan Wang, and Alex Jinpeng Wang. Glance: Accelerating diffusion models with 1 sample, 2025. 5

[9] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam Levi, Dominik ¨ Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024. 2   
[10] Weixi Feng, Wanrong Zhu, Tsu-jui Fu, Varun Jampani, Arjun Akula, Xuehai He, Sugato Basu, Xin Eric Wang, and William Yang Wang. LayoutGPT: Compositional visual planning and generation with large language models. In NeurIPS, 2024. 2   
[11] Alessandro Fontanella, Petru-Daniel Tudosiu, Yongxin Yang, Shifeng Zhang, and Sarah Parisot. Generating compositional scenes via text-to-image rgba instance generation. arXiv preprint arXiv:2411.10913, 2024. 2   
[12] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. arXiv preprint arXiv:2410.12557, 2024. 2   
[13] Yu Gao, Lixue Gong, Qiushan Guo, Xiaoxia Hou, Zhichao Lai, Fanshi Li, Liang Li, Xiaochen Lian, Chao Liao, Liyang Liu, et al. Seedream 3.0 technical report. arXiv preprint arXiv:2504.11346, 2025. 2   
[14] Lixue Gong, Xiaoxia Hou, Fanshi Li, Liang Li, Xiaochen Lian, Fei Liu, Liyang Liu, Wei Liu, Wei Lu, Yichun Shi, et al. Seedream 2.0: A native chinese-english bilingual image generation foundation model. arXiv preprint arXiv:2503.07703, 2025. 2   
[15] Julian Jorge Andrade Guerreiro, Naoto Inoue, Kento Masui, Mayu Otani, and Hideki Nakayama. LayoutFlow: Flow matching for layout generation. In ECCV, 2024. 2   
[16] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022. 5   
[17] Runhui Huang, Kaixin Cai, Jianhua Han, Xiaodan Liang, Renjing Pei, Guansong Lu, Songcen Xu, Wei Zhang, and Hang Xu. LayerDiff: Exploring text-guided multi-layered composable image synthesis via layer-collaborative diffusion model. In ECCV, 2024. 2   
[18] Mude Hui, Zhizheng Zhang, Xiaoyi Zhang, Wenxuan Xie, Yuwang Wang, and Yan Lu. Unifying layout generation with a decoupled diffusion model. In CVPR, 2023. 2   
[19] Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and Kota Yamaguchi. LayoutDM: Discrete diffusion model for controllable layout generation. In CVPR, 2023.   
[20] Naoto Inoue, Kotaro Kikuchi, Edgar Simo-Serra, Mayu Otani, and Kota Yamaguchi. Towards flexible multi-modal document models. In CVPR, 2023. 2   
[21] Naoto Inoue, Kento Masui, Wataru Shimoda, and Kota Yamaguchi. OpenCOLE: Towards reproducible automatic graphic design generation. In CVPR Workshops, 2024. 2   
[22] Peidong Jia, Chenxuan Li, Zeyu Liu, Yichao Shen, Xingru Chen, Yuhui Yuan, Yinglin Zheng, Dong Chen, Ji Li, Xiaodong Xie, et al. COLE: A hierarchical generation framework for graphic design. arXiv preprint arXiv:2311.16974, 2023. 2

[23] Zhaoyun Jiang, Shizhao Sun, Jihua Zhu, Jian-Guang Lou, and Dongmei Zhang. Coarse-to-fine generative modeling for graphic layouts. In AAAI, 2022. 2   
[24] Zhaoyun Jiang, Jiaqi Guo, Shizhao Sun, Huayu Deng, Zhongkai Wu, Vuksan Mijovic, Zijiang James Yang, Jian-Guang Lou, and Dongmei Zhang. LayoutFormer++: Conditional graphic layout generation via constraint serialization and decoding space restriction. In CVPR, 2023.   
[25] Kotaro Kikuchi, Naoto Inoue, Mayu Otani, Edgar Simo-Serra, and Kota Yamaguchi. Multimodal markup document models for graphic design completion. arXiv:2409.19051, 2024. 2   
[26] Yunji Kim, Jiyoung Lee, Jin-Hwa Kim, Jung-Woo Ha, and Jun-Yan Zhu. Dense text-to-image generation with attention modulation. In ICCV, 2023. 2   
[27] Xiang Kong, Lu Jiang, Huiwen Chang, Han Zhang, Yuan Hao, Haifeng Gong, and Irfan Essa. BLT: Bidirectional layout transformer for controllable layout generation. In ECCV, 2022. 2   
[28] Pengzhi Li, Qinxuan Huang, Yikang Ding, and Zhiheng Li. Layerdiffusion: Layered controlled image editing with diffusion models. In SIGGRAPH Asia 2023 Technical Communications, pages 1–4, 2023. 2   
[29] Yuheng Li, Haotian Liu, Qingyang Wu, Fangzhou Mu, Jianwei Yang, Jianfeng Gao, Chunyuan Li, and Yong Jae Lee. GLIGEN: Open-set grounded text-to-image generation. In CVPR, 2023. 2   
[30] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022. 2   
[31] Bingchen Liu, Ehsan Akhgari, Alexander Visheratin, Aleks Kamko, Linmiao Xu, Shivam Shrirao, Chase Lambert, Joao Souza, Suhail Doshi, and Daiqing Li. Playground v3: Improving text-to-image alignment with deep-fusion large language models. arXiv preprint arXiv:2409.10695, 2024. 2   
[32] Zeyu Liu, Weicong Liang, Zhanhao Liang, Chong Luo, Ji Li, Gao Huang, and Yuhui Yuan. Glyph-byt5: A customized text encoder for accurate visual text rendering. In European Conference on Computer Vision, pages 361–377. Springer, 2024. 2   
[33] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024. 2   
[34] Weijian Luo, Zemin Huang, Zhengyang Geng, J Zico Kolter, and Guo-jun Qi. One-step diffusion distillation through score implicit matching. Advances in Neural Information Processing Systems, 37:115377–115408, 2024. 2   
[35] Yihong Luo, Tianyang Hu, Jiacheng Sun, Yujun Cai, and Jing Tang. Learning few-step diffusion models by trajectory distribution matching, 2025. 5   
[36] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. Sit: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024. 2   
[37] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF inter-

national conference on computer vision, pages 4195–4205, 2023. 2   
[38] Yifan Pu, Yiming Zhao, Zhicong Tang, Ruihong Yin, Haoxing Ye, Yuhui Yuan, Dong Chen, Jianmin Bao, Sirui Zhang, Yanbin Wang, et al. Art: Anonymous region transformer for variable multi-layer transparent image generation. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 7952–7962, 2025. 2, 3, 5, 6   
[39] Qwen. Qwen-Image-Layered. https : / / github . com/QwenLM/Qwen-Image-Layered/tree/main/ assets/test\_images, 2025. 6   
[40] Vishnu Sarukkai, Linden Li, Arden Ma, Christopher Re, and´ Kayvon Fatahalian. Collage diffusion. In WACV, 2024. 2   
[41] Axel Sauer, Dominik Lorenz, Andreas Blattmann, and Robin Rombach. Adversarial diffusion distillation. In European Conference on Computer Vision, pages 87–103. Springer, 2024. 2   
[42] Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022. 2   
[43] Team Seedream, Yunpeng Chen, Yu Gao, Lixue Gong, Meng Guo, Qiushan Guo, Zhiyao Guo, Xiaoxia Hou, Weilin Huang, Yixuan Huang, et al. Seedream 4.0: Toward nextgeneration multimodal image generation. arXiv preprint arXiv:2509.20427, 2025. 2   
[44] Mohammad Amin Shabani, Zhaowen Wang, Difan Liu, Nanxuan Zhao, Jimei Yang, and Yasutaka Furukawa. Visual Layout Composer: Image-vector dual diffusion model for design layout generation. In CVPR, 2024. 2   
[45] Tomoyuki Suzuki, Kang-Jun Liu, Naoto Inoue, and Kota Yamaguchi. Layerd: Decomposing raster graphic designs into layers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 17783–17792, 2025. 2, 6   
[46] Zecheng Tang, Chenfei Wu, Juntao Li, and Nan Duan. LayoutNUWA: Revealing the hidden layout expertise of large language models. In ICLR, 2023. 2   
[47] Omost Team. Omost github page, 2024. 2   
[48] Petru-Daniel Tudosiu, Yongxin Yang, Shifeng Zhang, Fei Chen, Steven McDonagh, Gerasimos Lampouras, Ignacio Iacobacci, and Sarah Parisot. Mulan: A multi layer annotated dataset for controllable text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22413–22422, 2024. 2   
[49] VistaCreate Team. Vistacreate (formerly crello) graphic design platform. https://create.vista.com/, 2025. Accessed: 2025-11-09. 6   
[50] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025. 3   
[51] Xudong Wang, Trevor Darrell, Sai Saketh Rambhatla, Rohit Girdhar, and Ishan Misra. InstanceDiffusion: Instance-level control for image generation. In CVPR, 2024. 2

[52] X. Wang, Siming Fu, Qihan Huang, Wanggui He, and Hao Jiang. MS-Diffusion: Multi-subject zero-shot image personalization with layout guidance. arXiv:2406.07209, 2024. 2   
[53] Yilin Wang, Zeyuan Chen, Liangjun Zhong, Zheng Ding, Zhizhou Sha, and Zhuowen Tu. Dolfin: Diffusion layout transformers without autoencoder. In ECCV, 2024. 2   
[54] Haohan Weng, Danqing Huang, Yu Qiao, Zheng Hu, Chin-Yew Lin, Tong Zhang, and CL Chen. Desigen: A pipeline for controllable design template generation. In CVPR, 2024. 2   
[55] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025. 2, 3   
[56] Kota Yamaguchi. Canvasvae: Learning to generate vector graphic documents. arXiv preprint arXiv:2108.01249, 2021. 2   
[57] Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal LLMs. In ICML, 2024. 2   
[58] Tao Yang, Yingmin Luo, Zhongang Qi, Yang Wu, Ying Shan, and Chang Wen Chen. PosterLLaVa: Constructing a unified multi-modal layout generator with LLM. arXiv:2406.02884, 2024. 2   
[59] Shengming Yin, Zekai Zhang, Zecheng Tang, Kaiyuan Gao, Xiao Xu, Kun Yan, Jiahao Li, Yilei Chen, Yuxiang Chen, Heung-Yeung Shum, Lionel M. Ni, Jingren Zhou, Junyang Lin, and Chenfei Wu. Qwen-image-layered: Towards inherent editability via layer decomposition. 2025. 2, 6   
[60] Tianwei Yin, Michael Gharbi, Taesung Park, Richard Zhang, ¨ Eli Shechtman, Fredo Durand, and Bill Freeman. Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems, 37:47455–47487, 2024. 2, 5, 18   
[61] Tianwei Yin, Michael Gharbi, Richard Zhang, Eli Shecht- ¨ man, Fredo Durand, William T Freeman, and Taesung Park. One-step diffusion with distribution matching distillation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6613–6623, 2024. 2, 5, 18   
[62] Lvmin Zhang and Maneesh Agrawala. Transparent image layer diffusion using latent transparency. ACM Transactions on Graphics, 43(4):1–15, 2024. 2   
[63] Xinyang Zhang, Wentian Zhao, Xin Lu, and Jeff Chien. Text2layer: Layered image generation using latent diffusion model. arXiv preprint arXiv:2307.09781, 2023. 2   
[64] Xinchen Zhang, Ling Yang, Guohao Li, Yaqi Cai, Jiake Xie, Yong Tang, Yujiu Yang, Mengdi Wang, and Bin Cui. IterComp: Iterative composition-aware feedback learning from model gallery for text-to-image generation. arXiv:2410.07171, 2024. 2   
[65] Kaiwen Zheng, Yuji Wang, Qianli Ma, Huayu Chen, Jintao Zhang, Yogesh Balaji, Jianfei Chen, Ming-Yu Liu, Jun Zhu, and Qinsheng Zhang. Large scale diffusion distillation via score-regularized continuous-time consistency. arXiv preprint arXiv:2510.08431, 2025. 2

[66] Zhenyu Zhou, Defang Chen, Can Wang, Chun Chen, and Siwei Lyu. Simple and fast distillation of diffusion models. Advances in Neural Information Processing Systems, 37:40831–40860, 2024.   
[67] Yuanzhi Zhu, Xi Wang, Stephane Lathuili ´ ere, and Vicky \` Kalogeiton. Di [m] o: Distilling masked diffusion models into one-step generator. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18606– 18618, 2025. 2

# MRT: Masked Region Transformer for Layered Image Generation and Editing at Scale

Supplementary Material

<table><tr><td rowspan="2">Training Caption Length</td><td colspan="2">FIDmerged ↓</td></tr><tr><td>Short Cap.</td><td>Long Cap.</td></tr><tr><td>Short Cap.</td><td>17.64</td><td>18.56</td></tr><tr><td>Long Cap.</td><td>17.95</td><td>16.15</td></tr><tr><td>Mixed (50% short + 50% long)</td><td>16.13</td><td>15.93</td></tr></table>

Table 1. Effect of caption length during training. We train models with short captions, long captions, or a mixture of both, and evaluate FID on VC5K test set using short and long captions respectively.

<table><tr><td>method</td><td>Denoise steps</td><td>Latency(s)</td><td>Speed up</td><td> $FID_{merged} \downarrow$ </td></tr><tr><td>Baseline</td><td>50</td><td>14.4</td><td>-</td><td>16.02</td></tr><tr><td>+ Distill</td><td>16</td><td>4.5</td><td>3.2x</td><td>16.21</td></tr><tr><td>+ Distill</td><td>8</td><td>2.3</td><td>6.26x</td><td>18.58</td></tr></table>

Table 2. Multi-layer generator distillation with inference time.

<table><tr><td>#layer numbers</td><td>2~7</td><td>8~11</td><td>12~14</td><td>15~50</td></tr><tr><td> $PSNR_{merged} \uparrow$ </td><td>22.51</td><td>21.99</td><td>21.36</td><td>20.65</td></tr><tr><td> $SSIM_{merged} \uparrow$ </td><td>0.8932</td><td>0.8869</td><td>0.8780</td><td>0.8610</td></tr></table>

Table 3. Effect of layer numbers on image-to-layers (I2L) generation quality. We evaluate the model’s performance across different ranges of layer numbers in the generated results.

# 1. Additional ablation experiments

# 1.1. Mixed Training with Variable Caption Length

Table 1 demonstrates the importance of caption diversity during training. Models trained with mixed caption lengths achieve the best generalization, with FID of 16.13 on short captions and 15.93 on long captions. Training exclusively on one caption type creates a domain gap: short-captiononly training degrades to 18.56 FID on long captions, while long-caption-only training achieves 16.15 FID, showing better robustness but still suboptimal on short captions.

# 1.2. Effect of Layer Numbers on Image-to-layer

Table 3 demonstrates our method’s scalability across different layer counts for the image-to-layers task. The model handles compositions ranging from 2 to 50 layers effectively, maintaining stable performance across this wide range. This flexibility enables decomposition of both simple designs and complex multi-element compositions without architectural modifications.

# 1.3. Analysis of Distilled Models

To evaluate the real-world efficiency of our approach, we conducted inference speed benchmarks on a single NVIDIA H200 GPU. We compared the standard baseline method (operating at 50 denoising steps) against our distilled MRT model at reduced inference steps (16 and 8 steps). As shown in Table 2, the baseline model requires 14.4 seconds to complete the generation process. In contrast, applying DMD2 distillation significantly accelerates inference. Specifically, our model achieves a 3.2× speed-up (4.5s) at 16 steps with negligible degradation in generation quality (FID increases only slightly from 16.02 to 16.21). Furthermore, reducing the inference budget to just 8 steps yields a massive 6.26× speed-up (2.3s), showing that our method successfully balances high-fidelity generation with interactive-level latency. We also present the generated samples and compare the original and distilled models in Fig. 1.

# 2. Attention Analysis of Image-to-Layer Model

To validate that our model learns meaningful semantic representations rather than merely memorizing layout priors, we visualize the pixel-wise attention maps generated during the decomposition process. Fig. 2 illustrates the correspondence between the generated transparent layers and their associated attention activations. As observed, the attention mechanism exhibits strong spatial localization capabilities. For each predicted layer, the attention weights (visualized as heatmaps) highly correlate with the semantic boundaries of the target elements. For instance, when reconstructing high-frequency components such as text (e.g., “Bundle of Joy” in the second case, “Love NEVER FELT...” in the third one) or fine-grained graphical elements, the attention is tightly focused on the relevant character strokes and shapes, effectively suppressing background noise. Conversely, for background patterns or larger geometric shapes, the attention acts more broadly to capture the texture and spatial extent of the region. This visualization confirms that the model successfully disentangles the composite image by attending to distinct visual features guided by the layout, ensuring that the resulting RGBA layers possess clean alpha mattes and coherent textures.

# 3. User study details

# 3.1. User Study on Text-to-Layer Task

To evaluate the generation quality of our models on the text-to-layer task, we conducted a user study comparing our method (MRT) with the baseline (ART). We employed a blind, pairwise comparison setup. For each sample, participants were first shown the input text prompt, followed by the corresponding results generated by MRT and ART displayed side-by-side. To eliminate positional bias, the display order (left or right) of these two results was randomized for each evaluation. Participants were asked to cast a three-way forced-choice vote—”Method A is better,” ”Method B is better,” or ”Tie”—across four distinct dimensions: (1) elements (layout), (2) visual appeal (aesthetics), (3) correctness of the text (typography), and (4) coherence and quality of each layer (harmonization).

![](images/29c72c7e6aaaf0ce63193a29d8b0e858bd240f93b858c6cc4fcd55f642268415.jpg)

<details>
<summary>text_image</summary>

baseline 50-NFE
bloods
distilled 16-NFE
bloods
distilled 8-NFE
GIVEAWAY
Free product sample for every purchase of $60
GIVEAWAY
Free product sample for every purchase of $60
Flash sale
Special Offer Sale 50% off
Flash sale
Special Offer Sale 50% off
Flash sale
Special Offer Sale 50% off
New assisted GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES COLLECTION
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
GLASSES SOLUTIONS
ORDERHOW
</details>

Figure 1. Generation quality of distilled models. We achieve up to 6x speed up without sacrificing the quality and fidelity of images.

The web-based evaluation interface is shown in Fig. 33, where two generated results are displayed side-by-side with the text caption provided on the right panel.

# 3.2. User Study on Image-to-Layer Task

For the image-to-layers task, we conducted a comprehensive user study by performing three separate pairwise comparisons between our method and three state-of-the-art baselines: (1) Ours vs. LayerD, (2) Ours vs. Lovart, and (3) Ours vs. Roboneo. Each comparison was run as an independent blind test. Participants in each study were presented with a three-image layout: the original input image was displayed as a central reference, while our method’s result and the corresponding baseline’s result were shown side-by-side. To eliminate positional bias, the display order (left or right) of our result and the baseline’s result was fully randomized in every trial. Participants were asked to make a three-way forced-choice vote (”Method A is better,” ”Method B is better,” or ”Tie”) based on three key metrics: (1) granularity, (2) layer integrity, and (3) layer quality.

The evaluation interface is illustrated in Fig. 34, where the reference input image is shown at the center with decomposition results from two methods displayed on both sides.

# 4. Limitations

Although our model demonstrates strong performance in the image-to-layer task, it faces challenges when applied to real-world photographs. Specifically, our method often fails to correctly handle shadows, resulting in segmented object layers that exclude shadow regions and leaving the shadows on the background layer, which leads to visual inconsistency. We attribute this limitation primarily to our training data: our model was trained exclusively on design datasets, which are planar and lack physical effects such as shadows, reflections, and refractions that commonly appear in natural scenes. Despite this domain gap, we were pleasantly surprised to find that our method can still generalize reasonably well to real images, even without any supervision on real-world multi-layer data. As shown in our illustrations, most objects are successfully separated, which we believe stems from the strong visual understanding capability inherited from the Qwen-Image backbone, demonstrating the robustness, adaptability, and scalability of our approach. In future work, we plan to extend our method to real-world image scenarios by collecting and training on datasets that include realistic visual effects such as shadows and reflections. We believe such extensions will further enhance the model’s ability to produce coherent and physically plausible layer decompositions.

# 5. Visualizations and Qualitative Analysis

# 5.1. Diverse Text-to-Layer Generation

We visualize the qualitative results of our Text-to-Layer task in Fig. 3 through Fig. 9. Our Masked Region Transformer demonstrates exceptional versatility in generating highfidelity multi-layer designs solely from textual descriptions. As shown in Fig. 3 through Fig. 8, the model successfully synthesizes coherent compositions ranging from simple layouts to complex designs with over 25 layers and even more, maintaining strict spatial consistency and stylistic harmony. A key advantage of our approach is the native support for diverse typography; Fig. 9 illustrates our unique overflow generation capability. Unlike prior methods that truncate content at the canvas edge, our model generates complete, full-size RGBA layers that extend beyond the visible background boundary, thereby preserving full editability and reusability for downstream compositional tasks. Furthermore, Fig. 10 highlights the model’s capability to render

![](images/9e87e2642de79446eb859ed910160f624fa5adc3cbbc692fa2de5055ddc3937d.jpg)  
Figure 2. Attention map visualizations of image-to-layers task. We demonstrate the interpretability of our model by visualizing the internal attention weights during the layer generation process. Left: The input composite image and its corresponding layout. Right: The decomposition results. The top row displays the predicted transparent layers, while the bottom row shows the corresponding attention maps overlaid on the input image. Red regions indicate high activation values. The results highlight that the model’s attention is semantically selective, accurately aligning with specific visual elements (e.g., text, foreground objects, background patterns) to generate high-quality, disentangled layers.

accurate visual text across multiple languages, including Chinese, ensuring practical utility for global design applications.

# 5.2. Comparative Analysis of Image-to-Layer

In Fig. 11 through Fig. 18, we provide a comprehensive qualitative comparison between our approach and stateof-the-art baselines, including LayerD, Lovart, and Robo-Neo. The results consistently demonstrate that our method establishes a new standard for layer decomposition quality. While commercial systems like RoboNeo often introduce visual artifacts or fail to produce clean transparency, and academic baselines like LayerD tend to produce overly grouped layers that limit editing flexibility, our Masked Region Transformer achieves a superior balance. Our method excels in generating precise alpha mattes, maintaining semantic integrity, and achieving appropriate decomposition granularity (e.g., separating distinct visual elements rather than merging them). This is particularly evident in complex overlapping regions, where our model successfully disentangles elements that other methods fail to separate.

# 5.3. Scalability on Layer Counts in Image-to-Layer

To evaluate the robustness of our framework, we visualize image-to-layers decomposition results across varying degrees of complexity in Fig. 19 through Fig. 23, ranging from 6 layers up to 16 layers. These visualizations confirm that our architecture scales effectively without performance degradation. The model maintains consistent quality in boundary detection and content preservation in cases of a wide range of layer counts. This stability across diverse layer counts validates the efficacy of our masked attention mechanism, proving that the model can handle the structural complexity of professional-grade graphic designs.

# 5.4. Context-Aware Layer Addition

Fig. 24 demonstrates the capabilities of our layers-to-layers task, specifically focusing on layer addition. Here, we simulate a user editing workflow where new elements—such as text or decorative objects—are inserted into an existing design based on text prompts and specified bounding boxes. The results show that our model does not merely paste isolated objects; instead, it generates new layers that are contextually aware, matching the lighting, perspective, and artistic style of the existing layers. By conditioning on the full composition, the added layers harmonize seamlessly with the original design, preserving the global aesthetic while fulfilling the user’s semantic requirements.

# 5.5. Layer Restylization and Harmonization

In Fig. 25, we showcase the layer restylization capability, where user-provided assets are transformed to align with a target design’s visual identity. Our model effectively transfers style while preserving the geometric structure of the input asset. The visualization demonstrates that our singlepass generation approach ensures cross-layer consistency, successfully adapting the color palette, texture, and artistic rendering of external assets to match the pre-existing composition. This capability is essential for unifying disparate elements into a cohesive graphic design.

# 5.6. Layout Generalization in Text-to-Layer

Fig. 26 and Fig. 27 present an analysis of the interplay between text prompts and spatial controls. In these experiments, the text prompt contains implicit or explicit descriptions of element positions, while we simultaneously provide varying spatial layouts (bounding boxes) that may conflict with these textual descriptions. Remarkably, the results demonstrate that our model exhibits strong adherence to the user-provided layout, effectively overriding the spatial biases present in the text prompt while retaining the semantic content. This confirms that our framework successfully disentangles semantic generation from spatial arrangement, allowing users to enforce arbitrary layouts—such as moving a title from the top to the bottom—without compromising the generated content’s quality or the prompt’s semantic fidelity.

# 5.7. Layout-Guided Image Decomposition

For the Image-to-Layer task, Fig. 28 through Fig. 30 visualize the input raster images alongside their corresponding layout structures (bounding boxes and Z-order) used during inference. These examples illustrate how the model utilizes layout information—whether derived from automatic detectors or manual annotation—as a structural prior to guide the decomposition process. The visualizations show that the model accurately resolves ambiguities in the raster image by leveraging the provided spatial cues, resulting in semantically meaningful layers that strictly conform to the specified boundaries. This highlights the model’s ability to produce controllable and predictable decompositions essential for professional editing workflows.

# 5.8. Generalization to Natural Scenes

Although our model is trained exclusively on graphic design datasets (posters, flyers, etc.), Fig. 31 demonstrates its zeroshot generalization capability to real-world natural images. The model successfully segments objects from photographs into transparent layers, leveraging the strong visual understanding inherited from the Qwen-Image backbone. However, we observe a specific limitation due to the domain gap: unlike flat graphic designs, real-world scenes contain complex physical lighting effects. Consequently, the model often fails to associate cast shadows with their respective objects, leaving shadows on the background layer rather than the object layer. Despite this limitation regarding physical lighting effects, the structural decomposition remains surprisingly robust for out-of-domain data.

# 5.9. Failure Cases

Finally, we analyze representative failure cases in Fig. 32 to provide a balanced view of our method’s current limitations. We observe a common issue across all four tasks: some transparent backgrounds are decoded into gray instead of remaining transparent. This ambiguity arises because our VAE encoder currently uses a 3-channel input, which compresses transparent layers into a gray representation that the decoder sometimes misinterprets. Future work could address this by adopting a 4-channel encoder or alternative encoding schemes. Additionally, we identify task-specific limitations: 1) for text generation, the model sometimes struggles with rendering very small glyphs accurately; and 2) for layer-to-layer tasks, we occasionally observe failures in identity preservation (IP) and instruction following, particularly when complex style transfer or precise object insertion is required. These cases outline critical directions for future research in multi-layer generative modeling.

# Prompt

The image presents a cheerful and vibrant composition that celebrates the arrival of spring. The background is a soft, pastel pink, creating a warm and inviting atmosphere that evokes feelings of renewal and joy This gentle hue serves as a perfect canvas for the other elements, enhancing their visual appeal.At the center of the image, the phrase “Hello Spring"is prominently displayed ina playful and stylish font.The word“Hello"isrenderedinabold,dark pinkcolor,while“Spring is written ina flowing, cursive script that addsa touch of elegance.The text is enclosed within a whimsical, cloud-like shape that features decorative elements such as small droplets and a flower motif, further emphasizing the theme of springtime Along the bottom of the image, a row of fresh pink tulips adds a natural and lively touch. The tulips are in various stages of bloom, showcasing their delicate petals that range from soft pink to a slightly deeper shade. Their green leaves provide a vibrant contrast against the pink background, grounding the composition and adding a sense of freshness. The tulips are arranged in a way that draws the viewers eye across the image, creating a harmonious balance between the text and floral elements. The overall artistic style of the image is modernand graphic, with clean lines and a minimalistic approach that enhances its appeal. The use of soft edges and a cohesive color palette contributes to a light and airy feel, making the image feel uplifting and optimistic.The theme of the image is clearly celebratory, capturing the essence of spring asa time of growth and renewal.The mood conveyed isone of happinessand positivity,making it an ideal visual for various contexts.Thisimage could be effectively used for socialmedia posts,greeting cards, or promotional materials for spring events,floral shops, or seasonal sales.Its engaging design and cheerful message make it a versatile choice for anyonelooking to embrace the spirit of spring.

# Layout

![](images/bedf80f2c226841c27f7a8f5049a888e94e6324ada5331f77acf9ddb1bda2210.jpg)

<details>
<summary>natural_image</summary>

Two adjacent squares, one light blue and one orange, on a plain background (no text or symbols)
</details>

# Generated

![](images/4e8c042e0551552e0fb08078f40c23755d2034f7f3ef9a3e5ecb92a770560a58.jpg)

<details>
<summary>text_image</summary>

Hello
Spring
</details>

# Layer Images

![](images/414a248578006a3701a5227449c66acf8ec94f26892c0a8a824d32d41f10b997.jpg)

The image presents a whimsical and playful background that features a variety of celestial motifs, including planets, stars, and swirling designs, all rendered in a soft lavender hue.The background is predominantly white, which enhances the lightness and brightness of the overallcomposition,allowing the purple illustrations to stand out vividly. The celestial elements are scattered across the surface, creating a sense of movement and exploration, reminiscent of a child’s imagination. At the center of the image lies a young girl dressed in a delicate, peach-colored dress that features a bow at the waist. The dress has a soft, tulle-like texture that adds a sense of whimsy and elegance. She is positioned in a relaxed pose, with her arms raised and her hair styled in playful pigtails, contributing to the joyful and carefree atmosphere of the scene. The girl’s shoes are light pink,complementing the color of her dress and enhancing the overall pastel color scheme.Prominently displayed at the bottom of the image is a banner with the text "You're a Star! The text is presented in a bold,playful font that is easy to read and adds to the cheerful tone of the image.The banner hasa soft, brushstroke effect ina light beige color,which contrasts nicely with the white background and ties in with the peach tones of the dress. The artistic style of the image is modern and playful, with a digital illustration quality that appeals to both children and adults. The use of soft edges and light colors creates a dreamy, imaginative feel, making it suitable for a variety of contexts,such as children’sbirthday invitations,motivationalposters,orsocialmediapostscelebratingindividuality and creativity. Overal, the theme of the image is uplifting and celebratory, conveying a message of encouragement and positivity. The combination of celestial imagery and the phrase“You're a Star!" suggests a focus on self-worth and the idea that everyone has their own unique brilliance to shine.

![](images/2bf56d65941cf347e87e98bb5da6e9c76cd1ac11fbabafd585dced8bb7a9115e.jpg)

<details>
<summary>text_image</summary>

#1
#2 #3
</details>

![](images/525eebf83a14442269bf231a3566b61dee70c28cb349f4740414f4b5ec03cbf8.jpg)

<details>
<summary>text_image</summary>

You're a Star!
</details>

![](images/5946d3d4c6b945164165330a1cf6c0d8594309798b22a1a7766be996e38128a3.jpg)

The image features a vibrant and playful background in a deep blue hue, which serves asastrikingcanvasforthecentralelements.Thisrichcolorcreatesasenseof warmthandcoziness,perfectlycomplementing the theme of comfort during cold weather.At the center of the composition is a character depicted in a whimsical, cartoonish style. The character is wrapped in a large, bright red blanket adormed with abstract,leaf-like patterns, which adds a touch of visual interest and texture. Theblanket’sbold color contrasts beautifully with the bluebackground, enhancing the overall warmth of the image. The character is holding a steaming cup, likely of tea, with a stylized design that suggests warmth and comfort. Surrounding the character are playful, handwritten text elements that read,"\*A very hot tea for a cold winter\*."The text is in a white, flowing font that adds a lighthearted and inviting feel to the image.The placement of the text, curving around the character, creates a sense of movement and draws the viewer's eye toward the central theme of warmth and relaxation. The artistic style of the image is modern and illustrative, characterizedby bold colors and simplified forms.The use offlat design techniques, with minimal shading and soft edges, contributes to a cheerful and approachable aesthetic. The overalltheme of the image is cozy and inviting,evoking feelings of comfort and warmth during the winter months. The mood conveyed is one of relaxation andenjoyment,makingitanidealrepresentationofthesimplepleasures of sipping hot tea on a chilly day.This image could be effectively used in various contexts, such as social media posts promoting winter beverages, greeting cards for the holiday season,oraspart ofamarketing campaignforteabrands.Its cheerfulandengaging design makes it suitable for any platform aiming to evoke warmth and comfort.

![](images/35fd85d5f6f13ab3056feb14e22eaf4220ef43fa86f9d3e30316ebe47ea1c44e.jpg)

<details>
<summary>text_image</summary>

#1
#2
#3
#4
</details>

![](images/34ca5c2bd825488f084a3f7e99e4fc8c0efe2c0f3957f74aadf135691d657610.jpg)

<details>
<summary>text_image</summary>

a hot
hole cold
ciano!
tanning
tea too
cold
winter.
</details>

![](images/4b001da371b062f2b259c20479451046b94af5e79825f04593e8ed7ecb532e81.jpg)

The image presents a striking and modern promotional design, characterized by a bold colorscheme anddynamic composition.Thebackground featuresadramatic blackmarble texture, which adds depth and sophistication to the overall aesthetic. This dark backdrop contrasts sharply with the vibrant red and pink hues that form an abstract shape in the center, creating a visually engaging focal point. At the forefront, a figure is depicted in a stylish pose, wearing a black leatherjacket over a white top. The positioning of the figure, slightly off-center,draws the viewer's eye and adds a sense of movement to the composition. The interplay of light and shadow enhances the textures of the clothing, emphasizing the fashionable elements of the image.Prominentlydisplayedatthetopisthe text“NewCollction,"renderedinan elegant,bold font that is colored ina bright red. This choice of typography not only stands out against the background but also conveys a sense of urgency and excitement about the new offerings.Below the figure,the phrase"Special Offers Sale 50%offispresentedinaslightlysmalerfont,maintainingthesamecolorscheme, whichreinforces the promotional message.Additionally, a circular button labeled “Shop Now"is included, designed in a soft peach color that complements the overall palette while inviting immediate action. The artistic style of the image leans towards contemporary graphic design,utilizing clean lines and amix of textures to createa polished look.The combination of the marblebackground with the vibrant colors and modern typography evokes a sense of luxury and trendiness, appealing to a fashion-forward audience. The overall theme of the image is promotional and stylish, aimed at attracting attention to a new fashion collection.The mood conveyed is energetic and inviting, encouraging viewers to engage with the brand and explore the offerings. This image would be well-suited for use in digital marketing campaigns, social media advertisements, or as part of an online store's promotional materials, effectively capturing the essence of modern fashion retail.

![](images/5a3d2ea5ad0c83d9d5885234f484f9acf1132e675bfa54b45272cc12b5da33e1.jpg)

<details>
<summary>text_image</summary>

#1
#3
#2
#5
#6
#4
</details>

![](images/3d3a0c860a04e782856d6cdf7861ba92afb55cbb6c61f50ba5e3443331524e62.jpg)

<details>
<summary>text_image</summary>

New Collection
Shop Now
Special Offers Sale 50% off
</details>

![](images/f07448367ae2783f176ffa63f9ca2cac64e23c16120a36c38ad1047a37c0a496.jpg)

![](images/de99658fa29743dad993144cc094506b468b72e8b1c1c46dec1c8577dcfc0353.jpg)  
Figure 3. Text-to-layers generation examples. We visualize diverse text-to-layers generation results from our method, showing the input text prompts and corresponding multi-layer outputs. Each example displays individual transparent RGBA layers along with the merged composition. Our approach generates coherent multi-layer designs that maintain spatial consistency, stylistic harmony, and accurate layer boundaries, demonstrating strong alignment between text prompts and layered compositions.

# Prompt

The image presents a vibrant and playful design celebrating the “Festa della Repubblica Italiana,"or Italian Republic Day. The background is a bright turquoise blue, which creates a lively and cheerful atmosphere, evoking feelings of joy and festivity.At the center of the composition is the phrase“FESTA DELLA REPUBBLICA ITALIANA," prominently displayed inbold, white andred typography. The text is large and eye-catching, ensuring that it captures the viewer's attention immediately. The use of red for “TALIANAadds a dynamic contrast against the white, emphasizing the significance of the celebration. Surrounding the central text are various iconic landmarks and symbols associated with Italy, illustrated in a minimalist and cartoonish style.These elements include theColosseum,the Leaning Towerof Pisa the Arc de Triomphe, and the famous gondolas of Venice, among others. Each landmark is outlined in a simple black line,filled with soft colors that maintain a cohesive aesthetic. The landmarks are arranged in a circular pattern, creating a sense of unity and celebration, as if they are coming together to honor the Republic. Additionally,smalldecorativeelements suchas clouds,airplanes,and asunare scatered throughoutthedesign,enhancing thefestive themeandsuggestingasense of movement and travel, which is often associated with national pride and exploration. The overall artistic style is modern and graphic, with clean lines and a playful approach that appeals to a wide audience. The use of bright colors and simple shapes contributes to a lighthearted and celebratory mood,making the image suitable for various contexts, such as social media posts, event promotions, or educational materials related to Italian culture and history. In terms of symbolism, the image encapsulates national pride and unity, celebrating Italy's rich heritage and the significance of the Republic Day. It could be effectively used in promotional materials for events, festivals, or educational campaigns aimed at raising awareness about Italian culture and history.

# Layout

![](images/4a4c35cdbe6a41fd5a950bb6bd5c96c41f78c49f38e73593937b96a6318d4bea.jpg)

<details>
<summary>bar_stacked</summary>

| Category | Segment 1 | Segment 2 | Segment 3 | Segment 4 | Segment 5 | Segment 6 | Segment 7 |
|---|---|---|---|---|---|---|---|
| #1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| #2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| #3 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| #4 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| #5 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| #6 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| #7 | 0 | 0 | 0 | 0 | 0 | 1 | 1 |
</details>

# Generated

![](images/6b62d3a4706088547cfc30da97fd69f7b46021fd45d4fe4e83500aaccbe9cacb.jpg)

<details>
<summary>text_image</summary>

FESTA DELLA
REPUBBLICA
ITALIANA
</details>

# Layer Images

![](images/f053ec8db4b8f2e120b112e698a73e6c714b8f9de5256ef4a7f61905e54a7d99.jpg)

<details>
<summary>text_image</summary>

Image with a collage of colored panels, including a small illustration of Italian city names and the word 'ITALIANA' in red.
</details>

The image presents a vibrant and festive design celebrating the New Year, set against a deep navy blue background that evokes a sense of night and celebration. The color scheme predominantly features shades of blue, ranging from light to dark, which creates a harmonious and cool atmosphere. The background is smooth and uniform, providing a striking contrast to the other elements in the image. At the center of the composition are the bold, oversized numbers "2," \*2," and "4," which represent the year 2024.The first two “2"s are styled in a gradient of light blue to dark blue, while the last number "4" is a solid light blue, giving it a slightly different visual weight. The numbers are arranged vertically, with the "4" positioned to the right, creating a balanced and dynamic layout. Interspersed among the numbers is the phrase “Happy New Year,“ written in anelegant, flowing script.The text is white, which stands out prominently against the dark background, and is positioned vertically between the numbers, enhancing the celebratory message of the image. The font style is playful yet sophisticated, adding to the festive mood.To the right of the numbers, there is a stylized disco ball a classic symbol of celebration and parties.The disco ball is depicted in a light bluehue,surrounded by soft clouds and small stars, which adds a whimsical touch to the design. The reflective squares on the disco ball are illustrated in a way that suggests light and movement, further enhancing the festive atmosphere. The overall artistic style of the image is modern and graphic, with clean lines and a minimalist approach that emphasizes the key elements without overwhelming the viewer. The use of gradients and soft shapes contributes to a sense of depth and dimension, while the playful arrangement of the numbers and text creates a lively and engaging composition. The theme of the image is celebratory and joyful,perfectly capturing the excitement of welcoming a new year. Themood conveyed isone of optimism and festivitymaking it suitable for various contexts,such as social media posts,digital invitations,or promotional materials forNew Year's events.Thecombinationofbold typographyfestiveimagery,anda cohesive color palette makes this image an effective and appealing representation of New Year celebrations.

![](images/da1a89c2ea79e5c118f730d4a383df37fd74338b2832d43d472646d4493b3bad.jpg)

<details>
<summary>text_image</summary>

#1
#3
#5
#2
#4
#6
#7
#8
</details>

![](images/a59a99fe8392d9d8e090a9488c4804678fe6eae0c911e2cbbc361f1c2b84f356.jpg)

<details>
<summary>text_image</summary>

2
Happy New Year.
2
4
</details>

![](images/99fcda0e3260257ed5d98fb903bde0313879c4a3ba138f532a5aae076b47d76f.jpg)

<details>
<summary>text_image</summary>

2
2
4
</details>

The image presents a visually appealing advertisement for a home furnishings and decor shop, featuring a rich teal background with a textured, painterly effect that adds depth and sophistication. The color scheme is predominantly teal, which evokes a sense of calm and elegance, complemented by the warm tones of the food and decor elementsshowcasedinthepolaroid-styleimages.Attheforefront,twopolaroid framesare prominently displayed. The left frame features abeautifully arranged plate of food, showcasing slices of roasted pumpkin alongside a vibrant red apple and a glass of rosé wine. The plate is set against a backdrop of white dinnerware, with polished silver cutlery elegantly placed beside it.This arrangement suggests acozy inviting dining experience, perfect for gatherings or special occasions. The right frame captures a more elaborate table setting, adomed with a rustic wooden platter that holds an assortment of cheeses, nuts, and a smalloaf of bread. The table is further enhanced by decorative elements such as tal, lit candles and a bouquet of flowers, which add a touch of warmth and charm.The use of soft, neutral tones in the tableware contrasts beautifully with the vibrant colors of the food, creating a harmonious and inviting atmosphere. The text elements are strategically placed to draw attention. The phrase "Order online 24/7" is written in a stylish, modern font that is easy to read, while the larger text “Home furnishings& decora is presented in an elegant script, enhancing the overallaesthetic of the advertisement.The text is white, providing a striking contrast against the teal background, ensuring visibilityand clarity.The artistic style of the imageleans towards a contemporary and lifestyle-oriented aesthetic, with a focus on warmth and comfort. The use of natural light in the food photography creates soft shadows and highlights, adding a realistictouchthatinvites viewers to imagine theexperience ofenjoying themeals and decor The overall theme of the image is centered around home and hospitality, conveying a mood of warmth, comfort, and togetherness. It suggests a lifestyle that values quality dining experiences and beautifully curated home environments. In terms of symbolism, the food and decor elements represent nourishment and the joy of gathering,making theimage suitable for promotingproducts relatedto homedining and entertaining.This advertisement could effectively be used on social media platforms,in online marketing campaigns,or as part of a promotional flyer, appealing to customers looking to enhance their home dining experiences.

![](images/35e0c9373fc5972c487307faa1248b5685451733150d560f87c0fc09ec46168d.jpg)

<details>
<summary>text_image</summary>

#1
#4
#6
#7
#2
#5
#8
#3
</details>

![](images/db74428fcf6c1f8bf32009e4604b91a3998a76bd25da253d36961f199c552f82.jpg)

<details>
<summary>text_image</summary>

Order online 24/7
Home furnishings & decor
</details>

![](images/a441729b61067fd5fdefba44fddb520b1edff70af02c7a420db62612b5572c6e.jpg)

<details>
<summary>natural_image</summary>

Grid of six panels showing a dark teal background, four black squares, and two small food items on the right (no text or symbols)
</details>

The image presents a modem and inviting promotional graphic designed to announce thereopening of astudio store. The background features asoftly blurred interior of a clothing store, showcasing a rack of garments in various colors, including muted tones of blue, beige, and green. This creates a warm and welcoming atmosphere suggesting a casual yet stylish shopping experience. At the forefront, a prominent rectangularoverlay in alight cream color houses the main text.The bold, uppercase phrase “WERE BACK!" is centered within this overlay, rendered in a striking black font that captures immediate attention. Below this, in a smaller font, the message reads, \*Studio store is open now. No appointment necessary," providing essential information in a clear and concise manner. The text is easy to read against the light background, ensuring that the message is communicated effectively. A call-to-action buton,COMEFORAFG,"ispositionedcentrallythintheovrlayauinga rounded black background that contrasts sharply with the lighter text. This design choiceemphasizes the invitation for customers to visit the store, enhancing the overall accessibility of the message. The artistic style of the image leans towards a contemporary digital design,characterized by clean linesand aminimalist aesthetic. The use of soft edges and a subtle drop shadow around the overlay adds depth, making the text and button appearto stand out from the background. The overalltheme of the image is one of rejuvenation and excitement, conveying a sense of welcome and readiness to serve customers again. The mood is upbeat and inviting, encouraging potential shoppers to return to the store after a period of closure. In terms of symbolism, the image reflects themes of community and connection, suggesting that the store is eager to reconnect with its clientele. This graphic could be effectively used across various platforms, such as social media posts,email newsletters,or website banners,to attract attention and drive foot traffic to the store

![](images/237a5048b290c41b97b8e5d72af833b18723a633ca9f455421d724532766148d.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| #2 | 1 |
| #3 | 0 |
| #4 | 0 |
| #5 | 1 |
| #6 | 0 |
| #7 | 1 |
| #8 | 0 |
| #9 | 0 |
</details>

![](images/56e5376b92d48561e9e8c20bb2e82c0bc424b01c05d554b436e1c0ce17a211a0.jpg)

<details>
<summary>text_image</summary>

WE'RE BACK!
Studio store is open now.
No appointment necessary
COME FOR A FITTING
</details>

![](images/55df0ed449b61354093a52e5f6690422641df4ef8f786b57a9000ab70a4540da.jpg)

<details>
<summary>text_image</summary>

WE'RE BACK
WELCOME
WELCOME
WELCOME
</details>

Figure 4. Additional text-to-layers generation examples. More examples demonstrating our method’s capability to generate multi-layer designs from text descriptions. These results showcase the diversity of generated layouts, layer compositions, and visual styles.

#

The image presents a vibrant and enticing advertisement for a seafood restaurant's lunch special. The background is a rich teal color, providing a striking contrast to the other elements in the composition.This deephue is complemented by a lighter green rectangular banner at the top, which reads “Today’s Offer,"creating a clear focal point that draws the viewer's attention. At the center of the image is a beautifully plated dish featuring a whole fish, which is the star of the advertisement.Thefishispresentedwithmeticulousatentiontodetail,showcasing its grilled skin with distinct char marks, and garnished with fresh herbs and colorful edible flowers. The dish is artfully arranged, with a bed of vibrant orange garmishes that add a pop of color and texture, enhancing the overall visual appeal. Prominently displayed in bold, modern typography is the phrase GET YOUR SPECIAL LUNCH MENU," which is positioned just above the fish. The text is in a large, light green font that stands out against the darkerbackground, ensuring readability and impact. Below this, the price “ONLY \$5"is highlighted in a playful red starburst shape,further emphasizing the affordability of the offer and enticing potential customers. The overall artistic style of the image is contemporary and promotional, utilizing clean lines and a well-organized layout that effectively communicates the message. The use of high-quality photography for the food adds a realistic touch, making the dish look appetizing and inviting. The theme of the image is promotional and culinary, aimed at attracting customers to the restaurant for a special lunch deal. The mood conveyed is one of excitement and indulgence, encouraging viewers to take advantage of the offer. In terms of symbolism, the fish represents freshness and quality,key atributes foraseafoodrestaurant.The vibrant colorsand appealing presentation suggest a delightful dining experience.This image could be effectively used in various contexts, such as social media posts, flyers, or website banners, to promote the restaurant’s lunch specials and attract a wider audience.

# Layout

![](images/6336fe79816c6ebbbb39807ae090d8eb05866760fd2a2611fadbb85aaf78e746.jpg)

<details>
<summary>text_image</summary>

#1
#2
#5
#6
#7
#9
#8
#4
#3
#10
</details>

# Generated

![](images/74c870b511a8a0bca07a1aaf18dc87cd6d8ff9454c24c22b3c1dd171f4657de7.jpg)

<details>
<summary>text_image</summary>

Today's Offer
ONLY
$5
SCC: 2019A NO SCABRICAL POINT ELECIL
GET YOUR
SPECIAL
LUNCH MENU
www.etcally.greatsite.com
</details>

# Layer Images

![](images/b769d931f6f7a7abf38634b09c44cd009a3191dc814b1a45f12e687b6af3870f.jpg)

<details>
<summary>natural_image</summary>

Grid of transparent squares with a small colorful object in the center (no text or symbols)
</details>

The image presents a vibrant and engaging advertisement for children's eyewear, set against a playful and colorful background. The overall background features a rich purple hue, which serves as a bold canvas for the promotional elements. This purple is framed by a lively orange border adorned with small polka dots, adding a cheerful and inviting touch to the design.At the top of the image, the text “CHILDREN'S EYEWEAR" is prominently displayed in large, white, bold letters, ensuring that the message is clear and eye-catching. Below this, the phrase "GET 40% OFF"is presented in a smaller font, also in white, which contrasts well against the purple background, drawing attention to the promotional offer. The central part of the image includes a green oval shape that contains additional text and graphic elements. Inside this green area, the phrase “CRAFTED FOR COMFORT, STYLED FOR SMILES"is written in a friendlyandapproachablefont,reinforcingtheproduct'sappealtobothchildrenand parents.The green color of the oval contrasts nicely with the purple background, creating a sense of balance and harmony, Flanking the green oval are two playful graphic elements resembling puzzle pieces,colored in a bright yelow-orange.These elements add a whimsical touch to the composition, enhancing the overall theme of fun andcreativity associated with children’s products.The artistic style of the image is modern and graphic, utilizing bold colors and clear typography to convey its message effectively. The use of contrasting colors and playful shapes contributes to a lively and engaging atmosphere, making it suitable for a target audience of parents looking for stylish and comfortable eyewear for their children. Overall, the image conveys a theme of joy and acoessibility, with a mood that is both inviting and cheerful.It is well-suitedfor use inmarketing materials, social media promotions, or as part of a website dedicated to children's products, effectively capturing the attention of potential customers and encouraging them to take advantage of the discount offer.

![](images/2ba76cb9575ed0ac36e72816bbb37d7d0a8005f33011d0518d04f5ca2a8c4c69.jpg)

<details>
<summary>text_image</summary>

#2
#3
#6
#8
#5
#9
#4
#10
#11
#7
</details>

![](images/de6fe0822f830beff7e33b347f53dd786395da2192ff7c2c104770a8a0bd2550.jpg)

<details>
<summary>text_image</summary>

CHILDREN'S
EYE AWEAR
GET 40% OFF
BEIJING KAT BREAD
ENJERON/EBITDAY
CRAFTED FOR COMFORT, STYLED FOR SMILES
</details>

![](images/380e5c0b45126b253501c59757abfb47a71b8adfaf65ce447d5ea2af9c59cd67.jpg)

<details>
<summary>text_image</summary>

CHILDREN'S EVEWEAR
</details>

The image presents a warm and inviting scene that captures the essence of family camping during the autumn season. The background features a soft, cloudy sky with shades of gray, suggesting a cool, overcast day by the water. This natural setting is framed by the vibrant orange fabric of a tent,which creates a cozy and welcoming atmosphere. The orange hue contrasts beautifully with the cooler tones of the sky and water, adding a touch of warmth to the overall composition. In the foreground, a family of four stands together, facing away from the viewer.They are positioned centrally within the tent’s opening, creating a sense of intimacy and connection. The family members are dressed in autumn-appropriate clothing, with the adults wearing brown and green jackets, while the children are clad in a mix of colors, including a green coat and a gray beanie. This choice of attire not only reflects the season but also emphasizes the theme of outdoor adventure and togetherness.At the bottom of the image,a dark brown banner spans the width, providing a strong visual anchor. The text within the banner reads “FAMILY CAMPING" in bold, white letters, which stand out prominently against the dark background. Above this, in a smaller, golden font, the words“AUTUMN SEASON"and “JOIN OUR FESTIVAL EVENT"invite viewers to participate thefestivities.Acircular badgein the top right cormer states“LIMITED TICKET!"in a playful font, adding a sense of urgency and excitement to the event. The artistic style of the image is modern and clean, with a focus on realism that captures the genuine expressions and postures of the family. The use of soft edges and natural lighting enhances the inviting mood making it feel both relatable and aspirational. The overall theme is one of family bonding and outdoor exploration, evoking feelings of nostalgia and warmth associated with autumn activities. Symbolically, the image represents togetherness,adventure,andthejoyofspendingqualitytimewithloved ones in nature.It could be effectively used for promotionalmaterials related to family camping events, social media posts, or as part of a marketing campaign for outdoor gear and activities. The combination of visual elements and text creates a compelling invitation for families to engage in memorable experiences during the autumn season.

![](images/ca4d4839a7037f9081a5a4a82acaa8be5eb146ebf7d422436c731840cbf26362.jpg)

<details>
<summary>text_image</summary>

#1
#2
#3
#4
#5
#6
#7
#8
#9
#10
</details>

![](images/ee77e0e8eef8de3883f4059a41be33b04ac4b6bd487ec9e48a849ad1f79ade4e.jpg)

<details>
<summary>text_image</summary>

AUTUM SEASON
FAMILY CAMPING
JOIN IUR FESTIVAL EVENT
</details>

![](images/295193c84b81343b33570f76ef54f6186324345aad73a4780a6e33099ac7dfa4.jpg)

<details>
<summary>natural_image</summary>

Grid of transparent square tiles with a small inset image showing people inside a tunnel (no text or symbols)
</details>

The image presents a clean and modern design that emphasizes the theme of contemporary architecture with a focus on sustainability. The background is a soft, muted white, which provides a neutral canvas that allows the main elements to stand out prominently. Surrounding the central design is a thin green border, enhancing the overall aesthetic and framing the content effectively. At the center of the image is a stylized illustration of a modern house, characterized by its triangular roof and wooden exterior. The house features large windows that reflect a contemporary architectural style,anditiscomplementedbyasmalldeckareaadomedwitha railing.To the left of the house, there are two stylized trees, rendered in a vibrant green, which add a touch of nature and reinforce the theme of eco-friendliness. The house is depicted in a simplified, almost cartoonish style, which contributes to a friendly and approachable feel. Prominently displayed above the house is a circular badge that reads “SAVE 1o%"in a playful font, suggesting a promotional offr.Thiselement is designedtocatchtheviewer’sattentionandis colored in a bright green that harmonizes with the overall color scheme. Below the house,the text“MODERNARCHITECTURALDESIGNS"ispresented inbold,uppercase letters, using a darker green that contrasts well with the lighter background. This text is clear and assertive,effectively communicating the primary focus of the image. Further down, a smaller text block provides additional information: "Energy-effcient systems, renewable materials, and passive design strategies." This text is written in a more subdued font, ensuring that it complements the bolder headings without overwhelming them. The overall composition is well-balanced, with the house and trees anchoring the visual elements while the text guides the viewer's eye. The artistic style of the image leans towards a minimalist and modern illustration, utilizing flat design techniques with clean lines and simple shapes. The use of green tones throughout the image not only emphasizes the theme of sustainability but also evokes feelings of growth and harmony with nature. The overalltheme of theimage is promotional and informative,aimedat attracting potential customers interested in modern architectural solutions that prioritize energy eficiencyand sustainability.Themood conveyed is optimisticand forward-thinking, appealing to environmentally conscious individuals or businesses. This image could be effectively used inmarketing materials,such as flyers,social media posts, or website banners,to promote architectural services or products that align with modern, eco-friendly design principles.

![](images/ceec0bc631202eaadf1ed1341ec6e4ac27cb97000230105276d4412976d80485.jpg)

<details>
<summary>text_image</summary>

#1
#6
#10
#8
#9
#11
#3
#4
#2
#5
</details>

![](images/0c70229da2b8ad769ecc5bfefeeddc069f15fc9ebb681ff8038d8b96c15fb2e2.jpg)

<details>
<summary>text_image</summary>

SAVE
10%
MODERN
ARCHITECTURAL
DESIGNS
Energy-efficient systems, renewable materials,
and passive design strategies.
</details>

![](images/a52069991c0357facda7fc3104570f5c759bb0ea30fedc986e795a8d623048b6.jpg)

<details>
<summary>text_image</summary>

MODERN
ARCHITECTURAL
DESIGNS
</details>

Figure 5. Additional text-to-layers generation examples.

# Prompt

The image presents a vibrant and engaging design that captures the spirit of adventure.The background features a rich, dark green color that providesa strong contrast to the lighter elements in the foreground. This deep hue evokes a sense of nature and exploration, aligning perfectly with the theme of outdoor adventures. At the center of the image, two individualsare depicted, engaged ina moment of planning their journey. They are holding a map, which serves as a focal point, symbolizing exploration and discovery. The individuals are dressed in casual outdoor attire,suggestingtheyarereadyforanadventure.Thetexturesoftheirclothing, combined with the natural setting behind them, enhance the authenticity of the scene. The top of the image prominently displays the phrase “OUR TIME ADVENTURE" in a bold, playful font. The word "ADvENTURE" is highlighted in a warm, golden yellow, which stands out against the darker background and draws the viewers attention.This choice of color adds a sense of excitement and positivity to the overallmessage. Beneath the main title, there is a motivational tagline:“Embark on your next adventure with peace of mind. Prioritize your body’s safety and health every step of the way." This text is presented in a clean, white font that is easy to read and complements the overall design. The message encourages viewers to embrace adventure while also being mindful of their well-being. The artistic style of the image leans towards a modern, graphic design approach, with a mix of photography and digital elements. The use of dashed lines and a paper airplane graphic adds a whimsical touch,suggestingmovementandtheideaoftravel.Overal,thethemeoftheigeis adventurous and motivational, conveying a sense of excitement and readiness for exploration. The mood is uplifting and encouraging, making it suitable for various contexts,such as travel blogs, social media posts, or promotional materials for outdoor activities. This image effectively inspires viewers to embark on their own adventures while keeping their health and safety in mind.

# Layout

![](images/a11d9e615597fc8f4041de882a12d719a023cb4ccb5f8f294fd3aaf65db5f077.jpg)

<details>
<summary>text_image</summary>

#10
#3
#4
#2
#6
#5
#9
#7
#8
#11
</details>

# Generated

![](images/afab7dc297e3293ad12731cc97d181a6a7e940ec281290085c1669b62e57ff1b.jpg)

<details>
<summary>text_image</summary>

OUR TIME
ADVENTURE

Asians do your next adventure with respect
of mind. Priorize your body's safety and
health every step of the way
</details>

# Layer Images

![](images/681e092cc1703092b821b59a76e41ade67537680f43f3bf6269bdedbfac4c931.jpg)

<details>
<summary>natural_image</summary>

Grid of transparent background tiles with a small dark square icon in the top-left corner (no text or symbols)
</details>

The image presents a vibrant promotional design for a comedy event titled "Comedy Night."Thebackgroundfeaturesastagesetingwithsoft,drapedcurtainsthat create a sense of anticipation and excitement. The color scheme is predominantly dark, with shades of gray and black, which contrasts effectively with the bright and playful colors used in the text and graphic elements. At the top of the image, the date and location are prominently displayed: "MARCH 15 | CHICAGO, IL.° This information is set in a clean, sans-serif font that is easy to read, ensuring that the essential details are immediately noticeable. Below this, the main title “COMEDY NIGHT° is rendered in bold, yellow letters with a playful, cartoonish style. The text is outlined inpurple,addingdepthand afun,energetic vibe to the overall design. Beneath the title, there is a tagline that reads “EPIC STANDUP SHOW COMING," which is presented in a smaller, white font. This text is positioned to maintain a balanced composition, drawing the viewer's eye from the title down to the details of the event.Thelower section of the image features three circular portraits of the performers, each enclosed in a purple background that matches the outline of the title.The names of the comedians—Jake Davis, Noah Smith, and Leo Jones—are displayed beneath their respective images in a clean, white font, ensuring clarity and visibility. The use of circular frames for the portraits adds a moderm touch to the design.Theoverallartisticstyle of theimageis contemporaryand engaging, utilizing a mix of bold typography and vibrant colors to convey a sense of fun and entertainment.The mood is light-hearted and inviting,perfectly suited for acomedy show, suggesting an evening filled with laughter and enjoyment. This image could be effectively used for social media promotions,event flyers,or digital advertisements,appealing to audiences looking for anight of entertainment and humor. The combination of visual elements and text creates a cohesive and attractive promotional piece that captures the essence of a lively comedy night.

![](images/455082a2d9cdd8f537f08b535fb764a44ff585a87ca07edcbfe28e22dd5da800.jpg)

<details>
<summary>text_image</summary>

#1
#11
#10
#3
#2
#5
#7
#9
#4
#6
#8
</details>

![](images/8f09815ef43ee4a4f6d6ff6f8b6199a5871a6a3744c69cdeeed8545799613118.jpg)

<details>
<summary>text_image</summary>

MARCH 15 | CHICAGO, IL
COMEDY NIGHT
EPIC STANDUP SHOW COMING
Jake Davis
Noah Smith
Leo Jones
</details>

![](images/eba7e7bb96a88fd57b5a54461f102fee9a33098892c8c0d41db484bac36a56ad.jpg)

<details>
<summary>natural_image</summary>

Grid of transparent background with small purple rectangular shapes and a dark screen (no text or symbols)
</details>

The image presents a vibrant and engaging promotional graphic for a movie night event. The background is a deep gray providing a modern and sleek canvas that enhances the visibility of the foreground elements. This color choice creates a contrast that draws attention to the text and illustrations.At the top of the image, bold white text reads,“PLEASE JOIN US FOR," which is positioned above the main event title.“MOvIE NIGHT."rendered in large,eve-catching letters,The font is plavful yet clear, suggesting a fun and inviting atmosphere. The use of white against the dark background ensures that the text stands out prominently, The central composition featuresthree stylizedfiguresseatedin plush red cinemachairs,each holding popcorn.The characters are illustrated in a minimalist style, with simplified features and muted colors that convey a casual and relaxed vibe. The woman on the left is depicted withlong,flowing hair and a yellow top, while the manin the middle wears a purple shirt, and the figure on the right is dressed in a light blue shirt.This arrangement creates a sense of camaraderie and enjoyment, typical of a movie night experience.At the bottom of the image, additional details are provided in a clean, sans-serif font. The text includes the date, “SATURDAY, JULY 23|8 PM," and the location. “AT MOvIE CINEMA.all of which are clearly legible and wellorganized.The inclusion of a website link, "www.COMPANYNAME.com," suggests a professional touch, encouraging attendees to seek more information. The overall artistic style is digital and contemporary, characterized by flat colors and simple shapes, which contribute to a friendly and approachable aesthetic. The use of popcorn graphics in the bottom left comer adds a playful element, reinforcing the theme of a movie night. The mood conveyed by the image is festive and inviting, making it suitable for social gatherings and community events.The combination of engaging visuals and clear information makes this image ideal for use in social media promotions,flyers,or event announcements,effectively capturing the excitement ofa movie night.

![](images/4c314997d9e4b7850ffee5c359713a7559c005b42bccfd174601859f935d36d4.jpg)

<details>
<summary>text_image</summary>

#11
#5
#4
#3
#2
#8
#6
#7
#9
#10
</details>

![](images/afad405980294a75a9c28373a1c04652368e99fd6f10752e0cfba9734516b3ee.jpg)

<details>
<summary>text_image</summary>

PLEASE JOIN US FOR,
MOVIE
NIGHT

SATURDAY, JULY 23 | 8 PM
AT MOVIE CINEMA
www.COMPARTNAME.com
</details>

![](images/346b1e046388cf2a53e74fa24d7fd2544cf9f45e1f12f64d23416d430dc0ed0e.jpg)

<details>
<summary>natural_image</summary>

Grid of transparent background with black and red pixelated patterns, no text or symbols present
</details>

The image presents a stylish promotional graphic with a modern and elegant aesthetic.The background is a deep, rich green that provides a sophisticated canvas, enhancing the overall visual appeal. This dark backdrop contrasts beautifully with the lighter elements in the foreground, creating a striking visual hierarchy. At the center of the composition is a figure dressed in a fashionable outfit, which consists of a long, flowing dress paired with a stylish jacket and a wide belt that cinches at the waist. The clothing features a muted color palette, primarily in shades of green and gray, which harmonizes with the background while also allowing the figure to stand out. The outfit’s textures and layers suggest a contemporary fashion style, appealing to a modern audience. Surrounding the figure are decorative elements, including a circular shape that frames the model, along with star-like motifs that add a touch of whimsy and sparkle. These elements are rendered in white, which contrasts sharply against the dark background, drawing the viewer’s eye toward the central figure and enhancing the overall composition. Prominently displayed at the top of the image is the text “NEW ARRIVAL" in a bold, uppercase font, conveying a sense of urgency and excitement.Below this,the phrase “FLASH SALE"is presented in a slightly larger font, emphasizing the promotional aspect of the image. The text is rendered in a warm gold color, which adds a luxurious feel and complements the overall color scheme. At the bottom of the image,a banner announces a "50% DISCOUNT,"using a clean,sans-serif font that is easy to read.This call to action is crucial for engaging potential customers and encourages immediate interest in the sale. The website URL,“www.fashionstyle.com," is also included,providing a direct link for viewers to explore further. The overall artistic style of the image is modem and polished,witha focus on fashion and retail.The combination of elegant typography, a cohesive color palete, and stylish imagery createsamood that isboth invitingand aspirational.Thisimage is wellsuitedforuse indigitalmarketing campaigns, social media promotions, or as part of an online store's advertising materials, effectively capturing the essence of contemporary fashion retail.

![](images/1d870476a7cf48e201860c81c3557ab828c02d2831a15bd90debc0d92cd6236d.jpg)

<details>
<summary>text_image</summary>

#1
#8
#9
#3
#2
#4
#5
#7
#6
#10
#12
#11
</details>

![](images/939803c4b855b7c1d302648cf19fe5886a7e4e44a8850c56c05e58d2b843ace2.jpg)

<details>
<summary>text_image</summary>

NEW ARRIVAL
FLASH SALE
50% DISCOUNT
www.fashionstyle.com
</details>

![](images/4380494968fe6739fd9c78eb2623dd34c77d72b65898bda2963f6f506e090793.jpg)

<details>
<summary>natural_image</summary>

Grid of transparent background with a small dark object in the center (no text or symbols)
</details>

Figure 6. Additional text-to-layers generation examples.

# Prompt

The image presents a vibrant and playful composition centered around a delicious pastry, likely a cinnamon roll or a similar baked good.The background features a dynamic pattern of swirling stripes in shades of lavender and white, creating a sense ofmovement and energy that drawsthe viewer'seye.The alternating colorsof the background evoke a lighthearted and whimsical atmosphere, enhancing the overall appeal of the image.At the forefront, the pastry is prominently displayed showcasing its golden-brown, flaky layers that spiral inward. The texture of the pastry is rich and inviting, with a glossy sheen that suggests a fresh-baked quality The intricate details of the roll,including the visible swirls andlayer, highlight thecraftsmanship involved in its creation,making it the focal point of the image. Surrounding the pastry are bold, black letters that spell out “vPnOsS.\* with the letters varving in size and orientation adding to the playful and modern aesthetic. The typography is striking,with a mix of uppercase and lowercaseletters that create a sense of spontaneity. Below the pastry, a rounded black banner contains the text “SHELLYP TIssERIE."rendered inaclean,sans-serif font that complements the overal design.This text serves to identify the brand or establishment associated with the pastry, reinforcing its artisanal quality. The artistic style of the image leans towardsa contemporary graphic design approach, characterized by its use of bold colors,playful typography,and afocus on food aesthetics.The combination of the vibrant background and the detailed pastry creates a visually appealing contrast that is both eye-catching and appetizing. The theme of the image is celebratory and indulgent,evoking feelingsofjoyand satisfactionassociated withenjoyinga sweet treat. The mood conveyed is cheerful and inviting, making it an excellent choice for marketing materials related to a bakery or pastry shop.In terms of symbolism, the pastry represents comfort and indulgence, while the playful typography and background suggest a modorn, trendy approach to food. This image could be effectively used in variouscontexts,suchassocialmediapromotions,advertisements forabakery,or as partofamenudesign,appealingtoayouthfulandvibrantaudiencelooking for delightful culinary experiences.

# Layout

![](images/212098b0831e9bdd2e2e7a71ba8ae106475ab8ab22e85ad1f2e8f1ff6624f91a.jpg)

<details>
<summary>text_image</summary>

#1
#4
#5
#6
#7
#3
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#10
#12
#11
#8
</details>

# Generated

![](images/56dfe4aa00a20d132ea917051444e4a35b8932ef6ebc1dab2dfa7aa981c1f7cf.jpg)

<details>
<summary>text_image</summary>

SPLNOSO
SGLUP TOGETHE
</details>

# Layer Images

![](images/c7ac63e9dfd01f92932e8ea02831ff8e56387e2a0865847582d9b835022302d3.jpg)

<details>
<summary>text_image</summary>

n
S
y
P
s
</details>

The image presents a vibrant and modern design, primarily featuring a bright blue background that conveys a sense of professionalism and creativity. The blue is complemented byacontrastingyellowrectangle at the top,whichcontains the text "We are"in a bold, sans-serif font. This combination of colors creates a visually appealing and energetic atmosphere.At the center of the composition is a figure dressedinacasual yetstylishoutfit,consistingofalightpinkoversizedsweater and yellow shorts. The individual is holding a laptop, which is prominently displayed in Iront of them, suggesting a focus on digital technology and marketing services. Thelaptop screen is blank, whichmay symbolize the potential for creativity and innovation in digital marketing. The text “DIGITAL MARKETING AGENCY"is positioned prominently in white,bold letters, making it the focal point of the image.Below this,asmaller tagline reads,“We are ready to help yourbusinessmarketing," which reinforces the agency's commitment to supporting clients.The font used here is clean and modern, enhancing the professional tone of the image.In the lower section, there isayellowbox with the call-to-action°CALL US“in white text, accompanied by a phone icon and a contact number. This element is designed to encourage immediate engagement from potential clients, making it clear how to reach the agency. The overall artistic style of the image is contemporary and graphic, utilizing flat design techniques that emphasize clarity and directness. The bright color scheme and modemn typography contribute to a mood that is both inviting and professional, ideal for a digital marketing agency aiming to attract new clients. This image could be effectively used in various contexts,such as social media advertisements,website banners,or promotional materials,to convey the agency’s services and encourage potential clients to reach out for assistance.The combination ofengaging visuals and clear messaging makes it a strong representation of a modern marketing agency.

![](images/ca5ec51390006d317717149d2cebf70083a93daa07eeb8ecb57e8019d5c8b6e8.jpg)

<details>
<summary>text_image</summary>

#1
# =7
#5
#6
#10
#11
#9
#8
#2
# =13
</details>

![](images/4d6e031271b4f7293cba0d258c5f98766fd5a56b4e5cd0d294a90f578a3231b6.jpg)

<details>
<summary>text_image</summary>

We are
DIGITAL
MARKETING
AGENCY
We are ready to help
your business
marketing
CALL US
123 456 7990
</details>

![](images/2bafcf4585d6788a70f3bfabf866e7f90e0d4c116c94e9fbdb0a75c81c315674.jpg)

<details>
<summary>text_image</summary>

Grid-based diagram with blue shipping containers, yellow blocks, and a person reading, likely for a game or interactive interface.
</details>

The image presents a modem and stylish promotional design for a clothing item, specifically a hoodie.The background features a soft,creamy beige color that creates a warm and inviting atmosphere,This neutral backdrop is complemented by abstract shapes in darker green tones,adding a touch of sophistication and visual interest without overwhelming the main elements.At the center of the composition is a figure seated on a wooden stool dressed in a trendy outfit that includes a black hoodie with striking green accents.The hoodie is layered over a light-colored top, and the figure is wearing light-colored pants that contribute to a casual yet fashionable look.The choice of footwear, which consists of rugged boots,enhances theoverallstreetwearaesthetic.The figure’sposeisrelaxed,suggesting comfort and confidencein the attire.Text elementsare strategically placed around the figure.The phrase “New Arrivals” is prominently displayed at the top in a playful handwritten font, while the word Hoodie is featured in a bold,modern typeface that commands attention.Below themain title,asmallerline reads.Clotes are the spirit of fashion,“ whichappears to be a typographical error for “Clothes." This text is presented in a more understated font, allowing the main message to stand out. Additionally，there isa call-to-action button that says“Order Now,"designed in a contrasting dark green color that invites immediate engagement.The overall artistic style of the image is contemporary and clean, with a focus on minimalism that highlights the product. The use of soft shadows and subtle textures adds depth to the design,making it visually appealing.Tbe theme of the image revolves around fashion andlifestyle, conveying a mood that is both trendy and approachable.It suggests a sense of modernityand casual elegance,appealing to a youthful audience interested instylish clothing.Intermsofsymbolism,thehoodierepresentscomfort and versatility in fashion, while the overall design promotes a sense of accessibility to new trends.This image could be effectively used in various contexts,such as social mediaadvertisements,online clothing stores, or promotionalmaterials for fashion brands, aiming to attract customers loolking for thelatest in casual wear.

![](images/881fcd2b45152451499bd849534a21004dd0d9db9680f6a96fd6506791a12e34.jpg)

<details>
<summary>text_image</summary>

#1
#15
#3
#4
#5
#6
#7
#14
#8
#9
#2
#10
#13
#11
</details>

![](images/22ee109dec33b2decdd26846a9dfa725232939978c7aea585b731c897e75c84f.jpg)

<details>
<summary>text_image</summary>

New Arrivals
Hoodie
Clothes the spirit of fashion.
Order Now
</details>

![](images/563386356831c6ad83482334b796807181f1cbb09fb7b121c3fda62a52dbbec7.jpg)

<details>
<summary>natural_image</summary>

Grid of transparent background with small geometric shapes and no visible text or symbols
</details>

The image presents a vibrant and engaging promotional graphic centered around fresh fruit farming.The background features a clean, white grid pattern that addsa moderm and organized feel to the composition, enhancing the overall visual appeal. In the foreground, a wooden crate brimming with fresh apples takes center stage. The apples areamix of redandyellowhues,showcasing theirnatural freshnessand inviting appearance.Some applesarepartiallyobscuredby greenleaves,which addatouchof nature and vitality to the scene. The crate itself is positioned slightly to the right, creating adynamic balance with the text elements on the left.The text\*FRESH FRUITS FARMING" is prominently displayed in bold, green typography at the top of the image. The font is moderm and sans-serif, contributing to a clean and contemporary aesthetic.Below this,a striking“45% OFF"is highlighted inalarge, whitefont against asolid greenbackground, drawingimmediate attentiontothepromotional offer. This textis further emphasized by a playful, starburst-shaped "SALE!"graphic in yellow, which adds a sense of urgency and excitement. Additional text elements includea call-to-action“SHOPNOW”inabold, whitefont onadark greenrectangular button, accompanied by an arrow pointing to the right, guiding the viower's eye towards the next step,Below this, the address “123 Any St, Any City"is presented in asmaller,more understated font,ensuring that the focus remains onthe promotional message. The overall artistic style of the image is modern and graphic, utilizing a combination of bold colors and playful shapes to create an eye-catching advertisement. The use of green tones not only aligns with the theme of freshness and nature but also evolkes feelings of health and vitality. The theme of the image is promotional and inviting, aimed at attracting customers to purchase fresh fruits. The mood conveyed isenergeticand positive,encouraging viewers totakeadvantage of the sale.This image could be effectively used in various marketing contexts, such as social media advertisements, flyers, or website banners,particularly for businesses related to agriculture, grocery stores,or health-focused brands.The combination of appealing visuals and clear messaging makes it a versatile tool for engaging potential customers.

![](images/e8e9a58577c763cf80a44e30706c81fb7230e4740782b600364725de89928672.jpg)

<details>
<summary>text_image</summary>

#2
#3
#8
#15#21#20
#12
#4
#5
#6
#7
#9
#10
#11
#13
#14
#15
#16
#17
#18
</details>

![](images/417656acf4dff1663c40a5b3c18e79c3cecc47a1a7769e72979bcfa60feb07bb.jpg)

<details>
<summary>text_image</summary>

FRESH
FRUITS
FARMING
45%
OFF
SALE!
SHOP NOW —— 310000 Any St, Any City
</details>

![](images/c64da868b8359d2e2f964e64a17c82a863e7deebfd478385e4f20216b0a89225.jpg)

<details>
<summary>text_image</summary>

FRESH FAIR
FAIRING
</details>

Figure 7. Additional text-to-layers generation examples.

# Prompt

The image presents a vibrant and engaging promotional design for a winter bazaar themed around vinyl records. The background is a deep black, which serves to enhance the brightness of the other elements and create a striking contrast. Scattered throughout the background are colorful dots in various sizes, including white, orange, and purple, adding a playful and festive touch to the overall composition. At the center of the image, two stylized hands are depicted in a bold, graphic style, gripping a large pink vinyl record. The hands are illustrated in a deep blue hue, which contrasts effectively with the pink of the record. Surrounding the record are various musical elements, including colorful sound bars and musical notes, which suggest a lively atmosphere and the joy of music. The sound bars are rendered in shades of orange, yellow, and purple, creating a dynamic visual rhythm that complements the theme of the bazaar. Above the central illustration, the event title “Vinyl Lover Winter Bazaar” is prominently displayed in an elegant, cursive font that is white and slightly whimsical, evoking a sense of creativity and fun. Below this title, a tagline reads, “START THE COOLEST SEASON WITH A RECORD," in a simpler sans-serif font, reinforcing the event’s theme and inviting participation. The lower section of the image contains essential event details, including the date \*DEC 10, 20XX,\* the time “4:00 PM," and the location “15TH AVENUE, LONDON." This text is presented in a clean, modern font that is easy to read against the black background, ensuring that the information is accessible. The overallartistic style of the image is modern and playful, with a digital ilustration technique that emphasizes bold colors and graphic shapes. The use of contrasting colors and whimsical elements creates an inviting and energetic mood, perfect for attracting music lovers and potential attendees, In terms of symbolism, the image captures the essence of community and celebration associated with music and vinyI culture, It suggests a gathering of like-minded individuals who share a passion for records, making it an ideal visual for social media promotions, event flyers, or posters aimed at music enthusiasts. The lively design and engaging elements make it suitable for various marketing contexts, particularly in the realms of music, art, and community events.

# Layout

![](images/037539a44c62c9f307f8c476e37be20f1bbfbe8efa5adaeac2dfdfc600493c55.jpg)

# Generated

![](images/c48bdcf9be55430781bad8f0165844b7448ce3487494b55713491adf97763e70.jpg)

<details>
<summary>text_image</summary>

Vinyl Lover
Winter Bazaar
START THE COLOST SEASON WITH A RECORD
DEC 10, 201X X 22XX
X-4 XXX
4:00 PM
15TH AVENUE, LONDON
</details>

# Layer Images

![](images/194137e2e24a67819cea403868dd836a9ec2a85e00c192f0058231bf5bd0e2d9.jpg)

<details>
<summary>natural_image</summary>

Grid background with transparent squares and scattered colored dots, no text or symbols present
</details>

The image presents a promotional advertisement for a tanning salon, characterized by a clean and modern desigm, The background is a soft beige color, providing a warm and inviting atmosphere that complements the theme of sun-kissed skin. At the center of the composition is the word \*Tanning," rendered in a bold, elegant serif font that commands attention. The letters are predominantly black, creating a striking contrast against the lighter background. Below this, the word “Salon” is displayed in a more delicate, cursive font, adding a touch of sophistication to the overall desigm. Flanking the central text are two geometric shapes in a deep blue hue, which serve as visual anchors. These shapes are adorned with circular motifs that resemble stars, enhancing the theme of beauty and radiance associated with tanning, The use of these shapes adds a moderm flair to the advertisement, while the star motifs suggest a sense of glamour. Prominently featured in the corners of the image are bright yellow starburst shapes that read “50% OFF," effectively drawing the viewer’s eye and emphasizing the promotional aspect of the advertisement. This use of color not only hightights the discounts but also adds a playful element to the design. The overall artistic style is contemporary and graphic, utilizing clean lines and a balanced composition to convey a professional yet approachable image. The mood of the advertisement is upbeat and inviting, aiming to attract potential customers looking to enhance their appearance through tanning services, In terms of symbolism, the elements of the design—such as the stars and the warm color palette—evoke feelings of summer, beauty, and self-care. This image could be effectively used in various marketing contexts, such as social media promotions, flyers, or website banners, to attract clientele to the tanning salon and promote its services.

![](images/8e3f779ef34bb0dfafcd880b4be798b9d52ed58f3881c749294c7c73c9b44d35.jpg)

<details>
<summary>text_image</summary>

#1
#21 #12
#2 #13
#15
#16
#3 #19
#18
#4
#5 #6 #7
#8 #9
#10
#11
#24#5
#24#27
</details>

![](images/b684b0b0cb67e901e2a30a44f35d84d1c659fbd92ccdae9b75874651b37cb398.jpg)

<details>
<summary>text_image</summary>

Upcoming Sale
50%
OFF
50%
OFF
Tanning
Salon
</details>

![](images/3fc8603a72b6fada84c4dff8b7254863ed7c3173372c9cb413ef3d5d34a3a305.jpg)

<details>
<summary>text_image</summary>

Tanning
</details>

Figure 8. Additional text-to-layers generation examples. Our unified framework handles various design complexities, from simple compositions to intricate multi-element designs with over 25 layers, while maintaining generation quality.

![](images/4083df59b8063f406fecb99310aeebea0ffdebe8ba8863809d55be9b75cf529a.jpg)  
A fat. plawfsl Jlstration fesinated 3s yarn relae a sdcmasackgrodsvithbrigt tqeieadrage ccets alapetotbsshetteredca)o stliso,d,e wesiasd ge headineGRCAS”,aaler age ni d-tise ESCIsEME stTIENES DUDAs°, and atuquise con tactiine"HOLAgUNSoGENtALES,，elautotsecon e geometric sapes−a spiralpuschedractaea, rond 1 batingaedbeideis,gnasic,o vector feeL Corspeeition cesters the sotegaper slight y right of enter with icoes lagered arousd the corers :applenegatise spaoe franses te beattinead coeates dearyporapic iecarcly (vryargeeadlerd m call-to-action, smaller cestact), Typograpby appears basd-drss,bedoodeaaps yetlotrvithsbeoiwaesapes,ceatighe ale, thankgeu theceed desigs.   
AdngetWeter capagpotersettateeaysysraes tesSeedCHECTEFOECSTBE ORE YOU Got° anethe sbline\*Staysafe by pmpari tb Opoblethagssobigl elage dometentatlefteeteatacsterofabee atup at rightvthoefodingchair adeeore holding a lettle and cugs, and a srsall late shape tower d the herizon.Shapes are fat, cartoen/vecterwith bol d otins, sisple geotticcesSorthebant archa sd oesed lske, and repeat motis of triangular tree s Dhesettes; textsres ape smeeth flat celors with sabtle horizostal gradiest bands in the sky: Cuegosition pic m the beadline in large, proeninest uppercae chunky di sptaytypeacossett,bodr igh cestrast, whie the senteaoe-case triendly rounded saasaubbead is ooptsr-ciott bepeath z.cmating a cl eer typograghic hierarchy and 3eft-to-rigbt reeding flo w Eleceests are proportiened sethe tents large aod f gresrosod-ielt,sbepersesseatedjsrmsid-ssaedcester:r igtt, dtecgesd tresaleeedespig Jeavesgeoecossegativesikyspaeaoreandlaces belestsitetOe storratiseanttdoeyesdyasdcastiter thanalamist,singpysrodedtypogrtfrthe subbead aod abeid candensed Oecarativedisplayforth e beadlinetocoeseyanoutdoersadetytbetas   
wratedteracettabeckgodandargedegtgr eeochalodfrmedinafwie;secanaryacetsi aclsde pale sage groees, cean, and msted marsen Tae c onposities.cotaimsenelarge rectasgtlarctaldortce ateredgttagigptestdtope ooegeoetricngngpethtalpedpat (right), atamed picture (top-right), x table lamp (b ottom-right), and two bools on a cabinet (bottsm-lett) -several owpeated leaf shages and tvo hangisg pianter motih. Geotatey isarglyactanguar asdaogslar vtt 3 crisp vector outtises and thin dark strokes; the chel Khoand |s tbe3ergest element, occepsingrgeghty halft aecasaaddoistingcacleecateoects ucharuce ecet wth e tectare. Elecaents are lapsred with the chalkbeandi rec msedbebindscreafeaddecotiveobetsign ed alogtbeeasdigmais,atadbyee usnegativespearoadtbecetraltest.Tyeapyi sbeid, ssusded sare-serif, large asd ontered on the c Satbordicapreigtbeibteetigc int. The background is a usifonm warrs terracotta, produ cingscal,veoisigtlyagemodane   
Itbitbecgedwittacetadsaace loful sticloetisinpik,elleadbacke utlines;eroedveeccadstidb ysiteseretea csmserifbrceeedostheftdtl e the rightcard ooxtainssixsmal sed-and-greessemic xblackstyedeassignotstventecar ds asdalacgecveddecstietiteS isperca se back ares above, alletersexts erisp, fat, and gles acing asd centered heriontal aligsment creating a play Sutedacatioealmoodaodstrengtpogrpticearthya gxinst.thesegativetioa.feld.

Figure 9. Text-to-layers with overflow layer generation. Additional examples highlighting our method’s unique capability to generate overflow layers that extend beyond the background boundary. As discussed in Section 3 and shown in Fig. 3, over 60% of designs contain overflow layers. Our approach generates complete full-size RGBA layers on the canvas, preserving editability and reusability that previous methods sacrifice by truncating pixels at background boundaries.   
![](images/a123cdcdaab4eea5f5fcf9503f2c36260776ce847585e0d8c1b514f54ce6de63.jpg)  
画面采用亮橙色为主色调，搭配明亮黄色和天蓝作为副色与点编，整体为高饱和皮的平面画风格，画面中心曲两个重的简形牌块构成：上方大黄色长方牌属中且偏上，星主视定位，内含黑色租体无衬线文字“第一手吃瓜基地；下方白色长方牌暗向后下叠，中心有蓝色招状长方块，内含白色租体无衬线文字还不旺紧加关注。画面左右及顶部有放射状射线背票复多条射线增强动感，左上角有两只大形卡通与小闪电形装饰，右下角有美色闪电图形与信封图作为点，尺寸与比例上，黄色主牌占面面上平部的大觉比例为主，蓝色按钮的为主牌宽度的四分之三并作为次级信息突出，图标与装饰为小比例点。质感平失量，边明显，使用纯色填充无或田影质感元之间有明监层次与重，上牌通过果色指边与经影通区分前所关系，对齐以中安为主，留自适中使交字易读，字体为粗体无村线风格，文字水平中对齐，级清晰：大标题基、之。整体构图为横向报式对称中置布局，围活泼、随日、具有社交提体呼呼。  
面面以米白为主色调，辅以藏蓝和须船作为副绝与点，中心为一个白色影普深蓝色置转边框四角花，占把画面宽度的70%，上方有一个四花中大，中水 形的助射色章，内置白字“些”，辨面中央以深蓝手平中，面面两和底装饰有3-4处水星风荷花与荷（蓝、灰蓝与粉色），为重复元，形成左右呼应与视度平。整体采用平面无量结合水星于质感，边缩与文字为庆里镜利，花卉为柔和水彩交且有感，元素金现居中对齐的主次层次：大牌为主、章与文本为次，花卉为点并部分与牌通外缘不重，留白充。字体为无线，书法感的体中文手写字，文字居中对齐，版式湿规时的单列网格。背为浅白净色，品部有水云纹装饰，整体构图重主题偏课程/总场类，沉静且路带古典意境  
这是一张以深蓝择为主色调、金置与浅绿为点的中秋节活动海，画面中央为大号白色服体无线题中秋绝月催大费”，占犯宽度的主要视完比别；上方偷中位置以胶小的金黄色与白色文字组合出现“快快出你的”“n“赏月照”，形成图板对比；标题下方中有市圆角边的金色式元素，内含白色文字点告参与活动，为深色的，景有两和下方多处抽象面山丘与植物前纸风图形（重复元素约4-6处小叶子与3个主要丘陵），右上角为大的金策色满目排带象子笛影，整洁采用午量+纹理叠加的平面播画风格，元之问通过上下因中对齐与道留白构建层次，文字多为无线程体与的装体湿排以突出主灰，面结构呈三栏网格中心对齐，图节日强温。  
面面以附色为主色调，辅以浅和果白为副色，点缩色为解粉和红色，整体为向海按布，重复出观的文字约有7处，图形元素包括1个信封播画，2个小爱心1个手写式“lve字样和一个累色外的白色文本气；主要标题文宇省西西中上部的大比到位图，次 绿提示文宇位于下方和底部，比制上主称题约占画面高度的40%-50%，次要元账和装饰合计的点30%。风格为平面失量，文字为组医无大字，制附与粉色下划高光，感、光源、无影质。空间上主标中排列，装饰信封位于左上角盖达，有角对话框状模幅与背景供盘格渐交重，元素留白充足，次通过黑色边和投影分明，版式采用中对齐的重直网格，标题为大号程体无线，文字为中等相，整体围波湿可重且活，主题为爱小技巧的生活爽海报。引用可见文字有：建议收小技 模升温小技巧、高又浪湿、“维又心动”、 “双方限问上头”

Figure 10. Text-to-layers with multilingual support. Examples demonstrating our model’s capability to generate designs with multilingual text layers. Our dataset includes diverse languages (as shown in Fig. 2), enabling generation of visually rendered text in multiple languages including Chinese. This showcases the model’s ability to handle typography across different writing systems while maintaining design quality.

![](images/1046a496d1a88e6c5d6de9e61405f586b93d23355ee1e25eaef43f648fcfbfc8.jpg)

<details>
<summary>text_image</summary>

Bundle
of joy
Input
</details>

MRT   
![](images/68c98b3177d96b9e68f42792d4967b10ea5122194e6aedd7347a9b896625d997.jpg)

<details>
<summary>text_image</summary>

Bundle
of joy
</details>

![](images/c034bb595a372c209dfba09d60f244b864185e7a07161265edad2ff23ee82760.jpg)

![](images/e42a20509d05bcfd4a445738f0949665ca5395a3975d239a17283542d2125437.jpg)

![](images/974f664c78d97ac556d625a97ecd83504712fd5feb93e61c5bdb8812ebf6f611.jpg)

![](images/6716a5dde9598914e4432384ac57e167d896278789e4cd7c7549018f1c190fce.jpg)

<details>
<summary>natural_image</summary>

Woman holding a baby in a colorful, abstract background with cloud shapes (no text or symbols)
</details>

![](images/70d741e5f117e78ba8809cf2ba52266bb967fdb3db7bd1290edc59e104cdaf7f.jpg)

![](images/679fa99634095e843a1babc4a4c7a37b2aa650db78f53d8b1f43e53117525e4e.jpg)

![](images/9aff6bdca86cb381a909e00c9023182af09992d2843537a887a5bbed8cccc548.jpg)

<details>
<summary>text_image</summary>

Bundle
of joy
</details>

![](images/3628c2acaa8dc68abbade9b18b976bcfbadb50711f2945b847e4b82e4cfcd280.jpg)

![](images/f030879d7d842acb046c89477b7ac47916619f491447c6bf4474cae46ad8b4ca.jpg)

![](images/03369c23ce7ff75730a9411098374af9ba2d0b73428bedf3db7135a2e7173a3f.jpg)

<details>
<summary>natural_image</summary>

Woman holding a baby with a thought bubble saying 'Bundle of joy' (no other text or symbols)
</details>

![](images/12d6bf7989307022151f22344e150a9b6688a643833c9f4858af608fc83e09a1.jpg)

![](images/49e61f198ae6e6b566d722b1230fe1702d83b064bd49ed941b7d729bbff4af5f.jpg)

![](images/55133162c2ee0c63ca42df2065f0f1f952921bedbb69c64e54fdd355b520a487.jpg)

![](images/4527890c4ff742d98f55bdbfd11f79b23ef7d8602786ce2a66891f2c2dcee935.jpg)

![](images/f4f898b027001bc2812140fa15465f09c7af04258dfa1555d38aca6eabe8f0cd.jpg)

![](images/ebfbf5727620f6f75df86f87be6a751ec84c7618007598454bcac503279025b1.jpg)  
Figure 11. Qualitative comparison on image-to-layers task. We compare our method with LayerD, Lovart, and RoboNeo on decomposing a graphic design into transparent layers. Each panel shows: the input image (top-left), followed by our result and baseline results with their decomposed layers. Our method produces cleaner layer boundaries, better granularity, and more complete RGBA layers compared to the baselines.

![](images/83b99015b15f7c2d8274f24e61fdf6ab9e685303dc7b0870638384238f369bce.jpg)

<details>
<summary>text_image</summary>

MRT
Love
NEVER FELT
So Good.
Iovart
Love
NEVER FELT
So Good.
RoboNeo
Love
NEVER FELT
So Good.
LayerD
Love
NEVER FELT
So Good.
Input
Love
NEVER FELT
So Good.
Love
NEVER FELT
So Good.
Love
NEVER FELT
So Good.
</details>

Figure 12. Additional qualitative comparison on image-to-layers task. Our method demonstrates superior layer decomposition quality with better semantic correctness and transparency handling. The decomposed layers from our method maintain higher integrity and can faithfully reconstruct the input image, while baselines show issues with layer artifacts, improper grouping, or incomplete decomposition.

qndu   
![](images/c0694c40a0326b8c0ba11de50f6c2d85c23bd4e24bae40e1f187506bcea7266a.jpg)

<details>
<summary>text_image</summary>

Room For All
Best home interiors
</details>

MRT   
![](images/ca224d63affd62e0fe10f36e6d1b6bcd2a3b5ff1282f904b25971a1e7277a3b1.jpg)

<details>
<summary>text_image</summary>

Room For All
Best home interiors
</details>

![](images/dffc1ecf28b0eb25d692682cf8136b1e711f9c8316105ee2da1f9a772fea5113.jpg)

![](images/434b653659e870959519e1c1157e49d260b0eb41fe495e7c50093efbe754e087.jpg)

![](images/23c8ceab644c0ee7a0ce4468cd4afcdc47958d353e4c7d53d5712976e70e4ac5.jpg)

![](images/5c4b0dbf45552af16995a020b54a668054237cfc8ce607cd856015b6899de7cf.jpg)

<details>
<summary>text_image</summary>

Room For All
Best home interiors
</details>

![](images/29d5252a59f73c85b742102213adb7363cd59a26b4839a620b7af93b8d2a4bd0.jpg)

![](images/cb61a2bfe25e56d87e7ebaeb8d0ec11e8a01b2ba394b30b313be9d667b69d277.jpg)

![](images/ad1a2defa79c002c3e809de00f6fd05039dc65e33f8270144e99b3171651d51d.jpg)

<details>
<summary>text_image</summary>

Room For All
Best home Interiors
</details>

![](images/6a7658f3f0bc883e781c91599be591e1468adadaf49df05c106ffd96c1a6f4c5.jpg)

![](images/88de89463b44d5512c8d65a952eda3f6370a033a552fe6b3e88ef134c1bbe597.jpg)

![](images/726a39a50e5c5da3cc27f8bc9f293b3e43beb76934fd75b45273dcadb5594030.jpg)

![](images/a30561e773f5fbb8f742634017311e257ad8e94939ec0c1aca89559cad7c8506.jpg)

![](images/ad1a2defa79c002c3e809de00f6fd05039dc65e33f8270144e99b3171651d51d.jpg)

<details>
<summary>text_image</summary>

Room For All
Best home Interiors
</details>

![](images/85a6dd4efa94a5ce84e8ca67c54eaff3961432aaaec3b89b95ebf87cd4402e24.jpg)

![](images/05a19e1f89db4c0c07ad1839e6182f87d9159d5dbecb3c503c4cedb8b5b7b953.jpg)

![](images/a610a300bf58c6bbdb3b05979909dfc09a1ee48fd14b9b5efe0b2976c27d0e1f.jpg)

![](images/61c236b4bce260dcc78951dbeee7d9a2fa642286e9087dfd8e0120baa5346ef1.jpg)

![](images/dd86fc866c186adf2b5c72f56d1898ac1d7f062f33ac1b3bbd8aeba1c7de3d33.jpg)

<details>
<summary>text_image</summary>

Room For All
Best home interiors
</details>

![](images/46ddfd711c1654475b62fb337e9d87cdee494e6225bbed6a2f8c49f16e71ca7c.jpg)

![](images/9faaeca9aee0b46559b9152caaf16c7bd7e6485a4bba907c5587473de6a061d2.jpg)  
Figure 13. Additional qualitative comparison on image-to-layers task. This example further demonstrates our method’s advantages in layer quality, integrity, and appropriate granularity. Our approach successfully decomposes complex compositions while avoiding the overly grouped layers produced by LayerD or the artifacts present in commercial system outputs.

ndu   
![](images/9f777d1ea6cd3d0edc4efe94ffbd8e446e05494c65f97eb6b829191fa2aa44a1.jpg)

<details>
<summary>text_image</summary>

Ugly
SWEATER
DAY
</details>

![](images/35508b3f87a1f92553524d33a61389138d3a69a9b7ddd1df8bbd204725d47b06.jpg)

<details>
<summary>text_image</summary>

Ugly
SWEATER
DAY
</details>

![](images/e78f105d522554af7310178104b75f30d52f733df74f139b01a133f369243112.jpg)

![](images/eb91f4e3151f8a01d5c438c3273cd4607859a03575387ae3ace5f766cca66fd5.jpg)

<details>
<summary>text_image</summary>

SWEATER
</details>

![](images/20a995538ff359cdf7596392c12b673e7f84fa8acd8eb6fc3ec292b88a54f8c2.jpg)

<details>
<summary>text_image</summary>

DAY
Ugly
</details>

![](images/8aa67eaf11a7895ff102d768809019145a566aab9b90f4e96d70e929cfb709f8.jpg)

<details>
<summary>natural_image</summary>

Three-panel image showing transparent background with red dots, no text or symbols present
</details>

![](images/7366978950a60ddbf1646c60523a5580e56f339b9ec45937fc6f6486015aed81.jpg)

<details>
<summary>text_image</summary>

Ugly
SWEATER
DAY
</details>

![](images/2ba5622ea5690c39ff4e810026f7f156fae3dd44229f2febd1cea285f2881afc.jpg)

![](images/46ad1057594c40a02a8276445418c8636789514f59011f1d0ea28fe08cfe06c9.jpg)

<details>
<summary>text_image</summary>

Ugly DAY
SWEATTER
</details>

![](images/a2c8eb452bf65dfe04d34b93a4ba2576bbfab97f47e19f1e4319fa94883585dc.jpg)

<details>
<summary>text_image</summary>

Ugly
SWEATER
DAY
</details>

![](images/cce32a6eb54306d5abccb2f23a00d0c964f7bbc8241b62b5b1e00647a3be2f8f.jpg)

![](images/e34404b164873c9c51d2d860ebc5ba6de2ce311b61b4d58d794be22be9a848a4.jpg)

<details>
<summary>natural_image</summary>

Four red diamond shapes arranged in a 2x2 grid on transparent background (no text or symbols)
</details>

![](images/d0370031ab8f7aa0fefd4009d06ef18f286704881b953a52b98db9cae899fc27.jpg)

<details>
<summary>text_image</summary>

Ugly
SWEATER
DAY
</details>

![](images/d2706877d31041b34391e6b93750b10114004d629178aac9c4fff47936e647a0.jpg)

<details>
<summary>natural_image</summary>

Four-panel grid with red star shapes on transparent background, no text or symbols present
</details>

![](images/880583cf6ca16edc06889def2b7c0cd692376e1b2931f4b0db49d43f466632e5.jpg)

<details>
<summary>text_image</summary>

Ugly
SWEATER
DAY
</details>

![](images/1861bf4ed93239e648688598ce4c4e3497ab0fe04f1cf51f3bee30ec5b8eb5fd.jpg)

![](images/9f5cf30602c80091598c980d2da607c137d948636f2f6ddcce47d40759043362.jpg)

<details>
<summary>text_image</summary>

Ugly
DAY
</details>

![](images/aa55e4de98ed957307060434087af10bc93d98fcfe97347fbbf123c7a21d8975.jpg)

<details>
<summary>natural_image</summary>

Completely blank white image with no visible content, text, or symbols.
</details>

![](images/d82af0e20d691035ee5d16e139e3014a7c182d490ca2593775f7ca6a246b2c98.jpg)

<details>
<summary>natural_image</summary>

Completely blank white image with no visible content, text, or symbols.
</details>

Figure 14. Additional qualitative comparison on image-to-layers task. Our method consistently outperforms baselines across different design styles and complexities. The visualization shows that our approach produces high-quality transparent layers with accurate alpha channels and proper semantic decomposition, essential for downstream editing tasks.

ndu   
![](images/1b41fe2fdcb406cf0f129b43c1cedd7c62ff86cc2d15caf701c91cf97a932406.jpg)

<details>
<summary>text_image</summary>

CORPORATE
EVENT PLANNING
Event concept and vendor coordination
for your corporate-event.
15%
OFF
15% OFF
</details>

![](images/e501d8496c43228147a92181ce9f67c6a6170a392189e834ad26e7bb9e1617b7.jpg)

<details>
<summary>text_image</summary>

CORPORATE
EVENT PLANNING
Event conduct and vendor compensation
for your corporate event
15%
OFF
15% OFF
</details>

![](images/4d6a42670b26698493f8ca03f586abd80c48806474da0485d1bb7e7f4afa227e.jpg)

![](images/cc49cb039e71cdc83ab798f6deed8bf13ca6283d99aedfc0dede726fd46c54ef.jpg)

<details>
<summary>natural_image</summary>

Four-panel image showing a brown circle, a gray circle with a crescent shape, and a calendar icon on a transparent background (no text or symbols)
</details>

![](images/7d64d6b3c22cc024ccefc98e01e5633e7fc376d7e27f9a37c622f9cc0089bc52.jpg)

![](images/7c5eec9b61fc95bedb2bf0aaa107ac44078e48ad6ad39f5d3ce9c958ff5bb8d6.jpg)

![](images/d97993f320f68ce42be7231fc291b8be3dba8bcc94de213568eea01e103c37e9.jpg)

1   
![](images/3497137ee485fec84edcffde329c8729adfdc6afbe2e6c9b8f26a15a9e52bcb8.jpg)

<details>
<summary>text_image</summary>

CORPORATE
EVENT PLANNING
Event proposal and vendor confirmation
for your corporate events.
15% OFF
15% OFF
</details>

![](images/3125ba84266477ba24d71b4cf6e7200d2f55590ce5841c260c180fbdb89e91bb.jpg)

![](images/b60e73dcaf56afc2d6591dad42cb1ae1b8413d34798a1a7487f06c760e100048.jpg)

<details>
<summary>text_image</summary>

CORPORATE
EVENT PLANNING
15%
OFF
15% OFF
</details>

![](images/1b1b43b7d1e1dd1fe1c24135839dccb0f40371e1b47e0702ead48ee21cc70138.jpg)

<details>
<summary>text_image</summary>

CORPORATE
EVENT PLANNING
Event concept and vendor coordination
for your corporate event
15% OFF
</details>

![](images/daab65b5cb555b0a7b0bc67fa1356f9ac4502c026768995d6adeec50a8bd1ad8.jpg)

![](images/1580d60f88a8c4fa4cef9e896e2561d0bc852dad970ffc87d1de412590fb8e33.jpg)

![](images/5758165c847c228b44ac2ca23b262a13c18753f043c1b60bfd8458b6941f8aa3.jpg)

![](images/d60be655fc49f35cb735ba0c2c28d400c06f0483bb404c08c8e9b3aaa0d9f086.jpg)

![](images/77b58e200dc12593ec4eebb05340e0327ba91fbd62daf08a551fa8006938549b.jpg)

![](images/24c3cc825db7f1af5cc524bf744f49eff745f8a269db09b907e55048033cde93.jpg)

![](images/36a484807b9fb87df438afef56852c88dd46adfc0779bc8f45f977a80523f55f.jpg)

![](images/7b61afae664472941490235d989afe1c51071e5310baa4a5c0166ecc8daf294b.jpg)

<details>
<summary>text_image</summary>

CORPORATE
EVENT PLANNING
Event concept and vendor coordination
for your corporate event.
15%
OFF
15%
OFF
</details>

![](images/fd3ee90fe8f0254fceae0730557dec61f21e76757521b870b50c96eae2d626c5.jpg)

![](images/3d34e681f2f6415f059280fadd02e0aea1fda552ad886c026db0893197472939.jpg)

![](images/f6d04d7df656ada4b555805b603f9306274b5d34e92ac21decbf09286373d297.jpg)

![](images/7a8bc81e9c6ddeca55b15005447e89a9de45b8ea486a8e9a3266c7bea4c50652.jpg)  
Figure 15. Additional qualitative comparison on image-to-layers task. This case highlights our method’s ability to handle complex multi-element designs. While commercial systems like RoboNeo suffer from severe artifacts and LayerD produces overly grouped layers that compromise fine-grained editing flexibility, our method maintains both quality and appropriate decomposition granularity.

![](images/cccc0af786f6ff60b0bc6a8aea201c7f5862cb63e8499f3822dfa1e655c291a6.jpg)  
Figure 16. Additional qualitative comparison on image-to-layers task. Our method excels at decomposing designs with overlapping elements and complex visual hierarchies. The comparison demonstrates superior performance across all three evaluation dimensions: quality (semantic correctness and transparency), integrity (faithful reconstruction), and granularity (appropriate decomposition level).

![](images/9f9c1f066c6473596b9268ad2d77220647d3d00e49f7a23a82c51d2f79e32033.jpg)

<details>
<summary>text_image</summary>

THANK
YOU
MOM
</details>

![](images/bc632b3c575abb5f962fd05bf906d00d658d179cc670a4ffa29f6ed72f3d57b6.jpg)

<details>
<summary>text_image</summary>

THANK
YOU
MOM
</details>

![](images/9f8a6fbf8b5be34fbc4abf66bd1024ebc7b87e591e34e826d306a493956dcde8.jpg)

![](images/2ed68fc827ae9c8f209eb0f0b574c4977ab3394bca16ebdb434ca13921b6643a.jpg)

<details>
<summary>natural_image</summary>

Collage of four panels showing a woman holding flowers, with decorative star patterns in the corner (no text or symbols)
</details>

![](images/9ffba93afe7bd52e00d6c3668e0e196c048b34aecd857d124991eacdbb924afd.jpg)

![](images/4bf83b8aec3b4175e890cd0255ee657b8ce78073c5da2010742be171ae6e6e95.jpg)

![](images/14fc892a34519dc0fe0c28927f4d688658d8299e38427364b95ff7cf9920eb11.jpg)

<details>
<summary>text_image</summary>

THANK
YOU
MOM
</details>

![](images/725b0b2a4ba3e3fffbc0f61db1415fe4bc060697c0e2b4314bcbb77b1bbd46a8.jpg)

![](images/eb1c82f6ccfe15ad57d065ebccede8b4cdf7cef1d722d36827d858327234b04c.jpg)

![](images/845ec697e235ffe5f16c7d47d05cfdd6c851f5309639558b4f58a856d29ca537.jpg)

<details>
<summary>text_image</summary>

THANK
YOU
MOM
</details>

![](images/30f6326c069919e227fbdee5a04f14a50accced2c6be26fd2ecc9fed0f2d31e8.jpg)

![](images/e3bbef25c1525f200284cc1e7e472c741464e8d75c37a14783f4c9adf51da6d2.jpg)

<details>
<summary>natural_image</summary>

Collage of four panels showing a star, floral patterns, and a woman holding flowers (no text or symbols)
</details>

![](images/35624f9f1daedb7e2b400c3592683ec9122782e57e584ca37cf67299476ae9af.jpg)

![](images/409ecfd0ac49b427fc3b8e09bb6e794c7ebe51ab497fd44a6480dba43ef72bcc.jpg)

![](images/c900889558894409fc335d622e46cea6b765d3e073cf78f3761c5789aff792cb.jpg)

<details>
<summary>text_image</summary>

THANK
YOU
MOM
</details>

![](images/53ac5a112ea69c7f2abe174c84d050f9a0c9f01c00f7c10761a6d855140d3ddf.jpg)

![](images/5dee069091b3b21a7fa2e8f52e533794786f686f2b198c421dbd61ea2c1502e5.jpg)

![](images/c08ba3e4987cfb2e942d92361197c6509ed88fa0e989285aae2632ae7763f687.jpg)

![](images/a4d110b892652bc95540642aedc1a9eda7c0b564a779880dd5727673c2c677fb.jpg)  
Figure 17. Additional qualitative comparison on image-to-layers task. This example showcases our method’s robustness across different design categories. Our decomposed layers maintain sharp boundaries, clean transparency, and semantic coherence, enabling practical editing workflows that commercial and academic baselines struggle to support.

![](images/54d2e862b8077520fde9c93008ddb31b1d2307d42b6c8162065efae546243146.jpg)

<details>
<summary>text_image</summary>

ONLINE
ONLINE
ONLINE
ONLINE
PARTY
</details>

MRT   
![](images/9881fea9277771948c25e20764a151326c99cbb27b17e94e2d30e5383928866b.jpg)

<details>
<summary>text_image</summary>

ONLINE
ONLINE
ONLINE
ONLINE
PARTY
</details>

![](images/25b32049c4aa06114cff08dd04e3d5130e33f3a5f36a3db4c2316b42441be4c7.jpg)

<details>
<summary>text_image</summary>

ONLINE
PARTY
ONLINE
</details>

![](images/cd748627636ceab07dd3c240b4904f52bfcfc4eb9cfdd81d3330eeef63428512.jpg)

![](images/315bca64e01579bedadb3374160c0ff44f62d425de426014e72870f8e2aefc1d.jpg)

<details>
<summary>text_image</summary>

ONLINE
ONLINE
ON THE
ONLINE
PARTY
</details>

![](images/a8e7f85c421a5824c8c41fb5175fd343fe60284fdf09610b107a1ad42db56fc3.jpg)

<details>
<summary>text_image</summary>

OLINE
</details>

![](images/98ef1b32c34bff7bbb72cb39a385472b035aa7590f639f2628716cd731591b56.jpg)

<details>
<summary>text_image</summary>

ORLINE PARTY
ORLINE ONLINE
</details>

![](images/d0a8ffcf50589d48a62c5348284bfc51119350ef0290c0cf3bdbb55845769895.jpg)

<details>
<summary>text_image</summary>

ONI
INO
ONE
PARTY
</details>

![](images/e1b355e199423df2e1895015d826442d1ac7d39f8234fa8fd712dac15e0a18ae.jpg)

<details>
<summary>text_image</summary>

CMO PAR
CMO PAR
</details>

![](images/1ad0cbf2935f680a8160893b3985afc4889ffdfab3c14bdfa94c8122764c5275.jpg)

![](images/19dc22777146059250737586d96307914d8fc5d6a79820a803ad610b93b91c03.jpg)

<details>
<summary>text_image</summary>

ONLINE
ONLINE
ONLINE
PARTY
</details>

![](images/dd597c3e28fe70037fe0807d607d4e6edc9fcea83347f9b7f685cea1e38a2c3f.jpg)

<details>
<summary>text_image</summary>

NI INE
NI INE
ONLINE
ONLN
PARTY
ONI
ONI
I'E
</details>

![](images/869c1cddc977ccf72173fd988f1b6d3f3455eecb55b8be7e94bcd47642696d53.jpg)

<details>
<summary>natural_image</summary>

Two photos of people in a room, one seated at a table and the other standing with a book (no visible text or symbols)
</details>

Figure 18. Additional qualitative comparison on image-to-layers task. Final comparison case demonstrating consistent quality advantages of our method. The decomposition preserves layer reusability and editability while maintaining visual fidelity, confirming the effectiveness of our masked region transformer framework for the image-to-layers task.

![](images/7801a71548d94801a9ae2a32c80bc2b40d8f093bebce01c054aae281033a1be8.jpg)  
Figure 19. Image-to-layers visualization with 6 layers. We visualize the layer-by-layer decomposition process showing individual RGBA layers with transparency. Each layer is displayed separately along with its alpha mask, and the merged composition demonstrates faithful reconstruction of the input design. This visualization demonstrates our method’s ability to generate clean, reusable layers with accurate spatial boundaries and alpha channels.

![](images/cd8a236c066ef53697c52acf1b25d7ae8bc7e53546037b066acd6eac580ab00a.jpg)  
Figure 20. Image-to-layers visualization with 8 layers. Decomposition result showing increased layer complexity with 8 distinct transparent layers. Our method successfully handles more complex compositions, maintaining layer quality and proper decomposition granularity across the extended layer hierarchy. Each layer preserves semantic meaning and can be independently edited.

Input   
![](images/5f53ca2afa1bd515dac0074e015f1bcaed9f7acda1e078493db2ee18ffddc70c.jpg)  
Layout   
Merged and layer images   
Figure 21. Image-to-layers visualization with 10 layers. Further demonstrating scalability to compositions with 10 transparent layers. Our masked region transformer maintains stable performance across different layer counts, producing coherent decompositions without architectural modifications. The visualization shows consistent layer quality from background to foreground elements.

Input   
![](images/f08f33a7810b51c6048c5c86a18a3e24543ef5c869667e76cf22fe13d48e7e33.jpg)

Lavout   
![](images/f9c4f8a6a3c9b23245b8aab9cb3ba66c8d6ea169e5f4954e92aa3da3c0563cfc.jpg)

Merged and layer images   
![](images/f330384febde1070d9622ec3dd0824949dee18b7ade2bc82e72deb26a9866178.jpg)

<details>
<summary>natural_image</summary>

Abstract geometric shapes including a large circle, a small circle with a calendar icon, and scattered brown circles on transparent background (no text or symbols)
</details>

![](images/5e0ee94e5057f62b166a22f2cd8bd176be66886395742eb44f3303d48e668c4a.jpg)

![](images/934662ec921ba45a672788993f711482ba70d8979488ec7c625935ec670ce23d.jpg)

![](images/27be6083d348ae3274bfbc9670e8f7bd98e6874ac9d0a56f46d8e64295d3331f.jpg)

<details>
<summary>text_image</summary>

COMEDY TALK
Dairy Beans
COMEDY
TALK
Ameri Cheese
</details>

![](images/a1afa5b24d4031ba0b90fe6618520c0c55e44c2892687baf096bd31cfc374410.jpg)

![](images/fa994f1eefcbafba9d47ab9c937d75be3acbd9b7fb3714501fe2f1e24302c476.jpg)

![](images/0d20e9c7413167cdec05165ab0028064d8338a1f33523dd75883aa6e7de9316d.jpg)

<details>
<summary>text_image</summary>

DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
DENTIST
</details>

![](images/66de05eda2dbf80063478560a1779e398eec9df19b562719cdd045fab6dab55d.jpg)

![](images/9624d5b67a6dae03f608e83aabc125ac8782c9757d73fde15767f72dabd68706.jpg)

![](images/1e1c7ae2c06771da7ebc76f72930a192013da975252394dc4d3716c11509fa85.jpg)

<details>
<summary>text_image</summary>

CRYPTOCURRENCY TRADING
CYPICALS
1.2000-2005
2.2000-3000
3.2000-4000
4.2000-5000
5.2000-6000
6.2000-7000
7.2000-8000
8.2000-9000
9.2000-10000
10.2000-11000
11.2000-12000
12.2000-13000
13.2000-14000
14.2000-15000
15.2000-16000
16.2000-17000
17.2000-18000
18.2000-19000
19.2000-20000
20.2000-21000
21.2000-22000
22.2000-23000
23.2000-24000
24.2000-25000
25.2000-26000
26.2000-27000
27.2000-28000
28.2000-29000
29.2000-30000
30.2000-31000
31.2000-32000
32.2000-33000
33.2000-34000
34.2000-35000
35.2000-36000
36.2000-37000
37.2000-38000
38.2000-39000
39.2000-4000
40.2
</details>

![](images/ed869840a09c449ed82d2936bd3d18a95096e5e1ac7814eb0ddc8e7ce0800d9d.jpg)

![](images/148d70c2fa0c4125328a1d3ada6220fede8dbab764cebfb2bac1720bbc59fef5.jpg)

![](images/fb00b873bbb61e55bd4bf6f54d41d9372ccc8e2452a920bb35f0a909bd7e23ac.jpg)

<details>
<summary>text_image</summary>

WATER HOME DECOR
A LUTTER OF
WATER HOME DECOR
A LUTTER OF
</details>

![](images/6ef9e2c775185ce0144e3cf3be9448fb48338351e376dd62d5bd2f4ef9d0f627.jpg)

![](images/31f4a9380b679e764008cb26f511611b726834a1640dea2d3b8634fda190ada7.jpg)

![](images/8ae8d2bb0408d2fe0871e97cef8c9ce43b08bd14fee18f7869349c3e9c713fc3.jpg)

<details>
<summary>text_image</summary>

Cleaning Service
MIL 10.50% of the
down Cleaning
MIL 20% of the
down Cleaning
Cleaning Service
MIL 10.50% of the
down Cleaning
MIL 20% of the
down Cleaning
</details>

![](images/87421ef31c2c15c77c0c963148285575aace7be0fab39a8096e139e02c72c5b9.jpg)

![](images/985500a2631a3809e827910251bc7d34315d3da29a82d010e59c40e64ce4b372.jpg)

![](images/ed643557d8d6cb3965b7eca1e3081d4913d1f62a7afe6285ff0fa01ae4676a06.jpg)

<details>
<summary>text_image</summary>

ARCHITECTURAL
DESIGNS
</details>

![](images/259b2bf54ef819fd5b2b8e6e93781f5a056984279c9fc70ed9bdc9b1052d7395.jpg)

![](images/e782ee4826fc6e065013488ff48b63de593a1a5aa3e522b91259ea3f49f512f0.jpg)

![](images/1a8e81ed25a3d454a5629afe27f2d7dec92789114d42cf9101932be30b96bca7.jpg)

<details>
<summary>natural_image</summary>

Two-panel image showing a girl taking a selfie with colorful abstract shapes in the background (no text or symbols)
</details>

![](images/45b8d3e249cd09e1c4c186d6ada0ab6a8291b4ac29247c71d4664152be827e0d.jpg)

![](images/a4e839b65ddb0fb2020ff178bf0c261240096424d4a50e62dfac188cf3b4eed5.jpg)

![](images/dea146a283268a18b94d2a9ee82d01516de959bda9f4cd2a7034eea87b9e0716.jpg)

<details>
<summary>text_image</summary>

SUSTAINABLE DESIGN ←
SUSTAINABLE DESIGN ←
</details>

Figure 22. Image-to-layers visualization with 12 layers. Decomposition of a complex design into 12 transparent layers, demonstrating our method’s capability to handle high layer counts while maintaining decomposition quality. Each layer retains sharp boundaries and proper alpha masks, essential for professional editing workflows.

![](images/43b06e59005a508f8ef22486330dc326d016f2190d9f064f37cdd3677518eee9.jpg)

![](images/f4a7e1e9dc54a92f01b237589172154139fa5fcf24457fe565d186fa8efc7bc5.jpg)  
Figure 23. Image-to-layers visualization with 14 and 16 layers. Two examples showcasing our method’s scalability to very high layer counts (14 and 16 layers respectively). As shown in Table 3, our approach maintains stable performance across a wide range of layer numbers from 2 to 50 layers, demonstrating flexibility in handling both simple and complex multi-element compositions.

# Layer Caption

<layer> “School”. The text is styled in a playful, rounded font with a bold appearance. The color of the text is a bright, light green. </layer> </layer> COLOR YOUR STUDIES! </layer>

<layer> Light pink footed baby pants with elastic waistband. </layer>

</layer> Creamy white abstract flower or cloud shape with five rounded lobes. </layer>

<layer> “School”. The text is styled in a playful, rounded font with a bold appearance. The color of the text is a bright, light green. </layer> </layer> COLOR YOUR STUDIES! </layer>

<layer> Vibrant floral arrangement of pink, white, and red flowers with green leaves.</layer>

</layer> Bright pink horizontal line with arrowheads at both ends. </layer>

<layer> A pastel blue Easter egg decorated with stripes and white dots.</layer>

</layer> Text: Happy. The text is styled in a light blue color </layer>

# Existed Layer

![](images/2a2d2a398be40a169db1324edf735fc97038064445f211ce0d7974034a2406a0.jpg)

<details>
<summary>text_image</summary>

Back to
t4-<-->
School Supplies
<-->
</details>

![](images/ae805f4d14bd0986d46c046077c42e71d365091f000522b674351c2ab9fd503d.jpg)

<details>
<summary>text_image</summary>

love
Baby Shower
The活力
Cafte
June 05, 20XX
3:00 PM
St. Louis Only
1278 94
</details>

![](images/5b12f62d22e0459c009044f48630dd4ee2a4b58f395ee7bda0ff043750d1b4d1.jpg)

<details>
<summary>text_image</summary>

HAPPY BIRTH DAY!
ASHLEY MATSON
</details>

![](images/f56849cbb1766d68a211ace43db219135d8354b37c0c57861a54239eac7f8e1c.jpg)

<details>
<summary>text_image</summary>

JENNY
&
BENNY
SAVE THE DATE
01 JUNE 20XX
GRAND-AMERICAN HOTEL
</details>

![](images/32362380bb2e28bbbb1ae74a6a72449ea200d014dd8c75ed70c6234716ab7e5b.jpg)

<details>
<summary>text_image</summary>

Easter SALE
60%
yourname
</details>

# Merged result

![](images/f3abb23a979f8e6248472eecf78e7aa2833be2a3c15e63c488aa53803416b680.jpg)

<details>
<summary>text_image</summary>

Back to School
School Supplies
COLOR YOUR STUDIES!
</details>

![](images/8633cd40570bd93ced1b62c0dd2b458f1c22105c524723a4a4045af7de952274.jpg)

<details>
<summary>text_image</summary>

love
Baby Shower
The活力
Draft | June 09, 20XX
4:00 PM | St. Louis City
1278 04
</details>

![](images/0d69240c0f768ac9fb0419c8b85a010fbaf5de8486d760f3d2bcec6287dce165.jpg)

<details>
<summary>text_image</summary>

HAPPY BIRTH DAY!
ASHLEY MATSON
</details>

![](images/84ea1c429f0545d2b5a115f0c94d9849569333e35525f2ccb1c0571c24068f71.jpg)

<details>
<summary>text_image</summary>

JENNY
&
BENNY
SAVE THE DATE
01 JUNE 20XX
GRAND ADMISSION OF HOTEL
</details>

![](images/c75906756e435d537d4a118895c8b5038726c398f8d2e16cb0e76a21f1bd9060.jpg)

<details>
<summary>text_image</summary>

Happy Easter SALE
60%
yourname
</details>

# Added Layer

![](images/f75b4d6d0027e321bbf1a82c1ba222df658662a6b705e64e6781bf34008eecbc.jpg)

<details>
<summary>text_image</summary>

School
</details>

![](images/871394721cdab7f65b42069783cb33667801bbdfadcb4484bf76e583499c4241.jpg)

<details>
<summary>natural_image</summary>

Pink pants with white shorts against a checkered background (no text or symbols)
</details>

![](images/b0c7895bc9e345555f7f6301021fbffe5271b18cb8d4f006d9d5f913f0432ea9.jpg)

<details>
<summary>natural_image</summary>

Yellow smiley face emoji with closed eyes and white cheeks, isolated on transparent background (no text or symbols)
</details>

![](images/59a00127e1bd44278a26a2eaa6f40d89743d4f89cee265dc9ae22db1a055ebb4.jpg)

<details>
<summary>natural_image</summary>

Decorative floral border with pink, white, and red flowers on a checkered background (no text or symbols)
</details>

![](images/1e1635d3fc28fd1702c973203d327dcc88b9638614de309c0e1e2cf043bad36e.jpg)

<details>
<summary>natural_image</summary>

Simple illustration of a light blue egg-shaped object with white dots on a checkered background (no text or symbols)
</details>

# Added Layer

![](images/4858b79bda4fc4b063dadc81aeb667cb51e1c6084f142a8eead2290dea2d5d21.jpg)

<details>
<summary>text_image</summary>

COLOR YOUR STUDIES/
</details>

![](images/d338a380491db683d2b88aab8f3663e8129428a1823c142566c7b35b72571110.jpg)

<details>
<summary>natural_image</summary>

Simple illustration of a pink heart inside a white cloud-like shape on a checkered background (no text or symbols)
</details>

![](images/c1f326880511d1daafea19742123e84fe6313c67d419431f508e7a4b62cf3f18.jpg)

<details>
<summary>natural_image</summary>

Abstract geometric pattern with a central star-like shape surrounded by four curved arcs on a checkered background (no text or symbols)
</details>

![](images/6c58a91917275b15002684f0fab5344cab930a7264fa9a08b56fbed3bf0e7fb3.jpg)

<details>
<summary>natural_image</summary>

Simple horizontal arrow pointing left on a checkered background (no text or symbols)
</details>

![](images/3c5650b2fccc6970ebe5c684d2cf35c4cb10488dae7786a78bff45955d7558ce.jpg)

<details>
<summary>text_image</summary>

Happy
</details>

Figure 24. Additional examples for layer addition task. We demonstrate the layers-to-layers capability by adding new layers to existing compositions based on text prompts. Our method generates new layers that maintain cross-layer consistency and harmonize with the existing design’s spatial layout and visual style. By generating multiple layers in a single pass and conditioning on all existing layers, our approach better captures inter-layer relationships and produces coherent insertions that preserve global composition.

![](images/10bd57f5a59b0921aa2b12e12f020ea5b2faf57f62dd51482de656fac4a8337e.jpg)  
Figure 25. Additional examples for layer restylization task. We visualize the transformation of user-provided assets into stylistically harmonized layers that match the overall composition. Our method performs this restylization in a single pass for all target layers, preserving geometric structure while adapting appearance to align with the existing design’s visual identity. The results demonstrate effective style transfer while maintaining layer coherence and compositional harmony.

# Prompt:

Theimagepresentsasereneandomanticseasidesetingcharacteredbysoftpastelcolorshemedomnatedbyarmpinktoshe backgroundeaturesdientthatrasitiosfroaltpeachathtotoeperoalueartetooigailset atmosphere.Tisetlorpaleteisomplemetedbyeaturaltextusofteeachndteocanichsibletod cutoutthatfrattaItreoddalseeadtdodiet arepostionedstcdoofueacerethusetigoethstialotwateabe issetwithdeeeshaddtoal theromanticaaefteelementsetrateicalyedtftdefteematitleCS presentedinaldodefotlethbtileWGOMACRisihlymalrize thethemefrodsetdedcolocsselogrodi readabilityalltdoai softedgesandledingfctseatoondiimospreThtftaisearlymanticnlebatoingit suitableforoetssntsfoixpereestaeroosaladedef intimacyandathealouleinteateemablomntstoetsiiftoapicit furterreinfosteafspalasggtingtattiseasdnnxpereouldrtargelebatiohn anniversaryooalOallsielapestheemaasidimpelll marketing or promotional materials aimed at couples seeking unique dining experiences.

Layout A   
![](images/faf606f196ce911aad7000a0bd9eff354710034726f71a50ca7539126e202801.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| #1 | 4 |
| #2 | 4 |
| #3 | 3 |
| #4 | 4 |
| #5 | 5 |
| #6 | 7 |
| #7 | 7 |
| #8 | 8 |
</details>

Generated A

Layout B   
![](images/126c1dc2bedb233bf823c0d2fae553738c71b44a8667c6d5053a5ba4499c8751.jpg)

<details>
<summary>text_image</summary>

#1 #7
#8
#3
#2
#11
#9
#12
#4
#10
#13
#14
#5
</details>

Generated B

Layout C   
![](images/af1af5d9c8b497743f22d793c610b3265e49859ad09838e7487df6f74853362b.jpg)

<details>
<summary>text_image</summary>

#1
#2
#3
#4
#9
#5
#7
#6
#8
#10
#11
</details>

Generated C

![](images/ae081c6394b031ad7af0e818a406627726b2373a0fe42e8acb5bdced262ef7db.jpg)

<details>
<summary>text_image</summary>

SEASIDE
PLACES
WITH A SPECIAL GIFT FOR
A ROMANTICl DINNER
</details>

Layer images A

![](images/abdb83433ffd8eb0bf30d8c7faa1b896d2208aa9c977ed39274280bb047f4fa0.jpg)

<details>
<summary>text_image</summary>

SEASIDE PLACES
WITH A SPECIAL GIFT FOR A ROMANTIC DINNER
www.reallygreatsite.com
</details>

Layer images B

![](images/ffb4c490a0622eb7eeff5d750227c08054318dc0bed16f593d7fae01c00ddbd4.jpg)

<details>
<summary>text_image</summary>

SEASIDE PLACES
reallygreatsite

WITH A SPECIAL GIFT
FOR A ROMANTIC DINNER
SEASIDE PLACES
</details>

Layer images C

![](images/6713d89408bb12587007e639297f2c77a33eabbc8feb29a4fc970501a49247a6.jpg)

<details>
<summary>natural_image</summary>

Abstract graphic with pink gradient, a curved arch, and a small photo of two people at sunset (no text or symbols)
</details>

![](images/4f4d2135dd58c8bb117492129153d1321824a24cbfc1cdff030a648dafc01519.jpg)

<details>
<summary>text_image</summary>

Grid of transparent squares with partial text labels, likely from a graphic design or layout guide.
</details>

![](images/88c27c3cbb13f180dcf1926cdb7c935f2ea585c76fe6bc5c7b1305b5359429a3.jpg)

![](images/79a562ccfc7faf713d3aa6ad05de3167becf276e2896cca5aa81bb7e6c6a9d1c.jpg)

<details>
<summary>natural_image</summary>

Grid of six panels showing beach scenes and ocean views, no text or symbols present
</details>

Figure 26. Text-to-layers: Merged image vs. layout visualization. Additional example demonstrating our model’s ability to generate well-composed multi-layer designs from text prompts. The side-by-side comparison shows how textual descriptions are translated into visual compositions (left) with structured layer hierarchies (right), highlighting the model’s capability to learn both aesthetic and structural design principles.

# Prompt:

TheimagepresentsaisallngaginpromotiallayoutdsidtoiglighthiindetuesiniiaTheackgrondeatursus grentropicagtiaotploddi densegreenerytatcreateasenseofdvenureandonnectiotonature.Itheupperleftquadranttetext“EXPOEVI'sKG TRLis whimsicaltouchtcthdeuroteloistalltedeomplemtsthrovad organizedappeaanceTeuppritadantshocsesandialeainidebedatitotpecdotheirolderis elementddsfuldooesthdodifspdialsisa earthyoiadoli depictedendtlo greatoutdorsepersneseisiarthletehicataisthesiesthicoftageowrhtdat featuresthetetEXOAVELatoowdbE"inaeroldotetslieeng consistencywittheoverallolorsheme.Thuseoftarsaroudthtextddsaouchofandawsateiotothecalltoactio encouragingvewrsegeitteicehtisticlethasodedeanitocataltsd cohesivecolorlnuiallsoeaeit isbothinviiniaoalcouinsakoiouaouldielydo contextsucalmtidratut lookingtoexploreVirginia'shiking trails.

Layout A   
![](images/6d18b7b32575659c80f685ada0e5f67a6586974e70e63d8876046586681fd2ab.jpg)

<details>
<summary>bar_stacked</summary>

| Category | Value 1 | Value 2 | Value 3 | Value 4 | Value 5 | Value 6 | Value 7 | Value 8 |
|---|---|---|---|---|---|---|---|---|
| #2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| #3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| #4 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| #5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| #6 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| #7 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| #8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
</details>

Generated A   
![](images/b2bab9275d2d89e23fa6140d0982c174c1d9f05ab511b2292c1b024a66adb52a.jpg)

<details>
<summary>text_image</summary>

EXPLORE VIRGINIA'S
HIKING TRAILS
EXO TRAVEL
BOOKING ONLINE
</details>

Layer images A   
![](images/5d8ae1c05b276457c5346c8b386bb82436b2fb5669459cd5fa597e5c830a64ff.jpg)

<details>
<summary>natural_image</summary>

Grid of six image thumbnails showing a green forest, two small animal figures, and transparent backgrounds (no text or symbols)
</details>

Layout B   
![](images/d79b84361fa1904d5e6837382808fa8fcc102dfa91e2d7dd613a4f29445e9819.jpg)

<details>
<summary>text_image</summary>

#1 #7
#8
#10
#13
#3
#2
#11
#14
#9
#12
#4
#5
</details>

Generated B   
![](images/71ee1374e937f1f639e21db0209c4e946c0f39e16daa834704a72b521f6e4463.jpg)

<details>
<summary>text_image</summary>

EXPLORE
VIRGINIA'S
HIKING TRAILS
EXO TRAVEL
BOOKING ONLINE
</details>

Layer images B   
![](images/3bdb17f30383ce9235cae00718c87e694b66d521d6ad310a0e536c0162160633.jpg)

<details>
<summary>natural_image</summary>

Grid of transparent background with small green icons in the top-left corner (no text or symbols)
</details>

![](images/d725a823d3ae7427a3d326e39eb4e4a6c8a8b72bf4504aa2d98778dd44f93c18.jpg)

Layout C   
![](images/5787d54a1fc851f51a30ff2a8ad2708c3361d4abd5db9919debd2794ccf165c5.jpg)

<details>
<summary>text_image</summary>

#1
#2
#4
#9
#11
#10
#3
#5
#7
#6
#8
</details>

Generated C   
![](images/a866c51cb50f91fa6c1d08193715eab12abb604ce18eff0f819ece4d8c02495b.jpg)

<details>
<summary>text_image</summary>

EXPLORE
VIRGINIA'S
HIKING TRAILS
EXO TRAVEL
BOOKING
ONLINE
</details>

Layer images C   
![](images/356b974189b890fb52170b06abcb82f4168b8ff1424639fab894b3573c65c48b.jpg)

<details>
<summary>natural_image</summary>

Grid of transparent tiles containing small images of monkeys and green foliage, no text or symbols present
</details>

Figure 27. Text-to-layers: Merged image vs. layout visualization. Another example showing the relationship between the generated merged design and its underlying layer layout structure. The layout visualization reveals how our model organizes multiple layers with appropriate spatial relationships, z-ordering, and compositional balance to create aesthetically pleasing designs from text descriptions.

Input   
![](images/5081fb1c5a32c1a8cdcdf6800f76e8640c146f93c639b2daed1f6ff742beb2e1.jpg)

<details>
<summary>text_image</summary>

your store
HAPPY
BIRTHDAY
www.yourweb.com
</details>

Layout A   
![](images/64ce9887ba87b8666e1f34f5b9ab02f38d4fde4a036e84202a1040d906a25779.jpg)

Layout B   
![](images/a2d83fd042b7e41ff9315c342528cd68c6df5cfe2002dfe7356dc711f0ff1c79.jpg)

<details>
<summary>text_image</summary>

#1
#4
#13
#11
#16
#7
#5
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#2
#9
#10
#3
#14
#15
#8
#12
</details>

Layout C   
![](images/b9491f7d5b26082128c330c69626b8fe2a4a0c4c7f7da9d2c8d7d886b68aec35.jpg)

<details>
<summary>text_image</summary>

#1
#7
#6
#3
#4
#5
</details>

Layer images A   
![](images/c519b33b08e284587d3317e755d3d9b40d57395acd9d4ffcafff2ff36e3f0000.jpg)

<details>
<summary>text_image</summary>

HAPPY
MANNER
</details>

Layer images B   
![](images/4b02305db5f1802d6be1e2c69a26f2f36f47b606671d90628749479a1608cd7d.jpg)

<details>
<summary>text_image</summary>

HAPPY
BIRTHDAY
December 2017
</details>

Layer images C   
![](images/80ce08097cf716465e1b43b86a3ad627af781842d7747b0eebde36da4a41132e.jpg)

<details>
<summary>text_image</summary>

HAPPY
BIRTHDAY
Your choice
Happy
BIRTHDAY
You're you
You're your choice
</details>

Figure 28. Image-to-layers: Merged image vs. layout visualization. We visualize the input image alongside the extracted layer layout structure for the image-to-layers decomposition task. This demonstrates how our method decomposes raster images into semantically meaningful layers with well-defined spatial boundaries. The layout representation shows bounding boxes and z-order that guide the decomposition process.

![](images/123acdbeb43d3866e638a95dc5152e7e5f49f956ca799056ed3af346c8a442a8.jpg)  
Figure 29. Image-to-layers: Merged image vs. layout visualization. Another example illustrating the correspondence between input raster images and their layer layouts. Our method leverages layout information (either from automatic detectors or manual annotations) to perform accurate layer decomposition. The layer grouping augmentation strategy helps improve robustness to noisy or ambiguous layout specifications.

Input   
![](images/6bead4d5242b034498abf51f1231be85adf83999097b02836e32f659a5f2fdbf.jpg)

<details>
<summary>text_image</summary>

Vintage Shop
50%
Off
Shop now at yourname
</details>

Layout A   
![](images/abdd7c71de8e8e881e64f1c70e604f14454f84c0806e13c699ed4a51aab7f062.jpg)

Layout B   
![](images/eaaad75b34bf76797cbd402be181a11d9a6b29f25a18dd660801c8e33332e072.jpg)

<details>
<summary>text_image</summary>

#1
#2
#12 #5 #3
#13
#7 #10 #9
#4
#6
#11
</details>

Layout C   
![](images/cacdfb19826ed4814fedf695dbc45bd222a994f61d93bbe599f1ac35fca1dae2.jpg)

<details>
<summary>text_image</summary>

#1
#2
#7 #5 #3
#4
#6
</details>

Layer images A   
![](images/ab3fc084bbae5f2cc9dda5285932e49c26986d3d90d6fc6f9f8d39bdcadfaa8e.jpg)

<details>
<summary>text_image</summary>

Vintage Shop
Village Shop
*
</details>

Layer images B   
![](images/ca1cced27984e8c307b3a2a8893cb649a0f9e404a5ba0ed7e6bfe86e8a30675a.jpg)

<details>
<summary>text_image</summary>

Vintage Shop
Shop now at yourname
Vintage Shop
</details>

Layer images C   
![](images/948f1c45381bdd22d12b34702e0ee77ac1289ecba637f2ef1da4d6284868a781.jpg)

<details>
<summary>text_image</summary>

Vintage Shop
20% Off
Shop now at yourname

Vintage Shop

Shop now at yourname
</details>

Figure 30. Image-to-layers: Merged image vs. layout visualization. Final example showing the input-layout relationship in image-to-layers decomposition. This visualization confirms our method’s ability to handle diverse design categories and layout complexities, producing high-quality transparent layers that can be independently edited while maintaining faithful reconstruction of the original composition.

![](images/3ed7b99491db3dcc4add658f043801bfcb3f87706012e398e8fc963cddd409bb.jpg)  
Figure 31. Image-to-layers on real-world photographs: Limitation analysis. We demonstrate our method’s generalization to out-of-domain natural images. Despite being trained exclusively on design datasets, our model can decompose real photographs into layers. However, as discussed in the Limitations section, the model faces challenges with physical effects like shadows—often excluding shadow regions from object layers and leaving them on the background. This limitation stems from the domain gap between planar designs and real-world scenes with lighting effects. Nevertheless, the strong visual understanding from the Qwen-Image backbone enables reasonable generalization, with most objects successfully separated.

![](images/00666e2ad8bbd1708e54e0c9bc97fa8002cd4e22f8b66feb06e3252852c479b9.jpg)  
Figure 32. Failure cases and limitations. We present representative failure cases across our tasks. A common issue (top right) is the ”gray background” artifact, where transparent areas are decoded as gray due to the ambiguity of 3-channel VAE encoding. Other limitations include (bottom left) malformed glyphs when generating very small text, and (bottom right) occasional failures in identity preservation and instruction following during layer-to-layer editing.

![](images/652af9ebbb584cc92c1091cb2302797f2cd0143389b544436f85ad4d27262ac4.jpg)

<details>
<summary>text_image</summary>

S3 User Study
System
Image Group 1 / 20
Image Pair 1: case_0.png
Blind Test Mode
The positions of the
left and right images
have been randomized
to avoid position bias.
Total Image Pairs
20
Rated
0
Unrated
20
Navigation
Jump to image group
1
-
Export Results
Generate and
Download Ratings
Unrated List
First
Previous
Next
Last
Left Image
Right Image
Text Content
Caption
The image presents a vibrant and festive design celebrating the
Christmas season. The background is a rich, dark navy blue, which
serves as a striking canvas for the colorful holiday-themed illustrations
scattered throughout. These illustrations include a variety of
Christmas motifs such as decorated trees, snowflakes, bells, stockings,
and ornaments, all rendered in a playful, cartoonish style. The colors
used in these elements are bright and cheerful, featuring reds, greens,
yellows, and whites, which contribute to the overall festive
atmosphere.
At the center of the image is a bold, rectangular purple banner that
contacts deeply with the dark background. The banner contains the
Please Rate
Elements & Layout
Left is better ○ Tie ○ Right is better
Aesthetics
Left is better ○ Tie ○ Right is better
Typography
Left is better ○ Tie ○ Right is better
Overall Preference
Left is better ○ Tie ○ Right is better
</details>

Figure 33. User study interface for text-to-layers evaluation. Two generated results are displayed side-by-side with the text caption shown on the right. Participants vote across four dimensions: elements (layout), aesthetics, typography, and overall preference.

![](images/455ea8b49cf21c30bdd8d1ec1c717da6102432d421afa1fd2c7a5a28aee8f0fb.jpg)

<details>
<summary>text_image</summary>

S3 User Study System
Wind Text Mode
The positions of the left and right images have been randomized to avoid
position bias.
Total Image Plans
20
Rated
0
Unrated
20
Navigation
Jump to Image group
1
Export Results
Generate and Download Ratings
Unrated List
Unrated Image groups: 20
Image Pair 1: case_0.png
Left Image
MERRY
XMAS
Right Image
Path: ...case_0.png
Path: ...use_0.png
Text Content
Caption
The image presents a vibrant and festive design celebrating the Christmas season. The background is a rich, dark navy blue, which serves as a striking canvas for the colorful holiday-themed Illustrations scattered throughout. These Illustrations include a variety of Christmas motifs such as decorated trees, snowflakes, bells, stockings, and ornaments, all rendered in a playful, cartoonish style. The colors used in these elements are bright and cheerful, featuring reds, greens, yellows, and whites, which contribute to the overall festive atmosphere.
At the center of the image is a bold, rectangular purple banner that contrasts sharply with the dark background. The banner contains the text "MERRY XMAS" in large, white, sans-senif letters that are both eye-catching and easy to read. The font is playful and rounded, enhancing the cheerful tone of the design. Above the text, there is a small, hand-drawn heart in white, adding a personal touch and conveying warmth and affection associated with the holiday season.
The artistic style of the image is modern and whimsical, characterized by flat illustrations and a lack of intricate details, which gives it a lighthearted and approachable feel. The use of bright colors and simple shapes creates a sense of joy and celebration, making it suitable for a wide audience.
Please Rate
Elements & Layout
Left is better  Tie  Right is better
Aesthetics
Left is better  Tie  Right is better
Typography
Left is better  Tie  Right is better
Overall Preference
Left is better  Tie  Right is better
First  Previous  Next  Last
</details>

Figure 34. User study interface for image-to-layers evaluation. The reference input image is displayed at the center with decomposition results from two methods shown on both sides. Participants evaluate based on three metrics: granularity, layer integrity, and layer quality.