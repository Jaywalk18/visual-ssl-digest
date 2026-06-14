# Modality Forcing for Scalable Spatial Generation

Bardienus Pieter Duisterhof1,2 Deva Ramanan1 Jeffrey Ichnowski1

Justin Johnson2 Keunhong Park2

1Carnegie Mellon University 2World Labs

![](images/040b385e54180468a6f86e316daea749939a6605bcb37d325c1e1709221fca6f.jpg)

Project Page

![](images/53f3b3c4f5acfb62636439cdce771b2be10a0dc2431f9927fcdc72a879228475.jpg)

Code

![](images/1644f12e6246fbd58812e111b1e1d6e0153ca0a80fde5fe1d7861013e7c4c27b.jpg)

HuggingFace Demo

![](images/9af2d979b38c886a0493d392a02adb5043e67cd1485747fa5b730a81b071cf2a.jpg)

<details>
<summary>natural_image</summary>

Collage of nature-themed images including a person in a whale suit, a bear holding a sign reading 'Modality Forcing', and illuminated lamp scenes (no readable text or symbols)
</details>

“A whale-themed bedroom.”  
“An anime badger with a sign.”

![](images/8ec56a0f0962b7ee3e9687a1c4d20e421f86577f2a7b13af9a66d75d8d5092d1.jpg)

<details>
<summary>natural_image</summary>

Composite image showing a historical woman in traditional attire, a kitchen with steam stove, and four inset photos of interior scenes (no visible text or symbols)
</details>

“A Dutch woman.”  
“A traditional Korean kitchen.”

(a) Joint Generation: Text ⇒ Image + Depth  
![](images/2ae4fe80fc17e98cb685ba6ac0d6baf71dbd43c774329f151ef3fa7abab25747.jpg)

<details>
<summary>natural_image</summary>

Group of four puppies in striped outfits sitting on a grassy mat, with one dog and bowl nearby (no text or symbols visible)
</details>

(b) Conditional Generation: Image ⇌ Depth  
![](images/ebf391819cb74e9083975c93dd60c8df4dadd34d5614a41dffe1b62da7fa4f51.jpg)

<details>
<summary>natural_image</summary>

Silhouettes of three dogs in a dimly lit room with purple and orange gradient background (no text or symbols)
</details>

![](images/e006503eed9d23acf4ec937a79c3f01cf8ee9ce5159a18539c4fa4088b9746ce.jpg)

<details>
<summary>line chart</summary>

| Model Size | pretrained | scratch |
| ---------- | ---------- | ------- |
| 370M       | 0.957      | 0.933   |
| 800M       | 0.963      | 0.942   |
| 3B         | 0.972      | 0.944   |
</details>

(c) Scales with T2I model size  
Figure 1: We present Modality Forcing, a post-training recipe to extract spatial priors from text-toimage (T2I) models. A single DiT models the joint distribution over images and depth, enabling joint and conditional generation in arbitrary combinations. We demonstrate depth predictions improve with increasingly capable T2I pretraining, suggesting T2I is a scalable objective for spatial generation.

Abstract. Text-to-image (T2I) models contain rich spatial priors. Synthesizing photorealistic, cluttered scenes requires an understanding of geometry, including perspective and relative scale. Prior works adapt T2I models to leverage this prior for depth prediction, but they require dense depth data and involve complex recipes. We propose Modality Forcing, a simple, scalable post-training recipe for joint image-depth generation using a single DiT trained on sparse depth data. Modality Forcing enables conditional and joint generation of image and depth in any permutation by assigning separate noise levels per modality. Per-modality decoders let us train on sparse, real-world depth and achieve strong, generalizable depth prediction. We further show that Modality Forcing inherits the scalability of T2I pre-training: by training a set of T2I models from scratch (370M to 3.3B parameters), we find that larger models trained on more image data produce more accurate depth. Our strongest model is competitive with state-of-the-art monocular depth estimators and reduces AbsRel by 57% relative to existing joint image-depth generative models. These results provide strong evidence that image generation is a scalable pre-training objective for spatial perception.

![](images/12e75e48cc57de8a5376cb2c400a3b447c65b10a152cb43dd5d7ff88c87c17be.jpg)

<details>
<summary>natural_image</summary>

Scenic canal scene with bridge, bicycle, and person walking (no visible text or symbols)
</details>

Amsterdam canal at sunrise.

![](images/a348b9ab0d95b6edd4610c685e1f9514cdb68328e240cf6edaf0fa84106f7f55.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern restaurant with patrons dining and illuminated windows (no visible text or signage)
</details>

Modern restaurant at dusk.

![](images/84cef9683dc5d39d365ba3412567474d3b6b5ce1cd299274d8a45635a4990913.jpg)

<details>
<summary>natural_image</summary>

Group of pandas eating bamboo in a natural setting with decorative elements (no visible text or symbols)
</details>

Bamboo forest tea house.

![](images/c8bfdcd21ef653d8ccf85876e92a28f637ecfceab208152b88c6f9826c95a19e.jpg)

<details>
<summary>natural_image</summary>

Interior view of a futuristic exhibition space with large windows and a central display (no visible text or symbols)
</details>

Futuristic robotics lab.

![](images/92d9be577d7d25d159479a8baa8c8f0bed29fc69e2ab9f212e7f224b008cac5b.jpg)

<details>
<summary>natural_image</summary>

Tropical beach scene with vendors, shoppers, and palm trees (no visible text or symbols)
</details>

Tropical beach market.

![](images/a5975c2384784188d02bd7a053032252583889834e51e8085cd2382e2314d2f3.jpg)

<details>
<summary>natural_image</summary>

Interior view of a grand cathedral with ornate architecture and a large gathering of guests seated at tables (no visible text or signage)
</details>

Medieval castle feast.

![](images/d920c25d5909d1c87cae177464c525f1a91eb8bd4163e97755d2a33fd0f7e41c.jpg)

<details>
<summary>natural_image</summary>

Illustration of a surreal landscape with waterfalls, distant city skyline, and starry sky (no text or symbols)
</details>

Floating islands above clouds.

![](images/858389acaab1617529ffd558c29ab45ec0c3d194f342ab0579fc8e209c75c69d.jpg)

<details>
<summary>natural_image</summary>

Night city street scene with illuminated buildings and a large crowd, no visible text or signage
</details>

Neon city plaza at night.  
Figure 2: Modality Forcing generates rich RGB-Depth from text prompts. Unprojecting the points to 3D, Modality Forcing generates faithful and sharp geometry. The same checkpoint enables monocular depth estimation, and depth-to-image generation competitive with the best specialist models.

## 1 Introduction

Over the past several years, simple architectures and formulations fed with plentiful data have outperformed complex human-designed pipelines in many tasks. Examples include DUSt3R [40] for 3D reconstruction, SAM [16] for segmentation, and SAM-3D [33] for mesh generation. A central lesson is that general methods capable of absorbing more data are more effective than complex heuristics—often called the bitter lesson [32].

Models trained on task-specific data have led to considerable advances; however, 3D data is scarce which makes it difficult to scale to the quantities required for large models. One approach to ease this is to combine multiple synergetic tasks in a unified multi-task formulation. Language modeling is the clearest example, where a single generalist model trained on next-token prediction [5] exceeds specialist systems across translation, question answering, and coding.

Text-to-image (T2I) models contain similarly rich representations, but adapting them to new spatial tasks is difficult. First, they denoise latent tokens in an image VAE which is not designed for arbitrary spatial modalities. Depth as images limits training to dense (usually synthetic) samples and precludes encoding other spatial modalities such as meshes or point clouds. Second, post-training T2I models for spatial tasks may destroy its pretrained backbone and limit the pre-training benefit. Post-training generative models to produce spatial modalities requires a simple and scalable recipe.

We introduce Modality Forcing, a post-training recipe that enables a pre-trained T2I model to predict depth. Modality Forcing combines latent RGB tokens and pixel-space depth tokens into a single DiT to model the joint image-depth distribution. Pixel-space depth diffusion enables learning from sparse real-world depth annotations and achieves state-of-the-art results. Once trained, Modality Forcing allows setting RGB or depth as a target or conditioning, enabling joint generation, or conditional image-to-depth (I2D) or depth-to-image (D2I) generation.

Prior work has also modeled the joint RGB-depth distribution, but relied on learned adapters and was limited to dense depth data [6, 46]. We show that post-training a single DiT on the joint distribution unlocks training on real-world data and improves depth predictions by 57% on average. Concurrent work explored native multimodal pre-training [43] and instruction fine-tuning [12], representing depth as images. These models show encouraging results across visual tasks but share the same challenges with data, and leave it unclear whether depth quality scales with T2I capability.

We contribute a controlled scaling study to investigate the scalability of Modality Forcing with T2I model capability. We train a family of DiT’s from scratch, spanning from 370M to 3.3B parameters, and apply the same Modality Forcing recipe to each. We find that increasingly capable T2I models, larger models trained on more images, produce more accurate depth predictions. This suggests T2I is a scalable pre-training objective for spatial generation.

## Contributions.

• Modality Forcing, a simple post-training recipe that unifies three spatial tasks—monocular depth estimation (I2D), depth-to-image (D2I), and joint image-depth generation with one set of weights.  
• A controlled scaling study from 370M to 3.3B parameters and from zero to 1.92B images, showing that depth predictions scale with T2I model and training size.  
• State-of-the-art results: applied to FLUX.2-klein-9B, Modality Forcing competes with the best depth estimators and outperforms joint image-depth baselines by a margin of 57%.

## 2 Related Work

Text-to-image. Text-to-image (T2I) models have advanced rapidly over the past few years [10, 18, 27, 34]. The Stable Diffusion family of models began with latent diffusion models [27], which perform denoising in the latent space of a pretrained VAE rather than in pixel space; this became the dominant template for efficient image synthesis. A major architectural shift came with Stable Diffusion 3 [10], which replaced the U-Net with a Diffusion Transformer (DiT) [25] trained under a rectified-flow objective. The current state of the art for open-source models is largely set by FLUX [18] and Z-Image [34] — both flow transformers operating in the latent space of an image VAE and trained on large volumes of curated data. Whether this growing T2I capability reflects a correspondingly stronger underlying spatial prior remains untested; Modality Forcing answers this by post-training a controlled family of T2I checkpoints into joint image-depth generators under one recipe.

Monocular depth estimation. Monocular depth estimation has recently been transformed by foundation models trained on large corpora [38, 39, 44, 45]. Depth Anything V2 [45] scales by distilling a large teacher trained on synthetic data through pseudo-labeled real images, yielding strong zero-shot depth. Other works repurpose T2I models as depth priors: Marigold [15] fine-tunes Stable Diffusion on synthetic RGB-D pairs. The DiT architecture itself has also been adapted for depth: Pixel-Perfect Depth [42] shows that diffusing depth in pixel space avoids edge artifacts. Most recently, MoGe [38] reframes the problem as direct prediction of an affine-invariant 3D point map from which depth and intrinsics can be derived, and MoGe-2 [39] extends this to metric scale. Together, these lines of work establish that foundation-style learning on large 3D datasets yields high-quality depth and that T2I models carry exploitable spatial priors. Despite the rapid progress of T2I models, the extent to which their spatial priors transfer to depth has not been systematically studied.

Joint Image-Depth Generation. A growing body of work tackles jointly generating images and depth, motivated by the observation that a single joint distribution over RGB and depth unifies conditional and joint generation. LDM3D [30] retrains a Stable Diffusion VAE and U-Net on six-channel RGB-D inputs. JointNet [46] duplicates the pretrained U-Net into a parallel depth branch tightly coupled to a frozen RGB branch, supporting bidirectional prediction via channel-wise inpainting. UniCon [21] generalizes this with LoRA and disentangled per-branch noise schedules, recovering depth-conditioned generation, depth estimation, and joint sampling in a single model. JointDiT [6] ports the paradigm to diffusion transformers atop FLUX.1, introducing unbalanced per-modality timestep sampling. Orchid [17] instead trains a joint VAE over color, depth, and normals, paired with a single latent diffusion model. All of these methods rely on dense RGB-D supervision, typically from synthetic data or curated captured datasets. In contrast, Modality Forcing is a post-training recipe that scales with the base T2I model and removes the requirement for dense supervision.

![](images/730559908d97b03872dc58542bee8d2294657cd87b6f007da1103f877c040626.jpg)  
Figure 3: Modality Forcing is a recipe to post-train image-generation models for depth prediction. We encode RGB using a pretrained VAE and depth using a per-pixel tokenizer. During training (a), each modality is independently noised and trained with separate losses. At inference (b), each modality’s noise level can be controlled to yield image→depth, depth→image, or joint generation.

## 3 Method

Modality Forcing post-trains a pretrained text-to-image DiT to model the joint distribution over RGB and depth. The core idea is to give each modality its own noise level, so that a single set of weights supports joint generation and either conditional direction. Tokenizing depth in pixel space further lets the model learn from sparse, real-world depth supervision.

## 3.1 Problem Definition

We unify three generation tasks under a single model: joint image and depth generation, depth to image, and image to depth. Let c denote a text prompt, $\mathbf { \bar { x } } \in \mathbb { R } ^ { H \times W \times 3 }$ an RGB image, and $\mathbf { d } \in \bar { \mathbb { R } } ^ { H \times W }$ a depth map. We learn a joint generator

$$
p _ {\theta} (\mathbf {x}, \mathbf {d} \mid \mathbf {c}) \tag {1}
$$

from a set of $( \mathbf { c } , \mathbf { x } , \mathbf { d } )$ triples. A single model $p _ { \theta }$ then supports three tasks during inference:

$$
(\mathbf {x}, \mathbf {d}) \sim p _ {\theta} (\mathbf {x}, \mathbf {d} \mid \mathbf {c}) \quad \text { Joint   RGB - D   generation } \tag {2}
$$

$$
\mathbf {d} \sim p _ {\theta} (\mathbf {d} \mid \mathbf {x}, \mathbf {c}) \quad \text { Image - to - depth   (I2D) } \tag {3}
$$

$$
\mathbf {x} \sim p _ {\theta} (\mathbf {x} \mid \mathbf {d}, \mathbf {c}) \quad \text { Depth - to - image   (D2I) } \tag {4}
$$

Preliminaries. Diffusion models are an expressive class of generative models for high-quality generation. We discuss flow-matching with v-prediction [23] and x-prediction [20]. Let $\mathbf { x } _ { 0 } \sim q ( \mathbf { x } )$ denote a clean data distribution and $\mathbf { \epsilon } \gets \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ Gaussian noise. We interpolate linearly between the two with a scalar $t \in [ 0 , 1 ]$ to create training samples,

$$
\mathbf {x} _ {t} = (1 - t) \mathbf {x} _ {0} + t \boldsymbol {\epsilon}, \tag {5}
$$

so that $\mathbf { x } _ { \mathrm { 0 } }$ is clean and $\mathbf { x } _ { 1 }$ is pure noise. The constant velocity along this path is $\mathbf { v } = \mathbf { x } _ { 0 } - \epsilon$ . With v-prediction we parameterize the output as the velocity and regress a network $g _ { \boldsymbol { \theta } } ( \mathbf { x } _ { t } , t )$ directly:

$$
\mathcal {L} _ {\mathrm{v-pred}} = \mathbb {E} \left[ \| g _ {\theta} (\mathbf {x} _ {t}, t) - \mathbf {v} \| _ {2} ^ {2} \right]. \tag {6}
$$

Recent work has shown that v-prediction can struggle in high-dimensional spaces [20], instead suggesting to parameterize the output as the clean sample, $\hat { \mathbf { x } } _ { 0 } = g _ { \theta } ( \mathbf { x } _ { t } , t )$ . The implied velocity $\hat { \mathbf { v } } = ( \hat { \mathbf { x } } _ { 0 } - \mathbf { x } _ { t } ) / t$ is regressed to v:

$$
\mathcal {L} _ {\mathrm{x-pred}} = \mathbb {E} \left[ \left\| \left(g _ {\theta} (\mathbf {x} _ {t}, t) - \mathbf {x} _ {t}\right) / t - \mathbf {v} \right\| _ {2} ^ {2} \right]. \tag {7}
$$

This is well-conditioned when the data lies on a low-dimensional manifold (such as natural images or depth maps). Both methods sample the learned vector field from $t = 1 \ \mathrm { t o } \ t = 0$ with an ODE solver.

## 3.2 Modality Forcing

We introduce Modality Forcing, a diffusion algorithm with per-modality noise levels. Assigning separate noise levels allows conditional and joint generation in any permutation: an image with no noise conditions, while a full noise image is a generation target. Several methods use this idea, but differ in the axis along which the noise varies. Teacher Forcing [41] runs along sequence position, holding past tokens clean while denoising the next. Diffusion Forcing [7] varies noise level per token. Latent Forcing [1] denoises DINO [24] latents ahead of raw pixels. Modality Forcing extends this idea to the axis of modality: RGB and depth carry their own noise levels within a diffusion process, so that fixing a modality at $t = 0$ makes it conditioning. The schedule alone selects joint, image-to-depth, or depth-to-image generation.

We post-train existing T2I models to support all sampling schemes. During training, we sample per-modality noise levels in three different ways to support our three generation tasks. Joint RGB-D generation samples $t _ { \mathrm { r g b } } , t _ { \mathrm { d e p t h } } \in [ 0 , 1 ]$ ; I2D fixes $t _ { \mathrm { r g b } } = 0$ and samples $t _ { \mathrm { d e p t h } } \in [ 0 , 1 ] ;$ ; and D2I samples $t _ { \mathrm { r g b } } \in [ 0 , 1 ]$ and fixes $t _ { \mathrm { d e p t h } } = 0$ .

Depth tokenizer. Real-world videos typically only contain sparse depth annotation, since depth is estimated, e.g., using multi-view stereo (MVS) pipelines. To allow for post-training on these sparse annotations, we denoise depth directly in pixel space rather than the latent VAE space.

This is in contrast to prior work [6, 15, 21] which tokenizes depth through a pre-existing or new image VAE, with no obvious mechanism to accommodate partial depth supervision. We fill missing pixels with isotropic Gaussian noise to signal to the model that depth is not available at those locations, equivalent to how a fully missing depth map is encoded.

Per-modality timestep conditioning. Because each modality is denoised at its own noise level, every token must be modulated by its own timestep. We give RGB and depth separate timestep embedders: the RGB stream reuses the pretrained embedder, while depth receives a freshly initialized one. Since joint and conditional generation couple the two noise levels, we additionally let each stream’s modulation observe the other modality’s timestep through a lightweight cross-stream mixing module—one small embedder per direction. We initialize both to zero, so the embedding begins as an exact identity with no cross-communication and learns the coupling over training.

Depth detokenizer. The DiT blocks are followed by a depth detokenizer with n layers of self-attention, followed by a final linear layer that maps the depth tokens back to pixel space. The extra depth blocks help create depth-specific capacity without disrupting the RGB stream. We normalize depth supervision by scaling it to have unit mean, followed by spatial contraction [2] to ensure $d \in [ 0 , 2 ]$ .

Initialization. We warm-start the depth pathway rather than training it from scratch. We initialize the depth stream by cloning the pretrained image-stream weights; the remaining depth-specific modules—the pixel-space tokenizer, the depth timestep embedder, the detokenizer, and the crossstream mixers—are initialized from scratch.

## 3.3 Self-Distillation

T2I models are trained on billions of images; post-training them on millions of less diverse images may erode the rich prior. To preserve it, we introduce a self-distillation loss that penalizes the student for drifting from the original T2I checkpoint (similar in spirit to ‘Learning without Forgetting’ [22]). At train time, we pass the current noisy RGB through the frozen T2I model and record its predicted velocity $\mathbf { v } _ { \mathrm { t 2 i } }$ , then penalize the deviation of the student’s RGB velocity v:

$$
\mathcal {L} _ {\text { dist }} = (\lambda_ {\text { hi }} t _ {\text { depth }} + \lambda_ {\text { lo }} (1 - t _ {\text { depth }})) | | \mathbf {v} _ {\text { t2i }} - \mathbf {v} | | _ {2} ^ {2}, \tag {8}
$$

where $| | \mathbf { v } _ { \mathrm { t 2 i } } - \mathbf { v } | | _ { 2 } ^ { 2 }$ is the L2 loss between the predicted velocity and the T2I-predicted velocity. We set $\lambda _ { \mathrm { h i } } > \lambda _ { \mathrm { l o } }$ so the penalty is strongest at $t _ { \mathrm { d e p t h } } = 1 \colon$ : when depth is fully noised it carries no information. As depth is denoised toward $t _ { \mathrm { d e p t h } } = 0$ , it supplies context the T2I model never sees, so we relax the penalty rather than force agreement with a now-uninformed teacher.

Table 1: Training data. We train on twelve realworld and simulated datasets totaling 17M frames across 58k scenes, spanning indoor scanning, outdoor driving, and synthetic rendering.

<table><tr><td>Dataset</td><td># Scenes</td><td># Frames</td></tr><tr><td>Argoverse 2</td><td>700</td><td>108,139</td></tr><tr><td>Aria Project Sim</td><td>5,022</td><td>2,424,158</td></tr><tr><td>ARKitScenes</td><td>5,041</td><td>744,200</td></tr><tr><td>Blended MVS</td><td>493</td><td>113,209</td></tr><tr><td>FoundationStereo</td><td>40,936</td><td>40,936</td></tr><tr><td>Hypersim</td><td>457</td><td>74,519</td></tr><tr><td>MegaDepth</td><td>113</td><td>19,447</td></tr><tr><td>ParallelDomain</td><td>1,520</td><td>239,840</td></tr><tr><td>ScanNet v2</td><td>1,200</td><td>94,751</td></tr><tr><td>TartanAir v2</td><td>1,122</td><td>8,563,146</td></tr><tr><td>Taskonomy</td><td>520</td><td>4,385,534</td></tr><tr><td>Waymo Open</td><td>796</td><td>159,000</td></tr><tr><td>Total</td><td>57,920</td><td>16,966,879</td></tr></table>

Table 2: Base T2I model sizes used for scaling experiment, ranging from 370M to 3.3B parameters.

<table><tr><td>Param Count</td><td>Token Size</td><td>FFN Dim</td><td>Depth</td></tr><tr><td>370M</td><td>1024</td><td>3,072</td><td>30</td></tr><tr><td>800M</td><td>1536</td><td>4,608</td><td>30</td></tr><tr><td>3.3B</td><td>3072</td><td>9,216</td><td>30</td></tr></table>

each line = T2I pretraining images  
![](images/3140a1e59c22d208af084c38fa0c97fe8273ba0c3ccb0279810d881fe4f0e6fe.jpg)

<details>
<summary>line chart</summary>

| Model parameters | AbsRel | δ₁ |
| --- | --- | --- |
| ---------------- | ------ | -- |
| 370M | 0.057 | 0.958 |
| 800M | 0.051 | 0.963 |
| 3B | 0.044 | 0.971 |
| 3B | 0.063 | 0.945 |
| 3B | 0.062 | 0.945 |
| 3B | 0.061 | 0.945 |
| 3B | 0.060 | 0.945 |
| 3B | 0.059 | 0.945 |
| 3B | 0.058 | 0.945 |
| 3B | 0.057 | 0.945 |
| 3B | 0.056 | 0.945 |
| 3B | 0.055 | 0.945 |
| 3B | 0.054 | 0.945 |
| 3B | 0.053 | 0.945 |
| 3B | 0.052 | 0.945 |
| 3B | 0.051 | 0.945 |
| 3B | 0.050 | 0.945 |
| 3B | 0.049 | 0.945 |
| 3B | 0.048 | 0.945 |
| 3B | 0.047 | 0.945 |
| 3B | 0.046 | 0.945 |
| 3B | 0.045 | 0.945 |
| 3B | 0.044 | 0.945 |
| 3B | 0.043 | 0.945 |
| 3B | 0.042 | 0.945 |
| 3B | 0.041 | 0.945 |
| 3B | 0.040 | 0.945 |
| 3B | 0.039 | 0.945 |
| 3B | 0.038 | 0.945 |
| 3B | 0.037 | 0.945 |
| 3B | 0.036 | 0.945 |
| 3B | 0.035 | 0.945 |
| 3B | 0.034 | 0.945 |
| 3B | 0.033 | 0.945 |
| 3B | 0.032 | 0.945 |
| 3B | 0.031 | 0.945 |
| 3B | 0.030 | 0.945 |
| 3B | 0.029 | 0.945 |
| 3B | 0.028 | 0.945 |
| 3B | 0.027 | 0.945 |
| 3B | 0.026 | 0.945 |
| 3B | 0.025 | 0.945 |
| 3B | 0.024 | 0.945 |
| 3B | 0.023 | 0.945 |
| 3B | 0.022 | 0.945 |
| 3B | 0.021 | 0.945 |
| 3B | 0.020 | 0.945 |
| 3B | 0.019 | 0.945 |
| 3B | 0.018 | 0.945 |
| 3B | 0.017 | 0.945 |
| 3B | 0.016 | 0.945 |
| 3B | 0.015 | 0.945 |
| 3B | 0.014 | 0.945 |
| 3B | 0.013 | 0.945 |
| 3B | 0.012 | 0.945 |
| 3B | 0.011 | 0.945 |
| 3B | 0.010 | 0.945 |
| 3B | - | - |
</details>

Figure 4: Scaling experiments. Depth accuracy $( \delta _ { 1 } , \uparrow _ { : }$ , bottom) and AbsRel (↓, top) by T2I model size. Each line represents a T2I pre-training dataset size (none, 128M, 640M, 1.92B). Training larger T2I models on more image data yields better depth performance.

## 4 Results

We evaluate Modality Forcing across joint and conditional RGB-Depth tasks. First, we train a suite of T2I models from scratch to study how depth generation scales with T2I model and training dataset size. Next, we apply Modality Forcing to FLUX.2-klein-9B, and benchmark it against the best specialist models. Our experiments show that depth quality scales reliably with T2I model size and data, and that Modality Forcing is competitive with the best depth models.

## 4.1 Scaling experiment.

We conduct a controlled scaling experiment to answer: does depth performance scale with T2I capability? We train three T2I models on a large and diverse set of web images at 256×256 resolution and posttrain with Modality Forcing. We evaluate depth performance across model and training dataset sizes. There should be improvements across both axes if our hypothesis is true.

Each T2I model is a DiT trained with a flow-matching objective. For each training pair $\displaystyle ( \mathbf { x } , \mathbf { c } )$ , we encode the image into a clean latent $\mathbf { z } _ { 0 }$ with the frozen FLUX.2 VAE [18]. We sample a timestep $t \in [ 0 , 1 ]$ and add noise to the latent along the linear interpolant ${ \bf z } _ { t } = ( 1 - t ) { \bf z } _ { 0 } + t { \bf \epsilon }$ with $\mathbf { \epsilon } \epsilon \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ . The network output $g _ { \theta } ( \mathbf { z } _ { t } , t , \mathbf { c } )$ is parameterized as a clean-latent prediction $\hat { \mathbf { z } } _ { 0 } ,$ , and we regress the implied velocity to the true velocity $\mathbf { v } = \mathbf { z } _ { 0 } - \epsilon \mathbf { : }$ :

$$
\mathcal {L} _ {\mathrm{T2I}} = \left\| \left(g _ {\theta} (\mathbf {z} _ {t}, t, \mathbf {c}) - \mathbf {z} _ {t}\right) / t - \mathbf {v} \right\| _ {2} ^ {2}. \tag {9}
$$

The resulting checkpoints are the starting points for Modality Forcing post-training.

We vary model size from 370M to 3.3B (Table 2), and T2I pre-training data size from zero to 1.92B samples. For depth, we use ≈17M training frames, a combination of real and synthetic images (Table 1), and evaluate depth across NYUv2 [29], DIODE [35], ETH3D [28] and ScanNet [9].

Figure 4 shows depth performance across model and pre-training data size. The results suggest that the performance of Modality Forcing scales with T2I model size and training data. Larger T2I models and larger training data result in improved depth quality. This suggests Modality Forcing is a scalable recipe for depth generation. The results show that T2I pre-training and not just model size enables better depth quality, presenting direct evidence of the spatial prior present in T2I models.

![](images/5c38dce5803ca61f255fcf9f0b141c23246b95f3b2c796e79de31f05086c7ce0.jpg)  
Figure 5: Qualitative image-to-depth generation results. Modality Forcing generates robust and sharp depth maps yielding plausible 3D point clouds. We find JointDiT [6] sometimes fails catastrophically, missing structure or misestimating scale. PPD [42] produces much closer results, but Modality Forcing produces more robust results.

Implementation Details The T2I trunk is a flat-token DiT [25] with parallel attention and MLP branches per block, pre- and post-RMSNorm, grouped-query attention, and RoPE positional encoding [31]. Latents are produced by a frozen FLUX.2 VAE [18] and tokenized by a fresh patch embedding; text prompts are encoded by a frozen UMT5-XXL [8] into embeddings, and the trunk attends to these text tokens at every block. We use logit-normal sampling for RGB, and plateau logit-normal sampling for depth, both with a timestep shift [10] of $\mu = 1 . 1$ . Additionally, we set $p _ { \mathrm { i 2 d } } = 0 . 2$ and $p _ { \mathrm { d 2 i } } = 0 . 2$ , to focus the network on the case of a fully denoised image or depth map.

Captioning follows protocols similar to those in SD3 [10] and WAN [36]. Each model in the family has the same number of layers but a different token size (Table 2).

The metrics are averaged across NYUv2 [29], DIODE [35], ETH3D [28] and ScanNet [9] datasets and take $2 5 6 \times 2 5 6$ center crops. We use the robust affine-invariant alignment introduced in MoGe-2 [39].

Table 3: Affine-invariant depth estimation. Comparison of monocular depth estimation methods on five benchmarks. The results suggest Modality Forcing outperforms all models built for joint RGB-Depth generation, and generative depth estimators. Modality Forcing approaches the performance of the very best depth models such as MoGe-2 [39], scoring better on NYUv2 [29] and ETH3D [28]. The overall best are underlined, within-type best are bolded.

<table><tr><td rowspan="2">Method Type</td><td rowspan="2">Method</td><td colspan="2">NYUv2 [29]</td><td colspan="2">KITTI [13]</td><td colspan="2">ETH3D [28]</td><td colspan="2">ScanNet [9]</td><td colspan="2">DIODE [35]</td></tr><tr><td>AbsRel↓</td><td> $\delta_1 \uparrow$ </td><td>AbsRel↓</td><td> $\delta_1 \uparrow$ </td><td>AbsRel↓</td><td> $\delta_1 \uparrow$ </td><td>AbsRel↓</td><td> $\delta_1 \uparrow$ </td><td>AbsRel↓</td><td> $\delta_1 \uparrow$ </td></tr><tr><td rowspan="6">Discriminative depth estimation</td><td>ZoeDepth [3]</td><td>4.76</td><td>97.3</td><td>5.59</td><td>95.1</td><td>7.27</td><td>94.2</td><td>-</td><td>-</td><td>7.80</td><td>90.9</td></tr><tr><td>MASt3R [19]</td><td>4.67</td><td>96.7</td><td>5.79</td><td>95.1</td><td>4.64</td><td>97.0</td><td>-</td><td>-</td><td>5.79</td><td>94.1</td></tr><tr><td>DA-v2 [45]</td><td>4.16</td><td>97.9</td><td>6.77</td><td>94.3</td><td>4.63</td><td>97.2</td><td>-</td><td>-</td><td>5.41</td><td>94.6</td></tr><tr><td>Depth Pro [4]</td><td>3.67</td><td>98.2</td><td>5.12</td><td>96.8</td><td>4.97</td><td>96.4</td><td>-</td><td>-</td><td>4.66</td><td>95.6</td></tr><tr><td>UniDepth V2 [26]</td><td>2.96</td><td>98.6</td><td>3.85</td><td>98.1</td><td>2.95</td><td>98.5</td><td>-</td><td>-</td><td>4.05</td><td>96.5</td></tr><tr><td>MoGe-2 [39]</td><td>2.89</td><td>98.6</td><td>3.75</td><td>98.1</td><td>2.80</td><td>99.1</td><td>-</td><td>-</td><td>3.14</td><td>97.4</td></tr><tr><td rowspan="4">Generative depth estimation</td><td>Marigold [15]</td><td>4.88</td><td>96.8</td><td>9.05</td><td>90.3</td><td>4.90</td><td>97.2</td><td>5.89</td><td>95.2</td><td>6.13</td><td>94.5</td></tr><tr><td>Lotus [14]</td><td>4.31</td><td>97.3</td><td>8.89</td><td>90.1</td><td>5.25</td><td>96.7</td><td>5.01</td><td>96.5</td><td>6.70</td><td>93.8</td></tr><tr><td>GeoWizard [11]</td><td>4.89</td><td>96.6</td><td>10.97</td><td>86.6</td><td>6.15</td><td>95.3</td><td>5.49</td><td>95.7</td><td>6.37</td><td>94.0</td></tr><tr><td>PPD [42]</td><td>3.82</td><td>97.7</td><td>6.57</td><td>95.1</td><td>4.02</td><td>98.3</td><td>4.04</td><td>97.8</td><td>4.97</td><td>95.6</td></tr><tr><td rowspan="4">I2D from Joint Model</td><td>JointNet [46]</td><td>11.92</td><td>86.7</td><td>13.74</td><td>81.3</td><td>12.63</td><td>87.1</td><td>14.81</td><td>81.8</td><td>20.02</td><td>82.7</td></tr><tr><td>UniCon [21]</td><td>8.75</td><td>91.7</td><td>18.67</td><td>73.0</td><td>8.77</td><td>92.9</td><td>9.23</td><td>91.0</td><td>13.57</td><td>90.8</td></tr><tr><td>JointDiT [6]</td><td>5.12</td><td>96.9</td><td>10.90</td><td>88.4</td><td>5.32</td><td>97.0</td><td>5.76</td><td>96.3</td><td>10.22</td><td>93.9</td></tr><tr><td>Ours</td><td>2.52</td><td>98.9</td><td>5.37</td><td>96.6</td><td>2.37</td><td>99.3</td><td>2.32</td><td>98.9</td><td>3.35</td><td>97.7</td></tr></table>

![](images/c5d84abc30252d579d4dd43db47b8c583735b69e5db1e43d7b7daeb4b29924b7.jpg)

<details>
<summary>text_image</summary>

"A smiling woman."
"A warm, inviting kitchen."
"A Mediterranean feast."
JointDiT
Ours
Image
3D Point Cloud
Image
3D Point Cloud
Image
3D Point Cloud
</details>

Figure 6: Qualitative joint image-depth generation results. Modality Forcing samples RGB and geometry jointly, and proves to produce consistent and compelling point clouds. JointDiT [6] inherits compelling image generations from FLUX, but struggles with depth.

## 4.2 FLUX-Based Modality Forcing

After showing Modality Forcing scales with T2I capability, we now evaluate its performance when paired with a SoA T2I model. We post-train FLUX.2-klein-9B with the Modality Forcing recipe and evaluate it on conditional and joint generation tasks. We post-train this model using the real-world and synthetic depth data shown in Table 1. The training recipe is largely unchanged from Section 4.1, only now we apply the self-distillation loss to retain more of the T2I prior (Section 3.3).

Image-to-depth generation We evaluate the monocular depth estimation capability by evaluating affine-invariant depth on NYUv2 [29], DIODE [35], ETH3D [28], ScanNet [9] and KITTI [13]. We evaluate the Modality Forcing recipe against state-of-the-art depth estimation models. The comparison includes discriminative depth estimators, generative depth models, and existing joint image-depth generators. Discriminative depth estimators include the most performant models such as MoGe-2 [39] and Depth Pro [4]. We also include generative depth estimators such as pixel-perfect-depth (PPD) [42]. Finally, most comparable to Modality Forcing, we directly compare against methods which jointly generate image and depth such as JointDiT [6].

The results in Table 3 suggest that Modality Forcing convincingly outperforms existing joint imagedepth models and generative depth models. We attribute the performance delta to the scalability of our recipe to real-world depth data, the strong prior from the FLUX.2-klein-9B backbone, and our post-training recipe. Modality Forcing is competitive even to the strongest depth models such as PPD [42], Depth Anything V2 [45] and MoGe-2 [39]. Modality Forcing outperforms all prior baselines on NYUv2 [29] and ETH3D [28]. Our performance on ScanNet [9] is also strong, but confounded by the ScanNet training split in our data mixture.

Qualitatively, Figure 5 shows that the predicted depth is consistent with ground truth. We find that JointDiT [6] infers predictions that violate geometric constraints such as straight lines and vertical walls. We also observe JointDiT’s quantization artifacts for distant predictions.

Joint Generation Joint image-depth generation is a key capability enabling prompt-conditioned scene generation for artists, architects and embodied agents. Figure 6 shows a comparison between Modality Forcing and the existing SoA joint image-depth generator JointDiT [6]. JointDiT produces stunning images, but the depth predictions are not as robust as the Modality Forcing recipe.

Depth-to-Image Generation (D2I) Depth-toimage generation allows for generating various samples from a single depth map, conditioned on a text prompt. This unlocks generating various scenes with identical layout for embodied AI agents, or assets with identical geometry but different appearance. We follow the evaluation protocol from JointDiT [6] and UniCon [21] by evaluating on $6 { , } 0 0 0$ images in the OpenImages dataset with Depth Anything V2 [45] depth annotations. We conduct two experiments: (1) generate all $6 { , } 0 0 0$ images conditioned on DAv2 depth, and compute the FID vs GT images, (2) compute

Table 4: Depth-conditioned image generation on 6,000 OpenImages samples.

<table><tr><td rowspan="2">Method</td><td colspan="2">OpenImages 6K</td></tr><tr><td>FID↓</td><td>AbsRel↓</td></tr><tr><td>Readout-Guidance</td><td>18.72</td><td>23.19</td></tr><tr><td>ControlNet</td><td>13.68</td><td>9.85</td></tr><tr><td>UniCon [21]</td><td>13.21</td><td>9.26</td></tr><tr><td>JointDiT [6]</td><td>12.62</td><td>6.99</td></tr><tr><td>Modality Forcing (Ours)</td><td>11.41</td><td>9.26</td></tr></table>

AbsRel between DAv2(generated images) and DAv2(GT images), this is a proxy for depth-following.

Table 4 shows the results. Modality Forcing generates images with the lowest FID, considerably better than all the baselines. We find D2I performance does not match JointDiT [6], as our recipe appears to more loosely follow depth instructions for some scenes.

Implementation Details Unlike other T2I models, FLUX.2 uses double-stream layers (an image and text stream). We append a depth stream for those layers and freeze the RGB weights for the double stream layers. We train this model at $5 1 2 \times x$ variable aspect ratio, where $x \le 5 1 2$ .

We use logit-normal sampling for RGB, and plateau logit-normal sampling for depth, both with a timestep shift of $\mu = 1 . 1$ . Additionally, we set $p _ { \mathrm { i 2 d } } = 0 . 2$ and $p _ { \mathrm { d 2 i } } = 0 . 2$ , to focus the network on the case of a fully denoised image or depth map.

We use the robust and optimal alignment solver (ROE) [39], as well as the ensemble approach documented by Marigold [15] with $N = 1 0$ . We rerun the joint and generative models to match the protocols, we copy the discriminative results from the MoGe-2 paper as they match our protocol.

## 4.3 Denoising Trajectory Ablations

Modality Forcing allows setting arbitrary per-modality noise levels at inference time, a flexibility we previously showed enables competitive image-to-depth and depth-to-image generation. Here we leverage the same mechanism to extend our analysis to partial depth conditioning and to the model’s behavior across a range of different generation trajectories.

Partial Depth Conditioning In some creative applications, a user may want to control the conditioning strength between modalities — using a depth map to loosely fix coarse object pose, or to dictate precise geometry. Arbitrary denoising trajectories enable this partial conditioning: at every step we set

Depth Quality. Encoding per-modality timesteps independently allows for arbitrary denoising trajectories. We study how RGB and depth generations are affected by various trajectories through the noise landscape – prioritizing image or depth tokens early on in the denoising process. We follow Latent Forcing [1] by parameterizing the denoising trajectory according to $\begin{array} { r } { f _ { \alpha } ( t ) = \frac { \alpha t } { 1 + ( \alpha - 1 ) t } } \end{array}$ , with $t _ { \mathrm { d e p t h } } = f _ { \alpha } ( t _ { \mathrm { r g b } } )$ . For $\alpha > 1$ , we denoise RGB first, while $\alpha \in [ 0 , 1 )$ means depth is prioritized first. We sweep $\alpha \in [ 2 ^ { - 5 } , 2 ^ { 5 } ]$ and complete inference from the OpenImages 6k prompts. We then compute depth metrics between MoGe-2 [39] on the generated image and our own generated depth map. The results suggest that denoising RGB first yields more consistent depth. This suggests that generating natively in the latent VAE space may be an easier space to traverse than raw depth pixels, similar to what was found in Latent Forcing [1].

![](images/5cb4cc1e6058134a5809f56ff5aa2b169ac2ea7fff052fc27987235fa4515cc3.jpg)

<details>
<summary>line chart</summary>

| t_rgb | t_depth (AbsRel = 0.100) | t_depth (AbsRel = 0.125) | t_depth (AbsRel = 0.150) | t_depth (AbsRel = 0.175) | t_depth (AbsRel = 0.200) |
|-------|--------------------------|--------------------------|--------------------------|--------------------------|--------------------------|
| 1.0   | 1.0                      | 1.0                      | 1.0                      | 1.0                      | 1.0                      |
| 0.5   | ~0.8                     | ~0.9                     | ~1.0                     | ~1.1                     | ~1.2                     |
| 0.0   | ~1.0                     | ~1.0                     | ~1.0                     | ~1.0                     | ~1.0                     |
</details>

(a) Inference-time trajectories studied.

![](images/f56fe07346bd337f51a54f0b3e19682f1f44bea1b36eb766b05fff6f346e294e.jpg)

<details>
<summary>line chart</summary>

| x-axis       | AbsRel | δ₁   |
| ------------ | ------ | ---- |
| depth-first  | 0.20   | 0.75 |
| log₂ α       | 0.15   | 0.80 |
| rgb-first    | 0.10   | 0.90 |
</details>

(b) Denoising RGB first leads to better depth prediction.  
Figure 7: Modality Forcing inference-time analysis. We find that denoising RGB first acts as a kind of ‘scratch pad’ in latent space, leading to higherquality depth predictions.

$\mathbf { x } _ { \mathrm { d e p t h } } = ( 1 - t _ { \mathrm { d e p t h } } ) \mathbf { x } _ { \mathrm { c o n d } } + t _ { \mathrm { d e p t h } } \epsilon$ . Denoising depth earlier means more of RGB generation happens with clean depth, yielding stronger conditioning. Figure 8 shows that this mechanism effectively controls depth conditioning strength, and yields compelling and viable results. This simple test-time implementation adds another capability to the same model.

## 5 Conclusion and Limitations

We present Modality Forcing, a simple post-training recipe that turns a T2I model into a unified image-depth generator. The central idea is to assign each modality its own timestep and loss while sharing a single DiT backbone, which enables joint and conditional generation in any combination. Across controlled scaling experiments, the results suggest that stronger T2I pretraining transfers to stronger spatial prediction. Our strongest model based on FLUX.2-klein-9B achieves I2D performance competitive with the very best depth models. This supports the broader view that web-scale generative image models contain spatial structure that can be exposed and continues to scale.

The current model also leaves clear directions for improvement in future work. First, our scaling study is limited to at most 3B parameters and does not derive a full scaling law. Future work may explore scaling to larger T2I models and study scaling behavior in greater detail. Second, our experiments use relatively few depth samples, several orders of magnitude fewer than what is seen during T2I pretraining. Following recent work [37], we will likely see another leap in performance by scaling up depth data and the T2I backbone to > 9B parameters. Finally, architectural adjustments may further reduce artifacts or enable prediction of metric depth. With these extensions, we believe Modality Forcing provides a broad foundation for scalable generation across modalities. Applied to depth, Modality Forcing competes with the very best specialist models and is shown to continue to scale.

## 6 Acknowledgements

We thank Gengshan Yang, Katja Schwarz, Ben Mildenhall, Hao Zhang, Andy Cheng and other colleagues for productive discussions during the project and in revising the manuscript.

![](images/d64243d030260576d614e9ba516665cb4193020249fb0eb576449e1fe3702d53.jpg)

<details>
<summary>text_image</summary>

"Bernese"
"Penguin"
"VW Beetle"
"Cello"
"Sneaker"
Prompt Input Depth log₂ α = -5 log₂ α = -8 log₂ α = -11 log₂ α = -14 Pure d2i
Stronger depth conditioning
</details>

Figure 8: The denoising trajectory across depth and rgb dictates the strength of modality conditioning. Denoising more of depth early on means RGB will more rigidly match it. Here we show that this enables a new level of controllability of depth-conditioned image generation.

## References

[1] Alan Baade, Eric Ryan Chan, Kyle Sargent, Changan Chen, Justin Johnson, Ehsan Adeli, and Li Fei-Fei. Latent forcing: Reordering the diffusion trajectory for pixel-space image generation, 2026. URL https://arxiv.org/abs/2602.11401.  
[2] Jonathan T. Barron, Ben Mildenhall, Dor Verbin, Pratul P. Srinivasan, and Peter Hedman. Mip-nerf 360: Unbounded anti-aliased neural radiance fields. CVPR, 2022.  
[3] Shariq Farooq Bhat, Reiner Birkl, Diana Wofk, Peter Wonka, and Matthias Müller. Zoedepth: Zero-shot transfer by combining relative and metric depth, 2023. URL https://arxiv.org/ abs/2302.12288.  
[4] Aleksei Bochkovskii, Amaël Delaunoy, Hugo Germain, Marcel Santos, Yichao Zhou, Stephan R. Richter, and Vladlen Koltun. Depth pro: Sharp monocular metric depth in less than a second, 2025. URL https://arxiv.org/abs/2410.02073.  
[5] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya  
Sutskever, and Dario Amodei. Language models are few-shot learners. CoRR, abs/2005.14165, 2020. URL https://arxiv.org/abs/2005.14165.  
[6] Kwon Byung-Ki, Qi Dai, Lee Hyoseok, Chong Luo, and Tae-Hyun Oh. Jointdit: Enhancing rgbdepth joint modeling with diffusion transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 25261–25271, October 2025.  
[7] Boyuan Chen, Diego Martí Monsó, Yilun Du, Max Simchowitz, Russ Tedrake, and Vincent Sitzmann. Diffusion forcing: Next-token prediction meets full-sequence diffusion. Advances in Neural Information Processing Systems, 37:24081–24125, 2025.  
[8] Hyung Won Chung, Noah Constant, Xavier Garcia, Adam Roberts, Yi Tay, Sharan Narang, and Orhan Firat. Unimax: Fairer and more effective language sampling for large-scale multilingual pretraining, 2023. URL https://arxiv.org/abs/2304.09151.  
[9] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017.  
[10] Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, and Robin Rombach. Scaling rectified flow transformers for high-resolution image synthesis, 2024. URL https://arxiv.org/abs/2403.03206.  
[11] Xiao Fu, Wei Yin, Mu Hu, Kaixuan Wang, Yuexin Ma, Ping Tan, Shaojie Shen, Dahua Lin, and Xiaoxiao Long. Geowizard: Unleashing the diffusion priors for 3d geometry estimation from a single image, 2024. URL https://arxiv.org/abs/2403.12013.  
[12] Valentin Gabeur, Shangbang Long, Songyou Peng, Paul Voigtlaender, Shuyang Sun, Yanan Bao, Karen Truong, Zhicheng Wang, Wenlei Zhou, Jonathan T Barron, Kyle Genova, Nithish Kannen, Sherry Ben, Yandong Li, Mandy Guo, Suhas Yogin, Yiming Gu, Huizhong Chen, Oliver Wang, Saining Xie, Howard Zhou, Kaiming He, Thomas Funkhouser, Jean-Baptiste Alayrac, and Radu Soricut. Image generators are generalist vision learners. arXiv preprint arXiv:2604.20329, 2026.  
[13] A Geiger, P Lenz, C Stiller, and R Urtasun. Vision meets robotics: The kitti dataset. Int. J. Rob. Res., 32(11):1231–1237, September 2013. ISSN 0278-3649. doi: 10.1177/0278364913491297. URL https://doi.org/10.1177/0278364913491297.  
[14] Jing He, Haodong Li, Wei Yin, Yixun Liang, Leheng Li, Kaiqiang Zhou, Hongbo Zhang, Bingbing Liu, and Ying-Cong Chen. Lotus: Diffusion-based visual foundation model for high-quality dense prediction, 2025. URL https://arxiv.org/abs/2409.18124.  
[15] Bingxin Ke, Anton Obukhov, Shengyu Huang, Nando Metzger, Rodrigo Caye Daudt, and Konrad Schindler. Repurposing diffusion-based image generators for monocular depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.  
[16] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C. Berg, Wan-Yen Lo, Piotr Dollár, and Ross Girshick. Segment anything. arXiv:2304.02643, 2023.  
[17] Akshay Krishnan, Xinchen Yan, Vincent Casser, and Abhijit Kundu. Orchid: Image latent diffusion for joint appearance and geometry generation, 2025. URL https://arxiv.org/abs/ 2501.13087.  
[18] Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.  
[19] Vincent Leroy, Yohann Cabon, and Jérôme Revaud. Grounding image matching in 3d with mast3r, 2024. URL https://arxiv.org/abs/2406.09756.  
[20] Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise, 2026. URL https://arxiv.org/abs/2511.13720.  
[21] Xirui Li, Charles Herrmann, Kelvin CK Chan, Yinxiao Li, Deqing Sun, and Ming-Hsuan Yang. A simple approach to unifying diffusion-based conditional generation. In The Thirteenth International Conference on Learning Representations, 2025.  
[22] Zhizhong Li and Derek Hoiem. Learning without forgetting, 2017. URL https://arxiv.org/ abs/1606.09282.  
[23] Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023. URL https://arxiv.org/abs/2210.02747.  
[24] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Hervé Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision, 2024. URL https://arxiv.org/abs/2304.07193.  
[25] William Peebles and Saining Xie. Scalable diffusion models with transformers, 2023. URL https://arxiv.org/abs/2212.09748.  
[26] Luigi Piccinelli, Yung-Hsu Yang, Christos Sakaridis, Mattia Segu, Siyuan Li, Luc Van Gool, and Fisher Yu. UniDepth: Universal monocular metric depth estimation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024.  
[27] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. Highresolution image synthesis with latent diffusion models, 2022. URL https://arxiv.org/abs/ 2112.10752.  
[28] Thomas Schöps, Johannes L. Schönberger, Silvano Galliani, Torsten Sattler, Konrad Schindler, Marc Pollefeys, and Andreas Geiger. A multi-view stereo benchmark with high-resolution images and multi-camera videos. In 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2538–2547, 2017. doi: 10.1109/CVPR.2017.272.  
[29] Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In Andrew Fitzgibbon, Svetlana Lazebnik, Pietro Perona, Yoichi Sato, and Cordelia Schmid, editors, Computer Vision – ECCV 2012, pages 746–760, Berlin, Heidelberg, 2012. Springer Berlin Heidelberg. ISBN 978-3-642-33715-4.  
[30] Gabriela Ben Melech Stan, Diana Wofk, Scottie Fox, Alex Redden, Will Saxton, Jean Yu, Estelle Aflalo, Shao-Yen Tseng, Fabio Nonato, Matthias Muller, and Vasudev Lal. Ldm3d: Latent diffusion model for 3d, 2023. URL https://arxiv.org/abs/2305.10853.  
[31] Jianlin Su, Yu Lu, Shengfeng Pan, Ahmed Murtadha, Bo Wen, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding, 2023. URL https://arxiv.org/abs/ 2104.09864.  
[32] Richard Sutton. The bitter lesson, 2019. URL http://www.incompleteideas.net/IncIdeas/ BitterLesson.html.  
[33] SAM 3D Team, Xingyu Chen, Fu-Jen Chu, Pierre Gleize, Kevin J Liang, Alexander Sax, Hao Tang, Weiyao Wang, Michelle Guo, Thibaut Hardin, Xiang Li, Aohan Lin, Jiawei Liu, Ziqi Ma, Anushka Sagar, Bowen Song, Xiaodong Wang, Jianing Yang, Bowen Zhang, Piotr Dollár, Georgia Gkioxari, Matt Feiszli, and Jitendra Malik. Sam 3d: 3dfy anything in images. 2025. URL https://arxiv.org/abs/2511.16624.  
[34] Z-Image Team. Z-image: An efficient image generation foundation model with single-stream diffusion transformer. arXiv preprint arXiv:2511.22699, 2025.  
[35] Igor Vasiljevic, Nick Kolkin, Shanyi Zhang, Ruotian Luo, Haochen Wang, Falcon Z. Dai, Andrea F. Daniele, Mohammadreza Mostajabi, Steven Basart, Matthew R. Walter, and Gregory Shakhnarovich. DIODE: A Dense Indoor and Outdoor DEpth Dataset. CoRR, abs/1908.00463, 2019. URL http://arxiv.org/abs/1908.00463.  
[36] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen, Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models, 2025. URL https://arxiv.org/abs/2503.20314.  
[37] Jianyuan Wang, Minghao Chen, Shangzhan Zhang, Nikita Karaev, Johannes Schönberger, Patrick Labatut, Piotr Bojanowski, David Novotny, Andrea Vedaldi, and Christian Rupprecht. Vggt-ω, 2026. URL https://arxiv.org/abs/2605.15195.  
[38] Ruicheng Wang, Sicheng Xu, Cassie Dai, Jianfeng Xiang, Yu Deng, Xin Tong, and Jiaolong Yang. Moge: Unlocking accurate monocular geometry estimation for open-domain images with optimal training supervision. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 5261–5271, 2025.  
[39] Ruicheng Wang, Sicheng Xu, Yue Dong, Yu Deng, Jianfeng Xiang, Zelong Lv, Guangzhong Sun, Xin Tong, and Jiaolong Yang. Moge-2: Accurate monocular geometry with metric scale and sharp details, 2025. URL https://arxiv.org/abs/2507.02546.  
[40] Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, and Jerome Revaud. Dust3r: Geometric 3d vision made easy, 2024. URL https://arxiv.org/abs/2312.14132.  
[41] Ronald J. Williams and David Zipser. A learning algorithm for continually running fully recurrent neural networks. Neural Computation, 1(2):270–280, 1989. doi: 10.1162/neco.1989.1.2.270.  
[42] Gangwei Xu, Haotong Lin, Hongcheng Luo, Xianqi Wang, Jingfeng Yao, Lianghui Zhu, Yuechuan Pu, Cheng Chi, Haiyang Sun, Bing Wang, et al. Pixel-perfect depth with semantics-prompted diffusion transformers. arXiv preprint arXiv:2510.07316, 2025.  
[43] Ceyuan Yang, Zhijie Lin, Yang Zhao, Fei Xiao, Hao He, Qi Zhao, Chaorui Deng, Kunchang Li, Zihan Ding, Yuwei Guo, Fuyun Wang, Fangqi Zhu, Xiaonan Nie, Shenhan Zhu, Shanchuan Lin, Hongsheng Li, Weilin Huang, Guang Shi, and Haoqi Fan. Context unrolling in omni models, 2026. URL https://arxiv.org/abs/2604.21921.  
[44] Lihe Yang, Bingyi Kang, Zilong Huang, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024.  
[45] Lihe Yang, Bingyi Kang, Zilong Huang, Zhen Zhao, Xiaogang Xu, Jiashi Feng, and Hengshuang Zhao. Depth anything v2. arXiv:2406.09414, 2024.  
[46] Jingyang Zhang, Shiwei Li, Yuanxun Lu, Tian Fang, David McKinnon, Yanghai Tsin, Long Quan, and Yao Yao. Jointnet: Extending text-to-image diffusion for dense distribution modeling, 2023. URL https://arxiv.org/abs/2310.06347.