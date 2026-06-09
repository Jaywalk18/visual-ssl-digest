# A Mixed Diet Makes DINO An Omnivorous Vision Encoder

Rishabh Kabra1,2 Maks Ovsjanikov1 Drew A. Hudson1 Ye Xia1 Skanda Koppula1,2 Andre Araujo1 Joao Carreira1 Niloy J. Mitra2

1Google DeepMind 2University College London

{rkabra,movsani,dorarad,yexia,skandak,andrearaujo,joaoluis}@google.com n.mitra@ucl.ac.uk

## Abstract

Pre-trained vision encoders like DINOv2 have demonstrated exceptional performance on unimodal tasks. However, we observe that their features are poorly aligned across different visual modalities. For instance, the feature embedding for an RGB image and its corresponding depth map of the same scene exhibit a cosine similarity that is nearly identical to that of two random, unrelated images. To address this, we propose the Omnivorous Vision Encoder, a post-training framework that learns a modality-agnostic feature space. We fine-tune the encoder with a dual objective: first, to maximize the feature alignment between different modalities of the same scene; and second, a distillation objective that anchors the learned representations to a fully frozen teacher. The resulting student encoder becomes “omnivorous” by producing more consistent embeddings for a given scene, regardless of the input modality (RGB, Depth, Segmentation, etc.). This approach enables robust cross-modal understanding while retaining the discriminative semantics of the original foundation model. Omnivorous model weights are available at https://github. com/google-deepmind/representations4d.

## 1. Introduction

Human perception exhibits remarkable stability: whether we view a scene in daylight, shadow, or through glasses, our internal representation of the scene remains largely invariant [25, 50]. Ideally, a computer vision foundation model should possess this same “omnivorous” quality—mapping different modal views of the same scene (RGB, Depth, Segmentation) to almost identical points in its feature space.

Our empirical analysis, however, reveals that popular off-the-shelf encoders fall short of this. We find that for leading models such as DINOv2 [36], feature maps for paired RGB, Depth, and Segmentation images are not wellaligned. Specifically, the cosine similarity between the features of an RGB image $\left( x _ { r } \right)$ and its corresponding depth map $( x _ { d } )$ is surprisingly low, often comparable to the similarity between unrelated scenes: $\cos ( f ( x _ { r } ) , f ( x _ { d } ) )$ ≈ cos $\left( f ( x _ { r , 1 } ) , f ( x _ { r , 2 } ) \right)$ ), where f is the pretrained encoder.

We draw inspiration from the evolution of Natural Language Processing. Early NLP systems were languagespecific [42]. Later, it was demonstrated that aligning representations across languages [2, 23], or training shared multilingual encoders [5, 43], significantly improved generalization, particularly for low-resource languages. We argue that vision models face a similar inflection point (see Figure 1). By aligning abundant modalities (RGB) with structure-rich but scarcer signals (depth, segmentation), we can create a more robust, shared visual language.

Constructing this shared space presents a challenge. A trivial solution could simply collapse the feature space to achieve alignment, destroying the discriminative power of the encoder. Established methods like Contrastive Multiview Coding (CMC) [44] prevent collapse by pushing features apart when they are from different scenes. CMC relies on large sets of “negative” examples collected across datasets to ensure sample diversity, but these are typically limited to particular modalities (such as RGB) that are overrepresented.

To align extremely unbalanced modalities, while ensuring strong discriminative power, we propose a recipe that distills cross-modal alignment into an existing foundation model. Our approach preserves the encoder’s rich pretrained priors, reduces the need to collect negative examples across sparse modalities, and is lightweight. We adopt a parameter-efficient teacher-student framework, with the “student” encoder initialized from the pretrained foundation model, and update only the final high-level processing blocks to align representations across visual modalities. We also introduce an anchoring loss to preserve the expressivity of the original feature space.

We couple this architectural recipe with two data-centric contributions designed to further discourage trivial alignment solutions. As context, modalities such as depth and segmentation can be represented in many different ways as images, for example through colormap choices. First, we observe that standard colormaps (e.g., grayscale or jet colormaps) allow models to shortcut alignment by relying on low-level channel statistics. To counter this, we colorize depth and segmentation maps using a natural color palette derived from the corresponding RGB image. This creates “hard positives,” making the contrastive task as hard as possible by forcing the network to align features based on structural content rather than superficial signals such as color histograms. Second, we introduce a modality blending strategy. Rather than treating modalities as discrete states, we randomly blend RGB, depth, and segmentation images during training. This encourages the student to learn a degree of invariance across a continuous space of modalities, resulting in an “omnivorous” encoder that remains robust even when visual inputs are ambiguous.

![](images/eb2b100ab18099c32499cda0721b2350b7150e3ea9113ad3d8227724a501043a.jpg)

<details>
<summary>bar chart</summary>

| Method | Random RGB pairs | RGB and depth map | RGB and B&W |
| :--- | :--- | :--- | :--- |
| Patchwise cosine similarities | 0.24 | 0.26 | 0.86 |
</details>

Figure 1. Off-the-shelf vision encoders like DINO show poor cross-modal alignment. We show the similarity in feature space between randomly paired RGB images (top), between RGB images and depth maps of the same scene (middle), and between RGB and grayscale images of the same scene (bottom). While the numbers vary depending on the dataset, the pattern of misalignment between visual modalities remains consistent. Our proposed adapter aligns these modalities in an existing feature space.

## 2. Related Work

Unified encoders across visual modalities. A line of research seeks a single backbone that natively handles multiple visual modalities. Omnivore [13] trains one ViT to classify images, videos, and single-view 3D (RGB/depthlike inputs) with shared parameters, reporting benefits from joint training across modalities and an “omnivorous” design that reduces modality-specific heads. Beyond purely visual streams, ImageBind [14] learns a joint embedding that binds six modalities—image, text, audio, depth, thermal, and IMU—using only image-paired data, showing emergent alignment across unpaired modalities. Generalist architectures such as Uni-Perceiver [26, 55] unify many vision and language tasks with a single encoder–decoder interface, while Perceiver [19] and Perceiver IO [20] offer latent-bottlenecked Transformers designed to ingest heterogeneous inputs and emit structured outputs without modality-specific components. Autoregressive “all-in-one” systems like Unified-IO [29, 30] extend to diverse modalities (RGB, depth, segmentation masks, language), demonstrating broad task coverage under a common tokenization of inputs and outputs. These works motivate learning a shared space, but they typically co-train the backbone. By contrast, our approach targets alignment by fine-tuning a few layers on top of a frozen unimodal backbone.

Aligning RGB, depth, and 3D representations. Numerous papers study RGB–depth (and 2D–3D) alignment during pretraining. CLIP2Point [18] transfers CLIP knowledge to 3D by image–depth contrastive pretraining, providing a template for cross-modal InfoNCE on paired RGB/depth renders. CoMAE [48] proposes a single-model hybrid scheme that first learns cross-modal alignment contrastively and then injects masked-autoencoding objectives, explicitly targeting RGB–depth representation sharing on SUN RGB-D [41] and NYUv2 [34]. Mask3D [17] uses masked RGB-D pretraining to reconstruct depth and thereby embed 3D priors into a 2D backbone, an auxiliary signal that improves geometry awareness without labels. From a diagnostic perspective, Li and Heizmann [27] provides a unified framework comparing perspective-, modality-, and format-invariance, and empirically studies which crossformat pairs matter most. More recent works explore progressive multimodal pretraining (e.g., contrastive then masked-autoencoding) [21] and spatial-aware [3] multiscale contrastive losses for RGB-D dense prediction, reinforcing the value of explicit cross-modal objectives.

Adapters and parameter-efficient alignment. Rather than retraining large backbones, adapter methods add small trainable modules. ViT-Adapter [4] injects task/structure priors for dense prediction while keeping the ViT largely frozen, offering a strong blueprint for projector-style modules. For explicit cross-modal alignment with frozen encoders, MA-AVT [31] introduces blockwise contrastive alignment across audio-visual tokens in a parameter-efficient manner; its mechanics (blockwise objectives, shared/frozen trunk) inform projector designs that align modalities post-hoc. Recent “modality-disentangle adapters” [52] separate modality-invariant from modalityspecific components—useful when wanting both a unified embedding and optional modality-specific residuals.

Cross-modal distillation and source-free transfer. When some modalities are absent at test time, cross-modal knowledge distillation (CMKD) transfers supervision between modalities. SOCKET [1] performs source-free cross-modal transfer (e.g., RGB → depth/IR) without access to task-relevant source data, bridging modality gaps via paired task-irrelevant data and BN statistic matching. Newer CMKD variants [12] for RGB-D semantic segmentation incorporate disentanglement and contrastive terms to structure the internal spaces of single-modality students, offering alternative formulations to projector-based alignment. These methods underscore the value of contrastive/consistency losses across modalities and provide useful evaluation protocols (e.g., RGB-only inference after multimodal training).

Our contribution. Compared to unified co-training (Omnivore, ImageBind, Unified-IO) and RGB-D pretraining schemes (CLIP2Point, CoMAE, Mask3D), our method targets a pragmatic regime: post-hoc alignment of heterogeneous modalities by learning a single lightweight projector g on top of a fixed foundational backbone $f ^ { * }$ . We use a loss that directly maximizes cross-modal agreement while preserving scene-level discrimination. This design maintains the deployment benefits of strong unimodal encoders (e.g., $\mathrm { D I N O v } 2 ) ,$ delivering an “omnivorous” embedding at inference time without full-model finetuning. Our use of paired $( x ^ { M _ { 1 } } , x ^ { M _ { 2 } } )$ from the same scene and cosine/contrastive objectives follows established multimodal contrastive practice; TupleInfoNCE [28] also motivates constructing “hard” negatives by composing mismatched tuples.

## 3. Method

Our goal is to learn a unified mapping from arbitrary visual modalities to a shared embedding space. We aim to achieve not only modality-invariant representations, but also a modality-agnostic encoder that utilizes a single set of shared parameters for all inputs.

## 3.1. Architecture

We adopt a parameter-efficient teacher-student framework. We initialize a “student” encoder from the pre-trained foundation model. To balance stability and plasticity, the student shares the vast majority of its layers (the frozen backbone $f ^ { * } )$ with the teacher, updating only the final high-level processing blocks (the head $g )$ . The teacher’s head $( g ^ { \ast } )$ remains frozen to serve as a stable anchor. By distilling knowledge from the teacher $( f _ { T } = g ^ { * } \circ f ^ { * } )$ into the student $( f _ { S } = g \circ f ^ { * } )$ while simultaneously maximizing crossmodal alignment, we prevent catastrophic forgetting. We will refer to $g$ as the “adapter” module to disambiguate it from task-specific “heads” trained later in our experiments. The architecture is depicted in Figure 2.

Let a scene be represented by a set of multimodal images $\{ x _ { m } | m \in M \}$ , where M is the set of modalities $( \mathbf { e } . \mathbf { g } . , M =$ {RGB, Depth, Seg}). For every input $x _ { m } ,$ we compute two representations:

(i) Teacher Output: $h _ { m } ^ { * } = f _ { T } ( x _ { m } ) = g ^ { * } ( f ^ { * } ( x _ { m } ) )$ . This is the stable, pre-trained representation whose properties we aim to inherit.

![](images/2e17c3cc80eed5fcbd2e49b0879f169e075cf8b7400012c78340c28fa9812905.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Input Image"] --> B["RGB"]
  B --> C["Segmentation"]
  D["Input Image"] --> B
  E["Input Image"] --> B
  F["Input Image"] --> B
  G["Input Image"] --> B
  H["Input Image"] --> B
  I["Input Image"] --> B
  J["Frozen Backbone"] --> K["f*"]
  K --> L["z_D"]
  M["f*"] --> N["z_R"]
  O["f*"] --> P["z_S"]
  Q["Learnt Adapter"] --> R["g"]
  Q --> S["g"]
  Q --> T["h"]
```
</details>

Figure 2. Omnivorous Vision Encoder architecture. A frozen encoder $f ^ { * }$ extracts features $z _ { m } = f ^ { * } ( x _ { m } )$ from a spectrum of modalities denoted m (Segmentation, RGB, Depth). A trainable modality-agnostic adapter g maps these features into a common, aligned embedding space, producing a modality-invariant representation $h = g ( z _ { m } )$ . A convenient implementation of this architecture uses the early layers of a pretrained network as the frozen part $f ^ { * }$ , and the later layers as the adapter g.

(ii) Student Output: $h _ { m } = f _ { S } ( x _ { m } ) = g ( f ^ { * } ( x _ { m } ) )$ . This is the adapted representation we aim to align across modalities.

Both $h _ { m } ^ { * }$ and $h _ { m }$ are $L _ { 2 }$ normalized. Since our implementation distills from DINOv2, the network is a Vision Transformer [9], comprising 12 blocks in the Base model. Unless mentioned otherwise, we freeze the first $L = 8$ blocks and fine-tune the subsequent 4 for the student model.

## 3.2. Data

Our data pipeline consists of three processing steps:

1. Photometric augmentation (training). We first apply standard brightness, contrast, hue, and saturation augmentations to the RGB image of a scene.

2. Colorization (training and eval). For a given (photometrically augmented) RGB image $x _ { r } ^ { a u g }$ , we quantize its pixel values into 64 bins. These can then be used to colorize the corresponding segmentation or depth map, so the colorized maps $x _ { s }$ and $x _ { d }$ resemble the RGB image.

3. Modality mixup (training) [51]. We derive an aug-$x _ { s } ^ { m i x u p } : = ( 1 - \alpha _ { s } ) x _ { s } +$ $\alpha _ { s } x _ { r } ^ { a u g }$ , and augmented depth image $x _ { d } ^ { m i x u p } : = ( 1 ~ -$ xmixup := (1 − $\alpha _ { d } ) x _ { d } + \alpha _ { d } x _ { r } ^ { a u g }$ . The blending parameters $\alpha _ { s }$ and $\alpha _ { d }$ are stochastically sampled, independently of each other, per datapoint.

Theoretically, the space of mixed-up segmentations

$M _ { s } : = \{ x _ { s } ^ { m i x u p } | ( x _ { s } , x _ { r } ) \in \chi , \alpha _ { s } \in [ 0 , 1 ] \}$ and the space of mixed-up depth images $M _ { d } \ : = \ \{ x _ { d } ^ { m i x u p } | ( x _ { d } , x _ { r } ) \in $ xmixupd |(xd, xr) ∈ $\chi , \alpha _ { d } \in [ 0 , 1 ] \}$ together span a continuous space of modalities, loosely: Depth ↔ RGB ↔ Segmentation. In practice, we restrict the range of both α’s to [0, 0.5] while training to prevent depth and segmentation images from looking too similar to the RGB image. We ablate the choice of $\alpha _ { m a x } = 0 . 5$ in Sec 4.4. For evaluation (e.g., inter-modal retrieval), we set the α’s to 0.

Figure 3 illustrates our training data spanning six datasets (detailed further in Appendix 6).

## 3.3. Loss

Symmetric Cross-Modal Alignment To create a unified representation space, we employ a symmetric alignment strategy. We aim for the student embeddings from the same scene but different modalities to be close, while embeddings from different scenes should be distinct.

We use the InfoNCE (Information Noise-Contrastive Estimation) loss [35]. Given a batch of N scenes, we define positive pairs as the student embeddings of two different $i , ( h _ { m _ { 1 } } ^ { ( i ) } , h _ { m _ { 2 } } ^ { ( i ) } )$ $( h _ { m _ { 1 } } ^ { ( i ) } , h _ { m _ { 2 } } ^ { ( j ) } )$ where $i \neq j$ . The loss for a specific pair of modalities $( m _ { 1 } , m _ { 2 } )$ is:

$$
\begin{array}{l} \mathcal {L} _ {\text { InfoNCE }} (m _ {1}, m _ {2}) = \\ - \frac {1}{N} \sum_ {i = 1} ^ {N} \log \frac {\exp_ {\tau} (\text { sim } (h _ {m _ {1}} ^ {(i)} , h _ {m _ {2}} ^ {(i)}))}{\sum_ {j = 1} ^ {N} \exp_ {\tau} (\text { sim } (h _ {m _ {1}} ^ {(i)} , h _ {m _ {2}} ^ {(j)}))} \tag {1} \\ \end{array}
$$

Here sim(·, ·) denotes the cosine similarity, $\exp _ { \tau } ( x ) =$ $\exp ( x / \tau )$ , and τ is a learned temperature parameter (clipped to [0., 100.]). The total alignment loss, $\mathcal { L } _ { \mathrm { a l i g n } }$ , is the average of the symmetric InfoNCE losses computed over all modality pairs in the adapted space, i.e.:

$$
\mathcal {L} _ {\text { align }} = \frac {1}{3} \sum_ {k _ {1} = 1} ^ {3} \sum_ {k _ {2} > k _ {1}} ^ {3} \mathcal {L} _ {\text { InfoNCE }} (m _ {k _ {1}}, m _ {k _ {2}}) \tag {2}
$$

The three choices of pairs of modalities $( m _ { k _ { 1 } } , m _ { k _ { 2 } } )$ lead to the following pairs of augmented features: $( h _ { r } ^ { a u g } , h _ { s } ^ { m i x u p } ) , ( h _ { s } ^ { m i x u p } , h _ { d } ^ { m i x u p } )$ $( h _ { d } ^ { m i x u p } , h _ { r } ^ { a u g } )$ . This symmetric approach avoids the conflicting optimization targets inherent in aligning adapted features to potentially misaligned frozen features.

Anchoring Loss While $\mathcal { L } _ { \mathrm { a l i g n } }$ brings modalities together, it can lead to “representational drift” or collapse. The adapter might learn a trivial solution that satisfies alignment but discards the rich semantic information captured by the frozen backbone $f ^ { * }$ . To mitigate this, we introduce an anchoring loss, ${ \mathcal { L } } _ { \mathrm { a n c h o r } } .$ . This loss acts as a distillation mechanism, encouraging the student’s output $h _ { m }$ to remain close to the teacher’s output $h _ { m } ^ { * }$ of the same modality. We use the cosine distance for this objective:

![](images/9115308c4595beceabe53c60c418a2810daf433467d4f1665f4a77a8234abde4.jpg)

<details>
<summary>natural_image</summary>

Grid of 20 grayscale images showing various human and outdoor scenes with no visible text or symbols
</details>

Figure 3. Training data: depth and segmentation maps are first colorized using a natural color palette derived from the corresponding RGB image. We then apply a data augmentation: we blend the colorized depth image with up to 50% of the RGB image (and likewise for the segmentation image). The compositing alpha is randomly sampled (between 0% to 50%) for each datapoint. The idea is to interpolate between the modalities (Depth ↔ RGB ↔ Seg) smoothly and teach the model a degree of invariance across the full spectrum, while also providing more negative examples for between-scene contrastive learning. Other potential benefits: the augmentation (i) makes our representations naturally invariant to scene lighting; and (ii) helps us cope with imperfect depth and segmentation values.

$$
\mathcal {L} _ {\text { anchor }} = \frac {1}{| M |} \sum_ {m \in M} (1 - \mathrm{sim} (h _ {m}, h _ {m} ^ {*})) \tag {3}
$$

By anchoring $h _ { m }$ to the stable, pre-trained space of $h _ { m } ^ { * }$ we preserve the discriminative power of the original representation.

Total Objective and Implementation. The final training objective is a weighted sum of the two losses:

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\text { align }} + \lambda_ {\text { anchor }} \mathcal {L} _ {\text { anchor }}. \tag {4}
$$

The hyperparameter $\lambda _ { \mathrm { a n c h o r } }$ balances the trade-off between achieving cross-modal alignment and preserving the semantics of the input modality. A higher $\lambda _ { \mathrm { a n c h o r } }$ emphasizes fidelity to the teacher’s semantics, while a lower value prioritizes alignment. A non-zero $\lambda _ { \mathrm { a n c h o r } }$ is crucial when using symmetric alignment to prevent degenerate solutions. We use a default value of $\lambda _ { a n c h o r } = 1 0$ .

We compute the losses separately for the class token and the dense tokens output by the network. In the latter case, we subsample 64 dense tokens for each image before computing the loss. We use a mask to ensure we do not use intra-image dense tokens as negative examples for $\mathcal { L } _ { \mathrm { { I n f o N C E } } }$ .

## 4. Experiments

We evaluate a single Omnivorous checkpoint versus DI-NOv2 across the following settings: in Section 4.1 we assess retrieval between modalities without any additional training. In Section 4.2, we train linear and non-linear heads to evaluate on downstream tasks (classification, monocular depth prediction, and segmentation) on novel datasets. In Section 4.3, we train a depth prediction head from RGB images, then switch up the input modality beyond the training distribution. Finally, in Section 4.4 we present a set of ablations for our training pipeline.

## 4.1. Inter-Modal Retrieval

To assess the alignment of our features, we perform crossmodal retrieval evaluations. This task measures the ability to retrieve the correct scene in a target modality (e.g., Depth) given a query in a source modality (e.g., RGB).

Evaluation Protocol. We extract features for all scenes in the test sets of MOVi [15], ScanNet [7], and TartanAir [46]. We consider three modalities: RGB, Depth, and Segmentation. For every scene, we extract features using both the standard [CLS] token and Global Average Pooling (GAP) of the dense feature map. All feature vectors are $L _ { 2 }$ normalized. We compute the pairwise cosine similarity between the query and gallery sets. We report standard information retrieval metrics: Recall at k (R@k for k = {1, 5}), Mean Average Precision (mAP), and Median Rank (MedR).

To capture the holism of the shared space, the results reported in Table 1 are averaged over all 6 unique directed modality pairs (RGB→Depth, Depth→RGB, RGB→Seg, Seg→RGB, Depth→Seg, Seg→Depth).

Results. Table 1 compares our Omnivorous encoder against the frozen DINOv2 baseline. The baseline exhibits significant misalignment across modalities. On ScanNet, the DI-NOv2 features yield a Median Rank of 401.8 (GAP) and 382.5 (TOK), indicating that the embeddings for different views of the same scene are far apart in the latent space.

In contrast, the Omnivorous adapter significantly improves alignment without requiring fine-tuning of the backbone. On ScanNet (GAP), our method improves R@1 from 4.6% to 46.1% and reduces the Median Rank to 2.0. On the synthetic datasets (MOVi and TartanAir), where domain gaps are smaller, the alignment is near-perfect. For example, on MOVi, we achieve an R@1 of 86.2% compared to the baseline’s 15.5%.

## 4.2. Cross-Dataset and Cross-Task Transfer

We run a suite of downstream evaluations to assess whether the Omnivorous encoder successfully aligns modalities without compromising the semantic power of the underlying foundation model. Following the protocols established in DINOv2 and Probe3D [10], we evaluate on monocular depth estimation, semantic segmentation, and classification. Further results on normals estimation and 3D correspondence are in Appendix 7.3.

Monocular Depth Estimation. We evaluate geometric awareness by training lightweight decoders on top of the now-frozen student network. We report results on NYUv2 [34] and NAVI [22] in Table 2. When using a simple Linear readout, the Omnivorous encoder outperforms DINOv2, reducing the RMSE from 0.405 to 0.377, and improving the $\delta _ { 1 }$ accuracy (percentage of correctly predicted depth pixels) from 0.875 to 0.896. With the more expressive DPT decoder, performance remains at parity with the strong DI-NOv2 baseline (0.297 RMSE), confirming that our adapter preserves the fine-grained geometric information necessary for dense prediction.

Semantic Segmentation. To verify the utility of our aligned representations for dense semantic tasks, we evaluate on ADE20k [54], Cityscapes [6], and Pascal VOC [11] (Table 2). Our method achieves competitive performance, often surpassing the unimodal baseline. Notably, on ADE20k with a Linear readout, we improve the mIoU from 0.463 to 0.475. Similarly, on Cityscapes (Linear), we observe a gain from 0.622 to 0.632. These results demonstrate that enforcing alignment between RGB, depth, and segmentation maps does not degrade the high-level semantic understanding required for segmentation tasks; in fact, the multimodal regularization appears to offer slight benefits in generalization.

Classification. We first assess the linear separability of our representations by training a linear classifier on top of the frozen backbone for ImageNet-1k (Table 3). Our Omnivorous encoder demonstrates a substantial improvement over DINOv2 (top-1 accuracy of 83.8% compared to 80.4%). This marked improvement suggests that aligning structural modalities (depth, segmentation) with RGB enriches the semantic density of the shared feature space, making it significantly more discriminative for standard classification.

We further examine k-Nearest Neighbor (k-NN) classification to ensure the anchoring loss $\mathcal { L } _ { a n c h o r }$ effectively mitigated representational drift (Table 4). On ImageNet (soft voting), k-NN performance remains effectively at parity with the teacher (81.97% vs 81.94%), confirming that our student encoder has not forgotten the original pre-training. The results on downstream transfer datasets are mixed: we observe notable gains on RP2K (+3.65%), suggesting improved robustness for object-centric tasks. However, we note slight regressions on fine-grained datasets like iNaturalist and Google Landmarks v2. This could be explained by our training mix, which includes a significant amount of simulated multi-object data. We conclude that while “omnivorous” alignment generally preserves semantics, the choice of training data nevertheless matters.

## 4.3. Zero-Shot Cross-Modal Transfer

A key promise of a unified feature space is the ability to train a task head on one modality and deploy it on another without retraining. To test this, we train a depth prediction head (Linear or DPT) on the NYUv2 dataset using only RGB images as input. We then evaluate this head on the PACE [49] dataset, but we switch the input modality to Segmentation maps (which are within our Omnivorous backbone’s training distribution) and NOCS maps (which are out-of-distribution for both backbones).

Table 1. Cross-modal retrieval: average results across all 6 directed modality pairs. We sample 1 frame per test video, yielding N queries and targets per dataset. GAP: Global Average Pooling of dense features. TOK: CLS Token embedding.

<table><tr><td>dataset</td><td>feature type</td><td>model</td><td>R@1↑</td><td>R@5↑</td><td>mAP↑</td><td>MedR↓</td></tr><tr><td rowspan="4">movi(N=128)</td><td rowspan="2">gap</td><td>DINOv2 ViT-B/14</td><td>15.5</td><td>33.1</td><td>25.2</td><td>19.3</td></tr><tr><td>Omnivorous ViT-B/14</td><td>86.2</td><td>96.5</td><td>90.9</td><td>1.0</td></tr><tr><td rowspan="2">tok</td><td>DINOv2 ViT-B/14</td><td>18.2</td><td>34.5</td><td>27.2</td><td>16.8</td></tr><tr><td>Omnivorous ViT-B/14</td><td>76.6</td><td>92.7</td><td>83.4</td><td>1.0</td></tr><tr><td rowspan="4">scannet(N=3072)</td><td rowspan="2">gap</td><td>DINOv2 ViT-B/14</td><td>4.6</td><td>10.8</td><td>8.1</td><td>401.8</td></tr><tr><td>Omnivorous ViT-B/14</td><td>46.1</td><td>71.4</td><td>57.7</td><td>2.0</td></tr><tr><td rowspan="2">tok</td><td>DINOv2 ViT-B/14</td><td>3.9</td><td>9.0</td><td>6.9</td><td>382.5</td></tr><tr><td>Omnivorous ViT-B/14</td><td>30.2</td><td>55.8</td><td>42.2</td><td>5.3</td></tr><tr><td rowspan="4">tartanair(N=128)</td><td rowspan="2">gap</td><td>DINOv2 ViT-B/14</td><td>46.6</td><td>68.5</td><td>57.1</td><td>1.8</td></tr><tr><td>Omnivorous ViT-B/14</td><td>90.6</td><td>99.2</td><td>94.6</td><td>1.0</td></tr><tr><td rowspan="2">tok</td><td>DINOv2 ViT-B/14</td><td>43.4</td><td>66.7</td><td>54.7</td><td>2.1</td></tr><tr><td>Omnivorous ViT-B/14</td><td>84.5</td><td>98.4</td><td>90.5</td><td>1.0</td></tr></table>

Table 2. Downstream evals: monocular depth prediction and segmentation. We train either a Dense Prediction Transformer (DPT) or linear head on top of the frozen ViT backbone. Depth: The training minimizes a scale-invariant gradient loss and an edge-aware gradient loss. Evaluation is conducted on datasets like NYUv2 using standard metrics such as RMSE and threshold accuracy $( \delta _ { i } = 1 . 2 5 ^ { i } )$ . Segmentation: The decoder heads are trained for pixel-wise classification. During evaluation, we compute Mean Intersection-over-Union (mIoU) by aggregating confusion matrices across batches.

<table><tr><td rowspan="2">readout</td><td rowspan="2">dataset model</td><td colspan="2">depth delta1 ↑</td><td colspan="2">depth rmse ↓</td><td colspan="3">segmentation mean iou ↑</td></tr><tr><td>navi probe3d</td><td>nyuv2</td><td>navi probe3d</td><td>nyuv2</td><td>ade20k</td><td>cityscapes</td><td>pascal voc</td></tr><tr><td rowspan="2">Linear</td><td>DINOv2 ViT-B/14</td><td>0.697</td><td>0.875</td><td>0.076</td><td>0.405</td><td>0.463</td><td>0.622</td><td>0.814</td></tr><tr><td>Omnivorous ViT-B/14</td><td>0.706</td><td>0.896</td><td>0.074</td><td>0.377</td><td>0.475</td><td>0.632</td><td>0.826</td></tr><tr><td rowspan="2">DPT</td><td>DINOv2 ViT-B/14</td><td>0.779</td><td>0.948</td><td>0.061</td><td>0.297</td><td>0.496</td><td>0.737</td><td>0.855</td></tr><tr><td>Omnivorous ViT-B/14</td><td>0.781</td><td>0.948</td><td>0.061</td><td>0.297</td><td>0.505</td><td>0.732</td><td>0.857</td></tr></table>

Table 3. Downstream eval: linear-probe classification on ImageNet. We sweep over five learning rates, picking the best one for each row. TOK: CLS Token embedding. TOK & GAP: both the CLS embedding and Average-Pooled dense features are used.

<table><tr><td>feature type</td><td>model</td><td>accuracy ↑</td></tr><tr><td rowspan="2">tok</td><td>DINOv2 ViT-B/14</td><td>0.801</td></tr><tr><td>Omnivorous ViT-B/14</td><td>0.835</td></tr><tr><td rowspan="2">tok &amp; gap</td><td>DINOv2 ViT-B/14</td><td>0.804</td></tr><tr><td>Omnivorous ViT-B/14</td><td>0.838</td></tr></table>

As shown in Table 5, the frozen DINOv2 baseline fails catastrophically when the modality is switched. When fed Segmentation maps, the DINOv2 Linear head yields an RMSE of 1.536 (meters), which is effectively random guessing. In contrast, the Omnivorous encoder—which has mapped Segmentation inputs to the same semantic space as the RGB training data—achieves an RMSE of 0.532.

This advantage extends to unseen modalities. When testing on NOCS (Normalized Object Coordinate Space) [45], which neither model saw during training, the Omnivorous encoder still significantly outperforms the baseline (RMSE 1.075 vs 1.996). This suggests that by learning to align visual modalities, the Omnivorous encoder learns a more general representation that is more robust to modality shifts than the RGB-specific DINOv2 backbone.

## 4.4. Ablations

Loss. A key component of our method is the hyperparameter $\lambda _ { a n c h o r } .$ , which balances the symmetric alignment loss $( \mathcal { L } _ { a l i g n } )$ and the anchoring loss $( \mathcal { L } _ { a n c h o r } )$ as defined in Section 3.3. This parameter explicitly controls the trade-off between cross-modal alignment and preserving the original discriminative power of the frozen teacher’s features.

Figure 4 visualizes this frontier. The frozen DINOv2 baseline (light blue cross) exhibits high cross-scene discernibility (0.80) but suffers from poor cross-modal alignment (0.28), confirming our initial observations.

Table 4. Downstream eval: k-NN classification. On ImageNet [8], we follow the standard DINO evaluation protocol by using soft voting among the top-k neighbors (weighted by similarity) to predict classes, sweeping over multiple k values (e.g., 10, 20, 100) to report the best top-1 accuracy. On all other datasets (iNaturalist [16], SOP [40], GLDv2 [47], RP2K [38], Food2k [33]), we use universal embeddings, evaluating the “hard” k-NN accuracy by matching test query embeddings against a training index of embeddings.

<table><tr><td>model</td><td>imagenet soft</td><td>inat</td><td>sop</td><td>gldv2</td><td>rp2k</td><td>food2k</td></tr><tr><td>DINOv2 ViT-B/14</td><td>81.936</td><td>78.53</td><td>54.39</td><td>51.90</td><td>66.83</td><td>51.90</td></tr><tr><td>Omnivorous ViT-B/14</td><td>81.974</td><td>77.49</td><td>54.69</td><td>50.13</td><td>70.48</td><td>52.14</td></tr></table>

Table 5. Cross-modal transfer on the depth prediction task. Readout heads are trained on RGB images, but tested zero-shot on two novel modalities: Seg (within-distribution for Omnivorous) and NOCS images (out-of-distribution for both Omnivorous and DINO). The heads are trained on NYUv2, evaluated on PACE. We also show qualitative results for both models.

<table><tr><td>input</td><td>readout</td><td>model</td><td>delta1 ↑</td><td>rmse ↓</td></tr><tr><td rowspan="4">rgb</td><td rowspan="2">Linear</td><td>DINOv2 ViT-B/14</td><td>0.108</td><td>0.842</td></tr><tr><td>Omnivorous ViT-B/14</td><td>0.146</td><td>0.671</td></tr><tr><td rowspan="2">DPT</td><td>DINOv2 ViT-B/14</td><td>0.420</td><td>0.318</td></tr><tr><td>Omnivorous ViT-B/14</td><td>0.463</td><td>0.290</td></tr><tr><td rowspan="4">seg</td><td rowspan="2">Linear</td><td>DINOv2 ViT-B/14</td><td>0.003</td><td>1.536</td></tr><tr><td>Omnivorous ViT-B/14</td><td>0.184</td><td>0.532</td></tr><tr><td rowspan="2">DPT</td><td>DINOv2 ViT-B/14</td><td>0.042</td><td>0.792</td></tr><tr><td>Omnivorous ViT-B/14</td><td>0.169</td><td>0.507</td></tr><tr><td rowspan="4">nocs</td><td rowspan="2">Linear</td><td>DINOv2 ViT-B/14</td><td>0.001</td><td>1.996</td></tr><tr><td>Omnivorous ViT-B/14</td><td>0.023</td><td>1.075</td></tr><tr><td rowspan="2">DPT</td><td>DINOv2 ViT-B/14</td><td>0.014</td><td>0.979</td></tr><tr><td>Omnivorous ViT-B/14</td><td>0.029</td><td>0.822</td></tr></table>

![](images/028381d420c66ee5509b35890976a1f15937d491fb29796c7c454e057bfbc5c1.jpg)

Our adapted features create a clear Pareto frontier. By varying $\lambda _ { a n c h o r }$ , we can navigate this trade-off. Low values of $\lambda _ { a n c h o r }$ (e.g., 1.0) yield excellent cross-modal alignment (approaching 0.70 for dense features) but at the cost of reduced discriminative power, as the features drift significantly from the original encoder’s semantic space.

Conversely, as $\lambda _ { a n c h o r }$ increases (e.g., to 10.0 or 100.0), the anchoring loss dominates. This pulls the adapted features back towards the frozen ones, recovering most of the original cross-scene discernibility but sacrificing the alignment gains. This result confirms that $\lambda _ { a n c h o r }$ acts as a “knob” to tune the desired balance.

Training data. We ablate the mixup hyperparameter $\alpha _ { m a x }$ which controls the degree to which the modalities are blended during training (see Table 6). While depth prediction is an outlier, the performance on all other tasks continues to increase up to $\alpha _ { m a x } = 1$ , which implements full-spectrum blending of the three modalities. Our default value $\alpha _ { m a x } = 0 . 5$ was chosen to balance across the tasks.

Alternative parameterizations. See Appendix 7.3 for results on: (i) using an alternative foundation model (TIPS [32]) as the teacher network, (ii) learning an adapter on top of the teacher rather than adapting its final layers, and (iii) how many layers to freeze.

Table 6. Ablating modality mixup. We vary $\alpha _ { m a x }$ which controls the degree of blending between modalities during training. We report linear-probe performance on (i) classification (ImageNet accuracy using the TOK feature, without intermediate layers), (ii) depth prediction (δ1 on NYUv2), and (iii) segmentation (mean IoU on Cityscapes). We also report (iv) 3D correspondence (percentage of correct keypoints at threshold 0.0 on NAVI), which is assessed directly from features without a linear probe.

<table><tr><td> $\alpha_{max}$ </td><td>classif. ↑</td><td>depth ↑</td><td>segment. ↑</td><td>3D corresp. ↑</td></tr><tr><td>0</td><td>0.831</td><td>0.899</td><td>0.624</td><td>28.40</td></tr><tr><td>0.25</td><td>0.834</td><td>0.898</td><td>0.630</td><td>28.96</td></tr><tr><td>0.5</td><td>0.834</td><td>0.896</td><td>0.632</td><td>29.00</td></tr><tr><td>0.75</td><td>0.834</td><td>0.894</td><td>0.632</td><td>29.04</td></tr><tr><td>1.0</td><td>0.835</td><td>0.891</td><td>0.632</td><td>29.03</td></tr></table>

## 5. Discussion

To verify the alignment of our learned representations, we visualize the top three Principal Components of the features. Figure 5 presents two examples. The middle rows (“Frozen features”) illustrate the baseline DINOv2 output, where the feature maps for RGB, Depth, and Segmentation exhibit distinct color distributions, indicating that they occupy disjoint subspaces. In contrast, the bottom rows (“Adapted features”) demonstrate the effectiveness of our method: the feature maps for Depth and Segmentation align closely with the RGB features, sharing consistent colors and structural details. This qualitative evidence confirms that our adapter unifies the modalities into a shared semantic space without discarding spatial geometry.

![](images/70ac6997f8a4d2f7a0a8e0e76dc301ebb40258d9a2b5ecb5c14622b19da33985.jpg)

<details>
<summary>line chart</summary>

| Cross-Modal Alignment (<RGB, Depth > similarity) | Cross-Scene Discernability (1 - <RGB₁, RGB₂ > sim.) |
| ------------------------------------------------ | ----------------------------------------------- |
| 0.3                                              | 0.800                                           |
| 0.4                                              | 0.780                                           |
| 0.5                                              | 0.750                                           |
| 0.6                                              | 0.720                                           |
| 0.7                                              | 0.690                                           |
| 0.8                                              | 0.800                                           |
</details>

(a) Performance frontier (Alignment vs. Discernibility)

![](images/6dc1c5806dd738c9a3337885d1089fb0adb747101447c173356497dd7b669144.jpg)

<details>
<summary>line chart</summary>

| Depth: δ₁ accuracy | Segmentation: mean IoU |
| ------------------ | ---------------------- |
| 0.875              | 0.624                  |
| 0.880              | 0.626                  |
| 0.885              | 0.630                  |
| 0.890              | 0.633                  |
| 0.895              | 0.634                  |
| 0.900              | 0.629                  |
| 0.905              | 0.624                  |
</details>

(b) Performance frontier (Segmentation vs. Depth)  
Figure 4. Analysis of the anchoring loss. (a) Trade-off between cross-modal alignment and cross-scene discernibility, controlled by $\lambda _ { a n c h o r }$ . The x-axis measures alignment (cosine sim of <RGB, Depth>) and the y-axis measures discernibility (1 - cosine similarity of distinct RGB scenes) on ScanNet. Frozen DINOv2 (light blue) is discriminative but poorly aligned. (b) To pick a value for $\lambda _ { a n c h o r } .$ , we examine its effect on linear-head prediction performance, from Omnivorous features of RGB images, on Depth (NYUv2) and Segmentation (Cityscapes). We omit the datapoint for $\lambda _ { a n c h o r } = 0$ located at (x = 0.732, $y = 0 . 3 5 6 )$ for clarity, as it was too far below the remaining datapoints.

Future work. While in this work we focus on adapting a pre-existing feature space for cross-modality alignment, an interesting direction for exploration would be to align visual modalities while pre-training an encoder rather than post-hoc. This may unlock deeper benefits than fine-tuning the final layers of an existing model. In terms of potential downstream uses, having shown the benefits to cross-modal retrieval and depth prediction, we expect generative applications like monocular image-to-depth to benefit from conditioning on Omnivorous representations.

Limitations. DINOv2 undergoes high-resolution finetuning as a final training step. It is unclear whether this step would be required after training Omnivorous DINO too.

![](images/7852b77f5ee065654570f7fa68cd8bd9f51262f93aebf65542b79cac12434a85.jpg)  
Figure 5. PCA visualizations of frozen (DINO ViT-B/14) and adapted (Omnivorous ViT-B/14) features on two scenes.

Conclusion. Distilling from DINOv2, we have shown that our Omnivorous approach can exceed DINOv2’s performance on all 3D-relevant tasks in the Probe3D framework, semantic tasks such as classification, and cross-modal alignment. Our modality-agnostic encoder can also generalize to unseen visual modalities, paving the way for a more foundational vision model.

## Acknowledgments

We thank Kevis-Kokitsi Maninis for help with the evaluations, and Goker Erdogan for comments on the draft.

## References

[1] Faheem Ahmed et al. Cross-modal knowledge transfer without task-relevant source data. In Computer Vision – ECCV 2022, 2022. 2  
[2] Mikel Artetxe, Gorka Labaka, Eneko Agirre, and Kyunghyun Cho. Unsupervised neural machine translation. CoRR, abs/1710.11041, 2017. 1  
[3] Hao Chen, Zichao Chen, Yongliang Wu, and Hongzhuo Chen. Spatial-aware multi-modal contrastive learning for rgb-d salient object detection and beyond. Information Fusion, 124:103362, 2025. 2  
[4] Zhe Chen, Yinpeng Chen, Xiyang Dai, Mengchen Liu, Zihang Dai, Dongdong Chen, Lu Yuan, Lei Zhang, Zicheng Liu, Baining Guo, and Jingdong Wang. Vision transformer adapter for dense predictions. In Proceedings of the International Conference on Learning Representations (ICLR), 2023. 2  
[5] Alexis CONNEAU and Guillaume Lample. Cross-lingual language model pretraining. In Advances in Neural Information Processing Systems. Curran Associates, Inc., 2019. 1  
[6] Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016. 5  
[7] Angela Dai, Angel X. Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Nießner. Scannet: Richly-annotated 3d reconstructions of indoor scenes. In Proc. Computer Vision and Pattern Recognition (CVPR), IEEE, 2017. 5, 2  
[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009. 7, 3  
[9] Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020. 3  
[10] Mohamed El Banani, Amit Raj, Kevis-Kokitsi Maninis, Abhishek Kar, Yuanzhen Li, Michael Rubinstein, Deqing Sun, Leonidas Guibas, Justin Johnson, and Varun Jampani. Probing the 3d awareness of visual foundation models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21795–21806, 2024. 5, 2  
[11] Mark Everingham, Luc Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The pascal visual object classes (voc) challenge. International Journal of Computer Vision, 88(2):303–338, 2010. 5  
[12] Roger Ferrod, Cassio F. Dantas, Luigi Di Caro, and Dino ´ Ienco. Revisiting cross-modal knowledge distillation: A disentanglement approach for rgbd semantic segmentation, 2025. 3  
[13] Rohit Girdhar, Mannat Singh, Nikhila Ravi, Laurens van der Maaten, Armand Joulin, and Ishan Misra. Omnivore: A single model for many visual modalities. In Proceedings of  
the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16102–16112, 2022. 2  
[14] Rohit Girdhar, Alaaeldin El-Nouby, Zhuang Liu, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15180–15190, 2023. 2  
[15] Klaus Greff, Francois Belletti, Lucas Beyer, Carl Doersch, Yilun Du, Daniel Duckworth, David J Fleet, Dan Gnanapragasam, Florian Golemo, Charles Herrmann, et al. Kubric: A scalable dataset generator. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 3749–3761, 2022. 5, 2  
[16] Grant Van Horn, Oisin Mac Aodha, Yang Song, Yin Cui, Chen Sun, Alex Shepard, Hartwig Adam, Pietro Perona, and Serge Belongie. The inaturalist species classification and detection dataset, 2018. 7  
[17] Ji Hou, Xiaoliang Dai, Zijian He, Angela Dai, and Matthias Nießner. Mask3d: Pre-training 2d vision transformers by learning masked 3d priors. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2023. 2  
[18] Tianyu Huang, Bowen Dong, Yunhan Yang, Xiaoshui Huang, Rynson W. H. Lau, Wanli Ouyang, and Wangmeng Zuo. Clip2point: Transfer clip to point cloud classification with image-depth pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2023. 2  
[19] Andrew Jaegle, Felix Gimeno, Andy Brock, Oriol Vinyals, Andrew Zisserman, and Joao Carreira. Perceiver: General perception with iterative attention. In Proceedings of the 38th International Conference on Machine Learning, pages 4651–4664. PMLR, 2021. 2  
[20] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Maksym Andriushchenko, Sander Dieleman, and et al. Perceiver io: A general architecture for structured inputs & outputs. In Proceedings of the International Conference on Learning Representations (ICLR), 2022. 2  
[21] Muhammad Abdullah Jamal and Omid Mohareri. Multimodal contrastive masked autoencoders: A two-stage progressive pre-training approach for rgbd datasets. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pages 17947–17957, 2025. 2  
[22] Varun Jampani, Kevis-Kokitsi Maninis, Andreas Engelhardt, Arjun Karpur, Karen Truong, Kyle Sargent, Stefan Popov, Andre Araujo, Ricardo Martin-Brualla, Kaushal Patel, Daniel Vlasic, Vittorio Ferrari, Ameesh Makadia, Ce Liu, Yuanzhen Li, and Howard Zhou. NAVI: Categoryagnostic image collections with high-quality 3d shape and pose annotations. In NeurIPS, 2023. 5  
[23] Melvin Johnson, Mike Schuster, Quoc Le, Maxim Krikun, Yonghui Wu, Zhifeng Chen, Nikhil Thorat, Fernanda Viegas, ´ Martin Wattenberg, Greg Corrado, et al. Google’s multilingual neural machine translation system: Enabling zero-shot translation. Transactions of the Association for Computational Linguistics, 5:339–351, 2017. 1  
[24] Nikita Karaev, Ignacio Rocco, Benjamin Graham, Natalia Neverova, Andrea Vedaldi, and Christian Rupprecht. Dynamicstereo: Consistent dynamic depth from stereo videos. CVPR, 2023. 2  
[25] Edwin H Land. The retinex theory of color vision. Scientific american, 237(6):108–129, 1977. 1  
[26] Hao Li, Jinguo Zhu, Xiaohu Jiang, Xizhou Zhu, Hongsheng Li, Chun Yuan, Xiaohua Wang, Yu Qiao, Xiaogang Wang, Wenhai Wang, and Jifeng Dai. Uni-perceiver v2: A generalist model for large-scale vision and vision-language tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2691–2700, 2023. 2  
[27] Lanxiao Li and Michael Heizmann. A closer look at invariances in self-supervised pre-training for 3d vision. In Computer Vision – ECCV 2022, 2022. 2  
[28] Yunze Liu, Qingnan Fan, Shanghang Zhang, Hao Dong, Thomas Funkhouser, and Li Yi. Contrastive multimodal fusion with tupleinfonce. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 754–763, 2021. 3  
[29] Yuan Lu, Boyang Li, Philipp Randen, Marcella Cornia, Shiry Schiff, Ming-Wei Chang, and et al. Unified-io: A unified model for vision, language, and multi-modal tasks. In Proceedings of the International Conference on Learning Representations (ICLR), 2023. 2  
[30] Yuan Lu, Boyang Li, Philipp Randen, Marcella Cornia, Shiry Schiff, Ming-Wei Chang, and et al. Unified-io 2: Scaling autoregressive multimodal models with vision, language, audio, and action. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2024. 2  
[31] Tanvir Mahmud, Yizhe Tong, Dongliang Du, Ying Cao, Deliang Wang, and Deng Cai. Ma-avt: Modality alignment for parameter-efficient audio-visual transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), 2024. 2  
[32] Kevis-Kokitsi Maninis, Kaifeng Chen, Soham Ghosh, Arjun Karpur, Koert Chen, Ye Xia, Bingyi Cao, Daniel Salz, Guangxing Han, Jan Dlabal, Dan Gnanapragasam, Mojtaba Seyedhosseini, Howard Zhou, and Andre Araujo. TIPS: ´ Text-Image Pretraining with Spatial Awareness. In ICLR, 2025. 7  
[33] Weiqing Min, Zhiling Wang, Yuxin Liu, Mengjiang Luo, Liping Kang, Xiaoming Wei, Xiaolin Wei, and Shuqiang Jiang. Large scale visual food recognition, 2023. 7  
[34] Pushmeet Kohli Nathan Silberman, Derek Hoiem and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012. 2, 5  
[35] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018. 4  
[36] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy ´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 1  
[37] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy V. ´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. Featured Certification. 2  
[38] Jingtian Peng, Chang Xiao, and Yifan Li. Rp2k: A largescale retail product dataset for fine-grained image classification. arXiv preprint arXiv:2006.12634, 2020. 7  
[39] Mike Roberts, Jason Ramapuram, Anurag Ranjan, Atulit Kumar, Miguel Angel Bautista, Nathan Paczan, Russ Webb, and Joshua M. Susskind. Hypersim: A photorealistic synthetic dataset for holistic indoor scene understanding. In International Conference on Computer Vision (ICCV) 2021, 2021. 2  
[40] Hyun Oh Song, Yu Xiang, Stefanie Jegelka, and Silvio Savarese. Deep metric learning via lifted structured feature embedding. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. 7  
[41] Shuran Song, Samuel P Lichtenberg, and Jianxiong Xiao. Sun rgb-d: A rgb-d scene understanding benchmark suite. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 567–576, 2015. 2  
[42] Ilya Sutskever, Oriol Vinyals, and Quoc V Le. Sequence to sequence learning with neural networks. Advances in neural information processing systems, 27, 2014. 1  
[43] Gemini Team, Rohan Anil, Sebastian Borgeaud, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, Katie Millican, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023. 1  
[44] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In Computer Vision – ECCV 2020, pages 776–794, Cham, 2020. Springer International Publishing. 1  
[45] He Wang, Srinath Sridhar, Jingwei Huang, Julien Valentin, Shuran Song, and Leonidas J Guibas. Normalized object coordinate space for category-level 6d object pose and size estimation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 2642–2651, 2019. 6  
[46] Wenshan Wang, Delong Zhu, Xiangwei Wang, Yaoyu Hu, Yuheng Qiu, Chen Wang, Yafei Hu, Ashish Kapoor, and Sebastian Scherer. Tartanair: A dataset to push the limits of visual slam. In 2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), pages 4909–4916. IEEE, 2020. 5, 2  
[47] T. Weyand, A. Araujo, B. Cao, and J. Sim. Google Landmarks Dataset v2 - A Large-Scale Benchmark for Instance-Level Recognition and Retrieval. In Proc. CVPR, 2020. 7  
[48] Jiange Yang, Sheng Guo, Gangshan Wu, and Limin Wang. Comae: Single model hybrid pre-training on small-scale rgbd datasets. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI), 2023. 2  
[49] Yang You, Kai Xiong, Zhening Yang, Zhengxiang Huang, Junwei Zhou, Ruoxi Shi, Zhou Fang, Adam W Harley, Leonidas Guibas, and Cewu Lu. Pace: A large-scale dataset with pose annotations in cluttered environments. In European Conference on Computer Vision, pages 473–489. Springer, 2024. 6  
[50] Semir Zeki. Colour coding in the cerebral cortex: the reaction of cells in monkey visual cortex to wavelengths and colours. Neuroscience, 9(4):741–765, 1983. 1  
[51] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, and ´ David Lopez-Paz. mixup: Beyond empirical risk minimization. CoRR, abs/1710.09412, 2017. 3  
[52] Wen Zheng, Zhanpeng Zhang, Kai Zhang, Guangwei Zhou, Jie Yu, Yongdong Zhang, and Jian Cheng. Towards unified representation of invariant-specific features in missing modality face anti-spoofing. In Computer Vision – ECCV 2024, 2024. 2  
[53] Yang Zheng, Adam W. Harley, Bokui Shen, Gordon Wetzstein, and Leonidas J. Guibas. Pointodyssey: A large-scale synthetic dataset for long-term point tracking. In ICCV, 2023. 2  
[54] Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017. 5  
[55] Xizhou Zhu, Jinguo Zhu, Hao Li, Xiaoshi Wu, Hongsheng Li, Xiaohua Wang, and Jifeng Dai. Uni-perceiver: Pretraining unified architecture for generic perception for zeroshot and few-shot tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 16804–16815, 2022. 2

# A Mixed Diet Makes DINO An Omnivorous Vision Encoder

## Supplementary Material

This Appendix is organized into two broad sections: Section 6 describes our training and evaluation framework, while Section 7 extends the results of the main paper.

## 6. Training and Evaluation Details

We first present a summary of our training configuration in Table 7. Next, we describe our training data pipeline in Section 6.1. Finally, we elaborate on our evaluation protocols in Section 6.2.

## 6.1. Data Pipeline

Below we elaborate on all elements of the training-data processing steps previously introduced in Section 3.2.

## 6.1.1. Photometric Augmentation (RGB)

The photometric augmentation pipeline applies a sequence of standard distortions to the RGB image to encourage robustness against lighting variations and color shifts. The pipeline first adjusts brightness by adding a delta sampled from [−0.1, 0.1]. This is followed by a saturation adjustment, where the image is scaled by a factor drawn from [0.8, 1.2]. Next, the hue is shifted by a delta within the range $[ - 0 . 0 3 , 0 . 0 3 ]$ . Finally, the contrast is scaled by a factor sampled from [0.8, 1.2]. All random scalars are sampled independently for each distortion type per image instance.

## 6.1.2. Colorization (Depth & Segmentation)

Using standard colormaps (e.g., grayscale or jet) for the Depth and Segmentation images would allow the encoder to shortcut the alignment task by exploiting low-level channel statistics, thus learning modality-specific features. To counter this, we employ a natural colorization strategy.

$\Phi ( x _ { m } ^ { \mathrm { r a w } } , x _ { r } ^ { \mathrm { a u g } } )$ $x _ { m } ^ { \mathrm { r a w } }$ $x _ { r } ^ { \mathrm { a u g } } .$ This process creates “hard positives” for the contrastive objective: by forcing the structural map to share the same color histogram as the RGB image, we deny the network the ability to distinguish or align modalities based on superficial color signals. Consequently, the encoder must attend to the shared geometric content to solve the alignment task.

See Algorithm 1 for a pseudocode description of our nat-$x _ { m } ^ { \mathrm { r a w } }$ and discretize it into $B = 6 4$ intensity bins (step 1). Let $b _ { u , v } \in \{ 0 , \ldots , B - 1 \}$ denote the bin index of pixel $( u , v )$ $x _ { m } ^ { \mathrm { r a w } }$ $\mathcal { P } \in \mathbb { R } ^ { B \times 3 }$ by aggregating the RGB colors corresponding to each structural intensity bin (step 2). The accumulated color sum $\mathbf { S } _ { k }$ and pixel count ${ \bf N } _ { k }$ for bin k are computed as:

Algorithm 1 Natural Colorization  
Require: Scalar map $x_{m}^{\mathrm{raw}} \in \mathbb{R}^{H \times W}$ where $m \in \{\text{Depth, Segmentation}\}$ Require: Augmented RGB image $x_{r}^{\mathrm{aug}} \in \mathbb{R}^{H \times W \times 3}$ Require: Number of bins $B = 64$ , kernel size $K = 5$ , constant $\epsilon = 10^{-6}$ Ensure: Colorized map $x_{m} \in \mathbb{R}^{H \times W \times 3}$

1: Step 1: Normalization and Discretization
2: $x_{m}^{\text{norm}} \leftarrow \frac{x_{m}^{\text{raw}} - \min(x_{m}^{\text{raw}})}{\max(x_{m}^{\text{raw}}) - \min(x_{m}^{\text{raw}}) + \epsilon} \quad \triangleright$ Normalize modality m to [0, 1]
3: for each pixel $(u, v)$ do
4: $b_{u,v} \leftarrow \text{clip}(\lfloor x_{m}^{\text{norm}}[u, v] \cdot B \rfloor, 0, B - 1) \triangleright$ Compute bin indices
5: end for

6: Step 2: Palette Accumulation ▷ Aggregates $x_r^{\text{aug}}$ stats per bin
7: Initialize $S \in \mathbb{R}^{B \times 3}$ and $N \in \mathbb{R}^B$ with zeros
8: for each pixel $(u, v)$ do
9: $k \leftarrow b_{u,v}$ 10: $S[k] \leftarrow S[k] + x_r^{\text{aug}}[u, v]$ 11: $N[k] \leftarrow N[k] + 1$ 12: end for

13: Step 3: Palette Smoothing ▷ Fills gaps via 1D convolution
14: Define uniform kernel $w \in \mathbb{R}^K$ where $w_i = 1$ 15: $\tilde{S} \leftarrow \text{Convolve1D}(S, w)$ 16: $\tilde{N} \leftarrow \text{Convolve1D}(N, w)$

17: Step 4: Palette Normalization
18: for $k \in \{0, \ldots, B - 1\}$ do
19: $\mathcal{P}[k] \leftarrow \tilde{S}[k]/(\tilde{N}[k] + \epsilon) \triangleright$ Compute avg color per bin
20: end for

21: Step 5: Image Re-rendering
22: for each pixel $(u, v)$ do
23: $x_m[u, v] \leftarrow \mathcal{P}[b_{u,v}] \quad \triangleright$ Map bins to palette colors
24: end for
25: return $x_m$

$$
\mathbf {S} _ {k} = \sum_ {u, v} \mathbf {1} [ b _ {u, v} = k ] \cdot x _ {r} ^ {\text { aug }} (u, v), \quad \mathbf {N} _ {k} = \sum_ {u, v} \mathbf {1} [ b _ {u, v} = k ]
$$

Table 7. Training Configuration for Omnivorous DINO

<table><tr><td>Category</td><td>Details</td></tr><tr><td>Architecture</td><td>DINOv2 ViT-B/14 (173M parameters)Layers 0–7 are frozen, 8–11 are fine-tuned.</td></tr><tr><td>Optimizer</td><td>AdamW with learning rate  $1 \times 10^{-4}$ </td></tr><tr><td>Compute</td><td>TPU v4 ( $4 \times 4 \times 4$ ) for 20,000 steps, with a total runtime of 1 hour 14 minutes</td></tr><tr><td>Batch Size</td><td>512 (Global)</td></tr><tr><td>Datasets</td><td>ScanNet [7], TartanAir [46], Hypersim [39], MOVi [15], PointOdyssey [53], DynamicReplica [24]</td></tr><tr><td>Preprocessing</td><td> $224 \times 224$  resolution (RGB–bilinear resize; Depth &amp; Seg–nearest neighbor; center crop to square)Photometric Augmentation if training (RGB)Colorization (Depth and Seg)Normalization using ImageNet-1k mean and std (RGB, Depth, Seg)Modality Mixup ( $\alpha_{max} = 0.5$  if training else 0.0)</td></tr></table>

To ensure continuity, we apply a 1D smoothing convolution to S and N using a kernel of size 5 (step 3). The final palette value for bin k is $\mathcal { P } _ { k } = \tilde { \mathbf { S } } _ { k } / ( \tilde { \mathbf { N } } _ { k } + \epsilon )$ , where ϵ is set to 1e − 6 for numerical stability. The colorized map $x _ { m }$ is generated by mapping each pixel in the raw map to its corresponding palette entry: $x _ { m } ( u , v ) = \mathcal { P } _ { b _ { u , v } }$ .

## 6.1.3. Normalization (RGB, Depth, & Segmentation)

We use the ImageNet-1k mean pixel value (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225) to standardize all [0,1] images.

## 6.1.4. Modality Mixup

While natural colorization forces the encoder to focus on structure, it leaves the depth and segmentation maps stripped of textured. Due to this domain gap, the model may struggle to relate geometric shapes to rich photometric cues. To bridge the gap, we use modality mixup. By stochastically blending the colorized structural maps with the original RGB image, we span a continuous “modality spectrum” that interpolates between pure geometry (Depth/Segmentation) and pure texture (RGB). This exposes the encoder to a smooth space of inputs, encouraging it to learn representations that are invariant to the ratio of texture-to-structure, rather than overfitting to discrete modality tokens.

Let $x _ { m }$ be the naturally colorized map for modality $m \in$ {Depth, Segmentation} (from Algorithm 1) and $x _ { r } ^ { \mathrm { a u g } }$ be the photometrically augmented RGB image. We generate the final mixed input $x _ { m } ^ { \mathrm { { \bar { m i x u p } } } }$ mixup via convex combination:

$$
x _ {m} ^ {\mathrm{mixup}} = (1 - \alpha_ {m}) x _ {m} + \alpha_ {m} x _ {r} ^ {\mathrm{aug}}
$$

where the mixing coefficient $\alpha _ { m }$ is sampled uniformly from the range $[ 0 , \alpha _ { \mathrm { m a x } } ]$ independently for each training example. We set $\alpha _ { \mathrm { m a x } } = 0 . 5$ to ensure the structural signal remains dominant while re-introducing sufficient texture to facilitate alignment. This strategy effectively constructs a ”continuous bridge” between modalities, preventing the feature space from fragmenting into disjoint islands of geometry and texture.

## 6.2. Evaluation Protocols

We adopt the protocols established by DINOv2 [37] or Probe3D [10] wherever possible. We elaborate all details in the following subsections for completeness.

## 6.2.1. Cross-Modal Retrieval

For all datasets (ScanNet, MOVi, TartanAir), inputs are resized to 224×224 (using bilinear interpolation for RGB and nearest-neighbor for depth/segmentation) followed by a center crop. Single-channel structural inputs (depth and segmentation) are tiled to 3 channels and normalized using standard ImageNet statistics $( \mu ~ =$ $[ 0 . 4 8 5 , 0 . 4 5 6 , 0 . 4 0 6 ] , \sigma = [ 0 . 2 2 9 , 0 . 2 2 4 , 0 . 2 2 5 ] )$ after scaling pixel values to [0, 1]. Features are extracted using the frozen DINOv2 backbone and our adapter, applying $L _ { 2 }$ normalization to the final embeddings.

We compute pairwise cosine similarity between the query and gallery sets. To handle large-scale evaluation efficiently, similarity matrices are computed in batches of 2048. The rank for a given query is determined by counting the number of gallery items with a similarity score strictly greater than or equal to the ground-truth pair’s score (using a numerical stability threshold $\epsilon = 1 0 ^ { - 6 } )$ . As our evaluation setup assumes a strict one-to-one mapping between modalities (i.e., exactly one positive match per query), the Mean Average Precision (mAP) reported is equivalent to the Mean Reciprocal Rank (MRR). We average results over all six directed modality pairs.

## 6.2.2. Monocular Depth Estimation

Data and Preprocessing. We evaluate on NYUv2 [34] and NAVI Probe3D [10]. Unlike the classification or retrieval tasks which often resize inputs to a standard 224×224, we perform evaluation on high-resolution images to preserve geometric details (i.e., 480×640 for NYUv2, 512×512 for

NAVI). To process these variable resolutions with a ViT backbone trained on fixed patch sizes, we employ a “padto-patch” strategy: images are first center-cropped to the target resolution and then padded to the nearest multiple of the patch size (p=14). This allows the frozen backbone to process the dense grid of patches without interpolation artifacts. Standard photometric distortions and random rotations are applied during training, while horizontal flipping is used for test-time augmentation.

Decoder Architectures. We investigate the expressivity of our learned features using two distinct decoder heads.

• Linear Head: A lightweight baseline that projects the final layer’s patch tokens directly to depth bins using a single linear layer. The output is bilinearly upsampled to the input resolution. This setup tests the explicit geometric information present in the final semantic embedding.  
• DPT Head: A Dense Prediction Transformer (DPT) decoder that aggregates intermediate features from the backbone. Specifically, we gather tokens from layers 3, 6, 9, 12 (for the ViT-B/14 variant), fuse them using valid convolutions and upsampling blocks to recover highresolution details. This head evaluates the backbone’s ability to provide multi-scale hierarchical features suitable for dense prediction.

Training Objective. Both heads are trained (while keeping the backbone frozen) to classify pixels into 256 depth bins. We minimize a combined objective consisting of a Scale-Invariant Gradient Loss (sigloss) to enforce global structural consistency and an edge-aware gradient loss to sharpen local discontinuities. We train for 50,000 steps using AdamW with a compound learning rate schedule (constant, piecewise constant, and linear warmup).

## 6.2.3. Semantic Segmentation

Data and Preprocessing. We evaluate semantic segmentation on ADE20k, Cityscapes, and Pascal VOC. During training, we employ standard data augmentation techniques: input images undergo random resizing (ratio range [0.5, 2.0]), random horizontal flipping, and photometric distortion. The images are then randomly cropped to a fixed resolution of $5 1 2 \times 5 1 2$ .

Evaluation Protocol. Unlike the monocular depth evaluation which processes full images via padding, our segmentation evaluation employs a sliding window protocol to handle high-resolution inputs (e.g., Cityscapes) without downsampling artifacts. We perform inference on 512×512 crops with a stride of 341 pixels. Predictions from overlapping windows are averaged (mean logits) before the final argmax.

Decoder Architectures. We utilize the same two decoder configurations—Linear and DPT—as described in the Monocular Depth Estimation section (Appendix 6.2.2). The backbone remains frozen as before. The only modification is the final projection layer, which maps to K semantic classes (e.g., K = 150 for ADE20k) instead of depth bins.

Optimization. We train for 40,000 steps with a batch size of 16. We use the AdamW optimizer with a weight decay of $1 0 ^ { - 4 }$ . The learning rate follows a polynomial decay schedule (power 1.0) combined with a linear warmup for the first 1,500 steps. Performance is measured using the Mean Intersection-over-Union (mIoU), computed by aggregating confusion matrices over the entire validation set.

## 6.2.4. Multiview Correspondence

Data and Preprocessing. We evaluate 3D feature correspondence using the NAVI dataset. Image pairs are resized to $2 2 4 \times 2 2 4$ . We extract feature maps from the encoder, which correspond to a $1 6 \times 1 6$ grid of patches (given the patch size $p = 1 4 )$ . We do not employ a trained prediction head for this task; instead, we evaluate the raw feature representations directly.

Matching Protocol. For a given image pair, we compute the pairwise cosine similarity matrix between the flattened spatial tokens (N = 256) of the source and target views. We determine the predicted correspondence for each token by selecting the nearest neighbor (argmax of cosine similarity) in the other view. We evaluate bidirectional matches.

Metric: PCK@0. We report the Percentage of Correct Keypoints (PCK) at a strict threshold of 0.0. Since our evaluation operates on the discrete 16 × 16 token grid, a threshold of 0.0 requires the predicted token index to exactly match the ground-truth token index (i.e., the predicted patch must be the exact same patch as the ground truth). We generally report performance using the final layer’s features. That said, we also include an ablation (Table 10) measuring 3D correspondence across all fine-tuned Omnivorous ViT blocks (i.e., the last four).

## 6.2.5. Linear Probe Classification

Architecture and Training. To assess the linear separability of the learned representations, we train a linear classifier on top of the frozen backbone. We attach a single linear layer (projecting from the feature dimension D to the number of classes K = 1000) to the extracted features. The linear layer is trained to minimize the weighted softmax crossentropy loss, while the backbone remains frozen.

Evaluation Protocol. We evaluate on ImageNet-1k [8], reporting top-1 accuracy on the validation split. We employ standard data augmentation during training (random resized crops and horizontal flips), while validation images are resized to 256 pixels and center-cropped to $2 2 4 \times 2 2 4$ . We and train the prober for 10 epochs, sweeping over a range of learning rates (base values: [0.15, 0.2, 0.5, 1.0, 2.0]) along with the nesterov optimizer. We report the best accuracy achieved across the base learning rates. We report results using both the CLS token embedding and the concatenation of the CLS token and the global average pooled (GAP) features.

## 6.2.6. k-NN Classification

ImageNet. We follow the standard DINO evaluation protocol for ImageNet-1k. We extract features for the training set (index) and validation set (query) using the frozen backbone. The images are preprocessed by resizing the shorter side to 256 pixels, taking a central 224 × 224 crop, and normalizing with ImageNet statistics. We employ weighted soft voting: for each query, we compute the cosine similarity with its k nearest neighbors in the training set. These similarities are converted to weights using a softmax with temperature $\tau = 0 . 0 7$ . The class probabilities are summed across the neighbors, and the class with the highest aggregate probability is selected. We report the top-1 accuracy corresponding to the best k swept over {5, 10, 20, 50, 100}.

Transfer Datasets. For iNaturalist, SOP, Google Landmarks v2 (GLDv2), RP2K, and Food2k, we perform ”hard” k-NN classification (which is equivalent to Recall@1). We use the same image preprocessing as ImageNet (256 → 224 center crop).

• For GLDv2, we match queries from the test set against the distinct index set provided by the dataset (N ≈ 761k).  
• For iNaturalist, SOP, RP2K, and Food2k, we follow the standard metric learning protocol where the test set serves as both the query and the index. We compute the nearest neighbor for each query from the index, excluding the query itself (self-match), and check if the retrieved class label matches the query label.

## 6.2.7. Zero-Shot Modality Transfer

Protocol. To assess the universality of the learned feature space, we design a strict transfer protocol. We train a depth estimation head (either Linear or DPT as before) using only RGB images from the NYUv2 dataset. Once trained, we freeze the entire model (backbone + depth head) and evaluate it on the PACE dataset. This setup introduces a small domain shift and, crucially, a modality shift.

Modalities. We evaluate performance on three distinct input types:

• RGB: Serves as the baseline. The model encounters a domain shift (NYUv2 → PACE) but the modality remains consistent with training.  
• Segmentation: A modality seen by the Omnivorous backbone during pre-training, but never seen by the depth head. To render these inputs compatible with the frozen backbone, segmentation maps are preprocessed using our Natural Colorization scheme (Algorithm 1) to match the spectral statistics of RGB images. Unlike the backbone pre-training stage, we do not apply modality mixup during this evaluation.  
• NOCS (Normalized Object Coordinate Space): NOCS maps represent dense coordinate fields rather than pho-

tometric data. It is a modality that is completely outof-distribution; neither the Omnivorous backbone nor the depth head observes NOCS maps during training. The 3- channel coordinate maps are normalized using standard ImageNet RGB statistics before being fed into the model.

Success on NOCS and Segmentation inputs indicates that the encoder maps these diverse signals to a shared feature space that is interpretable by the RGB-trained head.

## 7. Extended Results

## 7.1. Diagnostic Metrics

Expanding on Fig 1, we report detailed cross-modal alignment and cross-scene discernibility metrics before and after Omnivorous training. Table 8 shows that our default checkpoint of Omnivorous DINO greatly improves cross-modal alignment while sacrificing some cross-scene discernibility (e.g., from 0.198 to $0 . 2 5 9 < R _ { 1 } , R _ { 2 } >$ similarity on Scan-Net). This echoes Fig 4a which showed the trade-off as a function of our $\lambda _ { a n c h o r }$ loss weight.

## 7.2. 3D Tasks

We revisit all tasks from the Probe3D framework for the Omnivorous DINO ViT-B/14 checkpoint introduced in Sec 4. We present two evaluations that were omitted in the main paper (normals estimation and multiview 3D correspondence), and add qualitative results for those already presented in the main paper (e.g., depth estimation and segmentation):

## 7.2.1. Normals Estimation

See Table 9. Omnivorous is consistently at par with DI-NOv2 across all metrics.

## 7.2.2. Multiview Correspondence

See Table 10. While our model is consistently more 3Dconsistent than the original DINOv2, the performance gap is a bit inconsistent with respect to the block where the features are taken from, i.e., there is no clear pattern of increasing/decreasing 3D-correspondence as a function of network depth. This merits future investigation.

## 7.2.3. Semantic Segmentation

See Fig 7 & 8 for a qualitative comparison between Omnivorous and DINOv2. We find that our model helps reduce over-segmentation, and is consistently more resilient to textural details in the input images.

## 7.2.4. Monocular Depth

See Fig 6 for a qualitative comparison between Omnivorous and DINOv2. As with predicted segmentations, we find that our model helps reduce high-frequency noise in the linear head’s depth predictions. Our model performs consistently better on flat surfaces, and cases where a flat object is placed on a flat surface (e.g., a painting on the wall).

![](images/8b5d9928a8d9008cac810479d8c96efe6262c656dc7a0d4e50923fa9ee15a7f7.jpg)

<details>
<summary>natural_image</summary>

Grid of 12 thermal imaging panels showing room layouts, furniture, and interior scenes with no visible text or symbols
</details>

(a) DINO ViT-B/14 depth prediction on NYUv2. Top: input images, middle: predictions, bottom: ground-truth.  
![](images/a3c8a77df6b28ee05c15012a724b746cc537208a2287fa0f2486d2d31a435fdf.jpg)

<details>
<summary>natural_image</summary>

Grid of 12 thermal imaging and spatial overlays showing room layouts, furniture, and building scenes (no text or symbols)
</details>

(b) Omnivorous DINO ViT-B/14 depth prediction on NYUv2. Top: input images, middle: predictions, bottom: ground-truth.

![](images/70b6debc5619715cef130f23e4e03d59ca16a75724a498b46209d46e7bfbdad6.jpg)

<details>
<summary>natural_image</summary>

Collage of 3D and 2D thermal imaging and classification results, featuring animals, objects, and human-like figures (no text or symbols)
</details>

(c) DINO ViT-B/14 depth prediction on NAVI Probe3D. Top: input images, middle: predictions, bottom: ground-truth.  
![](images/29831b20277bc20e0e6dfe19cfe3cc9e9dfdf32934fdf12c2e3d6214123dc8bd.jpg)

<details>
<summary>natural_image</summary>

Collage of 3D thermal or density maps showing various animal and food-related objects with no visible text or symbols
</details>

(d) Omnivorous DINO ViT-B/14 depth prediction on NAVI Probe3D. Top: input images, middle: predictions, bottom: ground-truth.  
Figure 6. Qualitative comparison (Omnivorous vs DINOv2) on depth prediction using a linear head. Please compare a versus b, and c versus d. We highlight notable differences using a black oval.

![](images/ec348a5bb6942e643af4ee375ef900478023e0caf72f83b39972f204f22fda37.jpg)

<details>
<summary>natural_image</summary>

Collage of 3D-rendered urban and rural scenes with colorful overlays, no visible text or symbols
</details>

(a) DINO ViT-B/14 segmentation prediction on ADE20k. Top: input images, middle: predictions, bottom: ground-truth.  
![](images/7bedbdb50624c3118fd5b4f77c2a73376bae55965aaecc1916f1f39833ea9522.jpg)

<details>
<summary>natural_image</summary>

Collage of 3D-rendered urban and landscape scenes with no visible text, numbers, or symbols
</details>

(b) Omnivorous DINO ViT-B/14 segmentation prediction on ADE20k. Top: input images, middle: predictions, bottom: ground-truth.  
![](images/367c6bb6b58904a9a8cb56bddcae182fff76d7c7ad6f567430a0fe7d7dc23e7f.jpg)

<details>
<summary>natural_image</summary>

Collage of various animal and regional images including animals, animals, animals in a habitat, airplane, and human figures (no text or symbols)
</details>

(c) DINO ViT-B/14 segmentation prediction on Pascal VOC. Top: input images, middle: predictions, bottom: ground-truth.  
![](images/582bde40b6339443834cb51c3cab483cf555c6ebda90ef7ae20ab359e4a08343.jpg)

<details>
<summary>natural_image</summary>

Collage of various animal and bird images including animals, animals in nature, a baby, airplane, and a red silhouette map (no text or symbols)
</details>

(d) Omnivorous DINO ViT-B/14 segmentation prediction on Pascal VOC. Top: input images, middle: predictions, bottom: ground-truth.  
Figure 7. Qualitative comparison (Omnivorous vs DINOv2) on segmentation prediction using a linear head. We highlight notable differences using a white oval.

Table 8. Diagnostic metrics: we expand Fig 1, showing cross-modal alignment and cross-scene discernibility metrics across three datasets for both pretrained DINOv2 and the adapted Omnivorous model (at our default $\lambda _ { a n c h o r } = 1 0 ) \rangle$ . We denote the three modalities R, D, and S (RGB, Depth, and Segmentation, respectively). The metrics are computed without modality-mixup $( \mathrm { i . e . , } \alpha _ { m a x } = 0 )$ . For $< R _ { 1 } , R _ { 2 } >$ , lower similarity is considered better.

<table><tr><td rowspan="2">dataset</td><td colspan="4">DINOv2 ViT-B/14</td><td colspan="4">Omnivorous ViT-B/14</td></tr><tr><td> $< R, D >$ </td><td> $< R, S >$ </td><td> $< D, S >$ </td><td> $< R_1, R_2 >$ </td><td> $< R, D >$ </td><td> $< R, S >$ </td><td> $< D, S >$ </td><td> $< R_1, R_2 >$ </td></tr><tr><td>movi</td><td>0.263</td><td>0.284</td><td>0.481</td><td>0.237</td><td>0.567</td><td>0.579</td><td>0.721</td><td>0.279</td></tr><tr><td>scannet</td><td>0.285</td><td>0.216</td><td>0.413</td><td>0.198</td><td>0.600</td><td>0.550</td><td>0.663</td><td>0.259</td></tr><tr><td>tartanair</td><td>0.345</td><td>0.359</td><td>0.543</td><td>0.172</td><td>0.607</td><td>0.603</td><td>0.736</td><td>0.223</td></tr></table>

Table 9. Downstream eval: normals estimation using a DPT head.

<table><tr><td>dataset</td><td>model</td><td>absrel ↓</td><td>diff 11.25 ↑</td><td>diff 22.50 ↑</td><td>diff 30.00 ↑</td><td>mean diff angle ↓</td><td>rmse angle ↓</td></tr><tr><td rowspan="2">navi</td><td>DINOv2 ViT-B/14</td><td>197.9</td><td>43.5</td><td>72.2</td><td>82.1</td><td>18.6</td><td>24.6</td></tr><tr><td>Omnivorous ViT-B/14</td><td>197.8</td><td>43.6</td><td>72.3</td><td>82.2</td><td>18.6</td><td>24.6</td></tr><tr><td rowspan="2">nyuv2</td><td>DINOv2 ViT-B/14</td><td>134.9</td><td>63.4</td><td>80.8</td><td>86.5</td><td>14.1</td><td>21.7</td></tr><tr><td>Omnivorous ViT-B/14</td><td>134.1</td><td>63.5</td><td>80.8</td><td>86.5</td><td>14.1</td><td>21.6</td></tr></table>

Table 10. Multiview correspondence: we report the Percentage of Correct Keypoints (↑) at the 0.0 level (i.e., only exact matches are counted). We measure correspondence for all the four blocks that are fine-tuned in the Omnivorous case, comparing them with their frozen DINOv2 counterparts.

<table><tr><td>block number model</td><td>9</td><td>10</td><td>11</td><td>12</td></tr><tr><td>DINOv2 ViT-B/14</td><td>29.76</td><td>28.49</td><td>27.68</td><td>28.57</td></tr><tr><td>Omnivorous ViT-B/14</td><td>29.76</td><td>28.93</td><td>28.63</td><td>29.00</td></tr></table>

## 7.3. Ablations

## 7.3.1. TIPS instead of DINOv2

As TIPS [32] shares the same ViT architecture as DINOv2, we can “ablate” our pretrained teacher by running Omnivorous training on TIPS instead of DINOv2. Two important distinctions are the shape of the position encoding parameter (TIPS uses $1 6 \times 1 6$ vs DINOv2’s $3 7 \times 3 7 )$ and the number of CLS tokens (TIPS uses two while DINOv2 uses one). We train Omnivorous TIPS ViT-B/14 using the default $\alpha _ { m a x } = 0 . 5$ , and freezing the first 8 blocks as we did for Omnivorous DINOv2.

Fig 9 shows that although it is harder for Omnivorous distillation to improve on the performance of TIPS (than in the case of DINOv2), $\lambda _ { a n c h o r } = 1 0 0$ nevertheless does exceed the depth and segmentation performance of higher values of $\lambda _ { a n c h o r }$ , which are anchored more strongly to the pretrained teacher. This attests to the generality of the Omnivorous framework regardless of the choice of pretrained teacher network.

## 7.3.2. Training an Adapter on Top vs. Fine-Tuning Final Blocks

We now ablate the parametrization of the student network. Rather than the default setting of fine-tuning the final blocks of a pretrained backbone, we train a zero-initialized adapter network on top of the frozen backbone. In this scenario, the student network is in fact larger than the teacher. All the teacher blocks are frozen and preserved in the student network; only the adapter blocks are trained. We use the same number of adapter blocks (four) as we fine-tuned for our default version of Omnivorous DINOv2.

We evaluate each scenario using a linear head on the final layer. We do not use a DPT head as it would require intermediate activations (typically from blocks [3, 6, 9, 12] in a 12-block ViT-B network), which cannot be consistently applied between the “adapter-on-top” and “finetune-finalblocks” settings, because the former in fact comprises 16 blocks rather than 12.

Table 11 shows comparable performance between the two settings, showing our distillation-based approach and training losses can easily be applied to alternative parametrizations of the student network.

## 7.3.3. Number of Blocks to Freeze

We assess how many ViT blocks can be inherited from the teacher network and kept frozen in Table 12. As before, we fine-tune only the final blocks of the network, keeping the preceding $L _ { \mathrm { s t o p - g r a d i e n t } }$ blocks frozen. We evaluate depth and segmentation prediction using both a DPT and linear head. Our default setting for Omnivorous ViT-B/14, $L _ { \mathrm { s t o p - g r a d i e n t } } = 8$ is chosen on this basis.

![](images/72506ae9883a94096dd5aafeba9bf3253d4d67e269a024e29e647af56ce1308b.jpg)

<details>
<summary>natural_image</summary>

Sequence of 3D street-level images showing urban street scenes, human figures, and color-coded bounding boxes (no text or symbols)
</details>

(a) DINO ViT-B/14 segmentation prediction on Cityscapes. Top: input images, middle: predictions, bottom: ground-truth.  
![](images/055e5038514896fb08c07c0d9d53f4995af9069678206ddec0c507649979cb1e.jpg)

<details>
<summary>natural_image</summary>

Sequence of 3D street scenes showing urban traffic, pedestrian activity, and human figures in various poses (no text or symbols)
</details>

(b) Omnivorous DINO ViT-B/14 segmentation prediction on Cityscapes. Top: input images, middle: predictions, bottom: ground-truth.  
Figure 8. Qualitative comparison (Omnivorous vs DINOv2) on segmentation prediction (contd.) using a linear head. We highlight notable differences using a white oval.

Table 11. Ablating the parametrization of the student: we either train a 4-block ViT on top of the DINOv2 ViT-B/14 backbone, or fine-tune the final 4 blocks of the backbone (ours). As before in Table 6, we report metrics (all ↑) on (i) classification, using either linear probes on TOK & GAP, or k-NN, (ii) depth prediction (linear head), (iii) segmentation (linear head), and (iv) multiview correspondence.

<table><tr><td rowspan="2">dataset parametrization</td><td colspan="2">Classification (acc.)</td><td colspan="2">Depth ( $\delta_1$ )</td><td colspan="3">Segmentation (mean IoU)</td><td rowspan="2">Corresp. (PCK) navi</td></tr><tr><td>inet (linear)</td><td>inet (k-NN)</td><td>navi</td><td>nyuv2</td><td>ade20k</td><td>cityscapes</td><td>pascal voc</td></tr><tr><td>Adapter on top</td><td>0.840</td><td>81.832</td><td>0.679</td><td>0.905</td><td>0.470</td><td>0.628</td><td>0.826</td><td>28.15</td></tr><tr><td>Fine-tune final blocks</td><td>0.838</td><td>81.974</td><td>0.706</td><td>0.896</td><td>0.475</td><td>0.632</td><td>0.826</td><td>29.00</td></tr></table>

Table 12. Ablating the number of blocks kept frozen, denoted by $L _ { \mathrm { s t o p - g r a d i e n t } } .$ , when training Omnivorous DINOv2. There are 12 total blocks in the ViT-B/14 architecture.

<table><tr><td rowspan="2">readout</td><td rowspan="2">dataset $L_{\text{stop-gradient}}$ </td><td colspan="2">Depth ( $\delta_1$ )</td><td colspan="3">Segmentation (mean IoU)</td></tr><tr><td>navi</td><td>nyuv2</td><td>ade20k</td><td>cityscapes</td><td>pascal voc</td></tr><tr><td rowspan="4">DPT</td><td>4</td><td>0.777</td><td>0.948</td><td>0.495</td><td>0.727</td><td>0.855</td></tr><tr><td>6</td><td>0.778</td><td>0.947</td><td>0.494</td><td>0.733</td><td>0.853</td></tr><tr><td>8</td><td>0.781</td><td>0.948</td><td>0.505</td><td>0.732</td><td>0.857</td></tr><tr><td>10</td><td>0.780</td><td>0.949</td><td>0.504</td><td>0.731</td><td>0.852</td></tr><tr><td rowspan="4">Linear</td><td>4</td><td>0.698</td><td>0.894</td><td>0.475</td><td>0.622</td><td>0.829</td></tr><tr><td>6</td><td>0.703</td><td>0.896</td><td>0.476</td><td>0.629</td><td>0.829</td></tr><tr><td>8</td><td>0.706</td><td>0.896</td><td>0.475</td><td>0.632</td><td>0.826</td></tr><tr><td>10</td><td>0.705</td><td>0.895</td><td>0.473</td><td>0.628</td><td>0.825</td></tr></table>

![](images/ac9b35a52c95c4485c27034c22f086697df0153b19b37a90f5b6655106a39b21.jpg)

<details>
<summary>line chart</summary>

| Cross-Modal Alignment (<RGB,Depth > similarity) | Cross-Scene Discernability (1 - V RGB₁, RGB₂ > sim.) |
| ------------------------------------------------ | --------------------------------------------------- |
| 0.60                                             | 0.63                                                |
| 0.63                                             | 0.62                                                |
| 0.70                                             | 0.60                                                |
| 0.77                                             | 0.57                                                |
| 0.80                                             | 0.55                                                |
| 0.83                                             | 0.54                                                |
| 0.85                                             | 0.54                                                |
</details>

(a) Performance frontier for Omnivorous TIPS (Alignment vs. Discernibility) on TartanAir. We omit the datapoint for $\lambda _ { a n c h o r } = 0 . 0$ , located at $( x = 0 . 7 8 3 , y = 0 . 8 5 9 )$ , for clarity.  
![](images/3ce92a8f7d48ea50ba5001c12b5738aa5f9ba4363033ca7106e3728385063ce1.jpg)

<details>
<summary>line chart</summary>

| Depth: δ₁ accuracy | Segmentation: mean IoU |
| ------------------ | ---------------------- |
| 0.870              | 0.550                  |
| 0.880              | 0.560                  |
| 0.885              | 0.562                  |
| 0.895              | 0.570                  |
| 0.900              | 0.575                  |
</details>

(b) Performance frontier for Omnivorous TIPS (Segmentation vs. Depth). As in Fig 4b, we use linear-head evaluation prediction performance for Depth (NYUv2) and Segmentation (Cityscapes). We omit the datapoint for $\lambda _ { a n c h o r } = 0 . 0$ , located at $( x = 0 . 7 4 5 , y =$ 0.435), for clarity.  
Figure 9. Behavior of Omnivorous TIPS.