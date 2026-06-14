# V-JEPA 2.1: Unlocking Dense Features in Video Self-Supervised Learning

Lorenzo Mur-Labadia1,2,∗, Matthew Muckley1, Amir Bar1, Mido Assran1, Koustuv Sinha1, Mike Rabbat1, Yann LeCun1, Nicolas Ballas1,†, Adrien Bardes1,†

1FAIR at Meta, 2Universidad de Zaragoza ∗Work done at Meta, †Joint last author

We present V-JEPA 2.1, a family of self-supervised models that learns dense, high-quality representations for visual scenes in both images and videos, while retaining strong global scene understanding. V-JEPA 2.1 combines four key ingredients: (i) a Dense Predictive Loss, a masking-based objective in which all tokens—visible context and masked tokens alike—contribute to the training loss, encouraging explicit spatial and temporal grounding; (ii) Deep Self-Supervision, which applies the self-supervised objective hierarchically at multiple intermediate encoder layers to improve representation quality ; (iii) Multi-Modal Tokenizers that support unified training over images and videos; and (iv) effective model and data scaling. These design choices substantially improve dense feature quality, yielding representations that are spatially structured, semantically coherent, and temporally consistent. Empirically, V-JEPA 2.1 achieves state-of-the-art results on a range of benchmarks: 7.71 mAP on Ego4D for short-term object-interaction anticipation, 40.8 Recall@5 on EPIC-KITCHENS for highlevel action anticipation, and a 20% improvement in real-robot grasping success rate over VJEPA-2 AC. The model also demonstrates state-of-art performances in robotic navigation (5.687 ATE on Tartan Drive), depth estimation (0.307 RMSE on NYUv2 with a linear probe), and global recognition (77.7% on Something-Something-V2). Our results demonstrate that V-JEPA 2.1 advances the state of the art in dense visual understanding and world modeling.

Date: June 12, 2026

Correspondence: lmur@unizar.es, abardes@meta.com

Code: https://github.com/facebookresearch/vjepa2

8Meta

![](images/b5644fe121f98ae6648c09e1139fd7e693404bc0fffc3fdd1b3fe3d374dba7b0.jpg)  
Figure 1 V-JEPA 2.1 unlocks high-quality dense features. We compute PCA on patch features extracted from the same image or video and map the top three components to RGB channels for both V-JEPA 2 (ViT-g) and V-JEPA 2.1 (ViT-G). Our novel V-JEPA 2.1 produces dense representations with strong spatial and temporal consistency, learning semantically coherent features where similar objects map to the same PCA components.

![](images/f46c6be4378928c7dd4827fb97e545334edd2d20bf2b06dcde82a1b0f51d3159.jpg)

<details>
<summary>bar chart</summary>

| Task | Model | Value | Change (%) |
| :--- | :--- | :--- | :--- |
| NYUv2 Depth Estimation (↓) | V-JEPA 2 | .642 | +52 |
| NYUv2 Depth Estimation (↓) | V-JEPA 2.1 | .307 | +52 |
| NYUv2 Depth Estimation (↓) | Previous SOTA using frozen backbone | .309 | +52 |
| Ego4D Short Term Anticipation | V-JEPA 2 | 6.02 | +35 |
| Ego4D Short Term Anticipation | V-JEPA 2.1 | 7.71 | +35 |
| Ego4D Short Term Anticipation | Previous SOTA using frozen backbone | 5.67 | +35 |
| DAVIS Object Tracking | V-JEPA 2 | 52.5 | +31 |
| DAVIS Object Tracking | V-JEPA 2.1 | 69.0 | +31 |
| DAVIS Object Tracking | Previous SOTA using frozen backbone | 71.1 | +31 |
| ADE20K Semantic Segmentation | V-JEPA 2 | 24.4 | +96 |
| ADE20K Semantic Segmentation | V-JEPA 2.1 | 47.9 | +96 |
| ADE20K Semantic Segmentation | Previous SOTA using frozen backbone | 54.8 | +96 |
| SSv2 Action Recognition | V-JEPA 2 | 77.3 | +77.7 |
| SSv2 Action Recognition | V-JEPA 2.1 | 77.7 | +77.7 |
| SSv2 Action Recognition | Previous SOTA using frozen backbone | 71.1 | +71.1 |
| EK Action Anticipation | V-JEPA 2 | 39.7 | +40.8 |
| EK Action Anticipation | V-JEPA 2.1 | 40.8 | +40.8 |
| EK Action Anticipation | Previous SOTA using frozen backbone | 27.6 | +27.6 |
| K400 Action Recognition | V-JEPA 2 | 87.3 | +87.3 |
| K400 Action Recognition | V-JEPA 2.1 | 87.7 | +87.7 |
| K400 Action Recognition | Previous SOTA using frozen backbone | 89.4 | +89.4 |
| IN1K Image Classification | V-JEPA 2 | 85.1 | +85.1 |
| IN1K Image Classification | V-JEPA 2.1 | 85.5 | +85.5 |
| IN1K Image Classification | Previous SOTA using frozen backbone | 88.1 | +88.1 |
</details>

Figure 2 V-JEPA 2.1 ViT-G performance across dense and global prediction tasks. We show the relative improvements of V-JEPA 2.1 compared to the previous V-JEPA 2 ViT-g model (Assran et al., 2025). We also report the performance of previous SOTA models using a frozen backbone evaluation: DINOv3 (Siméoni et al., 2025) is the reference model for depth estimation, object tracking, semantic segmentation, SSv2 action recognition, and image classification; InternVideo2s (Wang et al., 2024b) for K400 action recognition, STAformer (Mur-Labadia et al., 2024) for short-term object interaction anticipation, and PlausiVL (Mittal et al., 2024) for action anticipation. Tasks where V-JEPA 2.1 ViT-G obtains SOTA in frozen-backbone evaluation are underlined.

## 1 Introduction

World models hold the promise of enabling agents to perceive, predict, and plan effectively in the physical world (Sutton, 1981; Ha and Schmidhuber, 2018; Assran et al., 2023). At the core of these models lies the state-estimation problem: learning representations that reliably summarize the current world state from low-level, noisy perceptual inputs. Self-Supervised Learning (SSL) from video has recently emerged as a powerful route to this goal (Caron et al., 2021; Assran et al., 2025), because it can exploit large-scale, label-free data to learn representations that capture scene geometry, dynamics, and intrinsic physical properties (Siméoni et al., 2025; Garrido et al., 2025).

Despite rapid progress, learning representations that simultaneously preserve dense spatio-temporal structure (needed for localization, geometry, and tracking) while also capturing dynamics and supporting global understanding (needed for high-level recognition) remains an open challenge. Among recent advances, Joint Embedding Predictive Architectures (JEPA) (LeCun, 2022)—and in particular the V-JEPA family (Bardes et al., 2024; Assran et al., 2025)—have demonstrated strong global video understanding, especially in settings that require modeling motion and dynamics, and have shown promise for enabling prediction and planning in embodied agents (Assran et al., 2025). However, as illustrated in Figure 1, their learned representations can be less amenable to extracting fine-grained local spatial structure. In contrast, other SSL approaches such as DINO (Caron et al., 2021; Oquab et al., 2023; Siméoni et al., 2025) yield high-quality dense features for detection and segmentation, but are primarily image-based and therefore do not directly learn temporal dynamics from video.

In this work, we study self-supervised learning with a latent mask-denoising objective, where the model predicts masked segments of an image or video directly in a learned representation space. Our central finding is that high-quality dense spatio-temporal features—preserving fine-grained spatial layout and motion dynamics—do not emerge reliably when the prediction loss is applied only to masked regions. Instead, extending the predictive loss to the entire input, both masked and unmasked segments, substantially improves the low-level (dense) representations.

Building on this insight, we introduce V-JEPA 2.1, a self-supervised approach for learning unified image and video representations. V-JEPA 2.1 uses a dense predictive loss applied to all tokens (both visible context and masked tokens), grounding each token in its spatio-temporal location and preventing visible tokens from acting as global aggregators—an effect that is key to improving dense feature quality (Figure 3). Additionally, we find that deep self-supervision—applying the loss hierarchically at multiple intermediate encoder layers to provide training signals throughout the network—yields consistent gains on both dense and global downstream tasks. To enable native joint training across modalities, we use modality-specific learned tokenizers for images and videos within a single shared encoder. Finally, we show these improvements scale with data and model capacity: expanding the image component from 1M to 142M images using VisionMix-163M and scaling the model from 300M to 2B parameters leads to systematic downstream gains.

We train and release a suite of V-JEPA 2.1 models (ViT-g/G, 1B/2B), along with two distilled, smaller variants (ViT-B/L, 80M/300M). Empirically, V-JEPA 2.1 achieves state-of-the-art performance on predictive video benchmarks spanning both fine-grained and semantic forecasting: it reaches 7.71 mAP on Ego4D short-term object-interaction anticipation, which requires predicting where and when interactions will occur (localized interaction regions and time-to-interaction), and 40.8 Recall@5 on EPIC-KITCHENS-100 action anticipation, which evaluates the ability to forecast upcoming actions from partial temporal context.

We further show that better dense-features also improves performances on world modelling tasks. V-JEPA 2.1 dense-features leads to +20% success rate compared to VJEPA-2 AC (Assran et al., 2025) on grasping when when deploy our model on real Franka arms in new environment zero-shot. V-JEPA 2.1 is also suitable for robot navigation where it achieves state-of-art performances (5.687 ATE on Tartan Drive) while having 10x faster planning speed compared to previous work (Bar et al., 2025).

Beyond prediction and planning, V-JEPA 2.1 also delivers strong performance in both dense and global understanding tasks. For dense tasks, V-JEPA 2.1 ViT-G sets a new state of the art in linear-probe monocular depth estimation (0.307 RMSE on NYUv2), achieves competitive linear-probe semantic segmentation (85.0 mIoU on Pascal VOC), and produces temporally consistent features for video object segmentation (72.7 J &F-Mean on YouTube-VOS). At the global level, it also attains state-of-the-art accuracy in action recognition (77.7% on Something-Something-v2) and achieves competitive performances in Video Question Answering (VQA) tasks (83.1 accuracy on PerceptionTest).

We hope that these contributions will foster research in learning strong representations for physical world modelling, while empowering many applications in video understanding. We make our code and pretrained models publicly available to facilitate further research and applications.

## 2 Methodology

## 2.1 Preliminaries: Joint-Embedding Predictive Architectures

Joint-Embedding Predictive Architecture (JEPA) (LeCun, 2022) is a self-supervised learning framework designed to learn representations of data by making predictions in a learned latent space, rather than directly in the observation (input) space. JEPA models operate by encoding both a noise-corrupted version and an uncorrupted (clean) version of the same input. A predictor network is then trained to predict the representation of the clean input from the representation of the corrupted input.

Corrupted and clean inputs are processed by the encoder $E _ { \theta } ( \cdot )$ , which produces latent representations. The predictor, $P _ { \phi } ( \cdot )$ , is then used to map the representation of the corrupted input to the representation of the clean input. Given that the encoder and predictor are learned simultaneously, the system admits a trivial and uninformative solution in which the encoder $E _ { \theta } ( \cdot )$ outputs a constant vector regardless of its input. To avoid this representation collapse, explicit (Bardes et al., 2021; Balestriero and LeCun, 2025; Mo and Tong, 2024) or implicit (Grill et al., 2020a; Assran et al., 2023) regularization is used to promote representations that preserve input information.

In this work, we build upon the V-JEPA family of models, where video representations are learned through a mask-denoising objective in the representation space (Bardes et al., 2024; Assran et al., 2025). The V-JEPA objective aims to predict the representation of a video y from another view x of that video that has been corrupted through masking, i.e., from which patches have been randomly dropped. The encoder $E _ { \theta } ( \cdot )$ processes the masked video x, and outputs an embedding vector, or context tokens, for each visible patch. The outputs of the encoder are concatenated with a set of learnable mask tokens $\Delta _ { y }$ that specify the spatio-temporal position of the masked patches. The predictor network processes the combined tokens sequence, and outputs an embedding vector for each input token. The encoder and predictor are trained by minimizing the following

objective:

$$
\mathcal {L} _ {\text { predict }} = \frac {1}{| M |} \sum_ {i \in M} \| P _ {\phi} (E _ {\theta} (x), \Delta_ {y}) _ {i} - \mathrm{sg} (E _ {\overline {{\theta}}} (y) _ {i}) \| _ {1}, \tag {1}
$$

where M is a set containing the masked patch indexes from the x view. The loss use a stop-gradient operator, sg, to prevent representation collapse (Grill et al., 2020b) and an exponential moving average θ of θ is used to update the weight of the encoder $E _ { \overline { { \theta } } } ( y )$ processing the video $y . \mathcal { L } _ { \mathrm { p r e d i c t } }$ is only applied on masked tokens, and not on the context tokens.

Both encoder $E _ { \theta } ( \cdot )$ and predictor $P _ { \phi } ( \cdot )$ are parametrized with Vision Transformer (Dosovitskiy, 2020) and we rely on the same masking strategy as Assran et al. (2025). Refer to Appendix A for more details.

## 2.2 Analysis of V-JEPA Features for Dense Vision Tasks

While V-JEPA has proven to be an effective approach for understanding global semantic information from video, predicting future actions, and planning to reach specific goals (Bardes et al., 2024; Assran et al., 2025), previous works have not investigated the suitability of V-JEPA 2 features for dense vision tasks. To address this gap, we analyze the V-JEPA 2 feature maps through qualitative visualizations and dense downstream tasks evaluation.

For the qualitative visualizations, we compute the Principal Component Analysis (PCA) of patch features extracted from the V-JEPA 2 encoder, and we map the first three components to the RGB color channels. We assess the encoder performance on dense tasks using a linear probing protocol, in which we train a single linear layer on top of the frozen encoder features. We evaluate V-JEPA 2 on semantic segmentation using the ADE20K dataset (Zhou et al., 2017a) and depth estimation on NYUv2 (Silberman et al., 2012b). Refer to Appendix C for more details on the evaluation setup.

Observations and Hypothesis. Feature map visualizations of V-JEPA 2 are shown in Figure 1 and Figure 3. We observe that feature maps are noisy and show only fragmented local spatial structure. Additionally, V-JEPA 2 features obtain limited performance on dense tasks when using a simple linear probing protocol, such as semantic segmentation (22.2 mIoU on ADE20K) or depth estimation (0.682 RMSE on NYUv2), as reported in Table 1. Overall, these results support the conclusion that local information about the visual scene is not easily extractable from the V-JEPA 2 representation.

We hypothesize that the absence of local structure in the feature maps is due to the lack of self-supervision on patches that are not masked, i.e. the context patches. The predictor $P _ { \phi } ( \cdot )$ takes as input the concatenation of context tokens computed by $E _ { \theta } ( x )$ and a set of mask tokens $\Delta _ { y }$ that specify the masked positions to predict. The predictor outputs one token for each input, i.e., for both context and masked tokens. However, the original loss from V-JEPA 2 (Assran et al., 2025) is applied only to the masked tokens, as Equation 1 shows. Therefore, the model has no incentive to encode local information within the context tokens and can instead devote this computation to aggregating global information to minimize $\mathcal { L } _ { \mathrm { p r e d i c t i o n } ; }$ similarly to register tokens (Darcet et al., 2023).

![](images/688dd3ca24833e69e8a0335ef9c052f30651d5c4b7dd5f5811b60a885ecdbbd7.jpg)  
Figure 3 Influence of the context loss $\mathcal { L } _ { c t x } .$ We show PCA visualizations of the feature map representations learned with V-JEPA 2 and with a model trained using V-JEPA 2 plus our $\mathcal { L } _ { c t x }$ loss. While V-JEPA features only show fragmented local spatial structure, explicitly supervising unmasked regions with $\mathcal { L } _ { c t x }$ leads to feature maps that exhibit coherent spatial structure. Similar semantic parts $( \mathrm { e . g . }$ , heads of dogs, wheels of cars) are mapped to the same PCA components.

![](images/d5051823287680660a8e0b309eb3bbe99cfbd7d4a4d6d546d5d1de8227bf221a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Image"] --> B["3D Conv x-enc."]
  C["Image"] --> D["2D Conv x-enc."]
  B --> E["OR"]
  D --> F["OR"]
  E --> G["Add modality embedding"]
  F --> H["Add modality embedding"]
  G --> I["Remove masked tokens"]
  H --> J["Add modalities"]
  I --> K["Remove masked tokens"]
  J --> L["Add modalities"]
  K --> M["Remove masked tokens"]
  L --> N["Add modalities"]
  M --> O["Remove masked tokens"]
  N --> P["Output: x-Encoder, MLP, Multi-level Fusion"]
  O --> Q["Output: y-Encoder, Multi-level Fusion"]
  P --> R["Concatenate mask tokens"]
  Q --> S["P_φ Multi-level Predictor"]
  R --> T["Context loss λ_d·L_1"]
  S --> U["stop-grad"]
  T --> V["Prediction loss L_1"]
  U --> V
  V --> W["stop-grad"]
```
</details>

Figure 4 V-JEPA 2.1 Detailed Architecture. Images and videos are processed by respectively either a 2D or 3D Convolutional patch embedding. Then, 3D Rotational Positional Encoding (RoPE) and learnable modality embedding are added. The x−encoder processes the visible tokens and outputs multi-level embeddings formed by concatenating the normalized output from intermediate encoder blocks. Then, a MLP fuses this multi-level representation and reduces its dimensionality. These context tokens are concatenated with learnable mask tokens carrying spatio-temporal positional information. The predictor processes the combined sequence and produces multi-level predictions for the masked tokens. Training uses two different losses: (i) an L1 loss on masked-token predictions (the original V-JEPA objective), and (ii) a distance-weighted L1 loss on nearby context tokens, both supervised using the y-encoder multi-level outputs.

Context Self-Supervision. To verify this hypothesis, we propose to self-supervise both the mask and context patches and introduce a context loss ${ \mathcal L } _ { \mathrm { c t x } }$ , which is a weighted version of $\mathcal { L } _ { \mathrm { p r e d i c t } }$ applied on the context tokens:

$$
\mathcal {L} _ {\text { context }} = \frac {1}{| C |} \sum_ {i \in C} \lambda_ {i} \| P _ {\phi} (E _ {\theta} (x), \Delta_ {y}) _ {i} - \mathrm{sg} (E _ {\overline {{\theta}}} (y) _ {i}) \| _ {1}, \tag {2}
$$

where C is the set of indexed context tokens, and $\lambda _ { i }$ is a patch-specific weighting parameter described in the next section. The model is trained to minimize $\mathcal { L } _ { \mathrm { p r e d i c t } } + \mathcal { L } _ { \mathrm { c t x } }$ . Figure 3 shows that adding ${ \mathcal L } _ { \mathrm { c t x } }$ has a significant effect on the learned feature maps. With the context loss, local structure now clearly appears in the feature maps, and similar semantic parts (e.g., head of the dogs, wheel of the car) are mapped to the same PCA components. Additionally, adding ${ \mathcal L } _ { \mathrm { c t x } }$ significantly improves performance on dense-prediction tasks, achieving 33.9 mIoU on ADE20K (up from 22.2), and 0.473 RMSE on NYUv2 (down from 0.682). Hence, those results validate that by explicitly supervising context tokens, the model learns features that encode coherent local structure.

## 2.3 V-JEPA 2.1: Improving Dense Video SSL Features

Building on the previous observation, we introduce V-JEPA 2.1, a self-supervised training recipe for learning representations that combine high-quality dense local features with global semantic understanding. Our key algorithmic innovations are (1) Dense Prediction loss that applies self-supervision on both masked and unmasked tokens (Section 2.3.1) and, (2) Deep Self-Supervision of the encoder intermediate layers via a multi-level predictor (Section 2.3.2). Additionally, we explore (3) a Multi-Modal Tokenizer with modalityspecific patch embeddings for images and videos (Section 2.3.4); (4) Data Scaling through a more diverse and balanced image–video training distribution (Section 2.3.3); and Model Scaling to ViT-G (Section 2.3.5), enabling state-of-the-art downstream performance and effective distillation to smaller models (ViT-L, ViT-B, Section 3.10).

![](images/77d9a8fdc9d1ffb178777fe7b8a0d93d2d8a4a4fa5e7f46139e630c421bd941b.jpg)

<details>
<summary>bar chart</summary>

| Model | Segmentation (ADE20k) | Classification (SSv2) |
| :--- | :--- | :--- |
| V-JEPA 2 | 22.2 | 72.8 |
| + Context Loss | 33.8 | 62.5 |
| + Deep Self-Supervision. | 38.6 | 72.1 |
| + Data Scaling | 40.8 | 72.6 |
| + Multimodal Patch. | 41.4 | 72.6 |
| + Model Scaling | 47.1 | 76.1 |
| + High-Res. Annealing | 47.9 | 77.7 |
Acc. (SSv2) axis scale from 60 to 80.
</details>

Figure 5 Impact of individual components of our novel V-JEPA 2.1 training recipe. The ablation is conducted starting from a ViT-L architecture, on single-image semantic segmentation on ADE20k, and action classification on SSv2. Introducing Weighted-Context Self-Supervision with the context loss significantly improves segmentation, at the cost of classification. Deep Self-Supervision allows to retrieve the performance, and further improving segmentation. Scaling the image data with VisionMix 163M dataset, combined with a Multi-Modal Tokenizer further improve the results. Finally, the recipe scales with model size and high resolution cool-down.

VJEPA 2.1 Architecture. We illustrate the V-JEPA 2.1 architecture in Figure 4. An input, either an image or a video, is projected into a sequence of embedding vectors, or tokens, using a modality-specific patch embedding. Mask corruption is then applied to the sequence by randomly dropping patch tokens. The x-encoder processes the remaining visible context tokens and outputs representations from multiple encoder levels in addition to the final output. The multi-level representations are then concatenated along the channel axis and fed to an MLP to reduce their dimensionality. Context tokens are concatenated, along the sequence axis, with learnable mask tokens that carry spatio-temporal positional information of the masked patches. The predictor processes the combined sequence and produces multi-level predictions for each token. Training uses two different losses: (i) an L1 loss on masked-token predictions (the original V-JEPA objective), and (ii) a distance-weighted L1 loss for context tokens. Both use the y-encoder outputs as targets, which process the unmasked sequence of patches from the input images or videos. Losses are applied to several intermediate representation levels in addition to the encoder output.

Implementation Details. We follow the warmup-constant learning rate schedule of V-JEPA 2 and we train models for 135,000 iterations. We maintain the teacher EMA coefficient and weight decay at fixed values. Each video sample is a clip of 16 frames at a resolution of 256 × 256, and each image sample has a resolution of 256 × 256. Additionally, in a second stage we explore the effect of applying a cool-down phase, i.e., decaying the learning rate and increasing the input images and videos resolution. We further train our models for 12,000 iterations during this cool-down phase, increasing the input resolution: video clips now have 64 frames at a resolution of 384 × 384, and images have a resolution of 512 × 512. Ablation results are reported after the first training phase, whereas final downstream tasks results use the full warmup–constant–cooldown schedule. More details and all hyper-parameters are provided in Appendix A.

Empirical Evaluation. To evaluate our design choice, we rely on a set of dense-vision tasks (ADE20K and NYUv2) using a linear probing evaluation protocol following Siméoni et al. (2025) and global recognition tasks (Something-Somethingv2 for action recognition and ImageNet for object recognition) with an attentive probing protocol following Assran et al. (2025). We ablate the effect of each architecture component in Figure 5 and Table 1. In the following, we describe each component in more detail, as well as their impact on downstream performance.

Table 1 Impact of individual components of our novel V-JEPA 2.1 training recipe. Results on image classification (IN1K), video classification (SSv2), depth estimation (NYU), and semantic segmentation (ADE20K). Introducing the Context Loss improves dense tasks but reduces classification performance on SSv2. Incorporating our Deep Self-Supervision restores the classification performance. The VisionMix 163M dataset, the Multi-Modal Tokenizer, and scaling model size further improve the results.

<table><tr><td>IN1K Acc.</td><td>SSv2 Acc.</td><td>NYU RMSE ↓</td><td>ADE20K mIoU</td><td></td></tr><tr><td>82.2</td><td>72.8</td><td>0.682</td><td>22.2</td><td>V-JEPA 2</td></tr><tr><td>72.6</td><td>62.5</td><td>0.474</td><td>33.8</td><td>+ Context Loss</td></tr><tr><td>80.8</td><td>72.1</td><td>0.463</td><td>38.6</td><td>+ Multi-level Pred.</td></tr><tr><td>81.6</td><td>72.6</td><td>0.418</td><td>40.8</td><td>+ Vision Mix</td></tr><tr><td>81.6</td><td>72.6</td><td>0.415</td><td>41.4</td><td>+ Multi-modal Tok.</td></tr><tr><td>84.8</td><td>76.1</td><td>0.365</td><td>47.1</td><td>+ Model Scaling</td></tr><tr><td>85.5</td><td>77.7</td><td>0.307</td><td>47.9</td><td>+ Cool-down.</td></tr></table>

Table 2 Impact of the context loss weighting scheme. Results on semantic segmentation (ADE20K) and video classification (SSv2). Introducing the context loss with a fixed coefficient has a significant positive impact on the segmentation performance, however at the cost of classification performance. Warmup on the coefficient allows to mitigate part of the lost performance. Our weighted scheme further improves the performance on both segmentation and classification.

<table><tr><td>ADE20K mIoU</td><td>SSv2 Acc.</td><td> $\lambda$ </td><td>Scheme</td><td>Warmup</td></tr><tr><td>22.2</td><td>72.8</td><td>V-JEPA 2,  $\lambda = 0$ </td><td>cte</td><td></td></tr><tr><td>26.4</td><td>71.0</td><td>0.05</td><td>cte</td><td></td></tr><tr><td>29.6</td><td>62.5</td><td>0.2</td><td>cte</td><td></td></tr><tr><td>27.5</td><td>53.8</td><td>0.5</td><td>cte</td><td></td></tr><tr><td>24.6</td><td>51.1</td><td>1.0</td><td>cte</td><td></td></tr><tr><td>30.5</td><td>60.5</td><td>0.2</td><td>cte</td><td>√</td></tr><tr><td>32.2</td><td>61.5</td><td>0.5</td><td>cte</td><td>√</td></tr><tr><td>33.8</td><td>62.5</td><td>0.5</td><td>weighted</td><td>√</td></tr></table>

## 2.3.1 Dense Prediction Loss

We propose applying our self-supervised loss to both masked and visible patches by minimizing ${ \mathcal { L } } _ { \mathrm { d e n s e } } =$ $\mathcal { L } _ { \mathrm { p r e d i c t } } + \mathcal { L } _ { \mathrm { c t x } } .$ , where $\mathcal { L } _ { \mathrm { p r e d i c t } }$ is defined in $\operatorname { E q . }$ 1 and ${ \mathcal L } _ { \mathrm { c t x } }$ is defined in Eq. 2. Naive application of ${ \mathcal L } _ { \mathrm { c t x } }$ loss leads to poor performance on global semantic tasks, as the system can potentially find trivial solutions, such as copying the context features. We therefore explore various weighting coefficients $\lambda _ { i }$ in Eq. 2. Table 2 presents an ablation on various weighting schemes. First, we experiment with fixed values and set all $\lambda _ { i }$ to a constant $\lambda$ and try values in the range $\lambda = [ 0 . 0 , 0 . 0 5 , 0 . 2 , 0 . 5 , 1 . 0 ]$ . We observe that as we increase $\lambda ,$ the performance in semantic segmentation in the ADE20k dataset increases significantly up to certain point, but at the cost of the performance in action recognition in the SSv2 dataset decreases. We then introduce a progressive warm-up of $\lambda$ to restore action-recognition performance, with a schedule from epochs 50–100. We found empirically that it greatly stabilizes training. Following, we introduce a dynamic weighting scheme where ${ \mathcal L } _ { \mathrm { c t x } }$ for a given patch i is weighted by the inverse square root of its minimum spatio-temporal distance to any masked token in the video sequence: i.e. setting

$$
\lambda_ {i} = \frac {\lambda}{\sqrt {\mathrm{d} _ {\min} (i , M)}} \tag {3}
$$

in $\mathcal { L } _ { \mathrm { c t x } }$ in Eq. 2, where $\mathrm { { d } _ { \mathrm { { m i n } } } }$ is the distance, in number of blocks, between a context token and its closest mask token. This weighting emphasizes patches near masked regions by enforcing local continuity between masked and context areas, yielding a good trade-off between segmentation and action recognition performance.

Introducing our novel context loss $\mathcal { L } _ { c t x }$ with our weighted scheme improves the performance on dense vision tasks (22.2 → 33.9 mIoU on ADE20k, 0.682 → 0.473 RMSE on NYUv2). Qualitatively, this loss smooths the feature maps by removing noisy artifacts, as shown in Figure 3. However, Table 1 shows that there is still a degradation in video understanding (72.8 → 62.5 on SSv2) and image classification (82.2 → 72.6 on IN1K).

## 2.3.2 Deep Self-Supervision

We self-supervise the encoder representation not only at the output but also at multiple intermediate levels. We first concatenate, along the channel dimension, the outputs of three intermediate x-encoder blocks, in addition to the output layer. Then, a lightweight MLP fuses these multi-level representations and reduces their dimensionality before feeding them into the predictor. The predictor processes the fused multi-level sequence of context and mask tokens and produces four outputs corresponding to the four encoder layers. Both the prediction loss and the context loss are then applied at each one of these four levels.

Deep Self-Supervision leads to significant improvement on downstream performance for both global and dense tasks, as shows Figure 5. Furthermore, it allows local information to flow towards the final layers, effectively removing the need for intermediate layers in dense downstream tasks as we show in Appendix D.1. Deep Self-Supervision allows to recover the global understanding capabilities of V-JEPA 2 (72.0 on SSv2, 80.8 on IN1K) while improving on dense tasks with the context loss (38.6 mIoU on ADE20K, 0.463 RMSE on NYU).

Table 3 Comparison between VideoMix22M and VisionMix163M. We increase the visual diversity of the pretraining dataset by increasing the number of images, replacing ImageNet with LVD-142M, and update the YT-1B sampling weight to promote data-source with more visual diversity.

<table><tr><td>Source</td><td>Samples</td><td>Type</td><td>Total Hours</td><td>V-JEPA2 weights</td><td>V-JEPA2.1 weights</td></tr><tr><td>SSv2 (Goyal et al., 2017)</td><td>168K</td><td>EgoVideo</td><td>168</td><td>0.056</td><td>0.170</td></tr><tr><td>Kinetics (Carreira et al., 2019)</td><td>733K</td><td>ExoVideo</td><td>614</td><td>0.188</td><td>0.010</td></tr><tr><td>Howto100M (Miech et al., 2019)</td><td>1.1M</td><td>ExoVideo</td><td>134K</td><td>0.318</td><td>0.100</td></tr><tr><td>ImageNet (Deng et al., 2009)</td><td>1M</td><td>Images</td><td>n/a</td><td>0.250</td><td>0.</td></tr><tr><td>YT-1B (Zellers et al., 2022)</td><td>19M</td><td>ExoVideo</td><td>1.6M</td><td>0.188</td><td>0.720</td></tr><tr><td>LVD-142M (Oquab et al., 2023)</td><td>142 M</td><td>Curated images</td><td>n/a</td><td></td><td></td></tr></table>

## 2.3.3 Scaling Image Data.

DINOv2 (Oquab et al., 2023) introduced a cluster-based retrieval strategy to select images from a large pool of raw internet data, resulting in a curated set of 142 million images, referred to as the LVD-142M dataset. Using a similar approach, V-JEPA 2 (Assran et al., 2025) collected and curated video scenes from YT1B videos (Zellers et al., 2022), combined with other publicly available video datasets, yielding a large-scale collection of 19 million video samples from the internet, corresponding to 1.6 million hours. Both works demonstrated the positive effect of data scaling for SSL pretraining.

Building on these insights, we construct our VisionMix163M Dataset, combining large-scale curated sources from these two prior works. As we show in Table 3, we replace the 1M-image ImageNet subset from VJEPA-2 pretraining data with LVD-142M, providing a broader and more diverse appearance distribution. Since this extensive image collection already covers many static visual concepts, we shift the video sampling strategy towards more dynamic, motion-rich content, increasing the SSv2 sampling weight from 0.056 to 0.170. We also found beneficial to increase the contribution of YT-1B from 0.188 to 0.720, which contains much more heterogeneous video samples.

Rather than mixing images and videos within the same training batch, we leverage distributed training by assigning separate workers to each modality. After each iteration, gradients from video-only and image-only nodes are aggregated before updating the model. The image/video ratio is controlled through per-modality batch sizes. Empirically, we found optimal performance with 128 video clips (16 frames each) and 2,304 images per global batch. As we report in Table 1, the improvement of training on a more diverse and extend database is beneficial in all tasks (72.1 → 72.6 on SSv2, 80.8 → 81.6 on ImageNet, 38.6 → 40.8 on ADE20K, 0.463 → 0.418 RMSE on NYUv2).

## 2.3.4 Multi-Modal Tokenizer

Previous work exploring joint training on image and video, such as V-JEPA 2, was suboptimal: it employed a single 3D convolution as the patch-embedding layer. Images were duplicated temporally and treated as a 16-frame static video, which significantly increased their computational cost and introduced an incorrect representational bias (i.e., images were interpreted as static videos). Instead, we introduce a multi-modal tokenizer, which applies a 3D convolution of $1 6 \times 1 6 \times 2$ for processing videos and a 2D convolution of 16 × 16 for images. We also add a modality-learnable token to both the encoder and the predictor inputs, which explicitly encodes whether the input comes from the image or video pathway. These learnable tokens condition the processing on the input modality, helping the model disentangle stronger static appearance cues in images and temporal motion information in videos.

This design allows each modality to be processed in its native form, eliminating the need for temporal duplication of images and improving computational efficiency. Additionally, we observe that using the Multi-Modal Tokenizer has a positive effect on dense-task performance: it improves mIoU on ADE20K from 40.8 to 41.4, while performance on action or object recognition remains stable.

## 2.3.5 Scaling Model Size to ViT-G (2B) and High-Resolution Cool-Down

Next, we explore the effect of model scaling and high-resolution cooldown. Scaling the V-JEPA encoder from a ViT-L with 300 million parameters to a ViT-G with 2 billion parameters leads to significant improvements on all downstream tasks (72.6 → 76.1 on SSv2, 81.6 → 84.8 accuracy on ImageNet, 41.4 → 47.1 mIoU on ADE20K, 0.415 → 0.365 RMSE on NYUv2).

Additionally, introducing a cool-down phase for the learning rate, during which we increase both the spatial resolution of images (from 256 × 256 to 512 × 512) and the spatio-temporal resolution of videos (from 16 frames at 256 × 256 to 64 frames at 384 × 384), further improves performance on all tasks. This achieves an accuracy of 77.7 on SSv2, 85.5 on ImageNet, an mIoU of 47.9 on ADE20K, and an RMSE of 0.307 on NYUv2, resulting in our best performing model. The benefits of a cool-down training phase are particularly strong in depth estimation (0.365 → 0.307 RMSE on NYUv2).

## 2.3.6 Model Distillation

We scaled the size of the model not only to achieve top-tier performance but also to enable effective compression of the model through distillation Hinton et al. (2015). We distill our ViT-G (2B) model into smaller variants (ViT-B, 80M; ViT-L, 300M). The distillation protocol is adapted from our pretraining recipe, with key differences: i) we replace the Exponential-Moving-Average (EMA) of the target encoder by a frozen teacher model, ii) we keep an EMA copy of the student encoder, that is not used in the loss, but serves as the final model, iii) the distillation loss is identical to our pretraining loss, except it is only computed on the last layer of the teacher encoder, and it does not use deep self-supervision, iv) we use a predictor with only 12 blocks and a final linear layer matching the teacher embedding dimension. All other hyper-parameters—including masking ratios, cool-down schedules, learning rates, and data augmentations—remain identical to the original pretraining recipe. We provide more details on the distillation protocol in Appendix B.

## 3 Results

In this section, we evaluate the performance of V-JEPA 2.1 on a large variety of downstream tasks. Throughout the experiments, we employ V-JEPA 2.1 as a frozen encoder, demonstrating the versatility of its features. We first demonstrate the predictive capabilities of V-JEPA 2.1 in two forecasting tasks: short-term object interaction anticipation (Section 3.1) and action anticipation (Section 3.2). We then demonstrate that can leverage V-JEPA 2.1 to learn an action-condition world model and performs robot manipulation (Section 3.3) and navigation tasks (Section 3.4) in zero-shot setup. Then, we evaluate the quality of the learned dense features by assessing the V-JEPA 2.1 performance on single-image depth estimation and semantic segmentation (Section 3.5). Following, we analyze the temporal consistency of V-JEPA 2.1 representations through the video object segmentation task (Section 3.5). We also analyze V-JEPA 2.1 global understanding on two high-level understanding tasks: probe-based video classification and image classification (Section 3.7). Finally, we present results for our smaller distilled models (Section 3.10). We provide more details on the experimental setup and all pretraining hyper-parameters in Appendices A and C.

## 3.1 Short-Term Object Interaction Anticipation

Short-Term object-interaction Anticipation (STA) consists in predicting future object interaction in a egocentric scenario, predicting the next active object bounding box b, the object noun category N the action verb V and the time until contact (δ). This formulation evaluates fine-grained 2D localization, semantic understanding of objects, temporal reasoning over video dynamics, and the ability to forecast user intentions. Using the Ego-4D STA benchmark (Grauman et al., 2022), we show that V-JEPA 2.1 outperforms previous state-of-the-art performance by a significant margin due to its high-quality dense features and predictive capabilities.

![](images/d97574fab85c2e7ab046b8fe5b4c3571d6e66204e39274641bc3d2f821da1426.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Input video V:t"] --> B["t"]
  B --> C["δ"]
  C --> D["t + δ"]
  D --> E["Frame of contact"]
  F["Multiple predictions in the last observed frame V_t"] --> G["take puzzle_or_game_piece"]
  G --> H["topt playing_cords in TIC: 1.35s"]
  H --> I["ground-truth extracted from the frame of contact V_t+δ"]
```
</details>

Figure 6 Short-Term Anticipation task. Given a video observed up to time t (denoted $V _ { : t } )$ , the model predicts a set of future object–interaction events. Each prediction includes: (i) a bounding box localizing the object that will be interacted with, (ii) the noun class of that object, (iii) a verb class describing the upcoming interaction, and (iv) the anticipation time δ, i.e., the number of seconds between the current frame and the start of the interaction. Ground-truth labels are derived from the contact frame $V _ { t + \delta } ,$ , but are expressed in the coordinate frame of the last observed frame $V _ { t } .$ .

Task. We evaluate V-JEPA 2.1 on the STA v2 split of Ego4D (Grauman et al., 2022), which comprises 243 hours of annotated video clips spanning 128 noun and 81 verb categories, with a total of 98,276 training and 47,395 validation samples. We compare against state-of-the-art methods (Ragusa et al., 2023; Mur-Labadia et al., 2024; Thakur et al., 2023), and we further asses the performance of DINOv2 (Darcet et al., 2023) and DINOv3 (Siméoni et al., 2025) image encoders using our proposed attentive probe. Following the evaluation protocol of Grauman et al. (2022), we report Top-5 Average Precision (AP) and Top-5 mean Average Precision (mAP) metrics. As in standard mAP, predictions are matched to ground-truth boxes using IoU > 0.5 and additional class-specific criteria. For instance, the Top-5 mAP All variant requires a correct noun, correct verb, IoU above 0.5, and a time-to-contact δ prediction within a 0.25-second tolerance. To address the inherently multi-modal nature of future anticipation—where multiple plausible next-active objects may exist—the metric discounts up to four highest-scoring false positives per example, thereby avoiding penalties for plausible but unannotated predictions. Additionally, we report individual metrics to evaluate the temporal (δ), spatial (bounding boxes), and semantic (noun and verb) dimensions of the task.

Evaluation protocol. The model receives as input a low-resolution video clip (384 px, 8 frames spanning 0.5 seconds) and a high-resolution image (1080 px) corresponding to the last frame of the clip. Using the frozen encoder of V-JEPA 2.1, we extract last-level features independently from both inputs using the respective patchifier (a 2D convolution for the high-resolution image and a 3D convolution for the video) and adding the corresponding modality embedding. For the video features, we train a four-layer attentive probe, followed by the frame-guided temporal pooling module from Mur-Labadia et al. (2024). This module aggregates the 3D video tokens into a 2D representation that is spatially aligned with the spatial reference of the last frame. The pooled video features are then summed to the image features, and the resulting fused representation is rescaled into four multiscale feature maps, which serve as input to the detection head. To ensure a fair comparison with state-of-the-art methods (Ragusa et al., 2023; Mur-Labadia et al., 2024; Thakur et al., 2023), we adopt the same detection head (Ragusa et al., 2023) utilized in these approaches. This head extends Faster R-CNN by adding linear layers to additionally predict the verb category and the time to contact.

Table 4 Short-Term Object Interaction Anticipation on Ego4D. Following Grauman et al. (2022), we report different Top-5 Average Precision (AP) and mean Average Precision (mAP) metrics, where the winning score is the mAP All. We compare results reported in the literature with different frozen SSL encoders (DINOv2, DINOv2, V-JEPA 2 and our V-JEPA 2.1) extended with an attentive-based predicted head. V-JEPA 2.1 obtains the state-of-the-art due to its high-quality dense features and predictive capabilities.

<table><tr><td>Model</td><td>APb</td><td>APb+N</td><td>APb+V</td><td>APb+δ</td><td>APb+N+V</td><td>APb+δ+N</td><td>APb+δ+V</td><td>APAll</td><td>mAPN</td><td>mAPN+V</td><td>mAPN+δ</td><td>mAPAll</td></tr><tr><td colspan="13">Results Reported in the Literature</td></tr><tr><td>StillFast (Ragusa et al., 2023)</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>20.2</td><td>10.3</td><td>7.17</td><td>3.96</td></tr><tr><td>STAformer (Mur-Labadia et al., 2024)</td><td>38.3</td><td>28.3</td><td>16.6</td><td>12.27</td><td>12.8</td><td>8.89</td><td>5.47</td><td>4.06</td><td>29.4</td><td>15.3</td><td>9.94</td><td>5.67</td></tr><tr><td>GANO (Thakur et al., 2023)</td><td>45.3</td><td>27.0</td><td>12.2</td><td>16.6</td><td>6.54</td><td>9.0</td><td>4.18</td><td>2.47</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="13">Encoders using our same protocol</td></tr><tr><td>DINOv2 ViT-L (Oquab et al., 2023)</td><td>45.1</td><td>37.8</td><td>21.9</td><td>14.4</td><td>17.6</td><td>10.6</td><td>7.27</td><td>5.25</td><td>28.3</td><td>15.2</td><td>9.22</td><td>5.25</td></tr><tr><td>DINOv2 ViT-g (Oquab et al., 2023)</td><td>47.8</td><td>38.1</td><td>23.6</td><td>14.1</td><td>19.1</td><td>10.9</td><td>7.29</td><td>5.73</td><td>30.6</td><td>16.8</td><td>9.37</td><td>5.44</td></tr><tr><td>DINOv3 ViT-H+ (Siméoni et al., 2025)</td><td>48.9</td><td>40.2</td><td>23.4</td><td>13.6</td><td>19.4</td><td>10.8</td><td>6.71</td><td>5.45</td><td>32.4</td><td>17.2</td><td>9.53</td><td>5.20</td></tr><tr><td>DINOv3 ViT-7B (Siméoni et al., 2025)</td><td>50.2</td><td>40.9</td><td>23.8</td><td>14.2</td><td>19.8</td><td>11.5</td><td>7.33</td><td>5.82</td><td>33.8</td><td>17.4</td><td>10.3</td><td>5.68</td></tr><tr><td>V-JEPA 2 ViT-g (Assran et al., 2025)</td><td>45.7</td><td>34.4</td><td>22.2</td><td>16.7</td><td>16.9</td><td>12.1</td><td>8.56</td><td>6.26</td><td>27.2</td><td>14.9</td><td>10.4</td><td>6.02</td></tr><tr><td>V-JEPA 2.1 ViT-g</td><td>47.3</td><td>36.5</td><td>23.6</td><td>18.1</td><td>18.2</td><td>13.5</td><td>9.55</td><td>7.21</td><td>28.7</td><td>15.5</td><td>11.4</td><td>6.75</td></tr><tr><td>V-JEPA 2.1 ViT-G</td><td>50.7</td><td>39.3</td><td>25.8</td><td>20.2</td><td>20.3</td><td>15.1</td><td>10.8</td><td>8.20</td><td>30.9</td><td>17.2</td><td>12.8</td><td>7.71</td></tr></table>

![](images/2942787dc2266770b0dcb5122bb3be543df8fb77b889b7fa9702e1b9b2574442.jpg)  
Figure 7 Short-Term Anticipation (STA) qualitative results. We compare our model predictions with the ground-truth data. V-JEPA 2.1 predicts multiple scenarios, where the predicted object and verb classes correspond with plausible human-object interactions.

Results. Table 4 presents a comparison with state-of-the-art methods on the Short Term Anticipation task. V-JEPA 2.1 demonstrates achieves the absolute state-of-the-art performance with an overall mAP of 7.71. This represents a relative improvement of approximately 35% over the previous best method (Mur-Labadia et al., 2024), which introduces task-specific training components. As it is shown by the individual metrics, this gain is primarily driven by improved understanding of the next action 25.8 $\mathrm { A P _ { b + V } }$ and the time to contact $2 0 . 2 0 ~ \mathrm { A P _ { b + \delta } }$ , 12.9 $\mathrm { m A P _ { \ b + \delta } }$ . Furthermore, the high-quality V-JEPA 2.1 dense features enable the precise 2D detection of the next-active object bounding boxes, achieving 50.7 AP and surpassing DINOv3 ViT-7B in this category. Qualitative results are shown in Figure 7. V-JEPA 2.1 predicts plausible short-term interactions, with accurate bounding box localization, robust understanding of object categories, and plausible inference of the user next action intentions.

## 3.2 Action Anticipation

Action anticipation consists in predicting the future action given a contextual video clip leading up to some time before the action. Using the $\mathrm { E p i c } .$ -Kitchens-100 (EK100) benchmark (Damen et al., 2022), we show that V-JEPA 2.1 outperforms the action anticipation performance of V-JEPA 2, and sets a new state-of-the-art for

Table 5 Action Anticipation on EK100. Comparison with the state-of-the-art on the EK100 Action Anticipation benchmark. We report mean-class recall-at-5 for verb, noun and action on the validation set of EK100. V-JEPA 2.1 offers comparable performance to V-JEPA 2 for the ViT-g 1B model, but shows improved performance for the ViT-G 2B model, setting a new state-of-the-art for the task.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Param.</td><td colspan="3">Action Anticipation</td></tr><tr><td>Verb</td><td>Noun</td><td>Action</td></tr><tr><td>InAViT (Roy et al., 2024)</td><td>160M</td><td>51.9</td><td>52.0</td><td>25.8</td></tr><tr><td>Video-LLaMA (Zhang et al., 2023)</td><td>7B</td><td>52.9</td><td>52.0</td><td>26.0</td></tr><tr><td>PlausiVL (Mittal et al., 2024)</td><td>8B</td><td>55.6</td><td>54.2</td><td>27.6</td></tr><tr><td>V-JEPA 2 ViT-g (Assran et al., 2025)</td><td>1B</td><td>63.6</td><td>57.1</td><td>39.7</td></tr><tr><td>V-JEPA 2.1 ViT-g</td><td>1B</td><td>63.6</td><td>56.2</td><td>38.4</td></tr><tr><td>V-JEPA 2.1 ViT-G</td><td>2B</td><td>64.3</td><td>59.9</td><td>40.8</td></tr></table>

the task.

Task. The EK100 dataset is comprised of 100 hours of cooking activities recorded from an egocentric perspective across 45 kitchen environments. Each video in EK100 is annotated with action segments, which include a start timestamp, an end timestamp, and an action label. There are 3,568 unique action labels, each consisting of a verb and a noun category, with a total of 97 verb categories and 300 noun categories. The EK100 action anticipation task involves predicting noun, verb, and action (i.e., predicting verb and noun jointly) from a video clip, referred to as context, that occurs before the start timestamp of an action segment. The interval between the end of the context and the beginning of the action segment is the anticipation time, which is set to 1 second by default. Given that different future actions may be possible from a given context, mean-class recall-at-5 is used as the metric to measure performance (Damen et al., 2022).

Evaluation protocol. We employ the same protocol as V-JEPA 2 and use an attentive probe trained on top of the frozen V-JEPA 2.1 encoder and predictor to anticipate future actions. Specifically, we sample a video clip that ends 1 second before an action starts. This video context is fed to the encoder. The predictor takes the encoder representation, along with the mask tokens corresponding to the frame 1 second into the future, and predicts the representation of the future video frame. The outputs of the predictor and encoder are concatenated along the token dimension and fed to an attentive probe with a similar architecture to those used in our probe-based video classification protocol, with the difference being that the anticipation probe’s final cross-attention layer learns three query tokens (as opposed to one), and each query output is fed to a different linear classifier to predict the action category, the verb category, and the noun category respectively. A focal loss (Lin et al., 2017b) is applied to each classifier independently and then summed before back-propagating through the shared attention blocks of the probe.

Results. Table 5 summarizes the results on the validation set of the EK100 action anticipation benchmark. We compare V-JEPA 2.1 ViT-g 1B and ViT-G 2B encoders with V-JEPA 2 ViT-g 1B encoder. Both leverage 32 frames with 8 frames per second at resolution 384 × 384 as video context. V-JEPA 2.1 is comparable to V-JEPA 2 on the same 1B model size, but show scaling of the performance to the 2B models size, setting a new absolute state-of-the-art performance at 40.8 Action Recall@5, corresponding to a +2.8% improvement.

## 3.3 Robotic Arm Planning

Next, we evaluate V-JEPA 2.1 for robot manipulation. To that end, we train a frame-causal action-conditioned predictor, following the protocol of Assran et al. (2025). In particular, we use the public codebase from Assran et al. (2025) and modify it to compute features using our V-JEPA 2.1 video encoder; the predictor is an identical 24-layer transformer network containing approximately 300M parameters, and is trained on the Droid raw dataset (Khazatsky et al., 2024) using both a teacher-forcing and two-step rollout loss. The model is then deployed in our lab, zero-shot, on a table-top Franka Panda robot arm with a parallel-jaw gripper, and evaluated on reach, grasp, and pick-and-place tasks with visual goal specification via model-predictive control. We use the same exact task configuration files provided in Assran et al. (2025), and similar planning hyper-parameters.

VJEPA 2  
![](images/2c630df7525bcb21568b8a60c22c556a22ad2551c71776b4a182b2ab9adbb3a9.jpg)

<details>
<summary>natural_image</summary>

Interior view of a laboratory or lab setup with robotic arm, glassware, and equipment (no visible text or symbols)
</details>

![](images/e6b46840bce30ec919b06c59813b9d676b689e222a8e417d94eb888c7dad8dcc.jpg)

<details>
<summary>natural_image</summary>

Interior view of a workshop or lab space with equipment and no visible text or symbols
</details>

![](images/c5c8f8ec17a6d90a6f94104a24693e5a8ebf24a6ac57611cbd50fc019efdedf5.jpg)

<details>
<summary>natural_image</summary>

Laboratory setup with robotic arm, lab equipment, and cleaning items (no visible text or symbols)
</details>

VJEPA 2.1  
![](images/5587f1ff1d44a294dfa70c41b2805b0dba95ecce48cd229df815c78727283890.jpg)

<details>
<summary>natural_image</summary>

Industrial robotic arm operating a cylindrical device on a workbench (no visible text or symbols)
</details>

Goal Frame

![](images/3c811fc445bb11e86f33c955fac0619f27c95db7273e0936116d9d7d18eb9663.jpg)

<details>
<summary>natural_image</summary>

Laboratory setup with robotic arm and lab equipment (no visible text or symbols)
</details>

Start Frame

![](images/994a64704f2bc93a44ae3241cd539c781e7ac22c7e4cd30639175b03f937283f.jpg)

<details>
<summary>natural_image</summary>

Industrial robotic arm operating on a workbench with lab equipment in the background (no visible text or symbols)
</details>

Robot Execution

VJEPA 2.1 Energy Landscape  
![](images/4eb337acfe60f1fc7ce6613fac56cfd7065d1c314c3f0d14101244d699e4bb0b.jpg)

<details>
<summary>3d surface plot</summary>

| Δx   | Δy   | Energy |
|------|------|--------|
| -0.1 | 0.1  | 0.25   |
| 0    | 0    | 0.35   |
| 0.1  | -0.1 | 0.3    |
| -0.1 | 0    | 0.28   |
| 0    | 0.1  | 0.26   |
| -0.1 | -0.1 | 0.24   |
</details>

Figure 8 Zero-shot evaluation of VJEPA 2 and VJEPA 2.1 models on a table-top Franka Panda robot arm with a parallel-jaw gripper, demonstrating Grasp of a cup using visual goal specification via model-predictive control. VJEPA 2.1 features better encode depth information, leading to an improvement in robot manipulation when grasping the cup involves reasoning over actions along the depth axis of the camera.

Success rates are reported in Table 6. VJEPA 2.1 exhibits an improved depth understanding, leading to a 10% improvement in the success rate of Grasp compared to VJEPA 2 (cf. Figure 8). Moreover, we find that the VJEPA 2.1 model unlocks the benefit of planning with slightly longer rollouts, perhaps due to the improvement in dense features afforded by our model. In particular, by reducing the number of CEM samples and planning over 8 steps, we observe an additional improvement in the success rate on Grasp. By contrast, we actually observe a degradation in the success rate of VJEPA 2 when planning over longer horizons. Qualitatively, we further observe that task failures using the VJEPA 2.1 model on Pick-and-Place and Grasp are a result of poor planning over gripper actions, as opposed to failures in spatial understanding; e.g., closing the gripper too soon such that you cannot grasp the object, or slightly opening the gripper while in transit leading to the object being dropped.

Table 6 Planning Performance. Comparing closed-loop robot manipulation of a cup using MPC with VJEPA 2 versus VJEPA 2.1. In both cases we leverage the cross-entropy method (Rubinstein, 1997) for optimizing the sequence of actions using a single A100 GPU. For each robot skill, we evaluate each model across 10 tasks and average the results. The improved depth understanding and dense features of VJEPA 2.1 lead to a 20% improvement in the success rate of Grasp. Moreover, qualitatively, we find that any failures on Pick-and-Place and Grasp are a result of poor planning over gripper actions, as opposed to failures in spatial understanding (e.g., closing the gripper too soon, such that you cannot grasp the object, or slightly opening the gripper while in transit leading to the object being dropped.

Planning Details

<table><tr><td>Method</td><td>#Samples</td><td>Iter.</td><td>Horizon</td><td>Time</td><td>Reach</td><td>Grasp</td><td>Pick-&amp;-Place</td></tr><tr><td>VJEPA 2 (Assran et al., 2025)</td><td>800</td><td>10</td><td>1</td><td>3 sec.</td><td>100%</td><td>60%</td><td>80%</td></tr><tr><td rowspan="2">VJEPA 2.1 (ours)</td><td>800</td><td>10</td><td>1</td><td>3 sec.</td><td>100%</td><td>70%</td><td>80%</td></tr><tr><td>300</td><td>15</td><td>8</td><td>14 sec.</td><td>100%</td><td>80%</td><td>80%</td></tr></table>

![](images/212a8b2edc5fd1937477202cf051a6276c661e4f02aa8be312c7dc62b46e2086.jpg)

<details>
<summary>text_image</summary>

start plan
goal
</details>

Figure 9 Planning Navigation Trajectories in Latent Space. V-JEPA 2.1 enables 10× faster and more accurate navigation planning compared to (Bar et al., 2025). We show PCA visualizations of 8 denoising steps of the planned latent trajectory between a start frame and a goal frame.

Table 7 Open Loop Navigation Planning. We finetune V-JEPA 2.1 on robot navigation datasets and compare the open-loop planning performance to NWM. Reporting the average planning time (seconds), normalized Average Trajectory Error (ATE) and Relative Trajectory Error (RTE). V-JEPA 2.1 representation enables 10× speed-up without sacrificing performance.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Time</td><td colspan="2">Recon</td><td colspan="2">Tartan Drive</td><td colspan="2">Scand</td><td colspan="2">Sacson</td><td colspan="2">Average</td></tr><tr><td>ATE</td><td>RTE</td><td>ATE</td><td>RTE</td><td>ATE</td><td>RTE</td><td>ATE</td><td>RTE</td><td>ATE</td><td>RTE</td></tr><tr><td>NWM (Bar et al., 2025)</td><td>103.2 (s)</td><td>1.146</td><td>0.339</td><td>5.831</td><td>1.219</td><td>1.113</td><td>0.297</td><td>4.037</td><td>0.928</td><td>3.032</td><td>0.696</td></tr><tr><td>V-JEPA 2.1 ViT-g</td><td>10.6 (s)</td><td>1.146</td><td>0.338</td><td>5.758</td><td>1.200</td><td>1.094</td><td>0.299</td><td>3.904</td><td>0.921</td><td>2.975</td><td>0.690</td></tr><tr><td>V-JEPA 2.1 ViT-G</td><td>10.6 (s)</td><td>1.179</td><td>0.340</td><td>5.687</td><td>1.187</td><td>1.038</td><td>0.285</td><td>4.054</td><td>0.938</td><td>2.990</td><td>0.688</td></tr></table>

## 3.4 Navigation Planning

Next, we evaluate the utility of V-JEPA 2.1 representations for short-term robotic navigation. Specifically, given an agent’s most recent observation and a goal location specified by an image, we predict a 2-second navigation trajectory toward the goal. We adopt the navigation planning setup of NWM (Bar et al., 2025) and train a latent world model on top of the V-JEPA 2.1 representations. We find that V-JEPA 2.1 enables more accurate planning while achieving 10× faster planning speed than the previously used SD-VAE (Rombach et al., 2022). Results are reported in Table 7, with qualitative examples shown in Figure 9.

We train a Conditional Diffusion Transformer (CDiT) on top of the V-JEPA 2.1 representations, similarly to NWM. Our initial experiments indicate that directly training a diffusion model in the high-dimensional embedding space of the V-JEPA 2.1 is challenging: because the representations are high-dimensional (at least 80× larger than those of an SD-VAE), injecting noise in arbitrary directions can easily push samples off the data manifold. To improve robustness, we introduce two modifications. First, we train the model to predict a clean representation rather than noise (Li and He, 2025). Second, we use DDIM sampling (Song et al., 2020), which is less stochastic than DDPM (Ho et al., 2020) and empirically improves stability.

For planning, given an initial context embedding and a goal, we use the Cross-Entropy Method to sample N = 480 candidate trajectories of length 2 seconds at 4 FPS. We select the best candidate by simulating it in the world model and choosing the trajectory that minimizes the distance to the goal embedding. To simplify the search, we plan in a reduced 3-DoF action space (translation and yaw rotation). With V-JEPA 2.1, we require only 8 denoising steps, compared to the optimal SD-VAE setting which requires at least 128 steps. This yields lower Absolute Trajectory Error (ATE) and Relative Trajectory Error (RTE) on the validation set while reducing planning time by 10×.

Table 8 Performance on dense visual tasks. We report the Root Mean Squared Error (RMSE) for depth estimation (NYUv2, KITTI), mean Intersection-over-Union (mIoU) for semantic segmentation (ADE20K, Cityscapes, VOC12), and the J &F-Mean for video object segmentation (DAVIS, YouTube-VOS). Following Siméoni et al. (2025), ADE20K and VOC12 are evaluated at an input resolution of 448 × 448 for models with patch size 14 and 512 × 512 for those with patch size 16. Cityscapes, NYUv2, and KITTI use their default evaluation resolutions. For DAVIS and YouTube-VOS, we evaluate with the shorter image side set to 480 px (patch size 16) or 420 px (patch size 14). V-JEPA 2.1 VIT G demonstrates strong performance on dense vision tasks. Notably, it achieves state-of-art performances on NYUv2 depth estimation, outperforming a DINOv3 backbone with 7 billion parameters.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Param.</td><td colspan="2">Depth</td><td colspan="3">Segmentation</td><td colspan="2">VOS</td></tr><tr><td>NYUv2↓</td><td>KITTI↓</td><td>ADE20k</td><td>Citysc.</td><td>VOC</td><td>DAVIS-S</td><td>YT VOS-S</td></tr><tr><td colspan="9">7B models</td></tr><tr><td>Web-DINO (Fan et al., 2025)</td><td>7B/14</td><td>0.466</td><td>3.158</td><td>42.7</td><td>68.3</td><td>76.1</td><td>57.2</td><td>43.9</td></tr><tr><td>DINOv3 (Siméoni et al., 2025)</td><td>7B/16</td><td>0.309</td><td>2.346</td><td>55.9</td><td>81.1</td><td>86.6</td><td>71.1</td><td>74.1</td></tr><tr><td colspan="9">Models less than 2B parameters</td></tr><tr><td>PEcore (Bolya et al., 2025)</td><td>2B/14</td><td>0.590</td><td>4.119</td><td>38.9</td><td>61.1</td><td>69.2</td><td>48.2</td><td>34.7</td></tr><tr><td>PEspatial (Bolya et al., 2025)</td><td>2B/14</td><td>0.362</td><td>3.082</td><td>49.3</td><td>73.2</td><td>82.7</td><td>68.4</td><td>68.5</td></tr><tr><td>AM-RADIOv2.5 (Ranzinger et al., 2024)</td><td>1B/14</td><td>0.340</td><td>2.918</td><td>53.0</td><td>78.4</td><td>85.4</td><td>66.5</td><td>70.1</td></tr><tr><td>InternVideo2-1B (Wang et al., 2024b)</td><td>1B/14</td><td>0.471</td><td>4.739</td><td>28.4</td><td>35.6</td><td>65.2</td><td>50.6</td><td>51.2</td></tr><tr><td>SigLIP 2 (Tschannen et al., 2025)</td><td>1B/16</td><td>0.494</td><td>3.273</td><td>42.7</td><td>64.8</td><td>72.7</td><td>56.1</td><td>52.0</td></tr><tr><td>DINOv2 (Oquab et al., 2023)</td><td>1B/14</td><td>0.372</td><td>2.624</td><td>49.5</td><td>75.6</td><td>83.1</td><td>63.9</td><td>65.6</td></tr><tr><td>DINOv3 ViT-H+ (Siméoni et al., 2025)</td><td>0.8B/16</td><td>0.352</td><td>2.635</td><td>54.8</td><td>79.5</td><td>85.8</td><td>71.1</td><td>74.0</td></tr><tr><td>V-JEPA2 ViT-g (Assran et al., 2025)</td><td>1B/16</td><td>0.642</td><td>4.650</td><td>24.4</td><td>45.9</td><td>63.9</td><td>52.5</td><td>53.7</td></tr><tr><td>V-JEPA 2.1 ViT-g</td><td>1B/16</td><td>0.350</td><td>2.601</td><td>47.8</td><td>71.8</td><td>84.7</td><td>68.1</td><td>72.3</td></tr><tr><td>V-JEPA 2.1 ViT-G</td><td>2B/16</td><td>0.307</td><td>2.461</td><td>47.9</td><td>73.5</td><td>85.0</td><td>69.0</td><td>72.7</td></tr></table>

## 3.5 Depth Estimation and Semantic Segmentation

We evaluate the quality of the learned representations on semantic segmentation and depth estimation, two tasks that require fine-grained understanding of the image spatial structure.

Task. Semantic segmentation probes the model’s ability to capture category-level semantics and object boundaries. We evaluate our model on PASCAL VOC12 (Everingham et al., 2015), ADE20K (Zhou et al., 2017b), and Cityscapes (Cordts et al., 2016), reporting the mean Intersection over Union (mIoU). As in Siméoni et al. (2025), the input resolution of the images is adapted to 1024 patch tokens (i.e., 512 × 512 for patch size 16, 448 × 448 for patch size 14). On the other hand, depth estimation evaluates the model’s understanding of the scene’s geometric structure. For this task, we use the NYUv2 (Silberman et al., 2012a) and KITTI (Geiger et al., 2013b) benchmarks and report the Root Mean Squared Error (RMSE).

Evaluation protocol. Following the protocol of DINOv3 (Siméoni et al., 2025), we train a dense linear projection on top of the frozen final-layer features of the V-JEPA 2.1 encoder using the image patchifier. Importantly, we do not utilize intermediate layers from the encoder. As V-JEPA 2.1 adopts a 3D RoPe for positional encoding, the frequencies are interpolated according to the image resolution.

Results. Table 8 summarizes the performance of V-JEPA 2.1 and other visual encoders on depth estimation and single-image semantic segmentation. V-JEPA 2.1 ViT-G achieves state-of-the-art results on linear-probe monocular depth estimation, reaching 0.307 RMSE on NYUv2 and strong performance on KITTI with 2.461 RMSE, performing best across models that have less than 2 billion parameter. On

![](images/0d35eb82d447d2e2872a03cf479c6d6aee9f06c68834ca067612caa64a194a02.jpg)  
Figure 12 Qualitative comparative of V-JEPA 2.1 ViT-G vs DINOv3 ViT-H+. V-JEPA 2.1 captures fine-grained details (i.e the traffic light contour) and obtains better scale consistency. In the two rightmost columns, DINOv3 mislabel the depth of the green object on the bed, and the TV, and detect them closer to the camera than they are.

NYUv2, V-JEPA 2.1 surpasses prior image encoders, including DINOv3 ViT-7B (0.309 RMSE on NYU) and $\mathrm { P E _ { s p a t i a l } } \cdot$ -G (0.362 RMSE on NYU), and it represents a significant improvement over video encoders such as InternVideo2-1B (0.471 RMSE) and V-JEPA 2 (0.642 RMSE). Figure 10 provides a qualitative comparison of the depth maps predicted by V-JEPA 2 and V-JEPA 2.1. While V-JEPA 2 captures the overall scene geometry, the inconsistencies of its local features lead to noisy depth maps. In contrast, our novel V-JEPA 2.1 produces sharper and more coherent depth maps with well-defined object boundaries. We also visualize a more detailed qualitative comparative with DINOv3 ViT-H+ in Figure 12.

![](images/924ec78a7237590cb822d5b5894383a10de5ac4e5e60c470e8419d8ecdf8fa2d.jpg)

<details>
<summary>text_image</summary>

NYUv2
KITTI
Image
V-JEPA 2
V-JEPA 2.1
</details>

Figure 10 Depth estimation comparison on NYU and KITTI datasets. While V-JEPA 2 captures the overall scene geometry, its predictions lack local consistency and precise boundary structure. In contrast, our V-JEPA 2.1 produces sharper, more coherent, and fine-grained depth maps.

In semantic segmentation, V-JEPA 2.1 is also highly competitive with the state-of-the-art models, obtaining 85.0 mIoU on VOC12, 73.5 mIoU on Cityscapes and 47.9 mIoU on ADE20K. Compared with its previous version V-JEPA 2, the gains are remarkable across all datasets: +23.4 points in ADE20K, +27.6 on Cityscapes, and + 20.7 on VOC12; showing the benefits of explicitly supervising the context tokens and incorporating the multi-level predictor.

Performance on ADE20K and Cityscapes remains slightly behind the best image encoders. These datasets contain numerous object classes spanning large scale variations, with cluttered scene layouts that impose strict demands on fine-grained segmentation. We hypothesize that VisionMix contains comparatively fewer highly cluttered scenes, limiting exposure to the level of granularity required by such benchmarks. Figure 11 illustrates segmentation predictions on Cityscapes and VOC12. V-JEPA 2.1 yields detailed and spatially accurate masks, capturing multi-scale structures and fine object contours with high fidelity, showing competitive performance with state-of-the-art methods such as DINOv3 ViT-H+ (Siméoni et al., 2025).

## 3.6 Video Object Segmentation

We evaluate V-JEPA 2.1 on Video Object Segmentation (VOS) to assess the temporal consistency of its learned features.

Task. Video Object Segmentation consists in, given the ground-truth object mask in the first frame, propagating this mask accurately across all subsequent frames. This task evaluates the model’s ability to preserve long-range correspondences while remaining robust to camera motion, object deformation, and visual distractions. We evaluate on DAVIS 2017 (Pont-Tuset et al., 2017) and YouTube-VOS (Xu et al., 2018) datasets, reporting the standard J &F-mean metric (Perazzi et al., 2016) that jointly measures the region similarity and contour accuracy.

![](images/03562bb844b01988331f9c8984395b4bc335851df249798e19949f55b6f83da5.jpg)  
Figure 11 Semantic segmentation qualitative results in VOC12 and Cityscapes datasets. The previous version, V-JEPA 2, produced sparse semantic masks due to the presence noisy feature maps. However, V-JEPA 2.1 achieves competitive performance with state of the art methods such as DINOv3 ViT-H+.

Evaluation protocol. We adopt a non-parametric label propagation approach (Jabri et al., 2020) that matches local patch features across frames using cosine similarity in the embedding space. This procedure introduces no learnable parameters, making it a direct probe of the representation’s temporal stability. Following Siméoni et al. (2025), input videos are resized to produce a consistent number of patch tokens (short side of 420 px for patch size 14, and 480 px for patch size 16). On the training split of each dataset, we conduct a systematic search of the best hyper-parameters: maximum context length, neighborhood mask size, number of top-K nearest neighbors, and the temperature parameter used in similarity computation. The best configuration in DAVIS-S (15 context frames, circle mask of size 12, top-5 neighbors and temperature = 0.2), was applied to all the validation sets.

Results. V-JEPA 2.1 achieves 69.0 J &F on DAVIS-17 and 72.7 J &F on YouTube-VOS datasets, obtaining the second best performance and surpassing all prior encoders except DINOv3, which attains a slightly higher score. These results highlight the temporal consistency of the V-JEPA 2.1 features, which enable stable object tracking despite significant appearance changes across frames. Figure 13 illustrates two qualitative examples: even under fast motion and substantial visual variations, V-JEPA 2.1 maintains consistent object segmentation masks throughout the sequence.

## 3.7 Video and Image Classification

Real-world video reasoning requires recognizing static visual cues (i.e, objects, textures, scene layouts) as well as dynamic patterns (i.e, gestures, hand-object interactions, camera motion, long-term temporal evolution).

Initial GT frame  
![](images/38294ea52a775cd175b778df08b1fee9004dc16758e859640b7fac48fc555a3a.jpg)

![](images/84e0b7d903f60a12a1aef17622a8fb3fcc8eb37f79962011d7d436078e0aae23.jpg)

![](images/4552ab8f351de57174a8a513d9b6a86302b1d0c3827078b5f65f304a9a820438.jpg)

![](images/c2a986452fefe9aad82dc15b1effb156d39a078332285c7d67bc50b1383812c2.jpg)

![](images/33116968c89f6fe16704ba45cbdc0febac282cf5dd7bfeb501ba51be405ce6ff.jpg)

![](images/21439a3fad9b438f96e585b5c273b1a4b438eae52cd4cedd7d6f222a2b5ef096.jpg)

Frames sequence  
![](images/f27524f29cfbe4d53d0f94a9db5010741a278b6fc22816dd3782e7c0b6660223.jpg)

![](images/de7431b22db6ee1a8a105a0b58dc2d65323e8b0609153b4bcbe5f508cfba02cc.jpg)

![](images/3a0a3e8d8bff2a4b18bf61daf21fbc41e50e8346a86f8ecde72ab1aa86c0ef37.jpg)

![](images/a20b2fb2eaf4e33b50bafde7ebe944020c4af1fbe426ceca016d33e31c3a5d98.jpg)

![](images/8821fb9cb3f91eb28409d188036f5d9e193f3010b46e503c718f6b3357c659eb.jpg)

![](images/8396f35706f39d024499d9c3da4206e0aa00d5e2dd69dd9b96bb0049694baf7d.jpg)  
Figure 13 Video Object Segmentation qualitative examples. Given the ground-truth object masks for the initial frame, we propagate the instance masks to subsequent frames based on patch similarity within the V-JEPA 2.1 embedding space. All videos are processed at a resolution where the shorter side is set to 480 pixels.

Table 9 Performance on global tasks. We report Top-1 classification accuracy on action recognition benchmarks (SSv2, Diving-48, and K400) and image classification (IN1K). Our protocol uses a resolution of 256 × 256 by default, and 384 × 384 for V-JEPA 2 ViT-g and V-JEPA 2.1; and uses $1 6 \times 2 \times 3$ inputs for SSv2 (16-frame clips, 2 temporal crops, 3 spatial crops), $1 6 \times 8 \times 3$ for K400, and $3 2 \times 4 \times 3$ for Diving-48. V-JEPA 2.1 VIT-G achieves state-of-art performance on the SSv2 action recognition asks while being competitive on appearance understanding tasks.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Param.</td><td colspan="2">Motion Understanding</td><td colspan="2">Appearance Understanding</td></tr><tr><td>SSv2</td><td>Diving-48</td><td>K400</td><td>IN1K</td></tr><tr><td colspan="6">Results Reported in the Literature</td></tr><tr><td>VideoMAEv2 (Wang et al., 2023)</td><td>1B</td><td>56.1</td><td>-</td><td>82.8</td><td>71.4</td></tr><tr><td>InternVideo2-1B (Wang et al., 2024b)</td><td>1B</td><td>67.3</td><td>-</td><td>87.9</td><td>-</td></tr><tr><td>VideoPrism (Zhao et al., 2024)</td><td>1B</td><td>68.5</td><td>71.3</td><td>87.6</td><td>-</td></tr><tr><td>DINOv3 ViT-H+ (Siméoni et al., 2025)</td><td>0.8B</td><td>69.8</td><td>-</td><td>86.7</td><td>87.9</td></tr><tr><td>DINOv3 (Siméoni et al., 2025)</td><td>7B</td><td>70.1</td><td>-</td><td>87.8</td><td>88.4</td></tr><tr><td colspan="6">Image Encoders Evaluated Using the Same Protocol</td></tr><tr><td>DINOv2 (Darcet et al., 2023)</td><td>1.1B</td><td>50.7</td><td>82.5</td><td>83.6</td><td>86.1</td></tr><tr><td>PEcore G (Bolya et al., 2025)</td><td>1.9B</td><td>55.4</td><td>76.9</td><td>88.5</td><td>87.6*</td></tr><tr><td>SigLIP2 (Tschannen et al., 2025)</td><td>1.2B</td><td>49.9</td><td>75.3</td><td>87.3</td><td>88.0</td></tr><tr><td colspan="6">Video Encoders Evaluated Using the Same Protocol</td></tr><tr><td>InternVideo22s-1B (Wang et al., 2024b)</td><td>1B</td><td>69.7</td><td>86.4</td><td>89.4</td><td>85.8</td></tr><tr><td>V-JEPA ViT-H (Bardes et al., 2024)</td><td>600M</td><td>74.3</td><td>87.9</td><td>84.5</td><td>80.0</td></tr><tr><td>V-JEPA 2 ViT-g (Assran et al., 2025)</td><td>1B</td><td>77.3</td><td>90.2</td><td>87.3</td><td>85.1</td></tr><tr><td>V-JEPA 2.1 ViT-g</td><td>1B</td><td>76.9</td><td>89.0</td><td>87.0</td><td>84.8</td></tr><tr><td>V-JEPA 2.1 ViT-G</td><td>2B</td><td>77.7</td><td>89.2</td><td>87.7</td><td>85.5</td></tr></table>

To capture this dual nature, we evaluate the representations learned during pretraining on four complementary classification datasets. Something-Something v2 (Goyal et al., 2017) and Diving-48 (Li et al., 2018) assess motion-centric understanding, as their labels require modeling multi-frame dynamics to correctly identify the action. In contrast, Kinetics-400 (Kay et al., 2017) and ImageNet-1K (Deng et al., 2009) primarily measure appearance-based recognition, since many of their classes can be predicted from a single frame, even when the label describes an action.

Task. We compare the video classification performance of V-JEPA 2.1 against a broad set of visual encoders. For image-based self-supervised models, we include DINOv2 with registers (Darcet et al., 2023) and the more recent DINOv3 (Siméoni et al., 2025), both representing state-of-the-art in image-only pretraining. We further evaluate against leading image–text contrastive approaches, including SigLIP2 (Tschannen et al., 2025) and the Perception Encoder $\mathrm { P E } _ { \mathrm { c o r e } } \mathrm { G }$ (Bolya et al., 2025). For video models, we benchmark against the previous V-JEPA (Bardes et al., 2024) and V-JEPA 2 (Assran et al., 2025), as well as InternVideo2s2-1B (Wang et al., 2024b), a state-of-the-art video encoder trained primarily through vision–text contrastive objectives.

In addition, we report comparisons with results available in the literature for VideoMAEv2 (Wang et al., 2023), InternVideo2-1B (Wang et al., 2024b), VideoPrism (Zhang et al., 2024), and DINOv3 (Siméoni et al.,

Table 10 VideoQA results. We report the performance of VJEPA 2.1 trained with Llama 3.1 8B backbone on several popular temporal video understanding datasets. We train on a modified subset of the PerceptionLM (Cho et al., 2025) data, and compare to VJEPA 2 (Assran et al., 2025) by training it on the same data regime. VJEPA 2.1 achieves comparatively better performance than VJEPA 2 on several video understanding tasks which require both rich visual information and temporal understanding. For PerceptionTest (Pătrăucean et al., 2023), we report the validation accuracy, as the test server to compute the test accuracy (indicated by \*) is no longer active.

<table><tr><td>Method</td><td>ParamsEnc / LLM</td><td>Avg.</td><td>PerceptionTestTest Acc</td><td>MVPPaired-Acc</td><td>TempCompassmulti-choice</td><td>TemporalBench(MBA-short QA)</td><td>TOMATOAcc</td><td>TVBenchAcc</td><td>MVBenchAcc</td></tr><tr><td colspan="10">≤8B Video Language Models Results Reported in the Literature</td></tr><tr><td>InternVL-2.5 (Chen et al., 2024)</td><td>300M/7B</td><td>52.1</td><td>68.9</td><td>39.9</td><td>68.3</td><td>24.3</td><td>29.4</td><td>61.6</td><td>72.6</td></tr><tr><td>Qwen2VL (Wang et al., 2024a)</td><td>675M/7B</td><td>47.0</td><td>66.9</td><td>29.2</td><td>67.9</td><td>20.4</td><td>31.5</td><td>46.0</td><td>67.0</td></tr><tr><td>Qwen2.5VL (Qwen Team et al., 2025)</td><td>1B/7B</td><td>49.7</td><td>70.5</td><td>36.7</td><td>71.7</td><td>24.5</td><td>24.6</td><td>50.5</td><td>69.6</td></tr><tr><td>PLM 8B (Cho et al., 2025)</td><td>1B/8B</td><td>56.7</td><td>82.7*</td><td>39.7</td><td>72.7</td><td>28.3</td><td>33.2</td><td>63.5</td><td>77.1</td></tr><tr><td>VJEPA 2 ViT-g384 (Assran et al., 2025)</td><td>1B/8B</td><td>59.5</td><td>84.0*</td><td>44.5</td><td>76.9</td><td>36.7</td><td>40.3</td><td>60.6</td><td>73.5</td></tr><tr><td colspan="10">Models trained in this paper using filtered Perception LM data (Cho et al., 2025)</td></tr><tr><td>VJEPA 2 ViT-g384</td><td>1B/8B</td><td>57.8</td><td>80.1</td><td>40.9</td><td>74.1</td><td>32.3</td><td>41.4</td><td>62.6</td><td>73.3</td></tr><tr><td>VJEPA 2.1 ViT-G384</td><td>2B/8B</td><td>57.9</td><td>83.1</td><td>43.2</td><td>74</td><td>28.5</td><td>38</td><td>62.8</td><td>75.4</td></tr></table>

2025). Note that these models are evaluated under similar frozen-probe protocols, but their attentive heads differ from ours. For instance, DINOv3 augments its probe with explicit spatial and temporal positional embeddings and applies a 3D factorized RoPE across its attention blocks. In contrast, our V-JEPA 2.1 encoder already encodes spatio–temporal structure in its features, allowing our probe to operate without such additional components.

Evaluation protocol. Following the protocol proposed by Assran et al. (2025), we train a 4-layers attentive probe on top of the frozen encoder features using the respective training data from each task. The attentive probe consists of four transformer blocks followed by a final cross-attention mechanism that employs a learnable query token. At inference, we sampled several clips with a fixed number of frames from the video, and we averaged the classification logits across clips.

Results. Table 9 summarizes the global video understanding performance of V-JEPA 2.1, previous V-JEPA models, and other strong visual encoders. V-JEPA 2.1 ViT-G achieves a top-1 accuracy of 77.7 on SSv2, setting a new absolute state-of-the-art on the task compared to 77.5 for InternVideo2 full fine-tuning, 77.3 for V-JEPA 2, 70.1 for DINOv3 ViT-7B and 69.7 for InternVideo2 using the same protocol, demonstrating a strong understanding of video dynamics. Our model is also highly-competitive in appearance-based tasks, reaching 87.7 on K400 and 85.5 on IN1K. These results show that properly designing the loss over context tokens, together with deep self-supervision, enables fine-grained dense features that also capture strong global scene understanding. We also observe consistent improvements with model scaling from ViT-g to ViT-G with an average improvement of +0.6 points.

## 3.8 Video Question Answering

In this section, we evaluate V-JEPA 2.1 on Video Question Answering (VidQA), by training a Video Large Language Model (VidLLM) using V-JEPA 2.1 encoder and Llama 3.1 8B LLM as the backbone, following the recipe proposed by Assran et al. (2025).

Task. We compare the VidQA performance of V-JEPA 2.1 with popular open-source multimodal models reported in the literature, where the LLM backbone size is ≤ 8B. Following the protocol set of Assran et al. (2025), we report performance on video understanding benchmarks which require rich visual and temporal understanding, such as PerceptionTest (Pătrăucean et al., 2023), Minimal Video Pairs (MVP) (Krojer et al., 2024), TempCompass (Liu et al., 2024), TemporalBench (Cai et al., 2024), TOMATO (Shangguan et al., 2024) and MVBench (Li et al., 2024).

Evaluation protocol. We train our Video-LLM model using a subset of PerceptionLM that is publicly available (Cho et al., 2025). Starting with this data, we perform data quality filtering using domain filters, vision-text misalignment filters, and an external multimodal LLM, Qwen3VL, as a data-quality judge, assigning scores to each data samples. This process resulted us to filter out and remove 18% of the training data. It is important to note that we do not rephrase any data from PerceptionLM (Cho et al., 2025) using this setup - we only filter the data based on the model’s judgement.

We reproduced LLM alignment of VJEPA 2 on this new data, and compare our model, VJEPA 2.1, with it. We follow the same three-stage training procedure from Assran et al. (2025) to train VJEPA 2.1 LLM, starting from image-text captioning to progressively training on higher quality image and video-text captioning and QA data. We further add a fourth stage of alignment, where we expand the context length of the base LLM to support 64 frames of video input, compared to 32 frames supported by VJEPA 2 (Assran et al., 2025). We primarily use VJEPA 2.1 in video-only mode in our experiments, to better compare with VJEPA 2. Interestingly, we find adding a final post-training stage using PerceptionTest training set further improves performance on all downstream tasks.

Results. Table 10 summarizes the results on VidQA evaluation tasks. Comparing VJEPA 2 and VJEPA 2.1 on the same data, we find on average VJEPA 2.1 to be slightly better, with significant improvements on PerceptionTest, Minimal Video Pairs, and crucially, improvement on datasets requiring rich visual semantics - MVBench and TVBench. We observe VJEPA 2.1 to be worse on TemporalBench and TOMATO - both datasets requiring understanding of the motion events, but on shorter videos.

Compared to results reported in the literature, VJEPA 2.1 slightly underperforms results reported Assran et al. (2025) which uses significantly alignment data (88.5 million samples in (Assran et al., 2023) vs 72.5 million samples in VJEPA 2.1) Nevertheless, VJEPA 2.1 is competitive with state-of-the-art, outperforming most open-source multimodal LLMs with similar model size.

## 3.9 Qualitative results of V-JEPA 2.1 Dense Features

Figure 14 presents a comparison of dense feature maps obtained from a single image using V-JEPA 2.1 ViT-G, DINOv2 ViT-g (Oquab et al., 2023), DINOv3 ViT-H+ (Siméoni et al., 2025) and V-JEPA 2 ViT-g (Assran et al., 2025). We project each encoder’s feature space into three dimensions using PCA. Following Siméoni et al. (2025), we permute the three components across the six possible RGB channels permutations, and we display the most visually appealing. The qualitative results in Figure 14 illustrate the high-quality of V-JEPA 2.1 dense feature maps obtained from a single image, which show fine-grained and precise identification of objects, such as the cat’s fur details or pedestrians in Cityscapes scenes. Figure 15 further extends this comparison to full video sequences of 64 frames, showing temporal consistent dense maps across diverse scenarios, including egocentric videos from Ego4D (Grauman et al., 2022), Cityscapes sequences (Cordts et al., 2016), Diving48 videos (Li et al., 2018) and animal camouflage footage from MOC (Lamdouar et al., 2020).

## 3.10 Family of distilled models

We follow the same protocol to evaluate our distilled ViT-L and ViT-B models. Table 11 presents our results, where we observe a significant improvement in all our benchmarks between the ViT-L trained from scratch and the ViT-L distilled from ViT-G, almost closing the gap with ViT-G performance. On action recognition on SSv2, performance improves from 74.2% to 76.5%, almost reaching 77.7% from ViT-G. In image segmentation on ADE20k, performance improves from 42.0 mIoU to 46.7 mIoU. In depth estimation and video segmentation, the ViT-L distilled is even closer to ViT-G, with 2.490 RMSE on KITTI, close to 2.461 RMSE, and 68.7 J &F-Mean on DAVIS, close to 67.0 J &F-Mean. Finally, our ViT-B offers competitive performance with our ViT-L trained from scratch.

## 4 Related work

Self-Supervised Learning. Early SSL approaches in vision focus on simple pretext tasks such as predicting the relative position of image patches (Doersch et al., 2015), reordering shuffled patches (Noroozi and Favaro,

![](images/5aabc6ce2892fc6240a1355b3c3c4db80f26ed428a16260ad648107fff6a4b72.jpg)  
Figure 14 Comparison of dense features produced by different SSL methods on a single image. We compare DINOv2 ViT-g (Oquab et al., 2023), DINOv3 ViT-H+ (Siméoni et al., 2025) and V-JEPA 2 ViT-g (Assran et al., 2025) against V-JEPA 2.1 ViT-G. Images are resized so that their shorter side is set to 1024 pixels, after which they are processed using the 2D convolution image tokenizer.

![](images/ca73e57103f18d16da200d091200fa3f6a1d077bfa4a7daf85fd7718463a8bb2.jpg)  
Figure 15 V-JEPA 2.1 unlocks dense features from video. Qualitative results on Ego4D (1080px), Cityscapes (1080px), Diving48 (768px), and MOCA (768px) illustrate the strong temporal consistency of the dense features produced by our ViT-G model, particularly on dynamic objects. Since we process entire video sequences, we employ the 3D convolutional video tokenizer.

Table 11 Benefits of Distillation. Distilling the V-JEPA 2.1 ViT-G into smaller models (ViT-L, 300M; ViT-B, 80M) yields consistent improvements over training from scratch and approaches the performance of the ViT-G teacher.

<table><tr><td>Model</td><td>#Params</td><td>Distilled</td><td>SSv2</td><td>K400</td><td>IN1K</td><td>ADE20K</td><td>Citys.</td><td>VOC12</td><td>NYU (↓)</td><td>KITTI (↓)</td><td>DAVIS</td></tr><tr><td>ViT-G</td><td>2B</td><td></td><td>77.7</td><td>87.7</td><td>85.5</td><td>47.9</td><td>73.5</td><td>85.0</td><td>0.307</td><td>2.461</td><td>69.0</td></tr><tr><td>ViT-L</td><td>300M</td><td></td><td>74.2</td><td>84.3</td><td>81.8</td><td>42.0</td><td>66.7</td><td>79.6</td><td>0.381</td><td>2.914</td><td>64.5</td></tr><tr><td>ViT-L</td><td>300M</td><td>√</td><td>76.5</td><td>86.9</td><td>84.9</td><td>46.7</td><td>71.0</td><td>83.9</td><td>0.336</td><td>2.490</td><td>68.7</td></tr><tr><td>ViT-B</td><td>80M</td><td>√</td><td>74.5</td><td>85.1</td><td>84.0</td><td>42.0</td><td>64.5</td><td>78.0</td><td>0.408</td><td>2.850</td><td>67.0</td></tr></table>

2016; Misra and Maaten, 2020), inpainting missing regions (Pathak et al., 2016), re-colorizing grayscale images (Zhang et al., 2016), or predicting applied image transformations (Gidaris et al., 2018). Beyond handcrafted tasks, view-invariant approaches proposed to use a joint-embedding architecture to learn invariances to visual transformations, which led to significant advances in SSL, with contrastive approaches (Hénaff et al., 2019; He et al., 2020; Chen et al., 2020), non-contrastive approaches (Chen and He, 2020; Grill et al., 2020b; Bardes et al., 2021), or clustering-based techniques (Caron et al., 2018, 2020). Later, the adoption of vision transformers (Dosovitskiy, 2020) became the standard in self-supervised learning, first by revisiting existing view-invariant approaches (Chen et al., 2021; Caron et al., 2021), and then with masked image modeling approaches, which can operate in pixel space (Xie et al., 2021b; Wei et al., 2021), or in the latent space of the transformer (Bao et al., 2021). Most powerful approaches employ a combination of view-invariant and masked image-modeling techniques (Zhou et al., 2021; Oquab et al., 2023). The concept of learning representations by predicting missing information in latent space is formalized by the Joint-Embedding Predictive Architecture (JEPA) (LeCun, 2022), which have been successfully applied across multiple modalities, including audio (Baevski et al., 2022), images (Assran et al., 2023; Bar et al.), and video (Bardes et al., 2024).

Video Models. Motion cues from videos have inspired many early video pretext tasks. Early methods predicted camera transformations between image pairs (Agrawal et al., 2015) or future frames from egomotion (Jayaraman and Grauman, 2015). Others brought consecutive frame patches closer in feature space (Wang and Gupta, 2015) or used unsupervised object retrieval for segmentation (Pathak et al., 2017). The rise a large labeled video datasets such as Kinetics (Carreira and Zisserman, 2017) changed the focus of the community towards supervised video models. Early approaches used image sequences with late fusion (Donahue et al., 2015) or 3D convolutions (Tran et al., 2015), but 3D ConvNets are computationally expensive. Recent work improves efficiency with 2D spatial and 1D temporal convolutions (Tran et al., 2018; Xie et al., 2018), dual-pathways (Feichtenhofer et al., 2019), and vision transformers adapted for video (Arnab et al., 2021; Bertasius et al., 2021). Despite rapid progress, supervised learning requires extensive labeled data and may not fully exploit temporal information, and the focus went back to self-supervised learning.

Early self-supervised learning methods focused on temporal order verification (Misra et al., 2016; Xu et al., 2019), contrastive learning (Han et al., 2020; Dave et al., 2022), and future prediction (Han et al., 2019). Recent masked modeling approaches, like VideoMAE (Tong et al., 2022; Feichtenhofer et al., 2022), extend image-based masking to video. Other advances include masked modeling in CLIP space (Li et al., 2023), joint image-video training (OmniMAE (Girdhar et al., 2023)), and architectural innovations such as hierarchical transformers (Ryali et al., 2023) and decoupled encoders/decoders (Gupta et al., 2023). Despite progress, defining optimal pretext tasks and efficient video processing remain open challenges.

Scaling existing approaches, both in terms of data and model size, has demonstrated that generalist video encoders can be learned from extensive observation datasets of videos using self-supervised learning (Carreira et al., 2024; Wang et al., 2023; Rajasegaran et al., 2025). In particular, Joint-Embedding Predictive architectures show promising efficiency gains at large scale compared to generative approaches (Bardes et al., 2024), and open the door to applications in world modeling (Assran et al., 2025). Self-supervised models can also incorporated weak language supervision (Zhao et al., 2024; Wang et al., 2024b; Bolya et al., 2025), unlocking language-based tasks.

Learning dense features. Learning dense features is often achieved through designing local loss functions and objectives, such as leveraging spatio-temporal consistency in videos (Jabri et al., 2020), spatial alignment between image crops (Pinheiro et al., 2020; Bardes et al., 2022), and patch consistency (Yun et al., 2022). Contrastive learning approaches such as DetCon (Hénaff et al., 2021) and ORL (Xie et al., 2021a) use region proposals, while newer methods relax this requirement (Hénaff et al., 2022; Wen et al., 2022). Recent distillation-based methods combine strengths from multiple encoders, such as AM-RADIO (Ranzinger et al., 2024), Perception Encoder (Bolya et al., 2025), and DINOv3 (Siméoni et al., 2025), using objectives like cosine similarity and Gram matrix regularization to improve dense features. Some works focus on post-hoc improvements, including fine-tuning with clustering objectives (Ziegler and Asano, 2022), patch alignment (Salehi et al., 2023), and patch-sorting (Pariza et al., 2025). Others enhance features without fine-tuning, such as STEGO (Hamilton et al., 2022) and feature augmentation (Simoncini et al., 2024).

Beyond playing with the loss function, the standard Vision Transformer (ViT) (Dosovitskiy et al., 2021) can be adapted for dense prediction tasks, Ranftl et al. (2021) introduced the Dense Prediction Transformer (DPT), evolving from their prior CNN-based work on MiDaS (Ranftl et al., 2020). DPT solved the token-to-pixel problem using a "reassemble" operation that converts the 1D ViT sequence into multi-scale 2D feature maps, which are then fused by a convolutional decoder adapted from RefineNet (Lin et al., 2017a). This design, which leverages the global receptive field of the transformer backbone, proved effective for obtaining structural coherence, moving beyond the limitations of purely hierarchical CNNs. The work also catalyzed parallel research, including the development of purely transformer-based decoders like Segmenter (Strudel et al., 2021) and hierarchical transformer backbones like the Swin Transformer (Liu et al., 2021) and the Multiscale Vision Transformer (Fan et al., 2021). DPT architectural framework has proven to be the most influential design for scaling up, serving as the basis for current state-of-the-art foundation models like Depth Anything (Yang et al., 2024).

## 5 Conclusion and Future work

This work demonstrates how Joint-Embedding Predictive Architectures and Self-Supervised Learning from video unlock high-quality dense local features with strong global semantic understanding. Our key ingredient is a novel deep spatio-temporal self-supervision, composed by a weighted context loss and a multi-level predictor that enables supervision across the intermediate encoder layers. This design unlocks high-quality dense features that are spatially structured, semantically coherent and temporally consistent, as evidenced by PCA visualizations. Beyond the training objective, we demonstrate the importance of modality-specific tokenizers, large-scale and balanced image–video data curation, and scaling to ViT-G, which together enable an effective distillation to compact models. Our V-JEPA 2.1 ViT-G achieves absolute state-of-the-art performance in short-term object interaction anticipation, action anticipation and action recognition, showcasing its excellent predictive and motion understanding capabilities. Moreover, the dense feature maps achieve linear-probe state-of-the-art performance in monocular depth estimation and strong results in semantic segmentation and video object tracking. We hope these findings encourage further research on predictive physical world modeling.

Future work. Future work will focus on several promising directions. First, scaling along two axes: a) model size, we observe a very positive trend scaling from 1B to 2B and other work such as DINOv3 showed the benefit of scaling to 7B in vision; b) data size, our work on data curation highlighted that video self-supervised learning really benefit from large-scale datasets. Second, this work focus on learning better representation, whereas V-JEPA 2 showed the potential of building world models on top of these representations. Future work in this direction will explore world modeling aspects with the prism of learning dense prediction capabilities (Karypidis et al., 2024; Bojanowski, 2025). Finally, world models with dense understanding and prediction abilities will unlock many applications in robotics and autonomous agents, where a precise estimation of the state down to the pixel level is required, for example navigation in challenging real world environment, or fine-grained manipulation.

## References

Pulkit Agrawal, Joao Carreira, and Jitendra Malik. Learning to see by moving. In ICCV, 2015.  
Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Lucic, and Cordelia Schmid. Vivit: A video vision transformer. In ICCV, 2021.  
Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023.  
Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.  
Alexei Baevski, Wei-Ning Hsu, Qiantong Xu, Arun Babu, Jiatao Gu, and Michael Auli. Data2vec: A general framework for self-supervised learning in speech, vision and language. arXiv preprint arXiv:2202.03555, 2022.  
Randall Balestriero and Yann LeCun. Lejepa: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025.  
Hangbo Bao, Li Dong, and Furu Wei. Beit: Bert pre-training of image transformers. arXiv preprint arXiv:2106.08254, 2021.  
Amir Bar, Florian Bordes, Assaf Shocher, Mido Assran, Pascal Vincent, Nicolas Ballas, Trevor Darrell, Amir Globerson, and Yann LeCun. Stochastic positional embeddings improve masked image modeling. In Forty-first International Conference on Machine Learning.  
Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 15791–15801, June 2025.  
Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906, 2021.  
Adrien Bardes, Jean Ponce, and Yann LeCun. Vicregl: Self-supervised learning of local visual features. NeurIPS, 2022.  
Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.  
Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In ICML, 2021.  
Federico Baldassarre Marc Szafraniec Basile Terver Vasil Khalidov Francisco Massa Yann LeCun Patrick Labatut Maximilian Seitzer Piotr Bojanowski. Back to the features: Dino as a foundation for video world models. arXiv preprint arXiv:2507.19468, 2025.  
Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, et al. Perception encoder: The best visual embeddings are not at the output of the network. arXiv preprint arXiv:2504.13181, 2025.  
Mu Cai, Reuben Tan, Jianrui Zhang, Bocheng Zou, Kai Zhang, Feng Yao, Fangrui Zhu, Jing Gu, Yiwu Zhong, Yuzhang Shang, Yao Dou, Jaden Park, Jianfeng Gao, Yong Jae Lee, and Jianwei Yang. Temporalbench: Benchmarking fine-grained temporal understanding for multimodal video models. arXiv preprint arXiv:2410.10818, 2024.  
Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning. In ECCV, 2018.  
Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In NeurIPS, 2020.  
Mathilde Caron, Hugo Touvron, Ishan Misra, Herve Jegou, and Julien Mairal Piotr Bojanowski Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.  
Joao Carreira, Eric Noland, Chloe Hillier, and Andrew Zisserman. A short note on the kinetics-700 human action dataset. arXiv preprint arXiv:1907.06987, 2019.  
João Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In CVPR, 2017.  
João Carreira, Dilara Gokay, Michael King, Chuhan Zhang, Ignacio Rocco, Aravindh Mahendran, Thomas Albert Keck, Joseph Heyward, Skanda Koppula, Etienne Pot, Goker Erdogan, Yana Hasson, Yi Yang, Klaus Greff, Guillaume Le Moing, Sjoerd van Steenkiste, Daniel Zoran, Drew A. Hudson, Pedro Vélez, Luisa Polanía, Luke Friedman, Chris Duvarney, Ross Goroshin, Kelsey Allen, Jacob Walker, Rishabh Kabra, Eric Aboussouan, Jennifer Sun, Thomas Kipf, Carl Doersch, Viorica Pătrăucean, Dima Damen, Pauline Luc, Mehdi S. M. Sajjadi, and Andrew Zisserman. Scaling 4d representations. arXiv preprint arXiv:2412.15212, 2024.  
Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey E. Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.  
Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR, 2020.  
Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In ICCV, 2021.  
Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, Lixin Gu, Xuehui Wang, Qingyun Li, Yimin Ren, Zixuan Chen, Jiapeng Luo, Jiahao Wang, Tan Jiang, Bo Wang, Conghui He, Botian Shi, Xingcheng Zhang, Han Lv, Yi Wang, Wenqi Shao, Pei Chu, Zhongying Tu, Tong He, Zhiyong Wu, Huipeng Deng, Jiaye Ge, Kai Chen, Kaipeng Zhang, Limin Wang, Min Dou, Lewei Lu, Xizhou Zhu, Tong Lu, Dahua Lin, Yu Qiao, Jifeng Dai, and Wenhai Wang. Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271, 2024.  
Jang Hyun Cho, Andrea Madotto, Effrosyni Mavroudi, Triantafyllos Afouras, Tushar Nagarajan, Muhammad Maaz, Yale Song, Tengyu Ma, Shuming Hu, Suyog Jain, Miguel Martin, Huiyu Wang, Hanoona Rasheed, Peize Sun, Po-Yao Huang, Daniel Bolya, Nikhila Ravi, Shashank Jain, Tammy Stark, Shane Moon, Babak Damavandi, Vivian Lee, Andrew Westbury, Salman Khan, Philipp Krähenbühl, Piotr Dollár, Lorenzo Torresani, Kristen Grauman, and Christoph Feichtenhofer. Perceptionlm: Open-access data and models for detailed visual understanding. arXiv preprint arXiv:2504.13180, 2025.  
Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele. The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.  
Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Antonino Furnari, Jian Ma, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, and Michael Wray. Rescaling egocentric vision: Collection, pipeline and challenges for epic-kitchens-100. International Journal of Computer Vision (IJCV), 2022.  
Timothée Darcet, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Vision transformers need registers. arXiv preprint arXiv:2309.16588, 2023.  
Ishan Dave, Rohit Gupta, Mamshad Nayeem Rizve, and Mubarak Shah. Tclr: Temporal contrastive learning for video representation. Computer Vision and Image Understanding, 2022.  
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pages 248–255. Ieee, 2009.  
Carl Doersch, Abhinav Gupta, and Alexei A. Efros. Unsupervised visual representation learning by context. In ICCV, 2015.  
Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, and Kate Saenko. Long-term recurrent convolutional networks for visual recognition and description. In CVPR, 2015.  
Alexey Dosovitskiy. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.  
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mohammad Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.  
Mark Everingham, SM Ali Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. The pascal visual object classes challenge: A retrospective. International journal of computer vision, 111(1):98–136, 2015.  
David Fan, Shengbang Tong, Jiachen Zhu, Koustuv Sinha, Zhuang Liu, Xinlei Chen, Michael Rabbat, Nicolas Ballas, Yann LeCun, Amir Bar, and Saining Xie. Scaling language-free visual representation learning. arXiv preprint arXiv:2504.01017, 2025.  
Haoqi Fan, Bo Xiong, Karttikeya Mangalam, Yanghao Li, Zhicheng Yan, and Jitendra Malik. Multiscale vision transformers. In ICCV, 2021.  
Christoph Feichtenhofer, Haoqi Fan, Jitendra Malik, and Kaiming He. Slowfast networks for video recognition. In ICCV, 2019.  
Christoph Feichtenhofer, Haoqi Fan, Yanghao Li, and Kaiming He. Masked autoencoders as spatiotemporal learners. In NeurIPS, 2022.  
Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, and Yann LeCun. Intuitive physics understanding emerges from self-supervised pretraining on natural videos. arXiv preprint arXiv:2502.11831, 2025.  
Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. In The International Journal of Robotics Research, 2013a.  
Andreas Geiger, Philip Lenz, Christoph Stiller, and Raquel Urtasun. Vision meets robotics: The kitti dataset. The international journal of robotics research, 32(11):1231–1237, 2013b.  
S. Gidaris, P. Singh, and N. Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018.  
Rohit Girdhar, Alaaeldin El-Nouby, Mannat Singh, Kalyan Vasudev Alwala, Armand Joulin, and Ishan Misra. Omnimae: Single model masked pretraining on images and videos. In CVPR, 2023.  
Ross Girshick. Fast r-cnn. In Proceedings of the IEEE international conference on computer vision, pages 1440–1448, 2015.  
Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The" something something" video database for learning and evaluating visual common sense. In Proceedings of the IEEE international conference on computer vision, pages 5842–5850, 2017.  
Kristen Grauman, Andrew Westbury, Eugene Byrne, Zachary Chavis, Antonino Furnari, Rohit Girdhar, Jackson Hamburger, Hao Jiang, Miao Liu, Xingyu Liu, et al. Ego4d: Around the world in 3,000 hours of egocentric video. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18995–19012, 2022.  
Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020a.  
Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre H. Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Daniel Guo, Mohammad Gheshlaghi Azar, Bilal Piot, Koray Kavukcuoglu, Rémi Munos, and Michal Valko. Bootstrap your own latent: A new approach to self-supervised learning. In NeurIPS, 2020b.  
Ankush Gupta, Viorica Patraucean, Lucas Smaira, Larisa Markeeva, Menglin Chen, Teli He, Dylan Banarse, Antoine Miech, Alex Frechette, Raphael Koster, et al. Decoupled video coding with decoupled video prediction. arXiv preprint arXiv:2307.13532, 2023.  
David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2(3), 2018.  
Mark Hamilton, Zhoutong Zhang, Bharath Hariharan, Noah Snavely, and William T. Freeman. Unsupervised semantic segmentation by distilling feature correspondences. In ICLR, 2022.  
Tengda Han, Weidi Xie, and Andrew Zisserman. Video representation learning by dense predictive coding. In Workshop on Large Scale Holistic Video Understanding, ICCV, 2019.  
Tengda Han, Weidi Xie, and Andrew Zisserman. Self-supervised co-training for video representation learning. In NeurIPS, 2020.  
Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, 2020.  
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.  
Olivier J. Hénaff, Aravind Srinivas, Jeffrey De Fauw, Ali Razavi, Carl Doersch, S. M. Ali Eslami, and Aäron van den Oord. Data-efficient image recognition with contrastive predictive coding. In ICML, 2019.  
Olivier J. Hénaff, Skanda Koppula, Jean-Baptiste Alayrac, Aaron van den Oord, Oriol Vinyals, and João Carreira. Efficient visual pretraining with contrastive detection. arXiv preprint arXiv:2103.10957, 2021.  
Olivier J. Hénaff, Skanda Koppula, Evan Shelhamer, Daniel Zoran, Andrew Jaegle, Andrew Zisserman, João Carreira, and Relja Arandjelović. Object discovery and representation networks. In ECCV, 2022.  
Allan Jabri, Andrew Owens, and Alexei Efros. Space-time correspondence as a contrastive random walk. Advances in neural information processing systems, 33:19545–19560, 2020.  
D. Jayaraman and K. Grauman. Learning image representations tied to ego-motion. In ICCV, 2015.  
Efstathios Karypidis, Ioannis Kakogeorgiou, Spyros Gidaris, , and Nikos Komodakis. Dino-foresight: Looking into the future with dino. arXiv preprint arXiv:2412.11673, 2024.  
Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017.  
Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, Peter David Fagan, Joey Hejna, Masha Itkina, Marion Lepert, Yecheng Jason Ma, Patrick Tree Miller, Jimmy Wu, Suneel Belkhale, Shivin Dass, Huy Ha, Arhan Jain, Abraham Lee, Youngwoon Lee, Marius Memmel, Sungjae Park, Ilija Radosavovic, Kaiyuan Wang, Albert Zhan, Kevin Black, Cheng Chi, Kyle Beltran Hatch, Shan Lin, Jingpei Lu, Jean Mercat, Abdul Rehman, Pannag R Sanketi, Archit Sharma, Cody Simpson, Quan Vuong, Homer Rich Walke, Blake Wulfe, Ted Xiao, Jonathan Heewon Yang, Arefeh Yavary, Tony Z. Zhao, Christopher Agia, Rohan Baijal, Mateo Guaman Castro, Daphne Chen, Qiuyu Chen, Trinity Chung, Jaimyn Drake, Ethan Paul Foster, Jensen Gao, Vitor Guizilini, David Antonio Herrera, Minho Heo, Kyle Hsu, Jiaheng Hu, Muhammad Zubair Irshad, Donovon Jackson, Charlotte Le, Yunshuang Li, Kevin Lin, Roy Lin, Zehan Ma, Abhiram Maddukuri, Suvir Mirchandani, Daniel Morton, Tony Nguyen, Abigail O’Neill, Rosario Scalise, Derick Seale, Victor Son, Stephen Tian, Emi Tran, Andrew E. Wang, Yilin Wu, Annie Xie, Jingyun Yang, Patrick Yin, Yunchu Zhang, Osbert Bastani, Glen Berseth, Jeannette Bohg, Ken Goldberg, Abhinav Gupta, Abhishek Gupta, Dinesh Jayaraman, Joseph J Lim, Jitendra Malik, Roberto Martín-Martín, Subramanian Ramamoorthy, Dorsa Sadigh, Shuran Song, Jiajun Wu, Michael C. Yip, Yuke Zhu, Thomas Kollar, Sergey Levine, and Chelsea Finn. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.  
Benno Krojer, Mojtaba Komeili, Candace Ross, Quentin Garrido, Koustuv Sinha, Nicolas Ballas, and Mido Assran. A shortcut-aware video-qa benchmark for physical understanding via minimal video pairs. preprint, 2024.  
Hala Lamdouar, Charig Yang, Weidi Xie, and Andrew Zisserman. Betrayed by motion: Camouflaged object discovery via motion segmentation. In Proceedings of the Asian conference on computer vision, 2020.  
Yann LeCun. A path towards autonomous machine intelligence version 0.9. 2, 2022-06-27. Open Review, 62(1):1–62, 2022.  
Kai Li, Yulin Liu, Lei Li, et al. Unmasked teacher: Towards training-efficient video foundation models. arXiv preprint arXiv:2303.16035, 2023.  
Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Luo, Limin Wang, and Yu Qiao. Mvbench: A comprehensive multi-modal video understanding benchmark. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22195–22206, 2024.  
Tianhong Li and Kaiming He. Back to basics: Let denoising generative models denoise. arXiv preprint arXiv:2511.13720, 2025.  
Yingwei Li, Yi Li, and Nuno Vasconcelos. Resound: Towards action recognition without representation bias. In Proceedings of the European conference on computer vision (ECCV), pages 513–528, 2018.  
Guoliang Lin, Anton Milan, Chunhua Shen, and Ian Reid. Refinenet: Multi-path refinement networks for high-resolution semantic segmentation. In CVPR, 2017a.  
Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. Focal loss for dense object detection. In Proceedings of the IEEE international conference on computer vision, pages 2980–2988, 2017b.  
Yuanxin Liu, Shicheng Li, Yi Liu, Yuxiang Wang, Shuhuai Ren, Lei Li, Sishuo Chen, Xu Sun, and Lu Hou. TempCompass: Do video LLMs really understand videos? arXiv preprint arXiv:2403.00476, 2024.  
Ze Liu, Yutong Lin, Yue Cao, Han Hu, Yixuan Wei, Zheng Zhang, Stephen Lin, and Baining Guo. Swin transformer: Hierarchical vision transformer using shifted windows. In ICCV, 2021.  
Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF international conference on computer vision, pages 2630–2640, 2019.  
Ishan Misra and Laurens van der Maaten. Self-supervised learning of pretext-invariant representations. In CVPR, 2020.  
Ishan Misra, C. Lawrence Zitnick, and Martial Hebert. Shuffle and learn: Unsupervised learning using temporal order verification. In ECCV, 2016.  
Himangi Mittal, Nakul Agarwal, Shao-Yuan Lo, and Kwonjoon Lee. Can’t make an omelette without breaking some eggs: Plausible action anticipation using large video-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18580–18590, 2024.  
Shentong Mo and Shengbang Tong. Connecting joint-embedding predictive architecture with contrastive self-supervised learning. Advances in neural information processing systems, 37:2348–2377, 2024.  
Lorenzo Mur-Labadia, Ruben Martinez-Cantin, Jose J Guerrero, Giovanni Maria Farinella, and Antonino Furnari. Aff-ttention! affordances and attention models for short-term object interaction anticipation. In European Conference on Computer Vision, pages 167–184. Springer, 2024.  
M. Noroozi and P. Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV, 2016.  
Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.  
Valentinos Pariza, Mohammadreza Salehi, Gertjan J. Burghouts, Francesco Locatello, and Yuki M. Asano. Near, far: Patch-ordering enhances vision foundation models’ scene understanding. In ICLR, 2025.  
Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A. Efros. Context encoders: Feature learning by inpainting. In CVPR, 2016.  
Deepak Pathak, Ross Girshick, Piotr Dollár, Trevor Darrell, and Bharath Hariharan. Learning features by watching objects move. In CVPR, 2017.  
Federico Perazzi, Jordi Pont-Tuset, Brian McWilliams, Luc Van Gool, Markus Gross, and Alexander Sorkine-Hornung. A benchmark dataset and evaluation methodology for video object segmentation. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 724–732, 2016.  
Pedro O. Pinheiro, Amjad Almahairi, Ryan Y. Benmaleck, Florian Golemo, and Aaron Courville. Unsupervised learning of dense visual representations. In NeurIPS, 2020.  
Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017.  
Viorica Pătrăucean, Lucas Smaira, Ankush Gupta, Adrià Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alex Frechette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, Yusuf Aytar, Simon Osindero, Dima Damen, Andrew Zisserman, and João Carreira. Perception test: A diagnostic benchmark for multimodal video models. Advances in Neural Information Processing Systems, 36:42748–42761, 2023.  
Qwen Team, Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, Junyang Lin, An Yang, Binyuan Hui, Bowen Yu, Chen Cheng, Dayiheng Liu, Fan Hong, Fei Huang, Jiawei  
Liu, Jin Xu, Jianhong Tu, Jianyuan Zeng, Jie Zhang, Jinkai Wang, Jianwei Zhang, Jingren Zhou, Kexin Yang, Mei Li, Ming Yan, Na Ni, Rui Men, Songtao Jiang, Xiaodong Deng, Xiaoming Huang, Ximing Zhou, Xingzhang Ren, Yang Fan, Yichang Zhang, Yikai Zhu, Yuqiong Liu, and Zhifang Guo. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.  
Francesco Ragusa, Giovanni Maria Farinella, and Antonino Furnari. Stillfast: An end-to-end approach for short-term object interaction anticipation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2023.  
Jathushan Rajasegaran, Ilija Radosavovic, Rahul Ravishankar, Yossi Gandelsman, Christoph Feichtenhofer, and Jitendra Malik. An empirical study of autoregressive pre-training from videos. arXiv preprint arXiv:2501.05453, 2025.  
René Ranftl, Katrin Lasinger, and Vladlen Koltun. Toward robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer. In ECCV, 2020. Published in TPAMI in 2022.  
René Ranftl, Alexey Bochkovskiy, and Vladlen Koltun. Vision transformers for dense prediction. In ICCV, 2021.  
Mike Ranzinger, Greg Heinrich, Jan Kautz, and Pavlo Molchanov. Am-radio: Agglomerative vision foundation model reduce all domains into one. In CVPR, 2024.  
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.  
Debaditya Roy, Ramanathan Rajendiran, and Basura Fernando. Interaction region visual transformer for egocentric action anticipation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 6740–6750, 2024.  
Reuven Y. Rubinstein. Optimization of computer simulation models with rare events. European Journal of Operations Research, 99:89–112, 1997.  
Chaitanya Ryali, Yuan-Ting Hu, Daniel Bolya, Chen Wei, Haoqi Fan, Po-Yao Huang, Vaibhav Aggarwal, Arkabandhu Chowdhury, Omid Poursaeed, Judy Hoffman, et al. Hiera: A hierarchical vision transformer without the bells-andwhistles. arXiv preprint arXiv:2306.00989, 2023.  
Mohammadreza Salehi, Efstratios Gavves, Cees G. M. Snoek, and Yuki M. Asano. Time does tell: Self-supervised time-tuning of dense image representations. In ICCV, 2023.  
Ziyao Shangguan, Chuhan Li, Yuxuan Ding, Yanan Zheng, Yilun Zhao, Tesca Fitzgerald, and Arman Cohan. Tomato: Assessing visual temporal reasoning capabilities in multimodal foundation models. arXiv preprint arXiv:2410.23266, 2024.  
Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In European conference on computer vision, pages 746–760. Springer, 2012a.  
Nathan Silberman, Derek Hoiem, Pushmeet Kohli, and Rob Fergus. Indoor segmentation and support inference from rgbd images. In ECCV, 2012b.  
Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.  
Walter Simoncini, Andrei Bursuc, Spyridon Gidaris, and Yuki Asano. No train, all gain: Self-supervised gradients improve deep frozen representations. In NeurIPS, 2024.  
Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. arXiv preprint arXiv:2010.02502, 2020.  
Robin Strudel, Ricardo Garcia, Ivan Laptev, and Cordelia Schmid. Segmenter: Transformer for semantic segmentation. In ICCV, 2021.  
Richard S Sutton. An adaptive network that constructs and uses and internal model of its world. Cognition and Brain Theory, 4(3):217–246, 1981.  
Sanket Thakur, Cigdem Beyan, Pietro Morerio, Vittorio Murino, and Alessio Del Bue. Guided attention for next active object@ ego4d sta challenge. arXiv preprint arXiv:2305.16066, 2023.  
Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in neural information processing systems, 35:10078–10093, 2022.  
Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, and Manohar Paluri. Learning spatiotemporal features with 3d convolutional networks. In ICCV, 2015.  
Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. Look at spatiotemporal convolutions for action recognition. In CVPR, 2018.  
Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.  
Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 14549–14560, 2023.  
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024a.  
Xiaolong Wang and Abhinav Gupta. Unsupervised learning of visual representations using videos. In ICCV, 2015.  
Yi Wang, Kunchang Li, Xinhao Li, Jiashuo Yu, Yinan He, Guo Chen, Baoqi Pei, Rongkun Zheng, Zun Wang, Yansong Shi, et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pages 396–416. Springer, 2024b.  
Chen Wei, Haoqi Fan, Saining Xie, Chao-Yuan Wu, Alan Yuille, and Christoph Feichtenhofer. Masked feature prediction for self-supervised visual pre-training. arXiv preprint arXiv:2112.09133, 2021.  
Xin Wen, Bingchen Zhao, Anlin Zheng, Xiangyu Zhang, and Xiaojuan Qi. Self-supervised visual representation learning with semantic grouping. In NeurIPS, 2022.  
Jiahao Xie, Xiaohang Zhan, Ziwei Liu, Yew Soon Ong, and Chen Change Loy. Unsupervised object-level representation learning from scene images. In NeurIPS, 2021a.  
Saining Xie, Chen Sun, Jonathan Huang, Zhuowen Tu, and Kevin Murphy. Rethinking spatiotemporal feature learning: Speed-accuracy trade-offs in video classification. In ECCV, 2018.  
Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. arXiv preprint arXiv:2111.09886, 2021b.  
Dejing Xu, Jun Xiao, Zhou Zhao, Jian Shao, Di Xie, and Yueting Zhuang. Self-supervised spatiotemporal learning via video clip order prediction. In CVPR, 2019.  
Ning Xu, Linjie Yang, Yuchen Fan, Jianchao Yang, Dingcheng Yue, Yuchen Liang, Brian Price, Scott Cohen, and Thomas Huang. Youtube-vos: Sequence-to-sequence video object segmentation. In Proceedings of the European conference on computer vision (ECCV), pages 585–601, 2018.  
Lihe Yang, Binghui Kang, Zhiwen Huang, Xiaoxiao Xu, Jonathan Feng, and Hengshuang Zhao. Depth anything: Unleashing the power of large-scale unlabeled data. In CVPR, 2024.  
Sukmin Yun, Hankook Lee, Jaehyung Kim, and Jinwoo Shin. Patch-level representation learning for self-supervised vision transformers. In CVPR, 2022.  
Rowan Zellers, Jiasen Lu, Ximing Lu, Youngjae Yu, Yanpeng Zhao, Mohammadreza Salehi, Aditya Kusupati, Jack Hessel, Ali Farhadi, and Yejin Choi. Merlot reserve: Neural script knowledge through vision and language and sound. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16375–16387, 2022.  
Hang Zhang, Xin Li, and Lidong Bing. Video-llama: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858, 2023.  
Richard Zhang, Phillip Isola, and Alexei A. Efros. Colorful image colorization. In ECCV, 2016.  
Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024.  
Long Zhao, Nitesh B Gundavarapu, Liangzhe Yuan, Hao Zhou, Shen Yan, Jennifer J Sun, Luke Friedman, Rui Qian, Tobias Weyand, Yue Zhao, et al. Videoprism: A foundational visual encoder for video understanding. arXiv preprint arXiv:2402.13217, 2024.  
Bolei Zhou, Hang Zhao andXavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In CVPR, 2017a.  
Bolei Zhou, Hang Zhao, Xavier Puig, Sanja Fidler, Adela Barriuso, and Antonio Torralba. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 633–641, 2017b.  
Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832, 2021.  
Adrian Ziegler and Yuki M. Asano. Self-supervised learning of object parts for semantic segmentation. In CVPR, 2022.

Table 12 Pretraining hyperparameters for the Primary and Cooldown phases. Hyperparameters are identical across all our ViT architectures (ViT-L, -g and -G).

<table><tr><td>Parameter</td><td>Primary Phase</td><td>Cooldown Phase</td></tr><tr><td>Number of frames</td><td>16</td><td>64</td></tr><tr><td>Frames per Second</td><td>4.0</td><td>4.0</td></tr><tr><td>Video Crop Size</td><td>256</td><td>384</td></tr><tr><td>Image Crop Size</td><td>256</td><td>512</td></tr><tr><td>Random Resize Aspect Ratio</td><td>[0.75, 1.35]</td><td>[0.75, 1.35]</td></tr><tr><td>Random Resize Scale</td><td>[0.3, 1.0]</td><td>[0.3, 1.0]</td></tr><tr><td>Steps</td><td>135,000</td><td>12000</td></tr><tr><td>Warmup Steps</td><td>12,000</td><td>N/A</td></tr><tr><td>Image batch Size (global)</td><td>2,304</td><td>2,304</td></tr><tr><td>Video batch Size (global)</td><td>128</td><td>128</td></tr><tr><td>Starting Learning Rate</td><td>1e-4</td><td>6e-4</td></tr><tr><td>Final Learning Rate</td><td>5.25e-4</td><td>1e-6</td></tr><tr><td>Weight Decay</td><td>0.04</td><td>0.04</td></tr><tr><td>EMA</td><td>0.99925</td><td>0.99925</td></tr><tr><td>Spatial Mask Scale</td><td>[0.15, 0.7]</td><td>[0.15, 0.7]</td></tr><tr><td>Temporal Mask Scale</td><td>[1.0, 1.0]</td><td>[1.0, 1.0]</td></tr><tr><td>Mask Aspect Ratio</td><td>[0.75, 1.5]</td><td>[0.75, 1.5]</td></tr><tr><td>Tubelet Size</td><td>2</td><td>2</td></tr><tr><td>Patch Size</td><td>16</td><td>16</td></tr><tr><td>Predictor Blocks</td><td>24</td><td>24</td></tr><tr><td>Predictor Embedding Size</td><td>384</td><td>384</td></tr><tr><td>Encoder Intermediate Blocks (ViT-G)</td><td>[12, 24, 36, 48]</td><td>[12, 24, 36, 48]</td></tr><tr><td>Context Loss λ</td><td>0.5 for video, 0.7 for image</td><td>0.5 for videos, 0.7 for images</td></tr></table>

## Appendix

## A Pretraining details

In this section, we detail our pretraining protocol and provide hyper-parameters. We use identical hyperparameters for all our ViT architectures (ViT-L, -g and -G). Following Assran et al. (2025), pretraining consists of two phases: 1) the primary phase, lasting for 135k iterations, where the learning rate follows a warmup (from 1e-4 to 5.25e-4) for 12k iterations, then a constant schedule (5.25e-4); and 2) a cooldown phase, lasting for 12k iterations, where the learning rate is decayed to a small value (from 6e-4 to 1e-6).

During the primary phase, video samples consist of 16-frames clips sampled at 4 frames per second, both video and images have a resolution of 256 × 256. The cooldown phase is shorter, allowing for higher input resolution with manageable compute resources, video samples consist of 64-frames clips at resolution 384 × 384 and images are sampled at resolution $5 1 2 \times 5 1 2$ . During both phases, the masking strategy follows the one introduced in Bardes et al. (2024), the image and video batch sizes are set to, respectively, 2304 and 128, the weight-decay is set to 0.04 and the EMA coefficient is set to 0.99925.

We use the standard ViT architecture with a patch size of 16 for both images and videos, and a tubelet size of 2 for videos. Our predictor has 24 blocks (versus 12 in V-JEPA 2) and an embedding size of 384. For our deep self-supervision loss, we use 4 intermediate blocks of the encoder, at equally spaced indices. The parameter λ we use for Eq. 3 is different for video and images, with a value of 0.5 for video and 0.7 for images.

## B Distillation details

Our distillation protocol is adapted from the pretraining recipe, maintaining the same JEPA loss, masking, architecture, and hyperparameters, except for a few key modifications: the Exponential Moving Average (EMA) encoder is replaced by a frozen Teacher network. The predictor’s final layer embedding dimension is adapted to match the Teacher’s dimension to enable the computation of the loss. We do not use deep-self-supervision for distillation: the predictor operates only on the last layer of the encoder, and the loss is only calculated on this last layer. We do not initialize the predictor with the pretraining weights and instead retrain a new predictor, which has 12 blocks (instead of 24 for pretraining), greatly improving distillation stability. Throughout training, a separate EMA of the context encoder is maintained, corresponding to Polyak averaging, which is never used in the loss calculation but serves as the final, high-performing model released for downstream tasks.

Distillation follows a multi-stage training approach similar to pretraining. Stage 1 (Primary Phase) with a warmup-then-constant schedule and low-resolution data (16 frames, 256 × 256 pixels) using a pre-cooldown low-resolution V-JEPA 2.1 ViT-G as the teacher, and Stage 2 (Cooldown Phase) with a short annealing schedule and high-resolution data (64 frames, 384 ×384) using the corresponding post-cooldown high-resolution V-JEPA 2.1 ViT-G as the teacher. The Stage 2 student network is initialized with the EMA student network from Stage 1, a two-stage strategy found to produce the best-performing models compared to alternatives such as single-stage approaches or always distilling from the final high-resolution V-JEPA 2.1 teacher.

## C Evaluation protocols

## C.1 Depth Estimation

Datasets and Metrics. We evaluated the geometric quality of the dense features of V-JEPA 2.1 in depth estimation on NYUv2 (Silberman et al., 2012b) and KITTI (Geiger et al., 2013a), reporting the root mean square error (RMSE).

Evaluation Protocol. For fair comparison with the state-of-the-art, we adopted the same protocol reported by Siméoni et al. (2025), training a corresponding linear classifier on the training set of each data set. The linear layer is applied only on top of the V-JEPA 2.1 visual encoder frozen patch features, and it is further normalized using a learned BatchNorm layer. Note that we both train and evaluate single-image classifiers, as we process individual images using the 2D Convolution image tokenizer described in Sec.2.3.4. We sweep learning rates $\{ 1 . 5 \times 1 0 ^ { - 3 } , 1 \times 1 0 ^ { - 3 } , \bar { 8 } \times 1 0 ^ { - 4 } , 3 \times 1 0 ^ { - 4 } , 1 \times 1 0 ^ { - 4 } , 8 \times 1 0 ^ { - 5 } , 3 \times 1 0 ^ { - 5 } \}$ and weight decay $\{ 1 0 ^ { - 3 } , 1 0 ^ { - 4 } \}$ .

## C.2 Semantic Segmentation

Datasets and Metrics. We evaluate semantic segmentation with linear probing on the image semantic segmentation datasets ADE20K (Zhou et al., 2017a), Pascal VOC12 (Everingham et al., 2015), and Cityscapes (Cordts et al., 2016), reporting mean intersection-over-union (mIoU).

Evaluation Protocol. Similarly, we also follow the same protocol reported in DINOv3 Siméoni et al. (2025). We train a linear classifier on top of last-block patch features from the frozen encoder (already normalized via LayerNorm). Similar to the depth estimation evaluation protocol, we both train and evaluate single-image classifiers. Linear probes are optimized with AdamW using learning rates $\{ 1 . 5 { \times } 1 0 ^ { - 3 } , 8 { \times } 1 0 ^ { - 4 } , 5 { \times } 1 0 ^ { - 4 } , 8 { \times } 1 0 ^ { - 5 } \}$ and weight decay $\{ 0 . 1 , 0 . 0 1 , 1 0 ^ { - 4 } \}$ . We use a 512px resolution for ADE20K and VOC12, and 1024px height for Cityscapes.

## C.3 Video Object Segmentation Tracking

Dataset and Metrics. We evaluate on DAVIS 2017 (Pont-Tuset et al., 2017) and YouTube-VOS (Xu et al., 2018). DAVIS provides dense annotations for train/val (60/30 videos). In YouTube-VOS, as only the training set is publicly available, we divided this set both training (80%) and validation (20%) videos.

Evaluation Protocol. Following (Rajasegaran et al., 2025) and Siméoni et al. (2025), we implement a non-parametric label-propagation approach based on cosine similarity between patch features from a frozen backbone. Segmentation labels from the first frame are encoded as one-hot patch assignments. For each subsequent frame, we compute similarities with patches from the first frame and a small set of past frames, and propagate labels via a weighted k-NN scheme within a spatial neighborhood.

We sweep over several hyperparameters, including context length {4, 7, 10, 15}, neighborhood size {12, 24, 36}, neighborhood shape {square, circle}, top-k neighbors {3, 5, 10}, and similarity temperature {0.01, 0.1, 0.2, 0.7}. Hyperparameters are selected on the DAVIS training set and applied to all test splits. We use an input resolution of 480px.

## C.4 Short Term Object Interaction Anticipation

Dataset and Metrics. We evaluate V-JEPA 2.1 on the STA v2 split of Ego4D (Grauman et al., 2022), which comprises 243 hours of annotated video clips spanning 128 noun and 81 verb categories, with a total of 98,276 training and 47,395 validation samples. Following the established evaluation protocol (Grauman et al., 2022), we report Top-5 Average Precision (AP) and Top-5 mean Average Precision (mAP) metrics. As in standard mAP, predictions are matched to ground-truth boxes using IoU > 0.5 and additional class-specific criteria. For instance, the Top-5 mAP All variant requires a correct noun, correct verb, IoU above 0.5, and a time-to-contact δ prediction within a 0.25-second tolerance. To address the inherently multi-modal nature of future anticipation—where multiple plausible next-active objects may exist—the metric discounts up to four highest-scoring false positives per example, thereby avoiding penalties for plausible but unannotated predictions. Additionally, we report individual metrics to evaluate the temporal (δ), spatial (bounding boxes), and semantic (noun and verb) dimensions of the task.

Evaluation Protocol. For a fair comparison with the previous state-of-the-art, we follow the official StillFast baseline implementation (Ragusa et al., 2023), which extends Faster R-CNN (Girshick, 2015). Our attentive probe consists of four attention blocks followed by the frame-guided temporal pooling mechanism of (Mur-Labadia et al., 2024), since STA predictions must remain spatially aligned with the last video frame. This pooling mechanism adopts a residual cross-attention module between the 3D tokens of the full video clip and the 2D tokens of the last frame. Queries are computed from the last-frame tokens via a linear projection, while keys and values are obtained from the video tokens using corresponding projections. Once the video tokens are pooled in 2D, we apply a bilinear interpolation to match the resolution of the high-resolution image tokens. Both representations are summed and fused through a 2D convolution layer. Finally, we upsample this representation to produce a multi-scale pyramid at 1/4, 1/8, 1/16, 1/32, and 1/64 of the image input resolution. This pyramid forms the input to the prediction head, which follows the design used in prior work (Ragusa et al., 2023; Mur-Labadia et al., 2024; Thakur et al., 2023).

Probe Architecture. In this prediction head, first a Region Proposal Network (RPN) generates proposals from a feature pyramid, and RoIAlign extracts corresponding local features. To enrich these features with scene-level context, we concatenate to each local RoI feature the resulting pooled token from the 4-layers attentive probe. The fused representation is processed by two fully connected layers and added back to the local features through a residual connection, enabling global cues to modulate––rather than replace––local information. Noun, verb and time to contact δ are predicted using respective linear layers from the fused RoI features. A Softplus activation layers is applied to the δ output in order to obtain positive predictions. Training is end-to-end, combining standard Faster R-CNN losses with an additional verb loss $\mathcal { L } _ { v }$ and a Smooth-L1 TTC loss $\mathcal { L } _ { \mathrm { t t c } }$ , weighted by 0.1 and 0.5 respectively.

Optimization. Models are optimized with Adam using learning rates in $\{ 1 \times 1 0 ^ { - 5 } , 8 \times 1 0 ^ { - 5 } , 1 . 2 \times 1 0 ^ { - 4 } , 2 \times$ $1 0 ^ { - 4 } , 5 { \times } 1 0 ^ { - 4 } \}$ . We additionally explore different input configurations, including sampling rates {16, 8, 4, 2} fps, clip lengths {32, 16, 8, 4} frames, and spatial resolutions {256px, 384px}, covering videos with varying temporal spans. We follow the multi-scale training strategy of Faster R-CNN, sampling image short sides uniformly from {640, 672, 704, 736, 768, 800} with a maximum long side of 1333 px. Our final best configuration consist of 16 frames videos at 384px with a sampling rate of ${ \mathrm { 2 ~ f p s , } }$ and using a learning rate of $1 . 2 \times 1 0 ^ { - 4 }$ . We apply the same protocol to DINOv2 Oquab et al. (2023), DINOv3 Siméoni et al. (2025) and V-JEPA 2 Assran et al. (2025) models, reporting the best respective performance.

## C.5 Video and Image Classification

Dataset and Metrics. We use the evaluation benchmarks of V-JEPA 2 and evaluate our models on image classification on ImageNet Deng et al. (2009), and video classification on Kinetics-400 Kay et al. (2017), Something-Something-v2 Goyal et al. (2017) and Diving-48 Li et al. (2018).

Probe Architecture. We train an attentive probe on top of the frozen encoder output using the training data from each downstream task. Our attentive probe is composed of four transformer blocks, each using 16 heads in the attention layer. The first three blocks use standard self-attention; the final block uses a cross-attention layer with a learnable query token. The output of the cross-attention layer in the final block is added back to the query token as a residual connection before applying the rest of the block (LayerNorm, followed by MLP with a single GeLU activation). The transformer blocks are followed by a final linear classifier layer.

Evaluation Protocol. We follow the evaluation protocol of V-JEPA 2. For video evaluations, we sampled multiple clip segments from each input video. During validation, we also extracted three spatial views from each segment (instead of one view during training). The number of clip segments, frame step parameter, and global batch size vary for each evaluation. We use 64 $\times 2 \times 3$ input for SSv2 (16 frames clip, 2 temporal crops, 3 spatial crops), $1 6 \times 8 \times 3$ for K400, and $3 2 \times 4 \times 3$ for Diving-48. For all benchmarks, we use a spatial resolution of 384 × 384. We use the 2D or 3D tokenizer, respectively, for image and video benchmarks. We use a batch size of 256 except for ImageNet, we use 1024. For ImageNet, we do not use multiple clips or views per sample. For Diving-48, we employ a multilayer strategy. Instead of attending to the tokens from only the last layer of the encoder, we extract tokens from four encoder layers (the last layer and three intermediate layers) and attend to all of them.

Optimization. For each evaluation, we simultaneously train multiple classifier heads with different hyperparameters (learning rate and weight decay), reporting the accuracy of the best-performing classifier. For most of our evaluations (Kinetics, SSv2, and ImageNet), we train for 20 epochs and use 20 heads, each using one of five learning rate values and four weight decay values, and the learning rate decays according to a cosine schedule. For Diving-48, which benefit from longer training, we train for 50 epochs.

## C.6 Action Anticipation

Dataset and Metrics. We evaluated V-JEPA 2.1 on the Action Anticipation task of the Epic-Kitchen-100 benchmark Damen et al. (2022). The EK100 dataset is comprised of 100 hours of cooking activities recorded from an egocentric perspective across 45 kitchen environments. Each video in EK100 is annotated with action segments, which include a start timestamp, an end timestamp, and an action label. There are 3,568 unique action labels, each consisting of a verb and a noun category, with a total of 97 verb categories and 300 noun categories. The EK100 action anticipation task involves predicting noun, verb, and action (i.e., predicting verb and noun jointly) from a video clip, referred to as context, that occurs before the start timestamp of an action segment. The interval between the end of the context and the beginning of the action segment is the anticipation time, which is set to 1 second by default. Given that different future actions may be possible from a given context, mean-class recall-at-5 for noun, verb and action, is used as the metric to measure performance (Damen et al., 2022).

Evaluation Protocol. We follow the same evaluation protocol introduced in V-JEPA 2 and use a focal loss Lin et al. (2017b) with $\alpha = 0 . 2 5$ and $\gamma = 2 . 0$ when training the probe. We used a context of 32 frames with a frame-rate of 8 frames per second at resolution $3 8 4 \times 3 8 4$ . During probe training, we randomly sample an anticipation time between 0.25 and 1.75 seconds and an anticipation point between 0.0 and 0.25. Our probe architecture for action anticipation follows the architecture introduced in V-JEPA 2 and described in Appendix C.5, consisting of four transformer blocks, including a last cross-attention layer with a set of learnable query tokens, followed by a final linear classifier layer for each query token.

Table 13 Impact of deep self-supervision on the need for multi-scale evaluation. We conduct experiments with our V-JEPA 2.1 ViT-L trained from scratch. Last-Layer indicates that evaluation is conducted with only the last layer of the encoder as input to the probe. 4-Layers indicates that evaluation is conducted by concatenating 4 intermediate layers of the encoder and feeding it to the probe.

<table><tr><td rowspan="2">Deep Self-Supervision</td><td colspan="2">ADE20k</td><td colspan="2">NYU</td><td colspan="2">Diving-48</td></tr><tr><td>Last-Layer</td><td>4-Layers</td><td>Last-Layer</td><td>4-Layers</td><td>Last-Layer</td><td>4-Layers</td></tr><tr><td>✘</td><td>34.9</td><td>39.1</td><td>0.513</td><td>0.463</td><td>85.8</td><td>86.9</td></tr><tr><td>✓</td><td>42</td><td>43.9</td><td>0.381</td><td>0.370</td><td>87.2</td><td>88.1</td></tr></table>

Table 14 Impact of pretraining spatial resolution. We compare models trained during the cooldown phase at $2 5 6 \times 2 5 6$ spatial resolution against 384 × 384.

<table><tr><td>Model</td><td>Res</td><td>SSv2</td><td>D-48</td><td>K400</td><td>IN1K</td><td>EK100</td><td>ADE20k</td><td>Citys.</td><td>VOC</td><td>NYU</td><td>KITTI</td><td>DAVIS</td><td>YT-VOS</td></tr><tr><td>ViT-g</td><td>256</td><td>76.7</td><td>88.6</td><td>86.2</td><td>84.1</td><td>35.7</td><td>47.8</td><td>71.7</td><td>84.6</td><td>0.359</td><td>2.611</td><td>68.0</td><td>72.0</td></tr><tr><td>ViT-g</td><td>384</td><td>76.9</td><td>89</td><td>87</td><td>84.8</td><td>38.4</td><td>47.8</td><td>71.8</td><td>84.7</td><td>0.350</td><td>2.601</td><td>67.4</td><td>71.3</td></tr><tr><td>ViT-G</td><td>256</td><td>77.1</td><td>89.5</td><td>87.3</td><td>84.8</td><td>38.8</td><td>47.8</td><td>73.2</td><td>84.7</td><td>0.313</td><td>2.472</td><td>68.6</td><td>72.7</td></tr><tr><td>ViT-G</td><td>384</td><td>77.7</td><td>89.2</td><td>87.7</td><td>85.5</td><td>40.8</td><td>47.9</td><td>73.5</td><td>85.0</td><td>0.307</td><td>2.461</td><td>69.0</td><td>72.6</td></tr></table>

## D Additional Ablations

## D.1 Multi-scale evaluation

We evaluate the impact of Deep Self-Supervision on the need for multi-scale evaluation, i.e. using multiple layers of the encoder as input to the evaluation probe. Table 13 compares a model trained with Deep Self-Supervision with a model trained without, and we measure performance on image segmentation on ADE20k, depth estimation on NYU, and action recognition on Diving-48, using either; a) only the last layer of the encoder as input to the probe (Last-Layer) or b) 4 equally spaced intermediate layers (4-Layers). We observe that the model trained without Deep Self-Supervision has a wide gap in performance between using Last-Layer and 4-Layers, and benefits a lot from multi-scale evaluation. On the other hand, the model trained with Deep Self-Supervision is already stronger using Last-Layer, and only benefits a little from using 4-Layers.

## D.2 Pretraining Spatial Resolution

Table 14 presents the impact of the resolution used during the cooldown phase of pretraining on performance on most of our downstream tasks. We compare models trained with a video spatial resolution of 256 × 256 to the models we present in the main paper that are trained with a video spatial resolution of 384 × 384. V-JEPA 2.1 benefits from higher resolution pretraining across all of our downstream tasks.