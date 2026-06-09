# Self-Supervised Learning with a Multi-Task Latent Space Objective

Pierre-François De Plaen1 Abhishek Jha2,\* Luc Van Gool1,3,4,5 Tinne Tuytelaars1 Marc Proesmans1,5

1ESAT-PSI, KU Leuven, Belgium 2VIB.AI, KU Leuven, Belgium 3CVL, ETH Zürich, Switzerland 4INSAIT, Sofia University, Bulgaria 5TRACE vzw

## Abstract

We propose a multi-task formulation of self-predictive Siamese SSL in which each spatial transformation defines a distinct latent-space alignment task, solved by a dedicated predictor over a shared encoder. This perspective directly explains a long-standing failure of multi-crop training in self-predictive methods such as BYOL, SimSiam, and MoCo v3: a shared predictor is forced to solve heterogeneous alignment tasks simultaneously, leading to unstable optimization. Assigning one predictor per view type resolves this interference, unlocking linear evaluation gains of 3.8-4% across frameworks. This perspective also suggests a principled way to enrich pre-training by introducing additional spatial transformations as complementary tasks. We demonstrate this by introducing asymmetric cutout views, in which a masked online view is aligned with a complete target, forming a semantic inpainting objective. The resulting framework is stable, backboneagnostic, and consistently improves the performance of ResNet and ViT models on ImageNet and COCO.

## 1 Introduction

Self-supervised learning (SSL) has become a dominant paradigm for learning visual representations without supervision. It builds training signals directly from data, allowing models to discover visual structure on their own. Among SSL approaches, Siamese-based methods, often referred to as Joint Embedding Architectures [12, 30, 27, 64, 11], have proven especially effective. They learn by aligning representations of different augmented views of the same image. Recent large-scale efforts [46, 50] have further advanced this paradigm by leveraging larger datasets and more powerful architectures, yielding representations that generalize remarkably across a wide range of tasks.

An important factor behind this progress is the multi-crop strategy, which adds several small local crops to the pair of global views. This simple idea promotes spatial consistency and has been crucial to the success of clustering-based methods [10, 11]. Yet, it turns out to be unstable in self-predictive Siamese architectures, such as BYOL [27] and SimSiam [13], where the online and target branches play different roles, with the former including a prediction head. This instability has prevented these otherwise strong frameworks from benefiting from one of the most effective SSL augmentations.

We analyze this limitation and trace it to the shared predictor used across all views. A single predictor must align representations from both global and local crops, which differ strongly in scale and content, leading to unstable optimization. We resolve this by assigning a dedicated predictor to each view type, while keeping the encoder shared (Fig. 1b). This simple modification stabilizes multicrop training, yielding consistent accuracy gains across frameworks. On ImageNet, it improves linear evaluation by 3.8-4% across BYOL, SimSiam, and MoCo v3 (Tab. 1).

![](images/9653a0f93f52fb90504db3778e9f276ad0f7822d35370ea1c9ce864d77289806.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Targets"] --> B["Encoder"]
  C["Global"] --> D["Encoder"]
  E["Local"] --> D
  B --> F["Loss"]
  D --> F
  F --> G["p."]
```
</details>

(a) Naive multi-crop.

![](images/c7f601aaf0d08e0ff378a1608e2d06914ccb388ebd940075a6dc1be7e07bd720.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Targets"] --> B["Encoder"]
  C["Global"] --> D["Encoder"]
  E["Local"] --> D
  B --> F["Loss"]
  D --> F
  F --> G["p."]
  F --> H["p."]
```
</details>

(b) Multi-predictor multi-crop (ours, Sec. 3.2).  
![](images/5c8cf8a19dc6c89452632966ebffc4624bd8a5286ea5665c351be1cf77fcc2db.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Targets"] --> B["Encoder"]
  C["Global"] --> D["Encoder"]
  E["Local"] --> D
  F["Cutout"] --> D
  B --> G["Loss"]
  D --> G
  G --> H["p."]
  G --> I["p."]
  G --> J["p."]
```
</details>

(c) Multi-task (ours, Sec. 3.3).  
Figure 1: Overview of the proposed framework. Naive multi-crop (left) forces a single predictor to solve heterogeneous latent-space tasks simultaneously, leading to unstable optimization. Assigning one predictor per view type (middle) resolves this interference and stabilizes training. Introducing asymmetric cutout views (right) adds a complementary semantic inpainting task, further improving downstream performance. Here, p. denotes a predictor, and colors indicate distinct predictors.

This multi-predictor design suggests a broader perspective: each spatial transformation defines a distinct pre-training task, and new transformations can be introduced as complementary tasks without modifying the architecture or loss. For example, local crops define a local-to-global alignment task, encouraging the model to infer global context from partial observations. This raises a natural question: which additional transformations induce useful complementary tasks?

We study supervisory signals in SSL (Sec. 5.2) and find that spatial augmentations are the primary driver of representation learning. Motivated by this, we propose asymmetric cutout [19]: a region is masked in the online view while the target remains unaltered, creating a semantic inpainting objective in latent space. We first validate it as a standalone task and find that asymmetry is crucial, as masking both views causes performance to collapse, confirming that the model must predict the complete target representation from a masked view (Sec. 4.2).

Together, these elements form a unified and stable framework for self-predictive SSL (Fig. 1c). Each spatial transformation defines a distinct latent-space task with its own predictor. All tasks share a single encoder (backbone and projection head) and the same alignment loss, yielding a simple multitask formulation compatible with both CNNs and transformers.

In summary, our contributions are the following:

1. We identify the shared predictor as the primary cause of multi-crop instability in self-predictive SSL methods and show that decoupling predictors by view-type enables these methods to benefit from the multi-crop strategy.  
2. We study supervisory signals in SSL and confirm the central role of spatial transformations. We further provide evidence that multi-crop gains stem from increased spatial diversity rather than specifically from the inclusion of low-resolution crops.  
3. We interpret spatial transformations as latent-space tasks and extend our framework with a semantic inpainting task, which further improves performance.  
4. We validate the approach on key self-predictive methods with both ResNet and ViT backbones, achieving faster convergence and consistent performance gains across ImageNet (kNN, linear, semi-supervised, and fine-tuning) and COCO (detection and segmentation).

## 2 Related Work

SSL and Self-predictive Methods. Early work in self-supervised learning focused on handcrafted pretext tasks that required solving spatial or contextual prediction problems, including inpainting [48], jigsaw puzzles [45], relative patch prediction [21], rotation prediction [25], and colorization [67]. These methods demonstrated that purely spatial supervision can yield semantically meaningful representations. This perspective remains relevant to our approach, which leverages an asymmetric cutout as a lightweight spatial latent-space task.

Subsequent work shifted toward invariance-based objectives. Contrastive approaches such as Sim-CLR [12] and MoCo [30] align augmented views while repelling different images, while clusteringbased methods like DeepCluster [9] and SwAV [10] replace instance discrimination with prototype assignments. In parallel, non-contrastive methods explicitly enforcing feature diversity through decorrelation or variance constraints, as in Barlow Twins [64], VICReg [7], W-MSE [23], and ADM [17].

Self-predictive methods [27, 13, 15] showed that strong representations can be learned using a single invariance loss, without negatives, clustering, or reconstruction. Collapse is avoided through architectural asymmetry [49] via a predictor on the online branch. MoCo v3 [15] extended this paradigm to Vision Transformers [22], demonstrating the benefits of asymmetry even in contrastive formulations. Teacher-student approaches such as DINO [11] and iBOT [70] also rely on onlinetarget asymmetry with EMA updates, but omit a predictor head and instead stabilize training through centroid-based probability targets, or a centering and sharpening mechanism.

While self-predictive methods are stable in the two-crop regime, naively extending them to multicrop often leads to instability [11, 43, 42, 3]. Prior work proposed addressing this issue via auxiliary regularization [68]. In contrast, we show that the shared predictor itself is the source of instability: assigning one predictor per view type stabilizes multi-crop training.

Masked image modeling (MIM) constitutes another major direction, training models to reconstruct masked content either in pixel space [31, 60] or at the token level [6]. Hybrid approaches such as MSN [1] and iBOT [70] further unify masked prediction with prototype-based or distillation objectives, primarily in ViT-based settings [2, 16]. While these works broaden the SSL design space, our focus is on improving self-predictive Siamese methods in a backbone-agnostic manner.

Data Augmentations in SSL. Data augmentations are central in Siamese SSL because they define the proxy task. Cropping is particularly powerful, as it promotes spatial correspondence across views and is responsible for most of the accuracy gains [27, 44, 43]. The multi-crop strategy introduced in SwAV [10] and adopted in DINO [11] enriches supervision with local views, but poses challenges for self-predictive methods, which we address directly.

Several other approaches explicitly leveraged local features [59, 58, 57, 36]: they imposed regionor pixel-level consistency through additional losses or dense contrastive objectives. In contrast, our method follows SwAV and operates solely on final representations, introducing global-local consistency through augmentations rather than multi-level or dense supervision. In parallel, cutoutstyle perturbations, including random erasing, mixup, CutMix, and object-centric cropping, have been widely used to regularize supervised and self-supervised learning [69, 66, 63, 41]. In this work, we focus specifically on cutout as a form of within-image occlusion that isolates spatial masking effects without mixing images or labels, making it a simple building block.

Recent latent-space masked prediction frameworks such as I-JEPA [2] and CAPI [16] have shown that spatial prediction can yield strong semantic representations. Our asymmetric cutout shares this spirit but differs in a key respect: rather than predicting specific masked patch representations, we predict a single global embedding, which we show is sufficient to learn useful features while remaining backbone-agnostic.

Multi-Task SSL. Combining multiple pretext tasks has long been shown to improve representation learning. Early work [20] demonstrated the benefits of multi-task pretext learning with a shared backbone, and similar results were reported for 3D point clouds [28]. Multi-task SSL has also been explored in skeleton-based action recognition with shared encoders and task-specific heads [39]. With transformer-based masked modeling, multi-task learning became closely tied to multi-modality, with methods reconstructing multiple modalities within a unified architecture [24, 4, 61, 34]. However, these methods are largely reconstruction- or modality-driven and often rely on task-specific decoders. In contrast, we operate purely in latent space and show that multi-task behavior emerges from enforcing alignment across diverse spatial view types.

Closer to our setting, adaptive multi-head contrastive learning [56] and multi-target BYOL [51] introduce architectural branching for heterogeneous views but do not address predictor design or multi-crop instability. Our approach instead assigns a dedicated predictor per view type while sharing the encoder and loss, enabling stable multi-crop training and effective multi-view integration.

## 3 Method

## 3.1 Background: Self-Predictive Methods

Siamese self-supervised methods learn by aligning representations of different augmented views of the same image. Two identical encoders process the views, and a small projection head maps backbone outputs to the representation space used for the loss. Including this projector has been shown to improve the quality of representations [12].

Early methods [12, 30] use contrastive objectives that combine an alignment term with a repulsion term to prevent trivial solutions such as representational collapse. Later approaches, including BYOL [27] and SimSiam [13], remove the need for explicit negative samples. Instead, they introduce asymmetry by assigning different roles to the two branches: the online branch includes a learnable predictor trained via gradient descent, while the target branch is not directly optimized and omits this predictor. In BYOL, target parameters are updated as an exponential moving average of the online network; in SimSiam, the target is a stop-gradient copy of the online network.

## 3.2 Stabilizing Multi-Crop via Decoupled Predictors

The multi-crop strategy extends the standard two-view setup by introducing several smaller local crops alongside the usual global views. Each crop is encoded independently, and the model learns to align local representations with their corresponding global ones. This encourages consistency across scales and contextual levels.

Multi-crop significantly improves the performance of many SSL methods, including contrastive frameworks such as SimCLR [12] and MoCo [30], as well as clustering-based approaches like SwAV [10] and DINO [11]. However, this improvement is not universal. In self-predictive architectures [27, 13, 15], multi-crop leads to training instability and degraded performance compared to their standard baselines [11, 43, 42, 3]. Caron et al. [11] reported this issue for BYOL with both ResNet and Vision Transformer backbones (With a ViT-S, accuracy drops from 71.4% to 64.8% when using multi-crop). Morningstar et al. [43] observed analogous findings for MoCo v3, where performance drops by 1.5% with multi-crop, primarily because many runs exhibited training instability. While reducing the batch size or learning rate mitigated these failures, it again yielded models that underperformed the standard two-view baselines.

We hypothesize that this instability arises because global and local crops induce two fundamentally different latent-space tasks. Global-to-global alignment optimizes for augmentation invariance, whereas local-to-global alignment requires predicting global semantics from very limited contextual information (e.g., only an animal’s fur). In self-predictive methods, a single prediction head is therefore required to solve both tasks simultaneously, leading to degraded performance. We refer interested readers to App. D for an extended analysis.

To resolve this issue, we assign a separate predictor to each view type while keeping the encoder shared (see Fig. 3). Each predictor specializes in its corresponding view type, reducing interference between global and local alignment tasks. This modification does not alter the loss function or require new hyperparameters; it only marginally increases the number of learnable parameters and keeps training time unchanged, since each forward pass uses a single predictor. By isolating predictors across view types, we achieve stable optimization across different architectures. This stabilized formulation enables self-predictive methods to benefit from multi-crop augmentations (see Sec. 4.1) and serves as the foundation for the multi-task framework described next.

## 3.3 Spatial Transformations as Latent-Space Tasks

With predictors decoupled by view type, multi-crop can be reinterpreted as a set of alignment tasks sharing a common encoder, each spatial transformation defining a distinct pre-training objective solved by its dedicated predictor. This perspective naturally suggests extending the framework by incorporating additional transformations as complementary latent-space tasks.

We introduce asymmetric cutout [19] as one such task. While local crops require the model to infer global semantics from a reduced spatial context, random cutouts mask internal regions, encouraging robustness to partial occlusions. We apply cutout asymmetrically (see Fig. 2): the online branch receives the masked view while the target remains complete. In this setup, the online model is trained to predict the representation of the full image from a partially masked view, forming a semantic inpainting objective in latent space.

![](images/d7d93fda5529d63f19def5f1d1940759a88e28fa450757d7a63e82ae4b6dd342.jpg)

<details>
<summary>text_image</summary>

Online view
Target view
Asymmetric
(inpainting-like)
Symmetric
(invariance)
</details>

Figure 2: Asymmetric and symmetric cutout. Image from ImageNet val. set (№7011).

Formally, let $z ^ { v }$ denote the representation of view type $v \in$ {glob, loc, cutout}. The total alignment loss is computed as a weighted sum over view types:

$$
\mathcal {L} = \sum_ {v} \lambda_ {v} \mathbb {E} \left[ \left\| q _ {v} \left(\boldsymbol {z} _ {v}\right) - \boldsymbol {z} _ {\text { glob }} \right\| _ {2} ^ {2} \right] \tag {1}
$$

where $q _ { v }$ denotes the predictor for view type v and $\lambda _ { v }$ its relative weight. To keep the method simple and avoid tuning for dataset-specific statistics, we simply set all task weights to the same value: $\lambda _ { \mathrm { g l o b } } = \lambda _ { \mathrm { l o c } } = \lambda _ { \mathrm { c u t o u t } }$ .

This unified formulation (Fig. 3) treats each spatial transformation as a distinct latent-space task with its own predictor, while sharing the encoder and alignment loss across all view types. It integrates global, local, and cutout views into a single stable training recipe, is compatible across backbone architectures, and provides a principled way to incorporate additional spatial tasks in the future. We refer to this unified framework as MULAN (multi-task latent-space network).

![](images/530492206ec78ed9b0fa15c49cbc36d35a161c1c85618f0c030967acc675af53.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Image"] --> B["Targets"]
  B --> C["Backbone + Projection Head"]
  C --> D["stop-grad"]
  D --> E["Repres."]
  F["Task A"] --> G["Backbone + Projection Head"]
  G --> H["Repre."]
  H --> I["Prediction Head A"]
  I --> J["Prediction"]
  J --> K["loss"]
  L["Task B"] --> M["Backbone + Projection Head"]
  M --> N["Prediction Head B"]
  N --> O["Prediction"]
  O --> P["loss"]
  Q["Task C"] --> R["Backbone + Projection Head"]
  R --> S["Prediction Head C"]
  S --> T["Prediction"]
  T --> U["loss"]
    style A fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style Q fill:#f9f,stroke:#333
```
</details>

Figure 3: The proposed MULAN framework. A shared encoder processes all spatial views (global, local, cutout), while task-specific prediction heads align each view’s representation with the target. All tasks are optimized jointly under the same loss. Image from ImageNet val. set (№43632).

## 4 Experiments

## 4.1 Validation of Multi-Predictor Multi-Crop

We first validate our solution to the instability of multi-crop training in self-predictive frameworks. As discussed in Sec. 3.2, a shared predictor must align representations from both global and local crops, often causing instability. To address this, we assign a separate predictor to each view type while keeping the encoder shared, and train BYOL, SimSiam, and MoCo v3 under identical settings.

Setup. All experiments use a ResNet-50 [29] backbone and follow the original training recipes of each method, except that batch sizes are reduced to 1024 for BYOL and MoCo v3-1. Models are trained for 200 epochs on ImageNet-1k [18] with standard augmentations. Representation quality is measured using (1) linear evaluation, where a linear classifier is trained on frozen backbone features, and (2) non-parametric k-nearest neighbor (kNN) evaluation on the embedding space.

Results. Tab. 1 reports linear and kNN accuracies. Introducing view-specific predictors lets all three frameworks benefit from multi-crop training. In BYOL, linear accuracy rises from 70.7% to 74.7%. SimSiam and MoCo v3 show similar gains of around +4 points each. All runs converge reliably at standard learning rates, confirming that the shared predictor was the main source of instability. These results demonstrate that self-predictive SSL methods can fully benefit from multicrop augmentations once predictors are decoupled by view type. The modification is lightweight and generalizes across frameworks.

Table 1: Downstream performance of self-supervised ResNet-50 models trained with the multipredictor multi-crop (m-c) formulation. All models are pre-trained for 200 epochs.

<table><tr><td>Method</td><td>kNN</td><td> $\Delta$ </td><td>lin.</td><td> $\Delta$ </td></tr><tr><td>BYOL</td><td>63.1</td><td></td><td>70.7</td><td></td></tr><tr><td>w. multi-predictor m-c</td><td>69.5</td><td>+6.4</td><td>74.7</td><td>+4.0</td></tr><tr><td>SimSiam</td><td>60.2</td><td></td><td>69.9</td><td></td></tr><tr><td>w. multi-predictor m-c</td><td>65.3</td><td>+4.9</td><td>73.9</td><td>+4.0</td></tr><tr><td>MoCo v3</td><td>64.5</td><td></td><td>71.4</td><td></td></tr><tr><td>w. multi-predictor m-c</td><td>68.7</td><td>+4.2</td><td>75.2</td><td>+3.8</td></tr></table>

## 4.2 Asymmetric Cutout as a Standalone Task

Before adding cutout views to our multi-task formulation, we examine whether random cutout alone provides a valuable learning signal. We train BYOL using cutout as the only augmentation, masking a random region in the online view while keeping the target view complete. After 100 epochs on ImageNet, this simple setup reaches 46.8% accuracy, confirming that the model can learn meaningful representations by predicting the embedding of a full image from a partially masked view.

We then compare asymmetric and symmetric masking. When both views are masked, the model achieves near-random performance (3.4%). This confirms that asymmetry is crucial: the model must infer missing information from an unmasked reference rather than from another incomplete view. These results validate asymmetric cutout as a viable latent-space task and motivate its integration in our final multi-task formulation.

## 4.3 Evaluation of the Multi-Task Formulation

We now evaluate whether the proposed asymmetric cutout latent-space objective (Sec. 3.3) complements the two existing latent-space tasks (Sec. 3.2). Specifically, we assess whether models can leverage this additional pre-training signal to improve representation quality. We apply the multi-task formulation to BYOL, SimSiam, and MoCo v3, and compare each method against its multi-predictor multi-crop baseline under matched computational budgets.

Implementation details. We pre-train ResNet-50 backbones following the official training recipes of each method (see App. A for full settings). For a fair comparison, we adjust the number and type of views so that the total computational cost remains similar; specifically, we replace two local views with a single cutout view, since local views require roughly half the computation.

Results. Tab. 2 shows that adding cutout views further improves the performance of the three multi-crop frameworks by a significant margin. Even with only 200 pre-training epochs, all three multi-task models outperform their respective baselines. For instance, the multi-task SimSiam variant reaches 74.7% linear accuracy after only 200 pre-training epochs, surpassing the 71.3% achieved by the original SimSiam even after 800 epochs. These gains also translate into improved training efficiency. In the case of BYOL, the multi-task variant reaches 75.6% linear accuracy after 200 epochs (72 hours), surpassing the 74.3% achieved by the 1000-epoch baseline despite requiring only about one-third of the total training time (219h), even though its per-epoch cost is slightly higher (21:35 vs. 13:09) (see App. C, Tab. 9).

Table 2: Downstream performance of self-supervised ResNet-50 models under different augmentation strategies. Gray baselines correspond to the full-schedule results from the original papers.

<table><tr><td>Method</td><td>Strategy</td><td>Epochs</td><td>kNN</td><td>lin.</td></tr><tr><td>BYOL</td><td>baseline</td><td>1000</td><td>68.0</td><td>74.3</td></tr><tr><td rowspan="3">BYOL</td><td>baseline</td><td>200</td><td>63.1</td><td>70.7</td></tr><tr><td>multi-predictor m-c</td><td>200</td><td>69.5</td><td>74.7</td></tr><tr><td>multi-task</td><td>200</td><td>69.9</td><td>75.6</td></tr><tr><td>SimSiam</td><td>baseline</td><td>800</td><td>-</td><td>71.3</td></tr><tr><td rowspan="3">SimSiam</td><td>baseline</td><td>200</td><td>60.2</td><td>69.9</td></tr><tr><td>multi-predictor m-c</td><td>200</td><td>65.3</td><td>73.9</td></tr><tr><td>multi-task</td><td>200</td><td>66.5</td><td>74.7</td></tr><tr><td>MoCo v3</td><td>baseline</td><td>1000</td><td>68.9</td><td>74.6</td></tr><tr><td rowspan="3">MoCo v3</td><td>baseline</td><td>200</td><td>64.5</td><td>71.4</td></tr><tr><td>multi-predictor m-c</td><td>200</td><td>68.7</td><td>75.2</td></tr><tr><td>multi-task</td><td>200</td><td>69.2</td><td>75.7</td></tr></table>

## 4.4 Comparison with State-of-the-Art Methods

To assess the scalability to longer training schedules, competitiveness, and general applicability of our approach across backbones, we train MULAN on ImageNet-1k [18] using both convolutional and transformer backbones.

Implementation details. We adopt BYOL [27] as the base framework, given its strong results in our 200-epoch experiments. For the ResNet-50 [29] backbone, we follow the official BYOL configuration: two-layer projector and predictor networks (hidden dimension 4096, output 256), each with intermediate batch normalization [33] and ReLU activations. We pre-train the model for 800 epochs with the LARS optimizer [62], a linearly scaled [26] learning rate $( l r = 0 . 4 \times$ batch\_size/256), a cosine decay schedule, and a 10-epoch warm-up. The batch size is $1 0 2 4 ^ { - 1 }$ , and the weight decay is $1 . 5 \times 1 0 ^ { - 6 }$ , excluding batch normalization and bias parameters. The target network is updated using an EMA of the student weights [38], with the momentum coefficient increasing from 0.996 to 1.0 following a cosine schedule. For the ViT [22] backbones, we mostly follow the MoCo v3 [15] recipe and head design, except that we do not freeze the patch projection layer, as it led to lower performance in the multi-task setting. Instead, we found that lowering Adam’s $\beta _ { 2 }$ together with gradient clipping mitigated training instability (see App. E). We pre-train the backbone for 200 epochs using the AdamW optimizer [35], a linearly scaled learning rate (base $3 \times 1 0 ^ { - 4 } )$ , a batch size of 1024, and a weight decay of 0.1. The EMA base value is set to 0.998.

ImageNet results. Tab. 3 summarizes the results. Our MULAN framework significantly improves upon the BYOL baseline across both CNN and transformer backbones.

With a ResNet-50, it achieves 76.7% linear and 70.9% kNN accuracy, outperforming BYOL by +2.4% and +2.9%, respectively, and surpassing clustering-based methods such as SwAV and DINO. Performance is comparable to ReLIC v2, which attains slightly higher linear accuracy (77.1%) but a lower kNN score (70.5%). On ViT backbones, MULAN also yields consistent gains. With ViT-S, linear accuracy improves from 71.4% to 74.5% (+3.1%), and scaling to ViT-B further raises it to 78.3%, matching DINO. Notably, our method gains +3.8% from ViT-S to ViT-B compared to only +1.2% for DINO, suggesting MULAN interacts favorably with larger model capacity. Unlike DINO, which benefits from 800-epoch schedules, our method does not improve beyond 200 epochs on ViT backbones, yet already matches DINO’s accuracy. Thus, investigating how to unlock longerschedule training for self-predictive transformer methods is a promising direction for future work. Finally, our approach outperforms MIM-based methods on kNN and linear evaluation, consistent with evidence that MIM representations require full fine-tuning to reach peak performance [31, 40].

Table 3: Evaluation of SSL techniques pre-trained on ImageNet-1k. †MIM methods. ‡hybrid approaches. ⋆Checkpoints converted from JAX.

<table><tr><td>Ref.</td><td>Method</td><td>kNN</td><td>Lin.</td></tr><tr><td colspan="4">Backbone: ResNet-50</td></tr><tr><td>[55]</td><td>supervised</td><td>-</td><td>80.5</td></tr><tr><td>[12]</td><td>SimCLR</td><td>60.7</td><td>69.3</td></tr><tr><td>[14]</td><td>MoCo v2</td><td>61.9</td><td>71.1</td></tr><tr><td>[13]</td><td>SimSiam</td><td>-</td><td>71.3</td></tr><tr><td>[64]</td><td>Barlow Twins</td><td>66.0</td><td>73.2</td></tr><tr><td>[27]</td><td>BYOL</td><td>68.0*</td><td>74.3</td></tr><tr><td>[15]</td><td>MoCo v3</td><td>68.9</td><td>74.6</td></tr><tr><td>[10]</td><td>SwAV</td><td>65.7</td><td>75.3</td></tr><tr><td>[11]</td><td>DINO</td><td>67.5</td><td>75.3</td></tr><tr><td>[37]</td><td>C-BYOL</td><td>-</td><td>75.6</td></tr><tr><td>[52]</td><td>ReLIC v2</td><td>70.5*</td><td>77.1</td></tr><tr><td></td><td>MULAN</td><td>70.9</td><td>76.7</td></tr></table>

<table><tr><td>Ref.</td><td>Method</td><td>kNN</td><td>Lin.</td></tr><tr><td colspan="4">Backbone: ViT-S</td></tr><tr><td>[53]</td><td>supervised</td><td>-</td><td>79.8</td></tr><tr><td>[27]</td><td>BYOL</td><td>66.6</td><td>71.4</td></tr><tr><td>[14]</td><td>MoCo v2</td><td>64.4</td><td>72.7</td></tr><tr><td>[15]</td><td>MoCo v3</td><td>-</td><td>73.4</td></tr><tr><td>[10]</td><td>SwAV</td><td>66.3</td><td>73.5</td></tr><tr><td>[11]</td><td>DINO</td><td>74.5</td><td>77.0</td></tr><tr><td></td><td>MULAN</td><td>70.2</td><td>74.5</td></tr><tr><td colspan="4">Backbone: ViT-B</td></tr><tr><td>[53]</td><td>supervised</td><td>-</td><td>81.8</td></tr><tr><td>[60]</td><td>SimMIM $^{\dagger}$ </td><td>16.1</td><td>56.7</td></tr><tr><td>[31]</td><td>MAE $^{\dagger}$ </td><td>27.1</td><td>68.0</td></tr><tr><td>[2]</td><td>I-JEPA $^{\dagger}$ </td><td>-</td><td>72.9</td></tr><tr><td>[70]</td><td>iBOT $^{\ddagger}$ </td><td>77.1</td><td>79.5</td></tr><tr><td>[12]</td><td>SimCLR</td><td>-</td><td>73.9</td></tr><tr><td>[27]</td><td>BYOL</td><td>68.1</td><td>73.9</td></tr><tr><td>[15]</td><td>MoCo v3</td><td>71.4</td><td>76.7</td></tr><tr><td>[11]</td><td>DINO</td><td>76.1</td><td>78.2</td></tr><tr><td></td><td>MULAN</td><td>74.2</td><td>78.3</td></tr></table>

Overall, MULAN demonstrates consistent scalability across CNN and transformer backbones of varying capacities, yielding significant improvements in representation quality.

Transfer to dense tasks. We evaluate transfer to COCO object detection and segmentation (Tab. 4) using Mask R-CNN with FPN and a ResNet-50 backbone. Dense tasks require spatially detailed features, providing a complementary assessment of learned representations beyond classification. The MULAN framework outperforms supervised pretraining (41.8 vs. 39.0 AP for detection, 38.0 vs. 35.4 AP for segmentation) and prior SSL methods, achieving the highest AP on both tasks. These results suggest that the multi-task formulation produces representations that generalize well and could be used in a variety of downstream tasks.

We further report semi-supervised learning and fine-tuning results in App. B.

Table 4: Transfer performance (ResNet-50) on COCO using Mask R-CNN with FPN (1× schedule from detectron2).

<table><tr><td>Method</td><td>APdet.</td><td>APsegm.</td></tr><tr><td>supervised</td><td>39.0</td><td>35.4</td></tr><tr><td>MoCo v2</td><td>39.8</td><td>36.1</td></tr><tr><td>BYOL</td><td>40.4</td><td>37.0</td></tr><tr><td>SwAV</td><td>41.6</td><td>37.8</td></tr><tr><td>Barlow Twins</td><td>40.0</td><td>36.7</td></tr><tr><td>DINO</td><td>41.2</td><td>37.1</td></tr><tr><td>MULAN</td><td>41.8</td><td>38.0</td></tr></table>

Table 5: Effect of number and type of views on downstream performance. Each view type uses a different prediction head. Models are pre-trained for 200 epochs.

<table><tr><td></td><td>glob.</td><td>loc.</td><td>cutout</td><td>kNN</td><td>Lin.</td></tr><tr><td>BYOL</td><td>2</td><td>0</td><td>0</td><td>63.1</td><td>70.7</td></tr><tr><td>w. more views</td><td>4</td><td>0</td><td>0</td><td>64.3</td><td>71.7</td></tr><tr><td>w. local views</td><td>2</td><td>4</td><td>0</td><td>69.5</td><td>74.7</td></tr><tr><td>w. cutout views</td><td>2</td><td>0</td><td>2</td><td>68.6</td><td>73.7</td></tr><tr><td>w. multi-task</td><td>2</td><td>2</td><td>1</td><td>69.9</td><td>75.6</td></tr></table>

## 5 Ablation Studies

## 5.1 Influence of View Composition

Tab. 5 analyzes how the number and type of views affect BYOL performance. Increasing the number of global views from two to four yields only a modest improvement (70.7% to 71.7%), an effect that would likely diminish with longer training schedules. Introducing new view types instead produces substantial gains: local views raise performance to 74.7%, and global+cutout views reach 73.7%, confirming that view diversity is more valuable than quantity alone. Notably, the global+cutout configuration significantly outperforms the baseline, suggesting that the benefits of multi-crop strategies can be attributed to the increased spatial diversity, rather than to the inclusion of smaller-resolution crops specifically. The multi-task configuration achieves the best result with 75.6% linear accuracy despite using fewer total views than the multi-crop setup. This confirms that cutout and local views provide complementary training signals.

Table 6: Effect of individual augmentations on BYOL (ResNet-50). Removing cropping causes a large performance drop, while cropping or asymmetric cutout alone remains competitive. Spatial augmentations outperform all non-spatial combinations. Models are trained for 200 epochs, except the cutout-only variant, which saturates earlier.

<table><tr><td rowspan="2"></td><td colspan="3">Spatial augs.</td><td colspan="4">Other augs.</td><td rowspan="2">Lin acc.</td></tr><tr><td>crop</td><td>cutout</td><td>flip</td><td>jitter</td><td>gray</td><td>solar</td><td>blur</td></tr><tr><td>Baseline</td><td>√</td><td>✕</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>70.7</td></tr><tr><td>Remove crop</td><td>✕</td><td>✕</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>33.8</td></tr><tr><td>Crop only</td><td>√</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>55.3</td></tr><tr><td>Cutout only</td><td>✕</td><td>√</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>46.8</td></tr></table>

## 5.2 On the Importance of Spatial Augmentations

We introduced asymmetric cutout as a complementary task within our multi-task formulation, motivated by the disproportionate importance of spatial augmentations in SSL. While early work by Chen et al. [12] emphasized that strong augmentation compositions are crucial to define invariances, subsequent studies showed that cropping alone remains remarkably competitive. For instance, BYOL achieves 59.4% accuracy with only cropping, a trend also reported by Moutakanni et al. [44] for DINOv2 at scale.

Ablation results in Tab. 6 confirm that spatial transformations provide the primary training signal for SSL. In a 200-epoch schedule, cropping alone yields 55.3% accuracy on ImageNet, while removing cropping from the full augmentation pipeline reduces accuracy from 70.7% to 33.8%. Additionally, our asymmetric cutout strategy, which masks only the online view, achieves 46.8% accuracy and outperforms the combination of all non-spatial augmentations.

However, the modularity of our framework does not imply that all spatial augmentations can serve as effective latent-space tasks. As shown in App. F, simple transformations like random rotation yielded negligible additional improvements. We hypothesize that a latent-space task must be both semantically meaningful and sufficiently challenging to push the model to extract non-trivial features. We find that validating new tasks in a standalone setting (Sec. 4.2) is a reliable prerequisite to successful multi-task integration.

## 6 Conclusion

The key insight of this work is simple: spatial transformations in SSL are not just augmentations but latent-space tasks, and treating them as such unlocks substantial gains. Decoupling predictors by view type resolves the long-standing instability of multi-crop in self-predictive methods (BYOL, SimSiam, and MoCo v3), without changing the loss, backbone, or hyperparameters. Reframing this design as a multi-task objective then suggests incorporating additional pretext tasks. By adding asymmetric cutout as a complementary semantic inpainting task, we further improve accuracy across architectures. Beyond the specific gains reported, we believe the multi-task perspective offers a principled lens for future augmentation design in SSL. A natural next step is conditioning predictors on view-specific metadata such as cutout coordinates or crop scale, moving toward richer, taskconditional self-supervision. Extending this framework to video or 3D point clouds, where spatial transformations are even more diverse, is another promising direction.

## Acknowledgments

Most of the computational resources and services used in this work were provided by the VSC (Flemish Supercomputer Center), funded by the Research Foundation Flanders (FWO) and the Flemish Government – department WEWIS.

## References

[1] Mahmoud Assran, Mathilde Caron, Ishan Misra, Piotr Bojanowski, Florian Bordes, Pascal Vincent, Armand Joulin, Mike Rabbat, and Nicolas Ballas. Masked siamese networks for label-efficient learning. In ECCV, pages 456–473. Springer, 2022.  
[2] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a jointembedding predictive architecture. In CVPR, pages 15619–15629, 2023.  
[3] Arthur Aubret, Céline Teulière, and Jochen Triesch. Seeing the whole in the parts in selfsupervised representation learning. arXiv preprint arXiv:2501.02860, 2025.  
[4] Roman Bachmann, David Mizrahi, Andrei Atanov, and Amir Zamir. Multimae: Multi-modal multi-task masked autoencoders. In ECCV, pages 348–367. Springer, 2022.  
[5] Zhiwei Bai, Zhangchen Zhou, Jiajie Zhao, Xiaolong Li, Zhiyu Li, Feiyu Xiong, Hongkang Yang, Yaoyu Zhang, and Zhi-Qin John Xu. Adaptive preconditioners trigger loss spikes in adam. arXiv preprint arXiv:2506.04805, 2025.  
[6] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. In ICLR, 2021.  
[7] Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. In ICLR, 2021.  
[8] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.  
[9] Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In ECCV, pages 132–149, 2018.  
[10] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. NeurIPS, 33:9912–9924, 2020.  
[11] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, pages 9650–9660, 2021.  
[12] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, pages 1597–1607. PmLR, 2020.  
[13] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR, pages 15750–15758, 2021.  
[14] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020.  
[15] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In ICCV, pages 9640–9649, 2021.  
[16] Timothée Darcet, Federico Baldassarre, Maxime Oquab, Julien Mairal, and Piotr Bojanowski. Cluster and predict latent patches for improved masked image modeling. CoRR, 2025.  
[17] Pierre-François De Plaen, Tinne Tuytelaars, Marc Proesmans, and Luc Van Gool. Adversarial dependence minimization. arXiv preprint arXiv:2502.03227, 2025.  
[18] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, pages 248–255. IEEE, 2009.  
[19] Terrance DeVries and Graham W Taylor. Improved regularization of convolutional neural networks with cutout. arXiv preprint arXiv:1708.04552, 2017.  
[20] Carl Doersch and Andrew Zisserman. Multi-task self-supervised visual learning. In ICCV, pages 2051–2060, 2017.  
[21] Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, pages 1422–1430, 2015.  
[22] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR, 2021.  
[23] Aleksandr Ermolov, Aliaksandr Siarohin, Enver Sangineto, and Nicu Sebe. Whitening for self-supervised representation learning. In ICML, pages 3015–3024. PMLR, 2021.  
[24] Xinyang Geng, Hao Liu, Lisa Lee, Dale Schuurmans, Sergey Levine, and Pieter Abbeel. Multimodal masked autoencoders learn transferable representations. arXiv preprint arXiv:2205.14204, 2022.  
[25] Spyros Gidaris, Praveer Singh, and Nikos Komodakis. Unsupervised representation learning by predicting image rotations. In ICLR, 2018.  
[26] Priya Goyal, Piotr Dollár, Ross Girshick, Pieter Noordhuis, Lukasz Wesolowski, Aapo Kyrola, Andrew Tulloch, Yangqing Jia, and Kaiming He. Accurate, large minibatch sgd: Training imagenet in 1 hour. arXiv preprint arXiv:1706.02677, 2017.  
[27] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. NeurIPS, 33:21271–21284, 2020.  
[28] Kaveh Hassani and Mike Haley. Unsupervised multi-task feature learning on point clouds. In ICCV, pages 8160–8171, 2019.  
[29] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In CVPR, pages 770–778, 2016.  
[30] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In CVPR, pages 9729–9738, 2020.  
[31] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In CVPR, pages 16000–16009, 2022.  
[32] Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In European conference on computer vision, pages 646–661. Springer, 2016.  
[33] Sergey Ioffe and Christian Szegedy. Batch normalization: Accelerating deep network training by reducing internal covariate shift. In ICML, pages 448–456. PMLR, 2015.  
[34] Muhammad Abdullah Jamal and Omid Mohareri. Multi-modal contrastive masked autoencoders: A two-stage progressive pre-training approach for rgbd datasets. In CVPR, pages 17947–17957, 2025.  
[35] Diederik P Kingma. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.  
[36] Tim Lebailly and Tinne Tuytelaars. Global-local self-distillation for visual representation learning. In IEEE Winter Conf. Appl. Comput. Vis., pages 1441–1450, 2023.  
[37] Kuang-Huei Lee, Anurag Arnab, Sergio Guadarrama, John Canny, and Ian Fischer. Compressive visual representations. NeurIPS, 34:19538–19552, 2021.  
[38] Timothy P Lillicrap, Jonathan J Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, and Daan Wierstra. Continuous control with deep reinforcement learning. arXiv preprint arXiv:1509.02971, 2015.  
[39] Lilang Lin, Sijie Song, Wenhan Yang, and Jiaying Liu. Ms2l: Multi-task self-supervised learning for skeleton based action recognition. In Proceedings of the 28th ACM international conference on multimedia, pages 2490–2498, 2020.  
[40] Markus Marks, Manuel Knott, Neehar Kondapaneni, Elijah Cole, Thijs Defraeye, Fernando Perez-Cruz, and Pietro Perona. A closer look at benchmarking self-supervised pre-training with image classification. IJCV, pages 1–13, 2025.  
[41] Shlok Mishra, Anshul Shah, Ankan Bansal, Abhyuday Jagannatha, Janit Anjaria, Abhishek Sharma, David Jacobs, and Dilip Krishnan. Object-aware cropping for self-supervised learning. arXiv preprint arXiv:2112.00319, 2021.  
[42] Suhong Moon, Domas Buracas, Seunghyun Park, Jinkyu Kim, and John Canny. An embedding-dynamic approach to self-supervised learning. In IEEE Winter Conf. Appl. Comput. Vis., pages 2750–2758, 2023.  
[43] Warren Morningstar, Alex Bijamov, Chris Duvarney, Luke Friedman, Neha Kalibhat, Luyang Liu, Philip Mansfield, Renan Rojas-Gomez, Karan Singhal, Bradley Green, et al. Augmentations vs algorithms: What works in self-supervised learning. arXiv preprint arXiv:2403.05726, 2024.  
[44] Théo Moutakanni, Maxime Oquab, Marc Szafraniec, Maria Vakalopoulou, and Piotr Bojanowski. You don’t need domain-specific data augmentations when scaling self-supervised learning. NeurIPS, 37:116106–116125, 2024.  
[45] Mehdi Noroozi and Paolo Favaro. Unsupervised learning of visual representations by solving jigsaw puzzles. In ECCV, pages 69–84. Springer, 2016.  
[46] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.  
[47] Antonio Orvieto and Robert Gower. In search of adam’s secret sauce. Advances in Neural Information Processing Systems, 38:63404–63442, 2026.  
[48] Deepak Pathak, Philipp Krahenbuhl, Jeff Donahue, Trevor Darrell, and Alexei A Efros. Context encoders: Feature learning by inpainting. In CVPR, pages 2536–2544, 2016.  
[49] Pierre Harvey Richemond, Allison Tam, Yunhao Tang, Florian Strub, Bilal Piot, and Felix Hill. The edge of orthogonality: A simple view of what makes byol tick. In ICML, pages 29063–29081. PMLR, 2023.  
[50] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.  
[51] Azad Singh and Deepak Mishra. Branching out for better byol. In NeurIPS 2021 Workshop on Self-Supervised Learning: Theory and Practice, 2021.  
[52] Nenad Tomasev, Ioana Bica, Brian McWilliams, Lars Holger Buesing, Razvan Pascanu, Charles Blundell, and Jovana Mitrovic. Pushing the limits of self-supervised resnets: Can we outperform supervised learning without labels on imagenet? In First Workshop on Pretraining: Perspectives, Pitfalls, and Paths Forward at ICML 2022, 2022.  
[53] Hugo Touvron, Matthieu Cord, Matthijs Douze, Francisco Massa, Alexandre Sablayrolles, and Hervé Jégou. Training data-efficient image transformers & distillation through attention. In ICML, pages 10347–10357. PMLR, 2021.  
[54] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.  
[55] Vasilis Vryniotis. How to train state-of-the-art models using torchvision’s latest primitives. https://pytorch.org/blog/ how-to-train-state-of-the-art-models-using-torchvision-latest-primitives/, 2021.  
[56] Lei Wang, Piotr Koniusz, Tom Gedeon, and Liang Zheng. Adaptive multi-head contrastive learning. In European Conference on Computer Vision, pages 404–421. Springer, 2024.  
[57] Xinlong Wang, Rufeng Zhang, Chunhua Shen, Tao Kong, and Lei Li. Dense contrastive learning for self-supervised visual pre-training. In CVPR, pages 3024–3033, 2021.  
[58] Tete Xiao, Colorado J Reed, Xiaolong Wang, Kurt Keutzer, and Trevor Darrell. Region similarity representation learning. In ICCV, pages 10539–10548, 2021.  
[59] Enze Xie, Jian Ding, Wenhai Wang, Xiaohang Zhan, Hang Xu, Peize Sun, Zhenguo Li, and Ping Luo. Detco: Unsupervised contrastive learning for object detection. In ICCV, pages 8392–8401, 2021.  
[60] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In CVPR, pages 9653–9663, 2022.  
[61] Jiange Yang, Sheng Guo, Gangshan Wu, and Limin Wang. Comae: Single model hybrid pretraining on small-scale rgb-d datasets. In AAAI, pages 3145–3154, 2023.  
[62] Yang You, Igor Gitman, and Boris Ginsburg. Large batch training of convolutional networks. arXiv preprint arXiv:1708.03888, 2017.  
[63] Sangdoo Yun, Dongyoon Han, Seong Joon Oh, Sanghyuk Chun, Junsuk Choe, and Youngjoon Yoo. Cutmix: Regularization strategy to train strong classifiers with localizable features. In ICCV, pages 6023–6032, 2019.  
[64] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Selfsupervised learning via redundancy reduction. In ICML, pages 12310–12320. PMLR, 2021.  
[65] Xiaohua Zhai, Avital Oliver, Alexander Kolesnikov, and Lucas Beyer. S4l: Self-supervised semi-supervised learning. In ICCV, pages 1476–1485, 2019.  
[66] Hongyi Zhang, Moustapha Cisse, Yann N Dauphin, and David Lopez-Paz. mixup: Beyond empirical risk minimization. arXiv preprint arXiv:1710.09412, 2017.  
[67] Richard Zhang, Phillip Isola, and Alexei A Efros. Colorful image colorization. In ECCV, pages 649–666. Springer, 2016.  
[68] Tong Zhang, Congpei Qiu, Wei Ke, Sabine Süsstrunk, and Mathieu Salzmann. Leverage your local and global representations: A new self-supervised learning strategy. In CVPR, pages 16580–16589, 2022.  
[69] Zhun Zhong, Liang Zheng, Guoliang Kang, Shaozi Li, and Yi Yang. Random erasing data augmentation. In AAAI, pages 13001–13008, 2020.  
[70] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. Image bert pre-training with online tokenizer. In ICLR, 2021.

## A Experimental Settings

In this section, we detail the implementation settings required to reproduce the results reported in Secs. 4.1 to 4.4, 5.1 and 5.2. Unless specified otherwise, all experiments use ResNet-50 backbones, the standard ImageNet-1k training/validation splits, and synchronized batch normalization. All models are trained from scratch without labels, using mixed-precision training on a single 4×Nvidia A100 GPU node.

BYOL and MoCo v3 baselines. We closely follow the official training recipes for both methods, which use nearly identical hyperparameters. We pre-train ResNet-50 backbones with two-layer projection and prediction heads, each with a hidden dimension of 4096 and an output dimension of 256. All heads use intermediate batch normalization and ReLU activations. For MoCo v3, we additionally apply batch normalization on the outputs of both heads, following the official implementation. We pre-train the baselines for 200 epochs using the LARS optimizer, a linearly scaled learning rate (base value 0.4), a cosine decay schedule applied per training step, and a 10-epoch warm-up. For both methods, the target network is updated using an EMA of the online networks, with a momentum coefficient following a cosine schedule from 0.996 to 1.0. All experiments use a batch size of 1024.

SimSiam baseline. For SimSiam, we again follow the original training recipe. The projection layer is a three-layer MLP with hidden and output dimensions of 2048, and the prediction head is a two-layer bottleneck MLP with a hidden dimension of 512 and an output dimension of 2048. Both heads use intermediate batch normalization and ReLU activations, and the projection head additionally uses batch normalization on its output. We pre-train the baseline for 200 epochs with the SGD optimizer, a linearly scaled learning rate (base value 0.05), a batch size of 512, a cosine decay schedule applied per epoch, and no warmup. The prediction head uses a constant learning rate, as in the original paper.

Base data augmentations. We adopt the official data augmentation strategies from the respective methods. Unless stated otherwise, training views are obtained by applying the following sequence of transformations:

1. random resized cropping: a random patch of the image is selected, with an area uniformly sampled between 8% and 100% of that of the original image, and an aspect ratio logarithmically sampled between 3/4 and 4/3. For SimSiam and MoCo v3, the minimum area is set to 20%.  
2. random horizontal flipping with a probability of 50%.  
3. random color jitter: brightness, contrast, saturation, and hue are perturbed with random offsets uniformly sampled for each image. BYOL and MoCo v3 use the ranges (0.4, 0.4, 0.2, 0.1), while SimSiam uses (0.4, 0.4, 0.4, 0.1).  
4. random grayscale with a probability of 20%.  
5. random Gaussian blur: the image is blurred with a Gaussian blur kernel of size 23 and a standard deviation uniformly sampled in [0.1, 2]. In BYOL and MoCo v3, the transformation is applied with 100% probability for the first view and 10% for the second; in SimSiam, both views use a 50% probability.  
6. random solarization with probability 20% in the second view, for BYOL and MoCo v3.  
7. color normalization: finally, we normalize the color channels by subtracting the per-channel mean and dividing by the per-channel standard deviation estimated on the ImageNet training set.

Multi-predictor multi-crop and multi-task strategies. All multi-crop and multi-task experiments reuse the hyperparameters of their respective 2-view baselines to ensure a strictly controlled comparison. The multi-crop experiments use two global views and four local views, and the multitask runs use two global, two local, and one cutout view. Following previous works [10, 11], local crops have a resolution of 96×96, with the random crop area sampled in the interval [0.08, 0.25]. Global and cutout crops have a resolution of 224×224, with the random crop area sampled in the interval [0.25, 1.0]. The two global views follow BYOL’s base data augmentation setting described above. For local and cutout views, we apply random horizontal flipping with probability 50%, color jitter with BYOL’s parameter ranges, random grayscale with probability 20%, and Gaussian blur with probability 50%.

Table 7: Semi-supervised training on ImageNet-1k with 1% and 10% of labels. We report top-1 and top-5 validation set accuracies. ⋆We run the DINO semi-supervised evaluation under the same procedure.

<table><tr><td rowspan="2">Method</td><td colspan="2">Top-1</td><td colspan="2">Top-5</td></tr><tr><td>1%</td><td>10%</td><td>1%</td><td>10%</td></tr><tr><td>supervised</td><td>25.4</td><td>56.4</td><td>48.4</td><td>80.4</td></tr><tr><td>SimCLR</td><td>48.3</td><td>65.6</td><td>75.5</td><td>87.8</td></tr><tr><td>BYOL</td><td>53.2</td><td>68.8</td><td>78.4</td><td>89.0</td></tr><tr><td>DINO*</td><td>49.1</td><td>69.0</td><td>76.1</td><td>89.7</td></tr><tr><td>SwAV</td><td>53.9</td><td>70.2</td><td>78.5</td><td>89.9</td></tr><tr><td>Barlow Twins</td><td>55.0</td><td>69.7</td><td>79.2</td><td>89.3</td></tr><tr><td>NNCLR</td><td>56.4</td><td>69.8</td><td>80.7</td><td>89.3</td></tr><tr><td>C-BYOL</td><td>60.6</td><td>70.5</td><td>83.4</td><td>90.0</td></tr><tr><td>ReLIC v2</td><td>58.1</td><td>72.4</td><td>81.3</td><td>91.2</td></tr><tr><td>MULAN</td><td>60.4</td><td>72.5</td><td>84.0</td><td>91.3</td></tr></table>

Random cutout. Our random cutout implementation follows the hyperparameter sampling strategy of Torchvision’s RandomResizedCrop. We uniformly sample a cutout area between 20% and 40% of the original image area, and an aspect ratio logarithmically sampled between 3/4 and 4/3. The selected region is masked with a constant fill value equal to the ImageNet mean color, so that the masked pixels have an expected value of zero after color normalization.

Ablation Study on Data Augmentations. In the ablation on data augmentations, all experiments reuse the optimization hyperparameters of the 200-epoch BYOL baseline. We vary only the set of augmentations applied during pre-training.

kNN evaluation. For kNN evaluation, we follow the standard protocol used in prior work on selfsupervised learning. Each image is resized to a shorter side of 256 pixels and then center-cropped to 224 × 224 before feature extraction. For each validation sample, we retrieve its top-k nearest neighbors in the feature space using cosine similarity, and predict the label by a simple majority vote among them. We report the best results across k = 10 and k = 20. Similar to DINO, we observe that with vision transformers, the target network leads to slightly higher accuracy than the online network.

Linear evaluation. For linear probing, a linear classifier is trained on top of the frozen backbone features. To eliminate the need for per-model learning rate tuning, feature vectors are standardized using mean and variance statistics computed on ImageNet. For ResNet 50 architectures, features are extracted after the global average pooling layer, before the projection head. For Vision Transformers (ViT), the extraction protocol follows DINO [11]. For ViT-S, features consist of the concatenated [CLS] tokens from the final four layers. For ViT-B, features are formed by concatenating the [CLS] token with the global average-pooled output patch tokens. We train the linear head for 100 epochs with the SGD optimizer, a learning rate of 0.005, a batch size of 512, no weight decay, and a cosine learning rate schedule. During training, we apply random horizontal flipping with probability 50% and random cropping that keeps at least 8% of the image area, followed by resizing to 224 × 224 pixels. At evaluation time, images are resized so that the shorter side is 256 pixels, then centercropped to 224 × 224 pixels.

## B Additional Evaluations

Semi-supervised learning. We follow the semi-supervised learning protocol from [65, 12, 27], which we describe next for completeness. We fine-tune the pre-trained ResNet-50 backbone from Sec. 4.4 using subsets of ImageNet-1k labels (1% and 10%), as provided by [12]. During training, we apply the same data augmentations as for linear evaluation: random horizontal flipping and random cropping (with scales in the range 0.08-1), followed by resizing to 224×224 pixels. At test time, images are resized to 256 pixels along the shorter side using bicubic resampling, followed by a center crop to 224×224 pixels. In all cases, we use the same color normalization as during pre-training. We attach a linear classification head to the backbone and train the network with a softmax cross-entropy loss using SGD with Nesterov momentum 0.9, a batch size of 1024, and no additional regularization (e.g., no weight decay). We sweep over learning rate values in {0.005, 0.01, 0.02, 0.05, 0.1} and training schedules of 30 and 50 epochs, and report the test accuracy of the best configuration in Tab. 7.

Our approach significantly outperforms BYOL and achieves state-of-the-art performance. In the 1% setting, it outperforms ReLIC v2 by a large margin and matches the performance of C-BYOL [37], a compressive variant of BYOL. Notably, the C-BYOL objective could in principle be combined with our augmentation strategy, which we leave for future work.

Fine-tuning. We also fine-tune the pre-trained models on the full ImageNet-1k training set using the same simple protocol, with only two data augmentations and no explicit regularization. As reported in Tab. 8, MULAN improves over standard BYOL by +1.6 points in top-1 accuracy and by +1.1 points in top-5 accuracy.

Table 8: Fine-tuning on ImageNet-1k with neither heavy data augmentations nor regularization.

<table><tr><td>Method</td><td>Top-1</td><td>Top-5</td></tr><tr><td>SimCLR</td><td>76.0</td><td>93.1</td></tr><tr><td>BYOL</td><td>77.7</td><td>93.7</td></tr><tr><td>MULAN</td><td>79.3</td><td>94.8</td></tr></table>

## C Timing Analysis

Tab. 9 compares training cost and wall-clock efficiency across different BYOL training strategies. Although multi-task training increases the per-epoch cost, it is much more wall-clock efficient. The 200-epoch multi-task model (72h) outperforms the 1000-epoch baseline (219h) by 1.3% accuracy, corresponding to an approximately 3× gain in training efficiency.

Furthermore, the method is implemented with per-view forward and backward passes, keeping memory usage independent of the number of views. This reordering is mathematically equivalent to the standard multi-view formulation, yielding identical gradients. Under this setup, peak per-GPU memory consumption is 12.7 GB for all three augmentation strategies.

Table 9: Training efficiency on BYOL (ResNet-50) using a single node with 4× Nvidia A100 (80GB) GPUs.

<table><tr><td>Strategy</td><td>Epochs</td><td>Time/Ep</td><td>Total (h)</td><td>Lin.</td></tr><tr><td>baseline</td><td>200</td><td>13:09</td><td>44</td><td>70.7</td></tr><tr><td>multi-pred m-c</td><td>200</td><td>19:51</td><td>66</td><td>74.7</td></tr><tr><td>multi-task</td><td>200</td><td>21:35</td><td>72</td><td>75.6</td></tr><tr><td>baseline</td><td>1000</td><td>13:09</td><td>219</td><td>74.3</td></tr></table>

## D Extended Analysis of Multi-Crop Instability

To understand multi-crop instability in self-predictive methods, we combined insights from prior work with observations from our experiments. Caron et al. [11] reported that BYOL with multicrop suffers a significant accuracy drop for both ResNet-50 and ViT-S backbones, and Morningstar et al. [43] observed analogous drops for MoCo v3. Similar instability occurs in SimSiam [13], which does not use an EMA target. Together, these observations indicate that neither the EMA update nor the backbone architecture alone can explain the instability.

We also tested simple mitigation strategies reported in the literature, such as reducing batch size or learning rate. While these adjustments can prevent collapse, they do not restore performance to the level of the standard two-view baseline. This suggests that the optimization scale or rate is not the primary cause.

Taken together, these analyses support the conclusion that multi-crop instability is primarily associated with the predictor being asked to solve heterogeneous latent-space tasks, motivating the decoupled predictor solution described in the main text.

## E On Training Self-predictive Vision Transformers

This section details our training strategy for vision transformers, intending to facilitate the tuning of future methods.

We first adopt two changes relative to the ResNet setting: a deeper projection head following MoCo $\mathbf { v } 3 ,$ , and stochastic depth [32] applied to the online network with a dropout probability of 10%, following DINO.

Training vision transformers is known to require carefully tuned schedules and hyperparameters, particularly in self-supervised settings [53, 15, 11]. A notable challenge is training instability: MoCo v3 [15] reported sudden loss spikes that degrade performance and prevent the use of large learning rates, and showed that freezing the patch projection layer mitigates this issue across several methods (MoCo v3, BYOL, and SimCLR). In our multi-task setting, however, freezing the first layer leads to lower accuracy. Instead, we find that lowering the Adam optimizer’s $\beta _ { 2 }$ parameter (the exponential moving average of squared gradients) from 0.999 to 0.98, combined with gradient clipping, effectively mitigates these instabilities (Fig. 4) and unlocks substantial performance gains. This observation is consistent with findings from the natural language processing literature [47, 5, 8, 54].

![](images/776d601f690dcec19c7a6a96a18d1b349a530b4b7abceeb302dce9c6d22f66a4.jpg)

<details>
<summary>line chart</summary>

| Training Step | β₂ = 0.999 | β₂ = 0.98 |
| ------------- | ---------- | --------- |
| 0k            | ~10^1      | ~10^1     |
| 10k           | ~10^0      | ~10^0     |
| 20k           | ~10^0      | ~10^0     |
| 30k           | ~10^0      | ~10^0     |
| 40k           | ~10^0      | ~10^0     |
| 50k           | ~10^0      | ~10^0     |
| 60k           | ~10^0      | ~10^0     |
| 70k           | ~10^0      | ~10^0     |
| 80k           | ~10^0      | ~10^0     |
</details>

Figure 4: Training loss curves for MULAN with a ViT-B backbone on ImageNet (first 80k steps shown). We compare AdamW with $\beta _ { 2 } = 0 . 9 9 9$ and no gradient clipping against $\beta _ { 2 } = 0 . 9 8$ with gradient clipping at 0.5. The latter approach avoids loss spikes and achieves higher final accuracy.

A second challenge is that self-predictive methods tend to saturate early when trained with vision transformers. This is particularly pronounced for $\mathrm { { B Y O L } , }$ which does not benefit from training schedules longer than 300 epochs [11], and our multi-task variant saturates even earlier, around epoch 200. Notably, our method already matches DINO’s 800-epoch accuracy at 200 epochs, highlighting its potential. Enabling it to benefit from longer schedules, whether through regularization or refined hyperparameter tuning, remains an open challenge and a key direction for future work.

## F Negative Results and Future Directions

While augmenting the multi-crop strategy with asymmetric cutout views yielded significant gains (Tab. 5), several alternative spatial and mixing transformations proved ineffective. To guide the development of future work, we detail the online view-generation strategies that failed to improve performance:

• Random Rotation: Applying stochastic rotations to the input image before cropping.  
• Patch Shuffling: Dividing the input into a grid of $4 \times 4$ patches (similar to Vision Transformer tokenization) and performing a random shuffle.

• CutMix-like: Adopting a CutMix [63] approach where the online branch processes a blended image. The corresponding target was defined as a weighted average of the individual representations (passed separately to the target branch).

Table 10: Alternative pre-training strategies on BYOL (ResNet-50).

<table><tr><td>Pre-training Strategy</td><td>kNN</td><td>Lin.</td></tr><tr><td>glob. + loc.</td><td>69.5</td><td>74.7</td></tr><tr><td>glob. + loc. + rotate</td><td>68.9</td><td>75.1</td></tr><tr><td>glob. + loc. + shuffle</td><td>69.7</td><td>75.1</td></tr><tr><td>glob. + loc. + cutout</td><td>69.9</td><td>75.6</td></tr><tr><td>glob. + loc. + shuffle + cutout</td><td>70.0</td><td>75.7</td></tr></table>

The first two strategies yielded small gains (Tab. 10) that did not justify the increased computational overhead, while the third degraded performance. All three approaches failed to yield meaningful representations in the standalone validation setting of Sec. 4.2, confirming that this setting is a reliable prerequisite before integrating new tasks into the multi-task formulation.

An alternative approach for Random Rotation or Patch Shuffling would be to provide a conditioning signal to the predictors (e.g., the specific rotation angle). This modification would transform the method from a standard Siamese or Joint-Embedding Architecture into a Joint-Embedding Predictive Architecture, shifting the focus from learning invariant features to task-conditional prediction and enabling richer forms of self-supervision.