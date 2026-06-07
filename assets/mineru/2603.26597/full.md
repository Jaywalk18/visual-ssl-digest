# From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning

Yang Liu1 Qianqian Xu2,3,\* Peisong Wen1 Siran Dai4,5 Xilin Zhao6 Qingming Huang1,2,\*

1School of Computer Science and Technology, University of Chinese Academy of Sciences 2State Key Laboratory of AI Safety, Institute of Computing Technology, Chinese Academy of Sciences 3Beijing Academy of Artificial Intelligence

4Institute of Information Engineering, Chinese Academy of Sciences

5School of Cyber Security, University of Chinese Academy of Sciences

6School of Computer Science and Technology, Beijing Institute of Technology

liuyang232@mails.ucas.ac.cn xuqianqian@ict.ac.cn wenpeisong@ucas.ac.cn

daisiran@iie.ac.cn 1120230539@bit.edu.cn qmhuang@ucas.ac.cn

# Abstract

Recent studies have made notable progress in video representation learning by transferring image-pretrained models to video tasks, typically with complex temporal modules and video fine-tuning. However, fine-tuning heavy modules may compromise inter-video semantic separability, i.e., the essential ability to distinguish objects across videos. While reducing the tunable parameters hinders their intra-video temporal consistency, which is required for stable representations of the same object within a video. This dilemma indicates a potential trade-off between the intra-video temporal consistency and inter-video semantic separability during image-to-video transfer. To this end, we propose the Consistency-Separability Trade-off Transfer Learning (Co-Settle) framework, which applies a lightweight projection layer on top of the frozen image-pretrained encoder to adjust representation space with a temporal cycle consistency objective and a semantic separability constraint. We further provide a theoretical support showing that the optimized projection yields a better trade-off between the two properties under appropriate conditions. Experiments on eight image-pretrained models demonstrate consistent improvements across multiple levels of video tasks with only five epochs of self-supervised training. The code is available at https://github.com/yafeng19/Co-Settle.

# 1. Introduction

It has been a long-standing pursuit for the video community to exploit meaningful representations that benefit a wide range of video understanding scenarios. Driven by more comprehensive data, more powerful models, and more efficient algorithms, video representation learning has continuously evolved over the past decade, featuring heavy temporal processing mechanisms, such as 3D convolutions [32, 33, 101, 104], temporal attention [1, 9, 12], and inter-frame contrastive frameworks [39, 42, 50, 69, 117].

Recent studies [17, 46, 63, 66, 74, 115] have demonstrated that transferring image-pretrained models to the video domain can rival video-pretrained counterparts on multiple video downstream tasks. This observation raises a noteworthy question: How do image-pretrained models contribute to performance improvements on video downstream tasks?

In fact, image models are typically pretrained on largescale image datasets with diverse categories [25, 82]. Such pretraining encourages favorable inter-video semantic separability, i.e., semantic discrimination between different visual categories across videos [2, 47, 109]. Meanwhile, these models exhibit an approximate form of intra-video temporal consistency [73, 91], which produces relatively stable representations for the same object across frames within a video. However, since this consistency is obtained by pretraining with simple geometric changes (e.g., rotation), they fail to establish reliable temporal consistency, due to their lack of exposure to real-world temporal dynamics, such as a horse leaping over obstacles with complex transformations.

To leverage the advantages of image-pretrained, prior studies incorporate temporal modeling modules and finetune them on video datasets via indirect auxiliary tasks [17, 63, 66, 74, 115]. Yet, on the one hand, as the number of tunable parameters increases without proper constraint, the transfer process risks catastrophic forgetting of the semantic separability acquired from the image-pretraining stage [48, 83, 90]. On the other hand, if we restrict the number of tunable parameters to preserve separability, these methods fail to achieve sufficient temporal consistency. We argue that part of the parameters is occupied by indirect auxiliary tasks to learn information beyond consistency. This dilemma reveals a potential trade-off between the intravideo temporal consistency and inter-video semantic separability, thus calling for a careful balance between these two properties during image-to-video transfer.

In light of these challenges, we propose a Consistency-Separability Trade-off Transfer Learning (Co-Settle) framework, which applies a learnable lightweight projection on top of the frozen image encoder to adjust the representation space. To enhance temporal consistency, we design a cycle consistency objective for fine-grained correspondence learning across frames. To maintain semantic separability, we introduce a Kullback-Leibler divergence constraint to mitigate forgetting of semantic separability after the projection.

To provide an interpretable insight into the proposed framework, we further present theoretical justification for the trade-off mechanism between temporal consistency and semantic separability. Spectral analysis of the projection layer reveals that the optimized projection can increase the margin between inter- and intra-video distance in the representation space, leading to a more effective trade-off between the two properties. As shown in Figure 1, this improved trade-off results in better performance on video tasks.

Experimental results on eight ViT-based image-pretrained models demonstrate consistent improvements across several dense-level, frame-level, and video-level downstream video tasks, using only five epochs of self-supervised training on video datasets. These results suggest a potential solution for efficient image-to-video representation transfer learning.

The main contributions are summarized as follows:

• Methodologically, we propose the Co-Settle framework, which applies a lightweight projection layer on the imagepretrained encoder for representation space adjustment to balance temporal consistency and semantic separability.   
• Theoretically, through spectral analysis, we derive optimal conditions for the projection layer, which leads to a more effective trade-off between the two properties.   
• Experimentally, we evaluate our method on eight image foundation models, achieving consistent performance improvements across multiple granularity levels of video tasks after an efficient self-supervised learning process.

# 2. Related Work

Self-supervised visual representation learning. Selfsupervised learning has enabled models to develop generalizable representations for diverse downstream tasks. Contrastive learning methods leverage discriminative signals from different views [18–20, 24, 37, 96] to enforce semantical consistency for related contents. Masked modeling methods learn meaningful representations by predicting the raw pixels with encoder-decoder structures [3, 7, 8, 45, 78], or by reconstructing latent tokens within a self-distillation framework [14, 73, 91, 109, 120]. Extending these methods with the temporal dimension, early works introduce masked video modeling with high masking ratios [33, 38, 77, 101, 104, 111] while recent efforts [30, 39, 52, 69, 114] seek more efficient temporal modeling and prediction algorithms.

![](images/18fe02df5552e99ac1bc58a89e3b01e1829cb562296936b9af1f00f28536be20.jpg)

<details>
<summary>bubble</summary>

| Video Models | Image Models | Normalized Inter-video Distance (D_inter) | Normalized Intra-video Distance (D_intra) |
|---|---|---|---|
| SiamMAE | MAE | 0.50 | 0.13 |
| CropMAE | IJEPA | 0.56 | 0.24 |
| RSP | CLIP | 0.50 | 0.17 |
| MoCov3 | CLIP | 0.56 | 0.21 |
| iBOT | CLIP | 0.58 | 0.21 |
| DINO | CLIP | 0.62 | 0.24 |
| DINOV2 | CLIP | 0.63 | 0.20 |
| +Ours | CLIP Adapters | 0.59 | 0.19 |
| D=0.28 | CLIP Adapters | 0.47 | 0.21 |
| D=0.42 | CLIP Adapters | 0.51 | 0.19 |
| D=0.44 | CLIP Adapters | 0.53 | 0.20 |
| D=0.46 | CLIP Adapters | 0.56 | 0.24 |
| D=0.48 | CLIP Adapters | 0.58 | 0.25 |
| D=0.50 | CLIP Adapters | 0.59 | 0.19 |
| D=0.52 | CLIP Adapters | 0.61 | 0.21 |
| D=0.54 | CLIP Adapters | 0.62 | 0.23 |
| D=0.56 | CLIP Adapters | 0.63 | 0.21 |
| D=0.58 | CLIP Adapters | 0.64 | 0.24 |
| D=0.60 | CLIP Adapters | 0.65 | 0.21 |
mIoU on VIP (%) = 34.2, mIoU on VIP (%) = 38.4, mIoU on VIP (%) = 40.8, mIoU on VIP (%) = 39.8, mIoU on VIP (%) = 34.2, mIoU on VIP (%) = 38.8, mIoU on VIP (%) = 39.8, mIoU on VIP (%) = 40.8, mIoU on VIP (%) = 34.2, mIoU on VIP (%) = 38.8, mIoU on VIP (%) = 39.8, mIoU on VIP (%) = 34.2, mIoU on VIP (%) = 38.8, mIoU on VIP (%) = 39.8, mIoU on VIP (%) = 34.2, mIoU on VIP (%) = 38.8, mIoU on VIP (%) = 39.8, mIoU on VIP (%) = 34.2, mIoU on VIP (%). The chart highlights a 'Better Trade-off' indicated by an arrow from the lower-left to the upper-right region of the plot.
</details>

Figure 1. Comparison of video representation quality with recent visual representation learning models on the Kinetics-400 [55] validation set. Favorable video representations should exhibit strong intravideo temporal consistency (lower intra-video distance $D _ { i n t r a } )$ and clear inter-video semantic separability (higher inter-video distance $D _ { i n t e r } )$ jointly, yet the two objectives often compete since the two distances co-vary. Applying our method to image-pretrained models leads to consistent improvements on the margin of inter- and intra-video distance $D = D _ { i n t e r } - \gamma D _ { i n t r a }$ (detailed in Sec. 5.3), indicating a better trade-off between the two properties, and therefore leading to improved performance on video downstream tasks.

Image-to-video transfer learning. Recent works adopt parameter-efficient transfer strategies that update only a small subset of parameters while preserving comparable performance. Mainstream approaches insert adaptation modules into CLIP-pretrained Vision Transformers [28], enabling spatiotemporal adaptation via convolutional or attention-based operators [17, 66, 67, 74, 115]. However, due to the reliance on task-specific supervision [36, 55], these methods require separate fine-tuning when applied to different tasks.

Temporal cycle consistency. The inherent visual correspondence between adjacent frames provides natural supervisory signals to capture spatiotemporal coherence [4, 92]. Many studies exploit this property to learn semantically consistent representations within a cycle structure, benefiting downstream tasks such as classification [36, 55, 107, 108], video retrieval [57, 68], object segmentation [41, 61, 79, 84, 121], and point tracking [11, 15, 26]. Early methods focus on bidirectional patch-/object-level tracking [5, 65, 80, 105, 119], while others align feature distributions across related videos [29, 31, 40, 110]. Another line of work introduces random walk strategies [10, 51, 89], which guide representation learning by maximizing the probability that each patch returns to itself through a palindrome sequence.

![](images/868183d5eaea9058edf27b093136ac6f671e48fceb33996f50738ec5d590c936.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Video"] --> B["MAE"]
    B --> C["f"]
    C --> D["g"]
    D --> E["L_MSE"]
    F["CLIP"] --> G["ft"]
    G --> H["Cross-Attn."]
    H --> I["f"]
    I --> J["h"]
    J --> K["L_MIM"]
    L["iBOT"] --> M["f"]
    M --> N["h"]
    N --> O["f"]
    O --> P["h"]
    P --> Q["L_MIM"]
    R["Image Representation Space"] --> S["D_intra"]
    S --> T["D_inter"]
    T --> U["Transfer Learning"]
    V["Legend: Frozen Model, Trainable Module, Positional Encoding, Forward Pass"] --> W["v_t1^f, v_t2, v_t1^b"]
    W --> X["E_pos, E_pos"]
    X --> Y["Shared Model"]
    Y --> Z["z_t1^f, z_t2, z̃_t1^b"]
    Z --> AA["Shared Layer"]
    AA --> AB["p_t1^f, p_t2, p̃_t1^b"]
    AB --> AC["L_reg, L_cyc"]
    AC --> AD["L_reg, L_cyc"]
    AE["Enhancing Intra-video Consistency"] --> AF["Preserving Inter-video Separability"]
    AF --> AG["D = D_inter - γD_intra"]
    AG --> AH["Video Representation Space"]
```
</details>

Figure 2. Overview of our image-to-video transfer learning framework. Two frames are sampled from each video to construct a cyclic sequence. A frozen image-pretrained encoder extracts patch-level features, which are then mapped by a learnable projection layer. The projection layer is trained with a temporal cycle-consistency loss and a semantic separability constraint for representation adjustment, thereby promoting a better trade-off between intra-video temporal consistency and inter-video semantic separability.

# 3. Image-to-video Representation Transfer

To facilitate image-to-video transfer learning, we propose the Consistency-Separability Trade-off Transfer Learning (Co-Settle) framework as shown in Figure 2. We first encode frames with a frozen image-pretrained encoder and then leverage a learnable lightweight layer to project the representation space (Sec. 3.1). We optimize the projected representations for two goals: 1) intra-video temporal consistency, enforced by a cycle-consistency learning strategy (Sec. 3.2); and 2) inter-video semantic separability, preserved by a dimensionality constraint on the representations (Sec. 3.3).

# 3.1. Task definition

As a spatiotemporal volume [15, 88, 118], a video can be represented as an ordered sequence of T frames $V = \{ v _ { t } \in$ RH×W×C }T $\mathbb { R } ^ { \mathbf { \lambda } _ { H \times W \times C } ^ { \mathbf { \lambda } _ { T } } } \} _ { t = 1 } ^ { T }$ , where H, W , and C denote the height, width, and channels of each frame. Each frame ${ \mathbf { } } v _ { t }$ can be divided into $N = N _ { H } \times N _ { W } = \lceil H / p \rceil \times \lceil W / p \rceil$ nonoverlapping patches of $p ^ { 2 }$ pixels, where p is the patch size.

Given a video $V ,$ , we randomly sample two frames $\pmb { v } _ { t _ { 1 } }$ and $\mathbf { } v _ { t _ { 2 } }$ with a temporal offset determined by $\delta \in ( 0 , 1 )$ . Each frame is embedded through a frozen image-pretrained encoder $f ~ : ~ \mathbb { R } ^ { H \times W \times C } ~ \to ~ \mathbb { R } ^ { N \times d }$ with embedding dimension $d ,$ producing frame-wise representations $\begin{array} { r l } { z _ { t _ { 1 } } } & { { } = } \end{array}$ $f ( \pmb { v } _ { t _ { 1 } } ; \mathbf { E } _ { \mathrm { p o s } } )$ and $z _ { t _ { 2 } } = f ( \pmb { v } _ { t _ { 2 } } ; \mathbf { E } _ { \mathrm { p o s } } )$ , where $\mathbf { E } _ { \mathrm { p o s } }$ is the positional encoding of $f .$ . These representations inherit rich semantic priors from the image-pretrained model, yet still lack clear temporal correspondence. Then, we apply a lightweight layer $g : \dot { \mathbb { R } ^ { N \times d } }  \dot { \mathbb { R } ^ { N \times d } }$ with parameters limited to a linear layer and a LayerNorm to project $z _ { t _ { 1 } } , z _ { t _ { 2 } } \mathrm { a s } p _ { t _ { 1 } } = g ( z _ { t _ { 1 } } )$ and $p _ { t _ { 2 } } = g ( z _ { t _ { 2 } } )$ . This projection maps the static representations into a shared latent space, where we aim to enhance

temporal consistency while preserving semantic separability.

# 3.2. Intra-video temporal consistency learning

To provide direct and explicit guidance for learning temporal consistency, our core principle is to establish reliable temporal correspondences between patches. Without supervision for precise alignment, prior works [10, 51, 89] introduce Contrastive Random Walk (CRW) structure to enhance cross-frame correspondences in videos. CRW constructs a forward-backward sequence vt1 $\begin{array} { r } { \pmb { v } _ { t _ { 1 } } \xrightarrow { \mathrm { \ f o r w a r d } } \pmb { v } _ { t _ { 2 } } \xrightarrow { \mathrm { \ b a c k w a r d } } \pmb { v } _ { t _ { 1 } } } \end{array}$ vt2 vt1 , where the first and last frames are identical. For clarity, we denote the first forward frame as $\boldsymbol { v } _ { t _ { 1 } } ^ { f }$ and the last backward frame as $\pmb { v } _ { t _ { 1 } } ^ { b }$ . The objective of CRW is to maximize the probability that each patch in $\boldsymbol { v } _ { t _ { 1 } } ^ { f }$ returns to its original position in $\pmb { v } _ { t _ { 1 } } ^ { b }$ after traversing through the intermediate frame ${ \mathbf { } } v _ { t _ { 2 } }$ .

Let $p _ { t _ { 1 } } ^ { f } , \ p _ { t _ { 2 } } ,$ and $\boldsymbol { p } _ { t _ { 1 } } ^ { b }$ denote the projected representations of $\boldsymbol { v } _ { t _ { 1 } } ^ { f } , ~ \boldsymbol { v } _ { t _ { 2 } }$ , and $\boldsymbol { v } _ { t _ { 1 } } ^ { b }$ , respectively. We then calculate the forward and backward correlation matrices as $\begin{array} { r } { { \cal A } _ { t _ { 1 } } ^ { t _ { 2 } } = \operatorname { s o f t m a x } _ { \tau } ( p _ { t _ { 1 } } ^ { f } p _ { t _ { 2 } } ^ { \top } ) } \end{array}$ and $\begin{array} { r } { \pmb { A } _ { t _ { 2 } } ^ { t _ { 1 } } = \operatorname { s o f t m a x } _ { \tau } ( \pmb { p } _ { t _ { 2 } } \pmb { p } _ { t _ { 1 } } ^ { b ^ { \top } } ) } \end{array}$ . Concretely, each element in $\boldsymbol { A } _ { t _ { 1 } } ^ { t _ { 2 } }$ can be computed as:

$$
\boldsymbol {A} _ {t _ {1}} ^ {t _ {2}} (i, j) = \frac {\exp (d (\boldsymbol {p} _ {t _ {1}} (i) , \boldsymbol {p} _ {t _ {2}} (j)) / \tau)}{\sum_ {l = 1} ^ {N} \exp (d (\boldsymbol {p} _ {t _ {1}} (i) , \boldsymbol {p} _ {t _ {2}} (l)) / \tau)}, \tag {1}
$$

where $\tau \in \mathbb { R } ^ { + }$ represents the temperature hyperparameter and $d ( \cdot , \cdot )$ denotes the dot-product similarity. Intuitively, the matrix $A _ { t _ { 1 } } ^ { t _ { 2 } }$ quantifies the attention distribution from each patch in $\boldsymbol { v } _ { t _ { 1 } } ^ { f }$ to all patches in ${ \mathbf { } } v _ { t _ { 2 } }$ , indicating the transition paths of each patch between the two frames. Accordingly, the CRW objective can be formulated as aligning the correlation matrix product chain $A _ { t _ { 1 } } ^ { t _ { 2 } } A _ { t _ { 2 } } ^ { t _ { 1 } }$ with the identity matrix I via the Cross-Entropy loss $\begin{array} { r } { \dot { \mathcal { L } _ { C R W } } = \mathcal { L } _ { C E } \left( A _ { t _ { 1 } } ^ { t _ { 2 } } A _ { t _ { 2 } } ^ { t _ { 1 } } , I \right) } \end{array}$ .

![](images/87803730ddafbdfea984bf6041822ab635ebdb10c7d098a49aa1a6e0d74631ef.jpg)

<details>
<summary>line</summary>

| Category     | Arithmetic Function Value |
| ------------ | ------------------------- |
| Irrelevant   | 0                         |
| Shuffled     | 0                         |
</details>

![](images/e2e34894ebb209b14c019b9790fe736f508db873ec5be27b0db342e24a83754e.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Positional Encoding Augmentation Strategy"] -->|Interpolation| B["Crop"]
    B --> C["E_pos"]
    C --> D["Apply"]
    D --> E["v_t1^f"]
    E --> F["Forward Cycle"]
    F --> G["v_t2^f Backward Cycle"]
    G --> H["V"]
    subgraph Crop
        I1["1 2 3 4 5 6"] --> I2["7 8 9 10 11 12"]
        I3["13 14 15 16 17 18"]
        I4["19 20 21 22 23 24"]
        I5["25 26 27 28 29 30"]
        I6["31 32 33 34 35 36"]
    end
    subgraph E_pos
        E1["3 4 5 6"] --> E2["9 10 11 12"]
        E3["15 16 17 18"]
        E4["21 22 23 24"]
    end
    subgraph E
        E1 --> E2
        E2 --> E3
        E3 --> E4
        E4 --> E5
    end
```
</details>

![](images/d6ee75c672ffb79d58bff143a68c947d68248addfee78e1d870d26965e87d7cf.jpg)

<details>
<summary>line</summary>

| Step | Acc. (w/o PEA) | Acc. (w/ PEA) | J&Pm (w/o PEA) | J&Pm (w/ PEA) |
|------|----------------|---------------|----------------|---------------|
| 0    | 20             | 20            | 30             | 30            |
| 1000 | 70             | 70            | 45             | 45            |
| 2000 | 95             | 95            | 50             | 50            |
| 3000 | 98             | 98            | 50             | 50            |
| 4000 | 98             | 98            | 50             | 50            |
| 5000 | 98             | 98            | 50             | 50            |
</details>

Figure 3. Left: Observations on shortcuts. Patches with the same color box denote correspondence. Middle: Overview of our PEA strategy. Right: Cycle-consistent accuracy and downstream performance dynamics during training with or without our PEA strategy on MAE encoder.

Despite its elegant formation, CRW underperforms on modern Vision Transformer (ViT) backbones [28]. As shown in Figure 3 (Left), we test two cases: 1) Irrelevant setting: $v _ { t _ { 1 } } ^ { f } \mathbf { \Sigma } ( \bar { v } _ { t _ { 1 } } ^ { b } )$ and ${ \mathbf { } } v _ { t _ { 2 } }$ are sampled from unrelated videos; 2) Shuffled setting: the patches of $\mathbf { } v _ { t _ { 2 } }$ are randomly shuffled. In both cases, $A _ { t _ { 1 } } ^ { t _ { 2 } } A _ { t _ { 2 } } ^ { t _ { 1 } }$ rapidly converges to I, minimizing $\mathcal { L } _ { C R W }$ while failing to capture any meaningful visual correspondences. This phenomenon reveals a shortcut solution [89, 95], where the model reduces the loss by exploiting exact positional cues instead of learning semantic relations. We attribute this to the explicit positional encodings in ViTs, which are directly injected into the input tokens and propagated via the global attention, enabling patches to match by absolute location even under strong appearance changes.

To cope with this shortcut, we introduce Positional Encoding Augmentation (PEA), as illustrated in Figure 3 (Middle). PEA perturbs direct positional matching through an asymmetric design, inspired by information bottlenecks [98, 100] in self-distillation frameworks [14, 19, 37]. Specifically, PEA interpolates the pretrained positional encoding $\mathbf { E } _ { \mathrm { p o s } }$ with a controllable amplitude $\alpha \in \mathbb { R } ^ { + }$ and then applies a random crop to recover the original size, yielding an augmented version $\widetilde { \mathbf { E } } _ { \mathrm { p o s } }$ . Afterward, we use $\dot { \bf E } _ { \mathrm { p o s } }$ to encode the backward frame $\widetilde { \pmb { z } } _ { t _ { 1 } } ^ { b } = f ( \pmb { v } _ { t _ { 1 } } ^ { b } ; \widetilde { \pmb { \mathrm { E } } } _ { \mathrm { p o s } } )$ and obtain the ecorresponding projected representation $\widetilde { p } _ { t _ { 1 } }$ . This strategy ebreaks reliance on exact positional matches while preserving local positional relations, thereby leading to a stable learning process, as shown in Figure 3 (Right).

Given the forward correlation $\begin{array} { r } { { \bf \dot { A } } _ { t _ { 1 } } ^ { t _ { 2 } } = \operatorname { s o f t m a x } _ { \tau } ( { \bf q } _ { t _ { 1 } } ^ { f } { \bf q } _ { t _ { 2 } } ^ { \top } ) } \end{array}$ and the asymmetric backward correlation $\begin{array} { r l } { \widetilde { A } _ { t _ { 2 } } ^ { t _ { 1 } } } & { { } = } \end{array}$ softmaxτ $\mathbf { \nabla } _ { - } ( q _ { t _ { 2 } } \widetilde { { \mathbf { q } } } _ { t _ { 1 } } ^ { b ^ { \intercal } } )$ , we define the cycle-consistency loss as $\mathcal { L } _ { c y c } = \mathcal { L } _ { C E } \left( A _ { t _ { 1 } } ^ { t _ { 2 } } \widetilde { A } _ { t _ { 2 } } ^ { t _ { 1 } } , I \right)$ or formally:

$$
\mathcal {L} _ {c y c} = - \sum_ {i = 1} ^ {N} \log P \left(X _ {d} = \widetilde {\boldsymbol {q}} _ {t _ {1}} ^ {b} (i) | X _ {s} = \boldsymbol {q} _ {t _ {1}} ^ {f} (i)\right), \tag {2}
$$

where $X _ { s }$ and $X _ { d }$ denote the start and destination patches of the random walker. In this way, the cycle consistency objective can promote the model to learn effective temporal correspondences, facilitating the transfer from static image representations to dynamic video contexts.

# 3.3. Inter-video semantic separability constraint

While image-pretrained models exhibit favorable semantic separability, optimizing g solely for temporal alignment on video data with limited category diversity can lead to catastrophic forgetting of this property, and may further induce dimensional collapse [23, 34, 106, 122].

To mitigate this issue, we introduce a distributional regularization based on Kullback-Leibler (KL) divergence to preserve feature diversity across the projection layer. For each pair $( \pmb { p } , z ) \in \mathcal { S } = \{ ( \pmb { p } _ { t _ { 1 } } ^ { f } , z _ { t _ { 1 } } ^ { f } ) , ( \pmb { p } _ { t _ { 2 } } , z _ { t _ { 2 } } ) , ( \widetilde { \pmb { p } } _ { t _ { 1 } } ^ { \dot { b } } , \widetilde { z } _ { t _ { 1 } } ^ { b } ) \}$ , we compute normalized distributions via softmax along the feature dimension: $P = \operatorname { s o f t m a x } ( { p } ) , Z = \operatorname { s o f t m a x } ( z )$ . The regularization loss is then formulated as:

$$
\mathcal {L} _ {r e g} = \frac {1}{| \mathcal {S} |} \sum_ {(\boldsymbol {p}, \boldsymbol {z}) \in \mathcal {S}} \sum_ {i = 1} ^ {d} P (i) \log \frac {P (i)}{Z (i)}, \tag {3}
$$

where d denotes the feature dimension. This distributional regularization enforces a constraint on the global semantic structure while allowing flexible refinement.

By jointly optimizing the cycle consistency loss and the constraint term, we can achieve a balanced learning between intra-video temporal consistency and inter-video semantic separability. The final training objective for g is formulated as Eq. (4), where λ controls the strength of the constraint, and we will provide a justification for λ in the Sec. 4.

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\text { cyc }} + \lambda \mathcal {L} _ {\text { reg }}. \tag {4}
$$

# 4. Theoretical Analysis

In this section, we explore how the proposed method achieves our target, i.e., improving the intra-video temporal consistency without largely affecting the inter-video semantic separability. Generally, our analysis leads to two main conclusions: a) Within our proposed method, both linearbased and MLP projection rebalance different dimensions of the representation space in a similar mechanism (Theorem 1). b) This rebalance yields a better trade-off between the two properties under appropriate conditions (Theorem 2). This section is self-contained and may be skipped without affecting the overall understanding of our framework. The detailed derivations are provided in the Supplementary Material.

Formally, given the original representation of a patch $z _ { i } \in \mathbb { R } ^ { d }$ , we aim to learn a projection g that maps $z _ { i }$ to $\pmb { p } _ { i } = \pmb { g } ( \pmb { z } _ { i } ) \in \mathbb { R } ^ { d }$ . Since directly analyzing the original objectives in Eq. (2) and Eq. (3) is challenging, we introduce simplified yet equivalent surrogates to facilitate the analysis.

Objective 1 (Temporal Cycle Consistency). This term encourages alignment between temporally corresponding patches. We quantify it with the metric in Eq. (5). Note that minimizing $M _ { \mathrm { c y c } }$ is equivalent to minimizing the cycleconsistency loss $L _ { \mathrm { c y c } } ,$ since both decrease as temporal consistency improves and share the same optimality conditions.

$$
M _ {\text { cyc }} = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \big [ \| g (\boldsymbol {z} _ {1}) - g (\boldsymbol {z} _ {2}) \| ^ {2} \big ]. \tag {5}
$$

Objective 2 (Semantic Separability Constraint): The KL divergence constraint in Eq. (3) preserves the distance relationships between patches before and after the projection, which is equivalent to constraining the projection to be isometric. This property can be measured by the orthogonality of the Jacobian matrix [16, 49, 56, 60] of $^ { g , }$ as formulated in Eq. (6). Therefore, we use it as an approximation of $L _ { \mathrm { r e g } }$

$$
M _ {\text { reg }} = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {i}} \big [ \| \boldsymbol {J} _ {g} (\boldsymbol {z} _ {i}) \boldsymbol {J} _ {g} (\boldsymbol {z} _ {i}) ^ {\top} - \boldsymbol {I} \| _ {F} ^ {2} \big ]. \tag {6}
$$

Combining the two surrogates yields the overall objective:

$$
\min _ {g} M (g) = M _ {\text { cyc }} + \lambda M _ {\text { reg }}. \tag {7}
$$

We now consider two representative cases for g: i) A linear projection: $g ( z ) = W z ; \mathbf { i } ) \mathbf { A }$ two-layer $\begin{array} { r l } { \mathbf { M L P : } g ( \pmb { z } ) = } \end{array}$ $W _ { 2 } \phi ( W _ { 1 } z )$ with activation function $\phi ( \cdot ) = t a n h ( \cdot )$ , and this case represents more complex modules. The following theorem analyzes the spectral properties of the optimal solution under both cases, illustrating how the projection affects the quality of the transferred representation.

Theorem 1 (Spectral Properties of Optimal Projections, Informal). Denote the eigenvalues of intra-video covariance matrix Σ are $\{ \sigma _ { i } \} _ { i = 1 } ^ { d }$ . For case i), let $\{ \mu _ { i } \} _ { i = 1 } ^ { d }$ be the eigenvalues of W . Assume W and Σ are positive semi-definite. For case ii), let $\{ \mu _ { 1 , i } \} _ { i = 1 } ^ { d }$ and $\{ \mu _ { 2 , i } \} _ { i = 1 } ^ { d }$ be the eigenvalues $o f W _ { 1 }$ and $W _ { 2 } ,$ , respectively. Assume $\boldsymbol { z } _ { i } \sim \mathcal { N } ( \mathbf { 0 } , \boldsymbol { \Sigma } ) , \boldsymbol { W } _ { 1 }$ , $W _ { 2 }$ and Σ are positive semi-definite. Then the eigenvalues of the optimal projection $W ^ { \star }$ , $W _ { 1 } { } ^ { \star }$ and $W _ { 2 } ^ { \star }$ obey:

$$
\mu_ {i} ^ {\star} = \mu_ {1, i} ^ {\star} \cdot \mu_ {2, i} ^ {\star} = \left\{ \begin{array}{l l} 0, & \sigma_ {i} > 2 \lambda , \\ \sqrt {1 - \frac {\sigma_ {i}}{2 \lambda}}, & \sigma_ {i} \leq 2 \lambda . \end{array} \right. \tag {8}
$$

Remark 1. Theorem 1 reveals a soft thresholding behavior for both cases. Directions with large temporal variance $( \sigma _ { i } > 2 \lambda )$ are suppressed $( \mu _ { i } ^ { \star } = 0 \mathrm { o r } \mu _ { 1 , i } ^ { \star } \cdot \mu _ { 2 , i } ^ { \star } = 0 )$ , those with smaller variance are gradually scaled toward unit norm. Intuitively, components that already exhibit strong temporal consistency require less adjustment, whereas low-variance directions are amplified until the orthogonality penalty balances the consistency gain. For the two-layer MLP, when the inputs fall within the approximate linear region of the tanh function, the overall transformation presents similar representation scaling behavior as the linear layer. Similar conclusions hold for more complex architectures.

Since both cases yield similar spectral effects, we focus on examining whether a single linear layer is sufficient to improve the trade-off between temporal consistency and semantic separability. To this end, we first define two metrics to quantify these two competing objectives: 1) Intra-video distance: $D _ { i n t r a } ( z _ { 1 } , z _ { 2 } ) = \mathbb { E } _ { z _ { 1 } , z _ { 2 } } \left[ \| z _ { 1 } - z _ { 2 } \| ^ { 2 } \right]$ , which measures the average distance between temporally corresponding patches within a video. 2) Inter-video distance: $D _ { i n t e r } ( z _ { 1 } , z _ { 2 } ) = \mathbb { E } _ { \bar { z } _ { 1 } , \bar { z } _ { 2 } } \left\lceil \left\| \bar { z } _ { 1 } - \bar { z } _ { 2 } \right\| ^ { 2 } \right\rceil$ , calculating the average distance between video-level representations, where $\bar { z } _ { i } = \mathbb { E } _ { z \in f ( V _ { i } ) } \left[ z \right]$ is the mean representation of the video $V _ { i }$ . Then we define the margin of these two metrics as $D ( z _ { 1 } , z _ { 2 } ) = D _ { i n t e r } ( z _ { 1 } , z _ { 2 } ) - \gamma D _ { i n t r a } ( z _ { 1 } , z _ { 2 } )$ , reflecting the degree of separation between the two properties, where a larger value indicates a better trade-off. In the following theorem, we present how this margin metric evolves.

Theorem 2 (Trade-off Improvement, Informal). Let $\mathbf { \delta E } =$ $\mathbb { E } _ { z _ { 1 } , z _ { 2 } } [ ( z _ { 1 } - z _ { 2 } ) ( z _ { 1 } - z _ { 2 } ) ^ { \top } ] , \bar { \Sigma } = \mathbb { E } _ { \bar { z } _ { 1 } , \bar { z } _ { 2 } } [ ( \bar { z } _ { 1 } - \bar { z } _ { 2 } ) ( \bar { z } _ { 1 } -$ $\bar { z } _ { 2 } ) ^ { \top } ]$ denote the intra-video and inter-video covariance ma-Assume trices, with eigenvalues $\begin{array} { r } { \forall j , \tau _ { j } = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \sigma _ { i } = \tau } \end{array}$ P $\{ \sigma _ { i } \} _ { i = 1 } ^ { d }$ and i=1. For the linear projection $\{ \tau _ { i } \} _ { i = 1 } ^ { d }$ , respectively. $\pmb { p } = g ( \pmb { z } ) = \pmb { W } \pmb { z }$ with eigenvalues of the optimal $W ^ { \star }$ are $\begin{array} { r } { \mu _ { i } ^ { \star } = \sqrt { 1 - \frac { \sigma _ { i } } { 2 \lambda } } } \end{array}$ (when $\sigma _ { i } \le 2 \lambda )$ , the improvement in the margin metric $\dot { \Delta } = D ( \pmb { p } _ { 1 } , \pmb { p } _ { 2 } ) - D ( z _ { 1 } , z _ { 2 } )$ is given by:

$$
\Delta = \sum_ {\sigma_ {i} \leq 2 \lambda} (\tau - \sigma_ {i}) \left(1 - \frac {\sigma_ {i}}{2 \lambda}\right) > 0. \tag {9}
$$

Remark 2. Theorem 2 indicates that the margin metric between inter- and intra-video distance can be enhanced under the optimal linear projection, thus yielding a better trade-off between temporal consistency and semantic separability.

# 5. Experiments

In this section, we first introduce the implementation details in Sec. 5.1. Then, we evaluate the effectiveness and efficiency of our framework on multiple downstream tasks in Sec. 5.2 and justify with distance metrics in Sec. 5.3. Finally, we perform ablation studies in Sec. 5.4 to assess the impact of key configurations. Additional implementation details and results are provided in the Supplementary Material.

# 5.1. Implementation details

Fundamental image models. We evaluate our method across eight pretrained image encoders grouped into three categories: 1) Masked modeling: MAE [45], I-JEPA [3]; 2)

Table 1. Comparison with image and video representation learning methods on three dense-level video downstream tasks. Results marked with † are reported from prior works; ‡ denotes evaluations based on the official pretrained weights. The best results are highlighted in bold. INet, K400, WIT, and LAION represent ImageNet-1k [25], Kinetics-400 [55], WIT-400M [82], and LAION-400M [87] datasets. 

<table><tr><td rowspan="2" colspan="2">Type</td><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td rowspan="2">Dataset</td><td rowspan="2">Epoch</td><td rowspan="2">VIPmIoU</td><td colspan="3">DAVIS-2017</td><td colspan="2">JHMDB</td></tr><tr><td> $\mathcal{J}\&\mathcal{F}_{\mathrm{m}}$ </td><td> $\mathcal{J}_{\mathrm{m}}$ </td><td> $\mathcal{F}_{\mathrm{m}}$ </td><td>PCK@0.1</td><td>PCK@0.2</td></tr><tr><td rowspan="6" colspan="2">Video Pretrained</td><td>VideoMAE $^{\ddagger}$ [101]</td><td>ViT-L/16</td><td>K400</td><td>1600</td><td>25.6</td><td>45.0</td><td>43.6</td><td>46.5</td><td>43.3</td><td>70.5</td></tr><tr><td>MAE-ST $^{\dagger }$ [33]</td><td>ViT-L/16</td><td>K400</td><td>1600</td><td>33.2</td><td>54.6</td><td>55.5</td><td>53.6</td><td>44.4</td><td>72.5</td></tr><tr><td>DropMAE $^{\dagger }$ [111]</td><td>ViT-B/16</td><td>K400</td><td>1600</td><td>31.1</td><td>53.4</td><td>51.8</td><td>55.0</td><td>42.3</td><td>69.2</td></tr><tr><td>SiamMAE [39]</td><td>ViT-B/16</td><td>K400</td><td>400</td><td>36.1</td><td>60.9</td><td>59.4</td><td>62.4</td><td>47.0</td><td>74.9</td></tr><tr><td>CropMAE $^{\dagger }$ [30]</td><td>ViT-B/16</td><td>K400</td><td>400</td><td>33.0</td><td>57.8</td><td>56.9</td><td>58.7</td><td>45.3</td><td>73.3</td></tr><tr><td>RSP $^{\dagger }$ [52]</td><td>ViT-B/16</td><td>K400</td><td>400</td><td>34.0</td><td>60.5</td><td>57.8</td><td>63.2</td><td>46.0</td><td>74.6</td></tr><tr><td rowspan="16">Image Pretrained +Ours</td><td rowspan="4">Mask Modeling</td><td>MAE $^{\ddagger }$ [45]</td><td>ViT-B/16</td><td>INet</td><td>800</td><td>29.3</td><td>52.4</td><td>51.0</td><td>53.9</td><td>41.6</td><td>69.3</td></tr><tr><td>MAE +Ours</td><td>ViT-B/16</td><td>K400</td><td>+5</td><td>33.8</td><td>59.6</td><td>58.0</td><td>61.2</td><td>48.4</td><td>76.7</td></tr><tr><td>I-JEPA [3]</td><td>ViT-B/16</td><td>INet</td><td>800</td><td>31.5</td><td>53.9</td><td>52.9</td><td>54.8</td><td>42.6</td><td>71.4</td></tr><tr><td>I-JEPA +Ours</td><td>ViT-B/16</td><td>K400</td><td>+5</td><td>35.3</td><td>58.7</td><td>57.4</td><td>59.9</td><td>44.4</td><td>73.2</td></tr><tr><td rowspan="6">Contrastive Learning</td><td>CLIP $^{\ddagger }$ [82]</td><td>ViT-B/16</td><td>WIT</td><td>32</td><td>38.1</td><td>54.9</td><td>53.4</td><td>56.4</td><td>36.9</td><td>67.7</td></tr><tr><td>CLIP +Ours</td><td>ViT-B/16</td><td>K400</td><td>+5</td><td>39.2</td><td>58.3</td><td>57.0</td><td>59.7</td><td>40.6</td><td>71.4</td></tr><tr><td>BLIP $^{\ddagger }$ [62]</td><td>ViT-B/16</td><td>LAION</td><td>20</td><td>37.6</td><td>58.8</td><td>57.3</td><td>60.3</td><td>35.1</td><td>65.9</td></tr><tr><td>BLIP +Ours</td><td>ViT-B/16</td><td>K400</td><td>+5</td><td>39.6</td><td>62.0</td><td>60.2</td><td>63.8</td><td>38.9</td><td>70.2</td></tr><tr><td>MoCo v3 $^{\ddagger }$ [20]</td><td>ViT-B/16</td><td>INet</td><td>300</td><td>38.8</td><td>62.6</td><td>60.0</td><td>65.1</td><td>43.6</td><td>73.5</td></tr><tr><td>MoCo v3 +Ours</td><td>ViT-B/16</td><td>K400</td><td>+5</td><td>39.8</td><td>62.9</td><td>60.6</td><td>65.2</td><td>45.3</td><td>75.0</td></tr><tr><td rowspan="6">Self-Distillation</td><td>iBOT $^{\ddagger }$ [120]</td><td>ViT-B/16</td><td>INet</td><td>400</td><td>39.6</td><td>64.6</td><td>63.0</td><td>66.1</td><td>45.7</td><td>75.3</td></tr><tr><td>iBOT +Ours</td><td>ViT-B/16</td><td>K400</td><td>+5</td><td>40.8</td><td>65.1</td><td>63.3</td><td>66.9</td><td>46.1</td><td>75.8</td></tr><tr><td>DINO $^{\ddagger }$ [14]</td><td>ViT-B/16</td><td>INet</td><td>300</td><td>39.1</td><td>63.2</td><td>60.9</td><td>65.5</td><td>44.4</td><td>74.2</td></tr><tr><td>DINO +Ours</td><td>ViT-B/16</td><td>K400</td><td>+5</td><td>39.8</td><td>64.2</td><td>62.3</td><td>66.0</td><td>46.2</td><td>75.2</td></tr><tr><td>DINO v2 [73]</td><td>ViT-B/16</td><td>INet</td><td>100</td><td>38.4</td><td>63.1</td><td>61.6</td><td>64.5</td><td>46.6</td><td>76.3</td></tr><tr><td>DINO v2 +Ours</td><td>ViT-B/16</td><td>K400</td><td>+5</td><td>39.9</td><td>63.7</td><td>61.9</td><td>65.4</td><td>47.3</td><td>76.8</td></tr></table>

![](images/49514d4458446556d0c78facec382d874e747d23b24ea622416d2e42ed082c65.jpg)  
Figure 4. Evaluation results on frame-level and video-level tasks based on four representative image models.

Contrastive learning: CLIP [82], BLIP [62], MoCo v3 [20]; 3) Self-distillation: iBOT [120], DINO [14], DINO v2 [73]. Optimizing strategies. During training, we only update the projection layer $g$ with feature dimension $d = 7 6 8$ while the pretrained image encoders are kept frozen. The layer g is trained on the Kinetics-400 dataset for 5 epochs with a total batch size of 512 on 4 RTX4090 GPUs. Parameters optimization is performed by AdamW [71] with a basic learning rate of $1 \times 1 0 ^ { - 4 }$ and a cosine decay schedule.

# 5.2. Main results on different types of video tasks

# 5.2.1. Evaluation on dense-level benchmarks

Experiment setup. We evaluate the representations on three dense-level video benchmarks: video object segmentation on DAVIS-2017 [79], human part segmentation on VIP [121], and pose propagation on JHMDB [53]. All evaluations are under a zero-shot semi-supervised protocol [30, 39, 52, 69].

Quantitative results. The evaluation results on three denselevel benchmarks are reported in Tab. 1, from which we make the following conclusions: 1) Our method consistently improves performance across all eight fundamental image models, demonstrating its effectiveness in self-supervised image-to-video representation transfer. Specifically, the image-pretrained encoders yield average improvements of 1.98% mIoU on VIP, 2.63% $\mathcal { T } \& \mathcal { F } _ { \mathrm { m } }$ on DAVIS, and 2.59% PCK@0.1 on JHMDB. 2) The improvements hold across three categories of image models and, notably, also extend to supervised multimodal models (e.g., CLIP, BLIP), showing broad applicability to ViT-based image encoders under different pretraining paradigms. 3) Under comparable parameter scales, the transferred models match or surpass recent self-supervised video representation models while using simpler modules and lower training overhead, thus offering a lightweight alternative for dense-level video understanding.

Table 2. Results of integration into video dense tracking pipeline. 

<table><tr><td rowspan="2">Features</td><td colspan="2">BADJA</td></tr><tr><td> $\delta^{seg}$ </td><td> $\delta^{3px}$ </td></tr><tr><td>DINO v2</td><td>62.73</td><td>8.85</td></tr><tr><td>+Ours</td><td>70.52</td><td>8.86</td></tr><tr><td colspan="3"></td></tr><tr><td rowspan="2">Features</td><td colspan="2">TAP-DAVIS</td></tr><tr><td> $\delta^{x}_{avg}$ </td><td>OA</td></tr><tr><td>DINO v2</td><td>64.68</td><td>81.98</td></tr><tr><td>+Ours</td><td>64.02</td><td>85.40</td></tr></table>

![](images/6769c508e646052499e53e6cbdfb821cfedb382e53b8187f3be144b3c9c84c8b.jpg)

![](images/d427320d8baf1c0a21a7ae26473950d555acc4783c93123b05bb96e1c54c06a8.jpg)

![](images/fa105bfad30ee031fee19c142d04c78060e427ce02201fd83f5ea8f4da477714.jpg)

![](images/679133a2d92e9f752bb6bb8483186b94be5df13301812d3ccfd12405a08c541e.jpg)

Table 3. Efficiency and performance comparison with image-tovideo transfer methods. † marks the costs from prior studies. 

<table><tr><td>Method</td><td>Tunable Params (↓)</td><td>Peak GPU Mem. (↓)</td><td>Tuning GPU Hours (↓)</td><td>VIP  $\mathcal{J}\&\mathcal{F}_{m}$ </td><td>DAVIS17 mIoU</td><td>JHMDB PCK@0.1</td></tr><tr><td>MAE</td><td>/</td><td>/</td><td>/</td><td>29.3</td><td>52.4</td><td>41.6</td></tr><tr><td>Full Fine-tuning</td><td>111.66 M</td><td>13.9 GB</td><td> $20.1 \times \text{RTX4090}$ </td><td>29.8</td><td>53.9</td><td>42.6</td></tr><tr><td>Partial Fine-tuning</td><td>7.09 M</td><td>9.2 GB</td><td> $15.6 \times \text{RTX4090}$ </td><td>29.2</td><td>52.7</td><td>41.3</td></tr><tr><td>I2V Adapter</td><td>14.22 M</td><td>20.4 GB</td><td> $21.2 \times \text{RTX4090}$ </td><td>29.5</td><td>53.0</td><td>41.5</td></tr><tr><td>Ours</td><td>0.59 M</td><td>4.8 GB</td><td> $1.2 \times \text{RTX4090}$ </td><td>33.8</td><td>59.6</td><td>48.4</td></tr><tr><td>CLIP</td><td>/</td><td>/</td><td>/</td><td>38.1</td><td>54.9</td><td>36.9</td></tr><tr><td>AIM $^{\dagger}$  [115]</td><td>11.00 M</td><td>8.7 GB</td><td> $120 \times \text{V100}$ </td><td>34.2</td><td>51.6</td><td>35.8</td></tr><tr><td>ST-Adapter $^{\dagger}$  [74]</td><td>7.42 M</td><td>6.9 GB</td><td> $23 \times \text{V100}$ </td><td>36.5</td><td>54.4</td><td>37.5</td></tr><tr><td>Zerol2V $^{\dagger}$  [66]</td><td>14.00 M</td><td>7.6 GB</td><td> $100 \times \text{V100}$ </td><td>37.2</td><td>54.8</td><td>37.2</td></tr><tr><td>Ours</td><td>0.59 M</td><td>5.2 GB</td><td> $1.2 \times \text{RTX4090}$ </td><td>39.2</td><td>58.3</td><td>40.6</td></tr></table>

# 5.2.2. Evaluation on frame-level and video-level tasks

Experiment setup. We evaluate the models on several frame- and video-level tasks: temporal action localization on Breakfast [59] with the FACT [72] backbone, zero-shot video retrieval on UCF101 and HMDB51 [58, 93], action classification on Something-Something-v2 (SSV2) [36], and temporal order discrimination on Chiral SSV2 [6]. For SSV2, the models are fine-tuned on the training set for 25 epochs before evaluation on the validation set with single-clip sampling. For Chiral SSV2, we concatenate frame embeddings along the temporal dimension and train a linear probe.

Quantitative results. Figure 4 depicts the performance of transferred representations from four representative image models on both frame- and video-level downstream tasks. Our method consistently improves performance across these tasks. For instance, on the frame-level Breakfast task, it achieves an average gain of 2.80% Acc, indicating enhanced temporal awareness in the image models. On video-level tasks, it achieves a 2.58% R@1 improvement on HMDB51, a 1.53% Acc@1 gain on SSV2, and a 1.25% Acc gain on Chiral SSV2. These results reveal that our method generalizes well across different task granularities, highlighting its potential as a versatile solution for image-to-video transfer.

# 5.2.3. Integration into existing video pipeline

Experiment setup. We further explore the integration of the transferred representations into existing video analysis pipelines. Specifically, for the complex task of dense point tracking, we replace the original DINOv2 features in the DINO-Tracker [103] framework with our transferred DI-NOv2 representations and evaluate the performance on the BADJA [11] and TAP-DAVIS [26] tracking benchmarks.

Evaluation results. As shown in Tab. 2, our method

Table 4. Validation results on distance-based trade-off metrics. 

<table><tr><td>Type</td><td>Method</td><td> $D_{inter}$ </td><td> $D_{intra}$ </td><td>D(↑)</td><td>Cyc. Acc. (↑)</td></tr><tr><td rowspan="3">Video Pretrained</td><td>SiamMAE</td><td>0.5067</td><td>0.1330</td><td>0.4668</td><td>0.4100</td></tr><tr><td>CropMAE</td><td>0.5216</td><td>0.1736</td><td>0.4695</td><td>0.6522</td></tr><tr><td>RSP</td><td>0.4662</td><td>0.2130</td><td>0.4023</td><td>0.5990</td></tr><tr><td rowspan="16">Image Pretrained +Ours</td><td>MAE</td><td>0.3122</td><td>0.1131</td><td>0.2783</td><td>0.1366</td></tr><tr><td>MAE +Ours</td><td>0.5073</td><td>0.1834</td><td>0.4523</td><td>0.7203</td></tr><tr><td>I-JEPA</td><td>0.2572</td><td>0.1425</td><td>0.2145</td><td>0.1192</td></tr><tr><td>I-JEPA +Ours</td><td>0.5904</td><td>0.1745</td><td>0.5380</td><td>0.5906</td></tr><tr><td>CLIP</td><td>0.5603</td><td>0.2186</td><td>0.4947</td><td>0.3002</td></tr><tr><td>CLIP +Ours</td><td>0.6162</td><td>0.2626</td><td>0.5374</td><td>0.4608</td></tr><tr><td>BLIP</td><td>0.5858</td><td>0.1598</td><td>0.5378</td><td>0.2245</td></tr><tr><td>BLIP +Ours</td><td>0.6102</td><td>0.2457</td><td>0.5365</td><td>0.4527</td></tr><tr><td>MoCo v3</td><td>0.5547</td><td>0.2164</td><td>0.4898</td><td>0.3770</td></tr><tr><td>MoCo v3 +Ours</td><td>0.5503</td><td>0.1909</td><td>0.4930</td><td>0.5981</td></tr><tr><td>iBOT</td><td>0.6143</td><td>0.1862</td><td>0.5584</td><td>0.4488</td></tr><tr><td>iBOT +Ours</td><td>0.6399</td><td>0.2092</td><td>0.5772</td><td>0.5731</td></tr><tr><td>DINO</td><td>0.5756</td><td>0.2144</td><td>0.5112</td><td>0.4590</td></tr><tr><td>DINO +Ours</td><td>0.6246</td><td>0.2316</td><td>0.5551</td><td>0.5927</td></tr><tr><td>DINO v2</td><td>0.5926</td><td>0.1808</td><td>0.5384</td><td>0.4276</td></tr><tr><td>DINO v2 +Ours</td><td>0.6373</td><td>0.1976</td><td>0.5780</td><td>0.5383</td></tr></table>

achieves overall improvements on both datasets. The tracking trajectories suggest that our representations provide enhanced spatiotemporal coherence and better occlusion handling. These findings provide preliminary evidence of its potential for integration into existing video processing frameworks for its application in more real-world scenarios.

# 5.2.4. Efficiency analysis

Experiment setup. We compare our method with several image-to-video transfer methods, all of which involve posttraining or fine-tuning on Kinetics-400. We first compare with three common strategies: full fine-tuning, partial finetuning (i.e., updating only the final Transformer block), and the I2V adapter [74] applied to each block. Then we compare our method with CLIP-based supervised adaptation methods. Evaluation results. As shown in Tab. 3, the baseline methods yield suboptimal performance, primarily due to degraded discriminability caused by the limited semantic diversity of the video dataset. In contrast, our method delivers superior results with a ∼ 13× speed-up while updating only 0.59 M parameters. Additionally, our method outperforms CLIPbased adaptive baselines with lower training cost. These baselines emphasize global semantic separability but lack explicit temporal correspondence for dense video understanding. Overall, on dense-level tasks, our method delivers improved performance over common image-to-video transfer baselines while maintaining high computational efficiency. These outcomes also indicate that the performance gain chiefly arises from a more favorable trade-off between intra-video temporal consistency and inter-video semantic separability instead of image-to-video domain adaptation.

# 5.3. Distance-based trade-off metrics validation

Experiment setup. To provide an interpretable assessment of the method’s effectiveness, we validate the distance-based metrics proposed in Sec. 4. Specifically, we randomly sample 1000 videos from the Kinetics-400 validation set and measure four metrics for the original image-pretrained models and our transferred models. The metrics includes: 1) Inter-video distance $D _ { i n t e r } ~ = ~ D _ { i n t e r } ^ { o r i } / 2 R _ { i n t e r }$ Doriinter/2Rinter (normalized by its diameter); 2) Intra-video distance $D _ { i n t r a } ~ =$ $D _ { i n t r a } ^ { o r i } / 2 R _ { i n t r a }$ (normalized by its diameter); 3) Distance margin $D \ = \ D _ { i n t e r } \ - \gamma D _ { i n t r a }$ where the scale factor $\gamma = \mathbb { E } _ { \mathcal { M } } \left[ D _ { i n t r a } ^ { o r i } \right] / \mathbb { E } _ { \mathcal { M } } \left[ D _ { i n t e r } ^ { o r i } \right]$ is the average ratio of the original intra-/inter-video distance for each model $\mathcal { M } ; \pmb { 4 } )$ Cycle consistency accuracy $( C y c . A c c . )$ , defined as the proportion of patches that return to their original positions under a palindrome sequence constructed from two frames.

Table 5. Ablation results on the components, structures, and training configurations of our method. The best and secondary best results are highlighted in bold and underlined. Default settings are marked with blue .   
(a) Ablation on components of $\mathcal { L } _ { c y c } , \mathcal { L } _ { r e g }$ and PEA. 

<table><tr><td>Base Model</td><td> $\mathcal{L}_{cyc}$ </td><td> $\mathcal{L}_{reg}$ </td><td>PEA</td><td>VIP mIoU</td><td>DAVIS17  $\mathcal{J}\&\mathcal{F}_{m}$ </td><td>JHMDB PCK@0.1</td></tr><tr><td rowspan="4">MAE</td><td>√</td><td></td><td></td><td>16.2</td><td>26.2</td><td>38.5</td></tr><tr><td>√</td><td>√</td><td></td><td>23.1</td><td>42.3</td><td>42.4</td></tr><tr><td>√</td><td></td><td>√</td><td>33.3</td><td>59.3</td><td>48.1</td></tr><tr><td>√</td><td>√</td><td>√</td><td>33.8</td><td>59.6</td><td>48.4</td></tr><tr><td rowspan="4">DINO</td><td>√</td><td></td><td></td><td>17.5</td><td>30.6</td><td>39.3</td></tr><tr><td>√</td><td>√</td><td></td><td>33.6</td><td>58.9</td><td>46.4</td></tr><tr><td>√</td><td></td><td>√</td><td>38.0</td><td>61.8</td><td>46.1</td></tr><tr><td>√</td><td>√</td><td>√</td><td>39.8</td><td>64.2</td><td>46.2</td></tr></table>

(b) Ablation on the projection layer structures. 

<table><tr><td>Base Model</td><td>Projection Structure</td><td>VIP mIoU</td><td>DAVIS17 J&amp;Fm</td><td>JHMDB PCK@0.1</td></tr><tr><td rowspan="4">MAE</td><td>Vanilla</td><td>29.3</td><td>52.4</td><td>41.6</td></tr><tr><td>Linear-based layer</td><td>33.8</td><td>59.6</td><td>47.9</td></tr><tr><td>MLP (2 layers)</td><td>33.2</td><td>59.2</td><td>47.5</td></tr><tr><td>MLP (3 layers)</td><td>32.6</td><td>58.4</td><td>47.4</td></tr><tr><td rowspan="4">DINO</td><td>Vanilla</td><td>39.1</td><td>63.2</td><td>44.4</td></tr><tr><td>Linear-based layer</td><td>39.8</td><td>64.2</td><td>46.2</td></tr><tr><td>MLP (2 layers)</td><td>39.7</td><td>64.1</td><td>45.9</td></tr><tr><td>MLP (3 layers)</td><td>39.9</td><td>63.8</td><td>45.7</td></tr></table>

(c) Ablation on training dataset. 

<table><tr><td>Base Model</td><td>Training Dataset</td><td>VIP mIoU</td><td>DAVIS17 J&amp;Fm</td><td>JHMDB PCK@0.1</td></tr><tr><td rowspan="2">MAE</td><td>K400</td><td>33.8</td><td>59.6</td><td>48.4</td></tr><tr><td>SSV2</td><td>33.6</td><td>60.4</td><td>48.2</td></tr><tr><td rowspan="2">BLIP</td><td>K400</td><td>39.6</td><td>62.0</td><td>38.9</td></tr><tr><td>SSV2</td><td>38.2</td><td>60.0</td><td>37.7</td></tr><tr><td rowspan="2">DINO</td><td>K400</td><td>39.8</td><td>64.2</td><td>46.2</td></tr><tr><td>SSV2</td><td>39.4</td><td>63.8</td><td>45.8</td></tr></table>

(d) Ablation on the backbone scales. 

<table><tr><td>Backbone Model</td><td>Method</td><td>VIP mIoU</td><td>DAVIS17  $\mathcal{J}\&\mathcal{F}_{m}$ </td><td>JHMDB PCK@0.1</td></tr><tr><td rowspan="2">ViT-B/16</td><td>MAE</td><td>29.3</td><td>52.4</td><td>41.6</td></tr><tr><td>+Ours</td><td>33.8</td><td>59.6</td><td>48.4</td></tr><tr><td rowspan="2">ViT-L/16</td><td>MAE</td><td>29.9</td><td>55.8</td><td>44.6</td></tr><tr><td>+Ours</td><td>33.4</td><td>59.9</td><td>48.9</td></tr><tr><td rowspan="2">ViT-H/14</td><td>MAE</td><td>29.5</td><td>55.8</td><td>/</td></tr><tr><td>+Ours</td><td>33.4</td><td>60.1</td><td>/</td></tr></table>

(e) Ablation on the training epochs. 

<table><tr><td>Base Model</td><td>Training Epochs</td><td>VIP mIoU</td><td>DAVIS17 J&amp;Fm</td><td>JHMDB PCK@0.1</td></tr><tr><td rowspan="3">MAE</td><td>5</td><td>33.8</td><td>59.6</td><td>48.4</td></tr><tr><td>10</td><td>33.6</td><td>59.4</td><td>48.3</td></tr><tr><td>20</td><td>33.5</td><td>59.1</td><td>48.1</td></tr><tr><td rowspan="3">DINO</td><td>5</td><td>39.8</td><td>64.2</td><td>46.2</td></tr><tr><td>10</td><td>39.9</td><td>64.1</td><td>46.3</td></tr><tr><td>20</td><td>40.0</td><td>64.0</td><td>46.3</td></tr></table>

Evaluation results. As shown in Tab. 4, our method generally increases the margin D by widening the gap between inter- and intra-video distances. This yields a better trade-off between intra-video temporal consistency and inter-video semantic separability, supporting the conclusion of Theorem 2. In addition, our method significantly improves cycle consistency accuracy to a level comparable to video-pretrained models, indicating the effectiveness of introducing dense temporal correspondence into image-pretrained models.

# 5.4. Ablation Study

In this part, we conduct ablation studies to assess the contribution of each component in our method. All settings are kept consistent across variants, except for the ablated factors.

Analysis of components. We first examine the effect of cycle consistency loss, regularization loss, and PEA strategy. As shown in Tab. 5a, applying cycle consistency loss without the PEA strategy leads to obvious performance degradations, as the model exploits positional shortcuts (line 1-2). With the PEA strategy enabled (line 3), the model learns meaningful temporal correspondences, showing that cycle consistency

becomes effective only when shortcuts are suppressed. Additionally, incorporating the regularization term (line 4) yields further improvements, highlighting the importance of preserving semantic separability during transfer learning.

Effect of projection structure. We compare different designs of the projection structure, including a linear-based layer and MLPs with 2 or 3 layers in Tab. 5b. Empirically, the linear-based layer achieves equal or superior performance relative to the deeper MLPs, likely because more complex projections are prone to perturbing the semantic structure of image-pretrained representations. These results indicate that a simple linear projection is often sufficient, which is consistent with the theoretical analysis in Theorem 1.

Generalization on training configurations. We evaluate the generalization ability of our method under different training configurations as shown in Tabs. 5c to 5e. In addition to K400, training on the SSV2 dataset yields consistent improvements, demonstrating the robustness of the proposed method on datasets with stronger temporal dynamics. Scaling up the backbone from ViT-B to ViT-L and ViT-H still enhances downstream performance, indicating its adaptability with larger models. Moreover, our model converges within 5 epochs, which is selected as the default setting.

# 6. Conclusion

This work explores self-supervised image-to-video transfer learning for an effective trade-off between intra-video temporal consistency and inter-video semantic separability. We propose Co-Settle framework to project the representation space via a lightweight layer and provide a theoretical analysis. Experimental results with eight image models present the effectiveness of Co-Settle across multiple video tasks.

# Acknowledgments

This work was supported in part by National Natural Science Foundation of China: 62525212, U23B2051, 62236008, 62441232, 62521007 and U21B2038, in part by Youth Innovation Promotion Association CAS, in part by the Strategic Priority Research Program of the Chinese Academy of Sciences, Grant No. XDB0680201, in part by the project ZR2025ZD01 supported by Shandong Provincial Natural Science Foundation, in part by the China National Postdoctoral Program for Innovative Talents under Grant BX20250377, and in part by the Beijing Major Science and Technology Project under Contract No. Z251100008125059. This work was supported by Beijing Academy of Artificial Intelligence (BAAI).

# References

[1] Anurag Arnab, Mostafa Dehghani, Georg Heigold, Chen Sun, Mario Luciˇ c, and Cordelia Schmid. Vivit: A video vi-´ sion transformer. In International Conference on Computer Vision, pages 6836–6846, 2021. 1, 31   
[2] Mahmoud Assran, Randall Balestriero, Quentin Duval, Florian Bordes, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, and Nicolas Ballas. The hidden uniform cluster prior in self-supervised learning. International Conference on Learning Representations, 2023. 1   
[3] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15619–15629, 2023. 2, 5, 6, 24, 31   
[4] Fred Attneave. Some informational aspects of visual perception. Psychological review, 61(3):183, 1954. 2, 31   
[5] Alan Baade and Changan Chen. Self-supervised crossview correspondence with predictive cycle consistency. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16753–16763, 2025. 2   
[6] Piyush Bagad and Andrew Zisserman. Chirality in action: Time-aware video representation learning by latent straightening. arXiv preprint arXiv:2509.08502, 2025. 7, 23, 26   
[7] Hangbo Bao, Li Dong, Songhao Piao, and Furu Wei. Beit: Bert pre-training of image transformers. 2022. 2, 31   
[8] Amir Bar, Florian Bordes, Assaf Shocher, Mido Assran, Pascal Vincent, Nicolas Ballas, Trevor Darrell, Amir Globerson, and Yann LeCun. Stochastic positional embeddings improve masked image modeling. In International Conference on Machine Learning, 2024. 2, 31   
[9] Gedas Bertasius, Heng Wang, and Lorenzo Torresani. Is space-time attention all you need for video understanding? In International Conference on Machine Learning, 2021. 1, 31   
[10] Zhangxing Bian, Allan Jabri, Alexei A Efros, and Andrew Owens. Learning pixel trajectories with multiscale contrastive random walks. In IEEE/CVF Conference on Com-

puter Vision and Pattern Recognition, pages 6508–6519, 2022. 2, 3, 31   
[11] Benjamin Biggs, Thomas Roddick, Andrew Fitzgibbon, and Roberto Cipolla. Creatures great and smal: Recovering the shape and motion of animals from video. In Asian Conference on Computer Vision, pages 3–19. Springer, 2019. 2, 7, 31   
[12] Adrian Bulat, Juan Manuel Perez Rua, Swathikiran Sudhakaran, Brais Martinez, and Georgios Tzimiropoulos. Space-time mixing attention for video transformer. Advances in Neural Information Processing Systems, 34:19594– 19607, 2021. 1, 31   
[13] Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. Advances in Neural Information Processing Systems, 33: 9912–9924, 2020. 29   
[14] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jé- gou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In International Conference on Computer Vision, pages 9650–9660, 2021. 2, 4, 6, 24, 25, 26, 31   
[15] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017. 2, 3, 31   
[16] Nutan Chen, Alexej Klushyn, Francesco Ferroni, Justin Bayer, and Patrick Van Der Smagt. Learning flat latent manifolds with vaes. In International Conference on Machine Learning, pages 1587–1596, 2020. 5, 15   
[17] Shoufa Chen, Chongjian Ge, Zhan Tong, Jiangliu Wang, Yibing Song, Jue Wang, and Ping Luo. Adaptformer: Adapting vision transformers for scalable visual recognition. Advances in Neural Information Processing Systems, 35:16664–16678, 2022. 1, 2, 20, 31   
[18] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In International Conference on Machine Learning, pages 1597–1607. PMLR, 2020. 2, 29   
[19] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15750–15758, 2021. 4, 29   
[20] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In International Conference on Computer Vision, pages 9640– 9649, 2021. 2, 6, 24, 25, 29   
[21] Ho Kei Cheng, Yu-Wing Tai, and Chi-Keung Tang. Rethinking space-time networks with improved memory coverage for efficient video object segmentation. Advances in Neural Information Processing Systems, 34:11781–11794, 2021. 26   
[22] Ho Kei Cheng, Seoung Wug Oh, Brian Price, Joon-Young Lee, and Alexander Schwing. Putting the object back into video object segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3151–3161, 2024. 26   
[23] Siran Dai, Qianqian Xu, Peisong Wen, Yang Liu, and Qingming Huang. Exploring structural degradation in dense

representations for self-supervised learning. arXiv preprint arXiv:2510.17299, 2025. 4   
[24] Siran Dai, Qianqian Xu, Peisong Wen, Yang Liu, and Qingming Huang. Exploring non-contrastive self-supervised representation learning for image-based profiling. arXiv e-prints, pages arXiv–2506, 2025. 2   
[25] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 248–255. Ieee, 2009. 1, 6, 21   
[26] Carl Doersch, Ankush Gupta, Larisa Markeeva, Adria Recasens, Lucas Smaira, Yusuf Aytar, Joao Carreira, Andrew Zisserman, and Yi Yang. Tap-vid: A benchmark for tracking any point in a video. Advances in Neural Information Processing Systems, 35:13610–13626, 2022. 2, 7   
[27] Carl Doersch, Pauline Luc, Yi Yang, Dilara Gokay, Skanda Koppula, Ankush Gupta, Joseph Heyward, Ignacio Rocco, Ross Goroshin, João Carreira, et al. Bootstap: Bootstrapped training for tracking-any-point. In Asian Conference on Computer Vision, pages 3257–3274, 2024. 20, 31   
[28] Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2021. 2, 4, 24, 31   
[29] Debidatta Dwibedi, Yusuf Aytar, Jonathan Tompson, Pierre Sermanet, and Andrew Zisserman. Temporal cycleconsistency learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1801–1810, 2019. 2, 31   
[30] Alexandre Eymaël, Renaud Vandeghen, Anthony Cioppa, Silvio Giancola, Bernard Ghanem, and Marc Van Droogenbroeck. Efficient image pre-training with siamese cropped masked autoencoders. In European Conference on Computer Vision, 2024. 2, 6, 20, 21, 25, 26, 31   
[31] Mohammad Fahes, Tuan-Hung Vu, Andrei Bursuc, Patrick Pérez, and Raoul De Charette. Clip’s visual embedding projector is a few-shot cornucopia. pages 3254–3264, 2026. 2, 31   
[32] Christoph Feichtenhofer, Haoqi Fan, Bo Xiong, Ross Girshick, and Kaiming He. A large-scale study on unsupervised spatiotemporal representation learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3299–3309, 2021. 1, 31   
[33] Christoph Feichtenhofer, Yanghao Li, Kaiming He, et al. Masked autoencoders as spatiotemporal learners. Advances in Neural Information Processing Systems, 35:35946–35958, 2022. 1, 2, 6, 20, 21, 25, 26, 31   
[34] Quentin Garrido, Randall Balestriero, Laurent Najman, and Yann Lecun. Rankme: Assessing the downstream performance of pretrained self-supervised representations by their rank. In International Conference on Machine Learning, pages 10929–10974. PMLR, 2023. 4   
[35] Xavier Glorot and Yoshua Bengio. Understanding the difficulty of training deep feedforward neural networks. In Proceedings of the thirteenth international conference on

artificial intelligence and statistics, pages 249–256. JMLR Workshop and Conference Proceedings, 2010. 18   
[36] Raghav Goyal, Samira Ebrahimi Kahou, Vincent Michalski, Joanna Materzynska, Susanne Westphal, Heuna Kim, Valentin Haenel, Ingo Fruend, Peter Yianilos, Moritz Mueller-Freitag, et al. The" something something" video database for learning and evaluating visual common sense. In International Conference on Computer Vision, pages 5842–5850, 2017. 2, 7, 20, 21, 22, 23, 26, 31   
[37] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in Neural Information Processing Systems, 33:21271–21284, 2020. 2, 4, 29, 31   
[38] Agrim Gupta, Stephen Tian, Yunzhi Zhang, Jiajun Wu, Roberto Martín-Martín, and Li Fei-Fei. Maskvit: Masked visual pre-training for video prediction. In International Conference on Learning Representations, 2023. 2   
[39] Agrim Gupta, Jiajun Wu, Jia Deng, and Fei-Fei Li. Siamese masked autoencoders. Advances in Neural Information Processing Systems, 36:40676–40693, 2023. 1, 2, 6, 20, 21, 25, 26, 31   
[40] Isma Hadji, Konstantinos G Derpanis, and Allan D Jepson. Representation learning via global temporal alignment and cycle-consistency. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11068–11077, 2021. 2, 31   
[41] Boyu Han, Qianqian Xu, Zhiyong Yang, Shilong Bao, Peisong Wen, Yangbangyan Jiang, and Qingming Huang. Aucseg: Auc-oriented pixel-level long-tail semantic segmentation. Advances in Neural Information Processing Systems, 37:126863–126907, 2024. 2   
[42] Tengda Han, Weidi Xie, and Andrew Zisserman. Selfsupervised co-training for video representation learning. Advances in Neural Information Processing Systems, 33: 5679–5690, 2020. 1, 31   
[43] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In International Conference on Computer Vision, pages 1026–1034, 2015. 18   
[44] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9729–9738, 2020. 29, 31   
[45] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, and Ross Girshick. Masked autoencoders are scalable vision learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000–16009, 2022. 2, 5, 6, 24, 31   
[46] Yingdong Hu, Renhao Wang, Kaifeng Zhang, and Yang Gao. Semantic-aware fine-grained correspondence. In European Conference on Computer Vision, pages 97–115. Springer, 2022. 1, 20, 21, 31   
[47] Weiran Huang, Mingyang Yi, Xuyang Zhao, and Zihao Jiang. Towards the generalization of contrastive self-supervised

learning. In International Conference on Learning Representations, 2023. 1, 29   
[48] Xiaohu Huang, Hao Zhou, Kun Yao, and Kai Han. Froster: Frozen clip is a strong teacher for open-vocabulary action recognition. In International Conference on Learning Representations, 2024. 2   
[49] In Huh, Jae Myung Choe, YOUNGGU KIM, Daesin Kim, et al. Isometric quotient variational auto-encoders for structure-preserving representation learning. Advances in Neural Information Processing Systems, 36:39075–39087, 2023. 5, 15   
[50] Yuqi Huo, Mingyu Ding, Haoyu Lu, Nanyi Fei, Zhiwu Lu, Ji-Rong Wen, and Ping Luo. Compressed video contrastive learning. Advances in Neural Information Processing Systems, 34:14176–14187, 2021. 1, 31   
[51] Allan Jabri, Andrew Owens, and Alexei Efros. Space-time correspondence as a contrastive random walk. Advances in Neural Information Processing Systems, 33:19545–19560, 2020. 2, 3, 31   
[52] Huiwon Jang, Dongyoung Kim, Junsu Kim, Jinwoo Shin, Pieter Abbeel, and Younggyo Seo. Visual representation learning with stochastic frame prediction. In International Conference on Machine Learning, pages 21289–21305, 2024. 2, 6, 20, 21, 25, 26, 31   
[53] Hueihan Jhuang, Juergen Gall, Silvia Zuffi, Cordelia Schmid, and Michael J Black. Towards understanding action recognition. In Proceedings of the IEEE international conference on computer vision, pages 3192–3199, 2013. 6, 21, 22, 31   
[54] Kenji Kawaguchi, Zhun Deng, Xu Ji, and Jiaoyang Huang. How does information bottleneck help deep learning? In International Conference on Machine Learning, pages 16049– 16096. PMLR, 2023. 31   
[55] Will Kay, Joao Carreira, Karen Simonyan, Brian Zhang, Chloe Hillier, Sudheendra Vijayanarasimhan, Fabio Viola, Tim Green, Trevor Back, Paul Natsev, et al. The kinetics human action video dataset. arXiv preprint arXiv:1705.06950, 2017. 2, 6, 20, 31   
[56] Diederik P Kingma, Max Welling, et al. Auto-encoding variational bayes, 2013. 5, 15   
[57] Giorgos Kordopatis-Zilos, Giorgos Tolias, Christos Tzelepis, Ioannis Kompatsiaris, Ioannis Patras, and Symeon Papadopoulos. Self-supervised video similarity learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4756–4766, 2023. 2   
[58] Hildegard Kuehne, Hueihan Jhuang, Estíbaliz Garrote, Tomaso Poggio, and Thomas Serre. Hmdb: a large video database for human motion recognition. In International Conference on Computer Vision, pages 2556–2563. IEEE, 2011. 7, 22, 23, 26, 31   
[59] Hilde Kuehne, Ali Arslan, and Thomas Serre. The language of actions: Recovering the syntax and semantics of goaldirected human activities. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 780–787, 2014. 7, 22, 26   
[60] Yonghyeon Lee, Sangwoong Yoon, MinJun Son, and Frank C Park. Regularized autoencoders for isometric repre-

sentation learning. In International Conference on Learning Representations, 2022. 5, 15   
[61] Feiran Li, Qianqian Xu, Shilong Bao, Zhiyong Yang, Runmin Cong, Xiaochun Cao, and Qingming Huang. Sizeinvariance matters: Rethinking metrics and losses for imbalanced multi-object salient object detection. arXiv preprint arXiv:2405.09782, 2024. 2   
[62] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888– 12900. PMLR, 2022. 6, 21, 24   
[63] Rui Li and Dong Liu. Spatial-then-temporal self-supervised learning for video correspondence. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2279–2288, 2023. 1, 20, 21, 31   
[64] Rui Li, Shenglong Zhou, and Dong Liu. Learning finegrained features for pixel-wise video correspondences. In International Conference on Computer Vision, pages 9632– 9641, 2023. 20, 31   
[65] Xueting Li, Sifei Liu, Shalini De Mello, Xiaolong Wang, Jan Kautz, and Ming-Hsuan Yang. Joint-task self-supervised learning for temporal correspondence. Advances in Neural Information Processing Systems, 32, 2019. 2, 31   
[66] Xinhao Li, Yuhan Zhu, and Limin Wang. Zeroi2v: Zerocost adaptation of pre-trained transformers from image to video. In European Conference on Computer Vision, pages 425–443. Springer, 2024. 1, 2, 7, 20, 25, 31   
[67] Ziyi Lin, Shijie Geng, Renrui Zhang, Peng Gao, Gerard De Melo, Xiaogang Wang, Jifeng Dai, Yu Qiao, and Hongsheng Li. Frozen clip models are efficient video learners. In European Conference on Computer Vision, pages 388–404. Springer, 2022. 2, 20, 31   
[68] Yang Liu, Qianqian Xu, Peisong Wen, Siran Dai, and Qingming Huang. Not all pairs are equal: Hierarchical learning for average-precision-oriented video retrieval. In ACM International Conference on Multimedia, pages 3828–3837, 2024. 2, 23   
[69] Yang Liu, Qianqian Xu, Peisong Wen, Siran Dai, and Qingming Huang. When the future becomes the past: Taming temporal correspondence for self-supervised video representation learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24033–24044, 2025. 1, 2, 6, 20, 21   
[70] Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Objectcentric learning with slot attention. Advances in Neural Information Processing Systems, 33:11525–11538, 2020. 31   
[71] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. 6, 21   
[72] Zijia Lu and Ehsan Elhamifar. Fact: Frame-action crossattention temporal modeling for efficient action segmentation. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18175–18185, 2024. 7, 22, 26   
[73] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez,

Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. 1, 2, 6, 24, 25, 26, 31   
[74] Junting Pan, Ziyi Lin, Xiatian Zhu, Jing Shao, and Hongsheng Li. St-adapter: Parameter-efficient image-to-video transfer learning. Advances in Neural Information Processing Systems, 35:26462–26477, 2022. 1, 2, 7, 20, 21, 25, 31   
[75] Jungin Park, Jiyoung Lee, and Kwanghoon Sohn. Dual-path adaptation from image to video transformers. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2203–2213, 2023. 31   
[76] Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, et al. Pytorch: An imperative style, high-performance deep learning library. Advances in Neural Information Processing Systems, 32, 2019. 21   
[77] Gensheng Pei, Tao Chen, Xiruo Jiang, Huafeng Liu, Zeren Sun, and Yazhou Yao. Videomac: Video masked autoencoders meet convnets. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22733–22743, 2024. 2, 31   
[78] Zhiliang Peng, Li Dong, Hangbo Bao, Qixiang Ye, and Furu Wei. Beit v2: Masked image modeling with vector-quantized visual tokenizers. arXiv preprint arXiv:2208.06366, 2022. 2   
[79] Jordi Pont-Tuset, Federico Perazzi, Sergi Caelles, Pablo Arbeláez, Alex Sorkine-Hornung, and Luc Van Gool. The 2017 davis challenge on video object segmentation. arXiv preprint arXiv:1704.00675, 2017. 2, 6, 21, 22, 31   
[80] Rui Qian, Shuangrui Ding, and Dahua Lin. Rethinking image-to-video adaptation: An object-centric perspective. In European Conference on Computer Vision, pages 329– 348. Springer, 2024. 2, 31   
[81] Zhiwu Qing, Shiwei Zhang, Ziyuan Huang, Yingya Zhang, Changxin Gao, Deli Zhao, and Nong Sang. Disentangling spatial and temporal learning for efficient image-to-video transfer learning. In International Conference on Computer Vision, pages 13934–13944, 2023. 31   
[82] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021. 1, 6, 20, 21, 24, 31   
[83] Hanoona Rasheed, Muhammad Uzair Khattak, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. Fine-tuned clip models are efficient video learners. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6545–6554, 2023. 2   
[84] Nikhila Ravi, Valentin Gabeur, Yuan-Ting Hu, Ronghang Hu, Chaitanya Ryali, Tengyu Ma, Haitham Khedr, Roman Rädle, Chloe Rolland, Laura Gustafson, et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024. 2, 26   
[85] Yangjun Ruan, Saurabh Singh, Warren Richard Morningstar, Alexander A Alemi, Sergey Ioffe, Ian Fischer, and Joshua V

Dillon. Weighted ensemble self-supervised learning. In International Conference on Learning Representations, 2023. 25   
[86] Alexandre Sablayrolles, Matthijs Douze, Cordelia Schmid, and Hervé Jégou. Spreading vectors for similarity search. In International Conference on Learning Representations, 2019. 25   
[87] Christoph Schuhmann, Richard Vencu, Romain Beaumont, Robert Kaczmarczyk, Clayton Mullis, Aarush Katta, Theo Coombes, Jenia Jitsev, and Aran Komatsuzaki. Laion-400m: Open dataset of clip-filtered 400 million image-text pairs. arXiv preprint arXiv:2111.02114, 2021. 6, 21   
[88] Jianbo Shi and Jitendra Malik. Motion segmentation and tracking using normalized cuts. In International Conference on Computer Vision, pages 1154–1160. IEEE, 1998. 3   
[89] Ayush Shrivastava and Andrew Owens. Self-supervised anypoint tracking by contrastive random walks. In European Conference on Computer Vision, pages 267–284. Springer, 2024. 2, 3, 4, 31   
[90] Yang Shu, Xingzhuo Guo, Jialong Wu, Ximei Wang, Jianmin Wang, and Mingsheng Long. Clipood: Generalizing clip to out-of-distributions. In International Conference on Machine Learning, pages 31716–31731. PMLR, 2023. 2   
[91] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 1, 2   
[92] Eero P Simoncelli and Bruno A Olshausen. Natural image statistics and neural representation. Annual review of neuroscience, 24(1):1193–1216, 2001. 2, 31   
[93] K Soomro. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 7, 22, 23, 26, 31   
[94] Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024. 27   
[95] Yansong Tang, Zhenyu Jiang, Zhenda Xie, Yue Cao, Zheng Zhang, Philip HS Torr, and Han Hu. Breaking shortcut: Exploring fully convolutional cycle-consistency for video correspondence learning. arXiv preprint arXiv:2105.05838, 2021. 4   
[96] Chenxin Tao, Xizhou Zhu, Weijie Su, Gao Huang, Bin Li, Jie Zhou, Yu Qiao, Xiaogang Wang, and Jifeng Dai. Siamese image modeling for self-supervised vision representation learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2132–2141, 2023. 2, 31   
[97] Fida Mohammad Thoker, Hazel Doughty, and Cees GM Snoek. Tubelet-contrastive self-supervision for videoefficient generalization. In International Conference on Computer Vision, pages 13812–13823, 2023. 31   
[98] Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In 2015 ieee information theory workshop (itw), pages 1–5. Ieee, 2015. 4   
[99] Naftali Tishby and Noga Zaslavsky. Deep learning and the information bottleneck principle. In 2015 ieee information theory workshop (itw), pages 1–5. IEEE, 2015. 31

[100] Naftali Tishby, Fernando C Pereira, and William Bialek. The information bottleneck method. arXiv preprint physics/0004057, 2000. 4   
[101] Zhan Tong, Yibing Song, Jue Wang, and Limin Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. Advances in Neural Information Processing Systems, 35:10078–10093, 2022. 1, 2, 6, 20, 21, 25, 26, 31   
[102] Hugo Touvron, Andrea Vedaldi, Matthijs Douze, and Hervé Jégou. Fixing the train-test resolution discrepancy. Advances in Neural Information Processing Systems, 32, 2019. 25   
[103] Narek Tumanyan, Assaf Singer, Shai Bagon, and Tali Dekel. Dino-tracker: Taming dino for self-supervised point tracking in a single video. In European Conference on Computer Vision, 2024. 7   
[104] Limin Wang, Bingkun Huang, Zhiyu Zhao, Zhan Tong, Yinan He, Yi Wang, Yali Wang, and Yu Qiao. Videomae v2: Scaling video masked autoencoders with dual masking. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14549–14560, 2023. 1, 2, 20, 31   
[105] Xiaolong Wang, Allan Jabri, and Alexei A Efros. Learning correspondence from the cycle-consistency of time. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2566–2576, 2019. 2, 31   
[106] Xiao Wang, Haoqi Fan, Yuandong Tian, Daisuke Kihara, and Xinlei Chen. On the importance of asymmetry for siamese representation learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16570– 16579, 2022. 4   
[107] Zitai Wang, Qianqian Xu, Zhiyong Yang, Yuan He, Xiaochun Cao, and Qingming Huang. Openauc: Towards auc-oriented open-set recognition. Advances in Neural Information Processing Systems, 35:25033–25045, 2022. 2   
[108] Zitai Wang, Qianqian Xu, Zhiyong Yang, Zhikang Xu, Linchao Zhang, Xiaochun Cao, and Qingming Huang. A unified perspective for loss-oriented imbalanced learning via localization. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 2   
[109] Peisong Wen, Qianqian Xu, Siran Dai, Runmin Cong, and Qingming Huang. Semantic concentration for selfsupervised dense representations learning. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025. 1, 2   
[110] Haiping Wu and Xiaolong Wang. Contrastive learning of image representations with cross-video cycle-consistency. In International Conference on Computer Vision, pages 10149– 10159, 2021. 2, 31   
[111] Qiangqiang Wu, Tianyu Yang, Ziquan Liu, Baoyuan Wu, Ying Shan, and Antoni B Chan. Dropmae: Masked autoencoders with spatial-attention dropout for tracking tasks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14561–14571, 2023. 2, 6, 25, 31   
[112] Qiangqiang Wu, Tianyu Yang, Wei Wu, and Antoni B Chan. Scalable video object segmentation with simplified framework. In International Conference on Computer Vision, pages 13879–13889, 2023. 26

[113] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9653–9663, 2022. 31   
[114] Jiange Yang, Bei Liu, Jianlong Fu, Bocheng Pan, Gangshan Wu, and Limin Wang. Spatiotemporal predictive pre-training for robotic motor control. arXiv preprint arXiv:2403.05304, 2024. 2, 31   
[115] Taojiannan Yang, Yi Zhu, Yusheng Xie, Aston Zhang, Chen Chen, and Mu Li. Aim: Adapting image models for efficient video action recognition. International Conference on Learning Representations, 2023. 1, 2, 7, 20, 21, 25, 31   
[116] Zongxin Yang, Yunchao Wei, and Yi Yang. Associating objects with transformers for video object segmentation. Advances in Neural Information Processing Systems, 34: 2491–2502, 2021. 26   
[117] Youngjae Yu, Sangho Lee, Gunhee Kim, and Yale Song. Self-supervised learning of compressed video representations. In International Conference on Learning Representations, 2020. 1, 31   
[118] Lihi Zelnik-Manor and Michal Irani. Event-based analysis of video. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages II–II. IEEE, 2001. 3   
[119] Yurong Zhang, Liulei Li, Wenguan Wang, Rong Xie, Li Song, and Wenjun Zhang. Boosting video object segmentation via space-time correspondence learning. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2246–2256, 2023. 2, 31   
[120] Jinghao Zhou, Chen Wei, Huiyu Wang, Wei Shen, Cihang Xie, Alan Yuille, and Tao Kong. ibot: Image bert pre-training with online tokenizer. International Conference on Learning Representations, 2022. 2, 6, 24, 25, 26, 31   
[121] Qixian Zhou, Xiaodan Liang, Ke Gong, and Liang Lin. Adaptive temporal encoding network for video instancelevel human parsing. In Proceedings of the 26th ACM international conference on Multimedia, pages 1527–1535, 2018. 2, 6, 21, 22, 31   
[122] Zhijian Zhuo, Yifei Wang, Jinwen Ma, and Yisen Wang. Towards a unified theoretical understanding of non-contrastive learning via rank differential mechanism. International Conference on Learning Representations, 2023. 4

# From Static to Dynamic: Exploring Self-supervised Image-to-Video Representation Transfer Learning

Supplementary Material

# Appendix Contents

A. Symbol Definitions 14

B. Formal Theorems and Proofs 15

B.1. Spectral Properties of Optimal Projections . . 15

B.1.1. Settings for Linear Projection . . . . 15

B.1.2. Settings for MLP Projection . . . . . 15

B.1.3. Proof for Theorem 1 16

B.2. Trade-off Improvement . . 18

C. Supplementary Explanation of Method 20

C.1. Differences with Previous Methods . . . 20

C.2. Algorithm of the Framework . . . . . . 20

D. Detailed Description of Experiments 20

D.1. Training Datasets . . . 20

D.2. Training Settings . . . 21

D.3. Evaluation Settings . . . 21

D.3.1. Evaluation on Dense-level Benchmarks 21

D.3.2. Evaluation on Frame-/Video-level Benchmarks . . . . 22

D.3.3. Distance-based Trade-off Metrics Validation 23

D.4. Image-pretrained Fundamental Models . . . 24

D.5. Competitors . . . 25

E. Detailed Experiments Results 26

E.1. Comparison with Task-Specific SOTAs . . . 26

E.2. Detailed Results of Frame-/Video-Level Tasks 26

E.3. Training Dynamics . . 27

E.4. Shortcut Phenomenon in Training . . . 27

E.5. Additional Ablation Study . 2 7

F. Additional Visualizations 28

F.1. Inter-frame Correspondence 28

F.2. Downstream Task Performance 29

G. Detailed Related Work 29

G.1. Self-supervised Visual Representation . . . . 29

G.2. Image-to-video Transfer Learning . . . . . . 31

G.3. Temporal Cycle Consistency . . . . . . 31

H. Additional Discussions 31

H.1. Limitation and Future Work . . . . . 31

H.2. Broader Impact . . . . 31

# A. Symbol Definitions

We summarize the key notations used in the Method and Theoretical Analysis sections in Tab. 6 and Tab. 7.

Table 6. A summary of key notations and descriptions used in the Method Section. 

<table><tr><td>Notations</td><td>Descriptions</td></tr><tr><td> $\mathcal{D}$ </td><td>Training dataset.</td></tr><tr><td> $H/W/C$ </td><td>The height/width/channel dimension of a frame.</td></tr><tr><td> $T$ </td><td>The number of frames in a video.</td></tr><tr><td> $\boldsymbol{V}$ </td><td>A video containing  $T$  frames.</td></tr><tr><td> $\boldsymbol{v}_{t}$ </td><td>The frame at the moment  $t$  in a video,  $\boldsymbol{v}_{t} \in \mathbb{R}^{H \times W \times C}$ .</td></tr><tr><td> $\boldsymbol{v}_{t}(i)$ </td><td>A frame patch of  $\boldsymbol{v}_{t}, \boldsymbol{v}_{t} \in \mathbb{R}^{H \times W \times C}$ .</td></tr><tr><td> $\delta$ </td><td>The temporal offset between two frames,  $\delta \in (0,1)$ .</td></tr><tr><td> $p$ </td><td>The size of a frame patch.</td></tr><tr><td> $N_{H}$ </td><td>The patch number on the height dimension,  $N_{H} = H/p$ .</td></tr><tr><td> $N_{W}$ </td><td>The patch number on the width dimension,  $N_{W} = W/p$ .</td></tr><tr><td> $N$ </td><td>The patch number of a frame,  $N = N_{H} \times N_{W}$ .</td></tr><tr><td> $d$ </td><td>The embedding dimension of a frame patch.</td></tr><tr><td> $f(\cdot)$ </td><td>The image-pretrained encoder,  $f: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{N \times d}$ .</td></tr><tr><td> $g(\cdot)$ </td><td>The projection layer,  $g: \mathbb{R}^{N \times d} \to \mathbb{R}^{N \times d}$ .</td></tr><tr><td> $\alpha$ </td><td>The amplitude of the positional encoding interpolation.</td></tr><tr><td> $\mathbf{E}_{\text{pos}}$ </td><td>The positional encoding of  $f$ .</td></tr><tr><td> $\widetilde{\mathbf{E}}_{\text{pos}}$ </td><td>The augmented version of  $\mathbf{E}_{\text{pos}}$ .</td></tr><tr><td> $\boldsymbol{z}_{t}$ </td><td>The original representation of  $\boldsymbol{v}_{t}$ , where  $\boldsymbol{z}_{t} = f(\boldsymbol{v}_{t})$ .</td></tr><tr><td> $\boldsymbol{p}_{t}$ </td><td>The projected representation of  $\boldsymbol{z}_{t}$ , where  $\boldsymbol{p}_{t} = g(\boldsymbol{z}_{t})$ .</td></tr><tr><td> $\boldsymbol{A}_{t_{1}}^{t_{2}}$ </td><td>The transition matrix between representations  $\boldsymbol{p}_{t1}$  and  $\boldsymbol{p}_{t2}$ .</td></tr><tr><td> $\lambda$ </td><td>The strength of the constraint term.</td></tr></table>

Table 7. A summary of key notations and descriptions used in the Theoretical Analysis Section. 

<table><tr><td>Notations</td><td>Descriptions</td></tr><tr><td> $d$ </td><td>The embedding dimension of a frame patch.</td></tr><tr><td> $\lambda$ </td><td>The strength of the constraint term.</td></tr><tr><td> $f(\cdot)$ </td><td>The image-pretrained encoder,  $f: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{N \times d}$ .</td></tr><tr><td> $g(\cdot)$ </td><td>The projection layer,  $g: \mathbb{R}^{N \times d} \to \mathbb{R}^{N \times d}$ .</td></tr><tr><td> $z_{i}$ </td><td>The latent representation of an input patch.</td></tr><tr><td> $\tilde{z}_{i}$ </td><td>The mean representation of video  $V_{i}$ ,  $\tilde{z}_{i} = \mathbb{E}_{z \in f(V_{i})} [z]$ .</td></tr><tr><td> $p_{i}$ </td><td>The projected representation of  $z_{i}$ .</td></tr><tr><td> $W$ </td><td>The projection weight of the linear layer.</td></tr><tr><td> $W_{1}/W_{2}$ </td><td>The projection weight of the two-layer MLP.</td></tr><tr><td> $\phi(\cdot)$ </td><td>The tanh activation function.</td></tr><tr><td> $J_{g}(\cdot)$ </td><td>The Jacobian matrix of  $g$ .</td></tr><tr><td> $\Sigma$ </td><td>The intra-video covariance matrix between two patches.</td></tr><tr><td> $\bar{\Sigma}$ </td><td>The inter-video covariance matrix between two videos.</td></tr><tr><td> $U$ </td><td>The orthogonal basis for spectral decomposition.</td></tr><tr><td> $\Lambda_{W}/\Lambda_{1}/\Lambda_{2}$ </td><td>The eigenvalue matrices of  $W/W_{1}/W_{2}$ .</td></tr><tr><td> $\Lambda_{\Sigma}/\Lambda_{\bar{\Sigma}}$ </td><td>The eigenvalue matrices of  $\Sigma/\bar{\Sigma}$ .</td></tr><tr><td> $\mu_{i}/\mu_{1,i}/\mu_{2,i}$ </td><td>The eigenvalues of  $W/W_{1}/W_{2}$ .</td></tr><tr><td> $\sigma_{i}/\tau_{i}$ </td><td>The eigenvalues of  $\Sigma/\bar{\Sigma}$ .</td></tr><tr><td> $D_{intra}$ </td><td>The intra-video distance between two patches.</td></tr><tr><td> $D_{inter}$ </td><td>The inter-video distance between two videos.</td></tr><tr><td> $\gamma$ </td><td>The scale factor between  $D_{intra}$  and  $D_{inter}$ .</td></tr><tr><td> $D$ </td><td>The margin of inter-/intra-video distances.</td></tr><tr><td> $\Delta$ </td><td>The improvement of  $D(z_{1}, z_{2})$ .</td></tr></table>

# B. Formal Theorems and Proofs

In this section, we provide detailed proofs for the theoretical analysis of how the proposed method achieves our target, $i . e .$ , improving the intra-video temporal consistency without largely affecting the inter-video semantic separability. Generally, our analysis leads to two main conclusions: a) Within our proposed method, both linear-based and MLP projection rebalance different dimensions of the representation space in a similar mechanism (Theorem 3). b) This rebalance yields a better trade-off between the two properties under appropriate conditions (Theorem 4).

Formally, given the original representation of a patch $z _ { i } \in \mathbb { R } ^ { d }$ , we aim to learn a projection g that maps zi to $\mathbf { \nabla } _ { \mathbf { } ^ { p } i } =$ $g ( z _ { i } ) \in \mathbb { R } ^ { d }$ . Since directly analyzing the original objectives $\mathcal { L } _ { \mathrm { c y c } }$ and $\mathcal { L } _ { \mathrm { r e g } }$ is challenging, we introduce simplified yet equivalent surrogates to facilitate the analysis.

Objective 1 (Temporal Cycle Consistency). This term encourages alignment between temporally corresponding patches. We quantify it with the metric in Eq. (10). Note that minimizing $M _ { \mathrm { c y c } }$ is equivalent to minimizing the cycleconsistency loss $L _ { \mathrm { c y c } } ,$ , since both decrease as temporal consistency improves and share the same optimality conditions.

$$
M _ {\text { cyc }} = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \big [ \| g (\boldsymbol {z} _ {1}) - g (\boldsymbol {z} _ {2}) \| ^ {2} \big ]. \tag {10}
$$

Objective 2 (Semantic Separability Constraint): The KL divergence constraint $\mathcal { L } _ { \mathrm { r e g } }$ preserves the distance relationships between patches before and after the projection, which is equivalent to constraining the projection to be isometric. This property can be measured by the orthogonality of the Jacobian matrix [16, 49, 56, 60] of g, as formulated in Eq. (11). Therefore, we use it as an approximation of $L _ { \mathrm { r e g } } .$ .

$$
M _ {\text { reg }} = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {i}} \big [ \| \boldsymbol {J} _ {g} (\boldsymbol {z} _ {i}) \boldsymbol {J} _ {g} (\boldsymbol {z} _ {i}) ^ {\top} - \boldsymbol {I} \| _ {F} ^ {2} \big ]. \tag {11}
$$

Combining the two surrogates yields the overall objective:

$$
\min _ {g} M (g) = M _ {\text { cyc }} + \lambda M _ {\text { reg }}. \tag {12}
$$

We now consider two representative cases for g: i) A linear projection: $g ( z ) = W z ; \mathbf { i } ) \mathbf { A }$ two-layer MLP: $g ( z ) =$ $W _ { 2 } \phi ( W _ { 1 } z )$ with activation function $\phi ( \cdot ) = t a n h ( \cdot )$ , and this case represents more complex modules. The following theorem analyzes the spectral properties of the optimal solution under both cases, illustrating how the projection affects the quality of the transferred representation.

# B.1. Spectral Properties of Optimal Projections

# B.1.1. Settings for Linear Projection

For the linear projection $g ( z ) = W z$ , the Jacobian matrix can be expressed as $J _ { g } ( z _ { i } ) = { \frac { \partial g } { \partial z _ { i } } } = { \cal W }$ ∂zi ∂g , thereby the optimization objective can be reformulated as:

$$
\begin{array}{l} \min _ {\boldsymbol {W}} M (\boldsymbol {W}) = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \| \boldsymbol {W} \boldsymbol {z} _ {1} - \boldsymbol {W} \boldsymbol {z} _ {2} \| ^ {2} \right] \tag {13} \\ + \frac {\lambda}{2} \| \boldsymbol {W} \boldsymbol {W} ^ {\top} - \boldsymbol {I} \| _ {F} ^ {2}. \\ \end{array}
$$

To facilitate the analysis, we begin by introducing several definitions and assumptions.

Definition 1 (Intra-video Covariance Matrix). Define the intra-video covariance matrix as the covariance of the patch representations that exhibit corresponding relationships between different frames in a single video: $\Sigma = \mathbb { E } _ { z _ { 1 } , z _ { 2 } } \big \lceil ( z _ { 1 } -$ $z _ { 2 } ) ( z _ { 1 } - z _ { 2 } ) ^ { \top } ]$ , where $( z _ { 1 } , z _ { 2 } )$ denotes a pair of temporally aligned patch representations.

Definition 2 (Inter-video Covariance Matrix). Define the inter-video covariance matrix as the covariance of the videolevel representations across the dataset: $\bar { \Sigma } = \mathbb { E } _ { \bar { z } _ { 1 } , \bar { z } _ { 2 } } \big \lceil ( \bar { z } _ { 1 } -$ $\bar { z } _ { 2 } ) ( \bar { z } _ { 1 } - \bar { z } _ { 2 } ) ^ { \top } ]$ , where $\bar { z } _ { i } = \mathbb { E } _ { z \in f ( V _ { i } ) } \left[ z \right]$ ] denotes the mean representation of video $V _ { i } .$ .

Assumption 1 (Symmetric Operator). Without loss of generality, W , Σ and Σ¯ are constrained to be symmetric $( W ^ { \top } = W , \Sigma ^ { \top } = \Sigma , \bar { \Sigma } ^ { \top } = \bar { \Sigma } )$ . This is justified because any optimal W can be symmetrized without increasing (12).

Assumption 2 (Positive Semi-definite). The transformation operator W , the intra-video covariance matrix Σ, and the inter-video covariance matrix Σ¯ are positive semi-definite: $\pmb { W } \succeq 0 , \pmb { \Sigma } \succeq 0 , \bar { \pmb { \Sigma } } \succeq 0$ . This ensures all eigenvalues are non-negative.

Assumption 3 (Commutative Minimizer). we restrict the analysis to real symmetric commuting pairs $( W , \Sigma )$ and (W , Σ¯ ), i.e., ΣW = W Σ and $\bar { \Sigma } W = W \bar { \Sigma }$ . This allows simultaneous diagonalization with a common orthogonal basis U , yielding $\pmb { W } = \pmb { U } \pmb { \Lambda } _ { W } \pmb { U } ^ { \top } , \pmb { \Sigma } = \pmb { U } \pmb { \Lambda } _ { \Sigma } \pmb { U } ^ { \top }$ , and $\bar { \pmb { \Sigma } } = \pmb { U } \pmb { \Lambda } _ { \bar { \Sigma } } \pmb { U } ^ { \top }$ , where $\pmb { \Lambda } _ { W } , \pmb { \Lambda } _ { \Sigma } , \pmb { \Lambda } _ { \bar { \Sigma } }$ denote corresponding eigenvalue matrices.

# B.1.2. Settings for MLP Projection

For the MLP projection $g ( z ) = W _ { 2 } \phi ( W _ { 1 } z )$ , the optimization objective can be reformulated as:

$$
\min _ {g} M (g) = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \| g (\boldsymbol {z} _ {1}) - g (\boldsymbol {z} _ {2}) \| ^ {2} \right] \tag {14}
$$

$$
+ \frac {\lambda}{2} \mathbb {E} _ {\boldsymbol {z} _ {i}} \left[ \| \boldsymbol {J} _ {g} (\boldsymbol {z} _ {i}) \boldsymbol {J} _ {g} (\boldsymbol {z} _ {i}) ^ {\top} - \boldsymbol {I} \| _ {F} ^ {2} \right].
$$

To facilitate the analysis, we begin by introducing a set of assumptions analogous to those in Case i). Specifically, we replace the matrix W in Assumptions 1 to 3 with $W _ { 1 }$ and $W _ { 2 } .$ , respectively. In addition to these modifications, we introduce the following additional assumptions:

Assumption 4 (Gaussian Distribution). Without loss of generality, we assume that each patch representation $z _ { i } \in \mathbb { R } ^ { d }$ is independently drawn from a multivariate Gaussian distribution: $\boldsymbol { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { \boldsymbol { \Sigma } } )$ . This assumption is justified by the observation that patch-level features extracted from natural videos tend to exhibit approximately Gaussian behavior due to the high-dimensional embedding and the central limit effect. Consequently, $\mathbb { E } _ { z _ { i } } [ z _ { i } z _ { i } ^ { \top } ] = \bar { \Sigma }$ .

Assumption 5 (Linear Approximation). Assuming most values fall within the near-linear region of the tanh activation, we adopt the approximation $\phi ( U x ) \approx U \phi ( { \pmb x } )$ , where $U$ is an orthogonal matrix and $\phi ( \cdot ) = \operatorname { t a n h } ( \cdot )$ is applied element-wise.

# B.1.3. Proof for Theorem 1

Based on the settings above, we establish the following theorem, which characterizes the spectral properties of the optimal solution in both cases and illustrates how the projection affects the quality of the transferred representation.

Theorem 3 (Spectral Properties of Optimal Projections, Formal). Denote the intra-video covariance matrix as ${ \boldsymbol { \Sigma } } =$ $\mathbb { E } _ { z _ { 1 } , z _ { 2 } } \big [ ( z _ { 1 } - z _ { 2 } ) ( z _ { 1 } - z _ { 2 } ) ^ { \top } \big ]$ . Let $\{ \sigma _ { i } \} _ { i = 1 } ^ { d }$ be the eigenvalues of Σ.

positive semi-definite and mutually commuting. Let For case $i ) ,$ assume symmetric matrices W and Σ are $\{ \mu _ { i } \} _ { i = 1 } ^ { d }$ be the eigenvalues of W . Then the eigenvalues of the optimal projection $W ^ { \star }$ obey:

$$
\mu_ {i} ^ {\star} = \left\{ \begin{array}{l l} 0, & \sigma_ {i} > 2 \lambda , \\ \sqrt {1 - \frac {\sigma_ {i}}{2 \lambda}}, & \sigma_ {i} \leq 2 \lambda . \end{array} \right. \tag {15}
$$

For case $i i ) ,$ assume $\phi ( u z _ { i } ) ~ \approx ~ u \phi ( z _ { i } )$ holds for $z _ { i } \sim$ $\mathcal { N } ( \mathbf { 0 } , \pmb { \Sigma } )$ , and that symmetric matrices $W _ { 1 } , W _ { 2 } , \Sigma$ are positive semi-definite and mutually commuting. Let $\{ \mu _ { 1 , i } \} _ { i = 1 } ^ { d }$ and $\{ \mu _ { 2 , i } \} _ { i = 1 } ^ { d }$ 1 be the eigenvalues of $W _ { 1 }$ and $W _ { 2 } ,$ , respectively. Then the eigenvalues of the optimal projections $W _ { 1 } { \mathrm { : } }$ ⋆ , $W _ { 2 } ^ { \star }$ satisfy:

$$
\mu_ {1, i} ^ {\star} \mu_ {2, i} ^ {\star} = \left\{ \begin{array}{l l} 0, & \sigma_ {i} > 2 \lambda , \\ \sqrt {1 - \frac {\sigma_ {i}}{2 \lambda}}, & \sigma_ {i} \leq 2 \lambda . \end{array} \right. \tag {16}
$$

Proof. We first derive the case i) for the optimization objective of linear projection:

$$
\begin{array}{l} M (\boldsymbol {W}) = \underbrace {\frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1} , \boldsymbol {z} _ {2}} \left[ \| \boldsymbol {W} \boldsymbol {z} _ {1} - \boldsymbol {W} \boldsymbol {z} _ {2} \| ^ {2} \right]} _ {\text {Term A}} \tag {17} \\ + \underbrace {\frac {\lambda}{2} \| \boldsymbol {W} \boldsymbol {W} ^ {\top} - \boldsymbol {I} \| _ {F} ^ {2}} _ {\text { Term   B }}. \\ \end{array}
$$

The Term A can be derived as:

Term A

$$
\begin{array}{l} = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \| \boldsymbol {W} \boldsymbol {z} _ {1} - \boldsymbol {W} \boldsymbol {z} _ {2} \| ^ {2} \right] \\ = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \left(\boldsymbol {W} \boldsymbol {z} _ {1} - \boldsymbol {W} \boldsymbol {z} _ {2}\right) ^ {\top} \left(\boldsymbol {W} \boldsymbol {z} _ {1} - \boldsymbol {W} \boldsymbol {z} _ {2}\right) \right] \\ = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \operatorname{Tr} \left(\left(\boldsymbol {W} \boldsymbol {z} _ {1} - \boldsymbol {W} \boldsymbol {z} _ {2}\right) \left(\boldsymbol {W} \boldsymbol {z} _ {1} - \boldsymbol {W} \boldsymbol {z} _ {2}\right) ^ {\top}\right) \right] \\ = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \operatorname{Tr} \left(\boldsymbol {W} \left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right) \left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right) ^ {\top} \boldsymbol {W} ^ {\top}\right) \right] \\ = \frac {1}{2} \operatorname{Tr} \left( \right.\boldsymbol {W} ^ {\top} \boldsymbol {W} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \right.\left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right)\left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right) ^ {\top}\left. \right)\left. \right]\left. \right) \\ = \frac {1}{2} \operatorname{Tr} \left(\boldsymbol {W} ^ {\top} \boldsymbol {W} \boldsymbol {\Sigma}\right) \\ = \frac {1}{2} \operatorname{Tr} \bigl ((\boldsymbol {U} \boldsymbol {\Lambda} _ {W} \boldsymbol {U} ^ {\top}) ^ {\top} (\boldsymbol {U} \boldsymbol {\Lambda} _ {W} \boldsymbol {U} ^ {\top}) (\boldsymbol {U} \boldsymbol {\Lambda} _ {\Sigma} \boldsymbol {U} ^ {\top}) \bigr) \\ = \frac {1}{2} \operatorname{Tr} \left(\boldsymbol {\Lambda} _ {W} ^ {2} \boldsymbol {\Lambda} _ {\Sigma}\right). \tag {18} \\ \end{array}
$$

The derivation in Eq. (18) converts the squared $\ell _ { 2 }$ norm into a matrix trace and further reduces it to a product of eigenvalues via orthogonal decomposition.

The Term B can be derived as:

Term B

$$
\begin{array}{l} = \frac {\lambda}{2} \| \boldsymbol {W} \boldsymbol {W} ^ {\top} - \boldsymbol {I} \| _ {F} ^ {2} \\ = \frac {\lambda}{2} \operatorname{Tr} \left(\left(\boldsymbol {W} \boldsymbol {W} ^ {\top} - \boldsymbol {I}\right) \left(\boldsymbol {W} \boldsymbol {W} ^ {\top} - \boldsymbol {I}\right) ^ {\top}\right) \\ = \frac {\lambda}{2} \operatorname{Tr} \left(\left(\boldsymbol {W} ^ {4} - 2 \boldsymbol {W} ^ {2} + \boldsymbol {I}\right)\right) \tag {19} \\ = \frac {\lambda}{2} \operatorname{Tr} \left(\left(\boldsymbol {U} \boldsymbol {\Lambda} _ {W} \boldsymbol {U} ^ {\top}\right) ^ {4} - 2 \left(\boldsymbol {U} \boldsymbol {\Lambda} _ {W} \boldsymbol {U} ^ {\top}\right) ^ {2} + \boldsymbol {I}\right) \\ = \frac {\lambda}{2} \operatorname{Tr} \left(\boldsymbol {\Lambda} _ {W} ^ {4} - 2 \boldsymbol {\Lambda} _ {W} ^ {2}\right) + \frac {\lambda d}{2}. \\ \end{array}
$$

Similarly, this derivation transforms the Frobenius norm into a matrix trace and reduces it to a function of eigenvalues via orthogonal decomposition.

Then the original objective can be rewritten as:

$$
\begin{array}{l} M (\boldsymbol {W}) = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \| \boldsymbol {W} \boldsymbol {z} _ {1} - \boldsymbol {W} \boldsymbol {z} _ {2} \| ^ {2} \right] \\ + \frac {\lambda}{2} \| \boldsymbol {W} \boldsymbol {W} ^ {\top} - \boldsymbol {I} \| _ {F} ^ {2} \\ = \frac {1}{2} \operatorname{Tr} \left(\boldsymbol {\Lambda} _ {W} ^ {2} \boldsymbol {\Lambda} _ {\Sigma}\right) + \frac {\lambda}{2} \operatorname{Tr} \left(\boldsymbol {\Lambda} _ {W} ^ {4} - 2 \boldsymbol {\Lambda} _ {W} ^ {2}\right) + \frac {\lambda d}{2} \\ = \frac {1}{2} \sum_ {i = 1} ^ {d} (\mu_ {i} ^ {2} \sigma_ {i}) + \frac {\lambda}{2} \sum_ {i = 1} ^ {d} (\mu_ {i} ^ {4} - 2 \mu_ {i} ^ {2}) + \frac {\lambda d}{2}. \tag {20} \\ \end{array}
$$

Differentiating Eq. (20) w.r.t. $\mu _ { i }$ gives:

$$
\begin{array}{l} \frac {\partial M}{\partial \mu_ {i}} = \mu_ {i} \sigma_ {i} + 2 \lambda \mu_ {i} ^ {3} - 2 \lambda \mu_ {i} \tag {21} \\ = \mu_ {i} (2 \lambda \mu_ {i} ^ {2} - (2 \lambda - \sigma_ {i})). \\ \end{array}
$$

Setting the derivative of the objective function to zero yields two possible solutions for each eigenvalue $\mu _ { i } { : }$ :

• $\mu _ { i } = 0$ . This is always a solution. It is optimal whenever the cubic term renders the quartic penalization unnecessary, i.e., when $\sigma _ { i } > 2 \lambda$ .   
• $2 \lambda \mu _ { i } ^ { 2 } - ( 2 \lambda - \sigma _ { i } ) = 0$ . Solving for $\mu _ { i }$ yields the non-zero stationary points, which exist precisely when $\sigma _ { i } \leq 2 \lambda$ .

In summary, the objective Eq. (17) reaches its minimum at $\pmb { W } ^ { \star } = \pmb { U } \mathrm { d i a g } \big ( \mu _ { 1 } ^ { \star } , \ldots , \mu _ { d } ^ { \star } \big ) \pmb { U } ^ { \top }$ , where the eigenvalues of the optimal projection $W ^ { \star }$ obey:

$$
\mu_ {i} ^ {\star} = \left\{ \begin{array}{l l} 0, & \sigma_ {i} \geq 2 \lambda , \\ \sqrt {1 - \frac {\sigma_ {i}}{2 \lambda}}, & \sigma_ {i} <   2 \lambda . \end{array} \right. \tag {22}
$$

Afterward, we derive the case ii) for the optimization objective of MLP projection:

$$
\begin{array}{l} \min _ {g} M (g) = \underbrace {\frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1} , \boldsymbol {z} _ {2}} \left[ \| g (\boldsymbol {z} _ {1}) - g (\boldsymbol {z} _ {2}) \| ^ {2} \right]} _ {\text {Term C}} \tag {23} \\ + \underbrace {\frac {\lambda}{2} \mathbb {E} _ {\boldsymbol {z} _ {i}} \left[ \| \boldsymbol {J} _ {g} (\boldsymbol {z} _ {i}) \boldsymbol {J} _ {g} (\boldsymbol {z} _ {i}) ^ {\top} - \boldsymbol {I} \| _ {F} ^ {2} \right]} _ {\text {Term D}}. \\ \end{array}
$$

The Term C can be derived as:

Term C

$$
\begin{array}{l} = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \left\| g (\boldsymbol {z} _ {1}) - g (\boldsymbol {z} _ {2}) \right\| ^ {2} \right] \\ = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \| \boldsymbol {W} _ {2} \left(\phi (\boldsymbol {W} _ {1} \boldsymbol {z} _ {1}) - \phi (\boldsymbol {W} _ {2} \boldsymbol {z} _ {2})\right) \| ^ {2} \right] \\ = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \| \left(\boldsymbol {U} \boldsymbol {\Lambda} _ {2} \boldsymbol {U} ^ {\top}\right) \right. \\ \end{array}
$$

$$
\left. \left(\phi (\boldsymbol {U} \boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {1}) - \phi (\boldsymbol {U} \boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {2})\right) \| ^ {2} \right]. \tag {24}
$$

Following Assumption 5, the MLP projection becomes $g ( z ) = W _ { 2 } \phi ( W _ { 1 } z ) \approx U \Lambda _ { 2 } \phi ( \Lambda _ { 1 } U ^ { \top } z )$ , we can continue to derive Term C:

Term C

$$
\begin{array}{l} \approx \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \| (\boldsymbol {U} \boldsymbol {\Lambda} _ {2} \boldsymbol {U} ^ {\top}) \boldsymbol {U} \right. \\ \left. \left(\phi \left(\boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {1}\right) - \phi \left(\boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {2}\right)\right) \| ^ {2} \right] \tag {25} \\ = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \| \boldsymbol {\Lambda} _ {2} \left(\phi \left(\boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {1}\right) - \phi \left(\boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {2}\right)\right) \| ^ {2} \right]. \\ \end{array}
$$

Let $\pmb { Q } = \pmb { U } ^ { \top } \pmb { z }$ . Since U is orthogonal and $\boldsymbol { z } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { \Sigma } )$ , it follows that $ { Q } \sim \mathcal { N } ( \mathbf { 0 } ,  { \Sigma } )$ . Based on this property, we have:

Term C

$$
\begin{array}{l} = \frac {1}{2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \| \boldsymbol {\Lambda} _ {2} \left(\phi (\boldsymbol {\Lambda} _ {1} \boldsymbol {Q} _ {1}) - \phi (\boldsymbol {\Lambda} _ {1} \boldsymbol {Q} _ {2})\right) \| ^ {2} \right] \\ = \frac {1}{2} \sum_ {i = 1} ^ {d} \mu_ {2, i} ^ {2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \left(\phi \left(\mu_ {1, i} \boldsymbol {Q} _ {1, i}\right) - \phi \left(\mu_ {1, i} \boldsymbol {Q} _ {2, i}\right)\right) ^ {2} \right] \tag {26} \\ \approx \frac {1}{2} \sum_ {i = 1} ^ {d} \mu_ {2, i} ^ {2} \mu_ {1, i} ^ {2} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \left(\boldsymbol {Q} _ {1, i} - \boldsymbol {Q} _ {2, i}\right) ^ {2} \right] \\ = \frac {1}{2} \sum_ {i = 1} ^ {d} \mu_ {2, i} ^ {2} \mu_ {1, i} ^ {2} \sigma_ {i}. \\ \end{array}
$$

Next, we consider the derivation of the Term D. Note that $J _ { g }$ is the Jacobian matrix of $g ( z _ { i } )$ , it can be formulated as:

$$
\begin{array}{l} \boldsymbol {J} _ {g} (\boldsymbol {z} _ {i}) \approx \boldsymbol {U} \boldsymbol {\Lambda} _ {2} \phi (\boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {i}) \\ = \boldsymbol {U} \boldsymbol {\Lambda} _ {2} \phi^ {\prime} (\boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {i}) \boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \tag {27} \\ = \boldsymbol {U} \boldsymbol {\Lambda} _ {2} \left(1 - \phi^ {2} \left(\boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {i}\right)\right) \boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top}. \\ \end{array}
$$

Therefore, the Term D can be derived as:

Term D

$$
\begin{array}{l} = \frac {\lambda}{2} \mathbb {E} _ {\boldsymbol {z} _ {i}} \left[ \| \boldsymbol {J} _ {g} (\boldsymbol {z} _ {i}) \boldsymbol {J} _ {g} (\boldsymbol {z} _ {i}) ^ {\top} - \boldsymbol {I} \| _ {F} ^ {2} \right] \\ = \frac {\lambda}{2} \mathbb {E} _ {\boldsymbol {z} _ {i}} \left[ \| \left(\boldsymbol {U} \boldsymbol {\Lambda} _ {2} \left(1 - \phi^ {2} \left(\boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {i}\right)\right) \boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top}\right) \right. \\ \end{array}
$$

$$
\left. \left(\boldsymbol {U} \boldsymbol {\Lambda} _ {2} \left(1 - \phi^ {2} \left(\boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {i}\right)\right) \boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top}\right) ^ {\top} - \boldsymbol {I} \| _ {F} ^ {2} \right]
$$

$$
= \frac {\lambda}{2} \mathbb {E} _ {\boldsymbol {z} _ {i}} \left[ \| \boldsymbol {\Lambda} _ {2} \left(1 - \phi^ {2} \left(\boldsymbol {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {i}\right)\right) \boldsymbol {\Lambda} _ {1} \boldsymbol {\Lambda} _ {1} ^ {\top} \right.
$$

$$
\left(1 - \phi^ {2} (\mathbf {\Lambda} _ {1} \boldsymbol {U} ^ {\top} \boldsymbol {z} _ {i})\right) ^ {\top} \mathbf {\Lambda} _ {2} ^ {\top} - \boldsymbol {I} \| _ {F} ^ {2} ]
$$

$$
= \frac {\lambda}{2} \sum_ {i = 1} ^ {d} \left(\mu_ {1, i} ^ {2} \mu_ {2, i} ^ {2} \mathbb {E} _ {\boldsymbol {z} _ {i}} \left[ \left(1 - \phi^ {2} \left(\mu_ {1, i} \boldsymbol {Q} _ {i}\right)\right) ^ {2} \right] - 1\right) ^ {2}. \tag {28}
$$

Similarly, based on Assumption 5, we can approximately move the coefficient of $z _ { i }$ outside the activation function $\phi ( \cdot )$ , leading to the following transformation into the function of eigenvalues:

Term D

$$
\begin{array}{l} \approx \frac {\lambda}{2} \sum_ {i = 1} ^ {d} \left(\mu_ {1, i} ^ {2} \mu_ {2, i} ^ {2} \left(1 - \mu_ {1, i} ^ {2} \mathbb {E} _ {\boldsymbol {z} _ {i}} \left[ \boldsymbol {Q} _ {i} ^ {\top} \boldsymbol {Q} _ {i} \right]\right) ^ {2} - 1\right) ^ {2} \\ = \frac {\lambda}{2} \sum_ {i = 1} ^ {d} \left(\mu_ {1, i} ^ {2} \mu_ {2, i} ^ {2} \left(1 - 2 \mu_ {1, i} ^ {2} \mathbb {E} _ {\boldsymbol {z} _ {i}} \left[ \boldsymbol {Q} _ {i} ^ {\top} \boldsymbol {Q} _ {i} \right] \right. \right. \tag {29} \\ \left. + \mathbb {E} _ {\boldsymbol {z} _ {i}} \left[\left(\boldsymbol {Q} _ {i} ^ {\top} \boldsymbol {Q} _ {i}\right) ^ {\top} \left(\boldsymbol {Q} _ {i} ^ {\top} \boldsymbol {Q} _ {i}\right)\right]\right) - 1\left. \right) ^ {2} \\ = \frac {\lambda}{2} \sum_ {i = 1} ^ {d} \left(\mu_ {1, i} ^ {2} \mu_ {2, i} ^ {2} \left(1 - 2 \sigma_ {i} \mu_ {1, i} ^ {2} + 3 \sigma_ {i} ^ {2} \mu_ {1, i} ^ {4}\right) - 1\right) ^ {2}. \\ \end{array}
$$

Then the original objective can be rewritten as:

$$
\begin{array}{l} M (g) = \frac {1}{2} \sum_ {i = 1} ^ {d} \mu_ {2, i} ^ {2} \mu_ {1, i} ^ {2} \sigma_ {i} \\ + \frac {\lambda}{2} \sum_ {i = 1} ^ {d} \left(\mu_ {1, i} ^ {2} \mu_ {2, i} ^ {2} \left(1 - 2 \sigma_ {i} \mu_ {1, i} ^ {2} + 3 \sigma_ {i} ^ {2} \mu_ {1, i} ^ {4}\right) - 1\right) ^ {2}. \tag {30} \\ \end{array}
$$

For simplicity, we assume $\mu _ { 1 , i } \ \ll \ 1$ , which is a reasonable approximation at initialization when using Kaiming [43] or Xavier [35] schemes. Under this setting, the term $1 - 2 \sigma _ { i } \mu _ { 1 , i } ^ { 2 } + 3 \sigma _ { i } ^ { 2 } \mu _ { 1 , i } ^ { 4 }$ approaches 1, leading to the following simplification:

$$
M (g) = \frac {1}{2} \sum_ {i = 1} ^ {d} \mu_ {2, i} ^ {2} \mu_ {1, i} ^ {2} \sigma_ {i} + \frac {\lambda}{2} \sum_ {i = 1} ^ {d} \left(\mu_ {1, i} ^ {2} \mu_ {2, i} ^ {2} - 1\right) ^ {2}. \tag {31}
$$

Differentiating Eq. (31) w.r.t. $\mu _ { 1 , i }$ gives:

$$
\begin{array}{l} \frac {\partial M}{\partial \mu_ {1 , i}} = \mu_ {1, i} \mu_ {2, i} ^ {2} \sigma_ {i} + 2 \lambda \left(\mu_ {1, i} ^ {2} \mu_ {2, i} ^ {2} - 1\right) \mu_ {1, i} \mu_ {2, i} ^ {2} \tag {32} \\ = \mu_ {1, i} \mu_ {2, i} ^ {2} \big (\sigma_ {i} + 2 \lambda \big (\mu_ {1, i} ^ {2} \mu_ {2, i} ^ {2} - 1 \big) \big). \\ \end{array}
$$

Setting the derivative to zero produces two cases:

• $\mu _ { 1 , i } = 0$ when $\sigma _ { i } > 2 \lambda$ .   
• $\mu _ { 1 , i } = \frac { 1 } { \mu _ { 2 , i } } \sqrt { 1 - \frac { \sigma _ { i } } { 2 \lambda } }$ Solving for $\mu _ { 1 , i }$ yields the nonµ2,i zero stationary points, which exist precisely when $\sigma _ { i } \leq$ 2λ.

Differentiating Eq. (31) w.r.t. $\mu _ { 2 , i }$ gives:

$$
\begin{array}{l} \frac {\partial M}{\partial \mu_ {2 , i}} = \mu_ {2, i} \mu_ {1, i} ^ {2} \sigma_ {i} + 2 \lambda \left(\mu_ {2, i} ^ {2} \mu_ {1, i} ^ {2} - 1\right) \mu_ {2, i} \mu_ {1, i} ^ {2} \tag {33} \\ = \mu_ {2, i} \mu_ {1, i} ^ {2} \big (\sigma_ {i} + 2 \lambda \big (\mu_ {2, i} ^ {2} \mu_ {1, i} ^ {2} - 1 \big) \big). \\ \end{array}
$$

Setting the derivative to zero produces two cases:

• $\mu _ { 2 , i } = 0$ when $\sigma _ { i } > 2 \lambda$ .   
• $\mu _ { 2 , i } = \frac { 1 } { \mu _ { 1 , i } } \sqrt { 1 - \frac { \sigma _ { i } } { 2 \lambda } }$ . Solving for $\mu _ { 2 , \cdot }$ i yields the nonzero stationary points, which exist precisely when $\sigma _ { i } \leq$ 2λ.

In summary, the objective Eq. (23) reaches its minimum at $W _ { 1 } ^ { \star } ~ = ~ U$ diag $\mathbf { \Phi } : ( \mu _ { 1 , 1 } ^ { \star } , \ldots , \mu _ { 1 , d } ^ { \star } ) \mathbf { \Phi } U ^ { \top }$ , $\begin{array} { r l } { W _ { 2 } ^ { \star } } & { { } = } \end{array}$ $U \mathrm { d i a g } ( \mu _ { 2 , 1 } ^ { \star } , \ldots , \mu _ { 2 , d } ^ { \star } ) U ^ { \top }$ , where the eigenvalues of the optimal projection $W _ { 1 } ^ { \star } , W _ { 2 } ^ { \star }$ obey:

$$
\mu_ {1, i} ^ {\star} \mu_ {2, i} ^ {\star} = \left\{ \begin{array}{l l} 0, & \sigma_ {i} > 2 \lambda , \\ \sqrt {1 - \frac {\sigma_ {i}}{2 \lambda}}, & \sigma_ {i} \leq 2 \lambda . \end{array} \right. \tag {34}
$$

This completes the proof.

![](images/9c56b36b9dbd22d82002af7a6ac248ee6e0e39733518afdbc63a60e883f38ffc.jpg)

# B.2. Trade-off Improvement

Since both cases in Sec. B.2 yield similar spectral effects, we conduct the theoretical analysis based on a linear layer. To this end, we first provide the justification of the existence of the trade-off between temporal consistency and semantic separability. Subsequently, we define the distance-based metrics to quantify the two competing objectives and then present a theorem that reveals how the margin evolves after applying the optimal projection.

Lemma 1 (Trade-off between temporal consistency and semantic separability). For the objective $M ( W )$ consisting of a temporal consistency term and a semantic separability term, the gradients of these two terms induce opposing directions in a certain parameter space. This misalignment indicates an inherent trade-off between temporal consistency and semantic separability when optimizing M(W ).

Proof. According to Eq. (20), the objective $M ( W )$ of our method can be derived as:

$$
\begin{array}{l} M (\boldsymbol {W}) = \underbrace {\frac {1}{2} \mathbb {E} _ {z _ {1} , z _ {2}} \left[ \| \boldsymbol {W} z _ {1} - \boldsymbol {W} z _ {2} \| ^ {2} \right]} _ {\text { Temporal   Consistency }} + \underbrace {\frac {\lambda}{2} \| \boldsymbol {W} \boldsymbol {W} ^ {\top} - I \| _ {F} ^ {2}} _ {\text { Semantic   Separability }} \\ = \frac {1}{2} \mathrm{Tr} (\mathbf {\Lambda} _ {W} ^ {2} \mathbf {\Lambda} _ {\Sigma}) + \frac {\lambda}{2} \mathrm{Tr} (\mathbf {\Lambda} _ {W} ^ {4} - 2 \mathbf {\Lambda} _ {W} ^ {2}) + \frac {\lambda d}{2} \\ = \frac {1}{2} \sum_ {i = 1} ^ {d} (\mu_ {i} ^ {2} \sigma_ {i}) + \frac {\lambda}{2} \sum_ {i = 1} ^ {d} (\mu_ {i} ^ {4} - 2 \mu_ {i} ^ {2}) + \frac {\lambda d}{2}. \tag {35} \\ \end{array}
$$

According to the Assumption 2 , W and Σ are positive semi-definite, implying that $\mu _ { i }$ and $\sigma _ { i }$ are non-negative. Therefore, by differentiating $M ( W )$ w.r.t. $\mu _ { i }$ gives:

$$
\frac {\partial M}{\partial \mu_ {i}} = \underbrace {\mu_ {i} \sigma_ {i}} _ {\text { Temporal   Consistency }} + \underbrace {2 \lambda \mu_ {i} ^ {3} - 2 \lambda \mu_ {i}} _ {\text { Semantic   Separability }}. \tag {36}
$$

Based on the formulation in Eq. (36), the gradient of these two derived terms can be inferred as:

• Temporal Consistency: $\mu _ { i } \sigma _ { i } \geq 0$ .   
• Semantic Separability: $2 \lambda \mu _ { i } ^ { 3 } - 2 \lambda \mu _ { i } = 2 \lambda \mu _ { i } ( \mu _ { i } + 1 ) ( \mu _ { i } -$ 1) $< 0$ when $\mu _ { i } < 1$ .

Therefore, the two terms may change in opposite directions during optimization, since updates that increase temporal consistency tend to decrease semantic separability in the same eigen-direction, and vice versa. This behavior reflects an inherent trade-off between temporal consistency and semantic separability in our objective. A similar argument can be made for the trade-off in the MLP case.

![](images/3c51d66141809d51db53a4ea65f9a54f61e1ba6d739efac3aec31023372f71a7.jpg)

Definition 3 (Intra-video Distance). Define the intra-video distance as $D _ { i n t r a } ( z _ { 1 } , z _ { 2 } ) = \mathbb { E } _ { z _ { 1 } , z _ { 2 } } \left[ \| z _ { 1 } - z _ { 2 } \| ^ { 2 } \right]$ , which measures the average distance between temporally corresponding patches within a video.

Definition 4 (Inter-video Distance). Define the inter-video distance as $D _ { i n t e r } ( z _ { 1 } , z _ { 2 } ) = \mathbb { E } _ { \bar { z } _ { 1 } , \bar { z } _ { 2 } } \left[ \| \bar { z } _ { 1 } - \bar { z } _ { 2 } \| ^ { 2 } \right]$ , calculating the average distance between video-level representations, where $\bar { z } _ { i } = \mathbb { E } _ { z \in f ( V _ { i } ) } \left[ z \right]$ is the mean representation of the video $V _ { i }$ . This reflects the average distance between different video-level representations.

Definition 5 (Distance Margin). Define the margin of these two metrics as $D ( z _ { 1 } , z _ { 2 } ) ~ = ~ D _ { i n t e r } ( z _ { 1 } , z _ { 2 } ) ~ -$ $\gamma D _ { i n t r a } ( z _ { 1 } , z _ { 2 } )$ , reflecting the degree of separation between the two properties, where a larger value indicates a better trade-off between the two objectives.

Assumption 6 (Mean Eigenvalue Approximation). The eigenvalues of the inter-video covariance matrix approximate the average of those of the intra-video covariance matrix, i.e., $\begin{array} { r } { \forall j , \ \tau _ { j } = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \sigma _ { i } } \end{array}$ .

Theorem 4 (Trade-off Improvement, Formal). Let ${ \boldsymbol { \Sigma } } =$ $\mathbb { E } _ { z _ { 1 } , z _ { 2 } } \big [ ( z _ { 1 } - z _ { 2 } ) ( z _ { 1 } - z _ { 2 } ) ^ { \top } \big ] , \bar { \Sigma } = \mathbb { E } _ { \bar { z } _ { 1 } , \bar { z } _ { 2 } } \big [ ( \bar { z } _ { 1 } - \bar { z } _ { 2 } ) ( \bar { z } _ { 1 } -$ trices, with eigenvalues $\bar { z } _ { 2 } ) ^ { \top } ]$ be the intra-video and inter-video covariance ma- $\{ \sigma _ { i } \} _ { i = 1 } ^ { d }$ and $\{ \tau _ { i } \} _ { i = 1 } ^ { d }$ , respectively. Assume symmetric matrices W , Σ, and Σ¯ are positive d mutu. Let muting, and that be the eigenvalu $\forall j , \tau _ { j } =$ $\textstyle { \frac { 1 } { d } } \sum _ { i = 1 } ^ { d } \sigma _ { i } \ = \ \tau$ $\{ \mu _ { i } \} _ { i = 1 } ^ { d }$ For the linear projection $g ( z ) = W z$ , where the optimal eigenvalues are given by $\mu _ { i } ^ { \star } = \sqrt { 1 - \frac { \sigma _ { i } } { 2 \lambda } } f o r \sigma _ { i } \le 2 \lambda$ , the improvement in the margin metric is: given by:

$$
\Delta = D (g (\boldsymbol {z} _ {1}), g (\boldsymbol {z} _ {2})) - D (\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2})
$$

$$
= \sum_ {\sigma_ {i} \leq 2 \lambda} (\tau - \sigma_ {i}) \left(1 - \frac {\sigma_ {i}}{2 \lambda}\right) > 0. \tag {37}
$$

Proof. The margin metric of intra-video distance between the projected representations and the original representations can be derived as:

$$
\Delta_ {i n t r a} = D _ {i n t r a} \left(\boldsymbol {W} \boldsymbol {z} _ {1}, \boldsymbol {W} \boldsymbol {z} _ {2}\right) - \gamma D _ {i n t r a} \left(\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}\right)
$$

$$
= \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \left\| \boldsymbol {W} \boldsymbol {z} _ {1} - \boldsymbol {W} \boldsymbol {z} _ {2} \right\| ^ {2} \right]
$$

$$
- \gamma \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \left\| \boldsymbol {z} _ {1} - \boldsymbol {z} _ {2} \right\| ^ {2} \right]
$$

$$
= \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \left(\boldsymbol {W} \boldsymbol {z} _ {1} - \boldsymbol {W} \boldsymbol {z} _ {2}\right) ^ {\top} \left(\boldsymbol {W} \boldsymbol {z} _ {1} - \boldsymbol {W} \boldsymbol {z} _ {2}\right) \right]
$$

$$
- \gamma \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right) ^ {\top} \left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right) \right]
$$

$$
= \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \operatorname{Tr} \left(\boldsymbol {W} \left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right) \left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right) ^ {\top} \boldsymbol {W} ^ {\top}\right) \right]
$$

$$
- \gamma \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \operatorname{Tr} \big ((\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}) (\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}) ^ {\top} \big) \right]
$$

$$
= \operatorname{Tr} \left(\boldsymbol {W} ^ {\top} \boldsymbol {W} \mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right) \left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right) ^ {\top} \right]\right)
$$

$$
- \gamma \operatorname{Tr} \left(\mathbb {E} _ {\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}} \left[ \left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right) \left(\boldsymbol {z} _ {1} - \boldsymbol {z} _ {2}\right) ^ {\top} \right]\right)
$$

$$
= \operatorname{Tr} \left(\boldsymbol {\Lambda} _ {W} ^ {\top} \boldsymbol {\Lambda} _ {W} \boldsymbol {\Lambda} _ {\Sigma}\right) - \gamma \operatorname{Tr} \left(\boldsymbol {\Lambda} _ {\Sigma}\right)
$$

$$
= \sum_ {i = 1} ^ {d} (\mu_ {i} ^ {2} - \gamma) \sigma_ {i}. \tag {38}
$$

The margin metric of inter-video distance between the projected representations and the original representations can be derived as:

$$
\Delta_ {i n t e r} = D _ {i n t e r} (\pmb {W} \pmb {z} _ {1}, \pmb {W} \pmb {z} _ {2}) - D _ {i n t e r} (\pmb {z} _ {1}, \pmb {z} _ {2})
$$

$$
= \mathbb {E} _ {\bar {\boldsymbol {z}} _ {1}, \bar {\boldsymbol {z}} _ {2}} \left[ \| \boldsymbol {W} \bar {\boldsymbol {z}} _ {1} - \boldsymbol {W} \bar {\boldsymbol {z}} _ {2} \| ^ {2} \right]
$$

$$
- \gamma \mathbb {E} _ {\bar {z} _ {1}, \bar {z} _ {2}} \left[ \| \bar {z} _ {1} - \bar {z} _ {2} \| ^ {2} \right]
$$

$$
= \mathbb {E} _ {\bar {\boldsymbol {z}} _ {1}, \bar {\boldsymbol {z}} _ {2}} \left[ \left(\boldsymbol {W} \bar {\boldsymbol {z}} _ {1} - \boldsymbol {W} \bar {\boldsymbol {z}} _ {2}\right) ^ {\top} \left(\boldsymbol {W} \bar {\boldsymbol {z}} _ {1} - \boldsymbol {W} \bar {\boldsymbol {z}} _ {2}\right) \right]
$$

$$
- \gamma \mathbb {E} _ {\bar {\boldsymbol {z}} _ {1}, \bar {\boldsymbol {z}} _ {2}} \left[ (\bar {\boldsymbol {z}} _ {1} - \bar {\boldsymbol {z}} _ {2}) ^ {\top} (\bar {\boldsymbol {z}} _ {1} - \bar {\boldsymbol {z}} _ {2}) \right]
$$

$$
= \mathbb {E} _ {\bar {\boldsymbol {z}} _ {1}, \bar {\boldsymbol {z}} _ {2}} \left[ \operatorname{Tr} \left(\boldsymbol {W} (\bar {\boldsymbol {z}} _ {1} - \bar {\boldsymbol {z}} _ {2}) \left(\bar {\boldsymbol {z}} _ {1} - \bar {\boldsymbol {z}} _ {2}\right) ^ {\top} \boldsymbol {W} ^ {\top}\right) \right]
$$

$$
- \gamma \mathbb {E} _ {\bar {\boldsymbol {z}} _ {1}, \bar {\boldsymbol {z}} _ {2}} \left[ \mathrm{Tr} \big ((\bar {\boldsymbol {z}} _ {1} - \bar {\boldsymbol {z}} _ {2}) (\bar {\boldsymbol {z}} _ {1} - \bar {\boldsymbol {z}} _ {2}) ^ {\top} \big) \right]
$$

$$
= \operatorname{Tr} \left(\boldsymbol {W} ^ {\top} \boldsymbol {W} \mathbb {E} _ {\bar {\boldsymbol {z}} _ {1}, \bar {\boldsymbol {z}} _ {2}} \left[ (\bar {\boldsymbol {z}} _ {1} - \bar {\boldsymbol {z}} _ {2}) (\bar {\boldsymbol {z}} _ {1} - \bar {\boldsymbol {z}} _ {2}) ^ {\top} \right]\right)
$$

$$
\left. - \gamma \operatorname{Tr} \left(\mathbb {E} _ {\bar {\boldsymbol {z}} _ {1}, \bar {\boldsymbol {z}} _ {2}} \left[ (\bar {\boldsymbol {z}} _ {1} - \bar {\boldsymbol {z}} _ {2}) (\bar {\boldsymbol {z}} _ {1} - \bar {\boldsymbol {z}} _ {2}) ^ {\top} \right]\right) \right.
$$

$$
= \operatorname{Tr} \left(\boldsymbol {\Lambda} _ {W} ^ {\top} \boldsymbol {\Lambda} _ {W} \boldsymbol {\Lambda} _ {\bar {\Sigma}}\right) - \gamma \operatorname{Tr} \left(\boldsymbol {\Lambda} _ {\bar {\Sigma}}\right)
$$

$$
= \sum_ {i = 1} ^ {d} (\mu_ {i} ^ {2} - \gamma) \tau_ {i}. \tag {39}
$$

Then the improvement of the margin metrics can be formulated as:

$$
\Delta = (D _ {i n t e r} (\boldsymbol {W} \boldsymbol {z} _ {1}, \boldsymbol {W} \boldsymbol {z} _ {2}) - D _ {i n t r a} (\boldsymbol {W} \boldsymbol {z} _ {1}, \boldsymbol {W} \boldsymbol {z} _ {2}))
$$

$$
- \left(D _ {i n t e r} (\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}) - D _ {i n t r a} (\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2})\right)
$$

$$
= (D _ {i n t e r} (\boldsymbol {W} \boldsymbol {z} _ {1}, \boldsymbol {W} \boldsymbol {z} _ {2}) - D _ {i n t e r} (\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2}))
$$

$$
- \left(D _ {i n t r a} (\boldsymbol {W} \boldsymbol {z} _ {1}, \boldsymbol {W} \boldsymbol {z} _ {2}) - D _ {i n t r a} (\boldsymbol {z} _ {1}, \boldsymbol {z} _ {2})\right)
$$

$$
= \sum_ {i = 1} ^ {d} (\mu_ {i} ^ {2} - \gamma) \tau_ {i} - \sum_ {i = 1} ^ {d} (\mu_ {i} ^ {2} - \gamma) \sigma_ {i}
$$

$$
= \sum_ {i = 1} ^ {d} (\mu_ {i} ^ {2} - \gamma) (\tau_ {i} - \sigma_ {i}). \tag {40}
$$

Under Assumptions 1 to 3, the optimal linear projection $\pmb { W } ^ { \star } = \pmb { U } \mathrm { d i a g } \big ( \bar { \mu _ { 1 } ^ { \star } } , \dots , \mu _ { d } ^ { \star } \big ) \pmb { U } ^ { \top }$ has eigenvalues given by:

$$
\mu_ {i} ^ {\star} = \left\{ \begin{array}{l l} 0, & \sigma_ {i} \geq 2 \lambda , \\ \sqrt {1 - \frac {\sigma_ {i}}{2 \lambda}}, & \sigma_ {i} <   2 \lambda . \end{array} \right. \tag {41}
$$

Under Assumption $^ { 6 , }$ due to $\begin{array} { r } { \forall j , \tau _ { j } = \frac { 1 } { d } \sum _ { i = 1 } ^ { d } \sigma _ { i } = \tau , } \end{array}$ we have ${ \textstyle \sum _ { i = 1 } ^ { d } ( \tau - \sigma _ { i } ) = 0 }$ .

By substituting $\mu _ { i } ^ { \star } = \sqrt { 1 - \frac { \sigma _ { i } } { 2 \lambda } }$ and $\begin{array} { r } { \sum _ { i = 1 } ^ { d } ( \tau - \sigma _ { i } ) = 0 } \end{array}$ under the condition $\sigma _ { i } < 2 \lambda$ , the change in the margin metric can be expressed as:

$$
\Delta = \sum_ {i = 1} ^ {d} (\mu_ {i} ^ {2} - \gamma) (\tau_ {i} - \sigma_ {i}) = \sum_ {\sigma_ {i} <   2 \lambda} (\tau - \sigma_ {i}) \left(1 - \frac {\sigma_ {i}}{2 \lambda}\right). \tag {42}
$$

When $\lambda < \frac { \tau } { 2 } ( i . e . , 2 \lambda < \tau )$ , all $\sigma _ { i } \ \leq \ 2 \lambda$ necessarily obey $\sigma _ { i } < \tau .$ . In this case, each term in Eq. (42) satisfies $\left( \tau - \sigma _ { i } \right) \left( 1 - \frac { \sigma _ { i } } { 2 \lambda } \right) > 0$ because:

$\cdot \ \tau - \sigma _ { i } > 0$ follows from $\sigma _ { i } < \tau ,$   
$\mathbf { \partial } \cdot \mathrm { ~ 1 ~ - ~ } \frac { \sigma _ { i } } { 2 \lambda } \ge 0$ • 1 since $\sigma _ { i } \leq 2 \lambda .$

Therefore, $\Delta > 0$ holds whenever $\lambda < \frac { 1 } { 2 d } \sum _ { i = 1 } ^ { d } { \sigma _ { i } } .$ .

In summary, this section provides a theoretical analysis of the trade-off between temporal consistency and semantic separability, leading to the following two key insights:

1. Theorem 3 shows that linear projection exhibits similar behavior to shallow MLPs in adjusting representations, yielding comparable effects in similar feature scaling behavior as the linear layer.   
2. Theorem 4 demonstrates that under optimal conditions, a linear projection is sufficient to improve the trade-off between temporal consistency and semantic separability.

# C. Supplementary Explanation of Method

# C.1. Differences with Previous Methods

In Figure 5, we provide a comparative overview of several categories of video representation learning works alongside our method.

1) Video-pretrained methods extend the masked image modeling paradigm to the video domain by masking 3D volumes and reconstructing raw pixels for spatiotemporal learning [33, 101, 104]. Subsequent variants incorporate conditional frames to enhance temporal modeling [30, 39, 52, 69]. These approaches typically require large-scale video pretraining from scratch, incurring substantial computational cost due to video redundancy and pixel-level reconstruction overhead.   
2) Supervised adaptation methods adapt Vision Transformers pretrained with CLIP [82] by inserting lightweight adapters in serial or parallel configurations [17, 66, 67, 74, 115]. These adapters are usually trained on supervised action recognition datasets [36, 55], making them highly taskdependent and less generalizable without additional taskspecific fine-tuning.   
3) Video fine-tuning methods follow a two-stage training scheme: models are first pretrained on task-specific datasets to learn static features for instance-level discrimination, then fine-tuned on video datasets with additional temporal branches introduced to handle motion reasoning [27, 46, 63, 64]. Although it can perform well on specific video tasks, its increased model complexity and training cost make it difficult to perform fast cross-domain transfer.   
4) Our image-to-video transfer method takes a different approach by leveraging pretrained image representations and adapting them to video tasks via structure-preserving projection. The main advantages are as follows:

Algorithm 1: Consistency-Separability Trade-off Transfer Learning Algorithm   
Input: Unlabeled dataset D, number of iterations L, interpolation ratio $\alpha$ , constraint weight $\lambda$ .

Output: Parameters $\theta_{L+1}$ of projection layer g.

1 Initialize parameters $\theta_{1}$ for g.

2 for l = 1 to L do

3    Sample a batch of videos $\{V_{i}\}_{i=1}^{B}$ .

4    for i = 1 to B do

5    ▷ Temporal Correspondence Establishment

6    Select frames $v_{t_{1}}^{f}, v_{t_{2}}, v_{t_{1}}^{b}$ from $V_{m}$ and prepare position encoding $E_{pos}$ and $\widetilde{E}_{pos}$ .

7    Extract representations $z_{t_{1}}^{f}, z_{t_{2}}, \widetilde{z}_{t_{1}}^{b}$ with f and projections $p_{t_{1}}^{f}, p_{t_{2}}, \widetilde{p}_{t_{1}}^{b}$ via g.

8    Calculate correlation matrices $A_{t_{1}}^{t_{2}}$ and $\widetilde{A}_{t_{2}}^{t_{1}}$ .

9    ▷ Temporal Consistency and Semantic Separability Trade-off

10    Enhance temporal consistency of $p_{t_{1}}^{f}, p_{t_{2}}, \widetilde{p}_{t_{1}}^{b}$ via $L_{cyc}$ .

11    Align the semantic separability of $\{(p_{t_{1}}^{f}, z_{t_{1}}^{f}), (p_{t_{2}}, z_{t_{2}}), (\widetilde{p}_{t_{1}}^{b}, \widetilde{z}_{t_{1}}^{b})\}$ by $L_{reg}$ .

12    Update the projection layer g with $L_{total} = L_{cyc} + \lambda L_{reg}$ .

# 13 return

• Efficient transfer: We sample two frames per video and insert a lightweight linear-based projection head after a frozen image encoder, enabling fast transfer with reduced temporal and spatial cost.   
• Joint optimization: We simultaneously optimize temporal consistency and semantic separability via a temporal cycle-consistency objective and a semantic separability regularization term.   
• Label-free training: Our method is fully self-supervised, requiring no manual annotations, which enhances scalability and promotes better generalization across diverse video tasks of different granularity.

# C.2. Algorithm of the Framework

The complete optimization procedure of our framework is summarized in Algorithm 1. The batch-level for-loop can be implemented via matrix operations to reduce computational burden.

# D. Detailed Description of Experiments

# D.1. Training Datasets

Kinetics-400 [55] is a widely used large-scale video benchmark comprising 400 human action categories collected from YouTube. It provides 239,789 trimmed video clips, each lasting around 10 seconds, making it suitable for various video understanding tasks. In our experiments, we sample video frames at 2 FPS for pretraining to reduce redundancy while retaining sufficient temporal cues. In this work, unless otherwise noted, all the models equipped with our method are trained for 5 epochs using the Kinetics-400 training set.

![](images/416782681a04f15ab730a26ddbce2ec332c4d56b2fa9fabd28e28f89cdfa73cc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Video"] --> B["Layer Norm"]
    B --> C["Multi-Head Self-Attention"]
    C --> D["Layer Norm"]
    D --> E["Feed Forward"]
    E --> F["Decoder"]
    F --> G["Output Video"]
    G --> H["L^v MSE"]
    H --> A
    style A fill:#cce5ff,stroke:#333
    style G fill:#cce5ff,stroke:#333
```
</details>

(a) Video-pretrained methods [33, 101].

![](images/99b2c602bd0e611f6c28dfe8701192ec1a6b353ccbcafcb2fa205a1a64295fe4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Video"] --> B["Adapter"]
    B --> C["Layer Norm"]
    C --> D["Multi-Head Self-Attention"]
    D --> E["Layer Norm"]
    E --> F["Feed Forward"]
    F --> G["Sparse Features"]
    G --> H["L^v_CE"]
    I["Class"] --> B
    J["L^v_CE"] --> G
```
</details>

(b) Supervised adaptation methods [74, 115].

![](images/b6e1e856e0daccca5823df054e71e885e804db937fd4b8521916795f01195872.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Video"] --> B["Layer Norm"]
    B --> C["Multi-Head Self-Attention"]
    C --> D["Layer Norm"]
    D --> E["Feed Forward"]
    E --> F["×L"]
    F --> G["Head"]
    G --> H["Dense Features"]
    H --> I["L^v_align"]
    I --> G
```
</details>

(c) Video fine-tuning methods [46, 63].

![](images/2273344dc4a71ab7401e818fb3b023b891c3ee555221e29cdd1927ec9671ee2a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Frames"] --> B["Layer Norm"]
    B --> C["Multi-Head Self-Attention"]
    C --> D["Layer Norm"]
    D --> E["Feed Forward"]
    E --> F["Projection"]
    F --> G["Dense Features"]
    G --> H["L^f_cyc"]
    H --> I["Output"]
    style A fill:#f9f,stroke:#333
    style G fill:#ccf,stroke:#333
```
</details>

(d) Image-to-video transfer method (Ours).   
Figure 5. Comparison of several categories of video representation learning methods with ours.

SSV2 (Something-Something V2) [36] is a large-scale video benchmark emphasizing human-object interactions and temporal reasoning. It comprises 220,847 short, crowdsourced clips across 174 action classes, with each clip lasting a few seconds, making it well-suited for evaluating temporal understanding beyond appearance cues. In our experiments, we only use SSV2 in the ablation study on training datasets, training for 5 epochs on the SSV2 training split.

ImageNet-1k [25] is a well-known image dataset containing over 1.28 million training images across 1,000 real-world object categories. It has played a central role in the development of deep visual representation learning and serves as the pretraining corpus for most high-performance image encoders. In this work, most of the image models are already pretrained on this dataset, providing a strong foundation of semantic separability.

WIT-400M (WebImageText) [82] is a large-scale webcrawled dataset consisting of 400 million image-text pairs, designed to support vision-language pretraining. The dataset was constructed using 500,000 diverse natural language queries to guide image-text pair retrieval, with up to 20,000 pairs per query to encourage approximate class balancing. Its overall scale and linguistic richness make it suitable for training multimodal models such as CLIP [82].

LAION-400M [87] is a large-scale dataset of 400 million image-text pairs designed to support vision-language pretraining. The image-text pairs are extracted from Common Crawl web pages and filtered using CLIP-based similarity to retain pairs with stronger semantic alignment between images and captions. It is used as a pretraining dataset for vision-language models, including BLIP [62].

# D.2. Training Settings

During training, we freeze the pretrained image encoder f and update only the projection layer g. The training is performed on the Kinetics-400 dataset for 5 epochs with a total batch size of 512, using the first epoch for learning rate warm-up. We employ the AdamW optimizer [71] with a cosine learning rate decay schedule. The base learning rate is set to $b l r = 1 \times 1 0 ^ { - 4 }$ and scaled according to the batch size as $l r = b l r / 2 5 6$ . For each video clip, two frames are randomly sampled with a temporal interval of δ = 0.15 relative to the total video length. The softmax temperature is set to $\tau = 0 . 0 3$ . The output dimension of the projection head g is set to d = 768 for ViT-Base backbones. Detailed hyperparameter settings for training and method components are summarized in Tab. 8a and Tab. 8b. All experiments are implemented in PyTorch [76] and conducted on a Linux server equipped with an AMD EPYC 9654 96-Core CPU and 4 NVIDIA RTX4090 GPUs.

# D.3. Evaluation Settings

# D.3.1. Evaluation on Dense-level Benchmarks

We first evaluate the representations on three dense video downstream tasks: video object segmentation on DAVIS-2017 [79], human part segmentation on VIP [121], and human pose propagation on JHMDB [53]. Following previous works [30, 39, 52, 69], all tasks are evaluated under a semisupervised protocol in which the ground-truth mask of the first frame is given, and the model propagates predictions to subsequent frames without any task-specific fine-tuning. The hyperparameters used for each evaluation task are listed in Tab. 8c. To ensure fair comparisons, we keep the evaluation hyperparameter settings fixed across all methods and tasks without additional tuning.

Table 8. Summary of hyperparameter settings used during training and evaluation.   
(a) Training hyperparameters. 

<table><tr><td>Hyperparameter</td><td>Notation</td><td>Value</td></tr><tr><td>Image size</td><td> $H \times W$ </td><td> $224 \times 224$ </td></tr><tr><td>Patch size</td><td> $p$ </td><td>16</td></tr><tr><td>Optimizer</td><td>/</td><td>AdamW</td></tr><tr><td>Scheduler</td><td>/</td><td>Cosine</td></tr><tr><td>Weight decay</td><td>/</td><td>0.05</td></tr><tr><td>Momentum</td><td> $\beta_1, \beta_2$ </td><td>0.9, 0.95</td></tr><tr><td>Base learning rate</td><td> $blr$ </td><td> $1 \times 10^{-4}$ </td></tr><tr><td>Epochs</td><td>/</td><td>5</td></tr><tr><td>Warm-up Epoch</td><td>/</td><td>1</td></tr><tr><td>Batch size</td><td> $bs$ </td><td>512</td></tr></table>

(b) Method hyperparameters. 

<table><tr><td>Hyperparameter</td><td>Notation</td><td>Value</td></tr><tr><td>Temperature of Softmax</td><td>τ</td><td>0.03</td></tr><tr><td>Frame sampling interval</td><td>δ</td><td>0.15</td></tr><tr><td>Feature dim of g</td><td>d</td><td>768</td></tr></table>

(c) Evaluation hyperparameters. 

<table><tr><td>Hyperparameter</td><td>DAVIS-2017</td><td>VIP</td><td>JHMDB</td></tr><tr><td>Image size</td><td> $480 \times 880$ </td><td> $480 \times 880$ </td><td> $320 \times 320$ </td></tr><tr><td>Top-K</td><td>7</td><td>10</td><td>7</td></tr><tr><td>Queue Length</td><td>20</td><td>20</td><td>20</td></tr><tr><td>Neighborhood Size</td><td>20</td><td>20</td><td>20</td></tr></table>

DAVIS-2017 [79] is a widely used benchmark for video object segmentation. We report three standard metrics to assess overall segmentation quality:

1) ${ \mathcal { I } } _ { \mathrm { m } }$ (region similarity) computes the average IoU between the predicted mask $P _ { i }$ and the ground-truth mask $G _ { i }$ across all videos $V _ { i } { \mathrm { : } }$

$$
\mathcal {J} _ {\mathrm{m}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {\left| P _ {i} \cap G _ {i} \right|}{\left| P _ {i} \cup G _ {i} \right|}. \tag {43}
$$

2) $\mathcal { F } _ { \mathrm { m } }$ (contour accuracy) evaluates the alignment between the predicted and ground-truth boundaries by calculating the harmonic mean of precision $P r e _ { i }$ and recall $R e c _ { i } { \mathrm { : } }$ :

$$
\mathcal {F} _ {\mathrm{m}} = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {2 \cdot P r e _ {i} \cdot R e c _ {i}}{P r e _ {i} + R e c _ {i}}. \tag {44}
$$

3) $\mathcal { T } \& \mathcal { F } _ { \mathrm { m } }$ provides an overall performance measure by averaging ${ \mathcal { I } } _ { \mathrm { m } }$ and $\mathcal { F } _ { \mathrm { m } }$ :

$$
\mathcal {J} \& \mathcal {F} _ {\mathrm{m}} = \frac {\mathcal {J} _ {\mathrm{m}} + \mathcal {F} _ {\mathrm{m}}}{2}. \tag {45}
$$

VIP [121] focuses on fine-grained human part segmentation and is used to evaluate semantic part propagation. The main evaluation metric is the mIoU computed by averaging the IoU across all classes $C _ { j }$ and all videos $V _ { i }$ :

$$
\mathrm{mIoU} = \frac {1}{| C |} \sum_ {j = 1} ^ {| C |} \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {\left| P _ {i , j} \cap G _ {i , j} \right|}{\left| P _ {i , j} \cup G _ {i , j} \right|}. \tag {46}
$$

JHMDB [53] is commonly used for human pose estimation. We adopt it for the pose propagation task and evaluate performance using the PCK@k metric, which measures the proportion of keypoints predicted within a normalized distance threshold:

$$
\mathrm{PCK} @ k = \frac {1}{n} \sum_ {i = 1} ^ {n} \frac {1}{| S _ {i} |} \sum_ {j = 1} ^ {| S _ {i} |} \mathbb {1} \left[ D (\hat {p} _ {i, j}, p _ {i, j}) <   k \cdot d _ {i} \right], (4 7)
$$

where $S _ { i }$ is the keypoint set in video $V _ { i } , d _ { i }$ denotes the scale of the human body, $D ( \hat { p } _ { i , j } , p _ { i , j } )$ is the Euclidean distance between the predicted and ground-truth positions, and k is the threshold for the maximum allowable distance error. We report PCK@0.1 and PCK@0.2 in our experiments.

# D.3.2. Evaluation on Frame-/Video-level Benchmarks

We further evaluate the transferred models on several framelevel and video-level downstream tasks: temporal action localization on Breakfast [59], video retrieval on UCF101 and HMDB51 [58, 93], and action classification on Something-Something-v2 (SSV2) [36].

Breakfast [59] contains 1,712 untrimmed videos with frame-level annotations of fine-grained actions. We perform temporal action localization on this dataset by extracting frame-wise representations with our transferred image-tovideo model and training the FACT [72] backbone on these representations. Following a standard protocol, we train FACT on split2-4 and evaluate on split1. We report three standard metrics as follows:

1) Edit measures sequence-level similarity between the predicted and ground-truth label sequences after collapsing consecutive duplicates. Let $\textbf { y } = ~ ( y _ { 1 } , \dots , y _ { T } )$ and $\hat { \mathbf { y } } = ( \hat { y } _ { 1 } , \dots , \hat { y } _ { T } )$ be frame-wise labels, and let $\mathcal { C } ( \cdot )$ collapse consecutive identical labels. Denote $\operatorname { L e v } ( \cdot , \cdot )$ as the Levenshtein distance and $| \cdot |$ as the sequence length. The normalized edit score is

$$
\text { Edit } = 1 - \frac {\operatorname{Lev} \left(\mathcal {C} (\hat {\mathbf {y}}) , \mathcal {C} (\mathbf {y})\right)}{\max \left\{\left| \mathcal {C} (\hat {\mathbf {y}}) \right| , \left| \mathcal {C} (\mathbf {y}) \right| \right\}}. \tag {48}
$$

2) Acc is the frame-wise accuracy, representing the percentage of correctly labeled frames:

$$
\mathrm{Acc} = \frac {1}{T} \sum_ {t = 1} ^ {T} \mathbb {1} \left\{\hat {y} _ {t} = y _ {t} \right\}, \tag {49}
$$

where 1{·} is the indicator function.

3) F1@k is the segmental F1 at IoU threshold k. Let the ground-truth segment set be $\boldsymbol { \mathcal { S } } \ : = \ : \{ ( s _ { j } ^ { g } , e _ { j } ^ { g } , c _ { j } ^ { g } ) \}$ and the predicted set $\hat { S } = \{ ( s _ { i } ^ { p } , e _ { i } ^ { p } , c _ { i } ^ { p } ) \}$ , where $s / e$ are start/end frames and c is the class. For segments of the same class, define the temporal Intersection-over-Union as:

$$
\operatorname{IoU} \left(\left(s _ {i} ^ {p}, e _ {i} ^ {p}\right), \left(s _ {j} ^ {g}, e _ {j} ^ {g}\right)\right) = \frac {\left\{\min \left(e _ {i} ^ {p} , e _ {j} ^ {g}\right) - \max \left(s _ {i} ^ {p} , s _ {j} ^ {g}\right) \right\} _ {+}}{\max \left(e _ {i} ^ {p} , e _ {j} ^ {g}\right) - \min \left(s _ {i} ^ {p} , s _ {j} ^ {g}\right)}. \tag {50}
$$

A prediction is a true positive (TP) if it uniquely matches a ground-truth segment of the same class with $\mathrm { I o U } \geq k ;$ unmatched predictions are false positives (FP), and unmatched ground-truth segments are false negatives (FN). With precision $\begin{array} { r } { P = \frac { T P ^ { \bf { \bar { \Phi } } } } { T P + F P } } \end{array}$ and recall $\begin{array} { r } { R = \frac { T P } { T P + F N } } \end{array}$ TP+FN , we compute

$$
\mathrm{F1@k} = \frac {2 P \cdot R}{P + R}, \tag {51}
$$

where $k \in \{ 0 . 1 0 , 0 . 2 5 , 0 . 5 0 \}$ as standard thresholds.

UCF101 [93] comprises 13,320 videos from 101 human action classes, and HMDB51 [58] contains 6,766 videos from 51 action classes. For zero-shot video retrieval on the test set, we directly extract video representations using our transferred image-to-video model and perform retrieval following [68]: in each query round, one video is treated as the query and all remaining videos form the reference set. This process is repeated for every video. And we report the following metrics with the average.

1) mAP (Mean Average Precision) is the mean of per-query Average Precision (AP). Let |Q| be the number of queries, $n _ { j }$ the number of positives for query $j ,$ and $r _ { i }$ the rank of the i-th retrieved positive for that query. Then

$$
\mathrm{mAP} = \frac {1}{| \mathcal {Q} |} \sum_ {j = 1} ^ {| \mathcal {Q} |} \frac {1}{n _ {j}} \sum_ {i = 1} ^ {n _ {j}} \frac {i}{r _ {i}}. \tag {52}
$$

2) Recall@K is the fraction of queries for which at least one positive appears in the top-K results. Let ${ \mathcal { R } } _ { j } ^ { ( K ) }$ be the set of ranks $\le K$ among retrieved items for query $j ,$ , and let $\mathcal { P } _ { j }$ be the set of ranks of its positives. Then

$$
\text { Recall@ } K = \frac {1}{| \mathcal {Q} |} \sum_ {j = 1} ^ {| \mathcal {Q} |} \mathbb {1} \{\min (\mathcal {P} _ {j}) \leq K \}. \tag {53}
$$

Something-Something-v2 (SSV2) [36] is a large-scale action classification benchmark consisting of 220,847 short videos from 174 fine-grained action categories without public labels. It focuses on human-object interactions with subtle motion variations, and is widely used to evaluate a model’s capability for temporal reasoning and motion-sensitive action understanding.

For action classification on SSV2, each transferred imageto-video model is fine-tuned on the training set for 25 epochs and then evaluated on the validation set using single-clip sampling. Although this protocol is lighter than commonly used longer-schedule or multi-clip settings, it is applied uniformly to all compared methods to ensure effective and fair comparison. We report the standard top-k accuracy metric: $\operatorname { A c c } \ @ k$ measures the percentage of validation videos whose ground-truth label appears among the top-k predicted classes. Let $\mathbf { z } ^ { ( i ) } \in \mathbb { R } ^ { C }$ be the predicted logits for the i-th video over C classes, and let $y _ { i } \in \{ 1 , \ldots , C \}$ be the ground-truth label. Denote by $\pi _ { k } ( \mathbf { z } ^ { ( i ) } )$ the set of indices corresponding to the top-k largest entries in $\mathbf { z } ^ { ( i ) }$ . Then

$$
\operatorname{Acc} @ k = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} \left\{y _ {i} \in \pi_ {k} \left(\mathbf {z} ^ {(i)}\right) \right\}, \tag {54}
$$

where N is the number of validation videos and $\mathbb { 1 } \{ \cdot \}$ is the indicator function. Following common practice to present Acc@1 and Acc@5.

Chiral SSV2 [6] is a temporal order discrimination benchmark constructed from Something-Something-v2 [36]. It groups temporally opposite actions into chiral pairs, such as “sitting down” and “standing up”, and evaluates whether a video representation is sensitive to the ordering of visual change over time. Compared with standard action classification, this benchmark places stronger emphasis on timeawareness rather than semantic categorization.

Following [6], we evaluate each model using a linearprobe protocol on frozen representations. Specifically, for each chiral group, we extract frame-level representations from each video, concatenate representations along the temporal dimension to form the video representation, and train a linear classifier for binary classification. This procedure is repeated independently for every chiral group, and the final result is reported as the average classification accuracy across all groups.

Acc measures the percentage of correctly classified videos over all evaluation samples. Let $\mathbf { z } ^ { ( i ) } \in \mathbb { R } ^ { 2 }$ be the logits predicted by the linear classifier for the i-th video, and let $y _ { i } \in \{ 0 , 1 \}$ denote its ground-truth label within the corresponding chiral pair. Then

$$
\text { Acc } = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} \left\{\arg \max _ {c} z _ {c} ^ {(i)} = y _ {i} \right\}, \tag {55}
$$

where N is the total number of evaluation videos and 1{·} is the indicator function.

# D.3.3. Distance-based Trade-off Metrics Validation

To provide an interpretable assessment, we validate the distance-based metrics proposed in Sec.4. We randomly sample 1,000 videos from the Kinetics-400 validation set and compute each metric for both the original image-pretrained models and our transferred models. All metrics are computed on the same sample set for a fair comparison. We report four metrics as follows.

1) $D _ { i n t e r }$ (Inter-video distance). Let the sample set be V = {V (n)}M $\mathcal { V } = \{ V ^ { ( n ) } \} _ { n = 1 } ^ { M }$ n=1 with $M = 1 0 0 0$ . For each $V ^ { \bar { ( } n ) }$ , select the middle frame $v _ { t ^ { * } } ^ { ( n ) }$ and extract N patch representations $\{ z _ { t ^ { * } } ^ { ( n ) } ( i ) \} _ { i = 1 } ^ { N }$ . For each unordered pair $( u , v )$ with $u < v$ , define the pair-wise inter-video distance as:

$$
d (u, v) = \frac {1}{N} \sum_ {i = 1} ^ {N} \left\| \boldsymbol {z} _ {t ^ {*}} ^ {(u)} (i) - \boldsymbol {z} _ {t ^ {*}} ^ {(v)} (i) \right\| _ {2}. \tag {56}
$$

The unnormalized inter-video distance is

$$
D _ {i n t e r} ^ {o r i} = \frac {2}{M (M - 1)} \sum_ {1 \leq u <   v \leq M} d (u, v). \tag {57}
$$

Let the global center be the mean patch representation over videos, $\begin{array} { r } { \pmb { c } ( i ) = \frac { 1 } { M } \sum _ { n = 1 } ^ { M } z _ { t ^ { * } } ^ { ( n ) } ( i ) } \end{array}$ 1M , and define each video’s distance to the center as

$$
r (u) = \frac {1}{N} \sum_ {i = 1} ^ {N} \left\| \boldsymbol {z} _ {t ^ {*}} ^ {(u)} (i) - \boldsymbol {c} (i) \right\| _ {2}. \tag {58}
$$

The inter-video radius is $R _ { i n t e r } = \operatorname* { m a x } _ { u } { r ( u ) }$ , and the normalized metric is

$$
D _ {i n t e r} = \frac {D _ {i n t e r} ^ {o r i}}{2 R _ {i n t e r}}. \tag {59}
$$

2) $D _ { i n t r a }$ (Intra-video distance). For each $V ^ { ( n ) }$ , select a set of frame pairs $\mathcal { P } ^ { ( n ) } = \{ ( t _ { a } , t _ { b } ) \}$ . For a given pair, measure pair-wise intra-video distance as:

$$
d ^ {(n)} (t _ {a}, t _ {b}) = \frac {1}{N} \sum_ {i = 1} ^ {N} \left\| \boldsymbol {z} _ {t _ {a}} ^ {(n)} (i) - \boldsymbol {z} _ {t _ {b}} ^ {(n)} (i) \right\| _ {2}. \tag {60}
$$

The per-video unnormalized intra distance and its normalization radius are

$$
D _ {\text {intra}} ^ {\text {ori}, (n)} = \frac {1}{| \mathcal {P} ^ {(n)} |} \sum_ {\substack {(t _ {a}, t _ {b}) \in \mathcal {P} ^ {(n)} \\ N}} d ^ {(n)} \left(t _ {a}, t _ {b}\right), \tag{61}
$$

$$
R _ {i n t r a} ^ {(n)} = \max _ {t} \frac {1}{N} \sum_ {i = 1} ^ {N} \Big \| \pmb {z} _ {t} ^ {(n)} (i) - \bar {\pmb {z}} ^ {(n)} (i) \Big \| _ {2},
$$

where $\bar { z } ^ { ( n ) } ( i )$ is the per-video mean patch representation over the frames used for ${ \mathcal { P } } ^ { ( n ) }$ . We normalize each video by its own radius and then average:

$$
D _ {i n t r a} = \frac {1}{M} \sum_ {n = 1} ^ {M} \frac {D _ {i n t r a} ^ {o r i , (n)}}{2 R _ {i n t r a} ^ {(n)}}. \tag {62}
$$

3) D (Distance margin). The trade-off margin balances the two normalized distances with a scale factor γ:

$$
\begin{array}{l} D = D _ {i n t e r} - \gamma D _ {i n t r a}, \\ \gamma = \frac {\mathbb {E} _ {\mathcal {M}} \left[ D _ {i n t r a} ^ {o r i} \right]}{\mathbb {E} _ {\mathcal {M}} \left[ D _ {i n t e r} ^ {o r i} \right]}, \tag {63} \\ \end{array}
$$

where M indexes the set of models under comparison. Model-specific values of the scale factor γ are listed in Tab. 9 and concentrate within a narrow range. Therefore, to unify the setting, we use the average $\gamma = 0 . 3$ in practice.

4) $C y c .$ Acc. (Cycle-consistency accuracy). Given two frames forming a palindrome traversal and N patches per frame, let $A _ { t _ { a } } ^ { t _ { b } }$ and $A _ { t _ { b } } ^ { t _ { a } }$ be the patch-wise correlation transition matrices, and set $\dot { \boldsymbol { P } } = \boldsymbol { A } _ { t _ { a } } ^ { t _ { b } } \boldsymbol { A } _ { t _ { b } } ^ { t _ { a } }$ ta . The cycle-consistency accuracy is the proportion of patches returning to their original indices:

$$
C y c. A c c. = \frac {1}{N} \sum_ {i = 1} ^ {N} \mathbb {1} \left\{\arg \max _ {j} P _ {i j} = i \right\}. \tag {64}
$$

Table 9. The scale factor $\gamma = D _ { i n t r a } ^ { o r i } / D _ { i n t e r } ^ { o r i }$ and the average scale factor for each model. 

<table><tr><td>Method</td><td> $\gamma$ </td><td>Method</td><td> $\gamma$ </td></tr><tr><td>MAE</td><td>0.1855</td><td>MoCov3</td><td>0.3321</td></tr><tr><td>MAE +Ours</td><td>0.3289</td><td>MoCov3 +Ours</td><td>0.3053</td></tr><tr><td>I-JEPA</td><td>0.2283</td><td>iBOT</td><td>0.2817</td></tr><tr><td>I-JEPA +Ours</td><td>0.2645</td><td>iBOT +Ours</td><td>0.3084</td></tr><tr><td>CLIP</td><td>0.3365</td><td>DINO</td><td>0.3378</td></tr><tr><td>CLIP +Ours</td><td>0.3876</td><td>DINO +Ours</td><td>0.3505</td></tr><tr><td>BLIP</td><td>0.2489</td><td>DINOv2</td><td>0.2730</td></tr><tr><td>BLIP +Ours</td><td>0.3737</td><td>DINOv2 +Ours</td><td>0.2916</td></tr><tr><td colspan="4">Average  $\gamma : 0.3021$ </td></tr></table>

# D.4. Image-pretrained Fundamental Models

We evaluate our method using eight representative pretrained image encoders, which can be broadly categorized into three paradigms of self-supervised learning: 1) Masked modeling: MAE [45], I-JEPA [3]; 2) Contrastive learning: CLIP [82], BLIP [62], MoCo v3 [20]; 3) Self-distillation: iBOT [120], DINO [14], DINO v2 [73]. All models are pretrained on ImageNet-1k with self-supervised objectives, except for CLIP and BLIP, which are pretrained with natural language supervision. We adopt ViT-Base [28] architectures with a patch size of 16 as the backbone encoder for each model.

• MAE [45] follows an encoder-decoder architecture, where random image patches are masked and the model is trained to reconstruct the missing content at the pixel level.   
• I-JEPA [3] learns representations by predicting latent representations of masked regions. It discards the decoder and instead relies on semantic-level prediction to better capture high-level image structures.   
• CLIP [82] is a vision-language model trained with natural language supervision. It learns to align image and text embeddings in a shared feature space using a contrastive objective on WIT dataset.   
• BLIP [62] is a vision-language model that extends CLIPstyle contrastive pretraining with additional image-text matching and language modeling objectives. By jointly

Table 10. Evaluation results on frame-level and video-level tasks based on representative image models. The best results are marked in bold. 

<table><tr><td rowspan="3">Model</td><td colspan="5">Action Localization</td><td colspan="4">Video Retrieval</td><td colspan="2">Action Classification</td><td>Temporal Order Discrimination</td></tr><tr><td colspan="5">Breakfast</td><td colspan="2">UCF101</td><td colspan="2">HMDB51</td><td colspan="2">SSV2</td><td>Chiral SSV2</td></tr><tr><td>Edit</td><td>Acc</td><td>F1@0.10</td><td>F1@0.25</td><td>F1@0.50</td><td>mAP</td><td>R@1</td><td>mAP</td><td>R@1</td><td>Acc@1</td><td>Acc@5</td><td>Acc</td></tr><tr><td>CLIP</td><td>53.8</td><td>40.1</td><td>52.9</td><td>46.0</td><td>33.8</td><td>45.9</td><td>90.8</td><td>25.5</td><td>70.1</td><td>34.6</td><td>67.8</td><td>79.1</td></tr><tr><td>CLIP +Ours</td><td>54.9</td><td>40.9</td><td>52.5</td><td>46.4</td><td>34.8</td><td>49.0</td><td>96.0</td><td>27.1</td><td>71.3</td><td>35.5</td><td>68.8</td><td>80.5</td></tr><tr><td>BLIP</td><td>57.3</td><td>47.3</td><td>55.6</td><td>49.4</td><td>36.7</td><td>54.4</td><td>96.4</td><td>29.4</td><td>73.3</td><td>39.9</td><td>72.4</td><td>81.8</td></tr><tr><td>BLIP +Ours</td><td>58.6</td><td>49.1</td><td>57.5</td><td>51.0</td><td>38.1</td><td>55.2</td><td>97.0</td><td>29.9</td><td>73.5</td><td>41.4</td><td>72.6</td><td>82.8</td></tr><tr><td>iBOT</td><td>55.5</td><td>40.7</td><td>53.2</td><td>47.6</td><td>35.3</td><td>33.4</td><td>92.0</td><td>18.1</td><td>59.9</td><td>38.1</td><td>68.9</td><td>80.1</td></tr><tr><td>iBOT +Ours</td><td>56.3</td><td>42.9</td><td>53.5</td><td>47.5</td><td>37.3</td><td>34.6</td><td>94.9</td><td>18.8</td><td>63.9</td><td>40.6</td><td>71.3</td><td>82.3</td></tr><tr><td>DINO v2</td><td>58.5</td><td>44.1</td><td>57.0</td><td>51.3</td><td>38.2</td><td>37.1</td><td>93.3</td><td>18.7</td><td>62.1</td><td>39.2</td><td>71.0</td><td>84.8</td></tr><tr><td>DINO v2 +Ours</td><td>61.2</td><td>50.5</td><td>60.0</td><td>54.0</td><td>41.2</td><td>39.3</td><td>95.2</td><td>20.0</td><td>67.0</td><td>40.4</td><td>72.1</td><td>85.2</td></tr></table>

optimizing these objectives on large-scale web data, BLIP learns richer cross-modal representations and achieves stronger performance on image captioning and visual question answering.

• MoCo v3 [20] is a contrastive learning method that adapts Momentum Contrast to Vision Transformers, employing a siamese architecture with an online encoder and a momentum-updated target encoder, and discards the negative sample queue used in earlier versions.   
• DINO [14] adopts a self-distillation structure with Vision Transformers as the encoder. It encourages consistent representations across different views of the same image.   
• iBOT [120] builds upon DINO by introducing additional alignment on dense patch tokens. It aligns both the global [CLS] token and local patch-level features between two views, thus encouraging fine-grained spatial consistency in the learned representations.   
• DINO v2 [73] extends iBOT by incorporating various design improvements, including better centering techniques [85], regularization strategies like KoLeo loss [86], and resolution-adaptive training [102]. In our experiments, we exclude computationally intensive techniques to ensure a consistent and fair comparison across models.

# D.5. Competitors

We compare our approach against two categories of strong baselines: recent state-of-the-art (SOTA) video representation learning methods and image-to-video adaptation frameworks. For all baselines, we use the officially released pretrained weights without any additional training or finetuning.

1) Video representation learning methods: These methods are specifically designed to learn spatiotemporal representations from raw video inputs, often relying on temporal masking or reconstruction-based objectives.

• VideoMAE [101] extends the masked modeling paradigm to videos by randomly masking spatiotemporal tubes and reconstructing the missing pixels. It adopts a high masking

ratio to encourage the encoder to capture both appearance and motion features.

• MAE-ST [33] adapts MAE to spatiotemporal data by explicitly incorporating temporal modeling modules into the encoder to better capture dynamic patterns.   
• DropMAE [111] applies spatial-attention dropout in masked modeling, encouraging the model to attend to motion cues for temporal discriminability.   
• SiamMAE [39] adopts a Siamese structure where the past frame and a masked version of the current frame are jointly encoded. A conditional decoder is employed to reconstruct the missing patches, thereby promoting temporal consistency across frames.   
• CropMAE [30] generalizes SiamMAE by using different crops or augmentations of the same frame as input, encouraging invariance under intra-frame transformations.   
• RSP [52] formulates temporal modeling as a stochastic frame prediction task. It learns to reconstruct future frames from current ones by modeling both prior and posterior distributions over latent motion variables.

2) Image-to-video adaptation methods: These methods aim to adapt pretrained image models to instance-level video understanding tasks by integrating lightweight modules that enable temporal reasoning, while keeping most of the backbone parameters frozen and only updating a small subset.

• AIM [115] introduces a lightweight adapter into a frozen ViT backbone, enabling spatiotemporal adaptation along spatial and temporal dimensions, facilitating efficient transfer from static to dynamic inputs.

• ST-Adapter [74] proposes a 3D bottleneck adapter into the CLIP-pretrained ViT model, which enables the model to reason about dynamic video content at a small taskspecific parameter cost.

• ZeroI2V [66] introduces spatial-temporal dual-headed attention mechanism combined with a linear adaptation layer, thus enabling the transfer of frozen image models to video tasks and supporting zero additional inference cost via structural reparameterization.

![](images/a56ac85948c3e1ed9b076b7463bacc11a89d6a015d01fbb0da25db1facb62531.jpg)

<details>
<summary>line</summary>

| Step | Acc. (w/o PEA) | Acc. (w/ PEA) | J&Fm (w/o PEA) | J&Fm (w/ PEA) |
|------|----------------|---------------|----------------|---------------|
| 0    | 30             | 25            | 50             | 50            |
| 1000 | 100            | 70            | 45             | 60            |
| 2000 | 100            | 70            | 25             | 60            |
| 3000 | 100            | 70            | 20             | 60            |
| 4000 | 100            | 70            | 20             | 60            |
| 5000 | 100            | 70            | 20             | 60            |
</details>

![](images/b279d1d43977ffa947d1a14aafbd2da070002816793df4d00f8661974681c47e.jpg)

<details>
<summary>line</summary>

| Step | Acc. (w/o PEA) | Acc. (w/ PEA) | mIoU (w/o PEA) | mIoU (w/ PEA) |
| ---- | -------------- | ------------- | -------------- | ------------- |
| 0    | 28             | 22            | 36             | 32            |
| 1000 | 100            | 70            | 24             | 34            |
| 2000 | 100            | 72            | 20             | 34            |
| 3000 | 100            | 72            | 20             | 34            |
| 4000 | 100            | 72            | 20             | 34            |
| 5000 | 100            | 72            | 20             | 34            |
</details>

![](images/f334a8b3e10b846060b1f3ae030f589a193aa2d8a6c0ae8f91252538b401a628.jpg)

<details>
<summary>line</summary>

| Step | Acc. (w/o PEA) | Acc. (w/ PEA) | PCK@0.1 (w/o PEA) | PCK@0.1 (w/ PEA) |
| ---- | -------------- | ------------- | ----------------- | ---------------- |
| 0    | 25             | 25            | 43                | 43               |
| 1000 | 98             | 65            | 43                | 47               |
| 2000 | 100            | 68            | 25                | 47               |
| 3000 | 100            | 68            | 25                | 47               |
| 4000 | 100            | 68            | 25                | 47               |
| 5000 | 100            | 68            | 25                | 47               |
</details>

![](images/be748e17297ae94b36d14fbd0a6011b559b2e36d157d5393dcf6590021cdad66.jpg)

<details>
<summary>line</summary>

| Step | Acc. (w/o PEA) | Acc. (w/ PEA) | J&Fm (w/o PEA) | J&Fm (w/ PEA) |
|------|----------------|---------------|----------------|---------------|
| 0    | 45             | 35            | 65             | 65            |
| 1000 | 95             | 55            | 25             | 60            |
| 2000 | 100            | 55            | 35             | 60            |
| 3000 | 100            | 55            | 25             | 60            |
| 4000 | 100            | 55            | 20             | 60            |
| 5000 | 100            | 55            | 20             | 60            |
</details>

![](images/9d79265bab8f54f1037b862a55e1e8db732d2c19fa013b373ee164fa2d706753.jpg)

<details>
<summary>line</summary>

| Step | Acc. (w/o PEA) | Acc. (w/ PEA) | mIoU (w/o PEA) | mIoU (w/ PEA) |
|------|----------------|---------------|----------------|---------------|
| 0    | 45             | 35            | 90             | 90            |
| 1000 | 95             | 50            | 20             | 95            |
| 2000 | 100            | 55            | 18             | 100           |
| 3000 | 100            | 55            | 17             | 100           |
| 4000 | 100            | 55            | 16             | 100           |
| 5000 | 100            | 55            | 16             | 100           |
</details>

![](images/d2ea27ffa81accb3eb6ab5d136ea25b1044c620701e530a17e8c8d8dd10090b1.jpg)

<details>
<summary>line</summary>

| Step | Acc. (w/o PEA) | Acc. (w/ PEA) | PCK@0.1 (w/o PEA) | PCK@0.1 (w/ PEA) |
| ---- | -------------- | ------------- | ----------------- | ---------------- |
| 0    | 45             | 35            | -                 | 80               |
| 1000 | 95             | 55            | 85                | 95               |
| 2000 | 98             | 55            | 45                | 98               |
| 3000 | 98             | 55            | 35                | 98               |
| 4000 | 98             | 55            | 32                | 98               |
| 5000 | 98             | 55            | 32                | 98               |
</details>

Figure 6. Cycle-consistent accuracy and downstream performance during training with or without the PEA strategies across three tasks based on MAE (line 1) and DINO (line 2).

Table 11. Extended comparison with representative methods on the DAVIS-2017 validation set. Methods are grouped by their core settings for a broader reference. 

<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td rowspan="2">Backbone</td><td colspan="3">DAVIS-2017</td></tr><tr><td> $\mathcal{J}\&\mathcal{F}_{m}$ </td><td> $\mathcal{J}_{m}$ </td><td> $\mathcal{F}_{m}$ </td></tr><tr><td rowspan="4">Dedicated VOS Systems</td><td>STCN [21]</td><td>ResNet50</td><td>85.4</td><td>82.2</td><td>88.6</td></tr><tr><td>SwinB-AOT-L [116]</td><td>Swin-B</td><td>85.4</td><td>82.4</td><td>88.4</td></tr><tr><td>SimVOS-B [112]</td><td>ViT-B/16</td><td>88.0</td><td>85.0</td><td>91.0</td></tr><tr><td>Cutie-base [22]</td><td>ResNet50</td><td>87.9</td><td>84.6</td><td>91.1</td></tr><tr><td rowspan="2">Segmentation Foundation Models</td><td>SAM 2 [84]</td><td>Hiera-B+</td><td>90.2</td><td>87.0</td><td>93.4</td></tr><tr><td>SAM 2 [84]</td><td>Hiera-L</td><td>90.7</td><td>87.5</td><td>94.0</td></tr><tr><td rowspan="5">Self-Supervised Video Pre-training</td><td>VideoMAE [101]</td><td>ViT-L/16</td><td>45.0</td><td>43.6</td><td>46.5</td></tr><tr><td>MAE-ST [33]</td><td>ViT-L/16</td><td>54.6</td><td>55.5</td><td>53.6</td></tr><tr><td>SiamMAE [39]</td><td>ViT-B/16</td><td>60.9</td><td>59.4</td><td>62.4</td></tr><tr><td>CropMAE [30]</td><td>ViT-B/16</td><td>57.8</td><td>56.9</td><td>58.7</td></tr><tr><td>RSP [52]</td><td>ViT-B/16</td><td>60.5</td><td>57.8</td><td>63.2</td></tr><tr><td rowspan="6">Self-Supervised Image Pre-training +Ours</td><td>DINO [14]</td><td>ViT-B/16</td><td>63.2</td><td>60.9</td><td>65.5</td></tr><tr><td>DINO + Ours</td><td>ViT-B/16</td><td>64.2</td><td>62.3</td><td>66.0</td></tr><tr><td>DINOv2 [73]</td><td>ViT-B/16</td><td>63.1</td><td>61.6</td><td>64.5</td></tr><tr><td>DINOv2 + Ours</td><td>ViT-B/16</td><td>63.7</td><td>61.9</td><td>65.4</td></tr><tr><td>iBOT [120]</td><td>ViT-B/16</td><td>64.6</td><td>63.0</td><td>66.1</td></tr><tr><td>iBOT + Ours</td><td>ViT-B/16</td><td>65.1</td><td>63.3</td><td>66.9</td></tr></table>

# E. Detailed Experiments Results

# E.1. Comparison with Task-Specific SOTAs

To provide a broader view, we compare our method with several representative VOS systems and recent segmentation foundation models on DAVIS-2017 validation in Tab. 11. Our work focuses on general representation pre-training for direct transfer across multiple tasks rather than taskspecific designs, and thereby applies lightweight transfer for evaluation per standard self-supervised learning protocols. Thus, our datasets, computing resources, and architectures are not aligned with specialized SoTA methods for individual tasks such as video object segmentation (VOS).

# E.2. Detailed Results of Frame-/Video-Level Tasks

We further evaluate the transferred models on several frameand video-level downstream tasks: temporal action localization on Breakfast [59] using the FACT [72] backbone, zero-shot video retrieval on UCF101 and HMDB51 [58, 93], fine-tuned action classification on Something-Something-v2 (SSV2) [36], and temporal order discrimination via linear probing on Chiral SSV2 [6].

The quantitative results of transferred representations from four representative image models on both frame-level and video-level downstream tasks are depicted in Tab. 10. Our method delivers steady performance improvements across these tasks. For instance, on frame-level tasks, it achieves an average improvement of 2.80% Acc on Breakfast, indicating enhanced temporal awareness in image models. On video-level tasks, it brings a 2.58% R@1 improvement on HMDB51 and a 1.53% Acc@1 gain on SSV2, which validates the preserved semantic discrimination ability. These results indicate that our method generalizes well across different task granularities, highlighting its potential as a versatile solution for image-to-video transfer.

# E.3. Training Dynamics

We visualize the training dynamics of MAE and DINO across three downstream tasks in Figure 6. The plots show the cycle-consistency accuracy (i.e., the percentage of patches that return to their original positions after a cycle traversal) together with the downstream performance over training steps. Without the PEA strategy, the downstream performance drops sharply within the first two epochs, even when the cycle-consistency accuracy is close to 100%. This indicates that the model exploits the absolute positional encoding as a shortcut instead of learning temporal correspondences that remain reliable when the temporal distance between frames grows.

In contrast, when we apply the proposed PEA strategy, the cycle-consistency accuracy increases gradually, and the final value converges to a small stable range that depends on the model architecture and hyperparameter settings. This behavior is reasonable, since in real-world videos, correspondence quality naturally degrades as time passes: the first and last frames in a propagation chain can differ greatly due to camera motion and non-rigid object deformation, which leads to unavoidable information loss. On the Kinetics-400 dataset, the empirical cycle-consistency accuracy stabilizes around 50% ∼ 70% when the temporal interval is set to $\delta = 0 . 1 5$ . By promoting effective dense correspondences between frames and reducing reliance on positional cues, PEA leads to more stable improvements in downstream performance and highlights its role in learning robust temporal representations.

# E.4. Shortcut Phenomenon in Training

Tab. 12 compares the performance of our method trained with and without the proposed Positional Encoding Augmentation (PEA) strategy. As shown, removing PEA consistently leads to substantial performance degradation, with 4.4% ∼ 37.3% drop in $\mathcal { T } \& \mathcal { F } _ { \mathrm { m } }$ on DAVIS and 4.9% ∼ 22.8% drop in mIoU on VIP. This is primarily due to the model exploiting absolute positional encodings as shortcuts, resulting in dimensional collapse and degraded representations. The issue is particularly severe in self-distillation architectures, which rely heavily on positional alignment between teacher and student branches. This highlights the brittleness of image-pretrained representations when transferred to video and underscores that image-to-video transfer is a non-trivial challenge. In contrast, applying PEA consistently improves performance across all three downstream tasks, indicating the effectiveness of resisting shortcuts induced by the positional encoding mechanism of ViT.

# E.5. Additional Ablation Study

In Figure 7, we study the effects of the interpolation ratio α and the regularization weight λ. A moderate value of α gives the best performance since a small α cannot effectively suppress shortcut learning, while a large one disrupts relative positional cues and harms correspondence learning. Similarly, λ controls the strength of the semantic separability constraint: too small values may cause dimensional collapse in the projection space, whereas overly strong regularization reduces the flexibility needed to adapt the representations. Overall, both hyperparameters influence performance in a relatively mild range, and good results can be obtained with moderate choices.

Table 12. Impact of Positional Encoding Augmentation (PEA) strategy on representation quality across three downstream tasks. 

<table><tr><td>Image Model</td><td>Method</td><td>VIP mIoU</td><td>DAVIS17  $\mathcal{J}\&\mathcal{F}_{m}$ </td><td>JHMDB PCK@0.1</td></tr><tr><td rowspan="3">MAE</td><td>Vanilla</td><td>29.3</td><td>52.4</td><td>41.6</td></tr><tr><td>w/o PEA</td><td> $16.2_{-13.1}$ </td><td> $26.2_{-26.2}$ </td><td> $38.5_{-3.1}$ </td></tr><tr><td>w/ PEA</td><td> $33.8_{+4.5}$ </td><td> $59.6_{+7.2}$ </td><td> $48.4_{+6.8}$ </td></tr><tr><td rowspan="3">I-JEPA</td><td>Vanilla</td><td>31.5</td><td>53.9</td><td>42.6</td></tr><tr><td>w/o PEA</td><td> $26.6_{-4.9}$ </td><td> $49.5_{-4.4}$ </td><td> $44.1_{+1.5}$ </td></tr><tr><td>w/ PEA</td><td> $35.3_{+3.8}$ </td><td> $58.7_{+4.8}$ </td><td> $44.4_{+1.8}$ </td></tr><tr><td rowspan="3">MoCo v3</td><td>Vanilla</td><td>38.8</td><td>62.6</td><td>43.6</td></tr><tr><td>w/o PEA</td><td> $23.8_{-15.0}$ </td><td> $42.8_{-19.8}$ </td><td> $42.2_{-1.4}$ </td></tr><tr><td>w/ PEA</td><td> $39.8_{+1.0}$ </td><td> $62.9_{+0.3}$ </td><td> $45.3_{+1.7}$ </td></tr><tr><td rowspan="3">iBOT</td><td>Vanilla</td><td>39.6</td><td>64.6</td><td>45.7</td></tr><tr><td>w/o PEA</td><td> $16.8_{-22.8}$ </td><td> $27.3_{-37.3}$ </td><td> $38.2_{-7.5}$ </td></tr><tr><td>w/ PEA</td><td> $40.8_{+1.2}$ </td><td> $65.1_{+0.5}$ </td><td> $46.1_{+0.4}$ </td></tr><tr><td rowspan="3">DINO</td><td>Vanilla</td><td>39.1</td><td>63.2</td><td>44.4</td></tr><tr><td>w/o PEA</td><td> $17.5_{-21.6}$ </td><td> $30.6_{-32.6}$ </td><td> $39.3_{-5.1}$ </td></tr><tr><td>w/ PEA</td><td> $39.8_{+0.7}$ </td><td> $64.2_{+1.0}$ </td><td> $46.2_{+1.8}$ </td></tr><tr><td rowspan="3">DINO v2</td><td>Vanilla</td><td>38.4</td><td>63.1</td><td>46.6</td></tr><tr><td>w/o PEA</td><td> $17.7_{-20.7}$ </td><td> $30.0_{-33.1}$ </td><td> $39.1_{-7.5}$ </td></tr><tr><td>w/ PEA</td><td> $39.9_{+1.5}$ </td><td> $63.7_{+0.6}$ </td><td> $47.3_{+0.7}$ </td></tr></table>

Tab. 13 analyzes the sensitivity of the temporal sampling interval δ and the softmax temperature τ . A suitable δ balances visible motion and visual continuity, which is important for learning meaningful frame-level correspondences, while a moderate τ maintains an appropriate level of similarity sharpening. The model shows limited sensitivity to variations in these two hyperparameters, and the performance remains stable across a reasonable range. To ensure consistency and fair comparison across all experiments, we fix $\tau = 0 . 0 3$ and $\delta = 0 . 1 5$ .

We further conduct a robustness and generalization analysis by varying the PEA crop strategy (Tab. 14a) and the model patch size alongside positional encoding variants (Tab. 14b). PEA remains stable across a wide range of crop choices and generalizes well across various patch sizes (e.g., 8, 14, and 16). Moreover, PEA is compatible with modern designs such as RoPE [94]. By interpolating and cropping on the RoPE coordinate grid, PEA effectively mitigates shortcut behaviors, further demonstrating the robustness of our method.

As shown in Tab. 15, we investigate the impact of different regularization objectives. The KL-based regularization matches the distribution of transferred video representations to that of frozen image features. This helps preserve the inherited semantic geometry and prevents feature collapse by aligning distance relationships (as discussed in Sec. 4). Compared to a strict element-wise MSE loss, KL divergence provides a softer, distribution-level constraint. This allows for sufficient temporal adaptation while effectively maintaining semantic separability. Consequently, KL regulariza-

![](images/67f9a282e3a4e6d17b5a6b5695699524c329807ae088a42eada579da38fc9dac.jpg)

<details>
<summary>bar</summary>

| Weight λ | Ratio α | J&P_m on DAVIS (%) |
| -------- | ------- | ----------------- |
| 0        | 0       | 58.5              |
| 0        | 1       | 59.0              |
| 0        | 3       | 59.5              |
| 0        | 5       | 60.0              |
| 0        | 7       | 59.5              |
| 1        | 0       | 58.0              |
| 1        | 1       | 58.5              |
| 1        | 3       | 59.0              |
| 1        | 5       | 59.5              |
| 1        | 7       | 60.0              |
| 2        | 0       | 57.5              |
| 2        | 1       | 58.0              |
| 2        | 3       | 58.5              |
| 2        | 5       | 59.0              |
| 2        | 7       | 59.5              |
| 3        | 0       | 57.0              |
| 3        | 1       | 57.5              |
| 3        | 3       | 58.0              |
| 3        | 5       | 58.5              |
| 3        | 7       | 59.0              |
| 4        | 0       | 56.5              |
| 4        | 1       | 57.0              |
| 4        | 3       | 57.5              |
| 4        | 5       | 58.0              |
| 4        | 7       | 58.5              |
| 5        | 0       | 56.0              |
| 5        | 1       | 56.5              |
| 5        | 3       | 57.0              |
| 5        | 5       | 57.5              |
| 5        | 7       | 58.0              |
| 6        | 0       | 55.5              |
| 6        | 1       | 56.0              |
| 6        | 3       | 56.5              |
| 6        | 5       | 57.0              |
| 6        | 7       | 57.5              |
| ...      | ...     | ...               |
| ...      | ...     | ...               |
| ...      | ...     | ...               |
| ...      | ...     | ...               |
| ...      | ...     | ...               |
| ...      | ...     | ...               |
| ...      | ...     | ...               |
| ...      | ...     | ...               |
| ...      | ...     | ...               |
| ...      | ...     | ...               |
| ...      | ...     | ...<nl>
</details>

![](images/3c0c2e2d6729154a99f745dd1c2e92f9ed533c1d4ad83b212586cb7557af8ba5.jpg)

<details>
<summary>bar</summary>

| Weight λ | Ratio α = 1 | Ratio α = 3 | Ratio α = 5 | Ratio α = 7 |
| -------- | ----------- | ----------- | ----------- | ----------- |
| 0        | 30.0        | 31.0        | 32.0        | 33.0        |
| 1        | 30.5        | 31.5        | 32.5        | 33.5        |
| 2        | 31.0        | 32.0        | 33.0        | 34.0        |
| 3        | 31.5        | 32.5        | 33.5        | 34.0        |
| 4        | 32.0        | 33.0        | 34.0        | 34.0        |
| 5        | 32.5        | 33.5        | 34.0        | 34.0        |
| 6        | 33.0        | 34.0        | 34.0        | 34.0        |
| 7        | 33.5        | 34.0        | 34.0        | 34.0        |
| 8        | 34.0        | 34.0        | 34.0        | 34.0        |
| 9        | 34.0        | 34.0        | 34.0        | 34.0        |
| 10       | 34.0        | 34.0        | 34.0        | 34.0        |
| 11       | 34.0        | 34.0        | 34.0        | 34.0        |
| 12       | 34.0        | 34.0        | 34.0        | 34.0        |
| 13       | 34.0        | 34.0        | 34.0        | 34.0        |
| 14       | 34.0        | 34.0        | 34.0        | 34.0        |
| 15       | 34.0        | 34.0        | 34.0        | 34.0        |
| 16       | 34.0        | 34.0        | 34.0        | 34.0        |
| 17       | 34.0        | 34.0        | 34.0        | 34.0        |
| 18       | 34.0        | 34.0        | 34.0        | 34.0        |
| 19       | 34.0        | 34.0        | 34.0        | 34.0        |
| 20       | 34.0        | 34.0        | 34.0        | 34.0        |
</details>

![](images/e9fbb3514c8450240db91f2f3b74351db6e3c032151ba333150b88183e9402be.jpg)

<details>
<summary>bar</summary>

| Weight λ | Ratio α | PCK@0.1 on JHMDB (%) |
| :--- | :--- | :--- |
| 0 | 7 | 48 |
| 1 | 7 | 47 |
| 3 | 7 | 46 |
| 5 | 7 | 45 |
| 7 | 7 | 44 |
| 20 | 10 | 49 |
| 10 | 10 | 48 |
| 5 | 10 | 47 |
| 3 | 10 | 46 |
| 1 | 10 | 45 |
| 0 | 20 | 48 |
| 1 | 20 | 47 |
| 3 | 20 | 46 |
| 5 | 20 | 45 |
| 7 | 20 | 44 |
| 20 | 30 | 49 |
| 10 | 30 | 48 |
| 5 | 30 | 47 |
| 3 | 30 | 46 |
| 1 | 30 | 45 |
| 0 | 10 | 48 |
| 1 | 10 | 47 |
| 3 | 10 | 46 |
| 5 | 10 | 45 |
| 7 | 10 | 44 |
| 20 | 20 | 49 |
| 10 | 20 | 48 |
| 5 | 20 | 47 |
| 3 | 20 | 46 |
| 1 | 20 | 45 |
| 0 | 30 | 49 |
| 1 | 30 | 48 |
| 3 | 30 | 47 |
| 5 | 30 | 46 |
| 7 | 30 | 45 |
| 20 | 10 | 49 |
| 10 | 10 | 48 |
| 5 | 10 | 47 |
| 3 | 10 | 46 |
| 1 | 10 | 45 |
| 0 | 20 | 49 |
| 1 | 20 | 48 |
| 3 | 20 | 47 |
| 5 | 20 | 46 |
| 7 | 20 | 45 |
| 20 | 30 | 49 |
| 10 | 30 | 48 |
| 5 | 30 | 47 |
| 3 | 30 | 46 |
| 1 | 30 | 45 |
| 0 | 10 | 49 |
| 1 | 10 | 48 |
| 3 | 10 | 47 |
| 5 | 10 | 46 |
| 7 | 10 | 45 |
| 20 | 20 | 49 |
| 10 | 20 | 48 |
| 5 | 20 | 47 |
| 3 | 20 | 46 |
| 1 | 20 | 45 |
| 0 | 20 | 49 |
</details>

Figure 7. 3D bar charts for ablation results on interpolation ratio α and regularization weight λ across three tasks using MAE.

Table 13. Sensitivity analysis on temporal interval δ and softmax temporature τ . Default settings are highlighted with blue .   
(a) Ablation on temporal interval δ. 

<table><tr><td>Base Model</td><td> $\delta$ </td><td>VIP mIoU</td><td>DAVIS17  $\mathcal{J}\&\mathcal{F}_{\text{m}}$ </td><td>JHMDB PCK@0.1</td></tr><tr><td rowspan="5">MAE</td><td>0.05</td><td>33.9</td><td>59.6</td><td>48.6</td></tr><tr><td>0.10</td><td>33.9</td><td>59.7</td><td>48.4</td></tr><tr><td>0.15</td><td>33.8</td><td>59.6</td><td>48.4</td></tr><tr><td>0.20</td><td>33.6</td><td>59.7</td><td>48.5</td></tr><tr><td>0.25</td><td>33.6</td><td>59.6</td><td>48.4</td></tr><tr><td rowspan="5">DINO</td><td>0.05</td><td>39.7</td><td>63.9</td><td>46.2</td></tr><tr><td>0.10</td><td>39.9</td><td>64.0</td><td>46.1</td></tr><tr><td>0.15</td><td>39.8</td><td>64.2</td><td>46.2</td></tr><tr><td>0.20</td><td>39.9</td><td>64.2</td><td>46.1</td></tr><tr><td>0.25</td><td>39.9</td><td>64.2</td><td>46.1</td></tr></table>

(b) Ablation on softmax temporature τ . 

<table><tr><td>Base Model</td><td> $\tau$ </td><td>VIP mIoU</td><td>DAVIS17  $\mathcal{J}\&\mathcal{F}_{\text{m}}$ </td><td>JHMDB PCK@0.1</td></tr><tr><td rowspan="5">MAE</td><td>0.01</td><td>32.6</td><td>60.0</td><td>48.2</td></tr><tr><td>0.02</td><td>33.4</td><td>60.0</td><td>48.5</td></tr><tr><td>0.03</td><td>33.8</td><td>59.6</td><td>48.4</td></tr><tr><td>0.04</td><td>33.7</td><td>58.9</td><td>48.5</td></tr><tr><td>0.05</td><td>33.6</td><td>58.6</td><td>48.4</td></tr><tr><td rowspan="5">DINO</td><td>0.01</td><td>39.2</td><td>63.7</td><td>46.0</td></tr><tr><td>0.02</td><td>40.0</td><td>63.9</td><td>46.1</td></tr><tr><td>0.03</td><td>39.8</td><td>64.2</td><td>46.2</td></tr><tr><td>0.04</td><td>39.9</td><td>64.0</td><td>46.0</td></tr><tr><td>0.05</td><td>39.9</td><td>63.8</td><td>45.9</td></tr></table>

Table 14. Robustness and generalization analysis of PEA strategy across DINO series features.   
(a) Robustness across PEA crop manners. 

<table><tr><td>Base Model</td><td>PEA Crop</td><td>VIP mIoU</td><td>DAVIS17  $\mathcal{J}\&\mathcal{F}_{m}$ </td><td>JHMDB PCK@0.1</td></tr><tr><td rowspan="4">DINO</td><td>center</td><td>64.1</td><td>39.3</td><td>46.9</td></tr><tr><td>random</td><td>64.2</td><td>39.8</td><td>46.2</td></tr><tr><td>edge</td><td>64.1</td><td>39.9</td><td>46.2</td></tr><tr><td>multiple</td><td>64.3</td><td>39.8</td><td>46.2</td></tr></table>

(b) Generalization across PE variants and patch sizes. 

<table><tr><td>PEA</td><td> $L_{reg}$ </td><td>DINO (Abs. PE) ViT-S/8</td><td>DINO v2 (Abs. PE) ViT-S/14</td><td>DINO v3 (RoPE) ViT-S/16</td></tr><tr><td colspan="2">Vanilla</td><td>71.7</td><td>64.7</td><td>67.3</td></tr><tr><td>✗</td><td>√</td><td>71.1</td><td>63.9</td><td>65.8</td></tr><tr><td>√</td><td>√</td><td>72.3</td><td>65.1</td><td>67.9</td></tr></table>

Table 15. Ablation study on the choice of regularization loss $( L _ { r e g } )$ using the DINO backbone. 

<table><tr><td>PEA</td><td> $L_{reg}$ </td><td>DAVIS</td><td>VIP</td><td>JHMDB</td><td> $D_{inter}$ </td><td> $D_{intra}$ </td><td> $D(\uparrow)$ </td></tr><tr><td colspan="2">DINO</td><td>63.2</td><td>39.1</td><td>44.4</td><td>0.5756</td><td>0.2144</td><td>0.5112</td></tr><tr><td>✗</td><td>KL</td><td>61.8</td><td>38.0</td><td>46.1</td><td>0.6241</td><td>0.2404</td><td>0.5520</td></tr><tr><td>√</td><td>MSE</td><td>62.2</td><td>38.1</td><td>46.1</td><td>0.6142</td><td>0.2370</td><td>0.5431</td></tr><tr><td>√</td><td>KL</td><td>64.2</td><td>39.8</td><td>46.2</td><td>0.6246</td><td>0.2316</td><td>0.5551</td></tr></table>

tion tends to increase the normalized inter-video distance $( D _ { i n t e r } )$ , which aligns perfectly with the observed improvements in downstream task performance.

# F. Additional Visualizations

# F.1. Inter-frame Correspondence

We visualize the inter-frame correspondence learned by the projection layer g in Figure 8. The results indicate that most patches establish consistent matches across frames and successfully return to their original locations through the forward-backward cycle. Notably, due to factors such as camera motion and non-rigid object deformation, patch correspondences between $\boldsymbol { v } _ { t _ { 1 } } ^ { f }$ and ${ \mathbf { } } v _ { t _ { 2 } }$ are not strictly bijective. A single patch in $\boldsymbol { v } _ { t _ { 1 } } ^ { f }$ often correlates to multiple adjacent regions in $\mathbf { } v _ { t _ { 2 } } .$ , resulting in a correlation matrix product

![](images/917a9e953c0ca6b09e669694ddf1813ba32838e2281723bdfc050bb637283076.jpg)  
Figure 8. Cross-frame correspondence learned with our method. Patches with the same color box represent correspondence.

$A _ { t _ { 1 } } ^ { t _ { 2 } } \widetilde { A } _ { t _ { 2 } } ^ { t _ { 1 } }$ that exhibits a diagonally dominant structure rather than an exactly equal to the identity matrix I. This observation reveals the dilemma of the original contrastive random walk strategy: it needs to constrain the matrix to the identity matrix to ensure good cyclic consistency, but we cannot make it a perfect identity matrix because it would allow the model to take advantage of shortcuts in displaying positional encoding. This further justifies the necessity of our proposed PEA strategy, which effectively suppresses shortcut matching to stabilize correspondence learning.

# F.2. Downstream Task Performance

In Figures 9 and 10, we compare the performance of original image-pretrained models and our transferred models across three downstream tasks. Our method shows visible improvements in several challenging scenarios, such as rapid movements, complex object boundaries, and motioninduced artifacts, where the original models often underperform. These results suggest that incorporating temporal correspondence and strengthening semantic structure improves image-to-video representation transfer, validating the effectiveness of our method.

# G. Detailed Related Work

# G.1. Self-supervised Visual Representation

The rapid progress of self-supervised learning has enabled models to acquire generalizable representations for diverse downstream tasks in both the image and video domains. Depending on the nature of the pretraining objective, existing approaches can be broadly categorized into three paradigms.

Contrastive learning learns invariant representations by maximizing agreement between relevant instances while pushing apart representations of different instances. Early methods in the image domain construct positive and negative pairs [18, 20, 44] or apply diverse augmentations [13, 19, 37] to generate contrasting views. These approaches demonstrate strong generalization capabilities [47] and have been

![](images/6db6e56d73c975dacae181f4d4cb4544df3eb444ddc1930f2333d77013fbcebc.jpg)

Figure 9. Visualization comparison across three downstream tasks based on MAE.   
![](images/9eaf120fa96c74fae9e637ebbb3c3385b7b543d4a6930fd72de78f695b7dba3c.jpg)

<details>
<summary>text_image</summary>

CLIP
+Ours
0%
25%
50%
100%
0%
25%
50%
100%
0%
25%
50%
100%
(a) Video Object Segmentation on DAVIS-2017
</details>

![](images/47d2d495c2ab434d5170a126c04dcc9079f0be5357dc97ea41c14c8ff2a930c0.jpg)

<details>
<summary>text_image</summary>

CLIP
+Ours
0%
25%
50%
100%
0%
25%
50%
100%
</details>

(b) Semantic Part Propagation on VIP

![](images/c346cd907cc97af549cd08bc68ebf3a5ea043783efce9f6dd04747876f86fc37.jpg)

<details>
<summary>text_image</summary>

CLIP
0%
wath=weight@200°
50%
wath=weight@200°
100%
wath=weight@200°
100%
wath=weight@200°
100%
+Ours
0%
wath=weight@200°
50%
wath=weight@200°
100%
wath=weight@200°
100%
wath=weight@200°
100%
wath=weight@200°
100%
</details>

(c) Human Pose Propagation on JHMDB   
Figure 10. Visualization comparison across three downstream tasks based on CLIP.

successfully extended to the video domain. By leveraging 3D convolutions [32], temporal self-attention [1, 9, 12], or interframe contrastive objectives [42, 50, 97, 117], such methods benefit from spatiotemporal cues and have shown promising results on discriminative tasks such as action recognition and video retrieval [58, 93].

Masked modeling aims to reconstruct the original RGB values of masked image patches in the pixel space [3, 7, 8, 45, 96, 113]. A representative method is MAE [45], which employs an encoder-decoder architecture based on Vision Transformers [28] to restore the masked regions, thereby capturing structural dependencies within the images. By incorporating the additional temporal dimension, MAE can be naturally extended for video representation learning [33, 77, 101, 104, 111]. To alleviate the computational cost of dense modeling, recent methods focus on more efficient designs. SiamMAE [39] leverages sparsely sampled frames, asymmetric masking, and a conditional Siamese architecture, motivating subsequent works that improve frame selection and predictive mechanisms [30, 52, 114].

Self-distillation methods supervise a student network using outputs from a teacher network without relying on explicit labels, often focusing on restoring latent representations rather than raw pixels. This encourages the learning of high-level semantic information, aligning with principles of information compression [54, 99]. DINO [14] adopts a self-distillation framework with Vision Transformers to align patch-level representations across views. Subsequently, iBOT [120] and DINO v2 [73] extend this paradigm by enforcing consistency in both global [CLS] tokens and dense patch representations.

# G.2. Image-to-video Transfer Learning

Temporal structure enhancement methods typically design training objectives in a two-stage training manner based on self-supervised image contrastive learning frameworks [37, 44]. In the first stage, models are pretrained on image datasets to learn static representations for instancelevel discrimination [46, 63], or on synthetic videos to capture object motion patterns [27, 64]. In the second stage, the models are fine-tuned on real video datasets to refine temporal correspondences, enabling them to perform specific video tasks. However, the high spatiotemporal complexity hinders swift cross-domain representation transfer, motivating the exploration of parameter-efficient fine-tuning alternatives in subsequent works.

Parameter-efficient fine-tuning methods aim to adapt pretrained models to video tasks by updating only a small fraction of parameters. Specifically, several methods insert adapters into Vision Transformer [28] pretrained by CLIP [82] in a series or parallel way, enabling spatialtemporal joint adaptation through expanded convolution or attention modules [17, 66, 67, 74, 115]. Other approaches decouple spatial and temporal modeling using dualbranch architectures [75, 81], enabling separate reasoning across spatial and temporal dimensions. These adaptation methods are often trained on supervised action recognition datasets [36, 55], which require further fine-tuning when applied to different benchmarks. More recent work explores object-centric adaptation via slot attention [70], demonstrating the potential of using image-pretrained encoders for dense prediction tasks [80]. In a related direction, Pro-LIP [31] shows that fine-tuning only the visual projector is effective for few-shot CLIP adaptation, showing the strong transfer capacity of lightweight projection-based adaptation.

# G.3. Temporal Cycle Consistency

The inherent visual correspondence between temporally adjacent observations provides a powerful supervisory signal to capture spatiotemporal coherence in videos [4, 92]. Leveraging this property, numerous studies attempt to learn semantically consistent representations with a cycle structure, showing effectiveness in dense-level video tasks, including object segmentation [79, 121], motion estimation [53], and point tracking [11, 15]. Early methods mainly focus on tracking patches or objects across frames in a bidirectional manner [65, 105, 119], while others align the feature distributions among videos from the same category to enforce semantic consistency [29, 40, 110]. Another line of work introduces random walk strategies [10, 51, 89], where representation learning is guided by maximizing the probability of each patch returning to itself via a forward-backward cycle.

# H. Additional Discussions

# H.1. Limitation and Future Work

This work explores a more efficient and effective approach to transferring image representations to the video domain. In this work, we mainly focus on ViT-based backbones under the evaluated settings. For future work, we plan to extend the method to other visual backbones, including lightweight architectures (e.g., CNNs, ResNets) and emerging largescale vision models, to assess whether the observed tradeoff is a general property of visual representations for video understanding.

# H.2. Broader Impact

We examine the trade-off between intra-video temporal consistency and inter-video semantic separability in visual representations and, based on this view, propose a method for image-to-video representation transfer learning. The proposed method achieves competitive or superior performance compared with models pretrained on video from scratch, providing a lightweight alternative for video representation learning. It may also provide a useful perspective for future research on image-to-video transfer in broader scenarios.