# Improving Visual Token Reduction via Rectifying Distortions for Efficient Multimodal LLM Inference

Hyeonwoo Cho 1 Donghyeon Baek 1 Yewon Kim 1 Bumsub Ham 1 2

## Abstract

Recent advancements in Multimodal Large Language Models (MLLMs) have achieved remarkable success in vision-language tasks, yet the quadratic computational complexity arising from the vast number of visual tokens incurs significant memory and latency bottlenecks. While visual token reduction (VTR) strategies have been explored to mitigate this burden, existing methods overlook the positional and attentional consistency between the full and reduced sequences, resulting in a distorted representation. To this end, we propose RESTORE, a novel VTR framework that rectifies the positional and attentional distortions while maintaining efficiency. Specifically, we present a simple yet effective calibration method that restores lost visual attention by augmenting attention weights based on relative distances. We also introduce a distinctive anchor selection for token merging to mitigate information loss during feature averaging. Experimental results on multiple benchmarks demonstrate that our method consistently improves the accuracy of various reduction methods, achieving state-of-the-art performance while maintaining computational efficiency. Project page is available at https://cvlab.yonsei.ac.kr/ projects/RESTORE

## 1. Introduction

Recent advancements in Multimodal Large Language Models (MLLMs) (Liu et al., 2023; 2024a; Bai et al., 2025; Lin et al., 2024) have achieved remarkable success in interpreting complex visual information and translating it into coherent textual narratives, bridging the modality gap be-

1Yonsei University 2Korea Institute of Science and Technology (KIST). Correspondence to: Bumsub Ham <bumsub.ham@yonsei.ac.kr>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

tween vision and language. MLLMs leverage the generalization capabilities of Large Language Models (LLMs) (Radford et al., 2019; Brown et al., 2020; Touvron et al., 2023; Bai et al., 2023) with a pretrained visual encoder such as CLIP (Radford et al., 2021). An input image is projected into a sequence of visual tokens by the visual encoder, then processed via the LLM to generate text responses based on text inputs. Given that the computational complexity of the attention mechanism scales quadratically with sequence length, the vast number of visual tokens introduces a substantial bottleneck as they often reach into the thousands for high-resolution images or video inputs. Such an extensive visual sequence incurs significant memory and latency overheads.

To mitigate the computational burden, recent research has explored diverse strategies for visual token reduction (VTR) by pruning redundant tokens (Chen et al., 2024; Zhang et al., 2025b;a; Zou et al., 2025) or merging similar tokens (Bolya et al., 2023; Yang et al., 2025; Shang et al., 2025). Token pruning maintains the original information of retained tokens but sacrifices the context from pruned ones. Conversely, although token merging preserves global context by aggregating multiple tokens, averaging features during token merging leads to information loss of fine-grained details. A hybrid approach, VisionZip (Yang et al., 2025), attempts to combine the strengths of these approaches by first pruning and then merging the remaining tokens. Specifically, it preserves high-attention tokens and merges the rest into representatives (i.e., anchor tokens) chosen via uniform sampling. However, the anchor tokens might fail to represent their groups, leading to significant information loss during the merging process.

Beyond the reduction strategies, we observe that existing VTR methods fail to preserve the total attention weights of visual tokens compared to the full token sequence. This attenuation stems from the normalization property of the softmax function. As the number of visual tokens decreases, the probabilities originally allocated to the reduced tokens are redistributed to the retained tokens. Due to the exponential nature of softmax, the redistribution amplifies the remaining tokens that possessed large attention weights. The loss of visual attention weights causes the model to neglect visual contexts and rely on textual information, potentially leading to weak visual grounding or hallucinations. We show in Fig. 1 the token sequences and their attention weights when the last text token serves as the query. Figure 1(a) illustrates the full token sequence, where attention is distributed across all tokens without reduction. For the reduced token sequence, existing methods either assign new contiguous position indices (Chen et al., 2024; Xing et al., 2024; Zhang et al., 2025a) (Fig. 1(b)) or retain the initial position indices from the full token sequence (Yang et al., 2025; Alvar et al., 2025; Zou et al., 2025) (Fig. 1(c)). The reindexing strategy reduces the relative distances between tokens, which helps to partially preserve the total attention weights of visual tokens. That is, squeezing the relative distances mitigates the positional bias of rotary position embedding (RoPE) (Su et al., 2024) in the original indices of the full sequence. However, this strategy disturbs spatial relationships between tokens. For instance, the substantial distance between the first visual token and the last text token is shortened, disrupting the spatial relationships compared to the full token sequence. This problem could be addressed by retaining original position indices, but at the cost of a significant drop in visual attention. Due to the positional bias, the attention weights originally assigned to reduced visual tokens are redistributed towards text tokens that are closer to the query.

![](images/b5bf86170d9e333a3be2ebb0d7e439bbcd5f3526180c5ddeeee27a004da4f213.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Input"] --> B["Group 1"]
  B --> C["Group 2"]
  C --> D["Group 3"]
  D --> E["..."]
  E --> F["Group N"]
  F --> G["Group R"]
  G --> H["Group S"]
  H --> I["Group T"]
  I --> J["Output"]
    style A fill:#ccc
    style B fill:#ccc
    style C fill:#ccc
    style D fill:#ccc
    style E fill:#ccc
    style F fill:#ccc
    style G fill:#ccc
    style H fill:#ccc
    style I fill:#ccc
    style J fill:#ccc
```
</details>

(a) The full token sequence

![](images/b6bd4d727aea71b3c2eb2942b871f137d1b31131498b1d6e6c119022debec576.jpg)

![](images/86631b2f2343b7818b8f9f0fb912774fd06cd67890dc54c143eb6a3bdca77d38.jpg)  
(c) Retaining position indices

![](images/5fd3a4df9f159a0094c6ee63cf9a436d62e7f3f231abd2fa48d65cafc362bb28.jpg)

![](images/430c2f53e67eea883b05b3293c8ccc49504e5154169b602fb6a4262eadbbeb59.jpg)  
(b) Reindexing position indices

![](images/07edcae73d279645b97248f10d2728e2a34bd942a43f598d0e8bcfc5d1f59910.jpg)

![](images/38fe4f0df660534defd11343deb3645af94cac93287f07f34a84240ef9320c60.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["sys"] --> B["+"]
  B --> C["vis"]
  C --> D["+"]
  D --> E["txt"]
  E --> F["+"]
  F --> G["+"]
  G --> H["+"]
  H --> I["+"]
  I --> J["+"]
  J --> K["+"]
  K --> L["+"]
  L --> M["+"]
  M --> N["+"]
  N --> O["+"]
  O --> P["+"]
  P --> Q["+"]
  Q --> R["+"]
  R --> S["+"]
  S --> T["+"]
  T --> U["+"]
  U --> V["+"]
  V --> W["+"]
  W --> X["+"]
  X --> Y["+"]
  Y --> Z["+"]
  Z --> AA["+"]
  AA --> AB["+"]
  AB --> AC["+"]
  AC --> AD["+"]
  AD --> AE["+"]
  AE --> AF["+"]
  AF --> AG["+"]
  AG --> AH["+"]
```
</details>

![](images/2896734762ffc5884f527fd4c46ebe36ed9c83a12c0d812c5ea3e94dc5a37dba.jpg)  
(d) Rectifying position indices and attention

![](images/daf56b5b61ffa42913f11b50dcf21395e74a6efbdf65b0fcce9ac6aa87c644a5.jpg)

<details>
<summary>text_image</summary>

System token
Visual token
Text token
Retained visual token w/ insufficient attention
Retained visual token w/ sufficient attention
Text token w/ excessive attention
→ Positional bias
+ Rectifying positional bias
</details>

Figure 1. Illustration of the impact of visual token reduction on the internal attention mechanism of the LLM within MLLMs (e.g., LLaVA (Liu et al., 2024a)). (a) The full token sequence. (b) Reindexing position indices assigns contiguous indices to the reduced sequence. (c) Retaining position indices preserves the original indices of the retained tokens from (a). (d) We rectify distortions by retaining original position indices and calibrating the attention weights of the retained tokens.

In this paper, we propose RESTORE, a novel framework that REctifies diStortions in visual TOken REduction. To address this, we introduce a calibration method that restores attention weights while preserving original position indices (Fig. 1(d)). This attention calibration counteracts the positional bias of RoPE, recovering the lost total attention weights of visual tokens. We also introduce a novel anchor selection strategy for token merging that considers both representativeness and discriminativeness, minimizing information loss during feature averaging. Extensive experiments on several MLLM benchmarks demonstrate that our method consistently improves the accuracy of various VTR methods while maintaining efficiency. Our main contributions are summarized as follows:

• We analyze the attentional and positional distortions overlooked by existing VTR methods and propose a calibration method that rectifies the distortions.  
• We introduce a distinctive anchor token selection strategy, mitigating information loss during token merging by selecting representative and discriminative anchor tokens.  
• We provide comprehensive experimental results on multiple MLLM benchmarks, demonstrating that our method significantly improves accuracy across various VTR methods while preserving efficiency.

## 2. Related Work

Text-aware visual token reduction. Early VTR approaches for MLLMs focus on retaining visual tokens that are most relevant to the text input. The seminal work of FastV (Chen et al., 2024) introduces a text-aware token pruning method that leverages the cross-attention technique between visual and text tokens to identify and retain the most relevant visual tokens. Specifically, exploiting the autoregressive nature of LLMs, it selectively prunes visual tokens that exhibit low attention weights with respect to the last text token. FastV demonstrates that MLLMs maintain high accuracy even with a significantly reduced number of visual tokens. Subsequent works build upon this idea with advanced mechanisms to assess token importance. Sparse-VLM (Zhang et al., 2025b) introduces a pruning metric that averages cross-attention weights towards a selected subset of text tokens that are highly correlated with the visual input. FitPrune (Ye et al., 2025) identifies the importance score of visual tokens by multiplying visual self-attention weights with cross-attention weights, considering visual saliency and text relevance jointly. While these methods achieve high accuracy with text-relevant visual information, they still require significant computation within the LLM layers to compute attention weights before reduction, which limits the efficiency.

Text-agnostic visual token reduction. Another line of research focuses on reducing visual tokens without considering text relevance. These approaches reduce visual tokens before feeding the tokens into the LLM, avoiding the computational overhead of text-aware methods. They prune or merge redundant tokens based on attention weights relative to the [CLS] token for measuring importance or a selfcorrelation of visual features for measuring similarity. For pruning, VisPruner (Zhang et al., 2025a) incorporates both attention weights and feature similarity for token selection, while DART (Wen et al., 2025) leverages only feature similarity to eliminate redundancy. DivPrune (Alvar et al., 2025) emphasizes diversity between retained tokens by selecting a subset of tokens that are mutually distant. Our distinctive anchor token selection shares a similar motivation, but distinguishes itself in the criteria for token merging. To be specific, we incorporate the correlation between anchors and remaining tokens to mitigate information loss during feature averaging. HoloV (Zou et al., 2025) introduces a dynamic scoring based on feature variance and attention saliency to adaptively allocate the token budget across spatial partitions. Token pruning ensures the integrity of the retained tokens, but it discards information from pruned tokens. This exclusion often leads to the loss of text-relevant visual details, causing performance degradation. Merging-based approaches have been proposed to preserve information by aggregating multiple tokens into a representative anchor token. These methods primarily diverge in their strategies for selecting anchor tokens. PruMerge (Shang et al., 2025) selects anchor tokens with high attention weights, while VisionZip (Yang et al., 2025) performs merging with uniformly distributed anchor tokens. These strategies are likely to select suboptimal anchor tokens since they exhibit low similarity to the tokens they merge. Consequently, the anchor tokens fail to represent their local neighborhoods, exacerbating information loss during feature averaging. To overcome this limitation, we introduce a distinctive anchor selection for token merging that prioritizes tokens with high representativeness for their groups and discriminativeness from other anchor tokens.

Furthermore, we identify and address the overlooked issue of attentional distortion in existing VTR methods. Existing approaches differ in their position assignment to the reduced token sequence, exploiting either reindexing (Chen et al., 2024; Xing et al., 2024; Zhang et al., 2025a) or retaining original indices (Yang et al., 2025; Alvar et al., 2025; Zou et al., 2025). However, the impact of these position assignments on attention weights and subsequent model performance remains unexplored. To the best of our knowledge, we are the first to provide an in-depth analysis of the position assignment for the reduced token sequence. Building on the analysis, we propose a novel method to calibrate attention distortion.

## 3. Method

In this section, we first describe the MLLM architecture briefly (Sec. 3.1). We then provide a detailed description of our framework with an in-depth analysis (Sec. 3.2), and introduce a distinctive anchor selection for token merging (Sec. 3.3).

## 3.1. Preliminaries

VTR in MLLMs. We adopt the architecture of standard MLLMs, such as LLaVA (Liu et al., 2023), which comprises a visual encoder, a projector, and an LLM. Given an input image, the visual encoder extracts visual features, which are then mapped into the LLM’s embedding space via the projector to obtain a sequence of visual tokens $\mathbf { X } _ { \mathrm { v i s } } ~ \in ~ \bar { \mathbb { R } } ^ { N _ { \mathrm { v i s } } \times d }$ , where $N _ { \mathrm { v i s } }$ denotes the original number of visual tokens and d is the hidden dimension. To mitigate computational overhead, the VTR method is applied to ${ \bf X } _ { \mathrm { v i s } }$ before feeding it into the LLM. This process transforms the original sequence into a reduced set of visual tokens $\hat { \bf X } _ { \mathrm { v i s } } \in \mathbb { R } ^ { n _ { \mathrm { v i s } } \times d }$ , where $n _ { \mathrm { v i s } } \ll N _ { \mathrm { v i s } }$ . The reduced visual tokens are then concatenated with system tokens $\mathbf { X } _ { \mathrm { s y s } } ~ \in ~ \mathbb { R } ^ { N _ { \mathrm { s y s } } \times d }$ and text tokens $\mathbf { X } _ { \mathrm { t x t } } \in \mathbb { R } ^ { \tilde { N } _ { \mathrm { t x t } } \times d }$ to form the token sequence $\mathbf { X } = [ \mathbf { X } _ { \mathrm { s y s } } ; \hat { \mathbf { X } } _ { \mathrm { v i s } } ; \mathbf { X } _ { \mathrm { t x t } } ] \in \mathbb { R } ^ { N \times d }$ , where $N = N _ { \mathrm { s y s } } + n _ { \mathrm { v i s } } + N _ { \mathrm { t x t } } .$ . This concatenated sequence X serves as the input to the LLM for response generation.

Self-attention with RoPE. Modern LLMs primarily employ RoPE (Su et al., 2024) to encode spatial relationships between tokens. RoPE groups adjacent pairs of the feature dimension to form complex numbers, and then applies a rotation in the complex domain based on the token’s position index. Let ${ \bf x } _ { m } , { \bf x } _ { n } \in \mathbb { R } ^ { d _ { h } }$ denote the feature vectors of the tokens at position m and n within the sequence X, respectively, where $d _ { h }$ denotes the hidden dimension of each head in the multi-head attention (Vaswani et al., 2017). Given the frequency parameters $\Theta = \{ \theta _ { j } = 1 0 0 0 0 ^ { - 2 ( j - 1 ) / d _ { h } } \mid$ | $j \in [ 1 , \ldots , d _ { h } / 2 ] \}$ , the query and key vectors for tokens at positions m and n are as follows:

$$
\mathbf {q} _ {m} = \mathbf {W} _ {q} (\mathbf {x} _ {m}) e ^ {i m \Theta}, \quad \mathbf {k} _ {n} = \mathbf {W} _ {k} (\mathbf {x} _ {n}) e ^ {i n \Theta}, (1)
$$

where $\mathbf { W _ { q } }$ and $\mathbf { W _ { k } }$ are the projection matrices for the query and key in the complex domain, respectively. A logit of attention weight $z _ { m , n }$ is then computed as the real part of the inner product between the complex query and key vectors as follows:

$$
z _ {m, n} = \frac {\operatorname{Re} \left(\sum_ {j = 1} ^ {d _ {h} / 2} \mathbf {W} _ {q} (\mathbf {x} _ {m}) _ {j} \mathbf {W} _ {k} (\mathbf {x} _ {n}) _ {j} ^ {*} e ^ {i | m - n | \theta_ {j}}\right)}{\sqrt {d _ {h}}}. \tag {2}
$$

This formulation demonstrates that the attention weights depend on the relative distance $| m - n |$ through the phase shift $e ^ { i | m - n | \theta _ { j } }$ , enabling the attention mechanism to capture spatial relationships. However, VTR disrupts this mechanism by reducing tokens or altering positional indices, which distorts attentional or positional information. In the following section, we analyze these distortions and propose a method to rectify them.

## 3.2. Rectifying Distortions

Analysis. To analyze the impact of position assignments on the attention mechanism within the LLM, we compute the proportion of attention weights allocated to visual tokens averaged across all heads in each layer of LLaVA-1.5 (Liu et al., 2024a). That is, we compare the ratio of the original total attention weights of visual tokens that the reduced visual tokens retain. We measure this ratio for two distinct scenarios: when the query is a visual token (Visualto-Visual) and when it is a text token (Text-to-Visual). We conduct a comparative analysis using a pruning method of HoloV (Zou et al., 2025) on the GQA (Hudson & Manning, 2019) dataset under four settings: the baseline using the full token sequence (Fig. 1(a), $N _ { v i s } = 5 7 6 )$ , and reduced sequences $( n _ { v i s } ~ = ~ 6 4 )$ with reindexing indices (Fig. 1(b)), retaining original indices (Fig. 1(c)), and our proposed method (Fig. 1(d)). We show in Fig. 2(a) and Fig. 2(b) visual-to-visual and text-to-visual scenarios, respectively. These figures demonstrate that neither position assignment preserves the total attention weights of visual tokens at the baseline level. The reindexing strategy (dashed red line) exhibits a decline in attention weights, while retaining the original position indices (dashed green line) leads to a further substantial drop.

The attention attenuation of visual tokens is attributed to two primary reasons. First, the total attention weight of visual tokens inevitably diminishes in proportion to the reduced token count. Second, the further degradation observed in the retaining position indices arises from the positional bias of RoPE, where tokens with larger relative distances are penalized. This attenuation compromises the model’s ability to attend to visual information, resulting in weakened visual grounding and hallucinations. Therefore, it is essential for visual tokens to restore the attention weights of the reduced tokens to a level comparable to that of the full token sequence. To address these, we propose to calibrate attention weights, recovering the lost visual attention weights caused by the reduced token sequence and the long-term decay of RoPE.

![](images/3c7b6aef0934379c31dc1bdb646321e1eeb57ef64061409f1534d2b2f07ed82e.jpg)

<details>
<summary>line chart</summary>

| Layer | The full token sequence | Reindexing position indices | Retaining position indices | Rectifying position indices and attention |
|-------|--------------------------|------------------------------|-----------------------------|-------------------------------------------|
| 0     | 80                       | 40                           | 40                          | 40                                        |
| 5     | 15                       | 15                           | 15                          | 15                                        |
| 10    | 35                       | 25                           | 25                          | 35                                        |
| 15    | 25                       | 20                           | 20                          | 25                                        |
| 20    | 15                       | 15                           | 15                          | 15                                        |
| 25    | 15                       | 15                           | 15                          | 15                                        |
| 30    | 40                       | 30                           | 30                          | 40                                        |
</details>

(a) Visual-to-visual attention.  
![](images/17aee297cef89afdd5958aa7bd0979dd018fe633e54f2d40934bc5e484661432.jpg)

<details>
<summary>line chart</summary>

| Layer | The full token sequence | Reindexing position indices | Retaining position indices | Rectifying position indices and attention |
|-------|--------------------------|-----------------------------|-----------------------------|-------------------------------------------|
| 0     | 80                       | 35                          | 30                          | 75                                        |
| 1     | 40                       | 20                          | 15                          | 30                                        |
| 2     | 10                       | 5                           | 5                           | 10                                        |
| 3     | 5                        | 5                           | 5                           | 5                                         |
| 4     | 5                        | 5                           | 5                           | 5                                         |
| 5     | 5                        | 5                           | 5                           | 5                                         |
| 6     | 5                        | 5                           | 5                           | 5                                         |
| 7     | 5                        | 5                           | 5                           | 5                                         |
| 8     | 5                        | 5                           | 5                           | 5                                         |
| 9     | 5                        | 5                           | 5                           | 5                                         |
| 10    | 5                        | 5                           | 5                           | 5                                         |
| 11    | 5                        | 5                           | 5                           | 5                                         |
| 12    | 5                        | 5                           | 5                           | 5                                         |
| 13    | 5                        | 5                           | 5                           | 5                                         |
| 14    | 5                        | 5                           | 5                           | 5                                         |
| 15    | 5                        | 5                           | 5                           | 5                                         |
| 16    | 5                        | 5                           | 5                           | 5                                         |
| 17    | 5                        | 5                           | 5                           | 5                                         |
| 18    | 5                        | 5                           | 5                           | 5                                         |
| 19    | 5                        | 5                           | 5                           | 5                                         |
| 20    | 5                        | 5                           | 5                           | 5                                         |
| 21    | 5                        | 5                           | 5                           | 5                                         |
| 22    | 5                        | 5                           | 5                           | 5                                         |
| 23    | 5                        | 5                           | 5                           | 5                                         |
| 24    | 5                        | 5                           | 5                           | 5                                         |
| 25    | 5                        | 5                           | 5                           | 5                                         |
| 26    | 5                        | 5                           | 5                           | 5                                         |
| 27    | 5                        | 5                           | 5                           | 5                                         |
| 28    | 5                        | 5                           | 5                           | 5                                         |
| 29    | 5                        | 5                           | 5                           | 5                                         |
| 30    | 10                       | 10                          | 10                          | 10                                        |
| 31    | 10                       | 10                          | 10                          | 10                                        |
| Total                 | ~10                      | ~10                         | ~10                         | ~10                                       |
</details>

(b) Text-to-visual attention.  
Figure 2. Comparison of average attention proportions assigned to visual tokens within the LLM (a) when the query is a visual token and (b) when the query is a text token.

Attention calibration. To address the attention attenuation while preserving spatial relationships, we first measure the impact of the long-term decay of RoPE quantitatively. We derive this decay through the isolation of the positional encoding component from Eq. 2. Specifically, extracting the real part of the rotation components and averaging them across the feature dimension $d _ { h }$ , we quantify how the attention weight attenuates as a function of the relative distance $| m - n |$ . This yields the decay function $\mathcal { D } ( | m - n | )$ as:

$$
\mathcal {D} (| m - n |) = \frac {2}{d _ {h}} \sum_ {j = 1} ^ {d _ {h} / 2} \cos \left(| m - n | \theta_ {j}\right). \tag {3}
$$

This function demonstrates that attention weights diminish with an oscillation as |m−n| increases (Fig. 3 (blue)). When retaining original indices, the sparse visual tokens maintain large relative distances from the text query, causing their attention weights to be heavily penalized by the decay D. On the other hand, text tokens remain adjacent to each other, maintaining relatively high values of D. This disparity is further exacerbated by the exponential nature of the softmax function in the reduced sequence. Since text tokens possess larger softmax logits due to their close distance, the softmax redistributes the probability toward them. Consequently, the sparse visual tokens fail to preserve the total attention weights originally allocated to the visual tokens in the full sequence. This attenuation compels the model to neglect visual contexts and rely on textual information. To rectify this, we introduce a distance-aware calibration term to the attention weights. The goal is to compensate for the decay $\mathcal { D } ( | m - n | )$ , leading distant visual tokens to regain their significance. Incorporating the token size $s _ { n }$ for merging, the calibrated attention $\hat { A } _ { m , n }$ is formulated as:

$$
\hat {A} _ {m, n} = \frac {\exp \left(z _ {m , n} + \log s _ {n} (c - \mathcal {D} (| m - n |))\right)}{\sum_ {i = 1} ^ {N} \exp \left(z _ {m , i} + \log s _ {i} (c - \mathcal {D} (| m - i |))\right)}, \tag {4}
$$

where $s _ { n }$ denotes the number of tokens that have been merged into the n-th token, and c is a constant ensuring the rectification term remains positive. The logarithmic term log $s _ { n }$ ensures that a merged token receives weight proportional to the number of original tokens it represents, a technique introduced in ToMe (Bolya et al., 2023). In the presence of the positional bias induced by RoPE, attention weights are shifted toward text tokens as the visual sequence becomes sparse. Thus relying solely on $s _ { n }$ fails to prevent the suppression of visual context. To address this, we augment $s _ { n }$ with the distance-aware component $( c - \mathcal { D } ( | m - n | ) )$ to counteract the positional bias. The distance-aware term acts as a counterweight to the RoPE decay (Fig. 3 (orange)). This restores the magnitude of attention weights for distant visual tokens, aligning the distribution with that of the full token sequence as shown in Fig. 2 (solid green line). Note that while log $s _ { n }$ was successfully employed in ToMe for visual-only tasks, we argue that it is insufficient for MLLMs due to the interference of textual tokens.

Furthermore, while our derivation primarily focuses on standard 1D RoPE, the fundamental principle of distance-aware calibration seamlessly generalizes to multi-dimensional spatial encodings, such as the multimodal RoPE (M-RoPE) employed in recent models (e.g., Qwen2.5-VL (Bai et al., 2025)). We provide the theoretical generalization (Appendix A) and empirical validation for M-RoPE (Sec. 4).

## 3.3. Distinctive Anchor Token Selection

To minimize information loss during the merging process, it is crucial to select anchor tokens that are not only informative but also mutually exclusive to avoid redundancy. The previous methods rely on uniform sampling (Yang et al., 2025) or high-attention tokens (Shang et al., 2025) for selecting anchor tokens. However, these approaches suffer from two limitations: (1) Selecting anchor tokens with low correlation to non-anchor tokens results in poor representativeness. In this scenario, the anchor token fails to serve as a centroid for its cluster, exacerbating information loss during merging stage. (2) Selecting multiple anchor tokens that are highly correlated to each other introduces redundancy, as these tokens occupy overlapping regions in feature space and should ideally be merged rather than preserved individually. To address these, inspired by density peak clustering (Rodriguez & Laio, 2014), we propose an anchor token selection strategy based on feature similarity, guided by two criteria: representativeness that ensures anchor tokens are highly correlated to other tokens, and discriminativeness that prevents redundancy among selected anchors. We first define the representativeness R as the sum of pairwise correlations $\mathbf { C } _ { i j }$ with other tokens as follows:

![](images/365b66e03d95a07281da27f2856f254610f5b857ab969e22bfe40ca43d00e3a4.jpg)  
Figure 3. Visualizations of the long-term decay $\mathcal { D } ( | m - n | )$ (blue) and our calibration term (orange). The calibration term increases with relative distance to counteract the long-term decay.

$$
\mathcal {R} _ {i} = \sum_ {j = 1} ^ {N _ {\mathrm{vis}}} \mathbf {C} _ {i j}, \quad \text { where } \quad \mathbf {C} = \mathbf {X} _ {\mathrm{vis}} \mathbf {X} _ {\mathrm{vis}} ^ {T} / \| \mathbf {X} _ {\mathrm{vis}} \| ^ {2}. \tag {5}
$$

$\mathcal { R } _ { i }$ quantifies the accumulated similarity of the i-th token to all other visual tokens, indicating its potential to serve as an anchor token. However, relying solely on $\mathcal { R } _ { i }$ may lead to redundant anchor selection if candidates are highly correlated to each other. To identify and suppress such redundancy, we introduce a binary mask M that flags superior tokens (i.e., tokens with higher representativeness) as follows:

$$
\mathbf {M} _ {i j} = \mathbb {I} (\mathcal {R} _ {j} > \mathcal {R} _ {i}). \tag {6}
$$

Subsequently, we apply this mask to C to isolate correlation solely towards superior counterparts. The masked correlation matrix Cˆ is defined as:

$$
\hat {\mathbf {C}} _ {i j} = \left\{ \begin{array}{l l} \mathbf {C} _ {i j} & \text { if   } \mathbf {M} _ {i j} = 1 \\ - \infty & \text { otherwise } \end{array} \right.. \tag {7}
$$

Based on ${ \hat { \mathbf { C } } } ,$ we measure the redundancy of the i-th token by finding its maximum correlation with any superior token $( \operatorname* { m a x } _ { j } \hat { \mathbf { C } } _ { i j } )$ . A high maximum value implies that the token is substantially covered by a more representative token. Conversely, a low value indicates that the token captures unique features. We therefore define the discriminativeness as $1 \mathrm { ~ - ~ } \operatorname* { m a x } _ { j } \hat { \mathbf { C } } _ { i j }$ . Finally, the anchor set $\mathcal { A }$ is selected by weighting the representativeness with discriminativeness as follows:

$$
\mathcal {A} = \text { Top - K } \left(\mathcal {R} _ {i} \odot (1 - \max _ {j} \hat {\mathbf {C}} _ {i j})\right), \tag {8}
$$

where Top-K(·) selects the top-K tokens, and ⊙ denotes element-wise multiplication. With the optimal anchor set ${ \mathcal { A } } ,$ the other tokens are merged into the anchor tokens with which they share the highest correlation. Note that this process incurs no additional computational overhead, as we reuse the pairwise correlation values from the pre-computed correlation matrix C.

## 4. Experiments

In this section, we describe implementation details (Sec. 4.1), and present quantitative comparisons between previous VTR methods and our method (Sec. 4.2). We then conduct a detailed analysis of our framework (Sec. 4.3). Additional results and discussions are provided in the Appendix.

## 4.1. Implementation details

Datasets and models. We evaluate our method on multiple MLLM benchmarks, including GQA (Hudson & Manning, 2019), MMBench (Liu et al., 2024b), MME (Fu et al., 2025), POPE (Li et al., 2023), Science-QA (SQA) (Lu et al., 2022), VQAv2 (Goyal et al., 2017), TextVQA (Singh et al., 2019), SEED-Bench (Li et al., 2024) datasets for image question answering tasks. We perform experiments using LLaVA-1.5-7B, LLaVA-NeXT-7B (Liu et al., 2024a), and Qwen2.5-VL-7B (Bai et al., 2025).

Baselines. We compare several VTR approaches including text-aware pruning methods such as FastV (Chen et al., 2024), PDrop (Xing et al., 2024), SparseVLM (Zhang et al., 2025b), and text-agnostic methods such as VisionZip (Yang et al., 2025), DivPrune (Alvar et al., 2025), VisPruner (Zhang et al., 2025a), HoloV (Zou et al., 2025). We integrate our framework with the text-agnostic methods for evaluation, as they are more efficient than text-aware methods. Following the protocol of VisionZip, we adopt a hybrid reduction strategy that sequentially applies pruning and merging to integrate our merging method. Specifically, we first prune $\gamma n _ { v i s }$ tokens based on the pruning criteria of the four baselines, and then merge the remaining $( 1 - \gamma ) n _ { v i s }$ tokens. Unless otherwise specified, we set the pruning ratio to $\gamma = 0 . 5 ( i . e .$ ., the token budget is equally distributed between the pruning and merging stages). For the hyperparameter c, we set $c = 2$ by default, ensuring the minimum value of the calibration term is 1.

## 4.2. Results

We show in Table 1 the results on eight benchmarks across three different token retaining ratios. The experimental results demonstrate that our proposed framework consistently enhances the performance of various text-agnostic VTR baselines, including VisionZip, DivPrune, VisPruner, and HoloV. When retaining 192 or 128 tokens, our method achieves nearly lossless performance compared to the vanilla baseline. For instance, when integrated with Vis-Pruner and HoloV at a retention of 192 tokens, our approach maintains over 98% of the original performance. For an aggressive reduction to 64 tokens, our method with HoloV achieves an average accuracy of 96.5%, significantly outperforming the baseline HoloV by 4.3%. This indicates that our method effectively rectifies distortions and preserves visual information even under challenging reduction scenarios. We also show in Table 2 the results of LLaVA-NeXT-7B on multiple benchmarks. Given that this model processes extensive visual sequences, it inherently contains significant redundancy. Our method consistently achieves performance improvements across methods and retention ratios, demonstrating the versatility of our approach across different MLLM architectures. We further evaluate our framework on Qwen2.5-VL-7B-Instruct (Bai et al., 2025), which leverages M-RoPE. Since our distance-aware calibration naturally generalizes to such multi-dimensional encodings (Appendix A), it remains directly applicable. As shown in Table 3, our method consistently improves all baselines across every retention ratio.

## 4.3. Discussions

Ablations. We show in Table 4 an ablation study on the components of our framework. Unless otherwise specified, all studies in this section report accuracy averaged over the eight benchmarks in Table 1 under the aggressive retention ratio of $n _ { v i s } = 6 4$ , where each component is evaluated by isolating its effect while fixing the others. We adopt this challenging ratio to make the contribution of each component clearly observable, and use HoloV as the base method, which is the most recent baseline. The baseline ① represents HoloV using only pruning. The results in rows ② and ③ show that applying either retaining position indices or attention calibration leads to performance degradation compared to the baseline. Specifically, retaining original indices without attention calibration (③) drops the accuracy from 92.2% to 90.6%. This supports our analysis that

Table 1. Comparison with VTR methods for LLaVA-1.5-7B (Liu et al., 2024a) on eight benchmarks and the average score across them. For baselines, all experiment results are re-implemented from their official codebases under the same environments. Best results at each reduction ratio in bold.

<table><tr><td>Type</td><td>Method</td><td>Average</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td> $SQA^{IMG}$ </td><td> $VQA^{V2}$ </td><td> $VQA^{Text}$ </td><td>SEED</td></tr><tr><td colspan="11">Using All Visual Tokens, 576 Tokens (100%)</td></tr><tr><td>-</td><td>LLaVA-1.5-7B</td><td>100%</td><td>61.9</td><td>64.6</td><td>1862</td><td>85.9</td><td>69.5</td><td>78.5</td><td>58.2</td><td>58.6</td></tr><tr><td colspan="11">Retain 192 Tokens (33.3%)</td></tr><tr><td rowspan="3">Text-aware</td><td>FastV (ECCV24)</td><td>96.0%</td><td>57.1</td><td>64.4</td><td>1821</td><td>75.8</td><td>68.9</td><td>74.7</td><td>57.8</td><td>56.3</td></tr><tr><td>PDrop (CVPR25)</td><td>96.8%</td><td>58.0</td><td>62.9</td><td>1790</td><td>84.0</td><td>68.9</td><td>76.0</td><td>57.1</td><td>55.8</td></tr><tr><td>SparseVLM (ICML25)</td><td>98.1%</td><td>59.5</td><td>64.2</td><td>1782</td><td>85.4</td><td>68.7</td><td>77.0</td><td>57.7</td><td>57.3</td></tr><tr><td rowspan="9">Text-agnostic</td><td>ToMe (ICLR23)</td><td>96.9%</td><td>59.5</td><td>62.6</td><td>1727</td><td>86.9</td><td>69.0</td><td>75.9</td><td>55.8</td><td>56.6</td></tr><tr><td>VisionZip (CVPR25)</td><td>96.8%</td><td>59.2</td><td>62.5</td><td>1749</td><td>85.2</td><td>68.7</td><td>77.2</td><td>55.8</td><td>56.3</td></tr><tr><td>+ ours</td><td>98.0%</td><td>60.6</td><td>63.7</td><td>1782</td><td>86.6</td><td>69.1</td><td>77.0</td><td>54.9</td><td>58.0</td></tr><tr><td>DivPrune (CVPR25)</td><td>96.9%</td><td>58.9</td><td>63.1</td><td>1723</td><td>86.5</td><td>69.0</td><td>76.1</td><td>55.7</td><td>56.8</td></tr><tr><td>+ ours</td><td>98.7%</td><td>60.9</td><td>63.7</td><td>1813</td><td>86.6</td><td>69.1</td><td>77.4</td><td>56.5</td><td>58.2</td></tr><tr><td>VisPruner (ICCV25)</td><td>97.5%</td><td>59.4</td><td>62.5</td><td>1784</td><td>86.0</td><td>68.3</td><td>76.5</td><td>57.7</td><td>56.6</td></tr><tr><td>+ ours</td><td>98.8%</td><td>60.9</td><td>63.3</td><td>1816</td><td>86.1</td><td>69.5</td><td>77.6</td><td>57.0</td><td>58.1</td></tr><tr><td>HoloV (NeurIPS25)</td><td>96.5%</td><td>58.6</td><td>62.6</td><td>1779</td><td>85.0</td><td>67.3</td><td>76.0</td><td>55.8</td><td>56.3</td></tr><tr><td>+ ours</td><td>98.8%</td><td>61.0</td><td>63.7</td><td>1793</td><td>86.6</td><td>69.6</td><td>77.6</td><td>57.2</td><td>58.1</td></tr><tr><td colspan="11">Retain 128 Tokens (22.2%)</td></tr><tr><td rowspan="3">Text-aware</td><td>FastV (ECCV24)</td><td>91.8%</td><td>54.1</td><td>63.1</td><td>1694</td><td>68.3</td><td>69.3</td><td>70.7</td><td>56.3</td><td>54.0</td></tr><tr><td>PDrop (CVPR25)</td><td>84.5%</td><td>51.6</td><td>54.9</td><td>1403</td><td>67.8</td><td>68.8</td><td>65.6</td><td>52.6</td><td>47.0</td></tr><tr><td>SparseVLM (ICML25)</td><td>97.2%</td><td>58.4</td><td>64.4</td><td>1761</td><td>85.0</td><td>68.7</td><td>76.2</td><td>56.6</td><td>56.9</td></tr><tr><td rowspan="9">Text-agnostic</td><td>ToMe (ICLR23)</td><td>95.1%</td><td>58.7</td><td>60.7</td><td>1668</td><td>86.5</td><td>68.4</td><td>74.5</td><td>54.8</td><td>55.1</td></tr><tr><td>VisionZip (CVPR25)</td><td>95.6%</td><td>58.5</td><td>61.4</td><td>1705</td><td>83.2</td><td>68.8</td><td>76.7</td><td>55.6</td><td>55.2</td></tr><tr><td>+ ours</td><td>97.1%</td><td>59.8</td><td>62.7</td><td>1772</td><td>86.0</td><td>68.4</td><td>76.6</td><td>55.0</td><td>57.2</td></tr><tr><td>DivPrune (CVPR25)</td><td>96.3%</td><td>58.6</td><td>63.7</td><td>1702</td><td>86.5</td><td>68.9</td><td>75.2</td><td>55.2</td><td>55.8</td></tr><tr><td>+ ours</td><td>97.9%</td><td>60.6</td><td>63.1</td><td>1795</td><td>86.0</td><td>68.6</td><td>76.8</td><td>56.0</td><td>57.8</td></tr><tr><td>VisPruner (ICCV25)</td><td>96.3%</td><td>58.1</td><td>61.9</td><td>1778</td><td>84.5</td><td>68.8</td><td>75.3</td><td>56.9</td><td>55.0</td></tr><tr><td>+ ours</td><td>98.4%</td><td>60.9</td><td>63.3</td><td>1813</td><td>86.2</td><td>68.7</td><td>77.1</td><td>56.4</td><td>57.9</td></tr><tr><td>HoloV (NeurIPS25)</td><td>95.5%</td><td>57.5</td><td>62.5</td><td>1761</td><td>82.2</td><td>69.0</td><td>75.0</td><td>55.6</td><td>54.7</td></tr><tr><td>+ ours</td><td>98.3%</td><td>60.8</td><td>63.0</td><td>1807</td><td>86.0</td><td>68.7</td><td>77.1</td><td>56.6</td><td>58.0</td></tr><tr><td colspan="11">Retain 64 Tokens (11.1%)</td></tr><tr><td rowspan="3">Text-aware</td><td>FastV (ECCV24)</td><td>75.2%</td><td>46.4</td><td>51.4</td><td>1284</td><td>36.1</td><td>69.9</td><td>56.2</td><td>51.6</td><td>43.9</td></tr><tr><td>PDrop (CVPR25)</td><td>80.9%</td><td>49.0</td><td>55.6</td><td>1404</td><td>55.5</td><td>69.5</td><td>60.1</td><td>50.6</td><td>46.1</td></tr><tr><td>SparseVLM (ICML25)</td><td>90.6%</td><td>53.8</td><td>60.2</td><td>1591</td><td>77.5</td><td>69.7</td><td>70.3</td><td>53.5</td><td>51.2</td></tr><tr><td rowspan="9">Text-agnostic</td><td>ToMe (ICLR23)</td><td>91.6%</td><td>56.0</td><td>57.9</td><td>1588</td><td>84.3</td><td>68.0</td><td>71.7</td><td>52.6</td><td>52.8</td></tr><tr><td>VisionZip (CVPR25)</td><td>93.0%</td><td>56.0</td><td>60.5</td><td>1690</td><td>78.2</td><td>69.5</td><td>75.2</td><td>53.8</td><td>52.8</td></tr><tr><td>+ ours</td><td>94.8%</td><td>58.5</td><td>62.3</td><td>1726</td><td>84.6</td><td>68.0</td><td>74.3</td><td>52.2</td><td>55.2</td></tr><tr><td>DivPrune (CVPR25)</td><td>93.8%</td><td>57.1</td><td>60.2</td><td>1653</td><td>85.3</td><td>68.3</td><td>73.3</td><td>54.5</td><td>53.7</td></tr><tr><td>+ ours</td><td>95.9%</td><td>59.0</td><td>62.2</td><td>1748</td><td>84.8</td><td>68.0</td><td>75.0</td><td>54.4</td><td>56.3</td></tr><tr><td>VisPruner (ICCV25)</td><td>92.8%</td><td>55.8</td><td>60.0</td><td>1670</td><td>80.5</td><td>68.5</td><td>72.3</td><td>55.6</td><td>52.6</td></tr><tr><td>+ ours</td><td>96.0%</td><td>59.2</td><td>61.6</td><td>1722</td><td>85.1</td><td>68.0</td><td>75.4</td><td>55.4</td><td>56.5</td></tr><tr><td>HoloV (NeurIPS25)</td><td>92.2%</td><td>55.1</td><td>60.0</td><td>1699</td><td>76.8</td><td>68.6</td><td>72.3</td><td>54.9</td><td>52.5</td></tr><tr><td>+ ours</td><td>96.5%</td><td>59.0</td><td>61.9</td><td>1787</td><td>84.9</td><td>68.0</td><td>75.6</td><td>55.4</td><td>56.7</td></tr></table>

sparse positional indices induce severe attention decay due to the long-term property of RoPE, causing the model to neglect visual tokens, which is consistent with the analysis of Fig. 2. Applying the attention calibration technique alone (②) also results in a performance drop to 91.0%. The simultaneous deployment of both strategies (row ④) yields a performance gain, confirming their complementary nature in resolving positional and attentional distortions. Comparing the pruning-only results ① with the merging ⑤, we observe that merging improves performance from 92.2% to 93.5%. This confirms that aggregating information into distinctive anchors is more effective than discarding tokens. In contrast, augmenting merging with attention calibration under reindexed positions (row ⑥) slightly lowers the accuracy to 92.9%, as calibrating reindexed sequences amplifies attention to visual tokens excessively, which shares the same mechanism as the drop in row ② (see Fig. 5 in Appendix D). Finally, the full framework in row ⑧ achieves the highest accuracy of 96.5%, demonstrating the synergistic effect of integrating merging with calibration. A notable comparison arises between ⑦ and ⑧. In row ⑦, applying merging with only position rectification results in the lowest performance of 90.1%. This arises because retaining original indices places merged tokens at large relative distances from the query. As a result, RoPE suppresses these semantically rich tokens, preventing the model from attending to the aggregated global context. We address this via attention calibration, leading to a substantial improvement of 6.4%.

Table 2. Comparison with VTR methods on LLaVA-NeXT-7B (Liu et al., 2024a). Best results at each reduction ratio in bold.

<table><tr><td>Method</td><td>Average</td><td>GQA</td><td>POPE</td><td>VQA $^{\text{Text}}$ </td><td>SEED</td></tr><tr><td colspan="6">Using All Visual Tokens, 2880 Tokens (100%)</td></tr><tr><td>LLaVA-Next-7B</td><td>100%</td><td>64.2</td><td>86.5</td><td>64.9</td><td>70.2</td></tr><tr><td colspan="6">Retain 320 Tokens (11.1%)</td></tr><tr><td>Visionzip (CVPR25)</td><td>88.9%</td><td>59.0</td><td>82.7</td><td>55.2</td><td>58.3</td></tr><tr><td>+ ours</td><td>90.5%</td><td>60.6</td><td>87.2</td><td>53.0</td><td>59.9</td></tr><tr><td>DivPrune (CVPR25)</td><td>88.8%</td><td>60.2</td><td>84.0</td><td>51.6</td><td>59.5</td></tr><tr><td>+ ours</td><td>90.2%</td><td>60.7</td><td>87.3</td><td>51.4</td><td>60.5</td></tr><tr><td>VisPruner (ICCV25)</td><td>90.8%</td><td>59.3</td><td>83.2</td><td>59.1</td><td>58.6</td></tr><tr><td>+ ours</td><td>91.9%</td><td>60.9</td><td>87.2</td><td>55.9</td><td>60.2</td></tr><tr><td>HoloV (NeurIPS25)</td><td>87.5%</td><td>59.6</td><td>83.4</td><td>57.1</td><td>51.0</td></tr><tr><td>+ ours</td><td>92.0%</td><td>61.0</td><td>87.4</td><td>56.1</td><td>60.0</td></tr><tr><td colspan="6">Retain 160 Tokens (5.5%)</td></tr><tr><td>Visionzip (CVPR25)</td><td>84.9%</td><td>56.2</td><td>77.6</td><td>54.5</td><td>55.0</td></tr><tr><td>+ ours</td><td>87.9%</td><td>58.5</td><td>85.5</td><td>51.2</td><td>58.0</td></tr><tr><td>DivPrune (CVPR25)</td><td>86.3%</td><td>58.4</td><td>81.3</td><td>51.2</td><td>57.2</td></tr><tr><td>+ ours</td><td>87.2%</td><td>59.0</td><td>86.4</td><td>48.3</td><td>58.0</td></tr><tr><td>VisPruner (ICCV25)</td><td>86.7%</td><td>57.0</td><td>78.6</td><td>57.1</td><td>55.6</td></tr><tr><td>+ ours</td><td>88.5%</td><td>58.8</td><td>85.9</td><td>52.0</td><td>58.2</td></tr><tr><td>HoloV (NeurIPS25)</td><td>86.3%</td><td>57.2</td><td>78.1</td><td>56.0</td><td>55.9</td></tr><tr><td>+ ours</td><td>89.0%</td><td>59.2</td><td>86.1</td><td>53.1</td><td>57.8</td></tr></table>

Table 3. Comparison with VTR methods on Qwen2.5-VL-7B-Instruct (Bai et al., 2025). Best results at each reduction ratio in bold.

<table><tr><td>Method</td><td>Average</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td>TextVQA</td></tr><tr><td colspan="7">Using All Visual Tokens</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>100%</td><td>84.4</td><td>2323</td><td>86.7</td><td>77.8</td><td>77.7</td></tr><tr><td colspan="7">Retain 33.3% visual tokens</td></tr><tr><td>DivPrune (CVPR 25)</td><td>95.9%</td><td>79.6</td><td>2187</td><td>84.9</td><td>76.2</td><td>73.8</td></tr><tr><td>+ours</td><td>97.2%</td><td>81.1</td><td>2205</td><td>86.2</td><td>78.2</td><td>73.8</td></tr><tr><td>VisPruner (ICCV 25)</td><td>95.8%</td><td>80.2</td><td>2175</td><td>84.6</td><td>76.7</td><td>73.2</td></tr><tr><td>+ours</td><td>97.4%</td><td>82.0</td><td>2211</td><td>86.0</td><td>78.6</td><td>73.3</td></tr><tr><td>HoloV (NeurIPS 25)</td><td>93.6%</td><td>78.2</td><td>2121</td><td>83.6</td><td>75.9</td><td>70.0</td></tr><tr><td>+ours</td><td>96.4%</td><td>81.0</td><td>2193</td><td>85.7</td><td>78.1</td><td>71.9</td></tr><tr><td colspan="7">Retain 22.2% visual tokens</td></tr><tr><td>DivPrune (CVPR 25)</td><td>93.9%</td><td>77.6</td><td>2137</td><td>84.1</td><td>75.0</td><td>71.7</td></tr><tr><td>+ours</td><td>95.2%</td><td>79.8</td><td>2166</td><td>84.8</td><td>77.3</td><td>70.6</td></tr><tr><td>VisPruner (ICCV 25)</td><td>88.6%</td><td>78.0</td><td>1604</td><td>82.4</td><td>75.7</td><td>69.2</td></tr><tr><td>+ours</td><td>94.8%</td><td>79.0</td><td>2137</td><td>84.6</td><td>78.4</td><td>69.8</td></tr><tr><td>HoloV (NeurIPS 25)</td><td>90.6%</td><td>76.5</td><td>2031</td><td>81.9</td><td>74.4</td><td>65.9</td></tr><tr><td>+ours</td><td>93.9%</td><td>79.3</td><td>2166</td><td>84.4</td><td>76.6</td><td>67.4</td></tr><tr><td colspan="7">Retain 11.1% visual tokens</td></tr><tr><td>DivPrune (CVPR 25)</td><td>87.5%</td><td>71.1</td><td>1961</td><td>79.7</td><td>72.7</td><td>64.9</td></tr><tr><td>+ours</td><td>89.8%</td><td>74.5</td><td>1998</td><td>82.3</td><td>76.3</td><td>63.6</td></tr><tr><td>VisPruner (ICCV 25)</td><td>86.9%</td><td>69.8</td><td>2137</td><td>77.0</td><td>73.1</td><td>59.8</td></tr><tr><td>+ours</td><td>88.5%</td><td>74.4</td><td>1980</td><td>80.8</td><td>77.1</td><td>59.6</td></tr><tr><td>HoloV (NeurIPS 25)</td><td>83.5%</td><td>72.3</td><td>1834</td><td>77.6</td><td>70.8</td><td>56.4</td></tr><tr><td>+ours</td><td>88.4%</td><td>76.3</td><td>1992</td><td>81.3</td><td>75.4</td><td>58.3</td></tr></table>

Impact of calibration terms. We show in Table 5 the contribution of each term in our attention calibration. The baseline (w/o Attention Calibration) denotes that original position indices are retained but no attention calibration is applied to Eq. 4. Incorporating the merged group size term $s _ { n }$ yields an improvement across all methods, recovering the attention weights proportional to the number of merged tokens. The best performance is consistently achieved when the distance-aware term is weighted. This confirms that scaling by merged group size is insufficient, demonstrating that it is crucial to compensate for the positional bias to recover lost attention weights for visual tokens.

Effectiveness of distinctive anchor selection. Table 6 compares different reduction types: pruning only, hybrid reduction with uniformly distributed anchor selection, and hybrid reduction with our distinctive merging. While the hybrid approach generally outperforms pruning-only methods, uniformly distributed anchor selection shows suboptimal performance. For instance, in the case of DivPrune, employing uniform anchor selection degrades performance compared to pruning $( 9 5 . 8 \%  9 4 . 9 \% )$ , suggesting that aggregating tokens into unrepresentative anchors causes information distortion. In contrast, our distinctive anchor selection strategy consistently outperforms the pruning-only counterpart across all baselines.

Table 4. Ablation study on the components of our framework.

<table><tr><td></td><td>Incorporating Merging</td><td>Retaining Position</td><td>Attention Calibration</td><td>Average</td></tr><tr><td>1</td><td>X</td><td>X</td><td>X</td><td>92.2%</td></tr><tr><td>2</td><td>X</td><td>X</td><td>√</td><td>91.0%</td></tr><tr><td>3</td><td>X</td><td>√</td><td>X</td><td>90.6%</td></tr><tr><td>4</td><td>X</td><td>√</td><td>√</td><td>93.1%</td></tr><tr><td>5</td><td>√</td><td>X</td><td>X</td><td>93.5%</td></tr><tr><td>6</td><td>√</td><td>X</td><td>√</td><td>92.9%</td></tr><tr><td>7</td><td>√</td><td>√</td><td>X</td><td>90.1%</td></tr><tr><td>8</td><td>√</td><td>√</td><td>√</td><td>96.5%</td></tr></table>

Table 5. Comparison with attention calibration strategies.

<table><tr><td>Method</td><td>w/o Attention Calibration</td><td> $s_n$ </td><td> $s_n(c - \mathcal{D}(|m-n|))$ </td></tr><tr><td>VisionZip</td><td>89.5%</td><td>94.4%</td><td>94.8%</td></tr><tr><td>DivPrune</td><td>88.9%</td><td>94.9%</td><td>95.9%</td></tr><tr><td>VisPruner</td><td>88.7%</td><td>95.1%</td><td>96.0%</td></tr><tr><td>HoloV</td><td>90.1%</td><td>95.3%</td><td>96.5%</td></tr></table>

Position of merged tokens. Since a merged token serves as a representative for multiple tokens, assigning an appropriate positional index to this aggregate token is critical for preserving spatial context. We investigate four strategies for defining the position of the merged token: namely First, Median, Mean, and Last. These correspond to assigning the index of the earliest, central, arithmetic mean, and latest token within the group, respectively. As shown in Table 7, the ’Median’ yields superior accuracy (96.5%). We conjecture that the median position offers the most accurate spatial approximation of the group’s center, minimizing spatial distortion. Conversely, the ’Last’ strategy results in the lowest accuracy (95.2%). This is because our attention calibration assigns a smaller boosting weight to positions closer to the text query (i.e., the ’Last’ position), which may result in insufficient attention to the merged token.

Efficiency analysis. The primary source of additional computational overhead in our framework arises from the anchor selection process. Specifically, computing the selfcorrelation matrix involves matrix multiplication with a complexity of $O ( N _ { v i s } ^ { 2 } d )$ . However, this overhead is negligible compared to the total computational cost of MLLMs. As shown in Table 8, the anchor selection requires a constant 1.359 GFLOPs for LLaVA-1.5-7B (Liu et al., 2024a). Even with the most aggressive reduction to 64 tokens, it constitutes only 0.136% of the total inference FLOPs. The attention calibration is also computationally efficient in that the calibration requires an N × N matrix. The matrix is calculated once prior to the LLM input stage, and then applied to each layer, incurring minimal additional cost. We also show in Fig. 4 the trade-off between accuracy and latency. The results demonstrate that our method achieves a superior trade-off compared to the baseline. While incurring only a marginal increase in latency, it achieves significant accuracy gains. To further support our efficiency, we measure the endto-end latency, including both the total inference time and the prefill time, on the GQA dataset using an 8× NVIDIA A5000 GPU setup. As shown in Table 9, integrating our framework introduces only a marginal latency overhead over the baseline HoloV. For instance, at a retention of 64 tokens, it adds merely 4 seconds to the total inference time and 5.5ms to the prefill time. Despite this negligible cost, our framework narrows the accuracy gap to the full-token model, reducing the degradation of HoloV from 10.99% to 4.68% at 64 tokens. These results support that our method preserves the practical speedup of VTR while delivering significant accuracy gains.

Table 6. Comparison with reduction types. P denotes pruning and M denotes merging.

<table><tr><td>Method</td><td>P</td><td>P + M(uniform)</td><td>P + M(distinctive)</td></tr><tr><td>VisionZip</td><td>-</td><td>94.4%</td><td>94.8%</td></tr><tr><td>DivPrune</td><td>95.8%</td><td>94.9%</td><td>95.9%</td></tr><tr><td>VisPruner</td><td>94.2%</td><td>95.2%</td><td>96.0%</td></tr><tr><td>HoloV</td><td>93.1%</td><td>94.8%</td><td>96.5%</td></tr></table>

Table 7. Comparison with the position of merged tokens.

<table><tr><td>Position Type</td><td>Average</td></tr><tr><td>First</td><td>96.3%</td></tr><tr><td>Median</td><td>96.5%</td></tr><tr><td>Mean</td><td>96.0%</td></tr><tr><td>Last</td><td>95.2%</td></tr></table>

Table 8. Computational overhead of distinctive anchor token selection.

<table><tr><td> $n_{vis}$ </td><td>Overhead Ratio</td><td>Anchor Selection (FLOPs)</td><td>MLLMs Calculation (FLOPs)</td></tr><tr><td>192</td><td>0.074%</td><td rowspan="3">1.359 G</td><td>1.839 T</td></tr><tr><td>128</td><td>0.096%</td><td>1.417 T</td></tr><tr><td>64</td><td>0.136%</td><td>0.997 T</td></tr></table>

Table 9. End-to-end latency analysis on the GQA (Hudson & Manning, 2019) dataset.

<table><tr><td rowspan="2">Method</td><td colspan="2">Total Inference</td><td colspan="2">Prefill</td><td colspan="2">Accuracy</td></tr><tr><td>Time</td><td>Speedup</td><td>Time</td><td>Speedup</td><td>Acc.</td><td> $\Delta$ </td></tr><tr><td>LLaVA-1.5-7B</td><td>6m05s</td><td>1.00×</td><td>227.8ms</td><td>1.00×</td><td>61.9</td><td>-</td></tr><tr><td>HoloV (192 tokens)</td><td>4m23s</td><td>1.39×</td><td>178.2ms</td><td>1.28×</td><td>58.6</td><td>-5.33%</td></tr><tr><td>+ ours</td><td>4m28s</td><td>1.36×</td><td>181.4ms</td><td>1.26×</td><td>61.0</td><td>-1.45%</td></tr><tr><td>HoloV (128 tokens)</td><td>3m58s</td><td>1.53×</td><td>158.2ms</td><td>1.44×</td><td>57.5</td><td>-7.11%</td></tr><tr><td>+ ours</td><td>4m02s</td><td>1.51×</td><td>164.0ms</td><td>1.39×</td><td>60.8</td><td>-1.78%</td></tr><tr><td>HoloV (64 tokens)</td><td>3m39s</td><td>1.67×</td><td>145.6ms</td><td>1.56×</td><td>55.1</td><td>-10.99%</td></tr><tr><td>+ ours</td><td>3m43s</td><td>1.64×</td><td>151.1ms</td><td>1.51×</td><td>59.0</td><td>-4.68%</td></tr></table>

## 5. Conclusion

In this paper, we have proposed RESTORE, a novel framework to improve existing VTR methods for efficient MLLM inference. We have identified a critical yet overlooked issue in previous methods, attentional distortion caused by position assignment, and addressed it through an attention calibration mechanism. We have also introduced a distinctive anchor selection for token merging to mitigate information loss during token merging. Extensive experiments demonstrate that integrating our framework with various baselines consistently yields state-of-the-art performance across multiple benchmarks, maintaining efficiency.

![](images/67f3c884976c856498fe1cd4aa94d80c062cc8f4df54a69b0b007600791d3fc1.jpg)

<details>
<summary>line chart</summary>

| Latency (ms) | LLaVA-1.5-7B | HoioV | HoioV + Ours |
| ------------ | ------------ | ----- | ------------ |
| 160          | 62           | 55    | 59           |
| 180          | 62           | 58    | 61           |
| 200          | 62           | 59    | 61           |
| 220          | 62           | 60    | 61           |
</details>

(a) GQA.

![](images/7385058922b23c9f0c6e2449448a8b5201d645b2f7da302a492e2b2bc72dab3a.jpg)

<details>
<summary>line chart</summary>

| Latency (ms) | LLaVA-1.5-7B | HoloV | HoloV + Ours |
| ------------ | ------------ | ----- | ------------ |
| 150          | 64.5         | 60.0  | 61.5         |
| 160          | 64.5         | 62.5  | 63.0         |
| 180          | 64.5         | 62.5  | 63.5         |
| 200          | 64.5         | 62.5  | 63.5         |
| 220          | 64.5         | 62.5  | 63.5         |
</details>

(b) MMBench.  
Figure 4. Accuracy-latency trade-off curves on (a) GQA (Hudson & Manning, 2019) and (b) MMBench (Liu et al., 2024b) by varying the number of retained visual tokens.

## Impact Statement

Our method focuses on improving the computational efficiency of MLLMs. This direction has the potential to reduce energy consumption during inference and facilitate deployment in resource-constrained environments (e.g., mobile phones). We do not predict any negative societal consequences or ethical issues specific to this work.

## Acknowledgement

This work was partly supported by Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No.RS-2022-00143524, Development of Fundamental Technology and Integrated Solution for Next-Generation Automatic Artificial Intelligence System, No.RS-2025-09942968, AI Semiconductor Innovation Lab (Yonsei University)), the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (RS-2025-02216328), and the KIST Institutional Program (Project No.2E33001-24-086).

## References

Alvar, S. R., Singh, G., Akbari, M., and Zhang, Y. Divprune: Diversity-based visual token pruning for large multimodal models. In CVPR, 2025.  
Bai, J., Bai, S., Chu, Y., Cui, Z., Dang, K., Deng, X., Fan, Y., Ge, W., Han, Y., Huang, F., et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.  
Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.  
Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., and Hoffman, J. Token merging: Your vit but faster. In ICLR, 2023.  
Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J. D., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al. Language models are few-shot learners. NeurIPS, 2020.  
Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., and Chang, B. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. In ECCV, 2024.  
Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. In NeurIPS, 2025.  
Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017.  
Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In CVPR, 2019.  
Huttenlocher, D. P., Klanderman, G. A., and Rucklidge, W. J. Comparing images using the hausdorff distance. IEEE Transactions on pattern analysis and machine intelligence, 15(9):850–863, 2002.  
Li, B., Ge, Y., Ge, Y., Wang, G., Wang, R., Zhang, R., and Shan, Y. Seed-bench: Benchmarking multimodal large language models. In CVPR, 2024.  
Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, X., and Wen, J.-R. Evaluating object hallucination in large vision-language models. In EMNLP, 2023.  
Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-llava: Learning united visual representation by alignment before projection. In EMNLP, 2024.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. In NeurIPS, 2023.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In CVPR, 2024a.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? In ECCV, 2024b.

Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X.-C., Liu, C.-L., Jin, L., and Bai, X. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024c.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. NeurIPS, 2022.

Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I., et al. Language models are unsupervised multitask learners. OpenAI blog, 2019.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In ICML, 2021.

Rodriguez, A. and Laio, A. Clustering by fast search and find of density peaks. science, 344(6191):1492–1496, 2014.

Shang, Y., Cai, M., Xu, B., Lee, Y. J., and Yan, Y. Llavaprumerge: Adaptive token reduction for efficient large multimodal models. In ICCV, 2025.

Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In CVPR, 2019.

Su, J., Ahmed, M., Lu, Y., Pan, S., Bo, W., and Liu, Y. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Tong, J., Jin, W., Qin, P., Li, A., Zou, Y., Li, Y., Li, Y., and Li, R. Flowcut: Rethinking redundancy via information flow for efficient vision-language models. In NeurIPS, 2025.

Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.-A., Lacroix, T., Roziere, B., Goyal, N., Hambro, E., \` Azhar, F., et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. In NeurIPS, 2017.

Wen, Z., Gao, Y., Wang, S., Zhang, J., Zhang, Q., Li, W., He, C., and Zhang, L. Stop looking for important tokens in multimodal language models: Duplication matters more. In EMNLP, 2025.  
Xing, L., Huang, Q., Dong, X., Lu, J., Zhang, P., Zang, Y., Cao, Y., He, C., Wang, J., Wu, F., et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247, 2024.  
Yang, S., Chen, Y., Tian, Z., Wang, C., Li, J., Yu, B., and Jia, J. Visionzip: Longer is better but not necessary in vision language models. In CVPR, 2025.  
Ye, W., Wu, Q., Lin, W., and Zhou, Y. Fit and prune: Fast and training-free visual token pruning for multi-modal large language models. In AAAI, 2025.  
Zhang, Q., Cheng, A., Lu, M., Zhang, R., Zhuo, Z., Cao, J., Guo, S., She, Q., and Zhang, S. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. In ICCV, 2025a.  
Zhang, Y., Fan, C.-K., Ma, J., Zheng, W., Huang, T., Cheng, K., Gudovskiy, D. A., Okuno, T., Nakata, Y., Keutzer, K., et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In ICML, 2025b.  
Zou, X., Lu, D., Wang, Y., Yan, Y., Lyu, Y., Zheng, X., Zhang, L., and Hu, X. Don’t just chase “highlighted tokens” in mllms: Revisiting visual holistic context retention. In NeurIPS, 2025.

## Appendix

## A. Attention calibration for Qwen2.5-VL with M-RoPE

In standard 1D RoPE, our decay function D (Eq. 3) averages cosine terms across all frequency dimensions. In M-RoPE, the half-head dimension $d _ { h } / 2$ is partitioned into $G$ independent axes $( \mathrm { e . g . }$ ., temporal, height, width). Each axis $g \in \{ 1 , \ldots , G \}$ is allocated $d _ { g }$ dimensions and has its own positional distance $\Delta _ { g } .$ . To extend our attention calibration to $\mathbf { M } { \cdot } \mathbf { R o } \mathrm { P E }$ , we generalize the standard 1D RoPE formulation into a weighted sum of per-axis decay functions:

$$
\mathcal {D} _ {M - R o P E} = \sum_ {g = 1} ^ {G} w _ {g} \mathcal {D} _ {g} (\Delta_ {g}), \tag {9}
$$

where $\begin{array} { r } { \mathcal { D } _ { g } ( \Delta _ { g } ) = \frac { 1 } { | \Omega _ { g } | } \sum _ { j \in \Omega _ { g } } \cos ( \Delta _ { g } \cdot \theta _ { j } ) } \end{array}$ computes the specific decay for axis $g .$ Here, $\Omega _ { g }$ represents the set of frequency indices assigned to axis $g \left( \left| \Omega _ { g } \right| = d _ { g } \right)$ , and $\theta _ { j }$ is the rotation frequency for index $j .$ The weight $\begin{array} { r } { w _ { g } = \frac { 2 d _ { g } } { d _ { h } } } \end{array}$ 2dgd denotes the $^ { g , }$ $\textstyle \sum _ { g = 1 } ^ { G } w _ { g } = 1$ RoPE. This formulation preserves our fundamental principle: attention inherently decays as positional distance increases, requiring a proportional calibration bias in the VTR scenario.

## B. Theoretical Analysis of Textual Attention Amplification

In this section, we provide a theoretical analysis to elucidate the disproportionate amplification of textual attention described in Fig. 1(d). Specifically, we mathematically demonstrate how pruning visual tokens inadvertently exacerbates the initial attention disparity caused by positional bias, leading the model to over-attend to text tokens.

Let $S$ be the original set of all tokens $( | S | = N _ { s y s } + N _ { v i s } + N _ { t x t } )$ and $P$ be the set of pruned visual tokens. For any token i, let $z _ { i }$ be its attention logit. The original attention weight is defined as:

$$
a _ {i} = \frac {\exp (z _ {i})}{Z}, \quad \text { where } \quad Z = \sum_ {j \in S} \exp (z _ {j}). \tag {10}
$$

After pruning the visual tokens from $N _ { v i s }$ to $n _ { v i s }$ , the partition function reduces to $\begin{array} { r } { Z ^ { \prime } = Z - \sum _ { k \in P } \exp ( z _ { k } ) } \end{array}$ . The new attention weight for a remaining token $i \in S \setminus P$ is given by:

$$
a _ {i} ^ {\prime} = \frac {\exp (z _ {i})}{Z ^ {\prime}}. \tag {11}
$$

Consequently, the amplification in the attention weight for token i can be expressed as:

$$
\Delta a _ {i} = a _ {i} ^ {\prime} - a _ {i} = \exp (z _ {i}) \left(\frac {1}{Z ^ {\prime}} - \frac {1}{Z}\right) = a _ {i} \left(\frac {Z - Z ^ {\prime}}{Z ^ {\prime}}\right). \tag {12}
$$

Since $Z > Z ^ { \prime }$ , the term $\frac { Z - Z ^ { \prime } } { Z ^ { \prime } }$ is a positive constant across all remaining tokens. This derivation demonstrates that the redistributed probability mass from the pruned tokens is not shared uniformly. Instead, the absolute increase $\Delta a _ { i }$ is directly proportional to its original weight $a _ { i } .$ . That is, tokens that originally possessed larger attention weights absorb a larger portion of the redistributed weights.

## C. More Results

Text-aware baselines. In the main paper, we integrate our framework with text-agnostic VTR methods, as they avoid the computational overhead of computing attention weights within the LLM layers and are thus more efficient. Nevertheless, our framework is not inherently restricted to text-agnostic methods. It is equally feasible for text-aware approaches. This is because our attention calibration addresses the positional and attentional distortions that arise in any reduced token sequence, irrespective of how the retained tokens are selected, while our distinctive anchor selection operates on the merging stage that is orthogonal to the underlying pruning criterion. To verify this, we integrate our framework with three representative text-aware methods, including FastV (Chen et al., 2024), PDrop (Xing et al., 2024), and SparseVLM (Zhang et al., 2025b).

Table 10. Comparison with text-aware VTR methods for LLaVA-1.5-7B (Liu et al., 2024a) on eight benchmarks and the average score across them. For baselines, all experiment results are re-implemented from their official codebases under the same environments. Best results at each reduction ratio in Bold.

<table><tr><td>Method</td><td>Average</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td> $SQA^{IMG}$ </td><td> $VQA^{V2}$ </td><td> $VQA^{Text}$ </td><td>SEED</td></tr><tr><td colspan="10">Using All Visual Tokens, 576 Tokens (100%)</td></tr><tr><td>LLaVA-1.5-7B</td><td>100%</td><td>61.9</td><td>64.6</td><td>1862</td><td>85.9</td><td>69.5</td><td>78.5</td><td>58.2</td><td>58.6</td></tr><tr><td colspan="10">Retain 192 Tokens (33.3%)</td></tr><tr><td>FastV (ECCV24)</td><td>96.0%</td><td>57.1</td><td>64.4</td><td>1821</td><td>75.8</td><td>68.9</td><td>74.7</td><td>57.8</td><td>56.3</td></tr><tr><td>+ ours</td><td>99.0%</td><td>60.8</td><td>64.0</td><td>1853</td><td>86.0</td><td>68.4</td><td>77.4</td><td>57.4</td><td>58.1</td></tr><tr><td>PDrop (CVPR25)</td><td>96.8%</td><td>58.0</td><td>62.9</td><td>1790</td><td>84.0</td><td>68.9</td><td>76.0</td><td>57.1</td><td>55.8</td></tr><tr><td>+ ours</td><td>99.1%</td><td>61.0</td><td>64.1</td><td>1829</td><td>86.1</td><td>68.8</td><td>78.0</td><td>57.4</td><td>58.4</td></tr><tr><td>SparseVLM (ICML25)</td><td>98.1%</td><td>59.5</td><td>64.2</td><td>1782</td><td>85.4</td><td>68.7</td><td>77.0</td><td>57.7</td><td>57.3</td></tr><tr><td>+ ours</td><td>99.4%</td><td>61.4</td><td>64.1</td><td>1831</td><td>86.5</td><td>69.1</td><td>78.1</td><td>57.6</td><td>58.5</td></tr><tr><td colspan="10">Retain 128 Tokens (22.2%)</td></tr><tr><td>FastV (ECCV24)</td><td>91.8%</td><td>54.1</td><td>63.1</td><td>1694</td><td>68.3</td><td>69.3</td><td>70.7</td><td>56.3</td><td>54.0</td></tr><tr><td>+ ours</td><td>97.9%</td><td>59.7</td><td>63.6</td><td>1818</td><td>85.4</td><td>68.3</td><td>76.3</td><td>56.9</td><td>57.5</td></tr><tr><td>PDrop (CVPR25)</td><td>84.5%</td><td>51.6</td><td>54.9</td><td>1403</td><td>67.8</td><td>68.8</td><td>65.6</td><td>52.6</td><td>47.0</td></tr><tr><td>+ ours</td><td>95.3%</td><td>57.8</td><td>63.3</td><td>1721</td><td>84.2</td><td>68.0</td><td>74.6</td><td>53.2</td><td>56.4</td></tr><tr><td>SparseVLM (ICML25)</td><td>97.2%</td><td>58.4</td><td>64.4</td><td>1761</td><td>85.0</td><td>68.7</td><td>76.2</td><td>56.6</td><td>56.9</td></tr><tr><td>+ ours</td><td>98.5%</td><td>60.9</td><td>63.0</td><td>1806</td><td>86.7</td><td>68.5</td><td>77.7</td><td>56.3</td><td>58.4</td></tr><tr><td colspan="10">Retain 64 Tokens (11.1%)</td></tr><tr><td>FastV (ECCV24)</td><td>75.2%</td><td>46.4</td><td>51.4</td><td>1284</td><td>36.1</td><td>69.9</td><td>56.2</td><td>51.6</td><td>43.9</td></tr><tr><td>+ ours</td><td>89.9%</td><td>54.1</td><td>61.2</td><td>1621</td><td>75.9</td><td>66.4</td><td>69.0</td><td>51.5</td><td>52.8</td></tr><tr><td>PDrop (CVPR25)</td><td>80.9%</td><td>49.0</td><td>55.6</td><td>1404</td><td>55.5</td><td>69.5</td><td>60.1</td><td>50.6</td><td>46.1</td></tr><tr><td>+ ours</td><td>91.7%</td><td>54.9</td><td>61.6</td><td>1674</td><td>78.2</td><td>67.6</td><td>71.1</td><td>52.3</td><td>53.5</td></tr><tr><td>SparseVLM (ICML25)</td><td>90.6%</td><td>53.8</td><td>60.2</td><td>1591</td><td>77.5</td><td>69.7</td><td>70.3</td><td>53.5</td><td>51.2</td></tr><tr><td>+ ours</td><td>94.5%</td><td>57.1</td><td>61.9</td><td>1721</td><td>83.4</td><td>68.6</td><td>73.6</td><td>53.4</td><td>55.3</td></tr></table>

As shown in Table 10, our method consistently improves the accuracy of all text-aware baselines across every reduction ratio. The improvement becomes more pronounced under aggressive reduction. For instance, at a retention of 64 tokens, our framework improves FastV by 14.7% and PDrop by 10.8%. These results demonstrate that our framework is broadly compatible with diverse VTR strategies, rectifying the distortions overlooked by both text-aware and text-agnostic methods.

Experiment on fine-grained visual perception task. To examine whether our framework remains effective on tasks that demand fine-grained visual perception, we evaluate it on the OCRBench (Liu et al., 2024c) dataset, which requires precise recognition of text within images. Unlike coarse-grained recognition, OCR is highly sensitive to the loss of local visual details, making it a challenging testbed for VTR methods. We integrate our framework with four text-agnostic baselines, VisionZip, DivPrune, VisPruner, and HoloV, under three retention ratios $( n _ { v i s } = 1 9 2$ , 128, and 64). As shown in Table 11, our framework consistently improves the OCRBench score across nearly all baselines and retention ratios. The gains are substantial for several baselines; for instance, our framework improves DivPrune by 24 points at $n _ { v i s } = 1 2 8$ and VisionZip by 11 points at $n _ { v i s } = 6 4$ . The only minor exception arises for VisPruner under the most aggressive setting $( n _ { v i s } = 6 4 )$ , where the extreme compression leaves too few tokens to preserve the fine-grained textual cues. Overall, these results indicate that rectifying positional and attentional distortions helps the model retain fine-grained visual information, demonstrating that our framework is effective even on perception-intensive tasks.

Table 11. Comparison with VTR methods on the OCRBench (Liu et al., 2024c) dataset for LLaVA-1.5-7B (Liu et al., 2024a), evaluating fine-grained visual perception. Best results at each retention ratio in Bold.

<table><tr><td>Retained Tokens</td><td>Method</td><td>VisionZip</td><td>DivPrune</td><td>VisPruner</td><td>HoloV</td></tr><tr><td rowspan="2"> $n_{vis} = 192$ </td><td>Vanilla</td><td>286</td><td>281</td><td>295</td><td>295</td></tr><tr><td>+ ours</td><td>290</td><td>293</td><td>298</td><td>295</td></tr><tr><td rowspan="2"> $n_{vis} = 128$ </td><td>Vanilla</td><td>285</td><td>264</td><td>290</td><td>288</td></tr><tr><td>+ ours</td><td>287</td><td>288</td><td>294</td><td>301</td></tr><tr><td rowspan="2"> $n_{vis} = 64$ </td><td>Vanilla</td><td>258</td><td>257</td><td>287</td><td>279</td></tr><tr><td>+ ours</td><td>269</td><td>267</td><td>273</td><td>283</td></tr></table>

Table 12. Ablation on the pruning ratio γ on TextVQA (Singh et al., 2019) at $n _ { v i s } = 1 9 2$ for LLaVA-1.5-7B (Liu et al., 2024a). $\gamma = 0$ corresponds to merging only and $\gamma = 1$ to pruning only.

<table><tr><td>Method</td><td>Baseline</td><td> $\gamma = 0$ </td><td> $\gamma = 0.25$ </td><td> $\gamma = 0.5$ </td><td> $\gamma = 0.75$ </td><td> $\gamma = 1$ </td></tr><tr><td>VisionZip</td><td>55.8</td><td>53.6</td><td>54.6</td><td>54.9</td><td>54.7</td><td>55.8</td></tr><tr><td>DivPrune</td><td>55.7</td><td>56.8</td><td>56.5</td><td>56.5</td><td>56.5</td><td>56.8</td></tr><tr><td>VisPruner</td><td>57.7</td><td>56.8</td><td>56.9</td><td>57.0</td><td>57.0</td><td>57.7</td></tr><tr><td>HoloV</td><td>55.8</td><td>56.8</td><td>56.9</td><td>57.2</td><td>57.3</td><td>57.2</td></tr></table>

Table 13. Compatibility of our framework with the layer-wise, adaptive pruning method FlowCut (Tong et al., 2025) on LLaVA-1.5-7B (Liu et al., 2024a). We report results at two retention ratios with three pruning ratios γ. Best results at each retention ratio in Bold.

<table><tr><td>Method</td><td>Average</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td> $\text{SQA}^{\text{IMG}}$ </td><td> $\text{VQA}^{\text{V2}}$ </td><td> $\text{VQA}^{\text{Text}}$ </td><td>SEED</td></tr><tr><td colspan="10">Using All Visual Tokens, 576 Tokens (100%)</td></tr><tr><td>LLaVA-1.5-7B</td><td>100%</td><td>61.9</td><td>64.6</td><td>1862</td><td>85.9</td><td>69.5</td><td>78.5</td><td>58.2</td><td>58.6</td></tr><tr><td colspan="10">Retain 128 Tokens (22.2%)</td></tr><tr><td>FlowCut</td><td>97.0%</td><td>58.5</td><td>62.1</td><td>1792</td><td>85.2</td><td>68.6</td><td>76.0</td><td>57.3</td><td>56.2</td></tr><tr><td>+ ours ( $\gamma = 0.5$ )</td><td>97.2%</td><td>60.6</td><td>63.3</td><td>1736</td><td>86.3</td><td>66.9</td><td>77.3</td><td>55.7</td><td>57.3</td></tr><tr><td>+ ours ( $\gamma = 0.75$ )</td><td>97.7%</td><td>60.4</td><td>63.2</td><td>1764</td><td>86.3</td><td>68.3</td><td>77.4</td><td>55.8</td><td>57.5</td></tr><tr><td>+ ours ( $\gamma = 0.9$ )</td><td>97.5%</td><td>59.8</td><td>62.5</td><td>1819</td><td>85.3</td><td>68.3</td><td>77.2</td><td>55.6</td><td>57.3</td></tr><tr><td colspan="10">Retain 64 Tokens (11.1%)</td></tr><tr><td>FlowCut</td><td>93.7%</td><td>55.6</td><td>60.8</td><td>1744</td><td>80.2</td><td>69.1</td><td>72.8</td><td>55.6</td><td>53.5</td></tr><tr><td>+ ours ( $\gamma = 0.5$ )</td><td>95.3%</td><td>58.9</td><td>62.2</td><td>1690</td><td>84.8</td><td>67.9</td><td>75.6</td><td>53.3</td><td>56.2</td></tr><tr><td>+ ours ( $\gamma = 0.75$ )</td><td>95.7%</td><td>58.7</td><td>63.1</td><td>1749</td><td>83.8</td><td>68.0</td><td>75.5</td><td>53.9</td><td>55.5</td></tr><tr><td>+ ours ( $\gamma = 0.9$ )</td><td>94.2%</td><td>56.9</td><td>61.4</td><td>1746</td><td>81.9</td><td>67.8</td><td>74.3</td><td>54.0</td><td>54.4</td></tr></table>

Analysis of performance regression on TextVQA. While our framework consistently improves the overall average accuracy, we observe that it occasionally underperforms the baseline on TextVQA (Singh et al., 2019): for example, when integrated with VisionZip at a retention of 192 tokens (Table 1) and with LLaVA-NeXT-7B at a retention of 160 tokens (Table 2). We investigate the cause of these regressions and find that they do not originate from the attention calibration over-suppressing text-relevant tokens. Instead, they are attributed to the feature averaging in the token merging stage. TextVQA requires reading fine-grained, highly localized text within images, and such information is typically concentrated in a small number of visual tokens. When these tokens are merged, averaging their features with surrounding tokens dilutes the localized high-frequency details that are essential for accurate text recognition.

To verify this, we conduct an ablation on the pruning ratio γ while keeping the attention calibration enabled, evaluated on TextVQA at $n _ { v i s } = 1 9 2$ . Since γ governs the token budget allocated to pruning, a larger γ retains more tokens intact and merges fewer of them. As shown in Table 12, increasing γ consistently alleviates the regression, and bypassing merging entirely $( \gamma = 1 )$ recovers the performance to, or above, the baseline across all methods. As the attention calibration remains fixed throughout this ablation, the result confirms that the regression stems from the merging stage rather than from the calibration. We therefore note that, for tasks demanding fine-grained visual perception, the hybrid reduction strategy would benefit from adaptively suppressing merging to preserve critical localized information, which we leave as a promising direction for future work.

Compatibility with layer-wise and adaptive pruning methods such as FlowCut. Our framework is readily compatible with layer-wise, adaptive pruning methods such as FlowCut (Tong et al., 2025), which estimate token importance using multiple signals across layers. Although FlowCut aggregates multi-faceted criteria, the resulting importance ultimately reduces to a single scalar score per token at a given layer. Therefore, at any intermediate pruning stage i that retains $n _ { v i s , i }$ tokens, we can directly apply our hybrid reduction strategy. Specifically, we retain the top $\gamma \cdot n _ { v i s , i }$ tokens by pruning based on the FlowCut scores, and construct the remaining $( 1 - \gamma ) \cdot n _ { v i s , i }$ tokens by merging with our distinctive anchor selection, followed by attention calibration.

Table 13 reports the results of FlowCut and its integration with our framework on eight benchmarks at two retention ratios $( n _ { v i s } = 1 2 8$ and 64). Since FlowCut performs pruning over multiple stages, iterative merging may accumulate feature distortion; we therefore additionally examine higher pruning ratios $( \gamma = 0 . 7 5 \mathrm { a n d } 0 . 9 )$ that allocate a larger token budget to pruning. Our framework consistently improves the performance of FlowCut across all settings. Notably, $\gamma = 0 . 7 5$ achieves the best overall accuracy, indicating that a moderately pruning-oriented allocation is favorable for multi-stage adaptive methods, as it limits the distortion introduced by repeated merging while still benefiting from global context aggregation. These results demonstrate that our calibration and distinctive anchor selection generalize beyond single-stage reduction to layer-wise, adaptive pruning methods.

![](images/814a856e5641445e80f33f4167628815caf192d5039e704c67b32c07b2ec1804.jpg)

<details>
<summary>line chart</summary>

| Layer | The full token sequence | Reindexing position indices | Reindexing position indices and rectifying attention | Retaining position indices |
| --- | --- | --- | --- | --- |
| 0 | 78 | 36 | 85 | 39 |
| 1 | 68 | 32 | 78 | 35 |
| 2 | 28 | 18 | 44 | 10 |
| 3 | 12 | 10 | 14 | 5 |
| 4 | 10 | 8 | 12 | 3 |
| 5 | 12 | 10 | 16 | 5 |
| 6 | 14 | 12 | 20 | 8 |
| 7 | 16 | 14 | 24 | 10 |
| 8 | 20 | 16 | 28 | 12 |
| 9 | 24 | 18 | 32 | 14 |
| 10 | 30 | 20 | 36 | 16 |
| 11 | 34 | 22 | 40 | 18 |
| 12 | 30 | 20 | 36 | 16 |
| 13 | 28 | 18 | 32 | 14 |
| 14 | 26 | 16 | 30 | 12 |
| 15 | 24 | 14 | 28 | 10 |
| 16 | 22 | 12 | 26 | 8 |
| 17 | 20 | 10 | 24 | 6 |
| 18 | 18 | 8 | 22 | 4 |
| 19 | 16 | 6 | 20 | 2 |
| 20 | 14 | 4 | 18 | 0 |
| 21 | 12 | 2 | 16 | -2 |
| 22 | 10 | -2 | 14 | -4 |
| 23 | 8 | -4 | 12 | -6 |
| 24 | 6 | -6 | 10 | -8 |
| 25 | 4 | -8 | 8 | -10 |
| 26 | 2 | -10 | 6 | -12 |
| 27 | 0 | -12 | 4 | -14 |
| 28 | -2 | -14 | 2 | -16 |
| 29 | -4 | -16 | -2 | -18 |
| 30 | -6 | -18 | -4 | -20 |
| 31 | -8 | -20 | -6 | -22 |
| 32 | -10 | -22 | -8 | -24 |
| 33 | -12 | -24 | -10 | -26 |
| 34 | -14 | -26 | -12 | -28 |
| 35 | -16 | -28 | -14 | -30 |
| 36 | -18 | -30 | -16 | -32 |
| 37 | -20 | -32 | -18 | -34 |
| 38 | -22 | -34 | -20 | -36 |
| 39 | -24 | -36 | -22 | -38 |
| 40 | -26 | -38 | -24 | -40 |
| 41 | -28 | -40 | -26 | -42 |
| 42 | -30 | -42 | -28 | -44 |
| 43 | -32 | -44 | -30 | -46 |
| 44 | -34 | -46 | -32 | -48 |
| 45 | -36 | -48 | -34 | -50 |
| 46 | -38 | -50 | -36 | -52 |
| 47 | -40 | -52 | -38 | -54 |
| 48 | -42 | -54 | -40 | -56 |
| 49 | -44 | -56 | -42 | -58 |
| 50 | -46 | -58 | -44 | -60 |
| 51 | -48 | -60 | -46 | -62 |
| 52 | -50 | -62 | -48 | -64 |
| 53 | -52 | -64 | -50 | -66 |
| 54 | -54 | -66 | -52 | -68 |
| 55 | -56 | -68 | -54 | -70 |
| 56 | -58 | -70 | -56 | -72 |
| 57 | -60 | -72 | -58 | -74 |
| 58 | -62 | -74 | -60 | -76 |
| 59 | -64 | -76 | -62 | -78 |
| 60 | -66 | -78 | -64 | -80 |
| 61 | -68 | -80 | -66 | -82 |
| 62 | -70 | -82 | -68 | -84 |
| 63 | -72 | -84 | -70 | -86 |
| 64 | -74 | -86 | -72 | -88 |
| 65 | -76 | -88 | -74 | -90 |
| 66 | -78 | -90 | -76 | -92 |
| 67 | -80 | -92 | -78 | -94 |
| 68 | -82 | -94 | -80 | -96 |
| 69 | -84 | -96 | -82 | -98 |
| 70 | -86 | -98 | -84 | -100 |
| 71 | -88 | -100 | -86 | -102 |
| 72 | -90 | -102 | -88 | -104 |
| 73 | -92 | -104 | -90 | -106 |
| 74 | -94 | -106 | -92 | -108 |
| 75 | -96 | -108 | -94 | -110 |
</details>

(a) Visual-to-visual attention.

![](images/f52f7bfaa2a403d8fa95c9f859b6a52031114d3475179ccb580c5db3feaae4d8.jpg)

<details>
<summary>line chart</summary>

| Layer | The full token sequence | Reindexing position indices | Reindexing position indices and rectifying attention | Retaining position indices |
|-------|--------------------------|------------------------------|--------------------------------------------------|----------------------------|
| 0     | 78                       | 36                           | 80                                               | 32                         |
| 1     | 40                       | 10                           | 40                                               | 5                          |
| 2     | 10                       | 5                            | 10                                               | 2                          |
| 3     | 5                        | 3                            | 5                                                | 1                          |
| 4     | 3                        | 2                            | 3                                                | 1                          |
| 5     | 2                        | 1                            | 2                                                | 1                          |
| 6     | 1                        | 1                            | 1                                                | 1                          |
| 7     | 1                        | 1                            | 1                                                | 1                          |
| 8     | 1                        | 1                            | 1                                                | 1                          |
| 9     | 1                        | 1                            | 1                                                | 1                          |
| 10    | 1                        | 1                            | 1                                                | 1                          |
| 11    | 1                        | 1                            | 1                                                | 1                          |
| 12    | 1                        | 1                            | 1                                                | 1                          |
| 13    | 1                        | 1                            | 1                                                | 1                          |
| 14    | 1                        | 1                            | 1                                                | 1                          |
| 15    | 1                        | 1                            | 1                                                | 1                          |
| 16    | 1                        | 1                            | 1                                                | 1                          |
| 17    | 1                        | 1                            | 1                                                | 1                          |
| 18    | 1                        | 1                            | 1                                                | 1                          |
| 19    | 1                        | 1                            | 1                                                | 1                          |
| 20    | 1                        | 1                            | 1                                                | 1                          |
| 21    | 1                        | 1                            | 1                                                | 1                          |
| 22    | 1                        | 1                            | 1                                                | 1                          |
| 23    | 1                        | 1                            | 1                                                | 1                          |
| 24    | 1                        | 1                            | 1                                                | 1                          |
| 25    | 1                        | 1                            | 1                                                | 1                          |
| 26    | 1                        | 1                            | 1                                                | 1                          |
| 27    | 1                        | 1                            | 1                                                | 1                          |
| 28    | 1                        | 1                            | 1                                                | 1                          |
| 29    | 1                        | 1                            | 1                                                | 1                          |
| 30    | 20                       | 20                           | 20                                               | 20                         |
| 31    | 20                       | 20                           | 20                                               | 20                         |
| Total                 | ~78                      | ~36                          | ~80                                              | ~32                        |
</details>

(b) Text-to-visual attention.

Figure 5. Comparison of average attention proportions assigned to visual tokens within the LLM (a) when the query is a visual token and (b) when the query is a text token. We add a case of reindexing position indices with our proposed calibration.  
Table 14. Ablation study on the components of our framework with detailed benchmark results (expanding Table 4).

<table><tr><td></td><td>Incorporating Merging</td><td>Retaining Position</td><td>Attention Calibration</td><td>Average</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td> $\text{SQA}^{\text{IMG}}$ </td><td> $\text{VQA}^{\text{V2}}$ </td><td> $\text{VQA}^{\text{Text}}$ </td><td>SEED</td></tr><tr><td>1</td><td>✗</td><td>✗</td><td>✗</td><td>92.2%</td><td>55.1</td><td>60.0</td><td>1699</td><td>76.8</td><td>68.6</td><td>72.3</td><td>54.9</td><td>52.5</td></tr><tr><td>2</td><td>✗</td><td>✗</td><td>√</td><td>91.0%</td><td>54.5</td><td>57.6</td><td>1688</td><td>78.6</td><td>68.0</td><td>71.8</td><td>52.2</td><td>52.4</td></tr><tr><td>3</td><td>✗</td><td>√</td><td>✗</td><td>90.6%</td><td>54.5</td><td>60.2</td><td>1539</td><td>76.6</td><td>67.0</td><td>71.6</td><td>54.5</td><td>53.2</td></tr><tr><td>4</td><td>✗</td><td>√</td><td>√</td><td>93.1%</td><td>55.5</td><td>62.1</td><td>1739</td><td>76.0</td><td>68.5</td><td>72.3</td><td>54.8</td><td>53.9</td></tr><tr><td>5</td><td>√</td><td>✗</td><td>✗</td><td>93.5%</td><td>56.6</td><td>59.6</td><td>1679</td><td>82.1</td><td>68.7</td><td>73.2</td><td>55.4</td><td>53.4</td></tr><tr><td>6</td><td>√</td><td>✗</td><td>√</td><td>92.9%</td><td>57.1</td><td>62.1</td><td>1530</td><td>87.6</td><td>66.6</td><td>73.4</td><td>53.0</td><td>53.0</td></tr><tr><td>7</td><td>√</td><td>√</td><td>✗</td><td>90.1%</td><td>54.2</td><td>59.0</td><td>1419</td><td>83.4</td><td>66.4</td><td>71.0</td><td>53.4</td><td>53.4</td></tr><tr><td>8</td><td>√</td><td>√</td><td>√</td><td>96.5%</td><td>59.0</td><td>61.9</td><td>1787</td><td>84.9</td><td>68.0</td><td>75.6</td><td>55.4</td><td>56.7</td></tr></table>

Detailed results of tables in main text. We provide detailed results of Tables 4-7 in Tables 14-17, respectively.

## D. More Discussions

Attention calibration with reindexing. As observed in Fig. 2, the reindexing strategy yields an attention distribution across layers that is closer to the full token sequence compared to retaining position indices. This raises a natural question: can applying attention calibration to the reindexing strategy further improve alignment? We present the results of this experiment in Fig. 5. Contrary to the beneficial effect observed with retaining indices, applying attention calibration to reindexing causes the attention weights assigned to visual tokens to significantly overshoot those of the full token sequence (solid purple line). This leads to an excessive dominance of visual context, where visual tokens overshadow the necessary textual information. This imbalance leads to performance degradation, as evidenced by the quantitative results in Row ② of Table 4. The accuracy drops to 91.0%, falling behind the baseline of 92.2% (Row ①). This finding supports that our calibration term is specifically designed to counteract the positional decay inherent to retaining original indices.

Quantitative analysis of information loss in token merging. To validate that our distinctive anchor selection mitigates information loss during merging, we provide a quantitative analysis based on the Hausdorff distance (Huttenlocher et al., 2002; Alvar et al., 2025), which measures how well the merged token set preserves the original visual information. The standard Hausdorff distance computes the distance from each original token to its nearest anchor, but treats all anchors equally and thus disregards the broader semantic coverage of large merged groups. Since a merged anchor $y _ { j }$ serves as a proxy for a cluster of $s _ { j }$ tokens, a larger cluster inherently spans a wider semantic region, and the distance penalty to $y _ { j }$ should be discounted in proportion to its cluster size. To reflect this, we define a weighted Hausdorff distance between the set of original visual tokens X and the set of merged anchor tokens $Y$ as:

$$
h _ {w} (X, Y) = \max _ {x \in X} \min _ {y _ {j} \in Y} \frac {1 - \cos (x , y _ {j})}{s _ {j}}, \tag {13}
$$

Table 15. Full comparison results with attention calibration strategies (expanding Table 5). ’w/o’ denotes no calibration, ${ \overrightarrow { s } } _ { n } { \overrightarrow { \mathbf { \Gamma } } }$ denotes proportional calibration, and ’Ours’ denotes our proposed strategy $s _ { n } ( c - \overline { { \mathcal { D } ( | m - n | ) } } )$ .

<table><tr><td>Method</td><td>Calibration</td><td>Average</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td> $\text{SQA}^{\text{IMG}}$ </td><td> $\text{VQA}^{\text{V2}}$ </td><td> $\text{VQA}^{\text{Text}}$ </td><td>SEED</td></tr><tr><td rowspan="3">VisionZip</td><td>w/o</td><td>89.5%</td><td>54.0</td><td>58.3</td><td>1444</td><td>82.9</td><td>66.2</td><td>70.3</td><td>52.1</td><td>52.6</td></tr><tr><td> $s_n$ </td><td>94.4%</td><td>58.3</td><td>61.4</td><td>1660</td><td>83.8</td><td>68.2</td><td>74.4</td><td>53.5</td><td>55.2</td></tr><tr><td>Ours</td><td>94.8%</td><td>58.5</td><td>62.3</td><td>1726</td><td>84.6</td><td>68.0</td><td>74.3</td><td>52.2</td><td>55.2</td></tr><tr><td rowspan="3">DivPrune</td><td>w/o</td><td>88.9%</td><td>53.2</td><td>56.8</td><td>1400</td><td>84.8</td><td>64.8</td><td>70.5</td><td>51.9</td><td>53.3</td></tr><tr><td> $s_n$ </td><td>94.9%</td><td>58.5</td><td>62.1</td><td>1676</td><td>83.8</td><td>67.0</td><td>74.6</td><td>54.6</td><td>56.1</td></tr><tr><td>Ours</td><td>95.9%</td><td>59.0</td><td>62.2</td><td>1748</td><td>84.8</td><td>68.0</td><td>75.0</td><td>54.4</td><td>56.3</td></tr><tr><td rowspan="3">VisPruner</td><td>w/o</td><td>88.7%</td><td>53.2</td><td>56.6</td><td>1361</td><td>84.3</td><td>65.2</td><td>70.2</td><td>53.0</td><td>53.2</td></tr><tr><td> $s_n$ </td><td>95.1%</td><td>58.9</td><td>60.7</td><td>1661</td><td>84.2</td><td>67.8</td><td>75.1</td><td>55.6</td><td>56.2</td></tr><tr><td>Ours</td><td>96.0%</td><td>59.2</td><td>61.6</td><td>1722</td><td>85.1</td><td>68.0</td><td>75.4</td><td>55.4</td><td>56.5</td></tr><tr><td rowspan="3">HoloV</td><td>w/o</td><td>90.1%</td><td>54.2</td><td>59.0</td><td>1419</td><td>83.4</td><td>66.4</td><td>71.0</td><td>53.4</td><td>53.4</td></tr><tr><td> $s_n$ </td><td>95.3%</td><td>58.9</td><td>61.1</td><td>1694</td><td>83.9</td><td>67.3</td><td>75.2</td><td>55.5</td><td>56.1</td></tr><tr><td>Ours</td><td>96.5%</td><td>59.0</td><td>61.9</td><td>1787</td><td>84.9</td><td>68.0</td><td>75.6</td><td>55.4</td><td>56.7</td></tr></table>

Table 16. Full comparison results with reduction types (Expanding Table 6). P denotes Pruning and M denotes Merging. ’unif.’ stands for uniform anchor sampling, and ’dist.’ stands for distinctive anchor sampling (Ours).

<table><tr><td>Method</td><td>Reduction Type</td><td>Average</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td> $SQA^{IMG}$ </td><td> $VQA^{V2}$ </td><td> $VQA^{Text}$ </td><td>SEED</td></tr><tr><td rowspan="3">VisionZip</td><td>P</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>P + M (unif.)</td><td>94.4%</td><td>58.2</td><td>62.3</td><td>1663</td><td>83.0</td><td>68.8</td><td>74.6</td><td>52.7</td><td>55.3</td></tr><tr><td>P + M (dist.)</td><td>94.8%</td><td>58.5</td><td>62.3</td><td>1726</td><td>84.6</td><td>68.0</td><td>74.3</td><td>52.2</td><td>55.2</td></tr><tr><td rowspan="3">DivPrune</td><td>P</td><td>95.8%</td><td>59.0</td><td>60.8</td><td>1778</td><td>84.1</td><td>69.3</td><td>74.6</td><td>54.6</td><td>55.6</td></tr><tr><td>P + M (unif.)</td><td>94.9%</td><td>57.9</td><td>62.6</td><td>1699</td><td>83.7</td><td>68.2</td><td>74.1</td><td>53.4</td><td>56.0</td></tr><tr><td>P + M (dist.)</td><td>95.9%</td><td>59.0</td><td>62.2</td><td>1748</td><td>84.8</td><td>68.0</td><td>75.0</td><td>54.4</td><td>56.3</td></tr><tr><td rowspan="3">VisPruner</td><td>P</td><td>94.2%</td><td>57.0</td><td>61.1</td><td>1760</td><td>80.0</td><td>69.4</td><td>73.1</td><td>54.9</td><td>54.0</td></tr><tr><td>P + M (unif.)</td><td>95.2%</td><td>58.1</td><td>62.6</td><td>1721</td><td>83.2</td><td>68.6</td><td>74.5</td><td>54.0</td><td>55.7</td></tr><tr><td>P + M (dist.)</td><td>96.0%</td><td>59.2</td><td>61.6</td><td>1722</td><td>85.1</td><td>68.0</td><td>75.4</td><td>55.4</td><td>56.5</td></tr><tr><td rowspan="3">HoloV</td><td>P</td><td>93.1%</td><td>55.5</td><td>62.1</td><td>1739</td><td>76.0</td><td>68.5</td><td>72.3</td><td>54.8</td><td>53.9</td></tr><tr><td>P + M (unif.)</td><td>94.8%</td><td>57.4</td><td>62.8</td><td>1690</td><td>82.2</td><td>68.7</td><td>74.5</td><td>54.2</td><td>55.9</td></tr><tr><td>P + M (dist.)</td><td>96.5%</td><td>59.0</td><td>61.9</td><td>1787</td><td>84.9</td><td>68.0</td><td>75.6</td><td>55.4</td><td>56.7</td></tr></table>

Table 17. Full comparison results with the position of tokens (Expanding Table 7).

<table><tr><td>Position Type</td><td>Average</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td> $SQA^{IMG}$ </td><td> $VQA^{V2}$ </td><td> $VQA^{Text}$ </td><td>SEED</td></tr><tr><td>First</td><td>96.3%</td><td>58.6</td><td>61.9</td><td>1732</td><td>85.6</td><td>69.0</td><td>75.4</td><td>55.8</td><td>56.1</td></tr><tr><td>Median</td><td>96.5%</td><td>59.0</td><td>61.9</td><td>1787</td><td>84.9</td><td>68.0</td><td>75.6</td><td>55.4</td><td>56.7</td></tr><tr><td>Mean</td><td>96.0%</td><td>58.2</td><td>62.2</td><td>1741</td><td>85.3</td><td>68.1</td><td>75.6</td><td>55.0</td><td>56.3</td></tr><tr><td>Last</td><td>95.2%</td><td>57.7</td><td>62.0</td><td>1695</td><td>85.1</td><td>68.0</td><td>74.5</td><td>55.3</td><td>55.6</td></tr></table>

where $\cos ( \cdot , \cdot )$ denotes the cosine similarity and $s _ { j }$ is the number of tokens merged into the anchor $y _ { j } . \mathrm { A }$ lower $h _ { w }$ indicates that the merged tokens better cover the original visual information, implying less information loss.

We measure $h _ { w }$ on four baselines under three configurations: pruning only, pruning combined with merging using uniformly sampled anchors, and pruning combined with our distinctive anchors. The values are averaged over 100 non-overlapping images from the GQA (Hudson & Manning, 2019) dataset using LLaVA-1.5-7B (Liu et al., 2024a). As shown in Table 18, pruning alone incurs the highest information loss, since discarding tokens entirely removes their information. While uniform merging substantially reduces this loss by aggregating tokens, our distinctive anchor selection achieves the lowest $h _ { w }$ across all baselines. This confirms that selecting representative and discriminative anchors minimizes information loss during feature averaging, complementing the attentional and positional rectification of our framework.

Hyperparameter γ. The pruning ratio γ determines the allocation of the token budget between pruning and merging stages in our hybrid reduction strategy. We investigate the impact of varying $\gamma$ on performance (Table 19). The results indicate that a balanced allocation $( \gamma = 0 . 5 )$ yields optimal performance, demonstrating that both pruning and merging contribute significantly to effective token reduction. We conjecture that this is because the token pruning preserves intact information of selected tokens while the token merging captures global context via aggregating multiple tokens. Extreme allocations, such as $\gamma = 0 . 0$ (merging only) $\mathrm { o r } \gamma = 1 . 0$ (pruning only), lead to suboptimal performance, highlighting the importance of a hybrid approach.

Table 18. Quantitative analysis of information loss in token merging, measured by the weighted Hausdorff distance $h _ { w }$ (Eq. 13) on LLaVA-1.5-7B (Liu et al., 2024a). Lower is better. Values are averaged over 100 images from the GQA (Hudson & Manning, 2019) dataset. Best results in Bold.

<table><tr><td>Baseline</td><td>Pruning Only</td><td>P + M (Uniform)</td><td>P + M (Distinctive)</td></tr><tr><td>VisionZip</td><td>0.7168</td><td>0.0169</td><td>0.0077</td></tr><tr><td>DivPrune</td><td>0.2889</td><td>0.0177</td><td>0.0078</td></tr><tr><td>VisPruner</td><td>0.4590</td><td>0.0169</td><td>0.0078</td></tr><tr><td>HoloV</td><td>0.6701</td><td>0.0167</td><td>0.0076</td></tr></table>

Table 19. Ablation study on the hyperparameter $\gamma .$

<table><tr><td></td><td>Average</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td> $\text{SQA}^{\text{IMG}}$ </td><td> $\text{VQA}^{\text{V2}}$ </td><td> $\text{VQA}^{\text{Text}}$ </td><td>SEED</td></tr><tr><td>LLaVA-1.5-7B</td><td>100%</td><td>61.9</td><td>64.6</td><td>1862</td><td>85.9</td><td>69.5</td><td>78.5</td><td>58.2</td><td>58.6</td></tr><tr><td>HoloV (Baseline)</td><td>92.2%</td><td>55.1</td><td>60.0</td><td>1699</td><td>76.8</td><td>68.6</td><td>72.3</td><td>54.9</td><td>52.5</td></tr><tr><td> $\gamma = 0$ </td><td>95.7%</td><td>59.4</td><td>62.8</td><td>1727</td><td>84.4</td><td>68.3</td><td>74.6</td><td>53.4</td><td>56.7</td></tr><tr><td> $\gamma = 0.25$ </td><td>96.2%</td><td>59.4</td><td>62.8</td><td>1703</td><td>84.7</td><td>69.3</td><td>75.7</td><td>54.9</td><td>56.4</td></tr><tr><td> $\gamma = 0.5$ </td><td>96.5%</td><td>59.0</td><td>61.9</td><td>1787</td><td>84.9</td><td>68.0</td><td>75.6</td><td>55.4</td><td>56.7</td></tr><tr><td> $\gamma = 0.75$ </td><td>95.4%</td><td>57.3</td><td>63.7</td><td>1740</td><td>82.8</td><td>68.1</td><td>74.6</td><td>55.1</td><td>55.5</td></tr><tr><td> $\gamma = 1.0$ </td><td>93.1%</td><td>55.5</td><td>62.1</td><td>1739</td><td>76.0</td><td>68.5</td><td>72.3</td><td>54.8</td><td>53.9</td></tr></table>

## E. Detailed Computational Complexity Analysis

We provide a detailed calculation of the FLOPs presented in Table 8 and expanded in Table 20. To precisely estimate the inference cost, we compare the FLOPs occurring in the visual encoder, anchor selection, and the LLM. We exclude the projector from the total FLOPs calculation as its computational cost is negligible compared to the LLM and visual encoder.

Table 20. Detailed computational complexity analysis. The Total MLLM FLOPs is the sum of the Visual Encoder (constant) and LLM Decoder (variable based on $n _ { v i s } )$ . The projector’s cost is omitted as it is negligible. $N _ { s y s } = 3 5 , N _ { v i s } = 5 7 6 , N _ { t x t } = 2 5$ .

<table><tr><td rowspan="2">Module</td><td rowspan="2">Configuration / Formula</td><td rowspan="2">Params</td><td colspan="3">Target Visual Tokens ( $n_{vis}$ )</td></tr><tr><td>192</td><td>128</td><td>64</td></tr><tr><td rowspan="2">Visual Encoder(ViT-L/14, 24 Layers)</td><td>Input Tokens ( $N_{vis}$ )Hidden Size (D) / FFN (F)FLOPs (Attn) ≈  $4N_{vis} D^{2} + 2N_{vis}^{2} D$ FLOPs (MLP) ≈  $2N_{vis} FD$ </td><td>5761024 / 4096--</td><td colspan="3">576 (Fixed)Constant ParamsConstant Cost</td></tr><tr><td>FLOPs</td><td>-</td><td colspan="3">0.190 TFLOPs</td></tr><tr><td>Projector</td><td>Linear Projection</td><td>-</td><td colspan="3">Negligible</td></tr><tr><td rowspan="2">LLM Decoder(LLaMA-7B, 32 Layers)</td><td>System + Text ( $N_{sys} + N_{txt}$ )Total Input Tokens (N)Hidden Size (D) / FFN (F)FLOPs (Attn) ≈  $4ND^{2} + 2N^{2} D$ FLOPs (MLP) ≈ 3NFD</td><td>60 $60 + n_{vis}$ 4096 / 11008--</td><td>252</td><td>60 (Fixed)188Constant ParamsVariable Cost</td><td>124</td></tr><tr><td>FLOPs</td><td>-</td><td>1.649 TFLOPs</td><td>1.227 TFLOPs</td><td>0.807 TFLOPs</td></tr><tr><td>Total MLLM</td><td>Visual Encoder + LLM</td><td>-</td><td>1.839 T(TFLOPs)</td><td>1.417 T(TFLOPs)</td><td>0.997 T(TFLOPs)</td></tr><tr><td rowspan="2">Overhead</td><td>Anchor Selection Cost</td><td>-</td><td colspan="3">1.359 GFLOPs (≈ 0.001 TFLOPs)</td></tr><tr><td>Ratio (Overhead / Total)</td><td>-</td><td>0.074%</td><td>0.096%</td><td>0.136%</td></tr></table>