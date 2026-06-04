# TGV-KV: Text-Grounded KV Eviction for Vision-Language Models

Jizhihui Liu 1 Ruizi Han 1 Miao Zhang† 1 2 Rui Shao 1 Xuebo Liu 1 Weili Guan 1 2 Yaowei Wang 1 2

# Abstract

Vision-Language Models (VLMs) inherit the autoregressive generation paradigm and cache the keys and values (KV) of all previous tokens to accelerate inference, resulting in memory consumption that scales linearly with context length. This issue is particularly pronounced in VLMs due to substantial redundancy in the visual modality. Although KV cache eviction approaches can effectively reduce inference memory, they often incur significant performance degradation in VLMs, as most are designed for language models and overlook the inherent gap between text and vision. By systematically analyzing the modality gap in VLMs in this work, we argue that the importance of visual information should be grounded in textual guidance and accordingly propose a Text-Grounded KV Eviction method for VLMs (TGV-KV). TGV-KV comprises three submodules: (1) Text-Vision Budgeting (TVB) assigns budget to each layer based on the mutual information interaction. (2) Text-Weighted Ranking (TWR) assesses the priority of text and ranks vision importance based on weighted text-image attention. (3) Text-Prioritised Retention (TPR) policy strategically preserves text KV to avoid acute information loss. We evaluate TGV-KV across five models with different sizes and architectures, showing that TGV-KV preserves 99.2% full-KV accuracy on the VizWiz-VQA task with LLaVA-NeXT and boosts end-to-end throughput by 52.6% with an extreme retention budget of 5%. Code Link.

# 1. Introduction

Vision-Language Models (VLMs) (Bai et al., 2025; Liu et al., 2024; 2023) have revolutionized multimodal understanding and visual reasoning in recent years. Inheriting

†Corresponding Author. 1Harbin Institute of Technology, Shenzhen 2Peng Cheng Laboratory. Correspondence to: <danielement321@gmail.com>, <zhangmiao@hit.edu.cn>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

![](images/e23ff4b93da91f1341a7d5d60dd7dc17814b7c10c56af1e449fa960cdfb15c71.jpg)  
Figure 1. Visualization and Consequence of Modality Gap. (a) The average cosine similarity of vision and text tokens with k neighbours. (b) The layer 2 attention map (in log scale) in LLaVA, the intra-modality part (text-text, vision-vision) is much more intense than inter-modality (text-vision). (c) Accumulated attention score across all the layers, the eviction is highly uneven and most text KV pairs are evicted by cumulative attention score.

the auto-regressive architecture of Large Language Models (LLMs) (Grattafiori et al., 2024; Yang et al., 2025a), VLMs rely on the Key-Value (KV) cache mechanism to eliminate redundant computations and accelerate generation. However, the KV cache grows linearly with context length, incurring severe bottlenecks in memory consumption and inference latency. This challenge is more exacerbated in the multimodal setting since high-resolution images and long videos often take up thousands of tokens (Lin et al., 2024). Despite the large quantity, many studies (Chen et al., 2024a; Liu et al., 2025a) have found that most of these vision features are highly redundant and removing a large proportion of them negligibly degrades the model performance.

To overcome the memory issue in VLM, most current methods (Chen et al., 2024a; Liu et al., 2025a; Yang et al., 2025b) prune redundant vision tokens directly before or during the prefill at a time. However, these methods often influence model performance sharply since once a token is pruned, it is no longer accessed in subsequent model layers or decoding steps, and the contained unextracted information is permanently lost. With higher flexibility, KV cache eviction serves as a promising mitigation. However, such methods face critical performance degradation in VLM since most are designed for language models. To explain this, we identify three significant differences between text and vision and attribute this drop to the modality gap. (1) Vision tokens are very similar to each other, while text tokens are very diverse (Fig. 1 (a)). (2) This inherent gap causes a low-attention area in the text-vision part, making the unimodal attention fluctuate (Fig. 1 (b)). (3) This special distribution causes a sharp shift in accumulated attention at the intersection of vision and text, making the eviction extremely uneven (Fig. 1 (c)). Therefore, directly transferring current KV cache eviction methods from LLM to VLM overlooks the spatial redundancy of vision and the mutual modality interaction, suffering from the inconsistent attention distribution.

In this paper, we take a systematic study of the multimodal attention pattern in VLM and aim to elucidate the role of each component in KV cache eviction. Specifically, we reveal three key observations through extensive experiments, including text-vision attention’s effectiveness in layer budget division, an appropriate indicator to measure multimodal KV importance, and the relative priority of different modalities during eviction. Based on these key findings, we propose a robust and Text-Grounded KV eviction framework for VLMs, i.e., TGV1-KV. The key idea of TGV-KV is to overcome the modality inconsistency issue by taking full advantage of text features. To this end, we propose three synergistic components: a layer budget allocation policy Text-Vision Budgeting (TVB), a multimodal KV importance judge Text-Weighted Ranking (TWR), and an eviction criterion Text-Prioritised Retention (TPR). After the prefill, TVB extracts the text-vision attention and normalizes the summation layer-wisely to divide the total budget. TWR first evaluates the significance of each text token and applies a positional average to obtain the weight coefficient for vision token importance judgment. TPR adaptively evicts multimodal KV pairs based on the importance score while always keeping text KVs as long as the budget allows. These three modules take full consideration of text modality and maintain cross-modality consistency sufficiently, achieving outstanding performance in accuracy and efficiency.

We evaluate TGV-KV across diverse popular VLMs, covering basic LLaVA-series (Liu et al., 2023; 2024) to stateof-the-art Qwen-series (Bai et al., 2025). With an extreme compression ratio of 5%, TGV-KV retains 92.5% full-KV accuracy in DocVQA on Qwen3-VL-8B, while surpassing the best baseline by 33.0% on LLaVA-NeXT, along with a 95% reduction in memory and 52.6% acceleration.

Our main contributions are summarized as follows:

• We take a systematic study of the attention pattern in VLMs and propose TGV-KV, an out-of-the-box KV eviction approach for multimodal KV eviction.   
• We design three modules or policies, i.e., TVB, TWR, and TPR, to allocate budget, rank multimodal importance, and evict KV pairs while preserving performance, which can be adopted by subsequent VLM KV eviction studies seamlessly.

• Extensive studies across multiple VLMs demonstrate our TGV-KV achieves outstanding performance and substantial memory reduction, outperforming existing methods in both accuracy and efficiency.

# 2. Related Works

# 2.1. Vision-Language Models

Vision-Language Models (Liu et al., 2023; Achiam et al., 2023; Team et al., 2023) are usually composed of a Large Language Model decoder (Grattafiori et al., 2024; Yang et al., 2025a), a vision encoder (Radford et al., 2021; Zhai et al., 2023) and an adaptor (Alayrac et al., 2022). The vision encoder is a Vision Transformer (Dosovitskiy et al., 2021) that encodes visual inputs into abundant visual tokens, which are then projected into the text semantic space by the adaptor. The projected vision features are then concatenated with text embeddings, forming a unified multimodal input sequence. In the LLM decoder, all the counterparts are treated in the same way, performing unified causal selfattention (Vaswani et al., 2017) in each layer. During inference, VLM inherits the auto-regressive generation manner of LLM and stores the KVs of past tokens to accelerate generation. The memory consumption of KV cache grows linearly with the input sequence. For VLMs that adopt a dynamic-resolution vision encoder (Bai et al., 2025; Chen et al., 2024b), a video may take tens of thousands of tokens, incurring severe KV cache memory consumption.

# 2.2. Efficient Inference for VLMs

Many approaches have been proposed to overcome the memory burden in VLM generation, which can be roughly categorized into token pruning and KV eviction. Token pruning is usually applied before or during the prefill, which prunes tokens at a time. FastV (Chen et al., 2024a) and VisionZip (Yang et al., 2025b) prune tokens based on the attention score, while CDPruner (Zhang et al., 2025a) utilizes the text embeddings as a guidance. These methods lack flexibility and suffer from severe information loss. KV cache eviction evicts KV after the prefill and compresses the memory consumption during the decode phase. H2O (Zhang et al., 2023), SnapKV (Li et al., 2024b) assess KV importance by attention score, PyramidKV (Cai et al., 2024), SparseMM (Wang et al., 2025b), Ada-KV (Feng et al., 2024) allocate dynamic budget to layers or heads. Most of these methods do not consider the modality gap and handle different modalities as the same, which is suboptimal. AirCache (Huang et al., 2025) identifies key text tokens and judges vision importance with them, however, this requires extra computation and lacks mutual information flow analyses during budget allocation. In this work, we solve the mentioned problems by introducing text ground and mutual information flow during both budget allocation and importance assessment.

# 3. Method

In this section, we first revisit the principle of KV cache, and then present three key observations. Finally, we show the design of TGV-KV and depict it in Fig. 3.

# 3.1. Preliminary

Similar to LLM, the generation of a VLM can be divided into two phases, i.e., Prefill and Decode. Eviction methods are applied before the prefill phase and during the decode phase to control the total budget of KV cache.

Prefill Phase. The vision encoder and text tokenizer first encode multimodal inputs and text prompts into visual tokens $\mathbf { X } _ { \mathbf { v } } \in \mathbb { R } ^ { N _ { v } \times d }$ and text tokens $\mathbf { X _ { t } } \ \in \ \mathbb { R } ^ { N _ { t } \times d }$ , where $N _ { v }$ and $N _ { t }$ denote the number of vision and text tokens, d denotes the model dimension. All the counterparts are then concatenated into a unified sequence $\mathbf { X } \in \mathbb { R } ^ { ( \mathbf { \dot { N } } _ { v } + \boldsymbol { N } _ { t } ) \times { d } }$ . In the self-attention of each layer, the model calculates

$$
\mathbf {Q} = \mathbf {X W} _ {\mathbf {Q}}, \mathbf {K} = \mathbf {X W} _ {\mathbf {K}}, \mathbf {V} = \mathbf {X W} _ {\mathbf {V}}, \tag {1}
$$

where $\mathbf { W _ { Q } } , \mathbf { W _ { K } } , \mathbf { W _ { V } } \in \mathbb { R } ^ { d \times d }$ are pretrained projection matrices. The self-attention is then calculated by

$$
\mathbf {A} = \operatorname{softmax} (\mathbf {Q K} ^ {\mathsf {T}} + \mathbf {M}) \in \mathbb {R} ^ {(N _ {v} + N _ {t}) \times (N _ {v} + N _ {t})}, \tag {2}
$$

where M is an upper triangular matrix filled with −∞ to avoid information leak. Since the key and value of context tokens are accessed at each decoding step, all the K and V of each layer are cached to avoid re-computation.

Decode Phase. During per-token generation, the input of time step t is a single token $\mathbf { x } \in \mathbb { R } ^ { d }$ . The key and value of x are stored in the cache by

$$
\mathbf {K} = \operatorname{concat} (\mathbf {K}, \mathbf {x W} _ {\mathbf {k}}) \in \mathbb {R} ^ {(N _ {v} + N _ {t} + t) \times d}, \tag {3}
$$

$$
\mathbf {V} = \operatorname{concat} (\mathbf {V}, \mathbf {x W} _ {\mathbf {v}}) \in \mathbb {R} ^ {(N _ {v} + N _ {t} + t) \times d}. \tag {4}
$$

The memory footprint of K and V grows linearly with t. As the context grows longer, this issue can not be ignored.

# 3.2. Key Insights

KV cache eviction methods commonly include two procedures: budget allocation and KV eviction. In this subsection, we conduct experiments to identify different attention pattern combinations in layer budgeting and eviction criteria.

Inter-Layer Budget Allocation. Intuitively, different layers handle different semantics, thus need different budgets. Unlike previous works with a fixed rule-based layer budget allocation policy (Chen et al., 2024a; Cai et al., 2024), we aim to assign budgets dynamically based on the attention score since tokens interact with each other in self-attention. Note that we currently do not consider head-wise analyses since unpacking heads breaks the parallelism of attention computation and affects efficiency.

![](images/c868f801cf3c392af22a6723bacb72b2bbae1367db7a10b2d284d7e9b95dc061.jpg)

<details>
<summary>area</summary>

| Model | Layer | Vision | Text |
|---|---|---|---|
| LLaVA | 1 | High | High |
| LLaVA | 2 | Medium | Low |
| LLaVA | 3 | Low | Low |
| LLaVA | 4 | Low | Low |
| LLaVA | 5 | Low | Low |
| LLaVA | 6 | Low | Low |
| LLaVA | 7 | Low | Low |
| LLaVA | 8 | Low | Low |
| LLaVA | 9 | Low | Low |
| LLaVA | 10 | Low | Low |
| Qwen3-VL | 1 | High | High |
| Qwen3-VL | 2 | Medium | Low |
| Qwen3-VL | 3 | Low | Low |
| Qwen3-VL | 4 | Low | Low |
| Qwen3-VL | 5 | Low | Low |
| Qwen3-VL | 6 | Low | Low |
| Qwen3-VL | 7 | Low | Low |
| Qwen3-VL | 8 | Low | Low |
| Qwen3-VL | 9 | Low | Low |
| Qwen3-VL | 10 | Low | Low |
</details>

Figure 2. Attention Map of VLMs. We average all the heads and plot the attention map in LLaVA-1.5-7B and Qwen3-VL-8B. Each zoomed-in area denotes the text-text attention in a multimodal input. We label a few dominant text token in orange box. More visualizations can be found in Appendix C.1.

Table 1. Eviction Performance with Different Policies. We evaluate different layer budget allocation policies and KV eviction criteria on LLaVA. Note that even if evicting text first in (c), the system tokens are always kept. Percentage denotes retention ratio. 

<table><tr><td rowspan="2">Attention Pattern</td><td colspan="3">ChartQA↑</td><td colspan="3">TextVQA $^{lite}$  ↑</td></tr><tr><td>50%</td><td>20%</td><td>5%</td><td>50%</td><td>20%</td><td>5%</td></tr><tr><td>Vanilla Model</td><td>18.0</td><td>18.0</td><td>18.0</td><td>47.9</td><td>47.9</td><td>47.9</td></tr><tr><td colspan="7">(a) Same Eviction Criterion (TV+TT) + Different Layer Budget</td></tr><tr><td>Uniform</td><td>17.8</td><td>17.8</td><td>13.9</td><td>47.9</td><td>47.7</td><td>32.8</td></tr><tr><td>Vision-Vision (VV)</td><td>17.7</td><td>17.6</td><td>14.1</td><td>48.3</td><td>47.4</td><td>34.2</td></tr><tr><td>Text-Text (TT)</td><td>17.8</td><td>17.9</td><td>14.2</td><td>48.3</td><td>46.9</td><td>36.1</td></tr><tr><td>Text-Vision (TV)</td><td>17.8</td><td>17.8</td><td>14.3</td><td>48.3</td><td>47.5</td><td>36.4</td></tr><tr><td colspan="7">(b) Same Layer Budget (Uniform) + Different Importance Criteria</td></tr><tr><td>Observation Window</td><td>17.9</td><td>16.6</td><td>0.4</td><td>46.3</td><td>36.4</td><td>8.7</td></tr><tr><td>Self-Attention</td><td>4.7</td><td>3.9</td><td>4.8</td><td>31.8</td><td>28.4</td><td>23.5</td></tr><tr><td>VV+TT Attention</td><td>4.8</td><td>3.7</td><td>4.6</td><td>32.3</td><td>28.4</td><td>22.8</td></tr><tr><td>TV+TT Attention</td><td>17.9</td><td>17.7</td><td>11.0</td><td>48.4</td><td>47.8</td><td>37.3</td></tr><tr><td colspan="7">(c) Same Layer Budget (Uniform) + Different Eviction Modalities</td></tr><tr><td>Evict Vision First</td><td>16.7</td><td>14.8</td><td>10.0</td><td>46.3</td><td>40.4</td><td>31.0</td></tr><tr><td>Evict Text First</td><td>0.2</td><td>0.2</td><td>0.2</td><td>4.4</td><td>4.4</td><td>4.4</td></tr></table>

We mainly consider three types of attention as guidance, i.e., vision-vision (VV), text-vision (TV), and text-text (TT). Specifically, given the set $\mathbf { \mathcal { A } } ^ { ( \mathrm { x } ) } = \{ \mathbf { A } _ { l } ^ { ( \mathrm { x } ) } \} _ { l = 1 } ^ { L }$ for each mode x ∈ {VV, TV, TT}, where L denotes the number of layers in the decoder, we define the budget $b _ { l } ^ { \mathrm { ( x ) } }$ for each layer as:

$$
b _ {l} ^ {(x)} = \frac {\sum_ {i , j} [ \mathbf {A} _ {l} ^ {(x)} ] _ {i j}}{\sum_ {l ^ {\prime} = 1} ^ {L} \sum_ {i , j} [ \mathbf {A} _ {l ^ {\prime}} ^ {(x)} ] _ {i j}}. \tag {5}
$$

As shown in block (a) of Table 1, Observation 3.1. We hypothesize that the TV attention implies the intensity of information exchange, serving as an ideal indicator of budget allocation since higher exchange needs more KV retention.

Observation 3.1. The intensity of cross-modal attention serves as a proxy for semantic fusion, positively correlated with a layer’s demand for KV retention budget.

Intra-Layer KV Eviction. The remaining main problem is to evict “low-importance” KV pairs. We evaluate using an observation window (Li et al., 2024b), sum of self-attention, sum of intra-modality attention (VV+TT), and combination of intra- and inter- modality attention (TV+TT).

![](images/9ebe05431f0a2e0e2d408bf98619cad43761cb518826dc7d2f673462b4d90a0b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Attention"] --> B["Text-Vision"]
    B --> C["Text-Vision Budgeting (TVB)"]
    C --> D["Text-Weighted Ranking (TWR)"]
    D --> E["Text-Text"]
    E --> F["Column Sum"]
    F --> G["Text Score: s_l^(T) ∈ ℝ^N_t"]
    G --> H["Causal Average: +4 ÷3 ÷2 ÷1"]
    H --> I["×"]
    I --> J["Text Weight"]
    J --> K["Vision Score: s_l^(V) ∈ ℝ^N_v"]
    K --> L["Text-Prioritised Retention (TPR)"]
    L --> M["KV Index: b_l > N_t, b_l < N_t, b_l = N_t"]
```
</details>

Figure 3. Overview of TGV-KV framework. Our method consists of three submodules or policies: (1) Text-Vision Budgeting adaptively allocates layer-wise KV budgets via text-image attention summation; (2) Text-Weighted Ranking determines vision importance score by re-weighting text-image attention with averaged text attention, and text importance by attention summation; (3) Text-Prioritised Retention selectively evicts KV pairs based on the importance score while always preserving as many text features as the budget allows.

The results are presented in block (b) of Table 1, indicating Observation 3.2. This motivates us to design a robust KV importance criteria for VLM.

We show the results of randomly evicting text and vision KV pairs in block (c) in Table 1. It is worth noting that eviction of a small subset of text KV pairs cause much more damage to a larger subset of vision KV pairs, calling for extra protection to text KV pairs during eviction.

Observation 3.2. Integrating inter-modality interaction (TV) with intra-modality attention (TT) yields the most robust importance metric, indicating that visual saliency is intrinsically text-dependent.

Observation 3.3. Text features are highly sensitive to eviction, causing performance collapse, whereas visual features exhibit high redundancy, permitting aggressive eviction.

# 3.3. TGV-KV

Text-Vision Budgeting (TVB). The VLM handles different information levels in a hierarchical way (Liu et al., 2025a), and different budgets should be assigned to different layers (Qin et al., 2025). Instead of using a fixed rule-based policy (Cai et al., 2024) or a calibration dataset (Wang et al., 2025a;b), we adopt a cross-attention-driven policy to perform on-the-fly budgeting.

Based on Observation 3.1, we first slice text-vision attention A(TV)l ∈ $\mathbf { A } _ { l } ^ { ( \mathrm { T V } ) } \in \mathbb { R } ^ { N _ { t } \times N _ { v } }$ out of $\mathbf { A } _ { l } .$ , which is calculated by

$$
\mathbf {A} _ {l} ^ {\mathrm{(TV)}} = \operatorname{softmax} (\mathbf {Q} _ {l} ^ {\mathrm{(T)}} [ \mathbf {K} _ {l} ^ {\mathrm{(V)}} ] ^ {\mathsf {T}}) \in \mathbb {R} ^ {N _ {t} \times N _ {v}}, \tag {6}
$$

where ${ \bf Q } _ { l } ^ { \left( \mathrm { T } \right) } \in \mathbb { R } ^ { N _ { t } \times d }$ and ${ \bf K } _ { l } ^ { ( \vee ) } \in \mathbb R ^ { N _ { v } \times d }$ denotes the text query and vision key in layer l. We then sum up the textvision attention in each layer and normalize along the layer to get the budget bl by Eq. 5.

Text-Weighted Ranking (TWR). We assign an importance ratio to each vision token and rank them in TWR. As shown in set (b) of Table 1, the sum of text-vision attention is ideal for visual KV eviction. However, this straightforward criterion lacks the inner importance of text tokens. For example, for instructions “Describe this image.” and “Is there a taxi near the streetlight?”, the text KV pairs are not equally important (Zhang et al., 2025c), and the preserved KV pairs should demonstrate different focuses. To this end, we use text tokens to weight vision tokens.

In Fig. 2, when zooming the text-text attention area, there exists a set of text tokens that consistently take up a large proportion of attention in all the subsequent tokens (Xiao et al., 2024; Qin et al., 2025), manifesting as a continuous prominent vertical line. We denote these text tokens as dominant text tokens. Since the dominant text tokens are critical for text understanding and generation, it is necessary to assign a higher importance score to vision KV pairs that are attended by dominant text tokens.

Specifically, we slice text-text attention A(TT)l ${ \bf A } _ { l } ^ { ( \mathrm { T T } ) } \in \mathbb { R } ^ { N _ { t } \times N _ { t } }$ out of ${ \bf A } _ { l }$ and compute the column-wise sum of each text token. To account for the triangular causal mask, we divide each token’s sum by the position to get an average

$$
w _ {l, j} = \frac {\sum_ {i = j} ^ {N _ {t}} [ \mathbf {A} _ {l} ^ {(\mathrm{TT})} ] _ {i j}}{N _ {t} - j + 1}, j = 1 \dots N _ {t}. \tag {7}
$$

Upon computing the significance score for each text KV pair, we reuse A(TV)l ${ \bf A } _ { l } ^ { \left( \mathrm { T V } \right) }$ in Eq. 6 and multiply each i-th row with the normalized weight $\tilde { w } _ { l , i }$ to get a text-weighted crossmodality attention summation. We compute the final importance score s(V)l,j $s _ { l , j } ^ { ( \mathrm { V } ) }$ for each vision KV in layer l by

$$
s _ {l, j} ^ {\mathrm{(V)}} = \sum_ {i = 1} ^ {N _ {t}} \tilde {w} _ {l, i} [ \mathbf {A} _ {l} ^ {\mathrm{(TV)}} ] _ {i j}, j = 1, \dots , N _ {v}. \tag {8}
$$

For text KVs, we directly take the sum of self-attention score as the importance score $s _ { l , j } ^ { ( \mathrm { T } ) }$ since it has been proven to be effective and widely adopted (Zhang et al., 2023; Feng et al., 2024), which is calculated by

$$
s _ {l, j} ^ {\mathrm{(T)}} = \sum_ {i = j} ^ {N _ {t}} [ \mathbf {A} _ {l} ^ {\mathrm{(TT)}} ] _ {i j}, j = 1, \dots , N _ {t}. \tag {9}
$$

Text-Prioritised Retention (TPR). Similar to KV cache eviction, most pruning methods (Chen et al., 2024a; Yang et al., 2025b; Liu et al., 2025a) prune vision tokens only since pruning text tokens often leads to extreme damage to model performance. As shown in our Observation 3.3, pruning text tokens significantly damages the performance. To mitigate this issue, we always try to keep as many text KV pairs as budget allows. We use a text-prioritised retention criterion, which only evicts text KVs when the retained exceeds the budget, though all the vision KVs are removed.

Let $\mathcal { T } = \{ 1 , \cdots , N _ { t } \}$ and $\mathcal { V } = \{ 1 , \cdots , N _ { v } \}$ denote the sets of indices for text and vision KV pairs, respectively. For layer l, the set of retained indices $\mathcal { T } _ { l }$ is determined by:

$$
\mathcal {I} _ {l} = \left\{ \begin{array}{l l} \mathcal {T} \cup \operatorname{TopK} \left(\left\{s _ {l, j} ^ {\mathrm{(V)}} \right\} _ {j \in \mathcal {V}}, b _ {l} - N _ {t}\right), & \text { if } b _ {l} > N _ {t} \\ \operatorname{TopK} \left(\left\{s _ {l, j} ^ {\mathrm{(T)}} \right\} _ {j \in \mathcal {T}}, b _ {l}\right). & \text { if } b _ {l} \leq N _ {t} \end{array} \right. \tag {10}
$$

This criterion ensures that vision KVs are only retained after the entire text context is secured, and text KVs are only evicted under extreme budget constraints.

# 4. Experiments

# 4.1. Experiment Settings

Models. We evaluate TGV-KV on multiple VLMs with different architectures, including a basic model LLaVA-1.5- 7B (Liu et al., 2024), high-resolution models LLaVA-NeXT-7B (Liu et al., 2024) and LLaVA-OV (Li et al., 2024a), and state-of-the-art open-source model Qwen3-VL-series with various sizes of 4B and 8B (Bai et al., 2025). All the models are evaluated without finetuning.

Datasets. We evaluate TGV-KV on both image and video tasks. For image tasks, we choose two types of tasks where text or vision dominates to get a comprehensive evaluation of the model’s ability. Vision-dominant tasks require the model to observe the image carefully, and we choose four representative VQA tasks, including ChartQA (Masry et al., 2022), DocVQA (Mathew et al., 2021), VizWiz (Gurari et al., 2018), and TextVQA (Singh et al., 2019). Textdominant tasks mainly focus on the model’s generation quality based on the observation, and we choose TextCaps (Sidorov et al., 2020) and COCO-Caption-2017 (Lin et al., 2015) for this type. For video tasks, we adopt Video-TT (Zhang et al., 2025b), which comprises rephrased, wronglyled, and correctly-led adversarial open-ended questions to evaluate reasoning ability and robustness. More details on dataset description can be found in Appendix A.1.

Comparisons. We compare TGV-KV with multiple KV eviction methods, including those originally designed for LLMs (StreamingLLM (Xiao et al., 2024), SnapKV (Li et al., 2024b), $_ { \mathrm { H } _ { 2 } \mathrm { O } }$ (Zhang et al., 2023)) and VLMs (ElasticCache (Liu et al., 2025b), PrefixKV (Wang et al., 2025a)). Among them, $\mathrm { H } _ { 2 } \mathrm { O }$ and PrefixKV mainly adopt the sum of self-attention to allocate budget or evict KV pairs, SnapKV utilizes an observation window to assign importance score, and StreamingLLM preserves KV of the first and latest tokens that receive most attention. Since most results on new models are not reported by the original paper, we follow the original procedure and reproduce all the results ourselves. We set the observation window to 64 for SnapKV and the sink token number to 4 for StreamingLLM.

Implementation Details. We conduct all the accuracy evaluations with the LMMs-Eval toolkit (Zhang et al., 2024). All the experiments for image tasks are performed on a machine with 4× RTX 5090 (32G), while video tasks are implemented on a machine with 4× A800 (80G). Please refer to Appendix A.2 for more implementation details.

# 4.2. Accuracy Results

Vision-Dominant Results. We show the results for visiondominant tasks in Table 2. Notably, TGV-KV establishes a new state-of-the-art across nearly all tasks and KV budgets.

We utilize LLaVA as a representative baseline model due to its straightforward architecture. Unlike modern VLMs with sufficient training and optimization, LLaVA is highly susceptible to error evictions, and any suboptimal token eviction leads to severe performance degradation. We show that most methods struggle with this model with an extreme budget of 5%, with some performance metrics plummeting to as low as 5% of the original performance. In contrast, TGV-KV consistently outperforms these baselines. On LLaVA-NeXT, a model with a higher resolution and a longer input sequence, TGV-KV reserves 99.2% accuracy on VizWiz, and 97.4% on TextVQA with only 5% of the original KV cache size. These results highlight TGV-KV’s robustness under high-resolution scenarios.

Table 2. Accuracy Results on Vision-Dominant Tasks. Vanilla refers to the baseline model using the full KV cache. The percentages indicate the specific KV cache retention rates. The best results in each setting are highlighted in bold. 

<table><tr><td rowspan="2">Methods</td><td colspan="4"> $ChartQA^{Relaxed\ Acc.\uparrow}$ </td><td colspan="4"> $DocVQA^{ANLS\uparrow}$ </td><td colspan="4"> $VizWiz^{Acc.\uparrow}$ </td><td colspan="4"> $TextVQA^{Acc.\uparrow}$ </td></tr><tr><td>50%</td><td>20%</td><td>10%</td><td>5%</td><td>50%</td><td>20%</td><td>10%</td><td>5%</td><td>50%</td><td>20%</td><td>10%</td><td>5%</td><td>50%</td><td>20%</td><td>10%</td><td>5%</td></tr><tr><td colspan="17">LLaVA-1.5-7B (Liu et al., 2024)</td></tr><tr><td>Vanilla</td><td>18.0</td><td>18.0</td><td>18.0</td><td>18.0</td><td>23.9</td><td>23.9</td><td>23.9</td><td>23.9</td><td>54.4</td><td>54.4</td><td>54.4</td><td>54.4</td><td>47.9</td><td>47.9</td><td>47.9</td><td>47.9</td></tr><tr><td>StreamingLLM</td><td>15.2</td><td>14.4</td><td>14.3</td><td>13.4</td><td>17.8</td><td>15.3</td><td>14.4</td><td>13.6</td><td>53.0</td><td>52.5</td><td>52.1</td><td>48.2</td><td>40.5</td><td>35.4</td><td>33.8</td><td>32.5</td></tr><tr><td>SnapKV</td><td>17.9</td><td>16.6</td><td>0.4</td><td>0.4</td><td>23.0</td><td>20.5</td><td>1.7</td><td>1.7</td><td>54.1</td><td>53.6</td><td>6.5</td><td>5.0</td><td>47.3</td><td>44.6</td><td>9.2</td><td>9.2</td></tr><tr><td>H2O</td><td>4.6</td><td>3.7</td><td>3.7</td><td>3.7</td><td>13.5</td><td>7.4</td><td>5.9</td><td>4.5</td><td>13.6</td><td>10.0</td><td>8.9</td><td>7.6</td><td>34.3</td><td>27.6</td><td>24.7</td><td>21.0</td></tr><tr><td>ElasticCache</td><td>2.6</td><td>3.2</td><td>3.4</td><td>3.9</td><td>6.2</td><td>3.7</td><td>3.6</td><td>2.8</td><td>4.7</td><td>6.1</td><td>6.5</td><td>9.3</td><td>20.8</td><td>14.6</td><td>13.1</td><td>9.1</td></tr><tr><td>PrefixKV</td><td>16.0</td><td>2.0</td><td>1.0</td><td>0.9</td><td>23.5</td><td>7.5</td><td>2.9</td><td>1.6</td><td>22.5</td><td>3.9</td><td>1.4</td><td>0.5</td><td>47.5</td><td>19.6</td><td>12.4</td><td>4.8</td></tr><tr><td>TGV-KV</td><td>17.9</td><td>18.0</td><td>15.5</td><td>13.8</td><td>23.6</td><td>21.8</td><td>19.2</td><td>14.3</td><td>54.4</td><td>54.0</td><td>52.3</td><td>31.2</td><td>47.8</td><td>47.4</td><td>44.5</td><td>33.9</td></tr><tr><td colspan="17">LLaVA-NeXT-Mistral-7B (Liu et al., 2024)</td></tr><tr><td>Vanilla</td><td>52.9</td><td>52.9</td><td>52.9</td><td>52.9</td><td>63.7</td><td>63.7</td><td>63.7</td><td>63.7</td><td>63.7</td><td>63.7</td><td>63.7</td><td>63.7</td><td>65.7</td><td>65.7</td><td>65.7</td><td>65.7</td></tr><tr><td>StreamingLLM</td><td>39.0</td><td>30.0</td><td>29.7</td><td>30.0</td><td>46.3</td><td>33.2</td><td>29.5</td><td>28.1</td><td>62.4</td><td>60.8</td><td>60.5</td><td>60.5</td><td>58.0</td><td>47.4</td><td>44.6</td><td>43.1</td></tr><tr><td>SnapKV</td><td>52.4</td><td>49.6</td><td>46.2</td><td>39.2</td><td>63.3</td><td>60.8</td><td>55.1</td><td>44.0</td><td>63.5</td><td>63.0</td><td>62.5</td><td>61.5</td><td>65.3</td><td>63.0</td><td>59.8</td><td>54.9</td></tr><tr><td>H2O</td><td>50.1</td><td>34.5</td><td>27.2</td><td>27.8</td><td>62.9</td><td>44.2</td><td>35.1</td><td>29.4</td><td>52.0</td><td>26.5</td><td>24.2</td><td>24.0</td><td>65.1</td><td>60.4</td><td>53.6</td><td>48.7</td></tr><tr><td>ElasticCache</td><td>44.5</td><td>31.1</td><td>22.7</td><td>18.5</td><td>57.9</td><td>40.3</td><td>27.8</td><td>19.9</td><td>26.3</td><td>24.2</td><td>20.4</td><td>18.9</td><td>61.7</td><td>47.3</td><td>35.5</td><td>31.7</td></tr><tr><td>PrefixKV</td><td>52.2</td><td>38.0</td><td>12.6</td><td>10.4</td><td>62.6</td><td>38.3</td><td>18.8</td><td>16.3</td><td>63.5</td><td>62.8</td><td>52.5</td><td>19.2</td><td>65.1</td><td>58.7</td><td>39.9</td><td>34.5</td></tr><tr><td>TGV-KV</td><td>52.6</td><td>51.8</td><td>51.2</td><td>49.3</td><td>63.7</td><td>62.8</td><td>61.4</td><td>58.5</td><td>63.7</td><td>63.6</td><td>63.6</td><td>63.2</td><td>65.7</td><td>65.5</td><td>65.1</td><td>64.0</td></tr><tr><td colspan="17">LLaVA-OneVision-Qwen2-0.5B (Li et al., 2024a)</td></tr><tr><td>Vanilla</td><td>59.0</td><td>59.0</td><td>59.0</td><td>59.0</td><td>61.9</td><td>61.9</td><td>61.9</td><td>61.9</td><td>47.4</td><td>47.4</td><td>47.4</td><td>47.4</td><td>64.8</td><td>64.8</td><td>64.8</td><td>64.8</td></tr><tr><td>StreamingLLM</td><td>37.7</td><td>28.2</td><td>26.5</td><td>25.8</td><td>39.4</td><td>28.2</td><td>24.4</td><td>23.1</td><td>43.8</td><td>42.1</td><td>41.7</td><td>34.3</td><td>51.5</td><td>39.9</td><td>36.3</td><td>41.4</td></tr><tr><td>SnapKV</td><td>53.1</td><td>49.6</td><td>41.8</td><td>30.4</td><td>52.2</td><td>47.3</td><td>39.7</td><td>30.0</td><td>45.5</td><td>44.2</td><td>43.0</td><td>42.0</td><td>60.4</td><td>54.3</td><td>47.5</td><td>40.1</td></tr><tr><td>H2O</td><td>16.8</td><td>37.8</td><td>33.4</td><td>31.0</td><td>21.5</td><td>41.2</td><td>31.0</td><td>20.7</td><td>19.5</td><td>34.0</td><td>31.3</td><td>30.0</td><td>19.2</td><td>56.0</td><td>51.7</td><td>43.2</td></tr><tr><td>ElasticCache</td><td>29.8</td><td>21.0</td><td>13.4</td><td>9.8</td><td>19.1</td><td>12.4</td><td>9.7</td><td>8.0</td><td>29.4</td><td>29.7</td><td>32.8</td><td>21.4</td><td>41.9</td><td>35.2</td><td>27.9</td><td>31.3</td></tr><tr><td>PrefixKV</td><td>14.0</td><td>42.7</td><td>24.6</td><td>13.7</td><td>35.9</td><td>25.9</td><td>16.5</td><td>8.8</td><td>23.3</td><td>28.8</td><td>29.2</td><td>30.9</td><td>40.7</td><td>48.4</td><td>31.4</td><td>23.6</td></tr><tr><td>TGV-KV</td><td>53.5</td><td>50.7</td><td>47.6</td><td>42.5</td><td>52.5</td><td>48.5</td><td>42.8</td><td>36.4</td><td>45.9</td><td>44.7</td><td>43.7</td><td>43.1</td><td>61.5</td><td>57.0</td><td>52.0</td><td>46.0</td></tr><tr><td colspan="17">Qwen3-VL-4B-Instruct (Bai et al., 2025)</td></tr><tr><td>Vanilla</td><td>84.1</td><td>84.1</td><td>84.1</td><td>84.1</td><td>93.6</td><td>93.6</td><td>93.6</td><td>93.6</td><td>68.5</td><td>68.5</td><td>68.5</td><td>68.5</td><td>80.6</td><td>80.6</td><td>80.6</td><td>80.6</td></tr><tr><td>StreamingLLM</td><td>79.1</td><td>70.6</td><td>67.4</td><td>66.9</td><td>76.0</td><td>62.2</td><td>51.9</td><td>47.0</td><td>66.5</td><td>63.8</td><td>61.5</td><td>59.6</td><td>72.3</td><td>63.8</td><td>56.5</td><td>51.7</td></tr><tr><td>SnapKV</td><td>83.1</td><td>74.7</td><td>23.5</td><td>19.8</td><td>93.5</td><td>90.9</td><td>76.7</td><td>12.3</td><td>68.0</td><td>64.9</td><td>62.1</td><td>48.4</td><td>80.3</td><td>75.8</td><td>58.1</td><td>27.0</td></tr><tr><td>H2O</td><td>82.4</td><td>71.0</td><td>65.8</td><td>63.4</td><td>92.7</td><td>82.9</td><td>67.7</td><td>53.6</td><td>67.5</td><td>64.2</td><td>62.6</td><td>54.8</td><td>80.2</td><td>74.5</td><td>65.2</td><td>55.7</td></tr><tr><td>ElasticCache</td><td>80.7</td><td>74.6</td><td>68.1</td><td>58.8</td><td>78.6</td><td>57.8</td><td>42.9</td><td>33.1</td><td>55.0</td><td>54.7</td><td>54.0</td><td>52.8</td><td>67.7</td><td>57.4</td><td>47.0</td><td>39.8</td></tr><tr><td>PrefixKV</td><td>83.4</td><td>78.1</td><td>72.9</td><td>65.4</td><td>92.2</td><td>73.1</td><td>52.8</td><td>43.0</td><td>67.2</td><td>59.2</td><td>55.5</td><td>49.5</td><td>79.9</td><td>67.9</td><td>53.0</td><td>47.6</td></tr><tr><td>TGV-KV</td><td>83.8</td><td>82.5</td><td>80.2</td><td>67.2</td><td>93.5</td><td>93.1</td><td>91.9</td><td>87.3</td><td>68.4</td><td>67.8</td><td>66.2</td><td>63.1</td><td>80.6</td><td>80.1</td><td>78.1</td><td>70.4</td></tr><tr><td colspan="17">Qwen3-VL-8B-Instruct (Bai et al., 2025)</td></tr><tr><td>Vanilla</td><td>85.3</td><td>85.3</td><td>85.3</td><td>85.3</td><td>95.1</td><td>95.1</td><td>95.1</td><td>95.1</td><td>70.0</td><td>70.0</td><td>70.0</td><td>70.0</td><td>82.1</td><td>82.1</td><td>82.1</td><td>82.1</td></tr><tr><td>StreamingLLM</td><td>80.8</td><td>73.8</td><td>70.8</td><td>72.0</td><td>81.2</td><td>64.8</td><td>54.6</td><td>49.6</td><td>69.2</td><td>66.5</td><td>63.5</td><td>61.3</td><td>76.9</td><td>68.7</td><td>61.2</td><td>54.0</td></tr><tr><td>SnapKV</td><td>85.0</td><td>78.1</td><td>42.3</td><td>39.3</td><td>94.9</td><td>91.2</td><td>80.1</td><td>22.6</td><td>69.8</td><td>67.4</td><td>64.9</td><td>53.8</td><td>81.9</td><td>78.6</td><td>64.0</td><td>40.2</td></tr><tr><td>H2O</td><td>84.0</td><td>78.0</td><td>72.7</td><td>67.6</td><td>94.4</td><td>86.5</td><td>72.0</td><td>58.1</td><td>69.7</td><td>67.6</td><td>62.0</td><td>56.6</td><td>81.6</td><td>77.5</td><td>66.7</td><td>58.5</td></tr><tr><td>ElasticCache</td><td>78.8</td><td>68.5</td><td>67.0</td><td>67.7</td><td>87.1</td><td>71.4</td><td>57.0</td><td>45.3</td><td>48.7</td><td>51.7</td><td>53.4</td><td>53.6</td><td>70.0</td><td>64.7</td><td>57.2</td><td>48.1</td></tr><tr><td>PrefixKV</td><td>84.7</td><td>78.9</td><td>74.2</td><td>71.2</td><td>93.9</td><td>72.7</td><td>50.3</td><td>45.4</td><td>68.8</td><td>59.7</td><td>57.1</td><td>55.7</td><td>81.4</td><td>69.3</td><td>54.0</td><td>49.1</td></tr><tr><td>TGV-KV</td><td>85.3</td><td>84.4</td><td>82.1</td><td>73.1</td><td>95.0</td><td>94.7</td><td>93.2</td><td>88.0</td><td>69.8</td><td>69.2</td><td>67.6</td><td>64.4</td><td>82.1</td><td>81.8</td><td>79.4</td><td>72.0</td></tr></table>

Beyond the LLaVA series, we evaluate Qwen3-VL, which represents the state-of-the-art in multimodal understanding. Specifically, with a budget of 5%, TGV-KV preserves 93.3% performance on DocVQA, and 92.1% on VizWiz on Qwen3-VL-4B. On a larger version with 8B parameters, TGV-KV also achieves superior results, showcasing outstanding scaling ability to models with different sizes.

Text-Dominant Results. Results for image tasks where text dominates are presented in Table 3. On TextCaps, TGV-KV maintains the performance decrease less than 0.5% with both LLaVA and Qwen under a retained budget of 50%. It is notable that most existing methods suffer from catastrophic performance degradation when the budget is strictly limited, while TGV-KV still maintains accuracy under these scenarios, significantly outperforming the secondbest method by a large margin, gaining a 57.4% relative performance boost compared with StreamingLLM with LLaVA. These results highlight TGV-KV’s ability in text-dominant tasks and resource-limited devices.

Video Results. We plot the results on video tasks in the form of Pareto curves in Fig 4. Video-TT mainly evaluates VLM’s reasoning ability and robustness under misleading instructions, which incorporate long context generation and LLM-as-a-Judge assessing. Following the trend in image tasks, TGV-KV maintains performance comparable to the vanilla model baseline even as the retention ratio decreases sharply. In the high-compression setting where only 10% of the vanilla budget is retained, TGV-KV only decreases ≤2 percentage points on Rephased and Wrongly-Led instructions. On the primary subtask, TGV-KV consistently maintains over 95% of the original performance and stays close to the vanilla model under all the budgets.

![](images/ab51bbe35304a7753954d4bd6474270e9e95fa3f3b801c41a4422910770f3407.jpg)

<details>
<summary>line</summary>

| Retention Ratio | StreamingLLM | PrefixKV | SnapKV | H2O | Vanilla | TGV-KV |
| --------------- | ------------ | -------- | ------ | --- | ------- | ------ |
| 1.0             | 20.0         | 20.0     | 20.0   | 20.0 | 20.0    | 20.0   |
| 0.8             | 19.5         | 19.0     | 19.5   | 19.0 | 19.5    | 19.5   |
| 0.6             | 19.0         | 18.5     | 19.0   | 18.5 | 19.0    | 19.0   |
| 0.4             | 18.5         | 17.5     | 18.5   | 17.5 | 18.5    | 18.5   |
| 0.2             | 17.0         | 16.0     | 17.0   | 16.0 | 17.0    | 17.0   |
| 0.0             | 15.0         | 14.0     | 15.0   | 14.0 | 15.0    | 15.0   |
</details>

![](images/7e4f0f0d569df7f777e4e2e9958400f4834c675d115f4b4999c5561f07f46ec5.jpg)

<details>
<summary>line</summary>

| Retention Ratio | StreamingLLM | PrefixKV | SnapKV | H2O | Vanilla | TGV-KV |
| --------------- | ------------ | -------- | ------ | --- | ------- | ------ |
| 1.0             | 20.5         | 20.5     | 20.5   | 20.5| 20.5    | 20.5   |
| 0.8             | 20.0         | 20.0     | 20.0   | 20.0| 20.0    | 20.5   |
| 0.6             | 19.5         | 19.5     | 19.5   | 19.5| 19.5    | 21.0   |
| 0.4             | 19.0         | 19.0     | 19.0   | 19.0| 19.0    | 21.5   |
| 0.2             | 17.0         | 17.0     | 17.0   | 17.0| 17.0    | 19.5   |
| 0.1             | 16.0         | 16.0     | 16.0   | 16.0| 16.0    | 18.5   |
</details>

![](images/4ed01b931596a20f45798ee089027d2ee3711dec28a1ecba7a585b9b780f3ef4.jpg)

<details>
<summary>line</summary>

| Retention Ratio | StreamingLLM | PrefixKV | SnapKV | H2O | Vanilla | TGV-KV |
| --------------- | ------------ | -------- | ------ | --- | ------- | ------ |
| 1.0             | 26.5         | 26.0     | 26.0   | 26.0 | 26.0    | 26.5   |
| 0.8             | 26.5         | 25.5     | 25.5   | 25.0 | 25.5    | 26.5   |
| 0.6             | 26.5         | 25.0     | 25.0   | 24.0 | 25.0    | 27.0   |
| 0.4             | 26.0         | 24.0     | 24.0   | 21.0 | 24.0    | 26.5   |
| 0.2             | 25.0         | 23.0     | 23.0   | 11.0 | 23.0    | 26.0   |
| 0.1             | 24.0         | 21.0     | 21.0   | 9.0  | 21.0    | 25.5   |
| 0.0             | 21.0         | 19.0     | 19.0   | 1.0  | 19.0    | 25.0   |
</details>

Figure 4. Accuracy Results on Video-TT (Zhang et al., 2025b) using Qwen3-VL-4B. Vanilla refers to the original model with the full computation budget. Detailed breakdowns on subtasks and full numerical results are provided in Appendix B.2.

Table 3. Accuracy Results on Text-Dominant Tasks. The results of CIDEr and ROUGE-L are shown in percentage. The full version with more metrics can be found in Appendix B.1. 

<table><tr><td rowspan="2">Methods</td><td colspan="3"> $TextCaps^{CIDEr\uparrow}$ </td><td colspan="3"> $COCO-Cap^{ROUGE-L\uparrow}$ </td></tr><tr><td>50%</td><td>20%</td><td>10%</td><td>50%</td><td>20%</td><td>10%</td></tr><tr><td colspan="7">LLaVA-1.5-7B (Liu et al., 2024)</td></tr><tr><td>Vanilla</td><td>100.3</td><td>100.3</td><td>100.3</td><td>55.1</td><td>55.1</td><td>55.1</td></tr><tr><td>StreamingLLM</td><td>78.4</td><td>53.3</td><td>40.4</td><td>54.3</td><td>52.2</td><td>51.0</td></tr><tr><td>SnapKV</td><td>97.1</td><td>82.3</td><td>0.3</td><td>55.2</td><td>54.6</td><td>0.9</td></tr><tr><td> $H_2O$ </td><td>14.3</td><td>2.4</td><td>0.6</td><td>17.7</td><td>9.8</td><td>8.0</td></tr><tr><td>ElasticCache</td><td>3.3</td><td>1.3</td><td>1.1</td><td>9.6</td><td>8.7</td><td>8.4</td></tr><tr><td>PrefixKV</td><td>94.3</td><td>12.0</td><td>1.4</td><td>34.0</td><td>11.1</td><td>7.9</td></tr><tr><td>TGV-KV</td><td>99.8</td><td>87.8</td><td>63.6</td><td>55.3</td><td>55.0</td><td>52.6</td></tr><tr><td colspan="7">Qwen3-VL-8B-Instruct (Bai et al., 2025)</td></tr><tr><td>Vanilla</td><td>33.6</td><td>33.6</td><td>33.6</td><td>42.0</td><td>42.0</td><td>42.0</td></tr><tr><td>StreamingLLM</td><td>34.9</td><td>32.2</td><td>24.5</td><td>41.7</td><td>39.6</td><td>35.3</td></tr><tr><td>SnapKV</td><td>34.1</td><td>34.4</td><td>27.2</td><td>40.4</td><td>36.3</td><td>35.7</td></tr><tr><td> $H_2O$ </td><td>32.3</td><td>34.1</td><td>28.8</td><td>41.4</td><td>39.3</td><td>37.9</td></tr><tr><td>ElasticCache</td><td>27.4</td><td>26.6</td><td>20.1</td><td>39.9</td><td>38.1</td><td>34.6</td></tr><tr><td>PrefixKV</td><td>32.6</td><td>27.7</td><td>14.4</td><td>41.7</td><td>40.6</td><td>36.2</td></tr><tr><td>TGV-KV</td><td>33.5</td><td>34.9</td><td>30.9</td><td>42.0</td><td>41.6</td><td>39.5</td></tr></table>

# 4.3. Efficiency Results

Memory and Speed. We evaluate the computational efficiency of TGV-KV with different context lengths, with results on memory, per-token latency, and throughput in Table 4. Notably, with an extreme retention rate of 5%, TGV-KV reduces the memory consumption of KV cache size from 3.91 GB to a mere 0.20 GB under an 8k context. This substantial reduction effectively alleviates the memory bottleneck often encountered during the deployment of VLM. Furthermore, while FlashAttention (Dao, 2024) significantly accelerates the prefill stage by optimizing fullsequence attention computation, it provides limited or even negative speedup during decode due to the inherently low parallelism of single-token queries. These results showcase the efficiency of TGV-KV towards long context generation.

Table 4. Memory and Speed Analyses. Evaluated with LLaVA-1.5-7B on an NVIDIA A800 GPU. Memory refers to the KV cache consumption. Latency denotes the average time per decoding step. Throughput accounts for both prefill and decoding stages. 

<table><tr><td>Settings</td><td>Memory (GB)↓</td><td>Latency (ms)↓</td><td>Throughput (tokens/s)↑</td></tr><tr><td colspan="4">Context Length=8k tokens</td></tr><tr><td>Vanilla-Eager</td><td>3.91</td><td>40.1±1.5</td><td>23.6±0.1</td></tr><tr><td>Vanilla-FA2</td><td>3.91</td><td>44.1±0.9</td><td>25.3±0.5</td></tr><tr><td>TGV-KV/50%</td><td>1.95</td><td>29.4±0.4 (-26.7%)</td><td>26.2±0.4 (+11.0%)</td></tr><tr><td>TGV-KV/10%</td><td>0.39</td><td>30.9±3.5 (-22.9%)</td><td>29.8±0.3 (+26.3%)</td></tr><tr><td>TGV-KV/5%</td><td>0.20</td><td>27.9±0.4 (-30.4%)</td><td>31.0±0.1 (+31.4%)</td></tr><tr><td colspan="4">Context Length=16k tokens</td></tr><tr><td>Vanilla-Eager</td><td>7.81</td><td>57.3±0.7</td><td>21.1±0.3</td></tr><tr><td>Vanilla-FA2</td><td>7.81</td><td>57.4±0.3</td><td>21.3±0.3</td></tr><tr><td>TGV-KV/50%</td><td>3.91</td><td>33.6±1.2 (-41.4%)</td><td>24.1±0.1 (+14.2%)</td></tr><tr><td>TGV-KV/10%</td><td>0.78</td><td>28.1±0.1 (-51.0%)</td><td>30.8±0.3 (+46.0%)</td></tr><tr><td>TGV-KV/5%</td><td>0.39</td><td>27.9±0.4 (-51.3%)</td><td>32.2±0.3 (+52.6%)</td></tr></table>

# 4.4. Ablation Studies

To systematically validate the effectiveness of each component, we conduct ablation studies by evaluating all combinations of our three proposed modules. As reported in Table 5, we can draw three key conclusions on the role of each component. (1) TWR consistently yields superior importance criteria, as almost all the sets utilizing TWR for the importance score surpass those simply using naive self-attention. (2) TPR is crucial under extreme KV cache budgets, since TPR brings much better results under 5% budget, especially when TWR is absent. (3) TVB provides supplementary performance gains when the budget is sufficient, assisting in allocating extra budget to key layers. The ablation studies inside each submodule have been discussed in Table 1. These results corroborate the design philosophy of TGV-KV, underscoring the necessity of tailored strategies to bridge the modality gap in VLMs, rather than a straightforward transplantation of prevailing LLM paradigms.

![](images/3682afc5d10d6919ebd996a4bc8952441057c42d7aa326b7cbc21f46fa9e2343.jpg)

Figure 5. Visualization of Retained KV. We plot the retained KV of one sample from POPE (Li et al., 2023). The visualization is from LLaVA (32 layers) with an overall retention of 3%. The patches most related to the text instruction are circled in violet. Evicted KVs are masked or labelled in gray. More visualizations are in Appendix C.2.   
Table 5. Ablation Studies. Results are carried out with LLaVA-1.5-7B. Importance Score stands for the criterion to judge KV pair importance, and tokens with lower scores are first to be evicted. Ablation studies inside each submodule are presented in Table 1. 

<table><tr><td rowspan="2" colspan="3">TVB TWR TPR</td><td colspan="2">ChartQA↑</td><td colspan="2">COCO-Cap↑</td></tr><tr><td>20%</td><td>5%</td><td>20%</td><td>5%</td></tr><tr><td colspan="7">Attention Sum → Importance Score</td></tr><tr><td></td><td></td><td></td><td>3.8 (-14.2)</td><td>4.5 (-9.3)</td><td>20.0 (-35.0)</td><td>18.8 (-30.3)</td></tr><tr><td></td><td></td><td>✓</td><td>17.4 (-0.6)</td><td>13.6 (-0.2)</td><td>54.8 (-0.2)</td><td>47.9 (-1.4)</td></tr><tr><td>✓</td><td></td><td></td><td>15.0 (-3.0)</td><td>1.1 (-12.7)</td><td>54.9 (-0.1)</td><td>13.4 (-35.7)</td></tr><tr><td>✓</td><td></td><td>✓</td><td>17.3 (-0.7)</td><td>13.0 (-0.8)</td><td>54.8 (-0.2)</td><td>48.9 (-0.2)</td></tr><tr><td colspan="7">Text-Weighted Ranking → Importance Score</td></tr><tr><td></td><td>✓</td><td></td><td>17.8 (-0.2)</td><td>13.5 (-0.3)</td><td>55.1 (+0.1)</td><td>48.6 (-0.5)</td></tr><tr><td></td><td>✓</td><td>✓</td><td>17.8 (-0.2)</td><td>13.8 (-0.0)</td><td>55.1 (+0.1)</td><td>49.1 (-0.0)</td></tr><tr><td>✓</td><td>✓</td><td></td><td>18.0 (-0.0)</td><td>13.7 (-0.1)</td><td>55.0 (-0.0)</td><td>46.0 (-3.1)</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>18.0</td><td>13.8</td><td>55.0</td><td>49.1</td></tr></table>

Table 6. Accuracy Results on Token Pruning and KV Eviction. We evaluate TGV-KV against recent token pruning methods. The number denotes the equivalent retained token or KV size. 

<table><tr><td rowspan="2">Methods</td><td colspan="3">MME↑</td><td colspan="3">POPE↑</td><td colspan="3">GQA↑</td></tr><tr><td>128</td><td>64</td><td>32</td><td>128</td><td>64</td><td>32</td><td>128</td><td>64</td><td>32</td></tr><tr><td>Vanilla</td><td>1781</td><td>1781</td><td>1781</td><td>84.6</td><td>84.6</td><td>84.6</td><td>60.7</td><td>60.7</td><td>60.7</td></tr><tr><td>DivPrune</td><td>1724</td><td>1636</td><td>1600</td><td>86.9</td><td>85.7</td><td>81.4</td><td>59.2</td><td>57.5</td><td>55.3</td></tr><tr><td>VisionZip</td><td>1755</td><td>1693</td><td>1585</td><td>83.1</td><td>76.9</td><td>70.4</td><td>57.6</td><td>55.1</td><td>52.3</td></tr><tr><td>VisPruner</td><td>1766</td><td>1696</td><td>1552</td><td>84.5</td><td>78.3</td><td>73.3</td><td>58.5</td><td>55.3</td><td>52.8</td></tr><tr><td>CDPruner</td><td>1746</td><td>1707</td><td>1692</td><td>87.0</td><td>87.0</td><td>87.5</td><td>59.8</td><td>58.8</td><td>57.4</td></tr><tr><td>TGV-KV</td><td>1781</td><td>1781</td><td>1781</td><td>84.6</td><td>84.6</td><td>84.6</td><td>60.7</td><td>60.6</td><td>60.4</td></tr></table>

# 4.5. Discussions

Why Compress KV Rather Than Token? We prioritize KV cache eviction over token pruning for two main reasons. (1) Superior accuracy under the same budget. When evicting a token’s KV in one certain layer, its information is still visible in subsequent layers where the corresponding KV

is not evicted. However, once a token is pruned, its unextracted information can no longer be accessed. As shown in Table 6, under a same budget, TGV-KV barely degrades the VQA accuracy or increases hallucination. (2) System efficiency gains. Although token pruning reduces computation in both prefill and decoding, KV eviction mainly accelerates decoding. The overall latency is dominated by the iterative decoding stage rather than the one-time prefill computation (Liu et al., 2025b). Consequently, token pruning offers limited practical acceleration while significantly harming model quality. TGV-KV provides a more favorable trade-off between efficiency and performance in deployment.

Visualizations of Evicted KVs. We visualize the retained text and vision KV in different layers in Fig. 5. TVB tends to assign more budget to shallow layers, indicating that cross-modality interaction is intense in these layers, aligning with previous studies (Chen et al., 2024a; Xing et al., 2025). In some middle layers with very limited budget, all the vision KV and a few text KV are evicted, while the key part of the text instruction, e.g., “tennis ball” in this example, is reserved across all the layers. Besides, the image patches most related to these dominant text tokens are also accurately preserved in shallow layers. These findings support the rationale behind TWR and unveil its potential in visual grounding and hallucination alleviation.

# 5. Conclusion

In this paper, we take a systematic study of the multimodal attention pattern in VLMs and concludes three key observations vital to multimodal KV cache eviction design. Based on the analyses, we overcome the modality gap in VLM and propose a robust KV cache eviction approach, TGV-KV, which fully leverages the text to guide vision KV eviction. TGV-KV consists of three modules, where TVB allocates layer-wise budgets, TWR evaluates the KV importance score, and TPR preserves crucial text information. We evaluate TGV-KV across multiple models and benchmarks, proving its effectiveness in multimodal KV eviction. Our conclusions and observations are universal, and we believe subsequent researches can draw inspiration from our study.

# Acknowledgement

This work is partially sponsored by the National Natural Science Foundation of China under Grant 62306084 and U23B2051, Shenzhen College Stability Support Plan under Grant GXWD20231128102243003, and Shenzhen Science and Technology Program under Grant ZDSYS20230626091203008 and KJZD20230923115113026.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

# References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for fewshot learning. Advances in neural information processing systems, 35:23716–23736, 2022.   
Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.   
Cai, Z., Zhang, Y., Gao, B., Liu, Y., Li, Y., Liu, T., Lu, K., Xiong, W., Dong, Y., Hu, J., et al. Pyramidkv: Dynamic kv cache compression based on pyramidal information funneling. arXiv preprint arXiv:2406.02069, 2024.   
Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., and Chang, B. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. In European Conference on Computer Vision, pp. 19–35. Springer, 2024a.   
Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. Internvl:

Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198, 2024b.

Dao, T. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., and Houlsby, N. An image is worth 16x16 words: Transformers for image recognition at scale. ICLR, 2021.

Feng, Y., Lv, J., Cao, Y., Xie, X., and Zhou, S. K. Adakv: Optimizing kv cache eviction by adaptive budget allocation for efficient llm inference. arXiv preprint arXiv:2407.11550, 2024.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Gurari, D., Li, Q., Stangl, A. J., Guo, A., Lin, C., Grauman, K., Luo, J., and Bigham, J. P. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 3608–3617, 2018.

Huang, K., Zou, H., Wang, B., Xi, Y., Xie, Z., and Wang, H. Aircache: Activating inter-modal relevancy kv cache compression for efficient large vision-language model inference. arXiv preprint arXiv:2503.23956, 2025.

Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, X., and Wen, J.-R. Evaluating object hallucination in large visionlanguage models. In Bouamor, H., Pino, J., and Bali, K. (eds.), Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 292–305, Singapore, December 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main. 20. URL https://aclanthology.org/2023. emnlp-main.20/.

Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A., Ye, H., Cai, T., Lewis, P., and Chen, D. Snapkv: Llm knows what you are looking for before generation. Advances in Neural Information Processing Systems, 37:22947– 22970, 2024b.

Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-LLaVA: Learning united visual representation by alignment before projection. In Al-Onaizan, Y., Bansal, M., and Chen, Y.-N. (eds.), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 5971–5984, Miami, Florida, USA, November 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main. 342. URL https://aclanthology.org/2024. emnlp-main.342/.   
Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., and Dollar, P. Microsoft coco: Common objects in context, ´ 2015.   
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.   
Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26296–26306, 2024.   
Liu, J., Du, F., Zhu, G., Lian, N., Li, J., and Chen, B. Hiprune: Training-free visual token pruning via hierarchical attention in vision-language models. arXiv preprint arXiv:2508.00553, 2025a.   
Liu, Z., Liu, B., Wang, J., Dong, Y., Chen, G., Rao, Y., Krishna, R., and Lu, J. Efficient inference of vision instruction-following models with elastic cache. In Leonardis, A., Ricci, E., Roth, S., Russakovsky, O., Sattler, T., and Varol, G. (eds.), Computer Vision – ECCV 2024, pp. 54–69, Cham, 2025b. Springer Nature Switzerland. ISBN 978-3-031-72643-9.   
Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. In The 36th Conference on Neural Information Processing Systems (NeurIPS), 2022.   
Masry, A., Long, D. X., Tan, J. Q., Joty, S., and Hoque, E. ChartQA: A benchmark for question answering about charts with visual and logical reasoning. In Muresan, S., Nakov, P., and Villavicencio, A. (eds.), Findings of the Association for Computational Linguistics: ACL 2022, pp. 2263–2279, Dublin, Ireland, May 2022. Association for Computational Linguistics. doi: 10.18653/v1/2022. findings-acl.177. URL https://aclanthology. org/2022.findings-acl.177/.   
Mathew, M., Karatzas, D., and Jawahar, C. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2200–2209, 2021.

Neo, C., Ong, L., Torr, P., Geva, M., Krueger, D., and Barez, F. Towards interpreting visual information processing in vision-language models. In Yue, Y., Garg, A., Peng, N., Sha, F., and Yu, R. (eds.), International Conference on Representation Learning, volume 2025, pp. 57172– 57189, 2025.   
Qin, Z., Cao, Y., Lin, M., Hu, W., Fan, S., Cheng, K., Lin, W., and Li, J. CAKE: Cascading and adaptive KV cache eviction with layer preferences. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum? id=EQgEMAD4kv.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.   
Sidorov, O., Hu, R., Rohrbach, M., and Singh, A. Textcaps: a dataset for image captioning with reading comprehension. In European conference on computer vision, pp. 742–758. Springer, 2020.   
Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.   
Team, G., Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.   
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
Wang, A., Chen, H., Tan, J., Zhang, K., Cai, X., Lin, Z., Han, J., and Ding, G. PrefixKV: Adaptive prefix KV cache is what vision instruction-following models need for efficient generation. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025a. URL https://openreview.net/forum? id=tDG6bY48ch.   
Wang, J., Liu, Z., Rao, Y., and Lu, J. Sparsemm: Head sparsity emerges from visual concept responses in mllms. In Proceedings of the IEEE/CVF international conference on computer vision, 2025b.   
Xiao, G., Tian, Y., Chen, B., Han, S., and Lewis, M. Efficient streaming language models with attention sinks. In Kim, B., Yue, Y., Chaudhuri, S., Fragkiadaki, K., Khan,

M., and Sun, Y. (eds.), International Conference on Representation Learning, volume 2024, pp. 21875–21895, 2024.   
Xing, L., Huang, Q., Dong, X., Lu, J., Zhang, P., Zang, Y., Cao, Y., He, C., Wang, J., Wu, F., and Lin, D. Conical visual concentration for efficient large vision-language models. In Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR), pp. 14593–14603, June 2025.   
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.   
Yang, S., Chen, Y., Tian, Z., Wang, C., Li, J., Yu, B., and Jia, J. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 19792–19802, 2025b.   
Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 11975–11986, 2023.   
Zhang, K., Li, B., Zhang, P., Pu, F., Cahyono, J. A., Hu, K., Liu, S., Zhang, Y., Yang, J., Li, C., and Liu, Z. Lmms-eval: Reality check on the evaluation of large multimodal models, 2024. URL https://arxiv.org/ abs/2407.12772.   
Zhang, Q., Liu, M., Li, L., Lu, M., Zhang, Y., Pan, J., She, Q., and Zhang, S. Beyond attention or similarity: Maximizing conditional diversity for token pruning in mllms. arXiv preprint arXiv:2506.10967, 2025a.   
Zhang, Y., Chew, Y., Dong, Y., Leo, A., Hu, B., and Liu, Z. Towards video thinking test: A holistic benchmark for advanced video reasoning and understanding. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20626–20636, 2025b.   
Zhang, Y., Fan, C.-K., Ma, J., Zheng, W., Huang, T., Cheng, K., Gudovskiy, D., Okuno, T., Nakata, Y., Keutzer, K., et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. In International Conference on Machine Learning, 2025c.   
Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., Re, C., Barrett, C., et al. H2o: ´ Heavy-hitter oracle for efficient generative inference of large language models. Advances in Neural Information Processing Systems, 36:34661–34710, 2023.

# A. Experiment Details

# A.1. Dataset Description

# A.1.1. IMAGE TASKS

# Vision-Dominant Tasks.

ChartQA (Masry et al., 2022). ChartQA is a QA dataset on charts. It mainly evaluates the model’s ability in visual feature extraction and logical reasoning. It contains two subsets, i.e., human-authored and machine-generated. Humanauthored subset includes ∼9.6k high-quality human-written QA pairs, covering ∼4.8k charts. Machine-generated subset uses a T5 model to generate ∼23k QA pairs for ∼17k charts based on human-written chart captions. In our evaluation, we adopt both subsets and average them to get an overall score.   
• DocVQA (Mathew et al., 2021). The main goal of DocVQA is to test the model’s capability in understanding visual cues, identifying text in different format and extractive QA. The whole dataset contains ∼12k images and ∼50k QA pairs. We evaluate on the validation set with ∼1.2k images and ∼5k QA pairs. We report the results in Average Normalized Levenshtein Similarity (ANLS), a metric designed for QA tasks.   
• VizWiz (Gurari et al., 2018). VizWiz is a VQA task constructed with true questions and images from blind people. Beyond the traditional VQA task, VizWiz further includes an answerability prediction, which requires the model to first judge whether the question can be answered, since some images are blurry or have no meaningful objects. The whole dataset includes ∼31k items, and we evaluate on the validation set with a size of ∼3.1k.   
• TextVQA (Singh et al., 2019). TextVQA tests the model in reading image and comprehension. Common VQA tasks feature relatively fewer questions on reading the image, while TextVQA challenges include more OCR and reasoning ability. To answer the questions, the model needs to locate the related text and decide whether directly use the seen text as an answer or think before answering. The whole dataset features ∼45k questions, and we adopt the 5k validation set.

# Text-Dominant Tasks.

• TextCaps (Sidorov et al., 2020). TextCaps is a caption challenge that tests the model in reading text in the image and generating continuous text, which requires OCR capability, text understanding, and paraphrasing ability. It contains ∼28k images and ∼145k captions, and we conduct all the evaluation on the validation set, with ∼3.1k items. It uses multiple metrics, including Bleu, METEOR, ROUGE-L, and CIDEr. We report the results in CIDEr in our main paper.   
• COCO-Caption (Lin et al., 2015). COCO dataset is a large-scale computer vision dataset that comprises object detection, instance segmentation, semantic segmentation, image caption, and so on. In our study, we mainly evaluate the model’s ability in text-dominant tasks and only use the caption task. We utilize the validation set from COCO-2017, which features 5k data items on captioning. We report ROUGE-L in the main paper.

# A.1.2. VIDEO TASKS

• Video-TT (Zhang et al., 2025b). Video-TT mainly evaluates the model in two aspects, robustness and correctness. There are 1k videos selected from YouTube Shorts within 65 seconds, paired with one primary open-ended question and four adversarial questions. The adversarial questions include rephrased questions, questions with correct leads, questions with wrong leads, and multiple-choice questions. Most of the videos include complex visual transformation or story plot, thus making the benchmark challenging even for closed-sourced commercial models. The answers are sent to LLM-as-a-Judge to get a final score.

# A.2. Implementation Details

The max vision tokens for Qwen3-VL is set to 1024, and the vision aspect ratio is set to 2 for LLaVA-OV due to several extremely large images in the dataset, which may cause Out-of-Memory during evaluation. Following the common practice, we retain the first 4 and last 1 tokens for all the methods to avoid performance collision. We also manually set the max generated tokens to 32 for all the tasks, which is longer than any answer response, to shorten the evaluation time, because some methods cause the loss of EOS token and the generation falls into a dead loop. For Video-TT, we use DeepSeek-V3.2 with thinking mode off for LLM-as-a-Judge evaluation.

Table 7. Full performance comparison on text-dominant benchmarks. We report BLEU-x (B-x), CIDEr (C), METEOR (M), and ROUGE-L (R). Note that CIDEr scores are reported as raw values. 

<table><tr><td rowspan="2">Methods</td><td colspan="7">Retain 50% KV</td><td colspan="7">Retain 20% KV</td><td colspan="7">Retain 10% KV</td></tr><tr><td>B-1</td><td>B-2</td><td>B-3</td><td>B-4</td><td>C</td><td>M</td><td>R</td><td>B-1</td><td>B-2</td><td>B-3</td><td>B-4</td><td>C</td><td>M</td><td>R</td><td>B-1</td><td>B-2</td><td>B-3</td><td>B-4</td><td>C</td><td>M</td><td>R</td></tr><tr><td colspan="22">TextCaps (Sidorov et al., 2020) w/ LLaVA-1.5 (Liu et al., 2024)</td></tr><tr><td>Vanilla</td><td>0.71</td><td>0.53</td><td>0.38</td><td>0.27</td><td>1.00</td><td>0.23</td><td>0.47</td><td>0.71</td><td>0.53</td><td>0.38</td><td>0.27</td><td>1.00</td><td>0.23</td><td>0.47</td><td>0.71</td><td>0.53</td><td>0.38</td><td>0.27</td><td>1.00</td><td>0.23</td><td>0.47</td></tr><tr><td>StreamingLLM</td><td>0.68</td><td>0.49</td><td>0.34</td><td>0.24</td><td>0.78</td><td>0.21</td><td>0.44</td><td>0.64</td><td>0.44</td><td>0.30</td><td>0.19</td><td>0.53</td><td>0.19</td><td>0.41</td><td>0.61</td><td>0.41</td><td>0.26</td><td>0.17</td><td>0.40</td><td>0.17</td><td>0.39</td></tr><tr><td>SnapKV</td><td>0.71</td><td>0.52</td><td>0.37</td><td>0.26</td><td>0.97</td><td>0.23</td><td>0.47</td><td>0.68</td><td>0.49</td><td>0.34</td><td>0.24</td><td>0.82</td><td>0.22</td><td>0.45</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.01</td></tr><tr><td>H2O</td><td>0.08</td><td>0.05</td><td>0.03</td><td>0.02</td><td>0.14</td><td>0.08</td><td>0.17</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.02</td><td>0.04</td><td>0.09</td><td>0.03</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.04</td><td>0.07</td></tr><tr><td>ElasticCache</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.01</td><td>0.03</td><td>0.05</td><td>0.09</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.04</td><td>0.07</td><td>0.04</td><td>0.01</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.03</td><td>0.07</td></tr><tr><td>PrefixKV</td><td>0.70</td><td>0.52</td><td>0.38</td><td>0.27</td><td>0.94</td><td>0.22</td><td>0.47</td><td>0.06</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.12</td><td>0.06</td><td>0.14</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.04</td><td>0.08</td></tr><tr><td>TGV-KV</td><td>0.71</td><td>0.53</td><td>0.38</td><td>0.27</td><td>1.00</td><td>0.23</td><td>0.47</td><td>0.70</td><td>0.51</td><td>0.36</td><td>0.25</td><td>0.88</td><td>0.22</td><td>0.45</td><td>0.64</td><td>0.45</td><td>0.30</td><td>0.20</td><td>0.64</td><td>0.19</td><td>0.42</td></tr><tr><td colspan="22">COCO-Caption (Lin et al., 2015) w/ LLaVA-1.5 (Liu et al., 2024)</td></tr><tr><td>Vanilla</td><td>0.73</td><td>0.56</td><td>0.41</td><td>0.29</td><td>1.08</td><td>0.28</td><td>0.55</td><td>0.73</td><td>0.56</td><td>0.41</td><td>0.29</td><td>1.08</td><td>0.28</td><td>0.55</td><td>0.73</td><td>0.56</td><td>0.41</td><td>0.29</td><td>1.08</td><td>0.28</td><td>0.55</td></tr><tr><td>StreamingLLM</td><td>0.73</td><td>0.56</td><td>0.40</td><td>0.29</td><td>1.04</td><td>0.27</td><td>0.54</td><td>0.71</td><td>0.53</td><td>0.38</td><td>0.26</td><td>0.93</td><td>0.25</td><td>0.52</td><td>0.70</td><td>0.52</td><td>0.37</td><td>0.25</td><td>0.86</td><td>0.24</td><td>0.51</td></tr><tr><td>SnapKV</td><td>0.74</td><td>0.57</td><td>0.42</td><td>0.30</td><td>1.09</td><td>0.28</td><td>0.55</td><td>0.73</td><td>0.56</td><td>0.41</td><td>0.29</td><td>1.06</td><td>0.27</td><td>0.55</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.00</td><td>0.01</td><td>0.01</td></tr><tr><td>H2O</td><td>0.07</td><td>0.05</td><td>0.03</td><td>0.02</td><td>0.15</td><td>0.09</td><td>0.18</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.01</td><td>0.03</td><td>0.05</td><td>0.10</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.04</td><td>0.08</td></tr><tr><td>ElasticCache</td><td>0.04</td><td>0.03</td><td>0.01</td><td>0.01</td><td>0.03</td><td>0.05</td><td>0.10</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.04</td><td>0.09</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.04</td><td>0.08</td></tr><tr><td>PrefixKV</td><td>0.15</td><td>0.11</td><td>0.08</td><td>0.05</td><td>0.63</td><td>0.14</td><td>0.34</td><td>0.05</td><td>0.03</td><td>0.02</td><td>0.01</td><td>0.06</td><td>0.06</td><td>0.11</td><td>0.04</td><td>0.02</td><td>0.01</td><td>0.00</td><td>0.01</td><td>0.04</td><td>0.08</td></tr><tr><td>TGV-KV</td><td>0.74</td><td>0.57</td><td>0.42</td><td>0.30</td><td>1.09</td><td>0.28</td><td>0.55</td><td>0.74</td><td>0.57</td><td>0.41</td><td>0.29</td><td>1.07</td><td>0.28</td><td>0.55</td><td>0.71</td><td>0.54</td><td>0.39</td><td>0.27</td><td>0.95</td><td>0.26</td><td>0.53</td></tr><tr><td colspan="22">TextCaps (Sidorov et al., 2020) w/ Qwen3-VL-8B-Instruct (Bai et al., 2025)</td></tr><tr><td>Vanilla</td><td>0.47</td><td>0.30</td><td>0.20</td><td>0.14</td><td>0.34</td><td>0.25</td><td>0.39</td><td>0.47</td><td>0.30</td><td>0.20</td><td>0.14</td><td>0.34</td><td>0.25</td><td>0.39</td><td>0.47</td><td>0.30</td><td>0.20</td><td>0.14</td><td>0.34</td><td>0.25</td><td>0.39</td></tr><tr><td>StreamingLLM</td><td>0.46</td><td>0.29</td><td>0.19</td><td>0.12</td><td>0.35</td><td>0.24</td><td>0.38</td><td>0.44</td><td>0.27</td><td>0.17</td><td>0.11</td><td>0.32</td><td>0.21</td><td>0.36</td><td>0.40</td><td>0.23</td><td>0.14</td><td>0.08</td><td>0.25</td><td>0.18</td><td>0.33</td></tr><tr><td>SnapKV</td><td>0.47</td><td>0.29</td><td>0.20</td><td>0.13</td><td>0.34</td><td>0.24</td><td>0.38</td><td>0.45</td><td>0.29</td><td>0.18</td><td>0.11</td><td>0.34</td><td>0.22</td><td>0.36</td><td>0.40</td><td>0.24</td><td>0.14</td><td>0.09</td><td>0.27</td><td>0.17</td><td>0.31</td></tr><tr><td>H2O</td><td>0.47</td><td>0.30</td><td>0.20</td><td>0.14</td><td>0.32</td><td>0.25</td><td>0.39</td><td>0.44</td><td>0.28</td><td>0.19</td><td>0.14</td><td>0.34</td><td>0.23</td><td>0.39</td><td>0.40</td><td>0.23</td><td>0.14</td><td>0.10</td><td>0.29</td><td>0.19</td><td>0.36</td></tr><tr><td>ElasticCache</td><td>0.44</td><td>0.28</td><td>0.19</td><td>0.12</td><td>0.27</td><td>0.24</td><td>0.37</td><td>0.42</td><td>0.26</td><td>0.16</td><td>0.10</td><td>0.27</td><td>0.21</td><td>0.35</td><td>0.39</td><td>0.23</td><td>0.13</td><td>0.08</td><td>0.20</td><td>0.18</td><td>0.32</td></tr><tr><td>PrefixKV</td><td>0.46</td><td>0.30</td><td>0.20</td><td>0.13</td><td>0.33</td><td>0.25</td><td>0.38</td><td>0.42</td><td>0.26</td><td>0.16</td><td>0.10</td><td>0.28</td><td>0.22</td><td>0.35</td><td>0.35</td><td>0.20</td><td>0.11</td><td>0.06</td><td>0.14</td><td>0.16</td><td>0.30</td></tr><tr><td>TGV-KV</td><td>0.47</td><td>0.30</td><td>0.20</td><td>0.14</td><td>0.33</td><td>0.25</td><td>0.39</td><td>0.46</td><td>0.29</td><td>0.19</td><td>0.13</td><td>0.35</td><td>0.24</td><td>0.38</td><td>0.41</td><td>0.25</td><td>0.16</td><td>0.10</td><td>0.31</td><td>0.21</td><td>0.36</td></tr><tr><td colspan="22">COCO-Caption (Lin et al., 2015) w/ Qwen3-VL-8B-Instruct (Bai et al., 2025)</td></tr><tr><td>Vanilla</td><td>0.48</td><td>0.31</td><td>0.19</td><td>0.12</td><td>0.27</td><td>0.25</td><td>0.42</td><td>0.48</td><td>0.31</td><td>0.19</td><td>0.12</td><td>0.27</td><td>0.25</td><td>0.42</td><td>0.48</td><td>0.31</td><td>0.19</td><td>0.12</td><td>0.27</td><td>0.25</td><td>0.42</td></tr><tr><td>StreamingLLM</td><td>0.48</td><td>0.31</td><td>0.19</td><td>0.11</td><td>0.29</td><td>0.25</td><td>0.42</td><td>0.45</td><td>0.28</td><td>0.17</td><td>0.10</td><td>0.29</td><td>0.22</td><td>0.40</td><td>0.39</td><td>0.23</td><td>0.13</td><td>0.07</td><td>0.18</td><td>0.19</td><td>0.35</td></tr><tr><td>SnapKV</td><td>0.47</td><td>0.29</td><td>0.18</td><td>0.11</td><td>0.29</td><td>0.24</td><td>0.40</td><td>0.42</td><td>0.27</td><td>0.16</td><td>0.10</td><td>0.25</td><td>0.17</td><td>0.36</td><td>0.42</td><td>0.27</td><td>0.16</td><td>0.10</td><td>0.24</td><td>0.17</td><td>0.36</td></tr><tr><td>H2O</td><td>0.47</td><td>0.30</td><td>0.18</td><td>0.12</td><td>0.23</td><td>0.25</td><td>0.41</td><td>0.40</td><td>0.25</td><td>0.18</td><td>0.09</td><td>0.25</td><td>0.23</td><td>0.39</td><td>0.46</td><td>0.25</td><td>0.16</td><td>0.10</td><td>0.26</td><td>0.20</td><td>0.38</td></tr><tr><td>ElasticCache</td><td>0.45</td><td>0.28</td><td>0.17</td><td>0.10</td><td>0.21</td><td>0.24</td><td>0.40</td><td>0.42</td><td>0.26</td><td>0.16</td><td>0.09</td><td>0.23</td><td>0.22</td><td>0.38</td><td>0.38</td><td>0.22</td><td>0.13</td><td>0.07</td><td>0.16</td><td>0.19</td><td>0.35</td></tr><tr><td>PrefixKV</td><td>0.48</td><td>0.30</td><td>0.19</td><td>0.12</td><td>0.28</td><td>0.25</td><td>0.42</td><td>0.46</td><td>0.29</td><td>0.18</td><td>0.11</td><td>0.30</td><td>0.23</td><td>0.41</td><td>0.40</td><td>0.24</td><td>0.14</td><td>0.08</td><td>0.19</td><td>0.20</td><td>0.36</td></tr><tr><td>TGV-KV</td><td>0.48</td><td>0.31</td><td>0.19</td><td>0.12</td><td>0.27</td><td>0.25</td><td>0.42</td><td>0.47</td><td>0.30</td><td>0.19</td><td>0.11</td><td>0.29</td><td>0.25</td><td>0.42</td><td>0.44</td><td>0.27</td><td>0.17</td><td>0.10</td><td>0.27</td><td>0.22</td><td>0.40</td></tr></table>

Table 8. Extended Accuracy Results on Visual KV Eviction Methods. Vanilla denotes the vanilla model with full KV. The percentage denotes the vision KV budget retention ratio, while all the text KV are retained. The best result of each set is marked in bold. 

<table><tr><td rowspan="2">Methods</td><td colspan="4"> $ChartQA^{Relaxed\ Acc.\uparrow}$ </td><td colspan="4"> $DocVQA^{ANLS\uparrow}$ </td><td colspan="4"> $VizWiz^{Acc.\uparrow}$ </td><td colspan="4"> $TextVQA^{Acc.\uparrow}$ </td></tr><tr><td>50%</td><td>20%</td><td>10%</td><td>5%</td><td>50%</td><td>20%</td><td>10%</td><td>5%</td><td>50%</td><td>20%</td><td>10%</td><td>5%</td><td>50%</td><td>20%</td><td>10%</td><td>5%</td></tr><tr><td colspan="17">LLaVA-1.5-7B (Liu et al., 2024)</td></tr><tr><td>Vanilla</td><td>18.0</td><td>18.0</td><td>18.0</td><td>18.0</td><td>23.9</td><td>23.9</td><td>23.9</td><td>23.9</td><td>54.4</td><td>54.4</td><td>54.4</td><td>54.4</td><td>47.9</td><td>47.9</td><td>47.9</td><td>47.9</td></tr><tr><td>AirCache</td><td>17.6</td><td>17.0</td><td>15.6</td><td>15.0</td><td>23.0</td><td>20.6</td><td>18.6</td><td>16.8</td><td>54.3</td><td>53.9</td><td>53.5</td><td>52.9</td><td>47.8</td><td>46.1</td><td>43.8</td><td>41.1</td></tr><tr><td>TGV-KV</td><td>17.8</td><td>18.0</td><td>17.6</td><td>16.5</td><td>23.7</td><td>22.6</td><td>21.2</td><td>19.7</td><td>54.4</td><td>54.2</td><td>53.9</td><td>53.5</td><td>48.0</td><td>47.7</td><td>47.0</td><td>45.7</td></tr><tr><td colspan="17">Qwen3-VL-8B-Instruct (Bai et al., 2025)</td></tr><tr><td>Vanilla</td><td>85.3</td><td>85.3</td><td>85.3</td><td>85.3</td><td>95.1</td><td>95.1</td><td>95.1</td><td>95.1</td><td>70.0</td><td>70.0</td><td>70.0</td><td>70.0</td><td>82.1</td><td>82.1</td><td>82.1</td><td>82.1</td></tr><tr><td>AirCache</td><td>84.1</td><td>81.5</td><td>78.8</td><td>76.5</td><td>94.8</td><td>93.3</td><td>90.5</td><td>85.5</td><td>69.7</td><td>69.0</td><td>68.5</td><td>66.7</td><td>81.7</td><td>80.9</td><td>78.7</td><td>74.3</td></tr><tr><td>TGV-KV</td><td>85.3</td><td>84.9</td><td>84.2</td><td>82.8</td><td>95.0</td><td>94.8</td><td>94.0</td><td>92.3</td><td>70.0</td><td>69.7</td><td>69.1</td><td>68.1</td><td>82.1</td><td>81.9</td><td>81.1</td><td>79.2</td></tr></table>

Table 9. Detailed Results on Video-TT. We evaluate the primary task and three robustness tasks. 

<table><tr><td rowspan="2">Methods</td><td colspan="3">Primary↑</td><td colspan="3">Correctly-Led↑</td><td colspan="3">Wrongly-Led↑</td><td colspan="3">Paraphrase↑</td></tr><tr><td>50%</td><td>20%</td><td>10%</td><td>50%</td><td>20%</td><td>10%</td><td>50%</td><td>20%</td><td>10%</td><td>50%</td><td>20%</td><td>10%</td></tr><tr><td colspan="13">Qwen3-VL-4B-Instruct (Bai et al., 2025)</td></tr><tr><td>Vanilla</td><td>20.0</td><td>20.0</td><td>20.0</td><td>24.8</td><td>24.8</td><td>24.8</td><td>26.7</td><td>26.7</td><td>26.7</td><td>20.6</td><td>20.6</td><td>20.6</td></tr><tr><td>StreamingLLM</td><td>18.6</td><td>16.4</td><td>15.2</td><td>24.8</td><td>25.5</td><td>25.1</td><td>27.2</td><td>22.0</td><td>20.4</td><td>19.5</td><td>17.2</td><td>16.2</td></tr><tr><td>SnapKV</td><td>19.2</td><td>15.8</td><td>6.5</td><td>24.6</td><td>24.1</td><td>24.3</td><td>26.9</td><td>24.1</td><td>9.8</td><td>20.9</td><td>18.7</td><td>8.1</td></tr><tr><td> $H_2O$ </td><td>16.2</td><td>10.5</td><td>6.3</td><td>23.8</td><td>14.8</td><td>5.5</td><td>20.9</td><td>11.0</td><td>2.3</td><td>20.2</td><td>11.7</td><td>5.6</td></tr><tr><td>PrefixKV</td><td>14.9</td><td>9.9</td><td>4.1</td><td>22.4</td><td>13.9</td><td>3.0</td><td>22.0</td><td>8.5</td><td>1.5</td><td>17.8</td><td>10.1</td><td>4.8</td></tr><tr><td>TGV-KV</td><td>19.1</td><td>19.1</td><td>19.0</td><td>24.4</td><td>25.6</td><td>25.3</td><td>28.1</td><td>26.7</td><td>25.5</td><td>21.5</td><td>20.0</td><td>18.6</td></tr></table>

![](images/f4d7f297ac65c3ece45156a74f62fa7ce78591029c25c1110274c117190e672d.jpg)

<details>
<summary>bar</summary>

| Text Length | Vanilla | TGV/30% | TGV/20% | TGV/10% |
| ----------- | ------- | ------- | ------- | ------- |
| 1-50        | 70      | 70      | 70      | 70      |
| 51-100      | 60      | 60      | 60      | 60      |
| 101-150     | 50      | 50      | 50      | 50      |
| 151-200     | 40      | 40      | 40      | 40      |
| 201-250     | 55      | 55      | 55      | 55      |
| 251+        | 80      | 80      | 80      | 80      |
</details>

Figure 6. Accuracy Results on Different Lengths. We count the model accuracy with different text instruction lengths in ScienceQA (Lu et al., 2022). The results stay smooth and stable under different lengths, indicating removing all visual KV makes a small influence.

# B. Extended Experiment Results

# B.1. Text-Dominant Results

We provide a comprehensive evaluation of text-dominant tasks in Table 7, including more metrics (BLEU-1/2/3/4, CIDEr, METEOR, and ROUGE-L), which assess the model’s ability in generating descriptive text based on visual inputs. An interesting phenomenon is that TGV-KV surpasses the vanilla model in several settings, suggesting that TGV-KV acts as an effective noise filter that evicts irrelevant information.

# B.2. Video Results

Video understanding poses a significant challenge for KV eviction due to the extended context length and temporal redundancy. We report the raw results of Video-TT (Zhang et al., 2025b) in Table 9, which evaluates reasoning capabilities under adversarial conditions. On the primary task, TGV-KV retains 95.0% of the vanilla performance even when the budget is only 10% of the full KV. On the wrongly-led subtask, which tests the model’s ability to ignore hallucinatory instructions, TGV-KV achieves 95.5% of the vanilla performance. This indicates that our Text-Weighted Ranking (TWR) successfully prioritizes visual frames and patches that are semantically grounded in the query, allowing the model to answer correctly despite adversarial leads.

# B.3. Results Against Visual KV Eviction Method

Some related works (Huang et al., 2025) also conduct KV eviction for VLMs, however, they only evict vision KV and preserve all the text KV. TGV-KV evaluates the importance score of each vision KV and is also applicable for purely vision KV eviction. We remove the TPR policy and always keep all the text KV in each layer, in the same setting as these works. The retention budget is defined as the proportion of retained vision KV takes up in the full vision KV. In Table 8, we show the performance comparison. Notably, TGV-KV surpasses the comparison method across all the models and all the budget settings, proving its efficacy in vision KV eviction and further strengthening the design of TVB and TPR.

# C. Extended Visualizations

# C.1. Attention Gap and Dominant Tokens

In Fig. 7, we provide visualizations of the attention map across different VLM layers. The text-text part shows distinct vertical lines, corresponding to the dominant text tokens. Crucially, the text-vision part has low values in all the layers,

empirically validating our observation of the modality gap.

# C.2. Retained KV

We provide more examples of retained KV in Fig. 8. TGV-KV effectively preserves the image patches most related to the question and most critical text counterparts, ensuring the retained KV directly serves the text instruction. This visualization confirms that our method aligns KV cache eviction with the semantic intent of text prompt.

# D. Discussions

# D.1. Towards Long Text and Short Vision.

Under circumstances with long text, all the visual KV may be evicted due to our TPR policy. To evaluate the influence of evicting all vision tokens, we evaluate TGV-KV on ScienceQA (Lu et al., 2022), a VQA dataset with long text. As shown in Fig. 6, the performance drop across all the lengths under small budget are similar, indicating the eviction of evicting vision KV is relative small. Besides, relative studies point out that visual information fuse into text features gradually in the decoder (Neo et al., 2025), therefore TGV-KV still maintains performance for long text and short vision sequences.

![](images/880da5eb25f22620c914c8f024e402647f6bac1c3335ef7b249060ca5f31ec50.jpg)

Figure 7. Visualization of Attention Maps. The image is displayed in log scale.   
![](images/a5e6042ec4fa0221cb8253ae45b83dd8386b08647590c74e90a8166319a9c0c0.jpg)  
Figure 8. Visualization of Retained KV. The patches most related to the text instruction are circled in violet. Evicted KVs are masked or labelled in gray.