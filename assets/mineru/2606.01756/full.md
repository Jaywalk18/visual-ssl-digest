# EvoCut: Multi-Layer Evolution-Aware Visual Token Compression for Efficient Large Vision-Language Models

Hongyu Lu1,2 Feng Zhang2,‡ Wenwei Jin2 Huanling Hu3 Pengfei Zhang1 Yao Hu2 Jiawei Li2,∗ Shikai Jiang1,∗

1Harbin Institute of Technology 2Xiaohongshu 3Fudan University

24S021013@stu.hit.edu.cn, 22b921003@stu.hit.edu.cn

{zhangfeng4, wangdesheng}@xiaohongshu.com

hlhu24@m.fudan.edu.cn, wenwei1217.jin@gmail.com

yaoohu@gmail.com, jiangshikai@hit.edu.cn

# Abstract

Large vision-language models (LVLMs) achieve strong performance on image and video understanding tasks, but their inference efficiency is constrained by the large number of visual tokens produced by vision encoders. Most existing visual token compression methods estimate token importance from attention scores or representation properties at specific layers, overlooking how visual tokens evolve across the vision encoder. Such layer-specific criteria may provide incomplete importance estimates and limit performance preservation after compression. To address this issue, we analyze layer-wise visual token evolution directions and observe that tokens form multiple group evolution directions across vision-encoder layers. Our analysis further shows that informative tokens tend to exhibit persistent deviations from common group evolution directions. Based on this observation, we propose EvoCut, a training-free and attention-free visual token compression method that estimates token importance from multi-layer evolution deviation. Experimental results show that EvoCut can retain only 11.1% of the visual tokens on LLaVA-1.5-7B while preserving 94.4% of the average performance, demonstrating its effectiveness in balancing efficiency and accuracy.

# 1 Introduction

Large vision-language models (LVLMs) extend large language models to image and video understanding, and have shown strong performance across multimodal tasks (Alayrac et al., 2022; Li et al., 2023b; Liu et al., 2023; Bai et al., 2023; Lin et al., 2023). However, processing visual inputs requires the vision encoder to produce a large number of visual tokens. For high-resolution images and videos, the token sequence can easily grow to hundreds or thousands of tokens before it is passed to the language model (Liu et al., 2024a; Bai et al.,

![](images/b72a3119737673161b6e88982017d113e4e8062fab8339b43d6015dffb3699bf.jpg)

<details>
<summary>natural_image</summary>

Two basketball players in Lakers and Golden State jerseys smiling at the camera (no visible text or symbols)
</details>

![](images/0ca2daad896b9240d55883cf1eb92d69fd75cd4073a5f44632e40016473600d2.jpg)  
Describe this image.

The mage features two professindlbasketball players,one wearing a Lakers jersey and the other wearing a Golden State Warrors jersey. Theyare standing next toeachother,smiling and posing for a picture.

![](images/75af0af58883eaaae7fef86fee281dfc9549c997dd035a01a6dd57c0c0b4b276.jpg)

<details>
<summary>natural_image</summary>

Two basketball players in a Lakers jersey at the Celtics court (no visible text or symbols)
</details>

![](images/0ece15cdf4eff4834b1cf869295bccb586327b2c81703d2746880b29f011ae9d.jpg)

![](images/588263eb490da73c18bf560b02a1029af0a634e1f02988c829ab4caee0174a0e.jpg)

<details>
<summary>natural_image</summary>

Two basketball players in black uniforms with checkered jerseys, one wearing a 'JAKKA' jersey (no visible text or symbols on the players or background)
</details>

The image features two professional basketballplayers from the Golden State Warriors, Klay Thompson and Draymond Green, posing together for a photo.

The image features twoprofessional basketball players from the Golden State Warriors and the Los Angeles Lakers,both wearing their team uniforms.

Figure 1: Comparison between ApET and EvoCut. Red annotations indicate incorrect answers, while green annotations indicate correct answers.

2025). These visual tokens often dominate the prefilling cost and increase memory consumption, inference latency, and FLOPs. Therefore, reducing redundant visual tokens while preserving multimodal understanding ability is important for efficient LVLM deployment.

To reduce the inference cost of LVLMs, recent studies have explored visual token reduction and compression before or during language-model decoding (Chen et al., 2024a; Zhang et al., 2025; Yang et al., 2025; Shang et al., 2025; Ma et al., 2026; Wen et al., 2025; Jiang et al., 2025; Fan et al., 2025). Existing LVLM compression methods mainly estimate token importance from either attention responses or visual representations. Attentionbased methods prune or merge tokens based on attention scores, but these scores can be affected by positional bias and tend to concentrate on later token positions, as shown in Figure 2. They are also difficult to combine with efficient attention implementations such as FlashAttention (Dao et al., 2022; Dao, 2023). Representation-based methods avoid direct dependence on attention maps, but they often rely on single-layer features or local token relations. As shown in Figure 2, such layer-specific criteria can produce inconsistent importance estimates across different layers. These limitations suggest that many existing criteria provide only a partial view of token importance, making it difficult to consistently preserve informative visual tokens after compression.

![](images/29094efb3cf3cce85eb36f46f8ca2ae448cfbe9b4d6f38228ea1f5781c5ffd1f.jpg)  
Input Image

![](images/91e7ea9ebd671c091abd46bccae470415b1781440f68185e28ad485270ec103e.jpg)  
Attention Map

![](images/d64929e1e3b542ced02e27102bd6d59095285c53c704a056165000fbfabfebcc.jpg)  
Layer 3 V2Drop

![](images/16a7d8c4f510146c24ce81a1be3713be35fbb2fc9269f3268f9d5c1c542d8240.jpg)  
Layer 23 V2Drop

![](images/572c17d8d3163cee25fbb74ff5970008d3f42cc5ee67ec9524da00fcad81a229.jpg)  
Layer 3ApET

![](images/cfffccf596059e8876cf1ccd00e7788631aaa0f36fc319c64e84e83312bca8c2.jpg)  
Layer 23 ApET

![](images/1b405885ee3e0c7a83d7f285d79a7636ba6d33ac448d236eeb7c08fdf3d5020e.jpg)  
Figure 2: Visualization results show that attention scores are affected by positional bias, causing them to concentrate more on later tokens. Meanwhile, V2Drop and ApET produce inconsistent importance maps across different layers, suggesting that single-layer criteria can be unstable for identifying important visual tokens.

We argue that visual token importance should be estimated from its layer-wise evolution, rather than from a single-layer snapshot. To examine this hypothesis, we analyze token movements between adjacent vision-encoder layers and observe that visual tokens do not follow a single global evolution direction. Instead, they form multiple group evolution directions, and tokens that consistently deviate from these shared directions often correspond to visually informative regions. This observation suggests that persistent deviation from group evolution directions provides a useful signal for identifying important visual tokens. At the same time, single-layer deviation can be noisy, especially in deeper layers where token representations become highly mixed. Therefore, token importance should be modeled from cross-layer evolution behavior rather than from isolated layer-wise scores.

Motivated by these observations, we propose EvoCut, a training-free and attention-free visual token compression method for LVLMs. EvoCut estimates token importance from layer-wise token evolution inside the vision encoder. For each layer transition, it measures the deviation between each token’s evolution direction and the group evolution directions, and accumulates the resulting scores across layers with a history-aware update. This multi-layer scoring allows EvoCut to retain tokens that consistently provide distinctive visual information, as shown in Figure 1. Since EvoCut does not rely on attention maps, it can be integrated into existing LVLMs and remains compatible with efficient attention operators. Our main contributions are summarized as follows:

• We investigate visual-token evolution across vision-encoder layers and observe that token movements follow multiple group evolution directions. We further observe that tokens with persistent deviations from these shared directions are more likely to correspond to informative visual regions.   
• Based on these observations, we propose Evo-Cut, a training-free and attention-free visual token compression method for LVLMs. EvoCut estimates token importance by measuring deviation from multiple group evolution directions and accumulating this signal across layers, avoiding reliance on single-layer importance estimates.   
• Experiments on image and video understanding tasks show that EvoCut consistently improves the performance-efficiency trade-off across multiple LVLM backbones. For example, EvoCut preserves 94.4% of the average performance on LLaVA-1.5-7B with only 64 visual tokens while achieving a 1.44× total-time speedup.

# 2 Related Work

# 2.1 Large Vision-Language Models

Large Vision-Language Models (LVLMs) extend large language models (Brown et al., 2020; Touvron et al., 2023; OpenAI, 2023) to multimodal scenarios by incorporating visual encoders and projection modules. Most LVLMs build on Transformerbased (Vaswani et al., 2017) pretrained visual encoders such as ViT or CLIP to convert visual inputs into token sequences (Dosovitskiy et al., 2021; Radford et al., 2021). Typically, visual tokens are then mapped into the language model space, enabling the model to perform tasks such as visual question answering, image captioning, and multimodal reasoning. Models such as BLIP-2, InstructBLIP, MiniGPT-4, LLaVA, Qwen-VL, and

InternVL have demonstrated the effectiveness of this paradigm across various vision-language tasks (Li et al., 2023b; Dai et al., 2023; Zhu et al., 2023; Liu et al., 2023; Bai et al., 2023; Chen et al., 2024b). Recent LVLMs further improve fine-grained visual understanding by supporting high-resolution images and long videos (Liu et al., 2024a; Bai et al., 2025; Li et al., 2023c; Zhang et al., 2023; Lin et al., 2023). However, the resulting visual tokens are often much more numerous than textual tokens, introducing substantial computational and memory overhead. This makes reducing redundant visual tokens while preserving multimodal reasoning ability a key problem for efficient LVLMs.

# 2.2 Visual Token Compression for LVLMs

Existing visual token compression methods for LVLMs mainly estimate token importance from either attention responses or visual representations. Attention-based methods use visual self-attention, CLS-token attention, or text-to-vision attention to prune or merge visual tokens. For example, VisionZip compresses visual tokens based on CLStoken attention scores from the selected visionencoder layer used by the LVLM, which is typically the penultimate layer in LLaVA-style models (Yang et al., 2025). Although such methods are simple and effective, attention scores can be affected by positional bias and are difficult to combine with efficient attention operators such as FlashAttention (Dao et al., 2022; Dao, 2023). Representationbased methods avoid direct dependence on attention maps and instead identify redundant tokens from visual features, such as feature similarity, approximation error, token duplication, or local variation (Zhang et al., 2025; Ma et al., 2026; Wen et al., 2025). For instance, ApET reconstructs the original visual tokens with a compact set of basis tokens and uses the approximation error to measure token informativeness (Ma et al., 2026). However, many existing criteria still depend on layer-specific features or local token relations, which may provide only a partial estimate of token importance. These methods usually evaluate token importance from static features or local relationships, rather than explicitly modeling how each token changes between adjacent vision-encoder layers. In contrast, EvoCut models token importance from multi-layer evolution behavior and measures whether a token persistently deviates from multiple shared group evolution directions.

# 3 Method

# 3.1 Preliminaries

An LVLM turns visual content into a token sequence that can be consumed by a language model (Huang et al., 2023). Given an image or video, the vision encoder first extracts patch-level representations, and a projector maps these representations into the same embedding space as text tokens. We denote the resulting visual token sequence as V . Together with a textual instruction T , these visual tokens form the multimodal context used by the LLM to generate an answer autoregressively:

$$
y _ {t} \sim p _ {\theta} (y _ {t} \mid V, T, y _ {<   t}), \tag {1}
$$

where θ denotes the model parameters and $y _ { < t }$ denotes previously generated tokens.

# 3.2 Layer-wise Token Evolution Analysis

To understand how visual tokens evolve inside the vision encoder, we analyze the layer-wise evolution direction of patch tokens between adjacent Transformer layers. Let $x _ { i } ^ { ( t ) } \in \mathbb { R } ^ { d }$ denote the hidden state of the i-th patch token at the t-th visionencoder layer. We define its layer-wise evolution direction from layer t − 1 to layer t as

$$
\vec {\Delta} _ {i} ^ {(t)} = \frac {x _ {i} ^ {(t)} - x _ {i} ^ {(t - 1)}}{\| x _ {i} ^ {(t)} - x _ {i} ^ {(t - 1)} \| _ {2}}. (2)
$$

Discovery of Group Evolution Directions. We use HDBSCAN (Campello et al., 2013) to cluster token evolution directions at each layer transition, since it can discover density-based direction groups without specifying the number of clusters. Each cluster is treated as a shared evolution direction followed by a group of tokens, and its group direction is defined as the normalized mean of all member directions. As shown in the right part of Figure 3, the number of clustered directions is small in shallow layers and increases in deeper layers on both POPE and TextVQA. This indicates that visual tokens do not evolve along a single global direction, but gradually split into multiple group evolution paths as the network deepens.

High-Deviation Tokens Correlate with Informative Regions. To study the properties of informative tokens during layer-wise evolution, we examine their deviations from group evolution directions. For each token, we compute its cosine deviation from the nearest group direction at each layer transition. As shown in the left part of Figure 3, tokens corresponding to informative regions, such as the license plate and airplane, maintain large deviations across multiple layers. More generally, high-deviation tokens tend to appear on distinct foreground objects, text regions, and small targets, suggesting that directional deviation is associated with informative visual content.

![](images/12602539253f6aa6d696eea15756f6293155f958574f76e1ba90d7e8e8f35797.jpg)  
Figure 3: Visualization of multi-layer token evolution in LLaVA-1.5. (Left) Tokens with larger deviations from group evolution directions tend to correspond to visually informative regions. (Right) The number of clustered token evolution directions across vision-encoder layers on POPE and TextVQA, showing that token transitions form multiple group evolution directions. More results are provided in Appendix A.4.

Redundancy of Low-Deviation Tokens. In contrast, tokens that closely follow dominant group directions are mostly located in background regions, repetitive textures, or weakly informative edges. These tokens show highly predictable crosslayer behavior and carry less token-specific visual information for multimodal understanding. This complements the high-deviation cases and suggests that deviation from group directions can help distinguish informative tokens from less informative ones.

Deviation Noise in Deeper Layers. We also observe that single-layer deviation becomes less reliable in deeper layers. As self-attention repeatedly aggregates global context, token representations become more mixed, and background tokens may show large deviations due to their association with foreground regions. This makes isolated layer-wise scores unstable and motivates accumulating deviation signals across multiple layer transitions.

Overall, these observations suggest that token importance is better characterized by persistent deviation from group evolution directions than by a single-layer snapshot. This motivates EvoCut to score visual tokens using multi-layer evolution deviation.

# 3.3 Evolution-Aware Token Compression

Building on the above observations, we propose EvoCut, a training-free visual token compression method that estimates token importance from multilayer evolution in the vision encoder. Instead of relying on a single-layer representation snapshot, EvoCut tracks how each visual token evolves across adjacent layers and identifies tokens whose evolution directions deviate from shared group evolution directions. As illustrated in Figure 4, EvoCut consists of three steps: group evolution modeling, evolution-deviation scoring, and multi-layer score accumulation.

Group Evolution Modeling. Let $\begin{array} { r l } { X ^ { ( \ell ) } } & { { } = } \end{array}$ $\{ x _ { 1 } ^ { ( \ell ) } , \ldots , x _ { N } ^ { ( \ell ) } \}$ , x(ℓ)N } denote the vision-encoder hidden states at layer ℓ. For each adjacent layer pair, we compute the token evolution direction ∆⃗ (ℓ)i $\vec { \Delta } _ { i } ^ { ( \ell ) }$ i using the normalized layer-wise difference in Equation 2.

Since tokens may evolve along multiple shared directions, we cluster all token evolution directions for the transition from layer ℓ − 1 to layer ℓ with K-means. For simplicity and efficient inference, EvoCut uses a fixed number of clusters for all layer transitions. For each cluster, its center is computed as the mean of the assigned token evolution directions and then normalized to form a group evolution direction. EvoCut uses the M cluster centers as group evolution directions:

$$
\mathcal {U} ^ {(\ell)} = \{\vec {u} _ {1} ^ {(\ell)}, \vec {u} _ {2} ^ {(\ell)}, \dots , \vec {u} _ {M} ^ {(\ell)} \}. \tag {3}
$$

These directions capture the dominant evolution directions shared by visual tokens at the current layer transition.

Evolution-Deviation Scoring. For each token, EvoCut measures its deviation from the closest group evolution direction. We compute the best cosine alignment between ∆⃗ (ℓ)i $\vec { \Delta } _ { i } ^ { ( \ell ) }$ and the direction set $\mathcal { U } ^ { ( \ell ) }$ :

![](images/304b777971e6bdeb807239161fdf231942d4d3b43f6913109df5caeb6ea6fb40.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Large Language Model"] --> B["Textual tokens"]
    B --> C["EvoCut"]
    C --> D["Tokenizer"]
    D --> E["Vision Encoder"]
    E --> F["Question: How many cows can be seen in the image?"]
    
    subgraph Input Layer
        G["block L"] --> H["score^L"]
        I["..."] --> J["score^{l+1}"]
        K["block l+1"] --> L["score^l+1"]
        M["block l"] --> N["score^l"]
        O["..."] --> P["score^1"]
        Q["block 1"] --> R["score^1"]
    end
    
    subgraph Output Layer
        S["for token i and layer l"] --> T["evolution direction 2 u2"]
        U["evolution direction 3 u3"] --> V["evolution direction 1 u1"]
        W["token direction Δl"] --> X["score^l = α score^{l-1} + (1 - α) (1 - cos(Δl, u1))"]
    end
```
</details>

Figure 4: Overview of EvoCut. EvoCut estimates visual token importance from multi-layer token evolution, measures deviation from multiple group evolution directions, aggregates scores across layers, and retains top-ranked visual tokens before the multimodal projector and LLM prefilling.

$$
a _ {i} ^ {(\ell)} = \max _ {m} \cos (\vec {\Delta} _ {i} ^ {(\ell)}, \vec {u} _ {m} ^ {(\ell)}). (4)
$$

The layer-wise importance score is then defined as the directional deviation:

$$
r _ {i} ^ {(\ell)} = 1 - a _ {i} ^ {(\ell)}. \tag {5}
$$

A larger score indicates stronger deviation from shared group evolution directions, suggesting that the token is more likely to contain distinctive visual information.

Multi-Layer Score Accumulation. To reduce the instability of single-layer estimates, EvoCut accumulates token scores across layer transitions with an exponential moving average:

$$
s _ {i} ^ {(\ell)} = \alpha s _ {i} ^ {(\ell - 1)} + (1 - \alpha) r _ {i} ^ {(\ell)}, \tag {6}
$$

where α controls the contribution of historical evolution information. After the final vision-encoder layer, EvoCut keeps the top-K patch tokens according to si and sends the compressed visual sequence to the multimodal projector. The complete pseudocode is provided in Appendix A.1.

# 4 Experiments

# 4.1 Experimental Setup

Models and Baselines. To validate the effectiveness and generality of the proposed method, we evaluate it on representative LVLM backbones covering both image and video understanding: LLaVA-1.5-7B, LLaVA-NeXT-7B, Qwen-2.5-VL-7B, and Video-LLaVA-7B (Liu et al., 2023, 2024a; Bai et al., 2025; Lin et al., 2023). These models cover fixed-length, high-resolution, dynamicresolution, and temporal visual-token settings. We compare against representative training-free visual token reduction methods covering attention-based and representation-based compression, including FastV, SparseVLM, PDrop, V2Drop, VisionZip, and ApET (Chen et al., 2024a; Zhang et al., 2025; Xing et al., 2025; Chen et al., 2026; Yang et al., 2025; Ma et al., 2026).

Implementation Details. All methods are evaluated under the standard inference configurations of their corresponding LVLMs without additional training or fine-tuning. Our method can be integrated into all evaluated backbones: it accumulates token-importance scores from intermediate visionencoder layers and applies token selection to the final vision-encoder output before the multimodal projector. For EvoCut, we start computing token importance from the middle of the vision encoder, i.e., layer 12 for LLaVA-1.5, LLaVA-NeXT, and Video-LLaVA, and layer 16 for Qwen-2.5-VL. The number of K-means clusters is set to M = 8, and the EMA decay factor is set to $\alpha = 0 . 9 5$ for all experiments. All experiments are conducted on NVIDIA A800-80G GPUs.

# 4.2 Main Results

Image understanding tasks. We evaluate Evo-Cut on LLaVA-1.5-7B, LLaVA-NeXT-7B, and Qwen-2.5-VL-7B during inference without additional training, covering fixed-resolution, highresolution, and dynamic-resolution visual encoding settings. For LLaVA-1.5-7B, Table 1 reports results on a broad set of image understanding benchmarks, including visual question answering, perception, hallucination evaluation, science reasoning, textrich understanding, and multimodal reasoning tasks (Goyal et al., 2017; Hudson and Manning, 2019; Singh et al., 2019; Lu et al., 2022; Fu et al., 2023; Li et al., 2023d; Liu et al., 2024b). EvoCut achieves the best normalized average performance under all token budgets, retaining 98.4%, 97.0%, and 94.4% of the vanilla performance with 192, 128, and 64 tokens, respectively. Under the most aggressive 64-token setting, EvoCut still outperforms the strongest baseline by 0.8 percentage points in average retention, with clear gains on MMB, MMBCN, ， VQAV2, TextVQA, and SEED.

Table 1: Performance comparison on image understanding benchmarks with LLaVA-1.5-7B under different visual token budgets. Avg. reports the normalized average performance relative to the vanilla upper bound. 

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td> $MMB^{CN}$ </td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA^{V2}$ </td><td> $VQA^{Text}$ </td><td>SEED</td><td>LLaVA-B</td><td>Avg.</td></tr><tr><td colspan="12">Upper Bound, 576 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>61.9</td><td>64.7</td><td>58.1</td><td>1862</td><td>85.9</td><td>69.5</td><td>78.5</td><td>58.2</td><td>59.3</td><td>66.9</td><td>100.0%</td></tr><tr><td colspan="12">Retain 192 Tokens (↓66.7%)</td></tr><tr><td>FastV</td><td>52.7</td><td>61.2</td><td>57.0</td><td>1612</td><td>64.8</td><td>67.3</td><td>67.1</td><td>52.5</td><td>57.1</td><td>49.4</td><td>88.2%</td></tr><tr><td>SparseVLM</td><td>57.6</td><td>62.5</td><td>53.7</td><td>1721</td><td>83.6</td><td>69.1</td><td>75.6</td><td>56.1</td><td>55.8</td><td>66.1</td><td>95.7%</td></tr><tr><td>PDrop</td><td>57.3</td><td>63.2</td><td>56.8</td><td>1766</td><td>82.3</td><td>69.0</td><td>75.1</td><td>56.1</td><td>54.7</td><td>65.8</td><td>96.1%</td></tr><tr><td>VisionZip</td><td>59.3</td><td>63.0</td><td>57.3</td><td>1778</td><td>85.2</td><td>68.7</td><td>76.6</td><td>57.3</td><td>56.4</td><td>67.7</td><td>97.8%</td></tr><tr><td> $V^2Drop$ </td><td>58.5</td><td>63.7</td><td>56.6</td><td>1796</td><td>85.1</td><td>69.3</td><td>74.9</td><td>55.6</td><td>56.4</td><td>66.5</td><td>97.1%</td></tr><tr><td>ApET</td><td>60.2</td><td>63.4</td><td>57.9</td><td>1808</td><td>86.3</td><td>68.5</td><td>76.2</td><td>54.4</td><td>56.8</td><td>66.1</td><td>97.6%</td></tr><tr><td>EvoCut</td><td>60.2</td><td>64.2</td><td>57.6</td><td>1794</td><td>86.5</td><td>68.5</td><td>76.7</td><td>57.6</td><td>57.3</td><td>66.4</td><td>98.4%</td></tr><tr><td colspan="12">Retain 128 Tokens (↓77.8%)</td></tr><tr><td>FastV</td><td>49.6</td><td>56.1</td><td>56.4</td><td>1490</td><td>59.6</td><td>60.2</td><td>61.8</td><td>50.6</td><td>55.9</td><td>52.0</td><td>83.8%</td></tr><tr><td>SparseVLM</td><td>56.0</td><td>60.0</td><td>51.1</td><td>1696</td><td>80.5</td><td>67.1</td><td>73.8</td><td>54.9</td><td>53.4</td><td>62.7</td><td>92.5%</td></tr><tr><td>PDrop</td><td>57.1</td><td>61.1</td><td>56.6</td><td>1644</td><td>82.3</td><td>68.4</td><td>72.9</td><td>54.8</td><td>53.3</td><td>61.9</td><td>93.6%</td></tr><tr><td>VisionZip</td><td>57.6</td><td>62.0</td><td>56.2</td><td>1759</td><td>83.2</td><td>68.9</td><td>75.6</td><td>56.8</td><td>54.9</td><td>64.8</td><td>95.9%</td></tr><tr><td> $V^2Drop$ </td><td>56.3</td><td>61.8</td><td>54.5</td><td>1712</td><td>80.9</td><td>68.8</td><td>72.1</td><td>53.8</td><td>53.8</td><td>62.9</td><td>93.4%</td></tr><tr><td>ApET</td><td>58.9</td><td>62.3</td><td>56.4</td><td>1801</td><td>86.1</td><td>68.7</td><td>75.1</td><td>53.9</td><td>54.7</td><td>64.2</td><td>96.1%</td></tr><tr><td>EvoCut</td><td>59.2</td><td>62.3</td><td>56.7</td><td>1790</td><td>85.7</td><td>68.7</td><td>76.2</td><td>57.1</td><td>55.2</td><td>65.1</td><td>97.0%</td></tr><tr><td colspan="12">Retain 64 Tokens (↓88.9%)</td></tr><tr><td>FastV</td><td>46.1</td><td>48.0</td><td>52.7</td><td>1256</td><td>48.0</td><td>51.1</td><td>55.0</td><td>47.8</td><td>51.9</td><td>46.1</td><td>74.5%</td></tr><tr><td>SparseVLM</td><td>52.7</td><td>56.2</td><td>46.1</td><td>1505</td><td>75.1</td><td>62.2</td><td>68.2</td><td>51.8</td><td>51.1</td><td>57.5</td><td>85.7%</td></tr><tr><td>PDrop</td><td>47.5</td><td>58.8</td><td>50.5</td><td>1092</td><td>55.9</td><td>69.2</td><td>69.2</td><td>45.9</td><td>40.0</td><td>59.2</td><td>80.1%</td></tr><tr><td>VisionZip</td><td>55.1</td><td>60.1</td><td>55.3</td><td>1687</td><td>77.0</td><td>69.0</td><td>72.4</td><td>55.5</td><td>52.2</td><td>62.9</td><td>92.6%</td></tr><tr><td> $V^2Drop$ </td><td>50.5</td><td>55.2</td><td>49.7</td><td>1470</td><td>75.1</td><td>68.9</td><td>71.2</td><td>51.8</td><td>51.4</td><td>62.4</td><td>87.8%</td></tr><tr><td>ApET</td><td>56.9</td><td>61.2</td><td>54.4</td><td>1714</td><td>84.4</td><td>68.9</td><td>72.5</td><td>53.0</td><td>52.4</td><td>63.0</td><td>93.6%</td></tr><tr><td>EvoCut</td><td>56.6</td><td>61.8</td><td>56.0</td><td>1692</td><td>83.9</td><td>68.8</td><td>73.4</td><td>55.7</td><td>53.2</td><td>63.0</td><td>94.4%</td></tr></table>

Tables 2 and 3 further show consistent gains on LLaVA-NeXT-7B and Qwen-2.5-VL-7B. On LLaVA-NeXT-7B, EvoCut obtains the best average retention under all token budgets, reaching 98.0%, 94.9%, and 91.4% when retaining 640, 320, and 160 tokens. The advantage becomes larger as the token budget decreases, suggesting that multilayer evolution information is particularly useful under aggressive compression. On Qwen-2.5-VL-7B, EvoCut improves the best competing average retention from 95.5% to 96.2% at the 20% token budget and from 90.5% to 91.7% at the 10% token budget. These results indicate that multi-layer evolution deviation provides a robust token-importance criterion across different LVLM visual encoding settings.

Video understanding tasks. We next apply Evo-Cut to Video-LLaVA-7B to evaluate whether the proposed criterion generalizes to video understanding, where each input contains substantially more visual tokens across multiple frames. We evaluate on TGIF-QA, MSVD-QA, and MSRVTT-QA (Jang et al., 2017; Chen and Dolan, 2011; Xu et al., 2016). Detailed dataset descriptions are provided

Table 2: Main results on LLaVA-NeXT-7B. Avg. reports the normalized average performance relative to the vanilla upper bound. 

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td> $MMB^{CN}$ </td><td>MME</td><td>POPE</td><td> $VQA^{V2}$ </td><td> $VQA^{Text}$ </td><td>Avg.</td></tr><tr><td colspan="9">Upper Bound, 2880 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>64.2</td><td>67.4</td><td>60.6</td><td>1851</td><td>86.5</td><td>81.8</td><td>61.3</td><td>100%</td></tr><tr><td colspan="9">Retain 640 Tokens (↓ 77.8%)</td></tr><tr><td>PDrop</td><td>60.6</td><td>65.5</td><td>58.5</td><td>1781</td><td>83.7</td><td>78.3</td><td>57.4</td><td>95.8%</td></tr><tr><td>VisionZip</td><td>61.3</td><td>66.2</td><td>57.8</td><td>1787</td><td>85.9</td><td>79.1</td><td>60.2</td><td>97.1%</td></tr><tr><td>ApET</td><td>63.0</td><td>65.3</td><td>59.3</td><td>1815</td><td>87.2</td><td>79.2</td><td>57.9</td><td>97.6%</td></tr><tr><td>EvoCut</td><td>63.2</td><td>66.3</td><td>59.2</td><td>1804</td><td>86.3</td><td>79.2</td><td>59.7</td><td>98.0%</td></tr><tr><td colspan="9">Retain 320 Tokens (↓ 88.9%)</td></tr><tr><td>PDrop</td><td>56.4</td><td>63.4</td><td>56.2</td><td>1663</td><td>77.6</td><td>73.5</td><td>54.4</td><td>90.4%</td></tr><tr><td>VisionZip</td><td>59.3</td><td>63.1</td><td>55.3</td><td>1702</td><td>82.1</td><td>76.2</td><td>58.9</td><td>93.3%</td></tr><tr><td>ApET</td><td>61.0</td><td>63.5</td><td>56.6</td><td>1783</td><td>85.6</td><td>75.8</td><td>54.4</td><td>94.2%</td></tr><tr><td>EvoCut</td><td>60.9</td><td>63.7</td><td>56.9</td><td>1743</td><td>85.8</td><td>76.4</td><td>57.9</td><td>94.9%</td></tr><tr><td colspan="9">Retain 160 Tokens (↓ 94.4%)</td></tr><tr><td>PDrop</td><td>54.9</td><td>61.8</td><td>54.9</td><td>1513</td><td>72.3</td><td>70.2</td><td>52.7</td><td>86.4%</td></tr><tr><td>VisionZip</td><td>55.5</td><td>60.1</td><td>52.7</td><td>1628</td><td>74.8</td><td>71.4</td><td>56.2</td><td>88.0%</td></tr><tr><td>ApET</td><td>58.4</td><td>60.8</td><td>52.3</td><td>1680</td><td>82.6</td><td>72.7</td><td>53.8</td><td>90.1%</td></tr><tr><td>EvoCut</td><td>59.4</td><td>62.0</td><td>53.6</td><td>1653</td><td>82.2</td><td>73.0</td><td>57.3</td><td>91.4%</td></tr></table>

Table 4: Performance on video understanding tasks. The original number of visual tokens per video in Video-LLaVA is 2048, while all compression methods retain only 256 tokens. 

<table><tr><td rowspan="2">Method</td><td colspan="2">TGIF</td><td colspan="2">MSVD</td><td colspan="2">MSRVTT</td><td colspan="2">Avg.</td></tr><tr><td>Acc</td><td>Score</td><td>Acc</td><td>Score</td><td>Acc</td><td>Score</td><td>Acc</td><td>Score</td></tr><tr><td>Video-LLaVA</td><td>46.9</td><td>3.34</td><td>69.8</td><td>3.91</td><td>57.1</td><td>3.49</td><td>100%</td><td>100%</td></tr><tr><td rowspan="2">FastV</td><td>44.2</td><td>3.29</td><td>60.3</td><td>3.72</td><td>40.6</td><td>3.18</td><td rowspan="2">83.9%</td><td rowspan="2">94.9%</td></tr><tr><td>94.2%</td><td>98.5%</td><td>86.4%</td><td>95.1%</td><td>77.1%</td><td>91.1%</td></tr><tr><td rowspan="2">SparseVLM</td><td>45.9</td><td>3.32</td><td>68.6</td><td>3.90</td><td>32.9</td><td>3.02</td><td rowspan="2">84.6%</td><td rowspan="2">95.2%</td></tr><tr><td>98.9%</td><td>99.4%</td><td>98.3%</td><td>99.7%</td><td>57.6%</td><td>86.5%</td></tr><tr><td rowspan="2">PDrop</td><td>40.3</td><td>3.21</td><td>61.5</td><td>3.74</td><td>41.8</td><td>3.19</td><td rowspan="2">82.4%</td><td rowspan="2">94.4%</td></tr><tr><td>85.9%</td><td>96.1%</td><td>88.1%</td><td>95.7%</td><td>73.2%</td><td>91.4%</td></tr><tr><td rowspan="2">VisionZip</td><td>44.3</td><td>3.29</td><td>65.2</td><td>3.83</td><td>54.5</td><td>3.43</td><td rowspan="2">94.4%</td><td rowspan="2">98.2%</td></tr><tr><td>94.5%</td><td>98.5%</td><td>93.4%</td><td>98.0%</td><td>95.4%</td><td>98.3%</td></tr><tr><td rowspan="2">EvoCut</td><td>46.4</td><td>3.34</td><td>69.2</td><td>3.93</td><td>55.8</td><td>3.46</td><td rowspan="2">98.6%</td><td rowspan="2">99.9%</td></tr><tr><td>98.9%</td><td>100%</td><td>99.1%</td><td>100.5%</td><td>97.7%</td><td>99.1%</td></tr></table>

# in Appendix A.2.

As shown in Table 4, EvoCut retains 98.6% of the average accuracy and 99.9% of the average score while reducing the number of visual tokens per video from 2048 to 256. Compared with the strongest baseline, EvoCut improves average accuracy retention from 94.4% to 98.6% and average score retention from 98.2% to 99.9%. This shows that evolution-aware scoring remains effective when important visual evidence is distributed across both spatial and temporal dimensions.

Efficiency Analysis. We also compare total inference time, prefilling time, and FLOPs to evaluate practical efficiency gains. As shown in Tables 5 and 6, EvoCut consistently reduces total inference time and prefilling latency under matched token budgets. On LLaVA-1.5-7B, EvoCut reduces the total evaluation time from 17:05 to 11:53 and the prefilling time from 102ms to 68.4ms, achieving 1.44× and 1.49× speedups, respectively. On LLaVA-NeXT-7B, EvoCut further achieves 2.90× total-time and 2.98× prefilling speedups. Although VisionZip obtains marginally lower TFLOPs, Evo-Cut achieves the lowest wall-clock latency on both backbones, indicating limited scoring overhead while effectively reducing the cost of subsequent multimodal inference. A detailed FLOPs analysis is presented in Appendix A.3.

Table 3: Performance on Qwen-2.5-VL-7B for image understanding. The original number of visual tokens is dynamic, ranging from 256 to 2048. 

<table><tr><td>Method</td><td>GQA</td><td>POPE</td><td>SQA</td><td>MME</td><td>MMB</td><td>Avg.</td></tr><tr><td colspan="7">Upper Bound, 256–2048 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>60.5</td><td>86.2</td><td>76.7</td><td>2327</td><td>83.3</td><td>100%</td></tr><tr><td colspan="7">Retain 20% Tokens on Average (↓ 80%)</td></tr><tr><td>PDrop</td><td>55.1</td><td>78.4</td><td>70.9</td><td>2117</td><td>77.3</td><td>91.6%</td></tr><tr><td>VisionZip</td><td>56.8</td><td>82.4</td><td>76.3</td><td>2134</td><td>79.3</td><td>95.2%</td></tr><tr><td>ApET</td><td>57.0</td><td>83.6</td><td>75.5</td><td>2211</td><td>77.4</td><td>95.5%</td></tr><tr><td>EvoCut</td><td>57.1</td><td>83.4</td><td>76.7</td><td>2227</td><td>78.5</td><td>96.2%</td></tr><tr><td colspan="7">Retain 10% Tokens on Average (↓ 90%)</td></tr><tr><td>PDrop</td><td>52.0</td><td>74.8</td><td>69.7</td><td>1886</td><td>73.6</td><td>86.6%</td></tr><tr><td>VisionZip</td><td>52.4</td><td>78.9</td><td>74.1</td><td>2003</td><td>75.6</td><td>90.3%</td></tr><tr><td>ApET</td><td>53.4</td><td>79.3</td><td>74.0</td><td>2030</td><td>73.8</td><td>90.5%</td></tr><tr><td>EvoCut</td><td>52.8</td><td>79.2</td><td>74.5</td><td>2124</td><td>75.8</td><td>91.7%</td></tr></table>

Qualitative Analysis. Qualitative examples in Appendix A.5 show that EvoCut tends to retain tokens on foreground objects, text regions, and small targets, while filtering out background or repetitivetexture regions. This supports cross-layer evolution deviation as a useful signal for identifying informative visual tokens.

# 4.3 Ablation Studies

Ablation studies examine three key design choices in EvoCut: the number of group evolution directions, the starting layer for score accumulation, and the EMA decay factor. All ablations are conducted on LLaVA-1.5-7B with 64 retained visual tokens unless otherwise specified.

Effect of the number of evolution directions. To examine how the granularity of group evolution modeling affects token scoring, this ablation varies the number of group evolution directions M . As shown in Table 7, increasing M from 2 to 8 improves the normalized average performance from 91.8% to 95.0%, while further increasing it to 16 reduces the average to 93.5%. Accordingly, M = 8 is used as the default cluster number. This result supports our observation that visual tokens follow multiple evolution directions, while using too many directions may over-fragment shared directions and introduce noisy estimates.

Table 5: Efficiency analysis on LLaVA-1.5-7B using one NVIDIA A800 GPU on POPE. 

<table><tr><td>Methods</td><td>Token</td><td>Total Time↓</td><td> $\Delta \uparrow$ </td><td>Prefilling Time↓</td><td> $\Delta \uparrow$ </td><td>TFLOPs</td></tr><tr><td>LLaVA-1.5-7B</td><td>576</td><td>17:05</td><td>1.00×</td><td>102ms</td><td>1.00×</td><td>8.82</td></tr><tr><td>+ FastV</td><td>64</td><td>15:32</td><td>1.10×</td><td>87.3ms</td><td>1.17×</td><td>2.26</td></tr><tr><td>+ SparseVLM</td><td>64</td><td>15:57</td><td>1.07×</td><td>90.1ms</td><td>1.13×</td><td>2.31</td></tr><tr><td>+ PDrop</td><td>64</td><td>12:56</td><td>1.32×</td><td>72.5ms</td><td>1.41×</td><td>2.16</td></tr><tr><td>+ VisionZip</td><td>64</td><td>12:10</td><td>1.40×</td><td>69.2ms</td><td>1.47×</td><td>2.03</td></tr><tr><td>+ V2Drop</td><td>64</td><td>12:30</td><td>1.37×</td><td>71.0ms</td><td>1.44×</td><td>2.12</td></tr><tr><td>+ ApET</td><td>64</td><td>12:02</td><td>1.42×</td><td>68.9ms</td><td>1.48×</td><td>2.09</td></tr><tr><td>+ EvoCut</td><td>64</td><td>11:53</td><td>1.44×</td><td>68.4ms</td><td>1.49×</td><td>2.04</td></tr></table>

Table 7: Ablation on the number of evolution directions. All variants retain 64 visual tokens on LLaVA-1.5-7B, and M = 8 is used as the default setting. 

<table><tr><td>Variant</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA^{Text}$ </td><td>Avg.</td></tr><tr><td>M=2</td><td>55.1</td><td>56.7</td><td>1640</td><td>82.6</td><td>67.7</td><td>53.9</td><td>91.8%</td></tr><tr><td>M=4</td><td>54.6</td><td>58.7</td><td>1676</td><td>82.2</td><td>68.2</td><td>55.1</td><td>92.9%</td></tr><tr><td>M=8</td><td>56.6</td><td>61.8</td><td>1692</td><td>83.9</td><td>68.8</td><td>55.7</td><td>95.0%</td></tr><tr><td>M=16</td><td>55.6</td><td>59.4</td><td>1662</td><td>83.0</td><td>67.8</td><td>55.9</td><td>93.5%</td></tr></table>

Effect of the starting layer. To study how much cross-layer history is needed for stable token scoring, this ablation varies the layer from which Evo-Cut starts accumulating token-importance scores. Table 8 shows that starting from layer 12 achieves the best MME and POPE scores and ties for the best TextVQA result, while keeping prefilling latency close to the fastest setting. Accordingly, the middle-layer setting is used as the default. This suggests that starting too early may include noisy low-level variations, whereas starting too late loses useful cross-layer evolution history.

Table 8: Ablation on the starting layer for computing token importance. All variants retain 64 visual tokens on LLaVA-1.5-7B. 

<table><tr><td>Variant</td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA^{Text}$ </td><td>Prefilling Time↓</td></tr><tr><td>Start at layer 1</td><td>1691</td><td>83.2</td><td>69.2</td><td>55.6</td><td>69.5ms</td></tr><tr><td>Start at layer 8</td><td>1684</td><td>82.6</td><td>68.4</td><td>55.7</td><td>68.9ms</td></tr><tr><td>Start at layer 12</td><td>1692</td><td>83.9</td><td>68.8</td><td>55.7</td><td>68.4ms</td></tr><tr><td>Start at layer 20</td><td>1636</td><td>82.5</td><td>68.2</td><td>55.2</td><td>67.7ms</td></tr></table>

Effect of EMA decay factor. To assess the effect of historical-score weighting, this ablation studies the EMA decay factor α for multi-layer score accumulation. As shown in Table 9, α = 0.95 achieves the best normalized average performance of 95.0% and obtains the best results on most reported benchmarks. Accordingly, α = 0.95 is adopted as the default EMA decay factor. This indicates that retaining sufficient history is important for stable token scoring, but overly slow updates may suppress informative changes from later layers.

Table 6: Efficiency analysis on LLaVA-NeXT-7B using one NVIDIA A800 GPU on POPE. 

<table><tr><td>Methods</td><td>Token</td><td>Total Time↓</td><td> $\Delta \uparrow$ </td><td>Prefilling Time↓</td><td> $\Delta \uparrow$ </td><td>TFLOPs</td></tr><tr><td>LLaVA-NeXT-7B</td><td>2880</td><td>35:16</td><td>1.00×</td><td>206ms</td><td>1.00×</td><td>31.03</td></tr><tr><td>+ FastV</td><td>160</td><td>28:27</td><td>1.24×</td><td>117ms</td><td>1.76×</td><td>7.35</td></tr><tr><td>+ SparseVLM</td><td>160</td><td>30:26</td><td>1.16×</td><td>136ms</td><td>1.51×</td><td>7.62</td></tr><tr><td>+ PDrop</td><td>160</td><td>13:45</td><td>2.56×</td><td>79.5ms</td><td>2.59×</td><td>6.78</td></tr><tr><td>+ VisionZip</td><td>160</td><td>12:32</td><td>2.81×</td><td>69.9ms</td><td>2.95×</td><td>4.72</td></tr><tr><td>+ V2Drop</td><td>160</td><td>12:51</td><td>2.74×</td><td>73.8ms</td><td>2.79×</td><td>5.22</td></tr><tr><td>+ ApET</td><td>160</td><td>12:24</td><td>2.84×</td><td>69.7ms</td><td>2.96×</td><td>4.79</td></tr><tr><td>+ EvoCut</td><td>160</td><td>12:10</td><td>2.90×</td><td>69.1ms</td><td>2.98×</td><td>4.74</td></tr></table>

Table 9: Ablation on the EMA decay factor α for multilayer evolution scores. All variants retain 64 visual tokens on LLaVA-1.5-7B. 

<table><tr><td>Variant</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA^{Text}$ </td><td>Avg.</td></tr><tr><td> $\alpha = 0.80$ </td><td>54.2</td><td>58.7</td><td>1596</td><td>77.8</td><td>67.3</td><td>53.5</td><td>90.6%</td></tr><tr><td> $\alpha = 0.90$ </td><td>55.2</td><td>60.4</td><td>1643</td><td>77.0</td><td>68.7</td><td>54.8</td><td>92.2%</td></tr><tr><td> $\alpha = 0.95$ </td><td>56.6</td><td>61.8</td><td>1692</td><td>83.9</td><td>68.8</td><td>55.7</td><td>95.0%</td></tr><tr><td> $\alpha = 0.97$ </td><td>56.9</td><td>61.4</td><td>1657</td><td>84.2</td><td>67.9</td><td>55.0</td><td>94.3%</td></tr></table>

# 5 Conclusion

In this paper, we observe that visual tokens follow multiple group evolution directions across visionencoder layers, and that tokens persistently deviating from these group directions tend to carry important information. Based on this insight, we propose EvoCut, a training-free visual token compression method. EvoCut first clusters token evolution directions, then scores tokens by their deviation from these group directions, and accumulates these scores across layers. Experiments on image and video understanding across multiple LVLM backbones demonstrate that EvoCut consistently outperforms existing training-free compression methods under various token budgets.

# Limitations

Although EvoCut achieves strong performance on image and video understanding tasks, it still has several limitations. First, EvoCut requires access to the internal hidden states of the vision encoder to compute token evolution directions and perform clustering. This makes it incompatible with closedsource or API-only LVLMs where the vision encoder is not exposed, limiting its applicability in scenarios where only black-box model access is available. Second, our experiments focus on representative image and video understanding benchmarks; future work can further evaluate EvoCut on longer videos, higher-resolution inputs, more diverse multimodal reasoning tasks, and a broader range of vision-encoder backbones.

# Acknowledgments

This work is supported by the Natural Science Foundation of China under Grant No. 62305086, the China Postdoctoral Science Foundation under Grant No. 2023M740901, and the Natural Science Foundation of Heilongjiang Province of China under Grant No. LH2024F032.

# References

Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob L. Menick, Sebastian Borgeaud, and 8 others. 2022. Flamingo: A visual language model for few-shot learning. In Advances in Neural Information Processing Systems.   
Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond. arXiv preprint arXiv:2308.12966.   
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025. Qwen2.5-VL technical report. arXiv preprint arXiv:2502.13923.   
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, and 12 others. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems.   
Ricardo J. G. B. Campello, Davoud Moulavi, and Jörg Sander. 2013. Density-based clustering based on hierarchical density estimates. In Advances in Knowledge Discovery and Data Mining, pages 160–172. Springer.

David L. Chen and William B. Dolan. 2011. Collecting highly parallel data for paraphrase evaluation. In Annual Meeting of the Association for Computational Linguistics.   
Junjie Chen, Xuyang Liu, Zichen Wen, Yiyu Wang, Siteng Huang, and Honggang Chen. 2026. Variationaware vision token dropping for faster large visionlanguage models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition.   
Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024a. An image is worth 1/2 tokens after layer 2: Plug-andplay inference acceleration for large vision-language models. In European Conference on Computer Vision.   
Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, Bin Li, Ping Luo, Tong Lu, Yu Qiao, and Jifeng Dai. 2024b. InternVL: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198.   
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven C. H. Hoi. 2023. InstructBLIP: Towards general-purpose visionlanguage models with instruction tuning. In Advances in Neural Information Processing Systems.   
Tri Dao. 2023. FlashAttention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691.   
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. 2022. FlashAttention: Fast and memory-efficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems.   
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, Jakob Uszkoreit, and Neil Houlsby. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations.   
Yingqi Fan, Anhao Zhao, Jinlan Fu, Junlong Tong, Hui Su, Yijie Pan, Wei Zhang, and Xiaoyu Shen. 2025. VisiPruner: Decoding discontinuous cross-modal dynamics for efficient multimodal LLMs. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 18885–18902.   
Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, Yunsheng Wu, and Rongrong Ji. 2023. MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394.

Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In IEEE Conference on Computer Vision and Pattern Recognition.   
Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Barun Patra, Qiang Liu, Kriti Aggarwal, Zewen Chi, Johan Bjorck, Vishrav Chaudhary, Subhojit Som, Xia Song, and Furu Wei. 2023. Language is not all you need: Aligning perception with language models. In Advances in Neural Information Processing Systems.   
Drew A. Hudson and Christopher D. Manning. 2019. GQA: A new dataset for real-world visual reasoning and compositional question answering. In IEEE Conference on Computer Vision and Pattern Recognition.   
Yunseok Jang, Yale Song, Youngjae Yu, Youngjin Kim, and Gunhee Kim. 2017. TGIF-QA: Toward spatiotemporal reasoning in visual question answering. In IEEE Conference on Computer Vision and Pattern Recognition.   
Lei Jiang, Zixun Zhang, Yuting Zeng, Chunzhao Xie, Tongxuan Liu, Zhen Li, Lechao Cheng, and Xiaohua Xu. 2025. DCP: Dual-cue pruning for efficient large vision-language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 21191–21204.   
Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023a. SEED-Bench: Benchmarking multimodal LLMs with generative comprehension. arXiv preprint arXiv:2307.16125.   
Junnan Li, Dongxu Li, Silvio Savarese, and Steven C. H. Hoi. 2023b. BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning.   
Kunchang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. 2023c. VideoChat: Chat-centric video understanding. arXiv preprint arXiv:2305.06355.   
Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023d. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing.   
Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. 2023. Video-LLaVA: Learning united visual representation by alignment before projection. arXiv preprint arXiv:2311.10122.   
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024a. Improved baselines with visual instruction tuning. arXiv preprint arXiv:2310.03744.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In Advances in Neural Information Processing Systems.

Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, Kai Chen, and Dahua Lin. 2024b. MMBench: Is your multi-modal model an all-around player? In European Conference on Computer Vision, pages 216–233.   
Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, pages 2507–2521.   
Qiankun Ma, Ziyao Zhang, Haofei Wang, Jie Chen, Zhen Song, and Hairong Zheng. 2026. ApET: Approximation-error guided token compression for efficient VLMs. arXiv preprint arXiv:2602.19870.   
OpenAI. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning.   
Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. 2025. LLaVA-PruMerge: Adaptive token reduction for efficient large multimodal models. In IEEE/CVF International Conference on Computer Vision, pages 22857–22867.   
Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards VQA models that can read. In IEEE Conference on Computer Vision and Pattern Recognition.   
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Roziere, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. LLaMA: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in Neural Information Processing Systems.   
Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. 2025. Stop looking for “important tokens” in multimodal language models: Duplication matters more. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9961–9980.   
Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He,

Jiaqi Wang, Feng Wu, and Dahua Lin. 2025. PyramidDrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. In IEEE/CVF Conference on Computer Vision and Pattern Recognition.   
Jun Xu, Tao Mei, Ting Yao, and Yong Rui. 2016. MSR-VTT: A large video description dataset for bridging video and language. In IEEE Conference on Computer Vision and Pattern Recognition.   
Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. 2025. VisionZip: Longer is better but not necessary in vision language models. In IEEE/CVF Conference on Computer Vision and Pattern Recognition.   
Hang Zhang, Xin Li, and Lidong Bing. 2023. Video-LLaMA: An instruction-tuned audio-visual language model for video understanding. arXiv preprint arXiv:2306.02858.   
Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, and Shanghang Zhang. 2025. SparseVLM: Visual token sparsification for efficient vision-language model inference. In International Conference on Machine Learning.   
Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. MiniGPT-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.

# A Additional Method Details

# A.1 Pseudocode of EvoCut

Algorithm 1 summarizes the complete inferencetime procedure of EvoCut. Given the hidden states from the vision encoder, EvoCut starts from a selected layer $\ell _ { 0 } .$ , computes normalized token evolution directions between adjacent layers, estimates M group evolution directions, and accumulates token-level deviation scores across layers. After the final vision-encoder layer, the top-K patch tokens are retained and sent to the multimodal projector.

Algorithm 1 EvoCut visual token compression   
Require: Vision-encoder hidden states $\{X^{(\ell)}\}_{\ell=\ell_{0}-1}^{L}$ , number of retained tokens K, group direction number M, EMA decay factor $\alpha$ Ensure: Compressed visual token sequence $\hat{X}^{(L)}$ 1: Initialize token scores $s_{i}^{(\ell_{0}-1)} \leftarrow 0$ for all patch tokens $i = 1, \ldots, N$ 2: for $\ell = \ell_{0}$ to L do

3: for i = 1 to N do

4: $\vec{\Delta}_{i}^{(\ell)} \leftarrow \frac{x_{i}^{(\ell)} - x_{i}^{(\ell-1)}}{\|x_{i}^{(\ell)} - x_{i}^{(\ell-1)}\|_{2}}$ 5: end for

6: Cluster $\{\vec{\Delta}_{i}^{(\ell)}\}_{i=1}^{N}$ into M groups $\{C_{m}^{(\ell)}\}_{m=1}^{M}$ with K-means

7: for m = 1 to M do

8: $\vec{u}_{m}^{(\ell)} \leftarrow \text{Norm}\left(\frac{1}{|C_{m}^{(\ell)}|} \sum_{i \in C_{m}^{(\ell)}} \vec{\Delta}_{i}^{(\ell)}\right)$ 9: end for

10: for i = 1 to N do

11: $a_{i}^{(\ell)} \leftarrow \max_{m} \cos(\vec{\Delta}_{i}^{(\ell)}, \vec{u}_{m}^{(\ell)})$ 12: $r_{i}^{(\ell)} \leftarrow 1 - a_{i}^{(\ell)}$ 13: $s_{i}^{(\ell)} \leftarrow \alpha s_{i}^{(\ell-1)} + (1 - \alpha)r_{i}^{(\ell)}$ 14: end for

15: end for

16: $S \leftarrow$ indices of the top-K tokens according to $s_{i}^{(L)}$ 17: $\hat{X}^{(L)} \leftarrow \{x_{i}^{(L)} \mid i \in S\}$ 18: return $\hat{X}^{(L)}$

# A.2 Dataset Descriptions

We evaluate EvoCut on a diverse set of image and video understanding benchmarks to examine whether the proposed token-compression criterion preserves different types of multimodal information. The datasets used in our experiments are summarized below.

# Image understanding benchmarks.

VQAv2. VQAv2 evaluates general visual question answering over natural images and requires models to answer diverse questions about objects, attributes, actions, and scenes (Goyal et al., 2017).

GQA. GQA focuses on compositional visual reasoning and tests whether models can answer

questions that require object recognition, relation understanding, and multi-step reasoning (Hudson and Manning, 2019).

TextVQA. TextVQA measures text-rich image understanding, where models must recognize and reason over textual content appearing in images (Singh et al., 2019).

ScienceQA. ScienceQA is a multimodal science question answering benchmark that evaluates knowledge-intensive reasoning over imagequestion pairs (Lu et al., 2022).

MME. MME provides a comprehensive evaluation of LVLM perception and cognition abilities, covering tasks such as object existence, count, position, color, and reasoning (Fu et al., 2023).

POPE. POPE evaluates object hallucination in LVLMs through binary questions and includes random, popular, and adversarial sampling settings (Li et al., 2023d).

MMBench and MMBench-CN. MMBench evaluates multimodal understanding with multiplechoice questions, and MMBench-CN provides a Chinese counterpart for testing cross-lingual multimodal capability (Liu et al., 2024b).

SEED-Bench. SEED-Bench is a multidimensional benchmark for evaluating generative multimodal comprehension across images and videos (Li et al., 2023a).

LLaVA-Bench. LLaVA-Bench contains openended image-instruction examples and is used to assess instruction-following multimodal response quality (Liu et al., 2023).

# Video understanding benchmarks.

TGIF-QA. TGIF-QA evaluates video question answering over GIF-style videos and emphasizes temporal events, actions, and repeated motion patterns (Jang et al., 2017).

MSVD-QA. MSVD-QA is built on the MSVD video dataset and tests question answering over short web videos with diverse daily activities (Chen and Dolan, 2011).

MSRVTT-QA. MSRVTT-QA evaluates openended video question answering on a large collection of web videos covering broad topics and scenes (Xu et al., 2016).

# A.3 FLOPs Analysis

EvoCut reduces the computational cost mainly by shortening the visual sequence before the multimodal projector and the LLM prefilling stage. Since compression is performed at the output of the vision encoder, EvoCut does not reduce the FLOPs of the vision encoder itself. Instead, it reduces the FLOPs of all subsequent modules whose cost depends on the number of visual tokens.

Let N denote the original number of visual tokens and K denote the number of retained visual tokens after compression. Let T be the number of text tokens, d be the hidden size of the LLM, $d _ { \mathrm { f } }$ be the feed-forward dimension, and $L _ { \mathrm { l l m } }$ be the number of LLM layers. For a decoder-only Transformer layer, the prefilling FLOPs can be approximated, up to constant factors, as

$$
\mathcal {F} _ {\text { layer }} (S) \approx 4 S d ^ {2} + 2 S ^ {2} d + 2 S d d _ {\mathrm{ff}}, \tag {7}
$$

where $S = T { + } N$ before compression and $S = T \mathrm { - }$ 上 K after compression. The three terms correspond to attention projections, self-attention computation, and feed-forward layers, respectively. Therefore, the LLM-side FLOPs reduction is approximately

$$
\begin{array}{l} \Delta \mathcal {F} _ {\mathrm{llm}} \approx L _ {\mathrm{llm}} \left[ \mathcal {F} _ {\text {layer}} (T + N) \right. \tag {8} \\ \left. - \mathcal {F} _ {\mathrm{layer}} (T + K) \right]. \\ \end{array}
$$

This reduction contains both linear terms in the sequence length and the quadratic self-attention term. Thus, the benefit becomes more significant for high-resolution images and videos, where N is large.

The multimodal projector also benefits from token reduction. If the projector maps each visual token from dimension $d _ { v }$ to the LLM hidden size $d ,$ its FLOPs decrease from approximately $2 N d _ { v } d$ to $2 K d _ { v } d .$ . Although this saving is smaller than the LLM-side reduction, it further lowers the prefilling cost.

EvoCut introduces only a lightweight scoring overhead. For $L _ { e }$ evaluated layer transitions, token-direction computation costs $O ( L _ { e } N d _ { v } )$ . Kmeans clustering and nearest-direction scoring cost $O ( L _ { e } I N M d _ { v } )$ and $O ( L _ { e } N M d _ { v } )$ , respectively, where I is the number of K-means iterations and M is the number of group evolution directions. Since we use a small default value $M = 8$ and do not perform any additional vision-encoder forward pass, this overhead is small compared with the FLOPs saved in the LLM prefilling stage. For LLaVA-1.5- 7B, we estimate this overhead under the default setting with $N = 5 7 6 , d _ { v } = 1 0 2 4 , M = 8 , L _ { e } = 1 3 .$ , and I = 2 K-means iterations. The resulting Evo-Cut scoring cost is approximately 0.40 GFLOPs, i.e., 0.00040 TFLOPs. Compared with the 2.04 TFLOPs of LLaVA-1.5-7B after compression to 64 visual tokens, this overhead accounts for only about 0.020% of the total inference FLOPs.

The measured TFLOPs in Tables 5 and 6 are consistent with this analysis. On LLaVA-1.5-7B, reducing visual tokens from 576 to 64 lowers TFLOPs from 8.82 to 2.04, corresponding to a 76.9% reduction. On LLaVA-NeXT-7B, reducing visual tokens from 2880 to 160 lowers TFLOPs from 31.03 to 4.74, corresponding to an 84.7% reduction. These results show that visual token compression substantially reduces the dominant prefilling computation, while the remaining cost mainly comes from the unchanged vision encoder and lightweight EvoCut scoring.

A.4 Evolution Deviation Visualization   
![](images/df8de5d721d20c5d88858c56aa75fba8b792c39cdc60c4b7d948fcf50e7abf03.jpg)

<details>
<summary>bar</summary>

Qwen2.5-VL-7B — POPE
| VIT Encoder Layer Index | Qwen2.5-VL-7B — TextVQA |
|---|---|
| 1 | 2.3 |
| 2 | 2.5 |
| 3 | 4.8 |
| 4 | 4.9 |
| 5 | 5.1 |
| 6 | 8.3 |
| 7 | 6.6 |
| 8 | 11.1 |
| 9 | 3.5 |
| 10 | 5.7 |
| 11 | 2.8 |
| 12 | 6.7 |
| 13 | 3.9 |
| 14 | 3.2 |
| 15 | 6.7 |
| 16 | 6.4 |
| 17 | 4.6 |
| 18 | 4.4 |
| 19 | 4.1 |
| 20 | 5.3 |
| 21 | 5.9 |
| 22 | 9.7 |
| 23 | 11.6 |
| 24 | 15.4 |
| 25 | 12.4 |
| 26 | 12.2 |
| 27 | 14.8 |
| 28 | 15.0 |
| 29 | 13.8 |
| 30 | 15.3 |
| 31 | 7.8 |
| 32 | 8.4 |
The chart displays two vertical bar charts comparing the number of evolution directions for each VIT Encoder Layer Index in Qwen2.5-VL-7B and TextVQA models. The Y-axis represents the number of evolution directions, and the X-axis represents the VIT Encoder Layer Index (1 to 32). The top chart shows the absolute direction count per layer, while the bottom chart shows the corresponding direction count per layer. Both charts use blue bars to represent values for each layer, with numerical labels above each bar indicating the direction count for that layer. The text series 'TextVQA' is shown in the legend but not plotted on the charts.
</details>

Figure 5: Supplementary visualization for the layerwise token evolution analysis in Section 3.2. The figure further illustrates how token evolution directions deviate from group evolution directions across multiple visionencoder layers, complementing the findings in Figure 3.

# A.5 Qualitative Visualization

Figure 6 provides qualitative examples of token retention after EvoCut compression. Retained tokens are concentrated on visually informative regions, while filtered tokens mostly lie in background and repetitive-texture areas.

![](images/d7b899390faa9d52c836d19cfff07a4ee82fbf264e776ca538b99a29146bfc27.jpg)  
Figure 6: Qualitative visualization of token retention after EvoCut compression. Retained tokens (highlighted) are concentrated on visually informative regions, while filtered tokens mostly lie in background and repetitive-texture areas.