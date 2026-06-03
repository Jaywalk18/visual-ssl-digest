# InfoMerge: Information-aware Token Compression for Efficient Video Large Language Models

Xinxin Liu , Shiwei Gan\* , Xiao Liu , Yafeng Yin , Lei Xie , Sanglu Lu

State Key Laboratory of Novel Software Technology, Nanjing University Nanjing 210023, China

231880474@smail.nju.edu.cn,sw@nju.edu.cn, xiaoliu@smail.nju.edu.cn

{yafeng,lxie,sanglu}@nju.edu.cn

\*Corresponding author.

# Abstract

Video Large Language Models (Video-LLMs) achieve strong performance in video understanding, but their excessive visual tokens bring substantial computational overhead. Existing training-free compression methods improve inference efficiency by reducing visual tokens, yet they often rely on local adjacent-frame similarity for temporal redundancy estimation or allocate token budgets mainly according to segment length. Such designs are sensitive to frame-level noise and fail to capture the non-uniform information distribution of realworld videos. To address these challenges, we propose InfoMerge, a training-free visual token compression method that improves token utilization through robust redundancy estimation and content-aware budget allocation. Specifically, we propose the Temporal Fingerprint Difference: a segment-level secondorder temporal redundancy estimation strategy, which models the temporal similarity structure of tokens at the same spatial positions within each segment. We further introduce Content-Aware Budget Allocation (CABA), which dynamically allocates segment-level token budgets based on segment uniqueness and spectralentropy-based representational richness. By reducing repeated preservation of redundant static regions and allocating more tokens to informative segments, InfoMerge makes better use of the limited token budget while maintaining strong performance. Extensive experiments show that InfoMerge achieves strong efficiency–accuracy trade-offs across multiple benchmarks and backbones, with more pronounced advantages under aggressive compression. On LLaVA-OneVision-7B, InfoMerge retains 98.8% of the original average performance while reducing 85% of visual tokens and accelerating the prefill stage by 4.24×.

Frame 1

Frame 2

![](images/7a48ed55ad8d99765af4b429ead443ab586a24105048a6afcae36c414afa9033.jpg)

![](images/3f381e5b2415abb3848c53f426e6357f6862e8d68f76b793eae9474631fddf39.jpg)

![](images/8dfb375cd4ddf0b8501828b53707a775959b48e091a5d9101b13e0554d47ada9.jpg)

(a) Local similarity estima-similarity tion.   
![](images/fcea13f7ccdc7186b37c4a674adbe6c27dd1eb350e9be0fa59ae935e5c7a48a1.jpg)

![](images/df2086e706a4d8252b746f6efb5dc8e61a8da2fd396dbf58947de5c97a0aa334.jpg)

(b) Second-order temporaltemporal consistency consistency.   
![](images/95f3865056d2ff1bd6ded34e2e0a657412c8ba3a2356aa982c1edb794ac0dd81.jpg)

<details>
<summary>text_image</summary>

Frame 4
Frame 11
Frame 14
Frame 17
Question: Why does the roof object fly?
A. Explosion
B. Child throws
C. Balloon
D. Lunar approach
FastVID:
A. Explosion.
Ours:
D. Lunar approach.
(wrong)
(correct)
Ours only
FastVID only
</details>

(c) Qualitative comparison between FastVID and ours.   
Figure 1: Qualitative analysis of token selection. (a) Prior methods rely on local similarity between tokens at the same spatial position in adjacent frames. (b) We further model second-order temporal consistency across frames. (c) Compared with FastVID, our method retains more informative tokens related to the flying roof object, leading to the correct answer.

# 1 Introduction

Video-LLMs have achieved remarkable progress in complex video understanding tasks (Li et al., 2024a; Bai et al., 2025; Li et al., 2025; Cheng et al., 2024; Wang et al., 2024; Zhang et al., 2024). Despite their strong capabilities, the high computational and memory costs induced by dense video tokens make token compression essential for efficient Video LLM inference. Existing training-free methods can be broadly categorized into spatial pruning methods (Yang et al., 2025; Chen et al., 2024), temporal redundancy reduction methods (Tao et al., 2025), staged spatio-temporal compression methods (Shen et al., 2026; Shao et al., 2025), and unified spatio-temporal approaches (Du et al., 2026; Zhang et al., 2026). Despite their progress, most existing methods rely on a common underlying assumption: local pairwise similarity is sufficient to characterize redundancy. In practice, however, this assumption becomes fragile under real-world video noise, motion, and viewpoint variations, leading to unstable redundancy estimation and suboptimal token allocation.

Moreover, segment-based methods such as FastVID (Shen et al., 2026) allocate token budgets mainly according to segment length, ignoring the intrinsic information content of each segment. Based on these observations, we reformulate video token compression as a problem of second-order temporal redundancy estimation and informationaware budget allocation, aiming to reduce repeated preservation of static tokens and assign more budget to semantically distinctive and representationrich segments.

To this end, we propose InfoMerge, a trainingfree video token compression method from a segment-structured redundancy perspective. Unlike previous methods that estimate temporal redundancy using first-order adjacent-frame similarity at the same spatial position, InfoMerge introduces Temporal Fingerprint Difference (TFD) to evaluate temporal redundancy from a second-order perspective by modeling pairwise similarity structures within each segment. We further propose Content-Aware Budget Allocation (CABA), which jointly considers segment-level uniqueness and internal representation complexity to adaptively allocate token budgets according to semantic distinctiveness and spectral information density.

Finally, we integrate TFD and CABA into a spatio-temporal compression pipeline that performs redundancy-aware token merging in a training-free and plug-and-play manner. The proposed method can be seamlessly integrated into existing VLLMs. Figure 1 qualitatively shows that InfoMerge retains more informative visual evidence than FastVID under the same retention ratio, which helps the model produce the correct answer. Extensive experiments demonstrate that InfoMerge achieves a superior efficiency–accuracy trade-off. Our contributions are summarized as follows:

• We propose Temporal Fingerprint Difference, which estimates temporal redundancy from a structured second-order perspective and provides a robust static-token prior for more efficient token selection.   
• We introduce Content-Aware Budget Allo-

cation, which adaptively allocates token budgets according to segment-level uniqueness and representational richness, assigning more tokens to informative segments.

• We propose InfoMerge, a training-free and plug-and-play method that can be seamlessly integrated into existing Video-LLMs without fine-tuning. InfoMerge achieves strong efficiency–accuracy trade-offs, retaining most of the original model performance even under extremely low token retention ratios.

# 2 Related Work

Video Large Language Models. Recent Video-LLMs (Li et al., 2025; Cheng et al., 2024; Li et al., 2024b,a; Zhang et al., 2024; Bai et al., 2025; Wang et al., 2025) have achieved strong performance in video understanding by extending image-based VLMs to video inputs. However, these models still pass thousands of visual tokens to the downstream LLM, leading to substantial computational overhead due to the quadratic complexity of attention (Vaswani et al., 2017). Although some recent works (Lin et al., 2024; Liu et al., 2025) improve token efficiency through model-level optimization, they typically require additional training or fine-tuning. Therefore, efficient training-free token compression remains important for practical VLLM inference.

Visual Token Compression. Token compression is an effective way to reduce token redundancy in ViTs and LLMs. Training-free visual token compression aims to reduce the inference cost of VLMs and Video-LLMs without additional model training. Existing methods mainly rely on token pruning or merging. Spatial methods such as ToMe (Bolya et al., 2022), VisionZip (Yang et al., 2025), and FastV (Chen et al., 2024) reduce visual tokens according to token similarity or attention scores. Video-oriented methods further exploit temporal redundancy: DyCoke (Tao et al., 2025), TempMe (Shen et al., 2025), and PruneVID (Huang et al., 2025) merge similar tokens across frames, while FastVID (Shen et al., 2026) performs segmentation-based token selection and merging. HoliTom (Shao et al., 2025) further considers global redundancy for temporal token organization. However, two limitations remain. First, temporal redundancy is often characterized by firstorder local cues, such as adjacent-frame similarity or attention scores. These cues can be unstable under camera motion, illumination changes, or local perturbations, and they do not explicitly model whether a region is temporally static or dynamically informative. Second, segment-based methods usually allocate token budgets mainly according to segment length, which cannot capture the non-uniform information density of real videos. In contrast, our method uses Temporal Fingerprint Difference to model second-order temporal consistency within each segment, and Content-Aware Budget Allocation to allocate more tokens to semantically unique and representation-rich segments.

![](images/3ff7e0e3815f86dd4515b8b0b5b6192c9ae9811157939c0084150157991972f3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Large Language Model"] --> B["Tokenizer"]
    B --> C["Spatio-Temporal Compression"]
    C --> D["Segment Uniqueness"]
    C --> E["Information Density"]
    D --> F["CABA"]
    E --> F
    F --> G["TFD"]
    G --> H["seg1"]
    G --> I["seg2"]
    H --> J["Visual Encoder & Projector"]
    I --> J
    J --> K["Video Input"]
    
    C --> L["DTM hard mask"]
    L --> M["Frame1"]
    L --> N["Frame2"]
    L --> O["Frame3"]
    M --> P["DPC-KNN clustering"]
    N --> P
    O --> P
    P --> Q["merged context tokens"]
    
    C --> R["ATS with penalty"]
    R --> S["Frame1"]
    R --> T["Frame2"]
    R --> U["Frame3"]
    S --> V["Frame1"]
    S --> W["Frame2"]
    S --> X["Frame3"]
    T --> V
    U --> V
    V --> Y["selected salient tokens"]
    
    C --> Z["Concatenate compressed tokens"]
    
    Z --> AA["DSAT: DPC-KNN with hard mask"]
    
    AA --> AB["Frame1"]
    AA --> AC["Frame2"]
    AA --> AD["Frame3"]
    AB --> AE["Frame1"]
    AB --> AF["Frame2"]
    AB --> AG["Frame3"]
    AC --> AE
    AD --> AE
    AE --> AH["Selected Context Tokens"]
    
    Z --> AI["ATS: penalize repeated static positions"]
    
    AI --> AJ["Frame1"]
    AI --> AK["Frame2"]
    AI --> AL["Frame3"]
    AJ --> AM["Frame1"]
    AJ --> AN["Frame2"]
    AJ --> AO["Frame3"]
    AK --> AM
    AL --> AM
    AM --> AP["Selected Context Tokens"]
    
    Z --> AQ["TFD: temporal fingerprint difference"]
    
    AQ --> AR["seg1"]
    AQ --> AS["seg2"]
    
    AR --> AT["Segment tokens"]
    AT --> AU["Frame1"]
    AR --> AV["Segment tokens"]
    AV --> AW["Frame2"]
    AR --> AX["Segment tokens"]
    AX --> AY["Frame3"]
    
    AX --> AZ["Pairwise fingerprint"]
    
    AZ --> BA["Frame1"]
    AZ --> BB["Frame2"]
    
    AZ --> BC["Frame3"]
    
    BA --> BD["static token"]
    BD --> BE["dynamic token"]
    
    CA["CABA: Content-Aware Budget Allocation"] --> CB["Unique / dense segments receive more context tokens"]
    
    CB --> CC["Segment feature"]
    CC --> DD["Uniqueness"]
    DD --> DE["z-score fusion"]
    
    CE["Context token Budget"] --> CF["seg1"]
    CE --> CG["seg2"]
    
    CE --> DH["entropy"]
```
</details>

Figure 2: Overview of the proposed InfoMerge method.

# 3 Method

Overview. As illustrated in Fig. 2, InfoMerge consists of four components: dynamic temporal segmentation, Temporal Fingerprint Difference (TFD), Content-Aware Budget Allocation (CABA), and spatio-temporal compression. Given sampled video tokens, we first divide frames into temporally coherent segments. TFD then identifies temporally redundant spatial positions within each segment, while CABA adaptively allocates token budgets according to segment information content. Finally, we integrate both the TFD prior and CABA-guided segment budgets into spatio-temporal compression to suppress repeated preservation of static regions.

![](images/b6ff03e9e2c87b29840e6c65f31616ae86d5d5194b14c54e1151f072902bdd68.jpg)  
Figure 3: Illustration of the proposed TFD. For each spatial position, we construct an intra-segment temporal similarity matrix and compute the differences between adjacent temporal fingerprints to estimate temporal dynamics.

# 3.1 Dynamic Temporal Segmentation

Given a video consisting of L frames, where each frame contains N visual tokens with hidden dimension D, we first apply Dynamic Temporal Segmentation (DySeg) to divide the sampled video frames into temporally coherent segments. DySeg estimates transition strength using the cosine similarity between adjacent frame-level global features and selects low-similarity transitions as segment boundaries. This produces a sequence of ordered segments $\{ \gamma _ { k } \} _ { k = 1 } ^ { K }$ . For each segment $\nu _ { k }$ , we use local frame indices $l \in \{ 1 , \ldots , | \mathcal { V } _ { k } | \}$ , where $| \nu _ { k } |$ denotes the segment length. These segments serve as the basic units for our subsequent TFD, CABA, and Spatio-Temporal Compression.

# 3.2 Temporal Fingerprint Difference

Existing methods (Tao et al., 2025; Shao et al., 2025) estimate temporal redundancy mainly through local similarities between adjacent frames. However, such short-range measurements are sensitive to frame-level noise and often fail to reliably distinguish temporally static tokens from dynamically informative ones. We argue that truly redundant tokens should exhibit globally consistent temporal patterns within a segment, rather than only local frame-to-frame similarity.

To this end, we propose a Temporal Fingerprint Difference (TFD) module, which models segment-level temporal consistency for robust redundancy estimation, as illustrated in Fig. 3. The design is conceptually inspired by MADD (Sarkar and Ghosh, 2019), which characterizes highdimensional samples according to their relations to the data cloud. TFD constructs a segment-level second-order temporal fingerprint for each spatial token position.

Specifically, given video features ${ \textbf { H } } \in$ $\mathbb { R } ^ { L \times N \times D }$ , let $\mathbf { H } _ { l , p } ^ { ( k ) } \in \mathbb { R } ^ { D }$ denote the token at spatial position $p$ in frame l of the k-th segment $\nu _ { k }$ . For each spatial position $p ,$ we first normalize token features across frames and then construct a temporal similarity fingerprint matrix:

$$
\mathbf {F} _ {p} ^ {(k)} [ l, l ^ {\prime} ] = \operatorname{Cos} \left(\frac {\mathbf {H} _ {l , p} ^ {(k)}}{\| \mathbf {H} _ {l , p} ^ {(k)} \| _ {2}}, \frac {\mathbf {H} _ {l ^ {\prime} , p} ^ {(k)}}{\| \mathbf {H} _ {l ^ {\prime} , p} ^ {(k)} \| _ {2}}\right) \tag {1}
$$

where $l , l ^ { \prime } \in \{ 1 , \ldots , | \mathcal { V } _ { k } | \}$ , and $\cos ( \cdot , \cdot )$ denotes the cosine similarity function and F(k)p $\mathbf { F } _ { p } ^ { ( k ) } \in$ $\mathbb { R } ^ { | \mathcal { V } _ { k } | \times | \mathcal { V } _ { k } | }$ characterizes the pairwise temporal similarity structure of spatial position $p$ throughout the entire segment. Intuitively, temporally static regions tend to maintain stable similarity structures across time, while dynamic regions exhibit larger temporal variations. We measure temporal dynamics by computing the average absolute difference between adjacent temporal fingerprints, corresponding to adjacent rows of the fingerprint matrix:

$$
\operatorname{TFD} ^ {(k)} (p) = \frac {\sum_ {l = 1} ^ {| \mathcal {V} _ {k} | - 1} \sum_ {l ^ {\prime} = 1} ^ {| \mathcal {V} _ {k} |} \left| \mathbf {F} _ {p} ^ {(k)} [ l + 1 , l ^ {\prime} ] - \mathbf {F} _ {p} ^ {(k)} [ l , l ^ {\prime} ] \right|}{(| \mathcal {V} _ {k} | - 1) | \mathcal {V} _ {k} |} \tag {2}
$$

A smaller TFD value indicates stronger temporal consistency and thus higher redundancy. Accordingly, for each segment, we select the $\lfloor \rho N \rfloor$ spatial positions with the smallest TFD scores as static candidates $S ^ { ( k ) }$ .

Segment-wise Low-Rank Structure   
![](images/b163b6b1ab717090565cd3db29a3f906e002dc890d74e66dad9d18aec0b87eb7.jpg)

<details>
<summary>line</summary>

| Singular value index | Normalized spectral energy (log scale) |
| -------------------- | -------------------------------------- |
| 0                    | 10^-0                                  |
| 200                  | ~10^-4                                 |
| 400                  | ~10^-6                                 |
| 600                  | ~10^-8                                 |
| 800                  | ~10^-10                                |
| 900                  | ~10^-14                                |
</details>

Figure 4: Analysis of segment-wise low-rank structures in video token representations. The rapidly decaying singular-value spectra indicate strong redundancy within the segment.

# 3.3 Content-Aware Budget Allocation

Video information is often unevenly distributed across time: long segments may contain mostly static or repetitive content, while short segments can include key actions or scene changes. Therefore, allocating token budgets only according to segment length may under-preserve short but informative segments. To address this issue, we propose Content-Aware Budget Allocation (CABA), which estimates segment-level information content from segment uniqueness and representational richness.

Segment Uniqueness. We compute the segmentlevel representation of the k-th segment $\nu _ { k }$ as:

$$
\bar {\mathbf {x}} _ {k} = \frac {1}{| \mathcal {V} _ {k} | N} \sum_ {l = 1} ^ {| \mathcal {V} _ {k} |} \sum_ {p = 1} ^ {N} \mathbf {H} _ {l, p} ^ {(k)}, \tag {3}
$$

Further, we compute the global video representation $\bar { \bf x } _ { \mathrm { g l o b } }$ :

$$
\bar {\mathbf {x}} _ {\text {glob}} = \frac {\sum_ {k = 1} ^ {K} | \mathcal {V} _ {k} | \bar {\mathbf {x}} _ {k}}{\sum_ {k = 1} ^ {K} | \mathcal {V} _ {k} |}. \tag {4}
$$

We define the segment uniqueness score as the cosine distance between the segment representation and the global video representation:

$$
u _ {k} = 1 - \operatorname{Cos} (\bar {\mathbf {x}} _ {k}, \bar {\mathbf {x}} _ {\mathrm{glob}}). \tag {5}
$$

A larger $u _ { k }$ indicates that the segment is more semantically distinctive from the overall video.

Representational Richness. For the k-th segment $\nu _ { k } .$ , we flatten its visual tokens into $\mathbf { X } _ { k } \in \mathbb { R } ^ { | \mathcal { V } _ { k } | N \times D }$ and compute its singular values $\{ \sigma _ { j } \} _ { j = 1 } ^ { R }$ , where $R = \operatorname* { m i n } ( | \mathcal { V } _ { k } | N , D )$ . As shown in Fig. 4, video token representations exhibit a clear low-rank structure within a segment.The top-20 singular directions already explain 88.1% of the spectral energy. This motivates the use of spectral entropy to estimate segment-level representational richness. We estimate representational richness using normalized spectral entropy:

$$
\pi_ {j} = \frac {\sigma_ {j} ^ {2}}{\sum_ {\ell = 1} ^ {R} \sigma_ {\ell} ^ {2}} \tag {6}
$$

We then compute the normalized spectral entropy:

$$
e _ {k} = \frac {- \sum_ {j = 1} ^ {R} \pi_ {j} \log \pi_ {j}}{\log R}. \tag {7}
$$

A larger $e _ { k }$ indicates that the segment spans more diverse principal directions and is therefore less likely to be represented by a few redundant patterns.

Budget Allocation. We fuse segment uniqueness and representational richness after z-score normalization as $m _ { k } = \alpha z ( u _ { k } ) + \beta z ( e _ { k } )$ . where $\alpha$ and $\beta$ control the contributions of the two factors. We then use a sigmoid function to map the fused score to a smooth budget modulation coefficient and multiply it by the segment length:

$$
w _ {k} = \text { sigmoid } (\tau m _ {k}) \cdot | \mathcal {V} _ {k} |, \tag {8}
$$

where $\tau$ is the temperature parameter that controls the sharpness of budget redistribution. Finally, the context-token budget for the k-th segment is computed as

$$
B _ {k} = \max \left(1, \text { round } \left(\frac {L T _ {c} w _ {k}}{\sum_ {j = 1} ^ {K} w _ {j}}\right)\right), \tag {9}
$$

where $T _ { c }$ denotes the average context-token budget per frame. In this way, CABA allocates more tokens to semantically unique and representation-rich segments.

# 3.4 Spatio-Temporal Compression

We follow the Attention-Based Token Selection (ATS)–Density-Based Token Merging (DTM) compression pipeline, where the TFD-estimated static prior guides redundancy-aware token selection and merging, while the CABA-guided segment budgets adaptively control the context-token allocation across segments.

Attention-Based Token Selection with Penalty. We use the attention from the [CLS] token as a saliency estimate. For SigLIP-based Video-LLMs, we follow FastVID (Shen et al., 2026) to obtain these attention scores with the pretrained SigLIP head. However, high-attention static regions may be repeatedly selected across frames. To alleviate this issue, we introduce a redundancy-sensitive suppression mechanism guided by the static set of candidates $S ^ { ( k ) }$ . For frame l in segment $\nu _ { k }$ , let ${ \bf A } _ { l } \in \mathbb { R } ^ { N }$ denote flattened attention scores. We maintain a historical static set P (k)l−1, $\mathcal { P } _ { l - 1 } ^ { ( k ) }$ which records the static token positions selected by ATS in the previous $l - 1$ frames of the current segment. We only penalize positions that are both static candidates and have already been selected:

$$
\begin{array}{l} \mathbf {A} _ {l, p} \leftarrow \mathbf {A} _ {l, p} - \lambda \cdot \operatorname{std} (\mathbf {A} _ {l}), \tag {10} \\ p \in \mathcal {S} ^ {(k)} \cap \mathcal {P} _ {l - 1} ^ {(k)}. \\ \end{array}
$$

where λ controls the redundancy suppression strength. ATS then selects the top positions $T _ { s }$ according to the updated attention scores, forming the salient-token set $\mathcal { T } _ { l } ^ { \mathrm { A T S , ( k ) } }$ ATS,(k), where Ts denotes $T _ { s }$ the salient-token budget per-frame. If a selected position belongs to the static candidate set, it is added to the historical static set:

$$
\mathcal {P} _ {l} ^ {(k)} = \mathcal {P} _ {l - 1} ^ {(k)} \cup \left(\mathcal {T} _ {l} ^ {\mathrm{ATS}, (\mathrm{k})} \cap \mathcal {S} ^ {(k)}\right). \tag {11}
$$

For these selected static positions, we further replace their token representations in the current and subsequent frames of the segment with a segmentlevel merged representation at the same spatial position. Details are provided in Appendix A.7.

Hard-Masked Density-Based Token Merging. For each segment, we sample anchor frames at a fixed interval $p$ and evenly distribute the segmentlevel context-token budget across these anchor frames. Within each anchor frame, for token position $p ,$ , DTM computes a density-peak score $q _ { a , p } ^ { ( k ) } .$ where a denotes the index of the anchor frame in segment $\nu _ { k }$ . High-scoring tokens are selected as context anchors. However, without additional constraints, multiple anchor frames may repeatedly select the same static regions as cluster centers, leading to redundant context preservation. To improve contextual diversity, we introduce a hard mask. Let $\mathcal { Q } _ { a - 1 } ^ { ( k ) }$ denote the set of static positions already selected by the previous a − 1 anchor frames within $\nu _ { k }$ . For the current anchor frame $^ { a , }$ if position $p$ belongs to $\mathcal { Q } _ { a - 1 } ^ { ( k ) }$ Q(k) , its density-peak score is directly suppressed:

Table 1: Comparison on LLaVA-OneVision-7B. The A%/B% retention ratio indicates that A% of the LLM input tokens are retained, and subsequently compressed to B% during the LLM forward pass. The Avg. Score is computed over VideoMME Overall, MLVU, and LongVideoBench. Best results are in bold, second best underlined. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">FLOPs (T)↓</td><td rowspan="2">FLOPs Ratio</td><td rowspan="2">Retention Ratio</td><td colspan="4">VideoMME↑</td><td rowspan="2">MLVU↑</td><td rowspan="2">Long VideoBench↑</td><td colspan="2">Avg.↑</td></tr><tr><td>Short</td><td>Medium</td><td>Long</td><td>Overall</td><td>Score</td><td>%</td></tr><tr><td>LLaVA-OV-7B (Li et al., 2024a)</td><td>48.8</td><td>100.0%</td><td>100%</td><td>70.2</td><td>56.8</td><td>48.8</td><td>58.6</td><td>64.9</td><td>56.4</td><td>59.97</td><td>100.0</td></tr><tr><td>FastV (Chen et al., 2024)</td><td>9.4</td><td>19.2%</td><td>100%/15%</td><td>65.6</td><td>55.1</td><td>46.9</td><td>55.9</td><td>60.2</td><td>53.2</td><td>56.43</td><td>94.1</td></tr><tr><td>VisionZip (Yang et al., 2025)</td><td>6.3</td><td>12.9%</td><td>15%</td><td>67.4</td><td>57.6</td><td>48.1</td><td>57.7</td><td>60.0</td><td>55.8</td><td>57.83</td><td>96.4</td></tr><tr><td>MMG-Vid (Ma et al., 2026)</td><td>6.3</td><td>12.9%</td><td>15%</td><td>-</td><td>-</td><td>-</td><td>57.9</td><td>61.6</td><td>55.9</td><td>58.47</td><td>97.5</td></tr><tr><td>FastVID (Shen et al., 2026)</td><td>6.3</td><td>12.9%</td><td>15%</td><td>69.3</td><td>56.4</td><td>47.6</td><td>57.8</td><td>62.9</td><td>56.2</td><td>58.97</td><td>98.3</td></tr><tr><td>Ours</td><td>6.3</td><td>12.9%</td><td>15%</td><td>69.6</td><td>56.8</td><td>48.1</td><td>58.2</td><td>63.1</td><td>56.4</td><td>59.23</td><td>98.8</td></tr><tr><td>FastV (Chen et al., 2024)</td><td>7.4</td><td>15.1%</td><td>100%/10%</td><td>61.7</td><td>54.1</td><td>47.1</td><td>54.3</td><td>58.6</td><td>52.1</td><td>55.00</td><td>91.7</td></tr><tr><td>VisionZip (Yang et al., 2025)</td><td>4.2</td><td>8.5%</td><td>10%</td><td>63.6</td><td>56.0</td><td>48.2</td><td>55.9</td><td>59.7</td><td>54.5</td><td>56.70</td><td>94.5</td></tr><tr><td>FastVID (Shen et al., 2026)</td><td>4.2</td><td>8.5%</td><td>10%</td><td>67.2</td><td>55.7</td><td>47.6</td><td>56.8</td><td>62.1</td><td>55.7</td><td>58.20</td><td>97.1</td></tr><tr><td>Ours</td><td>4.2</td><td>8.5%</td><td>10%</td><td>68.9</td><td>55.9</td><td>48.2</td><td>57.7</td><td>61.7</td><td>55.2</td><td>58.20</td><td>97.1</td></tr><tr><td>FastV (Chen et al., 2024)</td><td>5.4</td><td>11.1%</td><td>100%/5%</td><td>56.3</td><td>51.7</td><td>44.1</td><td>50.7</td><td>56.1</td><td>48.5</td><td>51.77</td><td>86.3</td></tr><tr><td>VisionZip (Yang et al., 2025)</td><td>2.1</td><td>4.2%</td><td>5%</td><td>59.8</td><td>51.7</td><td>45.9</td><td>52.4</td><td>58.5</td><td>48.4</td><td>53.10</td><td>88.5</td></tr><tr><td>FastVID (Shen et al., 2026)</td><td>2.1</td><td>4.2%</td><td>5%</td><td>63.3</td><td>51.8</td><td>45.6</td><td>53.6</td><td>58.8</td><td>51.4</td><td>54.60</td><td>91.0</td></tr><tr><td>Ours</td><td>2.1</td><td>4.2%</td><td>5%</td><td>65.2</td><td>53.0</td><td>45.7</td><td>54.6</td><td>59.5</td><td>52.1</td><td>55.40</td><td>92.4</td></tr></table>

$$
q _ {a, p} ^ {(k)} \leftarrow - \infty , \quad p \in \mathcal {Q} _ {a - 1} ^ {(k)}. \tag {12}
$$

The remaining high-scoring positions are then selected as anchor tokens according to the allocated context-token budget.

After anchor selection, non-anchor tokens are assigned to their nearest anchor centers and aggregated into the corresponding anchor tokens through weighted merging. The salient tokens preserved by ATS and the contextual tokens generated by DTM are then concatenated in temporal order to form the compressed video-token sequence.

# 4 Experiments

# 4.1 Experimental Settings

Benchmarks and Baselines. We evaluate InfoMerge with LMMS-Eval (Li, 2024; Zhang et al., 2025) on three widely used video understanding benchmarks: VideoMME (Fu et al., 2025), MLVU (Zhou et al., 2025), and LongVideoBench (Wu et al., 2024). LLaVA-OneVision-7B (Li et al., 2024a) is used as the main evaluation backbone, and the original uncompressed model serves as the upper-bound baseline. To evaluate cross-backbone transferability, we further apply InfoMerge to LLaVA-Video-7B (Zhang et al., 2024) and report results on VideoMME and MVBench. We compare InfoMerge with representative training-free visual token compression methods, including FastV (Chen et al., 2024), VisionZip (Yang et al., 2025), and FastVID (Shen et al., 2026), MMG-Vid (Ma et al., 2026).

Implementation Details. We follow the default setting and sample 32 frames per video for LLaVA-OneVision-7B (Li et al., 2024a). For LLaVA-Video-7B (Zhang et al., 2024), we sample 64 frames by default. All experiments are conducted on one NVIDIA V100 GPU. We evaluate different compression strengths by setting the visual token retention ratios to 5%, 10%, and 15%. Unless otherwise specified, all ablation studies are conducted on LLaVA-OneVision-7B using the same frame sampling strategy, inference parameters, and evaluation protocol. Detailed hyperparameter settings are provided in Appendix C.

# 4.2 Comparisons

Performance on LLaVA-OneVision-7B. Table 1 compares InfoMerge with representative training-free token compression methods on LLaVA-OneVision-7B. Under the same retention ratio and FLOPs, InfoMerge improves over FastVID at 15% and 5% retention ratios, and achieves comparable overall performance at 10% retention. At the 15% retention ratio, InfoMerge retains 98.8% of the uncompressed model performance with only 12.9% FLOPs. Compared with the recent MMG-Vid (Ma et al., 2026), InfoMerge improves the retained performance from 97.5% to 98.8%. On VideoMME, InfoMerge also achieves the highest overall score of 58.2, outperforming both FastVID (Shen et al., 2026) and MMG-Vid (Ma et al., 2026). At the 10% retention ratio, InfoMerge shows clear gains on VideoMME. It improves the VideoMME overall score from 56.8 to 57.7 over FastVID (Shen et al., 2026), yielding a

Table 2: Comparison on LLaVA-Video-7B-Qwen2. We report results on MVBench and VideoMME. FLOPs are measured under the default 64-frame setting. “%” denotes the percentage of the Vanilla average score retained. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Retention Ratio R</td><td rowspan="2">FLOPs (T)↓</td><td rowspan="2">FLOPs Ratio</td><td rowspan="2">MVBench</td><td colspan="4">VideoMME</td><td rowspan="2">Avg. Score</td><td rowspan="2">%</td></tr><tr><td>Overall</td><td>Short</td><td>Medium</td><td>Long</td></tr><tr><td>LLaVA-Video-7B</td><td>100%</td><td>94.1</td><td>100.0%</td><td>60.4</td><td>64.1</td><td>77.0</td><td>62.3</td><td>53.1</td><td>62.3</td><td>100.0</td></tr><tr><td>FastV (Chen et al., 2024)</td><td>15%</td><td>16.9</td><td>18.0%</td><td>50.8</td><td>54.0</td><td>60.9</td><td>54.4</td><td>46.7</td><td>52.4</td><td>84.2</td></tr><tr><td>VisionZip (Yang et al., 2025)</td><td>15%</td><td>11.0</td><td>11.7%</td><td>55.2</td><td>60.3</td><td>70.3</td><td>58.8</td><td>51.7</td><td>57.8</td><td>92.8</td></tr><tr><td>MMG-Vid (Ma et al., 2026)</td><td>15%</td><td>11.0</td><td>11.7%</td><td>56.1</td><td>61.1</td><td>72.3</td><td>60.1</td><td>50.8</td><td>58.6</td><td>94.1</td></tr><tr><td>FastVID (Shen et al., 2026)</td><td>15%</td><td>11.0</td><td>11.7%</td><td>60.5</td><td>62.1</td><td>73.8</td><td>61.0</td><td>51.7</td><td>61.3</td><td>98.5</td></tr><tr><td>Ours</td><td>15%</td><td>11.0</td><td>11.7%</td><td>59.9</td><td>62.8</td><td>73.4</td><td>63.0</td><td>51.9</td><td>61.4</td><td>98.6</td></tr><tr><td>FastV (Chen et al., 2024)</td><td>10%</td><td>13.1</td><td>13.9%</td><td>43.2</td><td>49.6</td><td>54.0</td><td>50.3</td><td>44.6</td><td>46.4</td><td>74.5</td></tr><tr><td>VisionZip (Yang et al., 2025)</td><td>10%</td><td>6.9</td><td>7.3%</td><td>53.8</td><td>58.7</td><td>67.4</td><td>57.7</td><td>51.1</td><td>56.3</td><td>90.4</td></tr><tr><td>MMG-Vid (Ma et al., 2026)</td><td>10%</td><td>6.9</td><td>7.3%</td><td>54.9</td><td>59.4</td><td>71.0</td><td>57.9</td><td>49.2</td><td>57.2</td><td>91.8</td></tr><tr><td>FastVID (Shen et al., 2026)</td><td>10%</td><td>6.9</td><td>7.3%</td><td>58.5</td><td>60.2</td><td>72.0</td><td>58.1</td><td>50.4</td><td>59.4</td><td>95.3</td></tr><tr><td>Ours</td><td>10%</td><td>6.9</td><td>7.3%</td><td>60.0</td><td>60.9</td><td>72.6</td><td>59.1</td><td>51.0</td><td>60.5</td><td>97.1</td></tr></table>

Table 3: LLM-side latency comparison. Prefill time measures LLM decoder latency until the first generated token. Generate time measures LLM decoder latency for the full response, including prefill and subsequent decoding. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Time</td><td rowspan="2">Acc. (%)</td></tr><tr><td>Prefill (ms)</td><td>Generate (ms)</td></tr><tr><td>LLaVA-OV-7B</td><td>1505.7 (1.00×)</td><td>1538.5 (1.00×)</td><td>100.0</td></tr><tr><td>Ours r=15%</td><td>355.4 (4.24×)</td><td>389.2 (3.95×)</td><td>98.8</td></tr><tr><td>Ours r=10%</td><td>225.6 (6.67×)</td><td>257.6 (5.97×)</td><td>97.1</td></tr><tr><td>Ours r=5%</td><td>121.1 (12.43×)</td><td>150.3 (10.23×)</td><td>92.4</td></tr></table>

0.9-point improvement under the same FLOPs. InfoMerge improves the average score over FastV by 3.20 points. At the more aggressive 5% retention ratio, InfoMerge still maintains strong performance. Specifically, with 95% of visual tokens compressed, InfoMerge retains 92.4% of the original average performance, whereas the best-performing baseline FastVID retains 91.0%. Compared with the baseline methods, InfoMerge achieves consistently competitive overall performance across different retention ratios, with more pronounced advantages under extremely low token retention ratios.

Cross-backbone Transfer. Table 2 reports results on LLaVA-Video-7B. InfoMerge retains 97.1% of the original average performance with only 10% of visual tokens preserved, outperforming FastVID, which retains 95.3% under the same setting. This result suggests that InfoMerge can transfer to another Video-LLM backbone.

Efficiency Comparison. Beyond performance, Table 3 reports the LLM-side latency after visual features are obtained. InfoMerge has substantial acceleration in both prefill and generation. Un-

Table 4: Ablation study on different components. 

<table><tr><td colspan="3">Components</td><td colspan="3">VideoMME Acc.</td></tr><tr><td>TFD</td><td>Unique</td><td>Richness</td><td>r=5%</td><td>r=10%</td><td>r=15%</td></tr><tr><td></td><td></td><td></td><td>53.6</td><td>56.8</td><td>57.8</td></tr><tr><td>✓</td><td></td><td></td><td>54.2</td><td>57.5</td><td>58.1</td></tr><tr><td>✓</td><td>✓</td><td></td><td>54.5</td><td>57.6</td><td>58.1</td></tr><tr><td>✓</td><td></td><td>✓</td><td>54.3</td><td>56.9</td><td>58.2</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>54.6</td><td>57.7</td><td>58.2</td></tr></table>

Table 5: Ablation of temporal redundancy estimation on VideoMME. “Adjacent Diff.” follows DyCoke and computes first-order local feature differences within a temporal window of size 4, while TFD uses our secondorder temporal fingerprint difference. 

<table><tr><td>Method</td><td>r=5%</td><td>r=10%</td><td>r=15%</td></tr><tr><td>FastVID</td><td>53.6</td><td>56.8</td><td>57.8</td></tr><tr><td>+ Adjacent Diff. + CABA</td><td>54.4</td><td>57.5</td><td>58.2</td></tr><tr><td>+ TFD + CABA</td><td>54.6</td><td>57.7</td><td>58.2</td></tr></table>

der 15% retention, InfoMerge achieves substantial acceleration, with a 4.24× speedup in prefill and a 3.95× speedup in generation while maintaining 98.8% of the model performance on LLaVA-OneVision (Li et al., 2024a). This confirms that InfoMerge effectively improves inference efficiency under high compression rates.

# 4.3 Ablation Study

Effect of Temporal Fingerprint Difference. We first evaluate the effectiveness of TFD for temporal redundancy estimation. As shown in Table 4, it confirms the effectiveness of TFD as the basic redundancy prior: using TFD alone already improves over the original FastVID across all retention ratios. As shown in Table 5, we further compare TFD with a first-order local difference baseline following Dy-

Table 6: Ablation on the two-factor dynamic budget allocation on VideoMME. α and β denote the weights of segment uniqueness and representational richness, respectively. Retain Ratio= 10% 

<table><tr><td>α</td><td>β</td><td>VideoMME Acc.</td></tr><tr><td>1.00</td><td>0.00</td><td>57.6</td></tr><tr><td>0.90</td><td>0.10</td><td>57.7</td></tr><tr><td>0.80</td><td>0.20</td><td>57.4</td></tr><tr><td>0.70</td><td>0.30</td><td>57.2</td></tr><tr><td>0.60</td><td>0.40</td><td>57.0</td></tr><tr><td>0.50</td><td>0.50</td><td>57.1</td></tr><tr><td>0.30</td><td>0.70</td><td>57.4</td></tr><tr><td>0.00</td><td>1.00</td><td>56.9</td></tr></table>

Coke (Tao et al., 2025), which computes feature differences within a temporal window of size 4. Compared with this first-order window-based variant, TFD achieves better performance at stricter retention ratios, indicating that second-order temporal fingerprints are more effective for identifying temporally stable redundant regions when the token budget is limited.

# Effect of Content-Aware Budget Allocation.

We then study the contribution of the proposed dynamic budget allocation strategy. We first explore the fusion weights between segment uniqueness and representational richness to determine the best allocation configuration. Specifically, we fix the overall visual token retention ratio to 10% and vary the fusion weights of the two factors. As shown in Table 6, segment uniqueness is the stronger single factor, while representational richness provides complementary structural information. Their combination achieves the best performance, suggesting that segment-level budget allocation benefits from jointly considering segment uniqueness and representational richness. We further examine the contribution of each factor in Table 4. Adding segment uniqueness to TFD improves the performance at 5% and 10% retention ratios, showing that segment-level uniqueness is useful for budget reallocation. Adding representational richness provides complementary gains, especially at 15% retention, suggesting that representational richness helps capture the internal visual richness of each segment. The full model achieves the best results at 5% and 10% and remains competitive at 15%, demonstrating the overall effectiveness of the twofactor dynamic budget allocation.

Sensitivity to Temperature. We further conduct an ablation study on the temperature parameter τ in dynamic budget allocation. The temperature controls the sensitivity of the sigmoid modulation function to the segment score. A smaller value makes the budget distribution smoother, while a larger value concentrates the token budget on a few high-scoring segments. As shown in Figure 5, a moderate temperature achieves the best performance, indicating that overly smooth or overly aggressive redistribution can both be suboptimal.

![](images/b494aaf1b450a6c272b859b48b74bb0eb927329ac52d400097c8348d78399b16.jpg)

<details>
<summary>line</summary>

| Temperature τ | VideoMME Acc (%) |
| ------------- | ---------------- |
| 0.5           | 57.48            |
| 0.8           | 57.56            |
| 1.2           | 57.67            |
| 1.5           | 57.59            |
| 2.0           | 57.22            |
| 5.0           | 56.82            |
| 10.0          | 56.56            |
</details>

Figure 5: Ablation on the temperature parameter τ in dynamic budget allocation. Moderate temperature leads to the best trade-off between smooth and aggressive budget redistribution. Retain Ratio = 10%.

![](images/7d35c2d0ceafe111e339c5a6e52f7ae8649b8afd2e5d981e2c06a8376b614466.jpg)

<details>
<summary>line</summary>

| TFD penalty λ | VideoMME Acc (%) |
| ------------- | ---------------- |
| 1             | 57.3             |
| 2             | 57.1             |
| 5             | 57.3             |
| 8             | 57.4             |
| 10            | 57.7             |
| 12            | 57.7             |
| 15            | 57.7             |
| 10*           | 57.7             |
</details>

Figure 6: Ablation on the redundancy suppression strength λ on VideoMME. Stronger suppression reduces repeated preservation of static tokens and improves token utilization. Retain Ratio = 10%.

# Effect of Redundancy Suppression Strength.

Figure 6 analyzes the effect of the redundancy suppression strength λ. A larger λ more strongly penalizes previously selected static positions. The results show that strong suppression works well, suggesting that repeatedly selected static regions contribute limited additional information and should be explicitly discouraged during token selection.

# 5 Conclusion

In this paper, we propose InfoMerge, a novel training-free token compression method for efficient Video LLM inference. Our method is built upon two key components: (1) a Temporal Fingerprint Difference (TFD) module, which performs segment-level second-order redundancy identification; and (2) a Content-Aware Budget Allocation strategy, which adaptively distributes token budgets according to both segment uniqueness and representational richness across video segments. Extensive experiments across multiple Video-LLM architectures and benchmarks demonstrate the effectiveness and strong generalization ability of our method. Notably, InfoMerge exhibits more pronounced advantages under aggressive compression, achieving favorable efficiency–performance tradeoffs.

# 6 Limitations

Although InfoMerge achieves a favorable efficiency–performance trade-off for Video LLM inference, several limitations remain. The proposed Temporal Fingerprint Difference (TFD) module relies on segment-level temporal statistics. For regions with persistent jitter, the estimated temporal fingerprints may become less reliable, which can reduce the accuracy of identifying low-information regions such as static backgrounds. Besides, although InfoMerge generalizes well across several Video LLM architectures, our experiments mainly focus on representative open-source models and benchmarks. Its effectiveness on larger-scale multimodal systems or highly domain-specific video distributions remains to be further explored. While the proposed method significantly reduces inference FLOPs and memory usage, the additional computation introduced by redundancy estimation and spectral analysis may still incur non-negligible overhead under extremely resource-constrained environments. Finally, during the preparation of this manuscript, we leveraged AI assistants (Gemini and ChatGPT) for text polishing and grammatical corrections.

# References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.   
Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. 2022. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461.   
Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. 2024. An

image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer.   
Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, and 1 others. 2024. Videollama 2: Advancing spatial-temporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476.   
Junhao Du, Jialong Xue, Anqi Li, Jincheng Dai, and Guo Lu. 2026. Unified spatiotemporal token compression for video-llms at ultra-low retention. arXiv preprint arXiv:2603.21957.   
Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, and 1 others. 2025. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 24108–24118.   
Xiaohu Huang, Hao Zhou, and Kai Han. 2025. Prunevid: Visual token pruning for efficient video large language models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 19959–19973.   
Bo Li. 2024. Xinrun du, yuhao dong, haotian liu, yuanhan zhang, ge zhang, chunyuan li, and ziwei liu. Lmms-eval: Accelerating the development of large multimoal models, 6.   
Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and 1 others. 2024a. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326.   
Feng Li, Renrui Zhang, Hao Zhang, Yuanhan Zhang, Bo Li, Wei Li, Zejun Ma, and Chunyuan Li. 2024b. Llava-next-interleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895.   
KunChang Li, Yinan He, Yi Wang, Yizhuo Li, Wenhai Wang, Ping Luo, Yali Wang, Limin Wang, and Yu Qiao. 2025. Videochat: Chat-centric video understanding. Science China Information Sciences, 68(10):200102.   
Ji Lin, Hongxu Yin, Wei Ping, Pavlo Molchanov, Mohammad Shoeybi, and Song Han. 2024. Vila: On pre-training for visual language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26689–26699.   
Zhijian Liu, Ligeng Zhu, Baifeng Shi, Zhuoyang Zhang, Yuming Lou, Shang Yang, Haocheng Xi, Shiyi Cao, Yuxian Gu, Dacheng Li, and 1 others. 2025. Nvila: Efficient frontier visual language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 4122–4134.

Junpeng Ma, Qizhe Zhang, Ming Lu, Zhibin Wang, Qiang Zhou, Jun Song, and Shanghang Zhang. 2026. Mmg-vid: Maximizing marginal gains at segmentlevel and token-level for efficient video llms. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 24253–24261.   
Soham Sarkar and Anil K Ghosh. 2019. On perfect clustering of high dimension, low sample size data. IEEE transactions on pattern analysis and machine intelligence, 42(9):2257–2272.   
Kele Shao, Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. 2025. Holitom: Holistic token merging for fast video large language models. arXiv preprint arXiv:2505.21334.   
Leqi Shen, Guoqiang Gong, Tao He, Yifeng Zhang, Pengzhang Liu, Sicheng Zhao, and 1 others. 2026. Fastvid: Dynamic density pruning for fast video large language models. Advances in Neural Information Processing Systems, 38:123553–123581.   
Leqi Shen, Tianxiang Hao, Tao He, Sicheng Zhao, Yifeng Zhang, Yongjun Bao, Guiguang Ding, and 1 others. 2025. Tempme: Video temporal token merging for efficient text-video retrieval. In International Conference on Learning Representations, volume 2025, pages 60839–60860.   
Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. 2025. Dycoke: Dynamic compression of tokens for fast video large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18992–19001.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.   
Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, and 1 others. 2024. Qwen2- vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191.   
Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, and 1 others. 2025. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.   
Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. 2024. Longvideobench: A benchmark for longcontext interleaved video-language understanding. Advances in Neural Information Processing Systems, 37:28828–28857.   
Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. 2025. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the IEEE/CVF

Conference on Computer Vision and Pattern Recognition, pages 19792–19802.   
Evelyn Zhang, Fufu Yu, Aoqi Wu, Zichen Wen, Ke Yan, Shouhong Ding, Biqing Qi, and Linfeng Zhang. 2026. D2pruner: Debiased importance and structural diversity for mllm token pruning. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 12412–12420.   
Kaichen Zhang, Bo Li, Peiyuan Zhang, Fanyi Pu, Joshua Adrian Cahyono, Kairui Hu, Shuai Liu, Yuanhan Zhang, Jingkang Yang, Chunyuan Li, and 1 others. 2025. Lmms-eval: Reality check on the evaluation of large multimodal models. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 881–916.   
Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. 2024. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713.   
Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, and 1 others. 2025. Mlvu: Benchmarking multi-task long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 13691– 13701.

# A More Implementation Details

# A.1 Additional Experimental Settings

For latency measurement, we use batch size 1 and report the average latency after warm-up runs with CUDA synchronization. Latency is measured after visual features are extracted, excluding video decoding, frame sampling, vision-tower encoding, and projector computation. Therefore, the reported speedups reflect LLM-side acceleration rather than full end-to-end video-QA latency.

# A.2 FLOPs Calculation

We report MAC-style FLOPs, where one multiplyaccumulate operation is counted as one operation. Therefore, all FLOPs numbers are not multiplied by the conventional 2× MAC-to-FLOP conversion factor. We count only the prefill computation of the Qwen2 language backbone and exclude the vision encoder, projector, embedding lookup, RoPE, RMSNorm, softmax, and the LM head.

For a Qwen2 decoder layer with sequence length N, hidden size d, intermediate size $d _ { \mathrm { f } }$ , number of attention heads H, and number of key-value heads $H _ { \mathrm { k v } } .$ , the MAC-style FLOPs are computed as:

$$
\begin{array}{l} \mathrm{FLOPs} _ {\text {layer}} (N) = N d ^ {2} \left(2 + 2 \frac {H _ {\mathrm{kv}}}{H}\right) \tag {13} \\ + 2 N ^ {2} d + 3 N d d _ {\mathrm{ff}}. \\ \end{array}
$$

The first term counts the Q/K/V/O projections under grouped-query attention, the second term counts the two attention matrix multiplications, i.e., $Q K ^ { \top }$ and $\mathrm { A t t n } V$ , and the last term counts the SwiGLU MLP projections.

For an L-layer Qwen2 backbone, the total prefill FLOPs are:

$$
\mathrm{FLOPs} (N) = L \cdot \mathrm{FLOPs} _ {\text { layer }} (N). \tag {14}
$$

For Qwen2-7B, we use: $L = 2 8 , d =$ 3584, $d _ { \mathrm { f f } } = 1 8 9 4 4 , \quad H = 2 8 , \quad H _ { \mathrm { k v } } = 4$ . For outer-LLM token compression methods such as FastVID and InfoMerge, visual tokens are compressed before being fed into the Qwen2 backbone. Therefore, all decoder layers use the reduced sequence length $N _ { r }$ r:

$$
\mathrm{FLOPs} _ {\text { outer }} = L \cdot \mathrm{FLOPs} _ {\text { layer }} (N _ {r}). \tag {15}
$$

In contrast, for FastV, token pruning is applied after the first K full-token decoder layers. Following FastV (Chen et al., 2024), we set $K = 2$ and compute:

$$
\mathrm{FLOPs} _ {\text {FastV}} = K \cdot \mathrm{FLOPs} _ {\text {layer}} (N _ {0}) + (L - K) \cdot \mathrm{FLOPs} _ {\text {layer}} (N _ {r}), \tag {16}
$$

where $N _ { 0 }$ is the original sequence length and $N _ { r }$ is the sequence length after pruning. In our visualtoken-only FLOPs analysis, the original sequence length is computed as:

$$
N _ {0} = F \times P, \tag {17}
$$

where F denotes the number of sampled frames and P denotes the number of visual tokens per frame. The reduced sequence length $N _ { r }$ is determined by the corresponding visual-token retention ratio. Unless otherwise stated, text tokens are excluded from the FLOPs calculation.

# A.3 Source of Baseline Results

Unless otherwise stated, all results reported in our tables are evaluated locally under the same evaluation protocol. For Table 2, the accuracy results of FastV, VisionZip, and MMG-Vid on LLaVA-Video-7B are taken from MMG-Vid (Ma et al., 2026). The FastVID and InfoMerge results in Table 2 are evaluated locally under the same 64-frame setting. For all other tables, we locally evaluate all compared methods except MMG-Vid, whose results are taken from its original paper, since MMG-Vid is not publicly available at the time of our experiments.

# A.4 Details of Token Budget Decomposition

Given an overall visual-token retention ratio r, the total retained token budget for a video with $L$ sampled frames and N visual tokens per frame is

$$
B _ {\text { total }} = r L N. \tag {18}
$$

Following FastVID (Shen et al., 2026), we use a hyperparameter $d \in [ 0 , 1 ]$ to split the retained tokens into two branches:

$$
B _ {\mathrm{DTM}} = d B _ {\text {total}}, \quad B _ {\mathrm{ATS}} = (1 - d) B _ {\text {total}}. \tag {19}
$$

Here, $B _ { \mathrm { A T S } }$ is used for preserving fine-grained salient tokens selected by ATS, while $B _ { \mathrm { D T M } }$ is used for maintaining segment-level contextual information through DTM.

For ATS, the salient-token budget is uniformly assigned to each frame:

$$
T _ {s} = \left\lfloor \frac {B _ {\mathrm{ATS}}}{L} \right\rfloor = \left\lfloor (1 - d) r N \right\rfloor , \tag {20}
$$

where $T _ { s }$ denotes the number of salient tokens selected by ATS in each frame.

For DTM, the context-token budget is distributed across temporal segments according to the CABA described in Section 3.3. Given the segment allocation weight $w _ { k } .$ , the context-token budget for the k-th segment is computed as

$$
B _ {k} = \max \left(1, \text { round } \left(\frac {B _ {\mathrm{DTM}} w _ {k}}{\sum_ {j = 1} ^ {K} w _ {j}}\right)\right). \tag {21}
$$

In this way, the overall retention ratio r determines the total compression strength, while d controls the trade-off between ATS-based salient-token preservation and DTM-based context-token aggregation.

# A.5 Details of Dynamic Temporal Segmentation

We follow FastVID (Shen et al., 2026) and use Dynamic Temporal Segmentation (DySeg) to partition sampled video frames into temporally ordered segments. Given a video with L sampled frames, let ${ \bf g } _ { l }$ denote the frame-level global feature of the l-th frame. DySeg computes the transition similarity between adjacent frames as:

$$
t _ {l} = \cos (\mathbf {g} _ {l}, \mathbf {g} _ {l + 1}), \quad l = 1, \dots , L - 1. \tag {22}
$$

A smaller $t _ { s }$ indicates a stronger semantic transition between two adjacent frames and is therefore more likely to be selected as a segment boundary.

Let T = {tl}L−1 $\boldsymbol { \mathcal { T } } = \{ t _ { l } \} _ { l = 1 } ^ { L - 1 }$ denote all transition similarities. DySeg combines a minimum segment number constraint and a similarity threshold to determine the final boundary set:

$$
\mathcal {S} _ {1} = \operatorname{argmin} _ {c - 1} \mathcal {T}, \mathcal {S} _ {2} = \left\{l \mid t _ {l} <   \tau_ {\text {seg}} \right\}, \quad \mathcal {S} = \mathcal {S} _ {1} \cup \mathcal {O} \tag {23}
$$

Here, c denotes the minimum number of segments, and $\tau _ { \mathrm { s e g } }$ is the transition-similarity threshold. $S _ { 1 }$ ensures that each video is partitioned into at least c segments, while $S _ { 2 }$ further introduces boundaries at obvious semantic transitions. The resulting boundary set S divides the sampled frames into K temporally ordered segments $\{ V _ { k } \} _ { k = 1 } ^ { K }$ . All subsequent TFD-based redundancy estimation, information-density-aware budget allocation, and spatio-temporal token compression are performed independently within these segments.

# A.6 Details of Content-Aware Budget Allocation

Spectral entropy as representational richness. In CABA, we use normalized spectral entropy to estimate the internal representational richness of each segment. Given the flattened segment feature matrix $\mathbf { \bar { X } } _ { k } \in \mathbb { R } ^ { | \mathcal { V } _ { k } | N \times D }$ , let $\{ \sigma _ { j } \} _ { j = 1 } ^ { \bar { R } }$ denote its singular values, where $R = \mathrm { m i n } ( | \dot { \mathcal { V } } _ { k } | N , D )$ ). We define the normalized spectral energy distribution as

$$
\pi_ {j} = \frac {\sigma_ {j} ^ {2}}{\sum_ {\ell = 1} ^ {R} \sigma_ {\ell} ^ {2}}. \tag {24}
$$

We use squared singular values because they correspond to the spectral energy of the feature matrix. According to the Frobenius norm decomposition,

$$
\left\| \mathbf {X} _ {k} \right\| _ {F} ^ {2} = \sum_ {j = 1} ^ {R} \sigma_ {j} ^ {2}, \tag {25}
$$

where $\sigma _ { j } ^ { 2 }$ measures the amount of feature energy captured by the j-th singular direction. Therefore, $\{ \pi _ { j } \} _ { j = 1 } ^ { R }$ can be interpreted as an energy distribution over different principal directions.

Based on this distribution, we compute the normalized spectral entropy:

$$
e _ {k} = \frac {- \sum_ {j = 1} ^ {R} \pi_ {j} \log \pi_ {j}}{\log R}. \tag {26}
$$

The normalization term log R ensures that $e _ { k }$ lies in a comparable range across segments with different effective matrix sizes. A small $e _ { k }$ indicates that most energy is concentrated in a few dominant directions, suggesting a lower-dimensional and more redundant representation. In contrast, a large $e _ { k }$ indicates that the energy is distributed across more principal directions, suggesting richer and more complex visual semantics.

Relation to segment-level budget allocation. Segment uniqueness and representational richness capture complementary aspects of segment information. The uniqueness score $u _ { k }$ measures how much a segment deviates from the global video representation, while $e _ { k }$ measures the diversity of internal feature directions within the segment. Since the two factors have different numeric ranges, we apply z-score normalization before fusion:

$$
m _ {k} = \alpha z (u _ {k}) + \beta z (e _ {k}). \tag {27}
$$

The sigmoid function with temperature τ controls the sharpness of redistribution:

$$
w _ {k} = \sigma (\tau m _ {k}) \cdot | \mathcal {V} _ {k} |. \tag {28}
$$

Figure 7 provides a qualitative visualization of the proposed content-aware budget allocation. We observe that segments with lower information content, such as visually repetitive or less event-relevant clips, are assigned fewer context tokens. In contrast, segments containing key visual evidence or more diverse visual content receive larger token budgets. This suggests that InfoMerge redistributes the limited token budget toward more informative temporal segments.

# A.7 Details of Token Compression

Static-token replacement in ATS. Since many Video-LLMs employ SigLIP as the visual encoder, the corresponding [CLS] attention is unavailable during inference. Following FastVID (Shen et al., 2026), we re-attach the pretrained SigLIP head to obtain lightweight saliency estimation without modifying the original Video-LLM pipeline. When a selected ATS token position is identified as static by TFD, we reuse a segment-level merged representation for this spatial position to avoid repeatedly preserving near-duplicate static tokens. Specifically, for position $p \in { \mathcal { T } } _ { l } ^ { \mathrm { A T S } , ( k ) } \cap { S ^ { ( k ) } }$ ATS,(k)∩S(k), we compute a segment-level static representation by aggregating tokens at the same spatial position across the k-th segment:

$$
\mathbf {m} _ {p} ^ {(k)} = \frac {1}{| \mathcal {V} _ {k} |} \sum_ {r = 1} ^ {| \mathcal {V} _ {k} |} \mathbf {H} _ {r, p} ^ {(k)}. \tag {29}
$$

Then, for the current frame and all subsequent frames in the same segment, the token at position p is replaced by this merged representation:

$$
\mathbf {H} _ {r, p} ^ {(k)} \leftarrow \mathbf {m} _ {p} ^ {(k)}, \quad r = l, \dots , | \mathcal {V} _ {k} |. \tag {30}
$$

This operation reduces repeated preservation of visually similar static tokens while keeping a compact representation of the corresponding static region. This replacement is a feature-level substitution and does not introduce an additional tokenpruning step or change the number of retained tokens.

Density-peak anchor selection in DTM. Within each anchor frame, DTM follows density-peak clustering to select representative context anchors. For token $t _ { i } ,$ its local density is computed as

$$
\rho_ {i} = \exp \left(- \frac {1}{K _ {\mathrm{nn}}} \sum_ {t _ {j} \in \mathrm{kNN} (t _ {i})} d (t _ {i}, t _ {j}) ^ {2}\right), \tag {31}
$$

where $d ( \cdot , \cdot )$ denotes the token distance and $\mathrm { k N N } ( t _ { i } )$ denotes the $K _ { \mathrm { n n } }$ nearest neighbors of $t _ { i } .$ . The distance to higher-density tokens is defined as

$$
\delta_ {i} = \left\{ \begin{array}{l l} \min _ {j: \rho_ {j} > \rho_ {i}} d (t _ {i}, t _ {j}), & \exists j, \rho_ {j} > \rho_ {i}, \\ \max _ {j} d (t _ {i}, t _ {j}), & \text { otherwise. } \end{array} \right. \tag {32}
$$

The final density-peak score is

$$
s _ {i} = \rho_ {i} \delta_ {i}. \tag {33}
$$

Tokens with higher density-peak scores are selected as context anchors unless suppressed by the TFDguided hard mask described in Section 3.4.

Anchor-centric aggregation. After context anchors are selected, each non-anchor token is assigned to its nearest anchor token center. For anchor token a with assigned token set $\{ b _ { 1 } , \ldots , b _ { n } \}$ , the updated anchor representation is computed as

$$
a ^ {\star} = \mu a + (1 - \mu) \frac {1}{n} \sum_ {j = 1} ^ {n} b _ {j}, \tag {34}
$$

where $\mu$ balances the original anchor feature and the aggregated neighboring information. The updated anchors are used as contextual tokens and concatenated with the salient tokens selected by ATS.

# B Qualitative Analysis

To better understand the effect of content-aware budget allocation, we provide a qualitative case study in Figure 8. The question asks what caused the sudden fall of the woman, which requires the model to identify the causal event before the fall. FastVID allocates context-token budgets mainly according to segment length and may under-preserve short but evidence-critical segments. In contrast, InfoMerge assigns more context tokens to the prank segment according to its segment-level information content, thereby preserving key visual evidence for causal reasoning. As a result, InfoMerge correctly predicts “Scared by a prank”, while FastVID fails to determine the cause. This example demonstrates that InfoMerge can redistribute the limited token budget toward informative temporal segments instead of relying only on segment length.

# C Hyperparameter Settings

We use different hyperparameter settings for different retention ratios to balance salient-token preservation and contextual-token aggregation under varying compression strengths. The detailed settings are summarized in Table 7. Unless otherwise specified, all methods are evaluated under the same frame sampling strategy and evaluation protocol.

![](images/a9ec04419787dd2482488e0ccb2568399e4b9b699ead446f2315abdd06634d87.jpg)

<details>
<summary>text_image</summary>

input
VideMMME 001-1
VideMMME 027-1
VideMMME 001-1
VideMMME 027-1
VideMMME 001-1
VideMMME 027-1
VideMMME 001-1
VideMMME 027-1
VideMMME 001-1
VideMMME 027-1
VideMMME 001-1
</details>

Figure 7: Visualization of token selection in the proposed InfoMerge.Segments with lower information content receive fewer context tokens, while informative segments receive larger budgets.

Table 7: Hyperparameter settings under different visualtoken retention ratios. 

<table><tr><td>r</td><td>τ</td><td>ρ</td><td>λ</td><td>α</td><td>β</td></tr><tr><td>0.05</td><td>2.0</td><td>0.05</td><td>12.0</td><td>0.9</td><td>0.1</td></tr><tr><td>0.10</td><td>1.2</td><td>0.10</td><td>12.0</td><td>0.9</td><td>0.1</td></tr><tr><td>0.15</td><td>1.2</td><td>0.09</td><td>12.0</td><td>0.9</td><td>0.1</td></tr></table>

# VideoMME · Question: What caused the sudden fall of the woman?

![](images/86df8e7b8298f9a974ead8f21aa2ea9f3309460bc7dbdcb58ea949a6dbec3fba.jpg)

![](images/a9b9d80cfabeebac8d287eeccdbb2278100eda217b85975d60e24e8c535d406b.jpg)

![](images/0438a99d4c411694e86e47e8051af300509abbf429da9de169988e735ec9da2f.jpg)  
Bathroom setup

![](images/fc65e6483adf70a465b78964c56ad70d495523559a2f380f15216ad86511e8c7.jpg)

![](images/70d63504d8e6fb4c55395ffb8ff21c9748dcf0259ccfd2c3d67b39c523856f75.jpg)

![](images/a966c63130ced2b5077cc821e9b2116db2e7b9208bfec9dca4e9f6ee660b6b01.jpg)  
Prank appears

![](images/7d76371883f36125f47e375452eba0691bb728b75553e73518620d4e9fba8793.jpg)

![](images/4bb4a16c87b775c0d399e920346f7b737cb4a11a06425d3dd8ba1c1036989b5f.jpg)

![](images/29d17a22d07786cd50f941a1383837d0850ba60daf452e4edf0b99d9049e6b73.jpg)  
Woman falls

Context-token budget   
![](images/580cd7d90a67e8b9ed91aa88378628fefc3ba3c433ef55a7786e844e7d4b7edb.jpg)

<details>
<summary>bar</summary>

| Category         | Value |
| ---------------- | ----- |
| bathroom setup   | 50    |
| setup F0-6       | 25    |
| prank F7-18      | 90    |
| woman falls       | 90    |
</details>

FastVID: Cannot be determined (wrong)   
Ours: Scared by a prank (correct)

Segment uniqueness   
![](images/1bd1b816bafa9d2b0748c9f57bf54d2d981691ce5c67d24f9ceb954923609f25.jpg)

<details>
<summary>bar</summary>

| Category           | Value  |
| ------------------ | ------ |
| Bathroom setup seg1, F0-6 | 0.019  |
| Prank seg2, F7-18    | 0.136  |
| Woman falls eg3, F19-31 | 0.121  |
</details>

FastVID (length-based)   
Ours (CABA)   
Representational richness   
![](images/aac6e228915ec3088002c4be256ce98b82fe77538d724fa7ca9b580ca265256f.jpg)

<details>
<summary>bar</summary>

| Category           | Value |
| ------------------ | ----- |
| Bathroom setup seg1 | 5.21  |
| Prank seg2         | 6.65  |
| Woman falls eg3    | 3.77  |
</details>

Figure 8: Case study of evidence-aware budget allocation on VideoMME. The question asks what caused the sudden fall of the woman, requiring the model to identify the causal event before the fall. We show three temporal segments corresponding to the bathroom setup, the prank appearance, and the woman’s fall. Compared with FastVID’s length-based allocation, InfoMerge assigns more context-token budget to the evidence-critical prank segment, which also exhibits higher segment uniqueness and representational richness. This helps preserve the key visual evidence for causal reasoning. As a result, InfoMerge correctly predicts “Scared by a prank”, while FastVID fails to determine the cause.