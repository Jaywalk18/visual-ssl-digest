# Spectral Evolution-Guided Token Pruning in Multimodal Large Language Models

Bin Chen<sup>1,2</sup> Yuxiang Cai<sup>1,2</sup>\* Yadan Luo<sup>3</sup> Yi Zhang Jianwei Yin<sup>1,2</sup> Zhi Chen<sup>5</sup> \* <sup>1</sup> School of Software Technology, Zhejiang University, Ningbo, China <sup>2</sup> Zhejiang Key Laboratory of Digital-Intelligence Service Technology, China <sup>3</sup> The University of Queensland, St Lucia, QLD, Australia <sup>4</sup> Singapore Management University, Singapore <sup>5</sup> The University of Southern Queensland, Toowoomba, QLD, Australia caiyuxiang@zju.edu.cn, uqzhichen@gmail.com https://github.com/zjubinchen/CLSE

## Abstract

Reducing visual token redundancy is critical for accelerating Multimodal Large Language Models (MLLMs) without degrading cross-modal reasoning performance. Existing token pruning methods typically rely on single-layer signals, such as attention scores or token similarities, which overlook the cross-layer transformation of visual representations and may exhibit positional bias in multimodal token sequences. To address this limitation, we propose a training-free token pruning framework based on Cross-Layer Spectral Evolution (CLSE). Instead of measuring token importance from single-layer feature magnitudes, CLSE quantifies how token representations evolve across Transformer layers in the frequency domain. This evolution reflects the transition from high-frequency structural details to low-frequency semantic abstractions. We observe that tokens with stronger spectral redistribution across layers are more likely to be semantically active and should therefore be preserved. By modeling cross-layer token dynamics, CLSE provides a stable importance criterion that mitigates positional bias. Extensive experiments on both image and video benchmarks demonstrate that CLSE achieves a superior trade-off between efficiency and accuracy under aggressive token reduction. Across multiple MLLMs, CLSE reduces FLOPs, KV cache memory, and latency while maintaining competitive or improved performance.

## 1. Introduction

Multimodal Large Language Models (MLLM) [44, 72] have achieved remarkable progress in visual question answering, grounded dialogue, and multimodal reasoning. By coupling a vision encoder (e.g., ViT [20]) with a powerful LLM, these models enable strong cross-modal alignment and reasoning capabilities. However, this performance comes at a significant computational cost. Modern LLMs process hundreds to thousands of visual tokens per sample, leading to substantial inference latency, particularly in highresolution and video scenarios.

A key source of inefficiency lies in visual token redundancy. Spatial patches in images and even temporal patches in videos exhibit strong local and cross-frame correlations. Yet standard Transformer architectures process all tokens uniformly through deep stacks of self-attention layers. To mitigate this inefficiency, recent works propose dynamic token reduction strategies. For example, PriorTR [7], DynamicViT [50], A-ViT [76], and ATS [22] introduce adaptive token pruning mechanisms in ViTs. In the multimodal domain, FastV [6] prunes visual tokens after early layers of MLLMs, while SparseVLM [87] adopts progressive sparsification with token recycling. Complementary approaches such as ToMe [3] and PruMerge [51] aggregate similar tokens instead of discarding them, preserving complementary information under aggressive compression.

Despite the effectiveness, most existing methods rely on instantaneous signals computed within a single layer, such as attention weights, feature magnitudes, or similarity scores. However, attention is not always a reliable indicator of semantic contribution. As illustrated in Fig. 1, attention scores can exhibit strong positional bias, i.e., concentrating on visual tokens appearing later in the flattened sequence, even when those regions are not semantically critical [66]. Furthermore, static magnitude-based methods overlook a fundamental characteristic of LLM decoding behavior in an MLLM, where visual token representations undergo structured evolution across LLM layers, transitioning from low-level spatial details to high-level semantic abstractions while the text instruction is fused. This crosslayer transformation, rather than instantaneous attention intensity, more faithfully reflects a token’s true contribution to multimodal reasoning.

![](images/d0814e5e0d4a119a5bf7dda456c2c0bee42cf494aeb5c45a42ee63a09e35d8d3.jpg)  
Figure 1. Spectral evolution provides a more reliable token importance signal than attention. (a) Comparison between conventional text–vision attention and the proposed spectral evolution for estimating visual token importance across a Transformer block. (b) Layer wise visualization shows that the attention is biased toward the visual tokens appearing later in the flattened sequence due to positional encoding effects, whereas spectral evolution consistently highlights semantically evolving regions across layers.

Recent studies provide a complementary perspective by analyzing Transformers in the frequency domain. Works such as FNet [35] and Fourier-based vision architectures [49] demonstrate that spectral operations can effectively model global dependencies. More recently, frequency-aware compression strategies have been explored for multimodal models [59], highlighting the role of lowfrequency components in semantic representation. Empirical observations suggest that early Transformer layers preserve high-frequency spatial information, while deeper layers progressively emphasize low-frequency semantic structures. This depth-wise spectral shift reflects the abstraction process underlying multimodal reasoning.

Motivated by these observations, we propose a crosslayer spectral evolution method for visual token reduction in MLLMs, as shown in Fig. 1 (a). Our key insight is that token importance should be measured by how its representation evolves across layers. The tokens that contribute meaningfully to text-vision reasoning show stronger spectral evolution during the detail-to-semantics transformation, whereas redundant background tokens remain relatively stationary in the frequency domain. Specifically, we (1) project visual tokens into the frequency domain and apply a Gaussian high-pass filter to isolate global information; (2) quantify cross-layer spectral evolution as a tokenwise transformation intensity; and (3) integrate this evolution signal to obtain a robust importance estimate. Unlike purely attention-driven or magnitude-based pruning strategies, our approach captures representation dynamics across layers. Furthermore, our framework is orthogonal to token merging methods and can be seamlessly combined with merging strategies such as ToMe [3]. This complementarity is particularly beneficial in video settings, where temporal redundancy amplifies token correlations. Extensive experiments on image and video benchmarks demonstrate that CLSE consistently improves the trade-off between efficiency and accuracy across multiple MLLMs. In summary, our contributions are threefold:

• We introduce the cross-layer spectral evolution framework, dubbed CLSE, as a token importance estimation principle for MLLMs, which characterizes how visual representations evolve from structural details to semantic abstractions across LLM layers.

• We design a training-free spectral-guided token pruning framework that operationalizes CLSE to identify and remove redundant tokens during inference while preserving tokens undergoing meaningful representational evolution.

• We conduct extensive experiments on both image- and video-based multimodal benchmarks, showing that CLSE achieves a superior trade-off between efficiency and accuracy under aggressive token reduction. Across multiple MLLMs, our method consistently reduces FLOPs, KV cache memory, and latency while maintaining competitive or improved performance.

## 2. Related Work

Training-Free Token Reduction for MLLMs. Multimodal Large Language Models (MLLMs) [9, 42, 44, 72] incur substantial overhead due to long visual token sequences. In applications where efficiency matters [10– 19, 37, 40, 52, 57, 62, 63, 77, 78, 81, 86, 88–90], reducing the number of visual tokens during inference has emerged as an effective strategy for accelerating MLLMs without retraining. Existing training-free methods typically estimate token importance using heuristics derived from intermediate representations and can be broadly catego rized into attention-based, similarity-based, diversity-based, and hybrid approaches. Attention-based methods estimate token importance directly from cross-modal attention between text and visual tokens. FastV [6] is a representative plug-and-play approach that ranks visual tokens using attention scores at a selected transformer layer and prunes tokens with the lowest attention magnitude for subsequent layers. Several follow-up works extend this idea using different attention aggregation or dynamic selection strate gies [26, 28, 39, 45, 56, 58, 61, 68, 75, 84, 91]. These ap proaches are computationally efficient and easy to integrate with existing MLLMs, but they generally rely on absolute attention magnitude as the ranking signal. Similarity-based methods identify redundant visual tokens by measuring similarity in feature space and removing tokens that provide overlapping information. For example, DART [67] detects redundant visual tokens through similarity-based matching and selectively retains a compact subset of representative tokens. Related approaches also employ clustering or feature distance criteria [30, 33, 54, 73, 79]. Diversity-based methods aim to maintain a representative set of visual tokens by maximizing feature diversity among the selected tokens. For instance, DivPrune [1] encourages diversity in the retained tokens to avoid redundancy and improve coverage of different visual regions. Other approaches formulate token selection as an entropy or diversity maximization problem to preserve informative structures in the visual rep resentation [64, 83, 85]. While diversity criteria help pre vent redundancy, they may not explicitly capture task relevance. Hybrid methods combine multiple criteria to improve token selection robustness. VisPruner [21] integrates attention-based importance with diversity-aware selection, while ToDre [36] jointly considers diversity and attention signals when selecting tokens. Token merging approaches reduce sequence length by merging visually similar tokens rather than removing them. Representative methods include LightVLM [29], CrossGet [53], VisionZip [74], and Sparse VLM [87]. A concurrent work V²Drop [5] explores intrin sic cross-layer dynamics by progressively dropping ’lazy tokens’ with minimal representational variation. In contrast to existing methods that rely on instantaneous signals, we propose to measure cross-layer spectral evolution of tokens. By analyzing how token representations evolve across layers in the frequency domain, our approach cap tures structural-to-semantic transitions that are not visible from single-layer statistics, leading to more reliable identification of informative visual tokens.

Frequency Domain in Transformers. A growing body of work interprets Transformers through a frequency-domain lens. In language, FNet [35] replaces self-attention with Fourier mixing to accelerate computation, demonstrating that spectral operations can serve as efficient token mixers. VTC-LFC [65] compresses Vision Transformers by retaining only low-frequency spectral components of token representations. SFHformer [34] introduces frequencydomain token mixing via FFT to model global interactions with lower computational cost than standard self-attention, demonstrating the effectiveness of spectral operations in vision Transformers. In multimodal models, frequencydomain compression has also been explored; for instance, Fourier-VLM [59] compresses vision tokens by applying low-pass filtering in the frequency domain. Different from these approaches that apply spectral transforms for compression within a layer, we use cross-layer spectral evolution as an importance criterion, capturing whether a token undergoes meaningful detail-to-semantics transformation across layers.

## 3. Method

## 3.1. Preliminaries

Multimodal Large Language Models. An MLLM consists of a vision encoder $f _ { v } ,$ a LLM decoder $f _ { l } ,$ and a projection module $\phi ( \cdot )$ that maps visual embeddings into the LLM token space. Given an image or video frames I and text prompt $T ,$ , we obtain:

$$
\mathbf {X} ^ {0} = \phi (f _ {v} (I)) \in \mathbb {R} ^ {N \times d}, \qquad \mathbf {Z} ^ {0} = \operatorname{Embed} (T) \in \mathbb {R} ^ {M \times d},\tag{1}
$$

where $N$ and M are the numbers of visual and text tokens and d is the shared embedding dimension. We concatenate tokens into a joint sequence processed by L Transformer layers:

$$
\mathbf {H} ^ {0} = [ \mathbf {Z} ^ {0}; \mathbf {X} ^ {0} ] \in \mathbb {R} ^ {(M + N) \times d},\tag{2}
$$

$$
\mathbf {H} ^ {\ell} = \mathrm{Transformer} ^ {\ell} (\mathbf {H} ^ {\ell - 1}), \ell = 1, \ldots , L.\tag{3}
$$

For convenience, we denote the visual slice of layer ℓ as $\mathbf { X } ^ { \ell } \in \mathbb { R } ^ { N \times d }$

Token Redundancy and Reduction. Visual tokens typically exhibit strong spatial and semantic redundancy. Not all tokens contribute equally to cross-modal reasoning: background regions or visually repetitive patches may have limited influence on the final output. Token reduction aims to select a subset of $K \ll N$ informative tokens:

$$
\tilde {\mathbf {X}} = \mathcal {S} (\mathbf {X}), \quad \tilde {\mathbf {X}} \in \mathbb {R} ^ {K \times d},\tag{4}
$$

where $s$ is a selection operator that preserves tokens with high semantic contribution while discarding redundant ones. After selection, the reduced multimodal sequence becomes:

$$
\tilde {\mathbf {H}} = \left[ \mathbf {Z}; \tilde {\mathbf {X}} \right],\tag{5}
$$

![](images/8d3e4009bfd25c1a1769f2b4d569866547de5d4a6c572b4fdfaec6eac33e3f63.jpg)  
Figure 2. Overview of the proposed spectral evolution-guided token pruning framework. Left: Cross-layer spectral evolution is computed by transforming layer-wise token representations into the frequency domain, suppressing low-frequency components via a Gaussian highpass filter, and measuring the normalized spectral difference between adjacent layers. Tokens exhibiting stronger spectral redistribution are considered decision-critical and preserved. Right: The CLSE module is integrated into early LLM decoder layers to guide token selection. Compared with raw attention scores, CLSE provides a more reliable signal for identifying semantically evolving tokens.

which is then processed by subsequent Transformer layers. Fast Fourier Transform (FFT). Let $\textbf { X } \in \ \mathbb { R } ^ { H \times W }$ denote a 2D signal, where H and W are spatial height and width, and $X _ { h , w }$ is the value at coordinate (h, w) with $h \in \{ 0 , \ldots , H - 1 \}$ and $w \in \{ 0 , \ldots , W - 1 \}$ . The 2D Discrete Fourier Transform (DFT) maps X to a frequency representation $\hat { \mathbf { X } } \in \mathbb { C } ^ { H \times W }$

$$
\hat {X} _ {u, v} = \sum_ {h = 0} ^ {H - 1} \sum_ {w = 0} ^ {W - 1} X _ {h, w} e ^ {- j 2 \pi \left(\frac {u h}{H} + \frac {v w}{W}\right)},\tag{6}
$$

where $( u , v )$ are frequency indices and $j = \sqrt { - 1 }$ . We denote the forward and inverse transforms as $\hat { \mathbf { X } } = \mathcal { F } ( \mathbf { X } )$ and $\mathbf { X } = \mathcal { F } ^ { - 1 } ( \hat { \mathbf { X } } )$ , respectively. The DFT decomposes X into orthogonal complex exponentials of increasing spatial frequency: low-frequency coefficients encode global, slowly varying structures, while high-frequency coefficients capture fine-grained spatial variations.

## 3.2. Cross-Layer Spectral Evolution (CLSE)

In Fig. 2, we present an intuitive overview of our CLSE. Let $N = H$ · W and reshape the visual tokens $\mathbf { X } ^ { \ell } \in \mathbb { R } ^ { N \times d }$ into spatial feature maps $\mathbf { \bar { X } } _ { \mathrm { 2 D } } ^ { \ell } \in \mathbb { R } ^ { H \times W \times d }$ , we measure high-frequency structural energy instead of raw embedding magnitude. We first transform to the frequency domain and centre the spectrum:

$$
\widehat {\mathbf {X}} ^ {\ell} = \mathrm{fftshift} (\mathcal {F} (\mathbf {X} _ {\mathrm{2D}} ^ {\ell})).\tag{7}
$$

To suppress low-frequency global patterns and highlight local structure, we apply a Gaussian high-pass mask M ∈ $\mathbb { R } ^ { H \times W }$

$$
\mathbf {M} _ {u, v} = 1 - \exp \left(- \frac {(u - c _ {u}) ^ {2} + (v - c _ {v}) ^ {2}}{2 (r \cdot c _ {u}) ^ {2}}\right),\tag{8}
$$

where $\left( c _ { u } , c _ { v } \right)$ is the spectral centre and r is the cutoff ratio. We then invert the transform to obtain a spatial residual map and average over channels:

$$
\mathbf {R} _ {\ell} = \left| \mathcal {F} ^ {- 1} \Big (\mathrm{ifftshift} (\widehat {\mathbf {X}} ^ {\ell} \odot \mathbf {M}) \Big) \right|,\tag{9}
$$

$$
\mathbf {S} _ {\ell} = \mathrm{Avg} _ {d} (\mathbf {R} _ {\ell}) \in \mathbb {R} ^ {H \times W}.\tag{10}
$$

Flattening $\mathbf { S } _ { \ell }$ over spatial positions yields $\mathbf { S } _ { \ell } \in \mathbb { R } ^ { N }$ , a pertoken distribution of high-frequency energy. Crucially, $\mathbf { S } _ { \ell }$ is not the full feature norm. Instead, it isolates band-limited residual energy after low-frequency suppression, where an ablation study evidenced this in Fig. 5(c).

CLSE measures how a token’s band-limited structure changes across depth throughout the process of text-vision interaction in the transformer blocks. Background tokens tend to exhibit near-static residual patterns, while semantically engaged tokens often show pronounced redistribution. For a reference layer ℓ and a subsequent layer $\ell { + 1 }$ , we define the evolution intensity:

$$
\mathcal {I} ^ {(\ell)} = \mathrm{clip} \bigg (\frac {| \mathbf {S} _ {\ell + 1} - \mathbf {S} _ {\ell} |}{\mathbf {S} _ {\ell} + \overline {{\mathbf {S}}} _ {\ell} + \epsilon},   0,   \tau_ {\max} \bigg),\tag{11}
$$

$$
\overline {{\mathbf {S}}} _ {\ell} = \mathrm{mean} (\mathbf {S} _ {\ell} \text { over   tokens }),\tag{12}
$$

where $\boldsymbol { \mathcal { T } ^ { ( \ell ) } }$ is a relative metric to achieve scale-invariance. Dividing by $\mathbf { S } _ { \ell }$ prevents tokens with large embedding magnitudes from dominating the signal, while the global mean $\overline { { \mathbf { S } } } _ { \ell }$ acts as a regularizer to further suppress minor spectral fluctuations in inactive or redundant regions (where local energy is below the image-wide average). ϵ and $\tau _ { \mathrm { m a x } } \ \mathrm { e n - }$ sure numerical stability and outlier robustness.

To maximize prefilling speedups, we perform early-stage pruning in the LLM decoder. In our default setting, we compute CLSE from layer $0  1$ , which is between input embeddings $\mathbf { X } ^ { 0 }$ and the first decoder layer output $\mathbf { X } ^ { 1 }$ , and execute pruning before layer 2. For each sample, we select the top-K tokens based on the evolution intensities:

$$
\mathcal {K} = \operatorname{TopK} (\mathcal {I} ^ {(\ell)}, K), \quad \tilde {\mathbf {X}} ^ {\ell} = \mathbf {X} ^ {\ell} [ \mathcal {K} ] \in \mathbb {R} ^ {K \times d},\tag{13}
$$

and continue decoding with the truncated sequence $\tilde { \mathbf { H } } ^ { \ell } =$ $[ \mathbf { Z } ^ { \ell } ; \tilde { \mathbf { X } } ^ { \ell } ]$ for all subsequent layers. This yields a trainingfree, plug-and-play pruning module that significantly reduces computational overhead.

## 3.3. Theoretical Analysis

The above pruning rule admits a simple fidelity interpretation. Specifically, the calibrated spectral evolution score $\boldsymbol { \mathcal { T } ^ { ( \ell ) } }$ quantifies the amount of cross-layer band-limited structural redistribution associated with each token. For any retained token set Ω with $| \Omega | = K$ , let $P _ { \Omega } ( \mathbf { X } ^ { \ell } )$ denote the pruned token sequence. Then the pruning-induced perturbation of the subsequent decoder $g ( \cdot )$ admits the bound:

$$
\| g (\mathbf {X} ^ {\ell}) - g (P _ {\Omega} (\mathbf {X} ^ {\ell})) \| _ {2} \leq \alpha \sum_ {i \notin \Omega} \mathcal {I} _ {i} ^ {(\ell)},\tag{14}
$$

where the right-hand side corresponds to the discarded spectral evolution mass. Therefore, selecting the top-K tokens by CLSE minimizes this upper bound among all subsets of size K, i.e., it preserves the maximum amount of cross-layer spectral evolution under a fixed token budget. A derivation is provided in the appendix.

## 4. Experiments

To validate the effectiveness of our CLSE framework, we conduct experiments on image and video questionanswering datasets with different MLLM variants.

## 4.1. Image VQA Tasks

Experiment setup. We evaluate CLSE on nine diverse multimodal benchmarks spanning visual reasoning (GQA [31], SQA [48], VizWiz [25]), hallucination detection (POPE [38]), OCR (VQA<sub>TEXT</sub> [55], $\mathrm { O C R } _ { \mathrm { B e n c h } }$ [47]), and holistic understanding (MME [23], MMB [46], $\mathbf { M M B } _ { C N }$ [46], MMVet [80]). We compare against seven state-of-the-art training-free methods: ToMe [3], FastV [6], PDrop [69], SparseVLM [87], FiCoCo [27], HiRED [2], and PruMerge [51]. To assess cross-model generalizability, we evaluate on LLaVA-1.5-7B [44], LLaVA-1.5-13B [44], LLaVA-Next-7B [43], Qwen2-VL-7B [60], and Qwen2- VL-72B [60].

Implementation Details. We apply CLSE to each model’s standard inference pipeline without modifying model weights or requiring additional training. Unlike other methods that perform pruning in later layers, for example, FastV [6] performs at layer 3, CLSE performs token pruning at the first layer of the LLM decoder. All experiments are conducted on NVIDIA RTX PRO 6000 GPU.

Main Results. Table 1 reports results on LLaVA-1.5-7B under three token-retention budgets (192, 128, and 64 tokens), corresponding to pruning ratios of 66.7%, 77.8%, and 88.9%, respectively. CLSE applies spectral evolution scoring with simple hard pruning and no token recycling. Despite this simplicity, CLSE achieves the best overall average of 99.4%, 98.1%, and 94.8% across the three budgets, consistently outperforming both hard-pruning methods (FastV, PDrop) and token-merging methods (Sparse-VLM, PruMerge, FiCoCo-V). This demonstrates that spectral evolution provides a more reliable token-importance criterion than attention- or magnitude-based strategies, regardless of whether the baseline uses hard pruning or token merging.

Scaling to Larger Models. We further conduct experiments on the larger LLaVA-1.5-13B under $\begin{array} { r l } { K } & { { } = } \end{array}$ 192, 128, 64 tokens. Figure 3 shows that CLSE maintains the most balanced performance profile and achieves the strongest overall average across all budgets. The advantage becomes more evident as compression increases, where competing methods exhibit larger task-specific degradation. These results demonstrate that spectral evolution provides stable token importance estimation and scales effectively to larger models under aggressive pruning. Additionally, we provide an experimental comparison with baseline methods on an even larger model Qwen2-VL-72B in the appendix due to the page limit.

Generalization to Qwen2-VL-7B. Table 2 shows CLSE can generalize well to Qwen2-VL-7B under three token ratios. CLSE achieves 97.6%, 95.8% and 90.5% average accuracy relative to the vanilla model on three ratios. Under 66.7% pruning, CLSE achieves the highest average performance (97.6%), outperforming FastV and PDrop across

Table 1. Performance comparison with baselines on nine image understanding benchmarks with LLaVA-1.5-7B. The vanilla 576 tokens are pruned to 192, 128, and 64.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td> $MMB_{CN}$ </td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA_{Text}$ </td><td>VizWiz</td><td> $OCR_{Bench}$ </td><td>Avg.</td></tr><tr><td>LLaVA-1.5-7B</td><td colspan="10">Upper Bound, 576 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>61.9</td><td>64.7</td><td>58.1</td><td>1862</td><td>85.9</td><td>69.5</td><td>58.2</td><td>50.0</td><td>29.7</td><td>100.0%</td></tr><tr><td></td><td colspan="10">Retain 192 Tokens (↓66.7%)</td></tr><tr><td>ToMe (ICLR23)</td><td>54.3</td><td>60.5</td><td>53.2</td><td>1563</td><td>65.2</td><td>68.0</td><td>52.1</td><td>50.9</td><td>25.8</td><td>84.8%</td></tr><tr><td>FastV (ECCV24)</td><td>52.7</td><td>61.2</td><td>57.0</td><td>1612</td><td>67.3</td><td>67.1</td><td>52.5</td><td>50.8</td><td>29.1</td><td>92.1%</td></tr><tr><td>PruMerge (ICCV25)</td><td>54.3</td><td>59.6</td><td>52.9</td><td>1632</td><td>71.3</td><td>67.9</td><td>54.3</td><td>50.1</td><td>25.3</td><td>90.8%</td></tr><tr><td>PDrop (CVPR25)</td><td>57.1</td><td>63.2</td><td>56.8</td><td>1766</td><td>82.3</td><td>68.8</td><td>56.1</td><td>51.1</td><td>28.7</td><td>96.9%</td></tr><tr><td>HiRED (AAAI25)</td><td>58.7</td><td>62.8</td><td>54.7</td><td>1737</td><td>82.8</td><td>68.4</td><td>47.4</td><td>50.1</td><td>19.0</td><td>91.0%</td></tr><tr><td>SparseVLM (ICML25)</td><td>57.6</td><td>62.5</td><td>53.7</td><td>1721</td><td>83.6</td><td>69.1</td><td>56.1</td><td>50.5</td><td>29.2</td><td>96.3%</td></tr><tr><td>FiCoCo-V (AAAI26)</td><td>58.5</td><td>62.3</td><td>55.3</td><td>1732</td><td>82.5</td><td>67.8</td><td>55.7</td><td>51.0</td><td>28.6</td><td>96.2%</td></tr><tr><td>CLSE (Ours)</td><td>61.5</td><td>63.6</td><td>57.2</td><td>1817</td><td>85.1</td><td>70.1</td><td>57.5</td><td>50.6</td><td>30.1</td><td>99.4%</td></tr><tr><td></td><td colspan="10">Retain 128 Tokens (↓77.8%)</td></tr><tr><td>ToMe (ICLR23)</td><td>52.4</td><td>53.3</td><td>50.7</td><td>1343</td><td>62.8</td><td>59.6</td><td>49.1</td><td>51.6</td><td>25.0</td><td>84.1%</td></tr><tr><td>FastV (ECCV24)</td><td>49.6</td><td>56.1</td><td>56.4</td><td>1490</td><td>59.6</td><td>60.2</td><td>50.6</td><td>51.3</td><td>28.5</td><td>87.2%</td></tr><tr><td>PruMerge (ICCV25)</td><td>53.3</td><td>58.1</td><td>51.7</td><td>1554</td><td>67.2</td><td>67.1</td><td>54.3</td><td>50.3</td><td>24.8</td><td>88.9%</td></tr><tr><td>PDrop (CVPR25)</td><td>56.0</td><td>61.1</td><td>56.6</td><td>1644</td><td>82.3</td><td>68.3</td><td>55.1</td><td>51.0</td><td>28.7</td><td>95.3%</td></tr><tr><td>HiRED (AAAI25)</td><td>57.2</td><td>61.5</td><td>53.6</td><td>1710</td><td>79.8</td><td>68.1</td><td>46.1</td><td>51.3</td><td>19.1</td><td>89.8%</td></tr><tr><td>SparseVLM (ICML25)</td><td>56.0</td><td>60.0</td><td>51.1</td><td>1696</td><td>80.5</td><td>67.1</td><td>54.9</td><td>51.4</td><td>28.0</td><td>93.7%</td></tr><tr><td>FiCoCo-V (AAAI26)</td><td>57.6</td><td>61.1</td><td>54.3</td><td>1711</td><td>82.2</td><td>68.3</td><td>55.6</td><td>49.4</td><td>26.2</td><td>94.3%</td></tr><tr><td>CLSE (Ours)</td><td>60.9</td><td>63.0</td><td>55.8</td><td>1816</td><td>84.4</td><td>70.1</td><td>56.6</td><td>51.0</td><td>28.4</td><td>98.1%</td></tr><tr><td></td><td colspan="10">Retain 64 Tokens (↓88.9%)</td></tr><tr><td>ToMe (ICLR23)</td><td>48.6</td><td>43.7</td><td>47.1</td><td>1138</td><td>52.5</td><td>50.0</td><td>45.3</td><td>52.6</td><td>22.7</td><td>67.4%</td></tr><tr><td>FastV (ECCV24)</td><td>46.1</td><td>48.0</td><td>52.7</td><td>1256</td><td>48.0</td><td>51.1</td><td>47.8</td><td>50.8</td><td>24.5</td><td>78.0%</td></tr><tr><td>PruMerge (ICCV25)</td><td>51.9</td><td>55.3</td><td>49.1</td><td>1549</td><td>65.3</td><td>68.1</td><td>54.0</td><td>50.1</td><td>25.0</td><td>87.5%</td></tr><tr><td>PDrop (CVPR25)</td><td>41.9</td><td>33.3</td><td>50.5</td><td>1092</td><td>55.9</td><td>68.6</td><td>45.9</td><td>50.7</td><td>25.0</td><td>77.0%</td></tr><tr><td>HiRED (AAAI25)</td><td>54.6</td><td>60.2</td><td>51.4</td><td>1599</td><td>73.6</td><td>68.2</td><td>44.2</td><td>50.2</td><td>19.1</td><td>86.6%</td></tr><tr><td>SparseVLM (ICML25)</td><td>52.7</td><td>56.2</td><td>46.1</td><td>1505</td><td>75.1</td><td>62.2</td><td>51.8</td><td>50.1</td><td>18.0</td><td>84.3%</td></tr><tr><td>FiCoCo-V (AAAI26)</td><td>52.4</td><td>60.3</td><td>53.0</td><td>1591</td><td>76.0</td><td>68.1</td><td>53.6</td><td>49.8</td><td>22.6</td><td>89.8%</td></tr><tr><td>CLSE (Ours)</td><td>58.3</td><td>61.9</td><td>54.3</td><td>1809</td><td>78.3</td><td>69.6</td><td>55.5</td><td>50.9</td><td>25.1</td><td>94.8%</td></tr></table>

![](images/78e53fd05f71cbd690c1bb6657fd3615b908c479cc5aaf96f8bac8b772a7d9f3.jpg)  
(a) 64 Tokens

![](images/36bbc90115bfa21d36e6c41cd8714e34f073ed035b838d8211115a6b0ebdde90.jpg)  
(b) 128 Tokens

![](images/c7352463ba86e08684c9dfa9d183e8fe43b607b7320019ff397d9e0d873f110b.jpg)  
(c) 192 Tokens  
Figure 3. Radar plots comparing pruning methods on LLaVA-1.5-13B under different token budgets. All metrics, including the overall average (Avg), are normalized to the full-token model. Across all configurations, CLSE achieves the most balanced performance and maintains the highest overall accuracy under aggressive compression.

most benchmarks, with strong results on GQA (61.0), MMB (76.5), POPE (86.7), and SQA (84.2). As compression increases to 77.8%, CLSE maintains the same high average score (95.8%), demonstrating stable performance despite further token reduction. Even under extreme pruning (88.9%), CLSE continues to outperform competing methods with an average of 90.5%, showing clear advantages on reasoning-heavy tasks such as GQA, POPE, and $\mathrm { \Delta V Q A _ { T e x t } }$ . Across all settings, CLSE consistently delivers the best overall performance and exhibits strong robustness as token budgets shrink, validating that cross-layer spectral evolution generalizes beyond LLaVA and remains effective across different MLLMs architectures.

Table 2. Performance comparison with baseline methods under different token configurations on Qwen2-VL-7B.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td> $MMB_{CN}$ </td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA_{Text}$ </td><td>Avg.</td></tr><tr><td>Qwen2-VL-7B</td><td colspan="8">Upper Bound, All Tokens (100%)</td></tr><tr><td>Vanilla</td><td>62.2</td><td>79.1</td><td>77.8</td><td>2296</td><td>87.7</td><td>85.4</td><td>81.4</td><td>100%</td></tr><tr><td>Qwen2-VL-7B</td><td colspan="8">Token Reduction (↓ 66.7%)</td></tr><tr><td>+ FastV (ECCV24)</td><td>58.6</td><td>74.5</td><td>74.3</td><td>2083</td><td>85.6</td><td>83.1</td><td>76.1</td><td>94.9%</td></tr><tr><td>+ PDrop (CVPR25)</td><td>58.5</td><td>72.3</td><td>71.2</td><td>1991</td><td>86.1</td><td>82.8</td><td>76.4</td><td>93.1%</td></tr><tr><td>+ HiRED (AAAI25)</td><td>59.9</td><td>73.8</td><td>73.4</td><td>2158</td><td>86.6</td><td>80.7</td><td>71.0</td><td>94.1%</td></tr><tr><td>+ SparseVLM (ICML25)</td><td>59.2</td><td>77.1</td><td>76.0</td><td>2181</td><td>85.9</td><td>83.5</td><td>77.8</td><td>96.7%</td></tr><tr><td>+ CLSE (ours)</td><td>61.0</td><td>76.5</td><td>75.5</td><td>2254</td><td>86.7</td><td>84.2</td><td>78.3</td><td>97.6%</td></tr><tr><td>Qwen2-VL-7B</td><td colspan="8">Token Reduction (↓ 77.8%)</td></tr><tr><td>+ FastV (ECCV24)</td><td>56.2</td><td>70.1</td><td>72.4</td><td>1957</td><td>82.7</td><td>82.3</td><td>73.0</td><td>91.3%</td></tr><tr><td>+ PDrop (CVPR25)</td><td>56.4</td><td>67.5</td><td>67.2</td><td>1945</td><td>84.2</td><td>82.0</td><td>73.7</td><td>89.9%</td></tr><tr><td>+ HiRED (AAAI25)</td><td>58.3</td><td>70.5</td><td>70.7</td><td>2086</td><td>85.4</td><td>79.6</td><td>65.4</td><td>90.8%</td></tr><tr><td>+ SparseVLM (ICML25)</td><td>54.9</td><td>73.2</td><td>73.6</td><td>2047</td><td>81.1</td><td>83.2</td><td>72.2</td><td>91.9%</td></tr><tr><td>+ CLSE (ours)</td><td>59.8</td><td>74.1</td><td>74.2</td><td>2209</td><td>85.5</td><td>83.5</td><td>76.6</td><td>95.8%</td></tr><tr><td>Qwen2-VL-7B</td><td colspan="8">Token Reduction (↓ 88.9%)</td></tr><tr><td>+ FastV (ECCV24)</td><td>50.6</td><td>61.8</td><td>62.5</td><td>1760</td><td>72.2</td><td>80.5</td><td>68.6</td><td>82.2%</td></tr><tr><td>+ PDrop (CVPR25)</td><td>52.5</td><td>59.2</td><td>58.3</td><td>1756</td><td>74.1</td><td>80.2</td><td>67.7</td><td>81.5%</td></tr><tr><td>+ HiRED (AAAI25)</td><td>55.5</td><td>64.3</td><td>64.3</td><td>1834</td><td>82.8</td><td>75.7</td><td>56.5</td><td>83.6%</td></tr><tr><td>+ SparseVLM (ICML25)</td><td>47.5</td><td>59.5</td><td>61.1</td><td>1624</td><td>68.0</td><td>80.7</td><td>61.7</td><td>78.4%</td></tr><tr><td>+ CLSE (ours)</td><td>56.1</td><td>69.3</td><td>69.2</td><td>2008</td><td>81.2</td><td>82.6</td><td>73.4</td><td>90.5%</td></tr></table>

Generalization to LLaVA-Next-7B. To further evaluate the generalization ability of our method, we conduct comparative experiments on LLaVA-Next-7B. The results are summarized in Table 3. The vanilla model processes 2880 visual tokens and serves as the upper-bound performance reference. Under a strict token budget of 320 tokens (88.9% token reduction), we compare CLSE with several recent token reduction methods including FastV, HiRED, PruMerge, SparseVLM, and PDrop. Overall, CLSE achieves the best performance among all competing methods, reaching 94.7% of the vanilla performance, significantly outperforming other pruning approaches. In particular, CLSE preserves strong performance on reasoning-intensive benchmarks such as GQA (62.6) and MMB (67.1), and maintains competitive results across multimodal evaluation suites including MMB-CN, MME, POPE, and SQA. Compared with other pruning methods that suffer noticeable degradation under aggressive token reduction, CLSE consistently maintains higher accuracy across most benchmarks. Notably, CLSE achieves these improvements while operating under the same token budget as other methods, demonstrating that cross-layer spectral evolution provides a more reliable criterion for identifying informative visual tokens. This result indicates that modeling representation dynamics across layers, rather than relying on instantaneous signals, leads to more effective token selection in MLLMs.

## 4.2. Video Understanding Tasks

Experimental Settings. We apply CLSE to Video-LLaVA [41] and evaluate its performance on four representative video question-answering benchmarks: MSVD-QA [70], MSRVTT-QA [70], TGIF-QA [32], and ActivityNet-QA [82]. We compare against several advanced baselines: FastV [6], which performs hard pruning; PDrop [69], which adopts a pyramid-shaped progressive pruning schedule; SparseVLM [87] and FiCoCo-V [27], which execute token merging. We report Accuracy (%) and Score (0–5) using GPT-4o-mini and Gemini-2.5-Flash as assistant evaluators.

Main Results. Table 4 reports results when compressing video tokens from 2048 to 194 (over 90% reduction). CLSE applies per-frame 2D FFT to compute spectral evolution scores and performs hard pruning, as in FastV. Despite this simplicity, spectral evolution provides a substantially better importance signal: CLSE achieves 35.7% average accuracy, compared to 26.1% for FastV and 27.3% for PDrop, demonstrating that per-frame spatial frequency dynamics capture informative visual content more reliably than attentionbased selection. At such extreme compression ratios, hard pruning inevitably discards temporal content that is difficult to recover. We therefore introduce CLSE-M, which recalls the top-scored pruned tokens by their spectral evolution scores and compresses them into compact representatives via density-peak clustering. CLSE-M achieves 38.7% average accuracy (GPT), matching the vanilla model (38.7%), and outperforms SparseVLM—which adopts the same token recycling strategy but relies on attention-based scoring—by 0.6 points in accuracy (38.7% vs. 38.1%), confirming that spectral evolution provides a more reliable signal for identifying critical visual dynamics.

Table 3. Comparative experiments are performed on LLaVA-Next-7B.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MMB-CN</td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA^{Text}$ </td><td>VizWiz</td><td>OCRBench</td><td>Avg.</td></tr><tr><td>LLaVA-Next-7B</td><td colspan="10">Upper Bound, 2880 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>64.2</td><td>67.4</td><td>60.6</td><td>1851</td><td>86.5</td><td>70.1</td><td>64.9</td><td>57.6</td><td>51.7</td><td>100.0%</td></tr><tr><td>LLaVA-Next-7B</td><td colspan="10">Retain 320 Tokens (↓88.9%)</td></tr><tr><td>FastV (ECCV24)</td><td>55.9</td><td>61.6</td><td>51.9</td><td>1661</td><td>71.7</td><td>62.8</td><td>55.7</td><td>53.1</td><td>37.4</td><td>86.3%</td></tr><tr><td>HiRED (AAAI25)</td><td>59.3</td><td>64.2</td><td>55.9</td><td>1690</td><td>83.3</td><td>66.7</td><td>58.8</td><td>54.2</td><td>40.4</td><td>91.7%</td></tr><tr><td>PruMerge (ICCV2025)</td><td>53.6</td><td>61.3</td><td>55.3</td><td>1534</td><td>60.8</td><td>66.4</td><td>50.6</td><td>54.0</td><td>14.6</td><td>79.2%</td></tr><tr><td>SparseVLM (ICML25)</td><td>56.1</td><td>60.6</td><td>54.5</td><td>1533</td><td>82.4</td><td>66.1</td><td>58.4</td><td>52.0</td><td>27.0</td><td>85.8%</td></tr><tr><td>PDrop (CVPR25)</td><td>56.4</td><td>63.4</td><td>56.2</td><td>1663</td><td>77.6</td><td>67.5</td><td>54.4</td><td>54.1</td><td>25.9</td><td>86.5%</td></tr><tr><td>CLSE (Ours)</td><td>62.6</td><td>67.1</td><td>58.3</td><td>1794</td><td>84.4</td><td>67.9</td><td>58.9</td><td>56.1</td><td>41.4</td><td>94.7%</td></tr></table>

Table 4. Performance comparison on video understanding benchmarks with Video-LLaVA-7B. We consistently prune video tokens from 2048 to 194 (over 90% reduction). CLSE-M denotes CLSE with token merging, where pruned tokens are recalled and clustered into compact representatives based on their spectral evolution scores. GPT-4o-mini (GPT) and Gemini-2.5-Flash (Gem) are adopted as assistant evaluators.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Eval</td><td colspan="2">TGIF</td><td colspan="2">MSVD</td><td colspan="2">MSRVTT</td><td colspan="2">ActivityNet</td><td colspan="2">Avg.</td></tr><tr><td>Acc.</td><td>Score</td><td>Acc.</td><td>Score</td><td>Acc.</td><td>Score</td><td>Acc.</td><td>Score</td><td>Acc.</td><td>Score</td></tr><tr><td rowspan="2">Video-LLaVA-7B</td><td>GPT</td><td>11.3</td><td>1.19</td><td>58.9</td><td>3.31</td><td>42.7</td><td>2.69</td><td>42.0</td><td>2.20</td><td>38.7</td><td>2.35</td></tr><tr><td>Gem</td><td>14.2</td><td>0.77</td><td>57.3</td><td>2.93</td><td>47.5</td><td>2.32</td><td>42.5</td><td>2.12</td><td>40.4</td><td>2.04</td></tr><tr><td colspan="12">Hard Pruning</td></tr><tr><td rowspan="2">FastV</td><td>GPT</td><td>2.8</td><td>1.16</td><td>35.9</td><td>2.36</td><td>31.4</td><td>2.19</td><td>34.2</td><td>1.83</td><td>26.1</td><td>1.88</td></tr><tr><td>Gem</td><td>2.3</td><td>0.14</td><td>35.5</td><td>1.88</td><td>34.3</td><td>1.69</td><td>34.2</td><td>1.74</td><td>26.6</td><td>1.36</td></tr><tr><td rowspan="2">PDrop</td><td>GPT</td><td>5.0</td><td>0.96</td><td>42.2</td><td>2.59</td><td>27.9</td><td>1.95</td><td>34.3</td><td>1.78</td><td>27.3</td><td>1.82</td></tr><tr><td>Gem</td><td>6.1</td><td>0.37</td><td>52.2</td><td>2.54</td><td>34.6</td><td>1.65</td><td>39.4</td><td>1.93</td><td>33.0</td><td>1.62</td></tr><tr><td rowspan="2">FiCoCo-V</td><td>GPT</td><td>5.6</td><td>1.18</td><td>42.1</td><td>2.68</td><td>37.6</td><td>2.47</td><td>43.1</td><td>2.23</td><td>31.9</td><td>2.13</td></tr><tr><td>Gem</td><td>6.7</td><td>0.38</td><td>51.2</td><td>2.51</td><td>44.3</td><td>2.10</td><td>44.9</td><td>2.20</td><td>36.7</td><td>1.79</td></tr><tr><td rowspan="2">CLSE</td><td>GPT</td><td>9.6</td><td>1.06</td><td>54.1</td><td>3.11</td><td>39.3</td><td>2.56</td><td>39.9</td><td>2.03</td><td>35.7</td><td>2.19</td></tr><tr><td>Gem</td><td>12.9</td><td>0.69</td><td>56.3</td><td>2.88</td><td>47.7</td><td>2.27</td><td>39.9</td><td>1.99</td><td>39.2</td><td>1.95</td></tr><tr><td colspan="12">w/ Token Merging</td></tr><tr><td rowspan="2">SparseVLM</td><td>GPT</td><td>10.7</td><td>1.03</td><td>56.0</td><td>3.20</td><td>41.9</td><td>2.63</td><td>43.6</td><td>2.21</td><td>38.1</td><td>2.27</td></tr><tr><td>Gem</td><td>13.1</td><td>0.71</td><td>58.2</td><td>3.05</td><td>48.0</td><td>2.38</td><td>44.0</td><td>2.20</td><td>40.8</td><td>2.09</td></tr><tr><td rowspan="2">CLSE-M</td><td>GPT</td><td>11.2</td><td>1.05</td><td>57.0</td><td>3.25</td><td>42.2</td><td>2.61</td><td>44.6</td><td>2.25</td><td>38.7</td><td>2.29</td></tr><tr><td>Gem</td><td>14.7</td><td>0.79</td><td>60.2</td><td>3.07</td><td>49.5</td><td>2.35</td><td>45.0</td><td>2.24</td><td>42.3</td><td>2.11</td></tr></table>

## 4.3. Analysis

Efficiency Analysis. Table 5 compares inference efficiency on LLaVA-1.5-7B (192 tokens) and Video-LLaVA-7B (194 tokens). On images, CLSE achieves the lowest FLOPs (43.9%) and smallest KV cache (56.0 MB) with the highest performance (99.6%) at 1.50× speedup; FastV is faster (1.69×) but drops to 92.1%. On video, CLSE-M matches

![](images/1123491e3bd219e4180859038368d9abca47217c40373fa48acfc70b3db194eb.jpg)  
Figure 4. Latency Comparison to Attention Scoring.

SparseVLM in FLOPs (10.7%) and KV cache (121.1 MB) while achieving perfect performance retention (100% vs. 98.4%) at 2.73× speedup; FastV again leads in raw speed (3.32×) but suffers severe performance drop.

Efficiency Compared with Attention Scoring. In Fig. 4, we analyze the computational efficiency of our FFTbased scoring relative to the attention-extraction overhead ∆, benchmarked on a single simulated attention layer using LLaMA-3-8B’s configuration [24]. In modern inference engines, optimized kernels such as Scaled Dot-Product Attention (SDPA) achieve high throughput by avoiding the materialization of the full N ×N attention matrix. Consequently, any pruning strategy relying on dense attention weights necessitates falling back to a sub-optimal “Eager” implementation, which reintroduces a quadratic latency penalty $O ( N ^ { 2 } )$ . As illustrated, while the extraction overhead $\Delta$ (defined as $\mathrm { L a t e n c y } _ { \mathrm { E a g e r } } - \mathrm { L a t e n c y } _ { \mathrm { S D P A } } )$ escalates quadratically with sequence length N, our CLSE module maintains a near-linear O(N log N) complexity due to the efficiency of the Fast Fourier Transform. Even at a spatial resolution of $6 4 \times 6 4 ( N = 4 0 9 6 )$ , the latency of our spectral scoring remains significantly below the ∆ threshold. This gap demonstrates that spectral evolution also offers a more hardware-efficient pathway for token reduction.

Table 5. Efficiency–performance trade-off comparison on LLaVA-1.5-7B and Video-LLaVA-7B under comparable token budgets.

<table><tr><td>Methods</td><td>Tokens ↓</td><td>Prefilling Time ↓ (MS)</td><td>FLOPs ↓</td><td>KV Cache ↓ (MB)</td><td>Performance ↑</td><td>Throughput↑ (samples/s)</td><td>Speedup ↑ (Prefilling)</td></tr><tr><td>Vanilla LLaVA-1.5-7B</td><td>576</td><td>41.9</td><td>100%</td><td>313.0</td><td>100%</td><td>19.4</td><td>1.00×</td></tr><tr><td>+ FastV</td><td>192</td><td>24.8</td><td>46.9%</td><td>121.0</td><td>92.1%</td><td>30.3</td><td>1.69×</td></tr><tr><td>+ PDrop</td><td>192</td><td>31.9</td><td>74.0%</td><td>211.2</td><td>96.9%</td><td>24.6</td><td>1.31×</td></tr><tr><td>+ SparseVLM</td><td>192</td><td>33.0</td><td>49.7%</td><td>128.2</td><td>96.3%</td><td>23.7</td><td>1.27×</td></tr><tr><td>+ CLSE(ours)</td><td>192</td><td>27.8</td><td>43.9%</td><td>56.0</td><td>99.4%</td><td>27.9</td><td>1.50×</td></tr><tr><td>Vanilla Video-LLaVA-7B</td><td>2048</td><td>114.0</td><td>100%</td><td>1053</td><td>100%</td><td>7.6</td><td>1.00×</td></tr><tr><td>+ FastV</td><td>194</td><td>34.3</td><td>20.6%</td><td>212.9</td><td>67.4%</td><td>19.4</td><td>3.32×</td></tr><tr><td>+ PDrop</td><td>194</td><td>45.9</td><td>20.6%</td><td>209.2</td><td>70.5%</td><td>16.0</td><td>2.48×</td></tr><tr><td>+ SparseVLM</td><td>194</td><td>40.0</td><td>10.7%</td><td>121.1</td><td>98.4%</td><td>18.1</td><td>2.85×</td></tr><tr><td>+ CLSE-M(ours)</td><td>194</td><td>41.7</td><td>10.7%</td><td>121.1</td><td>100%</td><td>17.3</td><td>2.73×</td></tr></table>

Ablation Studies. Fig. 5 presents comprehensive ablation analyses on LLaVA-1.5-7B with 192 retained tokens. All results are normalized to the full-token model, where 1.0 indicates matching full performance. Fig. 5(a) evaluates the impact of the Gaussian high-pass cutoff parameter. In all experimental scenarios, we set the cutoff to 0.1. Performance improves substantially when moving from $r ~ = ~ 0 . 0 0$ to small values, indicating that mild suppression of low-frequency components effectively reduces redundancy. Across benchmarks, moderate cutoff values consistently yield the strong CLSE results. Fig. 5(b) evaluates the core components of our framework. We observe a catastrophic performance collapse in CLSE-Inertia (CLSE-I), which retains tokens with the lowest spectral evolution scores (the most static tokens), serving as counterfactual evidence that depth-wise spectral evolution is the primary driver of semantic activity. In contrast, while pure attentionbased pruning suffers from positional bias, our CLSE criterion significantly stabilizes performance. To further fortify the system, CLSE-Hybrid (CLSE-H) integrates CLSE with attention weights that can produce on-par performance like CLSE. Fig. 5(c) evaluates CLSE against alternative depthwise metrics. We compare with Spatial Feature Residual (SFR), which measures the L1-norm of feature residuals, and Mean Magnitude Difference (MMD), which monitors changes in channel-wise mean activations. SFR exhibits performance collapse as its reliance on raw pixel-level intensity makes it susceptible to high-frequency artifacts. MMD, while slightly more stable, remains a coarse-grained heuristic. By collapsing high-dimensional token dynamics into a scalar magnitude, it fails to capture the structural redistribution occurring within the frequency spectrum. In contrast, CLSE consistently establishes a new performance

upper bound.

Qualitative Analysis. Fig. 6 visualizes token importance maps under different strategies. Attention-based scores tend to spread over large background regions and occasionally over-emphasize visually salient but semantically irrelevant areas. In contrast, CLSE concentrates on structurally meaningful regions that actively evolve across layers, such as key objects or human subjects. When the cutoff ratio is set to r = 0.1, the maps exhibit sharper localization and reduced background noise compared to r = 0, confirming that moderate suppression of low-frequency components enhances semantic discrimination. These visualizations support our quantitative findings that cross-layer spectral evolution provides a more reliable token importance signal than raw attention.

Additional Experiments in Appendix. We provide complementary analyses in the appendix to further validate CLSE. These include experimental setup details (e.g., baseline methods, implementation, and datasets), details of a CLSE variant with token merging, ablation on pruning at different layers, experiments on additional MLLMs (InternVL, Qwen2-VL-72B), complete numeric results for LLaVA-1.5-13B, a detailed computational complexity analysis, channel dimension impact on run time, and impact of 3D vs 2D FFT on video MLLMs.

## 5. Conclusion

In this work, we introduced a token pruning framework, namely Cross-Layer Spectral Evolution (CLSE), that models how token representations evolve across depth, capturing the structural transition from high-frequency detail to low-frequency semantic abstraction. Extensive experiments on image and video benchmarks demonstrate that CLSE consistently achieves superior accuracy–efficiency tradeoffs across multiple architectures and compression levels. In particular, the performance gap widens under aggressive pruning, highlighting the robustness of spectral evolution as a token importance signal. We hope this work encourages further exploration of spectral representations for efficient multimodal reasoning.

![](images/52cb5f43be84073197bad4a01ec12f10b1892315028d7907c2134c32f92d0160.jpg)

![](images/79facc9fcf32fc5e099838760d2eec80e38c3d346f5d3815364a4ab03b36e368.jpg)  
(a) Cutoff Ratio r

![](images/9c0406d363cf90e78b3c4737107a2c618342f6e8b7fde4674edda39282584031.jpg)  
(b) Spectral Evolution

![](images/3c569949da88c87f0c4c8ebc9350b79a0e07cdcd7e0b8e790ae88e882559d083.jpg)  
(c) Token Importance Metrics  
Figure 5. Ablation studies on LLaVA-1.5-7B pruned after the first layer of LLM decoder with 192 retained tokens. Scores are normalized to the Vanilla model. (a) Effect of the Gaussian high-pass cutoff ratio. (b) Effect of spectral evolution and its integration with attention. (c) Impact of the magnitude measures for cross-layer component analysis.  
Q: What type of Q: What is the person Q: What is the person sandwich is shown? doing in this image? doing in this image?  
Q: Describe the train scene in this image.  
Q: Describe the kite flying scene.  
Q: What is the person doing in this image?  
Figure 6. Qualitative comparison of token importance maps. Top row: original images. Second row: attention-based token scores. Third and fourth rows: CLSE without frequency suppression (r = 0) and with moderate suppression (r = 0.1), respectively. Compared to attention, CLSE highlights structurally evolving and semantically relevant regions while suppressing redundant background areas. Moderate frequency filtering further improves focus on task-relevant objects.

## Acknowledgements

This work is supported by the National Natural Science Foundation of China under Grant No. 62502429, and the Zhejiang Key Laboratory Project (2024E10001)

## References

[1] Alvar, S.R., Singh, G., Akbari, M., Zhang, Y.: Divprune: Diversity-based visual token pruning for large multimodal models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 9392– 9401 (2025)

[2] Arif, K.H.I., Yoon, J., Nikolopoulos, D.S., Vandierendonck, H., John, D., Ji, B.: Hired: Attentionguided token dropping for efficient inference of highresolution vision-language models. In: Proceedings of the AAAI Conference on Artificial Intelligence.

vol. 39, pp. 1773–1781 (2025)

[3] Bolya, D., Fu, C.Y., Dai, X., Zhang, P., Feichtenhofer, C., Hoffman, J.: Token merging: Your vit but faster. In: The Eleventh International Conference on Learning Representations (2023)

[4] Chen, D., Dolan, W.B.: Collecting highly parallel data for paraphrase evaluation. In: Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies. pp. 190– 200 (2011)

[5] Chen, J., Liu, X., Wen, Z., Wang, Y., Huang, S., Chen, H.: Variation-aware vision token dropping for faster large vision-language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3489–3499 (2026)

[6] Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., Chang, B.: An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. In: European Conference on Computer Vision. pp. 19–35. Springer (2024)

[7] Chen, Z., Cai, Y., Guo, J., Cai, T., Yin, J., Chen, Z.: Accelerating multimodal large language models with prior-corrected token reduction. In: European Conference on Computer Vision. Springer (2026)

[8] Chen, Z., Wang, W., Cao, Y., Liu, Y., Gao, Z., Cui, E., Zhu, J., Ye, S., Tian, H., Liu, Z., et al.: Expanding performance boundaries of open-source multimodal models with model, data, and test-time scaling. arXiv preprint arXiv:2412.05271 (2024)

[9] Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al.: Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 24185–24198 (2024)

[10] Chen, Z., Li, J., Luo, Y., Huang, Z., Yang, Y.: Canzsl: Cycle-consistent adversarial networks for zero-shot learning from natural language. In: WACV 2020. pp. 874–883 (2020)

[11] Chen, Z., Luo, Y., Huang, Z., Li, J., Wang, S., Yu, X.: Distributed zero-shot learning for visual recognition. IEEE Transactions on Multimedia (2026)

[12] Chen, Z., Luo, Y., Qiu, R., Wang, S., Huang, Z., Li, J., Zhang, Z.: Semantics disentangling for generalized zero-shot learning. ICCV 2021 (2021)

[13] Chen, Z., Luo, Y., Wang, S., Li, J., Huang, Z.: Gsmflow: Generation shifts mitigating flow for generalized zero-shot learning. IEEE Transactions on Multimedia (2022)

[14] Chen, Z., Luo, Y., Wang, S., Qiu, R., Li, J., Huang, Z.: Mitigating generation shifts for generalized zero-shot learning. In: ACM MM 2021 (2021)

[15] Chen, Z., Wang, S., Li, J., Huang, Z.: Rethinking generative zero-shot learning: An ensemble learning perspective for recognising visual patches. In: ACM MM 2020. pp. 3413–3421 (2020)

[16] Chen, Z., Yu, X., Tao, X., Li, Y., Huang, Z.: Clusteraware prompt ensemble learning for few-shot vision– language model adaptation. Pattern Recognition p. 112596 (2025)

[17] Chen, Z., Zhang, P., Li, J., Wang, S., Huang, Z.: Zeroshot learning by harnessing adversarial samples. In: ACM MM 2023 (2023)

[18] Chen, Z., Zhao, Z., Guo, J., Li, J., Huang, Z.: Svip: Semantically contextualized visual patches for zeroshot learning. In: ICCV 2025 (2025)

[19] Chen, Z., Zhao, Z., Luo, Y., Li, Y., Tao, X., Huang, Z.: Fastedit: Fast text-guided single-image editing via semantic-aware diffusion fine-tuning. Pattern Recognition p. 112583 (2025)

[20] Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

[21] Fan, Y., Zhao, A., Fu, J., Tong, J., Su, H., Pan, Y., Zhang, W., Shen, X.: Visipruner: Decoding discontinuous cross-modal dynamics for efficient multimodal llms. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 18896–18913 (2025)

[22] Fayyaz, M., Koohpayegani, S.A., Jafari, F.R., Sengupta, S., Joze, H.R.V., Sommerlade, E., Pirsiavash, H., Gall, J.: Adaptive token sampling for efficient vision transformers. In: European conference on computer vision. pp. 396–414. Springer (2022)

[23] Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al.: Mme: A comprehensive evaluation benchmark for multimodal large language models. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2025)

[24] Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., et al.: The llama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024)

[25] Gurari, D., Li, Q., Stangl, A.J., Guo, A., Lin, C., Grauman, K., Luo, J., Bigham, J.P.: Vizwiz grand challenge: Answering visual questions from blind people. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 3608–3617 (2018)

[26] Han, Y., Liu, X., Ding, P., Wang, D., Chen, H., Yan, Q., Huang, S.: Rethinking token reduction in mllms:

Towards a unified paradigm for training-free acceleration. arXiv preprint arXiv:2411.17686 2(3) (2024)

[27] Han, Y., Liu, X., Zhang, Z., Ding, P., Chen, J., Wang, D., Chen, H., Yan, Q., Huang, S.: Filter, correlate, compress: Training-free token reduction for mllm acceleration. In: Proceedings of the 40th AAAI Conference on Artificial Intelligence (2025)

[28] He, Y., Chen, F., Liu, J., Shao, W., Zhou, H., Zhang, K., Zhuang, B.: Zipvl: Efficient large visionlanguage models with dynamic token sparsification. arXiv preprint arXiv:2410.08584 (2024)

[29] Hu, L., Shang, F., Feng, W., Wan, L.: Lightvlm: Acceleraing large multimodal models with pyramid token merging and kv cache compression. arXiv preprint arXiv:2509.00419 (2025)

[30] Huang, K., Zou, H., Xi, Y., Wang, B., Xie, Z., Yu, L.: Ivtp: Instruction-guided visual token pruning for large vision-language models. In: European conference on computer vision. pp. 214–230. Springer (2024)

[31] Hudson, D.A., Manning, C.D.: Gqa: A new dataset for real-world visual reasoning and compositional question answering. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 6700–6709 (2019)

[32] Jang, Y., Song, Y., Yu, Y., Kim, Y., Kim, G.: Tgif-qa: Toward spatio-temporal reasoning in visual question answering. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 2758– 2766 (2017)

[33] Jeddi, A., Baghbanzadeh, N., Dolatabadi, E., Taati, B.: Similarity-aware token pruning: Your vlm but faster. arXiv preprint arXiv:2503.11549 (2025)

[34] Jiang, X., Zhang, X., Gao, N., Deng, Y.: When fast fourier transform meets transformer for image restoration. In: European conference on computer vision. pp. 381–402. Springer (2024)

[35] Lee-Thorp, J., Ainslie, J., Eckstein, I., Ontanon, S.: Fnet: Mixing tokens with fourier transforms. In: Proceedings of the 2022 Conference of the north American chapter of the Association for Computational Linguistics: human language technologies. pp. 4296– 4313 (2022)

[36] Li, D., Yang, Z., Lu, S.: Todre: Visual token pruning via diversity and task awareness for efficient large vision-language models. arXiv e-prints pp. arXiv– 2505 (2025)

[37] Li, X., Zhang, D., Du, Z., Zhu, L., Chen, Z., Li, J.: Pataug: Augmentation of augmentation for testtime adaptation. In: ACM MM 2025, pp. 5080–5089 (2025)

[38] Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W.X., Wen, J.R.: Evaluating object hallucination in large visionlanguage models. In: Proceedings of the 2023 Confer-

ence on Empirical Methods in Natural Language Processing. pp. 292–305 (2023)

[39] Liang, X., Guan, C., Lu, J., Chen, H., Wang, H., Hu, H.: Dynamic token reduction during generation for vision language models. arXiv preprint arXiv:2501.14204 (2025)

[40] Lim, J.S., Chen, Z., Chen, Z., Baktashmotlagh, M., Yu, X., Huang, Z., Luo, Y.: Dipex: Dispersing prompt expansion for class-agnostic object detection. NeurIPS2024 (2024)

[41] Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., Yuan, L.: Video-llava: Learning united visual representation by alignment before projection. In: Proceedings of the 2024 conference on empirical methods in natural language processing. pp. 5971–5984 (2024)

[42] Liu, H., Li, C., Li, Y., Lee, Y.J.: Improved baselines with visual instruction tuning. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 26296–26306 (2024)

[43] Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., Lee, Y.J.: Llava-next: Improved reasoning, ocr, and world knowledge. In: Advances in Neural Information Processing Systems (NeurIPS) (2024)

[44] Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. In: Advances in Neural Information Processing Systems (NeurIPS) (2023)

[45] Liu, Y., Wu, F., Li, R., Tang, Z., Li, K.: Par: Promptaware token reduction method for efficient large multimodal models. arXiv preprint arXiv:2410.07278 (2024)

[46] Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? In: European conference on computer vision. pp. 216–233. Springer (2024)

[47] Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X.C., Liu, C.L., Jin, L., Bai, X.: Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences 67(12), 220102 (2024)

[48] Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.W., Zhu, S.C., Tafjord, O., Clark, P., Kalyan, A.: Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems 35, 2507–2521 (2022)

[49] Nguyen, T., Pham, M., Nguyen, T., Nguyen, K., Osher, S., Ho, N.: Fourierformer: Transformer meets generalized fourier integral theorem. Advances in Neural Information Processing Systems 35, 29319– 29335 (2022)

[50] Rao, Y., Zhao, W., Liu, B., Lu, J., Zhou, J., Hsieh, C.J.: Dynamicvit: Efficient vision transformers

with dynamic token sparsification. Advances in neural information processing systems 34, 13937–13949 (2021)

[51] Shang, Y., Cai, M., Xu, B., Lee, Y.J., Yan, Y.: Llavaprumerge: Adaptive token reduction for efficient large multimodal models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 22857–22867 (2025)

[52] Shao, Y., Lin, D., Yan, M., Chen, S., Zeng, F., Liao, M., Ma, A., Yan, Z., Wang, H., Wang, Y., et al.: Tr-dq: Time-rotation diffusion quantization. In: AAAI 2026 (2026)

[53] Shi, D., Tao, C., Rao, A., Yang, Z., Yuan, C., Wang, J.: Crossget: Cross-guided ensemble of tokens for accelerating vision-language transformers. arXiv preprint arXiv:2305.17455 (2023)

[54] Si, G., Yin, H., Li, X., Liao, W., He, T., Peng, P., Zhu, W., et al.: Infoprune: Revisiting visual token pruning from an information-theoretic perspective

[55] Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., Rohrbach, M.: Towards vqa models that can read. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 8317–8326 (2019)

[56] Song, D., Wang, W., Chen, S., Wang, X., Guan, M.X., Wang, B.: Less is more: A simple yet effective token reduction method for efficient multi-modal llms. In: Proceedings of the 31st International Conference on Computational Linguistics. pp. 7614–7623 (2025)

[57] Su, H., Li, J., Chen, Z., Zhu, L., Lu, K.: Distinguishing unseen from seen for generalized zero-shot learning. CVPR 2022 (2022)

[58] Sun, Z., Ma, Y., Liu, G., Chen, Y., Tang, X., Hu, Y., Xu, Y.: Ivc-prune: Revealing the implicit visual coordinates in lvlms for vision token pruning. arXiv preprint arXiv:2602.03060 (2026)

[59] Wang, H., Kai, J., Bai, H., Hou, L., Jiang, B., He, Z., Lin, Z.: Fourier-vlm: Compressing vision tokens in the frequency domain for large vision-language models. arXiv preprint arXiv:2508.06038 (2025)

[60] Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, Z., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024)

[61] Wang, Q., Ye, H., Chung, M.Y., Liu, Y., Lin, Y., Kuo, M., Ma, M., Zhang, J., Chen, Y.: Corematching: A co-adaptive sparse inference framework with token and neuron pruning for comprehensive acceleration of vision-language models. arXiv preprint arXiv:2505.19235 (2025)

[62] Wang, T., Guo, J., Li, D., Chen, Z.: On the discrim-

ination and consistency for exemplar-free class incremental learning. In: IJCAI 2025 (2025)

[63] Wang, W., Guo, J., Cai, Y., Chen\*, Z.: Learning multimodal prototypes for cross-domain few-shot object detection. In: CVPR 2026 Findings (2026)

[64] Wang, Y., Wu, J., Ni, Z., Yang, C., Liu, Y., Yang, L., Zhou, Y., Wen, Y., He, L.: Entropyprune: Matrix entropy guided visual token pruning for multimodal large language models. arXiv preprint arXiv:2602.17196 (2026)

[65] Wang, Z., Luo, H., Wang, P., Ding, F., Wang, F., Li, H.: Vtc-lfc: Vision transformer compression with low-frequency components. Advances in Neural Information Processing Systems 35, 13974–13988 (2022)

[66] Wen, Z., Gao, Y., Li, W., He, C., Zhang, L.: Token pruning in multimodal large language models: Are we solving the right problem? In: Findings of the Association for Computational Linguistics: ACL 2025. pp. 15537–15549 (2025)

[67] Wen, Z., Gao, Y., Wang, S., Zhang, J., Zhang, Q., Li, W., He, C., Zhang, L.: Stop looking for “important tokens” in multimodal language models: Duplication matters more. In: Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing. pp. 9972–9991 (2025)

[68] Xing, L., Wang, A.J., Yan, R., Shu, X., Tang, J.: Vision-centric token compression in large language model. arXiv preprint arXiv:2502.00791 (2025)

[69] Xing, L., Huang, Q., Dong, X., Lu, J., Zhang, P., Zang, Y., Cao, Y., He, C., Wang, J., Wu, F., et al.: Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. In: In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (2025)

[70] Xu, D., Zhao, Z., Xiao, J., Wu, F., Zhang, H., He, X., Zhuang, Y.: Video question answering via gradually refined attention over appearance and motion. In: Proceedings of the 25th ACM international conference on Multimedia. pp. 1645–1653 (2017)

[71] Xu, J., Mei, T., Yao, T., Rui, Y.: Msr-vtt: A large video description dataset for bridging video and language. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 5288–5296 (2016)

[72] Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025)

[73] Yang, C., Sui, Y., Xiao, J., Huang, L., Gong, Y., Li, C., Yan, J., Bai, Y., Sadayappan, P., Hu, X., et al.: Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision

A Implementation Details 15  
A.1. Datasets . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . A.2. Baselines . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . A.3. Experimental Setup . . . . . . . . . . . . . . . . . . . . . A.4. Method Pipeline of CLSE with Token-Merging 18   
B Computational Efficiency 19  
B.1. Theoretical FLOPs 19  
B.2. Latency Measurement 19   
C Additional Experiments 20  
C.1. Additional Ablation Studies 20  
C.2. Results on LLaVA-1.5-13B 20  
C.3. Results on InternVL-2.5-8B 21  
C.4. Results on Qwen2-VL-72B 22

language model. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 19803– 19813 (2025)

[74] Yang, S., Chen, Y., Tian, Z., Wang, C., Li, J., Yu, B., Jia, J.: Visionzip: Longer is better but not necessary in vision language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19792–19802 (2025)

[75] Ye, W., Wu, Q., Lin, W., Zhou, Y.: Fit and prune: Fast and training-free visual token pruning for multimodal large language models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 39, pp. 22128–22136 (2025)

[76] Yin, H., Vahdat, A., Alvarez, J.M., Mallya, A., Kautz, J., Molchanov, P.: A-vit: Adaptive tokens for efficient vision transformer. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10809–10818 (2022)

[77] You, F., Li, J., Chen, Z., Zhu, L.: Pixel exclusion: Uncertainty-aware boundary discovery for active cross-domain semantic segmentation. In: ACM MM 2022 (2022)

[78] You, F., Li, J., Zhu, L., Chen, Z., Huang, Z.: Domain adaptive semantic segmentation without source data. In: ACM MM 2021. pp. 3293–3302 (2021)

[79] Yu, H., Li, W., Qu, X., Wang, S., Chen, J., Zhu, J.: Visiontrim: Unified vision token compression for training-free mllm acceleration. arXiv preprint arXiv:2601.22674 (2026)

[80] Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., Wang, L.: Mm-vet: Evaluating large multimodal models for integrated capabilities. In: Fortyfirst International Conference on Machine Learning (2024)

[81] Yu, Z., Liao, Z., Li, J., Chen, Z., Zhu, L.: Dynamic target distribution estimation for source-free open-set domain adaptation. In: AAAI 2025. vol. 39, pp. 22254– 22262 (2025)

[82] Yu, Z., Xu, D., Yu, J., Yu, T., Zhao, Z., Zhuang, Y., Tao, D.: Activitynet-qa: A dataset for understanding complex web videos via question answering. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 33, pp. 9127–9134 (2019)

[83] Zamini, M., Shukla, D.: Delta-llava: Basethen-specialize alignment for token-efficient visionlanguage models. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 3648–3657 (2026)

[84] Zeng, Q.S., Li, Y., Wang, Q., Jiang, P.T., Wu, Z., Cheng, M.M., Hou, Q.: A glimpse to compress: Dynamic visual token pruning for large vision-language models. arXiv preprint arXiv:2508.01548 (2025)

[85] Zhang, Q., Liu, M., Li, L., Lu, M., Zhang, Y., Pan, J., She, Q., Zhang, S.: Beyond attention or similarity: Maximizing conditional diversity for token pruning in mllms. arXiv preprint arXiv:2506.10967 (2025)

[86] Zhang, Y., Wang, S., Chen, Z., Xu, X., Funiak, S., Liu, J.: Towards cost-efficient federated multi-agent rl with learnable aggregation. In: PAKDD 2024, pp. 171–183. Springer Nature Singapore (2024)

[87] Zhang, Y., Fan, C.K., Ma, J., Zheng, W., Huang, T., Cheng, K., Gudovskiy, D.A., Okuno, T., Nakata, Y., Keutzer, K., et al.: Sparsevlm: Visual token sparsification for efficient vision-language model inference. In: Forty-second International Conference on Machine Learning (2025)

[88] Zhao, Z., Chen, Z., Huang, Z., Sadiq, S., Chen, T.: Continual text-to-video retrieval with frame fusion and task-aware routing. In: SIGIR 2025. pp. 1011– 1021 (2025)

[89] Zhao, Z., Chen, Z., Huang, Z., Sadiq, S., Chen, T.: Generative recall, dense reranking: Learning multiview semantic ids for efficient text-to-video retrieval. arXiv preprint arXiv:2601.21193 (2026)

[90] Zhao, Z., Song, S., Chen, T., Chen, Z., Sadiq, S., Luo, Y.: Are synthetic videos useful? a benchmark for retrieval-centric evaluation of synthetic videos. In: ACM MM 2025 (2025)

[91] Zhuang, X., Zhu, Z., Xie, Y., Liang, L., Zou, Y.: Vasparse: Towards efficient visual hallucination mitigation via visual-aware token sparsification. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 4189– 4199 (June 2025)

## Appendix

D. Theoretical Analysis

## A. Implementation Details

## A.1. Datasets

Our experiments are conducted on a suite of widely recognized benchmarks spanning image understanding and video understanding, each targeting a distinct aspect of multimodal intelligence.

## Image Understanding Benchmarks.

GQA [31] targets visual scene understanding through a structured combination of scene graphs, image content, and compositional questions. Each image is accompanied by detailed spatial annotations and per-object attribute information, enabling fine-grained grounding of questionanswer pairs. The benchmark is designed to probe a model’s reasoning over object relationships, spatial layouts, and scene-level semantics beyond simple recognition.

MMBench [46] adopts a three-tier capability taxonomy to systematically profile multimodal models. At the broadest level, it distinguishes perception from reasoning; the intermediate level refines these into six cognitive sub-abilities; and the most granular level further decomposes performance across 20 fine-grained dimensions. This structured hierarchy enables interpretable and multi-faceted analysis of model strengths and weaknesses. MMBench-CN is a Chinese-language variant built upon the same evaluation framework.

MME [23] offers a systematic diagnostic of perceptual and cognitive abilities through 14 targeted subtasks, each defined by carefully formulated instruction-answer pairs. The benchmark prioritizes brevity and precision in its instructions to mitigate data leakage risks and ensure evaluation fairness. Spanning both low-level visual perception and high-level reasoning, MME provides a thorough and reliable measure of multimodal model capabilities.

POPE [38] addresses the problem of object hallucination through a binary probing paradigm: models are presented with yes/no questions about the existence of specific objects in an image. Performance is measured by accuracy, recall, precision, and F1 score across three sampling strategies (random, popular, and adversarial), enabling a controlled and multi-faceted analysis of hallucination tendencies in multimodal systems.

ScienceQA [48] is a large-scale multimodal benchmark drawing from elementary and middle school science curricula across three major domains: natural science, social science, and language science. Questions are organized hierarchically into 26 topics, 127 categories, and 379 fine-grained skills, yielding a richly structured evaluation space. Each question is accompanied by a natural language explanation and, for a subset, relevant context images, enabling assessment of not only answer accuracy but also multi-step reasoning chains and cross-modal interpretation. The breadth of subject matter combined with the depth of the skill taxonomy makes it a demanding testbed for cross-domain generalization.

TextVQA [55] probes a model’s capacity to jointly reason over visual scenes and the text embedded within them. Unlike purely visual QA tasks, answering TextVQA questions demands that a model first localize and read text regions in the image, then integrate that textual content with the broader visual context to produce accurate responses. This dual dependency on visual perception and text recognition makes it a particularly challenging benchmark at the intersection of OCR and visual reasoning.

VizWiz [25] is grounded in a real-world accessibility scenario: images were taken by blind individuals using smartphone cameras, and the accompanying questions reflect their actual information needs. The dataset comprises 20,523 training, 4,319 validation, and 8,000 test imagequestion pairs, each annotated by 10 crowd-sourced workers. A distinctive challenge is that a non-trivial fraction of questions are inherently unanswerable due to image quality issues, requiring models to simultaneously assess question answerability and produce accurate responses when possible.

OCRBench [47] provides a systematic evaluation of text perception and understanding across five task categories: text recognition in natural scenes, scene text-centric VQA, document-oriented VQA, key information extraction, and handwritten mathematical expression recognition. By spanning this diverse range of text-centric challenges, it enables thorough assessment of a multimodal model’s proficiency in handling textual content encountered across real-world visual environments.

## Video Understanding Benchmarks.

TGIF-QA [32] extends visual question answering to the video domain using animated GIFs sourced from Tumblr, comprising 165,000 QA pairs. It defines four task types that collectively demand temporal understanding beyond what static image QA can assess: repetition counting (how many times an action occurs), repeating action identification (what action is repeated a given number of times), state transition reasoning (what changes between two temporal states), and frame-level open-ended QA. This multi-task design makes TGIF-QA a comprehensive benchmark for evaluating both temporal and spatial reasoning in video understanding models.

MSVD-QA [4, 70] builds on the Microsoft Research Video Description corpus, transforming its 1,970 short video clips into an open-ended question answering benchmark with approximately 50.5K QA pairs. Questions are organized into five interrogative categories (what, who, how, when, and where), covering a broad spectrum of video content

Q: Which one is the correct caption of this image? A: A woman is riding a motorcycle down the street. POPE  
GQA  
![](images/8da76a3162ea34ca4dca32f2ae4624e0ae4d66ced8ba02674ef453824e958b08.jpg)  
Q: Are there beds next to the small outlet? A: yes

MMBench  
![](images/295efa8e61e5eaa1fcdc188b5e70124369207055486893d3527a1b6ebd37571c.jpg)

MMBench-CN  
![](images/78754e2d02f22912f0e240bb8adf0cfa73b9b7df88327f0608b4976b6ddf212c.jpg)  
Q: 以下哪个是这 图 的 确标 ？A: 一个 人在街上 车。

MME  
![](images/ae4a926bcf7ef9db57a93ec4cc5544c625df229479fd5e96df575405afba4369.jpg)  
Q: Are there two bowls in this image? A: Yes

![](images/6e6a776c88f6b3de95e04eb81caaf6a76acc8126299a31bcadddc286cd90d212.jpg)  
Q: Is there a bed in the image? A: yes

ScienceQA  
![](images/c7eab8cf54e179783479316756efd1025aed5c6e4cf52face92082ec100a7aad.jpg)  
Q: Is Rafflesia arnoldii made up of many cells? A: yes

TextVQA  
![](images/02cedceaac37dd2b66bf0952d961bcec581785dcf2ef9f28ae5807c242e93168.jpg)  
Q: What kind of beer is this? A: ale

VizWiz  
![](images/8767f090c613a90a86db0a372e44ad9f6bb72ca87d4ea5e682a906839a532ce5.jpg)  
Q: What is the name of the drink? A: irn bru

OCRBench  
![](images/4f94b3f9b6b3ad19310bf519074e563b6639de2587c1d22e2210a2e786af8e8c.jpg)  
Q: What is the Mosman Manly exit going to? A: Chatswood Epping

Figure 7. Representative examples from each image understanding benchmark used in our evaluation. Each cell shows the benchmark name, a sample image, the question posed to the model, and the ground-truth answer.

and temporal aspects, and supporting evaluation across both video QA and video captioning scenarios.

MSRVTT-QA [70, 71] scales video QA evaluation to 10,000 clips and 243,000 QA pairs drawn from the MSR-VTT dataset. Its central challenge is that accurate responses require models to jointly process visual appearance and temporal dynamics across the clip, neither alone is sufficient. Sharing the same five question-type taxonomy as MSVD-QA (what, who, how, when, where), it serves as a higher-volume and more diverse complement for bench marking video comprehension.

ActivityNet-QA [82] is a large-scale benchmark designed for understanding complex web videos through question answering. It contains 58K human-annotated QA pairs over 5,800 video clips sourced from the ActivityNet dataset, covering a wide range of real-world human activities. Questions require temporal reasoning and holistic video comprehension, making it a challenging testbed for long-form video understanding.

## A.2. Baselines

We compare against a range of representative tokenreduction methods for accelerating multimodal large language models (MLLMs). Despite sharing the common goal of eliminating redundant visual tokens, these methods differ substantially in their strategies, spanning token merging, attention-guided pruning, and adaptive budget allocation.

ToMe [3] consolidates redundant visual representations directly within transformer attention blocks by applying a bipartite soft-matching procedure at each layer. Tokens are partitioned into two complementary sets, and the most similar pairs across the boundary are selected and fused into unified embeddings, carrying proportionally combined features rather than discarding information outright. By folding this matching step into the standard forward pass, ToMe achieves near-linear throughput improvements as a function of the merge ratio with negligible additional memory overhead. No task-specific retraining or architectural modification is required, making it immediately applicable to any vision transformer backbone. This training-free property and architectural agnosticism make ToMe a generalpurpose compression primitive.

FastV [6] targets the computational bottleneck introduced by long visual token sequences by pruning low-value tokens at an early inference stage inside the LLM. Token importance is assessed using attention maps produced by the language model itself, rather than the upstream vision encoder, providing query-conditioned relevance estimates that are directly grounded in how the model processes the current input. Tokens that accumulate little attention across the early LLM layers are judged uninformative for the current query and discarded before reaching computationally heavier deeper layers. Because the pruning decision is made at inference time without any parameter updates, FastV integrates into existing MLLM pipelines as a plug-and-play module. The method exploits the empirical regularity that LLM attention is strongly concentrated on a sparse subset of visual tokens from the earliest layers, yielding substantial FLOPs reduction with modest accuracy degradation.

SparseVLM [87] formulates token selection as a crossmodal relevance problem, deriving per-token importance scores from the interaction between textual query representations and visual features. A key design choice is the adoption of sample-adaptive sparsity: rather than imposing a fixed retention ratio uniformly across all inputs, the method calibrates pruning aggressiveness to the estimated informational density of each image, retaining more tokens for visually complex or query-ambiguous scenes and pruning more aggressively when semantic content is concentrated. The approach further incorporates a token recycling mechanism to partially compensate for information loss—discarded tokens are not simply dropped but are redistributed into the retained set through a lightweight aggregation step, allowing low-attention regions to contribute to the final representation in compressed form. The combination of cross-modal scoring, content-adaptive budgeting, and partial information recovery enables SparseVLM to sustain strong task performance under significant compression without any model fine-tuning.

HiRED [2] addresses the token budget problem in highresolution MLLMs that decompose inputs into a global thumbnail paired with multiple local crops, where naive uniform allocation wastes representational resources on uninformative regions. Its compression strategy proceeds in two hierarchically nested stages: at the partition level, each crop’s token budget is set in proportion to its CLS attention score, ensuring that image regions attracting greater query-driven attention receive correspondingly more representation capacity. Within each budget-allocated partition, the locally most informative tokens are then retained and the remainder pruned, preserving fine-grained spatial structure where it is most needed. This two-stage design—global budget distribution followed by local token selection—maintains spatial awareness throughout the reduction process, a property that flat, context-insensitive methods cannot provide. HiRED is consequently well-suited to tasks involving fine-grained localization or structured scene understanding, where the informational value of image subregions is highly non-uniform.

LLaVA-PruMerge [51] integrates pruning and merging into a sequential two-stage compression pipeline tailored to the architectural conventions of LLaVA-style models. In the first stage, token importance is estimated through sparse CLS-to-visual cross-attention: tokens that attract low aggregate attention from the class token are flagged as lowpriority candidates and removed from the sequence. In the second stage, the surviving tokens are clustered according to the pairwise similarity of their key vectors, and tokens within each cluster are fused into a single unified embedding, reducing the token count further without discarding the information they collectively encode. This two-phase design achieves compression at two independent granularities—coarse-level elimination through pruning and fine-level consolidation through key-similarity merging—allowing the method to operate effectively under tight token budgets. The entire pipeline is training-free and introduces no additional parameters, relying exclusively on attention-derived and similarity-derived signals that are already available during the forward pass.

PyramidDrop [69] applies a staged compression schedule in which the proportion of tokens dropped increases progressively with layer depth, resulting in a pyramid-shaped token count profile across the transformer stack. The rationale is grounded in the observation that early layers process relatively distributed feature maps where broad coverage is beneficial, while deeper layers operate on increasingly abstracted, task-focused representations where a smaller, more focused token set is sufficient. By retaining a fuller complement of tokens in shallow layers and incrementally discarding a larger fraction at successive depth checkpoints, PyramidDrop avoids the abrupt representational discontinuity characteristic of single-stage pruning schemes. This depth-aware schedule allocates computational resources proportionally to the layer-wise marginal value of token diversity, delivering significant aggregate FLOPs savings across the full network. The pyramid structure thereby achieves a more principled trade-off between efficiency and representational fidelity than uniform or single-point pruning.

FiCoCo [26] addresses token redundancy through a unified three-stage filter–correlate–compress pipeline that tackles different sources of excess representation in a principled sequence. The filter stage performs an initial coarse selection, removing tokens whose predicted contribution to downstream task performance falls below a lightweight salience threshold, eliminating the most obvious redundancy at low cost. The correlate stage then analyses pairwise relationships among surviving tokens—both within the visual modality and across text-visual boundaries—to identify clusters of tokens whose encoded information substantially overlaps. The compress stage aggregates each identified cluster into a single compact representative, further condensing the sequence while retaining the essential semantic content of the group. By explicitly decoupling these three operations, FiCoCo can address complementary forms of redundancy independently and in sequence, enabling more thorough compression than single-criterion methods that optimize for only one dimension of token reduction at a time.

## A.3. Experimental Setup

All experiments are conducted on a server equipped with six NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs, each with 96 GB of VRAM, providing ample capacity to host large-scale vision-language models without cross-GPU tensor parallelism for most configurations. The software environment consists of Python 3.10, PyTorch 2.9.1, CUDA 12.8, and cuDNN 9.1.0. All models are loaded and evaluated in float16 precision. Inference is performed with a batch size of one to faithfully reflect single-sample latency, which is the practically relevant setting for interactive MLLM deployment. All baselines are reproduced under the same hardware and software stack using their official codebases. No task-specific fine-tuning is applied to any method; all token-reduction decisions are made at inference time.

## A.4. Method Pipeline of CLSE with Token-Merging

We describe the complete pipeline of CLSE, covering token importance scoring, rank-adaptive pruning, and the merging step that recovers information from discarded tokens.

Overall Flow. At a designated pruning layer ℓ, the model (i) runs a normal forward pass to obtain hidden states before and after the layer, (ii) computes a per-token importance score from the resulting representations, (iii) retains the topranked tokens and discards the rest, (iv) optionally merges a subset of discarded tokens into compact representatives, and (v) reassembles the sequence for all subsequent layers. Per-Frame 2D FFT vs. 3D Spatiotemporal FFT. A natural question for video inputs is whether a 3D FFT over the full spatiotemporal volume could capture richer temporal structure than per-frame 2D FFT. We compare both variants on Video-LLaVA-7B [41], where $\mathrm { C L S E _ { 2 d f } }$ applies 2D FFT independently to each sampled frame and $\mathrm { C L S E _ { 3 d } }$ applies a single 3D FFT across all frames. As shown in Table 6, per-frame 2D FFT substantially outperforms its 3D counterpart on every benchmark and evaluator. This is expected: with only $T = 8$ sparsely sampled frames, the temporal dimension is far too short for reliable 3D spectral analysis, and temporal coherence is more effectively captured by the cross-layer evolution factor. 3D FFT also loses the ability to process frames in parallel, introducing additional computational overhead without accuracy benefit.

Cross-Layer Spectral Evolution (CLSE). The CLSE score measures how much each token’s spectral structure changes between adjacent layers; its computation is detailed in the main paper. The final per-token importance multiplies the text-to-visual attention score by the CLSE factor. The number of tokens to retain is determined adaptively by the numerical rank of the text-visual attention matrix $\mathbf { A } _ { \mathrm { t e x t }  \mathrm { v i s } } ,$ rather than a fixed ratio: the top-r tokens by joint score are kept, where $r = \mathrm { r a n k } ( { \bf A } _ { \mathrm { t e x t  v i s } } )$

Density-Peak Clustering Merge. Pruned tokens are not simply discarded. A two-stage recycling mechanism recovers informative tokens from the pruned set. In Stage 1, we select the top 30% of pruned tokens by joint score, filtering out those with negligible attention. In Stage 2, we apply density-peak clustering to compress these K recovered tokens into $\lfloor K / 1 0 \rfloor + 1$ representative embeddings. Cluster centers are identified by jointly maximizing local density and distance to the nearest higher-density neighbor; each remaining token is then assigned to its closest center and compressed by uniform averaging within the cluster. The final reassembled sequence takes the form:

Table 6. Ablation on 2D per-frame FFT vs. 3D spatiotemporal FFT for video token scoring on Video-LLaVA-7B, retaining 194 tokens (↓90.5%). $\mathrm { C L S E _ { 2 d f } }$ applies 2D FFT independently to each sampled frame, while CLSE<sub>3d</sub> applies a single 3D FFT over the full spatiotemporal volume. Per-frame 2D FFT consistently and substantially outperforms 3D FFT across all benchmarks and evaluators.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Eval</td><td colspan="2">TGIF</td><td colspan="2">MSVD</td><td colspan="2">MSRVTT</td><td colspan="2">ActivityNet</td><td colspan="2">Avg.</td></tr><tr><td>Acc.</td><td>Score</td><td>Acc.</td><td>Score</td><td>Acc.</td><td>Score</td><td>Acc.</td><td>Score</td><td>Acc.</td><td>Score</td></tr><tr><td rowspan="2">Video-LLaVA-7B</td><td>GPT</td><td>11.3</td><td>1.19</td><td>58.9</td><td>3.31</td><td>42.7</td><td>2.69</td><td>42.0</td><td>2.20</td><td>38.7</td><td>2.35</td></tr><tr><td>Gem</td><td>14.2</td><td>0.77</td><td>57.3</td><td>2.93</td><td>47.5</td><td>2.32</td><td>42.5</td><td>2.12</td><td>40.4</td><td>2.04</td></tr><tr><td colspan="12">Retain 194 Tokens</td></tr><tr><td rowspan="2">CLSE $_{3d}$ </td><td>GPT</td><td>3.0</td><td>1.08</td><td>36.7</td><td>2.37</td><td>31.4</td><td>2.16</td><td>35.1</td><td>1.87</td><td>26.5</td><td>1.87</td></tr><tr><td>Gem</td><td>3.3</td><td>0.18</td><td>36.4</td><td>1.89</td><td>37.0</td><td>1.80</td><td>36.3</td><td>1.81</td><td>28.2</td><td>1.42</td></tr><tr><td rowspan="2">CLSE $_{2df}$ </td><td>GPT</td><td>9.6</td><td>1.06</td><td>54.1</td><td>3.11</td><td>39.3</td><td>2.56</td><td>39.9</td><td>2.03</td><td>35.7</td><td>2.19</td></tr><tr><td>Gem</td><td>12.9</td><td>0.69</td><td>56.3</td><td>2.88</td><td>47.7</td><td>2.27</td><td>39.9</td><td>1.99</td><td>39.2</td><td>1.95</td></tr></table>

$$
\left[ \underbrace {\text { pre - prompt }} _ {\text { unchanged }} \mid \underbrace {\text { kept   visual }} _ {r \text {   tokens }} \mid \underbrace {\text { merged   reps }} _ {\lfloor K / 1 0 \rfloor + 1 \text {   tokens }} \mid \underbrace {\text { text   tokens }} _ {\text { unchanged }} \right].\tag{15}
$$

This design preserves discriminative high-frequency tokens at full resolution while representing redundant-butinformative tokens compactly, achieving a favorable balance between compression ratio and information retention.

## B. Computational Efficiency

## B.1. Theoretical FLOPs

To characterize the computational cost of MLLMs, we analyze the two dominant components: the self-attention mechanism and the feed-forward network (FFN). The total FLOPs of a transformer with T layers, sequence length $n ,$ hidden dimension $d ,$ and FFN intermediate size m can be written as:

$$
\text { Total   FLOPs } = T \times \left(4 n d ^ {2} + 2 n ^ {2} d + 2 n d m\right).\tag{16}
$$

This expression reveals the quadratic dependence on sequence length n, motivating token reduction as an effective strategy for inference acceleration. Following FastV [6], we estimate FLOPs under token pruning as:

$$
\begin{array}{r l} \text { Post - Pruning   FLOPs } & = L \times (4 n d ^ {2} + 2 n ^ {2} d + 2 n d m) \\ & \quad + (T - L) \times (4 \hat {n} d ^ {2} + 2 \hat {n} ^ {2} d + 2 \hat {n} d m) \end{array}\tag{17}
$$

where $L$ is the pruning layer and nˆ is the sequence length after pruning. The theoretical FLOPs reduction ratio is then:

$$
1 - \frac {\text { Post - Pruning   FLOPs }}{\text { Total   FLOPs }}.\tag{18}
$$

FFT Scoring Overhead. The CLSE scoring step introduces an additional per-inference cost from the per-frame 2D FFT. Let $N _ { v }$ denote the total number of visual tokens, F the number of sampled frames, and $N _ { f } ~ = ~ N _ { v } / F$ the spatial tokens per frame. For each frame, a 2D FFT is applied to the $N _ { f }$ -point spatial grid independently across all d feature channels. A 2D FFT over $N _ { f }$ points (computed via torch.fft.fft2) requires $\mathcal { O } ( \dot { N } _ { f } \log _ { 2 } N _ { f } )$ operations per channel. Summing over all frames and channels yields:

$$
\text { FFT   FLOPs } = \mathcal {O} (d N _ {v} \log_ {2} (N _ {v} / F))  .\tag{19}
$$

For image inputs $( F = 1 )$ this reduces to $\mathcal { O } ( d N _ { v } \log _ { 2 } { N _ { v } } )$ To contextualize this cost, consider LLaVA-1.5-7B [42] with $d { = } 4 0 9 6 , \ N _ { v } { = } 5 7 6$ , and $F { = } 1 { : }$ : the FFT scoring step incurs approximately 108 M FLOPs, whereas a single transformer layer costs ≈97 B FLOPs under the formula above $( n { = } 6 0 0 , m { = } 1 1 0 0 8 )$ . The FFT overhead thus constitutes less than 0.12% of a single layer and under 0.004% of the total forward-pass cost, confirming that the spectral scoring step is computationally negligible.

## B.2. Latency Measurement

We measure wall-clock latency using torch.cuda.Event to ensure precise GPU-side timing. Benchmarks are conducted at batch size 1 with max new tokens=1 to isolate the prefill-stage cost. For each configuration, we perform 10 warm-up runs followed by $N = 1 0 0$ timed iterations, recording the mean $( \mu )$ and standard deviation (σ):

$$
\mu = \frac {1}{N} \sum_ {i = 1} ^ {N} t _ {i}, \quad \sigma = \sqrt {\frac {1}{N} \sum_ {i = 1} ^ {N} (t _ {i} - \mu) ^ {2}}\tag{20}
$$

The end-to-end inference latency $T _ { t o t a l }$ is decomposed into two primary components: (1) ViT Latency $( T _ { V i T } )$ the time consumed by the visual encoder and projector; and (2) Prefill Latency $( T _ { p r e f i l l } )$ , the language model’s forward pass over the multimodal sequence. $T _ { p r e f i l l }$ is measured by feeding pre-computed embeddings directly into the decoder to isolate language modeling from visual feature extraction.

The total latency is formulated as:

$$
T _ {t o t a l} \approx T _ {V i T} + T _ {p r e f i l l}, \quad \text { where } \quad T _ {p r e f i l l} = \sum_ {l = 1} ^ {D} \operatorname{Layer} (S _ {l})\tag{21}
$$

In this formulation, D is the number of decoder layers and $S _ { l }$ denotes the sequence length at layer l. This accounts for dynamic token reduction strategies where $S _ { l }$ may vary across layers. While $T _ { V i T }$ typically remains constant for a fixed input resolution, the overall efficiency is primarily determined by the adaptive reduction of $S _ { l }$ during the prefill stage.

![](images/6b33c8f05c1b90295153131816eab12c3f37ffb7817dca58552e0e918a7cbce5.jpg)  
Figure 8. FFT latency as a function of channel dimension C across four spatial resolutions $( N ~ \in ~ \{ 1 0 2 4 , 1 6 0 0 , 2 3 0 4 , 4 0 9 6 \} )$ ). Latency scales approximately linearly with both C and N, remaining in the low-millisecond range across all tested configurations.

Table 7. Ablation study on Z-Score normalization and temperature in CLSE-Hybird on LLaVA-1.5-7B. Without Z-score, the raw evolution intensity is used directly as the gating factor. Z-score normalization consistently improves performance, and $\tau = 0 . 1$ achieves the best results across all benchmarks. Excessively large τ flattens the sigmoid gate toward a uniform weight, degrading token selection.

<table><tr><td>Setting</td><td>Temp (τ)</td><td>GQA ↑</td><td>MMB ↑</td><td>MME ↑</td><td>POPE ↑</td></tr><tr><td>Vanilla (Full Tokens)</td><td>-</td><td>61.9</td><td>64.7</td><td>1862</td><td>85.9</td></tr><tr><td colspan="6">CLSE-H Pruning (Retain 192 Tokens, ↓66.7%)</td></tr><tr><td>No Z-Score</td><td>-</td><td>60.5</td><td>61.9</td><td>1796</td><td>82.7</td></tr><tr><td>With Z-Score</td><td>0.01</td><td>60.8</td><td>62.7</td><td>1805</td><td>83.7</td></tr><tr><td>With Z-Score</td><td>0.1</td><td>60.9</td><td>62.7</td><td>1815</td><td>83.8</td></tr><tr><td>With Z-Score</td><td>1</td><td>59.9</td><td>62.7</td><td>1781</td><td>81.6</td></tr><tr><td>With Z-Score</td><td>10</td><td>56.8</td><td>62.3</td><td>1751</td><td>74.6</td></tr></table>

## C. Additional Experiments

## C.1. Additional Ablation Studies

FFT Latency Scaling with Channel Dimension. Since CLSE applies a 2D FFT independently per feature channel, the spectral scoring cost scales as O(C · N log N ), where C is the channel dimension and N is the number of spatial tokens. The FFT is implemented via torch.fft.fft2 (cuFFT backend) over batched channel dimensions in a single kernel call, operating in bfloat16 precision on NVIDIA RTX PRO 6000 Blackwell GPUs. We benchmark the spectral scoring function in isolation using synthetic inputs, sweeping six channel sizes $( C \in$ {768, 1024, 1280, 2048, 3072, 4096}) and four spatial resolutions $( N ~ \in ~ \{ 1 0 2 4 , 1 6 0 0 , 2 3 0 4 , 4 0 9 6 \} )$ following the patch size=14 ViT convention. As shown in Fig. 8, latency grows approximately linearly with C at each resolution, consistent with the O(C · N log N ) analysis, and remains in the low-millisecond range across all tested configurations.

Z-Score in CLSE-Hybrid. In CLSE-Hybrid, the per-token evolution intensity $\delta _ { i }$ (defined in the main paper) is Z-score normalized across tokens before being used as a gating factor:

$$
\hat {\delta} _ {i} = \frac {\delta_ {i} - \mu_ {\delta}}{\sigma_ {\delta} + \epsilon}, \qquad e _ {i} = \sigma \Big (\hat {\delta} _ {i} / \tau \Big),\tag{22}
$$

where τ is a temperature parameter. This standardization makes the gate robust to the absolute scale of spectral intensities, which varies across images and models. As shown in Table 7, omitting Z-score consistently degrades performance, and $\tau = 0 . 1$ achieves the best results by producing a sharp yet smooth discrimination between high- and lowevolution tokens.

Pruning Layer Analysis. Table 8 reports the effect of varying the reference layer L and pruning layer K while retaining 192 visual tokens (↓66.7%). In early layers, visual tokens still preserve their original spatial diversity—semantically active regions (edges, fine structures) evolve rapidly across layers while background tokens remain spectrally static, yielding a highly discriminative CLSE signal; pruning here also maximizes latency savings. We therefore recommend $K \in \{ 1 , 3 \}$ as the sweet spot, balancing accuracy and efficiency. Mid-range pruning (e.g., $K = 1 0 )$ suffers from two compounding issues: repeated layer-wise mixing has already reduced the spectral diversity among visual tokens, weakening the CLSE signal, while most computation has already been incurred. At very deep layers $( K \ge 1 5 )$ , visual tokens have become mutually homogenized through extensive layer-wise abstraction—their representations are nearly interchangeable—so accuracy recovers, but all latency benefit is lost. Comparing reference choices at the same $K ,$ , the adjacent-layer reference $( L = K - 1 )$ is generally competitive with or superior to the global reference $( L = 0 )$ , suggesting that local spectral change is a reliable signal. Fig. 9 further visualizes the performance–latency trade-off across pruning layers, confirming that early-layer pruning (layers 1 and 3) achieves the best balance between efficiency and accuracy.

## C.2. Results on LLaVA-1.5-13B

To validate generalization to larger model scales, we evaluate CLSE on LLaVA-1.5-13B [42] and compare against representative baselines under three compression ratios. As shown in Table 9, CLSE achieves the highest average performance retention across all settings, attaining 99.1% at ↓66.7% and maintaining an 8.0 percentage-point lead over the next best method at the aggressive ↓88.9% ratio, demonstrating that spectral evolution scoring remains effective as

![](images/59da67413a6f37f5a2377dbfba15b42a476765adc20834e0265944092f24554e.jpg)  
(a) MME

![](images/95021ab84e3d2cd2eaeaedf44c4d77abea96ebe9ba9687f0df3226bdc69ed82e.jpg)  
(b) POPE  
Figure 9. Influence of Pruning Layer on Performance. We evaluated five pruning ratios per layer with CLSE-Hybrid. While pruning deeper layers maintains performance near or even above the Vanilla, the latency reduction remains marginal. Conversely, pruning shallower layers (e.g., layers 1 and 3) yields competitive results and, notably, outperforms the Vanilla on the MME dataset, offering a superior trade-off between efficiency and accuracy.

Table 8. Ablation study on CLSE Layer Selection. L and K denote the reference and scoring layers used to compute the cross-layer spectral evolution signal; pruning is applied at layer K. All settings retain 192 visual tokens (↓66.7%). Our default $( L = 0 , K = 1 )$ achieves competitive performance with the lowest evaluation time. Mid-range pruning (e.g., $K = 1 0 )$ causes noticeable degradation, while the adjacent-layer reference $( L = K - 1 )$ is generally competitive with or superior to the global reference $( L { = } 0 )$ at the same K.

<table><tr><td>Ref. Layer (L)</td><td>Prune Layer (K)</td><td>GQA ↑</td><td>MMB ↑</td><td>MME ↑</td><td>POPE (F1) ↑</td><td>Time (s) ↓</td></tr><tr><td>LLaVA-1.5-7B (Full Tokens)</td><td>-</td><td>61.9</td><td>64.7</td><td>1862</td><td>85.9</td><td>2348</td></tr><tr><td colspan="7">CLSE Pruning (Retain 192 Tokens, ↓66.7%)</td></tr><tr><td colspan="7">Pruning at Layer 1</td></tr><tr><td>0</td><td>1</td><td>61.1</td><td>62.6</td><td>1814</td><td>83.7</td><td>1691</td></tr><tr><td colspan="7">Pruning at Layer 3</td></tr><tr><td>0</td><td>3</td><td>58.8</td><td>62.2</td><td>1758</td><td>81.9</td><td>1732</td></tr><tr><td>2</td><td>3</td><td>60.4</td><td>62.5</td><td>1811</td><td>83.9</td><td>1738</td></tr><tr><td colspan="7">Pruning at Layer 5</td></tr><tr><td>0</td><td>5</td><td>56.4</td><td>61.1</td><td>1697</td><td>74.2</td><td>1756</td></tr><tr><td>4</td><td>5</td><td>54.7</td><td>59.5</td><td>1653</td><td>67.4</td><td>1746</td></tr><tr><td colspan="7">Pruning at Layer 10</td></tr><tr><td>0</td><td>10</td><td>49.4</td><td>58.0</td><td>1480</td><td>42.7</td><td>1927</td></tr><tr><td>9</td><td>10</td><td>54.3</td><td>59.3</td><td>1637</td><td>65.4</td><td>1932</td></tr><tr><td colspan="7">Pruning at Layer 15</td></tr><tr><td>0</td><td>15</td><td>57.6</td><td>64.0</td><td>1845</td><td>85.7</td><td>2001</td></tr><tr><td>14</td><td>15</td><td>60.8</td><td>64.1</td><td>1850</td><td>85.8</td><td>2007</td></tr><tr><td colspan="7">Pruning at Layer 20</td></tr><tr><td>0</td><td>20</td><td>61.6</td><td>64.3</td><td>1857</td><td>86.0</td><td>2066</td></tr><tr><td>19</td><td>20</td><td>61.0</td><td>64.2</td><td>1853</td><td>86.0</td><td>2075</td></tr></table>

Table 9. Comparative experiments on LLaVA-1.5-13B.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MMB-CN</td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA^{Text}$ </td><td>Avg.</td></tr><tr><td>LLaVA-1.5-13B</td><td colspan="8">Upper Bound, 576 Tokens (100%)</td></tr><tr><td>Vanilla</td><td>63.3</td><td>68.5</td><td>62.3</td><td>1816</td><td>85.9</td><td>74.9</td><td>61.2</td><td>100%</td></tr><tr><td>LLaVA-1.5-13B</td><td colspan="8">Retain 192 Tokens (↓66.7%)</td></tr><tr><td>FastV (ECCV24)</td><td>59.1</td><td>54.0</td><td>51.2</td><td>1641</td><td>82.3</td><td>56.4</td><td>51.6</td><td>86.4%</td></tr><tr><td>SparseVLM (ICML25)</td><td>58.7</td><td>67.4</td><td>61.0</td><td>1768</td><td>82.2</td><td>73.1</td><td>45.4</td><td>91.0%</td></tr><tr><td>PDrop (CVPR25)</td><td>59.6</td><td>67.5</td><td>62.4</td><td>1786</td><td>85.0</td><td>74.9</td><td>59.5</td><td>98.2%</td></tr><tr><td>FiCoCo-V (AAAI26)</td><td>58.1</td><td>65.2</td><td>58.3</td><td>1791</td><td>81.4</td><td>75.1</td><td>56.2</td><td>95.1%</td></tr><tr><td>CLSE (Ours)</td><td>62.4</td><td>67.4</td><td>60.8</td><td>1826</td><td>85.0</td><td>74.7</td><td>60.0</td><td>99.1%</td></tr><tr><td>LLaVA-1.5-13B</td><td colspan="8">Retain 128 Tokens (↓77.8%)</td></tr><tr><td>FastV (ECCV24)</td><td>57.7</td><td>57.9</td><td>48.8</td><td>1673</td><td>79.3</td><td>57.0</td><td>56.0</td><td>87.2%</td></tr><tr><td>SparseVLM (ICML25)</td><td>57.9</td><td>65.8</td><td>55.8</td><td>1774</td><td>81.1</td><td>69.9</td><td>49.9</td><td>94.1%</td></tr><tr><td>PDrop (CVPR25)</td><td>57.2</td><td>65</td><td>59.2</td><td>1744</td><td>82.3</td><td>75.4</td><td>58.3</td><td>96.5%</td></tr><tr><td>FiCoCo-V (AAAI26)</td><td>56.8</td><td>62.1</td><td>55.9</td><td>1710</td><td>77.3</td><td>75.0</td><td>54.8</td><td>92.0%</td></tr><tr><td>CLSE (Ours)</td><td>61.6</td><td>66.5</td><td>59.5</td><td>1785</td><td>82.9</td><td>75.0</td><td>58.3</td><td>98.2%</td></tr><tr><td>LLaVA-1.5-13B</td><td colspan="8">Retain 64 Tokens (↓88.9%)</td></tr><tr><td>FastV (ECCV24)</td><td>53.7</td><td>50.9</td><td>42.1</td><td>1567</td><td>69.3</td><td>56.8</td><td>47.1</td><td>81.0%</td></tr><tr><td>SparseVLM (ICML25)</td><td>50.6</td><td>61.3</td><td>54.8</td><td>1402</td><td>65.0</td><td>69.0</td><td>22.7</td><td>78.3%</td></tr><tr><td>PDrop (CVPR25)</td><td>50.7</td><td>59.2</td><td>50.7</td><td>1441</td><td>62.6</td><td>74.2</td><td>52.9</td><td>84.6%</td></tr><tr><td>FiCoCo-V (AAAI26)</td><td>54.4</td><td>55.8</td><td>49.4</td><td>1615</td><td>70.9</td><td>74.2</td><td>52.5</td><td>86.1%</td></tr><tr><td>CLSE (Ours)</td><td>58.6</td><td>63.3</td><td>56.7</td><td>1728</td><td>75.9</td><td>73.6</td><td>54.5</td><td>94.1%</td></tr></table>

model capacity increases.

## C.3. Results on InternVL-2.5-8B

We additionally evaluate CLSE on InternVL-2.5-8B [8] to assess cross-architecture generalization. InternVL encodes high-resolution images via dynamic tiling into $n _ { \mathrm { s u b } }$ sub-tiles $( 1 6 \times 1 6$ tokens each) plus a 256-token thumbnail. $\mathrm { C L S E _ { \mathrm { l o c a l } } }$ applies 2D FFT independently per sub-tile, while $\mathrm { C L S E _ { g l o b a l } }$ reassembles sub-tiles into their original spatial layout before a single global FFT. In both variants, sub-tiles and the thumbnail are scored with separate normalization statistics—since the thumbnail’s low-frequency characteristics would otherwise be systematically suppressed under a global mean and standard deviation—and a single global Top-K is then applied across all visual tokens to ensure fair cross-partition selection. As shown in Table 10, both variants consistently outperform FastV and achieve comparable accuracy to each other, confirming that the CLSE signal is robust to how spatial tokens are partitioned for FFT. At ↓88.9%, both achieve 96.2% average retention vs. 89.4% for FastV.

Table 10. Comparative experiments on InternVL-2.5-8B. $\mathrm { C L S E _ { \mathrm { l o c a l } } }$ applies 2D FFT independently within each 16×16 subtile, while $\mathrm { C L S E _ { g l o b a l } }$ reassembles sub-tiles into the full spatial layout before applying a single 2D FFT, capturing cross-tile global frequency structure. The thumbnail is scored separately in both variants.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MMB-CN</td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA^{Text}$ </td><td>Avg.</td></tr><tr><td>InternVL-2.5-8B</td><td colspan="8">Upper Bound, Full Tokens (100%)</td></tr><tr><td>Vanilla</td><td>63.1</td><td>84.7</td><td>82.6</td><td>2332</td><td>90.5</td><td>98.0</td><td>76.8</td><td>100%</td></tr><tr><td>InternVL-2.5-8B</td><td colspan="8">Token Reduction (↓ 66.7%)</td></tr><tr><td>FastV (ECCV24)</td><td>61.2</td><td>83.8</td><td>82.4</td><td>2252</td><td>90.4</td><td>96.4</td><td>75.5</td><td>98.4%</td></tr><tr><td>CLSE local</td><td>61.8</td><td>84.1</td><td>82.5</td><td>2312</td><td>90.5</td><td>96.6</td><td>75.9</td><td>99.1%</td></tr><tr><td>CLSE global</td><td>61.9</td><td>83.7</td><td>82.6</td><td>2325</td><td>90.5</td><td>96.5</td><td>75.9</td><td>99.1%</td></tr><tr><td>InternVL-2.5-8B</td><td colspan="8">Token Reduction (↓ 77.8%)</td></tr><tr><td>FastV (ECCV24)</td><td>59.6</td><td>81.1</td><td>80.5</td><td>2208</td><td>89.4</td><td>95.7</td><td>73.7</td><td>96.4%</td></tr><tr><td>CLSE local</td><td>60.7</td><td>83.6</td><td>82.0</td><td>2291</td><td>90.7</td><td>96.4</td><td>74.9</td><td>98.4%</td></tr><tr><td>CLSE global</td><td>60.9</td><td>83.5</td><td>81.8</td><td>2293</td><td>90.6</td><td>96.4</td><td>75.0</td><td>98.4%</td></tr><tr><td>InternVL-2.5-8B</td><td colspan="8">Token Reduction (↓ 88.9%)</td></tr><tr><td>FastV (ECCV24)</td><td>54.1</td><td>73.9</td><td>74.4</td><td>2052</td><td>85.2</td><td>92.5</td><td>66.4</td><td>89.4%</td></tr><tr><td>CLSE local</td><td>59.4</td><td>81.4</td><td>80.3</td><td>2201</td><td>89.7</td><td>95.8</td><td>72.5</td><td>96.2%</td></tr><tr><td>CLSE global</td><td>59.6</td><td>81.7</td><td>80.5</td><td>2204</td><td>89.6</td><td>95.7</td><td>71.9</td><td>96.2%</td></tr></table>

## C.4. Results on Qwen2-VL-72B

We further evaluate CLSE on Qwen2-VL-72B [60] with 4- bit weight quantization to assess scalability to very large models. As shown in Table 11, CLSE consistently outperforms both FastV and PDrop across all three compression ratios, with a performance retention gap over FastV that widens monotonically from 2.0 to 3.7 to 7.8 percentage points as pruning becomes more aggressive (↓66.7% → ↓88.9%). This demonstrates that cross-layer spectral evolution scoring remains effective even at 72B scale under quantized weights.

## D. Theoretical Analysis

We next show that the proposed pruning rule is not merely heuristic. Under a fixed token budget, selecting tokens by the proposed cross-layer spectral evolution score maximizes the preserved amount of informative structural dynamics. Moreover, under a mild downstream smoothness assumption, the same rule minimizes an upper bound on the perturbation induced by pruning.

Setup. Let $\mathcal { Z } ^ { ( \ell ) } ~ \in ~ \mathbb { R } ^ { N }$ denote the calibrated evolution scores defined in Eq. (9), where $N = H \cdot W$ is the number of visual tokens. For notational simplicity, we define the nonnegative token score as:

Table 11. Comparative experiments on Qwen2-VL-72B.

<table><tr><td>Method</td><td>GQA</td><td>MMB</td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA^{Text}$ </td><td>Avg.</td></tr><tr><td>Qwen2-VL-72B</td><td colspan="7">Upper Bound, Full Tokens (100%)</td></tr><tr><td>Vanilla</td><td>64.8</td><td>85.0</td><td>2476</td><td>87.5</td><td>92.5</td><td>82.0</td><td>100%</td></tr><tr><td>Qwen2-VL-72B</td><td colspan="7">Token Reduction (↓ 66.7%)</td></tr><tr><td>FastV (ECCV24)</td><td>59.6</td><td>79.0</td><td>2314</td><td>82.8</td><td>89.9</td><td>79.1</td><td>94.4%</td></tr><tr><td>PDrop (CVPR25)</td><td>60.9</td><td>81.3</td><td>2273</td><td>84.9</td><td>90.5</td><td>78.9</td><td>95.4%</td></tr><tr><td>CLSE (Ours)</td><td>61.4</td><td>82.4</td><td>2340</td><td>85.6</td><td>91.3</td><td>78.5</td><td>96.4%</td></tr><tr><td>Qwen2-VL-72B</td><td colspan="7">Token Reduction (↓ 77.8%)</td></tr><tr><td>FastV (ECCV24)</td><td>57.2</td><td>76.9</td><td>2173</td><td>78.0</td><td>88.7</td><td>76.6</td><td>90.8%</td></tr><tr><td>PDrop (CVPR25)</td><td>58.5</td><td>77.8</td><td>2178</td><td>82.0</td><td>89.2</td><td>76.7</td><td>92.2%</td></tr><tr><td>CLSE (Ours)</td><td>60.1</td><td>79.3</td><td>2309</td><td>84.7</td><td>90.6</td><td>76.0</td><td>94.5%</td></tr><tr><td>Qwen2-VL-72B</td><td colspan="7">Token Reduction (↓ 88.9%)</td></tr><tr><td>FastV (ECCV24)</td><td>53.0</td><td>66.0</td><td>1869</td><td>71.3</td><td>86.1</td><td>68.8</td><td>82.2%</td></tr><tr><td>PDrop (CVPR25)</td><td>54.6</td><td>67.5</td><td>2057</td><td>75.6</td><td>87.0</td><td>70.0</td><td>85.4%</td></tr><tr><td>CLSE (Ours)</td><td>57.2</td><td>75.3</td><td>2143</td><td>81.4</td><td>88.6</td><td>72.1</td><td>90.0%</td></tr></table>

$$
s _ {i} := \mathcal {I} _ {i} ^ {(\ell)}, \qquad i \in \{1, \dots , N \}.\tag{23}
$$

For any retained token set $\Omega \subseteq \{ 1 , \ldots , N \}$ with $| \Omega | = K ,$ let $P _ { \Omega } ( \mathbf { X } ^ { \ell } )$ denote the pruned token sequence that keeps tokens indexed by Ω and discards the others. We further define the retained spectral evolution mass as:

$$
\mathcal {Q} (\Omega) := \sum_ {i \in \Omega} s _ {i},\tag{24}
$$

which measures the total amount of cross-layer structural evolution preserved after pruning.

Proposition 1 (Optimal preservation under a cardinality budget). Let $\Omega _ { K } ^ { \star }$ be the index set of the top-K entries of $\{ s _ { i } \} _ { i = 1 } ^ { N }$ . Then

$$
\Omega_ {K} ^ {\star} \in \arg \max _ {| \Omega | = K} \mathcal {Q} (\Omega).\tag{25}
$$

Equivalently,

$$
\Omega_ {K} ^ {\star} \in \arg \min _ {| \Omega | = K} \sum_ {i \notin \Omega} s _ {i}.\tag{26}
$$

Proposition 1 states that, among all K-token subsets, the proposed top-K rule preserves the maximum amount of cross-layer spectral evolution, or equivalently discards the minimum amount of structural evolution mass.

Why spectral evolution matters. Unlike magnitudebased criteria, our score is computed from high-pass filtered residual maps:

$$
\mathbf {S} _ {\ell} = \mathrm{Avg} _ {d} \big (\big | \mathcal {F} ^ {- 1} \big (\text { ifftshift } (\text { fftshift } (\mathcal {F} (\mathbf {X} _ {2 D} ^ {\ell})) \odot \mathbf {M}) \big) \big | \big)\tag{27}
$$

and therefore quantifies band-limited structural activity rather than raw embedding norm. Large $s _ { i }$ indicates that token i undergoes substantial redistribution of local highfrequency content across adjacent layers, which is precisely the regime where text-conditioned visual reasoning actively reshapes token representations. Thus, preserving large-s<sub>i</sub> tokens preserves the most dynamically updated local structures. To relate this surrogate quantity to downstream fidelity, we consider the remaining decoder after pruning.

Assumption 1 (Downstream smoothness). Let g(·) denote the subsequent decoder mapping from the pruned visualtoken sequence to the final hidden representation or output logits. We assume g is locally Lipschitz:

$$
\left\| g (\mathbf {U}) - g (\mathbf {V}) \right\| _ {2} \leq L \left\| \mathbf {U} - \mathbf {V} \right\| _ {F},\tag{28}
$$

for some constant $L > 0$ and all U, V in a local neighborhood of interest.

Assumption 2 (Saliency alignment). We assume that the downstream influence of each token is upper bounded by its spectral evolution score. Formally, let $\mathbf { J } _ { i }$ denote the Jacobian block of g with respect to token $\mathbf { x } _ { i } ^ { \ell }$ . Then there exists $\alpha > 0$ such that:

$$
\| \mathbf {J} _ {i} \mathbf {x} _ {i} ^ {\ell} \| _ {2} \leq \alpha s _ {i}, \quad \forall i \in \{1, \dots , N \}.\tag{29}
$$

This assumption formalizes the inductive bias of our method: tokens with weak cross-layer spectral evolution have limited effect on the downstream computation, whereas tokens with strong evolution are more influential.

Theorem 1 (Top-K spectral evolution minimizes an output perturbation bound). Under Assumptions 1–2, for any retained set Ω with $| \Omega | = K$ , the pruning-induced perturbation satisfies:

$$
\| g (\mathbf {X} ^ {\ell}) - g (P _ {\Omega} (\mathbf {X} ^ {\ell})) \| _ {2} \leq \alpha \sum_ {i \notin \Omega} s _ {i}.\tag{30}
$$

Consequently, the top-K set $\Omega _ { K } ^ { \star }$ from Proposition 1 minimizes the above upper bound among all subsets of size $K .$

$$
\| g (\mathbf {X} ^ {\ell}) - g (P _ {\Omega_ {K} ^ {\star}} (\mathbf {X} ^ {\ell})) \| _ {2} \leq \alpha \min _ {| \Omega | = K} \sum_ {i \notin \Omega} s _ {i}.\tag{31}
$$

Proof. The optimality of $\Omega _ { K } ^ { \star }$ in Proposition 1 follows directly from sorting nonnegative scores: among all subsets of cardinality $K ,$ , the sum of retained scores is maximized by selecting the K largest entries.

We now prove the perturbation bound. Let $\Delta _ { \Omega } : = \mathbf { X } ^ { \ell } -$ $P _ { \Omega } ( \mathbf { X } ^ { \ell } )$ denote the perturbation introduced by pruning. By a first-order expansion of g around $\mathbf { X } ^ { \ell }$ , we have

$$
g (\mathbf {X} ^ {\ell}) - g (P _ {\Omega} (\mathbf {X} ^ {\ell})) = g (\mathbf {X} ^ {\ell}) - g (\mathbf {X} ^ {\ell} - \Delta_ {\Omega}) \approx \sum_ {i \notin \Omega} \mathbf {J} _ {i} \mathbf {x} _ {i} ^ {\ell}.\tag{32}
$$

![](images/71406762f43f64a2abd01af33f7bcaaf814c87c1f2f4a520484d2f61f91c0fbb.jpg)  
Figure 10. Multi-turn dialogue example on LLaVA-1.5-7B. The user asks about electronic devices on a desk, followed by a request for a detailed workspace description.

Applying the triangle inequality yields

$$
\| g (\mathbf {X} ^ {\ell}) - g (P _ {\Omega} (\mathbf {X} ^ {\ell})) \| _ {2} \leq \sum_ {i \notin \Omega} \| \mathbf {J} _ {i} \mathbf {x} _ {i} ^ {\ell} \| _ {2}.\tag{33}
$$

By Assumption 2,

$$
\| g (\mathbf {X} ^ {\ell}) - g (P _ {\Omega} (\mathbf {X} ^ {\ell})) \| _ {2} \leq \alpha \sum_ {i \notin \Omega} s _ {i}.\tag{34}
$$

Finally, Proposition 1 implies that the top- $\mathbf { \nabla } . K$ subset minimizes the discarded score mass $\textstyle \sum _ { i \not \in \Omega } s _ { i } ,$ , and therefore minimizes the above upper bound as well. □

Interpretation. Theorem 1 provides a principled justification for our pruning rule. The proposed criterion does not preserve tokens with merely large activations; instead, it preserves tokens with the strongest relative cross-layer structural evolution after low-frequency suppression. Under a fixed budget, this rule retains the maximum amount of informative spectral dynamics and minimizes a certified upper bound on pruning-induced distortion. This explains why CLSE is particularly effective in early decoding, where pruning must be aggressive yet reliable.

Remark. Our guarantee is stated with respect to a structure-preserving surrogate induced by spectral evolution, rather than a task-specific semantic objective. This is appropriate for training-free pruning, where explicit task supervision is unavailable at pruning time. Empirically, the strong alignment between spectral evolution and downstream utility is validated by the ablations in Sec. 4.3.

![](images/d4a231f3d3800c36a410b25c82125c80bc329043a41534811fda31a63aab8c57.jpg)  
Figure 11. Multi-turn dialogue example on LLaVA-1.5-7B. The user asks about a sign in a pub image, followed by a question about the type of place and items on display.

## E. Qualitative Multi-Turn Dialogue Examples

To illustrate the impact of token pruning on response quality in open-ended dialogue, we compare three settings on LLaVA-1.5-7B: (1) Base, which processes all 576 visual tokens with no pruning; (2) FastV [6], which prunes down to 192 visual tokens; and (3) CLSE, which also retains 192 tokens using our spectral evolution-guided criterion.

Figures 10 and 11 each show a two-turn conversation over a single image. The first turn asks a short, factual question, while the second turn requires a more detailed description of the scene. Despite using only 192/576 ≈ 33.4% of the visual tokens, CLSE consistently produces responses that match the Base model in accuracy and detail. FastV, by contrast, occasionally drops or misidentifies salient objects (e.g., misses the large monitor in Fig. 10; misreads the sign as “Drink” instead of “Pub” in Fig. 11), highlighting the benefit of retaining the most informative tokens as identified by spectral evolution.