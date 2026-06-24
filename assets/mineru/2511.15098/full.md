# A Comprehensive Study on Visual Token Redundancy for Discrete Diffusion-based Multimodal Large Language Models

Duo Li<sup>1</sup>\* Zuhao Yang<sup>1</sup>\* Xiaoqin Zhang<sup>2</sup> Ling Shao<sup>3</sup> Shijian Lu<sup>1†</sup> <sup>1</sup>CCDS, NTU, Singapore <sup>2</sup>CCST, ZJUT, China <sup>3</sup>Terminus AI Lab, UCAS, China

Email Contact: {duo001, yang0756}@e.ntu.edu.sg

## Abstract

Discrete diffusion-based multimodal large language models (dMLLMs) have emerged as a promising alternative to autoregressive MLLMs thanks to their advantages in parallel decoding and bidirectional context modeling, but most existing dMLLMs incur significant computational overhead during inference due to the full-sequence attention computation in each denoising step. Pioneering studies attempt to resolve this issue from a modality-agnostic perspective via key–value cache optimization or efficient sampling but most of them overlook modality-specific visual token redundancy. In this work, we conduct a comprehensive study on how visual token redundancy evolves with different dMLLM architectures and tasks and how visual token pruning affects dMLLM responses and efficiency. Specifically, our study reveals that visual redundancy emerges only in from-scratch dMLLMs while handling long-answer tasks. In addition, we validate that visual token pruning introduces non-negligible information loss in dMLLMs and only from-scratch dM-LLMs can recover the lost information progressively during late denoising steps. Furthermore, our study shows that layer-skipping is promising for accelerating AR-to-diffusion dMLLMs, whereas progressive or late-step pruning is more effective for from-scratch dMLLMs. Overall, this work offers a new perspective on efficiency optimization for dM-LLMs, greatly advancing their applicability across various multimodal understanding tasks. The code is available at: https://github.com/Yrdal3910/dMLLM-Visual-Token-Redundancy-Analysis.

## 1. Introduction

Discrete diffusion-based large language models (dLLMs) have recently emerged and advanced rapidly thanks to their promising potential in both efficiency and performance [17]. Most existing dLLMs are trained with two representative approaches: 1) from-scratch diffusion training [40, 59, 67] that trains standard Transformers [47] from scratch to directly learn the denoising-based text generation process and 2) AR-to-diffusion adaptation [14, 46, 60] that converts pretrained autoregressive large language models (LLMs) into diffusion-based generators by further training them with masking-based diffusion objectives. Leveraging the advances in dLLMs, several discrete diffusionbased multimodal LLMs (dMLLMs) [25, 61, 62] have been developed and achieved competitive performance as compared with autoregressive multimodal large language models (MLLMs) of similar scales.

However, most open-source dMLLMs involve intensive computation during inference, leading to a much longer inference process as compared with autoregressive MLLMs [36]. Several studies attempt to address this issue via different optimization strategies such as key–value cache optimization [33, 45, 51], efficient token sampling [15, 24, 49, 51], and variable-length generation [18, 23, 53]. However, most of these studies tackle the problem from a modalityagnostic perspective without considering the redundancy of visual tokens. Given that visual token redundancy and visual token pruning have been widely studied for autoregressive MLLMs [27, 42, 43], two straightforward questions naturally come to our mind: 1) whether visual token redundancy exists in prevalent dMLLMs, and if so, 2) how visual token pruning affects the inference speed and accuracy of prevalent dMLLMs. A direct extension to both questions is how those pruning approaches that have achieved great success for autoregressive MLLMs perform in dMLLMs.

We perform a systematic study on whether visual token redundancy exists, how it affects dMLLMs, and how it could guide visual token pruning in dMLLMs. The study focuses on two prevalent dMLLM backbones, namely, from-scratch dMLLMs (e.g., LLaDA-V [61]) and AR-todiffusion dMLLMs (e.g., LaViDa-Dream [25]), as well as six representative token pruning techniques that achieve impressive performance in autoregressive MLLMs. We first apply visual token pruning to the initial denoising steps, and the substantial performance drops in both dMLLM backbones (see Figure 1 left) indicate the essential role of visual tokens at the initial denoising stage. On the other hand, the two backbones exhibit very divergent behaviors while applying visual token pruning to the following denoising steps (see Figure 1 right). Specifically, AR-todiffusion dMLLMs still suffer from severe performance drops for both short-answer and long-answer tasks, while from-scratch dMLLMs suffer from a similar performance drop for short-answer tasks but achieve only minimal performance drops for long-answer tasks. In addition, we examine why the two dMLLM backbones perform very differently for long-answer tasks. Our study shows that visual token pruning leads to much more information loss in dMLLMs than in MLLMs, while from-scratch dMLLMs can better restore the pruning-induced information loss via the subsequent iterative bidirectional denoising and refinement, explaining their minimal performance drops. Further, our study shows that both attention scores during inference and step-wise output logits can be effective indicators to guide visual token pruning in dMLLMs.

![](images/cbdbb33e0e77e53f7090eb48ac568ecfc5198ce971eb14f65052e974fb8bf467.jpg)

![](images/37eb37004ca25e5098a9b24a8955ed83587fb79dcc4c7c869f51c80f32f940b6.jpg)  
Figure 1. LEFT: Visual token pruning affects dMLLMs much more than autoregressive MLLMs. Three representative pruning methods [1, 5, 64] are applied to LLaVA-NeXT [30] (solid lines), LaViDa-Dream [25], and LLaDA-V [61] (dotted lines). RIGHT: Performance vs. pruning start step. Unlike AR-to-diffusion LaViDa-Dream, the from-scratch LLaDA-V consistently recovers pruning-induced information loss, achieving much higher accuracy when pruning starts at different denoising steps (projected to 0–100 for visualization).

The contributions of this work are three-fold. First, we conduct the first comprehensive study of visual token redundancy in dMLLMs, revealing that AR-to-diffusion dM-LLMs suffer consistent and clear performance drops under pruning while from-scratch dMLLMs remain resilient on long-answer tasks. Second, we show that pruning causes much greater information loss in dMLLMs than in MLLMs, and that from-scratch dMLLMs can mitigate such loss via iterative bidirectional denoising whereas AR-to-diffusion dMLLMs lack this restoration ability. Third, we offer practical guidance: layer-skipping suits AR-to-diffusion dM-LLMs, while progressive or late-step pruning works better for from-scratch architectures, with both attention scores and logit signals serving as reliable pruning indicators.

## 2. Related Work

## 2.1. Visual Token Compression for MLLMs

MLLMs typically encode visual inputs into a large set of visual tokens. These visual tokens are inherently sparse in representing effective information compared to textual tokens [43, 56], thereby incurring considerable yet unnecessary computational cost. To combat this inefficiency, existing visual token compression methods attempt to prune redundant visual tokens or skip certain computational stages. These approaches can be broadly categorized into four groups: 1) transformation-based compression, which weakens the spatial and temporal redundancy inherent in visual inputs by applying pixel unshuffling [3, 7, 13, 48], pooling [58, 65], and convolutional transformations [8, 9]; 2) similarity-based compression, which prunes or merges redundant tokens according to inter-token similarity or distance metrics, typically measured via cosine similarity [1, 4, 55]; 3) attention-based compression, which employs various attention scores as the importance metric to guide token pruning in either the vision encoder [31, 57] or the LLM decoder [5, 54]; and 4) query-based compression, which leverages query prompts to guide the visual token compression and can be further classified into token distillation [2, 66] and cross-modal selection strategies [41, 64].

## 2.2. Acceleration Techniques for dMLLMs

Unlike LLMs that generate tokens sequentially with cached context, dLLMs perform iterative denoising across numerous steps, each involving bidirectional full-sequence attention and softmax computations, which collectively lead to significant slowdown in inference. To overcome these bottlenecks, recent approaches have accelerated dLLMs from three perspectives: 1) Optimizing key-value (KV) cache utilization—reusing KV caches across denoising steps [33, 51] and reducing cached representations within each step [6, 45]; 2) Designing efficient sampling algorithms—accelerating inference via dynamic and parallel decoding [15, 24, 49, 51]; and 3) Adopting variable-length generation—enabling the model to dynamically adjust the output sequence length during denoising [18, 23, 53]. While these methods can be naturally inherited by dMLLMs, a modality-specific yet overlooked factor is visual token redundancy. Our work aims to provide the first systematic study of this underexplored aspect and discuss its potential optimization directions.

## 3. From Autoregression to Diffusion: Modeling and Complexity

## 3.1. Model Foundation

Foundation of MLLMs. MLLMs adopt autoregressive modeling to define the model distribution over the output sequence [52]. Given a multimodal context $x \ { \mathrm { ( e . g . } }$ ., image–text pair) and target sequence $y = ( y ^ { 1 } , \ldots , y ^ { L } )$ , the model factorizes the conditional likelihood via the chain rule of probability:

$$
p _ {\theta} (y | x) = \prod_ {t = 1} ^ {L} p _ {\theta} (y ^ {t} \mid x, y ^ {<   t}),\tag{1}
$$

where $y ^ { < t } = ( y ^ { 1 } , \dots , y ^ { t - 1 } )$ , and $L$ is the total length of the target sequence. The parameters $\theta$ are optimized by maximum likelihood estimation:

$$
\mathcal {L} _ {\mathrm{AR}} (\theta) = - \mathbb {E} _ {(x, y) \sim p _ {\mathrm{data}}} \sum_ {t = 1} ^ {L} \log p _ {\theta} (y ^ {t} \mid x, y ^ {<   t}).\tag{2}
$$

This formulation enforces a causal and unidirectional dependency, where each token prediction conditions on all preceding tokens and the multimodal input context.

Foundation of dMLLMs. In contrast, dMLLMs parameterize a denoising Markov chain over discrete latent sequences [40, 60]. Let $y _ { 0 }$ denote the clean output sequence and $y _ { t }$ represent its masked version at diffusion step $t \in$ $[ 1 , T ]$ . The forward noising process $q _ { t \mid 0 } ( y _ { t } \mid y _ { 0 } )$ gradually injects noise or masks into $y _ { 0 } .$ , producing a sequence of intermediate noisy states $\{ y _ { t } \} _ { t = 1 } ^ { \bar { T } }$ . The reverse denoising process is modeled by a parameterized consistency kernel:

$$
p _ {\theta} (y _ {t - 1} \mid y _ {t}, x),\tag{3}
$$

which approximates the ideal reverse kernel that satisfies Chapman–Kolmogorov consistency [19]. The overall model distribution is given by marginalizing over all intermediate states:

$$
p _ {\theta} (y _ {0} \mid x) = \sum_ {y _ {1: T}} p (y _ {T}) \prod_ {t = 1} ^ {T} p _ {\theta} (y _ {t - 1} \mid y _ {t}, x),\tag{4}
$$

where $p ( y _ { T } )$ is the prior distribution $( \mathrm { e . g . , a }$ fully masked state where $p ( y _ { T } ) = \delta ( y _ { T } = M ) )$ ). The parameters $\theta$ are optimized by minimizing a negative log-likelihood upper bound:

$$
\mathcal {L} _ {\text { Diff }} (\theta) = - \mathbb {E} _ {y _ {0}, t, y _ {t}} \left[ w (t) \sum_ {i = 1} ^ {L} \mathbf {1} [ y _ {t} ^ {i} = M ] \log p _ {\theta} (y _ {0} ^ {i} \mid y _ {t}, x) \right],\tag{5}
$$

where M denotes the masked token, and $w ( t )$ is a weighting function derived from the diffusion noise schedule. Unlike causal factorization (Eq. (1)), Eq. (5) defines a non-causal and iterative denoising process that jointly models all tokens in each step, enabling bidirectional contextual reasoning across modalities.

## 3.2. Computational Cost Analysis

Following prior studies [5, 22, 50], the floating-point operations (FLOPs) of Transformer-based MLLMs can be formulated as:

$$
\mathrm{FLOPs} _ {\text {prefilling}} = T \times (4 n d ^ {2} + 2 n ^ {2} d + 2 n d m),\tag{6}
$$

$$
\begin{array}{c} \text {FLOPs} _ {\text {decoding}} = T \sum_ {t = 1} ^ {L} (4 d ^ {2} + 2 d (n + t - 1) + 2 d m) \\ = T (4 L d ^ {2} + 2 L d m + d L (2 n + L - 1)), \end{array}\tag{7}
$$

where $T$ denotes the number of Transformer layers; n is the input sequence length; $L$ stands for the output length; d represents the hidden size; m denotes the intermediate dimension of feed-forward networks.

For dMLLMs, since each denoising step performs bidirectional interactions among tokens, the total FLOPs can be defined as:

$$
\mathrm{FLOPs} _ {\text { decoding }} = T \sum_ {s = 1} ^ {S} \left(4 n _ {s} d ^ {2} + 2 n _ {s} ^ {2} d + 2 n _ {s} d m\right),\tag{8}
$$

<table><tr><td>Method</td><td>MME [11]</td><td>SQA [35]</td><td>GQA [16]</td><td>POPE [26]</td><td>MMB [32]</td><td>TVQA [44]</td><td>CQA [37]</td><td>MMMUP [63]</td><td>Avg.</td></tr><tr><td>LLaDA-V [61]</td><td> $1994.4_{5:47}$ </td><td> $89.8_{3:07}$ </td><td> $52.1_{22:25}$ </td><td> $87.3_{15:45}$ </td><td> $83.3_{7:24}$ </td><td> $31.7_{10:49}$ </td><td> $77.7_{10:08}$ </td><td> $18.7_{4:19}$ </td><td> $100.0\%^{79:44}$ </td></tr><tr><td colspan="10">Retention Ratio = 50%</td></tr><tr><td>ToMe (ICLR&#x27;23) [4]</td><td> $1918.7^{\uparrow 4.9\%}$ </td><td> $86.9^{\uparrow 3.7\%}$ </td><td> $50.0^{\uparrow 4.7\%}$ </td><td> $87.4^{\uparrow 6.1\%}$ </td><td> $79.3^{\uparrow 13.7\%}$ </td><td> $29.9^{\downarrow 0.9\%}$ </td><td> $53.2^{\downarrow 32.1\%}$ </td><td> $15.7^{\downarrow 8.9\%}$ </td><td> $91.3\%^{\downarrow 0.4\%}$ </td></tr><tr><td>FastV (ECCV&#x27;24) [5]</td><td> $1655.8^{\uparrow 15.3\%}$ </td><td> $78.3^{\downarrow 3.2\%}$ </td><td> $43.5^{\downarrow 0.7\%}$ </td><td> $82.1^{\downarrow 6.6\%}$ </td><td> $67.6^{\uparrow 7.2\%}$ </td><td> $25.6^{\downarrow 1.4\%}$ </td><td> $15.6^{\downarrow 26.5\%}$ </td><td> $12.1^{\uparrow 0.8\%}$ </td><td> $74.4\%^{\downarrow 3.4\%}$ </td></tr><tr><td>TRIM (COLING&#x27;25) [41]</td><td> $1590.4^{\uparrow 40.3\%}$ </td><td> $82.2^{\uparrow 22.5\%}$ </td><td> $42.8^{\uparrow 20.4\%}$ </td><td> $77.3^{\uparrow 35.9\%}$ </td><td> $75.9^{\uparrow 34.9\%}$ </td><td> $20.9^{\uparrow 42.1\%}$ </td><td> $32.7^{\uparrow 47.9\%}$ </td><td> $12.0^{\uparrow 25.9\%}$ </td><td> $75.6\%^{\uparrow 33.1\%}$ </td></tr><tr><td>SparseVLM (ICML&#x27;25) [64]</td><td> $1730.1^{\downarrow 9.5\%}$ </td><td> $80.6^{\downarrow 15.5\%}$ </td><td> $47.8^{\downarrow 20.5\%}$ </td><td> $83.1^{\downarrow 22.0\%}$ </td><td> $73.2^{\downarrow 5.9\%}$ </td><td> $23.1^{\downarrow 39.4\%}$ </td><td> $38.0^{\downarrow 72.0\%}$ </td><td> $14.5^{\downarrow 42.9\%}$ </td><td> $81.3\%^{\downarrow 28.8\%}$ </td></tr><tr><td>DivPrune (CVPR&#x27;25) [1]</td><td> $1793.7^{\uparrow 12.1\%}$ </td><td> $82.3^{\downarrow 10.7\%}$ </td><td> $48.3^{\downarrow 25.7\%}$ </td><td> $86.1^{\downarrow 6.3\%}$ </td><td> $74.6^{\downarrow 1.4\%}$ </td><td> $29.5^{\uparrow 5.1\%}$ </td><td> $64.8^{\uparrow 18.4\%}$ </td><td> $15.5^{\downarrow 1.5\%}$ </td><td> $90.2\%^{\downarrow 5.2\%}$ </td></tr><tr><td colspan="10">Retention Ratio = 25%</td></tr><tr><td>ToMe (ICLR&#x27;23) [4]</td><td> $1799.6^{\uparrow 20.2\%}$ </td><td> $84.6^{\uparrow 26.2\%}$ </td><td> $48.4^{\uparrow 28.0\%}$ </td><td> $87.1^{\uparrow 24.0\%}$ </td><td> $78.3^{\uparrow 27.5\%}$ </td><td> $28.8^{\uparrow 21.3\%}$ </td><td> $45.6^{\uparrow 7.2\%}$ </td><td> $14.3^{\downarrow 0.8\%}$ </td><td> $87.1\%^{\uparrow 21.4\%}$ </td></tr><tr><td>FastV (ECCV&#x27;24) [5]</td><td> $1374.1^{\uparrow 34.9\%}$ </td><td> $74.2^{\uparrow 15.0\%}$ </td><td> $39.1^{\uparrow 19.3\%}$ </td><td> $74.6^{\uparrow 17.9\%}$ </td><td> $58.2^{\uparrow 21.2\%}$ </td><td> $18.0^{\uparrow 25.9\%}$ </td><td> $10.8^{\uparrow 31.6\%}$ </td><td> $11.5^{\uparrow 23.2\%}$ </td><td> $64.2\%^{\uparrow 22.8\%}$ </td></tr><tr><td>TRIM (COLING&#x27;25) [41]</td><td> $1243.5^{\uparrow 47.0\%}$ </td><td> $78.2^{\uparrow 25.1\%}$ </td><td> $38.0^{\uparrow 31.7\%}$ </td><td> $68.0^{\uparrow 39.9\%}$ </td><td> $68.7^{\uparrow 39.4\%}$ </td><td> $16.2^{\uparrow 45.8\%}$ </td><td> $26.2^{\uparrow 53.3\%}$ </td><td> $12.3^{\uparrow 29.3\%}$ </td><td> $66.7\%^{\uparrow 39.4\%}$ </td></tr><tr><td>SparseVLM (ICML&#x27;25) [64]</td><td> $1437.1^{\uparrow 13.3\%}$ </td><td> $75.1^{\uparrow 19.3\%}$ </td><td> $43.2^{\uparrow 4.6\%}$ </td><td> $76.8^{\uparrow 2.2\%}$ </td><td> $62.1^{\uparrow 18.7\%}$ </td><td> $14.7^{\downarrow 2.3\%}$ </td><td> $16.4^{\uparrow 3.5\%}$ </td><td> $12.9^{\downarrow 13.5\%}$ </td><td> $67.2\%^{\uparrow 4.6\%}$ </td></tr><tr><td>DivPrune (CVPR&#x27;25) [1]</td><td> $1574.4^{\uparrow 25.4\%}$ </td><td> $80.5^{\uparrow 5.3\%}$ </td><td> $43.9^{\uparrow 10.0\%}$ </td><td> $84.2^{\uparrow 7.7\%}$ </td><td> $70.2^{\uparrow 7.9\%}$ </td><td> $26.5^{\uparrow 20.0\%}$ </td><td> $33.7^{\uparrow 40.3\%}$ </td><td> $13.8^{\uparrow 15.8\%}$ </td><td> $79.3\%^{\uparrow 15.8\%}$ </td></tr><tr><td colspan="10">Retention Ratio = 10%</td></tr><tr><td>ToMe (ICLR&#x27;23) [4]</td><td> $1664.0^{\uparrow 41.8\%}$ </td><td> $82.7^{\uparrow 30.5\%}$ </td><td> $45.5^{\uparrow 36.7\%}$ </td><td> $85.8^{\uparrow 35.8\%}$ </td><td> $75.2^{\uparrow 43.5\%}$ </td><td> $26.4^{\uparrow 37.4\%}$ </td><td> $33.0^{\uparrow 38.8\%}$ </td><td> $14.0^{\uparrow 5.4\%}$ </td><td> $81.5\%^{\uparrow 36.0\%}$ </td></tr><tr><td>FastV (ECCV&#x27;24) [5]</td><td> $1192.7^{\uparrow 40.9\%}$ </td><td> $73.4^{\uparrow 22.5\%}$ </td><td> $35.7^{\uparrow 21.3\%}$ </td><td> $63.7^{\uparrow 28.7\%}$ </td><td> $46.0^{\uparrow 27.0\%}$ </td><td> $9.1^{\uparrow 31.1\%}$ </td><td> $9.5^{\uparrow 48.5\%}$ </td><td> $11.3^{\uparrow 25.9\%}$ </td><td> $55.0\%^{\uparrow 29.8\%}$ </td></tr><tr><td>TRIM (COLING&#x27;25) [41]</td><td> $1092.3^{\uparrow 47.6\%}$ </td><td> $75.6^{\uparrow 29.4\%}$ </td><td> $35.3^{\uparrow 39.4\%}$ </td><td> $60.7^{\uparrow 49.0\%}$ </td><td> $59.7^{\uparrow 45.7\%}$ </td><td> $11.1^{\uparrow 48.2\%}$ </td><td> $13.3^{\uparrow 56.4\%}$ </td><td> $11.4^{\uparrow 32.4\%}$ </td><td> $57.6\%^{\uparrow 45.1\%}$ </td></tr><tr><td>SparseVLM (ICML&#x27;25) [64]</td><td> $1267.1^{\uparrow 23.3\%}$ </td><td> $73.4^{\uparrow 26.2\%}$ </td><td> $37.3^{\uparrow 18.8\%}$ </td><td> $67.6^{\uparrow 18.9\%}$ </td><td> $46.9^{\uparrow 21.4\%}$ </td><td> $9.2^{\uparrow 11.1\%}$ </td><td> $10.6^{\uparrow 27.5\%}$ </td><td> $12.0^{\downarrow 1.9\%}$ </td><td> $57.2\%^{\uparrow 18.6\%}$ </td></tr><tr><td>DivPrune (CVPR&#x27;25) [1]</td><td> $1468.4^{\uparrow 33.7\%}$ </td><td> $78.4^{\uparrow 8.0\%}$ </td><td> $40.3^{\uparrow 48.9\%}$ </td><td> $80.7^{\uparrow 11.5\%}$ </td><td> $62.8^{\uparrow 9.9\%}$ </td><td> $22.6^{\uparrow 26.5\%}$ </td><td> $16.5^{\uparrow 44.4\%}$ </td><td> $12.7^{\uparrow 21.6\%}$ </td><td> $70.8\%^{\uparrow 30.1\%}$ </td></tr><tr><td colspan="10">Retention Ratio = 0%</td></tr><tr><td>VTW (AAAI&#x27;25) [29]</td><td> $879.9^{\uparrow 36.6\%}$ </td><td> $72.9^{\uparrow 15.0\%}$ </td><td> $31.2^{\uparrow 25.4\%}$ </td><td> $42.0^{\uparrow 25.0\%}$ </td><td> $29.2^{\uparrow 21.4\%}$ </td><td> $6.4^{\uparrow 41.0\%}$ </td><td> $4.2^{\uparrow 52.8\%}$ </td><td> $11.6^{\uparrow 25.5\%}$ </td><td> $44.5\%^{\uparrow 31.0\%}$ </td></tr></table>

Table 1. Comparisons of visual token compression methods in the initial denoising step of LLaDA-V. Each cell reports the evaluation score. For the LLaDA-V baseline, bottom-right values indicate its reference inference time (min:sec), while for compression methods, the bottom-right values show the relative time change with respect to the baseline (↑ faster / time reduced, ↓ slower / time increased). Across different pruning ratios, existing compression methods achieve moderate efficiency gains while causing significant performance drops. Notably, VTW prunes all visual tokens at a specific layer and is therefore excluded from the comparison at other retention ratios.

## 4.1. Experimental Settings

where S is the number of denoising steps, $n _ { s }$ represents the number of tokens actively involved in computation at step s, T remains the number of Transformer layers, and the other symbols follow the same definitions as above. Note that this formulation only provides an approximation of FLOPs. The exact count may vary due to internal caching mechanisms and sampling strategies of individual dMLLMs, which determine the actual number of active tokens across different steps and computation stages.

Backbones and Benchmarks. We conduct experiments with LLaDA-V [61] and LaViDa-Dream [25], two representative dMLLM backbones that are trained via from-scratch and AR-to-diffusion, respectively. In addition, we evaluate both backbones over twelve widely adopted benchmarks, including ten for image understanding and two for video understanding. All experiments are conducted on $8 \times \mathrm { A 8 0 0 }$ GPUs. More details of the backbones, benchmarks and compression methods are provided in the Appendix.

Theoretically, dMLLMs incur heavier computational overhead than MLLMs, as each denoising step performs a prefilling-style computation with bidirectional attention. The linear and quadratic terms on the active token number $n _ { s }$ significantly increase the computational cost for multimodal inputs with excessive visual tokens, making it especially beneficial to reduce redundant visual tokens to alleviate this computational burden.

## 4. Experiments

Token Compression Methods. As described in Section 2.1, most existing visual token pruning methods adopt four underlying pruning mechanisms, namely, similarity-based pruning, attention-based pruning, querybased pruning, and transformation-based pruning. Unlike the other three visual token pruning approaches, transformer-based pruning is a built-in design in both MLLMs and dMLLMs which operates at a fixed rate and is non-adjustable at inference time [3, 21, 48]. We therefore exclude the transformer-based pruning and focus on the other three pruning approaches in our experiments. Specifically, we select two representative methods for each of the three pruning approaches, including ToMe [4] and DivPrune [1] for similarity-based pruning, FastV [5], VTW [29] for attention-based pruning, and Spar-

We conduct extensive experiments to investigate whether visual token redundancy exists in dMLLMs and how visual token pruning affects the inference efficiency and accuracy of dMLLMs. Specifically, we study dMLLMs by adapting representative pruning methods that have demonstrated impressive performance for autoregressive MLLMs.

<table><tr><td>Method</td><td>MME [11]</td><td>SQA [35]</td><td>GQA [16]</td><td>POPE [26]</td><td>MMB [32]</td><td>TVQA [44]</td><td>CQA [37]</td><td>MMMUP [63]</td><td>Avg.</td></tr><tr><td>LaViDa-Dream [25]</td><td>1857.0 5:37</td><td>75.3 4:28</td><td>56.4 27:53</td><td>87.0 20:04</td><td>73.6 9:31</td><td>42.4 11:35</td><td>60.4 6:35</td><td>13.5 4:22</td><td>100.0% 90:05</td></tr><tr><td colspan="10">Retention Ratio = 50%</td></tr><tr><td>ToMe (ICLR&#x27;23) [4]</td><td>1638.6↑ 3.9%</td><td>74.6↑ 0.4%</td><td>53.0↑ 2.2%</td><td>83.7↑ 3.7%</td><td>68.5↑ 2.5%</td><td>31.3↑ 4.3%</td><td>33.9↑ 4.1%</td><td>10.6↑ 4.2%</td><td>84.9%↑ 3.1%</td></tr><tr><td>FastV (ECCV&#x27;24) [5]</td><td>1771.1↓ 40.9%</td><td>73.4↓ 39.2%</td><td>52.2↓ 51.2%</td><td>84.3↓ 45.5%</td><td>69.3↓ 37.1%</td><td>30.6↓ 50.4%</td><td>22.8↓ 55.4%</td><td>12.6↓ 51.5%</td><td>85.0%↓ 47.4%</td></tr><tr><td>TRIM (COLING&#x27;25) [41]</td><td>1589.1↑ 1.5%</td><td>75.1↓ 0.7%</td><td>49.6↑ 0.0%</td><td>78.4↑ 0.8%</td><td>69.9↑ 0.4%</td><td>28.2↑ 2.6%</td><td>31.4↑ 2.8%</td><td>11.0↑ 3.4%</td><td>82.3%↑ 1.0%</td></tr><tr><td>SparseVLM (ICML&#x27;25) [64]</td><td>1798.9↓ 54.0%</td><td>56.2↓ 37.3%</td><td>54.5↓ 45.2%</td><td>83.9↓ 63.1%</td><td>69.7↓ 37.0%</td><td>29.3↓ 67.5%</td><td>27.1↓ 67.6%</td><td>10.8↓ 51.1%</td><td>81.6%↓ 53.3%</td></tr><tr><td>DivPrune (CVPR&#x27;25) [1]</td><td>1683.5↓ 1.2%</td><td>74.2↓ 1.5%</td><td>52.8↓ 0.7%</td><td>85.2↓ 2.5%</td><td>66.7↓ 0.2%</td><td>32.3↓ 0.6%</td><td>30.8↓ 0.5%</td><td>13.1↓ 0.8%</td><td>87.0%↓ 1.1%</td></tr><tr><td colspan="10">Retention Ratio = 25%</td></tr><tr><td>ToMe (ICLR&#x27;23) [4]</td><td>1606.5↑ 5.3%</td><td>74.3↑ 1.1%</td><td>51.8↑ 4.1%</td><td>81.7↑ 4.2%</td><td>67.4↑ 3.9%</td><td>28.9↑ 6.2%</td><td>29.4↑ 4.3%</td><td>10.6↑ 5.0%</td><td>82.2%↑ 4.3%</td></tr><tr><td>FastV (ECCV&#x27;24) [5]</td><td>1593.7↓ 30.3%</td><td>72.1↓ 31.3%</td><td>47.8↓ 28.9%</td><td>75.3↓ 30.7%</td><td>64.6↓ 26.4%</td><td>19.6↓ 35.0%</td><td>15.1↓ 40.3%</td><td>10.2↓ 33.6%</td><td>73.4%↓ 31.1%</td></tr><tr><td>TRIM (COLING&#x27;25) [41]</td><td>1552.1↑ 2.4%</td><td>74.3↓ 1.9%</td><td>46.1↑ 1.1%</td><td>70.8↑ 3.2%</td><td>67.6↑ 1.6%</td><td>24.1↑ 3.2%</td><td>27.3↑ 3.8%</td><td>10.9↑ 5.0%</td><td>77.5%↑ 2.2%</td></tr><tr><td>SparseVLM (ICML&#x27;25) [64]</td><td>1582.7↓ 41.5%</td><td>52.3↓ 31.3%</td><td>49.4↓ 31.4%</td><td>76.2↓ 48.9%</td><td>66.2↓ 32.9%</td><td>19.9↓ 54.7%</td><td>14.6↓ 54.4%</td><td>10.5↓ 42.7%</td><td>71.1%↓ 41.3%</td></tr><tr><td>DivPrune (CVPR&#x27;25) [1]</td><td>1442.6↑ 1.5%</td><td>73.8↓ 1.1%</td><td>48.8↑ 0.5%</td><td>77.4↑ 0.2%</td><td>59.1↑ 2.5%</td><td>23.9↑ 1.3%</td><td>20.7↑ 1.8%</td><td>12.2↑ 2.7%</td><td>76.6%↑ 0.9%</td></tr><tr><td colspan="10">Retention Ratio = 10%</td></tr><tr><td>ToMe (ICLR&#x27;23) [4]</td><td>1558.9↑ 6.5%</td><td>74.1↑ 3.7%</td><td>49.3↑ 4.8%</td><td>76.3↑ 5.3%</td><td>62.8↑ 4.0%</td><td>21.4↑ 7.2%</td><td>22.4↑ 6.8%</td><td>11.1↑ 4.6%</td><td>76.6%↑ 5.3%</td></tr><tr><td>FastV (ECCV&#x27;24) [5]</td><td>1374.1↓ 19.9%</td><td>71.2↓ 15.3%</td><td>42.8↓ 21.9%</td><td>61.2↓ 22.8%</td><td>56.8↓ 17.7%</td><td>12.6↓ 23.2%</td><td>12.9↓ 26.6%</td><td>10.8↓ 21.4%</td><td>65.4%↓ 21.7%</td></tr><tr><td>TRIM (COLING&#x27;25) [41]</td><td>1385.4↑ 3.0%</td><td>72.8↓ 1.5%</td><td>43.2↑ 1.9%</td><td>62.4↑ 3.9%</td><td>61.1↑ 1.8%</td><td>19.6↑ 3.6%</td><td>17.4↑ 3.5%</td><td>11.3↑ 8.0%</td><td>70.1%↑ 2.8%</td></tr><tr><td>SparseVLM (ICML&#x27;25) [64]</td><td>1531.4↓ 35.6%</td><td>51.4↓ 19.0%</td><td>45.2↓ 21.9%</td><td>67.4↓ 44.8%</td><td>60.8↓ 27.8%</td><td>15.0↓ 50.2%</td><td>11.6↓ 47.6%</td><td>10.1↓ 43.1%</td><td>65.0%↓ 34.9%</td></tr><tr><td>DivPrune (CVPR&#x27;25) [1]</td><td>1230.9↑ 3.6%</td><td>72.3↑ 0.4%</td><td>43.7↑ 3.8%</td><td>64.0↑ 2.9%</td><td>45.5↑ 3.5%</td><td>16.3↑ 3.2%</td><td>15.3↑ 3.3%</td><td>10.8↑ 3.8%</td><td>64.9%↑ 3.3%</td></tr><tr><td colspan="10">Retention Ratio = 0%</td></tr><tr><td>VTW (AAAI&#x27;25) [29]</td><td>1733.3↑ 0.9%</td><td>74.9↑ 2.6%</td><td>43.4↑ 0.4%</td><td>65.2↑ 0.5%</td><td>70.3↑ 1.6%</td><td>13.5↑ 2.2%</td><td>13.2↑ 0.3%</td><td>11.4↑ 1.1%</td><td>72.3%↑ 0.9%</td></tr></table>

Table 2. Comparison of visual token compression methods in the initial denoising step of LaViDa-Dream. Each cell reports the eval uation score. For the LaViDa-Dream baseline, bottom-right values indicate its reference inference time (min:sec), while for compression methods, the bottom-right values show the relative time change with respect to the baseline (↑ faster / time reduced, ↓ slower / time increased). Across different pruning ratios, existing compression methods achieve only negligible efficiency gains while causing significant performance drops. Notably, FastV and SparseVLM are even slower than the baseline due to their incompatibility with efficient attention operators [10].

seVLM [64] and TRIM [41] for query-based pruning.

## 4.2. Short-answer Tasks: No Visual Redundancy

We first conduct experiments on both LLaDA-V and LaViDa-Dream (see Table 1 and Table 2) under varying token retention ratios during the initial denoising step. For LLaDA-V, various token compression methods exhibit a consistent pattern: models either show slight degradation in performance with only marginal efficiency gains at a retention ratio of 50%, or achieve minimal acceleration at the cost of significant performance degradation when the reten tion ratio is below 50%. For LaViDa-Dream, existing compression methods perform even worse—exhibiting severe performance degradation while offering negligible speedups across all retention ratios; some are even slower than the baseline due to their incompatibility with efficient attention operators [10]. These findings indicate that for both architectures, the trade-off between efficiency and performance is largely ineffective.

applied). For LLaDA-V, pruning visual tokens provides only marginal efficiency gains while causing a noticeable performance drop, whereas for LaViDa-Dream, pruning yields almost no acceleration effect. The results remain consistent with those from applying pruning in the first denoising step: pruning visual tokens during intermediate steps still yields a poor trade-off between performance and efficiency across both architectures.

Overall, these findings suggest that existing compression techniques—though based on diverse underlying mechanisms—fail to effectively remove redundant visual tokens, implying that each token contributes meaningfully to the reasoning process in short-answer tasks.

## 4.3. Long-answer Tasks: Redundancy Diverges across Backbones

Given that each denoising step in dMLLMs performs a similar bidirectional computation, it is natural to further explore applying these compression methods to later denoising steps for potential acceleration. As shown in Table 3 (short-answer tasks), compression is applied to different portions of denoising steps (the percentage indicates the fraction of total denoising steps to which pruning is

We further extend our experiments to long-answer tasks. Notably, two backbones exhibit divergent behaviors (see Table 3): in LLaDA-V, visual redundancy gradually emerges as pruning is applied to the middle and later denoising steps, whereas in LaViDa-Dream, it remains largely absent. For instance, in LLaDA-V, DivPrune [1] prunes 75% of visual tokens in the middle decoding steps, yielding a

<table><tr><td rowspan="3">Method</td><td colspan="4">Short-answer Tasks</td><td colspan="10">Long-answer Tasks</td></tr><tr><td>MME [11]</td><td>POPE [26]</td><td colspan="2">VMME [12]</td><td colspan="3">IVQA [39]</td><td colspan="3">DVQA [38]</td><td colspan="4">VDC [34]</td></tr><tr><td>50%</td><td>50%</td><td colspan="2">50%</td><td>25%</td><td>50%</td><td>75%</td><td>25%</td><td>50%</td><td>75%</td><td>25%</td><td>50%</td><td>75%</td><td></td></tr><tr><td colspan="15">From-scratch dMLLM</td></tr><tr><td>LLaDA-V [61]</td><td>1994.4 5:47</td><td>87.3 15:45</td><td>56.0 22:02</td><td></td><td>66.2 42:03</td><td></td><td></td><td>83.9 1:22:52</td><td></td><td></td><td>2.8 56:16</td><td></td><td></td><td></td></tr><tr><td>FastV [5]</td><td>1377.0↓ 27.1%</td><td>73.8↓ 36.8%</td><td>46.3↓ 52.2%</td><td>45.9↑ 14.0%</td><td>64.0↓ 31.3%</td><td>65.3↓ 75.5%</td><td>58.2↑ 14.8%</td><td>81.2↓ 26.9%</td><td>83.0↓ 70.4%</td><td>2.2↓ 13.5%</td><td>2.6↓ 84.4%</td><td>2.7↓ 156.9%</td><td></td><td></td></tr><tr><td>VTW [29]</td><td>1213.9↑ 12.1%</td><td>55.4↑ 0.7%</td><td>46.2↑ 10.9%</td><td>45.3↑ 64.4%</td><td>63.5↑ 43.8%</td><td>65.2↑ 23.7%</td><td>56.5↑ 63.2%</td><td>80.9↑ 43.2%</td><td>83.0↑ 23.5%</td><td>1.9↑ 69.4%</td><td>2.4↑ 46.5%</td><td>2.7↑ 23.4%</td><td></td><td></td></tr><tr><td>SparseVLM [64]</td><td>1467.0↓ 40.1%</td><td>76.8↓ 49.3%</td><td>49.1↓ 71.7%</td><td>46.5↑ 12.5%</td><td>64.0↓ 34.7%</td><td>65.5↓ 75.9%</td><td>60.0↑ 13.1%</td><td>81.5↓ 30.7%</td><td>83.1↓ 71.7%</td><td>2.3↓ 13.9%</td><td>2.6↓ 86.7%</td><td>2.7↓ 160.7%</td><td></td><td></td></tr><tr><td>DivPrune [1]</td><td>1807.0↑ 11.5%</td><td>86.4↓ 2.0%</td><td>58.0↑ 7.7%</td><td>61.7↑ 47.1%</td><td>65.0↑ 30.5%</td><td>65.8↑ 14.0%</td><td>78.9↑ 46.4%</td><td>83.1↑ 30.7%</td><td>83.5↑ 13.7%</td><td>2.7↑ 53.2%</td><td>2.8↑ 35.1%</td><td>2.8↑ 17.0%</td><td></td><td></td></tr><tr><td colspan="15">AR-to-diffusion dMLLM</td></tr><tr><td>LaViDa-Dream [25]</td><td>1857.0 5:37</td><td>87.0 20:04</td><td>-</td><td></td><td>35.5 8:53</td><td></td><td></td><td>55.1 17:07</td><td></td><td></td><td>-</td><td>-</td><td>-</td><td></td></tr><tr><td>VTW [29]</td><td>1835.4↓ 0.9%</td><td>86.7↓ 0.8%</td><td>-</td><td>17.7↓ 2.8%</td><td>17.7↓ 1.1%</td><td>18.3↓ 1.3%</td><td>24.0↓ 3.8%</td><td>24.1↓ 1.4%</td><td>28.3↓ 3.0%</td><td>-</td><td>-</td><td>-</td><td></td><td></td></tr><tr><td>DivPrune [1]</td><td>1857.0↓ 2.4%</td><td>86.5↓ 2.2%</td><td>-</td><td>15.9↓ 0.6%</td><td>15.9↓ 1.1%</td><td>16.1↓ 1.9%</td><td>23.7↓ 0.3%</td><td>23.9↓ 1.9%</td><td>25.6↓ 2.4%</td><td>-</td><td>-</td><td>-</td><td></td><td></td></tr></table>

Table 3. Performance comparison when different token compression methods are applied at various denoising steps under a 25% retention ratio. The percentage indicates the fraction of total denoising steps to which pruning is applied; for short-answer tasks, the small number of steps permits only 50% as a valid ratio. For each compression method, the top value indicates the evaluation score, and the bottom-right value shows the relative change in inference time compared with the model’s baseline (↑ faster / time reduced, ↓ slower / time increased). Baseline bottom-right values show the reference inference time (min:sec). Note that LaViDa is incapable of video understanding<sup>1</sup>, and is thus excluded from video benchmarks. In short-answer tasks, existing compression methods for both LLaDA-V and LaViDa-Dream yield inefficient trade-offs between efficiency and performance. In long-answer tasks, these methods remain ineffective on LaViDa-Dream, whereas on LLaDA-V some of them (e.g., DivPrune and VTW) deliver notable efficiency gains with nearly lossless performance when pruning begins at middle or later denoising steps

1.44× speed-up on InfoVQA and DocVQA (−30.5% and −30.7% in time) with almost no performance degradation (65.0 vs. 66.2; 83.1 vs. 83.9). In contrast, in LaViDa-Dream, applying various compression techniques to different portions of denoising steps yields consistent yet suboptimal results, suffering from substantial performance drops (nearly 50%) alongside reduced efficiency.

These findings reveal that visual redundancy manifests differently between from-scratch and AR-to-diffusion dM-LLMs: in from-scratch dMLLMs, redundancy grows with longer generation length and more steps, whereas it remains negligible in AR-to-diffusion models. Unlike MLLMs, where visual redundancy consistently appears across both short- and long-answer tasks, dMLLMs exhibit more complex behavior: the emergence of redundancy depends not only on the modeling paradigm—from-scratch diffusion training versus AR-to-diffusion adaptation—but also on the task type, as short- and long-answer tasks may yield different redundancy patterns. This intriguing contrast motivates a deeper analysis in the following section.

## 5. Understanding Visual Token Redundancy in dMLLMs

In this section, we analyze the underlying causes of the divergent behaviors observed in from-scratch trained and AR-to-diffusion adapted dMLLMs across short-answer and long-answer tasks. These behaviors can be interpreted through two key factors: the information loss introduced by visual token pruning and the model’s ability to restore this lost information during later denoising. Viewed from these perspectives, visual redundancy persists in dMLLMs, offering new directions for future acceleration strategies.

## 5.1. Information Loss from Visual Token Pruning

Takeaway: Pruning visual tokens leads to more severe information loss in dMLLMs than in MLLMs, as both from-scratch and AR-to-diffusion dMLLMs exhibit a much stronger reliance on visual information compared with MLLMs.

As shown in Figure 2 (h), answer tokens in MLLMs exhibit near-zero attention to visual tokens (the orange line) throughout the entire decoding process. This phenomenon remains highly consistent across different model scales and both short- and long-answer tasks, despite the large number of visual tokens present. This pattern, also reported in prior studies [22, 29], provides strong evidence that a substantial portion of visual tokens can be pruned without harming performance.

In contrast, answer tokens in dMLLMs maintain a much stronger attention ratio toward visual tokens across different generation lengths and model backbones. In from-scratch dMLLMs, this ratio remains consistently high throughout dLLM layers (orange line in Figure 2 (e-f)), whereas in AR-to-diffusion dMLLMs, it displays a sharp spike at a specific layer (orange line in Figure 2 (d)). This strong visual dependence explains the pronounced performance degradation observed in Table 1 and Table 2 when pruning is applied, since removing key visual information that answer tokens rely heavily on tends to disrupt the inference process. This phenomenon is consistent with our experimental results for all tasks and model types, except for the long-answer setting in LLaDA-V, which will be discussed in the following subsection.

![](images/a3e176365e02d9ba5f77d56622da1b5b4441ecb20fde4aea5c28acff87e2731f.jpg)  
(a)

![](images/d0de515af3b4482c5e42a1d8c1cd40dac36b7fadbc9e6c88c72d171bf9ab35eb.jpg)

![](images/ce2397cf8208c2006632e4989010d3db1af4681fc05670af45c2747539eb8f5f.jpg)

![](images/24e6ca16cfa11b98fd8311e7518c816267cd8eb4be9fcf498d4b38190d2bff02.jpg)

(d)  
![](images/8dff243cc229cfbe837e539dd67410579a2af3bf99761cf0c66cc629e65b3a62.jpg)

![](images/78303b369825c616f72bc3f0604faa8b609357971bfa6c69fcb8914d1bf03309.jpg)

![](images/5ced23c7f84c216fe3e91da422b8f3dcec9a14f2b6d50cef11253334a8ac75f9.jpg)  
(b)

![](images/33fa2a1c08338143befa04860d82d3e7fb5c76b8d687f4aae66adfe6842a7e78.jpg)

![](images/10ccdb50e3a4cd642e32c80b009ec4856533bec22ec43f263dc0a3b3f78ce57f.jpg)  
(c)

![](images/5c61478f3ffe66b47cad42abddfb7b8f6ee2b85fad0491fcddc88a17364cf61a.jpg)

(e)  
![](images/0324f007a34b45d87967382035bad5b3a36f8e5b0ae23e4bd27e773edf5e3b1c.jpg)  
(f)

![](images/8b66daf3fd6637ccb29477bd23b9594e7de0e37d93ab4d32d425b50a681390e1.jpg)  
(h)  
Figure 2. Visualization of the fraction of attention from answer tokens to each token type across layers, and of logit dynamics across denoising steps. (a, d) show the heatmaps and attention ratio variations of LaViDa-Dream [25] (representing AR-to-diffusion dMLLMs) across both short- and long-answer tasks. (b, e) correspond to LLaDA-V [61] (representing from-scratch dMLLMs) on paragraph-level long-answer tasks (Video Detail Description [34]). (c, f) present LLaDA-V on sentence-level long-answer tasks (InfoVQA [39] and DocVQA [38]). (g) shows the attention ratio trends of LLaDA-V on short-answer tasks, and (h) depicts those of MLLMs on both shortand long-answer tasks. We observe three key patterns: (1) Compared with MLLMs, both dMLLM variants exhibit stronger reliance on visual tokens, leading to a more pronounced performance drop when visual tokens are pruned. (2) The self-attention intensity among answer tokens progressively increases from MLLMs to LaViDa-Dream and LLaDA-V, endowing dMLLMs, especially LLaDA-V, with a stronger capacity to recover lost information through bidirectional contextual refinement. (3) Steps 7–8 in (c) and (f) reveal a consistent pattern where a rise in logits follows a preceding surge in answer-token self-attention, suggesting that stronger self-attention facilitates information restoration during denoising.

## 5.2. Restoration Ability Under Information Loss

Takeaway: From-scratch dMLLMs demonstrate a strong capacity to recover lost visual information through contextual integration, whereas AR-to-diffusion dMLLMs lack such flexibility, retaining an autoregressive tendency that limits their ability to recover from loss. The restoration ability of dMLLMs depends on both the self-attention strength among answer tokens and the number of denoising steps.

A key distinction between dMLLMs and MLLMs lies in their decoding dynamics [40, 60]. Unlike MLLMs, which generate tokens autoregressively and freeze each token once produced, dMLLMs continuously perform bidirectional interactions and information fusion among all answer tokens throughout the denoising steps. This distinction is reflected in our observations: in from-scratch dMLLMs, answer tokens exhibit the strongest self-attention (red line in Figure 2 (e-g)), whereas in AR-to-diffusion dMLLMs, the selfattention is weaker yet still higher than that in MLLMs (red line in Figure 2 (d, h)). Such enhanced self-focus enables dMLLMs to mitigate the information loss caused by pruning visual tokens through continuous information exchange among answer tokens.

For long-answer tasks, this effect can explain our divergent experimental results in Table 3. A strong self-focus among answer tokens allows from-scratch dMLLMs to recover from pruning, while a weaker self-focus results in persistent degradation in AR-to-diffusion dMLLMs. This pattern can be further confirmed by examining cases from DocVQA [38] and InfoVQA [39], where we observe a consistent pattern unique to from-scratch dMLLMs: a sharp rise in answer-to-answer self-attention between denoising steps often precedes a surge in token confidence (e.g., the increase in logits at Step 8 in Figure 2 (c) coincides with the rise in self-attention at Step 7 in Figure 2 (f)). For tasks with longer generations, such as Video Detail Captioning [34], the same pattern persists but manifests more smoothly—without the sharp spike (Figure 2 (b and e)).

For short-answer tasks, aside from attention strength among answer tokens, the number of denoising steps is crucial for understanding the experimental results in Table 3. Since short-answer tasks involve very few total denoising steps, the correct answer token (e.g., a number, “Yes/No,” or a single choice) is typically determined within the first one or two denoising steps. Applying pruning too early leaves insufficient room for subsequent denoising to recover the missing information, leading to severe performance drops. Conversely, applying pruning at later steps yields only marginal efficiency gains. Both cases lead to a suboptimal trade-off between performance and efficiency.

Together, these findings show that effective restoration requires both strong answer-to-answer self-attention and a sufficient number of later denoising steps. This explains why visual redundancy emerges only in from-scratch dM-LLMs when handling long-answer tasks.

## 5.3. Visual Token Redundancy in dMLLMs

Takeaway: Unlike in MLLMs, where redundancy manifests as unnecessary visual tokens that incur extra computational overhead, redundancy in dMLLMs arises from the model’s ability to recover from missing information through iterative bidirectional refinement.

In MLLMs, visual redundancy typically refers to the unnecessary computational overhead of dispensable visual tokens, which contribute little to the inference process and can therefore be safely pruned without affecting performance.

In contrast, visual redundancy in dMLLMs reflects a fundamentally different and dynamic property. Since pruning inevitably causes information loss and dMLLMs can recover part of it through later bidirectional updates, visual redundancy thus describes how much computation can be reduced by leveraging the model’s ability to compensate for missing information while maintaining performance.

Viewed from this perspective, visual redundancy in dM-LLMs is tightly coupled with the model’s restoration ability discussed in Section 5.2. When models exhibit strong selfattention among answer tokens, as observed in from-scratch dMLLMs, they can reintegrate dispersed visual cues to reconstruct missing information, leading to a more resilient response to pruning. Conversely, when such self-attention is weak, as in AR-to-diffusion dMLLMs, visual redundancy diminishes because the model cannot sufficiently recover pruned information, and even a small reduction in visual tokens leads to noticeable degradation.

## 5.4. Insights for Visual Pruning in dMLLMs

Takeaway: For from-scratch dMLLMs, both attention scores and logits can serve as indicators of when and where to prune, with progressive pruning or pruning applied after certain decoding steps being more suitable. For AR-todiffusion dMLLMs, attention scores can serve as an effective metric, suggesting layer-skipping strategies.

Given the preceding analysis and our definition of visual redundancy, we attribute the absence of visual token redundancy in short-answer tasks for from-scratch dMLLMs to their strong reliance on visual information and the limited number of denoising steps, which together constrain their ability to restore missing information. For long-answer tasks, we observe a consistent correlation between the attention ratio and logits, as mentioned in Section 5.2. As the attention ratio shifts, the logits also vary accordingly, suggesting that both signals can jointly guide when and how to apply progressive pruning or pruning after certain decoding steps.

For AR-to-diffusion adapted models, another approach is to prevent information loss. Answer tokens exhibit consistently low attention to visual tokens across both shortand long-answer tasks, except for the sharp spike (the orange line at layer 4 in Figure 2 (d)). This pattern suggests a less harmful strategy: allow visual tokens to bypass layers with persistently low attention, while preserving those layers around the spike. In practice, this corresponds to attention-aware skipping that excludes visual tokens from low-dependency regions, leveraging the model’s limited restoration ability without excessive information loss.

## 6. Conclusion

We present the first comprehensive analysis of visual token redundancy in dMLLMs. Our study reveals that visual redundancy in dMLLMs is fundamentally different from that in MLLMs: it reflects the model’s capacity to recover pruning-induced information loss through iterative bidirectional refinement, rather than mere token-level dispensability. This property emerges only in from-scratch dMLLMs on long-answer tasks, where strong answer-token self-attention and sufficient denoising steps enable effective restoration. These insights lead to practical pruning guidelines: layer-skipping for AR-to-diffusion models and progressive or late-step pruning for from-scratch architectures.

## Acknowledgment

This study is funded by the Ministry of Education, Singapore, under the Tier-2 project scheme with project number MOET2EP20123-0003.

## References

[1] Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversity-based visual token pruning for large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9392–9401, 2025. 2, 4, 5, 6

[2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023. 3

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025. 2, 4

[4] Daniel Bolya, Cheng-Yang Fu, Xiaoliang Dai, Peizhao Zhang, Christoph Feichtenhofer, and Judy Hoffman. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022. 2, 4, 5

[5] Liang Chen, Haozhe Zhao, Tianyu Liu, Shuai Bai, Junyang Lin, Chang Zhou, and Baobao Chang. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large vision-language models. In European Conference on Computer Vision, pages 19–35. Springer, 2024. 2, 3, 4, 5, 6

[6] Xinhua Chen, Sitao Huang, Cong Guo, Chiyue Wei, Yintao He, Jianyi Zhang, Hai Li, Yiran Chen, et al. Dpad: Efficient diffusion language models with suffix dropout. arXiv preprint arXiv:2508.14148, 2025. 3

[7] Zhe Chen, Weiyun Wang, Yue Cao, Yangzhou Liu, Zhangwei Gao, Erfei Cui, Jinguo Zhu, Shenglong Ye, Hao Tian, Zhaoyang Liu, et al. Expanding performance boundaries of open-source multimodal models with model, data, and testtime scaling. arXiv preprint arXiv:2412.05271, 2024. 2

[8] Zesen Cheng, Sicong Leng, Hang Zhang, Yifei Xin, Xin Li, Guanzheng Chen, Yongxin Zhu, Wenqi Zhang, Ziyang Luo, Deli Zhao, et al. Videollama 2: Advancing spatialtemporal modeling and audio understanding in video-llms. arXiv preprint arXiv:2406.07476, 2024. 2

[9] Xiangxiang Chu, Limeng Qiao, Xinyang Lin, Shuang Xu, Yang Yang, Yiming Hu, Fei Wei, Xinyu Zhang, Bo Zhang, Xiaolin Wei, et al. Mobilevlm: A fast, strong and open vision language assistant for mobile devices. arXiv preprint arXiv:2312.16886, 2023. 2

[10] Tri Dao, Dan Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast and memory-efficient exact´ attention with io-awareness. In Advances in neural information processing systems, pages 16344–16359, 2022. 5

[11] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Zhenyu Qiu, Wei Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and Rongrong Ji. Mme: A comprehensive evaluation benchmark for multimodal large language models. ArXiv, abs/2306.13394, 2023. 4, 5, 6, 1

[12] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. arXiv preprint arXiv:2405.21075, 2024. 6, 1

[13] Zhangwei Gao, Zhe Chen, Erfei Cui, Yiming Ren, Weiyun Wang, Jinguo Zhu, Hao Tian, Shenglong Ye, Junjun He, Xizhou Zhu, et al. Mini-internvl: a flexible-transfer pocket multi-modal model with 5% parameters and 90% perfor mance. Visual Intelligence, 2(1):32, 2024. 2

[14] Shansan Gong, Shivam Agarwal, Yizhe Zhang, Jiacheng Ye, Lin Zheng, Mukai Li, Chenxin An, Peilin Zhao, Wei Bi, Jiawei Han, et al. Scaling diffusion language models via adaptation from autoregressive models. arXiv preprint arXiv:2410.17891, 2024. 1

[15] Pengcheng Huang, Shuhao Liu, Zhenghao Liu, Yukun Yan, Shuo Wang, Zulong Chen, and Tong Xiao. Pc-sampler: Position-aware calibration of decoding bias in masked diffusion models. arXiv preprint arXiv:2508.13021, 2025. 1, 3

[16] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709, 2019. 4, 5, 1

[17] Samar Khanna, Siddhant Kharbanda, Shufan Li, Harshit Varma, Eric Wang, Sawyer Birnbaum, Ziyang Luo, Yanis Miraoui, Akash Palrecha, Stefano Ermon, et al. Mercury: Ultra-fast language models based on diffusion. arXiv eprints, pages arXiv–2506, 2025. 1

[18] Jaeyeon Kim, Lee Cheuk-Kit, Carles Domingo-Enrich, Yilun Du, Sham Kakade, Timothy Ngotiaoco, Sitan Chen, and Michael Albergo. Any-order flexible length masked diffusion. arXiv preprint arXiv:2509.01025, 2025. 1, 3

[19] Andrei Kolmogoroff. Uber die analytischen methoden in der<sup>¨</sup> wahrscheinlichkeitsrechnung. Mathematische Annalen, 104 (1):415–458, 1931. 3

[20] Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International Journal of Computer Vision, 123:32 – 73, 2016. 1

[21] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024. 4

[22] Duo Li, Zuhao Yang, and Shijian Lu. Todre: Visual token pruning via diversity and task awareness for efficient large vision-language models. arXiv e-prints, pages arXiv–2505, 2025. 3, 6

[23] Jinsong Li, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Jiaqi Wang, and Dahua Lin. Beyond fixed: Training-free variablelength denoising for diffusion large language models. arXiv preprint arXiv:2508.00819, 2025. 1, 3

[24] Pengxiang Li, Yefan Zhou, Dilxat Muhtar, Lu Yin, Shilin Yan, Li Shen, Yi Liang, Soroush Vosoughi, and Shiwei Liu.

Diffusion language models know the answer before decoding. arXiv preprint arXiv:2508.19982, 2025. 1, 3

[25] Shufan Li, Konstantinos Kallidromitis, Hritik Bansal, Akash Gokul, Yusuke Kato, Kazuki Kozuka, Jason Kuen, Zhe Lin, Kai-Wei Chang, and Aditya Grover. Lavida: A large diffusion language model for multimodal understanding. arXiv preprint arXiv:2505.16839, 2025. 1, 2, 4, 5, 6, 7

[26] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355, 2023. 4, 5, 6, 1

[27] Youwei Liang, Chongjian Ge, Zhan Tong, Yibing Song, Jue Wang, and Pengtao Xie. Not all patches are what you need: Expediting vision transformers via token reorganizations. arXiv preprint arXiv:2202.07800, 2022. 1

[28] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollar, and´ C. Lawrence Zitnick. Microsoft coco: Common objects in context. In European Conference on Computer Vision, 2014. 1

[29] Zhihang Lin, Mingbao Lin, Luxi Lin, and Rongrong Ji. Boosting multimodal large language models with visual tokens withdrawal for rapid inference. In Proceedings of the AAAI Conference on Artificial Intelligence, pages 5334– 5342, 2025. 4, 5, 6, 2

[30] Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llavanext: Improved reasoning, ocr, and world knowledge, 2024. 2

[31] Xuyang Liu, Ziming Wang, Junjie Chen, Yuhang Han, Yingyao Wang, Jiale Yuan, Jun Song, Linfeng Zhang, Siteng Huang, and Honggang Chen. Global compression commander: Plug-and-play inference acceleration for highresolution large vision-language models. arXiv preprint arXiv:2501.05179, 2025. 2

[32] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024. 4, 5, 1

[33] Zhiyuan Liu, Yicun Yang, Yaojie Zhang, Junjie Chen, Chang Zou, Qingyuan Wei, Shaobo Wang, and Linfeng Zhang. dllm-cache: Accelerating diffusion large language models with adaptive caching. arXiv preprint arXiv:2506.06295, 2025. 1, 3

[34] LMMs-Lab. Videodetailcaption. Hugging Face Dataset, 2024. 6, 7, 8, 1

[35] Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. Learn to explain: Multimodal reasoning via thought chains for science question answering. In Advances in Neural Information Processing Systems, pages 2507–2521, 2022. 4, 5, 1

[36] Xinyin Ma, Runpeng Yu, Gongfan Fang, and Xinchao Wang. dkv-cache: The cache for diffusion language models. arXiv preprint arXiv:2505.15781, 2025. 1

[37] Ahmed Masry, Do Xuan Long, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. Chartqa: A benchmark for question an-

swering about charts with visual and logical reasoning. arXiv preprint arXiv:2203.10244, 2022. 4, 5, 1

[38] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209, 2021. 6, 7, 8, 1

[39] Minesh Mathew, Viraj Bagal, Ruben Tito, Dimosthenis\` Karatzas, Ernest Valveny, and CV Jawahar. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706, 2022. 6, 7, 8, 1

[40] Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, and Chongxuan Li. Large language diffusion models. arXiv preprint arXiv:2502.09992, 2025. 1, 3, 7

[41] Alfredo Garrachon Ruiz, Tom´ as de la Rosa, and Daniel´ Borrajo. Trim: Token reduction and inference modeling for cost-effective language generation. arXiv preprint arXiv:2412.07682, 2024. 3, 4, 5, 2

[42] Michael S Ryoo, AJ Piergiovanni, Anurag Arnab, Mostafa Dehghani, and Anelia Angelova. Tokenlearner: What can 8 learned tokens do for images and videos? arXiv preprint arXiv:2106.11297, 2021. 1

[43] Yuzhang Shang, Mu Cai, Bingxin Xu, Yong Jae Lee, and Yan Yan. Llava-prumerge: Adaptive token reduction for efficient large multimodal models. arXiv preprint arXiv:2403.15388, 2024. 1, 2

[44] Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326, 2019. 4, 5, 1

[45] Yuerong Song, Xiaoran Liu, Ruixiao Li, Zhigeng Liu, Zengfeng Huang, Qipeng Guo, Ziwei He, and Xipeng Qiu. Sparse-dllm: Accelerating diffusion llms with dynamic cache eviction. arXiv preprint arXiv:2508.02558, 2025. 1, 3

[46] Jaesung Tae, Hamish Ivison, Sachin Kumar, and Arman Cohan. Tess 2: A large-scale generalist diffusion language model. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 21171–21188, 2025. 1

[47] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. 1

[48] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024. 2, 4

[49] Qingyan Wei, Yaojie Zhang, Zhiyuan Liu, Dongrui Liu, and Linfeng Zhang. Accelerating diffusion large language models with slowfast: The three golden principles. arXiv eprints, pages arXiv–2506, 2025. 1, 3

[50] Zichen Wen, Yifeng Gao, Shaobo Wang, Junyuan Zhang, Qintong Zhang, Weijia Li, Conghui He, and Linfeng Zhang. Stop looking for important tokens in multimodal

language models: Duplication matters more. arXiv preprint arXiv:2502.11494, 2025. 3

[51] Chengyue Wu, Hao Zhang, Shuchen Xue, Zhijian Liu, Shizhe Diao, Ligeng Zhu, Ping Luo, Song Han, and Enze Xie. Fast-dllm: Training-free acceleration of diffusion llm by enabling kv cache and parallel decoding. arXiv preprint arXiv:2505.22618, 2025. 1, 3

[52] Jiayang Wu, Wensheng Gan, Zefeng Chen, Shicheng Wan, and Philip S Yu. Multimodal large language models: A survey. In 2023 IEEE International Conference on Big Data (BigData), pages 2247–2256. IEEE, 2023. 3

[53] Zirui Wu, Lin Zheng, Zhihui Xie, Jiacheng Ye, Jiahui Gao, Yansong Feng, Zhenguo Li, Victoria W., Guorui Zhou, and Lingpeng Kong. Dreamon: Diffusion language models for code infilling beyond fixed-size canvas, 2025. 1, 3

[54] Long Xing, Qidong Huang, Xiaoyi Dong, Jiajie Lu, Pan Zhang, Yuhang Zang, Yuhang Cao, Conghui He, Jiaqi Wang, Feng Wu, et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247, 2024. 2

[55] Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, et al. Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19803– 19813, 2025. 2

[56] Dingchen Yang, Bowen Cao, Anran Zhang, Weibo Gu, Winston Hu, and Guang Chen. Beyond intermediate states: Explaining visual redundancy through language. arXiv preprint arXiv:2503.20540, 2025. 2

[57] Senqiao Yang, Yukang Chen, Zhuotao Tian, Chengyao Wang, Jingyao Li, Bei Yu, and Jiaya Jia. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19792–19802, 2025. 2

[58] Linli Yao, Lei Li, Shuhuai Ren, Lean Wang, Yuanxin Liu, Xu Sun, and Lu Hou. Deco: Decoupling token compression from semantic abstraction in multimodal large language models. arXiv preprint arXiv:2405.20985, 2024. 2

[59] Jiasheng Ye, Zaixiang Zheng, Yu Bao, Lihua Qian, and Quanquan Gu. Diffusion language models can perform many tasks with scaling and instruction-finetuning. arXiv preprint arXiv:2308.12219, 2023. 1

[60] Jiacheng Ye, Zhihui Xie, Lin Zheng, Jiahui Gao, Zirui Wu, Xin Jiang, Zhenguo Li, and Lingpeng Kong. Dream 7b: Diffusion large language models. arXiv preprint arXiv:2508.15487, 2025. 1, 3, 7

[61] Zebin You, Shen Nie, Xiaolu Zhang, Jun Hu, Jun Zhou, Zhiwu Lu, Ji-Rong Wen, and Chongxuan Li. Llada-v: Large language diffusion models with visual instruction tuning. arXiv preprint arXiv:2505.16933, 2025. 1, 2, 4, 6, 7

[62] Runpeng Yu, Xinyin Ma, and Xinchao Wang. Dimple: Discrete diffusion multimodal large language model with parallel decoding. arXiv preprint arXiv:2505.16990, 2025. 1

[63] Xiang Yue, Tianyu Zheng, Yuansheng Ni, Yubo Wang, Kai Zhang, Shengbang Tong, Yuxuan Sun, Botao Yu, Ge

Zhang, Huan Sun, et al. Mmmu-pro: A more robust multidiscipline multimodal understanding benchmark. arXiv preprint arXiv:2409.02813, 2024. 4, 5, 1

[64] Yuan Zhang, Chun-Kai Fan, Junpeng Ma, Wenzhao Zheng, Tao Huang, Kuan Cheng, Denis Gudovskiy, Tomoyuki Okuno, Yohei Nakata, Kurt Keutzer, et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417, 2024. 2, 3, 4, 5, 6

[65] Yuanhan Zhang, Jinming Wu, Wei Li, Bo Li, Zejun Ma, Ziwei Liu, and Chunyuan Li. Llava-video: Video instruction tuning with synthetic data. arXiv preprint arXiv:2410.02713, 2024. 2

[66] Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592, 2023. 3

[67] Fengqi Zhu, Rongzhen Wang, Shen Nie, Xiaolu Zhang, Chunwei Wu, Jun Hu, Jun Zhou, Jianfei Chen, Yankai Lin, Ji-Rong Wen, et al. Llada 1.5: Variance-reduced preference optimization for large language diffusion models. arXiv preprint arXiv:2505.19223, 2025. 1

# A Comprehensive Study on Visual Token Redundancy for Discrete Diffusion-based Multimodal Large Language Models

Supplementary Material

## 1. Experimental Details

## 1.1. Benchmarks

We conduct our experiments across a diverse suite of multimodal benchmarks. For image understanding, we evaluate on ten datasets: MME [11], SQA [35], GQA [16], POPE [26], MMB [32], TVQA [44], CQA [37], MMMUP [63], IVQA [39], and DVQA [38]. For video understanding, we further assess performance on two benchmarks: VMME [12] and Video Detail Caption [34].

MME. MME is a comprehensive benchmark evaluating multimodal models on 14 perception and cognition subtasks, including OCR, counting, spatial localization, and visual recognition of scenes, landmarks, and artworks. All tasks are formulated as binary judgment questions with curated instruction–answer pairs to ensure fairness. We report the standard perception score on 2,374 image–question pairs.

SQA. ScienceQA evaluates multimodal reasoning and zero-shot generalization in scientific domains. It covers natural, language, and social sciences, with questions organized hierarchically across multiple topics and skills. Each question is a multiple choice question, often paired with an illustrative image. We evaluate on the image dataset of 2,017 question–answer pairs.

GQA. GQA evaluates structured visual reasoning using images, scene graphs, and automatically generated questions. Each image is paired with a scene graph from the Visual Genome dataset [20], providing detailed objects, attributes, and relations. We follow standard protocol and report accuracy on the test-dev set with 12,578 image–question pairs.

POPE. POPE evaluates object hallucination in vision–language models using binary questions about object presence in images from the MSCOCO dataset [28]. Performance is measured by the average F1 score over three sampling strategies, covering 8,910 image–question pairs.

MMB. MMBench provides a hierarchical evaluation of multimodal understanding across three levels—perception and reasoning (L1), six sub-skills (L2), and 20 tasks (L3)—each formulated as multiple-choice questions. It is available in English and Chinese versions, containing 4,377 and 4,329 image–question pairs, respectively. We evaluate on MMBench-EN subset.

TVQA. TextVQA benchmarks VQA models that must read and reason over text in natural images. It comprises 45,336 questions on 28,408 images (from text-rich Open Images categories), with 10 human answers per question. We follow the standard setting and evaluate accuracy on this dataset.

CQA. ChartQA benchmarks question answering over chart images that require both visual and logical reasoning. It includes 9,608 human-written questions and 23,111 questions generated from chart summaries, spanning 20,882 real-world charts collected from Statista, Pew Research, Our World in Data, and the OECD. Answers are often openvocabulary and may involve arithmetic or comparisons. We follow the dataset’s official evaluation protocol.

MMMUP. MMMU-Pro is a strengthened version of MMMU that aims to test genuine multimodal understanding and reasoning. It (i) filters out items solvable by textonly models, (ii) augments candidate options, and (iii) introduces the vision-only input setting MMMU-Pro Vision, where questions and options are embedded directly into images so models must truly “see” and read. In our experiments, we report results on the Vision subset following the paper’s protocol.

IVQA. InfoVQA evaluates VQA on infographics that require joint reasoning over layout, embedded text, graphical elements, and data visualizations. The dataset contains 5,485 images with 30,035 questions; answers are mainly extractive, with some numerical ones derived via counting, sorting, or simple arithmetic. We follow the official protocol and report accuracy.

DVQA. DocVQA focuses on question answering over real document images that require both reading and layout understanding. The dataset contains 12,767 document images of varied types and content, paired with about 50,000 human-annotated question–answer pairs. Each question involves information extraction, reasoning across text blocks, or interpreting document structure. We follow the dataset’s official evaluation protocol.

VMME. VideoMME is a large-scale benchmark for evaluating video understanding in LVLMs. It includes 900 videos (≈254 hours) from six domains and 30 subcategories, covering short (≤2 min), medium (4–15 min), and long (30–60 min) durations. Each video has three expert-authored multiple-choice questions, yielding 2,700 video–question pairs. We evaluate on the full dataset.

VDC. Video Detail Caption is a video captioning benchmark released by LMMs-Lab, where each video clip is paired with a detailed textual description. The test set contains 499 samples, each including a video name, a question prompt, and an answer paragraph. We follow the official evaluation protocol and assess performance using GPT-4omini as the evaluator.

## 1.2. Backbone Models

LLaDA-V. LLaDA-V [61] represents the pure diffusion paradigm in multimodal large language modeling. It extends the LLaDA diffusion language backbone with a SigLIP-2 vision encoder and a lightweight MLP projector, enabling multimodal understanding entirely through masked diffusion rather than next-token prediction. As a purely diffusion-trained model, LLaDA-V exemplifies nonautoregressive probabilistic reasoning and demonstrates strong scalability across image, document, and video understanding benchmarks.

LaViDa-Dream. LaViDa-Dream [25] represents the autoregressive-to-diffusion adaptation paradigm. It builds on Dream-7B, a discrete diffusion language model (DLM) adapted from autoregressive pretraining, and extends it to the multimodal setting through visual instruction tuning. By incorporating techniques such as complementary masking and Prefix-DLM caching, LaViDa-Dream achieves efficient multimodal reasoning while exemplifying the AR-todiffusion adaptation route in dMLLMs.

## 1.3. Token Compression Methods

ToMe. ToMe [4] is a training-free efficiency method that accelerates inference by merging similar tokens instead of pruning them. It computes pairwise similarity between attention keys and merges the most redundant token pairs during encoding through a fast bipartite matching algorithm.

DivPrune. DivPrune [1] formulates visual token pruning as a diversity-driven token selection problem. It defines a min–max diversity objective, encouraging the retained tokens to be maximally dissimilar to each other, and applies a greedy selection strategy to iteratively preserve the most informative and diverse subset of visual tokens.

<table><tr><td>Benchmark(s)</td><td>gen_length</td><td>block_length</td><td>gen_steps</td><td>think_mode</td></tr><tr><td colspan="5">LLaDA-V [61]</td></tr><tr><td>MME, SQA, GQA, POPE, MMB, TVQA, MMMUP, VMME</td><td>2</td><td>2</td><td>2</td><td>no_think</td></tr><tr><td>CQA</td><td>16</td><td>16</td><td>8</td><td>no_think</td></tr><tr><td>DVQA, IVQA</td><td>32</td><td>32</td><td>16</td><td>no_think</td></tr><tr><td>VDC</td><td>128</td><td>128</td><td>64</td><td>think</td></tr><tr><td colspan="5">LaViDa-Dream [25]</td></tr><tr><td>MME, SQA, GQA, POPE, MMB, TVQA, MMMUP</td><td>4</td><td>4</td><td>2</td><td>no_think</td></tr><tr><td>CQA</td><td>16</td><td>16</td><td>8</td><td>no_think</td></tr><tr><td>DVQA, IVQA</td><td>32</td><td>32</td><td>16</td><td>no_think</td></tr></table>

Table 1. Generation hyperparameters used for different benchmarks. The settings largely follow the default configurations of the respective backbone models, with minor adjustments to ensure stable decoding across short- and long-answer tasks.

FastV. FastV [5] is a training-free method that accelerates vision–language models by pruning redundant visual tokens in the early decoding stage. It removes the least informative tokens after the second LLM layer based on averaged attention scores.

VTW. VTW [29] is a training-free acceleration method that withdraws all visual tokens after a specific transformer layer to reduce inference cost in vision–language models. The withdrawal layer is chosen via a KL divergence criterion, enabling VTW to cut FLOPs and memory usage by over 40% without significant performance degradation.

SparseVLM. SparseVLM [64] introduces adaptive crossmodal sparsity to reduce redundancy in both visual and textual tokens. It ranks token importance via cross-modal attention, dynamically applies different sparsity ratios to vision and language streams, and employs a token recycling mechanism that reuses informative pruned tokens to preserve contextual completeness.

TRIM. TRIM [41] is a training-free token reduction method that measures text–image similarity in the CLIP representation space to rank visual tokens. It selects important tokens via an IQR-based threshold and appends an aggregated representation of unselected tokens.

## 2. Generation Hyperparameters

We summarize in Table 1 the generation hyperparameters used in our experiments for both the LLaDA-V and LaViDa-Dream backbone models. The generation settings generally follow the default configurations provided in the original model implementations, with minor adjustments to ensure stable decoding across short- and long-answer tasks. Specifically, for LaViDa-Dream, the parameters gen length, block length, and gen steps are set to 4, 4, and 2, respectively, for the group of benchmarks including MME, ScienceQA, and GQA to ensure decoding stability.