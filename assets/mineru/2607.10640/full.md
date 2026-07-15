![](images/6d94576bcd25d11b3d60e7409cb511be689210de156dba06f4af8b0fabb13850.jpg)

# Spectral Heat Flow for Conservative Token Condensation in Vision-Language Models

Zhaoyang Li <sup>\*</sup> <sup>1</sup> Yanjun Li <sup>\*</sup> <sup>1</sup> Wangkai Li <sup>1</sup> Yujia Chen <sup>1</sup> Tianzhu Zhang <sup>1</sup>

## Abstract

Vision-Language Models (VLMs) are costly at inference time because they must process long sequences of visual tokens. Existing token pruning methods often degrade under high compression by blindly discarding information, breaking spatial structure or collapsing diversity. We propose SpecFlow, a training-free framework that shifts the paradigm from destructive pruning to conservative condensation, strictly enforcing spatial coverage and statistical conservation to ensure stability. Treating visual tokens as nodes in a kNN graph, SpecFlow (i) computes a stable importance field via spectral heat flow to preserve structural coherence, (ii) allocates budgets via adaptive spatial partitioning to guarantee coverage, and (iii) aggregates discarded information into coreset sinks to maintain statistical conservation. The method is plug-and-play, requires no fine-tuning, and is compatible with FlashAttention. Experiments confirm that our SpecFlow outperforms SOTA methods across tasks, VLM architectures, and pruning ratios. Notably, LLaVA-1.5 with SpecFlow retains 95.6% of original performance despite pruning 88.9% of visual tokens, offering an exceptional efficiency-accuracy balance. Code is available at https://github. com/Lzy-dot/SpecFlow.

## 1. Introduction

Recent Vision-language models (VLMs) (Li et al., 2025a; Team et al., 2023; Liu et al., 2024c; Chen et al., 2024c) have shown remarkable progress in multimodal understanding, delivering strong results on visual question answering (Guo et al., 2023; Zhao et al., 2024; Huynh et al., 2025; Kuang et al., 2025), image captioning (Ghandi et al., 2023; Chen et al., 2024a), and video understanding (Lin et al., 2024; Maaz et al., 2024; Wang et al., 2024b).Despite this progress, efficient deployment remains challenging due to the large number of visual tokens produced by modern vision encoders. For example, LLaVA-1.5 (Liu et al., 2024a) typically encodes a 336 × 336 image into 576 patch tokens, which are then processed by the language model during the prefill stage, often resulting in substantially higher latency than text-only inference.

![](images/7e97049257fe40af6058c61bdc85be315b0c08b3f3710ae1ed0c958dbe3213c9.jpg)

![](images/70bcffd19d7f08c7d76b60eed831fd68aa6334b778cb1ae7f6ba65d375af8cdb.jpg)  
Figure 1. Global Top-K pruning vs. our proposed SpecFlow. (a) Global Top-K pruning based on [CLS] attention can yield fragmented selections due to spiky attention distributions, often creating spatial holes within objects. (b) SpecFlow diffuses attentionderived energy on a kNN token graph and applies coverage-aware regional budgeting, yielding region-coherent retained tokens under high compression.

A natural approach to accelerate inference is to reduce the number of visual tokens. Existing token pruning methods (Chen et al., 2024b; Yang et al., 2025; Zhang et al., 2024b; Zou et al., 2025) commonly perform token pruning via Top-K selection using attention-derived scores. While effective in reducing computation, such point-wise ranking can be brittle under high compression because it disregards two properties of visual token sets: spatial coherence (tokens corresponding to a region tend to form contiguous structures) and contextual dependency (background and surrounding regions can be essential for reasoning). Empirically, these limitations manifest as two recurring failure modes. First, attention scores can be highly concentrated, so Top-K selection often yields spatially fragmented token subsets, creating “holes” within objects and disrupting coherent visual evidence (Figure 1 (a)). Second, saliencydriven pruning tends to over-preserve foreground regions while aggressively removing contextual background, which can impair tasks that rely on spatial relations and scene-level semantics.

These observations suggest that token pruning should not be treated solely as independent, discrete selection. Instead, an effective pruning rule should respect the collective structure of visual tokens: importance should vary smoothly across coherent regions, while allocation should maintain coverage to avoid systematic blind spots. Motivated by this, we reformulate visual token pruning as an importance propagation problem on a token graph (Figure 1 (b)).

We propose SpecFlow, a training-free framework for efficient VLM inference that performs structure-preserving token pruning via spectral heat diffusion (Coifman & Lafon, 2006). SpecFlow constructs a graph over visual tokens and computes a stable importance field by propagating initial saliency through spectral heat flow, which naturally smooths spiky attention into region-consistent importance. To prevent spatial collapse under high compression, SpecFlow further allocates per-region budgets using adaptive quadtree partitioning, enforcing explicit spatial coverage. Finally, rather than discarding information, SpecFlow conservatively aggregates pruned tokens into coreset “sinks”, preserving summary statistics and diversity among the retained representations. SpecFlow is plug-and-play, requires no finetuning, and is compatible with efficient attention implementations such as FlashAttention (Dao et al., 2022; Dao, 2023).

In summary, our contributions to the community include:

1. We characterize two common failure modes of attention-score-based visual token pruning for VLMs: spatial fragmentation and loss of contextual evidence, especially under high compression.

2. We introduce spectral heat flow on a kNN token graph as a training-free importance propagation mechanism, producing a stable and region-coherent importance field for pruning.

3. We propose coverage-aware pruning via adaptive quadtree partitioning with energy-proportional budget allocation, and a conservative condensation step that aggregates removed tokens into a single coreset sink (with an additional diversity-preserving anchor token).

## 2. Related Work

## 2.1. Vision-Language Models

Recent progress in large language models has spurred visionlanguage models (VLMs) that transfer strong language capabilities to multimodal understanding and generation. Many recent VLMs (Liu et al., 2024a; Bai et al., 2023; Li et al., 2025b;c; Liu et al., 2024c; Lin et al., 2024; Liu et al., 2024b) improve accuracy by increasing the visual token budget through higher image resolution or multiple frames. However, the resulting visual sequences grow rapidly in length and introduce substantial computational overhead. For example, LLaVA-1.5 (Liu et al., 2024a) encodes a $3 3 6 \times 3 3 6$ image into 576 tokens, and LLaVA-NeXT (Liu et al., 2024b) scales to 2,880 tokens at $6 7 2 \times 6 7 2$ . Video models such as VideoLLaVA (Lin et al., 2024) and VideoPoet (Kondratyuk et al., 2023) further require thousands of tokens to represent multiple frames. Moreover, simply increasing the token budget does not fully eliminate visual deficiencies and hallucinations, while further amplifying inference cost. These challenges motivate token-efficient visual representations and sparsification methods that substantially reduce computation while maintaining comparable performance.

## 2.2. Visual Token Compression for VLMs

Visual token compression is crucial for scaling VLMs to high-resolution images and long videos under limited compute and context. Prior work mainly follows two routes. The first is pre-LLM compression, which reduces visual tokens before they enter the decoder. For example, LLaMA-VID (Li et al., 2024) adopts a compact frame representation, and VisionZip (Yang et al., 2025) selects dominant tokens using vision-encoder attention and can merge the remaining tokens. The second is decoder-stage sparsification, which prunes visual tokens during LLM inference based on their diminishing utility in deeper layers. In this category, FastV (Chen et al., 2024b) prunes low-importance visual tokens after an early layer using attention-based ranking in a plug-and-play manner, while SparseVLM (Zhang et al., 2024b) further introduces training-free, text-guided pruning with adaptive per-layer sparsity. Despite their effectiveness, training-free pruning often relies on spiky attention scores and some methods are not compatible with FlashAttention (Dao et al., 2022; Dao, 2023). Our SpecFlow instead diffuses saliency to produce region-consistent importance and adopts a FlashAttention-compatible pruning design.

## 3. Preliminary and Motivation

## 3.1. Preliminary

VLM inference and computation complexity. We consider a VLM $\mathcal { M } _ { \theta }$ composed of a vision encoder $f _ { \mathrm { v i s } }$ vision–language projection module $^ { g , }$ and an autoregressive decoder $f _ { \mathrm { l m } }$ with L Transformer layers. Given an image I and a text prompt $X = ( x _ { 1 } , \dots , x _ { T } )$ , the vision encoder produces patch features $V = f _ { \mathrm { v i s } } ( I ) \in \mathbb R ^ { N \times d _ { v } }$ , which are mapped to language-aligned visual tokens $Z = g ( V ) \in$ $\mathbb { R } ^ { N \times d }$ . The decoder processes the concatenated sequence $[ Z ; E ( X ) ]$ during prefill and then generates outputs autoregressively. The total decoder input length is $n = N + T$ where $T = n _ { \mathrm { s y s } } + n _ { \mathrm { q } }$ denotes the number of text tokens including the system prompt and the user query. Since Transformer self-attention in prefill scales quadratically (Vaswani et al., 2017) with the sequence length, the dominant cost is

$$
\mathcal {C} _ {\mathrm{prefill}} = \mathcal {O} \big (L \cdot n ^ {2} \cdot d \big),\tag{1}
$$

where d is the hidden dimension. In typical VLM settings, $N \gg T .$ , so computation is largely dominated by visual tokens, making the reduction of $N$ essential for improving VLM inference efficiency.

Graph diffusion for token importance. Let $G = ( \nu , \mathcal { E } )$ be a graph over N tokens and let $W \in \mathbb { R } ^ { N \times N }$ be a nonnegative row-stochastic matrix. Given an initial importance signal $e ^ { ( 0 ) } \in \mathbb { R } ^ { N }$ , a standard way to incorporate neighborhood consistency is the following diffusion update:

$$
e ^ {(t + 1)} = (1 - \alpha) e ^ {(0)} + \alpha W ^ {\top} e ^ {(t)},\tag{2}
$$

where $\alpha \in ( 0 , 1 )$ controls the propagation strength. This process is closely related to personalized PageRank and yields a smoothed importance estimate (Page et al., 1999; Jeh & Widom, 2003).

Proposition 3.1 (Restart diffusion as Dirichlet-regularized smoothing). Assume the transition matrix W is obtained by row-normalizing a symmetric affinity matrix $\mathbf { S } \in \mathbb { R } ^ { N \times N }$ i.e., $W = D ^ { - 1 } { \bf S }$ where $\mathbf { S } = \mathbf { S } ^ { \top } \overset { } { \geq } 0$ and $D = \mathrm { d i a g } ( \mathbf { S 1 } )$ . Then the diffusion in $E q . \ ( 2 )$ converges to a unique fixed point $e ^ { \star }$ satisfying

$$
(\mathbf {I} - \alpha W ^ {\top}) e ^ {\star} = (1 - \alpha) e ^ {(0)}.\tag{3}
$$

Moreover, $e ^ { \star }$ is equivalently characterized as the unique minimizer of a convex objective that balances fidelity to the seed and graph smoothness:

$$
\begin{array}{l} e ^ {\star} = \arg \min _ {e \in \mathbb {R} ^ {N}} \frac {1}{2} \big \| D ^ {- 1} e - D ^ {- 1} e ^ {(0)} \big \| _ {D} ^ {2} \\ \qquad + \frac {\alpha}{2 (1 - \alpha)} (D ^ {- 1} e) ^ {\top} (D - \mathbf {S}) (D ^ {- 1} e). \end{array}\tag{4}
$$

where $\| u \| _ { D } ^ { 2 } = u ^ { \top }$ Du and (D − S) is the (combinatorial) graph Laplacian.

Proof and additional discussion are deferred to $\mathsf { A p - }$ pendix A.1. Eq. (4) shows that diffusion is not a heuristic (Zhu et al., 2003; Zhou et al., 2003): it is the solution of a convex problem trading off seed fidelity and Dirichlet energy on the token graph.

Token graph and transition matrix. Given token embeddings $\bar { Z = } [ z _ { 1 } , \ldots , z _ { N } ] ^ { \top } \in \mathbb { R } ^ { N \times d }$ , define a k-nearestneighbor (kNN) graph under cosine similarity by assigning to each node i a neighbor set $\mathcal { N } _ { k } ( i )$ . A sparse random-walk (transition) matrix (Von Luxburg, 2007) $W \in \mathbb { R } ^ { N \times N }$ is obtained by softmax normalization over outgoing edges,

$$
W _ {i j} = \left\{ \begin{array}{l l} \frac {\exp \big (\tau \operatorname{sim} (z _ {i} , z _ {j}) \big)}{\sum_ {j ^ {\prime} \in \mathcal {N} _ {k} (i)} \exp \big (\tau \operatorname{sim} (z _ {i} , z _ {j ^ {\prime}}) \big)}, & j \in \mathcal {N} _ {k} (i), \\ 0, & \text {otherwise}, \end{array} \right.\tag{5}
$$

where sim $\begin{array} { r } { ( z _ { i } , z _ { j } ) = \frac { z _ { i } ^ { \top } z _ { j } } { \| z _ { i } \| _ { 2 } \| z _ { j } \| _ { 2 } } } \end{array}$ and $\tau > 0$ is a temperature. By construction, $\dot { W }$ is nonnegative and row-stochastic $( W \mathbf { 1 } = \mathbf { 1 } )$ , and thus admits the standard interpretation of one-step transition probabilities on the token graph. This matrix serves as the propagation operator in Eq. (2). <sup>1</sup>

## 3.2. Why Diffuse on a Feature-Similarity Graph Rather Than Raw Attention

Although attention maps provide a convenient training-free saliency cue, we do not use raw patch↔patch self-attention as the structural graph for diffusion. The reason is that in CLIP-style ViTs, raw self-attention is often not a reliable semantic affinity: it can connect tokens across inconsistent regions and does not preserve region coherence, as discussed in SCLIP (Wang et al., 2024a). Moreover, SC-CLIP (Bai et al., 2025) shows that anomaly tokens can emerge in deep layers and absorb attention mass from normal tokens, weakening spatial awareness and inducing feature homogenization. Therefore, using raw attention as a diffusion operator would propagate these artifacts, causing energy leakage to unrelated regions and reducing region-level discriminability under high compression. Accordingly, instead of diffusing on raw patch-to-patch attention links, we construct a kNN graph from token feature similarity and perform diffusion on this similarity-induced graph to preserve region coherence.

## 4. Method

Building on the above analysis, we propose SpecFlow, an efficient visual token pruning framework designed for highperformance vision-language modeling. SpecFlow leverages HeatFlow and adaptive quadtree allocation to better preserve the holistic contextual information and structural coherence of images. To avoid information loss, pruned tokens are compressed into compact coreset-style sink tokens to ensure statistical conservation.

## 4.1. HeatFlow: CLS-seeded energy diffusion

We instantiate the diffusion in Eq. (2) on the kNN graph in Eq. (5) to obtain a structure-aware token energy used for pruning. Algorithm 1 summarizes the overall procedure. We initialize the token energy from the attention distribution of the visual [CLS] token. In transformer-based vision encoders, [CLS] serves as a global aggregation token (Dosovitskiy, 2020; Radford et al., 2021) whose representation is optimized to summarize image-level semantics. Therefore, the outgoing attention weights from [CLS] provide a lightweight, training-free cue of which visual tokens contribute most to the global representation. Compared to local heuristics or post-hoc gradients, this signal is readily available during the forward pass and is naturally aligned with the model’s internal routing of information. Formally, let $A \in \mathbb { R } ^ { ( N + 1 ) \times ( N + 1 ) }$ be the self-attention matrix at a chosen layer/head, where index 0 corresponds to [CLS] and indices $1 , \ldots , N$ correspond to visual tokens. We define the initial energy as

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 CLS-seeded heat flow on a kNN token graph

input self-attention A, token embeddings Z
output diffused energy E

1: for b = 1 to B do

2: Seed from CLS attention. Select heads H and set  $s_{i} \leftarrow \frac{1}{|\mathcal{H}|} \sum_{h \in \mathcal{H}} A_{b,h,0,i}$  for  $i = 1, \ldots, N$ 

3:  $s \leftarrow s / (\mathbf{1}^{\top} s + \varepsilon)$ 

4: Construct token transition matrix. Form the row-stochastic matrix W from  $\{z_{b,i}\}_{i=1}^{N}$  by Eq. (5)

5:  $e \leftarrow s$ 

6: for t = 1 to  $T_{diff}$  do

7:  $e \leftarrow (1 - \alpha)s + \alpha W^{\top}e$ 

8:  $e \leftarrow e / (\mathbf{1}^{\top} e + \varepsilon)$ 

9: end for

10:  $E_{b} \leftarrow e$ 

11: end for

12: return E

Algorithm 2 Quadtree token pruning with energy allocation

input grid tokens Z (size  $H \times W$ ), diffused energy  $e^{\star}$ , budget K

output selected tokens  $\tilde{Z}$ 

1: for b = 1 to B do

2:  $E \leftarrow \text{Reshape}(e_{b}^{\star}, H, W)$ 

3: Quadtree splitting. Starting from  $(0, H, 0, W)$ , recursively split a crop c into four quadrants whenever split(c) holds (Eq. (8)); denote the resulting leaf crops by C

4: Energy-guided allocation. For each leaf crop  $c \in C$ , compute its energy mass S(c) from E as in Eq. (9) and initialize an integer token quota  $q_{c}$  by Eq. (13); adjust  $\{q_{c}\}$  to satisfy  $\sum_{c \in C} q_{c} = K$ 

5: Token selection.  $S \leftarrow \bigcup_{c \in C} TopK(E[c], q_{c})$  and  $\tilde{Z}_{b} \leftarrow Z_{b}[S]$ 

6: optionally: append sink tokens (mean and residual) from  $Z_{b}[\bar{S}]$  (Sec. 4.3)

7: end for
</div>

$$
e _ {i} ^ {(0)} = A _ {0, i}, \quad i = 1, \ldots , N,\tag{6}
$$

optionally aggregating multiple heads to reduce headspecific noise. This initialization highlights globally relevant regions while remaining computationally negligible, making it a suitable seed for the subsequent graph diffusion in Eq. (2). Starting from $e ^ { ( 0 ) }$ , we run a small number of iterations of Eq. (2) to obtain $e ^ { ( T ) }$ , and use it as the final energy $e ^ { \star }$ after normalization. Since W is k-sparse, the update costs $\mathcal { O } ( N k )$ per iteration. We $\ell _ { 1 }$ -normalize e after each iteration only for numerical stability, since W is rowstochastic and $e ^ { ( 0 ) }$ is normalized, the diffusion preserves mass in exact arithmetic (Appendix A.1).

## 4.2. Quadtree token pruning with energy allocation

Given the diffused token energy $e ^ { \star } \in \mathbb { R } ^ { N }$ from Sec. 4.1, we select a compact subset of visual tokens under a budget K while preserving spatial coverage. Since tokens are arranged on an $H \times W$ grid, we first reshape the energy into a 2D map $E \in \mathbb { R } ^ { H \times W }$ and then perform adaptive quadtree partitioning (Samet, 1984). Intuitively, we split regions with

$$
\tilde {Z}
$$

high energy variation into finer crops and keep low-variation regions coarse, enabling multi-scale selection. Algorithm 2 summarizes the overall pruning and allocation procedure.

Energy map. Let $E = { \mathrm { R e s h a p e } } ( e ^ { \star } , H , W )$ denote the energy map aligned with the token grid. For any rectangular crop $c = ( r _ { 0 } , r _ { 1 } , t _ { 0 } , t _ { 1 } )$ , we use $E [ c ]$ to denote the set of energies inside c and $| c | = ( r _ { 1 } - r _ { 0 } ) ( t _ { 1 } - t _ { 0 } )$ its area.

Quadtree splitting. Starting from the full region $c _ { \mathrm { r o o t } } =$ (0, H, 0, W ), we recursively split a crop into four quadrants if it is (i) sufficiently large and (ii) non-uniform in energy. Concretely, we compute the crop mean and standard deviation

$$
\begin{array}{l} \mu (c) = \frac {1}{| c |} \sum_ {(r, t) \in c} E _ {r, t}, \\ \sigma (c) = \sqrt {\frac {1}{| c |} \sum_ {(r , t) \in c} \left(E _ {r , t} - \mu (c)\right) ^ {2}}. \end{array}\tag{7}
$$

and apply the split criterion

$$
\operatorname{split} (c) = \bigl (h (c) \geq 2 m \bigr) \wedge \bigl (w (c) \geq 2 m \bigr) \wedge \bigl (\sigma (c) > \delta \bigr),\tag{8}
$$

where $h ( c ) = r _ { 1 } - r _ { 0 }$ and $w ( c ) = t _ { 1 } - t _ { 0 }$ are the crop height/width, m is the minimum crop size, and δ controls the sensitivity to energy variation. If split(c) is true, we partition c at midpoints $r _ { m } = \lfloor ( r _ { 0 } + r _ { 1 } ) / 2 \rfloor$ and $t _ { m } =$ $\lfloor ( t _ { 0 } + t _ { 1 } ) / 2 \rfloor$ to obtain four children crops; otherwise c becomes a leaf. We denote the set of all leaf crops by C, which forms a disjoint cover of the grid.

Energy-guided budget allocation. Token budget is distributed across leaf crops in proportion to their energy mass. For each $c \in { \mathcal { C } }$ , define

$$
M (c) = \sum_ {(r, t) \in c} E _ {r, t},\tag{9}
$$

and let $| c |$ denote the number of tokens in crop c. A fractional quota is given by

$$
\hat {q} _ {c} = K \cdot \frac {M (c)}{\sum_ {c ^ {\prime} \in \mathcal {C}} M (c ^ {\prime})}.\tag{10}
$$

This proportional rule admits an optimization interpretation: Proposition 4.1 (Energy-proportional allocation as proportional fairness). Assume $M ( c ) > 0$ for all $c \in { \mathcal { C } } .$ . The fractional quota $\{ \hat { q } _ { c } \} _ { c \in \mathcal { C } }$ defined by

$$
\hat {q} _ {c} = K \cdot \frac {M (c)}{\sum_ {c ^ {\prime} \in \mathcal {C}} M (c ^ {\prime})}\tag{11}
$$

is the unique optimizer of the concave program (Kelly et al., 1998)

$$
\max _ {\{q _ {c} > 0 \}} \sum_ {c \in \mathcal {C}} M (c) \log q _ {c} \quad s. t. \quad \sum_ {c \in \mathcal {C}} q _ {c} = K.\tag{12}
$$

We provide the proof and integer rounding details in Appendix A.2.

An initial integer allocation is obtained by flooring and clipping to the feasible range,

$$
q _ {c} \leftarrow \min \bigl (| c |, \lfloor \hat {q} _ {c} \rfloor \bigr),\tag{13}
$$

which allows $q _ { c } ~ = ~ 0$ when the budget is smaller than the number of crops. Since flooring and clipping imply $\textstyle \sum _ { c \in c } q _ { c } \leq K$ , the remaining budget $K - \textstyle \sum _ { c } q _ { c }$ is distributed to crops in descending order of $S ( c )$ among those with $q _ { c } < | c |$ , without exceeding |c|, until $\textstyle \sum _ { c \in c } q _ { c } = K$ Overall, the procedure enforces the global budget while maintaining $0 \leq q _ { c } \leq | c |$

Token selection within each crop. Within each crop c, the $q _ { c }$ highest-energy tokens are retained:

$$
\mathcal {S} = \bigcup_ {c \in \mathcal {C}} \operatorname{TopK} \left(E [ c ], q _ {c}\right), \quad | \mathcal {S} | = K.\tag{14}
$$

The pruned token sequence is then $\tilde { Z } = Z [ S ]$ , where indices in S follow the original token order.

Discussion. The quadtree split enforces spatial adaptivity: salient regions (high-variation energy) are represented at a finer granularity, while background regions remain coarse. Coupled with energy-proportional allocation, this yields an efficient subset that preserves both semantic relevance (via E) and coverage (via the multi-scale partition).

## 4.3. Sink tokens for preserving pruned information

Quadtree pruning keeps a budgeted subset of tokens $s$ but discards the complement S<sup>¯</sup>. While tokens in $\bar { \boldsymbol { S } }$ are assigned lower energy, they may still contain complementary context (e.g., background cues or fine-grained attributes). To mitigate information loss with negligible overhead, we compress the discarded set into a small number of additional sink tokens and feed them together with the selected tokens to the decoder. Let $\mathbf { Z } \in \bar { \mathbb { R } } ^ { N \times d }$ denote the token features, $\tilde { Z } = Z [ S ]$ the selected tokens, and $Z _ { \mathrm { p } } = Z [ \bar { S } ]$ the pruned tokens. We summarize $Z _ { \mathrm { p } }$ using two coreset-style sinks (Feldman & Langberg, 2011):

$$
u _ {\mathrm{mean}} = \frac {1}{| \bar {\mathcal {S}} |} \sum_ {i \in \bar {\mathcal {S}}} Z _ {i}, \qquad u _ {\mathrm{res}} = Z _ {\arg \max _ {i \in \bar {\mathcal {S}}} \| Z _ {i} - u _ {\mathrm{mean}} \| _ {2}}.\tag{15}
$$

The mean sink captures the overall context of discarded tokens, while the residual sink promotes diversity by retaining a representative token that is poorly explained by the mean. We append the sink tokens to the selected tokens to form the final visual token sequence

$$
Z ^ {\prime} = [ \tilde {Z}; u _ {\mathrm{mean}}; u _ {\mathrm{res}} ],\tag{16}
$$

which is passed to the language model.

## 4.4. Computational Complexity

We express inference cost in FLOPs and concentrate on visual tokens, as the text prompt is usually much shorter than the visual sequence $( \mathsf { i . e . , } T \ll N )$ . Let N denote the number of visual tokens produced by the encoder and K the number kept after pruning. Denote the hidden size by h and the FFN intermediate size by d. Coefficients a, b, c are implementation-dependent constants.

Prefill FLOPs. During prefill, the per-layer FLOPs as a function of the attended visual length n can be modeled as

$$
\Psi_ {\mathrm{pre}} (n) := a h n ^ {2} + n \left(b h ^ {2} + c h d\right),\tag{17}
$$

where the first term corresponds to attention and the latter terms capture per-token projections and the FFN. Pruning replaces $n = N$ with $n = K$ , hence the relative prefill saving is

$$
\Delta_ {\mathrm{pre}} := 1 - \frac {\Psi_ {\mathrm{pre}} (K)}{\Psi_ {\mathrm{pre}} (N)} = 1 - \frac {a h K ^ {2} + K (b h ^ {2} + c h d)}{a h N ^ {2} + N (b h ^ {2} + c h d)}.\tag{18}
$$

Let $K = ( 1 - R ) N$ . When attention dominates (typical for large $N )$ , the $n ^ { 2 }$ term governs and the reduction simplifies to

$$
\Delta_ {\mathrm{pre}} \approx 1 - \left(\frac {K}{N}\right) ^ {2} = 1 - (1 - R) ^ {2} = 2 R - R ^ {2}.\tag{19}
$$

highlighting that prefill gains are superlinear in the pruning ratio $R$ under this regime.

Table 1. Comparison of pruning methods on image-understanding benchmarks. We report the accuracy and average performance at varying pruning ratios. The best performance is marked in red.

<table><tr><td>Methods</td><td>GQA</td><td>MMB</td><td> $MMBCN$ </td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQAv2$ </td><td> $VQA_{Text}$ </td><td>VizWiz</td><td>Average</td></tr><tr><td>Upper Bound, 576 Tokens</td><td>61.9</td><td>64.7</td><td>58.1</td><td>1862</td><td>85.9</td><td>69.5</td><td>78.4</td><td>58.2</td><td>50.0</td><td>100%</td></tr><tr><td>LLaVA-1.5 (Liu et al., 2024a) 7B</td><td colspan="10">Retain 192 Tokens (↓ 66.7%)</td></tr><tr><td>ToMe (Bolya et al., 2022) (ICLR23)</td><td>54.3</td><td>60.5</td><td>-</td><td>1563</td><td>72.4</td><td>65.2</td><td>68.0</td><td>52.1</td><td>-</td><td>88.5%</td></tr><tr><td>FastV (Chen et al., 2024b) (ECCV24)</td><td>52.7</td><td>61.2</td><td>57.0</td><td>1612</td><td>64.8</td><td>67.3</td><td>67.1</td><td>52.5</td><td>50.8</td><td>90.5%</td></tr><tr><td>LLaVA-PruMerge (Shang et al., 2025) (ICCV25)</td><td>54.3</td><td>59.6</td><td>52.9</td><td>1632</td><td>71.3</td><td>67.9</td><td>70.6</td><td>54.3</td><td>50.1</td><td>91.4%</td></tr><tr><td>PDrop (Xing et al., 2024) (CVPR25)</td><td>57.1</td><td>63.2</td><td>56.8</td><td>1766</td><td>82.3</td><td>68.8</td><td>75.1</td><td>56.1</td><td>51.1</td><td>96.7%</td></tr><tr><td>HiRED (Arif et al., 2025) (AAAI25)</td><td>58.7</td><td>62.8</td><td>54.7</td><td>1737</td><td>82.8</td><td>68.4</td><td>74.9</td><td>47.4</td><td>50.1</td><td>94.6%</td></tr><tr><td>VisionZip (Yang et al., 2025) (CVPR25)</td><td>59.3</td><td>64.5</td><td>57.3</td><td>1767</td><td>86.4</td><td>68.9</td><td>76.8</td><td>57.3</td><td>51.6</td><td>98.1%</td></tr><tr><td>SparseVLM (Zhang et al., 2024b) (ICML25)</td><td>57.6</td><td>62.5</td><td>53.7</td><td>1721</td><td>83.6</td><td>69.1</td><td>75.6</td><td>56.1</td><td>50.5</td><td>96.1%</td></tr><tr><td>DART (Wen et al., 2025) (EMNLP25)</td><td>58.9</td><td>63.6</td><td>57.0</td><td>1856</td><td>82.8</td><td>69.8</td><td>76.7</td><td>57.4</td><td>51.1</td><td>98.5%</td></tr><tr><td>HoloV (Zou et al., 2025) (NeurIPS25)</td><td>58.6</td><td>64.6</td><td>56.9</td><td>1793</td><td>85.6</td><td>69.1</td><td>76.1</td><td>55.8</td><td>51.4</td><td>98.2%</td></tr><tr><td>SpecFlow (Ours)</td><td>58.3</td><td>65.8</td><td>56.1</td><td>1827</td><td>85.8</td><td>69.7</td><td>76.4</td><td>57.9</td><td>50.5</td><td>98.7%</td></tr><tr><td>LLaVA-1.5 (Liu et al., 2024a) 7B</td><td colspan="10">Retain 128 Tokens (↓ 77.8%)</td></tr><tr><td>ToMe (Bolya et al., 2022) (ICLR23)</td><td>52.4</td><td>53.3</td><td>-</td><td>1343</td><td>62.8</td><td>59.6</td><td>63.0</td><td>49.1</td><td>-</td><td>80.4%</td></tr><tr><td>FastV (Chen et al., 2024b) (ECCV24)</td><td>49.6</td><td>56.1</td><td>56.4</td><td>1490</td><td>59.6</td><td>60.2</td><td>61.8</td><td>50.6</td><td>51.3</td><td>85.4%</td></tr><tr><td>LLaVA-PruMerge (Shang et al., 2025) (ICCV25)</td><td>53.3</td><td>58.1</td><td>51.7</td><td>1554</td><td>67.2</td><td>67.1</td><td>68.8</td><td>54.3</td><td>50.3</td><td>89.4%</td></tr><tr><td>PDrop (Xing et al., 2024) (CVPR25)</td><td>56.0</td><td>61.1</td><td>56.6</td><td>1644</td><td>82.3</td><td>68.3</td><td>72.9</td><td>55.1</td><td>51.0</td><td>94.9%</td></tr><tr><td>HiRED (Arif et al., 2025) (AAAI25)</td><td>57.2</td><td>61.5</td><td>53.6</td><td>1710</td><td>79.8</td><td>68.1</td><td>73.4</td><td>46.1</td><td>51.3</td><td>93.1%</td></tr><tr><td>VisionZip (Yang et al., 2025) (CVPR25)</td><td>57.6</td><td>63.4</td><td>56.7</td><td>1768</td><td>84.7</td><td>68.8</td><td>75.6</td><td>56.8</td><td>52.0</td><td>97.2%</td></tr><tr><td>SparseVLM (Zhang et al., 2024b) (ICML25)</td><td>56.0</td><td>60.0</td><td>51.1</td><td>1696</td><td>80.5</td><td>67.1</td><td>73.8</td><td>54.9</td><td>51.4</td><td>93.8%</td></tr><tr><td>DART (Wen et al., 2025) (EMNLP25)</td><td>57.9</td><td>63.2</td><td>57.0</td><td>1845</td><td>80.1</td><td>69.1</td><td>75.9</td><td>56.4</td><td>51.7</td><td>97.5%</td></tr><tr><td>HoloV (Zou et al., 2025) (NeurIPS25)</td><td>57.4</td><td>62.8</td><td>55.0</td><td>1768</td><td>83.2</td><td>69.1</td><td>74.8</td><td>55.7</td><td>52.3</td><td>96.8%</td></tr><tr><td>SpecFlow (Ours)</td><td>57.6</td><td>64.3</td><td>55.9</td><td>1794</td><td>84.9</td><td>69.9</td><td>75.3</td><td>56.8</td><td>51.1</td><td>97.8%</td></tr><tr><td>LLaVA-1.5 (Liu et al., 2024a) 7B</td><td colspan="10">Retain 64 Tokens (↓ 88.9%)</td></tr><tr><td>ToMe (Bolya et al., 2022) (ICLR23)</td><td>48.6</td><td>43.7</td><td>-</td><td>1138</td><td>52.5</td><td>50.0</td><td>57.1</td><td>45.3</td><td>-</td><td>70.1%</td></tr><tr><td>FastV (Chen et al., 2024b) (ECCV24)</td><td>46.1</td><td>48.0</td><td>52.7</td><td>1256</td><td>48.0</td><td>51.1</td><td>55.0</td><td>47.8</td><td>50.8</td><td>76.7%</td></tr><tr><td>LLaVA-PruMerge (Shang et al., 2025) (ICCV25)</td><td>51.9</td><td>55.3</td><td>49.1</td><td>1549</td><td>65.3</td><td>68.1</td><td>67.4</td><td>54.0</td><td>50.1</td><td>87.7%</td></tr><tr><td>PDrop (Xing et al., 2024) (CVPR25)</td><td>41.9</td><td>33.3</td><td>50.5</td><td>1092</td><td>55.9</td><td>68.6</td><td>69.2</td><td>45.9</td><td>50.7</td><td>77.5%</td></tr><tr><td>HiRED (Arif et al., 2025) (AAAI25)</td><td>54.6</td><td>60.2</td><td>51.4</td><td>1599</td><td>73.6</td><td>68.2</td><td>69.7</td><td>44.2</td><td>50.2</td><td>89.4%</td></tr><tr><td>VisionZip (Yang et al., 2025) (CVPR25)</td><td>55.1</td><td>60.1</td><td>55.4</td><td>1690</td><td>77.0</td><td>69.0</td><td>72.4</td><td>55.5</td><td>52.9</td><td>94.5%</td></tr><tr><td>SparseVLM (Zhang et al., 2024b) (ICML25)</td><td>52.7</td><td>56.2</td><td>46.1</td><td>1505</td><td>75.1</td><td>62.2</td><td>68.2</td><td>51.8</td><td>50.1</td><td>87.3%</td></tr><tr><td>DART (Wen et al., 2025) (EMNLP25)</td><td>55.9</td><td>60.6</td><td>53.2</td><td>1765</td><td>73.9</td><td>69.8</td><td>72.4</td><td>54.4</td><td>51.6</td><td>93.9%</td></tr><tr><td>HoloV (Zou et al., 2025) (NeurIPS25)</td><td>55.1</td><td>61.3</td><td>53.7</td><td>1728</td><td>80.3</td><td>69.5</td><td>72.2</td><td>54.2</td><td>53.1</td><td>94.9%</td></tr><tr><td>SpecFlow (Ours)</td><td>55.3</td><td>63.7</td><td>54.3</td><td>1713</td><td>80.5</td><td>69.7</td><td>73.7</td><td>54.9</td><td>52.4</td><td>95.6%</td></tr></table>

Decode FLOPs. With KV caching, each decoding step scales approximately linearly with the context length:

$$
\Psi_ {\mathrm{dec}} (n) := b h ^ {2} + n (b h + c h d).\tag{20}
$$

Replacing N by K gives

$$
\Delta_ {\mathrm{dec}} := 1 - \frac {\Psi_ {\mathrm{dec}} (K)}{\Psi_ {\mathrm{dec}} (N)} = 1 - \frac {b h ^ {2} + K (b h + c h d)}{b h ^ {2} + N (b h + c h d)} \approx R.\tag{21}
$$

since the constant term $b h ^ { 2 }$ does not shrink with pruning.

Cost of pruning. Pruning is performed once per input; in particular, the kNN graph is computed only once before decoder-layer processing. The overhead comprises kNN graph construction (worst-case $\mathcal O ( N ^ { 2 } h )$ FLOPs), sparse diffusion $\mathcal { O } ( T _ { \mathrm { d i f f } } N k )$ , and quadtree splitting/TopK selection near-linear in N (i.e., O<sup>˜</sup>(N ), or O(DN ) with depth D). As these costs are not incurred per decoder layer, they are amortized by the per-layer savings from reducing the attended length from N to K, especially during prefill. A more detailed quantitative breakdown of the pruning overhead, end-to-end latency, and memory footprint, together with concrete numerical instantiations of the FLOPs formulas above, is provided in Appendix C.

## 5. Experiments

## 5.1. Experimental Setup

Benchmarks. We validate our approach on a diverse suite of established benchmarks for visual understanding. For image understanding task, we consider ten datasets, including GQA (Hudson & Manning, 2019), MMBench (MMB) and MMB-CN (Liu et al., 2024d), MME (Fu et al., 2023), POPE (Li et al., 2023), VizWiz (Bigham et al., 2010), ScienceQA (SQA) (Lu et al., 2022), VQA v2 (VQA ) (Goyal et al., 2017) and TextVQA $( { \mathrm { V Q A } } _ { \mathrm { T e x t } } )$ (Singh et al., 2019). For video understanding task, we additionally evaluate on MSVD-QA and MSRVTT-QA (Xu et al., 2017). We follow the official evaluation protocols and default settings for each benchmark. Further benchmark details are provided in Appendix B.1.

## 5.2. Main Results

Image Understanding Tasks. Table 1 summarizes the main results on nine image-understanding benchmarks under different token budgets. Overall, SpecFlow consistently achieves a strong accuracy–efficiency trade-off across datasets and compression regimes. At a 66.7% token compression ratio, SpecFlow attains the best average relative performance (98.7%), incurring only a 1.3% drop from the uncompressed full-token baseline (100%), while outperforming recent pruning baselines such as VisionZip, Sparse-VLM, and HoloV, and remaining competitive on individual benchmarks. Notably, on MMBench and SQA, SpecFlow even surpasses the unpruned baseline. As the token budget becomes more stringent, SpecFlow degrades gracefully and continues to deliver the best averages at 77.8% and 88.9% token compression ratios (97.8% and 95.6%, respectively).

Results on LLaVA-NeXT. For a more thorough evaluation, we further benchmark SpecFlow on LLaVA-NeXT across the same suite of datasets and compare against current state-of-the-art training-free pruning methods. Since LLaVA-NeXT adopts a revised image encoding pipeline, the number of visual tokens varies across inputs. To ensure a consistent evaluation protocol, we fix the token budget at 320 (pruned from up to 2880 raw tokens). As reported in Table 2, SpecFlow delivers the strongest overall performance, while retaining 95.8% of the unpruned baseline performance with an 88.9% reduction in visual tokens.

Video Understanding Tasks. We evaluate our method on two widely used video question-answering benchmarks. Following the protocol in (Chen et al., 2024b; Zhang et al., 2024b), we report results on the first 1,000 samples of each benchmark and use the Video-ChatGPT (Maaz et al., 2024) score as the primary metric. In Table 3, we treat Video-LLaVA with 2,048 video tokens as the unpruned upper bound, normalized to an average accuracy of 100.0% and a score difference of +0.00. For a fair comparison, all pruning methods retain 455 visual tokens (77.8% pruning ratio). Under this setting, SpecFlow preserves performance close to the unpruned baseline and consistently outperforms FastV, SparseVLM, and HoloV. These results indicate that SpecFlow remains effective for video inputs with temporal dynamics, producing accurate answers while substantially reducing token usage.

Results on Qwen2.5-VL. To assess architectural generality beyond the LLaVA family, we further evaluate SpecFlow on Qwen2.5-VL-7B, which differs from LLaVA in both the visual stack and the multimodal projector. As reported in Table 4, SpecFlow consistently outperforms HoloV across all three pruning ratios, with the margin growing under more aggressive compression (e.g., 89.4% vs. 87.3% relative accuracy at 88.9% pruning). This indicates that the proposed mechanism, which combines graph diffusion with adaptive regional allocation and sink-based context preservation, generalizes well to different MLLM architectures and is not tied to a specific CLIP-style encoder or projector design.

![](images/8f43e02c727b8d58b251111f2c14cc1cbf9c0360825628f3e3a68bf4e72ce8ec.jpg)

![](images/1b1b72ffdfd4b8edaf2091171b21aa32229b1fad32eb942ee021fe931d8b929b.jpg)

Figure 2. Ablation study on diffusion operators. Our KNN graph (built from token feature similarity) outperforms raw self-attention (Token attn) and [CLS]-initialized attention propagation ([CLS] attn).  
![](images/03d55faa3ab823058132d9b91b1913bc69d5c383a8544594c224ed3df258e7d5.jpg)

![](images/8680868402c9b1dd2645a56dd060e6ce4d772a0de2efb6ad701a35e954362104.jpg)  
Figure 3. Ablation on pruning strategies. Our Quadtree outperforms Top-K (direct selection of top-K tokens) and Uni Crop (uniformly partitioned crops), validating adaptive spatial pruning.

## 5.3. Ablation Studies

Ablation on Diffusion Operator. We validate the design choice of constructing a KNN graph based on token feature similarity by comparing it against two variants: raw visual token self-attention (Token attn) and [CLS]-initialized attention propagation ([CLS] attn). As illustrated in Figure 3, our KNN graph-based diffusion consistently outperforms both baselines. Specifically, it achieves 55.3% on GQA and 54.9% on TextVQA, surpassing the Token attn baseline by 5.7% and 4.6%, respectively. These results verify that feature-similarity kNN graphs provide a robust structural basis for diffusion, effectively preserving region coherence while mitigating the noise and artifacts inherent in raw attention maps.

Ablation on Adaptive Spatial Pruning Strategies. To assess the effectiveness of our adaptive quadtree-based pruning, we compare it against two non-adaptive strategies: Top-K, which selects high-energy tokens without spatial constraints; and Uni Crop, which employs rigid uniform partitioning. As shown in the results, our quadtree approach yields superior performance, reaching 55.3% on GQA and 54.9% on TextVQA. Critically, our method addresses the limitations of the baselines: while Top-K often results in spatial fragmentation and Uni Crop inefficiently allocates budget to low-importance backgrounds, our quadtree method adaptively refines granularity based on energy variation. This mechanism creates finer partitions in semantically salient regions to preserve detail, and coarser ones in uniform backgrounds to reduce redundancy. This design strikes an optimal balance between semantic preservation and spatial coverage, translating into significant performance gains.

(b)  
(a)  
Table 2. Comparison of pruning methods on Video QA benchmarks. We report the accuracy and average performance at 88.9% pruning ratios. The best performance is marked in red.

<table><tr><td>Methods</td><td>GQA</td><td>MMB</td><td> $MMBCN$ </td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQAv2$ </td><td> $VQA_{Text}$ </td><td>VizWiz</td><td>Average</td></tr><tr><td>Upper Bound, 2880 Tokens</td><td>64.2</td><td>67.4</td><td>60.6</td><td>1851</td><td>86.5</td><td>70.1</td><td>81.8</td><td>64.9</td><td>57.6</td><td>100%</td></tr><tr><td>LLaVA-NeXT (Liu et al., 2024b) 7B</td><td colspan="10">Retain 320 Tokens (↓ 88.9%)</td></tr><tr><td>FastV (Chen et al., 2024b) (ECCV24)</td><td>55.9</td><td>61.6</td><td>51.9</td><td>1661</td><td>71.7</td><td>62.8</td><td>71.9</td><td>55.7</td><td>53.1</td><td>88.0%</td></tr><tr><td>LLaVA-PruMerge (Shang et al., 2025) (ICCV25)</td><td>53.6</td><td>61.3</td><td>55.3</td><td>1534</td><td>60.8</td><td>66.4</td><td>69.7</td><td>50.6</td><td>54.0</td><td>85.6%</td></tr><tr><td>PDrop (Xing et al., 2024) (CVPR25)</td><td>56.4</td><td>63.4</td><td>56.2</td><td>1663</td><td>77.6</td><td>67.5</td><td>73.5</td><td>54.4</td><td>54.1</td><td>90.9%</td></tr><tr><td>FasterVLM (Zhang et al., 2024a) (ICCV25)</td><td>56.9</td><td>61.6</td><td>53.5</td><td>1701</td><td>83.6</td><td>66.5</td><td>74.0</td><td>56.5</td><td>52.6</td><td>91.1%</td></tr><tr><td>HiRED (Arif et al., 2025) (AAAI25)</td><td>59.3</td><td>64.2</td><td>55.9</td><td>1690</td><td>83.3</td><td>66.7</td><td>75.7</td><td>58.8</td><td>54.2</td><td>93.3%</td></tr><tr><td>SparseVLM (Zhang et al., 2024b) (ICML25)</td><td>56.1</td><td>60.6</td><td>54.5</td><td>1533</td><td>82.4</td><td>66.1</td><td>71.5</td><td>58.4</td><td>52.0</td><td>89.7%</td></tr><tr><td>DART (Wen et al., 2025) (EMNLP25)</td><td>61.7</td><td>65.3</td><td>58.2</td><td>1710</td><td>84.1</td><td>68.4</td><td>79.1</td><td>58.7</td><td>56.1</td><td>93.9%</td></tr><tr><td>HoloV (Zou et al., 2025) (NeurIPS25)</td><td>61.7</td><td>65.3</td><td>57.5</td><td>1738</td><td>83.9</td><td>68.9</td><td>79.5</td><td>58.7</td><td>55.3</td><td>95.6%</td></tr><tr><td>SpecFlow (Ours)</td><td>62.5</td><td>66.7</td><td>56.8</td><td>1707</td><td>85.0</td><td>68.6</td><td>80.1</td><td>59.5</td><td>54.2</td><td>95.8%</td></tr></table>

![](images/14c87f015508c8b5d5ac8f6c34fb79d90df2e299d362b29ef2862485f34a9104.jpg)

![](images/38f590d4948ee25d0e3f4d1a406d7ddbec2daa466ba8b8f26410c3851cd34a3d.jpg)  
Figure 4. Qualitative visualization of token selection and adaptive splitting. (a) Visualization of our energy-based quadtree splitting, showing the induced partitions and the final retained-token layouts across representative cases (b) Comparison of retained visual tokens at different pruning ratios for SpecFlow versus FastV and HoloV.

Table 3. Video QA Evaluations of different methods with 455 visual tokens retained.

<table><tr><td rowspan="2">Methods</td><td colspan="2">MSVD-QA</td><td colspan="2">MSRVT-QA</td><td colspan="2">Average</td></tr><tr><td>Acc.</td><td>Score</td><td>Acc.</td><td>Score</td><td>Acc.</td><td>Score</td></tr><tr><td>Video-LLaVA 7B</td><td>70.8</td><td>3.93</td><td>57.5</td><td>3.55</td><td>64.2</td><td>3.74</td></tr><tr><td>FastV (ECCV24)</td><td>68.2</td><td>3.75</td><td>54.1</td><td>3.42</td><td>61.2</td><td>3.59</td></tr><tr><td>SparseVLM (ICML25)</td><td>69.4</td><td>3.89</td><td>54.8</td><td>3.42</td><td>62.1</td><td>3.66</td></tr><tr><td>HoloV (NeurIPS25)</td><td>69.3</td><td>3.90</td><td>56.2</td><td>3.50</td><td>62.8</td><td>3.70</td></tr><tr><td>SpecFlow (Ours)</td><td>70.3</td><td>3.94</td><td>56.5</td><td>3.50</td><td>63.4</td><td>3.72</td></tr></table>

## 5.4. Discussion

SpecFlow’s advantage stems from coupling a structureaware importance signal with spatially balanced token retention. We perform energy diffusion on the token graph so that semantically similar tokens can propagate importance to each other, which mitigates the fragmentation of coherent objects under aggressive pruning and prevents a small set of peak tokens from dominating the budget. Meanwhile, the diffused energy remains sufficiently discriminative, avoid-

Table 4. SpecFlow Generalization on Qwen2.5-VL.

<table><tr><td>Methods</td><td>MME</td><td>POPE</td><td>SQA</td><td> $VQA_{Text}$ </td><td>Avg.</td></tr><tr><td>Upper Bound</td><td>2304</td><td>86.1</td><td>84.7</td><td>84.8</td><td>100.0%</td></tr><tr><td>Qwen2.5-VL-7B</td><td colspan="5">Retain 192 Tokens (↓66.7%)</td></tr><tr><td>HoloV (NeurIPS25)</td><td>2066</td><td>85.0</td><td>79.1</td><td>77.3</td><td>93.2%</td></tr><tr><td>SpecFlow (Ours)</td><td>2102</td><td>85.2</td><td>80.0</td><td>79.2</td><td>94.5%</td></tr><tr><td>Qwen2.5-VL-7B</td><td colspan="5">Retain 128 Tokens (↓77.8%)</td></tr><tr><td>HoloV (NeurIPS25)</td><td>2029</td><td>81.5</td><td>79.1</td><td>69.2</td><td>89.4%</td></tr><tr><td>SpecFlow (Ours)</td><td>2051</td><td>83.7</td><td>79.9</td><td>73.8</td><td>91.9%</td></tr><tr><td>Qwen2.5-VL-7B</td><td colspan="5">Retain 64 Tokens (↓88.9%)</td></tr><tr><td>HoloV (NeurIPS25)</td><td>1998</td><td>80.7</td><td>79.5</td><td>63.6</td><td>87.3%</td></tr><tr><td>SpecFlow (Ours)</td><td>2015</td><td>81.1</td><td>79.7</td><td>69.4</td><td>89.4%</td></tr></table>

![](images/1d19f8ed6b1e87167a7e58b60b47b6acbb89fb8bbc0292fd8a69fcf48eba3d7b.jpg)  
Figure 5. Cumulative mass (sorted) over visual token proportion for different importance measures.

ing the overly smooth behavior that text-vision attention can exhibit (Zhang et al., 2025; Zou et al., 2025), and thus better separates salient from non-salient tokens, as evidenced by the cumulative mass distribution in Figure 5. Building on this importance signal, we further introduce an energybased quadtree splitting strategy that adaptively partitions the image into regions of varying visual complexity and assigns a per-region token quota. As illustrated in Figure 4(a), this regional budgeting preserves object integrity while increasing spatial coverage, producing retained-token layouts that capture key structures across the scene instead of collapsing the budget onto a small area. Consequently, under extreme pruning ratios, SpecFlow maintains substantially better subject completeness than existing methods, as shown in Figure 4(b).

## 6. Conclusion

We presented SpecFlow, a training-free framework for efficient VLM inference that replaces destructive token pruning with conservative token condensation. SpecFlow computes a stable importance field via spectral heat diffusion on a kNN token graph, enforces spatial coverage through adaptive quadtree budgeting, and summarizes removed tokens with lightweight sink tokens to preserve context and diversity. Experiments on multiple image and video benchmarks show that SpecFlow achieves substantial visual token reduction with minimal accuracy loss, outperforming strong training-free baselines and remaining compatible with FlashAttention.

## Acknowledgments

This work was supported by the Open Fund of National Key Laboratory of Deep Space Exploration (NKDSEL2025008).

## Impact Statement

This paper presents SpecFlow, a training-free framework for efficient inference in vision-language models (VLMs) via conservative token condensation. By selecting a smaller set of region-coherent visual tokens and aggregating pruned information into lightweight sink tokens, SpecFlow can reduce the computational and memory cost of VLM inference, especially in the prefill stage. These efficiency gains may lower latency and monetary cost per query and reduce energy consumption, potentially enabling broader access to real-time and on-device VLM applications such as assistive interfaces, interactive education, and robotics.

However, improved efficiency can also lower the barrier to deploying VLMs at scale. When combined with powerful generative or retrieval systems, faster VLM inference could facilitate high-volume uses that raise ethical concerns, including privacy-invasive monitoring, large-scale content analysis, or the amplification of misleading or harmful multimodal content. While SpecFlow does not introduce new model capabilities or new training data by itself, it can increase throughput and thus the scale at which downstream systems are used. We therefore recommend that deployments preserve and strengthen existing safety measures, including content filtering, rate limiting, logging/auditing, and compliance with applicable privacy and data-protection regulations.

A further risk is that aggressive token budgets may degrade performance on some inputs (e.g., small objects, dense text, or rare visual concepts), which could be harmful in highstakes settings. We recommend validating token budgets on the target domain, monitoring failure cases, and providing fallbacks (e.g., less compression, full-token inference, or human oversight) when uncertainty is high. Finally, SpecFlow inherits limitations and potential biases of the underlying VLMs and evaluation datasets; practitioners should consider domain-specific bias and robustness testing before deployment.

## References

Arif, K. H. I., Yoon, J., Nikolopoulos, D. S., Vandierendonck, H., John, D., and Ji, B. Hired: Attention-guided token dropping for efficient inference of high-resolution vision-language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 1773– 1781, 2025.

Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-vl: A frontier large visionlanguage model with versatile abilities. arXiv preprint arXiv:2308.12966, 2023.

Bai, S., Liu, Y., Han, Y., Zhang, H., Tang, Y., Zhou, J., and Lu, J. Self-calibrated clip for training-free openvocabulary segmentation. IEEE Transactions on Image Processing, 2025.

Bigham, J. P., Jayant, C., Ji, H., Little, G., Miller, A., Miller, R. C., Miller, R., Tatarowicz, A., White, B., White, S., et al. Vizwiz: nearly real-time answers to visual questions. In Proceedings of the 23nd annual ACM symposium on User interface software and technology, pp. 333–342, 2010.

Bolya, D., Fu, C.-Y., Dai, X., Zhang, P., Feichtenhofer, C., and Hoffman, J. Token merging: Your vit but faster. arXiv preprint arXiv:2210.09461, 2022.

Boyd, S. and Vandenberghe, L. Convex optimization. Cambridge university press, 2004.

Chen, L., Li, J., Dong, X., Zhang, P., He, C., Wang, J., Zhao, F., and Lin, D. Sharegpt4v: Improving large multi-modal models with better captions. In European Conference on Computer Vision, pp. 370–387. Springer, 2024a.

Chen, L., Zhao, H., Liu, T., Bai, S., Lin, J., Zhou, C., and Chang, B. An image is worth 1/2 tokens after layer 2: Plug-and-play inference acceleration for large visionlanguage models. In European Conference on Computer Vision, pp. 19–35. Springer, 2024b.

Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 24185–24198, 2024c.

Coifman, R. R. and Lafon, S. Diffusion maps. Applied and computational harmonic analysis, 21(1):5–30, 2006.

Dao, T. Flashattention-2: Faster attention with better parallelism and work partitioning. arXiv preprint arXiv:2307.08691, 2023.

Dao, T., Fu, D., Ermon, S., Rudra, A., and Re, C. Flashat-´ tention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344–16359, 2022.

Dosovitskiy, A. An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929, 2020.

Feldman, D. and Langberg, M. A unified framework for approximating and clustering data. In Proceedings of the forty-third annual ACM Symposium on Theory of Computing, pp. 569–578, 2011.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al. MME: A comprehensive evaluation benchmark for multimodal large language models. arXiv:2306.13394, 2023.

Ghandi, T., Pourreza, H., and Mahyar, H. Deep learning approaches on image captioning: A review. ACM Computing Surveys, 56(3):1–39, 2023.

Goyal, Y., Khot, T., Summers-Stay, D., Batra, D., and Parikh, D. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6904–6913, 2017.

Guo, J., Li, J., Li, D., Tiong, A. M. H., Li, B., Tao, D., and Hoi, S. From images to textual prompts: Zero-shot visual question answering with frozen large language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10867–10877, 2023.

Horn, R. A. and Johnson, C. R. Matrix analysis. Cambridge university press, 2012.

Hudson, D. A. and Manning, C. D. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 6700– 6709, 2019.

Huynh, N. D., Bouadjenek, M. R., Aryal, S., Razzak, I., and Hacid, H. Visual question answering: from early developments to recent advances–a survey. arXiv preprint arXiv:2501.03939, 2025.

Jeh, G. and Widom, J. Scaling personalized web search. In Proceedings of the 12th international conference on World Wide Web, pp. 271–279, 2003.

Kelly, F. P., Maulloo, A. K., and Tan, D. K. H. Rate control for communication networks: shadow prices, proportional fairness and stability. Journal of the Operational Research society, 49(3):237–252, 1998.

Kondratyuk, D., Yu, L., Gu, X., Lezama, J., Huang, J., Schindler, G., Hornung, R., Birodkar, V., Yan, J., Chiu, M.-C., et al. Videopoet: A large language model for zeroshot video generation. arXiv preprint arXiv:2312.14125, 2023.

Kreyszig, E. Introductory functional analysis with applications. John Wiley & Sons, 1991.

Kuang, J., Shen, Y., Xie, J., Luo, H., Xu, Z., Li, R., Li, Y., Cheng, X., Lin, X., and Han, Y. Natural language understanding and inference with mllm in visual question answering: A survey. ACM Computing Surveys, 57(8): 1–36, 2025.

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large visionlanguage models. arXiv:2305.10355, 2023.

Li, Y., Wang, C., and Jia, J. Llama-vid: An image is worth 2 tokens in large language models. In European Conference on Computer Vision, pp. 323–340. Springer, 2024.

Li, Y., Zhang, Y., Wang, C., Zhong, Z., Chen, Y., Chu, R., Liu, S., and Jia, J. Mini-gemini: Mining the potential of multi-modality vision language models. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025a.

Li, Z., Qian, D., Su, K., Diao, Q., Xia, X., Liu, C., Yang, W., Zhang, T., and Yuan, Z. Bindweave: Subject-consistent video generation via cross-modal integration. arXiv preprint arXiv:2510.00438, 2025b.

Li, Z., Wang, Y., Xiong, G., Li, W., Pan, Y., and Zhang, T. Generalized few-shot point cloud segmentation via llmassisted hyper-relation matching. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 23063–23073, 2025c.

Lin, B., Ye, Y., Zhu, B., Cui, J., Ning, M., Jin, P., and Yuan, L. Video-llava: Learning united visual representation by alignment before projection. In Proceedings of the 2024 conference on empirical methods in natural language processing, pp. 5971–5984, 2024.

Liu, H., Li, C., Li, Y., and Lee, Y. J. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26296–26306, 2024a.

Liu, H., Li, C., Li, Y., Li, B., Zhang, Y., Shen, S., and Lee, Y. J. Llavanext: Improved reasoning, ocr, and world knowledge, 2024b.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36, 2024c.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024d.

Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.-W., Zhu, S.-C., Tafjord, O., Clark, P., and Kalyan, A. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in Neural Information Processing Systems, 35:2507–2521, 2022.

Maaz, M., Rasheed, H., Khan, S., and Khan, F. Videochatgpt: Towards detailed video understanding via large vision and language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 12585–12602, 2024.

Page, L., Brin, S., Motwani, R., and Winograd, T. The pagerank citation ranking: Bringing order to the web. Technical report, Stanford infolab, 1999.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.

Samet, H. The quadtree and related hierarchical data structures. ACM Computing Surveys (CSUR), 16(2):187–260, 1984.

Shang, Y., Cai, M., Xu, B., Lee, Y. J., and Yan, Y. Llavaprumerge: Adaptive token reduction for efficient large multimodal models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22857– 22867, 2025.

Singh, A., Natarajan, V., Shah, M., Jiang, Y., Chen, X., Batra, D., Parikh, D., and Rohrbach, M. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 8317–8326, 2019.

Team, G., Anil, R., Borgeaud, S., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., Millican, K., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., and Polosukhin, I. Attention is all you need. Advances in Neural Information Processing Systems, 30, 2017.

Von Luxburg, U. A tutorial on spectral clustering. Statistics and computing, 17(4):395–416, 2007.

Wang, F., Mei, J., and Yuille, A. Sclip: Rethinking selfattention for dense vision-language inference. In European Conference on Computer Vision, pp. 315–332. Springer, 2024a.

Wang, Y., Li, K., Li, X., Yu, J., He, Y., Chen, G., Pei, B., Zheng, R., Wang, Z., Shi, Y., et al. Internvideo2: Scaling foundation models for multimodal video understanding. In European Conference on Computer Vision, pp. 396– 416. Springer, 2024b.

Wen, Z., Gao, Y., Wang, S., Zhang, J., Zhang, Q., Li, W., He, C., and Zhang, L. Stop looking for important tokens in multimodal language models: Duplication matters more. arXiv preprint arXiv:2502.11494, 2025.

Xing, L., Huang, Q., Dong, X., Lu, J., Zhang, P., Zang, Y., Cao, Y., He, C., Wang, J., Wu, F., et al. Pyramiddrop: Accelerating your large vision-language models via pyramid visual redundancy reduction. arXiv preprint arXiv:2410.17247, 2024.

Xu, D., Zhao, Z., Xiao, J., Wu, F., Zhang, H., He, X., and Zhuang, Y. Video question answering via gradually refined attention over appearance and motion. In Proceedings of the ACM international conference on Multimedia, pp. 1645–1653, 2017.

Yang, S., Chen, Y., Tian, Z., Wang, C., Li, J., Yu, B., and Jia, J. Visionzip: Longer is better but not necessary in vision language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 19792–19802, 2025.

Zhang, Q., Cheng, A., Lu, M., Zhuo, Z., Wang, M., Cao, J., Guo, S., She, Q., and Zhang, S. [cls] attention is all you need for training-free visual token pruning: Make vlm inference faster. arXiv e-prints, pp. arXiv–2412, 2024a.

Zhang, Q., Cheng, A., Lu, M., Zhang, R., Zhuo, Z., Cao, J., Guo, S., She, Q., and Zhang, S. Beyond text-visual attention: Exploiting visual cues for effective token pruning in vlms. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 20857–20867, 2025.

Zhang, Y., Fan, C.-K., Ma, J., Zheng, W., Huang, T., Cheng, K., Gudovskiy, D., Okuno, T., Nakata, Y., Keutzer, K., et al. Sparsevlm: Visual token sparsification for efficient vision-language model inference. arXiv preprint arXiv:2410.04417, 2024b.

Zhao, H. H., Zhou, P., Gao, D., Bai, Z., and Shou, M. Z. Lova3: Learning to visual question answering, asking and assessment. Advances in Neural Information Processing Systems, 37:115146–115175, 2024.

Zhou, D., Bousquet, O., Lal, T., Weston, J., and Scholkopf,¨ B. Learning with local and global consistency. Advances in Neural Information Processing Systems, 16, 2003.

Zhu, X., Ghahramani, Z., and Lafferty, J. D. Semisupervised learning using gaussian fields and harmonic functions. In Proceedings of the 20th International conference on Machine learning (ICML-03), pp. 912–919, 2003.

Zou, X., Lu, D., Wang, Y., Yan, Y., Lyu, Y., Zheng, X., Zhang, L., and Hu, X. Don’t just chase” highlighted tokens” in mllms: Revisiting visual holistic context retention. arXiv preprint arXiv:2510.02912, 2025.

## A. Additional Theoretical Details

## A.1. HeatFlow Diffusion: Fixed Point, Mass Preservation, and Dirichlet View

This appendix provides proofs and auxiliary lemmas for Proposition 3.1. Throughout, $W \in \mathbb { R } ^ { N \times N }$ is the nonnegative row-stochastic transition matrix in Eq. $( 5 ) \ ( \mathrm { i . e . , } \ W \mathbf { 1 } = \mathbf { 1 } )$ , and $\alpha \in ( 0 , 1 )$ is the diffusion strength in Eq. (2).

A.1 Contraction and unique fixed point (no symmetry required). Define the restart-diffusion operator

$$
\mathcal {T} (e) \triangleq (1 - \alpha) e ^ {(0)} + \alpha W ^ {\top} e.\tag{22}
$$

Lemma A.1 (Contraction and unique fixed point of Eq. (2)). Assume $W \geq 0$ and $W \mathbf { 1 } = \mathbf { 1 }$ with $\alpha \in ( 0 , 1 )$ . Then T in Eq. (22) is an α-contraction under the $\ell _ { 1 }$ norm:

$$
\| \mathcal {T} (e) - \mathcal {T} (e ^ {\prime}) \| _ {1} \leq \alpha \| e - e ^ {\prime} \| _ {1}, \quad \forall e, e ^ {\prime} \in \mathbb {R} ^ {N}.\tag{23}
$$

Consequently, Eq. (2) admits a unique fixed point $e ^ { \star }$ satisfying

$$
(\mathbf {I} - \alpha W ^ {\top}) e ^ {\star} = (1 - \alpha) e ^ {(0)},\tag{24}
$$

and the iterates converge linearly: $\| e ^ { ( t ) } - e ^ { \star } \| _ { 1 } \leq \alpha ^ { t } \| e ^ { ( 0 ) } - e ^ { \star } \| _ { 1 }$

Proof. For any matrix $M , \| M x \| _ { 1 } \leq \| M \| _ { 1 } \| x \| _ { 1 }$ <sub>1</sub> where $\begin{array} { r } { \| M \| _ { 1 } = \operatorname* { m a x } _ { j } \sum _ { i } | M _ { i j } | } \end{array}$ (Horn & Johnson, 2012). Since $W \geq 0$ and each row of W sums to 1, we have

$$
\| W ^ {\top} \| _ {1} = \max _ {j} \sum_ {i} (W ^ {\top}) _ {i j} = \max _ {j} \sum_ {i} W _ {j i} = \max _ {j} \left(\sum_ {i} W _ {j i}\right) = 1.
$$

Therefore,

$$
\| \mathcal {T} (e) - \mathcal {T} (e ^ {\prime}) \| _ {1} = \alpha \| W ^ {\top} (e - e ^ {\prime}) \| _ {1} \leq \alpha \| W ^ {\top} \| _ {1} \| e - e ^ {\prime} \| _ {1} = \alpha \| e - e ^ {\prime} \| _ {1},
$$

proving Eq. (23). By Banach’s fixed-point theorem (Kreyszig, 1991), T has a unique fixed point $e ^ { \star }$ . The fixed-point equation $T ( e ^ { \star } ) = e ^ { \star }$ is equivalent to Eq. (24). The linear convergence bound follows directly from contraction. □

## A.2 Mass preservation (justifies optional renormalization).

Lemma A.2 (Mass preservation under restart diffusion). Assume $W \geq 0$ and $W \mathbf { 1 } = \mathbf { 1 } . \ : I f \mathbf { 1 } ^ { \top } e ^ { ( 0 ) } = \mathbf { 1 } \ : a n d e ^ { ( 0 ) } \geq 0 ,$ , then for all $t \geq 0$

$$
\mathbf {1} ^ {\top} e ^ {(t)} = 1, \qquad e ^ {(t)} \geq 0.\tag{25}
$$

Proof. Nonnegativity follows by induction since $e ^ { ( t + 1 ) } = ( 1 - \alpha ) e ^ { ( 0 ) } + \alpha W ^ { \top } e ^ { ( t ) }$ is a nonnegative combination of nonnegative vectors (as $W ^ { \top } \geq 0 )$ . For mass, use $\mathbf { 1 } ^ { \top } W ^ { \top } = ( W \mathbf { 1 } ) ^ { \top } = \mathbf { 1 } ^ { \top }$

$$
\mathbf {1} ^ {\top} e ^ {(t + 1)} = (1 - \alpha) \mathbf {1} ^ {\top} e ^ {(0)} + \alpha \mathbf {1} ^ {\top} W ^ {\top} e ^ {(t)} = (1 - \alpha) \cdot 1 + \alpha \mathbf {1} ^ {\top} e ^ {(t)}.
$$

If $\mathbf { 1 } ^ { \top } e ^ { ( t ) } = 1$ , then $\mathbf { 1 } ^ { \top } e ^ { ( t + 1 ) } = 1$ . Since it holds at $t = 0 ,$ , it holds for all t.

A.3 Dirichlet view under symmetrized affinity (proof of Prop. 3.1). Proposition 3.1 additionally assumes that $W$ is obtained by row-normalizing a symmetric affinity matrix. Concretely, define $\dot { \mathbf { S } } \in \mathbb { R } ^ { N \times N }$ and $D = \mathrm { d i a g } ( \mathbf { S 1 } )$ such that

$$
\mathbf {S} = \mathbf {S} ^ {\top} \geq 0, \qquad D _ {i i} > 0, \qquad W = D ^ {- 1} \mathbf {S}.\tag{26}
$$

This holds, for instance, when we use a symmetrized KNN edge set and set $S _ { i j } = \exp ( \tau \sin ( z _ { i } , z _ { j } ) )$ ) on edges and 0 otherwise.

Lemma A.3 (Dirichlet energy identity). $I f \mathbf { S } = \mathbf { S } ^ { \top } \geq 0$ and $D = \mathrm { d i a g } ( \mathbf { S 1 } )$ , then for any $f \in \mathbb { R } ^ { N }$

$$
f ^ {\top} (D - \mathbf {S}) f = \frac {1}{2} \sum_ {i = 1} ^ {N} \sum_ {j = 1} ^ {N} S _ {i j} (f _ {i} - f _ {j}) ^ {2}.\tag{27}
$$

Proof. Expand the RHS:

$$
\sum_ {i, j} S _ {i j} (f _ {i} - f _ {j}) ^ {2} = \sum_ {i, j} S _ {i j} (f _ {i} ^ {2} + f _ {j} ^ {2} - 2 f _ {i} f _ {j}) = 2 \sum_ {i} \Big (\sum_ {j} S _ {i j} \Big) f _ {i} ^ {2} - 2 \sum_ {i, j} S _ {i j} f _ {i} f _ {j}.
$$

Since $\textstyle \sum _ { i } S _ { i j } = D _ { i i }$ and $\textstyle \sum _ { i , j } S _ { i j } f _ { i } f _ { j } = f ^ { \top } \mathbf { S } f$ , the RHS equals $2 f ^ { \top } D f - 2 f ^ { \top } \mathbf { S } f = 2 f ^ { \top } ( D - \mathbf { S } ) f$ . Divide by $2$ to obtain Eq. (27). □

Proof of Proposition 3.1. We first show that the fixed point exists and is unique. Under Eq. (26), W is row-stochastic, so Lemma A.1 applies and guarantees a unique fixed point $e ^ { \star }$ solving $( { \bf I } - \alpha W ^ { \bar { \top } } ) e ^ { \star } = ( 1 - \bar { \alpha } ) e ^ { ( 0 ) }$

Next, define the objective in Proposition 3.1:

$$
\mathcal {J} (e) = \frac {1}{2} \big \| D ^ {- 1} e - D ^ {- 1} e ^ {(0)} \big \| _ {D} ^ {2} + \frac {\alpha}{2 (1 - \alpha)} \left(D ^ {- 1} e\right) ^ {\top} (D - \mathbf {S}) (D ^ {- 1} e),
$$

where $\| u \| _ { D } ^ { 2 } = u ^ { \top } D u$ . Let $f = D ^ { - 1 } e$ and $f _ { 0 } = D ^ { - 1 } e ^ { ( 0 ) }$ . Then $\mathcal { I } ( e )$ can be rewritten as a function of $f \colon$

$$
\mathcal {J} (e) \equiv \widetilde {\mathcal {J}} (f) = \frac {1}{2} \| f - f _ {0} \| _ {D} ^ {2} + \frac {\alpha}{2 (1 - \alpha)} f ^ {\top} (D - \mathbf {S}) f.\tag{28}
$$

Since $D \succ 0$ (diagonal with positive entries) and $( D - \mathbf { S } ) \succeq 0 ( \mathrm { g r a p h { L a p l a c i a n } } ) ,$ $\tilde { \mathcal { I } }$ is strictly convex and thus admits a unique minimizer (Boyd & Vandenberghe, 2004).

Compute the gradient w.r.t. $f \colon$

$$
\nabla_ {f} \widetilde {\mathcal {J}} (f) = D (f - f _ {0}) + \frac {\alpha}{1 - \alpha} (D - \mathbf {S}) f.
$$

Setting $\nabla _ { f } \tilde { \mathcal { I } } ( f ) = 0$ and multiplying by (1 − α) yields

$$
(1 - \alpha) D (f - f _ {0}) + \alpha (D - \mathbf {S}) f = 0 \quad \Longleftrightarrow \quad (D - \alpha \mathbf {S}) f = (1 - \alpha) D f _ {0}.
$$

Substituting $f = D ^ { - 1 } \epsilon$ and $f _ { 0 } = D ^ { - 1 } e ^ { ( 0 ) }$ gives

$$
(D - \alpha \mathbf {S}) D ^ {- 1} e = (1 - \alpha) e ^ {(0)} \quad \Longleftrightarrow \quad (\mathbf {I} - \alpha \mathbf {S} D ^ {- 1}) e = (1 - \alpha) e ^ {(0)}.
$$

Finally, since $W ^ { \top } = ( D ^ { - 1 } \mathbf { S ) } ^ { \top } = \mathbf { S } D ^ { - 1 }$ , we obtain

$$
(\mathbf {I} - \alpha W ^ {\top}) e = (1 - \alpha) e ^ {(0)},
$$

which matches the fixed-point equation. By uniqueness of the minimizer, the optimizer of $\mathcal { I } ( e )$ is exactly $e ^ { \star }$ . □

## A.2. Quota Allocation and Integer Rounding for Quadtree Pruning

This appendix provides proof for Proposition 4.1 and a feasibility guarantee for the integer rounding procedure described in Sec. 4.2. Recall that C denotes the set of quadtree leaf crops, $\begin{array} { r } { M ( c ) = \sum _ { ( r , t ) \in c } E _ { r , t } } \end{array}$ is the energy mass, and the fractiona quota is

$$
\hat {q} _ {c} = K \cdot \frac {M (c)}{\sum_ {c ^ {\prime} \in \mathcal {C}} M (c ^ {\prime})}.\tag{29}
$$

Proof of Proposition 4.1. Consider the concave optimization problem in Eq. (12) (Kelly et al., 1998):

$$
\max _ {\{q _ {c} > 0 \}} \sum_ {c \in \mathcal {C}} M (c) \log q _ {c} \quad \text { s.t. } \quad \sum_ {c \in \mathcal {C}} q _ {c} = K.
$$

The Lagrangian is $\begin{array} { r } { \mathcal { L } ( q , \lambda ) = \sum _ { c } M ( c ) \log q _ { c } - \lambda \big ( \sum _ { c } q _ { c } - K \big ) } \end{array}$ . Stationarity gives $\partial \mathcal { L } / \partial q _ { c } = M ( c ) / q _ { c } - \lambda = 0$ , hence $q _ { c } = M ( c ) / \lambda$ for all c. Imposing $\textstyle \sum _ { c } q _ { c } = K$ yields $\begin{array} { r } { \lambda = \frac { \sum _ { c } M ( c ) } { K } } \end{array}$ , and thus $\begin{array} { r } { q _ { c } = K \cdot \frac { M ( c ) } { \sum _ { c ^ { \prime } } M ( c ^ { \prime } ) } \equiv \hat { q } _ { c } } \end{array}$ . Since the objective is strictly concave over $q _ { c } > 0 ;$ , the optimizer is unique. □

Feasible integer rounding with capacity constraints. In Sec. 4.2, we form the initial integer quotas by flooring and clipping: $q _ { c } ^ { ( 0 ) } \gets \operatorname* { m i n } \bigl ( | c | , \lfloor \hat { q } _ { c } \rfloor \bigr )$ (Eq. (13)), and then distribute the remaining budget to crops with available capacity until $\textstyle \sum _ { c } q _ { c } = K$

Lemma A.4 (Feasibility of the floor+clip+fill rounding procedure). Assume $0 < K \le N$ and $\begin{array} { r } { \sum _ { c \in \mathcal { C } } | c | = N \left( i . e . , \mathcal { C } \right. } \end{array}$ is a disjoint cover of the $H \times W ~ g r i d )$ . Let $q _ { c } ^ { ( 0 ) } = \operatorname* { m i n } ( | c | , \ \lfloor \hat { q } _ { c } \rfloor )$ , and let $\begin{array} { r } { R = K - \sum _ { c \in \mathcal { C } } q _ { c } ^ { ( 0 ) } \geq 0 . } \end{array}$ . Consider any procedure that performs R unit increments, each time choosing a crop c with $q _ { c } < | c |$ and setting $q _ { c } \gets q _ { c } + 1 .$ . Then the resulting integer quotas satisfy

$$
\sum_ {c \in \mathcal {C}} q _ {c} = K, \quad 0 \leq q _ {c} \leq | c | \forall c \in \mathcal {C}.\tag{30}
$$

In particular, the “descending $M ( c ) ^ { \prime \prime }$ filling strategy in Sec. 4.2 is feasible.

Proof. By construction, $0 \leq q _ { c } ^ { ( 0 ) } \leq | c |$ and $\textstyle \sum _ { c } q _ { c } ^ { ( 0 ) } \leq \sum _ { c } \hat { q } _ { c } = K$ , hence $R \geq 0$ . The total remaining capacity after the initial step is

$$
\sum_ {c} (| c | - q _ {c} ^ {(0)}) \geq \sum_ {c} | c | - \sum_ {c} q _ {c} ^ {(0)} = N - \sum_ {c} q _ {c} ^ {(0)} \geq K - \sum_ {c} q _ {c} ^ {(0)} = R,
$$

where we used $K \leq N$ . Therefore, there exists sufficient capacity to perform R unit increments without violating $q _ { c } \leq | c |$ . Each increment increases $\textstyle \sum _ { c } q _ { c }$ by 1 while maintaining $q _ { c } \leq | c |$ by choice of $c .$ After exactly R increments, $\begin{array} { r } { \sum _ { c } q _ { c } = \sum _ { c } q _ { c } ^ { ( 0 ) } + R = K } \end{array}$ □

## A.3. Properties of Sink Tokens (Mean + Residual)

This appendix provides basic properties of the sink construction in Eq. (15). Let $\bar { \boldsymbol { S } }$ denote the pruned index set, and assume $| \bar { S } | > 0 . \left( \mathrm { I f } | \bar { S } \right| = 0$ , sink tokens are omitted.)

Lemma A.5 (Mean sink is the optimal 1-mean (least-squares prototype)). The mean sink $\begin{array} { r } { u _ { \mathrm { m e a n } } = \frac { 1 } { | \bar { \mathcal { S } } | } \sum _ { i \in \bar { \mathcal { S } } } X _ { i } } \end{array}$ is the unique minimizer of

$$
\min _ {u \in \mathbb {R} ^ {d}} \sum_ {i \in \bar {\mathcal {S}}} \| X _ {i} - u \| _ {2} ^ {2}.\tag{31}
$$

Proof. Expand the objective: $\begin{array} { r } { \sum _ { i } \| X _ { i } - u \| _ { 2 } ^ { 2 } = \sum _ { i } ( \| X _ { i } \| _ { 2 } ^ { 2 } - 2 u ^ { \top } X _ { i } + \| u \| _ { 2 } ^ { 2 } ) } \end{array}$ . Differentiating w.r.t. u gives $\nabla _ { u } \ : = \ :$ $- 2 \textstyle \sum _ { i } X _ { i } + 2 | { \bar { S } } | u$ . Setting $\nabla _ { u } = 0$ yields $\begin{array} { r } { u = \frac { 1 } { | \bar { S } | } \sum _ { i } X _ { i } = u _ { \mathrm { m e a n } } } \end{array}$ . Strict convexity in u implies uniqueness. □

Lemma A.6 (Residual sink as a radius certificate). Let $u _ { \mathrm { r e s } } = X _ { i }$ ⋆ where $i ^ { \star } = \operatorname * { a r g m a x } _ { i \in \bar { \mathcal { S } } } \| X _ { i } - u _ { \mathrm { m e a n } } \| _ { 2 } ,$ and define the residual radius $r \triangleq \lVert u _ { \mathrm { r e s } } - u _ { \mathrm { m e a n } } \rVert _ { 2 }$ . Then for all $i \in \bar { \mathcal { S } } , \| X _ { i } - u _ { \mathrm { m e a n } } \| _ { 2 } \leq r .$ Moreover, for any vector $v \in \mathbb { R } ^ { d }$

$$
\max _ {i \in \bar {\mathcal {S}}} \left| v ^ {\top} (X _ {i} - u _ {\mathrm{mean}}) \right| \leq \| v \| _ {2} r.\tag{32}
$$

Proof. The first claim follows directly from the definition of $i ^ { \star }$ . For the second, Cauchy–Schwarz yields $| v ^ { \top } ( X _ { i } - u _ { \mathrm { m e a n } } ) | \leq$ $\| v \| _ { 2 } \| X _ { i } - u _ { \mathrm { m e a n } } \| _ { 2 } \leq \| v \| _ { 2 } r$ , and taking the maximum over i completes the proof. □

## B. Detailed Experiment Settings

## B.1. Benchmarks

We evaluated our method on several widely used benchmarks for visual understanding. For image understanding, we considered nine datasets: GQA (Hudson & Manning, 2019); MMBench (MMB)(Liu et al., 2024d) and its Chinese split MMB-CN (Liu et al., 2024d); MME (Fu et al., 2023); POPE (Li et al., 2023); VizWiz (Bigham et al., 2010); SQA (ScienceQA) (Lu et al., 2022); VQA v2 (VQAV2) (Goyal et al., 2017); and TextVQA (VQAText) (Singh et al., 2019).

GQA (Hudson & Manning, 2019) GQA is a compositional visual question answering benchmark built around three core elements: images, structured scene annotations (scene graphs), and questions. Beyond the raw images, the dataset provides object-level information such as locations and attributes. Its questions are designed to probe not only visual recognition but also relational and compositional reasoning over the depicted scene.

MMBench (Liu et al., 2024d). MMBench is a multi-dimensional benchmark for evaluating multimodal models. It adopts a three-level ability taxonomy. The first level, L-1, measures two high-level capabilities, perception and reasoning. The second level, L-2, refines these into six sub-abilities. The third level, L-3, provides the most detailed view by specifying 20 fine-grained ability dimensions, enabling a comprehensive analysis of model behavior.

MME (Fu et al., 2023). MME serves as a broad benchmark for measuring multimodal model capabilities from multiple perspectives. It consists of 14 subtasks that separately test different aspects of perception and reasoning. The evaluation is formulated with instruction and answer pairs, using concise instructions to reduce potential leakage and improve consistency, which supports a more reliable and fair comparison across models.

POPE (Li et al., 2023). POPE is designed to quantify object hallucination in vision language models. It evaluates whether a model incorrectly claims that an object exists in an image by asking targeted yes or no questions about object presence. Performance is summarized with Accuracy, Recall, Precision, and F1 under three sampling protocols, which together offer a reliable view of hallucination tendencies and object grounded behavior.

ScienceQA (Lu et al., 2022). ScienceQA is a broad science question answering benchmark covering disciplines such as natural science, language-related subjects, and social science. It organizes questions with a hierarchical label system that includes topics, categories, and skills, totaling 26 topics, 127 categories, and 379 skills. This structure supports diverse problem types and enables evaluation of multimodal comprehension, multi-step reasoning, and interpretability.

VQA-V2 (Goyal et al., 2017). VQA-v2 is a large-scale visual question answering benchmark that tests visual understanding using open-ended questions about natural images. It contains 265,016 images covering diverse everyday scenes and objects. For each question, the dataset provides 10 human-annotated reference answers, which supports robust scoring and reduces sensitivity to individual annotator variation.

TextVQA (Singh et al., 2019). TextVQA targets settings where answering a question requires reading text that appears inside the image. The task evaluates whether a model can combine visual understanding with text recognition and use the extracted words to support reasoning. Questions therefore depend on both scene content and embedded textual cues, rather than purely visual signals.

MSVD-QA (Xu et al., 2017). MSVD-QA is a video question answering benchmark built from the Microsoft Research Video Description (MSVD) dataset. It contains 1,970 short video clips and about 50.5K associated question-answer pairs. The questions cover diverse aspects of the video content and are commonly used to evaluate video QA, and in some settings also video captioning related capabilities. Question types are grouped into five classes, namely what, who, how, when, and where.

MSRVTT-QA (Xu et al., 2017). MSRVTT-QA is a large-scale video QA benchmark comprising 10K videos and roughly 243K question-answer pairs. The dataset emphasizes reasoning over dynamic content, so accurate answers often depend on capturing both appearance information and temporal evolution across frames. As in MSVD-QA, questions are organized into five types, including what, who, how, when, and where, which supports fine-grained evaluation by question category.

## B.2. Implementation Details

All of our experiments are conducted on Nvidia A6000 GPU. The implementation was carried out in Python 3.10, utilizing PyTorch 2.1.2, and CUDA 12.1. All baseline settings follow the original paper. For our method, we set the number of diffusion steps for energy propagation to 2. In the quadtree partitioning process, we configure the minimum height and width of each crop (denoted as m in Eq. 8) to 4, which defines the finest granularity of spatial division.

## C. Detailed Computational Complexity Analysis

In this appendix, we provide a detailed empirical analysis of the computational cost of SpecFlow, complementing the formal treatment in Sec. 4.4. We report (i) end-to-end wall-clock latency, prefill time, and GPU memory, (ii) a fine-grained breakdown of the pruning-stage overhead, and (iii) a FLOPs comparison under aggressive pruning.

## C.1. End-to-End Wall-Clock Profiling

We measure practical efficiency on the NVIDIA RTX A6000 GPU using LLaVA-1.5-7B. We report prefill time, end-to-end latency, and peak GPU memory. Two pruning ratios are considered: a moderate setting (192 retained tokens, 66.7% pruning)

and an aggressive setting (64 retained tokens, 88.9% pruning). Results are summarized in Table 5.

Table 5. End-to-end efficiency of SpecFlow versus baselines on LLaVA-1.5-7B.

<table><tr><td>Method</td><td>Tokens</td><td>Prune %</td><td>Prefill (ms)</td><td>Latency (s)</td><td>Mem. (GB)</td></tr><tr><td>Vanilla</td><td>576</td><td>0.0</td><td>86.6</td><td>0.397</td><td>19.2</td></tr><tr><td>FastV</td><td>192</td><td>66.7</td><td>42.0</td><td>0.284</td><td>16.2</td></tr><tr><td>HoloV</td><td>192</td><td>66.7</td><td>22.0</td><td>0.248</td><td>15.8</td></tr><tr><td>SpecFlow (Ours)</td><td>192</td><td>66.7</td><td>24.5</td><td>0.259</td><td>15.9</td></tr><tr><td>FastV</td><td>64</td><td>88.9</td><td>29.0</td><td>0.249</td><td>15.8</td></tr><tr><td>HoloV</td><td>64</td><td>88.9</td><td>14.0</td><td>0.221</td><td>14.8</td></tr><tr><td>SpecFlow (Ours)</td><td>64</td><td>88.9</td><td>15.5</td><td>0.229</td><td>14.8</td></tr></table>

Two observations follow. First, SpecFlow achieves substantially faster inference than the unpruned baseline: at 88.9% pruning, end-to-end latency drops from 0.397 s to 0.229 s (a 1.73× speedup) and GPU memory drops from 19.2 GB to 14.8 GB. Second, under the same token budget, SpecFlow introduces only a modest additional overhead relative to the strongest pruning baseline HoloV (+2.5 ms prefill at 192 tokens and +1.5 ms at 64 tokens), while consistently delivering higher accuracy than HoloV under our reported settings.

## C.2. Breakdown of the Pruning-Stage Overhead

To attribute the runtime cost of SpecFlow to its individual components, we further measure the wall-clock time of each stage: kNN graph construction, spectral heat diffusion, and the combined quadtree partitioning, Top-K selection, and sink-token construction. Results are reported in Table 6.

Table 6. Wall-clock breakdown of the SpecFlow pruning module on RTX A6000 with LLaVA-1.5-7B. The total pruning overhead is small relative to the end-to-end latency.

<table><tr><td>Setting</td><td>kNN Graph</td><td>Diffusion</td><td>Quadtree+TopK+Sink</td><td>Total</td><td>Latency</td></tr><tr><td>192 tokens</td><td>1.1 ms</td><td>0.6 ms</td><td>0.9 ms</td><td>2.6 ms</td><td>0.259 s</td></tr><tr><td>64 tokens</td><td>1.1 ms</td><td>0.6 ms</td><td>1.2 ms</td><td>2.9 ms</td><td>0.229 s</td></tr></table>

In both settings, the total pruning overhead is at most 2.9 ms, which corresponds to roughly 1% of the end-to-end latency (0.229–0.259 s). The kNN graph and diffusion costs are essentially constant across pruning ratios, while the quadtree/Top-K/sink stage scales mildly with the chosen budget. This confirms that the pruning module is not the runtime bottleneck: the dominant cost still resides in the downstream LLM execution, particularly the decoding stage.

## C.3. FLOPs Comparison Under Aggressive Pruning

To complement wall-clock measurements with a hardware-agnostic metric, we report total inference FLOPs at the 64-token setting in Table 7. The unpruned LLaVA-1.5-7B requires 8.12 T FLOPs, while SpecFlow reduces this to 1.26 T FLOPs (−84.5%). SpecFlow also consumes fewer FLOPs than FastV at the same token budget (1.26 T vs. 1.64 T), even after accounting for the additional graph-diffusion and sink-construction operations introduced by our method.

Table 7. Total inference FLOPs at the 64-token setting on LLaVA-1.5-7B.

<table><tr><td>Method</td><td>FLOPs</td></tr><tr><td>LLaVA-1.5-7B (vanilla)</td><td>8.12 T</td></tr><tr><td>FastV (64 tokens)</td><td>1.64 T</td></tr><tr><td>SpecFlow (64 tokens)</td><td>1.26 T</td></tr></table>

## C.4. Discussion

Combining the three measurements above, we draw the following conclusions:

• Substantial inference speedup. At 88.9% pruning, SpecFlow reduces end-to-end latency from 0.397 s to 0.229 s and

peak GPU memory from 19.2 GB to 14.8 GB, while retaining 95.6% of the unpruned model’s accuracy.

• Negligible pruning overhead. The combined cost of kNN graph construction, spectral diffusion, and quadtree/sinktoken construction stays within 2.6–2.9 ms, i.e., about 1% of the total end-to-end latency. The added structural modules in SpecFlow are therefore not the runtime bottleneck.

• Favorable FLOPs. Despite introducing graph-based propagation and sink summarization, SpecFlow requires fewer total FLOPs than FastV at the same 64-token budget (1.26 T vs. 1.64 T), and only 15.5% of the FLOPs of the unpruned model.

Overall, these measurements indicate that SpecFlow is competitive with the strongest pruning baseline (HoloV) in runtime while achieving better accuracy, and it remains efficient in terms of latency, memory, and FLOPs. The added structural mechanisms do not become a practical inference bottleneck.

## D. Visualization of Attention and Energy Distribution

To further motivate our design of energy diffusion (HeatFlow), we visualize the distribution and spatial consistency of both raw [CLS] attention scores and our diffused energy scores. This visualization directly addresses the limitations of attention-based pruning highlighted earlier. As shown in Figure 6 (left), raw [CLS] attention yields a spiky distribution that concentrates on a small set of high-value tokens and leaves most tokens with negligible weights. This spiky pattern aligns with the brittle point-wise ranking of existing methods, which risks spatial fragmentation and contextual loss during token pruning. In contrast, Figure 6 (right) shows that HeatFlow transforms this distribution. The density curve spreads across a broader range of values (zoom in for better visualization), and the spatial overlay extends high-energy signals to coherent semantic regions instead of isolated points. This smoothing effect encodes the spatial coherence and contextual dependency of visual tokens, which are ignored by naive attention-based pruning. It ensures our importance scores support robust and structure-preserving token selection

## E. Limitations and Future Work

While SpecFlow achieves strong accuracy and efficiency trade-offs across multiple benchmarks and token budgets, several limitations remain. First, our current evaluation emphasizes accuracy under fixed token budgets and provides FLOPs-based analysis, end-to-end wall-clock measurements (latency, throughput, and peak memory) can depend on hardware, batch size, FlashAttention settings, and the overhead of preprocessing (kNN graph construction, diffusion, and quadtree selection). A thorough systems study that profiles these costs across resolutions and long-video regimes is an important next step. Second, SpecFlow relies on access to intermediate representations (visual token embeddings and attention from the vision encoder), this requirement may limit applicability to black-box VLM APIs or architectures that do not expose reliable CLS attention. Future work could explore more robust or model-agnostic seeding signals (e.g., multi-layer/ multi-head aggregation rules, text-guided cues, or lightweight learned selectors) and adaptive hyperparameters (e.g., diffusion strength/steps, kNN sparsity, and quadtree thresholds) that generalize across backbones without tuning.

Third, our coverage-aware pruning currently assumes a 2D token grid and uses a spatial quadtree. For video inputs, temporal structure is not explicitly modeled, and extending the method to spatiotemporal partitioning or 3D diffusion could further improve stability on dynamic scenes. Finally, the proposed mean + residual sink tokens provide a very compact summary of pruned content, which may be insufficient for inputs requiring fine-grained evidence (e.g., small objects, dense text/OCR, counting, or rare attributes). Exploring adaptive numbers of sink tokens, stronger coreset constructions, or task-aware condensation could better preserve rare but critical cues under extreme compression. We also plan to standardize token-budget accounting (including any appended sinks) and expand analyses of failure cases to clarify when conservative condensation may still break down.

Distribution Statistics of [CLS] AttentionScores and [CLS] Attention Overlay  
![](images/2e6d461db561946030ee0b6ef19cc8d232130eece742e3e6384795937c4f0f05.jpg)

![](images/b7a554ac0604b00a70fe43f593fa92f251bf13fb7cf111cdb21e3b5f018055c1.jpg)

![](images/7c0127a87846d9f6dd932ef4497f4601ba183d86442c7596c5a7a86e61562679.jpg)  
Distribution Statistics of Energy Scores and Energy Overlay

![](images/903d783e0e213f074b76fd4341aa80e67c7f8de575df8fc60d6c5b1e9f293b05.jpg)

![](images/97ca39426d66336e5a5de134a01a818a0fc36f830e4bec1ed594fd5340cf94a2.jpg)

![](images/8d1f84b6110075ae6c95cc55eb512a7a64c575d8a2039e1a4ecf94d2aa152172.jpg)

![](images/d920ea0cfbd8d75854adafe57ff3bd9f43ba243b666a8c4c13be64e81c1bc500.jpg)

![](images/d4bddddba0d6fc2390edd4fbba37a3f7820cd490b3940a62f91702cccae5480e.jpg)

![](images/51c50995abda1c108412cbcddd0c4de91ddc6f93ca8b09a1a1ce9c18daa19fe0.jpg)

![](images/87264d16939f28e4225c5176894c7ba362c7e4d9e588c14c6c432f3c68e47174.jpg)

![](images/57e7e29c1d87204fcf8f8d078684f4e06649892029bf19028efa9a95538a0e7c.jpg)

![](images/001041e0d0ff6c8bc8e4eba72694ee13f6fec77a76895d5cf94f6eb0b4ea409d.jpg)

![](images/6b7472f5cb6ccfb257da12ef5d1039473d3b70442fc5a258c8cdf273d005c811.jpg)

![](images/dd2d7905d4db7fcecc807e0c438a39b6248ec90c605dfbf9e8127457a9d0802e.jpg)

![](images/f01c6644b751be82913da31b113215084ae501222df34de2fc74a299a1f0753c.jpg)

![](images/459ba403ec72b3d8205a08dc8f0cea7ee2da574f35ae3f011a403eff1949fa67.jpg)

![](images/805e8583a07ce6aa540fca3c84f071f3157f888df590580ddc85c4360e2edde5.jpg)

![](images/d8c7493db2e1b4561de66ae6a19ea289528977935fb55c703bd3bd5a93564cad.jpg)

![](images/08e8e6963abb2144001cd4875de5b20a8c2f03e9b26a7ef0b7adedcb95e84e05.jpg)

![](images/60b784afedc99d9ebbf17d78a8768a95c4352e8d10579cfe3da1eecd6ef8b48a.jpg)  
Figure 6. Visualization of raw [CLS] attention and HeatFlow-diffused energy. Left: Raw attention exhibits a spiky distribution. Right: Diffused energy is smoothed and covers coherent semantic regions.