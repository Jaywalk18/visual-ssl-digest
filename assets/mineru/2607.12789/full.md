# AVQ-Attention: Adaptive Vector-Quantized Attention

Winfried van den Dool<sup>1,2</sup>, Patrick Forré<sup>3,4</sup>, Amir Habibian<sup>5</sup>, Yuki M. Asano<sup>6</sup>, and Max Welling<sup>2</sup>

<sup>1</sup> QUVA Lab, University of Amsterdam, The Netherlands w.v.s.o.vandendool@uva.nl

AMLab, Informatics Institute, University of Amsterdam, The Netherlands <sup>3</sup> AI4Science Lab, University of Amsterdam, The Netherlands

<sup>4</sup> Korteweg-de Vries Institute for Mathematics, University of Amsterdam, The Netherlands

5 Qualcomm AI Research, Amsterdam, The Netherlands 6 FunAI Lab, University of Technology Nuremberg, Germany

Abstract. The O(N<sup>2</sup>) complexity of attention over N tokens remains a computational bottleneck in transformer models. Vector-Quantized (VQ) attention reduces this to O(MN) by representing keys with M codewords, but applies uniform codebook capacity regardless of where attention mass concentrates: high-attention regions of key space may be coarsely approximated while low-attention regions waste representational capacity. We propose Adaptive Vector-Quantized (AVQ) Attention, which adaptively allocates codebook capacity based on attention importance. Starting from a small set of codewords, our method identifies the most important codes during the forward pass and refines them with pre-learned child codewords, achieving fine-grained quantization where it matters most while maintaining coarse quantization elsewhere. We develop an implementation using custom Triton kernels that enables the full adaptive refinement process, including importance scoring, child codeword insertion, and parent contribution replacement, to be carried out within the tiled computation paradigm of Flash Attention with minimal overhead. Our approach maintains O(MN) complexity while achieving improved accuracy-eficiency trade-ofs compared to fixed-codebook VQattention.

Keywords: Eficient Attention · Vector Quantization · Adaptive Codebook

## 1 Introduction

Attention mechanisms have become the cornerstone of modern deep learning, enabling transformers [33] to capture rich interactions between input tokens. However, this expressiveness comes at a computational cost: computing attention between all token pairs requires O(N <sup>2</sup>) operations for sequences of length N, making long-sequence processing increasingly prohibitive as models and datasets scale. Flash Attention [7] addressed the $\mathcal { O } ( N ^ { 2 } )$ memory bandwidth bottleneck through tiling strategies that keep intermediate results in fast SRAM. While this speeds up training and inference on longer sequences, the fundamental computational complexity remains quadratic. Unlike memory trafic, reducing computational cost without approximation is fundamentally impossible: N unique tokens communicating pairwise inherently generate $N ^ { 2 }$ interactions. The challenge therefore becomes how to best trade approximation error for computational eficiency.

![](images/9232a8a4897450e6c4db6318143b62691983008dd19060830f2efc3878a732aa.jpg)  
Fig. 1: key-space visualization of Adaptive VQ-Attention. Small dots represent inference key tokens, colored by how much attention they receive from a given query block; large squares represent pre-trained codewords. (1) Keys are assigned to their nearest codeword via vector quantization. (2) Per-codeword importance is computed from the attention mass each codeword receives. (3) The $\mathrm { t o p } { - } \mathcal { P }$ most important parents are selected and their pre-learned children are spawned, refining codebook resolution in high-attention regions while leaving low-attention regions at the coarser parent level.

Various approaches have explored this trade-of, from sparse attention patterns that limit which tokens interact [1, 4, 29] to token merging methods that reduce the number of tokens processed [2]. These methods generally modify the transformer’s structure by dropping or restricting token interactions. Vector-quantized attention ofers an alternative by clustering keys into M representative codewords where $M < N$ implies reducing complexity to $\mathcal { O } ( M N )$ . This approach has shown promise [19], but introduces a new challenge: the codebook must be chosen in advance, applying uniform quantization quality across all regions of the key space. During inference, regions receiving high attention mass may sufer from coarse approximation, while low-attention regions waste representational capacity on unnecessary precision. This challenge parallels a well-studied problem in quantization research. Traditional adaptive quantization methods allocate more representational capacity, i.e. bitwidth, to important features while applying coarser quantization elsewhere. These techniques have proven efective across various domains, from image compression to neural network quantization, by concentrating limited resources where they matter most [9, 36, 37].

Inspired by adaptive quantization techniques, we propose Adaptive Vector Quantized (AVQ) Attention. Attention naturally provides an importance signal: by measuring how much attention mass each cluster in key space receives during the forward pass, we obtain a direct importance measure—without the need to design one separately, as in standard adaptive quantization. We use this signal to dynamically allocate codebook capacity where it matters most. Concretely, we start from a set of parent codewords and compute VQ-Attention while extracting per-codeword importance scores. The top-P most important parents are then refined with their pre-learned child codewords, creating finer quantization in high-attention regions of key space while leaving low-attention regions at the coarser parent level. This refinement adapts dynamically to each input, concentrating codebook capacity based on where that specific input’s attention mass falls (see Fig. 1).

Implementing this adaptive refinement eficiently within Flash Attention’s tiled framework is non-trivial: the attention weights needed to determine per-codeword importance scores are never materialized—they only exist momentarily at the tile level and are immediately consumed. However, we show that Flash Attention’s incremental computation can in fact synergize with our approach: just as Flash Attention builds up attention over blocks of keys via online softmax, AVQ-Attention first computes attention over parent codewords, then incrementally refines it with blocks of children. A geometric constraint on the codebook—each parent equals the mean of its children—allows parent contributions to be recovered directly from the child logits already being computed, enabling eficient in-register correction without revisiting parent codewords. We show that this maintains O(MN) complexity matching standard VQ-attention while enabling adaptive allocation.

We experimentally validate AVQ-Attention on image classification (ImageNet-1k), semantic segmentation (ADE20K), and high-resolution image generation (Stable Difusion). We demonstrate improved accuracy-eficiency trade-ofs over fixed-codebook VQ-attention and competitiveness with a range of existing eficientattention methods. Moreover, we show that (A)VQ-attention can be applied post-hoc to pretrained transformers and fine-tuned in a small number of epochs, analogous to quantization-aware training for model compression. In summary, our contributions include the following.

– A hierarchical VQ-codebook with a constrained parent-child structure, and a training procedure that learns the codebook end-to-end.

– A Flash Attention-compatible mechanism that eficiently increments refinements of the attention output.

– Custom Triton [31] kernels that improve VQ-attention wall-clock performance by fusing operations and minimizing memory trafic, applicable to both flat and adaptive codebooks.

– A linear complexity attention variant combining these techniques to dynamically allocate compute capacity based on attention importance, adapting to each input at inference time. We validate this experimentally, demonstrating improved accuracy-eficiency trade-ofs compared to fixed-codebook VQ-attention.

## 2 Related Work

Adaptive quantization allocates representational capacity unevenly based on importance, applying higher precision where it matters most [36, 37]. While these methods typically require a separate mechanism to estimate importance, in the attention setting we can use attention weights directly, alleviating the need for sensitivity analysis [37], learned metrics [30], or auxiliary models [9]. This insight has also been used to guide mixed-precision KV-cache quantization, allocating higher bitwidth to tokens receiving more attention [40]. However, adaptivity in the context of VQ-attention can be significantly more efective: rather than varying the precision of N token interactions, VQ-attention reduces the number of interactions itself from N to M codewords, directly addressing the complexity bottleneck.

Vector-Quantization has been explored as a means to reduce the quadratic complexity of attention by clustering keys into a smaller set of representative codewords. By attending over codewords rather than individual keys, vectorquantized (VQ) attention reduces computational complexity from $\mathcal { O } ( N ^ { 2 } )$ to $\mathcal { O } ( M N )$ , where $M \ll N$ is the codebook size. Prior work on VQ-attention [19] demonstrates that this approach can achieve favorable eficiency–accuracy tradeofs, but relies on a fixed codebook that applies uniform quantization quality across the key space. We build most directly on this line of work, extending VQ-attention in two directions. First, we implement VQ-attention using custom Triton kernels compatible with Flash Attention’s tiled computation, minimizing memory trafic. Prior VQ-attention work targets the long-context regime where $N \gg M$ and the $\mathcal { O } ( M N ) \ll \mathcal { O } ( N ^ { 2 } )$ complexity gap alone provides large speedups. We first identify opportunities to fuse the sequential steps of VQattention into kernels that keep memory bandwidth low, widening the gap with standard attention at large N as the codebook grows. Additionally, it makes VQ-attention competitive already at moderate sequence lengths where the complexity gap is less pronounced and memory bandwidth plays a more significant role. Second, and more centrally, we replace the fixed codebook with an adaptive one that dynamically allocates additional codewords to regions of the key space receiving high attention mass, improving approximation quality where it matters most.

Clustered attention approximates full attention by grouping queries into clusters and computing attention once per cluster centroid, achieving linear complexity [35]. This assumes queries within a cluster produce similar attention patterns. Our approach clusters keys rather than queries, so every query retains its own attention pattern over the compressed codeword set. Token merging (ToMe) merges similar tokens into aggregated representations, either permanently reducing the token count and compounding the approximation across layers [2], or requiring per-block unmerging to restore the full token set [3]. Moreover, finding merge candidates requires computing pairwise token similarities, which is itself quadratic.

Sparse attention methods reduce the quadratic cost of attention by restricting computation to a subset of token–token interactions, whether through fixed patterns [1, 4], content-based routing [29], locality-sensitive hashing [16], attentionderived selection [39], KV-cache eviction [17], or hierarchical key selection for long-context inference [14, 23]. In contrast, our method preserves dense attention semantics: all keys contribute through their codeword, efectively replacing a binary keep-or-discard decision with a graduated one where less important regions receive coarser approximation rather than being removed entirely.

Linear and low-rank attention methods replace the softmax kernel with a decomposable feature map, enabling $\mathcal { O } ( N )$ complexity via associativity of matrix multiplication [15]. Low-rank methods such as Linformer [38] project keys and values to a lower-dimensional space. Both families modify the attention mechanism itself, whereas our approach preserves standard softmax attention.

## 3 Preliminaries

## 3.1 Self-Attention

Standard scaled dot-product attention [33] computes outputs as weighted averages of value vectors, where weights are determined by query-key similarities:

$$
Y _ {i} = \frac {\sum_ {j = 1} ^ {N} \exp (Q _ {i} K _ {j} ^ {\top}) V _ {j}}{\sum_ {l = 1} ^ {N} \exp (Q _ {i} K _ {l} ^ {\top})}\tag{1}
$$

where $Q _ { i } , K _ { j } , V _ { j } \in \mathbb { R } ^ { d }$ are query, key, and value vectors for tokens i and $j .$ . This formulation requires $\mathcal { O } ( N ^ { 2 } d )$ operations to compute all query-key dot products. It is common to scale key-query dot products by $1 / { \sqrt { d } } ;$ we omit this for clarity.

## 3.2 Vector-Quantized Attention

VQ-attention reduces complexity by replacing keys with quantized representations from a codebook $\{ C _ { a } \} _ { a = } ^ { M }$ where $M \ll N$ . Each key is assigned to its nearest codeword (with slight abuse of notation, we use a both as the assignment function and as a codeword index):

$$
\hat {K} _ {j} = C _ {a (K _ {j})}, \quad \mathrm{where} \quad a (K _ {j}) = \arg \min _ {a} \| C _ {a} - K _ {j} \| ^ {2}\tag{2}
$$

We can rewrite attention by grouping keys that map to the same codeword. Define:

$$
n _ {a} = | \{j: a (K _ {j}) = a \} |, \quad \bar {V} _ {a} = \sum_ {j: a (K _ {j}) = a} V _ {j}\tag{3}
$$

as the count and total aggregated value for codeword a. Then

$$
Y _ {i} \approx \frac {\sum_ {a = 1} ^ {M} \exp (Q _ {i} C _ {a} ^ {\top}) \bar {V} _ {a}}{\sum_ {a = 1} ^ {M} \exp (Q _ {i} C _ {a} ^ {\top}) n _ {a}},\tag{4}
$$

where the approximation error comes solely from key quantization. Indeed, the equivalence to quantized-key attention follows from regrouping the sums:

$$
\begin{array}{r l} \sum_ {j} \exp (Q _ {i} K _ {j} ^ {\top}) V _ {j} & \approx \sum_ {j} \exp (Q _ {i} \hat {K} _ {j} ^ {\top}) V _ {j} \\ & = \sum_ {a} \sum_ {j: a (K _ {j}) = a} \exp (Q _ {i} C _ {a} ^ {\top}) V _ {j} \\ & = \sum_ {a} \exp (Q _ {i} C _ {a} ^ {\top}) \bar {V} _ {a} \end{array}
$$

and similarly for the denominator.

## 3.3 Flash Attention

Flash Attention [6, 7] computes attention without materializing the full $N \times$ N matrix $\exp ( Q K ^ { \top } )$ , instead processing tiles that fit in on-chip SRAM. The algorithm maintains running numerators and denominators, scaled by a running maximum for numerical stability [24]. As we will see, this incremental structure synergizes naturally with our adaptive refinement. We adopt tile-level notation for the following sections: I and J denote contiguous index sets for query and key (or codeword) tiles respectively, with each tile assumed to fit in SRAM. Rewriting Eq. (1), the actual computation on hardware more closely resembles:

$$
\begin{array}{c} A _ {I J} = \exp (Q _ {I} K _ {J} ^ {\top}) \\ X _ {I} (J) = A _ {I J} V _ {J} \\ Z _ {I} (J) = \sum_ {j \in J} A _ {I j} \\ Y _ {I} = \frac {\sum_ {J} X _ {I} (J)}{\sum_ {J} Z _ {I} (J)}. \end{array}
$$

## 4 Method

## 4.1 Hierarchical Codebook

We describe the single-head, single-layer case; multi-head attention maintains one codebook per head. The codebook consists of M parent codewords $\{ C _ { j } \} _ { j = 1 } ^ { M } ,$ each with C children $\{ C _ { j , c } \} _ { c = 1 } ^ { \mathscr { C } }$ , giving $M ( 1 + { \mathcal { C } } )$ codewords in total, as children supplement rather than replace their parents.

Training. All codewords are learned via online k-means [22] with exponential moving averages (EMA) [25]. At each training step, keys are first quantized to their nearest parent, and parent centroids are updated via EMA. Each parent’s assigned keys are then further quantized among its $\mathcal { C }$ children, with the parent itself remaining as an option—keys that are already well-represented by the parent need not move to a child. Child centroids are updated via EMA on their respectively assigned keys (see Fig. 2).

![](images/de6eee988a90378c616f5f1bb42547bc970bce22adbae8fe46f60b066c19c1ef.jpg)  
Fig. 2: Hierarchical codebook in key space, illustrating the five online learning steps. Left: An incoming training batch (gray dots) arrives at the existing codebook. ○1 Keys are assigned to the nearest parent codeword (thick Voronoi boundaries), then ○2 further quantized to child codewords within each parent’s cell (thin boundaries). Right: Zoom into one parent’s cell showing the codebook update. ○3 The parent updates its position via EMA over all keys in its cell. ○4 Each child updates independently via countweighted EMA. ○5 Children are projected to restore the constraint $\begin{array} { r } { C _ { p } = \frac { 1 } { \mathcal { C } } \sum _ { c } C _ { p , c } ; } \end{array}$ heavier children (larger $n _ { c } )$ resist displacement more (Sec. A).

Parent-child constraint. We impose the constraint that each parent equals the mean of its children:

$$
C _ {p} = \frac {1}{\mathcal {C}} \sum_ {c = 1} ^ {\mathcal {C}} C _ {p, c}\tag{5}
$$

Since the unconstrained EMA updates on children will generally violate this, we project child positions back onto the constraint surface after each update via a closed-form mass-weighted projection (see Sec. A). This constraint is central to our method: it enables eficient removal of parent attention contributions at inference time, as we show in Sec. 4.3.

At inference, all codeword positions are fixed.

## 4.2 Vector Quantization for (A)VQ-Attention

Before computing attention, each key must be assigned to a codeword. Unlike standard vector quantization, which only requires assignments, the vector quantization step in VQ-attention must additionally aggregate the values and counts per codeword $( \bar { V } _ { a }$ and $n _ { a }$ in Eq. (3)), since these serve as the inputs to the subsequent attention computation. We compute both assignments and aggregates for the full codebook tree in a single fused kernel pass over the keys. Each key is first assigned to its nearest parent among $M _ { 0 }$ root codewords. Then, it is compared against only the C children of its assigned parent, and reassigned if a child is closer than the (cached) parent distance. At each level, the key’s value is scattered to the assigned codeword, accumulating $\bar { V } _ { a }$ and $n _ { a }$

![](images/5e1b808ad825639f312e089bb9a91f86dac6389dfafd7166c2f083a1f1314469.jpg)  
Fig. 3: Spatial locality of query tiles under Gilbert curve reordering [34] on a 28×28 patch grid. Tokens are reordered along a Gilbert space-filling curve (gray line) so that contiguous tiles (colored regions) form spatially compact 2D regions, regardless of tile size. Since each tile independently selects which parents to refine, spatial compactness ensures that queries sharing a refinement decision attend to similar regions.

The tree structure makes this eficient: the cost per key is $\mathcal { O } ( M _ { 0 } + \mathcal { C } )$ distance computations, yielding a codebook with $M _ { \mathrm { t o t a l } } = M _ { 0 } ( 1 { + } \mathcal { C } )$ codewords. One may be tempted to defer the computation of child codeword aggregates until after the most important parents have been identified, computing only for children that will actually be used. However, precomputing the full tree brings two key advantages:

– Reduced HBM trafic. The subsequent attention kernel can remain fully fused: after computing attention over parent codewords and determining importance, child aggregates are immediately available, and the kernel can proceed to refine without interrupting to perform additional vector quantization. This avoids writing and re-reading intermediate accumulators to global memory.

Per-query-tile amortization and adaptivity. The precomputed aggregates are shared across all query tiles, amortizing the VQ cost. Each tile may then independently select diferent parents for refinement, making the method adaptive not only per input, but per query tile. To ensure that queries within a tile attend to similar spatial regions, we reorder tokens along a Gilbert space-filling curve [34], a generalization of the Hilbert curve to non-power-of-two grids. This produces spatially compact tiles regardless of tile size, allowing diferent regions of the input to refine diferent parts of the codebook (see Fig. 3, Sec. H).

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 VQ Precompute kernel.

Input: $K, V \in \mathbb{R}^{BH \times N \times D}, C \in \mathbb{R}^{H \times M_{\text{total}} \times D}$ with $M_{\text{total}} = M_0(1 + \mathcal{C})$

Output: $\bar{V} \in \mathbb{R}^{BH \times M_{\text{total}} \times D}, n \in \mathbb{R}^{BH \times M_{\text{total}}}$

1: Initialize $\bar{V} = 0, n = 0$

2: for each $(bh, Block_N)$ in parallel do ▷ Separate GPU programs
3:    $k = K[bh, Block_N]$ ▷ [Block_N, D], stays in registers
4:    $v = V[bh, Block_N]$ ▷ [Block_N, D], stays in registers
// Parent assignment (tiled over $M_0$ if codebook exceeds SRAM)
5:    $c_p = C[h, 0:M_0]$ ▷ [M_0, D], loaded into SRAM
6:    $d = ||k - c_p||^2$ ▷ [Block_N, M_0] pairwise
7:    $a = \arg\min(d, axis=1); d_{\text{best}} = \min(d, axis=1)$ ▷ [Block_N] each
8:    $\bar{V}[bh, a] += v; n[bh, a] += 1$ ▷ Atomic add
// Child assignment — children of parent m at C[h, M_0 + mC:M_0 + (m+1)C]
9:    $c_0 = M_0 + a \cdot C$ ▷ [Block_N] first-child index per key
10:    $d_{\text{child}} = \text{full}(Block_N, \infty); c^* = c_0$
11:    for $c = 0, \ldots, C - 1$ do
12:    $c_{\text{code}} = C[h, c_0 + c]$ ▷ [Block_N, D]
13:    $d_c = ||k - c_{\text{code}}||^2$ ▷ [Block_N]
14:    where $d_c &lt; d_{\text{child}}$: $d_{\text{child}} = d_c, c^* = c_0 + c$
15:    end for
16:    where $d_{\text{child}} &lt; d_{\text{best}}$: $\bar{V}[bh, c^*] += v; n[bh, c^*] += 1$ ▷ Masked atomic add
17: end for
</div>

VQ-attention introduces several sequential operations — quantization, value aggregation, and attention — with intermediate results passing through global memory. While O(MN) complexity guarantees asymptotic eficiency, we improve wall-clock performance by fusing quantization and value aggregation into a single kernel that keeps each key in registers across both parent and child assignment. Each key’s value is then scattered directly to its assigned codeword’s accumulator via atomic operations at the tile level. The reduced memory traffic becomes increasingly important as N and M grow, and additionally allows VQ-attention to compete with Flash Attention already at moderate sequence lengths where the complexity gap alone is insuficient. While we present this for AVQ-attention, the fused VQ kernel is equally an improvement for standard VQ-attention. We provide pseudocode in Algorithm 1.

## 4.3 Computing (A)VQ-Attention with Flash Attention

Tiled VQ-attention. The VQ-attention formulation maps directly to Flash Attention’s tiling strategy. Continuing with the notation from Sec. 3.3, we replace $K _ { J }$ with $C _ { J } .$ , so that indices J and j now refer to codewords rather than keys. We write $\begin{array} { r } { \bar { V } _ { j } = \sum _ { k : a ( K _ { k } ) = j } V _ { k } } \end{array}$ for the aggregated values per codeword. The key diference from standard attention is that each codeword represents $n _ { j }$ keys, so the denominator accumulates counts rather than row sums:

$$
\begin{array}{c} A _ {I J} = \exp (Q _ {I} C _ {J} ^ {\top}) \\ X _ {I} (J) = A _ {I J} \bar {V} _ {J} \\ Z _ {I} (J) = A _ {I J} n _ {J} \\ \bar {X} _ {I} = \sum_ {J} X _ {I} (J), \quad \bar {Z} _ {I} = \sum_ {J} Z _ {I} (J) \\ Y _ {I} = \bar {X} _ {I} / \bar {Z} _ {I} \end{array}\tag{6}
$$

where $n _ { j }$ denotes the count of keys quantized to codeword $C _ { j }$ . The accumulators ${ \bar { X } } _ { I }$ and $\bar { Z } _ { I }$ are built up incrementally across tiles using the online softmax trick for numerical stability, as in Flash Attention. We leave the subtraction of a stable maximum in the exponent implicit throughout the text; the pseudocode in Algorithm 2 shows the full computation and Sec. D discusses the choice of running maximum in the presence of codeword counts.

Codeword importance. We can use $A _ { I J }$ and the accumulated denominator ${ \bar { Z } } _ { I }$ to extract per-codeword importance scores. We define importance for each query tile I as:

$$
w _ {j} (I) = \sum_ {i \in I} \frac {A _ {i j} \cdot n _ {j}}{\bar {Z} _ {i}}\tag{7}
$$

where ${ \bar { Z } } _ { i }$ is the i-th element of ${ \bar { Z } } _ { I }$ . Importance is thus computed from the denominator that the online softmax accumulation already maintains, at minimal extra cost. When $M _ { 0 }$ is small enough that all codewords fit in SRAM, ${ \bar { Z } } _ { i }$ is exact after a single pass; for larger $M _ { 0 }$ requiring tiling over codewords, an approximate denominator can be used (see Sec. B).

Child spawning. For each query tile $I ,$ the top-P parents by importance are selected for refinement. For each selected parent, its C (contiguous) children are looked up in the codebook tree. Child aggregated values $\hat { V } _ { c }$ and counts $n _ { c }$ are immediately available, having been precomputed by the VQ kernel (Algorithm 1). Together with the child codewords $C _ { p , c }$ loaded from the codebook, this provides everything needed for child attention.

Child attention. When children are added, some keys shift from their originally assigned parent to a closer child codeword. The parent’s aggregated values $\bar { V } _ { p }$ and counts $n _ { p }$ no longer fully represent these keys, so its contribution in the accumulators ${ \bar { X } } _ { I }$ and ${ \bar { Z } } _ { I }$ must be corrected. Crucially, we want to avoid recomputing the parent logits $S _ { i p } : = Q _ { i } C _ { p } ^ { \top }$ for two reasons: it would cost extra FLOPs, and it would require keeping parent codewords in SRAM while processing children. We structure the computation as a tiled Flash Attention pass where the first tile(s) consist of parents and subsequent tiles consist of children. To correct the parent contribution without revisiting it, we exploit the parent-child constraint (Eq. (5)): since $\begin{array} { r } { C _ { p } = \frac { 1 } { \mathcal { C } } \sum _ { c } C _ { p , c } } \end{array}$ , we have

$$
S _ {i p} = Q _ {i} C _ {p} ^ {\top} = \frac {1}{\mathcal {C}} \sum_ {c = 1} ^ {\mathcal {C}} Q _ {i} C _ {p, c} ^ {\top} = \frac {1}{\mathcal {C}} \sum_ {c = 1} ^ {\mathcal {C}} S _ {i c}\tag{8}
$$

so parent logits are recovered directly from the child logits already being computed. Since children are stored contiguously in the codebook, the sum in Eq. (8) reduces over adjacent entries in a register tile and adds negligible cost. We then define the correcting attention:

$$
\varDelta A _ {i c} = \exp (S _ {i c}) - \exp (S _ {i p})\tag{9}
$$

Updating the accumulators with ∆A in place of A implicitly removes the parent’s contribution for keys that moved to children and replaces it with their respective child’s attention weight (derivation in Sec. C). This requires only a single dot product per parent tile. Pseudocode for the full fused attention kernel is given in Algorithm 2.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 2 Flash AVQ-Attention kernel.

Input: $Q \in \mathbb{R}^{BH \times N \times D}$, $C \in \mathbb{R}^{H \times M_{\text{total}} \times D}$, $\bar{V} \in \mathbb{R}^{BH \times M_{\text{total}} \times D}$, $n \in \mathbb{R}^{BH \times M_{\text{total}}}$

Output: $Y \in \mathbb{R}^{BH \times N \times D}$

1: for each $(bh, Block_N)$ in parallel do ▷ Separate GPU programs
2:    $q = Q[bh, Block_N]$ ▷ [Block$_N$, D], stays in registers
3:    $c = C[h, 0:M_0]$; $\bar{v} = \bar{V}[bh, 0:M_0]$; $n = n[bh, 0:M_0]$ ▷ Into SRAM
4:    $s = q c^\top$ ▷ [Block$_N$, M$_0$] logits
5:    $m = \max_{j: n_j &gt; 0} s_{:,j}$ ▷ Stable max (empty codes excluded)
6:    $A = \exp(s - m)$; $A_{:,j} = 0 \forall j: n_j = 0$
7:    $\bar{x} = A \bar{v}$; $\bar{z} = \sum_j A_{:,j} \cdot n_j$
8:    $w_j = \sum_i A_{ij} n_j / \bar{z}_i$ ▷ [M$_0$] importance (Eq. (7))
// Top-$P$ selection and child refinement
9:    $S = \text{top-\( P(w)$} \) ▷ Selected parent indices
10:    for each selected parent $p \in S$ do ▷ Optionally in [$P/Block_P$] tiles
11:    Load $c_c, \bar{v}_c, n_c$ for children of p
12:    $s_c = q c_c^\top$ ▷ [Block$_N$, C] child logits
// Online softmax merge
13:    $m' = \max(m, \max_{c: n_c &gt; 0} s_{:,c})$
14:    $\bar{x} = \bar{x} \cdot \exp(m - m')$; $\bar{z} = \bar{z} \cdot \exp(m - m')$; $m = m'$
// Parent correction via Eq. (8)
15:    $s_p = \frac{1}{C} \sum_c s_c$ ▷ Recover parent logits from children
16:    $A_c = \exp(s_c - m)$; $A_p = \exp(s_p - m)$
17:    $ΔA = A_c - A_p$; $ΔA_{:,c} = 0 \forall c: n_c = 0$
18:    $\bar{x} += ΔA \bar{v}_c$; $\bar{z} += \sum_c ΔA_{:,c} \cdot n_c$
19:    end for
20:    $Y[bh, Block_N] = \bar{x}/\bar{z}$
21: end for
</div>

Summary. The full AVQ-attention forward pass consists of two fused kernels (Fig. 4): VQ Precompute (Algorithm 1) and Flash Attention (Algorithm 2). Table 1 compares per-step costs with flat VQ-attention.

![](images/f6fd77574a9ab692f080cb0e5e9d10701afc0d8a5899937f24c6cc3046e7c3fc.jpg)  
Fig. 4: AVQ-Attention inference pipeline. Kernel 1 (VQ Precompute): keys and values are quantized against the parent codebook $C _ { p } ,$ , producing aggregated values $\bar { V } _ { p }$ and counts $n _ { p } .$ . Child quantization reuses parent assignments ○1 , so each key is compared only against the $\mathcal { C }$ children of its assigned parent. Kernel 2 (Flash Attention): Attn<sub>p</sub> computes attention over parent codewords and extracts importance. The online softmax accumulators and per-tile importance scores are carried forward $\textcircled{2}$ to Attn<sub>c</sub>, which refines the top ${ } _ { - } \mathcal { P }$ most important parents with child attention using correcting attention weights.

Table 1: Complexity comparison: flat ${ \mathrm { V Q } } .$ -attention (M codewords) vs. AVQ-attention (M<sub>0</sub> parents, C children per parent, P parents refined).

<table><tr><td>Step</td><td>VQ-Attention</td><td>AVQ-Attention</td></tr><tr><td>VQ assignment</td><td> $\mathcal{O}(NMD)$ </td><td> $\mathcal{O}(N(M_0 + \mathcal{C})D)$ </td></tr><tr><td>Value aggregation</td><td> $\mathcal{O}(ND)$ </td><td> $\mathcal{O}(ND)$ </td></tr><tr><td>(Parent) Attention</td><td> $\mathcal{O}(NMD)$ </td><td> $\mathcal{O}(NM_0D)$ </td></tr><tr><td>Child attention</td><td>—</td><td> $\mathcal{O}(NPCD)$ </td></tr><tr><td>Total FLOPs</td><td> $\mathcal{O}(NMD)$ </td><td> $\mathcal{O}(N(M_0 + \mathcal{PC})D)$ </td></tr><tr><td>Codebook resolution</td><td> $M$ </td><td> $M_0(1 + \mathcal{C})$ </td></tr></table>

## 5 Experiments and Results

Starting from pretrained transformers, we replace attention layers with (A)VQattention and fine-tune for a small number of epochs. We evaluate on image classification (ImageNet-1k [8]) using a ViT-Base [10] (N=785 tokens, 85.8% top-1) and semantic segmentation (ADE20K [41]) using DPT-Large [27] (N=901 tokens, 49.0% mIoU). Full training details are in Sec. F. Figure 5 reports task performance vs. attention kernel time under identical training conditions. On both tasks, AVQ-attention consistently outperforms flat VQ-attention at comparable cost, confirming that adaptive codebook allocation improves the accuracy– eficiency trade-of. We further analyze codebook properties and attention-mass concentration in Sec. H.

![](images/bcdda2832c889773f88f5cdcef2c23a9e2b575a18494c3dfb180a556991fc7c3.jpg)

![](images/3049fbd517257148093f1a7e94053cf8f02b6c8d82f10ba4b3f52cd8f7758657.jpg)  
Fig. 5: Task performance vs. attention kernel time (ms) for VQ-attention (blue, labeled by M) and AVQ-attention (red, labeled by $M _ { 0 } / \mathcal { P } / \mathcal { C } )$ . Left: Top-1 accuracy on ImageNet-1k (N=785). Right: Mean IoU on ADE20K (N=901). AVQ achieves higher quality than VQ at comparable cost on both tasks.

## 5.1 Scaling Analysis

The eficiency advantages of (A)VQ-attention grow with sequence length. To validate the complexity analysis from Tab. 1, we benchmark wall-clock kernel time across sequence lengths; Fig. 6 in the Appendix confirms the predicted linear scaling with both N and the number of codewords (setup details in Sec. E). Table 2 reports absolute wall-clock times for the larger configurations most relevant to long-sequence deployment, including an unfused VQ baseline to isolate the speedup from fusing the VQ precompute step. At $N { = } 6 5 , 5 3 6$ , our fused AVQ-attention configurations are 81–127× faster than Flash Attention. For flat VQ-attention, comparing against the unfused baseline shows that fusing the VQ precompute step alone provides $\mathbf { a } \sim 2 \times$ speedup.

## 5.2 Elastic Inference via Top-P Adjustment

Since P can be changed at inference time without retraining, a single model can trade speed for accuracy. Table 6 shows this for $M _ { 0 } { = } 6 4 , c { = } 8$ , trained with ful spawning $\scriptstyle \left( { \mathcal { P } } = M _ { 0 } \right)$ . We also report capture: the fraction of true attention weight falling on keys assigned to the selected P parents. At P=16, capture exceeds 82% on both tasks and performance is within 0.3% of the maximum, explaining the diminishing returns at higher P.

## 5.3 Comparison Across Eficient-Attention Methods

We compare against several representative eficient-attention methods (Tab. 3). AVQ reaches the highest mIoU at comparable kernel cost, confirming it is competitive as a drop-in replacement for the attention layer.

Table 2: Wall-clock kernel time (ms) vs. sequence length N $\left( B { = } 4 , H { = } 1 2 , D { = } 6 4 \right)$ . For fair comparison, all methods use flash-attention-style kernels for the attention step; “unfused $\mathrm { V Q } ^ { \dag }$ uses torch.compile’d PyTorch only for VQ precompute, so the speedup to our fused rows isolates the contribution of fusing the VQ precompute step. AVQ configs denoted $M _ { 0 } / { \mathcal { P } } / { \mathcal { C } } .$ Setup details in Sec. E.

<table><tr><td>Configuration</td><td>N=1k</td><td>N=4k</td><td>N=16k</td><td>N=64k</td></tr><tr><td colspan="5">Baselines</td></tr><tr><td>Flash-Attn ( $\mathcal{O}(N^2)$ )</td><td>0.21</td><td>3.11</td><td>49.6</td><td>847</td></tr><tr><td>VQ-attn (unfused VQ) M=256</td><td>0.35</td><td>1.29</td><td>5.16</td><td>20.7</td></tr><tr><td>VQ-attn (unfused VQ) M=512</td><td>0.61</td><td>2.31</td><td>9.28</td><td>37.4</td></tr><tr><td colspan="5">Ours (fused kernels)</td></tr><tr><td>VQ-attn M=256</td><td>0.18</td><td>0.66</td><td>2.62</td><td>10.6</td></tr><tr><td>VQ-attn M=512</td><td>0.30</td><td>1.10</td><td>4.40</td><td>17.9</td></tr><tr><td>AVQ-attn 64/8/8</td><td>0.13</td><td>0.43</td><td>1.65</td><td>6.67</td></tr><tr><td>AVQ-attn 64/16/8</td><td>0.16</td><td>0.51</td><td>1.99</td><td>7.95</td></tr><tr><td>AVQ-attn 128/8/8</td><td>0.17</td><td>0.60</td><td>2.37</td><td>9.46</td></tr><tr><td>AVQ-attn 128/16/8</td><td>0.21</td><td>0.67</td><td>2.59</td><td>10.4</td></tr></table>

Table 3: ADE20K (DPT-L); mIoU ± std from 3 seeds. All methods use an identical 30-epoch recipe to give each the chance to saturate (Sec. F). <sup>†</sup>not FA-compatible.

<table><tr><td>Method</td><td>mIoU (%)</td><td>Kernel (ms)</td></tr><tr><td>Flash Attention-v2 (baseline)</td><td>49.0</td><td>0.313</td></tr><tr><td>AVQ ( $M_0=32, \mathcal{P}=8, \mathcal{C}=8$ )</td><td> $43.33 \pm 0.12$ </td><td>0.116</td></tr><tr><td>Flat VQ ( $M=128$ )</td><td> $42.70 \pm 0.08$ </td><td>0.103</td></tr><tr><td>Flat VQ ( $M=192$ )</td><td> $43.04 \pm 0.05$ </td><td>0.164</td></tr><tr><td>Swin (window 7) [21]</td><td> $42.90 \pm 0.09$ </td><td>0.108</td></tr><tr><td>NATTEN (window 5) [11]</td><td> $40.99 \pm 0.21$ </td><td>0.126</td></tr><tr><td>Linformer ( $k=64$ ) [38]</td><td>38.80</td><td>0.139</td></tr><tr><td>Performer ( $r=256$ ) [5]</td><td>25.29</td><td> $0.54^†$ </td></tr></table>

## 5.4 High-Resolution Difusion

We evaluate AVQ-attention in the Stable Difusion 1.5 (SD1.5) UNet [28] following the distillation setup of LinFusion [20]. We adopt the same distillation objective, data, and hyperparameters, training for 50k steps (half of their 100k). As the UNet’s inner-most self-attention blocks operate on only $N { \le } 2 5 6$ tokens, exact attention is already trivially cheap and a marginal part of the cost, leaving little to gain from replacing it. We therefore apply AVQ only to the five outer (level-0) blocks (N=4096) and for one experiment also to the five level-1 blocks (N=1024). Using ScaleCrafter [12] we further test the models on unseen 1024<sup>2</sup> inference resolutions, making AVQ-attention handle token numbers up to N≈16k. We report FID [13] and CLIP score [26] on COCO [18] under LinFusion’s evaluation protocol in Table 4.

Table 4: SD1.5 on COCO under the LinFusion protocol; FID, CLIP, and UNet forward time per denoising step. <sup>†</sup>additionally replaces the five level-1 (N=1k) blocks.

<table><tr><td rowspan="2">Method</td><td colspan="3"> $512^2$  (train)</td><td colspan="3"> $1024^2$  (w/ ScaleCrafter)</td></tr><tr><td>FID↓</td><td>CLIP↑</td><td>ms</td><td>FID↓</td><td>CLIP↑</td><td>ms</td></tr><tr><td>SD1.5 (FA-v2)</td><td>12.75</td><td>0.318</td><td>44.37</td><td>41.03</td><td>0.292</td><td>213.43</td></tr><tr><td>LinFusion (Mamba)</td><td>12.52</td><td>0.318</td><td>43.70</td><td>36.37</td><td>0.295</td><td>141.57</td></tr><tr><td>ToMe-SD (r=0.5)</td><td>12.40</td><td>0.317</td><td>41.99</td><td>43.00</td><td>0.290</td><td>163.30</td></tr><tr><td colspan="7">AVQ (ours)</td></tr><tr><td>M32P8C8</td><td>12.50</td><td>0.319</td><td>39.09</td><td>35.39</td><td>0.297</td><td>133.48</td></tr><tr><td>M64P16C8</td><td>12.51</td><td>0.320</td><td>39.50</td><td>35.65</td><td>0.296</td><td>135.06</td></tr><tr><td>M128P32C8</td><td>12.55</td><td>0.319</td><td>40.23</td><td>35.51</td><td>0.297</td><td>138.16</td></tr><tr><td>M64P16C8 (+L1) $^†$ </td><td>12.48</td><td>0.320</td><td>39.41</td><td>36.29</td><td>0.297</td><td>128.59</td></tr></table>

## 6 Discussion and Conclusion

We have presented AVQ-attention, an adaptive extension of vector-quantized attention that dynamically allocates codebook capacity to regions of key space receiving high attention mass. By combining a hierarchical codebook with importancedriven refinement, AVQ-attention achieves better task performance than flat VQ-attention at comparable cost. The experiments validate AVQ-attention as a general-purpose attention mechanism rather than targeting benchmark-specific performance: we take pretrained transformers, replace their attention layers, and fine-tune for only a few epochs. Our focus was on the controlled comparison between AVQ, VQ and other eficient attention types, and given the promise of (A)VQ-attention as a competitive layer type, we leave the design of eficient end-to-end transformer architectures around it as future research. Some training choices remain lightly explored. For instance, the optimal M<sub>0</sub>, P, and C likely vary across layers — some layers may need far fewer codewords (Sec. H)

— making per-layer configuration an attractive direction. AVQ-attention facilitates this by exposing informative measures such as captured attention mass and quantization error. The latter could additionally serve as a refinement criterion alongside importance, prioritizing parents whose children difer most from the parent representation. Together, these signals could enable eficient architecture search without full retraining. Beyond per-layer tuning, deeper hierarchies are a natural next step: each additional depth adds only PC codes to the attention cost while multiplying codebook resolution by C, potentially yielding exponential resolution growth for linear cost.

## Acknowledgements

This work is financially supported by Qualcomm Technologies Inc., the University of Amsterdam, and the allowance Top consortia for Knowledge and Innovation from the Netherlands Ministry of Economic Afairs and Climate Policy.

## References

1. Beltagy, I., Peters, M.E., Cohan, A.: Longformer: The long-document transformer. arXiv preprint 2004.05150 (2020), https://arxiv.org/abs/2004.05150

2. Bolya, D., Fu, C.Y., Dai, X., Zhang, P., Feichtenhofer, C., Hofman, J.: Token merging: Your vit but faster. In: International Conference on Learning Representations (ICLR) 2023 (2022), https://arxiv.org/abs/2210.09461, oral presentation

3. Bolya, D., Hofman, J.: Token merging for fast stable difusion. arXiv preprint 2303.17604 (2023), https://arxiv.org/abs/2303.17604

4. Child, R., Gray, S., Radford, A., Sutskever, I.: Generating long sequences with sparse transformers. arXiv preprint 1904.10509 (2019), https://arxiv.org/abs/ 1904.10509

5. Choromanski, K., Likhosherstov, V., Dohan, D., Song, X., Gane, A., Sarlós, T., Hawkins, P., Davis, J., Mohiuddin, A., Kaiser, Ł., Belanger, D., Colwell, L., Weller, A.: Rethinking attention with Performers. In: ICLR (2021), https://arxiv.org/ abs/2009.14794

6. Dao, T.: FlashAttention-2: Faster attention with better parallelism and work partitioning. In: ICLR (2024), https://arxiv.org/abs/2307.08691

7. Dao, T., Fu, D.Y., Ermon, S., Rudra, A., Ré, C.: Flashattention: Fast and memoryeficient exact attention with io-awareness. In: Advances in Neural Information Processing Systems. vol. 35 (2022), https://arxiv.org/abs/2205.14135

8. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet: A large-scale hierarchical image database. In: CVPR. pp. 248–255 (2009)

9. van den Dool, W., Zhdanov, M., Asano, Y.M., Welling, M.: Adaptive meshquantization for neural PDE solvers. arXiv preprint 2511.18474 (2025), https: //arxiv.org/abs/2511.18474

10. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: ICLR (2021), https://arxiv.org/abs/2010.11929

11. Hassani, A., Walton, S., Li, J., Li, S., Shi, H.: Neighborhood attention transformer. In: CVPR (2023), https://arxiv.org/abs/2204.07143

12. He, Y., Yang, S., Chen, H., Cun, X., Xia, M., Zhang, Y., Wang, X., He, R., Chen, Q., Shan, Y.: ScaleCrafter: Tuning-free higher-resolution visual generation with difusion models. In: ICLR (2024), https://arxiv.org/abs/2310.07702

13. Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., Hochreiter, S.: GANs trained by a two time-scale update rule converge to a local nash equilibrium. In: NeurIPS (2017), https://arxiv.org/abs/1706.08500

14. Hooper, C., Kim, S., Mohammadzadeh, H., Maheswaran, M., Zhao, S., Paik, J., Mahoney, M.W., Keutzer, K., Gholami, A.: Squeezed attention: Accelerating long context length LLM inference. In: Proc. Annual Meeting of the Association for Computational Linguistics (ACL) (2025), https://arxiv.org/abs/2411.09688

15. Katharopoulos, A., Vyas, A., Pappas, N., Fleuret, F.: Transformers are RNNs: Fast autoregressive transformers with linear attention. In: ICML (2020), https: //arxiv.org/abs/2006.16236

16. Kitaev, N., Kaiser, Ł., Levskaya, A.: Reformer: The eficient transformer. In: ICLR (2020), https://arxiv.org/abs/2001.04451

17. Li, Y., Huang, Y., Yang, B., Venkitesh, B., Locatelli, A., Ye, H., Cai, T., Lewis, P., Chen, D.: Snapkv: Llm knows what you are looking for before generation. Advances in Neural Information Processing Systems 37, 22947–22970 (2024)

18. Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft COCO: Common objects in context. In: ECCV (2014), https://arxiv.org/abs/1405.0312

19. Lingle, L.D.: Transformer-vq: Linear-time transformers via vector quantization. In: International Conference on Learning Representations. vol. 2024, pp. 6121–6150 (2024)

20. Liu, S., Yu, W., Tan, Z., Wang, X.: LinFusion: 1 GPU, 1 minute, 16K image. arXiv preprint arXiv:2409.02097 (2024), https://arxiv.org/abs/2409.02097

21. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., Guo, B.: Swin Transformer: Hierarchical vision transformer using shifted windows. In: ICCV (2021), https://arxiv.org/abs/2103.14030

22. Lloyd, S.P.: Least squares quantization in PCM. IEEE Transactions on Information Theory 28(2), 129–137 (1982)

23. Mao, Y., Wang, Q., Ester, M., Li, K.: IceCache: Memory-eficient KV-cache management for long-sequence LLMs. In: ICLR (2026), https://arxiv.org/abs/2604. 10539

24. Milakov, M., Gimelshein, N.: Online normalizer calculation for softmax. arXiv preprint arXiv:1805.02867 (2018), https://arxiv.org/abs/1805.02867

25. van den Oord, A., Vinyals, O., Kavukcuoglu, K.: Neural discrete representation learning. In: NeurIPS (2017), https://arxiv.org/abs/1711.00937

26. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: ICML (2021), https://arxiv.org/abs/2103.00020

27. Ranftl, R., Bochkovskiy, A., Koltun, V.: Vision transformers for dense prediction. In: ICCV. pp. 12179–12188 (2021), https://arxiv.org/abs/2103.13413

28. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: CVPR (2022), https://arxiv. org/abs/2112.10752

29. Roy, A., Safar, M., Vaswani, A., Grangier, D.: Eficient content-based sparse attention with routing transformers. Transactions of the Association for Computational Linguistics 9, 53–68 (2021)

30. Tang, C., Ouyang, K., Wang, Z., Zhu, Y., Ji, W., Wang, Y., Zhu, W.: Mixedprecision neural network quantization via learned layer-wise importance. In: European Conference on Computer Vision (ECCV) (2022)

31. Tillet, P., Kung, H.T., Cox, D.: Triton: An intermediate language and compiler for tiled neural network computations. In: Proceedings of the 3rd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages (MAPL). pp. 10–19 (2019)

32. Vali, M.H., Bäckström, T., Solin, A.: Diveq: Diferentiable vector quantization using the reparameterization trick. arXiv preprint arXiv:2509.26469 (2025)

33. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. In: NeurIPS. pp. 5998–6008 (2017), https://arxiv.org/abs/1706.03762

34. Červený, J.: Gilbert: Generalized Hilbert (“Gilbert”) space-filling curve for rectangular domains of arbitrary (non-power of two) sizes. https://github.com/ jakubcerveny/gilbert (2018), gitHub repository

35. Vyas, A., Katharopoulos, A., Fleuret, F.: Fast transformers with clustered attention. In: Advances in Neural Information Processing Systems (NeurIPS) (2020), https://arxiv.org/abs/2007.04825, arXiv:2007.04825

36. Wallace, G.K.: The JPEG still picture compression standard. Communications of the ACM 34(4), 30–44 (1991)

37. Wang, K., Liu, Z., Lin, Y., Lin, J., Han, S.: HAQ: Hardware-aware automated quantization with mixed precision. In: IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8612–8620 (2019)

38. Wang, S., Li, B.Z., Khabsa, M., Fang, H., Ma, H.: Linformer: Self-attention with linear complexity. arXiv preprint arXiv:2006.04768 (2020), https://arxiv.org/ abs/2006.04768

39. Yuan, J., Gao, H., Dai, D., Luo, J., Zhao, L., Zhang, Z., Xie, Z., Wei, Y., Wang, L., Xiao, Z., et al.: Native sparse attention: Hardware-aligned and natively trainable sparse attention. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 23078–23097 (2025)

40. Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., Ré, C., Barrett, C., Wang, Z., Chen, B.: H<sub>2</sub>O: Heavy-hitter oracle for eficient generative inference of large language models. In: Advances in Neural Information Processing Systems (NeurIPS) (2023)

41. Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., Torralba, A.: Scene parsing through ADE20K dataset. In: CVPR. pp. 633–641 (2017)

## A Codeword Clustering During Training

During training, we maintain codebook positions through exponential moving averages (EMA) while enforcing the parent-child constraint $\begin{array} { r } { \bar { C _ { p } } = \frac { 1 } { \mathcal { C } } \sum _ { c = 1 } ^ { \mathcal { C } } C _ { p , c } . } \end{array}$ This section describes how we update codewords when new data arrives at time $t + 1$

Dual representation for codewords. Let $p$ be a parent with C children. For each child $c ,$ we maintain two representations.

– Unconstrained EMA statistics: $( S _ { c } , N _ { c } )$ accumulate keys assigned to child c via standard EMA, tracking where it would naturally cluster without constraints. These define unconstrained means $M _ { c } = S _ { c } / N _ { c }$

– Constrained positions: $C _ { p , c }$ are the actual codeword positions used for quantization and attention, derived from the unconstrained statistics while satisfying the constraint $\begin{array} { r } { C _ { p } = \frac { 1 } { \mathcal { C } } \sum _ { c } C _ { p , c } } \end{array}$

The unconstrained statistics $( S _ { c } , N _ { c } )$ preserve the full EMA history of where data naturally lies, while the constrained positions enforce the geometric relationship with the parent.

Parent update and child adjustment. When new keys arrive at time $t + 1$ , we update each code’s EMA statistics:

$$
S _ {x} ^ {(t + 1)} = \lambda S _ {x} ^ {(t)} + (1 - \lambda) \sum_ {k: \hat {k} = C _ {x} ^ {(t)}} k\tag{10}
$$

$$
N _ {x} ^ {(t + 1)} = \lambda N _ {x} ^ {(t)} + (1 - \lambda) | \{k: \hat {k} = C _ {x} ^ {(t)} \} |\tag{11}
$$

where λ is the EMA decay rate. For parents we use no constraints and set $C _ { p } ^ { ( t + 1 ) } = M _ { p } ^ { ( t + 1 ) } = S _ { p } ^ { ( t + 1 ) } / N _ { p } ^ { ( t + 1 ) }$ directly. However, updating children using the same direct $C _ { c } ^ { ( t + 1 ) } = M _ { c } ^ { ( t + 1 ) }$ would violate the parent-child constraint. To restore the constraint with minimal disruption, we solve the mass-weighted leastsquares problem:

$$
\begin{array}{l} \min _ {C _ {p, 1}, \dots , C _ {p, \mathcal {C}}} \sum_ {c = 1} ^ {\mathcal {C}} N _ {c} \| M _ {c} - C _ {p, c} \| ^ {2} \\ \text { subject   to: } \quad \frac {1}{\mathcal {C}} \sum_ {c = 1} ^ {\mathcal {C}} C _ {p, c} = C _ {p} \end{array}
$$

Via Lagrange multipliers, this yields a closed-form projection. Defining the constraint residual $\begin{array} { r } { \delta ^ { ( t + \bar { 1 } ) } = \sum _ { c } M _ { c } ^ { \mathsf { \tilde { ( } { t + 1 } ) } } - \mathcal { C } C _ { p } ^ { ( t + 1 ) } } \end{array}$ and the harmonic weight $\sigma ^ { ( t + 1 ) } =$ $\textstyle \sum _ { c } 1 / N _ { c } ^ { ( t + 1 ) }$ :

$$
C _ {p, c} ^ {(t + 1)} = M _ {c} ^ {(t + 1)} - \frac {\delta^ {(t + 1)}}{N _ {c} ^ {(t + 1)} \cdot \sigma^ {(t + 1)}}\tag{12}
$$

Each child is pulled toward the constraint surface by an amount inversely proportional to its mass—heavier children (larger $N _ { c } )$ resist displacement more. These adjusted positions are used when children receive new data in the next step, while maintaining the constraint with the updated parent.

## B Computing Importance with Tiled Codewords

When $M _ { 0 }$ is too large for all codewords to fit in SRAM, the attention pass must tile over codewords. Computing importance $w _ { j } ~ ( \mathrm { E q . ~ ( 7 ) } )$ in this setting presents a challenge: the denominator $\begin{array} { r } { \bar { Z } _ { i } = \sum _ { j ^ { \prime } } A _ { i j ^ { \prime } } n _ { j ^ { \prime } } } \end{array}$ is only available after processing all codeword tiles, but we want to extract importance within each tile to avoid recomputing logits, and reduce it across queries within each tile to save memory. We address this by using an approximate denominator based on count-based extrapolation. Let t denote a tile index set for codewords, with $t \leq J$ meaning tile t is processed before or at tile $J .$ . After processing tiles up to $J ,$ , the partial denominator $\textstyle \sum _ { t < J } Z _ { I } ( t )$ accounts for only those keys whose parent codewords have been visited. Let $\begin{array} { r } { N _ { \mathrm { s e e n } } ( J ) = \sum _ { t \leq J } \sum _ { j \in t } n _ { j } } \end{array}$ be the total number of keys whose codewords have been processed so far. We extrapolate to the full key count:

$$
\tilde {Z} _ {I} (J) := \frac {N}{N _ {\mathrm{seen}} (J)} \sum_ {t \leq J} Z _ {I} (t)\tag{13}
$$

where $N$ is the total number of keys. Codeword importance is then computed as in Eq. (7), but with the approximate denominator:

$$
w _ {j} (I) = \sum_ {i \in I} \frac {A _ {i j} \cdot n _ {j}}{\tilde {Z} _ {i} (J)}, \quad j \in J,\tag{14}
$$

where J is the codeword tile containing $j .$ . After all codeword tiles are processed for a given query tile I, it has importance scores for all $M _ { 0 }$ codewords and independently selects its top- $\mathbf { \nabla } \cdot \mathcal { P }$ parents for refinement. In our experimental settings, all relevant codebook sizes already fit within a single SRAM tile. To evaluate the approximation nonetheless, we train $\mathrm { A V Q }$ models with $M _ { 0 } { = } 2 5 6 , \mathcal { C } { = } 8$ and force tiling into two tiles of 128 codewords each at evaluation time. Table 5 reports the fraction of true attention mass captured by the $\mathrm { t o p } { - } \mathcal { P }$ parents selected via three methods: tiled importance (using the approximate denominator $\operatorname { E q }$ . (13)), non-tiled importance (exact denominator from a single pass over all codewords), and exact selection (computing the full $N \times N$ attention matrix and selecting the $\mathcal { P }$ parents whose assigned keys receive the most true attention mass).

## C Correcting Attention Derivation

We show that updating the online softmax accumulators with the correcting attention $\varDelta A _ { i c } = A _ { i c } - A _ { i p }$ correctly replaces the parent’s contribution with finer-grained child contributions.

Table 5: Top-P capture (% of true attention mass) for $M _ { 0 } { = } 2 5 6$ , C=8 with tile size 128. Tiled importance uses the count-based extrapolated denominator; non-tiled uses the exact denominator; exact selects the P parents with highest true attention mass, computed from the full attention matrix.

<table><tr><td>Dataset</td><td>P</td><td>Tiled</td><td>Non-tiled</td><td>Exact</td></tr><tr><td>ImageNet (ViT-Base)</td><td>24</td><td>71.6%</td><td>75.9%</td><td>80.1%</td></tr><tr><td>ImageNet (ViT-Base)</td><td>32</td><td>77.7%</td><td>82.0%</td><td>85.5%</td></tr><tr><td>ADE20K (DPT-Large)</td><td>24</td><td>80.0%</td><td>83.2%</td><td>84.3%</td></tr><tr><td>ADE20K (DPT-Large)</td><td>32</td><td>85.9%</td><td>88.6%</td><td>89.4%</td></tr></table>

Let parent $p$ have $n _ { p }$ assigned keys with aggregated values ${ \bar { V } } _ { p } .$ After refinement, each child c receives $n _ { c }$ keys with aggregated values ${ \bar { V } } _ { c } ;$ the remaining keys stay with the parent:

$$
\bar {V} _ {\mathrm{stay}} = \bar {V} _ {p} - \sum_ {c = 1} ^ {\mathcal {C}} \bar {V} _ {c}, \qquad n _ {\mathrm{stay}} = n _ {p} - \sum_ {c = 1} ^ {\mathcal {C}} n _ {c}.\tag{15}
$$

The initial attention pass included the parent’s contribution $A _ { i p } \bar { V } _ { p }$ in the numerator accumulator. The correct post-refinement contribution is $A _ { i p } \bar { V } _ { \mathrm { s t a y } } +$ $\textstyle \sum _ { c } A _ { i c } { \bar { V } } _ { c }$ , so the required correction is:

$$
\sum_ {c} A _ {i c} \bar {V} _ {c} + A _ {i p} \bar {V} _ {\mathrm{stay}} - A _ {i p} \bar {V} _ {p} = \sum_ {c} A _ {i c} \bar {V} _ {c} - A _ {i p} \sum_ {c} \bar {V} _ {c}.\tag{16}
$$

Since $A _ { i p }$ is constant across children, this factorizes into a single dot product:

$$
\sum_ {c} \left(A _ {i c} - A _ {i p}\right) \bar {V} _ {c} = \sum_ {c} \varDelta A _ {i c} \bar {V} _ {c}\tag{17}
$$

The denominator correction $\textstyle \sum _ { c } { \varDelta A _ { i c } n _ { c } }$ follows identically. Thus the full correction for both accumulators is computed with one dot product of ∆A against the child aggregates, without needing to load the parent codeword $C _ { p }$ , its aggregated values $\bar { V } _ { p } ,$ , or counts $n _ { p } .$ , and without computing or storing the parent logits $Q _ { i } C _ { p } ^ { \top }$ separately.

## D Numerical Stability of Online Softmax in VQ-Attention

The online softmax maintains a running maximum $m _ { i }$ and computes all attention weights as exp $( S _ { i j } - m _ { i } )$ . The choice of $m _ { i }$ does not afect the final result (it cancels between numerator and denominator), but it controls numerical precision. In AVQ-attention, the refinement round introduces child codewords that may have zero assigned keys $( n _ { c } = 0 )$ . Since empty codewords are not constrained by any assigned keys, their logits $\dot { S } _ { i c } = Q _ { i } C _ { c } ^ { \top }$ can lie far from the populated region of key space. If such a codeword sets the running maximum to a value much larger than the previous maximum, the rescaling factor exp $( m _ { \mathrm { o l d } } - m _ { \mathrm { n e w } } )$ could underflow to zero, destroying all previously accumulated attention information — even though the empty codeword contributes nothing to the output $( \bar { V } _ { c } = 0$ 2 $n _ { c } = 0 )$ .

A natural idea is to redefine the logits as $S _ { i j } ^ { \prime } = Q _ { i } C _ { j } ^ { \top } + \ln n _ { j } ,$ so that empty codewords $( n _ { j } = 0 , \ln 0 = - \infty )$ are automatically excluded from the maximum. However, this complicates the parent logit recovery in Eq. (8): without folded counts, parent logits are recovered as a simple average of child logits $S _ { i p } = $ $\textstyle { \frac { 1 } { \mathcal { C } } } \sum _ { c } S _ { i c }$ , which reduces over adjacent register entries at negligible cost. With modified logits, the counts must first be subtracted out: $\begin{array} { r } { S _ { i p } = \frac { 1 } { \mathcal { C } } \sum _ { c } ( S _ { i c } ^ { \prime } - \ln n _ { c } ) + } \end{array}$ ln $n _ { p } .$ , requiring per-child counts alongside the logits.

A simpler solution is to restrict the running maximum to non-empty codewords:

$$
m _ {i} = \max _ {j: n _ {j} > 0} S _ {i j}\tag{18}
$$

This requires only a masked comparison (one bit per codeword) rather than an extra floating-point value per codeword. Empty codewords contribute nothing (both ${ \bar { V } } _ { j } = 0 { \mathrm { ~ a n d ~ } } n _ { j } = 0 )$ , but their attention weights exp $S _ { i j } - m _ { i } )$ may overflow in reduced precision since $S _ { i j }$ can exceed $m _ { i }$ . We therefore zero out $A _ { i j }$ for empty codewords in the parent pass and $\varDelta A _ { i c }$ for empty children in the refinement pass, preventing floating-point artifacts $( \infty \times 0 = \mathrm { N a N } )$ .

## E Benchmark Details

All wall-clock kernel times reported in the paper (Fig. 5, Tab. 2, Tab. 6, and Tab. 3) are measured on a single NVIDIA RTX 3090 (Ampere, SM 8.6, 24 GB) at batch size B=4 and head dimension $D { = } 6 4$ , with $H { = } 1 2$ heads for ViT-Base/ImageNet and H=16 for DPT-Large/ADE20K. The synthetic-length sweep in Tab. 2 uses N ∈ {1024, 4096, 16384, 65536} (H=12). VQ and AVQ timings include both the VQ precompute kernel and the attention kernel, launched via our fused two-kernel pipeline. Flash Attention uses PyTorch’s scaled\_dot\_product\_attention with the Flash Attention 2 backend. These kernel timings use Triton’s do\_bench utility (100 ms warmup, 200 ms measurement window, median reported), averaged over 3 independent runs. The per-denoising-step UNet times in Tab. 4 instead time the full UNet forward with CUDA graphs, on the same RTX 3090 at batch size $B { = } 2$

Figure 6 plots our VQ and AVQ kernel times against efective codebook size at each N, empirically confirming the predicted linear scaling with both N and codebook size.

## F Training Details

ImageNet-1k. We use a ViT-Base backbone (patch size 8, 224×224 input, N=785 tokens including class token) pretrained on ImageNet-21k [10]. All 12 attention layers are replaced with $( \mathrm { A } ) \mathrm { V Q } \cdot$ -attention. We fine-tune for 4 epochs using

![](images/19a6f09ee4a631b2db7115cc73c2af112bca8cf2cdfbf7066ea6a36523437a55.jpg)  
VQ codes per key (M or $M _ { 0 } { + } C )$ Attn codes per query (M or $M _ { 0 } + P C )$  
Fig. 6: Empirical verification of the complexity analysis in Tab. 1. Kernel time vs. effective codebook size on log-log axes for $\mathrm { V Q }$ -attention (blue) and AVQ-attention (red), at four sequence lengths $N$ (light to dark). Gray dashed lines indicate slope 1 (linear scaling). All kernels scale linearly with N (equidistant lines on the log scale for equal 4× increases in $N )$ and linearly with the number of codes, confirming $\mathcal { O } ( ( M _ { 0 } { + } C ) N D )$ for $\mathrm { V Q }$ precompute and $\mathcal { O } ( ( M _ { 0 } + \mathcal { P } \mathcal { C } ) N D )$ for attention. The precompute kernels appear slightly sub-linear in the number of codes; this is a fixed per-kernel overhead (launch cost, memory setup) that is amortized as the codebook grows.

AdamW with learning rate $1 0 ^ { - 5 }$ , constant schedule with 1 epoch linear warmup, per-device batch size 64, and FP16 mixed precision. Data augmentation follows standard practice: RandomResizedCrop and RandomHorizontalFlip for training, Resize and CenterCrop for evaluation. Learnable key normalization (LayerNorm) stabilizes the key space and improves codebook training. The EMA decay is $\lambda = 0 . 9 9$ and the commitment loss weight is $\beta = 0 . 2 5$ . For flat VQ-attention, we sweep codebook sizes $M \in \{ 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ . For AVQ-attention, children are initialized at scale 0.1 around their parent (i.e., $C _ { p , c } = C _ { p } + 0 . 1 \cdot \epsilon , \epsilon \sim \mathcal { N } ( 0 , I ) )$ Queries are reordered along a Gilbert space-filling curve for spatial tile locality.

ADE20K. We use DPT-Large [27] (pretrained on ADE20K, 480×480 input, N=901 tokens) and replace all attention layers. We fine-tune for 10 epochs using AdamW with learning rate $1 0 ^ { - 5 }$ , constant schedule with 10% linear warmup, per-device batch size 8, and FP16 mixed precision. The EMA decay warms up from 0.9 to 0.99 over the first 2 epochs (cosine schedule). All other VQ hyperparameters match the ImageNet setting. We select the best checkpoint by validation mIoU.

Eficient-attention comparison. For Tab. 3, all methods—including $\mathrm { ( A ) V Q - }$ use a common 30-epoch ADE20K recipe, otherwise identical to the above. The longer schedule, together with dead-code handling for VQ (Sec. G), lifts the (A)VQ numbers relative to the 10-epoch main experiments, so the values in Tab. 3 form a separate controlled comparison and are not directly comparable to Fig. 5.

Table 6: Elastic inference $( M _ { 0 } { = } 6 4 , \mathcal { C } { = } 8 ) \colon$ : a single model trained with $\mathcal { P } { = } M _ { 0 }$ evaluated at varying P. Capture denotes the fraction of true attention weight on keys assigned to the selected P parents.

<table><tr><td>Eval  $\mathcal{P}$ </td><td>4</td><td>8</td><td>12</td><td>16</td><td>24</td><td>32</td></tr><tr><td colspan="7"> $ADE20K (N=901, H=16)$ </td></tr><tr><td>mIoU (%)</td><td>41.44</td><td>42.24</td><td>42.33</td><td>42.42</td><td>42.49</td><td>42.48</td></tr><tr><td>Capture (%)</td><td>49.7</td><td>68.0</td><td>78.5</td><td>85.3</td><td>93.1</td><td>97.0</td></tr><tr><td>Kernel ms</td><td>0.14</td><td>0.16</td><td>0.18</td><td>0.20</td><td>0.24</td><td>0.28</td></tr><tr><td colspan="7">ImageNet ( $N=785, H=12$ )</td></tr><tr><td>Acc (%)</td><td>79.23</td><td>81.16</td><td>81.69</td><td>82.04</td><td>82.22</td><td>82.29</td></tr><tr><td>Capture (%)</td><td>47.5</td><td>65.4</td><td>75.8</td><td>82.7</td><td>91.0</td><td>95.6</td></tr><tr><td>Kernel ms</td><td>0.10</td><td>0.11</td><td>0.13</td><td>0.14</td><td>0.17</td><td>0.20</td></tr></table>

## G Alternative VQ Training Methods

We learn the codebook with EMA-based online k-means (Sec. 4), a simple choice that we do not claim to be optimal. Codebook learning is largely orthogonal to the attention mechanism, so improvements developed for vector quantization in other settings transfer directly to (A)VQ-attention. We briefly explored replacing the straight-through estimator with a diferentiable quantizer (DiVeQ [32]) and per-batch dead-code handling that reassigns nearest keys to underused codewords. Both improved results in our setting. We nonetheless report the simpler EMA recipe throughout for consistency; we expect the accuracy–eficiency tradeofs we present to be conservative, with room for orthogonal advances in vector quantization to improve them further.

## H Results Analysis

Below we report several supporting analyses of the trained AVQ models.

Elastic inference. Table 6 reports task performance, top-P capture, and kernel time for a single AVQ model $( M _ { 0 } { = } 6 4 , \ C { = } 8 )$ trained at $\mathcal { P } { = } M _ { 0 }$ and evaluated across a range of P.

Per-head attention-mass concentration. Figure 7 reports top-P capture—the fraction of true attention mass on the keys assigned to a head’s P selected parents—for every attention head of the trained AVQ model $( M _ { 0 } { = } 6 4 , \ C { = } 8 ,$ evaluated at $\mathcal { P } { = } 8 )$ . Capture varies considerably across heads within a layer on both backbones, and the head-mean is higher in the later layers.

![](images/628de1f10db30513093dc5c974f844714e29e0915b02bad30bd438aca61ea446.jpg)  
Fig. 7: Per-head $\mathrm { t o p } { - } \mathcal { P }$ capture (fraction of true attention mass on the $\mathcal { P } { = } 8$ selected parents) for $\mathrm { A V Q }$ -attention $( M _ { 0 } { = } 6 4 , \mathcal { C } { = } 8 )$ , by layer: min–max across heads (shaded), individual heads (points), and the head mean (line). Left: ImageNet (ViT-Base, $H { = } 1 2 ;$ the final layer attends only from the CLS query). Right: ADE20K (DPT-Large, $H { = } 1 6 )$

Per-layer codebook analysis. All experiments in this paper apply the same codebook configuration $( M _ { 0 } , \mathcal { C } )$ uniformly across all attention layers. To investigate whether this is eficient, we run inference on our trained AVQ models and measure the per-layer commitment loss $\mathbb { E } \left\lceil \| k - \hat { k } \| ^ { 2 } \right\rceil$ , the mean squared distance between each key and its assigned codeword. Table 7 reports this for two codebook sizes on each dataset.

Commitment loss varies 3–4× across layers, with a consistent pattern across both datasets and codebook sizes: low in early layers, peaking in the middle layers (4–6), and decreasing again toward the end. In DPT-Large, the second half of the network (layers 12–23) has markedly lower commitment loss than the first half—below 5 throughout at $M _ { 0 } { = } 1 2 8$ . Layer 0 combines low commitment loss with low codebook utilization (59–69% active ratio at $M _ { 0 } { = } 6 4 )$ ).

Spatial reordering ablation. As described in Sec. 4, we reorder queries along a Gilbert space-filling curve so that contiguous tiles form spatially compact 2D regions. Since each tile independently selects which P parents to refine, spatial coherence ensures that queries sharing a refinement decision attend to similar parts of the image. We ablate this by evaluating a trained model $( M _ { 0 } { = } 6 4 , \mathcal { P } { = } 1 6 .$ $\scriptstyle { \mathcal { C } } = 8 )$ under three query orderings—Gilbert (training configuration), raster (no reordering, tiles are thin horizontal strips), and random (destroys all locality)— and three tile sizes (Tab. 8).

On ADE20K, Gilbert reordering improves mIoU by ∼0.2 percentage points over raster order across all tile sizes; on ImageNet, Gilbert and raster perform identically (∼82.0%). In the ImageNet runs the final layer computes attention only for the CLS query, so patch-token ordering does not afect its parent selection. Random permutation is clearly harmful in both settings (−1.8% accuracy on ImageNet, −0.8% mIoU on ADE20K). We observe no meaningful diference across tile sizes at the current sequence lengths.

Table 7: Per-layer commitment loss $\mathbb { E } [ \| k - \hat { k } \| ^ { 2 } ]$ for AVQ-attention (C=8). Higher values may indicate that the codebook provides insuficient resolution for that layer’s key distribution. This metric depends only on the codebook geometry and is independent of P.

<table><tr><td colspan="3">ImageNet (ViT-Base)</td><td colspan="6">ADE20K (DPT-Large)</td></tr><tr><td>Layer</td><td> $M_0=64$ </td><td> $M_0=128$ </td><td>Layer</td><td> $M_0=64$ </td><td> $M_0=128$ </td><td>Layer</td><td> $M_0=64$ </td><td> $M_0=128$ </td></tr><tr><td>0</td><td>8.0</td><td>7.2</td><td>0</td><td>6.9</td><td>4.7</td><td>12</td><td>11.5</td><td>4.7</td></tr><tr><td>1</td><td>13.5</td><td>11.6</td><td>1</td><td>10.5</td><td>5.1</td><td>13</td><td>8.8</td><td>3.3</td></tr><tr><td>2</td><td>23.1</td><td>23.0</td><td>2</td><td>15.0</td><td>7.3</td><td>14</td><td>6.5</td><td>2.3</td></tr><tr><td>3</td><td>21.9</td><td>22.9</td><td>3</td><td>20.0</td><td>10.8</td><td>15</td><td>6.3</td><td>2.2</td></tr><tr><td>4</td><td>23.8</td><td>25.0</td><td>4</td><td>24.3</td><td>14.6</td><td>16</td><td>6.7</td><td>2.4</td></tr><tr><td>5</td><td>27.2</td><td>30.1</td><td>5</td><td>21.1</td><td>11.9</td><td>17</td><td>7.3</td><td>2.6</td></tr><tr><td>6</td><td>24.4</td><td>27.8</td><td>6</td><td>22.0</td><td>12.4</td><td>18</td><td>6.0</td><td>2.0</td></tr><tr><td>7</td><td>22.4</td><td>26.6</td><td>7</td><td>22.8</td><td>12.3</td><td>19</td><td>4.3</td><td>1.3</td></tr><tr><td>8</td><td>19.8</td><td>24.1</td><td>8</td><td>21.8</td><td>12.5</td><td>20</td><td>4.6</td><td>1.3</td></tr><tr><td>9</td><td>16.9</td><td>22.5</td><td>9</td><td>17.3</td><td>8.3</td><td>21</td><td>5.2</td><td>1.4</td></tr><tr><td>10</td><td>14.6</td><td>21.2</td><td>10</td><td>18.1</td><td>9.4</td><td>22</td><td>6.2</td><td>1.7</td></tr><tr><td>11</td><td>9.0</td><td>14.7</td><td>11</td><td>14.1</td><td>6.1</td><td>23</td><td>5.9</td><td>1.6</td></tr></table>

Table 8: Efect of query reordering on AVQ-attention $( M _ { 0 } { = } 6 4 , \mathcal { P } { = } 1 6 , \mathcal { C } { = } 8 )$ for ImageNet classification (accuracy %) and ADE20K segmentation (mIoU %). Random results averaged over 2 seeds.

<table><tr><td rowspan="2">Ordering</td><td colspan="3">ImageNet (acc. %)</td><td colspan="3">ADE20K (mIoU %)</td></tr><tr><td>32</td><td>64</td><td>128</td><td>32</td><td>64</td><td>128</td></tr><tr><td>Gilbert</td><td>82.04</td><td>82.03</td><td>82.08</td><td>42.40</td><td>42.35</td><td>42.37</td></tr><tr><td>Raster</td><td>82.07</td><td>82.00</td><td>82.02</td><td>42.16</td><td>42.20</td><td>42.15</td></tr><tr><td>Random</td><td>80.18</td><td>80.20</td><td>80.22</td><td>41.37</td><td>41.39</td><td>41.40</td></tr></table>

Codebook utilization. A practical concern with VQ-attention is codebook utilization: as M grows, some codewords attract few or no keys. We measure this via the active ratio: the fraction of codewords whose assignment count over the full validation set exceeds 1% of the fair-share count $N _ { \mathrm { v a l } } / M _ { \mathrm { t o t a l } }$ (e.g., $0 . 0 1 \times 5 0 , 0 0 0 \times 7 8 5 / 1 2 8 \approx 3 , 0 0 0$ for ImageNet at M=128), averaged across heads and layers.

Table 9 reports active ratios for flat VQ-attention at varying M and for AVQattention across several configurations, under identical training settings.

Flat VQ exhibits a clear downward trend in utilization as M grows: at M=512, nearly 10% of codewords fall below the active threshold. AVQ maintains substantially higher utilization at much larger total codebook sizes: at $M _ { \mathrm { t o t a l } } { = } 5 7 6$ AVQ achieves 97% utilization, and even at $M _ { \mathrm { t o t a l } } { = } 1 1 5 2$ it retains over 95%— better than flat VQ at M=256. The hierarchical structure naturally encourages full codebook usage: children are initialized near their parent, placing them in regions of key space where keys already concentrate.

Table 9: Codebook active ratio for flat VQ and AVQ under identical training settings. Flat VQ utilization degrades as M grows, while AVQ maintains high utilization even at much larger total codebook sizes. The active ratio measures what fraction of codewords receive key assignments, forming the pool of codewords available for attention. In AVQ-attention, each query attends to only a subset of this pool, with diferent queries selecting diferent subsets.

<table><tr><td>Method</td><td> $M_{\text{total}}$ </td><td>Active ratio</td></tr><tr><td>Flat VQ,  $M=64$ </td><td>64</td><td>97.6%</td></tr><tr><td>Flat VQ,  $M=128$ </td><td>128</td><td>95.7%</td></tr><tr><td>Flat VQ,  $M=256$ </td><td>256</td><td>93.4%</td></tr><tr><td>Flat VQ,  $M=512$ </td><td>512</td><td>90.4%</td></tr><tr><td>AVQ, 64/8/8</td><td>576</td><td>97.0%</td></tr><tr><td>AVQ, 64/16/8</td><td>576</td><td>97.0%</td></tr><tr><td>AVQ, 128/8/8</td><td>1152</td><td>95.4%</td></tr><tr><td>AVQ, 128/16/8</td><td>1152</td><td>95.5%</td></tr></table>