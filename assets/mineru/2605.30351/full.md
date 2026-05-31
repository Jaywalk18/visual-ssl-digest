# VideoMLA: Low-Rank Latent KV Cache for Minute-Scale Autoregressive Video Diffusion

Hidir Yesiltepe1 Jiazhen Hu1 Tuna Han Salih Meral1 Adil Kaan Akan2 Kaan Oktay2 Hoda Eldardiry1 Pinar Yanardag1 1Virginia Tech 2 fal

Project Page: https://videomla.github.io

# Abstract

Long-rollout causal video diffusion has converged on a fixed-size sliding-window KV cache, with recent progress innovating within this layout by changing which tokens occupy the window or how their positions are encoded. The per-head KV layout itself, a dominant contributor to streaming memory and latency, has been mostly left unchanged. In this paper, we present the first study of Multi-Head Latent Attention (MLA) in video diffusion. VideoMLA replaces per-head keys and values with a shared low-rank content latent and a shared decoupled 3D-RoPE positional key, reducing per-token KV memory by 92.7% at every cached layer. We further investigate why MLA succeeds in video diffusion even though the spectral assumption often used to motivate it in language models does not hold: pretrained video attention is not low-rank, with 99%-energy effective rank far above any practical latent dimension. VideoMLA retains quality at compression ratios where direct spectral approximation would predict large reconstruction error. We show that the MLA bottleneck, rather than the pretrained spectrum, determines the effective rank: both spectral and random initialization occupy nearly the full rank budget from initialization, and training preserves this budget while adapting within it. On VBench, VideoMLA matches short-horizon streaming video diffusion baselines, achieves the best overall score at long horizons among evaluated methods, and improves throughput by 1.23× on a single B200.

# 1 Introduction

Causal video diffusion models [9, 4, 16, 31, 25, 24, 17, 27, 22, 3, 23, 12, 30, 13] have gained traction as the dominant approach to streaming, long-horizon video generation. Distilled from bidirectional teachers, they generate frames [31, 5, 11, 8] or chunks autoregressively while attending to a rolling key-value (KV) cache of past frames, producing minute-long videos at interactive rates on a single GPU. As models scale toward longer rollouts, the per-head KV cache increasingly defines the operating point. At Wan-1.3B scale [21], each cached token stores 2 × 12 × 128 = 3,072 dense KV scalars per layer, accounting for keys and values across 12 heads with 128 channels each. With a 21-latent-frame cache, 1,560 tokens per latent frame, and 30 transformer layers, the dense KV cache contains 3.02B scalars, or about 6.0GB in bf16/fp16. This footprint explains why recent streaming systems use fixed-size sliding-window caches: retaining all past KV states would grow linearly with rollout length. However, fixing the window only bounds the number of cached tokens; it does not reduce the per-token, per-layer cost of the per-head KV layout. Reducing this layout is therefore a direct lever for longer horizons, larger batches, and faster inference.

The dominant line of recent work treats the cache as a fixed-size sliding window and innovates inside it. CausVid [27] initiated this thread by converting bidirectional diffusion into causal autoregressive generation via distribution matching distillation, with a sliding KV cache from inception.

![](images/f0c54f0fb1367aec6696bcb09af7cb67aa8fe0e9fc9c909c539dc85e2fb6156e.jpg)

<details>
<summary>line</summary>

| Rank k | σ_h / σ_1 |
| ------ | --------- |
| 10^0   | 10^0      |
| 10^1   | ~10^-1    |
| 10^2   | ~10^-2    |
| 10^3   | ~10^-3    |
</details>

![](images/94698d13b0621155d5281390285f710c73583d002553a8d1040569e866f24be9.jpg)

<details>
<summary>line</summary>

| Rank k | Cumulative energy E(k) |
| ------ | ------------------------ |
| 0      | 0.0                      |
| 200    | 0.5                      |
| 400    | 0.7                      |
| 600    | 0.8                      |
| 700    | 0.9                      |
</details>

![](images/0255df1dc865fc49b8e91296a4334fd27c1fc09b93a7dfac74888f4d0848271e.jpg)

<details>
<summary>line</summary>

| Layer index | τ = 0.90 | τ = 0.95 | τ = 0.99 |
| ----------- | -------- | -------- | -------- |
| 0           | 900      | 1100     | 1400     |
| 5           | 850      | 1050     | 1350     |
| 10          | 870      | 1070     | 1360     |
| 15          | 860      | 1060     | 1340     |
| 20          | 880      | 1120     | 1380     |
| 25          | 900      | 1080     | 1370     |
| 30          | 920      | 1100     | 1390     |
</details>

Figure 1: Pretrained video diffusion attention is not low-rank, unlike in language models. Singular value analysis of $[ W _ { K } ; W _ { V } ] \in \mathbb { R } ^ { 3 0 7 2 \times 1 5 3 6 }$ across the 30 transformer blocks of Wan2.1- T2V-1.3B. At $d _ { c } = 1 9 2$ , the median layer captures only $E _ { \mathrm { m e d } } = 0 . 4 5 8$ of the spectral energy, and the 99%-energy effective rank exceeds 1300 in every layer.

Self-Forcing [9] closed the train–test gap by conditioning training on self-generated frames within the same rolling cache. Subsequent work refined this recipe through attention-sink, token-selection, and compressed-memory mechanisms for long-range consistency [16, 25, 17, 28, 12, 29], training strategies for multi-minute rollouts and prompt switching [4, 22, 7, 24], improved distillation objectives [31, 17], and positional reparameterization such as Infinity-RoPE [24]. However, these methods all preserve the per-head KV layout that fills the window in the first place: they redistribute, reweight, compress over time, or reposition cached tokens without reducing the per-token KV state.

A second, complementary line changes the attention computation itself. SANA-Video [3] replaces softmax attention with block-causal linear attention, removing the conventional KV cache and using a constant-memory cumulative state for long-video generation. SCD [2] reduces cached state by routing temporal reasoning through a 25-layer causal encoder and using a 10-layer frame-wise decoder, so only the encoder layers cache. Under the same Wan cache geometry, this reduces dense KV storage by 16.7%. VideoMLA is orthogonal: it keeps all 30 self-attention layers cached but reduces each token’s cached state from 3072 to 224 scalars, yielding an 11.4× smaller cache than SCD for the same 21-latent-frame window. Thus, rather than changing which tokens are cached, how they are positioned, or how many layers cache, VideoMLA targets the remaining factor directly: the per-token KV layout at every cached self-attention layer.

In this paper, we intervene on the per-head layout itself. Building on Multi-Head Latent Attention (MLA) [14], we present VideoMLA, the first MLA-style latent KV cache for autoregressive video diffusion. VideoMLA replaces per-head keys and values with a shared low-rank content latent and a head-shared decoupled 3D-RoPE positional key, reducing per-token KV memory by 92.7% at every cached layer. This raises a puzzle: MLA is usually motivated by low-rank pretrained $W _ { K } , W _ { V }$ [14, 10], yet Wan-1.3B (Fig. 1) has 99%-energy rank far above practical latent dimensions. VideoMLA nonetheless retains quality where direct spectral approximation would incur large reconstruction error (Fig. 2). We show that the MLA bottleneck, not the pretrained spectrum, determines the effective rank: SVD and random initialization both nearly saturate the rank budget, which training preserves with little spectral change. The design question therefore shifts from what is the intrinsic rank? to what latent budget preserves video quality? Our contributions are summarized as follows:

• Latent KV caching for video diffusion. We introduce VideoMLA, an MLA-style autoregressive video diffusion model that replaces per-head keys and values with a shared content latent and a head-shared decoupled 3D-RoPE key, reducing per-token KV memory by 92.7% at every cached layer.   
• A spectral puzzle and rank-budgeted resolution. We show that Wan-1.3B video attention is not low-rank: the 99%-energy effective rank of $\left[ W _ { K } ; W _ { V } \right]$ far exceeds practical latent dimensions. VideoMLA nevertheless retains quality, while both SVD and random initialization saturate the imposed rank budget from initialization and preserve it during training.   
• Efficient long-horizon generation. We identify the NoPE/RoPE allocation that preserves visual fidelity and motion consistency at minute-scale horizons. On VBench, VideoMLA matches short-horizon baselines, achieves the best long-horizon overall score among evaluated methods, and improves throughput by 1.23× on a single B200.

![](images/e6b4f251b4e30aea8f54ed8eab0b598391194474597399d881ccdfdb57c1aa59.jpg)

<details>
<summary>line</summary>

| Rank k | σ_h/σ_1 (d_c = 64) | σ_h/σ_1 (d_c = 128) | σ_h/σ_1 (d_c = 256) | σ_h/σ_1 (d_c = 512) |
| ------ | ------------------- | -------------------- | -------------------- | -------------------- |
| 10^0   | ~1.0                | ~1.0                 | ~1.0                 | ~1.0                 |
| 10^1   | ~0.3                | ~0.3                 | ~0.3                 | ~0.3                 |
| 10^2   | ~0.1                | ~0.1                 | ~0.1                 | ~0.1                 |
| 10^3   | ~0.05               | ~0.05                | ~0.05                | ~0.05                |
</details>

![](images/523245b0572265333808f4ac67e18664e17c19e295710c80f21175eb49122075.jpg)

<details>
<summary>line</summary>

| Rank k | d_c = 64 | d_c = 128 | d_c = 256 | d_c = 512 |
| ------ | -------- | --------- | --------- | --------- |
| 0      | 0.0      | 0.0       | 0.0       | 0.0       |
| 100    | 1.0      | 0.8       | 0.6       | 0.4       |
| 200    | 1.0      | 0.95      | 0.75      | 0.55      |
| 300    | 1.0      | 1.0       | 0.85      | 0.65      |
| 400    | 1.0      | 1.0       | 0.9       | 0.75      |
| 500    | 1.0      | 1.0       | 0.95      | 0.85      |
| 600    | 1.0      | 1.0       | 1.0       | 0.95      |
</details>

![](images/8eb3f23f60b9e1146527052c80e6702df3a7e997af8b234f3d3a67933e4472a7.jpg)

<details>
<summary>line</summary>

| Layer index | d_e = 64 | d_e = 128 | d_e = 256 | d_e = 512 |
| ----------- | -------- | --------- | --------- | --------- |
| 0           | 70       | 130       | 250       | 500       |
| 5           | 70       | 130       | 250       | 500       |
| 10          | 70       | 130       | 250       | 500       |
| 15          | 70       | 130       | 250       | 500       |
| 20          | 70       | 130       | 250       | 500       |
| 25          | 70       | 130       | 250       | 500       |
| 30          | 70       | 130       | 250       | 500       |
</details>

Figure 2: The composed operator occupies its full rank-dc budget at every $d _ { c }$ and every layer. Singular value analysis of the composed operator $M _ { \mathrm { l e a r n e d } } \ = \ [ \breve { W } _ { \uparrow } ^ { K } W _ { \downarrow } ^ { K V } ; \breve { W } _ { \uparrow } ^ { V } W _ { \downarrow } ^ { K V } ]$ for SVDinitialized VideoMLA students at $d _ { c } \in \{ 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ . (a) Median normalized spectra share a common envelope, truncated at $d _ { c } .$ (b) Cumulative spectral energy. (c) Layer-wise 99%-energy effective rank: $r _ { 0 . 9 9 } \approx 0 . 9 8 d _ { c }$ at every budget, uniformly across depth. The composed operator’s rank is determined by the architectural bottleneck, not by the spectral structure of the dense source.

# 2 Related Work

Causal Video Generation. Causal video diffusion converts a bidirectional teacher into a streaming student that generates frames or chunks autoregressively with a rolling KV cache. CausVid [27] initiated this line with Distribution Matching Distillation (DMD) [26] based causal distillation, and Self-Forcing [9] reduced train–test mismatch by training on self-generated rollouts. Subsequent work improves long-horizon stability through joint denoising and attention sinks [16], teacher-guided correction [4], causal ODE initialization [31], reward-weighted distillation and EMA sinks [17], deep sink and cache pruning [25], KV recaching for prompt switches [22], and block-relative temporal RoPE [24]. These methods improve what is stored in the window or how it is positioned, but retain the dense per-head KV layout.

Efficient Causal Video Generation. A complementary line restructures attention to reduce memory or compute. SANA-Video [3] replaces softmax with block-causal linear attention and uses a constantsize cumulative state. SCD [2] separates temporal reasoning from frame-wise rendering, caching only the causal encoder. VideoSSM [28] augments sliding-window KV with an SSM-compressed global memory. These approaches reduce temporal or layer-wise memory, but do not compress the per-token, per-head KV state at every cached layer.

Multi-Head Latent Attention. DeepSeek-V2 [14] introduced Multi-Head Latent Attention (MLA), replacing per-head KV with a shared low-rank latent and a decoupled positional key; DeepSeek-V3 [15] scaled this design. MTLA [6] further compresses along time, while MHA2MLA [10] and TransMLA [18] convert pretrained MHA [20] or GQA [1] LLMs into MLA. These works target language deployment. We study MLA in video diffusion, where the memory profile and pretrained attention spectrum differ substantially.

# 3 Method

We write $x _ { t } \in \mathbb { R } ^ { d }$ for the attention input at current chunk t, where a chunk denotes a group of latent frames. Let d be the model dimension, $n _ { h }$ the number of heads, and $d _ { h }$ the per-head dimension, so that $d = n _ { h } d _ { h }$ . VideoMLA introduces a shared KV latent dimension $d _ { c }$ for cached content and splits each head into a NoPE content-scoring subspace and a RoPE positional subspace, The NoPE part is reconstructed from the shared latent and is not rotary-position $d _ { h } = d _ { h } ^ { \mathrm { n o p e } } + d _ { h } ^ { \mathrm { i o p e } }$ part uses a head-shared decoupled 3D-RoPE key.

# 3.1 Compressed KV Cache Construction

Each video latent token $x _ { t } \in \mathbb { R } ^ { d }$ produced by the backbone is first compressed into a low-rank latent that summarizes its key and value content for the rolling cache:

$$
c _ {t} ^ {K V} = W _ {\downarrow} ^ {K V} x _ {t} \in \mathbb {R} ^ {d _ {c}}, \tag {1}
$$

![](images/069c074c9c9e2aae70044f4121df26bc223b8ed1fd23fd849d368f427b76011b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Multi-Head Attention (MHA)"] --> B["Value Up"]
    A --> C["Key Up"]
    A --> D["Decoupled 3D-RoPE"]
    B --> E["KV Down"]
    C --> F["Key Rotation"]
    D --> G["Query Rotation"]
    D --> H["Query Up"]
    E --> I["Compressed KV Cache cKV"]
    F --> J["Video Latent xt"]
    G --> K["Query Down"]
    H --> L["Video Latent xt"]
    M["channel concat"] --> N["kt_rope"]
    O["token concat"] --> P["qt_rope"]
    Q["+"] --> R["+"]
    S["+"] --> T["+"]
    U["+"] --> V["+"]
    W["t"] --> X["Value Up"]
    Y["kt_nope"] --> Z["Key Up"]
    AA["dt_nope"] --> AB["Decoupled 3D-RoPE"]
    AC["dt_nope"] --> AD["Query Rotation"]
    AE["dt_nope"] --> AF["Query Up"]
    AG["dt_nope"] --> AH["Query Down"]
    AI["dt_nope"] --> AJ["Video Latent xt"]
    AK["dt_nope"] --> AL["Key Rotation"]
    AM["dt_nope"] --> AN["Query Rotation"]
    AO["dt_nope"] --> AP["Query Rotation"]
    AQ["dt_nope"] --> AR["Query Rotation"]
    AS["dt_nope"] --> AT["Query Rotation"]
    AU["dt_nope"] --> AV["Query Rotation"]
    AW["dt_nope"] --> AX["Query Rotation"]
    AY["w↑^V"] --> AZ["Value Up"]
    BA["c↑^KV"] --> BB["Key Up"]
    BC["w↓^KV"] --> BD["Key Rotation"]
    BE["w↓^K"] --> BF["Key Rotation"]
    BG["w↓^Q"] --> BH["Query Rotation"]
    BI["w↓^Q"] --> BJ["Query Rotation"]
```
</details>

Figure 3: Overview of VideoMLA. VideoMLA replaces dense per-head KV cache in Causal Wan 2.1- 1.3B with a low-rank latent obtained by jointly compressing keys and values through shared down/up projections, with positional information carried by a single decoupled rotated key. Orange blocks denote down projections, green blocks denote rotations, and white blocks denote up projections; latent frames are colored blue for the key/value stream and white for the query stream. Each block is annotated with the corresponding weight matrix from Section $^ { 3 , }$ named latents are shown in red.

where $W _ { \perp } ^ { K V } \in \mathbb { R } ^ { d _ { c } \times d }$ is the joint KV down-projection (Fig. 3, KV Down). The vector $c _ { t } ^ { K V }$ is the content object written into the compressed KV cache. It has dimension $d _ { c } \ll n _ { h } d _ { h } .$ , so it replaces the dense per-head content keys and values that would otherwise be stored for every head. Positional information is not folded into this latent; it is stored separately through the decoupled key $k _ { t } ^ { R }$ introduced in Section 3.2. Thus, each cached token stores the pair $( c _ { t } ^ { K V } , k _ { t } ^ { R } )$ rather than dense per-head KV states.

The per-head keys and values needed by attention are obtained from $c _ { t } ^ { K V }$ through two up-projections,

$$
k _ {t, h} ^ {\text { nope }} = W _ {\uparrow , h} ^ {K} c _ {t} ^ {K V}, \quad v _ {t, h} = W _ {\uparrow , h} ^ {V} c _ {t} ^ {K V}, \tag {2}
$$

where $h \in \{ 1 , \ldots , n _ { h } \}$ indexes attention heads and $W _ { \uparrow } ^ { K } , W _ { \uparrow } ^ { V }$ are the key and value up-projections (Fig. 3, Key Up and Value Up). Two properties of this construction are important. First, the same cached latent $c _ { t } ^ { K V }$ is shared across all heads: a single cache read produces $n _ { h }$ per-head keys and $n _ { h }$ per-head values through Eq. 2. Second, the reconstructed key cIt is the content-only component of the per-head key, denoted $k _ { t , h } ^ { \mathrm { n o p e } }$ no rotary positional information.; the positional component lives in the separate RoPE subspace.

Together with the decoupled positional key, the per-token cached state is reduced from the $2 n _ { h } d _ { h }$ scalars of a dense per-head KV cache to $d _ { c } + d _ { h } ^ { \mathrm { r o p e } }$ scalars. In our default setting, this is 224 scalars per token per layer, a 92.7% reduction.

The query path is per-token and uses an analogous down/up structure. From $x _ { t }$ , a query downprojection produces a query latent, and a content up-projection recovers the per-head NoPE query:

$$
c _ {t} ^ {Q} = W _ {\downarrow} ^ {Q} x _ {t} \in \mathbb {R} ^ {d _ {q}}, \tag {3}
$$

$$
q _ {t, h} ^ {\text { nope }} = W _ {\uparrow , h} ^ {Q} c _ {t} ^ {Q}, \tag {4}
$$

where $d _ { q }$ is the query latent dimension and $W _ { \uparrow } ^ { Q }$ is the Query Up projection in Fig. 3. Since queries are recomputed from the current block at every generation step, $c _ { t } ^ { Q }$ is internal to the layer and is never written to the KV cache. The head-sharing occurs only in the decoupled positional branch: VideoMLA uses a single RoPE key shared across heads, while the NoPE queries, NoPE keys, and values remain head-specific after up-projection.

<table><tr><td>Metric</td><td>Causal Full</td><td>Causal Local</td><td>Causal Linear</td><td>MLA Local</td></tr><tr><td>Memory</td><td>2ND</td><td>2WD</td><td> $D d_h$ </td><td> $W(d_c + d_h^{\text{rope}})$ </td></tr><tr><td>Comp. (N-th token)</td><td>ND</td><td>WD</td><td> $D d_h$ </td><td> $nW(d_c + d_h^{\text{rope}})$ </td></tr><tr><td>Comp. (N tokens)</td><td> $\frac{1}{2} N^2 D$ </td><td>NWD</td><td> $N D d_h$ </td><td> $nNW(d_c + d_h^{\text{rope}})$ </td></tr></table>

Table 1: Memory and compute costs across four attention variants. For a sequence of length N with hidden dimension $D ,$ , n heads, per-head dimension $d _ { h } = D / n$ , local window $W$ , latent KV dimension $d _ { c } ( d _ { c } \ll D )$ , and shared decoupled-RoPE dimension $d _ { h } ^ { \mathrm { { \mathrm { f o p e } } } }$ .

The dimension $d _ { c }$ is the layer’s main content-cache capacity knob: it controls how aggressively the cached content is compressed and how much shared subspace the model can use for joint key-value content. The choice of $d _ { c }$ is studied empirically in Figure 7 and Appendix.

# 3.2 Decoupled 3D-RoPE

The latent cache $c _ { t } ^ { K V }$ is kept position-free, so that the low-rank content path can be shared across heads and reused under sliding-window re-inseparate RoPE subspace. We split each head as $d _ { h } = d _ { h } ^ { \mathrm { n o p e } } + d _ { h } ^ { \mathrm { r o p e } }$ format, where $k _ { t , h } ^ { \mathrm { n o p e } }$ instead carried by a is the reconstructed content key and the remaining channels form a decoupled 3D-RoPE key. As in Wan, $d _ { h } ^ { \mathrm { r o p e } }$ is partitioned across temporal, height, and width axes, using the corresponding high-frequency rotary bands.

For each token, VideoMLA computes a single head-shared positional key

$$
k _ {t} ^ {R} = W _ {R} ^ {K} x _ {t} \in \mathbb {R} ^ {d _ {h} ^ {\text { rope }}}, \quad k _ {t} ^ {\text { rope }} = \mathrm{RoPE} _ {3 D} (k _ {t} ^ {R}), \tag {5}
$$

rather than $n _ { h }$ per-head RoPE keys. The cache stores the unrotated state $( c _ { t } ^ { K V } , k _ { t } ^ { R } )$ ; rotation is applied only when the active attention window is assemblabsolute rollout time and yields a per-token cache size of $d _ { c } + d _ { h } ^ { \mathrm { r o p e } }$ ps cached states independent of.

The query branch follows the same decomposition. From the query latent $c _ { t } ^ { Q }$ , the positional query for head h is

$$
q _ {t, h} ^ {R} = W _ {R, h} ^ {Q} c _ {t} ^ {Q}, \quad q _ {t, h} ^ {\text { rope }} = \mathrm{RoPE} _ {3 D} (q _ {t, h} ^ {R}). \tag {6}
$$

$( q _ { t , h } ^ { \mathrm { n o p e } } , q _ { t , h } ^ { \mathrm { r o p e } } )$ then com against $( k _ { t , h } ^ { \mathrm { n o p e } } , k _ { t } ^ { \mathrm { r o p e } } )$ e concatenated NoPE and RoPE components: each head uses, while values remain reconstructed only from the content latent.

# 3.3 Training-Time Forward Pass

During training, every video latent token writes its compressed cache state $( c _ { t } ^ { K V } , k _ { t } ^ { R } )$ , defined in Eqs. 1 and 5, into the KV cache as the block under denoising progresses. Attention is then computed in standard multi-head form, with the per-head content keys and values reconstructed on demand from the cached content latent through Eq. 2, and the shared positional key obtained by rotating the cached positional state $k _ { t } ^ { R }$ at use time.

For a query token at position i and a cached token at position $j ,$ attention head h combines the content and positional contributions into a single score

$$
\operatorname{score} _ {i, j} ^ {(h)} = \frac {q _ {i , h} ^ {\text { nope }} \cdot k _ {j , h} ^ {\text { nope }} + q _ {i , h} ^ {\text { rope }} \cdot k _ {j} ^ {\text { rope }}}{\sqrt {d _ {h} ^ {\text { nope }} + d _ {h} ^ {\text { rope}}}}, \tag {7}
$$

where qnopi,h $q _ { i , h } ^ { \mathrm { n o p e } }$ and qropi,h $q _ { i , h } ^ { \mathrm { r o p e } }$ are the content and rotated positional query components from Eqs. 4 and 6, $k _ { j , h } ^ { \mathrm { n o p e } }$ is the per-head content key from Eq. 2, and $k _ { j } ^ { \mathrm { r o p e } }$ is the rotated shared positional key obtainednope rope from Eq. 5. The two inner products live in subspaces of dimension $d _ { h } ^ { \mathrm { n o p e } }$ and $d _ { h } ^ { \mathrm { r o p e } }$ respectively, so the joint score is normalized by their combined dimension. A softmax over the active attention window followed by a weighted sum of $v _ { j , h }$ produces the per-head output, and the head outputs are mixed through the output projection $W ^ { O }$ .

![](images/2e9eb89ed6a6fceec9013822959435e8efd10af8a7c29d1ec23c5d001c2b1515.jpg)

<details>
<summary>natural_image</summary>

Collage of eight photos showing a kangaroo in a tent reading, overlooking a coastal cliff and ocean waves, with a person wearing a helmet in the foreground (no text or symbols)
</details>

Figure 4: Qualitative results. Samples generated by VideoMLA. Frames are shown at uniformly spaced timestamps from each 30s rollout, illustrating that the compressed latent KV cache preserves scene structure, subject identity, and visual fidelity over time.

The shape of score(h)i,j $\mathrm { s c o r e } _ { i , j } ^ { ( h ) }$ matches what a dense attention layer of the same per-head dimension would produce. As a consequence, VideoMLA substitutes for the dense self-attention module without any change to the surrounding training pipeline: chunkwise causal block masks, sink tokens, and FlexAttention kernels operate on the reconstructed per-head keys and values exactly as they would on dense ones. The only structural change relative to the dense baseline is internal to the attention layer: the cache holds by attention are recons $( \check { c } _ { t } ^ { K V } , k _ { t } ^ { R } )$ rather than per-head K and V , and the per-head views consumeduse time.

# 4 Experiments

# 4.1 Setup and Dataset

Implementation Details. We implement VideoMLA on top of the Wan-2.1 T2V-1.3B backbone [21], replacing only the self-attention layers while leaving the remaining architecture unchanged. The model has 30 transformer blocks, hidden dimension 1536, 12 heads, and per-head dimension 128. rwis and widt $d _ { c } ~ = ~ 1 9 2$ and ed 3 pairs $d _ { q } \ = \ 7 6 8$ , with the head dimension split intoannels are allocated across temporal, highest-frequency bands. This gives $d _ { h } ^ { \mathrm { n o p e } } = 9 6$ $d _ { h } ^ { \mathrm { r o p e } } = 3 2$ $( 6 , 5 , 5 )$ a per-token cache size of $\dot { d _ { c } } + \dot { d } _ { h } ^ { \mathrm { r o p e } } = \dot { 2 } 2 4$ scalars, corresponding to a $1 3 . 7 \times$ reduction from the dense $2 n _ { h } d _ { h } = 3 0 7 2$ -scalar KV cache. Training follows the three-stage Causal Forcing pipeline [31], including Teacher Forcing, Consistency Distillation initialization to four steps, and DMD, with total batch size 128. We use learning rates $5 \times 1 0 ^ { - 6 }$ for Teacher Forcing and $2 \times 1 0 ^ { - 6 }$ for Consistency Distillation and DMD. All training experiments are run in bf16 mixed precision on a 8 × B200 GPU.

Dataset. For the Consistency Distillation stage preceding DMD, we use 47,680 videos: 29,471 from OpenVid-1M [19] and 18,209 synthesized clips.

Baselines. We compare VideoMLA with recent causal video diffusion methods covering standard streaming pipelines, attention-architecture redesigns, and positional reparameterizations. The streaming baselines include CausVid [27], Self-Forcing [9], Rolling-Forcing [16], Causal Forcing [31],

![](images/da18b3599b353566291a7f63ad7b9d42ec667cf3ca53448e14f92c1ed30c8740.jpg)

<details>
<summary>text_image</summary>

Self Forcing
Causal Forcing
LongLive
Reward Forcing
VideoMLA
CausVid
Rolling Forcing
Deep Forcing
Infinity-RePE
</details>

Figure 5: Qualitative comparison. Long-rollout samples from VideoMLA and baseline causal video diffusion baselines under the same prompt. Each row shows uniformly spaced frames from one method.

<table><tr><td rowspan="2">Model</td><td colspan="7">Results on 30s ↑</td><td colspan="7">Results on 60s ↑</td><td colspan="4">User Study ↑</td></tr><tr><td>AQ</td><td>BC</td><td>DD</td><td>IQ</td><td>MS</td><td>SC</td><td>Overall</td><td>AQ</td><td>BC</td><td>DD</td><td>IQ</td><td>MS</td><td>SC</td><td>Overall</td><td>PA</td><td>TC</td><td>DC</td><td>Overall</td></tr><tr><td>Self-Forcing [9]</td><td>0.541</td><td>0.948</td><td>0.624</td><td>0.577</td><td>0.952</td><td>0.932</td><td>0.762</td><td>0.565</td><td>0.958</td><td>0.393</td><td>0.650</td><td>0.987</td><td>0.974</td><td>0.755</td><td>2.79</td><td>2.79</td><td>2.70</td><td>2.76</td></tr><tr><td>CausVid [27]</td><td>0.597</td><td>0.921</td><td>0.473</td><td>0.663</td><td>0.935</td><td>0.913</td><td>0.750</td><td>0.497</td><td>0.929</td><td>0.723</td><td>0.574</td><td>0.948</td><td>0.933</td><td>0.767</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Causal Forcing [31]</td><td>0.526</td><td>0.945</td><td>0.738</td><td>0.628</td><td>0.968</td><td>0.947</td><td>0.792</td><td>0.503</td><td>0.936</td><td> $\underline{0.847}$ </td><td>0.608</td><td>0.935</td><td>0.920</td><td>0.792</td><td>2.59</td><td>2.63</td><td> $\underline{2.81}$ </td><td>2.68</td></tr><tr><td>Rolling-Forcing [16]</td><td>0.620</td><td>0.953</td><td>0.742</td><td> $\underline{0.688}$ </td><td>0.982</td><td>0.960</td><td>0.824</td><td>0.580</td><td>0.958</td><td>0.380</td><td>0.670</td><td>0.988</td><td>0.977</td><td>0.759</td><td>2.55</td><td>2.68</td><td>2.60</td><td>2.61</td></tr><tr><td>Deep Forcing [25]</td><td>0.621</td><td>0.953</td><td>0.713</td><td>0.660</td><td>0.979</td><td>0.961</td><td>0.815</td><td>0.597</td><td>0.957</td><td>0.402</td><td>0.690</td><td>0.987</td><td>0.979</td><td>0.769</td><td>2.60</td><td>2.76</td><td>2.68</td><td>2.68</td></tr><tr><td>Reward Forcing [17]</td><td> $\underline{0.644}$ </td><td>0.956</td><td> $\underline{0.954}$ </td><td>0.683</td><td>0.981</td><td>0.957</td><td> $\underline{0.863}$ </td><td>0.585</td><td>0.952</td><td>0.676</td><td>0.673</td><td>0.985</td><td>0.974</td><td> $\underline{0.808}$ </td><td> $\underline{2.91}$ </td><td> $\underline{2.99}$ </td><td>2.83</td><td> $\underline{2.91}$ </td></tr><tr><td>LongLive [22]</td><td> $\underline{0.654}$ </td><td> $\underline{0.959}$ </td><td>0.649</td><td>0.678</td><td> $\underline{0.983}$ </td><td> $\underline{0.967}$ </td><td>0.816</td><td> $\underline{0.606}$ </td><td>0.961</td><td>0.433</td><td>0.664</td><td> $\underline{0.991}$ </td><td> $\underline{0.982}$ </td><td>0.773</td><td>2.56</td><td>2.70</td><td>2.58</td><td>2.61</td></tr><tr><td>Infinity-RoPE [24]</td><td>0.640</td><td>0.958</td><td>0.847</td><td>0.669</td><td>0.982</td><td>0.966</td><td>0.844</td><td> $\underline{0.607}$ </td><td>0.959</td><td>0.647</td><td>0.638</td><td>0.988</td><td>0.979</td><td>0.803</td><td>2.46</td><td>2.44</td><td>2.41</td><td>2.43</td></tr><tr><td>LongSANA [3]</td><td>0.573</td><td> $\underline{0.976}$ </td><td>0.149</td><td>0.683</td><td>0.974</td><td> $\underline{0.988}$ </td><td>0.723</td><td>0.529</td><td> $\underline{0.976}$ </td><td>0.103</td><td> $\underline{0.702}$ </td><td> $\underline{0.991}$ </td><td> $\underline{0.986}$ </td><td>0.714</td><td>2.48</td><td>2.63</td><td>2.56</td><td>2.56</td></tr><tr><td>VideoMLA (Ours)</td><td>0.601</td><td>0.942</td><td> $\underline{0.981}$ </td><td>0.697</td><td> $\underline{0.986}$ </td><td>0.952</td><td> $\underline{0.859}$ </td><td>0.569</td><td> $\underline{0.963}$ </td><td> $\underline{0.958}$ </td><td> $\underline{0.715}$ </td><td> $\underline{0.993}$ </td><td>0.954</td><td> $\underline{0.859}$ </td><td>3.04</td><td>3.24</td><td>3.22</td><td>3.17</td></tr></table>

Table 2: Long-horizon performance and user preference comparison. Results across 30s and 60s video generation, plus user study scores. AQ: Aesthetic Quality, BC: Background Consistency, DD: Dynamic Degree, IQ: Imaging Quality, MS: Motion Smoothness, SC: Subject Consistency. User study metrics are PA: Prompt Adherence, TC: Temporal Consistency, and DC: Dynamic Consistency. Bold: best; underline: second best.

Reward Forcing [17], Deep Forcing [25], and LongLive [22] and Infinity-RoPE[24]. We also compare with architectural efficiency method LongSANA [3].

# 4.2 Main Results

Qualitative Results. Figure 4 shows that VideoMLA preserves subject identity, scene structure, and visual fidelity over 30-second rollouts despite replacing the dense per-head KV cache with a compact latent cache. Finally, Figure 5 shows that VideoMLA generates results comparable to representative streaming causal video baselines while requiring faster inference and substantially lower memory. These qualitative results indicate that VideoMLA improves the efficiency–memory trade-off without the pronounced fidelity, dynamism, or long-horizon stability losses observed in more aggressive compression-based alternatives.

Quantitative Results. Table 2 reports long-horizon VBench results at 30s and 60s. VideoMLA achieves the best dynamic degree at both horizons, with 0.981 at 30s and 0.958 at 60s, indicating that latent KV compression does not suppress motion or lead to static generation. It also obtains the best imaging quality and motion smoothness, and reaches the highest 60s overall score of 0.859, substantially outperforming prior streaming baselines such as Reward Forcing, Infinity-RoPE, LongLive, and LongSANA. At 30s, VideoMLA is also competitive with the strongest baseline, achieving the second-best overall score while using a much smaller KV cache memory.

1 Minute Horizon   
![](images/fd0cb32a284ecd6bf515dce776d929ce126b0adddf3ff77593618f13055fa63b.jpg)

<details>
<summary>natural_image</summary>

Six-panel photo collage showing a skier in red gear on a snowy mountain peak, with no visible text or symbols.
</details>

Figure 6: Long-horizon generation quality. Frames sampled across a one-minute rollout of the same prompt. (Bottom) VideoMLA sustains visual fidelity with diverse, evolving motion, while (Top) LongSANA produces near-static content that degrades over time. VideoMLA yields higher visual fidelity and more diverse motion while achieving higher generation throughput and lower latency than LongSANA, and reduces KV cache size by 92.7% relative to the Self-Forcing baseline. 

<table><tr><td>Model</td><td>#Params</td><td>Resolution</td><td>Throughput↑</td><td>Latency↓</td><td>CLIP-T↑</td><td>CLIP-F↑</td><td>HPSv3↑</td></tr><tr><td colspan="8">Frame-wise autoregressive models</td></tr><tr><td>NOVA [5]</td><td>0.6B</td><td>768×480</td><td>2.26</td><td>14.63</td><td>0.2764</td><td>0.9673</td><td>2.95</td></tr><tr><td>Pyramid Flow [11]</td><td>2B</td><td>640×384</td><td>1.39</td><td>87.32</td><td>0.2888</td><td>0.9795</td><td>8.02</td></tr><tr><td colspan="8">Chunk-wise autoregressive models</td></tr><tr><td>Self-Forcing [9]</td><td>1.3B</td><td>832×480</td><td>18.06</td><td>4.19</td><td>0.3036</td><td>0.9689</td><td>9.86</td></tr><tr><td>LongSANA [3]</td><td>2B</td><td>832×480</td><td>19.35</td><td>4.48</td><td>0.2978</td><td>0.9887</td><td>7.54</td></tr><tr><td>VideoMLA (Ours)</td><td>1.3B</td><td>832×480</td><td>23.96</td><td>3.38</td><td>0.3278</td><td>0.9686</td><td>9.74</td></tr></table>

Table 3: Text-to-Video quantitative comparison on VBench. Models have similar parameter sizes and resolutions. Throughput ↑ (FPS) and latency ↓ (s) measured with batch size 1 on B200. Higher is better for CLIP-T, CLIP-F, and HPSv3 scores ↑. Bold: best; underline: second best.

Efficiency Results. Table 3 shows that VideoMLA achieves the highest throughput and lowest latency among chunk-wise autoregressive models, while also obtaining the best CLIP-T score. Although LongSANA has a slightly higher CLIP-F score, this is partly due to its more static generations, which preserve frame-level similarity but reduce motion dynamics. Consistently, VideoMLA obtains a higher HPSv3 score and, as shown in Figure 6, produces sharper, more dynamic, and more temporally stable long-rollout videos than LongSANA.

# 4.3 Ablations

Batch Scaling Under Fixed Memory. Fig. 7 shows that VideoMLA translates cache compression into practical serving headroom on a single B200. Dense MHA reaches the memory limit at B = 28, whereas MLA shifts the OOM cliff far to the right; with $d _ { c } = 6 4$ , it remains within budget even at $B = 3 2 0$ . The per-request memory slope drops from 6.26 GB/batch for MHA to 0.57–1.43 GB/batch for MLA, a 77–91% reduction across dc ∈ {64, 128, 192, 256, 512}. Consequently, MLA supports 4.6× to at least 11.4× larger non-OOM batches under the same memory cap, with our default $d _ { c } = 1 9 2$ giving 8.0× batch headroom.

# 5 Why MLA Works in Video Diffusion: Rank Budget vs. Spectral Structure

MLA is often motivated by the assumption that the pretrained key/value maps are approximately low-rank. We test whether this explanation holds for video diffusion by analyzing the joint dense operator $[ W _ { K } ; W _ { V } ]$ in Wan2.1-T2V-1.3B. Fig. 1 shows that this operator is not low-rank: at the default budget $d _ { c } = 1 9 2$ , the median layer preserves only 45.8% of the spectral energy, and the 99%-energy effective rank exceeds 1300 in every layer. Thus, a direct rank-dc spectral approximation would discard most of the dense key/value energy, even though VideoMLA retains generation quality at this cache size.

![](images/1c8a03aca1370a99aa7081bd595993dab6540841c512c11497f5752dd4996ec5.jpg)

<details>
<summary>line</summary>

| Batch size | MHA / SF | MLA d=64 | MLA d=128 | MLA d=192 | MLA d=256 | MLA d=512 |
| ---------- | -------- | -------- | --------- | --------- | --------- | --------- |
| 1          | 10       | 5        | 3         | 2         | 1         | 0.5       |
| 2          | 20       | 10       | 6         | 4         | 2         | 1         |
| 4          | 40       | 20       | 10        | 8         | 4         | 2         |
| 8          | 80       | 40       | 20        | 15        | 8         | 4         |
| 16         | 160      | 80       | 40        | 30        | 15        | 10        |
| 32         | 320      | 160      | 80        | 60        | 30        | 20        |
| 64         | 720      | 320      | 160       | 120       | 60        | 40        |
| 128        | 1880     | 720      | 320       | 240       | 120       | 80        |
| 256        | 3360     | 1880     | 640       | 480       | 240       | 160       |
</details>

![](images/5c6a1633450a8d26c91e1116c8477cb9c49241d6b244f9771ff9bb614e73b2e6.jpg)

<details>
<summary>bar</summary>

| MLAdc | Per-batch memory slope (GB) |
| ----- | --------------------------- |
| MHA   | 6.26                        |
| 64    | 0.57 (-91%)                 |
| 128   | 0.69 (-89%)                 |
| 192   | 0.82 (-87%)                 |
| 256   | 0.94 (-85%)                 |
| 512   | 1.43 (-77%)                 |
</details>

![](images/1e290d97507f5419aa2d5e23a6fe782ecf3d5ed8b5d897bb15adc4e60288167d.jpg)

<details>
<summary>bar</summary>

(c) Practical serving headroom
| MLA dc (MHA baseline at left) | Maximum non-OOM batch |
| :--- | :--- |
| MHA | 27.8 |
| 64 | 19.0 |
| 128 | 19.5 |
| 192 | 19.3 |
| 256 | 19.1 |
| 512 | 19.1 |
white labels: aggregate fps ≥ 11.4×
white labels: aggregate fps 4.6×
</details>

Figure 7: VideoMLA increases serving headroom under a fixed B200 memory budget. Compared with dense MHA, MLA greatly reduces per-batch memory growth and shifts the OOM limit to much larger batch sizes; the default $d _ { c } = 1 9 2$ gives 8.0× non-OOM batch headroom.

![](images/f77ddd81261aa44ad7f53ad66d7e724ee2bad54559174d9d2e76f8b3003dfd08.jpg)

<details>
<summary>line</summary>

| Training checkpoint step (k) | SVD init | Random init |
| ---------------------------- | -------- | ----------- |
| 0                            | 189      | 187         |
| 2                            | 189      | 187         |
| 4                            | 189      | 187         |
| 6                            | 189      | 187         |
| 8                            | 189      | 187         |
| 9.5                          | 189      | 187         |
</details>

![](images/6c291f2aeae3646de2b3d651f0be064af6ec01c09ab247c0e95ba325eda7e643.jpg)

<details>
<summary>line</summary>

| Training checkpoint step (k) | Median r0.0/dc (%) |
| ---------------------------- | ------------------ |
| 0                            | 98.4               |
| 9.5                          | 97.4               |
</details>

![](images/cd85e583a7937657cb47bcba04a0a9dcff2123f058091e9ffb18ca0e06222c99.jpg)

<details>
<summary>line</summary>

| Training checkpoint step (k) | SVD init | Random init |
| ---------------------------- | -------- | ----------- |
| 0                            | 0.21     | 0.40        |
| 9.5                          | 0.21     | 0.40        |
</details>

Figure 8: Rank-budget saturation during training. At $d _ { c } = 1 9 2 \AA$ , both SVD and random initialization occupy nearly the full latent rank budget from initialization, with stable effective rank and spectral tail throughout training.

This mismatch suggests that MLA should not be interpreted as recovering a hidden low-rank structure in the pretrained attention weights. Instead, MLA changes the optimization problem: the composed key/value operator

$$
M = [ W _ {\uparrow} ^ {K} W _ {\downarrow} ^ {K V}; W _ {\uparrow} ^ {V} W _ {\downarrow} ^ {K V} ]
$$

is constrained by construction to have rank at most $d _ { c }$ . Fig. 2 confirms that the learned composed operator uses this architectural budget almost fully across latent sizes. For $d _ { c } \in \{ 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \}$ , the normalized spectra share a common shape truncated at $d _ { c } ,$ and the layer-wise 99%-energy rank remains close to 0.98dc throughout the network. The effective rank is therefore set by the MLA bottleneck rather than by the spectrum of the original dense operator.

We further investigate whether this is an artifact of SVD initialization. Fig. 8 compares SVD and random initialization at $d _ { c } ~ = ~ 1 9 2$ during training. Both nearly saturate the rank budget from initialization, and training preserves the effective rank and spectral tail. Thus, training does not discover a lower-rank solution or collapse the spectrum; it adapts within the imposed budget.

# 6 Limitations and Broader Impact

VideoMLA reduces per-token KV cache, but the latent budget cannot shrink arbitrarily. Small budgets such as $d _ { c } = 6 4$ improve memory headroom but lose fine-grained details and degrade quality, making $d _ { c }$ a quality–efficiency trade-off. Our experiments focus on Wan2.1-T2V-1.3B and minute-scale generation; larger backbones, higher resolutions, longer horizons, and prompt switching remain future work. More efficient long-horizon generation can reduce deployment cost and broaden access to creative tools, simulation, education, and assistive media production.

# 7 Conclusion

We presented VideoMLA, the first MLA-style latent KV cache for autoregressive video diffusion. By replacing dense per-head keys and values with a shared low-rank content latent and a head-shared decoupled 3D-RoPE positional key, VideoMLA reduces per-token KV cache memory by 92.7% while preserving compatibility with standard chunk-causal generation. Our analysis shows that this success does not arise from an intrinsically low-rank pretrained key-value operator; instead, the MLA bottleneck defines a rank budget that the model uses nearly fully and adapts within during training. Empirically, VideoMLA preserves visual quality and motion at long horizons, achieves the best one-minute overall score among evaluated methods, and improves throughput with substantially lower cache memory. These results identify the per-token KV layout as an effective and complementary axis for scaling efficient long-horizon video diffusion.

# Acknowledgements

Pinar Yanardag is supported by the National Science Foundation under Grant No. 2543524.

# References

[1] Ainslie, J., Lee-Thorp, J., De Jong, M., Zemlyanskiy, Y., Lebrón, F., Sanghai, S.: Gqa: Training generalized multi-query transformer models from multi-head checkpoints. In: Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. pp. 4895–4901 (2023)   
[2] Bai, X., He, G., Li, Z., Shechtman, E., Huang, X., Wu, Z.: Causality in video diffusers is separable from denoising. arXiv preprint arXiv:2602.10095 (2026)   
[3] Chen, J., Zhao, Y., Yu, J., Chu, R., Chen, J., Yang, S., Wang, X., Pan, Y., Zhou, D., Ling, H., et al.: Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695 (2025)   
[4] Cui, J., Wu, J., Li, M., Yang, T., Li, X., Wang, R., Bai, A., Ban, Y., Hsieh, C.J.: Self-forcing++: Towards minute-scale high-quality video generation. arXiv preprint arXiv:2510.02283 (2025)   
[5] Deng, H., Pan, T., Diao, H., Luo, Z., Cui, Y., Lu, H., Shan, S., Qi, Y., Wang, X.: Autoregressive video generation without vector quantization. arXiv preprint arXiv:2412.14169 (2024)   
[6] Deng, K., Woodland, P.C.: Multi-head temporal latent attention. arXiv preprint arXiv:2505.13544 (2025)   
[7] Gao, J., Chen, Z., Liu, X., Feng, J., Si, C., Fu, Y., Qiao, Y., Liu, Z.: Longvie: Multimodal-guided controllable ultra-long video generation. arXiv preprint arXiv:2508.03694 (2025)   
[8] Henschel, R., Khachatryan, L., Poghosyan, H., Hayrapetyan, D., Tadevosyan, V., Wang, Z., Navasardyan, S., Shi, H.: Streamingt2v: Consistent, dynamic, and extendable long video generation from text. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2568–2577 (2025)   
[9] Huang, X., Li, Z., He, G., Zhou, M., Shechtman, E.: Self forcing: Bridging the train-test gap in autoregressive video diffusion. arXiv preprint arXiv:2506.08009 (2025)   
[10] Ji, T., Guo, B., Wu, Y., Guo, Q., Shen, L., Chen, Z., Qiu, X., Zhang, Q., Gui, T.: Towards economical inference: Enabling deepseek’s multi-head latent attention in any transformerbased llms. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 33313–33328 (2025)   
[11] Jin, Y., Sun, Z., Li, N., Xu, K., Jiang, H., Zhuang, N., Huang, Q., Song, Y., Mu, Y., Lin, Z.: Pyramidal flow matching for efficient video generative modeling. arXiv preprint arXiv:2410.05954 (2024)   
[12] Kim, Y., Hu, Q., Kuo, C.C.J., Beerel, P.A.: Memrope: Training-free infinite video generation via evolving memory tokens. arXiv preprint arXiv:2603.12513 (2026)

[13] Li, H., Liu, S., Lin, Z., Chandraker, M.: Rolling sink: Bridging limited-horizon training and open-ended testing in autoregressive video diffusion. arXiv preprint arXiv:2602.07775 (2026)   
[14] Liu, A., Feng, B., Wang, B., Wang, B., Liu, B., Zhao, C., Dengr, C., Ruan, C., Dai, D., Guo, D., et al.: Deepseek-v2: A strong, economical, and efficient mixture-of-experts language model. arXiv preprint arXiv:2405.04434 (2024)   
[15] Liu, A., Feng, B., Xue, B., Wang, B., Wu, B., Lu, C., Zhao, C., Deng, C., Zhang, C., Ruan, C., et al.: Deepseek-v3 technical report. arXiv preprint arXiv:2412.19437 (2024)   
[16] Liu, K., Hu, W., Xu, J., Shan, Y., Lu, S.: Rolling forcing: Autoregressive long video diffusion in real time. arXiv preprint arXiv:2509.25161 (2025)   
[17] Lu, Y., Zeng, Y., Li, H., Ouyang, H., Wang, Q., Cheng, K.L., Zhu, J., Cao, H., Zhang, Z., Zhu, X., et al.: Reward forcing: Efficient streaming video generation with rewarded distribution matching distillation. arXiv preprint arXiv:2512.04678 (2025)   
[18] Meng, F., Tang, P., Tang, X., Yao, Z., Sun, X., Zhang, M.: Transmla: Multi-head latent attention is all you need. arXiv preprint arXiv:2502.07864 (2025)   
[19] Nan, K., Xie, R., Zhou, P., Fan, T., Yang, Z., Chen, Z., Li, X., Yang, J., Tai, Y.: Openvid-1m: A large-scale high-quality dataset for text-to-video generation. arXiv preprint arXiv:2407.02371 (2024)   
[20] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I.: Attention is all you need. Advances in neural information processing systems 30 (2017)   
[21] Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)   
[22] Yang, S., Huang, W., Chu, R., Xiao, Y., Zhao, Y., Wang, X., Li, M., Xie, E., Chen, Y., Lu, Y., et al.: Longlive: Real-time interactive long video generation. arXiv preprint arXiv:2509.22622 (2025)   
[23] Yang, Y., Zhang, T., Huang, W., Chen, J., Wu, B., He, X., Cai, D., Li, B., Jiang, P.T.: Anchor forcing: Anchor memory and tri-region rope for interactive streaming video diffusion. arXiv preprint arXiv:2603.13405 (2026)   
[24] Yesiltepe, H., Meral, T.H.S., Akan, A.K., Oktay, K., Yanardag, P.: Infinity-rope: Actioncontrollable infinite video generation emerges from autoregressive self-rollout. arXiv preprint arXiv:2511.20649 (2025)   
[25] Yi, J., Jang, W., Cho, P.H., Nam, J., Yoon, H., Kim, S.: Deep forcing: Training-free long video generation with deep sink and participative compression. arXiv preprint arXiv:2512.05081 (2025)   
[26] Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., Freeman, B.: Improved distribution matching distillation for fast image synthesis. Advances in neural information processing systems 37, 47455–47487 (2024)   
[27] Yin, T., Zhang, Q., Zhang, R., Freeman, W.T., Durand, F., Shechtman, E., Huang, X.: From slow bidirectional to fast autoregressive video diffusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 22963–22974 (2025)   
[28] Yu, Y., Wu, X., Hu, X., Hu, T., Sun, Y., Lyu, X., Wang, B., Ma, L., Ma, Y., Wang, Z., et al.: Videossm: Autoregressive long video generation with hybrid state-space memory. arXiv preprint arXiv:2512.04519 (2025)   
[29] Zhang, L., Agrawala, M.: Packing input frame context in next-frame prediction models for video generation. arXiv e-prints pp. arXiv–2504 (2025)   
[30] Zhao, Z., Lu, Y., Liu, Z., Song, J., Deng, J., Patras, I.: Relax forcing: Relaxed kv-memory for consistent long video generation. arXiv preprint arXiv:2603.21366 (2026)

[31] Zhu, H., Zhao, M., He, G., Su, H., Li, C., Zhu, J.: Causal forcing: Autoregressive diffusion distillation done right for high-quality real-time interactive video generation. arXiv preprint arXiv:2602.02214 (2026)

# Table of Contents

A Videos and Website 1

B Details on User Study 1

C Background 1

C.1 Wan2.1-T2V-1.3B Backbone . . .

D Implementation Details 3

D.1 Backbone and Tokenization . . 3

D.2 VideoMLA Block . . . 3

D.3 NoPE/RoPE Split and 3D RoPE 4

D.4 Chunk-Causal Sliding-Window Attention . . . 4

D.5 Long-Horizon RoPE Re-indexing . . . 5

D.6 Training Pipeline . . . 5

E Inference-Time Reparameterization 5

F Additional Ablations 7

# A Videos and Website

To facilitate comprehensive evaluation and improve result accessibility, we provide video results covering qualitative examples, ablation studies, comparisons, and limitations in the https:// videomla.github.io.

# B Details on User Study

We conduct a user study to evaluate perceptual quality of one-minute generations. We compare nine models and ask 50 participants to rate each video using the interface shown in Fig. 9. For each generated video, participants answer three questions: Prompt Adherence, measuring how well the video follows the prompt; Temporal Consistency, measuring whether the video remains coherent from start to end; and Dynamic Consistency, measuring whether the video contains plausible and sustained motion. Each question is rated on a five-point Likert scale, from 1) Very Bad to 5) Very Good.

# C Background

# C.1 Wan2.1-T2V-1.3B Backbone

Our experiments use Wan2.1-T2V-1.3B as the base video diffusion backbone. Wan2.1-T2V-1.3B is a latent video diffusion transformer operating over spatiotemporal latent tokens rather than RGB pixels. The video is encoded by a 3D VAE that compresses the temporal dimension by 4× and each spatial dimension by $8 \times .$ , so an input video $V \in \dot { \mathbb { R } } ^ { F \times H \times W \times 3 }$ is mapped to a latent tensor with temporal length $1 + { \bf \bar { \Delta } } [ ( F - 1 ) / 4 ]$ and spatial resolution $H / 8 \times W / 8$ . The denoising model follows the rectified-flow formulation, where a clean latent $x _ { 0 }$ and Gaussian noise ϵ are linearly interpolated as

$$
x _ {t} = (1 - t) x _ {0} + t \epsilon , \qquad t \in [ 0, 1 ],
$$

and the reverse process is parameterized by a neural velocity field and solved with Euler integration at inference time.

Aclose-upviewofasspheecontainingatanqilngarden.side,smallEastedwarfwithweatheedsinandsereneexpiis rakingthesand,meticulouslyeatigtrcatepateswithambooake.Hismovementsaredeliberatendmditativeancingthe peacefulatmopeoftheenckgoudisuredeelinlyntsfreeencsdingtterei spereitsesib hisfocusedand contemplative pose.

![](images/3b2a8dd944593b1b2592d3e0daf509f775a1b57427f1c4e45ccb729688cbaae2.jpg)

<details>
<summary>natural_image</summary>

Child playing with a broom inside a glass sphere, surrounded by greenery (no text or symbols)
</details>

<table><tr><td></td><td>1) Very Bad</td><td>2) Bad</td><td>3) looks OK</td><td>4) Good</td><td>5) Very Good</td></tr><tr><td>Prompt Adherence. Overall, how good is this video given the prompt?</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>Temporal Consistency. How consistent does the video stay from start to end?</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr><tr><td>Dynamic Consistency. How dynamic is the generated video?</td><td>○</td><td>○</td><td>○</td><td>○</td><td>○</td></tr></table>

Figure 9: User Study Interface. User Study Interface for Long Video Generation

Wan2.1-T2V-1.3B uses a diffusion transformer with multi-head self-attention over video latent tokens. In our implementation, the backbone contains 30 transformer blocks, hidden dimension d = 1536, $n _ { h } = 1 2$ attention heads, and per-head dimension $d _ { h } = 1 2 8$ . In the dense baseline, each cached token stores both keys and values for all heads, giving a per-token, per-layer KV cache size of

$$
2 n _ {h} d _ {h} = 2 \times 1 2 \times 1 2 8 = 3 0 7 2
$$

scalars. With a 21-latent-frame cache, 1,560 tokens per latent frame, and 30 cached transformer layers, this corresponds to 3.02B cached scalars, or approximately 6.0GB in bf16/fp16. This dense per-head KV layout is the main memory target of VideoMLA.

The backbone uses 3D rotary position embeddings (3D-RoPE) to encode temporal and spatial token coordinates before self-attention. For latent features $\boldsymbol { x } \in \mathbb { R } ^ { \tilde { B } \times S \times C }$ with $S \overset { \cdot } { = } F H W$ , the channel dimension is partitioned across temporal, height, and width axes, and RoPE is applied separately to the corresponding coordinate subspaces before concatenation. In Wan, each RoPE dimension has a fixed maximum sequence length of 1024; although RoPE remains mathematically defined beyond this range, generation outside the positional regime observed during training can degrade attention quality.

For autoregressive long-video generation, Wan2.1-T2V-1.3B is commonly used after causal distillation or self-rollout training. The model generates latent frame chunks sequentially and conditions each chunk on a rolling KV cache of previous chunks. This cache enables efficient streaming generation because previous key and value states are reused rather than recomputed. However, in the dense Wan attention layout, the cache still stores full per-head keys and values for every retained token and every cached layer. VideoMLA keeps the Wan backbone and causal rollout setting intact, but replaces this dense per-token KV state with a shared latent content cache and a decoupled head-shared 3D-RoPE key.

Table 4: Training and model hyperparameters. Unless otherwise stated, all experiments use the default VideoMLA setting. 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Backbone</td><td>Wan2.1-T2V-1.3B</td></tr><tr><td>Transformer blocks</td><td>30</td></tr><tr><td>Hidden dimension  $d$ </td><td>1536</td></tr><tr><td>Number of heads  $n_h$ </td><td>12</td></tr><tr><td>Per-head dimension  $d_h$ </td><td>128</td></tr><tr><td>KV latent dimension  $d_c$ </td><td>192</td></tr><tr><td>Query latent dimension  $d_q$ </td><td>768</td></tr><tr><td>NoPE channels  $d_h^{\text{nope}}$ </td><td>96</td></tr><tr><td>RoPE channels  $d_h^{\text{rope}}$ </td><td>32</td></tr><tr><td>3D-RoPE complex pairs  $(t, h, w)$ </td><td> $(6, 5, 5)$ </td></tr><tr><td>Per-token cache size</td><td>224 scalars</td></tr><tr><td>KV cache reduction</td><td>92.7%</td></tr><tr><td>Training precision</td><td>bf16</td></tr><tr><td>Training GPUs</td><td> $8 \times \text{B200}$ </td></tr><tr><td>Total batch size</td><td>128</td></tr><tr><td>Teacher Forcing LR</td><td> $5 \times 10^{-6}$ </td></tr><tr><td>Consistency Distillation LR</td><td> $2 \times 10^{-6}$ </td></tr><tr><td>DMD LR</td><td> $2 \times 10^{-6}$ </td></tr><tr><td>Training pipeline</td><td> $\text{TF} \rightarrow \text{CD} \rightarrow \text{DMD}$ </td></tr><tr><td>Inference steps</td><td>4</td></tr></table>

# D Implementation Details

# D.1 Backbone and Tokenization

VideoMLA is implemented on top of Wan2.1-T2V-1.3B. The backbone contains L = 30 transformer blocks, hidden dimension $d = 1 5 3 6 , n _ { h } = 1 2$ attention heads, and per-head dimension $d _ { h } = 1 2 8$ so that $d = n _ { h } d _ { h }$ . The feed-forward hidden dimension is 8960. We keep the Wan text-conditioning branch and all non-attention modules unchanged, and replace only the temporal self-attention layers with the VideoMLA block described in Section 3.

We train on 5-second clips at $4 8 0 \times 8 3 2$ resolution and 16 fps. The Wan VAE encodes each clip into a latent tensor with 16 channels, 21 latent frames, and spatial size $6 0 \times 1 0 4$ . A 3D patch embedding with patch size (1, 2, 2) maps each latent frame into $3 0 \times 5 2 = 1 5 6 0$ visual tokens. Thus, each 5-second clip contains $2 1 \times 1 5 6 0 = 3 2 7 6 0$ self-attention tokens. Autoregressive generation is performed in chunks of 3 latent frames.

# D.2 VideoMLA Block

Each temporal self-attention layer follows the notation of Section 3. Given an attention input $x _ { t } \in \mathbb { R } ^ { d }$ , VideoMLA first forms a shared content cache latent

$$
c _ {t} ^ {K V} = W _ {\downarrow} ^ {K V} x _ {t} \in \mathbb {R} ^ {d _ {c}},
$$

and a query latent

$$
c _ {t} ^ {Q} = W _ {\downarrow} ^ {Q} x _ {t} \in \mathbb {R} ^ {d _ {q}}.
$$

In the default model, we use

$$
d _ {c} = 1 9 2, \qquad d _ {q} = 7 6 8, \qquad d _ {h} ^ {\mathrm{nope}} = 9 6, \qquad d _ {h} ^ {\mathrm{rope}} = 3 2,
$$

with $d _ { h } ^ { \mathrm { n o p e } } + d _ { h } ^ { \mathrm { r o p e } } = d _ { h } = 1 2 8$ . The down-projection shapes are

$$
W _ {\downarrow} ^ {K V} \in \mathbb {R} ^ {1 9 2 \times 1 5 3 6}, \qquad W _ {\downarrow} ^ {Q} \in \mathbb {R} ^ {7 6 8 \times 1 5 3 6}.
$$

Both $c _ { t } ^ { K V }$ and $c _ { t } ^ { Q }$ are normalized by RMSNorm before the corresponding up-projections.

The content key and value for head h are reconstructed from the shared cache latent:

$$
k _ {t, h} ^ {\mathrm{nope}} = W _ {\uparrow , h} ^ {K} c _ {t} ^ {K V}, \qquad v _ {t, h} = W _ {\uparrow , h} ^ {V} c _ {t} ^ {K V}.
$$

Aggregating all heads, the projection shapes are

$$
W _ {\uparrow} ^ {K} \in \mathbb {R} ^ {(n _ {h} d _ {h} ^ {\mathrm{nope}}) \times d _ {c}} = \mathbb {R} ^ {1 1 5 2 \times 1 9 2},
$$

$$
W _ {\uparrow} ^ {V} \in \mathbb {R} ^ {(n _ {h} d _ {h}) \times d _ {c}} = \mathbb {R} ^ {1 5 3 6 \times 1 9 2}.
$$

The NoPE query is reconstructed analogously:

$$
q _ {t, h} ^ {\text { nope }} = W _ {\uparrow , h} ^ {Q} c _ {t} ^ {Q}, \quad W _ {\uparrow} ^ {Q} \in \mathbb {R} ^ {(n _ {h} d _ {h} ^ {\text { nope }}) \times d _ {q}} = \mathbb {R} ^ {1 1 5 2 \times 7 6 8}.
$$

The RoPE branch is decoupled from the content cache. For each token, VideoMLA computes a single head-shared positional key

$$
k _ {t} ^ {R} = W _ {R} ^ {K} x _ {t} \in \mathbb {R} ^ {d _ {h} ^ {\mathrm{rope}}}, \qquad W _ {R} ^ {K} \in \mathbb {R} ^ {3 2 \times 1 5 3 6},
$$

and per-head positional queries

$$
q _ {t, h} ^ {R} = W _ {R, h} ^ {Q} c _ {t} ^ {Q}, \qquad W _ {R} ^ {Q} \in \mathbb {R} ^ {(n _ {h} d _ {h} ^ {\mathrm{rope}}) \times d _ {q}} = \mathbb {R} ^ {3 8 4 \times 7 6 8}.
$$

The output projection is the original attention output projection

$$
W _ {O} \in \mathbb {R} ^ {d \times (n _ {h} d _ {h})} = \mathbb {R} ^ {1 5 3 6 \times 1 5 3 6}.
$$

Therefore, each cached token stores only

$$
\left(c _ {t} ^ {K V}, k _ {t} ^ {R}\right) \in \mathbb {R} ^ {d _ {c} + d _ {h} ^ {\mathrm{rope}}},
$$

rather than dense per-head keys and values. With the default setting, this is $1 9 2 + 3 2 = 2 2 4$ scalars per token per layer, compared with $2 n _ { h } d _ { h } = 2 \cdot 1 2 \cdot 1 2 8 = 3 0 7 2$ scalars for dense MHA, corresponding to a 13.7× cache reduction.

# D.3 NoPE/RoPE Split and 3D RoPE

Each head is split as

$$
d _ {h} = d _ {h} ^ {\mathrm{nope}} + d _ {h} ^ {\mathrm{rope}},
$$

where the NoPE subspace is used for content matching and the RoPE subspace is used for positionaware matching. The per-head query and key are

$$
q _ {t, h} = [ q _ {t, h} ^ {\mathrm{nope}}; q _ {t, h} ^ {\mathrm{rope}} ], \qquad k _ {t, h} = [ k _ {t, h} ^ {\mathrm{nope}}; k _ {t} ^ {\mathrm{rope}} ],
$$

where

$$
q _ {t, h} ^ {\text { rope }} = \mathrm{RoPE} _ {3 D} (q _ {t, h} ^ {R}), \qquad k _ {t} ^ {\text { rope }} = \mathrm{RoPE} _ {3 D} (k _ {t} ^ {R}).
$$

The positional key is shared across heads, while the NoPE keys, NoPE queries, and values remain head-specific after up-projection.

For the default split $d _ { h } ^ { \mathrm { r o p e } } = 3 2$ , the RoPE subspace contains $d _ { h } ^ { \mathrm { r o p e } } / 2 = 1 6$ complex frequency pairs. Following the Wan 3D-RoPE factorization, these pairs are allocated across temporal, height, and width axes as

$$
(6, 5, 5),
$$

using the highest-frequency bands from the corresponding axis groups.

# D.4 Chunk-Causal Sliding-Window Attention

VideoMLA preserves the chunk-causal attention pattern used by the autoregressive Wan backbone. Tokens within the same 3-latent-frame chunk can attend to one another, while tokens in a later chunk cannot be attended to by earlier chunks. For long-horizon generation, attention is restricted to a fixed cache consisting of one sink latent frame and the most recent six latent frames. Since each latent frame contains 1560 tokens, the sink occupies 1560 cached token slots and the local window occupies $6 \times 1 5 6 0$ token slots.

During training, the same chunk-causal and sliding-window structure is enforced with block-sparse attention masks. During inference, the cache stores

$$
c ^ {K V} \in \mathbb {R} ^ {B \times T _ {\text { cache }} \times d _ {c}}, \quad k ^ {R} \in \mathbb {R} ^ {B \times T _ {\text { cache }} \times d _ {h} ^ {\text { rope }}},
$$

where $T _ { \mathrm { c a c h e } }$ is the number of cached tokens in the sink-plus-window context. When the cache is full, tokens outside the sink are evicted in FIFO order. The attention computation reads the active cache, reconstructs the content keys and values through $W _ { \uparrow } ^ { K }$ and $W _ { \uparrow } ^ { V }$ , applies 3D-RoPE to the active positional keys, and evaluates the standard per-head attention scores from Eq. (7).

# D.5 Long-Horizon RoPE Re-indexing

For rollouts beyond the 21 latent frames seen during 5-second training, cached positional keys are stored before RoPE is applied. When an attention window is assembled, the active cached keys are assigned local temporal coordinates inside the current sink-plus-window context and are then rotated by $\mathrm { { \bar { R o } P E } _ { 3 D } }$ following [24]. The current query chunk is rotated in the same local coordinate system. Thus, both queries and cached keys use a bounded, window-relative positional frame even after cache eviction.

This re-indexing keeps the RoPE phase within the short-horizon regime observed by the backbone during training, while allowing the generated video to extend beyond the original 21-latent-frame clip length. All reported long-video rollouts use one sink latent frame, a six-latent-frame local window, and the 4-step student sampler.

# D.6 Training Pipeline

We train the same VideoMLA architecture in three stages on 8 NVIDIA B200 GPUs, using FSDP full sharding, bf16 mixed precision, AdamW with $\beta _ { 1 } = 0$ and $\beta _ { 2 } = 0 . 9 9 9$ , and the rectified-flow denoising objective.

Stage 1: Teacher Forcing. We initialize the MLA projections from an SVD-style decomposition of the pretrained Wan dense attention matrices at the target configuration

$$
(d _ {c}, d _ {q}, d _ {h} ^ {\mathrm{nope}}, d _ {h} ^ {\mathrm{rope}}) = (1 9 2, 7 6 8, 9 6, 3 2).
$$

The model is then trained as a chunk-causal flow-matching student with clean previous-block context from the teacher-encoded latents. We use learning rate $5 ^ { ^ { - } } \times 1 0 ^ { - 6 }$ , per-GPU batch size 1, total batch size 2, gradient checkpointing, 1000 training timesteps, and timestep shift 5.0.

Stage 2: Consistency Distillation. Starting from the Stage-1 checkpoint, we distill the model to a 4-step sampling schedule

$$
[ 1 0 0 0, 7 5 0, 5 0 0, 2 5 0 ].
$$

We use timestep shift 5.0 and classifier-free guidance scale 3.0. The generator learning rate is $2 \times 1 0 ^ { - 6 }$ , the critic learning rate $\mathrm { i s 4 \times 1 0 ^ { - 7 } }$ , and the total batch size is 2 with gradient checkpointing enabled.

Stage 3: Distribution Matching Distillation. Finally, we initialize from the Stage-2 checkpoint at iteration 2500 and fine-tune with distribution matching distillation on the same 4-step schedule. The real score is provided by the frozen teacher and the fake score is learned online. We use five critic updates per generator update, EMA weight 0.99 starting from step 1, generator learning rate $2 \times 1 0 ^ { - 6 }$ , critic learning rate $\mathrm { \dot { 4 } } \times 1 0 ^ { - 7 }$ , guidance scale 3.0, and timestep shift 5.0. The total batch size is 16, obtained with per-GPU batch size 8 on 8 GPUs and gradient accumulation of 2.

The Stage-3 checkpoint is used for all reported VideoMLA results, including the long-horizon evaluations.

# E Inference-Time Reparameterization

The training-time formulation in Section 3 is written to make the connection to standard multi-head attention explicit: each cached token stores $( c _ { j } ^ { K V } , k _ { j } ^ { R } )$ , and the per-head NoPE keys and values are reconstructed through Eq. 2 before applying the usual attention computation. This is convenient duriandand raining because it allows VideoMLA to reuse the same block-causal masks, sink-tokenntion kernels as the dense baseline. At inference, however, explicitly reconstructing would partially undo the benefit of latent caching by materializing dense per-head t $k _ { j , h } ^ { \mathrm { n o p e } }$ $v _ { j , h }$ after every cache read. We therefore use an equivalent reparameterization that keeps the cache and the attention computation in latent form.

For the score computation, the only contribution of the reconstructed NoPE key is through its inner product with the reconstructed NoPE query. Substituting Eqs. 3, 4, and 2 into the content term of

Eq. 7 gives

$$
\begin{array}{l} q _ {i, h} ^ {\text { nope }} \cdot k _ {j, h} ^ {\text { nope }} = \left(W _ {\uparrow , h} ^ {Q} c _ {i} ^ {Q}\right) ^ {\top} \left(W _ {\uparrow , h} ^ {K} c _ {j} ^ {K V}\right) \\ = \left(c _ {i} ^ {Q}\right) ^ {\top} \left(W _ {\uparrow , h} ^ {Q}\right) ^ {\top} W _ {\uparrow , h} ^ {K} c _ {j} ^ {K V} \\ = \left(c _ {i} ^ {Q}\right) ^ {\top} A _ {h} c _ {j} ^ {K V}, \tag {8} \\ \end{array}
$$

where

$$
A _ {h} = \left(W _ {\uparrow , h} ^ {Q}\right) ^ {\top} W _ {\uparrow , h} ^ {K} \in \mathbb {R} ^ {d _ {q} \times d _ {c}}. \tag {9}
$$

The matrix $A _ { h }$ depends only on learned parameters and is independent of the current sequence, cache contents, diffusion timestep, and rollout position. It can therefore be precomputed once when the model is loaded. During inference, the NoPE content score for head h is computed directly from the query latent $c _ { i } ^ { Q }$ and the cached content latent $c _ { j } ^ { K V }$ , without forming either $q _ { i , h } ^ { \mathrm { n o p e } }$ or $k _ { j , h } ^ { \mathrm { n o p e } }$ as explicit per-head vectors.

The value path admits an analogous absorption. Let $W _ { h } ^ { O }$ denote the slice of the output projection applied to the output of head h. Using Eq. 2,

$$
\begin{array}{l} W _ {h} ^ {O} v _ {j, h} = W _ {h} ^ {O} W _ {\uparrow , h} ^ {V} c _ {j} ^ {K V} \\ = B _ {h} c _ {j} ^ {K V}, \tag {10} \\ \end{array}
$$

where

$$
B _ {h} = W _ {h} ^ {O} W _ {\uparrow , h} ^ {V}. \tag {11}
$$

Thus, the value up-projection can also be folded into the output mixer. In practice, after the attention weights for head h are computed, the weighted sum can be accumulated over the cached latents $c _ { i } ^ { K V }$ and then projected by $B _ { h }$ , rather than first reconstructing all dense values $v _ { j , h }$ and then applying the output projection.

The RoPE branch is kept separate from this absorption. The cache stores the unrotated, head-shared positional key $k _ { j } ^ { R }$ from Eq. 5. When the active attention window is assembled, $k _ { j } ^ { R }$ is rotated by $\mathrm { R o P E } _ { \mathrm { 3 D } } ( \cdot )$ using the current window indexing, and the positional score term in Eq. 7 is computed as

$$
q _ {i, h} ^ {\text { rope }} \cdot k _ {j} ^ {\text { rope }}. \tag {12}
$$

This separation is important because RoPE is position-dependent and cannot be folded into a fixed parameter matrix in the same way as the NoPE content projections. Storing $k _ { j } ^ { R }$ unrotated also preserves the ability to re-index cached tokens within a sliding window, as described in Section 3.2.

After reparameterization, the inference-time cache is never expanded into dense per-head keys and values. Each cached token contributes only a content latent $c _ { j } ^ { K \dot { V } } \in \mathbb { R } ^ { d _ { c } }$ and a head-shared positional key $k _ { j } ^ { R } \in \mathbb { R } ^ { d _ { h } ^ { \mathrm { r o p e } } }$ . Therefore, the per-token cached state remains

$$
d _ {c} + d _ {h} ^ {\text { rope }}, \tag {13}
$$

instead of the dense baseline cost

$$
2 n _ {h} d _ {h}. \tag {14}
$$

For an attention window of size W , cache memory traffic is reduced from

$$
\mathcal {O} (W 2 n _ {h} d _ {h}) \tag {15}
$$

to

$$
\mathcal {O} (W \left(d _ {c} + d _ {h} ^ {\text { rope }}\right)). \tag {16}
$$

With the default configuration $d _ { c } = 1 9 2$ and $d _ { h } ^ { \mathrm { r o p e } } = 3 2$ , this corresponds to 224 cached scalars per token per layer, compared with $2 n _ { h } d _ { h } = 3 0 \tilde { 7 } 2$ scalars for dense MHA. The reparameterization therefore preserves the mathematical attention computation of the training-time formulation while ensuring that the inference-time implementation realizes the intended latent-cache memory and bandwidth savings.

(a) Latent dimension dc 

<table><tr><td> $d_c$ </td><td>Semantic↑</td><td>Quality↑</td><td>Total↑</td><td>Mem.↓</td><td>FPS↑</td></tr><tr><td>64</td><td>77.42</td><td>79.18</td><td>78.30</td><td>32.00×</td><td>26.93</td></tr><tr><td>128</td><td>82.16</td><td>84.31</td><td>83.24</td><td>19.20×</td><td>27.00</td></tr><tr><td>256</td><td>82.74</td><td>84.58</td><td>83.66</td><td>10.67×</td><td>26.93</td></tr><tr><td>512</td><td>82.41</td><td>84.39</td><td>83.40</td><td>5.65×</td><td>26.79</td></tr></table>

(b) NoPE/RoPE split

<table><tr><td> $d_h^{\text{nope}}$ </td><td> $d_h^{\text{rope}}$ </td><td>Semantic↑</td><td>Quality↑</td><td>Total↑</td></tr><tr><td>112</td><td>16</td><td>74.62</td><td>78.31</td><td>76.47</td></tr><tr><td>64</td><td>64</td><td>79.54</td><td>82.12</td><td>80.83</td></tr><tr><td>32</td><td>96</td><td>75.88</td><td>80.74</td><td>78.31</td></tr><tr><td>96</td><td>32</td><td>83.02</td><td>84.76</td><td>83.89</td></tr></table>

Table 5: Ablation studies. Left: sweep over latent KV dimension $d _ { c }$ . Right: decoupled RoPE dimension ablation with $d _ { h } ^ { \mathrm { n o p e } } + d _ { h } ^ { \mathrm { r o p e } } = 1 2 8$ . Mem.: KV cache compression ratio relative to dense per-token KV cache. FPS: throughput at batch size 1 on 1×H100 80 GB.

# F Additional Ablations

Table 5 studies two architectural choices: the latent KV dimension $d _ { c }$ and the NoPE/RoPE channel split. The latent dimension controls the main quality–efficiency trade-off. At $d _ { c } = 6 4$ , VideoMLA gives the largest cache compression and memory headroom, but the budget is too restrictive: both semantic and quality scores drop, consistent with the loss of fine-grained visual details under overly aggressive compression. Increasing to $d _ { c } = $ 128 largely recovers quality while retaining a large compression ratio. Further increasing to $d _ { c } = 2 5 6$ or 512 gives only marginal gains, but substantially reduces the memory advantage. This suggests that the useful operating regime is not the largest possible latent dimension, but the smallest budget that preserves task-relevant video features.

The NoPE/RoPE split also has a clear effect. With only 16 RoPE channels, positional capacity is too limited, leading to weak temporal and spatial anchoring. Conversely, the RoPE-heavy $\mathrm { { \bar { 3 2 } / \bar { 9 6 } } }$ split leaves too little capacity for cached content and hurts semantic fidelity. The balanced 64/64 setting improves over these extremes but remains below the content-heavy default. The best result comes from the 96/32 split, indicating that streaming video benefits from allocating most channels to the cached content path while retaining a smaller dedicated RoPE subspace for positional structure.