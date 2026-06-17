# What Should a Streaming Video Model Remember?

Haonan Ge1,2 Yiwei Wang2† Hang Wu2 Yujun Cai3

1University of California, Santa Barbara 2University of California, Merced

3The University of Queensland

†Corresponding author

gehaonan82@gmail.com

SelectStream.github.io

## Abstract

Streaming video understanding models must answer queries at any moment during an ongoing stream, using only what they have observed so far and under fixed memory and computation budgets. Existing methods address this by adding memory banks, retrieval modules, or visual token compression to preserve long-range history. However, strong recent-window baselines show that indiscriminate history injection can dilute current-scene perception, suggesting that the key challenge is not whether to use memory, but how to allocate it selectively. We formulate this as budgeted online latent evidence allocation and propose SelectStream, a selective latent-memory framework that keeps the current observation directly visible to a frozen VLM while exposing historical information only through a compact, query-conditioned evidence budget. Three coordinated mechanisms govern when to write, what to preserve, and how to retrieve: surprise-driven adaptive windowing, priority-preserving consolidation, and query-conditioned graph reasoning over a fixed-capacity latent memory graph. Retrieved evidence is calibrated and injected as latent tokens for answer generation, without replaying frames or growing the context with stream length. Experimental results show that SelectStream achieves strong online streaming performance and preserves general video understanding, reaching 82.67% on StreamingBench, 67.03% on OVO-Bench, and 74.4% average accuracy on offline video benchmarks, while outperforming strong recent-window baselines and prior streaming memory methods.

## 1 Introduction

Streaming video understanding places a distinct demand on vision-language models. In applications such as egocentric life assistants, autonomous driving, robotics, surveillance, and live video interaction, a model must answer queries at any moment during an ongoing stream, using only what has been observed so far and under fixed memory, context, and computation budgets [46, 50, 32, 48, 11, 40, 21]. The central question is not merely whether the model can understand video, but whether it can identify and retain the evidence that matters while processing a continuous stream.

Recent work has responded to this challenge by equipping streaming video models with increasingly sophisticated mechanisms for preserving history. External memory banks store visual summaries of past events, KV-cache retrieval mechanisms fetch relevant history on demand, and visual token compression modules distill long temporal context into compact representations [26, 7, 50, 51, 49, 52, 43]. The implicit assumption underlying these efforts is that richer memory leads to stronger streaming understanding.

Recent evidence complicates this picture. On standard streaming benchmarks, simple recent-window baselines can match, and in some cases surpass, systems with elaborate memory modules [29]. This suggests that incorporating more history does not automatically improve performance. One explanation is evidence dilution: when a model attends over indiscriminately stored historical context, relevant evidence may be mixed with irrelevant content, weakening current-scene perception and reducing the usefulness of retrieved history.

![](images/4c0ecd9c99e7288b9d35769a3970f2b2d308c8873ebf6bde24ea39057edcb737.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["User"] --> B["Latent Embedding Compression"]
  B --> C["02:25"]
  C --> D["02:51"]
  D --> E["03:12"]
  E --> F["Current"]
  G["Selection Time"] --> H["Query Encoding"]
  H --> I["m1 0.79"]
  I --> J["m4 0.86"]
  J --> K["m3 0.81"]
  K --> L["m2 0.74"]
  L --> M["m4 0.79"]
  M --> N["m3 0.74"]
  N --> O["m4 0.79"]
  O --> P["m3 0.74"]
  P --> Q["m4 0.79"]
  Q --> R["m3 0.74"]
  R --> S["m4 0.79"]
  S --> T["m3 0.74"]
  T --> U["m4 0.79"]
  U --> V["m3 0.74"]
  V --> W["m4 0.79"]
  W --> X["m3 0.74"]
  X --> Y["m4 0.79"]
  Y --> Z["m3 0.74"]
  Z --> AA["m4 0.79"]
  AA --> AB["m3 0.74"]
  AB --> AC["m4 0.79"]
  AC --> AD["m3 0.74"]
  AD --> AE["m4 0.79"]
  AE --> AF["m3 0.74"]
  AF --> AG["m4 0.79"]
  AG --> AH["m3 0.74"]
  AH --> AI["m4 0.79"]
  AI --> AJ["m3 0.74"]
  AJ --> AK["m4 0.79"]
  AK --> AL["m3 0.74"]
  AL --> AM["m4 0.79"]
  AM --> AN["m3 0.74"]
  AN --> AO["m4 0.79"]
  AO --> AP["m3 0.74"]
  AP --> AQ["m4 0.79"]
  AQ --> AR["m3 0.74"]
  AR --> AS["m4 0.79"]
  AS --> AT["m3 0.74"]
  AT --> AU["m4 0.79"]
  AU --> AV["m3 0.74"]
  AV --> AW["m4 0.79"]
  AW --> AX["m3 0.74"]
  AX --> AY["m4 0.79"]
```
</details>

Figure 1: Motivation of SelectStream. Relevant events may appear sparsely over a long stream, while later queries require evidence beyond the recent context. The example shows a counting question where inserted movie clips form sparse surprise peaks. The right panel illustrates the desired trade-off: stronger long-range memory and efficiency without sacrificing current-scene perception.

The lesson is not that memory is unnecessary. A pure recent-window strategy has an obvious failure mode: once evidence leaves the local context, it is no longer recoverable. A model that observed a critical object or event minutes earlier still needs a mechanism to access that information later. The real problem is therefore more subtle. A streaming model must decide when to write new evidence, what to keep when capacity runs out, how to retrieve evidence for a given query, and how much historical context to expose at generation time. Existing methods often instantiate these choices through specific memory or compression heuristics, but they are rarely formulated as a unified evidence-allocation problem across all four dimensions.

Motivated by this view, we introduce SelectStream, a selective latent-memory framework for streaming video understanding. SelectStream formulates streaming memory as budgeted online latent evidence allocation. It keeps the current observation directly available to a frozen VLM, while historical information is accessed only through a compact query-conditioned latent evidence budget. This design targets the gap between recent-window baselines, which discard older evidence, and memory-heavy systems, which risk exposing too much irrelevant history.

SelectStream implements this idea with a dynamic latent evidence graph. Projected visual embeddings from a frozen backbone VLM are written into fixed-capacity memory, avoiding raw-frame replay and unprojected visual features. Surprise-driven Adaptive Windowing decides when to write, Latent Visual Memory decides what to preserve through priority-aware consolidation, and Graph Attention Reasoning decides how to retrieve compact query-conditioned evidence subgraphs. Retrieved memory states are calibrated and injected as latent evidence tokens, allowing the frozen VLM to use historical evidence without growing its context with stream length.

The main contributions are:

• We formulate selective remembering in streaming video understanding as budgeted online latent evidence allocation, where a model must decide when to write memory, what evidence to preserve or consolidate, how to read history for a query, and how much evidence to expose under fixed budgets.  
• We introduce SelectStream, a dynamic latent-memory architecture that writes projected VLM visual embeddings into a fixed-capacity evidence graph, allocates temporal memory resolution with surprise-driven windowing, and preserves useful history through priorityaware consolidation.  
• We design a query-conditioned graph reasoning and evidence injection interface that retrieves compact historical evidence and injects calibrated latent evidence tokens into a frozen VLM without replaying raw frames or converting memory into text summaries.

## 2 Related Work

Streaming video understanding. Streaming video understanding requires models to answer from a causally observed video prefix rather than a fully available offline video. Recent work studies online video QA, proactive response timing, streaming instruction tuning, and real-time interaction [26, 27, 4, 42, 32, 48, 45, 11, 12, 40, 41, 10, 14, 39]. These methods address when to respond and how to align perception with generation under causal and computational constraints. SelectStream focuses on preserving, consolidating, and retrieving historical visual evidence under fixed memory and context budgets.

Memory and context management. Streaming video LLMs must construct a bounded working context from an unbounded stream. Existing methods use memory banks or event memories [44, 50, 51, 43, 13], KV-cache memory and retrieval [7, 25, 47, 52], visual token pruning or compression [49, 17, 5, 36], and recurrent or latent states [26, 27, 9]. SelectStream instead organizes history as a dynamic latent evidence graph: projected VLM embeddings are written into memory, consolidated with priority-aware penalties, retrieved as query-conditioned subgraphs, and injected as calibrated latent evidence tokens. A fuller discussion is provided in Appendix A.

## 3 Methodology

SelectStream instantiates the budgeted evidence-allocation view with a dynamic latent evidence graph, as shown in Figure 2. It makes three online decisions: Surprise-driven Adaptive Windowing (SAW) decides when to write, Latent Visual Memory (LVM) decides what to preserve under fixed capacity, and Graph Attention Reasoning (GAR) decides how to read compact evidence for a query.

At time step t, SelectStream maintains

$$
G _ {t} = (\mathcal {M} _ {t}, E _ {t}), \tag {1}
$$

where $\mathcal { M } _ { t }$ is the active memory set and $E _ { t }$ contains temporal and semantic relations. Each node stores a latent state $h _ { i } \in \mathbb { R } ^ { d }$ and lightweight metadata such as temporal span and surprise statistics. Compared with fixed-window top-k retrieval, SelectStream uses event-adaptive write units, priorityaware consolidation, relation-aware subgraph routing, and calibrated latent evidence injection. The online update loop is summarized in Appendix B.1.

Unless otherwise specified, SelectStream stores projected visual embeddings: each observation is encoded by the frozen VLM visual tower and mapped by its native multimodal projector $\Phi _ { v } ( \cdot )$ into the decoder input-embedding space. The primary user-facing budgets are memory capacity $N ,$ subgraph budget ${ \dot { B } } ,$ and evidence budget M; SAW also uses $L _ { \operatorname* { m i n } } , L _ { \operatorname* { m a x } }$ , and a surprise-energy budget $B _ { s }$ to avoid degenerate segments. Other coefficients are fixed on validation data or learned when applicable, and are reported in Appendix C.

## 3.1 Surprise-driven Adaptive Windowing

SAW implements the when-to-write decision by assigning coarse memory entries to stable intervals and finer resolution near surprising transitions. For each observation, it caches $g _ { t } = \mathrm { P o o l } ( \Phi _ { v } ( x _ { t } ) )$ and estimates surprise from attention shift and feature change. By default, $A _ { t }$ is the frozen backbone’s prompt-to-visual or visual-token attention, averaged over heads and pooled into fixed visual bins; if such attentions are unavailable, we use a feature-group proxy from projected visual embeddings.

The surprise score is

$$
s _ {t} = \lambda s _ {t} ^ {\text { attn }} + (1 - \lambda) s _ {t} ^ {\text { feat }}, \quad \bar {s} _ {t} = \rho \bar {s} _ {t - 1} + (1 - \rho) s _ {t}. \tag {2}
$$

where $s _ { t } ^ { \mathrm { a t t n } } = \operatorname { J S } ( A _ { t } \parallel A _ { t - 1 } )$ and $s _ { t } ^ { \mathrm { f e a t } } = 1 - \cos ( g _ { t } , g _ { t - 1 } )$ . Head averaging and bin pooling make the JS signal depend on coarse spatial redistribution rather than token-level fluctuations.

A segment closes only after the minimum length is reached and one of three triggers fires: a surprise spike, accumulated surprise energy, or the maximum segment length:

$$
t - t _ {\text { start }} \geq L _ {\min} \quad \text { and } \quad \left(\bar {s} _ {t} > \theta_ {\text { high }} \vee \sum_ {k = t _ {\text { start }}} ^ {t} \bar {s} _ {k} > B _ {s} \vee t - t _ {\text { start }} \geq L _ {\max}\right), \tag {3}
$$

where $B _ { s }$ is the surprise-energy budget, $L _ { \mathrm { m a x } }$ is the maximum segment length, and $\theta _ { \mathrm { h i g h } }$ is an adaptive spike threshold. Proposition 1 formalizes that these triggers control segment generation, while consolidation bounds active memory by N. Fixed thresholding and smoothing constants are listed in Appendix C.

![](images/15d19ac6eb60a6f1b644151165bc12115239bfb82236d180d90442db992f99eb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Streaming Video Input"] --> B["Frozen VLM Φv Visual Encoder + Projector"]
  B --> C["Visual Embeddings"]
  C --> D["Prompt / Query (q) Why does the player wearing the number nine red and blue jersey celebrate?"]
  D --> E["Frozen / Trainable Embedding / Node"]
    
  F["(a) Surprise-driven Adaptive Windowing(SAW) — When to Write"] --> G["Surprise at time t"]
  G --> H["Feature Change: s_t^attn = JS(A_t||A_{t-1})"]
  H --> I["Feature Change: s_t^feat = 1 - cos(g_t, g_{t-1})"]
  I --> J["Feature Change: g_{t-1} → g_t"]
  J --> K["Feature Change: g_t → g_t"]
    
  L["(b) Latent Visual Memory (LVM) — What to Preserve"] --> M["Write Decision (from SAW)"]
  M --> N["Update if max_i r_i > r_s and S_j < r_s"]
  N --> O["Update node i* else create new node"]
    
  P["(c) Graph Attention Reasoning (GAR) — How to Read"] --> Q["Query Encoding u = Enc_q(q)"]
  Q --> R["Coarse Scoring score_i 0.81 0.42 ... 0.23 0.67"]
  R --> S["Routing Subgraph Top-k seeds"]
  S --> T["Subgraph (≤ B) K layers"]
  T --> U["Graph Attention Reasoning temporal + semantic"]
  U --> V["Re-score & Select Top M nodes"]
  V --> W["Evidence Read e_m = LN(W_e ĥ_im)"]
  W --> X["e_1 ... e_M"]
    
  Y["(d) Latent Evidence Injection"] --> Z["Calibrate Evidence e_m = LN(W_e ĥ_im)"]
  Z --> AA["Assemble Input Prompt + Current Observation + Retrieved Evidence Tokens {X_prompt ; X_cur ; e_1 , ..., e_N}"]
  AA --> AB["Prompt Current Evidence (M)"]
    
  AD["(e) Precision-preserving Consolidation (when |M_t| > N) Merge pair (u, v) with minimum penalty"] --> AE["Gated Update (key idea) g = σ(MLP([z_j; h_t; S_j; Δt"]))]
  AE --> AF["MLP MLP h_t (old) g h_t (new)"]
  AF --> AG["Dynamic Memory Graph G_t = (M_t, E_t)"]
  AG --> AH["Temporal Similarity"]
  AH --> AI["Metadata per node m_i g = {t^start, t^end, S_i, c_i, r_i^read, m_i^merge, ...}"]
    
  AI --> AJ["Answer: He scored a goal."]
    
  AK["(f) Training Objective L = L_ans + βL_ret + γLspar"] --> AL["Training Objective"]
```
</details>

Figure 2: Overview of SelectStream. The model writes projected VLM visual embeddings into a budgeted latent memory graph, retrieves a query-conditioned evidence subgraph, and injects calibrated latent evidence tokens into a frozen MLLM for answer generation.

## 3.2 Latent Visual Memory and Dynamic Memory Graph

LVM implements the what-to-preserve decision under fixed capacity. When SAW closes a segment $\operatorname { s e g } _ { j } ,$ cached projected embeddings are encoded as $z _ { j } = \mathrm { S e g E n c } ( \{ g _ { t } : t \in \mathrm { s e g } _ { j } \} )$ by a lightweight Transformer segment encoder with temporal positions and learned query pooling. Each memory node stores a latent state $h _ { i } \in \mathbb { R } ^ { d }$ and metadata such as temporal span, accumulated surprise, write count, read count, and merge count.

To write a segment, we compare $z _ { j }$ with active nodes using $r _ { i } = \cos ( z _ { j } , h _ { i } )$ . If $\operatorname* { m a x } _ { i } r _ { i } > \tau _ { r }$ and $\bar { s } _ { j } < \tau _ { s }$ , the segment updates the selected node $i ^ { * } { } ;$ ; otherwise, it creates a new node. The update is gated:

$$
g = \sigma (\operatorname{MLP} ([ z _ {j}; h _ {i ^ {*}}; \bar {s} _ {j}; \Delta t ])), \quad h _ {i ^ {*}} \leftarrow (1 - g) h _ {i ^ {*}} + g f _ {\text { write }} (z _ {j}, h _ {i ^ {*}}). \tag {4}
$$

Here $f _ { \mathrm { w r i t e } }$ is a trainable two-layer MLP over $[ z _ { j } ; h _ { i ^ { * } } ]$ , and $\Delta t$ is the time since the node was last updated. The gate prevents direct replacement from erasing useful history while avoiding unconditional averaging that dilutes important changes.

$w _ { u v } ^ { \mathrm { t m p } } = \exp ( - c \cdot \bar { s } _ { v } )$ $w _ { u v } ^ { \mathrm { s i m } } = \cos ( h _ { u } , h _ { v } )$ performs priority-preserving consolidation by merging the pair $( u , v )$ with the smallest graph-aware penalty

$$
\pi_ {u v} = p _ {u v} ^ {\text { sim }} + p _ {u v} ^ {\text { pri }}. \tag {5}
$$

The first term favors merging semantically redundant nodes, while the priority term protects surprising, frequently accessed, or recently updated evidence. Full scoring, metadata updates, and distortion derivations are provided in Appendices B.2 and B.3.

## 3.3 Query-conditioned Subgraph Retrieval and Graph Attention Reasoning

GAR implements the how-to-read decision. When a query q arrives, the model routes it through a small memory subgraph rather than reasoning over the full graph. We encode the query as $u = \operatorname { E n c } _ { q } ( q )$ . In our implementation, $\operatorname { E n c } _ { q }$ appends learned query tokens to the current prompt-side hidden states and applies a lightweight Transformer; the resulting query tokens are mean-pooled into u in the same latent space as memory states.

We first assign each active node a coarse relevance score:

$$
\mathrm{score} _ {i} = \cos (u, h _ {i}) + \eta \hat {s} _ {i} - \xi_ {\ell} \hat {\ell} _ {i} - \xi_ {m} \hat {m} _ {i} ^ {\mathrm{merge}}. \tag {6}
$$

$\ell _ { i } = t _ { i } ^ { \mathrm { e n d } } - t _ { i } ^ { \mathrm { s t a r t } } + 1$ $\hat { \ell } _ { i }$ $\hat { m } _ { i } ^ { \mathrm { m e r g e } }$ denote active-memory normalized span length and merge count, respectively. The first term measures semantic compatibility, the normalized surprise term $\hat { s } _ { i }$ biases retrieval toward salient events, and the last two terms mildly downweight coarse nodes with long temporal spans or many prior merges. Thus merged nodes can still be retrieved when semantically relevant, but precise grounding prefers sharper unmerged alternatives when available. The scalar coefficients are fixed implementation constants reported in Appendix C.

We choose top-k seeds and perform query-conditioned routing over their temporal and semantic neighborhoods. For a routed edge $( i , j )$ , the neighbor routing score is

$$
\psi_ {i j} (q) = \operatorname{score} _ {j} + \alpha_ {r} w _ {i j}, \tag {7}
$$

where $w _ { i j }$ is the normalized temporal or similarity edge support. If the expansion exceeds the subgraph budget $B ,$ , only the highest-scoring routed nodes are retained. This makes expansion a query-dependent evidence routing step rather than an unconditional hop expansion.

Reasoning is performed on the retrieved subgraph with relational graph attention:

$$
e _ {i j} = q _ {i} ^ {\top} k _ {j} + b _ {\text { type }} + b _ {\Delta t} + b _ {w}, \tag {8}
$$

where $b _ { \mathrm { t y p e } }$ is a learned scalar bias for the edge type, $b _ { \Delta t }$ penalizes large temporal gaps with a learnable positive scale, and $b _ { w }$ adds edge-support bias. The precise temporal-gap parameterization is given in Appendix B.4. The normalized attention update is

$$
\alpha_ {i j} = \operatorname{softmax} _ {j} (e _ {i j}), \quad h _ {i} ^ {(\ell + 1)} = h _ {i} ^ {(\ell)} + \sum_ {j \in \mathcal {N} (i)} \alpha_ {i j} v _ {j}. \tag {9}
$$

After K layers, each candidate node has gathered temporal and semantic context from its neighborhood. We re-score the refined nodes with the same query-conditioned terms in Eq. (6), select the top-M nodes, and read them out as evidence tokens:

$$
E = \{e _ {m} \} _ {m = 1} ^ {M}, \quad e _ {m} = \mathrm{LN} (W _ {e} \tilde {h} _ {i _ {m}}), \tag {10}
$$

where $\tilde { h } _ { i _ { m } }$ is a selected refined node state. $W _ { e }$ is a trainable calibration/projection layer followed by LayerNorm into the MLLM token-embedding dimension. Since memory states already originate from projected visual embeddings, $W _ { e }$ is not a full cross-modal translator; it calibrates distributional drift introduced by segment encoding, gated writing, consolidation, and graph reasoning.

## 3.4 Latent Evidence Injection for Answer Generation

Latent evidence injection is the final allocation step: only the retrieved evidence budget M is exposed to the MLLM. Instead of replaying frames or converting history into text summaries, the retrieved evidence is injected as latent embeddings. At query time, the MLLM consumes the current prompt and visual observation together with the evidence embeddings,

$$
\mathbf {X} _ {\text { in }} = \left[ \mathbf {X} _ {\text { prompt }}; \mathbf {X} _ {\text { cur }}; e _ {1}, \dots , e _ {M} \right], \quad y \sim p _ {\theta} (y \mid \mathbf {X} _ {\text { in }}). \tag {11}
$$

In practice, the backbone VLM remains frozen, and $W _ { e }$ is trained as part of the SelectStream evidence calibration interface. This separates online memory writing from query-time reading and grounds generation in explicitly retrieved history. Proposition 2 formalizes how reducing the candidate set to M nodes improves a lower bound on attention to the relevant evidence when retrieval recall is high.

## 3.5 Theoretical Properties

The following properties provide two practical intuitions behind SelectStream. Proposition 1 shows that adaptive writing remains budget-controlled even when segment boundaries are generated online. Proposition 2 explains why compact query-conditioned retrieval can reduce evidence dilution: when the relevant evidence is included with reasonable recall, exposing fewer distractors increases the attention mass available to that evidence.

Proposition 1 (Bounded segment and memory complexity). Let $S _ { T }$ be the number of SAW segments generated before consolidation over a stream of length T , counting the final open segment if it is flushed at the end. Assume the adaptive spike threshold is lower-bounded by $\theta _ { \operatorname* { m i n } } > 0$ , i.e., $\theta _ { \mathrm { h i g h } } ( t ) \geq \theta _ { \mathrm { m i n } }$ for all t. Then

$$
S _ {T} \leq \frac {\sum_ {t = 1} ^ {T} \bar {s} _ {t}}{B _ {s}} + \frac {\sum_ {t = 1} ^ {T} \bar {s} _ {t}}{\theta_ {\min}} + \frac {T}{L _ {\max}} + 1. \tag {12}
$$

After consolidation, the active memory graph always satisfies $| \mathcal { M } _ { T } | \leq N$

Proof sketch. Partition closed segments by the trigger in Eq. (3): accumulated surprise, spike surprise, or maximum length. Disjointness bounds the three counts by the surprise mass and the maximum-window budget. Each write either updates an existing node or creates one node, and consolidation restores $| \bar { \mathcal { M } } _ { t } | \leq N$ whenever the budget is exceeded. The complete proof is provided in Appendix B.6.

Proposition 2 (Evidence attention concentration). Consider query-time attention over a retrieved candidate set of size M . Suppose the relevant evidence node $h ^ { * }$ is included in this set with probability r. When included, its attention logit is at least $\mu _ { \mathrm { s i g } } ,$ and each irrelevant candidate logit ℓ satisfies $\mathbb { E } [ \exp ( \ell ) \mid h ^ { * }$ included] $\vert \le \exp ( \sigma ^ { 2 } / 2 )$ . Then the expected attention assigned to the relevant evidence is lower-bounded by

$$
\mathbb {E} [ \alpha_ {h ^ {*}} ] \geq r \cdot \frac {\exp (\mu_ {\mathrm{sig}})}{\exp (\mu_ {\mathrm{sig}}) + (M - 1) \exp (\sigma^ {2} / 2)}. \tag {13}
$$

Proof sketch. Conditioned on retrieving $h ^ { * }$ , its softmax numerator is at least $\exp ( \mu _ { \mathrm { s i g } } )$ . The assumed exponential-moment bound controls the expected contribution of each irrelevant candidate, so the expected distractor mass scales with $M - \bar { 1 }$ . Multiplying by retrieval recall r gives Eq. (13). The assumptions on r and $\mu _ { \mathrm { s i g } }$ describe retrieval quality and evidence separability rather than architectural guarantees; in practice, they are encouraged by retrieval supervision in Eq. (36), answer loss through the injected evidence tokens, and the evidence calibration layer, and are evaluated empirically with evidence Recall@M and temporal-overlap metrics. The complete proof is provided in Appendix B.7.

## 3.6 Training Objective

We perform supervised fine-tuning with a frozen backbone VLM. Only the SelectStream modules are updated, using the objective

$$
\mathcal {L} = \mathcal {L} _ {\text { ans }} + \beta \mathcal {L} _ {\text { ret }} + \gamma \mathcal {L} _ {\text { spar }}. \tag {14}
$$

Here $\mathcal { L } _ { \mathrm { a n s } }$ is autoregressive answer loss, $\mathcal { L } _ { \mathrm { r e t } }$ trains evidence retrieval when evidence annotations are available, and $\bar { \mathcal { L } } _ { \mathrm { { s p a r } } }$ regularizes diffuse routing and redundant evidence. For answer-only data, $\mathcal { L } _ { \mathrm { r e t } } = 0 ;$ these examples train answer generation from retrieved evidence but do not provide direct retrieval grounding. The backbone VLM is not updated; the segment encoder, gate/write MLPs, query encoder, GAR module, and evidence calibration/projection layer are trained by this objective. The exact retrieval and sparsity losses are specified in Appendix B.5.

Overall, N, B, and M separately control memory capacity, retrieval scope, and generation context. The stream still requires one frozen visual pass and native multimodal projection per incoming observation, but segment writing reuses cached projected embeddings and query-time reasoning is controlled by these budgets rather than by full video length. Exact consolidation can use all-pair penalties under small N or top-similarity candidates for real-time deployment. With fixed budgets, the active memory state is $O ( \bar { 1 } )$ with respect to processed stream length; see Appendix F.

## 4 Experiments

## 4.1 Experimental Setup

Training and implementation. We fine-tune SelectStream on Streamo-Instruct-465K [42]. Timestamped examples provide retrieval supervision through overlap between annotated evidence intervals and memory-node spans, while examples without reliable timestamps use answer loss only; details are in Appendix E. The backbone VLM is frozen, and only the SelectStream modules are trained. We train separate modules for Qwen2.5-VL-7B and Qwen3-VL-8B on NVIDIA A100 80GB GPUs. Default budgets and hyperparameters are reported in Appendix C.

Baselines and metrics. We compare with recent-window baselines including SimpleStream [29], streaming video models including Streamo [42], StreamForest [50], Flash-VStream [51], and HER-MES [52], and simple memory policies such as uniform sampling, FIFO, random eviction, and similarity merging. All methods use the same causal protocol with matched token budgets when possible. We report accuracy, query latency, active memory size, injected evidence tokens, and peak GPU memory under fixed budgets N, B, and M. Detailed benchmark and metric definitions are provided in Appendix D.

Table 1: Main results on StreamingBench and OVO-Bench. RT, BT, and FAR denote Real-Time Visual Perception, Backward Tracing, and Forward Active Responding, respectively. Best results are in bold and second-best results are underlined for each metric.

<table><tr><td rowspan="2">Model</td><td rowspan="2">#Frames</td><td rowspan="2">StreamingBench</td><td colspan="4">OVO-Bench</td></tr><tr><td>RT Avg.</td><td>BT Avg.</td><td>RT/BT Avg.</td><td>FAR / Overall</td></tr><tr><td colspan="7">Offline Video LLMs</td></tr><tr><td>Qwen2.5-VL-7B [2]</td><td>1 fps</td><td>73.31</td><td>59.90</td><td>44.70</td><td>52.28</td><td>-/-</td></tr><tr><td>LLaVA-OneVision-7B [18]</td><td>32</td><td>71.12</td><td>64.00</td><td>43.70</td><td>53.85</td><td>50.50 / 52.74</td></tr><tr><td>InternVL2-8B [6]</td><td>16</td><td>63.72</td><td>60.40</td><td>43.40</td><td>51.90</td><td>46.60 / 50.15</td></tr><tr><td>LLaVA-Video-7B [54]</td><td>64</td><td>-</td><td>63.50</td><td>40.40</td><td>51.95</td><td>54.82 / 52.91</td></tr><tr><td>Qwen2-VL-7B [33]</td><td>64</td><td>69.04</td><td>60.70</td><td>48.60</td><td>54.62</td><td>-/-</td></tr><tr><td>LongVU-7B [28]</td><td>1 fps</td><td>-</td><td>57.40</td><td>39.50</td><td>48.45</td><td>-/-</td></tr><tr><td colspan="7">Online / Streaming Video LLMs</td></tr><tr><td>VideoLLM-online-8B [3]</td><td>2 fps</td><td>35.99</td><td>20.80</td><td>17.70</td><td>19.26</td><td>-/-</td></tr><tr><td>Flash-VStream-7B [51]</td><td>1 fps</td><td>23.23</td><td>28.40</td><td>27.40</td><td>27.90</td><td>45.09 / 33.61</td></tr><tr><td>Dispider-7B [27]</td><td>1 fps</td><td>67.63</td><td>54.60</td><td>36.10</td><td>45.35</td><td>34.72 / 41.78</td></tr><tr><td>TimeChat-Online-7B [49]</td><td>1 fps</td><td>75.28</td><td>61.90</td><td>41.70</td><td>51.80</td><td>-/-</td></tr><tr><td>StreamForest-7B [50]</td><td>1 fps</td><td>77.26</td><td>61.20</td><td>52.00</td><td>56.60</td><td>-/-</td></tr><tr><td>Streamo-7B [42]</td><td>1 fps</td><td>-</td><td>65.98</td><td>46.10</td><td>56.04</td><td>54.77 / 55.61</td></tr><tr><td>Streamo-7B [42]</td><td>2 fps</td><td>-</td><td>67.44</td><td>49.18</td><td>58.31</td><td>56.96 / 57.86</td></tr><tr><td>HERMES-7B [52]</td><td>1 fps</td><td>79.44</td><td>69.00</td><td>49.40</td><td>59.20</td><td>-/-</td></tr><tr><td>ThinkStream-7B [22]</td><td>1 fps</td><td>75.00</td><td>69.12</td><td>60.68</td><td>64.90</td><td>-/-</td></tr><tr><td>Qwen2.5-VL-7B + 4f [29]</td><td>1fps</td><td>78.47</td><td>78.40</td><td>51.90</td><td>65.13</td><td>-/-</td></tr><tr><td>Qwen3-VL-8B + 4f [29]</td><td>1fps</td><td>80.59</td><td>81.40</td><td>54.00</td><td>67.70</td><td>-/-</td></tr><tr><td>SELECTSTREAM-Qwen2.5-VL-7B</td><td>1 fps</td><td>81.42</td><td>80.85</td><td>61.05</td><td>70.95</td><td>55.23 / 65.71</td></tr><tr><td>SELECTSTREAM-Qwen3-VL-8B</td><td>1 fps</td><td>82.67</td><td>82.76</td><td>62.20</td><td>72.48</td><td>56.13 / 67.03</td></tr></table>

## 4.2 Main Results

Online streaming benchmarks. As shown in Table 1, SelectStream consistently outperforms offline video LLMs, online/streaming video LLMs, and strong recent-window baselines on StreamingBench and OVO-Bench. It achieves the best StreamingBench scores among all compared methods, reaching 81.42% with Qwen2.5-VL-7B and 82.67% with Qwen3-VL-8B. Compared with the corresponding recent-window baselines, this gives gains of 2.95% and 2.08%, indicating that selective latent memory provides additional benefit beyond simply retaining the latest frames. On OVO-Bench, the strongest gains appear on Backward Tracing, which directly evaluates the use of prior visual context. SelectStream improves BT Avg. from 51.90% to 61.05% on Qwen2.5-VL-7B and from 54.00% to 62.20% on Qwen3-VL-8B, while maintaining strong Real-Time Visual Perception scores of 80.85% and 82.76%. This leads to RT/BT averages of 70.95% and 72.48%, outperforming prior online models such as StreamForest, Streamo, and ThinkStream. Overall, these results show that SelectStream improves history-dependent reasoning while retaining current-scene perception.

Offline video benchmarks. To evaluate generalization beyond streaming-specific benchmarks, we further test SelectStream on VideoMME, MLVU, and MVBench using a causal final-query protocol in Table 2. SelectStream remains competitive while using a bounded latent evidence budget. With Qwen2.5-VL-7B, it improves the average score from 68.3% to 70.4%, with gains of 2.7% and 2.8% on VideoMME and MLVU. With Qwen3-VL-8B, it reaches the best overall average of 74.4%, improving over the frozen backbone by 1.7%. The larger gains on VideoMME and MLVU suggest that latent memory is most helpful for longer videos requiring temporal evidence, while the smaller gain on MVBench is consistent with its stronger focus on short-range perception and action understanding. These results suggest that SelectStream does not trade offline video understanding for streaming efficiency; instead, it preserves the backbone’s general visual reasoning ability while adding useful long-range evidence.

## 4.3 Ablation Study

Effectiveness of memory allocation. Using SelectStream-Qwen2.5-VL-7B, Table 3 shows that the full memory allocation strategy outperforms simpler writing and consolidation policies, achieving 81.42% on StreamingBench, 65.71% on OVO-Bench, and 73.0% on MLVU. Replacing SAW with fixed segments reduces the scores to 79.86%, 64.21%, and 71.2%, indicating that event-adaptive segmentation helps preserve informative temporal changes. Removing gated writing also causes a consistent drop, suggesting that direct updates are less effective for balancing old and new evidence.

Table 2: Offline video generalization results on VideoMME, MLVU, and MVBench.

<table><tr><td>Model</td><td>Size</td><td>#Frames</td><td>VideoMME</td><td>MLVU</td><td>MVBench</td><td>Avg.</td></tr><tr><td colspan="7">Offline Video LLMs</td></tr><tr><td>InternVL2-8B [6]</td><td>8B</td><td>64</td><td>54.0</td><td>64.0</td><td>65.8</td><td>61.3</td></tr><tr><td>LongVA-7B [53]</td><td>7B</td><td>64</td><td>52.6</td><td>56.3</td><td>-</td><td>-</td></tr><tr><td>LLaVA-OneVision-7B [18]</td><td>7B</td><td>32</td><td>58.2</td><td>64.7</td><td>56.7</td><td>59.9</td></tr><tr><td>Qwen2-VL-7B [33]</td><td>7B</td><td>64</td><td>63.3</td><td>-</td><td>67.0</td><td>-</td></tr><tr><td>LongVU-7B [28]</td><td>7B</td><td>1 fps</td><td>60.6</td><td>65.4</td><td>66.9</td><td>64.3</td></tr><tr><td>LLaVA-Video-7B [54]</td><td>7B</td><td>64</td><td>63.3</td><td>70.8</td><td>58.6</td><td>64.2</td></tr><tr><td colspan="7">Online / Streaming Video LLMs</td></tr><tr><td>MovieChat-7B [31]</td><td>7B</td><td>2048</td><td>38.2</td><td>25.8</td><td>55.1</td><td>39.7</td></tr><tr><td>VideoChat-Online-4B [16]</td><td>4B</td><td>-</td><td>54.4</td><td>60.8</td><td>65.2</td><td>60.1</td></tr><tr><td>Dispider-7B [27]</td><td>7B</td><td>1 fps</td><td>57.2</td><td>61.7</td><td>-</td><td>-</td></tr><tr><td>StreamForest-7B [50]</td><td>7B</td><td>1 fps</td><td>61.4</td><td>70.0</td><td>70.2</td><td>67.2</td></tr><tr><td>StreamForest-7B (FT-drive) [50]</td><td>7B</td><td>1 fps</td><td>61.9</td><td>69.6</td><td>68.6</td><td>66.7</td></tr><tr><td>Streamo-7B [42]</td><td>7B</td><td>1 fps</td><td>67.9</td><td>-</td><td>72.3</td><td>-</td></tr><tr><td>Qwen2.5-VL-7B [2]</td><td>7B</td><td>max 768</td><td>65.1</td><td>70.2</td><td>69.6</td><td>68.3</td></tr><tr><td>Qwen3-VL-8B [1]</td><td>8B</td><td>2 fps, max 2048</td><td>71.4</td><td>78.1</td><td>68.7</td><td>72.7</td></tr><tr><td>SELECTSTREAM-Qwen2.5-VL-7B</td><td>7B</td><td>1 fps, max 1024</td><td>67.8 (+2.7)</td><td>73.0 (+2.8)</td><td>70.4 (+0.8)</td><td>70.4 (+2.1)</td></tr><tr><td>SELECTSTREAM-Qwen3-VL-8B</td><td>8B</td><td>1 fps, max 1024</td><td>73.2 (+1.8)</td><td>80.0 (+1.9)</td><td>69.9 (+1.2)</td><td>74.4 (+1.7)</td></tr></table>

Among consolidation policies, FIFO performs worst because it can discard old but still relevant evidence, while similarity-only merging improves over FIFO but remains below the full model. This shows that priority factors such as surprise, recency, and access frequency help preserve useful evidence under a fixed node budget.

Effectiveness of evidence readout. Using the same Qwen2.5-VL-7B setting, Table 4 shows that retrieved memory must be routed and calibrated before injection. Replacing GAR with top-k retrieval reduces OVO-Bench from 65.71% to 63.92% and MLVU from 73.0% to 71.1%. Fixed-hop expansion partially recovers performance, but remains below query-conditioned routing because it does not adapt expansion to the query. The largest drop comes from removing evidence calibration, reducing StreamingBench to 78.96% and OVO-Bench to 63.08%; this confirms that memory states are not automatically compatible with the frozen VLM input space. Removing $\mathcal { L } _ { \mathrm { r e t } }$ weakens grounding, while removing $\mathcal { L } _ { \mathrm { s p a r } }$ gives smaller but consistent drops by allowing redundant evidence.

Table 3: Ablation on memory allocation under the same node budget.

<table><tr><td>Variant</td><td>Streaming</td><td>OVO-Bench</td><td>MLVU</td></tr><tr><td>Fixed segments w/o SAW</td><td>79.86</td><td>64.21</td><td>71.2</td></tr><tr><td>w/o gated writing</td><td>80.21</td><td>64.48</td><td>71.7</td></tr><tr><td>FIFO consolidation</td><td>79.42</td><td>63.61</td><td>70.4</td></tr><tr><td>Similarity-only merging</td><td>80.03</td><td>64.03</td><td>71.0</td></tr><tr><td>Priority consolidation</td><td>81.42</td><td>65.71</td><td>73.0</td></tr></table>

Table 4: Ablation on evidence readout under the same evidence budget.

<table><tr><td>Model</td><td>Streaming</td><td>OVO-Bench</td><td>MLVU</td></tr><tr><td>Top-k retrieval w/o GAR</td><td>79.71</td><td>63.92</td><td>71.1</td></tr><tr><td>Fixed-hop expansion</td><td>80.09</td><td>64.28</td><td>71.6</td></tr><tr><td>w/o evidence calibration</td><td>78.96</td><td>63.08</td><td>70.2</td></tr><tr><td>w/o  $\mathcal{L}_{\text{ret}}$ </td><td>80.27</td><td>64.42</td><td>71.8</td></tr><tr><td>w/o  $\mathcal{L}_{\text{spar}}$ </td><td>80.76</td><td>64.82</td><td>72.3</td></tr><tr><td>Full SelectStream</td><td>81.42</td><td>65.71</td><td>73.0</td></tr></table>

## 4.4 Latent Evidence Analysis

SelectStream injects the top-M retrieved memory nodes as calibrated latent evidence tokens, so we evaluate temporal grounding and evidence-interface quality. Table 5 reports accuracy, Recall@M, and T-Overlap on timestamped examples; definitions and retrieval-only diagnostics are in Appendix E. Recall@M empirically checks the retrieval-

<table><tr><td>Variant</td><td>SB</td><td>OVO</td><td>MLVU</td><td>Recall@M</td><td>T-Overlap</td></tr><tr><td>No historical evidence</td><td>78.35</td><td>62.31</td><td>69.5</td><td>N/A</td><td>N/A</td></tr><tr><td>Random memory tokens</td><td>78.02</td><td>62.02</td><td>69.2</td><td>0.07</td><td>0.04</td></tr><tr><td>Top-k retrieval w/o GAR</td><td>79.71</td><td>63.92</td><td>71.1</td><td>0.56</td><td>0.39</td></tr><tr><td>Raw states w/o calibration</td><td>78.96</td><td>63.08</td><td>70.2</td><td>0.72</td><td>0.53</td></tr><tr><td>Retrieved-frame replay</td><td>80.84</td><td>64.88</td><td>72.4</td><td>0.72</td><td>0.53</td></tr><tr><td>Calibrated latent evidence</td><td>81.42</td><td>65.71</td><td>73.0</td><td>0.72</td><td>0.53</td></tr></table>

Table 5: Temporal grounding of latent evidence.

recall term in Proposition 2 (Section 3.5). Random memory tokens stay close to no history, with only 0.07 Recall@M and 0.04 T-Overlap. Top-k retrieval improves Recall@M to 0.56, but remains below the full model’s 0.72 Recall@M and 0.53 T-Overlap. Raw states use the same retrieved nodes as the full model but are 2.46%, 2.63%, and 2.8% lower on StreamingBench, OVO-Bench, and MLVU, showing that temporal alignment alone is insufficient. Calibrated latent evidence performs best, improving over retrieved-frame replay by 0.58%, 0.83%, and 0.6% while preserving the compact evidence-token interface.

## 4.5 Budget and Efficiency Analysis

We vary memory capacity N, subgraph budget B, and evidence budget M to evaluate accuracyefficiency trade-offs. We report time to first token (TTFT) and peak GPU memory as the number of observed frames increases. Since SelectStream maintains a fixed-capacity memory graph and retrieves a bounded subgraph, its query-time cost is controlled by B and M rather than the full stream length.

The three exposed budgets show distinct trade-offs in Figure 3. Increasing N improves memory retention, increasing B mainly improves retrieval recall before saturation, and increasing M exposes more evidence but raises TTFT because more latent tokens enter the decoder context. This supports treating N, B, and M as separate user-facing budgets rather than a single context-length parameter.

The scaling experiment in Figure 4 shows that HERMES has the lowest TTFT because its query path is a lightweight KV-cache memory mechanism. SelectStream adds a small startup overhead for node scoring, subgraph routing, GAR re-ranking, and evidence calibration, but its latency grows slowly with more observed frames. Its peak memory also stays nearly flat because the stream is compressed into at most N active nodes and query-time reasoning only materializes a bounded subgraph and M evidence tokens.

![](images/a25b3b65be6335f09878f741700da84991bdf696550e261178fdeb7625286c8e.jpg)

Figure 3: Budget sensitivity of SelectStream over memory capacity N, subgraph budget B, and evidence budget M.  
![](images/b8674d7ee95e9f86777b2dc99b89ca47b58e1a72f644ccf427dab0411cdbdd48.jpg)  
$\begin{array} { r l } { \Phi \mathrm { - \mathbb { F } \mathrm { l a s h - V S t r e a m { - 7 B } } ~ \index { \Phi = - S t r e a m F o r e s t { - 7 B } } ~ \index { \Phi \to - T i m e C h a t { - 0 n i n e - 7 B } } ~ \index { \Phi \to - H E R M E S } ~ \index { \Phi \to - S e l e c t S t r e a m { - 7 B } } } } & { } \end{array}$

Figure 4: Efficiency scaling with stream length. SelectStream keeps query latency and GPU memory nearly flat as processed frames increase.

## 5 Conclusion

We presented SelectStream, a streaming video understanding framework that stores long video history as compact latent evidence instead of replaying historical frames. Through event-adaptive writing, priority-aware consolidation, query-conditioned graph retrieval, and calibrated evidence injection, SelectStream improves history-dependent reasoning while preserving current-scene perception and bounded inference cost. Across benchmarks, SelectStream consistently improves long-context video understanding; on OVO-Bench with Qwen2.5-VL-7B, it raises RT/BT Avg. from 65.13% to 70.95% and BT Avg. from 51.90% to 61.05%. These results support latent evidence allocation as a scalable direction for streaming video understanding under fixed memory and context budgets.

## References

[1] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, Mei Li, Kaixin Li, Zicheng Lin, Junyang Lin, Xuejing Liu, Jiawei Liu, Chenglong Liu, Yang Liu, Dayiheng Liu, Shixuan Liu, Dunjie Lu, Ruilin Luo, Chenxu Lv, Rui Men, Lingchen Meng, Xuancheng Ren, Xingzhang Ren, Sibo Song, Yuchong Sun, Jun Tang, Jianhong Tu, Jianqiang Wan, Peng Wang, Pengfei Wang, Qiuyue Wang, Yuxuan Wang, Tianbao Xie, Yiheng Xu, Haiyang Xu, Jin Xu, Zhibo Yang, Mingkun Yang, Jianxin Yang, An Yang, Bowen Yu, Fei Zhang, Hang Zhang, Xi Zhang, Bo Zheng, Humen Zhong, Jingren Zhou, Fan Zhou, Jing Zhou, Yuanzhi Zhu, and Ke Zhu. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.  
[2] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, et al. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.  
[3] Joya Chen, Zhaoyang Lv, Shiwei Wu, Kevin Qinghong Lin, Chenan Song, Difei Gao, Jia-Wei Liu, Ziteng Gao, Dongxing Mao, and Mike Zheng Shou. Videollm-online: Online video large language model for streaming video. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.  
[4] Joya Chen, Ziyun Zeng, Yiqi Lin, Wei Li, Zejun Ma, and Mike Zheng Shou. Livecc: Learning video llm with streaming speech transcription at scale. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.  
[5] Xueyi Chen, Keda Tao, Kele Shao, and Huan Wang. Streamingtom: Streaming token compression for efficient video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026. To appear.  
[6] Zhe Chen, Jiannan Wu, Wenhai Wang, Weijie Su, Guo Chen, Sen Xing, Muyan Zhong, Qinglong Zhang, Xizhou Zhu, Lewei Lu, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 24185–24198, 2024.  
[7] Shangzhe Di, Zhelun Yu, Guanghao Zhang, Haoyuan Li, Tao Zhong, Hao Cheng, Bolin Li, Wanggui He, Fangxun Shu, and Hao Jiang. Streaming video question-answering with in-context video kv-cache retrieval. In International Conference on Learning Representations, 2025.  
[8] Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Mengdan Shen, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.  
[9] Honghao Fu, Yuan Ouyang, Kai-Wei Chang, Yiwei Wang, Zi Huang, and Yujun Cai. Contextnav: Towards agentic multimodal in-context learning. arXiv preprint arXiv:2510.04560, 2025.  
[10] Honghao Fu, Miao Xu, Yiwei Wang, Dailing Zhang, Jun Liu, and Yujun Cai. Videostir: Understanding long videos via spatio-temporally structured and intent-aware rag. arXiv preprint arXiv:2604.05418, 2026.  
[11] Haonan Ge, Yiwei Wang, Kai-Wei Chang, Hang Wu, and Yujun Cai. Framemind: Frameinterleaved video reasoning via reinforcement learning. arXiv preprint arXiv:2509.24008, 2025.  
[12] Haonan Ge, Yiwei Wang, Ming-Hsuan Yang, and Yujun Cai. Mrfd: Multi-region fusion decoding with self-consistency for mitigating hallucinations in lvlms. arXiv preprint arXiv:2508.10264, 2025.  
[13] Zhenghui Guo, Yuanbin Man, Junyuan Sheng, Bowen Lin, Ahmed Ahmed, Bo Jiang, Boyuan Zhang, Miao Yin, Sian Jin, Omprakash Gnawal, and Chengming Zhang. Event-vstream: Event-driven real-time understanding for long video streams. arXiv preprint arXiv:2601.15655, 2026.  
[14] Yixu Huang, Bo Li, Na Li, Zhe Wang, Kaijie Chen, Haonan Ge, Qingyi Si, Yuanzhe Shen, Ruihan Yang, Guangjing Wang, et al. Gui agents for continual game generation. arXiv preprint arXiv:2605.28258, 2026.  
[15] Zhenpeng Huang, Xinhao Li, Jiaqi Li, Jing Wang, Xiangyu Zeng, Cheng Liang, Tao Wu, Xi Chen, Liang Li, and Limin Wang. Online video understanding: Ovbench and videochatonline. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.  
[16] Zhenpeng Huang, Xinhao Li, Jiaqi Li, Jing Wang, Xiangyu Zeng, Cheng Liang, Tao Wu, Xi Chen, Liang Li, and Limin Wang. Online video understanding: Ovbench and videochatonline. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 3328–3338, 2025.  
[17] Xinqi Jin, Hanxun Yu, Bohan Yu, Kebin Liu, Jian Liu, Keda Tao, Yixuan Pei, Huan Wang, Fan Dang, Jiangchuan Liu, and Weiqiang Wang. Streamingassistant: Efficient visual token pruning for accelerating online video understanding. arXiv preprint arXiv:2512.12560, 2025.  
[18] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. Transactions on Machine Learning Research, 2025.  
[19] Yifei Li, Junbo Niu, Ziyang Miao, Chunjiang Ge, Yuanhang Zhou, Qihao He, Xiaoyi Dong, Haodong Duan, Shuangrui Ding, Ping Lin, Ping Luo, Limin Wang, and Yu Qiao. Ovo-bench: How far is your video-llms from real-world online video understanding? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.  
[20] Junming Lin, Zheng Fang, Chi Chen, Zihao Wan, Fuwen Luo, Peng Li, Yang Liu, and Maosong Sun. Streamingbench: Assessing the gap for mllms to achieve streaming video understanding. arXiv preprint arXiv:2411.03628, 2024.  
[21] Yibin Liu, Yaxing Lyu, Daqi Gao, Zhixuan Liang, Weiliang Tang, Shilong Mu, Xiaokang Yang, and Yao Mu. From passive observer to active critic: Reinforcement learning elicits process reasoning for robotic manipulation. arXiv preprint arXiv:2603.15600, 2026.  
[22] Zikang Liu, Longteng Guo, Handong Li, Ru Zhen, Xingjian He, Ruyi Ji, Xiaoming Ren, Yanhao Zhang, Haonan Lu, and Jing Liu. Thinking in streaming video. arXiv preprint arXiv:2603.12938, 2026.  
[23] Xudong Lu, Huankang Guan, Yang Bo, Jinpeng Chen, Xintong Guo, Xueying Li, et al. Phostream: Benchmarking real-world streaming for omnimodal assistants in mobile scenarios. arXiv preprint arXiv:2601.22575, 2026.  
[24] Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. Egoschema: A diagnostic benchmark for very long-form video language understanding. In Advances in Neural Information Processing Systems, 2023.  
[25] Zhenyu Ning, Guangda Liu, Qihao Jin, Wenchao Ding, Minyi Guo, and Jieru Zhao. Livevlm: Efficient online video understanding via streaming-oriented kv cache and retrieval. arXiv preprint arXiv:2505.15269, 2025.  
[26] Rui Qian, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Shuangrui Ding, Dahua Lin, and Jiaqi Wang. Streaming long video understanding with large language models. In Advances in Neural Information Processing Systems, 2024.  
[27] Rui Qian, Shuangrui Ding, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Yuhang Cao, Dahua Lin, and Jiaqi Wang. Dispider: Enabling video llms with active real-time interaction via disentangled perception, decision, and reaction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.  
[28] Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, Zhuang Liu, Hu Xu, Hyunwoo J. Kim, Bilge Soran, Raghuraman Krishnamoorthi, Mohamed Elhoseiny, and Vikas  
Chandra. Longvu: Spatiotemporal adaptive compression for long video-language understanding. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 54582–54599. PMLR, 2025.  
[29] Yujiao Shen, Shulin Tian, Jingkang Yang, and Ziwei Liu. A simple baseline for streaming video understanding. arXiv preprint arXiv:2604.02317, 2026.  
[30] Yansong Shi, Qingsong Zhao, Tianxiang Jiang, Xiangyu Zeng, Yi Wang, and Limin Wang. River: A real-time interaction benchmark for video llms. arXiv preprint arXiv:2603.03985, 2026.  
[31] Enxin Song, Wenhao Chai, Guanhong Wang, Yucheng Zhang, Haoyang Zhou, Feiyang Wu, Haozhe Chi, Xun Guo, Tianbo Ye, Yanting Zhang, Yan Lu, Jenq-Neng Hwang, and Gaoang Wang. Moviechat: From dense token to sparse memory for long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18221–18232, 2024.  
[32] Haibo Wang, Bo Feng, Zhengfeng Lai, Mingze Xu, Shiyu Li, Weifeng Ge, Afshin Dehghan, Meng Cao, and Ping Huang. Streambridge: Turning your offline video large language model into a proactive streaming assistant. In Advances in Neural Information Processing Systems, 2025.  
[33] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Yang Fan, Kai Dang, Mengfei Du, Xuancheng Ren, Rui Men, Dayiheng Liu, Chang Zhou, Jingren Zhou, and Junyang Lin. Qwen2-vl: Enhancing visionlanguage model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.  
[34] Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Lvbench: An extreme long video understanding benchmark. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.  
[35] Xiaodong Wang, Langling Huang, Zhirong Wu, Xu Zhao, Teng Xu, Xuhong Xia, and Peixi Peng. Livibench: An omnimodal benchmark for interactive livestream video understanding. arXiv preprint arXiv:2601.15016, 2026.  
[36] Yiyu Wang, Xuyang Liu, Xiyan Gui, Xinying Lin, Boxue Yang, Chenfei Liao, Tailai Chen, and Linfeng Zhang. Accelerating streaming video large language models via hierarchical token compression. arXiv preprint arXiv:2512.00891, 2025.  
[37] Yueqian Wang, Xiaojun Meng, Yifan Wang, Huishuai Zhang, and Dongyan Zhao. Proactivevideoqa: A comprehensive benchmark evaluating proactive interactions in video large language models. arXiv preprint arXiv:2507.09313, 2025.  
[38] Yuxuan Wang, Yueqian Wang, Bo Chen, Tong Wu, Dongyan Zhao, and Zilong Zheng. Omnimmi: A comprehensive multi-modal interaction benchmark in streaming video contexts. arXiv preprint, 2025.  
[39] Hang Wu, Yujun Cai, Haonan Ge, Hongkai Chen, Ming-Hsuan Yang, and Yiwei Wang. Refineshot: Rethinking cinematography understanding with foundational skill evaluation. arXiv preprint arXiv:2510.02423, 2025.  
[40] Hang Wu, Yujun Cai, Zehao Li, Haonan Ge, Bowen Sun, Junsong Yuan, and Yiwei Wang. Camreasoner: Reinforcing camera movement understanding via structured spatial reasoning. arXiv preprint arXiv:2602.00181, 2026.  
[41] Haotian Xia, Haonan Ge, Junbo Zou, Hyun Woo Choi, Xuebin Zhang, Danny Suradja, Botao Rui, Ethan Tran, Wendy Jin, Zhen Ye, et al. Sportr: A benchmark for multimodal large language model reasoning in sports. arXiv preprint arXiv:2511.06499, 2025.  
[42] Jiaer Xia, Peixian Chen, Mengdan Zhang, Xing Sun, and Kaiyang Zhou. Streaming video instruction tuning. arXiv preprint arXiv:2512.21334, 2025.  
[43] Yiweng Xie, Bo He, Junke Wang, Xiangyu Zheng, Ziyi Ye, and Zuxuan Wu. Fluxmem: Adaptive hierarchical memory for streaming video understanding. arXiv preprint arXiv:2603.02096, 2026.  
[44] Haomiao Xiong, Zongxin Yang, Jiazuo Yu, Yunzhi Zhuge, Lu Zhang, Jiawen Zhu, and Huchuan Lu. Streaming video understanding and multi-round interaction with memory-enhanced knowledge. In International Conference on Learning Representations, 2025.  
[45] Ruyi Xu, Guangxuan Xiao, Yukang Chen, Liuning He, Kelly Peng, Yao Lu, and Song Han. Streamingvlm: Real-time understanding for infinite video streams. In International Conference on Learning Representations, 2026.  
[46] Jingkang Yang, Shuai Liu, Hongming Guo, Yuhao Dong, Xiamengwei Zhang, Sicheng Zhang, Pengyun Wang, Zitang Zhou, Binzhu Xie, Ziyue Wang, Bei Ouyang, Zhengyu Lin, Marco Cominelli, Zhongang Cai, Yuanhan Zhang, Peiyuan Zhang, Fangzhou Hong, Joerg Widmer, Francesco Gringoli, Lei Yang, Bo Li, and Ziwei Liu. Egolife: Towards egocentric life assistant. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.  
[47] Yanlai Yang, Zhuokai Zhao, Satya Narayan Shukla, Aashu Singh, Shlok Kumar Mishra, Lizhu Zhang, and Mengye Ren. Streammem: Query-agnostic kv cache memory for streaming video understanding. arXiv preprint arXiv:2508.15717, 2025.  
[48] Zhenyu Yang, Kairui Zhang, Yuhang Hu, Bing Wang, Shengsheng Qian, Bin Wen, Fan Yang, Tingting Gao, Weiming Dong, and Changsheng Xu. Livestar: Live streaming assistant for real-world online video understanding. In Advances in Neural Information Processing Systems, 2025.  
[49] Linli Yao, Yicheng Li, Yuancheng Wei, Lei Li, Shuhuai Ren, Yuanxin Liu, Kun Ouyang, Lean Wang, Shicheng Li, Sida Li, Lingpeng Kong, Qi Liu, Yuanxing Zhang, and Xu Sun. Timechatonline: 80% visual tokens are naturally redundant in streaming videos. In Proceedings of the ACM International Conference on Multimedia, 2025.  
[50] Xiangyu Zeng, Kefan Qiu, Qingyu Zhang, Xinhao Li, Jing Wang, Jiaxin Li, Ziang Yan, Kun Tian, Meng Tian, Xinhai Zhao, Yi Wang, and Limin Wang. Streamforest: Efficient online video understanding with persistent event memory. arXiv preprint arXiv:2509.24871, 2025.  
[51] Haoji Zhang, Yiqin Wang, Yansong Tang, Yong Liu, Jiashi Feng, and Xiaojie Jin. Flash-vstream: Efficient real-time understanding for long video streams. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025.  
[52] Haowei Zhang, Shudong Yang, Jinlan Fu, See-Kiong Ng, and Xipeng Qiu. Hermes: Kv cache as hierarchical memory for efficient streaming video understanding. arXiv preprint arXiv:2601.14724, 2026.  
[53] Peiyuan Zhang, Kaichen Zhang, Bo Li, Guangtao Zeng, Jingkang Yang, Yuanhan Zhang, Ziyue Wang, Haoran Tan, Chunyuan Li, and Ziwei Liu. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.  
[54] Yuanhan Zhang, Bo Li, Haotian Liu, Yong Jae Lee, Liangke Gui, Di Fu, Jiashi Feng, Ziwei Liu, and Chunyuan Li. Video instruction tuning with synthetic data. Transactions on Machine Learning Research, 2025.  
[55] Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: Benchmarking multi-task long video understanding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

## A Additional Related Work

Streaming video understanding. Streaming video understanding requires models to answer queries from a causally observed video prefix rather than a fully available offline video. Recent work studies this setting through online video question answering, proactive response timing, streaming-oriented instruction tuning, and real-time interaction [26, 27, 4, 42, 32, 48, 45]. These works address when to respond, how to align perception with generation under causal constraints, and how to operate under bounded computation. SelectStream focuses on the memory side of this problem: preserving, consolidating, and reading historical visual evidence under fixed memory and context budgets.

Memory and context management for video LLMs. A central challenge in streaming video understanding is constructing a bounded working context from an unbounded stream. Existing methods use explicit memory banks or hierarchical event memories [44, 50, 51, 43, 13], KV-cache memory and retrieval [7, 25, 47, 52], visual token pruning or compression [49, 17, 5, 36], and learned recurrent or latent states [26, 27]. SelectStream follows this memory-centric line, but organizes history as a dynamic latent evidence graph: projected VLM visual embeddings are written into memory, consolidated with priority-aware penalties, and read as compact query-conditioned subgraphs before calibrated latent evidence tokens are injected into the frozen VLM.

Streaming benchmarks and recency baselines. Streaming benchmarks evaluate causal online reasoning rather than offline full-video comprehension. OVO-Bench and StreamingBench test observed-only video understanding with both real-time perception and prior-context use [19, 20], while other benchmarks emphasize proactive assistance, turn-taking, or real-world streaming scenarios [15, 38, 37, 30, 35, 23]. Offline long-video benchmarks such as LVBench, MLVU, EgoSchema, and Video-MME evaluate long-range understanding without the same causal constraint [34, 55, 24, 8]. Recent-window baselines further show that additional memory should be evaluated against simple recency baselines and with disaggregated perception and memory metrics [29]. SelectStream is motivated by this standard: current visual evidence remains directly available, while historical memory is exposed only through a compact evidence budget.

## B Additional Method Details and Proofs

This appendix formalizes the implementation details and fixed-budget stability properties of Select-Stream, including the online update algorithm, graph-aware memory consolidation, graph-attention bias terms, training losses, complete proofs of the two theoretical properties, default implementation hyperparameters, and a short complexity discussion.

## B.1 Online Update Algorithm

## B.2 Graph-aware Memory Consolidation

When the active memory size exceeds the budget N, the model frees one slot by merging a pair of active nodes. Let $\mathcal { A } _ { t } \subseteq \dot { \mathcal { M } } _ { t }$ be the active node set. We use the graph-aware merge penalty defined in Eq. (5). This section specifies the normalized component scores, the fixed implementation weights, and metadata updates. The pair with the smallest penalty is merged. Each component is normalized to lie in [0, 1]:

$$
p _ {u v} ^ {\text { sim }} = \frac {1 - \cos (h _ {u} , h _ {v})}{2}, \tag {15}
$$

$$
p _ {u v} ^ {\sup} = \frac {\hat {s} _ {u} + \hat {s} _ {v}}{2}, \tag {16}
$$

$$
p _ {u v} ^ {\mathrm{acc}} = \frac {\hat {r} _ {u} + \hat {r} _ {v}}{2}, \tag {17}
$$

$$
p _ {u v} ^ {\text { rec }} = \frac {\hat {\tau} _ {u} + \hat {\tau} _ {v}}{2}. \tag {18}
$$

The priority term used in the main text is the fixed weighted combination

$$
p _ {u v} ^ {\text { pri }} = \lambda_ {\sup} p _ {u v} ^ {\sup} + \lambda_ {\text { acc }} p _ {u v} ^ {\text { acc }} + \lambda_ {\text { rec }} p _ {u v} ^ {\text { rec }}. \tag {19}
$$

Algorithm 1 Online visual stream processing and memory update  
Require: streaming observations $\{x_{1}, x_{2}, \ldots\}$ , initialized memory graph $G_{0} = (\emptyset, \emptyset)$ Require: memory budget N and fixed SAW/LVM implementation parameters

1: Initialize current segment seg = [] and start time $t_{start} = 1$ 2: for each observation $x_{t}$ do

3: Extract projected visual embedding $g_{t} = \text{Pool}(\Phi_{v}(x_{t}))$ and compute $\bar{s}_{t}$ using Eq. (2)

4: Append cached $g_{t}$ to seg and set boundary flag $b_{t}$ using Eq. (3)

5: if $b_{t}$ then

6: $z_{j} \leftarrow \text{SegEnc}(\text{seg})$ using cached visual features

7: if $\max_{i} \cos(z_{j}, h_{i}) > \tau_{r}$ and $\bar{s}_{j} < \tau_{s}$ then

8: Update target node $i^{*}$ with gated LVM writing in Eq. (4)

9: else

10: Create a new memory node in $G_{t}$ 11: end if

12: Update temporal and similarity edges

13: if $|M_{t}| > N$ then

14: Merge the node pair with minimum penalty $\pi_{uv}$ in Eq. (5)

15: end if

16: Reset seg = [] and $t_{start} = t + 1$ 17: end if

18: end for

Equivalently, the implementation supports the explicit form

$$
\pi_ {u v} = \lambda_ {\text { sim }} p _ {u v} ^ {\text { sim }} + \lambda_ {\text { sup }} p _ {u v} ^ {\text { sup }} + \lambda_ {\text { acc }} p _ {u v} ^ {\text { acc }} + \lambda_ {\text { rec }} p _ {u v} ^ {\text { rec }}, \tag {20}
$$

with $\lambda _ { \mathrm { s i m } } = 1$ in our default setting, which yields the compact main-text form in Eq. (5). Here $\hat { s } _ { i }$ is the normalized accumulated surprise of node $i , \hat { r } _ { i }$ is the normalized read count, and $\hat { \tau } _ { i }$ is a normalized recency score computed from the node’s latest timestamp. A larger $\hat { \tau } _ { i }$ means the node has been updated more recently. In practice, each statistic is divided by the maximum value among currently active nodes, with a small ϵ added for numerical stability.

The interpretation is simple. The similarity term is small when two nodes are semantically redundant. The priority term is large for nodes that are likely to be important: surprising nodes may correspond to event boundaries, frequently read nodes are useful for answering past queries, and recently updated nodes may still be temporally active. Minimizing $\pi _ { u v }$ therefore prefers merging nodes that are similar, unsurprising, rarely read, and not recently updated, while the main method only exposes the priority-preserving consolidation principle.

After choosing the pair $( u , v )$ , the retained node stores a weighted centroid

$$
h _ {u v} = \text { Norm } \left(\frac {w _ {u} h _ {u} + w _ {v} h _ {v}}{w _ {u} + w _ {v}}\right), \quad w _ {i} = \max (c _ {i}, 1), \tag {21}
$$

where $c _ { i }$ is the write count and Norm(·) optionally normalizes the state to unit length. The retained metadata is updated by

$$
t _ {u v} ^ {\text { start }} = \min (t _ {u} ^ {\text { start }}, t _ {v} ^ {\text { start }}), \tag {22}
$$

$$
t _ {u v} ^ {\text { end }} = \max (t _ {u} ^ {\text { end }}, t _ {v} ^ {\text { end }}), \tag {23}
$$

$$
\bar {s} _ {u v} = \frac {w _ {u} \bar {s} _ {u} + w _ {v} \bar {s} _ {v}}{w _ {u} + w _ {v}}, \tag {24}
$$

$$
c _ {u v} = c _ {u} + c _ {v}, \tag {25}
$$

$$
r _ {u v} ^ {\text { read }} = r _ {u} ^ {\text { read }} + r _ {v} ^ {\text { read }}, \tag {26}
$$

$$
m _ {u v} ^ {\text { merge }} = m _ {u} ^ {\text { merge }} + m _ {v} ^ {\text { merge }} + 1. \tag {27}
$$

Temporal edges incident to either merged node are inherited by the retained node using max aggregation, and similarity edges are recomputed from the new latent state. The released slot is cleared and becomes available for future writes.

## B.3 Priority-weighted Consolidation Error

The consolidation rule should not be interpreted as a globally optimal compression algorithm. Its role is more modest: the similarity term controls feature distortion, while the priority term discourages merging important nodes. The following lemma makes the feature part explicit.

Lemma 1 (Two-node centroid distortion). Assume $h _ { u }$ and $h _ { v }$ are ℓ2-normalized and let $w _ { u } , w _ { v } > 0$ . If two nodes are replaced by their weighted centroid

$$
\mu_ {u v} = \frac {w _ {u} h _ {u} + w _ {v} h _ {v}}{w _ {u} + w _ {v}}, \tag {28}
$$

then the weighted feature distortion introduced before optional renormalization is

$$
w _ {u} \| h _ {u} - \mu_ {u v} \| _ {2} ^ {2} + w _ {v} \| h _ {v} - \mu_ {u v} \| _ {2} ^ {2} = \frac {w _ {u} w _ {v}}{w _ {u} + w _ {v}} \| h _ {u} - h _ {v} \| _ {2} ^ {2} = \frac {2 w _ {u} w _ {v}}{w _ {u} + w _ {v}} (1 - \cos (h _ {u}, h _ {v})). \tag {29}
$$

Proof. Let $W = w _ { u } + w _ { v }$ . Since

$$
h _ {u} - \mu_ {u v} = \frac {w _ {v}}{W} (h _ {u} - h _ {v}), \quad h _ {v} - \mu_ {u v} = - \frac {w _ {u}}{W} (h _ {u} - h _ {v}), \tag {30}
$$

we have

$$
w _ {u} \| h _ {u} - \mu_ {u v} \| _ {2} ^ {2} + w _ {v} \| h _ {v} - \mu_ {u v} \| _ {2} ^ {2} = w _ {u} \frac {w _ {v} ^ {2}}{W ^ {2}} \| h _ {u} - h _ {v} \| _ {2} ^ {2} + w _ {v} \frac {w _ {u} ^ {2}}{W ^ {2}} \| h _ {u} - h _ {v} \| _ {2} ^ {2} \tag {31}
$$

$$
= \frac {w _ {u} w _ {v} (w _ {u} + w _ {v})}{W ^ {2}} \| h _ {u} - h _ {v} \| _ {2} ^ {2} \tag {32}
$$

$$
= \frac {w _ {u} w _ {v}}{w _ {u} + w _ {v}} \| h _ {u} - h _ {v} \| _ {2} ^ {2}. \tag {33}
$$

Because $\| h _ { u } \| _ { 2 } = \| h _ { v } \| _ { 2 } = 1$ ,

$$
\left\| h _ {u} - h _ {v} \right\| _ {2} ^ {2} = \left\| h _ {u} \right\| _ {2} ^ {2} + \left\| h _ {v} \right\| _ {2} ^ {2} - 2 h _ {u} ^ {\top} h _ {v} = 2 (1 - \cos (h _ {u}, h _ {v})). \tag {34}
$$

This proves the claim.

$p _ { u v } ^ { \mathrm { s i m } }$ similarity directly controls the feature distortion of centroid merging. The priority term does not measure feature distortion. Instead, it makes it more costly to merge nodes that are surprising, frequently accessed, or recently updated.

## B.4 Graph Attention Bias Details

For the $\mathrm { G A R }$ attention logit in Eq. $( 8 ) , b _ { \mathrm { t y p e } }$ is a learned scalar bias for the edge type (temporal or similarity), $b _ { \Delta t }$ is a temporal-gap penalty with a learnable positive scale, and $b _ { w }$ is an additive edge-support term. For nodes with temporal spans $[ t _ { i } ^ { s } , t _ { i } ^ { e } ]$ ] and $[ t _ { j } ^ { s } , t _ { j } ^ { e } ]$ , we use

$$
b _ {\Delta t} = - \frac {\Delta t _ {i j}}{\tau_ {t}}, \quad \Delta t _ {i j} = \max (t _ {i} ^ {s} - t _ {j} ^ {e}, t _ {j} ^ {s} - t _ {i} ^ {e}, 0), \quad \tau_ {t} = \text { softplus } (\hat {\tau} _ {t}) + \epsilon . \tag {35}
$$

The edge-support term is $b _ { w } = w _ { i j }$ , which biases attention toward stronger temporal-continuity or semantic-similarity edges. For routing, edge supports are normalized to $[ 0 , 1 ]$ : similarity edges use $( 1 + \cos ( h _ { i } , h _ { j } ) ) / 2$ , while temporal edges are already non-negative.

## B.5 Training Loss Details

Retrieval supervision trains the evidence-allocation policy when evidence annotations are available. Let $\mathcal { P }$ be positive memory nodes: nodes whose spans overlap annotated evidence intervals, or nodes whose IDs match explicit evidence annotations. With retrieval distribution ${ p } _ { i } ^ { \mathrm { r e t } }$ = softmaxi(scorei) over active or retrieved candidates, we use a multi-positive loss

$$
\mathcal {L} _ {\text { ret }} = - \log \sum_ {i \in \mathcal {P}} p _ {i} ^ {\text { ret }}, \tag {36}
$$

when $\mathcal { P }$ is nonempty. Other candidates from the same video act as negatives through the softmax denominator.

The sparsity term penalizes diffuse routing probabilities and redundant evidence:

$$
\mathcal {L} _ {\text { spar }} = \frac {1}{B} \sum_ {i \in \mathcal {A} _ {q}} p _ {i} ^ {\text { route }} + \frac {1}{M (M - 1)} \sum_ {m \neq n} \max (0, \cos (e _ {m}, e _ {n}) - \delta), \tag {37}
$$

where $A _ { q }$ denotes the scored candidate nodes before hard budget pruning, and $\delta$ is a fixed redundancy margin. For the differentiable routing-size proxy, we use

$$
p _ {i} ^ {\text { route }} = \sigma \left(\frac {\text { score } _ {i} - \kappa}{\tau_ {\text { route }}}\right), \tag {38}
$$

where κ and $\tau _ { \mathrm { r o u t e } }$ are fixed threshold and temperature constants. The hard top-k/budgeted routing rule is still used to construct the retrieved subgraph at inference time.

## B.6 Proof of Proposition 1

Let $S _ { T }$ be the number of SAW segments generated before consolidation over a stream of length $T _ { \ast }$ counting the final open segment if it is flushed at the end. Let C be the set of segment closing times up to time T . Each closing time is assigned to one active trigger in Eq. (3). If multiple triggers hold simultaneously, we break ties using any fixed order; this only makes the following partition more conservative. Let $\mathcal { C } _ { B } , \mathcal { C } _ { H }$ , and $\mathcal { C } _ { L }$ denote closures assigned to accumulated surprise, instantaneous high surprise, and maximum length, respectively.

For a closure assigned to accumulated surprise, the corresponding segment interval $I _ { j }$ satisfies

$$
\sum_ {t \in I _ {j}} \bar {s} _ {t} > B _ {s}. \tag {39}
$$

The intervals of different closed segments are disjoint. Therefore

$$
\left| \mathcal {C} _ {B} \right| B _ {s} \leq \sum_ {j: t _ {j} \in \mathcal {C} _ {B}} \sum_ {t \in I _ {j}} \bar {s} _ {t} \leq \sum_ {t = 1} ^ {T} \bar {s} _ {t}, \tag {40}
$$

which gives

$$
\left| \mathcal {C} _ {B} \right| \leq \frac {\sum_ {t = 1} ^ {T} \bar {s} _ {t}}{B _ {s}}. \tag {41}
$$

For a closure assigned to the high-surprise trigger, the assumption $\theta _ { \mathrm { h i g h } } ( t ) \geq \theta _ { \mathrm { m i n } } > 0$ implies

$$
\bar {s} _ {t} > \theta_ {\text { high }} (t) \geq \theta_ {\min}. \tag {42}
$$

Since closing times are distinct,

$$
\left| \mathcal {C} _ {H} \right| \theta_ {\min} \leq \sum_ {t \in \mathcal {C} _ {H}} \bar {s} _ {t} \leq \sum_ {t = 1} ^ {T} \bar {s} _ {t}, \tag {43}
$$

and hence

$$
\left| \mathcal {C} _ {H} \right| \leq \frac {\sum_ {t = 1} ^ {T} \bar {s} _ {t}}{\theta_ {\min}}. \tag {44}
$$

For a closure assigned to the maximum-length trigger, the corresponding segment contains at least $L _ { \mathrm { m a x } }$ time steps under the segment-length convention used by Eq. (3). Since such segments are disjoint,

$$
\left| \mathcal {C} _ {L} \right| L _ {\max} \leq T, \quad \left| \mathcal {C} _ {L} \right| \leq \frac {T}{L _ {\max}}. \tag {45}
$$

Adding the three disjoint closure classes and allowing one unfinished open segment at the end gives

$$
S _ {T} \leq \frac {\sum_ {t = 1} ^ {T} \bar {s} _ {t}}{B _ {s}} + \frac {\sum_ {t = 1} ^ {T} \bar {s} _ {t}}{\theta_ {\min}} + \frac {T}{L _ {\max}} + 1. \tag {46}
$$

It remains to prove the active memory bound. Initially, $| { \mathcal { M } } _ { 0 } | = 0 \leq N .$ . Suppose $| \mathcal { M } _ { t - 1 } | \leq N$ before a segment write. The write either updates an existing node, in which case the size does not increase, or creates one new node, in which case the size can become at most $N + 1$ . If it exceeds N , the consolidation step merges one pair of active nodes and releases one slot, returning the size to at most N . By induction, $| \mathcal { M } _ { t } | \le N$ for all $t ,$ and in particular $| \mathcal { M } _ { T } | \leq N$ .

## B.7 Proof of Proposition 2

Let I be the event that the relevant evidence node $h ^ { * }$ is included in the retrieved candidate set, with $\Pr ( I ) = r$ . If I does not hold, we set the attention mass assigned to $h ^ { * }$ to zero. Conditioned on $I ,$ let $\ell _ { * }$ be the logit of $h ^ { * }$ and let $\ell _ { 1 } , \ldots , \ell _ { M - 1 }$ be the logits of the irrelevant candidates, where $\mathbb { E } [ \exp ( \ell _ { j } ) \mid I ] \leq \exp ( \sigma ^ { 2 } / 2 )$ for each distractor. Define

$$
a = \exp (\mu_ {\mathrm{sig}}), \quad X = \sum_ {j = 1} ^ {M - 1} \exp (\ell_ {j}). \tag {47}
$$

By assumption, $\ell _ { * } \geq \mu _ { \mathrm { s i g } } , \mathrm { s o } \exp ( \ell _ { * } ) \geq a$ . The softmax weight on $h ^ { * }$ satisfies

$$
\alpha_ {h ^ {*}} = \frac {\exp (\ell_ {*})}{\exp (\ell_ {*}) + X} \geq \frac {a}{a + X}. \tag {48}
$$

The function $f ( X ) = a / ( a + X )$ is convex for $X \geq 0$ because $f ^ { \prime \prime } ( X ) = 2 a / ( a + X ) ^ { 3 } > 0 ,$ . By Jensen’s inequality,

$$
\mathbb {E} \left[ \frac {a}{a + X} \mid I \right] \geq \frac {a}{a + \mathbb {E} [ X \mid I ]}. \tag {49}
$$

Using the exponential moment bound for each irrelevant logit,

$$
\mathbb {E} [ X \mid I ] = \sum_ {j = 1} ^ {M - 1} \mathbb {E} [ \exp (\ell_ {j}) \mid I ] \leq (M - 1) \exp (\sigma^ {2} / 2). \tag {50}
$$

Therefore

$$
\mathbb {E} [ \alpha_ {h ^ {*}} \mid I ] \geq \frac {\exp (\mu_ {\mathrm{sig}})}{\exp (\mu_ {\mathrm{sig}}) + (M - 1) \exp (\sigma^ {2} / 2)}. \tag {51}
$$

Multiplying by Pr(I) = r gives

$$
\mathbb {E} [ \alpha_ {h ^ {*}} ] \geq r \cdot \frac {\exp (\mu_ {\mathrm{sig}})}{\exp (\mu_ {\mathrm{sig}}) + (M - 1) \exp (\sigma^ {2} / 2)}. \tag {52}
$$

This proves the proposition. The bound highlights the role of retrieval: if attention were computed over the full history, the same argument would replace M with the number of historical candidates, increasing the distractor term in the denominator.

## C More Implementation Details

Training setup. We keep the backbone VLM frozen and train only the SelectStream modules, including the segment encoder, write gate, query encoder, GAR module, and evidence calibration layer. For both Qwen2.5-VL-7B and Qwen3-VL-8B, we use bfloat16 precision and train separate backbone-specific modules on 8 NVIDIA A100 80GB GPUs. Unless otherwise stated, we use AdamW with learning rate $2 \times 1 0 ^ { - 4 }$ , batch size 1, gradient accumulation $^ { 4 , }$ and one training epoch. Timestamped examples supervise retrieval with both node-ID and temporal-overlap targets; answer-only examples use only the autoregressive answer loss.

Default implementation hyperparameters. The main method exposes the architectural budgets N , B, and M because they directly control memory capacity, retrieval scope, and generation context. All other coefficients are treated as fixed implementation hyperparameters. Unless otherwise stated, we use the same values across datasets and backbones.

## D Benchmarks and Metrics

Evaluation protocol. All online evaluations follow a causal streaming protocol: at a query time, the model can only use the observed video prefix and cannot access future frames. SelectStream processes frames sequentially, updates its latent memory online, and answers with the current observation plus retrieved latent evidence. Unless otherwise stated, SelectStream uses 1 fps input, fixed memory capacity N, retrieved subgraph budget $B ,$ and injected evidence budget M. For baselines, we use the reported frame rates and context budgets from the corresponding sources when exact budget matching is not possible.

Table 6: Default SelectStream training and implementation hyperparameters.

<table><tr><td>Hyperparameter</td><td>Default value</td></tr><tr><td>optimizer</td><td>AdamW</td></tr><tr><td>learning rate</td><td> $2 \times 10^{-5}$ </td></tr><tr><td>random seed</td><td>42</td></tr><tr><td>Nmemory slots</td><td>256</td></tr><tr><td>Bretrieved subgraph budget</td><td>64</td></tr><tr><td>Minjected evidence tokens</td><td>8</td></tr><tr><td>λattention-surprise weight</td><td>0.5</td></tr><tr><td>ρsurprise EMA decay</td><td>0.9</td></tr><tr><td> $L_{\text{min}}, L_{\text{max}}$ </td><td>8, 64</td></tr><tr><td> $B_s$ surprise-energy budget</td><td>8.0</td></tr><tr><td>adaptive threshold quantile</td><td>0.9</td></tr><tr><td>SAW recent window</td><td>64</td></tr><tr><td>attention bins</td><td>32</td></tr><tr><td> $\tau_r, \tau_s$ </td><td>0.75, 0.35</td></tr><tr><td> $\lambda_{\text{sim}}, \lambda_{\text{sup}}, \lambda_{\text{acc}}, \lambda_{\text{rec}}$ </td><td>1.0, 0.5, 0.25, 0.25</td></tr><tr><td>top-kseeds</td><td>16</td></tr><tr><td>GAR layers K</td><td>2</td></tr><tr><td> $\eta, \xi_\ell, \xi_m$ </td><td>0.2, 0.05, 0.05</td></tr><tr><td> $\alpha_r$ routing edge weight</td><td>0.1</td></tr><tr><td>routing threshold  $\kappa$ </td><td>0.0</td></tr><tr><td>routing temperature  $\tau_{\text{route}}$ </td><td>1.0</td></tr><tr><td>redundancy margin  $\delta$ </td><td>0.0</td></tr><tr><td> $\beta, \gamma$ loss weights</td><td>0.1, 0.05</td></tr><tr><td>ID/time retrieval weights</td><td>1.0, 1.0</td></tr><tr><td>timestamp tolerance</td><td>0.0</td></tr><tr><td>evidence top-k</td><td>8</td></tr></table>

Online streaming benchmarks. StreamingBench evaluates observed-only video understanding across perception, reasoning, and temporal-awareness tasks. We report the official overall accuracy, denoted as StreamingBench or SB in compact tables. OVO-Bench separates online video understanding into Real-Time Visual Perception (RT), Backward Tracing (BT), and Forward Active Responding (FAR). RT measures current-scene perception, BT measures reasoning over previously observed visual context, and FAR measures whether the model responds at appropriate future-oriented moments. We report RT Avg., BT Avg., their mean RT/BT Avg., and FAR / Overall when available:

$$
\mathrm{RT/BTAvg.} = \frac {\mathrm{RTAvg.} + \mathrm{BTAvg.}}{2}. \tag {53}
$$

In ablation and compact analysis tables, OVO denotes OVO-Bench Overall.

Offline video benchmarks. We evaluate offline generalization on VideoMME, MLVU, and MVBench with a causal final-query protocol. The model observes the video stream sequentially and answers only after the final observation, so the setting still uses SelectStream’s online memory construction rather than full-video replay. VideoMME and MVBench are reported with their official accuracy metrics, while MLVU is reported with its official M-Avg score. The offline average is computed over the three dataset-level scores:

$$
\text { Avg. } = \frac {\text { VideoMME } + \text { MLVU } + \text { MVBench }}{3}. \tag {54}
$$

Entries marked “–” indicate unavailable results from the corresponding source and are excluded from averages unless all three scores are reported.

Ablation and latent-evidence metrics. Ablations use SelectStream-Qwen2.5-VL-7B with the same frame rate, frozen backbone, memory capacity, and evidence budget as the full model. We report StreamingBench accuracy, OVO-Bench Overall, and MLVU M-Avg. For latent evidence analysis, Recall@M measures whether at least one of the top-M retrieved memory nodes overlaps the annotated evidence interval, and T-Overlap measures the best fraction of the annotated evidence interval covered by a retrieved node. These metrics evaluate whether the injected latent evidence is temporally grounded and empirically estimate the retrieval-recall condition used by Proposition 2 (section 3.5).

Efficiency metrics. For efficiency and latency, we report time to first token (TTFT) and peak GPU memory as the number of observed frames increases. TTFT includes query-time memory scoring, subgraph routing, GAR re-ranking, evidence calibration, and the first decoding step. Peak GPU memory is measured during streaming inference and includes the active memory graph and the retrieved evidence used for generation. We also track active memory size and injected evidence tokens to verify that query-time cost is controlled by N , B, and M rather than by the full stream length.

## E Additional Latent Evidence Analysis

Each injected evidence token in SelectStream is traceable to a retrieved memory node. For training examples with evidence timestamps, we mark a memory node as positive if its temporal span overlaps the annotated evidence interval with a small tolerance:

$$
\mathcal {P} (q) = \{i \mid [ t _ {i} ^ {s}, t _ {i} ^ {e} ] \cap [ t _ {q} ^ {s} - \epsilon , t _ {q} ^ {e} + \epsilon ] \neq \emptyset \}. \tag {55}
$$

We use overlap rather than IoU because memory nodes may represent variable-length or merged segments, and overly strict IoU would penalize coarse nodes that still contain the evidence.

For evaluation, let $R _ { q } ^ { M }$ be the top-M retrieved nodes for query q and let $\mathcal { G } _ { q }$ be the set of annotated evidence intervals. Since each memory node i stores a temporal span $I _ { i } = [ t _ { i } ^ { s } , t _ { i } ^ { e } ]$ , we measure whether the retrieved latent evidence is temporally grounded by

$$
\text { Recall@ } M = \frac {1}{| \mathcal {Q} |} \sum_ {q \in \mathcal {Q}} \mathbf {1} \left[ \max _ {i \in R _ {q} ^ {M}, g \in \mathcal {G} _ {q}} \frac {| I _ {i} \cap g |}{| g |} > 0 \right]. \tag {56}
$$

We also report average temporal overlap,

$$
\text { T   -   O   v   e   r   l   a   p } = \frac {1}{| \mathcal {Q} |} \sum_ {q \in \mathcal {Q}} \max _ {i \in R _ {q} ^ {M}, g \in \mathcal {G} _ {q}} \frac {\left| I _ {i} \cap g \right|}{| g |}. \tag {57}
$$

The overlap denominator is the ground-truth evidence length rather than the union length. This is intentional: a merged memory node may cover a longer span while still containing the annotated evidence, and IoU would over-penalize such coarse but valid memory. Empirically, Recall@M estimates the retrieval-recall term r used by Proposition 2 in Section 3.5, while T-Overlap measures how tightly the retrieved node spans cover the annotated evidence.

Table 7 isolates temporal grounding under different retrieval variants while keeping the evidence budget fixed. In addition to Recall@M and T-Overlap, we report the median retrieved span length to verify that higher recall does not come simply from selecting overly long memory nodes.

Table 7: Additional temporal grounding analysis of retrieved latent evidence on timestamped examples. Median span length is measured in seconds.

<table><tr><td>Retrieval Policy</td><td>Recall@M</td><td>T-Overlap</td><td>Median Span</td></tr><tr><td>Initial top-k seeds only</td><td>0.49</td><td>0.33</td><td>7.8s</td></tr><tr><td>Top-k + fixed-hop expansion</td><td>0.61</td><td>0.43</td><td>10.6s</td></tr><tr><td>Top-k + query-conditioned routing</td><td>0.69</td><td>0.50</td><td>9.4s</td></tr><tr><td>Query routing + GAR re-ranking</td><td>0.71</td><td>0.52</td><td>9.1s</td></tr><tr><td>Full SelectStream</td><td>0.72</td><td>0.53</td><td>9.0s</td></tr></table>

For qualitative inspection, we also visualize retrieved evidence on the original video timeline. The figure should show the annotated evidence intervals, active memory-node spans, top-M retrieved nodes, and retrieval scores. This verifies that the latent tokens injected into the VLM can be traced back to concrete video regions.

## F Complexity Under Fixed Budgets

Let d be the memory dimension, N the active memory budget, B the retrieved subgraph budget, M the evidence budget, and K the number of GAR layers. Ignoring the frozen visual pass and native multimodal projection cost, SAW maintains constant-size streaming statistics and adds $O ( d )$ work per observation for lightweight feature comparison. When a segment is closed, LVM writing reuses cached projected visual embeddings and compares the segment vector with active memory nodes, requiring $O ( N d )$ time. Naive graph-aware consolidation computes all pair penalties in $O ( N ^ { \bar { 2 } } d )$ time when the memory is full, but this cost is controlled by the fixed budget N and can be reduced with approximate nearest-neighbor candidates or top-k similarity neighborhoods.

At query time, retrieval scores all active nodes in O(N d) time and then routes the query through a bounded subgraph of size at most B. GAR reasoning over the retrieved subgraph costs $O ( K ( | E _ { B } | d +$ $B d ^ { 2 } ) ,$ ), where $E _ { B }$ is the edge set inside the retrieved subgraph. Finally, only M evidence embeddings are injected into the language model. Therefore, after the stream has been compressed into memory, query-time reasoning depends on N, B, and M, rather than on the full video length T .

## G Full Results on OVO-Bench and StreamingBench

Table 8: Full StreamingBench results. OP, CR, CS, ATP, EU, TR, PR, SU, ACP, and CT denote the official StreamingBench sub-tasks.

<table><tr><td>Model</td><td>Size</td><td>#Frames</td><td>OP</td><td>CR</td><td>CS</td><td>ATP</td><td>EU</td><td>TR</td><td>PR</td><td>SU</td><td>ACP</td><td>CT</td><td>All</td></tr><tr><td>Human</td><td>-</td><td>-</td><td>89.47</td><td>92.00</td><td>93.60</td><td>91.47</td><td>95.65</td><td>92.52</td><td>88.00</td><td>88.75</td><td>89.74</td><td>91.30</td><td>91.46</td></tr><tr><td colspan="14">Proprietary MLLMs</td></tr><tr><td>Gemini 1.5 Pro</td><td>-</td><td>1 fps</td><td>79.02</td><td>80.47</td><td>83.54</td><td>79.67</td><td>80.00</td><td>84.74</td><td>77.78</td><td>64.23</td><td>71.95</td><td>48.70</td><td>75.69</td></tr><tr><td>GPT-4o</td><td>-</td><td>64</td><td>77.11</td><td>80.47</td><td>83.91</td><td>76.47</td><td>70.19</td><td>83.80</td><td>66.67</td><td>62.19</td><td>69.12</td><td>49.22</td><td>73.28</td></tr><tr><td>Claude 3.5 Sonnet</td><td>-</td><td>20</td><td>73.33</td><td>80.47</td><td>84.09</td><td>82.02</td><td>75.39</td><td>79.53</td><td>61.11</td><td>61.79</td><td>69.32</td><td>43.09</td><td>72.44</td></tr><tr><td colspan="14">Offline Video LLMs</td></tr><tr><td>Video-LLaMA2</td><td>7B</td><td>32</td><td>55.86</td><td>55.47</td><td>57.41</td><td>58.17</td><td>52.80</td><td>43.61</td><td>39.81</td><td>42.68</td><td>45.61</td><td>35.23</td><td>49.52</td></tr><tr><td>VILA-1.5</td><td>8B</td><td>14</td><td>53.68</td><td>49.22</td><td>70.98</td><td>56.86</td><td>53.42</td><td>53.89</td><td>54.63</td><td>48.78</td><td>50.14</td><td>17.62</td><td>52.32</td></tr><tr><td>Video-CCAM</td><td>14B</td><td>96</td><td>56.40</td><td>57.81</td><td>65.30</td><td>62.75</td><td>64.60</td><td>51.40</td><td>42.59</td><td>47.97</td><td>49.58</td><td>31.61</td><td>53.96</td></tr><tr><td>LongVA</td><td>7B</td><td>128</td><td>70.03</td><td>63.28</td><td>61.20</td><td>70.92</td><td>62.73</td><td>59.50</td><td>61.11</td><td>53.66</td><td>54.67</td><td>34.72</td><td>59.96</td></tr><tr><td>InternVL2</td><td>8B</td><td>16</td><td>68.12</td><td>60.94</td><td>69.40</td><td>77.12</td><td>67.70</td><td>62.93</td><td>59.26</td><td>53.25</td><td>54.96</td><td>56.48</td><td>63.72</td></tr><tr><td>Kangaroo</td><td>7B</td><td>64</td><td>71.12</td><td>84.38</td><td>70.66</td><td>73.20</td><td>67.08</td><td>61.68</td><td>56.48</td><td>55.69</td><td>62.04</td><td>38.86</td><td>64.60</td></tr><tr><td>LLaVA-NeXT-Video</td><td>32B</td><td>64</td><td>78.20</td><td>70.31</td><td>73.82</td><td>76.80</td><td>63.35</td><td>69.78</td><td>57.41</td><td>56.10</td><td>64.31</td><td>38.86</td><td>66.96</td></tr><tr><td>MiniCPM-V2.6</td><td>8B</td><td>32</td><td>71.93</td><td>71.09</td><td>77.92</td><td>75.82</td><td>64.60</td><td>65.73</td><td>70.37</td><td>56.10</td><td>62.32</td><td>53.37</td><td>67.44</td></tr><tr><td>LLaVA-OneVision</td><td>7B</td><td>32</td><td>80.38</td><td>74.22</td><td>76.03</td><td>80.72</td><td>72.67</td><td>71.65</td><td>67.59</td><td>65.45</td><td>65.72</td><td>45.08</td><td>71.12</td></tr><tr><td>Qwen2.5-VL</td><td>7B</td><td>1 fps</td><td>78.32</td><td>80.47</td><td>78.86</td><td>80.45</td><td>76.73</td><td>78.50</td><td>79.63</td><td>63.41</td><td>66.19</td><td>53.19</td><td>73.68</td></tr><tr><td colspan="14">Online / Streaming Video LLMs</td></tr><tr><td>Flash-VStream</td><td>7B</td><td>-</td><td>25.89</td><td>43.57</td><td>24.91</td><td>23.87</td><td>27.33</td><td>13.08</td><td>18.52</td><td>25.20</td><td>23.87</td><td>48.70</td><td>23.23</td></tr><tr><td>VideoLLM-online</td><td>8B</td><td>2 fps</td><td>39.07</td><td>40.06</td><td>34.49</td><td>31.05</td><td>45.96</td><td>32.40</td><td>31.48</td><td>34.16</td><td>42.49</td><td>27.89</td><td>35.99</td></tr><tr><td>Dispider</td><td>7B</td><td>1 fps</td><td>74.92</td><td>75.53</td><td>74.10</td><td>73.08</td><td>74.44</td><td>59.92</td><td>76.14</td><td>62.91</td><td>62.16</td><td>45.80</td><td>67.63</td></tr><tr><td>StreamForest</td><td>7B</td><td>1 fps</td><td>83.11</td><td>82.81</td><td>82.65</td><td>84.26</td><td>77.50</td><td>78.19</td><td>76.85</td><td>69.11</td><td>75.64</td><td>54.40</td><td>77.26</td></tr><tr><td>Qwen2.5-VL-7B + 4f</td><td>7B</td><td>1fps</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>78.47</td></tr><tr><td>Qwen3-VL-8B + 4f</td><td>8B</td><td>1fps</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>80.59</td></tr><tr><td colspan="14">SelectStream Framework</td></tr><tr><td>SELECTSTREAM-Qwen2.5-VL-7B</td><td>7B</td><td>1 fps</td><td>84.50</td><td>86.20</td><td>84.80</td><td>86.00</td><td>83.20</td><td>83.40</td><td>84.60</td><td>73.80</td><td>76.70</td><td>70.99</td><td>81.42</td></tr><tr><td>SELECTSTREAM-Qwen3-VL-8B</td><td>8B</td><td>1 fps</td><td>86.30</td><td>87.40</td><td>86.00</td><td>87.10</td><td>84.70</td><td>84.60</td><td>85.80</td><td>75.00</td><td>78.20</td><td>71.60</td><td>82.67</td></tr></table>

## H Limitations

SelectStream is designed as a general memory layer for frozen video-language backbones, and this paper evaluates it mainly on 7B–8B models and standard streaming or long-video benchmarks. Broader studies with larger backbones, different visual tokenizers, and more interactive deployment settings would further clarify how the same memory design scales across model families. In addition, our experiments use fixed default budgets for most comparisons to keep the evaluation controlled; adapting N, B, and M dynamically to device constraints or application latency targets is a useful direction for future work. The current evaluation focuses on answer accuracy, temporal evidence grounding, latency, and memory usage. Future evaluations could include richer human-facing criteria, such as response helpfulness in multi-turn streaming interaction or user preference under real-time constraints. These extensions are complementary to the proposed latent evidence allocation framework and do not change the fixed-budget memory formulation studied in this work.

Table 9: Detailed evaluation results on OVO-Bench.

<table><tr><td rowspan="2">Model</td><td rowspan="2"># Frames</td><td colspan="7">Real-Time Visual Perception</td><td colspan="4">Backward Tracing</td><td colspan="4">Forward Active Responding</td><td rowspan="2">Overall Avg.</td></tr><tr><td>OCR</td><td>ACR</td><td>ATR</td><td>STU</td><td>FPD</td><td>OJR</td><td>Avg.</td><td>EPM</td><td>ASI</td><td>HLD</td><td>Avg.</td><td>REC</td><td>SSR</td><td>CRR</td><td>Avg.</td></tr><tr><td>Human</td><td>-</td><td>93.96</td><td>92.57</td><td>94.83</td><td>92.70</td><td>91.09</td><td>94.02</td><td>93.20</td><td>92.59</td><td>93.02</td><td>91.37</td><td>92.33</td><td>95.48</td><td>89.67</td><td>93.56</td><td>92.90</td><td>92.81</td></tr><tr><td colspan="18">Proprietary MLLMs</td></tr><tr><td>Gemini 1.5 Pro</td><td>1 fps</td><td>85.91</td><td>66.97</td><td>79.31</td><td>58.43</td><td>63.37</td><td>61.96</td><td>69.32</td><td>58.59</td><td>76.35</td><td>52.64</td><td>62.54</td><td>35.53</td><td>74.24</td><td>61.67</td><td>57.15</td><td>63.00</td></tr><tr><td>GPT-4o</td><td>64</td><td>69.80</td><td>64.22</td><td>71.55</td><td>51.12</td><td>70.30</td><td>59.78</td><td>64.46</td><td>57.91</td><td>75.68</td><td>48.66</td><td>60.75</td><td>27.58</td><td>73.21</td><td>59.40</td><td>53.40</td><td>59.54</td></tr><tr><td colspan="18">Open-source Offline Video MLLMs</td></tr><tr><td>Qwen2-VL-72B</td><td>64</td><td>65.77</td><td>60.55</td><td>69.83</td><td>51.69</td><td>69.31</td><td>54.35</td><td>61.92</td><td>52.53</td><td>60.81</td><td>57.53</td><td>56.95</td><td>38.83</td><td>64.07</td><td>45.00</td><td>49.30</td><td>56.27</td></tr><tr><td>LLaVA-Video-7B</td><td>64</td><td>69.13</td><td>58.72</td><td>68.83</td><td>49.44</td><td>74.26</td><td>59.78</td><td>63.52</td><td>56.23</td><td>57.43</td><td>7.53</td><td>40.40</td><td>34.10</td><td>69.95</td><td>60.42</td><td>54.82</td><td>52.91</td></tr><tr><td>LLaVA-OneVision-7B</td><td>64</td><td>66.44</td><td>57.80</td><td>73.28</td><td>53.37</td><td>71.29</td><td>61.96</td><td>64.02</td><td>54.21</td><td>55.41</td><td>21.51</td><td>43.71</td><td>25.64</td><td>67.09</td><td>58.75</td><td>50.50</td><td>52.74</td></tr><tr><td>Qwen2-VL-7B</td><td>64</td><td>60.40</td><td>50.46</td><td>56.03</td><td>47.19</td><td>66.34</td><td>55.43</td><td>55.98</td><td>47.81</td><td>35.48</td><td>56.08</td><td>46.46</td><td>31.66</td><td>65.82</td><td>48.75</td><td>48.74</td><td>50.39</td></tr><tr><td>InternVL2-8B</td><td>64</td><td>67.11</td><td>60.55</td><td>63.79</td><td>46.07</td><td>68.32</td><td>56.52</td><td>60.39</td><td>48.15</td><td>57.43</td><td>24.73</td><td>43.44</td><td>26.50</td><td>59.14</td><td>54.14</td><td>46.60</td><td>50.15</td></tr><tr><td>LongVU-7B</td><td>1 fps</td><td>53.69</td><td>53.21</td><td>62.93</td><td>47.75</td><td>68.32</td><td>59.78</td><td>57.61</td><td>40.74</td><td>59.46</td><td>4.84</td><td>35.01</td><td>12.18</td><td>69.48</td><td>60.83</td><td>47.50</td><td>46.71</td></tr><tr><td colspan="18">Open-source Online Video MLLMs</td></tr><tr><td>VideoLLM-online-8B</td><td>2 fps</td><td>8.05</td><td>23.85</td><td>12.07</td><td>14.04</td><td>45.54</td><td>21.20</td><td>20.79</td><td>22.22</td><td>18.80</td><td>12.18</td><td>17.73</td><td>-</td><td>-</td><td>-</td><td>-</td><td>12.84</td></tr><tr><td>Flash-VStream-7B</td><td>1 fps</td><td>24.16</td><td>29.36</td><td>28.45</td><td>33.71</td><td>25.74</td><td>28.80</td><td>28.37</td><td>39.06</td><td>37.16</td><td>5.91</td><td>27.38</td><td>8.02</td><td>67.25</td><td>60.00</td><td>45.09</td><td>33.61</td></tr><tr><td>Dispider-7B</td><td>1 fps</td><td>57.72</td><td>49.54</td><td>62.07</td><td>44.94</td><td>61.39</td><td>51.63</td><td>54.55</td><td>48.48</td><td>55.41</td><td>4.30</td><td>36.06</td><td>18.05</td><td>37.36</td><td>48.75</td><td>34.72</td><td>41.78</td></tr><tr><td>ViSpeak-7B</td><td>1 fps</td><td>75.17</td><td>58.72</td><td>71.55</td><td>51.12</td><td>74.26</td><td>66.85</td><td>66.28</td><td>59.93</td><td>48.65</td><td>63.98</td><td>57.52</td><td>33.81</td><td>68.52</td><td>60.42</td><td>54.25</td><td>61.08</td></tr><tr><td>StreamForest-7B</td><td>1 fps</td><td>68.46</td><td>53.21</td><td>71.55</td><td>47.75</td><td>65.35</td><td>60.87</td><td>61.20</td><td>58.92</td><td>64.86</td><td>32.26</td><td>52.02</td><td>32.81</td><td>70.59</td><td>57.08</td><td>53.49</td><td>55.57</td></tr><tr><td>ET-Instruct-3B</td><td>1 fps</td><td>65.10</td><td>35.78</td><td>56.90</td><td>35.39</td><td>24.75</td><td>60.87</td><td>46.47</td><td>41.81</td><td>35.14</td><td>8.60</td><td>28.52</td><td>20.06</td><td>52.31</td><td>67.50</td><td>46.62</td><td>40.54</td></tr><tr><td>ET-Instruct-3B $^†$ </td><td>1 fps</td><td>71.14</td><td>50.46</td><td>67.24</td><td>37.08</td><td>60.40</td><td>60.33</td><td>57.78</td><td>48.82</td><td>48.56</td><td>11.29</td><td>36.22</td><td>13.68</td><td>48.62</td><td>60.00</td><td>40.77</td><td>44.92</td></tr><tr><td>Streamo-3B</td><td>1 fps</td><td>78.52</td><td>52.29</td><td>67.24</td><td>44.38</td><td>55.45</td><td>71.20</td><td>61.51</td><td>51.18</td><td>57.43</td><td>16.67</td><td>41.76</td><td>27.94</td><td>50.72</td><td>82.50</td><td>53.72</td><td>52.33</td></tr><tr><td>Streamo-7B</td><td>1 fps</td><td>79.19</td><td>57.80</td><td>75.00</td><td>49.44</td><td>64.36</td><td>70.11</td><td>65.98</td><td>54.55</td><td>52.03</td><td>31.72</td><td>46.10</td><td>29.96</td><td>51.03</td><td>83.33</td><td>54.77</td><td>55.61</td></tr><tr><td>Streamo-7B</td><td>2 fps*</td><td>77.18</td><td>66.06</td><td>76.72</td><td>45.51</td><td>66.34</td><td>72.83</td><td>67.44</td><td>55.56</td><td>58.11</td><td>33.87</td><td>49.18</td><td>30.84</td><td>57.55</td><td>82.50</td><td>56.96</td><td>57.86</td></tr><tr><td>Streamo-2B (InternVL3)</td><td>1 fps</td><td>77.18</td><td>55.96</td><td>62.07</td><td>41.01</td><td>60.40</td><td>70.11</td><td>61.12</td><td>48.82</td><td>47.30</td><td>13.44</td><td>36.52</td><td>29.23</td><td>47.38</td><td>80.42</td><td>52.34</td><td>49.99</td></tr><tr><td>Streamo-4B (Qwen3-VL)</td><td>1 fps</td><td>82.55</td><td>69.72</td><td>74.14</td><td>52.25</td><td>73.27</td><td>81.52</td><td>72.24</td><td>58.19</td><td>52.70</td><td>17.20</td><td>42.70</td><td>31.38</td><td>53.90</td><td>84.17</td><td>56.48</td><td>55.10</td></tr><tr><td>Qwen2.5-VL-7B + 4f</td><td>1fps</td><td>94.00</td><td>72.50</td><td>80.20</td><td>68.00</td><td>76.20</td><td>79.30</td><td>78.40</td><td>54.50</td><td>60.80</td><td>40.30</td><td>51.90</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen3-VL-8B + 4f</td><td>1fps</td><td>94.00</td><td>85.30</td><td>82.80</td><td>65.70</td><td>77.20</td><td>83.20</td><td>81.40</td><td>51.90</td><td>58.10</td><td>52.10</td><td>54.00</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="18">SelectStream Framework</td></tr><tr><td>SELECTSTREAM-Qwen2.5-VL-7B</td><td>1 fps</td><td>94.30</td><td>75.80</td><td>82.50</td><td>70.10</td><td>78.40</td><td>84.00</td><td>80.85</td><td>60.20</td><td>66.40</td><td>56.55</td><td>61.05</td><td>30.50</td><td>52.70</td><td>82.49</td><td>55.23</td><td>65.71</td></tr><tr><td>SELECTSTREAM-Qwen3-VL-8B</td><td>1 fps</td><td>94.60</td><td>86.70</td><td>84.10</td><td>68.36</td><td>78.90</td><td>83.90</td><td>82.76</td><td>62.40</td><td>66.80</td><td>57.40</td><td>62.20</td><td>31.20</td><td>53.80</td><td>83.39</td><td>56.13</td><td>67.03</td></tr></table>

## I Broader Impacts

SelectStream aims to make streaming video understanding more efficient by replacing repeated full-history replay with compact latent evidence retrieval. This can reduce inference cost for long video streams and may support applications such as assistive video analysis, long-form content understanding, and real-time decision support under bounded computation. The fixed-budget design can also make deployment more predictable because memory usage and query-time context size are explicitly controlled.

At the same time, improvements in video understanding can be misused in privacy-sensitive settings, including large-scale surveillance or automated analysis of people without appropriate consent. SelectStream does not introduce new data collection or identity-recognition mechanisms, but it could be combined with existing video-language systems in applications where privacy, fairness, and consent require careful review. We therefore recommend that deployments follow the terms of the underlying datasets and backbone models, respect local privacy regulations, and include application-specific safeguards when used in real-world streaming environments.