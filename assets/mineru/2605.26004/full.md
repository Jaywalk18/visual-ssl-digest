# MAGIC: Multimodal Alignment & Grounding-aware Instruction Coreset for Vision-Language Models

Shristi Das Biswas and Kaushik Roy

Purdue University

sdasbisw@purdue.edu, kaushik@purdue.edu

# Abstract

Instruction tuning of large vision-language models (LVLMs) increasingly depends on massive multimodal corpora, yet these datasets contain samples with substantial redundancy, low visual dependency, and highly imbalanced coverage of multimodal reasoning behaviors. As a result, uniform subsampling or naive scorebased selection often yields suboptimal training subsets. We introduce MAGIC, a trainingfree, forward-only coreset selection method designed to construct compact yet behaviorally faithful subsets for multimodal instruction tuning. MAGIC is built on three intrinsic signals extracted from a pretrained VLM: Multimodal Gain, which measures the likelihood improvement obtained from visual input; Bridging Relevance, which captures the sharpness of answer-token grounding over visual tokens; and Skill-Neuron Signatures, which characterize the functional computation elicited by each sample via top-activated feed-forward neurons. MAGIC combines these signals in a three-stage pipeline: filtering low-gain examples, ranking candidates by a normalized quality objective, and performing bucket-wise budget allocation over discrete neuron signatures to preserve latent multimodal skill coverage. This formulation avoids backpropagation, auxiliary selector training, and expensive clustering in continuous activation spaces, while remaining efficient and easily deployable in existing VLMs. Across LLaVA-665K and Vision-Flan datasets, and transfer settings to large target models, LLaVA-1.5-7B and -13B, MAGIC consistently improves over strong baselines under matched 20% budgets: it achieves 100.3% relative performance to full finetuning on LLaVA-665K and 101.6% relative performance on Vision-Flan-186K, while yielding a 73.7% reduction in wall-clock run time. Overall, our results establish that activation-aware behavioral coverage, when coupled with multimodal utility and grounding strength, provides a principled and practical foundation for VLM coreset selection.

![](images/b828d899d691138b3a7a1f8f14b987d8a22728c08bc09ba7b6836b2a027537f3.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Multimodal Gain\nDoes vision actually help reduce uncertainty?"] --> B["Takeaway: Visual Utility"]
    C["Bridging Relevance\nIs the answer grounded in the image?"] --> D["Takeaway: Grounding Quality"]
    E["Skill Signature\nWhat internal skill pattern is activated?"] --> F["Takeaway: Behavioral Diversity"]
    B --> G["MAGIC Selection Criterion\nSelect samples that are:\n✓ Visually Useful (High Multimodal Gain) + Strongly Grounded (High Bridging Relevance) + Behaviorally Diverse (Diverse Skill Signature)"]
    D --> G
    F --> G
    G --> H["Higher-quality, less redundant, and more transferable coreset"]
```
</details>

Figure 1: MAGIC characterizes each training sample using three complementary signals: Multimodal Gain; Bridging Relevance; and Skill Signature. These signals are combined to prioritize samples that are visually useful, strongly grounded, and behaviorally diverse for visual-language coreset selection.

# 1 Introduction

Large vision-language models (LVLMs) have recently achieved strong performance across a wide range of multimodal tasks (Dai et al., 2023; Alayrac et al., 2022; Yu et al., 2023a; Radford et al., 2021; Zhu et al., 2023). A standard LVLM pipeline typically consists of large-scale image-text pretraining followed by visual instruction tuning, where the model is adapted to follow task-oriented instructions. While this second stage is crucial for instruction-following ability, it has become increasingly expensive due to the rapid growth of visual instruction datasets. Moreover, these corpora are often highly redundant, contain many weakly grounded samples that require limited visual understanding, and exhibit severely imbalanced coverage of latent multimodal behaviors, making full-data tuning both costly and inefficient. Consequently, there is growing interest in multimodal data selection methods that can construct compact training subsets while preserving downstream performance.

Existing multimodal data selection methods largely inherit ideas from the broader coreset and instruction-tuning literature, including scalar scorebased ranking (Paul et al., 2021; Chen et al., 2024; Du et al., 2023; Zhou et al., 2023), redundancy pruning (Abbas et al., 2023), gradient-based selection (Wu et al., 2024; Liu et al., 2025; Tan et al., 2023), and clustering-based diversity promotion (Lee et al., 2024; Yu et al., 2025). However, these approaches face two major limitations in the visual instruction setting. First, single-score methods often over-select narrow modes of the dataset, since highly ranked samples may repeatedly reflect the same task pattern, concept type, or response style. Second, stronger selection methods often incur substantial overhead by relying on backward passes, auxiliary selector training, or expensive clustering in continuous activation spaces. Prior work has shown that overlooking sample relationships and latent data structure can lead to poor sample uniqueness or representativeness, ultimately limiting generalization (Yu et al., 2025).

In this paper, we argue that effective multimodal coreset selection should satisfy three desiderata: selected samples should exhibit strong multimodal utility, be strongly grounded in visual evidence, and sufficiently diverse in the latent computations they induce. Existing methods typically address only part of this problem. Utility-based ranking captures sample quality but not coverage, while diversityoriented methods improve spread but do not ensure that selected examples are both informative and visually grounded.

To address these limitations, we introduce MAGIC, a training-free, forward-only coreset selection method for visual instruction tuning. MAGIC is built on three intrinsic signals extracted from a pretrained VLM, as seen in Fig. 1: Multimodal Gain, which measures improvement in the model’s response likelihood when multimodal inputs are present relative to a unimodal text-only forward pass; Bridging Relevance, which measures the concentration of answer-token attention over visual tokens; and a Skill-Neuron Signature, a discrete fingerprint formed from top-activated feed-forward neurons induced by a sample. Together, these signals characterize not only whether a sample is useful for multimodal learning, but also what latent computation it elicits inside the model. MAGIC combines these signals in a three-stage pipeline. It first filters weakly multimodal samples, then ranks the remaining candidates with a normalized quality score combining Multimodal Gain and Bridging Relevance scores, and finally allocates budget across discrete Skill-Signature buckets to preserve behavioral coverage.

We evaluate MAGIC on two visual instruction tuning corpora, LLaVA-665K (Liu et al., 2024a) and Vision-Flan (Xu et al., 2024), using LLaVA-1.5-7B as the primary target model and further assessing transfer to the larger LLaVA-1.5-13B. Under matched 20% selection budgets, MAGIC achieves 100.3% relative performance with respect to full-data finetuning on LLaVA-665K and 101.6% on Vision-Flan-186K, while retaining 99.3% relative performance when transferred to LLaVA-1.5- 13B. Moreover, MAGIC not only reduces endto-end wall-clock runtime by 73.7% relative to full-data finetuning, yielding a substantially improved compute-performance trade-off, but also improves generalization to unseen tasks. These results show that activation-aware behavioral coverage, combined with multimodal utility and grounding strength, provides a principled and efficient foundation for multimodal coreset selection. Overall, our contributions are threefold: (i) we propose MAGIC, a training-free, and scalable coreset selection framework for visual instruction tuning that unifies multimodal utility, visual grounding, and activation-aware behavioral diversity; (ii) we introduce Multimodal Gain, Bridging Relevance, and Skill-Neuron Signature bucketing as intrinsic sample-level criteria for coreset construction, without any backward passes, proxy selector training, or expensive continuous activation-space clustering; and (iii) we demonstrate that MAGIC delivers consistent improvements in data efficiency at just 20% coreset selection and yields end-to-end compute-performance trade-offs across two distinct VIT datasets, LLaVA-665K and Vision-Flan, and different VLM target models, while substantially reducing total wall-clock run time by 74%. We further demonstrate improved generalization to unseen tasks, despite a 5× reduction in training data.

# 2 Related Works

Coreset selection aims to form a smaller training set whose optimization behavior and downstream performance closely match those obtained from the complete data corpus. Data selection methods can be categorized based on the types of information they utilize for selection (Hammoudeh and Lowd, 2024). Representation-based approaches (Abbas et al., 2023; Lee et al., 2024; Yu et al., 2025) leverage embeddings or latent features to capture sample structure and similarity. In particular, (Lee et al., 2024) clusters data based on representations associated with concept-skill compositions, while (Yu et al., 2025) proposes a collaborative framework combining spectral analysis and clustering-derived sample relationships for multimodal data valuation. Loss-trajectory-based methods (Mindermann et al., 2022) prioritize data points that contribute most significantly to reducing generalization error over training. Gradient-based techniques (Xia et al., 2024; Wu et al., 2024; Paul et al., 2021; Deng et al., 2024; Liu et al., 2025) select data based on gradient information or estimated downstream influence. Recent work has explored various approaches to select optimal visual instruction tuning datasets. On the other hand, authors in Chen et al. (2024) score VIT data using an external scoring model trained with the target model. More recently, works such as Yan et al. (2025) introduce a lightweight scorer that jointly optimizes importance and diversity using a coupled objective trained on a sampled subset of data. In contrast, MAGIC is entirely trainingfree and forward-only. Instead of depending on gradient computation, auxiliary scorer optimization, or clustering in continuous feature spaces, it selects samples using intrinsic signals that directly capture multimodal utility, grounding strength, and activation-aware behavioral coverage.

# 3 Methodology

In this section, we formalize the MAGIC framework. We begin by introducing the VLM setting and the internal representations accessible from a pretrained model under a forward pass. These quantities form the basis for the three intrinsic signals used by MAGIC, as seen in Fig. 2.

# 3.1 Problem Formulation

Let $\mathcal { D } = \{ ( I _ { i } , T _ { i } , Y _ { i } ) \} _ { i = 1 } ^ { N }$ be a visual instruction tuning dataset, where $I _ { i }$ is an image, $T _ { i }$ is a textual instruction, and $Y _ { i }$ is the target response. The goal of multimodal coreset selection is to construct a subset $\mathcal { C } \subseteq \mathcal { D }$ ,, where, $| { \mathcal { C } } | = M \ll N$ , such that finetuning a VLM on C retains the utility of finetuning on the full dataset D across multiple downstream tasks, whilst reducing visual instruction tuning costs.

# 3.2 Preliminaries

We consider a pretrained LVLM M composed of a visual encoder, a modality projector, and a transformer-based language model. For a sample $x _ { i } = ( I _ { i } , T _ { i } , Y _ { i } )$ , the image $I _ { i }$ is encoded into $N _ { v }$ visual tokens, projected into the language model space, and concatenated with $N _ { t }$ textual tokens from the instruction-response sequence (Dai et al., 2023; Liu et al., 2023). Here, we focus on the common setting where visual information is injected into a transformer-based LLM as input tokens.

At transformer layer ℓ, let

$$
X _ {i} ^ {(\ell)} = [ X _ {v, i} ^ {(\ell)}; X _ {t, i} ^ {(\ell)} ] \in \mathbb {R} ^ {(N _ {v} + N _ {t}) \times D} \tag {1}
$$

denote the concatenated hidden states, where D is the hidden dimension. The multi-head selfattention (MSA) block produces

$$
Z _ {i} ^ {(\ell)} = \mathrm{MSA} ^ {(\ell)} \big (\mathrm{LN} ^ {(\ell)} (X _ {i} ^ {(\ell)}) \big) + X _ {i} ^ {(\ell)}, \tag {2}
$$

where LN denotes layer normalization, and $Z _ { i } ^ { ( \ell ) } =$ $[ Z _ { v , i } ^ { ( \ell ) } ; Z _ { t , i } ^ { ( \ell ) } ]$ are the output visual and textual features, respectively. Let

$$
A _ {i} ^ {(\ell)} \in \mathbb {R} ^ {n _ {h} \times (N _ {v} + N _ {t}) \times (N _ {v} + N _ {t})} \tag {3}
$$

denote the corresponding head-wise attention weights induced by $\mathrm { \ u s A } ^ { ( \bar { \ell } ) }$ , where $n _ { h }$ is the number of attention heads. We further denote by

$$
H _ {i} ^ {(\ell)} \in \mathbb {R} ^ {N _ {t} \times D _ {\mathrm{ff}}} \tag {4}
$$

the token-wise textual FFN activations at layer $\ell ,$ where $D _ { \mathbb { H } }$ is the FFN width.

Under this notation, MAGIC derives three intrinsic forward-pass signals from a pretrained LVLM.

# 3.3 MAGIC Score Extraction

Multimodal Gain. To quantify the extent to which a training sample genuinely benefits from multimodal evidence, we compare the model’s answer-token prediction error under full multimodal conditioning and under a unimodal text-only setting $( I _ { i } = \emptyset )$ . Let $\mathcal { T } _ { i } ^ { a }$ denote the set of answertoken positions for sample i. We define the Multimodal Gain score as

$$
\Delta_ {i} ^ {(t)} = \mathrm{CE} (y _ {i, t}; y _ {i, <   t}, \varnothing , T _ {i}; \theta)
$$

$$
- \mathrm{CE} (y _ {i, t}; y _ {i, <   t}, I _ {i}, T _ {i}; \theta). \tag {5}
$$

$$
g _ {i} = \frac {1}{\left| \mathcal {T} _ {i} ^ {a} \right|} \sum_ {t \in \mathcal {T} _ {i} ^ {a}} \Delta_ {i} ^ {(t)} \tag {6}
$$

where $\mathrm { C E } ( \cdot )$ idenotes the per-token crossentropy, $I _ { i }$ is the image, $T _ { i }$ is the textual context, and $\emptyset$ denotes the absence of visual input. Thus, $g _ { i }$ quantifies the average reduction in answer-token prediction error attributable to the presence of the image. Samples with large $g _ { i }$ are those for which visual information materially improves target prediction, indicating high multimodal utility. In contrast, samples with small or near-zero values are largely recoverable from textual context alone and thus provide limited additional supervision for learning image-grounded behavior.

![](images/3f69795b09cdca90288b7d820b82583773db1396317b603341d6f4f01684c005.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Compute Multimodal Gain Score ĝᵢ = Norm(gᵢ)"] --> B["Filter out weakly multimodal samples E = Top_{[pN"]}({ĝᵢ})^{N}_{i=1}]
    C["Compute Bridging score bᵢ = Norm(bᵢ)"] --> D["Compute Quality-based shortlist qᵢ = αĝᵢ + βbᵢ S = Topₘᵢₙ(ᵢₘⱼ, |E|) (qᵢⱼ)ᵢ∈E"]
    E["Aggregate FFN activations across selected layers ℓ hᵢ⁽ᵉ⁾"] --> F["Keep the top activated neurons sᵢ⁽ᵉ⁾ = TopKIdx hᵢ⁽ᵉ⁾"]
    F --> G["Compute Skill Fingerprints φᵢ = {(ℓ, c) : c ∈ sᵢ⁽ᵉ⁾, ℓ ∈ L}"]
    G --> H["Partition S into buckets {B₁, ..., Bₖ} by φᵢ"]
    H --> I["Compute Bucket Mass mⱼ = Σ exp(qᵢ / τ)ᵢ∈Bⱼ"]
    I --> J["Normalize Bucket Mass pⱼ = mⱼ / Σ m"]
    J --> K["Assign Capped Initial Quota nⱼ ← min(|Bⱼ|, |γM|, |Mpⱼ|)"]
    K --> L["Intra-bucket top-score selection C ← U Topₙⱼ ( {qᵢⱼ}ᵢ∈Bⱼ )"]
    L --> M["Redistribute Leftover Budget By Descending Fraction of Mpⱼ"]
    M --> N["Backfill C With Any Unused Slots From The Remaining S and E"]
```
</details>

Figure 2: Overview of MAGIC. The method first extracts three forward-only intrinsic score signals from a pretrained VLM. It then filters weakly multimodal samples, builds a quality-based shortlist, partitions shortlisted samples into skill-signature buckets, and allocates the budget across buckets to construct the final coreset.

Bridging Relevance. Multimodal gain measures whether vision is useful, but not whether the response is strongly grounded in visual evidence. To capture this, we use the attention maps from selected transformer layers L. For each $\ell \in { \mathcal { L } } .$ , let $A _ { i } ^ { ( \ell ) }$ denote the head-wise attention tensor, and let $\bar { A } _ { i } ^ { ( \ell ) }$ be the average over heads. We consider the submatrix corresponding to answer-token queries t attending to visual-token keys v. To characterize how this attention mass is distributed within the visual-token block, we define the normalized image-token attention as

$$
\tilde {A} _ {i, t v} ^ {(\ell)} = \frac {\bar {A} _ {i , t v} ^ {(\ell)}}{\sum_ {u = 1} ^ {N _ {v}} \bar {A} _ {i , t u} ^ {(\ell)}}. \tag {7}
$$

This induces a distribution over visual tokens conditioned on answer-token query t. We then define its entropy as $\begin{array} { r } { e _ { i , t } ^ { ( \ell ) } = - \sum _ { v = 1 } ^ { \hat { N } _ { v } } \dot { \tilde { A } } _ { i , t v } ^ { ( \ell ) } } \end{array}$ ei,t log $\tilde { A } _ { i , t v } ^ { ( \ell ) }$ . Lower entropy corresponds to attention concentrating on more localized visual evidence, while high entropy indicates diffused attention on the entire image input. Hence, we define Bridging Relevance as

$$
b _ {i} = \frac {1}{| \mathcal {L} |} \sum_ {\ell \in \mathcal {L}} \frac {1}{| \mathcal {T} _ {i} ^ {a} |} \sum_ {t \in \mathcal {T} _ {i} ^ {a}} \left[ \left(\sum_ {v = 1} ^ {N _ {v}} \bar {A} _ {i, t v} ^ {(\ell)}\right) \left(1 - \frac {e _ {i , t} ^ {(\ell)}}{\log N _ {v}}\right) \right] \tag {8}
$$

Thus, $b _ { i }$ is high when answer tokens attend strongly to the visual input and concentrate this attention on a small set of evidence-bearing visual tokens.

Skill-Neuron Signature. To characterize the latent computation induced by a sample, we extract FFN activations from selected transformer layers, following prior work suggesting that these layers function as key-value memories and that individual neurons can encode interpretable concepts or localized knowledge (Geva et al., 2021, 2022; Dai et al., 2022). Let H (ℓ)i $H _ { i } ^ { \overline { { ( \ell ) } } }$ denote the token-wise textual FFN activations at layer ℓ. We aggregate them over the supervised answer-token positions:

$$
h _ {i} ^ {(\ell)} = \frac {1}{| \mathcal {T} _ {i} ^ {a} |} \sum_ {t \in \mathcal {T} _ {i} ^ {a}} H _ {i, t} ^ {(\ell)} \in \mathbb {R} ^ {D _ {\mathrm{ff}}}. \tag {9}
$$

We then retain the indices of the ${ \mathrm { t o p } } \lnot k _ { \ell }$ activated neurons,

$$
s _ {i} ^ {(\ell)} = \mathrm{TopKIdx} _ {k _ {\ell}} \Big (h _ {i} ^ {(\ell)} \Big), \tag {10}
$$

and form the multi-layer skill signature by layerneuron pairs as follows:

$$
\phi_ {i} = \{(\ell , c): c \in s _ {i} ^ {(\ell)}, \ell \in \mathcal {L} \}. \tag {11}
$$

This discrete signature serves as a compact behavioral fingerprint of the sample. The resulting triplet $( g _ { i } , b _ { i } , \phi _ { i } )$ , as seen in Alg. 1, forms the basis of the subsequent coreset selection procedure.

# 3.4 MAGIC Data Selection

Given the extracted sample descriptors $( g _ { i } , b _ { i } , \phi _ { i } )$ , MAGIC constructs the coreset through quality scoring, shortlist formation, and behavior-aware budget allocation, as shown in Alg. 2.

Quality scoring and Eligibility Filtering. Since multimodal gain and bridging relevance have different scales, we first apply robust normalization:

$$
\hat {g} _ {i} = \mathrm{Norm} (g _ {i}), \qquad \hat {b} _ {i} = \mathrm{Norm} (b _ {i}). \tag {12}
$$

We then define a joint quality score $q _ { i } = \alpha \hat { g } _ { i } + \beta \hat { b } _ { i }$ where $\alpha , \beta \ge 0$ control the relative contribution of multimodal utility and grounding strength. To suppress weakly multimodal examples, we first retain only the top $\lceil \rho N \rceil$ examples ranked by $g _ { i } \colon$

Algorithm 1 MAGIC Score Extraction   
Input: dataset $\mathcal{D} = \{(I_i, T_i, Y_i)\}_{i=1}^N$ , pretrained LVLM $\mathcal{M}$ , selected layers $\mathcal{L}$ , per-layer signature sizes $\{k_\ell\}_{\ell \in \mathcal{L}}$ Output: multimodal gain $\{g_i\}_{i=1}^N$ , bridging relevance $\{b_i\}_{i=1}^N$ , skill signatures $\{\phi_i\}_{i=1}^N$ 1: for $i = 1, \ldots, N$ do
2: Compute answer-token set $\mathcal{T}_i^a$ from $Y_i$ 3: for each $t \in \mathcal{T}_i^a$ do
4: $\Delta_i^{(t)} \leftarrow \text{CE}(y_{i,t}; y_{i,<t}, \emptyset, T_i; \theta) - \text{CE}(y_{i,t}; y_{i,<t}, I_i, T_i; \theta)$ 5: end for
6: $g_i \leftarrow \frac{1}{|\mathcal{T}_i^a|} \sum_{t \in \mathcal{T}_i^a} \Delta_i^{(t)}$ 7: for each $\ell \in \mathcal{L}$ do
8: $A_i^{(\ell)} \leftarrow \text{Attn}^{(\ell)}(\mathcal{M}(I_i, T_i, Y_i))$ 9: $\bar{A}_i^{(\ell)} \leftarrow \text{MeanHead}\left(A_i^{(\ell)}\right)$ 10: $H_i^{(\ell)} \leftarrow \text{FFNAct}^{(\ell)}(\mathcal{M}(I_i, T_i, Y_i))$ 11: $h_i^{(\ell)} \leftarrow \frac{1}{|\mathcal{T}_i^a|} \sum_{t \in \mathcal{T}_i^a} H_{i,t}^{(\ell)}$ 12: $s_i^{(\ell)} \leftarrow \text{TopKIdx}_{k_\ell}\left(h_i^{(\ell)}\right)$ 13: end for
14: $\tilde{A}_{i,tv}^{(\ell)} = \frac{\tilde{A}_{i,tv}^{(\ell)}}{\sum_{u=1}^{N_v} \tilde{A}_{i,tu}^{(\ell)}}$ 15: $e_{i,t}^{(\ell)} = -\sum_{v=1}^{N_v} \tilde{A}_{i,tv}^{(\ell)} \log \tilde{A}_{i,tv}^{(\ell)}$ 16: $b_i = \frac{1}{|\mathcal{L}|} \sum_{\ell \in \mathcal{L}} \frac{1}{|\mathcal{T}_i^a|} \sum_{t \in \mathcal{T}_i^a} \left[ \left( \sum_{v=1}^{N_v} \bar{A}_{i,tv}^{(\ell)} \left( 1 - \frac{e_{i,t}^{(\ell)}}{\log N_v} \right) \right] \right.\right)$ 17: $\phi_i \leftarrow \{ (\ell, c) : c \in s_i^{(\ell)}, \ell \in \mathcal{L}\}$ 18: end for
19: return $\{g_i\}_{i=1}^N, \{b_i\}_{i=1}^N, \{\phi_i\}_{i=1}^N$

$$
\mathcal {E} = \mathrm{Top} _ {\lceil \rho N \rceil} (\{g _ {i} \} _ {i = 1} ^ {N}). \tag {13}
$$

From this eligible set, we form a quality shortlist

$$
\mathcal {S} = \mathrm{Top} _ {\min (\lceil \eta M \rceil , | \mathcal {E} |)} (\{q _ {i} \} _ {i \in \mathcal {E}}), \tag {14}
$$

where $\eta \geq 1$ is a shortlist expansion factor. This stage ensures that the final selection is drawn from candidates with both strong multimodal utility and high overall quality.

Behavior-Aware Grouping. To preserve latent behavioral diversity, we partition the shortlist $s$ into discrete buckets according to the skill-neuron signatures $\phi _ { i }$ . Let

$$
\mathcal {B} = \{B _ {1}, \dots , B _ {K} \} \tag {15}
$$

denote the resulting partition, where samples sharing the same multi-layer signature are assigned to the same bucket. Unlike continuous clustering approaches (Yu et al., 2025), this discrete grouping is computationally lightweight and directly captures sample-induced behavioral similarity through shared skill-neuron activation patterns.

Final Selection. We allocate the coreset budget across buckets using a temperature-scaled quality mass. For each bucket $B _ { j } .$ , we define $m _ { j } = { \it { \Delta } }$ $\begin{array} { r } { \sum _ { i \in B _ { j } } \exp ( q _ { i } / \tau ) } \end{array}$ , and normalize across buckets:

$$
p _ {j} = \frac {m _ {j}}{\sum_ {r = 1} ^ {K} m _ {r}}. \tag {16}
$$

The initial quota for bucket $B _ { j }$ is then

$$
n _ {j} = \min (| B _ {j} |, \lceil \gamma M \rceil , \lfloor M p _ {j} \rfloor), \tag {17}
$$

where $\gamma$ caps over-allocation to dominant buckets. Any remaining budget $\begin{array} { r } { M - \sum _ { j = 1 } ^ { K } n _ { j } } \end{array}$ P is redistributed by descending fractional remainder of $M p _ { j }$ . Finally, we select the $\mathrm { t o p } { - } n _ { j }$ samples in each bucket according to $q _ { i } \colon$

$$
\mathcal {C} = \bigcup_ {j = 1} ^ {K} \mathrm{Top} _ {n _ {j}} (\{q _ {i} \} _ {i \in B _ {j}}). \tag {18}
$$

$\mathrm { I f } \ | { \mathcal { C } } | < M .$ , we backfill from $s \backslash { \mathcal { C } }$ , and then from ${ \mathcal { E } } \backslash { \mathcal { C } } .$ , in descending order of $q _ { i } .$ , until the target budget is reached. Here, $\mathcal A \backslash B$ denotes the elements in A that are not already contained in B. Overall, this procedure retains high-quality, strongly grounded, and multimodally useful samples while preserving coverage over distinct latent computation patterns.

# 4 Evaluation

# 4.1 Experimental Setup

Datasets. We conduct 20% coreset selection experiments on two visual instruction tuning datasets: LLaVA-665K (Liu et al., 2024a) and Vision-Flan-186K (Xu et al., 2024). LLaVA-665K contains approximately 665K multimodal instruction samples collected from diverse vision-language tasks, while Vision-Flan-186K comprises 191 vision-language tasks with roughly 1K expert-annotated instruction examples per task, totaling 186K samples.

<table><tr><td>Method</td><td>MME</td><td>SQA-I</td><td>POPE</td><td>VQAv2</td><td>LLaVA-W Bench</td><td>TextVQA</td><td>MMBench en</td><td>MMBench cn</td><td>GQA</td><td>VizWiz</td><td>MM-Vet</td><td>Rel. (%)</td></tr><tr><td>Full</td><td>1476.9</td><td>68.4</td><td>86.4</td><td>79.1</td><td>67.9</td><td>58.2</td><td>66.1</td><td>58.9</td><td>63.0</td><td>47.8</td><td>30.9</td><td>100.0</td></tr><tr><td>Random</td><td>1483.0</td><td>68.5</td><td>84.7</td><td>75.7</td><td>65.0</td><td>55.3</td><td>62.2</td><td>54.8</td><td>58.9</td><td>44.3</td><td>29.5</td><td>95.8</td></tr><tr><td>CLIP-Score</td><td>1331.6</td><td>65.0</td><td>85.3</td><td>73.4</td><td>66.2</td><td>54.7</td><td>55.2</td><td>52.0</td><td>51.4</td><td>43.0</td><td>-</td><td>91.2</td></tr><tr><td>EL2N</td><td>1439.5</td><td>65.5</td><td>84.3</td><td>76.2</td><td>64.9</td><td>53.0</td><td>53.2</td><td>47.4</td><td>58.7</td><td>43.7</td><td>21.1</td><td>89.8</td></tr><tr><td>Perplexity</td><td>1341.4</td><td>65.1</td><td>82.6</td><td>75.8</td><td>68.3</td><td>52.8</td><td>52.0</td><td>45.8</td><td>57.0</td><td>47.8</td><td>-</td><td>91.6</td></tr><tr><td>SemDeDup</td><td>1376.9</td><td>65.8</td><td>84.7</td><td>74.2</td><td>70.0</td><td>55.5</td><td>52.2</td><td>48.5</td><td>54.5</td><td>46.9</td><td>-</td><td>92.6</td></tr><tr><td>D2-Pruning</td><td>1391.2</td><td>69.3</td><td>85.7</td><td>73.0</td><td>63.9</td><td>51.8</td><td>65.7</td><td>57.6</td><td>58.4</td><td>41.9</td><td>-</td><td>94.8</td></tr><tr><td>Self-Sep</td><td>1335.9</td><td>67.8</td><td>83.5</td><td>74.9</td><td>63.3</td><td>49.3</td><td>61.4</td><td>53.8</td><td>59.5</td><td>46.0</td><td>-</td><td>93.4</td></tr><tr><td>Self-Filter</td><td>1306.2</td><td>61.4</td><td>83.8</td><td>73.7</td><td>64.9</td><td>52.9</td><td>48.8</td><td>45.3</td><td>58.3</td><td>53.2</td><td>26.6</td><td>90.5</td></tr><tr><td>COINCIDE</td><td>1495.6</td><td>69.2</td><td>86.1</td><td>76.5</td><td>67.3</td><td>55.6</td><td>63.1</td><td>54.5</td><td>59.8</td><td>46.8</td><td>-</td><td>97.4</td></tr><tr><td>RDS</td><td>1093.8</td><td>68.0</td><td>86.3</td><td>75.1</td><td>63.7</td><td>54.9</td><td>61.2</td><td>52.7</td><td>57.9</td><td>48.6</td><td>-</td><td>93.2</td></tr><tr><td>DataTailor</td><td>1476.1</td><td>-</td><td>85.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>49.5</td><td>31.8</td><td>-</td><td>85.9</td></tr><tr><td>COIDO</td><td>1450.2</td><td>69.4</td><td>85.4</td><td>77.2</td><td>70.1</td><td>55.6</td><td>63.8</td><td>56.7</td><td>60.4</td><td>47.1</td><td>-</td><td>98.2</td></tr><tr><td>ICONS</td><td>1485.7</td><td>70.8</td><td>87.5</td><td>76.3</td><td>66.1</td><td>55.6</td><td>63.1</td><td>55.8</td><td>60.7</td><td>50.1</td><td>29.7</td><td>98.4</td></tr><tr><td>MAGIC (ours)</td><td>1657.9</td><td>72.3</td><td>87.1</td><td>76.8</td><td>69.0</td><td>56.4</td><td>65.1</td><td>56.8</td><td>60.4</td><td>50.2</td><td>29.8</td><td>100.3</td></tr></table>

Table 1: Performance comparison of different data selection approaches when trained on 20% of the LLaVA-665K dataset. The best and second-best results for each benchmark are shown in bold and underlined, respectively. Our method MAGIC achieves the highest overall Rel. (100.3%), consistently outperforming existing approaches including ICONS (98.4%) and COIDO (98.2%).

<table><tr><td>Dataset</td><td>#Data</td><td>Method</td><td>VQAv2</td><td>GQA</td><td>SQA-I</td><td>TextVQA</td><td>POPE</td><td>MME</td><td>MMBench en</td><td>MMBench cn</td><td>LLaVA-W Bench Bench</td><td>Rel. (%)</td></tr><tr><td rowspan="3">Vision-Flan-186K</td><td>186k</td><td>Full-Finetune</td><td>68.0</td><td>49.2</td><td>60.8</td><td>50.4</td><td>83.4</td><td>1263.2</td><td>52.6</td><td>45.9</td><td>63.3</td><td>100.0</td></tr><tr><td>~37k</td><td>Random</td><td>64.1</td><td>45.8</td><td>58.7</td><td>45.3</td><td>82.9</td><td>1079.8</td><td>46.5</td><td>39.6</td><td>58.7</td><td>91.8</td></tr><tr><td>~37k</td><td>MAGIC (ours)</td><td>67.5</td><td>50.7</td><td>65.6</td><td>45.7</td><td>84.8</td><td>1351.6</td><td>52.0</td><td>46.5</td><td>63.2</td><td>101.6</td></tr></table>

Table 2: Selection results on Vision-Flan-186K. Performance comparison of different data selection approaches when trained on 20% of the Vision-Flan-186K. MAGIC achieves strong performance (101.6%) while using only 20% of the training data, significantly outperforming random selection and full performance.

Models for Selection and Target Finetuning. Following prior work (Lee et al., 2024), we use TinyLLaVA-2B (Zhou et al., 2024) as our reference model for efficient data selection, from which MAGIC extracts its scores. Unless otherwise specified, our primary target model is LLaVA-1.5-7B (Liu et al., 2024a). To evaluate transferability of the selected coreset to larger backbones, we additionally evaluate performance on LLaVA-1.5- 13B. Following the standard LLaVA-1.5 finetuning recipe, we finetune using LoRA (Hu et al., 2022) for one epoch with the official hyperparameter settings, using 4 NVIDIA A100 GPUs. The joint quality score uses weights $( \alpha , \beta ) = ( 0 . 5 , 0 . 5 )$ for MG and BR, respectively, and we extracted scores from 4 uniformly spaced layers from TinyLLaVA-2B, inspired by prior work (Lee et al., 2024). More details on experimental settings is in App. Sec. A.3.

Evaluation Benchmarks. We evaluate on a diverse benchmark suite covering key VLM capabilities, including multiple-choice understanding, visual question answering, text understanding in images, scientific reasoning, open-ended generation, and factual consistency. Concretely, our evaluation suite includes MMBench (Liu et al., 2024b), MME (Fu et al., 2023), VQAv2 (Goyal et al., 2017), GQA (Hudson and Manning, 2019), VizWiz (Gurari et al., 2018), TextVQA (Singh et al., 2019), ScienceQA (Saikh et al., 2022), LLaVA-W Bench (Lu et al., 2022), MM-Vet (Yu et al., 2023b), and POPE (Li et al., 2023b).

Baselines. We compare MAGIC against several strong baselines spanning different data selection paradigms. These include random selection, CLIP-Score (Radford et al., 2021) for measuring image-text alignment, EL2N (Paul et al., 2021) based on prediction difficulty, and Perplexity (Marion et al., 2023) using language modeling uncertainty. We also compare against SemDeDup (Abbas et al., 2023) for semantic deduplication, D2- Pruning (Maharana et al., 2023) for distributionaware data pruning, Self-Sup (Sorscher et al., 2022), which leverages self-supervised signals, and Self-Filter (Chen et al., 2024), which uses selffiltering to remove less informative samples. Additional baselines include methods designed specifically for vision-language coresets, such as CO-INCIDE (Lee et al., 2024), DataTailor (Yu et al., 2025), COIDO (Yan et al., 2025), ICONS (Wu et al., 2024), and representation-based selection (RDS) (Xia et al., 2024; Ivison et al., 2025).

# 4.2 Results and Discussion

Main results on LLaVA-665K. Table 1 summarizes coreset selection results on LLaVA-665K. MAGIC achieves the best overall relative score of 100.3%, outperforming all baselines, including ICONS (98.4%) and COIDO (98.2%). It also delivers the strongest results on several benchmarks, including MME, SQA-I, TextVQA, and MM-Vet, while remaining highly competitive across the remaining evaluation suite. Notably, MAGIC slightly surpasses full-data finetuning in aggregate relative performance despite using only 20% of the training data, suggesting that it effectively removes redundant or weakly informative samples while retaining high-value multimodal supervision. Overall, these results show that MAGIC provides a superior coreset construction strategy among both score-based and model-involved baselines. We further visualize top selected and bottom rejected samples in Fig. 4 to show the data qualitative preference induced by MAGIC, highlighting that the method favors samples with stronger multimodal utility and grounding while rejecting samples that are less visually informative or behaviorally redundant.

<table><tr><td>Method</td><td>VQAv2</td><td>GQA</td><td>SQA-I</td><td>TextVQA</td><td>POPE</td><td>MME</td><td>MMBench en</td><td>en</td><td>en</td><td>LLaVA-W Bench</td><td>MM-Vet</td><td>Rel. (%)</td></tr><tr><td>Full-Finetune</td><td>80.0</td><td>63.3</td><td>71.2</td><td>60.2</td><td>86.7</td><td>1541.7</td><td>68.5</td><td>61.5</td><td>69.5</td><td>36.1</td><td>100.0</td><td></td></tr><tr><td>Random</td><td>76.7</td><td>60.5</td><td>68.8</td><td>57.7</td><td>84.8</td><td>1484.9</td><td>62.8</td><td>55.2</td><td>68.6</td><td>35.5</td><td>95.7</td><td></td></tr><tr><td>CLIP-Score</td><td>75.3</td><td>52.6</td><td>69.7</td><td>57.3</td><td>85.4</td><td>1426.3</td><td>60.4</td><td>54.0</td><td>68.1</td><td>-</td><td>92.8</td><td></td></tr><tr><td>EL2N</td><td>77.2</td><td>59.6</td><td>69.9</td><td>56.1</td><td>84.1</td><td>1531.0</td><td>59.3</td><td>52.3</td><td>65.8</td><td>-</td><td>93.8</td><td></td></tr><tr><td>Perplexity</td><td>77.0</td><td>58.5</td><td>68.7</td><td>54.8</td><td>83.1</td><td>1508.8</td><td>57.5</td><td>50.3</td><td>68.7</td><td>-</td><td>92.7</td><td></td></tr><tr><td>SemDedup</td><td>75.6</td><td>57.5</td><td>70.5</td><td>57.7</td><td>85.3</td><td>1397.6</td><td>59.0</td><td>51.1</td><td>68.7</td><td>-</td><td>93.0</td><td></td></tr><tr><td>D2-Pruning</td><td>73.9</td><td>60.5</td><td>70.4</td><td>55.2</td><td>84.9</td><td>1463.0</td><td>67.3</td><td>59.9</td><td>66.5</td><td>-</td><td>95.9</td><td></td></tr><tr><td>Self-Sup</td><td>76.3</td><td>60.5</td><td>70.2</td><td>52.7</td><td>85.4</td><td>1463.8</td><td>63.7</td><td>57.6</td><td>64.9</td><td>-</td><td>94.5</td><td></td></tr><tr><td>Self-Filter</td><td>75.0</td><td>59.8</td><td>69.5</td><td>55.8</td><td>84.5</td><td>1446.9</td><td>58.8</td><td>51.8</td><td>69.1</td><td>-</td><td>93.3</td><td></td></tr><tr><td>COINCIDE</td><td>77.8</td><td>60.4</td><td>70.0</td><td>58.6</td><td>87.1</td><td>1516.8</td><td>64.0</td><td>57.7</td><td>67.4</td><td>-</td><td>96.8</td><td></td></tr><tr><td>MAGIC (Ours)</td><td>78.9</td><td>60.4</td><td>73.5</td><td>58.8</td><td>86.4</td><td>1551.5</td><td>67.4</td><td>61.3</td><td>69.0</td><td>36.3</td><td>99.3</td><td></td></tr></table>

Table 3: Transferring to the larger target model. We validate if the coresets selected from TinyLLaVA-2B are transferable to LLaVA-1.5-13B finetuning, and measure performance on various multimodal benchmarks.

<table><tr><td></td><td>AI2D</td><td>ChartQA</td><td>DocVQA</td><td>InfoVQA</td><td>MMVet</td><td>Naturalbench</td><td>RealworldQA</td><td>CMMMU</td><td>Rel. (%)</td></tr><tr><td>Full-Finetune</td><td>55.4</td><td>17.5</td><td>28.9</td><td>26.5</td><td>31.1</td><td>12.4</td><td>52.4</td><td>22.1</td><td>100.0</td></tr><tr><td>Random</td><td>50.2</td><td>15.1</td><td>25.2</td><td>24.3</td><td>29.6</td><td>11.1</td><td>49.8</td><td>21.9</td><td>91.8</td></tr><tr><td>CLIP-Score</td><td>52.0</td><td>16.4</td><td>27.1</td><td>24.9</td><td>29.2</td><td>11.6</td><td>49.3</td><td>20.8</td><td>93.9</td></tr><tr><td>EL2N</td><td>52.1</td><td>17.9</td><td>26.9</td><td>26.4</td><td>29.6</td><td>12.5</td><td>51.9</td><td>21.7</td><td>97.8</td></tr><tr><td>Perplexity</td><td>53.0</td><td>16.4</td><td>27.1</td><td>27.1</td><td>29.4</td><td>11.4</td><td>50.3</td><td>23.9</td><td>97.0</td></tr><tr><td>SemDeDup</td><td>53.2</td><td>16.4</td><td>26.4</td><td>26.6</td><td>29.1</td><td>13.2</td><td>51.1</td><td>22.8</td><td>97.8</td></tr><tr><td>D2-Pruning</td><td>52.1</td><td>17.3</td><td>26.5</td><td>24.7</td><td>33.1</td><td>11.3</td><td>52.0</td><td>21.9</td><td>96.7</td></tr><tr><td>Self-Sup</td><td>52.7</td><td>16.6</td><td>27.4</td><td>25.2</td><td>29.6</td><td>11.8</td><td>50.1</td><td>21.0</td><td>95.0</td></tr><tr><td>Self-Filter</td><td>52.1</td><td>16.0</td><td>27.1</td><td>25.0</td><td>29.3</td><td>11.5</td><td>49.5</td><td>20.6</td><td>94.0</td></tr><tr><td>COINCIDE</td><td>53.7</td><td>17.0</td><td>27.9</td><td>26.0</td><td>30.2</td><td>12.1</td><td>50.9</td><td>21.4</td><td>97.2</td></tr><tr><td>RDS</td><td>53.0</td><td>16.8</td><td>27.5</td><td>25.4</td><td>29.8</td><td>12.0</td><td>50.3</td><td>21.3</td><td>95.6</td></tr><tr><td>ICONS</td><td>53.9</td><td>17.1</td><td>27.9</td><td>27.5</td><td>29.7</td><td>12.8</td><td>55.0</td><td>25.2</td><td>101.6</td></tr><tr><td>MAGIC (ours)</td><td>54.9</td><td>17.1</td><td>27.2</td><td>26.9</td><td>29.8</td><td>13.2</td><td>54.5</td><td>25.3</td><td>101.7</td></tr><tr><td>Per-task Rel. (%)</td><td>99.1</td><td>97.7</td><td>94.1</td><td>101.5</td><td>95.8</td><td>106.5</td><td>104.0</td><td>114.5</td><td>-</td></tr></table>

Table 4: Detailed results of unseen task generalization. Performance comparison on unseen benchmarks when trained on selected 20% subsets. Notably, we observe improvements on InfoVQA (101.5%), RealWorldQA (104.0%), and CMMMU (114.5%), highlighting strong generalization to unseen tasks.

Main results on Vision-Flan. Table 2 reports coreset selection results on Vision-Flan-186K under a 20% budget. MAGIC achieves a relative score of 101.6%, outperforming random selection by a large margin (+9.8 points), even exceeding full-data finetuning. These gains are consistent across several benchmarks, including GQA, SQA-I, POPE, MME, and MMBench-cn, indicating that the selected subset preserves not only overall utility but also strong cross-task generalization. This result suggests that the combination of multimodal utility, grounding strength, and behavioral coverage can identify compact training subsets that match or exceed the effectiveness of the full dataset.

Transfer to a Larger Target Model. To test whether the selected coresets transfer when the target model has a much larger backbone than the reference model, we finetune LLaVA-1.5-13B on the 20% subsets selected from TinyLLaVA-2B. As shown in Table 3, MAGIC achieves the strongest overall transfer performance, reaching 99.3% relative performance to full-finetuning. MAGIC attains the best or second-best performance across all benchmarks, showing that the proposed criteria identify samples whose value is not tied to a single model scale and that MAGIC captures broadly useful multimodal training signals rather than overfitting to the inductive biases of the reference model.

Generalization to Unseen Tasks. We further evaluate whether the selected subsets support generalization beyond the benchmarks seen during coreset construction. More details on these datasets are provided in App. Sec. A.5. Tab. 4 shows that MAGIC achieves the best overall unseen-task relative score (101.7%), surpassing prior baselines. In particular, MAGIC yields strong relative improvements on InfoVQA (101.5%), NaturalBench (106.5%), RealWorldQA (104.0%), and CMMMU (114.5%). While performance on a few tasks remains slightly below full-data finetuning, the overall pattern suggests that preserving activation-level behavioral diversity improves robustness to task shift. This supports our central hypothesis that coreset construction should account not only for sample quality, but also for diversity in the latent computations induced by the selected data.

<table><tr><td>Method</td><td>End-to-End Cost (GPU-hr) ↓</td><td>Selection Cost (GPU-hr) ↓</td><td>Finetuning Cost (GPU-hr) ↓</td><td>External API</td><td>Rel. Perf. (%) ↑</td></tr><tr><td>Full-Finetune</td><td>76.0</td><td>-</td><td>68.0</td><td>No</td><td>100.0</td></tr><tr><td>COIDO</td><td>35.0</td><td>25.0</td><td>10.0</td><td>Yes</td><td>98.2</td></tr><tr><td>Self-Filter</td><td>81.0</td><td>71.0</td><td>10.0</td><td>No</td><td>90.5</td></tr><tr><td>ICONS</td><td>66.0</td><td>50.0</td><td>10.0</td><td>No</td><td>98.4</td></tr><tr><td>DataTailor</td><td>33.5</td><td>23.5</td><td>10.0</td><td>No</td><td>85.9</td></tr><tr><td>COINCIDE</td><td>66.5</td><td>55.5</td><td>10.0</td><td>No</td><td>97.4</td></tr><tr><td>MAGIC (Ours)</td><td>20.0</td><td>10.0</td><td>10.0</td><td>No</td><td>100.3</td></tr></table>

Table 5: End-to-end efficiency comparison. MAGIC achieves the best trade-off between compute cost and performance, matching or exceeding full-data performance while requiring the lowest overall compute.

<table><tr><td>Variant</td><td>Rel. (%) ↑</td><td>POPE ↑</td><td>GQA ↑</td></tr><tr><td>MAGIC (full)</td><td>100.0</td><td>87.0</td><td>59.9</td></tr><tr><td>w/o BR Score</td><td>98.4</td><td>86.7</td><td>58.3</td></tr><tr><td>w/o MG Score</td><td>98.1</td><td>86.7</td><td>58.1</td></tr><tr><td>w/o SN Score</td><td>98.3</td><td>86.8</td><td>57.6</td></tr></table>

Table 6: Ablation study of MAGIC. Removing any component degrades performance.

<table><tr><td>Hyperparameter</td><td>Setting</td><td>SQA ↑</td><td>POPE ↑</td></tr><tr><td rowspan="5">MG/BR weights</td><td>(1.0, 0.0)</td><td>69.8</td><td>86.7</td></tr><tr><td>(0.7, 0.3)</td><td>71.8</td><td>87.0</td></tr><tr><td>(0.5, 0.5)</td><td>72.3</td><td>87.1</td></tr><tr><td>(0.3, 0.7)</td><td>70.9</td><td>86.9</td></tr><tr><td>(0.0, 1.0)</td><td>71.2</td><td>86.8</td></tr><tr><td rowspan="4">top- $(k_{\ell})$ </td><td>[1,1,1,1]</td><td>71.8</td><td>86.8</td></tr><tr><td>[1,1,2,2]</td><td>71.1</td><td>86.9</td></tr><tr><td>[1,1,2,3]</td><td>73.5</td><td>87.0</td></tr><tr><td>[2,3,4,6]</td><td>70.2</td><td>85.5</td></tr></table>

Table 7: Hyperparameter sensitivity of MAGIC.

# 5 Ablations

Cost of Coreset Selection Table 5 compares end-to-end efficiency across data selection methods. MAGIC achieves the strongest computeperformance trade-off, amongst all baselines. At the same time, MAGIC attains the highest relative performance (100.3%). Notably, MAGIC also avoids external API usage, unlike COIDO. These results highlight the scalability and effectiveness of our proposed design.

Component Analysis. Table 6 evaluates the contribution of each component in MAGIC. Removing either the BR signal, the multimodal-gain eligibility filter, or the SN-based behavioral bucketing consistently degrades performance. These results confirm that the three components are complementary: multimodal gain filters out weakly visual examples, bridging relevance favors strongly grounded samples, and skill-signature bucketing preserves coverage over distinct behavioral modes.

Hyperparameter Sensitivity. Table 7 evaluates sensitivity to the MG/BR weighting and the perlayer signature size $k _ { \ell } ,$ which specifies how many top-activated neurons are retained from each selected layer. A balanced $( \alpha , \beta )$ weighting works best: (0.5, 0.5) achieves the strongest performance, indicating that multimodal utility and grounding strength are most effective when combined. For the signature design, intermediate granularity performs best: [1, 1, 2, 3], which retains 1, 1, 2, and 3 neurons from the four selected layers, yields the strongest overall score, which aligns with the experimental settings MAGIC uses. Smaller signatures under-represent latent behavior, while larger ones make buckets overly specific and less robust by fragmenting samples into overly fine groups.

# 6 Conclusion

We presented MAGIC, a training-free and forwardonly coreset selection method for visual instruction tuning. MAGIC combines three intrinsic signals from a pretrained VLM – Multimodal Gain, Bridging Relevance, and Skill-Neuron Signatures – to jointly model multimodal utility, grounding strength, and activation-level behavioral diversity. Comprehensive experiments on the LLaVA-1.5 and Vision-Flan datasets demonstrate that MAGIC consistently achieves stronger or comparable performance to full-data finetuning while using only 20% of the data, with the lowest data selection cost, showcasing its effectiveness and efficiency. These results show that high-value multimodal coresets should not be defined solely by scalar difficulty or redundancy, but also by the diversity of latent computations they elicit.

# 7 Limitations

MAGIC has a few limitations. First, like most VLM coreset selection methods, our score extraction depends on a pretrained reference LVLM, so selection quality may vary with the choice of reference model. That said, our transfer results suggest that the extracted signals remain useful beyond the exact model used for scoring. Second, MAGIC requires dataset-wide forward passes to extract selection signals. However, this cost is very modest compared to gradient-based or auxiliary-training methods, since MAGIC avoids backpropagation, scorer training, and iterative optimization. In practice, this yields a favorable efficiency-performance trade-off: MAGIC remains simple, training-free, and consistently effective across datasets and transfer settings.

# 8 Ethical Considerations and Broader Impact

MAGIC is designed to improve the efficiency of multimodal instruction tuning by selecting compact yet behaviorally representative coresets for large vision-language models. By reducing the amount of training data required for finetuning, MAGIC can lower computational cost, energy consumption, and environmental impact, enabling more sustainable multimodal model development.

Because MAGIC operates on large-scale webderived multimodal datasets, the selected subsets may still reflect demographic biases, unsafe content, or skewed visual-textual associations present in the original data. Although our bucket-aware balancing strategy improves coverage across diverse latent behaviors and reduces over-concentration on dominant modes, it does not explicitly enforce fairness or safety constraints. Nevertheless, the structured and diversity-aware design of MAGIC provides a promising foundation for incorporating fairness-aware or safety-aware objectives, which are beyond the scope of this work, into future coreset selection methods.

Finally, while more efficient coreset selection can broaden access to multimodal model training, responsible deployment remains important. We encourage practitioners to combine methods such as MAGIC with dataset auditing, safety filtering, and appropriate governance practices to support reliable and responsible use of large vision-language models.

# 9 AI Writing Statement

This paper utilized AI assistance for language polishing of the manuscript, including vocabulary correction and spell checking.

# References

Amro Abbas, Kushal Tirumala, Dániel Simig, Surya Ganguli, and Ari S Morcos. 2023. Semdedup: Dataefficient learning at web-scale through semantic deduplication. arXiv preprint arXiv:2303.09540.   
Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, and 1 others. 2022. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems, 35:23716– 23736.   
Ruibo Chen, Yihan Wu, Lichang Chen, Guodong Liu, Qi He, Tianyi Xiong, Chenxi Liu, Junfeng Guo, and Heng Huang. 2024. Your vision-language model itself is a strong filter: Towards high-quality instruction tuning with data selection. In Findings of the Association for Computational Linguistics: ACL 2024, pages 4156–4172.   
Damai Dai, Li Dong, Yaru Hao, Zhifang Sui, Baobao Chang, and Furu Wei. 2022. Knowledge neurons in pretrained transformers. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8493– 8502.   
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning. Advances in neural information processing systems, 36:49250–49267.   
Zhiwei Deng, Tao Li, and Yang Li. 2024. Influential language data selection via gradient trajectory pursuit. arXiv preprint arXiv:2410.16710.   
Qianlong Du, Chengqing Zong, and Jiajun Zhang. 2023. Mods: Model-oriented data selection for instruction tuning. arXiv preprint arXiv:2311.15653.   
Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, and 1 others. 2023. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394.   
Mor Geva, Avi Caciularu, Kevin Wang, and Yoav Goldberg. 2022. Transformer feed-forward layers build predictions by promoting concepts in the vocabulary space. In Proceedings of the 2022 conference on empirical methods in natural language processing, pages 30–45.

Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. 2021. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 5484–5495.   
Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. 2017. Making the v in vqa matter: Elevating the role of image understanding in visual question answering. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6904–6913.   
Danna Gurari, Qing Li, Abigale J Stangl, Anhong Guo, Chi Lin, Kristen Grauman, Jiebo Luo, and Jeffrey P Bigham. 2018. Vizwiz grand challenge: Answering visual questions from blind people. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3608–3617.   
Zayd Hammoudeh and Daniel Lowd. 2024. Training data influence analysis and estimation: A survey. Machine Learning, 113(5):2351–2403.   
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, and 1 others. 2022. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3.   
Drew A Hudson and Christopher D Manning. 2019. Gqa: A new dataset for real-world visual reasoning and compositional question answering. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6700–6709.   
Hamish Ivison, Muru Zhang, Faeze Brahman, Pang Wei Koh, and Pradeep Dasigi. 2025. Large-scale data selection for instruction tuning. arXiv preprint arXiv:2503.01807.   
Aniruddha Kembhavi, Mike Salvato, Eric Kolve, Minjoon Seo, Hannaneh Hajishirzi, and Ali Farhadi. 2016. A diagram is worth a dozen images. In European conference on computer vision, pages 235–251. Springer.   
Jaewoo Lee, Boyang Li, and Sung Ju Hwang. 2024. Concept-skill transferability-based data selection for large vision-language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 5060–5080.   
Baiqi Li, Zhiqiu Lin, Wenxuan Peng, Jean de Dieu Nyandwi, Daniel Jiang, Zixian Ma, Simran Khanuja, Ranjay Krishna, Graham Neubig, and Deva Ramanan. 2024. Naturalbench: Evaluating vision-language models on natural adversarial samples. Advances in Neural Information Processing Systems, 37:17044– 17068.   
Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi. 2023a. Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In International conference on machine learning, pages 19730–19742. PMLR.

Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023b. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 292–305.   
Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024a. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. Advances in neural information processing systems, 36:34892– 34916.   
Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, and 1 others. 2024b. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer.   
Zikang Liu, Kun Zhou, Wayne Xin Zhao, Dawei Gao, Yaliang Li, and Ji-Rong Wen. 2025. Less is more: High-value data selection for visual instruction tuning. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 3712–3721.   
Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan. 2022. Learn to explain: Multimodal reasoning via thought chains for science question answering. Advances in neural information processing systems, 35:2507–2521.   
Adyasha Maharana, Prateek Yadav, and Mohit Bansal. 2023. D2 pruning: Message passing for balancing diversity and difficulty in data pruning. arXiv preprint arXiv:2310.07931.   
Max Marion, Ahmet Üstün, Luiza Pozzobon, Alex Wang, Marzieh Fadaee, and Sara Hooker. 2023. When less is more: Investigating data pruning for pretraining llms at scale. arXiv preprint arXiv:2309.04564.   
Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of the association for computational linguistics: ACL 2022, pages 2263– 2279.   
Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. 2022. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706.   
Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209.

Sören Mindermann, Jan M Brauner, Muhammed T Razzak, Mrinank Sharma, Andreas Kirsch, Winnie Xu, Benedikt Höltgen, Aidan N Gomez, Adrien Morisot, Sebastian Farquhar, and 1 others. 2022. Prioritized training on points that are learnable, worth learning, and not yet learnt. In International Conference on Machine Learning, pages 15630–15649. PMLR.   
Mansheej Paul, Surya Ganguli, and Gintare Karolina Dziugaite. 2021. Deep learning on a data diet: Finding important examples early in training. Advances in neural information processing systems, 34:20596– 20607.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, and 1 others. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR.   
Tanik Saikh, Tirthankar Ghosal, Amish Mittal, Asif Ekbal, and Pushpak Bhattacharyya. 2022. Scienceqa: A novel resource for question answering on scholarly articles. International Journal on Digital Libraries, 23(3):289–301.   
Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 8317–8326.   
Ben Sorscher, Robert Geirhos, Shashank Shekhar, Surya Ganguli, and Ari Morcos. 2022. Beyond neural scaling laws: beating power law scaling via data pruning. Advances in Neural Information Processing Systems, 35:19523–19536.   
Haoru Tan, Sitong Wu, Fei Du, Yukang Chen, Zhibin Wang, Fan Wang, and Xiaojuan Qi. 2023. Data pruning via moving-one-sample-out. Advances in neural information processing systems, 36:18251–18262.   
Shengbang Tong, Ellis Brown, Penghao Wu, Sanghyun Woo, Manoj Middepogu, Sai C Akula, Jihan Yang, Shusheng Yang, Adithya Iyer, Xichen Pan, and 1 others. 2024. Cambrian-1: A fully open, visioncentric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310– 87356.   
Xindi Wu, Mengzhou Xia, Rulin Shao, Zhiwei Deng, Pang Wei Koh, and Olga Russakovsky. 2024. Icons: Influence consensus for vision-language data selection. arXiv preprint arXiv:2501.00654.   
xAI. 2024. Grok-1.5 vision preview. https://x.ai/ news/grok-1.5v. Online; accessed 14-November-2024.   
Mengzhou Xia, Sadhika Malladi, Suchin Gururangan, Sanjeev Arora, and Danqi Chen. 2024. Less: Selecting influential data for targeted instruction tuning. arXiv preprint arXiv:2402.04333.

Zhiyang Xu, Chao Feng, Rulin Shao, Trevor Ashby, Ying Shen, Di Jin, Yu Cheng, Qifan Wang, and Lifu Huang. 2024. Vision-flan: Scaling human-labeled tasks in visual instruction tuning. In Findings of the Association for Computational Linguistics: ACL 2024, pages 15271–15342.   
Yichen Yan, Ming Zhong, Qi Zhu, Xiaoling Gu, Jinpeng Chen, and Huan Li. 2025. Coido: Efficient data selection for visual instruction tuning via coupled importance-diversity optimization. arXiv preprint arXiv:2510.17847.   
Qifan Yu, Juncheng Li, Yu Wu, Siliang Tang, Wei Ji, and Yueting Zhuang. 2023a. Visually-prompted language model for fine-grained scene graph generation in an open world. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 21560–21571.   
Qifan Yu, Zhebei Shen, Zhongqi Yue, Yang Wu, Bosheng Qin, Wenqiao Zhang, Yunfei Li, Juncheng Li, Siliang Tang, and Yueting Zhuang. 2025. Mastering collaborative multi-modal data selection: A focus on informativeness, uniqueness, and representativeness. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 155–165.   
Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. 2023b. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490.   
Ge Zhang, Xinrun Du, Bei Chen, Yiming Liang, Tongxu Luo, Tianyu Zheng, Kang Zhu, Yuyang Cheng, Chunpu Xu, Shuyue Guo, and 1 others. 2024. Cmmmu: A chinese massive multi-discipline multimodal understanding benchmark. arXiv preprint arXiv:2401.11944.   
Baichuan Zhou, Ying Hu, Xi Weng, Junlong Jia, Jie Luo, Xien Liu, Ji Wu, and Lei Huang. 2024. Tinyllava: A framework of small-scale large multimodal models. arXiv preprint arXiv:2402.14289.   
Chunting Zhou, Pengfei Liu, Puxin Xu, Srinivasan Iyer, Jiao Sun, Yuning Mao, Xuezhe Ma, Avia Efrat, Ping Yu, Lili Yu, and 1 others. 2023. Lima: Less is more for alignment. Advances in Neural Information Processing Systems, 36:55006–55021.   
Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny. 2023. Minigpt-4: Enhancing vision-language understanding with advanced large language models. arXiv preprint arXiv:2304.10592.

# A Appendix

This appendix provides supplementary details for the main paper. We introduce the task of visual instruction-tuning, include implementation and experimental details, benchmark descriptions and task statistics, the full MAGIC data selection algorithm, additional coverage analysis in the MAGIC feature space, and qualitative examples of selected and rejected samples. These materials complement the main results by offering further detail on the methodology and additional evidence for the effectiveness of MAGIC.

# A.1 Visual Instruction Tuning

Recent vision-language models (VLMs), such as Flamingo (Alayrac et al., 2022), LLaVA (Liu et al., 2024a), BLIP-2 (Li et al., 2023a), and Cambrian (Tong et al., 2024), have substantially advanced multimodal intelligence by coupling visual perception with large language reasoning. Their downstream effectiveness relies heavily on visual instruction tuning (VIT) (Liu et al., 2023), a post-pretraining adaptation stage that finetunes the model on image-grounded instruction-response tuples. By exposing the model to diverse multimodal task formulations under natural language supervision, VIT improves instruction adherence, strengthens cross-modal alignment, and enables the model to operate as a general-purpose multimodal assistant (Liu et al., 2023). In parallel with the expansion of multimodal application domains, VIT datasets have grown to massive scale, now routinely comprising hundreds of thousands to millions of annotated instruction examples (Liu et al., 2024a; Tong et al., 2024).

# A.2 Datasets & model

For our experiments, we use the LLaVA-v1.5 checkpoint after Stage 1 pre-training for feature alignment, following the original LLaVA training pipeline, with the 7B model size and LLaVA-665K setting unless otherwise specified. Concretely, we use the checkpoint1, which represents the model after projector training and before any Stage 2 visual instruction tuning. Therefore, the checkpoint has no prior exposure to the LLaVA-665K visual instruction tuning data before data selection.

# A.3 Details on Experimental Settings

Unless otherwise specified, all coreset selection methods use a fixed selection budget of 20% of the original training set. For LLaVA-665K, this corresponds to 133,059 selected samples, while for Vision-Flan-186K it corresponds to approximately 37K samples. For all experiments, we use TinyLLaVA-2B as the reference model for score extraction and LLaVA-1.5-7B as the primary target model for finetuning; transfer experiments further evaluate the selected coresets on LLaVA-1.5-13B.

For score extraction in MAGIC, we use the pretrained LLaVA-v1.5 checkpoint after Stage 1 feature alignment, before any visual instruction tuning, so that the reference model has not previously seen the downstream VIT training data. We extract both Bridging Relevance and Skill-Neuron signals from the same four uniformly spaced transformer layers,

$$
\mathcal {L} _ {\mathrm{BR}} = \mathcal {L} _ {\mathrm{SN}} = \{8, 1 2, 1 6, 2 0 \},
$$

taking inspiration from prior work (Lee et al., 2024). We choose these layers to capture a broad spectrum of multimodal computations across early, middle, and late stages of the model, while keeping feature extraction efficient. Earlier layers tend to reflect lower-level alignment and local grounding signals, middle layers capture richer cross-modal interaction patterns, and higher layers encode more task-specific semantic reasoning. Using multiple layers distributed from lower to upper depths therefore provides a more stable view of both answer-tovision grounding and latent behavioral activation than relying on a single layer alone. At the same time, restricting extraction to four layers keeps the procedure efficient and avoids the redundancy and overhead of using the entire network. Score extraction is performed with a per-device batch size of 2, and the raw SN extraction retains the top 64 activated neurons per selected layer before the later signature compression step used in data selection.

For data selection, we use a target coreset size of 133,059, corresponding to a 20% subset of LLaVA-665K. We first apply a multimodal-gain eligibility filter with keep ratio $\rho = 0 . 6$ , then form a quality shortlist with factor $\eta \ : = \ : 2 . 0$ . Unless otherwise specified, the joint quality score uses weights $( \alpha , \beta ) = ( 0 . 5 , 0 . 5 )$ for MG and BR, respectively. For skill-signature bucketing, we use the per-layer signature configuration

$$
[ k _ {\ell} ] _ {\ell \in \mathcal {L}} = [ 1, 1, 2, 3 ],
$$

meaning that we retain 1, 1, 2, and 3 neurons from the four selected layers, respectively. Bucket allocation uses temperature τ = 0.2 and a maximum bucket fraction of $\gamma = 0 . 0 5$ . We then perform bucket-wise top-score selection followed by quota correction and backfilling as described in Algorithm 2.

For target-model finetuning, we follow the official LLaVA-1.5 recipe and train with LoRA for one epoch using the standard hyperparameter settings. All experiments are conducted on 4 NVIDIA A100 GPUs for a single run.

# A.4 Details on Evaluation Benchmarks.

We evaluate on a diverse collection of multimodal benchmarks that capture complementary capabilities of VLMs, including multiple-choice reasoning, visual question answering, text understanding in images, scientific reasoning, open-ended generation, and factual consistency. Specifically, VQAv2 (Goyal et al., 2017) measures open-ended question answering over natural images, while GQA (Hudson and Manning, 2019) emphasizes compositional reasoning over object attributes and relations. VizWiz (Gurari et al., 2018) evaluates robustness on real-world images captured by visually impaired users, and TextVQA (Singh et al., 2019) focuses on reading and reasoning over scene text. ScienceQA (Saikh et al., 2022) assesses scienceoriented reasoning grounded in visual content. POPE (Li et al., 2023b) is used to quantify object hallucination, whereas MME (Fu et al., 2023) evaluates perception and cognition through a collection of binary-choice subtasks. MMBench (Liu et al., 2024b) provides a broad testbed covering abilities such as recognition, OCR, and relational reasoning. In addition, LLaVA-W Bench (Lu et al., 2022) evaluates open-ended visual instruction-following behavior, and MM-Vet (Yu et al., 2023b) measures a wide range of vision-language skills, including recognition, OCR, knowledge, spatial reasoning, and mathematical understanding. More details on the task statistics are shown in Tab. 8.

# A.5 Unseen Tasks

We additionally evaluate generalization to unseentask benchmarks that probe transfer beyond the training-aligned evaluation suite. AI2D (Kembhavi et al., 2016) measures diagram understanding and visual reasoning over structured scientific illustrations. ChartQA (Masry et al., 2022) evaluates reasoning over charts and plots, requiring models to interpret visualized quantitative information. DocVQA (Mathew et al., 2021) tests document understanding from visually rich document images, while InfoVQA (Mathew et al., 2022) focuses on question answering over infographics that combine text, layout, and graphical elements. NaturalBench (Li et al., 2024) is designed to assess robust multimodal reasoning under more natural and diverse real-world inputs. RealWorldQA (xAI, 2024) further examines practical visual question answering ability on real-world images. Finally, CM-MMU (Zhang et al., 2024) extends evaluation to a broad, college-level multimodal benchmark that tests knowledge-intensive reasoning across diverse subjects. Together, these unseen tasks provide a complementary measure of out-of-distribution generalization and task transfer.

<table><tr><td rowspan="2">Task</td><td rowspan="2">MME</td><td rowspan="2">POPE</td><td rowspan="2">SQA-I</td><td colspan="2">MMBench</td></tr><tr><td>en</td><td>cn</td></tr><tr><td> $|\mathcal{D}_{\text{val}}|$ </td><td>986</td><td>500</td><td>424</td><td>1,164</td><td>1,164</td></tr><tr><td> $|\mathcal{D}_{\text{test}}|$ </td><td>2,374</td><td>8,910</td><td>4,241</td><td>1,784</td><td>1,784</td></tr><tr><td>Task Type</td><td>Y/N</td><td>Y/N</td><td>MCQ</td><td>MCQ</td><td>MCQ</td></tr><tr><td>Task</td><td>VQAv2</td><td>GQA</td><td>VizWiz</td><td>TextVQA</td><td>LLaVA-W</td></tr><tr><td> $|\mathcal{D}_{\text{val}}|$ </td><td>1,000</td><td>398</td><td>800</td><td>84</td><td>84</td></tr><tr><td> $|\mathcal{D}_{\text{test}}|$ </td><td>36,807</td><td>12,578</td><td>8,000</td><td>5,000</td><td>84</td></tr><tr><td>Task Type</td><td>VQA</td><td>VQA</td><td>VQA</td><td>VQA</td><td>VQA</td></tr></table>

Table 8: Statistics of Target Tasks. Our target tasks include diverse benchmarks and answer formats, covering different vision-language capabilities. Task types include Multiple-Choice Questions (MCQ), Visual Question Answering (VQA), and Yes/No Questions (Y/N).

# A.6 Coverage in the MAGIC Feature Space

To examine whether MAGIC preserves broad coverage of the training distribution, we visualize the joint MAGIC feature space using UMAP in Figure 3. Each point corresponds to one training sample embedded from its combined MAGIC descriptor, formed from Multimodal Gain (MG), Bridging Relevance (BR), and the processed Skill-Neuron (SN) signature. The orange points denote the full dataset, while the green points denote the selected coreset.

Two observations are notable. First, the selected coreset remains well distributed across the global support of the full dataset rather than collapsing onto a small number of dense regions. This suggests that MAGIC does not behave like a purely score-based filter that concentrates only on a narrow subset of samples. Second, the coreset also preserves many of the local structures present in the full dataset, indicating that the proposed skillsignature bucketing retains diverse latent behavioral modes while still favoring high-quality samples. Together, these results provide qualitative evidence that MAGIC preserves feature coverage in the multimodal training space while constructing a substantially smaller subset.

Algorithm 2 MAGIC Data Selection   
Input: dataset $D = \{x_i\}_{i=1}^N$ , multimodal gain $\{g_i\}_{i=1}^N$ , bridging relevance $\{b_i\}_{i=1}^N$ , skill signatures $\{\phi_i\}_{i=1}^N$ , target budget M, keep ratio $\rho$ , shortlist factor $\eta$ , weights $(\alpha, \beta)$ , temperature $\tau$ , bucket cap $\gamma$ Output: coreset C

1: $\hat{g}_i \leftarrow \text{Norm}(g_i)$ , $\hat{b}_i \leftarrow \text{Norm}(b_i)$ , $i = 1, \ldots, N$ 2: $q_i \leftarrow \alpha \hat{g}_i + \beta \hat{b}_i$ , $i = 1, \ldots, N$ 3: $E \leftarrow \text{Top}_{\lceil \rho N \rceil}(\{g_i\}_{i=1}^N)$ 4: $S \leftarrow \text{Top}_{\min(\lceil \eta M \rceil, |\mathcal{E}|)}(\{q_i\}_{i \in \mathcal{E}})$ 5: Partition S into buckets $B = \{B_1, \ldots, B_K\}$ by $\phi_i$ 6: $m_j \leftarrow \sum_{i \in B_j} \exp(q_i / \tau)$ , $j = 1, \ldots, K$ 7: $p_j \leftarrow m_j / \sum_{r=1}^K m_r$ , $j = 1, \ldots, K$ 8: $n_j \leftarrow \min(|B_j|, \lceil \gamma M \rceil, \lfloor Mp_j \rfloor)$ , $j = 1, \ldots, K$ 9: Redistribute $M - \sum_{j=1}^K n_j$ by descending fractional remainder of $Mp_j$ 10: $C \leftarrow \bigcup_{j=1}^K \text{Top}_{n_j}(\{q_i\}_{i \in B_j})$ 11: Backfill from S \ C, then E \ C, by descending $q_i$ until $|C| = M$ 12: return C

![](images/88927db69676afd2bc94d50164fcba73d8d51b7a4e75742ef2cf6839520264f7.jpg)

<details>
<summary>scatter</summary>

| UMAP Dimension 1 | UMAP Dimension 2 | Dataset          |
| ---------------- | ---------------- | ---------------- |
| -10              | 5                | Full Dataset     |
| -5               | 10               | Selected Coreset |
| 0                | 15               | Full Dataset     |
| 5                | 20               | Selected Coreset |
| 10               | 15               | Full Dataset     |
| 15               | 10               | Selected Coreset |
| 20               | 5                | Full Dataset     |
| -10              | -5               | Selected Coreset |
| -5               | -10              | Full Dataset     |
| 0                | -5               | Selected Coreset |
| 5                | 0                | Full Dataset     |
| 10               | 5                | Selected Coreset |
| 15               | 10               | Full Dataset     |
| 20               | 15               | Selected Coreset |
</details>

Figure 3: UMAP visualization of the MAGIC feature space. Each point represents one training sample embedded from its joint MAGIC descriptor, formed from Multimodal Gain (MG), Bridging Relevance (BR), and the processed Skill-Neuron (SN) signature. MAGIC preserves broad global coverage and local behavioral structure while selecting a compact subset.

# A.7 Qualitative Examples of Selected and Rejected Samples

Figure 4 provides qualitative examples of the samples prioritized and rejected by MAGIC. The left two examples are drawn from the top of the MAGIC ranking and illustrate the types of training instances favored by our selection criterion. These samples require genuine visual reasoning: the first

example asks the model to infer the country from a road sign, which depends on recognizing localized textual and structural cues in the image, while the second requires grounded regional description from a specified image crop. Such examples are visually informative, strongly grounded, and behaviorally non-trivial, and therefore tend to receive high multimodal gain and bridging relevance scores.

In contrast, the right two examples are drawn from the bottom of the ranking and illustrate the types of samples that MAGIC deprioritizes. These examples are either largely recoverable from language priors or only weakly dependent on the image content. For instance, the question about the “scientific name for the area protected by the pads” can be answered primarily from textual world knowledge, while the question about the Children’s Online Privacy Protection Act is unrelated to the visual evidence altogether. Such samples provide limited multimodal supervision and are therefore less useful for efficient visual instruction tuning.

Overall, these examples highlight that MAGIC favors samples where the image materially contributes to the answer and where the required computation is both grounded and behaviorally distinctive.

![](images/d03791b260b7b63173aadd8e5dcf257d8fe51ee701effec44c6d3f98cbf17b52.jpg)

<details>
<summary>text_image</summary>

Top Shortlisted Samples
<image> This sign for the airport is most likely in which country?
A. france
B. switzerland
C. germany
D. norway
Answer with the option's letter from the given choices directly.
C
Water Glass
Bottom Rejected Samples
<image> What is the scientific name for the area protected by the pads?
A. ulna
B. clavicle
C. mandible
D. patella
Answer with the option's letter from the given choices directly.
D
<image> When does the Children's Online Privacy Protection Act took effect in?
A. may 2000
B. aug 1990
C. sep 1999
D. apr 2000
Answer with the option's letter from the given choices directly.
D
</details>

Figure 4: Qualitative examples of MAGIC selection. The left two examples are highly ranked samples selected by MAGIC, while the right two are low-ranked samples rejected by the method. MAGIC prioritizes instances that require genuine visual grounding and multimodal reasoning, and deprioritizes samples that are largely answerable from language priors or provide weak visual supervision.