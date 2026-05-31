# UniNote: A Unified Embedding Model for Multimodal Representation and Ranking

Jinghan Zhao∗ zhaojinghan2003@gmail.com Xiaohongshu Inc. Beijing, China

Jintao Tong jintaotong@hust.edu.cn Huazhong University of Science and Technology Wuhan, China

Wenwei Jin† wenwei1217.jin@gmail.com Xiaohongshu Inc. Beijing, China

Luya Mo 3120230882@bit.edu.cn Beijing Institute of Technology Beijing, China

Anqi Li anqi.li@sjtu.edu.cn Shanghai Jiao Tong University Shanghai, China

Jiawei Li wangdesheng@xiaohongshu.com Xiaohongshu Inc. Beijing, China

Bin Li libin656712945@gmail.com Xiaohongshu Inc. Beijing, China

Yao Hu yaoohu@gmail.com Xiaohongshu Inc. Beijing, China

# Abstract

Item-to-Item (I2I) retrieval is a fundamental part of modern content platforms, supporting critical industrial workflows from recommendation engines to content auditing. While multimodal embedding methods have advanced general retrieval, they often falter in I2I scenarios due to the challenges of balancing global content representation with fine-grained local retrieval, the systemic inefficiency of decoupled embedding-and-ranking pipelines, and the inherent trade-offs between model precision and serving latency. To solve these issues, we propose UniNote, a unified embedding model designed for industrial I2I retrieval. Tailored retrieval strategies are introduced to support representation learning over complex, multimodal content at varying granularities. To operationalize these strategies, UniNote employs a two-stage training paradigm: the first stage leverages contrastive SFT to establish robust base embeddings, while the second stage refines ranking quality through a reinforcement learning (RL) process that aligns the model with content relevance. Our results show that UniNote achieves SOTA performance across diverse I2I tasks. Deployed at Xiaohongshu and integrated with Matryoshka Representation Learning (MRL), UniNote achieved significant improvements in retrieval quality and cost efficiency in large-scale applications.

# CCS Concepts

• Computing methodologies → Semantic networks; • Information systems → Similarity measures; Language models.

∗Work done during internship at Xiaohongshu Inc.

†Corresponding author.

![](images/607ebfeb10fe8e7d812176cdda9ec56c58a14060e3585b6f51c9f0b9c6f5ed99.jpg)

This work is licensed under a Creative Commons Attribution 4.0 International License. KDD 2026, Jeju Island, Republic of Korea.

© 2026 Copyright held by the owner/author(s).

ACM ISBN 979-8-4007-2259-2/2026/08

https://doi.org/10.1145/3770855.3818503

# Keywords

Multimodal LLMs, Representation Learning, Reinforcement Learning, Search Relevance

# ACM Reference Format:

Jinghan Zhao, Wenwei Jin, Anqi Li, Jintao Tong, Luya Mo, Jiawei Li, Bin Li, and Yao Hu. 2026. UniNote: A Unified Embedding Model for Multimodal Representation and Ranking. In Proceedings of the 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.2 (KDD 2026), August 9–13, 2026, Jeju Island, Republic of Korea. ACM, New York, NY, USA, 10 pages. https://doi.org/10.1145/3770855.3818503

# 1 Introduction

I2I retrieval constitutes a fundamental paradigm powering largescale ecosystems in modern internet content platforms. As content modalities diversify and information volumes surge, retrieval requirements have evolved to be more multifaceted and demanding. Contemporary retrieval frameworks must adapt to heterogeneous inputs across various modalities, including text, images, videos, and their complex combinations, to support a broad range of applications such as recommendation systems[16, 22–24], content moderation[1, 11, 25], and data management.

User-generated notes on the Xiaohongshu platform serve as a representative example of complex items that typically encompass heterogeneous modalities including text, images, videos. The retrieval of such notes entails several critical requirements. Effective retrieval of these notes imposes distinct functional requirements. First, the system must integrate multimodal information of each note into a cohesive global representation. Second, it necessitates fine-grained local retrieval capabilities, enabling tasks such as locating a full note based solely on an individual image. Third, the architecture is required to incorporate accurate relevance ranking while strictly adhering to the latency constraints of massive industrial databases.

![](images/acc97a71d5060da197ec20887dceb6c99a13e692d3f9c10dd4a001383cf09b07.jpg)

<details>
<summary>flowchart</summary>

Diagram illustrating the integration of item definition into unified embedding model, linking multimodal notes, query, LLM, and content relevance modules.
</details>

Figure 1: We propose UniNote, a unified retrieval-ranking framework specifically designed for I2I retrieval, which is capable of encoding various modality inputs into a unified space with cross-modal alignment, localized retrieval, and relevance-aware ranking capabilities.

Despite the progress in multimodal embedding methods, addressing these requirements remains challenging due to three fundamental limitations in existing paradigms. (1) Structural disconnect in dual-tower models. Standard retrieval architectures (e.g., CLIP[20] and its variants[26][12]) encode modalities independently to ensure efficiency. However, this late-interaction mechanism inherently fails to capture the fine-grained alignment between text and visual components required for composite notes, leading to a loss of local detail. (2) Granularity mismatch in generative baselines. While multimodal large language model (MLLM)-based embeddings[10][28] excel at global reasoning, they suffer from a granularity mismatch. These models are optimized for broad semantic descriptions or question-answering tasks, often overlooking the hierarchical structure needed to retrieve a whole note based on a partial input (i.e., local-to-global retrieval). (3) Inefficiency in relevance ranking. Regarding ranking, current industrial pipelines typically rely on a disjointed "retrieve-then-rerank" strategy[5][13]. They use heavy generative models to rerank candidates, creating a significant computational bottleneck. Furthermore, models like NoteLLM-2[27], trained on user interaction data (clicks/likes), optimize for user interest rather than objective content relevance, making them ill-suited for strict semantic retrieval tasks where accuracy and consistency are paramount.

To address these limitations, we propose UniNote, a unified embedding model tailored for the complex demands of I2I retrieval as illustrated in Figure 1. UniNote bridges the gap between holistic encoding and fine-grained retrieval through a novel two-stage training paradigm. First, we introduce a Contrastive Supervised Fine-tuning stage. Unlike rigid dual-tower models, we develop a robust embedding model that compresses heterogeneous inputs into a unified vector space while supporting various I2I retrieval types. We introduce a suite of tailored retrieval settings to facilitate data-driven representation learning for complex multimodal content. This stage transforms an MLLM from a generative model into an embedding model by utilizing the last token as the final

representation. Second, to overcome the ranking limitations of standard contrastive loss, we implement a Relative Reranking via Reinforcement Learning stage utilizing Group Relative Policy Optimization (GRPO)[21]. By constructing a hierarchical reward function that accounts for ranking positions and hard negatives, we directly optimize the embedding space for list-wise relevance. This enables the model to distinguish subtle content nuances without relying on computationally expensive generative rerankers, achieving high-precision ranking in a single pass.

The main contributions are summarized as follows:

• Comprehensive I2I Task Suite: We define ten distinct retrieval categories that span cross-modal alignment, information compression, and fine-grained local–global bidirectional retrieval. To support these, we construct a high-quality dataset with a robust hard-negative mining strategy.   
• Unified Retrieval-Ranking Framework: We propose UniNote, a unified model that bridges the gap between retrieval (embedding) and ranking (relevance). Through a novel two-stage training paradigm(Contrastive SFT followed by Relevance-aware Reinforcement Learning (via GRPO) ), we enable a single embedding to capture both coarse semantic spaces and fine-grained relevance.   
• SOTA Performance & Empirical Validation: Extensive evaluations demonstrate that UniNote achieves state-of-theart results across all ten I2I tasks. Ablation studies confirm that the two training stages synergize to balance embedding and ranking requirements in retrieval tasks.   
• Industrial Deployment Support & Validation: To meet large-scale industrial requirements, we adopt the MRL embedding scheme, allowing dimension selection based on cost–effectiveness trade-offs for flexible deployment. Production evaluations demonstrate that this approach maintains superior retrieval performance under practical constraints.

# 2 Related Works

# 2.1 Multimodal Embedding

Multimodal embedding aims to project heterogeneous data into a unified latent space. Early works, such as CLIP [20], BLIP [12], and SigLIP [26], pioneered the use of large-scale image-text pairs coupled with contrastive learning to achieve cross-modal alignment. These methods typically adopt a dual-tower architecture, where independent encoders project images and text into a shared manifold. While these models exhibit remarkable zero-shot generalization and have become the backbone for various downstream tasks[3, 4, 6, 7, 9, 14, 15, 19, 30, 31], their structural paradigm inherently limits them to single-modality inputs and often struggle with capturing fine-grained semantics or adhere to complex instructions due to their reliance on global, coarse-grained alignment.

The burgeoning demand for flexible and instruction-aware retrieval has catalyzed a shift toward leveraging MLLMs for representation learning. Most recent efforts have converged on a "last-tokenas-embedding" paradigm. E5-V [8] employs prompt-guided MLLMs to compress multimodal context into a single descriptor, significantly improving performance in composite retrieval. UniME-V2 [5] integrates multimodal contrastive learning with hard-negative mining via teacher-student distillation. Qwen3VL-Embedding [13] introduces a multi-stage training pipeline enriched with massive synthetic pairs and multi-task objectives, setting new benchmarks across diverse retrieval scenarios. By inheriting the reasoning capabilities of LLMs, these models transcend the limitations of traditional dual-towers, enabling a more nuanced understanding of complex multimodal queries. However, these models neglect the need to jointly capture global content representation and fine-grained local retrieval in I2I tasks. In addition, they lack a unified retrievalranking framework that integrates embedding with relevance ranking, thereby constraining their generality and consistent performance across diverse I2I retrieval scenarios.

# 2.2 Embedding Model for Rerank

For retrieval tasks, embedding models typically serve as the firststage coarse retriever, while the second-stage fine-grained reranking is commonly handled by dedicated reranking models such as UniME-V2[5], Lamra [18], and Qwen3VL-Embedding[13].In UniME-V2’s reranker, a generative MLLM is supervised through combined pairwise and listwise losses, and this paradigm is similarly adopted by Lamra and Qwen3VL-Embedding.

Some approaches instead employ a unified embedding model for ranking; for instance, TaoSearchEmb[17] proposes Retrieval-GRPO, which follows the contrastive learning phase with rerankingoriented training. Specifically, after encoding the query and candidate sequences, ANN-based retrieval samples top-k items to serve as RL outcomes. However, its reward model operates as a black-box with a single, rigid design, lacking generalizability and failing to ensure monotonic ranking consistency under conditional constraints defined by preferred recall quantity requirements, relative positional dependencies, and absolute positional dependencies in precise content-matching retrieval tasks.

# 3 Method

# 3.1 Task Definition

The I2I retrieval task is formally defined as follows: given a query $q \in { \cal Q }$ and a candidate set $C = \{ c _ { 1 } , c _ { 2 } , \ldots , c _ { M } \}$ , the objective is to retrieve the candidate $c ^ { * }$ that is semantically relevant to $q ,$ where both ?? and ?? can represent items of any modality.

Within the Xiaohongshu platform, the predominant form of user-generated content is referred to as a "Note", which serves as the primary medium for information sharing and interaction, constituting the core data format for our processing. A Note is primarily defined as:

$$
\mathcal {N} = \left(\{I _ {i} \} _ {i = 1} ^ {m}, \{O C R _ {i} \} _ {i = 1} ^ {m}, T _ {\text { title }}, T _ {\text { body }}\right) \tag {1}
$$

where $\{ I _ { i } \} _ { i = 1 } ^ { m }$ denotes a set of ?? images, and ???????? represents the text recognized within the ??-th image $I _ { i } , T _ { t i t l e }$ and $T _ { b o d y }$ refer to the title and body text of the Note, respectively. In this work, videos are represented as a sequence of images without specialized identifiers.

The objective of UniNote is to establish a unified representation for each Note, replacing previous isolated content representations that typically learned independent embeddings for each modality. Specifically, UniNote aims to provide a unified embedding-based representation for multimodal data, facilitating both retrieval and relevance ranking within a single model.

# 3.2 Pipeline Overview

The training of UniNote follows a two-stage pipeline, comprising: (1) Contrastive Supervised Fine-Tuning, and (2) Relevance Reranking via Reinforcement Learning. As illustrated in Figure 2. In the first stage (Sec. 3.3), we transform the generative MLLM into a high-quality embedding model. By leveraging the last-token hidden state as the representation. We adopt multi-granularity hard negative mining to improve the discrimination ability. In the second stage (Sec. 3.4), we bridge the gap between retrieval and reranking. By optimizing the model to perceive semantic overlap gradients through a reward-driven process, the embedding similarity can directly reflect the degree of content relevance, enabling a unified retrieval-ranking capability.

# 3.3 Stage1: Contrastive Supervised Fine-tuning (Contrastive SFT)

This phase focuses on adapting the MLLM into a specialized embedding model by utilizing contrastive learning techniques. Through a data-driven methodology, the model develops the capacity to establish global representations while maintaining local-global mutual retrieval capabilities, which effectively compresses complex information into a dense latent representation.

3.3.1 Training Data. In this stage, we need to construct matched training samples to support various retrieval sub-tasks within Note scenarios, while simultaneously enhancing the discriminative power of the model through hard negative mining.

In real-world application contexts, the requirements for Note retrieval scenarios are categorized into five meta-tasks, encompassing a total of ten specific retrieval tasks as shown in Table 1. (1)

![](images/bcf4e659d82c99454cb23528452770190d4c312b9a1dc962669a5264aedd5da6.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph (a) Data Construction & Mining
        A["Positive Retrieval Pair Synthesis"] --> B["Description Generation"]
        B --> C["Atomic Alignment"]
        C --> D["Subordinate & Semantic"]
        D --> E["OCR Extraction"]
        E --> F["OCR Perception"]
    end

    subgraph (b) Stage 1: Contrastive SFT & MRL
        G["MLLM"] --> H["Last Token"]
        H --> I["MRL"]
        I --> J["L_{k1}, L_{k2}, L_{k3}, L_{k4}"]
        J --> K["L_SFT"]
        K --> L["Target (Soft Labels)"]
        L --> M["Q(c_i | q)"]
        M --> N["P(c_i | q)"]
        N --> O["Prediction"]
    end

    subgraph (c) Stage 2: RL-based Reranking
        P["Candidate Pool e"] --> Q["GT Rank"]
        Q --> R["Policy Model"]
        R --> S["Sort by s_i"]
        S --> T["Reference Model"]
        T --> U["Group Computation"]
        U --> V["r_1, r_2, ..., r_n"]
        V --> W["Optimization (GPRO)"]
    end

    subgraph Multi-granularity Hard Negative Mining
        X["Similarity-based Mining"] --> Y["sim(φ(q),φ(c))"]
        Y --> Z["Hard Negative p'"]
        Z --> AA["τ_min, τ_max"]
        AA --> AB["τ_min, τ_max"]
        AB --> AC["Collection N"]
        AC --> AD["I_1, I_i"]
        AD --> AE["q = I_i, N \ {I_i}"]
    end

    subgraph Optimization (GPRO)
        AF["Reward Function"] --> AG["Irrelevant Note Penalty"]
        AG --> AH["Relevant Note Reward"]
        AH --> AI["Base Relevant Relevant"]
        AH --> AJ["Absolute Position Reward"]
        AH --> AK["Relative Order Reward"]
        AL["QR"] --> AM["gt"]
        AN["Dist"] --> AO["Pi"]
        AP["Grid"] --> AQ["r_1, r_2, ..., r_n"]
    end
```
</details>

Figure 2: The training pipeline, where (a) presents the training data construction strategy designed for item-to-item retrieval, including retrieval sample pairs and hard negative mining strategy; (b) represents the training stage 1: contrastive supervised fine-tuning (SFT); and (c) represents the training stage 2: relative rerank via reinforcement learning (RL).

Atomic Alignment represents a fundamental capability, establishing the mapping between a single image and its corresponding textual description to enable basic cross-modal search. (2) Subordinate Retrieval refers to application scenarios in which local content, such as specific images or text segments, is utilized to retrieve the complete Note. (3) Semantic Extraction functions as a complement to subordinate retrieval and is critically important in applications such as Note risk assessment. (4) OCR Perception addresses the prevalence of text-rich images on Xiaohongshu, creating the need to directly retrieve relevant images and Notes through OCR-based capabilities. (5) Content Relevance measures the degree of content relevance between different Notes.

Table 1: I2I Retrieval Tasks Taxonomy. 

<table><tr><td>Meta Task</td><td>Search Type</td></tr><tr><td>Atomic Alignment</td><td>image ↔ text</td></tr><tr><td>Subordinate Retrieval</td><td>image / text → note</td></tr><tr><td>Semantic Extraction</td><td>note → image / text</td></tr><tr><td>OCR Perception</td><td>image ↔ ocr, ocr → note</td></tr><tr><td>Content Relevance</td><td>note → note</td></tr></table>

The positive retrieval pair construction begins with high-quality notes collected from Xiaohongshu. First, based on each image $I _ { i }$ in note N, we use an annotator model $\phi _ { \mathrm { a n n } }$ to generate semantically aligned image-text pairs $( I _ { i } , S _ { \mathrm { d e s c } } ^ { i } )$ for Atomic-level Alignment. Then, for Subordinate Retrieval and Semantic Extraction, we leverage the inherent subordinate relationship between notes and their constituent elements, with ground-truth pairs defined

as $( I _ { i } , N )$ where $I _ { i } \in N .$ . In order to prevent the model from exploiting "low-level shortcuts" where it might match pixels or local features instead of understanding semantics, we introduce a Modal Replacement mechanism that substitutes the original image with its textual description:

$$
(I _ {i}, \mathcal {N} ^ {\prime}), \quad \text { where } \quad \mathcal {N} ^ {\prime} = \{S _ {\mathrm{desc}} ^ {i} \} \cup \mathcal {N} \setminus \{I _ {i} \} \tag {2}
$$

This formulation also supports text-query variants in the form of $( S _ { \mathrm { d e s c } } ^ { i } , N )$ . For OCR Perception. We employ expert OCR models to extract textual content embedded within images, yielding pairs $( I _ { i } , S _ { \mathrm { o c r } } ^ { i } )$ . These pairs provide positive samples for the I2OCR, OCR2I, and OCR2Note tasks.

Multi-granularity Hard Negative Mining. The embedding model aims to pull related samples closer and push unrelated ones apart via contrastive learning. In this context, hard negative samples are critical for enhancing the discriminative capability of the model. To facilitate robust hard negative mining across diverse scenarios, we utilize both relevance score techniques [2, 5, 29] and counterfactual hard negatives.

Given a query ?? and its corresponding positive samples $c ^ { + }$ , we aim to construct a set of hard negative samples $C ^ { - } = \{ c _ { 1 } ^ { - } , . . . , c _ { k } ^ { - } \}$ . Following the paradigm of UniME-V2, our approach integrates global mining with soft supervision. Let P denote the global candidate pool and $\phi ( \cdot )$ represent a reference encoder. The subset of valid hard negatives $\mathcal { P } ^ { \prime } \subset \mathcal { P }$ is defined as:

$$
\mathcal {P} ^ {\prime} = \{c \in \mathcal {P} \mid \tau_ {\mathrm{min}} \leq \mathrm{sim} (\phi (q), \phi (c)) \leq \tau_ {\mathrm{max}} \}, \tag {3}
$$

where $[ \tau _ { \mathrm { m i n } } , \tau _ { \mathrm { m a x } } ]$ defines the similarity range that identifies informative negatives while excluding false negatives. The similarity score $s = \sin ( \phi ( q ) , \phi ( c ^ { - } ) )$ for $c ^ { - } \in \mathcal { P } ^ { \prime }$ is subsequently utilized as a soft supervision signal.

To address scenarios where candidates are composite structures (e.g., I2Note or T2Note) rather than monolithic entities, we refine the scoring mechanism. Since standard encoders often fail to simultaneously capture global semantics and local fine-grained details, we define a refined soft score $s ^ { \prime }$ using a MaxSim operation:

$$
s ^ {\prime} (q, c ^ {-}) = \max _ {e \in c ^ {-}} \text { sim } (q, e), \tag {4}
$$

where ?? denotes the constituent elements of the composite candidate $c ^ { - }$ . This refined soft score $s ^ { \prime }$ subsequently serves as the supervision target for distribution alignment.

Furthermore, we incorporate heuristic rules based on the membership relation $( I _ { i } , N )$ , where $I _ { i } \in { \mathcal { N } }$ . We construct hard negative pairs via the following set subtraction:

$$
(q, c _ {\text { rule }} ^ {-}) = (I _ {i}, \mathcal {N} \setminus \{I _ {i} \}). \tag {5}
$$

This formulation ensures that the negative sample maintains maximum content overlap with the positive candidate context while strictly precluding any valid membership relationship.

3.3.2 Optimization Goal. We minimize the divergence between the embedding-based distribution ?? and the MLLM-derived softlabel distribution ??. Given a candidate set $\Omega _ { c } = \{ c ^ { + } , c _ { 1 } ^ { - } , \ldots , c _ { m } ^ { - } \}$ , the distributions are formulated as:

$$
Q (c _ {i} | q) = \frac {\exp (s _ {q , c _ {i}} / \tau)}{\sum_ {c _ {j} \in \Omega_ {c}} \exp (s _ {q , c _ {j}} / \tau)} \tag {6}
$$

$$
P (c _ {i} | q) = \frac {\exp (\cos (e _ {q} , e _ {c _ {i}}) / \tau)}{\sum_ {c _ {j} \in \Omega_ {c}} \exp (\cos (e _ {q} , e _ {c _ {j}}) / \tau)} \tag {7}
$$

To ensure learning robustness and symmetry, we employ the Jensen-Shannon (JS) divergence as the objective:

$$
\mathcal {D} _ {J S} (P \parallel Q) = \frac {1}{2} \mathrm{KL} (P \parallel Q) + \frac {1}{2} \mathrm{KL} (Q \parallel P) \tag {8}
$$

# 3.4 Stage 2: Relative Reranking via Reinforcement Learning

In this stage, we utilize Note2Note retrieval training data to enhance the perception of content relevance and the ranking capability of the model. We define the relevance between two notes as the degree of content overlap in our training settings. To construct training samples, we consider a given note N and partition both its visual and textual components into two equal sub-notes, $N _ { A }$ and $N _ { B }$ . This partition satisfies the conditions ${ \cal N } _ { A } \cap { \cal N } _ { B } = \emptyset$ and $N _ { A } \cup N _ { B } = N _ { \mathrm { \cdot } }$ ensuring that $N _ { A }$ and $N _ { B }$ initially have no content overlap.

With $N _ { A }$ serving as the query ??, we construct a candidate sequence with incrementally decreasing content overlap as follows: ${ \cal L } _ { r e l } = [ N _ { B } \cup \{ I _ { 1 } ^ { A } , I _ { 2 } ^ { A } , \ldots \} , \ldots , N _ { B } \cup \{ I _ { 1 } ^ { A } \} , N _ { B } ]$ . Furthermore, we introduce a noise sequence, denoted as $\{ N _ { n o i s e } , \ldots \}$ . The final ranking list $L _ { r a n k }$ is then systematically constructed by integrating the content-relevant sequence with the noise sequence, defined as: $L _ { r a n k } = [ N _ { B } \cup \{ I _ { 1 } ^ { A } , I _ { 2 } ^ { A } , \dotsc , I _ { M } ^ { A } \} , \dotsc , N _ { B } \cup \{ I _ { 1 } ^ { A } \} , N _ { B } , N _ { 1 } ^ { n o i s e } , N _ { 2 } ^ { n o i s e } , \dotsc ] .$ Here, M is the number of content-relevant notes in $L _ { r a n k }$ for the given query.

In contrast to existing frameworks such as UniME-V2 and Qwen3VL-Embedding, which typically utilize a separate model for the reranking stage, we unify representation learning and reranking within

![](images/39785d4bcc1097dcba4d2a93866637cd47311159c1dafc4c3fa3ab245231fb57.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["GT Rank"] --> B["Predict Rank"]
    B --> C{for each}
    C --> D["Irrelevant Note Penalty"]
    D --> E["i → penalty → i"]
    E --> F["+ Base Relevant Reward"]
    F --> G["+ Absolute Position Reward"]
    G --> H["where(V, p_i) dist"]
    H --> I["+ Relative Order Reward"]
    I --> J["where(V, p_i) + X"]
    J --> K["Output to (a)"]
    K --> L["Order Precision vs. Relevant Hit Rate"]
    L --> M["High Reward"]
    L --> N["Low Reward"]
```
</details>

Figure 3: Schematic illustration of the reward function. (a) The reward function comprises four components: Irrelevant Note Penalty, Base Relevant Reward, Absolute Position Reward, and Relative Order Reward. (b) The reward distribution for simulated prediction ranking is depicted. The reward increases in conjunction with the relevant hit rate and order precision, where the relevant hit rate plays a more decisive role in the overall reward value.

a single model architecture. We use GRPO framework for training, as the reranking process inherently aligns with the concept of group-based rewards. This process is illustrated in Figure 2(c), consisting of three stages: ranking prediction, reward design, and optimization.

3.4.1 Candidates Selection. This step simulates the group sampling process within the GRPO framework. Specifically, the top-?? (G>M) notes retrieved by the embedding model are treated as the candidate group. By computing the similarity scores, we establish a descending ranking order: $P = \left[ / p _ { 1 } , p _ { 2 } , . . . , p _ { G } \right]$ . Meanwhile, the corresponding ground truth ranking is denoted as $V = [ g _ { 1 } , g _ { 2 } , \dots , g _ { G } ]$ .

3.4.2 Reward Design. The reward function (Figure 3) is designed to strength two specific capabilities of the model: the ability to discriminate between relevant and irrelevant notes, and the ability to rank notes according to their relevance. Consequently, the reward function ?? is constructed across four dimensions: irrelevance penalty, relevance reward, absolute position reward, and relative position reward. We design a multi-dimensional reward function to supervise the quality of ranking across three levels of granularity:

• Irrelevant Note Penalty: We penalize noise notes $\pmb { \mathscr { p } } _ { i } \notin L _ { r e l }$ that are ranked above any relevant note $( \exists p _ { j } \in L _ { r e l } , j \ >$ ??). The penalty escalates as ?? approaches 1, reflecting the increased harm of ranking irrelevant content highly.   
• Base Relevant Reward : In our business context, we prioritize maximizing the recall of relevant samples in ?? over the ranking order of relevant notes. Therefore, we assign a fixed base reward $R _ { \mathrm { b a s e } }$ to any note where $\boldsymbol { p } _ { i } \in L _ { \mathrm { r e l } } .$ .   
• Absolute Position Reward: To minimize the displacement between predicted and ground-truth positions, we define a reward based on the distance dis $\mathrm { ~ t } = | p _ { i } - w h e r e ( V , \mathfrak { p } _ { i } ) | . \mathrm { ~ A ~ }$ smaller distance implies a more accurate absolute position.   
• Relative Order Reward: Since absolute distance alone cannot capture the global consistency of the sequence, we calculate a

relative order reward inspired by the concept of concordant pairs. For each relevant note $\mathbf { \nabla } \mathcal { P } i$ , we count the number of other relevant notes $\boldsymbol { \mathscr { P } } \boldsymbol { j }$ that are correctly ordered relative to $i .$

The reward $r _ { i }$ for the note at position ?? is formulated as follows:

$$
R _ {\mathrm{abs}} (i, p _ {i}) = 1 - \frac {| p _ {i} - w h e r e (V , p _ {i}) |}{G} \tag {9}
$$

$$
R _ {\text { rel }} (i, P) = \text { mean } (\mathbb {I} [ (p _ {i} - p _ {j}) (\text { where } (V, p _ {i}) - \text { where } (V, p _ {j})) > 0 ]) \tag {10}
$$

$$
r _ {i} = \left\{ \begin{array}{l l} C _ {\text { pen }} \cdot \left(1 + \frac {G - i}{G}\right) & \text { if   } p _ {i} \notin L _ {r e l} \text {   and   } \exists p _ {j} \in L _ {r e l}, j > i \\ 0 & \text { if   } p _ {i} \notin L _ {r e l} \text {   and   } \nexists p _ {j} \in L _ {r e l}, j > i \\ R _ {\text { base }} + R _ {\text { abs }} (p _ {i}, i) + R _ {\text { rel }} (p _ {i}, P) & \text { if   } p _ {i} \in L _ {r e l} \end{array} \right. \tag {11}
$$

where $C _ { \mathrm { p e n } }$ and $R _ { \mathrm { b a s e } }$ are fixed hyperparameters,

3.4.3 Optimization. We compute the intra-group advantage scores for the predicted candidates. The advantage $A _ { i }$ for each candidate in the group is computed as:

$$
A _ {i} = \frac {r _ {i} - m e a n (r)}{s t d (r) + \epsilon} \tag {12}
$$

Following the GRPO framework, the final loss function $\mathcal { L } _ { G R P O }$ is formulated as:

$$
\mathcal {L} _ {G R P O} = \mathbb {E} [ A _ {i} \cdot \log \pi_ {\theta} (P | q, L _ {r a n k}) ] \tag {13}
$$

where $\pi _ { \theta }$ denotes the policy model, which corresponds to our embedding model.

# 3.5 MRL Support

To support diverse deployment constraints, we integrate MRL into our alignment framework. We define a set of nested dimensions $\mathcal { K } = \{ 6 4 , 5 1 2 , 1 0 2 4 , 4 0 9 6 \}$ }. For each dimension $k \in \mathcal { K } ,$ we minimize the divergence between the embedding-based distribution $P ^ { ( k ) }$ and the MLLM-derived soft-label distribution ??. The final total loss $\mathcal { L } _ { S F T }$ is the sum across all nested dimensions:

$$
\mathcal {L} _ {S F T} = \sum_ {k \in \mathcal {K}} \mathcal {D} _ {J S} \left(P ^ {(k)} \| Q\right) \tag {14}
$$

This ensures that lower-dimensional truncations maintain high retrieval accuracy while inheriting the semantic depth of the fulldimensional representation.

# 4 Experiment Results

# 4.1 Evaluation Settings

We utilize Qwen3VL-8B-Instruct as the foundation framework for UniNote. During the RL stage, we employ the GRPO with the Adam for optimization. To address the lack of text in raw training data, we leverage Qwen3VL-8B-Instruct to generate descriptive captions and Qwen3VL-8B-Embedding for hard negative mining. The training is conducted on 8 H800 GPUs.

The evaluation is conducted on a large-scale multimodal dataset comprising 66k notes and over 500k items spanning image, text, and OCR modalities. The evaluation tasks correspond to the five meta-tasks specified in Table 1. Recall is adopted as the primary evaluation metric. For the Note2Note task, each query is associated with multiple relevant notes. In contrast, for the remaining four categories of tasks, each query contains only a single matching positive sample.

# 4.2 Comparison with State-of-the-Arts

We selected RzenEmbedding and Qwen3VL-Embedding-8B as our primary baselines, as both have achieved state-of-the-art (SOTA) performance in general retrieval tasks. As illustrated in Table 2, while existing SOTA methods exhibit robust performance in fundamental cross-modal alignment tasks (e.g., atomic alignment), but they falter in Subordinate Retrieval and Semantic Extraction. We attribute this to their training paradigms, which rely heavily on single image-text pairs. Consequently, these models lack the capacity to model the intricate global-local relationships inherent in complex, multimodal compositions, leading to an inability to balance coarse-grained global representation with fine-grained local retrieval.

Furthermore, in the OCR perception task, both RzenEmbed and Qwen3VL-Embedding-8B demonstrate robust I2OCR retrieval capabilities. However, the inverse capability of using OCR to retrieve corresponding images is significantly weaker. This performance gap is further amplified when attempting to retrieve entire notes based on OCR queries. In contrast, UniNote employs a data-driven approach to strike an effective balance across these diverse tasks. UniNote achieves the best performance across all tasks with the exception of I2OCR. It not only facilitates effective cross-modal alignment but also yields substantial improvements over RzenEmbed and Qwen3VL-Embedding-8B in local-to-global bidirectional retrieval tasks, such as I2Note, Note2I, and OCR2Note.

For the Note2Note retrieval task, we constructed a candidate pool by mixing irrelevant "distractor" notes with M relevant notes, requiring the model to identify and rank the top-M candidates. We found that both Rzen and Qwen3VL-Embedding performed respectably in this setting. This may be due to the nature of the task, where local overlap serves as a salient feature that is relatively easy for encoders to capture.

# 4.3 Ablation Result

To evaluate the efficacy of our proposed two-stage training framework, we conducted a series of ablation studies across four distinct configurations, as summarized in Table 3.

We evaluate the efficacy of our hard negative mining strategy in SFT stage: First, we examine the rand SFT variant, where negative samples are randomly selected from the global pool. This approach tends to introduce false negatives and suffers from an excessive semantic gap between positive and negative pairs. Such easy negatives fail to provide informative optimization gradients, thereby preventing the model from learning highly discriminative representations. Consequently, the performance of rand SFT on four meta-tasks is significantly lower than that of UniNote. This result demonstrates that purely random sampling is insufficient for capturing fine-grained cross-modal nuances.

Table 2: Performance comparison of different embedding models (%). 

<table><tr><td rowspan="2">Category</td><td rowspan="2">Search Type</td><td rowspan="2">n_queries</td><td rowspan="2">n_targets</td><td colspan="3">RzenEmbed</td><td colspan="3">Qwen3VL-Embed.</td><td colspan="3">UniNote (Ours)</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td rowspan="2">Atomic Alignment</td><td>I2T</td><td>50w</td><td>48w</td><td>63.6</td><td>82.7</td><td>87.1</td><td>56.0</td><td>76.1</td><td>81.6</td><td>75.2</td><td>90.2</td><td>92.9</td></tr><tr><td>T2I</td><td>48w</td><td>50w</td><td>69.5</td><td>86.6</td><td>90.2</td><td>52.9</td><td>72.6</td><td>78.4</td><td>74.0</td><td>89.1</td><td>92.0</td></tr><tr><td rowspan="2">Subordinate Retrieval</td><td>I2Note</td><td>50w</td><td>13w</td><td>37.9</td><td>59.6</td><td>65.5</td><td>41.2</td><td>61.1</td><td>67.4</td><td>93.7</td><td>100</td><td>100</td></tr><tr><td>T2Note</td><td>48w</td><td>13w</td><td>32.2</td><td>53.8</td><td>59.8</td><td>39.6</td><td>62.4</td><td>68.5</td><td>53.1</td><td>72.0</td><td>77.3</td></tr><tr><td rowspan="2">Semantic Extraction</td><td>Note2I</td><td>6.6w</td><td>6.6w</td><td>51.2</td><td>68.3</td><td>73.6</td><td>64.3</td><td>80.8</td><td>85.3</td><td>90.2</td><td>91.6</td><td>92.7</td></tr><tr><td>Note2T</td><td>6.6w</td><td>6.6w</td><td>43.1</td><td>61.5</td><td>67.4</td><td>44.6</td><td>63.3</td><td>69.3</td><td>64.2</td><td>83.8</td><td>87.7</td></tr><tr><td rowspan="3">OCR Perception</td><td>OCR2Note</td><td>18w</td><td>13w</td><td>33.2</td><td>55.2</td><td>60.6</td><td>37.7</td><td>60.6</td><td>66.4</td><td>62.1</td><td>80.5</td><td>84.5</td></tr><tr><td>I2OCR</td><td>20w</td><td>18w</td><td>70.4</td><td>84.0</td><td>86.9</td><td>79.0</td><td>91.3</td><td>93.3</td><td>53.5</td><td>66.7</td><td>69.9</td></tr><tr><td>OCR2I</td><td>18w</td><td>20w</td><td>55.8</td><td>71.2</td><td>75.5</td><td>64.3</td><td>79.0</td><td>82.5</td><td>82.8</td><td>90.9</td><td>92.5</td></tr><tr><td>Content Relevance</td><td>Note2Note</td><td>3.7w</td><td>37w</td><td>15.8</td><td>79.3</td><td>98.6</td><td>15.9</td><td>79.6</td><td>99.4</td><td>15.9</td><td>79.3</td><td>99.5</td></tr></table>

Next, we exclude hard negatives generated through rule-based heuristics. The substantial performance improvement of without rule-based mining compared to rand mining validates the significant value of global hard negative mining and soft supervision. Building upon this, the further performance gains achieved by incorporating counterfactual hard negatives demonstrate that this strategy forces the model to distinguish more subtle semantic differences, thereby increasing the confidence of the model in identifying subordinate relationships.

Finally, we evaluate the impact of the RL stage on the Note2Note task. As illustrated in Table 4, the introduction of the RL stage enhances both the precision and recall of the model for the Note2Note task. Specifically, the R@5 and P@5 metrics increase by 1.4% and 3.4%, respectively. The results indicate that, across various retrieval and matching tasks, UniNote achieves a higher relevant-sample recall rate than the SFT mode when the number of retrieved samples is controlled to be identical.

# 4.4 MRL Embedding Performance

To balance retrieval performance and storage overhead, we incorporate Matryoshka Loss to support multi-dimensional outputs. Figure 4 illustrates the impact of feature dimensions (64, 512, 1024, and 4096) on efficiency and Recall. The results demonstrate that the overall performance exhibits an upward trend as the feature dimension increases. Specifically, at a dimension of 512, the performance of most retrieval tasks approximates that of the full-dimensional features. Although a decline in performance is observed at a dimension of 64, the model retains 70% of the capability in tasks such as Atomic Alignment, Subordinate Retrieval, and Semantic Extraction. Consequently, the Matryoshka Embedding enables flexible deployment, catering to the requirements of diverse application scenarios.

# 5 Online Deployment

We conducted deployment testing of UniNote following the setup illustrated in Figure 5, operating in two modes: online and offline,

![](images/a31bee8626a2a6b65c8aa98cf1413ccd96809501b5e39c61906f7489bfe29172.jpg)

<details>
<summary>bar</summary>

| Category | 64 | 512 | 1024 | 4096 |
|---|---|---|---|---|
| Atomic Alignment | 0.54 | 0.73 | 0.75 | 0.75 |
| Subordinate Retrieval | 0.58 | 0.78 | 0.78 | 0.77 |
| Semantic Extraction | 0.62 | 0.75 | 0.73 | 0.74 |
| OCR Perception | 0.28 | 0.60 | 0.65 | 0.66 |
</details>

Figure 4: Impact of feature dimensions on efficiency and recall with Matryoshka embeddings.

corresponding to real-time production traffic and manually triggered tasks, respectively.

In the online mode, high-concurrency traffic streams are processed in real time by UniNote for embedding extraction and subsequent storage. Simultaneously, the Approximate Nearest Neighbor (ANN) index is updated to incorporate historical data. In parallel, the incoming traffic is matched against a pre-built small-scale sample repository for validation, and any matched notes are subsequently processed according to the downstream task requirements.

In the offline mode, operations personnel can initiate two categories of tasks. The first involves processing multimodal and multi-type samples (e.g., image, text, video, and note) for embedding extraction via UniNote, followed by storage and index updates—corresponding to the construction and maintenance of the aforementioned sample repository in the online mode. The second category enables retrieval from the historical data index using any given item as a query, returning the Top-K most relevant samples. Building upon the above definitions, we classify the evaluation tasks into two categories: (1) high-request, small-gallery verification, and (2) low-request, large-gallery retrieval.

Table 3: Hard Negative Sample Ablation Results (%). 

<table><tr><td>Method</td><td>Metric</td><td>Atomic Alignment</td><td>Subordinate Retrieval</td><td>Semantic Extraction</td><td>OCR Perception</td><td>Total</td></tr><tr><td rowspan="3">rand</td><td>R@1</td><td>57.6</td><td>66.2</td><td>62.6</td><td>42.8</td><td>57.3</td></tr><tr><td>R@5</td><td>76.5</td><td>79.5</td><td>81.6</td><td>56.4</td><td>73.5</td></tr><tr><td>R@10</td><td>81.7</td><td>83.1</td><td>86.0</td><td>60.2</td><td>77.7</td></tr><tr><td rowspan="3">UniNote (w/o Rule)</td><td>R@1</td><td>71.9</td><td>74.1</td><td>73.9</td><td>47.0</td><td>66.7</td></tr><tr><td>R@5</td><td>87.6</td><td>84.4</td><td>88.6</td><td>57.3</td><td>79.5</td></tr><tr><td>R@10</td><td>90.8</td><td>87.2</td><td>91.3</td><td>59.4</td><td>82.2</td></tr><tr><td rowspan="3">UniNote</td><td>R@1</td><td>74.6 (+17.0)</td><td>76.4 (+10.2)</td><td>77.2 (+14.6)</td><td>66.1 (+23.3)</td><td>73.6 (+16.3)</td></tr><tr><td>R@5</td><td>89.6 (+13.1)</td><td>86.0 (+6.5)</td><td>87.7 (+6.1)</td><td>79.3 (+22.9)</td><td>85.7 (+12.2)</td></tr><tr><td>R@10</td><td>92.5 (+10.8)</td><td>88.6 (+5.5)</td><td>90.2 (+4.2)</td><td>82.3 (+22.1)</td><td>88.4 (+10.7)</td></tr></table>

Table 4: Ablation study on RL module (%). 

<table><tr><td>Model</td><td>R@1</td><td>R@5</td><td>R@10</td><td>P@1</td><td>P@5</td><td>P@10</td></tr><tr><td>UniNote (w/o RL)</td><td>15.7</td><td>77.9</td><td>97.9</td><td>91.7</td><td>92.6</td><td>61.8</td></tr><tr><td>UniNote</td><td>15.9(+0.2)</td><td>79.3(+1.4)</td><td>99.5(+1.6)</td><td>96.0(+4.3)</td><td>96.0(+3.4)</td><td>64.5(+2.7)</td></tr></table>

Online mode (high-request, small-gallery verification). We evaluate UniNote in an online high-concurrency setting, matching daily incoming notes against a safety policy gallery. Over 7 consecutive days, 10% of daily traffic was randomly sampled for A/B testing. The baseline is a production CLIP-based image–text model trained on domain-specific data. The gallery contains three risk categories—images, videos, and notes—with 50 samples each (150 total). Similarity thresholds were tuned per category to match the baseline’s retrieval count, and professional annotators verified results. UniNote achieved 85.6%, 91.2%, and 93.6% recall retention in Note2Image, Note2Video, and Note2Text tasks using a single embedding extraction, while the baseline Image-to-Image method required 9.2× more storage and compute. UniNote thus delivers high recall with substantially lower resource cost.

Offline mode (low-request, large-gallery retrieval). In the backchecking scenario, known interest images (or notes and other related items) are used as queries to retrieve a collection of notes that contain content relevant to the given interest, from a large-scale historical note repository. In this experimental setting, we likewise employ the existing online CLIP-based image–text alignment model as the baseline and select one week of online data as the historical note collection. For the query configuration, we adopt the ANN search method and fix the number of returned results to the top-10 notes per query. A set of 1K images of various types was used as queries, collectively yielding 10K note retrieval results for annotation. After deduplication of identical notes, UniNote achieved a 23.5% gain in relevant sample recall. This demonstrates that retrieval methods based on note sub-content tend to result in

![](images/4d0dc3d5b97c9c137aa63d911b3b0cd5e2e661f485d6bf75c64cf85e996aff8d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Online"] --> B["UniNote"]
    C["Offline"] --> D["UniNote"]
    B --> E["Refere. Emb. Database"]
    D --> F["Corpus Emb. Database"]
    E --> G["ANN Index"]
    F --> G
    G --> H["Match"]
    H --> I["sim > thres."]
    I --> J["Handle"]
    G --> K["Recall Top-K Notes"]
    K --> L["ANN Search"]
    L --> M["Record"]
    L --> N["Query"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style E fill:#cfc,stroke:#333
    style F fill:#cfc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#cff,stroke:#333
    style I fill:#ffc,stroke:#333
    style J fill:#ffc,stroke:#333
    style K fill:#ffc,stroke:#333
    style L fill:#ffc,stroke:#333
    style M fill:#ffc,stroke:#333
```
</details>

Figure 5: Online deployment pipeline. The online mode refers to the process in which massive user-generated notes are stored into the corpus and simultaneously used as queries to perform matching verification against the reference smallsample library. The offline mode refers to the process where internal operations personnel update the reference smallsample library with risk-related data and perform retrieval and recall from the historical corpus

more duplicate recalls under identical computational cost, further highlighting the necessity of unified representation and retrieval.

The above two online experiments demonstrate the efficiency and practical viability of UniNote in industrial retrieval tasks.

# 6 Conclusion and Future Work

In this work, we address the item-to-item (I2I) retrieval challenge in multimodal scenarios, aiming to balance global representation and fine-grained retrieval while unifying multimodal representation and relevance search in a shared embedding space. Using real Xiaohongshu note data, we designed ten retrieval tasks of varying granularity and transformed them into high-quality embeddings via a Multimodal Large Language Model (MLLM). Through crossmodal alignment combined with similarity scoring and counterfactual hard negative mining, UniNote achieved substantial gains in I2I retrieval. In the Note2Note task, we introduced GRPO-based reinforcement learning, proposing a robust relevance-ranking reward that advances the unification of representation and ranking, with scalability to tasks such as image-to-note (I2Note). This approach offers an efficient, practical solution for industrial-scale heterogeneous item retrieval, and paves the way for leveraging reinforcement learning to further boost retrieval and ranking across diverse scenarios.

# References

[1] Premnarayan Arya, Amit Kumar Pandey, S Gopal Krishna Patro, Kretika Tiwari, Niranjan Panigrahi, Quadri Noorulhasan Naveed, Ayodele Lasisi, and Wahaj Ahmad Khan. 2024. MSCMGTB: A Novel Approach for Multimodal Social Media Content Moderation Using Hybrid Graph Theory and Bio-Inspired Optimization. IEEE Access 12 (2024), 73700–73718.   
[2] Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. 2024. Mllm-as-ajudge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In Forty-first International Conference on Machine Learning.   
[3] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. 2009. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition. Ieee, 248–255.   
[4] Mark Everingham, SM Ali Eslami, Luc Van Gool, Christopher KI Williams, John Winn, and Andrew Zisserman. 2015. The pascal visual object classes challenge: A retrospective. International journal of computer vision 111, 1 (2015), 98–136.   
[5] Tiancheng Gu, Kaicheng Yang, Kaichen Zhang, Xiang An, Ziyong Feng, Yueyi Zhang, Weidong Cai, Jiankang Deng, and Lidong Bing. 2025. Unime-v2: Mllm-as-a-judge for universal multimodal embedding learning. arXiv preprint arXiv:2510.13515 (2025).   
[6] Dan Hendrycks, Steven Basart, Norman Mu, Saurav Kadavath, Frank Wang, Evan Dorundo, Rahul Desai, Tyler Zhu, Samyak Parajuli, Mike Guo, et al. 2021. The many faces of robustness: A critical analysis of out-of-distribution generalization. In Proceedings of the IEEE/CVF international conference on computer vision. 8340– 8349.   
[7] Dan Hendrycks, Kevin Zhao, Steven Basart, Jacob Steinhardt, and Dawn Song. 2021. Natural adversarial examples. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 15262–15271.   
[8] Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. 2024. E5-v: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580 (2024).   
[9] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. 2014. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP). 787–798.   
[10] Chankyu Lee, Rajarshi Roy, Mengyao Xu, Jonathan Raiman, Mohammad Shoeybi, Bryan Catanzaro, and Wei Ping. 2024. Nv-embed: Improved techniques for training llms as generalist embedding models. arXiv preprint arXiv:2405.17428 (2024).   
[11] Anqi Li, Wenwei Jin, Jintao Tong, Pengda Qin, Weijia Li, and Guo Lu. 2025. Towards Trustworthy Multimodal Moderation via Policy-Aligned Reasoning and Hierarchical Labeling. arXiv preprint arXiv:2508.03296 (2025).   
[12] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. 2022. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International conference on machine learning. PMLR, 12888–12900.   
[13] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. 2026. Qwen3-VL-Embedding and Qwen3-VL-Reranker: A Unified Framework for State-of-the-Art Multimodal Retrieval and Ranking. arXiv preprint arXiv:2601.04720 (2026).   
[14] Tsung-Yi Lin, M Maire, S Belongie, J Hays, P Perona, D Ramanan, P Dollár, and CL Zitnick. [n. d.]. Microsoft coco: Common objects in context. Computer Vision-ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014. Proceedings, Part 13 ([n. d.]).   
[15] Fuxiao Liu, Yinghan Wang, Tianlu Wang, and Vicente Ordonez. 2021. Visual news: Benchmark and challenges in news image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing. 6761–6771.   
[16] Qidong Liu, Jiaxi Hu, Yutian Xiao, Xiangyu Zhao, Jingtong Gao, Wanyu Wang, Qing Li, and Jiliang Tang. 2024. Multimodal recommender systems: A survey. Comput. Surveys 57, 2 (2024), 1–17.   
[17] Xingxian Liu, Dongshuai Li, Tao Wen, Jiahui Wan, Gui Ling, Fuyu Lv, Dan Ou, and Haihong Tang. 2025. Taosearchemb: A multi-objective reinforcement learning framework for dense retrieval in taobao search. arXiv preprint arXiv:2511.13885 (2025).   
[18] Yikun Liu, Yajie Zhang, Jiayin Cai, Xiaolong Jiang, Yao Hu, Jiangchao Yao, Yanfeng Wang, and Weidi Xie. 2025. Lamra: Large multimodal model as your advanced retrieval assistant. In Proceedings of the Computer Vision and Pattern Recognition Conference. 4015–4025.   
[19] Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2200–2209.   
[20] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning transferable visual models from natural language supervision. In International conference on machine learning. PmLR, 8748–8763.   
[21] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024).

[22] Yinwei Wei, Xiang Wang, Liqiang Nie, Xiangnan He, Richang Hong, and Tat-Seng Chua. 2019. MMGCN: Multi-modal graph convolution network for personalized recommendation of micro-video. In Proceedings of the 27th ACM international conference on multimedia. 1437–1445.   
[23] Chuhan Wu, Fangzhao Wu, Tao Qi, Chao Zhang, Yongfeng Huang, and Tong Xu. 2022. Mm-rec: Visiolinguistic model empowered multimodal news recommendation. In Proceedings of the 45th international ACM SIGIR conference on research and development in information retrieval. 2560–2564.   
[24] Derong Xu, Wei Chen, Wenjun Peng, Chao Zhang, Tong Xu, Xiangyu Zhao, Xian Wu, Yefeng Zheng, Yang Wang, and Enhong Chen. 2024. Large language models for generative information extraction: A survey. Frontiers of Computer Science 18, 6 (2024), 186357.   
[25] Jialin Yuan, Ye Yu, Gaurav Mittal, Matthew Hall, Sandra Sajeev, and Mei Chen. 2024. Rethinking multimodal content moderation from an asymmetric angle with mixed-modality. In Proceedings of the IEEE/CVF winter conference on applications of computer vision. 8532–8542.   
[26] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF international conference on computer vision. 11975–11986.   
[27] Chao Zhang, Haoxin Zhang, Shiwei Wu, Di Wu, Tong Xu, Xiangyu Zhao, Yan Gao, Yao Hu, and Enhong Chen. 2025. Notellm-2: Multimodal large representation models for recommendation. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 1. 2815–2826.   
[28] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. 2024. GME: Improving Universal Multimodal Retrieval by Multimodal LLMs. arXiv preprint arXiv:2412.16855 (2024).   
[29] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Advances in neural information processing systems 36 (2023), 46595–46623.   
[30] Bolei Zhou, Agata Lapedriza, Aditya Khosla, Aude Oliva, and Antonio Torralba. 2017. Places: A 10 million image database for scene recognition. IEEE transactions on pattern analysis and machine intelligence 40, 6 (2017), 1452–1464.   
[31] Yuke Zhu, Oliver Groth, Michael Bernstein, and Li Fei-Fei. 2016. Visual7w: Grounded question answering in images. In Proceedings of the IEEE conference on computer vision and pattern recognition. 4995–5004.

# A Contrastive SFT Settings

We filter high-quality notes from the raw dataset of Xiaohongshu based on a threshold of more than 100 likes. Balanced processing is performed across topics and image counts per note to prevent weaknesses in specific dimensions. Hard negative samples are selected with $\tau _ { \mathrm { m i n } } = 0 . 5$ and $\tau _ { \operatorname* { m a x } } = 0 . 7 .$ , determined through manual sampling and inspection. This yields 900,000 training samples across nine task types (excluding note2note), each with 100,000 samples.

We follow UniME-V2 [5] training parameters to perform LoRA fine-tuning on Qwen3VL-8B-Instruct using eight H800 GPUs (80G). The specific parameters are listed in Table 5.

Table 5: SFT training hyperparameters 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Training samples</td><td>900k</td></tr><tr><td>Batch size</td><td>64</td></tr><tr><td>Learning rate</td><td>1e-4</td></tr><tr><td>Warmup ratio</td><td>0.1</td></tr><tr><td>LoRA rank</td><td>16</td></tr><tr><td>Training steps</td><td>4000</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Infra</td><td>GradCache</td></tr><tr><td>Max length</td><td>4096</td></tr><tr><td>Temperature</td><td>0.02</td></tr><tr><td>Hard negative num</td><td>8</td></tr><tr><td>GPU config</td><td>8× H800</td></tr></table>

# B Relative RL Reranking Settings

In this stage, we construct overlaps between notes by splitting them, producing 60k training samples. $C _ { \mathrm { p e n } }$ and $R _ { \mathrm { b a s e } }$ are hyperparameters in the reward function. The specific parameters are listed in Table 6.

Table 6: RL training hyperparameters 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Training samples</td><td>60k</td></tr><tr><td>Training epochs</td><td>1</td></tr><tr><td>Batch size</td><td>8</td></tr><tr><td>LoRA rank</td><td>16</td></tr><tr><td>Learning rate</td><td>5e-7</td></tr><tr><td>Precision</td><td>bf16</td></tr><tr><td>Penalty ( $C_{pen}$ )</td><td>-5</td></tr><tr><td>Base reward ( $R_{base}$ )</td><td> $3 \times len(L_{rel})$ </td></tr></table>