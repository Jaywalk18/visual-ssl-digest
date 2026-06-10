# One Token per Multimodal Evidence: Latent Memory for Resource-Constrained QA

Zhi Zheng∗ Ziqiao Meng Hao Luan Wei Liu Wee Sun Lee School of Computing, National University of Singapore

## Abstract

External memory effectively grounds large language models (LLMs) and visionlanguage models (VLMs)-based question answering (QA) in relevant multimodal evidence. However, existing memory paradigms represent each memory item in raw text and image forms, so retrieval-based systems must pass the retrieved text or images to the generation LLMs/VLMs, resulting in high token consumption and storage pressure, making it unaffordable for resource-constrained applications. We propose Latent Memory, a latent-space memory paradigm that replaces each raw text or image evidence item with a single high-dimensional latent token produced by a small compressor LLM/VLM. Rather than retrieving raw evidence for generation, Latent Memory operates in a unified latent representation space: the query is embedded into this space to retrieve relevant latent tokens, and the retrieved latent tokens are directly prompted to a pretrained LLM or VLM for answer generation. To make each latent token simultaneously informative for reconstruction, retrieval, and generation, we train the compressor with reconstruction, contrastive, and distillation objectives in a unified end-to-end manner. Latent Memory is evaluated on seven text-only QA benchmarks (e.g., HotpotQA) and multimodal QA benchmarks, where it achieves competitive QA performance compared to advanced RAG baselines while consuming 3× to 10× fewer generator tokens. It can also deliver the strongest image-grounded QA performance on WebQA.2

## 1 Introduction

Large language models (LLMs) and vision-language models (VLMs) have demonstrated remarkable capabilities in complex reasoning and knowledge-intensive tasks [1, 2, 3, 4], especially when equipped with an external memory [5] and then retrieving the relevant memory items [6] for faithful and reliable generation. This memory usually contains external knowledge evidence or dialogue history in a multimodal form, allowing models to ground their outputs in long-context tasks or multi-turn dialogues, thus becoming a core component of a wide range of applications, including long-context question answering (QA) [7, 4], coding agents [8, 9], and agentic AI assistants [10, 11].

Although external memory can improve answer generation by providing relevant evidence, the prevailing memory paradigm remains computationally expensive. The main bottleneck lies in generation, where the evidence in the memory is prompted to the LLM/VLM generator in uncompressed form, incurring substantial token costs and latency overhead. This problem is further amplified in handling the multimodal evidence in the memory, where each image may require megabytes of storage, and will expand into hundreds of visual tokens during generation [12, 13]. Together, these costs limit the scalability of external memory for long-context and multi-turn interactions, and make it difficult to deploy memory-augmented systems under tight storage or latency constraints, such as on-device assistants and edge-AI applications [14, 15, 16].

![](images/84a3f895fc76c8180c7c7994267532930238f46382f6fc984d2db3e6c5b64a94.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Contextual Memory C"] -->|RAG| B["Context1:......"]
  B --> C["Context2:......"]
  C --> D["Context3: <image>"]
  D --> E["Frozen Larger-Size LLM/VLM for Generation"]
  E --> F["Output"]
  F --> G["Answer A Terry Richardson"]
  H["Query Q"] --> I["Who is older, Annie Morton or Terry Richardson?"]
  J["High Storage Cost"] --> K["Thousands of Tokens"]
  K --> E
    style A fill:#f9f,stroke:#333
    style G fill:#bbf,stroke:#333
```
</details>

(a) The conventional RAG-based contextual generation

![](images/beb0e3177b0215540b5c9a15ce3da27460940f8f8e3c2d688da67390a1a8451d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Query Q"] --> B["Compress query to z̃q ∈ Rd r for retrieval"]
  B --> C["Top Similarity"]
  C --> D["One-Token Each"]
  D --> E["Frozen Larger-Size LLM/VLM for Generation"]
  E --> F["Output"]
  F --> G["Terry Richardson"]
  H["Contextual Memory C"] --> I["LMM/VLM with Trainable LoRA for Compression"]
  I --> J["Compressing each multimodal evidence unit to One-Token"]
  J --> K["Latent Memory M"]
  K --> L["Low Storage Cost"]
  L --> M["Unified Latent Representation Space for Retrieval and Generation"]
  N["Efficient QA with Retrieved Latent Memory (Ours)"] --> O["Query Q"]
  O --> P["Who is older, Annie Morton or Terry Richardson?"]
  P --> Q["Input"]
  Q --> R["Frozen Larger-Size LLM/VLM for Generation"]
  R --> S["Output"]
  S --> T["Terry Richardson"]
  U["1 Transforming Context C to Latent Memory (Ours)"] --> V["Contextual Memory C"]
  V --> W["Annie Morton: Terry Richardson: Image1: with Title: Image2: with Title: LLM/VLM with Compression"]
  W --> X["Compressing each multimodal evidence unit to One-Token"]
  X --> Y["Latent 1: z1 ∈ Rdθ, Latent 2: z2 ∈ Rdθ, Latent 3: z3 ∈ Rdθ"]
  Y --> Z["Low Storage Cost"]
  Z --> AA["Unified Latent Representation Space for Retrieval and Generation"]
  AB["2 Efficient QA with Retrieved Latent Memory (Ours)"] --> AC["Query Q"]
  AC --> AD["Who is older, Annie Morton or Terry Richardson?"]
  AD --> AE["Input"]
  AE --> AF["Frozen Larger-Size LLM/VLM for Generation"]
  AF --> AG["Output"]
  AG --> AH["Terry Richardson"]
  AI["1 Transforming Context C to Latent Memory (Ours)"] --> AJ["Contextual Memory C"]
  AJ --> AK["Annie Morton: Terry Richardson: Image1: with Title: Image2: with Title: LMM/VLM with Compression"]
  AK --> AL["Compressing each multimodal evidence unit to One-Token"]
  AL --> AM["Latent 1: z1 ∈ Rdθ, Latent 2: z2 ∈ Rdθ, Latent 3: z3 ∈ Rdθ"]
  AM --> AN["Low Storage Cost"]
  AN --> AO["Unified Latent Representation Space for Retrieval and Generation"]
  AP["3 Transforming Context C to Latent Memory (Ours)"] --> AQ["Contextual Memory C"]
  AQ --> AR["Annie Morton: Terry Richardson: Image1: with Title: Image2: with Title: LMM/VLM with Compression"]
  AR --> AS["Compressing each multimodal evidence unit to One-Token"]
  AS --> AT["Latent 1: z1 ∈ Rdθ, Latent 2: z2 ∈ Rdθ, Latent 3: z3 ∈ Rdθ"]
  AT --> AU["Low Storage Cost"]
  AU --> AV["Unified Latent Representation Space for Retrieval and Generation"]
  AW["4 Transforming Context C to Latent Memory (Ours)"] --> AX["Contextual Memory C"]
  AX --> AY["Annie Morton: Terry Richardson: Image1: with Title: Image2: with Title: LMM/VLM with Compression"]
  AY --> AZ["Compressing each multimodal evidence unit to One-Token"]
  AZ --> BA["Latent 1: z1 ∈ Rdθ, Latent 2: z2 ∈ Rdθ, Latent 3: z3 ∈ Rdθ"]
  BA --> BB["Low Storage Cost"]
  BB --> BC["Unified Latent Representation Space for Retrieval and Generation"]
  BD["5 Transforming Context C to Latent Memory (Ours)"] --> BE["Contextual Memory C"]
  BE --> BF["Annie Morton: Terry Richardson: Image1: with Title: Image2: with Title: LMM/VLM with Compression"]
  BF --> BG["Compressing each multimodal evidence unit to One-Token"]
  BG --> BH["Latent 1: z1 ∈ Rdθ, Latent 2: z2 ∈ Rdθ, Latent 3: z3 ∈ Rdθ"]
  BH --> BI["Low Storage Cost"]
  BI --> BJ["Unified Latent Representation Space for Retrieval and Generation"]
  BK["6 Transforming Context C to Latent Memory (Ours)"] --> BL["Contextual Memory C"]
  BL --> BM["Annie Morton: Terry Richardson: Image1: with Title: Image2: with Title: LMM/VLM with Compression"]
  BM --> BN["Compressing each multimodal evidence unit to One-Token"]
  BN --> BO["Latent 1: z1 ∈ Rdθ, Latent 2: z2 ∈ Rdθ, Latent 3: z3 ∈ Rdθ"]
  BO --> BP["Low Storage Cost"]
  BP --> BQ["Unified Latent Representation Space for Retrieval and Generation"]
  BR["7 Transforming Context C to Latent Memory (Ours)"] --> BS["Contextual Memory C"]
  BS --> BT["Annie Morton: Terry Richardson: Image1: with Title: Image2: with Title: LMM/VLM with Compression"]
  BT --> BU["Compressing each multimodal evidence unit to One-Token"]
  BU --> BV["Latent 1: z1 ∈ Rdθ, Latent 2: z2 ∈ Rdθ, Latent 3: z3 ∈ Rdθ"]
  BV --> BW["Low Storage Cost"]
  BW --> BX["Unified Latent Representation Space for Retrieval and Generation"]
  BY["8 Transforming Context C to Latent Memory (Ours)"] --> CA["Contextual Memory C"]
  CA --> CB["Annie Morton: Terry Richardson: Image1: with Title: Image2: with Title: LMM/VLM with Compression"]
  CB --> CC["Compressing each multimodal evidence unit to One-Token"]
  CC --> CD["Latent 1: z1 ∈ Rdθ, Latent 2: z2 ∈ Rdθ, Latent 3: z3 ∈ Rdθ"]
  CD --> CE["Low Storage Cost"]
  CE --> CF["Unified Latent Representation Space for Retrieval and Generation"]
  CG["9 Transforming Context C to Latent Memory (Ours)"] --> CH["Contextual Memory C"]
  CH --> CI["Annie Morton: Terry Richardson: Image1: with Title: Image2: with Title: LMM/VLM with Compression"]
  CI --> CJ["Compressing each multimodal evidence unit to One-Token"]
  CJ --> DA["Latent 1: z1 ∈ Rdθ, Latent 2: z2 ∈ Rdθ, Latent 3: z3 ∈ Rdθ"]
  DA --> DB["Low Storage Cost"]
  DB --> DC["Unified Latent Representation Space for Retrieval and Generation"]
  DD["10 Transforming Context C to Latent Memory (Ours)"] --> DE["Contextual Memory C"]
  DE --> DF["Annie Morton: Terry Richardson: Image1: with Title: Image2: with Title: LMM/VLM with Compression"]
  DF --> DG["Compressing each multimodal evidence unit to One-Token"]
  DG --> DH["Latent 1: z1 ∈ Rdθ, Latent 2: z2 ∈ Rdθ, Latent 3: z3 ∈ Rdθ"]
  DH --> DI["Low Storage Cost"]
  DI --> DJ["Unified Latent Representation Space for Retrieval and Generation"]
  DK["11 Transforming Context C to Latent Memory (Ours)"] --> DL["Contextual Memory C"]
  DL --> DV["Annie Morton: Terry Richardson: Image1: with Title: Image2: with Title: LMM/VLM with Compression"]
  DV --> DW["Compressing each multimodal evidence unit to One-Token"]
  DW --> DX["Latent 1: z1 ∈ Rdθ, Latent 2: z2 ∈ Rdθ, Latent 3: z3 ∈ Rdθ"]
  DX --> DY["Low Storage Cost"]
  DY --> DYD["Better Efficiency-Performance Trade-off"]
```
</details>

(b) Generation with retrievable Latent Memory (Ours)  
Figure 1: (a) shows the existing pipeline for memory-based generation. To improve storage efficiency and token efficiency, our Latent Memory (b) can compress each multimodal evidence into one latent token, which achieves better retrieval ability and competitive generation performances.

In pursuit of an efficient memory representation paradigm for applications with strict latency or storage constraints, this paper investigates whether multimodal contextual memory can be represented as highly compressed one-token latent representations capable of replacing raw evidence during generation. To this end, we propose Latent Memory, a framework in which each unit of evidence is compressed into a compact, retrievable, high-dimensional latent token that can be directly utilized by a frozen LLM/VLM generator without fine-tuning. As illustrated in Figure 1(b), Latent Memory uses a small compression LLM/VLM to encode each text or image context into a single latent token, forming a unified multimodal latent representation space. At query time, the question is embedded into the same space to retrieve relevant latent tokens through similarity search. These latent tokens are then projected into the hidden dimension of a frozen LLM or VLM for answer generation, replacing the conventional token-based context. This design enables:

(1) Token Efficiency and Storage Efficiency: The proposed Latent Memory leads to fewer token consumptions and lower storage pressure to handle contextual multimodal evidence.  
(2) Unified Representation and Retrieval Space: The latent tokens are used for both contextual representation and retrieval, making it easier to obtain better low-budget embeddings.

To obtain better latent tokens for multimodal evidence, we fine-tune a small LLM/VLM for tokens that preserve both the raw information and retrieval capability. So, the training loss combines: (i) a reconstruction loss to ensure latent tokens carry sufficient information of the original evidence; (ii) a contrastive loss to align queries with only their labeled supporting evidence in the latent space; and (iii) a distillation loss to align the generation behavior prompted on latent tokens with that of the behavior generator prompted with raw positive evidence. We evaluate Latent Memory on both text-only and multimodal benchmarks. On 2WikiMultihopQA and MuSiQue, one-token Latent Memory trained on HotpotQA attains competitive EM/F1 compared to raw-evidence-level retrieval augmented generation (RAG) baselines using only 3× less LLaMA-8B or Mistral-7B generator tokens. On multimodal QA benchmarks, Latent Memory yields 10× efficiency gains and the outstanding image-grounded QA results under the LLaVA-13B and Gemma3-12B generators. Our contributions are as follows:

• We propose Latent Memory, an efficient multimodal memory paradigm for resource-constrained scenarios, which compiles each evidence item into a single high-dimensional latent token.

• We show that one-token Latent Memory can support both retrieval and answer generation through a training objective that combines reconstruction, contrastive learning, and distillation.  
• Empirically, Latent Memory substantially improves the efficiency-performance trade-off across eight text-only and multimodal contextual QA benchmarks, and four generator LLM/VLMs.

## 2 Related Work

External Memory and RAG. For faithful QA, LLM/VLM usually takes the provided multimodal context as memory and cooperates with the external memory via full-context prompting [17] or RAG [18, 19, 20]. Dense Retrieval [21], BM25 Retrieval [22], and other recent RAG methods [23, 24] encode the feature of queries and evidence into a shared space for the nearest-neighbor search. In multimodal settings, recent multimodal embedding methods [25, 26, 27] retrieve images based on embedding models and prompt image tokens (576 in LLaVA VLMs [12] and 256 in Gemma VLMs [13]) to VLMs. However, all of these methods represent memory at the raw-evidence level; as the multimodal memory scales up, this representation becomes a storage and efficiency bottleneck. Latent Memory stores multimodal evidence as compact latent tokens for less storage space, and injects only latent vectors rather than raw evidence into the generation LLM for token efficiency.

Evidence Compression. Besides the RAG method, there are also Evidence Compression methods proposed to address the over-lengthened large memory. LLMLingua [28] learns to drop unnecessary tokens, AutoCompressor [29], ICAE [30], and LCC [31] learn to compress text documents into a small number of continuous tokens for efficient LLM-based QA generation, while xRAG [32], and the concurrent work, CLaRa [33] take the same latent for retrieval. However, these methods focus only on text-based situations, making them incompatible with recent multimodal memory settings. Moreover, as detailed in Appendix $\mathsf { A } . 2 ,$ considering that high-dimensional tokens are actually larger than the raw text evidence units, these algorithms cannot provide the storage saving as Latent Memory does in multimodal settings (discussed in Appendix B.5).

Latent-Space Reasoning. Latent reasoning methods [34, 35, 36, 37, 33, 31, 38, 39, 40] show that LLMs can reason and communicate through continuous latent states rather than discrete token sequences, while [41, 42, 43, 44] tries to represent multimodal comprehension and generation in a unified embedding space. Latent Memory applies the idea principle to design a new memory paradigm, where multimodal evidence is encoded, retrieved, and prompted in a unified latent representation.

## 3 Methodology: Latent Memory

To achieve efficient external memory, this paper proposes the Latent Memory paradigm. In this section, we first present how the one-token Latent Memory is built and used for QA generation. Then, we will describe how the compression LLM/VLM is trained to produce the Latent Memory.

## 3.1 Definition: Latent Memory for QA with Contextual Memory

A QA problem seeks the answer $\pmb { A } = \left( a _ { 1 } , a _ { 2 } , \dotsc , a _ { | \pmb { A } | } \right)$ of question $Q = \left( q _ { 1 } , q _ { 2 } , \dots , q _ { | Q | } \right)$ with N external contexts $\boldsymbol { \mathcal { C } } = \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N }$ . Prompting all the contexts C for generation will improve perplexity and may exceed the pre-trained context window in some cases. So RAG systems are usually employed to retrieve a subset of raw evidence $\mathcal { R } ( q , \mathcal { C } )$ and prompt them to a LLM/VLM generator as follows:

$$
P (\boldsymbol {A} \mid \boldsymbol {Q}, \mathcal {C}, \phi) = \prod_ {t = 1} ^ {| \boldsymbol {A} |} P (a _ {t} \mid a _ {<   t}, \boldsymbol {Q}, \mathcal {R} (\boldsymbol {Q}, \mathcal {C}), \phi). \tag {1}
$$

As shown in Figure 1(b), Latent Memory changes this interface by storing and passing raw evidence to a collection of compact latent tokens. Latent tokens serve as a retrieval representation for selecting relevant evidence, and then the retrieved tokens are directly prompted to the generator as a continuous evidence token. The full inference-time pipeline consists of four components.

Memory Compression. For each evidence item $\mathbf { { \boldsymbol { x } } } _ { i } \in { \mathcal { C } } ,$ , we use a small compressor LLM/VLM θ to produce a single Latent Memory token. Concretely, xi is appended with a learnable embedding (noted [MEM]), and we take the final hidden state of this token as the latent token:

$$
\boldsymbol {z} _ {i} = \theta (\boldsymbol {x} _ {i}, [ \mathrm{MEM} ]) \in \mathbb {R} ^ {d _ {\theta}}, \tag {2}
$$

![](images/2ce62825f3ebb59f413d6e37b8aed9767bc7b9c0ecf40db1e2399448a88e5755.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Amanda Lepore: Input"] --> B["LLM/VLM θ with Trainable LoRA for Compression"]
  B --> C["Output"]
  C --> D["Latent Token z ∈ Rd"]
  D --> E["LLM π with Trainable LoRA for Text Reconstruction"]
  E --> F["Supervision"]
  F --> G["Amanda Lepore: Supervision"]
  E --> H["MLP"]
  H --> I["Supervision"]
  I --> J["Ltext = CE(text), Limg = λimg MSE(image)"]
  J --> K["Embeddings of Image"]
```
</details>

Contrastive Loss ?Contrast for Latent Memory Retrieval

![](images/f83cd382e18336f35cb6dd34121eea76d6adef647499854f42a87ea428a46ac5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Positive Contexts C⁺"] --> B["Input"]
  C["Negative Contexts C⁻"] --> D["Input"]
  E["Query Q"] --> F["Input"]
  B --> G["LLM/VLM θ with Trainable LoRA for Compression"]
  D --> G
  F --> G
  G --> H["Output"]
  H --> I["Constructing Latent Memory M⁺"]
  G --> J["Output"]
  J --> K["Constructing Latent Memory M⁻"]
  G --> L["Output"]
  L --> M["Latent Query Token q̃ ∈ Rd"]
  I --> N["z ∈ M⁻"]
  K --> O["q̃ ∈ M⁻"]
  M --> P["LATENT Representation Space"]
  P --> Q["LContrast = (−1/(|M⁺|) ∑z_j ∈ M⁺) log( exp( z̃_jᵀq̃/τ) / ∑z̃_k ∈ M⁺ ∩ M⁻) exp( z̃_kᵀq̃/τ) "]
```
</details>

Distillation Loss $\pmb { L } _ { \mathrm { D i s t i l l } }$ for Latent Token Representations  
![](images/52563dbfb651579f95a5020e82e3ea6271549463207881f75408c8e50c868846.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Prompt with C+"] --> B["Context 1: Context 2: Who is older, Annie Morton or Terry Richardson?"]
  C["Latent Prompt with M+"] --> D["Latent context 1: z₁; Latent context 2: z₂ Who is older, Annie Morton or Terry Richardson?"]
  E["Frozen Larger-Size Generation LLM/VLM φ"] --> F["Input"]
  F --> G["Output"]
  G --> H["Answer A"]
  H --> I["Terry Richardson"]
  J["Ref Answer A"] --> K["Terry Richardson"]
  L["L_Distill = KL(A', A)"] --> H
```
</details>

Figure 2: The training process of the compressor and decoder consists of three losses. Reconstruction Loss ${ \mathcal { L } } _ { \mathrm { { R e c o n } } }$ aims at recovering the raw text and image in a teacher-forcing way. Contrastive Loss $\mathcal { L } _ { \mathrm { { C o n s t r a s t } } }$ aligns the query embedding to positive latent tokens. Distillation Loss $\mathcal { L } _ { \mathrm { D i s t i l l } }$ aligns the generator output between prompting raw evidence and latent tokens.

where $d _ { \theta }$ is the hidden dimension of the model θ. The Latent Memory $\mathcal { M } = \{ \boldsymbol { z } _ { i } \} _ { i = 1 } ^ { N }$ is then formed by collecting all these latent tokens while discarding the original raw evidence.

Retrieving Latent Memory. Prompting generators with compressed contexts may be enough for faithful generation within the context window. However, with a large amount of context, the corresponding latent tokens still pose a significant processing difficulty. So, we design to make the Latent Memory token retrievable, projecting $z _ { i }$ into a $d _ { r }$ dimensional retrieval space with an MLP projection, noted $\widetilde { \pmb z } _ { i } = { \bf M } { \bf L P } _ { r } ( { \pmb z } _ { i } )$ , $\check { \mathrm { M L P } } _ { r } : \mathbb { R } ^ { \check { d } _ { r } }  \mathbb { R } ^ { d _ { \theta } }$ . In retrieving the most relevant latent econtext for the query q, we compress the query into a query representation $\widetilde { \pmb q } \in \mathbb { R } ^ { d _ { r } }$ in this retrieval space. We then retrieve the top-k memories by inner product:

$$
\mathcal {I} = \underset {i} {\arg \text { top - k }} \widetilde {\boldsymbol {q}} ^ {\top} \widetilde {\boldsymbol {z}} _ {i}. \tag {3}
$$

Latent-Conditioned Generation. Retrieved latent tokens $\{ z _ { i } : i \in \mathcal { T } \}$ are mapped into the higherdimensional hidden space of the frozen LLM/VLM $\phi$ for generation. Let $\widehat { \boldsymbol { z } } _ { j }$ denote the projected blatent token corresponding to the j-th retrieved memory. We construct the input embeddings of the generator with another projector $\dot { W } _ { g } \in \mathbb { R } ^ { d _ { \phi } \times d _ { \theta } }$ and prompt them to ϕ as follows:

$$
P (\boldsymbol {A} \mid \boldsymbol {Q}, \mathcal {C}, \phi , \theta) = \prod_ {t = 1} ^ {| \boldsymbol {A} |} P (a _ {t} \mid a _ {<   t}, \boldsymbol {Q}, \widehat {\boldsymbol {z}} _ {1}, \dots , \widehat {\boldsymbol {z}} _ {k}, \phi), \quad \text { where } \quad \widehat {\boldsymbol {z}} _ {i} = \boldsymbol {W} _ {g} \boldsymbol {z} _ {i}. \tag {4}
$$

Reconstruction (Optional). To preserve some interpretability, Latent Memory can be roughly reconstructed back to the raw multimodal context. Through a fine-tuned decoder π, we can recover the raw-text for text-evidence or the caption of an image evidence in an autoregressive manner as follows:

$$
P (\boldsymbol {x} _ {i} \mid \boldsymbol {z} _ {i}) = \prod_ {t = 1} ^ {| \boldsymbol {x} _ {i} |} P (\boldsymbol {x} _ {i, t} \mid \boldsymbol {x} _ {i, <   t}, \boldsymbol {z} _ {i}, \pi). \tag {5}
$$

Latent Memory also allows image reconstruction for the latent tokens that are compressed from an image. We train a multi-layer perceptron (MLP) to predict the CLIP embedding of the image [45]. Then, the image can be roughly recovered with a pre-trained diffusion-based image generator unCLIP [46], conditioning on the recovered CLIP embedding.

## 3.2 Training the Compressor for Latent Memory

Latent Memory carries retrieval, information-providing, and optional reconstruction roles. In this subsection, we will introduce the algorithm to fine-tune a powerful compressor $\theta$ so that one latent token can support all these roles simultaneously.

As illustrated in Figure 2, the training procedure optimizes the compressor with three complementary signals. (1) A reconstruction objective encourages each compressed latent token $z _ { i }$ to preserve the content of its original evidence. (2) A contrastive objective shapes the retrieval space by pulling queries close only to their supporting evidence. (3) A distillation objective aligns the behavior of the frozen generator conditioned on latent memories with the behavior of the same generator conditioned on raw evidence. To avoid catastrophic forgetting, the large generator $\phi$ is kept frozen throughout training. We only optimize the LoRAs of compressor LLM/ LLM θ, reconstruction decoder LLM π, and retrieve and generate projections $W _ { r }$ and $W _ { g }$ . The training process only requires supervision from positive samples, and does not need other supervision signals, e.g., labeled answers.

Multimodal Reconstruction Loss. A one-token memory should not collapse into a purely discriminative retrieval identifier. It should still preserve recoverable information about the evidence item it represents. Therefore, for a text evidence item $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ (or the caption of an image), we fine-tune an LLM decoder π to reconstruct $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ in its Latent Memory $z _ { i }$ compressed with θ and a trainable LoRA form Eq. (2) in a teach-forcing way as follows:

$$
\mathcal {L} _ {\text { Recon }} ^ {\text { text }} = - \sum_ {i} \sum_ {t = 1} ^ {| \boldsymbol {x} _ {i} |} \log P _ {\pi} \left(x _ {i, t} \mid \boldsymbol {x} _ {i, <   t}, \boldsymbol {z} _ {i}\right). \tag {6}
$$

For image evidence, we do not reconstruct raw pixels. Instead, following the idea of unCLIP-style reconstruction [46], we reconstruct the CLIP image embedding of the original image. Given an image $\pmb { x } _ { i } ^ { \mathrm { i m g } }$ $z _ { i } ^ { \mathrm { i m g } }$ image embedding ${ \mathbf { } } v _ { i }$ with the loss function as follows:

$$
\mathcal {L} _ {\text { Recon }} ^ {\text { img }} = \sum_ {i} \| \boldsymbol {v} _ {i} - \mathrm{MLP} (\boldsymbol {z} _ {i}) \| _ {2} ^ {2}. \tag {7}
$$

In multimodal training, the reconstruction term combines the available text-side and image-side reconstruction signals as $\mathcal { L } _ { \mathrm { R e c o n } } = \mathcal { L } _ { \mathrm { R e c o n } } ^ { \mathrm { t e x t } } + \lambda _ { \mathrm { i m g } } \mathcal { L } _ { \mathrm { R e c o n } } ^ { \mathrm { i m g } }$ + λimgL Recon.

Contrastive Retrieval Loss. The retrieval projection in Section 3.1 maps each Latent Memory zi into a retrieval vector $\widetilde { z } _ { i }$ . To make this space useful for evidence selection, we train another LoRA eof compressor for query representation $\widetilde { \pmb q }$ to be close to the latent memories of supporting evidence eand far from irrelevant memories. For each question $Q ,$ , let $\mathcal { M } ^ { + }$ and $\mathcal { M } ^ { - }$ denote its positive and sampled negative latent evidence, respectively. We use a multi-positive contrastive objective, where each positive contributes one InfoNCE [47] term:

$$
\mathcal {L} _ {\text { Contrast }} = \frac {1}{| \mathcal {M} _ {i} ^ {+} |} \sum_ {\widetilde {\boldsymbol {z}} _ {j} \in \mathcal {M} ^ {+}} - \log \frac {\exp \left(\widetilde {\boldsymbol {q}} ^ {\top} \widetilde {\boldsymbol {z}} _ {j} / \tau\right)}{\sum_ {\widetilde {\boldsymbol {z}} _ {k} \in \mathcal {M} ^ {+} \cup \mathcal {M} ^ {-}} \exp \left(\widetilde {\boldsymbol {q}} ^ {\top} \widetilde {\boldsymbol {z}} _ {k} / \tau\right)}, \tag {8}
$$

where $\tau$ is the temperature. Since both text and image evidence are projected into the same retrieval space, the same loss supports unified retrieval over mixed multimodal memory.

Generation distillation loss. To ensure latent tokens have similar roles compared to raw evidence, we add a distillation objective. For each training example, the teacher distribution is obtained by autoregressively running the frozen generator $\phi$ with the raw supporting context $\mathcal { C } ^ { + }$ . This produces $\pmb { A } ^ { \mathrm { t e a } } = ( a _ { 1 } ^ { \mathrm { t e a } } , \cdot \cdot \cdot , a _ { | \pmb { A } ^ { \mathrm { t e a } } | } ^ { \mathrm { t e a } } )$ $\phi ,$ replaces the raw context with the projected latent memories $\widehat { z } _ { 1 } , \ldots , \widehat { z } _ { k }$ . We then minimize the btoken-level KL divergence along the teacher-generated trajectory:

$$
\mathcal {L} _ {\text { Distill }} = \sum_ {t = 1} ^ {| A ^ {\text { tea }} |} \mathbb {K L} \left(P (\cdot | a _ {<   t} ^ {\text { tea }}, Q, \mathcal {C} ^ {+}), \phi\right). \| P (\cdot | a _ {<   t} ^ {\text { tea }}, Q, \widehat {z} _ {1}, \dots , \widehat {z} _ {k}, \phi)\left. \right). \tag {9}
$$

This term teaches the compressor and generator projection to produce latent tokens that the frozen generator can interpret as evidence. In this way, distillation loss connects the latent retrieval interface to the final QA objective without fine-tuning the large LLM/VLM generator.

Table 1: Text-based QA results using Meta-Llama-3-8B-Instruct as the generation LLM. All methods use a frozen Meta-Llama-3-8B-Instruct generator. The Average columns report the out-of-domain average over 2WikiMultihopQA and MuSiQue. Bold indicates the best metric in each column, and underlining indicates the second-best one. R@k = Recall@k.

<table><tr><td colspan="17">Generation LLM (fixed): Meta-Llama-3-8B-Instruct</td></tr><tr><td rowspan="3">Dataset Method</td><td colspan="4">In-Domain</td><td colspan="12">Out-of-Domain</td></tr><tr><td colspan="4">HotpotQA</td><td colspan="4">2WikiMultihopQA</td><td colspan="4">MuSiQue</td><td colspan="4">Average</td></tr><tr><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td></tr><tr><td>Full Context</td><td>42.0</td><td>57.8</td><td>-</td><td>1462</td><td>17.7</td><td>39.2</td><td>-</td><td>1074</td><td>6.0</td><td>17.1</td><td>-</td><td>2580</td><td>11.9</td><td>28.2</td><td>-</td><td>1827</td></tr><tr><td colspan="17">Evidence Compression Baselines</td></tr><tr><td>LLMLingua (20%)</td><td>31.8</td><td>44.8</td><td>-</td><td>283</td><td>17.0</td><td>30.4</td><td>-</td><td>199</td><td>11.6</td><td>21.8</td><td>-</td><td>492</td><td>14.3</td><td>26.1</td><td>-</td><td>346</td></tr><tr><td>LLMLingua (10%)</td><td>25.0</td><td>36.1</td><td>-</td><td>154</td><td>14.8</td><td>24.7</td><td>-</td><td>108</td><td>6.9</td><td>14.9</td><td>-</td><td>259</td><td>10.9</td><td>19.8</td><td>-</td><td>184</td></tr><tr><td>LLMLingua (5%)</td><td>20.9</td><td>30.2</td><td>-</td><td>87</td><td>15.0</td><td>22.2</td><td>-</td><td>63</td><td>4.3</td><td>10.6</td><td>-</td><td>137</td><td>9.7</td><td>16.4</td><td>-</td><td>100</td></tr><tr><td colspan="17">RAG Baselines</td></tr><tr><td>BM25 Retrieval (k=1)</td><td>32.5</td><td>44.8</td><td>30.2</td><td>68</td><td>17.2</td><td>27.2</td><td>18.3</td><td>60</td><td>6.2</td><td>14.2</td><td>7.0</td><td>69</td><td>11.7</td><td>20.7</td><td>12.7</td><td>65</td></tr><tr><td>BM25 Retrieval (k=2)</td><td>36.8</td><td>49.9</td><td>45.5</td><td>106</td><td>16.3</td><td>28.3</td><td>29.2</td><td>94</td><td>8.0</td><td>16.7</td><td>12.2</td><td>106</td><td>12.2</td><td>22.5</td><td>20.7</td><td>100</td></tr><tr><td>BM25 Retrieval (k=5)</td><td>41.3</td><td>55.3</td><td>65.5</td><td>224</td><td>16.3</td><td>32.7</td><td>46.9</td><td>196</td><td>9.5</td><td>19.6</td><td>22.4</td><td>221</td><td>12.9</td><td>26.2</td><td>34.7</td><td>209</td></tr><tr><td>Dense Retrieval (k=1)</td><td>29.4</td><td>41.0</td><td>27.9</td><td>67</td><td>19.0</td><td>30.0</td><td>23.3</td><td>61</td><td>7.8</td><td>17.1</td><td>9.6</td><td>70</td><td>13.4</td><td>23.6</td><td>16.5</td><td>66</td></tr><tr><td>Dense Retrieval (k=2)</td><td>32.6</td><td>45.4</td><td>42.2</td><td>104</td><td>18.1</td><td>31.7</td><td>37.6</td><td>95</td><td>9.9</td><td>19.9</td><td>17.0</td><td>106</td><td>14.0</td><td>25.8</td><td>27.3</td><td>101</td></tr><tr><td>Dense Retrieval (k=5)</td><td>37.0</td><td>50.8</td><td>62.4</td><td>214</td><td>19.1</td><td>37.1</td><td>58.0</td><td>198</td><td>12.8</td><td>23.4</td><td>31.3</td><td>218</td><td>16.0</td><td>30.3</td><td>44.7</td><td>208</td></tr><tr><td>Qwen3-Emb-0.6B (k=1)</td><td>30.9</td><td>43.6</td><td>33.2</td><td>70</td><td>18.1</td><td>30.9</td><td>32.5</td><td>66</td><td>8.1</td><td>18.2</td><td>10.3</td><td>73</td><td>13.1</td><td>24.6</td><td>21.4</td><td>70</td></tr><tr><td>Qwen3-Emb-0.6B (k=2)</td><td>35.6</td><td>49.0</td><td>50.1</td><td>109</td><td>17.5</td><td>33.5</td><td>47.7</td><td>102</td><td>9.8</td><td>20.4</td><td>18.5</td><td>112</td><td>13.7</td><td>27.0</td><td>33.1</td><td>107</td></tr><tr><td>Qwen3-Emb-0.6B (k=5)</td><td>40.0</td><td>54.5</td><td>70.1</td><td>224</td><td>19.1</td><td>38.6</td><td>64.3</td><td>208</td><td>13.7</td><td>24.7</td><td>34.8</td><td>230</td><td>16.4</td><td>31.7</td><td>49.6</td><td>219</td></tr><tr><td colspan="17">Ours: Latent Memory</td></tr><tr><td>Latent Memory (k=1)</td><td>27.4</td><td>39.4</td><td>34.6</td><td>36</td><td>19.8</td><td>29.2</td><td>28.4</td><td>33</td><td>5.8</td><td>14.6</td><td>8.7</td><td>37</td><td>12.8</td><td>21.9</td><td>18.6</td><td>35</td></tr><tr><td>Latent Memory (k=2)</td><td>31.6</td><td>45.2</td><td>62.6</td><td>45</td><td>21.5</td><td>33.5</td><td>49.5</td><td>42</td><td>7.3</td><td>16.9</td><td>15.5</td><td>46</td><td>14.4</td><td>25.2</td><td>32.5</td><td>44</td></tr><tr><td>Latent Memory (k=5)</td><td>34.8</td><td>48.9</td><td>87.1</td><td>72</td><td>24.3</td><td>36.7</td><td>74.2</td><td>69</td><td>8.7</td><td>19.2</td><td>30.1</td><td>73</td><td>16.5</td><td>28.0</td><td>52.2</td><td>71</td></tr></table>

Finally, the overall training objective is as follows, where $\lambda _ { \mathrm { R e c o n } } , \lambda _ { \mathrm { C o n t r a s t } }$ , and $\lambda _ { \mathrm { D i s t i l l } }$ control the relative weights of reconstruction, retrieval, and distillation, respectively:

$$
\mathcal {L} = \lambda_ {\text {Recon}} \mathcal {L} _ {\text {Recon}} + \lambda_ {\text {Contrast}} \mathcal {L} _ {\text {Contrast}} + \lambda_ {\text {Distill}} \mathcal {L} _ {\text {Distill}}, \tag {10}
$$

## 4 Experiments

We evaluate Latent Memory in two settings: (1) text-only contextual QA and (2) multimodal contextual QA. In the main text-only setting, we use a frozen Meta-Llama-3-8B-Instruct [48] generator ϕ and fine-tune LLaMA-3.2-1B-Instruct [48] with LoRA adapter as the compressor θ. To compare against text-only latent-context baselines that are built around Mistral, we additionally report a frozen Mistral-7B-Instruct [49] generator setting, where both the Latent Memory compressor/encoder and the reconstruction decoder are LLaMA-3.2-1B-Instruct. In the multimodal setting, we use a frozen LLaVA-1.5-13B [12] generator and fine-tune LLaVA-1.5-7B [12] with LoRA adapters as the compressor. We also include another setting with Gemma-3-12B-Instruct [13] generator and Gemma-3-4B [13] compressor for multimodal QA in Appendix C.6. For all reported settings, the reconstruction decoder π is fine-tuned from LLaMA-3.2-1B-Instruct. Experiments are done on a Nvidia H200 141GB GPU. Full training and evaluation details are deferred to Appendix B.

Dataset. In the text-only setting, the Latent Memory is trained on HotpotQA training dataset and evaluated on the validation set of in-domain HotpotQA and out-of-domain 2WikiMultihopQA, MuSiQue. To investigate a larger transfer, Appendix C.2 adds a generalization-on-more-domains suite that spans open-domain factoid QA and scientific-document QA. WebQA [50] is used for multimodal training and evaluation. In testing on its validation set, we report image-grounded (n=2,511) and textgrounded (n=2,455) subsets separately, while retrieval itself remains unified over a mixed text-image candidate pool. We also consider multimodal domain transfer on SlideQA in Appendix C.2. For the Text-only setting, all evidence is processed in the “Title: Sentence“ form. We process evidence as “Title: Evidence“ and “Caption: Image“ forms for the multimodal setting.

Baselines. We include two categories of baselines. (1) Context-based baselines, including generation with full-context and evidence-compression baselines LLMLingua [28], xRAG [32], and CLaRa [33]. (2) RAG baselines, including BM25 Retrieval [22], Dense Retrieval [21], Retrieval with practical off-the-shelf Qwen-3-Embedding [51] (and its variant fine-tuned on in-domain), Nemo Retriever [52], and Qwen-3-VL-Embedding [25]. All baselines and the proposed Latent Memory-based QA use the same pre-trained generator. Baseline details are in Appendix E.

Table 2: Text-based QA results using Mistral-7B-Instruct as the generation LLM. The Latent Memory rows use LLaMA-3.2-1B-Instruct as both compressor/encoder and reconstruction decoder. xRAG and CLaRa use their pretrained Mistral-based checkpoints. The Average columns report the out-ofdomain average over 2WikiMultihopQA and MuSiQue.

<table><tr><td colspan="17">Generation LLM (fixed): Mistral-7B-Instruct</td></tr><tr><td rowspan="3">Dataset Method</td><td colspan="4">In-Domain</td><td colspan="12">Out-of-Domain</td></tr><tr><td colspan="4">HotpotQA</td><td colspan="4">2WikiMultihopQA</td><td colspan="4">MuSiQue</td><td colspan="4">Average</td></tr><tr><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td></tr><tr><td>Full Context</td><td>21.0</td><td>44.9</td><td>-</td><td>1701</td><td>3.8</td><td>27.1</td><td>-</td><td>1244</td><td>2.8</td><td>14.9</td><td>-</td><td>3012</td><td>3.3</td><td>21.0</td><td>-</td><td>2128</td></tr><tr><td colspan="17">Evidence Compression Baselines</td></tr><tr><td>LLMLingua (20%)</td><td>8.9</td><td>27.6</td><td>-</td><td>406</td><td>1.4</td><td>18.4</td><td>-</td><td>286</td><td>1.3</td><td>10.3</td><td>-</td><td>708</td><td>1.3</td><td>14.4</td><td>-</td><td>497</td></tr><tr><td>LLMLingua (10%)</td><td>6.8</td><td>22.6</td><td>-</td><td>217</td><td>0.6</td><td>15.1</td><td>-</td><td>149</td><td>0.9</td><td>7.4</td><td>-</td><td>371</td><td>0.8</td><td>11.2</td><td>-</td><td>260</td></tr><tr><td>LLMLingua (5%)</td><td>4.7</td><td>17.9</td><td>-</td><td>117</td><td>0.4</td><td>13.0</td><td>-</td><td>81</td><td>0.5</td><td>5.5</td><td>-</td><td>191</td><td>0.5</td><td>9.3</td><td>-</td><td>136</td></tr><tr><td>xRAG</td><td>12.1</td><td>25.3</td><td>-</td><td>42</td><td>3.5</td><td>17.6</td><td>-</td><td>38</td><td>1.4</td><td>9.5</td><td>-</td><td>42</td><td>2.5</td><td>13.6</td><td>-</td><td>40</td></tr><tr><td>CLaRa-16x (k=5)</td><td>5.2</td><td>15.3</td><td>55.8</td><td>119</td><td>5.1</td><td>16.2</td><td>26.5</td><td>115</td><td>0.7</td><td>6.0</td><td>22.3</td><td>119</td><td>2.9</td><td>11.1</td><td>24.4</td><td>117</td></tr><tr><td colspan="17">RAG and Latent-context Baselines</td></tr><tr><td>BM25 Retrieval (k=1)</td><td>11.0</td><td>29.6</td><td>30.2</td><td>76</td><td>0.4</td><td>14.4</td><td>18.3</td><td>66</td><td>0.9</td><td>8.1</td><td>6.9</td><td>75</td><td>0.6</td><td>11.3</td><td>12.6</td><td>71</td></tr><tr><td>BM25 Retrieval (k=2)</td><td>13.8</td><td>33.8</td><td>45.5</td><td>120</td><td>0.7</td><td>15.8</td><td>29.2</td><td>104</td><td>1.7</td><td>9.5</td><td>12.1</td><td>119</td><td>1.2</td><td>12.6</td><td>20.7</td><td>111</td></tr><tr><td>BM25 Retrieval (k=5)</td><td>15.9</td><td>37.8</td><td>65.5</td><td>255</td><td>1.0</td><td>18.6</td><td>46.9</td><td>221</td><td>1.7</td><td>10.7</td><td>22.4</td><td>250</td><td>1.3</td><td>14.7</td><td>34.6</td><td>236</td></tr><tr><td>Dense Retrieval (k=1)</td><td>9.0</td><td>26.4</td><td>27.9</td><td>75</td><td>0.6</td><td>15.8</td><td>23.3</td><td>67</td><td>1.0</td><td>10.2</td><td>9.5</td><td>77</td><td>0.8</td><td>13.0</td><td>16.4</td><td>72</td></tr><tr><td>Dense Retrieval (k=2)</td><td>11.3</td><td>30.2</td><td>42.2</td><td>117</td><td>1.0</td><td>17.9</td><td>37.6</td><td>107</td><td>1.2</td><td>11.1</td><td>16.9</td><td>119</td><td>1.1</td><td>14.5</td><td>27.3</td><td>113</td></tr><tr><td>Dense Retrieval (k=5)</td><td>13.9</td><td>34.0</td><td>62.4</td><td>244</td><td>2.7</td><td>22.5</td><td>58.0</td><td>225</td><td>1.6</td><td>12.8</td><td>31.3</td><td>247</td><td>2.2</td><td>17.6</td><td>44.7</td><td>236</td></tr><tr><td>Qwen3-Emb-0.6B (k=1)</td><td>10.3</td><td>28.4</td><td>33.2</td><td>79</td><td>0.7</td><td>17.1</td><td>32.5</td><td>74</td><td>1.0</td><td>10.6</td><td>10.3</td><td>81</td><td>0.9</td><td>13.8</td><td>21.4</td><td>77</td></tr><tr><td>Qwen3-Emb-0.6B (k=2)</td><td>12.6</td><td>32.7</td><td>50.1</td><td>124</td><td>1.2</td><td>19.8</td><td>47.7</td><td>115</td><td>1.5</td><td>11.2</td><td>18.5</td><td>126</td><td>1.3</td><td>15.5</td><td>33.1</td><td>121</td></tr><tr><td>Qwen3-Emb-0.6B (k=5)</td><td>15.1</td><td>36.6</td><td>70.1</td><td>256</td><td>3.0</td><td>24.0</td><td>64.3</td><td>236</td><td>1.6</td><td>13.4</td><td>34.8</td><td>262</td><td>2.3</td><td>18.7</td><td>49.6</td><td>249</td></tr><tr><td>Qwen3-Emb-0.6B-ft (k=1)</td><td>12.3</td><td>32.2</td><td>37.9</td><td>80</td><td>0.9</td><td>17.0</td><td>35.5</td><td>74</td><td>1.3</td><td>11.4</td><td>9.8</td><td>83</td><td>1.1</td><td>14.2</td><td>22.6</td><td>79</td></tr><tr><td>Qwen3-Emb-0.6B-ft (k=2)</td><td>16.3</td><td>38.4</td><td>59.7</td><td>127</td><td>1.7</td><td>21.5</td><td>54.4</td><td>119</td><td>1.4</td><td>12.1</td><td>16.5</td><td>133</td><td>1.6</td><td>16.8</td><td>35.4</td><td>126</td></tr><tr><td>Qwen3-Emb-0.6B-ft (k=5)</td><td>18.6</td><td>42.5</td><td>80.5</td><td>267</td><td>3.8</td><td>25.9</td><td>71.3</td><td>247</td><td>2.0</td><td>14.3</td><td>30.8</td><td>280</td><td>2.9</td><td>20.1</td><td>51.0</td><td>264</td></tr><tr><td colspan="17">Ours: Latent Memory</td></tr><tr><td>Latent Memory (k=1)</td><td>12.3</td><td>30.4</td><td>34.8</td><td>36</td><td>1.8</td><td>20.6</td><td>27.6</td><td>32</td><td>2.3</td><td>11.8</td><td>8.4</td><td>36</td><td>2.1</td><td>16.2</td><td>18.0</td><td>34</td></tr><tr><td>Latent Memory (k=2)</td><td>14.8</td><td>34.8</td><td>62.2</td><td>46</td><td>2.0</td><td>22.8</td><td>49.4</td><td>42</td><td>2.2</td><td>12.8</td><td>14.9</td><td>46</td><td>2.1</td><td>17.8</td><td>32.2</td><td>44</td></tr><tr><td>Latent Memory (k=5)</td><td>17.5</td><td>37.8</td><td>86.6</td><td>76</td><td>2.9</td><td>24.1</td><td>75.9</td><td>72</td><td>3.6</td><td>14.7</td><td>28.8</td><td>76</td><td>3.2</td><td>19.4</td><td>52.3</td><td>74</td></tr></table>

![](images/4ac6eca9aad65624a4b84ab9611f678327013d2bbcc008c6ac318da5386b4b40.jpg)

<details>
<summary>line chart</summary>

| Avg. Input Tokens to Generator | LLM Lingua | BM25 Retrieval | Dense Retrieval | Qwen3-Emb-0.6B | Latent Memory (1B, Ours) |
| ------------------------------ | ---------- | -------------- | --------------- | -------------- | ------------------------ |
| 50                             | 5%         | k=1            | k=1             | k=2            | k=2                      |
| 100                            | 10%        | k=2            | k=2             | k=2            | k=5                      |
| 200                            | 20%        | k=5            | k=5             | k=5            | -                        |
| 1827                           | -          | -              | -               | -              | -                        |
</details>

(a) LLaMA-8B

![](images/ec5ef8f64cb4194fcf097e084f5b1f3dd6022683fd40bfabd2eeb33bb050c9a3.jpg)

<details>
<summary>line chart</summary>

| Model | Avg. Input Tokens to Generator | OOD Average F1 |
|-------|----------------------------------|----------------|
| xRAG | 40 | 16 |
| CLAra-16x | 40 | 18 |
| LM Lingua | 40 | 14 |
| BM25 Retrieval | 40 | 12 |
| Dense Retrieval | 40 | 14 |
| Owen3-Emb-0.68 | 40 | 16 |
| Owen3-Emb-0.68-ft | 40 | 18 |
| Latent Memory (1B, Ours) | 40 | 20 |
| xRAG (k=5) | 40 | 16 |
| CLAra-16x (k=5) | 40 | 18 |
| LM Lingua (k=5) | 40 | 14 |
| BM25 Retrieval (k=5) | 40 | 12 |
| Dense Retrieval (k=5) | 40 | 14 |
| Owen3-Emb-0.68 (k=5) | 40 | 16 |
| Owen3-Emb-0.68-ft (k=5) | 40 | 18 |
| Latent Memory (1B, Ours) (k=5) | 40 | 20 |
| xRAG (k=1) | 40 | 16 |
| CLAra-16x (k=1) | 40 | 18 |
| LM Lingua (k=1) | 40 | 14 |
| BM25 Retrieval (k=1) | 40 | 12 |
| Dense Retrieval (k=1) | 40 | 14 |
| Owen3-Emb-0.68 (k=1) | 40 | 16 |
| Owen3-Emb-0.68-ft (k=1) | 40 | 18 |
| Latent Memory (1B, Ours) (k=5) | 40 | 20 |
| xRAG (k=5) | 40 | 16 |
| CLAra-16x (k=5) | 40 | 18 |
| LM Lingua (k=5) | 40 | 14 |
| BM25 Retrieval (k=5) | 40 | 12 |
| Dense Retrieval (k=5) | 50 | 14 |
| Owen3-Emb-0.68 (k=5) | 50 | 16 |
| Owen3-Emb-0.68-ft (k=5) | 50 | 18 |
| Latent Memory (1B, Ours) (k=5) | 50 | 20 |
| xRAG (k=5) | 50 | 16 |
| CLAra-16x (k=5) | 50 | 18 |
| LM Lingua (k=5) | 50 | 14 |
| BM25 Retrieval (k=5) | 50 | 12 |
| Dense Retrieval (k=5) | 50 | 14 |
| Owen3-Emb-0.68 (k=5) | 50 | 16 |
| Owen3-Emb-0.68-ft (k=5) | 50 | 18 |
| Latent Memory (1B, Ours) (k=5) | 50 | 20 |
| XLRGAL (k=5) | 50 | 16 |
| CLAra-16x (k=5) | 50 | 18 |
| LM Lingua (k=5) | 50 | 14 |
| BM25 Retrieval (k=5) | 50 | 12 |
| Dense Retrieval (k=5) | 50 | 14 |
| Owen3-Emb-0..68 (k=5) | 50 | 16 |
| Owen3-Emb-0.68-ft (k=5) | 50 | 18 |
| Latent Memory (1B, Ours) (k=5) | 50 | 20 |
Full: [star] indicates a full dataset.
</details>

(b) Mistral-7B-Instruct  
Figure 3: Text-only QA trade-off curves on the out-of-domain average F1 over 2WikiMultihopQA and MuSiQue.

Metrics. For text-only QA, we report Exact Match (EM), Token F1, Recall@k, and average generator input tokens (#Tok). For WebQA, we report F1, answer accuracy (Acc), Recall@k, and #Tok under the same unified retrieval setting. Acc follows the official WebQA evaluation protocol [50]. In all tables, #Tok reports the task-relevant generator-side prompt budget (including the context length and the query length) and excludes fixed chat-template scaffolding; in multimodal settings, it still reflects the effective tokenized cost after visual expansion.

## 4.1 Text-only Setting

Table 1 presents metrics of the proposed Latent Memory, context-based, and RAG baselines under a fixed 8B LLaMA generator. Qwen3-Emb-0.6B means using the Qwen3-Emedding-0.6B [51] for retrieval, which is a size similar to our 1B compressor. One-token Latent Memory shows the strongest out-of-domain average Recall@k among the reported text-only retrieval methods. As shown in Figure 3, Latent Memory achieves competitive EM/F1 performance with much fewer tokens. At k=5, the 1B model uses only 71 tokens on the out-of-domain average, about a third versus 209 for BM25 and 208 for Dense. We discuss the time complexity in Appendix B.4. These results indicate the superiority of using the same latent representation space for both retrieval and generation evidence.

Table 3: Multimodal QA (WebQA) with unified per-sample retrieval and unified generation over both images and text facts using a frozen LLaVA-1.5-13B generator.

<table><tr><td colspan="13">Generation VLM (fixed): LLaVA-1.5-13B</td></tr><tr><td rowspan="2">Method</td><td colspan="4">WebQA-Image</td><td colspan="4">WebQA-Text</td><td colspan="4">Average</td></tr><tr><td>F1</td><td>Acc</td><td>R@k</td><td>#Tok</td><td>F1</td><td>Acc</td><td>R@k</td><td>#Tok</td><td>F1</td><td>Acc</td><td>R@k</td><td>#Tok</td></tr><tr><td>Full Context</td><td>0.0</td><td>0.0</td><td>-</td><td>11655</td><td>6.0</td><td>10.0</td><td>-</td><td>8371</td><td>3.0</td><td>5.0</td><td>-</td><td>10013</td></tr><tr><td colspan="13">RAG Baselines: Text-only Retrieval Baselines</td></tr><tr><td>BM25 Retrieval (k=1)</td><td>14.5</td><td>21.5</td><td>21.8</td><td>295</td><td>39.9</td><td>42.7</td><td>31.8</td><td>133</td><td>27.2</td><td>32.1</td><td>26.8</td><td>214</td></tr><tr><td>BM25 Retrieval (k=2)</td><td>19.2</td><td>23.4</td><td>28.3</td><td>476</td><td>43.9</td><td>48.4</td><td>51.0</td><td>234</td><td>31.6</td><td>35.9</td><td>39.6</td><td>355</td></tr><tr><td>BM25 Retrieval (k=5)</td><td>32.6</td><td>29.5</td><td>37.9</td><td>932</td><td>46.7</td><td>54.3</td><td>73.0</td><td>552</td><td>39.6</td><td>41.9</td><td>55.5</td><td>742</td></tr><tr><td>Dense Retrieval (k=1)</td><td>18.9</td><td>24.5</td><td>39.7</td><td>476</td><td>34.9</td><td>40.4</td><td>25.5</td><td>264</td><td>26.9</td><td>32.4</td><td>32.6</td><td>370</td></tr><tr><td>Dense Retrieval (k=2)</td><td>30.0</td><td>30.1</td><td>53.2</td><td>814</td><td>38.5</td><td>49.3</td><td>40.1</td><td>538</td><td>34.2</td><td>39.7</td><td>46.6</td><td>676</td></tr><tr><td>Dense Retrieval (k=5)</td><td>49.8</td><td>36.6</td><td>67.1</td><td>1645</td><td>37.6</td><td>59.5</td><td>62.0</td><td>1396</td><td>43.7</td><td>48.1</td><td>64.6</td><td>1520</td></tr><tr><td colspan="13">RAG Baselines: Multimodal Retrieval Baselines</td></tr><tr><td>Nemo-Emb-1B (k=1)</td><td>19.9</td><td>25.2</td><td>48.7</td><td>507</td><td>41.1</td><td>44.0</td><td>40.4</td><td>130</td><td>30.5</td><td>34.6</td><td>44.6</td><td>319</td></tr><tr><td>Nemo-Emb-1B (k=2)</td><td>32.9</td><td>30.6</td><td>64.8</td><td>892</td><td>46.8</td><td>50.9</td><td>66.6</td><td>233</td><td>39.9</td><td>40.8</td><td>65.7</td><td>563</td></tr><tr><td>Nemo-Emb-1B (k=5)</td><td>53.0</td><td>37.2</td><td>81.4</td><td>1885</td><td>48.6</td><td>57.9</td><td>87.1</td><td>629</td><td>50.8</td><td>47.6</td><td>84.3</td><td>1257</td></tr><tr><td>Qwen3-VL-Emb-8B (k=1)</td><td>15.1</td><td>22.0</td><td>24.1</td><td>284</td><td>40.8</td><td>43.6</td><td>40.7</td><td>131</td><td>28.0</td><td>32.8</td><td>32.4</td><td>208</td></tr><tr><td>Qwen3-VL-Emb-8B (k=2)</td><td>20.1</td><td>24.5</td><td>32.9</td><td>465</td><td>46.1</td><td>50.2</td><td>66.4</td><td>235</td><td>33.1</td><td>37.3</td><td>49.6</td><td>350</td></tr><tr><td>Qwen3-VL-Emb-8B (k=5)</td><td>34.1</td><td>30.3</td><td>49.7</td><td>957</td><td>47.8</td><td>57.2</td><td>87.2</td><td>612</td><td>41.0</td><td>43.8</td><td>68.5</td><td>784</td></tr><tr><td colspan="13">Ours: Generation with retrieving Latent Memory</td></tr><tr><td>Latent Memory (k=1)</td><td>32.0</td><td>28.7</td><td>56.6</td><td>42</td><td>28.8</td><td>30.8</td><td>24.0</td><td>44</td><td>30.4</td><td>29.7</td><td>40.3</td><td>43</td></tr><tr><td>Latent Memory (k=2)</td><td>56.5</td><td>39.5</td><td>74.7</td><td>52</td><td>30.0</td><td>32.8</td><td>41.1</td><td>54</td><td>43.2</td><td>36.2</td><td>57.9</td><td>53</td></tr><tr><td>Latent Memory (k=5)</td><td>69.4</td><td>44.2</td><td>91.2</td><td>82</td><td>30.7</td><td>34.3</td><td>70.5</td><td>84</td><td>50.0</td><td>39.2</td><td>80.8</td><td>83</td></tr></table>

Table 2 further evaluates the same text-only setting under a frozen Mistral-7B-Instruct generator, which allows direct comparison with pretrained Mistral-based latent-context baselines such as xRAG and CLaRa. In this setting, our Latent Memory still uses the LLaMA-3.2-1B encoder/compressor and decoder. As shown in Figure 3, Ours-1B at $k { = } 5$ reaches the strongest out-of-domain Recall@k and competitive F1 while using 74 generator tokens, compared with 264 tokens for Qwen3-Emb-0.6B fine-tuned on HotpotQA at $k { = } 5$ .

Importantly, the retrieval behavior is not only an in-domain effect: the same HotpotQA-trained Latent Memory is evaluated without additional tuning on out-of-domain 2WikiMultihopQA and MuSiQue. Thus, the comparison tests the transfer of the learned latent interface rather than dataset-specific memorization. In Appendix C.2, we generalize the Latent Memory with a compressor trained on HotpotQA to four more datasets, where Latent Memory can still demonstrate a strong performanceefficiency trade-off. This competitive performance may be partly explained by its stronger retrieval behavior. Latent Memory reaches an out-of-domain average Recall@k of 52.2 at $k = 5 ,$ .

## 4.2 Multimodal Setting

We report the Latent Memory and baselines on the WebQA benchmark, which requires unified retrieval and multi-hop reasoning over a multimodal candidate pool. For BM25 and Dense retrieval, we retrieve only the caption for the image evidence. Table 3 shows the image-grounded $_ { ( n = 2 , 5 1 1 ) }$ and text-grounded $_ { ( n = 2 , 4 5 5 ) }$ WebQA subsets separately. One-token Latent Memory is strongest on the image-grounded subset while using far fewer generator tokens than raw-evidence retrieval baselines: at $k { = } 5 ,$ it reaches 69.4 image F1 with only 82 tokens, compared with 53.0 for Nemo-Emb at

1885 tokens. On the full benchmark, Latent Memory gives a competitive average F1 with a much smaller token budget, while Nemo-Emb gives the best average F1/Acc at substantially higher cost.

![](images/3292726d741eecc3cb2bcd253c29d9a57e8aff38ad1618b1fd774dfd79d3714c.jpg)

<details>
<summary>line chart</summary>

| Avg. Input Tokens to Generator | BM25 Retrieval | Dense Retrieval | NemaRetriever-1B | Owen3-VL-Emb-IBB | Latent Memory (7B, Ours) |
| ------------------------------ | -------------- | --------------- | ---------------- | ----------------- | ------------------------ |
| 10^1                           | 28             | 28              | 28               | 28                | 30                       |
| 10^2                           | 30             | 30              | 30               | 30                | 45                       |
| 10^3                           | 40             | 40              | 40               | 40                | 50                       |
</details>

Figure 4: LLaVA-based multimodal WebQA trade-off curves across k∈{1, 2, 5}.

Similar to the text-only setting, this behavior can be attributed to the fact that the unified representation of Latent Memory leads to a better Recall@K (10%+ higher compared to Dense Retrieval) on textgrounded and especially image-grounded questions. Moreover, as another reason, Raw-image prompting may exceed the pretrained context window for the generator, leading to poor-quality output (full-context often outputs meaningless content or blanks). To intuitively reflect the generation process augmented with Latent Memory, we provide a case study in Section 5.

## 5 Discussion

Ablations and Analysis. Table 4 conducts an ablation study over the core reconstruction loss ${ \mathcal { L } } _ { \mathrm { R e c o n } } ,$ where the results are averaged over HotpotQA, 2WikiMultihopQA, and MuSiQue. Removing reconstruction lowers both answer quality and retrieval accuracy, with a larger drop in EM/F1 than in Recall@k. Removing negative evidence from reconstruction also hurts retrieval and generation, supporting the view that negative evidence helps anchor the unified latent representation space. The full per-dataset breakdown and more ablation variants are reported in Appendix C.3 to C.5.

Table 4: Ablation on reconstruction. Colored subscripts indicate the gap to the default model.

<table><tr><td rowspan="2">k</td><td colspan="3">Original Latent Memory</td><td colspan="3">w/o Reconstruction Loss  $\mathcal{L}_{\text{Recon}}$ </td><td colspan="3">w/o  $\mathcal{L}_{\text{Recon}}$  on Negative Evidence  $\mathcal{M}^{-}$ </td></tr><tr><td>EM</td><td>F1</td><td>R@k</td><td>EM</td><td>F1</td><td>R@k</td><td>EM</td><td>F1</td><td>R@k</td></tr><tr><td>k=1</td><td>17.7</td><td>27.7</td><td>23.9</td><td>16.5-1.1</td><td>27.2-0.5</td><td>23.8-0.1</td><td>16.3-1.3</td><td>28.1+0.3</td><td>23.2-0.6</td></tr><tr><td>k=2</td><td>20.1</td><td>31.9</td><td>42.6</td><td>19.9-0.2</td><td>31.0-0.9</td><td>41.8-0.7</td><td>18.4-1.7</td><td>32.0+0.2</td><td>39.6-3.0</td></tr><tr><td>k=5</td><td>22.6</td><td>34.9</td><td>63.8</td><td>20.9-1.7</td><td>32.7-2.2</td><td>62.9-0.9</td><td>20.8-1.9</td><td>33.8-1.2</td><td>62.3-1.5</td></tr></table>

Better Latent Memory Capability with more Token Budget. One-token Latent Memory already gives a strong efficiency–quality trade-off, and allocating more latent tokens per evidence item can improve the quality. As summarized in Table 5, on the out-of-domain average over 2WikiMultihopQA and MuSiQue, upgrading to 8-token Latent Memory improves EM/F1 enough to surpass the strongest text-only RAG baseline (i.e., Qwen-3-Emb-0.6B) at each k, while still using fewer generator tokens. The gain mainly comes from stronger generation rather than retrieval, since Recall@k changes only modestly. Full in-domain and out-of-domain results are in Appendix C.1.

Table 5: Token-count ablation summary. We show the average Out-of-domain (2WikiMultihopQA and MuSiQue results). RAG\* denotes the strongest RAG baseline at the same k.

<table><tr><td rowspan="2">k</td><td colspan="4">RAG*</td><td colspan="4">1-token Latent Memory</td><td colspan="4">8-token Latent Memory</td><td colspan="4"> $\Delta (8-token - 1-token)$ </td></tr><tr><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td></tr><tr><td>k=1</td><td>13.1</td><td>24.6</td><td>21.4</td><td>70</td><td>12.8-0.3</td><td>21.9-2.7</td><td>18.6</td><td>35-35</td><td>14.4+1.3</td><td>24.7+0.1</td><td>19.9</td><td>42-28</td><td>+1.6</td><td>+2.8</td><td>+1.3</td><td>+7</td></tr><tr><td>k=2</td><td>13.7</td><td>27.0</td><td>33.1</td><td>107</td><td>14.4+0.7</td><td>25.2-1.8</td><td>32.5</td><td>44-63</td><td>17.3+3.6</td><td>29.0+2.0</td><td>34.5</td><td>58-49</td><td>+2.9</td><td>+3.8</td><td>+2.0</td><td>+14</td></tr><tr><td>k=5</td><td>16.4</td><td>31.7</td><td>49.6</td><td>219</td><td>16.5+0.1</td><td>28.0-3.7</td><td>52.2</td><td>71-148</td><td>19.9+3.5</td><td>32.5+0.9</td><td>53.7</td><td>106-113</td><td>+3.4</td><td>+4.6</td><td>+1.5</td><td>+35</td></tr></table>

A Representative Case Study. Figure 5 illustrates how Latent Memory behaves on an imagegrounded WebQA example. The retrieved latent evidence supports the correct answer and preserves the counting information required by the question. The optional reconstruction further shows that the latent token retains interpretable semantic content for both text and image evidence, rather than acting only as an opaque retrieval identifier.

Limitation on Current Modality Coverage and Future Directions. The current design assumes that evidence can be decomposed into atomic text or image units, and each unit can be compressed and retrieved independently. This assumption is reasonable for WebQA-style mixed evidence, where answers are often grounded in a small number of facts or images. However, it becomes limiting when the input meaning depends on the global structure. Complex tables require row-column relations and layout information; long videos require temporal ordering; and document pages may require spatial relations between captions, figures, and surrounding text. Compressing such inputs into isolated latent tokens may preserve local semantics but lose these structural dependencies. A natural next step is therefore to augment Latent Memory with structural axes such as position, layout, and time, so that retrieval and generation can operate over both local evidence semantics and global organization.

![](images/41098f80fd514a2162dce6a580f42790bc7af1990a84d969e5fc7f2412d5763a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Image: Window shopping: London's Burlington Arcade, which opened in 1819, positioned itself as an elegant and exclusive venue."] --> B["Text: Window shopping: London's Burlington Arcade, which opened in 1819, positioned itself as an elegant and exclusive venue."]
  B --> C["Text: Burlington Arcade: The present tenants include a range of clothing, footwear and accessory shops, art and antique dealers, and the jewellers and dea..."]
  C --> D["Text: ......"]
  D --> E["Retrievable Latent Memory"]
  E --> F["Input LLM/VLM θ with Trainable LoRA for Compression"]
  F --> G["Question: &quot;How many display windows can be found on the front of the Heming store at London's Burlington Arcade?&quot;"]
  G --> H["Latent context 1: z₁"]
  H --> I["① Latent Memory Construction"]
  I --> J["Output Frozen Larger-Size Generation LLM/VLM φ"]
  J --> K["Three"]
  K --> L["③ Generation with Latent Tokens"]
  L --> M["Reconstructed Context (Positive) Image: LLM π with Trainable LoRA/MLP for Decoding"]
  M --> N["Reconstructed Context (Negative) Image: Craven.. (CE = 0.8)"]
  N --> O["Text: View of... (CE = 0.7)"]
  O --> P["Text: Craven.. (CE = 0.8)"]
```
</details>

Figure 5: LLaVA-1.5-based Latent Memory on an image-grounded WebQA question. The figure consists of four parts. ① Compressing multimodal evidence forms a unified Latent Memory. ② The retrieval process aligns the query embedding with the latent token with the positive image, while irrelevant candidates are pushed away in the unified latent space. ③ The Latent Memory grounded QA preserves the counting ability. ④ The optional reconstruction for both image and text evidence.

## 6 Conclusion

We introduced Latent Memory, a novel memory paradigm that compiles each evidence item into one latent token, retrieves these latent memories by compressing the query, and feeds them directly to a frozen generator for QA. On text QA, it achieves competitive EM/F1 performance compared to RAG baselines with only 3× fewer tokens. On multimodal WebQA, it is especially effective for image-grounded questions and delivers the strongest average-F1 trade-off while sharply reducing generator cost by up to 10×. Latent Memory improves the Recall@k by employing a unified latent representation and prevents the context from exceeding the generator’s context window length.

Future Work Overall, Latent Memory provides an efficient alternative to the current token-level memory paradigm. It is promising for use in scenarios that require fast response and low storage pressure, such as edge devices and other resource-intensive scenarios. In the future, as discussed in Section 5, we will extend the Latent Memory to more modalities, including complex tables and complex videos. This paper focuses on external evidence, so we do not consider Agentic Memory [10, 53, 54], which are generated by models themselves. Future works also include extending the Latent Memory to resource-contained agents scenarios for better Agentic Memory comprehension.

## References

[1] Murong Yue. A survey of large language model agents for question answering. arXiv preprint arXiv:2503.19213, 2025.  
[2] Yu-Hsuan Lin, Qian-Hui Chen, Yi-Jie Cheng, Jia-Ren Zhang, Yi-Hung Liu, Liang-Yu Hsia, and Yun-Nung Chen. Llm inference enhanced by external knowledge: A survey. arXiv preprint arXiv:2505.24377, 2025.  
[3] Alphaeus Dmonte, Roland Oruche, Marcos Zampieri, Prasad Calyam, and Isabelle Augenstein. Claim verification in the age of large language models: A survey. arXiv preprint arXiv:2408.14317, 2024.  
[4] Zhi Zheng and Wee Sun Lee. Reasoning-cv: Fine-tuning powerful reasoning llms for knowledge-assisted claim verification. arXiv preprint arXiv:2505.12348, 2025.  
[5] Yaxiong Wu, Sheng Liang, Chen Zhang, Yichao Wang, Yongyue Zhang, Huifeng Guo, Ruiming Tang, and Yong Liu. From human memory to ai memory: A survey on memory mechanisms in the era of llms. arXiv preprint arXiv:2504.15965, 2025.  
[6] Shangyu Wu, Ying Xiong, Yufei Cui, Haolun Wu, Can Chen, Ye Yuan, Lianming Huang, Xue Liu, Tei-Wei Kuo, Nan Guan, et al. Retrieval-augmented generation for natural language processing: A survey. arXiv preprint arXiv:2407.13193, 2024.  
[7] Xavier Daull, Patrice Bellot, Emmanuel Bruno, Vincent Martin, and Elisabeth Murisasco. Complex qa and language models hybrid architectures, survey. arXiv preprint arXiv:2302.09051, 2023.  
[8] Xiaofei Dong, Xueqiang Zhang, Weixin Bu, Dan Zhang, and Feng Cao. A survey of llm-based agents: Theories, technologies, applications and suggestions. In 2024 3rd International Conference on Artificial Intelligence, Internet of Things and Cloud Computing Technology (AIoTC), pages 407–413. IEEE, 2024.  
[9] Zhi Zheng, Zhuoliang Xie, Zhenkun Wang, and Bryan Hooi. Monte carlo tree search for comprehensive exploration in llm-based automatic heuristic design. arXiv preprint arXiv:2501.08603, 2025.  
[10] Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.  
[11] Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. Swe-bench: Can language models resolve real-world github issues? In International Conference on Learning Representations, volume 2024, pages 54107–54157, 2024.  
[12] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.  
[13] Gemma Team. Gemma 3 technical report. ArXiv, abs/2503.19786, 2025.  
[14] Taehwan Park, Geonho Lee, and Min-Soo Kim. Mobilerag: A fast, memory-efficient, and energy-efficient method for on-device rag. arXiv preprint arXiv:2507.01079, 2025.  
[15] Onur Mutlu, Ataberk Olgun, and ˙Ismail Emir Yüksel. Memory-centric computing: solving computing’s memory problem. In 2025 IEEE International Memory Workshop (IMW), pages 1–4. IEEE, 2025.  
[16] Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024.  
[17] Jiaheng Liu, Dawei Zhu, Zhiqi Bai, Yancheng He, Huanxuan Liao, Haoran Que, Zekun Wang, Chenchen Zhang, Ge Zhang, Jiebin Zhang, et al. A comprehensive survey on long context language modeling. arXiv preprint arXiv:2503.17407, 2025.  
[18] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledgeintensive nlp tasks. Advances in neural information processing systems, 33:9459–9474, 2020.  
[19] Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang. Retrieval augmented language model pre-training. In International conference on machine learning, pages 3929–3938. PMLR, 2020.  
[20] Gautier Izacard, Patrick Lewis, Maria Lomeli, Lucas Hosseini, Fabio Petroni, Timo Schick, Jane Dwivedi-Yu, Armand Joulin, Sebastian Riedel, and Edouard Grave. Atlas: Few-shot learning with retrieval augmented language models. Journal of Machine Learning Research, 24(251):1–43, 2023.  
[21] Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 conference on empirical methods in natural language processing (EMNLP), pages 6769–6781, 2020.  
[22] Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond, volume 4. Now Publishers Inc, 2009.  
[23] Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, and Chao Huang. Lightrag: Simple and fast retrievalaugmented generation. 2024.  
[24] Muhammad Arslan, Hussam Ghanem, Saba Munawar, and Christophe Cruz. A survey on rag with llms. Procedia computer science, 246:3781–3790, 2024.  
[25] Mingxin Li, Yanzhao Zhang, Dingkun Long, Keqin Chen, Sibo Song, Shuai Bai, Zhibo Yang, Pengjun Xie, An Yang, Dayiheng Liu, et al. Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv preprint arXiv:2601.04720, 2026.  
[26] Haonan Jiang, Yuji Wang, Yongjie Zhu, Xin Lu, Wenyu Qin, Meng Wang, Pengfei Wan, and Yansong Tang. Embed-rl: Reinforcement learning for reasoning-driven multimodal embeddings. arXiv preprint arXiv:2602.13823, 2026.  
[27] Chenwei He, Xiangzhao Hao, Tianyu Yang, Yuxiang Ma, Yuheng Jia, Lingxiang Wu, Chaoyang Zhao, Haiyun Guo, and Jinqiao Wang. Plume: Latent reasoning based universal multimodal embedding. arXiv preprint arXiv:2604.02073, 2026.  
[28] Huiqiang Jiang, Qianhui Wu, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. Llmlingua: Compressing prompts for accelerated inference of large language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pages 13358–13376, 2023.  
[29] Alexis Chevalier, Alexander Wettig, Anirudh Ajith, and Danqi Chen. Adapting language models to compress contexts. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 3829–3846, 2023.  
[30] Tao Ge, Jing Hu, Lei Wang, Xun Wang, Si-Qing Chen, and Furu Wei. In-context autoencoder for context compression in a large language model. arXiv preprint arXiv:2307.06945, 2023.  
[31] Zeju Li, Yizhou Zhou, and Qiang Xu. Latent context compilation: Distilling long context into compact portable memory. arXiv preprint arXiv:2602.21221, 2026.  
[32] Xin Cheng, Xun Wang, Xingxing Zhang, Tao Ge, Si-Qing Chen, Furu Wei, Huishuai Zhang, and Dongyan Zhao. xrag: Extreme context compression for retrieval-augmented generation with one token. Advances in Neural Information Processing Systems, 37:109487–109516, 2024.  
[33] Jie He, Richard He Bai, Sinead Williamson, Jeff Z Pan, Navdeep Jaitly, and Yizhe Zhang. Clara: Bridging retrieval and generation with continuous latent reasoning. arXiv preprint arXiv:2511.18659, 2025.  
[34] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.  
[35] Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, and Yulan He. Codi: Compressing chain-of-thought into continuous space via self-distillation. arXiv preprint arXiv:2502.21074, 2025.  
[36] Xilin Wei, Xiaoran Liu, Yuhang Zang, Xiaoyi Dong, Yuhang Cao, Jiaqi Wang, Xipeng Qiu, and Dahua Lin. Sim-cot: Supervised implicit chain-of-thought. arXiv preprint arXiv:2509.20317, 2025.  
[37] Zhi Zheng and Wee Sun Lee. Beyond imitation: Reinforcement learning for active latent planning. arXiv preprint arXiv:2601.21598, 2026.  
[38] Muxin Fu, Xiangyuan Xue, Yafu Li, Zefeng He, Siyuan Huang, Xiaoye Qu, Yu Cheng, and Yang Yang. Latentmem: Customizing latent memory for multi-agent systems. arXiv preprint arXiv:2602.03036, 2026.  
[39] Guibin Zhang, Muxin Fu, and Shuicheng Yan. Memgen: Weaving generative latent memory for selfevolving agents. arXiv preprint arXiv:2509.24704, 2025.  
[40] Xinlei Yu, Chengming Xu, Guibin Zhang, Zhangquan Chen, Yudong Zhang, Yongbo He, Peng-Tao Jiang, Jiangning Zhang, Xiaobin Hu, and Shuicheng Yan. Vismem: Latent vision memory unlocks potential of vision-language models. arXiv preprint arXiv:2511.11007, 2025.  
[41] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a joint-embedding predictive architecture. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 15619–15629, 2023.  
[42] Delong Chen, Mustafa Shukor, Theo Moutakanni, Willy Chung, Jade Yu, Tejaswi Kasarla, Yejin Bang, Allen Bolourchi, Yann LeCun, and Pascale Fung. Vl-jepa: Joint embedding predictive architecture for vision-language. arXiv preprint arXiv:2512.10942, 2025.  
[43] Heejeong Nam, Quentin Le Lidec, Lucas Maes, Yann LeCun, and Randall Balestriero. Causal-jepa: Learning world models through object-level latent interventions. arXiv preprint arXiv:2602.11389, 2026.  
[44] Jingwen Sun, Wenyao Zhang, Zekun Qi, Shaojie Ren, Zezhi Liu, Hanxin Zhu, Guangzhong Sun, Xin Jin, and Zhibo Chen. Vla-jepa: Enhancing vision-language-action model with latent world model. arXiv preprint arXiv:2602.10098, 2026.  
[45] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.  
[46] Aditya Ramesh, Prafulla Dhariwal, Alex Nichol, Casey Chu, and Mark Chen. Hierarchical text-conditional image generation with clip latents. arXiv preprint arXiv:2204.06125, 1(2):3, 2022.  
[47] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.  
[48] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.  
[49] Yihang Jiang, Xiaoyang Li, Guangxu Zhu, Hang Li, Jing Deng, Kaifeng Han, Chao Shen, Qingjiang Shi, and Rui Zhang. 6g non-terrestrial networks enabled low-altitude economy: Opportunities and challenges. arXiv preprint arXiv:2311.09047, 2023.  
[50] Yingshan Chang, Mridu Narang, Hisami Suzuki, Guihong Cao, Jianfeng Gao, and Yonatan Bisk. Webqa: Multihop and multimodal qa. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16495–16504, 2022.  
[51] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025.  
[52] Mengyao Xu, Gabriel Moreira, Ronay Ak, Radek Osmulski, Yauhen Babakhin, Zhiding Yu, Benedikt Schifferer, and Even Oldridge. Llama nemoretriever colembed: Top-performing text-image retrieval model. arXiv preprint arXiv:2507.05513, 2025.  
[53] Yu Wang and Xi Chen. Mirix: Multi-agent memory system for llm-based agents. arXiv preprint arXiv:2507.07957, 2025.  
[54] Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.  
[55] Shi Yu, Chaoyue Tang, Bokai Xu, Junbo Cui, Junhao Ran, Yukun Yan, Zhenghao Liu, Shuo Wang, Xu Han, Zhiyuan Liu, et al. Visrag: Vision-based retrieval-augmented generation on multi-modality documents. arXiv preprint arXiv:2410.10594, 2024.  
[56] Maxime Louis, Hervé Déjean, and Stéphane Clinchant. Pisco: Pretty simple compression for retrievalaugmented generation. In Findings of the Association for Computational Linguistics: ACL 2025, pages 15506–15521, 2025.  
[57] Zhibin Lan, Liqiang Niu, Fandong Meng, Jie Zhou, and Jinsong Su. Ume-r1: Exploring reasoning-driven generative multimodal embeddings. arXiv preprint arXiv:2511.00405, 2026.  
[58] Rui-Jie Zhu, Tianhao Peng, Tianhao Cheng, Xingwei Qu, Jinfa Huang, Dawei Zhu, Hao Wang, Kaiwen Xue, Xuanliang Zhang, Yong Shan, et al. A survey on latent reasoning. arXiv preprint arXiv:2507.06203, 2025.  
[59] Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian Wang, Wenjie Li, and Xiaoyu Shen. Reasoning beyond language: A comprehensive survey on latent chain-of-thought reasoning. arXiv preprint arXiv:2505.16782, 2025.  
[60] Yue Liu, Jiaying Wu, Yufei He, Ruihan Gong, Jun Xia, Liang Li, Hongcheng Gao, Hongyu Chen, Baolong Bi, Jiaheng Zhang, et al. Efficient inference for large reasoning models: A survey. arXiv preprint arXiv:2503.23077, 2025.  
[61] Xinlei Yu, Zhangquan Chen, Yongbo He, Tianyu Fu, Cheng Yang, Chengming Xu, Yue Ma, Xiaobin Hu, Zhe Cao, Jie Xu, et al. The latent space: Foundation, evolution, mechanism, ability, and outlook. arXiv preprint arXiv:2604.02029, 2026.  
[62] Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, and Yisen Wang. Monet: Reasoning in latent visual space beyond images and language. arXiv preprint arXiv:2511.21395, 2025.  
[63] Zhen Zhang, Xuehai He, Weixiang Yan, Ao Shen, Chenyang Zhao, Shuohang Wang, Yelong Shen, and Xin Eric Wang. Soft thinking: Unlocking the reasoning potential of llms in continuous concept space. arXiv preprint arXiv:2505.15778, 2025.  
[64] Natasha Butt, Ariel Kwiatkowski, Ismail Labiad, Julia Kempe, and Yann Ollivier. Soft tokens, hard truths. arXiv preprint arXiv:2509.19170, 2025.  
[65] Chünhung Wu, Jinliang Lu, Zixuan Ren, Gangqiang Hu, Zhi Wu, Dai Dai, and Hua Wu. Llms are single-threaded reasoners: Demystifying the working mechanism of soft thinking. arXiv preprint arXiv:2508.03440, 2025.  
[66] Zhi Zheng and Wee Sun Lee. Soft-grpo: Surpassing discrete-token llm reinforcement learning via gumbel-reparameterized soft-thinking policy optimization. arXiv preprint arXiv:2511.06411, 2025.  
[67] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380, 2018.  
[68] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop qa dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, 2020.  
[69] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. Musique: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 10:539–554, 2022.  
[70] Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, et al. Natural questions: a benchmark for question answering research. Transactions of the Association for Computational Linguistics, 7:453–466, 2019.  
[71] Mandar Joshi, Eunsol Choi, Daniel S Weld, and Luke Zettlemoyer. Triviaqa: A large scale distantly supervised challenge dataset for reading comprehension. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 1601–1611, 2017.  
[72] Pradeep Dasigi, Kyle Lo, Iz Beltagy, Arman Cohan, Noah A Smith, and Matt Gardner. A dataset of information-seeking questions and answers anchored in research papers. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4599–4610, 2021.  
[73] Ryo Kamoi, Tanya Goyal, Juan Diego Rodriguez, and Greg Durrett. Wice: Real-world entailment for claims in wikipedia. arXiv preprint arXiv:2303.01432, 2023.  
[74] Ziyan Jiang, Xueguang Ma, and Wenhu Chen. Longrag: Enhancing retrieval-augmented generation with long-context llms. arXiv preprint arXiv:2406.15319, 2024.

## Appendix Contents

## 1. Related Work 16

(a) RAG and Embedding Retrieval . 16  
(b) Evidence Compression for Generation  
(c) Retrieval-Compression Interaction  
(d) Latent-Space Modeling for LLMs . .

## 2. Implementation Details 18

(a) Prompt Templates . . . . 18  
(b) Detailed Pipeline . . . . 19  
(c) Hyperparameter Settings . . . . . . . . 20  
(d) Time Complexity Analysis . . . . 21  
(e) Space Complexity Analysis . . . . 22

## 3. Additional Experiments and Discussion 23

(a) Token Count Ablation . 24  
(b) Generalization Ability on More Domains . . . 26  
(c) Ablation on Core Settings . . 27  
(d) Ablation on Stronger Text Compressors . 28  
(e) Direct Transfer to Similar Generator . 29  
(f) Multimodal Results with Gemma . . . . 30  
(g) Latent Tokens as Retrievers . 30

## 4. Case Study . . 32

(a) Reconstruction Quality of Latent Tokens . 32  
(b) Text-only Case Studies . 33  
(c) More Multimodal QA Case Studies . . . . 34

## 5. Baselines, Datasets, and Licenses . 35

## A Related Work

Table 6: Capability comparison with representative related work. Multimodal (Text & Image) discusses whether the method can adapt to multimodal documents or evidence. Efficient generation means whether this method aims at reducing the generator token consumption. Unified generation & retrieval means that the same compressed representation is used both for retrieval and as the object consumed by the generator (✓ making it unnecessary to work separately on compression and retrieval, xRAG (△) using the same retrieved embeddings as generator input at inference, but relies on a separately trained bridge rather than unified retrieval-generation training). No need to fine-tune generator LLM/VLM means the memory item can be directly fed into the generator without requiring it to comprehend. (✓ making it easier to deploy and avoid catastrophic forgetting)

<table><tr><td>Representative work</td><td>Multimodal(Text &amp; Image)</td><td>Efficientgeneration</td><td>Unified generation&amp; retrieval</td><td>No need to fine-tunegenerator LLM/VLM</td></tr><tr><td colspan="5">(1) RAG baselines: Raw-evidence RAG</td></tr><tr><td>BM25 / Dense RAG /Qwen3-Embedding [22, 21, 51]</td><td>✕</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>VisRAG / Qwen3-VL-Embedding /Nemo Retriever [55, 25, 52]</td><td>√</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td colspan="5">(2) Compression-based baselines: Evidence compression for generation</td></tr><tr><td>LLMLingua / ICAE [28, 30]</td><td>✕</td><td>√</td><td>✕</td><td>√</td></tr><tr><td>AutoCompressor [29]</td><td>✕</td><td>√</td><td>✕</td><td>✕</td></tr><tr><td>LCC / PISCO [31, 56]</td><td>✕</td><td>√</td><td>✕</td><td>✕</td></tr><tr><td colspan="5">(1) &amp; (2) Interaction baselines: Embedding-based retrieval then latent-context generation</td></tr><tr><td>xRAG [32]</td><td>✕</td><td>√</td><td>△</td><td>√</td></tr><tr><td>CLaRa [33]</td><td>✕</td><td>√</td><td>√</td><td>✕</td></tr><tr><td colspan="5">Ours: Compiling Latent Memory for retrieval &amp; generation</td></tr><tr><td>Latent Memory</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

In this section, I will discuss works that are related to the proposed Latent Memory paradigm, as well as a broad range of works that have a similar latent-representation idea. Table 6 presents a comprehensive collection of existing related works.

## A.1 RAG and Embedding Retrieval

BM25-based sparse RAG. Retrieval-augmented generation selects a small subset of evidence and then lets a pretrained generator answer from the retrieved content [18, 20]. BM25 is the classical sparse retrieval baseline, relying on lexical matching and term statistics rather than learned semantic representations [22]. It is simple and directly compatible with frozen LLMs, but the generator still consumes raw retrieved text, so the retrieval step does not reduce the per-evidence token cost of generation.

Text-only Dense retrieval and embedding models. Dense retrieval learns a neural text-embedding space for query-evidence matching, as in DPR-style retrieval [21]. As the most recent embedding model for dense retrieval, Qwen3-Embedding strengthens this retrieval front end with a modern text embedding and reranking model [51]. These methods improve semantic retrieval over textual evidence, but the representation is still used primarily as a retrieval key; after retrieval, the generator receives raw text rather than the embedding itself. Moreover, these works cannot generalize well to multimodal settings, especially on images with details that cannot be accurately captioned.

VisRAG and multimodal embedding retrieval. Multimodal RAG extends retrieval from text-only corpora to mixed text-image evidence. VisRAG is representative of raw-evidence multimodal RAG: it retrieves relevant visual documents but still hands visual inputs to the generator [55]. As pretrained embedding model for multimodal RAG, Qwen3-VL-Embedding provides a unified vision-language embedding and reranking framework for text-image retrieval [25]. Nemo Retriever is another strong text-image retrieval model used as a retrieval front end [52].

Recent reasoning-aware multimodal embedding methods further refine what the retrieval vector should encode: Embed-RL optimizes multimodal embeddings through reinforcement learning signals [26], UME-R1 explores reasoning-driven generative multimodal embeddings [57], and PLUME uses latent reasoning for universal multimodal embedding [27]. As summarized in Table 6, these methods support multimodal retrieval and can be used with pretrained VLM generators, but they remain raw-evidence RAG: retrieved images or text are still passed to the generator in their native input form. Thus, a retrieved image can still expand into many visual tokens, keeping generation expensive even when retrieval is accurate.

## A.2 Evidence Compression for Generation

Discrete prompt compression. Evidence-compression methods reduce the context consumed by the generator after the evidence has already been chosen. LLMLingua prunes or rewrites discrete prompt tokens, preserving compatibility with pretrained LLMs while shortening the input [28]. This improves generator-side efficiency, but it does not build a retrievable memory: the shortened context is produced for the current prompt rather than stored as a corpus item for future retrieval.

Latent or compact context compression. ICAE compresses context into learned memory-like latent states for language modeling [30]. AutoCompressor trains models to summarize previous context into compact summary vectors that can condition later generation [29]. LCC studies latent context compression for reducing long-context generation cost [31]. PISCO similarly targets compact softcontext representations for efficient generation [56]. These methods correspond to the second block of Table 6: they focus on efficient generation, but they generally do not retrieve over the compressed evidence representation. So, because there is no retrieval system, these methods can exceed the generation context window when the irrelevant context is very long, resulting in meaningless output. Several methods also require the generator side to be trained or adapted to understand the compressed tokens, so compression and retrieval remain separate problems.

## A.3 Retrieval-Compression Interaction

These are also works that combine the idea of retrieval and compression.

xRAG. xRAG first presents the idea of using a unified representation for generation & retrieval. However, as we mark it with △ for unified generation and retrieval in Table 6. xRAG relies on training a bridge based on pre-trained retrieval embeddings; noting that there are representations being important for generation but meaningless for retrieval (e.g., some details), the two-stage process makes it unable to find the optimal and unified generation & retrieval representation.

CLaRa. The most recent work, CLaRa, is closer to unified latent-context retrieval and generation because its latent context can be retrieved and then used for generation [33]. Both CLaRa and xRAG have the same limitation: it is text-only. Moreover, CLaRa requires a large amount of pre-training effort, and the latent representation in CLaRa is processed to fine-tune LLM/VLM Latent Memory, which might lead to catastrophic forgetting.

## Latent Memory vesus xRAG & CLaRa.

Compared to the two methods mentioned above, the Latent Memory paradigm extends the text-only problem to multimodal scenarios. The success of Latent Memory in multimodal scenarios not only improves generation token efficiency but also reduces storage pressure, which cannot be achieved by pure text-only Latent Memory. We provide a more detailed description in Appendix B.5, showing that saving text-based data does not significantly increase storage pressure compared to Latent Memory; on the contrary, high-dimensional vectors are often more difficult to store. However, for images, Latent Memory in LLaVA scenarios (4096-dimensional bf16 latent token) is more efficient than an uncompressed RGB image once the image is larger than roughly 53 × 53 pixels.

## A.4 Latent-Space Modeling for LLMs

Latent reasoning and hidden-state computation. Latent Memory is inspired by a broader direction of latent-space representation [58, 59, 60, 61]. Previous work shows that continuous hidden states can carry useful information in a wide collection of applications, including latent chain-of-thought math reasoning [34, 35, 36, 37, 62, 40], soft-thinking reasoning [63, 64, 65, 66], and agentic communication in multi-turn / multi-agent [38, 39]. It is worth noting that although some works also use a similar name, Memory [38, 39, 40], we are the first to use latent tokens to save storage and generation token consumption in multimodal contextual QA. Besides, some studies seek a unified latent representation space for multimodal understanding and generation [42, 44, 43]. These studies motivate the idea that hidden-state vectors have the ability to carry unified representations for contextual memory.

## B Implementation Details

## B.1 Prompt Templates

Teacher and student use the same system-user-assistant scaffold. Across text-only and LLaVA pipelines, the instruction is: “You are a helpful assistant. Answer the question concisely (a few words or a short phrase) based on the provided context.” The only difference is the evidence representation: the teacher sees raw evidence, while the student receives one latent token for each retrieved evidence item in the same retrieved order.

Text-only teacher prompt  
```ini
[system] You are a helpful assistant. Answer the question concisely (a few words or a short phrase) based on the provided context.
[user]
Context 1: <retrieved fact 1>
Context 2: <retrieved fact 2>
...
Context k: <retrieved fact k>
Question: <question>
[assistant]
Answer:
```

Text-only student prompt  
```ini
[system] You are a helpful assistant. Answer the question concisely (a few words or a short phrase) based on the provided context.
[user]
Latent context 1: [LATENT]
Latent context 2: [LATENT]
...
Latent context k: [LATENT]
Question: <question>
[assistant]
Answer:
```

Multimodal teacher prompt  
```ini
[system] You are a helpful assistant. Answer the question concisely (a few words or a short phrase) based on the provided context.
[user]
Context 1: <retrieved text fact 1>
Context 2: <image>
Title: <retrieved image title/caption 2>
...
Question: <question>
[assistant]
Answer:
```

Multimodal student prompt  
```ini
[system] You are a helpful assistant. Answer the question concisely (a few words or a short phrase) based on the provided context.
[user]
Latent context 1: [LATENT]
Latent context 2: [LATENT]
...
Latent context k: [LATENT]
Question: <question>
[assistant]
Answer:
```

## B.2 Detailed Pipeline

Retrieval projection MLP. The retrieval head maps the compressor hidden state into a 512- dimensional retrieval space. In all reported text-only and multimodal runs, it is implemented as

$$
\text { retrieval\_proj } (z) = \ell_ {2} \left(\text { LayerNorm } (\text { Linear } (z))\right),
$$

where the linear layer maps $d _ { \theta }  d _ { r }$ and $d _ { r } = 5 1 2$ in all reported runs. The $\ell _ { 2 }$ normalization is applied before FAISS storage and query-time similarity search, so inner product retrieval is equivalent to cosine similarity.

Generator projection MLP. The generator-side projector converts a retrieved Latent Memory vector into the frozen generator hidden dimension. It is a two-layer MLP with LayerNorm:

$$
\text { cross\_proj } (z) = \text { LayerNorm } (W _ {2} \text { GELU } (W _ {1} z)).
$$

For the text-only setting, it maps the Llama-3.2-1B hidden size $d _ { \theta } = 2 0 4 8$ to the Meta-Llama-3-8B generator hidden size $d _ { \phi } = 4 0 9 6$ , with intermediate dimension $( 2 0 4 8 + 4 0 9 6 ) / 2 = 3 0 7 2$ . For multimodal LLaVA, it maps the LLaVA-7B compressor hidden size $d _ { \theta } = 4 0 9 6$ to the $\mathrm { L L a V A – 1 } 3 \mathrm { B }$ generator hidden size $d _ { \phi } = 5 1 2 0$ , with intermediate dimension $( 4 0 9 6 + 5 1 2 0 ) / 2 = 4 6 0 8 .$ .

Decoder and image-reconstruction MLPs. For textual reconstruction, the projected latent token is inserted into the decoding prompt from Appendix B.1, and the decoder π is trained to autoregressively recover the original text evidence or image caption. The decoder projector maps compressor hidden states into the lightweight decoder hidden space with Linear $( { \bar { d } } _ { \theta } , { \bar { d } } _ { \pi } ) $ LayerNorm, where $d _ { \pi }$ denotes the hidden dimension of the decoder π. For image evidence, we do not reconstruct raw pixels. Instead, the image embedding reconstruction head predicts the frozen CLIP CLS hidden state. This MLP is

$$
\text { img\_embed\_decode\_proj } (z) = W _ {2} \text {   LayerNorm(GELU } (W _ {1} z)),
$$

where $W _ { 1 }$ maps $d _ { \theta }$ to $( d _ { \theta } + d _ { v } ) / 2$ and $W _ { 2 }$ maps to $d _ { v } = 1 0 2 4$ for the CLIP target used in our runs. For $\mathrm { L L a V A } { - } 7 \mathrm { \bar { B } }$ , this midpoint is $( 4 0 9 6 + 1 0 2 4 ) / 2 = 2 5 6 0 .$ .

Online question answering. At inference time, the query is encoded with the query adapter and projected into the same normalized retrieval space. FAISS returns the top-k memory entries. Their full latent vectors are ordered by retrieval score, projected with cross\_proj, and inserted into the frozen generator through inputs\_embeds. Ordinary prompt tokens are embedded with the generator embedding layer; projected latent memories are spliced between the prompt prefix and suffix. No generator weights are updated during training or evaluation.

Trainable components. The text model uses separate LoRA adapters for compress, query, decode, and query\_decode. The multimodal model additionally uses image\_decode. Encoderside adapters target q\_proj, k\_proj, v\_proj, and o\_proj. Decoder-side adapters also include gate\_proj and up\_proj. The trainable non-LoRA components are the [MEM] token, retrieval projector, generator projector, decoder projector, and image embedding reconstruction projector.

Training-Loss Distillation. The teacher is the frozen generator prompted with raw positive evidence, while the student uses the same frozen generator prompted with projected latent memories. For text-side distillation, the student latent context is randomly augmented with 0–3 sampled hardnegative latent memories during training, while the teacher prompt remains positive-only. This augmentation exposes the student generator to small retrieval noise without changing the teacher target distribution; the KL loss is still computed on the first 16 generated answer tokens.

## B.3 Hyperparameter Settings

Table 7: Main hyperparameters and LoRA configuration used in the reported runs.

<table><tr><td>Component</td><td>Text-only</td><td>Multimodal</td></tr><tr><td>Backbone / generator</td><td>Llama-3.2-1B-Instruct compressor → frozen Meta-Llama-3-8B-Instruct generator</td><td>LLaVA-1.5-7B → frozen LLaVA-1.5-13B</td></tr><tr><td>Compression / latent dimension  $d_{\theta}$ </td><td>2048 for Llama-3.2-1B; one latent token per evidence item in the main setting</td><td>4096 for LLaVA-1.5-7B; one latent token per text or image evidence item</td></tr><tr><td>Retrieval dimension  $d_r$ </td><td>512-dimensional normalized retrieval vector for FAISS inner-product search</td><td>512-dimensional normalized retrieval vector shared by text and image evidence</td></tr><tr><td>Generation dimension  $d_{\phi}$ </td><td>4096 for the frozen Meta-Llama-3-8B generator; cross_proj: 2048 → 3072 → 4096</td><td>5120 for frozen LLaVA-1.5-13B with cross_proj: 4096 → 4608 → 5120</td></tr><tr><td>Text reconstruction decoder</td><td>Llama-3.2-1B-Instruct</td><td>Llama-3.2-1B-Instruct for reported WebQA runs</td></tr><tr><td>Decoder / reconstruction dimension  $d_\pi$ </td><td>2048 decoder hidden size; decode_proj: 2048 → 2048</td><td>2048 LLaMA decoder hidden size for reported WebQA runs; LLaVA decode_proj: 4096 → 2048; image embedding target  $d_v$ =1024</td></tr><tr><td>Encoder-side LoRA targets</td><td>q_proj, k_proj, v_proj, o_proj</td><td>q_proj, k_proj, v_proj, o_proj</td></tr><tr><td>Decoder-side LoRA targets</td><td>q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj</td><td>q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj</td></tr><tr><td>LoRA rank / alpha / dropout</td><td>r=64, α=128, 0.05</td><td>r=64, α=128, 0.05</td></tr><tr><td>Optimizer and LR</td><td>AdamW, peak LR 1×10-4</td><td>AdamW, peak LR 1×10-4</td></tr><tr><td>Training batching</td><td>batch size 8, gradient accumulation 4</td><td>batch size 8, gradient accumulation 2</td></tr><tr><td>Training epochs</td><td>3 epochs (about 20 hours)</td><td>2 epochs (about 30 hours)</td></tr><tr><td>Text loss weights</td><td> $λ_{\text{recon}}$ =0.5,  $λ_{\text{contrast}}$ =0.2,  $λ_{\text{distill}}$ =1.0</td><td> $λ_{\text{recon}}$ =0.5,  $λ_{\text{contrast}}$ =0.2,  $λ_{\text{distill}}$ =1.0</td></tr><tr><td>Image loss weights</td><td>-</td><td> $λ_{\text{image-contrast}}$ =0.2,  $λ_{\text{image-distill}}$ =2.0</td></tr><tr><td>Embedding reconstruction</td><td>-</td><td> $λ_{\text{embed-recon}}$ =5.0</td></tr><tr><td>Hard negatives per sample</td><td>up to 8 text negatives</td><td>up to 4 text + 4 image negatives</td></tr><tr><td>Distillation supervision</td><td>first 16 answer tokens</td><td>first 16 answer tokens</td></tr></table>

The hyperparameter of loss is selected basically based on the scale.

Additional implementation choices are shared across settings. During training, retrieval scores are computed in memory without building a FAISS index. During offline evaluation, we build an FAISS inner-product index over normalized retrieval vectors. Negative reconstruction is enabled for text evidence and image captions.

Query reconstruction is disabled in the main runs. The token count considers from “Context 1:“ to the end, so introducing one more latent token may consume more than one generator token budget (usually 7 to 10 based on LLM/VLM tokenizer). For WebQA token accounting, raw retrieved images are counted after the generator’s native visual frontend; Latent Memory bypasses this raw visual expansion and inserts one projected latent token per retrieved item.

## B.4 Time Complexity Analysis

This section compares the dominant time complexity under two deployment settings: the evidence index or memory bank has already been compiled, or it must be compiled for the current corpus. Let $\boldsymbol { \mathcal { C } } = \{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N }$ evidence item. We denote the retrieval depth by $k ,$ the number of latent tokens per evidence item by T , the question length by |Q|, the frozen generator hidden size by $d _ { \phi } .$ , the retrieval embedding model hidden size by $d _ { e } ,$ and the Latent Memory compressor hidden size by $d _ { \theta }$ The table omits retrieval/search overhead and keeps only the dominant evidence-encoding and generator-prefill terms.

Table 8: Dominant time complexity comparison under precompiled and non-precompiled deployment. Retrieval/search overhead is omitted; the key term is the frozen generator prefill length.

<table><tr><td>Method</td><td>With precompiled index / memory</td><td>Without precompilation</td></tr><tr><td>Full Context</td><td> $\mathcal{O}\left((|Q| + N\bar{L})^{2}d_{\phi}\right)$ </td><td> $\mathcal{O}\left((|Q| + N\bar{L})^{2}d_{\phi}\right)$ </td></tr><tr><td>Raw-evidence RAG</td><td> $\mathcal{O}\left((|Q| + k\bar{L})^{2}d_{\phi}\right)$ </td><td> $\mathcal{O}\left(N\bar{L}^{2}d_{e} + (|Q| + k\bar{L})^{2}d_{\phi}\right)$ </td></tr><tr><td>Latent Memory</td><td> $\mathcal{O}\left((|Q| + kT)^{2}d_{\phi}\right)$ </td><td> $\mathcal{O}\left(N\bar{L}^{2}d_{\theta} + (|Q| + kT)^{2}d_{\phi}\right)$ </td></tr></table>

These formulas follow directly from Transformer prefill complexity: forwarding a sequence of length S through the frozen generator costs $\mathcal { O } ( S ^ { 2 } d _ { \phi } )$ . Full Context uses $S = | Q | + { \cal N } \bar { \cal L }$ , RAG uses $S = | Q | + k \bar { L }$ after selecting k raw evidence items, and Latent Memory uses $S = | \pmb { Q } | + k T$ because each retrieved evidence item is represented by T latent tokens.

• Compared with Full Context, Latent Memory avoids sending the whole evidence pool to the generator. Full Context has a generator-prefill term ${ \mathcal O } ( ( | Q | + N \mathbf { \bar { \bar { L } } } ) ^ { 2 } d _ { \phi } )$ , which grows quadratically with the number of evidence items N when L¯ is fixed. With precompilation, Latent Memory instead uses $\mathcal { O } ( ( \vert Q \vert + k T ) ^ { 2 } d _ { \phi } )$ , which no longer depends on N in the generator. Without precompilation, Latent Memory adds the evidence compilation term $\mathcal { O } ( N \bar { L } ^ { 2 } d _ { \theta } )$ , which is linear in N and can be highly parallelized across evidence items. As shown in the figure below, we plot the relationship of time consumption with the number of evidence items. Latent Memory shows linear complexity and leads to significantly less time complexity, even with compilation.

![](images/656f5e640ac1bb8806e2b5049f21eb533783da02d5444820221ca37b959cd191.jpg)

<details>
<summary>bar chart</summary>

| Number of evidence items | Full Context (s) | Latent Compile (s) | Latent Prefill (s) |
| :--- | :--- | :--- | :--- |
| 100 | 0.44 | 0.33 | 0.03 |
| 200 | 0.86 | 0.57 | 0.03 |
| 500 | 2.52 | 1.36 | 0.03 |
| 1000 | 6.39 | 2.69 | 0.06 |
</details>

Figure 6: Time analysis on HotpotQA, full-context vs Latent Memory. Compile means pre-compile, and prefill means the prefill process before answer-generation. We run 30 times for variance.

• Compared with raw-evidence RAG, Latent Memory has a similar non-precompiled setup structure: RAG embeds all evidence items, while Latent Memory compiles them into latent memories. The key difference is again the generator input after retrieval. RAG still sends k raw evidence items to the generator, giving $| \bar { Q | } + k \bar { L }$ , whereas Latent Memory sends $k T$ latent tokens, giving $| Q | + k T$ . Since $T \ll \bar { L }$ and usually k ≪ N, Latent Memory has the smallest online generatorprefill complexity among the three methods. Moreover, fewer tokens lead to less probability of out-of-memory, and a square-root less space complexity.

## B.5 Storage Complexity Analysis

Table 9 reports an average per-evidence storage comparison on WebQA. The text side uses the official WebQA text snippets in WebQA\_train\_val.json. The image side uses the official WebQA image files after extraction, which occupy about 75GB in total (the compressed package is about 50GB), averaged over the 350,777 unique image IDs appearing in the official train/validation annotations. We count only the stored evidence representation and do not include small metadata fields.

Table 9: Average per-item storage on WebQA. Raw text and raw image storage follow the official WebQA annotations and extracted image files. Latent text uses the text-only 1-token memory size (2048 bf16 values), while latent image uses the LLaVA 1-token memory size (4096 bf16 values).

<table><tr><td>Evidence type</td><td>Stored representation</td><td>Avg. storage / item</td><td>Storage effect</td></tr><tr><td>Text</td><td>Official raw text snippet</td><td>0.23 KB</td><td>Reference</td></tr><tr><td>Text</td><td>Latent Memory token</td><td>4.00 KB</td><td>17.6× larger</td></tr><tr><td>Image</td><td>Official extracted image file</td><td>209 KB</td><td>Reference</td></tr><tr><td>Image</td><td>Latent Memory token</td><td>8.00 KB</td><td>26.1× smaller</td></tr></table>

This comparison clarifies where the storage advantage comes from. On the pure language side, a short raw snippet is often smaller than a high-dimensional latent vector, so text-only Latent Memory should mainly be understood as reducing generator-side token usage rather than persistent corpus storage. On the image side, however, the raw evidence item is much larger; replacing each extracted WebQA image with a single LLaVA latent token gives a clear storage reduction while also avoiding raw visual-token expansion during generation.

The same conclusion also follows from a simple analytical threshold. A one-token LLaVA Latent Memory stores 4096 bf16 values, i.e.,

$$
S _ {\text { latent / image }} = 2 \times 4 0 9 6 = 8 1 9 2 \text { bytes }. \tag {11}
$$

For an uncompressed RGB image with height H and width W , raw storage is

$$
S _ {\text { image }} = 3 H W \text {   bytes. } \tag {12}
$$

Thus, Latent Memory is smaller whenever $3 H W > 8 1 9 2$ , or $H = W > \sqrt { 8 1 9 2 / 3 } \approx 5 2 . 3$ for square images. In other words, a 4096-dimensional bf16 latent token is more storage efficient than an uncompressed RGB image once the image is larger than roughly $5 3 \times 5 3$ pixels. If a separate 512-dimensional fp32 retrieval key is also stored, the per-item latent storage becomes 8192 + 4 × 512 = 10240 bytes, shifting the square-image threshold only to about 59 × 59 pixels.

The online activation footprint follows the same pattern. Raw multimodal RAG must instantiate visual embeddings for each retrieved image inside the VLM, giving an evidence activation size of $\mathcal { O } ( k \bar { L } d _ { \phi } )$ . Latent Memory instead instantiates only projected latent tokens, giving $\mathcal { O } ( k T d _ { \phi } )$ . Thus the online generation path reduces both token consumption and activation memory when $\bar { L } \gg T$ , which is typical for image evidence after visual-token expansion.

## C Additional Experiments and Discussion

The experiments in the main text answer the three questions as follows:

• 1-token Latent Memory can show a token-performance trade-off in multimodal settings, replacing the raw retrieved context.  
• Replacing raw evidence with 8-token Latent Memory results in a better trade-off, surpassing the best RAG baseline on out-of-domain performance.

The appendix then studies where this behavior comes from and how broadly it holds. We organize the additional evidence around the following questions:

• RQ-A1: How much token each latent evidence are needed for text-only and multimodal QA? Appendix C.1 varies the number of latent tokens per evidence item.  
• RQ-A2: Does the Latent Memory perform well across evidence domains, compressors, and generators? Appendices C.2, C.5, C.4, and C.6 test broader text and image QA benchmarks, compressors, and generator settings.  
• RQ-A3: Which training objectives support the unified representation space? Appendix C.3 ablates reconstruction, negative reconstruction, query reconstruction, and augmentation settings.  
• RQ-A5: How effective are compression and retrieval, respectively? Appendix C.7 excludes R@k improvements, how effective is compression? We considered a variant using latent tokens to provide the generator with raw evidence after retrieval to observe compression effectiveness.

## C.1 Token Count Ablation

This section varies the number of latent tokens allocated to each evidence item. The default model uses one token; we also evaluate 2-, 4-, and 8-token variants while keeping the evidence unit and retrieval granularity unchanged. For an evidence item xi, the multi-token variant emits

$$
\boldsymbol {Z} _ {i} = \left[ \boldsymbol {z} _ {i, 1}, \boldsymbol {z} _ {i, 2}, \dots , \boldsymbol {z} _ {i, T} \right], \quad \boldsymbol {z} _ {i, t} \in \mathbb {R} ^ {d _ {\theta}}. \tag {13}
$$

Retrieval still uses one key per evidence item by pooling these tokens before the retrieval projection:

$$
\bar {\boldsymbol {z}} _ {i} = \frac {1}{T} \sum_ {t = 1} ^ {T} \boldsymbol {z} _ {i, t}, \quad \widetilde {\boldsymbol {z}} _ {i} = \text { LayerNorm } (\boldsymbol {W} _ {r} \bar {\boldsymbol {z}} _ {i}). \tag {14}
$$

Thus, the ablation tests whether extra latent capacity improves generation enough to justify the larger token budget.

Table 10 reports the text-QA results, and Figures 7–9 summarize the accuracy, token, and recall trends.

Analysis. Three conclusions are consistent across text-only setting (Table 10 and Figures 7–9).

1. Increasing the latent-token budget consistently improves EM and F1, and the larger-token variants surpass the main retrieval baselines in answer quality.  
2. By contrast, increasing the latent-token budget does not lead to a comparable improvement in Recall@k at fixed k.  
3. These quality gains are achieved while preserving the overall trade-off: larger latent budgets do increase generator tokens, but they remain substantially more efficient than Full Context and still compare favorably with the raw retrieval baselines.

Table 10: Token-count ablation on text QA over HotpotQA, 2WikiMultihopQA, and MuSiQue. The Average columns report the out-of-domain average over 2WikiMultihopQA and MuSiQue. The 1-token setting is the main Latent Memory configuration; 2/4/8-token variants allocate more latent tokens per evidence item. #Tok reports the task-relevant generator budget.

<table><tr><td rowspan="3">Dataset Method</td><td colspan="4">In-Domain</td><td colspan="12">Out-of-Domain</td></tr><tr><td colspan="4">HotpotQA</td><td colspan="4">2WikiMultihopQA</td><td colspan="4">MuSiQue</td><td colspan="4">Average</td></tr><tr><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td></tr><tr><td>Full Context</td><td>42.0</td><td>57.5</td><td>-</td><td>1462</td><td>17.7</td><td>39.7</td><td>-</td><td>1074</td><td>6.0</td><td>15.5</td><td>-</td><td>2580</td><td>11.9</td><td>27.6</td><td>-</td><td>1827</td></tr><tr><td>LLMLingua (20%)</td><td>31.8</td><td>44.8</td><td>-</td><td>283</td><td>17.0</td><td>30.4</td><td>-</td><td>199</td><td>11.6</td><td>21.8</td><td>-</td><td>492</td><td>14.3</td><td>26.1</td><td>-</td><td>346</td></tr><tr><td>LLMLingua (10%)</td><td>25.0</td><td>36.1</td><td>-</td><td>154</td><td>14.8</td><td>24.7</td><td>-</td><td>108</td><td>6.9</td><td>14.9</td><td>-</td><td>259</td><td>10.9</td><td>19.8</td><td>-</td><td>184</td></tr><tr><td>LLMLingua (5%)</td><td>20.9</td><td>30.2</td><td>-</td><td>87</td><td>15.0</td><td>22.2</td><td>-</td><td>63</td><td>4.3</td><td>10.6</td><td>-</td><td>137</td><td>9.7</td><td>16.4</td><td>-</td><td>100</td></tr><tr><td colspan="17">Retrieval Augmented Generation Methods</td></tr><tr><td>BM25 Retrieval (k=1)</td><td>32.5</td><td>44.7</td><td>30.2</td><td>68</td><td>17.2</td><td>27.2</td><td>18.3</td><td>60</td><td>6.2</td><td>14.2</td><td>7.0</td><td>69</td><td>11.7</td><td>20.7</td><td>12.7</td><td>65</td></tr><tr><td>BM25 Retrieval (k=2)</td><td>36.8</td><td>49.9</td><td>45.5</td><td>106</td><td>16.2</td><td>28.4</td><td>29.2</td><td>94</td><td>8.0</td><td>16.7</td><td>12.2</td><td>106</td><td>12.1</td><td>22.6</td><td>20.7</td><td>100</td></tr><tr><td>BM25 Retrieval (k=5)</td><td>41.3</td><td>55.3</td><td>65.5</td><td>224</td><td>16.2</td><td>32.8</td><td>46.9</td><td>196</td><td>9.5</td><td>19.6</td><td>22.4</td><td>221</td><td>12.9</td><td>26.2</td><td>34.7</td><td>209</td></tr><tr><td>Dense Retrieval (k=1)</td><td>29.4</td><td>41.0</td><td>27.9</td><td>67</td><td>19.0</td><td>29.9</td><td>23.3</td><td>61</td><td>7.8</td><td>17.1</td><td>9.6</td><td>70</td><td>13.4</td><td>23.5</td><td>16.5</td><td>66</td></tr><tr><td>Dense Retrieval (k=2)</td><td>32.6</td><td>45.3</td><td>42.2</td><td>104</td><td>18.1</td><td>31.7</td><td>37.6</td><td>95</td><td>9.9</td><td>19.9</td><td>17.0</td><td>106</td><td>14.0</td><td>25.8</td><td>27.3</td><td>101</td></tr><tr><td>Dense Retrieval (k=5)</td><td>37.0</td><td>50.7</td><td>62.4</td><td>214</td><td>19.1</td><td>37.4</td><td>58.0</td><td>198</td><td>12.8</td><td>23.4</td><td>31.3</td><td>218</td><td>16.0</td><td>30.4</td><td>44.7</td><td>208</td></tr><tr><td>Qwen3-Emb-0.6B (k=1)</td><td>30.9</td><td>43.6</td><td>33.1</td><td>70</td><td>18.1</td><td>30.9</td><td>32.5</td><td>66</td><td>8.1</td><td>18.1</td><td>10.3</td><td>73</td><td>13.1</td><td>24.5</td><td>21.4</td><td>70</td></tr><tr><td>Qwen3-Emb-0.6B (k=2)</td><td>35.6</td><td>49.0</td><td>50.0</td><td>109</td><td>17.4</td><td>33.5</td><td>47.7</td><td>102</td><td>9.8</td><td>20.4</td><td>18.5</td><td>112</td><td>13.6</td><td>27.0</td><td>33.1</td><td>107</td></tr><tr><td>Qwen3-Emb-0.6B (k=5)</td><td>40.0</td><td>54.5</td><td>70.1</td><td>224</td><td>19.1</td><td>38.6</td><td>64.3</td><td>208</td><td>13.7</td><td>24.6</td><td>34.8</td><td>230</td><td>16.4</td><td>31.6</td><td>49.6</td><td>219</td></tr><tr><td colspan="17">Retrieval Augmented Generation for Latent Tokens in the Latent Memory (Ours)</td></tr><tr><td>1-token (k=1)</td><td>27.4</td><td>39.4</td><td>34.6</td><td>36</td><td>19.8</td><td>29.2</td><td>28.4</td><td>33</td><td>5.8</td><td>14.6</td><td>8.7</td><td>37</td><td>12.8</td><td>21.9</td><td>18.6</td><td>35</td></tr><tr><td>1-token (k=2)</td><td>31.6</td><td>45.2</td><td>62.6</td><td>45</td><td>21.5</td><td>33.5</td><td>49.5</td><td>42</td><td>7.3</td><td>16.9</td><td>15.5</td><td>46</td><td>14.4</td><td>25.2</td><td>32.5</td><td>44</td></tr><tr><td>1-token (k=5)</td><td>34.8</td><td>48.9</td><td>87.1</td><td>72</td><td>24.3</td><td>36.7</td><td>74.2</td><td>69</td><td>8.7</td><td>19.2</td><td>30.1</td><td>73</td><td>16.5</td><td>28.0</td><td>52.2</td><td>71</td></tr><tr><td>2-token (k=1)</td><td>27.3</td><td>39.5</td><td>34.6</td><td>37</td><td>20.2</td><td>29.9</td><td>28.0</td><td>34</td><td>5.7</td><td>15.2</td><td>8.7</td><td>38</td><td>13.0</td><td>22.6</td><td>18.4</td><td>36</td></tr><tr><td>2-token (k=2)</td><td>31.4</td><td>45.2</td><td>62.6</td><td>47</td><td>23.0</td><td>34.7</td><td>49.0</td><td>44</td><td>7.3</td><td>17.6</td><td>15.2</td><td>48</td><td>15.2</td><td>26.2</td><td>32.1</td><td>46</td></tr><tr><td>2-token (k=5)</td><td>35.2</td><td>49.3</td><td>87.1</td><td>77</td><td>26.6</td><td>38.0</td><td>73.8</td><td>74</td><td>8.5</td><td>19.8</td><td>29.4</td><td>78</td><td>17.6</td><td>28.9</td><td>51.6</td><td>76</td></tr><tr><td>4-token (k=1)</td><td>28.6</td><td>40.4</td><td>35.2</td><td>39</td><td>21.9</td><td>31.4</td><td>30.9</td><td>36</td><td>5.8</td><td>15.8</td><td>9.0</td><td>40</td><td>13.9</td><td>23.6</td><td>20.0</td><td>38</td></tr><tr><td>4-token (k=2)</td><td>33.0</td><td>46.9</td><td>63.2</td><td>51</td><td>25.1</td><td>36.2</td><td>53.5</td><td>48</td><td>7.6</td><td>18.1</td><td>16.0</td><td>52</td><td>16.4</td><td>27.2</td><td>34.8</td><td>50</td></tr><tr><td>4-token (k=5)</td><td>35.8</td><td>49.9</td><td>88.0</td><td>87</td><td>28.9</td><td>39.1</td><td>76.7</td><td>84</td><td>8.8</td><td>19.9</td><td>31.0</td><td>88</td><td>18.9</td><td>29.5</td><td>53.9</td><td>86</td></tr><tr><td>8-token (k=1)</td><td>31.6</td><td>44.7</td><td>35.1</td><td>43</td><td>21.2</td><td>31.6</td><td>30.5</td><td>40</td><td>7.5</td><td>17.7</td><td>9.2</td><td>44</td><td>14.4</td><td>24.7</td><td>19.9</td><td>42</td></tr><tr><td>8-token (k=2)</td><td>38.9</td><td>53.2</td><td>63.4</td><td>59</td><td>24.0</td><td>36.4</td><td>52.8</td><td>56</td><td>10.6</td><td>21.5</td><td>16.2</td><td>60</td><td>17.3</td><td>29.0</td><td>34.5</td><td>58</td></tr><tr><td>8-token (k=5)</td><td>42.8</td><td>58.5</td><td>88.0</td><td>107</td><td>26.8</td><td>40.5</td><td>76.2</td><td>104</td><td>13.0</td><td>24.5</td><td>31.1</td><td>108</td><td>19.9</td><td>32.5</td><td>53.7</td><td>106</td></tr></table>

![](images/9ba2f52f8b9eec2c55e55617fd5643c33dc8436177e31e4b308444b17eafddca.jpg)

<details>
<summary>line chart</summary>

| Method | k | Average Input Tokens to Generator | OOD Average EM |
| --- | --- | --- | --- |
| Full Context | 1 | 1000 | 12 |
| BM25 Retrieval | 1 | 100 | 12.5 |
| BM25 Retrieval | 2 | 100 | 13.5 |
| BM25 Retrieval | 5 | 100 | 14.5 |
| BM25 Retrieval | 10 | 100 | 15.5 |
| BM25 Retrieval | 20 | 100 | 16.5 |
| BM25 Retrieval | 50 | 100 | 17.5 |
| BM25 Retrieval | 100 | 100 | 18.5 |
| BM25 Retrieval | 200 | 100 | 19.5 |
| BM25 Retrieval | 500 | 100 | 20.5 |
| BM25 Retrieval | 1000 | 100 | 21.5 |
| Latent (2-token) | 1 | 100 | 13.5 |
| Latent (2-token) | 2 | 100 | 14.5 |
| Latent (2-token) | 4 | 100 | 15.5 |
| Latent (2-token) | 8 | 100 | 16.5 |
| Latent (4-token) | 1 | 100 | 13.5 |
| Latent (4-token) | 2 | 100 | 14.5 |
| Latent (4-token) | 4 | 100 | 15.5 |
| Latent (4-token) | 8 | 100 | 16.5 |
| Latent (4-token) | 10 | 100 | 17.5 |
| Latent (4-token) | 20 | 100 | 18.5 |
| Latent (4-token) | 50 | 100 | 19.5 |
| Latent (4-token) | 80 | 100 | 20.5 |
| Latent (4-token) | 100 | 100 | 21.5 |
| Latent (4-token) | 200 | 100 | 22.5 |
| Latent (4-token) | 500 | 100 | 23.5 |
| Latent (4-token) | 800 | 100 | 24.5 |
| Latent (4-token) | 1000 | 100 | 25.5 |
| LLLMingua (20%) | - | ~300 | - |
| LLLMingua (10%) | - | ~300 | - |
| LLLMingua (5%) | - | ~300 | - |
| Qwen3-Emb | - | ~300 | - |
| Latent (1-token) | - | ~300 | - |
| Latent (8-token) | - | ~300 | - |
| Latent (8-token) | - | ~300 | - |
| Latent (8-token) | - | ~300 | - |
| Latent (8-token) | - | ~300 | - |
| Latent (8-token) | - | ~300 | - |
| Latent (8-token) | - | ~300 | - |
| Latenta (k=1) | - | ~300 | - |
| Latenta (k=1) | - | ~300 | - |
| Latenta (k=1) | - | ~300 | - |
| Latenta (k=1) | - | ~300 | - |
| Latenta (k=1) | - | ~300 | - |
| Latenta (k=1) | - | ~300 | ~9.5 |
| Latenta (k=1) | - | ~300 | ~9.5 |
| Latenta (k=1) | - | ~300 | ~9.5 |
| Latenta (k=1) | - | ~300 | ~9.5 |
| Latenta (k=1) | - | ~300 | ~9.5 |
</details>

Figure 7: OOD average text-QA EM as a function of average generator tokens. The average is computed over 2WikiMultihopQA and MuSiQue.

![](images/e362b1c92c64192813353bd4275c4190232183241865e56369913ab7843212b6.jpg)

<details>
<summary>line chart</summary>

| Method | k | Average Input Tokens to Generator | OOD Average F1 |
| --- | --- | --- | --- |
| Full Context | 1 | 1000 | 27.5 |
| BM25 Retrieval | 1 | 200 | 26.0 |
| BM25 Retrieval | 2 | 300 | 26.5 |
| BM25 Retrieval | 3 | 400 | 27.0 |
| BM25 Retrieval | 4 | 500 | 27.5 |
| BM25 Retrieval | 5 | 600 | 28.0 |
| BM25 Retrieval | 6 | 700 | 28.5 |
| BM25 Retrieval | 7 | 800 | 29.0 |
| BM25 Retrieval | 8 | 900 | 29.5 |
| BM25 Retrieval | 9 | 1000 | 30.0 |
| BM25 Retrieval | 10 | 1100 | 30.5 |
| BM25 Retrieval | 11 | 1200 | 31.0 |
| BM25 Retrieval | 12 | 1300 | 31.5 |
| BM25 Retrieval | 13 | 1400 | 32.0 |
| BM25 Retrieval | 14 | 1500 | 32.5 |
| BM25 Retrieval | 15 | 1600 | 33.0 |
| BM25 Retrieval | 16 | 1700 | 33.5 |
| BM25 Retrieval | 17 | 1800 | 34.0 |
| BM25 Retrieval | 18 | 1900 | 34.5 |
| BM25 Retrieval | 19 | 2000 | 35.0 |
| BM25 Retrieval | 20 | 2100 | 35.5 |
| BM25 Retrieval | 21 | 2200 | 36.0 |
| BM25 Retrieval | 22 | 2300 | 36.5 |
| BM25 Retrieval | 23 | 2400 | 37.0 |
| BM25 Retrieval | 24 | 2500 | 37.5 |
| BM25 Retrieval | 25 | 2600 | 38.0 |
| BM25 Retrieval | 26 | 2700 | 38.5 |
| BM25 Retrieval | 27 | 2800 | 39.0 |
| BM25 Retrieval | 28 | 2900 | 39.5 |
| BM25 Retrieval | 29 | 3000 | 40.0 |
| BM25 Retrieval | 30 | 3100 | 40.5 |
| BM25 Retrieval | 31 | 3200 | 41.0 |
| BM25 Retrieval | 32 | 3300 | 41.5 |
| BM25 Retrieval | 33 | 3400 | 42.0 |
| BM25 Retrieval | 34 | 3500 | 42.5 |
| BM25 Retrieval | 35 | 3600 | 43.0 |
| BM25 Retrieval | 36 | 3700 | 43.5 |
| BM25 Retrieval | 37 | 3800 | 44.0 |
| BM25 Retrieval | 38 | 3900 | 44.5 |
| BM25 Retrieval | 39 | 4000 | 45.0 |
| BM25 Retrieval | 40 | 4100 | 45.5 |
| BM25 Retrieval | 41 | 4200 | 46.0 |
| BM25 Retrieval | 42 | 4300 | 46.5 |
| BM25 Retrieval | 43 | 4400 | 47.0 |
| BM25 Retrieval | 44 | 4500 | 47.5 |
| BM25 Retrieval | 45 | 4600 | 48.0 |
| BM25 Retrieval | 46 | 4700 | 48.5 |
| BM25 Retrieval | 47 | 4800 | 49.0 |
| BM25 Retrieval | 48 | 4900 | 49.5 |
| BM25 Retrieval | 49 | 5000 | 50.0 |
| BM25 Retrieval | 50 | 5100 | 50.5 |
| BM25 Retrieval | 51 | 5200 | 51.0 |
| BM25 Retrieval | 52 | 5300 | 51.5 |
| BM25 Retrieval | 53 | 5400 | 52.0 |
| BM25 Retrieval | 54 | 5500 | 52.5 |
| BM25 Retrieval | 55 | 5600 | 53.0 |
| BM25 Retrieval | 56 | 5700 | 53.5 |
| BM25 Retrieval | 57 | 5800 | 54.0 |
| BM25 Retrieval | 58 | 5900 | 54.5 |
| BM25 Retrieval | 59 | 6000 | 55.0 |
| BM25 Retrieval | 60 | 6100 | 55.5 |
| BM25 Retrieval | 61 | 6200 | 56.0 |
| BM25 Retrieval | 62 | 6300 | 56.5 |
| BM25 Retrieval | 63 | 6400 | 57.0 |
| BM25 Retrieval | 64 | 6500 | 57.5 |
| BM25 Retrieval | 65 | 6600 | 58.0 |
| BM25 Retrieval | 66 | 6700 | 58.5 |
| BM25 Retrieval | 67 | 6800 | 59.0 |
| BM25 Retrieval | 68 | 6900 | 59.5 |
| BM25 Retrieval | 69 | 7000 | 60.0 |
| BM25 Retrieval | 70 | 7100 | 60.5 |
| BM25 Retrieval | 71 | 7200 | 61.0 |
| BM25 Retrieval | 72 | 7300 | 61.5 |
| BM25 Retrieval | 73 | 7400 | 62.0 |
| BM25 Retrieval | 74 | 7500 | 62.5 |
| BM25 Retrieval | 75 | 7600 | 63.0 |
| BM25 Retrieval | 76 | 7700 | 63.5 |
| BM25 Retrieval | 77 | 7800 | 64.0 |
| BM25 Retrieval | 78 | 7900 | 64.5 |
| BM25 Retrieval | 79 | 8000 | 65.0 |
</details>

Figure 8: OOD average text-QA F1 as a function of average generator tokens. The average is computed over 2WikiMultihopQA and MuSiQue.

![](images/e62f86089a6f486f804a08452590f739723c772841dc87fc320686c967c92304.jpg)

<details>
<summary>line chart</summary>

| Kutoff | BM25 Retrieval | Dense Retrieval | Owen3-Emb | Latent (1-token) | Latent (2-token) | Latent (4-token) | Latent (8-token) |
| ------ | -------------- | --------------- | --------- | ---------------- | ---------------- | ---------------- | ---------------- |
| 1      | 10             | 17              | 22        | 19               | 18               | 20               | 21               |
| 2      | 21             | 27              | 33        | 32               | 33               | 35               | 36               |
| 5      | 35             | 45              | 50        | 52               | 53               | 54               | 55               |
</details>

Figure 9: OOD average Recall@k as a function of k. The average is computed over 2WikiMultihopQA and MuSiQue.

## C.2 Generalization Ability on More Domains

The experiments in the main body of this paper demonstrate that Latent Memory remains effective on the OOD dataset, but these results can be considered generalizations to similar corpora. In this section, we consider whether this effectiveness can be extended to more different situations and datasets that are very different from HotpotQA. Table 11 reports generalization to four additional text QA benchmarks: NQ (open-domain factoid), TriviaQA (trivia-style factoid), Qasper (scientific document QA), and WICE (claim verification). Unlike the three datasets in the main text, these datasets do not have a retrieval label design, so we did not calculate R@k. No fine-tuning is performed on these datasets; the same checkpoint trained on HotpotQA is evaluated directly. This tests whether the compressed latent representations generalize across domains and task types.

Table 11: Generalization ability on more text domains under the same frozen Meta-Llama-3-8B-Instruct generator. No fine-tuning is performed on these datasets. #Tok reports the task-relevant generator budget. Unlike the three datasets in the main text, these datasets do not have a retrieval label design, so we did not calculate R@k.

<table><tr><td rowspan="2">Method</td><td colspan="3">NQ</td><td colspan="3">TriviaQA</td><td colspan="3">Qasper</td><td colspan="3">WiCE</td><td colspan="3">Average</td></tr><tr><td>EM</td><td>F1</td><td>#Tok</td><td>EM</td><td>F1</td><td>#Tok</td><td>EM</td><td>F1</td><td>#Tok</td><td>EM</td><td>F1</td><td>#Tok</td><td>EM</td><td>F1</td><td>#Tok</td></tr><tr><td>Full Context</td><td>0.0</td><td>1.1</td><td>23588</td><td>18.9</td><td>24.7</td><td>10828</td><td>2.5</td><td>16.5</td><td>4701</td><td>51.1</td><td>51.1</td><td>2311</td><td>18.1</td><td>23.3</td><td>10357</td></tr><tr><td colspan="16">Raw-evidence retrieval</td></tr><tr><td>BM25 (k=1)</td><td>25.3</td><td>36.6</td><td>54</td><td>71.0</td><td>77.7</td><td>66</td><td>6.0</td><td>17.1</td><td>50</td><td>43.6</td><td>43.6</td><td>135</td><td>36.5</td><td>43.8</td><td>77</td></tr><tr><td>BM25 (k=2)</td><td>26.7</td><td>38.6</td><td>90</td><td>70.7</td><td>77.8</td><td>106</td><td>8.0</td><td>20.5</td><td>82</td><td>44.1</td><td>44.1</td><td>210</td><td>37.4</td><td>45.3</td><td>122</td></tr><tr><td>BM25 (k=5)</td><td>28.1</td><td>40.4</td><td>197</td><td>71.0</td><td>78.3</td><td>226</td><td>9.5</td><td>23.6</td><td>187</td><td>51.7</td><td>51.7</td><td>417</td><td>40.1</td><td>48.5</td><td>257</td></tr><tr><td>Dense (k=1)</td><td>28.2</td><td>40.5</td><td>58</td><td>70.2</td><td>77.1</td><td>57</td><td>5.5</td><td>16.6</td><td>46</td><td>39.9</td><td>39.9</td><td>116</td><td>36.0</td><td>43.5</td><td>69</td></tr><tr><td>Dense (k=2)</td><td>30.6</td><td>43.5</td><td>96</td><td>71.2</td><td>78.3</td><td>85</td><td>8.5</td><td>21.1</td><td>76</td><td>43.6</td><td>43.6</td><td>174</td><td>38.5</td><td>46.6</td><td>108</td></tr><tr><td>Dense (k=5)</td><td>33.1</td><td>46.5</td><td>210</td><td>71.7</td><td>79.2</td><td>171</td><td>11.5</td><td>26.8</td><td>168</td><td>46.4</td><td>46.4</td><td>351</td><td>40.7</td><td>49.7</td><td>225</td></tr><tr><td>Qwen3-Emb-0.6B (k=1)</td><td>30.8</td><td>43.0</td><td>60</td><td>71.9</td><td>79.0</td><td>70</td><td>8.5</td><td>19.8</td><td>47</td><td>42.2</td><td>42.2</td><td>126</td><td>38.4</td><td>46.0</td><td>76</td></tr><tr><td>Qwen3-Emb-0.6B (k=2)</td><td>32.0</td><td>45.1</td><td>99</td><td>72.1</td><td>79.8</td><td>110</td><td>10.0</td><td>26.6</td><td>80</td><td>42.5</td><td>42.5</td><td>201</td><td>39.1</td><td>48.5</td><td>122</td></tr><tr><td>Qwen3-Emb-0.6B (k=5)</td><td>34.5</td><td>48.2</td><td>218</td><td>72.8</td><td>80.9</td><td>221</td><td>8.5</td><td>30.2</td><td>171</td><td>47.5</td><td>47.5</td><td>389</td><td>40.8</td><td>51.7</td><td>250</td></tr><tr><td colspan="16">Latent Memory</td></tr><tr><td>Latent Memory (k=1)</td><td>22.2</td><td>33.7</td><td>26</td><td>66.5</td><td>72.9</td><td>34</td><td>3.0</td><td>10.9</td><td>25</td><td>45.5</td><td>45.5</td><td>59</td><td>34.3</td><td>40.8</td><td>36</td></tr><tr><td>Latent Memory (k=2)</td><td>22.7</td><td>34.6</td><td>35</td><td>66.4</td><td>73.0</td><td>44</td><td>4.5</td><td>12.2</td><td>34</td><td>52.5</td><td>52.5</td><td>68</td><td>36.5</td><td>43.1</td><td>45</td></tr><tr><td>Latent Memory (k=5)</td><td>23.4</td><td>35.0</td><td>62</td><td>65.1</td><td>71.7</td><td>70</td><td>6.0</td><td>12.3</td><td>61</td><td>55.9</td><td>55.9</td><td>94</td><td>37.6</td><td>43.7</td><td>72</td></tr><tr><td>8-token Latent Memory (k=1)</td><td>25.5</td><td>37.2</td><td>33</td><td>68.9</td><td>75.4</td><td>42</td><td>5.0</td><td>14.5</td><td>32</td><td>52.8</td><td>52.8</td><td>66</td><td>38.1</td><td>45.0</td><td>43</td></tr><tr><td>8-token Latent Memory (k=2)</td><td>24.9</td><td>36.8</td><td>49</td><td>68.1</td><td>74.8</td><td>58</td><td>5.0</td><td>14.3</td><td>48</td><td>57.0</td><td>57.0</td><td>82</td><td>38.7</td><td>45.7</td><td>59</td></tr><tr><td>8-token Latent Memory (k=5)</td><td>26.7</td><td>38.5</td><td>97</td><td>67.7</td><td>74.4</td><td>106</td><td>5.0</td><td>13.2</td><td>96</td><td>60.3</td><td>60.3</td><td>129</td><td>39.9</td><td>46.6</td><td>107</td></tr></table>

On these datasets, the RAG method based on Qwen-3-Embedding performs best, but we note that the 1-token and 8-token versions of Latent Memory exhibit good trade-offs. The 8-token Latent Memory method achieves similar performance to BM25 while requiring 2.5 times fewer tokens.

Multimodal generalization. Table 12 further evaluates multimodal generalization on a 20-image dataset SlideQA (Consider getting detailed answers from 1-2 correct slides.) with the same frozen LLaVA-1.5-13B generator. Latent Memory improves retrieval coverage and competitive EM value with far fewer generator tokens, but Nemo remains stronger on EM/F1.

Table 12: Generalization ability on the multimodal SlideQA domain under a frozen LLaVA-1.5-13B generator. Bold marks the best result in each metric column, and underlining marks the second best one. #Tok reports the generator-side input budget.

<table><tr><td colspan="5">Dataset: SlideQA Generation VLM (fixed): LLaVA-1.5-13B</td></tr><tr><td>Method</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td></tr><tr><td>Full-context baseline</td><td></td><td></td><td></td><td></td></tr><tr><td>Full Context</td><td>0.0</td><td>0.0</td><td>-</td><td>11871</td></tr><tr><td colspan="5">Raw-evidence multimodal retrieval</td></tr><tr><td>Nemo (k=1)</td><td>17.1</td><td>26.2</td><td>7.2</td><td>621</td></tr><tr><td>Nemo (k=2)</td><td>8.3</td><td>17.7</td><td>14.1</td><td>1212</td></tr><tr><td>Nemo (k=5)</td><td>4.7</td><td>13.2</td><td>30.5</td><td>2887</td></tr><tr><td colspan="5">Latent Memory</td></tr><tr><td>Latent Memory (k=1)</td><td>11.1</td><td>15.7</td><td>21.5</td><td>37</td></tr><tr><td>Latent Memory (k=2)</td><td>7.0</td><td>12.4</td><td>33.6</td><td>47</td></tr><tr><td>Latent Memory (k=5)</td><td>5.5</td><td>10.3</td><td>56.0</td><td>77</td></tr></table>

## C.3 Core Training Ablation

Table 13 ablates the reconstruction and distillation-negative designs used to train the text-only 1B compressor. The retrieval and generation settings are fixed; only the training targets or distillation context augmentation change. The variants are:

• Ours-1B: Full model that reconstructs both positive evidence and sampled negative evidence, with query reconstruction disabled.  
• w/o Constructing: Evidence reconstruction is removed entirely, so neither positive nor negative evidence is reconstructed.  
• w/o Constructing Negative Examples: Only positive evidence is reconstructed; sampled negative evidence is not used as a reconstruction target.  
• w/ Query Reconstruction: Adds query reconstruction to the default evidence-reconstruction setup.  
• w/ Distillation Negative Augmentation: During distillation, the student latent context is augmented with 0-3 randomly sampled irrelevant negative memories, while the teacher still sees only positive.

Table 13: Generator-side reconstruction ablation on text QA. All variants use the same frozen Meta-Llama-3-8B-Instruct generator and matched latent-token budget.

<table><tr><td rowspan="2">Variant</td><td colspan="3">HotpotQA</td><td colspan="3">2WikiMultihopQA</td><td colspan="3">MuSiQue</td><td colspan="3">Average</td></tr><tr><td>EM</td><td>F1</td><td>R@k</td><td>EM</td><td>F1</td><td>R@k</td><td>EM</td><td>F1</td><td>R@k</td><td>EM</td><td>F1</td><td>R@k</td></tr><tr><td colspan="13">Default model: reconstruct positive and negative evidence</td></tr><tr><td>Ours-1B (k=1)</td><td>27.4</td><td>39.4</td><td>34.6</td><td>19.8</td><td>29.2</td><td>28.4</td><td>5.8</td><td>14.6</td><td>8.7</td><td>17.7</td><td>27.7</td><td>23.9</td></tr><tr><td>Ours-1B (k=2)</td><td>31.6</td><td>45.2</td><td>62.6</td><td>21.5</td><td>33.5</td><td>49.5</td><td>7.3</td><td>16.9</td><td>15.5</td><td>20.1</td><td>31.9</td><td>42.6</td></tr><tr><td>Ours-1B (k=5)</td><td>34.8</td><td>48.9</td><td>87.1</td><td>24.3</td><td>36.7</td><td>74.2</td><td>8.7</td><td>19.2</td><td>30.1</td><td>22.6</td><td>34.9</td><td>63.8</td></tr><tr><td colspan="13">Reconstruction Loss</td></tr><tr><td>w/o Constructing (k=1)</td><td>23.4</td><td>37.8</td><td>33.4</td><td>20.3</td><td>28.8</td><td>29.8</td><td>5.9</td><td>15.1</td><td>8.3</td><td>16.5</td><td>27.2</td><td>23.8</td></tr><tr><td>w/o Constructing (k=2)</td><td>27.6</td><td>42.7</td><td>62.7</td><td>24.2</td><td>32.4</td><td>48.4</td><td>7.9</td><td>17.8</td><td>14.4</td><td>19.9</td><td>31.0</td><td>41.8</td></tr><tr><td>w/o Constructing (k=5)</td><td>30.6</td><td>45.2</td><td>85.2</td><td>22.7</td><td>33.8</td><td>73.3</td><td>9.4</td><td>19.2</td><td>30.4</td><td>20.9</td><td>32.7</td><td>62.9</td></tr><tr><td>w/o Constructing Negative (k=1)</td><td>24.3</td><td>39.3</td><td>34.1</td><td>19.0</td><td>30.1</td><td>28.0</td><td>5.7</td><td>14.8</td><td>7.6</td><td>16.3</td><td>28.1</td><td>23.2</td></tr><tr><td>w/o Constructing Negative (k=2)</td><td>29.1</td><td>44.9</td><td>58.0</td><td>19.8</td><td>33.9</td><td>47.0</td><td>6.3</td><td>17.2</td><td>13.8</td><td>18.4</td><td>32.0</td><td>39.6</td></tr><tr><td>w/o Constructing Negative (k=5)</td><td>31.7</td><td>46.2</td><td>86.2</td><td>22.3</td><td>36.1</td><td>72.2</td><td>8.3</td><td>18.9</td><td>28.7</td><td>20.8</td><td>33.8</td><td>62.3</td></tr><tr><td>w/ Query Reconstruction (k=1)</td><td>22.3</td><td>35.9</td><td>34.2</td><td>19.3</td><td>28.4</td><td>25.8</td><td>4.8</td><td>13.6</td><td>7.8</td><td>15.5</td><td>26.0</td><td>22.6</td></tr><tr><td>w/ Query Reconstruction (k=2)</td><td>26.5</td><td>41.7</td><td>60.7</td><td>22.6</td><td>31.8</td><td>45.4</td><td>6.3</td><td>15.4</td><td>13.7</td><td>18.5</td><td>29.6</td><td>39.9</td></tr><tr><td>w/ Query Reconstruction (k=5)</td><td>30.1</td><td>45.7</td><td>84.9</td><td>26.3</td><td>35.4</td><td>72.6</td><td>7.7</td><td>18.0</td><td>26.3</td><td>21.3</td><td>33.0</td><td>61.3</td></tr><tr><td colspan="13">Distillation Loss</td></tr><tr><td>w/o Negative Augmentation (k=1)</td><td>26.2</td><td>38.2</td><td>33.4</td><td>21.0</td><td>30.9</td><td>27.6</td><td>5.5</td><td>14.6</td><td>8.1</td><td>17.6</td><td>27.9</td><td>23.0</td></tr><tr><td>w/o Negative Augmentation (k=2)</td><td>31.3</td><td>44.4</td><td>59.0</td><td>22.8</td><td>34.8</td><td>48.4</td><td>6.6</td><td>16.2</td><td>13.8</td><td>20.2</td><td>31.8</td><td>40.4</td></tr><tr><td>w/o Negative Augmentation (k=5)</td><td>31.6</td><td>44.8</td><td>83.0</td><td>24.3</td><td>35.8</td><td>73.6</td><td>7.2</td><td>17.5</td><td>25.5</td><td>21.0</td><td>32.7</td><td>60.7</td></tr></table>

## Analysis.

1. Ours-1B is the strongest default overall, but the main takeaway is that the reconstruction objective stabilizes both retrieval and generation.  
2. Removing evidence reconstruction entirely (w/o Constructing) consistently hurts EM/F1 and also lowers Recall@k on average. This joint degradation supports the view that the same latent token is serving retrieval and generation in a unified representation space.  
3. Only reconstructing positive evidence (w/o Constructing Negative Examples) is less stable and degrades retrieval more clearly. This supports the role of negative evidence as a geometric anchor for the latent memory space.  
4. Adding query reconstruction (w/ Query Reconstruction) is harmful in this setting. It forces the query adapter to preserve surface reconstruction information, which conflicts with its role as a retrieval query representation.  
5. Adding 0–3 irrelevant negative memories during distillation (w/ Distillation Negative Augmentation) does not recover the full model, especially when k = 5.

Across the datasets, removing reconstruction weakens both retrieval and generation rather than trading one for the other. Negative evidence is useful when it is reconstructed as part of the memory representation, but query reconstruction and direct irrelevant-memory augmentation perturb the intended roles of the query encoder and generator-side conditioning target.

## C.4 Stronger Text Compressors

This section asks whether a larger or a different-family compressor improves one-token Latent Memory’s quality. The generator (frozen Meta-Llama-3-8B-Instruct), token budget, and training recipe are held fixed; only the compression backbone changes. Table 14 compares LLaMA-1B (the default), LLaMA-3B, and Qwen-1.5B, averaged over HotpotQA, 2WikiMultihopQA, and MuSiQue.

Table 14: Text-compressor ablation on text QA. The frozen generator and training recipe are fixed; only the compression backbone changes.

<table><tr><td rowspan="2">Encoder</td><td colspan="3">HotpotQA</td><td colspan="3">2WikiMultihopQA</td><td colspan="3">MuSiQue</td><td colspan="3">Average</td></tr><tr><td>EM</td><td>F1</td><td>R@k</td><td>EM</td><td>F1</td><td>R@k</td><td>EM</td><td>F1</td><td>R@k</td><td>EM</td><td>F1</td><td>R@k</td></tr><tr><td colspan="13">LLaMA-family compressors</td></tr><tr><td>LLaMA-1B (k=1)</td><td>27.4</td><td>39.4</td><td>34.6</td><td>19.8</td><td>29.2</td><td>28.4</td><td>5.8</td><td>14.6</td><td>8.7</td><td>17.7</td><td>27.7</td><td>23.9</td></tr><tr><td>LLaMA-1B (k=2)</td><td>31.6</td><td>45.2</td><td>62.6</td><td>21.5</td><td>33.5</td><td>49.5</td><td>7.3</td><td>16.9</td><td>15.5</td><td>20.1</td><td>31.9</td><td>42.6</td></tr><tr><td>LLaMA-1B (k=5)</td><td>34.8</td><td>48.9</td><td>87.1</td><td>24.3</td><td>36.7</td><td>74.2</td><td>8.7</td><td>19.2</td><td>30.1</td><td>22.6</td><td>34.9</td><td>63.8</td></tr><tr><td>LLaMA-3B (k=1)</td><td>29.6</td><td>42.5</td><td>35.7</td><td>20.1</td><td>30.5</td><td>29.1</td><td>6.3</td><td>15.8</td><td>9.1</td><td>18.7</td><td>29.6</td><td>24.6</td></tr><tr><td>LLaMA-3B (k=2)</td><td>34.4</td><td>48.9</td><td>64.2</td><td>23.0</td><td>35.9</td><td>51.9</td><td>8.9</td><td>19.4</td><td>16.0</td><td>22.1</td><td>34.7</td><td>44.0</td></tr><tr><td>LLaMA-3B (k=5)</td><td>35.4</td><td>49.7</td><td>88.6</td><td>25.1</td><td>37.3</td><td>77.6</td><td>9.9</td><td>19.9</td><td>30.1</td><td>23.5</td><td>35.6</td><td>65.4</td></tr><tr><td colspan="13">Qwen-family compressor</td></tr><tr><td>Qwen-1.5B (k=1)</td><td>25.5</td><td>37.2</td><td>34.8</td><td>20.6</td><td>30.3</td><td>30.0</td><td>5.0</td><td>13.2</td><td>8.8</td><td>17.0</td><td>26.9</td><td>24.5</td></tr><tr><td>Qwen-1.5B (k=2)</td><td>29.2</td><td>41.8</td><td>62.1</td><td>24.0</td><td>35.4</td><td>52.1</td><td>6.0</td><td>14.6</td><td>15.5</td><td>19.8</td><td>30.6</td><td>43.2</td></tr><tr><td>Qwen-1.5B (k=5)</td><td>30.7</td><td>43.4</td><td>86.4</td><td>27.4</td><td>37.4</td><td>76.5</td><td>6.2</td><td>16.3</td><td>30.1</td><td>21.4</td><td>32.3</td><td>64.3</td></tr></table>

## Analysis.

1. LLaMA-3B is consistently the strongest encoder in this comparison, improving both EM/F1 over LLaMA-1B and Qwen-1.5B at all reported k values.  
2. The gap is more pronounced in EM/F1 than in Recall@k, which suggests that encoder choice influences downstream answer quality more strongly than retrieval coverage alone.  
3. Since the token budget is matched across all three variants, the advantage of LLaMA-3B is best interpreted as a representational benefit rather than an efficiency effect.

The dataset-level breakdown further shows that the stronger compressor helps both in-domain and OOD evaluation. On HotpotQA, LLaMA-3B improves the default LLaMA-1B model from 34.8/48.9 EM/F1 to 35.4/49.7 at k=5, while maintaining similar high Recall@k. On 2WikiMultihopQA, the improvement is more visible: LLaMA-3B reaches 25.1 EM and 37.3 F1 at k=5, compared with 24.3/36.7 for LLaMA-1B, and also improves Recall@k from 74.2 to 77.6. MuSiQue remains the most difficult OOD dataset; LLaMA-3B improves EM/F1 modestly, but Recall@k is nearly unchanged at k=5, indicating that the encoder upgrade mainly improves the quality of the latent representation consumed by the generator rather than solving all retrieval coverage limitations. Qwen-1.5B obtains competitive Recall@k on 2Wiki and MuSiQue, but its lower EM/F1 suggests that high retrieval scores do not necessarily imply better latent-conditioned generation.

## C.5 Direct Transfer to Similar Generator

This experiment studies whether Latent Memory trained with the Meta-Llama-3-8B-Instruct generator can be directly reused by another similar LLaMA generator. We keep the compressor, latent memory bank, retrieval procedure, and projection interface fixed, and replace only the frozen answer generator with LLaMA-3.1-8B-Instruct. No additional compressor training or latent-token adaptation is performed. Table 15 reports the full text-QA results under this transferred generator.

Table 15: Direct generator transfer on text QA. Latent tokens are trained under Meta-Llama-3-8B-Instruct and evaluated directly with a frozen LLaMA-3.1-8B-Instruct generator. #Tok reports the generator-side input budget.

<table><tr><td colspan="17">Generation LLM (fixed): LLaMA-3.1-8B-Instruct</td></tr><tr><td rowspan="2">Method</td><td colspan="4">HotpotQA</td><td colspan="4">2WikiMultihopQA</td><td colspan="4">MuSiQue</td><td colspan="4">Average</td></tr><tr><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td></tr><tr><td>Full-context reference</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full Context</td><td>50.2</td><td>65.9</td><td>-</td><td>1462</td><td>35.2</td><td>48.7</td><td>-</td><td>1074</td><td>18.8</td><td>30.7</td><td>-</td><td>2580</td><td>34.7</td><td>48.4</td><td>-</td><td>1706</td></tr><tr><td>Sparse raw-evidence RAG baseline</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>BM25 Retrieval (k=1)</td><td>29.6</td><td>43.0</td><td>30.2</td><td>68</td><td>13.5</td><td>23.2</td><td>18.3</td><td>60</td><td>6.1</td><td>13.3</td><td>7.0</td><td>69</td><td>16.4</td><td>26.5</td><td>18.5</td><td>66</td></tr><tr><td>BM25 Retrieval (k=2)</td><td>34.6</td><td>48.4</td><td>45.5</td><td>106</td><td>16.7</td><td>25.4</td><td>29.2</td><td>94</td><td>7.8</td><td>15.2</td><td>12.2</td><td>106</td><td>19.7</td><td>29.7</td><td>29.0</td><td>102</td></tr><tr><td>BM25 Retrieval (k=5)</td><td>41.2</td><td>55.4</td><td>65.5</td><td>224</td><td>24.4</td><td>32.0</td><td>46.9</td><td>196</td><td>10.6</td><td>19.2</td><td>22.4</td><td>221</td><td>25.4</td><td>35.5</td><td>44.9</td><td>214</td></tr><tr><td>Dense raw-evidence RAG baselines</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Dense Retrieval (k=1)</td><td>26.5</td><td>39.3</td><td>27.9</td><td>67</td><td>15.0</td><td>25.5</td><td>23.3</td><td>61</td><td>6.6</td><td>15.1</td><td>9.6</td><td>70</td><td>16.0</td><td>26.6</td><td>20.3</td><td>66</td></tr><tr><td>Dense Retrieval (k=2)</td><td>30.6</td><td>44.1</td><td>42.2</td><td>104</td><td>18.9</td><td>29.0</td><td>37.6</td><td>95</td><td>8.3</td><td>17.5</td><td>17.0</td><td>106</td><td>19.3</td><td>30.2</td><td>32.3</td><td>102</td></tr><tr><td>Dense Retrieval (k=5)</td><td>36.9</td><td>50.9</td><td>62.4</td><td>214</td><td>27.8</td><td>37.0</td><td>58.0</td><td>198</td><td>12.6</td><td>22.7</td><td>31.3</td><td>218</td><td>25.7</td><td>36.8</td><td>50.6</td><td>210</td></tr><tr><td>Qwen3-Emb (k=1)</td><td>30.9</td><td>43.6</td><td>33.1</td><td>70</td><td>18.1</td><td>30.9</td><td>32.5</td><td>66</td><td>8.1</td><td>18.2</td><td>10.3</td><td>73</td><td>19.0</td><td>30.9</td><td>25.3</td><td>70</td></tr><tr><td>Qwen3-Emb (k=2)</td><td>35.6</td><td>49.0</td><td>50.0</td><td>109</td><td>17.4</td><td>33.5</td><td>47.7</td><td>102</td><td>9.8</td><td>20.4</td><td>18.5</td><td>112</td><td>21.0</td><td>34.3</td><td>38.8</td><td>108</td></tr><tr><td>Qwen3-Emb (k=5)</td><td>40.0</td><td>54.5</td><td>70.1</td><td>224</td><td>19.1</td><td>38.6</td><td>64.3</td><td>208</td><td>13.7</td><td>24.7</td><td>34.8</td><td>230</td><td>24.3</td><td>39.3</td><td>56.4</td><td>221</td></tr><tr><td>Direct transfer of Latent Memory</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>1-token Latent Memory (k=1)</td><td>23.7</td><td>36.8</td><td>34.6</td><td>36</td><td>21.4</td><td>29.0</td><td>28.3</td><td>33</td><td>5.1</td><td>13.3</td><td>8.7</td><td>37</td><td>16.7</td><td>26.4</td><td>23.9</td><td>36</td></tr><tr><td>1-token Latent Memory (k=2)</td><td>29.6</td><td>43.6</td><td>62.7</td><td>45</td><td>26.9</td><td>34.9</td><td>49.5</td><td>42</td><td>6.2</td><td>15.7</td><td>15.5</td><td>46</td><td>20.9</td><td>31.4</td><td>42.6</td><td>45</td></tr><tr><td>1-token Latent Memory (k=5)</td><td>34.8</td><td>48.3</td><td>87.0</td><td>72</td><td>33.0</td><td>39.9</td><td>74.2</td><td>69</td><td>8.1</td><td>18.4</td><td>30.1</td><td>73</td><td>25.3</td><td>35.5</td><td>63.8</td><td>72</td></tr><tr><td>2-token Latent Memory (k=1)</td><td>24.5</td><td>36.8</td><td>34.6</td><td>37</td><td>22.6</td><td>30.0</td><td>28.0</td><td>34</td><td>5.2</td><td>13.2</td><td>8.7</td><td>38</td><td>17.4</td><td>26.7</td><td>23.7</td><td>36</td></tr><tr><td>2-token Latent Memory (k=2)</td><td>29.9</td><td>43.6</td><td>62.5</td><td>47</td><td>27.3</td><td>34.9</td><td>49.1</td><td>44</td><td>7.4</td><td>16.9</td><td>15.3</td><td>48</td><td>21.5</td><td>31.8</td><td>42.3</td><td>46</td></tr><tr><td>2-token Latent Memory (k=5)</td><td>34.6</td><td>47.8</td><td>87.0</td><td>77</td><td>31.0</td><td>38.6</td><td>73.8</td><td>74</td><td>9.3</td><td>19.9</td><td>29.4</td><td>78</td><td>25.0</td><td>35.5</td><td>63.4</td><td>76</td></tr><tr><td>4-token Latent Memory (k=1)</td><td>25.5</td><td>38.3</td><td>35.2</td><td>39</td><td>23.4</td><td>30.8</td><td>30.9</td><td>36</td><td>5.4</td><td>14.0</td><td>9.2</td><td>40</td><td>18.1</td><td>27.7</td><td>25.1</td><td>38</td></tr><tr><td>4-token Latent Memory (k=2)</td><td>32.7</td><td>45.8</td><td>63.2</td><td>51</td><td>30.2</td><td>36.9</td><td>53.5</td><td>48</td><td>8.1</td><td>18.1</td><td>16.2</td><td>52</td><td>23.7</td><td>33.6</td><td>44.3</td><td>50</td></tr><tr><td>4-token Latent Memory (k=5)</td><td>36.4</td><td>49.4</td><td>88.0</td><td>87</td><td>33.7</td><td>40.0</td><td>76.7</td><td>84</td><td>9.3</td><td>19.6</td><td>31.0</td><td>88</td><td>26.5</td><td>36.3</td><td>65.2</td><td>86</td></tr><tr><td>8-token Latent Memory (k=1)</td><td>29.4</td><td>43.1</td><td>35.1</td><td>80</td><td>24.2</td><td>32.4</td><td>30.5</td><td>97</td><td>7.4</td><td>16.4</td><td>9.2</td><td>101</td><td>20.3</td><td>30.6</td><td>24.9</td><td>93</td></tr><tr><td>8-token Latent Memory (k=2)</td><td>38.2</td><td>52.7</td><td>63.4</td><td>96</td><td>29.5</td><td>38.1</td><td>52.7</td><td>113</td><td>11.1</td><td>21.1</td><td>16.3</td><td>117</td><td>26.2</td><td>37.3</td><td>44.1</td><td>109</td></tr><tr><td>8-token Latent Memory (k=5)</td><td>44.2</td><td>58.7</td><td>88.0</td><td>144</td><td>33.5</td><td>42.0</td><td>76.2</td><td>161</td><td>14.3</td><td>24.7</td><td>31.1</td><td>165</td><td>30.7</td><td>41.8</td><td>65.1</td><td>157</td></tr></table>

Analysis. The transferred latent tokens remain usable with the new generator without re-training. The one-token variant preserves the same token-efficiency pattern as in the main experiments, while larger latent-token budgets provide a clear capacity gain. In particular, the 8-token variant improves the average EM/F1 to 30.7/41.8 at k=5, exceeding the strongest raw-evidence embedding baseline at the same k while still using fewer generator tokens. This suggests that Latent Memory is not tightly bound to a single frozen LLaMA generator: once the latent tokens learn to act as compact evidence, another compatible LLaMA instruction model can consume them directly.

## C.6 Multimodal Results with Gemma

Table 16 reports WebQA results when the generator is switched to a frozen Gemma-3-12B-Instruct. This model-swap setting keeps the retrieval pool and evidence candidates fixed, and compares four groups: full-context prompting, text-only raw-evidence retrievers, multimodal raw-evidence retrievers, and direct generation from Latent Memory. We fine-tune Gemma-3-4B-PT as the compressor and use LLaMA-3.2-1B-Instruct as the reconstruction decoder.

Table 16: Multimodal QA (WebQA) with unified retrieval and generation using a frozen Gemma-3- 12B-Instruct generator.

<table><tr><td colspan="13">Generation VLM (fixed): Gemma-3-12B-Instruct</td></tr><tr><td rowspan="2">Method</td><td colspan="4">WebQA-Image</td><td colspan="4">WebQA-Text</td><td colspan="4">Avg</td></tr><tr><td>F1</td><td>Acc</td><td>R@k</td><td>#Tok</td><td>F1</td><td>Acc</td><td>R@k</td><td>#Tok</td><td>F1</td><td>Acc</td><td>R@k</td><td>#Tok</td></tr><tr><td>Full-context reference</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full Context</td><td>13.8</td><td>18.0</td><td>-</td><td>5856</td><td>54.9</td><td>54.9</td><td>-</td><td>4337</td><td>34.3</td><td>36.4</td><td>-</td><td>5097</td></tr><tr><td colspan="13">Raw-evidence text retrieval baselines</td></tr><tr><td>BM25 Retrieval (k=1)</td><td>8.2</td><td>9.2</td><td>21.8</td><td>167</td><td>38.6</td><td>40.4</td><td>31.8</td><td>106</td><td>23.4</td><td>24.8</td><td>26.8</td><td>137</td></tr><tr><td>BM25 Retrieval (k=2)</td><td>8.3</td><td>9.5</td><td>28.3</td><td>274</td><td>43.6</td><td>44.8</td><td>51.0</td><td>183</td><td>26.0</td><td>27.2</td><td>39.6</td><td>228</td></tr><tr><td>BM25 Retrieval (k=5)</td><td>8.8</td><td>10.5</td><td>37.9</td><td>559</td><td>49.5</td><td>50.3</td><td>73.0</td><td>415</td><td>29.2</td><td>30.4</td><td>55.5</td><td>487</td></tr><tr><td>Dense Retrieval (k=1)</td><td>10.6</td><td>15.8</td><td>39.7</td><td>241</td><td>36.2</td><td>37.0</td><td>25.5</td><td>158</td><td>23.4</td><td>26.4</td><td>32.6</td><td>200</td></tr><tr><td>Dense Retrieval (k=2)</td><td>10.3</td><td>14.6</td><td>53.1</td><td>411</td><td>39.0</td><td>40.0</td><td>40.1</td><td>303</td><td>24.6</td><td>27.3</td><td>46.6</td><td>357</td></tr><tr><td>Dense Retrieval (k=5)</td><td>10.6</td><td>14.7</td><td>67.1</td><td>848</td><td>45.2</td><td>45.9</td><td>62.0</td><td>750</td><td>27.9</td><td>30.2</td><td>64.5</td><td>799</td></tr><tr><td colspan="13">Raw-evidence multimodal retrieval baselines</td></tr><tr><td>Qwen3-VL-8B (k=1)</td><td>8.6</td><td>11.6</td><td>24.0</td><td>164</td><td>40.7</td><td>42.5</td><td>40.8</td><td>108</td><td>24.7</td><td>27.1</td><td>32.4</td><td>136</td></tr><tr><td>Qwen3-VL-8B (k=2)</td><td>9.1</td><td>11.1</td><td>32.6</td><td>270</td><td>47.6</td><td>48.7</td><td>66.3</td><td>187</td><td>28.4</td><td>29.9</td><td>49.5</td><td>228</td></tr><tr><td>Qwen3-VL-8B (k=5)</td><td>9.5</td><td>12.4</td><td>49.6</td><td>570</td><td>52.3</td><td>53.5</td><td>87.5</td><td>442</td><td>30.9</td><td>32.9</td><td>68.5</td><td>506</td></tr><tr><td>Nemo-Emb (k=1)</td><td>11.3</td><td>17.8</td><td>48.8</td><td>255</td><td>41.5</td><td>43.3</td><td>40.5</td><td>107</td><td>26.4</td><td>30.5</td><td>44.6</td><td>181</td></tr><tr><td>Nemo-Emb (k=2)</td><td>11.1</td><td>17.0</td><td>64.7</td><td>444</td><td>48.9</td><td>50.1</td><td>66.6</td><td>186</td><td>30.0</td><td>33.5</td><td>65.7</td><td>315</td></tr><tr><td>Nemo-Emb (k=5)</td><td>11.2</td><td>17.3</td><td>81.5</td><td>948</td><td>52.1</td><td>53.0</td><td>87.1</td><td>450</td><td>31.7</td><td>35.1</td><td>84.3</td><td>698</td></tr><tr><td colspan="13">Ours: direct generation from Latent Memory</td></tr><tr><td>Latent Memory (k=1)</td><td>11.7</td><td>18.7</td><td>52.5</td><td>38</td><td>31.6</td><td>31.9</td><td>23.7</td><td>39</td><td>21.6</td><td>25.3</td><td>38.1</td><td>38</td></tr><tr><td>Latent Memory (k=2)</td><td>11.3</td><td>18.0</td><td>70.1</td><td>47</td><td>31.9</td><td>32.3</td><td>41.3</td><td>48</td><td>21.6</td><td>25.1</td><td>55.7</td><td>48</td></tr><tr><td>Latent Memory (k=5)</td><td>11.3</td><td>17.5</td><td>89.0</td><td>74</td><td>31.3</td><td>31.7</td><td>70.5</td><td>75</td><td>21.3</td><td>24.6</td><td>79.8</td><td>74</td></tr></table>

Analysis. With Gemma-3-12B-Instruct, Latent Memory remains the most token-efficient option. Demonstrating the best WebQA-Image Accuracy and the second best F1 with 10× less tokens.

## C.7 Latent Tokens as Retrievers

Latent Memory’s latent token plays two roles at once: it is a retrieval key used to rank evidence and a generation input consumed by the frozen generator. This section isolates these two roles with a hybrid as-RAG mode. In this mode, latent tokens are used only for retrieval; after the top-k evidence items are selected, generation is performed from the retrieved raw text or images rather than from latent tokens. We report text-only and multimodal variants separately because the raw evidence format differs across settings.

• Ours-1B: full Latent Memory—latent tokens used for both retrieval and generation.  
• Ours-1B-RAG: latent tokens used for retrieval only; generator sees raw retrieved text sentences.  
• Latent-Token-based RAG: multimodal variant of the above—latent tokens rank the unified text-image pool; generator sees raw retrieved text and images.

Comparing these variants reveals how much of Latent Memory’s gain comes from better retrieval (latent key quality) versus better generation (latent token quality as generator input).

## Analysis.

1. In the plain text setting, the "Latent Memory as RAG" method based on raw evidence effectively isolates retrieval and compression. We observed that with Latent Memory k=1, it outperforms Latent-Memory-as-RAG (k=5) using the same number of tokens. This demonstrates that even disregarding the advantage of retrieval, the compression portion of Latent Memory still exhibits a trade-off, and the 8-token Latent Memory results show a better trade-off.

Table 17: Text-only-as-RAG results. Latent retrieval selects raw text evidence, and generation is performed from the retrieved raw text. #Tok excludes fixed prompt scaffolding.

<table><tr><td rowspan="2">Methods</td><td colspan="2">HotpotQA</td><td colspan="2">2WikiMultihopQA</td><td colspan="2">MuSiQue</td><td colspan="4">Avg</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>EM</td><td>F1</td><td>R@k</td><td>#Tok</td></tr><tr><td>Full-context reference</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Full Context</td><td>42.0</td><td>57.8</td><td>17.7</td><td>39.2</td><td>6.0</td><td>17.1</td><td>21.9</td><td>38.0</td><td>-</td><td>1706</td></tr><tr><td>Raw-evidence RAG baseline</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-Emb (k=1)</td><td>30.9</td><td>43.6</td><td>18.1</td><td>30.9</td><td>8.1</td><td>18.2</td><td>19.0</td><td>30.9</td><td>25.3</td><td>70</td></tr><tr><td>Qwen3-Emb (k=2)</td><td>35.6</td><td>49.0</td><td>17.4</td><td>33.5</td><td>9.8</td><td>20.4</td><td>21.0</td><td>34.3</td><td>38.8</td><td>108</td></tr><tr><td>Qwen3-Emb (k=5)</td><td>40.0</td><td>54.5</td><td>19.1</td><td>38.6</td><td>13.7</td><td>24.6</td><td>24.3</td><td>39.2</td><td>56.4</td><td>221</td></tr><tr><td>Direct generation from Latent Memory</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Latent Memory (k=1)</td><td>27.4</td><td>39.4</td><td>19.8</td><td>29.2</td><td>5.8</td><td>14.6</td><td>17.7</td><td>27.7</td><td>23.9</td><td>36</td></tr><tr><td>Latent Memory (k=2)</td><td>31.6</td><td>45.2</td><td>21.5</td><td>33.5</td><td>7.3</td><td>16.9</td><td>20.1</td><td>31.9</td><td>42.6</td><td>45</td></tr><tr><td>Latent Memory (k=5)</td><td>34.8</td><td>48.9</td><td>24.3</td><td>36.7</td><td>8.7</td><td>19.2</td><td>22.6</td><td>34.9</td><td>63.8</td><td>72</td></tr><tr><td>Latent retrieval as raw-evidence RAG</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Latent Memory-as-RAG (k=1)</td><td>37.0</td><td>49.8</td><td>18.6</td><td>30.7</td><td>10.6</td><td>19.9</td><td>22.1</td><td>33.5</td><td>23.9</td><td>75</td></tr><tr><td>Latent Memory-as-RAG (k=2)</td><td>44.0</td><td>58.9</td><td>18.4</td><td>35.4</td><td>13.2</td><td>22.8</td><td>25.2</td><td>39.0</td><td>42.6</td><td>125</td></tr><tr><td>Latent Memory-as-RAG (k=5)</td><td>49.9</td><td>66.2</td><td>21.9</td><td>42.8</td><td>18.0</td><td>30.2</td><td>29.9</td><td>46.4</td><td>63.8</td><td>259</td></tr></table>

Table 18: Multimodal-as-RAG results. Latent retrieval ranks the unified text-image pool, but generation is performed from retrieved raw text and raw images.

<table><tr><td rowspan="2">Method</td><td colspan="4">WebQA-Image</td><td colspan="4">WebQA-Text</td><td colspan="4">Avg</td></tr><tr><td>F1</td><td>Acc</td><td>R@k</td><td>#Tok</td><td>F1</td><td>Acc</td><td>R@k</td><td>#Tok</td><td>F1</td><td>Acc</td><td>R@k</td><td>#Tok</td></tr><tr><td colspan="13">Raw-evidence multimodal retrieval baselines</td></tr><tr><td>Qwen3-VL-8B (k=1)</td><td>15.1</td><td>22.0</td><td>24.1</td><td>284</td><td>40.8</td><td>43.5</td><td>40.7</td><td>131</td><td>27.9</td><td>32.8</td><td>32.4</td><td>208</td></tr><tr><td>Qwen3-VL-8B (k=2)</td><td>20.1</td><td>24.5</td><td>32.9</td><td>465</td><td>46.1</td><td>50.2</td><td>66.3</td><td>235</td><td>33.1</td><td>37.3</td><td>49.6</td><td>350</td></tr><tr><td>Qwen3-VL-8B (k=5)</td><td>34.1</td><td>30.3</td><td>49.7</td><td>957</td><td>47.8</td><td>57.2</td><td>87.2</td><td>612</td><td>41.0</td><td>43.8</td><td>68.5</td><td>784</td></tr><tr><td>Nemo-Emb (k=1)</td><td>19.9</td><td>25.2</td><td>48.7</td><td>508</td><td>41.1</td><td>44.0</td><td>40.4</td><td>130</td><td>30.5</td><td>34.6</td><td>44.6</td><td>319</td></tr><tr><td>Nemo-Emb (k=2)</td><td>32.9</td><td>30.6</td><td>64.8</td><td>892</td><td>46.8</td><td>50.9</td><td>66.6</td><td>233</td><td>39.9</td><td>40.8</td><td>65.7</td><td>563</td></tr><tr><td>Nemo-Emb (k=5)</td><td>53.0</td><td>37.2</td><td>81.4</td><td>1885</td><td>48.6</td><td>57.9</td><td>87.1</td><td>629</td><td>50.8</td><td>47.6</td><td>84.2</td><td>1257</td></tr><tr><td colspan="13">Direct generation from Latent Memory</td></tr><tr><td>Latent Memory (k=1)</td><td>32.0</td><td>28.7</td><td>56.6</td><td>42</td><td>28.8</td><td>30.8</td><td>24.0</td><td>44</td><td>30.4</td><td>29.7</td><td>40.3</td><td>43</td></tr><tr><td>Latent Memory (k=2)</td><td>56.5</td><td>39.5</td><td>74.7</td><td>52</td><td>30.0</td><td>32.8</td><td>41.1</td><td>54</td><td>43.2</td><td>36.2</td><td>57.9</td><td>53</td></tr><tr><td>Latent Memory (k=5)</td><td>69.4</td><td>44.2</td><td>91.2</td><td>82</td><td>30.7</td><td>34.3</td><td>70.5</td><td>84</td><td>50.0</td><td>39.2</td><td>80.8</td><td>83</td></tr><tr><td colspan="13">Latent retrieval as raw-evidence multimodal RAG</td></tr><tr><td>Latent Memory-as-RAG (k=1)</td><td>21.4</td><td>25.8</td><td>56.6</td><td>638</td><td>34.9</td><td>37.6</td><td>24.0</td><td>122</td><td>28.2</td><td>31.7</td><td>40.3</td><td>380</td></tr><tr><td>Latent Memory-as-RAG (k=2)</td><td>38.3</td><td>33.3</td><td>74.7</td><td>1233</td><td>40.3</td><td>43.9</td><td>41.1</td><td>207</td><td>39.3</td><td>38.6</td><td>57.9</td><td>720</td></tr><tr><td>Latent Memory-as-RAG (k=5)</td><td>54.7</td><td>37.0</td><td>91.2</td><td>2970</td><td>45.1</td><td>51.2</td><td>70.5</td><td>461</td><td>49.9</td><td>44.1</td><td>80.8</td><td>1715</td></tr></table>

2. In multimodal WebQA, the conclusion is different. The "Latent Memory as RAG" method based on raw evidence outperforms in the text portion, but it lags behind using latent tokens in the image portion. This indicates that the performance improvement on the image side is related to obtaining a more efficient representation, which helps alleviate the context processing pressure on large models.  
3. Furthermore, the improvement in recall@k itself may provide inspiration for the training of subsequent embedding models. This is an embedding design based on providing raw context, demonstrating the effectiveness of unified retrieval and representation space.

## D Case Study

The case studies are not additional leaderboard evidence; they diagnose how the compact representation behaves on individual examples. We use them to answer three qualitative questions:

• RQ-C1: What does reconstruction reveal about the latent token? Appendix D.1 checks whether reconstruction preserves key evidence rather than copying the full sentence verbatim.  
• RQ-C2: Can text-only latent retrieval recover complete multi-hop chains? Appendix D.2 shows examples where the retrieved latent memories cover all required supporting facts.  
• RQ-C3: How does unified text-image retrieval behave on concrete multimodal questions? Appendix D.3 shows additional WebQA cases using the same unified text-image memory pool as the main experiments.

## D.1 Reconstruction Quality of Latent Tokens

![](images/de3a2bfef9689f358557764b2d2208ff255666ee64ea2b46e97579d9b17e47a0.jpg)

<details>
<summary>line chart</summary>

| Training step | 1 token | 2 tokens | 4 tokens | 8 tokens |
| ------------- | ------ | -------- | -------- | -------- |
| 0             | 1.6    | 1.5      | 1.4      | 1.8      |
| 5k            | 0.7    | 0.6      | 0.4      | 0.3      |
| 10k           | 0.5    | 0.4      | 0.2      | 0.15     |
| 15k           | 0.4    | 0.3      | 0.15     | 0.1      |
| 20k           | 0.35   | 0.25     | 0.1      | 0.08     |
| 25k           | 0.35   | 0.25     | 0.1      | 0.08     |
| 30k           | 0.35   | 0.25     | 0.1      | 0.08     |
| 35k           | 0.35   | 0.25     | 0.1      | 0.08     |
</details>

Figure 10: Training reconstruction loss for 1-, 2-, 4-, and 8-token Latent Memory variants. More latent tokens consistently reduce the reconstruction CE, indicating that additional latent capacity preserves more evidence information during compression.

Table 19: Qualitative reconstruction examples on HotpotQA. CE is measured against the original evidence sentence under teacher forcing.

<table><tr><td>Original evidence</td><td>1-token reconstruction</td><td>CE</td><td>8-token reconstruction</td><td>CE</td></tr><tr><td>Shirley Temple: Shirley Temple Black (April 23, 1928–February 10, 2014) was an American actress, singer, dancer, businesswoman, and diplomat who was Hollywood&#x27;s number one box-office draw as a child actress from 1935 to 1938.</td><td>actress greater greater greater greater ...</td><td>0.226</td><td>Shirley Temple (April 23, 1928–February 10, 2014) was an actress, singer, dancewoman, and business actress who was Hollywood&#x27;s number one child ...</td><td>0.062</td></tr><tr><td>2014 S/S: 2014 S/S is the debut album of South Korean group WINNER.</td><td>S:#:#:#:#:#:#:#:#...</td><td>0.486</td><td>2014 is the debut album of South Korean group WINNER ...</td><td>0.016</td></tr></table>

The table above provides examples for latent reconstruction. 1-tokens are okay for low CE but cannot fully reconstruct the evidence; improving the number of tokens per latent evidence is helpful for reconstruction as the reconstruction loss in Figure 10 will drop accordingly.

The ability to reconstruct the text not only provides a stronger and more faithful representation of the original text but also enhances the interpretability of latent memory.

## D.2 Text-only Case Studies

Table 20 presents two additional text-only cases. Both require multi-hop composition rather than simple lexical matching: the first links a screenwriter to a Nicolas Cage film through two supporting facts, while the second links a person to a company and then to the company’s headquarters. In both cases, the required evidence chain is ranked at the top, and the final answer is recovered from a compact retrieved set.

Table 20: Two text-only case studies. Both examples achieve exact-match generation, and the retrieved evidence chain is fully covered by the top-ranked latent retrieval results.

<table><tr><td>Case 1</td><td colspan="5">Question: What screenwriter with credits for “Evolution” co-wrote a film starring Nicolas Cage and Téa Leoni?</td></tr><tr><td>Generation</td><td colspan="5">Gold answer: David Weissman. Predicted answer: David Weissman. EM: 1.0 Retrieved positives: 3/3.</td></tr><tr><td rowspan="4">Latent Memory</td><td>Rank Pos? (gold)</td><td>Retrieval score</td><td>Corresponding text of retrieved latent token</td><td>CE loss</td><td></td></tr><tr><td>1 Y</td><td>0.507</td><td>David Weissman: His film credits include “The Family Man” (2000), “Evolution” (2001), and “When in Rome” (2010).</td><td>0.260</td><td></td></tr><tr><td>2 Y</td><td>0.447</td><td>The Family Man: The Family Man is a 2000 American romantic comedy-drama film ... star-ring Nicolas Cage and Téa Leoni.</td><td>0.089</td><td></td></tr><tr><td>3 Y</td><td>0.394</td><td>David Weissman: David Weissman is a screen-writer and director.</td><td>0.010</td><td></td></tr><tr><td>Case 2</td><td colspan="5">Question: Where is the company that Sachin Warrier worked for as a software engineer headquartered?</td></tr><tr><td>Generation</td><td colspan="5">Gold answer: Mumbai. Predicted answer: Mumbai. EM: 1.0 Retrieved positives: 2/2.</td></tr><tr><td rowspan="3">Latent Memory</td><td>Rank Pos? (gold)</td><td>Retrieval score</td><td>Corresponding text of retrieved latent token</td><td>CE loss</td><td></td></tr><tr><td>1 Y</td><td>0.623</td><td>Tata Consultancy Services: Tata Consultancy Services Limited (TCS) is an Indian multinational information technology service company headquartered in Mumbai, Maharashtra.</td><td>0.342</td><td></td></tr><tr><td>2 Y</td><td>0.557</td><td>Sachin Warrier: He was working as a software engineer in Tata Consultancy Services in Kochi.</td><td>0.083</td><td></td></tr></table>

## D.3 More Multimodal QA Case Studies

Figure 11 provides additional WebQA examples beyond the main-text case study, showing the counting and comparison ability in multimodal reasoning. These examples illustrate how Latent Memory retrieves from a unified text-image candidate pool and then conditions the frozen LLaVA-1.5- 13B generator on the retrieved latent tokens. We include both successful and challenging multimodal cases to show the qualitative behavior behind the aggregate results: image-grounded examples highlight whether visual evidence can be preserved after compression, while text-grounded examples show how textual facts are selected and used under the same retrieval interface.

![](images/a11f723550652f8875d609c29d0430b78b4caceafb4e72fd1099266543f866fe.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Image: Campbell County Courthouse in Gillette, Wyoming"] --> B["Query: Positive img, Negative img"]
  B --> C["Retrievable Latent Memory"]
  C --> D["Question: &quot;How many flag poles are there near the entrance to the Campbell County Courthouse in Gillette, Wyoming??&quot;"]
  D --> E["Latent context 1: z₁"]
  E --> F["Reconstructed Context (Positive)"]
  E --> G["Reconstructed Context (Negative)"]
  F --> H["Text: Campbell County: Wyoming Campbell County is a county in the U.S. state of Wyoming. As of the 2010 United States Census, the population was 46,133, making it the third-most populous county in Wyoming."]
  G --> I["Text: Campbell County, Wyoming: As of the 2010 United States Census, the population was 46,133, making it the third-most ..."]
```
</details>

![](images/614cb484d550d6fddc310a172500926ea8ae123d20318797b07a82c45135c6ee.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Image: Bentley Mulsanne"] --> B["Caption: Bentley Mulsanne"]
  B --> C["Image: 1967 Cadillac Series 75..."]
  C --> D["Text: Bentley Mulsanne (2010): The Mulsanne W.O. Edition was presented at the 2018 Geneva Motor Show as a 100th anniversary celebration of the foundin..."]
  D --> E["Text: Bentley Mulsanne (1980–1992): The BMW Goldfisch V16 engine was tested in the Bentley Mulsanne a"]
  E --> F["Text: ....."]
    
  G["Retrievable Latent Memory"] --> H["PC2 Query"]
  H --> I["PC1 Query"]
  I --> J["② Retrieve based on Query"]
    
    K["Question: &quot;Which car has rounder headlights, the 1967 Cadillac Series or the Bentley Mulsanne?&quot;"]
    L["Latent context: z₁"]
    M["Latent context: z₂"]
    
  N["① Latent Memory Construction"] --> O["Output: LLM/VLM θ with Trainable LoRA for Compression"]
  P["Frozen Larger-Size Generation LLM/VLM φ"] --> Q["Output: Brentley Mulsanne"]
  R["Generation with Latent Tokens"] --> S["Output: Brentley Mulsanne"]
    
  T["Reconstructed Text Context in C⁻"] --> U["Input: Bentley Mulsanne (2010): The Mulsanne W.O. Edition was presented at the 2018 Geneva Motor Show as .....(CE=1.8)"]
  V["Reconstructed Positive Image Context"] --> W["Image: Bentley Mulsanne..."]
  X["Reconstructed Positive Image Context"] --> Y["Image: 1967 Cadillac Se..."]
    
  Z["Output: LLM π with Trainable LoRA/ MLP for Decoding"] --> AA["Reconstructed Text Context in C⁻"]
  AB["Reconstructed Positive Image Context"] --> AC["Image: Bentley Mulsanne..."]
  AD["Reconstructed Positive Image Context"] --> AE["Image: 1967 Cadillac Se..."]
```
</details>

Figure 11: Additional multimodal WebQA case studies with LLaVA-1.5-13B. Each example uses the same Latent Memory retrieval-and-generation pipeline as the main experiments.

## E Baselines, Datasets, and Licenses

Datasets. Table 21 summarizes all datasets used in the paper. For the main text-only setting, Latent Memory is trained on the HotpotQA training split and evaluated on HotpotQA validation, 2WikiMultihopQA, and MuSiQue without additional task-specific fine-tuning for the out-of-domain datasets. The generalization experiments further evaluate the same text checkpoint on NQ, TriviaQA, Qasper, and WICE to cover open-domain factoid QA, scientific-document QA, and claim verification.

For multimodal QA, WebQA is used for both training and evaluation under its public split; evaluation is reported separately for image-grounded and text-grounded questions, while retrieval is performed over the unified text-image candidate pool. SlideQA is a pure visual dataset covering slides and detail capture. We hope to find more multimodal datasets similar to WebQA that involve retrieval and reasoning, but unfortunately, this is almost the only one. The rest of the datasets often contain data outside the Latent Memory scope, such as tables and videos.

As shown in Table 20 and case study Figures, all evidence is processed in “Title: Sentence“ form for the Text-only setting, “Title: Evidence“ and “Caption: Image“ for the multimodal setting.

Table 21: Coverage of the datasets used in this work.

<table><tr><td>Dataset</td><td>Knowledge / Task</td><td>Domain</td><td>Primary Source</td></tr><tr><td>HotpotQA [67]</td><td>Multi-hop entity-centric QA with supporting-fact grounding</td><td>Encyclopedic factual knowledge</td><td>Wikipedia articles and supporting sentences</td></tr><tr><td>2WikiMultihopQA [68]</td><td>Cross-page factual reasoning and relation composition</td><td>Encyclopedic factual knowledge</td><td>Wikipedia-based evidence chains</td></tr><tr><td>MuSiQue [69]</td><td>More compositional multi-hop QA over multiple supporting paragraphs</td><td>Broad factual knowledge</td><td>Multi-paragraph textual QA collections</td></tr><tr><td>WebQA [50]</td><td>Unified text-image QA with visually grounded evidence</td><td>Web knowledge, mixed textual and visual evidence</td><td>Web text and web images</td></tr><tr><td>SlideQA</td><td>Multimodal slide-domain QA with text and visual evidence</td><td>Slide / presentation understanding</td><td>Slide pages, associated visual regions, and textual metadata</td></tr><tr><td>NQ [70]</td><td>Open-domain factoid QA, often short-answer retrieval</td><td>Open-domain factual knowledge</td><td>Search queries paired with Wikipedia evidence</td></tr><tr><td>TriviaQA [71]</td><td>Trivia-style factoid QA with strong entity coverage</td><td>Open-domain factual knowledge</td><td>Trivia questions with web / Wikipedia evidence</td></tr><tr><td>Qasper [72]</td><td>Document-grounded QA over scientific papers</td><td>Scientific document understanding</td><td>Research papers with caption, section, and paragraph structure</td></tr><tr><td>WICE [73]</td><td>Evidence-centered claim verification and explanation</td><td>Fact verification / evidence reasoning</td><td>Claim-evidence collections built from textual evidence sources</td></tr></table>

Baseline settings. For text-only QA, the main table uses the same fixed Meta-Llama-3-8B-Instruct generation model and the same question prompt format. Full Context concatenates all available context sentences in the sample and sends them directly to the generator. LLMLingua [28] compresses the raw text context first, and the compressed text is then inserted into the same generator prompt. BM25 Retrieval [22] ranks candidate sentences with sparse lexical matching and feeds the top-k raw sentences to the generator. Dense Retrieval [21] ranks the same candidate sentence pool with all-MiniLM-L6-v2 sentence embeddings and also feeds top-k raw sentences. Qwen3-Embedding [51] uses the Qwen3 text embedding model as a stronger dense retriever, with generation still performed from retrieved raw text. Since a series of embedding models, such as the Qwen-3-Embedding model, are usually pre-trained for our in-domain scenario (Wikipedia), fine-tuning on our chosen in-domain environment could disrupt the balance; so we believe that direct comparison with Latent Memory for retrieval is relatively fair.

Because xRAG [32] and CLaRa [33] are built around Mistral-style latent-context generation, we only compare to their pretrained models on the Mistral-7B-Instruct setting in Table 2. For xRAG, we use the pretrained Hannibal046/xrag-7b generator-side checkpoint together with the pretrained Salesforce/SFR-Embedding-Mistral retrieval encoder. For CLaRa, we use the released pretrained CLaRa checkpoints at the reported 16× compression settings (which are more suitable for our sentence-level evidence). Our Mistral-setting Latent Memory uses LLaMA-3.2-1B-Instruct as both the compressor/encoder and the reconstruction decoder, while the answer generator is frozen Mistral-7B-Instruct.

For multimodal QA, all baselines retrieve from the same unified WebQA candidate pool [50] containing both text facts and images. We report two frozen generator families: LLaVA-1.5-13B in the main WebQA setting and Gemma-3-12B-Instruct in Appendix C.6. In the implementation, the LLaVA setting is run through scripts/baselines\_llava.py, while the Gemma setting is run through scripts/baselines\_qwen.py, whose generator wrapper also supports model\_type=gemma3. For Gemma, the same system instruction is prepended inside the user message because the Gemma chat template used by the processor does not take a separate system role in our code. Full Context passes all candidate text and images to the generator, which is often expensive because images expand through the model’s visual frontend. BM25 Retrieval [22] performs sparse retrieval over textual fields; if an image candidate is retrieved through its title/caption metadata, the raw image is still provided to the VLM for generation. Dense Retrieval [21] uses textual surrogates for retrieval: text facts are encoded directly, while images are represented by their titles/captions for scoring; after retrieval, the original raw image is sent to the generator. Qwen3-VL-Embedding [25], and Nemo Retriever [52] are multimodal retrieval baselines that rank text-image candidates with pretrained vision-language retrieval models, but the generator still consumes the retrieved raw text or image rather than the retrieval embedding. This keeps the retrieval candidate pool unified while making the raw-evidence baselines comparable to Latent Memory.

For all retrieval baselines, k denotes the number of retrieved evidence items and is matched to the corresponding Latent Memory setting when reported. Token counts always measure the generatorside input budget after constructing the final prompt. Thus, text retrieval baselines count the retrieved text tokens, multimodal baselines count the effective tokens consumed after visual processing, and Latent Memory counts the projected Latent Memory tokens inserted into the frozen generator.

Licenses. Table 22 summarises the main datasets, pretrained models, and baselines used in this work. We report the license or usage terms stated on the official upstream release pages that we could verify directly. When an official public release does not state a license, we mark it as not stated rather than inferring one. For benchmarks that redistribute web pages or images, downstream use must additionally comply with the rights of the original content providers.

Table 22: Summary of datasets, baselines, and upstream license provenance.

<table><tr><td>Item</td><td>Type</td><td>License / Terms</td><td>Source</td></tr><tr><td>HotpotQA [67]</td><td>Dataset</td><td>CC BY-SA 4.0</td><td>Hugging Face: hotpotqa/hotpot_qa</td></tr><tr><td>2WikiMultihopQA [68]</td><td>Dataset</td><td>Apache License 2.0</td><td>Hugging Face: xanhho/2WikiMultihopQA</td></tr><tr><td>MuSiQue [69]</td><td>Dataset</td><td>Available Online</td><td>Hugging Face dataset cards: bdsaglam/musique</td></tr><tr><td>WebQA [50]</td><td>Dataset</td><td>Available Online</td><td>WebQA official projects</td></tr><tr><td>NQ [70, 74]</td><td>Dataset</td><td>GitHub: TIGER-Lab/LongRAG</td><td></td></tr><tr><td>TriviaQA [71]</td><td>Dataset</td><td>Apache License 2.0 in the Hugging Face release used by our data pipeline</td><td>Hugging Face: mandarjoshi/trivia_qa</td></tr><tr><td>Qasper [72]</td><td>Dataset</td><td>CC BY 4.0</td><td>Hugging Face: allenai/qasper</td></tr><tr><td>WICE [73]</td><td>Dataset</td><td>Available Online</td><td>WiCE project page by the authors and mirrored dataset pages</td></tr><tr><td>SlideQA</td><td>Dataset</td><td>Available Online</td><td>Hugging Face: NTT-hil-insight/SlideVQA</td></tr><tr><td>BM25 Retrieval [22]</td><td>Baseline</td><td>Available Online</td><td>Standard sparse retrieval baseline</td></tr><tr><td>Dense Retrieval [21]</td><td>Baseline</td><td>Apache License 2.0</td><td>Hugging Face: sentence-transformers/all-MiniLM-L6-v</td></tr><tr><td>Qwen3-Embedding [51]</td><td>Baseline</td><td>Apache License 2.0</td><td>Hugging Face: Qwen/Qwen3-Embedding-0.6B</td></tr><tr><td>Qwen3-VL-Embedding [25]</td><td>Baseline</td><td>Apache License 2.0</td><td>Hugging Face: Qwen/Qwen3-VL-Embedding-8B</td></tr><tr><td>Nemo Retriever [52]</td><td>Baseline</td><td>Apache License 2.0</td><td>Hugging Face: nvidia/llama-nemotron-embed-vl-1b-v2</td></tr><tr><td>LLMLingua [28]</td><td>Baseline</td><td>Apache License 2.0</td><td>Microsoft LLMLingua repository</td></tr><tr><td>xRAG [32]</td><td>Baseline</td><td>Available Online</td><td>Pretrained checkpoint: Hannibal046/xrag-7b; retrieval encoder: Salesforce/SFR-Embedding-Mistral</td></tr><tr><td>CLaRa [33]</td><td>Baseline</td><td>Not stated</td><td>Released pretrained CLaRa checkpoints for 16× compression settings</td></tr><tr><td>Llama-3.2-1B-Instruct [48]</td><td>Model</td><td>Llama 3.2 Community License</td><td>Hugging Face / Meta model card: meta-llama/Llama-3.2-1B-Instruct</td></tr><tr><td>Meta-Llama-3-8B-Instruct [48]</td><td>Model</td><td>Meta Llama 3 Community License</td><td>Hugging Face / Meta model card: meta-llama/Meta-Llama-3-8B-Instruct</td></tr><tr><td>Mistral-7B-Instruct [49]</td><td>Model</td><td>Apache License 2.0</td><td>Hugging Face / Mistral AI model release: mistralai/Mistral-7B-Instruct-v0.3</td></tr><tr><td>LLaVA-1.5-7B [12]</td><td>Model</td><td>Llama 2 Community License</td><td>Hugging Face: liuhaotian/llava-v1.5-7b; LLaVA website</td></tr><tr><td>LLaVA-1.5-13B [12]</td><td>Model</td><td>Llama 2 Community License</td><td>Hugging Face: liuhaotian/llava-v1.5-13b; LLaVA website</td></tr><tr><td>Gemma-3-4B-PT [13]</td><td>Model</td><td>Gemma Terms of Use / Gemma license</td><td>Hugging Face: google/gemma-3-4b-pt; Google AI for Developers Gemma Terms</td></tr><tr><td>Gemma-3-12B-IT [13]</td><td>Model</td><td>Gemma Terms of Use / Gemma license</td><td>Hugging Face: google/gemma-3-12b-it; Google AI for Developers Gemma Terms</td></tr><tr><td>CLIP ViT-L/14-336 [45]</td><td>Model</td><td>MIT for OpenAI CLIP code/re-lease</td><td>GitHub: openai/CLIP; Hugging Face mirror: openai/clip-vit-large-patch14-336</td></tr></table>