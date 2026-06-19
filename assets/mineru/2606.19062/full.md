# DREAM: Extending Vision-Language Models with Dual-Objective Encoding for Cross-Modal Retrieval

Kaleem Ullah, Altaf Hussain, Muhammad Munsif, Sung Wook Baik\*

Abstract—In today’s media-driven world, the exponential growth of video content across domains such as surveillance, education, and entertainment has made retrieving semantically relevant videos via natural language queries increasingly critical. Early video retrieval systems relied on handcrafted features or shallow cross-modal mappings, limiting their ability to capture complex semantics and temporal dynamics. While largescale vision-language models (VLMs) have improved cross-modal alignment, challenges remain in modeling fine-grained temporal dependencies and nuanced linguistic structures. In this paper, we introduce DREAM: Dual-path Representation Enhancement and Alignment Model, a novel multimodal framework that addresses these limitations through enhanced visual and textual encoding. DREAM incorporates a hybrid language modeling strategy that combines masked and permuted language modeling objectives to capture both local and global linguistic semantics. On the visual side, we design a hierarchical vision encoder with cascaded group attention, which integrates spatial and temporal information through multi-stage token interaction and coarse-to-fine attention refinement. We validate DREAM through comprehensive evaluations on the widely-used MSRVTT, MSVD and LSMDC benchmark datasets, where it achieves new state-of-the-art R@1 scores of 49.4%, 49.7% and 27.3%, respectively. Qualitative analyses further show the model’s ability to maintain coherent attention across frames and align complex queries with dynamic video content. These findings underscore the effectiveness of hierarchical attention and dual-objective textual modeling in enabling robust, context-aware video retrieval, and pave the way for future research in advancing cross-modal representation learning.

Index Terms—Text-Video Retrieval, Multimodal Understanding, Cross-Modal Alignment, Video Understanding, Vision-Language Models.

## I. INTRODUCTION

V IDEO has become the dominant form of media in today’smulti-modal world, driven by the rapid expansion of multi-modal world, driven by the rapid expansion of online platforms and the widespread deployment of cameras across domains such as surveillance, entertainment, education, and autonomous driving [1]. As a rich and immersive medium, video offers greater expressiveness and user engagement than static images or text-based content. Unlike still images or textual descriptions, videos capture complex, evolving activities and richer semantics over time.

This surge in video content, with millions of new videos produced daily has created a pressing need for automated

This work was supported by National Research Foundation of Korea(NRF) grant funded by the Korea government (MSIT), Grant /AwardNumber:(RS-2023-NR076686)) Kaleem Ullah and Sung Wook Baik are with Sejong University, Seoul 143-747, South Korea, Altaf Hussain is with KAIST InnoCORE PRISM-AI Center, Korea Advanced Institute of Science and Technology (KAIST), Daejeon, 34141, Republic of Korea and Muhammad Munsif is with Ulsan National Institute of Science and Technology (UNIST), Ulsan, 44919, Republic of Korea. (Email: kaleemullah@sju.ac.kr, a.hussain@ieee.org, munsif@ieee.org, sbaik@sejong.ac.kr, (\*Corresponding author: Sung Wook Baik)

systems capable of understanding, organizing, and retrieving relevant visual information. Consequently, video understanding technologies have seen significant progress, powering applications in classification, summarization, and retrieval [2] [3]. Among these, cross-modal video-text retrieval has emerged as a critical research area, enabling systems to locate semantically relevant videos in response to natural language queries.

Recent advances in text-video retrieval have been driven by transformer-based vision-language models and contrastive pretraining frameworks. Methods such as CLIP4Clip [4], MV-Adapter [5], VILT-CLIP [6], and MUSE [7] introduced stronger spatiotemporal reasoning through global attention, multiscale processing, or lightweight temporal fusion. While these approaches improve video representation quality, many rely on flat or uniform attention mechanisms that lack hierarchical refinement and fail to capture fine-grained temporal dynamics in complex scenes. On the language side, models such as BLIP [8], BridgeTower [9], Long-CLIP [10], and TeachText [11] have demonstrated the benefits of generative supervision or distillation for enhancing semantic expressiveness. However, most retrieval systems still rely on a single modeling paradigm either masked or contrastive and thus fail to jointly capture local token dependencies and global sentence structure. These limitations highlight the need for retrieval frameworks that integrate richer linguistic modeling with hierarchical and discriminative spatiotemporal representation learning.

To address these challenges, we propose a unified retrieval framework DREAM, that jointly models visual and textual representations using a hybrid language encoder and a hierarchical vision encoder. On the text side, we introduce a hybrid language modeling strategy that combines masked and permuted objectives to capture both local context and global semantic structure. We adopt this hybrid strategy to leverage the complementary strengths of both objectives, capturing finegrained dependencies as well as broader contextual flow. On the visual side, we develop a hierarchical vision encoder with cascaded group attention (CGAT) to refine spatiotemporal features through multi-stage token interaction. This hierarchical design effectively captures multi-scale spatial and temporal dependencies for more expressive visual representation. Unlike CLIP-based methods, which rely on simple framelevel feature aggregation, our hierarchical design progressively refines spatial and temporal dependencies, capturing multiscale spatiotemporal cues for more expressive visual representations. Together, these approaches enable more accurate and context-aware video retrieval, as demonstrated through our experiments on the benchmark datasets. Furthermore, to provide a concise overview and impact of our approach, we summarize the key contributions of this work as follows.

![](images/efde08aa0d6b9890417e08eb7060ab078b4052fd92ebc9ffe16ed37f0802b8ac.jpg)

<details>
<summary>timeline diagram</summary>

| Year | Value |
| ---- | ----- |
| 2017 | [13]  |
| 2019 | [30]  |
| 2021 | [4],[11] |
| 2023 | [5], [35] |
| 2025 | [6], [7] |
| 2017 | [29]  |
| 2019 | [31]  |
| 2021 | [32], [10] |
| 2023 | [20], [19] |
| 2025 | [2], [3]
</details>

Fig. 1. Progress of methodologies in text-to-video retrieval. With the establishment and broader adoption of standard benchmarks, research has shifted from earlier feature-engineering and statistical matching pipelines toward predominantly deep learning–based representation learning and video–text alignment methods.

## A. Main Contributions:

• We propose a novel dual-path text encoding framework that enhances semantic understanding through the integration of Masked Language Modeling (MLM) and Permuted Language Modeling (PLM) objectives. This hybrid structure captures both fine-grained local context and global sentence structure in textual queries, enabling more accurate interpretation of ambiguous and contextdependent queries. On top of a CLIP text encoder, our hybrid approach significantly improves the robustness and accuracy of text representations, leading to superior alignment with complex and temporally dynamic video content.  
• We propose a visual encoder that combines hierarchical feature extraction with efficient attention mechanisms for video representation learning. Drawing on principles from efficient transformer architectures, the model integrates multi-stage token embeddings and a Cascaded Group Attention module to progressively distill spatial semantics. This design improves the encoder’s ability to capture fine-grained visual patterns across multiple spatial and temporal scales, resulting in richer and more discriminative video embeddings that encapsulate context for text-to-video retrieval.  
• Extensive experiments on the MSR-VTT, MSVD and LSMDC datasets demonstrate the effectiveness of our proposed framework. Our model achieves state-of-the-art retrieval performance, surpassing previous methods with an R@1 of 49.4 on MSR-VTT, 49.7 on MSVD, and 27.3 on LSMDC along with the lowest median rank (MdR) on these datasets. These results highlight the strength of our design in capturing fine-grained video-text semantics and ensuring robust cross-modal retrieval.

The rest of this paper is organized as follows: Section II reviews prior work along two main dimensions relevant to textvideo retrieval, namely text modeling for cross-modal representation learning and video modeling for efficient spatiotemporal understanding, and identifies the limitations that motivate our approach. Section III details the proposed DREAM retrieval framework for cross-modal alignment mechanism. Section IV presents the experimental results, comparisons along with qualitative analysis. Finally, Section V concludes the article with a summary of findings, current limitations and potential directions for future work.

## II. RELATED WORK

Multimodal retrieval research has advanced primarily along two complementary dimensions: the development of expressive text encoders to better interpret natural language queries, and the design of efficient video encoders capable of capturing complex spatiotemporal patterns. Fig. 1 shows this evolution outlining the chronological development of key retrieval methods, these approaches have shaped the evolution of crossmodal retrieval from early image-based systems to modern transformer-driven frameworks. Below, we review representative developments in each direction, identify their limitations, and motivate the architectural choices underlying DREAM.

## A. Text Modeling for Cross-Modal Retrieval

The use of natural language as a retrieval query emerged in response to the limitations of visual-only systems, such as content-based image retrieval (CBIR), which relied heavily on color, texture, and structural descriptors. Although deep learning significantly improved CBIR by enabling more semantic feature extraction, the visual modality alone often failed to capture user intent when scenes contained ambiguous context or irrelevant background content. As a result, text-guided retrieval methods gained prominence due to the richer and more explicit semantic information provided by natural language. Early text-to-image retrieval (TIR) relied on shallow statistical correlation models, including latent semantic indexing and canonical correlation analysis [12]–[14], which struggled to capture high-level semantics and non-linear relationships between text and images. Deep learning approaches subsequently introduced CNN-RNN pipelines, such as the model proposed by Karpathy and Fei-Fei [15], enabling joint embedding learning and improving alignment between modalities. The introduction of transformers and large-scale multimodal pretraining further advanced this direction. In particular, CLIP [16] demonstrated that contrastive training on large text-image corpora can yield highly transferable textual embeddings, even though its text encoder is trained without any masked or autoregressive language-modeling objective. BLIP [8] extended multimodal understanding by incorporating generative supervision that enriches textual semantics, and BridgeTower [9] improved cross-modal fusion through deeper cross-layer interactions.

Despite these advancements, most retrieval-oriented text encoders derive their linguistic representations from either masked language modeling, as in BLIP and BridgeTower, or from architectures optimized solely through contrastive objectives, as in CLIP and Long-CLIP [10]. Masked modeling captures fine-grained local context but provides limited control over global sentence structure, while contrastive-only encoders optimize for sentence-level alignment but do not explicitly model token-wise dependencies. Other approaches, such as TeachText [11], enhance semantic diversity through distillation, yet still rely on a single modeling perspective determined by the teacher network. As a result, existing text encoders tend to emphasize either local semantics or global structure, making it difficult to represent queries involving multi-stage activities, compositional phrasing, or subtle semantic distinctions. These limitations highlight the need for a textual modeling strategy that jointly captures fine-grained dependencies and global semantic organization within a unified framework. DREAM addresses this gap by introducing a dual-path semantic modeling approach that integrates complementary linguistic cues, enabling richer and more discriminative textual representations tailored for alignment with dynamic video content.

## B. Video Modeling and Efficient Spatiotemporal Attention

Text-to-video retrieval (TVR) introduces the additional challenge of reasoning over motion and temporal structure. Early methods such as Two-Stream CNNs [17] and C3D networks [18] learned coarse spatiotemporal features but were constrained by limited receptive fields and difficulty modeling long-term dependencies. Transformer-based architectures led to significant progress by using self-attention to capture temporal relations explicitly. TimeSformer [19], for example, factorized attention into spatial and temporal components to improve efficiency, while ViT-based video transformers extended image-level architectures to video settings. CLIP4Clip [4] further demonstrated that CLIP embeddings could be transferred to video retrieval by aggregating frame-level features, and frameworks such as VILT-CLIP [6] and MV-Adapter [5] proposed lightweight fusion modules to enhance cross-modal interactions without excessive computational overhead.

Despite these advances, many of these models employ flat global attention or uniform temporal aggregation, treating all tokens equally regardless of their semantic salience. This uniformity can limit a model’s ability to capture subtle transitions, multi-stage actions, or object-action interactions, and often leads to high computational cost when processing densely sampled videos. To overcome these issues, recent methods have explored structured or hierarchical mechanisms. MUSE [7] introduced multi-scale temporal transformers to model varying temporal resolutions UMP [20] incorporated modalityaware prompts to guide attention and LSDO [21] emphasized low-salient object cues to improve fine-grained retrieval. However, these systems typically use fixed attention patterns or single-stage fusion and lack mechanisms for progressive refinement across hierarchical levels.

The remaining challenge lies in designing video encoders that capture coarse contextual patterns early while refining fine-grained spatial and temporal details in later stages. DREAM addresses this challenge by employing a hierarchical visual encoder equipped with cascaded group attention, allowing for progressive, coarse-to-fine feature refinement. This structure enhances temporal sensitivity and spatial discrimination while maintaining computational efficiency, enabling more robust alignment with complex textual queries. Although prior work has significantly advanced text and video representation learning, important gaps persist. Text encoders often fail to integrate local and global semantics within a single modeling framework, while video encoders frequently rely on flat or weakly structured attention mechanisms that limit their ability to model rich, multi-scale spatiotemporal patterns. DREAM introduces two complementary approaches to address these challenges: a dual-path text encoder that captures both fine-grained and global linguistic semantics, and a hierarchical video encoder that progressively refines representations through cascaded group attention (CGAT). Together, these components form a unified retrieval framework that advances beyond existing designs and addresses longstanding challenges in text-video alignment.

## III. METHOD

In this section, we introduce our overall approach for TVR. DREAM is designed to effectively capture and align information from both textual and visual modalities. On the language side, we enhance the model’s ability to understand and represent complex descriptions. For the visual component, we utilize a computationally optimized backbone that processes high-dimensional video data while preserving spatial and temporal fidelity. Additionally, we introduce mechanisms to better integrate spatial and temporal information across video frames. Together, these design choices lead to improved semantic alignment between text and video. To address the two core challenges fine-grained temporal dependencies and nuanced linguistic structures our design choices in DREAM are purposefully aligned with these motivations. The hierarchical vision encoder, equipped with CGAT, models subtle temporal transitions and multi-scale spatiotemporal cues by progressively refining token representations across stages and recursively fusing them through coarse-to-fine attention groups. This enables the model to track object, local transitions and temporal context that are often overlooked in singlestage or static pooling approaches. On the language side, the dual-stream hybrid modeling strategy is introduced to capture both short-range syntactic cues and long-range compositional structure. MLM resolves phrase-level ambiguities by focusing on masked tokens, while PLM models global ordering and semantic relations, enabling the encoder to interpret complex multi-event descriptions. Together, these components ensure that DREAM effectively addresses the temporal and linguistic challenges inherent in text-video retrieval. The following subsections provide detailed insight into each component of our proposed framework.

![](images/adbdbde00768c54692ce7140ac29b82a47530a7018253748a0109e0b89f31f0b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Stage1[Stage 1]
  A["Patch Emb"] --> B["Encoder"]
  B --> C["Layer 1: F₁: H/8 × w/8 × C₁"]
  C --> D["Layer 2: F₂: H/16 × W/16 × C₂"]
  D --> E["Layer 3: F₃: H/32 × W/32 × C₃"]
  E --> F["Layer 4: F₄: H/64 × W/64 × C₄"]
    end

    subgraph Stage2[Stage 2]
  G["Patch Emb"] --> H["Encoder"]
  H --> I["Layer 2: F₂: H/16 × W/16 × C₂"]
  I --> J["Layer 3: F₃: H/32 × W/32 × C₃"]
  J --> K["Layer 4: F₄: H/64 × W/64 × C₄"]
    end

    subgraph Stage3[Stage 3]
  L["Patch Emb"] --> M["Encoder"]
  M --> N["Layer 3: F₃: H/32 × W/32 × C₃"]
  N --> O["Layer 4: F₄: H/64 × W/64 × C₄"]
    end

    subgraph Stage4[Stage 4]
  P["Patch Emb"] --> Q["Encoder"]
  Q --> R["Layer 4: F₄: H/64 × W/64 × C₄"]
    end

    subgraph Split
  S["Head 1"] --> T["Token Interaction"]
  T --> U["Stacked Group Attention"]
  U --> V["Stacked Group Attention"]
  V --> W["Stacked Group Attention"]
  X["Head 2"] --> Y["Token Interaction"]
  Y --> Z["Stacked Group Attention"]
  Z --> AA["Stacked Group Attention"]
  AB["Head n"] --> AC["Token Interaction"]
  AC --> AD["Stacked Group Attention"]
  AE["Concat & Project"] --> AF["Output"]
    end

    subgraph TokenInteraction
  AG["Feed Forward Network"] --> AH["Input"]
  AI["Token Interaction"] --> AJ["Input"]
  AK["Cascaded Group Attention"] --> AL["Input"]
  AM["Feed Forward Network"] --> AN["Input"]
  AO["Token Interaction"] --> AP["Input"]
    end

    subgraph TokenInteraction
  AQ["Input"] --> AR["Input"]
  AS["Input"] --> AT["Input"]
    end

    subgraph Concat & Project
  AU["Input"] --> AV["Output"]
    end
```
</details>

Fig. 2. Overview of the visual feature encoder architecture. The top (a) shows a four-stage hierarchical transformer that encodes video frames into multiscale token representations. The bottom-left (b) illustrates the Cascaded Group Attention module, which enhances feature refinement through token interaction and attention. The bottom-right (c) section shows grouped tokens processed across multiple attention heads for feature integration.

## A. Multi-Stage Visual Feature Extraction

Understanding and modeling visual semantics is fundamental to effective TVR, where the goal is to align highdimensional visual content with natural language descriptions. To bridge the semantic gap between textual queries and visual content in video retrieval, we design a hierarchical vision encoder that effectively transforms raw frame sequences into rich, discriminative feature embeddings. Specifically, the video frames $\{ I _ { t } \} _ { t = 1 } ^ { T }$ , where each frame ${ { I } _ { t } } ~ \in ~ { \mathbb { R } ^ { H \times W \times 3 } }$ , are processed through a series of stages, generating visual representations denoted as $Z _ { s } \in \mathbb { R } ^ { N _ { s } \times d _ { s } }$ at each stage s. Here, $N _ { s }$ is the number of tokens at stage s, and $d _ { s }$ is the feature dimension at that stage. DREAM builds on the strengths of multi-stage transformers, CGAT, and token interaction mechanisms to progressively distill spatiotemporal information into compact visual representations. The architecture is depicted in Fig. 2.

The vision encoder operates over a sequence of video frames $\{ I _ { t } \} _ { t = 1 } ^ { T }$ , where each frame ${ { I } _ { t } } ~ \in ~ \bar { \mathbb { R } } ^ { H \times W \times 3 }$ is first partitioned into fixed-size non-overlapping patches. At each stage $s \in \{ 1 , 2 , 3 , 4 \}$ , a patch embedding layer projects the flattened patches into a token embedding space via a linear transformation $E _ { s } : \mathbb { R } ^ { p ^ { 2 } \cdot 3 } \  \ \mathbb { R } ^ { d _ { s } }$ , where $p$ is the patch size and $d _ { s }$ denotes the feature dimension at stage s. The resulting token sequence is then processed by a transformer encoder block $\vartheta _ { s }$ , producing hidden representations $Z _ { s } ~ =$ ϑs $\left( E _ { s } \left( \mathrm { p a t c h e s } ( I _ { t } ) \right) \right)$ . In practice, the input to stage s is the refined token representation produced by stage s−1, enabling progressive hierarchical abstraction rather than independent processing of raw frames.

To enhance intra- and inter-token communication beyond standard self-attention, each stage is augmented with a TokenInteract module. Given input tokens $Z ~ \in ~ \mathbb { R } ^ { N \times d } .$ , the module performs a lightweight local transformation followed by pairwise token interaction. First, a non-linear token-wise projection mixes local channel information using eq. 1:

![](images/2b0e72fd761a8e77b26a5b61e7de238b04bf1db15be4dbf06cee06b8a2d499bc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph a[Transformer Encoder]
  A1["Add & Norm"] --> A2["Feed Forward"]
  A2 --> A3["Add & Norm"]
  A3 --> A4["Multi-Head Attention"]
  A4 --> A5["Add & Norm"]
  A5 --> A6["Feed Forward"]
  A6 --> A7["Add & Norm"]
  A7 --> A8["Multi-Head Attention"]
  A8 --> A9["Positional Encoding"]
  A9 --> B["Byte Pair Encoding"]
  B --> C["cake is placed on a plate"]
    end

    subgraph b[Masked And Permuted Language Modeling]
  D1["Can See Self"] --> D2["Add & Norm"]
  D2 --> D3["Feed Forward"]
  D3 --> D4["Add & Norm"]
  D4 --> D5["Multi-Head Attention"]
  D5 --> D6["Positional Encoding"]
  D6 --> E["MLM X1 X2 X3 X4 X5 X6 X7 X8 X9"]
    end

    subgraph c
  F1["Cannot See Self"] --> F2["Add & Norm"]
  F2 --> F3["Feed Forward"]
  F3 --> F4["Add & Norm"]
  F4 --> F5["Multi-Head Attention"]
  F5 --> F6["Positional Encoding"]
  F6 --> G["PLM X1 X2 X3 X4 X5 X6 X7 X8 X9"]
    end
```
</details>

Fig. 3. Text encoder architecture illustrating two complementary language modeling strategies within a multi-head attention framework. (a) A transformerbased encoder processes input text, which is first tokenized using byte pair encoding and enriched with positional encodings. (b) The architecture branches into two parallel modeling objectives: masked language modeling (MLM), where selected tokens are masked and predicted based on visible context (including self), and permuted language modeling (PLM), where input tokens are shuffled and processed through a dual-stream attention mechanism. In PLM, the content stream can see all tokens, while the query stream cannot attend to itself.

$$
Z ^ {\mathrm{TI}} = \sigma (Z W _ {1}) W _ {2}, \tag {1}
$$

where $W _ { 1 } \in \mathbb { R } ^ { d \times d _ { h } }$ and $W _ { 2 } \in \mathbb { R } ^ { d _ { h } \times d }$ are learnable matrices, $d _ { h }$ is hidden dimension, and $\sigma ( \cdot )$ is GELU for non-linearity. Furthermore to capture lightweight pairwise relations, we compute an affinity matrix through feature similarity with eq. 2 which is aggregated for contextual evidence in eq. 3:

$$
A = \text { softmax } \left(\frac {Z Z ^ {\top}}{\sqrt {d}}\right), \tag {2}
$$

$$
Z ^ {\text { pair }} = A Z. \tag {3}
$$

The output of the TokenInteract module is then formed in eq. 4 by combining the two components:

$$
\operatorname{TokenInteract} (Z) = Z ^ {\mathrm{TI}} + Z ^ {\text { pair }}. \tag {4}
$$

Thus, the overall token interaction step becomes:

$$
Z ^ {\prime} = Z + \text { TokenInteract } (Z), \quad Z ^ {\prime \prime} = Z ^ {\prime} + \text { FFN } (Z ^ {\prime}), \tag {5}
$$

where FFN(·) denotes the position-wise feedforward network. This module introduces localized contextualization and lightweight pairwise refinement prior to the deeper selfattention computation.

Central to our encoder design is the CGAT mechanism, which fuses multiscale features from different encoder stages in a progressive fine-to-coarse manner. Feature maps from all stages are partitioned into non-overlapping spatial windows, forming groups $\{ G _ { k } \} _ { k = 1 } ^ { K }$ . Each group corresponds to a fixed spatial region of the frame (e.g., 8×8, 16×16) patches depending on the stage. This spatial grouping ensures computational efficiency and provides a natural hierarchy for progressively refining local details into broader semantic representations. Unlike conventional visual encoders, the CGAT mechanism explicitly models fine-grained temporal dependencies by enabling hierarchical information flow from lower-level local groups to higher-level semantic groups. This bottom-up refinement allows the encoder to capture micro-temporal variations, object interactions, and action transitions across frames, elements that are often missed by standard CLIP-based encoders or temporally pooled representations. To achieve this, the outputs from lower-level groups are recursively injected into higher-level groups, promoting progressive fusion of local and global semantics. Within each group, standard self-attention (∂) is performed using learned queries, keys, and values $( Q , K , V )$ derived from the grouped tokens, as formalized in Eq. 6.

$$
\partial (Q, K, V) = \text { softmax } \left(\frac {Q K ^ {\top}}{\sqrt {d}}\right) V \tag {6}
$$

Each group output is then recursively refined through bottom-up fusion. Specifically, self-attention is first applied within each group to model intra-group dependencies, after which the output is refined using the representation from group $k { - } 1$ , enabling information flow from lower to higher semantic levels. This recursive formulation is written as in Eq. 7

$$
Z _ {k} = \mathrm{SA} (Q _ {k}, K _ {k}, V _ {k}) + f (Z _ {k - 1}) \tag {7}
$$

where $f ( \cdot )$ denotes a linear projection or residual mapping. Finally, the outputs of all group attention branches are concatenated along the token dimension and passed through a projection layer $W _ { p } ~ \in ~ \mathbb { R } ^ { d \times d }$ , yielding a unified video embedding presented in eq. 8

$$
\mathcal {V} = W _ {p} \cdot \left[ Z _ {1} \| Z _ {2} \| \dots \| Z _ {K} \right] \tag {8}
$$

where ∥ denotes token-wise concatenation. This embedding $\boldsymbol { \nu } \in \mathbb { R } ^ { \ddot { N } ^ { \prime } \times d }$ , where $N ^ { \prime }$ is the total number of tokens after concatenation, captures multi-scale visual information and serves as the visual representation used for alignment with textual embeddings in downstream retrieval tasks.

## B. Dual-Stream Language Modeling

Capturing the nuanced semantics of natural language is crucial for bridging the gap between textual queries and dynamic visual content in TVR, particularly when addressing linguistic ambiguity in the context of dynamic visual events. To solve this, we design a hybrid text encoder that enhances the representational strength of CLIP’s transformer with a dual-language modeling mechanism. The hybrid MLM–PLM structure is specifically designed to capture both fine-grained local dependencies and global structural semantics. MLM focuses on resolving localized dependencies, such as verbobject relationships, by reconstructing randomly masked tokens, while PLM models longer-range syntactic structures by predicting tokens within a permuted sequence. This dual approach allows the model to handle both short-range linguistic patterns and complex, global dependencies, directly addressing the ambiguity and compositionality often present in natural language queries for video retrieval. As illustrated in Fig. 3(b), this dual-path approach strengthens the encoder’s capacity to understand both local and global linguistic cues, providing a richer semantic representation that aligns well with the complex structure of video content. We formalize the input caption as a tokenized sequence derived from the video’s textual annotation, as shown in eq. 9.

$$
X = \{x _ {1}, x _ {2}, \dots , x _ {T} \}, \quad x _ {i} \in \mathcal {W} \tag {9}
$$

Where T is the sequence length and W denotes the vocabulary. This sequence is passed through the CLIP text encoder, denoted as ${ \mathcal { E } } _ { \mathrm { C L I P } }$ in eq. 10 to obtain contextualized embeddings:

$$
H _ {\text { clip }} = \mathcal {E} _ {\text { CLIP }} (X), \quad H _ {\text { clip }} \in \mathbb {R} ^ {T \times d} \tag {10}
$$

To refine the semantic representation, we apply MLM and PLM objectives in parallel. For MLM, a subset M of tokens is masked and the model is trained using the remaining visible context as given in eq. 11

$$
\mathcal {L} _ {\mathrm{MLM}} = - \sum_ {i \in \mathcal {M}} \log P (x _ {i} \mid \tilde {X} _ {\mathrm{MLM}}) \tag {11}
$$

This encourages the encoder to infer token-level semantics based on surrounding context. In contrast, in eq. 12 PLM applies a random permutation π to the input sequence indices and trains the model to autoregressively predict the permuted tokens.

$$
\mathcal {L} _ {\mathrm{PLM}} = - \sum_ {t = 1} ^ {T} \log P \left(x _ {\pi (t)} \mid x _ {\pi (1)}, \dots , x _ {\pi (t - 1)}\right) \tag {12}
$$

To effectively model dependencies while preventing information leakage in the PLM path, we incorporate a dualstream attention mechanism, as illustrated in Fig. 3(b). In addition to $\mathrm { C L I P } ^ { \prime } \mathrm { s }$ text encoder, our architecture introduces two parallel attention streams: a content stream, which provides keys and values using full context, and a query stream, which restricts tokens from attending to themselves, thereby enforcing a causal structure. For each token $x _ { \pi ( t ) }$ in the permuted sequence, the attention computation follows the standard formulation in Eq. 6 In this setup, the content stream generates K and V from the entire sequence, while the query stream computes Q using a masked view that excludes $x _ { \pi ( t ) }$ itself. This dual-stream design ensures that autoregressive prediction remains faithful to the temporal structure of the permuted sequence. By decoupling the query and context perspectives, the model effectively learns global structure without information leakage, thereby enhancing the semantic richness of the resulting text representations.

Finally, the outputs from the MLM and PLM branches are fused with the original CLIP embeddings using residual summation as shown in Eq. 13:

$$
H _ {\text { text }} = H _ {\text { clip }} + \lambda \cdot (H _ {\mathrm{MLM}} + H _ {\mathrm{PLM}}) \tag {13}
$$

where HMLM and $H _ { \mathrm { P L M } }$ denote the hidden representations produced by the MLM and PLM branches, respectively, and λ is a scalar that balances their contribution. The final embedding $H _ { \mathrm { t e x t } } \in \mathbb { R } ^ { T \times d }$ encodes multimodal priors, local context, and global semantics, making it well-suited for alignment with complex visual content in downstream retrieval tasks. While our PLM branch adopts the dual-stream attention mechanism inspired by XLNet [22], our implementation differs in two key ways: (i) it is embedded within CLIP’s transformer encoder

![](images/116f1e84536e4ca242fb7f98378c59e0de674d3f04d7e8de1dec816d902125d6.jpg)

1．A Heavy metal melting and shaping.  
2.Large tell being melted in a factory.  
3.Men are working at a steel manufacturing plant.  
4.Acompany producing steel.  
5.Someone is melting a metal in a industry.

1.Basketball teams play crowd watches.  
2.Players are playing basketball.  
3.Players play basketball，one had put a ball in basket  
4.Ultimate players' basketball technique  
5.A nba player does cross over and scores a basket

1.Man balances and pushing bicycle.  
2.Man with bicycle falls on sign.  
3.A man and bike fell on road sign.  
4.Man is tiying to use pole to push off bike ends up falling.  
5.A man fell by the road with his bike and a signboard.

Fig. 4. Examples of video-text pairs from benchmark datasets. Each clip is represented by four sampled frames (left) and five corresponding human-annotated sentences (right). The first two examples (Top Row) are drawn from the MSR-VTT dataset, while the remaining examples are from the MSVD dataset.

rather than a standalone language model, and (ii) it is trained jointly with video embeddings under a retrieval objective. This retrieval-specific conditioning ensures that the learned global dependencies are not generic but tuned to align with temporally dynamic video content. DREAM introduces a hybrid language encoding framework that combines MLM and PLM within a contrastive retrieval setting, a first in video-text retrieval. This approach allows DREAM to address complex challenges, such as multi-event query alignment and fine-grained temporal modeling, which traditional single-path models cannot capture. The MLM and PLM objectives are used as auxiliary training losses rather than being directly added to the text embeddings. Therefore, the overall training objective is formulated as Eq. 14, where $\mathcal { L } _ { \mathrm { r e t } }$ denotes the videotext retrieval loss.

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\text { ret }} + \mathcal {L} _ {\text { MLM }} + \mathcal {L} _ {\text { PLM }} \tag {14}
$$

## C. Cross-Modality Alignment

A fundamental requirement for TVR is the construction of a unified embedding space in which heterogeneous modalities visual and textual data can be directly compared based on their semantic content. This necessitates the projection of both video and language representations into a common latent space wherein cross-modal alignment is both geometrically meaningful. Let $\gamma \in \mathbb { R } ^ { d }$ denote the aggregated video embedding and $\mathcal { T } \in \mathbb { R } ^ { d }$ the corresponding textual embedding, both derived from their respective modality-specific encoders.

To quantify the semantic relatedness between these representations, we employ the cosine similarity in Eq. 15, a scaleinvariant metric that captures the angular distance between two vectors in high-dimensional space

$$
S (\mathcal {V}, \mathcal {T}) = \frac {\mathcal {V} \cdot \mathcal {T}}{\| \mathcal {V} \| \| \mathcal {T} \|} \tag {15}
$$

Following the groundwork established in [4] we aggregate frame-level features using mean pooling, which has been empirically validated as an effective and efficient strategy for making video features comparable with textual embeddings. This ensures that both video and textual representations are aligned in a common latent space prior to similarity computation. Furthermore both embeddings undergo $\ell _ { 2 }$ normalization to constrain them to the unit hypersphere, thereby ensuring that similarity comparisons reflect pure alignment rather than disparities. During inference, this similarity metric is evaluated between a given textual query and the video embeddings within the retrieval corpus. The resulting scores are then ranked to identify the top-k candidate videos whose semantic representations exhibit maximal alignment with the input query. A key design decision in the DREAM framework is the use of global video-text alignment instead of frame-word or region-phrase correspondence. Methods, such as token-wise cross-attention, prevent efficient pre-indexing of video representations, while Global design enables real-time retrieval by decoupling video and text embeddings. CGAT mechanism in the DREAM implicitly attends to local spatiotemporal details, allowing the global token to capture essential features for temporally complex queries. Additionally, global-alignment approach is more robust to asynchronous data in real-world datasets, where fine-grained alignment methods struggle with loose temporal synchronization.

TABLE I ABLATION STUDY ON THE MSRVTT DATASET. WE REPORT R@1 (%) WHILE INCREMENTALLY ADDING KEY COMPONENTS TO THE BASELINE. ✓ = COMPONENT INCLUDED, ✗ = EXCLUDED. MLM AND PLM DENOTE MASKED AND PERMUTED LANGUAGE MODELING, RESPECTIVELY, WHILE CGAT STANDS FOR CASCADED GROUP ATTENTION AND HVE FOR THE HIERARCHICAL VISION ENCODER

<table><tr><td>Exp</td><td>MLM</td><td>PLM</td><td>CLIP Text</td><td>CGAT</td><td>HVE</td><td>R@1</td></tr><tr><td>1</td><td>✗</td><td>✗</td><td>√</td><td>✗</td><td>✗</td><td>44.2 ± 0.13</td></tr><tr><td>2</td><td>√</td><td>✗</td><td>√</td><td>✗</td><td>✗</td><td>45.5 ± 0.11</td></tr><tr><td>3</td><td>✗</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>45.2 ± 0.12</td></tr><tr><td>4</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>✗</td><td>48.1 ± 0.11</td></tr><tr><td>5</td><td>√</td><td>√</td><td>√</td><td>√</td><td>✗</td><td>48.8 ± 0.13</td></tr><tr><td>6</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>49.4 ± 0.12</td></tr></table>

## IV. RESULTS AND DISCUSSION

This section provides a comprehensive evaluation of the proposed method using publicly available benchmark datasets for video retrieval. We assess the performance from multiple perspectives, incorporating both quantitative and qualitative analyses. The following subsections offer a detailed discussion of the results obtained through this evaluation.

## A. Experimental Setup:

All the experiments were conducted on a PC running the Ubuntu operating system, equipped with four NVIDIA RTX A5000 GPUs, 128 GB RAM, and an AMD Ryzen 9 3900 24-core processor operating at a base clock speed of 4.2 GHz and 5.3 GHz boost frequency. The proposed framework is implemented in Python 3.10 using PyTorch [23] version 2.4+cu118. The training and evaluation procedures including a batch size of 32 for training and 16 for validation, a learning rate of 1e-4, and a total of 5 training epochs. We utilized a maximum of 12 frames per video and 32 tokens per caption following [4] . The Adam [24] optimizer was used for training, and inference was performed with frame-based inputs at a rate of 1 frame per second. All experiments were repeated with three random seeds. We report mean ± standard deviation across runs to ensure statistical robustness.

## B. Evaluation Metrics:

To evaluate the performance of DREAM, we utilize three standard metrics commonly used in information retrieval and ranking task. Recall at K (R@K), Median Rank (Md@R), and Mean Rank (Mn@R). These metrics provide a comprehensive understanding of the effectiveness and efficiency of our approach.

1) Recall at K (R@K): Recall at K (R@K) measures the proportion of queries for which at least one relevant item appears in the top-K retrieved results. Let N denote the total number of queries, and let $r _ { i } ^ { ( K ) } \in \{ 0 , 1 \}$ be an indicator variable that equals 1 if a relevant item for query i is retrieved within the top-K results, and 0 otherwise. Then, R@K is defined as in Eq. 16

$$
\mathrm{R} @ K = \frac {1}{N} \sum_ {i = 1} ^ {N} r _ {i} ^ {(K)} \tag {16}
$$

This metric captures how often the retrieval system succeeds within the top-K positions, aligning with user expectations in practical search scenarios.

2) Median Rank (Md@R): The Median Rank (M d@R) is the median of the ranks of the first relevant item retrieved for all queries. It provides a robust central tendency measure of the ranks and is less sensitive to outliers compared to the mean. A lower M d@R value indicates that the relevant items are ranked closer to the top on average. This metric is useful for understanding typical query performance without being skewed by extreme cases.  
3) Mean Rank (Mn@R): The Mean Rank (Mn@R) calculates the average rank of the first relevant item retrieved for all queries. It is calculated as in Eq. 17

$$
\mathrm{Mn@R} = \frac {\sum_ {i = 1} ^ {N} r _ {i}}{N} \tag {17}
$$

where $r _ { i }$ is the rank of the first relevant item for the i-th query, and N is the total number of queries. While M n@R is sensitive to outliers, it provides insight into the overall ranking performance across the dataset.

## C. Datasets:

The proposed method was evaluated using three benchmark datasets: MSRVTT [25], MSVD [26] and LSMDC [27]. Fig. 4 shows sample videos and their annotated sentences. MSRVTT is a large-scale web video dataset comprising 10,000 videos totaling 41.2 hours and 200K clip-sentence pairs. Each video is accompanied by 20 natural language annotations, with an average duration of 30 seconds per video. For evaluation, the test set used is ‘test1k-A’, which contains 1,000 cliptext pairs following the JSFusion setup [28]. MSVD contains 1,970 videos, each with a length that ranges from one to 62 seconds. Train, validation and, test splits contain 1,200, 100, and 670 videos, respectively. Each video has approximately 40 associated sentences in English. The LSMDC [27] dataset consists of 118,081 video clips, each ranging from 2 to 30 seconds in length. These clips are extracted from 202 movies. The validation set includes 7,408 videos, while the test set contains 1,000 videos, all sourced from movies that are separate from those used in the training and validation sets.

## D. Ablation Study:

We perform the ablation study on the MSRVTT dataset to evaluate the individual contributions of each component in our framework. MSRVTT offers a larger scale, greater diversity, and more complex textual annotations, making it a more challenging and representative benchmark for retrieval performance. Table I presents Recall@1 (R@1) scores for various configurations of our model, incrementally adding or removing key components. Starting from CLIP text embeddings and simple temporal pooling, we observe a performance of 44.2%

TABLE II TEXT-VIDEO RETRIEVAL PERFORMANCE COMPARISON OF METHODS ON MSRVTT DATASET. SECOND BEST IS UNDERLINED.

<table><tr><td>Methods</td><td>Year</td><td>R@1 ↑</td><td>R@5 ↑</td><td>R@10 ↑</td><td>MdR ↓</td><td>MnR ↓</td></tr><tr><td>CT-SAN [29]</td><td>2017</td><td>4.4</td><td>16.6</td><td>22.3</td><td>35.0</td><td>-</td></tr><tr><td>JSFusion [30]</td><td>2018</td><td>10.2</td><td>31.2</td><td>43.2</td><td>13.0</td><td>-</td></tr><tr><td>CE [31]</td><td>2019</td><td>20.9</td><td>48.8</td><td>62.4</td><td>6.0</td><td>28.2</td></tr><tr><td>MMT [32]</td><td>2020</td><td>24.6</td><td>54.0</td><td>67.1</td><td>4.0</td><td>-</td></tr><tr><td>TT-CE+ [33]</td><td>2021</td><td>29.6</td><td>61.6</td><td>74.2</td><td>3.0</td><td>-</td></tr><tr><td>Clip4Clip [4]</td><td>2022</td><td>42.1</td><td>71.9</td><td>81.4</td><td>2.0</td><td>15.7</td></tr><tr><td>ESAT [34]</td><td>2023</td><td>-</td><td>57.9</td><td>69.9</td><td>4.0</td><td>21.8</td></tr><tr><td>Zhu et al. [35]</td><td>2023</td><td>25.5</td><td>51.7</td><td>62.6</td><td>5.0</td><td>-</td></tr><tr><td>Align and Tell [36]</td><td>2023</td><td>43.2</td><td>72.3</td><td>81.5</td><td>2.0</td><td>-</td></tr><tr><td>Lee et. [37]</td><td>2024</td><td>28.7</td><td>58.6</td><td>70.9</td><td>4.0</td><td>-</td></tr><tr><td>UMP [20]</td><td>2024</td><td>43.7</td><td>70.0</td><td>79.7</td><td>-</td><td>15.2</td></tr><tr><td>LSDO [21]</td><td>2025</td><td>46.2</td><td>74.0</td><td>83.5</td><td>2.0</td><td>13.5</td></tr><tr><td>CMA [38]</td><td>2025</td><td>45.0</td><td>72.3</td><td>81.0</td><td>2.0</td><td>14.6</td></tr><tr><td>HTVR [39]</td><td>2025</td><td>48.0</td><td>74.7</td><td>83.7</td><td>2.0</td><td>12.9</td></tr><tr><td>VideoAligner [40]</td><td>2026</td><td>45.9</td><td>73.2</td><td>82.5</td><td>-</td><td>13.1</td></tr><tr><td>Ours</td><td>-</td><td> $49.4 \pm 0.12$ </td><td> $78.4 \pm 0.15$ </td><td> $84.3 \pm 0.10$ </td><td> $2.0 \pm 0.00$ </td><td> $12.3 \pm 0.11$ </td></tr></table>

TABLE III VIDEO-TEXT RETRIEVAL PERFORMANCE COMPARISON OF METHODS ON MSRVTT DATASET. SECOND BEST IS UNDERLINED.

<table><tr><td>Methods</td><td>Year</td><td>R@1 ↑</td><td>R@5 ↑</td><td>R@10 ↑</td><td>MdR ↓</td><td>MnR ↓</td></tr><tr><td>CE [31]</td><td>2019</td><td>20.6</td><td>50.3</td><td>64.0</td><td>5.3</td><td>25.1</td></tr><tr><td>MMT [32]</td><td>2020</td><td>24.4</td><td>56.0</td><td>67.8</td><td>4.0</td><td>-</td></tr><tr><td>TT-CE+ [33]</td><td>2021</td><td>32.1</td><td>62.7</td><td>75.0</td><td>3.0</td><td>-</td></tr><tr><td>Clip4Clip [4]</td><td>2022</td><td>43.1</td><td>70.5</td><td>81.2</td><td>3.0</td><td>-</td></tr><tr><td>ESAT [34]</td><td>2023</td><td>-</td><td>57.0</td><td>70.3</td><td>4.0</td><td>19.43</td></tr><tr><td>Zhu et al. [35]</td><td>2023</td><td>25.3</td><td>53.2</td><td>63.4</td><td>5.0</td><td>-</td></tr><tr><td>Align and Tell [36]</td><td>2023</td><td>41.6</td><td>69.3</td><td>78.6</td><td>2.0</td><td>-</td></tr><tr><td>UMP [20]</td><td>2024</td><td>44.8</td><td>69.9</td><td>80.6</td><td>-</td><td>11.0</td></tr><tr><td>CMA [38]</td><td>2025</td><td>44.6</td><td>72.1</td><td>83.6</td><td>2.0</td><td>9.8</td></tr><tr><td>Ours</td><td>-</td><td> $45.7 \pm 0.12$ </td><td> $72.9 \pm 0.14$ </td><td> $84.2 \pm 0.11$ </td><td> $2.0 \pm 0.00$ </td><td> $9.3 \pm 0.09$ </td></tr></table>

R@1. Introducing MLM improves this to 45.5%, indicating its effectiveness in capturing localized semantic cues. Similarly, PLM contributes 1.0% gain when used alone. When combined, the hybrid MLM+PLM configuration achieves 48.1%, confirming the benefit of dual-objective text encoding in modeling both local and global dependencies. These gains confirm that the hybrid MLM+PLM design is not merely a reuse of XLNetstyle PLM, but a retrieval-oriented adaptation. On the visual side, the introduction of CGAT alone (without hierarchical stages) further improves performance to 48.8%. This shows that fine-grained spatial reasoning and coarse-to-fine feature fusion enhance video understanding even without structural depth. Finally, incorporating the full hierarchical visual encoder results in the best overall performance, reaching 49.4% R@1. This confirms that our visual architecture contributes significantly by capturing multi-scale temporal-spatial features that better align with complex textual queries. Overall, the ablation study confirms that both sides of the architecture, hybrid text encoding and hierarchical visual modeling are essential for achieving robust video-text alignment and improving retrieval accuracy.

## E. SOTA Comparison

To comprehensively evaluate the effectiveness and generalizability of DREAM, we conduct extensive comparisons against recent state-of-the-art methods on standard videotext retrieval benchmarks. Our evaluation covers both text-tovideo and video-to-text retrieval tasks across three datasets: MSRVTT, MSVD and LSMDC. These benchmarks differ in scale and complexity, with MSRVTT offering broader diversity and richer annotations, while MSVD provides more concise descriptions and a smaller but well-curated set of video clips.

TABLE IV TEXT-VIDEO PERFORMANCE COMPARISON OF METHODS ON MSVD DATASETS. SECOND BEST IS UNDERLINED

<table><tr><td>Methods</td><td>Year</td><td>R@1 ↑</td><td>R@5 ↑</td><td>R@10 ↑</td><td>MdR ↓</td><td>MnR ↓</td></tr><tr><td>VSE [41]</td><td>2017</td><td>12.3</td><td>30.1</td><td>42.3</td><td>14.0</td><td>-</td></tr><tr><td>Multi Cues [42]</td><td>2018</td><td>20.3</td><td>47.8</td><td>61.1</td><td>6.0</td><td>-</td></tr><tr><td>CE [31]</td><td>2019</td><td>19.8</td><td>49.0</td><td>63.8</td><td>6.0</td><td>-</td></tr><tr><td>TT-CE+ [33]</td><td>2021</td><td>25.4</td><td>56.9</td><td>71.3</td><td>4.0</td><td>-</td></tr><tr><td>Clip4Clip [4]</td><td>2022</td><td>46.2</td><td>76.1</td><td>84.6</td><td>2.0</td><td>10</td></tr><tr><td>QB-Norm [43]</td><td>2022</td><td>47.6</td><td>77.6</td><td>86.1</td><td>2.0</td><td>-</td></tr><tr><td>Zhu et al. [35]</td><td>2023</td><td>25.5</td><td>51.7</td><td>62.6</td><td>5.0</td><td>-</td></tr><tr><td>Align and Tell [36]</td><td>2023</td><td>47.1</td><td>77.0</td><td>85.6</td><td>2.0</td><td>-</td></tr><tr><td>HTVR [39]</td><td>2025</td><td>47.7</td><td>75.6</td><td>85.5</td><td>2.0</td><td>9.9</td></tr><tr><td>LSDO [21]</td><td>2025</td><td>48.0</td><td>78.1</td><td>86.7</td><td>2.0</td><td>8.8</td></tr><tr><td>VideoAligner [40]</td><td>2026</td><td>48.5</td><td>78.2</td><td>86.5</td><td>-</td><td>8.6</td></tr><tr><td>Ours</td><td>-</td><td>49.7 ±0.15</td><td>79.1 ±0.14</td><td>87.3 ±0.12</td><td>2.0 ±0.00</td><td>8.5 ±0.07</td></tr></table>

TABLE V VIDEO-TEXT PERFORMANCE COMPARISON OF METHODS ON MSVD DATASETS.

<table><tr><td>Methods</td><td>Year</td><td>R@1 ↑</td><td>R@5 ↑</td><td>R@10 ↑</td><td>MdR ↓</td><td>MnR ↓</td></tr><tr><td>VSE [41]</td><td>2017</td><td>34.7</td><td>59.9</td><td>70.0</td><td>3.0</td><td>-</td></tr><tr><td>Multi Cues [42]</td><td>2018</td><td>31.5</td><td>51.0</td><td>61.5</td><td>5.0</td><td>-</td></tr><tr><td>TT-CE+ [33]</td><td>2021</td><td>27.1</td><td>55.3</td><td>67.1</td><td>4.0</td><td>-</td></tr><tr><td>Clip4Clip [4]</td><td>2022</td><td>56.6</td><td>79.7</td><td>84.3</td><td>1.0</td><td>7.6</td></tr><tr><td>Align and Tell [36]</td><td>2023</td><td>61.8</td><td>87.5</td><td>92.7</td><td>1.0</td><td>-</td></tr><tr><td>Ours</td><td>-</td><td>62.6 ±0.14</td><td>87.9 ±0.12</td><td>92.5 ±0.10</td><td>1.0 ±0.00</td><td>4.1 ±0.06</td></tr></table>

By benchmarking on both datasets, we aim to demonstrate the robustness of our model under varying retrieval scenarios.

MSRVTT: The results on MSRVTT are presented in Table II. Our method consistently outperforms all previous SOTA approaches. For the text-to-video retrieval task, our model obtains an R@1 score of 49.4%, surpassing the best previously reported result of 46.2% by LSDO [21]. It also achieves strong performance at R@5 (78.4%) and R@10 (84.3%), closely matching the best-performing baselines. The Median Rank (MdR) of 2 and a Mean Rank (MnR) of 12.3 further demonstrate the model’s ability to consistently retrieve relevant results early in the ranking list. These improvements can be attributed to our proposed CGAT mechanism, which refines spatial-temporal dependencies in the visual modality and enables the model to construct richer video representations. In contrast to prior approaches, such as CLIP4Clip [4] and UMP [20], which rely on static pooling or basic temporal modeling, our approach utilizes the hierarchical structure of video content to generate more discriminative embeddings.

In the video-to-text retrieval task on MSRVTT, our method in Table III again establishes a new performance benchmark, achieving an R@1 score of 45.7%, R@5 of 72.9%, and R@10 of 84.2%. These results outperform previous baselines by a significant margin. The model also maintains an MdR of 2 and reduces MnR to 9.3, the best reported on this dataset. These gains stem from the design of our hybrid language encoder, which combines MLM and PLM in a dual-stream attention setup. This structure allows the model to capture both localized semantic dependencies and global sentence structure, resulting in text embeddings that are more robust to ambiguity and better aligned with temporally structured video content.

MSVD: To further validate the generalizability of our proposed framework across datasets of varying scale and complexity, we conduct additional evaluations on the MSVD benchmark. Compared to MSRVTT, MSVD contains fewer training text-video pairs, making retrieval performance more sensitive to model generalization. As shown in Table IV, in the TVR setting, our model achieves an R@1 of 49.7%, R@5 of 79.1%, and R@10 of 87.3%. While LSDO [21] performs slightly lower at R@1 (48.0%), both methods achieve comparable MdR (2.0) and MnR (8.5). These results suggest that our model maintains robust performance even under data-constrained scenarios, and the small dataset size may exaggerate performance differences due to subtle variations in visual or linguistic representations.

TABLE VI TEXT-VIDEO PERFORMANCE COMPARISON OF METHODS ON LSMDC DATASETS.

<table><tr><td>Methods</td><td>Year</td><td>R@1 ↑</td><td>R@5 ↑</td><td>R@10 ↑</td><td>MdR ↓</td><td>MnR ↓</td></tr><tr><td>CT-SAN [29]</td><td>2017</td><td>5.1</td><td>16.3</td><td>25.2</td><td>46.0</td><td>-</td></tr><tr><td>JSFusion [30]</td><td>2018</td><td>9.1</td><td>21.2</td><td>34.1</td><td>36.0</td><td>-</td></tr><tr><td>CE [31]</td><td>2019</td><td>11.2</td><td>26.9</td><td>34.8</td><td>25.3</td><td>-</td></tr><tr><td>MMT [32]</td><td>2020</td><td>12.9</td><td>29.9</td><td>40.1</td><td>19.3</td><td>75.0</td></tr><tr><td>TT-CE+ [33]</td><td>2021</td><td>17.2</td><td>36.5</td><td>46.3</td><td>13.7</td><td>-</td></tr><tr><td>Clip4Clip [4]</td><td>2022</td><td>22.6</td><td>41.0</td><td>49.1</td><td>11.0</td><td>61.0</td></tr><tr><td>Align and Tell [36]</td><td>2023</td><td>23.1</td><td>41.2</td><td>49.6</td><td>11.0</td><td>-</td></tr><tr><td>ESAT [34]</td><td>2023</td><td>-</td><td>32.3</td><td>40.8</td><td>17.0</td><td>70.92</td></tr><tr><td>UMP [20]</td><td>2024</td><td>21.6</td><td>41.1</td><td>49.9</td><td>-</td><td>59.5</td></tr><tr><td>LSDO [21]</td><td>2025</td><td>26.4</td><td>44.8</td><td>55.1</td><td>8.0</td><td>52.1</td></tr><tr><td>Ours</td><td>-</td><td>27.3 ±0.11</td><td>44.6 ±0.13</td><td>56.4 ±0.12</td><td>8.0 ±0.00</td><td>49.6 ±0.17</td></tr></table>

TABLE VII VIDEO-TO-TEXT RETRIEVAL PERFORMANCE COMPARISON ON THE LSMDC DATASET.

<table><tr><td>Methods</td><td>Year</td><td>R@1 ↑</td><td>R@5 ↑</td><td>R@10 ↑</td><td>MdR ↓</td><td>MnR ↓</td></tr><tr><td>CLIP4Clip [4]</td><td>2022</td><td>20.6</td><td>39.4</td><td>47.5</td><td>13.0</td><td>56.7</td></tr><tr><td>Align and Tell [36]</td><td>2023</td><td>21.3</td><td>40.2</td><td>50.1</td><td>11.0</td><td>-</td></tr><tr><td>ESAT [34]</td><td>2023</td><td>-</td><td>31.9</td><td>42.2</td><td>18.0</td><td>71.61</td></tr><tr><td>UMP [20]</td><td>2024</td><td>22.5</td><td>40.7</td><td>50.8</td><td>-</td><td>51.7</td></tr><tr><td>Ours</td><td>-</td><td> $23.1 \pm 0.4$ </td><td> $40.9 \pm 0.5$ </td><td> $51.3 \pm 0.6$ </td><td> $10.0 \pm 0.0$ </td><td> $51.0 \pm 0.7$ </td></tr></table>

In Table V, for the video-to-text retrieval task, our method achieves the best overall performance, with an R@1 of 62.6%, R@5 of 87.9%, and R@10 of 92.5%. While Align and Tell [36] slightly outperforms our method on R@10 (92.7%), our model achieves a lower MnR of 4.1, highlighting its ability to retrieve relevant captions more consistently and with fewer errors. These improvements can be attributed to the synergy between our hierarchical visual encoder and hybrid text modeling approach, which together enhance semantic alignment across modalities.

LSMDC: We also evaluated our method on the LSMDC dataset, with results shown in Table VI for text-to-video retrieval and Table VII for video-to-text retrieval. In the textto-video task, our model achieves an R@1 score of 27.3%, outperforming LSDO (26.4%) and other recent baselines. We also report R@5 (44.6%) and R@10 (56.4%), demonstrating consistent performance across different retrieval ranks. In video-to-text retrieval, our model achieves R@1 of 23.1%, surpassing previous approaches such as CLIP4Clip (20.6%) and Align and Tell (21.3%), with strong performance at higher ranks (R@5: 40.9%, R@10: 51.3%). These results highlight the effectiveness of our dual-objective text encoder and hierarchical visual encoder in improving retrieval performance across both tasks on the LSMDC dataset.

## F. Qualitative Analysis:

To better understand how our proposed framework interprets visual information and aligns it with textual semantics, we conduct a qualitative analysis using attention heatmaps and token-frame alignment visualizations on samples from the MSRVTT and MSVD datasets. Fig. 5 presents attended regions across several video frames, each paired with its

![](images/635e94211f44fd6d72ee6613e6a576e42f25ebd4a90465dc1f1a4b475b7649cf.jpg)  
Fig. 5. Attention heatmaps from the final transformer layer, illustrating the model’s focus on semantically relevant regions. The left set of examples shows the model attending to key actions in a simulated emergency scenario involving human interaction with a mannequin, capturing motion and contact points. The right set highlights consistent attention on a glowing fireplace within a virtual Minecraft environment, despite variations in viewpoint. These visualizations demonstrate the effectiveness of the CGAT mechanism in localizing salient content, maintaining coherent focus across scene.

![](images/18d1cbbb9c906cfca9c0af71eca07af3179e97c9f1eec30e3a784ccc3642306b.jpg)

<details>
<summary>network diagram</summary>

| Category | Value |
|---|---|
| CLS | 0.21 |
| athlete | 0.23 |
| is | 0.18 |
| swinging | 0.17 |
| and | 0.16 |
| throwing | 0.23 |
| metal | 0.19 |
| ball | 0.22 |
| at | 0.17 |
| scottish games | 0.20 |
| EOT | 0.18 |
</details>

Fig. 6. Visualization of attention weights in the alignment module. We select three query embeddings and top-3 attention weights for clearer interpretation. The visualization illustrates how the model captures fine-grained temporal alignment between textual concepts and visual content.

![](images/5dc59e2ce4b7dd86c0b3afb6c2b890847ec27d4efee2edaea05663d2584a2b24.jpg)

<details>
<summary>text_image</summary>

Query: tourists walking around the mount fuji visitor center
LSDO
w/o PLM & MLM
DREAM
bbc talking to guys with very expensive red sports car
LSDO
a person is putting the vegetable in to the water and boil it
w/o PLM & MLM
DREAM
a basketball player runs and dunks a basketball
</details>

Fig. 7. Qualitative comparison of text-to-video retrieval results on the MSRVTT [25] dataset. Given each text query, we show the R@1 retrieved video produced by LSDO [21], a variant without MLM and PLM and the proposed DREAM model. Blue box highlights correct results.

corresponding attention activation map. The visualizations demonstrate that the CGAT mechanism consistently highlights semantically meaningful regions relevant to the textual query. For example, in the left sequence, the model successfully identifies and tracks a simulated emergency scenario involving a mannequin and human actors. Attention consistently focuses on critical regions such as the body, limbs, and points of interaction during actions like dragging or assisting, capturing both motion and human-object engagement. In contrast, the right sequence, taken from a Minecraft environment, shows the model’s attention remaining tightly centered on a glowing fireplace across varying camera angles and frames. This indicates the model’s robust ability to maintain spatial and temporal coherence, a key strength for aligning dynamic visual content with natural language queries.

To further examine how our model performs fine-grained cross-modal alignment, we present an additional visualization in Fig. 6, which shows the sentence “athlete is swinging and throwing a metal ball at the Scottish Games.” This example demonstrates the model’s capacity to decompose complex textual input into semantically meaningful components that align with temporally distributed video frames. The green head focuses on early-stage cues which correspond to the preparatory phase of the action. The orange head emphasizes the core motion “throwing” linked to the middle frames capturing the throw. The blue head attends to the outcome and context, which align with the final frames showing the airborne object and the event setting. The attention distribution across heads illustrates the model’s ability to distinguish between subject, action, and context within a video, and to associate each component of the query with the most relevant temporal segment.

In addition to attention visualizations, we performed qualitative comparison across semantic complexity, Fig. 7 presents a qualitative comparison of R@1 retrieval results between LSDO, a variant without MLM and PLM, and the proposed DREAM model across scenarios of increasing semantic complexity. The top row (Single Action) corresponds to relatively simple queries involving a single dominant activity, where all methods retrieve visually related videos, however, DREAM exhibits more consistent alignment with both the action and scene context, while baseline methods occasionally emphasize background content missing the tourists. The second row (left) illustrates Human-Object Interaction, where accurate retrieval requires understanding both the manipulated object and the associated action. In this case, LSDO retrieves videos containing relevant objects but fails to capture the precise interaction (e.g., boiling vegetables), whereas DREAM successfully retrieves videos reflecting both object presence and the intended manipulation. The second row (right) corresponds to a Multi-Actor Scene, which is the most challenging scenario due to interactions among multiple agents and temporally ordered actions. For the query “a basketball player runs and dunks a basketball,” LSDO retrieves a video where the dunk is performed without the running action, missing a key semantic component. The variant without MLM and PLM retrieves a running player however, the retrieved sport is volleyball rather than basketball, leading to semantic mismatch. In contrast, DREAM correctly retrieves a video capturing both the running motion and dunk action within the appropriate basketball context.

![](images/a45b5aafb658b8f999098e809d6c47ff83dd297388372b112ffc1cd62dd1c3f4.jpg)

<details>
<summary>text_image</summary>

Query: a little kid is playing in the bathtub with his toys
CLIP4Clip
LSDO
DREAM
Query: little kid playing with the toys in a bathtub his is
Helping in the war a soldier another soldier is
CLIP4Clip
LSDO
DREAM
</details>

Fig. 8. Qualitative comparison of text-to-video retrieval results on the MSRVTT [25] dataset. Given each text query, we show the R@1 retrieved video produced by CLIP4Clip [4], LSDO [21], and DREAM model. The left column presents the retrieval results with the original queries, while the right column shows the retrieval results after perturbing the input queries to observe the effect of permutation modeling. Blue box highlights correct results.

Further to this, Fig. 8 also presents a comparison of retrieval results between the original input query (left column) and the permuted query (right column). Despite changes in word order, DREAM consistently retrieves relevant videos, highlighting its robustness to query perturbations. In contrast, the baseline models and the variant without MLM and PLM objectives show noticeable performance degradation when the query is permuted, emphasizing the role of DREAM’s dual-objective modeling in maintaining semantic alignment despite linguistic variations.

Both visualizations demonstrate that our model’s hierarchical attention design and dual-modality encoding enable it to learn semantically grounded, temporally aware alignments that surpass the flat or diffuse attention patterns typical of prior approaches. Compared to conventional vision transformers or CLIP-based retrieval models, our method achieves sharper, more discriminative attention, reinforcing the superior alignment performance observed in our quantitative evaluations. While most attention patterns are interpretable and consistent, we note that in rare cases such as cluttered or fast-moving scenes attention may become more diffuse. These instances suggest directions for future work, including the integration of motion-specific priors and optical flow cues to further improve temporal reasoning and retrieval robustness.

## V. CONCLUSION

In this paper, we propose a novel framework for TVR that comprehensively tackles the semantic and temporal complexity of video content. Our model leverages a hybrid language encoder, combining masked and permuted language modeling to enrich textual representations, alongside a hierarchical vision encoder equipped with CGAT for fine-grained spatial-temporal feature extraction. This dual-modality design enables robust alignment between natural language queries and dynamic video sequences. Extensive experiments on the MSR-VTT, MSVD and LSMDC benchmarks demonstrate that our method outperforms previous state-of-the-art approaches in both retrieval accuracy and interpretability. We further validated the effectiveness of our approach through qualitative analyses, including heatmap visualizations and tokenframe attention alignments, which reveal that our model can consistently localize semantically meaningful content across time. These insights confirm the model’s ability to track complex actions and align them with linguistic structures, making it particularly suitable for real-world multimedia retrieval scenarios. Despite its strong performance, our model has certain limitations. In highly cluttered scenes or under poor lighting conditions, attention maps occasionally become diffuse, reducing alignment precision. Additionally, while the current framework captures temporal context using frame-level modeling, it does not explicitly incorporate motion features such as optical flow or velocity cues, which may limit its performance on fast-paced or subtle temporal events. Future work can explore the integration of motion-aware priors and temporal dynamics through video-specific modules, such as 3D convolutions and transformer-based temporal modeling. Furthermore, extending the model to handle multilingual queries, longer video sequences and exploring concept-level alignment and concept permutation/masking could improve the model’s ability to handle more abstract, high-level semantic relationships within videos, allowing for greater flexibility and robustness in text-video retrieval tasks.

## REFERENCES

[1] Y. Tang, J. Bi, S. Xu, L. Song, S. Liang, T. Wang, D. Zhang, J. An, J. Lin, R. Zhu et al., “Video understanding with large language models: A survey,” IEEE Transactions on Circuits and Systems for Video Technology, 2025.  
[2] H. Qu, R. Yan, X. Shu, H. Gao, P. Huang, and G.-S. Xie, “Mvp-shot: Multi-velocity progressive-alignment framework for few-shot action recognition,” IEEE Transactions on Multimedia, 2025.  
[3] Z. Ma, H. Chen, W. Zeng, L. Su, and S. Zhang, “Multi-modal reference learning for fine-grained text-to-image retrieval,” IEEE Transactions on Multimedia, 2025.  
[4] H. Luo, L. Ji, M. Zhong, Y. Chen, W. Lei, N. Duan, and T. Li, “Clip4clip: An empirical study of clip for end to end video clip retrieval and captioning,” Neurocomputing, vol. 508, pp. 293–304, 2022.  
[5] X. Jin, B. Zhang, W. Gong, K. Xu, X. Deng, P. Wang, Z. Zhang, X. Shen, and J. Feng, “Mv-adapter: Multimodal video transfer learning for video text retrieval,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 27 144–27 153.  
[6] H. Wang, F. Liu, L. Jiao, J. Wang, Z. Hao, S. Li, L. Li, P. Chen, and X. Liu, “Vilt-clip: Video and language tuning clip with multimodal prompt learning and scenario-guided optimization,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 38, no. 6, 2024, pp. 5390–5400.  
[7] H. Tang, M. Cao, J. Huang, R. Liu, P. Jin, G. Li, and X. Liang, “Muse: Mamba is efficient multi-scale learner for text-video retrieval,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 39, no. 7, 2025, pp. 7238–7246.  
[8] J. Li, D. Li, C. Xiong, and S. Hoi, “Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation,” in International conference on machine learning. PMLR, 2022, pp. 12 888–12 900.  
[9] X. Xu, C. Wu, S. Rosenman, V. Lal, W. Che, and N. Duan, “Bridgetower: Building bridges between encoders in vision-language representation learning,” in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 37, no. 9, 2023, pp. 10 637–10 647.  
[10] B. Zhang, P. Zhang, X. Dong, Y. Zang, and J. Wang, “Long-clip: Unlocking the long-text capability of clip,” in European Conference on Computer Vision. Springer, 2024, pp. 310–325.  
[11] I. Croitoru, S.-V. Bogolin, M. Leordeanu, H. Jin, A. Zisserman, S. Albanie, and Y. Liu, “Teachtext: Crossmodal generalized distillation for text-video retrieval,” in Proceedings of the IEEE/CVF international conference on computer vision, 2021, pp. 11 583–11 593.  
[12] P. Chandrika and C. Jawahar, “Multi modal semantic indexing for image retrieval,” in Proceedings of the ACM International Conference on Image and Video Retrieval, 2010, pp. 342–349.  
[13] J. Shao, Z. Zhao, F. Su, and T. Yue, “Towards improving canonical correlation analysis for cross-modal retrieval,” in Proceedings of the on Thematic Workshops of ACM Multimedia 2017, 2017, pp. 332–339.  
[14] S. Wang, D. Guo, X. Xu, L. Zhuo, and M. Wang, “Cross-modality retrieval by joint correlation learning,” ACM Transactions on Multimedia Computing, Communications, and Applications (TOMM), vol. 15, no. 2s, pp. 1–16, 2019.  
[15] A. Karpathy and L. Fei-Fei, “Deep visual-semantic alignments for generating image descriptions,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 3128–3137.  
[16] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.  
[17] K. Simonyan and A. Zisserman, “Two-stream convolutional networks for action recognition in videos,” Advances in neural information processing systems, vol. 27, 2014.  
[18] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Paluri, “Learning spatiotemporal features with 3d convolutional networks,” in Proceedings of the IEEE international conference on computer vision, 2015, pp. 4489–4497.  
[19] G. Bertasius, H. Wang, and L. Torresani, “Is space-time attention all you need for video understanding?” in ICML, vol. 2, no. 3, 2021, p. 4.  
[20] H. Zhang, P. Zeng, L. Gao, J. Song, and H. T. Shen, “Ump: Unified modality-aware prompt tuning for text-video retrieval,” IEEE Transactions on Circuits and Systems for Video Technology, 2024.  
[21] Y. Zheng, B. Huang, Z. Chen, and D. Yu, “Enhancing text-video retrieval performance with low-salient but discriminative objects,” IEEE Transactions on Image Processing, 2025.  
[22] Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. R. Salakhutdinov, and Q. V. Le, “Xlnet: Generalized autoregressive pretraining for language understanding,” Advances in neural information processing systems, vol. 32, 2019.  
[23] A. Paszke, “Pytorch: An imperative style, high-performance deep learning library,” arXiv preprint arXiv:1912.01703, 2019.  
[24] D. P. Kingma and J. Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.  
[25] J. Xu, T. Mei, T. Yao, and Y. Rui, “Msr-vtt: A large video description dataset for bridging video and language,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2016, pp. 5288– 5296.  
[26] D. Chen and W. B. Dolan, “Collecting highly parallel data for paraphrase evaluation,” in Proceedings of the 49th annual meeting of the association for computational linguistics: human language technologies, 2011, pp. 190–200.  
[27] A. Rohrbach, M. Rohrbach, and B. Schiele, “The long-short story of movie description,” in German conference on pattern recognition. Springer, 2015, pp. 209–221.  
[28] Y. Yu, J. Kim, and G. Kim, “A joint sequence fusion model for video question answering and retrieval,” in Proceedings of the European conference on computer vision (ECCV), 2018, pp. 471–487.  
[29] Y. Yu, H. Ko, J. Choi, and G. Kim, “End-to-end concept word detection for video captioning, retrieval, and question answering,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 3165–3173.  
[30] Y. Yu, J. Kim, and G. Kim, “A joint sequence fusion model for video question answering and retrieval,” in Proceedings of the European Conference on Computer Vision (ECCV), September 2018.  
[31] Y. Liu, S. Albanie, A. Nagrani, and A. Zisserman, “Use what you have: Video retrieval using representations from collaborative experts,” arXiv preprint arXiv:1907.13487, 2019.  
[32] V. Gabeur, C. Sun, K. Alahari, and C. Schmid, “Multi-modal transformer for video retrieval,” in European Conference on Computer Vision. Springer, 2020, pp. 214–229.  
[33] I. Croitoru, S.-V. Bogolin, M. Leordeanu, H. Jin, A. Zisserman, S. Albanie, and Y. Liu, “Teachtext: Crossmodal generalized distillation for text-video retrieval,” in Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), October 2021, pp. 11 583– 11 593.  
[34] X. Wu, J. Qian, and T. Wang, “Text-video retrieval method based on enhanced self-attention and multi-task learning,” Multimedia Tools and Applications, vol. 82, no. 16, pp. 24 387–24 406, 2023.  
[35] J. Zhu, P. Zeng, L. Gao, G. Li, D. Liao, and J. Song, “Complementarityaware space learning for video-text retrieval,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 8, pp. 4362– 4374, 2023.  
[36] X. Wang, L. Zhu, Z. Zheng, M. Xu, and Y. Yang, “Align and tell: Boosting text-video retrieval with local alignment and fine-grained supervision,” IEEE Transactions on Multimedia, vol. 25, pp. 6079–6089, 2022.  
[37] S. Lee, H.-I. Kim, and Y. M. Ro, “Text-guided distillation learning to diversify video embeddings for text-video retrieval,” Pattern Recognition, vol. 156, p. 110754, 2024.  
[38] H. Jiang, J. Zhang, R. Huang, C. Ge, Z. Ni, S. Song, and G. Huang, “Cross-modal adapter for vision–language retrieval,” Pattern Recognition, vol. 159, p. 111144, 2025.  
[39] D. Zhang, Z. Wang, Z. Hu, and X.-J. Wu, “Htvr: Hierarchical textto-video retrieval based on relative similarity,” Pattern Recognition, p. 112145, 2025.  
[40] Z. Feng, S. Mao, Y. Wang, and S. Zhang, “Videoaligner: Text-driven feature decomposition for precise video-text alignment,” Pattern Recognition, p. 113971, 2026.  
[41] F. Faghri, D. J. Fleet, J. R. Kiros, and S. Fidler, “Vse++: Improving visual-semantic embeddings with hard negatives,” arXiv preprint arXiv:1707.05612, 2017.  
[42] N. C. Mithun, J. Li, F. Metze, and A. K. Roy-Chowdhury, “Learning joint embedding with multimodal cues for cross-modal video-text retrieval,” in Proceedings of the 2018 ACM on international conference on multimedia retrieval, 2018, pp. 19–27.  
[43] S.-V. Bogolin, I. Croitoru, H. Jin, Y. Liu, and S. Albanie, “Cross modal retrieval with querybank normalisation,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 5194–5205.