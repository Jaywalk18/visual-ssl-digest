# Vortex: Multi-Modal Fusion System for Intelligent Video Retrieval

Duc-Tho Nguyen1,2 ⋆, Hieu-Hoc Tran-Minh1,2 , Khanh-Hoa Lam1,2 Hoang-Nhut Ly1,2 , Huu-Phuc Huynh1,2 , Thanh-Tien Tran1,2 , and Trung-Nghia Le1,2

1 University of Science, VNU-HCM, Ho Chi Minh City, Vietnam

2 Vietnam National University, Ho Chi Minh City, Vietnam

Abstract. This paper presents Vortex, the multimodal video retrieval system developed by our team, FocusOnFun, for the Ho Chi Minh City AI Challenge 2025, designed to advance intelligent multimedia search and temporal reasoning. The system integrates adaptive keyframe extraction, multimodal metadata generation from vision-language and speech models, and a hybrid retrieval strategy that fuses CLIP and SigLIP2 embeddings through Reciprocal Rank Fusion to balance global and fine-grained semantics. To enhance interactivity, Vortex incorporates Rocchio-based relevance feedback and a multi-stage temporal search mechanism for sequential event alignment. Built on Milvus and Elasticsearch, the architecture enables scalable indexing and efficient retrieval. Evaluated in the official competition, our FocusOnFun team’s system achieved a score of 79.6/88 (90.5%) in the Preliminary Round and was further evaluated in the Final Round, achieving an ‘Excellent’ overall performance with ‘Outstanding’ results in the question-answering (QA) task. This demonstrating the complementary strengths of CLIP and SigLIP2 and confirming the effectiveness of the hybrid retrieval approach. The system establishes a robust foundation for future research in intelligent, context-aware, and interactive video retrieval.

Keywords: Lifelog Event Retrieval · Video Retrieval System · Relevance Feedback · Interactive Retrieval System · Multimodal Search · Reciprocal Rank Fusion · Large Language Models

## 1 Introduction

The Ho Chi Minh AI Challenge (AIC) [5] is a recurring scientific competition in Vietnam dedicated to advancing intelligent multimedia retrieval. Its design closely follows and aligns with leading international benchmarks, including the Video Browser Showdown (VBS) [11] and the Lifelog Search Challenge (LSC) [13]. The competition focuses on building intelligent assistant systems capable of performing deep semantic analysis and retrieval across large-scale multimedia databases.

The AIC’25 Final Round featured four distinct search tasks. The first is Textual Known-Item Search (Textual KIS), a core task in which systems must locate a specific video segment based on a natural language description. The second, introduced in the final round, is Video Known-Item Search (Video KIS), which requires finding a segment based on a short query video clip. The third task, Question Answering (Q&A), moves beyond traditional retrieval by requiring systems to not only identify the relevant video segment but also comprehend its content and generate a precise textual answer. The fourth and most complex task, Temporal Retrieval and Alignment of Key Events (TRAKE), challenges systems to retrieve a video containing an entire sequence of described events and to accurately align each event with its corresponding semantic keyframe.

In this paper, we present Vortex, a comprehensive multi-modal video retrieval system, for the AIC’25. Designed as an end-to-end solution, Vortex integrates an adaptive keyframe extraction pipeline (AutoShot [18] with L2-norm filtering) to efficiently manage extensive video datasets by minimizing redundancy while preserving essential visual information. To enable deep content-based retrieval, the system generates rich multi-modal metadata. Qwen2.5-VL [2] extracts textual cues from video frames through optical character recognition and captioning, while Whisper [10] provides temporally aligned automatic speech recognition transcriptions. We propose a hybrid retrieval strategy that balances search breadth and precision by generating dual embeddings using CLIP [9] for global semantic context and SigLIP2 [15] for fine-grained detail recognition. Their results are combined through Reciprocal Rank Fusion (RRF) [4] to produce an optimized ranking. To address the competition’s most challenging queries, Vortex also integrates advanced interactive components such as a multi-stage Temporal Search for sequential "before, main, and after" event alignment and an interactive Relevance Feedback loop based on the Rocchio algorithm [12], enabling users to iteratively refine their searches and achieve higher retrieval accuracy across all task categories.

We participated in all four tasks of AIC’25 as the FocusOnFun team, achieved a score of 79.6/88 (90.5%) in the Preliminary Round and achieved excellent overall performance in the Final Round. Our system demonstrated strong robustness, particularly in handling new and diverse query types. For the Video KIS task, our rich metadata pipeline performed exceptionally well. The system generated natural-language descriptions for each query video and leveraged high-quality OCR, ASR, and object-filtering modules to accurately locate the target content, resulting in excellent performance. We also attained very good results on the TRAKE task. Most notably, we achieved outstanding performance on the Q&A task, underscoring the effectiveness of our proposed system.

Our main contributions are summarized as follows:

– We propose Vortex, an end-to-end multi-modal video retrieval framework designed to address all four tasks of the Ho Chi Minh City AI Challenge 2025, integrating efficient data processing, semantic understanding, and interactive retrieval within a unified system.

– We develop a two-stage keyframe extraction pipeline using AutoShot and $L _ { 2 }$ -norm filtering to efficiently capture representative visual content.  
– We build a multi-modal metadata preprocessing pipeline employing Qwen2.5- VL for OCR and captioning, and Whisper for aligned audio transcriptions.  
– We introduce a hybrid retrieval strategy that fuses dual embeddings from CLIP and SigLIP2 via RRF for improved ranking accuracy.  
– We propose interactive modules, including multi-stage Temporal Search and Rocchio-based Relevance Feedback, to enhance complex and iterative queries.

## 2 Related Work

The field of interactive multimedia retrieval has advanced considerably through benchmarking initiatives such as the LSC and VBS, which evaluate systems for event retrieval from large multimodal datasets using textual, visual, and ad hoc queries [5, 11]. LSC focuses on personal multimodal archives, emphasizing real-time interactive search across Known-Item Search (KIS), Question Answering (QA), and ad hoc tasks [1]. Recent reviews highlight key trends, including embedding-based retrieval (e.g., CLIP, BLIP), Large Language Model (LLM) integration for query optimization, and improved user interfaces for temporal and collaborative search [13]. In LSC’25, systems such as U-Cker further enhanced retrieval through LLM-driven query refinement and temporal reasoning, improving performance on ambiguous lifelog queries [16].

VBS targets large-scale video archives with tasks involving textual KIS, visual KIS, and QA [11]. The 2025 results demonstrate the effectiveness of hybrid vision-language models for contextual understanding, with top systems like ViFi leveraging SigLIP for precise multimodal matching [8]. Other notable examples, such as VideoEase, integrated ASR, OCR, and temporal reranking for complex sequence retrieval [14], while VITRIVR by Heller et al. [7] exemplified effective temporal query mechanisms.

Building on these foundations, the previous AIC’24 adopted similar principles, emphasizing Vietnamese-language data and LLM-assisted query expansion. Systems like NewsInsight2.0 focused on LLM-based query optimization and temporal algorithms [17]. Notably, top-performing systems at AIC’24 introduced multimodal fusion, particularly through CLIP–BEiT-3 hybridization, along with shot-based reranking and LLM-driven query optimization to handle temporal reasoning and ambiguity [3, 6].

Vortex extends these prior efforts by unifying key advances while introducing novel components. Instead of the established CLIP-BEiT-3 fusion, we propose a new hybrid retrieval strategy pairing CLIP with SigLIP2 [15]. SigLIP2, a state-of-the-art model, offers exceptional fine-grained detail recognition and localization, providing a powerful complement to CLIP’s global semantic understanding. Furthermore, Vortex integrates this novel embedding fusion with a complete interactive loop, combining a multi-stage Temporal Search (Section 3.5) for sequential queries with a Rocchio-based Relevance Feedback mechanism (Section 3.4). This combination allows our system to not only execute complex temporal and semantic searches but also to iteratively refine results based on direct user feedback, achieving robust and adaptive performance.

![](images/53e5668fb80f880e03321ec489842c35248634ebdc016a4216b3068321d2b867.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Query"] --> B["User interface"]
  C["Feedback"] --> D["Filter"]
  B --> E["Text"]
  B --> F["Image"]
  B --> G["Temporal Search Input"]
  G --> H["Previous event"]
  G --> I["Current event"]
  G --> J["Next event"]
  D --> K["elasticsearch"]
  K --> L["Retrieval Module"]
  L --> M["Temporal Re-ranking"]
  M --> N["Result"]
  N --> O["Cache"]
  O --> P["redis"]
  P --> Q["Milvus"]
  Q --> R["Positive Images"]
  Q --> S["Negative Images"]
  R --> T["Feedback"]
```
</details>

Fig. 1: Overall system architecture with two main workflows. (A) Query Processing supports text, image, temporal, and filter queries. Temporal queries include three text fields for previous, current, and next events, processed through the Retrieval Module and refined using the Temporal Re-ranking algorithm. Filter queries access scene descriptions or OCR metadata via Elasticsearch, and final results are cached in Redis for efficiency. (B) Feedback Loop allows users to label retrieved images as positive or negative, with feedback stored in Milvus to enhance retrieval accuracy.

## 3 Proposed System

## 3.1 System Architecture

Our system is a comprehensive multimodal video retrieval platform designed for complex, large-scale search tasks. It supports both text and image queries, integrating hybrid vector search, metadata filtering, and advanced temporal and interactive retrieval. The overall architecture is shown in Fig. 1.

At its core, the system uses a dual-database backend: Milvus for high-dimensional vector search and Elasticsearch for text indexing and metadata filtering, with Redis providing low-latency caching. The workflow consists of two main phases. In the Preprocessing and Indexing phase, the system extracts keyframes (AutoShot with norm-based filtering), generates multimodal metadata (Qwen2.5-VL for OCR and captions, Whisper for ASR), and indexes CLIP and SigLIP2 embeddings. The Search and Interaction phase handles user queries through the Retrieval Module, which performs hybrid search with RRF, metadata filtering, temporal re-ranking, and Rocchio-based relevance feedback for efficient and adaptive retrieval.

![](images/9a2c05f63a9bb43526a5006ebde1b947786c24b1bdf19e7725ed9ab219958138.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Video"] --> B["Audio extraction"]
  B --> C["Audio"]
  C --> D["Whisper"]
  D --> E["Transcriptions"]
  E --> F["Qwen2.5-VL"]
  F --> G["Metadata (OCR, Description)"]
  G --> H["1152-dim"]
  H --> I["CLIP"]
  I --> J["SigLIP 2"]
  J --> K["1024-dim"]
  K --> L["Indexing"]
  L --> M["Milvus"]
  F --> N["Indexing"]
  N --> O["elasticsearch"]
  C --> P["Autoshot"]
  P --> Q["Shots"]
  Q --> R["Keyframe selection"]
  R --> S["Keyframe"]
  S --> T["Keyframe"]
  T --> U["SigLIP 2"]
  U --> V["1024-dim"]
  V --> W["Indexing"]
  W --> X["Milvus"]
```
</details>

Fig. 2: Data pre-processing pipeline.

## 3.2 Data Pre-processing

The overall data pre-processing pipeline is shown in Fig. 2, including three main steps:

Keyframe Extraction. Efficient handling of data plays a vital role in the performance of video retrieval systems, particularly when dealing with large-scale video collections. Analyzing every single frame within a video is highly resourceconsuming, resulting in unnecessary computational overhead, increased storage demand, and reduced retrieval efficiency. Therefore, we employed an optimized two-stage keyframe selection strategy, inspired by the TycheVid team [6].

First, we utilize AutoShot [18] to segment the video into distinct shots, enabling us to focus on semantically consistent parts of the video.

Second, within each shot, we apply an optimized keyframe selection algorithm. We use the CLIP model (ViT-L-14-quickgelu pre-trained with DFN2B) for semantic feature extraction. To optimize performance, features (embeddings) are extracted from every eighth frame within the shot. Let $\mathbf { e } _ { \mathrm { c u r r e n t } }$ be the embedding vector of the current sampled frame and $\mathbf { e } _ { \mathrm { p r e v } }$ be the vector of the last retained keyframe.

To quantify the visual change, we calculate the relative difference, rel\_diff, using the Euclidean $\left( L _ { 2 } \right)$ norm:

$$
\text { rel\_diff } = \frac {\| \mathbf {e} _ {\text { current }} - \mathbf {e} _ {\text { prev }} \|}{\| \mathbf {e} _ {\text { prev }} \|}. \tag {1}
$$

A frame is retained as a new keyframe only if the computed rel\_diff (Eq. 1) exceeds 0.4, a threshold empirically determined to yield effective results.

This two-stage strategy represents a deliberate trade-off between efficiency and granularity. The initial sampling of every eighth frame is a performance optimization to manage large-scale data. The adaptive component is the subsequent L2-norm filtering. This filter ensures that only frames representing a significant visual change from this sampled set are retained, thus avoiding high redundancy in static scenes. While it is possible for a highly transient event occurring between the 8-frame intervals to be missed, this approach was calibrated to prioritize computational efficiency and robustly capture major scene changes, which proved highly effective for the event-level retrieval tasks in the competition.

![](images/ed699b9c8a8a285638b4669cfd121fab25e289a6367c0701b0c89102214d0551.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Text"] --> C["Feature extraction"]
  B["Image"] --> C
  C --> D["SigLIP 2\n1152-dims"]
  C --> E["CLIP\n1024-dims"]
  D --> F["Database"]
  E --> F
  F --> G["Ranked candidates"]
  G --> H["Rerank with RRF"]
  H --> I["Top k"]
```
</details>

Fig. 3: Retrieval Module processes both text and image queries. SigLIP2 and CLIP query embeddings are stored in Milvus Retrieved candidates are then reranked using Reciprocal Rank Fusion (RRF), obtaining top-k results.

Multi-modal Feature Extraction. Extracted keyframes are processed through a multimodal pipeline to generate metadata and embeddings. The Qwen2.5- VL-3B-Instruct model performs OCR and captioning, offering a strong tradeoff between accuracy and efficiency [2]. For embedding generation, we employ CLIP (DFN5B) for global semantic context and SigLIP2 for fine-grained recognition [9, 15]. These complementary models enhance retrieval flexibility across diverse query types.

Automatic Speech Recognition. Audio features are extracted using Whisper [10], which outputs timestamped transcriptions. To align them with keyframes, each frame timestamp is matched to transcription intervals (a, b) where $a \leq t _ { k } \leq$ b. The last spoken text is propagated through silent gaps, ensuring full coverage. The resulting transcription field enriches each keyframe’s metadata with synchronized speech content.

## 3.3 Retrieval with Reciprocal Rank Fusion

Our system employs hybrid embeddings from CLIP and SigLIP2 for multimodal retrieval. Each user query, whether text or image, is embedded using both models, producing two independent similarity searches in Milvus and generating separate ranked lists of keyframes.

To merge these results into a single, more accurate ranking, we apply RRF [4], an effective and robust data-fusion method. For each keyframe d, the RRF score is computed as:

$$
\operatorname{RRF} _ {-} \operatorname{Score} (d) = \sum_ {i = 1} ^ {N} \frac {1}{k + \operatorname{rank} _ {i} (d)}, \tag {2}
$$

where $N = 2$ (CLIP and SigLIP2) and k is a constant $( \mathrm { e . g . , 6 0 } )$ that reduces sensitivity to lower-ranked results. Keyframes are then re-ranked by descending RRF\_Score(d). This approach combines $\mathrm { C L I P } \mathrm { { ^ { \circ } s } }$ global semantic understanding with SigLIP2’s fine-grained recognition, substantially improving retrieval relevance and robustness.

## 3.4 Query Refinement with Relevance Feedback

To enhance interactivity and support iterative refinement, we implemented a relevance feedback mechanism based on the Rocchio algorithm [12], a classical approach within the Vector Space Model. The algorithm updates the original query vector $\bf ( q _ { 0 } )$ by shifting it toward the centroid of relevant keyframes and away from the centroid of non-relevant ones:

$$
\boldsymbol {q} _ {\boldsymbol {m}} = \alpha \boldsymbol {q} _ {\boldsymbol {0}} + \beta \frac {1}{| C _ {r} |} \sum_ {\boldsymbol {d} _ {j} \in C _ {r}} \boldsymbol {d} _ {j} - \gamma \frac {1}{| C _ {n r} |} \sum_ {\boldsymbol {d} _ {j} \in C _ {n r}} \boldsymbol {d} _ {j}, \tag {3}
$$

where $\alpha , \beta ,$ and $\gamma$ control the influence of the original query, relevant set $\left( C _ { r } \right)$ , and non-relevant set $\left( C _ { n r } \right)$ .

In our system’s user interface, after an initial search, the user can provide explicit feedback by selecting "Prefer this answer" (like) or "Not prefer" (dislike) for any number of the returned images (keyframes). The system gathers all "liked" keyframes into the relevant set $\left( C _ { r } \right)$ and all "disliked" keyframes into the non-relevant set $\left( C _ { n r } \right)$ .

This feedback is used to calculate the new query vector $\mathbf { \delta } \mathbf { q } _ { m }$ using Equation 3. This refined vector is then used to re-query the Milvus database, presenting the user with a new list of results that is more closely aligned with their stated preferences.

## 3.5 Temporal Search

Standard retrieval systems often fail when handling complex temporal queries, such as those seen in the TRAKE challenge, as well as complex Known-Item Search (KIS) queries where the query itself defines a target frame based on its surrounding temporal context. These queries require the system to not only find individual events but also to verify that they occur in a specific sequence (e.g., an event A happening before an event B) within the same video. A simple, single-stage search is insufficient for this task.

To address this, we developed a multi-stage temporal re-ranking algorithm (Algorithm 1). Our approach requires the user to decompose their query into three distinct components: a current query $\left( Q _ { c u r r e n t } \right)$ , a previous query $( Q _ { p r e v i o u s } )$ , and a next query $( Q _ { n e x t } )$ .

Our algorithm processes these inputs as three independent searches. First, the system retrieves three separate ranked lists of keyframes: $R _ { c u r r e n t } , R _ { p r e v i o u s } ,$ and $R _ { n e x t }$ . The final ranking is then determined by re-scoring the main results in Rcurrent. $R _ { c u r r e n t }$

For each current result $r _ { c }$ (which belongs to a specific video\_ID) in the $R _ { c u r r e n t }$ list, the algorithm searches the $R _ { p r e v i o u s }$ and $R _ { n e x t }$ lists to find the highest-scoring keyframes, $r _ { p }$ and $r _ { n }$ , that belong to the exact same video\_ID.

A new, temporally-boosted score $( S _ { f i n a l } )$ is then calculated for $r _ { c }$ by summing its original score with the highest scores found for its corresponding "previous" and "next" keyframes (if any exist within that same video). This scoring can be formalized as:

$$
S _ {f i n a l} (r _ {c}) = S (r _ {c}) + S _ {\max} (r _ {p} \in \text { video\_ID }) + S _ {\max} (r _ {n} \in \text { video\_ID }). \tag {4}
$$

If no matching previous or next keyframes are found in that video, their respective $S _ { \mathrm { m a x } }$ scores are treated as zero.

Finally, the $R _ { c u r r e n t }$ list is re-sorted in descending order based on this new $S _ { f i n a l }$ . This re-ranking method ensures that a video containing the entire described sequence of events (Previous → Current → Next) will receive a significantly higher aggregate score, thus directly addressing the temporal alignment challenge and improving retrieval accuracy for complex sequential queries.

Here is the detailed complexity analysis of the multi-stage temporal reranking mechanism (Algorithm 1). Let K be the number of top candidates retrieved from each of the three independent searches (for $Q _ { p r e v i o u s } , Q _ { c u r r e n t }$ , and $Q _ { n e x t } )$ , which is a value set by the system.

– Step 1 (Independent Retrieval): The Milvus HNSW index is employed to perform fast and high-recall Approximate Nearest Neighbor (ANN) retrieval, serving as the baseline computational cost for this step.  
– Step 2 (Compute Best Scores): This step iterates through $R _ { p r e v i o u s }$ (K items) and $R _ { n e x t }$ (K items). To ensure high performance, we implement bestPrev and bestNext as hash maps (or dictionaries). We build these maps by iterating through the two lists, a process with a time complexity of $O ( K ) +$ $O ( K ) = O ( K )$ .  
– Step 3 (Temporal Re-scoring): This step iterates through the main $R _ { c u r r e n t }$ list (K items). For each candidate $r _ { c } .$ , retrieving its corresponding bestPrev[video\_id] and bestNext[video\_id] scores from the hash maps is an O(1) operation on average. Therefore, the total complexity for this step is $O ( K )$ ).  
– Step 4 (Re-ranking): The final step sorts the re-scored list $R _ { c u r r e n t } ^ { * }$ , which has a size of K. This sorting operation has a time complexity of O(K log K). Thus, the total additional computational overhead of our entire temporal reranking algorithm (Steps 2-4) is dominated by the final sorting step, resulting in a complexity of O(K log K). This is a negligible cost compared to the initial ANN retrieval (Step 1) and confirms the algorithm’s suitability for a real-time, interactive system.

The heuristic re-ranking approach is intentionally chosen over classical Dynamic Programming (DP) methods. Traditional DP algorithms are designed for dense alignment between two known, finite sequences and often incur substantial computational overhead, making them impractical to apply across the entire database in an interactive retrieval setting.

Algorithm 1 Multi-Stage Temporal Re-ranking for Sequential Queries  
Require: Queries: $Q_{previous}$ , $Q_{current}$ , $Q_{next}$ .  
Ensure: Re-ranked list $R_{current}^{*}$ with temporally boosted scores.

1: Step 1: Independent Retrieval
2: $R_{previous} \leftarrow \text{Search}(Q_{previous})$ 3: $R_{current} \leftarrow \text{Search}(Q_{current})$ 4: $R_{next} \leftarrow \text{Search}(Q_{next})$ 5: Step 2: Compute Best Scores per Video
6: for all (video_id, score) in $R_{previous}$ do
7:    bestPrev[video_id] ← max(bestPrev[video_id], score)
8: end for
9: for all (video_id, score) in $R_{next}$ do
10:    bestNext[video_id] ← max(bestNext[video_id], score)
11: end for
12: Step 3: Temporal Re-scoring
13: for all (video_id, frame_id, S(r_c)) in $R_{current}$ do
14: $S_{final}(r_c) \leftarrow S(r_c) + \text{bestPrev}[video\_id] + \text{bestNext}[video\_id]$ 15:    Append (video_id, frame_id, S_final(r_c)) to $R^{*}_{current}$ 16: end for
17: Step 4: Re-ranking and Output
18: Sort $R^{*}_{current}$ in descending order of S_final
19: return $R^{*}_{current}$

Our task, which includes both the Temporal Retrieval and Alignment of Key Events (TRAKE) and complex sequential KIS queries, requires finding a sparse sequence of high-level semantic events ("Before", "Now", "After") within a massive video database.

Our method acts as a lightweight and effective heuristic. It leverages the speed of vector retrieval to find candidate events independently and then applies an O(K log K) re-ranking process to boost the scores of videos that contain the complete described sequence. This two-stage design - fast retrieval followed by efficient re-ranking - is far more scalable and better suited for an interactive retrieval system than a complex DP-based alignment that would be computationally infeasible to run against the entire database.

## 3.6 LLM-Assisted Query Interpretation

While many contemporary retrieval systems employ Large Language Models (LLMs) for automatic query rewriting, this approach introduces risks such as intent drift and hallucination, where the rewritten query may diverge from the user’s original goal. Given the real-time, interactive nature of the competition, maintaining user control and query fidelity is crucial.

![](images/1afc6b01d23c6c03843b92d109dce493811eb4e6712090cbc1baa4e764412de7.jpg)

<details>
<summary>text_image</summary>

ID: 52e3a371
Building
RE-SEARCH
Search or paste an image...
L22_V007
L21_V006
K05_V025
K10_V005
L21_V012
K11_V020
K06_V018
L29_V002
K13_V005
L22_V006
L22_V006
L22_V006
</details>

Fig. 4: Overview of the system user interface. The left sidebar displays the query details and search management options, including the query ID and a re-search button. The right panel contains the search bar for text or image input at the top and presents the retrieved video keyframes as a grid of ranked results below.

![](images/148211ae0e1ae7bc4345afbf1fcc369ed6e69de44709374fa260ae11d3490948.jpg)

<details>
<summary>text_image</summary>

ID: e882c7c6
Temporal search: Before: "House",
Now: "Tree", After: "Girl"
RE-SEARCH
Before
House
Now
Tree
After
Girl
SEARCH TEMPORAL
/L27_V015/004809 (3 frames)
4809
3777
5137
/L27_V015/004745 (3 frames)
4745
3777
5137
</details>

Fig. 5: Temporal search mode interface, featuring three input fields for specifying temporal relations: Before, Now, and After.

To mitigate this, Vortex implements a hybrid, iterative refinement architecture. The LLM functions as a Query Interpretation Assistant rather than an autonomous rewriter. Instead of silently modifying the query, the system analyzes the user’s input and proposes several explicit suggestions back to the user. This ensures the user retains full control and the query is not altered without their consent. This strategy is particularly effective for resolving semantic ambiguity. For instance, given a succinct or ambiguous query (e.g., “building”), the LLM generates specific alternatives (e.g., “a tall office building,” “a building under construction,” “a university campus building”). The user can then select the suggestion that best matches their intent, creating an iterative feedback loop that combines the LLM’s semantic power with explicit user guidance.

This design establishes a multi-layer refinement workflow. It combines LLMassisted suggestions before the search (pre-query refinement) with Rocchio-based Relevance Feedback after the search (post-query refinement), providing a robust, transparent, and user-controlled retrieval experience.

## 3.7 User interface

To ensure efficient user interaction, we designed a clean and intuitive interface (Fig. 4) with a left sidebar for navigation and a right panel for queries and visualization. The sidebar manages search sessions, history, and retrieval modes such as CLIP-only, SigLIP2-only, or the default RRF mode. The main panel provides a unified query bar for text and image inputs and displays ranked keyframes as search results. A dedicated Temporal Search Mode (Fig. 5) supports complex sequential queries through three input fields labeled Before, Now, and After, allowing users to specify event order. This integrated design supports both standard and temporal search workflows seamlessly.

## 4 Experiments

## 4.1 Dataset Overview

Our system was evaluated on the official dataset of the $\mathrm { A I C ^ { 2 5 ^ { 3 } } }$ . The dataset includes videos from major Vietnamese media channels such as 60 Giay Official, HTV Sports, Bao Thanh Nien, ViVu TV, HTV Giai Tri, HTV Entertainment, and Bao Tuoi Tre, covering diverse topics including news, sports, entertainment, and social events. In addition to the raw videos, the organizers provided keyframes, object annotations generated by a Faster R-CNN model pretrained on OpenImagesV4, CLIP (ViT-B/32) embeddings in .npy format, YouTube metadata (date, channel, and title), and a mapping file linking each frame ID to its timestamp and frame rate.

## 4.2 Evaluation Metrics

We evaluated the performance of Vortex using the official Mean of Top-k R-Score protocol defined by the AIC’25 organizers, which jointly measures retrieval accuracy and ranking quality. For each query, the system submitted up to 100 ranked results, with each result assigned an R-Score in the range [0, 1] according to its correctness against the ground truth. The scoring criteria varied by task: Textual-KIS required both the video name and frame index to fall within the reference range, Visual-QA required correct video, frame, and textual answer, and Temporal Alignment granted partial credit proportional to the number of correctly matched frames within the allowed tolerance.

Table 1: Official Final Scores of the proposed system across the three preliminary rounds of the AIC’25 competition.

<table><tr><td>Round</td><td>Final Score</td><td>Modules Integrated</td></tr><tr><td>Round 1 (24 queries)</td><td>20.6</td><td>Baseline (CLIP-only search)</td></tr><tr><td>Round 2 (30 queries)</td><td>27.8</td><td>Hybrid RRF (CLIP + SigLIP2)</td></tr><tr><td>Round 3 (35 queries)</td><td>31.2</td><td>Temporal Search + Relevance Feedback</td></tr><tr><td>Total</td><td>79.6/88 (90.5%)</td><td>Overall performance across all rounds</td></tr></table>

For each query, the best R-Score within the top-k results was computed at five cutoff levels (k ∈ 1, 5, 20, 50, 100): R@k = max1≤i≤k R-Score(ri). The Final Score for each query was defined as the mean of the five R@k values, rewarding systems that retrieve correct answers early in the ranked list while maintaining consistent precision across different cutoff levels.

## 4.3 Results

The performance of our system progressively improved as additional modules were integrated. The official scores for each competition round in the Preliminary Round are reported in Table 1. Overall, the system achieved a final total of 79.6 out of 88 points, corresponding to approximately 90.5% of the maximum possible score, demonstrating robust and consistent performance across all evaluation rounds.

The system’s performance in the Final Round was preliminarily evaluated by the Jury Board, and our Vortex system achieved an overall rating of Excellent. The task-level evaluations are as follows: Excellent for TKIS, Very Good for VKIS, Very Good for TRAKE, and Outstanding for Q&A. These results demonstrate the robustness of our system architecture in addressing the diverse and complex queries presented in the final round, and they highlight its particularly strong content-comprehension capabilities in the Q&A task.

## 4.4 System Usage Examples

To demonstrate the practicality and interactivity of the Vortex system, we illustrate strategies used for several representative competition queries (Fig. 6):

Query tkis-query-02 (Textual KIS): The hint referenced a multiplechoice question about a reaction that “releases hydrogen gas” (non-diacritic: “giai phong khi hidro”). Because this was a highly specific cue, we skipped semantic search and directly applied the global OCR filter: /filterall ocr{hidro}. This immediately surfaced the correct video (Fig. 6a).

Query tkis-query-08 (Textual KIS): The hint described “a cave in the country famous for Gaulois roosters, with animal and human engravings.” We issued the semantic query “Hang dong co hinh khac dong vat” (Eng: “Cave with animal engravings”), and the system retrieved relevant clips via ASR transcripts and generated captions (Fig. 6b).

![](images/9e18392591a3d1c75d6bb99693b34c08ec11e247b8dcf7fdf33743f2c6415d92.jpg)

<details>
<summary>text_image</summary>

Screenshot of a web interface displaying a form with input fields, a red highlighted section, and a grid of product images.
</details>

(a) tkis-02 : Global OCR filter for "hidro".

![](images/c5cb28cb5eb8d7e509ca683e2f8c22c2b6fdc193499f4a2aa7b7a1e6680d89ca.jpg)

<details>
<summary>text_image</summary>

Screenshot of a photo editing interface showing a grid of nature photos with a red selection overlay and a dropdown menu in Chinese.
</details>

(b) tkis-08 : Semantic search for "Hang dong...".

![](images/7542aa46ab9a3f23f6b43c82d587000bb18e85399704e6c354641d366fc93d1c.jpg)

<details>
<summary>text_image</summary>

Screenshot of a social media or video feed interface displaying a grid of photo thumbnails with red highlights and a dropdown menu in Chinese.
</details>

(c) tkis-12 : Semantic search for "Cong trinh...".

![](images/4edc7de7461d69d28a68c8590db61f871ebf8ecf028e05f05d413c289a138ce0.jpg)

<details>
<summary>text_image</summary>

Screenshot of a food web interface displaying a grid of food images with Chinese labels and a 'OK' button.
</details>

(d) qa-02 : Initial search for ingredients.

![](images/84df6cad88a892eb23e6c95a99627f08eba7de16425334811aa023f01d511113.jpg)

<details>
<summary>text_image</summary>

Screenshot of a mobile photo gallery interface showing a grid of product photos with one highlighted item.
</details>

(e) qa-02 : "Nearby frame" feature showing the cut.

![](images/27b11203dcf60e621461ab6ce430cb54a92ada36c0a113b8ca3243632eaf78f9.jpg)

<details>
<summary>text_image</summary>

Screenshot of a social media or video feed interface displaying a grid of image thumbnails with Chinese text labels and UI elements.
</details>

(f) vkis-07 : OCR filter for "DI TICH KIM LONG".

![](images/daabeb08a8026b85f58fc5241e897aac9c2f339abefdc86334c8fb9401507dec.jpg)

<details>
<summary>text_image</summary>

Screenshot of a video editing interface showing a grid of video thumbnails with Chinese text labels and a red highlighted area.
</details>

(g) trake-03 (E1, Step 1): OCR search for "PHI 1 0 BRU" finds a frame ‘after’ the goal.

![](images/ab26fa46bf1640cb81ce729268f3e30354ab150d6c170a1e24505f1858926261.jpg)

<details>
<summary>text_image</summary>

CHI 00 BRU
APP
MILK SPORTS
25 WOEN
ASEAN
máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, nào
máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo bái bái, máo cǎo bái, máo cǎo bái, máo cǎo bái, máo cǎo bói
</details>

(h) trake-03 (E1, Step 2): Exploring nearby frames prior to the OCR result to find the event sequence.

![](images/0049f5ee28c4b405c40b5be59e8fb0a9b4f16058fe2af1657102d6a3200e7f0b.jpg)

<details>
<summary>text_image</summary>

Screenshot of a video editing software interface showing a grid of video thumbnails and a red highlighted selection panel.
</details>

(i) trake-03 (E1, Step 3): Visual Search pinpoints the exact keyframe of the goal.  
Fig. 6: Screenshots of the Vortex system in action for various query types.

Query tkis-query-12 (Textual KIS): The hint mentioned “an elliptical structure made of baked bricks.” Using the semantic query “Cong trinh co dang vong elip bang gach dat nung,” the system returned the correct segment based on scene descriptions (Fig. 6c).

Query qa-query-02 (Q&A): The question asked how many pieces a sandwich was cut into. The first hint listed the ingredients (“2 strips of watermelon, 1 strip of pineapple, 1 strip of pear”). Searching this phrase returned the clip’s starting point (Fig. 6d). Using the “nearby frame” feature, we quickly navigated to the cutting step, confirming the answer is two (Fig. 6e).

Query vkis-07 (Video KIS): This query provided no textual hints. By observing the displayed video, we identified the phrase “DI TICH KIM LONG” (Eng: “KIM LONG HISTORICAL SITE”) on an artifact and used the OCR filter to search it. The system immediately located the correct footage (Fig. 6f).

Query trake-03 (TRAKE): This query required identifying three events: E1 (first goal), E2 (penalty save), and E3 (second goal). Our coarse-to-fine approach for E1 illustrates the system’s interactivity:

– Step 1 (Coarse Filter): We applied an OCR filter for the scoreboard update “PHI 1 0 BRU,” which returned a frame immediately after the goal (Fig. 6g).  
– Step 2 (Temporal Search): Using the “nearby frame” tool, we navigated backward to inspect the sequence leading up to the goal (Fig. 6h).  
– Step 3 (Fine-Grained Search): We then captured a frame from this sequence and performed a Query-by-Example visual search, pinpointing the exact moment the ball crossed the line (Fig. 6i).  
This multi-stage strategy, such as text filtering, temporal navigation, and visual refinement, was also used to locate events E2 and E3.

## 5 Conclusion

This paper introduced Vortex, a unified multi-modal video retrieval system developed for the AIC’25, which integrates semantic understanding, fine-grained visual cues, and interactive query refinement within a single framework. In the competition, Vortex delivered strong and consistent results, scoring 79.6/88 (90.5%) in the Preliminary Round and earning an "Excellent" overall rating in the Final Round. Notably, it achieved "Outstanding" performance in Q&A and excelled across the remaining tasks. These outcomes highlight the effectiveness of combining semantic retrieval with fine-grained content analysis, as well as the value of interactive search mechanisms. Vortex demonstrates a powerful direction for context-aware, adaptable, and user-centered video retrieval systems, providing a solid foundation for future extensions in large-scale and real-world multimedia search.

Acknowledgments. This research is supported by research funding from Faculty of Information Technology, University of Science, Vietnam National University - Ho Chi Minh City.

## References

1. LSC ’25: Proceedings of the 8th Annual ACM Workshop on the Lifelog Search Challenge (2025)  
2. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report (2025)  
3. Bao, L.L., Phong, H.N., Tien, T.N., Thuan, P.N., Minh, T.H., Doan, T.T., Le, K., Do, T., Le, D.D., Ngo, T.D.: A hybrid video retrieval system using clip and beit-3 for enhanced object and contextual understanding. In: Buntine, W., Fjeld, M., Tran, T., Tran, M.T., Huynh Thi Thanh, B., Miyoshi, T. (eds.) Information and Communication Technology. pp. 206–218. Singapore (2025)  
4. Cormack, G.V., Clarke, C.L.A., Büttcher, S.: Reciprocal rank fusion outperforms condorcet and individual rank learning methods. ACM SIGIR (2009)  
5. Do, T.L., Huynh, V.T., Nguyen, H.D., Nguyen-Quang, T., Tran, M.K., Nguyen, T.T., Ninh, T.V., Le, T.K., Ngo, T.D., Dang-Nguyen, D.T., Ngo, T.T., Schöffmann, K., Gurrin, C., Tran, M.T.: Toward abstraction-level event retrieval in large video collections: Leveraging human knowledge and LLM-based reasoning in the Ho Chi Minh City AI Challenge 2025. In: Proceedings of the 14th International Symposium on Information and Communication Technology (SOICT 2025). CCIS, Springer, Nha Trang, Vietnam (2025)  
6. Gia, B.T., Khanh, T.B.C., Thanh, T.L.T., Tran, K., Trong, H.H., Doan, T.T., Le, K., Do, T., Le, D.D., Ngo, T.D.: Addressing ambiguous queries in video retrieval with advanced temporal search. In: Buntine, W., Fjeld, M., Tran, T., Tran, M.T., Huynh Thi Thanh, B., Miyoshi, T. (eds.) Information and Communication Technology. pp. 167–180. Singapore (2025)  
7. Heller, S., Gasser, R., Illi, C., Pasquinelli, M., Sauter, L., Spiess, F., Schuldt, H.: Towards explainable interactive multi-modal video retrieval with vitrivr. In: Lokoč, J., Skopal, T., Schoeffmann, K., Mezaris, V., Li, X., Vrochidis, S., Patras, I. (eds.) MMM. pp. 435–440 (2021)  
8. Quan, K.A.C., Nguyen, Q.N., Tran, M.T.: Vifi: A video finding system at video browser showdown 2025. In: MMM. p. 378–384. Berlin, Heidelberg (2025)  
9. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision (2021)  
10. Radford, A., Kim, J.W., Xu, T., Brockman, G., McLeavey, C., Sutskever, I.: Robust speech recognition via large-scale weak supervision (2022)  
11. Rossetto, L., Schoeffmann, K., Gurrin, C., Lokoč, J., Bailer, W.: Results of the 2025 video browser showdown (2025)  
12. Siradjuddin, I.A., Triyanto, A., S., M.K.: Content based image retrieval with rocchio algorithm for relevance feedback using 2d image feature representation. In: MLMI. p. 16–20. MLMI ’19, New York, NY, USA (2019)  
13. Tran, A., Bailer, W., Dang-Nguyen, D.T., Healy, G., Hodges, S., Þór Jónsson, B., Rossetto, L., Schoeffmann, K., Tran, M.T., Vadicamo, L., Gurrin, C.: The state-ofthe-art in lifelog retrieval: A review of progress at the acm lifelog search challenge workshop 2022-24 (2025)  
14. Tran, Q.L., Nguyen, B., Jones, G.J.F., Gurrin, C.: Videoease at vbs2025: An interactive video retrieval system. In: Ide, I., Kompatsiaris, I., Xu, C., Yanai, K., Chu, W.T., Nitta, N., Riegler, M., Yamasaki, T. (eds.) MMM. pp. 363–370 (2025)  
15. Tschannen, M., Gritsenko, A., Wang, X., Naeem, M.F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., Hénaff, O., Harmsen, J., Steiner, A., Zhai, X.: Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features (2025)  
16. Ueki, K., Muto, R., Wada, T., Akaba, R., Fernandez, G.F.L.: U-cker: A prototype system for the lifelog search challenge 2025. In: Proceedings of the 8th Annual ACM Workshop on the Lifelog Search Challenge. p. 58–62. LSC ’25, New York, NY, USA (2025)  
17. Vuong, G.H., Nguyen, V.L., Ho, V.S., Nguyen-Dang, T.T., Tran, N.D., Ninh, V.T., Tran, M.T.: Newsinsight2.0: An enhanced version integrating large language model-based query optimization with advanced temporal mechanisms. In: Buntine, W., Fjeld, M., Tran, T., Tran, M.T., Huynh Thi Thanh, B., Miyoshi, T. (eds.) Information and Communication Technology. pp. 337–348. Singapore (2025)  
18. Zhu, W., Huang, Y., Xie, X., Liu, W., Deng, J., Zhang, D., Wang, Z., Liu, J.: Autoshot: A short video dataset and state-of-the-art shot boundary detection (2023)