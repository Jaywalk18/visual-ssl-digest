# ELVA: Exploring Ranking-Driven Universal Multimodal Retrieval

Yuhan Liu1†‡, Pei Fu2†, Hang Li2, Yukun Qi2, Chao Jiang2, Jingwen Fu3§, Zhen Liu1, Bin Qin2, Zhenbo Luo2, Jian Luan2, and Jingmin Xin1§

1 National Key Laboratory of Human-Machine Hybrid Augmented Intelligence, Institute of Artificial Intelligence and Robotics, Xi’an Jiaotong University

2 MiLM Plus, Xiaomi Inc

3 Zhongguancun Academy, Beijing, China

Abstract. Leveraging Multimodal Large Language Models (MLLMs) via contrastive learning has become a mainstream paradigm for improving the performance of Universal Multimodal Retrieval (UMR). However, previous works have ignored the grain blindness when adapting the contrastive paradigm into retrieval tasks. Grain blindness refers to the tendency of the model to overlook grain-level information contained in the query, which is crucial for effectively handling complex queries. This stems from contrastive learning treating samples as a binary classification (positive/negative), while ignoring the different information carried by each negative sample. To address this, we argue that negatives should be treated differently according to their similarity to the positive sample, enabling the model to learn distinct grain information from each negative. In this paper, we introduce a simple but effective framework, called ELVA, a novel rule-based RL framework that mitigates grain blindness through ranking-driven MLLMs. 1) Instead of relying on reward models, we extend Reinforcement Learning with Verifiable Rewards (RLVR) to retrieval tasks, allowing the model to explore new ranking behaviors without explicit ranking labels. 2) By utilizing rule-based rewards, our approach jointly optimizes the ranking of negative samples while enlarging the similarity gap between positive and negative. To more precisely measure grain blindness, we further introduce MRBench, a new benchmark specifically designed for multi-grain query scenarios. ELVA achieves state-of-the-art results across standard retrieval benchmarks, and its notable 13.1% improvement on MRBench further demonstrates its effectiveness in alleviating grain blindness.

## 1 Introduction

Universal Multimodal Retrieval (UMR) refers to a general retrieval paradigm that unifies diverse retrieval tasks within a single framework and enables generalization to unseen retrieval tasks [20, 35, 38, 78]. This marks a substantial shift from prior efforts, which primarily focused on modality-specific retrieval tasks, including text-to-text [39,74], text-to-image [6,71], and image-to-image [3,47] retrieval. Recently, researchers have begun exploring Multimodal Large Language Models (MLLMs) [5, 23, 52, 62] for UMR, leveraging their extensive pretrained knowledge and strong generalization ability. Since MLLMs are originally trained for generative objectives (e.g., next-token prediction), recent works have adapted them to retrieval tasks via contrastive learning, effectively transferring their generative abilities to retrieval tasks [8, 30, 40, 73].

![](images/b3f3bb93fea803fdf32cd535f9d23881deb16910a92df2320f9c071089bfb90b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Query: Retrieve Similar Image"] --> B["+"]
  B --> C["grain 1: Charmander"]
  B --> D["grain 2: fire-breathing"]
  E["(a) Contrastive Objective Tuning"] --> F["Fail to capture the comprehensive grain info. e.g., lose &quot;fire-breathing&quot;"]
  F --> G["Ret Results"]
  H["(b) Ranking-Driven Tuning"] --> I["Capture the comprehensive grain info."]
  I --> J["Ret Results"]
```
</details>

Fig. 1: The main idea of our proposed ELVA. Previous works [31, 35] fail on multi-grain queries due to grain blindness that emerges during contrastive training, as illustrated in (a). Build on its basis, our ELVA leverages the ranking-driven tuning with verifiable rewards to capture the comprehensive information, accurately retrieve the precise candidates shown in (b).

Although existing methods achieve impressive performance, our study reveals that they still suffer from grain blindness when adapting the contrastive paradigm to retrieval tasks. Grain blindness refers to the tendency of the model to overlook grain-level information contained in a query. Consequently, these methods struggle with multi-grain queries, as shown in Figure 1. In such a case, the query contains multiple levels of grain information, such as the action “fire-breathing" and the entity “Charmander", which place high demands on the model’s ability to comprehensively capture multi-grain information. To tackle the grain blindness, our study highlights two key challenges that need to be confronted.

Challenge I: Which properties that the training paradigm have to equip in order to reduce the grain blindness? Previous works [31, 35, 38] leverage contrastive objectives, learning embeddings by distinguishing positive and negative samples. However, as illustrated in Figure 1 (a), this training paradigm is suboptimal for retrieval tasks, as it fails to comprehensively capture the multiple levels of grain information contained in a query. Intuitively, the positive sample needs to be contrasted against diverse negative samples in order to learn distinct grain information. Yet, the contrastive paradigm treats all negatives equally [16,79], ignore the differential information carried by each negative, which is crucial for accurate retrieval. To address this issue, we argue that the model should treat negative samples differently based on their similarity to the positive, which means negatives with higher similarity should be positioned closer to the positive. Toward this goal, we propose a ranking-driven approach that ranks candidates according to their relevance to the positive sample, enabling the model to capture comprehensive grain-level information, as shown in Figure 1 (b).

Challenge II: How to incentivize the ranking ability without the ranking labels? Previous retrieval models typically rely on supervised ranking objectives (e.g., listwise loss) to learn candidate ordering [9, 14, 28]. However, in UMR scenarios, obtaining ranking labels such as precisely ranking all negative samples by their exact relevance is extremely difficult and expensive. Meanwhile, forcing models to fit static labels severely restricts their capacity to explore subtle, grainlevel hierarchical differences. To overcome this, we introduce an explorationdriven RL framework that operates without explicit ranking targets, utilizes the reward function as dynamic evaluators. Driven by the GRPO algorithm [48], the model autonomously discovers optimal inter-negative hierarchies by comparing the relative ranking quality of its variant generated candidate lists.

In this paper, we propose ELVA: ExpLoring Ranking-Driven UniVersal Multimodal RetrievAl, a novel rule-based RL framework designed to overcome the grain blindness through ranking-driven MLLMs. Different from previous works that rely solely on contrastive objectives [35,38], ELVA adopts a rankingdriven tuning to capture richer grain-level information while simultaneously overcome the absence of ranking labels. Specifically, we propose two verifiable reward function to optimize the policy model: 1) Ranking Reward, which encourages the model to rank candidates based on their relevance while rewarding the model for placing positive samples at higher ranks inspired by the NDCG [15]. Our Ranking Reward is a continuous reformulation for RL, ensuring continuous reward signal while optimizing negatives hierarchies; 2) Margin Reward, which enforces explicit similarity-gap constraints, ensuring that positive samples remain closer to the query than negative ones. Additionally, we introduce a balanced negative sampling strategy to construct a ranking customized dataset for RL, filtering out excessively difficult negatives to ensure stable optimization.

For a more comprehensive evaluation, we construct a new benchmark, MR-Bench, derived from the M-BEIR dataset [56]. MRBench is specifically designed for multi-grain retrieval, where each query contains two or more grain-level attributes (e.g., an entity and an action), making it particularly challenging to preserve multi-grained information. Our method achieves a substantial 13.1% improvement in retrieval accuracy on this benchmark, demonstrating its effectiveness in mitigating grain blindness.

To summarize, we make the following contributions:

– We identify the issue of grain blindness when adapting the contrastive learning paradigm to retrieval tasks, and highlight two key challenges that need to be confronted.

– We propose ELVA to enable comprehensive multi-grain information acquisition by jointly optimizing ranking order and enforcing similarity-gap constraints.  
– We construct a new dataset to evaluate model performance in complex multigrain scenarios, and ELVA achieves state-of-the-art performance across diverse benchmarks including MRBench.

## 2 Related Work

Universal Multimodal Retrieval. Multimodal retrieval serves as the core task in information retrieval [2, 16, 45, 51], focusing on retrieving related content across diverse data modalities [22, 26]. As the landscape of information retrieval expands, more recent studies have shifted attention toward universal multimodal retrieval (UMR) [20, 35, 38, 78], where a unified model is capable of handling heterogeneous modalities and diverse retrieval tasks simultaneously. While earlier work in this domain often relied on small foundation models such as CLIP [57], recent advances [31, 35, 76] have demonstrated the promise of employing Multimodal Large Language Models (MLLMs) [24, 33, 43, 52, 65, 77] to further enhance retrieval performance. As MLLMs are primarily trained with generative objectives, recent researches [21,31,35] adapt them for retrieval tasks via contrastive learning, utilize embeddings extracted from MLLMs performing similarity-based retrieval, leveraging their strong cross-modal representation capabilities. However, it remains a significant challenge to capture comprehensive grain-level information in order to retrieve complex queries with high precision. We propose a simple yet effective approach to incentivize the model’s ranking ability, thus improving the UMR performance.

RL for Ranking Learning. Reinforcement learning (RL) has become a promising approach that enables models to adjust their behavior during training based on continuous feedback signals [37, 49, 67]. ReasonRank [34] optimizes discrete metric-based rewards (e.g., NDCG, Recall, RBO) defined over the entire ranked list. MM-R5 [64] introduces a position-weighted ranking reward, where each retrieved item receives a score according to its ranking position. These discrete rewards which means it only jumps when the ranking order changes, the optimization signal is discontinuous and high-variance, leading to unstable learning and poor convergence [61, 75]. In contrast, our reward enforces continuous feedback offering smoother gradients than purely discrete metric-based rewards. Search-R3 [10] incorporates continuous similarity scores, however the feedback easily saturates once the top-ranked positions converge, providing weak supervision for representation learning. In this paper, our ELVA not only maintains non-saturating learning signals via margin reward, but also models the intranegative ranking structure via continuous ranking reward. Moreover, the ranking reward encourages negatives with high relevance ranked nearer to the positive, thereby enriching the grain-level information within the embedding space.

## 3 Method

![](images/1a7ce8b8fa3126beb1465056c2dbea0f50cfeb5e699c0872cdd28724025c37cc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Embedding"] --> B["Post-Training"]
  B --> C["Contrastive Learning"]
  C --> D["Post-Training"]
  D --> E["Pre-training & Instruction Tuning"]
  F["Multimodal Large Language Models"] --> G["Image"]
  F --> H["Can you help me find a daily image?"]
  F --> I["Similar Images."]
  F --> J["Text"]
  F --> K["Summarize above image and sentence in one word."]
  F --> L["Summarization Prompt"]
  M["Stage 1-2"] --> N["Preliminary Model"]
  N --> O["Fail to handle complex query due to grain blindness."]
  P["Stage 3"] --> Q["Reinforcement Learning with Verifiable Reward"]
  Q --> R["Query"]
  Q --> S["Positive"]
  Q --> T["Negative"]
  R --> U["Policy Model"]
  S --> U
  T --> U
  U --> V["KL Constraint"]
  U --> W["G rollouts"]
  U --> X["Reward Functions"]
  U --> Y["Ranking Reward"]
  V --> Z["Reference Model"]
  W --> Z
  X --> Z
  Y --> Z
  Z --> AA["Margin Reward"]
  AA --> AB["Pos. > Q."]
  AA --> AC["Pos. Neg."]
  AA --> AD["Ranking Reward"]
  AB --> AE["Pos. No.1"]
  AB --> AF["Pos. No.2"]
  AB --> AG["Pos. Neg. 2"]
```
</details>

Fig. 2: Overview of the proposed ELVA framework. ELVA leverage the threestage training framework for UMR tasks. Stage 1-2 as the pre-training and instruction tuning stage following [35], obtained the preliminary model struggle with the grain blindness. Stage 3 employ the RL tuning to incentivizing the ranking ability to address the issue. Given the input query q and candidates including pos. and N x neg., we first perform G rollouts to output G independent sets of embeddings from policy model. Then we compute the reward $r _ { i }$ for each output $o _ { i }$ using our proposed reward functions, detailed in Section 3.4. Finally, we optimize the policy model with GRPO [11] while ensuring that the model remains close to the reference policy model, via KL divergence.

## 3.1 Preliminary

This paper proposes a novel framework ELVA in Figure 2, to tackle the grain blindness in UMR. In this task, given a query q originating from any modality (text, image, or interleaved formats), the objective is to retrieve the most relevant sample from a set $\varOmega = \{ c _ { n } \} _ { n = 1 } ^ { N }$ with N candidates. To perform retrieval, we use a unified embedding extractor to encode both the query and candidates. Then we compute the cosine similarity $s _ { i }$ between the query and each candidate, and rank all candidates based on their similarity scores. The final retrieval output corresponds to the top-k candidates:

$$
\mathcal {C} = \varPhi_ {\mathrm{ret}} (q, \varOmega) = \operatorname{Top} \text {-} k \big (\{s _ {1}, s _ {2}, \ldots , s _ {N} \} \big).
$$

## 3.2 Formulation

To theoretically ground the concept of grain blindness, we formally define the mathematical formulation and this representational collapse to theoretically ground our framework.

Definition 1 (Grain and Query Composition). A grain g is defined as the minimal, atomic semantic unit $( e . g .$ , an entity, an attribute, or an action) within a multimodal query. A query q is formulated as a set of these interdependent units, $q = \{ g _ { 1 } , g _ { 2 } , . . . , g _ { K } \}$ . Grains possess inherent structural dependencies; for instance, in the query “red dress,” the attribute grain red is necessarily coupled with the entity grain dress to form a coherent search intent.

Definition 2 (Grain Blindness). Grain blindness is formalized as a representational collapse where the distance between the full query embedding and its de-grained version (where a specific grain $g _ { k }$ is removed) falls below a discriminative threshold δ:

$$
d (f _ {\theta} (q), f _ {\theta} (q \setminus \{g _ {k} \})) <   \delta , \tag {1}
$$

where $f _ { \theta } ( \cdot )$ denotes the embedding function and $d ( \cdot , \cdot )$ is a distance metric. This inequality indicates that the model fails to preserve the discriminative features of the specific grain $g _ { k }$ in the embedding space.

Theoretically, the emergence of grain blindness during contrastive learning can be attributed to Gradient Starvation [42, 46]. If certain dominant grains within q (e.g., a primary entity) provide a sufficient similarity margin to distinguish between positive and negative pairs, the contrastive loss drops rapidly. Consequently, other salient but secondary grains lose their necessary gradient signals for optimization. This premature convergence forces the model to ignore the suppressed grains, directly leading to the representational collapse described in Definition 2.

## 3.3 Pre-training & Instruction Tuning

To effectively leverage the contrastive learning paradigm in transforming the generative capability of MLLMs into discriminative representations, we first employ a two-stage training framework following [35, 38]. Since MLLMs are primarily trained for generative objectives such as next-token prediction, their inherent retrieval capability remains limited. The first stage conducts languageonly pretraining on NLI datasets [7], enabling the generative model to produce more effective embeddings. The second stage, instruction tuning, further aligns the MLLMs with various retrieval tasks for better adaptability. These tasks include image-to-image retrieval, composed image retrieval, and image/questionto-multimodal-document retrieval, among others. Further details on the instruction templates and the M-BEIR datasets [56] are provided in the Suppl.

Training Objective. We adopt contrastive learning with the InfoNCE loss [41] as the optimization objective for both the language-only pretraining and instructiontuning phases. During training, we input query q and instruction i into the model to obtain the query representation $\textstyle e _ { q } .$ Similarly, each candidate c is fed into the model to derive its representation $e _ { c }$ . The training objective maximizes the similarity of positive samples and minimizes the similarity of negative samples, formulated as:

$$
\mathcal {L} = - \frac {1}{N} \sum_ {n = 1} ^ {N} \log \left[ \frac {\exp \left[ \cos \left(e _ {q} , e _ {c} ^ {+}\right) / \tau \right]}{\sum_ {m = 1} ^ {N} \exp \left[ \cos \left(e _ {q} , e _ {c _ {m}}\right) / \tau \right]} \right]
$$

where τ denotes the temperature parameter and N is the batch size. This approach enables the model to learns to discriminate between relevant and irrelevant information across various modalities.

## 3.4 ELVA

After the above training stages, we obtain a preliminary retrieval model. However, the model lacks sufficiently grain-level information, making it less effective for complex queries. To address this, we propose ELVA, a reinforcement learning based framework incentivizes the ranking ability of MLLMs for UMR. In contrast to conventional contrastive learning, ELVA optimizes model behavior through rule-based reward signals. However, a critical challenge in applying Policy Optimization to UMR tasks is that traditional EOL extraction [35, 38] (i.e., directly extracting the hidden state of a fixed prompt token [17]) yields no representational variance, rendering RL exploration impossible. To confront this, we formulate the feature extraction process as a generative paradigm. The model is mandated to first autoregressively generate a textual synthesis of the input, followed by a designated special token [RET], which serves as the information bottleneck. We utilize the following prompt template:

## Template for Generative Embeddings

USER: {Iimage} <Instruction> {Iquery}.

Analyze and summarize the key information of the above input. Finally, append the special token [RET] to represent the entire input.

ASSISTANT: {Generated Summary} [RET]

Here, Iimage denotes the image input, while Iquery refers to the query text input. These modalities can be flexibly combined, and the corresponding instructions are adjusted accordingly. [RET] is a special token registered in the LLM, and we take the hidden state at this token’s position as the retrieval embedding. In our RL framework, an action consists of generating embeddings for the query, positive, and all negatives by the policy model shown in the bottom of Figure 2. To facilitate policy optimization, the model generates G independent rollouts for a given query q and candidate set Ω through GRPO. This results in G groups of embeddings, denoted as {e(g)q , e(g)pos, {e(g)n,i}Ni=1}Gg=1, which serve as the basis for $\{ \mathbf { e } _ { q } ^ { ( g ) } , \mathbf { e } _ { p o s } ^ { ( g ) } , \{ \mathbf { e } _ { n , i } ^ { ( g ) } \} _ { i = 1 } ^ { N } \} _ { g = 1 } ^ { G }$ computing relative rewards.

Verifiable Reward Design. The reward model serves as a key component in reinforcement learning (RL), guiding the model’s behavior to align with predefined correctness objectives. While conventional RL paradigms typically depend on human preference-based reward modeling [19, 32], recent advances such as DeepSeek-R1 [11] have shown that verifiable reward functions can substantially enhance reasoning capability. Building upon this insight, we extend Reinforcement Learning with Verifiable Rewards (RLVR) to the multimodal retrieval domain by developing a rule-driven, multi-criteria reward function that jointly evaluates ranking quality and distance between candidates. This design not only produces accurate retrieval results but also encourages the reasonable order among negative samples, thereby improving both robustness and interpretability. Our framework evaluates model output in terms of two complementary dimensions: ranking orders and the distance between positive and negative.

Margin Reward. The margin reward is designed to encourage the model to maintain a sufficient similarity gap between positive and negative samples, inspired by the triplet loss [12,36]. To achieve this, we compute the similarity scores between the query and all negative candidates, and select the hardest negative that has the highest similarity to the query. The reward then encourages the model to increase the similarity gap between the positive pair and the hardest negative. Given the input of query q and candidates $\varOmega = \{ c _ { p } , c _ { n } ^ { 0 } , \cdots , c _ { n } ^ { k } \}$ , where p and n represent positive samples and negative samples, the margin reward is defined as:

$$
R _ {\mathrm{Margin}} = \max (0, \cos (q, c _ {p}) - \cos (q, c _ {n}) _ {\max} - \delta),
$$

where δ is a predefined hyperparameter that specifies the minimum required similarity gap. This formulation effectively drives the positive sample to rank at the top of the candidate list while imposing explicit similarity gap constraints. This reward enhances both retrieval precision and the discriminative structure of the learned representations.

Ranking Reward. We propose the ranking reward to explicitly encourage the model to rank the positive sample at the top of the candidate list while simultaneously promoting the ordering among negative samples to capture sufficient grained information. In contrast to the margin reward, which enforces a fixed similarity gap between positive and negative pairs, the ranking reward introduces rank-dependent weighting to optimize both rank precision and inter-negative structure.

Given a query and a candidate set, to translate these candidates into a ranked list, we compute cosine similarity scores $s _ { i } = \cos ( q , c _ { i } )$ between the query and candidates within each specific rollout. These scores are then sorted to form the ranking used to calculate the reward. The reward is defined as:

$$
R _ {\mathrm{Rank}} = s _ {(r)} \cdot \frac {1}{1 + \log r} - \gamma \sum_ {k \neq r} s _ {(k)} \cdot (\log k - 1),
$$

where $s _ { ( r ) }$ denotes the similarity of the positive sample ranked at position r, k denotes the rank of negatives and γ controls the penalty strength for high-scoring negatives. The first term encourages the model to assign a higher similarity score to the positive sample and place it near the top of the ranking. The second term encourages more similar negative samples to be ranked closer to the top, thereby receiving smaller penalties, while still constraining their similarity to the query.

Moreover, $R _ { R a n k }$ is a continuous reformulation designed for RL, ensuring smoother reward landscape while optimizing negatives hierarchies. This rankaware formulation enhances retrieval quality by encouraging precise rank positioning and structured ranking among negatives, thereby improving the model’s ability to learn adequate grained, discriminative representations.

Final Reward Function. The total reward combines the above rewards to optimize both ranking quality and similarity gap:

$$
R _ {\mathrm{total}} = \alpha R _ {\mathrm{Margin}} + \varepsilon R _ {\mathrm{Ranking}},
$$

where hyperparameters α and ε balance reward contributions and avoid reward hacking. This joint optimization enables the model to produce more precise retrieval outcomes by effectively capturing rich grained information, particularly in the context of complex or compositional queries.

Negative Sampling Strategy We introduce a balanced negative sampling strategy tailored for the RL stage. Prior work [35] directly sampled the top-100 candidates for SFT, which we find problematic for RL: when candidates are overly similar to the query, the reward distribution becomes too narrow, yielding weak or vanishing gradients [66] and hindering effective policy learning. To address this, we construct each query’s candidate set using a balanced mix of negatives: (1) 50 filtered hard negatives, obtained by removing candidates above a similarity threshold and selecting the top 50 remaining ones [8, 29]; and (2) 50 randomly sampled negatives from the full pool. This combination increases reward variance and provides richer learning signals: the filtered subset offers controlled difficulty to enhance ranking capability, while the random subset adds distributional diversity and prevents overfitting. Overall, this mixed sampling strategy yields a more stable and informative signal for RL optimization.

## 4 Experiments

## 4.1 Experimental Setup

Datasets and Metrics. We employ the NLI dataset [7] for pre-training and the M-BEIR dataset [56] for instruction tuning, following [35,38]. M-BEIR spans eight retrieval tasks across ten datasets, containing approximately 1.1M training instances. For the RL stage, we apply our negative sampling strategy to construct a training set of 11k instances from M-BEIR by sampling 1% of data from each dataset. We evaluate ELVA on the M-BEIR test set to assess its versatility across diverse retrieval scenarios. To further examine its generalization ability, we also evaluate ELVA on several unseen datasets [2, 69]. For multi-grain scenarios, we introduce a new benchmark, MRBench (Multi-gRain Benchmark), derived from M-BEIR. We first employ Qwen2.5-VL-7B [1] to automatically identify and filter queries containing at least two grain-level attributes, followed by human sampling verification to ensure data quality. Finally, we sample an equal number of instances from each task, resulting in a benchmark of 1k queries across 3 datasets and 4 retrieval tasks. We follow standard evaluation protocols for all datasets, using Recall@K as the primary metric for retrieval tasks.

Table 1: Comparison with recent state-of-the-art methods on the M-BEIR test set. The first row denotes the retrieval task configuration, where $q ^ { t }$ and $q ^ { i }$ represent text and image queries, respectively, and $c ^ { t }$ and $c ^ { i }$ denote text and image candidates. Dataset abbreviations include VN for VisualNews, F200K for Fashion200K, InfoS for InfoSeek, and FIQ for FashionIQ. Following the UniIR [56] evaluation protocol, Recall@10 is reported for FashionIQ and Fashion200K, while Recall@5 is used for all other datasets. The best results are highlighted.

<table><tr><td rowspan="3">Methods</td><td colspan="3"> $q^{t} \rightarrow c^{i}$ </td><td colspan="2"> $q^{t} \rightarrow c^{4}$ </td><td colspan="2"> $q^{t} \rightarrow (c^{i}, c^{t})$ </td><td colspan="3"> $q^{i} \rightarrow c^{4}$ </td><td> $q^{i} \rightarrow c^{i}$ </td><td colspan="2"> $(q^{i}, q^{t}) \rightarrow c^{4}$ </td><td colspan="2"> $(q^{i}, q^{t}) \rightarrow c^{i}$ </td><td colspan="2"> $(q^{i}, q^{t}) \rightarrow (c^{i}, c^{t})$ </td><td rowspan="3">Avg.</td></tr><tr><td>VN</td><td>COCO</td><td>F200K</td><td colspan="2">WebQA</td><td>EDIS</td><td>WebQA</td><td>VN</td><td>COCO</td><td>F200K</td><td>NIGHTS</td><td>OVEN</td><td>InfoS</td><td>FIQ</td><td>CIRR</td><td>OVEN</td><td>InfoS</td></tr><tr><td>R@5</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@5</td><td>R@5</td><td>R@10</td><td>R@5</td><td>R@5</td><td>R@5</td></tr><tr><td colspan="19">Zero-shot</td></tr><tr><td>CLIP-L [45]</td><td>43.3</td><td>61.1</td><td>6.6</td><td>36.2</td><td>43.3</td><td>45.1</td><td>41.3</td><td>79.0</td><td>7.7</td><td>26.1</td><td>24.2</td><td>20.5</td><td>7.0</td><td>13.2</td><td>38.8</td><td>26.4</td><td>32.5</td><td></td></tr><tr><td>SigLIP [68]</td><td>30.1</td><td>75.7</td><td>36.5</td><td>39.8</td><td>27.0</td><td>43.5</td><td>30.8</td><td>88.2</td><td>34.2</td><td>28.9</td><td>29.7</td><td>25.1</td><td>14.4</td><td>22.7</td><td>41.7</td><td>27.4</td><td>37.2</td><td></td></tr><tr><td>BLIP [25]</td><td>16.4</td><td>74.4</td><td>15.9</td><td>44.9</td><td>26.8</td><td>20.3</td><td>17.2</td><td>83.2</td><td>19.9</td><td>27.4</td><td>16.1</td><td>10.2</td><td>2.3</td><td>10.6</td><td>27.4</td><td>16.6</td><td>26.8</td><td></td></tr><tr><td>BLIP2 [24]</td><td>16.7</td><td>63.8</td><td>14.0</td><td>38.6</td><td>26.9</td><td>24.5</td><td>15.0</td><td>80.0</td><td>14.2</td><td>25.4</td><td>12.2</td><td>5.5</td><td>4.4</td><td>11.8</td><td>27.3</td><td>15.8</td><td>24.8</td><td></td></tr><tr><td>Qwen2-VL-7B [52]</td><td>9.3</td><td>55.1</td><td>5.0</td><td>42.0</td><td>26.2</td><td>9.4</td><td>5.4</td><td>46.6</td><td>4.0</td><td>21.3</td><td>21.4</td><td>22.5</td><td>4.3</td><td>16.3</td><td>43.6</td><td>36.2</td><td>23.0</td><td></td></tr><tr><td>Qwen2.5-VL-7B [1]</td><td>40.2</td><td>71.9</td><td>20.3</td><td>71.9</td><td>49.4</td><td>64.5</td><td>29.3</td><td>84.6</td><td>19.4</td><td>25.5</td><td>42.4</td><td>32.1</td><td>25.0</td><td>55.1</td><td>60.8</td><td>54.9</td><td>46.7</td><td></td></tr><tr><td colspan="19">Supervised - Dual Encoder</td></tr><tr><td>UniIR-BLIP $_{FF}$  [57]</td><td>23.4</td><td>79.7</td><td>26.1</td><td>80.0</td><td>50.9</td><td>79.8</td><td>22.8</td><td>89.9</td><td>28.9</td><td>33.0</td><td>41.0</td><td>22.4</td><td>29.2</td><td>52.2</td><td>55.8</td><td>33.0</td><td>46.8</td><td></td></tr><tr><td>UniIR-CLIP $_{SF}$  [57]</td><td>42.6</td><td>81.1</td><td>18.0</td><td>84.7</td><td>59.4</td><td>78.7</td><td>43.1</td><td>92.3</td><td>18.3</td><td>32.0</td><td>45.5</td><td>27.9</td><td>24.4</td><td>44.6</td><td>67.6</td><td>48.9</td><td>50.6</td><td></td></tr><tr><td colspan="19">Supervised - MLLMs</td></tr><tr><td>Vision-R1-7B [13]</td><td>41.9</td><td>75.0</td><td>22.0</td><td>70.6</td><td>51.3</td><td>69.1</td><td>35.4</td><td>85.1</td><td>22.4</td><td>25.9</td><td>48.8</td><td>44.0</td><td>29.2</td><td>57.7</td><td>66.2</td><td>59.0</td><td>50.2</td><td></td></tr><tr><td>VLM-R1-7B [49]</td><td>40.5</td><td>77.2</td><td>22.5</td><td>72.3</td><td>50.0</td><td>67.9</td><td>36.2</td><td>86.3</td><td>20.9</td><td>26.4</td><td>48.8</td><td>37.5</td><td>29.9</td><td>57.4</td><td>64.0</td><td>62.3</td><td>50.0</td><td></td></tr><tr><td>MM-Embed-7B [31]</td><td>41.0</td><td>71.3</td><td>17.1</td><td>95.9</td><td>68.8</td><td>85.0</td><td>41.3</td><td>90.1</td><td>18.4</td><td>32.4</td><td>42.1</td><td>42.3</td><td>25.7</td><td>50.0</td><td>64.1</td><td>57.7</td><td>52.7</td><td></td></tr><tr><td>PUMA-3B [38]</td><td>35.7</td><td>79.5</td><td>25.8</td><td>86.2</td><td>58.2</td><td>78.4</td><td>35.2</td><td>90.1</td><td>29.0</td><td>31.4</td><td>52.7</td><td>48.3</td><td>30.6</td><td>49.9</td><td>74.0</td><td>65.2</td><td>54.4</td><td></td></tr><tr><td>LamRA-Ret-2B [35]</td><td>30.8</td><td>78.8</td><td>23.1</td><td>82.5</td><td>54.3</td><td>77.8</td><td>31.2</td><td>88.5</td><td>27.1</td><td>28.7</td><td>51.1</td><td>44.2</td><td>28.9</td><td>47.7</td><td>72.3</td><td>60.8</td><td>51.6</td><td></td></tr><tr><td>LamRA-Ret-7B [35]</td><td>41.6</td><td>81.5</td><td>28.7</td><td>86.0</td><td>62.6</td><td>81.2</td><td>39.6</td><td>90.6</td><td>30.4</td><td>32.1</td><td>54.1</td><td>52.1</td><td>33.2</td><td>53.1</td><td>76.2</td><td>63.3</td><td>56.6</td><td></td></tr><tr><td>ELVA-2B (Ours)</td><td>35.6</td><td>80.3</td><td>25.0</td><td>88.0</td><td>56.1</td><td>80.5</td><td>33.4</td><td>90.2</td><td>25.9</td><td>29.3</td><td>52.0</td><td>47.4</td><td>30.9</td><td>50.0</td><td>72.8</td><td>61.3</td><td>53.8+4.3%</td><td></td></tr><tr><td>ELVA-7B (Ours)</td><td>43.5</td><td>83.0</td><td>29.2</td><td>91.0</td><td>63.5</td><td>83.1</td><td>41.7</td><td>92.2</td><td>32.1</td><td>32.8</td><td>56.0</td><td>55.5</td><td>34.6</td><td>55.4</td><td>77.5</td><td>67.1</td><td>58.7+3.9%</td><td></td></tr></table>

Implementation Details. Our framework is implemented in PyTorch, by default, built upon Qwen2-VL-7B [52]. During the retrieval pretraining stage, experiments are conducted on 8 × H 20 GPUs with a batch size of 576, a learning rate of $4 \times 1 0 ^ { - 5 }$ , and trained for two epochs (3h completed). In the instruction tuning stage, we use 16 × H 20 GPUs with a batch size of 960 and a learning rate of $1 \times 1 0 ^ { - 4 }$ for one epoch following [35] (48h completed). For the RL stage, training is performed for one epoch on 8 × H 20 GPUs with a learning rate of $1 \times 1 0 ^ { - 6 }$ , using 8 rollouts and $\beta = 0 . 2$ (16h completed). Across all stages, the vision encoder remains frozen, while the language model is fine-tuned using LoRA. During M-BEIR evaluation, we conduct experiments in a local retrieval pool with generative embedding extract method. The weight hyperparameter set to $\alpha = 0 . 4$ and $\varepsilon = 0 . 6$ . More details are shown in Suppl.

Table 2: Experimental results on unseen datasets. The first row denotes the type of retrieval task: $q ^ { t }$ represents text queries, $q ^ { i }$ image queries, $q ^ { \mathrm { d i a l o g } }$ dialog-based queries, and $( q ^ { i } \oplus q ^ { t } )$ denotes interleaved image-text queries; $\hat { c } ^ { t }$ and $c ^ { i }$ correspond to text and image candidates, respectively, while ITM refers to the Image-Text Matching task. Dataset abbreviations include Share4V for ShareGPT4V, Urban for Urban-1k, VisD for Visual Dialog, and MT-FIQ for Multi-round FashionIQ. The ∗ symbol indicates that images in these datasets originate from COCO or FashionIQ; however, due to notable differences in captions and query structures, they are still treated as unseen datasets. We follow the standard evaluation metrics defined for each dataset, and the best-performing results are highlighted.

<table><tr><td rowspan="3">Methods</td><td colspan="3"> $q^{t} \rightarrow c^{i}$ </td><td colspan="3"> $q^{i} \rightarrow c^{t}$ </td><td colspan="2"> $(q^{i}, q^{t}) \rightarrow c^{i}$ </td><td> $q^{\text{dialog}} \rightarrow c^{i}$ </td><td> $(q^{i} \oplus q^{t}) \rightarrow c^{i}$ </td><td colspan="2">ITM</td></tr><tr><td>Share4V</td><td>Urban*</td><td>Flickr</td><td>Share4V</td><td>Urban*</td><td>Flickr</td><td>CIRCO*</td><td>GeneCIS*</td><td>VisD*</td><td>MT-FIQ*</td><td>CC-Neg</td><td>Sugar-Crepe*</td></tr><tr><td>R@1</td><td>R@1</td><td>R@1</td><td>R@1</td><td>R@1</td><td>R@1</td><td>MAP@5</td><td>R@1</td><td>R@1</td><td>R@5</td><td>Acc.</td><td>Acc.</td></tr><tr><td>CLIP-L [45]</td><td>84.0</td><td>52.8</td><td>67.3</td><td>81.8</td><td>68.7</td><td>87.2</td><td>4.0</td><td>13.3</td><td>23.7</td><td>17.7</td><td>66.7</td><td>73.0</td></tr><tr><td>Long-CLIP-L [69]</td><td>95.6</td><td>86.1</td><td>76.1</td><td>95.8</td><td>82.7</td><td>89.3</td><td>5.7</td><td>16.3</td><td>37.9</td><td>18.5</td><td>76.3</td><td>80.9</td></tr><tr><td>UniIR-CLIP [56]</td><td>85.8</td><td>75.0</td><td>78.7</td><td>84.1</td><td>78.4</td><td>94.2</td><td>12.5</td><td>16.8</td><td>26.8</td><td>39.4</td><td>79.9</td><td>80.3</td></tr><tr><td>E5-V [18]</td><td>86.7</td><td>84.0</td><td>79.5</td><td>84.0</td><td>82.4</td><td>88.2</td><td>24.8</td><td>18.5</td><td>54.6</td><td>19.2</td><td>83.2</td><td>84.7</td></tr><tr><td>MagicLens-L [70]</td><td>85.5</td><td>59.3</td><td>72.5</td><td>60.9</td><td>24.2</td><td>84.6</td><td>29.6</td><td>16.3</td><td>28.0</td><td>22.6</td><td>62.7</td><td>75.9</td></tr><tr><td>EVA-CLIP-8B [50]</td><td>91.2</td><td>77.8</td><td>80.8</td><td>93.1</td><td>80.4</td><td>95.6</td><td>6.0</td><td>13.1</td><td>23.2</td><td>22.1</td><td>59.4</td><td>81.7</td></tr><tr><td>EVA-CLIP-18B [50]</td><td>92.1</td><td>81.7</td><td>83.3</td><td>94.0</td><td>83.3</td><td>96.7</td><td>6.1</td><td>13.6</td><td>24.7</td><td>21.9</td><td>63.8</td><td>83.1</td></tr><tr><td>LamRA-Ret-7B [35]</td><td>93.3</td><td>95.1</td><td>82.8</td><td>88.1</td><td>94.3</td><td>92.7</td><td>33.2</td><td>18.9</td><td>62.8</td><td>60.9</td><td>79.6</td><td>85.8</td></tr><tr><td>ELVA-7B (Ours)</td><td>96.6</td><td>96.1</td><td>84.4</td><td>92.0</td><td>95.5</td><td>95.2</td><td>34.5</td><td>20.2</td><td>65.3</td><td>61.2</td><td>87.3</td><td>91.1</td></tr></table>

## 4.2 Experimental Results

Comparison of Effectiveness. We begin by evaluating the effectiveness of ELVA on the M-BEIR test set. Table 1 reports results in terms of Recall@K, covering 16 sub-tasks across 8 combinations of query and candidate modalities. To examine scalability, we present results for both ELVA-2B and ELVA-7B. We compare against three categories of methods: 1) Zero-shot general-purpose MLLMs, including BLIP-2 [24] and Qwen-VL [52]; 2) prior RL-based MLLMs, such as Vision-R1 [13] and VLM-R1 [49]; and 3) Retrieval-specialized MLLMs, including MM-Embed [31] and LamRA [35]. As shown in Table 1, ELVA consistently achieves state-of-the-art (SOTA) results across most settings. Notably, 1) even the 2B variant surpasses larger models such as MM-Embed-7B on most sub-tasks, and 2) on particularly challenging configurations like $( q ^ { i } , q ^ { t } )  ( c ^ { i } , c ^ { t } )$ on the InfoS dataset with 6.0% improvement, ELVA attains a substantial performance lead over previous methods. These results demonstrate the robustness and universality of ELVA, highlighting its strong retrieval capability across diverse multimodal inputs. We further apply LamRA-Rank in the reranking stage to boost accuracy shown in Suppl.

Comparison of Generalization on Unseen Dataset. To assess the generalization capability of our approach, we conduct extensive experiments on multiple unseen retrieval datasets. As shown in Table 2, ELVA consistently delivers strong performance across all evaluation settings, demonstrating robust generalization to diverse data modalities and task types. In ITM tasks, our ELVA achieves over a 9.7% improvement to other methods. Similarly, in fixed-modal retrieval tasks such as text-to-image, our method also achieves substantial improvements in performance. These findings underscore the strong adaptability and scalability of ELVA, highlighting its promise as a unified framework for broader multimodal applications.

Table 3: Experimental results on held-out tasks. ∗ indicates training on other tasks without exposure to the three held-out tasks.

<table><tr><td rowspan="2">Methods</td><td> $q^{i} \rightarrow c^{i}$ </td><td colspan="2"> $(q^{i}, q^{t}) \rightarrow c^{t}$ </td><td colspan="2"> $(q^{i}, q^{t}) \rightarrow (c^{i}, c^{t})$ </td><td rowspan="2">Avg.</td></tr><tr><td>NIGHTS R@5</td><td>OVEN R@5</td><td>InfoS R@5</td><td>OVEN R@5</td><td>InfoS R@5</td></tr><tr><td colspan="7">Supervised</td></tr><tr><td>UniIR-BLIP $_{FF}$ </td><td>33.0</td><td>41.0</td><td>22.4</td><td>55.8</td><td>33.0</td><td>37.0</td></tr><tr><td>UniIR-CLIP $_{SF}$ </td><td>32.0</td><td>45.5</td><td>27.9</td><td>67.6</td><td>48.9</td><td>44.4</td></tr><tr><td colspan="7">Zero-shot</td></tr><tr><td>Qwen2.5-VL</td><td>20.3</td><td>38.5</td><td>40.4</td><td>53.6</td><td>44.9</td><td>39.5</td></tr><tr><td>Vision-R1</td><td>22.9</td><td>39.8</td><td>42.9</td><td>57.4</td><td>46.5</td><td>41.9</td></tr><tr><td>LamRA-Ret*</td><td>27.2</td><td>44.7</td><td>44.0</td><td>62.8</td><td>49.5</td><td>45.6</td></tr><tr><td>ELVA-7B*</td><td>28.2</td><td>46.5</td><td>49.2</td><td>64.4</td><td>53.0</td><td>48.3</td></tr></table>

Table 6: Experimental results on MRBench datasets. The ∗ symbol indicates that dataset are filtered for the multi-grain scene.

<table><tr><td rowspan="2">Methods</td><td colspan="2"> $q^{t} \to c^{i}$ </td><td> $q^{i} \to c^{t}$ </td><td> $q^{i} \to c^{i}$ </td><td rowspan="2"> $(q^{i}, q^{t}) \to c^{i}$ </td><td rowspan="2">Avg.</td></tr><tr><td>COCO*R@5</td><td>COCO*R@5</td><td>NIGHTS*R@5</td><td>CIRR*R@5</td></tr><tr><td>Qwen2-VL-7B [52]</td><td>39.2</td><td>32.0</td><td>15.6</td><td>10.8</td><td>34.3</td><td></td></tr><tr><td>LamRA-Ret-7B [35]</td><td>50.4</td><td>54.0</td><td>22.4</td><td>25.7</td><td>38.1</td><td></td></tr><tr><td>ELVA-7B (Ours)</td><td>55.5</td><td>60.6</td><td>24.1</td><td>32.5</td><td>43.2</td><td></td></tr></table>

Table 4: Comparison of Reward Weighting.

<table><tr><td>Method</td><td>VN</td><td>COCO</td><td>F200K</td><td>Avg.</td></tr><tr><td> $\alpha = 0.6, \varepsilon = 0.4$ </td><td>42.9</td><td>82.2</td><td>28.8</td><td>58.2</td></tr><tr><td> $\alpha = 0.5, \varepsilon = 0.5$ </td><td>43.2</td><td>82.5</td><td>29.1</td><td>58.4</td></tr><tr><td> $\alpha = 0.4, \varepsilon = 0.6$  (ELVA)</td><td>43.5</td><td>83.0</td><td>29.2</td><td>58.7</td></tr></table>

Table 5: Generalizability of our ranking-driven RL framework.

<table><tr><td>Method</td><td colspan="2">M-BEIR MRBench</td></tr><tr><td>PUMA [38]</td><td>54.4</td><td>35.1</td></tr><tr><td>PUMA+ELVA</td><td>56.3</td><td>37.0</td></tr><tr><td>MM-Embed [31]</td><td>52.7</td><td>34.6</td></tr><tr><td>MM-Embed+ELVA</td><td>54.9</td><td>36.2</td></tr></table>

Table 7: Zero-shot text-to-video retrieval performance.

<table><tr><td rowspan="2">Method</td><td colspan="3">MSR-VTT</td><td colspan="3">MSVD</td></tr><tr><td>R@1</td><td>R@5</td><td>R@10</td><td>R@1</td><td>R@5</td><td>R@10</td></tr><tr><td>InternVideo [55]</td><td>40.0</td><td>65.3</td><td>74.1</td><td>43.4</td><td>69.9</td><td>79.1</td></tr><tr><td>ViCLIP [53]</td><td>42.4</td><td>-</td><td>-</td><td>49.1</td><td>-</td><td>-</td></tr><tr><td>UMT-L [27]</td><td>42.6</td><td>64.4</td><td>73.1</td><td>49.9</td><td>77.7</td><td>85.3</td></tr><tr><td>InternVideo $2_{s2}$ -6B [54]</td><td>55.9</td><td>78.3</td><td>85.1</td><td>59.3</td><td>84.4</td><td>89.6</td></tr><tr><td>LamRA-7B [35]</td><td>44.7</td><td>68.6</td><td>78.6</td><td>52.4</td><td>79.8</td><td>87.0</td></tr><tr><td>ELVA-7B (Ours)</td><td>46.4</td><td>70.5</td><td>78.9</td><td>53.9</td><td>80.7</td><td>87.9</td></tr></table>

Comparison of Generalization on Unseen Task. Evaluate ELVA on unseen retrieval tasks by excluding specific tasks during training and testing the retrained model on these omitted tasks. As shown in Table 3, our method exhibits strong performance on unseen retrieval tasks. At the same parameter scale, our ELVA achieves 5.9% improvement over previous SOTA method. This strong generalization capability indicates that our method can effectively extend to unseen retrieval tasks without further training, showing great potential for broader real-world applications.

Comparison of Multi-Grain scene on MRBench. Table 6 reports the results on the MRBench benchmark. ELVA achieves superior performance compared to previous methods SOTA LamRA with 13.1% improvements and zeroshot models, highlighting the effectiveness of our approach in accurately retrieving queries with multi-grain information. As illustrated in Figure 3, when a query includes multiple grain-level attributes, such as “snow-capped mountains” and “trees”, existing methods frequently fail to retrieve the correct result due to their inability to capture all grain components adequately. The qualitative comparisons show that our method captures the complex intent of the query and successfully retrieves the desired target. These results further confirm the effectiveness of our proposed approach in substantially alleviating the grain blindness problem. For more qualitative examples, please refer to Suppl.

## 4.3 Ablation Study

Ablating Reward Functions. To evaluate the effect of reward functions, we ablate ranking rewards and margin rewards, analyzing their effect across M-

Table 8: Ablation study. Avg. refers to the average recall performance across the M-BEIR test set. We select the image-to-text retrieval tasks as example.

<table><tr><td>#</td><td>Method</td><td>VN</td><td>COCO</td><td>F200K</td><td>Avg.</td></tr><tr><td>1</td><td>w/o Ranking Rewards</td><td>41.7 ↓1.8</td><td>82.0 ↓1.0</td><td>28.4 ↓0.8</td><td>57.2 ↓1.5</td></tr><tr><td>2</td><td>w/o Margin Rewards</td><td>42.3 ↓1.2</td><td>82.2 ↓0.8</td><td>28.5 ↓0.7</td><td>58.1 ↓0.6</td></tr><tr><td>3</td><td>w/o Negative Ranking</td><td>42.8 ↓0.7</td><td>82.0 ↓1.0</td><td>28.5 ↓0.7</td><td>58.1 ↓0.6</td></tr><tr><td>4</td><td>w/o Negative Sampling Strategy</td><td>43.1 ↓0.4</td><td>82.2 ↓0.8</td><td>28.5 ↓0.7</td><td>58.2 ↓0.5</td></tr><tr><td>5</td><td>w/o Randomly Sampled Negatives</td><td>43.3 ↓0.2</td><td>82.6 ↓0.4</td><td>28.9 ↓0.3</td><td>58.4 ↓0.3</td></tr><tr><td>6</td><td>ELVA-7B (Ours)</td><td>43.5</td><td>83.0</td><td>29.2</td><td>58.7</td></tr></table>

Instruction: Find a daily life image that is identical to the given one.  
![](images/d8ba6505ce0d965da2706bdcf812eee4c1cd5aa498a18a9ab949ce2a091a30cb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["LamRA"] -->|q^i → c^i| B["Original"]
  B --> C["ELVA (Ours)"]
  C --> D["(q^i, q^t) → c^i"]
  D --> E["Change to a large horse, and man in a black suit"]
```
</details>

Fig. 3: Qualitative examples. We show the results of our method across different retrieval tasks, with the correct result indicated by the green box. Here, qt for text queries, qi for image queries, ci for image candidates.

BEIR test set, as shown in Table 8. Excluding the ranking rewards leads to noticeable performance drops, highlighting its importance for enhancing the ranking quality, in order to learn sufficient grain information. Removing the margin rewards also reduces performance, indicating its role in ensuring the similarity gap for precise retrieval. In addition, we conduct another ablation in ranking reward, removing the second term designed for negative rank in row 3. The performance indicates that optimizing the negative ranking leads to more accurate retrieval. We conduct more ablation studies about hyperparameters in Suppl.

Ablating Negative Sampling Strategy. To evaluate the impact of the negative sampling strategy, we analyze different sampling strategies. As shown in Table 8, removing the sampling strategy leads to performance degradation, suggesting its importance in maintaining training data moderate difficulty in row 4. Introducing a random subset further enhances distributional diversity and prevents the model from overfitting in row 5. The results indicate that a properly training dataset is essential for model convergence, ensuring robust performance across datasets. More comparison results are shown in Suppl.

Ablating Reward Weighting. Additionally, we examine how different weighting schemes between the margin-based reward and the ranking-based reward affect the final performance in Table 4. This observation indicates that while the margin reward provides effective local pairwise guidance, it is the ranking reward that captures more holistic ordering signals and aligns more strongly with the final evaluation metric (Recall/K). Therefore, assigning a slightly higher weight to the ranking reward allows the model to better optimize global ranking behavior without losing the corrective constraints introduced by the margin term.

## 4.4 Deep Analysis

Post-Training Extension. Beyond the complete three stages pipeline, our proposed RL paradigm (Stage 3) functions as a highly adaptable, modular enhancement for existing multimodal retrievers. Future works can bypass the supervised fine-tuning stages and directly apply our ranking-driven RL to off-the-shelf models to achieve consistent performance gains. As shown in Table 5 integrating S3 as a post-training step significantly improves the average performance of PUMA [38] and MM-Embed [31]. This demonstrates our RL framework to be a universal "plug-and-play" booster that seamlessly scales to various multimodal retrieval architectures.

Extending to Video Retrieval. As presented in Table 7, we evaluate our method on the MSR-VTT [63] and MSVD [4] datasets under a zero-shot textto-video retrieval setting. The results show that our approach achieves strong performance. For example, on MSR-VTT, our model achieves 16.5% improvements over InternVideo [55] , while on MSVD, it outperforms UMT-L [27] by 8.4%. It is noteworthy that our model has not been exposed to any video data during fine-tuning, yet it still retains Qwen2-VL’s inherent video understanding ability. Although the current performance remains below the state-of-the-art InternVideo2 [54], we plan to incorporate video data in future work to further narrow this gap [44, 58].

Empirical and Qualitative Analysis of the Embedding Space. To validate the mitigation of grain blindness (Def. 2), we measure the representation distance d( $f _ { \theta } ( q ) , f _ { \theta } ( q \backslash \{ g _ { k } \} ) )$ on 100 sampled MRBench queries. By systematically masking a single phrase-level grain (e.g., dropping ‘standing” from ‘standing dog”), we compute the average cosine

![](images/56a92d3b78b3cd3661e64a50bd8c11a10d0d463cb257f86d429ade38f7f00702.jpg)

<details>
<summary>scatterplot</summary>

| Sentiment Type         | Count |
| ---------------------- | ----- |
| Positive (Standing Dog) | 100   |
| Hard Reg (Lying Dog)    | 50    |
| Hard Reg (Standing Cat) | 30    |
</details>

Baseline

![](images/fdeb948d53aae66225b68b2a94d995dcec9a4e086f9a36fd505bc6687f24c267.jpg)

<details>
<summary>scatterplot</summary>

| Category | Count |
| -------- | ----- |
| Positive (Standing Dog) | 100 |
| Hard Reg (Crying Dog) | 80 |
| Hard Reg (Standing Cat) | 60 |
| Easy Negatives | 20 |
</details>

ELVA  
Fig. 4: Distribution of embeddings from Baseline (left) and ELVA (right).

distance between full and masked query embeddings. The baseline yields a collapsed distance of 0.07, indicating the dropped grain was largely ignored. Conversely, ELVA significantly widens this gap to 0.15, quantitatively confirming its preservation of grain-level semantics. Furthermore, t-SNE visualization in Figure 4 of the query ‘Standing Dog”, shows that while the baseline entangles positives with hard negatives (e.g., ‘Lying Dog” or “Standing Cat”), ELVA clearly separates them. Together, these results demonstrate ELVA’s ability to prevent granularity loss and construct a highly discriminative embedding space.

## 5 Limitations and Future Work

The limitation of MLLM-based retrieval is high inference costs. This overhead can be further mitigated via feature precomputation, layer pruning [38], efficient MLLM designs [59, 60, 72], or deploying the lightweight ELVA-2B. Future works includes exploring joint retrieval-reranking frameworks to reduce pipeline complexity and scaling to larger MLLMs.

## 6 Conclusion

In this paper, we present ELVA, a novel framework designed to address the grain blindness when adapting the MLLMs via contrastive paradigm to Universal Multimodal Retrieval (UMR). We identify two challenges for addressing the grain blindness: 1) training paradigm’s properties impact on retrieval; 2) how to incentivize the new ability without the ranking labels. We also introduce a new benchmark specifically designed to evaluate the grain blindness mitigation. Extensive experiments demonstrate that ELVA achieves significant gains and effectively mitigates grain blindness. We expect that our framework will offer meaningful guidance for advancing multimodal information retrieval and encourage continued exploration in this domain.

## References

1. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al.: Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923 (2025) 10  
2. Baldrati, A., Agnolucci, L., Bertini, M., Del Bimbo, A.: Zero-shot composed image retrieval with textual inversion. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 15338–15347 (2023) 4, 10  
3. Baldrati, A., Bertini, M., Uricchio, T., Del Bimbo, A.: Effective conditioned and composed image retrieval combining clip-based features. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 21466– 21474 (2022) 2  
4. Chen, D., Dolan, W.B.: Collecting highly parallel data for paraphrase evaluation (2011) 14  
5. Dong, H., Kang, Z., Yin, W., Liang, X., Feng, C., Ran, J.: Scalable vision language model training via high quality data curation. arXiv preprint arXiv:2501.05952 (2025) 2  
6. Fu, Z., Zhang, L., Xia, H., Mao, Z.: Linguistic-aware patch slimming framework for fine-grained cross-modal alignment. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 26307–26316 (2024) 2  
7. Gao, T., Yao, X., Chen, D.: Simcse: Simple contrastive learning of sentence embeddings (2021) 6, 9  
8. Gu, T., Yang, K., Feng, Z., Wang, X., Zhang, Y., Long, D., Chen, Y., Cai, W., Deng, J.: Breaking the modality barrier: Universal embedding learning with multimodal llms. arXiv preprint arXiv:2504.17432 (2025) 2, 9  
9. Gu, T., Yang, K., Zhang, K., An, X., Feng, Z., Zhang, Y., Cai, W., Deng, J., Bing, L.: Unime-v2: Mllm-as-a-judge for universal multimodal embedding learning. arXiv preprint arXiv:2510.13515 (2025) 3  
10. Gui, Y., Cheng, J.: Search-r3: Unifying reasoning and embedding generation in large language models. arXiv preprint arXiv:2510.07048 (2025) 4  
11. Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al.: Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 (2025) 5, 8  
12. Hong, M., Lu, Y., Ye, N., Lin, C., Zhao, Q., Liu, S.: Unsupervised homography estimation with coplanarity-aware gan. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 17663–17672 (2022) 8  
13. Huang, W., Jia, B., Zhai, Z., Cao, S., Ye, Z., Zhao, F., Xu, Z., Hu, Y., Lin, S.: Vision-r1: Incentivizing reasoning capability in multimodal large language models. arXiv preprint arXiv:2503.06749 (2025) 10, 11  
14. Huang, X., Peng, H., Zou, D., Liu, Z., Li, J., Liu, K., Wu, J., Su, J., Yu, P.S.: Cosent: Consistent sentence embedding via similarity ranking. IEEE/ACM Transactions on Audio, Speech, and Language Processing 32, 2800–2813 (2024) 3  
15. Järvelin, K., Kekälëinen, J.: Cumulated gain-based evaluation of ir techniques. ACM Transactions on Information Systems (TOIS) 20(4), 422–446 (2002) 3  
16. Jia, C., Yang, Y., Xia, Y., Chen, Y.T., Parekh, Z., Pham, H., Le, Q., Sung, Y.H., Li, Z., Duerig, T.: Scaling up visual and vision-language representation learning with noisy text supervision. In: International conference on machine learning. pp. 4904–4916. PMLR (2021) 2, 4  
17. Jiang, T., Huang, S., Luan, Z., Wang, D., Zhuang, F.: Scaling sentence embeddings with large language models. arXiv preprint arXiv:2307.16645 (2023) 7  
18. Jiang, T., Song, M., Zhang, Z., Huang, H., Deng, W., Sun, F., Zhang, Q., Wang, D., Zhuang, F.: E5-v: Universal embeddings with multimodal large language models. arXiv preprint arXiv:2407.12580 (2024) 11  
19. Kaufmann, T., Weng, P., Bengs, V., Hüllermeier, E.: A survey of reinforcement learning from human feedback (2024) 8  
20. Kong, F., Zhang, J., Liu, Y., Zhang, H., Feng, S., Yang, X., Wang, D., Tian, Y., Zhang, F., Zhou, G., et al.: Modality curation: Building universal embeddings for advanced multimodal information retrieval. arXiv preprint arXiv:2505.19650 (2025) 1, 4  
21. Lan, Z., Niu, L., Meng, F., Zhou, J., Su, J.: Llave: Large language and vision embedding models with hardness-weighted contrastive learning. arXiv preprint arXiv:2503.04812 (2025) 4  
22. Lee, K.H., Chen, X., Hua, G., Hu, H., He, X.: Stacked cross attention for imagetext matching. In: Proceedings of the European conference on computer vision (ECCV). pp. 201–216 (2018) 4  
23. Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., Li, C.: Llava-nextinterleave: Tackling multi-image, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895 (2024) 2  
24. Li, J., Li, D., Savarese, S., Hoi, S.: Blip-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: International conference on machine learning. pp. 19730–19742. PMLR (2023) 4, 10, 11  
25. Li, J., Li, D., Xiong, C., Hoi, S.: Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In: International conference on machine learning. pp. 12888–12900. PMLR (2022) 10  
26. Li, J., Selvaraju, R., Gotmare, A., Joty, S., Xiong, C., Hoi, S.C.H.: Align before fuse: Vision and language representation learning with momentum distillation. Advances in neural information processing systems 34, 9694–9705 (2021) 4  
27. Li, K., Wang, Y., Li, Y., Wang, Y., He, Y., Wang, L., Qiao, Y.: Unmasked teacher: Towards training-efficient video foundation models. In: ICCV (2023) 12, 14  
28. Li, M., Zhang, Y., Long, D., Chen, K., Song, S., Bai, S., Yang, Z., Xie, P., Yang, A., Liu, D., Zhou, J., Lin, J.: Qwen3-vl-embedding and qwen3-vl-reranker: A unified framework for state-of-the-art multimodal retrieval and ranking. arXiv (2026) 3  
29. Li, X., Li, C., Chen, S.Z., Chen, X.: U-marvel: Unveiling key factors for universal multimodal retrieval via embedding learning with mllms. arXiv preprint arXiv:2507.14902 (2025) 9  
30. Lin, L., Long, J., Wan, Z., Wang, Y., Yang, D., Yang, S., Yao, Y., Chen, X., Guo, Z., Li, S., et al.: Sail-embedding technical report: Omni-modal embedding foundation model. arXiv preprint arXiv:2510.12709 (2025) 2  
31. Lin, S.C., Lee, C., Shoeybi, M., Lin, J., Catanzaro, B., Ping, W.: Mm-embed: Universal multimodal retrieval with multimodal llms. arXiv preprint arXiv:2411.02571 (2024) 2, 4, 10, 11, 12, 14  
32. Liu, C.Y., Zeng, L., Liu, J., Yan, R., He, J., Wang, C., Yan, S., Liu, Y., Zhou, Y.: Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451 (2024) 8  
33. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 34892–34916 (2023) 4  
34. Liu, W., Ma, X., Sun, W., Zhu, Y., Li, Y., Yin, D., Dou, Z.: Reasonrank: Empowering passage ranking with strong reasoning ability. arXiv preprint arXiv:2508.07050 (2025) 4  
35. Liu, Y., Zhang, Y., Cai, J., Jiang, X., Hu, Y., Yao, J., Wang, Y., Xie, W.: Lamra: Large multimodal model as your advanced retrieval assistant. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 4015–4025 (2025) 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12  
36. Liu, Y., Huang, Q., Hui, S., Fu, J., Zhou, S., Wu, K., Li, P., Wang, J.: Semanticaware representation learning for homography estimation. In: Proceedings of the 32nd ACM International Conference on Multimedia. pp. 2506–2514 (2024) 8  
37. Liu, Z., Sun, Z., Zang, Y., Dong, X., Cao, Y., Duan, H., Lin, D., Wang, J.: Visualrft: Visual reinforcement fine-tuning. arXiv preprint arXiv:2503.01785 (2025) 4  
38. Lyu, Y., Shao, R., Chen, G., Zhu, Y., Guan, W., Nie, L.: Puma: Layer-pruned language model for efficient unified multimodal retrieval with modality-adaptive learning. arXiv preprint arXiv:2507.08064 (2025) 1, 2, 3, 4, 6, 7, 9, 10, 12, 14  
39. Ma, X., Wang, L., Yang, N., Wei, F., Lin, J.: Fine-tuning llama for multi-stage text retrieval. In: Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval. pp. 2421–2425 (2024) 2  
40. Meng, R., Jiang, Z., Liu, Y., Su, M., Yang, X., Fu, Y., Qin, C., Chen, Z., Xu, R., Xiong, C., et al.: Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590 (2025) 2  
41. Oord, A.v.d., Li, Y., Vinyals, O.: Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 (2018) 7  
42. Pezeshki, M., Kaba, O., Bengio, Y., Courville, A.C., Precup, D., Lajoie, G.: Gradient starvation: A learning proclivity in neural networks. Advances in Neural Information Processing Systems 34, 1256–1272 (2021) 6  
43. Qi, Y., Fu, P., Li, H., Liu, Y., Jiang, C., Qin, B., Luo, Z., Luan, J.: Patchcue: Enhancing vision-language model reasoning with patch-based visual cues. arXiv preprint arXiv:2603.05869 (2026) 4  
44. Qi, Y., Zhao, Y., Zeng, Y., Bao, X., Huang, W., Chen, L., Chen, Z., Zhao, J., Qi, Z., Zhao, F.: Vcr-bench: A comprehensive evaluation framework for video chainof-thought reasoning. arXiv preprint arXiv:2504.07956 (2025) 14  
45. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PmLR (2021) 4, 10, 11  
46. Robinson, J., Sun, L., Yu, K., Batmanghelich, K., Jegelka, S., Sra, S.: Can contrastive learning avoid shortcut solutions? Advances in neural information processing systems 34, 4974–4986 (2021) 6  
47. Saito, K., Sohn, K., Zhang, X., Li, C.L., Lee, C.Y., Saenko, K., Pfister, T.: Pic2word: Mapping pictures to words for zero-shot composed image retrieval. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 19305–19314 (2023) 2  
48. Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al.: Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300 (2024) 3  
49. Shen, H., Liu, P., Li, J., Fang, C., Ma, Y., Liao, J., Shen, Q., Zhang, Z., Zhao, K., Zhang, Q., et al.: Vlm-r1: A stable and generalizable r1-style large vision-language model. arXiv preprint arXiv:2504.07615 (2025) 4, 10, 11  
50. Sun, Q., Wang, J., Yu, Q., Cui, Y., Zhang, F., Zhang, X., Wang, X.: Eva-clip-18b: Scaling clip to 18 billion parameters. arXiv preprint arXiv:2402.04252 (2024) 11  
51. Tang, Y., Yu, J., Gai, K., Zhuang, J., Xiong, G., Gou, G., Wu, Q.: Missing targetrelevant information prediction with world model for accurate zero-shot composed image retrieval. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 24785–24795 (2025) 4  
52. Wang, P., Bai, S., Tan, S., Wang, S., Fan, Z., Bai, J., Chen, K., Liu, X., Wang, J., Ge, W., et al.: Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191 (2024) 2, 4, 10, 11, 12  
53. Wang, Y., He, Y., Li, Y., Li, K., Yu, J., Ma, X., Li, X., Chen, G., Chen, X., Wang, Y., et al.: Internvid: A large-scale video-text dataset for multimodal understanding and generation. In: ICLR (2024) 12  
54. Wang, Y., Li, K., Li, X., Yu, J., He, Y., Chen, G., Pei, B., Zheng, R., Xu, J., Wang, Z., et al.: Internvideo2: Scaling video foundation models for multimodal video understanding. In: ECCV (2024) 12, 14  
55. Wang, Y., Li, K., Li, Y., He, Y., Huang, B., Zhao, Z., Zhang, H., Xu, J., Liu, Y., Wang, Z., et al.: Internvideo: General video foundation models via generative and discriminative learning. arXiv preprint arXiv:2212.03191 (2022) 12, 14  
56. Wei, C., Chen, Y., Chen, H., Hu, H., Zhang, G., Fu, J., Ritter, A., Chen, W.: Uniir: Training and benchmarking universal multimodal information retrievers. In: ECCV (2024) 3, 6, 9, 10, 11  
57. Wei, C., Chen, Y., Chen, H., Hu, H., Zhang, G., Fu, J., Ritter, A., Chen, W.: Uniir: Training and benchmarking universal multimodal information retrievers. In: European Conference on Computer Vision. pp. 387–404. Springer (2024) 4, 10  
58. Wu, K., Li, P., Fu, J., Li, Y., Wu, Y., Liu, Y., Wang, J., Zhou, S.: Event-equalized dense video captioning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 8417–8427 (June 2025) 14  
59. Wu, Y., Deng, Y., Hui, S., Liu, Y., Wu, K., Huang, W., Wang, J.: Hierarchical frequency adaptation for all-in-one image restoration. Knowledge-Based Systems p. 116049 (2026) 14  
60. Wu, Y., Deng, Y., Zhou, S., Liu, Y., Huang, W., Wang, J.: Cr-former: Single-image cloud removal with focused taylor attention. IEEE Transactions on Geoscience and Remote Sensing 62, 1–14 (2024) 14  
61. Xiao, T., Wang, S.: Towards off-policy learning for ranking policies with logged feedback. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 36, pp. 8700–8707 (2022) 4  
62. Xiaomi, L., Xia, B., Shen, B., Zhu, D., Zhang, D., Wang, G., Zhang, H., Liu, H., Xiao, J., Dong, J., et al.: Mimo: Unlocking the reasoning potential of language model–from pretraining to posttraining. arXiv preprint arXiv:2505.07608 (2025) 2  
63. Xu, J., Mei, T., Yao, T., Rui, Y.: Msr-vtt: A large video description dataset for bridging video and language. In: CVPR (2016) 14  
64. Xu, M., Dong, J., Hou, J., Wang, Z., Li, S., Gao, Z., Zhong, R., Cai, H.: Mm-r5: Multimodal reasoning-enhanced reranker via reinforcement learning for document retrieval. arXiv preprint arXiv:2506.12364 (2025) 4  
65. Yang, Z., Liu, Y., Fu, J., Sugiyama, M., Zheng, N., et al.: Shaping schema via language representation as the next frontier for llm intelligence expanding. arXiv preprint arXiv:2605.09271 (2026) 4  
66. Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al.: Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476 (2025) 9  
67. Yu, T., Yao, Y., Zhang, H., He, T., Han, Y., Cui, G., Hu, J., Liu, Z., Zheng, H.T., Sun, M., et al.: Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13807–13816 (2024) 4  
68. Zhai, X., Mustafa, B., Kolesnikov, A., Beyer, L.: Sigmoid loss for language image pre-training. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 11975–11986 (2023) 10  
69. Zhang, B., Zhang, P., Dong, X., Zang, Y., Wang, J.: Long-clip: Unlocking the longtext capability of clip. In: European Conference on Computer Vision (2024) 10, 11  
70. Zhang, K., Luan, Y., Hu, H., Lee, K., Qiao, S., Chen, W., Su, Y., Chang, M.W.: Magiclens: Self-supervised image retrieval with open-ended instructions (2024) 11  
71. Zhang, Q., Lei, Z., Zhang, Z., Li, S.Z.: Context-aware attention network for imagetext retrieval. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3536–3545 (2020) 2  
72. Zhang, S., Fang, Q., Yang, Z., Feng, Y.: Llava-mini: Efficient image and video large multimodal models with one vision token. arXiv preprint arXiv:2501.03895 (2025) 14  
73. Zhang, Y., Li, M., Long, D., Zhang, X., Lin, H., Yang, B., Xie, P., Yang, A., Liu, D., Lin, J., et al.: Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176 (2025) 2  
74. Zhao, W.X., Liu, J., Ren, R., Wen, J.R.: Dense text retrieval based on pretrained language models: A survey. ACM Transactions on Information Systems 42(4), 1–60 (2024) 2  
75. Zhou, J., Wang, X., Yu, J.: Optimizing preference alignment with differentiable ndcg ranking. arXiv preprint arXiv:2410.18127 (2024) 4  
76. Zhou, J., Xiong, Y., Liu, Z., Liu, Z., Xiao, S., Wang, Y., Zhao, B., Zhang, C.J., Lian, D.: Megapairs: Massive data synthesis for universal multimodal retrieval. In: Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 19076–19095 (2025) 4  
77. Zhu, D., Chen, J., Shen, X., Li, X., Elhoseiny, M.: Minigpt-4: Enhancing visionlanguage understanding with advanced large language models. arXiv preprint arXiv:2304.10592 (2023) 4  
78. Zhu, L., Ji, D., Chen, T., Wu, H., Wang, S.: Retrv-r1: A reasoning-driven mllm framework for universal and efficient multimodal retrieval. arXiv preprint arXiv:2510.02745 (2025) 1, 4  
79. Zhu, T., Jung, M.C., Clark, J.: Generalized contrastive learning for multi-modal retrieval and ranking. In: Companion Proceedings of the ACM on Web Conference 2025. pp. 661–670 (2025) 2