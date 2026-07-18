# Beyond Single Expert: Harmonizing Diverse Visual Priors in MLLMs for Spatial Understanding

Xiao Lin Xiaohu Huang Kai Han<sup>∗</sup> The University of Hong Kong {lllinxiao, huangxiaohu}@connect.hku.hk, kaihanx@hku.hk

## Abstract

Multimodal Large Language Models (MLLMs) have demonstrated substantial promise in spatial understanding. Existing works typically incorporate prior knowl edge extracted from a pre-trained foundation model to further enhance the spatial awareness of MLLMs. In this paper, we first reveal that when integrating diverse foundation models into MLLMs, different models provide complementary spatial priors that benefit different tasks. Motivated by this, we propose ViPS, a novel multi-model prior framework designed to fully unleash the potential of incorporating multiple Visual Priors from diverse models into MLLMs for Spatial understanding. Specifically, ViPS introduces an Efficient Prior Proxy to generate multiple foundational priors with minimal inference overhead, and a Dynamic Prior Fusion mechanism to achieve harmonious and context-aware prior fusion and injection from the prior proxies. Extensive experiments demonstrate that ViPS successfully harmonizes diverse visual priors, establishing new state-of-the-art performance across multiple complex spatial reasoning and 3D spatial understanding benchmarks. Project page: https://visual-ai.github.io/vips

![](images/e6f3a4e1ac076a9126e9bb0bcfa8ed0d6926cf6b40204db15a0807bbcc4516a8.jpg)

![](images/93f550aa3b3dbce19b927dffcb7a04c541e827dc21bba45f69a2cf0f3a555d57.jpg)

![](images/ab50b31f2979852e4a7df1134a54d24286aee3b82a9934cd31e992b345699879.jpg)  
Figure 1: Comparison of Existing Single-Expert Paradigms and Our Multi-Prior Framework. Top-left: Existing paradigms typically rely on a single external encoder (e.g., VGGT) to provide visual priors for MLLMs. Bottom-left: In contrast, our approach integrates diverse knowledge from multiple expert models into the MLLM. Middle and right: Extensive evaluations demonstrate that our method achieves state-of-the-art performance across multiple benchmarks.

## 1 Introduction

Spatial understanding serves as the foundation for real-world reasoning and interaction, acting as a pivotal technology for critical applications such as robotic navigation and embodied intelligence. The rapid advancement of Multimodal Large Language Models (MLLMs)[Wang et al., 2024, Li et al., 2024, Chen et al., 2024d, Hurst et al., 2024, Team et al., 2024, Bai et al., 2025] has brought substantial breakthroughs to this domain. By bridging the general reasoning capabilities of Large

Language Models (LLMs) with visual or point-cloud encoders [Xu et al., 2024, Qi et al., 2024], MLLMs facilitate sophisticated spatial perception and reasoning within complex environments.

Recently, numerous studies have explored incorporating knowledge from pre-trained foundation models (e.g., VGGT [Wang et al., 2025b], WAN [Wan et al., 2025]) into MLLMs [Huang et al., 2025, Wu et al., 2026, Li et al., 2026, Zheng et al., 2025a]. Typically, this paradigm involves leveraging foundational features as auxiliary inputs or aligning the latent representations of MLLMs with foundation models through distillation or feature injection. Such a paradigm has proven effective in augmenting MLLMs with domain-specific expert knowledge to boost spatial awareness. Despite their success, existing efforts are predominantly restricted to exploiting prior knowledge from a single expert, while overlooking the potential of fusing diverse knowledge from multiple foundation models. Intuitively, different foundation models capture distinct priors shaped by their varied pre-training data and objectives, making it highly desirable to aggregate their respective advantages in MLLMs. Therefore, how to synergistically harmonize these complementary strengths from distinct models remains a significant, yet unresolved challenge.

In this paper, we seek to explore two questions that form our core motivations: (i) Do different foundation models exhibit distinct specializations across various spatial understanding tasks when serving as priors for MLLMs? (ii) How can we synergistically integrate the expert knowledge from multiple foundation models into MLLMs? Regarding the first question, we conduct comprehensive empirical studies by integrating the priors of different foundation models [Wang et al., 2025b, Lin et al., 2025, Liu et al., 2025a, Wan et al., 2025, Heinrich et al., 2025] into MLLMs across a wide range of spatial understanding tasks [Yang et al., 2025, Chen et al., 2020, Zhang et al., 2023b, Chen et al., 2021, Azuma et al., 2022, Ma et al., 2022]. Our results reveal that these models contribute differently to specific tasks, indicating a strong complementarity among diverse visual priors and underscoring the necessity of multi-model integration. Based on these findings, a straightforward solution to the second question would be a naive ensemble of multiple priors. However, such an approach suffers from several critical drawbacks: First, the computational overhead becomes prohibitive, as extracting priors from each foundation model requires an independent forward pass during inference. Second, since different visual priors inherently exhibit distinct strengths and specializations, a naive ensemble not only fails to prioritize the most relevant prior knowledge for a given task, but also risks confusing the MLLM due to the distribution disparities among different foundation models.

To address these challenges, we propose ViPS (illustrated in Figure 1), a novel multi-model prior framework designed to fully unleash the potential of multiple Visual Priors in MLLMs for Spatial understanding. ViPS harmoniously integrates priors from multiple models into the MLLM by introducing an Efficient Prior Proxy and a Dynamic Prior Fusion mechanism. Specifically, the Efficient Prior Proxy is designed to generate multiple foundational priors without requiring multiple forward passes of foundation models. Instead of directly deploying all foundation models independently, we employ one base model alongside several lightweight proxies to estimate the priors of the foundation models and align the proxy outputs with the ground-truth priors during training. This design is motivated by the insight that different foundation models often share common low- and mid-level visual and geometric features. Therefore, a robust base model providing sufficient representations is enough, from which other distinct priors can be effectively distilled using lightweight proxies. Furthermore, the Dynamic Prior Fusion is proposed to achieve harmonious and context-aware prior selection from proxies. To this end, we first generate dynamic weights based on the input task, and then use these weights to aggregate the priors from different foundation models into a fused multiexpert prior, which is subsequently injected into the MLLM. Additionally, a set of zero-initialized convolutional layers is applied before aggregation to ensure that diverse priors are harmoniously fused. Extensive experiments demonstrate that ViPS achieves state-of-the-art performance across diverse spatial reasoning and spatial understanding benchmarks.

To summarize, we make the following contributions: First, we conduct an empirical study on integrating diverse foundation models into MLLMs, revealing that different visual priors exhibit distinct specializations across various tasks. Second, we propose ViPS, a novel multi-model prior framework for spatial understanding, which features an Efficient Prior Proxy to generate multimodel priors with minimal overhead, and a Dynamic Prior Fusion mechanism for harmonious and context-aware prior integration. Third, extensive experiments demonstrate that ViPS achieves new state-of-the-art performance across multiple spatial reasoning and spatial understanding benchmarks.

![](images/b73eefcc78c7bfa3a9704a35fe8633321137a464da648bb12126820c859af1db.jpg)  
Figure 2: Prior Analysis of Diverse Foundation Models. Left: The t-SNE of visual prior features extracted from different foundation models. Right: Spatial heatmaps for different models

## 2 Method

## 2.1 Empirical Study on Diverse Model Priors

Preliminary. Our work focuses on spatial understanding based on MLLMs. Formally, given a visual input comprising a sequence of frames (e.g., a video clip or multi-view images of a scene) $\mathcal { V } = \{ \stackrel { \cdot } { v _ { 1 } , } , \stackrel { \cdot } { v _ { 2 } } , \dotsc , \stackrel { \cdot } { \in } \mathbb { R } ^ { T \times H \times W \times 3 }$ , alongside a corresponding textual task instruction $\chi _ { t e x t } =$ $\{ x _ { 1 } , x _ { 2 } , \ldots , x _ { N } \}$ , the objective of the MLLM is to generate a coherent and accurate textual response $\mathcal { Y } = \{ y _ { 1 } , y _ { 2 } , . . . , y _ { M } \}$ that adequately answers the query by reasoning over the holistic spatial structures of the scene. In standard video-based MLLM architectures (e.g., LLaVA [Li et al., 2024], Qwen2-VL [Wang et al., 2024]), the visual sequence V is first processed by a generic vision encoder $\Phi _ { v i s } .$ , producing visual embeddings $F _ { v i s } = \mathbf { \bar { \Phi } } _ { v i s } ( \mathcal { V } )$ . A modality projector $\mathcal { P }$ then aligns these embeddings into the text semantic space. The LLM, denoted as $f _ { L L M }$ , takes the concatenated tokens of the projected visual features and the text instruction to auto-regressively predict the response:

$$
P (\mathcal {Y} \mid \mathcal {V}, \mathcal {X} _ {t e x t}) = \prod_ {i = 1} ^ {M} P (y _ {i} \mid \mathcal {P} (F _ {v i s}), \mathcal {X} _ {t e x t}, y _ {<   i}; \theta),\tag{1}
$$

where θ encapsulates the learnable parameters of the system. To bridge the gap in 3D spatial awareness, recent approaches explicitly introduce external foundation model priors. Given a pretrained visual expert E (e.g., encoders from foundation models like VGGT), it extracts domainspecific prior features $F _ { p r i o r } = \mathcal { E } ( \mathcal { V } )$ . The LLM leverages this supplementary knowledge to enhance generation, yielding $P ( \mathcal { V } \mid F _ { v i s } , \dot { F } _ { p r i o r } , \mathcal { X } _ { t e x t } )$ . While integrating a single prior model provides distinct utility, relying exclusively on an individual E inevitably limits the representation capacity, forming the motivation for our extensive empirical investigations.

![](images/7be6c2735130983c09ee41ecb52a6932d493b37e390d7b0f26f0c7c539e3ae3a.jpg)

![](images/64d9d1aa1c64020f9a4010f5ef7a02ad7e5ea91dbdff651c005608b1daae29e4.jpg)  
Figure 3: Relative Performance of Diverse Foundation Models. We evaluate the performance of various foundation models when serving as priors for MLLMs across a wide range of spatial understanding tasks. Notably, no single model dominates all metrics, thoroughly motivating the need for the multi-model prior integration.

Key Finding and Motivation. As highlighted in the introduction, existing efforts are restricted to exploiting the priors of a single model, overlooking the potential of fusing diverse knowledge from multiple foundation models. Fundamentally, different foundation models encapsulate distinct representations and spatial semantics, thereby endowing them with disparate advantages when serving as priors for MLLMs. To demonstrate this, we select a diverse set of foundation models for evaluation, including VGGT [Wang et al., 2025b], DepthAnything3 [Lin et al., 2025], TraceAnything [Liu et al., 2025a], Wan2.1 [Wan et al., 2025], and RADIO [Heinrich et al., 2025]. First, we conduct a prior analysis on these diverse models. As illustrated in Figure 2 (Left), the visualization demonstrates that priors extracted from distinct foundation models cluster into disparate regions within the latent space. Furthermore, the activation heatmaps in Figure 2 (Right) reveal that different models focus on varying structural and semantic cues within the identical scene. These results validate that different foundation models indeed encapsulate distinct representations and spatial cues due to their diverse training paradigms. Subsequently, we explore their relative advantages across a wide range of spatial understanding tasks when serving as priors for MLLMs (See A.3 for more details). The results are shown in Figure 3. From a column-wise perspective, the best-performing model varies across different sub-tasks. From a row-wise perspective, each specific model achieves its optimal performance on disparate sub-tasks. These results demonstrate that no single foundation model achieves universal dominance; instead, different models exhibit distinct strengths and specializations. The highly complementary nature of these diverse visual priors directly motivates our core objective: to efficiently extract and adaptively harmonize distinct prior knowledge from multiple foundation models for comprehensive spatial reasoning.

## 2.2 Proposed Framework: ViPS

Building upon the above insights, we introduce ViPS (Visual Priors for Spatial understanding), a novel framework designed to fully unleash the potential of leveraging diverse prior models in MLLMs. The overall framework is illustrated in Figure 4. ViPS first processes the input visual sequence V through a standard vision encoder $\Phi _ { v i s }$ [Zhai et al., 2023], producing the base visual embeddings $F _ { v i s } .$ Instead of relying on a single expert or naively ensembling multiple heavy models, ViPS seamlessly integrates comprehensive prior knowledge through two pivotal mechanisms: Efficient Prior Proxy and Dynamic Prior Fusion.

Specifically, the Efficient Prior Proxy utilizes a base foundation model alongside lightweight MLPs to efficiently estimate the prior features of various external models, generating a set of diverse prior representations $\{ F _ { p r i o r } ^ { 1 } , \dot { F _ { p r i o r } ^ { 2 } } , \ldots , F _ { p r i o r } ^ { K } \}$ . Subsequently, these representations are fed into the Dynamic Prior Fusion part, which first applies independent, zero-initialized convolutional layers [Zhang et al., 2023a] to each prior branch and then employs a context-aware weighting mechanism, guided by the final token of the input instruction $\chi _ { t e x t } .$ , to dynamically compute fusion weights. Finally, the harmoniously fused multi-expert prior $\hat { F } _ { p r i o r }$ is incorporated into the MLLM, empowering the LLM to leverage the most relevant visual knowledge tailored to the specific spatial reasoning task. We detail the formulations of these two components below.

Efficient Prior Proxy. Our framework aims to effectively integrate highly complementary priors from various foundation models for comprehensive spatial understanding. Directly extracting visual features from a set of K distinct foundation models, denoted as $\{ \mathcal { E } _ { 1 } , \mathcal { \bar { E } } _ { 2 } , \ldots , \mathcal { E } _ { K } \}$ , conventionally requires K independent forward passes, i.e., computing $\mathcal { E } _ { k } ( \mathcal { V } )$ for $k \in \{ 1 , \ldots , K \}$ . This paradigm introduces a severe computational bottleneck, as the inference latency and memory footprint scale linearly with the number of integrated priors. To decouple the computational overhead from the integration of multiple models while effectively preserving the rich diversity of visual prior knowledge, we propose the efficient prior proxy mechanism. Specifically, we use a single robust vision encoder as the base model $\mathcal { E } _ { b a s e }$ to extract a unified foundational representation:

$$
F _ {b a s e} = \mathcal {E} _ {b a s e} (\mathcal {V}) \in \mathbb {R} ^ {S \times D _ {b a s e}},\tag{2}
$$

where S and $D _ { b a s e }$ denote the sequence length and channel dimension of the visual tokens, respectively. In our experiments, the base model is directly initialized with the encoder of one of the foundation models (we present experiments using encoders from different foundation models as the base model in the appendix).

To approximate the distinct knowledge of the K targeted foundation models, we instantiate a set of K lightweight proxy networks. Each proxy $\phi _ { k } ( \cdot )$ is implemented as a simple Multi-Layer Perceptron (MLP) designed to extract specific prior semantics (e.g., depth cues, geometric boundaries, or detailed semantics) from the shared foundational feature $F _ { b a s e }$ . The representation for the k-th prior is uniformly formulated as:

![](images/f526335bfa0b08e3fa87fcdb4c46910617858845c30f17eaaf5941d41f4f164e.jpg)  
Figure 4: Overview of the Proposed ViPS Framework. The framework integrates distinct prior knowledge from multiple foundation models via the Efficient Prior Proxy and coordinates them using Dynamic Prior Fusion for comprehensive spatial reasoning.

$$
F _ {p r i o r} ^ {k} = \phi_ {k} (F _ {b a s e}) \in \mathbb {R} ^ {S \times D _ {p r i o r}},\tag{3}
$$

where $D _ { p r i o r } ^ { k }$ is the feature dimension for the k-th prior feature.

To guarantee the fidelity of these estimated prior representations, we explicitly supervise the output of prior proxies during the training phase. Specifically, the features extracted by the K distinct foundation models are utilized as the ground-truth targets, denoted as $F _ { g t } ^ { k } = \check { \mathcal { E } _ { k } } ( \mathcal { V } )$ . We apply an $L _ { 2 }$ loss to enforce strict alignment between the proxy outputs $F _ { p r i o r } ^ { k }$ and their corresponding ground-truth priors $F _ { g t } ^ { k }$ :

$$
\mathcal {L} _ {a l i g n m e n t} = \sum_ {k = 1} ^ {K} \left\| F _ {p r i o r} ^ {k} - F _ {g t} ^ {k} \right\| _ {2} ^ {2}.\tag{4}
$$

The viability of this proxy-based estimation stems from the intuition that different foundation models inherently share substantial low- and mid-level visual semantics; thus, a robust base representation contains sufficient foundational knowledge from which distinct, high-level prior semantics can be efficiently derived via shallow proxies. Given that the computational cost of a shallow ML $\mathrm { ~ \bf ~ P ~ } \phi _ { k }$ is negligible compared to a full foundation model, our framework can scale to an arbitrary number of diverse priors without linearly increasing inference overhead, while ensuring that the estimated visual priors remain accurate through the alignment loss.

Dynamic Prior Fusion. With the diverse prior representations $\{ F _ { p r i o r } ^ { 1 } , \ldots , F _ { p r i o r } ^ { K } \}$ efficiently established, the remaining challenge is seamlessly integrating them into the MLLM. Directly adding these distinct and heterogeneous priors fails to prioritize the most effective priors for the specific input task, and can overwhelm the language model, consequently degrading overall performance. To achieve harmonious and context-aware integration, we propose the Dynamic Prior Fusion mechanism.

Accordingly, we design a dynamic weighting mechanism to selectively focus on the precise knowledge required by the given task. Specifically, we extract the representation of the final token of the text query, denoted as $x _ { l a s t } ^ { t e x t }$ , which naturally encapsulates the aggregated semantics of the entire instruction due to the causal mechanism of the LLM. This context vector is passed through an MLP to compute a set of K-dimensional logits $z \in \mathbb { R } ^ { K }$

$$
z = \mathrm{MLP} _ {w e i g h t} (x _ {l a s t} ^ {t e x t}).\tag{5}
$$

A Softmax activation is subsequently applied to z to obtain the normalized fusion weights $w \in [ 0 , 1 ] ^ { K }$ These weights will be utilized to dynamically combine the diverse priors. However, our experiments show that directly doing so fails to yield progressive performance improvements. We attribute this phenomenon to the disparate prior distributions originating from different source models, which tend to confuse the MLLM during the early stages (see Section 2.1 and Section 3.4). To overcome this issue and enable progressive prior injection, we apply an independent zero-initialized convolutional layer, denoted as $\mathrm { Z e r o C o n v } _ { k } ( \cdot )$ , to each proxy branch prior to the weighted fusion. This strategy ensures that the outputs of the convolution are initially zero, preserving the original MLLM representations and preventing the prior perturbations from disrupting early training, while gradually scaling up as the MLLM progressively learns from the diverse prior knowledge. The processed prior $\tilde { F } _ { p r i o r } ^ { \bar { k } }$ is computed as:

$$
\tilde {F} _ {\text { prior }} ^ {k} = \operatorname{ZeroConv} _ {k} (F _ {\text { prior }} ^ {k}).\tag{6}
$$

Finally, the context-aware weights are utilized to linearly combine the aligned priors, yielding the overall harmonized multi-expert prior $\hat { F } _ { p r i o r }$ :

$$
\hat {F} _ {p r i o r} = \sum_ {k = 1} ^ {K} w _ {k} \cdot \tilde {F} _ {p r i o r} ^ {k}.\tag{7}
$$

This fused representation ${ \hat { F } } _ { p r i o r } ,$ which dynamically captures the most pertinent visual prior requested by the query, is directly added element-wise to the corresponding projected visual embeddings ${ \bar { \mathcal { P } } } ( F _ { v i s } ) { \mathrm { ~ ( i . e . } }$ , the image tokens) of the MLLM. In this way, Dynamic Prior Fusion seamlessly integrates diverse priors into the MLLM, adaptively tailoring knowledge selection to the input task while leveraging zero-initialization to ensure progressive learning without disrupting early training.

It is worth noting that, for simplicity of presentation, the above formulation describes the prior injection process at a single layer. In practice, we uniformly apply this Dynamic Prior Fusion mechanism across five layers of the MLLM. This multi-layer injection ensures that the diverse prior knowledge is deeply integrated, progressively guiding the spatial reasoning of the model.

## 3 Experiments

## 3.1 Dataset and Evaluation Metric

We evaluate our method across a total of six datasets: VSI-Bench [Yang et al., 2025] for spatial reasoning tasks, and five ScanNet-series benchmarks for 3D spatial understanding (ScanRefer [Chen et al., 2020], Multi3DRefer [Zhang et al., 2023b], Scan2Cap [Chen et al., 2021], ScanQA [Azuma et al., 2022], and SQA3D [Ma et al., 2022]). Specifically, VSI-Bench comprises eight fine-grained spatial reasoning tasks: object counting, absolute distance estimation, object size estimation, room size estimation, relative distance, relative direction, route planning, and object appearance order. Following the standard protocol, we report the accuracy (%) across all its sub-tasks. The five ScanNet-series benchmarks, derived from the ScanNet corpus, span three primary tasks: (i) 3D visual grounding, where ScanRefer targets free-form object localization and Multi3DRefer extends this to multi-target and zero-target ambiguity; (ii) dense captioning, using Scan2Cap to generate natural language descriptions for localized objects; and (iii) embodied question answering, employing ScanQA for geometry-grounded open-ended questions and SQA3D for complex situated reasoning from a specific agent perspective. Following standard protocols, we report Acc@0.25/0.5 for ScanRefer, F1@0.25/0.5 for Multi3DRefer, CIDEr@0.5 for Scan2Cap, CIDEr and Exact Match (EM) for ScanQA, and Exact Match for SQA3D. More details can be found in the Appendix.

## 3.2 Implementation Detail

We use both Qwen2-VL-7B [Wang et al., 2024] and Qwen3-VL-8B [Bai et al., 2025] as base MLLMs on VSI-Bench; for all other experiments, we adopt Qwen2-VL-7B as the default model. To comprehensively capture spatial knowledge, we integrate $\bar { K } = 5$ distinct foundation models to extract complementary visual priors. Specifically, we employ VGGT [Wang et al., 2025b] for general visual geometric feature extraction, which also serves as the base foundation model in the Efficient Prior Proxy. We further incorporate DepthAnything3 [Lin et al., 2025] for precise monocular depth cues, TraceAnything [Liu et al., 2025a] for object-centric motion and relationship tracking, Wan2.1 [Wan et al., 2025] for rich spatio-temporal dynamics, and RADIO [Heinrich et al., 2025] for robust generic visual representations. These diverse priors collectively provide a holistic understanding of 3D scenes. We uniformly sample 32 frames from each video as the input to the vision encoder. During training, we first apply the alignment loss to train the Efficient Prior Proxy, and then freeze the proxies while fine-tuning the MLLM. We freeze the vision encoder and apply Low-Rank Adaptation (LoRA) [Hu et al., 2022] to the LLM backbone during MLLM fine-tuning. We optimize the model using the Adam optimizer [Kingma and Ba, 2014] with a batch size of 16 and a maximum learning rate of $1 \times 1 0 ^ { - 5 }$ The model is trained for 1 epoch on each dataset. More details can be found in the Appendix.

Table 1: Performance Comparison on VSI-Bench. ViPS achieves a leading average score across the eight spatial reasoning sub-tasks, surpassing prior spatial-enhanced MLLMs.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Avg.</td><td>Obj. Count</td><td>Abs. Dist.</td><td>Obj. Size</td><td>Room Size</td><td>Rel. Dist.</td><td>Rel. Dir.</td><td>Route Plan</td><td>Appr. Order</td></tr><tr><td colspan="4">Numerical Answer</td><td colspan="4">Multiple-Choice Answer</td></tr><tr><td colspan="10">Proprietary Models (API)</td></tr><tr><td>GPT-4o [Hurst et al., 2024]</td><td>34.0</td><td>46.2</td><td>5.3</td><td>43.8</td><td>38.2</td><td>37.0</td><td>41.3</td><td>31.5</td><td>28.5</td></tr><tr><td>Gemini-1.5-Pro [Team et al., 2024]</td><td>45.4</td><td>56.2</td><td>30.9</td><td>64.1</td><td>43.6</td><td>51.3</td><td>46.3</td><td>36.0</td><td>34.6</td></tr><tr><td colspan="10">Open-source Models</td></tr><tr><td>LongVA-7B [Zhang et al., 2024]</td><td>29.2</td><td>38.0</td><td>16.6</td><td>38.9</td><td>22.2</td><td>33.1</td><td>43.3</td><td>25.4</td><td>15.7</td></tr><tr><td>LongVILA-8B [Chen et al., 2024c]</td><td>21.6</td><td>29.1</td><td>9.1</td><td>16.7</td><td>0.0</td><td>29.6</td><td>30.7</td><td>32.5</td><td>25.5</td></tr><tr><td>InternVL2-8B [Chen et al., 2024d]</td><td>34.6</td><td>23.1</td><td>28.7</td><td>48.2</td><td>39.8</td><td>36.7</td><td>30.7</td><td>29.9</td><td>39.6</td></tr><tr><td>InternVL2-40B [Chen et al., 2024d]</td><td>36.0</td><td>34.9</td><td>26.9</td><td>46.5</td><td>31.8</td><td>42.1</td><td>32.2</td><td>34.0</td><td>39.6</td></tr><tr><td>VILA-1.5-40B [Liu et al., 2025b]</td><td>31.2</td><td>22.4</td><td>24.8</td><td>48.7</td><td>22.7</td><td>40.5</td><td>25.7</td><td>31.5</td><td>32.9</td></tr><tr><td>LLaVA-OneVision-7B [Li et al., 2024]</td><td>32.4</td><td>47.7</td><td>20.2</td><td>47.4</td><td>12.3</td><td>42.5</td><td>35.2</td><td>29.4</td><td>24.4</td></tr><tr><td>LLaVA-OneVision-72B [Li et al., 2024]</td><td>40.2</td><td>43.5</td><td>23.9</td><td>57.6</td><td>37.5</td><td>42.5</td><td>39.9</td><td>32.5</td><td>44.6</td></tr><tr><td>LLaVA-NeXT-Video-7B [Liu et al., 2024]</td><td>35.6</td><td>48.5</td><td>14.0</td><td>47.8</td><td>24.2</td><td>43.5</td><td>42.4</td><td>34.0</td><td>30.6</td></tr><tr><td>LLaVA-NeXT-Video-72B [Liu et al., 2024]</td><td>40.9</td><td>48.9</td><td>22.8</td><td>57.4</td><td>35.3</td><td>42.4</td><td>36.7</td><td>35.0</td><td>48.6</td></tr><tr><td colspan="10">Spatial-Enhanced Models</td></tr><tr><td>vsGRPO-V-7B [Liao et al., 2025]</td><td>40.7</td><td>59.9</td><td>29.6</td><td>50.8</td><td>48.3</td><td>35.4</td><td>35.6</td><td>34.0</td><td>31.5</td></tr><tr><td>SPAR-8B [Zhang et al., 2025]</td><td>41.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SpaceR-7B [Ouyang et al., 2025]</td><td>45.6</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VG-LLM-4B [Zheng et al., 2025a]</td><td>45.9</td><td>65.6</td><td>37.4</td><td>54.8</td><td>60.2</td><td>42.3</td><td>46.3</td><td>33.0</td><td>25.9</td></tr><tr><td>VG-LLM-8B [Zheng et al., 2025a]</td><td>50.1</td><td>67.2</td><td>38.0</td><td>59.3</td><td>63.2</td><td>47.0</td><td>43.9</td><td>33.0</td><td>49.4</td></tr><tr><td>3DRS-7B [Huang et al., 2025]</td><td>45.9</td><td>68.7</td><td>34.8</td><td>53.6</td><td>56.6</td><td>40.9</td><td>43.2</td><td>30.4</td><td>39.2</td></tr><tr><td>Vega-3D [Wu et al., 2026]</td><td>50.5</td><td>69.7</td><td>35.9</td><td>58.0</td><td>60.8</td><td>45.1</td><td>43.1</td><td>30.9</td><td>60.5</td></tr><tr><td>VLM-3R [Fan et al., 2025]</td><td>57.2</td><td>70.2</td><td>49.4</td><td>69.2</td><td>67.1</td><td>65.4</td><td>80.5</td><td>45.4</td><td>40.1</td></tr><tr><td>ViPS (qwen2-vl-7b)</td><td>63.8</td><td>71.5</td><td>60.8</td><td>70.7</td><td>71.6</td><td>65.5</td><td>84.7</td><td>51.0</td><td>34.3</td></tr><tr><td>ViPS (qwen3-vl-8b)</td><td>63.8</td><td>82.6</td><td>50.5</td><td>81.4</td><td>64.1</td><td>62.8</td><td>64.1</td><td>46.2</td><td>58.3</td></tr></table>

## 3.3 Comparison with State-of-the-Art Methods

We evaluate ViPS against leading MLLMs and specialized 3D spatial understanding models. Table 1 and Table 2 summarize our results on VSI-Bench and the ScanNet-series benchmarks respectively.

Spatial Reasoning on VSI-Bench. As shown in Table 1, ViPS achieves a leading average score of 63.8%, surpassing the previous best spatial-enhanced model VLM-3R [Fan et al., 2025] (57.2%). While trailing VG-LLM-8B [Zheng et al., 2025a] in Appearance Order, ViPS dominates most other categories, proving the advantage of dynamically injecting multiple priors over relying solely on a single foundation model like VLM-3R or 3DRS [Huang et al., 2025].

3D spatial understanding on ScanNet-series. Table 2 shows our results across grounding, captioning, and QA tasks. Compared to generalist 3D-LLMs like Vega-3D [Wu et al., 2026] and 3DRS [Huang et al., 2025], ViPS delivers highly competitive performance. It yields top scores on visual grounding (ScanRefer, 64.6% Acc@0.25) and QA (ScanQA, 107.9 CIDEr), outperforming Vega-3D. Although ViPS slightly trails 3DRS on Scan2Cap, these results confirm that our unified proxy injection effectively harnesses multiple foundation priors for complex 3D spatial understanding tasks.

## 3.4 Ablation Study

Effectiveness of Individual Visual Priors. Table 3 evaluates the impact of individual foundation models compared to a baseline MLLM without prior injection. While each single prior yields distinct performance gains, our full ViPS framework integrating all five priors achieves the highest scores across all metrics. This confirms both the effectiveness of individual priors and the strong complementarity of harmonizing heterogeneous visual knowledge for 3D spatial understanding.

Effectiveness of Dynamic Prior Injection. Table 4 compares the full ViPS model against two injection variants. First, replacing zero-initialized convolutions with standard random initialization (w/o Zero-init) causes a severe performance drop. This confirms our motivation that zero-init is essential to prevent the diverse, unaligned prior distributions from confusing the MLLM during early training, preserving the original latent space and enabling progressive prior injection. Second, substituting the dynamic proxy fusion with a straightforward feature sum (Vanilla Addition) degrades performance. This demonstrates that our dynamic mechanism, which adapts to the input query, achieves more effective prior injection than static feature aggregation.

Effectiveness of Efficient Prior Proxy. To demonstrate the efficiency and accuracy of our Efficient Prior Proxy, we compare our method against an upper-bound setting (w/ GT Priors) where groundtruth features from all five foundation models are used to replace priors from the Efficient Prior Proxy. As shown in Table 5, our method employs multiple lightweight proxies to replace full foundation models, significantly reducing parameter overhead and inference latency $( 1 \times \mathrm { v s . } \sim 5 \times )$ with only a marginal drop in overall performance. We also report the estimation error (0.252 in cosine distance) between the estimated priors and the ground-truth priors during inference. These experimental results consistently demonstrate the effectiveness of our proposed approach. Furthermore, we conduct an ablation study by removing the alignment loss $( \mathcal { L } _ { a l i g n m e n t } )$ while retaining all other prior proxy structures (in this setting, the whole framework, including the MLLM and Efficient Prior Proxy, is trained end-to-end). Surprisingly, although the performance is inferior to the setting with the alignment loss, it still achieves considerable improvements compared to the baseline. We attribute this to the fact that even without the alignment loss, different prior proxies can still serve as projectors onto distinct sub-feature spaces, enabling the MLLM to learn diverse knowledge from the base model, akin to the multi-head mechanism in attention networks.

Table 2: Performance Comparison on ScanNet-Series Benchmarks. ViPS delivers competitive performance across 3D visual grounding (ScanRefer, Multi3DRefer), dense captioning (Scan2Cap), and embodied question answering (ScanQA, SQA3D).

<table><tr><td rowspan="2">Method</td><td colspan="2">ScanRefer</td><td colspan="2">Multi3DRefer</td><td>Scan2Cap</td><td colspan="2">ScanQA</td><td>SQA3D</td></tr><tr><td>Acc@0.25</td><td>Acc@0.5</td><td>F1@0.25</td><td>F1@0.5</td><td>C@0.5</td><td>CIDER</td><td>EM</td><td>EM</td></tr><tr><td colspan="9">Specialists</td></tr><tr><td>ScanRefer [Chen et al., 2020]</td><td>37.3</td><td>24.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>M3DRef-CLIP [Zhang et al., 2023b]</td><td>51.9</td><td>44.7</td><td>42.8</td><td>-</td><td>38.4</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Scan2Cap [Chen et al., 2021]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>35.2</td><td>-</td><td>-</td><td>-</td></tr><tr><td>ScanQA [Azuma et al., 2022]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>64.9</td><td>21.1</td><td>47.2</td></tr><tr><td>3D-VisTA [Zhu et al., 2023]</td><td>50.6</td><td>45.8</td><td>-</td><td>-</td><td>66.9</td><td>69.6</td><td>22.4</td><td>48.5</td></tr><tr><td colspan="9">Generalists</td></tr><tr><td>3D-LLM (BLIP2-flant5) [Hong et al., 2023]</td><td>30.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>69.4</td><td>20.5</td><td>-</td></tr><tr><td>Chat-3D v2 [Huang et al., 2023]</td><td>42.5</td><td>38.4</td><td>45.1</td><td>41.6</td><td>63.9</td><td>87.6</td><td>-</td><td>54.7</td></tr><tr><td>SceneLLM [Fu et al., 2024]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>80.0</td><td>27.2</td><td>53.6</td></tr><tr><td>Grounded 3D-LLM [Chen et al., 2024b]</td><td>47.9</td><td>44.1</td><td>45.2</td><td>40.6</td><td>70.6</td><td>72.7</td><td>-</td><td>-</td></tr><tr><td>PQ3D [Zhu et al., 2024b]</td><td>57.0</td><td>51.2</td><td>-</td><td>50.1</td><td>80.3</td><td>-</td><td>-</td><td>47.1</td></tr><tr><td>ChatScene [Huang et al., 2023]</td><td>55.5</td><td>50.2</td><td>57.1</td><td>52.4</td><td>77.1</td><td>87.7</td><td>21.6</td><td>54.6</td></tr><tr><td>LLaVA-3D [Zhu et al., 2024a]</td><td>54.1</td><td>42.4</td><td>-</td><td>-</td><td>79.2</td><td>91.7</td><td>27.0</td><td>55.6</td></tr><tr><td>Inst3D-LLM [Yu et al., 2025]</td><td>57.8</td><td>51.6</td><td>58.3</td><td>53.5</td><td>79.7</td><td>88.6</td><td>24.6</td><td>-</td></tr><tr><td>3D-LLaVA [Deng et al., 2025]</td><td>51.2</td><td>40.6</td><td>-</td><td>-</td><td>78.8</td><td>92.6</td><td>-</td><td>54.5</td></tr><tr><td>Video-3D LLM [Zheng et al., 2025b]</td><td>58.1</td><td>51.7</td><td>58.0</td><td>52.7</td><td>83.8</td><td>102.1</td><td>30.1</td><td>58.6</td></tr><tr><td>3DRS [Huang et al., 2025]</td><td>62.9</td><td>56.1</td><td>60.4</td><td>54.9</td><td>86.1</td><td>104.8</td><td>30.3</td><td>60.6</td></tr><tr><td>Vega-3D [Wu et al., 2026]</td><td>63.2</td><td>56.2</td><td>60.8</td><td>55.1</td><td>83.2</td><td>106.3</td><td>30.4</td><td>61.3</td></tr><tr><td>ViPS</td><td>64.6</td><td>57.6</td><td>62.0</td><td>56.5</td><td>85.5</td><td>107.9</td><td>31.6</td><td>62.5</td></tr></table>

Table 3: Ablation on Individual Visual Priors. Each foundation-model prior yields distinct gains over the no-prior baseline, and combining all five priors achieves the best results across the ScanNet series benchmarks.

<table><tr><td rowspan="2">Method</td><td colspan="2">ScanRefer</td><td colspan="2">Multi3DRefer</td><td>Scan2Cap</td><td colspan="2">ScanQA</td><td>SQA3D</td></tr><tr><td>Acc@0.25</td><td>Acc@0.5</td><td>F1@0.25</td><td>F1@0.5</td><td>C@0.5</td><td>C</td><td>EM</td><td>EM</td></tr><tr><td>Baseline</td><td>62.1</td><td>54.6</td><td>59.6</td><td>54.4</td><td>81.4</td><td>104.6</td><td>30.5</td><td>60.8</td></tr><tr><td>+ RADIO</td><td>62.9</td><td>56.0</td><td>61.2</td><td>55.5</td><td>82.6</td><td>105.1</td><td>30.7</td><td>61.1</td></tr><tr><td>+ DepthAnything3</td><td>62.9</td><td>56.2</td><td>61.0</td><td>55.7</td><td>81.8</td><td>106.2</td><td>30.9</td><td>61.2</td></tr><tr><td>+ TraceAnything</td><td>62.8</td><td>55.8</td><td>60.6</td><td>55.2</td><td>82.7</td><td>106.7</td><td>30.8</td><td>61.4</td></tr><tr><td>+ VGGT</td><td>63.2</td><td>56.3</td><td>61.1</td><td>55.8</td><td>81.9</td><td>106.6</td><td>31.1</td><td>61.6</td></tr><tr><td>+ Wan2.1</td><td>62.7</td><td>55.8</td><td>61.1</td><td>55.5</td><td>81.3</td><td>105.5</td><td>30.6</td><td>61.2</td></tr><tr><td>ViPS (Full)</td><td>64.6</td><td>57.6</td><td>62.0</td><td>56.5</td><td>85.5</td><td>107.9</td><td>31.6</td><td>62.5</td></tr></table>

## 3.5 Analysis of Dynamic Prior Weights

To further understand the behavior of our Dynamic Prior Fusion mechanism, we visualize the learned fusion weights $( w _ { k }$ in Equation 7) across different spatial understanding tasks. As illustrated in Figure 5, the weight distribution adapts dynamically depending on different downstream tasks (e.g., ScanQA, ScanRefer, Multi3DRefer, SQA3D, and Scan2Cap), as well as across distinct question types within the same dataset $( \mathrm { e . g . }$ , VSI-Bench). This observation aligns with our motivation that different foundation models exhibit distinct specializations, which shows that our framework successfully achieves context-aware prior selection by dynamically adjusting the fusion weights based on the specific scenario rather than statically relying on a single expert.

Table 4: Ablation on the Dynamic Prior Injection. Both the zero-initialized convolution and the context-aware dynamic fusion are essential for harmonizing diverse priors without disrupting early training.

<table><tr><td rowspan="2">Method</td><td colspan="2">ScanRefer</td><td colspan="2">Multi3DRefer</td><td>Scan2Cap</td><td colspan="2">ScanQA</td><td>SQA3D</td></tr><tr><td>Acc@0.25</td><td>Acc@0.5</td><td>F1@0.25</td><td>F1@0.5</td><td>C@0.5</td><td>C</td><td>EM</td><td>EM</td></tr><tr><td>Baseline</td><td>62.1</td><td>54.6</td><td>59.6</td><td>54.4</td><td>81.4</td><td>104.6</td><td>30.5</td><td>60.8</td></tr><tr><td>w/o Zero-init</td><td>63.1</td><td>56.1</td><td>61.0</td><td>55.6</td><td>76.4</td><td>104.7</td><td>30.8</td><td>61.1</td></tr><tr><td>Vanilla Addition</td><td>64.2</td><td>57.4</td><td>61.2</td><td>56.0</td><td>82.6</td><td>106.4</td><td>30.9</td><td>61.9</td></tr><tr><td>ViPS (Full)</td><td>64.6</td><td>57.6</td><td>62.0</td><td>56.5</td><td>85.5</td><td>107.9</td><td>31.6</td><td>62.5</td></tr></table>

Table 5: Ablation on the Efficient Prior Proxy. The lightweight proxies match the upper bound ( w/ GT Priors , where features are extracted via independent forward passes of the original foundation models) with only a marginal performance drop, while reducing parameter and inference cost from ∼5× to 1×.

<table><tr><td rowspan="2">Method</td><td colspan="3">Efficiency</td><td colspan="2">ScanRefer</td><td colspan="2">Multi3DRefer</td><td>S2C</td><td colspan="2">ScanQA</td><td>SQA</td></tr><tr><td>Param.</td><td>Latency</td><td>Err. ↓</td><td>Acc@0.25</td><td>Acc@0.5</td><td>F1@0.25</td><td>F1@0.5</td><td>C@0.5</td><td>C</td><td>EM</td><td>EM</td></tr><tr><td>w/ GT Priors</td><td> $\sim 5\times$ </td><td> $\sim 5\times$ </td><td>0</td><td>65.4</td><td>58.3</td><td>62.9</td><td>57.4</td><td>86.0</td><td>107.9</td><td>32.1</td><td>63.2</td></tr><tr><td>w/o  $\mathcal{L}_{alignment}$ </td><td>1×</td><td>1×</td><td>-</td><td>64.0</td><td>57.2</td><td>61.3</td><td>55.9</td><td>83.4</td><td>106.3</td><td>30.7</td><td>61.1</td></tr><tr><td>ViPS (Ours)</td><td>1×</td><td>1×</td><td>0.252</td><td>64.6</td><td>57.6</td><td>62.5</td><td>56.8</td><td>85.5</td><td>107.9</td><td>31.6</td><td>62.5</td></tr></table>

## 4 Related Work

## 4.1 Spatial Understanding with Large Language Models

Spatial understanding, a foundational pillar for real-world interaction and reasoning, has witnessed a paradigm shift with the advent of LLMs [Brown et al., 2020, Ouyang et al., 2022, Touvron et al., 2023, Achiam et al., 2023]. Early attempts, such as PointLLM [Xu et al., 2024], PointBind [Guo et al., 2023], GPT4Point [Qi et al., 2024], MiniGPT-3D [Tang et al., 2024], and Chat-3D [Wang et al., 2023], focused on aligning 3D point-cloud encoders directly with the LLM embedding space. To facilitate more effective cross-modal feature fusion, subsequent frameworks like Grounded-3D-LLM [Chen et al., 2024b], LL3DA [Chen et al., 2024a], 3D-LLaVA [Deng et al., 2025], and Inst3D-LLM [Yu et al., 2025] introduced advanced representation learning schemes. However, the inherent scarcity and noise of 3D point-cloud data often limit the scalability of these methods.

Recent research has gravitated towards video-based inputs. Prominent works such as 3D-LLM [Hong et al., 2023], Scene-LLM [Fu et al., 2024], Video-3D LLM [Zheng et al., 2025b], GPT4Scene [Qi et al., 2025] and SpatialStack [Zhang et al., 2026] establish dense correlations between 2D features and 3D scenes by building upon powerful pre-trained MLLMs [Li et al., 2024, Wang et al., 2024]. Specifically, Scene-LLM [Fu et al., 2024] captures fine-grained 3D knowledge through efficient 3D visual representation learning, while Video-3D LLM [Zheng et al., 2025b] introduces position-aware encodings for video sequences. Similarly, LLaVA-3D [Zhu et al., 2024a] achieves robust perception by learning a set of 3D voxels. While our work also falls within the video-input MLLM paradigm, we diverge from these approaches by exploring how to synergistically integrate prior knowledge from diverse foundation models to further elevate 3D spatial understanding.

## 4.2 Integration of Foundation Model Prior in MLLMs

The rapid evolution of foundation models—such as VGGT [Wang et al., 2025b], DepthAnything3 [Lin et al., 2025], and WAN [Wan et al., 2025]—has inspired a new line of research that injects diverse foundational priors into MLLMs to bolster their perceptual capabilities. Recent methods including VG-LLM [Zheng et al., 2025a], 3DRS [Huang et al., 2025], MiLO [Cao et al., 2025], VLM-3R [Fan et al., 2025], Vega-3D [Wu et al., 2026], ROSS3D [Wang et al., 2025a], and GeoThinker [Li et al., 2026] have demonstrated the efficacy of this paradigm. Specifically, VG-LLM employs a 3D visual geometry encoder to extract geometric priors from video sequences, while 3DRS aligns the latent space of MLLMs with VGGT via knowledge distillation. GeoThinker introduces an active perception mechanism for MLLMs to retrieve necessary geometric features, and Vega-3D injects world knowledge from video generation models, predicated on the hypothesis that these models inherently capture the underlying dynamics of scene transitions. In summary, existing efforts have predominantly focused on: (i) more efficient mechanisms for utilizing prior knowledge, and (ii) evaluating the performance of different individual model priors. In contrast, we propose ViPS, an efficient multi-model prior framework. By adaptively harmonizing various priors from disparate sources, our method significantly improves MLLM performance in spatial understanding tasks.

![](images/a1b6afaaa2e27cb0f0d8c19714cc695ad0218ae381b20b74b9ec6b29834ece7d.jpg)  
Figure 5: Distribution of Dynamic Prior Weights. This figure illustrates the distribution of the learned fusion weights (w<sub>k</sub> in Equation 7) assigned to different foundation models. Left: The weight distribution on the test sets of ScanQA, ScanRefer, Multi3DRefer, SQA3D, and Scan2Cap. Right: The weight distribution across different question types in VSI-Bench.

## 5 Conclusion

In this paper, we have introduced ViPS, a multi-model prior framework that unifies heterogeneous foundation-model priors within a single MLLM for spatial understanding. Our empirical study first reveals that no single foundation model dominates across tasks: different models supply distinct and complementary spatial knowledge. Building on this insight, ViPS couples an Efficient Prior Proxy, which distills the knowledge of multiple foundation models into lightweight branches sharing a common backbone and thereby avoids the linear cost of independent forward passes, with a Dynamic Prior Fusion mechanism that adaptively re-weights the resulting priors according to the input query and injects them through zero-initialized convolutions for stable training. On VSI-Bench and five ScanNet-series benchmarks, ViPS delivers state-of-the-art results in spatial reasoning, 3D visual grounding, dense captioning, and embodied question answering, while retaining single-encoder-level inference cost. These results suggest that context-aware harmonization of complementary priors is a promising path toward spatially grounded multimodal reasoning.

## Acknowledgements

This work is supported by Hong Kong Research Grants Council – General Research Fund (Grant No. 17213825), Hong Kong Innovation and Technology Commission – Innovation and Technology Fund (Grant No. ITS/488/24FP), and HKU Seed Fund for PI Research.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.

Daichi Azuma, Taiki Miyanishi, Shuhei Kurita, and Motoaki Kawanabe. Scanqa: 3d question answering for spatial scene understanding. In CVPR, 2022.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

Tom Brown, Benjamin Mann, Nick Ryder, et al. Language models are few-shot learners. In NeurIPS, 2020.

Meng Cao, Haokun Lin, Haoyuan Li, et al. Seeing through imagination: Learning scene geometry via implicit spatial world modeling. arXiv preprint arXiv:2512.01821, 2025.

Dave Zhenyu Chen, Angel X Chang, and Matthias Nießner. Scanrefer: 3d object localization in rgb-d scans using natural language. In ECCV, 2020.

Sijin Chen, Xin Chen, Chi Zhang, et al. Ll3da: Visual interactive instruction tuning for omni-3d understanding reasoning and planning. In CVPR, 2024a.

Yilun Chen, Shuai Yang, Haifeng Huang, et al. Grounded 3d-llm with referent tokens. arXiv preprint arXiv:2405.10370, 2024b.

Yukang Chen, Fuzhao Xue, Dacheng Li, et al. Longvila: Scaling long-context visual language models for long videos. In ICLR, 2024c.

Zhe Chen, Jiannan Wu, Wenhai Wang, et al. Internvl: Scaling up vision foundation models and aligning for generic visual-linguistic tasks. In CVPR, 2024d.

Zhenyu Chen, Ali Gholami, Matthias Nießner, and Angel X Chang. Scan2cap: Context-aware dense captioning in rgb-d scans. In CVPR, 2021.

Jiajun Deng, Tianyu He, Li Jiang, et al. 3d-llava: Towards generalist 3d lmms with omni superpoint transformer. In CVPR, 2025.

Zhiwen Fan, Jian Zhang, Renjie Li, et al. Vlm-3r: Vision-language models augmented with instruction-aligned 3d reconstruction. arXiv preprint arXiv:2505.20279, 2025.

Rao Fu, Jingyu Liu, Xilun Chen, et al. Scene-llm: Extending language model for 3d visual understanding and reasoning. arXiv preprint arXiv:2403.11401, 2024.

Ziyu Guo, Renrui Zhang, Xiangyang Zhu, et al. Point-bind & point-llm: Aligning point cloud with multimodality for 3d understanding, generation, and instruction following. arXiv preprint arXiv:2309.00615, 2023.

Greg Heinrich, Mike Ranzinger, Hongxu Yin, et al. Radiov2.5: Improved baselines for agglomerative vision foundation models. In CVPR, 2025.

Yining Hong, Haoyu Zhen, Peihao Chen, et al. 3d-llm: Injecting the 3d world into large language models. In NeurIPS, 2023.

Edward J Hu, Yelong Shen, Phillip Wallis, et al. Lora: Low-rank adaptation of large language models. In ICLR, 2022.

Haifeng Huang, Zehan Wang, Rongjie Huang, et al. Chat-3d v2: Bridging 3d scene and large language models with object identifiers. arXiv preprint arXiv:2312.08168, 2023.

Xiaohu Huang, Jingjing Wu, Qunyi Xie, and Kai Han. 3drs: Mllms need 3d-aware representation supervision for scene understanding. arXiv preprint arXiv:2506.01946, 2025.

Aaron Hurst, Adam Lerer, Adam P Goucher, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

Bo Li, Yuanhan Zhang, Dong Guo, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

Haoyuan Li, Qihang Cao, Tao Tang, et al. Thinking with geometry: Active geometry integration for spatial reasoning. arXiv preprint arXiv:2602.06037, 2026.

Zhenyi Liao, Qingsong Xie, Yanhao Zhang, et al. Improved visual-spatial reasoning via r1-zero-like training. arXiv preprint arXiv:2504.00883, 2025.

Haotong Lin, Sili Chen, Junhao Liew, et al. Depth anything 3: Recovering the visual space from any views. arXiv preprint arXiv:2511.10647, 2025.

Haotian Liu, Chunyuan Li, Yuheng Li, et al. Llava-next: Improved reasoning, ocr, and world knowledge. arXiv preprint, 2024.

Xinhang Liu, Yuxi Xiao, Donny Y Chen, et al. Trace anything: Representing any video in 4d via trajectory fields. arXiv preprint arXiv:2510.13802, 2025a.

Zhijian Liu, Ligeng Zhu, Baifeng Shi, et al. Nvila: Efficient frontier visual language models. In CVPR, 2025b.

Xiaojian Ma, Silong Yong, Zilong Zheng, et al. Sqa3d: Situated question answering in 3d scenes. arXiv preprint arXiv:2210.07474, 2022.

Kun Ouyang, Yuanxin Liu, Haoning Wu, et al. Spacer: Reinforcing mllms in video spatial reasoning. arXiv preprint arXiv:2504.01805, 2025.

Long Ouyang, Jeffrey Wu, Xu Jiang, et al. Training language models to follow instructions with human feedback. In NeurIPS, 2022.

Zhangyang Qi, Ye Fang, Zeyi Sun, et al. Gpt4point: A unified framework for point-language understanding and generation. In CVPR, 2024.

Zhangyang Qi, Zhixiong Zhang, Ye Fang, et al. Gpt4scene: Understand 3d scenes from videos with visionlanguage models. arXiv preprint arXiv:2501.01428, 2025.

Yuan Tang, Xu Han, Xianzhi Li, et al. Minigpt-3d: Efficiently aligning 3d point clouds with large language models using 2d priors. In ACM MM, 2024.

Gemini Team, Petko Georgiev, Ving Ian Lei, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023.

Team Wan, Ang Wang, Baole Ai, et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.

Haochen Wang, Yucheng Zhao, Tiancai Wang, et al. Ross3d: Reconstructive visual instruction tuning with 3d-awareness. In ICCV, 2025a.

Jianyuan Wang, Minghao Chen, Nikita Karaev, et al. Vggt: Visual geometry grounded transformer. In CVPR, 2025b.

Peng Wang, Shuai Bai, Sinan Tan, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.

Zehan Wang, Haifeng Huang, Yang Zhao, et al. Chat-3d: Data-efficiently tuning large language model for universal dialogue of 3d scenes. arXiv preprint arXiv:2308.08769, 2023.

Xianjin Wu, Dingkang Liang, Tianrui Feng, et al. Generation models know space: Unleashing implicit 3d priors for scene understanding. arXiv preprint arXiv:2603.19235, 2026.

Runsen Xu, Xiaolong Wang, Tai Wang, et al. Pointllm: Empowering large language models to understand point clouds. In ECCV, 2024.

Jihan Yang, Shusheng Yang, Anjali W Gupta, et al. Thinking in space: How multimodal large language models see, remember, and recall spaces. In CVPR, 2025.

Hanxun Yu, Wentong Li, Song Wang, et al. Inst3d-lmm: Instance-aware 3d scene understanding with multimodal instruction tuning. In CVPR, 2025.

Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023.

Jiahui Zhang, Yurui Chen, Yanpeng Zhou, et al. From flatland to space: Teaching vision-language models to perceive and reason in 3d. arXiv preprint arXiv:2503.22976, 2025.

Jiang Zhang, Shijie Zhou, Bangya Liu, et al. Spatialstack: Layered geometry-language fusion for 3d vlm spatial reasoning. arXiv preprint arXiv:2603.27437, 2026.

Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In ICCV, 2023a.

Peiyuan Zhang, Kaichen Zhang, Bo Li, et al. Long context transfer from language to vision. arXiv preprint arXiv:2406.16852, 2024.

Yiming Zhang, ZeMing Gong, and Angel X Chang. Multi3drefer: Grounding text description to multiple 3d objects. In ICCV, 2023b.

Duo Zheng, Shijia Huang, Yanyang Li, and Liwei Wang. Learning from videos for 3d world: Enhancing mllms with 3d vision geometry priors. arXiv preprint arXiv:2505.24625, 2025a.

Duo Zheng, Shijia Huang, and Liwei Wang. Video-3d llm: Learning position-aware video representation for 3d scene understanding. In CVPR, 2025b.

Chenming Zhu, Tai Wang, Wenwei Zhang, et al. Llava-3d: A simple yet effective pathway to empowering lmms with 3d-awareness. arXiv preprint arXiv:2409.18125, 2024a.

Ziyu Zhu, Xiaojian Ma, Yixin Chen, et al. 3d-vista: Pre-trained transformer for 3d vision and text alignment. In ICCV, 2023.

Ziyu Zhu, Zhuofan Zhang, Xiaojian Ma, et al. Unifying 3d vision-language understanding via promptable queries. In ECCV, 2024b.

## Appendix

## A Detailed Training Dataset Description

## A.1 ScanNet-series Dataset

For the 3D spatial understanding tasks, our training protocol aligns with the configurations established by previous works such as Video-3D LLM and 3DRS, which aggregate data from five distinct benchmarks, yielding a combined training set of approximately 223K sample pairs. Specifically, the composition includes:

• ScanRefer and Scan2Cap: Each dataset contributes 36,665 QA instances.

• Multi3DRefer: This multi-target grounding dataset provides 43,838 data entries.

• ScanQA: We utilize 26,515 question-answering pairs from this dataset.

• SQA3D: Serving as the largest subset in our collection, SQA3D contributes 79,445 samples.

With the exception of SQA3D, which is sourced from 518 unique 3D environments, the remaining four datasets are constructed upon 562 unique scans from the ScanNet corpus.

## A.2 VSI-Bench Dataset

For the spatial reasoning evaluation, we follow the dataset composition presented in VLM-3R to train our model on VSI-Bench. Our training subset comprises a total of 207,658 instruction-tuning QA pairs. The dataset is structurally diverse, encompassing various scene typologies and specific reasoning tasks. The detailed distribution is as follows:

• ScanNet++ QA: The majority of the data originates from the high-fidelity ScanNet++ corpus, accounting for 135,119 QA pairs.

• ScanNet QA: An additional 51,630 QA pairs are derived from ScanNet.

• Absolute Distance Estimation: 16,805 samples are dedicated to object absolute distance estimation tasks.

• Route Planning: The dataset also includes 4,104 instances specifically designed to evaluate and enhance the model’s route planning and navigational reasoning capabilities.

## A.3 Details of Section 2.1

To ensure computational efficiency during our exploration, here we train the models on a 10% subset of the original training datasets used in 3DRS [Huang et al., 2025] and VLM-3R [Fan et al., 2025]. We conduct our empirical assessment on the VSI-Bench [Yang et al., 2025] and ScanNetseries benchmarks including ScanRefer [Chen et al., 2020], Multi3DRefer [Zhang et al., 2023b], Scan2Cap [Chen et al., 2021], ScanQA [Azuma et al., 2022], and SQA3D [Ma et al., 2022]. We employ Qwen2-VL [Wang et al., 2024] as our base MLLM. For each evaluated variant, we extract features from each foundation model via its respective encoder and project them into the same hidden dimension as the MLLM using an MLP. Then inject them into the image tokens of the MLLM via addition. The evaluation metrics are consistent with those in the main text.

## B Training Detail

A comprehensive summary of all detailed hyperparameters and configurations is provided in Table 6 and Table 7. To align the external visual priors with the internal representations, we downsample all features extracted by the prior models to match the resolution of the MLLM’s image tokens.

The trainable parameters in our ViPS framework include the LLM backbone (updated via LoRA), the multimodal projector (mm\_projector), the zero-initialized convolutions and MLPs responsible for generating dynamic weights within the Dynamic Prior Fusion, and all parameters within the Efficient Prior Proxy. Under our hardware configuration, the training process takes approximately 24 hours for the ScanNet-series datasets and 30 hours for the VSI-Bench dataset.

Table 6: Training Hyperparameters and Hardware Configuration. Detailed settings used to train ViPS (qwen2-vl-7b) on the ScanNet-series benchmarks.

<table><tr><td>Configuration / Hyperparameter</td><td>Value</td></tr><tr><td>Model Initialization</td><td></td></tr><tr><td>Base MLLM</td><td>LLaVA-Video-7B-Qwen2</td></tr><tr><td>Vision Tower</td><td>google/siglip-so400m-patch14-384</td></tr><tr><td>Training Settings</td><td></td></tr><tr><td>Hardware</td><td> $8 \times 48GB$  GPUs</td></tr><tr><td>Distributed Strategy</td><td>DeepSpeed ZeRO Stage 2</td></tr><tr><td>Data Precision</td><td>bfloat16 (bf16)</td></tr><tr><td>Gradient Checkpointing</td><td>True</td></tr><tr><td>Optimization Hyperparameters</td><td></td></tr><tr><td>Training Epochs</td><td>1</td></tr><tr><td>Per-device Batch Size</td><td>1</td></tr><tr><td>Gradient Accumulation Steps</td><td>2</td></tr><tr><td>Total Effective Batch Size</td><td>16</td></tr><tr><td>Base Learning Rate</td><td> $2 \times 10^{-5}$ </td></tr><tr><td>Warmup Ratio</td><td>0.03</td></tr><tr><td>LoRA Configurations</td><td></td></tr><tr><td>LoRA Rank</td><td>512</td></tr><tr><td>LoRA Alpha</td><td>1024</td></tr><tr><td>Prior Features Alignment</td><td></td></tr><tr><td>Target Spatial Resolution</td><td> $14 \times 14$ </td></tr><tr><td>Downsampling Method</td><td>Bilinear</td></tr></table>

## C Impact of Different Base Model

In the main paper, we default to using VGGT as the base foundation model within our Efficient Prior Proxy. To further investigate the robustness and flexibility of our ViPS framework, we conduct additional experiments by substituting VGGT with other foundation models as the base model, while keeping the rest of the framework unchanged.

As shown in Table 8, using different base models yields comparable and consistently strong performance across the ScanNet-series benchmarks. Although VGGT yields the best overall performance, we observe that TraceAnything achieves better results on certain metrics (e.g., 62.02 F1@0.25 on Multi3DRefer and 85.53 C@0.5 on Scan2Cap). Furthermore, all other foundation models maintain highly comparable performance when used as the base model. These results demonstrate that our Efficient Prior Proxy and Dynamic Prior Fusion mechanisms are robust and not strictly dependent on a specific base model choice.

## D Visualization Result

To further demonstrate the effectiveness of our proposed ViPS framework, we provide visualization results in Figure 6. In this comparison, the baseline model is obtained by fine-tuning the Qwen2-VL architecture on the same training dataset used for our model. As illustrated, by synergistically harmonizing diverse visual priors, our ViPS framework achieves a more precise understanding of complex spatial relationships and generates more accurate responses compared to the baseline.

## E Broader Impact

Positive Impacts. The capabilities introduced by our ViPS framework significantly advance the spatial reasoning and 3D spatial understanding of MLLMs. By effectively harmonizing diverse visual priors, our approach paves the way for more intelligent embodied agents, such as domestic robots and autonomous navigation systems. Furthermore, our Efficient Prior Proxy minimizes inference overhead, facilitating the deployment of sophisticated 3D perception systems on resource-constrained edge devices. This efficiency can democratize access to advanced AI, fostering innovations in assistive applications for visually impaired individuals to navigate complex environments safely.

Question\_type: obj\_appearance\_order  
![](images/9454f8b021281803884998d050b39a22809771d92462b98973a053930c318fc4.jpg)

Question: What will be the first-time appearance order of the following categories in the video: blanket, door, basket, table? Options: A. door, table, basket, blanket B. door, blanket, basket, table C. blanket, door, basket, table D. door, basket, blanket, table GT: D Our Prediction:D Baseline Prediction:A

Question\_type: route\_planning  
![](images/5310b86384a531e498a319e72d41f8a3827ccbc0b64bdced2b64d5937b659a53.jpg)  
Question: You are a robot beginning at the tv facing the bed. You want to navigate to the trash bin. You will perform the following actions (Note: for each [please fill in], choose either 'turn back,' 'turn left,' or 'turn right.'): 1. [please fill in] 2. Go forward until the cabinet 3. [please fill in] 4. Go forward until the trash bin is on your right. You have reached the final destination. Options: A. Turn Left, Turn Left B. Turn Right, Turn Left C. Turn Back, Turn Left D. Turn Right, Turn Right GT: B Our Prediction:B Baseline Prediction:D

Question\_type: object\_rel\_distance  
![](images/4b0c26516dcf2f6b18d884f25f6c1c0627fb2dce64440d82ec48003f7b09179d.jpg)

Question: Measuring from the closest point of each object, which of these objects (power strip, computer tower, bookshelf, laptop) is the closest to the whiteboard? Options: A. power strip B. computer tower C. bookshelf D. laptop GT: C Our Prediction:C Baseline Prediction:D

Question\_type: object\_abs\_distance  
![](images/dcd8a32a85b943fd389c8c7cd672a3f05294c18ae65a28d696510dd2afdcbfda.jpg)  
Question: Measuring from the closest point of each object, what is the distance between the tv and the stove (in meters)? GT: 2.9 Our Prediction:2.9 Baseline Prediction:2.4

Figure 6: Qualitative Visualization on VSI-Bench. We compare our ViPS framework against the baseline, which is obtained by fine-tuning Qwen2-VL on our identical training dataset, showing that ViPS produces more accurate responses to complex spatial questions.

Table 7: Training Hyperparameters and Hardware Configuration. Detailed settings used to train ViPS (qwen3-vl-8b) on the ScanNet-series and VSI-Bench datasets.

<table><tr><td>Configuration / Hyperparameter</td><td>Value</td></tr><tr><td>Model Initialization</td><td></td></tr><tr><td>Base MLLM</td><td>Qwen/Qwen3-VL-8B-Instruct</td></tr><tr><td>Vision Tower</td><td>Qwen3VLVisionModel</td></tr><tr><td>Training Settings</td><td></td></tr><tr><td>Hardware</td><td> $8 \times 48GB$  GPUs</td></tr><tr><td>Distributed Strategy</td><td>DeepSpeed ZeRO Stage 2</td></tr><tr><td>Data Precision</td><td>bfloat16 (bf16)</td></tr><tr><td>Gradient Checkpointing</td><td>True</td></tr><tr><td>Optimization Hyperparameters</td><td></td></tr><tr><td>Training Epochs</td><td>1</td></tr><tr><td>Per-device Batch Size</td><td>1</td></tr><tr><td>Gradient Accumulation Steps</td><td>2</td></tr><tr><td>Total Effective Batch Size</td><td>16</td></tr><tr><td>Base Learning Rate</td><td> $2 \times 10^{-5}$ </td></tr><tr><td>Warmup Ratio</td><td>0.03</td></tr><tr><td>LoRA Configurations</td><td></td></tr><tr><td>LoRA Rank</td><td>256</td></tr><tr><td>LoRA Alpha</td><td>256</td></tr><tr><td>Prior Features Alignment</td><td></td></tr><tr><td>Target Spatial Resolution</td><td> $15 \times 20$ </td></tr><tr><td>Downsampling Method</td><td>Bilinear</td></tr></table>

Table 8: Robustness to the Choice of Base Model. Substituting VGGT with other foundation models in the Efficient Prior Proxy yields comparable performance across the ScanNet-series benchmarks, indicating that ViPS is not strictly dependent on a specific base-model choice.

<table><tr><td rowspan="2">Base Model</td><td colspan="2">ScanRefer</td><td colspan="2">Multi3DRefer</td><td>Scan2Cap</td><td colspan="2">ScanQA</td><td>SQA3D</td></tr><tr><td>Acc@0.25</td><td>Acc@0.5</td><td>F1@0.25</td><td>F1@0.5</td><td>C@0.5</td><td>CIDER</td><td>EM</td><td>EM</td></tr><tr><td>Baseline</td><td>62.1</td><td>54.6</td><td>59.6</td><td>54.4</td><td>81.4</td><td>104.6</td><td>30.5</td><td>60.8</td></tr><tr><td>VGGT</td><td>64.6</td><td>57.6</td><td>62.0</td><td>56.5</td><td>85.5</td><td>107.9</td><td>31.6</td><td>62.5</td></tr><tr><td>Wan</td><td>64.4</td><td>57.4</td><td>62.0</td><td>56.3</td><td>83.0</td><td>106.0</td><td>31.1</td><td>61.4</td></tr><tr><td>TraceAnything</td><td>63.8</td><td>56.7</td><td>62.0</td><td>56.4</td><td>85.5</td><td>107.0</td><td>31.3</td><td>61.8</td></tr><tr><td>DepthAnything</td><td>63.7</td><td>56.8</td><td>61.8</td><td>56.4</td><td>82.2</td><td>107.0</td><td>31.1</td><td>61.7</td></tr><tr><td>RADIO</td><td>64.1</td><td>57.2</td><td>61.7</td><td>56.4</td><td>82.6</td><td>107.3</td><td>31.2</td><td>62.5</td></tr></table>

Potential Negative Impacts and Mitigation. Endowing AI systems with highly accurate 3D spatial understanding capabilities inherently presents privacy and security risks. If deployed irresponsibly, such technologies could be exploited for unauthorized surveillance or the invasive reconstruction of private indoor spaces. To mitigate these ethical concerns, future real-world applications must incorporate robust anonymization pipelines—such as the automatic obfuscation of human faces and sensitive documents—prior to processing scene data. Additionally, strict access controls, transparent user consent mechanisms, and compliance with data protection regulations are essential to safeguard individual privacy.

## F Limitation and Future Work

While ViPS demonstrates state-of-the-art performance in 3D spatial understanding, several limitations remain. First, due to computational resource constraints, our experiments are currently restricted to a 7B-parameter base MLLM and training datasets on the scale of a few hundred thousand samples. Scaling up both the model size and data volume could further enhance performance. Second, extracting spatial awareness from external visual priors is ultimately a sub-optimal workaround. A more fundamental future direction is to internalize this geometric knowledge directly during the MLLM pre-training phase, which could be achieved by introducing more diverse spatial datasets, finegrained 3D sub-tasks, and tailored loss functions. Finally, our evaluations are currently confined to standard offline benchmarks. Deploying and validating spatial-aware MLLMs in real-world robotics and physical embodied AI scenarios remains an exciting avenue for future exploration.