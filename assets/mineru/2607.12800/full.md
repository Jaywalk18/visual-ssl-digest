# UniVR: Thinking in Visual Space for Unified Visual Reasoning

Zhongwei Ren<sup>1,2,∗</sup>, Yunchao Wei<sup>1</sup>, Yao Zhao<sup>1</sup>, Weibo Gong<sup>2</sup>, Xiao Liu<sup>2</sup>, Anran Wang<sup>2</sup>, Xiangtai Li<sup>2,∗,†</sup>, Xiaojie Jin<sup>1,∗</sup>

<sup>1</sup>Beijing Jiaotong University, <sup>2</sup>ByteDance

<sup>∗</sup>Equal Contribution, <sup>†</sup>Project Lead

## Abstract

Learning broad world knowledge directly from raw visual data is a fundamental capability of intelligence. We introduce UniVR, the first investigation into simultaneously learning complex reasoning, fine-grained physical dynamics, and long-term planning from pure visual demonstrations. At its core, UniVR features VR-GRPO, a reinforcement learning paradigm with complementary global and step-level rewards. This approach enforces logical coherence and physical consistency throughout the reasoning process without requiring task-specific heuristics or image-text pairs. To train and evaluate UniVR, we construct VR-X, a large-scale benchmark curated from 16 diverse sources spanning long-horizon manipulation, spatial puzzles, and physical reasoning. It is the first comprehensive suite to assess these heterogeneous capabilities under a purely visual protocol. Remarkably, UniVR achieves up to a 25% improvement on VR-X, and its superior visual reasoning also boosts performance on various multimodal understanding benchmarks. These findings underscore the vast potential of reasoning within visual spaces, with all code, data, and models are open-sourced for further research.

Date: July 15, 2026 Correspondence: Xiaojiao Jin, Yunchao Wei Project Page: https://UniVR.github.io/

## 1 Introduction

Current AI models primarily derive world knowledge from text [2, 4, 6, 39, 64, 73, 74], performing reasoning [27, 29, 35, 57] and planning [32, 66, 68, 84] within the textual space. However, text is an abstract representation of the world, which is unable to fully encompass the rich information of the real visual world, such as complex dynamics, spatial relationships, and underlying physical laws. In contrast, vision serves as the most direct medium for world knowledge and remains the primary source for animals and humans to acquire information. In most scenarios, humans can perform complex reasoning without relying on language by directly simulating task execution and scene transitions in their minds, which constitutes our innate visual reasoning capability. Given the vast abundance of video content available on the internet, equipping AI with the capacity for complex reasoning and planning within the visual space holds significant promise for enhancing its world modeling abilities and pushing the frontier of eficient task execution in the real visual world.

Recent research has explored two primary pathways for advancing visual reasoning. On one hand, MLLMs, such as Gemini-3 [23] and GPT-5 [67], can generate detailed textual reasoning chains that are subsequently converted into visual reasoning traces via generative models like Nano Banana [14] and GPT-Image [50]. While this paradigm conveniently leverages pre-existing linguistic strengths, textual abstractions inherently struggle to capture the intricate dynamics and spatial relationships of the physical world. Such limitations prevent a seamless transfer of textual reasoning proficiency into the visual domain. As shown in Fig 1, even with state-of-the-art reasoning and high-fidelity rendering, models frequently fail to maintain logical coherence and physical consistency in tasks requiring fine-grained, long-horizon visual evolution. On the other hand, some works [49, 58–60, 70] leverage video generation models to model task strategies directly in visual space, while others employ unified generation models [15, 17, 81, 87, 90] for more comprehensive reasoning through joint vision-language representations. Despite their promise in acquiring complex knowledge from visual data, their reasoning still remains heavily dependent on text guidance, requiring dense image-text pairs for both training and inference. These limitations constrain the scaling of visual reasoning and drive our core inquiry:

![](images/ad49b0aa61000c9090395b1d8661103a51386da2ef5586f4a6989d942d6a638c.jpg)

![](images/e5086b36d96ced8bcd9f268f0fd9f9bb9277887ac71f26fd46e4f1230fbab694.jpg)

![](images/2218313fe9dbfc16471c0776ba96c7746cbd4da9a2497ee3d094daaffc727f4d.jpg)

![](images/cd922580e93072465278081888f4eaa3c7eb82082ddde758693c0702aa1cade2.jpg)  
Figure 1 UniVR focuses on advancing native visual-space reasoning and planning across diverse tasks. Compared to LMMs that reason within textual space, UniVR facilitates a more profound comprehension of real visual world, improving policy learning in various scenarios.

## How can models utilize raw visual data to boost their visual reasoning capabilities across diverse tasks?

To investigate this problem, we first establish VR-X, a comprehensive benchmark designed to assess visua reasoning capabilities across diverse tasks. As shown in Fig 4, VR-X encompasses two primary task categories. The first focuses on long-horizon complex planning across various environments, featuring minute-scale tasks with fine-grained dynamics (e.g. tying knots, folding clothes) and diverse domains spanning cooking, crafts, robotic control, and navigation. The second focuses on general visual reasoning, evaluating fundamenta cognitive skills such as visual search, puzzles, spatial perception, and editing. Models are tasked to directly reason and generate in visual space, with assessment centered on the logical coherence and task completion of the visual reasoning trace. Such task diversity and evaluation paradigm is rarely explored in existing benchmarks, demanding robust learning capabilities to master heterogeneous task knowledge.

Based on this benchmark, we evaluate state-of-the-art MLLMs, including Gemini [14, 23], GPT [67], and Qwen [3, 54], by prompting them to generate detailed textual reasoning chains to guide generative models, alongside visual generation models such as Emu3.5 [15]. Results indicate that while they excel at textual reasoning, visual comprehension, or high-fidelity generation, they still struggle to accurately execute tasks in the benchmark. As shown in Fig. 1 and Fig. 7, their predictions contain errors such as logical gaps, physical inconsistency, and violations of rules, ultimately failing to produce visually coherent task sequences. This underscores an urgent need for frameworks that efectively master those heterogeneous knowledge.

Motivated by these observations, we propose UniVR, a unified next-token prediction framework with strong visual reasoning capabilities. At its core is VR-GRPO, a novel RL paradigm that learns diverse knowledge directly in visual space. A VLM evaluator first provides holistic assessment of generated sequences for task completion and visual quality. However, as detailed in Sec. 3.2, this vanilla reward fails to capture errors like logical gaps and physical inconsistency. We attribute this to current VLMs’ limited visual world knowledge to identify fine-grained errors in minute-level multi-step sequences, leaving final states and appearance quality to dominate judgments. We therefore propose a Step-Focal reward that can proactively targets error-prone substeps for more precise assessment. Combined with global evaluation, this design ensures both overall task completion and fine-grained reasoning coherence, without relying on dense image-text pairs or task-specific rules.

In addition to these qualitative analyses, we benchmark the performance of UniVR against existing methods on VR-X. As shown in Fig. 1, UniVR significantly boosts visual reasoning capabilities without compromising the foundational strengths of the base model (Emu3.5). Notably, with only 34B parameters, UniVR approaches the performance of the Gemini 3 Pro [23] + Nano Banana 2 [21] pipeline and even surpasses Gemini 3 in long-horizon manipulation tasks. This demonstrates the superior eficiency and efectiveness of our visual reasoning framework. Further visualization results are provided in Fig. 8 to demonstrate its robust performance across diverse scenarios.

Our contributions are summarized as follows:

• We are the first to explore learning heterogeneous tasks, from long-term planning to general cognitive reasoning, directly in a unified visual space without language supervision.

• We propose VR-GRPO, featuring a novel Step-Focal reward that proactively targets error-prone reasoning substeps alongside a global reward. This significantly improves logical coherence and physical consistency in visual reasoning without relying on image-text pairs or task-specific rules.

• We introduce VR-X, a benchmark for evaluating diverse tasks in visual space, spanning from fine-grained long-term planning to general reasoning, to facilitate future research on visual reasoning.

## 2 Related Works

Visual Reasoning. The rapid advancement of LLMs [2, 6, 13, 33, 79, 93, 94] and MLLMs [3, 4, 40, 72, 88] has spurred extensive research on text-centric reasoning paradigms, such as chain-of-thought [48, 85, 102] and latent-space reasoning [12, 28]. Recent eforts like visual CoT [37, 53, 80] extend this paradigm to multimodal inputs, yet they still project visual features into a linguistic space, leaving reasoning fundamentally text-bounded . Another line of work explores video generation for non-linguistic world modeling in autonomous driving [49, 60] and robotics [26, 70]. However, these approaches are typically confined to short-horizon dynamics or narrow, single-task settings. UniVR departs from both paradigms by directly reasoning in the visual space, unifying and improving heterogeneous tasks, which demand long-horizon planning, complex policy and intricate physics.

Unified Model [17, 31, 38, 52, 87, 90, 96] aims to encode world knowledge in a single model space, combining the textual reasoning of MLLMs with the visual generation capabilities. Pioneering architectures [15, 81] tokenize images, text, and video into a unified discrete space to enable flexible cross-modal generation. However, these models are predominantly trained on entertainment-oriented or artistic editing objectives, and their visual knowledge acquisition remains heavily constrained by dense image-text supervision. Our UniVR breaks this dependency by incorporating a text-free visual reasoning framework into the unified architecture, enabling the model to learn complex reasoning and planning directly from raw visual inputs.

Reinforcement Learning in Generative Model. The success of RL in MLLMs [19, 39, 43, 51, 56, 61, 71, 97, 98] has been extended to visual generation, with pioneering works such as DDPO [5] and ReFL [91] employing PPO [61] or RLHF [51] to align models with human preferences regarding image fidelity. Following the success of DeepSeek-R1 [25], GRPO [41, 64] has been widely adopted across various vision tasks to enhance multimodal understanding and image/video generation. Various approaches [83, 92, 100] also incorporate specialized reward, such as boosting aesthetic appeal via HPSv3 [47, 89] or enforcing semantic alignment through CLIP [55] and VLM-based scoring. However, these reward mechanisms primarily target perceptual quality, text-image consistency, or single-step correctness. In contrast, our VR-GRPO is specifically designed to optimize for logical consistency and physical plausibility in the context of multi-step visual reasoning and policy learning.

![](images/76122a6906536a9dcfc0ef97ab014b659bdbedf530c11b30a6a312fa9b35e135.jpg)  
Figure 2 Overview of UniVR architecture. Via a unified next-token prediction objective, UniVR processes instructions and image queries to directly generate visual reasoning traces for task execution.

## 3 UniVR

In this section, we introduce UniVR, shown in Fig. 2, which adopts an autoregressive generation model as its basic framework. It has a two-stage training pipeline: cold initialization and reinforcement learning. We describe how to conduct visual space reasoning using this framework, together with cold initialization in Sec. 3.1, and RL training in Sec. 3.2.

## 3.1 Thinking in Visual Space

Given an image sequence $x _ { 1 : t }$ and an instruction, we model the next-frame distribution: $p ( x _ { t + 1 } \mid x _ { 1 : t } )$ . We use image sequences containing demonstration trajectories of task execution across diverse scenarios, encompassing various kinds of planning and reasoning knowledge. Our formulation does not require dense textual reasoning chains, but instead directly models the state transitions and underlying policy dynamics in these trajectories, encouraging the model to reason in visual space.

Specifically, we adopt Emu3.5 as our baseline, a state-of-the-art unified generative model that produces variable-length image sequences. It employs a VQ-VAE-style autoencoder [77] to encode images and text into a unified discrete vocabulary, also enabling text generation. This allows us to further investigate how enhanced visual reasoning afects multimodal understanding (See Sec. 5.2). Despite pretraining on extensive image-text pairs, this baseline still struggles to capture complex real-world physics, fine-grained action dynamics, and visual cognitive abilities, as shown in Sec. 5.1 and Fig. 7. We attribute this to its heavy reliance on dense textual reasoning chains for world knowledge acquisition, reducing vision to a mere renderer. We therefore first perform supervised fine-tuning on a curated dataset of diverse visual tasks as cold initialization, standardizing all samples as query image, instruction, visual reasoning trajectory. This endows the model with visual reasoning priors for subsequent RL. Training configuration and data construction are detailed in Appendix A.

![](images/e7b53ea6ddb38b256d094de79f274a306b504a8965c2136dde12caba44a5bc3f.jpg)  
Figure 3 The proposed Visual Reasoning GRPO. (up) VR-GRPO integrates global and step-focal rewards to ensure both task completion and the physical coherence of generated reasoning traces. (down) Failure case of global rewards that overlook local inconsistencies within long reasoning traces.

## 3.2 Visual Reasoning GRPO

Despite the versatility of casting various tasks into a unified visual space, we observe that vanilla SFT still struggles to reconcile heterogeneous multi-source tasks. For instance, 2D puzzles and long-term human planning edifer drastically in temporal scale, domain knowledge, and visual appearance. As detailed in Sec. 5.2, dense supervision in SFT training fails to yield consistent improvements across diverse tasks. While RL method, e.g. GRPO [64], ofer a promising paradigm for performance enhancement by enabling models to autonomously explore optimal policies across diverse scenarios, existing methods prioritize visual fidelity or cross-modal alignment rather than visual reasoning.

To address this, we introduce VR-GRPO, an RL methodology that eschews the requirement for dense image-text pairs or task-specific heuristics, focusing on logical coherence and task completion. In Sec. 5.2, we further demonstrate that VR-GRPO can also seamlessly integrate with textual reasoning tasks, facilitating a synergistic advancement in overall multimodal understanding.

Reward design. The VR-GRPO has two reward components: format reward $R _ { \mathrm { f o r m a t } }$ and visual reasoning reward $R _ { \mathrm { r e a s o n } } . \ R _ { \mathrm { r e a s o n } }$ consists of a global reward $R _ { \mathrm { g } }$ and a novel step-focal reward $R _ { \mathrm { s } } . \ R _ { \mathrm { f o r m a t } }$ ensures that the generated image sequences satisfy structural constraints, such as uniform resolution and the number of

reasoning steps prescribed by the task instructions.

For $R _ { \mathrm { r e a s o n } }$ , we use a general-purpose prompt to guide an online reward model (i.e. Qwen3-VL-30B) in providing an overall quality assessment of rollout samples against ground-truth references. While the VLM efectively assesses task completion and visual fidelity, as shown in Fig. 6, our preliminary analysis indicates that this global reward often overlooks intermediate physical violations and logical gaps, over-prioritizing terminal success and pixel clarity, especially in long-horizon, multi-step tasks. We attribute this to current VLMs’ reliance on text-derived world knowledge, which constrains their ability to pinpoint fine-grained visual dynamic errors in minute-level multi-step sequences.

Consequently, we introduce a step-focal reasoning reward $R _ { s }$ designed to identify the most error-prone steps in the reasoning trajectory. By focusing the VLM on these critical steps, we deliver precise, fine-grained rewards to ensure the model maintains logical and physical coherence throughout complex, long-horizon tasks. Specifically, for a set of K rollout trajectories $y _ { 1 : K }$ , we first assume a uniform length T . We employ a CLIP image encoder to extract per-frame feature embeddings as $z _ { k } ( t ) \in \mathbb { R } ^ { d }$ and calculate the inter-trajectory variance at each timestep t as:

$$
\sigma (t) = \sqrt {\frac {1}{K} \sum_ {k = 1} ^ {K} \| \mathbf {z} _ {k} (t) - \bar {\mathbf {z}} (t) \| _ {2} ^ {2}}\tag{1}
$$

where $\bar { \bf { z } } ( t )$ is the mean embedding across all trajectories at time t. A high variance σ(t) indicates a state of maximum uncertainty where the model’s reasoning paths diverge. Given a window size W , we identify the peak uncertainty at $t ^ { * } = \mathrm { a r g m a x } _ { t } \sigma ( t )$ and extract a reasoning segment from $[ t ^ { * } - W / 2 , t ^ { * } + W / 2 ]$ . For samples that are excessively challenging, this process reverts to random sampling. In practice, generated trajectories often vary in length. To ensure temporal alignment, we partition each trajectory into an equal number of segments. We then treat the segment index as the synchronized timestep and use the average CLIP features within each segment to perform the uncertainty analysis described above. Following [82], we employ a pairwise evaluation protocol for both global rewards and the identified sub-steps to derive relative win rates. This approach is designed to further mitigate the inherent scoring bias that often accompanies direct VLM-based scalar assessments. Upon obtaining the global score $R _ { \mathrm { g } }$ and step-focal scores $R _ { \mathrm { s } }$ for each trajectory, we integrate them using the following formula:

$$
R _ {\mathrm{reason}} = R _ {\mathrm{g}} - \lambda | R _ {\mathrm{g}} - R _ {\mathrm{s}} |\tag{2}
$$

This formulation prevents the model from taking reasoning shortcuts, as it necessitates that the model not only predicts the terminal state accurately but also maintains procedural integrity and physical coherence. The coeficient λ controls the strength of this alignment.

## 4 VR-X Benchmark

Our objective is to advance the capacity for reasoning natively within the visual domain across a wide array of scenarios. We prioritize a comprehensive benchmark capable of evaluating multi-step planning, delicate manipulations and cognitive reasoning. However, current benchmarks mostly emphasize visual quality or text-matching, limited to restricted environments and short-term tasks.

Therefore, we introduce VR-X, the first large-scale benchmark designed for diverse and heterogeneous visual reasoning. As shown in Fig. 4, it includes six tasks, each providing detailed visual reasoning traces: visual guidance, robotic manipulation, puzzle games, editing, search, and spatial perception.

## 4.1 Dataset Generation

VR-X is characterized by its vast diversity and involves fine-grained, visually complex manipulations that are hard to describe linguistically. We curate 1.5M raw samples from 16 diverse sources (e.g., AgiBot [7], Action100M [10], EgoDex [30], VisualCoT [63]), spanning minute-long planning (robotic manipulation, cooking, handcrafting) to single-step reasoning (mazes, visual search). Rigorously curated into 310k cold-start training, 3k RL, and 1.8k benchmark evaluation samples, all follow a unified format: query image, textual instruction, visual reasoning trajectory. We further annotate these sequences with fine-grained textual chain-of-thought (CoT) descriptions to support multimodal learning. See Appendix A for more details.

![](images/b917bec7f6b8b3c33f5d3b236af9cc6a138427ed92520f0be1dd7428a69f9acc.jpg)  
Figure 4 Overview of VR-X benchmark.

## 4.2 Evaluation Metrics

Our evaluation focuses on two main aspects: the logical accuracy of visual reasoning and the adherence to real physical dynamics. Accordingly, we employ the following metrics for evaluation:

VLM score: We conduct an automated evaluation using Qwen3.5-397B [54] to quantitatively assess reasoning traces. The model evaluates each sample based on task completion, procedural coherence, visual informativeness, and image fidelity. These dimensions are aggregated into a final normalized score (0–100), providing a robust metric for measuring the logical and visual quality of the results.

JEPA similarity: Despite strong reasoning capabilities, VLMs’ dependency on textual knowledge often hinders their ability to perceive intrinsic physical laws. As shown in Fig. 6, this may result in hallucinations where the evaluator ignores fine-grained physical inconsistencies. To bridge this gap, we incorporate JEPA similarity as an additional metric. Building on prior research [45] demonstrating that V-JEPA [1] encoders can capture high-level latent physical dynamics, we map sequences into latent space and compute maximum mean discrepancy against ground-truth distributions. Lower scores indicate closer alignment with real-world physics. See Appendix A for more evaluation details.

Table 1 Comparison on VR-X. Qwen, Gemini 2.5/3, and GPT-5 are paired with Qwen-image-edit, Nano Banana 1/2, and GPT-image-1.5, respectively.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Visual Thinking</td><td colspan="2">Long-term planning</td><td colspan="4">General Reasoning</td><td rowspan="2">Overall↑</td><td rowspan="2">JEPA↓</td></tr><tr><td>Guidance</td><td>Robot</td><td>Editing</td><td>Spatial</td><td>Puzzle</td><td>Search</td></tr><tr><td colspan="10">Large Multimodal Model + T2I Model</td></tr><tr><td>Qwen3-VL-235B [3]</td><td>✕</td><td>48.2</td><td>62.8</td><td>42.1</td><td>38.2</td><td>64.1</td><td>59.8</td><td>52.5</td><td>18.08</td></tr><tr><td>Qwen3.5-397B [54]</td><td>✕</td><td>47.0</td><td>63.2</td><td>39.8</td><td>40.4</td><td>65.6</td><td>64.5</td><td>53.4</td><td>18.64</td></tr><tr><td>GPT-5 [67]</td><td>✕</td><td>68.2</td><td>64.1</td><td>58.0</td><td>49.3</td><td>64.0</td><td>77.4</td><td>63.5</td><td>12.17</td></tr><tr><td>Gemini-2.5-pro [14]</td><td>✕</td><td>58.4</td><td>67.9</td><td>54.0</td><td>40.5</td><td>67.7</td><td>76.3</td><td>60.8</td><td>14.39</td></tr><tr><td>Gemini-3-pro [23]</td><td>✕</td><td>66.2</td><td>67.1</td><td>63.7</td><td>55.1</td><td>65.5</td><td>79.0</td><td>66.1</td><td>11.07</td></tr><tr><td colspan="10">Unified Generation Model</td></tr><tr><td>Janus-pro [11]</td><td>✕</td><td>9.2</td><td>18.2</td><td>5.4</td><td>10.2</td><td>27.1</td><td>21.5</td><td>15.3</td><td>68.79</td></tr><tr><td>Show-o2 [90]</td><td>✕</td><td>15.1</td><td>22.5</td><td>13.0</td><td>17.1</td><td>29.4</td><td>35.8</td><td>22.2</td><td>59.93</td></tr><tr><td>ILLUME+ [31]</td><td>✕</td><td>13.1</td><td>11.5</td><td>5.8</td><td>14.6</td><td>22.2</td><td>27.5</td><td>15.8</td><td>61.12</td></tr><tr><td>OneCAT [38]</td><td>✕</td><td>15.6</td><td>13.5</td><td>10.3</td><td>20.2</td><td>16.2</td><td>30.1</td><td>17.7</td><td>77.06</td></tr><tr><td>STAR [52]</td><td>✕</td><td>22.7</td><td>28.5</td><td>14.2</td><td>21.5</td><td>27.2</td><td>37.7</td><td>25.3</td><td>51.98</td></tr><tr><td>OmniGen2 [87]</td><td>✕</td><td>20.4</td><td>29.4</td><td>15.2</td><td>16.9</td><td>30.5</td><td>42.1</td><td>25.8</td><td>47.09</td></tr><tr><td>Bagel [17]</td><td>✕</td><td>25.2</td><td>34.7</td><td>20.9</td><td>21.3</td><td>35.1</td><td>47.7</td><td>30.8</td><td>40.88</td></tr><tr><td>Emu3.5 [15]</td><td>✕</td><td>38.6</td><td>42.8</td><td>32.7</td><td>35.3</td><td>43.4</td><td>46.2</td><td>39.8</td><td>33.62</td></tr><tr><td>UniVR</td><td>√</td><td>59.5</td><td>68.0</td><td>48.5</td><td>46.5</td><td>62.2</td><td>64.3</td><td>58.2</td><td>13.01</td></tr><tr><td>△ v.s. Emu 3.5</td><td></td><td>↑20.9</td><td>↑25.2</td><td>↑15.8</td><td>↑11.2</td><td>↑18.8</td><td>↑18.1</td><td>↑18.4</td><td>↓20.61</td></tr></table>

## 5 Experiment

We leverage Emu3.5 34B to initialize our unified model, which undergoes full-parameter SFT and RL. Our RL pipeline is implemented using the verl framework [65], utilizing a rollout size of 8. For sub-step selection, we set the default window size to 4 frames and the consistency coeficient λ to 2.0. During training, video resolution is scaled to a short-side of 512px, and the maximum sequence length is capped at 20k tokens. See Appendix A for more implementation details.

## 5.1 Results on VR-X

Tab. 1 provides a comprehensive comparison on the VR-X benchmark across two dominant technical paradigms: the integration of large multimodal models with T2I models and unified generation models. This evaluation aims to investigate the capacity of current approaches for visual-space reasoning while validating the eficacy of UniVR.

LMMs with T2I models. We first evaluate of-the-shelf LMMs (rows 1-5), including Qwen 3-VL [3] and Qwen 3.5 [54] with Qwen-Image-Edit [86], Gemini 2.5/3 Pro [14, 23] with Nano Banana 1/2 [21, 22], and GPT-5 [67] with GPT-image 1.5 [50]. These systems follow a two-stage pipeline: the LMM first produces step-level textual instructions from the input image and task prompt. The generation module then renders the sequence frame-by-frame conditioned on previous frames and textual guidance. Gemini 3 Pro with Nano Banana 2 achieves the best performance, benefiting from strong textual reasoning and superior rendering quality. However, as shown in Fig. 7, visual inconsistencies persist in tasks with complex dynamics and planning, even with detailed instructions. Notably, this issue does not improve substantially with the evolution of language models, comparisons between Gemini 3/2.5 Pro and Qwen 3.5/3-VL show marginal gains in visual logicality. This underscores the urgent need for intrinsic visual world knowledge beyond mere linguistic scaling.

Unified generation models. Rows 9-16 evaluate various vision-language unified models. Except for Emu3.5, most lack contiguous image generation and must iteratively unroll for long-horizon planning, generally underperforming with a mere 30% peak success rate. This likely stems from training objectives centered on entertainment generation and artistic editing rather than logical reasoning. Emu3.5 achieves the highest score in this group, benefiting from native pre-training on diverse daily-life and handcrafting sequences. Yet textual reasoning remains indispensable for its execution, and its performance stagnates at 35% when confronted with VR-X’s intricate visual dynamics. Meanwhile, these models also exhibit poor JEPA scores, lagging significantly behind sequences rendered by text-based MLLMs, with feature distribution divergences from ground truth approximately 2–4.5× larger. This underscores that visual space reasoning remains a critical bottleneck for existing models.

<table><tr><td>Method</td><td>MMMU [101]</td><td>MME(P) [20]</td><td>MME(C) [20]</td><td>MMBench [42]</td><td>MathVista [44]</td><td>MM-Vet [99]</td></tr><tr><td>Emu 3.5</td><td> $\underline{0.292}$ </td><td>781.1</td><td> $\underline{324.6}$ </td><td>0.183</td><td> $\underline{41.7}$ </td><td>28.0</td></tr><tr><td>Text-only training</td><td>0.290</td><td> $\underline{782.0}$ </td><td>323.4</td><td> $\underline{0.199}$ </td><td>40.8</td><td> $\underline{28.3}$ </td></tr><tr><td>UniVR*</td><td> $\underline{0.337}$ </td><td> $\underline{799.3}$ </td><td> $\underline{338.5}$ </td><td> $\underline{0.198}$ </td><td> $\underline{44.0}$ </td><td> $\underline{35.6}$ </td></tr><tr><td> $\triangle v.s.$  Emu3.5</td><td>↑ 0.045</td><td>↑ 18.2</td><td>↑ 13.9</td><td>↑ 0.015</td><td>↑ 2.3</td><td>↑ 7.6</td></tr></table>

(a) Results on understanding benchmarks. <sup>∗</sup> means training with with interleaved image-text sequences.

<table><tr><td rowspan="2">Cold Start</td><td colspan="3">Reward</td><td rowspan="2">LP</td><td colspan="2">VR-X</td></tr><tr><td>Global</td><td>Step</td><td>Pairwise</td><td>GR</td><td>JEPA↓</td></tr><tr><td>√</td><td></td><td></td><td></td><td>48.2</td><td>42.4</td><td>18.44</td></tr><tr><td>√</td><td>√</td><td></td><td></td><td>45.7</td><td>46.0</td><td>22.30</td></tr><tr><td>√</td><td></td><td>√</td><td></td><td>54.9</td><td>45.8</td><td>15.87</td></tr><tr><td>√</td><td>√</td><td>√</td><td></td><td>61.6</td><td>53.7</td><td>12.89</td></tr><tr><td>√</td><td>√</td><td>√</td><td>√</td><td>63.8</td><td>55.4</td><td>13.01</td></tr></table>

(b) Reward in VR-GRPO.

<table><tr><td rowspan="2">Training Data</td><td rowspan="2">Cold Start</td><td colspan="3">Reward</td><td rowspan="2">LP</td><td colspan="2">VR-X</td></tr><tr><td>HPSv3</td><td>CLIP</td><td>VR-GRPO</td><td>GR</td><td>JEPA↓</td></tr><tr><td>V</td><td>√</td><td></td><td></td><td></td><td>48.2</td><td> $\underline{42.4}$ </td><td>18.44</td></tr><tr><td>V&amp;T</td><td>√</td><td></td><td></td><td></td><td>49.7</td><td>41.0</td><td>18.37</td></tr><tr><td>V&amp;T</td><td>√</td><td>√</td><td></td><td></td><td>49.0</td><td>41.7</td><td>18.09</td></tr><tr><td>V&amp;T</td><td>√</td><td>√</td><td>√</td><td></td><td> $\underline{52.6}$ </td><td>42.0</td><td> $\underline{17.17}$ </td></tr><tr><td>V&amp;T</td><td>√</td><td>√</td><td>√</td><td>√</td><td>65.4</td><td>57.8</td><td>12.97</td></tr></table>

(c) Training with text-based RL method.

<table><tr><td>Method</td><td>Training Data</td><td>LP</td><td>GR</td></tr><tr><td rowspan="2">Cold-Start</td><td>Separate</td><td>53.6</td><td>44.2</td></tr><tr><td>Join</td><td>48.2</td><td>42.4</td></tr><tr><td rowspan="2">UniVR</td><td>Separate</td><td>61.5</td><td>55.0</td></tr><tr><td>Join</td><td>63.8</td><td>55.4</td></tr></table>

(d) Ablation on joint training.

<table><tr><td>Method</td><td>WorldArena [62]</td><td>Uni-MMMU [103]</td><td>RBench [18]</td></tr><tr><td>Emu3.5</td><td>40.3</td><td>28.7</td><td>31.2</td></tr><tr><td>Cold-Start</td><td>42.4</td><td>37.6</td><td>36.8</td></tr><tr><td>UniVR</td><td>49.5</td><td>54.4</td><td>47.7</td></tr><tr><td> $\triangle v.s.$  Emu3.5</td><td>↑ 9.2</td><td>↑ 25.7</td><td>↑ 16.5</td></tr></table>

(e) Evaluation on other benchmark data.  
Table 2 Ablation studies. “LP” and “GR” denote long-term planning and general reasoning in VR-X, respectively.

UniVR. In contrast, UniVR significantly enhances performance across all tasks. Despite the absence of fine-grained text procedural annotations, directly training in visual space achieves a 60% success rate in long-term planning and 70% in general reasoning. Remarkably, at a 34B parameter scale, UniVR outperforms Gemini 2.5 Pro on several key metrics. This shows that our training pipeline efectively empowers the model to distill essential task-centric policies directly from visual signals, bypassing the need for textual mediation. Furthermore, by fostering stronger visual reasoning, the generated sequences exhibit enhanced physical dynamics, leading to better JEPA metrics.

## 5.2 Ablation Study

Visual reasoning benefit multimodal understanding. Tab. 2a presents an ablation on standard multimodal understanding benchmarks. Following the protocol in [46], row one reports the baseline performance of Emu3.5 on benchmarks such as MMMU [101] and MME [20], where the model is prompted to autonomously generate intermediate visual reasoning images and the final textual answer. In row two, we train solely on textual reasoning chains derived from VR-X (details of texual reasoning annotation provided in Appendix A) to align with the conventional training paradigm of multimodal understanding models, yet this yields negligible gains.In contrast, integrating the UniVR training paradigm (row three) leads to more significant gains across all six metrics. This indicates that enhanced visual reasoning serves as a powerful complement to textua supervision, efectively bolstering overall multimodal comprehension.

Reward in VR-GRPO. Tab. 2b ablates the individual components of VR-GRPO. Row one establishes the baseline performance of the cold-start model before RL. While the global reward alone improves general reasoning (short sequences) by 3.6% , it causes degradation in long-term planning and JEPA scores, indicating the occurrence of reward hacking. Conversely, introducing the step-focal reward alone shows positive improvements across all three metrics. Combining both rewards results in a synergistic efect, further enhancing overall performance. Finally, transitioning the VLM from absolute scalar scoring to a pairwise comparison protoco further enhances the robustness of the rewards and improves the final results.

VR-GRPO is compatible with text-based RL. Tab. 2c investigates the compatibility of VR-GRPO with existing multimodal rewards. In the cold-start phase, adding textual reasoning chains (row 2) yields no significant gains over pure visual reasoning (row 1). Furthermore, incorporating aesthetic-centric rewards like HPSv3 or vision-language alignment via CLIP (rows 3-4) provides limited assistance for tasks requiring complex visual reasoning, with HPSv3 showing negligible impact. In contrast, VR-GRPO (row 5) significantly boosts visua reasoning capabilities and remains compatible with other reward functions, highlighting the generalizability and versatility of our approach.

VR-GRPO stabilizes training with heterogeneous knowledge. Long-term planning and general reasoning in VR-X exhibit significant disparities in terms of temporal scale, environmental context, visual appearance, and knowledge domains. Rows 1-3 in Tab. 2d investigate the joint learning capabilities across these tasks without the incorporation of VR-GRPO. The results indicate that during the pure cold-start phase, joint training struggles to efectively balance heterogeneous task knowledge, failing to surpass the performance of models trained individually on each task. In contrast, the introduction of VR-GRPO in row 4 empowers the model to autonomously explore optimal policies, yielding performance gains in both categories. This demonstrates the eficacy of our approach in facilitating unified visual reasoning training.

Evaluate on other visual reasoning benchmarks. To further assess UniVR’s generalization, we evaluate it on test data from three external benchmarks (Tab. 2e): two focusing on embodied intelligence scenarios [18, 62] and one on cognitive reasoning [103]. Using the instructions and query images from these benchmarks, we prompt UniVR to generate corresponding visual reasoning trajectories, then evaluate them with the same VLM scoring protocol as in Sec. 5. Despite significant domain gaps from our training data, encompassing unseen robotic manipulation tasks, environments, camera viewpoints, and mathematical/physical reasoning questions—UniVR substantially improves baseline performance (up to 24.3%) on these new tests. Moreover, with VR-GRPO, this gain significantly exceeds that of vanilla SFT, demonstrating that VR-GRPO fosters more generalizable visual reasoning.

## 6 Conclusion

In this work, we presented UniVR, a unified framework that enables complex reasoning and planning directly within visual space, free from dense language supervision. Central to our approach is VR-GRPO, which enforces both global task completion and fine-grained step-level physical coherence through complementary rewards. Trained and evaluated on the diverse VR-X benchmark, UniVR achieves substantial gains over strong text-based reasoning pipelines and unified generation models, demonstrating that raw visual demonstrations alone can support sophisticated, long-horizon policy learning. Beyond task execution, our findings reveal that strengthened visual reasoning confers broader benefits to multimodal understanding, suggesting vision-native computation as a powerful complement to textual abstraction. We believe this work opens promising avenues toward more grounded, eficient world models that learn directly from the abundant visual world.

## References

[1] Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.

[2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

[3] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

[5] Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training difusion models with reinforcement learning. arXiv preprint arXiv:2305.13301, 2023.

[6] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[7] Qingwen Bu, Jisong Cai, Li Chen, Xiuqi Cui, Yan Ding, Siyuan Feng, Shenyuan Gao, Xindong He, Xuan Hu, Xu Huang, et al. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

[8] Joao Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the kinetics dataset. In proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6299–6308, 2017.

[9] Brandon Castellano and PySceneDetect Contributors. Pyscenedetect: Video scene detection and analysis tool. GitHub repository, 2025. URL https://github.com/Breakthrough/PySceneDetect. Accessed: 2026-05-06.

[10] Delong Chen, Tejaswi Kasarla, Yejin Bang, Mustafa Shukor, Willy Chung, Jade Yu, Allen Bolourchi, Theo Moutakanni, and Pascale Fung. Action100m: A large-scale video action dataset. arXiv preprint arXiv:2601.10592, 2026.

[11] Xiaokang Chen, Zhiyu Wu, Xingchao Liu, Zizheng Pan, Wen Liu, Zhenda Xie, Xingkai Yu, and Chong Ruan. Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811, 2025.

[12] Xinghao Chen, Anhao Zhao, Heming Xia, Xuan Lu, Hanlin Wang, Yanjun Chen, Wei Zhang, Jian Wang, Wenjie Li, and Xiaoyu Shen. Reasoning beyond language: A comprehensive survey on latent chain-of-thought reasoning. arXiv preprint arXiv:2505.16782, 2025.

[13] Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. Palm: Scaling language modeling with pathways. Journal of Machine Learning Research, 24(240):1–113, 2023.

[14] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[15] Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.

[16] Dima Damen, Hazel Doughty, Giovanni Maria Farinella, Sanja Fidler, Antonino Furnari, Evangelos Kazakos, Davide Moltisanti, Jonathan Munro, Toby Perrett, Will Price, et al. The epic-kitchens dataset: Collection, challenges and baselines. IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(11):4125–4141, 2020.

[17] Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.

[18] Yufan Deng, Zilin Pan, Hongyu Zhang, Xiaojie Li, Ruoqing Hu, Yufei Ding, Yiming Zou, Yan Zeng, and Daquan Zhou. Rethinking video generation model for the embodied world. arXiv preprint arXiv:2601.15282, 2026.

[19] Kaituo Feng, Kaixiong Gong, Bohao Li, Zonghao Guo, Yibing Wang, Tianshuo Peng, Junfei Wu, Xiaoying Zhang, Benyou Wang, and Xiangyu Yue. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025.

[20] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394, 2023.

[21] Google. Introducing Gemini 2.5 Flash Image, our state-of-the-art image model. Google Developers Blog, 2025. URL https://developers.googleblog.com/en/introducing-gemini-2-5-flash-image/. Accessed: 2025-05- 06.

[22] Google. Nano banana 2: Google’s latest AI image generation model. Google Blog, feb 2026. URL https: //blog.google/innovation-and-ai/technology/ai/nano-banana-2/. Accessed: 2026-05-06.

[23] Google DeepMind. A new era of intelligence with Gemini 3, November 2025. URL https://blog.google/ products-and-platforms/products/gemini/gemini-3/. Accessed: 2026-05-05.

[24] Jiawei Gu, Yunzhuo Hao, Huichen Will Wang, Linjie Li, Michael Qizhe Shieh, Yejin Choi, Ranjay Krishna, and Yu Cheng. Thinkmorph: Emergent properties in multimodal interleaved chain-of-thought reasoning. arXiv preprint arXiv:2510.27492, 2025.

[25] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

[26] Yanjiang Guo, Lucy Xiaoyang Shi, Jianyu Chen, and Chelsea Finn. Ctrl-world: A controllable generative world model for robot manipulation. arXiv preprint arXiv:2510.10125, 2025.

[27] Kunyang Han, Yibo Hu, Mengxue Qu, Hailin Shi, Yao Zhao, and Yunchao Wei. Rose: Revolutionizing open-set dense segmentation with patch-wise perceptual large multimodal model, 2024. URL https://arxiv.org/abs/ 2412.00153.

[28] Shibo Hao, Sainbayar Sukhbaatar, DiJia Su, Xian Li, Zhiting Hu, Jason Weston, and Yuandong Tian. Training large language models to reason in a continuous latent space. arXiv preprint arXiv:2412.06769, 2024.

[29] Alex Havrilla, Yuqing Du, Sharath Chandra Raparthy, Christoforos Nalmpantis, Jane Dwivedi-Yu, Maksym Zhuravinskyi, Eric Hambro, Sainbayar Sukhbaatar, and Roberta Raileanu. Teaching large language models to reason with reinforcement learning. arXiv preprint arXiv:2403.04642, 2024.

[30] Ryan Hoque, Peide Huang, David J Yoon, Mouli Sivapurapu, and Jian Zhang. Egodex: Learning dexterous manipulation from large-scale egocentric video. arXiv preprint arXiv:2505.11709, 2025.

[31] Runhui Huang, Chunwei Wang, Junwei Yang, Guansong Lu, Yunlong Yuan, Jianhua Han, Lu Hou, Wei Zhang, Lanqing Hong, Hengshuang Zhao, et al. Illume+: Illuminating unified mllm with dual visual tokenization and difusion refinement. arXiv preprint arXiv:2504.01934, 2025

[32] Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In International conference on machine learning, pages 9118–9147. PMLR, 2022.

[33] Albert Q Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, et al. Mistral 7b. arXiv preprint arXiv:2310.06825, 2023.

[34] Alexander Khazatsky, Karl Pertsch, Suraj Nair, Ashwin Balakrishna, Sudeep Dasari, Siddharth Karamcheti, Soroush Nasiriany, Mohan Kumar Srirama, Lawrence Yunliang Chen, Kirsty Ellis, et al. Droid: A large-scale in-the-wild robot manipulation dataset. arXiv preprint arXiv:2403.12945, 2024.

[35] Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa. Large language models are zero-shot reasoners. Advances in neural information processing systems, 35:22199–22213, 2022.

[36] Ang Li, Charles Wang, Deqing Fu, Kaiyu Yue, Zikui Cai, Wang Bill Zhu, Ollie Liu, Peng Guo, Willie Neiswanger, Furong Huang, et al. Zebra-cot: A dataset for interleaved vision language reasoning. arXiv preprint arXiv:2507.16746, 2025.

[37] Bangzheng Li, Ximeng Sun, Jiang Liu, Ze Wang, Jialian Wu, Xiaodong Yu, Hao Chen, Emad Barsoum, Muhao Chen, and Zicheng Liu. Latent visual reasoning. arXiv preprint arXiv:2509.24251, 2025.

[38] Han Li, Xinyu Peng, Yaoming Wang, Zelin Peng, Xin Chen, Rongxiang Weng, Jingang Wang, Xunliang Cai, Wenrui Dai, and Hongkai Xiong. Onecat: Decoder-only auto-regressive model for unified understanding and generation. arXiv preprint arXiv:2509.03498, 2025.

[39] Baijiong Lin, Weisen Jiang, Yuancheng Xu, Hao Chen, and Ying-Cong Chen. Parm: Multi-objective test-time alignment via preference-aware autoregressive reward model. arXiv preprint arXiv:2505.06274, 2025.

[40] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.

[41] Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-grpo: Training flow matching models via online rl. arXiv preprint arXiv:2505.05470, 2025.

[42] Yuan Liu, Haodong Duan, Yuanhan Zhang, Bo Li, Songyang Zhang, Wangbo Zhao, Yike Yuan, Jiaqi Wang, Conghui He, Ziwei Liu, et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pages 216–233. Springer, 2024.

[43] Yuqi Liu, Bohao Peng, Zhisheng Zhong, Zihao Yue, Fanbin Lu, Bei Yu, and Jiaya Jia. Seg-zero: Reasoning-chain guided segmentation via cognitive reinforcement. arXiv preprint arXiv:2503.06520, 2025.

[44] Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.

[45] Ge Ya Luo, Gian Mario Favero, Zhi Hao Luo, Alexia Jolicoeur-Martineau, and Christopher Pal. Beyond fvd: Enhanced evaluation metrics for video generation quality. arXiv preprint arXiv:2410.05203, 2024.

[46] Yinyi Luo, Wenwen Wang, Hayes Bai, Hongyu Zhu, Hao Chen, Pan He, Marios Savvides, Sharon Li, and Jindong Wang. Torchumm: A unified multimodal model codebase for evaluation, analysis, and post-training. arXiv preprint arXiv:2604.10784, 2026.

[47] Yuhang Ma, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. Hpsv3: Towards wide-spectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15086–15095, 2025.

[48] Aman Madaan and Amir Yazdanbakhsh. Text and patterns: For efective chain of thought, it takes two to tango. arXiv preprint arXiv:2209.07686, 2022.

[49] Grégoire Mialon, Clémentine Fourrier, Craig Swift, Thomas Wolf, Yann LeCun, and Thomas Scialom. Gaia: a benchmark for general ai assistants. arXiv preprint arXiv:2311.12983, 2023.

[50] OpenAI. Introducing our latest image generation model in the API. OpenAI Blog, apr 2025. URL https: //openai.com/index/image-generation-api/. Accessed: 2025-05-06.

[51] Long Ouyang, Jefrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.

[52] Jie Qin, Jiancheng Huang, Limeng Qiao, and Lin Ma. Star: Stacked autoregressive scheme for unified multimodal learning. arXiv preprint arXiv:2512.13752, 2025.

[53] Yiming Qin, Bomin Wei, Jiaxin Ge, Konstantinos Kallidromitis, Stephanie Fu, Trevor Darrell, and XuDong Wang. Chain-of-visual-thought: Teaching vlms to see and think better with continuous visual tokens. arXiv preprint arXiv:2511.19418, 2025

[54] Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026. URL https://qwen.ai/blog?id= qwen3.5.

[55] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

[56] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. Direct preference optimization: Your language model is secretly a reward model. Advances in neural information processing systems, 36:53728–53741, 2023.

[57] Zhongwei Ren, Zhicheng Huang, Yunchao Wei, Yao Zhao, Dongmei Fu, Jiashi Feng, and Xiaojie Jin. Pixellm: Pixel reasoning with large multimodal model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 26374–26383, 2024.

[58] Zhongwei Ren, Yunchao Wei, Xun Guo, Yao Zhao, Bingyi Kang, Jiashi Feng, and Xiaojie Jin. Videoworld: Exploring knowledge learning from unlabeled videos. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29029–29039, 2025.

[59] Zhongwei Ren, Yunchao Wei, Xiao Yu, Guixun Luo, Yao Zhao, Bingyi Kang, Jiashi Feng, and Xiaojie Jin. Videoworld 2: Learning transferable knowledge from real-world videos. arXiv preprint arXiv:2602.10102, 2026.

[60] Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. Gaia-2: A controllable multi-view generative world model for autonomous driving. arXiv preprint arXiv:2503.20523, 2025.

[61] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

[62] Yu Shang, Zhuohang Li, Yiding Ma, Weikang Su, Xin Jin, Ziyou Wang, Lei Jin, Xin Zhang, Yinzhou Tang, Haisheng Su, et al. Worldarena: A unified benchmark for evaluating perception and functional utility of embodied world models. arXiv preprint arXiv:2602.08971, 2026.

[63] Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. Visual cot: Advancing multi-modal language models with a comprehensive dataset and benchmark for chain-of-thought reasoning. Advances in Neural Information Processing Systems, 37:8612–8642, 2024.

[64] Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

[65] Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. Hybridflow: A flexible and eficient rlhf framework. In Proceedings of the Twentieth European Conference on Computer Systems, pages 1279–1297, 2025.

[66] Wentao Shi, Xiangnan He, Yang Zhang, Chongming Gao, Xinyue Li, Jizhi Zhang, Qifan Wang, and Fuli Feng. Enhancing long-term recommendation with bi-level learnable large language model planning. arXiv preprint arXiv:2403.00843, 2024.

[67] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

[68] Chan Hee Song, Jiaman Wu, Clayton Washington, Brian M Sadler, Wei-Lun Chao, and Yu Su. Llm-planner: Few-shot grounded planning for embodied agents with large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2998–3009, 2023.

[69] C. Spearman. The proof and measurement of association between two things. The American Journal of Psychology, 15(1):72–101, 1904. doi: 10.2307/1412159.

[70] Boming Tan, Xiangdong Zhang, Ning Liao, Yuqing Zhang, Shaofeng Zhang, Xue Yang, Qi Fan, and Yanyong Zhang. Dreamworld: Unified world modeling in video generation. arXiv preprint arXiv:2603.00466, 2026.

[71] Huajie Tan, Yuheng Ji, Xiaoshuai Hao, Xiansheng Chen, Pengwei Wang, Zhongyuan Wang, and Shanghang Zhang. Reason-rft: Reinforcement fine-tuning for visual reasoning of vision language models. arXiv preprint arXiv:2503.20752, 2025.

[72] Qwen Team. Qwen3. 5-omni technical report. arXiv preprint arXiv:2604.15804, 2026.

[73] Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. Llama: Open and eficient foundation language models, 2023. URL https: //arxiv.org/abs/2302.13971.

[74] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

[75] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786 2025.

[76] Thomas Unterthiner, Sjoerd Van Steenkiste, Karol Kurach, Raphael Marinier, Marcin Michalski, and Sylvain Gelly. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

[77] Aaron van den Oord, Oriol Vinyals, and Koray Kavukcuoglu. Neural discrete representation learning. In NeurIPS, 2017.

[78] Homer Rich Walke, Kevin Black, Tony Z Zhao, Quan Vuong, Chongyi Zheng, Philippe Hansen-Estruch, Andre Wang He, Vivek Myers, Moo Jin Kim, Max Du, et al. Bridgedata v2: A dataset for robot learning at scale. In Conference on Robot Learning, pages 1723–1736. PMLR, 2023.

[79] Jiacong Wang, Bohong Wu, Haiyong Jiang, Zhou Xun, Xin Xiao, Haoyuan Guo, and Jun Xiao. World to code: Multi-modal data generation via self-instructed compositional captioning and filtering. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 4608–4623, 2024.

[80] Qixun Wang, Yang Shi, Yifei Wang, Yuanxing Zhang, Pengfei Wan, Kun Gai, Xianghua Ying, and Yisen Wang. Monet: Reasoning in latent visual space beyond images and language. arXiv preprint arXiv:2511.21395, 2025.

[81] Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869, 2024.

[82] Yibin Wang, Zhimin Li, Yuhang Zang, Yujie Zhou, Jiazi Bu, Chunyu Wang, Qinglin Lu, Cheng Jin, and Jiaqi Wang. Pref-grpo: Pairwise preference reward-based grpo for stable text-to-image reinforcement learning. arXiv preprint arXiv:2508.20751, 2025.

[83] Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025

[84] Zihao Wang, Shaofei Cai, Guanzhou Chen, Anji Liu, Xiaojian Ma, Yitao Liang, and Team CraftJarvis. Describe, explain, plan and select: interactive planning with large language models enables open-world multi-task agents. In Proceedings of the 37th International Conference on Neural Information Processing Systems, pages 34153–34189, 2023.

[85] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.

[86] Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, Yuxiang Chen, Zecheng Tang, Zekai Zhang, Zhengyi Wang, An Yang, Bowen Yu, Chen Cheng, Dayiheng Liu, Deqing Li, Hang Zhang, Hao Meng, Hu Wei, Jingyuan Ni, Kai Chen, Kuan Cao, Liang Peng, Lin Qu, Minggang Wu, Peng Wang, Shuting Yu, Tingkun Wen, Wensen Feng, Xiaoxiao Xu, Yi Wang, Yichang Zhang, Yongqiang Zhu, Yujia Wu, Yuxuan Cai, and Zenan Liu. Qwen-image technical report, 2025. URL https://arxiv.org/abs/2508.02324.

[87] Chenyuan Wu, Pengfei Zheng, Ruiran Yan, Shitao Xiao, Xin Luo, Yueze Wang, Wanli Li, Xiyan Jiang, Yexin Liu, Junjie Zhou, et al. Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871, 2025.

[88] Jialong Wu, Shaofeng Yin, Ningya Feng, Xu He, Dong Li, Jianye Hao, and Mingsheng Long. ivideogpt: Interactive videogpts are scalable world models. NeurIPS, 37:68082–68119, 2024.

[89] Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341, 2023.

[90] Jinheng Xie, Zhenheng Yang, and Mike Zheng Shou. Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564, 2025.

[91] Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.

[92] Zeyue Xue, Jie Wu, Yu Gao, Fangyuan Kong, Lingting Zhu, Mengzhao Chen, Zhiheng Liu, Wei Liu, Qiushan Guo, Weilin Huang, et al. Dancegrpo: Unleashing grpo on visual generation. arXiv preprint arXiv:2505.07818, 2025.

[93] An Yang, Baosong Yang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Zhou, Chengpeng Li, Chengyuan Li, Dayiheng Liu, Fei Huang, et al. Qwen2 technical report. arXiv preprint arXiv:2407.10671, 2024.

[94] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

[95] Cheng Yang, Haiyuan Wan, Yiran Peng, Xin Cheng, Zhaoyang Yu, Jiayi Zhang, Junchi Yu, Xinlei Yu, Xiawu Zheng, Dongzhan Zhou, et al. Reasoning via video: The first evaluation of video models’ reasoning abilities through maze-solving tasks. arXiv preprint arXiv:2511.15065, 2025.

[96] Ling Yang, Ye Tian, Bowen Li, Xinchen Zhang, Ke Shen, Yunhai Tong, and Mengdi Wang. Mmada: Multimodal large difusion language models. arXiv preprint arXiv:2505.15809, 2025.

[97] Yi Yang, Xiaoxuan He, Hongkun Pan, Xiyan Jiang, Yan Deng, Xingtao Yang, Haoyu Lu, Dacheng Yin, Fengyun Rao, Minfeng Zhu, et al. R1-onevision: Advancing generalized multimodal reasoning through crossmodal formalization. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2376–2385, 2025.

[98] Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

[99] Weihao Yu, Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Zicheng Liu, Xinchao Wang, and Lijuan Wang. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

[100] Shihao Yuan, Yahui Liu, Yang Yue, Jingyuan Zhang, Wangmeng Zuo, Qi Wang, Fuzheng Zhang, and Guorui Zhou. Ar-grpo: Training autoregressive image generation models via reinforcement learning. arXiv preprint arXiv:2508.06924, 2025.

[101] Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 9556–9567, 2024.

[102] Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493, 2022.

[103] Kai Zou, Ziqi Huang, Yuhao Dong, Shulin Tian, Dian Zheng, Hongbo Liu, Jingwen He, Bin Liu, Yu Qiao, and Ziwei Liu. Uni-mmmu: A massive multi-discipline multimodal unified benchmark. arXiv preprint arXiv:2510.13759, 2025.

## Appendix

In this supplementary appendix, we provide additional details and analyses:

• Appendix A covers implementation details of the two-stage training pipeline

• Appendix B presents further analysis of the VR-X benchmark and evaluation metrics

• Appendix C includes additional ablation studies and visualizations.

## A Implementation Details

## A.1 Cold Initialization

Data Construction. We construct our SFT training data from multiple sources, as summarized in Tab. 4. These datasets vary in size and diversity; we weight all subsets according to their sample counts during training. Using the pipeline shown in Fig. 5, we filter raw data into curated training samples. This data processing pipeline consists of four stages. First, we select raw video sequences from the source data and perform scene-aware temporal sampling via PySceneDetect [9] at 0.27 FPS, which preserves richer information than random sampling. This is followed by SigLIP2-based [75] deduplication and VLM-based filtering to remove low-quality frames such as facial close-ups, blurred frames, and blank screens, typically yielding hundreds of frames from a minute-long video. Second, we leverage Qwen3.5-397B [54] to synthesize reasoning-oriented questions and corresponding textual answers from the sampled sequences, with approximately 10 key steps per trajectory. Third, conditioned on these QA pairs, we prompt Qwen3.5 to identify the most relevant query image and key-step frames. Finally, we apply a rigorous quality filter to discard low-quality samples, including those with image-text mismatches, trivial questions requiring no reasoning or planning, and poor visual quality. This pipeline filters out nearly 80% of the candidate pool, ensuring that the remaining samples represent the most informative and logically sound visual reasoning trajectories.

For non-video sources such as VisualCoT and ZebraCoT, which are already formatted as image sequences, we bypass temporal sampling and standardize them to match video-derived data. Overall, we curate 310k cold initialization samples from 1.5M raw candidates.

Optimization. For cold initialization, we initialize from Emu3.5-34B and train UniVR using the configurations in Tab. 3 (second column) on 32 GPUs with full parameters. Each image is resized to 512 on the short side and tokenized by Emu3.5’s VQ tokenizer, yielding 1,000–1,500 tokens per image. We cap the maximum sequence length per sample at 15,000 to reduce training overhead while accommodating reasoning trajectories spanning approximately several minutes.

## A.2 Reinforcement Learning

Data Construction. UniVR employs a novel visual reasoning reward that requires sequences with complex reasoning procedures. To this end, we filter hard samples from the original 310k training set using the post-initialization model, yielding approximately 3k samples. This subset comprises roughly 2k long-term planning trajectories, predominantly spanning 6–10 reasoning steps—and 1k general reasoning data to preserve task diversity.

Optimization. For reinforcement learning, we initialize from the SFT checkpoint and train with full parameters on 32 GPUs using the configuration in Tab. 3 (third column). The Qwen evaluator is deployed on an additional 8 GPUs. To improve training eficiency, we use Qwen3-VL-30B-A3B as the evaluator. Larger variants (e.g., Qwen3-VL-235B or Qwen3.5-397B) yield more accurate rewards but significantly increase training latency. Classifier-Free Guidance (CFG) is disabled during rollout.

## B Details on VR-X Benchmark

The VR-X evaluation set comprises 1.8k high-quality reasoning trajectories curated by professional annotators from a held-out subset (detailed distribution in Tab. 4). VR-X employs two metrics: VLM score and JEPA similarity score, detailed below.

![](images/6bf14a34170331e307dfaa1a2203faf55905d1a1de10356116f8dfc83ffaaf46.jpg)

![](images/f08e53a340b84a55bf707d1ce52d727f5d9f720ca034cfc9b1254afe53345be9.jpg)

Stage 3: Select the frames most relevant to the text description  
![](images/6ea028c64908008c444b7e2e0fc5e382576d9aaae38529969c416cd550d83490.jpg)

![](images/9e7384c26f01e99f6a5a5a081ccf2c9d84483a552f9dafb50755d101784f26db.jpg)  
Figure 5 Data Generation Pipeline.

VLM Evaluation Detail. For the VLM score, we design a unified prompt covering visual quality, task completion, logical coherence, physical dynamics, and temporal consistency. Both the GT and generated sequence are fed into the VLM, enabling it to reference the GT logic process and action dynamics for more accurate judgment. To assess the alignment between VLM and human evaluation, we first have professional annotators score model outputs on the benchmark. The samples are then shufled and presented with corresponding VLM and human scores, allowing annotators to judge which is superior. From this, we compute the Spearman correlation [69] between human and VLM scores, which reaches approximately 0.85, indicating a high degree of human alignment.

Failure cases falsely scored as high-reward by global reasoning alone.  
![](images/b25ac939b129be23ab0e6072be70fb124ac82ca8f4a2283110198b6ddfcc3230.jpg)

![](images/44d932244e31524e70c7db7a1bf69baa96f0a8c6f28e808eb6d94597bf19ba24.jpg)

![](images/e6c4ef0a5245411d604a5d54064c58cd211463a62995ecbe5683579c3969800d.jpg)  
Figure 6 Analysis of the Visual Reasoning Reward. (Up) Relying solely on the global reward misses step-level errors in long-horizon planning tasks, e.g. hangers penetrating garments (top row), erroneous liquid pouring dynamics (middle row), and jittering transitions between paper towels (bottom row). (Down) RL reward curves with and without the step-level reward component.

JEPA Evaluation Detail. For the JEPA score, we follow the implementation in [45]. Specifically, its computation resembles traditional video metrics such as FVD [76], but replaces the I3D [8] feature extractor with a V-JEPA [1] encoder. Image sequences are compressed into 1280-dimensional latent vectors, and the distance between two feature distributions is computed via Maximum Mean Discrepancy (MMD) with a polynomial kernel—smaller distances indicate higher similarity. Since the V-JEPA encoder is trained to encode spatiotemporal coherence and physical dynamics, the JEPA score serves as a complement to the VLM score, assessing whether generated sequences conform to real-world transitions. This metric is applied only to the long-term planning subset, as it targets visual sequences containing real-world dynamics and is unsuitable for single-step reasoning tasks or simulated scenarios in general reasoning. Compared to FVD, the JEPA score converges with approximately 1,000 samples, aligning with the scale of the VR-X benchmark.

Table 3 Training hyperparameters for Cold-Start and RL stages.

<table><tr><td>Hyperparameters</td><td>Cold-Start</td><td>RL</td></tr><tr><td>Learning rate</td><td> $5 \times 10^{-4}$ </td><td> $1 \times 10^{-5}$ </td></tr><tr><td>LR scheduler</td><td>Cosine</td><td>Cosine</td></tr><tr><td>Weight decay</td><td>0.1</td><td>0.1</td></tr><tr><td>Gradient norm clip</td><td>5.0</td><td>5.0</td></tr><tr><td>Warm-up steps</td><td>700</td><td>0</td></tr><tr><td>Training steps</td><td>10k</td><td>250</td></tr><tr><td>Sequence length</td><td>15000</td><td>15000</td></tr><tr><td>Batch Size</td><td>128</td><td>128</td></tr><tr><td>Resolution</td><td>[512, 640]</td><td>[512, 640]</td></tr><tr><td> $\lambda$ </td><td>-</td><td>2.0</td></tr></table>

Table 4 Data composition.

<table><tr><td>Category</td><td>Data Source</td><td>Frames</td><td>Ratios</td></tr><tr><td rowspan="4">Visual Guidance</td><td>EgoDex [30]</td><td>289,053</td><td>23.7%</td></tr><tr><td>Action100M [10]</td><td>109,029</td><td>5.0%</td></tr><tr><td>Epic-kitchen [16]</td><td>53,605</td><td>2.5%</td></tr><tr><td>VideoCraftBench [59]</td><td>2,272</td><td>1.0%</td></tr><tr><td rowspan="4">Robot Manipulation</td><td>AgiBot [7]</td><td>427,267</td><td>24.5%</td></tr><tr><td>Droid [34]</td><td>13,000</td><td>1.0%</td></tr><tr><td>Bridge [78]</td><td>14,850</td><td>1.6%</td></tr><tr><td>ZebraCoT-Robot [36]</td><td>54,270</td><td>3.5%</td></tr><tr><td>Editing</td><td>ZebraCoT-Multiobject [36]</td><td>128,075</td><td>6.5%</td></tr><tr><td rowspan="2">Spatial Perception</td><td>ThinkMorph-Navigation [24]</td><td>68,568</td><td>11.2%</td></tr><tr><td>ZebraCoT-Embodiment [36]</td><td>58,931</td><td>3.2%</td></tr><tr><td rowspan="2">Visual Search</td><td>VisualCoT [63]</td><td>30,000</td><td>4.9%</td></tr><tr><td>ThinkMorph-Search [24]</td><td>13,980</td><td>2.3%</td></tr><tr><td rowspan="3">Puzzle &amp; Game</td><td>VRBench [95]</td><td>13,000</td><td>1.0%</td></tr><tr><td>Zebra-Jigsaw [36]</td><td>43,798</td><td>7.1%</td></tr><tr><td>ThinkMorph-VisPuzzle [24]</td><td>13,000</td><td>1.0%</td></tr></table>

## C More Analysis and Visualizations

Visual reasoning rewards mitigates reward hacking. Fig. 6 presents additional cases where the global reasoning reward alone yields near-perfect VLM scores, yet manual inspection reveals physical and logical errors, such as implausible hanger-garment interactions, incorrect wine-pouring dynamics, and flawed towel-replacement logic. These samples span extended durations (30+ seconds), with errors localized to merely a few frames that global VLM assessment easily overlooks. This phenomenon is reflected in the training curves (Fig. 6, bottom left) the global-reward curve exhibits sharp spikes and top-end oscillations, indicating shortcut-seeking behavior that neglects intermediate-step correctness. In contrast, the VR-GRPO curve (bottom right) ascends more smoothly, demonstrating substantially lower reward hacking risk.

## VR-GRPO stabilizes long-horizon visual reasoning.

We analyze the scalability of VR-GRPO across varying temporal horizons by partitioning test samples into four duration-based groups: <10s, 10–30s, 30–60s, and >60s. As shown in Tab. 5, VR-GRPO delivers the most pronounced gains on >30s sequences. We attribute this to its step-level reward design, which explicitly maintains logical coherence

<table><tr><td>Method</td><td>&lt;10s</td><td>10-30s</td><td>30-60s</td><td>&gt;60s</td></tr><tr><td>Emu3.5</td><td>54.2</td><td>48.0</td><td>38.9</td><td>21.7</td></tr><tr><td>Cold-Start</td><td>58.7</td><td>54.5</td><td>41.1</td><td>26.9</td></tr><tr><td>UniVR</td><td>71.1</td><td>61.0</td><td>56.7</td><td>45.6</td></tr><tr><td> $\triangle v.s.$  Emu3.5</td><td>↑ 16.9</td><td>↑ 13.0</td><td>↑ 17.8</td><td>↑ 23.9</td></tr></table>

Table 5 Performance across diferent time span.

and physical consistency at intermediate stages. By correcting error-prone steps, VR-GRPO efectively mitigates compounding errors that typically accumulate during long-range prediction, thereby exhibiting superior stability on extended reasoning traces.

Please tie a simple knot in the gift on the table.  
![](images/42b55c1d0ebd2685605cdc60befca8220432e1c228eb56f21781ea07bb83dfe1.jpg)  
Figure 7 Comparison with Gemini and Emu3.5. In each group, the first, second, and third rows correspond to Gemini 3 Pro + Nano Banana 2, Emu3.5, and UniVR, respectively.

More Comparisons with Baselines. Fig. 7 provides additional side-by-side comparisons with Gemini 3 Pro + Nano Banana 2 and Emu3.5 on identical test samples. While both baselines generate high-fidelity visual appearances, they exhibit execution inaccuracies in fine-grained tasks such as rope manipulation, garment unfolding and folding, paper-folding dynamics, and environmental consistency during policy execution. Benefiting from our visual reasoning training pipeline, UniVR achieves comparable visual fidelity while demonstrating superior logical coherence and physical consistency.

More Visualization. Fig. 8 shows results of UniVR across additional scenarios, generating long-horizon sequences with coherent logic, physical dynamics, and temporal consistency.

Limitations. Despite these promising results, UniVR presents several limitations worth noting. First, our training on 34B-scale models with long visual sequences demands substantial computational resources, potentially limiting accessibility. Second, although VR-GRPO improves evaluation quality via its step-level design, our reward mechanism still relies on general purpose VLMs with limited fine-grained physical-world knowledge. A more powerful reward system natively grounded in visual world dynamics is needed. Finally, although visual reasoning holds significant potential for learning complex world knowledge and enhancing multimodal understanding, achieving optimal synergy among native visual, textual, and auditory reasoning remains an open question for future work.

![](images/dcf00c567bb8c2fad66bef769fea02d926853d1cb8d636f25f9d7e32fb70699c.jpg)  
Figure 8 More visualization of UniVR.