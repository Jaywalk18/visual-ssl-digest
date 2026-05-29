# OmniVerifier-M1: Multimodal Meta-Verifier with Explicit Structured Recalibration

Xinchen Zhang1∗† Bowei Liu1∗ Jiale Liu2 Chufan Shi3 Yizhen Zhang1 Junhong Liu4 Youliang Zhang1 Zhiheng Li1 Yujiu Yang1‡ Ling Yang5‡

1Tsinghua University 2Pennsylvania State University 3University of Southern California 4Microcyto 5Princeton University ∗: Equal Contribution. †: Project Leader. ‡: Corresponding Authors.

# zhangxc24@mails.tsinghua.edu.cn, yangling0818@163.com   
Ð https://github.com/Cominclip/OmniVerifier   
 https://omniverifier.github.io/

# Abstract

Visual outcomes are increasingly central to multimodal large language models, making reliable and fine-grained verification essential for scaling generalist foundation models. In this work, we investigate multimodal meta-verification, which leverages verifier-generated rationales rather than decision-only signals, and explore how to effectively incorporate meta-verification feedback into multimodal verifier training. We identify two key findings. First, symbolic verifier outputs (e.g., bounding boxes) outperform textual explanations as meta-verification rationales, enabling efficient rule-based reinforcement learning rewards while avoiding reliance on modelbased rewards from auxiliary judge models. Second, decoupling reinforcement learning objectives for binary judgment and meta-verification substantially outperforms joint reward optimization, due to intrinsic differences in output structure and learning dynamics. Based on these insights, we train OmniVerifier-M1, a generalist visual verifier leveraging symbolic meta-verification and decoupled reinforcement learning. OmniVerifier-M1 provides robust verification and finegrained error localization, and further enables M1-TTS, a verifier-driven agentic generation system achieving dynamic region-level self-correction. This approach paves the way for more reliable, interpretable, and fine-grained multimodal verification, supporting safer and more controllable foundation model deployment.

# 1 Introduction

Current multimodal large language models (MLLMs) demonstrate powerful reasoning and generative capabilities in a variety of inference scenarios and reasoning modes (Guo et al., 2025b; Seed, 2026; Comanici et al., 2025; Zhang et al., 2025). Visual outcomes serve as a crucial bridge connecting multimodal understanding and generation, whether they are produced via agentic tool-use (OpenAI, 2025; Zheng et al., 2025) or through native generative processes (Liao et al., 2025; Gu et al., 2025; Deng et al., 2025). In interleaved multimodal reasoning and interactive systems, enabling precise, fine-grained, and reliably evaluable verification of visual outcomes is a key requirement for scaling unified multimodal models and advancing generative intelligence.

Universal verification of visual outcomes remains at an early stage. Most existing image reward models (Xu et al., 2024; Zhang et al., 2024c), such as RewardDance (Wu et al., 2025b) and UnifiedReward (Wang et al., 2025a), focus primarily on training and evaluation in traditional text-to-image generation scenarios. OmniVerifier (Zhang et al., 2025) marks an important step toward more general, world-modeling-oriented visual verification by leveraging reinforcement learning with binary (True/False) judgments of visual outcomes. However, feedback limited to binary decisions without supervision from detailed generative critiques can be coarse and uninformative, reducing the granularity needed for precise and effective judge model improvment (Shao et al., 2025; Wang et al., 2026).

In this work, we move beyond binary verifier judgments and examine the reliability of verifier-generated rationales and explanations, a process referred to as meta-verification (Shao et al., 2025; Wang et al., 2026). Instead of relying on decision-level signals, meta-verification operates at the level of explanations to guide the learning objective, yielding more informative and more restrictive feedback. In the investigation of how to improve multimodal verifier training by integrating meta-verification feedback, this work identifies two core findings:

# Finding 1: Symbolic verifier outputs beat textual ones in meta-verification, enabling scalable and reliable rule-based RL rewards.

Motivated by the highly structured and spatial nature of visual representations, we use symbolic outputs (e.g., bounding boxes or points) as rationales for meta-verification feedback when training the verifier, instead of relying on textual explanations. Textual rationales require additional judge models for evaluation, which slows down meta-verification feedback and increases the risk of reward hacking. In contrast, symbolic rationales provide a structured approximation of explanatory intent that can be directly assessed with explicit rules. Experiments show that in meta-verification training, symbolic rationales consistently match or outperform textual explanations, allowing rule-based feedback to replace model-based rewards, improving training efficiency while inherently preventing reward hacking.

# Finding 2: Decoupling RL rewards for binary judgment and meta-verification outperforms joint training in leveraging meta-verification feedback.

In exploring how to better leverage meta-verification feedback for training the verifier, we find that combining binary judgment accuracy and meta-verification reward into a single joint reward for each sample offers little improvement in judgement accuracy. This is due to intrinsic differences in task structure and difficulty: binary judgments operate in a highly discrete output space, allowing the model to occasionally score well by chance, whereas meta-verification provides continuous, stronger supervision that effectively constrains such random behavior. To address this, we design a decoupling strategy that treats binary judgment and meta-verification as separate tasks with distinct reward systems for mixed data. Both empirical results and theoretical analysis confirm the superiority of decoupled training over joint training in the using of meta-verification.

Based on these observations, we train OmniVerifier-M1, a multimodal verifier adaptable to diverse multimodal foundation models (Cui et al., 2025; Cao et al., 2025). We adopt a decoupled training paradigm that leverages meta-verification feedback derived from symbolic outputs, enabling more effective and stable verifier optimization. Beyond serving as a multimodal visual verifier, OmniVerifier-M1 functions as a fine-grained multimodal optimizer that can precisely localize erroneous regions and provide actionable correction guidance. Building on this capability, we further develop a fine-grained multimodal agentic generation system, M1-TTS, in which verifier-driven decisions are expressed as heterogeneous, tool-level actions, including symbolic region localization and structured textual edit operations, and are iteratively coordinated through replanning to guide a unified foundation model toward region-level self-correction. Experimental results show that M1-TTS substantially outperforms conventional global-level multi-turn editing methods in correction effectiveness.

Our contributions can be summarized as follows:

• Multimodal Meta-Verification Paradigm: We bring meta-verification to multimodal setting, enabling fine-grained verifier feedback beyond binary judgment.   
• Symbolic Meta-Verification Rationales: We show that symbolic verifier outputs outperform textual explanations as meta-verification rationales, supporting efficient rule-based RL without reward hacking.   
• Decoupled Meta-Verification Training: We theoretically and empirically demonstrate that decoupling reinforcement learning objectives for binary judgment and meta-verification substantially outperforms joint reward optimization.

• Generalist Verifier and Agentic Correction System: We develop OmniVerifier-M1 and M1- TTS, a generalist multimodal verifier and an agentic correction system that support robust visual verification and effective region-level self-correction across diverse generative foundation models.

# 2 Related Work

Generative Veirifer or Reward Models. Unlike traditional reward models that only output a scalar reward (Ouyang et al., 2022; Zhang et al., 2024a; Xu et al., 2023; Luo et al., 2025; 2026), generative verifiers provide interpretable, generative critiques, offering immense potential for scaling test-time computation or reinforcement learning (Zhang et al., 2025; Liu et al., 2025; Wang et al., 2026; Yang et al., 2026b). LLMor VLM-as-a-Judge (Zhu et al., 2023; Chen et al., 2025; 2024) methods leverage the reasoning capabilities of large models to make evaluations more transparent and accurate, pioneering the use of foundation models as evaluators. DeepSeekMath-V2 (Shao et al., 2025) introduces meta-verification to assesses whether issues identified by the verifier indeed exist, which enhance verifier training by providing strict supervision. OmniVerifier (Zhang et al., 2025) identifies three fundamental atomic capabilities for verifying visual outcomes, marking a first step toward a general-purpose mutlimodal verifier for universal scenarios. However, exploration of multimodal verifiers is still in an early stage. Starting from the essence of visual representations, we develop a robust multimodal verifier training paradigm based on symbolic outputs with decoupled reinforcement learning.

Iterative Refinement for Visual Generation. As we move towards more general visual generation scenarios, especially complex compositional generation (Zhang et al., 2024b; Yang et al., 2024; 2025b) or world-knowledge reasoning tasks (Wang et al., 2025b; Hu et al., 2025; Yang et al., 2026a), it is difficult to achieve perfect results in a single attempt. Many approaches address this by combining a visual verifier with a generative model, employing a generate-reflect-refine loop to progressively improve generated images (Qin et al., 2025; Jiang et al., 2025; Huang et al., 2025a; Jaiswal et al., 2026). ReflectionFlow (Zhuo et al., 2025) constructs large-scale dataset to perform reflection tuning on diffusion transformer to achieve multiround refinement. OmniVerifier-TTS (Zhang et al., 2025) bridge the image generation and edit within unifed multimodal models through the guidence of visual verifier. These methods optimize images from a high-level, macro perspective. However, erroneous regions are often small and can be easily confused with visually similar attributes, making precise, multi-dimensional control via textual descriptions challenging. To address this, we build an agentic generation system based on symbolic verifier outputs, allowing targeted, region-level corrections through efficient multi-round refinement.

# 3 Problem Formulation

We study reinforcement learning-based training of a pointwise multimodal verifier under the RLVR (Reinforcement Learning with Verifier Rewards) paradigm (Shao et al., 2024; Guo et al., 2025a). Our goal is to train a verifier that not only determines whether a visual outcome satisfies the given prompt, but also produces transparent, fine-grained, and actionable critiques, providing reliable supervision for model reflection and refinement.

# 3.1 Baseline RLVR Training for Multimodal Verifiers

Let $\boldsymbol { \mathcal { D } } = \{ ( x _ { i } , y _ { i } ) \}$ denote the training set, where $x _ { i } = ( I _ { i } , P _ { i } )$ consists of an image $I _ { i }$ and its corresponding prompt $P _ { i }$ , and $y _ { i } \in \{ \mathsf { T r u e } , \mathsf { F a l s e } \}$ is the ground-truth judgment of whether the image satisfies the prompt.

A visual verifier $\pi _ { \theta }$ takes $( I , P )$ as input and generates a textual output o. A binary decision $\hat { y } \in$ {True, False} is then deterministically parsed from o according to a predefined output format:

$$
(o, \hat {y}) = \pi_ {\theta} (I, P). \tag {1}
$$

The RL objective for training the verifier is:

$$
\max_{\pi_{\theta}} \mathbb{E}_{\substack{(I_{i},P_{i},y_{i})\sim \mathcal{D},\\ (o_{i},\hat{y}_{i})\sim \pi_{\theta}(\cdot |I_{i},P_{i})}}\bigl [\mathcal{R}_{\mathrm{f}}(o_{i}) + \mathcal{R}_{\mathrm{acc}}(\hat{y}_{i},y_{i})\bigr ]. \tag{2}
$$

![](images/5a0da0f383a81451cb489700acf3f6f2f80e03f6e9e0b91e2628a4f2cc4774bf.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Prompt: A person with red backpack sitting on a bench"] --> B["Textual Verifier"]
    B --> C["True/False Textual Explanation"]
    B --> D["GT Explanation"]
    B --> E["Model-based Meta-Verifier"]
    E --> F["Slow, Hacking Reward"]
    G["Symbolic Verifier"] --> H["True/False Symbolic BBox"]
    G --> I["GT BBox"]
    G --> J["Rule-based Meta-Verifier IoU, L2 Dis,... Fast, Reliable Reward"]
    K["Joint RL Training"] --> L["BatchSize: B"]
    L --> M["0.5B True R_Acc"]
    L --> N["0.5B False R_Acc R_meta"]
    O["Decoupled RL Training"] --> P["BatchSize: 1.5B"]
    P --> Q["0.5B True R_Acc"]
    P --> R["0.5B False R_Acc"]
    P --> S["0.5B Grounding R_meta"]
    T["Verification Accuracy Policy"] --> U["Grounding Policy"]
    V["GT Bbox"] --> W["Rule-based Meta-Verifier"]
```
</details>

Figure 1: Pipeline of two key findings. Left: the advantage of symbolic bounding boxes over textual explanations, enabling rule-based rewards to inherently prevent reward hacking and accelerate training. Right: the comparison between joint training and decoupled training.

Format Reward. The format reward $\mathcal { R } _ { \mathrm { f } } ( \cdot )$ requires the verifier ouput $o _ { i }$ to perform an explicit reasoning step before giving the final judgment, where the verifier is instructed to include its intermediate analysis within <think> and </think> tags. The reward is realized as an indicator function checking strict adherence to this structure.

Accuracy Reward. The accuracy reward $\mathcal { R } _ { \mathrm { a c c } } ( \cdot )$ is a binary reward defined as

$$
\mathcal {R} _ {\mathrm{acc}} (\hat {y}, y) = \left\{ \begin{array}{l l} 1, & \text { if } \hat {y} = y, \\ 0, & \text { otherwise. } \end{array} \right. \tag {3}
$$

This reward provides supervision only at the decision level, without considering the correctness of the verifier’s reasoning or generative critique. While it can guide the model to learn coarse judgments, the learning signal is limited and easily exploitable: the model can achieve high reward by guessing or following superficial patterns, rather than performing meaningful verification. Consequently, this formulation fails to encourage fine-grained, interpretable, and reliable verification behavior.

# 3.2 Meta-Verification Enhanced RLVR Training

To overcome the limitations of decision-only supervision, meta-verification is used to enhance RLVR training of the verifier (Shao et al., 2025). In this setting, the verifier is required to produce not only a binary decision, but also an explicit rationale when the decision is negative. Formally, the verifier outputs:

$$
(o, \hat {y}, e) = \pi_ {\theta} (I, P), \tag {4}
$$

where $\hat { y } \in \{ \mathsf { T r u e } , \mathsf { F a l s e } \}$ and e denotes an explanation, which is only required when $\hat { y } = \mathsf { F a l s e }$

By integrating meta-verification feedback into the reward function, the enhanced verifier RL objective is formulated as:

$$
\max_ {\pi_ {\theta}} \mathbb {E} _ {\substack {(I _ {i}, P _ {i}, y _ {i}) \sim \mathcal {D}, \\ (o _ {i}, \hat {y} _ {i}, e _ {i}) \sim \pi_ {\theta} (\cdot | I _ {i}, P _ {i})}} [ \mathcal {R} _ {\mathrm{f}} (o _ {i}) + \mathcal {R} _ {\text {acc}} (\hat {y} _ {i}, y _ {i}) \cdot (\mathbb {I} [ y _ {i} = \text {True} ] + \mathbb {I} [ y _ {i} = \text {False} ] \cdot \mathcal {R} _ {\text {meta}} (e _ {i})) ]. \tag{5}
$$

Meta-Verification Reward. The meta-verification reward $\mathcal { R } _ { \mathrm { m e t a } } ( \cdot )$ evaluates the correctness and validity of the verifier-generated rationale eˆ. Specifically, a separate meta-verifier $\mathcal { M } _ { \phi }$ is used to assess whether the explanation correctly identifies genuine issues in the visual outcome:

$$
\mathcal {R} _ {\text { meta }} = \mathcal {M} _ {\phi} (I, P, \hat {e}) \in \mathbb {R}. \tag {6}
$$

This reward provides supervision at the explanation level, encouraging the verifier to produce faithful and informative rationales rather than spurious or hallucinated justifications. By incorporating metaverification feedback, the verifier receives denser and more restrictive learning signals that go beyond binary correctness, enabling improved reliability, interpretability, and training efficiency.

![](images/0d2b43446987dbd011f26827cc68b527984749beeabdda67bfe73ce7c2fd5a09.jpg)

<details>
<summary>line</summary>

| Training Steps | Qwen3-VL 8B Symbolic Bbox | Qwen3-VL 8B Textual Exp | OmniVerifier 7B Symbolic Bbox | OmniVerifier 7B Textual Exp |
| -------------- | ------------------------- | ----------------------- | ----------------------------- | ---------------------------- |
| 0              | 0.73                      | 0.73                    | 0.73                          | 0.73                         |
| 10             | 0.75                      | 0.75                    | 0.75                          | 0.75                         |
| 20             | 0.76                      | 0.76                    | 0.76                          | 0.76                         |
| 30             | 0.77                      | 0.77                    | 0.77                          | 0.77                         |
| 40             | 0.78                      | 0.78                    | 0.78                          | 0.78                         |
| 50             | 0.79                      | 0.79                    | 0.79                          | 0.79                         |
| 60             | 0.80                      | 0.80                    | 0.80                          | 0.80                         |
| 70             | 0.81                      | 0.81                    | 0.81                          | 0.81                         |
| 80             | 0.82                      | 0.82                    | 0.82                          | 0.82                         |
</details>

![](images/b09cdd2a90fa97fcb4577f1175a347ac0af2de8187984cfe60bea2c2e21fda19.jpg)

<details>
<summary>line</summary>

| Training Steps | Qwen3-VL 8B Symbolic Bbox | Qwen3-VL 8B Textual Exp | OmniVerifier 7B Symbolic Bbox | OmniVerifier 7B Textual Exp |
| -------------- | -------------------------- | ------------------------ | ----------------------------- | ---------------------------- |
| 0              | 0.72                       | 0.72                     | 0.75                          | 0.75                         |
| 10             | 0.73                       | 0.74                     | 0.75                          | 0.76                         |
| 20             | 0.74                       | 0.75                     | 0.76                          | 0.76                         |
| 30             | 0.75                       | 0.76                     | 0.77                          | 0.77                         |
| 40             | 0.76                       | 0.78                     | 0.78                          | 0.78                         |
| 50             | 0.77                       | 0.78                     | 0.78                          | 0.79                         |
| 60             | 0.78                       | 0.79                     | 0.79                          | 0.79                         |
| 70             | 0.78                       | 0.79                     | 0.79                          | 0.79                         |
| 80             | 0.78                       | 0.79                     | 0.79                          | 0.79                         |
</details>

![](images/05da4334f251c4902a4a00814946cc3f45321a28bc2b1fc6c8fdac8cdae81842.jpg)

<details>
<summary>line</summary>

| Training Steps | Qwen3-VL 8B Symbolic Bbox | Qwen3-VL 8B Textual Exp | OmniVerifier 7B Symbolic Bbox | OmniVerifier 7B Textual Exp |
| -------------- | ------------------------- | ----------------------- | ----------------------------- | ---------------------------- |
| 0              | 0.650                     | 0.654                   | 0.650                         | 0.650                        |
| 10             | 0.658                     | 0.660                   | 0.652                         | 0.654                        |
| 20             | 0.660                     | 0.662                   | 0.654                         | 0.656                        |
| 30             | 0.662                     | 0.664                   | 0.656                         | 0.658                        |
| 40             | 0.664                     | 0.666                   | 0.658                         | 0.660                        |
| 50             | 0.666                     | 0.668                   | 0.660                         | 0.662                        |
| 60             | 0.668                     | 0.670                   | 0.662                         | 0.664                        |
| 70             | 0.670                     | 0.672                   | 0.664                         | 0.666                        |
| 80             | 0.672                     | 0.674                   | 0.666                         | 0.668                        |
</details>

Figure 2: Comparison between symbolic bounding boxes and textual explanations as meta-verification signals in verifier RLVR training.

In subsequent sections, we further analyze how different forms of rationales and reward coupling strategies affect optimization dynamics, and show that symbolic rationales combined with decoupled reinforcement learning objectives yield substantially better performance.

# 4 Symbolic Rationales for Rule-Based Multimodal Meta-Verification

Drawbacks of Model-Based Meta-Verifiers. Model-based reward models in RLVR aim to leverage the core capabilities of LLMs, particularly their advanced reasoning skills to produce more accurate judgments and rewards (Chen et al., 2025; Whitehouse et al., 2025). Their flexibility mitigates the rigidity of rule-based rewards, which often struggle to generalize across diverse patterns. However, in dynamic reinforcement learning settings, these approaches are highly vulnerable to reward hacking: models may exploit weaknesses in the verifier to obtain high rewards without genuine improvements in reasoning, and in some cases even at the cost of degraded reasoning performance (Huang et al., 2025b; Wang et al., 2026). Moreover, applying model-based reward to large batches of samples generated during RL rollouts increases both the training cost and the overall training time (Wang et al., 2026).

Revisiting Rule-Based Meta-Verifiers. Beyond domains such as code and mathematics with structured answer, the diversity of output formats and the complexity of semantic composition make it difficult to directly apply rule-based signals as reinforcement learning rewards. In contrast, images constitute highly structured, spatially grounded, and high-dimensional representations. In visual outcome verification, errors in images are not only expressible through textual explanations, they can be captured through symbolic, structured outputs such as bounding boxes, keypoints, or line segments that explicitly localize and characterize failure regions. For example, instead of generating verbose textual explanations, a verifier can output symbolic cues that spatially localize mismatched regions, providing concise and actionable feedback for correction, as shown in Fig. 1. Such grounded symbolic feedback forms a natural basis for rule-based meta-verification, enabling precise error attribution without dependence on unconstrained textual reasoning.

Experimental Setup. We apply DAPO (Yu et al., 2025) to perform RL training on OmniVerifier-7B (Zhang et al., 2025; Bai et al., 2025b) and Qwen3-VL-8B (Bai et al., 2025a). For each training sample, we provide ground-truth binary judgments (True/False) together with ground-truth textual explanations and bounding boxes for meta-verification. For textual explanation, we use Qwen3-4B (Yang et al., 2025a) to perform model-based comparation between the groundtruth explanation and verifier generated explanation to answer whether the two is semantically equal. For symbolic bounding box, we use intersection over union (IoU) as rule-based reward to provide meta-verification feedback. All the two models are trained for 80 steps on 16 NVIDIA A800-80G GPUs. We evaluate both models on ViVerBench (Zhang et al., 2025), a comprehensive and challenging benchmark designed for visual-outcome verification.

Table 1: Performance on ViVerBench and efficiency analysis. 

<table><tr><td>Model</td><td>ViVerBench (Overall)</td><td>Per-Card GPU Memory (GB)</td><td>Per-Sample Reward Computation Time (ms)</td><td>Training Time per Step (min)</td><td>Mean Response Length (tokens)</td></tr><tr><td>OmniVerifier 7B</td><td>0.650</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>OmniVerifier 7B(Bbox)</td><td>0.661</td><td>48.6</td><td>0.021</td><td>8.13</td><td>384</td></tr><tr><td>OmniVerifier 7B(Exp)</td><td>0.662</td><td>56.9</td><td>20.2</td><td>10.27</td><td>340</td></tr><tr><td>Qwen 3-VL 8B</td><td>0.654</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Qwen 3-VL 8B(Bbox)</td><td>0.671</td><td>49.9</td><td>0.021</td><td>8.74</td><td>516</td></tr><tr><td>Qwen 3-VL 8B(Exp)</td><td>0.670</td><td>58.3</td><td>20.2</td><td>11.08</td><td>488</td></tr></table>

Experimental Analysis. From Fig. 2, we observe that during training, the accuracy on the training set exhibits highly similar trends for both models, whether using symbolic bounding boxes or textual explanations as meta-verification signals. Moreover, their performance on both in-domain test sets and ViVerBench is also remarkably similar. This indicates that employing a rule-based IoU reward as metaverification can serve as a reliable proxy for textual explanations. It effectively guides the verifier to improve its capabilities, while the symbolic format allows direct adherence to rule-based reward modeling, elegantly mitigating the issue of reward hacking at its source.

Additionally, as shown in Table 1, we compare the computational overhead of rule-based and model-based meta-verification from both training and inference perspectives. During training, symbolic outputs show clear efficiency advantages over textual explanations by reducing GPU memory usage, per-sample reward computation time, and per-step training time, while maintaining comparable inference efficiency with similar response lengths. Therefore, in multimodal verification scenarios, symbolic bounding box outputs can effectively replace textual explanations, providing comparable supervisory strength and inference-side overhead while substantially mitigating reward hacking and reducing training costs.

Finding 1. Symbolic verifier outputs beat textual ones, unlocking rule-based RL rewards in meta-verification.

# 5 Decoupled Reinforcement Learning Incentivizing Meta-Verification

We investigate a general reinforcement learning paradigm for multimodal verifier training with metaverification training. The formulation in Eq. 5 as joint training: for each training sample, we first assess the correctness of the binary judgment. When both the model prediction and the ground-truth label are False, we further employ a rule-based verifier (e.g., IoU) to generate meta-verification feedback.

A careful analysis of joint training reveals two intrinsic limitations. First, the meta-verification reward $\mathcal { R } _ { \mathrm { m e t a } } ( \cdot )$ is activated only when both the prediction of the model and the ground-truth label are False, leading to a conditional and discontinuous gradient flow for the meta-verification objective. Second, binary judgment and meta-verification differ fundamentally in output structure and optimization landscape: the former operates over a discrete, low-entropy label space, while the latter requires learning continuous, finegrained outputs. Jointly optimizing these heterogeneous objectives induces conflicting learning dynamics, which motivates an in-depth examination of the joint training paradigm:

Lemma 5.1. In joint RLVR training of multimodal verifier, all gradient terms related to the explanation e are multiplicatively gated by the accuracy reward $\mathcal { R } _ { a c c } ( \cdot )$ .

All proofs are provided in Appendix A. This lemma reveals that under joint training, the verifier must first learn to make correct binary judgments before it can receive reward signals about where the error occurs. Based on this lemma, we have:

Theorem 5.2. Let the verifier’s decision (classification) accuracy on the data distribution be denoted as:

$$
p _ {a c c} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}} \left[ \mathbb {P} _ {\pi_ {\theta}} (\hat {y} = y \mid x) \right], \tag {7}
$$

Then, in joint training, the gradients related to meta-verification satisfy:

$$
\left\| \nabla_ {\theta} J _ {j o i n t} ^ {(e)} \right\| \leq p _ {a c c} (\theta) \cdot \mathbb {E} [ \| \mathcal {R} _ {m e t a} (e) \nabla_ {\theta} \log \pi_ {\theta} (e | x, \hat {y}) \| ]. \tag {8}
$$

![](images/378b18b2bba31557e508be93716858015e3a8455b3c57595e48df46e55ae7f89.jpg)  
Figure 3: Performance comparison between decoupled and joint RL training with symbolic bounding boxes as meta-verification signals.

From this theorem, we observe that in the early stage of RL training, if $p _ { \mathrm { a c c } } ( \theta ) \ll 1$ , we have $\nabla _ { \theta } J _ { \mathrm { i o i n t } } ^ { ( e ) } \approx 0$ This implies that meta-verification can hardly be optimized effectively. In particular, for smaller or less capable models, there exists an inherent gap between binary judgment and meta-verification.

Based on the above analysis of these limitations, we decompose binary judgment and meta-verification into two separate tasks, each served by an independent reward model, rather than coupling the two rewards in a sequential manner. We refer to this strategy as decoupled training. Specifically, as shown in Fig. 1, we start from original dataset $\mathcal { D } = \{ ( x , y ) \}$ , where positive and negative labels $( y = \mathsf { T r u e }$ and $y = \mathsf { F a l s e } )$ are balanced at a 1:1 ratio. The full dataset is used exclusively to supervise the accuracy reward $\mathcal { R } _ { \mathrm { a c c } } ( \cdot )$ In addition, we duplicate all samples with $y = \mathsf { F a l s e }$ , this duplicated subset is supervised solely by the meta-verification reward $\mathcal { R } _ { \mathrm { m e t a } } ( \cdot )$ . In this way, we explicitly decouple the verifier and meta-verification objectives at the dataset level and conduct mixed training across the two tasks.

We provide a detailed gradient-level analysis of both joint training and decoupled training:

Theorem 5.3. Consider the gradient estimator for meta-verification in joint RLVR training:

$$
\mathcal {G} _ {\text { joint }} = \mathcal {R} _ {\text { acc }} (x, \hat {y}) \cdot \mathcal {R} _ {\text { meta }} (e) \cdot \nabla_ {\theta} \log \pi_ {\theta} (e \mid x, \hat {y}), \tag {9}
$$

and the gradient estimator in decoupled training:

$$
\mathcal {G} _ {\text { dec }} = \mathcal {R} _ {\text { meta }} (e) \cdot \nabla_ {\theta} \log \pi_ {\theta} (e \mid x, \hat {y}), \tag {10}
$$

where samples are drawn from the conditional distribution $x \sim \mathcal { D } \mid y = F a l s e$ . Then, the gradient variance in joint training satisfies:

$$
\operatorname{Var} \left(\mathcal {G} _ {\text {joint}}\right) = p _ {\text {acc}} (\theta) \operatorname{Var} \left(\mathcal {G} _ {\text {dec}}\right) + \tag {11}
$$

$$
p _ {a c c} (\theta) \left(1 - p _ {a c c} (\theta)\right) \| \mathbb {E} [ \mathcal {G} _ {d e c} ] \| ^ {2},
$$

and consequently,

$$
\operatorname{Var} (\mathcal {G} _ {\text { joint }}) \geq p _ {a c c} (\theta) \operatorname{Var} (\mathcal {G} _ {d e c}), \tag {12}
$$

with strict inequality when $\mathbb { E } [ \mathcal { G } _ { d e c } ] \neq 0$ and $p _ { a c c } ( \theta ) \in ( 0 , 1 )$ .

Corollary 5.4. Let the signal-to-noise ratio (SNR) of a gradient estimator be defined as

$$
\mathrm{SNR} (\mathcal {G}) = \frac {\left\| \mathbb {E} [ \mathcal {G} ] \right\| ^ {2}}{\mathrm{Var} (\mathcal {G})}. \tag {13}
$$

Table 2: Performance on ViVerBench of joint training and decoupled training. 

<table><tr><td rowspan="2">Model / Metric</td><td colspan="3">Concept Existence</td><td colspan="2">Object Relation</td><td colspan="2">World Dynamics</td><td colspan="3">Image Annotation</td><td colspan="4">State Value Evaluation</td><td colspan="2">STEM</td><td rowspan="2">Overall</td></tr><tr><td>Obj.</td><td>Attr.</td><td>Abs.P.</td><td>Spat.</td><td>N-Spat.</td><td>S-Phy</td><td>D-Phy</td><td>BBox</td><td>Point</td><td>Count</td><td>Maze</td><td>F.Lake</td><td>Robot.</td><td>GUI</td><td>Chart</td><td>LaTeX</td></tr><tr><td>OmniVerifier 7B</td><td>0.701</td><td>0.703</td><td>0.521</td><td>0.808</td><td>0.739</td><td>0.525</td><td>0.596</td><td>0.770</td><td>0.659</td><td>0.527</td><td>0.490</td><td>0.482</td><td>0.728</td><td>0.634</td><td>0.600</td><td>0.918</td><td>0.650</td></tr><tr><td>OmniVerifier 7B(Joint)</td><td>0.723</td><td>0.733</td><td>0.513</td><td>0.833</td><td>0.761</td><td>0.487</td><td>0.564</td><td>0.827</td><td>0.716</td><td>0.640</td><td>0.497</td><td>0.436</td><td>0.601</td><td>0.694</td><td>0.623</td><td>0.928</td><td>0.661</td></tr><tr><td>OmniVerifier 7B(Decoupled)</td><td>0.741</td><td>0.754</td><td>0.506</td><td>0.846</td><td>0.769</td><td>0.467</td><td>0.535</td><td>0.854</td><td>0.741</td><td>0.710</td><td>0.441</td><td>0.443</td><td>0.589</td><td>0.722</td><td>0.639</td><td>0.931</td><td>0.668</td></tr><tr><td>Qwen 3-VL 8B</td><td>0.710</td><td>0.690</td><td>0.562</td><td>0.642</td><td>0.716</td><td>0.604</td><td>0.593</td><td>0.831</td><td>0.741</td><td>0.489</td><td>0.420</td><td>0.693</td><td>0.671</td><td>0.787</td><td>0.540</td><td>0.773</td><td>0.654</td></tr><tr><td>Qwen 3-VL 8B(Joint)</td><td>0.732</td><td>0.724</td><td>0.534</td><td>0.704</td><td>0.754</td><td>0.595</td><td>0.582</td><td>0.847</td><td>0.773</td><td>0.523</td><td>0.458</td><td>0.664</td><td>0.639</td><td>0.815</td><td>0.568</td><td>0.824</td><td>0.671</td></tr><tr><td>Qwen 3-VL 8B(Decoupled)</td><td>0.750</td><td>0.733</td><td>0.527</td><td>0.717</td><td>0.768</td><td>0.583</td><td>0.596</td><td>0.863</td><td>0.784</td><td>0.565</td><td>0.380</td><td>0.717</td><td>0.652</td><td>0.838</td><td>0.572</td><td>0.835</td><td>0.680</td></tr></table>

Table 3: Performance on RefCOCO of joint training and decoupled training. 

<table><tr><td>Model / Metric</td><td>RefCOCO val</td><td>RefCOCO test-A</td><td>RefCOCO test-B</td><td>RefCOCO+ val</td><td>RefCOCO+ test-A</td><td>RefCOCO+ test-B</td><td>RefCOCOg val</td><td>RefCOCOg test</td><td>Overall</td></tr><tr><td>OmniVerifier 7B</td><td>0.807</td><td>0.890</td><td>0.750</td><td>0.757</td><td>0.810</td><td>0.707</td><td>0.733</td><td>0.733</td><td>0.773</td></tr><tr><td>OmniVerifier 7B (Joint)</td><td>0.810</td><td>0.897</td><td>0.765</td><td>0.760</td><td>0.773</td><td>0.733</td><td>0.758</td><td>0.744</td><td>0.780</td></tr><tr><td>OmniVerifier 7B (Decoupled)</td><td>0.837</td><td>0.913</td><td>0.783</td><td>0.737</td><td>0.776</td><td>0.753</td><td>0.766</td><td>0.763</td><td>0.791</td></tr><tr><td>Qwen 3-VL 8B</td><td>0.860</td><td>0.883</td><td>0.820</td><td>0.747</td><td>0.867</td><td>0.743</td><td>0.837</td><td>0.893</td><td>0.831</td></tr><tr><td>Qwen 3-VL 8B (Joint)</td><td>0.887</td><td>0.910</td><td>0.834</td><td>0.763</td><td>0.873</td><td>0.753</td><td>0.855</td><td>0.901</td><td>0.847</td></tr><tr><td>Qwen 3-VL 8B (Decoupled)</td><td>0.898</td><td>0.917</td><td>0.855</td><td>0.770</td><td>0.910</td><td>0.777</td><td>0.870</td><td>0.931</td><td>0.866</td></tr></table>

Then, the SNR of the meta-verification gradient under joint training satisfies:

$$
\operatorname{SNR} \left(\mathcal {G} _ {\text { joint }}\right) \leq p _ {\text { acc }} (\theta) \operatorname{SNR} \left(\mathcal {G} _ {\text { dec }}\right), \tag {14}
$$

with strict inequality whenever $p _ { a c c } ( \theta ) \in ( 0 , 1 )$ .

Theorem 5.3 shows that in joint RL training, the meta-verification gradient is effectively multiplied by a Bernoulli variable controlled by $p _ { \mathrm { a c c } } ( \theta )$ , which both suppresses the expected gradient magnitude and introduces an additional variance term. Corollary 5.4 further indicates this gating directly reduces the signalto-noise ratio of the meta-verification gradient by a factor of $p _ { \mathrm { a c c } } ( \theta )$ . Consequently, when the verifier’s judgment accuracy is imperfect, joint training yields sparse and noisy learning signals for meta-verification, whereas decoupled training removes this dependency and provides a more stable optimization signal.

Experimental Setup. Following the same experimental setting as in Section 4, we evaluate the effectiveness of decoupled training in comparison with joint training. We decouple binary judgment and meta-verification into two independent learning objectives, each supervised by a dedicated reward model. Specifically, all samples with y = False are duplicated and treated as grounding-only data, supervised exclusively by the meta-verification reward (e.g., IoU), while the remaining samples are supervised solely by the accuracy reward. These two data streams are jointly mixed during reinforcement learning. We evaluate all models on ViVerBench (Zhang et al., 2025) and RefCOCO (Yu et al., 2016), which respectively measure visual outcome judgment and visual grounding capability.

Experimental Analysis. From Fig. 3, we observe that decoupled training consistently outperforms joint training on both OmniVerifier-7B and Qwen3-VL-8B. In particular, on ViVerBench tasks that are closely related to visual grounding, such as Bounding Box , Counting, and Pointing, decoupled training yields substantially better performance. This improvement can be attributed to the more stable meta-verification gradients provided by decoupling, which enable the verifier to learn more precise and reliable groundingoriented visual judgments. As further evidenced in Table 3, models trained with the decoupled strategy also exhibit clear advantages on RefCOCO, demonstrating stronger visual grounding capability. These results indicate that the error localization ability learned by visual verifiers can effectively generalize to generic grounding tasks, rather than being confined to verifier-specific supervision. These findings suggest that, for visual verifier training, decoupled optimization constitutes a more robust and effective meta-verification reinforcement learning strategy than joint training, due to its ability to disentangle heterogeneous learning objectives and stabilize the training dynamics.

Finding 2. Decoupling RL rewards for binary judgment and meta-verification outperforms joint training.

![](images/45c42cbe5bea787a4dca236cf2563fda163bb13a012613d6488b885ba322aae8.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Verifier Agent"] --> B["OmniVerifier-M1"]
    B --> C["Spatial Action (Symbolic Bbox)"]
    B --> D["Semantic Action (Textual Edit Instruction)"]
    B --> E["Edit Instruction: Add a red backpack inside the box."]
    C --> F["Final Image"]
    D --> G["False"]
    E --> H["Multi-Turn Iterative Loop"]
    H --> I["Agentic Calling"]
    I --> J["Unified Multimodal Model"]
    J --> K["Grounding Edit"]
    L["Prompt: A person with red backpack sitting on a bench"] --> B
    M["Image Symbolic Bbox Edit Instruction"] --> J
```
</details>

Figure 4: Pipeline of M1-TTS: a fine-grained agentic generation system for unified multimodal models.

# 6 Multimodal Verifier for Agentic Generation

# 6.1 OmniVerifier-M1: Generalist Multimodal Verifier

Based on the above two experimental findings, we train OmniVerifier-M1, a generalist multimodal verifier built on Qwen3-VL-8B (Bai et al., 2025a), which uses symbolic bounding boxes as output rationales and leverages rule-based meta-verification reward feedback through decoupled reinforcement training.

As shown in the last row of Table 2, OmniVerifier achieves a score of 0.68 on ViVerBench (Zhang et al., 2025), with notable gains on key text-to-image verification tasks such as Object, Attribute, Spatial Relationship, and Bounding Box. This approach also significantly reduces training overhead and demonstrates the potential of a robust reinforcement learning paradigm for integrating meta-verification into verifier training as a generalizable framework.

# 6.2 M1-TTS: Fine-Grained Agentic Generation

OmniVerifier-M1 provides fine-grained, actionable feedback that precisely localizes erroneous regions in images, rather than offering only coarse, text-level explanations from a global perspective. Building on this capability, we design M1-TTS, an agentic generation system that leverages a visual verifier and a unified multimodal model (UMM) to perform fine-grained, precise, and high-difficulty image world modeling tasks. As shown in Fig. 4, M1-TTS consists of two components: a Verifier Agent and a UMM Agent.

Verifier Agent. The Verifier Agent serves as both the evaluator and diagnostician of input images. Given the current image whether newly generated by the UMM or edited from the previous iteration, if it is not aligned with the input prompt, OmniVerifier-M1 produces a structured action composed of two parts:

• Spatial Action. OmniVerifier-M1 generates symbolic bounding boxes to precisely localize erroneous regions in the image. By explicitly identifying where corrections are needed, this spatial signal directly simplifies the UMM’s perception and reasoning burden, allowing a smooth and effective transition from error diagnosis to targeted image editing.   
• Semantic Action. In addition to erroneous localization, OmniVerifier-M1 provides explicit semantic editing instructions. We predefine a set of atomic editing actions such as Add, Delete, and Modify, and require OmniVerifier-M1 to compose accurate, actionable edit commands grounded in these atomic operations. This design enforces structured, interpretable refinement steps

UMM Agent. The Unified Multimodal Model (UMM) performs image editing by taking as input the image, symbolic bboxes, and explicit editing instructions. The use of symbolic bboxes eliminates the need for the

![](images/3339af4be31f224ec2562314f2b248a1e27cbbbe1297532ef606dcfb9329f6a9.jpg)

<details>
<summary>text_image</summary>

Prompt
A fast-paced racquet sport that is extremely popular in Indonesia, requiring agility and precision
Explanation
This refers to badminton, so the image should include a badminton shuttlecock and a racquet

Edit Prompt 1
Modify the tennis ball by replacing it with a badminton shuttlecock, with the shuttlecock's impact point positioned closer to the racket.

Edit Prompt 2
Modify the tennis racket by replacing it with a badminton racket.

A red-haired princess wearing a black dress stands, raising both hands in front of her chest with open palms. Beshe her, a white-haired princess in a red dress makes a clear "three" gesture with her right hand.

Modify the right-hand gesture of the girl wearing the red dress, changing it from a "two" gesture to a clear "three" gesture.

Modify the left hand of the girl wearing the black dress, changing it from four fingers to five fingers.

A laboratory table displaying one plaster model each of a cylinder, sphere, cone, cube, and rectangular prism, with the cylinder being the tallest object and the sphere being the shortest.

Delete the larger of the two spherical plaster models, keeping the smaller sphere.

Modify the spherical plaster model by reducing its size so that it becomes shorter than the rectangular prism.

Prompt
Four playing cards are arranged in sequence, overlapping by about a quarter, from left to right: Ace of Hearts, Ace of Diamonds, Ace of Clubs, and Ace of Spades.

Edit Prompt 1
Modify the bottom-right suit symbol on the Ace of Spades, changing the club symbol to a spade.

Prompt
A clock with black hour and minute hands and a red second hand, showing the time 10:12.

Edit Prompt 1
Delete the black hand pointing toward 12.
</details>

Figure 5: Qualitative visualization of M1-TTS.

Table 4: Performance comparison on WISE and T2I-CoreBench. 

<table><tr><td rowspan="2">Model</td><td colspan="7">WISE</td><td colspan="3">T2I-CoreBench</td></tr><tr><td>Cultural</td><td>Time</td><td>Space</td><td>Biology</td><td>Physics</td><td>Chemistry</td><td>Overall</td><td>Composition</td><td>Reasoning</td><td>Overall</td></tr><tr><td>Qwen-Image</td><td>0.62</td><td>0.63</td><td>0.77</td><td>0.57</td><td>0.75</td><td>0.40</td><td>0.62</td><td>0.780</td><td>0.493</td><td>0.589</td></tr><tr><td>Qwen3-VL 8B+RePlan</td><td>0.66</td><td>0.64</td><td>0.77</td><td>0.56</td><td>0.72</td><td>0.47</td><td>0.65</td><td>0.793</td><td>0.547</td><td>0.629</td></tr><tr><td>OmniVerifier-M1+RePlan</td><td>0.71</td><td>0.61</td><td>0.77</td><td>0.56</td><td>0.74</td><td>0.57</td><td>0.68</td><td>0.817</td><td>0.626</td><td>0.690</td></tr><tr><td>GPT-Image-1.5</td><td>0.89</td><td>0.69</td><td>0.88</td><td>0.80</td><td>0.76</td><td>0.78</td><td>0.83</td><td>0.855</td><td>0.746</td><td>0.782</td></tr><tr><td>Qwen3-VL 8B+GPT-Image-1.5</td><td>0.88</td><td>0.80</td><td>0.91</td><td>0.89</td><td>0.85</td><td>0.81</td><td>0.86</td><td>0.857</td><td>0.752</td><td>0.787</td></tr><tr><td>OmniVerifier-M1+GPT-Image-1.5</td><td>0.90</td><td>0.80</td><td>0.92</td><td>0.91</td><td>0.88</td><td>0.83</td><td>0.88</td><td>0.863</td><td>0.769</td><td>0.800</td></tr></table>

UMM to first fully parse complex editing instructions and then infer the corresponding spatial regions during editing, thereby significantly simplifying the editing process and improving editing precision.

M1-TTS supports dynamic multi-round image optimization, iteratively refining the image until the verifier outputs True or a maximum number of iterations is reached. Within M1-TTS, the verifier not only injects prior world knowledge into the UMM’s self-refinement process through its strong reasoning and judgment capabilities, but also compensates for the UMM’s limitations in visual perception and spatial localization. As a result, the verifier serves as a critical supervisory and guiding component that enables accurate, fine-grained, and iterative image refinement.

Experimental Setup. We conduct M1-TTS experiments on two strong generative models, RePlan (Wu et al., 2025a; Qu et al., 2025) and GPT-Image-1.5, with the maximum number of iterative steps set to 10. We evaluate M1-TTS on WISE (Niu et al., 2025) and T2I-CoreBench (Li et al., 2025) to assess its capabilities in world-knowledge–driven generation and complex image generation, respectively. All experiments are conducted on 8 NVIDIA A800 GPUs

Experimental Analysis. As shown in Fig. 5, M1-TTS perform dynamic image generation both accurately and efficiently. For regions that are misaligned with the prompt or contain severe errors, OmniVerifier-M1 provides precise guidance through bounding boxes and explanatory signals. This is especially important for complex images with objects that share similar attributes, where bounding boxes more effectively highlight issues. Furthermore, as reported in Table 4, M1-TTS achieves substantial improvements on world knowledge (WISE) and complex text-to-image benchmarks (T2I-CoreBench), whether using RePlan or GPT-Image-1.5.

These gains stem from OmniVerifier-M1’s ability to inject world knowledge while leveraging its interactive bbox outputs to guide the generative model in refining the image.

# 7 Conclusion

In this work, we present Multimodal Meta-Verification, a framework that extends verifier training beyond binary judgments by leveraging symbolic localization feedback. We identify two key findings: (1) Symbolic outputs provide structured and efficient rationales that outperform textual explanations, enabling rulebased reinforcement learning while mitigating reward hacking; (2) Decoupled RL objectives for binary judgment and meta-verification facilitates more robust and efficient optimization, yielding consistently higher verification accuracy than joint training. Building on these insights, we develop a generalist multimodal verifier OmniVerifier-M1 and further introduce M1-TTS, a verifier-driven agentic generation system that performs region-level self-correction through iterative reasoning and action. We provide a robust training paradigm for multimodal meta-verification and leave the exploration of broader applications of verifiers to future work.

# Impact Statements

This paper introduces OmniVerifier-M1 and M1-TTS, and presents a robust framework for training multimodal verifiers with meta-verification feedback.

The methods and findings in this work are intended to advance research in efficient and scalable machine learning systems. We do not anticipate immediate negative societal impacts beyond those commonly associated with deploying more capable and efficient language model systems.

# References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a.   
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.   
Siyu Cao, Hangting Chen, Peng Chen, Yiji Cheng, Yutao Cui, Xinchi Deng, Ying Dong, Kipper Gong, Tianpeng Gu, Xiusen Gu, et al. Hunyuanimage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.   
Dongping Chen, Ruoxi Chen, Shilin Zhang, Yaochen Wang, Yinuo Liu, Huichi Zhou, Qihui Zhang, Yao Wan, Pan Zhou, and Lichao Sun. Mllm-as-a-judge: Assessing multimodal llm-as-a-judge with vision-language benchmark. In Forty-first International Conference on Machine Learning, 2024.   
Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, and Bingsheng He. Judgelrm: Large reasoning models as a judge. arXiv preprint arXiv:2504.00050, 2025.   
Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.   
Yufeng Cui, Honghao Chen, Haoge Deng, Xu Huang, Xinghang Li, Jirong Liu, Yang Liu, Zhuoyan Luo, Jinsheng Wang, Wenxuan Wang, et al. Emu3. 5: Native multimodal models are world learners. arXiv preprint arXiv:2510.26583, 2025.   
Chaorui Deng, Deyao Zhu, Kunchang Li, Chenhui Gou, Feng Li, Zeyu Wang, Shu Zhong, Weihao Yu, Xiaonan Nie, Ziang Song, et al. Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683, 2025.   
Jiawei Gu, Yunzhuo Hao, Huichen Will Wang, Linjie Li, Michael Qizhe Shieh, Yejin Choi, Ranjay Krishna, and Yu Cheng. Thinkmorph: Emergent properties in multimodal interleaved chain-of-thought reasoning. arXiv preprint arXiv:2510.27492, 2025.   
Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025a.   
Dong Guo, Faming Wu, Feida Zhu, Fuxing Leng, Guang Shi, Haobin Chen, Haoqi Fan, Jian Wang, Jianyu Jiang, Jiawei Wang, et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025b.   
Yushi Hu, Reyhane Askari-Hemmat, Melissa Hall, Emily Dinan, Luke Zettlemoyer, and Marjan Ghazvininejad. Multimodal rewardbench 2: Evaluating omni reward models for interleaved text and image. arXiv preprint arXiv:2512.16899, 2025.   
Wenxuan Huang, Shuang Chen, Zheyong Xie, Shaosheng Cao, Shixiang Tang, Yufan Shen, Qingyu Yin, Wenbo Hu, Xiaoman Wang, Yuntian Tang, et al. Interleaving reasoning for better text-to-image generation. arXiv preprint arXiv:2509.06945, 2025a.   
Yuzhen Huang, Weihao Zeng, Xingshan Zeng, Qi Zhu, and Junxian He. From accuracy to robustness: A study of rule- and model-based verifiers in mathematical reasoning, 2025b. URL https://arxiv.org/ abs/2505.22203.   
Shantanu Jaiswal, Mihir Prabhudesai, Nikash Bhardwaj, Zheyang Qin, Amir Zadeh, Chuan Li, Katerina Fragkiadaki, and Deepak Pathak. Iterative refinement improves compositional image generation. arXiv preprint arXiv:2601.15286, 2026.

Dongzhi Jiang, Renrui Zhang, Haodong Li, Zhuofan Zong, Ziyu Guo, Jun He, Claire Guo, Junyan Ye, Rongyao Fang, Weijia Li, et al. Draco: Draft as cot for text-to-image preview and rare concept generation. arXiv preprint arXiv:2512.05112, 2025.   
Ouxiang Li, Yuan Wang, Xinting Hu, Huijuan Huang, Rui Chen, Jiarong Ou, Xin Tao, Pengfei Wan, Xiaojuan Qi, and Fuli Feng. Easier painting than thinking: Can text-to-image models set the stage, but not direct the play? arXiv preprint arXiv:2509.03516, 2025.   
Chao Liao, Liyang Liu, Xun Wang, Zhengxiong Luo, Xinyu Zhang, Wenliang Zhao, Jie Wu, Liang Li, Zhi Tian, and Weilin Huang. Mogao: An omni foundation model for interleaved multi-modal generation. arXiv preprint arXiv:2505.05472, 2025.   
Zijun Liu, Peiyi Wang, Runxin Xu, Shirong Ma, Chong Ruan, Peng Li, Yang Liu, and Yu Wu. Inference-time scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495, 2025.   
Ruilin Luo, Zhuofan Zheng, Lei Wang, Yifan Wang, Xinzhe Ni, Zicheng Lin, Songtao Jiang, Yiyao Yu, Chufan Shi, Ruihang Chu, et al. Unlocking multimodal mathematical reasoning via process reward model. Advances in Neural Information Processing Systems, 38:49851–49899, 2025.   
Ruilin Luo, Chufan Shi, Yizhen Zhang, Cheng Yang, Songtao Jiang, Tongkun Guan, Ruizhe Chen, Ruihang Chu, Peng Wang, Mingkun Yang, et al. From narrow to panoramic vision: Attention-guided cold-start reshapes multimodal reasoning. In International Conference on Learning Representations, 2026.   
Yuwei Niu, Munan Ning, Mengren Zheng, Weiyang Jin, Bin Lin, Peng Jin, Jiaqi Liao, Chaoran Feng, Kunpeng Ning, Bin Zhu, et al. Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265, 2025.   
OpenAI. Thinking with images. https://openai.com/index/thinking-with-images/, 2025.   
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730–27744, 2022.   
Luozheng Qin, Jia Gong, Yuqing Sun, Tianjiao Li, Mengping Yang, Xiaomeng Yang, Chao Qu, Zhiyu Tan, and Hao Li. Uni-cot: Towards unified chain-of-thought reasoning across text and vision. arXiv preprint arXiv:2508.05606, 2025.   
Tianyuan Qu, Lei Ke, Xiaohang Zhan, Longxiang Tang, Yuqi Liu, Bohao Peng, Bei Yu, Dong Yu, and Jiaya Jia. Replan: Reasoning-guided region planning for complex instruction-based image editing. arXiv preprint arXiv:2512.16864, 2025.   
Bytedance Seed. Seed1. 8 model card: Towards generalized real-world agency. arXiv preprint arXiv:2603.20633, 2026.   
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.   
Zhihong Shao, Yuxiang Luo, Chengda Lu, ZZ Ren, Jiewen Hu, Tian Ye, Zhibin Gou, Shirong Ma, and Xiaokang Zhang. Deepseekmath-v2: Towards self-verifiable mathematical reasoning. arXiv preprint arXiv:2511.22570, 2025.   
Yibin Wang, Yuhang Zang, Hao Li, Cheng Jin, and Jiaqi Wang. Unified reward model for multimodal understanding and generation. arXiv preprint arXiv:2503.05236, 2025a.   
Zhaokai Wang, Penghao Yin, Xiangyu Zhao, Changyao Tian, Yu Qiao, Wenhai Wang, Jifeng Dai, and Gen Luo. Genexam: A multidisciplinary text-to-image exam. arXiv preprint arXiv:2509.14232, 2025b.   
Zongqi Wang, Rui Wang, Yuchuan Wu, Yiyao Yu, Pinyi Zhang, Shaoning Sun, Yujiu Yang, and Yongbin Li. Reward modeling from natural language human feedback. arXiv preprint arXiv:2601.07349, 2026.

Chenxi Whitehouse, Tianlu Wang, Ping Yu, Xian Li, Jason Weston, Ilia Kulikov, and Swarnadeep Saha. J1: Incentivizing thinking in llm-as-a-judge via reinforcement learning. arXiv preprint arXiv:2505.10320, 2025.   
Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Sheng-ming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-image technical report. arXiv preprint arXiv:2508.02324, 2025a.   
Jie Wu, Yu Gao, Zilyu Ye, Ming Li, Liang Li, Hanzhong Guo, Jie Liu, Zeyue Xue, Xiaoxia Hou, Wei Liu, et al. Rewarddance: Reward scaling in visual generation. arXiv preprint arXiv:2509.08826, 2025b.   
Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. Imagereward: Learning and evaluating human preferences for text-to-image generation. Advances in Neural Information Processing Systems, 36:15903–15935, 2023.   
Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, et al. Visionreward: Fine-grained multi-dimensional human preference learning for image and video generation. arXiv preprint arXiv:2412.21059, 2024.   
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.   
Cheng Yang, Chufan Shi, Yaxin Liu, Bo Shui, Junjie Wang, Mohan Jing, Linran Xu, Xinyu Zhu, Siheng Li, Yuxiang Zhang, et al. Chartmimic: Evaluating lmm’s cross-modal reasoning capability via chart-to-code generation. In International Conference on Learning Representations, volume 2025, pp. 26590–26646, 2025b.   
Cheng Yang, Chufan Shi, Bo Shui, Yaokang Wu, Muzi Tao, Huijuan Wang, Ivan Yee Lee, Yong Liu, Xuezhe Ma, and Taylor Berg-Kirkpatrick. Ureason: Benchmarking the reasoning paradox in unified multimodal models. arXiv preprint arXiv:2602.08336, 2026a.   
Ling Yang, Zhaochen Yu, Chenlin Meng, Minkai Xu, Stefano Ermon, and Bin Cui. Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms. In Forty-first International Conference on Machine Learning, 2024.   
Ling Yang, Xinchen Zhang, Ye Tian, Shiyi Zhang, Chenming Shang, Minghao Xu, Wentao Zhang, and Bin Cui. Hermesflow: Seamlessly closing the gap in multimodal understanding and generation. Advances in Neural Information Processing Systems, 38:62248–62272, 2026b.   
Licheng Yu, Patrick Poirson, Shan Yang, Alexander C Berg, and Tamara L Berg. Modeling context in referring expressions. In European conference on computer vision, pp. 69–85. Springer, 2016.   
Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Yu Yue, Weinan Dai, Tiantian Fan, Gaohong Liu, Lingjun Liu, et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.   
Lunjun Zhang, Arian Hosseini, Hritik Bansal, Mehran Kazemi, Aviral Kumar, and Rishabh Agarwal. Generative verifiers: Reward modeling as next-token prediction. arXiv preprint arXiv:2408.15240, 2024a.   
Xinchen Zhang, Ling Yang, Yaqi Cai, Zhaochen Yu, Kai-Ni Wang, Ye Tian, Minkai Xu, Yong Tang, Yujiu Yang, Bin Cui, et al. Realcompo: Balancing realism and compositionality improves text-to-image diffusion models. Advances in Neural Information Processing Systems, 37:96963–96992, 2024b.   
Xinchen Zhang, Ling Yang, Guohao Li, Yaqi Cai, Jiake Xie, Yong Tang, Yujiu Yang, Mengdi Wang, and Bin Cui. Itercomp: Iterative composition-aware feedback learning from model gallery for text-to-image generation. arXiv preprint arXiv:2410.07171, 2024c.   
Xinchen Zhang, Xiaoying Zhang, Youbin Wu, Yanbin Cao, Renrui Zhang, Ruihang Chu, Ling Yang, and Yujiu Yang. Generative universal verifier as multimodal meta-reasoner. arXiv preprint arXiv:2510.13804, 2025.

Ziwei Zheng, Michael Yang, Jack Hong, Chenxiao Zhao, Guohai Xu, Le Yang, Chao Shen, and Xing Yu. Deepeyes: Incentivizing" thinking with images" via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.   
Lianghui Zhu, Xinggang Wang, and Xinlong Wang. Judgelm: Fine-tuned large language models are scalable judges. arXiv preprint arXiv:2310.17631, 2023.   
Le Zhuo, Liangbing Zhao, Sayak Paul, Yue Liao, Renrui Zhang, Yi Xin, Peng Gao, Mohamed Elhoseiny, and Hongsheng Li. From reflection to perfection: Scaling inference-time optimization for text-to-image diffusion models via reflection tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15329–15339, 2025.

# A Theoretical Proof

# A.1 Proof of Lemma 5.1

Proof. The expected objective for joint training is defined as:

$$
J _ {\text { joint }} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}} \mathbb {E} _ {(o, \hat {y}, e) \sim \pi_ {\theta} (\cdot | x)} \left[ \mathcal {R} _ {\mathrm{acc}} (\hat {y}, y) \cdot \mathcal {R} _ {\mathrm{meta}} (e) \right], \tag {15}
$$

The policy gradient of the joint objective can be written as:

$$
\nabla_ {\theta} J _ {\text { joint }} = \mathbb {E} \left[ \mathcal {R} _ {\mathrm{acc}} (\hat {y}, y) \cdot \mathcal {R} _ {\mathrm{meta}} (e) \cdot \nabla_ {\theta} \log \pi_ {\theta} (o, \hat {y}, e \mid x) \right]. \tag {16}
$$

Since the prediction yˆ and the meta-verification output e are jointly generated by the same policy, the joint log-probability can be factorized as:

$$
\log \pi_ {\theta} (o, \hat {y}, e \mid x) = \log \pi_ {\theta} (\hat {y} \mid x) + \log \pi_ {\theta} (e \mid x, \hat {y}). \tag {17}
$$

Substituting this decomposition into Eq. 16, we obtain:

$$
\begin{array}{l} \nabla_ {\theta} J _ {\text { joint }} = \mathbb {E} \left[ \mathcal {R} _ {\text { acc }} (\hat {y}, y) \cdot \mathcal {R} _ {\text { meta }} (e) \cdot \nabla_ {\theta} \left(\log \pi_ {\theta} (\hat {y} \mid x) + \log \pi_ {\theta} (e \mid x, \hat {y})\right) \right] (18) \\ = \mathbb {E} \left[ \mathcal {R} _ {\text {acc}} (\hat {y}, y) \cdot \mathcal {R} _ {\text {meta}} (e) \cdot \nabla_ {\theta} \log \pi_ {\theta} (\hat {y} \mid x) \right] + \mathbb {E} \left[ \mathcal {R} _ {\text {acc}} (\hat {y}, y) \cdot \mathcal {R} _ {\text {meta}} (e) \cdot \nabla_ {\theta} \log \pi_ {\theta} (e \mid x, \hat {y}) \right]. (19) \\ \end{array}
$$

The gradient terms related to explanations:

$$
\nabla_ {\theta} J _ {\text { joint }} ^ {(e)} = \mathbb {E} _ {x \sim \mathcal {D}, (\hat {y}, e) \sim \pi_ {\theta}} \left[ \mathcal {R} _ {\text { acc }} (\hat {y}, y) \cdot \mathcal {R} _ {\text { meta }} (e) \nabla_ {\theta} \log \pi_ {\theta} (e \mid x, \hat {y}) \right]. \tag {20}
$$

Note that:

$$
\mathcal {R} _ {\mathrm{acc}} (\hat {y}, y) = \mathbf {1} [ \hat {y} = y ]. \tag {21}
$$

Therefore, when the verifier makes an incorrect prediction $( \mathrm { i } . \mathrm { e } . , \hat { y } \neq y )$ :

$$
\mathcal {R} _ {\mathrm{acc}} (\hat {y}, y) = 0 \quad \Rightarrow \quad \nabla_ {\theta} J _ {\text { joint }} ^ {(e)} = 0. \tag {22}
$$

![](images/f1e22e3778eecc21430d8f8a35b604069e9b5421aa4546dbfb94166dbbd0c7e9.jpg)

# A.2 Proof of Theorme 5.2

Proof. According to the definition of the joint training objective, the gradient term $\nabla _ { \boldsymbol { \theta } } J _ { \mathrm { j o i n t } } ^ { ( e ) }$ with respect to the explanation generation component e can be expressed as:

$$
\nabla_ {\theta} J _ {\text { joint }} ^ {(e)} = \mathbb {E} _ {x \sim \mathcal {D}, (\hat {y}, e) \sim \pi_ {\theta}} \left[ \mathcal {R} _ {\text { acc }} (\hat {y}, y) \cdot \mathcal {R} _ {\text { meta }} (e) \nabla_ {\theta} \log \pi_ {\theta} (e \mid x, \hat {y}) \right]. \tag {23}
$$

Considering that the accuracy reward $\mathcal { R } _ { \mathrm { a c c } } ( \hat { y } , y )$ is an indicator function $\mathbf { 1 } [ \hat { y } = y ]$ , we apply Jensen’s inequality to the $\ell _ { 2 } { \mathrm { - n o r m } }$ of the above gradient, and the derivation proceeds as follows:

$$
\left\| \nabla_ {\theta} J _ {\text { joint }} ^ {(e)} \right\| = \left\| \mathbb {E} _ {x \sim \mathcal {D}} \left[ \mathbb {E} _ {(\hat {y}, e) \sim \pi_ {\theta}} [ \mathbf {1} [ \hat {y} = y ] \cdot \mathcal {R} _ {\text { meta }} (e) \nabla_ {\theta} \log \pi_ {\theta} (e \mid x, \hat {y}) ] \right] \right\| \tag {24}
$$

$$
\leq \mathbb {E} _ {x \sim \mathcal {D}} \left[ \mathbb {E} _ {(\hat {y}, e) \sim \pi_ {\theta}} [ \| \mathbf {1} [ \hat {y} = y ] \cdot \mathcal {R} _ {\text { meta }} (e) \nabla_ {\theta} \log \pi_ {\theta} (e \mid x, \hat {y}) \| ] \right] \tag {25}
$$

$$
= \mathbb {E} _ {x \sim \mathcal {D}} \left[ \mathbb {E} _ {\hat {y} \sim \pi_ {\theta} (\cdot | x)} \left[ \mathbb {E} _ {e \sim \pi_ {\theta} (\cdot | x, \hat {y})} [ \mathbf {1} [ \hat {y} = y ] \cdot \| \mathcal {R} _ {\text { meta }} (e) \nabla_ {\theta} \log \pi_ {\theta} (e | x, \hat {y}) \| ] \right] \right]. \tag {26}
$$

Using the property of conditional expectation, when $\hat { y } \neq y .$ the indicator function $\mathbf { 1 } [ \hat { y } = y ] = 0$ , so only the case where $\hat { y } = y$ needs to be retained in the expectation term:

$$
\left\| \nabla_ {\theta} J _ {\text { joint }} ^ {(e)} \right\| \leq \mathbb {E} _ {x \sim \mathcal {D}} \left[ \mathbb {P} _ {\pi_ {\theta}} (\hat {y} = y \mid x) \cdot \mathbb {E} _ {e \sim \pi_ {\theta} (\cdot | x, y)} \left[ \| \mathcal {R} _ {\text { meta }} (e) \nabla_ {\theta} \log \pi_ {\theta} (e \mid x, y) \| \right] \right] \tag {27}
$$

$$
\leq \mathbb {E} _ {x \sim \mathcal {D}} \left[ \mathbb {P} _ {\pi_ {\theta}} (\hat {y} = y \mid x) \right] \cdot \sup _ {x} \mathbb {E} _ {e \sim \pi_ {\theta}} \left[ \| \mathcal {R} _ {\text { meta }} (e) \nabla_ {\theta} \log \pi_ {\theta} (e \mid x, y) \| \right] \tag {28}
$$

$$
= p _ {\mathrm{acc}} (\theta) \cdot C, \tag {29}
$$

where $\begin{array} { r } { C = \operatorname* { s u p } _ { x } \mathbb { E } _ { e } \left[ \Vert \mathcal { R } _ { \mathrm { m e t a } } ( e ) \nabla _ { \theta } \log \pi _ { \theta } ( e \mid x , y ) \Vert \right] } \end{array}$ denotes the finite upper bound of the gradient term. Based on the definition of the verification accuracy $\mathsf { \bar { p } } _ { \mathrm { a c c } } ( \theta ) = \mathbb { E } _ { x \sim \mathcal { D } } [ \mathbb { P } _ { \pi _ { \theta } } ( \bar { y } = y \mid x ) ]$ ], the proof is complete.

![](images/a54204a045a59fea3d14f48e6e9d2c347acc3d7d2bf579aea63786c936ed23fd.jpg)

# A.3 Proof of Throrme 5.3

Proof. Let $I = \mathcal { R } _ { \mathrm { a c c } } ( x , \hat { y } ) = \mathbf { 1 } [ \hat { y } = y ]$ be the indicator variable representing the accuracy of the verifier. By definition, I follows a Bernoulli distribution with parameter $p _ { \mathrm { a c c } } ( \theta ) = \mathbb { P } ( \hat { y } = y )$ , such that $\mathbb { E } [ I ] = p _ { \mathrm { a c c } } ( \boldsymbol { \dot { \theta } } )$ and $I ^ { 2 } = I$ . The joint gradient estimator can be written as $\mathcal { G } _ { \mathrm { j o i n t } } = I \cdot \mathcal { G } _ { \mathrm { d e c } } .$ .

First, we calculate the first and second moments of $\lvert \mathcal { G } _ { \mathrm { j o i n t } }$ :

$$
\mathbb {E} [ \mathcal {G} _ {\text { joint }} ] = \mathbb {E} [ I \cdot \mathcal {G} _ {\text { dec }} ] = p _ {\text { acc }} (\theta) \mathbb {E} [ \mathcal {G} _ {\text { dec }} ], \tag {30}
$$

$$
\mathbb {E} [ \| \mathcal {G} _ {\text { joint }} \| ^ {2} ] = \mathbb {E} [ \| I \cdot \mathcal {G} _ {\text { dec }} \| ^ {2} ] = \mathbb {E} [ I ^ {2} \cdot \| \mathcal {G} _ {\text { dec }} \| ^ {2} ] = p _ {\text { acc }} (\theta) \mathbb {E} [ \| \mathcal {G} _ {\text { dec }} \| ^ {2} ]. \tag {31}
$$

Using the variance identity $\operatorname { V a r } ( Z ) = \mathbb { E } [ \| Z \| ^ { 2 } ] - \| \mathbb { E } [ Z ] \| ^ { 2 }$ for a vector-valued random variable $Z ,$ we have:

$$
\operatorname{Var} \left(\mathcal {G} _ {\text { joint }}\right) = \mathbb {E} \left[ \left\| \mathcal {G} _ {\text { joint }} \right\| ^ {2} \right] - \left\| \mathbb {E} \left[ \mathcal {G} _ {\text { joint }} \right] \right\| ^ {2} \tag {32}
$$

$$
= p _ {\text { acc }} (\theta) \mathbb {E} [ \| \mathcal {G} _ {\text { dec }} \| ^ {2} ] - \| p _ {\text { acc }} (\theta) \mathbb {E} [ \mathcal {G} _ {\text { dec }} ] \| ^ {2} \tag {33}
$$

$$
= p _ {\mathrm{acc}} (\theta) \left(\operatorname{Var} \left(\mathcal {G} _ {\mathrm{dec}}\right) + \| \mathbb {E} \left[ \mathcal {G} _ {\mathrm{dec}} \right] \| ^ {2}\right) - p _ {\mathrm{acc}} (\theta) ^ {2} \| \mathbb {E} \left[ \mathcal {G} _ {\mathrm{dec}} \right] \| ^ {2} \tag {34}
$$

$$
= p _ {\mathrm{acc}} (\theta) \operatorname{Var} \left(\mathcal {G} _ {\mathrm{dec}}\right) + \left(p _ {\mathrm{acc}} (\theta) - p _ {\mathrm{acc}} (\theta) ^ {2}\right) \| \mathbb {E} [ \mathcal {G} _ {\mathrm{dec}} ] \| ^ {2} \tag {35}
$$

$$
= p _ {\mathrm{acc}} (\theta) \operatorname{Var} \left(\mathcal {G} _ {\mathrm{dec}}\right) + p _ {\mathrm{acc}} (\theta) \left(1 - p _ {\mathrm{acc}} (\theta)\right) \| \mathbb {E} \left[ \mathcal {G} _ {\mathrm{dec}} \right] \| ^ {2}. \tag {36}
$$

Since $p _ { \mathrm { a c c } } ( \theta ) \in [ 0 , 1 ]$ , the term $p _ { \mathrm { a c c } } ( \theta ) ( 1 - p _ { \mathrm { a c c } } ( \theta ) ) \lVert \mathbb { E } [ \mathcal { G } _ { \mathrm { d e c } } ] \rVert ^ { 2 }$ is always non-negative. Therefore:

$$
\operatorname{Var} \left(\mathcal {G} _ {\text { joint }}\right) \geq p _ {\text { acc }} (\theta) \operatorname{Var} \left(\mathcal {G} _ {\text { dec }}\right). \tag {37}
$$

The equality holds if and only $\mathrm { i f } p _ { \mathrm { a c c } } ( \theta ) ( 1 - p _ { \mathrm { a c c } } ( \theta ) ) \lVert \mathbb { E } [ \mathcal { G } _ { \mathrm { d e c } } ] \rVert ^ { 2 } = 0$ . Given $p _ { \mathrm { a c c } } ( \theta ) \in ( 0 , 1 )$ and $\mathbb { E } [ \mathcal { G } _ { \mathrm { d e c } } ] \neq 0 .$ , the inequality is strict.

# A.4 Proof of Corollary 5.4

Proof. Based on the results from Theorem 5.3, we have the following expressions for the first moment and the variance of the joint gradient estimator:

$$
\left\| \mathbb {E} \left[ \mathcal {G} _ {\text { joint }} \right] \right\| ^ {2} = p _ {\text { acc }} (\theta) ^ {2} \left\| \mathbb {E} \left[ \mathcal {G} _ {\text { dec }} \right] \right\| ^ {2}, \tag {38}
$$

$$
\operatorname{Var} \left(\mathcal {G} _ {\text { joint }}\right) = p _ {\text { acc }} (\theta) \operatorname{Var} \left(\mathcal {G} _ {\text { dec }}\right) + p _ {\text { acc }} (\theta) \left(1 - p _ {\text { acc }} (\theta)\right) \| \mathbb {E} \left[ \mathcal {G} _ {\text { dec }} \right] \| ^ {2}. \tag {39}
$$

Substituting these into the definition of $\operatorname { S N R } ( \mathcal { G } _ { \mathrm { j o i n t } } )$ , we obtain:

$$
\operatorname{SNR} \left(\mathcal {G} _ {\text { joint }}\right) = \frac {\left\| \mathbb {E} \left[ \mathcal {G} _ {\text { joint }} \right] \right\| ^ {2}}{\operatorname{Var} \left(\mathcal {G} _ {\text { joint }}\right)} \tag {40}
$$

$$
= \frac {p _ {\mathrm{acc}} (\theta) ^ {2} \| \mathbb {E} [ \mathcal {G} _ {\mathrm{dec}} ] \| ^ {2}}{p _ {\mathrm{acc}} (\theta) \operatorname{Var} (\mathcal {G} _ {\mathrm{dec}}) + p _ {\mathrm{acc}} (\theta) (1 - p _ {\mathrm{acc}} (\theta)) \| \mathbb {E} [ \mathcal {G} _ {\mathrm{dec}} ] \| ^ {2}} \tag {41}
$$

$$
= \frac {p _ {\mathrm{acc}} (\theta) \| \mathbb {E} [ \mathcal {G} _ {\mathrm{dec}} ] \| ^ {2}}{\operatorname{Var} (\mathcal {G} _ {\mathrm{dec}}) + (1 - p _ {\mathrm{acc}} (\theta)) \| \mathbb {E} [ \mathcal {G} _ {\mathrm{dec}} ] \| ^ {2}}. \tag {42}
$$

Since $1 - p _ { \mathrm { a c c } } ( \theta ) \geq 0$ , it follows that the denominator satisfies:

$$
\operatorname{Var} \left(\mathcal {G} _ {\text { dec }}\right) + \left(1 - p _ {\text { acc }} (\theta)\right) \| \mathbb {E} \left[ \mathcal {G} _ {\text { dec }} \right] \| ^ {2} \geq \operatorname{Var} \left(\mathcal {G} _ {\text { dec }}\right). \tag {43}
$$

By applying this inequality to the denominator of the SNR expression, we have:

$$
\mathrm{SNR} (\mathcal {G} _ {\text { joint }}) \leq \frac {p _ {\mathrm{acc}} (\theta) \| \mathbb {E} [ \mathcal {G} _ {\mathrm{dec}} ] \| ^ {2}}{\operatorname{Var} (\mathcal {G} _ {\mathrm{dec}})} \tag {44}
$$

$$
= p _ {\mathrm{acc}} (\theta) \cdot \mathrm{SNR} (\mathcal {G} _ {\mathrm{dec}}). \tag {45}
$$

For $p _ { \mathrm { a c c } } ( \theta ) \in ( 0 , 1 )$ , the term $( 1 - p _ { \operatorname { a c c } } ( \theta ) ) \| \mathbb { E } [ \mathcal { G } _ { \operatorname { d e c } } ] \| ^ { 2 }$ is strictly positive (assuming a non-vanishing signal $\lVert \mathbb { E } [ \mathcal { G } _ { \mathrm { d e c } } ] \rVert > 0 )$ , which makes the denominator strictly larger than $\mathrm { V a r } ( \mathcal G _ { \mathrm { d e c } } )$ , thus confirming the strict inequality. □

Table 5: Performance Comparison of Rule-Based Symbolic Rewards and Model-Based Textual Rewards as Meta-Verification Signals on ViVerBench 

<table><tr><td>Backbone</td><td>Reward Metric</td><td>ViVerBench</td></tr><tr><td>OmniVerifier 7B</td><td>-</td><td>0.6501</td></tr><tr><td>OmniVerifier 7B</td><td>textual explanation (model-based)</td><td>0.6617</td></tr><tr><td>OmniVerifier 7B</td><td>symbolic bbox (rule-based)</td><td>0.6613</td></tr><tr><td>OmniVerifier 7B</td><td>symbolic point (rule-based)</td><td>0.6619</td></tr><tr><td>Qwen 3-VL 8B</td><td>-</td><td>0.6539</td></tr><tr><td>Qwen 3-VL 8B</td><td>textual explanation (model-based)</td><td>0.6698</td></tr><tr><td>Qwen 3-VL 8B</td><td>symbolic bbox (rule-based)</td><td>0.6717</td></tr><tr><td>Qwen 3-VL 8B</td><td>symbolic point (rule-based)</td><td>0.6709</td></tr></table>

Table 6: Evaluation of error localization capability on synthetic and real-world data. 

<table><tr><td>Backbone</td><td>Synthetic Data</td><td>Real-World Data</td></tr><tr><td>OmniVerifier 7B</td><td>0.290</td><td>0.265</td></tr><tr><td>OmniVerifier 7B (Joint)</td><td>0.545</td><td>0.495</td></tr><tr><td>OmniVerifier 7B (Decoupled)</td><td>0.710</td><td>0.670</td></tr><tr><td>Qwen 3-VL 8B</td><td>0.375</td><td>0.325</td></tr><tr><td>Qwen 3-VL 8B (Joint)</td><td>0.665</td><td>0.605</td></tr><tr><td>Qwen 3-VL 8B (Decoupled)</td><td>0.780</td><td>0.725</td></tr></table>

# B Additional Experiments

# B.1 Symbolic Point as Meta-Verification Signals

We replace the symbolic bounding box with a symbolic point as the rule-based reward signal. In the bounding-box setting, for each negative sample, we compute the IoU between the predicted and groundtruth boxes and apply a threshold of 0.6 to obtain a binary gated reward, rather than using the continuous IoU value directly.

For consistency, we define the point-based reward in the same binary form. Specifically, if the predicted point falls inside the ground-truth bounding box, the error region is considered correctly localized and a reward of 1 is assigned; otherwise, the reward is set to 0. As shown in Table 5, rule-based symbolic point rewards also serve as an effective alternative to model-based textual explanations for meta-verification under the joint training setting.

# B.2 Evaluation of the Verifier’s Localization Accuracy

To directly evaluate the verifier’s ability to localize errors, we carefully construct a test set of 400 False samples that are not used during training, including 200 synthetic samples and 200 real-world samples. We compute the IoU between predicted and ground-truth bounding boxes and use a threshold of 0.6 consistent with the training setting to determine whether the error is successfully localized.

As shown in Table 6, the significant improvement demonstrates that our decoupled symbolic rule-based RL effectively teaches the verifier to perform precise spatial error localization. Such high-precision localization capability ensures that the UMM agent receives reliable and fine-grained guidance

# B.3 Impact of Batch Size on Decoupled Training versus Joint Training

To further verify whether the observed performance gains come from the training strategy rather than the batch size, we conduct additional experiments on both OmniVerifier and Qwen3-VL-8B. Specifically, (i) we increase the batch size of joint training to 1.5B and compare it with decoupled training under the same batch size; and (ii) we reduce the batch size of decoupled training to B and compare it with joint training under the same setting.

Table 7: Analysis of batch size on performance on ViVerBench and RefCOCO. 

<table><tr><td>Backbone</td><td>Batch Size</td><td>ViVerBench</td><td>RefCOCO</td></tr><tr><td>OmniVerifier 7B</td><td>-</td><td>0.6501</td><td>0.7734</td></tr><tr><td>OmniVerifier 7B (Joint)</td><td>1B</td><td>0.6610</td><td>0.7800</td></tr><tr><td>OmniVerifier 7B (Decoupled)</td><td>1B</td><td>0.6672</td><td>0.7898</td></tr><tr><td>OmniVerifier 7B (Joint)</td><td>1.5B</td><td>0.6617</td><td>0.7813</td></tr><tr><td>OmniVerifier 7B (Decoupled)</td><td>1.5B</td><td>0.6680</td><td>0.7910</td></tr><tr><td>Qwen3-VL 8B</td><td>-</td><td>0.6539</td><td>0.8313</td></tr><tr><td>Qwen3-VL 8B (Joint)</td><td>1B</td><td>0.6710</td><td>0.8470</td></tr><tr><td>Qwen3-VL 8B (Decoupled)</td><td>1B</td><td>0.6792</td><td>0.8642</td></tr><tr><td>Qwen3-VL 8B (Joint)</td><td>1.5B</td><td>0.6708</td><td>0.8473</td></tr><tr><td>Qwen3-VL 8B (Decoupled)</td><td>1.5B</td><td>0.6800</td><td>0.8660</td></tr></table>

As shown in Table 7, decoupled training consistently outperforms joint training under the same batch size. This indicates that the performance gains do not simply come from using a larger batch size, but instead stem from the decoupled optimization strategy itself.

In joint training, although the batch size is B, the same 0.5B negative samples are simultaneously used to optimize both the judgment objective and the grounding objective, meaning that each sample contributes to two different supervision signals. In contrast, in decoupled training, although the total batch size is 1.5B, the same 0.5B negative samples are explicitly separated into objective-specific supervision: one part is used for the judgment objective, while the other is used for the grounding objective. Therefore, decoupled training changes how supervision signals are applied without increasing data diversity.

Joint training suffers from sparse and entangled reward signals, whereas decoupled training reduces interference between optimization objectives and provides denser and more stable learning signals, leading to better performance.

# C Data Construction Pipeline

we further provide a more detailed description of our two automated data construction pipelines.Our training data is constructed through two automated pipelines, ensuring that every false sample is associated with a meaningful and well-defined bounding box. Importantly, our training data is entirely derived from OmniVerifier (Zhang et al., 2025), enabling a fair comparison with the verifier and thereby allowing us to clearly demonstrate the advantages of meta-verification. We construct the dataset using both synthetic data (ShareGPT-4o-Image) and real-world data (LVIS) through two automated methods to obtain both aligned and misaligned image–text pairs.

Method 1: Image-fixed, Prompt-modified. For each complex image, we first use GPT-5 to generate a detailed prompt, which serves as the true prompt. We then modify the prompt using GPT-5 by adding or removing objects, altering attributes, or modifying spatial relationships to construct a mismatched (false) prompt, while GPT-5 simultaneously generates the corresponding ground-truth bounding boxes for the regions associated with these modifications.

Method 2: Prompt-fixed, Image-inpainting. We treat each complex image as the true image and first apply SAM 2.1 to segment it, obtaining masks and bounding boxes for all objects. To balance dataset difficulty, we dynamically select one object based on its mask area. We then perform inpainting using the selected mask to remove the object, thereby constructing a false image. Finally, we use GPT-5 to generate a detailed prompt from the true image, which serves as the fixed prompt. This construction naturally yields accurate and meaningful bounding boxes.

# D Limitations and Future Works

Despite the strong capabilities demonstrated by OmniVerifier-M1 in multimodal verifier training and M1-TTS in dynamic agentic generation, there remain two limitations that need to be addressed:

• The verifier training paradigm proposed in this work requires validation on larger-scale backbone models and backbones with different architectures. Larger models tend to achieve higher binary judgment accuracy during early training, which may slightly mitigate the disadvantages of joint training but far from resolve them. Therefore, in future work, we plan to evaluate our approach on models with larger sizes, as well as on architectures such as MoE.   
• The performance of M1-TTS is still strongly constrained by the editing capability of the underlying unified generative model. Our experiments show that although OmniVerifier-M1 can provide accurate bounding boxes and precise edit instructions, current image editing models are rarely trained to follow region-grounded editing commands. As a result, they may fail to restrict modifications to the specified bounding-box regions and instead introduce unnecessary or even harmful changes in unrelated areas. This highlights an important direction for future research: developing finegrained, region-level image editing models that can faithfully execute localized instructions while preserving the rest of the image. Such grounding-aware editing capability is essential for enabling reliable dynamic image refinement and supporting more general, complex, and interactive generation scenarios.