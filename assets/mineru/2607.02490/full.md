# Visually Grounded Self-Reflection for Vision-Language Models via Reinforcement Learning

Liyan Tang $^{*}$ ♠, Fangcong Yin $^{*}$ ♦, Greg Durrett $^{\diamond}$ $^{\spadesuit}$ The University of Texas at Austin $\diamondsuit$ New York University
lytang@utexas.edu, {fy666, gdurrett}@nyu.edu

## Abstract

Large vision-language models can reason over multimodal inputs by generating textual chains of thought (CoT). A key capability exhibited in CoT reasoning is self-reflection: revisiting earlier decisions and correcting previous errors. However, existing LVLMs often fail to properly attend to visual inputs during reflection, limiting their ability to translate feedback into grounded corrections, especially for out-of-distribution images. To address this issue, we propose a novel reinforcement learning training framework VRRL, with two components explicitly designed to elicit visually grounded self-reflection. First, we randomly mask trajectory prefixes during training to emphasize recovery from incorrect intermediate predictions rather than making early mistakes. Second, we introduce buffered roll-ins from an experience replay buffer to expose the model to diverse failure states that it must learn to correct. We evaluate our approach on visual grounding tasks involving tables and charts, as well as spatial navigation benchmarks. While off-the-shelf and conventionally fine-tuned models degrade substantially under distribution shift, our method substantially improves average out-of-distribution accuracy over standard RL and reflection-oriented fine-tuning baselines by using self-reflection effectively. $^{1}$

## 1 Introduction

Large vision-language models (LVLMs) can solve complex multimodal reasoning tasks over text and images by generating CoT traces (Zhang et al., 2024; Hu et al., 2024). Textual reasoning models use CoT to improve performance through a number of cognitive skills (Gandhi et al., 2025), among them self-reflection. Self-reflection is the act of a model reasoning about the correctness of a candidate answer or solution step, then potentially correcting it or revising it if an error was made (Madaan et al., 2023; Guo et al., 2025). However, this skill remains under-developed in LVLMs. Prior work suggests that a key bottleneck stems from the modality gap (Yi et al., 2024): LVLMs often fail to attend to relevant visual tokens (Ma et al., 2026) and struggle to translate visual evidence into grounded corrective behaviors (Huang et al., 2025; Zhang et al., 2026), particularly for complex or out-of-distribution (OOD) images that differ substantially from pre-training distributions. Achieving visual self-reflection in LVLMs requires additional post-training.

To address this limitation, we first use supervised fine-tuning (SFT) to teach the model the basic structure of multi-turn visual feedback and then propose a novel Visual Reflection RL (VRRL) training recipe explicitly designed to instill reflective error-correction from such feedback. During RL, VRRL combines two complementary strategies that expose the model to diverse error-recovery scenarios. First, for newly generated on-policy trajectories, we introduce Random Turn Masking, which computes policy updates only on randomly selected suffixes of a rollout. This teaches the model to learn to correct errors while masking the steps which may have led to those errors. Second, we introduce a Buffered Roll-In strategy that samples historical “mistake prefixes” from a replay buffer of past failures and asks the model to continue the rollout and correct previous errors. By explicitly training the model to reason over multi-turn visual feedback, these strategies improve both the robustness and OOD generalization of visual reasoning.

We evaluate VRRL on two tasks where a VLM acts in an environment that provides visual feedback: (1) Visual grounding, where the model predicts the coordinates of a queried visual element in an image; and (2) Spatial navigation, where the model solves maze-like navigation tasks from visual inputs. For both tasks, we first establish basic self-reflection capabilities through SFT, and then enhance them using several RL baselines and our proposed method under the same training distribution (Guo et al., 2025; KimiTeam et al., 2025; Zhai et al., 2024; Feizi et al., 2026). We evaluate whether each approach generalizes its reflective correction behavior to OOD settings.

![](images/d917573acf82b36ad798f0a2391bddbd6135cb3f37ce8c0b323a1adfcc83484a.jpg)  
Figure 1: A motivating example of multi-turn reflection for visual grounding. In this illustrative task, the goal is to find the pixel coordinate for the value “11” in a table image. In a single turn, an LVLM can reason over text in the instruction but may fail to provide accurate pixel-level grounding output. However, if given its prediction as visual feedback (a red dot on the image), the LVLM can reflect to iteratively correct its inaccurate grounding and determine when to stop and give an answer.

Our experiments show that off-the-shelf LVLMs struggle to generalize OOD under direct prompting, and that prompt engineering alone fails to elicit meaningful self-reflection, often leading to repetitive behaviors that degrade performance. Explicit training for self-reflection improves OOD generalization, while VRRL further outperforms standard RL and existing reflection-oriented fine-tuning methods by inducing visually grounded error correction more effectively.

The main contributions of this work are as follows: (1) We demonstrate that visually grounded self-reflection is an effective mechanism for improving LVLM robustness under distribution shifts, outperforming non-reflective and weakly grounded reflection baselines. (2) We introduce VRRL, a novel RL framework that combines Random Turn Masking and Buffered Roll-In to train models to recover from diverse intermediate errors, leading to stronger OOD generalization across visual feedback environments.

## 2 Problem Formulation: Multi-turn Inference with Reflection

Reasoning with Multi-turn Inference. We consider a multimodal reasoning task in which a model interacts with an environment E. Given an input image I and a natural language instruction Q, the goal is to answer Q by taking a sequence of actions $a_{t} \in A$ and receiving image observations

$I_{t} \in O$ from the environment. A standard LVLM $\pi_{\theta}(a \mid I, Q)$ typically makes a one-shot decision by producing a single action and immediately terminating with a final answer. While simple, this single-turn setting does not allow the model to verify its prediction or recover from mistakes.

In this work, we formulate multimodal reasoning as a multi-turn sequential decision process. Instead of relying on a single prediction, the model can iteratively propose an action, receive visual feedback from the environment, and decide whether to refine its prediction or terminate. Figure 1 illustrates the difference between single-turn and multi-turn inference. We define this process by the tuple $(\mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{R})$ .

State and Observation. At turn t, the underlying state $s_{t} = (I, Q, \mathcal{H}_{t}) \in \mathcal{S}$ with the interaction history $\mathcal{H}_{t} = \{(a_{0}, I_{0}), \ldots, (a_{t-1}, I_{t-1})\}$ for $a_{t} \in A$ . Upon taking an action $a_{t}$ , the model receives an image $I_{t} \in O$ as the visual observation rendered by the environment, and can reflect on the observation to adjust its future actions. For example, in the visual grounding task in Figure 1, the model predicts a coordinate tuple for the given query. The environment then marks the predicted location on the image, e.g., with a red point, and returns a fixed-size crop centered at the predicted coordinates. This crop serves as the image observation $I_{t}$ for the next turn.

Action Space. An action $a_{t} \in A$ corresponds to a reasoning step followed by an answer proposal or a termination: the policy $\pi_{\theta}(a_{t} \mid s_{t})$ can propose an answer candidate that could be changed in later turns, or terminate the trajectory $\tau$ by finalizing an answer candidate from the last turn that it considers already correct. Upon termination, the final answer will be evaluated. In the visual grounding task, for example, an answer proposal at turn t is a selection of pixel coordinates $p_{t} = (x_{t}, y_{t})$ within the image space, and a final answer needs to be locked in with a termination function call.

![](images/b45bffe331a0785408e42d679484e0bbc48f4963ae8631b7aa881ef27dc46847.jpg)  
Figure 2: (a) Random Turn Masking masks gradient updates on a random number of prefix steps to avoid training on potentially erroneous steps. (b) Buffered Roll-In begins roll-outs from a potentially erroneous step in the replay buffer, enabling correction of this mistake. Note that partial rewards are assigned for correct reflection steps.

Trajectory Reward. The reward function R is only evaluated when the trajectory $\tau$ is terminated at turn t = T. The learning objective is to optimize the policy $\pi_{\theta}$ to maximize the final answer accuracy, enabling the model to refine its predictions iteratively based on visual feedback. Note that R involves multiple components that evaluate different aspects of the entire multi-turn trajectory, including final answer correctness, response format, and task progress at each turn (e.g., the reflection reward in Section 3.2).

## 3 Methods

Training Framework. Our training pipeline follows a two-stage SFT → RL recipe designed to instill and then refine reflective capabilities (Guo et al., 2025; KimiTeam et al., 2025; Zhai et al., 2024; Feizi et al., 2026; Sprague et al., 2026). Our goal is to improve the generalizability of the trained model over OOD tasks.

## 3.1 Stage 1: Supervised Fine-Tuning (SFT)

In the first stage, we first construct an offline dataset $D_{SFT}$ with trajectories containing both immediate correct answers (single-turn) and iterative corrections (multi-turn): single-turn trajectories aim at teaching the model the visual reasoning task, and multi-turn trajectories teach the model the format of reflection based on visual feedback from the environment.

Each instance consists of an image I, an instruction Q, and a trajectory $\tau$ . Single-turn trajectories consist of a single action leading to the correct answer, the visual feedback $I_{0}$ , and a termination action $a_{1}, \tau = \{a_{0}, I_{0}, a_{1}\}$ . A multi-turn trajectory $\tau = \{a_{0}, I_{0}, \ldots, a_{T-1}, I_{T-1}, a_{T}\}$ consists of alternating visual feedback $I_{t}$ and assistant responses $a_{t}$ (CoT and answer candidates). Note that the last action $a_{T}$ is always a termination action.

On multi-turn trajectories, we optimize the standard auto-regressive cross-entropy loss over the assistant turns $a_{1}, \ldots, a_{T}$ . Note that $a_{0}$ is excluded from the multi-turn loss update because $a_{0}$ is constructed to be erroneous so that the model needs to correct in the following assistant turns. This stage establishes the interaction format and initializes the policy with basic error-correction behaviors.

## 3.2 Stage 2: RL with Random Turn Masking and Buffered Roll-In

To explicitly train the model to recover from errors, we do reinforcement learning on a set of examples $D_{RL}$ . Our reward R is as defined below. We employ Group Relative Policy Optimization (GRPO; Shao et al. (2024b)) augmented with two novel mechanisms: Random Turn Masking (RTM) and Buffered Roll-In (see Figure 2).

Reward Function. We combine three reward components: format validity, answer correctness, and reflection shaping. (1) Format Reward: We assign $r_{fmt} = 1$ if the model response utilizes the valid tool-call format, and 0 otherwise. Invalid formats result in a total trajectory reward of 0. (2) Outcome reward: We assign $R_{answer} = 1.0$ if the final answer is correct based on the task-specific metric, and 0 otherwise. (3) Reflection reward: To encourage convergence toward the target through multi-turn inference, we define a reflection reward $R_{refl}$ based on an improvement-based reward function $\phi(\tau)$ . Reflection reward is task-specific, and details for the evaluation tasks can be found in Section 4; ablations in Section 7 show that reflection reward elicits self-reflection better than the outcome reward.

(a) Table Related Tasks

<table><tr><td rowspan="2" colspan="2"></td><td colspan="2">SQuAD2.0</td><td colspan="2">HotpotQA</td></tr><tr><td>EM</td><td>F1</td><td>EM</td><td>F1</td></tr><tr><td rowspan="3">3/4B</td><td>Qwen2.5</td><td>73.8</td><td>77.9</td><td>45.6</td><td>61.8</td></tr><tr><td>Llama3</td><td>71.9</td><td>76.1</td><td>43.8</td><td>59.7</td></tr><tr><td>Qwen3</td><td>75.1</td><td>79.3</td><td>47.2</td><td>63.5</td></tr><tr><td rowspan="3">7/8B</td><td>Qwen2.5</td><td>79.6</td><td>83.4</td><td>51.8</td><td>67.6</td></tr><tr><td>Llama3</td><td>77.8</td><td>81.6</td><td>49.9</td><td>65.4</td></tr><tr><td>Qwen3</td><td>81.3</td><td>85.0</td><td>54.1</td><td>69.8</td></tr></table>

Q: Locate "F1" in the "HotpotQA" section.  
Q: Locate "47.2" in the table.

![](images/5a8fe3e8988ed5e04804070315f7a1a97b5f1b300217c5034b546f2e628faac6.jpg)

![](images/8ce2aac21af7fc256dd179b3c361e374d6d234c8efc7bd9e24f50968eb05fbb6.jpg)  
Q: Please locate the cross representing H.

![](images/83b96249e9aed91cffac17f3650dd77cb5cf057d8be765e8a04aee47983966ad.jpg)  
Q: Find the shortest path from the sprite to the chest without falling into holes.  
Figure 3: Evaluation tasks for visually grounded self-reflection. (1) Visual Grounding. models are trained to localize headers of synthetic small tables. OOD tasks include generalization to larger tables (Large Table) queries of inner cell values rather than headers (Cell Query), and queries about bar charts (b), and scatter plots (c). (2) Spatial Navigation. The model is given a maze-style map and needs to find the shortest path from the start to the goal without running into obstacles. We train models on smaller maps and evaluate on larger maps as OOD tests.

The final scalar reward R for a rollout is:

$$
R = \left\{ \begin{array}{l l} 0 & \text { if } r _ {\mathrm{fmt}} = 0, \\ \max (R _ {\mathrm{answer}}, R _ {\mathrm{refl}}) & \text { if } r _ {\mathrm{fmt}} = 1 \end{array} \right.
$$

For incorrect predictions ( $R_{answer} = 0$ ), we provide partial credits based on improvements.

For a given question $q \in D_{RL}$ , we sample a group of G trajectories $\{\tau_{1}, \ldots, \tau_{G}\}$ from the old policy $\pi_{\theta_{old}}$ and follow the standard GRPO training objective (Appendix A).

Random Turn Masking (RTM). Given a full trajectory rollout $\tau$ of length T, we sample a start index $k \sim \text{Unif}(\{1, \ldots, T\})$ and compute policy-gradient updates only on the suffix from turn k to T, masking the loss for earlier turns. Formally, let $\mathcal{J}(\theta)$ denote the RL objective. Under RTM, the gradient estimate is

$$
\nabla \mathcal {J} _ {\mathrm{RTM}} (\theta) = \mathbb {E} _ {\tau \sim \pi_ {\theta}} \mathbb {E} _ {k \sim U (1, T)} \left[ \sum_ {t = k} ^ {T} \nabla \log \pi_ {\theta} (a _ {t} | s _ {t}) \hat {A} _ {t} \right],
$$

where $\hat{A}_{t}$ is the GRPO advantage estimate. Although prefix turns before k do not directly contribute to the gradient, they determine the conditioning state $s_{k}$ . RTM therefore trains the policy to optimize returns from arbitrary intermediate states, including potentially erroneous ones, without learning to make the mistakes in the erroneous states. We interpret RTM as a form of reweighted per-decision policy gradient; a derivation is provided in Appendix A.

Buffered Roll-In. RTM reweights gradient updates across turns, but it still relies on on-policy rollouts. As the policy improves, failure states become less frequent, reducing the amount of training signal for error recovery. To maintain a diverse set of difficult recovery scenarios, we introduce Buffered Roll-In. We maintain a replay buffer B of previously generated prefixes. During rollout generation on training examples from $D_{RL}$ , if a trajectory terminates with an incorrect final answer, we treat the state immediately before termination as a valid but unresolved intermediate state. We remove the termination action and store the resulting prefix $\tau_{pre}$ in B. During training, instead of always generating rollouts from scratch, we also sample prefixes from B. For each sampled prefix $\tau_{pre}$ , the policy generates a group of G suffix completions $\{\tau_{\mathrm{suf}}^{(1)},\ldots,\tau_{\mathrm{suf}}^{(G)}\}$ , and GRPO is applied only to these generated suffixes. This directly trains the model to recover from previously failed intermediate states.

To balance exploring new questions with addressing past failures, we construct each training batch by sampling questions from $D_{RL}$ for RTM with probability $\rho$ and prefixes from the buffer B for Buffered Roll-In with probability $1 - \rho$ . The final optimization objective is:

$$
\mathcal {J} _ {\mathrm{Total}} (\theta) = \rho \mathcal {J} _ {\mathrm{RTM}} (\theta) + (1 - \rho) \mathcal {J} _ {\mathrm{Buff}} (\theta)
$$

This creates a self-paced curriculum: as the policy improves with RTM, the buffer naturally accumulates “harder” failure modes that the current policy struggles to resolve.

## 4 Task Setup

We study multi-turn inference with self-reflection in two visual feedback environments: visual grounding and spatial navigation. In both tasks, the model proposes an intermediate answer, receives an image observation $I_{t}$ from the environment, and either refines its prediction or terminates with a final answer.

## 4.1 Visual Grounding

Visual grounding (Kazemzadeh et al., 2014; Plummer et al., 2017; Yu et al., 2018; Mao et al., 2016) aims to localize image regions referred to by natural language. We focus on data visualizations, where precise localization is challenging because tables and charts differ substantially from the real-world image distributions commonly seen during pre-training (Wang et al., 2024b; Tang et al., 2025). This provides a controlled testbed for studying whether models can use visual feedback to self-correct under distribution shifts.

Task and Environment. Given an image I and instruction Q, the model must predict the target coordinate $p = (x, y)$ . At each turn t, it either proposes a coordinate $p_{t}$ or terminates with a final prediction. After each proposal, the environment returns visual feedback $I_{t}$ : a $200 \times 200$ crop centered at $p_{t}$ with a red marker indicating the proposed location.

Reward and Evaluation. The outcome reward is 1.0 if the final coordinate $p_T$ falls within a Euclidean distance threshold $\delta_{\mathrm{tol}} = 40 \, \mathrm{px}$ of the ground-truth coordinate, and 0 otherwise. We report accuracy under this criterion, except for Bar Chart, where a prediction is correct if it falls within the target bar's bounding box. We also use a distance-based reflection reward during training (Appendix A).

Data and Splits. We synthesize tables and charts following prior data generation procedures (Long et al., 2025; Zheng et al., 2025). The training and in-distribution test sets use small arXiv-style tables with row- or column-header queries. We evaluate OOD generalization on four 1K-example test sets: (1) Large Tables, which increases table size; (2) Cell Query, which asks for inner table cells; (3) Bar Chart, which transfers grounding to bar charts; (4) and Scatter Plot, which requires localizing labeled points. Figure 3 illustrates these tasks, with full generation details in Appendix C.

## 4.2 Spatial Navigation

Spatial navigation is challenging for LVLMs (Wang et al., 2024a; Stogiannidis et al., 2025) and provides a natural visual feedback environment because agent movement can be inspected and corrected over multiple turns. We use FrozenLake (Wu et al., 2025b; Brockman et al., 2016), a grid-based navigation task where the agent must reach a goal while avoiding holes and impassable obstacles. Prior work shows that textual chain-of-thought alone is insufficient for such tasks (Xu et al., 2026).

Task and Environment. Given an input map image I and a fixed instruction Q, the model predicts the shortest, valid path from the start to the goal that does not run into holes or walls. A path proposal is a sequence of actions $v_{t} = (v_{t}^{1}, v_{t}^{2}, \ldots, v_{t}^{L})$ , where each action is in $\{left, right, up, down\}$ . At each turn, the model either proposes a path or terminates with a final prediction. After each proposal, the environment returns visual feedback $I_{t}$ by drawing a segmented red line for the predicted path on the map. Figure 10 shows a detailed example.

Reward and Evaluation. Following Xu et al. (2026), we use exact match (EM) as both the outcome reward and evaluation metric. A trajectory receives reward 1.0 if the final path $v_{T}$ is valid and matches the optimal solution length, and 0 otherwise. We additionally use an improvement-based reflection reward that measures partial progress toward the optimal path, detailed in Appendix A.

Data and Training. We adopt the training and evaluation data from Xu et al. (2026), using 3–5 grid maps as the in-distribution training setting and larger $6 \times 6$ and $7 \times 7$ maps as OOD evaluations. Because spatial navigation is difficult for LVLMs, we first warm-start the model with direct-answer training on in-distribution examples without visual feedback, following Xu et al. (2026), and then apply all subsequent training methods to this warm-started model. Details can be found in Appendix D.

<table><tr><td rowspan="2"></td><td rowspan="2">In-Distribution</td><td colspan="5">Out-of-Distribution</td></tr><tr><td>Large Table</td><td>Cell Query</td><td>Bar Chart</td><td>Scatter Plot</td><td>OOD Avg</td></tr><tr><td colspan="7">Zero-shot</td></tr><tr><td>Qwen2.5-VL-3BSingle</td><td>5.3</td><td>4.8</td><td>3.4</td><td>13.1</td><td>2.9</td><td>6.0</td></tr><tr><td>Qwen2.5-VL-3B*Multi</td><td>5.6 (77.3%)</td><td>2.2 (69.2%)</td><td>1.3 (86.1%)</td><td>12.0 (14.7%)</td><td>2.4 (75.5%)</td><td>4.5</td></tr><tr><td>Qwen2.5-VL-7BSingle</td><td>17.9</td><td>9.7</td><td>6.1</td><td>15.1</td><td>4.5</td><td>8.8</td></tr><tr><td>Qwen2.5-VL-7B*Multi</td><td>19.8 (58.0%)</td><td>8.9 (49.3%)</td><td>7.2 (76.8%)</td><td>14.7 (15.5%)</td><td>4.6 (38.2%)</td><td>8.8</td></tr><tr><td>VL-Rethinker-7B</td><td>15.1</td><td>7.9</td><td>5.7</td><td>1.0</td><td>3.7</td><td>4.6</td></tr><tr><td>VL-Rethinker-32B</td><td>42.4</td><td>18.7</td><td>27.6</td><td>9.5</td><td>32.7</td><td>22.1</td></tr><tr><td colspan="7">Qwen2.5-VL-3B-Instruct</td></tr><tr><td colspan="7">SFT</td></tr><tr><td>Single-SFT</td><td>80.4</td><td>46.1</td><td>2.4</td><td>25.7</td><td>23.8</td><td>24.5</td></tr><tr><td>Multi-SFT</td><td>84.7</td><td>50.4</td><td>1.6</td><td>13.1</td><td>24.8</td><td>22.5</td></tr><tr><td>Reflection Tuning</td><td>92.7</td><td>52.5</td><td>7.0</td><td>25.0</td><td>27.4</td><td>28.0</td></tr><tr><td colspan="7">RL</td></tr><tr><td>Single-SFT → GRPO</td><td>96.2</td><td>53.3</td><td>5.3</td><td>27.1</td><td>34.7</td><td>30.1</td></tr><tr><td>Multi-SFT → GRPO</td><td>99.6</td><td>78.6</td><td>13.5</td><td>30.7</td><td>37.2</td><td>40.0</td></tr><tr><td>VRRL(Ours)</td><td>99.6</td><td>88.6</td><td>20.3</td><td>33.5</td><td>40.3</td><td>45.7</td></tr><tr><td colspan="7">Qwen2.5-VL-7B-Instruct</td></tr><tr><td colspan="7">SFT</td></tr><tr><td>Single-SFT</td><td>83.6</td><td>62.8</td><td>34.0</td><td>20.3</td><td>68.9</td><td>46.5</td></tr><tr><td>Multi-SFT</td><td>84.8</td><td>66.2</td><td>39.1</td><td>20.9</td><td>73.3</td><td>49.9</td></tr><tr><td>Reflection Tuning</td><td>95.3</td><td>75.1</td><td>51.8</td><td>14.2</td><td>81.3</td><td>55.6</td></tr><tr><td colspan="7">RL</td></tr><tr><td>Single-SFT → GRPO</td><td>99.6</td><td>89.6</td><td>68.3</td><td>38.6</td><td>84.3</td><td>70.2</td></tr><tr><td>Multi-SFT → GRPO</td><td>99.6</td><td>91.4</td><td>68.4</td><td>46.8</td><td>86.0</td><td>73.2</td></tr><tr><td>VRRL(Ours)</td><td>99.7</td><td>89.6</td><td>77.3</td><td>57.0</td><td>89.7</td><td>78.4</td></tr></table>

Table 1: In-distribution and OOD evaluation results for visual grounding across models. We perform paired bootstrap tests to compare the best-performing model with the second-best model in each column. Bold indicates that the best result is better by a statistically significant margin ( $p < 0.05$ ). For Qwen2.5-VL-3B $_{\text{Multi}}$ and Qwen2.5-VL-7B $_{\text{Multi}}$ , we report the percentage of traces where the model's reflection turns repeat the same predictions as previous turns.

## 5 Experimental Setup

We compare our proposed method against baseline methods below. Implementation details and example outputs can be found in Appendix B and F.

Zero-Shot Baselines. We evaluate Qwen2.5-VL-3B-Instruct and 7B models (Bai et al., 2025b) under two settings: direct pointing and a multi-turn reflection prompt that asks the model to critique and refine its coordinates without parameter updates. For spatial navigation tasks, we also evaluate Qwen3-VL-4B-Instruct (Bai et al., 2025a).

Supervised Fine-Tuning (SFT) Baselines. We compare two SFT strategies for Qwen2.5-VL-3B/7B on visual grounding, and Qwen2.5-VL-3B and Qwen3-VL-4B on spatial navigation. Single-SFT trains on perfect single-turn trajectories, where the model outputs the correct coordinate and terminates immediately. Multi-SFT trains on a mixture of perfect and synthetic recovery trajectories, where the model first predicts an incorrect coordinate and then iteratively uses visual feedback to refine its prediction until correct.

RL Baselines. We apply standard GRPO to Single-SFT and Multi-SFT (Single/Multi-SFT → GRPO). These baselines (1) use a sparse binary outcome reward (1 for success, 0 for failure) without the reflection reward shaping $R_{refl}$ ; (2) calculate loss over the full trajectory without Random Turn Masking; and (3) do not use Buffered Roll-In.

Reflection-Oriented Baselines. We include prior fine-tuning approaches designed to instill self-reflection in LVLMs as baselines: (1) VL-Rethinker (Wang et al., 2026a) augments GRPO training with selective sampling and forced rethinking to encourage self-reflection through textual reasoning only by scaling long CoT traces. This differs from our setting, which focuses on visually grounded self-reflection driven by visual feedback. We directly evaluate the released VL-Rethinker models trained on diverse multimodal reasoning tasks for their OOD generalization capabilities. (2) Reflection Tuning (Wu et al., 2025a) trains

![](images/6cf89407089e9b3eaf32530dea9631f78619369100b8fd3f9e459fed9ea8f337.jpg)  
Figure 4: Performance for multi-turn reflection across in-distribution and OOD tasks for visual grounding tasks. The top row illustrates the progression of cumulative accuracy across turns. Single-SFT → GRPO is a single-turn baseline method where its per-turn accuracy remains the same. The bottom row shows the percentage of examples reaching turn X. VRRL generally uses reflection better, improving more over iterations.

LVLMs to perform self-reflection based on visual feedback through iterative online SFT, where the model learns from corrections of its own generated error trajectories. Since this approach is naturally compatible with our visually grounded setting, we apply reflection tuning on top of our multi-turn SFT model using the same in-distribution training data.

VRRL (Ours). Our method applies our full training recipe on top of Multi-SFT.

Training. For visual grounding, we use 15K training examples for SFT and 6K for RL. For spatial navigation, we use 4K training examples for SFT and 2K for RL. Implementation details, dataset details, and training configurations can be found in Appendix B.

## 6 Results

## 6.1 Visual Grounding

Prompting does not elicit reliable self-reflection. Table 1 shows that off-the-shelf LVLMs struggle with precise spatial localization. In the zero-shot setting, both 3B and 7B models achieve low accuracy across tasks: Prompting models to reflect provides little benefit and can even hurt performance. In many multi-turn traces, zero-shot models simply repeat previous predictions without meaningful correction.

Furthermore, VL-Rethinker models that have been trained to reflect with textual CoT reasoning underperform on OOD tasks; their reflective CoT traces generally fail to correct mistakes. These results suggest that visually grounded self-correction cannot be reliably elicited through prompting or textual CoT alone (Wu et al., 2026b, 2025c; Huang et al., 2025; Jiang et al., 2025).

SFT is limited. SFT methods, including Multi-SFT and reflection tuning that specifically instill visual self-reflection, mainly teach the in-distribution knowledge, but remain brittle on the evaluated OOD tasks. For example, Multi-SFT only learns the format of reflection rather than robust error-correcting behavior.

RL on multi-turn reflection models improves OOD generalization. RL substantially improves OOD performance over SFT. Even single-turn RL, Single-SFT → GRPO, raises Large Table accuracy to 53.3% for 3B and 89.6% for 7B. However, the main gains come from combining RL with a multi-turn reflection formulation: Multi-SFT → GRPO improves the 3B model to 78.6% on Large Table, 13.5% on Cell Query, 30.7% on Bar Chart, and 37.2% on Scatter Plot, yielding 3–25% absolute gains over its single-turn counterpart. VRRL further improves over Multi-SFT → GRPO by 3–10% on most OOD tasks while maintaining near-perfect in-distribution accuracy for both model scales. These gains are notable because the OOD splits require generalization across table size, query type, and visual domain, indicating that VRRL induces a more robust visual grounding capability.

VRRL teaches effective reflection. Figure 4 shows that VRRL's gains come from improved multi-turn correction rather than stronger one-shot grounding alone. While Single-SFT → GRPO improves single-turn accuracy, it lacks the turn-by-turn refinement behavior of multi-turn RL. Multi-SFT → GRPO improves across early reflection turns, but VRRL converts this behavior into stronger OOD generalization. The turn-distribution plots further show that VRRL adapts its number of refinement steps to task familiarity: it terminates early on in-distribution examples, but continues refining on OOD tasks and achieves higher accuracy in later turns.

<table><tr><td rowspan="3"></td><td colspan="4">Qwen2.5-VL-3B-Instruct</td><td colspan="4">Qwen3-VL-4B-Instruct</td></tr><tr><td>ID</td><td colspan="3">OOD</td><td>ID</td><td colspan="3">OOD</td></tr><tr><td>Avg.</td><td>6 × 6</td><td>7 × 7</td><td>Avg.</td><td>Avg.</td><td>6 × 6</td><td>7 × 7</td><td>Avg.</td></tr><tr><td colspan="9">Zero-shot</td></tr><tr><td>Single</td><td>2.4</td><td>1.2</td><td>1.6</td><td>1.4</td><td>8.4</td><td>2.4</td><td>2.0</td><td>2.2</td></tr><tr><td>Multi</td><td>2.8</td><td>0.8</td><td>2.0</td><td>1.4</td><td>34.0</td><td>8.8</td><td>2.4</td><td>5.6</td></tr><tr><td>VL-Rethinker-7B</td><td>11.3</td><td>3.2</td><td>3.2</td><td>3.2</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VL-Rethinker-32B</td><td>26.5</td><td>10.8</td><td>3.6</td><td>7.2</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="9">SFT</td></tr><tr><td>Base*</td><td>77.7</td><td>33.2</td><td>4.8</td><td>19.0</td><td>88.1</td><td>49.6</td><td>5.6</td><td>27.6</td></tr><tr><td>Single-SFT</td><td>83.5</td><td>39.2</td><td>8.8</td><td>24.0</td><td>90.7</td><td>56.8</td><td>8.0</td><td>32.4</td></tr><tr><td>Multi-SFT</td><td>81.2</td><td>41.6</td><td>10.2</td><td>25.9</td><td>86.9</td><td>60.4</td><td>21.2</td><td>40.8</td></tr><tr><td>Reflection Tuning</td><td>85.7</td><td>49.6</td><td>12.4</td><td>31.0</td><td>93.6</td><td>63.6</td><td>13.2</td><td>38.4</td></tr><tr><td colspan="9">RL</td></tr><tr><td>Single-SFT → GRPO</td><td>85.9</td><td>49.2</td><td>8.8</td><td>29.0</td><td>93.2</td><td>62.4</td><td>5.6</td><td>34.0</td></tr><tr><td>Multi-SFT → GRPO</td><td>85.2</td><td>42.8</td><td>9.2</td><td>26.0</td><td>87.9</td><td>61.6</td><td>10.4</td><td>36.0</td></tr><tr><td>VRRL (Ours)</td><td>83.9</td><td>54.8</td><td>23.6</td><td>39.2</td><td>89.1</td><td>65.2</td><td>39.2</td><td>52.2</td></tr></table>

Table 2: In-distribution (ID) and OOD evaluation for spatial navigation on Qwen2.5-VL-3B-Instruct and Qwen3-VL-4B-Instruct. Base\*: We first train models on ID data with direct answering without visual feedback to obtain Base as a warm-started model on the task distribution, and then apply all training methods based upon this model. We perform paired bootstrap tests to compare the best-performing model with the second-best model in each column. Bold indicates that the best result is statistically significant $p < 0.05$ .

<table><tr><td rowspan="2"></td><td colspan="2">ID</td><td colspan="2">OOD</td></tr><tr><td># Turns</td><td> $\Delta_{ref}$ </td><td># Turns</td><td> $\Delta_{ref}$ </td></tr><tr><td colspan="5">Qwen2.5-VL-3B-Instruct</td></tr><tr><td>Single-SFT → GRPO</td><td>2.00</td><td>+ 0</td><td>2.00</td><td>+ 0</td></tr><tr><td>Multi-SFT</td><td>2.41</td><td>+ 0.1</td><td>3.51</td><td>+ 2.8</td></tr><tr><td>Multi-SFT → GRPO</td><td>2.00</td><td>+ 0</td><td>2.00</td><td>+ 0</td></tr><tr><td>Reflection Tuning</td><td>2.23</td><td>+ 1.3</td><td>3.17</td><td>+ 3.4</td></tr><tr><td>VRRL(Ours)</td><td>2.37</td><td>+ 6.0</td><td>3.39</td><td>+ 13.6</td></tr><tr><td colspan="5">Qwen3-VL-4B-Instruct</td></tr><tr><td>Single-SFT → GRPO</td><td>2.00</td><td>+ 0</td><td>2.00</td><td>+ 0</td></tr><tr><td>Multi-SFT</td><td>2.24</td><td>+ 1.1</td><td>4.57</td><td>+ 11.0</td></tr><tr><td>Multi-SFT → GRPO</td><td>2.00</td><td>+ 0</td><td>2.00</td><td>+ 0</td></tr><tr><td>Reflection Tuning</td><td>2.24</td><td>+ 1.7</td><td>4.69</td><td>+ 7.4</td></tr><tr><td>VRRL(Ours)</td><td>2.22</td><td>+ 1.1</td><td>4.64</td><td>+ 23.0</td></tr></table>

Table 3: Reflection behaviors of training methods for spatial navigation. # Turns denotes the average number of turns, including a termination turn, of the model responses, and $\Delta_{ref}$ is the improvement of task accuracy by multi-turn reflection inference.

## 6.2 Spatial Navigation

VRRL improves OOD spatial navigation. Table 2 shows results on our spatial navigation task. VRRL achieves comparable in-distribution accuracy to the baselines while outperforming all baselines on OOD settings for both models. On OOD tasks, VRRL improves multi-turn SFT by 13.3% on Qwen2.5-VL-3B and 11.4% on Qwen3-VL-4B, and it outperforms the second-best baselines by 9.8% on average across two models.

VRRL uses reflection more efficiently. We next analyze reflection behavior in spatial navigation. Table 3 reports the average number of turns, including the required termination call, as well as the performance improvement from multi-turn reflection. Standard GRPO with only outcome reward (Multi-SFT → GRPO) largely suppresses the reflection behavior learned during multi-turn SFT in this setting. In contrast, VRRL uses reflection more effectively and selectively: it improves performance over turns while requiring similar number of turns on average than other reflection-oriented baselines.

## 7 Ablations

Table 4 and 5 show ablation studies for visual grounding and spatial navigation, respectively, to isolate the impact of each proposed component. When added individually to the Multi-SFT → GRPO baseline, RTM and Buffered roll-in produce mixed results. In particular, buffered roll-in alone causes the model to collapse into mostly single-turn behavior, losing the reflection capability learned during multi-turn SFT. However, by exposing the model to diverse intermediate states, it still improves average single-turn performance, leading to 42.6% on visual grounding and substantially outperforming Single-SFT → GRPO at 30.1%.

<table><tr><td rowspan="2"></td><td colspan="5">Out-of-Distribution</td></tr><tr><td>Large Table</td><td>Cell Query</td><td>Bar Chart</td><td>Scatter Plot</td><td>Average</td></tr><tr><td>Single-SFT → GRPO</td><td>53.3</td><td>5.3</td><td>27.1</td><td>34.7</td><td>30.1</td></tr><tr><td>Multi-SFT → GRPO</td><td>78.6</td><td>13.5</td><td>30.7</td><td>37.2</td><td>40.0</td></tr><tr><td>+ RTM</td><td>70.5</td><td>11.4</td><td>33.8</td><td>34.1</td><td>37.5</td></tr><tr><td>+ Buffered Roll-In</td><td>63.2</td><td>26.1</td><td>43.9</td><td>37.1</td><td>42.6</td></tr><tr><td>+ RTM + Buffered Roll-In</td><td>78.6</td><td>18.1</td><td>36.9</td><td>39.9</td><td>43.4</td></tr><tr><td>+ Reflection Reward</td><td>79.1</td><td>19.5</td><td>38.0</td><td>34.1</td><td>42.7</td></tr><tr><td>VRRL (Ours)</td><td>88.6</td><td>20.3</td><td>33.5</td><td>40.3</td><td>45.7</td></tr></table>

Table 4: Ablation study on VRRL over the 3B model (Reflection Reward + RTM + Buffered Roll-In) on visual grounding. We find that with only the Buffered Roll-In method, the model fails to learn multi-turn reflection and terminates immediately after one pointing call. However, it still outperforms the single-turn model Single-SFT → GRPO on all settings by a large margin.

<table><tr><td rowspan="2"></td><td>ID</td><td colspan="3">OOD</td></tr><tr><td>Avg</td><td>6 × 6</td><td>7 × 7</td><td>Avg</td></tr><tr><td colspan="5">Qwen2.5-VL-3B-Instruct</td></tr><tr><td>Single-SFT → GRPO</td><td>85.9</td><td>49.2</td><td>8.8</td><td>29.0</td></tr><tr><td>Multi-SFT → GRPO</td><td>85.2</td><td>42.8</td><td>9.2</td><td>26.0</td></tr><tr><td>+ RTM</td><td>84.4</td><td>48.0</td><td>9.2</td><td>28.6</td></tr><tr><td>+ Buffered Roll-In</td><td>85.3</td><td>47.6</td><td>7.6</td><td>27.6</td></tr><tr><td>+ RTM + Buffered Roll-In</td><td>83.3</td><td>52.4</td><td>13.2</td><td>32.8</td></tr><tr><td>+ Reflection Reward</td><td>86.9</td><td>51.6</td><td>10.0</td><td>30.8</td></tr><tr><td>VRRL (Ours)</td><td>83.9</td><td>54.8</td><td>23.6</td><td>39.2</td></tr></table>

Table 5: Ablation study on VRRL over the 3B model (Reflection Reward + RTM + Buffered Roll-In) on spatial navigation.

Importantly, combining RTM with buffered roll-in restores and enhances multi-turn behavior. This combination resolves the single-turn collapse, restoring Large Table performance to the baseline level and improving all other OOD settings, demonstrating that these two components are highly complementary. Reflection reward is also necessary to achieve strong performance for both tasks, indicating that the best OOD robustness is obtained only when all three components are combined. Furthermore, the reflection reward alone proves to be an effective individual addition (for example, 42.7% average on visual grounding). By providing a reward shaping signal rather than a sparse binary reward, it teaches the model to iteratively move toward better states even when the final answer remains incorrect, which is particularly useful when the query type is unseen. The full model outperforms all ablated baselines, indicating that the best OOD robustness is obtained only when all three components are combined.

## 8 Related Work

Self-Reflection of LVLMs. Self-reflection, the process of inspecting previously generated outputs and correcting potential mistakes, has emerged as an important capability of LLMs (Madaan et al., 2023; Gou et al., 2024) and has been applied to a variety of downstream tasks (Shinn et al., 2023; Wadhwa et al., 2024). As pre-trained LVLMs only exhibit limited self-reflection capabilities (Cheng et al., 2025), post-training methods have been explored to further improve this behavior (Li et al., 2025; Huang et al., 2024). As existing approaches mainly train LVLMs for self-reflection by scaling CoT reasoning in text (Wan et al., 2025; Yang et al., 2025a; Chung et al., 2025; Jian et al., 2025; Wang et al., 2026a), they often over-rely on the textual information in prompts (Vo et al., 2026; Tang et al., 2026), and fail to utilize visual feedback to adjust predictions. In contrast, visually grounded self-reflection, where models learn self-verification and error recovery by leveraging image inputs and intermediate visual feedback, is less explored.

Thinking with Images. A recent line of work studies the paradigm of “thinking with images” (Su et al., 2025b), in which VLMs are augmented with external tools (e.g., OCR, depth analysis, zooming, and image segmentation) that provide new visual information as tool outputs across multiple turns (Su et al., 2025a; Hu et al., 2024; Shao et al., 2024a). These methods equip VLMs with the ability to make accurate visual tool calls (Wu et al., 2026a) and to integrate tool outputs into their reasoning process (Huang et al., 2025; Wang et al., 2026b; Yang et al., 2025b), but they mainly focus on incentivizing correct tool calls from VLMs rather than reflective capability of VLMs over tool outputs to validate their candidate answers.

## 9 Conclusion

In this paper, we showed that multi-turn reflection can improve robustness of visual grounding and spatial navigation for LVLMs. To this end, we propose VRRL, an RL training method that combines Random Turn Masking and Buffered Roll-In to teach models both when to stop and how to recover from intermediate mistakes using visual feedback. Our method achieves strong in-distribution performance while substantially improving OOD generalization over zero-shot, SFT, and standard RL baselines. Our model successfully uses multiple steps of refinement to achieve this performance. Our work highlights multi-turn self-reflection as a key direction for improving the robustness and generalization of LVLMs.

## Limitations

We acknowledge several limitations in our work. First, our evaluation assumes visual feedback is available from the environment, which is not readily applicable to some multi-modal reasoning tasks such as open-ended VQA on real-world images. While this design choice was intentional, allowing us to isolate multi-turn grounding and refinement mechanics, it may not fully capture the complexity and diversity of real-world tasks.

Second, due to computational constraints, all training experiments are conducted with a single model family of Qwen (Qwen2.5-VL and Qwen3-VL), up to 7B model scale. While our results have clearly demonstrated the effectiveness of our training method, we have not evaluated whether the same conclusions hold consistently across larger model scale or different architectures.

## Acknowledgments

This work was supported by NSF CAREER Award IIS-2145280, NSF grant IIS-2433071, the NSF AI Institute for Foundations of Machine Learning (IFML), the NSF under Cooperative Agreement 2421782 and the Simons Foundation grant MPS-AI-00010515 awarded to the NSF-Simons AI Institute for Cosmic Origins — CosmicAI, https:

//www.cosmicai.org/, and an award from Exxon-Mobil. This work was also partially supported by the Sloan Foundation. Finally, this work has been supported by a compute grant from NVIDIA. We also acknowledge use of the research computing resources of the Empire AI Consortium, Inc., with support from the State of New York, the Simons Foundation, and the Secunda Family Foundation.

## References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025a. Qwen3-vl technical report. Preprint, arXiv:2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025b. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923.

Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. 2016. OpenAI Gym. Preprint, arXiv:1606.01540.

Kanzhi Cheng, Li YanTao, Fangzhi Xu, Jianbing Zhang, Hao Zhou, and Yang Liu. 2025. Vision-language models can self-improve reasoning via reflection. In Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8876–8892, Albuquerque, New Mexico. Association for Computational Linguistics.

Jiwan Chung, Junhyeok Kim, Siyeol Kim, Jaeyoung Lee, Min Soo Kim, and Youngjae Yu. 2025. v1: Learning to point visual tokens for multimodal grounded reasoning. Preprint, arXiv:2505.18842.

Aarash Feizi, Shravan Nayak, Xiangru Jian, Kevin Qinghong Lin, Kaixin Li, Rabiul Awal, Xing Han Lù, Johan Obando-Ceron, Juan A. Rodriguez, Nicolas Chapados, David Vazquez, Adriana Romero-Soriano, Reihaneh Rabbany, Perouz Taslakian, Christopher Pal, Spandana Gella, and Sai Rajeswar. 2026. Grounding computer use agents on human demonstrations. In The Fourteenth International Conference on Learning Representations.

Kanishk Gandhi, Ayush K Chakravarthy, Anikait Singh, Nathan Lile, and Noah Goodman. 2025. Cognitive behaviors that enable self-improving reasoners, or, four habits of highly effective STars. In Second Conference on Language Modeling.

Zhibin Gou, Zhihong Shao, Yeyun Gong, yelong shen, Yujiu Yang, Nan Duan, and Weizhu Chen. 2024. CRITIC: Large language models can self-correct with tool-interactive critiquing. In The Twelfth International Conference on Learning Representations.

Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, Xiaokang Zhang, Xingkai Yu, Yu Wu, Z. F. Wu, Zhibin Gou, Zhihong Shao, Zhuoshu Li, Ziyi Gao, Aixin Liu, and 175 others. 2025. Deepseek-r1 incentivizes reasoning in llms through reinforcement learning. Nature, 645(8081):633–638.

Yushi Hu, Weijia Shi, Xingyu Fu, Dan Roth, Mari Ostendorf, Luke Zettlemoyer, Noah A. Smith, and Ranjay Krishna. 2024. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024. Large language models cannot self-correct reasoning yet. In The Twelfth International Conference on Learning Representations.

Xinyu Huang, Yuhao Dong, Weiwei Tian, Bo Li, Rui Feng, and Ziwei Liu. 2025. High-resolution visual reasoning via multi-turn grounding-based reinforcement learning. Preprint, arXiv:2507.05920.

Pu Jian, Junhong Wu, Wei Sun, Chen Wang, Shuo Ren, and Jiajun Zhang. 2025. Look again, think slowly: Enhancing visual reflection in vision-language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 9251–9270, Suzhou, China. Association for Computational Linguistics.

Dongzhi Jiang, Renrui Zhang, Ziyu Guo, Yanwei Li, Yu Qi, Xinyan Chen, Liuhui Wang, Jianhan Jin, Claire Guo, Shen Yan, Bo Zhang, Chaoyou Fu, Peng Gao, and Hongsheng Li. 2025. MME-CoT: Benchmarking Chain-of-Thought in Large Multimodal Models for Reasoning Quality, Robustness, and Efficiency. In Forty-second International Conference on Machine Learning.

Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. 2014. ReferItGame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 787–798, Doha, Qatar. Association for Computational Linguistics.

KimiTeam, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, Chuning Tang, Congcong Wang, Dehao Zhang, Enming Yuan, Enzhe Lu, Fengxiang Tang, Flood Sung, Guangda Wei, Guokun Lai, and 77 others. 2025. Kimi k1.5: Scaling reinforcement learning with llms. Preprint, arXiv:2501.12599.

Jiaze Li, Hao Yin, Wenhui Tan, Jingyang Chen, Boshen Xu, Yuxun Qu, Yijing Chen, Jianzhong Ju, Zhenbo Luo, and Jian Luan. 2025. Revisor: Beyond textual reflection, towards multimodal introspective reasoning in long-form video understanding. arXiv preprint arXiv:2511.13026.

Hui Long, Haoli Yu, Jingqiao Wang, Jian Long, and Changxin Fang. 2025. ChartBench: A Comprehensive Evaluation Benchmark for Chart Understanding Capabilities of Vision-Language Models. In Proceedings of the 2025 2nd International Conference on Virtual Reality, Image and Signal Processing, VRISP '25, page 253–257, New York, NY, USA. Association for Computing Machinery.

Ji Ma, Wei Suo, Peng Wang, and Yanning Zhang. 2026. Understanding and mitigating hallucinations in multimodal chain-of-thought models. arXiv preprint arXiv:2603.27201.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. In Thirty-seventh Conference on Neural Information Processing Systems.

Junhua Mao, Jonathan Huang, Alexander Toshev, Oana Camburu, Alan Yuille, and Kevin Murphy. 2016. Generation and comprehension of unambiguous object descriptions. In CVPR.

Ahmed Masry, Mohammed Saidul Islam, Mahir Ahmed, Aayush Bajaj, Firoz Kabir, Aaryaman Kartha, Md Tahmid Rahman Laskar, Mizanur Rahman, Shadikur Rahman, Mehrad Shahmohammadi, Megh Thakkar, Md Rizwan Parvez, Enamul Hoque, and Shafiq Joty. 2025. ChartQAPro: A more diverse and challenging benchmark for chart question answering. In Findings of the Association for Computational Linguistics: ACL 2025, pages 19123–19151, Vienna, Austria. Association for Computational Linguistics.

Bryan A. Plummer, Liwei Wang, Christopher M. Cervantes, Juan C. Caicedo, Julia Hockenmaier, and Svetlana Lazebnik. 2017. Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models. IJCV, 123(1):74–93.

Hao Shao, Shengju Qian, Han Xiao, Guanglu Song, Zhuofan Zong, Letian Wang, Yu Liu, and Hongsheng Li. 2024a. Visual CoT: Advancing Multi-Modal Language Models with a Comprehensive Dataset and Benchmark for Chain-of-Thought Reasoning. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, and 1 others. 2024b. DeepSeekMath: Pushing the limits of mathematical

reasoning in open language models. arXiv preprint arXiv:2402.03300.

Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik R Narasimhan, and Shunyu Yao. 2023. Reflexion: language agents with verbal reinforcement learning. In Thirty-seventh Conference on Neural Information Processing Systems.

Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, Akshay Nathan, Alan Luo, Alec Helyar, Aleksander Madry, Aleksandr Efremov, Aleksandra Spyra, Alex Baker-Whitcomb, Alex Beutel, Alex Karpenko, and 467 others. 2026. Openai gpt-5 system card. Preprint, arXiv:2601.03267.

Zayne Sprague, Jack Lu, Manya Wadhwa, Sedrick Keh, Mengye Ren, and Greg Durrett. 2026. SkillFactory: Self-Distillation For Learning Cognitive Behaviors. In Proceedings of the International Conference on Learning Representations (ICLR).

Ilias Stogiannidis, Steven McDonagh, and Sotirios A. Tsaftaris. 2025. Mind the gap: Benchmarking spatial reasoning in vision-language models. Preprint, arXiv:2503.19707.

Zhaochen Su, Linjie Li, Mingyang Song, Yunzhuo Hao, Zhengyuan Yang, Jun Zhang, Guanjie Chen, Jiawei Gu, Juntao Li, Xiaoye Qu, and 1 others. 2025a. OpenThinkIMG: Learning to Think with Images via Visual Tool Reinforcement Learning. arXiv preprint arXiv:2505.08617.

Zhaochen Su, Peng Xia, Hangyu Guo, Zhenhua Liu, Yan Ma, Xiaoye Qu, Jiaqi Liu, Yanshu Li, Kaide Zeng, Zhengyuan Yang, and 1 others. 2025b. Thinking with images for multimodal reasoning: Foundations, methods, and future frontiers. arXiv preprint arXiv:2506.23918.

Kaihua Tang, Jiaxin Qi, Jinli Ou, Yuhua Zheng, and Jianqiang Huang. 2026. Scaling test-time robustness of vision-language models via self-critical inference framework. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR).

Liyan Tang, Grace Kim, Xinyu Zhao, Thom Lake, Wenxuan Ding, Fangcong Yin, Prasann Singhal, Manya Wadhwa, Zeyu Leo Liu, Zayne Rea Sprague, Ramya Namuduri, Bodun Hu, Juan Diego Rodriguez, Puyuan Peng, and Greg Durrett. 2025. Chartmuseum: Testing visual reasoning capabilities of large vision-language models. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

An Vo, Khai-Nguyen Nguyen, Mohammad Reza Taesiri, Vy Tuong Dang, Anh Totti Nguyen, and Daeyoung Kim. 2026. Vision language models are biased. In The Fourteenth International Conference on Learning Representations.

Manya Wadhwa, Xinyu Zhao, Junyi Jessy Li, and Greg Durrett. 2024. Learning to refine with fine-grained natural language feedback. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 12281–12308, Miami, Florida, USA. Association for Computational Linguistics.

Zhongwei Wan, Zhihao Dou, Che Liu, Yu Zhang, Dongfei Cui, Qinjian Zhao, Hui Shen, Jing Xiong, Yi Xin, Yifan Jiang, Chaofan Tao, Yangfan He, Mi Zhang, and Shen Yan. 2025. SRPO: Enhancing multimodal LLM reasoning via reflection-aware reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Haozhe Wang, Chao Qu, Zuming Huang, Wei Chu, Fangzhen Lin, and Wenhu Chen. 2026a. VL-rethinker: Incentivizing self-reflection of vision-language models with reinforcement learning. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Jiacong Wang, Zijian Kang, Haochen Wang, Liang Xiao, Ya Wang, Jiawen Li, Bohong Wu, Ran Jiao, Haiyong Jiang, ChaoFeng, and Jun Xiao. 2026b. VGR: Visual grounded reasoning. In The Fourteenth International Conference on Learning Representations.

Jiayu Wang, Yifei Ming, Zhenmei Shi, Vibhav Vineet, Xin Wang, Yixuan Li, and Neel Joshi. 2024a. Is A Picture Worth A Thousand Words? Delving Into Spatial Reasoning for Vision Language Models. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Zirui Wang, Mengzhou Xia, Luxi He, Howard Chen, Yitao Liu, Richard Zhu, Kaiqu Liang, Xindi Wu, Hao-tian Liu, Sadhika Malladi, Alexis Chevalier, Sanjeev Arora, and Danqi Chen. 2024b. Charxiv: Charting gaps in realistic chart understanding in multimodal LLMs. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track.

Mingyuan Wu, Jingcheng Yang, Jize Jiang, Meitang Li, Kaizhuo Yan, Hanchao Yu, Minjia Zhang, ChengXiang Zhai, and Klara Nahrstedt. 2026a. VTool-r1: VLMs learn to think with images via reinforcement learning on multimodal tool use. In The Fourteenth International Conference on Learning Representations.

Penghao Wu, Shengnan Ma, Bo Wang, Jiaheng Yu, Lewei Lu, and Ziwei Liu. 2025a. GUI-reflection: Empowering multimodal GUI models with self-reflection behavior. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Qiucheng Wu, Handong Zhao, Michael Saxon, Trung Bui, William Yang Wang, Yang Zhang, and Shiyu Chang. 2025b. VSP: Assessing the dual challenges of perception and reasoning in spatial planning tasks for VLMs. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

Xueqing Wu, Yuheng Ding, Bingxuan Li, Pan Lu, Da Yin, Kai-Wei Chang, and Nanyun Peng. 2025c. Visco: Benchmarking fine-grained critique and correction towards self-improvement in visual reasoning. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9527–9537.

Yuan Wu, Zongxian Yang, Jiayu Qian, Songpan Gao, Guanxing Chen, Qiankun Li, Yu-An Huang, and Zhi-An Huang. 2026b. Better eyes, better thoughts: Why vision chain-of-thought fails in medicine. arXiv preprint arXiv:2603.06665.

Yi Xu, Chengzu Li, Han Zhou, Xingchen Wan, Caiqi Zhang, Anna Korhonen, and Ivan Vulić. 2026. Visual planning: Let's think only with images. In The Fourteenth International Conference on Learning Representations.

Shuo Yang, Yuwei Niu, Yuyang Liu, Yang Ye, Bin Lin, and Li Yuan. 2025a. Look-back: Implicit visual re-focusing in MLLM reasoning. arXiv preprint arXiv:2507.03019.

Wenxi Yang, Yuzhong Zhao, Fang Wan, and Qixiang Ye. 2025b. Thinking with images via self-calling agent. arXiv preprint arXiv:2512.08511.

Chao Yi, Yuhang He, De-Chuan Zhan, and Han-Jia Ye. 2024. Bridge the modality and capability gaps in vision-language model selection. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Licheng Yu, Zhe Lin, Xiaohui Shen, Jimei Yang, Xin Lu, Mohit Bansal, and Tamara L. Berg. 2018. Mattnet: Modular attention network for referring expression comprehension. In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 1307–1315.

Qiying Yu, Zheng Zhang, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, YuYue, Weinan Dai, Tiantian Fan, Gaohong Liu, Juncai Liu, LingJun Liu, Xin Liu, Haibin Lin, Zhiqi Lin, Bole Ma, Guangming Sheng, Yuxuan Tong, Chi Zhang, Mofan Zhang, and 17 others. 2026. DAPO: An open-source LLM reinforcement learning system at scale. In The Thirty-ninth Annual Conference on Neural Information Processing Systems.

Yuexiang Zhai, Hao Bai, Zipeng Lin, Jiayi Pan, Shengbang Tong, Yifei Zhou, Alane Suhr, Saining Xie, Yann LeCun, Yi Ma, and Sergey Levine. 2024. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. In The Thirty-eighth Annual Conference on Neural Information Processing Systems.

Haoyu Zhang, Yuwei Wu, Pengxiang Li, Xintong Zhang, Zhi Gao, Rui Gao, Mingyang Gao, Che Sun, and Yunde Jia. 2026. MIRROR: Multimodal Iterative Reasoning via Reflection on Visual Regions. arXiv preprint arXiv:2602.18746.

Zhuosheng Zhang, Aston Zhang, Mu Li, hai zhao, George Karypis, and Alex Smola. 2024. Multimodal chain-of-thought reasoning in language models. Transactions on Machine Learning Research.

Mingyu Zheng, Zhifan Feng, Jia Wang, Lanrui Wang, Zheng Lin, Hao Yang, and Weiping Wang. 2025. TableDreamer: Progressive and weakness-guided data synthesis from scratch for table instruction tuning. In Findings of the Association for Computational Linguistics: ACL 2025, pages 7290–7315, Vienna, Austria. Association for Computational Linguistics.

## A Method Details

GRPO. For a given question q, we sample a group of G trajectories $\{\tau_{1},\ldots,\tau_{G}\}$ from the old policy $\pi_{\theta_{old}}$ and optimize the standard GRPO objective:

$$
\mathcal {J} _ {\mathrm{GRPO}} (\theta) = \mathbb {E} _ {q, \{\tau_ {i} \} _ {i = 1} ^ {G}} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \frac {1}{| \tau_ {i} |} \sum_ {t = 1} ^ {| \tau_ {i} |} \ell_ {i, t} (\theta) \right],
$$

where $\{\tau_i\}_{i = 1}^G\sim \pi_{\theta_{\mathrm{old}}}$ and

$$
\ell_ {i, t} (\theta) = \mathcal {L} _ {i, t} ^ {\mathrm{clip}} (\theta) - \beta \mathbb {D} _ {\mathrm{KL}} \left(\pi_ {\theta} \parallel \pi_ {\mathrm{ref}}\right).
$$

The clipped objective is defined as

$$
\begin{array}{c} r _ {i, t} (\theta) = \frac {\pi_ {\theta} (a _ {i , t} \mid s _ {i , t})}{\pi_ {\theta_ {\mathrm{old}}} (a _ {i , t} \mid s _ {i , t})}, \\ \mathcal {L} _ {i, t} ^ {\mathrm{clip}} (\theta) = \min {(r _ {i, t} (\theta) A _ {i}, \operatorname{clip} {(r _ {i, t} (\theta) , 1 - \epsilon , 1 + \epsilon)} A _ {i})}. \end{array}
$$

$D_{KL}$ is the regularization term ensuring the policy does not deviate too much from the reference model and $A_{i}$ is the advantage computed by normalizing the trajectory reward within the group:

$$
A _ {i} = \frac {r _ {i} - \operatorname{mean} (\{r _ {1} , \dots , r _ {G} \})}{\operatorname{std} (\{r _ {1} , \dots , r _ {G} \})}.
$$

Random Turn Masking (RTM) Interpretation. We can interpret RTM as a form of reweighted per-decision policy gradient. By expanding the expectation over the uniform starting index k, we observe that the contribution of each turn t to the total gradient is weighted by the probability of it being included in the suffix. Let $w_{t}$ be the weight assigned to the gradient at turn t. Since a turn t is included in the loss whenever the sampled start index $k \leq t$ , and k is sampled uniformly, the effective weight is:

$$
w _ {t} = P (k \leq t) = \sum_ {j = 1} ^ {t} \frac {1}{T} = \frac {t}{T}
$$

Substituting this back into the gradient formulation, the expected RTM gradient is equivalent to a weighted standard policy gradient:

$$
\mathbb {E} \left[ \nabla \mathcal {J} _ {\mathrm{RTM}} (\theta) \right] = \mathbb {E} _ {\tau \sim \pi_ {\theta}} \left[ \sum_ {t = 1} ^ {T} \frac {t}{T} \cdot \nabla \log \pi_ {\theta} (a _ {t} | s _ {t}) \hat {A} _ {t} \right]
$$

This suggests that RTM applies a linear weighting schedule $w_{t} \propto t$ . Later turns, which correspond to refinement and reflection steps, receive higher gradient magnitude than early turns. This implicitly prioritizes the optimization of recovery behavior over initial exploration.

Reward Function for Visual Grounding. In this work, we define potential function $\phi(d)$ based on the Euclidean distance d to the target:

$$
\phi (d) = \frac {1}{2} \left(\exp \left(- \frac {d ^ {2}}{\sigma_ {1} ^ {2}}\right) + \exp \left(- \frac {d ^ {2}}{\sigma_ {2} ^ {2}}\right)\right).
$$

We select different $\sigma$ values to ensure the model receives meaningful feedback signals when the prediction is both far from (via $\sigma_{2}$ ) and near (via $\sigma_{1}$ ) the target. We choose to use improvement based reward function as we find that when the reflection reward is only based on the distance of final prediction to the ground truth, then the model will improve its prediction in the first turn and hence lose the ability to keep the reflection behavior. There is a large space in the design of reflection rewards and finding the best reflection reward is beyond the scope of this work as the best reflection reward may be task dependent.

We define the raw, unshaped improvement-based reward $r_{\mathrm{refl}}$ based on $\phi(d)$ . We define the following shaping to obtain the final reflection reward $R_{refl}$ .

$$
R _ {\mathrm{refl}} = 0. 1 + 0. 9 \cdot \max (0, r _ {\mathrm{refl}}) \text { if } r _ {\mathrm{fmt}} = 1
$$

We give a weight of 0.9 to the reflection reward and 0.1 indicates the format reward when the format of the output is correct $r_{fmt} = 1$ . For correct predictions, note that $r_{refl}$ is capped to 1.0 given the definition of $\phi(d)$ , so the total reward R is capped to 1.0 in combination with the format reward. For incorrect predictions, we provide partial credit based on improvement, using $\max(0, r_{\text{refl}})$

to avoid over-penalizing regressive reflection attempts: empirically, allowing negative shaping reduced the model's tendency to engage in reflection, since it can always achieve the base format reward (0.1) without attempting corrective moves.

Reward Function for Spatial Navigation. Following Xu et al. (2026), we build reflection reward upon progress rate for the spatial navigation task as follows. Given a model predicted path $\hat{v}$ and M ground truth optimal paths $v_{m}$ for $m \in \{1, \ldots, M\}$ , the progress rate is computed as

$$
\mathrm{PR} = \max _ {m \in \{1, \dots , M \}} \frac {1}{n} \sum_ {j = 1} ^ {n} \left[ \prod_ {k = 1} ^ {j} \mathbb {I} \left(\hat {v} _ {k} = v _ {k} ^ {(m)}\right) \right]
$$

In other words, progress rate measures the ratio of consecutive correct steps from the start that align with at least one ground truth trajectory.

Then, the unshaped reflection reward is

$$
r _ {\text { refl }} = \operatorname{clip} \left(\sum_ {t = 2} ^ {T} w (\Delta \mathrm{PR} _ {t}), 0, 1\right)
$$

where T is the total number of turns, $\Delta PR_{t} = PR_{t} - PR_{t-1}$ , and

$$
w (\Delta \mathrm{PR}) = \left\{ \begin{array}{l l} \Delta \mathrm{PR} & \text { if } \Delta \mathrm{PR} \geq 0 \\ \lambda_ {\deg} \cdot \Delta \mathrm{PR} & \text { if } \Delta \mathrm{PR} <   0 \end{array} \right.
$$

where $\lambda_{\mathrm{deg}}$ is a hyperparameter.

In other words, the design rewards improvement towards the optimal trajectories over multiple turns of refinement, and penalize unhelpful revisions that lead to regression in progress.

We define the following shaping to obtain the final reflection reward $R_{refl}$ when the response has the correct format (i.e., $r_{fmt} = 1$ ).

$$
R _ {\text { refl }} = \left\{ \begin{array}{l l} 0. 1 + 0. 9 \cdot \max (0, r _ {\text { refl }}) + \alpha r _ {\text { refl }} - \gamma T & \text { if } r _ {\text { coord }} = 1, \\ 0. 1 + 0. 9 \cdot \max (0, r _ {\text { refl }}) - \gamma T & \text { if } r _ {\text { coord }} = 0 \end{array} \right.
$$

Where $r_{coord}$ is the correctness of the final answer, $\alpha$ is the reflection bonus coefficient, and $\gamma$ is the step cost penalty coefficient.

The first two terms (format reward and weighted raw reflection reward) are the same as the reflection shaping for visual grounding, and we added the last two terms for spatial navigation specifically: (1) Reflection bonus: if the final answer is correct, we provide the model with a reward bonus of $\alpha$ if the response uses reflections. This provides incentive for the model to perform more multi-turn reflections during training for this difficult task; (2) Step cost: to prevent the model from over-reflecting to hack the reflection bonus, we apply a step cost of $\gamma$ to the number of turns T in the response. By combining these two terms, we empirically incentivize the model to use more reflections during training for spatial navigation while not over-reflecting. We set $\alpha = 0.2$ and $\gamma = 0.05$ by default.

## B Implementation Details

## B.1 VRRL

Visual Grounding. All training experiments are conducted on Qwen2.5-VL-3B-Instruct and Qwen2.5-VL-7B-Instruct using 4 NVIDIA A100 (80GB) GPUs. For the SFT stage, we train on 15K small table header lookup data with a learning rate of 5e-6 and a global batch size of 48. For the RL stage, We utilize a learning rate of 1e-6 and a global batch size of 32 on the SFT model that is trained for 5 epochs, where the model has already developed self-reflection output format. For GRPO, we sample G = 8 rollouts per question and set the maximum trajectory length to T = 8 turns. The KL coefficient is set to $\beta = 0.01$ . For the reward function, we set the shaping parameters $\sigma_{1} = 2\delta_{tol}$ and $\sigma_{2} = 5\delta_{tol}$ (where $\delta_{tol} = 40$ ) to provide gradient signals at both coarse and fine granularities. We set the per-step cost to 0. For the training curriculum, we set the mixing probability $\rho = 2/3$ , meaning 66% of training samples come from standard on-policy generation (with RTM applied) and 33% from the replay buffer (Buffered Roll-In).

The replay buffer $\mathcal{B}$ has a maximum capacity of 500 prefixes. To prevent the model from overfitting to failure states, which could cause the policy to unlearn immediate termination, we enforce a constraint where $30\%$ of the examples in $\mathcal{B}$ are forced to be correct states (prefixes ending in a successful point). This ensures the policy practices validation (terminating when correct) alongside correction (refining when wrong). The buffer operates as a First-In-First-Out (FIFO) queue; given the small capacity, this ensures the “mistake states” remain relevant to the current policy’s capabilities.

We run RL training for 1200 steps for all of our trained models and most models converge in 600 steps. We then use the checkpoints that achieve the best performance on a 200-example holdout set on the Large Table task for all OOD evaluations.

Spatial Navigation. All training experiments are conducted on Qwen2.5-VL-3B-Instruct and Qwen3-VL-4B-Instruct using 4 NVIDIA A100 (80GB) GPUs. Following the settings and training configurations in Xu et al. (2026), we first warm-up the model with 10 epochs of SFT on the 3K single-turn examples of maps of size 3 – 5 with only the direct answer as the task is very out-of-distribution of the instruction-tuned model.

For the SFT stage of Single-SFT and Multi-SFT, we train on 4K examples from maps of size 4 - 5 as performance on $3 \times 3$ maps is already saturated after the warm-up stage. We run SFT with a learning rate of 2e-6 and a global batch size of 48.

For the RL stage, we use 2K data from maps of size 4 - 5. we apply online data filtering adapted from DAPO (Yu et al., 2026) to stabilize training by filtering out examples where all rollouts get 1 or 0 reward uniformly. For Qwen3-VL-4B, because its base capability is strong enough for multi-turn reflection, we use a smaller reflection bonus coefficient in the reflection reward to prevent reward hacking with $\alpha = 0.1$ and $\gamma = 0.01$ . We utilize a learning rate of 1e-6 for Qwen2.5-VL-3B and 5e-7 for Qwen3-VL-4B. We use a global batch size of 32 on the SFT model that is trained for 5 epochs. For GRPO, we sample $G = 8$ rollouts per question and set the maximum trajectory length to $T = 8$ turns. The KL coefficient is set to $\beta = 0.01$ . We use the same training curriculum and replay buffer configurations as the one for the visual grounding task. We use $\lambda_{\mathrm{deg}} = 0.5$ to balance between incentivizing reflections and penalizing over revisions.

We run RL training for 1200 steps for all of our trained models and most models converge in 250 steps given the strong warm-up-ed model. We then use the checkpoints that achieve the best performance on a 250-example heldout set on the in-distribution maps.

## B.2 Baselines

VL-Rethinker. We directly evaluate the official released models of VL-Rethinker (Wang et al., 2026a) with two model sizes (7B and 32B) on our evaluation tasks to test their OOD generalization capability when using textual CoT for self-reflection, since these models have already been trained on chart and diagram data during fine-tuning. At inference time, we follow Wang et al. (2026a) and prepend reflection prefixes after each assistant turn to encourage the model to reflect. We set the maximum number of reflections to 5.

Reflection Tuning. Wu et al. (2025a) proposes reflection tuning to train models to self-reflect using visual feedback through online iterative SFT, where a teacher model provides supervision signals for revising incorrect generations. The original setup in Wu et al. (2025a) focuses on GUI tasks. It first performs SFT on multi-turn self-reflection data to obtain a warm-up model, and then applies iterative SFT on top of this model. To adapt this method to our setting, we use Multi-SFT as the base model and apply 5 iterations of iterative SFT, following the same SFT-then-RL pipeline as our method. At each iteration, SFT data is created by sampling trajectories from the training prompts and collecting both correct and incorrect trajectories. For incorrect trajectories, we inject reflection by generating thinking traces together with the ground-truth answer, following Wu et al. (2025a).

## C Data Generation for Visual Grounding Tasks

In this section, we describe the pipeline for synthesizing our training and evaluation datasets. The core advantage of programmatic generation is the ability to extract perfect ground-truth spatial metadata, which we leverage to construct precise visual question-answering pairs. Examples of each setting can be found in Figure 3.

## C.1 Dataset Splits and Evaluation Tasks

We train on a single in-distribution task, Table Lookup, and evaluate on both held-out in-distribution examples and four OOD tasks that test different axes of generalization: scale, query type, and visual domain. Each test set contains 1K examples.

Table Lookup: Training and In-distribution Evaluation. We generate small arXiv-style tables with 5 to 15 rows and columns across 15 academic domains. Queries ask the model to localize row or column headers, e.g., “Find the column header ‘F1-Score’.” This task serves as both the training distribution and the in-distribution evaluation setting.

OOD-Large Tables: Size Generalization. We evaluate on substantially larger tables with 20 to 50 rows and columns, while preserving the same header-localization query format as the training task.

OOD-Cell Query: Query-Type Generalization. We evaluate on small tables but query cell contents (body text) rather than headers. This tests the model's ability to generalize the pointing mechanism to unseen query types without explicit training. Note that this represents a harder task: localizing a row or column header is effectively a 1D search problem along a single axis, whereas finding an exact inner cell requires precise 2D spatial localization.

OOD-Bar Chart: Domain Generalization. We synthetically generate bar chart images with fine-grained control over chart metadata. This evaluates the model's ability to transfer its self-reflection skills to a novel visual domain where the objective is to point to the highest, lowest bar, or any bar given a specific category label. While visually identifying these bars is straightforward, we show that precisely localizing them within the image coordinate space remains highly error-prone (Table 1).

OOD-Scatter Plot: Domain Generalization. We synthetically generate scatter plots where each plot contains 10 to 20 dots labeled with unique letters, and the model must point to the dot corresponding to a queried label. This reflects realistic data-extraction scenarios, where accurate spatial grounding of densely packed, uniquely labeled data points is an essential prerequisite for downstream tasks, such as reasoning over complex charts (Tang et al., 2025; Masry et al., 2025).

## C.2 Table Image Generation

To generate table content, we sample terminology from 15 predefined academic domains, including machine learning, computer vision, natural language processing, reinforcement learning, and bioinformatics. For each domain, we define tailored vocabularies for column groups (e.g., dataset names like ImageNet or COCO), row groups (e.g., model architectures or size categories), evaluation metrics, and baseline methods. The cell values are populated using custom generators that simulate realistic numeric formats, such as percentages, floating-point numbers, and integer scores. We ensure that every data cell within a single table contains a unique value. This prevents spatial ambiguity during evaluation, ensuring there is only one correct answer per question.

We model four distinct table structures to reflect the layout diversity found in scientific literature:

1. Single-level: A basic table with a single row of column headers and a single column of row headers.

2. Multi-column: Introduces hierarchical column groups, where an overarching category (e.g., a dataset) spans multiple sub-metrics.

3. Multi-row: Introduces hierarchical groupings along the vertical axis, categorizing specific methods under broader taxonomies.

4. Fully hierarchical: The most complex structure, combining multi-level spanning across both rows and columns.

The visual rendering of the tables is implemented programmatically using Python. The output image dimensions dynamically scale to accommodate the generated table layout. During rendering, the absolute pixel coordinates of every individual cell, header, and spanning group are recorded. Leveraging this precise spatial metadata, we synthesize templated natural language question-answer pairs that require the model to locate specific elements, such as pointing to a specific row header, a nested column header, or an individual data cell.

## C.3 Bar Chart Image Generation

We synthesize bar chart instances by independently sampling layout parameters for each example.

1. Overall layout: The number of bars is drawn uniformly randomly between 5 and 10 bars. The height of the bar chart is randomly drawn between 512 and 768 pixels, and the width is fixed at a 16:9 aspect ratio relative to the sampled height.

2. Category labels: X-axis category labels are randomly sampled from several semantic schemas, including calendar months, weekdays, fiscal quarters, years, hours, business functions, fruits, and geographic regions, to ensure labels remain concise and contextually coherent.

3. Bars: Bar values are sampled i.i.d. from Uniform(0, 100). Bar colors are assigned by cycling through a fixed palette of 15 visually distinct hues, with the number of unique colors clamped to the sampled color count.

Based on the layout above, SVG charts are rendered programmatically via a layout engine. Each SVG is subsequently rasterized to a PNG at the target resolution. Because all element positions are determined analytically, pixel-accurate ground-truth coordinates can be computed directly from the chart parameters without any post-hoc image processing. Question-answer pairs are synthesized from three types of templated natural language queries, including locating the bar with the largest or smallest value and identifying the bar corresponding to a named category. The models are prompted to answer the question by pointing to the exact pixel coordinate within the bounding box that defines the target bar.

## C.4 Scatter Plot Image Generation

Scatter plot generation begins by establishing a continuous two-dimensional coordinate system. For each plot, the visible data ranges for the x- and y-axes are determined by randomly cropping sub-spans from a broader, predefined global value range. A variable number of data points are then uniformly sampled as integer coordinates within these visible boundaries.

To prevent visual occlusion and spatial ambiguity, we employ a rejection-sampling collision avoidance mechanism. This ensures that every plotted element, both the data points and their corresponding alphanumeric text labels, occupies a strictly unique coordinate pair. If overlaps are detected between points and labels, the coordinates are iteratively resampled.

## D Data for Spatial Navigation Tasks

We directly adopt the training and evaluation setup of FrozenLake from Xu et al. (2026), training on maps with sizes ranging from 3–5 grids and evaluating on larger maps with sizes of 6–7 grids to evaluate generalization. Each map size contains 250 examples in the evaluation set.

To induce self-reflection behavior through Multi-SFT prior to RL training, we do not employ templated reasoning traces as used for visual grounding tasks, since Xu et al. (2026) show that such templates are ineffective for training VLMs on this task. Instead, we distill CoT trajectories from GPT-5.4 (Singh et al., 2026) that demonstrate corrections of randomly injected errors on the training maps and use them as self-reflective SFT data. Following Xu et al. (2026), we retain the same prompt format with ‘think’, ‘answer’, and ‘final’ tags: when the model generates an answer proposal in the first turn, it emits an ‘answer’ without justification (thinking). In the subsequent turn, the model needs to either finalize the answer with ‘final’ tag (no justification needed), or revise with ‘think’ and then a new answer proposal with ‘answer.’ This scheme preserves the direct answer capabilities of the warm-up model in the first turn while still allowing models to perform reflection and revision. We further encourage reflection by prepending an instruction that asks the model either to revise its previous answer or terminate the trajectory after receiving visual feedback.

## E Licenses

We use the following publicly available datasets from prior works with open licenses.

FrozenLake We access the FrozenLake data via Gym (Brockman et al., 2016) that uses the MIT license and data is available at: gymnasium.farama.org.

## F Output Examples

We show example outputs from our model on the OOD tasks.

![](images/7d4e656463b96c7a2274fd77d6c6cfcf65ffc8779eacf42654748c207d185c76.jpg)  
Figure 5: Example output for Large Table question.

![](images/8fd08a52cd79ba847b04e9bf3d290b15f4f17f7127ae7a98302bed0d96f48755.jpg)  
Figure 6: Figure 5 continuation.

![](images/2817510d3c90f975a15185a3e242bc2e49c2b702681ae05eb2add6dd268632f2.jpg)  
[User] Question: Locate the bar where the category is “Mango”. Return its pixel coordinate in a list format [x\_coordinate, y\_coordinate].

```json
[Assistant]
```json
{
    "think": "I need to locate the bar for the category 'Mango'. I'll point to its location.",
    "function_call": {
    "name": "pointing",
    "arguments": {
    "points": [
    {
    "description": "Mango",
    "x": 840,
    "y": 297
    }
    ]
    }
    }
}
```  
Figure 7: Example output for Bar Chart question.

```txt
Turn 2
[User]
[Assistant]
```json
{
    "think": "I'm pointing at 'Berry' instead of 'Mango'. Let me move right.",
    "function_call": {
    "name": "pointing",
    "arguments": {
    "points": [
    {
    "description": "Mango",
    "x": 940,
    "y": 315
    }
    ]
    }
}
```
Turn 3
[User]
[Assistant]
```json
{
    "think": "I'm pointing at 'Fig' instead of 'Mango'. Let me move right.",
    "function_call": {
    "name": "pointing",
    "arguments": {
    "points": [
    {
    "description": "Mango",
    "x": 1007,
    "y": 334
    }
    ]
    }
}
...
```  
Figure 8: Figure 7 continuation.

![](images/738590dbbe946d66e8e052068ae7ce51ed98948b2c96665f497ae0ed71281803.jpg)  
Figure 9: Figure 8 continuation.

![](images/7e24784726bd6502c7fe7eb4fc0d99859858c5b1582dd3d9868a88678d737244.jpg)  
Figure 10: Example output for FrozenLake question.

![](images/d73a2392f5f09b68172eba29a4cb04c83a15546a78715166c0ba0801af6d22da.jpg)  
Figure 11: Figure 10 continuation.