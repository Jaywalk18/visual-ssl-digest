# Groc-PO: GROunded Context Preference Optimization for Truthful Multimodal LLMs

Zhixiao Zheng zhixiao.zheng@mail.ustc.edu.cn University of Science and Technology of China Hefei, China

Chunxiao Liu chunxiao6liu@gmail.com Xiaomi Corporation Beijing, China

Zheren Fu<sup>∗</sup> fzr@ustc.edu.cn University of Science and Technology of China Hefei, China

Dongming Zhang zhangdongming@people.cn State Key Laboratory of Communication Content Cognition, People’s Daily Online Beijing, China

Zhiyuan Yao yaozhiyuan@mail.ustc.edu.cn University of Science and Technology of China Hefei, China

Zhendong Mao zdmao@ustc.edu.cn University of Science and Technology of China Hefei, China

## Abstract

Despite the rapid progress of Multimodal Large Language Models (MLLMs), they still sufer from untruthfulness issues, such as visual hallucinations, content fabrication, and unfaithful reason ing, which substantially undermine their faithfulness and practical utility. Alignment methods based on human preference, such as Direct Preference Optimization (DPO), have been widely adopted to address these issues. However, multimodal reasoning errors of ten propagate across stages, and final-answer errors can often be traced to mistakes in early grounding stages, yet standard DPO typically applies preference optimization at the final-answer level. This credit-assignment challenge means that supervision for early grounding stages is indirect rather than stage-specific, making it dificult to suppress error propagation arising from grounding drift and context inconsistency. To address this, we propose Grounded Context Preference Optimization (Groc-PO), a grounded prefer ence optimization framework for MLLMs. We further construct the Grounded Context Preference Dataset (GCPD), organizing multi stage preference samples around three stages of Object Grounding, Contextual Grounding, and Grounded Reasoning, to capture the formation, integration, and utilization of grounded context. By introducing more explicit preference supervision over multiple grounded stages, Groc-PO strengthens context-dependent reason ing and mitigates cross-stage error propagation. Extensive experi ments show that, compared with standard DPO and other strong baselines, Groc-PO achieves improved performance in hallucination mitigation, faithful reasoning, and overall reliability, supporting the value of more explicit grounded supervision for trustworthy multimodal reasoning.

![](images/677308f9be71907b6ffe94a0a05160e3b3618b9b49d9bca3518b54d67b149f79.jpg)  
What is the man on the right holding, and what does this suggest about his intention ?

(a)  
![](images/c62a88d10dd01607e4c63880c29761152b014d199f727b609c42bc8aff1a6435.jpg)  
(b)  
Figure 1: Motivating example of error propagation across stages in MLLMs. (a) A case where an early grounding error propagates to the later reasoning stage and leads to an incorrect answer. (b) Statistical experiments with LLaVA-v1.5- 7B [20] on GCPD dataset (constructed from RLHF-V [35]), showing that introducing errors into 0, 1, or 2 grounding stages is associated with progressively lower final reasoning accuracy, consistent with error propagation in MLLMs.

## CCS Concepts

• Computing methodologies → Machine learning.

## Keywords

medical analysis [24, 32, 36]. However, despite these advancements, MLLMs sufer from the unfaithfulness problems, such as generat ing content that contradicts visual facts (e.g., fabricated objects, attributes, or relationships) [5, 10]. This significantly hinders their reliability and utility in real-world applications [18, 22].

To mitigate the above issues, preference learning based on human feedback has become a mainstream alignment paradigm. Among these methods, Direct Preference Optimization (DPO) [25] has been widely adopted because it directly optimizes the policy from prefer ence data without requiring an explicit reward model. Existing DPO methods apply preference supervision at the final-answer-level to guide the model in generating more accurate and reliable outputs. However, multimodal reasoning often involves several stages, in cluding early grounding and later reasoning, and the quality of the final answer depends on the robustness of the full process. As illus trated in Fig. 1a, an error arising in an early grounding stage can propagate to later reasoning stages and lead to reasoning failure. Therefore, it remains unclear whether standard DPO, which mainly relies on holistic final-answer-level preferences, can adequately address reasoning errors rooted in early grounding stages.

To this end, we conduct further controlled explorations. As shown in Fig. 1b, introducing errors into more grounding stages is associated with lower final reasoning accuracy, indicating that errors can propagate and accumulate across stages along the reasoning process. This observation reveals a key structural limitation of final-answer-level preference optimization, namely the propa gation of MLLM errors: later reasoning failures often do not arise solely at the answer-generation stage, but instead originate from early grounding stages and then accumulate along the reasoning process, afecting the final reasoning outcome. Yet standard DPO typically supervises only at the final-answer-level, ofering holistic guidance while providing little stage-specific supervision for these early grounding stages.

Consequently, an important question for MLLMs preference alignment is how to provide more targeted preference supervi sion for early grounding stages, so as to reduce error propagation and improve faithful multimodal reasoning. To address these prob lems, we propose Grounded Context Preference Optimization (Groc-PO), a grounded preference optimization framework for MLLMs. Built upon the newly constructed Grounded Context Preference Dataset (GCPD), Groc-PO leverages preference signals from three contextual stages: Object Grounding, Contextual Grounding, and Grounded Reasoning. This design provides more explicit supervi sion for upstream grounded stages, thereby enhancing the model’s ability to robustly construct and faithfully utilize the grounded context. The framework incorporates the full context and adopts an adaptive stage-aware optimization strategy, improving complex reasoning while mitigating the propagation of upstream errors. Ex tensive experiments demonstrate that Groc-PO yields notable gains in hallucination mitigation, contextual understanding, and faithful reasoning. The main contributions of this paper are as follows:

• We propose the Groc-PO framework. By introducing explicit supervision over grounded pre-final stages, it improves faith ful multimodal reasoning under multi-round contexts and efectively mitigates error propagation in MLLMs.

• We construct the Grounded Context Preference Dataset (GCPD). GCPD organizes multi-stage preference data around Visual Grounding, Context Grounding, and Faithful Complex Reasoning, providing support for explicit stage-wise supervision of grounded context.

• Built on GCPD, Groc-PO employs an adaptive grounded preference optimization mechanism that dynamically allocates learning emphasis across diferent stages and sample complexities, enabling more targeted alignment.

• We conduct systematic experiments across multiple datasets and benchmarks. The results show that Groc-PO consistently outperforms standard DPO and several strong baselines on hallucination and complex capability evaluations, validating the efectiveness of explicit supervision for pre-final grounded stages.

## 2 Related Works

## 2.1 Unfaithfulness in MLLMs

Unfaithfulness (or hallucination) in MLLMs refers to the generation of content inconsistent with the visual input, typically manifested as fabricated objects, incorrect attributes, or misinterpreted relationships [5]. Hallucinations stem from training data flaws [16]; module biases [9]; suboptimal training paradigms [6]; and inference-stage defects [12]. To address hallucinations, approaches fall into two categories: training-free methods (e.g., Opera [12], VCD [14]) and training-based techniques (e.g., RLHF, PPO) [26, 33].

## 2.2 Preference Learning for faithful MLLMs

Preference learning initially applied to LLM alignment via RLHF, but DPO has recently gained widespread adoption as a simpler and stable alternative. V-DPO [34] extends DPO by incorporating visual context learning. POVID [38] creates a fine-grained dataset by injecting noise to texts and images. RLHF-V [35] collects segmentlevel human preference data and performs dense DPO. In addition, SPO-Task Planning [17] constructs preference pairs using curriculum learning to improve long-horizon planning. MM-RLHF [37] constructed a preference dataset and proposed a novel reward model to achieve MLLM alignment. SPO [27] treats questioning and answering jointly as a policy trajectory, co-optimizing them via a structured reward function to enhance the model’s consideration of visual dependency in dialogue. The mrDPO [29] utilizes multiround DPO and Rebirth Tuning to optimize audio-visual LLMs.

## 3 Methodology

The overview of our Groc-PO framework is illustrated in Fig. 2. We first introduce the DPO, followed by the novel Grounded Context Preference Dataset (GCPD) and its generation pipeline, and the adaptive Groc-PO Loss.

## 3.1 Preliminaries: Direct Preference Optimization

DPO directly optimizes the model through a contrastive learning objective, making it more inclined to generate human-preferred responses while reducing the probability of generating dispreferred responses. DPO learns from preference data $( x , y ^ { + } , y ^ { - } ) \sim \mathcal { D }$ , where <sup>??</sup> is the input prompt, $y ^ { + }$ is the human-preferred /chosen response, $y ^ { - }$ is the dispreferred /rejected response, and D is the dataset.

![](images/8169e40208d8db1da04eecce470856fc68d26737dd57ab5f20bef10a3123f4bf.jpg)  
Figure 2: Overview of our Grounded Context Preference Optimization (Groc-PO) framework, including GCPD dataset construction. The left panel shows dataset construction, where multi-stage preference pairs are generated through teacher-assisted drafting, model-centric sampling, iterative correction, and human verification. The middle panel presents three stages of grounded preference supervision: Stage 1 for object grounding, Stage 2 for contextual grounding, and Stage 3 for grounded reasoning. The right panel shows Groc-PO training, which jointly uses preference pairs from all three stages with a stage-aware, hardness-aware loss to improve context-dependent reasoning and mitigate cross-stage error propagation.

The DPO objective function assumes that the human preference probability $p ^ { * } ( y ^ { + } \succ y ^ { - } \mid x )$ can be modeled via a latent reward function $r ^ { * } ( x , y ) \colon p ^ { * } ( y ^ { + } \succ y ^ { - } \mid x ) = \sigma ( r ^ { * } ( x , y ^ { + } ) - r ^ { * } ( x , y ^ { - } ) )$ . DPO further relates the reward function to the model’s policy <sup>??</sup>?? and a reference policy $\pi _ { \mathrm { r e f } } \colon r \ast ( x , y ) = \beta ( \log ( \pi _ { \theta } ( y \mid x ) ) - \log ( \pi _ { \mathrm { r e f } } ( y \mid x ) ) )$

where $\beta$ is a hyperparameter controlling the ratio between reward function and policy deviation. DPO’s loss can directly optimize MLLM to maximize the probability of generating $y ^ { + }$ and minimize generating $y ^ { - }$ . Let us define the log-likelihood ratio for the preferred response as $r ^ { + } = \log \left( \pi _ { \theta } ( y ^ { + } \mid x ) / \pi _ { \mathrm { r e f } } ( y ^ { + } \mid x ) \right)$ and for the dispreferred response as $r ^ { - } = \log \left( \pi _ { \theta } ( y ^ { - } \mid x ) / \pi _ { \mathrm { r e f } } ( y ^ { - } \mid x ) \right)$ ). Then the DPO loss function is defined as:

$$
\mathcal {L} _ {\mathrm{DPO}} = - \log \sigma \left(\beta \left(r ^ {+} - r ^ {-}\right)\right).\tag{1}
$$

By minimizing this loss function, the model <sup>??</sup>?? is trained to increase the diference between the log-probabilities of $y ^ { + }$ and $y ^ { - }$ , It makes DPO simpler and demonstrates comparable or superior performance to RLHF.

## 3.2 Grounded Context Preference Dataset (GCPD)

Failures in multimodal reasoning often do not emerge only at the final response, but can originate from imperfect grounding and accumulated inconsistencies in pre-final stages. To explicitly super vise these upstream stages, we construct the Grounded Context Preference Dataset (GCPD), a structured preference dataset or ganized around progressively accumulated grounded context.

Specifically, GCPD is built as a 3-stage context-dependent preference dataset. Its three stages move from basic visual grounding, to context-grounded understanding, and finally to faithful complex reasoning. This formulation places preference supervision not only on the final answer, but also on the grounded context that supports it, providing a more direct signal for reducing error propagation and improving multimodal faithfulness.

Cumulative Multi-stage Context: For any stage <sup>??</sup>, the prompt includes all historical context from stage 1 to <sup>??</sup> − 1, ensuring continuous information flow. Let I denote the image, Q denote the question, A denote the answer, $A _ { s } ^ { + }$ means the chosen response (human-preferred). Then, we will have the following prompt structure for every stage:

• Stage 1 (S1): Prompt = {<sup>??,</sup> <sup>??</sup><sub>1</sub>}

• Stage 2 (S2): Prompt = {<sup>??,</sup> <sup>??</sup><sub>1</sub><sup>,</sup> <sup>??+</sup><sub>??</sub> <sup>,</sup> <sup>??</sup><sub>2</sub>}

• Stage 3 (S3): Prompt = {<sup>??,</sup> <sup>??</sup><sub>1</sub><sup>,</sup> <sup>??+</sup><sub>??</sub> <sup>,</sup> <sup>??</sup><sub>2</sub><sup>,</sup> <sup>??+</sup><sub>??</sub> <sup>,</sup> <sup>??</sup><sub>3</sub>}

Then, we detail the progressive stages design:

• S1: Object Grounding. S1 marks the starting point of CoT process—identifying basic facts. We present the model with a standardized question (e.g., List each entity and its key attributes) to extract core visual elements.

• S2: Contextual Grounding. S2 simulates the intermediate steps of CoT. Building upon S1, S2 focuses on tasks such as relationship description, comprehensive captioning, or visual question answering. It requires the model not only to identify individual entities but also to understand how they form a meaningful whole.

• S3: Grounded Reasoning. S3 is the culmination of the CoT simulation. We pose complex questions that require integrating the image with context from S1 and S2, and performing logical inference, intent prediction, or reasoning tasks. This compels the model to perform high-level cognition based on established, reliable context.

In this way, our GCPD dataset is no longer fragmented question answer pairs but ofers a progressive learning process aligned with human cognitive laws.

## 3.3 Pipeline of GCPD’s Generation and Features

3.3.1 Basic Workflow. Our pipeline begins with a widely recog nized RLHF-V dataset [35], which contains 5,733 images along with human-annotated, high-quality preference pair for tasks like cap tioning, relational description. Our goal is to generate a structured 3-stage context for each image. For any stage <sup>??</sup>, the core tasks is to generate a specific question $Q _ { s }$ and a high-quality preference pair $( y _ { s } ^ { + } , y _ { s } ^ { - } )$ , consisting of a chosen and a rejected response. This three-stage structure is designed with progressively increasing com plexity, following a perception-understanding-reasoning path.

For the Stage 1, we define a universal base question $\mathbf { Q } _ { 1 }$ : “List each entity and its key attributes,” to establish a factual foundation. The chosen response, $y _ { s _ { 1 } } ^ { + }$ , is initially generated by an advanced teacher model; the rejected response, $y _ { s _ { 1 } } ^ { - } ,$ is derived from $y _ { s _ { 1 } } ^ { + }$ via introduced rule-based deficiencies, retaining structural soundness while including factual inaccuracies. Subsequently, $y _ { s _ { 1 } } ^ { + }$ undergoes rigorous verification and refinement to ensure its accuracy.

For the Stage 2, aiming for both eficiency and quality, we adopt the existing data from the RLHF-V dataset corresponding to each image, as its design philosophy aligns perfectly with our goals for this stage. This means that the question $Q _ { 2 } ,$ , the chosen response $y _ { s _ { 2 } } ^ { + }$ and the rejected response $y _ { s _ { 2 } } ^ { - }$ are all sourced from this high-quality, human-validated dataset. It guarantees the superiority of the data for relational description and understanding.

In the Stage 3, our objective is to enhance model’s ability of complex thinking. We leverage an advanced teacher model to generate a new, more profound, and complex question, $Q _ { 3 }$ , based on the context of the original image and the preceding dialogue $( Q _ { 1 } , y _ { s _ { 1 } } ^ { + } , Q _ { 2 } , y _ { s _ { 2 } } ^ { + } )$ . Following the pattern of the 1st stage, we then generate a high-quality chosen response, $y _ { s _ { 3 } } ^ { + }$ , via the teacher model combined with rigorous verification and refinement, and a rejected response, $y _ { s _ { 3 } } ^ { - }$ , from our LLaVA base model to complete the final preference pair.

3.3.2 Iterative Self-Correction for Chosen Samples $( y _ { s } ^ { + } ) .$ To maxi mize the chosen response quality and factual accuracy, we introduce an iterative self-correction mechanism.

• In Stage 1, the initial list of entities in the chosen response, generated by the advanced teacher model, is fed back into the teacher model with a detailed verification prompt. The teacher model is instructed to comprehensively check and provide a score. If the response contains hallucinations or the average score is too low, it triggers a rewrite by the advanced teacher model.

• Similarly, in Stage 3, the complex reasoning answer in the chosen response is sent back for a second review. In this step, the teacher model acts as ${ \mathrm { ~  ~ a ~ } } " { \mathrm { C r i t i c } } "$ , inspecting the reasoning chain for logical fallacies and ensuring it is fully grounded in the provided visual and textual context.

This "generate-and-refine" closed-loop process significantly enhances the quality of our chosen responses, providing the model with a clear and reliable learning target.

3.3.3 Model-Centric Sampling. To further improve data quality and make training more eficient, we draw inspiration from the on-policy concept in RLHF [7] and adopt a Model-Centric Sampling strategy. The core idea is to ensure that the distribution of training data, aligns as closely as possible with the generation distribution of our targeted fine-tuned model. This approach enables the model to directly confront and rectify its own predominant error patterns, making the fine-tuning process highly targeted. This strategy is reflected in two key aspects:

• The rejected responses in Stage $3 ~ ( y _ { s _ { 3 } } ^ { - } )$ are generated by our target LLaVA model. Consequently, these samples are representative of the model’s intrinsic failure modes, particularly in areas like long-range dependency, contextual understanding, and complex reasoning, which manifest as logical fallacies or cumulative hallucinations. In contrast to negative samples from a more capable, external model teacher model, which often sufer from a distribution mismatch in $y _ { r } ^ { - }$ . These ’model-centric’ samples provide a highly targeted and valuable learning signal for DPO.

• In Stage 2, a key characteristic of the RLHF-V dataset lies in its preference pair construction: the rejected responses $( y _ { s _ { 2 } } ^ { - } )$ are generated by the MLLM family, while the chosen responses $( y _ { s _ { 2 } } ^ { + } )$ are human-revised versions of these same rejected samples. This approach ensures high distributional and stylistic alignment with our target model, efectively forming a tailored "problem-solution" paradigm for its specific weaknesses. This mechanism provides the highly valuable and targeted learning signal that is the core rationale for our adoption of this dataset.

It is worth noting that the Stage 1 responses are highly uniform, making generation variance across models minimal, and thus Model-Centric Sampling has little impact.

3.3.4 Human-in-the-Loop Verification. To ensure the rigor and quality of our GCPD dataset, we introduced Human-in-the-Loop verification. The audit team consisted of three MLLM-familiar PhD students who adhered to a guideline for all checks and corrections. The samples reviewed included: first, a 10% random sample of the entire dataset; second, a targeted review of $y _ { s } ^ { + }$ samples flagged by the teacher model Critic as having major issues during the "Iterative Self-Correction" process.

Overall, approximately 12% (∼2k) of the $y _ { s } ^ { + }$ samples were manually audited, leading to the revision or rewriting of nearly 2% of severely problematic samples (primarily in R3). The total time cost was approximately 57 hours per reviewer (30h for auditing, 27h for revision). This mechanism ensures the reliability and faithfulness of the dataset.

## 3.4 Customized Groc-PO Loss

The standard DPO loss (Equation 1) treats all samples in the dataset equally. This uniform approach overlooks the inherent gradient of cognitive depth and sample dificulty within our GCPD dataset. To better leverage this rich and structured information, we propose a Groc-PO Loss, with a sample-level adaptive weight, $w _ { i } ,$ , enabling the model to dynamically focus on samples that are more informative and have higher learning value. The Groc-PO loss is defined as:

Table 1: Performance comparison with leading methods on various hallucination and general benchmarks. On LLaVA-v1.5-7B, LLaVA-v1.5-13B [20], Groc-PO achieves significant leads on key faithfulness metrics (e.g., AMBER, MM-Hal) and simultaneously enhances general abilities (e.g., LLAVA-Bench, SEED). Here, AMBER-Gene. refers to AMBER-Generation, and AMBER-Discri. denotes the AMBER-Discrimination.

<table><tr><td rowspan="2">Methods</td><td colspan="2">MM-Hal [28]</td><td colspan="3">AMBER-Gene. [31]</td><td colspan="2">AMBER-Discri. [31]</td><td rowspan="2">LLaVA [21]</td><td rowspan="2">SEED [15]</td></tr><tr><td>Score↑</td><td>Hal-Rate↓</td><td>CHAIR↓</td><td>Hal-Rate↓</td><td>Cog↓</td><td>Acc↑</td><td>F1↑</td></tr><tr><td>LLaVA-1.5-7B [20]</td><td>2.01</td><td>61.4</td><td>7.8</td><td>36.4</td><td>4.2</td><td>71.7</td><td>74.3</td><td>65.6</td><td>66.1</td></tr><tr><td>+ DPO [25]</td><td>2.14</td><td>58.3</td><td>5.7</td><td>27.3</td><td>2.6</td><td>71.3</td><td>82.1</td><td>69.1</td><td>66.4</td></tr><tr><td>+ CSR [39]</td><td>2.05</td><td>60.4</td><td>5.4</td><td>25.5</td><td>2.6</td><td>73.2</td><td>76.1</td><td>68.9</td><td>65.9</td></tr><tr><td>+ POVID [38]</td><td>2.26</td><td>55.2</td><td>5.7</td><td>26.9</td><td>3.0</td><td>71.9</td><td>74.7</td><td>68.2</td><td>66.1</td></tr><tr><td>+ V-DPO [34]</td><td>2.16</td><td>56.0</td><td>5.6</td><td>27.3</td><td>2.7</td><td>-</td><td>81.6</td><td>-</td><td>-</td></tr><tr><td>+ RLHF-V [35]</td><td>2.02</td><td>60.4</td><td>5.5</td><td>26.3</td><td>2.5</td><td>74.8</td><td>78.5</td><td>68.0</td><td>66.1</td></tr><tr><td>+ mDPO [30]</td><td>2.39</td><td>54.0</td><td>4.4</td><td>24.5</td><td>2.4</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>+ Groc-PO (Ours)</td><td>2.76</td><td>47.0</td><td>4.2</td><td>25.2</td><td>1.5</td><td>78.0</td><td>85.0</td><td>72.9</td><td>67.1</td></tr><tr><td>LLaVA-1.5-13B [20]</td><td>2.38</td><td>53.1</td><td>7.0</td><td>33.1</td><td>3.3</td><td>71.4</td><td>73.1</td><td>73.1</td><td>68.2</td></tr><tr><td>+ DPO [25]</td><td>2.47</td><td>51.0</td><td>6.1</td><td>26.3</td><td>2.7</td><td>71.9</td><td>82.1</td><td>72.8</td><td>68.6</td></tr><tr><td>+ RLHF-V [35]</td><td>2.50</td><td>52.1</td><td>6.3</td><td>25.1</td><td>2.1</td><td>79.2</td><td>82.3</td><td>76.7</td><td>68.2</td></tr><tr><td>+ Groc-PO (Ours)</td><td>2.85</td><td>45.0</td><td>3.8</td><td>24.4</td><td>1.3</td><td>83.5</td><td>88.2</td><td>76.5</td><td>68.8</td></tr></table>

$$
\mathcal {L} _ {\mathrm{Groc-PO}} = - \mathbb {E} _ {(x _ {i}, y _ {w, i}, y _ {l, i}) \sim \mathcal {D}} \left[ w _ {i} \cdot \log \sigma (r _ {i} (\theta)) \right]\tag{2}
$$

where $\begin{array} { r } { r _ { i } ( \theta ) = \beta \log \frac { \pi _ { \theta } ( y _ { i } ^ { + } | x _ { i } ) } { \pi _ { \mathrm { r e f } } ( y _ { i } ^ { + } | x _ { i } ) } - \beta \log \frac { \pi _ { \theta } ( y _ { i } ^ { - } | x _ { i } ) } { \pi _ { \mathrm { r e f } } ( y _ { i } ^ { - } | x _ { i } ) } } \end{array}$ is the implicit reward diference for preference pair <sup>??</sup>. The core innovation lies in the design of adaptive weight $w _ { i } ,$ which is composed of two multi plicative components:

$$
w _ {i} = \lambda_ {r (i)} \cdot \gamma_ {i}\tag{3}
$$

3.4.1 Stage-aware Importance Weight $( \lambda _ { r } ) .$ This weight is designed to reflect the learning value of diferent rounds. Later rounds repre sent more complex tasks that demand stronger long-range depen dency and comprehensive abilities. Thus, for any stage <sup>??</sup>, we design $\lambda _ { s }$ as a monotonically increasing function of <sup>??</sup> to encourage model to focus on these advanced knowledge:

$$
\lambda_ {r} = 1 + \alpha (r - 1)\tag{4}
$$

where $r \in \{ 1 , 2 , 3 \}$ is the dataset stage, and <sup>??</sup> $\geq 0$ is a hyperparame ter that controls the growth rate of stage importance. When $\alpha > 0 _ { : }$ samples from the latter 2 stages are assigned a higher loss weight.

3.4.2 Hardness-aware Focusing Weight (<sup>??</sup>??). This weight aims to make model focus more on "hard samples" that are dificult to distinguish. When the model can easily distinguish between $y ^ { + }$ and $y ^ { - }$ , the sample is "easy" and has low learning value. Conversely, when model perceives two responses as having similar quality, the sample is "hard" and should be prioritized. We define $\gamma _ { i }$ as:

$$
\gamma_ {i} = \big (1 - \sigma (r _ {i} (\theta)) \big) ^ {\eta}\tag{5}
$$

where $\eta \geq 0$ is a focusing parameter. For $\eta > 0$ , this term signif icantly decreases the loss for well-distinguished samples (where $\sigma ( r _ { i } )  1 ) ;$ , thereby directing the optimization process toward the most challenging pairs.

Through this dual-weighting mechanism, our Groc-PO Loss adaptively evaluates the importance of each training sample, considering both its role in the progressive rounds (via $\lambda _ { t } )$ and sample’s learning value (via $\gamma _ { i } )$

## 4 Experiments

## 4.1 Datasets, Metrics and Implementation Details

Training Data: Based on our GCPD dataset generation pipeline described in Section 3.2, we constructed the 3-stages preference dataset comprising 5,733 diverse images and 17,199 high-quality preference pairs.

Evaluation Benchmarks: To comprehensively evaluate models’ performance, we employ several widely used benchmarks: For faithfulness evaluation, AMBER [31] is a LLM-free benchmark for evaluating hallucinations, which has two components: (a) Discrimination: deciding whether a statement is correct; (b) Generation: describing for an image. MM-Hal [28] evaluates response-level hallucination rate and informativeness. For general capability evaluation, LLaVA-Bench [21] is a benchmark spanning diverse scenarios. SEED-Bench [15] is a large-scale benchmark to assess model abilities likes visual understanding and reasoning. For complex tasks, some sub-tasks of benchmarks are adopted, such as LLaVA-Bench-complex reasoning [21], LLaVA-Bench-conversation [21], SEEDvisual reasoning [15], MME-commonsense-reasoning [8], and multiturn dialogue benchmark MM-MT [1].

Implementation Details: Our experiments leverage the widely adopted LLaVA-v1.5-7B and 13B [20] and Qwen2.5-VL-7B [4] models to evaluate scalability and efectiveness of our method. We employed LoRA [11] and AdamW optimizer [23]. Training was performed over 2 epochs with an efective batch size of 32. The teacher model we used is GPT-4o [13].

Table 2: Comparison with diferent preference optimization methods on Qwen2.5-VL-7B [4].

<table><tr><td>Method</td><td>MM-Hal Score (↑)</td><td>MM-Hal Rate (↓)</td></tr><tr><td>Qwen2.5-VL-7B [4]</td><td>3.55</td><td>0.40</td></tr><tr><td>+ DPO [25]</td><td>3.59</td><td>0.38</td></tr><tr><td>+ POVID [38]</td><td>3.73</td><td>0.37</td></tr><tr><td>+ CSR [39]</td><td>3.71</td><td>0.41</td></tr><tr><td>+ Groc-PO (Ours)</td><td>3.83</td><td>0.32</td></tr></table>

Table 3: Comparison of Groc-PO and DPO on complex tasks. Tasks cover reasoning (LLaVA-Bench-complex reasoning, SEED-visual reasoning, MME-commonsense-reasoning), single-turn conversation (LLaVA-Bench-conversation), and multi-turn dialogue (MM-MT ). Groc-PO shows superiority.

<table><tr><td>Models</td><td>LLaVA-complexreason [21]</td><td>LLaVA-conversation[21]</td><td>SEED-reason[15]</td><td>MME-reason(%) [8]</td><td>MM-MT[1]</td></tr><tr><td>DPO</td><td>56.9</td><td>62.0</td><td>74.3</td><td>38.3</td><td>1.88</td></tr><tr><td>Groc-PO</td><td>82.5(+45%)</td><td>67.2(+8%)</td><td>76.5(+3%)</td><td>51.8(+35%)</td><td>2.55(+36%)</td></tr></table>

## 4.2 Main Results

On multiple mainstream faithfulness and general benchmarks, we conduct a comprehensive comparison of Groc-PO with a series of representative baselines, such as llava-v1.5-7B [20], DPO [25], CSR [39], POVID [38], RLHF-V [35], V-DPO [34], and mDPO [30].

On Faithfulness and General Abilities Evaluation. Table 1 shows that Groc-PO achieves leading performance across almost all key evaluation metrics. These evaluations cover diferent types of faithfulnesss and general ability tests. These demonstrate that through our progressive preference data and adaptive training framework, the model can not only significantly suppress halluci nations but also improve comprehensive abilities

Furthermore, Groc-PO shows its scalability by delivering consistent and substantial gains across models from 7B to 13B, validating its eficiency and broad applicability.

In addition, to validate the scalability and generalizability of our framework, we applied Groc-PO to Qwen2.5VL-7B [4]. As detailed in Table 2, the model achieved improvements on the faithfulness test.

On Complex Understanding and Reasoning. Table 3 demon strates the particularly prominent superiority of Groc-PO in com plex reasoning tasks. Across demanding benchmarks, including LLaVA-bench-complex, LLaVA-conversation, SEED-reasoning, and MME-Commonsense-reasoning, our model consistently outper forms the baseline, achieving up to a 45% relative improvement.

Furthermore, Groc-PO demonstrates superior conversational ca pabilities on MM-MT benchmark (multi-round dialogue), validating the efectiveness of its structured context.

Table 4: Ablation study of the loss components and hyperparameters (<sup>??,</sup> <sup>??</sup>) on MM-Hal [28], as described in Section 3.4.

<table><tr><td>Loss Setting</td><td> $\alpha$ </td><td> $\eta$ </td><td>MM-Hal Score (↑)</td><td>MM-Hal Rate (↓)</td></tr><tr><td colspan="5">(A) Component Ablation</td></tr><tr><td>DPO Loss</td><td>0</td><td>0</td><td>2.24</td><td>60.0</td></tr><tr><td>+ Stage-aware</td><td>0.25</td><td>0</td><td>2.46</td><td>55.0</td></tr><tr><td>+ Difficulty-aware</td><td>0</td><td>2</td><td>2.42</td><td>57.0</td></tr><tr><td>+ Groc-PO Loss</td><td>0.25</td><td>2</td><td>2.76</td><td>47.0</td></tr><tr><td colspan="5">(B) Sensitivity to  $\alpha$ </td></tr><tr><td>w/  $\alpha = 0$ </td><td>0</td><td>2</td><td>2.42</td><td>57.0</td></tr><tr><td>w/  $\alpha = 0.25$ </td><td>0.25</td><td>2</td><td>2.76</td><td>47.0</td></tr><tr><td>w/  $\alpha = 0.5$ </td><td>0.5</td><td>2</td><td>2.40</td><td>55.5</td></tr><tr><td colspan="5">(C) Sensitivity to  $\eta$ </td></tr><tr><td>w/  $\eta = 0$ </td><td>0.25</td><td>0</td><td>2.46</td><td>55.0</td></tr><tr><td>w/  $\eta = 1$ </td><td>0.25</td><td>1</td><td>2.57</td><td>51.0</td></tr><tr><td>w/  $\eta = 2$ </td><td>0.25</td><td>2</td><td>2.76</td><td>47.0</td></tr></table>

## 4.3 Ablation Study

4.3.1 Contribution of Groc-PO Loss Components. To verify the efectiveness of our Groc-PO Loss, we compared it against 3 variants: (1) DPO Loss: uses GCPD data with standard DPO loss; (2) Stage-aware Only; and (3) Dificulty-aware Only. Table 4 shows that the full Groc-PO Loss achieves the best performance. The individual components each provide significant gains.

In addition, We also performed ablation studies on the key hyperparameters of the loss, <sup>??</sup> (Stage-aware) and <sup>??</sup> (Hardness-aware).

4.3.2 Efect of Model-Centric Sampling. We conduct an ablation of the data construction strategy, comparing a model trained solely on Teacher-generated preference data with one trained on our final dataset using "model-centric sampling". Figure 3a shows that the latter performs better, suggesting that learning from its own imperfect responses provides a closer data distribution and more targeted alignment signals, thereby improving self-alignment.

4.3.3 Contribution of Each Stage. To quantify the contribution of each context stage, we established three independent training settings: S1, S2, and S3. These models were exclusively trained on data from their respective stages, lacking historical context in S2 and S3.

Figure 3b revealed that S2 outperformed S1 and S3. Because S3 relies on preceding information, the absence of context leads to misalignment and performance decline. This afirms the necessity of collaboration among three context rounds, asserting that optimal performance requires structured integration.

4.3.4 Impact of Multi-stage Context Depth. To investigate the influence of contextual learning depth, we used three training settings: (1) using the 1st-round data (S1-Only); (2) using the data of 1st and 2nd rounds (S1+S2); and (3) using full data of 3 stages. Figure 3c shows the model performance monotonically improves with increasing context depth and complexity.

This demonstrates that our designed "perception→understanding→reasoning" progressive learning path is indispensable for building model’s compressive capabilities and faithfulness.

![](images/5caaf9996395388c11da728c885c5f996b0e304a1a6f781197e29dadd9b450ba.jpg)  
(a) Model-Centric Sampling or Not

![](images/75c8249877c677129409ac19f98d476aaaf2191c3e4e561d30565d1f7314703e.jpg)  
(b) Contribution of each Round

![](images/48bcedaf9f6a8cf1d4b3dff02376b2249645304a15600a68dd387d2c3360bb27.jpg)  
(c) Impact of Context Depth

![](images/b032d00951df79b968d92c94d600c2b4f8a2412e1445341bfb99b73c07598bc1.jpg)  
(d) Sensitivity to Context History

Figure 3: Ablations and Analysis. (a) Efectiveness of Model-Centric Sampling or not. (b) Contribution of individual stages: S2 peaks while S3 degrades from misalignment without history. (c) Impact of context depth: monotonic improvement with progressive history. (d) Sensitivity to history length: performance on a fixed stage-3 query improves as context is added.  
![](images/645fd7d87762fb5f0fa4a8c7495b62748b961fcc0e0222eeb6f22929197681cd.jpg)  
Complex Question: What is the setting or environment in which the image takes place?

![](images/0b6a269866ba89761a5b60c4bf38dee141ceb2448b46fad4f91040fcd6725d7a.jpg)  
Figure 4: Token-level log-probability (log-prob) comparison on chosen responses <sup>??+</sup><sub>??</sub> for complex tasks: Groc-PO vs. DPO. The curves (green:Groc-PO, brown:DPO) show log-prob assigned to each token. The top yellow bar chart illustrates the diference (Groc-PO - DPO), which is almost entirely positive. This suggests Groc-PO exhibits higher internal confidence on complex tasks.

Table 5: Comparison of training with and without grounded context. The "Flattened DPO" (the same 3-stage 17k dataset but without grounded context) shows lower faithfulness than Groc-PO (with grounded context).

<table><tr><td>Setting</td><td>MM-Hal Score(↑)</td><td>Hal-Rate(↓)</td></tr><tr><td>Flattened DPO</td><td>2.17</td><td>61.0</td></tr><tr><td>Full Structured Context</td><td>2.76</td><td>47.0</td></tr></table>

Table 6: Training Overhead Comparison. The average persample processing time and peak memory usage in DPO and Groc-PO, indicating that Groc-PO adds only minor overhead.

## 4.4 Analysis and Discussion

4.4.1 Dependence on Structured Context. To verify the impor tance of structured context history, we designed a baseline named "Flattened DPO", which uses the same 17k preference pairs but removes all context history, degrading all training data to singlestage question-answer pairs. Table 5 shows the "Flattened DPO" is far below structured context Groc-PO. A likely reason is that many Stage 3 require information from the first two stages, and removing history has left the model misaligned. This result underscores the necessity of incorporating structured context to cultivate contextual coherence and enhance faithfulness.

<table><tr><td>Model</td><td>Avg. time (s)</td><td>Peak Memory Usage (MB)</td></tr><tr><td>DPO</td><td>1.99</td><td>40420</td></tr><tr><td>Groc-PO (Ours)</td><td>2.05 (+3%)</td><td>40432 (+0.03%)</td></tr></table>

![](images/5269d9358cd92b858b7721789c3b8733c4c750772c97675335642371653cdb21.jpg)  
Figure 5: Comparison on complex tasks: Groc-PO vs. DPO. While DPO shows certain degree of errors, Groc-PO demonstrates robust, evidence-based, and coherent reasoning.

4.4.2 Sensitivity to History Length. We evaluated the model’s ability to leverage contextual history of varying lengths by testing the same third-stage question (S3) under three conditions: Zeroshot (image + S3), 1-stage context (image + S1 + S3), and 2-stage context (full history: image $+ \ S 1 + \ S 2 + \ S 3 )$ . Figure 3d shows that performance exhibited a clear improvement: 2-stage <sup>></sup> 1-stage <sup>></sup> Zero-shot. This empirically validates the critical role of multi-stage context in multimodal dialogue, demonstrating that more complete history enables the model to better localize the question, perform logical reasoning, and generate accurate responses.

4.4.3 Training Overhead. We compared the training overhead of Groc-PO with DPO. Table 6 reports the average processing time per sample and peak memory usage. Results show that Groc-PO introduces only marginal overhead.

4.4.4 Discussion: Efectiveness of Grounded Context Supervision. Our analysis highlights the efectiveness of grounded context supervision. As shown in Figure 3c, model performance consistently improves as more complete grounded context is incorporated during training. This suggests that more targeted and stage-specific supervision on early grounding stages may improve the reliability of later-stage reasoning. Notably, the improvement is most evident on complex reasoning tasks. As reported in Table 3, the full Groc-PO achieves clear gains on these tasks. At the same time, it maintains strong performance on general and faithfulness benchmarks. These results indicate that grounded context supervision improves reasoning quality, is beneficial for challenging multimodal tasks, and maintains competitive performance on general and faithfulness benchmarks.

## 4.5 Case Study and Visualization

4.5.1 Token-level Log-Probabilities on Chosen Responses of Complex Tasks: Groc-PO vs. DPO. To further investigate Groc-PO’s generation confidence for complex tasks, we conducted a case study for token-level log-probability (log-prob) comparison.

In Figure 4, we observe that Groc-PO (green curve) assigns a higher log-prob to the majority of tokens in the chosen responses compared to DPO (brown curve), where the diference (yellow bars) is almost entirely positive. This indicates that Groc-PO exhibits higher internal confidence when handling complex tasks.

4.5.2 Visualization of Qualitative Comparison. Figure 5 vi sually confirms that while the DPO relatively fails on complex queries, Groc-PO generates logically accurate responses that are well-supported by explicit visual evidence.

## 5 Conclusion

In this paper, we proposed Grounded Context Preference Optimization (Groc-PO), a framework that improves MLLM faithfulness through explicit preference supervision over grounded pre-final stages. To support this objective, we introduced the Grounded Con text Preference Dataset (GCPD) and novel adaptive loss function. Extensive evaluations demonstrate that Groc-PO comprehensively enhances multiple capabilities, contributing to developing more faithful MLLMs.

## 6 Acknowledgements

This work was supported by the Artificial Intelligence-National Science and Technology Major Project (2023ZD0121200) and the

Fundamental and Interdisciplinary Disciplines Breakthrough Plan of the Ministry of Education of China (No. JYB2025XDXM103).

## References

[1] Pravesh Agrawal, Szymon Antoniak, Emma Bou Hanna, Baptiste Bout, Devendra Chaplot, Jessica Chudnovsky, Diogo Costa, Baudouin De Monicault, Saurabh Garg, Theophile Gervet, et al. 2024. Pixtral 12B. arXiv preprint arXiv:2410.07073 (2024).

[2] Jean-Baptiste Alayrac, Jef Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al. 2022. Flamingo: a visual language model for few-shot learning. Advances in neural information processing systems 35 (2022), 23716–23736.

[3] Elmira Amirloo, Jean-Philippe Fauconnier, Christoph Roesmann, Christian Kerl, Rinu Boney, Yusu Qian, Zirui Wang, Afshin Dehghan, Yinfei Yang, Zhe Gan, et al. 2024. Understanding alignment in multimodal llms: A comprehensive study. arXiv preprint arXiv:2407.02477 (2024).

[4] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. 2025. Qwen2.5-VL Technical Report. arXiv preprint arXiv:2502.13923 (2025).

[5] Zechen Bai, Pichao Wang, Tianjun Xiao, Tong He, Zongbo Han, Zheng Zhang, and Mike Zheng Shou. 2024. Hallucination of multimodal large language models: A survey. arXiv preprint arXiv:2404.18930 (2024).

[6] Assaf Ben-Kish, Moran Yanuka, Morris Alper, Raja Giryes, and Hadar Averbuch-Elor. 2023. Mocha: Multi-objective reinforcement mitigating caption hallucina tions. arXiv preprint arXiv:2312.03631 2 (2023).

[7] Paul F Christiano, Jan Leike, Tom Brown, Miljan Martic, Shane Legg, and Dario Amodei. 2017. Deep reinforcement learning from human preferences. Advances in neural information processing systems 30 (2017).

[8] Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. 2023. MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv preprint arXiv:2306.13394 (2023).

[9] Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. 2024. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 14375–14385.

[10] Anisha Gunjal, Jihan Yin, and Erhan Bas. 2024. Detecting and preventing hallucinations in large vision language models. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 38. 18135–18143.

[11] Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2022. Lora: Low-rank adaptation of large language models. ICLR 1, 2 (2022), 3.

[12] Qidong Huang, Xiaoyi Dong, Pan Zhang, Bin Wang, Conghui He, Jiaqi Wang, Dahua Lin, Weiming Zhang, and Nenghai Yu. 2024. Opera: Alleviating hallucination in multi-modal large language models via over-trust penalty and retrospection-allocation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13418–13427.

[13] Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276 (2024).

[14] Sicong Leng, Hang Zhang, Guanzheng Chen, Xin Li, Shijian Lu, Chunyan Miao, and Lidong Bing. 2024. Mitigating object hallucinations in large vision-language models through visual contrastive decoding. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13872–13882.

[15] Bohao Li, Rui Wang, Guangzhi Wang, Yuying Ge, Yixiao Ge, and Ying Shan. 2023. Seed-bench: Benchmarking multimodal llms with generative comprehension. arXiv preprint arXiv:2307.16125 (2023).

[16] Yifan Li, Yifan Du, Kun Zhou, Jinpeng Wang, Wayne Xin Zhao, and Ji-Rong Wen. 2023. Evaluating object hallucination in large vision-language models. arXiv preprint arXiv:2305.10355 (2023).

[17] Xiwen Liang, Min Lin, Weiqi Ruan, Rongtao Xu, Yuecheng Liu, Jiaqi Chen, Bingqian Lin, Yuzheng Zhuang, and Xiaodan Liang. 2025. Structured preference optimization for vision-language long-horizon task planning. arXiv preprint arXiv:2502.20742 (2025).

[18] Zijing Liang, Yanjie Xu, Yifan Hong, Penghui Shang, Qi Wang, Qiang Fu, and Ke Liu. 2024. A Survey of Multimodel Large Language Models. In Proceedings of the 3rd International Conference on Computer, Artificial Intelligence and Control Engineering. 405–409.

[19] Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 26296–26306.

[20] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual Instruc tion Tuning.

[21] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual in struction tuning. Advances in neural information processing systems 36 (2023), 34892–34916.

[22] Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. 2024. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253 (2024)

[23] Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101 (2017).

[24] Renjie Pi, Tianyang Han, Wei Xiong, Jipeng Zhang, Runtao Liu, Rui Pan, and Tong Zhang. 2024. Strengthening multimodal large language model with bootstrapped preference optimization. In European Conference on Computer Vision. Springer, 382–398.

[25] Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. Advances in Neural Information Processing Systems 36 (2023), 53728–53741.

[26] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347 (2017).

[27] Guohao Sun, Can Qin, Yihao Feng, Zeyuan Chen, Ran Xu, Sohail Dianat, Majid Rabbani, Raghuveer Rao, and Zhiqiang Tao. 2025. Structured Policy Optimiza tion: Enhance Large Vision-Language Model via Self-referenced Dialogue. In Proceedings of the IEEE/CVF International Conference on Computer Vision. 741–751.

[28] Zhiqing Sun, Sheng Shen, Shengcao Cao, Haotian Liu, Chunyuan Li, Yikang Shen, Chuang Gan, Liang-Yan Gui, Yu-Xiong Wang, Yiming Yang, et al. 2023. Aligning large multimodal models with factually augmented rlhf. arXiv preprint arXiv:2309.14525 (2023).

[29] Changli Tang, Yixuan Li, Yudong Yang, Jimin Zhuang, Guangzhi Sun, Wei Li, Zujun Ma, and Chao Zhang. 2024. Enhancing multimodal LLM for detailed and accurate video captioning using multi-round preference optimization. arXiv preprint arXiv:2410.06682 (2024).

[30] Fei Wang, Wenxuan Zhou, James Y Huang, Nan Xu, Sheng Zhang, Hoifung Poon, and Muhao Chen. 2024. mdpo: Conditional preference optimization for

multimodal large language models. arXiv preprint arXiv:2406.11839 (2024).

[31] Junyang Wang, Yuhang Wang, Guohai Xu, Jing Zhang, Yukai Gu, Haitao Jia, Ming Yan, Ji Zhang, and Jitao Sang. 2023. An LLM-free Multi-dimensional Benchmark for MLLMs Hallucination Evaluation. arXiv preprint arXiv:2311.07397 (2023).

[32] Wenyi Xiao, Ziwei Huang, Leilei Gan, Wanggui He, Haoyuan Li, Zhelun Yu, Fangxun Shu, Hao Jiang, and Linchao Zhu. 2024. Detecting and mitigating hallucination in large vision language models via fine-grained ai feedback. arXiv preprint arXiv:2404.14233 (2024).

[33] Wenyi Xiao, Ziwei Huang, Leilei Gan, Wanggui He, Haoyuan Li, Zhelun Yu, Fangxun Shu, Hao Jiang, and Linchao Zhu. 2025. Detecting and mitigating hallucination in large vision language models via fine-grained ai feedback. In Proceedings of the AAAI Conference on Artificial Intelligence, Vol. 39. 25543–25551.

[34] Yuxi Xie, Guanzhen Li, Xiao Xu, and Min-Yen Kan. 2024. V-dpo: Mitigating hallucination in large vision language models via vision-guided direct preference optimization. arXiv preprint arXiv:2411.02712 (2024).

[35] Tianyu Yu, Yuan Yao, Haoye Zhang, Taiwen He, Yifeng Han, Ganqu Cui, Jinyi Hu, Zhiyuan Liu, Hai-Tao Zheng, Maosong Sun, et al. 2024. Rlhf-v: Towards trustworthy mllms via behavior alignment from fine-grained correctional human feedback. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 13807–13816.

[36] Mengxi Zhang, Wenhao Wu, Yu Lu, Yuxin Song, Kang Rong, Huanjin Yao, Jianbo Zhao, Fanglong Liu, Haocheng Feng, Jingdong Wang, et al. 2024. Automated multi-level preference for mllms. Advances in Neural Information Processing Systems 37 (2024), 26171–26194.

[37] Yi-Fan Zhang, Tao Yu, Haochen Tian, Chaoyou Fu, Peiyan Li, Jianshu Zeng, Wulin Xie, Yang Shi, Huanyu Zhang, Junkang Wu, et al. 2025. Mm-rlhf: The next step forward in multimodal llm alignment. arXiv preprint arXiv:2502.10391 (2025).

[38] Yiyang Zhou, Chenhang Cui, Rafael Rafailov, Chelsea Finn, and Huaxiu Yao. 2024. Aligning modalities in vision large language models via preference fine-tuning. arXiv preprint arXiv:2402.11411 (2024).

[39] Yiyang Zhou, Zhiyuan Fan, Dongjie Cheng, Sihan Yang, Zhaorun Chen, Chenhang Cui, Xiyao Wang, Yun Li, Linjun Zhang, and Huaxiu Yao. 2024. Calibrated self-rewarding vision language models. arXiv preprint arXiv:2405.14622 (2024).