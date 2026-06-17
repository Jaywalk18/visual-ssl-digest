# Position: The Systemic Lack of Agency in Visual Reasoning

Yizhao Huang 1 2 \* Haoyang Chen 1 2 3 \* Shiqin Wang 1 2 \* Pohsun Huang 1 2 \* Jiayuan Li 3 4 Haoyuan Du 1 2 Yandong Shi 1 2 Zheng Wang 1 2 3 † Zhixiang Wang 5 †

## Abstract

This paper argues that a systemic lack of Agency constrains the implicit reasoning capabilities of current Vision-Language Models (VLMs). Implicit reasoning refers to the ability to autonomously discover and utilize hidden visual evidence to bridge information gaps, rather than merely relying on explicitly specified targets. This capacity underlies human visual understanding and everyday reasoning. We argue that this limitation arises from a tendency to approach visual reasoning primarily as passive semantic retrieval, rather than as active, situated reasoning that depends on autonomous visual exploration. As a result, most existing benchmarks primarily assess Passive Capacity, leaving this aspect of reasoning largely unmeasured. To address this gap, we introduce the Visual Implicit Reasoning Diagnosing Benchmark (V-IRD), which targets this missing quadrant by requiring models to derive answers strictly through autonomous visual analysis. Our results show that, despite strong retrieval abilities, prominent VLMs struggle to utilize reference objects and to attend to visual evidence that requires self-directed inquiry. Simply put, strong semantic recognition does not equate to active visual exploration, revealing a critical gap in current VLMs. More information can be found at https://haoychen.github.io/ Implicit-Reasoning/.

\*Equal Contribution, †Corresponding author. 1National Engineering Research Center for Multimedia Software, Institute of Artificial Intelligence, School of Computer Science, Wuhan University 2Hubei Key Laboratory of Multimedia and Network Communication Engineering 3Zhongguancun Academy, Beijing, China. 100094 4School of Automation, Beijing Institute of Technology 5Shanda AI Research Tokyo. Correspondence to: Zheng Wang <wangzwhu@whu.edu.cn>, Zhixiang Wang <zhixiang.wang@shanda.com>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

## 1. Introduction

In cognitive science, perception has long been understood as an active process driven by goal-directed information acquisition rather than passive stimulus reception (Gibson, 2014; Ballard, 1991). Does such agency exist in VLMs? Recent advancements in VLMs have demonstrated remarkable proficiency in semantic recognition and explicit instructionfollowing (Liu et al., 2023; Singh et al., 2025; Wang et al., 2025a; Yang et al., 2025).

However, this proficiency heavily relies on explicit guidance, creating a fundamental gap when transitioning to the unconstrained physical world. In natural settings, visual reasoning is predominantly implicit. Human observers do not wait for explicit instructions to check relevant information for reasoning. Instead, they demonstrate agency: proactively shifting attention from salient foregrounds to subtle background cues, and mining unmentioned evidence to construct coherent reasoning chains (Rose et al., 2023; Wang et al., 2025b).

Our position is that current VLMs suffer from a systemic Visual Implicit Reasoning Deficit, defined as the inability to autonomously search for and utilize visual cues not explicitly mentioned in the text prompt. While VLMs possess the capacity to perceive details when given specific guidance, they lack the agency to construct physical arguments from raw visual data. Consequently, when a prompt focuses solely on a target object without pointing to background information, the model shows no strong tendency to actively discover critical hidden information beyond the target. Instead of treating the visual context as a landscape of evidence to be explored, the model acts as a passive observer, overlooking critical visual information.

The complexity of real-world visual reasoning rarely presents itself through explicit instructions. In natural settings, the solution depends on implicit visual evidence which consists of critical geometric or physical cues (Chen et al., 2024; Lu et al., 2024) that are necessary for deduction but absent from the user’s prompt. For instance, accurately estimating the dimensions of a non-standard bottle requires the autonomous discovery of a background reference, such as a standard ID card (Yang et al., 2024), rather than relying on the semantic label bottle. The core challenge lies not in recognizing the object itself, but in the autonomous retrieval of unmentioned, supportive visual details to build a valid physical argument (Zhao et al., 2025).

![](images/23e2621ff67eceb040ff5484911d2482736571a118b770e8f590ed6a38995143.jpg)

<details>
<summary>text_image</summary>

Prompt: "How tall is this bottle?" (Answer: 20cm)
Coin is 2.5cm in diameter.-> Bottle is 8 coins high -> 20cm
Looks like a bottle -> most of bottles are 15cm.
Human/Ideal:
Active Agency
VLM:
Passive Capacity
</details>

![](images/f7de1ebccd0f7f429e33ffea9d686e64c56c8d5772e16790e0349407f5c2bddf.jpg)

<details>
<summary>scatterplot</summary>

| Cognitive Task | Context Dependency (Explicit -> Implicit) | Question |
| :--- | :--- | :--- |
| Q1 Explicit Recognition simple identificaiton | Q1 | Explicit Recognition simple identificaiton |
| Q2 Explicitly Reasoning guide for reasoning | Q2 | Explicitly Reasoning guide for reasoning |
| Q3 Implicit Perception passive checks (e.g.hallucination) | Q3 | Implicit Perception passive checks (e.g.hallucination) |
| the missing Q4 Autonomous Information Retrieval | Q4 | The missing Q4 Autonomous Information Retrieval? |
</details>

Figure 1. Agency in visual reasoning. Left: Comparison of active agency vs. passive capacity in visual reasoning. Unlike humans actively retrieve implicit visual cues to reason about physical properties, current VLMs tend to ignore the implicit information and give wrong answers. Right: Our taxonomy highlights Autonomous Information Retrieval (Q4) as the critical gap. Success here requires visual agency, the ability to actively seek unmentioned visual evidence without explicit prompting which current models lack.

In this paper, we formalize this gap as the distinction between visual capacity (what models can see when guided) and visual agency (whether models autonomously seek evidence). Through a series of diagnostic experiments, we provide evidence of this deficit. We demonstrate that without explicit instructions, visual perception and physical reasoning often become decoupled, making it difficult for the model to ground its judgments in implicit visual evidence.

Our contributions are summarized as follows:

• Concept Definition. We formalize the Visual Implicit Reasoning Deficit, positing that current models fail to autonomously ground physical reasoning in implicit visual cues due to a lack of search agency.  
• V-IRD Benchmark. To decouple visual discovery from instruction-following, we introduce V-IRD. By enforcing a strict information gap via Target-Exclusive Prompting, it compels models to autonomously identify visual cues across four domains, rather than relying on linguistic hints.  
• Evaluation and Analysis. We propose Threshold Accuracy to separate precise reasoning from estimation. Results reveal a significant performance drop in mainstream VLMs under implicit settings, highlighting a critical deficit in visual agency.

## 2. The Blind Spot in Current Evaluations

Current evaluation methodologies are systematically biased: they measure visual capacity while neglecting visual agency . We analyze three key areas where this blind spot persists.

## 2.1. Explicit VQA: Externalized Attention

While benchmarks like ColorBench (Liang et al., 2026) and MMIE (Xia et al., 2025) target fine-grained recognition, and V\* (Wu & Xie, 2024) introduces guided visual search, they all predominantly operate with explicit instruction.

These benchmarks (Zhang et al., 2025a; Weng et al., 2025; Chia et al., 2024) effectively externalize the visual planning process. The user acts as the attention manager (Hutchins, 1995) by explicitly pointing to the target. High scores in these metrics confirm the model’s grounding capability but mask its inability to autonomously identify relevant visual information without direct supervision.

Recent concurrent works like AdaptVision (Lin et al., 2025) and DeepEyes (Zheng et al., 2025) address active visual execution by employing reinforcement learning to dynamically zoom in on local details. However, effective execution requires foundational intent. While these methods provide mechanisms for active perception, our work formalizes its essential prerequisite: the lack of visual agency. Without the intrinsic agency to autonomously seek out unmentioned clues, models struggle to determine where or why to focus, fundamentally bottlenecking the potential of such advanced zooming mechanisms.

## 2.2. Hallucination: Commission vs. Omission

In work on hallucination mitigation, prevailing research such as the advanced diagnostic suite HallusionBench (Guan et al., 2024) and NOPE (Lovenia et al., 2024) predominantly targets errors of commission. These efforts (Li et al., 2023; Rostamkhani et al., 2025; Fu et al., 2024) specifically address the fabrication of non-existent objects or the over-reliance on language priors. Such a perspective largely overlooks implicit neglect which constitutes a critical error of omission.

In this scenario the failure lies not in generating false information but in failing to utilize existing visual context (Tong et al., 2024; Li et al., 2024c). When a model bypasses the necessary visual search for details like a reference scale to directly generate an answer, it is not hallucinating in the traditional sense but rather suffering from a lack of agency to visually verify its reasoning (Seth et al., 2025).

![](images/0138b81cd9920202806993f1e0d7ae9237c0ae3ba19ea439eb04e1aa59510cf9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["Start: Prompt: 'How tall is this bottle?'"] --> B["Validate Prior Knowledge"]
  B --> C["Sufficient Knowledge"]
  C --> D["Visual Implicit Reasoning"]
  D --> E["Threshold: 5%, 10%, 20%"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#fcc,stroke:#333
    style E fill:#ffc,stroke:#333
```
</details>

Figure 2. Framework for verifying implicit reasoning capability. To rigorously assess implicit reasoning independent of knowledge retrieval, we employ a filtered evaluation pipeline. We first validate prior knowledge across domains to ensure the model possesses the necessary factual basis, eliminating errors caused by knowledge gaps. Verified instances then proceed to Visual Implicit Reasoning, where the model needs to autonomously chain Implicit Information to derive the Target answer. This design ensures that performance metrics strictly reflect the model’s capability to reason with unstated visual cues rather than its memorized knowledge base.

## 2.3. Physical Reasoning in Closed and Open Sets

Benchmarks such as PhysBench (Chow et al., 2025), Phys-Reason (Zhang et al., 2025b) and MMMU (Yue et al., 2024; 2025) have significantly advanced the evaluation of physical knowledge. However, these tasks are frequently presented in structured and closed-set formats where necessary variables are usually explicitly provided.

In contrast, authentic physical reasoning constitutes an openset inference problem. This process requires autonomous discovery of unmentioned support evidence, such as noticing floor unevenness to determine stability (Li et al., 2026a). Consequently, existing benchmarks test the processing of physical rules (Xu et al., 2025) but fail to evaluate the acquisition of the visual evidence required to apply those rules.

## 3. The Visual Implicit Reasoning Deficit

We posit that the current limitations of VLMs stem primarily from a specific deficit in implicit visual reasoning rather than a general lack of cognitive capacity. In this section we formalize the distinction between explicit and implicit visual reasoning and map the current capabilities of VLMs onto a cognitive quadrant to isolate this specific deficit.

## 3.1. Formalizing Visual Implicit Reasoning

To understand the deficit, we mathematically characterize the reasoning process. Let a VLM M take an image I and a text query Q to produce an answer A.

Explicit Reasoning. In the dominant paradigm of current training, Q explicitly contains pointers to the necessary visual evidence E. For example, if Q is “Is the red cable connected to the battery?”, the attention mechanism is linguistically guided to the specific regions of the red cable and battery. The task is reduced to a verification problem:

$$
A \leftarrow M (I, Q _ {\text { explicit }}) \tag {1}
$$

Here, the model’s role is passive because it executes a predefined visual search plan provided by the user.

Implicit Reasoning. In contrast, implicit reasoning presents an under-specified query where the necessary E is unmentioned. For example, Q might simply be “What is the diameter of this badge?”.

To answer this question accurately, the model needs to autonomously identify and utilize implicit visual information in the image, such as a coin with a known diameter. This involves a coherent two-stage reasoning process. First, the model executes a Plan phase, leveraging prior world knowledge K to formulate a specific search intent from Q. It then performs an autonomous Search operation, examining I guided by this intent to uncover relevant but unmentioned evidence E.

$$
E \leftarrow \operatorname{Search} (I, \text { Plan } (Q, K)) \tag {2}
$$

Upon the successful retrieval of E, the cognitive task transitions from open-ended exploration to well-founded reasoning. The model is expected to integrate this discovered visual context with the original query to derive the final conclusion. This ultimate inference step is formalized as:

$$
A \leftarrow M (E, Q) \tag {3}
$$

The Deficit. We define the Visual Implicit Reasoning Deficit as a critical breakdown in the autonomous retrieval phase defined in Equation (2). When the necessary E is not explicitly named in Q, the model fails to initiate the search mechanism Plan(Q, K). Instead of treating the image as a field of potential evidence to be explored, the model tends to restrict its visual attention primarily to the entities explicitly mentioned in the text. Consequently, the reasoning process degenerates from the grounded deduction of Equation (3) to a constrained inspection of the target object:

$$
A \leftarrow M (I | _ {Q _ {\text { target }}}, Q) \tag {4}
$$

where $I | _ { Q _ { \mathrm { t a r g e t } } }$ denotes the visual content restricted solely to the target object explicitly named in Q.

In this state, the model exhibits attention tunneling: it accurately processes the pixels of the object mentioned in the prompt (Visual Capacity) but treats the surrounding context, which contains the crucial unmentioned evidence, as irrelevant background. Lacking the retrieved context E, the model cannot construct a valid physical argument and often defaults to parametric hallucinations.

## 3.2. The Missing Quadrant: Autonomous Information Retrieval

To further contextualize this deficit, we propose a taxonomy of visual-language tasks structured along two axes. The first axis is Information Availability distinguishing between explicit and implicit inputs while the second is Cognitive Demand distinguishing between recognition and reasoning. As illustrated in Figure 1, this framework divides the landscape into four operational quadrants.

Quadrant I represents explicit recognition where targets are named and the goal is simple identification. In this domain current models exhibit high visual capacity. Quadrant II covers explicit reasoning typified by mathematical benchmarks where models perform well because the textual query provides a guided reasoning path. Quadrant III involves implicit perception which addresses passive quality control issues such as object hallucination but does not require active evidence seeking. Quadrant IV is defined as autonomous information Retrieval where visual evidence is decisive but entirely unmentioned in the prompt.

The critical gap lies within Quadrant IV which we identify as the Missing Quadrant. Unlike explicit tasks success here requires the model to autonomously deduce that specific unnamed features need to be retrieved to answer the high-level query. Current VLMs function as lazy readers operating effectively only when told where to look but failing fundamentally in this domain. They lack the visual agency to transform a high-level goal into a low-level visual search operation. This deficit explains the paradox where a model can perfectly describe a specific visual defect if explicitly asked yet confidently overlooks the same defect when making a holistic safety judgment simply because it never autonomously initiated the search for evidence.

![](images/94230e5175f123f1bcf37c08e942417062fc982fd547c5a3ebc1ff482ff7e659.jpg)

<details>
<summary>pie chart</summary>

| Category | Value (%) |
|---|---|
| Kinematics | 3.7 |
| Electricity | 5.6 |
| Weight | 8.0 |
| Temperature | 13.0 |
| Physical Property | 14.2 |
| Environmental | 14.2 |
| Length | 21.6 |
| Distance | 8.0 |
| Volume | 7.4 |
| Area | 3.7 |
| Remark | 14.8 |
The chart displays a single ring with categories labeled around it. The values for 'Spatial Geometry' are shown in blue, 'Contextual Inference' in orange, and 'Physical Loss' in red. The label 'm²' appears in the bottom right corner.
</details>

Figure 3. Statistics of categories and tasks in V-IRD.

## 4. Empirical Analysis: Boundaries and Mechanisms of Implicit Visual Reasoning in VLMs

To validate our hypothesis of the Visual Implicit Reasoning Deficit, we designed a comprehensive experimental framework encompassing mainstream VLMs shown in Figure 2. Rather than relying on large-scale automated benchmarks, which may contain statistical biases, we adopt a targeted evaluation strategy. Our experiments are structured logically to peel back the layers of the deficit: first establishing capacity boundaries, then exposing the lack of agency, and finally analyzing the cognitive breakdown mechanisms.

## 4.1. The V-IRD Benchmark

Current mainstream benchmarks like PhysBench (Chow et al., 2025) typically structure prompts with direct pointers, such as asking “What is the color of the leftmost spectrum in the picture?”. Recent works on visual reasoning like Point-It-Out (Xue et al., 2025) further highlight that models rely heavily on text-conditioned attention or visual markers to locate relevant regions. These prompts effectively leak the solution path. By explicitly naming the supporting evidence, these benchmarks evaluate the verification capacity of the model rather than its search agency.

Table 1. Validation of prerequisite capabilities. The accuracy (%) of Visual Recognition and Parametric Knowledge is reported to ensure models possess the foundational knowledge required for subsequent reasoning tasks.

<table><tr><td rowspan="2">Model Category</td><td colspan="2">Task</td></tr><tr><td>Visual Recognition %</td><td>Parametric Knowledge %</td></tr><tr><td>Lightweight (&lt;7B)</td><td>99.74</td><td>74.81</td></tr><tr><td>Medium-Scale (7-30B)</td><td>100.00</td><td>87.01</td></tr><tr><td>Large-Scale (30-80B)</td><td>99.55</td><td>90.91</td></tr><tr><td>Closed-source Models</td><td>100.00</td><td>96.00</td></tr></table>

Benchmark Construction and Taxonomy. We adopted a multi-source collection strategy to guarantee visual diversity, sourcing images from manual photography, web crawling, and AI generation. As illustrated in Figure 3, we structured the V-IRD benchmark into a hierarchical taxonomy spanning four core domains to ensure comprehensive cognitive coverage. Spatial Geometry constitutes the largest portion (41%), focusing on precise metrology tasks such as Length, Distance, Volume, and Area. Contextual Inference (29%) challenges the model to deduce abstract information like Environment and Remark. The remaining data comprises Physical Properties (21%), covering Temperature and Weight, and Physical Logic (9%), which involves Electricity and Kinematics. In addition, we conduct a rigorous physical consistency check. We manually screened all images, especially those generated by AI, to remove logical errors or physically impossible content. This process guarantees that every visual clue is realistic and reliable.

The Target-Exclusive Prompting Strategy. To ensure that performance stems solely from intrinsic visual agency rather than instruction-following, we implement a strict Target-Exclusive Strategy. Unlike traditional visual prompting techniques that guide attention via spatial referents or bounding boxes, our prompts are rigorously sanitized to mention only the semantic target subject (e.g., How tall is this bottle?). We explicitly forbid any textual references to background reference objects such as coins, environmental markers, or relative positions. This constraint forces the model to independently realize that unmentioned visual information is critical to the solution, effectively turning the evaluation into a pure test of Autonomous Visual Discovery.

## 4.2. Evaluation Metrics

To evaluate performance across different task types, we employ specific criteria to define prediction correctness.

1) Discrete Tasks. For reasoning tasks involving categorical outcomes such as logical reasoning and environment inference, standard Accuracy (ACC) is employed. A prediction is considered correct only if the predicted category exactly matches the ground-truth label. Mathematically, for a prediction cˆ and ground truth c, the scoring function is:

$$
\mathrm{ACC} = \mathbb {I} (\hat {c} = c) \tag {5}
$$

where I(·) is the indicator function, which equals 1 if the condition is met and 0 otherwise.

2) Continuous Tasks. For estimation tasks as geometric scaling and physical properties, Threshold Accuracy $( \operatorname { A C C } _ { \delta } )$ is employed. Unlike regression metrics that average continuous errors, we adopt a strict binary success criterion. A prediction is considered correct only if its relative error falls within a fixed threshold δ:

$$
\mathrm{ACC} _ {\delta} = \mathbb {I} \left(\frac {| y - \hat {y} |}{y} \leq \delta\right) \tag {6}
$$

In this case, y represents the ground-truth value, and yˆ is the predicted value. Note that for edge cases where the ground truth is zero $( e . g . , 0 ^ { \circ } \mathrm { C } )$ , the metric degenerates to standard ACC, requiring an exact match. To evaluate reasoning precision at different granularities, we adopt three distinct thresholds $\delta \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 0 . 3 0 \}$ (representing 5%, 10%, 20% and 30% tolerance margins). If the error exceeds the given threshold, the score is 0. This strict evaluation regime penalizes vague guesses and distinguishes high-precision visual measurements from coarse approximations.

## 4.3. Verification of Explicit Capability

Before studying implicit reasoning, the factor of lack of knowledge needs to be excluded. We need to verify whether the failures stem from a lack of agency or simply because the model lacks the required fundamental knowledge.

We extracted the core Implicit Information Components underpinning the V-IRD benchmark, which broadly encompass key visual reference objects (e.g., standard coins, ID cards), fundamental physical rules, and environmental information. To ensure a comprehensive evaluation, we categorized the models into four distinct groups based on parameter size and accessibility: Lightweight (< 7B), Medium-Scale (7 − 30B), Large-Scale (30 − 80B), and Closed-source Models. We structured the Explicit Capability Probe into two distinct verification stages. The first stage evaluates Visual Recognition to determine if the model can accurately identify the implicit objects from the visual input. The second stage evaluates Parametric Knowledge by querying the model via text to confirm if it knows the specific physical attributes corresponding to these objects. This process effectively acts as a unit test for the model’s knowledge base, isolating its foundational capabilities from its reasoning agency.

Results. As presented in Table 1, the evaluation of prerequisite capabilities reveals that visual recognition has reached saturation, with accuracy exceeding 99.55% across all categories and achieving a perfect 100.00% in Medium-Scale and Closed-source models, indicating robust perceptual systems. Concurrently, Parametric Knowledge demonstrates a positive correlation with model scale, where performance steadily improves from a baseline of 74.81% in Lightweight models to 87.01% and 90.91% in Medium-Scale and Large-Scale models respectively, culminating in a peak of 96.00% for Closed-source architectures. These findings confirm that the essential atomic knowledge required for V-IRD tasks is already encoded within these systems, thereby isolating the variable of interest and verifying that subsequent failures in implicit reasoning stem from an agency deficit rather than a fundamental lack of capability.

Table 2. Main results on V-IRD. We benchmark a broad range of VLMs across four domains and ten fine-grained tasks. Each cell reports accuracy (%) under relative-error thresholds of 5% and 10%, formatted as $\left( \mathrm { A C C } _ { 5 \% } , \mathrm { A C C } _ { 1 0 \% } \right)$ . The Average column gives the mean accuracy across all ten tasks at the two thresholds. Within each model group, the best score per column is highlighted in bold. Across all VLM groups, the top-3 entries per column are shaded with decreasing intensity: 1st , 2nd , 3rd .

<table><tr><td rowspan="2">Model</td><td colspan="4">Spatial Geometry</td><td colspan="2">Physical Properties</td><td colspan="2">Physical Logic</td><td colspan="2">Contextual Inference</td><td rowspan="2">Average</td></tr><tr><td>Length</td><td>Area</td><td>Volume</td><td>Distance</td><td>Temp.</td><td>Weight</td><td>Electricity</td><td>Kinematics</td><td>Remark</td><td>Env.</td></tr><tr><td colspan="12">Open-source VLMs: &lt; 7B</td></tr><tr><td>InternVL3.5-1B (Wang et al., 2025a)</td><td>(7.14, 10.00)</td><td>(0.00, 16.67)</td><td>(0.00, 4.17)</td><td>(11.54, 34.62)</td><td>(54.76, 59.52)</td><td>(34.62, 34.62)</td><td>(11.11, 11.11)</td><td>(50.00, 50.00)</td><td>(16.67, 16.67)</td><td>(43.48, 43.48)</td><td>(22.93, 28.08)</td></tr><tr><td>InternVL3.5-2B (Wang et al., 2025a)</td><td>(0.00, 8.57)</td><td>(0.00, 8.33)</td><td>(8.33, 12.50)</td><td>(0.00, 7.69)</td><td>(69.05, 76.19)</td><td>(34.62, 34.62)</td><td>(38.89, 38.89)</td><td>(33.33, 33.33)</td><td>(25.00, 33.33)</td><td>(56.52, 56.52)</td><td>(26.57, 31.00)</td></tr><tr><td>InternVL3-1B (Zhu et al., 2025)</td><td>(4.29, 8.57)</td><td>(8.33, 8.33)</td><td>(4.17, 12.50)</td><td>(3.85, 19.23)</td><td>(28.57, 30.95)</td><td>(26.92, 26.92)</td><td>(11.11, 11.11)</td><td>(41.67, 41.67)</td><td>(27.08, 31.25)</td><td>(52.17, 52.17)</td><td>(20.82, 24.27)</td></tr><tr><td>InternVL3-2B (Zhu et al., 2025)</td><td>(7.14, 20.00)</td><td>(0.00, 0.00)</td><td>(12.50, 20.83)</td><td>(3.85, 19.23)</td><td>(78.57, 85.71)</td><td>(42.31, 42.31)</td><td>(55.56, 55.56)</td><td>(41.67, 58.33)</td><td>(52.08, 52.08)</td><td>(54.35, 54.35)</td><td>(34.80, 40.84)</td></tr><tr><td>Qwen3-VL-2B (Yang et al., 2025)</td><td>(8.57, 15.71)</td><td>(8.33, 16.67)</td><td>(4.17, 8.33)</td><td>(3.85, 11.54)</td><td>(71.43, 71.43)</td><td>(53.85, 57.69)</td><td>(33.33, 33.33)</td><td>(33.33, 33.33)</td><td>(39.58, 47.92)</td><td>(54.35, 54.35)</td><td>(31.08, 35.03)</td></tr><tr><td>Qwen3-VL-4B (Yang et al., 2025)</td><td>(20.00, 31.43)</td><td>(0.00, 8.33)</td><td>(0.00, 16.67)</td><td>(0.00, 0.00)</td><td>(71.43, 71.43)</td><td>(76.92, 76.92)</td><td>(38.89, 50.00)</td><td>(33.33, 41.67)</td><td>(25.00, 25.00)</td><td>(67.39, 67.39)</td><td>(33.30, 38.89)</td></tr><tr><td>Qwen2.5-VL-3B (Bai et al., 2025)</td><td>(8.57, 20.00)</td><td>(16.67, 16.67)</td><td>(4.17, 16.67)</td><td>(11.54, 26.92)</td><td>(71.43, 71.43)</td><td>(38.46, 42.31)</td><td>(27.78, 27.78)</td><td>(16.67, 25.00)</td><td>(35.42, 35.42)</td><td>(60.87, 60.87)</td><td>(29.16, 34.31)</td></tr><tr><td colspan="12">Open-source VLMs: 7-8B</td></tr><tr><td>Qwen3-VL-8B (Yang et al., 2025)</td><td>(10.00, 22.86)</td><td>(0.00, 8.33)</td><td>(12.50, 20.83)</td><td>(11.54, 19.23)</td><td>(90.48, 92.86)</td><td>(61.54, 65.38)</td><td>(44.44, 44.44)</td><td>(41.67, 41.67)</td><td>(35.42, 37.50)</td><td>(86.96, 86.96)</td><td>(39.45, 44.01)</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2025)</td><td>(7.14, 20.00)</td><td>(8.33, 16.67)</td><td>(0.00, 8.33)</td><td>(3.85, 23.08)</td><td>(52.38, 52.38)</td><td>(65.38, 69.23)</td><td>(27.78, 27.78)</td><td>(8.33, 8.33)</td><td>(29.17, 31.25)</td><td>(65.22, 65.22)</td><td>(26.76, 32.23)</td></tr><tr><td>InternVL3.5-8B (Wang et al., 2025a)</td><td>(7.14, 15.71)</td><td>(0.00, 0.00)</td><td>(8.33, 16.67)</td><td>(7.69, 11.54)</td><td>(90.48, 90.48)</td><td>(50.00, 50.00)</td><td>(33.33, 33.33)</td><td>(50.00, 50.00)</td><td>(43.75, 45.83)</td><td>(71.74, 71.74)</td><td>(36.25, 38.53)</td></tr><tr><td>InternVL3-8B (Zhu et al., 2025)</td><td>(4.29, 11.43)</td><td>(0.00, 33.33)</td><td>(0.00, 25.00)</td><td>(3.85, 3.85)</td><td>(50.00, 59.52)</td><td>(42.31, 42.31)</td><td>(22.22, 22.22)</td><td>(58.33, 58.33)</td><td>(60.42, 66.67)</td><td>(73.91, 73.91)</td><td>(31.53, 39.66)</td></tr><tr><td>LLaVA-OV-7B (Li et al., 2024a)</td><td>(10.00, 27.14)</td><td>(0.00, 0.00)</td><td>(4.17, 12.50)</td><td>(11.54, 23.08)</td><td>(59.52, 59.52)</td><td>(46.15, 46.15)</td><td>(5.56, 5.56)</td><td>(33.33, 50.00)</td><td>(43.75, 47.92)</td><td>(71.74, 71.74)</td><td>(28.58, 34.36)</td></tr><tr><td colspan="12">Open-source VLMs: 10-30B</td></tr><tr><td>InternVL3.5-14B (Wang et al., 2025a)</td><td>(7.14, 18.57)</td><td>(0.00, 8.33)</td><td>(8.33, 12.50)</td><td>(3.85, 15.38)</td><td>(80.95, 80.95)</td><td>(50.00, 50.00)</td><td>(27.78, 27.78)</td><td>(83.33, 83.33)</td><td>(66.67, 70.83)</td><td>(69.57, 69.57)</td><td>(39.76, 43.73)</td></tr><tr><td>InternVL3-14B (Zhu et al., 2025)</td><td>(12.86, 22.86)</td><td>(8.33, 8.33)</td><td>(12.50, 41.67)</td><td>(23.08, 34.62)</td><td>(85.71, 85.71)</td><td>(53.85, 53.85)</td><td>(33.33, 33.33)</td><td>(50.00, 50.00)</td><td>(50.00, 54.17)</td><td>(73.91, 73.91)</td><td>(40.36, 45.84)</td></tr><tr><td colspan="12">Open-source VLMs: 30-70B</td></tr><tr><td>InternVL3-38B (Zhu et al., 2025)</td><td>(11.43, 20.00)</td><td>(0.00, 0.00)</td><td>(12.50, 16.67)</td><td>(11.54, 23.08)</td><td>(90.48, 95.24)</td><td>(69.23, 69.23)</td><td>(27.78, 33.33)</td><td>(58.33, 58.33)</td><td>(62.50, 66.67)</td><td>(78.26, 78.26)</td><td>(42.20, 46.08)</td></tr><tr><td>InternVL3.5-38B (Wang et al., 2025a)</td><td>(12.86, 24.29)</td><td>(0.00, 0.00)</td><td>(4.17, 16.67)</td><td>(15.38, 34.62)</td><td>(90.48, 90.48)</td><td>(69.23, 69.23)</td><td>(50.00, 55.56)</td><td>(66.67, 66.67)</td><td>(58.33, 62.50)</td><td>(76.09, 76.09)</td><td>(44.32, 49.61)</td></tr><tr><td>Qwen2.5-VL-32B (Bai et al., 2025)</td><td>(11.43, 24.29)</td><td>(8.33, 16.67)</td><td>(0.00, 8.33)</td><td>(0.00, 3.85)</td><td>(76.19, 76.19)</td><td>(73.08, 80.77)</td><td>(44.44, 44.44)</td><td>(50.00, 50.00)</td><td>(47.92, 52.08)</td><td>(69.57, 69.57)</td><td>(38.10, 42.62)</td></tr><tr><td>Qwen3-VL-32B (Yang et al., 2025)</td><td>(15.71, 22.86)</td><td>(0.00, 16.67)</td><td>(12.50, 16.67)</td><td>(7.69, 11.54)</td><td>(92.86, 95.24)</td><td>(73.08, 73.08)</td><td>(44.44, 44.44)</td><td>(50.00, 58.33)</td><td>(54.17, 54.17)</td><td>(84.78, 84.78)</td><td>(43.52, 47.78)</td></tr><tr><td colspan="12">Open-source VLMs: 70-80B</td></tr><tr><td>LLaVA-NEXT-72B (Li et al., 2024b)</td><td>(15.71, 35.71)</td><td>(0.00, 0.00)</td><td>(0.00, 12.50)</td><td>(0.00, 3.85)</td><td>(73.81, 73.81)</td><td>(38.46, 38.46)</td><td>(11.11, 11.11)</td><td>(33.33, 33.33)</td><td>(14.58, 16.67)</td><td>(54.35, 54.35)</td><td>(24.14, 27.98)</td></tr><tr><td>LLaVA-OV-72B (Li et al., 2024a)</td><td>(17.14, 30.00)</td><td>(0.00, 0.00)</td><td>(8.33, 16.67)</td><td>(0.00, 11.54)</td><td>(88.10, 90.48)</td><td>(61.54, 61.54)</td><td>(38.89, 38.89)</td><td>(66.67, 66.67)</td><td>(47.92, 47.92)</td><td>(78.26, 78.26)</td><td>(40.68, 44.20)</td></tr><tr><td>Qwen2.5-VL-72B (Bai et al., 2025)</td><td>(11.43, 21.43)</td><td>(16.67, 25.00)</td><td>(0.00, 0.00)</td><td>(3.85, 3.85)</td><td>(80.95, 80.95)</td><td>(84.62, 88.46)</td><td>(50.00, 50.00)</td><td>(50.00, 50.00)</td><td>(60.42, 66.67)</td><td>(78.26, 78.26)</td><td>(43.62, 46.46)</td></tr><tr><td>InternVL3-78B (Zhu et al., 2025)</td><td>(14.29, 25.71)</td><td>(8.33, 25.00)</td><td>(8.33, 20.83)</td><td>(11.54, 23.08)</td><td>(90.48, 90.48)</td><td>(69.23, 69.23)</td><td>(22.22, 22.22)</td><td>(66.67, 66.67)</td><td>(66.67, 70.83)</td><td>(84.78, 84.78)</td><td>(44.25, 49.88)</td></tr><tr><td colspan="12">Open-source VLMs: &gt;200B</td></tr><tr><td>Qwen3-VL-235B (Yang et al., 2025)</td><td>(14.29, 27.14)</td><td>(16.67, 25.00)</td><td>(0.00, 16.67)</td><td>(7.69, 7.69)</td><td>(90.48, 97.62)</td><td>(76.92, 76.92)</td><td>(44.44, 50.00)</td><td>(66.67, 66.67)</td><td>(62.50, 62.50)</td><td>(84.78, 84.78)</td><td>(46.44, 51.50)</td></tr><tr><td colspan="12">Proprietary VLMs</td></tr><tr><td>GPT-5.2 (Singh et al., 2025)</td><td>(14.29, 27.14)</td><td>(16.67, 16.67)</td><td>(8.33, 16.67)</td><td>(0.00, 3.85)</td><td>(92.86, 100.00)</td><td>(80.77, 84.62)</td><td>(50.00, 55.56)</td><td>(58.33, 58.33)</td><td>(75.00, 77.08)</td><td>(78.26, 78.26)</td><td>(47.45, 51.82)</td></tr><tr><td>Claude-Sonnet-4.5 (Anthropic, 2025)</td><td>(27.14, 44.29)</td><td>(16.67, 33.33)</td><td>(29.17, 37.50)</td><td>(0.00, 3.85)</td><td>(88.10, 88.10)</td><td>(73.08, 73.08)</td><td>(38.89, 44.44)</td><td>(66.67, 75.00)</td><td>(72.92, 75.00)</td><td>(86.96, 86.96)</td><td>(49.96, 56.15)</td></tr><tr><td>Claude-Sonnet-4.5-Thinking (Anthropic, 2025)</td><td>(25.71, 41.43)</td><td>(16.67, 16.67)</td><td>(12.50, 16.67)</td><td>(3.85, 15.38)</td><td>(85.71, 88.10)</td><td>(69.23, 69.23)</td><td>(33.33, 33.33)</td><td>(83.33, 83.33)</td><td>(75.00, 79.17)</td><td>(91.30, 91.30)</td><td>(49.66, 53.46)</td></tr><tr><td>Gemini-3-Flash (Google DeepMind, 2025)</td><td>(35.71, 58.57)</td><td>(16.67, 41.67)</td><td>(29.17, 33.33)</td><td>(3.85, 19.23)</td><td>(95.24, 100.00)</td><td>(65.38, 69.23)</td><td>(66.67, 66.67)</td><td>(75.00, 83.33)</td><td>(79.17, 79.17)</td><td>(84.78, 84.78)</td><td>(55.16, 63.60)</td></tr><tr><td>Gemini-3-Pro (Google DeepMind, 2025)</td><td>(44.29, 54.29)</td><td>(25.00, 50.00)</td><td>(16.67, 29.17)</td><td>(11.54, 11.54)</td><td>(95.24, 97.62)</td><td>(80.77, 80.77)</td><td>(66.67, 66.67)</td><td>(75.00, 83.33)</td><td>(77.08, 77.08)</td><td>(91.30, 91.30)</td><td>(58.36, 64.18)</td></tr><tr><td colspan="12">Human reference</td></tr><tr><td>Human</td><td>(40.00, 60.00)</td><td>(16.67, 25.00)</td><td>(50.00, 50.00)</td><td>(38.46, 46.15)</td><td>(100.00, 100.00)</td><td>(92.31, 92.31)</td><td>(66.67, 66.67)</td><td>(83.33, 83.33)</td><td>(83.33, 83.33)</td><td>(91.30, 91.30)</td><td>(66.21, 69.81)</td></tr></table>

## 4.4. Core Evaluation: The Deficit of Active Visual Reasoning

Experimental Settings. We evaluated standard VLMs on the V-IRD benchmark using the Target-Exclusive Strategy in Table 2. In this setting, prompts explicitly request the final target value (Target T ) but intentionally omit any mention of the available visual evidence (Implicit Information I). This forces the model to autonomously discover and utilize the visual context. Performance is measured using $\operatorname { A C C } _ { \delta }$ at strict $( \delta ~ = ~ 5 \% )$ and relaxed $( \delta \ : = \ : 1 0 \% )$ margins for continuous tasks, and standard Accuracy for discrete tasks.

Severe Collapse in Spatial Geometry. The results reveal a significant performance divergence, with the most catastrophic failure observed in Spatial Geometry. While models demonstrated high precision in explicit pre-experiments (where reference objects were named), their performance declined significantly under the target-exclusive setting. For instance, even at the relaxed threshold $( \delta = 1 0 \% )$ , most models struggled to achieve meaningful accuracy. This degradation suggests that without explicit text pointing to reference objects, models frequently struggle to actively look for them, which leads to hallucinations based on prior training distributions rather than situated visual measurement.

Performance in Physical Tasks and Contextual Inference. Other domains exhibited varying degrees of fragility. Physical Properties and Physical Logic maintained relatively good performance where visual components were salient. Conversely, in Contextual Inference tasks requiring context deduction, models frequently ignored background evidence in favor of generic foreground features. Overall, Closedsource models generally outperformed Open-source models across these domains, exhibiting stronger robustness in active retrieval, although the visual agency deficit remains a widespread challenge across current architectures.

![](images/f3e82c7636f274063b0925d73b28b8457cc2cf44453640af0015a241597b09b4.jpg)

<details>
<summary>text_image</summary>

Category 1:
Physical Properties
[Image1]
Question:
What is the
temperature of the
water in the cup?
True value:
0 degrees Celsius.
[Image 1]
Question:
What is the
resistance
of the light
bulb in ohms?
True value: 488.89 Ohms
[Image 1]
Question:
What is the
resistance
of the light
bulb in ohms?
True value: 270W
[Image 1]
Question:
What is the
resistance
of the light
bulb in ohms?
True value: 881.667 Ohms
[Image 1]
Question:
What is the
resistance
of the light
bulb in ohms?
True value: 488.89 Ohms
[Image 1]
Question:
What is the
resistance
of the light
bulb in ohms?
True value: 270W
[Image 2]
Question:
What is the
resistance
of the light
bulb in ohms?
True value: 270W
[Image 2]
Question:
What is the
resistance
of the light
bulb in ohms?
True value: 270W
[Image 2]
Question:
What is the
resistance
of the light
bulb in ohms?
True value: 270W
[Image 2] Question:
What is the
resistance
of the light
bulb in ohms?
True value: French Revolution
[Image 2] Question:
What is the
resistance
of the light
bulb in ohms?
True value: French Revolution
[Image 2] Question:
What is the
resistance
of the light
bulb in ohms?
True value: French Revolution
[Image 3] Question:
What is the
resistance
of the light
bulb in ohms?
True value: Turkish flight
[Image 3] Question:
Which country's
flight is this person
taking?
True value: Turkish flight
[Image 3] Question:
What is the
resistance
of the light
bulb in ohms?
True value: Turkish flight
</details>

Figure 4. Representative instances of implicit reasoning. Category 1: Physical Properties infers intrinsic attributes, ranging from thermodynamic states to mass equivalence calculations based on conservation principles. Category 2: Physical Logic applies specific physical laws to analyze functional systems. Category 3: Contextual Inference deduces non-visual contexts from environmental clues. Category 4: Spatial Geometry performs precise metrology using reference objects. Success in these domains requires visual agency, which is the ability to actively retrieve unmentioned evidence for high-level queries.

## 4.5. Probing Agency Deficit in Visual Reasoning

To further probe the robustness of the reasoning agency, we conducted a focused qualitative analysis on a curated set of 10 complex samples. These images are characterized by high information density with multiple potential visual references, yet contain sparse valid cues specifically applicable to the reasoning target. We selected the top-performing models at the 5% threshold across different scales (InternVL3-2B, Qwen3-VL-8B, InternVL3-14B, InternVL3.5-38B, InternVL3-78B, and Gemini-3-pro) and instructed them to generate explicit Chain-of-Thought (Wei et al., 2022) sequences to solve these tasks. A prediction is considered correct if the relative error falls within a 10% threshold (δ<10%). By analyzing their generated traces, we pinpoint exactly where the cognitive chain breaks. We formulated a hierarchical taxonomy to categorize failures into three sequential stages:

Stage I: Active Discovery Failure. The model provides a detailed description of the clutter but fails to acknowledge the presence of the specific implicit information required for the task.

Stage II: Valuation and Selection Failure. The model explicitly notices the valid visual evidence but fails to establish a logical connection with the target. In these informationrich scenarios, the model often treats the critical cue as irrelevant background noise, overwhelmed by other salient but non-functional objects.

Stage III: Logical Calculation Failure. The model successfully bridges the anchor and the target but fails at the stage of physical modeling or numerical computation.

Analysis Results. As shown in Table 3, the quantitative results reveal a decisive skew towards early-stage perceptual deficits. On average, 75.82% of failures are classified as Stage I, indicating that models predominantly fail to perceive the implicit cues entirely. This is particularly pronounced in smaller models, where Stage I errors reach 90%. Stage II accounts for an additional 14.42%, where evidence is noticed but treated as noise. Consequently, the combined Agency Deficit (Stage I & II) constitutes over 90% of all failures. In contrast, only 9.76% of errors result from Stage III calculation failures. Notably, smaller models exhibit 0% logic failure simply because they rarely survive the discovery phase to attempt calculation. Even for the strongest model (Gemini-3-pro), the Agency Deficit remains at 80%, confirming that the primary bottleneck is not calculation capacity, but the agency to initiate search.

Table 3. Quantitative analysis of failure stages. Errors are decomposed into Agency Deficit (Stage I: Discovery, Stage II: Association) and Capacity Deficit (Stage III: Logic). Results confirm that reasoning is primarily bottlenecked by an inability to actively find the correct implicit information, which accounts for the vast majority of failures and significantly outweighs logic-based errors.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Accuracy (%)</td><td colspan="2">Agency Deficit (%)</td><td>Capacity Deficit (%)</td></tr><tr><td>I: Discovery</td><td>II: Association</td><td>III: Logic</td></tr><tr><td>InternVL3-2B</td><td>0.00</td><td>90.00</td><td>10.00</td><td>0.00</td></tr><tr><td>Qwen3-VL-8B</td><td>10.00</td><td>88.89</td><td>11.11</td><td>0.00</td></tr><tr><td>InternVL3-14B</td><td>10.00</td><td>88.89</td><td>11.11</td><td>0.00</td></tr><tr><td>InternVL3.5-38B</td><td>0.00</td><td>70.00</td><td>20.00</td><td>10.00</td></tr><tr><td>InternVL3-78B</td><td>30.00</td><td>57.14</td><td>14.29</td><td>28.57</td></tr><tr><td>Gemini-3-Pro</td><td>50.00</td><td>60.00</td><td>20.00</td><td>20.00</td></tr><tr><td>Average</td><td>16.67</td><td>75.82</td><td>14.42</td><td>9.76</td></tr></table>

## 4.6. Diagnostic Experiment: Explicit Information Injection

To conclusively verify that the performance collapse stems from an agency deficit rather than capability limitations, we conducted a gradient-based information injection experiment. We used the same model and data configurations as in the previous experiment, employing the same carefully curated high-information-density samples and representative model set.

We designed a four-stage protocol to observe performance evolution as visual planning is offloaded to the prompt. The experiment starts at Level 0 (Implicit Baseline) with original underspecified instructions. We then introduce Level 1 (object awareness) to explicitly prompt attention to specific references, followed by Level 2 (attribute awareness) which specifies the physical attributes to inspect. Finally, Level 3 (oracle guidance) provides ground truth data for comparison and reduces the task to pure logical calculation. Consistent with the previous experiment, a prediction is considered correct if the relative error falls within a 10% threshold (δ<10%).

Results and Implications. As shown in Figure 5, for large-scale models, the highly explicit physical information injected into the prompts may conflict with their extensive pre-trained knowledge. Such models generally depend on robust parametric priors. Therefore, discrepancies between the provided physical cues and their inherent knowledge may induce reasoning conflicts, leading to stagnant or diminished performance. Conversely, medium-scale models demonstrate an improvement. This positive trend indicates that explicit external guidance mitigates their intrinsic deficit in active visual exploration to a certain degree. Meanwhile, the performance of small-scale models remains relatively static. This limitation is presumably attributable to their restricted baseline capacity, which impedes their ability to process and integrate complex visual cues into the final reasoning procedures.

![](images/e8b6719b30563acd07fe2d479d7495902a94b4ae0f00e93f33427eccf5e22f9c.jpg)

<details>
<summary>line chart</summary>

| Level | Gemini-3-pro | Qwen3-VL-8B | InternVL3.5-38B | InternVL3-2B | InternVL3-78B | InternVL3-14B |
|-------|--------------|-------------|-----------------|--------------|---------------|---------------|
| Level 0 | 50 | 10 | 0 | 0 | 30 | 10 |
| Level 1 | 40 | 10 | 0 | 10 | 20 | 30 |
| Level 2 | 40 | 20 | 30 | 0 | 20 | 20 |
| Level 3 | 30 | 20 | 20 | 10 | 10 | 0 |
</details>

Figure 5. Performance trends across difficulty levels (level 0-3). The scale-dependent effects of explicit information injection on model performance. While medium-scale models benefit from explicit guidance, large-scale and small-scale models exhibit stagnant or diminished performance due to conflicts with parametric priors and limited baseline capacity, respectively.

## 5. Alternative Views

A prevalent perspective in recent VLMs research is that the observed limitations in visual implicit reasoning are primarily a consequence of insufficient scale or suboptimal elicitation, rather than a fundamental shortcoming of current models. From this viewpoint, increasing model capacity, training data, or prompt specificity should naturally resolve the reported failures.

Scaling Capacity vs. Scaling Agency. One prominent hypothesis argues that implicit reasoning is an emergent capability that will reliably manifest as models continue to scale (Wei et al., 2022). Indeed, our results confirm that scaling consistently improves visual and physical knowledge, suggesting substantial gains in representational capacity. However, we observe that improvements in implicit visual reasoning are markedly slower and less stable (Huang et al., 2025). This divergence indicates that scaling preferentially enhances what a model knows, but does not guarantee when or why that knowledge is autonomously deployed. In this sense, scaling capacity does not equate to scaling agency.

Prompt Sensitivity and Evaluation Intent. Another alternative explanation attributes the observed failures to prompt design or evaluation artifacts, arguing that more explicit instructions or chain-of-thought prompts (Wei et al., 2022; Zhang et al., 2024) would elicit stronger reasoning behavior (Li et al., 2026b). While this interpretation is plausible, it overlooks the specific intent of our evaluation. Our focus on implicit reasoning is motivated by whether models can proactively uncover and exploit relevant information in an image without rich, task-specific guidance. Prompt minimalism is therefore a deliberate probe of default model behavior, rather than a limitation of the evaluation setup. If reasoning only emerges when explicitly requested, it reflects a gap between possessing knowledge and deploying it by default.

Scaling and prompting improve guided performance but do not enable autonomous visual reasoning. Addressing the failure to initiate reasoning in implicit settings necessitates solutions rooted in fundamental model architecture, rather than simple scale or guidance.

## 6. Discussion and Conclusion

This paper establishes a formal framework for visual implicit reasoning and reveals a fundamental limitation of current VLMs: a gap between passive recognition and the active agency required for genuine visual understanding. Both theoretical analysis and evidence from V-IRD show that contemporary models function primarily as probabilistic semantic retrievers rather than grounded visual reasoners.

Our analysis further indicates that this limitation is not due to missing atomic knowledge, but to a pronounced agency deficit. When faced with implicit instructions, models abandon situated visual measurement and default to internal memory. These findings suggest that scaling model size alone is insufficient; progress instead requires training objectives that promote autonomous visual discovery and active perception.

## Acknowledgement

This work was funded by the National Natural Science Foundation of China (Grant No. 62571379) and the Hubei Provincial Key Research and Development Program (Grant No. 2024BAB050). The numerical calculations in this paper have been done on the supercomputing system in the Supercomputing Center of Wuhan University.

## References

Anthropic. Claude sonnet 4.5 system card. Technical report, Anthropic, September 2025. URL https://www.anthropic.com/

claude-sonnet-4-5-system-card.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Ballard, D. H. Animate vision. Artificial intelligence, 48(1): 57–86, 1991.

Chen, B., Xu, Z., Kirmani, S., Ichter, B., Sadigh, D., Guibas, L., and Xia, F. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14455–14465, 2024.

Chia, Y. K., Toh, V., Ghosal, D., Bing, L., and Poria, S. Puzzlevqa: Diagnosing multimodal reasoning challenges of language models with abstract visual patterns. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 16259–16273, 2024.

Chow, W., Mao, J., Li, B., Seita, D., Campagnolo Guizilini, V., and Wang, Y. Physbench: Benchmarking and enhancing vision-language models for physical world understanding. In International Conference on Learning Representations, volume 2025, pp. 97959–98108, 2025.

Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N. A., Ma, W.-C., and Krishna, R. Blink: Multimodal large language models can see but not perceive. In European Conference on Computer Vision, pp. 148–166. Springer, 2024.

Gibson, J. The ecological approach to visual perception: classic edition, 2014.

Google DeepMind. Gemini 3: A new era of intelligence, November 2025. URL https://blog.google/ products/gemini/gemini-3.

Guan, T., Liu, F., Wu, X., Xian, R., Li, Z., Liu, X., Wang, X., Chen, L., Huang, F., Yacoob, Y., et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14375– 14385, 2024.

Huang, Y., He, Q., Chen, Z., Zhang, H., Yu, H., and Zhao, Z. Autonomous multimodal reasoning via implicit chainof-vision. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2963–2972, 2025.

Hutchins, E. Cognition in the Wild. MIT press, 1995.

Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al. Llavaonevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024a.

Li, D., Fang, Y., Chen, Y., Yang, S., Cao, S., Wong, J., Luo, M., Wang, X., Yin, H., Gonzalez, J., et al. Worldmodelbench: Judging video generation models as world models. Advances in Neural Information Processing Systems, 38, 2026a.  
Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., and Li, C. Llava-next-interleave: Tackling multiimage, video, and 3d in large multimodal models. arXiv preprint arXiv:2407.07895, 2024b.  
Li, X., Yu, Z., Zhang, Z., Chen, X., Zhang, Z., Zhuang, Y., Sadagopan, N., and Beniwal, A. When thinking fails: The pitfalls of reasoning for instruction-following in llms. Advances in Neural Information Processing Systems, 38: 77925–77962, 2026b.  
Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, X., and Wen, J.-R. Evaluating object hallucination in large vision-language models. In Proceedings of the 2023 conference on empirical methods in natural language processing, pp. 292–305, 2023.  
Li, Z., Yang, B., Liu, Q., Ma, Z., Zhang, S., Yang, J., Sun, Y., Liu, Y., and Bai, X. Monkey: Image resolution and text label are important things for large multi-modal models. In proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 26763–26773, 2024c.  
Liang, Y., Li, M., Fan, C., Li, Z., Nguyen, D., Cobbina, K., Bhardwaj, S., Chen, J., Liu, F., and Zhou, T. Colorbench: Can vlms see and understand the colorful world? a comprehensive benchmark for color perception, reasoning, and robustness. Advances in Neural Information Processing Systems, 38, 2026.  
Lin, Z., Liu, Y., Yang, Y., Tao, L., and Ye, D. Adaptvision: Efficient vision-language models via adaptive visual acquisition. arXiv preprint arXiv:2512.03794, 2025.  
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36:34892–34916, 2023.  
Lovenia, H., Dai, W., Cahyawijaya, S., Ji, Z., and Fung, P. Negative object presence evaluation (nope) to measure object hallucination in vision-language models. In Proceedings of the 3rd Workshop on Advances in Language and Vision Research (ALVR), pp. 37–58, 2024.  
Lu, P., Bansal, H., Xia, T., Liu, J., Li, C., Hajishirzi, H., Cheng, H., Chang, K.-W., Galley, M., and Gao, J. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. In International Conference on Learning Representations, volume 2024, pp. 23439– 23554, 2024.  
Rose, D., Himakunthala, V., Ouyang, A., He, R., Mei, A., Lu, Y., Saxon, M., Sonar, C., Mirza, D., and Wang, W. Y. Visual chain of thought: bridging logical gaps with multimodal infillings. arXiv preprint arXiv:2305.02317, 2023.  
Rostamkhani, M., Ansari, B., Sabzevari, H., Rahmani, F., and Eetemadi, S. Illusory vqa: Benchmarking and enhancing multimodal models on visual illusions. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 2995–3004, 2025.  
Seth, A., Manocha, D., and Agarwal, C. Hallucinogen: Benchmarking hallucination in implicit reasoning within large vision language models. In Proceedings of the 2nd Workshop on Uncertainty-Aware NLP (UncertaiNLP 2025), pp. 89–102, 2025.  
Singh, A., Fry, A., Perelman, A., Tart, A., Ganesh, A., El-Kishky, A., McLaughlin, A., Low, A., Ostrow, A., Ananthram, A., et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.  
Tong, P., Brown, E., Wu, P., Woo, S., IYER, A. J. V., Akula, S. C., Yang, S., Yang, J., Middepogu, M., Wang, Z., et al. Cambrian-1: A fully open, vision-centric exploration of multimodal llms. Advances in Neural Information Processing Systems, 37:87310–87356, 2024.  
Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025a.  
Wang, Z., Chen, C., Luo, F., Dong, Y., Zhang, Y., Xu, Y., Wang, X., Li, P., and Liu, Y. Actiview: Evaluating active perception ability for multimodal large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7605–7633, 2025b.  
Wei, J., Wang, X., Schuurmans, D., Bosma, M., Xia, F., Chi, E., Le, Q. V., Zhou, D., et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in neural information processing systems, 35:24824–24837, 2022.  
Weng, T., Wang, J., Jiang, W., and Ming, Z. Visnumbench: Evaluating number sense of multimodal large language models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3830–3840, 2025.  
Wu, P. and Xie, S. V\*: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13084–13094, 2024.  
Xia, P., Han, S., Qiu, S., Zhou, Y., Wang, Z., Zheng, W., Chen, Z., Cui, C., Ding, M., Li, L., et al. Mmie: Massive multimodal interleaved comprehension benchmark for large vision-language models. In International Conference on Learning Representations, volume 2025, pp. 25842–25875, 2025.  
Xu, X., Xu, Q., Xiao, T., Chen, T., Yan, Y., Zhang, J., Diao, S., Yang, C., and Wang, Y. Ugphysics: A comprehensive benchmark for undergraduate physics reasoning with large language models. In International Conference on Machine Learning, pp. 69849–69877. PMLR, 2025.  
Xue, H., Ge, Y., Zeng, Y., Li, Z., Liu, M.-Y., Chen, Y., and Fan, J. Point-it-out: Benchmarking embodied reasoning for vision language models in multi-stage visual grounding. arXiv preprint arXiv:2509.25794, 2025.  
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.  
Yang, L., Kang, B., Huang, Z., Xu, X., Feng, J., and Zhao, H. Depth anything: Unleashing the power of large-scale unlabeled data. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10371–10381, 2024.  
Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.  
Yue, X., Zheng, T., Ni, Y., Wang, Y., Zhang, K., Tong, S., Sun, Y., Yu, B., Zhang, G., Sun, H., et al. Mmmu-pro: A more robust multi-discipline multimodal understanding benchmark. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 15134–15186, 2025.  
Zhang, J., Yao, D., Pi, R., Liang, P. P., and Fung, Y. R. Vlm2- bench: A closer look at how well vlms implicitly link explicit matching visual cues. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7510–7545, 2025a.  
Zhang, X., Dong, Y., Wu, Y., Huang, J., Jia, C., Fernando, B., Shou, M. Z., Zhang, L., and Liu, J. Physreason: A comprehensive benchmark towards physics-based reasoning. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16593–16615, 2025b.  
Zhang, Z., Zhang, A., Li, M., Zhao, H., Karypis, G., and Smola, A. Multimodal chain-of-thought reasoning in language models. Transactions on Machine Learning Research, 2024, 2024.  
Zhao, Q., Lu, Y., Kim, M. J., Fu, Z., Zhang, Z., Wu, Y., Li, Z., Ma, Q., Han, S., Finn, C., et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1702–1713, 2025.  
Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., and Yu, X. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.  
Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

## A. Supplementary Quantitative Analysis

To systematically investigate whether the observed performance deficits arise from strict metric precision requirements or a fundamental lack of visual grounding, we analyze model performance under relaxed error tolerances (δ ∈ {20%, 30%}), as detailed in Table 4. This ablation effectively isolates calibration sensitivity from reasoning capability. The results demonstrate a distinct performance bifurcation that implies different error mechanisms across domains.

In semantic-centric tasks, specifically Physical Properties and Contextual Inference, relaxing the error threshold yields immediate and substantial performance gains. Proprietary models such as GPT-5.2 and Gemini-3-Pro approach ceiling performance (near 100% accuracy) even at the 20% threshold. This rapid saturation suggests that failures in these domains are largely attributable to minor calibration variances rather than flawed reasoning logic.

Conversely, Spatial Geometry tasks exhibit a structural failure mode that is resistant to threshold relaxation. Even with a generous 30% tolerance, performance in Volume and Distance estimation remains critically low. For instance, InternVL3- 78B attains only ∼ 33% accuracy on Volume estimation at the 30% threshold, in sharp contrast to its 95% accuracy on Temperature estimation. This persistent stagnation suggests that the error in spatial tasks is, to a certain degree, rooted in foundational deficiencies. Instead of actively establishing a visual reference scale, models revert to unimodal parametric priors, resulting in hallucinations that deviate fundamentally from the visual reality, a limitation that cannot be masked by simply loosening evaluation constraints.

## B. Inference Prompts

## B.1. Universal System Prompt

All models are conditioned with a unified system instruction.

## System Instruction Prompt

Role: You are a multimodal AI assistant specializing in precise physical reasoning and geometric estimation.

Core Directive: Answer the user’s question by deriving the result strictly from the visual content provided. You must first provide an explicit reasoning process explaining how you calculated or deduced the answer, grounded solely in observable pixel data.

## Negative Constraints:

1. Do NOT rely on generic parametric knowledge (e.g., “standard sizes”) if it conflicts with the visual evidence.  
2. Do NOT output ranges (e.g., “20-30cm”) or uncertainty terms (e.g., “approx”, “maybe”).  
3. Do NOT provide verbose conclusions; output strictly the requested value.

Output Schema: You must output a single JSON object. The content of the “answer” field must start with the exact phrase “The answer is”.

{ "reasoning":"Step-by-step derivation...", "answer": "The answer is [Exact Value]" }

## B.2. Domain-Specific Evaluation Prompts

We categorize the benchmark into four reasoning quadrants. For each instance, we employ a dual-query strategy to ablate the efficacy of explicit reasoning triggers.

## Q1. Physical Properties

Evaluates the capacity to deduce invisible object states (e.g., thermal properties) via implicit secondary visual artifacts like condensation, rather than semantic recognition.

Standard Prompt: "What is the temperature (◦C) of the water in the picture?"

CoT Prompt: "What is the temperature (◦C) of the water in the picture? Please think step by step."

Table 4. Supplementary results on the V-IRD (20% and 30% thresholds). We evaluate a wide range of VLMs across four distinct domains under relaxed constraints. The values in parentheses denote the Accuracy achieved under relative error thresholds of 20% and 30%, respectively. The Average column represents the mean accuracy across all tasks for these two thresholds. Within each model group, the best score per column is highlighted in bold. Across all VLM groups, the top-3 entries per column are shaded with decreasing intensity: 1st , 2nd , 3rd .

<table><tr><td rowspan="2">Model</td><td colspan="4">Spatial Geometry</td><td colspan="2">Physical Properties</td><td colspan="2">Physical Logic</td><td colspan="2">Contextual Inference</td><td rowspan="2">Average</td></tr><tr><td>Length</td><td>Area</td><td>Volume</td><td>Distance</td><td>Temp.</td><td>Weight</td><td>Electricity</td><td>Kinematics</td><td>Remark</td><td>Env.</td></tr><tr><td colspan="12">Open-source VLMs : &lt; 7B</td></tr><tr><td>InternVL3.5-1B (Wang et al., 2025a)</td><td>(11.43, 18.57)</td><td>(16.67, 16.67)</td><td>(4.17, 12.50)</td><td>(34.62, 46.15)</td><td>(59.52, 64.29)</td><td>(42.31, 50.00)</td><td>(11.11, 11.11)</td><td>(50.00, 66.67)</td><td>(16.67, 16.67)</td><td>(43.48, 43.48)</td><td>(29.00, 34.61)</td></tr><tr><td>InternVL3.5-2B (Wang et al., 2025a)</td><td>(20.00, 24.29)</td><td>(25.00, 25.00)</td><td>(12.50, 20.83)</td><td>(7.69, 26.92)</td><td>(76.19, 78.57)</td><td>(38.46, 50.00)</td><td>(38.89, 38.89)</td><td>(33.33, 58.33)</td><td>(33.33, 33.33)</td><td>(56.52, 56.52)</td><td>(34.19, 41.27)</td></tr><tr><td>InternVL3-1B (Zhu et al., 2025)</td><td>(14.29, 22.86)</td><td>(33.33, 33.33)</td><td>(12.50, 16.67)</td><td>(19.23, 23.08)</td><td>(35.71, 35.71)</td><td>(26.92, 38.46)</td><td>(11.11, 11.11)</td><td>(41.67, 41.67)</td><td>(33.33, 33.33)</td><td>(52.17, 52.17)</td><td>(28.03, 30.84)</td></tr><tr><td>InternVL3-2B (Zhu et al., 2025)</td><td>(25.71, 35.71)</td><td>(0.00, 0.00)</td><td>(20.83, 20.83)</td><td>(23.08, 30.77)</td><td>(85.71, 85.71)</td><td>(50.00, 57.69)</td><td>(55.56, 61.11)</td><td>(58.33, 58.33)</td><td>(56.25, 60.42)</td><td>(54.35, 54.35)</td><td>(42.98, 46.50)</td></tr><tr><td>Qwen3-VL-2B (Yang et al., 2025)</td><td>(27.14, 34.29)</td><td>(25.00, 25.00)</td><td>(20.83, 25.00)</td><td>(11.54, 23.08)</td><td>(73.81, 76.19)</td><td>(73.08, 76.92)</td><td>(38.89, 38.89)</td><td>(33.33, 50.00)</td><td>(47.92, 50.00)</td><td>(54.35, 54.35)</td><td>(40.59, 45.37)</td></tr><tr><td>Qwen3-VL-4B (Yang et al., 2025)</td><td>(47.14, 55.71)</td><td>(16.67, 16.67)</td><td>(20.83, 29.17)</td><td>(0.00, 0.00)</td><td>(73.81, 78.57)</td><td>(92.31, 92.31)</td><td>(50.00, 50.00)</td><td>(50.00, 50.00)</td><td>(25.00, 25.00)</td><td>(67.39, 67.39)</td><td>(44.32, 46.48)</td></tr><tr><td>Qwen2.5-VL-3B (Bai et al., 2025)</td><td>(42.86, 45.71)</td><td>(41.67, 41.67)</td><td>(16.67, 25.00)</td><td>(26.92, 30.77)</td><td>(76.19, 80.95)</td><td>(46.15, 53.85)</td><td>(33.33, 33.33)</td><td>(25.00, 25.00)</td><td>(35.42, 37.50)</td><td>(60.87, 60.87)</td><td>(40.51, 43.47)</td></tr><tr><td colspan="12">Open-source VLMs : 7-8B</td></tr><tr><td>Qwen3-VL-8B (Yang et al., 2025)</td><td>(44.29, 55.71)</td><td>(25.00, 33.33)</td><td>(20.83, 33.33)</td><td>(19.23, 19.23)</td><td>(92.86, 95.24)</td><td>(73.08, 80.77)</td><td>(44.44, 44.44)</td><td>(58.33, 58.33)</td><td>(37.50, 41.67)</td><td>(86.96, 86.96)</td><td>(50.25, 54.90)</td></tr><tr><td>Qwen2.5-VL-7B (Bai et al., 2025)</td><td>(38.57, 50.00)</td><td>(25.00, 41.67)</td><td>(8.33, 12.50)</td><td>(23.08, 42.31)</td><td>(52.38, 52.38)</td><td>(76.92, 84.62)</td><td>(27.78, 27.78)</td><td>(16.67, 16.67)</td><td>(33.33, 33.33)</td><td>(65.22, 65.22)</td><td>(36.73, 42.65)</td></tr><tr><td>InternVL3.5-8B (Wang et al., 2025a)</td><td>(41.43, 44.29)</td><td>(16.67, 33.33)</td><td>(25.00, 29.17)</td><td>(11.54, 15.38)</td><td>(90.48, 95.24)</td><td>(65.38, 69.23)</td><td>(38.89, 38.89)</td><td>(50.00, 75.00)</td><td>(52.08, 52.08)</td><td>(71.74, 71.74)</td><td>(46.32, 52.44)</td></tr><tr><td>InternVL3-8B (Zhu et al., 2025)</td><td>(28.57, 45.71)</td><td>(33.33, 33.33)</td><td>(25.00, 25.00)</td><td>(11.54, 23.08)</td><td>(61.90, 66.67)</td><td>(57.69, 57.69)</td><td>(22.22, 22.22)</td><td>(83.33, 91.67)</td><td>(68.75, 72.92)</td><td>(73.91, 76.09)</td><td>(46.63, 51.44)</td></tr><tr><td>LLaVA-OV-7B (Li et al., 2024a)</td><td>(42.86, 54.29)</td><td>(8.33, 16.67)</td><td>(12.50, 16.67)</td><td>(23.08, 26.92)</td><td>(59.52, 69.05)</td><td>(46.15, 57.69)</td><td>(5.56, 5.56)</td><td>(58.33, 66.67)</td><td>(52.08, 52.08)</td><td>(71.74, 71.74)</td><td>(38.02, 43.73)</td></tr><tr><td colspan="12">Open-source VLMs : 10-30B</td></tr><tr><td>InternVL3.5-14B (Wang et al., 2025a)</td><td>(41.43, 50.00)</td><td>(16.67, 33.33)</td><td>(12.50, 20.83)</td><td>(26.92, 34.62)</td><td>(80.95, 85.71)</td><td>(65.38, 73.08)</td><td>(27.78, 27.78)</td><td>(83.33, 100.00)</td><td>(70.83, 70.83)</td><td>(69.57, 69.57)</td><td>(49.54, 56.58)</td></tr><tr><td>InternVL3-14B (Zhu et al., 2025)</td><td>(42.86, 54.29)</td><td>(8.33, 8.33)</td><td>(45.83, 45.83)</td><td>(46.15, 46.15)</td><td>(85.71, 85.71)</td><td>(73.08, 80.77)</td><td>(38.89, 38.89)</td><td>(58.33, 91.67)</td><td>(62.50, 62.50)</td><td>(73.91, 73.91)</td><td>(53.56, 58.81)</td></tr><tr><td colspan="12">Open-source VLMs : 30-70B</td></tr><tr><td>InternVL3-38B (Zhu et al., 2025)</td><td>(44.29, 61.43)</td><td>(25.00, 33.33)</td><td>(20.83, 37.50)</td><td>(38.46, 53.85)</td><td>(95.24, 95.24)</td><td>(84.62, 92.31)</td><td>(33.33, 33.33)</td><td>(66.67, 83.33)</td><td>(66.67, 70.83)</td><td>(78.26, 78.26)</td><td>(55.34, 63.94)</td></tr><tr><td>InternVL3.5-38B (Wang et al., 2025a)</td><td>(45.71, 50.00)</td><td>(16.67, 41.67)</td><td>(20.83, 29.17)</td><td>(34.62, 34.62)</td><td>(90.48, 95.24)</td><td>(92.31, 92.31)</td><td>(55.56, 55.56)</td><td>(66.67, 100.00)</td><td>(64.58, 66.67)</td><td>(76.09, 80.43)</td><td>(56.35, 64.57)</td></tr><tr><td>Qwen2.5-VL-32B (Bai et al., 2025)</td><td>(40.00, 45.71)</td><td>(25.00, 41.67)</td><td>(8.33, 16.67)</td><td>(15.38, 15.38)</td><td>(78.57, 80.95)</td><td>(80.77, 92.31)</td><td>(44.44, 44.44)</td><td>(58.33, 75.00)</td><td>(54.17, 58.33)</td><td>(69.57, 69.57)</td><td>(47.46, 54.00)</td></tr><tr><td>Qwen3-VL-32B (Yang et al., 2025)</td><td>(34.29, 47.14)</td><td>(33.33, 41.67)</td><td>(16.67, 25.00)</td><td>(19.23, 26.92)</td><td>(95.24, 95.24)</td><td>(80.77, 96.15)</td><td>(44.44, 44.44)</td><td>(66.67, 75.00)</td><td>(56.25, 58.33)</td><td>(84.78, 84.78)</td><td>(53.17, 59.47)</td></tr><tr><td colspan="12">Open-source VLMs : 70-80B</td></tr><tr><td>LLaVA-NEXT-72B (Li et al., 2024b)</td><td>(52.86, 54.29)</td><td>(25.00, 33.33)</td><td>(12.50, 12.50)</td><td>(11.54, 15.38)</td><td>(73.81, 83.33)</td><td>(38.46, 38.46)</td><td>(11.11, 11.11)</td><td>(33.33, 50.00)</td><td>(20.83, 20.83)</td><td>(54.35, 54.35)</td><td>(33.38, 37.36)</td></tr><tr><td>LLaVA-OV-72B (Li et al., 2024a)</td><td>(51.43, 60.00)</td><td>(16.67, 41.67)</td><td>(33.33, 33.33)</td><td>(11.54, 19.23)</td><td>(90.48, 95.24)</td><td>(69.23, 73.08)</td><td>(38.89, 38.89)</td><td>(66.67, 83.33)</td><td>(52.08, 56.25)</td><td>(78.26, 78.26)</td><td>(50.86, 57.93)</td></tr><tr><td>Qwen2.5-VL-72B (Bai et al., 2025)</td><td>(41.43, 47.14)</td><td>(50.00, 50.00)</td><td>(0.00, 16.67)</td><td>(7.69, 11.54)</td><td>(88.10, 92.86)</td><td>(88.46, 96.15)</td><td>(50.00, 50.00)</td><td>(50.00, 75.00)</td><td>(66.67, 68.75)</td><td>(78.26, 78.26)</td><td>(52.06, 58.64)</td></tr><tr><td>InternVL3-78B (Zhu et al., 2025)</td><td>(52.86, 65.71)</td><td>(25.00, 33.33)</td><td>(20.83, 33.33)</td><td>(30.77, 50.00)</td><td>(90.48, 95.24)</td><td>(80.77, 80.77)</td><td>(22.22, 22.22)</td><td>(66.67, 83.33)</td><td>(75.00, 79.17)</td><td>(84.78, 84.78)</td><td>(54.94, 62.79)</td></tr><tr><td colspan="12">Open-source VLMs : &gt;200B</td></tr><tr><td>Qwen3-VL-235B (Yang et al., 2025)</td><td>(44.29, 55.71)</td><td>(25.00, 33.33)</td><td>(25.00, 41.67)</td><td>(23.08, 30.77)</td><td>(100.00, 100.00)</td><td>(88.46, 92.31)</td><td>(50.00, 50.00)</td><td>(75.00, 83.33)</td><td>(64.58, 68.75)</td><td>(84.78, 84.78)</td><td>(58.02, 64.07)</td></tr><tr><td colspan="12">Proprietary VLMs</td></tr><tr><td>GPT-5.2 (Singh et al., 2025)</td><td>(44.29, 71.43)</td><td>(33.33, 41.67)</td><td>(16.67, 25.00)</td><td>(7.69, 7.69)</td><td>(100.00, 100.00)</td><td>(84.62, 96.15)</td><td>(55.56, 55.56)</td><td>(58.33, 83.33)</td><td>(79.17, 79.17)</td><td>(82.61, 82.61)</td><td>(56.23, 64.26)</td></tr><tr><td>Claude-Sonnet-4.5 (Anthropic, 2025)</td><td>(70.00, 72.86)</td><td>(41.67, 41.67)</td><td>(54.17, 58.33)</td><td>(7.69, 15.38)</td><td>(90.48, 95.24)</td><td>(73.08, 76.92)</td><td>(44.44, 44.44)</td><td>(75.00, 100.00)</td><td>(83.33, 83.33)</td><td>(86.96, 86.96)</td><td>(62.68, 67.51)</td></tr><tr><td>Claude-Sonnet-4.5-Thinking (Anthropic, 2025)</td><td>(70.00, 78.57)</td><td>(25.00, 33.33)</td><td>(25.00, 33.33)</td><td>(30.77, 34.62)</td><td>(90.48, 95.24)</td><td>(73.08, 73.08)</td><td>(33.33, 33.33)</td><td>(83.33, 100.00)</td><td>(83.33, 83.33)</td><td>(91.30, 91.30)</td><td>(60.56, 65.61)</td></tr><tr><td>Gemini-3-Flash (Google DeepMind, 2025)</td><td>(70.00, 85.71)</td><td>(75.00, 75.00)</td><td>(50.00, 75.00)</td><td>(46.15, 65.38)</td><td>(100.00, 100.00)</td><td>(73.08, 80.77)</td><td>(66.67, 66.67)</td><td>(83.33, 100.00)</td><td>(83.33, 83.33)</td><td>(84.78, 84.78)</td><td>(73.23, 81.67)</td></tr><tr><td>Gemini-3-Pro (Google DeepMind, 2025)</td><td>(81.43, 81.43)</td><td>(58.33, 75.00)</td><td>(45.83, 54.17)</td><td>(23.08, 38.46)</td><td>(100.00, 100.00)</td><td>(88.46, 92.31)</td><td>(66.67, 66.67)</td><td>(100.00, 100.00)</td><td>(81.25, 81.25)</td><td>(91.30, 91.30)</td><td>(73.64, 78.06)</td></tr></table>

## Q2. Physical Logic

Probes the synthesis of visual perception with fundamental physical laws. The model must actively extract variables and apply principles such as Ohm’s Law or rigid body dynamics.

Standard Prompt: "How many grams does the bread in the picture weigh in total?"

CoT Prompt: "How many grams does the bread in the picture weigh in total? Please think step by step."

## Q3. Contextual Inference

Targets visual commonsense, requiring the identification of geolocation, institutional identity, or cultural context from fine-grained visual markers rather than salient foreground objects.

Standard Prompt: "What is the dialect of the region depicted in the picture?"

CoT Prompt: "What is the dialect of the region depicted in the picture? Please think step by step."

## Q4. Spatial Geometry

Addresses the agency in spatial understanding, requiring active identification of reference objects and precise metric estimation (distance, volume) of targets.

Standard Prompt: "What is the capacity (ml) of the bottle in the picture?"

CoT Prompt: "What is the capacity (ml) of the bottle in the picture? Please think step by step."

Table 5. Hierarchical Taxonomy and Statistics of the V-IRD Benchmark. The benchmark covers four primary domains and 10 fine-grained sub-tasks. For each sub-task, the sample count (#), a precise definition, and a representative query are reported to illustrate the diversity of physical and spatial reasoning challenges.

<table><tr><td>Sub-task</td><td>#</td><td>Description</td><td>Sample Questions</td></tr><tr><td colspan="4">Domain I: Spatial Geometry</td></tr><tr><td>Length</td><td>35</td><td>Measure the linear dimension of a target object relative to visual reference scales.</td><td>What is the length (cm) of the pencil on the desk?</td></tr><tr><td>Distance</td><td>13</td><td>Quantify the spatial interval between two distinct entities in 3D space.</td><td>What is the distance (cm) between the ID card and the passport?</td></tr><tr><td>Volume</td><td>12</td><td>Estimate the fluid capacity or displacement volume of containers based on geometry.</td><td>What is the capacity (ml) of the transparent bottle?</td></tr><tr><td>Area</td><td>6</td><td>Calculate the 2D surface coverage of specific planar regions or screens.</td><td>What is the area ( $cm^2$ ) of the computer monitor?</td></tr><tr><td colspan="4">Domain II: Contextual Inference</td></tr><tr><td>Remark</td><td>24</td><td>Identify specific entity metadata, such as airline brands, logos, or institutional names.</td><td>Which country&#x27;s airline is the person in the picture taking?</td></tr><tr><td>Environment</td><td>23</td><td>Infer geolocation, temporal context, or cultural dialects from environmental markers.</td><td>What is the characteristic dialect of the region de-picted?</td></tr><tr><td colspan="4">Domain III: Physical Properties</td></tr><tr><td>Temperature</td><td>21</td><td>Deduce thermal states from phase-change artifacts (e.g., steam, ice, condensation).</td><td>What is the temperature (°C) of the alcohol in the beaker?</td></tr><tr><td>Weight</td><td>13</td><td>Estimate object mass by integrating visual material properties and approximate volume.</td><td>How many grams does the bread weigh in total?</td></tr><tr><td colspan="4">Domain IV: Physical Logic</td></tr><tr><td>Electricity</td><td>9</td><td>Apply abstract circuit theories (e.g., Ohm&#x27;s Law) to visual component states.</td><td>What is the resistance (Ω) of the light bulb filament?</td></tr><tr><td>Kinematics</td><td>6</td><td>Analyze force equilibrium, torque requirements, or motion trajectories.</td><td>What is the minimum force (N) required to balance the lever?</td></tr></table>

## C. Task Definitions and Dataset Statistics

To rigorously evaluate the physical and spatial agency of Multimodal Large Language Models (MLLMs), we introduce V-IRD, a benchmark grounded in a hierarchical taxonomy. As illustrated in Figure 3, the dataset is stratified into four primary reasoning domains comprising 10 fine-grained sub-tasks. Table 5 provides a comprehensive breakdown of the statistics, definitions, and representative queries for each category.

## Hierarchical Reasoning Domains

We categorize the reasoning challenges into the following four quadrants, designed to probe distinct facets of visual intelligence:

1. Spatial Geometry. This domain addresses the “Agency Deficit” in metric spatial understanding. Unlike generic object detection, tasks in this category (including Length, Distance, Volume, and Area) require the model to establish an internal metric scale from visual references. As shown in Table 5, this is the largest category in our benchmark, assessing the fundamental capability of implicit information seeking and precise metric estimation from 2D pixel inputs.  
2. Contextual Inference. Beyond salient foreground objects, this domain targets “Visual Commonsense.” It requires the active discovery and identification of subtle cues, which include institutional logos (Remark) or environmental markers such as vegetation and architecture (Environment), to deduce geolocation, temporal context, or cultural identity.

3. Physical Properties. This domain evaluates the capacity to actively deduce invisible object states via secondary visual artifacts. For instance, the Temperature task requires inferring thermal states from steam or condensation, while the Weight task demands the synthesis of estimated volume and material density.

4. Physical Logic. Representing the high level of abstract reasoning, this domain probes the synthesis of visual perception with fundamental physical laws. Models must actively extract implicit visual variables and apply principles such as Ohm’s Law (Electricity) or rigid body dynamics (Kinematics) to solve complex reasoning problems.

## D. Human Evaluation

To assess the degree of alignment between VLMs and human physical understanding, and to establish a robust high-quality reference for the V-IRD benchmark, a human performance evaluation was conducted.

We recruited 3 human participants, all holding bachelor’s degrees with engineering or science backgrounds, representing competent human cognition and reasoning ability in physical and spatial tasks. The assessment process did not set strict time limits, and the average completion time was approximately 5 hours. To guarantee statistical reliability, the participants worked collaboratively to complete two full iterations of the entire V-IRD benchmark. Specifically, the workload was distributed among the evaluators such that the dataset was fully annotated two times, providing a consistent consensus metric for human accuracy.

Importantly, the images and text prompts shown to the evaluators were strictly consistent with the inputs provided to the models. This setting aimed to place humans and the models at comparable initial conditions at the input stage, thereby minimizing potential information leakage. We then aggregated these consensus responses to derive the final human performance scores reported in Table 2 and Table 4.