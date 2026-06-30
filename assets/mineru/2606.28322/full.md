# PerceptionRubrics: Calibrating Multimodal Evaluation to Human Perception

Yana Wei <sup>\*</sup> <sup>1</sup> Hongbo Peng <sup>\*</sup> <sup>2</sup> Yanlin Lai <sup>\*</sup> <sup>3</sup> Liang Zhao <sup>2</sup> Kangheng Lin <sup>2</sup> En Yu <sup>2</sup> Keyu Lv <sup>2</sup> Han Zhou <sup>2</sup> Yin Tang <sup>4</sup> Haodong Li <sup>2</sup> Mitt Huang <sup>2</sup> Hangyu Guo <sup>2</sup> Jianjian Sun <sup>2</sup> Zheng Ge <sup>2</sup> Xiangyu Zhang <sup>2</sup> Daxin Jiang <sup>2</sup> Vishal M. Patel <sup>1</sup>

## Abstract

We introduce PERCEPTIONRUBRICS, a rubricbased evaluation framework that addresses the gap between saturated benchmark scores and realworld brittleness. Shifting evaluation from holistic semantic matching to rigorous atomic auditing, PERCEPTIONRUBRICS pairs 1,038 informationdense images with over 12,000 instance-specific rubrics. These criteria are derived from golden captions constructed via a novel Circular Peer-Review consensus pipeline and then distilled into a dual-stream system of Must-Right (essential facts) and Easy-Wrong (fine-grained details) rubrics. Crucially, PERCEPTIONRUBRICS implements a Gated Scoring mechanism: unlike linear averages, failure on mandatory visual facts triggers sharp binary penalties. Extensive evaluation yields critical insights: (1) The Reliability Gap: models often verify fragmented elements correctly yet fail strict conjunctive constraints, exposing brittleness in dense domains; (2) Open-Closed Stratification: contrary to reasoning trends, we reveal a persistent 8% perception deficit between open-source and proprietary frontiers; and (3) Human-Aligned Rigor: our gated metrics substantially out-align conventional benchmarks, validating that strict perceptual fidelity is the prerequisite for reliable generation. Code and data can be found at project page.

## 1. Introduction

Despite the rapid evolution of Multimodal Large Language Models (MLLMs), a fundamental evaluation crisis persists: current perception benchmarks do not reliably reflect genuine perceptual capability. This has led to a evaluation paradox where leaderboards are increasingly saturated in the high-score regime as illustrated in Figure 1, yet models remain perceptually brittle in real-world deployment. Toptier systems often appear nearly tied on metrics but exhibit drastically different failure modes—such as miscounting objects or inverting spatial relations—that are highly salient to users even when reported metric scores (Dong et al., 2024) remain high. This discrepancy suggests that benchmark rewards are misaligned with human perceptual sensitivity, creating a false sense of progress and failing to provide the diagnostic resolution needed to steer the next generation of MLLMs.

![](images/340fd225a1e8437d1df99decdd1d1a18477e3eb81d731cd88929e8fa21d5afbd.jpg)

![](images/fb994fa87b6cf247932683caa1197c6574c9425096ef0291754c23f46c554773.jpg)  
Figure 1. Motivation of PERCEPTIONRUBRICS. Top: An existing benchmark favors GPT-4o despite key omissions, while humans prefer responses that capture more perceptually important details. Bottom: Compared with DetailCaps and DOCCI, PER-CEPTIONRUBRICS more clearly distinguishes model capabilities.

We trace this failure to two systemic flaws in current benchmark design. First, the visual content and task design lack sufficient perceptual detail coverage. Many benchmarks rely on information-poor images or narrow domains (Onoe et al., 2024), often framing tasks as closed-form questions that allow models to “shortcut” through linguistic priors rather than genuine visual grounding (Zhou et al., 2023; Zhang et al., 2025). Even in open-ended captioning, references are frequently imprecise, biased, or too sparse (Dong et al., 2024) to challenge the long-tail visual knowledge of frontier models. Second, current reward signals are fundamentally uncalibrated. Conventional metrics, such as single-number similarity scores (e.g., CLIPScore (Radford et al., 2021)) or averaged multi-aspect schemes (Dong et al., 2024), rely on linear averaging that effectively “dilutes” fatal localized errors with general semantic overlap. Consequently, a caption plagued by hallucinations can still achieve a high metric score, severing the link between numerical performance and genuine reliability. In contrast, human perception is strictly non-linear: a single-digit hallucination in a financial table is not a permissible fluctuation but a binary failure (Poznanski et al., 2025). Existing metrics fail to reflect this, making it difficult to distinguish acceptable descriptive variation from critical perceptual failures.

![](images/f967eb815e6a50b401293c5e862f2b9a522a47be36e006b90f855424c919fba9.jpg)  
Figure 2. Rubric Demonstration of PERCEPTIONRUBRICS. Representative examples are selected for each task, highlighting “Must Right” ( ; essential features) and “Easy Wrong” pitfalls ( ; error-prone fine-grained details).

To bridge this gap, we propose PERCEPTIONRUBRICS, a benchmark that repurposes image captioning—the most fundamental proxy for integrated perception, recognition, and reasoning—into a rigorous diagnostic testbed. To address the data deficit, we curate 1,038 images characterized by extreme information density and distributional diversity. Crucially, to bypass the visual grounding gap that limits direct image-to-rubric generation, we adopt a caption-centric construction pipeline as an intermediary strategy. Instead of relying on noisy raw predictions, we establish ground truth via a Circular Peer-Review consensus mechanism: an ensemble of state-of-the-art MLLMs iteratively critiques and refines descriptions, followed by human verification. This process yields “Golden Captions” that serve as highfidelity textual references for the visual content, filtering out the noise and biases prevalent in traditional datasets.

Building on this foundation, we address the calibration gap by distilling Golden Captions into a granular, rubric-based auditing system. We extract over 12,000 atomic rubrics and organize them into two complementary streams: Must-Right rubrics, which capture essential visual facts that a response must satisfy, and Easy-Wrong rubrics, which target common hallucinations, omissions, and misinterpretations mined from model error patterns. We then introduce a gated scoring mechanism calibrated to human sensitivity: the Must-Right rubrics serve as mandatory gatekeepers, so failure to satisfy any essential criterion sharply penalizes the final score. This design ensures that the metric reflects not just coarse semantic proximity, but genuine perceptual reliability, effectively distinguishing between acceptable approximations and catastrophic failures.

Comprehensive evaluation and analysis across leading MLLMs on PERCEPTIONRUBRICS yields critical insights:

• Unveiling the “Reliability Gap”. We expose a disconnect between fragmented recognition and coherent understanding: models often pass atomic checks but fail strict conjunctive constraints. This reveals that despite high partial scores, current MLLMs lack the perceptual consistency required for information-dense domains like GUIs.

• Quantifying the Open-Closed Gap. Contrasting the convergence in reasoning tasks, we identify a persistent 8% perception deficit between the open-source frontier (e.g., Qwen3.5 (Team, 2026a)) and proprietary leaders (e.g., Seed-2.0 (ByteDance-Seed, 2026c)). Basic visual precision thus remains a decisive bottleneck distinguishing intrinsic model capacity.

• Superior Human Alignment. PERCEPTIONRUBRICS aligns substantially better with human judgment than conventional benchmarks (e.g., DOCCI (Onoe et al., 2024)), an effect amplified by our gated scoring. Furthermore, a near-perfect correlation between basic perception and hallucination resistance confirms strict fidelity as a prerequisite for reliable generation.

![](images/3da18bc00fed877fdc277bada0579be96dcb10df35cdb6afd5ad55acb827477d.jpg)  
Figure 3. Benchmark Statistics of PERCEPTIONRUBRICS: The distribution of tasks across 7 main categories.

## 2. Related Work

Visual Perception Benchmarks in MLLMs. Evaluating visual perception remains pivotal for assessing MLLMs (Team, 2025; OpenAI, 2025a). Current benchmarks generally fall into two categories: holistic suites and task-specific datasets. Comprehensive frameworks like MM-Bench (Liu et al., 2024b), MM-Vet (Yu et al., 2023), and MME (Fu et al., 2024) evaluate broad capabilities but increasingly face leaderboard saturation in recent flagship models (Bai et al., 2025a; Huang et al., 2026). Conversely, task-specific benchmarks target distinct skills, such as OCR in OCRBench (Liu et al., 2024c), open-world recognition in SimpleVQA (Cheng et al., 2025b) and spatial understanding in VSIBench (Yang et al., 2025). However, these benchmarks heavily rely on closed-ended formats (e.g., single or multiple-choice). Such designs often allow models to exploit linguistic priors or random guessing to bypass genuine visual grounding (Zhou et al., 2023; Zhang et al., 2025), limiting their ability to diagnose perceptual brittleness.

Evaluation of Image Captioning. Image captioning serves as a holistic proxy for perception, requiring models to autonomously prioritize and describe visual elements. Recent methods have moved beyond generic similarity metrics (Papineni et al., 2002) or object-set matching heuristics (Rohrbach et al., 2018) towards model-based evaluation. DOCCI (Onoe et al., 2024) targets detailed description using reference-based metrics; DetailCaps (Dong et al., 2024) employs multi-expert annotation to score object and attribute matching; RePer (Wei et al., 2025) utilizes an LLM-judge for aspect-based evaluation; and CapArena (Cheng et al., 2025a) aligns assessments with human preference via pairwise battles. Despite these advancements, a critical gap persists: existing methods often rely on sparse, biased references and linear scoring mechanisms that dilute fatal localized hallucinations with high holistic similarity, failing to reflect the non-linear sensitivity of human verification (Poznanski et al., 2025).

Rubric-Based Reward Modeling. To improve evaluation reliability, the field is shifting from opaque scalar scoring (Liu et al., 2024a) to rubric-based auditing. In text generation, structured criteria have effectively mitigated reward hacking (Rezaei et al., 2025). Approaches like RM-R1 (Chen et al., 2025) and SPCT (Liu et al., 2025) formulate evaluation as a reasoning process via chain-ofrubrics, while frameworks such as RaR (Gunjal et al., 2025) and ResearchRubrics(Sharma et al., 2025) leverage LLMs to decompose subjective judgments into atomic, verifiable checks. While this paradigm has standardized text-centric evaluation, comparable fine-grained auditing systems for multimodal perception remain under-explored. Existing vision benchmarks lack the mechanism to decompose complex visual scenes into verifiable atomic facts, highlighting the need for a rigorous standard to distinguish precise perception from approximation.

## 3. PerceptionRubrics

To align multimodal evaluation with the rigor of human judgment, we first outline our guiding design principles (Section 3.1) and data curation strategy (Section 3.2), followed by our novel caption-centric pipeline for generating atomic rubrics (Section 3.3) and the gated scoring mechanism that enforces calibration (Section 3.4).

## 3.1. Design Criteria

To rigorously stress-test the upper bounds of state-of-the-art models and bridge the gap between reported metrics and real-world reliability, the design of PERCEPTIONRUBRICS is governed by two overarching principles:

Enforcing Perceptual Persistence. To probe comprehensive perceptual capabilities, we prioritize complexity over scale. We posit that a robust benchmark must utilize images with extreme information density that ranging from crowded scenes to document-heavy layouts, therefore invalidate the linguistic “shortcuts” often taken by models. This design criterion compels models to exhibit perceptual persistence, requiring active, fine-grained exploration of long-tail visual details rather than reliance on rough global understanding or parametric priors.

![](images/6c3fa39af3f4e7e9a469bc707a24d8070027967356b0a62b2a7f66f0c712a36d.jpg)  
Figure 4. The PERCEPTIONRUBRICS Construction Pipeline. Adopting a caption-centric approach, we first synthesize golden captions via circular peer-review (Top). These captions then serve as anchors to generate Must-Right and Easy-Wrong rubrics through domain-specific prompting (Bottom).

Calibrating to Human Sensitivity. To resolve the paradox where high semantic scores mask brittle performance, we prioritize precision over approximation. We argue that an effective metric must mirror the error-sensitive nature of human judgment, where localized errors (e.g., hallucinating a single digit in a chart) represent binary failures rather than minor fluctuations. Consequently, our criterion mandates atomic verifiability and task-adaptive penalties: evaluation must be grounded in objective, fact-based checks (True/- False) and rigorously penalize hallucinations, ensuring the metric reflects practical perceptual utility rather than mere statistical similarity.

## 3.2. Image Curation

To ensure the benchmark probes the perceptual limits of flagship models, we curate an image collection that emphasizes visual diversity and complexity, targeting inputs rich in perceptually critical details that maximize error potential.

Task Domains. As illustrated in Figure 3, we structure our data across seven diverse categories to cover the full spectrum of multimodal capabilities: Natural Scenes (complex real-world environments); Document & OCR (text-dense documents, forms, and handwritten content); Digital UI & UX (web pages, mobile UIs, and dashboards); Structured Data (charts, plots, and tables); STEM & Expert (scientific diagrams, geometric figures, and medical imaging); Logic & Puzzle (visual riddles and spatial reasoning tasks); and Creative & Cultural (artworks, cultural artifacts, and design concepts).

Density-Aware Filtering. We employ the advanced MLLM, Step3-VL-10B (Huang et al., 2026), as a scorer to filter the curated images based on complexity and informativeness. Specifically, given a candidate image, the model evaluates its visual complexity (via object richness) and informativeness (via semantic density), assigning a score from 1 to 10 (see details in Section C.1). To ensure a balanced distribution across categories, we retain images that surpass domain-specific thresholds.

## 3.3. Caption-Centric Perception Rubric Construction

To instantiate the rigorous design criteria outlined above, we construct a caption-centric pipeline. Given that generating rubrics directly from raw pixels often suffers from the visual grounding gap inherent in current vision encoders (Darcet et al., 2023) and MLLMs (Kang et al., 2025), we choose an intermediary strategy: first explicitly transcribing visual information into text, then distilling rules from it. This approach prioritizes constructing a comprehensive, precise, and exhaustive golden caption to capture image details. This textual foundation enables the subsequent rubric generator to cover extreme visual granularity and detect subtle failure modes with significantly higher reliability than direct imageto-rubric methods.

## 3.3.1. GENERATING GOLDEN CAPTION

As illustrated in the top half of Figure 4, we construct golden reference captions $C _ { g o l d }$ through a two-step consensusdriven pipeline. This approach treats heterogeneous

MLLMs as a collaborative filter to minimize human annotation costs while ensuring high precision.

Step 1: Circular Peer-Review. Three distinct top-tier MLLMs (e.g., GPT-5.2, Gemini-3-Pro, and Seed-1.8) serve as a “jury-and-generator” ensemble. For each image, they first generate independent descriptions to form an initial candidate pool. To reduce hallucinations and self-preference bias, we implement a circular peer-review mechanism (Figure 4, top middle). In this phase, models iteratively compare candidates against visual evidence, rank them based on accuracy, and rewrite descriptions to synthesize a superior version. This review cycle runs for limited iterations $( N \leq 2 )$ to efficiently drive the ensemble toward a unified consensus.

Step 2: Strict Consensus Filtering. To strictly control quality and annotation costs, human experts intervene only as final verifiers rather than creators. We adopt a discardon-divergence protocol: samples where the models fail to reach a unanimous agreement are discarded. Only when the ensemble converges on a single optimal caption (i.e., high consensus) do human annotators perform a lightweight verification to finalize the golden reference $C _ { g o l d } .$ . This ensures that human effort is spent exclusively on high-confidence samples.

## 3.3.2. GENERATING PERCEPTION RUBRIC

Building upon the verified golden reference $C _ { g o l d } .$ , we employ Gemini-3-Pro (Team, 2025) as the rubric proposer to construct dual-stream evaluation criteria (Figure 4, bottom). This pipeline mirrors the error-sensitive nature of human judgment by generating rubrics from two complementary perspectives: a priori essential facts and a posteriori common pitfalls.

A Priori: Must-Right Rubrics. From a positive perspective, the rubric proposer distills a set of atomic perceptual facts from I and $C _ { \mathrm { g o l d } }$ that a candidate must correctly identify. Crucially, we employ domain-specific adaptive prompts (detailed in Section C.2) to align with varying perceptual demands: rubrics for text-centric images prioritize character precision, while those for natural scenes emphasize spatia relations and object attributes.

A Posteriori: Easy-Wrong Rubrics. From a negative perspective, we challenge model robustness by targeting likely failure modes. We first construct a response pool P by collecting predictions from a diverse set of baseline MLLMs. By analyzing the discrepancies between these actual outputs P and the reference $C _ { g o l d } .$ , the rubric proposer identifies frequent hallucinations and subtle misinterpretations. These empirically observed errors are converted into Easy-Wrong rubrics, ensuring the evaluation penalizes realistic mistakes

Table 1. Detailed statistics of the PERCEPTIONRUBRICS benchmark for images, captions and rubrics.

<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Number of images</td><td>1,038</td></tr><tr><td>Number of captions</td><td>1,038</td></tr><tr><td>Average caption length (words)</td><td>770.42</td></tr><tr><td>Total number of rubrics</td><td>12,004</td></tr><tr><td>Must-right rubrics</td><td>4,232</td></tr><tr><td>Easy-wrong rubrics</td><td>7,772</td></tr><tr><td>Average rubrics per image</td><td>11.56</td></tr><tr><td>Must-right per image</td><td>4.08</td></tr><tr><td>Easy-wrong per image</td><td>7.49</td></tr></table>

rather than hypothetical ones.

## 3.4. Evaluation Metric

We employ an LLM-as-a-Judge framework to perform finegrained evaluation, aiming to balance effectiveness and efficiency. We select GPT-OSS-120B (OpenAI, 2025b) as the judge due to its proven capability for highly calibrated assessments (Huang et al., 2026). Specifically, a model prediction $P ,$ and a set of rubrics $\mathcal { R } = \mathcal { R } _ { m } \cup \mathcal { R } _ { e }$ covering Must-Right and Easy-Wrong cases, the judge evaluates each rubric item yielding a boolean output (True for compliance, False otherwise). To prioritize factual correctness, we implement a gated scoring logic:

Must-Right as the Gate. Let $\mathcal { R } _ { m } = \{ r _ { m , 1 } , . . . , r _ { m , j } \}$ be the set of Must-Right rubrics, which serve as a mandatory gatekeeper. If the model fails even a single criterion in $\mathcal { R } _ { m } ,$ , the description is deemed factually compromised, penalizing the final score to zero:

$$
G = \prod_ {i = 1} ^ {j} \mathbb {I} (r _ {m, i} = \text { True })\tag{1}
$$

where $G \in \{ 0 , 1 \}$ represents the gate status.

Easy-Wrong for Granular Differentiation. For models that pass the gate $( G = 1 )$ , we calculate the final score based on the Easy-Wrong rubrics $\mathcal { R } _ { e } = \{ r _ { e , 1 } , . . . , r _ { e , k } \}$ These rubrics assess whether the response correctly captures error-prone fine-grained details, including details that are commonly hallucinated, omitted, or misinterpreted. The final score S is defined as:

$$
S = G \cdot \frac {1}{k} \sum_ {i = 1} ^ {k} \mathbb {I} (r _ {e, i} = \text { True })\tag{2}
$$

This scoring philosophy ensures that a high score reflects not only the absence of basic hallucinations but also a superior discernment of subtle, density-rich visual details.

## 4. Experiments

## 4.1. Benchmark Statistics

As summarized in Table 1, the resulting benchmark contains 1,038 information-dense images, each paired with a verified golden caption and a set of instance-specific perception rubrics. In total, PERCEPTIONRUBRICS includes 12,004 atomic rubrics, consisting of 4,232 Must-Right rubrics and 7,772 Easy-Wrong rubrics, with an average of 11.56 rubrics per image. Beyond rubric density, our benchmark is also characterized by highly detailed textual references. As shown in Figure 5, the golden caption lengths exhibit a right-skewed distribution: most captions concentrate around 400–500 words, while a long tail extends to captions exceeding 3,400 words. The mean caption length reaches 770.42 words, higher than the median of 569 words. This long-tailed caption distribution reflects the high information density of our images and provides a rich textual anchor for constructing fine-grained and verifiable rubrics.

## 4.2. Experimental Setup

We evaluate a diverse suite of 25 models, spanning proprietary frontier models (e.g., Gemini-3-Pro (Team, 2025), Gemini-3.5-Flash (Gemini Team, Google Deep-Mind, 2026), GPT-5.4 (OpenAI, 2026a), GPT-4o (OpenAI, 2024), Seed-2.0 (ByteDance-Seed, 2026c), Seed-1.8 (ByteDance-Seed, 2026b), Seed-1.6 (ByteDance-Seed, 2026a), GLM-5V-Turbo (Hong et al., 2026), Qwen3.5- Plus (Team, 2026a)) and leading open-weights models (e.g.,Qwen3.5-397B (Team, 2026a), Qwen3-VL (Bai et al., 2025a), Qwen2.5-VL (Bai et al., 2025b),Step3-VL-10B (Huang et al., 2026), Step-3.7-Flash (StepFun Team, 2026), MiniMax-M3 (Lai et al., 2026), MiMo-V2.5 (Team, 2026b),Kimi-K2.5 (moonshot, 2026)).

## 4.3. Main Results

Compliance Scores. Table 2 summarizes the performance of all evaluated models, which reveals a pronounced performance stratification that is largely obscured by traditional holistic benchmarks. Seed-2.0-Lite leads the leaderboard with an overall score of 70.07%, outperforming the runner-up (Gemini-3.5-Flash) by 0.19%. In contrast, despite being a widely used proprietary model, GPT-4o-2024- 05-13 exhibits the weakest perceptual performance among its category, achieving an overall accuracy of only 12.59%. Across models, performance is consistently higher on natural image domains (e.g., reaching 79.20% for Seed-2.0- Lite), aligning with human perceptual intuition and reflecting the relative maturity of models in handling real-world visual scenes. Conversely, almost all models struggle most in the GUI domain (e.g., Qwen2.5-VL-7B drops to 5.13%), indicating that robust visual grounding for future agents remains an unresolved challenge. Moreover, unlike in reasoning tasks where open-sourced models often rival proprietary flagships (Huang et al., 2026; Bai et al., 2025a), our results show a distinctive performance gap. The best-performing open-source model (Qwen3.5, 61.61%) still trails the proprietary state-of-the-art by over 8%. This suggests that open-source models still have significant ground to cover in fine-grained perception and open-world recognition, also confirming our benchmark’s sensitivity in distinguishing intrinsic model capacity beyond reasoning capabilities.

![](images/7baa886c29ed2b8257bc8cdd2afad6c38690708a4b8805c16b8c2688a97c710c.jpg)  
Figure 5. Distribution of golden caption lengths in our benchmark. The histogram shows the word count frequency across the dataset.

Domain-Specific Failure Modes. To diagnose where models fundamentally fail, we analyze cases in which predictions do not pass the Must-Right gate (i.e. G = 0), indicating a breakdown in basic perceptual capability. Figure 6 (Left) presents the distribution of such failure cases across domains for six representative models. A similar pattern emerges: GUI constitutes the dominant source of perceptual failures. In contrast, domains such as Natural and STEM are comparatively easier, exhibiting substantially fewer failures. This trend suggests that current models continue to struggle with inputs characterized by high information density and strict spatial constraints.

Atomic vs. Holistic Perception. To evaluate perceptual reliability at different granularities, we compare performance metrics derived from individual rubrics versus the aggregate gate status. Specifically, we define Atomic Accuracy as the mean accuracy of all individual rubrics (r<sub>i</sub>), representing local precision. In contrast, the Must-Right Pass Rate is calculated as the average value of the binary gate status G across the dataset (i.e., the expectation <sup>E</sup>[G]), representing the probability of a record successfully passing the mandatory gatekeeper. As shown in Figure 6 (Right), models consistently achieve high Atomic Accuracy, indicating that most individual $r _ { i }$ predictions are correct. However, the Must-Right Pass Rate (average G) is substantially lower, revealing a systematic failure to satisfy the strict conjunction of all constraints. We term this discrepancy the Reliability Gap. Notably, this gap narrows as model capability increases, suggesting that stronger models are better able to maintain consistent perception abilities required to keep the gate G open.

Table 2. Fine-grained performance breakdown across 7 domains on PERCEPTIONRUBRICS. Models are categorized into Open-Source and Proprietary groups and sorted by Overall Score in ascending order. All values are reported in percentage (%).

<table><tr><td>Model</td><td>Params</td><td>Doc</td><td>Logic</td><td>Creative</td><td>GUI</td><td>Natural</td><td>STEM</td><td>Structured</td><td>Overall</td></tr><tr><td colspan="10">Open-Source Models</td></tr><tr><td>Qwen2.5-VL-7B</td><td>7B</td><td>6.53</td><td>3.06</td><td>6.71</td><td>5.13</td><td>20.70</td><td>14.74</td><td>6.14</td><td>8.37</td></tr><tr><td>Qwen2.5-VL-32B</td><td>32B</td><td>14.22</td><td>8.89</td><td>13.32</td><td>14.39</td><td>36.60</td><td>20.13</td><td>19.07</td><td>17.79</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>8B</td><td>39.24</td><td>23.20</td><td>31.18</td><td>29.04</td><td>55.83</td><td>36.35</td><td>28.44</td><td>34.13</td></tr><tr><td>Step3-VL-10B</td><td>10B</td><td>32.07</td><td>33.90</td><td>34.23</td><td>23.25</td><td>54.55</td><td>48.94</td><td>38.70</td><td>35.97</td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking</td><td>235B</td><td>43.08</td><td>35.84</td><td>39.24</td><td>33.28</td><td>56.73</td><td>53.42</td><td>41.04</td><td>41.88</td></tr><tr><td>MiniMax-M3</td><td>428B</td><td>34.42</td><td>35.17</td><td>37.82</td><td>40.58</td><td>54.85</td><td>58.39</td><td>55.40</td><td>44.82</td></tr><tr><td>MiMo-V2.5</td><td>310B</td><td>44.06</td><td>37.46</td><td>42.82</td><td>35.25</td><td>55.35</td><td>61.04</td><td>53.23</td><td>45.65</td></tr><tr><td>Step-3.7-Flash</td><td>196B</td><td>45.81</td><td>36.45</td><td>48.20</td><td>42.60</td><td>62.72</td><td>65.40</td><td>46.40</td><td>48.62</td></tr><tr><td>Kimi-K2.5</td><td>1T</td><td>46.85</td><td>49.27</td><td>48.84</td><td>46.37</td><td>60.07</td><td>59.57</td><td>50.49</td><td>50.78</td></tr><tr><td>Kimi-K2.6</td><td>1T</td><td>46.21</td><td>50.05</td><td>49.31</td><td>47.81</td><td>58.87</td><td>60.15</td><td>54.67</td><td>51.77</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>397B</td><td>60.17</td><td>58.29</td><td>56.85</td><td>54.76</td><td>68.51</td><td>77.59</td><td>64.00</td><td>61.61</td></tr><tr><td colspan="10">Proprietary Models</td></tr><tr><td>GPT-4o-2024-05-13</td><td>-</td><td>10.32</td><td>10.35</td><td>10.00</td><td>7.01</td><td>23.89</td><td>12.14</td><td>17.33</td><td>12.59</td></tr><tr><td>Seed-1.6</td><td>-</td><td>49.38</td><td>27.52</td><td>40.23</td><td>43.94</td><td>57.47</td><td>48.40</td><td>43.00</td><td>44.54</td></tr><tr><td>GLM-5V-Turbo</td><td>-</td><td>50.37</td><td>46.70</td><td>46.08</td><td>39.84</td><td>61.48</td><td>54.23</td><td>47.12</td><td>48.18</td></tr><tr><td>Seed-1.8</td><td>-</td><td>58.77</td><td>41.33</td><td>53.31</td><td>50.26</td><td>70.62</td><td>55.44</td><td>51.73</td><td>54.34</td></tr><tr><td>GPT-5.5</td><td>-</td><td>43.14</td><td>52.03</td><td>43.82</td><td>59.53</td><td>57.75</td><td>61.78</td><td>64.54</td><td>55.23</td></tr><tr><td>Gemini-3-Flash</td><td>-</td><td>54.68</td><td>58.59</td><td>57.49</td><td>51.55</td><td>71.17</td><td>77.55</td><td>59.49</td><td>59.83</td></tr><tr><td>GPT-5.4</td><td>-</td><td>55.19</td><td>59.25</td><td>47.66</td><td>62.61</td><td>60.43</td><td>68.46</td><td>70.61</td><td>60.81</td></tr><tr><td>Seed-2.0-Pro</td><td>-</td><td>64.95</td><td>62.41</td><td>65.59</td><td>48.22</td><td>75.74</td><td>71.67</td><td>56.40</td><td>61.44</td></tr><tr><td>Qwen3.5-Plus</td><td>-</td><td>56.87</td><td>57.82</td><td>55.95</td><td>53.94</td><td>69.47</td><td>72.81</td><td>70.85</td><td>61.61</td></tr><tr><td>Qwen3.6-Plus</td><td>-</td><td>60.78</td><td>59.77</td><td>56.82</td><td>52.69</td><td>70.16</td><td>77.25</td><td>68.74</td><td>62.30</td></tr><tr><td>Gemini-3-Pro</td><td>-</td><td>68.35</td><td>63.68</td><td>71.51</td><td>57.57</td><td>76.65</td><td>77.83</td><td>74.50</td><td>68.79</td></tr><tr><td>Gemini-3.1-Pro</td><td>-</td><td>67.86</td><td>63.66</td><td>70.00</td><td>59.37</td><td>74.85</td><td>80.37</td><td>74.80</td><td>69.02</td></tr><tr><td>Gemini-3.5-Flash</td><td>-</td><td>71.64</td><td>64.32</td><td>72.03</td><td>54.10</td><td>78.89</td><td>86.05</td><td>76.23</td><td>69.88</td></tr><tr><td>Seed-2.0-Lite</td><td>-</td><td>73.56</td><td>61.48</td><td>72.62</td><td>59.07</td><td>79.20</td><td>80.85</td><td>72.59</td><td>70.07</td></tr></table>

Consistency of Perceptual Capabilities. We further examine the correlation between models’ basic perceptual reliability and their hallucination resistance to fine-grained details. As shown in Figure 7, there is a near-perfect linear correlation $( R ^ { 2 } \approx 0 . 9 8 )$ between Must-Right Pass Rate and Easy-Wrong accuracy. This implies that models failing to ground essential visual facts (low X-axis) inevitably struggle with subtle details and hallucination (low Y-axis). Therefore, robust fine-grained understanding critically depends on foundational perception, in particular, the coherent recognition of multiple salient elements.

## 5. Analysis

Beyond model performance, we conduct a systematic metaevaluation to assess the rigor and reliability of the benchmark itself from multiple perspectives.

## 5.1. Alignment with Human Preference

To validate whether PERCEPTIONRUBRICS reflects humanperceived model quality, we compare its model ranking against the Vision Arena (Chou et al., 2024) leaderboard, which aggregates large-scale human preferences over MLLM responses into Elo ratings. In Figure 9, we focus on the five models: GPT-5.4, Qwen3-VL-235B, GPT-4o, Kimi-K2.6, and MiMo-V2.5. For each benchmark, we plot the evaluation score of these models against the Vision Arena score.

PERCEPTIONRUBRICS exhibits the strongest alignment with human preference among the compared benchmarks, achieving a Pearson correlation of 0.916 and a Spearman rank correlation of 1.000. In contrast, existing captioning benchmarks such as DOCCI (Onoe et al., 2024) and DetailCaps (Dong et al., 2024) show substantially weaker agreement with human-preference scores. DOCCI, in particular, assigns nearly indistinguishable scores to models with markedly different human-preference ratings, indicating limited discriminative power. These results suggest that PERCEPTIONRUBRICS provides a more human-aligned and discriminative signal for fine-grained perception evaluation.

![](images/8d2d94339713450372db2780834afdcea3bddfe78fd9a808a60cb4a86db7ab23.jpg)

![](images/9ee0c70de2da5dbf7234436913fd0d521d6ad25d2380b55b78a8feb123ed96fd.jpg)

<sub>Digital</sub> <sub>UI</sub> <sub>&</sub> <sub>UX</sub>Digital UI & UXigital UI & UXFigure 6. Comprehensive Failure Analysis. (Left) Distribution of error sources across different models. (Right) Reliability Gap Analysis comparing Atomic Accuracy (the average pass rate over individual rubrics) with the stricter Must-Right-All-Pass Rate, highlighting the<sup>2026/6/22</sup> <sup>下午7:51</sup> difficulty of maintaining consistency across all constraints.  
![](images/d32e42b8e4c07a74f55ccc6312dcd80385c79fffe7834111b37aaef68588d5dc.jpg)  
Figure 7. Correlation Analysis between basic perceptual reliability (Must-Right) and fine-grained understanding (Easy-Wrong) across six representative models.

## 5.2. Resistance to Length Bias.

We analyze the correlation between predicted caption length and performance on PERCEPTIONRUBRICS to assess potential length bias. As shown in Figure 10 (a-b), Gemini-3.1-Pro shows no statistically significant correlation (r = $- 0 . 0 7 9 , p = 0 . 0 7 5 8 )$ , while Kimi-K2.6 exhibits a weak positive correlation $( r = 0 . 1 7 2 , p = 1 . 0 9 \times 1 0 ^ { - 4 } )$ . This result indicates that PERCEPTIONRUBRICS effectively decouples verbosity from evaluation outcomes, rewarding precise and verifiable perception rather than longer descriptions.

## 5.3. Evaluation Robustness

In Figure 10 (c), we selected three representative models spanning different capability levels: Seed-2.0-Lite, Step3-

![](images/960f98c9e3b741015cdc168043a5c454f757bd3c19b419f5f5161327abb3e3dd.jpg)  
Figure 8. Rubric Coverage vs. Evaluation Stability. As the sampled rubric ratio increases from 20% to 80%, the standard deviation of model scores decreases.

VL-10B, and Kimi-K2.6. Then we performed repeated evaluations using two distinct judges with the same inputs: GPT-OSS-120B (OpenAI, 2025b) and GPT-5.5 (OpenAI, 2026b). Despite GPT-OSS-120B exhibiting a slightly stricter scoring distribution (systematically lower by ∼6.0%), both judges yielded an identical ranking order. The black error bars represent the standard deviation across these independent runs. The results demonstrate high stability, with standard deviations remaining consistently low across all configurations. Overall, these results demonstrate the robustness of both our rubric generation pipeline and the resulting evaluation metrics to judge choice and sampling variability.

## 5.4. Rubric Coverage vs. Evaluation Stability

As shown in Figure 8, we analyze the effect of rubric quantity on evaluation stability. Using 25 models, we subsample

![](images/5c6f17a221f36aa7edd605f80b71191e847e7f6849e145d7bb364ae7fd87c6be.jpg)

![](images/f70287b3244a567ac6c07b3a3d769e63dfa8f7b5c9a9d577b5d9636c6a2f1968.jpg)

![](images/a93fd00209c482c3158cdce3dcca0960c401db29c54a223b04590d05d89b0472.jpg)

Figure 9. Alignment with Human Preference. We compare benchmark scores from DOCCI (Onoe et al., 2024), DetailCaps (Dong et al., 2024), and PERCEPTIONRUBRICS against human preference scores from Vision Arena for the five overlapping models. Each point denotes one model. PERCEPTIONRUBRICS shows the strongest correlation with Vision Arena, achieving Pearson 0.916 and Spearman 1.000.  
![](images/d25691bf6b13e413b79957872d2fd8732eec21a46591e915d4c6d5b7fefd06e2.jpg)  
(a)

![](images/5f6211e9759829a4bb42d409f58df194f3562b284adb6e734d21f6fb4f8433be.jpg)  
(b)

![](images/b91040ce9fd8aef72d54028ff03ed0f5edbf92721af8e873960a2f2b021bcd30.jpg)  
(c)  
Figure 10. (a-b) Length Bias. The two figures examine the correlation between response length (word count) and benchmark scores. (c) Evaluation Robustness. Results obtained with different judges exhibit consistent and stable performance trends.

20%, 40%, 60%, and 80% of rubrics from both the Must-Right and Easy-Wrong sets. For each sampling ratio, we perform three independent runs and compute the standard deviation of model scores to measure stability. The figure visualizes the distribution of these standard deviations across models at each ratio using violin plots, with embedded boxes indicating the interquartile range and medians; the dashed line denotes the mean stability trend. Evaluation stability improves monotonically as rubric coverage increases, with standard deviation consistently decreasing, highlighting sufficient rubric coverage as a prerequisite for stable and reproducible perception assessment.

mechanism, our framework exposes perceptual failures that are often hidden by existing metrics. Experiments across 25 MLLMs reveal a clear reliability gap between individual fact recognition and consistent conjunctive perception, persistent weaknesses in information-dense domains such as GUIs, and strong alignment between our scores and human preferences. These findings suggest that reliable multimodal evaluation should move beyond coarse similarity and explicitly audit critical visual facts. We hope PERCEPTION-RUBRICS provides a sharper diagnostic tool for measuring perceptual reliability and guiding the development of more trustworthy MLLMs.

## 6. Conclusion

We present PERCEPTIONRUBRICS, a rubric-based benchmark that calibrates multimodal evaluation to human perceptual judgment. By decomposing dense image understanding into atomic, verifiable rubrics and enforcing a gated scoring

## Impact Statement

This work aims to advance machine learning by improving the reliability of multimodal evaluation. While this may affect downstream MLLM development, we do not identify specific societal consequences requiring special discussion.

## References

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report, 2025a. URL https://arxiv.org/abs/2511.21631.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-vl technical report, 2025b. URL https://arxiv.org/abs/2502.13923.

ByteDance-Seed. Seed1.6, 2026a. URL https://seed. bytedance.com/en/seed1\_6/.

ByteDance-Seed. Seed1.8, 2026b. URL https://seed. bytedance.com/en/seed1\_8/.

ByteDance-Seed. Seed2.0, 2026c. URL https://seed. bytedance.com/en/seed2.

Chen, X., Li, G., Wang, Z., Jin, B., Qian, C., Wang, Y., Wang, H., Zhang, Y., Zhang, D., Zhang, T., et al. Rm-r1: Reward modeling as reasoning. arXiv preprint arXiv:2505.02387, 2025.

Cheng, K., Song, W., Fan, J., Ma, Z., Sun, Q., Xu, F., Yan, C., Chen, N., Zhang, J., and Chen, J. Caparena: Benchmarking and analyzing detailed image captioning in the llm era, 2025a. URL https://arxiv.org/ abs/2503.12329.

Cheng, X., Zhang, W., Zhang, S., Yang, J., Guan, X., Wu, X., Li, X., Zhang, G., Liu, J., Mai, Y., Zeng, Y., Wen, Z., Jin, K., Wang, B., Zhou, W., Lu, Y., Li, T., Huang, W., and Li, Z. Simplevqa: Multimodal factuality evaluation for multimodal large language models, 2025b. URL https: //arxiv.org/abs/2502.13059.

Chou, C., Dunlap, L., Mashita, K., Mandal, K., Darrell, T., Stoica, I., Gonzalez, J. E., and Chiang, W.-L. Visionarena: 230k real world user-vlm conversations with preference labels. 2024. URL https://arxiv.org/ abs/2412.08687.

Darcet, T., Oquab, M., Mairal, J., and Bojanowski, P. Vision transformers need registers. arXiv preprint arXiv:2309.16588, 2023.

Dong, H., Li, J., Wu, B., Wang, J., Zhang, Y., and Guo, H. Benchmarking and improving detail image caption, 2024. URL https://arxiv.org/abs/2405.19092.

Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., Wu, Y., and Ji, R. Mme: A comprehensive evaluation benchmark for multimodal large language models, 2024. URL https: //arxiv.org/abs/2306.13394.

Gemini Team, Google DeepMind. Gemini 3.5: Frontier intelligence with action, 2026. URL https: //blog.google/innovation-and-ai/ models-and-research/gemini-models/ gemini-3-5/.

Gunjal, A., Wang, A., Lau, E., Nath, V., He, Y., Liu, B., and Hendryx, S. Rubrics as rewards: Reinforcement learning beyond verifiable domains. arXiv preprint arXiv:2507.17746, 2025.

Hong, W., Gu, X., Pan, Z., Yang, Z., Wang, Y., Wang, Y., Yue, Y., Wang, Y., Wang, Y., Wang, Y., Liu, X., Yu, W., Wang, W., Li, W., Duan, S., Yang, S., Lv, R., Liu, M., Pan, L., Ning, K., Ji, J., Wang, J., Chen, J., Xu, J., Zhu, J., Cheng, J., Qi, J., Gan, G., Wang, G., Yao, C., et al. GLM-5V-Turbo: Toward a native foundation model for multimodal agents. arXiv preprint arXiv:2604.26752, 2026.

Huang, A., Yao, C., Han, C., Wan, F., Guo, H., Lv, H., Zhou, H., Wang, J., Zhou, J., Sun, J., et al. Step3-vl-10b technical report. arXiv preprint arXiv:2601.09668, 2026.

Kang, S., Kim, J., Kim, J., and Hwang, S. J. See what you are told: Visual attention sink in large multimodal models. arXiv preprint arXiv:2503.03321, 2025.

Lai, X., Xu, W., Yang, Y., Chen, Q., Xu, Y., Zeng, L., Li, X., Sun, H., Zhu, H., Zhang, V., and Zhao, P. Mini-Max Sparse Attention. arXiv preprint arXiv:2606.13392, 2026.

Liu, C. Y., Zeng, L., Liu, J., Yan, R., He, J., Wang, C., Yan, S., Liu, Y., and Zhou, Y. Skywork-reward: Bag of tricks for reward modeling in llms. arXiv preprint arXiv:2410.18451, 2024a.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024b.

Liu, Y., Li, Z., Huang, M., Yang, B., Yu, W., Li, C., Yin, X.-C., Liu, C.-L., Jin, L., and Bai, X. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12),

December 2024c. ISSN 1869-1919. doi: 10.1007/ s11432-024-4235-6. URL http://dx.doi.org/ 10.1007/s11432-024-4235-6.

Liu, Z., Wang, P., Xu, R., Ma, S., Ruan, C., Li, P., Liu, Y., and Wu, Y. Inference-time scaling for generalist reward modeling. arXiv preprint arXiv:2504.02495, 2025.

moonshot. kimi-k2-5, 2026. URL https://www.kimi. com/blog/kimi-k2-5.html/.

Onoe, Y., Rane, S., Berger, Z., Bitton, Y., Cho, J., Garg, R., Ku, A., Parekh, Z., Pont-Tuset, J., Tanzer, G., et al. Docci: Descriptions of connected and contrasting images. In European Conference on Computer Vision, pp. 291– 309. Springer, 2024.

OpenAI. Hello gpt-4o, 2024. URL https://openai. com/index/hello-gpt-4o/.

OpenAI. Introducing gpt-5.2, 2025a. URL https://openai.com/index/ introducing-gpt-5-2/.

OpenAI. Gpt-oss-120b and gpt-oss-20b model card. arXiv preprint arXiv:2508.10925, 2025b. URL https:// arxiv.org/abs/2508.10925.

OpenAI. Introducing GPT-5.4, 2026a. URL https://openai.com/index/ introducing-gpt-5-4/.

OpenAI. GPT-5.5 System Card. https://openai. com/index/gpt-5-5-system-card/, 2026b.

Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311–318, 2002.

Poznanski, J., Soldaini, L., and Lo, K. olmocr 2: Unit test rewards for document ocr. arXiv preprint arXiv:2510.19817, 2025.

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020.

Rezaei, M., Vacareanu, R., Wang, Z., Wang, C., Liu, B., He, Y., and Akyurek, A. F. Online rubrics elicitation from¨ pairwise comparisons. arXiv preprint arXiv:2510.07284, 2025.

Rohrbach, A., Hendricks, L. A., Burns, K., Darrell, T., and Saenko, K. Object hallucination in image captioning. arXiv preprint arXiv:1809.02156, 2018.

Sharma, M., Zhang, C. B. C., Bandi, C., Wang, C., Aich, A., Nghiem, H., Rabbani, T., Htet, Y., Jang, B., Basu, S., et al. Researchrubrics: A benchmark of prompts and rubrics for evaluating deep research agents. arXiv preprint arXiv:2511.07685, 2025.

StepFun Team. Step 3.7 flash: A high-efficiency flash model for real-world agents, 2026. URL https://static. stepfun.com/blog/step-3.7-flash/.

Team, G. Gemini 3 pro: the frontier of vision ai, 2025. URL https://blog.google/ innovation-and-ai/technology/ developers-tools/gemini-3-pro-vision/ /.

Team, Q. Qwen3.5: Accelerating productivity with native multimodal agents, February 2026a. URL https:// qwen.ai/blog?id=qwen3.5.

Team, X. M. Mimo-v2.5, 2026b. URL https://huggingface.co/collections/ XiaomiMiMo/mimo-v25. Hugging Face model collection.

Wei, Y., Zhao, L., Lin, K., Yu, E., Peng, Y., Dong, R., Sun, J., Wei, H., Ge, Z., Zhang, X., et al. Perception in reflection. arXiv preprint arXiv:2504.07165, 2025.

Yang, J., Yang, S., Gupta, A. W., Han, R., Fei-Fei, L., and Xie, S. Thinking in space: How multimodal large language models see, remember, and recall spaces. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 10632–10643, 2025.

Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., and Wang, L. Mm-vet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490, 2023.

Zhang, H., Li, C., and Fernando, B. Mitigating easy option bias in multiple-choice question answering. arXiv preprint arXiv:2508.13428, 2025.

Zhou, Y., Cui, C., Yoon, J., Zhang, L., Deng, Z., Finn, C., Bansal, M., and Yao, H. Analyzing and mitigating object hallucination in large vision-language models. arXiv preprint arXiv:2310.00754, 2023.

## A. Dataset Statistics

In this section, we provide detailed statistics and comparisons for the PERCEPTIONRUBRICS benchmark.

## A.1. Comparison with Other Benchmarks

Compared to existing benchmarks, PerceptionRubrics distinguishes itself in three critical dimensions: annotation granularity, data source diversity, and domain coverage, as shown in Table 3.

• Dense and Comprehensive Captions: Unlike DetailCaps-4870 (Dong et al., 2024) and DOCCI (Onoe et al., 2024), which typically provide brief descriptions (averaging 122.1 and 135.9 words, respectively), PerceptionRubrics focuses on dense captioning. With an average of 770.42 words per image, our benchmark captures fine-grained visual details, spatial relationships, and implicit reasoning, offering a significantly more challenging testbed for evaluating the upper bounds of MLLMs.

• Broad Domain Coverage: Unlike existing benchmarks that are predominantly restricted to natural scenes, Perception-Rubrics spans seven distinct domains to provide a more comprehensive evaluation. These range from everyday natural scenes to specialized areas such as GUIs, OCR-heavy documents, and STEM-related diagrams. This diversity is crucial for assessing the general-purpose capabilities of agents in complex, real-world applications that go far beyond simple object recognition.

• Diverse and High-Quality Sources: Instead of relying solely on web-crawled data or specific author donations, our dataset aggregates high-quality samples from existing visual benchmarks. Furthermore, we employ a hybrid annotation pipeline combining advanced reasoning models (e.g., GPT-5.2-Thinking) with human expert verification, ensuring both the scalability and reliability of the ground truth.

Table 3. Comparison of our proposed benchmark with existing datasets. By transposing the table, detailed descriptions are easier to read.

<table><tr><td>Benchmark</td><td>DetailCaps-4870</td><td>DOCCI</td><td>PerceptionRubrics</td></tr><tr><td>Specific Sources</td><td>COCO, SAM, LAION, CC, SBU, Coyo, Flickr</td><td>Author Donation</td><td>Open-source Visual Benchmarks</td></tr><tr><td>Image Domains Annotator</td><td>Natural sceneGPT-4V, GPT-4o, Gemini-1.5-Pro</td><td>Natural sceneHuman</td><td>Multi-domain (GUI, OCR, STEM...)GPT-5.2-Thinking, Seed-1.8, Gemini-3-Pro, Human Experts</td></tr><tr><td>Images</td><td>4,870</td><td>14,847</td><td>1,038</td></tr><tr><td>Avg. Words</td><td>122.1</td><td>135.9</td><td>770.42</td></tr></table>

## A.2. Distributions

## A.2.1. CAPTION LENGTH DISTRIBUTION

As illustrated in Figure 5, we analyze the word count distribution of the golden captions. The distribution follows a typical long-tail pattern: while the majority of captions are concentrated between 300 and 700 words (with a median of 569), a significant portion extends beyond 1,000 words, reaching up to 3,461 words. This diversity in length ensures that our benchmark covers both concise summaries and highly detailed descriptions, providing a robust basis for evaluating model performance across different levels of information density.

## A.2.2. RUBRIC DISTRIBUTION

To ensure a granular and balanced evaluation, we analyze the distribution of rubrics across the dataset in Figure 11. (a) The total number of rubrics per sample primarily ranges from 10 to 20, with a clear peak at approximately 12, indicating a consistently high level of evaluation detail across the benchmark. (b) When broken down by category, Must-Right rubrics exhibit a sharp distribution centered around 4 items, representing the core facts that a model must capture. In contrast, Easy-Wrong rubrics show a broader distribution peaking around 8 items. This design places a heavier emphasis on penalizing common hallucinations and subtle errors, thereby increasing the discriminative power of the benchmark for high-performing models.

## Rubric Count Distribution

![](images/a3834717b38f26250311c072384d7afa9f6f7dab9c1b0119d75337830083ff8b.jpg)

![](images/0242c61eb942fabe7f2b7d49589135fd6d0eb37288304f02e9ada7b266810430.jpg)  
Figure 11. Distribution analysis of rubrics. (a) Frequency distribution of the total rubrics count across the dataset. (b) Probability density comparison of rubrics count between Must-Right and Easy-Wrong categories.

## B. Model Roles and Pipeline Details

To construct and evaluate PerceptionRubrics, we utilized a diverse set of models, assigning specific roles based on their capabilities. The detailed assignments are listed below:

• Complexity Judger: STEP-3-VL-10B. Responsible for filtering images based on visual complexity and informativeness.

• Rubric Generator: Gemini-3-Pro. Generated the initial set of perception rubrics from the images.

• Panel of Judges: Gemini-3-Pro, GPT-5.2, Seed-1.8. Acted as a consensus panel to validate the quality of generated captions.

• Final Judger: GPT-OSS-120B. Used for final scoring during the evaluation phase.

## C. Prompts

We provide the full system prompts used in our pipeline to ensure reproducibility.

## C.1. Complexity Filtering Prompt

The following prompt is used by the Complexity Judger to select high-quality images.

```txt
Image Filtering Prompt

You are an extremely strict computer vision data expert. Please analyze the provided image and perform a rigorous evaluation based on the two dimensions of "Visual Complexity" and "Informativeness".

**Core Principles:** 1. **Do NOT** give a high score simply because the image contains text. You must evaluate the **density** and **semantic depth** of the text.
```

```txt
2. **Severely penalize** low-quality images: images that are blurry, noisy, contain scribbled handwriting, or have excessive empty backgrounds should receive low scores.
3. If the majority of the image is white space or a single background, the score must be determined by the richness of the subject content, not by the image dimensions.

Please score based on the following strict standards (1-10 points):

1. Visual Complexity:
  - Definition: The quantity of independent visual elements (objects, lines, textures), spatial occupancy, and clarity of details within the image.
  - **1-3 Points (Low)**: Minimalist composition, massive white space, simple handwriting, single isolated objects, blurry snapshots, low-resolution screenshots.
  - **4-7 Points (Medium)**: Clear composition, good foreground-background separation, natural scenes with some texture detail, standard object close-ups
  - **8-10 Points (High)**: Extremely high density of details (e.g., crowds, dense forests, complex mechanical structures), frame filled, no large areas of solid color, high-frequency textures.

2. Informativeness:
  - Definition: The amount of information when the image is translated into a text description, the richness of context, and its knowledge value.
  - **1-3 Points (Low)**: Simple mathematical formulas, single words/numbers, scribbles without context, illegible content, generic decorative patterns, extremely low information entropy.
  - **4-7 Points (Medium)**: Complete sentences, clear recognition of single objects (e.g., "a red apple"), scenes with distinct actions, standard street views or portraits.
  - **8-10 Points (High)**: Dense documents (e.g., full-page newspapers, academic papers), complex infographics (containing multiple data sets), historical photos rich in narrative detail, scenes requiring long-form text to describe clearly.

**Output Format Requirements:** Please strictly output in XML format. Do not use Markdown code blocks (do not use ``'xml). Output the XML string directly.

XML Template:

<image_evaluation>
    <visual_complexity>
    <reasoning>Briefly describe the density of visual elements. If the image is blurry or mostly empty, explain here and provide a reason for the low score.</reasoning>
    <score>Integer between 1 and 10</score>
    </visual_complexity>
    <informativeness>
    <reasoning>Briefly describe the richness of semantic content. If it is a simple formula or phrase, explicitly state that the information content is limited.</reasoning>
    <score>Integer between 1 and 10</score>
    </informativeness>
</image_evaluation>
```

## C.2. Rubric Generation Prompt

The prompts used for generating rubrics are as follows:

```txt
Rubric Generation Prompt for Nature Scene

You are an expert evaluator for Multimodal Large Language Models (MLLMs), specializing in creating "Gating Rubrics" for natural imagery.

Your goal is to extract a concise set of **Critical Perception Checkpoints** from the provided Image and Ground Truth (GT) caption. These rubrics define the minimum acceptable standard for a model's response.

### CRITICAL EVALUATION PROTOCOL
This is a **Zero-Tolerance Gating Task**. If a candidate model fails **ANY** of the checkpoints you generate, it receives a score of 0.
Therefore, your rubrics must strictly adhere to the following principles:
1. **Undeniable Visibility:** Only select elements that are clearly visible and prominent in the image.
2. **Essentiality:** Only select elements that are critical to the image's core meaning. Ignore background clutter or minor details.
3. **Verifiability:** Each rubric must be a binary (Pass/Fail) check.

### WORKFLOW INSTRUCTIONS

**Step 1: Rubric Generation Strategy (Semantic Generalization)**
Apply the following abstract rules to ensure the rubrics are robust to varying levels of descriptive detail:

* **Entity Abstraction:** Identify the fundamental semantic category of the dominant object, strictly discarding specific instance names, brands, or fine-grained biological sub-species. (e.g., use "car" instead of "Tesla Model 3"; use "dog" instead of "Golden Retriever").
* **Attribute Decoupling:** Decouple the object's existence from its descriptive attributes. Exclude color, material, or state adjectives from the rubric criteria to prevent penalizing valid but concise responses. (e.g., require "the presence of a flower" rather than "a yellow flower"; require "clothing" rather than "a silk dress").
* **Contextual Necessity:** Only include attributes if they serve as the sole differentiator between multiple objects of the same class. (e.g., "the red player" vs "the blue player").
**Step 2: Final Filtering (Grounding Check)**
Review your list. Ensure every rubric meets the "Grounding Check":
* The element must be present in **BOTH** the Image and the GT Caption.
* If the GT caption describes a hidden detail or hallucinates something not clearly visible, **discard it**.

### OUTPUT FORMAT
Return a strictly valid JSON list containing 3 to 5 strings.
Example: '[The response mentions <Generalized Object>.", "The response indicates the weather is <Condition>."]
---
### FEW-SHOT EXAMPLES

**Example 1: Natural Scene**
* **Context:** Image shows a Black Tesla Model 3 on a rainy highway. GT describes it specifically as a Tesla Model 3. User asks "Describe this image."
* **Thought Process:** Apply Entity Abstraction: "Tesla Model 3" -> "Car". Apply Attribute Decoupling: Ignore "Black". "Rainy" is global context, keep it.
* **Generated Rubrics:** [
    "The response mentions a car or vehicle.",
    "The response indicates the weather is rainy or the road is wet.",
    "The response mentions the vehicle is on a road or highway."
```

```txt
]
**Example 2: Animal Interaction**
* **Context:** Image shows a Golden Retriever catching a frisbee in a park. GT says "A purebred Golden Retriever leaps to catch a red frisbee."
* **Thought Process:** Apply Entity Abstraction: "Golden Retriever" -> "Dog". Apply Attribute Decoupling: Ignore "Red" (frisbee color). Keep the interaction "catching/leaping".
* **Generated Rubrics:** 
[ "The response mentions a dog.", "The response mentions the dog is interacting with a frisbee or disc.", "The response captures the action of jumping or catching." ]
```

## Rubric Generation Prompt for Digital UI & UX

```markdown
You are an expert evaluator for Multimodal Large Language Models (MLLMs), specializing in creating "Gating Rubrics" for Graphical User Interfaces (GUIs), including mobile screenshots, web pages, and software interfaces.

Your goal is to extract a concise set of **Critical Perception Checkpoints** from the provided Image and Ground Truth (GT) caption. These rubrics define the minimum acceptable standard for a model's response.

### CRITICAL EVALUATION PROTOCOL
This is a **Zero-Tolerance Gating Task**. If a candidate model fails **ANY** of the checkpoints you generate, it receives a score of 0.
Therefore, your rubrics must strictly adhere to the following principles:
1. **Undeniable Visibility:** Only select elements that are clearly visible and prominent.
2. **Functional Criticality:** Only select elements that are essential for operating or navigating the interface (e.g., "Submit" button, "Back" arrow). Ignore decorative banners or ads.
3. **Verifiability:** Each rubric must be a binary (Pass/Fail) check.

### WORKFLOW INSTRUCTIONS

**Step 1: Rubric Generation Strategy (Interaction & Structure)**
Apply the following abstract rules to ensure the rubrics cover the interface's functionality:
* **Functional Semantics:** Identify interactive elements by their function, not just their shape. Map icons to their standard meaning. (e.g., "The response identifies the magnifying glass as a 'Search' button/feature"; "The response identifies the 'hamburger' icon as a menu").
* **Textual Anchoring:** Enforce exact matching for critical labels, headers, and button text. (e.g., The page title "Settings", the button label "Log In").
* **State Awareness:** Check for visual cues that indicate the system status. (e.g., "The response notes that the 'Home' tab is currently selected/active"; "The response mentions the toggle is in the 'On' position"; "The response notes a notification badge/red dot").
* **Structural Hierarchy:** Identify the major navigation zones. (e.g., "The response mentions the navigation bar at the bottom"; "The response identifies the header containing the logo").
**Step 2: Final Filtering (Grounding Check)**
Review your list. Ensure every rubric meets the "Grounding Check":
* The element must be present in **BOTH** the Image and the GT Caption.
* If the GT caption describes a functional flow not visible in the static image (e.g., "Clicking this opens a modal"), **discard it**. Only evaluate what is
```

```markdown
currently visible.

### OUTPUT FORMAT
Return a strictly valid JSON list containing 3 to 5 strings.
Example: 'The response identifies the screen title as <Title>.", "The response mentions the <Button Name> button at the bottom."]
---
### FEW-SHOT EXAMPLES

**Example 1: Mobile App (Settings Page)**
* **Context:** A screenshot of a Settings page. Title "Settings". Top item is "Airplane Mode" (Toggle is OFF). Bottom is a Tab Bar with "General" selected.
* **Thought Process:** Title is critical context. "Airplane Mode" is the first functional item. The state of the toggle (OFF) is a detail, but if prominent, keep it. The selected tab defines where we are.
* **Generated Rubrics:** 
[ "The response identifies the screen title as 'Settings'.", "The response mentions the 'Airplane Mode' option.", "The response indicates that the 'General' tab is currently selected or active .", "The response mentions the presence of a navigation bar at the bottom."
]

**Example 2: E-Commerce Product Page**
* **Context:** A product page for "Nike Air Max". Price "$120". Big red button "Add to Cart". Review stars (4.5).
* **Thought Process:** Product Name is the core entity. Price is critical data (OCR) . The primary action is "Add to Cart".
* **Generated Rubrics:** 
[ "The response identifies the product name as 'Nike Air Max'.", "The response correctly mentions the price as $120.", "The response identifies the primary action button labeled 'Add to Cart'.", "The response mentions the presence of a star rating or reviews." ]
```

## General System Instruction Template

```markdown
You are an expert VLM (Vision-Language Model) evaluator and Hallucination Analyst.

### Task
Your task is to generate a set of **Rubrics (Evaluation Criteria)** for an image captioning task.
You will be provided with:
1. **Ground Truth Caption (GT):** A factual, accurate description of the image.
2. **Model Response Pool:** A collection of captions generated by various VLMs. These responses may contain hallucinations, perceptual errors, or correct details

Your goal is to identify **common or severe perceptual errors** in the 'Response Pool' by comparing them against the 'Ground Truth', and then formulate strict criteria to penalize these errors.

### Process
1. **Analyze Errors:** Scan the 'Model Response Pool' to find discrepancies against the 'Ground Truth'. Focus on:
* **Hallucinations:** Objects mentioned in responses but not present in the GT
```

```markdown
* **Attribute Errors:** Wrong colors, shapes, materials, or textures.
* **Counting/Quantification:** Incorrect numbers of objects.
* **Spatial Relations:** Wrong relative positions (e.g., left vs. right).
* **OCR/Text:** Incorrect reading of text visible in the image.
* **Action/State:** Wrong interpretation of what an agent is doing.

2. **Filter for Perception (Crucial):**
* **INCLUDE:** Visual perception issues (e.g., calling a "red helmet" a "blue helmet"; seeing "3 people" instead of "4"; reading "STOP" as "SHOP").
* **EXCLUDE:** Knowledge gaps or Entity linking issues. If the model fails to recognize a specific character (e.g., "Genshin Impact character") but correctly describes their visual appearance (e.g., "a girl with blonde hair"), do NOT create a rubric for the specific name. Focus on the visual description.

3. **Formulate Rubrics:**
* Convert the identified high-frequency or severe errors into **Binary Checklists**.
* If models frequently hallucinate an object, create a **Negative Constraint** (e.g., "The response must NOT...").
* If models get an attribute wrong, create a **Positive Constraint** (e.g., "The response must identify...").
### Rubric Style Guidelines
* **Format:** Use imperative statements. Do NOT use questions.
* **Structure:** Start with "The response must...".
* **Granularity:** Each rubric must check a single, atomic fact.
* **Tone:** Objective and strict.

### Example
**Ground Truth:** A black cat sitting on a white refrigerator. There is a magnet shaped like a banana on the door.
**Response Pool Analysis:**
- Model A: "A black dog on a fridge." (Error: Dog vs Cat)
- Model B: "A black cat on a grey fridge." (Error: Grey vs White)
- Model C: "A cat near a fridge with an apple magnet." (Error: Apple vs Banana)

**Output Rubrics:**
{
    "rubrics": [
    "The response must identify the animal as a cat.",
    "The response must state that the refrigerator is white.",
    "The response must identify the magnet shape as a banana.",
    "The response must NOT mention the presence of a dog or an apple."
    ]
}

### Output Format
Return the result strictly in valid JSON format without markdown code blocks.
{
    "rubrics": [
    "string",
    "string"
    ]
}

Here is the data for the current image:

[Ground Truth Caption]
{gt_caption}

[Model Response Pool]
1. {response_1}
```

```txt
2. {response_2}
3. {response_3}
...
8. {response_8}
Please generate the perception rubrics based on the analysis of the responses above.
```

## C.3. Panel of Judges Prompt

To ensure the objectivity and correctness of the generated rubrics, a panel of models (Gemini-3-Pro (Team, 2025), GPT-5.2 (OpenAI, 2025a), Seed-1.8 (ByteDance-Seed, 2026b)) performs a cross-verification using the following prompt.

## Caption Verification Prompt (Panel of Judges)

```txt
**Role:**
You are the "Expert Visual Truth Adjudicator". Your task is to perform a rigorous comparative analysis of multiple AI-generated image descriptions against a provided image to identify the most faithful representation.

**Evaluation Dimensions:** 
1. **Factuality:** Are there hallucinations? (e.g., objects, colors, or text that don't exist).
2. **Spatial Precision:** Are positional relationships (left, right, above, behind) accurate?
3. **Attribute Accuracy:** Are textures, materials, lighting, and colors correctly identified?
4. **Detail Density:** Does the caption capture nuanced elements without being redundant?

**Task Workflow:** 
1. **Independent Verification:** Analyze the image first, then audit each Candidate (1, 2, and 3) individually.
2. **Conflict Resolution:** Identify discrepancies between candidates (e.g., Candidate 1 says 'vintage', Candidate 2 says 'modern'). Inspect the image to resolve these.
3. **Ranking:** Select the "Best" baseline based on the highest fidelity to the visual evidence.

**Input Candidates:** 
[Candidate 1]: {candidate_1_text}
[Candidate 2]: {candidate_2_text}
[Candidate 3]: {candidate_3_text}

**Strict Output Format:** 
You must output your response in valid XML format only. No preamble, no markdown formatting outside the XML, and no conversational filler.

**XML Output Schema:** 
<voting_result>
    <analysis>
    <candidate_1_critique>Briefly note strengths/hallucinations for C1.</candidate_1_critique>
    <candidate_2_critique>Briefly note strengths/hallucinations for C2.</candidate_2_critique>
    <candidate_3_critique>Briefly note strengths/hallucinations for C3.</candidate_3_critique>
    </analysis>
    <best_candidate_id>Candidate ID (1, 2, or 3)</best_candidate_id>
```

```erb
<rationale>A concise explanation of why this candidate won, specifically citing why it outperformed the others in terms of accuracy or detail.</rationale></voting_result>
```

## C.4. Evaluation Prompt

We utilize GPT-OSS-120B (OpenAI, 2025b) to evaluate models’ generated captions using the following prompts.

## Prompt for model evaluation

```markdown
You are an expert Rubric Evaluator for Vision-Language Models.

### Task
Your task is to verify whether a model's generated **Caption** satisfies a specific set of **Rubrics** (Evaluation Criteria).

You will receive three inputs:
1. **Model Caption:** The text description generated by the model.
2. **Group A (Critical Rubrics):** A list of fundamental perception criteria. These are "bottom-line" facts.
3. **Group B (Granular Rubrics):** A list of fine-grained or high-frequency error checks.

### Judgment Logic
For each rubric in both groups, determine if the **Model Caption** complies with the requirement.
* **True (Pass):** The caption explicitly meets the criteria or implies it without ambiguity.
* **False (Fail):** The caption contradicts the criteria, fails to mention a required element, or triggers a negative constraint.

**Handling Different Rubric Types:** 
1. **Positive Constraints** (e.g., "Must identify the car as red"):
    * Pass: "A red car is parked..."
    * Fail: "A blue car..." (Contradiction) OR "A car is parked..." (Missing specific detail).

2. **Negative Constraints** (e.g., "Must NOT mention a dog"):
    * Pass: "A cat sits on the mat." (No dog mentioned).
    * Fail: "A dog and a cat..." (Hallucination detected).

### Crucial Requirement
You must evaluate **Group A** and **Group B** independently and return the results in separate lists. The order of boolean results in the output must strictly match the order of the input rubrics.

### Output Format
Return the result strictly in valid XML format. Do not use Markdown code blocks.
<Assessment>
    <GroupA>
    <Result>true</Result>
    <Result>false</Result>
    <!-- Add more Result tags matching the number of rubrics in Group A -->
    </GroupA>
    <GroupB>
    <Result>true</Result>
    <Result>true</Result>
    <!-- Add more Result tags matching the number of rubrics in Group B -->
    </GroupB>
</Assessment>

Please evaluate the following caption against the provided rubric groups.
```

```ini
[Model Caption]
{caption}

[Group A: Critical Rubrics]
{group_a_rubrics}

[Group B: Granular Rubrics]
{group_b_rubrics}
```

## D. Human Annotation Feedback

To ensure the high quality of the benchmark, we involved human annotators in the loop. Given the extreme complexity of the images and the exceptional length of the golden captions (averaging 770.42 words), we employed the “Model-Ensemble-Vote-then-Human-Refine” pipeline. We utilized state-of-the-art multimodal models (specifically Gemini-3-Pro, GPT-5.2, and Seed-1.8) to generate initial drafts via a voting mechanism, followed by meticulous human verification.

Annotators reported that the AI-generated drafts were surprisingly sophisticated, significantly reducing the need for structural rewriting. However, the process introduced specific challenges regarding vigilance and fine-grained verification.

Hard Cases and Visual Nuances. The primary difficulty lay in fine-grained visual semantic alignment, particularly in regions with blurred edges, complex lighting, or severe occlusion. Annotators identified three recurrent types of “hard cases”:

• Material and Boundary Misinterpretation: Models occasionally merged ephemeral visual features with solid objects. A cited example involved a racing car where the model incorrectly described the “dust kicked up by the wheels” as a physical extension of the car’s bodywork.

• Precise Spatial Reasoning: Subtle prepositional errors were common. For instance, a model described a pig as standing “outside the pen,” whereas a closer inspection revealed it was actually standing “at the doorway” (threshold ambiguity).

• Hallucination in Low-Visibility Areas: In shadowed or blurry regions, models tended to hallucinate specific, irrelevant objects to complete the scene.

Annotation Policy: Determinism over Ambiguity. Our annotators adhered to a strict standard of determinism. Unlike models that might produce vague descriptions for unclear regions (e.g., “a blurry object”), humans preferred to delete hallucinations entirely rather than retaining ambiguous text. If an object was recognizable (e.g., via tool-assisted zooming), it was described explicitly; otherwise, it was removed to ensure the caption contained only grounded, high-confidence information.

Diversity of Caption Styles. Interestingly, annotators noted that the golden captions naturally exhibited distinct stylistic modalities, reflecting the versatile capabilities of the underlying models. The captions generally fell into two categories:

• Literary Narrative: Highly fluent, prose-style descriptions that focus on immersion and flow. These captions tend to be exceptionally long and use varied sentence structures to weave visual details into a cohesive story.

• Structured Representation: Captions that utilize Markdown formatting (e.g., bolding key terms, using bullet points for distinct regions) to present information in a highly organized, hierarchical manner.

We preserved this stylistic diversity in the final benchmark to evaluate models on both narrative generation and structured information extraction.

## E. Additional Experimental Results

Table 4 presents the comprehensive evaluation results across all models.

Table 4. Main evaluation results on PerceptionRubrics. Models are categorized into Open-Source and Proprietary groups and sorted by Overall Score in ascending order. All values are reported in percentage (%). M-R Item: Must-Right Item Accuracy; E-W Item: Easy-Wrong Item Accuracy; Gate Pass: The sample-level pass rate where all Must-Right items are correct (Must-Right All True); E-W Avg: The sample-level mean of per-case Easy-Wrong accuracy.

<table><tr><td>Model</td><td>Overall</td><td>M-R Item</td><td>E-W Item</td><td>Gate Pass</td><td>E-W Avg</td></tr><tr><td colspan="6">Open-Source Models</td></tr><tr><td>Qwen2.5-VL-7B</td><td>8.37</td><td>64.99</td><td>30.69</td><td>26.20</td><td>30.52</td></tr><tr><td>Qwen2.5-VL-32B</td><td>17.79</td><td>76.30</td><td>44.04</td><td>39.40</td><td>43.68</td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>34.13</td><td>85.33</td><td>59.72</td><td>56.65</td><td>59.26</td></tr><tr><td>Step3-VL-10B</td><td>35.97</td><td>85.63</td><td>59.05</td><td>58.96</td><td>58.95</td></tr><tr><td>Qwen3-VL-235B-A22B-Thinking</td><td>41.88</td><td>88.61</td><td>64.40</td><td>64.16</td><td>64.30</td></tr><tr><td>MiniMax-M3</td><td>44.82</td><td>89.21</td><td>66.40</td><td>65.70</td><td>65.94</td></tr><tr><td>MiMo-V2.5</td><td>45.65</td><td>88.49</td><td>65.77</td><td>66.76</td><td>65.57</td></tr><tr><td>Step-3.7-Flash</td><td>48.62</td><td>90.39</td><td>70.04</td><td>68.21</td><td>69.80</td></tr><tr><td>Kimi-K2.5</td><td>50.78</td><td>91.43</td><td>71.31</td><td>71.10</td><td>70.94</td></tr><tr><td>Kimi-K2.6</td><td>51.77</td><td>91.45</td><td>71.04</td><td>72.16</td><td>70.92</td></tr><tr><td>Qwen3.5-397B-A17B</td><td>61.61</td><td>93.64</td><td>78.01</td><td>78.90</td><td>77.59</td></tr><tr><td colspan="6">Proprietary Models</td></tr><tr><td>GPT-4o-2024-05-13</td><td>12.59</td><td>70.01</td><td>36.00</td><td>32.27</td><td>35.67</td></tr><tr><td>Seed-1.6</td><td>44.54</td><td>88.95</td><td>67.09</td><td>65.32</td><td>66.82</td></tr><tr><td>GLM-5V-Turbo</td><td>48.18</td><td>90.84</td><td>68.79</td><td>69.17</td><td>68.60</td></tr><tr><td>Seed-1.8</td><td>54.34</td><td>91.69</td><td>72.12</td><td>73.60</td><td>71.91</td></tr><tr><td>GPT-5.5</td><td>55.23</td><td>93.21</td><td>69.34</td><td>78.23</td><td>69.47</td></tr><tr><td>Gemini-3-Flash</td><td>59.83</td><td>93.25</td><td>75.44</td><td>78.23</td><td>75.01</td></tr><tr><td>GPT-5.4</td><td>60.81</td><td>93.60</td><td>74.88</td><td>79.58</td><td>74.73</td></tr><tr><td>Seed-2.0-Pro</td><td>61.44</td><td>92.95</td><td>78.29</td><td>77.65</td><td>77.97</td></tr><tr><td>Qwen3.5-Plus</td><td>61.61</td><td>93.86</td><td>77.86</td><td>79.09</td><td>77.40</td></tr><tr><td>Qwen3.6-Plus</td><td>62.30</td><td>93.96</td><td>77.48</td><td>79.67</td><td>77.04</td></tr><tr><td>Gemini-3-Pro</td><td>68.79</td><td>95.26</td><td>81.96</td><td>83.62</td><td>81.67</td></tr><tr><td>Gemini-3.1-Pro</td><td>69.02</td><td>95.69</td><td>81.47</td><td>84.49</td><td>81.22</td></tr><tr><td>Gemini-3.5-Flash</td><td>69.88</td><td>95.52</td><td>82.81</td><td>84.01</td><td>82.35</td></tr><tr><td>Seed-2.0-Lite</td><td>70.07</td><td>95.59</td><td>84.69</td><td>82.85</td><td>84.23</td></tr></table>

## F. Qualitative Examples

We provide concrete examples of the generated rubrics across diverse domains in Figure 12 and Figure 13.

As shown in the figures, our benchmark covers seven major categories, ranging from daily natural scenes to highly specialized STEM diagrams and logic puzzles. (a) For each image, we generate a comprehensive set of fine-grained rubrics. The items marked with the “OK” icon (Must-Right) represent core factual elements and primary subjects that are essential for a basic understanding of the scene. (b) The items marked with the “Thumbs-up” icon (Easy-Wrong) target more challenging details, including spatial relationships, fine-grained text recognition, negative constraints (e.g., “must NOT mention...”), and complex logical reasoning. These rubrics are specifically designed to be “Easy-Wrong” for current large multi-modal models, effectively exposing hallucinations and subtle comprehension errors. For instance, in the “Structured Data” and “STEM & Expert” cases, the rubrics require precise reading of axis scales, curve styles, and hierarchical biological relationships, which demand a high level of visual-logical alignment.

![](images/9fe6b3d3469256b81fb3099c220620d035de094441ddf28cad127b3ad322bedf.jpg)  
<sub>Logic & Puzzle</sub>Figure 12. Qualitative examples of the fine-grained rubrics across four categories: Natural Scene, Document & OCR, Digital UI & UX, and Structured Data. Each example consists of an image and two tiers of rubrics: Must-Right (top group) focusing on core facts, and Easy-Wrong (bottom group) focusing on challenging details, negative constraints, and logical reasoning

## Logic & Puzzle

• a w orksheet w ith the title “H O W M A N Y” o r the instructio n t o co unt the shapes

• m ultiple geom etric shapes ( e.g., hearts, stars, circles, rectangles, pentago ns) scattered i n the central area

• the answ er section a t the bo tto m a s co nsisting o f specific shapes paired w ith em pty boxes

• the instructio n text a s “C ount the sim ilar shapes and w rite the correct num ber”

• the yellow geo m etric shapes a s rectangles o r slanted bars (no t squares)

• the black geo m etric shapes a s triangles

• exactly tw o black triangles i n the m ain shape

• the dark blue shapes a s octagons (o r stop-sign shaped)

• the answ er entry boxes i n the bottom section are em pty o r blank

• m ust N O T state that the dark blue octagon appears i n the bottom answ er key section

## STEM & Expert

the hierarchical trophic levels labeled 'Producers', 'Herbivores', 'Scavengers', and 'D etritivo res'

the central tier label explicitly a s 'C arnivores and O m nivores' (o r no tes the specific typo 'O m viores' visible i n the im age).

specific organism s depicted i n the photographic thum bnails ( e.g., Acacia Tree, G iraffe, Lion, Hyena, o r Vulture)

the to p 'D etritivo res' tie r fe a ture s te xt la be ls (Te rm ite s, Ba cte ria , F ungi) ra the r tha n photographs

![](images/b5f450b455b5b75bc47dbaa43f083d8cc6258a7cecdb38d87dfa70cded3fcc11.jpg)

a food w eb o r ecosystem chart consisting o f five hierarchical tiers/levels

• the tw o Scavengers depicted a s a Hyena and a Vulture

the specific anim als i n the Carnivo re tier, including the Egyptian M ongoose and Black M am ba

the specific anim als i n the Herbivore tie r, including the Grasshopper and Klipspringer

• the three distinct tree species labeled: A cacia Tree, Baobab Tree, distinguish betw een the tw o specific grass types labeled: Elephant Grass and Berm uda Grass

the visual flow of the diagram , noting that arrow s point upw ard from the Producers a t the bottom t o the higher trophic levels

m ust NO T m ention the presence o f anim als not depicted i n the im age, such a s leopards, hippopotam uses, o r crocodiles

## Creative & Cultural

• a m ovie poster

• the title text 'BROTHERHOOD O F BLADES'

• three central figures o r m en i n the co m po sitio n

the red Rom an num eral 'II'

• the subtitle text a s 'THE IN FER N A L BA TTLEFIELD'

• the Ro m a n num e ra l 'II' i s red o r red d ish i n co lo r the central figure's pose a s having arm s extended w ide o r spread out

actor nam es ( e.g., CHANG CHEN, YANG M I) along the top edge

the figure i n the background (left) i s w earing a tall black ha

Chinese calligraphy o r characters overlaid o n the central figures

m ust N O T describe the background a s w arm , sunny, o r brightly colored (it i s co ld/desaturated/gray)

Figure 13. Qualitative examples of the fine-grained rubrics across three additional categories: Logic & Puzzle, STEM & Expert, and Creative & Cultural. Each example consists of an image and two tiers of rubrics: Must-Right (top group) focusing on core facts, and Easy-Wrong (bottom group) focusing on challenging details, negative constraints, and logical reasoning.