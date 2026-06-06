# VideoKR: Towards Knowledge- and Reasoning-Intensive Video Understanding

Lin Fu 1 \* Zheyuan Yang 2 \* Yang Wang 1 Tingyu Song 3 Arman Cohan 4 Yilun Zhao 4

![](images/0114680b733a1d2338ec50394007a296b7c72b3b18c3887b17c3f786c25ee9dd.jpg)

VideoKR

![](images/2e9ba58d263aa11357584020f7d51c5b19350a837c2ac85a87235347a84b69ad.jpg)

VideoKR

# Abstract

We introduce VideoKR, the first large-scale training corpus specifically designed to strengthen knowledge- and reasoning-intensive video understanding. It comprises 315K video reasoning examples over 145K newly collected, CClicensed, expert-domain videos. We develop a human-in-the-loop, skill-oriented example generation pipeline that targets progressively deeper video reasoning capabilities while ensuring the difficulty, diversity, and reliability of both the examples and their CoT rationales. We also curate VideoKR-Eval, a new expert-annotated benchmark where questions require genuine video understanding and knowledge-intensive reasoning rather than textual shortcuts. Our experiments show that, under a standard SFT→GRPO pipeline, models post-trained on VideoKR outperform prior post-training approaches on knowledge-intensive video reasoning while remaining competitive on general video reasoning, highlighting data design as a key driver of progress in video reasoning. We further conduct comprehensive ablations to isolate the contributions of VideoKR, providing actionable insights for future work.

# 1. Introduction

Multimodal foundation models for video understanding have achieved rapid progress in recent years, driven by architectural advances (Shu et al., 2025; Zohar et al., 2025; Ren et al., 2025; Li et al., 2024d; 2026), large-scale pretraining (Zhang et al., 2025a; Wang et al., 2025c; Bai et al.,

![](images/71136d177577ff8dc04f4260946d609994acf25781224c0d5e90a0a700b5efe5.jpg)

<details>
<summary>donut</summary>

Knowledge Distribution
| Category | Percentage (%) |
| :--- | :--- |
| Healthcare | 3.2 |
| Social Science | 40.1 |
| Engineering | 18.8 |
| Natural Science | 37.9 |

| Core Skills | Example of Collected CC-licensed Video |
|---|---|
| Basic Video Perception | 00:35 |
| Reasoning | 00:38 |
| VideoKR-SFT-201K | VideoKR-RL-114K |
| Legend: Medical Laboratory Science, Biomedical Informatics, Public Health, Radiography, Nursing, Others | 00:44 |
| Legend: Software Engineering, Mechatronics & Robotics, Mechanical Engineering, Electrical Engineering, Civil Engineering, Others | 01:09 |
| Legend: Chemistry, Biochemistry, Atmospheric Science, Mathematics, Geology, Others | 01:09 |
The chart displays a donut chart with two main segments: the left segment (e.g., Social Science) shows 40.1% coverage while the right segment (e.g., Engineering) shows 18.8%. The bottom section is a small icon labeled 'Knowledge Reasoning' pointing to a graph symbolizing visual perception. The bottom-right image shows a video capture timeline from 00:35 to 01:09.
</details>

QA Examples via Skill-Oriented Example Generation

Q: What is the overall trend in thermometer reading?

Answer: Rising Reasoning: (...abbreviated...) The liquid column between 1:06 – 1:12 is rising (...abbreviated...)

Q: What is the role of the black solid?

A. Homogeneous catalyst B. Oxidant C. Heterogeneous catalyst D. Dehydrating agent

Answer: C Reasoning: (...abbreviated...) ;<#!(solid) catalyzes "!#! (liquid), therefore it is a heterogeneous catalysis.

Q: Using the same solution as in the video (35 .0(), 18.0%(./.), density 1.081 ⋅ ()!") and excess black powder, determine the volume of dry gas produced at STP.

Answer: 2.24L Reasoning: From 00:30 to 00:52 in the video, we (...abbreviated...), therefore, we identify the demo as the decomposition of hydrogen peroxide to water and oxygen ( 2"!#! → 2"!# + #! ). (...abbreviated...) For 35.0\*+ of an 18.0%(0/0) solution(3 = 1.085/\*+), the solution mass is 35.0×1.08 = 37.805 ; therefore, \* "!#! = (...abbreviated...) the dry oxygen volume is 0.1×22.414 ≈ 2.24+.

Figure 1. An overview of the VideoKR training corpus. All videos are newly collected and CC licensed, and span a wide range of professional domains. We develop a skill oriented QA synthesis pipeline in which every example is grounded in one of three core skills essential for advanced video reasoning, and examples in the CoT subset are further paired with a high quality reasoning trace.

2025; Chen et al., 2025a), and sophisticated post-training strategies (Feng et al., 2025a; Meng et al., 2025; Wang et al., 2025a; Li et al., 2025a;b). However, current models still face significant limitations when transitioning from surfacelevel video perception to video reasoning tasks that demand domain knowledge and multi-step inference (Zhao et al., 2025b; Hu et al., 2025; Deng et al., 2025; Song et al., 2025). A key bottleneck lies in the nature of the training corpora used to develop these models.

Table 1. Comparison of VideoKR with prior post-training corpora for video understanding. %Video denotes the fraction of video understanding examples, CC indicates whether all videos are Creative Commons (CC) licensed. †: data has not been open-sourced. 

<table><tr><td></td><td>Video Source</td><td>% Video</td><td># Video</td><td>Avg Duration</td><td>CC</td><td># Example</td><td>Example Source</td><td>Expert-domain</td><td>Example/CoT Generator</td></tr><tr><td>LLaVA-Video (Zhang et al., 2025c)</td><td>Existing Dataset</td><td>100%</td><td>178K</td><td>36.9 seconds</td><td>✕</td><td>1156K</td><td>Newly Generated</td><td>✕</td><td>GPT-4o</td></tr><tr><td>VideoEspresso (Han et al., 2025)</td><td>Existing Dataset</td><td>100%</td><td>259K</td><td>47.7 seconds</td><td>✕</td><td>202K</td><td>Newly Generated</td><td>✕</td><td>GPT-4o</td></tr><tr><td>Video-R1 (Feng et al., 2025a)</td><td>Existing Dataset</td><td>52%</td><td>61K</td><td>36.9 seconds</td><td>✕</td><td>260K</td><td>Existing Dataset</td><td>✕</td><td>Qwen2.5-VL-72B-Instruct</td></tr><tr><td>VideoRFT (Wang et al., 2025a)</td><td>Existing Dataset</td><td>56%</td><td>127K</td><td>24.7 seconds</td><td>✕</td><td>310K</td><td>Existing Dataset</td><td>✕</td><td>GPT-4o-mini + DeepSeek-R1</td></tr><tr><td> $^{\dagger}$  Video-CoT (Zhang et al., 2025b)</td><td>Existing Dataset</td><td>100%</td><td>Unknown</td><td>Unknown</td><td>✕</td><td>192K</td><td>Existing Dataset</td><td>✕</td><td>Qwen2.5-VL-72B-Instruct</td></tr><tr><td>OneThinker (Feng et al., 2025b)</td><td>Existing Dataset</td><td>42%</td><td>158K</td><td>90.9 seconds</td><td>✕</td><td>600K</td><td>Existing Dataset</td><td>✕</td><td>Seed1.5-VL</td></tr><tr><td>VideoAuto-R1 (Liu et al., 2026a)</td><td>Existing Dataset</td><td>59%</td><td>35K</td><td>63.8 seconds</td><td>✕</td><td>83K</td><td>Existing Dataset</td><td>✕</td><td>-</td></tr><tr><td>VideoKR (Ours)</td><td>Newly Collected</td><td>100%</td><td>145K</td><td>344.1 seconds</td><td>√</td><td>315K</td><td>Newly Generated</td><td>√</td><td>Expert-validated selection from a pool of 7 frontier models</td></tr></table>

Existing large-scale video datasets are predominantly constructed for perceptual objectives such as action recognition, event localization, and short-range temporal understanding (Feng et al., 2025a; Wang et al., 2025a; Meng et al., 2025; Chen et al., 2025b; 2026). Their content is heavily skewed toward everyday activities, with limited coverage of specialized domains and little support for knowledgeand reasoning-intensive video understanding. Consequently, models trained on current corpora often struggle with tasks requiring multi-hop inference, scientifically grounded explanations, or interpretation of events governed by nonobservable principles, limiting their reliability in real-world applications that demand accurate, domain-aware reasoning.

To bridge this gap, we open-source VideoKR, the first large-scale training corpus targeted for knowledge- and reasoning-intensive video understanding. We collect 145K CC-licensed videos across 82 professional subjects using a knowledge-driven collection protocol that targets realworld manifestations of domain knowledge. To transform these raw videos into effective video reasoning training data, we design a skill-oriented QA generation framework that decomposes knowledge- and reasoning-intensive video understanding into three complementary capabilities: basic video reasoning, knowledge-enhanced video perception, and knowledge-intensive video reasoning. For each video, the framework generates challenging QA examples tailored to each skill category, each paired with a high-quality CoT rationale. We apply rigorous quality control with humanexpert involvement, yielding a high-quality supervised finetuning corpus, VideoKR-SFT-201K, and a reinforcement learning corpus, VideoKR-RL-114K. In addition, through a manual audit of existing knowledge-intensive video reasoning benchmarks, we find that many examples are solvable with little video understanding. To address these issues, we construct a new evaluation benchmark, VideoKR-Eval.

We adopt a standard SFT→GRPO pipeline to isolate data design as the primary bottleneck and attribute performance gains more cleanly to VideoKR. We further establish a standardized evaluation framework to enable fair, reproducible model comparisons. Experiments show that even without sophisticated post-training algorithmic design, base models (i.e., Qwen2.5-VL-7B-Instruct and Qwen3-VL-8B-Instruct) post-trained on VideoKR already outperform prior posttraining approaches. To inform how VideoKR can advance future work in video reasoning, we conduct comprehensive ablations that disentangle its key contributors, including the effect of CoT supervision, the impact of skill-based data composition, and controlled SFT and RL studies that compare VideoKR against prior post-training corpora.

We summarize our main contributions as follows:

• We open-source VideoKR, the first large-scale training corpus designed for knowledge- and reasoning-intensive video understanding. We apply rigorous quality control to ensure consistently high-quality training data (§3).   
• We construct VideoKR-Eval, a new evaluation benchmark that mitigates single-frame answerability in prior benchmarks through multi-model single-frame probing and expert re-annotation of filtered videos (§4).   
• We establish a standardized evaluation framework to ensure fair and reproducible model comparisons (§5.2).   
• Models post-trained on VideoKR achieve the best knowledge-intensive performance among similar-sized models, while remaining competitive on general video benchmarks (§6).   
• We conduct comprehensive ablations to isolate the contributions of VideoKR, including the effectiveness of CoT supervision, the impact of skill-based data composition, comparisons against prior post-training corpora, yielding actionable insights for future work (§6.4 to §6.5).

# 2. Related Work

Video Understanding Datasets. Recent video understanding benchmarks have expanded their scope to evaluate a broader range of multimodal and reasoning capabilities (Wu et al., 2021; Zhou et al., 2025; Fu et al., 2025; Zhao et al., 2025b; Deng et al., 2025; Li et al., 2024c; Liu et al., 2026b; Xu et al., 2025). General-purpose benchmarks such as Video-MME (Fu et al., 2025), MVBench (Li et al., 2024b), VSI-Bench (Yang et al., 2025b), and Video-Vista (Li et al., 2024c) assess perceptual skills, spatiotemporal comprehension, and cross-modal reasoning, providing a solid foundation for evaluating video understanding. Building on this trend, a growing set of knowledge-, science-, and reasoning-intensive evaluation benchmarks focuses on deeper, domain-aware reasoning that goes beyond surface-level video perception (Xu et al., 2025; Liu et al., 2026b). For instance, MMVU (Zhao et al., 2025b) requires models to reason over specialized-domain videos and apply relevant domain knowledge; VideoMMMU (Hu et al., 2025) and Video-MMLU (Song et al., 2025) target expert-level understanding of subject-specific lecture videos; and SciVideoBench (Deng et al., 2025) evaluates advanced reasoning over scientific videos.

![](images/1800676153157c0ddab0170ddf4db3393908450654ffd9d4333ec1593dd9386f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Domain Knowledge Bank Construction (§3.1)"] --> B["Construct knowledge bank"]
    B --> C["63,745 Knowledge Points"]
    C --> D["Knowledge Point"]
    D --> E["Generate search keywords"]
    E --> F["1-3 Scenarios"]
    F --> G["Search Videos with CC license"]
    G --> H["Corresponding Search Keywords"]
    H --> I["Judge video relevance to targeted scenarios"]
    
    J["Knowledge-Driven Video Collection (§3.2)"] --> K["82 subjects across 4 disciplines"]
    K --> L["Name: Redox reaction Definition: A chemical process in which one substance loses electrons...abbreviated..."]
    L --> M["Course Lecture"]
    M --> N["Subject"]
    
    O["Skill-Oriented Example Generation (§3.3)"] --> P["Core Skills"]
    P --> Q["Basic Video Perception"]
    Q --> R["Knowledge Reasoning"]
    R --> S["Expert annotate seed examples"]
    S --> T["150 examples × 3 skills × 4 disciplines"]
    
    U["Data Quality Control (§3.4)"] --> V["Human-Validated Model Selection for Each Pipeline Step"]
    V --> W["For each step, sample 100 examples per model, have them validated by experts, select models that meet quality criteria for that step"]
    W --> X["Data Contamination Mitigation"]
    X --> Y["1. YouTube-ID Filtering"]
    X --> Z["2. Near-Duplicate Video Filtering via frame hashing then sequence matching"]
    Y --> AA["Self-consistency Check"]
    Z --> AB["Video Dependency Filter"]
    Z --> AC["CoT Rationale Validation"]
```
</details>

<table><tr><td>Feature</td><td>SFT-201K</td><td>RL-114K</td></tr><tr><td># Examples</td><td>201,156</td><td>114,381</td></tr><tr><td># Multi-choice</td><td>99,843</td><td>54,461</td></tr><tr><td>Question Length</td><td>30.6</td><td>31.7</td></tr><tr><td># Open-ended</td><td>101,313</td><td>59,920</td></tr><tr><td>Question Length</td><td>11.6</td><td>13.1</td></tr><tr><td>% VIDR</td><td>43.48%</td><td>35.66%</td></tr><tr><td>% KNOWVID</td><td>33.02%</td><td>33.38%</td></tr><tr><td>% KNOWVIDR</td><td>23.49%</td><td>30.96%</td></tr><tr><td># Knowledge Points</td><td>20,372</td><td>12,446</td></tr><tr><td># Videos</td><td>85,934</td><td>59,625</td></tr><tr><td>Avg Length (second)</td><td>339.0</td><td>351.6</td></tr><tr><td>25s - 5min</td><td>57.61%</td><td>55.16%</td></tr><tr><td>5min - 10min</td><td>25.35%</td><td>26.61%</td></tr><tr><td>10min - 30min</td><td>17.04%</td><td>18.23%</td></tr><tr><td>CoT Rationale Length</td><td>196.9</td><td>-</td></tr></table>

Figure 2. (Left) Overview of data construction pipeline. (Right) Statistics of VideoKR-SFT-201K and VideoKR-RL-114K training corpus.

Post-training for Video Understanding. Current reasoning models are typically trained through a two-stage posttraining pipeline that combines SFT and RL (Guo et al., 2025; Team et al., 2025; Yang et al., 2025a; Li et al., 2025c). To enhance video reasoning capabilities, the SFT stage is usually initialized on video reasoning datasets that include explicit chain-of-thought annotations, temporal cues, and spatial grounding signals, helping the model form more structured and interpretable reasoning patterns (Munasinghe et al., 2025; Zohar et al., 2024; Wu et al., 2025; Feng et al., 2025b; Wang et al., 2025b). In the RL phase, recent work has concentrated on adapting reinforcement learning with verifiable rewards (RLVR) to video reasoning, exploring complex reward engineering that emphasizes spatial understanding (Ouyang et al., 2025; Tang et al., 2025a), temporal dynamics (Wang et al., 2025d; Chen et al., 2025b; Feng et al., 2025a; Zhao et al., 2025a), or integrated spatiotemporal relationships (Meng et al., 2025; Tang et al., 2025b). Despite these advances, most post-training approaches still build on repurposed video understanding datasets that target basic perception: as shown in Table 1, existing open-source corpora largely rely on short videos from datasets released years ago, and synthesis-based efforts (Zhang et al., 2025c; Han et al., 2025) typically depend on a single model, which can introduce systematic biases. We address this need by open-sourcing VideoKR: every video is newly collected, depicts expert-domain scenarios, and is released under CC licenses, and we adopt a human-in-the-loop, skill-oriented example generation framework to ensure the difficulty, diversity, and reliability of the data. Our experiments show that, under a standard SFT→GRPO pipeline, models posttrained on VideoKR already outperform prior post-training approaches, suggesting that better data remains crucial.

# 3. VideoKR Training Corpus Construction

This section describes the VideoKR data construction process, with an overview of the pipeline shown in Figure 2. Because our goal is large-scale corpus synthesis, exhaustive manual construction is infeasible. However, model based generation can introduce systematic artifacts. We therefore adopt a quality-controlled, semi-automated pipeline: whenever a step involves model outputs, it is audited and validated by human experts (detailed in Section 3.4). We engage 34 domain experts, each with graduate-level background in the relevant discipline (see Appendix A.2 for annotator biographies), to enforce quality criteria throughout the process.

# 3.1. Domain Knowledge Bank Construction

To achieve comprehensive coverage of domain-related videos, we begin by constructing a Domain Knowledge Bank, where each entry represents a knowledge point consisting of a term and its corresponding definition. The authors manually reviewed undergraduate curricula from top universities worldwide and identified 82 representative subjects distributed across four major disciplines: Natural Sciences, Healthcare, Humanities and Social Sciences, and Engineering. The complete subject list is provided in Appendix A.1. To ensure systematic and fine-grained representation of domain knowledge within each discipline, we adopt a hierarchical knowledge organization framework with four layers: Subject → Course → Lecture → Knowledge Point. Specifically, for each subject, we ask expert annotators to provide a list of 4 to 8 core undergraduate courses consistent with standard academic programs. For each selected course, annotators compile a structured syllabus based on well-established curricula from top universities, outlining major lecture topics and learning objectives. Then for each lecture, we prompt LLMs (see Section 3.4 for our expertinvolved procedure to select and validate the models used here and in subsequent steps) to identify several key knowledge points, each paired with a term and a paragraph-length definition. We collect a total of 63,745 knowledge points.

# 3.2. Knowledge-Driven Video Collection

Building upon the constructed domain knowledge bank, we then collect a large-scale video corpus, outlined below.

Knowledge-based Video Scenario Generation. In practice, directly using knowledge point terms (e.g., “Newton’s Second Law”) as search queries often yields lecture recordings or purely instructional materials. While informative, such videos often lack the diversity and situational richness found in real-world contexts where the knowledge is implicitly applied. To obtain more authentic and engaging content, we first ask LLMs to generate 1–3 short scenarios that describe realistic situations involving each knowledge point. For example, instead of searching “Newton’s Second Law”, a generated scenario might be “a rocket launching into the sky”, which inherently reflects Newton’s Second Law. These scenarios are then transformed into semantically relevant search keywords, helping retrieve videos that embody the knowledge rather than merely explain it.

Scenario-Guided Video Search and Filtering. For each generated search keyword, we employ the YouTube Data API1 to retrieve metadata (e.g., titles, descriptions, and durations) for the top-10 candidate videos. We restrict the search to videos released under CC licenses to ensure legal reusability, a critical aspect that has been ambiguous in prior training corpora. For each candidate video, we instruct models to evaluate its relevance to the expected knowledge point and scenario based on the textual metadata. Videos exceeding 30 minutes are excluded, as long-context video understanding falls beyond the scope of this work. For the remaining candidates, we download the videos and prompt MLLMs to perform a secondary relevance assessment using visual content to confirm alignment with the intended knowledge context. To remove potentially harmful or sensitive content, we randomly sample four frames from each video and run Azure AI’s image moderation APIs2 to filter out unsafe videos. We collect 146,567 CC-licensed videos.

# 3.3. Skill-Oriented Example Generation

For each video, we generate multiple QA examples. Following recent video post-training work (Feng et al., 2025a; Meng et al., 2025), we adopt multi-choice and open-ended QA formats, as they offer verifiable supervision suitable for RLVR. To enable scalable and high-quality data creation, we design a skill-based example generation pipeline:

Core Skill Categorization. As illustrated in Figure 1, three complementary dimensions, i.e., perception, knowledge, and reasoning, are essential for knowledge- and reasoning-intensive video understanding. Accordingly, we define three core skills that guide the example generation process: (1) Basic Video Reasoning (VIDR), which involves direct comprehension of events observable from the visual sequence, such as tracking actions, spatial relations, or temporal order, without relying on external domain knowledge. (2) Knowledge-enhanced Video Perception (KNOWVID), where visual perception is enriched by explicit domain knowledge. The model must align observed visual cues with relevant concepts across both spatial and temporal dimensions, for example, recognizing laboratory apparatus such as a “burette” or “condenser” and understanding their roles in a sequence of chemical procedures. (3) Knowledge-Intensive Video Reasoning (KNOWVIDR), which focuses on integrating visual understanding with domain knowledge to perform sophisticated, multi-hop inference, e.g., estimating the quantity of chemical product formed from observed reactant amounts, or inferring a patient’s likely diagnosis by interpreting symptoms and medical procedures depicted in a clinical video.

Seed Examples Curation By Human Experts. To ensure the quality and domain accuracy of the generated examples, we engage expert annotators to curate a seed set of examples for every core skill defined above. For each core skill, the annotators select representative knowledge points and their corresponding collected videos, then construct question–answer pairs accompanied by detailed, step-by-step reasoning processes that clearly articulate how visual evidence, domain knowledge, and logical inference jointly lead to the final answer. Each annotated example then undergoes a manual review by the authors to verify the accuracy of the QA content and reasoning process. In total, 150 examples are created for each skill within every discipline, resulting in 1,800 high-quality, expert-curated seed examples. To improve reliability, each example is independently reviewed by a second annotator; 74 examples are revised in this stage.

Example Generation. Building on the expert-curated seed set, we use frontier MLLMs to scale up example creation in a controlled, skill-aware manner. For each video, the model generates two examples per skill, producing six examples in total through six independent generation rounds (one per example). During each round, the model is provided with (1) video frames uniformly sampled at 0.2 fps with timestamps, (2) three randomly sampled human-curated examples from the same discipline and skill category, and (3) the knowledge point and associated subjects when targeting the KNOWVID or KNOWVIDR skills. The model is instructed to emulate the seed examples in QA formulation and reasoning process generation, while maintaining fidelity to the video’s unique visual content and knowledge context.

Example Validation and Filtering. To mitigate generation errors and reasoning bias, we adopt three complementary strategies: (1) Self-Consistency Verification: The model is re-prompted with the generated question and corresponding video frames to produce a detailed, step-by-step answer. An example is retained only if the re-derived answer matches the original, and the reasoning process from this verification step is used as the final reasoning trace. (2) Video Dependency Filtering: To ensure that each example in VideoKR genuinely requires visual understanding rather than relying on textual cues or shortcut reasoning, we instruct InternVL3.5-38B and Qwen3-VL-32B-Instruct to answer the question using only the text and four randomly sampled video frames. If both models successfully predict the correct answer under this limited setting, the example is removed from the dataset. Notably, this filter is stricter than what existing evaluation benchmarks use, which commonly rely on text-only (Fu et al., 2025; Di & Xie, 2024) or singleframe (Saravanan et al., 2025; Plizzari et al., 2025) settings. (3) CoT Rationale Validation: To mitigate systematic generator bias in the reasoning traces, we use an independent strong MLLM as a verifier: given the question, reasoning trace, and videos, it checks that each key step is supported by observable evidence or standard domain knowledge, and that the reasoning decisively distinguishes the chosen answer from plausible alternatives. We discard examples with critical unsupported steps.

# 3.4. VideoKR Data Quality Control

Beyond the automated example validation and filtering described above, we further strengthen VideoKR through pipeline-level quality control and contamination mitigation.

Human-Validated Model Selection for Each Pipeline Step. As summarized in Table 1, prior work on video reasoning corpus construction typically relies on a single model throughout the pipeline. This design can introduce model specific artifacts. To improve synthesis diversity and avoid overcommitting to any single model’s biases, we use a pool of seven frontier models (i.e., GPT-5.2, GPT-5- mini, Claude-4.5-Sonnet, Gemini-3-Flash, DeepSeek-V3.2, Qwen3-VL-235B-A22B, and GLM-4.6V). However, our pipeline is difficulty stratified: lightweight steps such as metadata relevance screening can be handled reliably by multiple models, whereas demanding stages such as QA example generation and verification are reliable only for a subset of models. We therefore introduce a human validated model selection protocol to determine stage eligibility: For each candidate model and each pipeline step, we sample 100 instances from that step’s real inputs and ask domain expert annotators to assess the model outputs and label errors. A model is eligible for a step only if its error rate falls below a predefined threshold. We provide details of this process and the human-validated models for each step in Appendix A.4. During large-scale synthesis, for each instance at a given step, we randomly select one qualified model from the pool.

Data Contamination Mitigation. To prevent evaluation leakage, we traverse all video benchmarks supported by LMMs-Eval (Li et al., 2024a) and apply a two-stage decontamination protocol over the videos in VideoKR: (1) YouTube-ID Filtering: When a benchmark (e.g., MMVU) provides YouTube video IDs, we directly filter out any training video whose YouTube ID matches an evaluation video ID, resulting in 131 videos removed. (2) Near-Duplicate Video Filtering: We also perform duplicate detection using frame level perceptual hashing and windowed sequence matching, resulting in 877 videos removed. We detail the process in Appendix A.5.

Manual Quality Assessment. To further assess the endto-end quality of the finalized corpus and quantify residual noise, we randomly sample 800 examples from VideoKR-SFT-201K and ask ten expert annotators, who previously participated in seed-example curation, to evaluate them end to end. Of the sampled items, 52 questions are flagged as potentially non visual solvable. For reasoning traces, annotators identify 32 errors. Among these, 17 cases change the final answer, while 15 cases preserve the correct answer but rely on unsupported domain claims or fail to ground key steps in the relevant video evidence. These error rates are comparable to the error levels observed during human expert seed example curation (§3.3) and are therefore acceptable.

# 3.5. VideoKR-SFT-201K & VideoKR-RL-114K

We then randomly partition the generated 315,537 examples into two subsets while preserving video-level grouping, resulting in VideoKR-SFT-201K for supervised fine-tuning and VideoKR-RL-114K for RLVR training. For VideoKR-SFT-201K, each example retains its validated CoT rationale as the supervision target, whereas VideoKR-RL-114K keeps only the question and verifiable answer, since RLVR optimizes against the verifiable answer while the policy model generates its own reasoning during training. Figure 2 presents the key data statistics. Randomly-sampled examples from VideoKR are shown in Appendix A.3.

Table 2. Single-frame answerability rates across existing benchmarks. A QA example is classified as single-frame-solvable for a model only if the model answers it correctly in all three independent trials using only the question, answer options, and one randomly sampled video frame. 

<table><tr><td>Model</td><td>VidMMMU (900)</td><td>MMVU (1,000)</td><td>SciVidBench (1,000)</td><td>VideoKR-Eval (ours) (2,000)</td></tr><tr><td>Claude-4.5-Sonnet</td><td>35.3</td><td>41.3</td><td>21.8</td><td>9.5</td></tr><tr><td>Qwen3-VL-235B-A22B</td><td>39.3</td><td>45.2</td><td>13.2</td><td>10.1</td></tr><tr><td>GPT-5.2</td><td>38.3</td><td>49.7</td><td>23.0</td><td>10.7</td></tr></table>

# 4. VideoKR-Eval Evaluation Benchmark

We next discuss the motivation and build of VideoKR-Eval.

# 4.1. Limitations of Existing Benchmarks

We observe that existing knowledge-intensive video reasoning benchmarks (e.g., VideoMMMU, MMVU, and SciVideoBench) contain a substantial fraction of examples that can be answered without continuous video understanding. We quantify this issue through single-frame probing: each model is given only the question, answer options, and one randomly sampled frame from the video, and each example is evaluated in three independent trials. As shown in Table 2, frontier models achieve surprisingly high single-frame answerability rates on existing benchmarks (e.g., >35% on MMVU and VideoMMMU).

# 4.2. VideoKR-Eval Benchmark Construction

We construct VideoKR-Eval from VideoMMMU, MMVU, and SciVideoBench by retaining 1,254 original examples that require continuous video understanding under multimodel single-frame probing, and augmenting them with 746 expert-reannotated examples from filtered videos.

Multi-Model Single-Frame Filtering. For each example in VideoMMMU, MMVU, and SciVideoBench, we run single-frame probing with three frontier models: Qwen3- VL-235B-A22B, Claude-4.5-Sonnet, and GPT-5.2. Each model receives only the question, answer options, and one randomly sampled frame from the video, and is evaluated with three independent trials. For each model, an example is considered single-frame-solvable if the model answers it correctly in all three trials; otherwise, it is treated as requiring continuous video understanding for that model. We retain only the intersection of examples judged as requiring continuous video understanding by all three models, yielding 1,254 original examples.

Expert Re-annotation of Filtered Videos. For examples outside this intersection, we discard the original QA pairs and ask domain experts to re-annotate new QA examples using the corresponding videos. Annotators are required to write questions grounded in clearly observable video evidence, requiring relevant domain knowledge, and paired with uniquely determined ground truth answers. This process yields 746 expert-reannotated examples. Together with the 1,254 retained original examples, VideoKR-Eval contains 2,000 examples. Detailed statistics for VideoKR-Eval are provided in Appendix B.1.

# 5. Experiment Setup

In this section, we discuss the experiment setup for posttraining on VideoKR and the subsequent model evaluation.

# 5.1. Post-Training on VideoKR

Recent post-training work for video reasoning emphasizes sophisticated RL variants and reward engineering. In contrast, we aim to isolate a different, and arguably more fundamental, bottleneck: whether data design is the primary limiting factor for knowledge and reasoning intensive video understanding. Accordingly, we deliberately adopt a standard, widely used SFT→GRPO pipeline as a controlled scaffold, ensuring that algorithmic complexity does not become a confounder and that observed gains can be attributed more cleanly to the training data.

Specifically, we use Qwen2.5-VL-7B-Instruct and Qwen3- VL-8B-Instruct as base models to assess whether VideoKR yields consistent gains under various architectural designs and pretraining priors. For SFT, we fine-tune each base model on VideoKR-SFT-201K for one epoch. Starting from the resulting SFT checkpoint, we then run GRPO on VideoKR-RL-114K for one epoch. For Qwen3-VL-8B-Instruct, we also conduct Zero-RL training by directly running GRPO on VideoKR-RL-114K for one epoch. We set batch size as 32 for both SFT and GRPO. For the GRPO accuracy reward, we follow prior video reasoning work (Feng et al., 2025a; Wang et al., 2025a) and use ROUGE for openended QAs and Exact Match for multiple-choice QAs. The maximum video token number is 4,096, and the maximum number of frames is 128. The training hyperparameters and details are provided in Appendix C.1.

# 5.2. Evaluation Setup

We next describe our evaluation benchmarks and the standardized protocol for fair, reproducible model comparisons.

Evaluation Benchmarks. We evaluate models on seven benchmarks grouped into two categories: (1) General Video Reasoning, including Video-MME (Fu et al., 2025), MVBench (Li et al., 2024b) and LongVideoBench (Wu et al., 2024), which measure broad video understanding; and (2) Knowledge-intensive Video Reasoning, including VideoM-MMU (Hu et al., 2025), MMVU (Zhao et al., 2025b),

Table 3. Benchmark results across general and knowledge-intensive video reasoning. Models are grouped into (i) Other Models and (ii) methods built on Qwen2.5-VL-7B-Instruct or Qwen3-VL-8B-Instruct (with the indicated input Frames). Within each group for (ii), the best score is bold and the second-best is underlined. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Release</td><td rowspan="2">Frames</td><td colspan="4">General Video Reasoning</td><td colspan="5">Knowledge-Intensive Video Reasoning</td></tr><tr><td>Video-MME</td><td>MVBench</td><td>LongVideoBench</td><td>Average</td><td>VideoMMMU</td><td>MMVU</td><td>SciVideoBench</td><td>VideoKR-Eval</td><td>Average</td></tr><tr><td colspan="12">Closed-source models</td></tr><tr><td>GPT-5.4</td><td>2026-03</td><td>64</td><td>86.0</td><td>78.3</td><td>76.7</td><td>80.3</td><td>87.1</td><td>82.0</td><td>52.9</td><td>63.2</td><td>71.3</td></tr><tr><td>Gemini 3 Pro</td><td>2025-11</td><td>64</td><td>87.7</td><td>74.1</td><td>77.4</td><td>79.7</td><td>87.6</td><td>77.5</td><td>50.4</td><td>60.3</td><td>69.0</td></tr><tr><td>Claude Opus 4.5</td><td>2025-11</td><td>64</td><td>81.4</td><td>67.2</td><td>67.2</td><td>71.9</td><td>84.4</td><td>77.3</td><td>48.6</td><td>56.3</td><td>66.7</td></tr><tr><td colspan="12">Other Models</td></tr><tr><td>Qwen3-VL-32B-Thinking</td><td>2025-10</td><td>128</td><td>72.9</td><td>71.7</td><td>63.8</td><td>69.5</td><td>69.6</td><td>67.8</td><td>42.1</td><td>50.2</td><td>57.4</td></tr><tr><td>Qwen3-VL-32B-Instruct</td><td>2025-10</td><td>128</td><td>74.4</td><td>72.6</td><td>65.5</td><td>70.8</td><td>72.0</td><td>68.2</td><td>39.7</td><td>45.0</td><td>56.2</td></tr><tr><td>Qwen2.5-VL-72B</td><td>2025-02</td><td>128</td><td>72.8</td><td>67.4</td><td>63.2</td><td>67.8</td><td>67.0</td><td>65.1</td><td>38.9</td><td>42.6</td><td>53.4</td></tr><tr><td>InternVL3.5-8B</td><td>2025-08</td><td>128</td><td>65.5</td><td>73.3</td><td>61.0</td><td>66.6</td><td>57.2</td><td>54.0</td><td>24.5</td><td>35.4</td><td>42.8</td></tr><tr><td>LLaVA-OneVision-7B</td><td>2024-08</td><td>32</td><td>59.0</td><td>58.1</td><td>56.5</td><td>57.9</td><td>36.2</td><td>43.1</td><td>16.2</td><td>23.5</td><td>29.8</td></tr><tr><td>LLaVA-NeXT-Video-34B</td><td>2024-07</td><td>32</td><td>51.0</td><td>48.0</td><td>49.9</td><td>49.6</td><td>19.1</td><td>39.4</td><td>16.0</td><td>18.8</td><td>23.3</td></tr><tr><td>LLaVA-NeXT-Video-7B</td><td>2024-07</td><td>32</td><td>32.0</td><td>38.1</td><td>38.9</td><td>36.3</td><td>21.8</td><td>28.1</td><td>10.9</td><td>14.7</td><td>18.9</td></tr><tr><td colspan="12">Qwen2.5-VL-7B-Instruct or Qwen3-VL-8B-Instruct as Base Models</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>2025-02</td><td>16</td><td>57.1</td><td>65.0</td><td>55.2</td><td>59.1</td><td>48.4</td><td>52.5</td><td>23.1</td><td>31.3</td><td>38.8</td></tr><tr><td>Video-R1</td><td>2025-03</td><td>16</td><td>59.7</td><td>65.5</td><td>55.3</td><td> $60.2_{+1.1}$ </td><td>51.1</td><td>53.3</td><td>26.6</td><td>28.9</td><td> $40.0_{+1.2}$ </td></tr><tr><td>VideoRFT</td><td>2025-05</td><td>16</td><td>57.6</td><td>61.7</td><td>53.6</td><td> $57.6_{-1.5}$ </td><td>51.1</td><td>53.6</td><td>26.3</td><td>29.8</td><td> $40.2_{+1.4}$ </td></tr><tr><td>VideoKR (SFT + RL)</td><td>2026-05</td><td>16</td><td>56.6</td><td>66.6</td><td>57.0</td><td> $60.1_{+1.0}$ </td><td>52.6</td><td>59.2</td><td>27.3</td><td>37.7</td><td> $44.2_{+5.4}$ </td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>2025-02</td><td>128</td><td>65.1</td><td>66.3</td><td>60.9</td><td>64.1</td><td>51.1</td><td>55.7</td><td>28.1</td><td>32.7</td><td>41.9</td></tr><tr><td>VideoAuto-R1</td><td>2026-01</td><td>128</td><td>66.8</td><td>70.2</td><td>59.7</td><td> $65.6_{+1.5}$ </td><td>52.1</td><td>55.7</td><td>32.7</td><td>36.5</td><td> $44.3_{+2.4}$ </td></tr><tr><td>VideoKR (SFT + RL)</td><td>2026-05</td><td>128</td><td>66.4</td><td>68.9</td><td>61.3</td><td> $65.5_{+1.4}$ </td><td>52.2</td><td>60.5</td><td>32.5</td><td>41.2</td><td> $46.6_{+4.7}$ </td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>2025-10</td><td>128</td><td>68.2</td><td>67.9</td><td>61.6</td><td>65.9</td><td>61.8</td><td>59.6</td><td>33.4</td><td>39.0</td><td>48.5</td></tr><tr><td>OneThinker</td><td>2025-12</td><td>128</td><td>65.8</td><td>69.3</td><td>61.4</td><td> $65.5_{-0.4}$ </td><td>62.9</td><td>61.6</td><td>33.8</td><td>38.3</td><td> $49.2_{+0.7}$ </td></tr><tr><td>VideoAuto-R1</td><td>2026-01</td><td>128</td><td>68.7</td><td>68.8</td><td>58.8</td><td> $65.4_{-0.5}$ </td><td>63.1</td><td>59.6</td><td>32.7</td><td>43.8</td><td> $49.8_{+1.3}$ </td></tr><tr><td>Qwen3-VL-8B-Thinking</td><td>2025-10</td><td>128</td><td>67.6</td><td>68.0</td><td>60.0</td><td> $65.2_{-0.7}$ </td><td>64.9</td><td>60.5</td><td>33.0</td><td>41.5</td><td> $50.0_{+1.5}$ </td></tr><tr><td>VideoKR (SFT)</td><td>2026-05</td><td>128</td><td>64.8</td><td>63.6</td><td>58.5</td><td> $62.3_{-3.6}$ </td><td>61.7</td><td>63.0</td><td>28.3</td><td>43.6</td><td> $49.2_{+0.7}$ </td></tr><tr><td>VideoKR (zero RL)</td><td>2026-05</td><td>128</td><td>67.4</td><td>65.5</td><td>60.0</td><td> $64.3_{-1.6}$ </td><td>61.9</td><td>63.5</td><td>32.5</td><td>44.6</td><td> $50.6_{+2.1}$ </td></tr><tr><td>VideoKR (SFT + RL)</td><td>2026-05</td><td>128</td><td>67.8</td><td>67.0</td><td>61.5</td><td> $65.4_{-0.5}$ </td><td>63.0</td><td>64.8</td><td>32.8</td><td>45.3</td><td> $51.5_{+3.0}$ </td></tr></table>

SciVideoBench (Deng et al., 2025), and VideoKR-Eval, which focus on domain-specific, expert-level reasoning.

Reproducibility Challenges in Prior Post-Training Work. We observe substantial cross-paper inconsistencies in reported results, particularly for base models used as posttraining starting points. Based on careful follow-up experiments, we attribute these discrepancies primarily to prompt misalignment: base models are sometimes evaluated under prompt conditions that are misaligned with their intended inference mode. For example, Qwen2.5-VL-Instruct is not a “reasoning” model, yet some papers evaluate it with elaborate self-reflection and forced reasoning-trace instructions designed for post-trained ‘reasoning” variants.

Standardizing Evaluation for Fair Model Comparisons. To ensure fair and reproducible comparisons, for each model, we use the official prompt released by the original paper whenever available; otherwise, we adopt the default prompt templates from LMMs-Eval (Li et al., 2024a). For input frames, we follow the model-recommended inference configuration whenever it is specified by the model release. We run each model three times with independent sampling and report the mean. All evaluations are performed using the LMMs-Eval framework (Li et al., 2024a). Full evaluation details (e.g., model prompts and inference parameters) are provided in Appendix C.2.

# 6. Experiment Results and Analysis

We next discuss our main findings and ablation analysis.

# 6.1. Main Results

Table 3 presents the main experimental results. Posttraining on VideoKR consistently improves Qwen2.5- VL-7B-Instruct across all evaluated benchmarks, and yields clear gains for Qwen3-VL-8B-Instruct especially on knowledge-intensive video reasoning benchmarks. The gains are most pronounced on knowledge-intensive tasks: SFT followed by RL on VideoKR raises the knowledgeintensive average of Qwen2.5-VL-7B from 41.9 to 46.6 (+4.7) and of Qwen3-VL-8B from 48.5 to 51.5 (+3.0), with the largest per-dataset improvements on MMVU and VideoKR-Eval (e.g., +4.8 and +8.5 points for Qwen2.5-VL-7B). Notably, the post-trained Qwen3-VL-8B attains the best knowledge-intensive average among 7/8B-scale models (51.5, versus 50.0 for the strongest competing model, Qwen3-VL-8B-Thinking). These results underscore the value of our training data, which tightly integrates domain knowledge, visual grounding, and structured reasoning. Starting from an SFT-initialized model, subsequent RL training consistently delivers higher performance than the SFTonly baseline, highlighting that combining SFT with RL is important for fully leveraging the strengths of VideoKR data. Moreover, the RL-only variant also generally outperforms the SFT-only model, indicating that RL alone can induce stronger generalizable reasoning abilities.

![](images/5bf639adff5e6de9f46c6a9bb9675c474c0b0961dedc779034bbaf7e070e6a4e.jpg)

<details>
<summary>line</summary>

| Input Frames | Qwen2.5-VL-7B | Qwen2.5-VL-7B (SFT+RL) |
| ------------ | ------------- | ---------------------- |
| 16           | 59.1          | 60.1                   |
| 32           | 61.8          | 62.2                   |
| 64           | 62.8          | 63.8                   |
| 128          | 64.1          | 65.5                   |
</details>

(a) General Reasoning

![](images/f17ba814082b6536996ad89881d0b6eb1eb90ea710f36991d1af706462a0626d.jpg)

<details>
<summary>line</summary>

| Input Frames | Qwen2.5-VL-7B | Qwen2.5-VL-7B (SFT+RL) |
| ------------ | ------------- | ---------------------- |
| 16           | 38.8          | 44.2                   |
| 32           | 40.5          | 44.3                   |
| 64           | 41.0          | 45.8                   |
| 128          | 41.9          | 46.6                   |
</details>

(b) Knowledge-intensive Reasoning   
Figure 3. Inference-time frame scaling results on general and knowledge-intensive video reasoning benchmarks. The figure shows category-wise average accuracies for Qwen2.5-VL-7B-Instruct and its VideoKR post-trained variant (SFT+RL) under different input frame budgets. Appendix D.1 provides the full per-benchmark results for post-trained Qwen2.5-VL-7B-Instruct and Qwen3-VL-8B-Instruct models.

# 6.2. Case Study

To better understand how post-training on VideoKR changes model behavior, we randomly sample 100 examples from VideoKR-Eval and compare outputs from different models. Our case study shows that the post-trained Qwen3-VL-8B model can integrate visual evidence with relevant domain knowledge to perform complex reasoning. It also exhibits “aha-moment” reasoning patterns indicative of deeper understanding. Examples are shown in Appendix D.3.

# 6.3. Analysis of Inference-Time Frame Scaling

To analyze how input frame count affects performance, we evaluate our post-trained model, which was trained with 128 frames, under varying numbers of frames at inference. Specifically, we test 16, 32, 64, and 128 frames while keeping all other inference settings fixed. As shown in Figure 3, increasing the number of input frames consistently improves performance for both the base model and our post-trained model. For example, on general video reasoning benchmarks, Qwen2.5-VL-7B (SFT+RL) improves from 60.1% at 16 frames to 65.5% at 128 frames. On knowledge-intensive video reasoning benchmarks, it further improves from 44.2% to 46.6%. These results suggest that our model benefits from richer visual and temporal evidence at inference time, and that the gains from VideoKR remain consistent across different frame budgets.

In the following subsections, we conduct controlled SFT and RL ablations on VideoKR to provide insights for future work. Unless otherwise specified, all experiments use Qwen2.5-VL-7B-Instruct as the base model and are trained and evaluated with 128 input frames.

# 6.4. Ablations on VideoKR-SFT-201K

Using VideoKR-SFT-201K, we ablate two design choices, the skill composition of the training examples and the use of CoT supervision, to quantify their individual contributions.

Skill-Oriented Data Composition. To assess the contribution of each skill component, we fine-tune models on cumulative subsets of our skill-oriented data. Specifically, we construct three 80K-example variants from VideoKR-SFT-201K: (1) VIDR only; (2) a balanced mixture of VIDR and KNOWVID (1:1); and (3) a randomly sampled subset from the full VideoKR-SFT-201K (i.e., VIDR +KNOWVID +KNOWVIDR, we reuse examples in the previous ablation). We fine-tune Qwen2.5-VL-7B-Instruct on each variant for one epoch with a batch size of 16. We observe that incorporating all three skill components yields the best knowledge-intensive performance: training on VIDR alone obtains 41.4% on knowledge-intensive benchmarks. Adding Knowledge-Enhanced Perception (VIDR +KNOWVID) gives 41.3%, while incorporating Knowledge-Intensive Reasoning (VIDR +KNOWVID +KNOWVIDR) further improves performance to 42.4%. The same trend holds on VideoKR-Eval, where accuracy rises monotonically from 35.3 (VIDR) to 35.9 (VIDR +KNOWVID) to 36.8 (VIDR +KNOWVID +KNOWVIDR). These results indicate that combining domain knowledge and complex reasoning supervision is crucial.

CoT vs. Direct Output. To validate the necessity of explicit CoT rationales in SFT, we randomly sample 80K examples from VideoKR-SFT-201K and create two variants: one with CoT rationales and one without. We fine-tune Qwen2.5-VL-7B-Instruct on each variant for one epoch with a batch size of 16. As illustrated in Table 4, the CoT-trained model improves over the Direct Output baseline on knowledge-intensive reasoning, lifting the average from 39.4% to 42.4% (a 3.0-point gain), underscoring the importance of high-quality CoT supervision for advanced knowledge-intensive video reasoning.

Table 4. Ablation studies on post-training data. All experiments use Qwen2.5-VL-7B-Instruct as the base model, with 128 input frames. The complete results are provided in Appendix D.2. 

<table><tr><td rowspan="2">Ablation Setting</td><td>General</td><td colspan="2">Knowledge-Intensive</td></tr><tr><td>Average</td><td>VideoKR-Eval</td><td>Average</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>64.1</td><td>32.7</td><td>41.9</td></tr><tr><td colspan="4">Skill-Oriented Data Composition (SFT, 80K examples, one epoch)</td></tr><tr><td>VIDR</td><td> $58.0_{-6.1}$ </td><td> $35.3_{+2.6}$ </td><td> $41.4_{-0.5}$ </td></tr><tr><td>VIDR + KNOWVID</td><td> $58.4_{-5.7}$ </td><td> $35.9_{+3.2}$ </td><td> $41.3_{-0.6}$ </td></tr><tr><td>VIDR + KNOWVID + KNOWVIDR</td><td> $58.3_{-5.8}$ </td><td> $36.8_{+4.1}$ </td><td> $42.4_{+0.5}$ </td></tr><tr><td colspan="4">CoT Supervision Format (SFT, 80K examples, one epoch)</td></tr><tr><td>Direct Output</td><td> $61.4_{-2.7}$ </td><td> $35.9_{+3.2}$ </td><td> $39.4_{-2.5}$ </td></tr><tr><td>Chain-of-Thought</td><td> $58.3_{-5.8}$ </td><td> $36.8_{+4.1}$ </td><td> $42.4_{+0.5}$ </td></tr><tr><td colspan="4">Comparison with Other SFT Corpora (SFT, 80K examples, one epoch)</td></tr><tr><td>Video-R1-CoT-165k</td><td> $57.3_{-6.8}$ </td><td> $27.5_{-5.2}$ </td><td> $36.2_{-5.7}$ </td></tr><tr><td>OneThinker-SFT-340k</td><td> $60.5_{-3.6}$ </td><td> $29.7_{-3.0}$ </td><td> $38.3_{-3.6}$ </td></tr><tr><td>VideoRFT-CoT-102K</td><td> $59.6_{-4.5}$ </td><td> $32.1_{-0.6}$ </td><td> $38.4_{-3.5}$ </td></tr><tr><td>VideoKR-SFT-201K (Ours)</td><td> $58.3_{-5.8}$ </td><td> $36.8_{+4.1}$ </td><td> $42.4_{+0.5}$ </td></tr><tr><td colspan="4">Comparison with Other RL Corpora (GRPO, 50K examples, one epoch)</td></tr><tr><td>Video-R1-260k</td><td> $63.7_{-0.4}$ </td><td> $33.1_{+0.4}$ </td><td> $41.6_{-0.3}$ </td></tr><tr><td>OneThinker-600k</td><td> $60.3_{-3.8}$ </td><td> $33.2_{+0.5}$ </td><td> $42.3_{+0.4}$ </td></tr><tr><td>VideoRFT-RL-310K</td><td> $63.8_{-0.3}$ </td><td> $33.5_{+0.8}$ </td><td> $42.3_{+0.4}$ </td></tr><tr><td>VideoAuto-R1-83K</td><td> $62.8_{-1.3}$ </td><td> $33.3_{+0.6}$ </td><td> $42.7_{+0.8}$ </td></tr><tr><td>VideoKR-RL-114K (Ours)</td><td> $61.7_{-2.4}$ </td><td> $34.5_{+1.8}$ </td><td> $43.0_{+1.1}$ </td></tr></table>

# 6.5. VideoKR vs Prior Post-Training Corpus

We next conduct a comprehensive comparison of VideoKR with prior open-source post-training corpora, under both SFT and zero-RL settings. For SFT, we randomly sample 80K examples from each SFT corpus and fine-tune Qwen2.5-VL-7B-Instruct on each variant for one epoch with a batch size of 16. For zero-RL, we randomly sample 50K QA examples from each RL corpus and train Qwen2.5- VL-7B-Instruct on each variant for one epoch with a batch size of 16; we exclude non-QA tasks since they require task-specific reward functions that cannot be unified under our standard GRPO training pipeline.

Main Findings. As illustrated in Table 4, under SFT, the model trained on the VideoKR-SFT subset reaches a 42.4 knowledge-intensive average and is the only corpus to surpass the base model (41.9), whereas prior corpora such as Video-R1 and VideoRFT lower it to 36.2 and 38.4, respectively. Under zero-RL, VideoKR with VideoKR-RL-114K achieves the strongest gain (43.0, +1.1 over the base model, ahead of the next-best VideoAuto-R1 at 42.7), indicating that high-quality data is key to maximizing post-training benefits for advanced video reasoning.

Training-Data Difficulty Analysis. To diagnose why prior corpora provide limited improvements relative to VideoKR, we analyze training-data difficulty with respect to the base models. Concretely, we randomly sample 3,000 video QA examples from each corpus and measure the zeroshot accuracy of Qwen2.5-VL-7B and Qwen3-VL-8B in the 128-frame setting. As shown in Table 5, all evaluated models attain high accuracy on prior corpora (e.g., Qwen3- VL-8B scores between 49.1% and 57.1%), suggesting these datasets are effectively saturated for current frontier base models and thus offer weak learning signals. In contrast, accuracy on VideoKR remains lower (42.3% for the same model), indicating a more challenging distribution that better supports continued capability gains during post-training.

Table 5. Accuracy of Qwen2.5/3-VL models on 3,000 randomly sampled QA examples from various post-training corpora. 

<table><tr><td>Model</td><td>Video-R1</td><td>VideoRFT</td><td>OneThinker</td><td>VidAuto-R1</td><td>VideoKR</td></tr><tr><td>2.5-VL-7B-Inst.</td><td>55.3</td><td>47.8</td><td>45.8</td><td>57.1</td><td>39.2</td></tr><tr><td>3-VL-8B-Inst.</td><td>57.1</td><td>51.1</td><td>49.1</td><td>54.5</td><td>42.3</td></tr><tr><td>3-VL-8B-Think</td><td>59.0</td><td>52.3</td><td>49.3</td><td>54.3</td><td>43.5</td></tr></table>

# 7. Conclusion

This work offers a corpus-centric perspective on posttraining foundation models for advanced video reasoning. Instead of viewing visual perception, domain knowledge, and advanced reasoning as loosely linked elements, we show that integrating structured domain concepts with visually grounded examples yields stronger reasoning performance, without relying on sophisticated RL reward engineering. Extensive experiments and analyses confirm that post-training on the VideoKR dataset produces strong improvements, especially on knowledge-intensive video reasoning.

# Impact Statement

All videos in VideoKR are licensed under CC license, which enables reusers to distribute, remix, adapt, and build upon the material in any medium or format, so long as attribution is given to the creator. By restricting our dataset to CC-licensed videos, we ensure legal reusability and clear provenance, a critical aspect that has been ambiguous in prior training corpora. For human expert annotation and validation, we compensated annotators at an average rate of \$13 USD per hour, which exceeds the prevailing hourly rates for comparable local work. All expert annotators provided informed consent to participate and explicitly authorized the public release and redistribution of their annotations as part of the resulting dataset and accompanying materials. The VideoKR and VideoKR-Eval data construction incurs approximately 70.4K US dollars in model inference costs.

# Acknowledgments

We thank Dr. Yu Rong for valuable suggestions on the posttraining experiment design and paper writeup improvement.

# References

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.   
Chen, G., Li, Z., Wang, S., Jiang, J., Liu, Y., Lu, L., Huang, D.-A., Byeon, W., Le, M., Rintamaki, T., Poon, T., Ehrlich, M., Rintamaki, T., Poon, T., Lu, T., Wang, L., Catanzaro, B., Kautz, J., Tao, A., Yu, Z., and Liu, G. Eagle 2.5: Boosting long-context post-training for frontier vision-language models, 2025a. URL https: //arxiv.org/abs/2504.15271.   
Chen, R., Fan, Z., Luo, T., Zou, H., Feng, Z., Xie, G., Zhang, H., Wang, Z., Liu, Z., and Zhang, H. Datasets and recipes for video temporal grounding via reinforcement learning, 2025b. URL https://arxiv.org/abs/ 2507.18100.   
Chen, Y., Huang, W., Shi, B., Hu, Q., Ye, H., Zhu, L., Liu, Z., Molchanov, P., Kautz, J., QI, X., Liu, S., Yin, H., Lu, Y., and Han, S. Scaling RL to long videos. 2026. URL https://openreview.net/forum? id=TxedB8hI5O.   
Deng, A., Yang, T., Yu, S., Spencer, L., Bansal, M., Chen, C., Yeung-Levy, S., and Wang, X. Scivideobench: Benchmarking scientific video reasoning in large multimodal models, 2025. URL https://arxiv.org/abs/ 2510.08559.   
Di, S. and Xie, W. Grounded question-answering in long egocentric videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12934–12943, 2024.   
Feng, K., Gong, K., Li, B., Guo, Z., Wang, Y., Peng, T., Wu, J., Zhang, X., Wang, B., and Yue, X. Video-r1: Reinforcing video reasoning in mllms. arXiv preprint arXiv:2503.21776, 2025a.   
Feng, K., Zhang, M., Li, H., Fan, K., Chen, S., Jiang, Y., Zheng, D., Sun, P., Zhang, Y., Sun, H., et al. Onethinker: All-in-one reasoning model for image and video. arXiv preprint arXiv:2512.03043, 2025b.   
Fu, C., Dai, Y., Luo, Y., Li, L., Ren, S., Zhang, R., Wang, Z., Zhou, C., Shen, Y., Zhang, M., Chen, P., Li, Y., Lin, S., Zhao, S., Li, K., Xu, T., Zheng, X., Chen, E., Shan, C., He, R., and Sun, X. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis, 2025. URL https: //arxiv.org/abs/2405.21075.

Guo, D., Yang, D., Zhang, H., Song, J., Zhang, R., Xu, R., Zhu, Q., Ma, S., Wang, P., Bi, X., et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.   
Han, S., Huang, W., Shi, H., Zhuo, L., Su, X., Zhang, S., Zhou, X., Qi, X., Liao, Y., and Liu, S. Videoespresso: A large-scale chain-of-thought dataset for fine-grained video reasoning via core frame selection. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26181–26191, 2025.   
Hu, K., Wu, P., Pu, F., Xiao, W., Zhang, Y., Yue, X., Li, B., and Liu, Z. Video-mmmu: Evaluating knowledge acquisition from multi-discipline professional videos, 2025. URL https://arxiv.org/abs/2501.13826.   
Li, B., Zhang, P., Zhang, K., Pu, F., Du, X., Dong, Y., Liu, H., Zhang, Y., Zhang, G., Li, C., and Liu, Z. Lmmseval: Accelerating the development of large multimodal models, March 2024a. URL https://github.com/ EvolvingLMMs-Lab/lmms-eval.   
Li, K., Wang, Y., He, Y., Li, Y., Wang, Y., Liu, Y., Wang, Z., Xu, J., Chen, G., Luo, P., Wang, L., and Qiao, Y. Mvbench: A comprehensive multi-modal video understanding benchmark, 2024b. URL https://arxiv. org/abs/2311.17005.   
Li, X., Yan, Z., Meng, D., Dong, L., Zeng, X., He, Y., Wang, Y., Qiao, Y., Wang, Y., and Wang, L. Videochat-r1: Enhancing spatio-temporal perception via reinforcement fine-tuning. arXiv preprint arXiv:2504.06958, 2025a.   
Li, Y., Chen, X., Hu, B., Wang, L., Shi, H., and Zhang, M. Videovista: A versatile benchmark for video understanding and reasoning, 2024c. URL https://arxiv. org/abs/2406.11303.   
Li, Y., Jiang, S., Hu, B., Wang, L., Zhong, W., Luo, W., Ma, L., and Zhang, M. Uni-moe: Scaling unified multimodal llms with mixture of experts, 2024d. URL https:// arxiv.org/abs/2405.11273.   
Li, Y., Chen, X., Li, Z., Liu, Z., Wang, L., Luo, W., Hu, B., and Zhang, M. Veripo: Cultivating long reasoning in video-llms via verifier-gudied iterative policy optimization, 2025b. URL https://arxiv.org/abs/ 2505.19000.   
Li, Y., Liu, Z., Li, Z., Zhang, X., Xu, Z., Chen, X., Shi, H., Jiang, S., Wang, X., Wang, J., Huang, S., Zhao, X., Jiang, B., Hong, L., Wang, L., Tian, Z., Huai, B., Luo, W., Luo, W., Zhang, Z., Hu, B., and Zhang, M. Perception, reason, think, and plan: A survey on large multimodal reasoning models, 2025c. URL https://arxiv.org/abs/ 2505.04921.

Li, Y., Liu, Z., Hu, B., Wang, W., Ding, Y., Cao, X., and Zhang, M. Vision enhancing llms: Empowering multimodal knowledge storage and sharing in llms. IEEE Transactions on Image Processing, 35:858–871, 2026. doi: 10.1109/TIP.2025.3649356.   
Liu, S., Zhuge, M., Zhao, C., Chen, J., Wu, L., Liu, Z., Zhu, C., Cai, Z., Zhou, C., Liu, H., Chang, E., Suri, S., Xu, H., Qian, Q., Wen, W., Varadarajan, B., Liu, Z., Xu, H., Bordes, F., Krishnamoorthi, R., Ghanem, B., Chandra, V., and Xiong, Y. Videoauto-r1: Video auto reasoning via thinking once, answering twice. 2026a. URL https://arxiv.org/abs/2601.05175.   
Liu, Y., Ouyang, K., Wu, H., Liu, Y., Sui, L., Li, X., Zhong, Y., Charles, Y., Zhou, X., and Sun, X. Videoreasonbench: Can mllms perform vision-centric complex video reasoning?, 2026b. URL https://arxiv.org/abs/ 2505.23359.   
Meng, J., Li, X., Wang, H., Tan, Y., Zhang, T., Kong, L., Tong, Y., Wang, A., Teng, Z., Wang, Y., et al. Open-o3- video: Grounded video reasoning with explicit spatiotemporal evidence. arXiv preprint arXiv:2510.20579, 2025.   
Munasinghe, S., Gani, H., Zhu, W., Cao, J., Xing, E., Khan, F. S., and Khan, S. Videoglamm: A large multimodal model for pixel-level visual grounding in videos, 2025. URL https://arxiv.org/abs/2411.04923.   
Ouyang, K., Liu, Y., Wu, H., Liu, Y., Zhou, H., Zhou, J., Meng, F., and Sun, X. Spacer: Reinforcing mllms in video spatial reasoning, 2025. URL https://arxiv. org/abs/2504.01805.   
Plizzari, C., Tonioni, A., Xian, Y., Kulshrestha, A., and Tombari, F. Omnia de egotempo: Benchmarking temporal understanding of multi-modal llms in egocentric videos. 2025. URL https://arxiv.org/abs/ 2503.13646.   
Ren, W., Ma, W., Yang, H., Wei, C., Zhang, G., and Chen, W. Vamba: Understanding hour-long videos with hybrid mamba-transformers. arXiv preprint arXiv:2503.11579, 2025.   
Saravanan, D., Gupta, V., Singh, D., Khan, Z., Gandhi, V., and Tapaswi, M. Velociti: Benchmarking video-language compositional reasoning with strict entailment. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18914–18924, 2025.   
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.

Sheng, G., Zhang, C., Ye, Z., Wu, X., Zhang, W., Zhang, R., Peng, Y., Lin, H., and Wu, C. Hybridflow: A flexible and efficient rlhf framework. arXiv preprint arXiv: 2409.19256, 2024.   
Shu, Y., Liu, Z., Zhang, P., Qin, M., Zhou, J., Liang, Z., Huang, T., and Zhao, B. Video-xl: Extra-long vision language model for hour-scale video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 26160–26169, 2025.   
Song, E., Chai, W., Xu, W., Xie, J., Liu, Y., and Wang, G. Video-mmlu: A massive multi-discipline lecture understanding benchmark. arXiv preprint arXiv:2504.14693, 2025.   
Tang, H., Cao, M., Liu, R., Liang, X., Li, L., Li, G., and Liang, X. Video spatial reasoning with object-centric 3d rollout. 2025a. URL https://arxiv.org/abs/ 2511.13190.   
Tang, Y. Y., Shimada, D., Hua, H., Huang, C., Bi, J., Feris, R., and Xu, C. Video-r4: Reinforcing text-rich video reasoning with visual rumination. 2025b. URL https: //arxiv.org/abs/2511.17490.   
Team, K., Du, A., Gao, B., Xing, B., Jiang, C., Chen, C., Li, C., Xiao, C., Du, C., Liao, C., et al. Kimi k1.5: Scaling reinforcement learning with llms. arXiv preprint arXiv:2501.12599, 2025.   
Wang, Q., Yu, Y., Yuan, Y., Mao, R., and Zhou, T. Videorft: Incentivizing video reasoning capability in mllms via reinforced fine-tuning. arXiv preprint arXiv:2505.12434, 2025a.   
Wang, S., Jin, J., Wang, X., Song, L., Fu, R., Wang, H., Ge, Z., Lu, Y., and Cheng, X. Video-thinker: Sparking” thinking with videos” via reinforcement learning. arXiv preprint arXiv:2510.23473, 2025b.   
Wang, Y., Li, X., Yan, Z., He, Y., Yu, J., Zeng, X., Wang, C., Ma, C., Huang, H., Gao, J., Dou, M., Chen, K., Wang, W., Qiao, Y., Wang, Y., and Wang, L. Internvideo2.5: Empowering video mllms with long and rich context modeling, 2025c. URL https://arxiv.org/abs/ 2501.12386.   
Wang, Y., Wang, Z., Xu, B., Du, Y., Lin, K., Xiao, Z., Yue, Z., Ju, J., Zhang, L., Yang, D., et al. Time-r1: Posttraining large vision language model for temporal video grounding. arXiv preprint arXiv:2503.13377, 2025d.   
Wu, B., Yu, S., Chen, Z., Tenenbaum, J. B., and Gan, C. STAR: A benchmark for situated reasoning in realworld videos. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021. URL https://openreview. net/forum?id=EfgNF5-ZAjM.

Wu, H., Li, D., Chen, B., and Li, J. Longvideobench: A benchmark for long-context interleaved video-language understanding. In The Thirty-eight Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2024. URL https://openreview. net/forum?id=3G1ZDXOI4f.   
Wu, R., Ma, X., Ci, H., Fan, Y., Wang, Y., Zhao, H., Li, Q., and Wang, Y. Longvitu: Instruction tuning for long-form video understanding. arXiv preprint arXiv:2501.05037, 2025.   
Xu, Y., Wu, Y., Yu, J., Yan, Z., Jiang, T., He, Y., Zhao, Q., Chen, K., Qiao, Y., Wang, L., Okumura, M., and Wang, Y. Expvid: A benchmark for experiment video understanding & reasoning, 2025. URL https://arxiv.org/ abs/2510.11606.   
Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.   
Yang, J., Yang, S., Gupta, A. W., Han, R., Fei-Fei, L., and Xie, S. Thinking in space: How multimodal large language models see, remember, and recall spaces, 2025b. URL https://arxiv.org/abs/2412.14171.   
Zhang, B., Li, K., Cheng, Z., Hu, Z., Yuan, Y., Chen, G., Leng, S., Jiang, Y., Zhang, H., Li, X., Jin, P., Zhang, W., Wang, F., Bing, L., and Zhao, D. Videollama 3: Frontier multimodal foundation models for image and video understanding. 2025a. URL https://arxiv. org/abs/2501.13106.   
Zhang, S., Hao, X., Tang, Y., Zhang, L., Wang, P., Wang, Z., Ma, H., and Zhang, S. Video-cot: A comprehensive dataset for spatiotemporal understanding of videos based on chain-of-thought, 2025b. URL https://arxiv. org/abs/2506.08817.   
Zhang, Y., Wu, J., Li, W., Li, B., MA, Z., Liu, Z., and Li, C. LLaVA-video: Video instruction tuning with synthetic data. Transactions on Machine Learning Research, 2025c. ISSN 2835-8856. URL https:// openreview.net/forum?id=EElFGvt39K.   
Zhao, F., Zhang, L., Shi, D., Gao, Y., Ye, C., Cai, Y., Gao, J., and Yan, D. Videoperceiver: Enhancing fine-grained temporal perception in video multimodal large language models. arXiv preprint arXiv:2511.18823, 2025a.   
Zhao, Y., Xie, L., Zhang, H., Gan, G., Long, Y., Hu, Z., Hu, T., Chen, W., Li, C., Song, J., Xu, Z., Wang, C., Pan, W., Shangguan, Z., Tang, X., Liang, Z., Liu, Y., Zhao, C., and Cohan, A. Mmvu: Measuring expertlevel multi-discipline video understanding, 2025b. URL https://arxiv.org/abs/2501.12380.

Zheng, Y., Zhang, R., Zhang, J., Ye, Y., Luo, Z., Feng, Z., and Ma, Y. Llamafactory: Unified efficient fine-tuning of 100+ language models, 2024. URL https://arxiv. org/abs/2403.13372.

Zhou, J., Shu, Y., Zhao, B., Wu, B., Liang, Z., Xiao, S., Qin, M., Yang, X., Xiong, Y., Zhang, B., Huang, T., and Liu, Z. Mlvu: Benchmarking multi-task long video understanding. 2025. URL https://arxiv.org/ abs/2406.04264.

Zohar, O., Wang, X., Bitton, Y., Szpektor, I., and Yeung-Levy, S. Video-star: Self-training enables video instruction tuning with any supervision, 2024. URL https: //arxiv.org/abs/2407.06189.

Zohar, O., Wang, X., Dubois, Y., Mehta, N., Xiao, T., Hansen-Estruch, P., Yu, L., Wang, X., Juefei-Xu, F., Zhang, N., et al. Apollo: An exploration of video understanding in large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 18891–18901, 2025.

# A. VideoKR Data Construction

# A.1. Domain Knowledge Bank Construction

Table 6. Complete subject list by major disciplines. Columns list subfields under each of the four major disciplines. 

<table><tr><td>Natural Sciences (20)</td><td>Engineering (20)</td><td>Healthcare (18)</td><td>Humanities &amp; Social Sciences (24)</td></tr><tr><td>Chemistry</td><td>Electrical Engineering</td><td>Medicine</td><td>History</td></tr><tr><td>Physics</td><td>Computer Science</td><td>Public Health</td><td>Psychology</td></tr><tr><td>Biology</td><td>Materials Science &amp; Engineering</td><td>Pharmacy</td><td>Arts</td></tr><tr><td>Astrophysics &amp; Astronomy</td><td>Mechanical Engineering</td><td>Dentistry</td><td>Education</td></tr><tr><td>Earth Science</td><td>Civil &amp; Environmental Engineering</td><td>Nursing</td><td>Sociology</td></tr><tr><td>Mathematics</td><td>Chemical Engineering</td><td>Biomedical Sciences</td><td>Philosophy</td></tr><tr><td>Statistics &amp; Data Science</td><td>Biomedical Engineering</td><td>Medical Laboratory Science</td><td>Linguistics</td></tr><tr><td>Environmental Science</td><td>Aerospace Engineering</td><td>Nutrition &amp; Dietetics</td><td>Anthropology</td></tr><tr><td>Ecology &amp; Evolutionary Biology</td><td>Industrial &amp; Operations Research</td><td>Physiotherapy</td><td>Archaeology</td></tr><tr><td>Microbiology &amp; Immunology</td><td>Systems Engineering</td><td>Occupational Therapy</td><td>Political Science</td></tr><tr><td>Biochemistry &amp; Molecular Biology</td><td>Nuclear Engineering</td><td>Speech &amp; Language Therapy</td><td>International Relations</td></tr><tr><td>Biophysics</td><td>Energy Engineering</td><td>Radiography / Imaging Sciences</td><td>Economics</td></tr><tr><td>Neuroscience</td><td>Mechatronics &amp; Robotics</td><td>Health / Biomedical Informatics</td><td>Law</td></tr><tr><td>Genetics &amp; Genomics</td><td>Software Engineering</td><td>Epidemiology &amp; Biostatistics</td><td>Geography</td></tr><tr><td>Cell &amp; Developmental Biology</td><td>AI Engineering</td><td>Global Health</td><td>Communication &amp; Media Studies</td></tr><tr><td>Marine Science</td><td>Computer Engineering</td><td>Veterinary Medicine</td><td>Literature &amp; Comparative Literature</td></tr><tr><td>Atmospheric Science</td><td>Communications Engineering</td><td>Optometry / Vision Science</td><td>Modern Languages &amp; Cultures</td></tr><tr><td>Geology &amp; Geophysics</td><td>Control &amp; Automation</td><td>Health Policy &amp; Management</td><td>Theology &amp; Religious Studies</td></tr><tr><td>Paleontology</td><td>Structural Engineering</td><td></td><td>Business Administration</td></tr><tr><td>Scientific Computing</td><td>Geotechnical Engineering</td><td></td><td>Finance</td></tr><tr><td></td><td></td><td></td><td>Accounting</td></tr><tr><td></td><td></td><td></td><td>Architecture &amp; Urban Planning</td></tr><tr><td></td><td></td><td></td><td>Public Policy &amp; Administration</td></tr><tr><td></td><td></td><td></td><td>Gender &amp; Sexuality Studies</td></tr></table>

Based on a manual review of undergraduate curricula from leading universities worldwide, we identified 82 representative subjects spanning four major disciplines. Table 6 organizes these subjects across Natural Sciences, Healthcare, Humanities and Social Sciences, and Engineering, forming the top-level index of our four-layer knowledge base of subject, course, lecture, and knowledge point, and enabling broad cross-domain coverage and balanced sampling.

# A.2. Annotator Information

Table 7. Biographies of 34 annotators involved in the VideoKR construction pipeline. The table details their participation in: Know. Bank (Domain Knowledge Bank Construction), Seed Ex. (Seed Example Curation), Model Val. (Human-Validated Model Selection), Quality (Manual Quality Assessment), and Eval Bench. (VideoKR-Eval Construction). 

<table><tr><td rowspan="2">ID</td><td rowspan="2">Year</td><td rowspan="2">Major</td><td rowspan="2">Assigned Discipline</td><td colspan="5">Annotation Tasks</td></tr><tr><td>Know. Bank</td><td>Seed Ex.</td><td>Model Val.</td><td>Quality</td><td>Eval Bench.</td></tr><tr><td>1</td><td>3rd yr PhD</td><td>Electrical Eng.</td><td>Engineering</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>2</td><td>3rd yr PhD</td><td>Mechanical Eng.</td><td>Engineering</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>3</td><td>1st yr PhD</td><td>Computer Science</td><td>Engineering</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>4</td><td>2nd yr PhD</td><td>Civil Engineering</td><td>Engineering</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>5</td><td>2nd yr Master</td><td>Chemical Eng.</td><td>Engineering</td><td>✓</td><td>✓</td><td></td><td></td><td></td></tr><tr><td>6</td><td>1st yr Master</td><td>Materials Science</td><td>Engineering</td><td></td><td>✓</td><td>✓</td><td></td><td>✓</td></tr><tr><td>7</td><td>1st yr Master</td><td>Aerospace Eng.</td><td>Engineering</td><td></td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>8</td><td>2nd yr Master</td><td>Biomedical Eng.</td><td>Engineering</td><td>✓</td><td>✓</td><td></td><td></td><td>✓</td></tr><tr><td>9</td><td>2nd yr Master</td><td>Software Eng.</td><td>Engineering</td><td></td><td>✓</td><td></td><td></td><td></td></tr><tr><td>10</td><td>1st yr Master</td><td>Electronic Eng.</td><td>Engineering</td><td></td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>11</td><td>1st yr Master</td><td>Industrial Eng.</td><td>Engineering</td><td>✓</td><td>✓</td><td></td><td></td><td></td></tr><tr><td>12</td><td>3rd yr PhD</td><td>Physics</td><td>Natural Sciences</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>13</td><td>2nd yr PhD</td><td>Chemistry</td><td>Natural Sciences</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>14</td><td>2nd yr PhD</td><td>Biology</td><td>Natural Sciences</td><td></td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>15</td><td>1st yr PhD</td><td>Mathematics</td><td>Natural Sciences</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>16</td><td>2nd yr Master</td><td>Statistics</td><td>Natural Sciences</td><td>✓</td><td>✓</td><td></td><td></td><td>✓</td></tr><tr><td>17</td><td>1st yr Master</td><td>Earth Science</td><td>Natural Sciences</td><td></td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>18</td><td>1st yr Master</td><td>Astrophysics</td><td>Natural Sciences</td><td></td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>19</td><td>1st yr Master</td><td>Environmental Sci.</td><td>Natural Sciences</td><td>✓</td><td>✓</td><td></td><td></td><td></td></tr><tr><td>20</td><td>2nd yr Master</td><td>Geology</td><td>Natural Sciences</td><td>✓</td><td>✓</td><td></td><td></td><td></td></tr><tr><td>21</td><td>1st yr Master</td><td>Ecology</td><td>Natural Sciences</td><td></td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>22</td><td>2nd yr PhD</td><td>Economics</td><td>Humanities &amp; Social Sciences</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>23</td><td>2nd yr PhD</td><td>Psychology</td><td>Humanities &amp; Social Sciences</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>24</td><td>3rd yr PhD</td><td>Sociology</td><td>Humanities &amp; Social Sciences</td><td></td><td>✓</td><td></td><td></td><td>✓</td></tr><tr><td>25</td><td>1st yr Master</td><td>Political Science</td><td>Humanities &amp; Social Sciences</td><td>✓</td><td>✓</td><td></td><td></td><td></td></tr><tr><td>26</td><td>1st yr Master</td><td>Philosophy</td><td>Humanities &amp; Social Sciences</td><td>✓</td><td>✓</td><td>✓</td><td></td><td>✓</td></tr><tr><td>27</td><td>1st yr Master</td><td>History</td><td>Humanities &amp; Social Sciences</td><td>✓</td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>28</td><td>1st yr Master</td><td>Law</td><td>Humanities &amp; Social Sciences</td><td>✓</td><td>✓</td><td></td><td></td><td>✓</td></tr><tr><td>29</td><td>2nd yr Master</td><td>Linguistics</td><td>Humanities &amp; Social Sciences</td><td></td><td>✓</td><td></td><td></td><td>✓</td></tr><tr><td>30</td><td>1st yr Master</td><td>Education</td><td>Humanities &amp; Social Sciences</td><td></td><td>✓</td><td>✓</td><td></td><td></td></tr><tr><td>31</td><td>4th yr PhD</td><td>Public Health</td><td>Healthcare</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>32</td><td>2nd yr PhD</td><td>Clinical Medicine</td><td>Healthcare</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr><tr><td>33</td><td>2nd yr PhD</td><td>Dentistry</td><td>Healthcare</td><td></td><td>✓</td><td></td><td></td><td></td></tr><tr><td>34</td><td>2nd yr Master</td><td>Pharmacy</td><td>Healthcare</td><td>✓</td><td>✓</td><td></td><td></td><td>✓</td></tr></table>

# A.3. VideoKR Data Example

![](images/af2e6cfc7e31aebf5c55f93468596081192844a892d21a596476febf79e32884.jpg)

<details>
<summary>text_image</summary>

Ksp of Ca(OH)₂ and Mg(OH)₂
NCSOM Online
250
PH of Saturated Ca(OH)₂
NCSOM Online
10.10
3
4
5
6
7
8
NCSOM Online
OIL sensor
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/OFF
ON/Off
</details>

![](images/13ff7807fa4fbe31b783f14fe2ae86d8428291cc2793647cbed29ef6caf7d043.jpg)  
Q: What value is shown on the display at around 01:38?

Answer: −6.419

Reasoning: The user is asking about what value is displayed at approximately 1 minute and 38 seconds (98 seconds) into the video. Looking at the timestamp at around 98 seconds, I can see a Texas Instruments TI -83 Plus calculator display. The screen shows statistical calculation results, including: - "INTERCEPT= -6.419" - "SLOPE=11.824" The question asks for "the value" shown at this time, and given the correct answer provided is -6.419, this corresponds to the intercept value displayed on the calculator screen.

![](images/45a63ecb4d8e3c3da0d9e03687fe8c26b4083eade445016d540077bad8d48e86.jpg)  
Q: Which tested sample shows formation of an insoluble solid during measurement?

Answer: magnesium hydroxide sample

Reasoning: Let me analyze the video content around the relevant measurement mark. Looking at the sequence: At around 01:53, the text "PH of Saturated Ca(OH) ₂" appears,(...abbreviated...) The question asks about which sample forms an insoluble solid during measurement. However, I need to consider the chemistry involved:(...abbreviated...) The video is demonstrating the difference in solubility between these two hydroxides, and the formation of insoluble solid would be most notable with the magnesium hydroxide sample due to its much lower Ksp value.

![](images/bd82419445815b79b2d504d7e19311d91a3667c8bcb7c82769f0e67cf77dfb17.jpg)  
Q: A saturated solution of Mg(OH)₂ exhibits a pH value measured in the experiment. Calculate the Ksp of Mg(OH)₂ using the observed pH.

A. 1.0 × 10⁻¹²

B. 5.6 × 10⁻¹²

C. 1.8 × 10⁻¹¹

D. 4.0 × 10⁻¹²

Answer: A

Reasoning: To calculate the Ksp of Mg(OH) ₂, first extract the saturated solution's pH = 10.10 from the video(...abbreviated...). Calculate pOH = 14.00 - 10.10 = 3.90, then [OH⁻] = 10⁻³.⁹⁰ ≈ 1.2589 × 10⁻⁴ M(...abbreviated...). From Mg(OH)₂ ⇌ Mg²⁺ + 2OH⁻, [Mg²⁺] = [OH⁻]/2 ≈ 6.2945 × 10⁻⁵ M(...abbreviated...). Substitute into Ksp = [Mg²⁺][OH⁻]²: Ksp = (6.2945 × 10⁻⁵)(1.2589 × 10⁻⁴)² ≈ 9.98 × 10⁻¹³, rounding to 1.0 × 10⁻¹².

Figure 4. A VideoKR-SFT-201K example from the natural science domain. The reasoning process is presented in a concise and abbreviated form to improve readability.   
![](images/a42d4c8703c2489e7bef3f59b4037a6b3bfc974bd375c3629e1a93f4f9360c50.jpg)

<details>
<summary>text_image</summary>

Part II: Learn How to Use a Sound Measuring App
Part II: Noise Mapping
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
</details>

![](images/df26ca767c2cead47c0f4e4eff70a682941525ee2b88f31d28d20741a770a9dc.jpg)  
Q: What device is physically connected to the phone before measurements begin?

Answer: External speakers

Reasoning: I need to examine the video carefully to identify what device is physically connected to the phone before the measurement phase begins.(...abbreviated...)Then at around 194.6 seconds onwards, I can observe a table setup showing a phone positioned between two black external speakers. The speakers appear to be wired and connected to the phone with visible cables. This setup is shown clearly before the person begins walking around the room taking measurements (which occurs around 225.7 seconds).(...abbreviated...)The external speakers are part of the controlled setup for the noise mapping exercise - they likely serve as the sound source that the participants are trying to locate and map using the sound measuring app on their phone.

![](images/74e48c52c5753f4d7788f2baa0254f966f242f25365f6acba7873dcd62a572a2.jpg)  
Q: At the moment the handheld display shows its highest reading, which category applies to that spot on the diagram?

Answer: Red

Reasoning: Looking at the video timeline, I need to identify when the handheld SPL meter shows its highest reading and then determine which category that corresponds to on the noise map.(...abbreviated...)The highest reading on the handheld SPL meter is 79.1 dBA, recorded at around 101 and 109 seconds.(...abbreviated...)This peak reading corresponds to the red category on the noise map’s Green-Yellow -Orange-Red color legend, which is reserved for the loudest measured noise levels in the room.

![](images/82c3e2f18739fca6f131bf05dffffb815e55d6bef8953a8d7ca113702ee1d546.jpg)  
Q: During the SPL meter demonstration, at the moment when the current SPL value is displayed as 77.2 dBA, calculate the ratio of (maximum value - average value) to (average value - minimum value). Round to two decimal places.

A. 1.85

B. 2.03

C. 2.15

D. 3.03

Answer: B

Reasoning: Extract the three key values of the SPL meter shown in the video when the current SPL value is 77.2 dBA(...abbreviated...), the maximum value is 79.5 dBA, the minimum value is 76.5 dBA and the average value is 77.49 dBA. First calculate the numerator: 79.5 - 77.49 = 2.01, then calculate the denominator: 77.49 - 76.5 = 0.99(...abbreviated...). Divide the two results to get the ratio: 2.01 ÷ 0.99 ≈ 2.0303, and round it to two decimal places to get 2.03.

Figure 5. A VideoKR-SFT-201K example from the healthcare domain. The reasoning process is presented in a concise and abbreviated form to improve readability.

![](images/f2cb08eb47590bb0eb9dc61dffdcac9dbfaef60953381b25c20765cfe5a659e0.jpg)

<details>
<summary>text_image</summary>

Screenshot of a web-based document editing interface with eight numbered sections showing user profiles, tables, and content thumbnails.
</details>

![](images/72c8d267c0cea05bc029f1fe4ab40feef7a5bb13c8a881d94f76778b04c9328b.jpg)  
Q: Which view tab is selected immediately after creating the object set from the dialog?

A. References

B. Classes

C. Allocations

D. Time

# Answer: A

Reasoning: I check the video’s sequence of creating an object set via the dialog and the subsequent active view tab(...abbreviated...). After the object set creation dialog is closed and the new object set is generated, the interface displays content with "Incoming references" and a corresponding tree structure(...abbreviated...). This visual confirms the References tab is the active view immediately after the action, so the answer is A.

![](images/9a384289009ac5c3caa0cde5b3bc1db15f7827b898a500442f9e3f1fb002ed3b.jpg)  
Q: Which specific field keeps the selected objects reachable and prevents reclamation?

![](images/c98d5dc91917e7fd251f491f10b6bc229dc8ad7fbc044d0a97f3cab327f6a119.jpg)

# Answer: leakMap field

Reasoning: I analyze the video’s content about object reachability and garbage collection prevention in a Java profiling tool(...abbreviated...). The tool displays a hierarchical tree of object references for specific instances, and a dedicated internal field is used to maintain references to selected objects during analysis (...abbreviated...). The leakMap field is the specific one that keeps these objects reachable, stopping the garbage collector from reclaiming them during profiling.

![](images/19539ffdceff0d33495a98bc581cba87633bdfd3146bee142769a15e99ac8112.jpg)  
Q: In the heap analysis of new objects since the last mark heap action, what percentage of these objects are instances of the class java.awt.geom.GeneralPath? Round your answer to two decimal places.

A. 16.89%

B. 17.29%

C. 17.50%

D. 18.00%

# Answer: B

Reasoning: Extract two key values from the video’s heap analysis interface (...abbreviated...): the total number of new objects since the last mark heap action is 3384, and the instance count of java.awt.geom.GeneralPath is 585. Calculate the percentage with the formula (target class instances / total new objects) × 100(...abbreviated...). Substitute the values: (585 ÷ 3384) × 100 ≈ 17.2872%, round to two decimal places to get 17.29%, which corresponds to option B.

Figure 6. A VideoKR-SFT-201K example from the engineering domain. The reasoning process is presented in a concise and abbreviated form to improve readability.   
![](images/b93e52c5ee885e0788cd9650c3c4060a66b1c395e0ae0eeebb7b365b400cb6df.jpg)

<details>
<summary>flowchart</summary>

Comparison flowchart of analytical and predictive models, covering typical flow to solving time series, types of models, and their respective models.
</details>

![](images/b196b98db3d0c0c505b15db4dac9bc156af53a85b35bcc2c1f2449b16ccf5049.jpg)

Q: In the illustration with a warehouse, how many helmeted figures stand in a row?   
![](images/3a957fd27462251513ebac38118f486739822e9fe77f00470a433850389647d3.jpg)

A. Four

B. Three

C. Six

D. Five

# Answer: D

Reasoning: I examine the warehouse illustration with helmeted stick figures in the video(...abbreviated...). This illustration appears under the "Step 1: Define Problem" section, with a business/observer figure on the left and a row of helmeted figures on the right (...abbreviated...). Counting the helmeted figures in this row from the visual shows exactly five, so the answer is D.

![](images/ecdd1f9af2e526a9c3e56dd304f8be29b0ac3e458f11f3ef556c26b662ffb148.jpg)  
Q: Why are two separate models shown when predicting different future horizons?

![](images/38017d9fbadd3a57acb1777a266ca36b5dc73dfc65ff434f57beab6ca4332443.jpg)

# Answer: different target labels

Reasoning: I analyze the content in the video that addresses using separate models for predicting different future horizons(...abbreviated...). The video explains that predicting values for two distinct future time points (e.g., 3 days and 10 days out) requires individual models, each trained to forecast a specific time horizon(...abbreviated...). Each model corresponds to a unique prediction objective, meaning they are trained on different target labels—one for the nearer future horizon and one for the farther one, which is why two separate models are needed.

![](images/bc170eff980cc7bb558f47f1257bb1984eb520320815a1f9d2847fb77016f05d.jpg)  
Q: Based on the table shown in the video under “Step 3: What is the data saying?”, which lists the number of laptop inbound volumes for five consecutive dates, what is the average number of laptops received per day across the entire period?

# Answer: 164

Reasoning: To determine the average number of laptops received per day, I first extract the five daily inbound laptop volumes presented in the table in the video: 90 (2020-11-15), 213 (2020-11- 16), 178 (2020-11-17), 187 (2020-11-18), and 152 (2020-11-19). I then calculate the total inbound volume by summing these values: 90 + 213 + 178 + 187 + 152 = 820. Finally, I divide the total by the number of days (5) to find the daily average: 820 ÷ 5 = 164. This calculation confirms the average number of laptops received per day across the period is 164.

Figure 7. A VideoKR-SFT-201K example from the engineering domain. The reasoning process is presented in a concise and abbreviated form to improve readability.   
![](images/b0d294b69eef609a2a18f95bae7a46c96362b57a9dc3dd944982fbd78449ff75.jpg)

<details>
<summary>text_image</summary>

①
Herbert Auguth
Prime Minister
②
VOTES
Women
③
Lady
Constance
Bulwer-Lytton
④
Do us
Justice
⑤
⑥
⑦ Votes
For
Women
⑧
Cast:
Constance Lyber: "Davonar......" (Ricardo Kure)
As Nishakam, Erby, Women......... (Jin Whitshake)
Alicia, Pina, Dabie, and a red party. Thest Vomies.
Suffignates and other party played by members of the cast.
</details>

![](images/3ed76096480d39fcd3af7a12005c0820b643c1b3168a5a82b580b2d568cbf75e.jpg)

Q: At around 03:32, which item is shown immediately after the text sequence ends?   
![](images/ff61ea48621fc4b1d528a46d19eee0f22d89b8e1110888557bd40e1dc2df30c6.jpg)

A. A sign image

B. A portrait image

C. A window image

D. A crowd image

# Answer: C

Reasoning: Let me work through the timeline to understand what happens at the timestamp mentioned in the question.(...abbreviated...)Examining the sequence around that time shows a window image at 208.0 seconds, with faces and prison bar visuals following shortly after.(...abbreviated...)Given the timeline around 3:32 (212 seconds) and the sequence of visuals, the window image at 208.0 seconds would be what appears after a text sequence in that general timeframe.

![](images/ade0661927cca4cf4236d92be773ca9c35b6b0e607cb9a42a0eefefe5e0ddcc7.jpg)  
"What form of collective action is illustrated by the confrontations with authorities?

![](images/931c315feb3e97678069c56b3efb52d3130f0e88885669d2da66ed6872d5643f.jpg)

# Answer: militant protest

Reasoning: This video is about the women's suffrage movement in 1909, specifically focusing on "Deeds" as stated at around 00:18.(...abbreviated...)The title specifically uses "Deeds" and the historical context of 1909 points to the tactics of the Women's Social and Political Union (WSPU), which advocated for direct action rather than peaceful petition. (...abbreviated...)all indicate this is illustrating militant protest tactics - involving civil disobedience, direct confrontation with authorities, and willingness to face arrest and imprisonment for the cause.

![](images/eb67c9c94a3a24cb922377e2e2df74afe26525f3e828a6a47dfa240a1254cb79.jpg)  
Q: Based on the protesters’ tactics shown, what type of political strategy is illustrated?

![](images/1bd465b8a3f09d9047043377e9a7d025ce0032d2f9df0afc13326f566ed7a03a.jpg)

![](images/1131d6231e7c4433bf514b7d24982f36c0b17388092958c8f9b505a774f32a3d.jpg)

# Answer: militant direct action

Reasoning: To answer this question, I need to examine the protest tactics and imagery shown throughout the video. Key observations:(...abbreviated...)The phrase "DEEDS NOT WORDS" is particularly telling - it explicitly contrasts action-oriented protest with mere verbal advocacy. The combination of street protests, apparent physical confrontations with authorities, and subsequent imprisonment indicates these suffragettes went beyond traditional peaceful lobbying. This matches the historical record of the Women's Social and Political Union (WSPU), which adopted increasingly confrontational tactics including civil disobedience, property damage, and direct confrontation with authorities - tactics that deliberately provoked arrest to draw attention to their cause.

Figure 8. A VideoKR-SFT-201K example from the humanities and social science domain. The reasoning process is presented in a concise and abbreviated form to improve readability.

# A.4. Human-Validated Model Selection Protocol

To ensure that the VideoKR corpus is constructed with the highest possible quality while maximizing data diversity, we implemented a rigorous, human-in-the-loop qualification protocol for all foundation models used in our pipeline. Instead of relying on a single model (which risks imprinting specific model biases), we maintain a dynamic pool of eligible models for each synthesis stage. This section details the qualification methodology and the resulting model assignments.

We evaluate seven frontier models for potential inclusion in each pipeline step: GPT-5.2, GPT-5-mini, Claude-4.5-Sonnet, Gemini-3-Flash, DeepSeek-V3.2, Qwen3-VL-235B-A22B, and GLM-4.6V. For every pipeline stage defined in §3.1–§3.4, we conducted a controlled pilot study by sampling 100 representative input instances. Domain experts evaluated the model outputs against strict criteria, distinguishing between hard compliance failures (e.g., JSON format violations) and soft content failures (e.g., hallucinations or weak reasoning). A model was deemed eligible for a specific stage only if its total error rate $\leq 3 \%$ . Table 8 shows the human-validated models for each pipeline step.

Table 8. The eligible models at each VideoKR construction pipeline step. “-” indicates the model is not optimized for the modality required at that step; “✗” indicates the model fails human validation for that step. 

<table><tr><td>Pipeline Stage</td><td>Modality</td><td>GPT-5.2</td><td>GPT-5-mini</td><td>Claude-4.5-Sonnet</td><td>Gemini-3-Flash</td><td>DeepSeek-V3.2</td><td>Qwen3-VL-235B-A22B</td><td>GLM-4.6V</td></tr><tr><td colspan="9">§3.1 Domain Knowledge Bank Construction</td></tr><tr><td>Lecture → Knowledge Point</td><td>Text</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>-</td><td>-</td></tr><tr><td colspan="9">§3.2 Knowledge-Driven Video Collection</td></tr><tr><td>Scenario Generation</td><td>Text</td><td>√</td><td>√</td><td>√</td><td>√</td><td>✕</td><td>-</td><td>-</td></tr><tr><td>Search Keyword Generation</td><td>Text</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>-</td><td>-</td></tr><tr><td>Metadata Relevance Judge</td><td>Text</td><td>√</td><td>✕</td><td>√</td><td>✕</td><td>✕</td><td>-</td><td>-</td></tr><tr><td>Visual Relevance Judge</td><td>Vision</td><td>√</td><td>✕</td><td>✕</td><td>√</td><td>-</td><td>✕</td><td>✕</td></tr><tr><td colspan="9">§3.3 Skill-Oriented Example Generation</td></tr><tr><td>Example Generation</td><td>Vision</td><td>√</td><td>✕</td><td>√</td><td>√</td><td>-</td><td>✕</td><td>✕</td></tr><tr><td>CoT Rationale Validation</td><td>Vision</td><td>√</td><td>✕</td><td>√</td><td>√</td><td>-</td><td>✕</td><td>✕</td></tr></table>

# A.5. Data Contamination Mitigation

For Near-Duplicate Video Filtering, we uniformly sample both benchmark and VideoKR videos at 1 fps using ffmpeg and compute 64-bit perceptual hashes per frame. We partition each training video into overlapping 20-second windows with a 1-second stride and build an index over window-level hash sequences to enable scalable retrieval. For each benchmark video, we retrieve the top-10 candidate training windows and verify them by aligned-frame Hamming distance; we flag an overlap when the best 20-second window has at least 70% of frames with distance ≤ 30, and remove any matched training video.

# B. VideoKR-Eval Benchmark

# B.1. Detailed Statistics of VideoKR-Eval

As shown in Table 9, we construct VideoKR-Eval from three source benchmarks: VideoMMMU, MMVU, and SciVideoBench. We first perform multi-model single-frame probing and retain only original examples that are judged to require continuous video understanding by all three probing models. This yields 1,254 retained original examples. For the remaining filtered videos, domain experts re-annotate new visually grounded QA examples, contributing 746 additional examples. The final benchmark comprises 2,000 high-quality examples designed to require genuine video-level understanding and knowledge-intensive reasoning.

Table 9. Detailed statistics for the VideoKR-Eval benchmark construction. We retain original examples that are judged to require continuous video understanding by all three single-frame probing models, and add expert-reannotated examples for the filtered videos. 

<table><tr><td rowspan="2">Source Benchmark</td><td rowspan="2">Candidate Count</td><td colspan="3">VideoKR-Eval Composition</td><td rowspan="2">Final Count</td></tr><tr><td>Filtered</td><td>Retained Original</td><td>Expert-Reannotated</td></tr><tr><td>MMVU</td><td>1,000</td><td>639</td><td>361</td><td>398</td><td>759</td></tr><tr><td>VideoMMMU</td><td>900</td><td>560</td><td>340</td><td>241</td><td>581</td></tr><tr><td>SciVideoBench</td><td>1,000</td><td>447</td><td>553</td><td>107</td><td>660</td></tr><tr><td>Total</td><td>2,900</td><td>1,646</td><td>1,254</td><td>746</td><td>2,000</td></tr></table>

# C. Experiment Setup

# C.1. Post-Training Details

GRPO Reward Design. We employ GRPO (Shao et al., 2024) as our reinforcement learning algorithm. Following the standard RLVR-style reward formulation, the total reward is defined as $R = 0 . 1 \cdot R _ { f } + 0 . 9 \cdot R _ { a } .$ , where $R _ { f }$ and $R _ { a }$ denote the format and accuracy rewards, respectively. Specifically, $R _ { f }$ is set to 1.0 if the model output strictly satisfies the required format: <think>...</think><answer>...</answer>. For the accuracy reward $R _ { a }$ , we adopt the ROUGE metric for open-ended QA, while employing Exact Match (EM) for multiple-choice tasks.

Training Details. We train all models on up to 8 NVIDIA A800 GPUs (80 GB). For SFT, we use a learning rate of $1 \times 1 0 ^ { - 5 }$ , while for RL we use a learning rate of $5 \times 1 0 ^ { - 6 }$ . Both stages are optimized with AdamW, and the maximum response length is set to 2,048 tokens. For GRPO rollout generation, we set the rollout size G to 8 and use a temperature of 1.0 to encourage exploration. The KL penalty coefficient β is set to 0.01. Supervised fine-tuning is implemented with LLaMA-Factory (Zheng et al., 2024), while reinforcement learning is implemented with verl (Sheng et al., 2024).

# C.2. Evaluation Setup

To ensure fair and reproducible comparisons, we standardize the inference configuration for all evaluations by setting the temperature to 0.1. The maximum response token is set to 8,192 tokens.

All Benchmarks – For VideoKR and OneThinker   
```txt
{Question}
Please answer this question based on the visual content.
Provide your thinking process between the <think> and </think> tags, and then give your final answer between the <answer> and </answer> tags.
At the end, you must output the final answer in the format:
<answer><your answer here></answer> 
```

All Benchmarks – For VideoAuto-R1   
```txt
Video-MME
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option
{question}
Put your final answer in \\boxed{}. 
```

Other benchmarks   
```txt
{question}
Put your final answer in \\boxed{}. 
```

All Benchmarks – For Video-R1 and VideoRFT   
```txt
{Question}
Please think about this question as if you were a human pondering deeply.
Engage in an internal dialogue using expressions such as 'let me think', 'wait', 'Hmm', 'oh, I see', 'let's break it down', etc, or other natural language thought expressions.
It's encouraged to include self-reflection or verification in the reasoning process.
Provide your detailed reasoning between the <think></think> tags, and then give your final answer between the <answer></answer> tags.
For multiple-choice: Please provide only the single option letter (e.g., A, B, C, D, etc.) within the <answer></answer> tags.
For open-ended: Please provide your text answer within the <answer></answer> tags. 
```

Video-MME | MVBench – For Qwen series instruct models   
```txt
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
Question: {question} Possible answer choices:
{options}
The best answer is: 
```

Video-MME | MVBench – For Qwen series thinking models   
```txt
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
Question: {question} Possible answer choices:
{options}
Please reason step-by-step, identify relevant visual content, analyze key timestamps and clues, and then provide the final answer. 
```

Video-MME – For other models   
```txt
Select the best answer to the following multiple-choice question based on the video. Respond with only the letter (A, B, C, or D) of the correct option.
Question: {question} Possible answer choices:
{options}
Answer with the option's letter from the given choices directly. 
```

MVBench – For other models   
```txt
{question}
{options}
Only give the best option. 
```

LongVideoBench – For other models   
```txt
{question}
{options}
Answer with the option's letter from the given choices directly. 
```

VideoMMMU – For other models   
```txt
Perception & Comprehension:
{question}
{options}
Please ignore the Quiz question in last frame of the video. 
```

Adaptation-multiple-choice(open-ended):   
```txt
You should watch and learn the video content. Then apply what you learned to answer the following multi-choice(open-ended) question. The image for this question is at the end of the video.
{question}
{options} 
```

# MMVU – For Qwen series models and InternVL3.5-8B

# multiple-choice:

```txt
{question}
{options} 
```

Visual Information: processed video

Answer the given multiple-choice question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: ”Therefore, the final answer is: \$LETTER” (without quotes), where \$LETTER is one of the options. Think step by step before answering.

# open-ended:

```txt
{question} 
```

Visual Information: processed video

Answer the given question step by step. Begin by explaining your reasoning process clearly.

Conclude by stating the final answer using the following format: ”Therefore, the final answer is: \$ANSWER” (without quotes), where \$ANSWER is the final answer of the question. Think step by step before answering.

# MMVU – For other models

# multiple-choice:

```txt
{question}
{options} 
```

Visual Information: processed video

Do not generate any intermediate reasoning process. Answer directly with the option letter from the given choices.

# open-ended:

```txt
{question} 
```

Visual Information: processed video

Do not generate any intermediate reasoning process. Directly output the final answer.

# SciVideoBench – For Qwen series thinking models, GPT-5.4, Gemini 3 Pro, Claude Opus 4.5 and InternVL3.5-8B

```txt
{question}
{options} 
```

Answer the given question step by step. Begin by explaining your reasoning process clearly.

Conclude by stating the final answer using the following format: ”Therefore, the final answer is: \$ANSWER” (without quotes), where \$ANSWER is the final answer of the question. Think step by step before answering.

# SciVideoBench – For Qwen series instruct models and other models

```txt
{question}
{options} 
```

Answer with the option’s letter from the given choices directly.

VideoKR-Eval– For Qwen series thinking models, GPT-5.4, Gemini 3 Pro, Claude Opus 4.5 and InternVL3.5-8B   
```txt
multiple-choice:
{question}
{options}
Answer the given multiple-choice question step by step. Begin by explaining your reasoning process clearly. Conclude by stating the final answer using the following format: "Therefore, the final answer is: $LETTER" (without quotes), where $LETTER is one of the options. Think step by step before answering. 
```

```txt
open-ended:
{question}
Answer the given question step by step. Begin by explaining your reasoning process clearly.
Conclude by stating the final answer using the following format: "Therefore, the final answer is: $ANSWER" (without quotes), where $ANSWER is the final answer of the question. Think step by step before answering. 
```

VideoKR-Eval– For Qwen series instruct models and other models   
```txt
multiple-choice:  
{question}  
{options}  
Answer with the option's letter from the given choices directly. 
```

```txt
open-ended:
{question}
Please answer the question using a single word or phrase. 
```

# D. Experiment Results and Analysis

# D.1. Performance with Different Frames

Table 10. Detailed accuracy on general and knowledge-intensive video reasoning benchmarks for post-trained models across different input frames. 

<table><tr><td rowspan="2">Model</td><td rowspan="2">Release</td><td rowspan="2">Frames</td><td colspan="4">General Video Reasoning</td><td colspan="5">Knowledge-Intensive Video Reasoning</td></tr><tr><td>Video-MME</td><td>MVBench</td><td>LongVideoBench</td><td>Average</td><td>VideoMMMU</td><td>MMVU</td><td>SciVideoBench</td><td>VideoKR-Eval</td><td>Average</td></tr><tr><td colspan="12">Post-trained Qwen2.5-VL-7B-Instruct</td></tr><tr><td>VideoKR (SFT+RL)</td><td>2026-05</td><td>16</td><td>56.6</td><td>66.6</td><td>57.0</td><td>60.1</td><td>52.6</td><td>59.2</td><td>27.3</td><td>37.7</td><td>44.2</td></tr><tr><td>VideoKR (SFT+RL)</td><td>2026-05</td><td>32</td><td>60.1</td><td>68.2</td><td>58.2</td><td>62.2</td><td>51.2</td><td>58.9</td><td>27.4</td><td>39.8</td><td>44.3</td></tr><tr><td>VideoKR (SFT+RL)</td><td>2026-05</td><td>64</td><td>64.0</td><td>68.6</td><td>58.9</td><td>63.8</td><td>52.6</td><td>60.6</td><td>29.8</td><td>40.0</td><td>45.8</td></tr><tr><td>VideoKR (SFT+RL)</td><td>2026-05</td><td>128</td><td>66.4</td><td>68.9</td><td>61.3</td><td>65.5</td><td>52.2</td><td>60.5</td><td>32.5</td><td>41.2</td><td>46.6</td></tr><tr><td colspan="12">Post-trained Qwen3-VL-8B-Instruct</td></tr><tr><td>VideoKR (SFT+RL)</td><td>2026-05</td><td>16</td><td>57.2</td><td>65.5</td><td>54.7</td><td>59.1</td><td>60.2</td><td>63.8</td><td>29.3</td><td>40.5</td><td>48.5</td></tr><tr><td>VideoKR (SFT+RL)</td><td>2026-05</td><td>32</td><td>61.7</td><td>65.7</td><td>56.6</td><td>61.3</td><td>60.7</td><td>64.7</td><td>30.7</td><td>42.3</td><td>49.6</td></tr><tr><td>VideoKR (SFT+RL)</td><td>2026-05</td><td>64</td><td>63.9</td><td>66.2</td><td>57.9</td><td>62.7</td><td>62.1</td><td>63.8</td><td>31.5</td><td>44.6</td><td>50.5</td></tr><tr><td>VideoKR (SFT+RL)</td><td>2026-05</td><td>128</td><td>67.8</td><td>67.0</td><td>61.5</td><td>65.4</td><td>63.0</td><td>64.8</td><td>32.8</td><td>45.3</td><td>51.5</td></tr></table>

# D.2. Ablation Studies

Table 11. Ablation studies on post-training data. All experiments use Qwen2.5-VL-7B-Instruct as the base model, with 128 input frames. 

<table><tr><td rowspan="2">Ablation Setting</td><td colspan="4">General Video Reasoning</td><td colspan="5">Knowledge-Intensive Video Reasoning</td></tr><tr><td>Video-MME</td><td>MVBench</td><td>Long VideoBench</td><td>Average</td><td>VideoMMMU</td><td>MMVU</td><td>SciVideoBench</td><td>VideoKR-Eval</td><td>Average</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>65.1</td><td>66.3</td><td>60.9</td><td>64.1</td><td>51.1</td><td>55.7</td><td>28.1</td><td>32.7</td><td>41.9</td></tr><tr><td colspan="10">Skill-Oriented Data Composition</td></tr><tr><td>VR (Basic Reasoning)</td><td>61.0</td><td>60.8</td><td>52.3</td><td> $58.0_{-6.1}$ </td><td>51.3</td><td>51.7</td><td>27.3</td><td>35.3</td><td> $41.4_{-0.5}$ </td></tr><tr><td>VR + KV (Perception)</td><td>61.5</td><td>60.0</td><td>53.7</td><td> $58.4_{-5.7}$ </td><td>51.0</td><td>52.3</td><td>26.1</td><td>35.9</td><td> $41.3_{-0.6}$ </td></tr><tr><td>VR + KV + KVR (Full)</td><td>60.6</td><td>60.9</td><td>53.4</td><td> $58.3_{-5.8}$ </td><td>50.6</td><td>51.6</td><td>30.7</td><td>36.8</td><td> $42.4_{+0.5}$ </td></tr><tr><td colspan="10">Supervision Format</td></tr><tr><td>Direct Output</td><td>65.6</td><td>62.8</td><td>55.8</td><td> $61.4_{-2.7}$ </td><td>45.8</td><td>51.8</td><td>24.1</td><td>35.9</td><td> $39.4_{-2.5}$ </td></tr><tr><td>Chain-of-Thought (CoT)</td><td>60.6</td><td>60.9</td><td>53.4</td><td> $58.3_{-5.8}$ </td><td>50.6</td><td>51.6</td><td>30.7</td><td>36.8</td><td> $42.4_{+0.5}$ </td></tr><tr><td colspan="10">Comparison with Other SFT Corpora (SFT-only)</td></tr><tr><td>Video-R1-CoT-165k</td><td>56.9</td><td>63.2</td><td>51.9</td><td> $57.3_{-6.8}$ </td><td>45.3</td><td>50.4</td><td>21.4</td><td>27.5</td><td> $36.2_{-5.7}$ </td></tr><tr><td>OneThinker-SFT-340k</td><td>59.1</td><td>66.7</td><td>55.8</td><td> $60.5_{-3.6}$ </td><td>45.7</td><td>52.2</td><td>25.5</td><td>29.7</td><td> $38.3_{-3.6}$ </td></tr><tr><td>VideoRFT-CoT-102K</td><td>61.8</td><td>62.9</td><td>54.2</td><td> $59.6_{-4.5}$ </td><td>47.3</td><td>50.3</td><td>24.0</td><td>32.1</td><td> $38.4_{-3.5}$ </td></tr><tr><td>VideoKR-SFT-201K (Ours)</td><td>60.6</td><td>60.9</td><td>53.4</td><td> $58.3_{-5.8}$ </td><td>50.6</td><td>51.6</td><td>30.7</td><td>36.8</td><td> $42.4_{+0.5}$ </td></tr><tr><td colspan="10">Comparison with Other RL Corpora (RL-only)</td></tr><tr><td>Video-R1-260k</td><td>64.8</td><td>67.5</td><td>58.7</td><td> $63.7_{-0.4}$ </td><td>51.3</td><td>54.9</td><td>26.9</td><td>33.1</td><td> $41.6_{-0.3}$ </td></tr><tr><td>OneThinker-600k</td><td>63.7</td><td>64.5</td><td>52.7</td><td> $60.3_{-3.8}$ </td><td>53.7</td><td>54.5</td><td>27.6</td><td>33.2</td><td> $42.3_{+0.4}$ </td></tr><tr><td>VideoRFT-RL-310K</td><td>65.2</td><td>67.9</td><td>58.3</td><td> $63.8_{-0.3}$ </td><td>50.9</td><td>55.7</td><td>29.0</td><td>33.5</td><td> $42.3_{+0.4}$ </td></tr><tr><td>VideoAuto-R1-83K</td><td>65.0</td><td>66.9</td><td>56.6</td><td> $62.8_{-1.3}$ </td><td>52.4</td><td>55.9</td><td>29.0</td><td>33.3</td><td> $42.7_{+0.8}$ </td></tr><tr><td>VideoKR-RL-114K (Ours)</td><td>64.2</td><td>65.7</td><td>55.3</td><td> $61.7_{-2.4}$ </td><td>51.8</td><td>56.2</td><td>29.6</td><td>34.5</td><td> $43.0_{+1.1}$ </td></tr></table>

# D.3. Case Study

![](images/4538f64afdbba8097b9143927eb4374a63d09d1a491d7dc6c58f7f8379b17573.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["1: Source"] --> B["2: Source"]
    B --> C["3: Distribution"]
    C --> D["4: Output"]
    D --> E["5: Stream"]
    E --> F["6: Stream"]
    F --> G["7: Distribution"]
    G --> H["8: Output"]
```
</details>

Question: For the company shown in the video, what is one of the methods to handle associated risks?

Options: (A) diversification (B) vertical integration (C) horizontal integration (D) outsourcing (E) speculation

Explanation: A coffee brand and its upstream chain are shown; to hedge green - bean price risk, the firm can vertically integrate—own/establish plantations to secure supply and stabilize costs.

Qwen3-VL-8B-Thinking: <think> The video showcases a supply chain process: procurement, manufacturing, distribution, and trading. Among the various options, there is no indication of diversification or speculation; nor is there any vertical or horizontal integration. Phased handover and third-party transportation imply outsourcing some processes to mitigate non-core risks. Therefore, option D (outsourcing) is the most suitable. </think><answer>D </answer>

Error Reason: Misread generic supply-chain stages as proof of outsourcing, ignoring cues (coffee beans, upstream cost volatility) that point to vertical integration as the risk-mitigation strategy. The correct inference is owning upstream production (e.g., plantations), not contracting it out.

OneThinker: <think> Video context: a supply-chain flowchart (sourcing → manufacturing → distribution → transactional management) indicates production and logistics stages. Option check: A) Diversification —portfolio risk spreading; not depicted. B) Vertical integration —end-to-end control; no evidence. C) Horizontal integration—mergers; absent. D) Outsourcing— contracting external suppliers; truck/boxed goods suggest third-party handling, letting the firm focus on core tasks. E) Speculation—risky bets; irrelevant here. Thus, the risk-handling method most consistent with the video is outsourcing. Final: D. </think><answer>D</answer>

Error Reason: Misread the scene as generic logistics and inferred outsourcing, ignoring that the bubbles indicate coffee beans and the focus is upstream cost volatility. For a coffee company, the correct risk mitigation depicted is vertical integration (own plantations), not outsourcing.

Qwen3-VL-8B-Instruct (SFT+RL on VideoKR): <think> The video depicts a truck departing from a coffee plantation adorned with coffee bean icons and the Folgers logo, heading towards a roasting/processing plant, and then to a retail store. The process traces the supply chain from SOURCING → MANUFACTURING → DISTRIBUTION. This highlights exposure to upstream green-coffee price and supply volatility. A fitting mitigation is vertical integration —the brand establishes/owns plantations to secure supply, stabilize input costs, and tighten quality control. Therefore, the answer is B, vertical integration. </think><answer> B</answer>

Note: From coffee-bean icons and the SOURCING → MANUFACTURING → DISTRIBUTION flow, it identifies upstream risk and correctly maps the mitigation to vertical integration (owning plantations), not outsourcing.

Figure 9. Comparison of model responses on a knowledge-intensive video reasoning sample.

![](images/0947ffaed2ce4d4e136d6d8a203f1bcbf5be73bb991ae98dac2fca0403ee4425.jpg)

<details>
<summary>natural_image</summary>

Sequence of eight lab equipment photos showing laboratory glassware and lab equipment (no visible text or symbols)
</details>

Question: What is the main purpose of the overnight magnetic stirring step starting at 02:31?

# Options:

(A) Enhancing ultrasonication effects for smaller droplet size   
(B) Increasing the solvent concentration for better dispersion   
(C) Mixing to evenly distribute quantum dots in solution   
(D) Preventing sedimentation of formed nanobeads   
(E) Cooling the solution to induce polymer crystallization   
(F) Breaking the emulsion into larger droplets   
(G) Promoting chemical reaction between polymer and quantum dots   
(H) Reducing temperature fluctuations during particle formation   
(I) Removing excess water to concentrate the emulsion   
(J) Complete evaporation of solvent leading to solidification

Explanation: This step is central to the emulsion–solvent evaporation route: ultrasonication (02:23) forms a chloroform emulsion, and overnight stirring renews the solvent–air interface to drive evaporation. As chloroform leaves, non -volatile PSMA and encapsulated QDs precipitate and harden into stable nanobeads; without it, solidification fails—thus the answer is J.

Qwen3-VL-8B-Thinking: <think> At 02:31 in the video, the procedure introduces an overnight magnetic stirring step during the preparation of QD-nanobead LFIA. In fact, prolonged gentle stirring helps homogenization, and more importantly, it keeps the particles suspended, preventing precipitation. The following options were evaluated: A (enhanced sonication), B (increased solvent concentration), E (cooling/crystallization), F (increased droplet size), G (promoted reaction), H (temperature control), and I/J (concentration/evaporation). None of these align with the typical purpose of stirring. C (uniform distribution of QD), while feasible, is less important thanmaintaining suspension —the core objective. Therefore, the primary purpose of this step is D: preventing the precipitation of the formed nanobeads. </think><answer>D</answer>

Error Reason: Misread the step as anti-sedimentation (D). The sequence shows a volatile-solvent emulsion and later dry beads, indicating overnight stirring primarily drives complete solvent evaporation and solidification (J).

OneThinker: <think> The video demonstrates an overnight magnetic stirring step. In laboratory practice, prolonged gentle stirring is used to maintain the homogeneity of the dispersion, prevent precipitation/aggregation, and improve mass transfer, thereby ensuring a uniform distribution of the suspension (in this case, quantum dots) in the next step. It is not used to change solvent concentration (B), cool (E), control temperature (H), concentrate/evaporate (I, J), or increase droplet size (F). Enhanced sonication (A) is also not applicable—stirring is a low-shear process, typically performed \*\*after\*\* sonication, and its role is to maintain droplet integrity, not to further break down the droplets. Preventing precipitation (D) and any accidental interactions (G) may occur, but they are secondary. Therefore, the purpose is to mix to ensure a uniform distribution of the quantum dots in the solution—option C. </think><answer>C </answer>

Error Reason: Misread overnight stirring as simple homogenization (C), ignoring cues of a volatile-solvent emulsion and later dry beads. In this workflow, stirring primarily drives complete solvent evaporation and solidification (J), not just preventing settling.

Qwen3-VL-8B-Instruct (SFT+RL on VideoKR): <think> The video at 02:31 demonstrates the overnight magnetic stirring step in the nanobead preparation process. Overnight magnetic stirring is a routine homogenization step in the quantum dot-nanobead preparation process, and is therefore likely for maintaining uniform dispersion (C) and inhibiting precipitation (D). Comparing other options, many (B, E, F, H, I) do not conform to the typical function of stirring; D seems best suited for the overnight step. Wait— rewatching the segment around 02:23, we can see ultrasonic treatment forming an emulsion in a volatile solvent (chloroform), followed by footage of dried, solid nanobeads. During the emulsion-solvent evaporation process, the prolonged gentle stirring primarily serves to renew the solvent-air interface between droplets, thereby promoting complete solvent evaporation, allowing the polymer + quantum dots to precipitate and harden. Therefore, preventing precipitation is secondary; the primary goal is complete evaporation, ultimately leading to solidification—J. </think><answer>J </answer>

Note: Initial reasoning favored homogenization/anti-settling (C/D), but the “aha moment” came when linking the ultrasonication-created chloroform emulsion (02:23) to the later dry beads—revealing that overnight stirring chiefly drives complete solvent evaporation and solidification (J).

Figure 10. Comparison of model responses on a knowledge-intensive video reasoning sample.

![](images/e50227a5d2966e7c9fea55dea529c8c0680dc46489b3b647cd29b9364f1bb211.jpg)

<details>
<summary>text_image</summary>

1
FELVIC
RADIOGRAPHIC
2
3D REMERED
PELVIS
4
3D REMERED
PELVIS
5
WEIGHT
BEARING
6
WEIGHT
BEARING
7
PELVIC
LOGAMENTS
8
PELVIC
LOGAMENTS
</details>

Question: Can you identify the abnormality on this plain film of the pelvis? <image 1>

Options:

(A) Bone cyst   
(B) Acute hip fracture   
(C) Osteoarthritis   
(D) Surgical hardware   
(E) Resection of the pubic symphysis   
(F) Bone tumor   
(G) Hip dislocation   
(H) Osteoporosis   
(I) Paget's disease   
(J) Bone infection

Explanation: The key finding is a smooth cortical bone defect at the pubic symphysis, suggesting a previous pubic symphysis resection rather than acute separation or trauma. The sacroiliac and hip joints are well-matched, with no fracture lines or metal implants, and no signs of osteoarthritis, Paget's disease, osteoporosis, infection, or tumor.

Qwen3-VL-8B-Thinking: <think> The video shows a diffuse radiolucent area in the pelvis, accompanied by widespread cortical thinning and loss of normal trabecular detail. The hip joint is symmetrical, without displacement, and no focal osteolytic or sclerotic lesions, fracture lines, periosteal reaction, or metallic density shadows are observed. The joint space is intact, with no osteophyte formation, suggesting the absence of osteoarthritis, and lacks the typical features of Paget's disease (bone expansion, cortical thickening) or infection (invasive destruction, soft tissue swelling). The lesion is diffuse rather than focal, therefore osteoporosis is the most plausible explanation.</think><answer>H</answer>

Error Reason: The model incorrectly overinterpreted diffuse radiolucent areas as osteoporosis, neglecting a crucial finding—smooth cortical spaces at the pubic symphysis, consistent with surgical resection. The apparent "demineralization" may be related to bone protrusion/exposure, and there is no corroborating evidence of osteopenia prevalent in other sites.

OneThinker: <think> Based on the pelvis plain film and the listed options, the abnormality most plausibly involves the hip joint. Typical signs of a bone cyst, tumor, osteoporosis, Paget’s disease, osteoarthritis, pubic symphysis resection, hip dislocation, or bone infection are not clearly demonstrated, and an acute hip fracture is not convincingly visible. With the absence of definitive features for these alternatives, the most specific and common explanation in this context is a hip-related abnormality consistent with surgical hardware. Therefore, the best answer is D. Surgical hardware. </think><answer>D </answer>

Error Reason: Although no metallic density was observed on the imaging, the model incorrectly inferred the presence of a "surgical implant" and overlooked the smooth pubic symphysis and cortical absence—characteristic features of previous resections. Furthermore, the diagnosis incorrectly described the abnormality as related to the hip joint, rather than located at the pubic symphysis.

Qwen3-VL-8B-Instruct (SFT+RL on VideoKR): <think> The video shows significant destruction at the pubic symphysis, with a marked gap or separation, which is uncommon in a normal pelvis. Therefore, there is a significant abnormality in the pubic symphysis region. The interpubic cortical space is smooth, and the surrounding bone structure appears intact, but this is consistent with previous surgical resection rather than acute trauma. The hip joint is well-aligned, without displacement or dislocation; no cortical fracture or trabecular bone destruction is observed, nor is there an acute hip fracture; no focal osteolytic/sclerotic lesions, bone cysts, tumors, Paget's disease, or infection are seen. Diffuse demineralization suggests osteoporosis, but is insufficient for a definitive diagnosis; typical features of osteoarthritis such as joint space narrowing or osteophyte formation are also absent. No metallic feel or surgical implants are observed. Therefore, the pathological findings are most consistent with a diagnosis following pubic symphysis resection. </think><answer>E </answer>

Note: VideoKR focuses on the pubic symphysis and utilizes radiological prior information (depicting the pelvic ring and checking joint consistency) to avoid the biases of other models that are hip-centric and diffuse osteoporosis.

Figure 11. Comparison of model responses on a knowledge-intensive video reasoning sample.