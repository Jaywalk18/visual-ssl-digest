# VGIF-Score: Interpretable and Diagnostic Evaluation of Spatio-Temporal Instruction Following in Video Generation

Songyu Xu<sup>1</sup>, Xin Wang<sup>1</sup>, Qiang Chen<sup>2</sup>, Xinran Wang<sup>1</sup>, Muxi Diao<sup>1</sup>, Yuxuan Zhang<sup>1</sup>, Kongming Liang<sup>1</sup>, Rui Lin<sup>2⋆</sup>, and Zhanyu Ma<sup>1</sup>

<sup>1</sup> Beijing University of Posts and Telecommunications, Beijing, China <sup>2</sup> China Telecommunications Group Co., Ltd., Beijing, China linrui@chinatelecom.cn

Abstract. Recent video generation models (VGMs) have made substantial progress in visual fidelity, yet their ability to follow long, compositional instructions remains insuficiently evaluated. Existing evaluation protocols often rely on prompts that are short and semantically shallow, with limited atomic constraints and weak spatio-temporal dependencies. They also frequently depend on costly human evaluation or handcrafted vision pipelines, while providing little diagnostic insight into which instruction constraints succeed or fail. To address this gap, we propose VGIF-Score, a highly automated and interpretable framework for evaluating instruction following in video generation. VGIF-Score consists of two complementary components: an objective completion branch that parses prompts into a Spatio-Temporal Directed Acyclic Graph (ST-DAG) and performs dependency-aware QA with short-circuit diagnostics, and a subjective satisfaction branch that uses instructionconditioned AutoRubric to assess cinematography, visual purity, motion smoothness, and physics adherence. Together, these components produce a unified score that captures both objective completion and perceptual satisfaction. We instantiate this framework on VGIF-Bench, a benchmark of 223 long, structurally entangled prompts paired with approximately 4.3K fine-grained evaluation items. Experiments on 14 proprietary and open-source VGMs across more than 3K generated videos show that VGIF-Score provides reliable, interpretable, and diagnostically useful evaluation of video generation instruction following. The code will be available at https://github.com/PRIS-CV/VGIF-SCORE.

Keywords: Video Generation Models · Instruction Following

## 1 Introduction

Video Generation Models (VGMs) have rapidly progressed from early generative paradigms such as VAEs, GANs, and autoregressive modeling [17,10,41] to difusion and difusion-transformer architectures [14,2,8,9,24,23,36]. Recent large-scale systems [35,31] can produce visually faithful and increasingly cinematic videos through improved spatio-temporal modeling. Despite this progress, instruction following remains a key bottleneck. A model may handle “a girl walking in a park,” but struggle with “a girl in a red coat drops a glass, the glass shatters, and a nearby dog turns toward the sound”—a prompt that tests whether the model captures the logic of an event sequence, not merely individual visual phenomena. This gap reveals the need to evaluate spatio-temporal instruction following beyond visual plausibility.

Existing evaluation protocols, however, remain insuficient. Traditional metrics such as FVD [34] and CLIP-based scores [13] capture only low-level similarity or coarse semantics. Recent benchmarks broaden the landscape through multidimensional evaluation [15,16,45], compositional or physical reasoning tests [26,3], and human-aligned or MLLM-based judging [21]. Yet two fundamental limitations persist. First, prompts are typically short and semantically shallow. Even benchmarks with longer prompts [16] often evaluate coarse dimensions rather than dependency-aware execution. We argue that instruction dificulty is governed not by length alone, but by compositional depth—the number of atomic constraints and the dependencies among them—where existing benchmarks remain shallow (Table 1). Second, scoring is largely aggregate, ofering little diagnostic insight into which constraints fail and how failures propagate through causal chains.

To address these limitations, we propose VGIF-Score, a fine-grained and automated framework for evaluating instruction following in video generation (Figure 1). Inspired by Davidsonian Scene Graphs [4], we decompose each prompt into a Spatio-Temporal Directed Acyclic Graph (ST-DAG) of atomic semantic units—entities, attributes, locations, actions, states, and causal relations— connected by explicit dependency edges. From this graph, we derive dependencyaware QA pairs with a short-circuit mechanism that propagates failures along the dependency structure, and complement them with an instruction-conditioned AutoRubric [40] that assesses cinematography, visual purity, motion smoothness, and physics adherence. We instantiate this framework on VGIF-Bench, a diagnostic benchmark of 223 long-form, dependency-rich prompts and approximately 4.3K fine-grained evaluation items. Experiments on 14 opensource and proprietary VGMs across more than 3K generated videos show that VGIF-Score provides reliable and interpretable evaluation, and reveals two systematic failure modes: weak causal instruction following and strong sensitivity to depencey-depth and prompt position.

## 2 Related Works

Video Generation Models VGMs have evolved from early generative models to large-scale difusion and difusion-transformer systems [17,10,41]. Recent models have greatly improved visual fidelity, motion quality, and temporal consistency [35,31], enabling increasingly realistic and cinematic video synthesis from open-ended text prompts. These advances are largely driven by stronger spatio-temporal modeling, larger training corpora, and more expressive generative backbones. However, improved visual realism does not necessarily imply faithful instruction following. A video may look plausible at the frame or clip level while still omitting later constraints, confusing object states, or breaking the causal relation between events.

![](images/82abb9a9316b802105f586d7dd75bbcde8427683f06d15a09364620ac56b3a6c.jpg)  
Fig. 1. Overview of VGIF-Score. The framework evaluates spatio-temporal instruction following via objective QA-based scoring and subjective rubric-based assessment

Table 1. Comparison of representative video-generation evaluation benchmarks.

<table><tr><td>Benchmark</td><td>#P</td><td>W</td><td>U</td><td>Dep.</td><td>Depth</td><td>Obj.</td><td>Subj.</td><td>ST-DAG</td><td>Diag.</td></tr><tr><td>VBench [15]</td><td>946</td><td>7.7</td><td>1.7</td><td>0.1</td><td>-</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>VBench++ [16]</td><td>3330</td><td>8.8</td><td>1.7</td><td>0.2</td><td>-</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>VBench-2.0 [45]</td><td>1230</td><td>20.2</td><td>3.8</td><td>1.4</td><td>-</td><td>✓</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>T2V-CompBench [26]</td><td>1400</td><td>10.4</td><td>1.6</td><td>0.4</td><td>-</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>TC-Bench [7]</td><td>270</td><td>12.0</td><td>1.4</td><td>1.2</td><td>-</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>VMBench [19]</td><td>1050</td><td>26.3</td><td>4.7</td><td>1.9</td><td>-</td><td>-</td><td>-</td><td>✘</td><td>✘</td></tr><tr><td>T2VWorldBench [3]</td><td>1260</td><td>11.2</td><td>1.6</td><td>0.4</td><td>-</td><td>✓</td><td>✘</td><td>✘</td><td>✘</td></tr><tr><td>ChronoMagic [44]</td><td>1649</td><td>45.2</td><td>8.4</td><td>4.8</td><td>-</td><td>-</td><td>-</td><td>✘</td><td>✘</td></tr><tr><td>GenAI-Bench [18]</td><td>512</td><td>12.5</td><td>2.2</td><td>0.5</td><td>-</td><td>✘</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>MJ-Video [33]</td><td>1085</td><td>46.5</td><td>7.8</td><td>2.9</td><td>-</td><td>✘</td><td>✓</td><td>✘</td><td>✘</td></tr><tr><td>VGIF-Bench (Ours)</td><td>223</td><td>78.2</td><td>11.3</td><td>6.5</td><td>4.9</td><td>✓</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

Abbr. W: average words; U: atomic units; Dep.: estimated dependencies; Obj.: objective assessment; Subj.: subjective assessment; ST-DAG: explicit spatio-temporal directed acyclic graph; Diag.: diagnostic evaluation. U and Dep. are uniformly estimated from prompt text for fair comparison. Depth is reported only when an explicit graph structure is available. For VGIF-Bench, the explicit ST-DAG contains 16.4 nodes and 17.7 edges per prompt on average.

Video Generation Benchmarks and Metrics VGM evaluation has progressed from automatic metrics such as FVD [34] and CLIP-based similarity [13,27] to comprehensive benchmark suites. Recent works evaluate video quality and consistency across multiple dimensions [15,45,16], incorporate humanaligned or preference-oriented assessment [21], study compositionality and textvideo alignment [26], or explore physical reasoning, world knowledge, long-context prompts, and MLLM-based judging [3,37,38,42,28,29]. These eforts have substantially broadened the scope of video generation evaluation, covering visual quality, temporal consistency, motion realism, text-video alignment, and user preference. Nevertheless, most protocols still treat prompts primarily as flat text, and their evaluation targets are often defined at the dimension or video level. As a result, they provide limited information about which atomic constraints are satisfied, which prerequisite events are missing, and how an early failure afects downstream state changes or causal outcomes.

## 3 VGIF-Score

## 3.1 Framework Overview

Given a text prompt p and a generated video $x ,$ VGIF-Score evaluates how faithfully x follows the instruction in $p .$ As shown in Figure 1, it consists of two complementary branches: objective completion and subjective satisfaction.

The objective branch uses an LLM to parse p into a Spatio-Temporal Directed Acyclic Graph (ST-DAG). Based on this graph, we construct dependency-aware QA pairs and use a VLM evaluator to answer them against the generated video. The QA accuracy yields an objective completion score.

The subjective branch uses an LLM to generate an instruction-conditioned AutoRubric tailored to the specific prompt $p ,$ each rubric dimension produces scoring criteria and anchor descriptions. The VLM evaluator rates cinematography, visual purity, motion smoothness, and physics adherence on a 1–5 scale, normalized to [0, 1], and equally weighted. The final VGIF-Score combines the two branches with equal weights.

## 3.2 ST-DAG-Based Objective Completion

We represent each prompt p as a Spatio-Temporal Directed Acyclic Graph:

$$
\mathcal {G} (p) = (\mathcal {V}, \mathcal {E}), \quad \mathcal {Q} (p) = \{(q _ {i}, a _ {i}) \} _ {i = 1} ^ {N},\tag{1}
$$

where each node $v \in \mathcal V$ denotes an atomic semantic unit and each directed edge $e \in { \mathcal { E } }$ denotes a dependency relation. The node set covers six types—entity, attribute, location, action, state, and causal —progressing from static scene elements to dynamic events and their consequences. Edge types include solid dependencies (compositional prerequisites) and causal dependencies (consequence relations). Based on $\mathcal { G } ( \boldsymbol { p } )$ , we derive N dependency-aware QA pairs, where $q _ { i }$ is a binary question associated with a graph node and $a _ { i }$ is its expected answer.

Dependency-aware evaluation. Given video x, a VLM evaluator answers every question independently. Before crediting a node, we verify that its dependency expression is satisfied. Dependencies follow the logical connectives specified in the ST-DAG annotation: conjunctive (AND) dependencies require all upstream nodes to be correct, while disjunctive (OR) dependencies require at least one. If the dependency expression evaluates to false, the node is marked incorrect regardless of its own answer, and this failure propagates to all downstream nodes along the dependency chain.

Formally, let $\hat { a } _ { i }$ be the VLM’s answer for question $q _ { i } ,$ and let dep(i) denote its dependency expression over upstream $\mathrm { Q A }$ indices. The per-node correctness is defined recursively:

$$
c _ {i} = \mathbf {1} [ \hat {a} _ {i} = a _ {i} ] \wedge \operatorname{eval} \bigl (\mathrm{dep} (i), \{c _ {j} \} _ {j <   i} \bigr),\tag{2}
$$

where eval(·) evaluates the Boolean dependency expression (supporting AND, OR, and parentheses) against previously computed correctness values. The objective completion score is:

$$
S _ {\mathrm{obj}} (p, x) = \frac {1}{N} \sum_ {i = 1} ^ {N} c _ {i}.\tag{3}
$$

## 3.3 Instruction-Conditioned AutoRubric

While the objective branch verifies whether individual semantic units are realized, it does not capture holistic perceptual qualities that afect user satisfaction. We therefore introduce an instruction-conditioned AutoRubric that evaluates four complementary dimensions:

Cinematography: whether camera work, composition, lighting, and pacing match the narrative tone specified in the prompt.

Visual purity: whether the video contains only elements specified in the prompt, with no extraneous objects, identity drift, or artifacts.

Motion smoothness: whether movements and interactions are temporally smooth, continuous, and free of jitter or freezing.

Physics adherence: whether prompt-specified interactions and state changes follow plausible physical behavior.

Crucially, the rubric is not generic: for each prompt, the LLM generates prompt-specific scoring criteria and score anchor descriptions for every dimension, so that the evaluator judges the video against what was actually requested rather than an abstract quality standard.

For each dimension $k \in \{ 1 , 2 , 3 , 4 \}$ , the evaluator assigns a raw score $r _ { k } ( p , x ) \in$ {1, 2, 3, 4, 5}. We normalize and aggregate:

$$
\tilde {r} _ {k} (p, x) = \frac {r _ {k} (p , x) - 1}{4}, \quad S _ {\text { rubric }} (p, x) = \frac {1}{4} \sum_ {k = 1} ^ {4} \tilde {r} _ {k} (p, x).\tag{4}
$$

## 3.4 Final Score

The final VGIF-Score combines objective completion and subjective satisfaction:

$$
S _ {\mathrm{VGIF}} (p, x) = \frac {1}{2} S _ {\mathrm{obj}} (p, x) + \frac {1}{2} S _ {\mathrm{rubric}} (p, x).\tag{5}
$$

![](images/4104a0c2fb07feb9152138a3a2e69ed88b70c5f518999bc5120f90f39c414000.jpg)  
Fig. 2. Overview of VGIF-Bench. The figure presents (a) the hierarchical prompt taxonomy, (b) graph depth distribution, (c) ST-DAG node-type composition, and (d) multi-parent node distribution. These statistics illustrate the coverage and structural complexity of the benchmark.

For a benchmark of M prompt-video pairs, the overall score is:

$$
\mathrm{VGIF-Score} = \frac {1}{M} \sum_ {m = 1} ^ {M} S _ {\mathrm{VGIF}} (p _ {m}, x _ {m}).\tag{6}
$$

## 4 VGIF-Bench

VGIF-Bench is designed to evaluate whether video generation models can faithfully execute long, structurally entangled instructions rather than merely depict isolated objects or short event fragments. Unlike prior benchmarks that treat prompts as flat text, VGIF-Bench represents each prompt as an explicit spatiotemporal directed acyclic graph (ST-DAG), aligns graph nodes with dependencyaware QA, and complements structural verification with instruction-conditioned autorubric scoring. This design makes the benchmark not only more challenging, but also substantially more interpretable and diagnostic.

## 4.1 Benchmark Construction

We construct VGIF-Bench through a largely automated pipeline with human verification, prioritizing structural instruction complexity over benchmark scale. Starting from a hierarchical taxonomy of video-generation scenarios, GPT-5.2 [22] drafts each benchmark sample together with (1) a long-form prompt, (2) its ST-DAG decomposition, (3) dependency-aware QA pairs, and (4) instructionconditioned autorubric specifications. Automatic validation then enforces schema consistency, dependency validity, duration normalization, and sample deduplication, after which human annotators verify graph semantics, QA answerability, and rubric alignment. This design allows the expensive human efort to focus on verification and correction, rather than writing all prompts and evaluation items from scratch.

The final benchmark contains 223 prompts, 3,445 dependency-aware QA pairs, and 892 autorubric dimension specifications, yielding approximately 4.3K fine-grained evaluation items across the objective and subjective branches. Across the benchmark, the ST-DAG annotations contain 3,656 nodes and 3,940 edges in total, corresponding to 16.4 nodes and 17.7 edges per prompt on average.

## 4.2 Benchmark Structure and Distribution

Figure 2 summarizes the coverage and structural complexity of VGIF-Bench. As shown in Figure 2(a), the benchmark spans 8 macro categories and 38 subcategories, covering product, narrative, spatial, emotional, physical, performative, natural, and surreal scenarios. This taxonomy is intended to capture both common real-world generation requests and compositionally dificult prompts that require coordinated multi-step execution.

Figure 2(b) shows the distribution of graph depth across prompts. Most samples exhibit non-trivial hierarchical structure, with the mass concentrated in the mid-to-high depth range rather than near-flat dependency chains. This indicates that VGIF-Bench is challenging not only because prompts are long, but because many instructions require multi-stage semantic execution under explicit dependency constraints.

Figure 2(c) presents the composition of ST-DAG node types. While entity and action nodes form the dominant backbone, VGIF-Bench also contains substantial numbers of state, location, attribute, and causal nodes. This makes temporal evolution and inter-event dependency first-class evaluation targets rather than incidental byproducts of prompt wording.

Figure 2(d) highlights the prevalence of non-linear dependency patterns. Many prompts contain multiple nodes whose realization depends on several upstream conditions simultaneously, rather than simple left-to-right chains. Such multi-parent structures are important because they enable fine-grained diagnosis of failure propagation and reveal whether a model can maintain coherent execution across intertwined semantic constraints.

Taken together, these properties make VGIF-Bench dificult not because it is large, but because successful generation requires coordinated realization of entities, actions, states, and causal outcomes under explicit structural dependencies. These properties directly motivate our evaluation design: objective QA localizes constraint-level failures, while AutoRubric captures perceptual degradation caused by broken event execution.

## 5 Evaluation

## 5.1 Experimental Setup

Models. We evaluate 14 representative VGMs, including five proprietary models with undisclosed parameters (Kling-V3, Seedance-2.0, Wan-2.7, ViduQ3-Turbo, and PixVerse-V6) and nine open-source models with publicly released weights: LTX-2.0 (19B) [12], Wan2.2-A14B (27B total, 14B active) [35], HyVideo-1.5 (HunyuanVideo-1.5, 8.3B) [39], LongCat-Video (13.6B) [32], Mochi-1 (10B) [30], CogVideoX-1.5 (5B) [43], MAGI-1 (4.5B) [1], URSA (1.7B) [6], and InfinityStar (8B) [20]. This model pool covers both difusion-transformer variants and autoregressive video generation, and spans a broad range of parameter scales from compact 1.7B models to large-scale 27B systems, providing a comprehensive testbed for evaluating spatio-temporal instruction following.

Table 2. Main results on VGIF-Bench. The first-ranked result is highlighted in blue, and the second-ranked result in green. Columns are organized as Entity, Attribute, Location, Action, State, Causal, Objective Score, Cinematography, Visual Purity, Motion Smoothness, Physics Adherence, Subjective Score, and VGIF-Score.

<table><tr><td rowspan="2">Model</td><td colspan="7">Objective</td><td colspan="5">Subjective</td><td rowspan="2">VGIF</td></tr><tr><td>Ent.</td><td>Attr.</td><td>Loc.</td><td>Act.</td><td>Sta.</td><td>Cau.</td><td>Obj.</td><td>Cin.</td><td>Pur.</td><td>Mot.</td><td>Phy.</td><td>Sub.</td></tr><tr><td colspan="14">Commercial Models</td></tr><tr><td>Kling-V3</td><td>71.07</td><td>57.36</td><td>75.14</td><td>20.73</td><td>12.60</td><td>4.21</td><td>42.18</td><td>50.58</td><td>74.35</td><td>49.06</td><td>37.58</td><td>52.89</td><td>46.30</td></tr><tr><td>Seedance-2.0</td><td>70.18</td><td>54.01</td><td>75.00</td><td>19.34</td><td>11.65</td><td>2.98</td><td>40.96</td><td>55.67</td><td>71.52</td><td>55.76</td><td>43.41</td><td>56.59</td><td>47.59</td></tr><tr><td>Wan-2.7</td><td>76.46</td><td>70.91</td><td>82.73</td><td>22.12</td><td>13.21</td><td>3.46</td><td>46.29</td><td>45.88</td><td>64.89</td><td>42.35</td><td>33.21</td><td>46.58</td><td>46.44</td></tr><tr><td>ViduQ3-Turbo</td><td>74.35</td><td>66.07</td><td>78.73</td><td>35.57</td><td>10.95</td><td>3.68</td><td>44.76</td><td>44.93</td><td>67.62</td><td>46.19</td><td>34.35</td><td>48.27</td><td>45.35</td></tr><tr><td>PixVerse-V6</td><td>75.41</td><td>66.97</td><td>81.22</td><td>25.91</td><td>15.50</td><td>4.21</td><td>46.73</td><td>48.79</td><td>66.73</td><td>44.84</td><td>35.34</td><td>48.93</td><td>47.18</td></tr><tr><td colspan="14">Open-Source Models</td></tr><tr><td>LTX-2.0</td><td>57.38</td><td>45.65</td><td>61.88</td><td>8.95</td><td>4.34</td><td>0.53</td><td>31.06</td><td>40.00</td><td>57.49</td><td>44.48</td><td>36.59</td><td>44.64</td><td>36.50</td></tr><tr><td>Wan2.2-A14B</td><td>69.33</td><td>60.06</td><td>74.31</td><td>14.37</td><td>8.88</td><td>1.84</td><td>39.48</td><td>39.10</td><td>56.77</td><td>37.31</td><td>28.61</td><td>40.45</td><td>39.96</td></tr><tr><td>HyVideo-1.5</td><td>59.79</td><td>50.15</td><td>65.75</td><td>12.37</td><td>6.20</td><td>0.79</td><td>33.76</td><td>44.30</td><td>60.36</td><td>47.53</td><td>37.40</td><td>47.40</td><td>39.18</td></tr><tr><td>LongCat-Video</td><td>64.42</td><td>55.26</td><td>66.02</td><td>11.07</td><td>5.79</td><td>0.53</td><td>35.27</td><td>39.01</td><td>52.74</td><td>42.06</td><td>32.02</td><td>41.46</td><td>38.47</td></tr><tr><td>Mochi-1</td><td>56.03</td><td>50.45</td><td>65.19</td><td>9.66</td><td>5.37</td><td>0.26</td><td>31.76</td><td>35.53</td><td>52.65</td><td>33.15</td><td>28.07</td><td>37.35</td><td>33.28</td></tr><tr><td>CogVideoX-1.5</td><td>52.75</td><td>51.65</td><td>63.26</td><td>9.54</td><td>4.13</td><td>0.00</td><td>30.37</td><td>28.43</td><td>45.74</td><td>31.21</td><td>25.47</td><td>32.71</td><td>31.54</td></tr><tr><td>MAGI-1</td><td>44.74</td><td>36.34</td><td>59.94</td><td>3.30</td><td>2.27</td><td>0.00</td><td>24.63</td><td>24.04</td><td>41.97</td><td>26.28</td><td>23.14</td><td>28.86</td><td>26.74</td></tr><tr><td>URSA</td><td>51.69</td><td>49.25</td><td>60.22</td><td>5.18</td><td>2.89</td><td>0.00</td><td>28.33</td><td>29.60</td><td>41.08</td><td>34.80</td><td>27.09</td><td>33.14</td><td>30.92</td></tr><tr><td>InfinityStar</td><td>62.68</td><td>59.46</td><td>72.10</td><td>11.07</td><td>2.89</td><td>0.00</td><td>35.33</td><td>38.61</td><td>52.24</td><td>44.38</td><td>31.24</td><td>35.52</td><td>35.43</td></tr></table>

Evaluation model and protocol. We use Gemini-3.1-Pro [11] as the unified VLM evaluator for both QA-based objective completion and AutoRubricbased subjective assessment. For each prompt-video pair, VGIF-Score is computed at the video level and then averaged over the benchmark. For dimensionwise analysis, such as entity, action, and causal relation, we aggregate QA accuracy across all questions belonging to the corresponding semantic dimension.

## 5.2 Main Results

Table 2 reports the main results on VGIF-Bench.

Overall performance. Proprietary models achieve an average VGIF-Score of 46.57, compared with 34.67 for open-source models, showing a clear but not overwhelming gap. Seedance-2.0 obtains the highest overall VGIF-Score among proprietary models, while Wan2.2-A14B leads the open-source group. Nevertheless, even the best-performing models remain far from fully satisfying VGIF-Bench, indicating that spatio-temporal instruction following is still a challenging capability beyond visual fidelity.

Objective vs. subjective gap. High subjective quality does not necessarily imply strong objective completion. Several models obtain reasonable visual purity or motion scores but remain weak on action, state, and causal dimensions.

Table 3. VGIF-Score by scenario category. The first-ranked result is highlighted in blue, and the second-ranked result in green.

<table><tr><td>Model</td><td>Product</td><td>Narrative</td><td>Surreal</td><td>Physics</td><td>Emotion</td><td>Spatial</td><td>Performance</td><td>Nature</td></tr><tr><td colspan="9">Commercial Models</td></tr><tr><td>Kling-V3</td><td>46.13</td><td>43.12</td><td>44.85</td><td>43.25</td><td>50.39</td><td>48.78</td><td>49.31</td><td>45.46</td></tr><tr><td>Seedance-2.0</td><td>47.26</td><td>46.14</td><td>48.82</td><td>40.93</td><td>53.18</td><td>49.66</td><td>49.21</td><td>46.83</td></tr><tr><td>Wan-2.7</td><td>49.77</td><td>43.03</td><td>53.17</td><td>50.62</td><td>65.48</td><td>58.20</td><td>61.60</td><td>58.06</td></tr><tr><td>ViduQ3-Turbo</td><td>46.45</td><td>38.63</td><td>42.77</td><td>42.61</td><td>51.47</td><td>46.50</td><td>50.06</td><td>45.97</td></tr><tr><td>PixVerse-V6</td><td>46.20</td><td>42.74</td><td>46.05</td><td>44.17</td><td>50.24</td><td>47.24</td><td>48.29</td><td>54.59</td></tr><tr><td colspan="9">Open-Source Models</td></tr><tr><td>LTX-2.0</td><td>33.00</td><td>34.78</td><td>34.65</td><td>34.03</td><td>37.70</td><td>40.18</td><td>38.79</td><td>40.20</td></tr><tr><td>Wan2.2-A14B</td><td>38.16</td><td>37.65</td><td>41.59</td><td>39.57</td><td>32.23</td><td>39.96</td><td>39.95</td><td>47.58</td></tr><tr><td>HyVideo-1.5</td><td>40.43</td><td>29.47</td><td>35.06</td><td>35.62</td><td>47.67</td><td>39.36</td><td>43.47</td><td>45.26</td></tr><tr><td>LongCat-Video</td><td>37.73</td><td>34.09</td><td>33.23</td><td>36.56</td><td>44.95</td><td>40.12</td><td>38.95</td><td>43.86</td></tr><tr><td>Mochi-1</td><td>33.87</td><td>27.40</td><td>34.39</td><td>30.42</td><td>39.44</td><td>34.72</td><td>32.14</td><td>34.80</td></tr><tr><td>CogVideoX-1.5</td><td>28.81</td><td>29.87</td><td>29.71</td><td>26.00</td><td>36.37</td><td>35.92</td><td>30.43</td><td>30.43</td></tr><tr><td>MAGI-1</td><td>27.55</td><td>24.46</td><td>25.87</td><td>28.55</td><td>29.39</td><td>29.39</td><td>24.06</td><td>26.80</td></tr><tr><td>URSA</td><td>33.19</td><td>25.23</td><td>32.70</td><td>30.64</td><td>31.97</td><td>29.61</td><td>30.66</td><td>34.12</td></tr><tr><td>InfinityStar</td><td>33.44</td><td>27.06</td><td>32.93</td><td>33.26</td><td>41.65</td><td>40.47</td><td>34.46</td><td>42.02</td></tr></table>

This discrepancy supports the need for a dual-branch metric: subjective quality alone may overestimate visually plausible but semantically incomplete videos, while objective QA alone cannot capture perceptual degradation.

Causal reasoning bottleneck. Causal instruction following is the most dificult objective dimension across nearly all models. Even the strongest commercial models obtain causal scores below 5, while multiple open-source models are close to zero. This indicates that current VGMs still struggle to bind events into coherent cause-efect chains, rather than simply rendering local objects or short actions. Importantly, causal failures are not isolated errors: once a triggering action or prerequisite state is missed, downstream state changes and causal outcomes often become impossible to realize. This explains why causal scores are substantially lower than entity or location scores, and motivates the dependencyaware short-circuit design of VGIF-Score.

## 5.3 Category-wise Analysis

We further analyze performance across scenario categories in Table 3.

Scenario dificulty. Diferent categories expose distinct model weaknesses. Emotion, performance, and nature scenarios tend to receive higher scores, partly because they often rely more on appearance, style, or short-range motion. In contrast, narrative and physics-related scenarios are more challenging because they require richer temporal evolution, state transitions, and causal dependencies. This trend is consistent with the structural statistics in Figure 2, where deeper dependency chains and multi-parent structures are common sources of dificulty.

Model bias. Models also exhibit category-specific biases. Some systems perform competitively on appearance-driven categories but degrade in structured or multi-entity interactions. For example, a model may generate visually appealing product or nature videos while failing to maintain coherent event progression in narrative or physics scenarios. Such results show that high visual quality alone does not guarantee robust generalization to dependency-rich instructions.

Implication. The category-wise results suggest that future evaluation should not only report a single overall score, but also expose which types of instructions stress a model. A model optimized for visual appeal may rank highly on product or nature prompts, yet still fail on categories that require event-level reasoning. VGIF-Bench therefore provides a more diagnostic view of model capability by linking scenario-level performance with explicit structural properties.

## 5.4 Structural Analysis

VGIF-Bench’s ST-DAG representation makes it possible to measure where in the prompt and at which compositional depth instruction following degrades. We analyze accuracy along two orthogonal axes: relative position of constraints in the prompt text, and dependency depth in the ST-DAG.

![](images/857d034596f33fdabfa908be37918ac4983841230a6f12426f8d7d5816042358.jpg)

(b) Accuracy Heatmap: Model x Dependency Depth  
![](images/d0fbb9b3e9b8418da64c91240db0ebe25c9e570a57e25373706e931163407050.jpg)  
Fig. 3. Structural factors governing instruction-following accuracy. (a) QA accuracy vs. relative position in the prompt. Accuracy drops from 67.9% (first 20%) to 10.1% (final 20%), a 6.7× decline universal across all 14 VGMs. (b) Heatmap of accuracy by model and dependency depth. Accuracy decreases monotonically with depth; depths 8–12 are merged. All values to one decimal place.

Prompt position. Fig. 3a reveals a strong recency bias: averaged across all 14 models, the accuracy falls from 67.9% for constraints appearing in the first 20% of the prompt to 48.9% at 0.2–0.4, 27.3% at 0.4–0.6, 14.9% at 0.6–0.8, and 10.1% in the final 20%—a 6.7× decline. Even PixVerse-V6 drops from 85.5% at early positions to 14.4% at late positions. Entity questions retain 50.3% accuracy at late positions, while causal questions drop to 1.0% past the midpoint, showing that position sensitivity is most severe for semantically complex constraints.

Dependency depth. Fig. 3b shows that precision decreases monotonically with ST-DAG depth: averaged across all 14 models, 80.6% at depth 0 (independent questions), 58.5% at depth 1, 36.4% at depth 2, 15.1% at depth 3 and

![](images/02eba30efd1c225ea8087fc6d7e49c6ad7517118a4fc7fbbbd135410c4d0f547.jpg)  
Fig. 4. Dependency-aware causal chain diagnosis. Kling-V3 executes the full perfume transformation chain (12/12 QA, 2/2 causal). CogVideoX-1.5 preserves the local scene and early mist formation (q1–q8), but misses the trigger action q9; the downstream causal nodes q10 and q12 fail under dependency-aware evaluation.

5.6% at depth 4—an average drop of ∼19% per level. Beyond depth 4, accuracy falls below 3% for all 14 models. PixVerse-V6 retains 62.2% at depth 2 while MAGI-1 drops to 19.0%, yet all models converge to near-zero beyond depth 4. The depth 0→1 decline (80.6%→58.5%, a 22.1% drop) quantifies the immediate cost of even a single dependency.

Interaction. Position and depth efects compound multiplicatively: a causal constraint appearing late in the prompt and sitting at depth ≥3 faces near-zero success probability across all models. Together, these two orthogonal axes— temporal attention decay and compositional reasoning depth—explain the dominant share of the instruction-following gap.

## 5.5 Diagnostic Analysis

To illustrate how structural factors interact with both objective and subjective evaluation, Figure 4 presents a side-by-side diagnosis of Kling-V3 and CogVideoX 1.5 on the same VGIF-Bench prompt. The prompt describes a surreal perfume advertisement with an explicit causal chain: the bottle sprays mist, the mist forms a floating ring, a gold key rotates through the ring, the liquid shifts color, the mirror reflection changes, and a silk glove animates to applaud.

Objective diagnosis. Kling-V3 executes the full dependency chain, answering all 12 dependency-aware QA pairs correctly, including both causal questions. In contrast, CogVideoX-1.5 satisfies the local scene and early state constraints but fails at the bridge action where the key should rotate through the mist ring. Because this action gates downstream causal and state nodes, the short-circuit mechanism propagates the single failure to later constraints. The ST-DAG therefore localizes the failure to the causal transition edge, rather than only reporting a flat aggregate accuracy.

Table 4. Human validation and alignment.  
(a) Evaluator Efectiveness

<table><tr><td>Signal</td><td>Human Reference</td><td>Statistic</td><td>Value</td></tr><tr><td>ST-DAG QA</td><td>QA labels</td><td>Agreement</td><td>96.3%</td></tr><tr><td>ST-DAG QA</td><td>QA labels</td><td>Cohen&#x27;s κ</td><td>0.92</td></tr><tr><td>AutoRubric</td><td>Rubric scores</td><td>Spearman ρ</td><td>0.83</td></tr><tr><td>VGIF-Score</td><td>Human-derived VGIF</td><td>Spearman ρ</td><td>0.87</td></tr><tr><td colspan="4">(b) Human Alignment</td></tr><tr><td>Automatic Score</td><td>Human-Comp.</td><td>Human-Sat.</td><td>Human-Overall</td></tr><tr><td>Objective Score</td><td>0.78</td><td>0.52</td><td>0.65</td></tr><tr><td>AutoRubric Score</td><td>0.41</td><td>0.81</td><td>0.72</td></tr><tr><td>VGIF-Score</td><td>0.71</td><td>0.83</td><td>0.89</td></tr></table>

## 5.6 Human Validation and Alignment

We randomly sampled 200 generated videos from VGIF-Bench and collected annotations from three human annotators per sample. Human annotators answer the same ST-DAG QA pairs, score videos with the same AutoRubric criteria, and provide overall ratings. We report Cohen’s κ [5] for categorical QA agreement and Spearman rank correlation [25] for rating-based scores. Table 4 summarizes two aspects of validation. First, Gemini-3.1-Pro shows strong consistency with human annotations under the same evaluation protocol, supporting its effectiveness as the VLM evaluator. Second, the objective branch better aligns with human completion judgments, while AutoRubric better aligns with subjective satisfaction. The final VGIF-Score achieves the highest correlation with human overall ratings, demonstrating that combining structural correctness and perceptual quality provides a more comprehensive evaluation signal.

## 6 Conclusion

We introduced VGIF-Score, an interpretable and diagnostic framework for evaluating spatio-temporal instruction following in video generation. By combining ST-DAG-based objective completion with instruction-conditioned AutoRubric assessment, VGIF-Score measures both structural correctness and perceptual satisfaction while localizing where failures occur. We further built VGIF-Bench, a dependency-rich benchmark of 223 long-form prompts and approximately 4.3K fine-grained evaluation items, designed to evaluate multi-entity interactions, state transitions, and causal event chains. Experiments on 14 proprietary and open-source VGMs show that current models still struggle with deep instruction following, especially under causal chains, deep dependency structures, and late-position constraints. Human validation further supports the reliability of the VLM evaluator and the necessity of the dual-branch design. We hope our work can support more diagnostic evaluation and guide future video generation models toward stronger semantic and causal instruction following.

## 7 Acknowledgements

This work was supported by the National Nature Science Foundation of China (Grant U23B2052, 62225601, 62476029) and the Beijing Key Laboratory of Multimodal Data Intelligent Perception and Governance.

## References

1. ai, S., Teng, H., Jia, H., Sun, L., Li, L., Li, M., Tang, M., Han, S., Zhang, T., Zhang, W.Q., Luo, W., Kang, X., Sun, Y., Cao, Y., Huang, Y., Lin, Y., Fang, Y., Tao, Z., Zhang, Z., Wang, Z., Liu, Z., Shi, D., Su, G., Sun, H., Pan, H., Wang, J., Sheng, J., Cui, M., Hu, M., Yan, M., Yin, S., Zhang, S., Liu, T., Yin, X., Yang, X., Song, X., Hu, X., Zhang, Y., Li, Y.: Magi-1: Autoregressive video generation at scale (2025), https://arxiv.org/abs/2505.13211

2. Chen, J., Zhou, Z., Tong, Y., Chang, D., Luo, Y., Ma, Z.: Seeing as experts do: A knowledge-augmented agent for open-set fine-grained visual understanding. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 41446–41455 (2026)

3. Chen, Y., Guo, X., Shi, Z., Song, Z., Zhang, J.: T2vworldbench: A benchmark for evaluating world knowledge in text-to-video generation. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 6474–6485 (2026)

4. Cho, J., Hu, Y., Baldridge, J.M., Garg, R., Anderson, P., Krishna, R., Bansal, M., Pont-Tuset, J., Wang, S.: Davidsonian scene graph: Improving reliability in fine-grained evaluation for text-to-image generation. In: The Twelfth International Conference on Learning Representations

5. Cohen, J.: A coeficient of agreement for nominal scales. Educational and psychological measurement 20(1), 37–46 (1960)

6. Deng, H., Pan, T., Zhang, F., Liu, Y., Luo, Z., Cui, Y., Shen, C., Shan, S., Zhang, Z., Wang, X.: Uniform discrete difusion with metric path for video generation. arXiv preprint arXiv:2510.24717 (2025)

7. Feng, W., Li, J., Saxon, M., Fu, T.j., Chen, W., Wang, W.Y.: Tc-bench: Benchmarking temporal compositionality in text-to-video and image-to-video generation. arXiv preprint arXiv:2406.08656 (2024)

8. Gao, Y., Chang, D., Yu, B., Qin, H., Diao, M., Chen, L., Liang, K., Ma, Z.: Toward generalizable forgery detection and reasoning. IEEE Transactions on Image Processing 35, 3395–3410 (2026)

9. Gao, Y., Lin, W., Xu, J., Xu, W., Chen, P.: Self-supervised adversarial training for robust face forgery detection. In: BMVC. p. 718 (2023)

10. Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. Advances in neural information processing systems 27 (2014)

11. Google DeepMind: Gemini 3.1 pro (2026), https://deepmind.google/models/ model-cards/gemini-3-1-pro/

12. HaCohen, Y., Brazowski, B., Chiprut, N., Bitterman, Y., Kvochko, A., Berkowitz, A., Shalem, D., Lifschitz, D., Moshe, D., Porat, E., et al.: Ltx-2: Eficient joint audio-visual foundation model. arXiv preprint arXiv:2601.03233 (2026)

13. Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., Choi, Y.: Clipscore: A referencefree evaluation metric for image captioning. In: Proceedings of the 2021 conference on empirical methods in natural language processing. pp. 7514–7528 (2021)

14. Ho, J., Salimans, T., Gritsenko, A., Chan, W., Norouzi, M., Fleet, D.J.: Video difusion models. Advances in neural information processing systems 35, 8633– 8646 (2022)

15. Huang, Z., He, Y., Yu, J., Zhang, F., Si, C., Jiang, Y., Zhang, Y., Wu, T., Jin, Q., Chanpaisit, N., et al.: Vbench: Comprehensive benchmark suite for video generative models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 21807–21818 (2024)

16. Huang, Z., Zhang, F., Xu, X., He, Y., Yu, J., Dong, Z., Ma, Q., Chanpaisit, N., Si, C., Jiang, Y., et al.: Vbench++: Comprehensive and versatile benchmark suite for video generative models. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

17. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013)

18. Li, B., Lin, Z., Pathak, D., Li, J., Fei, Y., Wu, K., Ling, T., Xia, X., Zhang, P., Neubig, G., et al.: Genai-bench: Evaluating and improving compositional text-tovisual generation. arXiv preprint arXiv:2406.13743 (2024)

19. Ling, X., Zhu, C., Wu, M., Li, H., Feng, X., Yang, C., Hao, A., Zhu, J., Wu, J., Chu, X.: Vmbench: A benchmark for perception-aligned video motion generation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 13087–13098 (2025)

20. Liu, J., Han, J., Yan, B., Wu, H., Zhu, F., Wang, X., Jiang, Y., Peng, B., Yuan, Z.: Infinitystar: Unified spacetime autoregressive modeling for visual generation (2025), https://arxiv.org/abs/2511.04675

21. Liu, Y., Cun, X., Liu, X., Wang, X., Zhang, Y., Chen, H., Liu, Y., Zeng, T., Chan, R., Shan, Y.: Evalcrafter: Benchmarking and evaluating large video generation models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 22139–22149 (2024)

22. OpenAI: Gpt-5.2. https://openai.com/index/introducing-gpt-5-2/ (2025), large language model

23. Qin, H., Chang, D., Gao, Y., Tan, Y., Chen, L., Ma, Z.: Increfa: Breaking the static wall of generative model attribution. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 35405–35415 (2026)

24. Qin, H., Chang, D., Gao, Y., Yu, B., Chen, L., Ma, Z.: Multimodal conditional information bottleneck for generalizable ai-generated image detection. arXiv preprint arXiv:2505.15217 (2025)

25. Spearman, C.: The proof and measurement of association between two things. (1961)

26. Sun, K., Huang, K., Liu, X., Wu, Y., Xu, Z., Li, Z., Liu, X.: T2v-compbench: A comprehensive benchmark for compositional text-to-video generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 8406–8416 (2025)

27. Tang, Z., Wang, Z., Peng, B., Dong, J.: Clip-agiqa: Boosting the performance of ai-generated image quality assessment with clip. In: International Conference on Pattern Recognition. pp. 48–61. Springer (2024)

28. Tang, Z., Yang, S., Peng, B., Wang, Z., Dong, J.: Revisiting mllm based image quality assessment: Errors and remedy. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 9475–9483 (2026)

29. Tang, Z., Yang, S., Wang, Z., Peng, B., Li, Y., Dong, B., Dong, J.: Endogenous reprompting: Self-evolving cognitive alignment for unified multimodal models. arXiv preprint arXiv:2601.20305 (2026)

30. Team, G.: Mochi 1. https://github.com/genmoai/models (2024)

31. Team, K., Chen, J., Ci, Y., Du, X., Feng, Z., Gai, K., Guo, S., Han, F., He, J., He, K., et al.: Kling-omni technical report. arXiv preprint arXiv:2512.16776 (2025)

32. Team, M.L., Cai, X., Huang, Q., Kang, Z., Li, H., Liang, S., Ma, L., Ren, S., Wei, X., Xie, R., et al.: Longcat-video technical report. arXiv preprint arXiv:2510.22200 (2025)

33. Tong, H., Wang, Z., Chen, Z., Ji, H., Qiu, S., Han, S., Geng, K., Xue, Z., Zhou, Y., Xia, P., et al.: Mj-video: Fine-grained benchmarking and rewarding video preferences in video generation. arXiv preprint arXiv:2502.01719 (2025)

34. Unterthiner, T., Van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., Gelly, S.: Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717 (2018)

35. Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.W., Chen, D., Yu, F., Zhao, H., Yang, J., et al.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)

36. Wang, J., Deng, H., Pan, T., Liu, Y., Wang, C., Zhang, F., Qi, Y., Wang, X.: Udmgrpo: Stable and eficient group relative policy optimization for uniform discrete difusion models. arXiv preprint arXiv:2604.18518 (2026)

37. Wang, X., Xu, S., Xiangxuan, S., Zhang, Y., Diao, M., Duan, X., Liang, K., Ma, Z., et al.: Cinetechbench: A benchmark for cinematographic technique understanding and generation. Advances in Neural Information Processing Systems 38 (2026)

38. Wang, X., Zhang, Y., Zhang, X., Yan, H., Diao, M., Xu, S., Yan, Z., Li, H., Liang, K., Ma, Z.: Detailverifybench: A benchmark for dense hallucination localization in long image captions (2026), https://arxiv.org/abs/2604.05623

39. Wu, B., Zou, C., Li, C., Huang, D., Yang, F., Tan, H., Peng, J., Wu, J., Xiong, J., Jiang, J., et al.: Hunyuanvideo 1.5 technical report. arXiv preprint arXiv:2511.18870 (2025)

40. Xie, L., Huang, S., Zhang, Z., Zou, A., Zhai, Y., Ren, D., Zhang, K., Hu, H., Liu, B., Chen, H., et al.: Auto-rubric: Learning from implicit weights to explicit rubrics for reward modeling. arXiv preprint arXiv:2510.17314 (2025)

41. Yan, W., Zhang, Y., Abbeel, P., Srinivas, A.: Videogpt: Video generation using vq-vae and transformers. arXiv preprint arXiv:2104.10157 (2021)

42. Yang, S., Zhong, H., Zhang, R., Zhao, X., Li, S., Zheng, K., Yang, X., Wang, Z., Tang, Z., Li, Y., Gu, B., Peng, Z., Huang, Y., Luo, M., Bo, Y., Feng, D., Zhang, Y., Ma, J., Wang, R., Zhang, L., Guo, Y., Guan, F., Agrawala, M., Fu, H., Zhao, A., Rao, A.: Evalverse: Pipeline-aware and expert-calibrated benchmarking for professional cinematic video generation (2026), https://arxiv.org/abs/2605. 23271

43. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., et al.: Cogvideox: Text-to-video difusion models with an expert transformer. arXiv preprint arXiv:2408.06072 (2024)

44. Yuan, S., Huang, J., Xu, Y., Liu, Y., Zhang, S., Shi, Y., Zhu, R., Cheng, X., Luo, J., Yuan, L.: Chronomagic-bench: A benchmark for metamorphic evaluation of text-to-time-lapse video generation. Advances in Neural Information Processing Systems 37, 21236–21270 (2024)

45. Zheng, D., Huang, Z., Liu, H., Zou, K., He, Y., Zhang, F., Gu, L., Zhang, Y., He, J., Zheng, W.S., et al.: Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755 (2025)