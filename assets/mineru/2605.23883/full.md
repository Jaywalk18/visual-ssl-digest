# PGT: Procedurally Generated Tasks for improving visual grounding in MLLMs

Rim Assouel 1 2 3 Amir Bar 3 Michal Drozdzal 3 \* Adriana Romero-Soriano 1 3 4 5 \*

# Abstract

Despite remarkable progress in Multimodal Large Language Models (MLLMs), these models still struggle with fine-grained understanding tasks. In this work, we propose Procedurally Generated Tasks (PGT), a simple data-driven framework that serves a dual purpose: inducing fine-grained visual understanding and acting as a low-cost diagnostic tool to identify the source of perception failures. By overlaying unambiguous geometric primitives on images, PGT generate additional dense supervision that disentangles visual grounding capability from semantic priors. Extensive experiments on relational, quantitative, and 3D/depth understanding benchmarks show that PGT yields remarkable gains across diverse architectures. Instruction tuning MLLMs on LLaVA-v1.5-Instruct augmented with PGT data results in improvements of up to +20% on the What’sUp benchmark and +13.3% on CV-Bench-2D, while maintaining general perception capabilities. Moreover, finetuning state-of-the-art MLLMs on PGT data leads to boosts of up to +5.5% on What’sUp and +8.3% on CV-Bench-2D. These findings demonstrate that PGT effectively address the bottleneck of fine-grained perception, revealing that many spatial reasoning deficits stem from inadequate supervision signals rather than inherent architectural or resolution limitations.

# 1. Introduction

Multimodal Large Language Models (MLLMs) have achieved remarkable proficiency (Bai et al., 2025; Zhu et al., 2025; Li et al., 2024; Tong et al., 2024; Yu et al., 2025; Huang et al., 2025) in high-level semantic tasks, 1Mila - Quebec AI Institute ´ 2Universite de Montr ´ eal ´ 3FAIR at Meta Superintelligence Labs 4McGill University 5Canada CIFAR AI Chair. Correspondence to: Rim Assouel <assouelr@mila.quebec>, Adriana Romero-Soriano <adrianars@meta.com>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

spanning image captioning, visual question answering, and open-ended dialogue. However, these strong general perception capabilities often mask a fundamental deficiency in fine-grained spatial intelligence tasks (Liu et al., 2025; Tong et al., 2024; Fu et al., 2024) that humans can solve effortlessly. While models can eloquently describe the semantic content of a scene, they frequently falter on basic compositional reasoning tasks such as precisely counting overlapping objects, determining relative depth, or distinguishing between ”left” and ”right” in complex arrangements (Assouel et al., 2025; Kamath et al., 2023).

To mitigate these deficits, recent work has largely bifurcated into two approaches. The first involves model-centric improvements, which seek to improve perception through visual representations enhancements —such as integrating higher-resolution vision encoders (Li et al., 2024), employing a mixture of Vision Transformers (ViTs) (Tong et al., 2024; Karamcheti et al., 2024) to capture multi-scale visual features, or distilling visual information from multiple experts encoders (Yoon et al., 2025). The second approach relies on data-centric scaling, either utilizing massive and costly human-annotated datasets or implementing complex multi-stage training protocols (Li et al., 2025a; Sarch et al., 2025; Huang et al., 2025; Meng et al., 2025; Yu et al., 2025). Common strategies in this domain include generating thinking traces interleaved with coordinates grounding (Sarch et al., 2025) or enforcing auxiliary objectives like object enumeration. While effective, these methods incur high annotation bottlenecks and computational overhead, often overfitting to specific domain (Liu et al., 2025) distributions without fully solving the underlying grounding problem.

Meanwhile, recent work (Fu et al., 2024) demonstrates that the failure in easy vision-centric tasks does not stem from a lack of visual representation in the encoder, but rather from insufficient training incentives to override linguistic priors (Yamada et al., 2024). Current MLLMs often bypass genuine visual grounding, behaving akin to ”bag-of-words” classifiers that associate objects (e.g., ”sky”) with spatial concepts (e.g., ”up”) based on text statistics rather than grounded geometry (Sarch et al., 2025; Chen et al., 2026). This dependence is exacerbated by the ambiguity of natural image datasets (Jian et al., 2025; Yang et al., 2025), where spatial relationships are often implicit, providing a noisy signal that discourages the model from attending to the

PGT: We overlay procedurally generated geometric tasks and efficiently inject them into standard training   
![](images/aa7c59a0e00c0ba6d9a2edd75c31a5c0db076ff444b41b3f220c5c850603f38c.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Standard Dataset\nLLaVa-Instruct or Cambrian-7M\nBaseline training data"] --> B["+"]
    B --> C["Spatial Relations\nGeneral Q: &quot;Describe the street\nPGT Q: &quot;Where is red vs green?&quot;\nA: &quot;Right&quot;"]
    B --> D["Abstract Counting\nGeneral Q: &quot;What room is this?&quot;\nPGT Q: &quot;Count orange circles?&quot;\nA: &quot;3&quot;"]
    B --> E["Distance & Analogy\nGeneral Q: &quot;Where is it?&quot;\nPGT Q: &quot;Closest to D?&quot;\nA: &quot;B&quot;"]
    B --> F["Enhanced Dataset\nStandard Data + PGT\nSample-efficient training"]
    
    G["▲ language prior\nQ: Is the bowl above the table?\nModel: yes"] --> H["↓"]
    I["▲ incorrect 2D shortcut\nQ: Which object is closer to the camer.\nModel: lamp"] --> J["↑"]
    
    K["Relation\n0.6M SFT Scale\nBaseline\n7M SFT Scale +PGTs (Ours)"] --> L["90\n40-90"]
    M["Depth\n0.6M SFT Scale\nBaseline\n7M SFT Scale +PGTs (Ours)"] --> N["90\n40-90"]
    O["Count\n0.6M SFT Scale\nBaseline\n7M SFT Scale +PGTs (Ours)"] --> P["50-90"]
```
</details>

Figure 1. Overview of PGT. Top: The construction of our procedurally generated data to augment instruction tuning training datasets. Abstract geometric primitives are overlaid to training data, when available. Bottom: (Left) Examples of failure modes in fine-grained relational and spatial understanding of state-of-the-art MLLMs. In the first example the model can rely on the fact that a bowl is usually on a table and in the second example it can rely on the shortcut where object higher up as usually further from the camera (Right) PGT performance boosts in relational, quantitative, and 3D/depth understanding over the baseline when using different instruction tuning dataset sizes.

visual latent space.

A theoretically efficient way to force a model to ignore real-world induced correlations (e.g., ”sky is blue” or ”ceiling is up”) is to train on fully synthetic, abstract visual reasoning tasks where such priors do not exist. However, given the immense capacity of SOTA models, it remains unclear if this efficiency comes at the cost of relevance. We ask a fundamental question: Can fine-grained spatial skills efficiently transfer from abstract, non-ambiguous objects to real-world scenarios?. To answer this, we introduce Procedurally Generated Tasks (PGT), a set of synthetic tasks that use unambiguous geometric primitives—such as colored bounding boxes, labeled points, and counting markers—overlaid directly onto existing training images (see Figure 1). Unlike methods relying on costly human annotation, PGT automatically generates data with dense, verifiable low-cost supervision. By forcing the model to reason about abstract primitives where language priors do not apply (e.g., a green box has no semantic correlation with being ”above” or ”under” a red box), we compel the visual attention mechanism to engage with the actual image geometry, testing the hypothesis that these abstract skills unlock better grounding in complex semantic environments.

By training MLLMs on PGT data, we demonstrate a surprising phenomenon: skills learned from PGT’s abstract geometric overlays transfer effectively to real-world semantic objects. For example, we show that a model trained to estimate the distance between two abstract points can better estimate the relative depth of multiple objects in a natural

image. Through extensive experimentation, we demonstrate that PGT’s simple, low-cost intervention yields state-of-theart improvements across 11 diverse benchmarks. Augmenting a Llama-3-8B-based (Grattafiori et al., 2024) MLLM with PGTs yields a +19.6% gain on the What’s Up (Kamath et al., 2023) spatial benchmark and +13.3% on CV-Bench 2D (Tong et al., 2024). Most notably, we observe emergent 3D capabilities, with improvements of up to +9.7% on CV-Bench 3D, despite the training signal being purely 2D. This indicates the possibility of a shared distance estimation circuit between the 2D and 3D settings. Our contributions are summarized as follows:

• A Suite of Procedurally Generated Tasks: We propose PGT, a set of abstract geometric tasks designed to disentangle visual signals from linguistic priors. (Section 3)   
• Controlled Multimodal Training across Diverse Backbones: We demonstrate the generalizability of our method through a rigorous study across four distinct LLM backbones. The consistent gains suggest that PGT addresses a fundamental modality alignment bottleneck common to a wide range of architectures. (Section 4.1)   
• Impact on SOTA & Diagnostic Value: We push the state-of-the-art by fine-tuning advanced models (including Qwen-2.5-VL and InternVL3), showing that PGT yields significant improvements even on top of strong pre-training. Furthermore, we position PGT as an complementary diagnostic signal: its ability to

improve performance without architectural changes suggests it can serve as a baseline to justify complex model-centric improvements, or be combined with them for additive gains. (Section 4.2)

# 2. Related Work

Causes of Finegrained Understanding Limitations in MLLMs. Recent work (Liu et al., 2025) attributes finegrained spatial understanding limitations of MLLMs to three primary root causes spanning architecture, training pipelines, and data. Covert et al. (2025); Chen et al. (2025); Assouel et al. (2025); Tschannen et al. (2025); Ranzinger et al. (2025) underline the impact of leveraging vision backbones (e.g., CLIP (Radford et al., 2021)) optimized for global semantic alignment on the ability of MLLMs to preserve the spatial structure needed for downstream reasoning. Fu et al. (2025); Dorkenwald et al. (2024) argue for finegrained supervision and explicit spatial grounding during training. Yet, there is a lack of large-scale, high-quality datasets annotated with explicit spatial relations (Kamath et al., 2023; Liu et al., 2025). Unlike semantic pre-training data, spatially-grounded data are expensive to annotate and often restricted to synthetic domains or limited relations. Determining the root cause of the bottleneck for a specific task typically requires expensive ablation studies or architectural modifications. PGT offers a cost-efficient alternative to verify whether heavy modifications (e.g., replacing the encoder or retraining with RL) are actually justified, or if the model simply lacks a clear training signal to unlock its spatial understanding capabilities.

Improving Finegrained Understanding in MLLMs. Efforts have been devoted to enhancing vision backbones architectures and training objectives, either through self-supervised post-training objectives (Covert et al., 2025) that encourage the recovery of local semantic details or through the aggregation of features from multiple expert encoders (e.g., DINOv2, CLIP) via specialized connectors or distillation objectives, explicitly fusing spatially rich representations with semantic embeddings to support finegrained tasks. Efforts have also been devoted to devising training-free approaches such as ViperGPT (Sur´ıs et al., 2023) that enhance both interpretability and performance of the models by decomposing complex visual tasks into structured sequences of subgoals. By contrast, hybrid prompting strategies integrate explicit spatial markers into the reasoning stream. For example, Lei et al. (2024); Izadi et al. (2025) propose to overlay coordinate matrices or visual cues directly onto images while embedding corresponding references in the text, thereby bridging the modality gap. Similarly, Li et al. (2025b); Wu et al. (2024) elicit spatial reasoning of MLLMs by visualizing their reasoning traces. Another line of work (Zhou et al., 2024; Hu et al., 2024) extends the Visualization-of-Thought prompting idea to be used during inference, enabling the model to further process the visual input to support its reasoning. Finally, the focus has most recently shifted to inducing robust spatial reasoning behaviors—such as backtracking, reflection, and sequential processing—via reinforcement learning (RL). Following this line, ViGoRL (Sarch et al., 2025) employs a visually grounded RL framework with Monte Carlo Tree Search to generate reasoning traces which anchor every step to specific image coordinates, effectively forcing the model to ”point” to evidence and backtrack when necessary. SpatialLadder (Li et al., 2025a) adopts a progressive training curriculum – using RL – to transition models from basic perception to complex multi-step reasoning. Parallel to this, Visual Jigsaw (Wu et al., 2025) leverages RL from verifiable rewards (RLVR) on temporal and spatial ordering tasks, encouraging the model to develop sequential processing capabilities and structural awareness without reliance on expensive human annotations.

# 3. Procedurally Generated Tasks

To address the limitations of MLLMs in fine-grained visioncentric reasoning, we introduce procedurally generated tasks (PGT), a set of tasks that utilize procedurally generated geometric overlays to provide dense, verifiable supervision signals to improve MLLMs. Unlike traditional data augmentation, our PGT uses these overlays as pretext tasks that require the model to perform precise spatial and quantitative reasoning without augmenting the total number of samples.

# 3.1. Taxonomy of PGT

We design three core tasks to target specific reasoning deficits observed in current MLLMs.

Spatial Relationship Understanding. We overlay two distinct geometric primitives—typically a green and a red bounding box—onto a real-world image. We prompt the model to solve two sub-tasks: (1) relative positioning – determining the spatial relationship of one box relative to the other (e.g., ”above”, ”below”, ”left”, ”right”); and (2) coordinate regression – providing the normalized coordinates of a specific box to enforce precise localization. See Figure 1 (top) for an example.

Counting. To decouple counting ability from semantic object recognition, we project a variable number of semi-transparent colored circles (e.g., orange, purple, green, etc) onto the scene. We prompt the model to identify and count only the circles of the requested color, ignoring both the background objects and circles of other colors. See Figure 1 (top) for an example.

Analogy and Relative Distance estimation. We utilize multi-point overlays labeled with identifiers (e.g., A, B, C, D) and prompt the model to solve two sub-tasks : (1)

distance estimation – identifying which labeled point is closest to a target point (e.g., ”Which one is closest to D: A, B, or C?”); and (2) simple analogy – identifying circles that share the same properties, (e.g., color) across different locations in the image. See Figure 1 (top) for an example.

# 3.2. Procedural Generation and Overlaying

A primary advantage of our PGT is that the entire generation pipeline is automated and requires no human annotation. Given an image I from an existing training dataset, and its associated set of question-answer pairs $\begin{array} { r } { { \cal { S } } = \{ ( q _ { i } , a _ { i } ) \} _ { i = 1 } ^ { n } , } \end{array}$ we augment each data sample $X = ( I , S )$ in the training set through the following procedure:

1. Task Selection: We uniformly sample a task from the previously defined PGT taxonomy (e.g., spatial relationship understanding, counting, or analogy and relative distance estimation). Unless stated otherwise, we sample a task for each data point. We further ablate this design choice in section 5.3.   
2. Rendering: We sample the necessary geometric primitives P (e.g., bounding boxes, colored circles, or labeled points) with randomized parameters for location, color, and scale. We render these primitives in an overlay layer L and produce a modified image $I _ { P G T } = I$ ⊕ L, where ⊕ denotes overlay.   
3. Procedural QA Generation: We procedurally generate a new question-answer pair $( Q _ { P G T } , A _ { P G T } )$ based on the ground-truth geometric properties of P.   
4. Sample Integration: We append the newly generated question-answer pair to the original set of pairs S, resulting in an augmented question-answer pairs set $\mathcal { S } _ { a u g } = S \cup \{ ( Q _ { P G T } , A _ { P G T } ) \}$ , and a PGT-augmented data sample $X _ { a u g } = ( I _ { P G T } , S _ { a u g } )$ .

This approach preserves the original semantic integrity of the image while introducing a layer of abstract reasoning at a negligible computational overhead. It is worth noting, that when having access to real data, leveraging the overlay mechanism to alter the real data avoids augmenting the dataset size—which would necessitate more training iterations. Therefore, when real data is available, we apply these tasks as a targeted augmentation to existing datasets, such as LLaVA-Instruct-1.5 (Liu et al., 2023b). By projecting abstract primitives directly onto real-world scenes, we compel the model’s attention mechanism to disentangle the ”base” semantic layer from the ”abstract” geometric layer. This prevents the model from over-relying on linguistic priors (e.g., assuming a ”sky” token implies an ”above” relationship) and forces it to ground its reasoning in specific visual tokens because there is no shortcut to rely on. Furthermore, because the tasks are procedurally generated, we ensure that the objects of interest and their spatial configurations remain non-ambiguous and verifiable. We give further details and exact template in Appendix A.

# 4. Results

In this section, we evaluate the performance of PGT in different settings. First, we assess the impact of PGT when building MLLMs and using PGT in their instruction tuning phase. This is done in a controlled setting in subsection 4.1. Then, we validate the potential of PGT when finetuning state-of-the-art MLLMs in subsection 4.2. To align our evaluation with the core focus of this work, our primary analysis targets metrics that isolate fine-grained visual understanding and spatial grounding. Specifically, we evaluate our models using a diverse range of benchmarks spanning relational understanding (What’s up-COCO, CV-Bench-2D, VSR), vision-centric and 3D understanding (CV-Bench 3D, TallyQA, MMVP, RealworldQA, and single image perception subsets of BLINK, namely spatial, depth, relational, localization. Finally, we include general image understanding benchmarks (MMSTAR, GQA, POPE, SEED, AI2D) as a sanity check to ensure that our synthetic intervention effectively overrides linguistic priors without degrading the models’ broader semantic understanding capabilities.

# 4.1. PGT for instruction tuning

We investigate the utility of our PGT data for instruction tuning of MLLMs, i.e., we assume access to a pre-trained vision backbone as well as a pre-trained large language model (LLM) backbone and build a MLLM by adding an adapter between the 2 backbones. We follow the 2 training stages of Karamcheti et al. (2024); Tong et al. (2024); Li et al. (2024): we first pre-train the adapter for captioning alignment and then intruction-tune both the adapter and the LLM backbone, while keeping the vision backbone frozen. We use PGT data for instruction tuning.

Experimental Setup. We use CLIP-ViT-L/14@336px (Radford et al., 2021) as vision backbone, and consider the following LLM backbones: Vicuna-1.5-7binstruct (Liu et al., 2023b), Llama3-8b-Instruct (Grattafiori et al., 2024), Qwen2.5-7b-Instruct (Qwen et al., 2025) and Qwen2.5-14b-Instruct (Qwen et al., 2025). We utilize the LLaVA-v1.5-Instruct dataset (Liu et al., 2023b) as our primary semantic training data source and augment it with PGT for instruction tuning. All MLLM are trained for 1 epoch for captioning alignment (adapter only) followed by 1 or 2 epochs for instruction tuning (adapter and LLM). Similar to Liu et al. (2023b); Karamcheti et al. (2024), we use a cosine learning rate schedule with a 0.1 warmup period, a global batch size of 128 and a learning rate of $2 e ^ { - } 5 .$ . To evaluate the impact of our proposed PGT, we compare models trained on the original dataset against those augmented with our tasks. As described in section 3, the augmented dataset maintains the exact same number of training samples as the original dataset, with PGT data injected directly into the existing visual-instruction pairs. Exact hyperparameters for this section are given in Appendix B.1.

Table 1. Quantitative comparison of the performance achieved by instruction tuned MLLMs across diverse benchmarks. MLLMs are built leveraging a pre-trained vision backbone and varying pre-trained base LLM backbones. For each model family, we compare the baseline instruction tuning protocol (prismatic training) against the PGT-augmented (+ PGT) variant. Results demonstrate that PGT yields remarkable improvements in spatial, fine-grained, and 3D reasoning without degrading general perception. All values are accuracies (%). 

<table><tr><td rowspan="2">BASE LLM</td><td colspan="3">RELATIONAL REASONING</td><td colspan="5">VISION-CENTRIC / 3D</td><td colspan="5">GENERAL PERCEPTION</td></tr><tr><td>W-UP</td><td>CV-2D</td><td>VSR</td><td>CV-3D</td><td>TALLY</td><td>MMVP</td><td>RWQA</td><td>BLINK</td><td>MMSTAR</td><td>GQA</td><td>POPE</td><td>SEED</td><td>AI2D</td></tr><tr><td>VICUNA-1.5-7B</td><td>75.9</td><td>55.8</td><td>54.7</td><td>58.4</td><td>63.7</td><td>26.0</td><td>53.5</td><td>49.2</td><td>32.7</td><td>64.7</td><td>88.0</td><td>62.1</td><td>52.4</td></tr><tr><td>+ PGT</td><td>95.9</td><td>68.1</td><td>68.8</td><td>67.1</td><td>66.0</td><td>28.0</td><td>54.9</td><td>57.3</td><td>35.3</td><td>64.1</td><td>87.6</td><td>64.3</td><td>54.5</td></tr><tr><td>Improvement</td><td>+20.0</td><td>+12.3</td><td>+14.1</td><td>+8.7</td><td>+2.3</td><td>+2.0</td><td>+1.4</td><td>+8.1</td><td>+2.6</td><td>-0.6</td><td>-0.4</td><td>+1.2</td><td>+2.1</td></tr><tr><td>LLAMA-3-8B</td><td>77.8</td><td>58.5</td><td>63.8</td><td>61.1</td><td>63.4</td><td>31.3</td><td>59.0</td><td>58.0</td><td>38.7</td><td>67.6</td><td>82.3</td><td>65.7</td><td>57.3</td></tr><tr><td>+ PGT</td><td>97.4</td><td>71.8</td><td>72.3</td><td>70.8</td><td>66.4</td><td>33.3</td><td>58.9</td><td>62.8</td><td>39.8</td><td>67.1</td><td>83.7</td><td>66.3</td><td>57.5</td></tr><tr><td>Improvement</td><td>+19.6</td><td>+13.3</td><td>+8.5</td><td>+9.7</td><td>+3.0</td><td>+1.2</td><td>-0.1</td><td>+4.8</td><td>+1.1</td><td>-0.5</td><td>+1.4</td><td>+0.6</td><td>+0.2</td></tr><tr><td>QWEN-2.5-7B</td><td>91.8</td><td>60.8</td><td>72.9</td><td>65.3</td><td>67.0</td><td>38.0</td><td>54.9</td><td>58.9</td><td>40.7</td><td>60.0</td><td>79.7</td><td>66.2</td><td>62.0</td></tr><tr><td>+ PGT</td><td>96.3</td><td>72.7</td><td>73.2</td><td>73.6</td><td>70.9</td><td>40.7</td><td>55.8</td><td>62.2</td><td>41.5</td><td>59.2</td><td>79.4</td><td>67.9</td><td>63.7</td></tr><tr><td>Improvement</td><td>+4.5</td><td>+11.9</td><td>+0.3</td><td>+8.3</td><td>+3.9</td><td>+2.7</td><td>+0.9</td><td>+3.3</td><td>+0.8</td><td>-0.8</td><td>-0.3</td><td>+1.7</td><td>+1.7</td></tr><tr><td>QWEN-2.5-14B</td><td>89.1</td><td>64.5</td><td>68.9</td><td>63.1</td><td>70.6</td><td>34.0</td><td>58.2</td><td>61.6</td><td>41.3</td><td>59.1</td><td>86.1</td><td>65.9</td><td>64.7</td></tr><tr><td>+ PGT</td><td>98.3</td><td>74.6</td><td>75.2</td><td>73.1</td><td>69.5</td><td>43.3</td><td>58.2</td><td>63.8</td><td>43.1</td><td>62.0</td><td>88.0</td><td>69.1</td><td>66.6</td></tr><tr><td>Improvement</td><td>+9.2</td><td>+10.1</td><td>+6.3</td><td>+10.0</td><td>-0.9</td><td>+9.3</td><td>+0.0</td><td>+2.2</td><td>+1.8</td><td>+2.9</td><td>+1.9</td><td>+3.2</td><td>+1.9</td></tr></table>

Results. Table 1 presents the results of leveraging PGT data across diverse LLM backbones. As shown in the table, PGT yields substantial performance gains in relational reasoning benchmarks as well as vision-centric benchmarks that require precise spatial, relational and quantitative reasoning. The most notable improvements are observed in the What’s Up benchmark, where VLMs based on the Vicuna-1.5-7B and Llama-3-8B models demonstrate absolute increases of +20.0% and +19.6% respectively. This improvement of spatial relationship understanding extends to CV-Bench-2D, which witnesses a consistent boost ranging from +10.1% to +13.3% across all tested LLM backbones, suggesting that the localization skills learned from abstract geometric primitives transfer effectively to real-world object relationships.

Furthermore, although the PGT tasks are formulated in 2D, we observe an emergent improvement in 3D and finegrained vision-centric perception; with absolute gains ranging from +8.3% to +10.0% on CV-Bench-3D and from +2.2% to +8.1% on BLINK. We further confirm and ablate that general 2D relative distances comparisons directly help with relative depth estimation of real world objects. We further identified heuristic failures in baseline models (see Appendix C) and hypothesize that PGT are effective by reinforcing a possible shared distance estimation circuit between the 2D and 3D setting. Finally while specialized benchmarks like VSR show sharp increases for the Vicuna baseline (+14.1%), the framework also proves beneficial for quantitative reasoning, as evidenced by a +3.9% improvement on TallyQA for the Qwen-2.5-7B backbone. Critically, these enhancements do not compromise the models’ foundational capabilities; performance on general perception benchmarks such as MMSTAR remains stable or improves, while scores on general tasks like GQA and POPE fluctuate minimally, confirming that PGT acts as a complementary, effective training signal that improves fine-grained image understanding without harming broader semantic and general understanding.

# 4.2. Finetuning state-of-the-art MLLMs with PGT

In this subsection, we aim to showcase that PGT finetuning on state-of-the-art MLLMs is an effective way to improve their fine-grained visual understanding. To do so, we use PGT to finetune a suite of state-of-the-art MLLMs, including Qwen-2.5-VL (3B, 7B) (Bai et al., 2025), LLaVA-Next (Vicuna-1.5-7B, Llama3-8B) (Li et al., 2025a), and InternVL3-8B (Zhu et al., 2025). In this case, to avoid assumptions about real data access, we built PGT on uniformly gray images.

Experimental Setup. We conduct controlled supervised fine-tuning experiments where each model is trained for 2 epochs with a learning rate of $1 \times 1 0 ^ { - 4 }$ . For these experiments, we utilize LoRA-based (Hu et al., 2021) fine-tuning on a set of 5k PGT samples rendered on neutral gray backgrounds; we do not use any real data and assume that all images I are uniformly gray and S = ∅. Similar to our previous experimental settting 4.1, we uniformly sample a PGT task for each of the 5k samples. We use the same evaluation benchmarks as in subsection 4.1. Exact hyperparameters for this section are given in Appendix B.2.

Table 2. Quantitative comparison of the impact of finetuning state-of-the-art MLLMs with PGT-augmented data over baselines and specialized methods. Results demonstrate that PGT effectively improves the baselines’ performance while being competitive or improving specialized methods. Accuracies are reported in (%).∗ means that the result is reported from original paper and not recomputed with our codebase. 

<table><tr><td rowspan="2">MLLM</td><td colspan="3">RELATIONAL REASONING</td><td colspan="5">VISION-CENTRIC / 3D</td><td colspan="5">GENERAL PERCEPTION</td></tr><tr><td>W-UP</td><td>VSR</td><td>CV-2D</td><td>CV-3D</td><td>TALLY</td><td>RWQA</td><td>BLINK</td><td>MMVP</td><td>SEED</td><td>MMSTAR</td><td>GQA</td><td>AI2D</td><td>POPE</td></tr><tr><td colspan="14">Improving SOTA Backbones</td></tr><tr><td>LLAVA-NEXT-7B</td><td>85.2</td><td>64.7</td><td>59.5</td><td>51.6</td><td>45.2</td><td>58.4</td><td>55.4</td><td>32.7</td><td>63.4</td><td>34.8</td><td>66.7</td><td>64.5</td><td>87.8</td></tr><tr><td>+ PGT</td><td>90.7</td><td>70.9</td><td>67.8</td><td>59.9</td><td>60.1</td><td>60.1</td><td>58.8</td><td>38.7</td><td>64.3</td><td>37.1</td><td>66.5</td><td>64.5</td><td>86.1</td></tr><tr><td>LLAVA-NEXT-LLAMA3-8B</td><td>93.8</td><td>71.8</td><td>62.9</td><td>68.3</td><td>59.2</td><td>59.5</td><td>63.1</td><td>38.7</td><td>68.1</td><td>41.5</td><td>69.3</td><td>71.5</td><td>87.6</td></tr><tr><td>+ PGT</td><td>93.8</td><td>71.8</td><td>65.7</td><td>77.3</td><td>63.9</td><td>59.5</td><td>64.2</td><td>42.0</td><td>67.2</td><td>45.3</td><td>68.1</td><td>71.3</td><td>86.6</td></tr><tr><td>INTERNVL3-8B</td><td>97.2</td><td>85.2</td><td>81.4</td><td>85.7</td><td>77.2</td><td>65.2</td><td>74.6</td><td>60.7</td><td>76.4</td><td>68.1</td><td>66.8</td><td>83.8</td><td>91.0</td></tr><tr><td>+ PGT</td><td>97.9</td><td>85.3</td><td>82.5</td><td>86.0</td><td>77.4</td><td>68.5</td><td>74.5</td><td>62.7</td><td>77.0</td><td>68.5</td><td>68.3</td><td>84.0</td><td>90.8</td></tr><tr><td colspan="14">PGT vs. Specialized Data &amp; Models</td></tr><tr><td>QWEN2.5-VL-3B</td><td>93.8</td><td>80.4</td><td>70.9</td><td>71.5</td><td>66.4</td><td>59.0</td><td>67.6</td><td>37.3</td><td>73.7</td><td>55.3</td><td>64.9</td><td>79.2</td><td>87.5</td></tr><tr><td>+ PGT</td><td>96.1</td><td>84.0</td><td>74.4</td><td>79.3</td><td>67.5</td><td>62.9</td><td>68.4</td><td>45.3</td><td>74.2</td><td>56.5</td><td>65.4</td><td>78.1</td><td>87.5</td></tr><tr><td>+ SPECIALIZED MIX</td><td>93.7</td><td>82.8</td><td>70.8</td><td>79.9</td><td>63.4</td><td>62.7</td><td>71.5</td><td>42.0</td><td>74.4</td><td>56.9</td><td>65.2</td><td>79.6</td><td>87.6</td></tr><tr><td>VIGORL-3B</td><td>96.2</td><td>74.1</td><td>78.0</td><td>77.3</td><td>64.8</td><td>47.3</td><td>65.0</td><td>44.7</td><td>-</td><td>37.9</td><td>50.7</td><td>71.2</td><td>86.1</td></tr><tr><td>SPATIAL-LADDER-3B</td><td>88.8</td><td>60.4</td><td>72.4*</td><td>74.9*</td><td>43.1</td><td>52.5</td><td>58.6</td><td>48.1</td><td>68.2</td><td>48.9</td><td>52.0</td><td>73.5</td><td>84.7</td></tr><tr><td>QWEN2.5-VL-7B</td><td>96.8</td><td>83.8</td><td>77.7</td><td>83.5</td><td>72.4</td><td>67.5</td><td>72.8</td><td>53.3</td><td>76.4</td><td>62.2</td><td>65.8</td><td>82.7</td><td>87.4</td></tr><tr><td>+ PGT</td><td>98.0</td><td>85.7</td><td>78.2</td><td>84.6</td><td>73.5</td><td>69.3</td><td>74.8</td><td>54.0</td><td>76.4</td><td>63.5</td><td>67.1</td><td>83.2</td><td>88.2</td></tr><tr><td>+ SPECIALIZED MIX</td><td>96.4</td><td>85.5</td><td>78.7</td><td>82.5</td><td>72.4</td><td>68.9</td><td>74.7</td><td>52.0</td><td>76.6</td><td>62.5</td><td>66.2</td><td>82.7</td><td>86.8</td></tr><tr><td>IMAGE JIGSAW</td><td>97.4</td><td>85.4</td><td>77.8</td><td>83.0</td><td>70.4</td><td>68.5</td><td>73.7</td><td>58.0</td><td>75.9</td><td>62.9</td><td>66.2</td><td>82.9</td><td>87.5</td></tr><tr><td>THINKLITE-VL</td><td>98.3</td><td>83.3</td><td>76.6</td><td>80.5</td><td>73.9</td><td>67.5</td><td>75.9</td><td>32.0</td><td>64.3</td><td>72.3</td><td>71.0</td><td>83.4</td><td>88.5</td></tr></table>

Results. Results are presented in Table 2. As shown in the table, the application of PGT to advanced models like Qwen-2.5-VL and InternVL3-8B consistently yields performance gains, particularly in vision-centric benchmarks that demand high-precision localization and counting. In particular, we observe that the LLaVA-Next-7B model experiences a substantial boost of +8.3% on CV-Bench-2D , +6.2% on VSR and +14.9% on TallyQA after PGT fine-tuning. Importantly, finetuning on PGT does not hurt the performance of models trained on data mixes targeting improved fine-grained understanding like InternVL3-8B, and even improves the performance of the model on most vision-centric benchmarks – e.g., with boosts +1.1% on CV-Bench-2D and +3.3% on RealWorldQA. These results indicate that the spatial primitives learned from geometric overlays are not redundant to the large-scale pre-training of state-of-the-art models but can rather provide a complementary specialized grounding signal that enhances their fine-grained perception.

A key question is whether PGT data is substantially less effective than specialized real-world data for teaching fine-grained understanding. A significant challenge in vision-centric training is the inherent noise in humanannotated datasets. To quantify the effectiveness of our PGT compared to annotated specialized datasets, we construct a Specialized Mix that mimics the taxonomy of PGT. To do so, we source samples from the training sets of TallyQA (counting), VSR (relational), and the distance-estimation subset of the Spatial-Ladder-26k dataset (Li et al., 2025a). This mix is created to be the same size as our PGT – i.e., 5k samples. We use the constructed specialized mix to finetune Qwen2.5-VL-3B and Qwen2.5-VL-7B, and compare their performance to that of the same models finetuned on

PGT-augmented data. The results of the controlled baseline trained on the above-described Specialized Mix of real data are designated as +Specialized mix in Table 2. When comparing PGT results to those of the Specialized Mix, we observe that PGT achieves higher results on the vast majority of relational reasoning, vision-centric and 3D benchmarks. In particular, even if some evaluations are indistribution w.r.t. the specialized mix, PGT leads to a +3.6% (compared to +2.4% for the specialized mix) boosts in VSR on the Qwen2.5-VL-3B backbone. Interestingly, we observe that gains related to the 2D relative distance estimation PGT are comparable to those achieved by adding real world 3D relative distance estimation data in the Specialized Mix, leading to a +7.8% boost for PGT compared to a +8.4% boost using the Specialized Mix on Qwen-2.5-VL-3B. It is worth noting that neither PGT nor the specialized mix result in excessive degradation of general perception benchmarks.

We further compare the Qwen2.5-VL models finetuned with PGT data against other specialized models, which are specifically post-trained to improve finegrained understanding. More specifically, we compare Qwen2.5-VL-3B to its ViGoRL-3B (Sarch et al., 2025) and Spatial-Ladder-3B (Li et al., 2025a) counterparts. ViGoRL consists of a post-training pipeline designed to encourage the model to incorporate spatial coordinates grounding (x,y) within its language thought. Spatial-Ladder designs a progressive RL training framework using a specialized dataset. Similarly, we compare Qwen2.5-VL-7B to Image-Jigsaw-7B (Wu et al., 2025) and Thinklite-VL (Wang et al., 2025). Image-Jigsaw consists of an RFT phase leveraging self-supervised jigsaw puzzle tasks, and Thinklite-VL focuses on high quality data selection based on the reasoning iterations to do RL finetuning. Notably, despite our significantly smaller fine-tuning budget (5k samples vs. larger specialized corpora) and no human annotations, we observe that our PGT finetuning can be as effective or even more effective than a specialized training dataset on relational reasoning, and vision-centric and 3D tasks.

# 5. Ablations

In this section, we aim to systematically characterize which components of the proposed PGT training lead to the observed improvements. To do so, we run the ablations on a single backbone, namely Llama-3-8B (Grattafiori et al., 2024), and follow the experimental setup of subsection 4.1 leveraging the LLava-Instruct-1.5 dataset (Liu et al., 2023b). To rigorously assess how procedurally generated tasks transfer to specific vision-centric domains, we group our benchmarks into four primary axes of evaluation:

• Relational Reasoning: Evaluates spatial binding and relative positioning. We report average results across What’s Up-COCO (Kamath et al., 2023), VSR (Liu et al., 2023a), the relational subset of CV-Bench-2D (Tong et al., 2024), and the spatial splits of SEED (Li et al., 2023a) and BLINK (Fu et al., 2024).   
• Counting: Focuses on precise object enumeration. We average results across TallyQA (Acharya et al., 2018) and the counting splits of MMSTAR (Chen et al., 2024), SEED, CV-Bench, and BLINK.   
• 3D/Depth Understanding: Probes implicit depth perception and 3D geometry using CV-Bench-3D and BLINK (depth). We report average results across these benchmarks.   
• General Perception: Ensures no degradation in standard VQA or hallucination metrics. We average results from GQA (Hudson & Manning, 2019), AI2D (Hiippala et al., 2020), POPE (Li et al., 2023b), RealWorldQA (xAI, 2024), MMVP, and general aggregated results of SEED and MMSTAR.

We perform a set of ablations that isolate critical design choices that we made for our PGT training: the impact of each task in PGT (subsection 5.1), the impact of PGT on increasing real data sizes(subsection 5.2), and finally the impact of the proportion of PGT-augmented data w.r.t. to the base instruction tuning real training data(subsection 5.3). Exact hyperparameters for this section are detailed in Appendix B.2.

# 5.1. The impact of each task in PGT

To evaluate the specific contribution of each task in PGT, we conduct a leave-one-out ablation study across tasks, and compare results with the baseline leveraging regular finetuning (reg-FT) and with full PGT instruction tuning. We compute results by averaging performance across the benchmarks targeting different skills and report them in Table 3. We note that the removal of targeted tasks significantly degrades the corresponding skill axis. For example, removing the relational understanding PGT results in a 8.3% drop on average in relational understanding, while removing the counting PGT results in a 1.2% performance decrease in counting tasks. Importantly, removing any PGT does not result in notable variations in general understanding tasks. We also observe that excluding the relative distance estimation task causes the 3D/Depth understanding score to substantially reduce from 70.9% to 61.7%, nearly matching the 60.8% baseline performance. This empirical result demonstrates that the model’s ability to compare relative 2D distances between abstract primitives serves as the foundational primitive for emergent relative object depth comparison and 3D understanding. We also observe that the exclusion of the analogy task does not consistently degrade performance across different axes, suggesting that direct spatial and quantitative grounding provide the most robust transfer signals for the evaluated benchmarks.

Table 3. Leave-one-out task ablation: performance (%) when removing specific procedurally generated tasks from the full PGT suite. All MLLM are based on Llama-3-8B LLM backbone. 

<table><tr><td>Configuration</td><td>Relational</td><td>Counting</td><td>3D/Depth</td><td>General</td></tr><tr><td>Baseline (reg-FT)</td><td>65.2</td><td>52.5</td><td>60.8</td><td>61.9</td></tr><tr><td>Full PGT</td><td>75.7</td><td>55.2</td><td>72.9</td><td>62.2</td></tr><tr><td>w/o Relational</td><td>66.0</td><td>54.0</td><td>68.4</td><td>61.7</td></tr><tr><td>w/o Counting</td><td>73.9</td><td>51.5</td><td>69.3</td><td>61.9</td></tr><tr><td>w/o Distance</td><td>72.8</td><td>53.8</td><td>61.7</td><td>61.9</td></tr><tr><td>w/o Analogy</td><td>76.0</td><td>52.0</td><td>69.2</td><td>61.5</td></tr></table>

# 5.2. The benefits of PGT when scaling real data sizes

We validate whether the benefits of PGT persist as the size of the underlying real dataset increases or whether PGT is mostly beneficial in the low instruction tuning data regime. To do so, we propose to compare a MLLM model based on Llama-3-8b trained at the size of the Llava-Instruct-1.5 dataset (with ∼0.6M samples) to the same model trained on real data from Cambrian-7M (Tong et al., 2024) training set (with ∼7M). Both models are compared by considering instruction tuning with and without PGT data. As shown in Table 4, scaling the real dataset from 0.6M samples to 7M samples results in performance improvements across different tasks. Leveraging PGT provides further boosts, notably increasing the performance of the MLMM trained on Cambrian-7M by +6.1% to +7.8% on relational and depth reasoning tasks, respectively. This finding confirms that the current bottleneck for fine-grained reasoning in MLLMs is not the sheer volume of semantic instruction tuning data, but rather the density of high-quality spatial supervision. Our ablation shows that PGT effectively addresses this bottleneck regardless of the real (semantic) data training scale.

Table 4. Impact of PGT across different real, semantic training dataset scales (0.6M vs. 7M dataset sizes). MLLM is based on Llama-3-8b LLM backbone. 

<table><tr><td>Dataset Scale</td><td>Relational</td><td>Counting</td><td>3D/Depth</td><td>General</td></tr><tr><td>0.6M (Baseline, reg-FT)</td><td>65.2</td><td>52.5</td><td>60.8</td><td>61.9</td></tr><tr><td>0.6M + PGT</td><td>75.7</td><td>55.2</td><td>72.9</td><td>62.2</td></tr><tr><td>7M (Baseline, reg-FT)</td><td>69.7</td><td>57.5</td><td>62.5</td><td>64.9</td></tr><tr><td>7M + PGT</td><td>75.8</td><td>59.4</td><td>70.3</td><td>65.8</td></tr></table>

Table 5. Impact of the proportion of PGT-augmented samples on MLLM’s performance. MLLM is based on Llama-3-8b LLM backbone. Real data from lava-Instruct-1.5 used for captioning alignment and instruction tuning. When highlighted, PGT-augmented data is used for instruction tuning. 

<table><tr><td>Training</td><td>Relational</td><td>Counting</td><td>3D/Depth</td><td>General</td></tr><tr><td>Baseline (reg-FT)</td><td>65.2</td><td>52.5</td><td>60.8</td><td>61.9</td></tr><tr><td>PGT 100%</td><td>75.7</td><td>55.2</td><td>72.9</td><td>62.2</td></tr><tr><td>PGT 50%</td><td>74.3</td><td>57.1</td><td>71.2</td><td>62.1</td></tr><tr><td>PGT 10%</td><td>74.2</td><td>56.4</td><td>70.3</td><td>61.7</td></tr><tr><td>PGT 5%</td><td>73.8</td><td>53.0</td><td>69.6</td><td>61.9</td></tr></table>

# 5.3. The impact of the proportion of PGT-augmented data w.r.t. real data

We ablate the impact of the overall proportion of PGTaugmented data on the MLLM’s performance. To do so, we consider applying PGT on 100%, 50%, 10% and only 5% of the real, semantic training data from Llava-Instruct-1.5, and train MLLMs based on Llama-3-8b LLM backbone. Results are presensed in Table 5. We observe that applying PGT, even to small amounts of data, results in substantial performance improvements across different skills, with boosts being more pronounced in relational and 3D/Depth understanding. Although the benefits are stronger when applying PGT to at least 50% of the data, most of the gains are already realized when leveraging PGT on the smallest fraction of data considered (5%) – with improvements of over 8% in relational and 3D/Depth understandings. Counting is the only skill that appears to notably benefit from additional PGT augmentations during training. Importantly, the performance on general benchmarks remains roughly constant across different proportions of PGT-augmented data used for instruction tuning, and is comparable to the performance of the baseline trained on non-augmented data.

# 6. Conclusion and Future Work

Conclusion. In this work, we introduced Procedurally Generated Tasks (PGT) designed to tackle finegrained compositional understanding of MLLMs. We leverage PGT to train or finetune MLLMs, pushing the model to not rely on linguistic priors and semantic shortcuts to answer visual question. By forcing models to reason over unambiguous geometric primitives, we encourage the visual attention mechanism to ground itself in the actual image content rather than falling back on statistical correlations in the text. Through extensive experimentation, including both instruction tuning of MLLMs and finetuning of state-of-the-art MLLMs, we showed that leveraging PGT-augmented data not only transfers to real-world images but consistently boosts relational, quantitative, and 3D/Depth understanding, while maintaining general perception capabilities of the models. Moreover, we observed that 3D capabilities may emerge from unambiguous geometric primitives overlaid on real 2D data, highlighting the effectiveness of procedurally generated data and suggesting that this data holds the potential to address the existing bottleneck of fine-grained perception without requiring architectural modification or massive data scaling.

Beyond immediate performance gains, PGT serve a critical role as a low-cost diagnostic instrument. The design space for MLLMs is combinatorially complex—spanning architecture choices, training stages, and data curation—, making it difficult to pinpoint whether a failure stems from a lack of visual capacity or a lack of alignment. Echoing the findings of Fu et al. (2025), who showed that visual representations often contain details that the language model simply ignores, our work confirms that the bottleneck for fine-grained reasoning is often not the visual encoder’s resolution or the LLM’s size, but the training signal itself. PGT provides a low-cost and effective method to validate the influence of this bottleneck: if a model’s performance spikes with PGT, the capability was likely present but dormant, waiting for the right grounding signal.

Limitations and Future Work. We position PGT not as a replacement for data scaling or architectural enhancements, but as an orthogonal and highly efficient complementary signal. While model-centric approaches focus on extending a model’s theoretical capacity (e.g., via higher resolution or larger backbones), PGT addresses the alignment bottleneck that prevents models from accessing that capacity. Given the substantial performance gains observed across diverse benchmarks with negligible computational overhead, we argue that PGT is justified as a standard integration in modern training recipes, regardless of the underlying architecture.

Consequently, PGT should also serve as a necessary diagnostic baseline. Before attributing spatial failures to ”insufficient resolution” or ”weak vision backbones,” we encourage researchers to first verify if the model can solve PGT. Success on PGT indicates that the necessary visual features are present but dormant, waiting for a clearer training signal to override linguistic priors. We acknowledge, however, that the specific suite of tasks introduced in this paper is a proof-of-concept rather than an exhaustive catalog. While our current suite is not meant to be a comprehensive list of all possible tasks, we provide a detailed discussion of promising extension avenues for the PGT framework—including 3D primitives, temporal understanding, and multi-step reasoning—in Appendix F.

# Impact Statement

This paper presents work whose goal is to advance the field of machine learning. In particular, this paper focuses on improving fine-grained spatial understanding. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Acharya, M., Kafle, K., and Kanan, C. Tallyqa: Answering complex counting questions, 2018. URL https:// arxiv.org/abs/1810.12440.   
Assouel, R., Astolfi, P., Bordes, F., Drozdzal, M., and Romero-Soriano, A. Object-centric binding in contrastive language-image pretraining, 2025. URL https: //arxiv.org/abs/2502.14113.   
Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., and Lin, J. Qwen2.5-vl technical report, 2025. URL https://arxiv.org/abs/2502.13923.   
Chen, L., Li, J., Dong, X., Zhang, P., Zang, Y., Chen, Z., Duan, H., Wang, J., Qiao, Y., Lin, D., and Zhao, F. Are we on the right way for evaluating large vision-language models?, 2024. URL https://arxiv.org/abs/ 2403.20330.   
Chen, R., Guo, X., Liu, K., Liang, S., Liu, S., Zhang, Q., Wang, L., Zhang, H., and Cao, X. Where mllms attend and what they rely on: Explaining autoregressive token generation, 2026. URL https://arxiv.org/abs/ 2509.22496.   
Chen, Y., Yan, Z., Zhou, C., Dai, B., and Luo, A. F. Vision transformers with self-distilled registers, 2025. URL https://arxiv.org/abs/2505.21501.   
Covert, I., Sun, T., Zou, J., and Hashimoto, T. Locality alignment improves vision-language models, 2025. URL https://arxiv.org/abs/2410.11087.   
Dorkenwald, M., Barazani, N., Snoek, C. G. M., and Asano, Y. M. Pin: Positional insert unlocks object localisation abilities in vlms, 2024. URL https://arxiv.org/ abs/2402.08657.

Feng, S., Wang, S., Ouyang, S., Kong, L., Song, Z., Zhu, J., Wang, H., and Wang, X. Reasonmap: Towards finegrained visual reasoning from transit maps, 2026. URL https://arxiv.org/abs/2505.18675.

Fu, S., Bonnen, T., Guillory, D., and Darrell, T. Hidden in plain sight: Vlms overlook their visual representations, 2025. URL https://arxiv.org/abs/ 2506.08008.

Fu, X., Hu, Y., Li, B., Feng, Y., Wang, H., Lin, X., Roth, D., Smith, N. A., Ma, W.-C., and Krishna, R. Blink: Multimodal large language models can see but not perceive, 2024. URL https://arxiv.org/abs/ 2404.12390.

Grattafiori, A., Dubey, A., Jauhri, A., Pandey, A., Kadian, A., Al-Dahle, A., Letman, A., Mathur, A., Schelten, A., Vaughan, A., Yang, A., Fan, A., Goyal, A., Hartshorn, A., Yang, A., Mitra, A., Sravankumar, A., Korenev, A., Hinsvark, A., Rao, A., Zhang, A., Rodriguez, A., Gregerson, A., Spataru, A., Roziere, B., Biron, B., Tang, B., Chern, B., Caucheteux, C., Nayak, C., Bi, C., Marra, C., McConnell, C., Keller, C., Touret, C., Wu, C., Wong, C., Ferrer, C. C., Nikolaidis, C., Allonsius, D., Song, D., Pintz, D., Livshits, D., Wyatt, D., Esiobu, D., Choudhary, D., Mahajan, D., Garcia-Olano, D., Perino, D., Hupkes, D., Lakomkin, E., AlBadawy, E., Lobanova, E., Dinan, E., Smith, E. M., Radenovic, F., Guzman, F., Zhang, F., ´ Synnaeve, G., Lee, G., Anderson, G. L., Thattai, G., Nail, G., Mialon, G., Pang, G., Cucurell, G., Nguyen, H., Korevaar, H., Xu, H., Touvron, H., Zarov, I., Ibarra, I. A., Kloumann, I., Misra, I., Evtimov, I., Zhang, J., Copet, J., Lee, J., Geffert, J., Vranes, J., Park, J., Mahadeokar, J., Shah, J., van der Linde, J., Billock, J., Hong, J., Lee, J., Fu, J., Chi, J., Huang, J., Liu, J., Wang, J., Yu, J., Bitton, J., Spisak, J., Park, J., Rocca, J., Johnstun, J., Saxe, J., Jia, J., Alwala, K. V., Prasad, K., Upasani, K., Plawiak, K., Li, K., Heafield, K., Stone, K., El-Arini, K., Iyer, K., Malik, K., Chiu, K., Bhalla, K., Lakhotia, K., Rantala-Yeary, L., van der Maaten, L., Chen, L., Tan, L., Jenkins, L., Martin, L., Madaan, L., Malo, L., Blecher, L., Landzaat, L., de Oliveira, L., Muzzi, M., Pasupuleti, M., Singh, M., Paluri, M., Kardas, M., Tsimpoukelli, M., Oldham, M., Rita, M., Pavlova, M., Kambadur, M., Lewis, M., Si, M., Singh, M. K., Hassan, M., Goyal, N., Torabi, N., Bashlykov, N., Bogoychev, N., Chatterji, N., Zhang, N., Duchenne, O., C¸ elebi, O., Alrassy, P., Zhang, P., Li, P., Vasic, P., Weng, P., Bhargava, P., Dubal, P., Krishnan, P., Koura, P. S., Xu, P., He, Q., Dong, Q., Srinivasan, R., Ganapathy, R., Calderer, R., Cabral, R. S., Stojnic, R., Raileanu, R., Maheswari, R., Girdhar, R., Patel, R., Sauvestre, R., Polidoro, R., Sumbaly, R., Taylor, R., Silva, R., Hou, R., Wang, R., Hosseini, S., Chennabasappa, S., Singh, S., Bell, S., Kim, S. S., Edunov, S., Nie, S., Narang,

S., Raparthy, S., Shen, S., Wan, S., Bhosale, S., Zhang, S., Vandenhende, S., Batra, S., Whitman, S., Sootla, S., Collot, S., Gururangan, S., Borodinsky, S., Herman, T., Fowler, T., Sheasha, T., Georgiou, T., Scialom, T., Speckbacher, T., Mihaylov, T., Xiao, T., Karn, U., Goswami, V., Gupta, V., Ramanathan, V., Kerkez, V., Gonguet, V., Do, V., Vogeti, V., Albiero, V., Petrovic, V., Chu, W., Xiong, W., Fu, W., Meers, W., Martinet, X., Wang, X., Wang, X., Tan, X. E., Xia, X., Xie, X., Jia, X., Wang, X., Goldschlag, Y., Gaur, Y., Babaei, Y., Wen, Y., Song, Y., Zhang, Y., Li, Y., Mao, Y., Coudert, Z. D., Yan, Z., Chen, Z., Papakipos, Z., Singh, A., Srivastava, A., Jain, A., Kelsey, A., Shajnfeld, A., Gangidi, A., Victoria, A., Goldstand, A., Menon, A., Sharma, A., Boesenberg, A., Baevski, A., Feinstein, A., Kallet, A., Sangani, A., Teo, A., Yunus, A., Lupu, A., Alvarado, A., Caples, A., Gu, A., Ho, A., Poulton, A., Ryan, A., Ramchandani, A., Dong, A., Franco, A., Goyal, A., Saraf, A., Chowdhury, A., Gabriel, A., Bharambe, A., Eisenman, A., Yazdan, A., James, B., Maurer, B., Leonhardi, B., Huang, B., Loyd, B., Paola, B. D., Paranjape, B., Liu, B., Wu, B., Ni, B., Hancock, B., Wasti, B., Spence, B., Stojkovic, B., Gamido, B., Montalvo, B., Parker, C., Burton, C., Mejia, C., Liu, C., Wang, C., Kim, C., Zhou, C., Hu, C., Chu, C.-H., Cai, C., Tindal, C., Feichtenhofer, C., Gao, C., Civin, D., Beaty, D., Kreymer, D., Li, D., Adkins, D., Xu, D., Testuggine, D., David, D., Parikh, D., Liskovich, D., Foss, D., Wang, D., Le, D., Holland, D., Dowling, E., Jamil, E., Montgomery, E., Presani, E., Hahn, E., Wood, E., Le, E.-T., Brinkman, E., Arcaute, E., Dunbar, E., Smothers, E., Sun, F., Kreuk, F., Tian, F., Kokkinos, F., Ozgenel, F., Caggioni, F., Kanayet, F., Seide, F., Florez, G. M., Schwarz, G., Badeer, G., Swee, G., Halpern, G., Herman, G., Sizov, G., Guangyi, Zhang, Lakshminarayanan, G., Inan, H., Shojanazeri, H., Zou, H., Wang, H., Zha, H., Habeeb, H., Rudolph, H., Suk, H., Aspegren, H., Goldman, H., Zhan, H., Damlaj, I., Molybog, I., Tufanov, I., Leontiadis, I., Veliche, I.-E., Gat, I., Weissman, J., Geboski, J., Kohli, J., Lam, J., Asher, J., Gaya, J.-B., Marcus, J., Tang, J., Chan, J., Zhen, J., Reizenstein, J., Teboul, J., Zhong, J., Jin, J., Yang, J., Cummings, J., Carvill, J., Shepard, J., McPhie, J., Torres, J., Ginsburg, J., Wang, J., Wu, K., U, K. H., Saxena, K., Khandelwal, K., Zand, K., Matosich, K., Veeraraghavan, K., Michelena, K., Li, K., Jagadeesh, K., Huang, K., Chawla, K., Huang, K., Chen, L., Garg, L., A, L., Silva, L., Bell, L., Zhang, L., Guo, L., Yu, L., Moshkovich, L., Wehrstedt, L., Khabsa, M., Avalani, M., Bhatt, M., Mankus, M., Hasson, M., Lennie, M., Reso, M., Groshev, M., Naumov, M., Lathi, M., Keneally, M., Liu, M., Seltzer, M. L., Valko, M., Restrepo, M., Patel, M., Vyatskov, M., Samvelyan, M., Clark, M., Macey, M., Wang, M., Hermoso, M. J., Metanat, M., Rastegari, M., Bansal, M., Santhanam, N., Parks, N., White, N., Bawa, N., Singhal, N., Egebo, N., Usunier, N., Mehta,

N., Laptev, N. P., Dong, N., Cheng, N., Chernoguz, O., Hart, O., Salpekar, O., Kalinli, O., Kent, P., Parekh, P., Saab, P., Balaji, P., Rittner, P., Bontrager, P., Roux, P., Dollar, P., Zvyagina, P., Ratanchandani, P., Yuvraj, P., Liang, Q., Alao, R., Rodriguez, R., Ayub, R., Murthy, R., Nayani, R., Mitra, R., Parthasarathy, R., Li, R., Hogan, R., Battey, R., Wang, R., Howes, R., Rinott, R., Mehta, S., Siby, S., Bondu, S. J., Datta, S., Chugh, S., Hunt, S., Dhillon, S., Sidorov, S., Pan, S., Mahajan, S., Verma, S., Yamamoto, S., Ramaswamy, S., Lindsay, S., Lindsay, S., Feng, S., Lin, S., Zha, S. C., Patil, S., Shankar, S., Zhang, S., Zhang, S., Wang, S., Agarwal, S., Sajuyigbe, S., Chintala, S., Max, S., Chen, S., Kehoe, S., Satterfield, S., Govindaprasad, S., Gupta, S., Deng, S., Cho, S., Virk, S., Subramanian, S., Choudhury, S., Goldman, S., Remez, T., Glaser, T., Best, T., Koehler, T., Robinson, T., Li, T., Zhang, T., Matthews, T., Chou, T., Shaked, T., Vontimitta, V., Ajayi, V., Montanez, V., Mohan, V., Kumar, V. S., Mangla, V., Ionescu, V., Poenaru, V., Mihailescu, V. T., Ivanov, V., Li, W., Wang, W., Jiang, W., Bouaziz, W., Constable, W., Tang, X., Wu, X., Wang, X., Wu, X., Gao, X., Kleinman, Y., Chen, Y., Hu, Y., Jia, Y., Qi, Y., Li, Y., Zhang, Y., Zhang, Y., Adi, Y., Nam, Y., Yu, Wang, Zhao, Y., Hao, Y., Qian, Y., Li, Y., He, Y., Rait, Z., DeVito, Z., Rosnbrick, Z., Wen, Z., Yang, Z., Zhao, Z., and Ma, Z. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.21783.

Hiippala, T., Alikhani, M., Haverinen, J., Kalliokoski, T., Logacheva, E., Orekhova, S., Tuomainen, A., Stone, M., and Bateman, J. A. Ai2d-rst: a multimodal corpus of 1000 primary school science diagrams. Language Resources and Evaluation, 55(3):661–688, December 2020. ISSN 1574-0218. doi: 10.1007/ s10579-020-09517-1. URL http://dx.doi.org/ 10.1007/s10579-020-09517-1.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of large language models, 2021. URL https://arxiv. org/abs/2106.09685.

Hu, Y., Shi, W., Fu, X., Roth, D., Ostendorf, M., Zettlemoyer, L., Smith, N. A., and Krishna, R. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models, 2024. URL https://arxiv. org/abs/2406.09403.

Huang, W., Jia, B., Zhai, Z., Cao, S., Ye, Z., Zhao, F., Xu, Z., Hu, Y., and Lin, S. Vision-r1: Incentivizing reasoning capability in multimodal large language models, 2025. URL https://arxiv.org/abs/2503.06749.

Hudson, D. A. and Manning, C. D. GQA: a new dataset for compositional question answering over real-world

images. CoRR, abs/1902.09506, 2019. URL http: //arxiv.org/abs/1902.09506.   
Izadi, A., Banayeeanzade, M. A., Askari, F., Rahimiakbar, A., Vahedi, M. M., Hasani, H., and Baghshah, M. S. Visual structures helps visual reasoning: Addressing the binding problem in vlms, 2025. URL https://arxiv.org/abs/2506.22146.   
Jian, P., Yu, D., Yang, W., Ren, S., and Zhang, J. Teaching vision-language models to ask: Resolving ambiguity in visual questions. In Che, W., Nabende, J., Shutova, E., and Pilehvar, M. T. (eds.), Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 3619–3638, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-251- 0. doi: 10.18653/v1/2025.acl-long.182. URL https: //aclanthology.org/2025.acl-long.182/.   
Kamath, A., Hessel, J., and Chang, K.-W. What’s ”up” with vision-language models? investigating their struggle with spatial reasoning, 2023. URL https://arxiv.org/ abs/2310.19785.   
Karamcheti, S., Nair, S., Balakrishna, A., Liang, P., Kollar, T., and Sadigh, D. Prismatic vlms: Investigating the design space of visually-conditioned language models, 2024. URL https://arxiv.org/abs/2402.07865.   
Langley, P. Crafting papers on machine learning. In Langley, P. (ed.), Proceedings of the 17th International Conference on Machine Learning (ICML 2000), pp. 1207–1216, Stanford, CA, 2000. Morgan Kaufmann.   
Lei, X., Yang, Z., Chen, X., Li, P., and Liu, Y. Scaffolding coordinates to promote vision-language coordination in large multi-modal models, 2024. URL https: //arxiv.org/abs/2402.12058.   
Li, B., Wang, R., Wang, G., Ge, Y., Ge, Y., and Shan, Y. Seed-bench: Benchmarking multimodal llms with generative comprehension, 2023a. URL https://arxiv. org/abs/2307.16125.   
Li, F., Zhang, R., Zhang, H., Zhang, Y., Li, B., Li, W., Ma, Z., and Li, C. Llava-next-interleave: Tackling multiimage, video, and 3d in large multimodal models, 2024. URL https://arxiv.org/abs/2407.07895.   
Li, H., Li, D., Wang, Z., Yan, Y., Wu, H., Zhang, W., Shen, Y., Lu, W., Xiao, J., and Zhuang, Y. Spatialladder: Progressive training for spatial reasoning in vision-language models, 2025a. URL https://arxiv.org/abs/ 2510.08531.

Li, R., Li, S., Kong, L., Yang, X., and Liang, J. Seeground: See and ground for zero-shot open-vocabulary 3d visual grounding, 2025b. URL https://arxiv. org/abs/2412.04383.   
Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large visionlanguage models, 2023b. URL https://arxiv. org/abs/2305.10355.   
Liu, D., Liang, T., Hu, Z., Peng, J., Lu, Y., Xu, Y., Fu, Y., and Yin, Y. Spatial intelligence in vision-language models: A comprehensive survey, November 2025. URL https://www.techrxiv.org/users/ 992599/articles/1354538/master/file/ data/Spatial\_VLM\_v1/Spatial\_VLM\_v1. pdf?inline=true. TechRxiv. Preprint.   
Liu, F., Emerson, G., and Collier, N. Visual spatial reasoning, 2023a. URL https://arxiv.org/abs/ 2205.00363.   
Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning, 2023b. URL https://arxiv.org/abs/ 2304.08485.   
Meng, F., Du, L., Liu, Z., Zhou, Z., Lu, Q., Fu, D., Han, T., Shi, B., Wang, W., He, J., Zhang, K., Luo, P., Qiao, Y., Zhang, Q., and Shao, W. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning, 2025. URL https://arxiv. org/abs/2503.07365.   
Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., and Qiu, Z. Qwen2.5 technical report, 2025. URL https: //arxiv.org/abs/2412.15115.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020.   
Ranzinger, M., Heinrich, G., Molchanov, P., Kautz, J., Catanzaro, B., and Tao, A. Featsharp: Your vision model features, sharper, 2025. URL https://arxiv.org/ abs/2502.16025.   
Sarch, G., Saha, S., Khandelwal, N., Jain, A., Tarr, M. J., Kumar, A., and Fragkiadaki, K. Grounded reinforcement learning for visual reasoning, 2025. URL https:// arxiv.org/abs/2505.23678.

Sur´ıs, D., Menon, S., and Vondrick, C. Vipergpt: Visual inference via python execution for reasoning, 2023. URL https://arxiv.org/abs/2303.08128.   
Tong, S., Brown, E., Wu, P., Woo, S., Middepogu, M., Akula, S. C., Yang, J., Yang, S., Iyer, A., Pan, X., Wang, Z., Fergus, R., LeCun, Y., and Xie, S. Cambrian-1: A fully open, vision-centric exploration of multimodal llms, 2024. URL https://arxiv.org/abs/ 2406.16860.   
Tschannen, M., Gritsenko, A., Wang, X., Naeem, M. F., Alabdulmohsin, I., Parthasarathy, N., Evans, T., Beyer, L., Xia, Y., Mustafa, B., Henaff, O., Harmsen, J., Steiner, ´ A., and Zhai, X. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features, 2025. URL https://arxiv. org/abs/2502.14786.   
Wang, X., Yang, Z., Feng, C., Lu, H., Li, L., Lin, C.-C., Lin, K., Huang, F., and Wang, L. Sota with less: Mctsguided sample selection for data-efficient visual reasoning self-improvement, 2025. URL https://arxiv. org/abs/2504.07934.   
Wu, P. and Xie, S. V\*: Guided visual search as a core mechanism in multimodal llms, 2023. URL https: //arxiv.org/abs/2312.14135.   
Wu, P., Zhang, Y., Diao, H., Li, B., Lu, L., and Liu, Z. Visual jigsaw post-training improves mllms, 2025. URL https://arxiv.org/abs/2509.25190.   
Wu, W., Mao, S., Zhang, Y., Xia, Y., Dong, L., Cui, L., and Wei, F. Mind’s eye of llms: Visualization-of-thought elicits spatial reasoning in large language models, 2024. URL https://arxiv.org/abs/2404.03622.   
xAI. Realworldqa: A benchmark for real-world spatial understanding, 2024. URL https://x.ai/news/ grok-1.5v. Dataset released with Grok-1.5 Vision Preview.   
Xing, S., Sun, Z., Xie, S., Chen, K., Huang, Y., Wang, Y., Li, J., Song, D., and Tu, Z. Can large vision language models read maps like a human?, 2025. URL https: //arxiv.org/abs/2503.14607.   
Yamada, Y., Tang, Y., Zhang, Y., and Yildirim, I. When are lemons purple? the concept association bias of visionlanguage models, 2024. URL https://arxiv.org/ abs/2212.12043.   
Yang, J., Yang, S., Gupta, A. W., Han, R., Fei-Fei, L., and Xie, S. Thinking in space: How multimodal large language models see, remember, and recall spaces, 2025. URL https://arxiv.org/abs/2412.14171.

Yoon, H., Jung, J., Kim, J., Choi, H., Shin, H., Lim, S., An, H., Kim, C., Han, J., Kim, D., Eom, C., Hong, S., and Kim, S. Visual representation alignment for multimodal large language models, 2025. URL https://arxiv. org/abs/2509.07979.   
Yu, E., Lin, K., Zhao, L., Yin, J., Wei, Y., Peng, Y., Wei, H., Sun, J., Han, C., Ge, Z., Zhang, X., Jiang, D., Wang, J., and Tao, W. Perception-r1: Pioneering perception policy with reinforcement learning, 2025. URL https: //arxiv.org/abs/2504.07954.   
Zhou, Q., Zhou, R., Hu, Z., Lu, P., Gao, S., and Zhang, Y. Image-of-thought prompting for visual reasoning refinement in multimodal large language models, 2024. URL https://arxiv.org/abs/2405.13872.   
Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., Gao, Z., Cui, E., Wang, X., Cao, Y., Liu, Y., Wei, X., Zhang, H., Wang, H., Xu, W., Li, H., Wang, J., Deng, N., Li, S., He, Y., Jiang, T., Luo, J., Wang, Y., He, C., Shi, B., Zhang, X., Shao, W., He, J., Xiong, Y., Qu, W., Sun, P., Jiao, P., Lv, H., Wu, L., Zhang, K., Deng, H., Ge, J., Chen, K., Wang, L., Dou, M., Lu, L., Zhu, X., Lu, T., Lin, D., Qiao, Y., Dai, J., and Wang, W. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models, 2025. URL https://arxiv.org/abs/2504.10479.

# A. Additional Dataset details

![](images/a1294eb3367b698e124d72d8aa1ee0ff05a0ef10eabc74cf14db07ad9ba84458.jpg)

<details>
<summary>natural_image</summary>

Street view of a dense urban street with tall buildings and visible road markings (no readable text or symbols)
</details>

# Spatial Relations

2D Relation : “Where is the green box relative to the red box? Answer with left or right.”

Answer: “Right”

Localization : “What are the normalized coordinates of the center of the red box?”

Answer: “[0.6, 0.2]”

![](images/37aa38feff6a235d9df99b2314d1e00a2a9d4110587c3481355150f8ba2d8a24.jpg)

<details>
<summary>natural_image</summary>

Modern living room interior with sofa, coffee table, and large windows (no visible text or symbols)
</details>

# Abstract Counting

“There are several colored overlaid circles in this image. How many orange circles are there?

A. 2   
B. 3   
C. 1 ?

Answer with the letter from the correct option directly.”

Answer: “B”

![](images/7b19fa54c11a7a7f43e81524b65ca7850af356a0a8898c90ee04fd261ceeb672.jpg)

<details>
<summary>text_image</summary>

A
C
D
B
</details>

# Distance & Analogy

Distance : “There are four colored circles labeled A, B, C, D in this image. Which one is closest to C? B, A or D ?”

Answer: “B”

Analogy : What is the letter of the circle that has the same color as the circle with a B in it ?

Answer : “A”

Figure 2. Our suite of PGT: (left) spatial relationship reasoning, (center) abstract counting, and (right) 2D relative distance estimation.

In this section, we provide the specific prompts and templates used for generating the Procedurally Generated Tasks (PGT) as well as the prompts used for the Specialized Mix (constructed from TallyQA, VSR, and Spatial Ladder).

# A.1. Handling Occlusion and Semantic Preservation

To mitigate the potential occlusion of critical semantic features in the original image, specific geometric primitives—such as the colored circles used in the counting tasks—are explicitly rendered as semi-transparent during our procedural generation pipeline. Our empirical results validate that this theoretical interference does not negatively manifest in practice. As demonstrated in Section 4.1, performance on general perception benchmarks (e.g., MMSTAR, GQA, and POPE) remains stable or fluctuates minimally across all model backbones when PGT is applied. This confirms that the semantic integrity of the original image is sufficiently preserved, allowing the model to effectively learn from the original instruction-tuning QA pairs alongside the PGT augmentations.

# A.2. Procedurally Generated Tasks (PGT) Prompts

For each task in our PGT suite, we utilize a standardized prompt template to ensure consistency. The geometric primitives (e.g., bounding boxes, colored circles) are overlaid on the image prior to feeding it into the model. We varied templates with and without formatting instructions. When several template are available one is sampled uniformly at random.

Spatial Relationship Understanding. This task involves two overlaid bounding boxes (typically green and red). The model is asked to identify the spatial relationship of the green box relative to the red box. The model is also tasked to predict the center coordinates of one the boxes as shown in Figure 2.

# Spatial Relationship Prompt

Template A: USER: Where is the [color A] box relative to the [rel B] box? Answer with [rel A] or [rel B]. GPT : [rel A]

Template B: USER: Based on the image, is this statement True or False? The [color A] box is [rel A] of [color B] box? Answer with True or False directly. GPT : True

Template C: USER :Considering the relative positions of the [color A] box and the [color B] box in the image provided, where is the [color A] box located with respect to the [color A] box? Select from the following choices. (A) [rel A] (B) [rel B] GPT : [rel A]

Counting. In this task, semi-transparent circles of various colors are projected onto the scene. The model must count only the circles of a specific target color.

# Counting Prompt

Template A : USER : How many [color] overlay circles are there? Answer with the number directly GPT : [num]

Template B : USER : How many [color] overlay circles are there? (A) [option A] (B) [option B] (C) [option C] Answer with the option’s letter from the given choices directly GPT : [letter]

Analogy and Relative Distance. This task utilizes multi-point overlays labeled with identifiers (e.g., A, B, C, D). The model is prompted to estimate distances (e.g., finding the closest point to a target) or solve a simple color-based analogy.

# Relative Distance Prompt

Template Distance: USER: There are four colored circles labeled A, B, C, D in this image. Which one is closes to [target letter] : [option A], [option B], or [option C] ? GPT : [letter]

Template Analogy: USER: What is the letter of the circle that has the same color as the circle with a [target letter] in it ? GPT: [letter]

# A.3. Specialized Training Mix Prompts

To compare PGT against real-world annotated data, we constructed a Specialized Mix using samples from TallyQA (Acharya et al., 2018), VSR (Liu et al., 2023a), and Spatial Ladder-26k (Li et al., 2025a). When feasible, we standardized the answer formatting for these datasets to align with our PGT format and diversity.

Counting (TallyQA). Counting taks are sourced from TallyQA (Acharya et al., 2018), like our PGT set we alternate between free form answers and multiple choice. We only vary the answering format based on the answer. The question is taken from the original dataset directly.

# TallyQA Prompt (Specialized Mix)

Template A : USER : How many [obj] are there? Answer with the number directly GPT : [num]

Template B : USER : How many [obj] are there? (A) [option A] (B) [option B] (C) [option C] Answer with the option’s letter from the given choices directly GPT : [letter]

Relational Reasoning (VSR). For relational tasks sourced from Visual Spatial Reasoning (VSR) (Liu et al., 2023a), we use true/false validation prompts regarding the spatial configuration of objects. Note that VSR employs more relationships than our PGT set which is restricted 2D to left/right, above/below spatial statements.

# VSR Prompt (Specialized Mix)

Template: USER: Based on the image, is this statement True or False? [statement] Answer with True or False directly. GPT : True

Distance Estimation (Spatial Ladder). For distance estimation, we utilize the distance-estimation subset from the Spatial-Ladder-26k dataset (Li et al., 2025a) without changing the original formatting of the answer.

# Spatial Ladder Prompt (Specialized Mix)

Example: user : Measuring from the closest point of each object, which of these two objects (tv, pillow) is closer to the table? A. tv B. pillow Answer with the option’s letter from the given choices directly. gpt : A.

# B. Additional Training Details

In this section, we provide the detailed hyperparameters and training configurations used for our experiments. We categorize our experiments into two main settings: (1) Instruction Tuning and Ablations (Sections 4.1 and 5), which utilize the Prismatic (Karamcheti et al., 2024) training framework, and (2) Finetuning of State-of-the-Art (SOTA) MLLMs (Section 4.2), which utilizes the LLaMA-Factory framework.

# B.1. Instruction Tuning and Ablations

For the multimodal training experiments (Section 4.1) and the ablation studies (Section 5), we adopted the Prismatic training framework (Karamcheti et al., 2024). We followed a two-stage training protocol: first, pre-training the adapter for captioning alignment, followed by instruction tuning of both the adapter and the LLM backbone while keeping the vision encoder frozen. We adopt a 2 layer MLP for the adapter and fix the vision backbone to the most commonly used CLIP-L-14-336px.

Following Karamcheti et al. (2024) the captioning alignment training is done with a global batch size of 256 and a learning rate of 1e − 3. The instruction tuning training is done with a global batch size of 128 and a learning rate of $2 \times 1 0 ^ { - 5 } ,$ , a cosine schedule and a 0.1 warmup . Tables 6 and 7 detail the specific optimization and system hyperparameters used in this setting.

The ablation experiment measuring the impact of the multimodal training data 5.3 using the Cambrian-7M dataset follows the hyperparameters given in Tong et al. (2024), we also detail them in Tables 8 and 9.

Table 6. Optimization Hyperparameters (LLaVA-Instruct-1.5). 

<table><tr><td>Hyperparameter</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>1e-3</td><td>2e-5</td></tr><tr><td>LR Schedule</td><td>Cosine</td><td>Cosine</td></tr><tr><td>Warmup Ratio</td><td>0.03</td><td>0.1</td></tr><tr><td>Global Batch Size</td><td>256</td><td>128</td></tr><tr><td>Weight Decay</td><td>0.0</td><td>0.1</td></tr><tr><td>Grad. Clipping</td><td>1</td><td>1</td></tr></table>

Table 8. Optimization Hyperparameters (Cambrian). 

<table><tr><td>Hyperparameter</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>1e-3</td><td>4e-5</td></tr><tr><td>LR Schedule</td><td>Cosine</td><td>Cosine</td></tr><tr><td>Warmup Ratio</td><td>0.03</td><td>0.1</td></tr><tr><td>Global Batch Size</td><td>512</td><td>512</td></tr><tr><td>Weight Decay</td><td>0</td><td>0.1</td></tr><tr><td>Grad. Clipping</td><td>1</td><td>1</td></tr></table>

Table 7. Training & System Config (LLaVA-Instruct-1.5). 

<table><tr><td>Config</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>Epochs</td><td>1</td><td>2</td></tr><tr><td>Precision</td><td>BF16</td><td>BF16</td></tr><tr><td>Max Seq. Len</td><td>2048</td><td>2048</td></tr><tr><td>Num GPUs</td><td>8</td><td>16</td></tr><tr><td>Sharding</td><td>hybrid_shard_zero2</td><td>hybrid_shard</td></tr></table>

Table 9. Training & System Config (Cambrian). 

<table><tr><td>Config</td><td>Stage 1</td><td>Stage 2</td></tr><tr><td>Epochs</td><td>1</td><td>1</td></tr><tr><td>Precision</td><td>BF16</td><td>BF16</td></tr><tr><td>Num GPUs</td><td>8</td><td>16</td></tr><tr><td>Max Seq. Len</td><td>2048</td><td>2048</td></tr><tr><td>Sharding</td><td>hybrid_shard_zero2</td><td>hybrid_shard</td></tr></table>

# B.2. Finetuning SOTA MLLMs

For the experiments involving the finetuning of state-of-the-art MLLMs (Section 4.2), such as Qwen-2.5-VL, LLava-NexT and InternVL3, we utilized the LLaMA-Factory framework. These models were finetuned using Low-Rank Adaptation (LoRA) on the PGT-augmented data (or the specialized mix). The exact instruction tuning prompt for this data are given in Section B.

We trained the models on 5000 samples, for 2 epochs with a learning rate of $1 \times 1 0 ^ { - 4 }$ . We used a LoRA rank of 8 and alpha of 32 targeting all the keys, queries, and values linear layer of the language model. The multimodal projector and vision backbone are kept frozen. Table 10 lists the full configuration and hyperparameters.

Table 10. Hyperparameters for SOTA MLLM Finetuning (LLaMA-Factory). 

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="2">Optimization</td></tr><tr><td>Learning Rate</td><td> $1 \times 10^{-4}$ </td></tr><tr><td>LR Schedule</td><td>Cosine</td></tr><tr><td>LR ratio</td><td>0.1</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Global Batch Size</td><td>64</td></tr><tr><td>Max Gradient Norm</td><td>2</td></tr><tr><td colspan="2">LoRA Configuration</td></tr><tr><td>LoRA Rank (r)</td><td>8</td></tr><tr><td>LoRA Alpha (α)</td><td>32</td></tr><tr><td colspan="2">Training Duration</td></tr><tr><td>Num Epochs</td><td>2</td></tr><tr><td>Data Sample Size</td><td>5k (PGT/Specialized Mix)</td></tr></table>

# C. Additional Qualitative Analyses

# C.1. Identified Spatial Shortcuts

In this section, we provide a qualitative analysis of the failure modes and reasoning shortcuts observed in current state-ofthe-art MLLMs. Echoing the findings of Fu et al. (2025), who showed that visual representations often contain details that the language model simply ignores or overlooks, our work confirms that the bottleneck for fine-grained reasoning is often not the visual encoder’s resolution or the LLM’s size, but the training signal itself. By examining the models’ responses and reasoning traces (for thinking models), we identify recurring non-robust heuristics that models employ to solve spatial tasks.

We specifically observed these common shortcut mechanisms:

• Scale-Depth Correlation: The model assumes larger objects are always closer, probably based on real world statistics.   
• 2D Verticality vs. Depth: The model conflates the 2D y-coordinate with depth, assuming objects higher in the image plane are further away, regardless of the actual scene geometry.   
• Detailedness vs. Depth : The model relies on the common fact that more detailed ( and less blurry objects) are usually located closer to the camera.

Below, we present examples illustrating these behaviors.

# Scale-Depth Correlation

# Model: VigoRL-3B

Question:Which object is closer to the camera taking this photo, the door (highlighted by a red box) or the books (highlighted by a blue box)?

(A) door

(B) books

Answer only with the single capital letter corresponding to the correct choice.

![](images/8579c093f3b03e7989b1768076df199bcdc546a5046e2b4c82d70a2c6caa3c6b.jpg)

<details>
<summary>natural_image</summary>

Interior view of a minimalist room with a doorway and bookshelves, no visible text or symbols
</details>

# Model’s Thinking Trace):

To determine which object is closer to the camera, let’s consider their relative positions. The door is located at (670, 150), and the books are inside the blue box at (350, 350). Since the books are inside the blue box and the blue box is closer to the camera than the door, the books are closer to the camera. Let’s verify this by comparing the distances. The door is at (670, 150), and the books are at (350, 350). The distance from the camera to the door is greater than the distance from the camera to the books. Now, let’s consider the perspective of the image. The door appears smaller than the books because it is farther away. This confirms that the books are indeed closer to the camera. Another factor to consider is the depth of the scene. The books are positioned inside the blue box, which is closer to the camera than the door. This further supports that the books are closer to the camera. To ensure accuracy, let’s also consider the lighting and shadows. The books cast a shadow on the floor, indicating they are closer to the light source, which is consistent with them being closer to the camera. Lastly, let’s examine the scale of objects. The books appear larger in the image compared to the door, suggesting they are closer to the camera. This is another indicator that the books are indeed closer.

# Analysis:

The model relies on a size heuristic, equating pixel area with proximity. It fails to account for the intrinsic size difference between the objects, demonstrating a lack of genuine depth perception.

# 2D Verticality/Details vs. 3D Depth

# Model:VigoRL-3B

# Question:

Which object is closer to the camera taking this photo, the door (highlighted by a red box) or the lamp (highlighted by a blue box)?

(A) door

(B) lamp

Answer only with the single capital letter corresponding to the correct choice.

![](images/654646810e21db3c2533a382ec0e17faa13d659bdca32fc7cd9744f7bfbbf59b.jpg)

<details>
<summary>natural_image</summary>

Modern kitchen interior with dark tiled walls, a modern kitchen counter, and hanging kitchenware (no visible text or symbols)
</details>

# Model Response (Thinking Trace):

To determine which object is closer to the camera, let’s consider their relative positions. The door (highlighted by a red box) is located near the center-left of the image at (400, 300). The lamp (highlighted by a blue box) is positioned higher up and further back in the room at (550, 50). Since the door is closer to the bottom edge of the image and the lamp is higher up, the door appears closer to the camera. Let’s verify this by comparing their distances from the camera’s perspective. The door is located at (400, 300), which is closer to the bottom of the image. The lamp is at (550, 50), which is higher up and further back. Considering the perspective, the door at (400, 300) seems closer to the camera than the lamp at (550, 50). To further confirm, let’s consider the depth of the scene. The door at (400, 300) is closer to the foreground compared to the lamp at (550, 50). This suggests that the door is indeed closer to the camera. Another factor to consider is the size of the objects. The door at (400, 300) appears larger and more detailed, indicating that it is closer to the camera than the smaller and farther lamp at (550, 50). To ensure accuracy, let’s also examine the lighting. The door at (400, 300) is well-lit, suggesting it is closer to the light source, whereas the lamp at (550, 50) is less illuminated, indicating it is farther away from the light source. This further supports the conclusion that the door is closer to the camera. Lastly, let’s consider the shadows. The shadow cast by the door at (400, 300) is more pronounced and closer to the camera, reinforcing that the door is indeed closer to the camera compared to the lamp at (550, 50).

# Analysis:

The model applies a projective geometry prior indiscriminately. While often true for objects on a ground plane, this heuristic fails in indoor scenes with various views , where the 2D y-coordinate does not map linearly to depth. It also relies on another correlation where more detailed objects are usually closer to the camera.

# Binding Erros

Model: Spatial-Ladder-3B

# Question:

Based on the image, is this statement ”True” or ”False”?The suitcase is under the cat. Respond with ”True” or ”False” directly.

![](images/eacde4f0bcd806c8048d22718844af2461c43eaf26b148634baa3c3ec13c86be.jpg)

<details>
<summary>natural_image</summary>

A tabby cat resting on a black suitcase, surrounded by folded clothes (no text or symbols visible)
</details>

# Model Response (Thinking Trace):

To determine if the statement ”The suitcase is under the cat” is true, let’s analyze the image. The cat is positioned on top of the suitcase, not underneath it. The cat’s body is aligned with the top edge of the suitcase, indicating that the cat is on top. (250, 150) To further confirm this observation, let’s consider the relative positions of the cat and the suitcase. The cat’s head is near the top of the suitcase, and its body extends down towards the bottom of the suitcase. This confirms that the cat is indeed on top of the suitcase, not underneath. (350, 200) Another way to verify this is by examining the shadows and lighting in the image. The shadow cast by the cat appears to fall directly beneath the cat, suggesting that the cat is on top of the suitcase rather than underneath it. (400, 250) To ensure accuracy, let’s also look at the texture and material of the objects. The suitcase appears to be made of a hard material, while the cat’s fur is soft and fluffy. This further supports the idea that the cat is on top of the suitcase, not underneath. (500, 300) Lastly, let’s consider the perspective and depth of the image. The cat’s position relative to the background elements suggests that it is elevated above the ground, which aligns with the cat being on top of the suitcase rather than underneath. (600, 350)

# Analysis:

As the reasoning progresses the shifted towards evaluating the statement ”The cat is under the suitcase” instead of the initial one ”The suitcase is under the cat”. It did not remain consistent with the initial relationship binding, switching the subject of the assessed relationship to the most probable statistically (animated subject) in real-world captioned images.

We also give few example of corrected answer in the CVBench-3D dataset. Answers are taken from LLava-Next-8B where we observed significant boost after finetuning the model on our PGT mix. Notably, in all those examples, the models systematically predicts that the bigger object is closer to the camera. While this observation is not causal of the observed effect, we want to draw’s the reader’s attention to the potential shortcuts the model might rely on and how that can inform dataset design.

# Examples of corrected answers (LLaVA-NeXt-8b)

![](images/e160a8bc9797c1af6b08da45b0cf111da268cedad90cd1e02dfc9862e00d29fc.jpg)

<details>
<summary>natural_image</summary>

Interior view of a cozy living room with wooden furniture, TV, and wall-mounted artwork (no visible text or symbols)
</details>

Question: Which object is closer to the camera taking this photo, the books (highlighted by a red box) or the chair (highlighted by a blue box)?

(A) books

(B) chair

Answer only with the single capital letter corresponding to the correct choice.

Original Prediction: B

Prediction after PGT ft: A

![](images/4dba3d766dfebb6399ae205273d113955ca4607972add023741f434864ca54d0.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern dining or hotel lobby with wooden furniture, decorative wall art, and a potted plant (no visible text or symbols)
</details>

Question: Which object is closer to the camera taking this photo, the lamp (highlighted by a red box) or the table (highlighted by a blue box)?

(A) lamp

(B) table

Answer only with the single capital letter corresponding to the correct choice.

Original Prediction: B

Prediction after PGT ft: A

![](images/18d10286228f3d0192663de988e12905a4ac36dd3160a3041d065005bce0443c.jpg)

<details>
<summary>natural_image</summary>

Interior view of a modern kitchen with dark tiled walls, a black counter, and hanging kitchen utensils (no visible text or symbols)
</details>

Question: Which object is closer to the camera taking this photo, the door (highlighted by a red box) or the lamp (highlighted by a blue box)?

(A) door

(B) lamp

Answer only with the single capital letter corresponding to the correct choice.

Original Prediction: A

Prediction after PGT ft: B

![](images/ce64399f33478686299fe21d0a4b7e58c01f7701750abaa11a39589f3847aa11.jpg)

<details>
<summary>natural_image</summary>

Street scene with trees and a road, featuring two marked boxes (blue and red) highlighting areas of interest near buildings and a fence.
</details>

Question: Which object is closer to the camera taking this photo, the bicycle (highlighted by a red box) or the bus (highlighted by a blue box)?

(A) bicycle

(B) bus

Answer only with the single capital letter corresponding to the correct choice.

Original Prediction: B

Prediction after PGT ft: A

# D. Additional Ablations

Table 11. Impact of the overlay strategy vs. separate synthetic data. All results use the Llama-3-8b backbone. 

<table><tr><td>Strategy</td><td>Relational</td><td>Counting</td><td>3D/Depth</td><td>General</td></tr><tr><td>Baseline (reg-FT)</td><td>65.2</td><td>52.5</td><td>60.8</td><td>61.9</td></tr><tr><td>PGT (Overlay)</td><td>75.7</td><td>55.2</td><td>72.9</td><td>62.2</td></tr><tr><td>Separate Data</td><td>75.8</td><td>55.3</td><td>75.4</td><td>62.8</td></tr></table>

In this section we analyze the efficiency of our overlay mechanism compared to a traditional strategy of appending synthetic data as a separate dataset (Separate Data) and thus augmenting the total number of training samples. While training on a separate synthetic mix yields a slight absolute performance advantage—reaching 75.8% on relational reasoning and 75.4% on 3D/Depth—it requires a substantial increase in training iterations and computational cost. Our overlay strategy captures the vast majority of these performance gains while maintaining the original training sample count. This demonstrates that overlaying geometric primitives directly onto semantic scenes is not only a computationally superior trade-off but also forces the model to disentangle abstract geometry from real-world features within a single visual context.

# E. Extended Limitations: Task Saturation and Clutter

While our main ablation study (Section 5.3) demonstrates that applying PGT to just 5% of the training data yields significant performance improvements, and scaling up to 100% maintains stable general perception performance, there is a theoretical upper limit to task density. Specifically, injecting multiple different PGT overlays per image (exceeding a 100% ratio) introduces the risk of severe image clutter.

To directly evaluate the impact of exceeding a 100% ratio, we conducted an additional experiment simulating task accumulation. We finetuned the LLaVA-NeXt-7B model with 1, 2, and 3 distinct PGT tasks overlaid per image sample. The results are summarized in Table 12.

Table 12. Impact of task accumulation (injecting multiple PGTs per sample) on LLaVA-NeXt-7B performance. 

<table><tr><td>Number of PGTs</td><td>Relational (%)</td><td>Count (%)</td><td>3D/Depth (%)</td><td>General (%)</td></tr><tr><td>1 task</td><td>69.8</td><td>53.7</td><td>61.4</td><td>63.1</td></tr><tr><td>2 tasks</td><td>71.1</td><td>54.8</td><td>60.4</td><td>62.0</td></tr><tr><td>3 tasks</td><td>70.8</td><td>52.5</td><td>54.8</td><td>61.1</td></tr></table>

These results show that while the model can accommodate a moderate increase in task density (peaking at 2 tasks for relational reasoning and counting), pushing the augmentation to 3 tasks causes a noticeable degradation across all metrics, with 3D/Depth understanding experiencing the most severe drop. We acknowledge that this specific setup isolates task accumulation and does not strictly evaluate the occlusion of real-world background information. However, it clearly indicates that injecting too many tasks per image is inherently detrimental to the model’s learning capacity, irrespective of background clutter.

# F. Extended Future Work

While the tasks presented in the main text serve as a foundational proof-of-concept, the design space for Procedurally Generated Tasks (PGT) is vast. We outline several highly promising avenues for extending the visual PGT framework to address more complex modalities and advanced spatial logic.

3D Geometric Primitives. As demonstrated in Section 4.1, our 2D relative distance tasks elicited emergent 3D depth perception, strongly supporting the hypothesis that these tasks reinforce a shared distance estimation circuit. A natural and highly relevant extension is to explicitly target 3D spatial understanding by procedurally generating 3D geometric primitives. Injecting rendered shapes—such as cubes or spheres with simulated lighting, shadows, and projective perspective—directly onto images holds the potential to push models’ 3D understanding significantly further than 2D abstractions alone.

Temporal Reasoning in Video. Extending the PGT framework beyond static images to the video modality represents a critical next step. Procedural geometric tasks focusing on temporal reasoning—such as tracking a procedurally generated marker across frames or predicting the trajectory of a moving overlay—could provide the dense, unambiguous supervision required to improve object permanence and temporal grounding in MLLMs.

Multi-Step Reasoning and Advanced Spatial Logic. Our current PGT suite focuses on fixing the foundational visual understanding layer (e.g., binary spatial predicates, basic counting, and depth estimation). However, recent evaluation benchmarks such as ReasonMap (Feng et al., 2026), MapBench (Xing et al., 2025), and V\* (Wu & Xie, 2023) demonstrate that MLLMs still heavily struggle with complex visual reasoning and planning, such as multi-hop route finding on topological maps or systematic visual search. We hypothesize that PGT can act as a bridge between basic visual grounding and these complex reasoning capabilities. Future task designs could introduce multi-step geometric pathfinding between labeled nodes or require intersection area estimation of overlapping semi-transparent shapes. Applying the PGT methodology to these complex reasoning tasks via step-by-step prompt expansion (akin to chain-of-thought, but explicitly grounded in procedurally generated visual anchors) is a natural and exciting evolution of this work.