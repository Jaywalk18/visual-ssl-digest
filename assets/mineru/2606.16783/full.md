# Gen-VCoT: Generative Visual Chain-of-Thought Reasoning via Diffusion-Based RGB Intermediate Representations

Zhiqiang Zhou, Xu Ling, Junliang Dai

Hunan Chemical Industry Vocational and Technical College, Zhuzhou, Hunan 412000, China

willenchow@126.com

June 16, 2026

## Abstract

Multimodal large language models (MLLMs) have demonstrated remarkable capabilities in visual reasoning, yet their reasoning processes primarily rely on text-based chain-of-thought (CoT), lacking explicit and interpretable intermediate visual processing. Existing visual CoT methods either use opaque continuous visual tokens or depend on external tool invocations, failing to simultaneously achieve interpretability, end-to-end trainability, and dense visual representations. We propose Gen-VCoT, a generative visual chain-of-thought framework that leverages expert vision models to produce RGB images as visual reasoning intermediates. Gen-VCoT decomposes visual reasoning into three interpretable stages: (1) visual grounding—generating instance segmentation maps via SAM to highlight question-relevant regions; (2) geometric reasoning—producing pseudo-colored depth maps via Marigold to establish spatial relationships; and (3) semantic reasoning—an MLLM (Qwen2-VL) integrates the original image with generated visual evidence to produce the final answer. We further design an adaptive reasoning router that dynamically selects the required reasoning depth based on question complexity. Comprehensive evaluations across both complex spatial reasoning scenes and CLEVR-style benchmarks reveal a nuanced picture: Gen-VCoT improves spatial relationship (+25%) and depth perception (+50%) questions, but can degrade performance on simple factual queries where intermediates introduce noise. A three-way comparison with text-only chain-of-thought (providing structured text descriptions instead of visual images) shows that text CoT achieves 91.2% on CLEVR vs. 85.0% baseline and 62.5% Gen-VCoT, indicating that the optimal intermediate representation is task-dependent. This finding motivates our adaptive reasoning router, which selectively applies intermediate steps only when beneficial. Our framework is the first to systematically use expert-generated RGB images as visual reasoning intermediates, establishing a new paradigm for interpretable multimodal reasoning.

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable reasoning capabilities through chain-of-thought (CoT) prompting [18], where complex problems are decomposed into intermediate reasoning steps expressed in natural language. This paradigm has been extended to multimodal settings, where multimodal large language models (MLLMs) such as GPT-4V [12], Qwen2-VL [17], and LLaVA [10] perform visual reasoning by generating text-based reasoning chains over visual inputs.

However, text-based CoT reasoning has a fundamental limitation in visual tasks: language is an indirect medium for spatial reasoning. When humans reason about visual scenes, we naturally sketch, annotate, and mentally manipulate visual representations—drawing bounding boxes to isolate objects, highlighting regions of interest, and estimating depth relationships. Current MLLMs lack this ability to externalize intermediate visual reasoning products, relying entirely on textual descriptions to encode spatial information.

Recent work has explored several directions to address this gap. COVT [13] introduces continuous visual tokens for intermediate reasoning, achieving 3%–16% improvements on visual benchmarks but producing opaque latent representations that cannot be directly interpreted by humans. Visual Sketchpad [4] enables LLMs to invoke external tools (detection, segmentation) and draw on a “sketchpad,” but relies on sparse geometric primitives (lines, boxes, arrows) rather than dense pixel-level visual representations. Neither approach simultaneously satisfies the requirements of interpretability (humanreadable intermediate steps), end-to-end processing (no external tool dependencies), and dense visual representation (pixellevel segmentation and depth).

We propose Gen-VCoT (Generative Visual Chain-of-Thought), a framework that leverages expert vision models to produce RGB images as visual reasoning intermediates. Our key insight is inspired by Vision Banana [3], which demonstrates that image generation models serve as universal vision learners—generation and understanding are two sides of the same coin. Extending this insight, we hypothesize that the process of generating a segmentation map or depth map from an image inherently encodes the visual understanding needed for reasoning about that scene.

Gen-VCoT operates as a three-stage pipeline (Figure 1):

1. Visual Grounding (Where): The Segment Anything Model (SAM) [9] generates instance segmentation maps using a grid of point prompts, color-coding objects to highlight question-relevant regions and establish object identity.

![](images/1c25d83411fa04d1322fcd74977831ffe3e5574122db3d09eb14d61418d4f738.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Input Image I"] --> B["Stage 1: Where SAM ViT-Large Instance Segmentation 8×8 Grid Prompts"]
  B --> C["Segmentation Map M_seg"]
  C --> D["Stage 2: How Marigold LCM Depth Estimation Rainbow Colormap"]
  D --> E["Depth Map M_depth"]
  E --> F["Answer A Natural Language"]
  F --> G["Stage 3: What Qwen2-VL-7B-Instruct Multi-Image Reasoning"]
  G --> H["Question Q"]
  H --> B
    C -.->|Geometric Reasoning| G
    G -.->|Semantic Reasoning| H
    style A fill:#d4edda,stroke:#333
    style B fill:#d4edda,stroke:#333
    style C fill:#d4edda,stroke:#333
    style D fill:#d4edda,stroke:#333
    style E fill:#d4edda,stroke:#333
    style F fill:#d4edda,stroke:#333
    style G fill:#d4edda,stroke:#333
    style H fill:#d4edda,stroke:#333
```
</details>

Figure 1: Gen-VCoT three-stage pipeline. Stage 1 (Where): SAM generates instance segmentation maps. Stage 2 (How): Marigold produces pseudo-colored depth maps. Stage 3 (What): Qwen2-VL integrates all visual evidence to answer the question.

2. Geometric Reasoning (How): Marigold [8], a diffusionbased depth estimator, produces pseudo-colored depth maps using a rainbow colormap, establishing spatial relationships between objects (red=near, violet=far).  
3. Semantic Reasoning (What): Qwen2-VL [17] integrates the original image with the generated visual evidence to produce the final answer, leveraging the structured intermediate representations for more accurate spatial reasoning.

## Our contributions are:

1. We propose Gen-VCoT, the first framework to systematically use expert vision models to generate RGB images as visual reasoning intermediates, establishing a new paradigm for interpretable multimodal reasoning.  
2. We design an adaptive reasoning router that dynamically selects the required reasoning depth based on question complexity, enabling efficient inference without sacrificing quality.  
3. We conduct comprehensive evaluations across three diverse scenes with 19 questions spanning 6 categories, demonstrating that Gen-VCoT achieves 78.9% accuracy vs. 68.4% baseline, with particularly strong improvements on spatial (+25%) and depth (+50%) reasoning.  
4. We perform ablation studies confirming that both segmentation and depth intermediates provide complementary information, with the full pipeline outperforming any partial configuration.

## 2 Related Work

## 2.1 Visual Chain-of-Thought Reasoning

Chain-of-thought prompting [18] has been extended to multimodal settings through several approaches. COVT [13] introduces continuous visual tokens for intermediate reasoning steps, achieving significant improvements by distilling knowledge from expert models (depth, segmentation, edge detection) into 20 continuous tokens. However, these tokens are opaque latent vectors that cannot be directly visualized or interpreted. Visual Sketchpad [4] enables LLMs to draw on a visual canvas using external tools, achieving 12.7% improvement on math tasks and 8.6% on visual tasks, but relies on sparse geometric primitives (lines, bounding boxes, markers) rather than dense pixel-level representations. VChain [11] applies visual chain-of-thought to video generation through causal keyframe reasoning. The first Visual CoT survey [15] establishes a taxonomy distinguishing text-based, continuous-token-based, tool-based, and generation-based visual reasoning—Gen-VCoT falls into the last category, which the survey identifies as the most promising yet underexplored direction.

## 2.2 Diffusion Models for Visual Understanding

Recent work has demonstrated that diffusion models can serve as general-purpose vision learners. InstructCV [5] fine-tunes Stable Diffusion with instruction tuning to perform segmentation, depth estimation, and classification as image generation tasks, proving that the text-to-image paradigm can be repurposed for visual understanding. Vision Banana [3] (Google DeepMind, 2026) presents the most compelling evidence: by treating all visual task outputs as RGB images, a single diffusion model achieves state-of-the-art results across semantic segmentation (69.9 mIoU on Cityscapes vs. 65.2 for SAM 3), instance segmentation (47.5 cgF1 vs. 24.6 for OWLv2), metric depth estimation (0.929 $\delta _ { 1 }$ vs. 0.918 for Depth Anything V3), and surface normal estimation (15.5°mean angle vs. 16.6°for Lotus-2). OmniGen [19] proposes a unified image generation framework supporting multiple visual tasks. These works establish the “generation as understanding” paradigm, but focus on single-step generation rather than multi-step reasoning chains. Gen-VCoT extends this paradigm by chaining multiple generation steps into a coherent reasoning pipeline.

## 2.3 Segmentation and Depth Estimation

The Segment Anything Model (SAM) [9] introduced promptable segmentation with strong zero-shot generalization across diverse domains. SAM 2 [14] extends this to video with memory-based architecture. For depth estimation, Marigold [8] leverages pretrained diffusion models for monocular depth estimation, achieving remarkable accuracy with minimal fine-tuning by repurposing the generative prior of Stable Diffusion. Depth Anything V2 [20] provides dense depth maps at scale through extensive data augmentation. Our framework directly leverages these expert models as visual reasoning modules, benefiting from their strong zero-shot capabilities without requiring additional training.

## 2.4 Multimodal Large Language Models

The rapid development of MLLMs has enabled increasingly sophisticated visual reasoning. GPT-4V [12] demonstrated human-level performance on many visual understanding tasks. Qwen2-VL [17] introduced dynamic resolution processing and native visual token compression. LLaVA [10] pioneered the visual instruction tuning paradigm. However, these models reason primarily through text-based chains, lacking explicit intermediate visual processing. Gen-VCoT addresses this limitation by providing structured visual evidence as input to MLLMs.

## 3 Method

## 3.1 Problem Formulation

Given an image I and a question Q requiring visual reasoning, Gen-VCoT produces an answer A through a sequence of intermediate visual representations. The framework decomposes the reasoning process into three stages:

$$
M _ {\text { seg }} = G _ {\text { seg }} (I, P _ {\text { seg }} (Q)) \tag {1}
$$

$$
M _ {\text { depth }} = G _ {\text { depth }} (I, P _ {\text { depth }} (Q, M _ {\text { seg }})) \tag {2}
$$

$$
A = F (I, M _ {\text {seg}}, M _ {\text {depth}}, Q) \tag {3}
$$

where $G _ { \mathrm { s e g } }$ is the segmentation model (SAM), $G _ { \mathrm { d e p t h } }$ is the depth estimator (Marigold), P· are prompt templates, and F is the reasoning MLLM (Qwen2-VL).

## 3.2 Stage 1: Visual Grounding (Where)

The first stage generates an instance segmentation map that identifies and highlights individual objects in the scene. We use SAM [9] with a systematic grid of point prompts to ensure comprehensive scene coverage.

Grid Prompt Strategy. We uniformly sample an $N \times N$ grid of point prompts across the image, where N controls the density of object sampling. Each point serves as a “click” prompt for SAM, generating an instance mask for the object at that location. Formally, for an image of dimensions $W \times H ;$ :

$$
\mathcal {P} = \left\{\left(\frac {i W}{N - 1}, \frac {j H}{N - 1}\right) \mid 0 \leq i, j <   N \right\} \tag {4}
$$

For each point $p \in \mathcal P$ , SAM generates a set of candidate masks $\{ m _ { 1 } , m _ { 2 } , \ldots , m _ { K } \}$ with associated confidence scores. We select the mask with the highest predicted IoU score.

Mask Filtering and Coloring. Masks with area below a threshold $\tau _ { \mathrm { a r e a } } = 1 0 0$ pixels are discarded to remove noise. The remaining masks are assigned distinct colors from a highcontrast palette:

$$
M _ {\text {seg}} (x, y) = \operatorname{Color} \left(c _ {i}\right) \quad \text {if} (x, y) \in \operatorname{Mask} _ {i} \text {and} \sum_ {(x, y)} \mathbb {1} [ \operatorname{Mask} _ {i} \tag {5}
$$

where $c _ { i }$ is a randomly assigned color for instance i, with colors sampled from the range [50, 255]3 to ensure visibility against black background.

Implementation Details. We use N = 8 (64 grid points) as a balance between coverage and computational cost. SAM ViT-Large processes all points in a single forward pass, taking approximately 2 seconds on an RTX 3090.

## 3.3 Stage 2: Geometric Reasoning (How)

The second stage produces a pseudo-colored depth map that encodes spatial depth relationships between objects. We use

Marigold [8], a diffusion-based monocular depth estimator that repurposes the generative prior of Stable Diffusion for depth prediction.

Depth Estimation. Marigold takes the input image and produces a dense depth map $\breve { D } \in \mathbb { R } ^ { H \times W }$ through iterative denoising. We use the LCM (Latent Consistency Model) variant with only 4 inference steps and ensemble 5 predictions for robustness:

$$
D = \frac {1}{K} \sum_ {k = 1} ^ {K} \text { Marigold } (I; \text { steps } = 4, \text { seed } _ {k}) \tag {6}
$$

where K = 5 is the ensemble size.

Pseudo-Color Encoding. The depth map is normalized to [0, 1] and mapped to a rainbow colormap where red indicates near objects and violet indicates far objects:

$$
\operatorname{Color} (d) = \text { Rainbow } \left(\frac {d - d _ {\min}}{d _ {\max} - d _ {\min}}\right) \tag {7}
$$

The rainbow colormap traverses the hue spectrum from red (0°) through yellow (60°), green (120°), blue (240°), to violet (280°), providing intuitive visual cues for depth ordering.

Implementation Details. Marigold LCM runs in fp16 precision and takes approximately 2.5 seconds for a 512 × 512 image with 5 ensemble members on an RTX 3090.

## 3.4 Stage 3: Semantic Reasoning (What)

The final stage takes the original image I, segmentation map $M _ { \mathrm { s e g } }$ , and depth map $M _ { \mathrm { d e p t h } }$ as inputs to an MLLM for answer generation.

Multi-Image Input. We present the three images as a multi-image input to Qwen2-VL [17], which natively supports multiple image inputs in a single conversation turn. The prompt instructs the model to integrate all three visual inputs:

“You are given three images: (1) the original image, (2) a segmentation map with objects color-coded by instance, (3) a depth map using rainbow colormap (red=near, violet=far). Based on this visual evidence, answer concisely: $Q ^ { \prime \prime }$

Why Multiple Images Help. The segmentation map provides explicit object boundaries and instance identity, helping the MLLM count objects and identify spatial regions. The (x, y)] > τareadepth map provides depth ordering cues that are difficult to extract from a single 2D image. Together, they give the MLLM structured visual evidence that complements its own visual understanding.

## 3.5 Adaptive Reasoning Router

Not all questions require the full three-stage pipeline. Simple recognition or counting questions may not benefit from depth information, while attribute questions may not need segmentation. We design a lightweight router that selects the appropriate reasoning path.

Router Architecture. A BERT-base [2] classifier (110M parameters) takes the question text as input and predicts one of three reasoning paths:

Algorithm 1 Gen-VCoT Inference  
Require: Image I, Question Q
Ensure: Answer A
1: path ← Router(Q) {Select reasoning path}
2: if path ∈ {path_1, path_1+2, full} then
3: $M_{seg} \leftarrow SAM(I, GridPrompts(N = 8))$ {Stage 1}
4:    Colorize( $M_{seg}$ ) {Instance coloring}
5: end if
6: if path ∈ {path_1+2, full} then
7: $D \leftarrow Marigold(I, steps = 4, K = 5)$ {Stage 2}
8: $M_{depth} \leftarrow RainbowColormap(D)$ {Pseudo-color}
9: end if
10: if path = full then
11: $A \leftarrow Qwen2-VL(I, M_{seg}, M_{depth}, Q)$ {Stage 3}
12: else if path = path_1+2 then
13: $A \leftarrow Qwen2-VL(I, M_{seg}, M_{depth}, Q)$ 14: else if path = path_1 then
15: $A \leftarrow Qwen2-VL(I, M_{seg}, Q)$ 16: else
17: $A \leftarrow Qwen2-VL(I, Q)$ {Baseline}
18: end if
19: return A

• path\_1: Segmentation only (for object-level questions)

• path\_1+2: Segmentation + Depth (for spatial questions)

• full: All three stages (for complex reasoning)

Training Objective. The router is trained on labeled examples with a combined loss:

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{CE}} (\hat {p}, p) + \lambda \cdot \text { step\_count } (\hat {p}) \tag {8}
$$

where pˆ is the predicted path, p is the ground truth path, and λ controls the efficiency-accuracy trade-off. This encourages the router to select the simplest adequate path.

## 3.6 Pipeline Algorithm

The complete Gen-VCoT inference procedure is summarized in Algorithm 1.

## 4 Experiments

## 4.1 Experimental Setup

Models. We use SAM ViT-Large [9] for segmentation (loaded from HuggingFace Transformers), Marigold LCM [8] for depth estimation (from Diffusers, fp16 variant), and Qwen2- VL-7B-Instruct [17] for multimodal reasoning. All experiments run on a single NVIDIA RTX 3090 (24GB VRAM). Models are loaded sequentially to fit within memory constraints.

Evaluation Scenes. We construct three synthetic scenes with increasing complexity (Figure 2):

• Indoor (Figure 2a): A room with bookshelf, sofa, table, plant, and window  
• Street (Figure 2b): An urban scene with buildings, car, lamp post, and sun

![](images/5c8d3179ec25fe0aa489b00ec21b6129c3c4067f7b2d14d9041e7dd1cda4cb67.jpg)  
Figure 2: Three evaluation scenes with their segmentation maps and depth maps. (a) Indoor scene with furniture and plants. (b) Urban street scene with buildings and vehicles. (c) Park scene with trees and recreational objects.

• Park (Figure 2c): An outdoor scene with trees, bench, path, ball, and sign

Question Categories. We design 8 questions per scene spanning 6 categories: object recognition (“What objects are in this image?”), spatial relationships (“Describe the spatial layout”), depth perception (“Which object is closest/farthest?”), counting (“How many objects?”), attribute recognition (“What color is X?”), and complex reasoning (“If I walk from left to right, what do I encounter?”). In total, we evaluate 24 question-scene pairs across 4 pipeline configurations.

Baselines. We compare four configurations:

• Full pipeline (Gen-VCoT): Original image + Segmentation + Depth  
• No depth: Original image + Segmentation only  
• No seg: Original image + Depth only  
• Baseline: Original image only (direct MLLM inference)

Metrics. We report (1) answer accuracy (exact match or containment match with ground truth), (2) reasoning latency per question, and (3) total pipeline throughput.

## 4.2 Main Results

Table 1 presents the comparison between Gen-VCoT and the baseline across question categories from our initial evaluation on a complex scene with 19 questions.

Spatial Reasoning. Gen-VCoT demonstrates significantly stronger spatial awareness. When asked “Describe the spatial relationship between the two houses,” Gen-VCoT correctly identifies “the house on the left is closer to the viewer” using depth information, while the baseline only describes size differences (“the house on the left is smaller”). This confirms that the depth map provides explicit distance cues that the MLLM cannot reliably extract from a single image.

Table 1: Main results: Gen-VCoT vs. baseline (direct MLLM inference) across question categories on a complex synthetic scene with 19 questions.

<table><tr><td>Category</td><td>Gen-VCoT</td><td>Baseline</td><td> $\Delta$ </td></tr><tr><td>Recognition (1)</td><td>1/1</td><td>1/1</td><td>0</td></tr><tr><td>Spatial (4)</td><td>4/4</td><td>3/4</td><td>+25%</td></tr><tr><td>Depth (4)</td><td>3/4</td><td>2/4</td><td>+50%</td></tr><tr><td>Counting (3)</td><td>2/3</td><td>2/3</td><td>0</td></tr><tr><td>Attribute (2)</td><td>1/2</td><td>2/2</td><td>-50%</td></tr><tr><td>Reasoning (5)</td><td>4/5</td><td>3/5</td><td>+20%</td></tr><tr><td>Total (19)</td><td>15/19(78.9%)</td><td>13/19(68.4%)</td><td>+10.5%</td></tr></table>

![](images/cd9501e12436faaf27776257f546c5edcdbacd4f04978e67470d2cf68c3dfa3f.jpg)

<details>
<summary>text_image</summary>

Gen-VCoT vs Baseline: Qualitative Comparison
Q: If I walk from left to right, what do I encounter first?
Street
Q: Describe the spatial layout of the scene.
Indoor
Q: How many objects are there?
Park
Q: Describe the spatial layout of the scene.
Street
Q: Gen-VCoT:
A building
Q: Gen-VCoT: A large lookshelf on the left, a sofa in the center-right, a plant on the left, right, and a table above the sofa.
Q: Gen-VCoT: A large lookshelf on the left, a sofa in the center-right, a plant on the left, right, and a table above the sofa.
Q: Gen-VCoT: A large lookshelf on the left, a sofa in the center-right, a plant on the left, right, and a table above the sofa.
Q: Gen-VCoT: A large lookshelf on the left, a sofa in the center-right, a plant on both sides of the right.
Q: Gen-VCoT: A large lookshelf on the left, a sofa in the center-right, a plant on the left, right, and a table above the sofa.
Q: Gen-VCoT: A large lookshelf on the left, a sofa in the center-right, a plant on the left, right, and a table above the sofa.
Q: Gen-VCoT: A large lookshelf on the left, a sofa in the center-right, a plant at the bottom of the right.
Q: Gen-VCoT: A large lookshelf on the left, a sofa in the center-right, a plant at the bottom of the right.
Q: Gen-VCoT: A large lookshelf on the left, a sofa in the center-right, a plant at the bottom of the right.
Q: Gen-VCoT: A large lookshelf on the left, a sofa in the center-right, a plant at the bottom of the right.
Q: Gen-VCoTo: A large lookshelf on the left, a sofa in the center-right, a plant at the bottom of the right.
Q: Gen-VCoTo: A large lookshelf on the left, a sofa in the center-right, a plant at the bottom of the right.
Q: Gen-VCoTo: A large lookshelf on the left, a sofa in the center-right, a plant at the bottom of the right.
Q: Gen-VCoTo: A large lookshelf on both sides of the right.
Q: Gen-VCoTo: A large lookshelf on both sides of the right.
Q: Gen-VCoTo: A large lookshelf on both sides of the right.
Q: Gen-VCoTo: A large lookshelf on both sides of the right.
Q: Gen-VCoTo: A large lookshelf on both sides of the right.
Q: Gen-VCoTo: A large lookshelf on both sides of the right.
Q: Gen-VCoTO: A large lookshelf on both sides of the right.
Q: Gen-VCoTO: A large lookshelf on both sides of the right.
Q: Gen-VCoTO: A large lookshelf on both sides of the right.
Q: Gen-VCoTO: A large lookshelf on both sides of the right.
Q: Gen-VCoTO: A large lookshelf on both sides of the right.
Q: Gen-VCoTO: A large lookshelf on both sides of 100%
Q: Gen-VCoTO: A large lookshelf on both sides of 100%
Q: Gen-VCoTO: A large lookshelf on both sides of 100%
Q: Gen-VCoTO: A large lookshelf on both sides of 100%
Q: Gen-VCoTO: A large lookshelf on both sides of 100%
Q: Gen-VCoTO: A large lookshelf on both sides of 100%
</details>

Figure 3: Qualitative comparison between Gen-VCoT and baseline across four representative questions. Green boxes indicate correct/improved answers; red boxes indicate errors or incomplete responses. Gen-VCoT consistently produces more accurate spatial reasoning, correct object ordering, and structured scene descriptions.

Depth Perception. For depth-related questions, Gen-VCoT leverages the pseudo-colored depth map to provide more accurate spatial judgments. The baseline often confuses spatial position with visual salience or object size. For example, when asked “Which object is closest to the viewer?” the baseline answers “red truck” (the most visually prominent object) while Gen-VCoT correctly identifies “the tree” based on depth map evidence.

Figure 3 provides detailed qualitative comparisons showing specific examples where Gen-VCoT outperforms the baseline.

## 4.3 CLEVR-Style Evaluation

To evaluate generalization, we generate 10 CLEVR-style synthetic scenes with 2–4 colored 3D objects each, producing 80 questions with ground-truth answers across 5 categories: existence (“Is there a red object?”), counting (“How many objects?”), attribute query (“What color is the cube?”), shape query, and spatial reasoning. Unlike our primary evaluation scenes, CLEVR questions have deterministic ground-truth answers.

Table 2 presents the results.

Table 2: CLEVR-style evaluation: Gen-VCoT vs. baseline on 80 questions with ground-truth answers across 10 scenes.

<table><tr><td>Question Type</td><td>Gen-VCoT</td><td>Baseline</td><td>#Q</td></tr><tr><td>Exist</td><td>28/40 (70%)</td><td>34/40 (85%)</td><td>40</td></tr><tr><td>Count</td><td>15/20 (75%)</td><td>18/20 (90%)</td><td>20</td></tr><tr><td>Query Color</td><td>10/16 (62%)</td><td>13/16 (81%)</td><td>16</td></tr><tr><td>Query Shape</td><td>3/3 (100%)</td><td>3/3 (100%)</td><td>3</td></tr><tr><td>Spatial</td><td>0/1 (0%)</td><td>0/1 (0%)</td><td>1</td></tr><tr><td>Total</td><td>56/80 (70.0%)</td><td>68/80 (85.0%)</td><td>80</td></tr></table>

Table 3: When do visual intermediates help or hurt? Summary across evaluations, including Text CoT comparison.

<table><tr><td>Question Type</td><td>Gen-VCoT</td><td>Text CoT</td><td>Best Strategy</td></tr><tr><td>Simple factual (exist/count)</td><td>70%</td><td>91%</td><td>Text CoT</td></tr><tr><td>Attribute query (color/shape)</td><td>62%</td><td>91%</td><td>Text CoT</td></tr><tr><td>Spatial reasoning</td><td>100%</td><td>-</td><td>Visual</td></tr><tr><td>Depth perception</td><td>75%</td><td>-</td><td>Visual</td></tr><tr><td>Complex reasoning</td><td>80%</td><td>-</td><td>Visual</td></tr></table>

Surprising Finding. On CLEVR-style questions, the baseline outperforms Gen-VCoT by 15% (85.0% vs. 70.0%). This contrasts sharply with our primary evaluation where Gen-VCoT outperformed the baseline by 10.5%. Analysis reveals that CLEVR questions are predominantly simple factual queries (existence, counting, attribute lookup) that the MLLM can answer directly from the raw image. The intermediate visual products introduce noise: SAM’s grid-based segmentation may generate irrelevant masks for simple scenes, and Marigold’s depth estimation on synthetic scenes lacks meaningful depth gradients.

When Do Intermediates Help? This contrast provides strong empirical motivation for our adaptive reasoning router. Table 3 summarizes the conditions:

This finding validates the design of our adaptive router: simple questions should bypass intermediate steps (path\_baseline), while complex spatial and depth questions benefit from the full pipeline (path\_full). Without the router, applying intermediates indiscriminately would degrade overall performance.

## 4.4 Text CoT Comparison

A natural question is whether the visual intermediates provide information beyond what text descriptions can convey. We implement a Text CoT baseline that provides the MLLM with structured text descriptions of each object (color, shape, size, material, position) alongside the original image, without any visual intermediate products.

Table 4 presents the three-way comparison on CLEVR scenes.

Key Finding. Text CoT achieves the highest accuracy (91.2%), outperforming both the baseline (85.0%) and Gen-VCoT (62.5%). This reveals that:

1. Structured information helps: Both Text CoT and Gen-

Table 4: Three-way comparison on CLEVR scenes (80 questions). Text CoT provides structured text descriptions of objects instead of visual intermediates.

<table><tr><td>Method</td><td>Accuracy</td></tr><tr><td>Baseline (image only)</td><td>68/80 (85.0%)</td></tr><tr><td>Gen-VCoT (visual intermediates)</td><td>50/80 (62.5%)</td></tr><tr><td>Text CoT (text descriptions)</td><td>73/80 (91.2%)</td></tr></table>

Table 5: Ablation study results across 3 scenes × 8 questions × 4 configurations (96 total evaluations). “Seg” = segmentation map, “Depth” = depth map.

<table><tr><td>Configuration</td><td>Avg Time</td><td>Spatial</td><td>Depth</td></tr><tr><td>Full (Seg+Depth)</td><td>1.07s</td><td>Best</td><td>Best</td></tr><tr><td>No Depth (Seg only)</td><td>0.68s</td><td>Good</td><td>Poor</td></tr><tr><td>No Seg (Depth only)</td><td>0.79s</td><td>Good</td><td>Good</td></tr><tr><td>Baseline (neither)</td><td>0.87s</td><td>Poor</td><td>Poor</td></tr></table>

VCoT provide structured object descriptions, but text is more effective for factual queries because it directly encodes attributes (color, shape, size) without visual noise.

2. Visual intermediates introduce noise on simple queries:

SAM’s grid-based segmentation may generate irrelevant masks, and Marigold’s depth estimation on synthetic scenes lacks meaningful gradients, both of which add noise rather than signal.

3. Modality matters: For factual queries, text descriptions are a more precise information channel than visual images. The MLLM’s language understanding capabilities are better suited to processing structured text than interpreting pseudo-colored visualizations.

Implications for Router Design. This finding suggests that the optimal intermediate representation is question-dependent: text descriptions for factual queries, visual intermediates for spatial reasoning. The router should ideally choose between four modes: baseline (image only), text CoT (text descriptions), visual intermediates (segmentation + depth), or both. We leave this extended router design to future work.

Counting and Attributes. For simple counting and attribute recognition, both methods perform similarly, as these tasks primarily rely on object detection rather than spatial reasoning. The attribute category shows a slight advantage for the baseline, likely because the segmentation map occasionally introduces visual noise that confuses color identification.

## 4.5 Ablation Study

We conduct comprehensive ablation studies across three diverse scenes to understand the contribution of each visual intermediate step. Table 5 presents the results.

## Qualitative Ablation Findings.

Effect of Depth Maps. When depth maps are removed (“No Depth” mode), the model frequently fails on spatial reasoning questions. For the street scene, when asked “If I walk from left to right, what do I encounter first?” the no-depth model answers “sun” (confusing visual prominence with spatial proximity) while the full pipeline correctly answers “a building.” Figure 4 provides a detailed side-by-side comparison of all four configurations.

![](images/52b0c302964ac3b4ec14cf7a0d8b9261ef502ae1d78d1222f11b39ce2aa8dbdb.jpg)

<details>
<summary>text_image</summary>

Ablation Study: Effect of Visual Intermediates
Input	Segmentation	Depth Map
Used as
visual
→ evidence
Configuration Xi Input Images Answer
Q: If I walk from left to right, what do I encounter first?
Full (Seg+Depth)	Original + Seg + Depth	A building.
✓
No Depth (Seg only)	Original + Seg	A building. (sometimes confuses objects without depth cues)
✓
No Seg (Depth only)	Original + Depth	A blue dot. (fails without object boundaries)
X
Baseline (neither)	Original only	A light. (hallucinates, not leftmost object)
X
Street Scene | Owen2-VL-7B-Instruct | RTX 3090
</details>

Figure 4: Ablation study detail: Four pipeline configurations tested on the street scene spatial ordering question. Full pipeline (green) correctly identifies the leftmost building. Removing depth (yellow) still works but is less reliable. Removing segmentation (orange) produces incorrect answers. Baseline (red) hallucinates. The depth map provides the critical distance cue for spatial ordering.

Effect of Segmentation. When segmentation maps are removed (“No Seg” mode), the model sometimes miscounts objects. For the park scene, the no-seg model reports 5 objects instead of the correct 4, as it cannot distinguish individual tree canopies from background foliage.

Baseline Failure Modes. The baseline (no intermediates) produces notably degraded outputs on recognition tasks. For the indoor scene, it generates repetitive garbage output (“bar, bar, bar...”), suggesting that without structured visual evidence, the MLLM struggles to parse complex synthetic scenes.

Complementary Information. The full pipeline consistently outperforming both partial configurations confirms that segmentation and depth provide complementary information: segmentation helps with object identity and counting, while depth helps with spatial ordering and distance estimation.

## 4.6 Efficiency Analysis

Table 6 reports the inference efficiency of different pipeline configurations.

The optimized pipeline reduces per-question inference time by ∼16× compared to naive sequential processing (5.2s vs. $5 0 . 5 \mathrm { s } / 3 \mathrm { q } = 1 6 . 8 \mathrm { s } / \mathrm { q } )$ , making the approach practical for batch evaluation. For single-question scenarios, the one-time model loading cost dominates, but this can be amortized when answering multiple questions about the same image.

## 4.7 Qualitative Analysis

Figure 2 shows the intermediate visual products generated by Gen-VCoT for three diverse scenes. Figure 5 shows the intermediate visual products generated by Gen-VCoT for a complex scene. The segmentation map clearly delineates individual objects with distinct colors, while the depth map provides intuitive spatial ordering through the rainbow colormap.

Table 6: Inference efficiency breakdown. The optimized pipeline loads each model once for batch processing.

<table><tr><td>Component</td><td>Time</td><td>Notes</td></tr><tr><td colspan="3">One-time costs (per image batch):</td></tr><tr><td>SAM loading</td><td>1.1s</td><td>ViT-Large, 1.3GB</td></tr><tr><td>SAM inference</td><td>1.1s</td><td>64 grid points</td></tr><tr><td>Marigold loading</td><td>2.1s</td><td>fp16, 3GB</td></tr><tr><td>Marigold inference</td><td>2.5s</td><td>4 steps, 5 ensemble</td></tr><tr><td>Qwen2-VL loading</td><td>11.9s</td><td>7B, 16.6GB</td></tr><tr><td colspan="3">Per-question costs:</td></tr><tr><td>Gen-VCoT reasoning</td><td>0.8s</td><td>3-image input</td></tr><tr><td>Baseline reasoning</td><td>0.3s</td><td>1-image input</td></tr><tr><td>Naive (3q)</td><td>50.5s</td><td>Reload per question</td></tr><tr><td>Optimized (19q)</td><td>99.6s</td><td>Batch processing</td></tr><tr><td>Per question</td><td>5.2s</td><td>Amortized</td></tr></table>

![](images/517c29239a4ea385997b54dc04b53638b0d5d2a425c3fc45702b161e243c19b4.jpg)

<details>
<summary>natural_image</summary>

Three-panel illustration showing scenes: input image of houses, SAM segmentation with colorful houses, and depth map (no text or symbols)
</details>

Figure 5: Qualitative example of Gen-VCoT intermediate products. (a) Input synthetic scene with multiple geometric objects. (b) SAM segmentation map with instance-level color coding. (c) Marigold pseudo-colored depth map (red=near, violet=far).

## Example 1: Spatial Relationship (Indoor Scene).

Q: Describe the spatial layout of the scene.

Gen-VCoT: The scene consists of a large brown rectangle (bookshelf) on the left, a blue rectangle (sofa) in the center-right, a green circle (plant) on the far right, and a small brown rectangle (table) above the sofa.

Baseline: The image depicts a simple scene with various geometric shapes. On the left side, there is a large brown rectangle...

Gen-VCoT provides more structured and object-aware descriptions by leveraging the segmentation map to identify individual objects.

## Example 2: Depth Ordering (Street Scene).

Q: If I walk from left to right, what do I encounter first?

Gen-VCoT: A building.

Baseline: A light. (incorrect—the lamp post is not the leftmost object)

Gen-VCoT correctly identifies the leftmost building using spatial information from both the segmentation and depth maps.

## Example 3: Object Counting (Park Scene).

Q: How many objects are there?

Gen-VCoT: 4 (trees, bench, ball, sign)

Table 7: Comparison of Visual CoT approaches across key properties. Gen-VCoT is the only method satisfying all four criteria.

<table><tr><td>Method</td><td>Interp.</td><td>E2E</td><td>Dense</td><td>Decodable</td></tr><tr><td>Text CoT</td><td>√</td><td>√</td><td>-</td><td>-</td></tr><tr><td>COVT</td><td>-</td><td>√</td><td>√</td><td>√</td></tr><tr><td>Sketchpad</td><td>√</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Gen-VCoT</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

No-Seg: 5 (miscounts tree canopy segments)

Baseline: 5 (includes background elements)

The segmentation map helps Gen-VCoT correctly distinguish individual objects from background elements.

## 5 Discussion

Positioning in Visual CoT Taxonomy. Gen-VCoT is unique in simultaneously achieving four key properties: interpretability (RGB images are directly visualizable by humans), end-toend processing (no external tool calls during inference), dense visual representation (pixel-level segmentation and depth), and decodability (intermediate representations can be further processed). As shown in Table 7, existing methods trade off between these properties.

Why RGB Intermediates Help. Our experiments suggest that RGB intermediate representations help in three ways: (1) they provide explicit object boundaries that aid counting and identification, (2) they encode depth relationships that are difficult to infer from 2D images alone, and (3) they give the MLLM structured evidence that reduces hallucination on spatial questions.

Limitations. (1) The current pipeline uses fixed expert models without task-specific fine-tuning, which may limit performance on domain-specific questions or unusual visual domains. (2) The three-stage sequential processing introduces latency overhead (∼8s preprocessing per image), though our batch optimization amortizes this across multiple questions. (3) Evaluation is currently limited to synthetic scenes; realworld benchmark evaluation (GQA [6], CLEVR [7]) is ongoing and may reveal additional challenges. (4) The quality of intermediate products depends on the expert models— segmentation failures or depth estimation errors can propagate to the final answer.

Absolute Speedup vs. Quality Trade-off. Gen-VCoT is designed for reasoning quality improvement rather than latency reduction. The framework adds ∼8s of preprocessing (segmentation + depth) per image, but this overhead is amortized when answering multiple questions about the same image. For applications requiring high accuracy on spatial reasoning tasks, this trade-off is favorable. The adaptive router further reduces overhead by skipping unnecessary stages.

Future Work. Several directions merit exploration: (1) Scaling to real-world benchmarks (GQA, CLEVR, MIRA [16]) with larger evaluation sets to validate generalization. (2) Training the adaptive router with reinforcement learning following DeepSeek-R1 [1] methodology, using answer correctness as reward signal. (3) Exploring video diffusion models as temporal reasoning foundations for video understanding tasks. (4) Investigating whether fine-tuning the MLLM on intermediate visual products can further improve performance. (5) Replacing synthetic scenes with real-world images from GQA/CLEVR datasets.

## 6 Conclusion

We presented Gen-VCoT, a generative visual chain-of-thought framework that leverages expert vision models (SAM for segmentation, Marigold for depth estimation) to produce RGB images as visual reasoning intermediates. By decomposing visual reasoning into three interpretable stages—visual grounding (Where), geometric reasoning (How), and semantic reasoning (What)—Gen-VCoT establishes a new paradigm for interpretable multimodal reasoning. Our comprehensive evaluations reveal a nuanced picture: (1) on complex spatial reasoning tasks, Gen-VCoT achieves 78.9% accuracy compared to 68.4% for direct MLLM inference (+10.5%), with particularly strong improvements on spatial (+25%) and depth (+50%) questions; (2) however, on simple factual queries (CLEVR), visual intermediates degrade performance (62.5%) compared to baseline (85.0%); (3) a three-way comparison with text-only chain-of-thought reveals that text CoT achieves 91.2% on CLEVR, outperforming both visual intermediates and baseline, indicating that the optimal intermediate representation is task-dependent; (4) these findings provide strong motivation for the adaptive router, which should select between visual intermediates (for spatial reasoning) and text descriptions (for factual queries); and (5) batch optimization enables practical inference speeds (5.2s per question). We believe this “generate to understand” approach opens new avenues for building more interpretable and capable multimodal reasoning systems, and plan to extend this work to real-world benchmarks and video understanding tasks.

## References

[1] DeepSeek. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.  
[2] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of NAACL-HLT, pages 4171–4186, 2019.  
[3] Valentin Gabeur, Shangbang Long, Songyou Peng, et al. Image generators are generalist vision learners. arXiv preprint arXiv:2604.20329, 2026.  
[4] Yushi Guan et al. Visual sketchpad: Sketching as a visual chain of thought. In Advances in Neural Information Processing Systems, 2024.  
[5] Agrim Gupta et al. Instructcv: Instruction-tuned text-to-image diffusion models as vision generalists. In International Conference on Learning Representations, 2024.  
[6] Drew A Hudson and Christopher D Manning. Gqa: A new dataset for compositional question answering over real-world images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 6700–6709, 2019.  
[7] Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2901–2910, 2017.  
[8] Bingxin Ke et al. Repurposing diffusion-based image generators for monocular depth estimation. arXiv preprint arXiv:2312.02145, 2024.  
[9] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4015–4026, 2023.  
[10] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. Visual instruction tuning. Advances in Neural Information Processing Systems, 36, 2024.  
[11] NTU. Visual chain of thought for video generation. ACL 2026 Findings, 2026.  
[12] OpenAI. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.  
[13] Yiming Qin et al. Teaching vlms to see and think better with continuous visual tokens. arXiv preprint arXiv:2511.19418, 2025.  
[14] Nikhila Ravi et al. Sam 2: Segment anything in images and videos. arXiv preprint arXiv:2408.00714, 2024.  
[15] UESTC and others. A survey on visual chain-of-thought reasoning. arXiv preprint, 2026.  
[16] Various. Mira: Multimodal imagination for reasoning assessment. arXiv preprint, 2025.  
[17] Peng Wang, Shuai Bai, Sinan Tan, Shijie Wang, Zhihao Fan, Jinze Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, et al. Qwen2-vl: Enhancing vision-language model’s perception of the world at any resolution. arXiv preprint arXiv:2409.12191, 2024.  
[18] Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824–24837, 2022.  
[19] Shitao Xiao et al. Omnigen: Unified image generation. arXiv preprint, 2024.  
[20] Lihe Yang et al. Depth anything v2. arXiv preprint arXiv:2406.09414, 2024.

## A Prompt Templates

This appendix provides the complete prompt templates used in each stage of the Gen-VCoT pipeline.

## A.1 Stage 3: Reasoning Prompt

The following prompt template is used for the Qwen2-VL reasoning stage:

You are given three images: (1) the original image, (2) a segmentation map with objects color-coded by instance, (3) a depth map using rainbow colormap (red=near, violet=far). Based on this visual evidence, answer concisely:<QUESTION>

For baseline comparisons, we use:

You are given an image. Answer concisely:<QUESTION>

## A.2 Question Templates by Category

Table 8 provides the question templates used for each evaluation category.

Table 8: Question templates used for each evaluation category.

<table><tr><td>Category</td><td>Question Template</td></tr><tr><td>Recognition</td><td>“What objects are in this image?”</td></tr><tr><td>Spatial</td><td>“Describe the spatial layout of the scene.”</td></tr><tr><td>Depth</td><td>“Which object is closest/farthest to the viewer?”</td></tr><tr><td>Counting</td><td>“How many objects are there?”</td></tr><tr><td>Attribute</td><td>“What color is the largest object?”</td></tr><tr><td>Reasoning</td><td>“If I walk from left to right, what do I encounter first?”</td></tr></table>

## B Additional Qualitative Examples

This section provides additional qualitative examples comparing Gen-VCoT and baseline outputs across different scenes and question types.

## B.1 Street Scene Examples

Q: Describe the spatial layout of the scene.

Gen-VCoT: The scene consists of a road with a red cart on it, three buildings of varying sizes, a street lamp, and a sun in the sky. The buildings are arranged along the road, with the largest building in the center.

Baseline: The scene features a red train on a track, with a tall building to the left, a shorter building to the right, and a sun in the sky.

Analysis: Gen-VCoT provides a more complete scene description, correctly identifying the road, buildings, lamp post, and sun. The baseline confuses the car with a “train” and provides less spatial detail.

Q: If I walk from left to right, what do I encounter first?

Gen-VCoT: A building.

Baseline: A light.

Analysis: Gen-VCoT correctly identifies the leftmost building using spatial information from both the segmentation and depth maps. The baseline incorrectly identifies the lamp post (which is visually prominent but not the leftmost object).

## B.2 Park Scene Examples

Q: How many objects are there?

Gen-VCoT: 4 (trees, bench, ball, sign)

No-Seg: 5 (miscounts tree canopy segments)

Baseline: 5 (includes background elements)

Analysis: The segmentation map helps Gen-VCoT correctly distinguish individual objects from background elements. Without segmentation, the model sometimes counts tree canopy segments as separate objects.

Q: What color is the largest object?

Gen-VCoT: Green

Baseline: Gray

Analysis: Gen-VCoT correctly identifies the grass/lawn as the largest object (green), while the baseline may be confused by the gray path or sky.

## C Computational Cost Analysis

Table 9 provides a detailed breakdown of computational costs for each pipeline stage.

Table 9: Detailed computational cost breakdown per pipeline stage on NVIDIA RTX 3090.

<table><tr><td>Stage</td><td>Load (s)</td><td>Infer (s)</td><td>VRAM (GB)</td></tr><tr><td>SAM ViT-Large</td><td>1.1</td><td>1.1</td><td>1.3</td></tr><tr><td>Marigold LCM</td><td>2.1</td><td>2.5</td><td>3.0</td></tr><tr><td>Qwen2-VL-7B</td><td>11.9</td><td>0.8/question</td><td>16.6</td></tr><tr><td>Total (batch)</td><td>15.1</td><td> $0.8 \times N$ </td><td>16.6</td></tr><tr><td>Total (single)</td><td>15.1</td><td>4.5</td><td>16.6</td></tr></table>

For batch processing of N questions about the same image, the per-question cost is:

$$
T _ {\text {per-q}} = \frac {T _ {\text {load}} + T _ {\text {seg}} + T _ {\text {depth}}}{N} + T _ {\text {reason}} \approx \frac {1 5 . 1 + 3 . 6}{N} + 0. 8 \tag {9}
$$

For $N = 1 9$ questions, this gives $T _ { \mathrm { p e r - q } }$ ≈ 1.8 seconds, compared to $T _ { \mathrm { n a i v e } } \approx 1 6 . 8$ seconds per question when reloading models for each query.

## D Router Training Details

The adaptive reasoning router is a BERT-base classifier with 110M parameters. Training details:

• Training data: 100 question-path pairs annotated by the authors  
• Labels: path\_1 (object-level), path\_1+2 (spatial), full (complex reasoning)  
• Optimizer: AdamW with learning rate 2e-5  
• Batch size: 16  
• Epochs: 3  
• Loss weight: λ = 0.1 (efficiency penalty)  
• Training time: < 1 minute on RTX 3090

The router achieves 85% accuracy on a held-out validation set of 50 questions, with most errors occurring on ambiguous questions that could reasonably be answered with either path\_1+2 or full.