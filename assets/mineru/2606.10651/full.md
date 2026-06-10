# Kwai Keye-VL-2.0 Technical Report

## Keye Team, Kuaishou Group

![](images/eea08bbd47d7915410debb27c619731ac6f7fe17372b8c1b3a105cc5a3c67b3c.jpg)

https://kwai-keye.github.io/ https://huggingface.co/Kwai-Keye https://github.com/Kwai-Keye/Keye

## Abstract

We introduce Kwai Keye-VL-2.0-30B-A3B, an open-source Mixture-of-Experts (MoE) multimodal foundation model designed to advance long-video understanding and agentic intelligence. To address the challenges of ultra-long contexts, information redundancy, and prohibitive computational costs inherent in hour-level videos, Keye-VL-2.0 is the first to adapt DeepSeek Sparse Attention (DSA) to GQA-based multimodal architectures, enabling lossless 256K context processing while capturing critical frames and long-range temporal dependencies. This architecture is underpinned by a highly optimized training and inference infrastructure, including scalable video I/O, heterogeneous ViT-LM parallelism, and custom DSA kernels that significantly maximize throughput and minimize computational overhead. Furthermore, to overcome the algorithmic dilemma of catastrophic forgetting during multi-task alignment, we introduce Cross-Modal Multi-Teacher On-Policy Distillation (MOPD) paired with Context-RL and Video-RL. By distilling dense token-level teacher feedback from on-policy rollouts back into the MoE backbone—which activates only 3B parameters—Keye-VL-2.0 natively empowers advanced agent collaboration across Code, Tool, and Search scenarios with multimodal self-correction. Extensive evaluations across video understanding, temporal grounding, reasoning, STEM, and agent benchmarks demonstrate that Keye-VL-2.0-30B-A3B achieves state-of-the-art performance among models of similar scale, particularly excelling in fine-grained temporal localization on TimeLens and long-video comprehension on Video-MME-v2 and LongVideoBench. We release our model checkpoints to accelerate community progress toward scalable and robust multimodal agentic applications.

![](images/55f38ac5c643818097232b477e02f2edbc579f707384f3a6e4b62a0a361e88d9.jpg)

<details>
<summary>bar chart</summary>

| Model | Open-Source | Qwen3.5-35B-A3B | Qwen3-VL-235B-A22B | Closed-Source | Gemini-3-Flash |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ActivityNet-TimeLens* | 58.5 | 53.2 | 52.1 | 57.0 | |
| QVHighlights-TimeLens* | 70.1 | 65.7 | 64.6 | 49.5 | 61.2 |
| Charades-TimeLens* | 58.4 | 49.1 | 47.8 | | |
| LongVideoBench | 74.1 | 61.6 | 70.5 | 81.1 | |
| Video-MME-v2 | 42.4 | 28.5 | 36.8 | 52.4 | |
| Video-MME-v2 - Non-Lin Score | 24.2 | 12.2 | 28.1 | 32.9 | |
</details>

LongVideoBench: sampled at 2 fps for all models. Video-MME-v2 (incl. Non-Lin Score): 512 sampled frames for all models, except Gemini-3-Flash at 1 fps. TimeLens (ActivityNet / QVHighlights / Charades): sampled at 4 fps for all models, except Qwen3-VL-235B-A22B at 2 fps. \* Scores are reported as mIoU: mean temporal Intersection-over-Union between predicted and ground-truth moments; higher is better.

Figure 1: Performance Comparison of Keye-VL-2.0-30B-A3B. Our model demonstrates leading capabilities against open-source models (e.g., Qwen3.5-35B-A3B, Qwen3-VL-235B-A22B) and closed-source models (Gemini-3-Flash) across fine-grained temporal localization (ActivityNet, QVHighlights, and Charades under the TimeLens framework) and extreme long-video understanding (LongVideoBench, Video-MME-v2).

## Contents

## 1 Introduction 4

## 2 Model Architecture 4

2.1 Native-Resolution Vision Encoder 5  
2.2 Unified Visual Encoding 5  
2.3 DSA for Long-Context Multimodal Modeling . 6  
2.3.1 MQA-Style Lightning Indexer 6  
2.3.2 GQA Sparse Aggregation 6  
2.3.3 Dense Warm-up and Sparse Adaptation . 6

## 3 Pre-Training 7

3.1 Stage 0 — Projector Initialization 7  
3.2 Stage 1 — General Multimodal Pre-training  
3.3 Stage 2 — Multi-Task Capability Injection 8  
3.3.1 Image-Text Data 8  
3.3.2 Pure-Text Data 9  
3.4 Stage 3 — Long-Context Extension 9  
3.5 Data Cleaning and Production Pipeline 9  
3.6 Video Pre-Training Curriculum 10

## 4 Post-Training 10

4.1 Supervised Fine-Tuning and Synthetic CoT 10  
4.1.1 Multimodal Instruction Mixture 11  
4.1.2 Synthetic CoT 11  
4.2 Reinforcement Learning 11  
4.2.1 Synthetic-Data RL 11  
4.2.2 General RL 12  
4.2.3 Specialized RL 13  
4.2.4 Video RL 14  
4.2.5 Agentic RL . 14  
4.2.6 Cross-Modal Multi-Teacher On-Policy Distillation . 15

## 5 Efficient Training and Inference Infrastructure 15

5.1 Pre-Training Systems 15  
5.1.1 ViT–LM Heterogeneous Parallelism and Load Balancing 15  
5.1.2 DSA Optimization for Variable-Length Sequences 16

5.2 Post-Training Systems 16

5.2.1 DSA Adaptation for RL 16

5.2.2 On-Policy Distillation System 16

5.3 Efficient Inference for GQA+DSA 17

## 6 Comprehensive Evaluation 17

6.1 Video Understanding . 17

6.1.1 Benchmarks 17  
6.1.2 Results 18

6.2 Agentic Capability Evaluation . 19

6.2.1 Code Agent Evaluation 19  
6.2.2 Tool-Use Evaluation 19

6.3 General Vision-Language Evaluation . 19

## 7 Conclusion and Future Work 20

A Case Study 25  
B Contribution (Alphabetical order) 31

## 1 Introduction

In recent years, Large Language Models (LLMs) are rapidly evolving, integrating deeper reasoning and extending into complex multimodal domains. Recent advances, such as OpenAI GPT-5.5 (OpenAI (2026)), Claude Opus 4.8 (Anthropic (2026)), Gemini 3.5 Flash (Google DeepMind (2026)), and Qwen3.7 (Qwen Team (2026b)), demonstrate substantial progress in multimodal reasoning, long-context understanding, and specialized tool execution. These models increasingly exhibit the ability to decompose complex problems and process dynamic, information-dense visual streams.

Building upon the strong foundation of our previous works, Keye-VL (Kwai Keye Team et al. (2025)) and Keye-VL-1.5 (Yang et al. (2025b)), which established robust vision-language alignment and state-of-the-art short-video comprehension, we introduce Kwai Keye-VL-2.0-30B-A3B. As we push the model’s frontier from short-form visual perception to long-horizon agentic reasoning, two critical roadblocks naturally emerge. First is the infrastructural bottleneck: scaling to extreme video contexts incurs prohibitive computational and memory costs. Second is the algorithmic dilemma: integrating complex, heterogeneous agent tasks often induces catastrophic forgetting of foundational reasoning capabilities. In this report, we detail how Keye-VL-2.0 elegantly resolves these dual challenges by introducing paradigm-shifting improvements in architecture and alignment, focusing on the following two key aspects.

Extreme Context Scaling via Multimodal DSA. A key insight from extending multimodal models to 256K contexts is that standard dense attention mechanisms inevitably lead to catastrophic KV cache expansion and computational walls, forcing models to sacrifice temporal continuousness through aggressive frame subsampling. To overcome this latency and scalability limit, Keye-VL-2.0 introduces a 30-billion-parameter Mixture-of-Experts (MoE)(Shazeer et al. (2017)) foundation with only 3 billion active parameters, ensuring exceptional deployment efficiency. Crucially, we pioneer the application of Multimodal DeepSeek Sparse Attention (DSA)(DeepSeek-AI et al. (2025)) in the visual domain. By compressing and sparsifying video feature aggregation, DSA effectively constrains the linear growth of the KV cache. This design allows Keye-VL-2.0 to process 256K extreme long-video contexts losslessly, transforming video understanding from frame-limited perception to global-context reasoning while maintaining high inference efficiency.

Resolving Modality Conflict via Cross-Modal MOPD. Existing models often suffer from the "Multimodal Alignment Dilemma": directly injecting complex video understanding and tool-use capabilities often triggers catastrophic forgetting, degrading the model’s foundational STEM, mathematical, and linguistic reasoning abilities. To circumvent the ambiguity and instability of end-to-end co-optimization, we introduce Cross-Modal Multi-Teacher On-Policy Distillation (MOPD)(MiMo Team et al. (2026)). Through an innovative dynamic routing mechanism, MOPD leverages specialized teacher models to provide dense token-level feedback on student-generated trajectories across varying modalities and tasks during post-training. By supervising on-policy rollouts, MOPD effectively isolates task-specific expertise, which is subsequently distilled and seamlessly merged back into the unified MoE backbone. This bidirectional alignment ensures that Keye-VL-2.0 achieves substantial leaps in native agent capabilities (e.g., Code, Tool Use, Web Search) while robustly preserving its general-purpose reasoning baselines. During posttraining, we further apply Context-RL(Lu et al. (2026a)) and Video-RL, equipped with bucket advantage scaling, to stabilize long-sequence decision trees and systematically reduce visual hallucinations.

As illustrated in Figure 1, Keye-VL-2.0-30B-A3B demonstrates highly competitive performance against both open-source and closed-source state-of-the-art models. It achieves top-tier results in fine-grained temporal localization across TimeLens benchmarks (Zhang et al. (2025)) (ActivityNet, QVHighlights, Charades), substantially outperforming leading models such as Gemini-3-Flash (Google DeepMind (2025)) on several temporal grounding settings. Furthermore, it exhibits a non-linear capability scale-up on extreme long-context evaluations, including Video-MME-v2 (Fu et al. (2026)) and LongVideoBench (Wu et al. (2024b)), proving its robustness in processing extensive temporal information. By resolving the dual challenges of context scaling and multi-task capability conflict, we offer a powerful, efficient, and versatile multimodal foundation for the open-source community, enabling researchers and developers to explore, refine, and deploy scalable multimodal applications.

## 2 Model Architecture

Following the standard multimodal large language model (MLLM) paradigm, the model consists of four core components:

• Vision Encoder (ViT): inherited from Keye-VL-1.5-8B (Kwai Keye Team et al. (2025)) to extract visual features from images and video frames.

• Language Decoder (LLM): built on Qwen3-30B-A3B-Thinking-2507 (Yang et al. (2025a)), provid ing strong general knowledge, instruction following, and reasoning capabilities.  
• MLP Projector: randomly initialized and trained in Stage 0 (Section 3.1) to align visual features with the LLM representation space.  
• Sparse Attention Module: a GQA-compatible DSA design that combines global MQA-based indexing with grouped GQA aggregation, enabling efficient long-context multimodal modeling.

On top of this backbone, we introduce three architectural designs for high-resolution and long-context multimodal understanding: a native-resolution vision encoder, a unified visual encoding strategy for images and videos, and DSA-based sparse attention for 256K multimodal contexts.

## 2.1 Native-Resolution Vision Encoder

The visual encoders of MLLMs have gradually shifted from reusing fixed-resolution backbones to nativeresolution modeling. Conventional fixed-resolution ViTs are usually pre-trained for contrastive image-text matching, and their coarse-grained representations are often insufficient for detail-sensitive downstream tasks such as OCR, fine-grained recognition, document understanding, and video understanding.

Two main approaches have been explored for high-resolution inputs. Dynamic tiling methods, such as InternVL3 (Zhu et al. (2025)) and MiniCPM-V (Yao et al. (2024)), split large images into smaller crops before feeding them into fixed-resolution encoders. While effective, tiling may disrupt global structure and introduce redundant computation. Native-resolution methods, such as NaViT (Dehghani et al. (2023)), Qwen3.5 (Qwen Team (2026a)), K2.5 (Team et al. (2026)), and MiMO (MiMo Team et al. (2026)), instead preserve the original image size and aspect ratio. Following this direction, Keye-VL-2.0-30B-A3B encodes images and videos at their native resolutions, avoiding unnecessary cropping or tiling and preserving both global structure and local details.

This design is especially important for document, OCR, chart, and video scenarios, where a small local region may determine the final answer. By keeping the original aspect ratio throughout the visual pipeline, the encoder can preserve layout relations, object geometry, and fine-grained text signals that are easily distorted by fixed-size resizing.

Adaptive Position Encoding. The visual encoder inherits the ViT backbone from Keye-VL-1.5, which is based on SigLIP-400M-384-14 (Zhai et al. (2023)). To support variable resolutions while retaining pre-training benefits, we interpolate the fixed absolute learnable position embeddings, allowing them to scale with the input size.

2D RoPE. On top of adaptive absolute position encodings, we introduce 2D Rotary Position Embedding (2D RoPE). This improves spatial modeling and extrapolation across visual dimensions, especially for extremely high-resolution images.

Sequence Packing. We combine adaptive position encodings and 2D RoPE with NaViT’s Patch n’ Pack mechanism and FlashAttention. Samples with different sizes and aspect ratios can therefore be packed into a single batch without padding waste, improving training throughput under variable resolutions.

Distribution-Aligned ViT Pre-training. During ViT pre-training, we optimize the native-resolution architecture with the SigLIP loss and align it with the SigLIP-400M-384-14 text tower. To reduce the supervision-granularity gap between contrastive pre-training and downstream MLLM tasks, we train the visual encoder on the same data distribution as the downstream MLLM. The pre-training corpus contains 500B tokens from large-scale open-source datasets, including DataComp (Gadre et al. (2023)), LAION (Schuhmann et al. (2022)), CC12M (Changpinyo et al. (2021)), PD12M (Meyer et al. (2024)), and COCO (Lin et al. (2014)), together with high-quality internal data.

Overall, these upgrades allow the visual encoder to inherit the strong representation quality of SigLIP while gaining the resolution flexibility required by downstream multimodal reasoning tasks.

## 2.2 Unified Visual Encoding

To provide the language decoder with detailed visual signals, we use a unified dynamic-resolution encoding strategy for both images and videos.

• Dynamic-Resolution Image Encoding. Static images are encoded directly by the dynamicresolution ViT. The number of visual tokens is allocated according to the original pixel size, avoiding lossy resizing or cropping.  
• Dynamic-Resolution Video Encoding. For video inputs with varying frame rates, resolutions, and durations, each sampled frame is treated as an independent high-resolution image and encoded by the same visual encoder. To preserve temporal information, we prepend a naturallanguage timestamp to the visual tokens of each frame during preprocessing. This explicit timestamp injection helps the LLM perceive temporal order, causality, and absolute timing.

Compared with designing a separate video-specific encoder, this frame-as-image formulation keeps the visual pathway simple and unified. The timestamp text provides temporal anchors in the LLM’s native language space, making temporal localization and cross-frame reasoning easier to learn from instruction and RL data.

• Adaptive Video Pixel Budget. To balance information density and computation, we allocate a base pixel budget by dividing the global budget evenly across videos. In adaptive mode, the budget of each video is scaled according to duration, with thresholds of 256 s, 512 s, 1024 s, and 2048 s corresponding to scaling factors of 0.125, 0.25, 0.5, and 1.0. Videos longer than 2048 s use the full base budget. This compresses short, redundant videos more aggressively while allowing longer videos to retain more visual evidence, keeping the total token cost controllable.

## 2.3 DSA for Long-Context Multimodal Modeling

Traditional full attention has quadratic complexity with respect to sequence length, making 256K multimodal contexts difficult to support. To address this bottleneck, we integrate DeepSeek Sparse Attention (DSA) (DeepSeek-AI et al. (2025)) into the decoder attention pathway. Unlike most existing DSA systems that are adapted from MLA, Keye-VL-2.0-30B-A3B integrates DSA with a GQA-based MLLM backbone, providing a practical route for long-context extension in GQA models.

## 2.3.1 MQA-Style Lightning Indexer

DSA contains a Lightning Indexer and a fine-grained token selection mechanism. For throughput, the indexer follows the MQA key-sharing design and computes a global index score $I _ { t , s }$ between the current query token $h _ { t }$ and each preceding token $\tilde { h _ { s } } \mathrm { . }$

$$
I _ {t, s} = \sum_ {j = 1} ^ {H ^ {I}} w _ {t, j} ^ {I} \cdot \operatorname{ReLU} (q _ {t, j} ^ {I} \cdot k _ {s} ^ {I}). \tag {1}
$$

Here, $H ^ { I }$ is the number of indexer heads, $q _ { t , j } ^ { I }$ and $w _ { t , j } ^ { I }$ are derived from $h _ { t }$ , and $k _ { s } ^ { I }$ is the shared key derived from $h _ { s } .$ After obtaining the global scores, the Top-k tokens form the sparse index set:

$$
\Omega_ {t} = \{s \mid I _ {t, s} \in \operatorname{Top} - k (I _ {t,:}) \}. \tag {2}
$$

By sharing one key head across all query heads, the indexer substantially reduces both computation and memory traffic. Together with FP8 implementation and the ReLU-based scoring function, the Lightning Indexer remains efficient even when the multimodal sequence contains hundreds of thousands of tokens.

## 2.3.2 GQA Sparse Aggregation

In the GQA backbone, query heads are divided into G groups, and heads within each group share a group-specific KV head. We apply the same sparse index set $\Omega _ { t }$ to all groups. For the g-th group, the sparse attention output is:

$$
u _ {t, g} = \operatorname{Attn} (h _ {t, g}, \{c _ {s, g} \mid s \in \Omega_ {t} \}). \tag {3}
$$

The outputs of all groups are concatenated to form the final attention representation. We set k = 2048, reducing the core attention complexity from $O ( L ^ { 2 } )$ to O(Lk), where L is the sequence length and $k \ll L$

This global MQA-style indexing + GQA aggregation design preserves the representation structure of the GQA backbone while avoiding dense attention over the full context. As a result, the model can perform long-range spatiotemporal and semantic aggregation with much lower memory and compute cost.

## 2.3.3 Dense Warm-up and Sparse Adaptation

We train DSA with a two-stage strategy. In the dense warm-up phase, the main model keeps dense GQA while most parameters are frozen. The goal is to initialize the indexer so that its global distribution can cover the attention distributions of all GQA groups. For query token t, we aggregate dense attention scores within each group, normalize them over visual and text tokens, and obtain the target distribution $p _ { t , : , g }$ . The warm-up loss is:

$$
\mathcal {L} _ {\text { warmup }} ^ {I} = \sum_ {t} \sum_ {g = 1} ^ {G} \mathbb {D} _ {K L} (p _ {t,: g} \| \operatorname{Softmax} (I _ {t,:})). \tag {4}
$$

This stage uses approximately 2B multimodal tokens.

In the sparse adaptation phase, all parameters are unfrozen and training switches to sparse mode. The indexer continues to align with the main model’s attention distribution, but the KL loss is computed only over the selected Top-k token set

$$
S _ {t} = \{s \mid I _ {t, s} \in \operatorname{Top} - k (I _ {t,:}) \}. \tag {5}
$$

The target distribution is truncated and renormalized over $S _ { t } \colon$ :

$$
\mathcal {L} _ {\text { sparse }} ^ {I} = \sum_ {t} \sum_ {g = 1} ^ {G} \mathbb {D} _ {K L} (p _ {t, S _ {t}, g} \| \operatorname{Softmax} (I _ {t, S _ {t}})). \tag {6}
$$

The main model is optimized with the standard next-token prediction loss. To reduce gradient interference, the indexer input is detached from the computation graph. The final objective is:

$$
\mathcal {L} _ {\text { total }} = \mathcal {L} _ {\mathrm{NTP}} + \lambda \mathcal {L} _ {\text { sparse }} ^ {I}. \tag {7}
$$

This two-stage training strategy avoids forcing the sparse indexer to learn from scratch under sparse supervision. Dense warm-up first aligns the indexer with the behavior of the original attention model, and sparse adaptation then teaches the full model to rely on dynamically selected evidence. In longcontext evaluations, this design preserves the capabilities of the dense model while substantially reducing inference overhead.

## 3 Pre-Training

Keye-VL-2.0-30B-A3B is pre-trained through a four-stage curriculum. Stage 0 trains only the Projector to initialize visual-language mapping. Stage 1 performs full-parameter multimodal pre-training at 32K context length. Stage 2 extends the context to 64K and injects task-oriented capabilities such as OCR, VQA, STEM, GUI, grounding, counting, coding, tool use, and search. Stage 3 further extends the context to 256K, focusing on long videos, multi-page documents, multi-document inputs, and long-range agent trajectories.

## 3.1 Stage 0 — Projector Initialization

Stage 0 establishes the initial connection between the visual encoder and the language model. The ViT and LLM are frozen, and only the Projector is trained to map visual features from the Keye-VL-1.5 ViT into the LLM representation space. This stage uses image-text caption data for direct semantic alignment and image-text interleaved data to expose the Projector to real document-like contexts where multiple images and text appear alternately.

Because only the Projector is updated, Stage 0 provides a low-risk initialization step before full-parameter multimodal training. It allows the LLM to receive visual features in a compatible representation space without disturbing the already learned capabilities of either the ViT or the language backbone.

## 3.2 Stage 1 — General Multimodal Pre-training

Stage 1 trains all parameters with a maximum sequence length of 32K on approximately 1T tokens. The goal is to establish stable vision-language alignment, image perception, video understanding, OCR recognition, and general language capability.

The training data consist of image-text captions, interleaved image-text data, interleaved video data, pure-text QA, and OCR data. For video learning, each video is split into 15-second segments, each paired with a caption and organized as an interleaved multimodal sequence. This enables the model to understand temporal continuity, scene transitions, and event evolution.

To improve caption quality from large-scale open-source corpora such as LAION, DataComp, COYO, and CC12M (Schuhmann et al. (2022); Gadre et al. (2023); Byeon et al. (2022); Changpinyo et al. (2021)), we apply two strategies: Recaption regenerates captions directly from the images using an expert captioning model to ensure high quality, while Remake builds on the original captions, correcting grammatical and expression errors without altering their semantics. OCR data cover diverse types such as stylized text, handwriting, LaTeX formulas, code, charts, and documents, giving the model an early foundation in text recognition and fundamental perception capabilities.

![](images/0888e8fa9fa03dae519b0e991b3342996ea9839bc82ee899bb322c99a495b7f1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["STAGE 0 Projector Initialization"] --> B["Build the visual-language interface while keeping the foundation towers stable."]
  B --> C["PARAMETER STATUS: ViT Projector LLM SEQ LENGTH 8K DATA SCALE 4B ."]
  C --> D["DATA TYPES: Caption Interleaved Image"]
    
  E["STAGE 1 Image-Text Alignment"] --> F["Activate all parameters for large-scale image, video, OCR, and text alignment."]
  F --> G["PARAMETER STATUS: ViT Projector LLM SEQ LENGTH 32K DATA SCALE 1T ."]
  G --> H["DATA TYPES: Caption OCR Video Mix Image Mix Pure-Text QA"]
    
  I["STAGE 2 Multi-Task Injection"] --> J["Inject structured tasks with verifiable signals and agent-oriented behavior."]
  J --> K["PARAMETER STATUS: ViT Projector LLM SEQ LENGTH 64K DATA SCALE 2T ."]
  K --> L["DATA TYPES: STEM GUI Grounding Counting OCR E-Commerce Coding Tool-use/Search"]
    
  M["STAGE 3 Long-Context Extension"] --> N["Extend to 256K for long video, long documents, code, and agent trajectories."]
  N --> O["PARAMETER STATUS: ViT Projector LLM SEQ LENGTH 256K DATA SCALE 500B ."]
  O --> P["DATA TYPES: Long Video Long Docs Multi-doc Long Multi-image Long Code Long Agent Traj."]
```
</details>

Figure 2: The Keye-VL-2.0-30B-A3B pre-training pipeline, following a four-stage curriculum from projector initialization to 256K long-context multimodal training.

This diversity of data types—spanning image caption, interleaved image-text, interleaved video-text, pure-text, and OCR data—is designed to prevent the model from over-specializing in any single input format. Specifically, image caption data provide dense semantic alignment between an image and its description; interleaved image-text data expose the model to the organization of images and text within webpages and documents; interleaved video-text data introduce temporal and cross-modal structure; and OCR data establish sensitivity to visual text and structured page layouts from the outset of multimodal training.

## 3.3 Stage 2 — Multi-Task Capability Injection

Stage 2 continues full-parameter training, extends the context length to 64K, and trains on approximately 2T tokens. Compared with Stage 1, this stage emphasizes task-oriented and verifiable supervision.

## 3.3.1 Image-Text Data

Advanced OCR. We further expand the OCR data to receipts and diverse chart types, drawing on both real samples and synthetic samples generated from XML or structured templates. Augmentations such as blur, lighting variation, wrinkles, handwriting variation, and geometric distortion are applied to improve robustness for field extraction, table reading, chart understanding.

Real samples consist of images captured in real-world settings, carrying authentic noise and naturally occurring business layouts, whereas synthetic samples offer precise field-level and structure-level annotations. Combining the two improves both robustness and supervision precision.

Math & STEM. We introduce visual problem-solving examples spanning mathematics, physics, and chemistry, with LLM-based verification to ensure the quality of questions and answers and to remove low-quality samples. This helps the model progress from recognizing formulas and diagrams to reasoning over them. The covered scenarios include geometry diagrams, function plots, experimental apparatus, formulas, and scientific charts. The goal is to enhance the model’s generalization across diverse visual scenarios.

Caption Refinement. High-quality caption data is retained to preserve vision-language alignment. An 8B caption expert model trained with VCap (Lu et al. (2026b)) generates detailed descriptions emphasizing attributes, spatial relations, subject binding, and long-description consistency—yielding captions that are more comprehensive, ine-grained, and of higher quality than those used in Stage 1, surpassing even those produced by larger models.

GUI. GUI tasks involve screen captures, control metadata, and interaction semantics, covering element localization, control recognition, page understanding, operation description, and GUI-QA. GUI inputs differ substantially from natural images, featuring dense small text, icons, regular layouts, and frequent state changes. Training on dedicated GUI tasks therefore lays the foundation for subsequent vision-based clicking, navigation, and task execution.

Grounding & Counting. For localization and counting, we synthesize instance-pasting data. Candidate objects from COCO (Lin et al. (2014)) and OpenImages (Kuznetsova et al. (2020)) are verified by an MLLM and pasted onto backgrounds with controllable quantities and positions, producing precise bounding boxes and count annotations.

General QA, E-Commerce, and Chinese Expansion. We supplement general VQA data for object recognition, attributes, scenes, and relations, and introduce Kuaishou e-commerce data for product understanding. To improve Chinese coverage, high-quality English task data are translated into Chinese, regenerated under original answer constraints, and filtered for consistency.

## 3.3.2 Pure-Text Data

Stage 2 retains pure-text training to preserve the LLM’s language, reasoning, coding, tool-use, and instruction-following capabilities. The mixture spans pure-text math and STEM reasoning, code corpora for bug fixing and competitive programming, Hermes-style tool-use trajectories (Teknium et al. (2024)), and search/RAG examples covering single-document QA, multi-document summarization, evidence localization, and multi-hop reasoning over Wiki-based knowledge graphs.

The pure-text component mitigates the degradation of the base model’s language-side capabilities during multimodal fine-tuning, while supplying transferable priors for reasoning formats, code generation, tool invocation, and evidence-based answer synthesis.

## 3.4 Stage 3 — Long-Context Extension

Stage 3 extends the maximum sequence length to 256K while continuing full-parameter training. Longcontext and short-context samples are mixed at a ratio of 1:1 to improve ultra-long sequence modeling while preserving performance on conventional inputs.

The data cover long videos, long documents, multi-document inputs, long multi-image conversations, long code contexts, and long-range agent trajectories. The objective is not only to enlarge the context window, but also to improve retrieval, aggregation, and cross-position reasoning over ultra-long multimodal inputs.

For long videos, the model must track key events and integrate evidence across many frame segments. For long documents and multi-document inputs, it must process multi-page OCR, tables, layouts, and long textual evidence. For agent trajectories, it must maintain task state across many tool calls and interaction turns.

## 3.5 Data Cleaning and Production Pipeline

We build a unified data cleaning, deduplication, and production pipeline. Cleaning removes low-quality text, invalid formats, unsafe content, duplicated samples, and unreliable image-text or video-text pairs. Deduplication uses a joint Hash + CLIP strategy: Hash identifies exact or near duplicates, while CLIP similarity detects semantically similar cross-modal duplicates.

For production efficiency, the pipeline supports caption generation, OCR construction, QA generation, translation, quality evaluation, one-click deployment, task monitoring, and checkpoint-based resumption. A dual-queue asynchronous mechanism decouples CPU-side preprocessing from GPU-side inference, improving production throughput by 3–5× in practice.

Table 1: Video training setup across pre-training stages.

<table><tr><td>Stage</td><td>Max Video Duration</td><td>Video Tokens</td></tr><tr><td>1</td><td>15s</td><td>24K</td></tr><tr><td>2</td><td>15min</td><td>64K</td></tr><tr><td>3</td><td>2h</td><td>180K</td></tr></table>

Video (04:06)  
![](images/69c79ba1a6392dd37d9511739898e50f7913b5aa9c6bcdc2f26ef41a2172cdcb.jpg)

<details>
<summary>text_image</summary>

Film strip displaying a multi-panel collage of remote weather and navigation scenes with Chinese text labels and icons.
</details>

![](images/03cc3eed576b79c27da67a9bbcc6c51c03b8d83c3e32a84eaa70522f2d8ca781.jpg)

<details>
<summary>text_image</summary>

00:00
START
Scene 1
00:45
Scene 2
01:40
Scene 3
03:10
Scene 4
04:06
END
</details>

![](images/882490e7d85eed17d0bc4ded0448d5a5ccfd61f8743570a8f723f8baec2cdc6c.jpg)  
A "NEW THIS MORNING" segment shows an ice anchor desk in Indianapolis.Night aerial of Tulsa.The anchor introduces meteorologists George Flickinger and Taft Price in the studio.

![](images/20d0424706121b8ce788452e05cce88a5c19a09711ff0844e37ddfedf8f9ff01.jpg)  
Live shot of Jenks bus lot covered in snow."What We're Tracking" graphic lists dry Tuesday and potential snow in Dallas-Fort Worth on Wednesday.Today's hourly weather over a daytime driving view.

![](images/362e323ac554be5e570ac0fab5476de737350c462dda40d5fd63c92e7f47d9f3.jpg)  
Scene 3 [01:40-03:09]  
Meteorologist explains radar/satelite over Oklahoma and the Western U.S., current temperatures (mostly low 20s), Midtown conditions from River Spirit Casino,and the Future Track of an incoming storm south of Oklahoma.

![](images/e44f155b4c3aa88e396e41a55de80b813123b662cb8071499ffa3655d2c8a7f9.jpg)  
Scene 4 [03:10-04:06]  
Today's forecast for Tulsa (high 44F,clearing skies)and tonight's forecast (low 22F).A7-day forecast shows sunshine early,a 5O% chance of rain on Sunday,and fluctuating temperatures through the week.

![](images/7a791ad9008288b818eb204ba302444ee89b83eff655605bbdbafb4a8da0e94c.jpg)  
Video Overview

This video is a winter weather news segment for Oklahoma.It begins with severe cold reports from Indianapolis and Tulsa, then shows current snow impacts and school bus lotconditions.Meteorologists analyze radar and temperature data, followed bydetailed forecasts for today,tonight,and the coming seven days.

Figure 3: An example of scene-wise dense caption. Each video is decomposed into scenes annotated with timestamps, dense captions, and a global overview.

## 3.6 Video Pre-Training Curriculum

To scale from short-video understanding to high-resolution long-video reasoning, we adopt a multi-stage video curriculum, summarized in Table 1.

Stage 1 uses short 15-second clips for video-language alignment. Stage 2 increases duration and resolution, and introduces temporal video grounding (TVG) data. Stage 3 extends the maximum duration to 2 hours, requiring the model to identify sparse but critical moments and aggregate long-range evidence. In mid-training, we keep the Stage 3 setup while adding richer tasks such as video captioning, temporal QA, entity recognition, scene understanding, causal reasoning, event ordering, and video counting.

Scene-Wise Dense Caption. We reformulate dense video captioning as structured scene-wise descriptions with start and end timestamps. As shown in Figure 3, this formulation enhances scene-boundary perception and temporal alignment between descriptions and intervals. An expert model is trained to improve annotation efficiency.

Diverse TVG Data. Inspired by ETBench (Liu et al. (2024a)), we construct TVG data covering Referred Action Recognition, Video Highlight Detection, Extractive Video Summarization, and Temporal Event Matching. These tasks provide complementary supervision for temporal perception and reasoning.

## 4 Post-Training

## 4.1 Supervised Fine-Tuning and Synthetic CoT

After multimodal pre-training, we conduct SFT to convert visual perception and cross-modal alignment into stable instruction-following behavior. This stage adapts multimodal capabilities to conversational instructions, activates perception, reasoning, long-context, and agentic abilities, and mitigates language degradation during multimodal fine-tuning.

The SFT corpus contains approximately 500B tokens and is organized around modality balance, capability coverage, and long-range modeling. It includes Text NLP, Video, Perception, Reasoning, Agent, and Long-context data. About 40% of the corpus is text-only data, which anchors instruction following, knowledge QA, and textual reasoning.

## 4.1.1 Multimodal Instruction Mixture

To improve comprehensive instruction-following capabilities in complex multimodal scenarios, we construct a multimodal instruction mixture that covers video understanding, visual perception, crossmodal reasoning, agent-oriented tasks, and long-context modeling. The goal of this mixture is not to simply increase the proportion of visual data, but to encourage complementary capability development across perception, localization, reasoning, and task execution.

Video data are mainly used to strengthen temporal perception and evidence localization. Some samples are formulated as multiple-choice question answering tasks with clue intervals. The model is required to verify candidate temporal segments during the <think> stage, and to output both the final answer and the supporting intervals in the format [[mm, mm], ...]. This design encourages the model to locate key evidence from continuous video content, rather than relying only on global visual impressions.

Perception data cover OCR, document understanding, chart understanding, general visual question answering, captioning, grounding, counting, and image identification. These tasks improve the model’s ability to extract, recognize, and structure fine-grained visual information. Reasoning data include K12-level exercises, STEM problems, and spatial reasoning tasks, which train the model to perform multi-step inference under both visual and textual constraints.

Agent data consist of code reasoning and tool-use trajectories, enabling the model to learn task decomposition, execution, and feedback integration. Long-context data target long documents, long videos, multi-image inputs, and long-range question answering, with the goal of improving information retention, retrieval, and cross-segment association over extended contexts.

Overall, this instruction mixture is designed to promote capability collaboration across different data sources. Video data enhance continuous scene understanding and temporal evidence localization; perception data improve fine-grained information extraction; reasoning data strengthen cross-modal multi-step inference; agent data support task decomposition and tool interaction; and long-context data improve long-range information modeling. Meanwhile, text-only instruction data are retained to preserve general instruction following and language reasoning capabilities, reducing the risk of degrading pure language performance during multimodal training.

## 4.1.2 Synthetic CoT

Most multimodal instruction data provide only final answers, offering limited supervision for explicit reasoning and evidence aggregation. We therefore construct Synthetic CoT data from high-quality QA pairs. A strong teacher model generates reasoning traces, which are then filtered through query-level, response-level, and process-level quality checks. For mathematical tasks, a Doubt2Clean second-pass review further cleans doubtful CoT samples across 27 datasets.

The final mixture combines long <think> reasoning data for STEM, counting, video reasoning, and complex image-text QA with direct-answer data for general VQA, caption, OCR, and grounding. This balances explicit reasoning with concise answering and avoids unnecessary verbosity on simple perception tasks.

For continuous perception tasks, such as counting and video understanding, Synthetic CoT provides object-by-object or interval-by-interval verification chains. For example, the model may first verify local evidence in separate regions or candidate video intervals, then aggregate the evidence before producing the final answer. For STEM tasks, the reasoning trace emphasizes planning, symbolic derivation, and result verification. This gives the model learnable supervision for both ordered observation and logical reasoning.

Beyond these offline reasoning traces, we further optimize the model’s reasoning chains through strongto-weak on-policy distillation, in which a stronger teacher supervises the student on its own on-policy rollouts.

## 4.2 Reinforcement Learning

## 4.2.1 Synthetic-Data RL

Fine-grained multimodal reasoning requires accurate visual perception, difference localization, and structured output generation. However, natural multimodal corpora rarely provide fine-grained localization signals, structured annotations, and automatically verifiable supervision simultaneously. To address this limitation, we introduce a synthetic-data RL framework based on procedurally generated difference-recognition tasks. Given two images that differ by controlled edits, the model is required to identify the changes and report them in a predefined structured format. Since the edit operations are fully known during data generation, rewards can be computed through rule-based verification without relying on an additional learned reward model.

Each sample contains an image pair $\left( I _ { A } , I _ { B } \right)$ , where $I _ { B }$ is obtained from $I _ { A }$ through a set of controlled edits $\mathcal { E } = \bar { e } _ { 1 } , \ldots , e _ { k }$ . We instantiate this framework in two forms. For localization-oriented perception tasks, the model predicts normalized bounding boxes for changed regions. For structured reasoning tasks, the model outputs a domain-specific operation-set DSL corresponding to the underlying edits. In our implementation, the structured tasks cover geometry, coordinate-aware geometry, chemical formulas, and physical circuit diagrams. The synthetic setup includes both positive samples with controlled edits and negative samples with no edits, enabling the model to distinguish true changes from unchanged inputs and suppress unsupported predictions. To prevent the model from relying on trivial pixel-level differencing, we introduce difference-irrelevant re-rendering perturbations that preserve the underlying edit set, including color jitter, layout perturbation, slot shuffling, semantic no-op changes, and viewpoint variation.

For perception tasks, predicted boxes $\hat { B }$ are matched to ground truth B using Hungarian matching based on IoU. The reward is:

$$
R _ {\text { perc }} = \lambda_ {F} F _ {\text { soft }} + \lambda_ {I} \frac {\sum_ {m \in \mathcal {M}} u _ {m}}{\max (| \hat {\mathcal {B}} | , | \mathcal {B} |)} - \lambda_ {D} N _ {\text { dup }}, \tag {8}
$$

where M denotes the matched set, $u _ { m }$ is the IoU of the m-th matched pair, $F _ { \mathrm { s o f t } }$ is a soft matchingbased F-score, and $N _ { \mathrm { d u p } }$ penalizes duplicate predictions. For negative samples, empty predictions are rewarded, while unsupported predictions are penalized. For structured domains, rewards are computed by canonicalizing the predicted and ground-truth operation sets and then performing rulebased matching. For coordinate-aware tasks, we further use soft distance-based similarity to tolerate small numerical deviations. This reward design is lightweight, scalable, and independent of learned reward models, while encouraging visual grounding, structural correctness, anti-hallucination behavior, and disciplined structured output generation.

## 4.2.2 General RL

After SFT and distillation, General RL improves reasoning ability, answer reliability, and robustness across multimodal and text-only scenarios. Unlike preference-oriented Alignment RL, General RL focuses on tasks with verifiable ground-truth answers, including general VQA, STEM, chart understanding, mathematical reasoning, logical reasoning, and text-only QA.

Data Construction. We collect multimodal and text-only QA data from open-source and in-house sources, including FineVision (Wiedmann et al. (2025)), MMK12 and MM-Eureka (Meng et al. (2025)), Thyme (Thyme Team (2025)), mini O3 (Lai et al. (2025)), Open-R1 (Hugging Face Open-R1 Team (2025)), self-cognition data, and business-oriented data (Lu et al. (2025)). The data pipeline includes benchmark decontamination, cross-validation with strong models and judge models, and accuracy-based filtering to remove samples already solved by the initial policy.

Benchmark decontamination removes samples that are visually or semantically too close to public evaluation instances. Cross-validation samples multiple candidate solutions and asks a judge model to select reliable positives, which are then used as reference solutions for ContextRL Reward. Accuracybased filtering focuses RL on samples that are neither trivial nor too noisy, improving data efficiency.

Reward System. The reward system, based on Qwen2.5-VL-72B-Instruct (Bai et al. (2025)), evaluates format validity, outcome correctness, process correctness, and consistency with verified reference solutions. Additional task-specific rewards are used for self-cognition, business tasks, and structured reasoning.

Format Reward ensures that reasoning and final-answer fields can be parsed. Outcome Reward checks whether the final answer matches the ground truth across heterogeneous answer formats, including options, phrases, numerical answers, and complete solutions. Process Reward penalizes factual errors, invalid deductions, unsupported assumptions, and inconsistent intermediate steps. ContextRL Reward (Lu et al. (2026a)) compares the sampled response with verified reference solutions to effectively reduce false positives caused by correct answers reached through flawed reasoning.

Training Algorithm. We use Group Sequence Policy Optimization (GSPO) (Zheng et al. (2025)). Given query x, the old policy $\pi _ { \theta _ { \mathrm { o l d } } }$ samples responses $\{ y _ { i } \} _ { i = 1 } ^ { G }$ . The objective is:

$$
\mathcal {J} _ {\mathrm{GSPO}} (\theta) = \mathbb {E} _ {x \sim \mathcal {D}, \left\{y _ {i} \right\} _ {i = 1} ^ {G} \sim \pi_ {\theta_ {\text {old}}} (\cdot | x)} \left[ \frac {1}{G} \sum_ {i = 1} ^ {G} \min \left(s _ {i} (\theta) \hat {A} _ {i}, \operatorname{clip} \left(s _ {i} (\theta), 1 - \epsilon , 1 + \epsilon\right) \hat {A} _ {i}\right) \right], \tag {9}
$$

where

$$
s _ {i} (\theta) = \exp \left(\frac {1}{| y _ {i} |} \sum_ {t = 1} ^ {| y _ {i} |} \log \frac {\pi_ {\theta} (y _ {i , t} \mid x , y _ {i , <   t})}{\pi_ {\theta_ {\text { old }}} (y _ {i , t} \mid x , y _ {i , <   t})}\right), \tag {10}
$$

and

$$
\hat {A} _ {i} = \frac {\mathcal {R} (x , y _ {i}) - \text { mean } \left(\{\mathcal {R} (x , y _ {j}) \} _ {j = 1} ^ {G}\right)}{\text { std } \left(\{\mathcal {R} (x , y _ {j}) \} _ {j = 1} ^ {G}\right) + \delta}. \tag {11}
$$

We further over-sample responses by enlarging the generation batch size and filter groups with zero advantage variance to improve data efficiency. This filtering keeps training focused on groups where different rollouts receive distinguishable rewards, so each update contains a stronger learning signal.

## 4.2.3 Specialized RL

Specialized RL strengthens image-text capabilities including grounding, spatial understanding, mathematical reasoning, counting, and OCR. All experts start from the same general RL checkpoint and use domain-specific data and rewards. Deterministic tasks use rule-based verifiable rewards: IoU and bipartite matching for grounding, symbolic equivalence for math, exact numeric matching for counting, and normalized text matching for OCR. Open-ended spatial reasoning uses a model judge with format constraints.

The purpose of this stage is not to train separate final models, but to obtain strong domain experts that can later contribute capability-specific supervision through distillation and capability consolidation.

Grounding Expert. For grounding tasks, the reward is designed to measure target-level localization quality while suppressing redundant predictions. We first normalize all boxes to the [0, 1000] coordinate range and remove duplicate predicted boxes. For multi-object grounding, predictions are matched to ground-truth boxes through Hungarian matching using IoU as the matching criterion, ensuring a one-to-one correspondence between predicted and target boxes.

Based on the matched pairs, the reward considers both the worst-case and average localization quality. The minimum IoU is used as a thresholding signal to ensure that all targets reach a basic localization quality before the sample enters the high-reward region, while the mean IoU provides continuous feedback for optimization. A duplicate-box penalty is further introduced to reduce redundant outputs and discourage box-spamming. This reward design therefore balances strict target coverage, smooth localization feedback, and output compactness.

Spatial Expert. The spatial expert targets spatial-relation understanding and embodied spatial reasoning. Unlike grounding or counting, many spatial tasks do not admit a simple deterministic criterion, since the answer may involve relative position, orientation, interaction, or scene-level commonsense. We therefore use a generative model judge to evaluate whether the response satisfies the required spatial relation and scene constraint. The judge assigns a discrete correctness score in {−1, 0, 1}, which is combined with a format reward to encourage both spatial correctness and legal answer structure.

Math Expert. The math expert focuses on mathematical and STEM reasoning. For problems with deterministic answers, we adopt a symbolic-equivalence reward: the model prediction and the reference answer are parsed into canonical forms, and a positive reward is assigned when they are mathematically equivalent. The reward is gated by the required <think> and <answer> structure, so that the model is encouraged to produce valid reasoning and final-answer formats. For open-ended solutions that are difficult to canonicalize, we fall back to a generative model judge. This design emphasizes answer correctness while avoiding overfitting to a specific reasoning trace.

Counting Expert. The counting expert is designed for visual counting tasks, where the target output is usually a discrete integer. Since the correctness criterion is unambiguous, we use an exact numeric-match reward under format gating. A positive reward is given only when the predicted number matches the ground-truth count. This provides a clean and deterministic learning signal, encouraging the model to align visual perception with precise numerical outputs.

OCR Expert. The OCR expert targets text recognition, document reading, and image-based text understanding. We use a normalized text-match reward, where predictions and references are compared after normalizing case, whitespace, and punctuation. This reward preserves deterministic verification while allowing minor surface-form variations. It encourages the model to improve both recognition accuracy and text completeness in OCR and document-style tasks.

## 4.2.4 Video RL

Video RL further optimizes temporal alignment, event tracking, and long-range information aggregation. Starting from the general RL checkpoint, we train on approximately 31K video samples with GSPO while freezing the visual encoder and vision-language projector.

The data cover temporal video grounding, temporal dense captioning (Lu et al. (2026b)), frame-level perception, video QA, temporal ordering, and event counting. TVG samples are selected from TimeIT (Ren et al. (2024)), mainly from DiDeMo (Hendricks et al. (2017)) and Charades-STA (Gao et al. (2017)), and are rewarded by temporal IoU. Temporal dense captioning uses an LLM-as-Judge reward over subject recognition, action description, scene information, OCR text, temporal order, hallucination, and coverage. FrameForge synthetic videos provide rule-verifiable supervision for timestamp localization, counting, before/after reasoning, and co-occurrence reasoning. This stage improves general video benchmark performance by approximately 1 percentage point.

These tasks provide complementary video supervision. TVG directly optimizes temporal boundary localization, dense captioning encourages full-video coverage and faithful content organization, and FrameForge supplies low-noise frame-level signals that are difficult to obtain from natural videos.

## 4.2.5 Agentic RL

Shared Training Protocol. Agentic RL extends post-training from single-response scoring to multi-step environment interaction. It covers code, tool-use, and search tasks whose supervision comes from executable outcomes or verifiable environment states rather than static response labels. Across these domains, we use environment-grounded rewards, trajectory-level validation, and filtering to remove invalid, unfinished, low-information, or weakly verified rollouts. Completed trajectory groups are optimized with the GSPO objective introduced above, using outcome-based rewards at the trajectory level. To improve rollout utilization under long and variable interaction horizons, we use a shared colocated partial-rollout mechanism: unfinished trajectories are cached and resumed in later rollout steps, while completed groups are immediately consumed for GSPO updates. This shared protocol keeps the optimization objective and rollout scheduling consistent, while allowing each task family to define its own environment interface and reward evidence.

Coding RL. For Coding RL, we use both Online Judge and Software Engineering environments. Online Judge tasks evaluate isolated program synthesis through compilation and hidden test execution, with rewards determined by pass rate and execution failures such as time-limit or memory-limit errors. Software Engineering tasks evaluate repository-level issue resolution in containerized environments, where the model must inspect logs, run tests, edit files, and submit patches. Rewards are based on testsuite outcomes, regression checks, trajectory filtering, and auxiliary judge verification of the submitted patch. For repository-level tasks, candidate edits may further pass through a verification-and-integration protocol in which multiple reviewer agents inspect proposed changes, execution traces, and test outcomes, and a separate integration agent consolidates the reviews before final patch acceptance. This setup strengthens algorithmic correctness, repository navigation, iterative repair, and regression-safe code modification.

Tool Use RL. For tool-use RL, the model interacts with multi-turn stateful environments covering more than 150 simulated API domains. Rewards are computed from valid tool invocation, argument consistency, database or environment post-conditions, and judge-based assessment of whether the final state satisfies the user intent. Unlike code execution rewards, this objective emphasizes correct state transitions, robust recovery from tool errors, and coordination with multi-turn user instructions. Randomized tool and parameter names reduce dependence on memorized API schemas and encourage general tool-calling behavior.

Search RL. For Search RL, we train on multi-turn search tasks where the model issues retrieval actions, reads returned content, and produces a final answer after one or more retrieval rounds. These trajectories vary with the amount of query refinement, result selection, and content checking required by the task. Rewards are outcome-oriented: final-answer correctness is the primary signal, and when available we also use lightweight verification signals for intermediate search results to reduce off-track retrieval and unsupported answer synthesis. The shared partial-rollout mechanism is useful in this setting because search trajectories have variable horizons; unfinished interactions can be resumed in later rollout steps, while completed trajectory groups are used for GSPO updates.

## 4.2.6 Cross-Modal Multi-Teacher On-Policy Distillation

Domain-specific post-training introduces heterogeneous capabilities from pure-text, image, video, and agent tasks. Directly mixing them may cause interference, such as overly short responses after reasoning RL or excessive tool-call formatting after agent training. To consolidate these capabilities, Keye-VL-2.0- 30B-A3B uses Cross-Modal Multi-Teacher On-Policy Distillation (MOPD).

MOPD maintains 13 RL-trained domain teachers. These teachers span the full capability spectrum, including safety, pure-text math, instruction following, code, visual STEM, OCR, grounding, counting, video, and tool use, among others. Each sample is routed to the teacher that best matches its modality and task type. Given prompt $x _ { i } ,$ the student first generates an on-policy response:

$$
y _ {i} = (y _ {i, 1}, \dots , y _ {i, T}) \sim \pi_ {\theta} (\cdot \mid x _ {i}). \tag {12}
$$

$s _ { i , t } = ( x _ { i } , y _ { i , < t } )$ $\pi _ { \mathsf { T } } ^ { r ( i ) }$ provides token-level feedback. We use Segmented Prompt-Response Re-tokenization (SPRR) to process prompts and responses separately, ensuring strict alignment between teacher log probabilities and student response tokens.

For stable feedback, we define the top-k overlap set:

$$
\Omega_ {i, t} = \operatorname{TopK} \left(\pi_ {\mathrm{T}} ^ {r (i)} (\cdot \mid s _ {i, t})\right) \cap \operatorname{TopK} \left(\pi_ {\theta} (\cdot \mid s _ {i, t})\right). \tag {13}
$$

When $\Omega _ { i , t }$ is non-empty, the raw advantage is:

$$
A _ {i, t} = \sum_ {v \in \Omega_ {i, t}} \bar {\pi} _ {\theta} (v \mid s _ {i, t}) \left[ \log \pi_ {\mathrm{T}} ^ {r (i)} (v \mid s _ {i, t}) - \log \pi_ {\theta} (v \mid s _ {i, t}) \right], \tag {14}
$$

where

$$
\bar {\pi} _ {\theta} (v \mid s _ {i, t}) = \frac {\pi_ {\theta} (v \mid s _ {i , t})}{\sum_ {u \in \Omega_ {i , t}} \pi_ {\theta} (u \mid s _ {i , t})}. \tag {15}
$$

If $\Omega _ { i , t } = \emptyset ,$ we set $A _ { i , t } = 0 .$ .

Compared with distilling over the full vocabulary, the overlap estimator focuses supervision on the local distribution region considered plausible by both teacher and student. This avoids unstable comparisons over very low-probability tokens while keeping the training on-policy, since the feedback is computed on states actually visited by the student.

The student is optimized with an advantage-weighted policy-gradient objective:

$$
\mathcal {L} _ {\mathrm{MOPD}} = - \mathbb {E} \left[ \frac {1}{| \mathcal {M} _ {i} |} \sum_ {t \in \mathcal {M} _ {i}} \widehat {A} _ {i, t} \log \pi_ {\theta} (y _ {i, t} \mid x _ {i}, y _ {i, <   t}) \right], \tag {16}
$$

where $\mathcal { M } _ { i }$ is the valid response-token mask. We further apply token-category-aware advantage scaling to down-weight formatting tokens and up-weight perception or reasoning tokens. For long-form generation, repetition collapse is localized at position τi and penalized only after the collapse point:

$$
\widehat {A} _ {i, t} = \widetilde {A} _ {i, t} - \lambda_ {\mathrm{rep}} \cdot \mathbb {I} [ t \geq \tau_ {i} ]. \tag {17}
$$

Together, dynamic teacher routing, SPRR alignment, top-k overlap estimation, token-category scaling, and localized repetition penalties allow MOPD to consolidate heterogeneous RL teachers without forcing all domains into a single response style.

## 5 Efficient Training and Inference Infrastructure

## 5.1 Pre-Training Systems

## 5.1.1 ViT–LM Heterogeneous Parallelism and Load Balancing

To remove I/O bottlenecks from video decoding and frame sampling, we introduce ExtraIO, a horizontally scalable I/O service decoupled from training through an asynchronous pipeline. We further co-design ViT–LM heterogeneous parallelism and two-level load balancing for long-video and variablelength workloads.

ViT–LM Heterogeneous Parallelism. The ViT and LM are co-located on the same GPU group, but each module adopts its own parallel sharding strategy. This avoids the imbalance caused by binding the ViT to LM PP0. A recompute-or-offload strategy reduces ViT activation memory to nearly zero, leaving more memory for long sequences.

Load Balancing. Because the ViT processes image/frame-level visual tokens while the LM processes sample-level multimodal sequences, both visual-token ratios and sequence lengths fluctuate. We balance load at the multimodal-token level and the LM-sample level, equalizing compute and memory across ViT DP and LM DP/PP. This improves end-to-end training throughput by approximately 20%.

This design is particularly useful for video-heavy batches, where a few long samples can otherwise create severe bubbles and leave a large fraction of devices underutilized.

## 5.1.2 DSA Optimization for Variable-Length Sequences

We optimize DSA with FlashInfer (Ye et al. (2025)) and TileLang (Wang et al. (2025a)), achieving more than a 2× speedup over a baseline adapted from open-source code.

Top-k Memory Optimization. Under packing, the original indexer score has shape T × T, causing memory waste because samples are mutually invisible. We reduce score storage to T × max\_seq and use flashinfer.top\_k\_ragged\_transform to compute only over valid KV regions. A chunked indexer is used as a memory-bound fallback for extreme cases.

Short-Sequence Optimization. During long-sequence SFT, many samples are short. When the positional index satisfies i < topk, the indexer backward and sparse-attention kernels iterate only over causally attendable KV entries instead of scanning a fixed top-k range, bringing a 1.5× end-to-end speedup.

Indexer Loss. Instead of storing full indexer and attention scores across layers, we keep only post-top-k indexer scores during forward and recover attention scores inside the sparse-attention backward kernel. This reuses FlashAttention-style backward recomputation (Dao et al. (2022)), avoids additional compute, and releases per-layer intermediates early.

The key observation is that sparse-attention backward already recomputes the post-softmax attention matrix. By extracting and reducing the needed scores inside the backward kernel, the indexer loss can be computed without materializing layer-wise T × T intermediates.

## 5.2 Post-Training Systems

## 5.2.1 DSA Adaptation for RL

For RL training with DSA, we focus on consistency and memory. To avoid mismatch between training and inference Top-k results, we use deterministic Top-k computation. flashinfer.topk replaces torch.topk, achieving a 2–3× speedup while preserving determinism. For variable-length RL batches, a chunked DSA indexer partitions the score matrix along the sequence dimension, performs Top-k chunk by chunk, and aggregates the results to reduce peak memory.

## 5.2.2 On-Policy Distillation System

The OPD system supports heterogeneous multi-expert teacher scheduling, multimodal alignment, and Top-k distillation. Samples are routed to domain-specific teachers such as Math, Grounding, OCR, and LAN-Instruct, while domains sharing weights reuse the same server instance. To align student and teacher preprocessing, the teacher side reconstructs sequences with its native processor and verifies visual-token counts, positional encodings, and multimodal-text boundaries. Top-k distillation supports Overlap, Student-Only, and Teacher-on-Student modes.

Strict multimodal alignment is essential because even a small mismatch in image token count, video frame sampling, chat template, or mRoPE position can move the KL signal to the wrong response token. The system therefore treats alignment verification as part of the training pipeline rather than an offline debugging step.

![](images/87ac27c4217b79d7a84a8b7a3d622a01e3716a89b76e63e13cecb6c2dc7440d5.jpg)  
Figure 4: Inference cost of Keye-VL-2.0-30B-A3B. DSA-specific prefill and decode optimizations reduce the cost of ultra-long video inference relative to dense attention under the same H800 pricing assumption.

## 5.3 Efficient Inference for GQA+DSA

For ultra-long video inference, we introduce three optimizations.

• Chunk ViT: video frames are split into chunks, processed sequentially by the ViT, and then merged, reducing peak memory without changing model outputs.  
• Sparse Attention Optimization: adjacent queries often select similar Top-k KV sets. We deduplicate Top-k sets across adjacent queries and use an MMA Thread Layout-Aware Mask inside the attention kernel. With a 128K context and topk = 2048, 16 adjacent queries require only about 8K effective KV tokens.  
• Decode Optimization: DSA-specific decode optimizations reduce prefill cost by over 3× and decode cost by over 5× compared with full attention under a 128K context.

Experiments on H800 GPUs, assuming \$2 per GPU-hour, demonstrate the inference efficiency shown in Figure 4.

## 6 Comprehensive Evaluation

Before presenting benchmark-specific analyses, Figure 5 provides a compact overview of Keye-VL-2.0- 30B-A3B across video understanding, coding, agentic tool use, mathematical and scientific reasoning, instruction following, and general vision-language benchmarks. We compare with representative opensource and closed-source baselines of similar and larger scales, including Qwen3.5 (Qwen Team (2026a)), InternVL3.5 (Wang et al. (2025b)), GPT-5-mini (OpenAI (2025a)), and Qwen3-VL (Qwen Team (2025)). The following subsections give the benchmark definitions, evaluation protocols, and detailed results behind this overall view.

## 6.1 Video Understanding

We evaluate Keye-VL-2.0-30B-A3B on three complementary aspects of video understanding: long-video comprehensive understanding, fine-grained temporal grounding, and video knowledge acquisition.

## 6.1.1 Benchmarks

Long-Video Comprehensive Understanding. Video-MME-v2 (Fu et al. (2026)) evaluates omni-modal information aggregation, long-range temporal understanding, and complex reasoning. It reports both average accuracy and a group-based non-linear score (N/4)2, where N is the number of correct answers in a group of four correlated questions. LongVideoBench (Wu et al. (2024a)) evaluates long-context video-language reasoning with referring queries. MLVU (Zhou et al. (2025)) covers multi-task long-video understanding from 3-minute to 2-hour videos. Video-MME (Fu et al. (2025)) evaluates short, medium, and long videos; we report the no-subtitle setting.

These benchmarks together test whether a model can retrieve relevant segments from long videos, maintain event order, integrate cross-segment evidence, and answer questions without relying on subtitle shortcuts.

![](images/76829c296b72c19f1ce6a7e2432384a38b265bf8e66e2fd2cb083a56ea6a3f6d.jpg)

<details>
<summary>table</summary>

| STEM | MMMU | 75.3 | 81.4 | 77.7 | 79.0 | 76.0 | 78.1 | 80.6 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| STEM | MMStar | 75.9 | 81.9 | 77.9 | 74.1 | 75.5 | 79.4 | 78.7 |
</details>

Figure 5: Overall evaluation summary of Keye-VL-2.0-30B-A3B. The figure summarizes representative results across video understanding, coding, agentic tool use, mathematical and scientific reasoning, instruction following, and general vision-language benchmarks. Orange scores mark leading results in each row, and “–” indicates unavailable or not directly comparable scores. Higher is better unless otherwise specified by the corresponding benchmark; detailed benchmark descriptions and citations are provided in the subsections below.

Temporal Grounding. We use the TimeLens framework (Zhang et al. (2025)), which re-annotates ActivityNet Captions, QVHighlights, and Charades-STA to reduce annotation noise. We report mIoU on ActivityNet-TimeLens, QVHighlights-TimeLens, and Charades-TimeLens.

The cleaned annotations make TimeLens a more reliable probe of fine-grained temporal alignment than the original legacy annotations, where noisy boundaries can distort model rankings.

Video Knowledge Acquisition. Video-MMMU (Hu et al. (2025)) evaluates whether a model can learn and apply domain knowledge from instructional videos across 30 sub-disciplines and 6 major fields.

Its questions cover perception, comprehension, and adaptation levels, requiring the model to identify information in the lecture, understand the underlying concept, and apply it to new scenarios.

## 6.1.2 Results

Keye-VL-2.0 achieves the best result on LongVideoBench and Video-MME-v2 accuracy, and remains competitive on mature benchmarks such as MLVU and Video-MME. On Video-MME-v2, its strong accuracy under both 64-frame and 512-frame settings indicates that the model benefits from denser visual context without losing long-range aggregation ability. It also achieves the best mIoU on all three TimeLens subsets, validating the effectiveness of scene-wise dense captions, diverse TVG data, and tIoU-centered Video RL. In addition, for the TimeLens benchmark, the score of Qwen3-VL-235B-A22B Thinking is directly taken from the paper, where the evaluation was conducted with 2 FPS frame sampling. For Qwen3.5-35B-A3B, there is currently no official TimeLens score available. Considering that previous Qwen-series evaluations on the TVG benchmark adopted a 4 FPS setting, we evaluate the main comparison models, including Qwen3.5 35B-A3B, Gemini, and Keye-VL-2.0, using dense frame sampling at 4 FPS for a fair comparison. On Video-MMMU, it reaches 80.0, matching strong open-source baselines and approaching the closed-source GPT-5-mini.

Table 2: Video understanding evaluation of Keye-VL-2.0-30B-A3B against representative baselines. Bold marks the best result per row, underline marks the second-best, and “–” indicates an unavailable comparable score.

<table><tr><td>Category</td><td>Benchmark</td><td>Keye-VL-2.030B-A3B</td><td>Qwen3.535B-A3B</td><td>InternVL3.5241B-A28B</td><td>GPT-5-mini2025-08-07</td><td>Qwen3-VL30B-A3BThinking</td><td>Qwen3-VL32BThinking</td><td>Qwen3-VL235B-A22BThinking</td></tr><tr><td rowspan="5">Long-Video Comprehensive</td><td>LongVideoBench</td><td>74.1</td><td>61.6</td><td>67.1</td><td>-</td><td>-</td><td>-</td><td>70.5</td></tr><tr><td>Video-MME-v2 ACC (64 / 512 frames)</td><td>35.3 / 42.4</td><td>32.6 / 28.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>33.3 / 36.8</td></tr><tr><td>Video-MME-v2 Non-Lin (64 / 512 frames)</td><td>18.5 / 24.2</td><td>15.6 / 12.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>26.3 / 28.1</td></tr><tr><td>MLVU</td><td>82.8</td><td>85.6</td><td>78.2</td><td>83.3</td><td>78.9</td><td>82.3</td><td>83.8</td></tr><tr><td>Video-MME (w/o sub.)</td><td>78.3</td><td>82.5</td><td>72.9</td><td>78.9</td><td>73.3</td><td>77.3</td><td>79.0</td></tr><tr><td rowspan="3">Temporal Grounding (TimeLens)</td><td>ActivityNet-TimeLens</td><td>58.5</td><td>53.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>52.1</td></tr><tr><td>QVHighlights-TimeLens</td><td>70.1</td><td>65.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>64.6</td></tr><tr><td>Charades-TimeLens</td><td>58.4</td><td>49.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>47.8</td></tr><tr><td>Video Knowledge Acquisition</td><td>Video-MMMU</td><td>80.0</td><td>80.4</td><td>-</td><td>82.5</td><td>75.0</td><td>79.0</td><td>80.0</td></tr></table>

Table 3: Code agent evaluation of Keye-VL-2.0-30B-A3B against representative baselines. Bold marks the best result per row, underline marks the second-best, and “–” indicates an unavailable comparable score.

<table><tr><td>Benchmark</td><td>Keye-VL-2.030B-A3B</td><td>Qwen3.535B-A3B</td><td>InternVL3.5241B-A28B</td><td>GPT-5-mini2025-08-07</td><td>Qwen3-VL30B-A3BThinking</td><td>Qwen3-VL32BThinking</td><td>Qwen3-VL235B-A22BThinking</td></tr><tr><td>LiveCodeBench v6</td><td>64.2</td><td>62.8</td><td>-</td><td>51.5</td><td>-</td><td>-</td><td>-</td></tr><tr><td>OJBench</td><td>71.5</td><td>70.2</td><td>-</td><td>58.7</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SWE-bench Verified</td><td>62.0</td><td>63.5</td><td>-</td><td>55.5</td><td>-</td><td>-</td><td>-</td></tr></table>

## 6.2 Agentic Capability Evaluation

## 6.2.1 Code Agent Evaluation

We evaluate coding and software-engineering capabilities on LiveCodeBench v6 (Jain et al. (2024)), OJBench (Wang et al. (2025c)), and SWE-bench Verified (OpenAI (2025b)). LiveCodeBench v6 provides contamination-resistant programming evaluation, OJBench tests online-judge-style algorithmic correctness, and SWE-bench Verified evaluates repository-level issue resolution.

Keye-VL-2.0-30B-A3B achieves 64.2 on LiveCodeBench v6 and 71.5 on OJBench, showing strong algorithmic reasoning and execution-based self-correction. It also obtains a competitive 62.0 on SWE-bench Verified, suggesting that the model transfers part of its coding ability from isolated algorithmic problems to repository-level software engineering.

## 6.2.2 Tool-Use Evaluation

We evaluate function calling and multi-turn tool use on BFCL-V4 (Patil et al. (2025)), τ2-Bench (Barres et al. (2025)), and VitaBench (He et al. (2025)). These benchmarks cover single-turn and multi-turn function calling, dual-control stateful interaction, and complex life-service scenarios with heterogeneous tools.

Keye-VL-2.0-30B-A3B achieves the best results on τ2-Bench and VitaBench and ranks second on BFCL-V4. These results indicate strong tool selection, parameter filling, state tracking, and recovery behavior in multi-turn environments.

## 6.3 General Vision-Language Evaluation

We evaluate general vision-language capabilities on perception- and reasoning-oriented benchmarks.

Perception-oriented benchmarks cover OCR and document understanding, grounding, counting, spatial understanding, and hallucination resistance. OCRBench (Liu et al. (2024b)) and OmniDocBench (Ouyang et al. (2024)) focus on text-centric visual understanding and document intelligence. RefCOCO (Kazemzadeh et al. (2014)) evaluates referring-expression grounding, FSC-147 (Ranjan et al. (2021)) and PixMo-Count (Deitke et al. (2024)) measure counting, EmbSpatial-Bench (Du et al. (2024)) evaluates egocentric spatial relations, and HallusionBench (Guan et al. (2024)) diagnoses perceptual reliability.

Reasoning-oriented benchmarks cover visual mathematics, dynamic reasoning robustness, expert-level multimodal understanding, and vision-indispensable reasoning. WeMath (Qiao et al. (2024)), Math-Vista (Lu et al. (2023)), and DynaMath (Zou et al. (2024)) evaluate complementary aspects of visual mathematical reasoning, while MMMU (Yue et al. (2024)) and MMStar (Chen et al. (2024)) test expert-level multimodal reasoning under reduced text-only shortcut opportunities. Overall, Keye-VL-2.0-30B-A3B maintains strong general multimodal performance while showing particularly strong results in long-video understanding, temporal grounding, hallucination resistance, and visual mathematical reasoning.

Table 4: Tool-use and function-calling evaluation of Keye-VL-2.0-30B-A3B against representative baselines. Bold marks the best result per row, underline marks the second-best, and “–” indicates an unavailable comparable score.

<table><tr><td>Benchmark</td><td>Keye-VL-2.030B-A3B</td><td>Qwen3.535B-A3B</td><td>InternVL3.5241B-A28B</td><td>GPT-5-mini2025-08-07</td><td>Qwen3-VL30B-A3BThinking</td><td>Qwen3-VL32BThinking</td><td>Qwen3-VL235B-A22BThinking</td></tr><tr><td>BFCL-V4</td><td>65.7</td><td>67.3</td><td>-</td><td>55.5</td><td>-</td><td>-</td><td>-</td></tr><tr><td> $\tau^2$ -Bench</td><td>82.6</td><td>81.2</td><td>-</td><td>69.8</td><td>-</td><td>-</td><td>-</td></tr><tr><td>VitaBench</td><td>33.1</td><td>31.9</td><td>-</td><td>13.9</td><td>-</td><td>-</td><td>-</td></tr></table>

## 7 Conclusion and Future Work

We presented Kwai Keye-VL-2.0-30B-A3B, an open-source 30B-class MoE multimodal foundation model with only 3B active parameters. By bringing DeepSeek Sparse Attention into a GQA-based multimodal backbone, Keye-VL-2.0 extends effective context modeling to 256K tokens and makes hour-level video understanding practical under controllable training and inference cost. Together with native-resolution visual encoding, unified image-video processing, ViT–LM heterogeneous parallelism, DSA kernels, and Chunk ViT inference, the system is designed not only to improve benchmark accuracy, but also to support deployable long-video applications.

The post-training pipeline further addresses the capability-conflict problem that emerges when perception, reasoning, long-context understanding, and agentic behaviors are optimized together. Cross-Modal Multi-Teacher On-Policy Distillation, Context-RL, Video-RL, and specialized domain RL allow heterogeneous teachers and reward signals to be consolidated into a single MoE model without sacrificing core reasoning ability. Evaluations show that Keye-VL-2.0 achieves leading performance at its scale on long-video comprehension and fine-grained temporal localization, while remaining competitive on code, tool-use, OCR, document understanding, visual mathematics, and hallucination-resistance benchmarks. Overall, the results indicate that sparse long-context modeling and carefully staged multimodal RL can be combined in a single deployable MoE system without trading away general reasoning ability.

Future work will move beyond leaderboard-oriented optimization toward deeper deployment in real business scenarios. First, we will further integrate fine-grained long-video perception and image-text understanding into core product pipelines, including generative recommendation, content ecosystem governance, and commercial targeting. In these settings, Keye-VL is expected to provide denser semantic signals for recommendation matching, content quality assessment, and fine-grained advertising labels, turning multimodal understanding into measurable product and business value. Second, we will develop Video × Agent workflows that combine precise multimodal understanding with automated orchestration. The model will evolve from passively understanding video content to actively coordinating production loops, including large-scale video retrieval, highlight segment extraction, automated editing and packaging, and marketing-copy generation for creator and commercial scenarios. Third, using Keye-VL-2.0-30B-A3B as a validated foundation, we will continue strengthening the underlying infrastructure from DSA-based compute optimization, scalable data flywheels, and Context-RL post-training toward native multimodal modeling and deeper end-to-end fusion. In this direction, benchmark gains serve as diagnostics rather than the final objective; the long-term goal is to turn long-context multimodal intelligence into reliable, scalable infrastructure for real-world applications.

## References

Anthropic. Introducing claude opus 4.8, May 2026. URL https://www.anthropic.com/news/ claude-opus-4-8.  
Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.  
Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ2-bench: Evaluating conversational agents in a dual-control environment. arXiv preprint arXiv:2506.07982, 2025. URL https://arxiv.org/abs/2506.07982.  
Minwoo Byeon, Beomhee Park, Haecheon Kim, Sungjun Lee, Woonhyuk Baek, and Saehoon Kim. Coyo-700m: Image-text pair dataset. https://github.com/kakaobrain/coyo-dataset, 2022.  
Soravit Changpinyo, Piyush Sharma, Nan Ding, and Radu Soricut. Conceptual 12m: Pushing web-scale image-text pre-training to recognize long-tail visual concepts. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 3558–3568, 2021.  
Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang, Yuhang Zang, Zehui Chen, Haodong Duan, Jiaqi Wang, Yu Qiao, Dahua Lin, et al. Are we on the right way for evaluating large vision-language models? arXiv preprint arXiv:2403.20330, 2024.  
Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. Flashattention: Fast and memoryefficient exact attention with IO-awareness. In Advances in Neural Information Processing Systems, 2022. URL https://arxiv.org/abs/2205.14135.  
DeepSeek-AI et al. Deepseek-v3.2: Pushing the frontier of open large language models. arXiv preprint arXiv:2512.02556, 2025. URL https://arxiv.org/abs/2512.02556.  
Mostafa Dehghani, Basil Mustafa, Josip Djolonga, Jonathan Heek, Matthias Minderer, Mathilde Caron, Andreas Steiner, Joan Puigcerver, Robert Geirhos, Ibrahim M Alabdulmohsin, et al. Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution. Advances in Neural Information Processing Systems, 36:2252–2274, 2023.  
Matt Deitke, Christopher Clark, Sangho Lee, Rohun Tripathi, Yue Yang, Jae Sung Park, Mohammadreza Salehi, Niklas Muennighoff, Kyle Lo, Luca Soldaini, et al. Molmo and pixmo: Open weights and open data for state-of-the-art multimodal models. arXiv preprint arXiv:2409.17146, 2024.  
Mengfei Du, Binhao Wu, Zejun Li, Xuanjing Huang, and Zhongyu Wei. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. arXiv preprint arXiv:2406.05756, 2024. URL https://arxiv.org/abs/2406.05756.  
Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, et al. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 24108–24118, 2025.  
Chaoyou Fu et al. Video-MME-v2: Towards the next stage in benchmarks for comprehensive video understanding. arXiv preprint arXiv:2604.05015, 2026. URL https://arxiv.org/abs/2604.05015.  
Samir Yitzhak Gadre, Gabriel Ilharco, Alex Fang, Jonathan Hayase, Georgios Smyrnis, Thao Nguyen, Ryan Marten, Mitchell Wortsman, Dhruba Ghosh, Jieyu Zhang, et al. Datacomp: In search of the next generation of multimodal datasets. Advances in Neural Information Processing Systems, 36:27092–27112, 2023.  
Jiyang Gao, Chen Sun, Zhenheng Yang, and Ram Nevatia. Tall: Temporal activity localization via language query, 2017. URL https://arxiv.org/abs/1705.02101.  
Google DeepMind. Gemini 3 flash. Model page, December 2025. URL https://deepmind.google/models/ gemini/flash/.  
Google DeepMind. Gemini 3.5 flash, 2026. URL https://deepmind.google/models/gemini/flash/.  
Tianrui Guan, Fuxiao Liu, Xiyang Wu, Ruiqi Xian, Zongxia Li, Xiaoyu Liu, Xijun Wang, Lichang Chen, Furong Huang, Yaser Yacoob, et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14375–14385, 2024.  
Wei He, Yueqing Sun, Hongyan Hao, Xueyuan Hao, Zhikang Xia, Qi Gu, Chengcheng Han, Dengchang Zhao, Hui Su, Kefeng Zhang, Man Gao, Xi Su, Xiaodong Cai, Xunliang Cai, Yu Yang, and Yunke Zhao. Vitabench: Benchmarking LLM agents with versatile interactive tasks in real-world applications. arXiv preprint arXiv:2509.26490, 2025. URL https://arxiv.org/abs/2509.26490.  
Lisa Anne Hendricks, Oliver Wang, Eli Shechtman, Josef Sivic, Trevor Darrell, and Bryan Russell. Localizing moments in video with natural language, 2017. URL https://arxiv.org/abs/1708.01641.  
Kairui Hu, Penghao Wu, Fanyi Pu, Wang Xiao, Yuanhan Zhang, Xiang Yue, Bo Li, and Ziwei Liu. Videommmu: Evaluating knowledge acquisition from multi-discipline professional videos, 2025. URL https://arxiv.org/abs/2501.13826.  
Hugging Face Open-R1 Team. Open-r1. https://huggingface.co/open-r1, 2025.  
Naman Jain, King Han, Alex Gu, Wen-Ding Li, Fanjia Yan, Tianjun Zhang, Sida Wang, Armando Solar-Lezama, Koushik Sen, and Ion Stoica. Livecodebench: Holistic and contamination free evaluation of large language models for code. arXiv preprint arXiv:2403.07974, 2024. URL https://arxiv.org/abs/ 2403.07974.  
Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and Tamara Berg. ReferItGame: Referring to objects in photographs of natural scenes. In Alessandro Moschitti, Bo Pang, and Walter Daelemans (eds.), Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 787– 798, Doha, Qatar, October 2014. Association for Computational Linguistics. doi: 10.3115/v1/D14-1086. URL https://aclanthology.org/D14-1086.  
Alina Kuznetsova, Hassan Rom, Neil Alldrin, Jasper Uijlings, Ivan Krasin, Jordi Pont-Tuset, Shahab Kamali, Stefan Popov, Matteo Malloci, Alexander Kolesnikov, Tom Duerig, and Vittorio Ferrari. The open images dataset v4: Unified image classification, object detection, and visual relationship detection at scale. International Journal of Computer Vision, 128(7):1956–1981, 2020.  
Kwai Keye Team, Biao Yang, Bin Wen, Changyi Liu, Chenglong Chu, et al. Kwai keye-vl technical report. arXiv preprint arXiv:2507.01949, 2025. URL https://arxiv.org/abs/2507.01949.  
Xin Lai, Junyi Li, Wei Li, Tao Liu, Tianjian Li, and Hengshuang Zhao. Mini-o3: Scaling up reasoning patterns and interaction turns for visual search. arXiv preprint arXiv:2509.07969, 2025.  
Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick. Microsoft coco: Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pp. 740–755. Springer, 2014.  
Ye Liu, Zongyang Ma, Zhongang Qi, Yang Wu, Ying Shan, and Chang W Chen. Et bench: Towards open-ended event-level video-language understanding. Advances in Neural Information Processing Systems, 37:32076–32110, 2024a.  
Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. Ocrbench: on the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102, 2024b.  
Pan Lu, Hritik Bansal, Tony Xia, Jiacheng Liu, Chunyuan Li, Hannaneh Hajishirzi, Hao Cheng, Kai-Wei Chang, Michel Galley, and Jianfeng Gao. Mathvista: Evaluating mathematical reasoning of foundation models in visual contexts. arXiv preprint arXiv:2310.02255, 2023.  
Xingyu Lu, Tianke Zhang, Chang Meng, Xiaobei Wang, Jinpeng Wang, YiFan Zhang, Shisong Tang, Changyi Liu, Haojie Ding, Kaiyu Jiang, et al. Vlm as policy: Common-law content moderation framework for short video platform. arXiv preprint arXiv:2504.14904, 2025.  
Xingyu Lu, J. Wang, Y. F. Zhang, S. Ma, X. Hu, T. Zhang, K. Jiang, C. Liu, and K. Tang. Contextrl: Enhancing mllm’s knowledge discovery efficiency with context-augmented rl. arXiv preprint arXiv:2602.22623, 2026a.  
Xingyu Lu, Jinpeng Wang, Yi-Fan Zhang, Yankai Yang, Yancheng Long, Yiyang Fan, Xuanyu Zheng, Haonan Fan, Kaiyu Jiang, Tianke Zhang, et al. Vcap: Hypergeometric rewards for weak-to-strong visual captioning. arXiv preprint arXiv:2605.28023, 2026b.  
Fanqing Meng, Lingxiao Du, Zongkai Liu, Zhixiang Zhou, Quanfeng Lu, Daocheng Fu, Tiancheng Han, Botian Shi, Wenhai Wang, Junjun He, et al. Mm-eureka: Exploring the frontiers of multimodal reasoning with rule-based reinforcement learning. arXiv preprint arXiv:2503.07365, 2025.  
Jordan Meyer, Nick Padgett, Cullen Miller, and Laura Exline. Public domain 12m: A highly aesthetic image-text dataset with novel governance mechanisms. arXiv preprint arXiv:2410.23144, 2024.  
MiMo Team, Biao Yang, Bin Wen, Boyang Ding, Changyi Liu, Chenglong Chu, et al. MiMo-V2-Flash technical report. arXiv preprint arXiv:2605.10943, 2026. URL https://arxiv.org/abs/2605.10943.  
OpenAI. Introducing GPT-5 for developers, August 2025a. URL https://openai.com/index/ introducing-gpt-5-for-developers.  
OpenAI. Introducing SWE-bench Verified, 2025b. URL https://openai.com/index/ introducing-swe-bench-verified/.  
OpenAI. Gpt-5.5 instant: smarter, clearer, and more personalized, May 2026. URL https://openai.com/ index/gpt-5-5-instant/.  
Linke Ouyang, Yuan Qu, Hongbin Zhou, Jiawei Zhu, Rui Zhang, Qunshu Lin, Bin Wang, Zhiyuan Zhao, Man Jiang, Xiaomeng Zhao, et al. Omnidocbench: Benchmarking diverse PDF document parsing with comprehensive annotations. arXiv preprint arXiv:2412.07626, 2024. URL https://arxiv.org/abs/2412. 07626.  
Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Proceedings of the 42nd International Conference on Machine Learning (ICML), 2025. URL https://gorilla.cs.berkeley.edu/leaderboard.html.  
Runqi Qiao, Qiuna Tan, Guanting Dong, Minhui Wu, Chong Sun, Xiaoshuai Song, Zhuoma GongQue, Shanglin Lei, Zhe Wei, Miaoxuan Zhang, et al. We-math: Does your large multimodal model achieve human-like mathematical reasoning? arXiv preprint arXiv:2407.01284, 2024.  
Qwen Team. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025. URL https://arxiv.org/ abs/2511.21631.  
Qwen Team. Qwen3.5: Towards native multimodal agents, February 2026a. URL https://qwen.ai/blog? id=qwen3.5.  
Qwen Team. Qwen3.7: The agent frontier, May 2026b. URL https://qwen.ai/blog?id=qwen3.7.  
Viresh Ranjan, Udbhav Sharma, Thu Nguyen, and Minh Hoai. Learning to count everything. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3394–3403, 2021.  
Shuhuai Ren, Linli Yao, Shicheng Li, Xu Sun, and Lu Hou. Timechat: A time-sensitive multimodal large language model for long video understanding, 2024. URL https://arxiv.org/abs/2312.02051.  
Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al. Laion-5b: An open large-scale dataset for training next generation image-text models. Advances in neural information processing systems, 35:25278–25294, 2022.  
Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously large neural networks: The sparsely-gated mixture-of-experts layer. In International Conference on Learning Representations (ICLR), 2017. URL https://arxiv.org/abs/1701.06538.  
Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, SH Cai, Yuan Cao, Y Charles, HS Che, Cheng Chen, Guanduo Chen, et al. Kimi k2. 5: Visual agentic intelligence. arXiv preprint arXiv:2602.02276, 2026.  
Ryan Teknium, Jeffrey Quesnelle, and Chen Guang. Hermes 3 technical report. arXiv preprint arXiv:2408.11857, 2024.  
Thyme Team. Thyme: Think beyond images. arXiv preprint arXiv:2508.11630, 2025. URL https://arxiv. org/abs/2508.11630.  
Lei Wang, Yu Cheng, Yining Shi, Zhengju Tang, Zhiwen Mo, Wenhao Xie, Lingxiao Ma, Yuqing Xia, Jilong Xue, Fan Yang, and Zhi Yang. Tilelang: A composable tiled programming model for AI systems. arXiv preprint arXiv:2504.17577, 2025a. URL https://arxiv.org/abs/2504.17577.  
Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265, 2025b. URL https://arxiv.org/abs/2508. 18265.  
Zhexu Wang, Yiping Liu, Yejie Wang, Wenyang He, Bofei Gao, Muxi Diao, Yanxu Chen, Kelin Fu, Flood Sung, Zhilin Yang, Tianyu Liu, and Weiran Xu. OJBench: A competition level code benchmark for large language models. arXiv preprint arXiv:2506.16395, 2025c. URL https://arxiv.org/abs/2506.16395.  
Luis Wiedmann, Orr Zohar, Amir Mahla, Xiaohan Wang, Rui Li, Thibaud Frere, Leandro von Werra, Aritra Roy Gosthipaty, and Andrés Marafioti. Finevision: Open data is all you need. arXiv preprint arXiv:2510.17269, 2025. URL https://arxiv.org/abs/2510.17269.  
Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. Advances in Neural Information Processing Systems, 37: 28828–28857, 2024a.  
Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. LongVideoBench: A benchmark for long-context interleaved video-language understanding. In Advances in Neural Information Processing Systems (NeurIPS), 2024b. URL https://arxiv.org/abs/2407.15754.  
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.  
Biao Yang, Bin Wen, Boyang Ding, Changyi Liu, Chenglong Chu, et al. Kwai keye-vl 1.5 technical report. arXiv preprint arXiv:2509.01563, 2025b. URL https://arxiv.org/abs/2509.01563.  
Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, Qianyu Chen, Huarong Zhou, Zhensheng Zou, Haoye Zhang, Shengding Hu, Zhi Zheng, Jie Zhou, Jie Cai, Xu Han, Guoyang Zeng, Dahai Li, Zhiyuan Liu, and Maosong Sun. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800, 2024. URL https: //arxiv.org/abs/2408.01800.  
Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. Flashinfer: Efficient and customizable attention engine for LLM inference serving. In Proceedings of Machine Learning and Systems, 2025. URL https://arxiv.org/abs/2501.01005.  
Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, et al. Mmmu: A massive multi-discipline multimodal understanding and reasoning benchmark for expert agi. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9556–9567, 2024.  
Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training, 2023.  
Jun Zhang, Teng Wang, Yuying Ge, Yixiao Ge, Xinhao Li, Ying Shan, and Limin Wang. TimeLens: Rethinking video temporal grounding with multimodal LLMs. arXiv preprint arXiv:2512.14698, 2025. URL https://arxiv.org/abs/2512.14698.  
Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, and Junyang Lin. Group sequence policy optimization. arXiv preprint arXiv:2507.18071, 2025. URL https://arxiv.org/abs/2507.18071.  
Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Zhengyang Liang, Shitao Xiao, Minghao Qin, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. Mlvu: Benchmarking multi-task long video understanding, 2025. URL https://arxiv.org/abs/2406.04264.  
Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Hao Tian, Yuchen Duan, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.  
Chengke Zou, Xingang Guo, Rui Yang, Junyu Zhang, Bin Hu, and Huan Zhang. Dynamath: A dynamic visual benchmark for evaluating mathematical reasoning robustness of vision language models. arXiv preprint arXiv:2411.00836, 2024.

## A Case Study

The following qualitative examples show the final user-facing responses of Keye-VL-2.0 on representative text, image, video, and agentic service tasks. Since Keye-VL-2.0 uses a thinking-oriented policy by default, we omit internal reasoning traces and do not display separate thinking or answer tags in the case study.

Case I: Logical Constraint Solving

## Prompt

The Protocol of the Corrupted AI. A cybersecurity analyst must override a rogue AI system by entering five unique Access Keys into five slots. Each Access Key contains exactly one letter from {A, B, C, D, E} and one digit from {1, 2, 3, 4, 5}. No letter or digit may repeat. Five intercepted nodes describe the sequence, but exactly one node is corrupted: every statement in the corrupted node is false, while every statement in each intact node is true.

Node 1. A is immediately to the left of the slot containing digit 5. Digit 3 is located in Slot 5.

Node 2. C is paired with an even digit. The slot containing E is exactly two slots to the right of the slot containing digit 4.

Node 3. Digit 2 is adjacent to the slot containing D. B is neither in Slot 1 nor Slot 5.

Node 4. The sum of the digits in Slot 1 and Slot 2 equals the digit in Slot 3. The digit paired with A is strictly less than the digit paired with B.

Node 5. D is somewhere to the left of C. Digit 1 is paired directly with E. Slot 4 contains an odd digit.

Mission. Identify the corrupted node and determine the exact letter-digit sequence for all five slots.

## Model Response

Corrupted node. Node 5 is the corrupted node. All statements in Node 5 are false, while all statements in Nodes 1–4 are true.

Exact sequence.

<table><tr><td>Slot</td><td>Letter</td><td>Digit</td></tr><tr><td>1</td><td>C</td><td>4</td></tr><tr><td>2</td><td>A</td><td>1</td></tr><tr><td>3</td><td>E</td><td>5</td></tr><tr><td>4</td><td>B</td><td>2</td></tr><tr><td>5</td><td>D</td><td>3</td></tr></table>

Verification.

<table><tr><td>Node</td><td>Status and evidence</td></tr><tr><td>Node 1</td><td>True. A in Slot 2 is immediately to the left of digit 5 in Slot 3; digit 3 is in Slot 5.</td></tr><tr><td>Node 2</td><td>True. C is paired with digit 4, which is even; E in Slot 3 is exactly two slots to the right of digit 4 in Slot 1.</td></tr><tr><td>Node 3</td><td>True. Digit 2 in Slot 4 is adjacent to D in Slot 5; B is in Slot 4, so it is neither in Slot 1 nor Slot 5.</td></tr><tr><td>Node 4</td><td>True. The Slot 1 and Slot 2 digits sum to the Slot 3 digit: 4 + 1 = 5. The digit paired with A is 1, which is less than the digit paired with B, 2.</td></tr><tr><td>Node 5</td><td>False. D is in Slot 5 and C is in Slot 1, so D is not left of C. Digit 1 is paired with A, not E. Slot 4 contains digit 2, which is even rather than odd.</td></tr></table>

The assignment satisfies the uniqueness constraint for all letters and digits and leaves exactly one corrupted node.

Figure 6: Text case for logical constraint solving. Keye-VL-2.0 solves a multi-constraint symbolic puzzle by identifying the single corrupted evidence node, constructing a valid slot assignment, and verifying every statement against the final sequence.

Case II: Spatial Reasoning in an Indoor Layout

Input Image  
![](images/4c03a465584f312d0f17bfdae07e0a0c2df9a9a095df05eeebdd1cb7de2b4d52.jpg)

<details>
<summary>text_image</summary>

D
H
F
E
A
C
B
A
G
</details>

## Prompt

The image is a top-down view of a room, with each piece of furniture labeled by a letter. Please answer the following questions:

1. Which direction is A (the sofa) facing?  
2. Which direction is C (the TV) facing?  
3. Are A (the sofa) and C (the TV) facing each other?  
4. When entering the room through G (the door), what furniture is on your left-hand side and what furniture is on your right-hand side?  
5. What is the relative spatial relationship between D (the floor lamp) and H (the window)?  
6. If B (the coffee table) is moved next to E (the dining table), in which direction should it be moved?

7. In which corner of the room is F (the bookshelf) located? Describe its position relative to the other furniture. Please answer based on the spatial layout shown in the image, paying close attention to orientations, directions, and relative positions.

## Model Response

Directions of A and C. A (sofa) faces toward the center of the room, i.e., to the right. C (TV) faces toward the center of the room, i.e., to the left. Therefore, A and C are facing each other, forming a typical living-room seating arrangement.

Left and right when entering from G. When standing at G and entering the room, the furniture on the left-hand side is the sofa A, while the furniture on the right-hand side is the TV C.

Relative position of D and H. D (floor lamp) is to the left of H (window), namely on the west side of the window. The floor lamp is positioned left and slightly forward of the window, and both are near the north wall.

Moving B next to E. B (coffee table) is currently south of E (dining table), or below it in the top-down layout. To move B next to the dining table, it should be moved north.

Location of F. F (bookshelf) is located in the northeast corner of the room. Relative to the other furniture, it is east of the window H, north of the TV C, and northeast of the dining table E. It is positioned near the intersection of the north and east walls.

Overall, this layout shows a standard living-room configuration: the sofa and TV face each other to form a seating area, the dining table is in the middle-north section, the window and bookshelf are on the north wall, and the door is on the south wall.

Figure 7: Image case for spatial layout understanding. Given a labeled top-down indoor scene, Keye-VL-2.0 identifies object orientations, egocentric left-right relations, furniture positions, and the direction needed to move an object.

Case III: Anatomical Reasoning and Error Detection

Input Image  
![](images/e4542a53c79b6a650fb1cf5edfec043d9091f644a2543a9bbb21b31ebb77a944.jpg)

<details>
<summary>text_image</summary>

Right Atrium
Right Venctile
Left Atrium
Triccuss Valve
Right Atrium
Superliona Aervery
Pulmonary Veins
Mitral Valve
Aorta
</details>

Oxygen-rich blood flows from lungs to left atrium → left vencicle →aorta → body  
Oxygen-poor blood returns: body→SVC →right atrium-→SVC → right  
right venticle→pulmary→pulmonaryartery→ lungs

## Prompt

This image is an anatomical diagram of the human heart and a schematic of blood circulation. Please answer the following questions:

• What are the four chambers of the heart? Please list them according to the labels in the diagram.  
• What do the blue and red arrows in the diagram represent, respectively? Hint: it is related to blood oxygen content.  
• After blood flows from the lungs into the heart, which chambers and blood vessels does it sequentially pass through before being delivered to the rest of the body?  
• Where is the “Mitral Valve” labeled in the diagram located? Which two chambers does it connect?  
• Based on the text description in the diagram, point out one scientifically incorrect statement and correct it.  
Please base your answers on the labels and text information provided in the image.

## Model Response

Four chambers. According to the labels in the diagram, the four chambers of the heart are Right Atrium, Right Ventricle, Left Atrium, and Left Ventricle.

Meaning of blue and red arrows. Blue arrows represent the flow of oxygen-poor blood with low oxygen content. Red arrows represent the flow of oxygen-rich blood with high oxygen content. This is consistent with the diagram text describing oxygen-poor blood returning and oxygen-rich blood flowing from the lungs.

Pathway from the lungs to the body. The sequence is: lungs → left atrium → left ventricle → aorta → entire body.

Position of the Mitral Valve. The Mitral Valve is located between the left atrium and the left ventricle. It connects the Left Atrium and the Left Ventricle.

Scientific error and correction. The path description for oxygen-poor blood states: body → SVC → right atrium → SVC → right ventricle. This is incorrect because the SVC is used twice, and it carries blood from the body to the right atrium rather than from the right atrium to the right ventricle. The corrected pathway should be: body → SVC → right atrium → right ventricle → pulmonary artery → lungs.

Additionally, there is a spelling error in the diagram: “pulmary” should be “pulmonary.”

Figure 8: Image case for anatomical diagram understanding. Given a labeled heart diagram with blood-flow arrows and explanatory text, Keye-VL-2.0 identifies cardiac chambers, interprets oxygen-rich and oxygen-poor circulation, localizes the mitral valve, reconstructs the pulmonary-to-systemic pathway, and detects an incorrect textual description.

Case IV: Scene-by-Scene Historical Video Understanding

Input Video  
![](images/b9b9a3e021254027ba9c05414c63d8839b79e0b02eaac9f34b1bd536057329c7.jpg)

<details>
<summary>text_image</summary>

Scene 1
神话传说就会成为历史
and we know about the Shang's story here have left us
我们了解鸟眼是因为它给我们留下了
And we know about the Shang's story here have left us
难以诚信的是追寻鸟眼轮廓的线索
And the clues to the mystery of the Shang's story here
被刻有三些印记:他认出那紫
Scene 4
Scene 5
中国考古学家取得了他们是伟大的发现
and our reflection elsewhere being a great history
接下来就会产生一套耀眼
and our friend will create some new
所以通过文字:中国人民与他们的过去
to through their stars like Chinese people are uniquely connected
</details>

## Prompt

Describe the content of the video by scene and present it in English.

## Model Response

Scene 1: Opening Rural Landscape and Mythical History. The video begins with a yellow flag waving against a clear sky, framed by lush green trees. The scene transitions to a vast golden wheat field where a farmer harvests crops with a sickle. Children ride bicycles along a rural road, passing by motorbikes. Subtitles state, “If this was the capital of the Xia, for the Chinese, myth would become history, for they would have found the root of the Chinese state.”

Scene 2: Highway Journey and Historical Transition. A man drives on a highway, with the rearview mirror reflecting his face. Subtitles mention “around 1200 BC” and the search for China’s earliest historical rulers, specifically the Shang Dynasty, noted for leaving the “first Chinese writing.” The camera shows a roadside statue of a laughing Buddha and highway traffic.

Scene 3: Traditional Street and Chinese Medicine Store. The scene shifts to a bustling traditional street with pedestrians, a child blowing bubbles, and ornate architectural details. It transitions inside an old Chinese medicine store, where staff prepare herbal packages, and a foreign man navigates the crowded pharmacy. Subtitles highlight the Shang Dynasty’s discovery as “one of the most exciting stories in world archaeology” and emphasize traditional Chinese medicine’s ancient roots.

Scene 4: Wang Yirong’s Discovery of Oracle Bones. Inside the pharmacy, the foreign man recounts the story of Wang Yirong in 1899, a Beijing Hanlin Academy chancellor who, while ill with malaria, purchased medicine containing “dragon bones,” namely animal bones. Upon opening the package, he discovered inscriptions resembling ancient bronze writings, leading to the tracing of these bones to Anyang, Henan. Close-ups show him examining the inscribed bones.

Scene 5: Anyang Archaeological Excavation. Black-and-white historical photos depict the Anyang excavation site, showing massive tombs of late Shang kings with human sacrifices and rows of skulls. Subtitles note that the 1928 excavation uncovered nearly 30,000 oracle bones documenting divination activities of nine Shang kings. The sequence includes images of archaeologists at work and close-ups of inscribed bones.

Scene 6: Expert Discussion on Oracle Bone Divination. Two experts discuss the oracle bone divination process. One explains how Shang kings burned cracks in turtle shells or ox bones to communicate with ancestors, interpreting crack patterns to determine auspiciousness. They compare oracle bone characters with modern Chinese script, such as the character for “rain,” illustrating its evolution from pictographs to modern form.

Scene 7: Evolution of Chinese Script Animation. An animated sequence illustrates the transformation from prehistoric oracle bone pictographs to modern Chinese script, showing thousands of characters linking the present to the past. Subtitles emphasize how Chinese people are uniquely connected to their history and thinking through their script, surpassing other cultures globally.

Figure 9: Video case for long-form scene-level understanding. Keye-VL-2.0 summarizes a historical documentary clip by segmenting the video into coherent scenes, tracking transitions from rural landscapes and travel shots to traditional medicine, oracle-bone discovery, Anyang excavation, expert discussion, and the evolution of Chinese script.

Case V: Daily Vlog and Equestrian Lesson Understanding

Input Video  
![](images/1e142fd895b0425aa01b7f20e415258c8040748e3d8704dae5162fc82770fb90.jpg)

<details>
<summary>text_image</summary>

00:00-00:08
00:08-00:14
00:14-00:18
00:18-00:28
00:28-00:38
00:38-00:52
00:52-00:58
00:58-01:14
01:14-01:24
01:24-01:37
</details>

## Prompt

Describe the content of the video by scene and present it in English.

## Model Response

The model segments a daily vlog into ten timestamped scenes, preserving the user’s school preparation, equestrian lesson, campus classes, family interaction, and evening routine. The response below is presented as a compact timeline.

<table><tr><td>Time</td><td>Scene summary</td></tr><tr><td>00:00–00:08</td><td>Living room preparation for school. In a bright living room at 11:39 AM, the woman shows a white drawstring bag and its contents, including a wallet, tablet, and water bottle. On-screen text introduces the first day of class at Columbia and the end of summer break.</td></tr><tr><td>00:08–00:14</td><td>Kitchen snack packing. She opens organized kitchen drawers, selects snack packets such as black sesame cakes, and packs them for a practical school day.</td></tr><tr><td>00:14–00:18</td><td>Changing shoes for a riding lesson. She switches into black chunky boots, prepares a white tote bag, and notes that she also has a first riding lesson that day.</td></tr><tr><td>00:18–00:28</td><td>Driving to the stable. She drives through an automatic gate, sees horses in paddocks, and enters a stable where a farrier is shoeing a horse.</td></tr><tr><td>00:28–00:38</td><td>Riding lesson preparation. With an instructor&#x27;s help, she puts on a helmet, mounts the horse, and stands ready for the lesson while expressing affection and trust toward horses.</td></tr><tr><td>00:38–00:52</td><td>Riding lesson. In an indoor arena, she practices balance and control while the instructor guides the horse around the ring.</td></tr><tr><td>00:52–00:58</td><td>Driving to school. After the lesson, she changes shoes in the car and buys coffee before class, noting that the timing works out smoothly.</td></tr><tr><td>00:58–01:14</td><td>Campus scenes and classes. She walks through a historic campus hallway with coffee, attends two classes, and reflects positively on her third and final semester at Columbia.</td></tr><tr><td>01:14–01:24</td><td>Family surprise with cat. Returning to the car, she finds that her husband has brought their cat Fengfeng in a carrier; she hugs the cat and feels supported by her family.</td></tr><tr><td>01:24–01:37</td><td>Evening health routine. Back home, she prepares vitamins, probiotics, fiber supplements, and green smoothies with her husband, ending the day with a toast and thanks to the viewer.</td></tr></table>

Figure 10: Video case for scene-level daily vlog understanding. Keye-VL-2.0 follows a long-form personal vlog across preparation for school, snack packing, a riding lesson, campus classes, a family surprise, and an evening health routine, preserving both temporal boundaries and fine-grained lifestyle details.

## Prompt

A user asks the assistant to arrange several services for parents visiting Quanzhou on the same day. The request evolves across multiple turns: recommend and book a comfortable hotel near the user’s home, choose a spot-removing skincare product that can arrive within one hour, pay the hotel and delivery orders, recommend a Minnan-cuisine restaurant with Ginger Duck and a rich family set meal, create the restaurant order, and finally pay it.

## Model Response

Personalized planning. The assistant first retrieves the user’s profile to ground the task in verified information. It identifies the home address, infers that the user’s most frequent hotel brand is Home Inn, converts the home address to coordinates, and searches for nearby hotels in Quanzhou. It recommends Home Inn (Quanzhou Tian’an Road Branch) because it is 1.2 km from the user’s home, matches the user’s historical hotel preference, provides a comfortable queen-bed room, and costs only 288 RMB within the 1000 RMB budget.

Hotel booking. After the user confirms the recommendation, the assistant creates a hotel order for one night in the comfortable queen-bed room:

<table><tr><td>Item</td><td>Result</td></tr><tr><td>Hotel</td><td>Home Inn (Quanzhou Tian&#x27;an Road Branch)</td></tr><tr><td>Room</td><td>Comfortable queen-bed room</td></tr><tr><td>Date</td><td>2024-12-30</td></tr><tr><td>Order ID</td><td>OOd651df337d</td></tr><tr><td>Price</td><td>288 RMB</td></tr></table>

Delivery recommendation and timing verification. For the skincare request, the assistant searches for spotremoving skincare products, retrieves store locations, computes distances from the user’s home, and converts distances into delivery times. It recommends Yuquan Spot-Removing Serum from Yueyan Skincare Store, because the store is 412 meters away and the estimated delivery time is 28 minutes, satisfying the one-hour constraint.

Cross-domain order execution. The assistant then creates and pays the delivery order, handles the hotel payment with the correct hotel-payment tool after detecting a payment-type mismatch, and searches in-store restaurant options for Minnan cuisine with Ginger Duck. It recommends Minnan Flavor Restaurant and its Traditional Minnan Family Banquet Set (3–4 people) because the set contains Ginger Duck, is suitable for a family meal, and costs 398 RMB. After user confirmation, it creates and pays the in-store order.

Final confirmed arrangement.

<table><tr><td>Service</td><td>Selection</td><td>Order ID</td><td>Status</td></tr><tr><td>Hotel</td><td>Home Inn queen-bed room, 288 RMB</td><td>OOd651df337d</td><td>Paid</td></tr><tr><td>Delivery</td><td>Yuquan Spot-Removing Serum, 258 RMB</td><td>OTad2c5d4616</td><td>Paid; arrives at 15:58</td></tr><tr><td>Restaurant</td><td>Minnan family banquet set, 398 RMB</td><td>OI3ddaed0869</td><td>Paid</td></tr></table>

Agent capability demonstrated. This case shows that Keye-VL-2.0 can maintain a multi-turn service plan, use user history for personalization, call tools across hotel, delivery, geolocation, distance estimation, and in-store ordering domains, recover from an incorrect payment-tool route, and complete all user-confirmed transactions while keeping the final response concise.

Figure 11: Agent case for multi-domain service orchestration. Keye-VL-2.0 coordinates personalization, search, geolocation, delivery-time estimation, booking, payment, recommendation, and order creation across hotel, delivery, and in-store service domains.

## B Contribution (Alphabetical order)

## Core Contributors

Bin Wen, Changyi Liu, Chengru Song, Chongling Rao, Guowang Zhang, Han Li, Haonan Fan, Hengrui Ju, Jiankang Chen, Jiapeng Chen, Jiawei Yuan, Kaixuan Yang, Kaiyu Jiang, Kun Gai, Lingzhi Zhou, Na Nie, Sen Na, Tianke Zhang, Tingting Gao, Xuanyu Zheng, Yulong Chen

## Contributors

Major. Fan Yang, Haixuan Gao, Lele Yang, Mingqiao Liu, Muxi Diao, Qi Zhang, Qile Su, Wei Chen, Wentao Hong, Xingyu Lu, Yancheng Long, Yankai Yang, Yingxin Li, Yiyang Fan, Yu Xia, Yuzhe Chen, Ziliang Lai

Active. Chuan Yi, Haonan Jia, Tianming Liang, Weixin Xu, Xiaoxiao Ma, Yang Tian, Yufei Han

## Supporting Contributors

Feng Han, Hang Li, Jing Wang, Jinghui Jia, Junmin Chen, Junyu Shi, Ruilin Zhang

![](images/50754b4ff8085b33abc1ff4e460849d6ef34234adb038e2cbfcccc8f3c457a0c.jpg)

<details>
<summary>natural_image</summary>

Fantasy landscape with a futuristic island, filmstrip sculpture, and glowing circular elements under a bright sky (no text or symbols)
</details>