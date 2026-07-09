# Segmentation before Answering: Pixel Grounding for MLLM Visual Reasoning

Yake Wei <sup>1</sup> <sup>2</sup> Yuan Wang <sup>3</sup> Fengyun Rao <sup>4</sup> Jing LYU <sup>4</sup> Di Hu <sup>1</sup> <sup>2</sup>

## Abstract

Recent advancements in Multimodal Large Language Models (MLLMs) have evolved from static perception to interleaved visual-language reasoning, often referred to as “thinking with images”. A basic operation in this reasoning process is to zoom in on regions of interest (often represented with bounding boxes) to acquire finer visual details. In this paper, we propose Segmentation before Answering (SegAnswer), which shifts the unit of zoom-in from the popular bounding box to pixel-level segmentation mask. By employing fine-grained masks to isolate the target area from cluttered environments, segmented visual input yields a more precise region of interest, effectively filtering out redundant background and interfering objects. Furthermore, the discrete patches of segmented visual input align more seamlessly with how MLLMs structure visual tokens via positional embeddings. In experiments, we evaluate SegAnswer across diverse benchmarks, including high-resolution perception, general perception, and hallucination. It achieves consistent improvements and also exhibits considerable performance on segmentation tasks, validating its capability for reliable pixel grounding.

## 1. Introduction

Recent advancements in Multimodal Large Language Models (MLLMs) have demonstrated remarkable progress across a broad range of vision-language tasks (Li et al., 2025a; Bai et al., 2025), covering from fundamental image captioning to complex visual question answering. Moving beyond static visual perception where the visual inputs of MLLMs are fixed, immutable contexts processed in a single forward pass, explorations about interleaved visual-language reasoning have raised wide attention recently (Wang et al., 2025b;

Zheng et al., 2025). These methods are often referred to as “thinking with images”. In these methods, models often actively manipulate visual inputs to capture and re-examine regions of interest, thereby acquiring finer details beyond the original input image to generate the final answer.

Among the various operations employed when thinking with images, a basic operation in this reasoning process is to zoom in on regions of interest (Wang et al., 2025b; Zheng et al., 2025). Concretely, when executing a zoom-in operation, the model would typically generate bounding boxes (BBox) that delineate the area. Subsequently, this localized area is cropped from the original image, serving as a new visual input for the model’s next step of reasoning. Despite its widespread adoption, this kind of BBox-based zoomin operation encounters significant and often overlooked challenges in the complex practical applications.

As illustrated in Figure 1a, a primary limitation arises from the inherent irregularity of target object shapes. The rigid rectangular prior of bounding boxes fails to capture the di verse geometric characteristics of natural objects, which are often irregular. Consequently, rectangular boxes inevitably encompass substantial redundant background, resulting in inefficient token consumption. Furthermore, in scenarios involving overlapping objects, simple bounding boxes often fail to disentangle the target from adjacent entities. This inability to precisely isolate the region of interest would lead to semantic interference, where the model struggles to distinguish the intended object from its surroundings. Fundamentally, these issues stem from the coarse granularity of bounding boxes, which lack the capacity to provide a precise observation of the region of interest.

To mitigate these limitations of the current BBox-based zoom-in operation, we shift from rectangular bounding boxes to pixel-level segmentation masks. As shown in Figure 1b, this segmentation-based pixel grounding requires the model to generate a fine-grained and accurate segmentation mask of the region of interest, instead of coarse and rectangular bounding boxes. By leveraging these segmented visual inputs, it effectively filters out redundant background and extraneous visual signals, ensuring that only the regions of interest are preserved for subsequent reasoning. Furthermore, for the newly introduced segmented visual input, we keep the position index of each segmented image patch as it is in the original image, thereby sparse segmented image patches can effectively reflect spatial relations. This also aligns more seamlessly with the nature that MLLMs structure visual tokens via positional embedding. Then, these cleaner, unambiguous visual inputs are further fed into models to provide finer details, enhancing visual reasoning.

![](images/9b55edcfa52918c6c77be19dfdeabd202b5f953a1902b628296e4ed98c55811f.jpg)  
(a) BBox-based zoom-in operation.

![](images/ea28360a897876c3e5231b916562664db2859059a3deef858a96bbccfc54b64d.jpg)  
(b) Our segmentation-based pixel grounding.  
Figure 1. Comparison between the BBox-based zoom-in operation and our segmentation-based pixel grounding. (a) Rectangular bounding boxes inevitably introduce redundant background regions (e.g., the background around the sloping tennis racket) and fail to precisely disentangle the region of interest from overlapping objects (e.g., the fork and napkin), leading to visual noise and semantic ambiguity. (b) Pixel-level segmentation can precisely isolate the region of interest, effectively eliminating background noise and decoupling adjacent entities. In addition, by keeping the position index of the original image, sparse segmented image patches can also effectively reflect spatial relations.

Based on the above pixel grounding, we propose the Segmentation before Answering (SegAnswer) method in this paper, which employs the segmentation during the visual reasoning process. Concretely, the training of SegAnswer has three stages. We first conduct pixel grounding to equip the model with semantic segmentation ability. Then, a multimodal interleaved supervised fine-tuning is conducted to instruct the model to perform pixel grounding as an intermediate conversation step, utilizing the segmented images for generating the final answer. Finally, reasoning with pixel grounding is carried out to enhance the MLLM visual reasoning by reinforcement learning.

In experiments, we first evaluate our method across a wide range of benchmarks for MLLMs, including highresolution perception (V\* (Wu & Xie, 2024), HR-Bench 4K, and 8K (Wang et al., 2025c)), general perception (MM-Bench (Liu et al., 2024b), VisuLogic (Xu et al., 2025), and MMVP (Tong et al., 2024)), and hallucination benchmarks (POPE (Li et al., 2023b) and Hallusionbench (Guan et al., 2024)). A broad range of empirical results across these diverse benchmarks demonstrates that our method achieves consistent and considerable improvements over existing visual reasoning baselines. Beyond reasoning outcomes, we also assess the pixel grounding ability of SegAnswer. On segmentation benchmarks (RefCOCO (Kazemzadeh et al., 2014), RefCOCO+ (Kazemzadeh et al., 2014), and RefCOCOg (Mao et al., 2016)), our method exhibits strong pixel-level grounding performance, surpassing prior segmentation-specific approaches. In a nutshell, our contributions are threefold:

• We propose to shift from rigid rectangular bounding boxes to pixel-level segmentation for localizing the region of interest more precisely.

• We propose the SegAnswer method, which utilizes pixel-level grounding to achieve more accurate capture of regions of interest, thereby facilitating the visual reasoning ability of MLLMs.

• Our method achieves consistent improvements over previous visual reasoning methods across multiple types of evaluation benchmarks.

## 2. Related Work

## 2.1. Multimodal Large Language Models

The integration of visual encoders with powerful Large Language Models (LLMs) endowed MLLMs with visual perception capabilities (Alayrac et al., 2022; Awadalla et al., 2023). Representative architectures, such as LLaVA (Liu et al., 2024a) and BLIP-2 (Li et al., 2023a), established a prevailing paradigm where a visual connector projects image features into a text-aligned embedding space, enabling LLMs to process visual signals alongside textual inputs. While these advancements have significantly boosted performance across general vision-language tasks, the perception of visual inputs remains static. In other words, conventional MLLMs typically process visual inputs through a single forward pass as a fixed and immutable context. Consequently, the visual information available for reasoning is restricted to the initial abstraction, preventing the model from verifying details that were not prominent in the global view.

## 2.2. MLLMs Visual Reasoning

Beyond static perception, recent advancements in MLLMs have embraced visual reasoning, which is referred to as “thinking with image”. These methods empower models to actively manipulate visual content by diverse operations, including drawing auxiliary lines (Hu et al., 2024), shifting image styles (Liu et al., 2025), highlighting sub-regions (Fu et al., 2025), and zooming in on specific areas (Wang et al., 2025b; Zheng et al., 2025). Among these operations, the zoom-in mechanism is prevalent and effective for capturing fine-grained details for the region of interest. Nevertheless, natural objects are rarely perfect rectangles, and their shapes are often irregular or non-convex. However, typical implementations of zoom-in rely on generating bounding boxes to define the region of interest. This BBox-based zoom-in operation would capture redundant background information and fail to effectively disentangle overlapping objects, leading to semantic ambiguity. To address these limitations, we introduce SegAnswer, a method that replaces coarse bounding boxes with precise semantic segmentation. By leveraging pixel-level grounding, SegAnswer isolates the target object accurately, eliminating visual interference and providing a clean context for subsequent reasoning.

## 3. Method

In this paper, we introduce SegAnswer, a novel framework designed to enhance the MLLM visual reasoning by precise, pixel-level grounding. Unlike former approaches that rely on bounding boxes for visual content capture, SegAnswer empowers the MLLM to actively segment regions of interest during the reasoning process. This capability allows the model to eliminate background noise and resolve semantic ambiguities inherent in complex visual scenes. As illustrated in Figure 2, our training has three stages: pixel grounding, multimodal interleaved supervised fine-tuning (SFT), and reasoning with pixel grounding.

## 3.1. Stage 1: Pixel Grounding

This stage aims to align textual semantics with pixel-level visual features, effectively enabling the model’s pixel grounding ability, catching and segmenting the region of interest based on textual instructions.

Our base model is Qwen2.5-VL-7B (Bai et al., 2025). To enable mask prediction for segmentation, we integrate a projector and the SAM 2.1 (Ravi et al., 2025) as the mask decoder. The projector is a simple MLP, and it aims to help align the hidden tokens of the MLLM with the mask decoder. To obtain the mask based on the hidden tokens of the MLLM, a special token, <|seg|>, is utilized as the interface. During the forward pass, we extract the lastlayer hidden states corresponding to this token. Then, these hidden states are processed by the projector and mapped into the input space of the mask decoder. Then, the mask decoder will decode the mask result as in (Ravi et al., 2025). An example of the prompt template is as follows:

Prompt 1: <image>. Segment the “food wrapped up closest to plate” in the image. Response 1: It is <|seg|>.

For this stage, the training objective is a linear combination of the next token prediction loss and segmentation-specific losses. For the segmentation-specific losses, we adopt the loss configuration from SAM 2.1 (Ravi et al., 2025), which includes a focal loss and dice loss for mask prediction, a mean-absolute-error loss for IoU prediction, and a crossentropy loss for object prediction. To ensure stable feature alignment, we implement a two-phase training strategy. Initially, we first freeze both the MLLM backbone and the mask decoder, updating only the newly introduced projector. Subsequently, Low-Rank Adaptation (LoRA) (Hu et al., 2022) is applied to the MLLM. The projector, mask decoder, and LoRA layers of the MLLM are jointly fine-tuned. After Stage 1, the LoRA weights are merged into the MLLM parameters, equipping the model with the ability to perform pixel-level grounding.

## 3.2. Stage 2: Multimodal Interleaved SFT

Building upon the pixel grounding ability established in Stage 1, this stage aims to evolve the model’s capability to handle multimodal interleaved reasoning conversation patterns, where segmented images of pixel grounding are employed as an intermediate conversation step.

As illustrated in the prompt template below, the model is instructed to first analyze the image and question to decide if a specific region requires isolation. If affirmative, it will generate a <|seg|> token to execute the segmentation. Then, the segmented image will be decoded as described in Section 3.1. The resulting segmented image is then fed into the model, allowing the model to observe fine-grained visual information of the region of interest.

![](images/f90869c374be9e292b4b3672e29897a828fcd9a8b6c35382549b7d1667e5cb23.jpg)  
Figure 2. Overview of our SegAnswer method. The training pipeline progresses through three stages: Stage 1: Pixel Grounding aligns textual semantics with pixel-level features, training the MLLM to generate segmentation masks via a specialized <|seg|> token and a mask decoder. Stage 2: Multimodal Interleaved SFT enables the model to employ segmentation as an intermediate conversation step, using the generated mask to focus the visual context before answering. Stage 3: Reasoning with Pixel Grounding utilizes reinforcement learning to enhance MLLM visual reasoning by precise and finer segmented visual inputs.

![](images/39507fef986faff535976155c17d741d75011466d5628d97e2803b2fa54e4354.jpg)

It should be noted that we keep the position index of each segmented image patch as it is in the original image, as shown in Figure 1b. In this way, sparse segmented image patches can effectively reflect spatial relations. This way also aligns more seamlessly with the nature that MLLMs structure visual tokens through positional embeddings.

In addition, the optimization of the model relies solely on the next-token prediction loss derived from the language modeling objective in this stage. We do not apply explicit supervision to the segmentation mask output. Therefore, the projector and mask decoder are frozen to preserve the segmentation ability learned in Stage 1. We unfreeze and fine-tune the full parameters of the MLLM backbone.

By the end of this stage, the model is expected to be equipped with preliminary capabilities to handle multimodal interleaved scenarios, effectively treating segmentation as an intrinsic tool and establishing a solid behavioral initialization for subsequent reinforcement learning.

## 3.3. Stage 3: Reasoning with Pixel Grounding

This stage focuses on optimizing the reasoning strategy through Reinforcement Learning (RL) to enhance the visua reasoning ability of the MLLM.

Unlike text-only RL, where the state consists solely of text tokens, the visual reasoning process incorporates visual tokens of segmented areas derived from pixel grounding. At each time step t, the i-th rollout sequence $o _ { i } ^ { t }$ is defined as the interleaved sequence of text and visual history:

$$
o _ {i} ^ {t} = \{\mathbf {X} _ {\leq t}, \mathbf {I} _ {\leq t} \} = (X _ {0}, I _ {0}), \ldots , (X _ {t}, I _ {t}),\tag{1}
$$

where $\mathbf { X } _ { \leq t }$ represents the accumulated text tokens, $I _ { 0 }$ represents the visual tokens of the original image, and $\mathbf { I } _ { \geq 1 }$ represents the segmented images. Also, as stated in Section 3.2, the position index of segmented images is also kept the same as it is in the original image, to well preserve the spatial relations. Then, given this state, the policy model (i.e., the MLLM) generates the next token in the sequence. This process continues iteratively until the model generates a final answer or reaches a maximum step limit.

During visual reasoning, intermediate visual operations (here is pixel grounding) lack explicit ground truth supervision. Therefore, we adopt an outcome-driven reward formulation that evaluates the entire reasoning trajectory based on result quality. The total reward has two parts: an accuracy reward $R _ { \mathrm { a c c } } ,$ and a format reward $R _ { \mathrm { f o r m a t } }$ . Given a final reasoning trajectory τ , the total reward is defined as:

$$
R (\tau) = w _ {a} * R _ {\mathrm{acc}} (\tau) + w _ {f} * R _ {\mathrm{format}} (\tau),\tag{2}
$$

where $w _ { a }$ and $w _ { f }$ is the weight of $R _ { \mathrm { a c c } } ( \tau )$ and $R _ { \mathrm { f o r m a t } } ( \tau )$ repsectively.

For the RL algorithm, we leverage the decoupled clip and dynamic sampling Policy Optimization (DAPO) (Yu et al., 2025). The effectiveness and efficiency of this algorithm have been verified across different tasks. Also, for the final multimodal trajectories, we use a token-wise masking setting that restricts loss calculation solely to model-predicted tokens, omitting the observation inputs.

After three stages, it completes the training pipeline of SegAnswer, equipping the MLLM with a visual reasoning capability with pixel grounding.

## 4. Experiment

## 4.1. Training Data

Stage 1: In this stage, we utilize multiple object segmentation datasets to enable the model with pixel grounding ability. The used datasets include RefCOCO (Kazemzadeh et al., 2014), RefCOCO+ (Kazemzadeh et al., 2014), RefCOCOg (Mao et al., 2016), RefClef (Kazemzadeh et al.,

2014), ReasonSeg (Lai et al., 2024), ADE20K (Zhou et al., 2017), COCOStuff (Caesar et al., 2018), Mapillary Vistas (Neuhold et al., 2017), PACO-LVIS (Ramanathan et al., 2023) and PASCAL-Part (Chen et al., 2014). Stage 2: In this stage, we use VisualCOT (Shao et al., 2024) as the training data. This is a large-scale VQA dataset that contains 438k question–answer pairs, each annotated with bounding boxes that mark the critical regions needed to derive the answer. It should be noted that we do not use the bounding box supervision. We only use the ground truth answer to supervise the final result. Stage 3: For RL training, we utilize ViRL39K (Wang et al., 2025a) as the training data. It is a curated collection of 39k verifiable question-answer pairs for vision-language RL training. This dataset is built on top of newly collected problems and existing datasets through cleaning, reformatting, rephrasing, and verification.

## 4.2. Training settings

For Stage 1, when training the projector, the learning rate is $1 e - 3$ . This stage uses RefCOCO, RefCOCO+, RefCOCOg, and RefClef datasets and trains the model for 5 epochs. Then, when training LoRA layers, the projector and the mask decoder, all datasets of Stage 1 are utilized to train the model for 3 epochs. The learning rate is $2 e - 5$ . For Stage 2, the learning rate is $2 e - 6 .$ , and the total epoch is 1. For Stage 3, the learning rate is $2 e - 5 ,$ , and the total epoch is 1. The number of rollouts is 4. $w _ { a }$ is 0.8, and $w _ { f }$ is 0.2.

## 4.3. Benchmarks

High-resolution perception: We evaluate our SegAnswer method on the visual detail understanding benchmarks with high-resolution visual inputs. These benchmarks require the model have a fine-grained understanding of the highresolution image. In this part, we adopt V\* (Wu & Xie, 2024), HR-Bench 4K, and HR-Bench 8K (Wang et al., 2025c) benchmarks. General perception: Besides the fine-grained visual understanding benchmarks, we also assess the model’s ability on widely-used general perception benchmarks, including MMBench (Liu et al., 2024b), VisuLogic (Xu et al., 2025), and MMVP (Tong et al., 2024). Hallucination: We also evaluate the model’s ability on typical hallucination benchmarks, including POPE (Li et al., 2023b) and Hallusionbench (Guan et al., 2024). Segmentation benchmarks: To verify the pixel grounding ability of our SegAnswer, we also evaluate our model on representative segmentation datasets, RefCOCO (Kazemzadeh et al., 2014), RefCOCO+ (Kazemzadeh et al., 2014), and RefCOCOg (Mao et al., 2016).

## 4.4. Main Results

We first evaluate SegAnswer across three distinct categories of multimodal benchmarks: High-Resolution Perception,

Table 1. Comparison with other methods on diverse benchmarks. We report performance across three categories: High-resolution perception (V\*, HR-Bench 4K and HR-Bench 8K), General perception (MMBench, VisuLogic, MMVP), and Hallucination evaluation (POPE, HallusionBench). The base model of our SegAnswer method is Qwen2.5-VL-7B.

<table><tr><td rowspan="2">Model</td><td colspan="3">High-resolution perception</td><td colspan="3">General perception</td><td colspan="2">Hallucination</td></tr><tr><td>V*</td><td>HR-4K</td><td>HR-8K</td><td>MMBench</td><td>VisuLogic</td><td>MMVP</td><td>POPE</td><td>HallusionBench</td></tr><tr><td>LLaVA-OneVision-9B† (Li et al., 2025a)</td><td>71.7</td><td>62.1</td><td>54.5</td><td>81.8</td><td>22.7</td><td>67.8</td><td>85.1</td><td>31.4</td></tr><tr><td>Qwen2.5-VL-7B† (Bai et al., 2025)</td><td>77.5</td><td>68.7</td><td>63.4</td><td>83.0</td><td>26.1</td><td>70.7</td><td>86.0</td><td>44.1</td></tr><tr><td>Pixel Reasoner† (Wang et al., 2025b)</td><td>85.5</td><td>73.9</td><td>66.4</td><td>84.7</td><td>25.3</td><td>71.1</td><td>86.8</td><td>44.6</td></tr><tr><td>DeepEyes† (Zheng et al., 2025)</td><td>84.3</td><td>73.5</td><td>69.8</td><td>85.4</td><td>26.7</td><td>71.3</td><td>87.6</td><td>45.3</td></tr><tr><td>SegAnswer</td><td>86.4</td><td>74.8</td><td>71.3</td><td>87.5</td><td>27.1</td><td>72.3</td><td>87.8</td><td>46.3</td></tr><tr><td>Δ over base model</td><td>+8.9</td><td>+6.6</td><td>+8.6</td><td>+4.5</td><td>+1.0</td><td>+1.6</td><td>+1.8</td><td>+2.2</td></tr></table>

† Re-evaluated using its official model and evaluation code.

Table 2. Performance comparison on the fine-grained visua perception benchmark, V\*. Other methods that target this task are further compared.

<table><tr><td>Model</td><td>V*</td></tr><tr><td>LLaVA-OneVision-9B† (Li et al., 2025a)</td><td>71.7</td></tr><tr><td>Qwen2.5-VL-7B† (Bai et al., 2025)</td><td>77.5</td></tr><tr><td>SEAL (Wu &amp; Xie, 2024)</td><td>75.4</td></tr><tr><td>DyFo (Li et al., 2025b)</td><td>81.2</td></tr><tr><td>Chain-of-Focus (Zhang et al., 2025)</td><td>88.0</td></tr><tr><td>Pixel Reasoner† (Wang et al., 2025b)</td><td>85.5</td></tr><tr><td>DeepEyes† (Zheng et al., 2025)</td><td>84.3</td></tr><tr><td>SegAnswer</td><td>86.4</td></tr><tr><td>Δ over base model</td><td>+8.9</td></tr></table>

† Re-evaluated using its official model and evaluation code.

General Perception, and Hallucination Evaluation. As presented in Table 1, we compare our framework against leading open-source MLLMs, including LLaVA-OneVision-9B (Li et al., 2025a) and our backbone model Qwen2.5- VL-7B (Bai et al., 2025). In addition, we also compare our method with recent MLLM visual reasoning methods, such as Pixel Reasoner (Wang et al., 2025b) and Deep-Eyes (Zheng et al., 2025), which utilize BBox-based visual perception operation.

Based on Table 1, not surprisingly, the advantages of our approach are most demonstrated in high-resolution tasks where fine-grained detail is vital. Notably, on the V\* benchmark, our method achieves a score of 86.4, delivering a substantial improvement over the base Qwen2.5-VL-7B model. In addition, on benchmarks including V\*, HR-Bench 4K, and HR-Bench 8K, SegAnswer achieves superior performance, surpassing the other visual reasoning baselines. These results validate that compared to rectangular bounding boxes that introduce redundant background noise, precise pixellevel segmentation effectively isolates the target, allowing the model to focus exclusively on the relevant visual features required for complex visual recognition tasks.

We also assess the model performance on the general perception and representative hallucination benchmarks. According to Table 1, SegAnswer still demonstrates consistent improvements on these benchmarks. By reasoning with pixel grounding, the region of interest is accurately captured and segmented, and then the model can further show general perception improvement, besides the target fine-grained visual perception scenarios.

## 4.5. Fine-grained Visual Perception

To further assess the model’s efficacy in handling complex visual details, more comparisons on the challenging V\* benchmark are provided. V\* benchmark assesses MLLMs ability to perform fine-grained visual detail search and relative spatial reasoning. This dataset is challenging since the visual input is high resolution, and it needs a detailed visual search in the image to answer the question correctly.

As shown in Table 2, we further compare with several methods that target this task, including SEAL (Wu & Xie, 2024), DyFo (Li et al., 2025b), and Chain-of-Focus (Zhang et al., 2025). Based on the results, for this challenging task, our SegAnswer method exhibits considerable improvement. By employing pixel-level segmentation, SegAnswer strictly isolates the target features, thereby minimizing semantic interference and enabling more accurate visual reasoning in this fine-grained visual search scenario.

## 4.6. Qualitative Analysis

To observe the concrete reasoning process of SegAnswer, we visualize several inference trajectories on the V\* benchmark. The results are provided in Figure 3. These examples illustrate how the model invokes pixel grounding to observe small objects in the high-resolution image, avoiding the interference of complex backgrounds.

![](images/afd36bf01cbb570fae93a20cae1f5b8e7590708e1e37b58ab57499af660c264c.jpg)

![](images/f9740bb9a8bd6361edb7be23358d026b8fa315659903a728de91487ae31edf61.jpg)  
(a)

![](images/13d6142e989d353c8cdd72e75e75bbc3cbfc25887bfe0e039a8096986de72771.jpg)

![](images/7e3b2fa276242a3264b4f47deeb98b7c0df02ee3b46f5e23c638bc0376d7bd74.jpg)  
(b)

![](images/78edd462274c5c94606fb430f0969cd510f24ced2803115cd5a27f2c1462547d.jpg)

![](images/65199691d248246527c34a1537d924300587569baf953bc19e549912fba8067d.jpg)  
(c)

![](images/07a3ec8b24b6ae0f39235ce3a125be682b181d060d773e01fc74defbce898b09.jpg)

![](images/a738ee0f6e33ab2299add92ce98474dd860888d70f3ee4a63e0b7e7e35e96fff.jpg)  
(d)  
Figure 3. Qualitative visualization of reasoning trajectories with SegAnswer on the V\* benchmark. We showcase four examples (a-d) where the model answers fine-grained visual perception questions. In each case, the model autonomously determines the need for visual refinement, predicts the <|seg|> token to isolate the target object (e.g., the woman in the sample a, and utilizes the segmented visual context to derive the correct answer.

Based on the visualization results, our SegAnswer can not only accurately catch the single object that is related to the question (e.g., woman in Figure 3a), it can also handle the question that is related to multiple objects in the image. Consider the scenario in Figure 3b, where the model is asked to identify the spatial relationship between the broom and the folded chair, SegAnswer explicitly predicts the need for focus of both objects, generates the <|seg|> token, and produces a precise mask that isolates both the broom and the folded chair. By reasoning over this segmented region, the model correctly identifies that the broom is on the left side of the folded chair.

In addition, according to the visualization results, compared to rigid rectangular bounding boxes, our segmentation results effectively filter out redundant background and ambiguous visual information, providing precise regions of interest. These qualitative analyses further demonstrate the effectiveness of the pixel grounding ability of SegAnswer and the reliability of the reasoning process.

## 4.7. Evaluation of Pixel Grounding Capability

Since our SegAnswer method utilizes pixel grounding to enhancing visual reasoning of MLLMs, its efficacy is intrinsically dependent on the precision of the underlying segmentation. To verify the effectiveness of our model’s grounding capabilities, we evaluated SegAnswer on representative referring segmentation benchmarks: RefCOCO, RefCOCO+, and RefCOCOg. We also compare SegAnswer with other segmentation-specific methods, including LISA (Lai et al., 2024), Groundhog (Zhang et al., 2024), LaSagnA (Wei et al., 2024), VideoLISA (Bai et al., 2024), VISA (Yan et al., 2024) and Vitron (Fei et al., 2024).

Table 3. Performance comparison on referring segmentation benchmarks. We evaluate the pixel grounding quality on RefCOCO, RefCOCO+, and RefCOCOg. SegAnswer is compared against other segmentation-specific methods.

<table><tr><td rowspan="2">Model</td><td colspan="3">RefCOCO</td><td colspan="3">RefCOCO+</td><td colspan="2">RefCOCOg</td></tr><tr><td>val</td><td>test-A</td><td>test-B</td><td>val</td><td>test-A</td><td>test-B</td><td>val-u</td><td>test-u</td></tr><tr><td>LISA (Lai et al., 2024)</td><td>74.9</td><td>79.1</td><td>72.3</td><td>62.4</td><td>67.4</td><td>56.5</td><td>66.4</td><td>68.5</td></tr><tr><td>Groundhog (Zhang et al., 2024)</td><td>78.5</td><td>79.9</td><td>75.7</td><td>70.5</td><td>75.0</td><td>64.9</td><td>74.1</td><td>74.6</td></tr><tr><td>LaSagnA (Wei et al., 2024)</td><td>76.8</td><td>78.7</td><td>73.8</td><td>66.4</td><td>70.6</td><td>60.1</td><td>70.6</td><td>71.9</td></tr><tr><td>VideoLISA (Bai et al., 2024)</td><td>73.8</td><td>76.6</td><td>68.8</td><td>63.4</td><td>68.8</td><td>56.2</td><td>68.3</td><td>68.8</td></tr><tr><td>VISA (Yan et al., 2024)</td><td>72.4</td><td>75.5</td><td>68.1</td><td>59.8</td><td>64.8</td><td>53.1</td><td>65.5</td><td>66.4</td></tr><tr><td>Vitron (Fei et al., 2024)</td><td>75.5</td><td>79.5</td><td>72.2</td><td>66.7</td><td>72.5</td><td>58.0</td><td>67.9</td><td>68.9</td></tr><tr><td>SegAnswer</td><td>79.9</td><td>82.3</td><td>76.9</td><td>73.6</td><td>78.7</td><td>67.5</td><td>76.0</td><td>76.4</td></tr></table>

![](images/0e1b0ddc7b84c052eec2e1be5207f84309b8fdf930522134edb6877cc9002c80.jpg)  
(a) Accuracy reward curve.

![](images/6c2116bbead779c5cede24417c8d61bfcdde120df43bd46e81207e2d96947247.jpg)  
(b) Format reward curve.

![](images/b721b80ca65a29eb71d0d616a9dc747eda6e06ad93f3f4cc2c97eab4c11abc4a.jpg)  
(c) Total reward curve.  
Figure 4. Training reward curves during Stage 3: Reasoning with Pixel Grounding. The solid lines represent the smoothed moving average, while the shaded areas indicate the raw variation at each step.

Based on the results in Table 3, despite being designed for the visual reasoning task, SegAnswer demonstrates considerable performance across different representative referring segmentation datasets. Specifically, SegAnswer achieves more substantial performance gains on the more challenging RefCOCO+ and RefCOCOg datasets compared to Ref-COCO with relatively simple descriptions, demonstrating SegAnswer’s effectiveness in tackling more demanding semantic comprehension scenarios. These results confirm that our training pipeline can ensure the model executes the segmentation with high quality, providing a trustworthy pixel grounding ability for complex visual reasoning.

## 4.8. Learning Dynamics of RL Training

During the RL training of Stage 3, we further observe the learning dynamics of the two reward components (the accuracy reward $R _ { \mathrm { a c c } }$ and the format reward $R _ { \mathrm { f o r m a t } } )$ as well as the total reward. The results are shown in Figure 4.

The accuracy reward curve (Figure 4a) exhibits a consistent upward trend, rising from an initial value of approximately 0.35 to over 0.55. This steady ascent indicates that the model is successfully learning to leverage pixel grounding to derive correct answers. The format reward curve (Figure 4b) remains consistently high from the very onset of training. This stability exhibits the effectiveness of Stage 2, where the model is instructed to handle multimodal interleaved conversation scenarios, alleviating the cold-start issue before reinforcement learning begins (e.g., how to correctly use <|seg|>, <answer>, and /<answer> tags). Overall, the total reward curve also shows a consistent upward trend, indicating a steady training process.

## 5. Conclusion

In this work, SegAnswer is introduced to handle the finegrained visual reasoning of MLLMs by integrating pixellevel grounding. We identify that the prevailing BBox-based zoom-in operation for visual reasoning often suffers from inherent redundancy and semantic ambiguity, particularly when processing irregular shapes or overlapping objects. By replacing inflexible bounding boxes prediction with precise semantic segmentation, SegAnswer empowers the MLLM to actively isolate and focus on exact regions of interest, thereby eliminating visual noise and resolving conflicting semantic signals. Extensive empirical evaluations across diverse benchmarks demonstrate that SegAnswer consistently shows considerable performance. The model exhibits impressive intrinsic segmentation capabilities, validating its high-quality pixel grounding ability.

## References

Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for fewshot learning. Advances in neural information processing systems, 35:23716–23736, 2022.

Awadalla, A., Gao, I., Gardner, J., Hessel, J., Hanafy, Y., Zhu, W., Marathe, K., Bitton, Y., Gadre, S., Sagawa, S., et al. Openflamingo: An open-source framework for training large autoregressive vision-language models. arXiv preprint arXiv:2308.01390, 2023.

Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., et al. Qwen2. 5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

Bai, Z., He, T., Mei, H., Wang, P., Gao, Z., Chen, J., Zhang, Z., and Shou, M. Z. One token to seg them all: Language instructed reasoning segmentation in videos. Advances in Neural Information Processing Systems, 37:6833–6859, 2024.

Caesar, H., Uijlings, J., and Ferrari, V. Coco-stuff: Thing and stuff classes in context. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1209–1218, 2018.

Chen, X., Mottaghi, R., Liu, X., Fidler, S., Urtasun, R., and Yuille, A. Detect what you can: Detecting and representing objects using holistic models and body parts. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1971–1978, 2014.

Fei, H., Wu, S., Zhang, H., Chua, T.-S., and Yan, S. Vitron: A unified pixel-level vision llm for understanding, generating, segmenting, editing. Advances in neural information processing systems, 37:57207–57239, 2024.

Fu, X., Liu, M., Yang, Z., Corring, J. R., Lu, Y., Yang, J., Roth, D., Florencio, D., and Zhang, C. Refocus: Visual editing as a chain of thought for structured image understanding. 2025.

Guan, T., Liu, F., Wu, X., Xian, R., Li, Z., Liu, X., Wang, X., Chen, L., Huang, F., Yacoob, Y., et al. Hallusionbench: an advanced diagnostic suite for entangled language hallucination and visual illusion in large vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14375– 14385, 2024.

Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W., et al. Lora: Low-rank adaptation of large language models. ICLR, 1(2):3, 2022.

Hu, Y., Shi, W., Fu, X., Roth, D., Ostendorf, M., Zettlemoyer, L., Smith, N. A., and Krishna, R. Visual sketchpad: Sketching as a visual chain of thought for multimodal language models. Advances in Neural Information Processing Systems, 37:139348–139379, 2024.

Kazemzadeh, S., Ordonez, V., Matten, M., and Berg, T. Referitgame: Referring to objects in photographs of natural scenes. In Proceedings of the 2014 conference on empirical methods in natural language processing (EMNLP), pp. 787–798, 2014.

Lai, X., Tian, Z., Chen, Y., Li, Y., Yuan, Y., Liu, S., and Jia, J. Lisa: Reasoning segmentation via large language model. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9579– 9589, 2024.

Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al. Llavaonevision: Easy visual task transfer. Transactions on Machine Learning Research, 2025a.

Li, G., Xu, J., Zhao, Y., and Peng, Y. Dyfo: A trainingfree dynamic focus visual search for enhancing lmms in fine-grained visual understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 9098–9108, 2025b.

Li, J., Li, D., Savarese, S., and Hoi, S. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International conference on machine learning, pp. 19730–19742. PMLR, 2023a.

Li, Y., Du, Y., Zhou, K., Wang, J., Zhao, W. X., and Wen, J.-R. Evaluating object hallucination in large visionlanguage models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 292–305, 2023b.

Liu, D., Wang, Z., Ruan, M., Luo, F., Chen, C., Li, P., and Liu, Y. Visual abstract thinking empowers multimodal reasoning. arXiv preprint arXiv:2505.20164, 2025.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning. Advances in neural information processing systems, 36, 2024a.

Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al. Mmbench: Is your multi-modal model an all-around player? In European conference on computer vision, pp. 216–233. Springer, 2024b.

Mao, J., Huang, J., Toshev, A., Camburu, O., Yuille, A. L., and Murphy, K. Generation and comprehension of unambiguous object descriptions. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 11–20, 2016.

Neuhold, G., Ollmann, T., Rota Bulo, S., and Kontschieder, P. The mapillary vistas dataset for semantic understanding of street scenes. In Proceedings of the IEEE international conference on computer vision, pp. 4990–4999, 2017.

Ramanathan, V., Kalia, A., Petrovic, V., Wen, Y., Zheng, B., Guo, B., Wang, R., Marquez, A., Kovvuri, R., Kadian, A., et al. Paco: Parts and attributes of common objects. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 7141–7151, 2023.

Ravi, N., Gabeur, V., Hu, Y.-T., Hu, R., Ryali, C., Ma, T., Khedr, H., Radle, R., Rolland, C., Gustafson, L., et al.¨ Sam 2: Segment anything in images and videos. In The Thirteenth International Conference on Learning Representations, 2025.

Shao, H., Qian, S., Xiao, H., Song, G., Zong, Z., Wang, L., Liu, Y., and Li, H. Visual cot: Unleashing chainof-thought reasoning in multi-modal language models. CoRR, 2024.

Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., and Xie, S. Eyes wide shut? exploring the visual shortcomings of multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 9568–9578, 2024.

Wang, H., Qu, C., Huang, Z., Chu, W., Lin, F., and Chen, W. Vl-rethinker: Incentivizing self-reflection of visionlanguage models with reinforcement learning. arXiv preprint arXiv:2504.08837, 2025a.

Wang, H., Su, A., Ren, W., Lin, F., and Chen, W. Pixel reasoner: Incentivizing pixel-space reasoning with curiosity-driven reinforcement learning. arXiv preprint arXiv:2505.15966, 2025b.

Wang, W., Ding, L., Zeng, M., Zhou, X., Shen, L., Luo, Y., Yu, W., and Tao, D. Divide, conquer and combine: A training-free framework for high-resolution image perception in multimodal large language models. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 7907–7915, 2025c.

Wei, C., Tan, H., Zhong, Y., Yang, Y., and Ma, L. Lasagna: Language-based segmentation assistant for complex queries. arXiv preprint arXiv:2404.08506, 2024.

Wu, P. and Xie, S. V?: Guided visual search as a core mechanism in multimodal llms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13084–13094, 2024.

Xu, W., Wang, J., Wang, W., Chen, Z., Zhou, W., Yang, A., Lu, L., Li, H., Wang, X., Zhu, X., et al. Visulogic: A benchmark for evaluating visual reasoning in multi-modal large language models. arXiv preprint arXiv:2504.15279, 2025.

Yan, C., Wang, H., Yan, S., Jiang, X., Hu, Y., Kang, G., Xie, W., and Gavves, E. Visa: Reasoning video object segmentation via large language models. In European Conference on Computer Vision, pp. 98–115. Springer, 2024.

Yu, Q., Zhang, Z., Zhu, R., Yuan, Y., Zuo, X., Yue, Y., Dai, W., Fan, T., Liu, G., Liu, L., et al. Dapo: An open-source llm reinforcement learning system at scale. arXiv preprint arXiv:2503.14476, 2025.

Zhang, X., Gao, Z., Zhang, B., Li, P., Zhang, X., Liu, Y., Yuan, T., Wu, Y., Jia, Y., Zhu, S.-C., et al. Chain-of-focus: Adaptive visual search and zooming for multimodal reasoning via rl. arXiv preprint arXiv:2505.15436, 2025.

Zhang, Y., Ma, Z., Gao, X., Shakiah, S., Gao, Q., and Chai, J. Groundhog: Grounding large language models to holistic segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 14227–14238, 2024.

Zheng, Z., Yang, M., Hong, J., Zhao, C., Xu, G., Yang, L., Shen, C., and Yu, X. Deepeyes: Incentivizing” thinking with images” via reinforcement learning. arXiv preprint arXiv:2505.14362, 2025.

Zhou, B., Zhao, H., Puig, X., Fidler, S., Barriuso, A., and Torralba, A. Scene parsing through ade20k dataset. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 633–641, 2017.