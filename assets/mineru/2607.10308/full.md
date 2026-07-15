# Generalize LMMs to Versatile Visual Modalities via Fabricated Modality Synthesis

Shihao Yuan<sup>1</sup> , Yuanze Li<sup>1</sup> , Ruyi Zhang<sup>1</sup>, Ming Liu<sup>1()</sup> , and Wangmeng Zuo<sup>1</sup>

{csshihao, sqleopop, csmliu}@outlook.com, {ruyi.zhang.maggie, cswmzuo}@gmail.com

<sup>1</sup>Faculty of Computing, Harbin Institute of Technology, Harbin, China

Abstract. Despite the advancements of Large Multimodal Models (LMMs in RGB vision, their ability to generalize to unseen visual modalities remains a largely unexplored challenge. We argue that diferent visual modalities are merely distinct samplings of the same physical world. Therefore, efective generalization requires models to possess both modalityagnostic perception of scene semantics and the adaptability to modalityspecific characteristics. To achieve this, we propose a training framework, VVM-Tuning, to equip LMMs with these capabilities through modality synthesis and modality contexts. Specifically, we synthesize diverse appearance-varied images from RGB scenes, training the model to disentangle invariant semantics from varying visual appearances, and align these appearances with language for visual concepts decoupled from modalities. We then introduce modality contexts in the prompt and use instruction tuning to assist the model in mapping these appearance variations back to modality-related attributes, enabling zero-shot adaptation to unseen modalities during inference. To facilitate research in this direction, we introduce VVM-Bench, a comprehensive benchmark featuring 6 real and synthetic modalities to evaluate semantic perception and modality understanding. Experiments demonstrate that, via our training on synthetic modalities, 5 tested models exhibit consistent improvements on both real-world and novel synthetic modalities without in-modality training. Source code and data will be publicly available at https://github.com/Hunter-Will/VVM-Tuning.

Keywords: LMMs · Instruction Tuning · Synthetic Data

## 1 Introduction

Current Large Multimodal Models (LMMs) demonstrate impressive visual performance on RGB images [10]; however, the generalization boundary on other visual modalities (e.g. infrared, depth, etc.) remains largely unexplored yet. Existing non-RGB vision in current LMMs primarily depends on the non-RGB data incorporated during the data scaling process [1,4,33]. This leads to a generalization gap when encountering unseen modalities, as the non-RGB vision is built on in-modality data and limited modality coverage.

![](images/8d8db324b1bafc1b99676f81f793ac8abda9b1351270ed40d84f49aa79f3aa3c.jpg)  
Fig. 1: An overview of our idea. Diferent visual modalities (Thermal and RGB) are both signal samplings of our physical world, sharing the same underlying semantics (formed by the person and the car). They also share basic visual concepts, as the color itself is invariant across modalities. The modality-specific physical meaning is encoded by the rearrangement and remapping of basic visual concepts. Thus, we introduce textual context to complement such knowledge during inference.

To address this gap and explore the generalization boundary of LMMs, we raise the question: Is it possible for LMMs to generalize across versatile visual modalities without training on in-modality data?

The solution to the problem starts from the nature of visual modalities, as shown in Fig. 1, though with diferent appearances, they are essentially digital signals collected by sensors (thermal and RGB camera in Fig. 1) from the same physical space. Therefore, images from diferent visual modalities can be viewed as distinct samplings of identical real-world scenes. This commonality lies the foundation for pan-modal generalization, as the understanding of underlying semantics is modality-agnostic. Furthermore, through observation, we find that in human vision, images from other visual modalities can still be seen even without understanding the modality. Take Fig. 2 as an example, our brain can still read the semantics of the image from basic visual elements. This gives us inspiration that LMMs can learn this process to recognize semantics even without certain modality knowledge, which we call Modality-unaware Perception. Also, those visual elements, belonging to general visual concepts, are not completely combined with semantics. Some of them (like the colors in the thermal image in Fig. 1) might not be a necessary part for understanding the whole image semantics under this modality-unaware circumstance. Instead, they are supposed to combine with the physical meaning of the visual modality to achieve a higher understanding of the image, such as the color-coded thermal image in Fig. 1. Although they can only be understood superficially without modality knowledge, they are still basic elements of both human vision and machine vision, which are completely generalized among diverse visual modalities, as long as they are 3-channel digital images. That means they are still perceivable modality-unawarely, for example, the thermal color in Fig. 1 can still be understood as colors without knowing its connection with temperature. We refer to this part that only perceives basic visual elements without understanding further as photographic perception. Correspondingly, the part that comprehends underlying semantics is referred to as semantic perception. Both parts compose a modality-unaware perception without modality knowledge. This modality-unaware perception demonstrates a highly possible approach of unseen visual modality generalization; however, it doesn’t mean that every single visual modality can be understood in the same way. Diferent visual modalities have their unique sampling function of the real world, which is defined by the sensor and signal processing procedure. This often leads to varying appearances of the same underlying semantics and usually means a connection between the physical attributes of entities in the real-world scene and the unique visual appearances in digital images, which is, of course, diferent from regular RGB camera images. Besides, non-RGB modalities usually map an additional physical quantity to the color of the image, known as the pseudo-color [13,28], which actually enriches the meaning of basic visual elements. Understanding such a connection or mapping requires modality knowledge, and such modality knowledge is not reachable during training for the unseen visual modalities in our discussion. Thus, we introduce modality contexts to complement the modality understanding by utilizing the instruction following and in-context learning ability of LMMs during inference.

![](images/181256625c2f02a72bb5b5c44231fe536f73f4ad7516f22330fa351a961d5d4a.jpg)  
Fig. 2: An example of how human vision adapts to unseen visual modalities. Our eyes can capture basic visual elements regardless of modalities, and our brain recognizes the semantics even through a few visual elements. This demonstrates a compatible modality-unaware perception, which we could implement in LMMs’ training.

Viewing LMMs’ training from this perspective, we argue that the current LMMs are largely overlooking the modality-unaware perception. The visuallanguage alignment, building primarily on RGB images, doesn’t distinguish the invariant semantics and varying visual appearances. This would cause an overalignment and isolate other visual modalities, as the model needs to learn new alignments for each diferent visual modality incorporated in training. Based on the above insights, we proposed a novel training framework called VVM-tuning to: 1) Build up a modality-unaware perception via fabricated images; 2) Enable the modality-aware understanding of LMMs through modality synthesis.

For the first part, we leverage RGB images to fabricate non-RGB images to mimic varying appearances from diferent visual modalities, keeping the semantics invariant from RGB images. Therefore, the model has a chance to learn how to read invariant semantics through varying appearances. Furthermore, we design two sets of descriptions and VQA tasks to disentangle the semantic parts and perceptual parts of the visual-language alignments. By separate alignment, the model is taught to extract the semantics and keep perceptual elements aligned with language at the same time, which would benefit the modality-aware understanding for reconnecting the physical meanings and visual appearances. Because of the fabricated image, we can largely diversify the visual appearances of common scenes; thus, we establish a more general perception of visual appearances, which leads to a base for modality-aware understanding.

For the second part, we utilize modality synthesis to build a general synthetic instruction set to train the model with modality contexts. These modality contexts are fabricated to construct diverse synthesized visual modalities combined with the fabricated images. From these synthetic data, the model learns to connect the modality-specific physical attributes to varying visual appearances and further integrates with the semantics to achieve better modality understanding. With the combination of the above two parts, we enable the zero-shot generalization of LMMs to unseen visual modalities without in-modality data for training.

Finally, we assemble a diverse visual modality benchmark, VVM-bench, containing 6 real and synthesized imaging modalities. From our perspective, this benchmark evaluates the generalization ability through 11 VQA tasks from modality-unaware perception and modality-aware understanding, respectively. Half of the 6 non-RGB visual modalities are collected from VS-TDX [7], and 3 fabricated modalities to mimic unseen visual modalities. Experiments show an average of 8.2% improvements of perception tasks and 3.8% improvements of understanding tasks on this benchmark among 5 diferent base models and in 6 synthesized and real visual modalities for both perception and understanding.

In summary, our contribution can be listed as follows:

– To our best knowledge, we are the first to explore the potential of LMMs to generalize to unseen visual modalities, and demonstrate a possible approach through modality context and modality synthesis.

– We introduce synthetic data for other visual modalities and propose a fabricated modality synthesis data pipeline, which can produce diverse fabricated images and synthesized modality contexts.

– We propose a training framework establishing modality-unaware perception and modality-aware understanding, and confirm that LMMs can generalize to other visual modalities via modality synthesis and disentangled tasks.

– We assemble a versatile visual modality benchmark to evaluate the visual ability and generalization on visual modalities for LMMs.

## 2 Related Works

Since the emergence of LLaVA [19], establishing the visual instruction tuning paradigm, both LMMs and benchmarks targeting their RGB visual capabilities have seen significant development [10]. However, the generalization boundary of LMMs to non-RGB visual modalities remains underexplored. Modern LMMs, such as LLaVA-OneVision [1, 16], Qwen-VL [3, 4, 17], and InternVL [6, 33, 41], incorporate diverse and abundant visual data, which may contain other modalities in their training subsets like DocQA [24] and ScienceQA [23]. However, their primary focus remains on scaling data and diversity to improve general visual abilities, rather than investigating the generalization of unseen visual modality.

Meanwhile, some previous works, such as Imagebind [12] and PandaGPT [30], also involve thermal and depth images. However, the basic idea of Imagebind [12] is “binding” other general modalities (audio, text, etc.) to RGB images in order to eliminate the need for pairwise alignments between these modalities. As a result, RGB-D and RGB-T paired data are still required during the training of Imagebind [12], which are not regarded as unseen visual modality and not within our scope of discussion. Similarly, the cross-modal capabilities discussed in PandaGPT [30] refer to the zero-shot instruction following ability among 6 seen modalities inherited from Imagebind [12], which is also beyond the discussion. Concurrently, research dedicated to non-RGB modalities has largely been confined to fine-tuning LMMs on specialized datasets. For instance, Infrared-LLaVA [14] adapts LLaVA for infrared-specific tasks by tailoring the model to the infrared modality. SpatialBot [5] focuses on the spatial understanding with both RGB and depth images, also relying on RGB-D training to inject depth knowledge. Specialized LMMs, for instance, EarthGPT [38] in remote sensing, XrayGPT [32] and RadVLM [8] in radiology, are trained with domain-specific data. Though they also involve multiple visual modalities, they are focusing on application tasks, leaving their inherent ability to generalize across diferent useen visual modalities behind.

This gap is further reflected in existing evaluation suites. Mainstream benchmarks for general-purpose LMMs, such as MMBench [20], MME [9], MMMU [37], MM-Vet [35], and SEED-Bench [18], are predominantly centered on RGB images. While a few recent eforts have extended evaluation to non-RGB modalities for LMMs, the RGB-Th-Bench [26] focuses on the cross-reference understanding of RGB-T paired images, ignoring the possible generalization to unseen modalities on general visual abilities. The VS-TDX [7] benchmark evaluates general visual abilities on 3 modalities, but still lacks focus on the principles and testing of unseen visual modality generalization. Other specialized evaluations, like MedSG-Bench [36] in medical imaging and RSVQA [21] in remote sensing, are highly domain-specific and not designed to assess general visual adaptability. Consequently, there remains a lack of benchmarks tailored to systematically evaluate the cross-visual-modal generalization capabilities of LMMs. These limitations underscore a critical gap in understanding the true visual versatility of modern LMMs beyond the RGB vision, motivating our study to investigate their generalization potential across diferent visual modalities.

![](images/5ac0826b7429102007c86bd999943e188e08bcf516bbb97ce084e202581d89b6.jpg)  
Fig. 3: An overview of our modality synthesis pipeline, composed of the image synthesis and modality context fabrication.

## 3 Methods

## 3.1 Preliminary

From the base idea of how diferent visual modalities follow the basic imaging principles to map the real-world scenarios into photos, our method tries to imitate the same process of how humans understand versatile visual modalities, which leads to two diferent abilities: modality-unaware perception and modalityaware understanding. To equip LMMs with these two abilities, we design two sets of generated instruction tasks [27] and train the model to learn the adaptability from these tasks with fabricated images and modalities. Our VVM-Tuning mainly follows the paradigm of visual instruction tuning [19]. The training objective follows the common approach of Supervised Fine-Tuning (SFT), which involves the Cross Entropy loss function $\mathcal { L } _ { S F T }$

$$
\mathcal {L} _ {S F T} = - \sum_ {i = 1} ^ {n} \log \pi_ {\theta} (y _ {i} | X, y _ {1}, y _ {2},..., y _ {i - 1}),\tag{1}
$$

The whole formalized expression of the training is to optimize $\mathcal { L } _ { S F T } ( T _ { a } , T _ { g t } )$ to minimize the diferences between the generated $T _ { a }$ and the ground truth $T _ { g t }$

## 3.2 Modality Synthesis

Efective generalization requires cross-modal perception and context-guided modality adaptation, which needs the training data covering versatile modal characteristics. Thus, we introduce a visual modality synthesis pipeline, shown in Fig. 3. The pipeline is mainly composed of two parts: 1) image synthesis and 2) modality context fabrication.

The first part is to imitate the unique characteristics that difer from regular RGB photos, such as the pseudo-color pattern [39] and histogram distribution. Here, we introduce a simple pipeline to fabricate non-RGB images from RGB images. As shown in Fig. 3, the RGB source images are first converted to grayscale for histogram adjustments. We apply cubic spline interpolation [25] to randomize the shape of the histogram [29] and increase the diversity by flipping the whole histogram randomly. Then the grayscale images are painted with pseudo-color by randomly applying 22 color maps [40] that are commonly used in scientific computing and visualization. After color mapping, several image augmentations [34], such as noise injection, smoothing, and blurring, are employed to simulate the noisy and low-resolution modalities. With such a simple pipeline, we manage to fabricate diverse images with similar characteristics to the real non-RGB visual modalities, shown by the examples in Fig. 4.

Besides fabricated images, we also make up corresponding physical meanings and explanations to compose modality contexts. This is done by two approaches according to whether there are structure annotations (segmentation mask or object information) of source RGB images, shown in Fig. 3. For the images with annotations, the modality contexts are assembled structure texts by connecting the color maps with objects in the image by labels such as segmentation masks. Then, the value interval mapping colors is ascribed with human-made physical attributes and fit into manual templates to form modality contexts. For the nonannotated images, we leverage the in-context learning ability of LMM to generate color and object-related modality context by prompting it to follow manually written examples. Combining fabricated images and synthesized modality contexts results in modality synthesis that contains diverse characteristics to teach LMMs the varying and invariant features of versatile visual modalities. Further details of the modality synthesis pipeline can be found in the Appendix.

## 3.3 Modality-unaware Perception

The foundation of generalizing to unseen visual modalities is an invariant perception ability across them. This ability is called modality-unaware perception, which contains two parts: 1) the underlying semantic perception and 2) the photographic perception of basic visual elements. In regular RGB visual-language alignment, these two parts of perception are usually entangled together, aligned with the language as a whole. However, when other visual modalities are involved, the entangled perception isolates the shared semantics between visual modalities due to diferent appearances. This causes the LMMs need to learn new alignments for each diferent visual modality. As the underlying semantics are encoded in diferent forms of expression by diferent visual modalities, we first design scene semantic tasks to equip LMM with the ability to extract the underlying semantics. Then, photographic perception tasks are involved to capture unique visual clues introduced by diferent visual modalities. With the image synthesis pipeline, the fabricated images are convenient to create the same semantics with diferent visual appearances. Based on them, we design several instructions to train the model to recognize the physical entities in various appearances and activate the common knowledge with such entities. Furthermore, the photographic perception tasks ensure the LMM captures the visual details and remains loyal to the image instead of being confused by semantics.

![](images/4facebe9d15b7160b9dc61dbbddbc690991e16303b94cad16e7c2a044e2bf6ab.jpg)  
Fig. 4: Diferent examples of fabricated images with diferent color maps produced by our image synthesis pipeline.

![](images/edaf0ec826773746abbbd91110d16f12685a2e51558a8aef25efa577c92b1327.jpg)  
Fig. 5: An overview of the task construction of VVM-Tuning, including modalityunaware perception tasks and modality-aware understanding tasks. Starting from our core insight at the top, the region illustrates the construction of the semantic description and photographic description tasks to form modality-aware descriptions. Then, the surrounding VQA tasks are generated from task templates with descriptions.

For better adaptability, both semantic perception and photographic perception are composed of two forms of tasks, image description [15] and visual question-answering (VQA) [2, 19]. As shown in the left part of Fig. 5, they form four tasks: semantic description, semantic QA, photographic description, and photographic QA. First, for the semantic description, we leverage the common semantics of fabricated images and prompt an LMM to extract the overlapped parts from the description of the RGB image according to the RGB image and the corresponding fabricated image. That leads to a semantic description that extracts the shared semantics of fabricated and RGB images. Then, the photographic description is generated by prompting the LMM to extract the unique parts from the fabricated image that are not in the semantic description. However, due to the lack of perception of these fabricated images, the colors in the photographic description need to be recorrected by introducing the color names during the image synthesis pipeline. As for the VQA instructions, we design 11 diferent sub-tasks for semantic and photographic QA as human-made question templates and utilize a large-sized LMM to generate diverse questions and answers according to RGB images, fabricated images, and corresponding descriptions, respectively. Finally, the questions are randomly organized as a mix of non-choice questions and choice questions to combine with the descriptions to create a completely modality-unaware training set. Further details of the task distribution and templates can be found in the Appendix.

## 3.4 Modality-aware Understanding

The purpose of this part of VVM-Tuning is to enhance the understanding of modality contexts. Modality context is a textual prompt, including specific characteristics and physical meanings of the modality of the image. The two ways of synthesizing modality context are described in Sec. 3.2. The modality-aware understanding tasks leverage synthesized modality contexts to equip the model with the ability to understand modality contexts and connect the physical meaning to specific visual concepts or appearances. Same as the modality-unaware perception, the modality-aware training also consists of image description and VQA tasks, called modality description and modality QA, shown in the right part of Fig. 5. The construction of modality description is achieved by leveraging a large language model to combine the semantic description, photographic description, and synthesized modality context together to form an image description with modality knowledge. The modality QA is generated by LMM according to the modality context and human-made templates. We don’t use the modality description here as the VQA tasks are concentrated to particular part of the image and the modality description only provides a general description of the entire image. However, scene and photographic descriptions are involved to make sure the LMM for generation understands the non-RGB images enough. The question templates of modality QA are specially designed to mainly cover diferent circumstances that require a knowledge of the visual modality. Further details can be found in the Appendix.

## 3.5 Benchmark

To mitigate the lack of evaluation data and confirm the feasibility of our training framework, we assembled a benchmark that contains multiple real and synthesized visual modalities to test the perception and understanding abilities of LMMs. The VVM-Bench, as shown in Fig. 6, is composed of 6 visual modalities and 11 VQA tasks belonging to perception and understanding, respectively. Among them, three modalities (thermal, depth, and X-ray) are collected from VS-TDX [7] with minimal changes of output formation to insert modality contexts in understanding tasks. The other three modalities are fabricated from

![](images/7487713c6a1362835edeced7ea28555f55bb54ed2ffc48ddbc4e7d7ec6c898e8.jpg)  
Fig. 6: An overview of VVM-Bench. Our VVM-Bench contains 6 visual modalities and 11 VQA tasks for perception and understanding evaluation, including 3 real modalities collected from previous work and 3 fabricated from RGB images.

RGB images, including Optical Flow, Synthesized 1, and Synthesized 2. Specifically, the optical flow images are visualizations of predicted optical flows from RAFT [31] to serve as signals from an almost entirely new and unknown imaging sensor with real physical meanings. The other two visual modalities are completely synthesized as unseen visual modalities. The images in Synthesized 1 are fabricated using a style transfer algorithm, StyleShot [11], to apply a visualization style on normal RGB images. As for images for Synthesized 2, we use the same image synthesis method as described in Sec. 3.2 with changed hyperparameters to explore the generative diversity of our image synthesis pipeline.

Our benchmark evaluates two diferent aspects of LMMs corresponding to our insights, the modality-unaware perception and modality-aware understanding, abbreviated as perception and understanding in Fig. 6. The perception aspect contains 7 sub-tasks, and all tasks focus on the semantics or visual attributes perceivable without modality knowledge. No modality contexts are provided in these tasks. Instead, the understanding tasks are all about the modality understanding or knowledge association with the visual modality, which provides modality contexts in the prompts. All modality contexts are written by human and we use choice questions in VVM-bench for the convenience of evaluation. Detailed task templates and modality contexts can be found in the Appendix.

## 4 Experiments

## 4.1 Implementation Details

We have employed the training framework on multiple base LMMs, including LLaVA-1.5 [19], Qwen2.5-VL [17], and Qwen3-VL [4] with diferent sizes in our capabilities. All the experiments are conducted on a 4 NVIDIA A800 GPU server with a total batch size of 32. Following the common setting of Supervised Fine-Tuning (SFT), we apply the AdamW [22] as optimizer and set the learning rate to 1e − 5. We adopt a single-staged fine-tuning with roughly 50k training samples in total for 1 epoch, involving all tasks in modality-unaware perception and modality-aware understanding. To preserve the general abilities of LMMs, we retain 12k RGB visual instruction data in the 50k training dataset.

Our benchmark follows the common VQA benchmark setting, with the form of choice questions. To quantify the results, the LMMs are instructed to output the option’s letter only to calculate accuracy. Specifically, for tasks that need modality contexts, all modality contexts are inserted between the image and the question as contexts. Further evaluation details can be found in the Appendix.

## 4.2 Results of Modality-unaware Perception

The results in Tab. 1 show the VQA accuracy on the perception subset of VVM-Bench. The results in Tab. 1 report the total accuracy of all samples in each modality for both baselines and our tuned models. From the results in Tab. 1, experiments show our VVM-tuning brought an average 8.2% improvement for all base models among 6 modalities on our VVM-Bench perception subset. Across all tested models, LLaVA-1.5 gained the biggest average improvement of 13.6% because of its relatively low base. The 4B model of Qwen-3-VL got the lowest improvement, but still exceeded 5%. This exhibits the adaptability of our modalityunaware perception training among diferent base models and confirms that the perception of LMMs on unseen visual modalities still has room for improvement.

Table 1: The results of the Modality-unaware Perception subset of VVM-bench. “∆” represents accuracy diference between the model we tuned and the baselines. “Average” shows the average accuracy and diference across all tested modalities. “Average ∆” shows the average improvements across all tested models for each visual modality. Note that there is no modality context for this perception subset.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="3">Collected Visual Modalities</td><td colspan="3">Fabricated Visual Modalities</td><td rowspan="2">Average</td></tr><tr><td colspan="2">Thermal Depth</td><td>X-Ray</td><td colspan="3">Optical Flow Synthesized 1 Synthesized 2</td></tr><tr><td>LLaVA-1.5</td><td>7B</td><td>55.6%</td><td>62.4%</td><td>58.1%</td><td>53.2%</td><td>70.7%</td><td>62.6%</td><td>60.4%</td></tr><tr><td>LLaVA-1.5 (Ours)</td><td>7B</td><td>69.1%</td><td>69.8%</td><td>67.3%</td><td>75.4%</td><td>85.8%</td><td>77.0%</td><td>74.1%</td></tr><tr><td> $\Delta$ </td><td>-</td><td>+13.5%</td><td>+7.4%</td><td>+9.2%</td><td>+22.2%</td><td>+15.1%</td><td>+14.4%</td><td>+13.6%</td></tr><tr><td>Qwen-2.5-VL</td><td>3B</td><td>72.0%</td><td>67.9%</td><td>68.1%</td><td>67.2%</td><td>83.0%</td><td>74.9%</td><td>72.2%</td></tr><tr><td>Qwen-2.5-VL (Ours)</td><td>3B</td><td>81.7%</td><td>77.1%</td><td>77.5%</td><td>83.1%</td><td>88.9%</td><td>83.0%</td><td>81.9%</td></tr><tr><td> $\Delta$ </td><td>-</td><td>+9.7%</td><td>+9.2%</td><td>+9.4%</td><td>+15.9%</td><td>+5.9%</td><td>+8.1%</td><td>+9.7%</td></tr><tr><td>Qwen-2.5-VL</td><td>7B</td><td>74.5%</td><td>73.3%</td><td>75.7%</td><td>77.0%</td><td>87.1%</td><td>81.1%</td><td>78.1%</td></tr><tr><td>Qwen-2.5-VL (Ours)</td><td>7B</td><td>79.8%</td><td>79.2%</td><td>79.6%</td><td>88.4%</td><td>91.2%</td><td>86.4%</td><td>84.1%</td></tr><tr><td> $\Delta$ </td><td>-</td><td>+5.3%</td><td>+5.9%</td><td>+3.9%</td><td>+11.4%</td><td>+4.1%</td><td>+5.3%</td><td>+6.0%</td></tr><tr><td>Qwen-3-VL</td><td>4B</td><td>80.6%</td><td>77.8%</td><td>76.7%</td><td>73.0%</td><td>81.6%</td><td>75.6%</td><td>77.6%</td></tr><tr><td>Qwen-3-VL (Ours)</td><td>4B</td><td>82.6%</td><td>84.4%</td><td>79.5%</td><td>81.5%</td><td>86.9%</td><td>82.1%</td><td>82.8%</td></tr><tr><td> $\Delta$ </td><td>-</td><td>+2.0%</td><td>+6.6%</td><td>+2.8%</td><td>+8.5%</td><td>+5.3%</td><td>+6.5%</td><td>+5.3%</td></tr><tr><td>Qwen-3-VL</td><td>8B</td><td>79.8%</td><td>78.5%</td><td>73.8%</td><td>73.3%</td><td>82.4%</td><td>78.0%</td><td>77.6%</td></tr><tr><td>Qwen-3-VL (Ours)</td><td>8B</td><td>82.1%</td><td>84.2%</td><td>80.3%</td><td>86.5%</td><td>89.5%</td><td>82.1%</td><td>84.1%</td></tr><tr><td> $\Delta$ </td><td>-</td><td>+2.3%</td><td>+5.7%</td><td>+6.5%</td><td>+13.2%</td><td>+7.1%</td><td>+4.1%</td><td>+6.5%</td></tr><tr><td>Average  $\Delta$ </td><td>-</td><td>+6.6%</td><td>+7.0%</td><td>+6.4%</td><td>+14.2%</td><td>+7.5%</td><td>+7.7%</td><td>+8.2%</td></tr></table>

Table 2: The results of the Modality-aware Understanding subset of VVM-bench with modality contexts. “∆” represents accuracy diference between the model we tuned and the baselines. “Average” shows the average accuracy and diference across all tested modalities. “Average ∆” shows the average improvements across all tested models for each visual modality.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="3">Collected Visual Modalities</td><td colspan="3">Fabricated Visual Modalities</td><td rowspan="2">Average</td></tr><tr><td>Thermal</td><td>Depth</td><td>X-Ray</td><td colspan="3">Optical Flow Synthesized 1 Synthesized 2</td></tr><tr><td>LLaVA-1.5</td><td>7b</td><td>77.8%</td><td>80.3%</td><td>72.4%</td><td>76.7%</td><td>84.1%</td><td>84.8%</td><td>79.4%</td></tr><tr><td>LLaVA-1.5 (Ours)</td><td>7b</td><td>82.5%</td><td>81.2%</td><td>79.3%</td><td>89.2%</td><td>87.6%</td><td>89.8%</td><td>84.9%</td></tr><tr><td>Δ</td><td>-</td><td>+4.7%</td><td>+0.9%</td><td>+6.9%</td><td>+12.5%</td><td>+3.5%</td><td>+5.0%</td><td>+5.6%</td></tr><tr><td>Qwen-2.5-VL</td><td>3b</td><td>83.8%</td><td>85.5%</td><td>83.5%</td><td>89.6%</td><td>86.9%</td><td>91.0%</td><td>86.7%</td></tr><tr><td>Qwen-2.5-VL (Ours)</td><td>3b</td><td>89.3%</td><td>89.5%</td><td>85.9%</td><td>93.8%</td><td>91.0%</td><td>93.8%</td><td>90.6%</td></tr><tr><td>Δ</td><td>-</td><td>+5.5%</td><td>+4.0%</td><td>+2.4%</td><td>+4.2%</td><td>+4.1%</td><td>+2.8%</td><td>+3.8%</td></tr><tr><td>Qwen-2.5-VL</td><td>7b</td><td>86.4%</td><td>83.4%</td><td>84.1%</td><td>84.6%</td><td>88.8%</td><td>89.8%</td><td>86.2%</td></tr><tr><td>Qwen-2.5-VL (Ours)</td><td>7b</td><td>88.5%</td><td>90.2%</td><td>85.9%</td><td>93.8%</td><td>91.9%</td><td>93.5%</td><td>90.6%</td></tr><tr><td>Δ</td><td>-</td><td>+2.1%</td><td>+6.8%</td><td>+1.8%</td><td>+9.2%</td><td>+3.1%</td><td>+3.7%</td><td>+4.5%</td></tr><tr><td>Qwen-3-VL</td><td>4b</td><td>87.3%</td><td>83.8%</td><td>84.3%</td><td>90.0%</td><td>90.0%</td><td>91.8%</td><td>87.9%</td></tr><tr><td>Qwen-3-VL (Ours)</td><td>4b</td><td>89.6%</td><td>87.0%</td><td>88.0%</td><td>92.1%</td><td>90.5%</td><td>92.0%</td><td>89.9%</td></tr><tr><td>Δ</td><td>-</td><td>+2.3%</td><td>+3.2%</td><td>+3.7%</td><td>+2.1%</td><td>+0.5%</td><td>+0.2%</td><td>+2.0%</td></tr><tr><td>Qwen-3-VL</td><td>8b</td><td>89.3%</td><td>87.2%</td><td>83.1%</td><td>90.8%</td><td>87.9%</td><td>91.5%</td><td>88.3%</td></tr><tr><td>Qwen-3-VL (Ours)</td><td>8b</td><td>91.6%</td><td>91.5%</td><td>89.8%</td><td>91.7%</td><td>90.3%</td><td>93.8%</td><td>91.5%</td></tr><tr><td>Δ</td><td>-</td><td>+2.3%</td><td>+4.3%</td><td>+6.7%</td><td>+0.9%</td><td>+2.4%</td><td>+2.3%</td><td>+3.2%</td></tr><tr><td>Average Δ</td><td>-</td><td>+3.4%</td><td>+3.8%</td><td>+4.3%</td><td>+5.8%</td><td>+2.7%</td><td>+2.8%</td><td>+3.8%</td></tr></table>

Meanwhile, although training on synthetic data, all base models show at least an average 6% improvement on visual modalities with real physical meanings, as shown in Thermal, Depth, X-ray, and Optical Flow. This confirms the existence of the generalization foundation, and diferent real visual modalities do share commonalities in both semantics and perceptual elements. Also, the performance on synthesized modalities indicates that this improvement could generalize to completely unseen visual modalities and further confirm the efectiveness of our simple image synthesis pipeline.

## 4.3 Results of Modality-aware Understanding

From the results in Tab. 2, we can observe further improvements on modality understanding tasks. These tasks focus on understanding the modality context and linking the modality knowledge and common sense, which is more associated with the language model in LMM. As shown in Tab. 2, our VVM-tuning enhanced the ability to utilize modality context and reinterprete the perceptual features by a 3.8% average improvement across all models and modalities. Similar to before, among all baselines, LLaVA-1.5 has the greatest improvement of 5.6%, and the 4B Qwen-3-VL has the smallest improvement of 2.0%. This result further confirms the possibility of generalization across diverse unseen visual modalities. Tested LMMs demonstrate strong potential for learning from synthesized modalities and adapting to the other unseen visual modalities.

Among the evaluated modalities, the Optical Flow fabricated by us is the most improved, and the other three real modalities are next to it; our synthesized modalities get the lowest improvements. This shows that a synthetic gap still exists between made-up physical meaning and realistic sensing from the physical world. However, this gap does not reflect in the perception subset, indicating that modality meanings are more specific and harder to imitate than their visual appearances. Nevertheless, the 2.7%-5.8% improvements still support that the model can learn from synthesized modality contexts to generalize to real visual modalities and keep the potential for completely unseen visual modalities.

Table 3: The results of 3 real visual modalities in the understanding subset with/out modality contexts. “M.C.” is an abbreviation for Modality Context. “∆” represents the accuracy diference with or without modality context for each model and modality. “Average ∆” demonstrates the average improvements of providing modality contexts for each modality across all models.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Size</td><td colspan="3">Thermal</td><td colspan="3">Depth</td><td colspan="3">X-Ray</td></tr><tr><td>w/o M.C.</td><td>w/ M.C.</td><td> $\Delta$ </td><td>w/o M.C.</td><td>w/ M.C.</td><td> $\Delta$ </td><td>w/o M.C.</td><td>w/ M.C.</td><td> $\Delta$ </td></tr><tr><td>LLaVA-1.5</td><td>7b</td><td>47.7%</td><td>77.8%</td><td>+30.2%</td><td>40.7%</td><td>80.3%</td><td>+39.6%</td><td>61.0%</td><td>72.4%</td><td>+11.5%</td></tr><tr><td>LLaVA-1.5 (Ours)</td><td>7b</td><td>63.1%</td><td>82.5%</td><td>+19.4%</td><td>43.5%</td><td>81.2%</td><td>+37.6%</td><td>73.0%</td><td>79.3%</td><td>+6.3%</td></tr><tr><td>Qwen-2.5-VL</td><td>3b</td><td>76.0%</td><td>83.8%</td><td>+7.8%</td><td>54.4%</td><td>85.5%</td><td>+31.0%</td><td>79.7%</td><td>83.5%</td><td>+3.8%</td></tr><tr><td>Qwen-2.5-VL (Ours)</td><td>3b</td><td>79.9%</td><td>89.3%</td><td>+9.5%</td><td>64.7%</td><td>89.5%</td><td>+24.8%</td><td>84.4%</td><td>85.9%</td><td>+1.5%</td></tr><tr><td>Qwen-2.5-VL</td><td>7b</td><td>69.6%</td><td>86.4%</td><td>+16.8%</td><td>46.3%</td><td>83.4%</td><td>+37.1%</td><td>82.9%</td><td>84.1%</td><td>+1.2%</td></tr><tr><td>Qwen-2.5-VL (Ours)</td><td>7b</td><td>74.5%</td><td>88.5%</td><td>+13.9%</td><td>54.3%</td><td>90.2%</td><td>+35.9%</td><td>84.1%</td><td>85.9%</td><td>+1.8%</td></tr><tr><td>Qwen-3-VL</td><td>4b</td><td>77.5%</td><td>87.3%</td><td>+9.8%</td><td>47.9%</td><td>83.8%</td><td>+35.9%</td><td>83.4%</td><td>84.3%</td><td>+0.9%</td></tr><tr><td>Qwen-3-VL (Ours)</td><td>4b</td><td>73.8%</td><td>89.6%</td><td>+15.8%</td><td>55.9%</td><td>87.0%</td><td>+31.1%</td><td>86.9%</td><td>88.0%</td><td>+1.1%</td></tr><tr><td>Qwen-3-VL</td><td>8b</td><td>71.4%</td><td>89.3%</td><td>+17.9%</td><td>53.8%</td><td>87.2%</td><td>+33.4%</td><td>82.3%</td><td>83.1%</td><td>+0.8%</td></tr><tr><td>Qwen-3-VL (Ours)</td><td>8b</td><td>79.1%</td><td>91.6%</td><td>+12.6%</td><td>70.4%</td><td>91.5%</td><td>+21.1%</td><td>88.2%</td><td>89.8%</td><td>+1.6%</td></tr><tr><td>Average  $\Delta$ </td><td>-</td><td>-</td><td>-</td><td>+15.37%</td><td>-</td><td>-</td><td>+32.75%</td><td>-</td><td>-</td><td>+3.05%</td></tr></table>

## 4.4 Ablation Study

We also conduct ablation experiments to further observe the efectiveness of our training strategy. The first ablation demonstrates the efect of introducing modality context on real visual modalities. The results in Tab. 3 indicate that the modality context is important for modality understanding when encountering unseen modalities, as removing modality context leads to a 32.75% decrease at most. However, as the data scales of modern LMMs are continuously expanding, common non-RGB modalities like Thermal and X-ray are inevitably involved in their knowledge base. Thus, we observe a rising trend as the model size and date increase in Thermal and X-ray compared to Depth, even without modality context. Also, the results confirm that our training does not inject modality knowledge during fine-tuning; instead, improved modality-unaware perception brought an increase in modality understanding even without modality context.

Further ablation provided a result about the efectiveness of two aspects of VVM-tuning. In Tab. 4, we conduct an ablation study on our tuned 7B Qwen-2.5-VL model to explore the influence of modality-unaware perception tasks and modality-aware understanding tasks. As shown in Tab. 4, both parts are necessary for complete modality understanding. The modality-unaware perception tasks solely brought roughly half of the total improvement on both perception and understanding subsets. And the modality-aware understanding training further enhances the improvements on the basis of modality-unaware perception.

Table 4: Ablation results for the modality-unaware perception and modality-aware understanding in VVM-Tuning on Qwen 2.5-VL 7B model. “Perception” and “Understanding” represent modality-unaware perception tasks and modality-aware understanding tasks in VVM-Tuning, respectively. Instead, ${ } ^ { 6 6 } \mathrm { P } ^ { 9 }$ and “U” represent the perception subset and understanding subset in VVM-Bench for evaluation. “Average” shows an average accuracy across all visual modalities. $" \mathbf { \nabla } _ { \mathbf { X } } , , . . .$ and $\mathfrak { s } _ { \pmb { \nu } }$ indicate whether the corresponding task set is used or not used during training.

<table><tr><td rowspan="2" colspan="2">Perception Understanding</td><td rowspan="2">Subset</td><td colspan="3">Collected Visual Modalities</td><td colspan="3">Fabricated Visual Modalities</td><td rowspan="2">Average</td></tr><tr><td>Thermal Depth</td><td>X-Ray</td><td colspan="4">Optical Flow Synthesized 1 Synthesized 2</td></tr><tr><td rowspan="2">✘</td><td rowspan="2">✘</td><td>P</td><td>74.5%</td><td>73.3%</td><td>75.7%</td><td>77.0%</td><td>87.1%</td><td>81.1%</td><td>78.1%</td></tr><tr><td>U</td><td>86.4%</td><td>83.4%</td><td>84.1%</td><td>84.6%</td><td>88.8%</td><td>89.8%</td><td>86.2%</td></tr><tr><td rowspan="2">✓</td><td rowspan="2">✘</td><td>P</td><td>75.3%</td><td>75.6%</td><td>79.6%</td><td>81.7%</td><td>90.7%</td><td>87.6%</td><td>81.7%</td></tr><tr><td>U</td><td>86.7%</td><td>82.5%</td><td>86.3%</td><td>92.9%</td><td>90.3%</td><td>91.3%</td><td>88.3%</td></tr><tr><td rowspan="2">✘</td><td rowspan="2">✓</td><td>P</td><td>76.8%</td><td>74.1%</td><td>75.4%</td><td>74.1%</td><td>85.8%</td><td>81.7%</td><td>78.0%</td></tr><tr><td>U</td><td>87.6%</td><td>85.4%</td><td>85.1%</td><td>92.1%</td><td>88.8%</td><td>90.3%</td><td>88.2%</td></tr><tr><td rowspan="2">✓</td><td rowspan="2">✓</td><td>P</td><td>79.8%</td><td>79.2%</td><td>79.6%</td><td>88.4%</td><td>91.2%</td><td>86.4%</td><td>84.1%</td></tr><tr><td>U</td><td>88.5%</td><td>90.2%</td><td>85.9%</td><td>93.8%</td><td>91.9%</td><td>93.5%</td><td>90.6%</td></tr></table>

However, without modality-unaware tasks, the perception base is too weak to learn the adaptability of modality understanding for unseen visual modalities. Therefore, the performance nearly drops back to baseline.

## 5 Conclusion

Through our exploration of unseen visual modality generalization of LMMs, we have demonstrated that the disentangled alignments of semantics and photographic perception are efective as a generalization foundation. Via fabricating images, the modality-unaware tasks equip the model with unified perception across multiple real and synthesized visual modalities. Furthermore, we introduce modality context in the prompt and enable the adaptation to unseen visual modalities through modality synthesis. Our experiments have shown that the training framework is efective on multiple base models and helps them improve 8.2% and 3.8% performance on perception and understanding respectively without in-modality data. Therefore, the results confirmed that generalizing among versatile unseen visual modalities is possible for LMMs with enough perception and modality contexts. We also propose a benchmark to evaluate LMMs’ general abilities on diverse visual modalities. We hope our eforts could facilitate the research on the visual modality adaptation of LMMs.

This attempt also gives us a new view of understanding the underlying mechanism of LMMs’ vision. By distinguishing the diferences and commonalities between visual modalities, we have the idea of separating shared semantics and photographic elements, which leads to a rethinking of machine vision: Do LMMs understand the diference between the image and the real world behind it? Or is its “real world” just the image? We wish this could inspire others to a deeper understanding of LMMs’ visual abilities and continue to explore the boundaries of LMMs’ vision.

## Acknowledgements

This work was supported by the National Key R&D Program of China under Grant No. 2022YFA1004100 and the National Natural Science Foundation of China (NSFC) under Grant No. 62501191.

Besides, we express our gratitude to the anonymous reviewers for their invaluable suggestions and comments. We would like to thank all members of the ILL Lab for their helpful discussions and the computing resources supported by the ILL Lab, Faculty of Computing, HIT. Also, thanks to Sangyun Chung and other authors of VS-TDX for the accessibility of Non-RGB data.

## References

1. An, X., Xie, Y., Yang, K., Zhang, W., Zhao, X., Cheng, Z., Wang, Y., Xu, S., Chen, C., Zhu, D., et al.: LLaVA-OneVision-1.5: Fully open framework for democratized multimodal training. arXiv preprint arXiv:2509.23661 (2025)

2. Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C.L., Parikh, D.: VQA: Visual Question Answering. In: Proceedings of the IEEE International Conference on Computer Vision. pp. 2425–2433 (2015)

3. Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., Zhou, J.: Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond. arXiv preprint arXiv:2308.12966 (2023)

4. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., et al.: Qwen3-VL Technical Report. arXiv preprint arXiv:2511.21631 (2025)

5. Cai, W., Ponomarenko, I., Yuan, J., Li, X., Yang, W., Dong, H., Zhao, B.: Spatial-Bot: Precise Spatial Understanding with Vision Language Models. In: 2025 IEEE International Conference on Robotics and Automation (ICRA). pp. 9490–9498. IEEE (2025)

6. Chen, Z., Wu, J., Wang, W., Su, W., Chen, G., Xing, S., Zhong, M., Zhang, Q., Zhu, X., Lu, L., et al.: InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 24185–24198 (2024)

7. Chung, S., Yu, Y., Kim, S.Y., Chee, Y., Ro, Y.M.: Enhanced Vision-Language Models for Diverse Sensor Understanding: Cost-Eficient Optimization and Benchmarking. arXiv preprint arXiv:2412.20750 (2024)

8. Deperrois, N., Matsuo, H., Ruipérez-Campillo, S., Vandenhirtz, M., Laguna, S., Ryser, A., Fujimoto, K., Nishio, M., Sutter, T.M., Vogt, J.E., et al.: RadVLM: A Multitask Conversational Vision-Language Model for Radiology. arXiv preprint arXiv:2502.03333 (2025)

9. Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al.: MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models. arXiv preprint arXiv:2306.13394 (2023)

10. Fu, C., Zhang, Y.F., Yin, S., Li, B., Fang, X., Zhao, S., Duan, H., Sun, X., Liu, Z., Wang, L., et al.: MME-Survey: A Comprehensive Survey on Evaluation of Multimodal LLMs. arXiv preprint arXiv:2411.15296 (2024)

11. Gao, J., Sun, Y., Liu, Y., Tang, Y., Zeng, Y., Qi, D., Chen, K., Zhao, C.: StyleShot: A Snapshot on Any Style. IEEE Transactions on Pattern Analysis and Machine Intelligence (2025)

12. Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., Misra, I.: ImageBind: One Embedding Space To Bind Them All. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 15180– 15190 (2023)

13. Hu, J., Peng, X., Xu, Z.: Study of Gray Image Pseudo-Color Processing algorithms. In: 6th International Symposium on Advanced Optical Manufacturing and Testing Technologies: Large Mirrors and Telescopes. vol. 8415, pp. 323–328. SPIE (2012)

14. Jiang, S., Chen, Z., Liang, J., Zhao, Y., Liu, M., Qin, B.: Infrared-LLaVA: Enhancing Understanding of Infrared Images in Multi-Modal Large Language Models. In: Findings of the Association for Computational Linguistics: EMNLP 2024. pp. 8573–8591 (2024)

15. Kulkarni, G., Premraj, V., Ordonez, V., Dhar, S., Li, S., Choi, Y., Berg, A.C., Berg, T.L.: BabyTalk: Understanding and Generating Simple Image Descriptions. IEEE Transactions on Pattern Analysis and Machine Intelligence 35(12), 2891– 2903 (2013)

16. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., Li, C.: LLaVA-OneVision: Easy Visual Task Transfer (2024)

17. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al.: Qwen2.5-VL Technical Report. arXiv preprint arXiv:2511.21631 (2025)

18. Li, B., Ge, Y., Ge, Y., Wang, G., Wang, R., Zhang, R., Shan, Y.: SEED-Bench: Benchmarking Multimodal Large Language Models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13299– 13308 (2024)

19. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual Instruction Tuning. Advances in neural information processing systems 36 (2024)

20. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: MMBench: Is Your Multi-modal Model an All-around Player? In: European Conference on Computer Vision. pp. 216–233. Springer (2024)

21. Lobry, S., Marcos, D., Murray, J., Tuia, D.: RSVQA: Visual Question Answering for Remote Sensing Data. IEEE Transactions on Geoscience and Remote Sensing 58(12), 8555–8566 (2020)

22. Loshchilov, I., Hutter, F.: Decoupled Weight Decay Regularization. In: International Conference on Learning Representations (2017)

23. Lu, P., Mishra, S., Xia, T., Qiu, L., Chang, K.W., Zhu, S.C., Tafjord, O., Clark, P., Kalyan, A.: Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering. In: The 36th Conference on Neural Information Processing Systems (NeurIPS) (2022)

24. Mathew, M., Karatzas, D., Jawahar, C.: DocVQA: A Dataset for VQA on Document Images. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 2200–2209 (2021)

25. McKinley, S., Levine, M.: Cubic Spline Interpolation. College of the Redwoods 45(1), 1049–1060 (1998)

26. Moshtaghi, M., Khajavi, S.H., Pajarinen, J.: RGB-Th-Bench: A Dense benchmark for Visual-Thermal Understanding of Vision Language Models. arXiv preprint arXiv:2503.19654 (2025)

27. Peng, B., Li, C., He, P., Galley, M., Gao, J.: Instruction Tuning with GPT-4. arXiv preprint arXiv:2304.03277 (2023)

28. Radewan, C.H.: Digital Image Processing With Pseudo-Color. In: Acquisition and Analysis of Pictorial Data. vol. 48, pp. 50–56. SPIE (1975)

29. Schoenberg, I.: Splines and Histograms. In: Spline Functions and Approximation Theory: Proceedings of the Symposium held at the University of Alberta, Edmonton May 29 to June 1, 1972. pp. 277–327. Springer (1973)

30. Su, Y., Lan, T., Li, H., Xu, J., Wang, Y., Cai, D.: PandaGPT: One Model To Instruction-Follow Them All. In: Proceedings of the 1st Workshop on Taming Large Language Models: Controllability in the era of Interactive Assistants! pp. 11–23 (2023)

31. Teed, Z., Deng, J.: RAFT: Recurrent All-Pairs Field Transforms for Optical Flow. In: European Conference on Computer Vision. pp. 402–419. Springer (2020)

32. Thawakar, O.C., Shaker, A.M., Mullappilly, S.S., Cholakkal, H., Anwer, R.M., Khan, S., Laaksonen, J., Khan, F.: XrayGPT: Chest Radiographs Summarization using Large Medical Vision-Language Models. In: Proceedings of the 23rd Workshop on Biomedical Natural Language Processing. pp. 440–448 (2024)

33. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: InternVL3.5: Advancing Open-Source Multimodal Models in Versatility, Reasoning, and Eficiency. arXiv preprint arXiv:2508.18265 (2025)

34. Xu, M., Yoon, S., Fuentes, A., Park, D.S.: A Comprehensive Survey of Image Augmentation Techniques for Deep Learning. Pattern Recognition 137, 109347 (2023)

35. Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., Wang, L.: MM-Vet: Evaluating Large Multimodal Models for Integrated Capabilities. arXiv preprint arXiv:2308.02490 (2023)

36. Yue, J., Zhang, S., Jia, Z., Xu, H., Han, Z., Liu, X., Wang, G.: MedSG-Bench: A Benchmark for Medical Image Sequences Grounding. arXiv preprint arXiv:2505.11852 (2025)

37. Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al.: MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9556– 9567 (2024)

38. Zhang, W., Cai, M., Zhang, T., Zhuang, Y., Mao, X.: EarthGPT: A Universal Multimodal Large Language Model for Multisensor Image Comprehension in Remote Sensing Domain. IEEE Transactions on Geoscience and Remote Sensing 62, 1–20 (2024)

39. Zhang, X., Bai, T., Li, H.: Pseudo-color coding method of infrared images based on human vision system. In: Infrared Materials, Devices, and Applications. vol. 6835, pp. 403–410. SPIE (2008)

40. Zhou, L., Hansen, C.D.: A Survey of Colormaps in Visualization. IEEE Transactions on Visualization and Computer Graphics 22(8), 2051–2069 (2015)

41. Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al.: InternVL3: Exploring Advanced Training and Test-Time Recipes for Open-Source Multimodal Models. arXiv preprint arXiv:2504.10479 (2025)

# Supplementary Material

## A Additional Results

## A.1 Quantity Results on VS-TDX

As discussed in the main paper, we have collected 3 modalities from VS-TDX [7] and modified the prompt formation to insert modality context. The results in Tab. A1 are evaluated under the original protocol of VS-TDX without our modification. The “Perception Mean” and “Understanding Mean” are calculated by averaging the accuracies of perception tasks (the first four tasks), and the understanding tasks (the last two tasks), following VS-TDX. Our modification has been shown in Tab. A2, where the blue text represents the modality contexts added by us, and the rest of the texts remain unchanged as VS-TDX.

Table A1: The results under the evaluation protocol of VS-TDX.

<table><tr><td>Model</td><td>Modality</td><td>Existence</td><td>Counting</td><td>Position</td><td>Description</td><td>Perception Mean</td><td>Contextual Understanding</td><td>Sensor Understanding</td><td>Understanding Mean</td></tr><tr><td rowspan="3">LLaVA-1.5 7B</td><td>Thermal</td><td>57.1%</td><td>32.7%</td><td>68.8%</td><td>59.9%</td><td>54.6%</td><td>73.1%</td><td>38.4%</td><td>55.8%</td></tr><tr><td>Depth</td><td>69.7%</td><td>23.2%</td><td>55.2%</td><td>74.7%</td><td>55.7%</td><td>68.9%</td><td>22.6%</td><td>45.8%</td></tr><tr><td>X-ray</td><td>64.6%</td><td>35.3%</td><td>54.6%</td><td>73.4%</td><td>57.0%</td><td>76.2%</td><td>52.0%</td><td>64.1%</td></tr><tr><td rowspan="3">LLaVA-1.5 7B (Ours)</td><td>Thermal</td><td>71.5%</td><td>57.3%</td><td>75.0%</td><td>69.9%</td><td>68.4%</td><td>79.3%</td><td>58.5%</td><td>68.9%</td></tr><tr><td>Depth</td><td>74.5%</td><td>50.0%</td><td>67.3%</td><td>77.6%</td><td>67.3%</td><td>76.3%</td><td>20.6%</td><td>48.5%</td></tr><tr><td>X-ray</td><td>66.2%</td><td>60.2%</td><td>67.3%</td><td>78.7%</td><td>68.1%</td><td>77.9%</td><td>68.3%</td><td>73.1%</td></tr><tr><td rowspan="3">Qwen-2.5-VL 3B</td><td>Thermal</td><td>75.7%</td><td>46.7%</td><td>72.7%</td><td>78.4%</td><td>68.4%</td><td>83.5%</td><td>73.9%</td><td>78.7%</td></tr><tr><td>Depth</td><td>75.5%</td><td>40.0%</td><td>62.9%</td><td>80.0%</td><td>64.6%</td><td>74.3%</td><td>40.6%</td><td>57.4%</td></tr><tr><td>X-ray</td><td>75.0%</td><td>34.1%</td><td>61.8%</td><td>82.4%</td><td>63.3%</td><td>82.3%</td><td>77.1%</td><td>79.7%</td></tr><tr><td rowspan="3">Qwen-2.5-VL 3B (Ours)</td><td>Thermal</td><td>83.5%</td><td>70.4%</td><td>78.1%</td><td>85.2%</td><td>79.3%</td><td>87.1%</td><td>77.8%</td><td>82.4%</td></tr><tr><td>Depth</td><td>80.3%</td><td>59.5%</td><td>75.7%</td><td>83.8%</td><td>74.8%</td><td>80.2%</td><td>53.9%</td><td>67.0%</td></tr><tr><td>X-ray</td><td>76.6%</td><td>72.3%</td><td>76.9%</td><td>87.3%</td><td>78.3%</td><td>85.7%</td><td>83.2%</td><td>84.5%</td></tr><tr><td rowspan="3">Qwen-2.5-VL 7B</td><td>Thermal</td><td>78.9%</td><td>47.2%</td><td>77.3%</td><td>80.2%</td><td>70.9%</td><td>81.6%</td><td>66.2%</td><td>73.9%</td></tr><tr><td>Depth</td><td>78.2%</td><td>51.6%</td><td>70.5%</td><td>82.1%</td><td>70.6%</td><td>71.5%</td><td>28.7%</td><td>50.1%</td></tr><tr><td>X-ray</td><td>78.9%</td><td>44.6%</td><td>78.1%</td><td>92.6%</td><td>73.5%</td><td>84.9%</td><td>80.9%</td><td>82.9%</td></tr><tr><td rowspan="3">Qwen-2.5-VL 7B (Ours)</td><td>Thermal</td><td>79.7%</td><td>68.3%</td><td>86.7%</td><td>82.7%</td><td>79.4%</td><td>86.7%</td><td>71.0%</td><td>78.9%</td></tr><tr><td>Depth</td><td>83.2%</td><td>60.5%</td><td>78.4%</td><td>84.2%</td><td>76.6%</td><td>80.2%</td><td>36.2%</td><td>58.2%</td></tr><tr><td>X-ray</td><td>79.0%</td><td>71.1%</td><td>79.3%</td><td>90.6%</td><td>80.0%</td><td>85.3%</td><td>83.0%</td><td>84.2%</td></tr><tr><td rowspan="3">Qwen-3-VL 4B</td><td>Thermal</td><td>86.4%</td><td>60.3%</td><td>75.0%</td><td>84.2%</td><td>76.5%</td><td>84.1%</td><td>75.6%</td><td>79.9%</td></tr><tr><td>Depth</td><td>88.2%</td><td>51.6%</td><td>72.3%</td><td>86.9%</td><td>74.7%</td><td>79.9%</td><td>25.5%</td><td>52.7%</td></tr><tr><td>X-ray</td><td>78.5%</td><td>48.6%</td><td>82.1%</td><td>93.0%</td><td>75.5%</td><td>85.7%</td><td>81.2%</td><td>83.5%</td></tr><tr><td rowspan="3">Qwen-3-VL 4B (Ours)</td><td>Thermal</td><td>85.1%</td><td>75.4%</td><td>84.4%</td><td>82.5%</td><td>81.8%</td><td>90.0%</td><td>69.2%</td><td>79.6%</td></tr><tr><td>Depth</td><td>89.2%</td><td>60.0%</td><td>82.6%</td><td>92.6%</td><td>81.1%</td><td>87.0%</td><td>34.1%</td><td>60.5%</td></tr><tr><td>X-ray</td><td>80.0%</td><td>63.1%</td><td>81.7%</td><td>92.2%</td><td>79.2%</td><td>88.2%</td><td>85.7%</td><td>86.9%</td></tr><tr><td rowspan="3">Qwen-3-VL 8B</td><td>Thermal</td><td>85.4%</td><td>60.8%</td><td>78.1%</td><td>81.9%</td><td>76.6%</td><td>87.1%</td><td>66.9%</td><td>77.0%</td></tr><tr><td>Depth</td><td>86.1%</td><td>53.2%</td><td>74.7%</td><td>87.4%</td><td>75.3%</td><td>78.0%</td><td>36.9%</td><td>57.4%</td></tr><tr><td>X-ray</td><td>76.4%</td><td>42.2%</td><td>76.5%</td><td>93.4%</td><td>72.1%</td><td>83.8%</td><td>80.9%</td><td>82.3%</td></tr><tr><td rowspan="3">Qwen-3-VL 8B (Ours)</td><td>Thermal</td><td>83.9%</td><td>73.9%</td><td>77.3%</td><td>84.8%</td><td>80.0%</td><td>92.2%</td><td>75.3%</td><td>83.8%</td></tr><tr><td>Depth</td><td>88.4%</td><td>68.4%</td><td>82.0%</td><td>90.0%</td><td>82.2%</td><td>87.0%</td><td>58.8%</td><td>72.9%</td></tr><tr><td>X-ray</td><td>79.4%</td><td>69.5%</td><td>81.3%</td><td>93.9%</td><td>81.0%</td><td>88.1%</td><td>88.3%</td><td>88.2%</td></tr></table>

Table A2: An example of our modification, we added the modality context to the original prompt of VS-TDX for understanding tasks.

## Question

Thermal images visualize infrared radiation emitted by objects using heat-sensing sensors. They can be used to analyze temperature distribution, detect objects, and inspect equipment conditions. Usually, the color of thermal images represents for temperatures, for example, high temperature usually appears reddish or bright in thermal image. Here shows a thermal image, please answer question according to above information. <image> Question: What is the temperature characteristic of the figure in the center of the image? Choices: A. The figure has the same temperature as the surroundings. B. The figure is warmer than the surroundings. C. The figure is cooler than the surroundings. D. The figure is not distinguishable temperature-wise. Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.

![](images/35a645d69b683b1191e4c7c01caf5e3eb1719db6123f69a583a29c48639285a2.jpg)

## A.2 Data Scaling on Qwen-2.5-VL 7B

Results in Tab. A3 reveal an ascending trend with increasing data scales. However, scaling the number of samples is less meaningful than increasing image styles and task diversity for generalization, which we expect in future work.

Table A3: Perception results for data scales on Qwen-2.5-VL 7B.

<table><tr><td>Data Scale</td><td>Thermal</td><td>Depth</td><td>X-Ray</td><td>Optical Flow</td><td>Synthesized 1</td><td>Synthesized 2</td><td>Mean</td></tr><tr><td>baseline</td><td>74.5%</td><td>73.3%</td><td>75.7%</td><td>77.0%</td><td>87.1%</td><td>81.1%</td><td>78.1%</td></tr><tr><td>25%</td><td>78.3%</td><td>77.3%</td><td>78.7%</td><td>83.3%</td><td>89.7%</td><td>84.1%</td><td>81.9%</td></tr><tr><td>50%</td><td>76.7%</td><td>78.3%</td><td>78.4%</td><td>77.8%</td><td>88.8%</td><td>85.0%</td><td>80.8%</td></tr><tr><td>75%</td><td>77.7%</td><td>79.1%</td><td>81.5%</td><td>82.8%</td><td>91.3%</td><td>86.1%</td><td>83.1%</td></tr><tr><td>100%</td><td>79.8%</td><td>79.2%</td><td>79.6%</td><td>88.4%</td><td>91.2%</td><td>86.4%</td><td>84.1%</td></tr></table>

## A.3 Target-tuning with a Samll amount of Real Data

We provide the results of tuning Qwen-2.5-VL 7B with TDX data (Extra Thermal Depth, and X-Ray data, which is not in VVM-Bench) in Tab. A4. With 600 samples (each 200 for Thermal, Depth, and X-Ray) and 10 epochs, the performance is close to our method. With 3600 samples (1200 per modality) and 10 epochs, the performance on these modalities surpasses ours. However, the performance on the other modalities remains close to baseline, indicating that simply training with several modalities benefits few in generalization. That makes our work meaningful, as we don’t expect that synthetic modality tuning could replace real modality tuning. Instead, our method aims to enhance LMMs when real modality tuning is unavailable.

Table A4: Perception results with small amount of TDX data on Qwen-2.5-VL 7B and Qwen-3-VL 8B.

<table><tr><td>Model</td><td>Thermal</td><td>Depth</td><td>X-Ray</td><td>Optical Flow</td><td>Synthesized 1</td><td>Synthesized 2</td><td>Mean</td></tr><tr><td>Qwen-2.5-VL</td><td>74.5%</td><td>73.3%</td><td>75.7%</td><td>77.0%</td><td>87.1%</td><td>81.1%</td><td>78.1%</td></tr><tr><td>Qwen-2.5-VL (Ours)</td><td>79.8%</td><td>79.2%</td><td>79.6%</td><td>88.4%</td><td>91.2%</td><td>86.4%</td><td>84.1%</td></tr><tr><td>Qwen-2.5-VL (600)</td><td>79.5%</td><td>78.2%</td><td>80.0%</td><td>75.4%</td><td>85.2%</td><td>79.6%</td><td>79.6%</td></tr><tr><td> $\Delta$ (-ours)</td><td>-0.3%</td><td>-1.0%</td><td>+0.4%</td><td>-13.0%</td><td>-6.0%</td><td>-6.8%</td><td>-4.5%</td></tr><tr><td>Qwen-2.5-VL (3600)</td><td>87.4%</td><td>82.2%</td><td>85.2%</td><td>76.5%</td><td>86.4%</td><td>82.6%</td><td>83.4%</td></tr><tr><td> $\Delta$ (-ours)</td><td>+7.6%</td><td>+3.0%</td><td>+5.6%</td><td>-11.9%</td><td>-4.8%</td><td>-3.8%</td><td>-0.7%</td></tr><tr><td> $\Delta$ (-baseline)</td><td>+12.9%</td><td>+8.9%</td><td>+9.5%</td><td>-0.5%</td><td>-0.7%</td><td>+1.5%</td><td>+5.3%</td></tr></table>

## A.4 Quality Results

We observe the diference in the behavior of the baselines and our tuned model and display some examples here. Quality result 1 in Tab. A5, serving as a failure case, demonstrates an unnecessary preference for guessing the visual modality of the non-RGB images in the base model, which is unfortunately inherited by our tuned model. This preference comes from the insuficient quantity and diversity of non-RGB modalities during the visual-language alignments, which may require complete realignment with diverse non-RGB images. However, our tuned model still exhibits a more detailed perception of colors.

Quality result 2 in Tab. A6 shows the ability to connect the modality context with visual perception. The base model wrongly assumes the red area in the center of the plant represents high fungal signals, indicating an inability to link visual perceptions with the physical meanings in the modality context. Meanwhile, our tuned model correctly links the color of areas to the modality context, resulting in a more consistent understanding.

Table A5: Quality result 1: modality-unaware description task on Qwen-2.5 VL 3B and our tuned model.

## Quality Result 1

![](images/9927461daf7f3b0e3491a1644e8b9c03d45ec50b817609dcbe0899c6436c1be2.jpg)

## Instruction

Describe the right image.

## Model Response

## Qwen-2.5 VL 3B:

The right image is a color-coded representation of a plant, likely an infrared or thermal imaging scan. The colors indicate different temperature levels, with warmer areas appearing in red and cooler areas in blue. The plant appears to have a healthy appearance with no visible signs of disease or damage. The detailed coloration helps in identifying specific regions of the plant that might require further attention or monitoring.

## Qwen-2.5 VL 3B (Ours):

The image on the right is a thermal scan of a plant, showing heat signatures in various colors such as blue, green, orange, and red.

Table A6: Quality result 2: modality-aware understanding on Qwen-2.5 VL 3B and our tuned model.  
Quality Result 2  
![](images/c0dc0e6a70d3b935da4c107c501894d0af54ff7a2bdd640bf1cd59aae61de4b8.jpg)

The question in Tab. A7 tests the ability of modality-unaware perception by asking about the existence of objects in the image. The base model, Qwen-2.5 VL 7B, is still trying to guess the modality of the image and refuses to answer the question by requiring more contexts. However, it is enough to determine the existence of the forest just by viewing the image itself, which is successfully achieved by our tuned model.

The Tab. A8 shows an open understanding task with modality context. The base model answers with a lengthy statement that falsely believes the trees are concentrated and located in a lower altitude because of their green or blue colors.

This error clearly reflects how erroneous perception afects the understanding, even with modality contexts, while our tuned model exhibits a correct understanding based on robust perception instead.

Table A7: Quality result 3: modality-unaware perception on Qwen-2.5 VL 7B and our tuned model.  
Quality Result 3  
![](images/474865389623c66bb019448e9668d90cc2fd564817fe3b9679ab0fc705507883.jpg)

## Model Response

Qwen-2.5 VL 7B: The image appears to be a topographical map or a digital elevation model (DEM) of an area, not a photograph of a forest. The colors represent different elevations and possibly vegetation types, but they do not depict a forest in the traditional sense. The blue areas likely represent lower elevations, while the green and yellow areas could indicate higher elevations and potentially areas with vegetation. However, without additional context or a legend, it’s not possible to definitively state that there is a forest present. If you have more information about the location or the purpose of this image, it might help in identifying specific features like forests.

Table A8: Quality result 4: modality-aware understanding on Qwen-2.5 VL 7B and our tuned model.

## Quality Result 4

![](images/f09204896daafabf55084ea952c5822e52e5504febe5d43c860dca19f4c63064.jpg)

## Instruction

Context:

The image shows an altitude map of a landscape. The color closer to red means higher altitude. The color closer to blue means lower altitude.

Question:

What can we learn about the trees in the image?

## Model Response

Qwen-2.5 VL 7B:

From the image, which appears to be an altitude map of a landscape, we can infer some characteristics about the trees based on their distribution and the colors representing different altitudes:

Tree Distribution: The trees seem to be concentrated in certain areas rather than being uniformly spread across the landscape. This suggests that there might be specific environmental conditions (such as soil type, water availability, or sunlight) that favor tree growth in those areas.

Altitude Influence: The trees appear to be more prevalent in areas with lower altitudes, as indicated by the blue and green colors. This could imply that the trees are adapted to growing in these lower-altitude regions, possibly due to factors such as temperature, precipitation, or soil composition.

Topography: The presence of trees in specific areas might also indicate that these areas have more favorable topographical features for tree growth, such as gentle slopes or areas where water collects. Vegetation Patterns: The distinct patches of different colors suggest that there might be variations in vegetation types or densities. The transition from one color to another could indicate changes in the type of vegetation or the health of the trees. Overall, the image provides insights into the relationship between tree distribution and altitude, suggesting that trees are influenced by both topographical and environmental factors in this landscape. Qwen-2.5 VL 7B (Ours): The trees are located at a medium altitude, as indicated by the orange and yellow colors surrounding them.

## A.5 Discussion of Limitations

As discussed before, the generalization to unseen visual modalities is based on a modality-unaware perception, from which the visual-language alignment is inherited from RGB vision. Our tuning only disentangles that alignment by semantics and photographics, not executing a completely new alignment or injecting new knowledge. Besides, our modality context only introduces the mapping of physical meanings, without injecting new knowledge or abilities into the model. That means the generalization to unseen modalities is limited within the existing knowledge and abilities of known objects learned by base LMMs. To be honest, that would constrain the application of such generalization, as the non-RGB images are usually highly related to specific downstream tasks, which a general LMM could not solve without specialized domain knowledge. Although how such new knowledge could be absorbed by the model from the context during inference remains an unexplored challenge, the current research on LMMs with long-CoT and reinforcement learning still demonstrates a promising prospect.

Meanwhile, the evaluation format of VQA we employed in VVM-Bench is quite simple and is unable to measure the complex reasoning ability of the tested LMMs. While it’s enough to serve as a perception-evaluating metric, it’s easy to be hacked by guessing the answer from options in complex understanding tasks, especially with competent contexts. We are looking forward to further development on multi-hop inference and complex context reasoning, as well as benchmarks and evaluation metrics in the direction of generalization to unseen visual modalities. We believe our method could serve as a perceptual basis for further exploration.

## B Evaluation Details

## B.1 Evaluation Templates

The templates of prompts used in evaluation are shown in Tab. A9 in our VVM-Bench. Basically, the prompts are composed of four parts: image, question, options, and a format instruction for calculating accuracy. For understanding tasks, modality contexts are inserted at “<context>” as shown in Tab. A9.

Table A9: Prompt Templates of VVM-Bench in evaluation for all modalities and task sets.

<table><tr><td>Modality</td><td>Task set</td><td>Template</td></tr><tr><td rowspan="9">Collected Modalities</td><td rowspan="4">Perception</td><td>&lt;image&gt;</td></tr><tr><td>&lt;question&gt;</td></tr><tr><td>&lt;options&gt;</td></tr><tr><td>Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.</td></tr><tr><td rowspan="5">Understanding</td><td>&lt;context&gt;</td></tr><tr><td>&lt;image&gt;</td></tr><tr><td>&lt;question&gt;</td></tr><tr><td>&lt;options&gt;</td></tr><tr><td>Please directly answer the question and provide the correct option letter, e.g., A, B, C, D.</td></tr><tr><td rowspan="9">Fabricated Modalities</td><td rowspan="4">Perception</td><td>&lt;image&gt;</td></tr><tr><td>&lt;question&gt;</td></tr><tr><td>&lt;options&gt;</td></tr><tr><td>Answer with the option&#x27;s letter from the given choices directly.</td></tr><tr><td rowspan="5">Understanding</td><td>&lt;image&gt;</td></tr><tr><td>&lt;context&gt;</td></tr><tr><td>&lt;question&gt;</td></tr><tr><td>&lt;options&gt;</td></tr><tr><td>Answer with the option&#x27;s letter from the given choices directly.</td></tr></table>

## B.2 Benchmark Details

Our VVM-Bench contains 13317 VQA samples in total, including 6991 samples in the perception subset and 6326 samples in the understanding subset. The detailed modalities and tasks distribution is listed in Tab. A10. All samples follow the templates shown in Tab. A9.

Table A10: Number of samples in VVM-Bench.

<table><tr><td>Modality</td><td>Thermal</td><td>Depth</td><td>X-Ray</td><td>Optical Flow</td><td>Synthesized 1</td><td>Synthesized 2</td><td>Total</td></tr><tr><td>P</td><td>1330</td><td>1873</td><td>1692</td><td>378</td><td>1018</td><td>700</td><td>6991</td></tr><tr><td>U</td><td>1386</td><td>2304</td><td>1575</td><td>240</td><td>421</td><td>400</td><td>6326</td></tr><tr><td>Total</td><td>2716</td><td>4177</td><td>3267</td><td>618</td><td>1439</td><td>1100</td><td>13317</td></tr><tr><td>P</td><td>Cnt.(885)</td><td>Log.Inf.(1758)</td><td>Exis.(2235)</td><td>R.Pos.(1372)</td><td>Color(247)</td><td>Shape(247)</td><td>R.Size(247)</td></tr><tr><td>U</td><td colspan="2">Modal.Und.(3502)</td><td colspan="2">Scene.Und.(2284)</td><td colspan="2">Event Ext.(261)</td><td>Modal.Reas.(279)</td></tr></table>

Table A11: Modality Contexts of VVM-Bench in evaluation. “Synthesized Context” means the context is fabricated by us.

<table><tr><td>Modality</td><td>Context</td></tr><tr><td>Thermal</td><td>Thermal images visualize infrared radiation emitted by objects using heat-sensing sensors. They can be used to analyze temperature distribution, detect objects, and inspect equipment conditions. Usually, the color of thermal images represents for temperatures, for example, high temperature usually appears reddish or bright in thermal image. Here shows a thermal image, please answer question according to above information.</td></tr><tr><td>Depth</td><td>Depth images visualize the distance between a sensor and objects in a scene by capturing depth information. They can be used to measure object dimensions, map environments in 3D, and assist in object recognition and navigation tasks. The color of depth image usually represents the depth or the distance between that point and the depth camera. Here shows a depth image, please answer question according to above information.</td></tr><tr><td>Xray</td><td>X-ray images visualize the internal structures of objects by capturing the varying absorption of X-rays. They can be used to inspect internal components, identify structural defects, and analyze materials or biological tissues for diagnostic purposes. X-ray images usually shows different structures overlapping as the X-ray can go through the object, the more absorption of X-ray, more brighter in the image. Here shows a xray image, please answer question according to above information.</td></tr><tr><td>Optical Flow</td><td>The image is an optical flow visualization. The color of the pixels represents for the movement of the object. The saturation of the color means the magnitude of the movement vector. The brighter the color, the larger the movement is. The paler color means smaller movement. The direction of the movement is represented by the hue of the color. Red means moving right, orange means moving lower right, yellow means moving down, green means moving lower left, blue means moving upper right, purple means moving up, pink means moving upper right.</td></tr><tr><td rowspan="2">Synthesized 1</td><td>Synthesized Context (Hardness)</td></tr><tr><td>Synthesized Context (Humidity)</td></tr><tr><td rowspan="6">Synthesized 2</td><td>Synthesized Context (Hardness)</td></tr><tr><td>Synthesized Context (Humidity)</td></tr><tr><td>Synthesized Context (Heat Signals)</td></tr><tr><td>Synthesized Context (Radiation)</td></tr><tr><td>Synthesized Context (Reflective Level)</td></tr><tr><td>Synthesized Context (Illuminance Level)</td></tr></table>

The detailed modality contexts are shown in Tab. A11. Among the 6 modalities, 4 of them use modality contexts with real physical meanings. For the 2 synthesized modalities, we manufacture multiple meanings to compose 6 synthesized contexts as shown in Tab. A12. The synthesized modalities have multiple synthesized contexts among their samples, with the inclusion relationships listed in Tab. A11.

Table A12: Synthesized Modality Contexts of VVM-Bench for synthesized modalities.

<table><tr><td>Modality</td><td>Context</td></tr><tr><td>Hardness</td><td>The image visualizes the hardness of objects. Different colors represent different hardnesses. Red means solid hard, yellow means medium hardness, green means slightly soft, blue means medium soft, purple means very soft.</td></tr><tr><td>Humidity</td><td>The image visualizes the humidity in the air. Different colors represent different humidity. Red means low humidity, yellow means slightly dry, green means medium humidity, blue means humid, purple means high humidity.</td></tr><tr><td>Heat Signals</td><td>The image visualizes the heat signals in the air. The colors in the picture range from black to white. The darker area means a weak signal, and the brighter area means a strong signal.</td></tr><tr><td>Radiation</td><td>The image visualizes the radiation signals in the air. The colors in the picture range from red to yellow. The redder area means lower value, and the yellower area means higher value.</td></tr><tr><td>Reflective Level</td><td>The image visualizes the reflective level of the object. The colors in the picture range from blue to red. The bluer area means low reflective level, and the redder area means high reflective level.</td></tr><tr><td>Illuminance Llevel</td><td>The image visualizes the illuminance level in the area. The colors in the picture range from dark pink to white. The darker area means lower value, and the brighter area means higher value.</td></tr></table>

## C Modality Synthesis Details

## C.1 Image Synthesis

The histogram adjusting method in Fig. 3 is shown in Algorithm A1. We first generate 1-4 keypoints randomly in the value plane, ranging from 0 to 255. Then, we interpolate the cubic spline to form a map function S(x) on [0, 255] → [0, 255]. That function could change the histogram by changing the pixel value. However, random splines may overflow from [0, 255]. Thus, we use rejection sampling to make sure the value range of S(x) is valid.

The Fig. A1 shows more examples with corresponding colormap bars, indicating that the random splines are simple but efective in imitating the characteristics of non-RGB images.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm A1 Adjust Histogram with Random Splines
Input Grayscale Image I; Hyperparameters The max number of keypoints  $N_{max}$ ; Max try times  $T_{fail}$ 
1: Randomize the number of keypoints  $N = \text{Random}(1, N_{max})$ 
2: for cnt = 1, ...,  $T_{fail}$  do
3: Generate N random keypoints  $(x_{1}, y_{1}) \ldots (x_{N}, y_{N}) \in [0, 255]^{2}$ .
4: Interpolate a cubic spline  $S(x), x \in [0, 255]$  using keypoints  $(x_{1}, y_{1}) \ldots (x_{N}, y_{N})$ .
5: if  $S(x) \in [0, 255]$  then
6: break
7: else
8: The value range of  $S(x)$  is illegal, reject  $S(x)$ .
9: if cnt ==  $T_{fail}$  then
10:  $S(x) = x$ 
11: Adjust the histogram by mapping pixel value using the spline,  $\hat{I} = S(I)$ .
Output Adjusted Image  $\hat{I}$
</div>

More Examples of Fabricated Images  
![](images/6495ece9a54d4c9a6ef589e76b71e2c7a58c7135866a659238f5de1effaac39e.jpg)  
Fig. A1: More examples of fabricated images produced by our image synthesis pipeline with diferent color maps.

## C.2 Modality Context Template

Table A13: An example of templates used to fabricate modality context.

![](images/f59cf1429f4250dc51ca9805409aab84982e7e2a88cc130be59af514a5695002.jpg)

For the modality context synthesis, we employ both structured text and LMMgenerated text to build diverse modality contexts. The structured template is shown in Tab. A13, with human-made physical attributes, corresponding color range, and the mapping between value and color. We assemble the modality context with the color and value information from the image synthesis pipeline.

However, such structured text could be meaningless as it is blind to the content in the image, so we utilize the annotation from the source RGB images to filter out the meaningless expression texts. For the LMM-generated contexts, we merge the generation of modality context with the generation of modality-aware VQA tasks in practice to ensure the LMM won’t generate useless modality contexts. We use a 72B Qwen-2.5-VL [17] LMM to complete all multimodal generation and a 32B Qwen-3 [4] for language-only generation. The Tab. A14 shows two examples of LMM-generated modality context. Detailed generation prompts are listed in Tab. A22.

## Table A14: Generated Modality Context Examples

```txt
Generated Modality Context Examples
Example 1:
The special color filter in the image represents varying temperature levels. Darker shades indicate colder temperatures, while lighter shades represent warmer temperatures. This allows us to infer how the temperature distribution affects the objects and environment within the bathroom.
Example 2:
In this image, the color filter emphasizes emotional states: blue represents calmness and serenity, while red signifies urgency and alertness. The girl's blue attire and the blue tones around her suggest she is in a state of calm, whereas the red patterns on the blanket indicate areas of potential concern or urgency within the hospital environment.
```

## D Training Task Details

## D.1 Modality-unaware Description Tasks

The modality-unaware tasks have description tasks and VQA tasks. The generation prompt of description tasks is shown in Tab. A15 and Tab. A16 for semantic descriptions and photographic descriptions, respectively. The prompt of semantic description in Tab. A15 is designed to tell the LMM to extract the shared semantics of the RGB image and the non-RGB image and generate a semantic description from the original RGB descriptions. Instead, the prompt of photographic descriptions in Tab. A16 is designed to capture the unique visual clues from the non-RGB image and generate a detailed description covering the basic visual elements in the image.

Table A15: Semantic Description Generation Prompt Template

## Semantic Description Generation Prompt Template

```txt
You are an expert at understanding images and describe the scene in the image.
You are given two images: one regular and one non-regular, both are shot from same scene, your task is to describe the scene behind two images.
Regular Image:
<image>
Non-Regular Image:
<image>
Reference Description:
"<caption>"
Your task is to:
1. Analyze the two images, ignore the differences between them, focus on the same scene behind them. Make sure you understand the scene fully.
2. Generate a description that covers the scene. You can check reference description if you are not sure about the scene, but DO NOT copy it.
3. Note that you are describing the scene not the two images, so do not describe any features that affected by the camera or photographic methods.
4. Generate a diverse question that require a subject to describe the scene from a image, please note the subject only can see one image.
5. Output the scene description and the question in format of JSON.
[REQUIREMENT]
1. Please ensure that the description is consistent with both images and DO NOT add any features that not exists in the images to the description.
2. The description should focus on the scene in the images, avoid describing details like colors and texture. Also DO NOT describe the features that not showing in both images.
3. Give your output in a format of JSON, which has the same structure like this:...
```

Table A16: Photographic Description Generation Prompt Template

![](images/bf245db72267a6b3a2c87297b03435d32fd1d727d5f992b5bf601f9335a2e78b.jpg)

## D.2 Modality-unaware VQA Tasks

The categories of modality-unaware VQA tasks are listed in Tab. A17 and Tab. A18 with exemplar questions. As described before, the semantic QA tasks in Tab. A17 are focused on understanding the scene semantics, including object relationships and a broader understanding of the physical world behind the image. And the photographic QA tasks in Tab. A18 are more focused on the visual elements and attributes like number, color and position.

Table A17: Semantic QA Tasks and Question Templates

<table><tr><td>Task</td><td>Question Example</td></tr><tr><td>Scene Understanding</td><td>What are the person and the dog doing?What happened in the scene?</td></tr><tr><td>Scene Reasoning</td><td>Why is the man holding a phone?Why does the dog have its tone out of its mouth?What causes the man to run?</td></tr><tr><td>Event Extension</td><td>What will happen if the man drops his phone?To solve the problem, which tool does the man need?How to prevent the car from crashing?</td></tr><tr><td>Logical Inference</td><td>Does the phone belong to the woman?What is the most possible relationship between the man and the woman?Which person is the bravest one in the scene?</td></tr><tr><td>Knowledge Inference</td><td>How does the person feel according to the scene?When does this event likely take place?What place is it in the image?</td></tr></table>

Table A18: Photographic QA Tasks and Question Templates

<table><tr><td>Task</td><td>Question Example</td></tr><tr><td>Color</td><td>What is the color of the dog in the image?How many colors can you see on the human?</td></tr><tr><td>Object Shape</td><td>What is the shape of the desk?Can you describe the shape of the desk?</td></tr><tr><td>Relative Position</td><td>What object is next to the person?Is the dog to the left of the cat?Which side of the desk to the person?</td></tr><tr><td>Relative Size</td><td>Which object looks biggest in the image?Which object is bigger in the cat and the human?Are the two apples on the desk the same size?</td></tr><tr><td>Object Existence</td><td>Is there a door in the image?Which object does not appear in the image?Do all numbers from 0-9 appear in the image?</td></tr><tr><td>Counting</td><td>How many objects are there in the image?Are there 5 red apples on the desk?</td></tr></table>

The prompt template used to generate these VQA tasks is listed in Tab. A19, which uses the question type and examples to generate corresponding questions and answers. The “Scene description” is replaced with semantic descriptions or photographic descriptions when generating VQA separately. The prompt ensures the output is organized in a JSON format and easily converted to training and evaluation data.

Table A19: Modality-unaware QA generation prompt template  
![](images/bfa2d237daf435188bd3d35d338393dc6567b66257262015217d14a013973a8d.jpg)

```txt
3. Create questions according to the question type, but please do not completely copy the content of the example questions. You need to generated all kinds of questions related to the question type and the ability behind the question type.
4. You can generate diverse questions requiring broad common knowledge, but ensure the connection between question and answer to be clear and reasonable. Avoid generating question and answers with a lot of guessing and assuming.
5. Give your output in a format of JSON, which has the same structure like this:
[FORMAT] Your output MUST be in JSON format as follows:
{
    "strategy": "[HOW TO MAKE IT CHALLENGING]",
    "question": "[YOUR QUESTION]",
    "answer": "[YOUR CORRECT ANSWER]",
    "ability_needed": "[ABILITY TESTED OF YOUR QUESTION TYPE]",
    "incorrect_answers": [
    "[INCORRECT ANSWER 1]",
    "[INCORRECT ANSWER 2]",
    "[INCORRECT ANSWER 3]", ...
    ]
}
```

## D.3 Modality-aware Description Tasks

The modality-aware training also contains description tasks and VQA tasks. The prompt used to generate modality descriptions is listed in Tab. A20. Note that this is a language-only task, for which we use a 32B Qwen-3 language model to generate modality descriptions by combining semantic description, photographic description and modality contexts. The modality contexts are generated along with modality QA tasks to ensure their usefulness. Other two descriptions are generated in the modality-unaware tasks before.

## D.4 Modality-aware VQA Tasks

The sub-tasks and their question examples are listed in Tab. A21. They are generated using the prompt listed in Tab. A22 along with the modality context. We use a similar generation format to ensure the usability of the LMMs’ output. Note that the “modality description” in Tab. A22 actually refers to modality context, as it is asking the model to generate a piece of text describing the mapping between color patterns and physical meanings.

Table A20: Modality Description Generation Prompt Template

## Modality Description Generation Prompt Template

You are an expert at understanding image descriptions and rewrite them to a better description. You are given tow description about a special colored image, one is focusing on describe the scene and the other one is focusing on describe the detailed colors and shapes: Scene description: <semantic\_caption> Photographic description: <photographic\_caption> Modality description: <modality\_context> Your task is to: 1. Analyze the tow descriptions, make sure you understand the image according to the description. 2. Analyze the modality description which explained what the color filter is trying to emphasize. 3. According to the modality description, combine the scene description and photographic description to a more completed description that cover all features in the image combined with the modality description. 4. Create one question that asking a subject describe the image based on the modality description and image. You can assume the subject only can see the image and modality description. [REQUIREMENT]1. Please ensure that the image information and the modality description are fully utilized in the completed description. Every details in the image is described in the answer with the modality description. 2. Avoid generating question with a lot of guessing and assuming. However you can make up some extra knowledge in the modality description, it will be provided to subject. 3. Give your ouput in a format of JSON, which has the same structure like this:

<semantic\_caption>: Semantic Description generated before. <photographic\_caption>: Photographic Description generated before. <modality\_context>: Modality Context fabricated before.

Table A21: Modality QA Tasks and Question examples

<table><tr><td>Task</td><td>Question Example</td></tr><tr><td>Modality Understanding</td><td>Based on the color meaning, what are the person and the dog doing?What happened in the scene shown by the special color filter?</td></tr><tr><td>Modality Reasoning</td><td>Why does the dog look red by the color filter?Why does the human look irregularly under the color patterns?What causes the man to run?</td></tr><tr><td>Event Extension</td><td>What will happen if the man drops his phone, considering the ground is red?To solve the problem, which tool does the man need if the color red represents hardness?Which person should be quarantined if the color means temperature?</td></tr><tr><td>Logical Inference</td><td>Is the car running?Which are the most possible tools to open the door?Which person is the happiest one in the scene based on the temperature shown by the color?</td></tr><tr><td>Knowledge Inference</td><td>How does the person feel according to the scene if the color represents blood flow?When does this event likely take place, considering the ultraviolet level?What place is it in the image if the color indicates the gravity level?</td></tr></table>

Table A22: Modality-aware QA generation prompt template

## Modality-aware QA Generation Prompt Template

<table><tr><td></td></tr><tr><td>You are an expert at understanding images and generating relevant questions and answers based on them to test the ability of a visual assistant.</td></tr><tr><td>You are given a non-regular image that has special color filter and a question type with several examples of questions:</td></tr><tr><td>Scene description: {caption}</td></tr><tr><td>Question Type: {question_type}</td></tr><tr><td>Question Examples:</td></tr><tr><td>{question_examples}</td></tr><tr><td>Your task is to:</td></tr><tr><td>1. Analyze the image, scene description, question type, and question examples thoroughly and make sure you understand the scene in the image and the ability demanded by the question type without copy the example questions.</td></tr><tr><td>2. Analyze the special color pattern of the image and generate a modality description to explain what the color filter is trying to emphasize.</td></tr><tr><td>3. Make a strategy to create challenging questions about the image and its color filter to meet the requirements as below to test the ability of the subject (a visual assistant).</td></tr><tr><td>4. Create one question of the given question type based on the strategy and image according to the modality description. Your question should be clear and meaningful, requiring the subject having a strong ability that connecting common knowledge to the modality description to solve the question.</td></tr><tr><td>5. Provide the correct answer to your question according to the image and the modality description. The answer should be clear and distinct.</td></tr><tr><td>6. Generate at most {negative_samples_num} different plausible incorrect answers that the subject might give if it is doesn&#x27;t understand the the modality description or lacks abilities. Adjust the number of incorrect answers according to your question (i.e. if your question only has &#x27;yes&#x27; or &#x27;no&#x27; as answers, only generate one incorrect answer).</td></tr><tr><td>[REQUIREMENT]</td></tr><tr><td>1. Please ensure that the image information and the modality description are fully utilized, and make sure the answer cannot be inferred from the question alone.</td></tr><tr><td>2. Make the length of the incorrect answers similar to the correct answer to prevent that the correct answer is always the longest one.</td></tr><tr><td>3. Create diverse questions according to the question type, but please DO NOT completely copy the content of the example questions. You need to generated all kinds of questions related to the question type and the ability behind the question type.</td></tr></table>

```json
4. You can generate diverse questions requiring broad common knowledge, but ensure the connection between question and answer to be clear and reasonable. Avoid generating question and answers with a lot of guessing and assuming. However you can make up some extra knowledge in the modality description, it will be provided to subject.
5. Give your output in a format of JSON, which has the same structure like this:
[FORMAT] Your output MUST be in JSON format as follows:
{
    "strategy": "[HOW TO MAKE IT CHALLENGING]",
    "modality_description": "[MAKE UP SOME MEANINGS FOR THE COLOR PATTERN]",
    "question": "[YOUR QUESTION]",
    "answer": "[YOUR CORRECT ANSWER]",
    "ability_needed": "[ABILITY TESTED OF YOUR QUESTION TYPE]",
    "incorrect_answers": [
    "[INCORRECT ANSWER 1]",
    "[INCORRECT ANSWER 2]",
    "[INCORRECT ANSWER 3]", ...
    ]
}
```