# MLLM-DataEngine: Closing the Loop of Multimodal Instruction Tuning Data Generation

Zhiyuan Zhao<sup>∗</sup>, Bin Wang<sup>∗</sup>, Linke Ouyang<sup>∗</sup>, Yiqi Lin, Pan Zhang, Xiaoyi Dong, Jiaqi Wang, Conghui He<sup>†</sup>

Shanghai AI Laboratory

Shanghai, China

{zhaozhiyuan, wangbin, ouyanglinke, dongxiaoyi, heconghui}@pjlab.org.cn, wjqdev@gmail.com, linyq29@gmail.com

Abstract—In this paper, we propose MLLM-DataEngine, a novel closed-loop system that bridges data generation, model training, and evaluation. Within each loop iteration, the MLLM-DataEngine first analyzes the weakness of the model based on the evaluation results, then generates a proper incremental dataset for the next training iteration, and enhances the model capability iteratively. Compared with previous instruction finetuning dataset collection methods which are separate from the benchmarking, MLLM-DataEngine shows better targeting and can improve MLLMs’s capabilities more effectively. Firstly, we propose an Adaptive Bad-case Sampling module, which can effectively analyze model weakness based on the benchmarking results and adjust the generation of incremental datasets flexibly. Secondly, in order to ensure high-quality data for specific capability types, the most representative in-context examples and abundant information are provided to GPT-4, which helps GPT-4 fully comprehend the model’s weakness and further guarantees high-quality generated data. Through extensive experiments, we find MLLM-DataEngine could boost the MLLMs capability in a targeted and automatic manner without human participants. We hope MLLM-DataEngine could be a general solution for the following MLLMs data curation. Code, data, and model are available at https://github.com/opendatalab/MLLM-DataEngine.

Keywords—Multimodal Large Language Models, Instruction Tuning, Data Engine

## I. INTRODUCTION

The field of Multimodal Large Language Models (MLLMs) has greatly advanced recently [1]–[7]. To equip these models with vision-language understanding capabilities, a two-stage fine-tuning process becomes common practice [1]–[3]. In the first stage, the image-text feature alignment is inadequate to ensure robust multi-modal understanding capabilities. More importantly, the second stage utilizes high-quality annotated data for instruction fine-tuning, which is pivotal in guaranteeing exceptional model performance.

In the pursuit of high-quality multi-modal instruction tuning data, recent studies have begun to explore high-quality data generation. For example, several efforts [3], [8] have been undertaken to construct data from public datasets hand-craftly. In contrast, some recent advancements like ChatCaptioner [9] and IdeaGPT [10] using Large Language Models (LLMs) like ChatGPT [11] and a Vision-Language Model (VLM)

for creating caption data. For example, LLaVA [2] harnesses GPT-4 [11], a superior text model, with image annotation for multimodal data generation. Similarly, LRV [12] pre-defines over 20 tasks and combines them with image annotation, enabling GPT-4 to generate high-quality instruction fine-tuning data.

Despite their great efforts, current methods remain isolated from model evaluation and feedback, which severely hampers the opportunity for model improvement and fails to effectively mitigate model shortcomings. We argue that to improve model capabilities, the construction of high-quality instruction tuning data was supposed to be closely integrated with proficiency benchmarking. Unfortunately, combining benchmarking with data generation is still a remaining challenge. The current benchmarks for MLLMs such as SEED-Bench [13] and MM-Benchmark [14] provide comprehensive assessment of model capabilities and can point out the model weakness [15], while it is non-trivial to use it as guidance for model improvement, especially when the weakness includes several different aspects. A straightforward solution is to annotate or collect new data by humans, while its cost is quite large, especially when the models and benchmarks are updated iteratively.

To solve this problem, we propose MLLM-DataEngine to bridge instruct tuning data generation, model training, and evaluation. As shown in Fig. 1, different from previous approaches, our method introduces a closed-loop cycle of Evaluation-Guidance-Generation-Evaluation where model weaknesses from the evaluation phase are harnessed to guide the data generation process. Updating MLLMs in the loop can more effectively improve model capabilities and mitigate model weaknesses. Specifically: (1) Evaluation: To comprehensively identify model weakness, we collect bad cases of each fine-grained capability through wide-coverage evaluation and build a bad case pool in each iteration. (2) Guidance: As an essential connection between model evaluation and data generation, Adaptive Bad-case Sampling module (ABS) is proposed to select proper question types and in-context examples from the bad case pool according to model weakness, which helps guide further data generation. (3) Generation We utilize GPT-4 to generate diverse and accurate instruct-tuning data for each fine-grained capability. To make GPT-4 fully comprehend the weakness of the model, we feed the most representative in-context examples to GPT-4. Meanwhile, rich and sufficient information is provided to GPT-4 to ensure the correctness of the generated data. Our contributions are as follows:

![](images/926ffa1c8ea58d89ce5367da022486d0e5819628bba12fd011312147237265ad.jpg)  
Fig. 1: Comparison of existing methods and our proposed MLLM-DataEngine. Existing instruct tuning data generation method are separated from model evaluation and feedback. In contrast, our proposed MLLM-DataEngine is a closed-loop of generation training-evaluation-generation which leads to targeted and effective model improvement.

• We present MLLM-DataEngine, a multimodal engine that fosters a closed loop for data generation, model training, and evaluation, thus facilitating iterative improvement of model capabilities.

• Different from current methods, MLLM-DataEngine innovatively guides data generation using model feedback through a carefully designed process, which helps bridge the gap between model improvement and evaluation.

• We perform extensive experiments across various evaluation benchmarks. Results confirm MLLM-DataEngine’s ability to iteratively enhance model performance and compensate for model deficiencies.

## II. OUR APPROACH

The framework of MLLM-DataEngine, showcased in this paper, utilizes feedback from the model evaluation to guide the data generation and harnesses generated data to encounter the weakness of the model, which establishes a cyclical process for iterative enhancement between the model and data. As Fig. 2 illustrates, each iteration involves four steps:

• Model Evaluation. Firstly, The model’s capabilities are systematically evaluated across various dimensions. Then its bad cases are collected from identified weaknesses.

• Prompt Construction. Secondly, proper prompts targeted at model weakness are constructed through Adaptive Bad-case Sampling (ABS) for GPT-4 data generation.

• Data Generation. Thirdly, the prompt constructed in the previous step is fed into GPT-4 for data generation. Meanwhile, all generated data are carefully filtered and processed to ensure high-quality instruction fine-tuning.

• Model fine-tuning. Finally, the model is fine-tuned with newly generated data, and MLLM-DataEngine loops back to the first Model Evaluation step for new model performance evaluation and weakness identification.

## A. Model Evaluation

In the first step, we evaluate the model’s performance to identify its weaknesses, and we use bad cases (questions that the model answers incorrectly) as feedback to guide further data generation. In our proposed method, we utilize the open-source, Supervised, Fine-Tuned (SFT) Multimodal Large Language Models (MLLMs) as the initial model.

As for the evaluation benchmark, we choose SEED-Bench [13] as the evaluation benchmark, given that traditional singletask evaluation methods, such as VQA and Caption, cannot comprehensively and accurately assess the capabilities of MLLMs. SEED-Bench is a high-quality and generative evaluation system for multimodal large language models, which involves 19K multiple-choice questions with accurate human annotations. Moreover, SEED-Bench evaluation spans over 9 evaluation dimensions for image understanding, including multiple complex vision-language recognition and reasoning tasks (such as Instance Recognition, Text Recognition, Visual Reasoning, et al.), which ensures comprehensive capability evaluation and enhancement in our proposed MLLM-DataEngine.

After the model’s performance is comprehensively evaluated, in order to provide targeted guidance for the further data generation process, we collect those bad cases that are answered by MLLMs incorrectly and construct bad case pool. The bad case pool contains questions the model answered incorrectly in each type of SEED-Bench evaluation dimension and can reflect the model’s shortcomings and defects in specific capability dimensions. During the bad case pool construction, we find that a significant portion of the examples in the bad case pool is similar to each other (such as those about color and et al.). To make the examples in the bad case pool more representative and diverse, we adopt a simple yet effective duplicate-reduction process to iteratively remove duplicate questions and only keep the most representative examples (see Algorithm 1).

## B. Prompt Construction

After the bad case pool is established, we construct delicate prompts for further data generation, which is composed of two key components: (1) Detailed image information described in language and (2) Well-demonstrated question type. Each component plays a crucial role in the generation of highquality data.

![](images/b970eb993cebac35bc76eee00ecefe123bf8d3260faf12290876caf7035c9988.jpg)  
Fig. 2: Illustration of proposed MLLM-DataEngine. The whole process is divided into 4 steps. (1) Model Evaluation. We first test the base model on the public benchmark to get the bad cases and build the bad case pool. (2) Prompt Construction. After bad cases are obtained, Adaptive Bad-case Sampling (ABS) is proposed to select the proper question type and the most representative in-context examples. Meanwhile, rich image information is provided. (3) Data Generation. Constructed prompt are fed to GPT-4 to generate data. (4) Model Training. The model is fine-tuned on the latest generated data and loops back to the beginning of the data engine

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Reduce Duplicates in the Bad Case Pool
Input: P - Original set of bad cases.
Output: Q - Reduced bad cases.
1: Initialize  $Q \leftarrow \emptyset$ 
2: while  $P \neq \emptyset$  do
3:  $P_{s} = \text{RandomSelect}(P)$ 
4:  $Q \leftarrow Q \cup P_{s}$ 
5: for  $P_{i}$  in P do
6: if spacy( $P_{i}, P_{s}$ ) &gt; 0.9 then
7:  $P \leftarrow P.\text{remove}(P_{i})$ 
8: end if
9: end for
10: end while
11: return Q
</div>

Firstly, for detailed image information, we randomly choose images from Visual Genome (VG) Dataset [17], which provides a range of annotations for each image, such as Region Descriptions, Object Instances, and Object Attributes, et al..

Among those, Region Descriptions, which contain multiple instances along with detailed descriptions of objects as well as their position coordinates, are extracted as Image Information. Moreover, for the Text Understanding task, we employ PaddleOCR [18] to extract texts that appear in images, which serves as an auxiliary supplement to the image information.

Secondly, to clearly demonstrate the specific question type expected to be generated from GPT-4, a clear definition of this question type and in-context learning examples are needed. To ensure the effective selection of appropriate in-context examples and to align the generated data with the model’s shortcomings, we introduce the Adaptive Bad-case Sampling (ABS) strategy. It comprises two steps: (1) A question type is chosen randomly based on an adaptive sampling ratio. The sampling ratio for each question type is based on the corresponding evaluation dimensions scores, which is represented by $r _ { i } = \sqrt { 1 - a _ { i } } .$ , where $r _ { i }$ is the sampling ratio of $i ^ { t h }$ question type and $a _ { i }$ denotes the accuracy attained by MLLM in $i ^ { t h }$ question type. (2) Ten instances of this question type are then randomly selected from the bad case pool to serve as in-context examples, assisting GPT-4 in understanding the required question type effectively and guiding GPT-4 to generate valuable data. ABS ensures the flexibility and [Question] Count the number of dogs visible in the image. [Choices] (A) None (B) One (C) Two (D) Three [Answer] There is only one dog visible in the image. [Choice Answer] B

[Question] What is placed on the edge of the bathtub? [Choices] (A) A soap dish (B) An air vent (C) Towels (D) A bath mat [Answer] There are two white towels and a pair of towels placed on the edge of the bathtub. [Choice Answer] C

![](images/b8d2de8c483c2a0eddc69d2db9c944b5219238dd064a102d0003b09df3db40af.jpg)

[Question] What does the visual context suggest about the owner of the items on the table? [Choices] (A) A college student (B) A reader of children's literature (C) A professional chef (D) A carpenter [Answer] Given the presence of the book "The Tiger Came to Tea", it’s a children's literature reader. [Choice Answer] B

[Question] What language is the text on the sign in? [Choices] (A) English (B) Spanish (C) 2 different languages (D) French [Answer] The text on the sign is in 2 different languages. [Choice Answer] C

Fig. 3: Example of generated data for four out of nine fine-grained abilities.  
TABLE I: Experiments on LLaVA-1.5. The None (baseline) row indicates the baseline model results without incremental dataset in instruction fine-tuning.

<table><tr><td>Incremental Dataset</td><td>Data Amount</td><td> $SEED^I$ </td><td> $MMB^{Dev}$ </td><td>MME</td><td>GQA</td><td>VizWiz</td><td>VQAv2</td><td>ScienceQA</td></tr><tr><td>None (baseline)</td><td>-</td><td>66.04</td><td>66.66</td><td>1475/290(1765)</td><td>57.27</td><td>49.07</td><td>77.56</td><td>70.67/68.27</td></tr><tr><td>LRV [12]</td><td>320k</td><td>67.71</td><td>67.61</td><td>1462/304(1766)</td><td>58.23</td><td>50.41</td><td>78.22</td><td>71.80/69.46</td></tr><tr><td>SVIT [16]</td><td>1.5M</td><td>68.22</td><td>66.66</td><td>1415/282(1697)</td><td>58.55</td><td>46.96</td><td>78.91</td><td>70.34/68.72</td></tr><tr><td>DataEngine, Round1</td><td>80k</td><td>67.72</td><td>67.47</td><td>1484/270(1754)</td><td>57.33</td><td>51.26</td><td>77.72</td><td>72.41/71.29</td></tr><tr><td>DataEngine, Round2</td><td>170k</td><td>68.30</td><td>67.61</td><td>1486/286(1772)</td><td>57.74</td><td>48.95</td><td>78.18</td><td>72.37/71.19</td></tr><tr><td rowspan="2">DataEngine, Round3</td><td rowspan="2">220k</td><td>68.57</td><td>67.18</td><td>1511/303(1814)</td><td>58.02</td><td>52.90</td><td>78.18</td><td>73.17/71.15</td></tr><tr><td>↑2.53</td><td>↑0.52</td><td>↑36/13(49)</td><td>↑0.75</td><td>↑3.83</td><td>↑0.62</td><td>↑2.50/2.88</td></tr></table>

adaptability of data generation and more data will be generated for dimensions where the model’s performance is poorer. Consequently, ABS effectively guides the generation of data to address the weaknesses of the model.

## C. Data Generation

With the prompt constructed, we utilize gpt-4-1106-preview version of GPT-4 to generate instruct fine-tuning data. During data generation, GPT-4 is prompted to construct diverse and complex questions, along with corresponding accurate Direct Answers (DA) based on the provided information. In addition to direct answers, GPT-4 is also instructed to reformat questions into a Multiple-Choice format (MC), comprising four options (A, B, C, and D) and a correct answer. Meanwhile, to further enhance the data diversity, we randomly shuffle options in each multi-choice question. Example generated data are shown in Fig. 3.

## III. EXPERIMENTS

## A. Implementation Details

We adopt three mainstream, open-source MLLMs in experiments: LLaVA-1.5 [7], MiniGPT4-v2 [19], and MiniGPT-4 [1]. We incorporate these MLLMs into MLLM-DataEngine for iterative refinement. Specifically, we use both the original instruct-tuning data and MLLM-DataEngine generated incremental data in the instruct tuning of each iteration. The refined model is evaluated using various downstream benchmarks for a comprehensive assessment. Details can be found in supplementary materials.

## B. Main Results

In this section, we conduct multiple rounds of model refinement using our proposed MLLM-DataEngine to validate its effectiveness. In each round of refinement in MLLM-DataEngine, we combine the incremental data generated by MLLM-DataEngine (data generated in the current and the previous rounds) with the original instruction fine-tuning data for fine-tuning.

Experiments are conducted on LLaVA-1.5 and MiniGPT4- v2, whose results are demonstrated in Table I and Table II, respectively. The None (baseline) row indicates baseline model results without incremental dataset. SEED<sup>I</sup> refers to SEED-Bench (Image). MMB<sup>Dev</sup> refers to MMBench Dev. MME score stands for Perception/Cognition (total score). ScienceQA score stands for Overall score/Image score. Besides, we compare our proposed MLLM-DataEngine with two comparison methods: LRV [12] and SVIT [16], which also seek to enhance MLLMs by scaling up instruction fine-tuning data. From the results, we can see that: (1) MLLM-DataEngine can effectively and iteratively enhance model performance. Results clearly show that the model performance increases in each round of MLLM-DataEngine across various benchmarks as the generated data

TABLE II: Experiments on MiniGPT4-v2. The None (baseline) row indicates the baseline model results.

<table><tr><td>Incremental Dataset</td><td>Data Amount</td><td> $SEED^I$ </td><td> $MMB^{Dev}$ </td><td>OK-VQA</td><td>VizWiz</td><td>VSR</td></tr><tr><td>None (baseline)</td><td>-</td><td>49.21</td><td>38.83</td><td>56.03</td><td>53.08</td><td>61.37</td></tr><tr><td>LRV [12]</td><td>320k</td><td>49.24</td><td>40.72</td><td>56.66</td><td>54.30</td><td>61.78</td></tr><tr><td>SVIT [16]</td><td>1.5M</td><td>49.75</td><td>43.64</td><td>56.95</td><td>54.31</td><td>58.67</td></tr><tr><td>DataEngine, Round1</td><td>100k</td><td>61.84</td><td>51.80</td><td>56.76</td><td>53.50</td><td>62.09</td></tr><tr><td>DataEngine, Round2</td><td>180k</td><td>63.80</td><td>53.43</td><td>57.05</td><td>53.62</td><td>62.52</td></tr><tr><td rowspan="2">DataEngine, Round3</td><td rowspan="2">270k</td><td>63.83</td><td>52.92</td><td>56.87</td><td>54.39</td><td>62.43</td></tr><tr><td>↑14.62</td><td>↑14.09</td><td>↑0.74</td><td>↑1.31</td><td>↑1.06</td></tr></table>

![](images/de12099bce9eb52d0e1f6352bd3a675c6fecf763e934b0b823235d64b313ef40.jpg)  
Fig. 4: Comparison between uniform sampling and Adaptive Bad-case Sampling (ABS). Weak capabilities of the baseline model are highlighted.

TABLE III: Experiments of different synthetic data formats on LLaVA-1.5. DA refers to direct answer and MC refers to multiple choice answer.

<table><tr><td>Incremental Dataset</td><td>Data Format</td><td>SEEDI</td><td>MMBDev</td><td>GQA</td></tr><tr><td>None (baseline)</td><td>-</td><td>66.04</td><td>66.66</td><td>57.27</td></tr><tr><td rowspan="2">DataEngine, Round1</td><td>DA</td><td>66.21</td><td>67.26</td><td>58.21</td></tr><tr><td>DA + MC</td><td>67.72</td><td>67.47</td><td>57.33</td></tr><tr><td rowspan="2">DataEngine, Round2</td><td>DA</td><td>67.25</td><td>67.18</td><td>58.19</td></tr><tr><td>DA + MC</td><td>68.30</td><td>67.61</td><td>57.74</td></tr><tr><td rowspan="2">DataEngine, Round3</td><td>DA</td><td>67.33</td><td>66.15</td><td>58.60</td></tr><tr><td>DA + MC</td><td>68.57</td><td>67.18</td><td>58.02</td></tr></table>

amount grows. For instance, three rounds of refinement on LLaVA-1.5 result in improvements of 2.53% for SEED-Bench, 0.52% for MMBench Dev, and a total score increase of 49 for MME when compared to baseline. Additionally, superior results are also achieved on traditional VQA benchmarks under MLLM-DataEngine refinement. (2) MLLM-DataEngine presents a more effective data generation strategy. MLLM-DataEngine achieves significantly better improvements by using a more targeted but lesser volume of data compared with LRV and SVIT, which produce 320k and 1.5M instruction fine-tuning data for model enhancement respectively. This indicates the superiority of MLLM-DataEngine in terms of data generation strategy, as well as the effectiveness of the underlying iterative refinement philosophy. (3) The improvement brought by data scaling is not unlimited. Comparing results between 3 rounds of refinement in MLLM-DataEngine, it can be seen that: In the beginning, there was a significant improvement through data scaling. However, as the volume of data grows to a certain extent, The trade-off between data scaling and model improvements is becoming increasingly smaller. For example, MiniGPT4-v2’s third round of refinement demonstrated significantly less improvement compared to the first and second rounds. We speculate that further improvement is constrained due to potential model limitations such as input resolution, visual feature granularity, and so on.

## C. Ablation Studies

1) Comparison between ABS and Uniform Sampling: To further validate the ability of Adaptive Bad-case Sampling (ABS) to enhance model weakness, we compared its effects with uniform sampling. For comparison, we directly sample from all generated data, using both ABS and uniform sampling, with each method fixedly sampling 90K data. We finetune the baseline MiniGPT4-v2 model using data obtained from those two sampling strategies separately. Results on SEED-Bench are shown in Fig. 4 (detailed result in supplementary). Results demonstrate that for the weaknesses of the baseline model, our proposed Adaptive Bad-case Sampling (ABS) can provide a more targeted enhancement.

2) Effects of Different Instruction Tuning Data Formats: To explore the effects of different data formats, we experiment with different formats on LLaVA-1.5, and results are shown in Table III. DA refers to direct answer and MC refers to multiple choices question. Firstly, regardless of the format, significant performance improvements can be achieved by generated data. For example, the three-round DA also has a significant improvement compared to the baseline model. Secondly, the choice of data format influences performance across different benchmarks. For example, the multiple choices (MC) leads to substantial improvements on challenging multiple-choice benchmarks such as SEED and MMBench when compared to Direct Answer (DA); however, on reasoning VQA benchmarks, using only DA yields the best results (58.60% accuracy on GQA).

TABLE IV: MiniGPT4-v2 (A) → MiniGPT4 (B).

<table><tr><td>Incremental Dataset</td><td>SEED $^{\text{I}}$ </td><td>MMB $^{\text{Dev}}$ </td></tr><tr><td>None, baseline</td><td>21.30</td><td>23.00</td></tr><tr><td>MiniGPT4-v2, Round1</td><td>54.31</td><td>38.91</td></tr><tr><td>MiniGPT4-v2, Round2</td><td>56.80</td><td>46.30</td></tr><tr><td>MiniGPT4-v2, Round3</td><td>58.12</td><td>49.31</td></tr></table>

TABLE V: LLaVA-1.5 (A) → MiniGPT4 (B).

<table><tr><td>Incremental Dataset</td><td>SEED $^{\text{I}}$ </td><td>MMB $^{\text{Dev}}$ </td></tr><tr><td>None, baseline</td><td>21.30</td><td>23.00</td></tr><tr><td>LLaVA-1.5, Round1</td><td>51.96</td><td>39.00</td></tr><tr><td>LLaVA-1.5, Round2</td><td>57.11</td><td>48.45</td></tr><tr><td>LLaVA-1.5, Round3</td><td>58.34</td><td>45.79</td></tr></table>

3) Generalization Ability of Generated Data: Lastly, we explore the generalization capability of the data generated by MLLM-DataEngine. More specifically, we address the question, “Is the data generated for Model A also beneficial for Model B?”. We investigate this by using model A’s MLLM-DataEngine generated data to instruction finetune model B and examining subsequent performance improvements (represented as A→B). Table IV and Table V demonstrate the results of fine-tuning MiniGPT-4 (model B) using the data generated with MiniGPT4-v2 and LLaVA-1.5 in the MLLM-DataEngine loop, respectively (model A). MLLM-DataEngine generated data for MiniGPT4-v2 and LLaVA-1.5 achieves 36.82%/26.31% and 37.04%/22.79% improvement in MiniGPT4, respectively. We further extend this investigation with cross-validation on two strong MLLMs, MiniGPT4-v2 and LLaVA-1.5. Results are shown in Table VI and Table VII, which also exhibit performance improvements, confirming that the data generated by MLLM-DataEngine is not only modelspecific but also broadly applicable across different MLLMs.

## IV. CONCLUSION

This paper presents MLLM-DataEngine, a framework for generating high-quality, targeted instruction fine-tuning data, addressing model weaknesses, and forming a closed training loop for large multi-modal models. We hope this approach will advance data construction on multi-modal research.

## REFERENCES

[1] Deyao Zhu, Jun Chen, Xiaoqian Shen, and et al., “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” arXiv preprint arXiv:2304.10592, 2023.

TABLE VI: LLaVA-1.5 (A) → MiniGPT4-v2 (B).

<table><tr><td>Incremental Dataset</td><td>SEED $^{\text{I}}$ </td><td>MMB $^{\text{Dev}}$ </td></tr><tr><td>None, baseline</td><td>49.21</td><td>38.83</td></tr><tr><td>LLaVA-1.5, Round1</td><td>59.81</td><td>48.71</td></tr><tr><td>LLaVA-1.5, Round2</td><td>62.72</td><td>51.67</td></tr><tr><td>LLaVA-1.5, Round3</td><td>63.09</td><td>52.14</td></tr></table>

TABLE VII: MiniGPT4-v2 (A) → LLaVA-1.5 (B).

<table><tr><td>Incremental Dataset</td><td>SEED $^{\text{I}}$ </td><td>MMB $^{\text{Dev}}$ </td></tr><tr><td>None, baseline</td><td>66.04</td><td>66.66</td></tr><tr><td>MiniGPT4-v2, Round1</td><td>67.40</td><td>66.23</td></tr><tr><td>MiniGPT4-v2, Round2</td><td>67.95</td><td>66.87</td></tr><tr><td>MiniGPT4-v2, Round3</td><td>68.46</td><td>67.04</td></tr></table>

[2] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee, “Visual instruction tuning,” in NeurIPS, 2024.

[3] Wenliang Dai, Junnan Li, Dongxu Li, and et al., “Instructblip: Towards general-purpose vision-language models with instruction tuning,” arXiv, 2023.

[4] Zhiliang Peng, Wenhui Wang, Li Dong, and et al., “Kosmos-2: Grounding multimodal large language models to the world,” arXiv preprint arXiv:2306.14824, 2023.

[5] Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, and et al., “Flamingo: a visual language model for few-shot learning,” in NeurIPS, 2022.

[6] Xiaoyi Dong, Pan Zhang, Yuhang Zang, and et al., “Internlmxcomposer2: Mastering free-form text-image composition and comprehension in vision-language large model,” arXiv preprint arXiv:2401.16420, 2024.

[7] Haotian Liu, Chunyuan Li, Yuheng Li, and et al., “Improved baselines with visual instruction tuning,” arXiv preprint arXiv:2310.03744, 2023.

[8] Bo Li, Yuanhan Zhang, Liangyu Chen, and et al., “Mimic-it: Multimodal in-context instruction tuning,” arXiv preprint arXiv:2306.05425, 2023.

[9] Deyao Zhu, Jun Chen, Kilichbek Haydarov, and et al., “Chatgpt asks, blip-2 answers: Automatic questioning towards enriched visual descriptions,” arXiv preprint arXiv:2303.06594, 2023.

[10] Haoxuan You, Rui Sun, Zhecan Wang, and et al., “Idealgpt: Iteratively decomposing vision and language reasoning via large language models,” arXiv preprint arXiv:2305.14985, 2023.

[11] Josh Achiam, Steven Adler, Sandhini Agarwal, , and et al., “Gpt-4 technical report,” arXiv preprint arXiv:2303.08774, 2023.

[12] Fuxiao Liu, Kevin Lin, Linjie Li, and et al., “Mitigating hallucination in large multi-modal models via robust instruction tuning,” in ICLR, 2023.

[13] Bohao Li, Rui Wang, Guangzhi Wang, and et al., “Seed-bench: Benchmarking multimodal llms with generative comprehension,” arXiv preprint arXiv:2307.16125, 2023.

[14] Yuan Liu, Haodong Duan, Yuanhan Zhang, and et al., “Mmbench: Is your multi-modal model an all-around player?,” arXiv preprint arXiv:2307.06281, 2023.

[15] Alexander Kirillov, Eric Mintun, Nikhila Ravi, and et al., “Segment anything,” arXiv preprint arXiv:2304.02643, 2023.

[16] Bo Zhao, Boya Wu, and Tiejun Huang, “Svit: Scaling up visual instruction tuning,” arXiv preprint arXiv:2307.04087, 2023.

[17] Ranjay Krishna, Yuke Zhu, Oliver Groth, and et al., “Visual genome: Connecting language and vision using crowdsourced dense image annotations,” IJCV, 2017.

[18] Yuning Du, Chenxia Li, Ruoyu Guo, and et al., “Pp-ocr: A practical ultra lightweight ocr system,” arXiv preprint arXiv:2009.09941, 2020.

[19] Jun Chen, Deyao Zhu, Xiaoqian Shen, and et al., “Minigpt-v2: large language model as a unified interface for vision-language multi-task learning,” arXiv preprint arXiv:2310.09478, 2023.

# Supplementary Materials for “MLLM-DataEngine: Closing the Loop of Multimodal Instruction Tuning Data Generation”

![](images/921d766e848fb60c0f2c5735fc36b0e629aeb57f309a054fa1a79bfe07ed18bb.jpg)  
Fig. 1: Results of each capability of MiniGPT4-v2 during the refinement process of MLLM-DataEngine.

In Section I, we provide prompts used in MLLM-DataEngine data generation. In Section II, we provide implementation details. In Section III, we provide anylysis on MLLM-DataEngine generated data. In Section IV, we provide detailed results during each optimization ieration of MLLM-DataEngine. In Section V, we provide eaxmple MLLM-DataEngine generated data.

## I. PROMPTS USED IN MLLM-DATAENGINE

Fig. 3 is the prompt template used in MLLM-DataEngine when generating data with GPT-4. Among it, Region Description is extracted from Visual Genome (VG) Dataset. OCR Result is the output of PaddleOCR when generating Text Understanding questions. Question Types and its Definition are acquire from SEED Benchmark. In-Context Examples are randomly selected from Bad Case Pool of MLLMs in that question type.

## II. IMPLEMENTATION DETAILS

## A. LLaVA-1.5

is the SOTA (state-of-the-art) open-source MLLM with strong capabilities in image content recognition and understanding. The instruction fine-tuning data covering openknowledge VQA [1]–[4], OCR [5], [6], conversation [7], [8], and region-level VQA [9]–[11], leading to a total of 665k instruction fine-tuning data. Evaluations are conducted on comprehensive multi-dimensional benchmarks (SEED-Bench [12], MMBench [13], MME [14]) and conventional VQA datasets (VQAv2, GQA, VizWiz [15], ScienceQA [16]). As for the experimental setting, we adopt LLaVA-1.5-7b-lora version of LLaVA-1.5, where the model is fine-tuned for 1 epoch using LoRA [17] with rank set to 128 and alpha set to 256, following its original setting. The cosine learning rate scheduler is adopted with an initial learning rate set to $1 e ^ { - 4 }$ and a warm-up ratio set to 0.03. The model is fine-tuned using 8×A100 GPUs for about 20 hours.

## B. MiniGPT4-v2

MiniGPT4-v2 is another strong-performed MLLM that is fine-tuned using abundant multi-task instruction data. The multimodel instruction fine-tuning data of MiniGPT4-v2 consists of multiple tasks: image captioning [6], [18], referring expression comprehension/generation [10], [11], [19], VQA [1]– [5], and multimodel conversation [7], [20]. During evaluation, except from three VQA datasets used in the original setting (OKVQA [2], VizWiz [15], VSR [21]), we carry out evaluations on SEED-Bench and MMBench. As for the experimental setting, we follow the original setting and finetune the model for 10/20/30 epochs with 1k iterations per epoch as the amount of data grows in each round of MLLM-DataEngine refinement. LoRA with rank set to 16 and alpha set to 64 is used. The cosine learning rate scheduler is adopted with an initial learning rate set to $\bar { 1 } e ^ { - 5 }$ and a warm-up step set to 1k. The model is fine-tuned using 8×A100 GPUs for about 10 hours.

## C. MiniGPT4

MiniGPT4 the preliminary version of MiniGPT4-v2, has basic image comprehension and instruction-following abilities but is inferior across various capability dimensions. The model is fine-tuned using 3.5k filtered image captioning data [22]. SEED-Bench and MMBench are adopted as evaluation benchmarks. In terms of the experimental setup, we employ LoRA for one epoch with rank set to 16 and alpha to 64. The model uses a cosine learning rate scheduler with an initial rate of $1 e ^ { - 5 }$ and 1k warm-up step, fine-tuned on 8×A100 GPUs over approximately 2-3 hours.

TABLE I: SEED-Bench results of LLaVA-1.5 during MLLM-DataEngine refinement.

<table><tr><td>Incremental Dataset</td><td>Scene Understanding</td><td>Instance Identity</td><td>Instance Attributes</td><td>Instance Location</td><td>Instance Counting</td><td>Spatial Relation</td><td>Instance Interaction</td><td>Visual Reasoning</td><td>Text Understanding</td><td>Overall</td></tr><tr><td>None (baseline)</td><td>74.89</td><td>68.32</td><td>66.87</td><td>60.94</td><td>57.42</td><td>49.92</td><td>73.20</td><td>76.13</td><td>27.06</td><td>66.04</td></tr><tr><td>Data-Engine, round1</td><td>73.75</td><td>69.52</td><td>70.02</td><td>59.30</td><td>59.95</td><td>54.34</td><td>71.13</td><td>76.13</td><td>47.06</td><td>67.22</td></tr><tr><td>Data-Engine, round2</td><td>74.57</td><td>71.00</td><td>71.67</td><td>60.63</td><td>58.52</td><td>53.42</td><td>69.07</td><td>75.83</td><td>67.06</td><td>68.30</td></tr><tr><td>Data-Engine, round3</td><td>74.32</td><td>70.51</td><td>72.10</td><td>62.17</td><td>58.19</td><td>55.10</td><td>72.16</td><td>77.04</td><td>60.00</td><td>68.57</td></tr></table>

TABLE II: SEED-Bench results of MiniGPT4-v2 during MLLM-DataEngine refinement.

<table><tr><td>Incremental Dataset</td><td>Scene Understanding</td><td>Instance Identity</td><td>Instance Attributes</td><td>Instance Location</td><td>Instance Counting</td><td>Spatial Relation</td><td>Instance Interaction</td><td>Visual Reasoning</td><td>Text Understanding</td><td>Overall</td></tr><tr><td>None (baseline)</td><td>63.39</td><td>54.72</td><td>47.41</td><td>43.46</td><td>36.41</td><td>36.38</td><td>57.73</td><td>66.77</td><td>32.94</td><td>49.66</td></tr><tr><td>Data-Engine, round1</td><td>71.03</td><td>65.10</td><td>64.70</td><td>52.25</td><td>48.55</td><td>43.53</td><td>68.04</td><td>75.23</td><td>68.24</td><td>61.84</td></tr><tr><td>Data-Engine, round2</td><td>71.79</td><td>68.10</td><td>66.72</td><td>56.13</td><td>51.04</td><td>45.05</td><td>68.04</td><td>73.72</td><td>71.76</td><td>63.80</td></tr><tr><td>Data-Engine, round3</td><td>72.39</td><td>67.56</td><td>66.90</td><td>55.42</td><td>50.84</td><td>45.05</td><td>65.98</td><td>74.02</td><td>71.76</td><td>63.83</td></tr></table>

![](images/ebe58b239fe65d95cd1839779f57118f24a57bdc7e61089780305daa8cb3bb5f.jpg)  
(a) Visual Reasoning

![](images/0bfb45c395f616e70e2160ae6d515f4cd064cea854b587860b2307c256dc03c0.jpg)  
(b) Spatial Relation

![](images/33baab7c76bffecaf957caf7fcfdf4568b38dd9230e3030a9645e477edcd2670.jpg)  
(c) Object Counting  
Fig. 2: Words distributions analysis. Generated instruct data aligns closely to the given question type.

## III. DATA STATISTICS AND ANALYSIS

The average lengths of the generated questions and answers are 67.60 and 76.81 characters, respectively and questions contain both interrogative and declarative sentences. To validate whether the generated instructions align with the given question type, we further analyze the distribution of the first and second words in the generated questions. As shown in Fig. 2, the inner circle of the diagram represents the first word of the question, while the outer circle signifies the second word. It can be concluded that the generated data accurately reflects the given capability dimensions. For instance, Visual Reasoning requires an accurate comprehension of image content and context, hence questions like “What can”, “What might”, and “What could” are prevalent.

## IV. DETAILED MLLM-DATAENGINE RESULTS

Detailed SEED-Bench results in each round of LLaVA-1.5 and MiniGPT4-v2 are shown in Table I and Table II, respectively. Results of each capability of MiniGPT4-v2 during the refinement process of MLLM-DataEngine is illustrated in Fig. 1.

## V. QUALITY EXAMPLES IN MLLM-DATAENGINEGENERATED DATA

High-quality and diverse examples generated by MLLM-DataEngine are demonstrated. Fig. 4 are examples of Scene Understanding, instance Identity, and Instance Attribute. Fig. 5 are examples of Instance Localization, Instance Counting, and Spatial Relation. Fig. 6 are examples of Instance Interaction, Visual Reasoning, and Text Recognition.

## REFERENCES

[1] Yash Goyal, Tejas Khot, Douglas Summers-Stay, and et al., “Making the V in VQA matter: Elevating the role of image understanding in visual question answering,” in CVPR, 2017.

[2] Kenneth Marino, Mohammad Rastegari, Ali Farhadi, and et al., “OK-VQA: A visual question answering benchmark requiring external knowledge,” in CVPR, 2019.

[3] Dustin Schwenk, Apoorv Khandelwal, Christopher Clark, and et al., “A-OKVQA: A benchmark for visual question answering using world knowledge,” in ECCV, 2022.

[4] Drew A. Hudson and Christopher D. Manning, “GQA: A new dataset for real-world visual reasoning and compositional question answering,” in CVPR, 2019.

[5] Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and et al., “OCR-VQA: visual question answering by reading text in images,” in ICDAR, 2019.

[6] Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and et al., “Textcaps: A dataset for image captioning with reading comprehension,” in ECCV, 2020.

[7] Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee, “Visual instruction tuning,” in NeurIPS, 2024.

[8] “Sharegpt,” 2023, https://sharegpt.com/.

[9] Ranjay Krishna, Yuke Zhu, Oliver Groth, and et al., “Visual genome: Connecting language and vision using crowdsourced dense image annotations,” IJCV, 2017.

[10] Junhua Mao, Jonathan Huang, Alexander Toshev, and et al., “Generation and comprehension of unambiguous object descriptions,” in CVPR, 2016.

[11] Sahar Kazemzadeh, Vicente Ordonez, Mark Matten, and et al., “Referitgame: Referring to objects in photographs of natural scenes,” in EMNLP, 2014.

[12] Bohao Li, Rui Wang, Guangzhi Wang, and et al., “Seed-bench: Benchmarking multimodal llms with generative comprehension,” arXiv preprint arXiv:2307.16125, 2023.

[13] Yuan Liu, Haodong Duan, Yuanhan Zhang, and et al., “Mmbench: Is your multi-modal model an all-around player?,” arXiv preprint arXiv:2307.06281, 2023.

[14] Chaoyou Fu, Peixian Chen, Yunhang Shen, and et al., “Mme: A comprehensive evaluation benchmark for multimodal large language models,” arXiv preprint arXiv:2306.13394, 2023.

[15] Danna Gurari, Qing Li, Abigale J. Stangl, and et al., “Vizwiz grand challenge: Answering visual questions from blind people,” in CVPR, 2018.

[16] Pan Lu, Swaroop Mishra, Tanglin Xia, and et al., “Learn to explain: Multimodal reasoning via thought chains for science question answering,” in NeurIPS, 2022.

[17] Edward J. Hu, Yelong Shen, Phillip Wallis, and et al., “Lora: Low-rank adaptation of large language models,” in ICLR, 2022.

[18] Tsung-Yi Lin, Michael Maire, Serge J. Belongie, and et al., “Microsoft COCO: common objects in context,” in ECCV, 2014.

[19] Licheng Yu, Patrick Poirson, Shan Yang, and et al., “Modeling context in referring expressions,” in ECCV, 2016.

[20] Bryan A. Plummer, Liwei Wang, Chris M. Cervantes, and et al., “Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models,” in ICCV, 2015.

[21] Fangyu Liu, Guy Emerson, and Nigel Collier, “Visual spatial reasoning,” Transactions of the Association for Computational Linguistics, 2023.

[22] Deyao Zhu, Jun Chen, Xiaoqian Shen, and et al., “Minigpt-4: Enhancing vision-language understanding with advanced large language models,” arXiv preprint arXiv:2304.10592, 2023.

![](images/d828e3ec1d32e15b7658d9c176af5cbbb1a09a0664f85a2c346c106958cbd6f8.jpg)  
Fig. 3: Prompt Template for data generation in MLLM-DataEngine.

[Scene Understanding]

[Question] Identify the type of event happening in the image. [Choices] (A) Car show (B) Biker's convention (C) Marathon (D) Street fair [Answer] Based on the large number of bikes and people, it appears to be a biker's convention.

[Question] What is happening in the background of the image? [Choices] (A) A truck is unloading cargo from a plane (B) A truck is loading cargo onto a plane (C) There is a clear, sunny day in the background (D) There are trees without any fog in the background [Answer] There is a truck loading cargo onto a large commercial airplane, and there is fog obscuring trees in the distance. [Choice Answer] A

[Question] What is the main activity that appears to be taking place in this image? [Choices] (A) Hiking (B) Surfing (C) Swimming (D) Picnicking [Answer] The main activity in the image is surfing as evidenced by the presence of multiple surfboards. [Choice Answer] B

## [Instance Identity]

[Question] What are the two animals standing next to each other in the image? [Choices] (A) Cows (B) Horses (C) Sheep (D) Goats [Answer] The two animals standing next to each other in the image are sheep. [Choice Answer] C

[Question] Describe the main object in the center of the image. [Choices] (A) A car (B) A boat (C) A plane (D) A train [Answer] The main object in the image is a black train engine on display in a park. [Choice Answer] D

![](images/41be1403bd1dad515955a02ef59c022ea6a30b904eb8965ceb912034a3045c21.jpg)

[Question] What is the man in the image looking at? [Choices] (A) Television (B) Cell phone (C) Basketball player on TV (D) Picture on wall [Answer] The man in the image is looking at his cell phone. [Choice Answer] B

[Instance Attribute]

[Question] What is the color of the bird's neck? [Choices] (A) White (B) Gray (C) Black (D) Brown [Answer] The bird's neck is white. [Choice Answer] A

[Question] What is the color of the frisbee that the man is holding? [Choices] (A) Blue (B) Red (C) White (D) Green [Answer] The frisbee that the man is holding is white. [Choice Answer] C

[Question] Please describe the shape and style of the clock on the building. [Choices] (A) Circular clock with Roman numerals (B) Square clock with Arabic numerals (C) Diamond-shaped clock with Arabic numerals (D) Diamond-shaped clock with Roman numerals [Answer] The clock on the building is diamond shaped and features Roman numerals [Choice Answer] D

Fig. 4: Examples of MLLM-DataEngine generated data in Scene Understanding, Instance Identity, and Instance Attribute.

![](images/565af75698c3f22f961536765f6178217b6308fa04f649159374b7c5535bf5f6.jpg)

## [Instance Localization]

[Question] Where is the baseball player who is swinging located in the image? [Choices] (A) Top left (B) Bottom right (C) Right center (D) Left center [Answer] The baseball player swinging is located towards the right side in the image, close to the center but slightly towards the upper side. [Choice Answer] C

![](images/15003943d2b1775e2956021b9423aadf060f5a8b26ae7559b467b2fa31ce83e8.jpg)

[Question] Can you find the location of the sun that is shining through the clouds? [Choices] (A) Bottom Left (B) Top Left (C) Top Right (D) Bottom Right [Answer] The sun shining through the clouds is located towards the right side of the image, more towards the top. [Choice Answer] C

![](images/b2db05279921ccf9320023b362caef1f12295ad7166d88fd9e176e08c8acfc07.jpg)

[Question] Where is the pink bow on one of the figurines located? [Choices] (A) Upper Left (B) Bottom Right (C) Upper Right (D) Bottom Left [Answer] The pink bow on one of the figurines is located at the upper right section of the image. [Choice Answer] C

## [Instance Counting]

[Question] How many people are there in the image? [Choices] (A) One person (B) Two people (C) Three people (D) Four people [Answer] There are two people in the image. [Choice Answer] B

[Question] Determine the number of people standing behind the bus. [Choices] (A) One (B) Two (C) Three (D) None [Answer] There is one person standing behind the bus. [Choice Answer] A

![](images/87b11ea214ae9fbe404647e746461d9040d55a032e7ac3ec857f8ab3ce97120c.jpg)

![](images/cdd4ebd3bed11bb2bd120ba5ea98afef98d68fe3b9adc21fd21fe0cf3523bd25.jpg)

[Question] How many men in the image are smiling? [Choices] (A) None (B) One (C) Two (D) Three [Answer] At least one man in the image is smiling. [Choice Answer] B

![](images/8eabac87a29332a40841e140bb85d4813b6e41f9253c1ae49d297a45446f6b1b.jpg)

[Spatial Relation]

![](images/cda08e6254c766d9dc2d297d1b0993dc86f16f40eb031a03f42b99f320e7b253.jpg)

[Question] Describe the position of the fork in relation to the bagel. [Choices] (A) The fork is to the right of the bagel (B) The fork is underneath the bagel (C) The fork is to the top left of the bagel (D) The fork is on top of the bagel [Answer] The fork is positioned to the top left of the bagel. [Choice Answer] C

![](images/56240aa9e6cee12543e909dcf83b3e3132ea37b6790f6a08649b717345f0aacd.jpg)

[Question] What is the position of the baseball bat in relation to the running player? [Choices] (A) The bat is on the ground (B) The bat is in the player's hand (C) The bat is in mid air (D) The bat is with the catcher

[Choice Answer] C

[Answer] The baseball bat is in mid air, implying the player might have just thrown or hit it.

![](images/c336c565c17b9c6ddc9b73c82cbf84b44793c5153e9f4c68f52953d5cf3b8463.jpg)

[Question] Describe the position of the person who is helping the girl surf in relation to the girl. [Choices] (A) To the left and in front of the girl (B) Directly behind the girl (C) To the right and behind the girl (D) Directly in front of the girl [Answer] The person helping the girl surf is positioned slightly to the right of the girl and a bit behind her. [Choice Answer] C

Fig. 5: Examples of MLLM-DataEngine generated data in Instance Localization, Instance Counting, and Spatial Relation.

[Answer] The man is standing on a surfboard, practicing riding it in the sand. [Choice Answer] C

[Question] Describe what the two skiers are doing in the image. [Choices] (A) The skiers are racing (B) The skiers are skiing downhill (C) The skiers are jumping off a cliff (D) The skiers are facing each other [Answer] The two skiers are on a mountain slope, facing each other. They are both wearing helmets, jackets, and pants, and they have their skis on. One of them is holding up a ski pole, while the other has a ski pole pointed into the ground. [Choice Answer] D

[Question] Which part of the cat's body is playing with the man's tie? [Choices] (A) Cat's tail (B) Cat's ears (C) Cat's nails (D) Cat's whiskers [Answer] The cat is using its nails to play with the man's tie. [Choice Answer] C

## [Visual Reasoning]

[Question] What is the event taking place in the image? [Choices] (A) Wine tasting event (B) Sports event (C) Music festival (D) Birthday party [Answer] The picture appears to be a wine tasting event. [Choice Answer] A

[Question] What is the activity taking place in the image? [Choices] (A) Playing soccer (B) Playing frisbee (C) Running (D) Kneeling [Answer] The image shows two people playing frisbee on a field. [Choice Answer] B

[Question] Apart from the parking meter, what other object seems to have a metallic part that is rusted?

![](images/39b723fcfb451cccd419ec468f631e5491bcef4c260bf82eeaedf857bae46bb1.jpg)

[Text Recognition]

[Question] What religious figure is mentioned on the sign in the image? [Choices] (A) Adidas (B) FutureShop (C) Jesus (D) Milestones [Answer] Jesus. [Choice Answer] C

Fig. 6: Examples of MLLM-DataEngine generated data in Instance Interaction, Visual Reasoning, and Text Recognition.