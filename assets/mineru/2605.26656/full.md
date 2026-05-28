# DV-SFT: Direct Vision Supervision for Fine-Grained Visual Understanding

Jianfei Zhao1,2, Feng Zhang1, Xin Sun1, Chong Feng1,5, Bing Wang3, Zhixing Tan4

1School of Computer Science and Technology, Beijing Institute of Technology,

2Zhongguancun Academy, 3Beihang University, 4Zhongguancun Laboratory

5Southeast Academy of Information Technology, Beijing Institute of Technology,

zhqingan@bit.edu.cn

# Abstract

Multimodal large language models are typically trained end-to-end to predict ground-truth answers, yet supervision signals are applied exclusively to text tokens. Visual tokens, the core carriers of visual information, are optimized only implicitly as part of the context, leading to coarse-grained visual understanding. Prior works attempt to supervise visual inputs but inevitably rely on auxiliary components such as additional decoders or forward passes, because visual tokens lack readily interpretable labels. This limits their practical applicability. In this work, we propose Direct Vision Supervised Fine-Tuning (DV-SFT), which constructs explicit, token-level supervision for visual tokens and trains them through the same next-token prediction objective used for text. Specifically, we exploit the direct vision–text correspondence in OCR-related scenarios and automatically label each visual token with the word in its corresponding image patch. DV-SFT treats the MLLM as a black box, requiring no architectural modifications or additional forward passes. Extensive experiments demonstrate the superiority of direct vision supervision. DV-SFT consistently outperforms standard SFT across three in-domain and four outof-domain benchmarks. Further analyses show that vision supervision effectively enhances fine-grained visual understanding and achieves higher multimodal alignment efficiency.

# 1 Introduction

Multimodal large language models (MLLMs) (Liu et al., 2023; Dai et al., 2023; Bai et al., 2025) augment large language models with a vision encoder (Dosovitskiy et al., 2021; Zhai et al., 2023), mapping images into the language feature space. Endto-end training aligns multimodal features and enables MLLMs to achieve near-human performance on various visual understanding tasks. Despite this success, a critical limitation remains: all explicit supervision signals are applied solely to text tokens. Visual tokens—the primary carriers of visual information—are only optimized implicitly as part of the context, receiving no direct semantic guidance. Scaling up end-to-end training, though beneficial in the aggregate, still supervises visual tokens only implicitly through data statistics, leaving finegrained details easily overlooked or hallucinated (Chen et al., 2023; Tang et al., 2025).

To improve the precision of visual understanding, a number of recent efforts attempt to inject additional supervision for visual tokens. However, these signals are often indirect and coarse-grained, such as devising vision-targeted auxiliary rewards (Li et al., 2026b; Liu et al., 2026) or distilling visual features through representation alignment (Tang et al., 2025; Li et al., 2026a; Jiang et al., 2025a). In other cases, the training objective of visual tokens is misaligned with visual understanding, as seen in next-visual-token prediction (Wang et al., 2025a, 2024; Zou et al., 2026) or pixel-level reconstruction (Wang et al., 2025b; Sun et al., 2025). More critically, most of these methods require architectural modifications such as additional decoders or forward passes, which hinder their practical adoption. A more direct, intuitive, and architecturally simple form of vision supervision thus remains an open need.

An intriguing observation from recent studies (Zhao et al., 2025a; Tang et al., 2025; Jiang et al., 2025b) suggests a more direct path: end-to-end trained MLLMs can extract semantic information from visual tokens without any dedicated vision supervision. Specifically, for each visual token, the words corresponding to peak values in its logits (hereafter visual logits) often accurately describe the content of the associated image patch, including objects, attributes, text strings, and so on. Inspired by this finding, we hypothesize that explicitly supervising visual tokens by the words that describe their corresponding image patches can substantially enhance fine-grained visual understanding.

To validate this idea, we concentrate on OCRrelated scenarios, where the correspondence between vision and text is most straightforward, facilitating fine-grained alignment between image tokens and words. As illustrated in Figure 1, an MLLM’s visual logits in an OCR image can roughly recognize the textual content, but the granularity remains coarse and the alignments imprecise—a direct consequence of the absence of fine-grained vision supervision. The OCR scenarios thus serve as a clean, well-controlled testbed for studying direct vision supervision.

We therefore propose Direct Vision Supervised Fine-Tuning (DV-SFT), a simple and principled method that uses words in visual tokens as vision labels and applies the next-token prediction loss directly on visual tokens. Critically, DV-SFT requires no architectural modifications, no additional forward passes, and treats the MLLM as a black box, making it readily applicable to any existing MLLM pipeline. We conduct comprehensive evaluations across eight benchmarks, including document and chart understanding (Rajpurkar et al., 2016; Mathew et al., 2021, 2022; Masry et al., 2022), OCR (Mishra et al., 2019; Liu et al., 2024b; Singh et al., 2019), and general VQA (Fu et al., 2026).

In summary, our main contributions are as follows:

• We propose DV-SFT, which enables direct token-level supervision on visual tokens without any modification to model architecture or forward path, keeping a fully unified, end-toend training paradigm.   
• We design an offline vision label construction pipeline that leverages OCR and drawing tools to produce fine-grained, token-level supervision for visual tokens.   
• Extensive experiments show that DV-SFT significantly improves visual capability and outperforms standard SFT across a range of benchmarks, with detailed analyses confirming that the gains stem from refined visual representations.

# 2 Related Work

Multimodal large language models (Liu et al., 2023; Bai et al., 2025) typically adopt an end-toend training paradigm, where visual tokens receive supervision only indirectly through the correctness of the final text output. Although this approach effectively models visual features, the learning outcomes are often unstable, leading to coarse-grained interpretation and hallucination (Liu et al., 2024a; Chen et al., 2023; Zhao et al., 2025b). To mitigate this, a growing body of research has explored incorporating supervision on the visual tokens.

Existing vision supervision methods can be broadly grouped into four paradigms. (1) The first provides indirect semantic guidance by using reward signals. Reward functions offer a more flexible form of supervision, and these methods (Li et al., 2026b; Liu et al., 2026) reward the model based on the visual consistency observed in sampled outputs. While these approaches effectively strengthen visual understanding, they remain an indirect form of supervision. (2) The second paradigm employs representation alignment. These methods (Tang et al., 2025; Li et al., 2026a; Jiang et al., 2025a) distill high-quality features to supervise the forward pass of visual tokens. However, the poor interpretability of continuous representations can inject noise and limit the reliability of the supervision. (3) The third paradigm is nextvisual-token prediction. The emergence of discrete visual tokens has bridged the gap between the learning forms of visual and text tokens. Methods in this category supervise visual tokens through autoregressive image generation tasks (Wang et al., 2024; Zou et al., 2026). Despite their explicitness, these approaches suffer from a misalignment between the generative objective and the goal of visual understanding Wang et al. (2025a) retain the architecture of visual understanding models and use only discrete visual tokens as supervisory signals, but the objective misalignment remains. (4) The fourth paradigm aims for reconstructing visual input. These studies (Wang et al., 2025b; Sun et al., 2025) supervise the correctness of visual token understanding by reconstructing the original visual input from visual representation; however, they face a conflict (Song et al., 2026) between pixel-level and semantic features, resulting in suboptimal learning outcomes. Importantly, nearly all of these methods introduce additional architectural components, such as visual tokenizers, diffusion-based decoders, or extra forward passes, which complicate training and limit their general applicability.

In contrast, our proposed DV-SFT avoids all of the above limitations. It constructs direct tokenlevel labels for visual tokens based on the visual semantics in the image patches, closely aligning with the goal of visual understanding. By modeling vision supervision as a standard next-token prediction task, DV-SFT achieves a fully unified, end-to-end training paradigm. More importantly, DV-SFT introduces no additional modules or forward propagation passes, offering high practical usability.

![](images/888fb520239524f7addbe6c66503ce41cb88b4dd321ded68b1f32c739def3d7a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["MLLM"] --> B["Multi"]
    A --> C["Modal"]
    A --> D["Language"]
    A --> E["Model"]
    B --> F["(Visual Tokens)"]
    C --> F
    D --> F
    E --> F
    F --> G["Image"]
    style A fill:#cce5ff,stroke:#333
    style B fill:#ffcccc,stroke:#333
    style C fill:#ffcccc,stroke:#333
    style D fill:#ffcccc,stroke:#333
    style E fill:#ffcccc,stroke:#333
    style F fill:#ffcccc,stroke:#333
    style G fill:#ffcccc,stroke:#333
```
</details>

Figure 1: An example of visual logits in Qwen3-VL-2B showing the top-20 tokens at the final image patch of each word. Numbers indicate rank; punctuation and meaningless characters are removed. This observation inspires us to supervise visual tokens using the words in the image patches.

# 3 Preliminaries

An MLLM typically consists of a vision encoder that processes visual inputs, a projector that aligns features across modalities, and a large language model. In MLLMs’ training process, the input image is first tokenized into patches, and the patch sequence is input into the visual encoder. After successively processed by the encoder and projector, the input image finally forms a sequence of visual embeddings $\pmb { I } = [ \pmb { v } _ { 1 } , \pmb { v } _ { 2 } , \pmb { \cdot } \cdot \pmb { \cdot } , \pmb { v } _ { m } ] \in \mathbb { R } ^ { m \times D }$ where m is the number of visual tokens and D is the model dimension. When concatenated with the text input embeddings, the entire sequence is fed into the large language model, which outputs the logits for each token. The language modeling loss will act on the text logits under the teacher-forcing paradigm:

$$
\mathcal {L} _ {t} = - \frac {1}{n} \sum_ {i} ^ {n} \log \mathrm{P} (y _ {i} | \boldsymbol {I}, \boldsymbol {X}, \boldsymbol {Y} _ {<   i}; \theta), \tag {1}
$$

where X is the prompt and Y of length n is the ground-truth response.

![](images/a6eea0a2a687eb07e0e7e6ca45b481f49c965e6412d15078407ea420008b1510.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input"] --> B["Hello World (Image)"]
    C["Input"] --> D["Do OCR (Prompt)"]
    E["Input"] --> F["Hello World (Answer)"]
    G["Input"] --> H["Logits"]
    I["Input"] --> J["Hello World"]
    K["Input"] --> L["Do OCR"]
    M["Input"] --> N["Hello World"]
    O["Input"] --> P["Logits"]
    Q["Input"] --> R["Hello World"]
    S["Input"] --> T["Do OCR"]
    U["Input"] --> V["Hello World"]
    W["Input"] --> X["Logits"]
    Y["Input"] --> Z["Hello World"]
    AA["Input"] --> AB["Do OCR"]
    AC["Input"] --> AD["Hello World"]
    AE["Input"] --> AF["Logits"]
    AG["Input"] --> AH["Hello World"]
    AI["Input"] --> AJ["Do OCR"]
    AK["Input"] --> AL["Hello World"]
    AM["Input"] --> AN["Logits"]
    AO["Input"] --> AP["Hello World"]
    AQ["Input"] --> AR["Do OCR"]
    AS["Input"] --> AT["Hello World"]
    AU["Input"] --> AV["Logits"]
    AW["Input"] --> AX["Hello World"]
    AY["Input"] --> AZ["Word"]
    BA["Input"] --> BB["Word"]
    BC["Input"] --> BD["Word"]
    BE["Input"] --> BF["Word"]
    BG["Input"] --> BH["Word"]
    BI["Input"] --> BJ["Word"]
    BK["Input"] --> BL["Word"]
    BM["Input"] --> BN["Word"]
    BO["Input"] --> BP["Word"]
    BP --> BQ["Word"]
    BR["Input"] --> BS["Word"]
    BT["Input"] --> BU["Word"]
    BV["Input"] --> BW["Word"]
    BX["Input"] --> BY["Word"]
    BZ["Input"] --> CA["Word"]
    CB["Input"] --> CD["Word"]
    CE["Input"] --> CF["Word"]
    CG["Input"] --> CH["Word"]
    CI["Input"] --> CJ["Word"]
    CK["Input"] --> CL["Word"]
    CM["Input"] --> CN["Word"]
    CO["Input"] --> CP["Word"]
    CS["Input"] --> CY["Word"]
    CZ["Input"] --> DA["Word"]
    DB["Input"] --> DC["Word"]
    DD["Input"] --> DE["Word"]
    DF["Input"] --> DG["Word"]
    DH["Input"] --> DI["Word"]
    DJ["Input"] --> DK["Word"]
    DL["Input"] --> DM["Word"]
    DN["Input"] --> DE
    DG --> DE
```
</details>

Figure 2: Diagram of DV-SFT. The training procedure of DV-SFT is identical to that of standard SFT, where all tokens are trained end-to-end on the next-token prediction task. The difference lies in the fact that a significant portion of visual tokens have valid labels.

Like text tokens, the model also produces logits for all visual tokens, referred to as visual logits. Recent studies (Zhao et al., 2025a; Jiang et al., 2025b; Tang et al., 2025) found that the distribution of visual logits contains concrete semantic features in each visual token. Sepcifically, the top words in $\mathrm { P } ( . | \pmb { I } _ { \leq i } )$ always reflect the visual content in the patch of vi. Intuitively, the generalizable feature patterns that emerge spontaneously in models under end-to-end training serve as high-quality supervisory signals. Therefore, inspired by this, we use words that are semantically aligned with the visual content to supervise the learning of visual tokens in MLLMs:

$$
\mathcal {L} _ {v} = - \frac {1}{m} \sum_ {i} ^ {m} \log \mathrm{P} (w _ {i} \mid I _ {\leq i}; \theta), \tag {2}
$$

where $w _ { i }$ is the label for the visual token ${ \boldsymbol { v } } _ { i }$

# 4 Direct Vision Supervision

We propose Direct Vision Supervised Fine-Tuning (DV-SFT), which utilizes words that describe image patches as direct supervision signals for visual tokens, thereby enhancing the model’s learning of visual tokens. Due to the information-rich nature of the visual modality, constructing labels for visual tokens is highly challenging. To address this, we exploit the straightforward association between vision and text in OCR-related scenarios to construct labels for visual tokens. Specifically, we use the word contained in a visual token as its label and let the model learn through the next-token prediction task, as illustrated in Figure 2. To ensure parallel training, we perform only one-step prediction for vision labels. Specifically, the loss is computed using only the first token after tokenizing the vision label. This practice also conforms to the first-token preference property of visual logits (Zhao et al., 2025a).

# 4.1 Vision Label Construction

We attempt to construct labels for visual tokens in OCR-related scenarios, where the direct correspondence between text and vision makes the construction of vision labels feasible. The challenge in constructing vision labels lies in establishing a oneto-one mapping between words in the image and visual tokens. Specifically, it involves identifying which words appear in the image and determining which visual tokens each word should supervise.

We achieve this through two approaches: (1) Image-to-Label, and (2) Label-to-Image.

Image-to-Label. We first use a text detection model and a text recognition model to conduct word-level OCR for the image, obtaining all words along with their spatial locations. We identify word concatenation errors in the OCR results by checking the tokenization length and discard words whose tokenization length exceeds three. Extremely large words (with a height exceeding three times the patch size) are also discarded to ensure the annotation precision of the vision labels. Considering the autoregressive nature of MLLMs, we align each word to the visual token corresponding to its bottom-right corner and use it as the label for that visual token. Finally, we filter out multiple vision labels assigned to the same visual token to avoid confusing the model.

Label-to-Image. The visual encoder tokenizes images through a mechanical strategy, which prevents perfect alignment between text in the image and visual tokens. To address this issue, we construct fine-grained vision labels through a text-toimage approach. Specifically, we use a unimodal document question-answering dataset and employ drawing tools to render documents by aligning each word individually onto the patch grid of a blank image. Words that span multiple visual tokens are assigned as labels only to the last visual token, in accordance with the autoregressive property.

# 4.2 Vision Smoothing

Although MLLMs are causal models with unidirectional attention, the visual encoder possesses bidirectional characteristics. This means that visual information will interact across tokens—that is, a visual token may contain semantic information from subsequent visual tokens in the sequence. The example in Figure 1 illustrates this phenomenon: the last word “model” in the image appears in the logits of the fourth token.

To address this phenomenon, we propose Vision Smoothing method inspired by label smoothing, in which a visual token learns its own vision label while also learning other vision labels in the image. Accordingly, Eq. 2 is modified as:

$$
\begin{array}{l} \mathcal {L} _ {v} = - \frac {1}{m} \sum_ {i = 1} ^ {m} \left[ (1 - \beta) \log \mathrm{P} \left(w _ {i} \mid I _ {\leq i}; \theta\right) \right. \tag {3} \\ \left. + \frac {\beta}{| V | - 1} \sum_ {t \neq w _ {i}} ^ {V} \log \mathrm{P} (t \mid I _ {\leq i}; \theta) \right], \\ \end{array}
$$

where V denotes all vision labels in an image and β is the smoothing factor.

The final loss of DV-SFT is:

$$
\mathcal {L} = \mathcal {L} _ {t} + \lambda \mathcal {L} _ {v}, \tag {4}
$$

where λ is a hyperparameter.

# 5 Experiments

# 5.1 Experimental Setup

Datasets. We employ DocVQA (Mathew et al., 2021), InfographicVQA (Mathew et al., 2022), and SQuAD (Rajpurkar et al., 2016) to build training data. Both DocVQA and InfographicVQA are multimodal datasets, and we construct vision labels using the data-to-label approach. SQuAD is a unimodal document understanding dataset with extractive answers, which aligns well with the model’s need for accurate visual perception. We convert this dataset into a multimodal task through the label-todata approach. The statistics of the training data are shown in Table 1.

Benchmarks. For in-domain data, we use the test sets of DocVQA and InfographicVQA, as well as the validation set of SQuAD. For out-of-domain data, we use the test set of ChartQA (Masry et al., 2022), OCRBench (Liu et al., 2024b), the test set of OCRVQA (Mishra et al., 2019), and the validation set of TextVQA (Singh et al., 2019). For OCRVQA, we randomly select 3,000 test samples and ensure that the ground truth of each sample is visible in the image.

<table><tr><td>Dataset</td><td>Samples</td><td>Images</td><td>Text Labels</td><td>Vision Labels</td><td>Vision Coverage</td></tr><tr><td>DocVQA</td><td>39,463</td><td>10,194</td><td>190,895</td><td>5,402,739</td><td>6.97%</td></tr><tr><td>InfographicVQA</td><td>23,946</td><td>4,406</td><td>61,608</td><td>4,173,608</td><td>11.94%</td></tr><tr><td>SQuAD</td><td>24,929</td><td>13,183</td><td>60,957</td><td>2,607,031</td><td>31.44%</td></tr></table>

Table 1: Statistical information of training data. Visual Coverage refers to the percentage of visual tokens that are assigned vision labels.

Baselines. We choose the standard SFT method as our baseline, which trains the model using only text labels. In addition, we reproduce BASIC (Tang et al., 2025) method, which is a self-distilled representation supervision method.

Implementation Details. We select Qwen3-VL-2B/8B-Instruct and Qwen3-1.7B as the base models. For Qwen3-VL, we randomly select one QA pair (two for InfographicVQA) per image to avoid duplicate vision labels. For Qwen3, we load the visual encoder from Qwen3-VL-2B and use all training data. We set $\beta = 0 . 3$ in Eq. 3 and $\lambda = 2 \mathrm { e } { \mathrm { - } } 3$ in Eq. 4. We employ greedy decoding during inference with the default setting.

Detailed experimental setups are provided in Appendix A.

# 5.2 Main Results

We validate the superiority of our method on both in-domain and out-of-domain data, and the results are illustrated in Table 2.

In Domain. Our method, DV-SFT, achieves consistent advantages on in-domain data, and the experimental results across three base models fully validate its effectiveness. Moreover, the next-token prediction learning paradigm adopted by DV-SFT aligns more naturally with the inherent characteristics of language models, yielding significant advantages over BASIC. The intuitively comprehensible supervisory signal is another strength of DV-SFT. The representation supervision approach employed by BASIC can suffer from severe model degradation when the quality of the representations is suboptimal. This issue is confirmed in the experiments on the 2B model, where the training results of BA-SIC fall far behind the original capabilities of the base model. In the experiments on the 8B model, BASIC benefits from higher-quality supervised representations and achieves stable performance improvements; however, its gains are not significantly superior to those of SFT. Thanks to direct vision supervision, DV-SFT enables the model to understand visual features more accurately, thereby achieving a significant advantage in text-oriented visual understanding.

Out of Domain. DV-SFT also demonstrates significant advantages on out-of-domain tasks, indicating its strong domain generalization capability. Due to the limited knowledge capacity of the 2B model, DV-SFT achieves the best overall performance but does not consistently outperform across all datasets. In contrast, in the experiments with the 8B model, where the larger parameter count affords greater knowledge capacity, DV-SFT consistently improves performance on out-of-domain datasets. Experimental results on out-of-domain tasks indicate that DV-SFT does not simply improve taskspecific performance but genuinely enhances the model’s visual capability.

In summary, constructing vision supervision for MLLMs can more effectively enhance their visual understanding capability. Compared with SFT, DV-SFT achieves consistent advantages on both in-domain and out-of-domain tasks. Moreover, the direct vision supervision established by DV-SFT aligns perfectly with the next-token prediction learning paradigm of the model and does not rely on any additional network modules or forward passes. The end-to-end black box training approach requires no intermediate states of the model. These advantages in the training paradigm make DV-SFT highly practical and readily applicable.

# 5.3 Multimodal Alignment

To investigate the advantages of vision supervision over text-only supervision, we analyze the effectiveness of the two methods in the process of a unimodal model learning visual features from scratch. As shown in Table 2, vision supervision demonstrates comprehensive advantages over text-only supervision, indicating that with limited training data, vision supervision enables the model to better understand visual information. The advantage of vision supervision is even more pronounced on out-of-domain tasks. Compared with an average performance improvement of 0.74% on in-domain tasks, DV-SFT achieves an average improvement of 1.6% on out-of-domain tasks. This suggests that vision supervision genuinely enhances the model’s visual understanding capability, rather than merely improving task-specific learning.

<table><tr><td rowspan="2">Method</td><td colspan="4">In Domain</td><td colspan="5">Out of Domain</td></tr><tr><td>DocVQA</td><td>InfoVQA</td><td>SQuAD</td><td>Avg.</td><td>ChartQA</td><td>OCRBench</td><td>OCRVQA</td><td>TextVQA</td><td>Avg.</td></tr><tr><td>Qwen3-VL-2B</td><td>92.25</td><td>70.18</td><td>77.43</td><td>79.95</td><td>70.56</td><td>72.00</td><td>72.97</td><td>84.88</td><td>75.10</td></tr><tr><td>SFT</td><td>93.54</td><td>71.66</td><td>86.99</td><td>84.06</td><td>69.60</td><td>73.10</td><td>75.00</td><td>84.10</td><td>75.45</td></tr><tr><td>BASIC</td><td>84.02</td><td>57.15</td><td>72.95</td><td>71.37</td><td>54.96</td><td>65.90</td><td>66.33</td><td>77.12</td><td>66.08</td></tr><tr><td>DV-SFT</td><td>93.68</td><td>72.34</td><td>87.38</td><td>84.47</td><td>69.80</td><td>74.40</td><td>75.17</td><td>84.08</td><td>75.86</td></tr><tr><td>Qwen3-VL-8B</td><td>95.97</td><td>81.88</td><td>80.27</td><td>86.23</td><td>77.12</td><td>82.30</td><td>76.03</td><td>87.80</td><td>80.81</td></tr><tr><td>SFT</td><td>96.28</td><td>82.24</td><td>87.60</td><td>88.71</td><td>77.52</td><td>83.70</td><td>77.17</td><td>88.44</td><td>81.71</td></tr><tr><td>BASIC</td><td>96.38</td><td>82.39</td><td>86.72</td><td>88.50</td><td>77.44</td><td>83.40</td><td>77.00</td><td>88.14</td><td>81.50</td></tr><tr><td>DV-SFT</td><td>96.40</td><td>82.49</td><td>88.14</td><td>89.01</td><td>77.76</td><td>84.00</td><td>77.17</td><td>88.48</td><td>81.85</td></tr><tr><td>Qwen3-1.7B</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SFT</td><td>86.49</td><td>58.02</td><td>77.60</td><td>74.04</td><td>48.60</td><td>57.90</td><td>60.67</td><td>66.74</td><td>58.48</td></tr><tr><td>DV-SFT</td><td>86.57</td><td>58.51</td><td>78.69</td><td>74.59</td><td>49.08</td><td>58.60</td><td>62.20</td><td>67.82</td><td>59.43</td></tr></table>

Table 2: Main results. Avg. denotes the average accuracy. The bolded results indicate the best performance in each setting.

![](images/86458da9d4e4c79fcc33015849ffdaa824ffcf0e63b82d76f9996dd292a6f39c.jpg)

<details>
<summary>line</summary>

| Epoch | In Domain Avg. Acc. | In Domain +3.78% | In Domain +0.84% | In Domain +0.69% | In Domain +0.62% | In Domain +0.83 | Out of Domain Avg. Acc. | Out of Domain +2.50% | Out of Domain +6.89% | Out of Domain +2.70% | Out of Domain +1.62% |
|-------|---------------------|-------------------|-------------------|-------------------|-------------------|-----------------|--------------------------|------------------------|------------------------|------------------------|------------------------|
| 0     | 51.0                | 51.0              | 51.0              | 51.0              | 51.0              | 51.0            | 51.0                     | 51.0                   | 51.0                   | 51.0                   | 51.0                   |
| 0.2   | 59.0                | 59.0              | 59.0              | 59.0              | 59.0              | 59.0            | 53.0                     | 53.0                   | 53.0                   | 53.0                   | 53.0                   |
| 0.4   | 64.0                | 64.0              | 64.0              | 64.0              | 64.0              | 64.0            | 54.0                     | 54.0                   | 54.0                   | 54.0                   | 54.0                   |
| 0.6   | 67.0                | 67.0              | 67.0              | 67.0              | 67.0              | 67.0            | 57.0                     | 57.0                   | 57.0                   | 57.0                   | 57.0                   |
| 0.8   | 68.0                | 68.0              | 68.0              | 68.0              | 68.0              | 68.0            | 58.0                     | 58.0                   | 58.0                   | 58.0                   | 58.0                   |
| 1     | 68.3                | 68.3              | 68.3              | 68.3              | 68.3              | 68.3            | 59.0                     | 59.0                   | 59.0                   | 59.0                   | 59.0                   |
</details>

Figure 3: Test results of Qwen3-1.7B at each checkpoint during training. Results under In-Domain are obtained on the validation set.

We further analyze the visual learning process of the unimodal model, as shown in Figure 3. It can be observed that in the early stages of training, vision supervision exhibits a more pronounced advantage over text-only supervision. Similarly, this advantage is more significant on out-of-domain tasks. This indicates that applying direct supervision to visual tokens leads to higher multimodal alignment efficiency compared to indirect supervision of textonly labels.

In summary, the experimental results on the unimodal model further validate the superiority of direct vision supervision.

# 5.4 Character Recognition Ability

We use a plain text OCR task to evaluate the improvement in character recognition ability brought by DV-SFT. Specifically, we construct test samples from the news dataset XSUM (Narayan et al., 2018), convert the documents into images using drawing tools, and then ask the model to recognize the text in the images. To avoid the influence of the model’s prior knowledge on character recognition, we further construct a Non-Contextual text recognition task. We collect all words from the dataset and sample them based on word frequency to form semantically meaningless word sequences, which are used to test the model’s pure character recognition ability.

The results on both tasks, each with 500 test samples, are shown in the left part of Figure 4. First, we can observe that DV-SFT achieves superior performance on both OCR tasks, indicating that vision supervision can effectively improve the model’s character recognition accuracy. Furthermore, the advantage of DV-SFT is amplified on the Non-Contextual OCR task, which further demonstrates that direct vision supervision enhances the model’s fine-grained visual understanding capability rather than merely improving task performance.

During DV-SFT training, vision labels supervise visual tokens in a one-to-one manner. To investigate whether this learning pattern can generalize to one-to-many or many-to-one scenarios, we evaluate the model’s ability to recognize text in images at different resolutions. As shown in the results in the right part of Figure 4, DV-SFT consistently outperforms SFT across different resolutions. Although the model learns vision–text correspondences in a one-to-one mode during training, it effectively acquires visual features rather than mechanically learning word prediction.

![](images/8fac940c6f34c1f9be691a84e39dd35a7afee63439a7ec2b37092da209e9f552.jpg)

Figure 4: Test results on naive text OCR tasks. Contextual OCR refers to the recognition task on normal text, while Non-Contextual OCR refers to the recognition task on unordered word sequences. NED denotes Normalized Edit Distance, where lower values indicate better performance. Left part: test results at the original image resolution. Right part: test results of the 2B model at different image resolutions, with the X-axis representing the visual token length. 

<table><tr><td rowspan="2">Method</td><td colspan="2">Qwen3-VL-2B</td><td colspan="2">Qwen3-VL-8B</td></tr><tr><td>Acc.</td><td>F1</td><td>Acc.</td><td>F1</td></tr><tr><td>SFT</td><td>81.21</td><td>80.07</td><td>87.95</td><td>88.46</td></tr><tr><td>DV-SFT</td><td>80.88</td><td>80.12</td><td>88.08</td><td>88.47</td></tr></table>

Table 3: The results on MME.

![](images/ce2c40683abbf146673df643bf897e9c1310d42d9ad455cef4fceb18fe580190.jpg)  
Figure 5: A case study of visual logits in a general visual scene. The image is resized to dimensions of 10 × 10 visual tokens. Each box represents a visual token, with the word inside indicating its Top-1 logit. Green boxes denote matches with the visual content; red boxes indicate mismatches.

# 5.5 General Visual Capability

We leverage the straightforward association between vision and text in OCR-related scenarios to construct supervisory signals for visual tokens. Specifically, we use the word within the patch corresponding to each visual token to supervise the model’s learning of that token. Extensive experiments demonstrate that vision supervision effectively enhances the model’s character recognition ability. However, whether purely text-based vision supervision leads to degradation in visual understanding on non-text scenarios remains an open question.

We evaluate the model’s visual capability in general scenarios using MME (Fu et al., 2026), a benchmark comprising 10 visual perception tasks and 4 visual cognition tasks. The experimental results are shown in Table 3. From the results, we can observe that purely word-based vision supervision does not degrade the model’s visual capabilities in general scenarios; rather, it slightly improves the model’s overall visual performance. This suggests that although vision supervision constructed around OCR scenarios cannot fully annotate visual information, this form of direct vision supervision facilitates the development of fine-grained visual understanding mechanisms in the model. As can be seen from Figure 5, although the model learns visual features under word-based vision supervision, this direct supervision enables the model to develop more concrete semantic understanding of visual tokens, and this visual advantage generalizes effectively to general scenarios.

In summary, although training the model on OCR-related scenarios yields effective performance improvements on general vision tasks, vision supervision can effectively enhance the model’s semantic understanding of visual tokens, and this semantic understanding generalizes from text to general visual scenarios. This phenomenon further validates the effectiveness and necessity of vision supervision.

![](images/188fb737c0215c740364519c0fcf42e18c0b4773178c534d7e7ad35ab89b0e47.jpg)

<details>
<summary>line</summary>

| x     | λ      | β      |
|-------|--------|--------|
| 0.0   | 76.4   | 76.3   |
| 0.1   | 76.3   | 76.6   |
| 0.2   | 76.8   | 76.4   |
| 0.3   | 76.4   | 76.8   |
| 0.4   | 76.1   | 76.5   |
</details>

Figure 6: Ablation studies on the validation set of DocVQA and InfographicVQA. Avg. Acc. denotes average accuracy.

# 5.6 Ablation Study

We conduct ablation experiments on the two key mechanisms in DV-SFT: Vision Supervision (λ in Eq. 4) and Visual Smoothing (β in Eq. 3). The Results are shown in Figure 6.

Vision Supervision. The experimental results show that incorporating only weak vision supervision on top of SFT can yield stable performance improvements for the model. However, we observe that excessively large weights for vision supervision lead to negative effects. We attribute this phenomenon to three factors: (1) Label imbalance. As shown in the dataset statistics in Table 1, the number of vision labels is several tens of times that of text labels. Although label normalization alleviates the quantitative imbalance to some extent, a larger amount of supervisory signals can still lead to training imbalance. (2) Initial gradient. We compute the validation loss on 1,024 training samples for the 2B model. The average loss for vision supervision is 24.30, while that for text supervision is 0.98. Since vision labels and text labels follow exactly the same training procedure, the large discrepancy in initial loss causes the gradient to be heavily biased toward learning visual tokens. (3) Task objective. The ultimate goal of MLLMs is to generate responses that meet task requirements, whereas the role of vision supervision is to assist the model in understanding visual information. In summary, incorporating vision supervision on top of text supervision can effectively enhance the model’s visual understanding capability, but the weight of vision supervision should be properly balanced to prevent degradation of the model’s generation ability.

Vision Smoothing. We can observe that without vision smoothing, the advantage of DV-SFT over SFT is very limited, which is due to the information interaction across visual tokens. Although the forward propagation in MLLMs follows a strictly unidirectional mask, visual tokens first pass through the vision encoder to extract features before being fed into the MLLM. During this process, bidirectional information flow occurs, such that each visual token contains not only the visual information of its corresponding patch but also the context of the entire visual input. Consequently, having each visual token learn only its own label restricts its information representation. In contrast, our proposed visual smoothing method better accommodates the global nature of visual tokens and can more effectively supervise the learning of visual tokens.

In summary, the experimental results validate the necessity of balancing vision loss and text loss, as well as the positive effect of vision smoothing on vision supervision.

# 6 Conclusion

In this paper, we propose DV-SFT, a method motivated by the semantic alignment observed in visual logits that constructs explicit token-level supervision for visual tokens. By exploiting the straightforward vision–text correspondence in OCR-related scenarios, we construct fine-grained vision labels through two complementary pipelines: an imageto-label approach that extracts words from images using OCR engines, and a label-to-image approach that synthesizes images from word labels using drawing tools. To further address the contextualization of visual tokens caused by the bidirectional attention in the vision encoder, we introduced Vision Smoothing, which optimizes the learning of vision labels. Experimental results on both in-domain and out-of-domain benchmarks demonstrate that DV-SFT consistently outperforms standard supervised fine-tuning, and further analyses confirm that direct vision supervision effectively enhances finegrained visual understanding and improves multimodal alignment efficiency.

In the future, we plan to extend direct vision supervision to broader visual domains beyond OCR.

# Limitations

We proposed DV-SFT, which enhances the model’s visual capability through direct vision supervision. Although the importance of direct vision supervision has been thoroughly validated, there remains room for improvement in its specific implementation, particularly in the following aspects:

OCR Scenarios. We exploited the natural correspondence between vision and text in OCR tasks to construct labels for visual tokens. However, this approach to constructing vision labels may not be easily generalizable to broader visual scenes. In general visual scenes, visual tokens often carry more complex and diverse information, making it highly challenging to construct sufficiently matched vision labels.

Single Vision Label. In OCR tasks, visual tokens and text tend to exhibit a one-to-one correspondence, which facilitates the construction of vision labels. In general visual scenes, however, a single visual token often contains highly rich visual information, and may need multiple labels to represent the visual information within a patch.

Vision Coverage. Although our method achieves substantial vision coverage, some visual tokens still remain unlabeled. In future work, we will explore how to improve the construction of vision labels to achieve higher-quality vision supervision.

Despite the above limitations, the experiments fully validate the effectiveness and superiority of direct vision supervision and offer a new perspective for future MLLM training paradigms.

# Ethical Considerations

This work does not introduce any new datasets and therefore raises no privacy or personally identifiable information concerns. Our method relies solely on existing, publicly available tools for label construction. Consequently, DV-SFT poses negligible unique ethical risks beyond those already associated with general MLLM training and deployment.

# References

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. 2025. Qwen3- vl technical report. arXiv preprint arXiv:2511.21631.

Zhiyang Chen, Yousong Zhu, Yufei Zhan, Zhaowen Li, Chaoyang Zhao, Jinqiao Wang, and Ming Tang. 2023. Mitigating hallucination in visual language models with visual supervision. arXiv preprint arXiv:2311.16479.   
Cheng Cui, Ting Sun, Manhui Lin, Tingquan Gao, Yubo Zhang, Jiaxuan Liu, Xueqing Wang, Zelun Zhang, Changda Zhou, Hongen Liu, et al. 2025. Paddleocr 3.0 technical report. arXiv preprint arXiv:2507.05595.   
Wenliang Dai, Junnan Li, Dongxu Li, Anthony Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale N Fung, and Steven Hoi. 2023. Instructblip: Towards general-purpose vision-language models with instruction tuning. In NIPS, pages 49250–49267.   
Alexey Dosovitskiy, Lucas Beyer, Alexander Kolesnikov, Dirk Weissenborn, Xiaohua Zhai, Thomas Unterthiner, Mostafa Dehghani, Matthias Minderer, Georg Heigold, Sylvain Gelly, et al. 2021. An image is worth 16x16 words: Transformers for image recognition at scale. In ICLR.   
Chaoyou Fu, Peixian Chen, Yunhang Shen, Yulei Qin, Mengdan Zhang, Xu Lin, Jinrui Yang, Xiawu Zheng, Ke Li, Xing Sun, et al. 2026. Mme: A comprehensive evaluation benchmark for multimodal large language models. In NIPS.   
Jiachen Jiang, Jinxin Zhou, Bo Peng, Xia Ning, and Zhihui Zhu. 2025a. Analyzing fine-grained alignment and enhancing vision understanding in multimodal language models. In NIPS, volume 38, pages 874– 899.   
Nick Jiang, Anish Kachinthaya, Suzanne Petryk, and Yossi Gandelsman. 2025b. Interpreting and editing vision-language representations to mitigate hallucinations. In ICLR, volume 2025, pages 63582–63605.   
Hengzhuang Li, Xinsong Zhang, Qiming Peng, Bin Luo, Han Hu, Dengyang Jiang, Han-Jia Ye, Teng Zhang, and Hai Jin. 2026a. Unleashing the intrinsic visual representation capability of multimodal large language models. In CVPR, pages 11975–11986.   
Zongxia Li, Wenhao Yu, Chengsong Huang, Zhenwen Liang, Rui Liu, Fuxiao Liu, Jingxi Chen, Dian Yu, Jordan Lee Boyd-Graber, Haitao Mi, and Dong Yu. 2026b. Vision-sr1: Self-rewarding vision-language model via reasoning decomposition and multi-reward policy optimization. In ICLR.   
Hanchao Liu, Wenyuan Xue, Yifei Chen, Dapeng Chen, Xiutian Zhao, Ke Wang, Liping Hou, Rongjun Li, and Wei Peng. 2024a. A survey on hallucination in large vision-language models. arXiv preprint arXiv:2402.00253.   
Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In NIPS, volume 36, pages 34892–34916.

Jiazhen Liu, Yuchuan Deng, and Long Chen. 2026. Empowering small vlms to think with dynamic memorization and exploration. In ICLR.   
Yuliang Liu, Zhang Li, Mingxin Huang, Biao Yang, Wenwen Yu, Chunyuan Li, Xu-Cheng Yin, Cheng-Lin Liu, Lianwen Jin, and Xiang Bai. 2024b. Ocrbench: On the hidden mystery of ocr in large multimodal models. Science China Information Sciences, 67(12):220102.   
Ahmed Masry, Xuan Long Do, Jia Qing Tan, Shafiq Joty, and Enamul Hoque. 2022. Chartqa: A benchmark for question answering about charts with visual and logical reasoning. In Findings of ACL, pages 2263– 2279.   
Minesh Mathew, Viraj Bagal, Rubèn Tito, Dimosthenis Karatzas, Ernest Valveny, and CV Jawahar. 2022. Infographicvqa. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 1697–1706.   
Minesh Mathew, Dimosthenis Karatzas, and CV Jawahar. 2021. Docvqa: A dataset for vqa on document images. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pages 2200–2209.   
Anand Mishra, Shashank Shekhar, Ajeet Kumar Singh, and Anirban Chakraborty. 2019. Ocr-vqa: Visual question answering by reading text in images. In International conference on document analysis and recognition, pages 947–952. IEEE.   
Shashi Narayan, Shay B Cohen, and Mirella Lapata. 2018. Don’t give me the details, just the summary! topic-aware convolutional neural networks for extreme summarization. In EMNLP, pages 1797–1807.   
Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. Squad: 100,000+ questions for machine comprehension of text. In EMNLP, pages 2383–2392.   
Amanpreet Singh, Vivek Natarjan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach. 2019. Towards vqa models that can read. In CVPR, pages 8317–8326.   
Wei Song, Yuran Wang, Zijia Song, Yadong Li, Zenan Zhou, Long Chen, Xu Jhua, Jiaqi Wang, and Kaicheng Yu. 2026. Dualtoken: Towards unifying visual understanding and generation with dual visual vocabularies. In ICLR.   
Zhen Sun, Yunhang Shen, Jie Li, Xing Sun, Pingyang Dai, Liujuan Cao, and Rongrong Ji. 2025. Ds-vlm: Diffusion supervision vision language model. In ICML.   
Jianting Tang, Yubo Wang, Haoyu Cao, and Linli Xu. 2025. Basic: Boosting visual alignment with intrinsic refined embeddings in multimodal large language models. In ICCV, pages 20582–20592.

Dianyi Wang, Wei Song, Yikun Wang, Siyuan Wang, Kaicheng Yu, Zhongyu Wei, and Jiaqi Wang. 2025a. Autoregressive semantic visual reconstruction helps vlms understand better. arXiv preprint arXiv:2506.09040.   
Haochen Wang, Anlin Zheng, Yucheng Zhao, Tiancai Wang, Zheng Ge, Xiangyu Zhang, and Zhaoxiang Zhang. 2025b. Reconstructive visual instruction tuning. In ICLR, volume 2025, pages 14374–14399.   
Xinlong Wang, Xiaosong Zhang, Zhengxiong Luo, Quan Sun, Yufeng Cui, Jinsheng Wang, Fan Zhang, Yueze Wang, Zhen Li, Qiying Yu, et al. 2024. Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869.   
Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. 2023. Sigmoid loss for language image pre-training. In CVPR, pages 11975–11986.   
Jianfei Zhao, Feng Zhang, Xin Sun, Chong Feng, and Zhixing Tan. 2025a. Tell model where to look: Mitigating hallucinations in mllms by vision-guided attention. arXiv preprint arXiv:2511.20032.   
Jianfei Zhao, Feng Zhang, Xin Sun, Lingxing Kong, Zhixing Tan, and Chong Feng. 2025b. Cross-image contrastive decoding: Precise, lossless suppression of language priors in large vision-language models. arXiv preprint arXiv:2505.10634.   
Kai Zou, Hongbo Liu, Dian Zheng, Jianxiong Gao, Zhiwei Zhao, and Bin Liu. 2026. Echogen: Cycleconsistent learning for unified layout-image generation and understanding. In AAAI, volume 40, pages 14068–14076.

# A Detailed Experimental Setup

# A.1 Datasets

We employ DocVQA (Mathew et al., 2021), InfographicVQA (Mathew et al., 2022), and SQuAD (Rajpurkar et al., 2016) to build training data. Both DocVQA and InfographicVQA are multimodal datasets, and we construct vision labels using the image-to-label method. We use PaddleOCR (Cui et al., 2025) to perform word-level OCR, which specifically includes a text detection model (PP-OCRv5\_server\_det) and a text recognition model (PP-OCRv5\_server\_rec). An example of image-tolabel is shown in Figure 8.

SQuAD is a unimodal document understanding dataset with extractive answers, which aligns well with the model’s need for accurate visual perception. We convert this dataset into a multimodal task through the label-to-image approach. We randomly select one dark and one light color for rendering the background and text, respectively, to enhance training diversity. An example of label-to-image is shown in Figure 9. To strengthen the model’s reliance on visual information, we filter out questionanswer pairs in SQuAD that exhibit weak relevance to the document. We also remove answers with low fluency. We achieve this using the perplexity of a unimodal LLM, Qwen3-0.6B. Specifically, each training sample satisfies the following conditions: PPL $( A \mid Q ) > 1 0 0$ and PPL $( A \mid D , Q ) < 1 0$ .

To ensure the verifiability of the model’s responses, we randomly incorporate diverse formatcontrol instructions into the training data. The detailed instructions are shown in Figure 7.

# A.2 Benchmarks

For in-domain data, we use the test sets of DocVQA and InfographicVQA, as well as the validation set of SQuAD. For out-of-domain data, we use the test set of ChartQA (Masry et al., 2022), OCRBench (Liu et al., 2024b), the test set of OCRVQA (Mishra et al., 2019), and the validation set of TextVQA (Singh et al., 2019). For OCRVQA, we randomly select 3,000 test samples and ensure that the ground truth of each sample is visible in the image. Results for DocVQA and InfographicVQA are obtained through online evaluation 2, while the remaining benchmarks are evaluated by ground truth.

During testing, we use a uniform format-control instruction: “Answer the question using a single word or phrase.”

# Instructions

1. Answer the question using a single word or phrase.   
2. Respond with only the answer, no explanation.   
3. Give a concise answer in one word or short phrase.   
4. Output only the answer, nothing else.   
5. Reply with a single word or phrase, no full sentences.   
6. Provide the answer as a single word or brief phrase.   
7. Provide a brief answer without additional text.   
8. Keep your response to a single word or short phrase.   
9. State the answer concisely in a word or phrase.   
10. No explanations―just the answer word or phrase.

Figure 7: Format-control instructions. 

<table><tr><td colspan="2">Qwen3-VL-2B / Qwen3-1.7B</td></tr><tr><td>Min Pixels</td><td>32*32*64</td></tr><tr><td>Max Pixels</td><td>32*32*2048</td></tr><tr><td>Global Batch Size</td><td>64</td></tr><tr><td>Learning Rate</td><td>5e-6</td></tr><tr><td>Warmup Ratio</td><td>0.03</td></tr><tr><td>Max Grad Norm</td><td>1</td></tr><tr><td>LR Scheduler</td><td>cosine</td></tr><tr><td>Weight Decay</td><td>0</td></tr><tr><td>Model Max Length</td><td>8192</td></tr><tr><td colspan="2">Additional Settings for Qwen3-VL-8B</td></tr><tr><td>Learning Rate</td><td>2e-5</td></tr><tr><td>LoRA Rank</td><td>64</td></tr><tr><td>LoRA Alpha</td><td>128</td></tr><tr><td>LoRA Dropout</td><td>0</td></tr></table>

Table 4: Detailed training settings.

# A.3 Baselines

We choose the standard SFT method as our baseline, which trains the model using only text labels. In addition, we reproduce BASIC (Tang et al., 2025) method. This approach constructs supervisory signals from the hidden states of visual tokens in the early-to-middle layers of the model and trains the projector and the first layer of the model to learn visual features through representation supervision. To ensure compatibility with FlashAttention, we disable the attention-distribution-based loss normalization in BASIC and adopt uniform normalization instead. All hyperparameters follow the settings in the original paper.

# A.4 Implementation Details

We select Qwen3-VL-2B/8B-Instruct and Qwen3- 1.7B as the base models. For Qwen3-VL-2B/8B-

Instruct, we randomly select one QA pair (two for InfographicVQA) per image to avoid duplicate vision labels. For Qwen3-1.7B, we load the visual encoder from Qwen3-VL-2B and use all training data. The visual encoder parameters are frozen in all training settings. Qwen3-VL-2B and Qwen3- 1.7B are fully fine-tuned, while Qwen3-VL-8B is fine-tuned using LoRA. All models are trained for one epoch, and the final checkpoint is selected as the final model. The detailed experimental settings are shown in Table 4.

We set $\beta = 0 . 3$ in Eq. 3 and $\lambda = 2 \mathrm { { e } \mathrm { { - } 3 } }$ in Eq. 4. We employ greedy decoding during inference with the default setting. All experiments were conducted on two NVIDIA A100 80G GPUs.

When constructing the Non-Contextual OCR task, we randomly generate unordered word sequences of 200–500 characters in length based on word frequency to create OCR images. The average ground-truth length for the Contextual OCR task is 550.22, while that for the Non-Contextual OCR task is 348.63.

withmodifierwas necessary for the polyimide sorbents due tothe stronger intermolecular forees.M. Krieger(Indiana University）and S.Hawthome(University of N.Dakota) experimented with polyurethane foamPUF)with SFEinquantitativelyrecovering wide varietiesofoganicompouncsTheseicluded semi-volatilesphenolsaromatisand PAH's.APUFsamplingofairinamoker'sofficeandfolowedbySFE-GC(S.Hawtoe) showed the presence of phenolsnicotineand fatty acids.PUFand SFE-GC were also demonstrated in quantitative measurementsof phenolicsinwoodsmokeanalysis.W.T Foreman U.sGeological SurveyCO）extractedtheCcartridgewith SFEtoecove pesticidesin highyield

# DETERMINATIONOF POLAR VOLATILE ORGANICSPVOC)INAMBIENTAIR

Thepolar compounds are those containing hetero-atoms such as nitrogen,sulfur and oxygen.The single most difficult problem in developingprotocols for analyzing polar compoundsat uacelevel in air is probably moisture.Sampling of sidesuream smoke ompoentssedsidemoiseinebietoguphen trapand prevented sample enrichment The evaporation ofwate vaporinthe souree of the massspecrometeriteredieighvaaeetecfctoos Thepresent EPA TO-14 methodrequirestheuseof Naphion dryertoeliminatewater UnfortunatelytheNaphionubeisalsopemeableomanypolarompoundssuchcarbonyl andalcohols.MethodTO14 withcanistersampling isonly fornonpolarorganiccompounds e.g.aromaticsandhydrocarbons.

Pleil(U.s.EPAResearch Triangle Park,Nsummarized theresearchactivity in PVOC.Twodevelopmentsare worthmentioning here.ASUMMA passivatedanisterwhich replaced sampling bags inairsamplingwas evaluated foritsstability towards PvOCby using astandarditure consistingacetonitrile,methanol,acetone,acrylnitrile,btanal,soronol methylethyl ketoneethylacrylate and methylmethacrylate.Wideyariationoftedata2 70%RSD)in the subsequent analysis over several weeks indicated thepresence of fesidual activityfrom the canistersmetal surface. Another approach toovercome themoistureproblem isto eliminate the sampling step altogether by using real-time detection. Iontrapmass spectrometer (TMS)is beingevaluatedas a dynamic deteetion and identification system Atmospherc pressure inletandglowdischarge ionization (API/GD)isbeing investigated by EPA forair sample introduction intoIMs.Another inlet systemusing adirect sniffr probe fitedwithneede valve wasreported byDBerberichMonsantoCompany）Both systems showed lowppbsensitivity(160ppb)andtherMSprovided massscanningandms/ms capability for identification cf unknown.

# NICOTINE.INENVIRONMENTALTOBACCO SMOKE

D.Eatough (Brigham Young University)presented theresults on cabin air quality study incommercial aircraft Future study was halted when the smoking ban became effective January of this year.Thisstudy was conducted inaDC-1O aircrat with the following objectives(1）to quantify concentrations ofETS species2）toidentifythe factors

-2-

Figure 8: An example of constructing vision labels by the image-to-label approach. Each rectangle represents a visual token span that has been successfully aligned with a word.

ThereisaveyacieradiftigofmallmedimsiedwldeinidandobaHinisriedutthfrsad aidedbytheuseootteiluefstrasdeesoiatesprteo hungiccnceesnaeymaoetDleeaosueomesad mlioninaiatsressomecoctatheaeitntbesstaindiheeaeatpresetgisathe openseasonisopaaiemoteaisctesertsded tothatthereisatiindecatiackmketforachdldmesldandetssicalucsedsexpeieuy deicaciesadtebesoomacesinertsopresmdoeasrsuteosof nese thutoeeitsejueecesst redbrocketdeerpopuatiosbeenextiratedonTobagoasaresultfovetinioserosucksdvestegrenigaae goldteguthespecacledcaianandtecommnooumaresocomonyntedandpachdeeisalsosomepacigoffuly protetedspeesndwoeysadpuoessuntmasraanrupieselotses Tniadiseaoaeseo nfasetiaeeistieitssis mostntingftspmlproomagetsdestdedss iteintefeeisoeaaetied wildifesanctuarieereissomeindicationtatteovementiseiningttaethessfwiifemanaementmoreserisit wdraedeetiesesi supportedbytectandftuegovemensandiftegeneralppueWillovetowsagreateaarenessoftemoac widlifeconservation and changethe cultureof wanton consumption tooneofsustainablemanageent.

Figure 9: An example of constructing vision labels by the label-to-image approach. The grid represents the vision tokenization result, with each box corresponding to a visual token.