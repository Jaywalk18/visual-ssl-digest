# Towards Fast and Effective Long Video Understanding of Multimodal Large Language Models via Adaptive Quasi-Gaussian Sampling

Kun Zhang Chenxin Fang Tao Chen Baiyang Song Yunhang Shen Yiyi Zhou<sup>†</sup> Rongrong Ji

Key Laboratory of Multimedia Trusted Perception and Efficient Computing, Ministry of Education of China, Xiamen University, 361005, P.R. China

{kunzhang,fangchenxin,chentao,songbaiyang,yhshen}@stu.xmu.edu.cn {zhouyiyi,rrji}@xmu.edu.cn

## Abstract

Long video understanding remains a daunting challenge for Multimodal Large Language Models (MLLMs) due to the excessive computation and memory footprint. Thus, keyframe selection is often adopted to mitigate this shortcoming, which however still suffers from low flexibility and high noise due to its hard sampling principle. In this paper, we define video frame selection as a problem of Quasi-Gaussian Sampling, and propose an adaptive and training-free approach termed AdaQ. Inspired by the 3-σ rule of Gaussian distribution, the objective of AdaQ is to achieve the optimal 3-σ interval for different examples, i.e., a smaller 3-σ interval for the local query and a larger one for the global query, thereby facilitating robust and adaptive frame sampling. To validate AdaQ, we apply it to four MLLMs with three embedding models. The extensive experimental results not only show its obvious performance gains over the default MLLMs and the SOTA keyframe selection methods, e.g., helping Qwen3-VL-8B outperform GPT4o by 15.8% on average by using only 64 frames, but also confirm its superior robustness and high efficiency for long-video understanding, e.g., only 1 hyper-parameter needs to be set. Our code project is given at https://github.com/Zkayovo-xmu/AdaQ.

## 1 Introduction

For a year or two, the research of Multimodal Large Language Models (MLLMs) Tong et al. [2025], Luo et al. [2025], Wu et al. [2024b], Heurtel-Depeiges et al. [2025], Sun et al. [2025], Yang et al. [2025b], OpenAI [2025] has made great breakthroughs in various vision-language (VL) tasks. However, long video understanding is still a daunting challenge for existing MLLMs due to the large number of video frames to process, resulting in excessive computation and memory overhead Chen et al. [2025]. For instance, to process 1k frames for an hour-long video, the advanced Qwen3-VL-8B Bai et al. [2025a] requires about 58G GPU memory and 3.28 minutes for one inference. Moreover, the ineffective video understanding not only impedes the performance of MLLMs on video tasks, but also greatly narrows down their application scopes in resource-limited scenarios.

To this end, keyframe selection is arguably a straightforward solution to handle long-video understanding for MLLMs. Inspired by the advancements in LLM-RAG [Ram et al., 2023, Jiang et al., 2023, Shi et al., 2024], recent endeavors [Shen et al., Tang et al., 2025b, Liu et al., 2025a, Zhang et al., 2025, Chen et al., 2025, Song et al., 2026] often adopt keyframe selection strategies to mitigate computational costs by reducing the number of input video frames for MLLMs. In practice, these methods will select a limited number of key frames (or clips) based on the frame-query similarities computed by VL embedding models, e.g., CLIP [Radford et al., 2021] or BLIP [Li et al., 2022]. Consequently, the input sequence length for MLLMs is significantly reduced, while maintaining the ability to perform video understanding tasks effectively using only the selected frames.

Although effective, the existing keyframe selection still suffers from two main limitations. The first one is the flexibility of handling different types of video tasks. Specifically, while keyframe selection is often superior in local video understanding, e.g., Needle-in-Haystack Wang et al. [2024a], it tends to perform poorly on the global understanding ones. This is because the keyframe selection paradigm often focus on local video snippets, leading to a much narrower vision compared to the default uniform sampling of MLLMs in terms of global understanding. The other issue is the impact of frame-query similarity noise. In the existing frame selection methods Tang et al. [2025b], Zhu et al. [2025], Zhang et al. [2025], their VL embedding models are commonly pre-trained with plain image-caption pairs, exhibiting obvious gaps to the video examples of MLLMs Chen et al. [2025]. Under this rigid selection setting, similarity noise not only declines the precision of frame selections, but also hinders the adaptive adjustment to input queries. Overall, enhancing the flexibility and robustness of frame selection remains a key challenge for MLLMs.

To achieve the above target, we redefine video frame selection as a problem of Adaptive Quasi-Gaussian Sampling, and propose a novel and training-free method termed AdaQ for MLLMs. Specifically, AdaQ is motivated and supported by the classical 3-σ rule of Gaussian distribution, where samples outside the 3-σ interval (≈ 99.73%) are regarded as the low-quality or abnormal ones Casella and Berger [2002]. To this end, the objective of AdaQ is to build an optimal 3-σ interval for the probabilistic frame sampling, i.e., a larger frame coverage for the global query while a smaller one for the local instruction. However, a naive transformation from query-frame similarities to Gaussian distribution is hard to meet this objective. As shown in Fig. 1, although the similarity scores vary across frames, the large number of frames causes the resulting probabilistic distributions become too flat to distinguish their importances Zhang et al. [2025], Tang et al. [2025a].

To address this bottleneck, the design of AdaQ is further supported by an important observation, i.e., the similarity variance of different queries provides a highly informative and robust indicator to distinguish frames. As reported in Fig. 1-c, the average similarity variances of LVBench Wang et al. [2024a] and VideoMME Fu et al. [2025] are significantly different, which are two representative benchmarks for the local and global video understanding, respectively. Motivated by this, we adopt the similarity variance of each example to change the temperature of the Quasi-Gaussian distribution, thereby adaptively adjusting 3-σ interval. In this way, AdaQ can adaptively change its sampling strategies to meet the properties of different queries, which also alleviates the noisy impact via its soft sampling manner.

To validate the proposed AdaQ, we apply it to four advanced MLLMs, including LLaVA-OneVision Li et al. [2025a], LLaVA-Video-7B Li et al. [2024], Qwen2.5-VL Bai et al. [2025b] and Qwen3-VL Bai et al. [2025a], and three VL embedding models, namely CLIP Radford et al. [2021], LongCLIP Zhang et al. [2024] and BLIP Li et al. [2022]. Extensive experiments are conducted on a set of mainstream video benchmarks Wu et al. [2024a], Fu et al. [2025], Wang et al. [2024a], Zhou et al. [2024], where a bunch of SOTA keyframe selection methods Liu et al. [2025a], Tang et al. [2025b], Zhang et al. [2025] are also comprehensively compared. The experimental results not only witness the consistent advantages of our AdaQ over the default MLLMs as well as the SOTA keyframe selection methods, e.g., improving Qwen3-VL-8B and AKS by +9.0% and +3.2% on average, but also confirm its superior robustness and better efficiency for long-video understanding. For instance, AdaQ only requires 1 hyper-parameter for most video tasks.

Overall, our contributions are three-fold:

• Motivated by the 3-σ rule of Gaussian distribution, we define video frame selection as a problem of Adaptive Quasi-Gaussian Sampling, and propose a novel and training-free framework termed AdaQ for long video understanding of MLLMs.

• We identify an important observation that different video tasks exhibit distinct task-wise similarity variance patterns, and further leverage this property to adaptively adjust the Quasi-Gaussian sampling distribution for flexible frame selection.

![](images/4fdb9e48f7206452ded91f687c978b5cc86e5059b222483434bb87bbe5fb6078.jpg)  
Figure 1: The statistics of keyframe selection and video examples. (a) Frame-query similarity scores for two types of queries, i.e., the local (top) and global (bottom) ones. (b) The Quasi-Gaussian distributions adjusted by our AdaQ based on the frame-query similarities. Without AdaQ, the default distributions are overly smooth and have excessively large 3-σ intervals. (c) Average similarity variances of LVBench and VideoMME across different embedding models. Different types of video tasks exhibit obvious distinct frame-query similarity variances. Based on this observation, we use the similarity variance to adaptively adjust the 3-σ intervals for more effective frame sampling.

• Extensive experiments on four MLLMs, three VL embedding models, and multiple longvideo benchmarks demonstrate that AdaQ consistently outperforms existing keyframe selection methods, while requiring only one hyper-parameter.

## 2 Related Work

Recent years have witnessed the rapid development of multimodal large language models (MLLMs) towards strong and generalized vision-language (VL) capabilities Hurst et al. [2024], Team et al. [2024], Lin et al. [2024], Maaz et al. [2024], Li et al. [2025a, 2024], Bai et al. [2025b,a], Wang et al. [2025]. Despite a big leap forward in image-centric tasks, long video understanding is still challenging for existing MLLMsLiu et al. [2024a]. In particular, existing MLLMs often treat an input video as a set of image patches, each encoded into hundreds of visual tokens Liu et al. [2024b], so the token sequence quickly becomes prohibitively large as videos get longer.

To this end, numerous efforts are recently devoted to efficient long video understanding for MLLMs Kim et al. [2025a], Alvar et al. [2025], Kim et al. [2025b], Xu et al. [2024], Li et al. [2025b], Shutova et al. [2025], Feng et al. [2024], Yang et al. [2025a]. One popular solution is to select query-related key frames (clips) for MLLMs Tang et al. [2025a], Ren et al. [2025], Luo et al. [2024], Huang et al. [2025], Liu et al. [2025a], Zhang et al. [2025], Chen et al. [2025], Zhu et al. [2025]. Motivated by LLM-RAG works Shi et al. [2024], Ram et al. [2023], Jiang et al. [2023], keyframe selection methods aim to identify the most informative video frames based on an external VL embedding model Radford et al. [2021], thereby reducing the length of input tokens. There are also some works similar to our probabilistic sampling settings Zhang et al. [2025], Liu et al. [2025a]. In particular, Q-Frame Zhang et al. [2025] also uses Softmax to normalize the similarity scores into a probabilistic distribution, based which Gumbel noisy is added to enhance the randomness. However, its sampling is still a hard selection based on the disturbed probabilistic values of frames. The principle of BOLT Liu et al. [2025a] is more closed to ours, which is also weighted probabilistic sampling. BOLT first adopts the cumulative distribution function to improve the sampling probabilities of key video frames, and then forms a inverse transform sampling manner. Compared with BOLT, our AdaQ targets at the adaptive and optimal 3-σ interval for frame sampling, which adopts the similarity variances of different examples to more largely adjust the Quasi-Gaussian distributions.

In addition to the above frame–query based methods, another line of research Wang et al. [2024b], Luo et al. [2024], Ma et al. [2025] leverages external tools for MLLMs. There are also alternative methodologies studied for efficient long video understanding of MLLMs, such as token pruning Liu et al. [2025b], Shao et al. [2025], Yang et al. [2025a], Tao et al. [2025] and KV cache compression

![](images/1623661397e45447b732a741ed1dc9f6d33ef7d882462373c59ca5400d554258.jpg)  
Figure 2: Illustration of the proposed AdaQ approach for the long-video understanding of MLLMs. Given a long video and a user query, a pretrained Vision-Language (VL) embedding model is used to compute the frame-query similarities, e.g., CLIP Radford et al. [2021]. Afterwards, AdaQ will transform these similarity scores into a Quasi-Gaussian probability distribution, and adopt the similarity variance, Var(s), to adjust the temperature τ of the Quasi-Gaussian distribution, thereby achieving adaptive 3-σ interval for different frame samplings.

Kim et al. [2025a], Li et al. [2025b], Shutova et al. [2025]. However, our principle and contribution are orthogonal to them, thus they are not in the scope of our comparison.

## 3 Preliminary

We first revisit the 3-σ rule of Gaussian distribution. Specifically, given a Gaussian distribution $X \sim { \mathcal { N } } ( \mu , \sigma ^ { 2 } )$ , the majority of its probability mass is concentrated within a bounded interval:

$$
\mathbb {P} (| X - \mu | \leq 3 \sigma) \approx 99.7 \%.\tag{1}
$$

In this case, the effective sampling range is regarded as the 3-σ interval, where samples outside this interval can be regarded as the low-quality or noisy ones Casella and Berger [2002].

This principle is also applicable to long video understanding. An intuitive solution is to transfer the query-frame similarities of an example into a Quasi Gaussian distribution, and define frame selection as a probability sampling task.

$$
p _ {i} \sim \hat {G} (\hat {\mu}, \hat {\sigma} (\tau)), \quad \mathrm{where} \quad p _ {i} = \frac {\exp (s _ {i} / \tau)}{\sum_ {j = 1} ^ {T} \exp (s _ {j} / \tau)},\tag{2}
$$

where $p _ { i }$ denotes the probability of video frame $V _ { i }$ being sampled, and $s _ { i }$ is the similarity score computed by the VL embedding models like CLIP Radford et al. [2021], T denotes the number of candidate frames, and $\hat { G }$ is a discrete Quasi-Gaussian distribution. Here, τ is the temperature factor that controls the shape of the resulting Quasi-Gaussian distribution, the detailed relationship between τ and the effective sampling range will be further discussed in the next section. In this case, the video frames within the 3-σ interval of $\hat { G }$ can be regarded as the query-relevant ones, while the rest ones will not be considered during sampling.

However, this direct transformation is hard to meet the target of adaptive and flexible sampling. As shown in Fig. 1, although the similarity scores yield differences across frames, the obtained Quasi Gaussian distribution becomes overly smooth, resulting in an excessively large 3-σ interval. In this case, we focus on achieving the adaptive and optimal 3-σ intervals for different examples, facilitating the flexible and robust frame sampling.

## 4 Method

In this paper, we propose a novel and training-free method termed AdaQ towards effective and efficient long video understanding of MLLMs, as illustrated in Fig. 2.

Given a long video $V$ and a user query $Q ,$ an MLLM is expected to generate an accurate answer A based on the most informative video frames. A widely used paradigm is keyframe selection based on

frame-query similarities:

$$
V _ {\text { key }} = \{V _ {i} \mid s _ {i} \in \text { Top - } K (\{s _ {t} \} _ {t = 1} ^ {T}) \}, \quad s _ {i} = \text { Emb } (V _ {i}, Q).\tag{3}
$$

Here, $s _ { i }$ denotes the similarity between frame $V _ { i }$ and the query $Q ,$ , and $T$ is the total number of candidate frames. However, this hard selection manner suffers from low flexibility and high noise, especially when the similarity distribution is flat or multi-modal.

To overcome these limitations, we formulate frame selection as an adaptive Quasi-Gaussian sampling problem as defined in Eq.2. Specifically, this Quasi-Gaussian distribution is constructed based on the peak index ${ \hat { \mu } } = \left\{ j \ | \ { \bar { p _ { j } } } = \operatorname { i n a x } P \right\}$ and the distribution fluctuation coefficient $\hat { \sigma } ( \tau )$ . In particular, $\hat { \sigma } ( \tau )$ controls the effective sampling width of the probability mass over the reordered index set $\mathcal { T } _ { : }$ and it quantifies how widely the sampling probability mass spreads around the center ${ \hat { \mu } } .$

Inspired by the $3 { - } \sigma$ principle as discussed above, the objective of AdaQ is to construct a bounded high-probability sampling interval around the most relevant region, $i . e . , 3 \ – \sigma$ interval. Frames inside this interval are preserved as valid candidates, while frames outside it are regarded as low-quality or noisy ones and suppressed from sampling. Therefore, our Quasi-Gaussian formulation explicitly models an effective sampling range, which is also the key reason why AdaQ is termed Quasi-Gaussian Sampling rather than standard probabilistic sampling.

As shown in Fig. 1, directly normalizing frame-query similarity scores may lead to an overly flat probability distribution, making it difficult to distinguish truly informative frames from ordinary ones. Under our formulation, a viable solution is to adaptively change the temperature τ to adjust the distribution curve of $\hat { G } .$ . More importantly, changing $\tau$ also means adjusting the effective sampling range of the resulting Quasi-Gaussian distribution. But when and how to dynamically set the temperature is still challenging.

To overcome this challenge, we leverage the observed variance pattern of similarity scores to dynamically adjust τ , thereby adaptively reshaping the Quasi-Gaussian distribution. Specifically, we define τ as an adaptive temperature in a variance-driven manner:

$$
\tau = \gamma \cdot \frac {1}{T} \sum_ {i = 1} ^ {T} (\tilde {s} _ {i} - \mu) ^ {2}, \quad \text { where } \tilde {s} _ {i} = \frac {s _ {i} - \min _ {j} s _ {j}}{\max _ {k} s _ {k} - \min _ {j} s _ {j}}, \quad \mu = \frac {1}{T} \sum_ {i = 1} ^ {T} \tilde {s} _ {i}.\tag{4}
$$

Here, $\gamma$ is a scaling factor for the variance term, and it is used to control the sampling temperature $\tau ,$ thus adjusting the sharpness of the resulting probability distribution. Notably, γ is also the only hyper-parameter of AdaQ. Since τ determines the spread of the Quasi-Gaussian distribution, this variance-driven design also enables AdaQ to adaptively adjust the corresponding $3 { - } \sigma$ range for different queries. In particular, a smaller variance tends to produce a narrower and more concentrated range, while a larger variance leads to a broader range with richer temporal coverage.

Based on this adaptive adjustment, we further introduce a probabilistic sampling strategy to draw keyframes from the adjusted Quasi-Gaussian distribution. Specifically, given the sampling probabilities $P = \{ p _ { i } \} _ { i \in \mathbb { Z } }$ , we first determine the effective sampling range over the reordered frame indices according to the bounded high-probability region of $\hat { G } .$ Following the intuition of Gaussian sampling, frames outside the $3 { - } \sigma$ range are regarded as low-quality candidates and directly suppressed from receiving sampling probability. Then, we perform probability-weighted sampling Tillé [2006], denoted by Sample(·), to obtain the final keyframe set $V _ { \mathrm { k e y } }$ . Each frame $V _ { i }$ within the effective range is selected with probability proportional to $p _ { i }$ , and we sample K keyframes without replacement to encourage both relevance and temporal diversity:

$$
V _ {\text { key }} \sim \text { Sample } \left(\{V _ {i} \} _ {i \in Z}, P _ {Z}, K\right),\tag{5}
$$

where $Z \subseteq \mathbb { Z }$ denotes the valid frame index set within the effective range, and $P _ { Z }$ is the renormalized probability distribution over $Z .$ Finally, the sampled frames are sorted by their original timestamps before being fed into the MLLM.

Discussion. Here, we analyze how the adaptive temperature τ adjusts the 3-σ interval of the Quasi-Gaussian distribution to achieve proper frame coverage for different types of user queries.

First of all, the temperature τ adjusts the peak value $p _ { \hat { \mu } }$ of this distribution. Specifically, for any $i \neq \hat { \mu }$ , we have

$$
\frac {p _ {i}}{p _ {\hat {\mu}}} = \exp \left(- \frac {s _ {\hat {\mu}} - s _ {i}}{\tau}\right), \qquad \tau \in (0, \infty).\tag{6}
$$

A smaller τ leads to faster exponential decay of non-peak probabilities with a higher peak value, producing a sharper 3-σ interval. In contrast, a larger τ results in a more uniform distribution with slower decay and a lower peak value, which can be defined by

$$
\lim _ {\tau \to + \infty} p _ {i} = \frac {1}{T}, \lim _ {\tau \to 0 ^ {+}} p _ {j} = \left\{ \begin{array}{l l} 1, & j = \hat {\mu}, \\ 0, & j \neq \hat {\mu}. \end{array} \right.\tag{7}
$$

Based on the cumulative distribution function (CDF) of the standard Gaussian distribution and Eq. 6, we obtain the following formulation:

$$
P (\hat {\mu} - 3 \hat {\sigma} \leq i \leq \hat {\mu} + 3 \hat {\sigma}) = \sum_ {i \in Z} p _ {i} = p _ {\hat {\mu}} \sum_ {i \in Z} \exp \left(- \frac {s _ {\hat {\mu}} - s _ {i}}{\tau}\right) = \theta ,\tag{8}
$$

where $\theta \in [ 0 , 1 ]$ is a probability value, and $Z \subseteq \mathbb { Z }$ denotes the valid subset of reordered frame indices within the effective sampling range.

Eq. 8 indicates that AdaQ does not simply sample from the full probability space, but instead constructs a bounded high-quality region centered at $\hat { \mu } ,$ following the 3-σ principle. Therefore, the size of Z directly reflects the temporal coverage of valid candidate frames.

As demonstrated in Eq. 7, we have $\tau \propto - p _ { \hat { \mu } }$ . Finally, we can obtain the relation between the temperature τ and the temporal coverage of our proposed Quasi-Gaussian sampling technique according to Eq. 8:

$$
| Z | \propto - p _ {\hat {\mu}} \exp \left(- \frac {s _ {\hat {\mu}} - s _ {i}}{\tau}\right) \propto \tau .\tag{9}
$$

Here, $| Z |$ is the size of the valid frame index set, indicating the coverage of candidate frames, and $\propto$ denotes the sign of correlation rather than strict proportionality: x ∝ y indicates a positive correlation between x and y, whereas x ∝ −y indicates a negative correlation.

According to Eq. 4 and Eq. 9, needle-in-a-haystack queries typically yield smaller similarity variance and thus smaller τ , resulting in a more compact $| Z |$ and a narrower visual receptive scope. In contrast, global understanding queries tend to have larger variance and larger τ , leading to a larger $| Z |$ and broader temporal coverage. In this case, AdaQ enables flexible and robust frame selection for video MLLMs, making it well suited to real-world scenarios with diverse user queries.

## 5 Experiments

## 5.1 Experimental Settings

Benchmarks and metrics. We evaluate AdaQ on four mainstream video benchmarks, including LongVideoBench Wu et al. [2024a], Video-MME Fu et al. [2025], LVBench Wang et al. [2024a], and MLVU Zhou et al. [2024]. LongVideoBench (LVB) and LVBench mainly target at long video understanding with needle-in-a-haystack style tasks under extended temporal contexts. And LVBench still has a small proportion of examples about global understanding, as shown in Fig.3. Video-MME emphasizes global understanding and diverse video genres across multiple duration scales. MLVU is a mixed benchmark that combines both single-detail and holistic understanding over videos with varied lengths. We use Accuracy (Acc) as the metric.

Implementation Details. We validate AdaQ on four advanced MLLMs, including LLaVA-OneVision Li et al. [2025a], LLaVA-Video Li et al. [2024], Qwen2.5-VL Bai et al. [2025b] and Qwen3-VL Bai et al. [2025a]. For each benchmark, we uniformly sample candidate frames at 1 FPS from the full video, and employ pretrained VL embedding models to compute query–frame similarity scores. The validated embedding models include CLIP Radford et al. [2021], LongCLIP Zhang et al. [2024] and BLIP Li et al. [2022]. Considering the randomness of probabilistic sampling, all AdaQ results are averaged over three trials, and the full statics are provided in the Appendix. Regarding the only hyper-parameter γ, it is set to 0.5 in most cases, except 1.5 for LongCLIP on VideoMME. As baselines, we adopt uniform sampling (uniform) and keyframe selection (Top-K), both using a sample frequency of 1 FPS. For compared keyframe selection methods, we use their official codes to reproduce results on new MLLMs and embedding models, e.g., Qwen3-VL and LongCLIP. Unless otherwise specified, all reported AdaQ results are based on the 3-σ interval setting. More analyses about different interval settings are provided in the Appendix B.1.

Table 1: Comparison between our AdaQ and existing methods on four MLLMs. Our AdaQ consistently outperforms the compared methods under most settings. Frame denotes the number of sampled video frames. Avg is the average gains compared to the default MLLM. <sup>†</sup>denotes the reproduced results based on their official codes. The best results are in bold while the second ones are underlined.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Embedding Model</td><td rowspan="2">Frames</td><td colspan="2">LongVideoBench</td><td colspan="2">Video-MME</td><td rowspan="2">LVBench</td><td rowspan="2">MLVU</td><td rowspan="2">Avg</td></tr><tr><td>Long</td><td>Overall</td><td>Long</td><td>Overall</td></tr><tr><td>LLaVA-OneVision-7B</td><td>-</td><td>32</td><td>49.5</td><td>56.8</td><td>48.1</td><td>58.3</td><td>38.6</td><td>63.7</td><td>-</td></tr><tr><td>Top-k</td><td>CLIP</td><td>32</td><td>50.7 (+1.2)</td><td>58.3 (+1.5)</td><td>48.2 (+0.1)</td><td>58.9 (+0.6)</td><td>43.8 (+5.2)</td><td>66.9 (+3.2)</td><td>+5.5%</td></tr><tr><td>FRAG</td><td>MLLM (7B)</td><td>32</td><td>-</td><td>57.6 (+0.8)</td><td>-</td><td>58.6 (+0.3)</td><td>-</td><td>65.3 (+1.6)</td><td>+1.5%</td></tr><tr><td> $OneClip-RAG^†$ </td><td>CLIP</td><td>32</td><td>54.4 (+4.9)</td><td>58.1 (+1.3)</td><td>47.7 (-0.4)</td><td>57.9 (-0.4)</td><td>44.1 (+5.5)</td><td>67.6 (+3.9)</td><td>+5.5%</td></tr><tr><td>AKS</td><td>BLIP</td><td>32</td><td>53.0 (+3.5)</td><td>59.3 (+2.6)</td><td>49.3 (+1.2)</td><td>58.4 (+0.1)</td><td>43.5 (+4.9)</td><td>66.8 (+3.1)</td><td>+5.5%</td></tr><tr><td> $Q-Frame^†$ </td><td>LongCLIP</td><td>32</td><td>54.0 (+4.5)</td><td>59.4 (+2.6)</td><td>48.3 (+0.2)</td><td>58.4 (+0.1)</td><td>44.3 (+5.7)</td><td>68.5 (+4.8)</td><td>+6.8%</td></tr><tr><td>BOLT</td><td>CLIP</td><td>32</td><td>-</td><td>59.6 (+2.8)</td><td>49.6 (+1.5)</td><td>59.9 (+1.6)</td><td>-</td><td>66.8 (+3.1)</td><td>+4.2%</td></tr><tr><td>AdaQ (ours)</td><td>CLIP</td><td>32</td><td>53.4 (+3.9)</td><td>59.8 (+3.0)</td><td>49.6 (+1.5)</td><td>59.3 (+1.0)</td><td>44.3 (+5.7)</td><td>68.5 (+4.8)</td><td>+7.3%</td></tr><tr><td>AdaQ (ours)</td><td>LongCLIP</td><td>32</td><td>55.3 (+5.8)</td><td>60.0 (+3.2)</td><td>49.0 (+0.9)</td><td>59.5 (+1.2)</td><td>44.8 (+6.2)</td><td>69.4 (+5.7)</td><td>+8.2%</td></tr><tr><td>LLaVA-Video-7B</td><td>-</td><td>64</td><td>49.6</td><td>58.9</td><td>53.2</td><td>64.2</td><td>41.9</td><td>69.5</td><td>-</td></tr><tr><td>Top-k</td><td>CLIP</td><td>64</td><td>54.6 (+5.0)</td><td>59.5 (+0.6)</td><td>53.2 (+0.0)</td><td>64.4 (+0.2)</td><td>46.9 (+5.0)</td><td>71.2 (+1.7)</td><td>+3.9%</td></tr><tr><td>FRAG</td><td>MLLM (7B)</td><td>64</td><td>-</td><td>60.6 (+1.7)</td><td>-</td><td>63.7 (-0.5)</td><td>-</td><td>69.2 (-0.3)</td><td>+0.6%</td></tr><tr><td>BOLT</td><td>CLIP</td><td>64</td><td>-</td><td>62.2 (+3.3)</td><td>-</td><td>64.6 (+0.4)</td><td>-</td><td>70.3 (+0.8)</td><td>+2.5%</td></tr><tr><td>E-VRAG</td><td>VLM (2B)</td><td>64</td><td>-</td><td>63.1 (+4.2)</td><td>-</td><td>65.4 (+1.2)</td><td>-</td><td>70.2 (+0.7)</td><td>+3.3%</td></tr><tr><td>OneClip-RAG</td><td>CLIP</td><td>64</td><td>56.2 (+6.6)</td><td>62.5 (+3.6)</td><td>54.0 (+0.8)</td><td>65.2 (+1.0)</td><td>48.2 (+6.3)</td><td>71.2 (+1.7)</td><td>+6.3%</td></tr><tr><td>AKS</td><td>BLIP</td><td>64</td><td>54.7 (+5.1)</td><td>62.7 (+3.8)</td><td>55.0 (+1.8)</td><td>65.3 (+1.1)</td><td>47.6 (+5.7)</td><td>71.8 (+2.3)</td><td>+6.3%</td></tr><tr><td> $Q-Frame^†$ </td><td>LongCLIP</td><td>64</td><td>56.9 (+7.3)</td><td>61.5 (+2.6)</td><td>53.9 (+0.7)</td><td>64.7 (+0.5)</td><td>47.1 (+5.2)</td><td>72.4 (+2.9)</td><td>+5.4%</td></tr><tr><td>AdaQ (ours)</td><td>CLIP</td><td>64</td><td>57.2 (+7.6)</td><td>63.1 (+4.2)</td><td>55.0 (+1.8)</td><td>66.0 (+1.8)</td><td>48.5 (+6.6)</td><td>72.6 (+3.1)</td><td>+7.5%</td></tr><tr><td>AdaQ (ours)</td><td>LongCLIP</td><td>64</td><td>57.4 (+8.0)</td><td>63.2 (+4.3)</td><td>54.4 (+1.2)</td><td>65.3 (+1.1)</td><td>49.1 (+7.2)</td><td>74.2 (+4.7)</td><td>+8.2%</td></tr><tr><td>Qwen2.5-VL-7B</td><td>-</td><td>64</td><td>50.7</td><td>60.1</td><td>52.9</td><td>63.7</td><td>39.3</td><td>65.5</td><td>-</td></tr><tr><td>Top-k</td><td>CLIP</td><td>64</td><td>56.4 (+5.7)</td><td>63.5 (+3.4)</td><td>54.8 (+1.9)</td><td>65.0 (+1.3)</td><td>46.7 (+7.4)</td><td>70.7 (+5.2)</td><td>+8.6%</td></tr><tr><td> $OneClip-RAG^†$ </td><td>CLIP</td><td>64</td><td>55.7 (+5.0)</td><td>62.2 (+2.1)</td><td>56.3 (+3.4)</td><td>65.3 (+1.6)</td><td>46.5 (+7.2)</td><td>68.6 (+3.1)</td><td>+7.3%</td></tr><tr><td> $AKS^†$ </td><td>BLIP</td><td>64</td><td>56.6 (+5.9)</td><td>63.8 (+3.7)</td><td>54.4 (+1.5)</td><td>64.6 (+0.9)</td><td>46.4 (+7.1)</td><td>69.6 (+4.1)</td><td>+8.0%</td></tr><tr><td> $Q-Frame^†$ </td><td>LongCLIP</td><td>64</td><td>58.4 (+7.7)</td><td>64.8 (+4.7)</td><td>54.0 (+1.1)</td><td>64.5 (+0.8)</td><td>46.5 (+7.2)</td><td>72.8 (+7.3)</td><td>+9.6%</td></tr><tr><td>AdaQ (ours)</td><td>CLIP</td><td>64</td><td>57.7 (+7.0)</td><td>64.4 (+4.3)</td><td>55.4 (+2.5)</td><td>65.5 (+1.8)</td><td>47.0 (+7.7)</td><td>71.8 (+6.3)</td><td>+9.8%</td></tr><tr><td>AdaQ (ours)</td><td>LongCLIP</td><td>64</td><td>58.9 (+8.2)</td><td>65.5 (+5.4)</td><td>55.6 (+2.7)</td><td>65.9 (+2.2)</td><td>47.1 (+7.8)</td><td>73.1 (+7.6)</td><td>+11.0%</td></tr><tr><td>Qwen3-VL-8B</td><td>-</td><td>64</td><td>50.7</td><td>62.2</td><td>56.9</td><td>67.6</td><td>43.6</td><td>71.0</td><td>-</td></tr><tr><td>Top-k</td><td>CLIP</td><td>64</td><td>57.6 (+6.9)</td><td>64.1 (+1.9)</td><td>57.3 (+0.4)</td><td>67.7 (+0.1)</td><td>49.2 (+5.6)</td><td>73.2 (+2.2)</td><td>+4.8%</td></tr><tr><td> $OneClip-RAG^†$ </td><td>CLIP</td><td>64</td><td>58.3 (+7.6)</td><td>65.1 (+2.9)</td><td>56.8 (-0.1)</td><td>67.3 (-0.3)</td><td>50.5 (+6.9)</td><td>71.5 (+0.5)</td><td>+5.2%</td></tr><tr><td> $AKS^†$ </td><td>BLIP</td><td>64</td><td>56.9 (+8.0)</td><td>65.2 (+2.4)</td><td>59.0 (+2.1)</td><td>68.6 (+1.0)</td><td>49.0 (+5.6)</td><td>74.2 (+3.2)</td><td>+5.8%</td></tr><tr><td> $Q-Frame^†$ </td><td>LongCLIP</td><td>64</td><td>58.3 (+7.6)</td><td>65.0 (+2.8)</td><td>57.2 (+0.3)</td><td>67.9 (+0.3)</td><td>50.2 (+6.6)</td><td>74.7 (+3.7)</td><td>+6.3%</td></tr><tr><td>AdaQ (ours)</td><td>CLIP</td><td>64</td><td>59.1 (+8.4)</td><td>66.5 (+4.3)</td><td>59.6 (+2.7)</td><td>69.8 (+2.2)</td><td>50.6 (+7.0)</td><td>75.0 (+4.0)</td><td>+8.0%</td></tr><tr><td>AdaQ (ours)</td><td>LongCLIP</td><td>64</td><td>59.8 (+9.1)</td><td>66.9 (+4.7)</td><td>59.4 (+2.5)</td><td>69.6 (+2.0)</td><td>51.4 (+7.8)</td><td>76.4 (+5.4)</td><td>+9.0%</td></tr></table>

## 5.2 Experimental Results

## 5.2.1 Quantitative Analysis

Comparison with existing keyframe selection methods. We first compare our AdaQ to existing keyframe selection methods on four MLLMs, as reported in Tab. 1. We first observe that the naive keyframe selection method is a strong baseline despite its simplicity. But its advantages mainly lie in the local understanding tasks as discussed above, e.g., LongVideo and LVBench. Similar cases can also be seen on recent approach OneClip-RAG, which focuses on selecting relevant clips via frame-query similarities. In comparison, advanced keyframe selection methods often exhibit more balanced performance across tasks. For instance, AKS and Q-Frame obtain decent results on both LVBench and MLVU, mainly due to their adaptive and flexible sampling designs. Compared with these SOTA methods, our AdaQ achieves the best performance under most settings with average improvements of 3.2% and 2.7% on Qwen3-VL, respectively. More importantly, our advantages are consistent across MLLMs. Also shown in Tab. 2, AdaQ still outperforms the two SOTA methods on all three embedding models. Overall, these results greatly validate our superiority and robustness.

Analysis of keyframe selection. We further make a thorough comparison to keyframe selection (Top K) in Fig. 3 and Tab. 2-3. The results of Fig. 3 confirm our argument about the inferior flexibility of keyframe selection. Although keyframe selection performs well on the local understanding tasks of LVBench and MLVU, i.e., NIH and Single-Detail, it also suffers from obvious drops in the global (holistic) ones, e.g., -6.2% than uniform sampling (Uniform) of Qwen3-VL-8B on LVBench-Global.

![](images/47f36fc3e22b1552966f01240e45505cc1152ece823f70a6556f8d34a4a88c3b.jpg)

![](images/52b5fdf6dd8b767868832ebeeaa1a8b346b70e096b4b1ff9c7ba9b67ac7600be.jpg)

![](images/129bb5e46a807d351ae96043f98f979d3f680fc5f6851aed5e47a2cee18d6cb5.jpg)

![](images/ebcc26e6904890240094d813a7a7f0806e419b58936eec32f7fb32f7131ce891.jpg)  
Figure 3: Comparison between AdaQ, keyframe selection (Top-K) and uniform sampling. Keyframe selection often perform well on the local understanding tasks, i.e., needle-in-a-haystack (NIH) and single-detail, but it is inferior in the global ones due to its rigid sampling strategy. In contrast, our AdaQ is capable of adaptive and flexible sampling for different queries.

Table 2: Comparisons with different VL embedding models. LCLIP denotes LongCLIP. The number of frames is 64. In comparison, AdaQ is more robust to embedding models.

<table><tr><td rowspan="2">Method</td><td colspan="3">LongVideoBench</td><td colspan="3">MLVU</td></tr><tr><td>CLIP</td><td>BLIP</td><td>LCLIP</td><td>CLIP</td><td>BLIP</td><td>LCLIP</td></tr><tr><td>LLaVA-Video</td><td colspan="3">58.9</td><td colspan="3">69.5</td></tr><tr><td>Top-k</td><td>59.5+0.6</td><td>61.3+2.4</td><td>62.0+3.1</td><td>71.0+1.5</td><td>72.4+2.9</td><td>73.0+3.5</td></tr><tr><td>Q-Frame</td><td>62.4+3.5</td><td>62.6+3.7</td><td>62.1+3.2</td><td>71.5+2.0</td><td>72.8+3.3</td><td>72.9+3.4</td></tr><tr><td>AKS</td><td>62.2+3.3</td><td>62.7+3.8</td><td>62.0+3.1</td><td>71.2+1.7</td><td>71.8+2.3</td><td>72.0+2.5</td></tr><tr><td>AdaQ</td><td>63.1+4.2</td><td>63.2+4.3</td><td>63.2+4.3</td><td>72.0+2.5</td><td>73.4+3.9</td><td>74.2+4.7</td></tr><tr><td>Qwen3-VL</td><td colspan="3">62.2</td><td colspan="3">71.0</td></tr><tr><td>Top-k</td><td>64.1+1.9</td><td>65.1+2.9</td><td>64.7+2.5</td><td>73.2+2.2</td><td>74.2+3.2</td><td>74.6+3.6</td></tr><tr><td>Q-Frame</td><td>64.7+2.5</td><td>66.0+3.8</td><td>65.0+2.8</td><td>73.9+2.9</td><td>74.0+3.0</td><td>74.7+3.7</td></tr><tr><td>AKS</td><td>64.6+2.4</td><td>65.2+3.0</td><td>66.0+3.8</td><td>73.1+2.1</td><td>73.9+2.9</td><td>75.3+4.3</td></tr><tr><td>AdaQ</td><td>66.5+4.3</td><td>66.7+4.5</td><td>66.9+4.7</td><td>75.0+4.0</td><td>75.4+4.4</td><td>76.4+5.4</td></tr></table>

Table 3: Ablation of the number of sampled video frames.

<table><tr><td>Model</td><td>Frames</td><td>LVB</td><td>Video-MME</td><td>LVBench</td><td>MLVU</td><td>Avg</td></tr><tr><td colspan="7">Qwen3-VL-8B</td></tr><tr><td>Uniform</td><td>32</td><td>59.8</td><td>64.5</td><td>40.6</td><td>67.1</td><td>-</td></tr><tr><td>Top-k</td><td>32</td><td>62.2 +2.4</td><td>66.1 +1.6</td><td>43.4 +2.8</td><td>70.4 +3.3</td><td>+4.6%</td></tr><tr><td>AdaQ</td><td>32</td><td>63.8 +4.0</td><td>67.7 +3.2</td><td>46.8 +6.2</td><td>72.0 +4.9</td><td>+8.6%</td></tr><tr><td>Uniform</td><td>64</td><td>62.2</td><td>66.9</td><td>43.6</td><td>71.0</td><td>-</td></tr><tr><td>Top-k</td><td>64</td><td>64.1 +1.9</td><td>68.1 +1.2</td><td>49.2 +5.6</td><td>73.2 +2.2</td><td>+5.2%</td></tr><tr><td>AdaQ</td><td>64</td><td>66.5 +4.3</td><td>69.8 +2.9</td><td>50.6 +7.0</td><td>75.0 +4.0</td><td>+8.2%</td></tr><tr><td>Uniform</td><td>128</td><td>64.8</td><td>69.9</td><td>47.6</td><td>74.7</td><td>-</td></tr><tr><td>Top-k</td><td>128</td><td>66.6 +1.8</td><td>70.2 +0.3</td><td>51.8 +4.2</td><td>75.3 +0.6</td><td>+3.2%</td></tr><tr><td>AdaQ</td><td>128</td><td>66.8 +2.0</td><td>70.6 +0.7</td><td>52.8 +5.2</td><td>76.4 +1.7</td><td>+4.3%</td></tr><tr><td>Uniform</td><td>256</td><td>65.7</td><td>70.5</td><td>52.0</td><td>77.0</td><td>-</td></tr><tr><td>Top-k</td><td>256</td><td>66.7 +1.0</td><td>70.9 +0.4</td><td>54.5 +2.5</td><td>77.3 +0.3</td><td>+1.8%</td></tr><tr><td>AdaQ</td><td>256</td><td>67.7 +2.0</td><td>71.3 +0.8</td><td>55.0 +3.0</td><td>77.7 +0.7</td><td>+2.7%</td></tr></table>

In stark contrast, AdaQ obtains consistent gains for both local and global tasks compared to the default uniform sampling, e.g., it improves 3.1% and 4.0% under LLaVA-Video-7B and Qwen3-VL-8B on MLVU, respectively. These results well confirm the superior flexibility of our AdaQ for video tasks.

Tab. 2 illustrates the dependence of keyframe selection on the capabilities of VL embedding models. Given a weaker embedding model like CLIP, LLaVA-Video-7B receives obviously worse results on the video benchmarks, especially in the local understanding task where it excels, e.g., -2.5% compared to LongCLIP on LongVideoBench. These results also confirm our argument about the similarity noise impact of keyframe selection in terms of its hard selection manner. Compared with it, our AdaQ shows much better robustness to embedding models. Tab. 3 shows the results of keyframe selection and our AdaQ using different numbers of sampled frames. It can be seen that using more video frames is a solution to mitigate the shortcomings of keyframe selection, but its gains to uniform sampling also become marginal. In comparison, our AdaQ reaches the performance upper-bound by using much fewer frames. Overall, these results not only confirm the argued shortcomings of keyframe selection, but also confirm the soft and adaptive principle of our AdaQ.

Comparison with SOTA Video-MLLMs. In Tab. 4, we also apply AdaQ to two Qwen3-VL models and compare them with a set of SOTA video-MLLMs. It can be seen that although Qwen3-VL series are advanced and powerful MLLMs, they still lag behind other MLLMs which uses larger parameter size or more video frames. These two factors are critical for existing MLLMs to pursuit stronger video capabilities. However, with the help of AdaQ, Qwen3-VL-8B and Qwen3-VL-30B can boost their performance on all benchmarks, even outperform-

Table 4: Comparison between Qwen3-VL+AdaQ and existing SOTA MLLMs. With AdaQ, Qwen3-VL can outperform several larger MLLMs.

<table><tr><td>Method</td><td>Frames</td><td>LVB</td><td>Video-MME</td><td>LVBench</td><td>MLVU</td></tr><tr><td>GPT-4o</td><td>256/0.5fps</td><td>66.7</td><td>71.9</td><td>34.7</td><td>64.6</td></tr><tr><td>Gemini-1.5-Pro</td><td>256</td><td>64.0</td><td>75.0</td><td>33.1</td><td>-</td></tr><tr><td>Qwen2.5-VL-72B</td><td>1fps</td><td>60.7</td><td>73.3</td><td>47.3</td><td>74.6</td></tr><tr><td>Kimi-VL-16B-A3B</td><td>64</td><td>64.5</td><td>67.8</td><td>-</td><td>74.2</td></tr><tr><td>LLaVA-Video-72B</td><td>64</td><td>63.9</td><td>70.0</td><td>45.5</td><td>74.4</td></tr><tr><td>InternVL3.5-38B</td><td>64</td><td>65.7</td><td>70.9</td><td>-</td><td>77.0</td></tr><tr><td>Qwen3-VL-8B</td><td>64</td><td>62.2</td><td>66.9</td><td>43.6</td><td>71.0</td></tr><tr><td>+AdaQ</td><td>64</td><td>66.9</td><td>69.4</td><td>51.4</td><td>76.4</td></tr><tr><td>Qwen3-VL-30B</td><td>64</td><td>67.2</td><td>69.9</td><td>44.0</td><td>72.8</td></tr><tr><td>+AdaQ</td><td>64</td><td>70.2</td><td>71.1</td><td>52.4</td><td>77.8</td></tr></table>

![](images/65ee0305ccf19cdc63b2ab8016305519a7958795acc12055cf2db256a2be5dd4.jpg)  
Figure 5: Visualized comparisons between Top-k, AKS and our AdaQ. For each example, their selected frames are given at the bottom, and the curves are the similarity distribution for Top-K and AKS, while the ones of AdaQ are the adjusted probability distribution. The right pannels indicate that whether the required cues are covered, e.g., calculator and playing cards. In comparison, AdaQ can provide more adaptive sampling strategies to obtain more suitable temporal coverages.

ing the close-source MLLMs by a large margin. For instance, Qwen3-VL-30B obtains 3.5%, 17.7% and 13.2% gains over GPT-4o on LVB, LVBench and MLVU, respectively. These notable results further validate the effectiveness of our AdaQ and also solidify our contribution to the community.

Ablation Studies of hyper-parameter. In Fig. 4, we report the sensitivity of AdaQ to its only hyperparameter, i.e., γ in Eq. 4 for adjusting the Quasi-Gaussian distribution. From these plots, we can see that AdaQ is not very sensitive to the selection of γ. Although it changes still bring slight performance variations, a direct choice of 0.5 can already obtain high performance for all MLLMs on most benchmarks. These results demonstrate that our AdaQ is a novel and neat method, and its simple and robust setup can well facilitate its practical use. More analyses are in Appendix.

![](images/2733860b90e17972dcefc9f7fce5f042b848e9e679e1433e019ab9c0505b248b.jpg)  
Figure 4: Ablation of hyper-parameter γ. AdaQ is robust and γ = 0.5 works well across settings.

## 5.2.2 Qualitative Analysis

In Fig. 5, we visualize the sampled frames and prediction of keyframe selection (Top-k), AKS and AdaQ. For each example, the right panels show whether the required clues are covered by retrieved visual evidence, e.g., calculator and playing cards. As observed, AdaQ consistently retrieves more instruction-relevant frames with better temporal coverage, leading to more complete evidence grounding. In contrast, Top-k and AKS may miss key clues or over-focus on local peaks, resulting in fragmented evidence and unreliable reasoning. Conclusively, AdaQ enables the MLLM to answer more accurately in both multi-detail reasoning (left) and global understanding (right) tasks.

## 6 Conclusion

In this paper, we propose a novel and training-free approach termed AdaQ for the effective long video understanding of MLLMs. Different from existing hard selection approaches, AdaQ defines video frame selection as a problem of Adaptive Quasi-Gaussian sampling, and applies the observed task-wise similarity variance to adaptively change the optimal 3-σ intervals, thereby achieving a flexible and robust sampling strategies for different queries. The extensive experiments on 4 MLLMs and 3 VL embedding models not only witness its superiority than the default MLLMs and the compared methods, e.g., helping Qwen3-VL-8B outperform GPT4o by 15.8% on average, but also well confirm its robustness to real-world applications.

## References

Saeed Ranjbar Alvar, Gursimran Singh, Mohammad Akbari, and Yong Zhang. Divprune: Diversitybased visual token pruning for large multimodal models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 9392–9401, 2025.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025a. doi: 10.48550/arXiv.2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025b.

George Casella and Roger L. Berger. Statistical Inference. Duxbury, Pacific Grove, CA, 2nd edition, 2002. ISBN 978-0534243128.

Tao Chen, Shaobo Ju, Qiong Wu, Chenxin Fang, Kun Zhang, Jun Peng, Hui Li, Yiyi Zhou, and Rongrong Ji. Towards effective and efficient long video understanding of multimodal large language models via one-shot clip retrieval. arXiv preprint arXiv:2512.08410, 2025.

Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, and S Kevin Zhou. Ada-kv: Optimizing kv cache eviction by adaptive budget allocation for efficient llm inference. arXiv preprint arXiv:2407.11550, 2024.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, Ran He, and Xing Sun. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. In CVPR, pages 24108–24118, 2025.

David Heurtel-Depeiges, Anian Ruoss, Joel Veness, and Tim Genewein. Compression via pre-trained transformers: A study on byte-level multimodal data. In Forty-second International Conference on Machine Learning, 2025.

De-An Huang, Subhashree Radhakrishnan, Zhiding Yu, and Jan Kautz. Frag: Frame selection augmented generation for long video and long document understanding. arXiv preprint arXiv:2504.17447, 2025.

Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, et al. Gpt-4o system card. arXiv preprint arXiv:2410.21276, 2024.

Zhengbao Jiang, Frank F Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. Active retrieval augmented generation. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 7969–7992, 2023.

Jang-Hyun Kim, Jinuk Kim, Sangwoo Kwon, Jae W. Lee, Sangdoo Yun, and Hyun Oh Song. Kvzip: Query-agnostic KV cache compression with context reconstruction. CoRR, abs/2505.23416, 2025a. doi: 10.48550/ARXIV.2505.23416.

Junhyuck Kim, Jongho Park, Jaewoong Cho, and Dimitris Papailiopoulos. Lexico: Extreme KV cache compression via sparse coding over universal dictionaries. In Forty-second International Conference on Machine Learning, ICML 2025. OpenReview.net, 2025b.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, et al. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. Trans. Mach. Learn. Res., 2025, 2025a.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International conference on machine learning, pages 12888–12900. PMLR, 2022.

Junyan Li, Yang Zhang, Muhammad Yusuf Hassan, Talha Chafekar, Tianle Cai, Zhile Ren, Pengsheng Guo, Foroozan Karimzadeh, Colorado Reed, Chong Wang, and Chuang Gan. Commvq: Commutative vector quantization for KV cache compression. In Forty-second International Conference on Machine Learning, ICML 2025. OpenReview.net, 2025b.

Bin Lin, Yang Ye, Bin Zhu, Jiaxi Cui, Munan Ning, Peng Jin, and Li Yuan. Video-llava: Learning united visual representation by alignment before projection. In EMNLP, pages 5971–5984, 2024.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. Improved baselines with visual instruction tuning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 26296–26306, 2024a.

Haotian Liu, Chunyuan Li, Yuheng Li, Bo Li, Yuanhan Zhang, Sheng Shen, and Yong Jae Lee. Llava-next: Improved reasoning, ocr, and world knowledge, January 2024b.

Shuming Liu, Chen Zhao, Tianqi Xu, and Bernard Ghanem. Bolt: Boost large vision-language model without training for long-form video understanding, 2025a.

Yudong Liu, Jingwei Sun, Yueqian Lin, Jianyi Zhang, Jingyang Zhang, Ming Yin, Qinsi Wang, Hai Li, and Yiran Chen. Keyframe-oriented vision token pruning: Enhancing efficiency of large vision language models on long-form video processing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 20802–20811, 2025b.

Gen Luo, Yiyi Zhou, Yuxin Zhang, Xiawu Zheng, Xiaoshuai Sun, and Rongrong Ji. Feast your eyes: Mixture-of-resolution adaptation for multimodal large language models. In ICLR, 2025.

Yongdong Luo, Xiawu Zheng, Xiao Yang, Guilin Li, Haojia Lin, Jinfa Huang, Jiayi Ji, Fei Chao, Jiebo Luo, and Rongrong Ji. Video-rag: Visually-aligned retrieval-augmented long video comprehension. arXiv Preprint, 2024. https://arxiv.org/abs/2411.13093.

Ziyu Ma, Chenhui Gou, Hengcan Shi, Bin Sun, Shutao Li, Hamid Rezatofighi, and Jianfei Cai. Drvideo: Document retrieval based long video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18936–18946, 2025.

Muhammad Maaz, Hanoona Abdul Rasheed, Salman Khan, and Fahad Khan. Video-chatgpt: Towards detailed video understanding via large vision and language models. In ACL, pages 12585–12602, 2024.

OpenAI. Gpt-4o system card. https://openai.com/index/gpt-4o-system-card/, 2025. Accessed: 2026-01-19.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.

Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 11:1316–1331, 2023.

Xubin Ren, Lingrui Xu, Long Xia, Shuaiqiang Wang, Dawei Yin, and Chao Huang. Videorag: Retrieval-augmented generation with extreme long-context videos. arXiv Preprint, 2025. https: //arxiv.org/abs/2502.01549.

Kele Shao, Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Holitom: Holistic token merging for fast video large language models. arXiv preprint arXiv:2505.21334, 2025.

Xiaoqian Shen, Yunyang Xiong, Changsheng Zhao, Lemeng Wu, Jun Chen, Chenchen Zhu, Zechun Liu, Fanyi Xiao, Balakrishnan Varadarajan, Florian Bordes, et al. Longvu: Spatiotemporal adaptive compression for long video-language understanding. In Forty-second International Conference on Machine Learning.

Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Richard James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. Replug: Retrieval-augmented black-box language models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Compu tational Linguistics: Human Language Technologies (Volume 1: Long Papers), pages 8371–8384, 2024.

Alina Shutova, Vladimir Malinovskii, Vage Egiazarian, Denis Kuznedelev, Denis Mazur, Nikita Surkov, Ivan Ermakov, and Dan Alistarh. Cache me if you must: Adaptive key-value quantization for large language models. In Forty-second International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025. OpenReview.net, 2025.

Baiyang Song, Jun Peng, Yuxin Zhang, Guangyao Chen, Feidiao Yang, and Jianyuan Guo. KTV: Keyframes and key tokens selection for efficient training-free video LLMs. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 40, pages 9060–9068, 2026. doi: 10.1609/aaai.v40i11.37862. URL https://doi.org/10.1609/aaai.v40i11.37862.

Zhen Sun, Yunhang Shen, Jie Li, Xing Sun, Pingyang Dai, Liujuan Cao, and Rongrong Ji. Dsvlm: Diffusion supervision vision language model. In Forty-second International Conference on Machine Learning, 2025.

Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive keyframe sampling for long video understanding. In CVPR, pages 29118–29128, 2025a.

Xi Tang, Jihao Qiu, Lingxi Xie, Yunjie Tian, Jianbin Jiao, and Qixiang Ye. Adaptive keyframe sampling for long video understanding. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 29118–29128, 2025b.

Keda Tao, Can Qin, Haoxuan You, Yang Sui, and Huan Wang. Dycoke: Dynamic compression of tokens for fast video large language models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 18992–19001, 2025.

Gemini Team, Petko Georgiev, Ving Ian Lei, Ryan Burnell, Libin Bai, Anmol Gulati, Garrett Tanzer, Damien Vincent, Zhufeng Pan, Shibo Wang, et al. Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context. arXiv preprint arXiv:2403.05530, 2024.

Yves Tillé. Sampling Algorithms. Springer, New York, NY, 2006. doi: 10.1007/0-387-34240-0.

Bo Tong, Bokai Lai, Yiyi Zhou, Gen Luo, Yunhang Shen, Ke Li, Xiaoshuai Sun, and Rongrong Ji. Flashsloth : Lightning multimodal large language models via embedded visual compression. In CVPR, pages 14570–14581, 2025.

Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. Lvbench: An extreme long video understanding benchmark. arXiv Preprint, 2024a. https://arxiv.org/abs/2406.08035.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, Zhaokai Wang, Zhe Chen, Hongjie Zhang, Ganlin Yang, Haomin Wang, Qi Wei, Jinhui Yin, Wenhao Li, Erfei Cui, Guanzhou Chen, Zichen Ding, Changyao Tian, Zhenyu Wu, JingJing Xie, Zehao Li, Bowen Yang, Yuchen Duan, Xuehui Wang, Zhi Hou, Haoran Hao, Tianyi Zhang, Songze Li, Xiangyu Zhao, Haodong Duan, Nianchen Deng, Bin Fu, Yinan He, Yi Wang, Conghui He, Botian Shi, Junjun He, Yingtong Xiong, Han Lv, Lijun Wu, Wenqi Shao, Kaipeng Zhang, Huipeng Deng, Biqing Qi, Jiaye Ge, Qipeng Guo, Wenwei Zhang, Songyang Zhang, Maosong Cao, Junyao Lin, Kexian Tang, Jianfei Gao, Haian Huang, Yuzhe Gu, Chengqi Lyu, Huanze Tang, Rui Wang, Haijun Lv, Wanli Ouyang, Limin Wang, Min Dou, Xizhou Zhu, Tong Lu, Dahua Lin, Jifeng Dai, Weijie Su, Bowen Zhou, Kai Chen, Yu Qiao, Wenhai Wang, and Gen Luo. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv Preprint, 2025. https://arxiv.org/abs/2508.18265.

Xiaohan Wang, Yuhui Zhang, Orr Zohar, and Serena Yeung-Levy. Videoagent: Long-form video understanding with large language model as agent. In European Conference on Computer Vision, pages 58–76. Springer, 2024b.

Haoning Wu, Dongxu Li, Bei Chen, and Junnan Li. Longvideobench: A benchmark for long-context interleaved video-language understanding. In NeurIPS, 2024a.

Qiong Wu, Wenhao Lin, Weihao Ye, Yiyi Zhou, Xiaoshuai Sun, and Rongrong Ji. Accelerating multimodal large language models via dynamic visual-token exit and the empirical findings. arXiv Preprint, 2024b. https://arxiv.org/abs/2411.19628.

Yuhui Xu, Zhanming Jie, Hanze Dong, Lei Wang, Xudong Lu, Aojun Zhou, Amrita Saha, Caiming Xiong, and Doyen Sahoo. Think: Thinner key cache by query-driven pruning. arXiv preprint arXiv:2407.21018, 2024.

Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Chendi Li, Jinghua Yan, Yu Bai, Ponnuswamy Sadayappan, Xia Hu, et al. Topv: Compatible token pruning with inference time optimization for fast and low-memory multimodal vision language model. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 19803–19813, 2025a.

Rui Yang, Lin Song, Yicheng Xiao, Runhui Huang, Yixiao Ge, Ying Shan, and Hengshuang Zhao. Haplovl: A single-transformer baseline for multi-modal understanding. In Forty-second International Conference on Machine Learning, 2025b.

Beichen Zhang, Pan Zhang, Xiaoyi Dong, Yuhang Zang, and Jiaqi Wang. Long-clip: Unlocking the long-text capability of clip. In European conference on computer vision, pages 310–325. Springer, 2024.

Shaojie Zhang, Jiahui Yang, Jianqin Yin, Zhenbo Luo, and Jian Luan. Q-frame: Query-aware frame selection and multi-resolution adaptation for video-llms. arXiv preprint arXiv:2506.22139, 2025.

Junjie Zhou, Yan Shu, Bo Zhao, Boya Wu, Shitao Xiao, Xi Yang, Yongping Xiong, Bo Zhang, Tiejun Huang, and Zheng Liu. MLVU: A comprehensive benchmark for multi-task long video understanding. arXiv Preprint, 2024. https://arxiv.org/abs/2406.04264.

Zirui Zhu, Hailun Xu, Yang Luo, Yong Liu, Kanchan Sarkar, Zhenheng Yang, and Yang You. Focus: Efficient keyframe selection for long video understanding. arXiv preprint arXiv:2510.27280, 2025.

Table 5: Comparison of different post-sampling interval settings on Qwen3-VL-8B across four longvideo benchmarks. Probs-Ori samples from the original probability distribution over all frames, while Probs-Adjusted uses the variance-adjusted distribution. AdaQ further restricts sampling to a high-probability interval, where $2 { - } \sigma$ and 3-σ denote intervals with different cumulative probability coverage. Avg denotes the average relative gain over uniform sampling. The best results are in bold.

<table><tr><td>Method</td><td>Interval</td><td>LongVideoBench</td><td>Video-MME</td><td>LVBench</td><td>MLVU</td><td>Avg</td></tr><tr><td>Uniform</td><td>-</td><td>62.2</td><td>67.6</td><td>43.6</td><td>71.0</td><td>-</td></tr><tr><td>Probs-Ori</td><td>All</td><td>62.5 (+0.3)</td><td>68.0 (+0.4)</td><td>44.1 (+0.5)</td><td>70.8 (-0.2)</td><td>+0.5%</td></tr><tr><td>Probs-Adjusted</td><td>All</td><td>66.0 (+3.8)</td><td>69.4 (+1.8)</td><td>50.0 (+6.4)</td><td>74.8 (+3.8)</td><td>+7.2%</td></tr><tr><td>AdaQ</td><td>2-σ</td><td>65.7 (+3.5)</td><td>69.1 (+1.5)</td><td>51.9 (+8.3)</td><td>74.6 (+3.6)</td><td>+8.0%</td></tr><tr><td>AdaQ</td><td>3-σ</td><td>66.5 (+4.3)</td><td>69.8 (+2.2)</td><td>50.6 (+7.0)</td><td>75.0 (+4.0)</td><td>+8.0%</td></tr></table>

Table 6: Per-run results (test1/2/3) and mean scores across benchmarks in Tab. 1.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Embedding Model</td><td colspan="4">LongVideoBench</td><td colspan="4">Video-MME</td><td colspan="4">LVBench</td><td colspan="4">MLVU</td></tr><tr><td>t1</td><td>t2</td><td>t3</td><td>mean</td><td>t1</td><td>t2</td><td>t3</td><td>mean</td><td>t1</td><td>t2</td><td>t3</td><td>mean</td><td>t1</td><td>t2</td><td>t3</td><td>mean</td></tr><tr><td>LLaVA-OneVision-7B</td><td rowspan="4">CLIP</td><td>60.6</td><td>59.6</td><td>59.2</td><td>59.8</td><td>59.0</td><td>59.3</td><td>59.5</td><td>59.3</td><td>44.1</td><td>44.9</td><td>43.8</td><td>44.3</td><td>67.8</td><td>68.8</td><td>68.9</td><td>68.5</td></tr><tr><td>LLaVA-Video-7B</td><td>63.6</td><td>62.8</td><td>63.0</td><td>63.1</td><td>66.4</td><td>65.7</td><td>65.8</td><td>66.0</td><td>48.1</td><td>49.4</td><td>48.0</td><td>48.5</td><td>72.2</td><td>73.0</td><td>72.6</td><td>72.6</td></tr><tr><td>Qwen2.5-VL-7B</td><td>64.0</td><td>64.4</td><td>64.8</td><td>64.4</td><td>65.4</td><td>65.4</td><td>65.8</td><td>65.5</td><td>47.6</td><td>46.0</td><td>47.3</td><td>47.0</td><td>72.4</td><td>71.1</td><td>71.8</td><td>71.8</td></tr><tr><td>Qwen3-VL-8B</td><td>66.4</td><td>66.6</td><td>66.5</td><td>66.5</td><td>69.9</td><td>70.1</td><td>69.5</td><td>69.8</td><td>49.7</td><td>51.3</td><td>50.4</td><td>50.5</td><td>75.1</td><td>74.8</td><td>75.1</td><td>75.0</td></tr><tr><td>LLaVA-OneVision-7B</td><td rowspan="4">LongCLIP</td><td>59.5</td><td>60.4</td><td>60.1</td><td>60.0</td><td>59.9</td><td>59.3</td><td>59.2</td><td>59.5</td><td>44.3</td><td>44.9</td><td>45.3</td><td>44.8</td><td>69.8</td><td>69.0</td><td>69.4</td><td>69.4</td></tr><tr><td>LLaVA-Video-7B</td><td>63.7</td><td>63.1</td><td>62.7</td><td>63.2</td><td>65.4</td><td>65.5</td><td>65.1</td><td>65.3</td><td>49.3</td><td>49.1</td><td>48.9</td><td>49.1</td><td>74.4</td><td>74.0</td><td>74.2</td><td>74.2</td></tr><tr><td>Qwen2.5-VL-7B</td><td>66.4</td><td>65.3</td><td>64.7</td><td>65.5</td><td>65.3</td><td>66.1</td><td>66.2</td><td>65.9</td><td>47.5</td><td>47.1</td><td>46.6</td><td>47.1</td><td>73.3</td><td>73.1</td><td>72.9</td><td>73.1</td></tr><tr><td>Qwen3-VL-8B</td><td>66.9</td><td>66.8</td><td>66.9</td><td>66.9</td><td>69.9</td><td>69.3</td><td>69.5</td><td>69.6</td><td>51.1</td><td>52.2</td><td>50.8</td><td>51.4</td><td>76.5</td><td>76.3</td><td>76.4</td><td>76.4</td></tr></table>

## A LVBench NIH/Global Subset Analysis

To better interpret the trends in Fig. 3, we additionally report results on two LVBench subsets split by the official time\_reference annotation, which indicates the temporal span of the ground-truth evidence. Specifically, we categorize instances with time\_reference > 4 minutes as global understanding (Global), where answering typically requires aggregating information across a long temporal context. The remaining instances are treated as Needle-In-A-Haystack (NIH), in which the required clue lies in a relatively short window and thus emphasizes precise evidence localization. This subset analysis helps reveal whether the performance changes in Fig. 3 stem from improved long-range coverage (Global) or more accurate retrieval of sparse yet critical clues (NIH).

## B Additional Experimental Details

## B.1 Comparison of different post-sampling interval settings

In Tab. 5, we further study the effect of different post-sampling interval settings. Directly sampling from the original probability distribution over all frames, i.e., Probs-Ori, brings only marginal gains over uniform sampling, indicating that the original similarity-induced distribution is still too flat to provide effective frame selection. After applying our variance-based adjustment, Probs-Adjusted achieves much stronger improvements, $\mathrm { e . g . , + 7 . 2 \% }$ average gain, which verifies the importance of adaptively reshaping the Quasi-Gaussian distribution. Moreover, by further restricting sampling to a high-probability interval, AdaQ achieves consistently strong performance under both $2 { - } \sigma$ and 3-σ settings. In particular, the $3 { - } \sigma$ interval achieves the best results on LongVideoBench, Video-MME, and MLVU, while the $2 \sigma$ interval performs better on LVBench. These results demonstrate that suppressing low-probability frames outside the effective interval is beneficial, and the broader 3-σ setting provides a robust default choice across different long-video benchmarks.

## B.2 Repeated-run Evaluation Results

In this appendix, we report detailed per-run results that are omitted in the main paper due to space limitations. All numbers in Tab. 1 are averaged over three repeated evaluations under identical inference settings (same backbone, frame budget, and evaluation protocol). For transparency, Tab. 6 lists the score of each run together with the mean, allowing readers to directly inspect run-to-run variability induced by probabilistic sampling. Furthermore, Tab. 7 provides the same three-run breakdown and mean results when varying the scoring backbones (CLIP/BLIP/LongCLIP), which complements the robustness trends summarized in Tab. 2.

Table 7: Repeated-run results under different embedding models (CLIP/BLIP/LongCLIP) in Tab. 2, reporting test1/2/3 and the mean to quantify run-to-run variability.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Dataset</td><td colspan="4">CLIP</td><td colspan="4">BLIP</td><td colspan="4">LongCLIP</td></tr><tr><td>test1</td><td>test2</td><td>test3</td><td>mean</td><td>test1</td><td>test2</td><td>test3</td><td>mean</td><td>test1</td><td>test2</td><td>test3</td><td>mean</td></tr><tr><td rowspan="2">Qwen3-VL-8B</td><td>LongVideoBench</td><td>66.4</td><td>66.6</td><td>66.5</td><td>66.5</td><td>66.2</td><td>66.5</td><td>67.5</td><td>66.7</td><td>66.9</td><td>66.8</td><td>66.9</td><td>66.9</td></tr><tr><td>MLVU</td><td>75.1</td><td>74.8</td><td>75.1</td><td>75.0</td><td>75.0</td><td>75.6</td><td>75.6</td><td>75.4</td><td>76.5</td><td>76.3</td><td>76.4</td><td>76.4</td></tr><tr><td rowspan="2">LLaVA-Video-7B</td><td>LongVideoBench</td><td>63.6</td><td>62.8</td><td>63.0</td><td>63.1</td><td>62.4</td><td>62.8</td><td>63.6</td><td>62.9</td><td>63.7</td><td>63.1</td><td>62.7</td><td>63.2</td></tr><tr><td>MLVU</td><td>72.2</td><td>73.0</td><td>72.6</td><td>72.6</td><td>73.5</td><td>73.3</td><td>73.3</td><td>73.4</td><td>74.4</td><td>74.0</td><td>74.2</td><td>74.2</td></tr></table>

Table 8: Efficiency comparison between Uniform sampling and AdaQ on Qwen3-VL-8B. Candidate Frames denotes the frames used for CLIP-based similarity computation, and Input Frames denotes the final frames fed into the MLLM.

<table><tr><td>Method</td><td>Candidate Frames</td><td>Input Frames</td><td>Memory Overhead</td><td>CLIP Cost</td><td>Selection Cost</td><td>MLLM Inference</td><td>All Time</td></tr><tr><td>Uniform</td><td>-</td><td>64</td><td>17.98 GB</td><td>-</td><td>-</td><td>2.61s</td><td>21.3s</td></tr><tr><td>Uniform</td><td>-</td><td>256</td><td>22.83 GB</td><td>-</td><td>-</td><td>7.80s</td><td>26.8s</td></tr><tr><td>AdaQ</td><td>512</td><td>64</td><td>17.98 GB</td><td>0.53s</td><td>0.001s</td><td>2.67s</td><td>21.89s</td></tr><tr><td>AdaQ</td><td>1024</td><td>64</td><td>17.98 GB</td><td>0.66s</td><td>0.001s</td><td>2.61s</td><td>22.04s</td></tr></table>

Per-run results for the main comparison. Tab. 6 presents the results of three repeated runs, along with their mean values across benchmarks, demonstrating that AdaQ consistently outperforms previous methods across multiple trials, despite the inherent randomness introduced by different random seeds.

Repeated-run results under different scoring backbones. Tab. 7 further evaluates stability when changing the embedding model used for frame–query relevance (CLIP/BLIP/LongCLIP), and shows that the observed improvements remain reliable across scoring backbones, consistent with Tab. 2.

## C Compute Resources and Runtime Analysis

Although AdaQ requires computing frame-query similarities over all candidate frames, this cost is also shared by most existing query-aware frame selection methods. Moreover, this step is performed by CLIP-like VL embedding models, which are much smaller and computationally cheaper than MLLMs. In practice, the runtime is still dominated by MLLM inference rather than similarity scoring. Therefore, the additional cost of candidate-frame scoring is relatively minor compared with the cost of directly feeding more video frames into the MLLM. To provide a concrete runtime analysis, we report the actual cost of AdaQ on Qwen3-VL-8B using one NVIDIA A800 GPU in Table 8. Compared with uniform sampling using 256 input frames, AdaQ only feeds 64 selected frames into the MLLM and thus keeps the memory overhead at 17.98 GB, while uniform sampling with 256 frames increases the memory overhead to 22.83 GB. Moreover, the selection step of AdaQ is nearly negligible, taking only 0.001s, and the total runtime remains close to uniform sampling with 64 frames even when 512 or 1024 candidate frames are scored. These results show that AdaQ introduces little selection overhead while avoiding the much higher inference cost caused by directly increasing the number of MLLM input frames.

## D Limitations

Although AdaQ achieves consistent improvements, its performance is still limited by two factors. First, some failures arise from the capability boundary of the underlying MLLM, especially for complex temporal reasoning or ambiguous visual content. Second, AdaQ relies on external VL embedding models for frame-query similarity estimation, where adaptive sampling can alleviate but not fully resolve noisy or misaligned similarities. Future work may combine adaptive frame sampling with stronger temporal reasoning and video-language alignment.

## E Broader Impacts

AdaQ aims to improve the efficiency and effectiveness of long-video understanding for MLLMs by reducing redundant visual inputs and enabling more adaptive frame sampling. This may lower the computational cost of video-based AI systems and make long-video understanding more accessible in resource-limited scenarios. However, stronger and more efficient video understanding models may also introduce potential risks when used in privacy-sensitive applications, such as surveillance or large-scale video analysis without proper consent. In addition, AdaQ relies on pretrained MLLMs and VL embedding models, and may inherit their biases, hallucinations, or failure modes, which could lead to unreliable predictions in high-stakes scenarios. Therefore, real-world deployment should follow dataset licenses, privacy regulations, and appropriate human oversight.

## F Licenses for Existing Assets

This work builds upon existing public datasets, pretrained models, VL embedding models, and baseline codebases. Specifically, we use public long-video understanding benchmarks, including LongVideoBench, Video-MME, LVBench, and MLVU, as well as publicly available MLLM backbones and VL embedding models, such as LLaVA-OneVision, LLaVA-Video, Qwen2.5-VL, Qwen3-VL, CLIP, LongCLIP, and BLIP. We properly cite the original papers and official releases of these assets in the main paper. All datasets, models, and codebases are used only for research purposes, and we follow their official licenses, terms of use, and evaluation protocols. Our released code does not redistribute the original datasets or pretrained model weights, but instead provides instructions and scripts for using the assets from their official sources. We will also include the corresponding asset names, sources, and license or usage information in the released project documentation.