# DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers

Mengping Yang1,2 Zhiyu Tan1,2† Binglei Li1,2,3 Xiaomeng Yang2 Hesen Chen1,2 Hao Li1,2,3∗ 1Fudan University 2Shanghai Academy of AI for Science 3Shanghai Innovation Institute

![](images/8c19f707b7d5f98f673ce0a7e363e154bd32f05903f80b83ae85c230e58e1c6b.jpg)  
Figure 1. Comparison between Representation Alignment [70], DispLoss [62] and our proposed DiverseDiT in learning representations. (a) REPA [70] employs external encoders as guidance and different blocks’inputs are homogeneous. (b) DispLoss [62] encourage internal representations to spread out but still with homogeneous input and without block-wise diversity. (c) We propose long residual connections to enhance input diversity and diversity loss to encourage diverse feature representations across blocks. (d) On ImageNet 256 × 256, our proposed method consistently reflects training efficiency and effectiveness when applied to both SiT and REPA.

## Abstract

Recent breakthroughs in Diffusion Transformers (DiTs) have revolutionized the field of visual synthesis due to their superior scalability. To facilitate DiTs’ capability of capturing meaningful internal representations, recent works such as REPA incorporate external pretrained encoders for representation alignment. However, the underlying mechanisms governing representation learning within DiTs are not well understood. To this end, we first systematically investigate the representation dynamics of DiTs. Through analyzing the evolution and influence of internal representations under various settings, we reveal that representation diversity across blocks is a crucial factor for effective learning. Based on this key insight, we propose DiverseDiT, a novel framework that explicitly promotes representation diversity. DiverseDiT incorporates long residual connections to diversify input representations across blocks and a representation diversity loss to encourage blocks to learn distinct features. Extensive experiments on ImageNet 256 × 256 and 512 × 512 demonstrate that our DiverseDiT yields consistent performance gains and convergence acceleration when applied to different backbones with various sizes, even when tested on the challenging one-step generation setting. Furthermore, we show that DiverseDiT is complementary to existing representation learning techniques, leading to further performance gains. Our work provides valuable insights into the representation learning dynamics of DiTs and offers a practical approach for enhancing their performance. Our code is available at https: //github.com/kobeshegu/DiverseDiT.

## 1. Introduction

Diffusion models [25, 55], particularly diffusion transformers (DiT) [2, 49] which demonstrate superior scalability in learning the data distribution, have significantly advanced the field of visual synthesis including text-toimage [51, 66], text-to-video generation [20, 69], etc.

Recent studies identified that top-performing diffusion models capture more discriminative internal representations [10, 45, 65], yielding an implicit connection between diffusion generative models and representation learning. Following this philosophy, methods like REPA [70] (Fig. 1 (a)) align latent noisy representations with features derived from pre-trained visual encoders to guide the representation learning. Subsequent work REPA-E [35] extends such alignment in a joint end-to-end training manner with VAE tuning and REG [64] entangles low-level visual latents and high-level class tokens for a two-level alignment. However, this reliance on powerful external foundation models, which require massive resources for training, poses a significant drawback. Other approaches aim to improve representations without external guidance. SRA [27] aligns representations between a student and an EMA teacher model, while Wang et al. [62] propose a dispersive loss to encourage separation between internal representations (Fig. 1 (b)). Despite their considerable advancements, the underlying mechanisms governing representation learning within DiT models remain largely opaque. Key questions persist: How do DiT models learn meaningful representations, and why are external alignment techniques effective? This lack of fundamental understanding hinders the development of more principled and efficient training paradigms.

To address this gap, we perform a systematic investigation into the representation learning process of DiT models (Sec. 2). First, we analyze how representations evolve throughout training by measuring the discrepancy between internal representations across different blocks. Then, we examine how representations are influenced when external models are employed for alignment on different blocks with multiple encoders. Through these analyses, we reveal several key findings: 1) Representation discrepancy across blocks naturally increases as training progresses; 2) Aligning a single block with a pre-trained model significantly increases its discrepancy from other blocks; 3) Crucially, aligning more blocks or using more encoders does not necessarily improve performance, suggesting that excessive alignment can harm the overall diversity of the model’s representations. These observations provide a new perspective for understanding the representation learning of DiTs and offer a plausible explanation for the effectiveness of existing techniques like REPA: the rationale behind effective representation learning in DiTs lies in improving the representation diversity across different blocks.

Capitalizing on the above insights, we propose DiverseDiT, a novel and effective framework for promoting diverse representation learning in DiTs. Specifically, DiverseDiT introduces two simple yet powerful components. First, we incorporate long-range residual connections to diversify the inputs to different blocks, preventing representational homogenization. Second, we introduce a representation diversity loss that explicitly penalizes similarity between features from different blocks. This encourages each block to specialize and capture unique, complementary aspects of the data. Together, these components promote diverse representation learning through both diverse inputs and inter-block diversity constraints, without requiring external guidance models (Fig. 1 (c)). We conduct extensive experiments to testify the effectiveness of DiverseDiT on

ImageNet 256×256 and ImageNet 512×512 datasets. The results demonstrate that our method consistently improves the training convergence and synthesis quality when applied to different baselines, with or without external guidance (Fig. 1 (d)) across a wide range of model scales, even when tested on the challenging one-step setting [17]. Moreover, we show that our proposed method is complementary to existing alternatives such as Disp [62] and SRA [27], yielding further improvement.

We summarize our primary contributions as: 1) We conduct a comprehensive analysis on how representations are learned in DiTs, revealing that improving representation diversity across blocks is a key factor for effective training. To our knowledge, this work is the first to elucidate this representation relationship and provide valuable insight to understanding representation dynamics of DiTs. 2) We propose DiverseDiT, an efficient and effective framework that facilitates representation diversity with long residual connections that enable diverse input and block diversity loss that encourages representations to be distinct. 3) Extensive experiments with different baseline models across various scales on both multi-step and one-step settings demonstrate the effectiveness of our method in accelerating convergence and improving performance.

## 2. How Representations Are Learned?

## 2.1. Preliminaries

Scalable Interpolant Transformers (SiT). Our work is based on SiT, which unifies flow [39] and diffusion [25] models that transform Gaussian noise ϵ into samples x∗:

$$
\mathbf {x} _ {t} = \alpha_ {t} \mathbf {x} _ {*} + \sigma_ {t} \epsilon , \tag {1}
$$

where $\alpha _ { t }$ decreases and $\sigma _ { t }$ increases with time t. Flowbased models interpolate between noise and data while diffusion models define a stochastic differential equation to approach Gaussian distribution as $t  \infty$ . Sampling is performed via a reverse SDE for diffusion or a probability flow ODE for flow models: $\dot { \mathbf x } _ { t } = \mathbf v ( \mathbf x _ { t } , t )$ , the velocity field $\mathbf { v } ( \mathbf { x } _ { t } , t )$ can be formulated with conditional expectation:

$$
\mathbf {v} (\mathbf {x}, t) = \mathbb {E} [ \dot {\mathbf {x}} _ {t} | \mathbf {x} _ {t} = \mathbf {x} ] = \dot {\alpha} _ {t} \mathbb {E} [ \mathbf {x} _ {*} | \mathbf {x} _ {t} = \mathbf {x} ] + \dot {\sigma} _ {t} \mathbb {E} [ \epsilon | \mathbf {x} _ {t} = \mathbf {x} ]. \tag {2}
$$

The velocity for the velocity field $\mathbf { v } ( \mathbf { x } _ { t } , t )$ is derived by a model $\mathbf { v } _ { \theta } ( \mathbf { x } _ { t } , t )$ trained to minimize:

$$
\mathbb {E} _ {\mathbf {x} _ {*}, \epsilon , t} \left[ \| \mathbf {v} _ {\theta} (\mathbf {x} _ {t}, t) - \dot {\alpha} _ {t} \mathbf {x} _ {*} - \dot {\sigma} _ {t} \epsilon \| ^ {2} \right]. \tag {3}
$$

Once trained, we can synthesize samples from random noises by the reverse SDE via computing the velocity filed:

$$
d \mathbf {x} _ {t} = \mathbf {v} (\mathbf {x} _ {t}, t) d t - \frac {1}{2} w _ {t} \mathbf {s} (\mathbf {x} _ {t}, t) d t + \sqrt {w _ {t}} d \overline {{\mathbf {w}}} _ {t}, \tag {4}
$$

![](images/373ba9c069b0ef38b50f82c2c7826baa0484867dff20cd89e6ccbf457e38b734.jpg)  
Figure 2. CKA representation similarities of models trained on various settings. We can observe that 1) the discrepancies between different blocks increases as training progresses; 2) aligning specific blocks significantly increases the dissimilarity between the corresponding block and other blocks; 3) aligning on more blocks with different pretrained encoders brings marginal performance improvements. Detailed quantitative results are provided in Sec. E.

where score $\mathbf { s } ( \mathbf { x } _ { t } , t )$ is obtained via conditional expectation:

$$
\mathbf {s} (\mathbf {x} _ {t}, t) = - \sigma_ {t} ^ {- 1} \mathbb {E} [ \epsilon | \mathbf {x} _ {t} = \mathbf {x} ] = \sigma_ {t} ^ {- 1} \frac {\alpha_ {t} \mathbf {v} (\mathbf {x} , t) - \dot {\alpha} _ {t} \mathbf {x}}{\alpha_ {t} \dot {\sigma} _ {t} - \dot {\alpha} _ {t} \sigma_ {t}}. (5)
$$

Representation Alignment (REPA). To leverage external models to aid representation learning for DiTs, REPA proposes to perform patch-wise projection alignment between the model’s intermediate hidden states h with features $\mathbf { y } _ { * }$ derived from pretrained visual encoders:

$$
\mathcal {L} _ {\mathrm{REPA}} (\theta , \phi) := - \mathbb {E} _ {\mathbf {x} _ {*}, \epsilon , t} \left[ \frac {1}{N} \sum_ {n = 1} ^ {N} \operatorname{sim} (\mathbf {y} _ {*} ^ {[ n ]}, h _ {\phi} (\mathbf {h} _ {t} ^ {[ n ]})) \right], \tag {6}
$$

where x∗ denotes clean images, $h _ { \phi }$ is MLP projectors and $\sin ( \cdot , \cdot )$ denotes similarity function.

Centered Kernel Alignment (CKA). CKA is a widely used similarity index for quantifying neural network representations [11, 12, 32]. Accordingly, we adopt CKA to calculate the similarities of representations across DiT blocks for our analysis. Formally, CKA is normalized from Hilbert-Schmidt Independence Criterion (HSIC) [18] to be invariant to orthogonal transformation and isotropic scaling:

$$
\mathrm{CKA} (\mathrm{X}, \mathrm{Y}) = \frac {\mathrm{HSIC} (\mathrm{x} , \mathrm{y})}{\sqrt {\mathrm{HSIC} (\mathrm{x} , \mathrm{x}) \mathrm{HSIC} (\mathrm{y} , \mathrm{y})}}. \tag {7}
$$

HSIC identifies whether two distributions $( \mathrm { X } , \mathrm { Y } )$ are independent: $\begin{array} { r } { \mathrm { H S I C } ( K , L ) = \frac { 1 } { ( n - 1 ) ^ { 2 } } \operatorname { T r } ( K H L H ) , K _ { i j } = } \end{array}$ $k \left( \mathbf { x } _ { i } , \mathbf { x } _ { j } \right)$ and $L _ { i j } = l \left( \mathrm { y } _ { i } , \mathrm { y } _ { j } \right)$ , where k and l are kernels.

## 2.2. Our Observations

With CKA as the representation similarity index, we systematically investigate how representations are learned and how they are affected when external representation alignment is enabled in three settings. 1) SiT training stage analysis: we track the evolution of internal representations and quantify the change in similarity between different blocks as the model learns. 2) REPA block-specific alignment: we identify the effect of aligning pretrained visual features on different blocks to assess how external knowledge alters the representation of specific blocks. 3) REPA multiple block guidance from multiple encoders: we explore the impact of applying external guidance to multiple blocks with multiple encoders to probe whether guiding multiple blocks leads to improvement. All implementation details strictly follow the settings of SiT-B/2 and REPA-B/2 on Imagenet $2 5 6 \times 2 5 6$ for 450K iterations. We use DINOv2-B, MAE-L and Mo-Cov3 for REPA alignment. The visualized results are shown in Fig. 2 and detailed quantitative results are given in Sec. $\mathrm { E , }$ we can observe several interesting findings from the results.

(1). Representation diversity across different blocks increases during training: The similarity heatmaps of SiT at different training steps (5K, 50K, 200K, 450K) show a clear trend of increasing representational diversity. Specifically, the heatmap becomes more diagonal as training progresses, and the representation between different layers becomes less similar. Intuitively, different blocks specialize, develop more distinct and complementary representations.

Such observation aligns with the broader understanding that deep models learn hierarchical representations.

(2). External alignment enhances block differentiation: The REPA heatmaps exhibit more distinct (less similar) patterns around the red mark compared to the corresponding regions in the SiT heatmaps, indicating that aligning specific blocks significantly increases the dissimilarity between the representations of the targeted block and other blocks. Additionally, consistent with REPA [70], aligning earlier blocks (i.e., Block 5, Block 8) yields better performance than aligning later blocks (Block 10). This demonstrates that external alignment effectively promotes specialization by making the selected block’s representation more different from other blocks. In other words, REPA encourages each block to learn more distinct and complementary features, leading to a more diverse and more effective representation. More importantly, these observations provide insight into why REPA-like external alignment is effective: by enforcing specialization, it prevents representational collapse and encourages the network to explore a wider range of features. This specialization-driven perspective may also explain why aligning with larger models (DINOv2-L, -g) brings only marginal improvements compared to aligning with smaller models (DINOv2-B) in the original REPA.

(3). Aligning on more blocks with more external models does not necessarily improve performance: While REPA with single blocks shows clear differentiation, using multiple blocks for guidance (e.g., Block:[2,5,8], [3,6,9]) does not bring similar improvements to the performance. In some cases, the FID score is even slightly worse (Block [2,5,8]), suggesting that applying guidance to more blocks might counterintuitively reduce the overall diversity between blocks. We hypothesize that this is due to the introduction of conflicting constraints, preventing individual blocks from effectively specializing. Furthermore, aligning multiple blocks with different external encoders ([5/Dinov2+10/MAE], i.e., aligning DinoV2 features on block 5 and MAE features on block 10) also provides limited benefit and shows limited representation diversity. Such observation further reflects that the representation diversity across blocks is a crucial factor for high-quality synthesis.

In general, our systematic analysis provides a comprehensive understanding of representation dynamics for DiTs and reveals that the key for representation learning is increasing the discrepancies of block representations. Our findings offer a novel perspective for explaining existing methods, showing how models learn representations during training and highlighting the critical role of block specialization. These observations motivate us to design more effective methods to enhance representation diversity for performance improvement and accelerated training.

## 3. Methodology

In light of the observations from our systematic analysis of representation dynamics in Sec. 2, we introduce, DiverseDiT, a novel method to explicitly enhance representation diversity. Our approach focuses on encouraging specialization via long residual connections to enhance the input diversity of different blocks and a representation diversity loss to explicitly promote diverse feature representations across all blocks. We detail each of them below.

## 3.1. Long Residual Connections

Motivated by the findings in Sec. 2, we argue that the diversity of inputs for each block also plays a crucial role in shaping the learned representations. However, conventional diffusion transformers often suffer from a lack of input diversity because each block’s input is typically homogeneous and is derived solely from the output of the preceding layer. To address this, we employ a long residual connection to inject diversity into the inputs of each block. This mechanism selectively injects the output of earlier layers into later layers, promoting feature reuse and preventing representational collapse. Formally, suppose the model consists of L DiT blocks, we connect the i-th block’s output to the $( L - i ) -$ th block via:

$$
f _ {l} = \mathcal {R} _ {\text { res }} ^ {i} (f _ {i}, f _ {l - 1}) = \text { Linear } (\text { Norm } (f _ {i} \oplus f _ {l - 1})), \tag {8}
$$

where $i \in [ 0 , . . . , L / / 2 - 1 ]$ , and $\mathcal { R } _ { \mathrm { r e s } } ^ { i }$ denotes the residual connection. $f _ { i } ~ \in ~ \mathbb { R } ^ { N \times T \times D }$ is the representation of the i-th block, ⊕ denotes concatenating the representation of two blocks $f _ { i }$ and $f _ { l - 1 }$ , which is further processed by a layer normalization and a linear layer for linear transformation. By injecting skip connections, we break the chain of homogeneous inputs and encourage the network to learn more varied and informative representations from different sources.

## 3.2. Representation Diversity Loss

To further encourage specialization and promote diversity in the learned representations, we introduce a representation diversity loss to explicitly promote diverse feature representations within each block. Specifically, our representation diversity comprises three key components: an orthogonality loss, a mutual-information minimization loss, and a feature dispersion loss. Notably, to reduce computational cost, we only consider a subset of all possible pairs from L blocks: $\mathcal { P } \subseteq \{ ( i , j ) : i < j , \ i , j \in L \}$ . For each block, we define token-wise mean feature along the N and T dimension as:

$$
\boldsymbol {\mu} _ {l} = \frac {1}{N T} \sum_ {n = 1} ^ {N} \sum_ {t = 1} ^ {T} f _ {l} [ n, t,: ] \in \mathbb {R} ^ {D}. \tag {9}
$$

Then, we compute the orthogonality loss by penalizing high cosine similarity between block-wise mean representations, encouraging cross-block orthogonality:

$$
\mathcal {L} _ {\text { orth }} = \frac {1}{| \mathcal {P} |} \sum_ {(i, j) \in \mathcal {P}} \cos (\boldsymbol {\mu} _ {i}, \boldsymbol {\mu} _ {j}) = \frac {1}{| \mathcal {P} |} \sum_ {(i, j) \in \mathcal {P}} \frac {\boldsymbol {\mu} _ {i} ^ {\top} \boldsymbol {\mu} _ {j}}{\| \boldsymbol {\mu} _ {i} \| _ {2} \| \boldsymbol {\mu} _ {j} \| _ {2}}. \tag {10}
$$

Next, we minimize mutual information between block representations to ensure statistical independence within block-wise representations. However, directly computing mutual information is computationally intractable for highdimensional features. Therefore, we use a computationally efficient proxy based on the average cosine similarity of normalized feature vectors as the estimation of mutual information. Specifically, we define flattened, ℓ2-normalized token representations along the N and T dimension as:

$$
\hat {\boldsymbol {f}} _ {l, n, t} = \frac {f _ {l} [ n , t , : ]}{\| f _ {l} [ n , t , : ] \| _ {2}} \in \mathbb {R} ^ {D}. \tag {11}
$$

Then, we compute the proxy mutual-information loss as:

$$
\mathcal {L} _ {\mathrm{MI}} = \frac {1}{| \mathcal {P} |} \sum_ {(i, j) \in \mathcal {P}} \frac {1}{N T} \sum_ {n = 1} ^ {N} \sum_ {t = 1} ^ {T} \hat {\boldsymbol {f}} _ {i, n, t} ^ {\top} \hat {\boldsymbol {f}} _ {j, n, t}. \tag {12}
$$

In this way, we avoid directly calculating covariance matrices for efficiency and meanwhile minimizing the correlation between representations. Further, we employ a feature dispersion loss to encourage diverse channel usage by maximizing the variance of feature activations. The representations of each block are flattened to $\tilde { f } _ { l } \in \mathbb { R } ^ { ( N T ) \times D }$ and normalized along the sample axis to obtain $\widehat { \tilde { f } _ { b } }$ . Then we compute the averaged activation per dimension:

$$
a = \frac {1}{| \mathcal {P} |} \sum_ {p \in \mathcal {P}} \text { mean } _ {n, t} (\widehat {\tilde {f}} _ {b} [ n, t,: ]), \tag {13}
$$

a is then normalized to $a ^ { \prime }$ by $a ^ { \prime } = a / \operatorname* { m a x } _ { k } a _ { k }$ , and its variance is maximized to obtain the feature dispersion loss:

$$
\mathcal {L} _ {\mathrm{disp}} = - \frac {1}{D} \sum_ {k = 1} ^ {D} (a _ {k} ^ {\prime} - \bar {a} ^ {\prime}) ^ {2}, \bar {a} ^ {\prime} = \frac {1}{D} \sum_ {k = 1} ^ {D} a _ {k} ^ {\prime}. \tag {14}
$$

Finally, the overall representation diversity loss aggregates the above three components as:

$$
\mathcal {L} _ {\text { div }} = \lambda_ {\text { orth }} \mathcal {L} _ {\text { orth }} + \lambda_ {\mathrm{MI}} \mathcal {L} _ {\mathrm{MI}} + \lambda_ {\text { disp }} \mathcal {L} _ {\text { disp }}, \tag {15}
$$

where $\lambda _ { \mathrm { o r t h } } , \lambda _ { \mathrm { M I } } , \lambda _ { \mathrm { d i s p } }$ control the relative weight of each loss, we set them as 0.33 in default without any parameter searching. In practice, we find that when ${ \mathcal { L } } _ { \mathrm { d i v } }$ is optimized too small $( e . g .$ ., close to 0), the model tends to diverge and becomes unable to effectively model the underlying data distribution. This phenomenon potentially arises because overly emphasizing the separation of representations hinders the model’s ability to specialize and learn meaningful, shared representations across the data. We thus develop an adaptive weight w for the overall ${ \mathcal { L } } _ { \mathrm { d i v } } ;$ :

Table 1. Variation in model-scale on ImageNet 256×256 without CFG. Our proposed method brings consistent performance gains across all model-scales when applied to both SiT and REPA.

<table><tr><td>Model</td><td>Iter.</td><td> $FID_{\downarrow}$ </td><td> $sFID_{\downarrow}$ </td><td> $IS_{\uparrow}$ </td><td> $Prec.\uparrow$ </td><td> $Rec.\uparrow$ </td></tr><tr><td>SiT-B</td><td>400k</td><td>36.80</td><td>6.77</td><td>40.09</td><td>0.51</td><td>0.63</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>28.05</td><td>6.04</td><td>50.66</td><td>0.57</td><td>0.63</td></tr><tr><td>REPA-B</td><td>400k</td><td>22.99</td><td>6.70</td><td>64.73</td><td>0.59</td><td>0.65</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>17.29</td><td>6.56</td><td>79.92</td><td>0.62</td><td>0.65</td></tr><tr><td>SiT-L</td><td>400k</td><td>18.77</td><td>5.27</td><td>71.44</td><td>0.64</td><td>0.63</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>16.10</td><td>5.08</td><td>79.47</td><td>0.66</td><td>0.64</td></tr><tr><td>REPA-L</td><td>400k</td><td>9.57</td><td>5.34</td><td>113.32</td><td>0.69</td><td>0.66</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>8.47</td><td>5.42</td><td>123.03</td><td>0.69</td><td>0.67</td></tr><tr><td>SiT-XL</td><td>400k</td><td>17.43</td><td>5.11</td><td>76.00</td><td>0.64</td><td>0.64</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>12.42</td><td>4.85</td><td>95.01</td><td>0.68</td><td>0.63</td></tr><tr><td>REPA-XL</td><td>400k</td><td>8.73</td><td>5.21</td><td>118.68</td><td>0.69</td><td>0.65</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>8.09</td><td>5.02</td><td>123.23</td><td>0.70</td><td>0.65</td></tr></table>

$$
w = \left\{ \begin{array}{l l} 1, & \text { if } \mathcal {L} _ {\mathrm{div}} > 0. 5, \\ \frac {\mathcal {L} _ {\mathrm{div}} - 0 . 1}{0 . 5}, & \text { if } 0. 1 <   \mathcal {L} _ {\mathrm{div}} \leq 0. 5, \\ 0, & \text { otherwise }. \end{array} \right.
$$

## 4. Experiments

## 4.1. Experiment Setup

Implementation Details We incorporate DiverseDiT into several popular baselines to evaluation its versatility and effectiveness including SiT [44], REPA [70] and Mean-Flow [17], leaving other details untouched. To ensure fair comparison, we strictly follow the training configurations of SiT [44], REPA [70] and MeanFlow [17] for experimental evaluation on the ImageNet [13] dataset with the resolution of $2 5 6 \times 2 5 6$ and $5 1 2 \times 5 1 2$ , images and pre-computed VAE features are preprocessed following REPA [70] with Stable Diffusion VAE [51]. We adapt the B/2, L/2, and XL/2 model configurations from SiT with a patch size of 2 to evaluate the scalability. For training, we use AdamW [41] with a constant learning rate of 1e-4, $( \beta _ { 1 } , \beta _ { 1 } ) = ( 0 . 9 , 0 . 9 9 9 )$ without decay, the batchsize is fixed to 256. For performance evaluation, we strictly follow the ADM setup [14] and adopt several popular metrics: Frechet Inception Dis- ´ tance (FID) [23], structural FID (sFID) [46], Inception Score (IS) [52], Precision (Prec.) and Recall (Rec.) [33], all calculated from 50K generated images. Classifier-free guidance (CFG) [24] is not employed unless specified. We use Euler-Maruyama sampler with 250 steps for sampling. All experiments are conducted on 8 × 80GB H800 GPUs. More implementation details are provided in Sec. B.

Compared baselines For multi-step comparison, we compare our method against existing alternatives from three categories: 1) pixel-diffusion: ADM [14], VDM++ [29], CDM [26]; 2) latent-diffusion with UNet: LDM [51]; 3)

![](images/dc79d810e6ec8bb76149994118da6912bd47fbf2b03e2c96195d3de7e9a70273.jpg)

<details>
<summary>text_image</summary>

100K 200K 400K 100K 200K 400K
SIT-XL2
Our DiverseBT
SIT-XL2
Our DiverseBT
</details>

Figure 3. Generated samples from different training iterations. Images are sampled using the same seed, noise and class label. We use a classifier-free guidance scale of 4.0 during sampling.

Table 2. Comparison results on ImageNet 256×256 with CFG.

<table><tr><td>Method</td><td>Epochs</td><td> $\text{FID}_{\downarrow}$ </td><td> $\text{sFID}_{\downarrow}$ </td><td> $\text{IS}_{\uparrow}$ </td><td> $\text{Pre.}_{\uparrow}$ </td><td> $\text{Rec.}_{\uparrow}$ </td></tr><tr><td>ADM-U [14]</td><td>400</td><td>3.94</td><td>6.14</td><td>186.70</td><td>0.82</td><td>0.52</td></tr><tr><td>VDM++ [29]</td><td>560</td><td>2.40</td><td>-</td><td>225.30</td><td>-</td><td>-</td></tr><tr><td>CDM [26]</td><td>2160</td><td>4.88</td><td>-</td><td>211.80</td><td>-</td><td>-</td></tr><tr><td>LDM-4 [51]</td><td>200</td><td>3.60</td><td>-</td><td>247.70</td><td>0.87</td><td>0.48</td></tr><tr><td>SD-DiT [75]</td><td>480</td><td>3.23</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>MakDiT [73]</td><td>1600</td><td>2.28</td><td>5.67</td><td>276.70</td><td>0.80</td><td>0.62</td></tr><tr><td>MDTv2-XL/2 [16]</td><td>1080</td><td>1.58</td><td>4.52</td><td>314.70</td><td>0.79</td><td>0.65</td></tr><tr><td>DiT-XL/2 [49]</td><td>1400</td><td>2.27</td><td>4.60</td><td>278.20</td><td>0.83</td><td>0.57</td></tr><tr><td>SiT-XL/2 [44]</td><td>1400</td><td>2.06</td><td>4.50</td><td>270.30</td><td>0.82</td><td>0.59</td></tr><tr><td>REPA [70]</td><td>200</td><td>1.96</td><td>4.49</td><td>264.00</td><td>0.82</td><td>0.60</td></tr><tr><td>REPA [70]</td><td>800</td><td>1.80</td><td>4.50</td><td>284.00</td><td>0.81</td><td>0.61</td></tr><tr><td>REG [64]</td><td>800</td><td>1.36</td><td>4.25</td><td>299.40</td><td>0.77</td><td>0.66</td></tr><tr><td>E2E-REPA [35]</td><td>800</td><td>1.69</td><td>4.17</td><td>219.30</td><td>0.77</td><td>0.67</td></tr><tr><td>SRA [27]</td><td>800</td><td>1.58</td><td>4.65</td><td>311.40</td><td>0.80</td><td>0.63</td></tr><tr><td>DispLoss [62]</td><td>800</td><td>1.97</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DiverseDiT (Ours)</td><td>80</td><td>1.89</td><td>4.44</td><td>276.85</td><td>0.81</td><td>0.66</td></tr><tr><td>DiverseDiT (Ours)</td><td>200</td><td>1.52</td><td>4.23</td><td>282.72</td><td>0.81</td><td>0.66</td></tr></table>

Table 3. Comparison results on ImageNet 512×512 with CFG.

<table><tr><td>Method</td><td>Epochs</td><td> $\text{FID}_{\downarrow}$ </td><td> $\text{sFID}_{\downarrow}$ </td><td> $\text{IS}_{\uparrow}$ </td><td> $\text{Pre.}_{\uparrow}$ </td><td> $\text{Rec.}_{\uparrow}$ </td></tr><tr><td>ADM-G [14]</td><td>400</td><td>2.85</td><td>5.86</td><td>221.70</td><td>0.84</td><td>0.53</td></tr><tr><td>VDM++ [29]</td><td>-</td><td>2.65</td><td>-</td><td>278.10</td><td>-</td><td>-</td></tr><tr><td>MakDiT [73]</td><td>800</td><td>2.50</td><td>5.10</td><td>256.30</td><td>0.83</td><td>0.56</td></tr><tr><td>DiT-XL/2 [49]</td><td>600</td><td>3.04</td><td>5.02</td><td>240.80</td><td>0.84</td><td>0.54</td></tr><tr><td>SiT-XL/2 [44]</td><td>600</td><td>2.62</td><td>4.18</td><td>252.20</td><td>0.84</td><td>0.57</td></tr><tr><td>REPA [70]</td><td>200</td><td>2.08</td><td>4.19</td><td>274.60</td><td>0.83</td><td>0.58</td></tr><tr><td>DiverseDiT (Ours)</td><td>80</td><td>2.21</td><td>4.31</td><td>241.08</td><td>0.82</td><td>0.61</td></tr><tr><td>DiverseDiT (Ours)</td><td>200</td><td>1.99</td><td>4.19</td><td>267.12</td><td>0.83</td><td>0.61</td></tr></table>

diffusion transformers: SiT [44], DiT [49], SD-DiT [75], MaskDiT [73], MDTv2 [16]; and 4) current state-of-the-art models: REPA [70], REG [64], E2E-REPA [35], SRA [27], DispLoss [62]. For one-step comparison, we compare our method with recent popular methods: iCT-XL/2 [57], Shortcut-XL/2 [15], IMM-XL/2 [74] and Meanflow [17]. Detailed descriptions of these methods are given in Sec. D.

## 4.2. Main Results

In this part, we first present comparison results of applying our methods on different baselines across various scales to investigate its effectiveness and scalability, and then compare our methods against existing SoTA models.

Improving representation learning across various model scales. Tab. 1 presents the quantitative results of applying our proposed techniques to SiT and REPA across various model scales on ImageNet 256×256 without CFG. We could observe that incorporating our method consistently yields substantial improvements on all evaluation metrics across all model scales, demonstrating its effectiveness and generalization regardless of the underlying training paradigm. Notably, our method achieves an FID of 17.29 on the REPA-B setting with 400K iterations, which is better than that of SiT-L (i.e., 18.77) with the same training iterations. Similarly, the performance of applying our method on the REPA-L outperforms that of the REPA-XL, i.e., 8.47 vs 8.73 on FID and 123.03 vs 118.68 on IS. Moreover, Fig. 3 shows the images generated by SiT-XL and our proposed method at different training iterations. Our generated images exhibit more details, better structures, and fewer artifacts, demonstrating that our method leads to faster convergence and higher visual quality compared to the baseline models. That is, our design towards improving the diversity of representations across different blocks contributes to a scalable and efficient learning process.

Comparison with SoTA Models. Tab. 2 presents the comparison results with recent state-of-the-art (SoTA) methods using CFG On ImageNet 256 × 256. As can be observed from the table, our method achieves competitive performance compared to SoTA models while requiring significantly fewer training epochs. At 80 epochs, our method attains an FID score of 1.89, outperforming REPA trained for 200 epochs (1.96) and surpassing the performance of several established methods trained for hundreds or even thousands of epochs. For instance, the SiT-XL/2 model requires 1400 epochs to reach an FID of 2.06, while we achieve 1.52 with only 200 epochs. While REG achieves a slightly better FID of 1.36, it requires 800 epochs, four times the training cost of ours. Furthermore, Tab. 3 shows the comparison results on ImageNet 512 × 512. Similarly, our DiverseDiT achieves a comparable FID of 2.21 with only 80 epochs and obtains the best FID score when trained for 200 epochs. The consistent strong performance across multiple metrics, coupled with the significantly reduced training time, shows the efficiency and effectiveness of our model in learning diverse and high-quality representations. Additionally, we provide selected samples generated by our method in Fig. 4, the generated images demonstrate that DiverseDiT produces images with excellent quality. More quantitative and qualitative results CFG are given in Sec. F and Sec. I.

Improving representation learning for one-step generation. Regarding one-step generation, we applied our proposed techniques to MeanFlow (MF) [17] to assess the generalization ability. Tab. 4 presents the quantitative results across different model scales. Similar to our previous findings, incorporating our method consistently improves the performance with different model sizes, e.g., our method improves the FID score of MF-B/2 from 9.44 to 8.51 and the IS from 152.55 to 158.84. Additionally, Tab. 5 shows a comparison of our method with other one-step generative models. Notably, we achieve a new SoTA with an FID score of 2.99 by applying our method to MeanFlow-XL/2. These results identify the effectiveness of our method in improving representation learning for one-step generation.

![](images/65f8b4506840af0fadb5fb9d19473cb2233b758ee34445d59276641fadc07755.jpg)

<details>
<summary>natural_image</summary>

Collage of diverse images including crab, birds, animals, and food items with no visible text or symbols
</details>

Figure 4. Generated samples on ImageNet 256×256 from our DiverseDiT. We use a classifier-free guidance scale of 4.0.

Table 4. Variation in model scale on ImageNet 256×256 for one-step generation without CFG.

<table><tr><td>Model</td><td>Iter.</td><td>Step.</td><td> $FID_{\downarrow}$ </td><td> $sFID_{\downarrow}$ </td><td> $IS_{\uparrow}$ </td><td> $Prec.\uparrow$ </td><td> $Rec.\uparrow$ </td></tr><tr><td>MF-B/2</td><td>400K</td><td>1</td><td>9.44</td><td>6.21</td><td>152.55</td><td>0.77</td><td>0.42</td></tr><tr><td>+ (Ours)</td><td>400K</td><td>1</td><td>8.51</td><td>5.95</td><td>158.84</td><td>0.78</td><td>0.44</td></tr><tr><td>MF-L/2</td><td>400k</td><td>1</td><td>8.73</td><td>6.11</td><td>161.69</td><td>0.79</td><td>0.40</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>1</td><td>7.15</td><td>5.91</td><td>199.66</td><td>0.81</td><td>0.41</td></tr><tr><td>MF-XL/2</td><td>400k</td><td>1</td><td>5.94</td><td>6.10</td><td>213.13</td><td>0.83</td><td>0.41</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>1</td><td>5.69</td><td>5.88</td><td>214.92</td><td>0.83</td><td>0.41</td></tr></table>

Table 5. Comparison results on ImageNet 256×256 for onestep generation with CFG.

<table><tr><td>Method</td><td>Params</td><td>Step</td><td>NFE</td><td> $\mathbf{FID}_{\downarrow}$ </td></tr><tr><td>iCT-XL/2 [57]</td><td>675M</td><td>1</td><td>1</td><td>34.24</td></tr><tr><td>Shortcut-XL/2 [15]</td><td>675M</td><td>1</td><td>1</td><td>10.60</td></tr><tr><td>IMM-XL/2 [74]</td><td>675M</td><td>1</td><td>2</td><td>7.77</td></tr><tr><td>MF-XL/2 [17]</td><td>676M</td><td>1</td><td>1</td><td>3.43</td></tr><tr><td>MF-XL/2 + DispLoss [62]</td><td>676M</td><td>1</td><td>1</td><td>3.21</td></tr><tr><td>MF-XL/2 +ours</td><td>713M</td><td>1</td><td>1</td><td>2.99</td></tr></table>

## 4.3. Ablation Analysis

In this part, we perform ablation studies to testify the efficacy of each component and the design choices used in our main experiments.

Ablation on designed components. In Tab. 6, we present an ablation study that analyzes the contribution of different components of our method by removing each component. The results clearly demonstrate the importance of both the representation diversity loss and the long residual connections for optimal performance. Removing the diversity loss (w/o diversity) worsens FID scores for both SiT-B (from 28.05 to 32.77) and REPA-B (from 17.29 to

Table 6. Ablation analysis on different components.

<table><tr><td>Component</td><td> $\text{FID}_\downarrow$ </td><td> $\text{sFID}_\downarrow$ </td><td> $\text{IS}_\uparrow$ </td><td> $Prec._\uparrow$ </td><td> $Rec._\uparrow$ </td></tr><tr><td>SiT-B + full</td><td>28.05</td><td>6.04</td><td>50.66</td><td>0.57</td><td>0.63</td></tr><tr><td>w/o diversity</td><td>32.77</td><td>6.42</td><td>44.85</td><td>0.54</td><td>0.63</td></tr><tr><td>w/o residual</td><td>33.72</td><td>6.53</td><td>43.97</td><td>0.53</td><td>0.63</td></tr><tr><td>REPA-B + full</td><td>17.29</td><td>6.56</td><td>79.92</td><td>0.62</td><td>0.65</td></tr><tr><td>w/o diversity</td><td>20.66</td><td>6.66</td><td>72.72</td><td>0.61</td><td>0.64</td></tr><tr><td>w/o residual</td><td>18.18</td><td>6.62</td><td>75.49</td><td>0.61</td><td>0.65</td></tr></table>

Table 7. Ablation analysis on different loss variants.

<table><tr><td>Component</td><td> $\text{FID}_{\downarrow}$ </td><td> $\text{sFID}_{\downarrow}$ </td><td> $\text{IS}_{\uparrow}$ </td><td> $\text{Prec.}_{\uparrow}$ </td><td> $\text{Rec.}_{\uparrow}$ </td></tr><tr><td>REPA-B + full</td><td>17.29</td><td>6.56</td><td>79.92</td><td>0.62</td><td>0.65</td></tr><tr><td>only  $\mathcal{L}_{\text{orth}}$ </td><td>18.97</td><td>6.64</td><td>75.44</td><td>0.61</td><td>0.65</td></tr><tr><td>only  $\mathcal{L}_{\text{MI}}$ </td><td>17.70</td><td>6.62</td><td>78.34</td><td>0.62</td><td>0.65</td></tr><tr><td>only  $\mathcal{L}_{\text{div}}$ </td><td>20.85</td><td>6.78</td><td>68.74</td><td>0.61</td><td>0.65</td></tr></table>

20.66). Similarly, removing the long residual connections (w/o residual) also noticeably increases FID for both baseline models. These results confirm that both components of DiverseDiT play a crucial role in promoting diverse representation learning and improving the model performance. More ablation analysis results are presented in Sec. G.

Effect of diversity loss variants. In the main experiments, we adopt a combination of different discrepancies for the representation diversity loss. This part validates the contribution of different components of the diversity loss in Tab. 7. The results show that using all components of the diversity loss (REPA-B + full) achieves the best performance (FID 17.29, IS 79.92). Moreover, removing any single loss component degrades performance. More importantly, adding any component consistently outperforms the REPA-B baseline (Tab. 1), demonstrating their effectiveness in encouraging diverse representation learning.

Effect of the adaptive range of diversity loss. In Tab. 8, we test the impact of different adaptive ranges of our diversity loss on the performance. A constant weight leads to divergence, possibly due to excessive representation discrepancy as discussed in Sec. 3. By contrast, adaptively controlling the diversity loss based on the loss value enables stable training. In particular, [0.1, 0.5] yields the best performance, suggesting that a narrower adaptive range effectively promotes diversity without sacrificing image quality. Combining with existing methods for further improvement. Tab. 9 shows DiverseDiT’s compatibility with DispLoss [62] and SRA [27], yielding further performance gains. The results demonstrate that our method can be effectively combined with existing approaches for further performance improvements, reflecting the flexibility of our approach. Noticeably, when combining our proposed method with both DispLoss and SRA, we achieve an FID of 21.95, which is better than that of REPA (22.99 in Tab. 1) at the same iterations. Recall that REPA requires external models for representation alignment, while here we do not rely on any external guidance, demonstrating the potential for representation learning through internal mechanisms.

Table 8. Ablation study on adaptive range of diversity loss.

<table><tr><td>Range</td><td> $\mathbf{FID}_{\downarrow}$ </td><td> $sFID_{\downarrow}$ </td><td> $IS_{\uparrow}$ </td><td> $Prec._{\uparrow}$ </td><td> $Rec._{\uparrow}$ </td></tr><tr><td>Adap [0.1, 0.5]</td><td>28.05</td><td>6.04</td><td>50.66</td><td>0.57</td><td>0.63</td></tr><tr><td>Adap [0.2, 0.7]</td><td>30.59</td><td>6.34</td><td>48.07</td><td>0.55</td><td>0.63</td></tr><tr><td>Adap [0.3, 0.9]</td><td>31.85</td><td>6.43</td><td>45.98</td><td>0.54</td><td>0.63</td></tr><tr><td>Constant</td><td>diverge</td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

Table 9. Combining our method with prior approaches. All results are calculated from 400K iterations without CFG.

<table><tr><td>Component</td><td> $\text{FID}_{\downarrow}$ </td><td> $\text{sFID}_{\downarrow}$ </td><td> $\text{IS}_{\uparrow}$ </td><td> $\text{Prec.}_{\uparrow}$ </td><td> $\text{Rec.}_{\uparrow}$ </td></tr><tr><td>SiT-B</td><td>36.80</td><td>6.77</td><td>40.09</td><td>0.51</td><td>0.63</td></tr><tr><td>+ Ours</td><td>28.05</td><td>6.04</td><td>50.66</td><td>0.57</td><td>0.63</td></tr><tr><td>++ DispLoss [62]</td><td>24.98</td><td>6.01</td><td>57.04</td><td>0.59</td><td>0.63</td></tr><tr><td>+++ SRA [27]</td><td>21.95</td><td>5.92</td><td>64.64</td><td>0.60</td><td>0.64</td></tr></table>

## 5. Related Works

Diffusion Models. Diffusion probabilistic models [25, 55, 56], which generate images via iteratively denoising Gaussian noises, have become the dominating paradigm for image [1, 6, 51] and video generation [20, 31, 69], driven by improved training stability with flow matching [39, 40] and exceptional model scalability from conventional UNetbased models [3, 14, 51] to the Transformer-based architectures [2, 44, 49, 61]. Besides these advancements, many efforts improve diffusion models from the perspective of accelerating sampling process [17, 43, 58], designing noise schedules [28, 42, 47], developing novel architectures like linear transformers [7, 66], MoE-based models [53, 63], etc.

Representation Learning. Central to representation learning is to learn rich and meaningful representations for downstream tasks. This field has evolved through several key paradigms, including discriminative, generative, and multimodal approaches. Discriminative methods, exemplified by contrastive learning based methods including BYOL [19], DINO [4, 48, 54], and MoCo [8, 9, 21], capture discriminative signals between images to learn strong representations. The generative variant learns the underlying data distribution via reconstructing input images, representative works including auto-encoder methods VAE [30], MAE [22], and masked image modeling [67]. Similarly, diffusion models also learn informative features as inherent denoising autoencoders [45, 68, 72]. To enable cross-model understanding and retrieval, multimodal methods [37, 50, 60, 71] align textual and visual signals in a shared representation space. Despite these advancements, it remains unclear what representations should be learned for different tasks, especially for diffusion generative models.

Representation Learning of Diffusion Models. Many prior approaches identify that improved representation learning of diffusion models advances both synthesis quality and downstream tasks [36, 65]. Following this philosophy, REPA [70] incorporated external encoders to align diffusion representations with pre-trained representations, extended by REPA-E [35] that enabled end-to-end training with VAE and SARA [5] that introduced structural alignment. Further, SoftREPA [34] performed alignment on textual embeddings and REG [64] entangled image latents and class tokens to harness discriminative representations. Unlike these approaches that requires external guidances for knowledge alignment, SRA [27] leveraged representations from later layers with lower noise to guide representations of earlier layers with higher noise. Further, Wang et al. [62] developed DispLoss to regularize internal features to disperse in the embedding space, thus encouraging the model to capture informative representations. Despite these advancements, it remains unclear how meaningful representations are learned and what representations within models are more suitable. We thus perform a systematic investigation on this and develop DiverseDiT to learn diverse and effective representations for diffusion transformers.

## 6. Conclusion

In this work, we first present a comprehensive analysis of the representation learning process within diffusion transformers (DiTs), revealing the critical role of representation diversity across different blocks. Based on these insights, we introduce DiverseDiT, a novel and efficient framework explicitly designed to enhance representation diversity with long residual connections to diversify input and representation diversity loss to encourage distinct features across blocks, without relying on external guidance. Extensive experiments demonstrate that DiverseDiT consistently improves training convergence and model performance across various model scales and settings, including multi-step and one-step generation. These findings contribute to a deeper understanding of representation learning dynamics in DiTs and offer a practical, effective strategy for boosting their performance, paving the way for future research in learning representations for generative models.

## Acknowledgments

This work was supported by AI for Science Program, Shanghai Municipal Commission of Economy and Informatization (Grant No. 2025-GZL-RGZN-BTBX-02017).

## References

[1] Jason Baldridge, Jakob Bauer, Mukul Bhutani, Nicole Brichtova, Andrew Bunner, et al. Imagen 3. arXiv preprint arXiv:2408.07009, 2024. 8  
[2] Fan Bao, Shen Nie, Kaiwen Xue, Yue Cao, Chongxuan Li, Hang Su, and Jun Zhu. All are worth words: A vit backbone for diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 22669–22679, 2023. 1, 8  
[3] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al. Stable video diffusion: Scaling latent video diffusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023. 8  
[4] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou, ´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9650–9660, 2021. 8  
[5] Hesen Chen, Junyan Wang, Zhiyu Tan, and Hao Li. SARA: Structural and adversarial representation alignment for training-efficient diffusion models. arXiv preprint arXiv:2503.08253, 2025. 8  
[6] Junsong Chen, Jincheng Yu, Chongjian Ge, Lewei Yao, Enze Xie, Yue Wu, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, et al. Pixart-α: Fast training of diffusion transformer for photorealistic text-to-image synthesis. In International Conference on Learning Representations, 2024. 8  
[7] Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, et al. Sana-video: Efficient video generation with block linear diffusion transformer. arXiv preprint arXiv:2509.24695, 2025. 8  
[8] Xinlei Chen, Haoqi Fan, Ross Girshick, and Kaiming He. Improved baselines with momentum contrastive learning. arXiv preprint arXiv:2003.04297, 2020. 8, 2  
[9] Xinlei Chen, Saining Xie, and Kaiming He. An empirical study of training self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 9640–9649, 2021. 8, 2  
[10] Xinlei Chen, Zhuang Liu, Saining Xie, and Kaiming He. Deconstructing denoising diffusion models for self-supervised learning. arXiv preprint arXiv:2401.14404, 2024. 1  
[11] Nello Cristianini, John Shawe-Taylor, Andre Elisseeff, and Jaz Kandola. On kernel-target alignment. In Advances in Neural Information Processing Systems, 2001. 3  
[12] MohammadReza Davari, Stefan Horoi, Amine Natik, Guillaume Lajoie, Guy Wolf, and Eugene Belilovsky. Reliability of cka as a similarity measure in deep learning. arXiv preprint arXiv:2210.16156, 2022. 3  
[13] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009. 5  
[14] Prafulla Dhariwal and Alexander Nichol. Diffusion models beat gans on image synthesis. In Advances in Neural Information Processing Systems, pages 8780–8794, 2021. 5, 6, 8, 2, 3  
[15] Kevin Frans, Danijar Hafner, Sergey Levine, and Pieter Abbeel. One step diffusion via shortcut models. In International Conference on Learning Representations, 2025. 6, 7, 4  
[16] Shanghua Gao, Pan Zhou, Ming-Ming Cheng, and Shuicheng Yan. Mdtv2: Masked diffusion transformer is a strong image synthesizer. arXiv preprint arXiv:2303.14389, 2023. 6, 3  
[17] Zhengyang Geng, Mingyang Deng, Xingjian Bai, J Zico Kolter, and Kaiming He. Mean flows for one-step generative modeling. arXiv preprint arXiv:2505.13447, 2025. 2, 5, 6, 7, 8, 1, 4  
[18] Arthur Gretton, Olivier Bousquet, Alex Smola, and Bernhard Scholkopf. Measuring statistical dependence with hilbert- ¨ schmidt norms. In International Conference on Algorithmic Learning Theory, pages 63–77, 2005. 3  
[19] Jean-Bastien Grill, Florian Strub, Florent Altche, Corentin ´ Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. In Advances in Neural Information Processing Systems, pages 21271–21284, 2020. 8  
[20] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and Jose Lezama. ´ Photorealistic video generation with diffusion models. In European Conference on Computer Vision, pages 393–411. Springer, 2024. 1, 8  
[21] Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9729–9738, 2020. 8, 2  
[22] Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollar, and Ross Girshick. Masked autoencoders are scalable ´ vision learners. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 16000– 16009, 2022. 8, 2, 4  
[23] Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in Neural Information Processing Systems, 30, 2017. 5, 3  
[24] Jonathan Ho and Tim Salimans. Classifier-free diffusion guidance. arXiv preprint arXiv:2207.12598, 2022. 5, 1  
[25] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in Neural Information Processing Systems, 33:6840–6851, 2020. 1, 2, 8  
[26] Jonathan Ho, Chitwan Saharia, William Chan, David J Fleet, Mohammad Norouzi, and Tim Salimans. Cascaded diffusion models for high fidelity image generation. Journal of Machine Learning Research, 23(47):1–33, 2022. 5, 6, 3  
[27] Dengyang Jiang, Mengmeng Wang, Liuzhuozheng Li, Lei Zhang, Haoyu Wang, Wei Wei, Guang Dai, Yanning Zhang, and Jingdong Wang. No other representation component is needed: Diffusion transformers can provide representation guidance by themselves. arXiv preprint arXiv:2505.02831, 2025. 2, 6, 8, 1, 4, 9  
[28] Tero Karras, Miika Aittala, Timo Aila, and Samuli Laine. Elucidating the design space of diffusion-based generative models. pages 26565–26577, 2022. 8  
[29] Diederik Kingma and Ruiqi Gao. Understanding diffusion objectives as the elbo with simple data augmentation. pages 65484–65516, 2023. 5, 6, 3  
[30] Diederik P Kingma and Max Welling. Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114, 2013. 8  
[31] Weijie Kong, Qi Tian, Zijian Zhang, Rox Min, Zuozhuo Dai, Jin Zhou, Jiangfeng Xiong, Xin Li, Bo Wu, Jianwei Zhang, et al. Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603, 2024. 8  
[32] Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International Conference on Machine Learning, pages 3519–3529. PMLR, 2019. 3  
[33] Tuomas Kynka¨anniemi, Tero Karras, Samuli Laine, Jaakko ¨ Lehtinen, and Timo Aila. Improved precision and recall metric for assessing generative models. In Advances in Neural Information Processing Systems, 2019. 5, 3  
[34] Jaa-Yeon Lee, Byunghee Cha, Jeongsol Kim, and Jong Chul Ye. Aligning text to image in diffusion models is easier than you think. arXiv preprint arXiv:2503.08250, 2025. 8  
[35] Xingjian Leng, Jaskirat Singh, Yunzhong Hou, Zhenchang Xing, Saining Xie, and Liang Zheng. REPA-E: Unlocking vae for end-to-end tuning with latent diffusion transformers. arXiv preprint arXiv:2504.10483, 2025. 1, 6, 8, 4  
[36] Daiqing Li, Huan Ling, Amlan Kar, David Acuna, Seung Wook Kim, Karsten Kreis, Antonio Torralba, and Sanja Fidler. Dreamteacher: Pretraining image backbones with deep generative models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 16698– 16708, 2023. 8  
[37] Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pages 12888– 12900. PMLR, 2022. 8  
[38] Tianhong Li, Dina Katabi, and Kaiming He. Return of unconditional generation: A self-supervised representation generation method. In Advances in Neural Information Processing Systems, pages 125441–125468, 2024. 2  
[39] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In International Conference on Learning Representations, 2023. 2, 8  
[40] Xingchao Liu, Chengyue Gong, et al. Flow straight and fast: Learning to generate and transfer data with rectified flow. In International Conference on Learning Representations, 2023. 8  
[41] Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101, 2017. 5, 1  
[42] Cheng Lu and Yang Song. Simplifying, stabilizing and scaling continuous-time consistency models. arXiv preprint arXiv:2410.11081, 2024. 8  
[43] Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. Dpm-solver: A fast ode solver for diffusion probabilistic model sampling in around 10 steps. In Advances in Neural Information Processing Systems, pages 5775–5787, 2022. 8  
[44] Nanye Ma, Mark Goldstein, Michael S Albergo, Nicholas M Boffi, Eric Vanden-Eijnden, and Saining Xie. SiT: Exploring flow and diffusion-based generative models with scalable interpolant transformers. In European Conference on Computer Vision, pages 23–40. Springer, 2024. 5, 6, 8, 1, 4  
[45] Sarthak Mittal, Korbinian Abstreiter, Stefan Bauer, Bernhard Scholkopf, and Arash Mehrjou. Diffusion based represen- ¨ tation learning. In International Conference on Machine Learning, pages 24963–24982. PMLR, 2023. 1, 8  
[46] Charlie Nash, Jacob Menick, Sander Dieleman, and Peter Battaglia. Generating images with sparse representations. In International Conference on Machine Learning, pages 7958–7968. PMLR, 2021. 5, 3  
[47] Alexander Quinn Nichol and Prafulla Dhariwal. Improved denoising diffusion probabilistic models. In International Conference on Machine Learning, pages 8162–8171. PMLR, 2021. 8, 5  
[48] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, Huy ´ Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023. 8, 2  
[49] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 4195–4205, 2023. 1, 6, 8, 4  
[50] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pages 8748–8763, 2021. 8  
[51] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High-resolution image ¨ synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022. 1, 5, 6, 8, 3  
[52] Tim Salimans, Ian Goodfellow, Wojciech Zaremba, Vicki Cheung, Alec Radford, and Xi Chen. Improved techniques for training gans. In Advances in Neural Information Processing Systems, 2016. 5, 3  
[53] Minglei Shi, Ziyang Yuan, Haotian Yang, Xintao Wang, Mingwu Zheng, Xin Tao, Wenliang Zhao, Wenzhao Zheng,  
Jie Zhou, Jiwen Lu, et al. Diffmoe: Dynamic token selection for scalable diffusion transformers. arXiv preprint arXiv:2503.14487, 2025. 8  
[54] Oriane Simeoni, Huy V Vo, Maximilian Seitzer, Federico ´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, ¨ et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025. 8  
[55] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. Deep unsupervised learning using nonequilibrium thermodynamics. In International Conference on Machine Learning, pages 2256–2265. PMLR, 2015. 1, 8  
[56] Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising diffusion implicit models. In International Conference on Learning Representations, 2021. 8  
[57] Yang Song and Prafulla Dhariwal. Improved techniques for training consistency models. In International Conference on Learning Representations, 2024. 6, 7, 4  
[58] Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency models. In International Conference on Machine Learning, pages 32211–32252. PMLR, 2023. 8  
[59] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jon Shlens, and Zbigniew Wojna. Rethinking the inception architecture for computer vision. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 2818–2826, 2016. 3  
[60] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025. 8  
[61] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, 2017. 8  
[62] Runqian Wang and Kaiming He. Diffuse and disperse: Image generation with representation regularization. arXiv preprint arXiv:2506.09027, 2025. 1, 2, 6, 7, 8, 9  
[63] Yujie Wei, Shiwei Zhang, Hangjie Yuan, Yujin Han, Zhekai Chen, Jiayu Wang, Difan Zou, Xihui Liu, Yingya Zhang, Yu Liu, et al. Routing matters in moe: Scaling diffusion transformers with explicit routing guidance. arXiv preprint arXiv:2510.24711, 2025. 8  
[64] Ge Wu, Shen Zhang, Ruijing Shi, Shanghua Gao, Zhenyuan Chen, Lei Wang, Zhaowei Chen, Hongcheng Gao, Yao Tang, Jian Yang, et al. Representation entanglement for generation: Training diffusion transformers is much easier than you think. arXiv preprint arXiv:2507.01467, 2025. 2, 6, 8, 4  
[65] Weilai Xiang, Hongyu Yang, Di Huang, and Yunhong Wang. Denoising diffusion autoencoders are unified self-supervised learners. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15802–15812, 2023. 1, 8  
[66] Enze Xie, Junsong Chen, Junyu Chen, Han Cai, Haotian Tang, Yujun Lin, Zhekai Zhang, Muyang Li, Ligeng Zhu,  
Yao Lu, et al. Sana: Efficient high-resolution text-to-image synthesis with linear diffusion transformers. In International Conference on Learning Representations, 2025. 1, 8  
[67] Zhenda Xie, Zheng Zhang, Yue Cao, Yutong Lin, Jianmin Bao, Zhuliang Yao, Qi Dai, and Han Hu. Simmim: A simple framework for masked image modeling. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 9653–9663, 2022. 8  
[68] Xingyi Yang and Xinchao Wang. Diffusion model as representation learner. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 18938–18949, 2023. 8  
[69] Zhuoyi Yang, Jiayan Teng, Wendi Zheng, Ming Ding, Shiyu Huang, Jiazheng Xu, Yuanming Yang, Wenyi Hong, Xiaohan Zhang, Guanyu Feng, et al. Cogvideox: Text-to-video diffusion models with an expert transformer. In International Conference on Learning Representations, 2025. 1, 8  
[70] Sihyun Yu, Sangkyung Kwak, Huiwon Jang, Jongheon Jeong, Jonathan Huang, Jinwoo Shin, and Saining Xie. Representation alignment for generation: Training diffusion transformers is easier than you think. In International Conference on Learning Representations, 2024. 1, 4, 5, 6, 8  
[71] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 11975–11986, 2023. 8  
[72] Zijian Zhang, Zhou Zhao, and Zhijie Lin. Unsupervised representation learning from pre-trained diffusion probabilistic models. In Advances in Neural Information Processing Systems, pages 22117–22130, 2022. 8  
[73] Hongkai Zheng, Weili Nie, Arash Vahdat, and Anima Anandkumar. Fast training of diffusion models with masked transformers. arXiv preprint arXiv:2306.09305, 2023. 6, 4  
[74] Linqi Zhou, Stefano Ermon, and Jiaming Song. Inductive moment matching. arXiv preprint arXiv:2503.07565, 2025. 6, 7, 4  
[75] Rui Zhu, Yingwei Pan, Yehao Li, Ting Yao, Zhenglong Sun, Tao Mei, and Chang Wen Chen. Sd-dit: Unleashing the power of self-supervised discrimination in diffusion transformer. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8435– 8445, 2024. 6, 4

# DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers

Supplementary Material

![](images/8d123f3b61019444efcf13526b8b30dd01e4540dd376fb8167effc2e847e6bd2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["SiT/DiT block"] --> B["SiT/DiT block"]
  B --> C["SiT/DiT block"]
  C --> D["SiT/DiT block"]
  D --> E["SiT/DiT block"]
  E --> F["SiT/DiT block"]
  F --> G["nonised images"]
    
  H["f1-i"] --> I["+"]
  J["f1"] --> I
  I --> K["Norm"]
  I --> L["Linear"]
  K --> M["f1"]
  L --> M
  M --> N["long residual connections"]
    
  O["Orthogonality Loss Independence"] --> P["Feature Dispersion Loss Variance"]
  P --> Q["Mutual Information Loss Correlation"]
  Q --> R["Diversity Loss"]
    
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    
    subgraph Feature Dispersion Loss Variance
        P
        Q
        R
    end
    
    style Feature Dispersion Loss Variance fill:#e6f7ff,stroke:#333
    style Feature Dispersion Loss Variance fill:#e6f7ff,stroke:#333
```
</details>

Figure A1. Detailed diagram of our proposed DiverseDiT. DiverseDiT incorporates long residual connections to diversify input representations across blocks and a representation diversity loss to encourage blocks to learn distinct features.

## A. Appendix Overview

This supplementary material is organized as follows: First, we present more implementation details in Sec. B. Followed by the evaluation details and brief introduction of comparison baseline methods in Sec. C and Sec. D, respectively. Then, Sec. E shows the detailed quantitative results of our comprehensive analysis in Sec. 2. In the following, we present more quantitative comparison results under various settings in Sec. F and more ablation results in Sec. G. Moreover, Sec. H discusses the limitations and potential future works of our method. Finally, Sec. I illustrates more uncurated images generated by our proposed method.

## B. More Implementation Details

Detailed diagram of our DiverseDiT. Our diversity loss is theoretically motivated from the following perspectives: 1) introducing an explicit inductive bias that encourages block-wise diversity to model the underlying observed distribution; 2) improving the representational orthogonality across blocks, thus reducing the redundancy and mutual correlation of different blocks; 3) promoting the coverage of the representation space, enabling blocks to specialize in complementary structures. For our proposed long residual connections and representation diversity loss, we first concatenate the hidden features of two blocks and perform layer normalization and a lightweight linear layer, as illustrated in Fig. A1.

More implementation details. We implement our proposed techniques on the original SiT [44] and REPA [70] implementation, leaving other details unchanged. Regarding the representation diversity loss, we calculate the corresponding loss items following the definition of each loss in Sec. 3, we randomly select 10 layers for computing the loss, as we find that computing on more layers gains similar performance improvement. Such observation aligns with that of DispLoss [62], where the effect of representation diversity loss propagates to other blocks, even though it is not directly applied to them. Further, the detailed setup for hyperparameters is presented in Tab. A1. To speed up the training process and save pre-processing time, we adopt mixed-precision (fp16) with gradient clipping and pre-compute VAE latents with stable diffusion VAE [51] (sd-vae-ft-mse) following REPA. For all optimization, we adopt AdamW [41] with a learning rate of 1e-4, and the training batchsize is 256. When classifier-free guidance (CFG) [24] is applied for generating images, we use the same guidance interval as that of REPA, which has been identified to improve the model performance, also illustrated in our results in Tab. A4. Additionally, we implement our method on the official MeanFlow [17] and follow their default configurations for training and evaluation for one-step generation experimental evaluation.

Sampler. Following the practice of REPA, we employ the Euler-Maruyama sampler with the SDE sampling with a diffusion coefficient $\sigma _ { t } .$ The sampling step for generating each image is set to 250.

Discussion on hyperparameter sensitivity. In our evaluation, we use the same hyperparameter settings when applying our method to various models, namely SiT [44], REPA [70], DispLoss [62], SRA [27] and MeanFlow [17]. Our method consistently improves performance across different backbones and different sampler settings (multi-step and one-step sampling). These results indicate that although our diversity loss introduces several hyperparameters, it is robust to hyperparameter choices.

Computing resources. All experiments were conducted on NVIDIA H100 (80GB) or H200 (141GB) GPUs. For training, the speed is about 5.8 steps/s for training SiT-XL +

Table A1. Hyperparameter setup for the main experiment. We strictly follow the setup of the baseline SiT and REPA for our training and evaluation for fair comparison.

<table><tr><td></td><td>Table 1 (SiT-B)</td><td>Table 1 (SiT-L)</td><td>Table 1 (SiT-XL)</td><td>Table 2 (SiT-XL)</td><td>Table 3 (SiT-XL)</td></tr><tr><td colspan="6">SiT + Ours</td></tr><tr><td>Input dim.</td><td> $32 \times 32 \times 4$ </td><td> $32 \times 32 \times 4$ </td><td> $32 \times 32 \times 4$ </td><td> $32 \times 32 \times 4$ </td><td> $64 \times 64 \times 4$ </td></tr><tr><td>Num. layers</td><td>12</td><td>24</td><td>28</td><td>28</td><td>28</td></tr><tr><td>Hidden dim.</td><td>768</td><td>1,024</td><td>1,152</td><td>1,152</td><td>1,152</td></tr><tr><td>Num. heads</td><td>12</td><td>16</td><td>16</td><td>16</td><td>16</td></tr><tr><td colspan="6">REPA + Ours</td></tr><tr><td> $\lambda$ </td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td><td>0.5</td></tr><tr><td>Alignment depth</td><td>5</td><td>8</td><td>8</td><td>8</td><td>8</td></tr><tr><td>sim( $\cdot$ ,  $\cdot$ )</td><td>cos. sim.</td><td>cos. sim.</td><td>cos. sim</td><td>cos. sim.</td><td>cos. sim.</td></tr><tr><td>Encoder  $f(x)$ </td><td>DINOv2-B</td><td>DINOv2-B</td><td>DINOv2-B</td><td>DINOv2-B</td><td>DINOv2-B</td></tr><tr><td colspan="6">Optimization</td></tr><tr><td>Loss adaptive range</td><td>[0.1, 0.5]</td><td>[0.1, 0.5]</td><td>[0.1, 0.5]</td><td>[0.1, 0.5]</td><td>[0.1, 0.5]</td></tr><tr><td>Training iteration</td><td>400K</td><td>400K</td><td>400K</td><td>4M</td><td>1M</td></tr><tr><td>Training Batch size</td><td>256</td><td>256</td><td>256</td><td>256</td><td>256</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>lr</td><td>0.0001</td><td>0.0001</td><td>0.0001</td><td>0.0001</td><td>0.0001</td></tr><tr><td> $(\beta_1, \beta_2)$ </td><td>(0.9, 0.999)</td><td>(0.9, 0.999)</td><td>(0.9, 0.999)</td><td>(0.9, 0.999)</td><td>(0.9, 0.999)</td></tr><tr><td colspan="6">Interpolants</td></tr><tr><td> $\alpha_t$ </td><td> $1 - t$ </td><td> $1 - t$ </td><td> $1 - t$ </td><td> $1 - t$ </td><td> $1 - t$ </td></tr><tr><td> $\sigma_t$ </td><td> $t$ </td><td> $t$ </td><td> $t$ </td><td> $t$ </td><td> $t$ </td></tr><tr><td> $w_t$ </td><td> $\sigma_t$ </td><td> $\sigma_t$ </td><td> $\sigma_t$ </td><td> $\sigma_t$ </td><td> $\sigma_t$ </td></tr><tr><td>Training objective</td><td>v-prediction</td><td>v-prediction</td><td>v-prediction</td><td>v-prediction</td><td>v-prediction</td></tr><tr><td>Sampler</td><td>Euler-Maruyama</td><td>Euler-Maruyama</td><td>Euler-Maruyama</td><td>Euler-Maruyama</td><td>Euler-Maruyama</td></tr><tr><td>Sampling steps</td><td>250</td><td>250</td><td>250</td><td>250</td><td>250</td></tr><tr><td>Guidance</td><td>1.0</td><td>1.0</td><td>1.0</td><td>1.35</td><td>1.35</td></tr></table>

Ours, and it takes about 1.38 hours to generate 50,000 images for evaluation (10.04 images/s). We have uploaded the compute report for detailed GPU hours used for our analysis and evaluation.

Pretrained vision foundation models for representation alignment. In our systematic analysis and experimental evaluation, we use three pretrained visual encoders as external representation guidance, namely DINOv2-B [48], MAE [22], and MoCov3 [9]. For DINOv2-B1 and MAE2 models, we download the pretrained model officially released by the original authors. Regarding the MoCov3 model, we download the -L version from the implementation of RCG3 [38]; following REPA. For representation alignment, we perform projection with three MLP layers with SiLU activations following the exact configuration of REPA.

• DINOv2 [48]: DINOv2 employs a vision transformer (ViT) architecture and learns self-supervised representations by enforcing consistency between different views of an image. It measures the feature distance between the representations of real and generated images, capturing high-level semantic information.

• MAE [22]: MAE trains an encoder and a lightweight decoder with a reconstruction objective. It learns to reconstruct masked patches of an image, learning robust representations.

• MoCov3 [9]: based on the philosophy of contrastive learning, MoCov3 empirically revisits prior MoCo series [8, 21] and scales to larger model sizes to learn representations by maximizing the similarity between different views of the same image while minimizing the similarity between views of different images.

## C. Evaluation Details

Implementation details for evaluation. We strictly follow the setup and use the same reference batches of ADM [14] for evaluation, following their official implementation. Specifically, for 256×256 evaluation, we generate 50,000 images and convert them into a .npz file and compute the quantitative metrics from the reference batch (VIR-TUAL imagenet256 labeled.npz) of ADM4. Similarly, we quantify the result of 512×512 evaluation via computing the metrics between our generated images and the reference batch (VIRTUAL imagenet512.npz).

Evaluation metrics. We adopt several popular metrics for evaluation: Frechet Inception Distance (FID) [ ´ 23], structural FID (sFID) [46], Inception Score (IS) [52], Precision (Prec.) and Recall (Rec.) [33]. Their main concepts are:

• FID [23] computes the Frechet Distance between two ´ observed data distributions, which represent the feature distributions of synthesized and real images extracted by the pre-trained Inception-V3 [59]. Formally, FID is calculated by

$$
\operatorname{FID} (X, Y) = \left\| \mu_ {s} - \mu_ {r} \right\| ^ {2} + \operatorname{Tr} \left(\Sigma_ {s} + \Sigma_ {r} - 2 \left(\Sigma_ {s} \Sigma_ {r}\right) ^ {\frac {1}{2}}\right), \tag {A1}
$$

where X and Y represent the synthesized distribution and real distribution, respectively. µ and Σ correspond to the mean and variance of the distribution, and Tr(·) is the trace operation.

• sFID [46] is a variant of FID that aims to be more robust to structural differences between real and generated images. Instead of using the standard Inception-V3 features, sFID uses features extracted from different layers of the network, focusing on structural information. This makes it more sensitive to the arrangement of objects and their parts, and less sensitive to color or texture differences.

• IS [52] measures the quality and diversity of generated images. It uses the Inception-V3 model to predict the class of each generated image. A good Inception Score means that the generated images are clear and belong to a specific class (high confidence), and that the generated images cover a wide range of classes (high diversity). Formally, Inception Score is calculated by:

$$
\mathrm{IS} = \exp (\mathbb {E} _ {\mathrm{x} \sim \mathrm{X}} [ D _ {K L} (p (\mathrm{y} | \mathrm{x}) | | p (\mathrm{y})) ]) \tag {A2}
$$

where x represents the generated images, X is the distribution of generated images. $p ( \mathrm { y } | \mathrm { x } )$ is the conditional probability distribution of the class y given the image x, predicted by the Inception model, p(y) is the marginal probability of class $\mathrm { y } . ~ D _ { K L }$ is the Kullback-Leibler divergence.

• Precision and Recall [33] are used to evaluate the quality of generated images by comparing them to real images. Precision measures how much the generated images resemble real images, while Recall measures how much of the real image distribution is captured by the generated images.

• Centered Kernel Alignment (CKA) is a widely adopted metric for quantifying neural network representations [12, 32], which has been demonstrated with several advantages: 1) CKA is invariant to orthogonal transformation and isotropic scaling, thus it is stable under various image transformations; 2) CKA can capture the non-linear correspondence between representations benefit from its kernel mapping in the kernel space; and 3) CKA can quantify the correspondence between different features across different widths, whereas previous metrics fail [32]. Formally, CKA is normalized from Hilbert-Schmidt Independence Criterion (HSIC) [18] to be invariant to orthogonal transformation and isotropic scaling:

$$
\mathrm{CKA} (\mathrm{X}, \mathrm{Y}) = \frac {\mathrm{HSIC} (\mathrm{x} , \mathrm{y})}{\sqrt {\mathrm{HSIC} (\mathrm{x} , \mathrm{x}) \mathrm{HSIC} (\mathrm{y} , \mathrm{y})}}. \tag {A3}
$$

HSIC identifies whether two distributions (X, Y) are independent: HSIC(K, L) = 1(n−1)2 $\begin{array} { r } { \mathrm { H S I C } ( K , L ) = \frac { 1 } { ( n - 1 ) ^ { 2 } } } \end{array}$ Tr(KHLH), where $K _ { i j } = k \left( \mathbf { x } _ { i } , \mathbf { x } _ { j } \right)$ and $L _ { i j } = l \left( \mathrm { y } _ { i } , \mathrm { y } _ { j } \right)$ , where k and l are kernels. Note that for kernel selections of k and l in Eq. (A3), we find that different kernels (RBF, polynomial, and linear) reflect similar discrepancies across various representations of DiTs, while the RBF kernel contributes to the distinguishability of quantitative results.

## D. Comparison Baselines

In this part, we briefly introduce the main concept of baseline methods that are used for our evaluation.

## D.1. Multi-step baseline models

• ADM [14] achieved improved synthesis performance with architectural improvement on traditional Unetbased diffusion models and developed classifier guidance to improve the synthesis fidelity for classconditional tasks.  
• VDM++ [29] demonstrated that commonly used diffusion model objectives equate to a weighted integral of ELBOs over different noise levels, where the weighting depends on the specific objective used. Based on this, a sample adaptive noise schedule was introduced for improved training efficiency.  
• CDM [26] proposed a cascaded architecture that trains multiple models across different resolutions, starting from the lowest resolution to higher resolution.  
• LDM [51] developed latent diffusion models that train diffusion in a low-dimensional compressed latent space to improve the training efficiency. Specifically, the images are first encoded into latent codes and then added noise for training, and the denoised latents are decoded back to pixel space for sampling.  
• MDTv2 [16] introduced an asymmetric encoderdecoder paradigm for efficient training of diffusion transformer. To stabilize the training and improve model performance, they further employ U-Net-like

Table A2. Detailed quantitative results of our systematic analysis. All implementation details strictly follow the default settings of SiT and REPA for our investigation. All baselines are reported using vanilla-REPA [70] for training.

<table><tr><td>Model</td><td>Alignment?</td><td>Encoder.</td><td>Align Depth.</td><td>Iter.</td><td> $FID_{\downarrow}$ </td><td> $sFID_{\downarrow}$ </td><td> $IS_{\uparrow}$ </td><td> $Prec.\uparrow$ </td><td> $Rec.\uparrow$ </td></tr><tr><td colspan="10">SiT-B</td></tr><tr><td rowspan="5"></td><td>X</td><td>X</td><td>X</td><td>50k</td><td>89.65</td><td>12.58</td><td>18.28</td><td>0.34</td><td>0.43</td></tr><tr><td>X</td><td>X</td><td>X</td><td>100k</td><td>40.46</td><td>6.15</td><td>36.26</td><td>0.52</td><td>0.49</td></tr><tr><td>X</td><td>X</td><td>X</td><td>200k</td><td>31.42</td><td>5.87</td><td>58.08</td><td>0.61</td><td>0.53</td></tr><tr><td>X</td><td>X</td><td>X</td><td>400k</td><td>12.69</td><td>5.29</td><td>106.21</td><td>0.71</td><td>0.54</td></tr><tr><td>X</td><td>X</td><td>X</td><td>450k</td><td>12.04</td><td>5.24</td><td>110.19</td><td>0.71</td><td>0.54</td></tr><tr><td colspan="10">REPA</td></tr><tr><td rowspan="15"></td><td>√</td><td>DINOv2-B</td><td>5</td><td>450k</td><td>5.37</td><td>5.35</td><td>175.07</td><td>0.75</td><td>0.58</td></tr><tr><td>√</td><td>DINOv2-B</td><td>8</td><td>450k</td><td>7.67</td><td>5.60</td><td>150.87</td><td>0.72</td><td>0.58</td></tr><tr><td>√</td><td>DINOv2-B</td><td>10</td><td>450k</td><td>10.85</td><td>6.12</td><td>128.34</td><td>0.70</td><td>0.58</td></tr><tr><td>√</td><td>DINOv2-B</td><td>[2,5,8]</td><td>450k</td><td>13.25</td><td>5.27</td><td>105.64</td><td>0.70</td><td>0.55</td></tr><tr><td>√</td><td>DINOv2-B</td><td>[3,6,9]</td><td>450k</td><td>13.54</td><td>5.50</td><td>104.88</td><td>0.69</td><td>0.56</td></tr><tr><td>√</td><td>MAE</td><td>5</td><td>450k</td><td>10.15</td><td>5.11</td><td>123.24</td><td>0.72</td><td>0.55</td></tr><tr><td>√</td><td>MAE</td><td>8</td><td>450k</td><td>11.40</td><td>5.22</td><td>115.20</td><td>0.72</td><td>0.55</td></tr><tr><td>√</td><td>MAE</td><td>10</td><td>450k</td><td>12.11</td><td>5.30</td><td>111.01</td><td>0.71</td><td>0.55</td></tr><tr><td>√</td><td>DINOv2, MAE</td><td>5</td><td>450k</td><td>5.77</td><td>5.10</td><td>166.15</td><td>0.76</td><td>0.57</td></tr><tr><td>√</td><td>DINOv2, MAE</td><td>8</td><td>450k</td><td>7.44</td><td>5.37</td><td>150.12</td><td>0.73</td><td>0.57</td></tr><tr><td>√</td><td>DINOv2, MAE</td><td>10</td><td>450k</td><td>11.03</td><td>6.17</td><td>124.89</td><td>0.70</td><td>0.55</td></tr><tr><td>√</td><td>DINOv2, MAE</td><td>[3,8]</td><td>450k</td><td>11.46</td><td>5.20</td><td>113.95</td><td>0.72</td><td>0.55</td></tr><tr><td>√</td><td>DINOv2, MAE</td><td>[5,10]</td><td>450k</td><td>11.73</td><td>5.22</td><td>111.77</td><td>0.71</td><td>0.54</td></tr><tr><td>√</td><td>DINOv2, MoCoV3</td><td>[3,8]</td><td>450k</td><td>11.36</td><td>5.24</td><td>115.75</td><td>0.71</td><td>0.55</td></tr><tr><td>√</td><td>DINOv2, MoCoV3</td><td>[5,10]</td><td>450k</td><td>12.71</td><td>6.08</td><td>104.46</td><td>0.66</td><td>0.56</td></tr></table>

long-shortcuts in the encoder and dense input-shortcuts in the decoder.

• MaskDiT [73] used a similar encoder-decoder architecture with MDTv2, while the model was trained with an auxiliary reconstruction objective like [22] to reconstruct masked inputs.  
• SD-DiT [75] extended the reconstruction-based MaskDiT architecture, while introducing a selfsupervised discrimination objective with a momentum encoder for improved training.  
• DiT [49] proposed to replace the conventional Unetbased architectures with transformers and further explored different condition injection mechanisms for conditional generation.  
• SiT [44] systematically investigated the connections between discrete diffusion to continuous flow matching and developed practical training configurations for achieving strong synthesis performance.  
• REPA [70] connected diffusion training dynamics and representation learning, revealing that pretrained external guidance could facilitate the representation learning of diffusion transformers.  
• REG [64] further advanced REPA with a decoupled representation alignment technique, which entangled image latents and class tokens to imporve the conditional discrimination capability.  
• E2E-REPA [35] unlocked a end-to-end training paradigm for joint tuning both the VAE and diffusion

models throughout the training process, improving the VAE itself and downstream generation performance simultaneously.

• SRA [27] leveraged representations from later layers with lower noise of the EMA teacher to guide representations of earlier layers with higher noise, enabling a scheme of self-alignment.  
• DispLoss [27] introduced a regularized dispersive loss to encourage internal features to spread out in the embedding space, thus facilitating the model to learn informative representations.

## D.2. One-step baseline models

• MeanFlow [17] introduced average velocity that was defined as the ratio of displacement to a time interval, with displacement given by the time integral of the instantaneous velocity. An intrinsic relation between the average and instantaneous velocities was then derived to guide efficient and effective one-step generative training.  
• Shortcut [15] enhanced the few-step flow matching by adding a self-consistency loss, designed to learn the relationships between flow behaviors observed at different discrete time points.  
• IMM [74] learned a model that enforces selfconsistency among stochastic interpolants evaluated at different points in time.  
• iCT [57] leveraged consistency constraints across net-

work outputs at different time steps to ensure that they predict the same endpoints along the trajectory.

## E. Detailed Results of Our Analysis

Detailed Quantitative Results. Tab. A2 presents the detailed quantitative results of our systematic analysis in Sec. 2. These quantitative results consistently reflect the findings in our analysis: 1) aligning external representations on more blocks (e.g., aligning DINOv2-B features on [2, 5, 6]-th blocks and [3, 6, 9]-th blocks) does not bring obvious performance improvements, indicating that indiscriminate alignment can be detrimental and reduce the overall diversity between blocks, such observation is also reflected by the CKA similarity heatmaps in Fig. 2. 2) aligning with earlier blocks (e.g., Block 5) generally results in better performance than aligning with later blocks (e.g., Block 10), as evidenced by the lower FID scores, which is also identified in the original REPA. 3) combining different external encoders (DINOv2 and MAE) on different blocks does not consistently improve performance, further indicating that the representation diversity across blocks is a crucial factor for high-quality synthesis. Together, the quantitative results and CKA similarity heatmaps in Fig. 2 consistently reveal that the key for representation learning is increasing the discrepancies of block representations. Which provides explainable motivations for our proposed method in explicitly encouraging the representation diversity from the perspective of input and internal features’ correlations.

CKA similarity across various timesteps. In Fig. 2, we present the CKA similarity heatmaps between representations of different blocks at the final denoising timestep. To investigate the difference of block representations across different timesteps, we calculate their representational discrepancies of different timesteps in Fig. A2. The results are computed from aligning DINOv2-B features on REPA-B for 400K training iterations like Fig. 2. We could observe that the representation similarities between different blocks across different timesteps show a very similar pattern. That is, the representational discrepancy across diffusion transformer blocks originates from the internal representation instead of different denoising timesteps. Moreover, we can see that as the inference steps increases, the representational discrepancy between different blocks at different timesteps tends to slightly increase as well. Such observation is reasonable because the noisy hidden states become less noisy throughout the sampling process.

## F. More Quantitative Results

Improving representation learning across various model scales on ImageNet 512×512. Tab. A3 presents the quantitative results of applying our proposed techniques to SiT and REPA across various model scales on ImageNet 512×512 without CFG. Similar to the results of ImageNet 256×256 in Tab. 1, our method consistently improves the performance of both SiT and REPA models across all scales, as evidenced by the reduction in FID and sFID scores and the increase in IS. Specifically, when applied to SiT-B, our method achieves a significant improvement in FID score (from 43.46 to 33.18 and sFID from 7.53 to 6.92), while also improving the IS score from 36.80 to 45.09. Similar improvements can be observed for REPA-B, with FID improving from 30.13 to 23.82 and IS increasing from 53.92 to 64.62. The benefits of our method are also evident for larger models. For SiT-XL, our approach reduces FID from 19.65 to 17.68 and increases IS from 71.57 to 76.45. For REPA-XL, the FID decreases from 7.91 to 7.18, and the IS increases from 127.83 to 137.09. These results further indicate that our method is effective in improving the representation learning capabilities of both SiT and REPA models, regardless of their scale. The consistent improvements in FID, sFID, and IS across different model sizes demonstrate the robustness and generalizability of our approach. The improvements in Precision and Recall also suggest that our method leads to better alignment between the generated images and the real data distribution.

Table A3. Variation in model-scale on ImageNet 512×512 without CFG. Our proposed method brings consistent performance gains across all model-scales when applied to both SiT and REPA. All baselines are reported using vanilla-REPA [70] for training.

<table><tr><td>Model</td><td>Iter.</td><td> $FID_{\downarrow}$ </td><td> $sFID_{\downarrow}$ </td><td> $IS_{\uparrow}$ </td><td> $Prec.\uparrow$ </td><td> $Rec.\uparrow$ </td></tr><tr><td>SiT-B</td><td>400k</td><td>43.46</td><td>7.53</td><td>36.80</td><td>0.60</td><td>0.64</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>33.18</td><td>6.92</td><td>45.09</td><td>0.67</td><td>0.65</td></tr><tr><td>REPA-B</td><td>400k</td><td>30.13</td><td>7.79</td><td>53.92</td><td>0.68</td><td>0.64</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>23.82</td><td>7.76</td><td>64.62</td><td>0.70</td><td>0.63</td></tr><tr><td>SiT-L</td><td>400k</td><td>22.75</td><td>5.78</td><td>64.05</td><td>0.73</td><td>0.63</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>19.19</td><td>5.74</td><td>71.78</td><td>0.76</td><td>0.61</td></tr><tr><td>REPA-L</td><td>400k</td><td>10.82</td><td>5.52</td><td>106.43</td><td>0.78</td><td>0.63</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>9.83</td><td>5.49</td><td>114.00</td><td>0.78</td><td>0.64</td></tr><tr><td>SiT-XL</td><td>400k</td><td>19.65</td><td>5.55</td><td>71.57</td><td>0.75</td><td>0.60</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>17.68</td><td>5.48</td><td>76.45</td><td>0.77</td><td>0.60</td></tr><tr><td>REPA-XL</td><td>400k</td><td>7.91</td><td>5.41</td><td>127.83</td><td>0.79</td><td>0.65</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>7.18</td><td>5.38</td><td>137.09</td><td>0.78</td><td>0.64</td></tr></table>

Comparison results across different model scales with different CFG scales. We mainly present comparison results without using classifier-free guidance (CFG) [47] in the main paper. In this part, we present comparison results across different model scales with CFG enabled to further investigate its impact and our performance. Specifically, we conduct experiments on ImageNet 256x256 on REPA using the DINOv2-B encoder for 400K training iterations across different model sizes (REPA-B, REPA-L, and REPA-XL). We systematically evaluated the performance with different CFG scales (1.0, representing no classifier-free guidance, and 1.35). Tab. A4 presents a detailed analysis of the impact of Classifier-Free Guidance (CFG) scale on the performance of our proposed method when applied to REPA models of varying sizes (REPA-B, REPA-L, and REPA-XL).

![](images/e122dfc88c153bd7ef0b97e9e82096b67bd1debad612a293dc6eee5aff41b3a6.jpg)  
Figure A2. CKA representation similarities across different timesteps. The representational discrepancies across different timesteps show similar correlations.

First, the results consistently demonstrate that increasing the CFG scale from 1.0 to 1.35 leads to significant improvements in image quality and diversity across all model scales. This is evidenced by the substantial increase in IS scores and the decrease in FID scores observed across all REPA model sizes when CFG is enabled. Second, our proposed method also gains consistent performance improvement across different model scales when CFG is enabled.

For instance, our model advances the FID score of REPA-B from 12.47 to 8.33 and IS score from 107.38 to 134.16 with CFG=1.35, attaining a >32% performance improvement on FID. Similarly, our model advances the FID score of REPA-XL from 3.50 to 3.16 and the IS score from 188.96 to 194.36 with CFG=1.35.

Furthermore, Tab. A5 presents the comparison results on on ImageNet 512×512 with CFG=1.35. Across all model scales, our method consistently improves the FID and sFID scores when CFG is used, indicating enhanced image quality and fidelity. For example, when applied to SiT-B, our method reduces the FID from 43.46 to 33.18 and the sFID from 7.53 to 6.92. Similarly, for REPA-B, the FID decreases from 30.13 to 23.82. Together with the results that were tested without using CFG, these results demonstrate the scalability and effectiveness of our proposed method to higher resolutions and different model sizes.

Table A4. Variation in alignment depth on ImageNet 256×256 with different CFG scales. CFG=1.0 means no classifier-free guidance is applied. Our proposed method brings consistent performance gains across all model-scales when applied to REPA with different alignment depths and evaluated with different CFG scales.

<table><tr><td>Model</td><td>Iter.</td><td>Encoder.</td><td>Align Depth.</td><td>CFG.</td><td> $FID_{\downarrow}$ </td><td> $sFID_{\downarrow}$ </td><td> $IS_{\uparrow}$ </td><td> $Prec.\uparrow$ </td><td> $Rec.\uparrow$ </td></tr><tr><td>REPA-B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.0</td><td>22.99</td><td>6.70</td><td>64.73</td><td>0.59</td><td>0.65</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.35</td><td>12.47</td><td>5.95</td><td>107.38</td><td>0.67</td><td>0.61</td></tr><tr><td>+Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.0</td><td>17.29</td><td>6.56</td><td>79.92</td><td>0.62</td><td>0.65</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.35</td><td>8.33</td><td>5.84</td><td>134.16</td><td>0.70</td><td>0.63</td></tr><tr><td>REPA-B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.0</td><td>27.94</td><td>7.19</td><td>54.32</td><td>0.56</td><td>0.64</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.35</td><td>16.46</td><td>6.38</td><td>90.97</td><td>0.64</td><td>0.62</td></tr><tr><td>+Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.0</td><td>23.27</td><td>6.82</td><td>62.63</td><td>0.59</td><td>0.65</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.35</td><td>12.92</td><td>6.08</td><td>104.46</td><td>0.66</td><td>0.63</td></tr><tr><td>REPA-L</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.0</td><td>12.02</td><td>6.77</td><td>40.09</td><td>0.51</td><td>0.63</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.35</td><td>5.14</td><td>4.74</td><td>157.71</td><td>0.75</td><td>0.61</td></tr><tr><td>+Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.0</td><td>10.01</td><td>5.47</td><td>107.68</td><td>0.69</td><td>0.64</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.35</td><td>4.18</td><td>4.61</td><td>172.26</td><td>0.76</td><td>0.61</td></tr><tr><td>REPA-L</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.0</td><td>9.57</td><td>5.34</td><td>113.42</td><td>0.69</td><td>0.66</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.35</td><td>3.86</td><td>4.82</td><td>183.18</td><td>0.75</td><td>0.63</td></tr><tr><td>+Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.0</td><td>8.47</td><td>5.42</td><td>123.03</td><td>0.69</td><td>0.67</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.35</td><td>3.39</td><td>4.80</td><td>196.08</td><td>0.76</td><td>0.63</td></tr><tr><td>REPA-XL</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.0</td><td>8.27</td><td>5.19</td><td>123.85</td><td>0.69</td><td>0.66</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.35</td><td>3.33</td><td>4.73</td><td>196.52</td><td>0.75</td><td>0.64</td></tr><tr><td>+Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.0</td><td>8.18</td><td>5.01</td><td>126.63</td><td>0.70</td><td>0.65</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>5</td><td>1.35</td><td>3.17</td><td>4.71</td><td>198.30</td><td>0.77</td><td>0.62</td></tr><tr><td>REPA-XL</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.0</td><td>8.73</td><td>5.21</td><td>118.68</td><td>0.69</td><td>0.65</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.35</td><td>3.50</td><td>4.72</td><td>188.96</td><td>0.76</td><td>0.63</td></tr><tr><td>+Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.0</td><td>8.09</td><td>5.02</td><td>123.23</td><td>0.70</td><td>0.65</td></tr><tr><td></td><td>400K</td><td>DINOv2-B</td><td>8</td><td>1.35</td><td>3.16</td><td>5.60</td><td>194.36</td><td>0.77</td><td>0.62</td></tr></table>

Quantitative results of applying our method on REPA with alignment on different blocks. Tab. A4 also provides insight into the impact of the depth of alignment on the performance of our proposed method. We evaluated the models with alignment depths of 5 and 8, while keeping other parameters constant. The results suggest that increasing the alignment depth from 5 to 8 can have varying effects depending on the model size, suggesting that the optimal alignment depth may depend on the interplay between model size. Despite these variations, our method consistently improves upon the baseline REPA models when performing alignment on different blocks, with or without CFG. For example, REPA-XL with our method and an alignment depth of 5 achieves an FID score of 3.17 at CFG 1.35, compared to 3.33 for the baseline. Similarly, the IS score improves from 196.52 to 198.30. This consistent trend of improvement, regardless of alignment depth, demonstrating the effectiveness of our approach in enhancing image generation. The consistent improvements observed across different alignment depths and model sizes further demonstrate the robustness and generalizability of our approach.

Table A5. Variation in model-scale on ImageNet 512×512 with CFG=1.35. Our proposed method brings consistent performance gains across all model-scales when applied to both SiT and REPA.

<table><tr><td>Model</td><td>Iter.</td><td> $FID_{\downarrow}$ </td><td> $sFID_{\downarrow}$ </td><td> $IS_{\uparrow}$ </td><td> $Prec.\uparrow$ </td><td> $Rec.\uparrow$ </td></tr><tr><td>SiT-B</td><td>400k</td><td>32.77</td><td>6.95</td><td>50.85</td><td>0.67</td><td>0.62</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>23.96</td><td>6.45</td><td>63.07</td><td>0.73</td><td>0.61</td></tr><tr><td>REPA-B</td><td>400k</td><td>21.27</td><td>7.34</td><td>78.25</td><td>0.73</td><td>0.62</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>16.44</td><td>7.30</td><td>92.11</td><td>0.75</td><td>0.63</td></tr><tr><td>SiT-L</td><td>400k</td><td>14.85</td><td>5.41</td><td>91.49</td><td>0.78</td><td>0.60</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>12.44</td><td>5.41</td><td>101.08</td><td>0.79</td><td>0.58</td></tr><tr><td>REPA-L</td><td>400k</td><td>5.57</td><td>5.35</td><td>158.39</td><td>0.80</td><td>0.62</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>4.66</td><td>5.58</td><td>173.94</td><td>0.80</td><td>0.64</td></tr><tr><td>SiT-XL</td><td>400k</td><td>12.50</td><td>5.17</td><td>102.38</td><td>0.79</td><td>0.58</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>11.24</td><td>5.28</td><td>107.73</td><td>0.81</td><td>0.58</td></tr><tr><td>REPA-XL</td><td>400k</td><td>4.30</td><td>5.09</td><td>174.70</td><td>0.81</td><td>0.61</td></tr><tr><td>+ (Ours)</td><td>400k</td><td>3.98</td><td>5.48</td><td>184.73</td><td>0.80</td><td>0.62</td></tr></table>

Table A6. Ablation analysis on different components with CFG=1.35.

<table><tr><td>Component</td><td> $\text{FID}_{\downarrow}$ </td><td> $\text{sFID}_{\downarrow}$ </td><td> $\text{IS}_{\uparrow}$ </td><td> $Prec.\uparrow$ </td><td> $Rec.\uparrow$ </td></tr><tr><td>SiT-B Baseline</td><td>23.28</td><td>6.00</td><td>65.23</td><td>0.61</td><td>0.60</td></tr><tr><td>SiT-B + full</td><td>16.21</td><td>5.45</td><td>84.00</td><td>0.66</td><td>0.60</td></tr><tr><td>w/o diversity</td><td>20.07</td><td>5.72</td><td>73.65</td><td>0.63</td><td>0.60</td></tr><tr><td>w/o residual</td><td>20.76</td><td>5.77</td><td>69.23</td><td>0.61</td><td>0.61</td></tr><tr><td>REPA</td><td>12.47</td><td>5.85</td><td>107.38</td><td>0.61</td><td>0.62</td></tr><tr><td>REPA-B + full</td><td>8.34</td><td>5.64</td><td>134.16</td><td>0.70</td><td>0.63</td></tr><tr><td>w/o diversity</td><td>10.75</td><td>5.75</td><td>115.42</td><td>0.68</td><td>0.62</td></tr><tr><td>w/o residual</td><td>11.02</td><td>5.77</td><td>112.93</td><td>0.64</td><td>0.62</td></tr></table>

Table A7. Ablation analysis on selecting different number of layers for diversity loss

<table><tr><td> $\mathcal{P}$ </td><td>SiT-XL</td><td>5</td><td>10</td><td>15</td><td>20</td><td>all</td></tr><tr><td> $FID_{\downarrow}$ </td><td>18.77</td><td>16.85</td><td>16.10</td><td>16.01</td><td>15.84</td><td>15.77</td></tr><tr><td> $IS_{\uparrow}$ </td><td>71.44</td><td>77.62</td><td>79.47</td><td>82.05</td><td>83.95</td><td>85.64</td></tr><tr><td>Time (h)</td><td>18.66</td><td>19.90</td><td>21.02</td><td>23.45</td><td>25.96</td><td>28.50</td></tr></table>

## G. More Ablation and Analysis Results

Ablation on layer selection P. In our implementation for layer selection P, we randomly select 10 layers to compute the diversity loss for experiments. To investigate its impact, here we testify the impact of P on SiT-L (24 layers) for 400K training steps. The results in Tab. A7 show that selecting more layers improves the performance but increases the training time. In particular, selecting 10 layers yields a better trade-off between performance and efficiency.

Ablation on different loss variants. Here we further perform ablation on each loss component of the proposed diversity loss on SiT-B baseline. The results in Tab. A8 show the effectiveness of each loss, consistent with the findings of REPA results in Tab. 7. Specifically, using all components of the diversity loss (SiT-B + full) achieves the best performance and removing any single loss component degrades performance.

Table A8. Ablation analysis on different loss variants on SiT-B baseline.

<table><tr><td>Component</td><td> $\text{FID}_{\downarrow}$ </td><td> $\text{sFID}_{\downarrow}$ </td><td> $\text{IS}_{\uparrow}$ </td><td> $Prec.\uparrow$ </td><td> $Rec.\uparrow$ </td></tr><tr><td>SiT-B + full</td><td>28.05</td><td>6.04</td><td>50.66</td><td>0.57</td><td>0.63</td></tr><tr><td>only  $\mathcal{L}_{orth}$ </td><td>31.32</td><td>6.45</td><td>47.09</td><td>0.56</td><td>0.63</td></tr><tr><td>only  $\mathcal{L}_{MI}$ </td><td>29.97</td><td>6.21</td><td>48.23</td><td>0.57</td><td>0.63</td></tr><tr><td>only  $\mathcal{L}_{div}$ </td><td>36.12</td><td>6.64</td><td>45.04</td><td>0.55</td><td>0.62</td></tr></table>

Ablation on designed components with CFG. Tab. A6 presents the ablative results on the designed components of our DiverseDiT with CFG. We could see that applying the CFG consistently improves the overall scores. Similar to the results in Tab. 6, the results clearly demonstrate the importance of both the representation diversity loss and the long residual connections for optimal performance. Removing the diversity loss (w/o diversity) worsens the FID scores for both SiT-B (from 23.28 to 20.07) and REPA-B (from 12.47 to 10.75). Similarly, removing the long residual connections (w/o residual) also noticeably increases FID for both baseline models. Despite some performance degradation, we can observe that applying any of our proposed techniques to the baseline methods, i.e., REPA-B and SiT-B, brings substantial performance improvements. For instance, with only long residual connections (w/o diversity), we achieve an FID of 20.07 on SiT-B and an FID of 10.75 on REPA-B, which are better than the original baseline results (23.28 for SiT-B and 12.47 for REPA-B). Similar conclusions could be observed from the results of only diversity loss (w/o residual) as well. These results confirm that both components of DiverseDiT play a crucial role in promoting diverse representation learning and improving the performance.

Effect of diversity loss variant with CFG. Tab. A9 presents an ablation analysis on different loss variants with CFG=1.35. Similar to the previous results, the table demonstrates the importance of each loss component for optimal performance. Removing any of the loss components, namely Lorth, LMI, or Ldiv, degrades the FID score compared to the REPA-B + full configuration (8.34). While using only Lorth results in an FID of 10.98, using only LMI gives an FID of 10.78, and using only Ldiv improves the FID to 8.59. These results confirm that each loss component plays a role in improving the model’s performance, which is also reflected by the better results compared with the REPA baseline when each loss is used in isolation.

Combining with existing methods for further improvement with CFG. Tab. A10 further explores the effect of combining our method with existing approaches, specifically DispLoss [62] and SRA [27], on the SiT-B baseline with CFG=1.35. Adding our method to the SiT-B baseline improves the FID from 23.28 to 16.21. Further combining with DispLoss results in an even lower FID of 13.73. This demonstrates that our method is complementary to existing techniques and can be combined with them to achieve further improvements in image generation quality. Note that SRA and DispLoss require no additional external models for representation alignment, and combining our proposed method with them achieves a better performance than that of REPA, which needs pretrained models as guidance, demonstrating the potential for representation learning through internal mechanisms.

Table A9. Ablation analysis on different loss variants with CFG=1.35.

<table><tr><td>Component</td><td> $\text{FID}_\downarrow$ </td><td> $\text{sFID}_\downarrow$ </td><td> $\text{IS}_\uparrow$ </td><td> $\text{Prec.}_\uparrow$ </td><td> $\text{Rec.}_\uparrow$ </td></tr><tr><td>REPA Baseline</td><td>12.47</td><td>5.85</td><td>107.38</td><td>0.61</td><td>0.62</td></tr><tr><td>REPA-B + full</td><td>8.34</td><td>5.64</td><td>134.16</td><td>0.70</td><td>0.63</td></tr><tr><td>only  $\mathcal{L}_{\text{orth}}$ </td><td>10.98</td><td>5.78</td><td>115.03</td><td>0.68</td><td>0.62</td></tr><tr><td>only  $\mathcal{L}_{\text{MI}}$ </td><td>10.78</td><td>5.76</td><td>115.95</td><td>0.69</td><td>0.63</td></tr><tr><td>only  $\mathcal{L}_{\text{div}}$ </td><td>8.59</td><td>5.77</td><td>131.69</td><td>0.70</td><td>0.63</td></tr></table>

Table A10. Combining our method with prior approaches with CFG=1.35.

<table><tr><td>Component</td><td> $\mathbf{FID}_{\downarrow}$ </td><td> $sFID_{\downarrow}$ </td><td> $IS_{\uparrow}$ </td><td> $Prec._{\uparrow}$ </td><td> $Rec._{\uparrow}$ </td></tr><tr><td>REPA Baseline</td><td>12.47</td><td>5.85</td><td>107.38</td><td>0.61</td><td>0.62</td></tr><tr><td>SiT-B Baseline</td><td>23.28</td><td>6.00</td><td>65.23</td><td>0.61</td><td>0.60</td></tr><tr><td>+ Ours</td><td>16.21</td><td>5.45</td><td>84.00</td><td>0.66</td><td>0.60</td></tr><tr><td>++ DispLoss [62]</td><td>13.73</td><td>5.76</td><td>95.31</td><td>0.68</td><td>0.60</td></tr><tr><td>+++ SRA [27]</td><td>11.25</td><td>5.37</td><td>108.15</td><td>0.69</td><td>0.61</td></tr></table>

## H. Limitations and Future Work

Limitations. Despite a comprehensive investigation, our analysis could be extended in several aspects: For instance, whether DiverseDiT can be effectively adapted to diverse generation tasks, such as text-to-image synthesis or image editing, remains an open question. Besides, performing similar analysis on representation learning of other models like large-language models might reveal more interesting findings. Additionally, we do not perform extensive hyperparameter searching for the optimal performance in our experiments, the full potential of our proposed representation diversity loss could be further unlocked. Nevertheless, our study could provide potential guidelines for developing more effective methods in learning informative representations.

Future work. For future work, we plan to extend our analysis and evaluation on text-to-image synthesis tasks. We aim to investigate the application of our representation diversity loss to other generative models and modalities, such as video generation and 3D shape generation. Exploring different architectures and training strategies in conjunction with our proposed loss function could potentially lead to even more significant improvements in the quality and diversity of generated content. Meanwhile, we intend to explore theoretical connections between representation diversity and other desirable properties of generative models, such as robustness to adversarial attacks and generalization to unseen data distributions. Furthermore, considering that our proposed diversity loss alone could likely be applied as a fine-tuning step for pre-trained models without any architectural changes, we plan to explore this in our ongoing research.

## I. More Qualitative Results

We present more uncurated generation results of our DiverseDiT-XL on ImageNet 256×256 in Fig. A3 - Fig. A19 with CFG (w = 4.0).

![](images/988607354087e5a1844535d6e56aaca60d81c8163141cb448492c5c498ea34bd.jpg)

<details>
<summary>natural_image</summary>

Underwater photos of multiple sharks in full view, showing dorsal and lateral views (no text or symbols)
</details>

Figure A3. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with $w = 4 . 0 ,$ , the lass label is “Great white shark” (2).

![](images/1f39a3b21ec86bf5af4607051606d8150d3ec300d6ed8e1dc85eae9fc0858d7a.jpg)  
Figure A4. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with $w = 4 . 0 ,$ the lass label is “Chickadee” (19).

![](images/734d69842018dab68acb63e08fb22978442ab739b78072f46cad915c41ba6da9.jpg)

<details>
<summary>natural_image</summary>

Grid of twelve turtle species in various poses, including shell, turtle, and sea turtle, with no visible text or symbols.
</details>

Figure A5. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with $w = 4 . 0 ,$ , the lass label is “Terrapin” (36).

![](images/4f83ba3af0fc3ce6f0bb0efe7a165465605cd9f4f3d7b11208e8cef42ffbec5b.jpg)

<details>
<summary>natural_image</summary>

Grid of ten black-and-white photos of blue and gray herons, including birds, wetlands, and water, all in natural habitat (no text or symbols)
</details>

Figure A6. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with w = 4.0, the lass label is “Little blue heron, Egretta caerulea” (131).

![](images/5198e3d61d3f52d0444d90e24e277ce915c939c65c9702fcc0fb5d5395612706.jpg)

<details>
<summary>natural_image</summary>

Grid of eleven different brown and white cowl dogs in various poses, each with a distinct headband and photo frame (no text or symbols visible)
</details>

Figure A7. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with w = 4.0, the lass label is “Blenheim spaniel” (156).

![](images/4ed2d96f893470495a6fe7f9aadd9e7ea03f5e8936d7771ac5309fa16b4bd258.jpg)

<details>
<summary>natural_image</summary>

Grid of 12 photos of Golden F------ dog photos, including sitting, standing, and relaxing (no text or symbols)
</details>

Figure A8. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with w = 4.0, the lass label is “Golden retriever” (207).

![](images/4e63a9ae7a91d4169904b1a9ac237030a4d6b93092504aa75b663733da48031b.jpg)  
Figure A9. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with w = 4.0, the lass label is “Arctic fox, White fox, Alopex lagopus” (279).

![](images/50c9edf5854f885e09b8d06b9c4bfe2be02a025d6cb1c505414f19ce2e19a60c.jpg)

<details>
<summary>natural_image</summary>

Grid of 12 photos of a red panda in various poses and ecosystems, including climbing branches, peckies, and walking through snow (no text or symbols visible)
</details>

Figure A10. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with w = 4.0, the lass label is “lesser panda, Red panda, Panda, Bear cat, Cat bear, Ailurus fulgens” (387).

![](images/b53aea811e870a59e5ca419200157f884a065f03a5ab28f3f733902d793dc0f6.jpg)  
Figure A11. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with $w = 4 . 0 ,$ the lass label is “Balloon’ (417).

![](images/d1558a6973f78736fafee754095f976159a0a6817b82adcf82ce52c993c97bee.jpg)

<details>
<summary>natural_image</summary>

Collage of 16 historic UK-style architecture photos including castle towers, gardens, and water reflections (no visible text or symbols)
</details>

Figure A12. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with $w = 4 . 0 ,$ , the lass label is “Castle’ (483).

![](images/99017ba423ea98e51da93922731c9c4ea46ba0e585f89f096582ff2475c64c5d.jpg)

<details>
<summary>natural_image</summary>

Collage of various classic car models including convertible, red convertible, and blue convertible cars, displayed in a grid layout with no visible text or symbols.
</details>

Figure A13. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with w = 4.0, the lass label is “Check, Convertible’ (511).

![](images/7b8df6d43823e91fdf33c6c8c31706d3af45d3ea14cf0b055b2bc692dc0122a0.jpg)

<details>
<summary>text_image</summary>

Collage of laptops displaying various content and interface elements, including game design, video, and text content.
</details>

Figure A14. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with $w = 4 . 0 ,$ , the lass label is “Laptop, Laptop computer’ (620).

![](images/eb049e45541ee6f31d3be4ba7a52e51e947c260b104c109288ad92e120406a8b.jpg)  
Figure A15. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with $w = 4 . 0 ,$ , the lass label is “Pillow’ (721).

![](images/fb727f3f076a9c661ed60c2061d969123f2e1f5e267695b16cc7568dd417cc8c.jpg)

<details>
<summary>natural_image</summary>

Collage of twelve vintage tram tracks in various colors and styles, including vintage buildings, highways, and urban buildings (no visible text or signage)
</details>

Figure A16. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with w = 4.0, the lass label is “Check, Streetcar, Tram, Tramcar, Trolley, Trolley car’ (829).

![](images/7b31c20e36ff25926a01b029ca957e74726db9ac4baecbcfebcdb7823c726453.jpg)

<details>
<summary>natural_image</summary>

Underwater coral and reef scenes with various marine and natural environments (no text or symbols visible)
</details>

Figure A17. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with $w = 4 . 0 ,$ , the lass label is “Coral reef’ (973).

![](images/a094a26fefcf2af9dc57bf2f25f5a7a8c170b2fb385907ea06f653bf38fdf117.jpg)

<details>
<summary>natural_image</summary>

Collage of scenic lake views including mountains, forests, and a lakeside with trees (no text or symbols visible)
</details>

Figure A18. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with w = 4.0, the lass label is “Lakeside, lakeshore’ (975).

![](images/72139c7e258e48d46bc936dfced26146b1ab551d913947d5b70ff9f4bc0e44ad.jpg)  
Figure A19. Uncurated generation results of our DiverseDiT-XL on ImageNet 256×256. We use classifier-free guidance with w = 4.0, the lass label is “Volcano’ (980).