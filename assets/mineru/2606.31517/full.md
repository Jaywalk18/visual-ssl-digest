# UNSUPERVISED DATA-EFFICIENT CROSS-MODAL RETRIEVAL WITH GLOBAL-NEIGHBORHOOD ALIGNMENT HASHING

Runhao Li $^{1}$ , Xiaoxu Ma $^{2}$ , Zhenyu Weng $^{2,*}$ , Yue Zhang $^{3}$ , Guibo Luo $^{4}$ , Huiping Zhuang $^{2}$ , Zhiping Lin $^{1}$ , Yap-Peng Tan $^{5,1}$

$^{1}$ School of Electrical and Electronic Engineering, Nanyang Technological University, Singapore $^{2}$ Shien-Ming Wu School of Intelligent Engineering, South China University of Technology, China $^{3}$ College of Computer and Information Engineering, Henan Normal University, China $^{4}$ Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology, Peking University Shenzhen Graduate School, China $^{5}$ VinUniversity, Viet Nam

## ABSTRACT

Compared to supervised cross-modal hashing (CMH), unsupervised CMH reduces the reliance on manual labeling by learning binary codes from unlabeled image-text pairs. However, existing unsupervised CMH methods often rely on large-scale image-text pairs, which are costly to collect. To address this limitation, we propose Global-Neighborhood Alignment Hashing (GNAH), a novel approach that preserves the semantic structure of vision-language foundation models within a compact binary Hamming space using only a limited number of image-text pairs. Specifically, GNAH captures global structural information from the continuous latent space and transfers it into the binary Hamming space through a Prototype-Anchored Global Alignment module. In addition, GNAH extends conventional pairwise contrastive learning by modeling stochastic neighborhood relationships via a Contrastive Stochastic Neighborhood Alignment module, thereby alleviating overfitting to sparse pairwise correlations. Extensive experiments demonstrate that GNAH consistently outperforms existing unsupervised cross-modal retrieval methods under data-constrained settings, offering a practical solution for real-world CMH applications.

Index Terms— Unsupervised cross-modal retrieval, data-efficient learning

## 1. INTRODUCTION

The surge in multimedia data, coupled with the advancements in retrieval-augmented generation, has made cross-modal retrieval a compelling topic in both academia and industry. The goal of cross-modal retrieval is to retrieve related instances from one modality (e.g., image) using a query from another modality (e.g., text). To achieve efficient cross-modal retrieval, cross-modal hashing (CMH) methods bridge the heterogeneity gap between different modalities by learning a common Hamming space, where instances from different modalities are mapped into compact binary hash codes.

Existing CMH methods can be roughly classified into supervised $[1-4]$ and unsupervised categories $[5-11]$ . Unsupervised CMH methods learn hash functions to preserve data correlation without intensive data labeling. Although these methods reduce the dependence on labeled data, they still require a considerable amount of image-text pairs to effectively train the hash functions. These image-text pairs are often scarce and expensive to acquire, particularly when dealing with specialized or private datasets. Meanwhile, recent vision-language foundation models, such as Contrastive Language-Image Pretraining (CLIP) $[12]$ , have demonstrated remarkable effectiveness in modality alignment, a crucial aspect of cross-modal hashing. This has led to increasing interest in leveraging CLIP for unsupervised CMH $[5,6,8,9]$ . However, most existing approaches limit CLIP's role to a feature extractor or focus only on instance-level relationships, failing to exploit its full potential. Given CLIP's Internet-scale pretraining and proven knowledge transfer capabilities in data-efficient settings for many downstream tasks $[13,14]$ , a critical question remains: How can we fully harness CLIP for data-efficient unsupervised CMH?

To bridge this gap, a promising direction is to move beyond simple feature extraction and instance-to-instance correlation learning, instead focusing on leveraging CLIP's global-neighborhood structural priors to capture semantic rerelationships across diverse image-text pairs. By transitioning from an instance-to-instance learning paradigm to instance-to-prototype and instance-to-neighborhood approaches, unsupervised CMH can reduce overfitting to individual samples and produce more robust and meaningful hash codes, particularly under data-efficient conditions. Driven by these motivations, we propose Global-Neighborhood Alignment Hashing (GNAH), a novel approach that enhances data-efficient hashing performance by maintaining the semantic structure of the dataset, facilitating effective knowledge transfer from vision-language foundation models into a compact binary Hamming space. The main contributions and novelty of this work are summarized as follows:

![](images/cfee2f07f514e821ae1611e75792cee879e43e8fd2a20595609b26c6fd16aa6b.jpg)  
Fig. 1: Framework of GNAH. Image and text features from CLIP are mapped into a Hamming space via modality-specific hash functions. Structural information is preserved through Prototype-Anchored Global Alignment for global semantics and Contrastive Stochastic Neighborhood Alignment for local neighborhood priors.

\- Our GNAH is a novel method to explore unsupervised cross-modal hashing under data-efficient conditions, reducing the need for well-aligned image-text pairs.

\- We introduce Prototype-Anchored Global Alignment, which employs representative prototypes as robust semantic references to capture main semantics and maintain global structure in the binary Hamming space.

\- We develop Contrastive Stochastic Neighborhood Alignment to exploit CLIP's local structural prior by retaining stochastic neighborhood relationships, which mitigates overfitting to limited pairwise correlations.

\- Extensive experiments show that GNAH outperforms state-of-the-art unsupervised CMH methods and remains competitive with the performance of supervised CMH methods under data-efficient conditions.

## 2. METHODOLOGY

## 2.1. Methodology Overview

First, inspired by the success of prototype-based methods in data-efficient learning [15-17], we leverage prototypes as robust semantic anchors. Because prototypes offer stable representations that are inherently less vulnerable to noise, they achieve two core goals: capturing the main semantic information for preservation in compact binary codes, and providing resilience to noise through instance-to-prototype relationships. This forms the basis of our Prototype-Anchored Global Alignment (PAGA).

Second, while recent contrastive learning methods [18] have demonstrated strong effectiveness in cross-modal retrieval, their strict reliance on pairwise alignment makes them prone to overfitting when pairwise correspondences are scarce. To mitigate this, we excavate CLIP's structural prior and align paired samples with their pairwise neighborhood distributions. Transitioning from instance-to-instance to instance-to-neighborhood alignment forms the foundation of our Contrastive Stochastic Neighborhood Alignment (CSNA), which enhances local neighborhood coherence between the latent and Hamming spaces.

## 2.2. Problem Formulation

Let $D = \{\mathbf{x}_i, \mathbf{y}_i\}_{i=1}^n$ denote a cross-modal dataset with $n$ image-text pairs, where $\mathbf{x}_i \in \mathbb{R}^{d \times 1}$ is the $i^{th}$ instance from image modality, $\mathbf{y}_i \in \mathbb{R}^{d \times 1}$ is the $i^{th}$ instance from text modality, $d$ is the dimension of image and text features. Simultaneously, we obtain the features of the dataset through the image and text encoders of CLIP as $D_f = \{\mathbf{f}_i^x, \mathbf{f}_i^y\}_{i=1}^n$ . Let $B^x = \{\mathbf{b}_i^x\}_{i=1}^n$ and $B^y = \{\mathbf{b}_i^y\}_{i=1}^n$ respectively denote the binary codes of image and text modalities, where $\mathbf{b}_i^* \in \{-1, +1\}^L, * \in \{x, y\}$ , and $L$ is the length of hash codes. We formulate the two hash functions for image and text modalities as $f^x(\mathbf{f}^x | \Theta^x)$ and $f^y(\mathbf{f}^y | \Theta^y)$ respectively, where $\Theta^x$ and $\Theta^y$ are the network parameters to be learned. The outputs of the hash functions are relaxed hash codes defined as $\mathbf{h}_i^x = f^x(\mathbf{f}_i^x)$ and $\mathbf{h}_i^y = f^y(\mathbf{f}_i^y)$ for the $i^{th}$ image-text pair.

The binary hash code of an image-text pair is computed as:

$$
\mathbf {b} _ {i} ^ {*} = \operatorname{sgn} (\mathbf {h} _ {i} ^ {*}), * \in \{x, y \},\tag{1}
$$

where $\operatorname{sgn}(\cdot)$ is the element-wise sign function.

## 2.3. Prototype-Anchored Global Alignment

Alignment with prototypical anchors. In this study, we adopt the K-means algorithm to select the prototypical anchors. The K-means cluster centroids are denoted by $C = \{c_{j}\}_{j=1}^{K}$ , where the cluster centroids $C \in R^{d \times K}$ are obtained by the K-means algorithm from the multi-modal features $D_{f}$ , and the cluster number K is determined by the Elbow Method [19] to provide a good coverage of the features and represent the global structure. Meanwhile, in the Hamming space, we initialize K binary prototypical anchors $C_{b} = \{c_{bj}\}_{j=1}^{K}$ corresponding to the prototypical anchors in the latent space, where $c_{bj} \in \{-1, 1\}^{L \times 1}$ . It can be represented in a matrix form as $C_{b} \in \{-1, 1\}^{L \times K}$ . We enforce $||h_{i}^{*}|| = 1 (* \in \{x, y\})$ and $||c_{bj}|| = 1$ via $\ell_{2}$ -normalization as in other methods [18, 20]. We avoid directly generating binary prototypical anchors by projecting the latent prototypical anchors to the binary Hamming space through hash functions to decouple their relationships, reducing the risk of overfitting to specific feature distributions. Then, we quantify the relationships between a sample and prototypical anchors as:

$$
\mathbf {p} _ {i} ^ {*} = \operatorname{softmax} \left(\mathbf {C} ^ {T} \mathbf {f} _ {i} ^ {*} / \tau\right), \quad \hat {\mathbf {p}} _ {i} ^ {*} = \operatorname{softmax} \left(\mathbf {C} _ {b} ^ {T} \mathbf {h} _ {i} ^ {*} / \tau\right),\tag{2}
$$

where $\tau$ is the temperature parameter. Since prototypical anchors possess global structural information, we maintain the global structure by aligning the relationships between each sample and prototypical anchors. The loss function for Prototype-Anchored Global Alignment (PAGA) is defined as:

$$
\ell_ {p} = \sum_ {*} \sum_ {i = 1} ^ {n} \mathcal {D} _ {\mathcal {K L}} (\mathbf {p} _ {i} ^ {*} | | \hat {\mathbf {p}} _ {i} ^ {*}),\tag{3}
$$

where $\mathcal{D}_{\mathcal{KL}}(\cdot ||\cdot)$ denotes the KL divergence.

Binary prototypical anchor update. As directly optimizing the binary prototypical anchors is an NP-hard problem [21], we attempt to make the binary prototypical anchors learnable. Specifically, we create prototypical anchors $\{s_{j}\}_{j=1}^{K}$ using continuous values, and then determine binary prototypical anchors using $\mathbf{c}_{bj} = \operatorname{sgn}(\mathbf{s}_{j})$ . This binarization step helps mitigate binarization error, especially given the many samples that gather around these anchors. Inspired by the standard K-means clustering algorithm, the new $j^{th}$ prototypical anchor is calculated as:

$$
\mathbf {s} _ {j} = \frac {\sum_ {i = 1} ^ {n} (\mathbb {1} _ {j} (l _ {i} ^ {x}) \mathbf {h} _ {i} ^ {x} + \mathbb {1} _ {j} (l _ {i} ^ {y}) \mathbf {h} _ {i} ^ {y})}{\sum_ {i = 1} ^ {n} (\mathbb {1} _ {j} (l _ {i} ^ {x}) + \mathbb {1} _ {j} (l _ {i} ^ {y}))},\tag{4}
$$

where $l_{i}^{x} \in \{1, \ldots, K\}$ denotes the nearest prototypical anchor assignment of the $i^{th}$ instance from image modality, $l_{i}^{y} \in \{1, \ldots, K\}$ denotes the nearest prototypical anchor assignment of the $i^{th}$ instance from text modality, and $\mathbb{1}_{j}(\cdot)$ is an indicator function defined as:

$$
\mathbb {1} _ {j} (a) = \left\{ \begin{array}{l l} 1 & \quad i f \quad a = j \\ 0 & \quad \text { otherwise. } \end{array} \right.\tag{5}
$$

## 2.4. Contrastive Stochastic Neighborhood Alignment

Given an image-text pair $(\mathbf{x}_{i},\mathbf{y}_{i})$ , we define its adaptive neighborhood representation by estimating a neighborhood mean $\mu_{i}$ and variance $\sigma_{i}^{2}$ , which serve as statistical descriptors for a Gaussian initialization:

$$
\pmb {\mu} _ {i} = \mathrm{norm} (\mathbf {f} _ {i} ^ {x} + \mathbf {f} _ {i} ^ {y}), \quad \sigma_ {i} ^ {2} = \frac {1}{4} \| \mathbf {f} _ {i} ^ {x} - \mathbf {f} _ {i} ^ {y} \| _ {2} ^ {2},\tag{6}
$$

where $\text{norm}(\cdot)$ denotes the $\ell_{2}$ -normalization. Using the computed $\mu_{i}$ and $\sigma_{i}^{2}$ , we sample from a Gaussian distribution to generate a perturbed representation that captures the neighborhood structure:

$$
\tilde {\mathbf {f}} _ {i} = \mathrm{norm} (\pmb {\mu} _ {i} + \sigma_ {i} \pmb {\epsilon}), \quad \pmb {\epsilon} \sim \mathcal {N} (0, \mathbf {I}).\tag{7}
$$

Contrastive neighborhood alignment. Once the adaptive neighborhood representations are generated, we project them into the relaxed Hamming space:

$$
\tilde {\mathbf {h}} _ {i} ^ {x} = f ^ {x} (\tilde {\mathbf {f}} _ {i}), \quad \tilde {\mathbf {h}} _ {i} ^ {y} = f ^ {y} (\tilde {\mathbf {f}} _ {i}).\tag{8}
$$

To enforce alignment within the relaxed Hamming space, we define the contrastive loss function for an image-text pair $(\mathbf{x}_{i}, \mathbf{y}_{i})$ as follows:

$$
\begin{array}{l} \ell (\mathbf {x} _ {i}, \mathbf {y} _ {i}) = \\ - \log \frac {\exp (\mathbf {h} _ {i} ^ {x T} \tilde {\mathbf {h}} _ {i} ^ {y} / \tau)}{\exp (\mathbf {h} _ {i} ^ {x T} \tilde {\mathbf {h}} _ {i} ^ {y} / \tau) + \sum_ {*} \sum_ {j = 1 , j \neq i} ^ {n} \exp (\mathbf {h} _ {i} ^ {x T} \mathbf {h} _ {j} ^ {*} / \tau)}. \end{array}\tag{9}
$$

This loss function considers the i-th instance from the image modality as an anchor and evaluates similarity over both image and text modalities. A symmetric loss term, $\ell(\mathbf{y}_{i},\mathbf{x}_{i})$ , is computed similarly by treating the text modality as the anchor. The loss function for Contrastive Stochastic Neighborhood Alignment (CSNA) is defined as:

$$
\ell_ {c} = \sum_ {i = 1} ^ {n} \ell (\mathbf {x} _ {i}, \mathbf {y} _ {i}) + \ell (\mathbf {y} _ {i}, \mathbf {x} _ {i}).\tag{10}
$$

![](images/3b7bc4850ccc589a960d5a2f7a8c69e1b1ac214b07c6329b981e595a088a471f.jpg)  
Fig. 2: Retrieval performance of unsupervised CMH methods through data-efficient learning at 16 and 32 bits. “I2T” denotes image-to-text retrieval and “T2I” denotes text-to-image retrieval.

## 2.5. Binarization and Overall Optimization

Binarization error reduction. Similar to the prior method [1], to mitigate binarization errors caused by continuous relaxation and enhance the representational integrity of our hash functions, we incorporate a binarization loss function defined as:

$$
\ell_ {b} = \sum_ {*} \sum_ {i = 1} ^ {n} \| \mathbf {h} _ {i} ^ {*} - \mathbf {b} _ {i} ^ {*} \| _ {2} ^ {2}.\tag{11}
$$

Overall objective. Finally, by combining $\ell_{p}$ , $\ell_{c}$ , and $\ell_{b}$ , we formulate the overall objective of GNAH as:

$$
\ell = \beta \exp (- \gamma t) \ell_ {p} + (1 - \beta \exp (- \gamma t)) \ell_ {c} + \ell_ {b},\tag{12}
$$

where $0 \leq \beta \leq 1$ is a hyper-parameter to control the strength of the global alignment, $\gamma \geq 0$ determines the exponential decay of PAGA, and t represents the number of epochs. We utilize a curriculum learning approach, where PAGA initially establishes a global structural foundation, then gradually decays to prioritize local refinement via CSNA.

## 3. EXPERIMENTS

## 3.1. Implementation Details

Our method is evaluated using three widely used datasets: MIR Flickr [22], Pascal Sentence [23], and NUS-WIDE [24].

For all datasets, training samples are randomly drawn from the retrieval sets with a fixed random seed 42. To assess the model's performance under varying levels of data scarcity, we conduct evaluations using 20, 40, 80, and 160 training sample pairs per dataset. Following the convention [18], mean average precision (mAP) is adopted for performance evaluation.

We compare our method against five unsupervised CMH methods, CIRH [7], DSAH [25], UCCH [18], CAGAN [10], and CFRH [11]. We also compare our method against two recent supervised CMH methods, DNPH [2] and DSPH [3]. To ensure a fair comparison, all methods use CLIP's image and text encoders as feature extractors. The hyperparameters of GNAH and other methods are tuned based on the dataset or set as default values. The GNAH model is trained using the Adam [26] optimizer for 500 epochs with a batch size of 100. The learning rate adopted for training is 0.0001. Comparisons are conducted at 16 and 32 bits.

## 3.2. Data-Efficient Learning for Cross-Modal Hashing

Comparison with unsupervised CMH methods. As shown in Fig. 2, GNAH consistently outperforms all baselines across datasets. For instance, at 32 bits with 80 training samples, GNAH achieves 0.686 (I2T) and 0.685 (T2I) on MIR Flickr, exceeding the next-best methods by 4.1% and 3.8%. On Pascal Sentence, GNAH reaches 0.480 (I2T) and 0.481 (T2I), outperforming the next-best methods by 10.0% and 8.6%. On NUS-WIDE, GNAH achieves 0.575 (I2T) and 0.582 (T2I), surpassing CAGAN by 2.2% for both tasks.

Table 1: Performance of GNAH and its variants at 32 bits with 80 training sample pairs.

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Full GNAH</td><td colspan="2">w/o  $\ell_p$ </td><td colspan="2">w/o  $\ell_c$ </td><td colspan="2">w/o  $\ell_b$ </td><td colspan="2">GNAH-A</td><td colspan="2">GNAH-B</td><td colspan="2">GNAH-C</td></tr><tr><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td><td>I2T</td><td>T2I</td></tr><tr><td>MIR Flickr</td><td>0.686</td><td>0.685</td><td>0.669</td><td>0.669</td><td>0.576</td><td>0.580</td><td>0.682</td><td>0.681</td><td>0.656</td><td>0.647</td><td>0.681</td><td>0.678</td><td>0.671</td><td>0.677</td></tr><tr><td>Pascal Sentence</td><td>0.480</td><td>0.481</td><td>0.448</td><td>0.449</td><td>0.123</td><td>0.119</td><td>0.468</td><td>0.469</td><td>0.337</td><td>0.369</td><td>0.447</td><td>0.473</td><td>0.453</td><td>0.454</td></tr><tr><td>NUS-WIDE</td><td>0.575</td><td>0.582</td><td>0.571</td><td>0.571</td><td>0.500</td><td>0.475</td><td>0.568</td><td>0.573</td><td>0.564</td><td>0.554</td><td>0.580</td><td>0.574</td><td>0.578</td><td>0.587</td></tr></table>

![](images/6eee846bfd899941164879b251295061e0b6f8a580dc23a85fd53b9a9a86b0a7.jpg)  
Fig. 3: Retrieval performance of GNAH and supervised CMH methods through data-efficient learning at 16 and 32 bits.

Comparison with supervised CMH methods. We further compare GNAH with two supervised CMH methods on Pascal Sentence. As shown in Fig. 3, GNAH consistently outperforms supervised baselines with 20–80 sample pairs despite using no labels. Although supervised methods improve with 160 samples as more labeled data reduces overfitting, our unsupervised approach remains competitive without requiring any manual annotations.

## 3.3. Ablation Study

Effect of $\beta$ and $\gamma$ . To analyze how the global alignment weight $\beta$ and exponential decay factor $\gamma$ affect performance, we conduct a grid search on MIR Flickr. As shown in Fig. 4, the average I2T and T2I mAP at 32 bits (with 80 training pairs) reveals clear interactions between these hyperparameters. When $\beta = 0$ , the model relies solely on CSNA, yielding lower performance and indicating that CSNA alone cannot sufficiently capture dataset-level semantics. When $\gamma = 0$ , PAGA does not decay, which also harms performance, since PAGA is not a perfect modality aligner and should not dominate throughout training. Allowing PAGA to shape the initial global structure, then shifting focus to CSNA for local refinement, yields better results.

![](images/a1da47ca6ff987fc6880d4a971a33474f03509b0da6f9b3f30138f4af8a24115.jpg)  
Fig. 4: Effect of $\beta$ and $\gamma$ at 32 bits with 80 training sample pairs. Results are averaged between I2T and T2I.

Variants of GNAH. To investigate the impact of different components and explore alternative configurations, we construct several variants of GNAH, including: (1) GNAH without $\ell_{p}$ , (2) GNAH without $\ell_{c}$ , (3) GNAH without $\ell_{b}$ , (4) GNAH-A: replacing the proposed contrastive loss with a traditional pairwise contrastive loss for modality alignment, (5) GNAH-B: relaxing the binary prototypical anchors by removing the binarization step, and (6) GNAH-C: generating binary prototypical anchors by directly projecting the latent prototypical anchors through hash functions and subsequently binarizing the relaxed binary prototypical anchors. Table 1 reports the performance of GNAH and its variants at 32 bits with 80 training pairs. The full model consistently outperforms variants (1)–(3), highlighting the importance of each component. Notably, removing $\ell_{c}$ results in a sharp performance drop, particularly on Pascal Sentence (declining from 0.480 to 0.123 in I2T and from 0.481 to 0.119 in T2I). This degradation happens because PAGA focuses mainly on capturing global semantics while overlooking modality alignment and local neighborhood structure.

## 4. CONCLUSION

In this paper, we propose Global-Neighborhood Alignment Hashing (GNAH) for unsupervised data-efficient cross-modal retrieval. By integrating Prototype-Anchored Global Alignment and Contrastive Stochastic Neighborhood Alignment, GNAH effectively transfers semantic structures from foundation models into a compact Hamming space while mitigating overfitting in low-data regimes. Experimental results across multiple benchmarks demonstrate that GNAH outperforms state-of-the-art unsupervised methods and remains competitive with supervised approaches.

## 5. REFERENCES

[1] Qing-Yuan Jiang and Wu-Jun Li, “Deep cross-modal hashing,” in Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 3232–3240.

[2] Qibing Qin, Yadong Huo, Lei Huang, Jiangyan Dai, Huihui Zhang, and Wenfeng Zhang, “Deep neighborhood-preserving hashing with quadratic spherical mutual information for cross-modal retrieval,” IEEE Transactions on Multimedia, vol. 26, pp. 6361–6374, 2024.

[3] Yadong Huo, Qibing Qin, Jiangyan Dai, Lei Wang, Wenfeng Zhang, Lei Huang, and Chengduan Wang, “Deep semantic-aware proxy hashing for multi-label cross-modal retrieval,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 34, no. 1, pp. 576–589, 2024.

[4] Runhao Li, Zhenyu Weng, Huiping Zhuang, Yongming Chen, and Zhiping Lin, “Neighborhood learning from noisy labels for cross-modal retrieval,” in 2023 IEEE International Symposium on Circuits and Systems (ISCAS). IEEE, 2023, pp. 1–5.

[5] Xinyu Xia, Guohua Dong, Fengling Li, Lei Zhu, and Xiaomin Ying, “When clip meets cross-modal hashing retrieval: A new strong baseline,” Information Fusion, vol. 100, pp. 101968, 2023.

[6] Zhang Xi, Xiumei Wang, and Peitao Cheng, “Unsupervised hashing retrieval via efficient correlation distillation,” IEEE Transactions on Circuits and Systems for Video Technology, vol. 33, no. 7, pp. 3529–3541, 2023.

[7] Lei Zhu, Xize Wu, Jingjing Li, Zheng Zhang, Weili Guan, and Heng Tao Shen, “Work together: Correlation-identity reconstruction hashing for unsupervised cross-modal retrieval,” IEEE Transactions on Knowledge and Data Engineering, vol. 35, no. 9, pp. 8838–8851, 2022.

[8] Yaoxin Zhuo, Yikang Li, Jenhao Hsiao, Chiuman Ho, and Baoxin Li, “Clip4hashing: unsupervised deep hashing for cross-modal video-text retrieval,” in Proceedings of the 2022 international conference on multimedia retrieval, 2022, pp. 158–166.

[9] Jiaxing Li, Wai Keung Wong, Lin Jiang, Xiaozhao Fang, Shengli Xie, and Yong Xu, “Ckdh: Clip-based knowledge distillation hashing for cross-modal retrieval,” IEEE Transactions on Circuits and Systems for Video Technology, 2024.

[10] Yewen Li, Mingyuan Ge, Mingyong Li, Tiansong Li, and Sen Xiang, “Clip-based adaptive graph attention network for large-scale unsupervised multi-modal hashing retrieval,” Sensors, vol. 23, no. 7, pp. 3439, 2023.

[11] Li Mingyong, Li Yewen, Ge Mingyuan, and Ma Longfei, "Clip-based fusion-modal reconstructing hashing for large-scale unsupervised cross-modal retrieval," International Journal of Multimedia Information Retrieval, vol. 12, no. 1, pp. 2, 2023.

[12] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PMLR, 2021, pp. 8748–8763.

[13] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu, “Learning to prompt for vision-language models,” International Journal of Computer Vision, vol. 130, no. 9, pp. 2337–2348, 2022.

[14] Runhao Li, Yongming Chen, Zhenyu Weng, Zhiping Lin, and Yap-Peng Tan, “Class-specific prompt learning for vision-language models,” IEEE Transactions on Neural Networks and Learning Systems, 2025.

[15] Jake Snell, Kevin Swersky, and Richard Zemel, “Prototypical networks for few-shot learning,” Advances in neural information processing systems, vol. 30, 2017.

[16] Kaixin Wang, Jun Hao Liew, Yingtian Zou, Daquan Zhou, and Jiashi Feng, “Panet: Few-shot image semantic segmentation with prototype alignment,” in proceedings of the IEEE/CVF international conference on computer vision, 2019, pp. 9197–9206.

[17] Kamalesh Palanisamy, Yu-Wei Chao, Xinya Du, Yu Xiang, et al., “Proto-clip: Vision-language prototypical network for few-shot learning,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS). IEEE, 2024, pp. 2594–2601.

[18] Peng Hu, Hongyuan Zhu, Jie Lin, Dezhong Peng, Yin-Ping Zhao, and Xi Peng, “Unsupervised contrastive cross-modal hashing,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 3, pp. 3877–3889, 2023.

[19] Edy Umargono, Jatmiko Endro Suseno, and SK Vincensius Gunawan, “K-means clustering optimization using the elbow method and early centroid determination based on mean and median formula,” in The 2nd international seminar on science and technology (ISSTEC 2019). Atlantis Press, 2020, pp. 121–129.

[20] Zhirong Wu, Yuanjun Xiong, Stella X Yu, and Dahua Lin, "Unsupervised feature learning via non-parametric instance discrimination," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2018, pp. 3733-3742.

[21] Fumin Shen, Chunhua Shen, Wei Liu, and Heng Tao Shen, "Supervised discrete hashing," in Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 37-45.

[22] Mark J Huiskes and Michael S Lew, “The mir flickr retrieval evaluation,” in Proceedings of the 1st ACM international conference on Multimedia information retrieval, 2008, pp. 39–43.

[23] Cyrus Rashtchian, Peter Young, Micah Hodosh, and Julia Hockenmaier, “Collecting image annotations using amazon’s mechanical turk,” in Proceedings of the NAACL HLT 2010 workshop on creating speech and language data with Amazon’s Mechanical Turk, 2010, pp. 139–147.

[24] Tat-Seng Chua, Jinhui Tang, Richang Hong, Haojie Li, Zhiping Luo, and Yantao Zheng, “Nus-wide: a real-world web image database from national university of singapore,” in Proceedings of the ACM international conference on image and video retrieval, 2009, pp. 1–9.

[25] Dejie Yang, Dayan Wu, Wanqian Zhang, Haisu Zhang, Bo Li, and Weiping Wang, “Deep semantic-alignment hashing for unsupervised cross-modal retrieval,” in Proceedings of the 2020 international conference on multimedia retrieval, 2020, pp. 44–52.

[26] Diederik P Kingma and Jimmy Ba, “Adam: A method for stochastic optimization,” arXiv preprint arXiv:1412.6980, 2014.