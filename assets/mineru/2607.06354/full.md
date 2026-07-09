# Generalized Synthetic Image Detection with Enhanced RGB-Noise Representation Learning

Zhen Li<sup>a</sup>, Gang Cao<sup>a,b,∗</sup>, Tian Zhang<sup>a</sup>, Lifang Yu<sup>c</sup> and Shaowei Weng<sup>d</sup>

<sup>a</sup>School of Computer and Cyber Sciences, Communication University of China, Beijing, 100024, China

<sup>b</sup>School of Information Engineering, Changsha Medical University, Changsha, 410219, China

<sup>c</sup>Department of Information Engineering, Beijing Institute of Graphic Communication, Beijing, 100026, China

<sup>d</sup>Fujian Provincial Key Laboratory of Big Data Mining and Applications, Fujian University of Technology, Fuzhou, 350118, China

## A R T I C L E I N F O

Keywords: Image forensics AI-generated image Synthetic image detection Contrastive learning Feature fusion

## A B S T R A C T

The rapid advancement of large-scale generative models has accelerated the spread of highly deceptive AI-generated images, making generalized synthetic image detection a critical im perative. Existing forensic networks often struggle with cross-model generalization and real world degradations due to their reliance on single-domain representations and conventional binary classification optimization. To overcome these limitations, we propose RNSIDNet, a novel forensic framework that achieves robust detection through enhanced RGB-Noise repre sentation learning. Specifically, our method employs a dual-branch architecture where global RGB semantics, extracted by an attention-refined CLIP backbone, dynamically modulate high frequency noise artifacts captured by Bayar convolutions via a Feature-wise Linear Modulation (FiLM) module. To further enhance the learned representations, we design a Hard Sample-aware Contrastive Learning (HSCL) strategy. By explicitly penalizing challenging training samples, HSCL reshapes the latent feature space to maximize the discriminative margin between pristine and synthetic domains. Extensive experiments across eight public benchmark datasets verify that our model achieves state-of-the-art performance, delivering superior generalization ability, robustness, and computational eficiency. Code and dataset will be publicly available on https: //github.com/multimediaFor/RNSIDNet.

## 1. Introduction

The rapid evolution of deep learning techniques, especially large-scale generative models [13, 17, 37, 33], has significantly advanced image synthesis while simultaneously facilitating the spread of deceptive AIGC content and deepfakes. In response, researchers have developed various forensic methods to identify these forgeries within a learning-based framework [44, 29, 2, 26].

While numerous detection methods have been proposed, they frequently encounter a dual bottleneck in feature representation and model optimization when facing unseen generative models or complex real-world degradations, limiting their real-world applicability. A primary limitation of existing detectors is their reliance on single-domain representation. Spatial models are highly prone to overfitting dataset-specific semantics, whereas frequency-based approaches degrade sharply under common image corruptions like JPEG compression or Gaussian blur [44, 11]. To<sup>.</sup> capture comprehensive forensic traces, recent studies [47, 41] have attempted to fuse diverse feature representations. Unfortunately, most existing methods rely on naive fusion strategies, such as direct concatenation or element-wise addition. These rigid operations fail to capture the complex contextual dependencies between diferent modalities. Consequently, such inefective integration ultimately forces heterogeneous features to interfere with each other rather than achieve synergy, where dominant semantic information may obscure subtle frequency-domain artifacts.

Beyond representation, traditional optimization objectives further constrain detection capabilities. Most detectors [44, 29, 46] frame synthetic image detection as a standard binary classification task supervised by a Binary Cross-Entropy (BCE) loss. However, it creates a vulnerable decision boundary that is easily bypassed by the highquality outputs of modern AIGC models. Previous works [7, 21] adopt contrastive learning to improve the feature representation, but they treat all sample pairs uniformly. For deepfake detection, most mismatched pairs which are visually distinct from the anchor sample can be easily pushed away, while the most discriminative information comes from a small set of hard samples that lie extremely close to the anchor in feature space. In this way, the contribution of these critical samples is diluted by the overwhelming number of easy ones.

To tackle these challenges, we propose an RGB-Noise dual-branch Synthetic Image Detection Network (RNSID Net). It aims to construct a robust framework for generalized synthetic image detection, utilizing enhanced RGB-Noise representation learning to deeply synergize heterogeneous features and reshape the representation space. At the architectural level, we design a gating-based dynamic feature fusion mechanism. Instead of conventional direct concatenation, it innovatively utilizes the RGB features as conditioning variables to dynamically generate scale and bias parameters via Feature-wise Linear Modulation (FiLM) [32], which subsequently modulate the noise branch features. Furthermore, we introduce a Hard Sample-aware Contrastive Learning (HSCL) strategy. It achieves dynamic re-weighting within a compact low-dimensional manifold, actively tracking and assigning larger gradient penalties to highly deceptive images. This spatial optimization efectively strips away gradient interference from simple samples, significantly enhancing intra-class clustering and inter-class repulsion, thereby successfully widening and reshaping the decision boundary against extremely photorealistic forgeries. Finally, building upon existing research, we construct and release a large-scale Aligned Multi-source Synthetic Image Dataset (AMSID). By enforcing strict pixel alignment between pristine and synthetic sample pairs across diverse generators, AMSID enables the network to suppress content driven distribution biases and focus on intrinsic generative artifacts.

Through such specialized designs, RNSIDNet achieves high generalization capabilities across various generative models and diverse content types. Notably, even when trained on limited data and with a mere fraction of the trainable parameters, RNSIDNet yields performance comparable to massive models trained on substantially larger datasets, demonstrating exceptional eficiency and robustness.

In summary, the main contributions of this paper are as follows:

• We propose RNSIDNet, an RGB-noise synthetic image detection network, by dynamically modulating noise features with RGB modal, achieving an efective fusion of heterogeneous representations.

• We propose a representation enhancement strategy guided by a Hard Sample-aware Contrastive Loss (HSCL). It forces the feature space to achieve higher inter-class separability and intra-class compactness by amplifying the repulsion against hard cross-class samples.

• Extensive experiments on eight public datasets demonstrate that the proposed method achieves the state-of-theart detection performance, exhibiting outstanding generalization capabilities in cross-model forgery detection.

The rest of this paper is organized as follows. Section II reviews previous related works. The proposed RNSIDNet network and its training are elaborated in Section III. The experimental results and discussion are depicted in Section IV, followed by the conclusion drawn in Section V.

## 2. Related Works

## 2.1. Single-Domain Feature

Most end-to-end synthetic image detection methods focus on capturing visual artifacts or statistical anomalies within a single modality of the feature domain. In the spatial domain, researchers typically utilize CNNs to automatically learn generalizable pixel-level discriminative features. For instance, Wang et al. [44] systematically verifies the existence of cross-GAN fingerprints. With the rise of large vision-language models, the powerful representation capability of pre-trained CLIP [34] models is used to enhance cross-model detection generalization [29, 9]. The subsequent RINE method [211 leverages the intermediate features of CLIP encoder blocks. In the frequency domain spectral analysis provides another perspective for image forensics. Early works [50, 11] rely on basic transforms (DFT, DCT) to expose spectrum replication. Recent methods capture subtle generative artifacts more eficiently through diverse schemes, like frequency-aware networks [41], and hybrid approaches that extract noise residuals via a pretrained denoising model or specialized filter to amplify high-frequency anomalies [24, 27, 51, 16].

![](images/bdd42d47fa3e1a6080755a00d1087763294aa2dc14c9f4b53760deff0f53f1a6.jpg)  
Figure 1: Overview of the proposed RNSIDNet framework. The input image is processed through two parallel branches: an RGB branch that extracts multi-scale representations using a frozen CLIP-ViT encoder and a Balanced Attention Module (BAM), and a noise branch that captures high-frequency residuals via Bayar convolution followed by Global Average Pooling (GAP). The heterogeneous features are subsequently integrated using a dynamic Feature-wise Linear Modulation (FiLM) module guided by the RGB features. Finally, the network is jointly optimized by a classification loss $\left( \boldsymbol { L } _ { c l s } \right)$ and a contrastive loss $\left( L _ { c o n t r a } \right)$ driven by the Hard Sample-aware Contrastive Learning (HSCL) strategy, explicitly enforcing intra-class compactness and inter-class separation.

## 2.2. Multi-Dimensional Feature Fusion

To mitigate the inherent vulnerabilities and domain-specific overfitting of single-modality representations, multidimensional fusion frameworks have emerged as a promising paradigm. By concurrently capturing high-level semantics and low-level statistical artifacts, these frameworks significantly bolster detection resilience and generalization. In practice, some recent methodologies [18] leverage attention mechanisms to dynamically integrate global semantics with fine-grained local patches, while others exploit cross-modal alignments between textual prompts and visual content [38, 9]. Furthermore, advanced approaches synergize multi-domain cues—such as frequency spectra and loca textures—with the robust representational priors of vision-language models [47, 25].

## 2.3. Training Paradigm and Optimization Strategy

Recently, image reconstruction has emerged as a powerful proxy for exposing generative artifacts by measuring pixel-level reconstruction discrepancies. To mitigate the "generalization illusion" caused by overfitting to datasetspecific biases [15], recent studies align real and synthetic data distributions via autoencoders or self-conditioned reconstruction [35, 46, 8]. Moreover, to overcome the rigid decision boundaries of traditional BCE loss, contrastive learning is increasingly integrated to directly enforce intra-class compactness and inter-class separability. Recent methods tailor this contrastive objective to specific feature levels, for instance, Chen et al. [7] construct contrastive pairs by widening low-level reconstruction trajectories, whereas Tan et al. [40] apply contrastive optimization within the high-level semantic space using prompt-guided CLIP.

## 3. Proposed scheme

In this section, the architecture design, training dataset generation and training strategy of our proposed synthetic image detection network, i.e., RNSIDNet, are presented in detail.

## 3.1. Network architecture

## 3.1.1. Overview

The overall architecture of the proposed RNSIDNet is illustrated in Figure 1. It employs a collaborative dual-branch structure introducing an RGB branch and a noise branch. The blind detection of AI-generated images is achieved by fusing high-level spatial representations with low-level noise residual signals.

Given an input image $\mathbf { \bar { \boldsymbol { X } } } \in \mathbb { R } ^ { H \times W \times 3 }$ , the RGB branch extracts the multi-scale spatial feature ?? from a frozen CLIP-ViT [34] encoder, which are further refined by a Balanced Attention Module (BAM) [31] to yield the RGB representation $Z _ { r }$ . Concurrently, the Noise Branch extracts high-frequency residual signals via a Bayar-constrained filter, followed by a lightweight CNN to produce the noise representation $Z _ { n }$ . During the feature fusion stage, we design a dual-branch fusion mechanism based on Feature-wise Linear Modulation (FiLM) [32]. This module leverages the extracted RGB image feature $Z _ { r }$ as conditioning guidance to dynamically weight and filter the noise feature $Z _ { n } .$ Finally, $Z _ { f }$ is passed through a classification head to output the forgery probability of the image, thereby completing the end-to-end detection task.

## 3.1.2. Attention-Refined Spatial Feature Extraction

In the RGB branch, the input image ?? is fed into the frozen CLIP-ViT encoder. The layer-normalized output feature $F _ { i } \in \mathbb { R } ^ { D }$ is extracted from the ??-th Transformer [43] block. Thereafter, these intermediate features are stacked along the layer dimension to construct a cross-layer feature representation $\boldsymbol { M } \in \mathbb { R } ^ { N \times D }$ , where ?? denotes the number of extracted blocks. Such a representation successfully captures hierarchical semantic information across various network depths. Given that diferent layers contribute unequally to forgery detection, an adaptive re-weighting mechanism is designed to highlight salient cues.

To this end, a BAM module is employed to enhance the feature representation ??. Specifically, the channel attention branch compresses the layer dimension to aggregate global context. The intermediate representation is processed by a shared Multi-Layer Perceptron (MLP) to derive the channel-wise weight vector $w _ { c } \in \bar { \mathbb { R } ^ { 1 \times D } }$

$$
w _ {c} = \sigma (\mathrm{MLP(AvgPool} (M)) + \mathrm{MLP(MaxPool} (M)))\tag{1}
$$

where $\sigma ( \cdot )$ denotes the sigmoid activation function, and pooling operations are performed along the layer dimension ??. Concurrently, the layer-wise attention branch captures the scaling factors across diferent network depths. It aggregates the feature dimension via mean and max pooling, and applies a 1D convolution to generate the layer-wise weight vector $w _ { l } \in \mathbb { R } ^ { N \times 1 }$

$$
w _ {l} = \sigma (\operatorname{Conv1d} ([ \operatorname{MeanPool} (M); \operatorname{MaxPool} (M) ]))\tag{2}
$$

where [⋅; ⋅] represents the concatenation along the feature dimension. Subsequently, $w _ { c }$ and $w _ { l }$ are broadcasted to $\mathbb { R } ^ { N \times D }$ by expanding along the layer and feature dimensions, respectively. They are then summed and activated by another sigmoid function to construct the joint attention map. Through a residual connection, the intermediate refined feature representation ??<sup>̃</sup> is obtained:

$$
\tilde {M} = M + M \odot \sigma (w _ {c} + w _ {l})\tag{3}
$$

where ⊙ represents element-wise multiplication. Ultimately, to preserve the stability of the original semantic structure during training, a learnable gating parameter $\gamma$ is introduced to dynamically blend the original and intermediate features, yielding the final representation $M ^ { \prime }$ :

$$
M ^ {\prime} = (1 - \sigma (\gamma)) M + \sigma (\gamma) \tilde {M}.\tag{4}
$$

Following this gating mechanism, $M ^ { \prime }$ is transformed into a ??-dimensional vector by a projection layer and processed by a hybrid pooling layer. The final RGB feature vector $Z _ { r } \in \mathbb { R } ^ { d }$ is computed as the mean of max and average-pooled representations.

## 3.1.3. Constrained High-Frequency Noise Residual Learning

In the noise branch, a constrained convolutional layer is utilized to extract high-frequency residual from the input image ??. Existing methods typically rely on standard CNNs or Spatial Rich Model (SRM) [12] filters, both of which have notable limits. Driven by the inherent optimization bias to minimize loss efortlessly, standard CNNs tend to greedily learn dominant low-frequency semantics, often degenerating into mere "content extractors" and causing severe feature homogenization with the RGB branch. Conversely, the hand-crafted SRM filter employs fixed kernels, lacking the dynamic adaptability required to capture the novel and evolving microscopic generative artifacts incurred by modern AIGC models.

To overcome these bottlenecks, we introduce Bayar convolution [4], which restricts the parameter space through structural constraints, guaranteeing that the filter maintains stable high-pass characteristics. Let the convolutional kernel size be $K \times K$ , with its center coordinate denoted as ??. This filter updates the gradients only for the $K ^ { 2 } - 1$ non-center parameters, defined as $\theta = \{ w _ { i } \mid i \neq c \}$ . During the forward propagation phase, the network first normalizes the non-center parameters, forcing them to satisfy $\textstyle \sum _ { i \neq c } w _ { i } = 1$ . Subsequently, a fixed weight $w _ { c } = - 1$ is inserted at the center of the kernel, constructing the complete convolutional kernel $W = \{ w _ { 1 } , \ldots , w _ { c - 1 } , - 1 , w _ { c + 1 } , \ldots , w _ { K ^ { 2 } } \}$ . Since the sum of the non-center weights is 1 and the center weight is fixed at -1, the convolutional kernel strictly satisfies a zero-sum constraint, i.e., $\textstyle \sum _ { i = 1 } ^ { K ^ { 2 } } w _ { i } = 0$ . This zero-sum architecture renders the convolutional kernel as a learnable high-pass filter. It captures the diference between the current pixel and the weighted average of its local neighbors. Consequently, the low-frequency semantic structural information of the image is efectively suppressed, while the local statistical anomaly and artifact residual are amplified. When dealing with multi-channel inputs, this constrained convolution operates independently on each input channel and is subsequently recombined along the output channel dimension, realizing cross-channel residual signal modeling. Let $B ( \cdot )$ be the constrained filtering operation defined by $W$ , then the extracted residual response map is yielded as

$$
R = \mathcal {B} (X).\tag{5}
$$

The resulting residual map ?? contains high-frequency information, such as interpolation artifacts, compression traces, and the local textural anomaly unique to generative models. Subsequently, ?? is fed into a lightweight convolutional encoding network $f _ { n } ( \cdot )$ , which consists of five stacked convolutional blocks. Each block sequentially applies a $3 \times 3$ convolution with stride 2, batch normalization, and a ReLU activation function, while doubling the channel dimension at each stage. Finally, a flatten operation compresses the spatial dimensions to yield the fixed-length noise feature representation $Z _ { n } \in \mathbb { R } ^ { d }$ as

$$
Z _ {n} = f _ {n} (R)\tag{6}
$$

where ?? is the final dimension of the flattened feature. This process gradually expands the receptive field while achieving a global statistical aggregation of local residual signals, thereby distilling highly discriminative high frequency pattern features.

## 3.1.4. Dynamic Feature Fusion Mechanism

Upon obtaining the RGB feature vector $Z _ { r } \in \mathbb { R } ^ { d }$ and the noise feature vector $Z _ { n } \in \mathbb { R } ^ { d }$ , the appropriate fusion of such heterogeneous representations also afects the ultimate discriminative eficacy of the model. To this end, we design a dynamic feature modulation mechanism guided by spatial features to achieve the deep fusion of the dual-branch representations.

As illustrated in Figure 1, we adopt a modulation strategy based on FiLM. Using $Z _ { r }$ as a priori input, a Multi-Layer Perceptron (MLP) dynamically generates modulation parameters. The MLP consists of two linear projection layers with channel dimensions expanding as $d  2 d  3 d$ . It generates a scaling coeficient $\gamma ,$ , a shift coeficient $\beta ,$ and a gating vector ??, which can be formulated as

$$
[ \gamma , \beta , g ] = f _ {f i l m} (Z _ {r})\tag{7}
$$

where $f _ { f i l m } ( \cdot )$ is the mapping that includes two linear transformations and a ReLU non-linear activation function. A Sigmoid function is applied to ?? to ensure that the values of the gating weights fall within a reasonable and stable range. Subsequently, the generated afine parameters $( \gamma , \beta )$ and the gating vector (??) are utilized to perform feature-wise linear modulation and secondary filtering on the noise feature $Z _ { n }$ :

$$
\hat {Z} _ {n} = g \odot (\gamma \odot Z _ {n} + \beta)
$$

(8)

where ⊙ denotes element-wise multiplication. In this integrated operation, the afine parameters ?? and ?? perform precise recalibration and shifting of the low-level noise features driven by the conditional guidance, enabling the noise representation to adaptively align with the structural and content distribution of the current image. Concurrently, the gating vector ?? acts as an information valve along the feature dimension to further suppress irrelevant noise interference. By dynamically controlling the transmission ratio of the modulated features, it prevents the excessive amplification of anomalous statistical artifacts. Ultimately, the joint cross-branch representation is constructed via a residual connection:

$$
Z _ {f} = Z _ {r} + \hat {Z} _ {n}\tag{9}
$$

Consequently, the final joint representation $Z _ { f } \in \mathbb R ^ { d }$ accurately captures and amplifies intrinsic generative artifacts, providing a purer and more robust discriminative basis. To map these fused features into the final decision space, $Z _ { f }$ is first processed by a sequence of dense layers, yielding a highly compact and abstract representation $Z _ { f } ^ { \prime }$ . Ultimately, $Z _ { f } ^ { \prime }$ is fed into the classification head, which outputs the forgery probability via a Sigmoid activation function.

## 3.1.5. Hard Sample-aware Contrastive Learning

The conventional training paradigm typically relies on cross-entropy loss for global supervision. However, it overlooks the hard-to-distinguish samples near the decision boundary, thereby limiting the model’s generalization capability against highly deceptive synthetic images. To attenuate such deficiency, we formulate a joint loss function comprising two components: a binary classification loss for branch supervision and a contrastive loss for constraining topological structure of the feature space. The overall objective function $\mathcal { L } _ { t o t a l }$ is defined as

$$
\mathcal {L} _ {t o t a l} = \mathcal {L} _ {c l s} + \lambda \mathcal {L} _ {c o n t r a}\tag{10}
$$

where $\mathcal { L } _ { c l s }$ is the classification loss, $\mathcal { L } _ { c o n t r a }$ is the contrastive loss, and ?? is the weighting coeficient.

Following the protocol of supervised contrastive learning [20], for each anchor, samples with the same label form positive pairs, while those with diferent labels form negative pairs. During training, the gradient is dominated by easily distinguishable pairs, which causes the network to quickly hit an optimization bottleneck. The most critical "hard negatives"–samples that carry a diferent label yet lie perilously close to the anchor in the feature space–are virtually ignored. This leaves the model blind to the subtle, discriminative cues that separate authentic and generated images [36]. To address this, we propose a Hard Sample-aware Contrastive Loss (HSCL) based on dynamic re-weighting. It adjusts the penalty adaptively for hard negative samples, guiding the network to focus its optimization on the most confusable boundary features.

The computation of HSCL proceeds as follows. Let ?? be the total number of samples in the current mini-batch. For each sample ??, the final embedding vector $z _ { i } \in \mathbb { R } ^ { d }$ is first $L _ { 2 } \cdot$ -normalized to ensure that similarities are measured strictly by angular distance. Subsequently, the cosine similarity matrix ?? between all pairs of samples within the batch is computed and scaled by a temperature parameter ?? to control the smoothness of the distribution:

$$
s _ {i, j} = \frac {z _ {i} \cdot z _ {j} ^ {T}}{\tau}.\tag{11}
$$

Specifically, for each anchor $i , N ( i )$ denotes the set of its negative samples. To counteract the dilution caused by easy negatives, we dynamically select the top-?? negatives in $N ( i )$ with the highest similarity $s _ { i j }$ and assign them an extra penalty. Denoting the exponentiated similarity as $\exp ( { s _ { i j } } )$ , the re-weighted negative term is defined as

$$
\tilde {E} _ {i j} ^ {-} = \exp (s _ {i j}) \cdot \left(1 + \alpha H _ {i j}\right)\tag{12}
$$

where $H _ { i j } \in \{ 0 , 1 \}$ indicates whether sample ?? is among the selected hard negatives, and ?? controls the penalty intensity. For positive pairs, we simply set $E _ { i j } ^ { + } = \exp ( s _ { i j } )$ when ?? and ?? share the same label (and 0 otherwise). The proposed HSCL is then defined as

$$
\mathcal {L} _ {\text {contra}} = - \frac {1}{B} \sum_ {i = 1} ^ {B} \log \left(\frac {\sum_ {j = 1} ^ {B} E _ {i j} ^ {+}}{\sum_ {j = 1} ^ {B} E _ {i j} ^ {+} + \sum_ {j = 1} ^ {B} \tilde {E} _ {i j} ^ {-} + \epsilon}\right)\tag{13}
$$

where ?? is a small constant for numerical stability. By amplifying the repulsion on those cross-class neighbors that lie nearest to the anchor, HSCL forces the feature boundaries to be sharpest in the most confusable regions. This yields a highly discriminative feature space and acts as an efective complement to the classification loss, significantly improving generalization ability to unseen generative models.

For the binary classification task, we employ the Binary Cross-Entropy (BCE) loss to measure the discrepancy between the predicted probabilities and the ground-truth labels. Assuming that for sample ??, the model’s predicted output is $p _ { i } \in [ 0 , 1 ]$ and the ground-truth label is $y _ { i } \in \{ 0 , 1 \}$ , the classification loss $\mathcal { L } _ { c l s }$ is defined as

$$
\mathcal {L} _ {c l s} = - \frac {1}{B} \sum_ {i = 1} ^ {B} [ y _ {i} \cdot \log (p _ {i}) + (1 - y _ {i}) \cdot \log (1 - p _ {i}) ]
$$

(14)

Throughout the training process, the HSCL objective is responsible for structuring a well-clustered, highly separable feature space, while the BCE loss is tasked with delineating the optimal classification decision boundary within this optimized manifold.

Real image  
Zoom-in view  
Stable Diffusion output  
GAN output  
![](images/f934c46069652ac057013d45019e7a7f41e2a5f4666d65e681875105b05b467d.jpg)  
Figure 2: Comparison of local details between synthetic and pristine images. From left to right: original image, zoomed-in local patch, SD XL [33] generated image, and Real-ESRGAN [45] generated image.

## 3.2. Construction of Aligned Multi-source Synthetic Image Dataset (AMSID)

Existing deepfake detection datasets often sufer from limited content diversity and outdated generative models, which cause models to overfit to dataset-specific biases rather than learning generalizable forgery artifacts. As observed by Cozzolino et al. [9], incorporating diverse real-image sources is crucial to bridging the generalization gap. To this end, we construct the Aligned Multi-source Synthetic Image Dataset (AMSID), a large-scale benchmark of approximately 235,000 images designed to force the model to capture semantic-independent generation traces. By pairing diverse real images with their synthetic counterparts produced through modern generators, we explicitly eliminate content-driven distribution gaps and strengthen generalization.

AMSID collects pristine source images from six open-source datasets spanning a wide range of scenes, resolutions, and semantic domains: MS-COCO [23], MMP [6], DIV2K [1], BDD100K [48], RAISE [10], and Flickr2K [22]. To ensure that the network encounters a rich variety of generative fingerprints, we reconstruct these real images using two representative families of generators: Difusion Models (DMs) and Generative Adversarial Networks (GANs). Specifically, Stable Difusion XL (SD XL) [33] is employed to regenerate images from the MS-COCO, MMP, and BDD100K subsets, injecting difusion-specific artifacts. Real-ESRGAN [45] processes the DIV2K, RAISE, and Flickr2K subsets, imprinting GAN-related high-frequency texture biases. All synthetic images are generated strictly from their source counterparts, guaranteeing pixel-level alignment and identical semantic content. A visual comparison is provided in Figure 2.

Table 1  
Detailed information of our training datasets. <sup>∗</sup> means the typical spatial resolution of images in the dataset.

<table><tr><td>Dataset</td><td>Generator</td><td>Source</td><td>Real / Fake</td><td>Resolution</td><td>Format</td></tr><tr><td>Bias-Free [15]</td><td>SD 2.1 [37]</td><td>MS-COCO [23]</td><td>51,515 / 51,515</td><td> $512 \times 512$ </td><td>PNG</td></tr><tr><td rowspan="6">AMSID</td><td rowspan="3">SD XL [33]</td><td>MS-COCO [23]</td><td>27,935 / 27,935</td><td> $480 \times 640^{*}$ </td><td>JPG / PNG</td></tr><tr><td>MMP [6]</td><td>2,000 / 2,000</td><td> $768 \times 1024^{*}$ </td><td>JPG / PNG</td></tr><tr><td>BDD100K [48]</td><td>10,000 / 10,000</td><td> $720 \times 1280$ </td><td>JPG / PNG</td></tr><tr><td rowspan="3">Real-ESRGAN [45]</td><td>Flickr2K [22]</td><td>2,650 / 2,650</td><td> $1356 \times 2040^{*}$ </td><td>PNG</td></tr><tr><td>DIV2K [1]</td><td>900 / 900</td><td> $1356 \times 2040^{*}$ </td><td>PNG</td></tr><tr><td>RAISE [10]</td><td>5,000 / 5,000</td><td> $3264 \times 4928^{*}$ </td><td>TIF / PNG</td></tr></table>

Table 2  
Summary of eight test datasets.

<table><tr><td>Datasets</td><td>Real/Fake</td><td>Source of Real</td><td>Generator Type</td><td>Generators</td></tr><tr><td>Synthbuster [3]</td><td>1K / 9K</td><td>RAISE</td><td>DM</td><td>9</td></tr><tr><td>AIGCDetectionBenchmark [52]</td><td>74.3K / 74.3K</td><td>LSUN &amp; ImageNet, etc.</td><td>DM &amp; GAN</td><td>16</td></tr><tr><td>UniversalFakeDetect [29]</td><td>52K / 52K</td><td>LAION &amp; ImageNet, etc.</td><td>DM &amp; GAN</td><td>18</td></tr><tr><td>GenImage [53]</td><td>50K / 50K</td><td>ImageNet</td><td>DM &amp; GAN</td><td>8</td></tr><tr><td>DDA-COCO [8]</td><td>5K / 25K</td><td>MSCOCO</td><td>DM</td><td>5</td></tr><tr><td>DIF [39]</td><td>37.7K / 37.7K</td><td>LAION</td><td>DM &amp; GAN</td><td>13</td></tr><tr><td>Chameleon [47]</td><td>14.9K / 11.2K</td><td>Internet</td><td>Unknown</td><td>Unknown</td></tr><tr><td>WildRF [5]</td><td>1.1K / 1.2K</td><td>Reddit, FB, X</td><td>Unknown</td><td>Unknown</td></tr></table>

The generation pipeline is carefully controlled to preserve structural fidelity while maximizing the injection of model-intrinsic fingerprints. For the difusion-based branch, we integrate the ControlNet Tile module [49] into the SD XL img2img pipeline to spatially confine local redrawing and prevent semantic drift. A DPM++ 2M sampler [28] with 30 steps and Karras scheduling [19] is adopted for eficient high-quality synthesis, and the denoising strength is set to 0.25 to balance artifact visibility and global consistency. For the GAN-based branch, Real-ESRGAN’s highorder degradation modeling is leveraged in an end-to-end manner; it performs blind restoration at original resolutions, using its adversarial prior to re-infer and sharpen textures while naturally embedding the discriminator’s generative fingerprints. Both tracks produce synthetic images that are highly realistic and spatially aligned with the originals, yielding a clean, diverse training set explicitly built for learning generic forgery representations.

## 4. Experiments

## 4.1. Experimental Settings

## 4.1.1. Datasets

Training Datasets. The training data configuration of our model is detailed in Table 1. About 200,000 images are collected from our created AMSID dataset and the self-conditioned subset of the public Bias-Free [15] benchmark, which reconstructs MS-COCO [23] images at pixel-level via Stable Difusion 2.1 [37]. During the training phase, a strict paired alternating sampling is enforced to keep equal real and fake samples within each batch. Specifically, a pristine source image and its reconstructed synthetic counterpart are fed into the network in a mandated alternating sequence (i.e., ${ R e a l } _ { A } , { F a k e } _ { A } , { R e a l } _ { B } , { F a k e } _ { B } , \ldots )$ . This strictly paired formulation not only eliminates the distribution bias caused by varying semantic contents, but also provides sample pairs for contrastive learning. As such, the network could focus purely on capturing the microscopic generative artifacts.

Data Augmentation. To enhance the robustness against real-world degradations and prevent overfitting to superficial artifacts, we employ a probability-driven data augmentation pipeline comprising four sequential operations:

1) Center Cropping: Input images are center-cropped to 224 × 224 pixels to standardize dimensions and ensure semantic alignment.

2) Geometric Transformation: Applied with a probability of 0.5, randomly selecting a mutually exclusive operation (horizontal/vertical flip, 90<sup>◦</sup> rotation, or transposition) to foster spatial invariance.

3) Image Compression: JPEG compression $\left( \mathrm { Q F } \in \left[ 3 0 , 1 0 0 \right] \right)$ is applied with a probability of 0.5. We randomly alternate between OpenCV and PIL encoders to emulate diverse cross-platform transmission artifacts.

4) Gaussian Blurring: Gaussian blur $( \sigma \in [ 0 . 0 , 3 . 0 ] )$ is applied with a probability of 0.5 to obscure fragile highfrequency noise, compelling the network to capture robust, degradation-invariant structural representations.

Testing Datasets. To comprehensively evaluate the cross-domain generalization of our model, we employ 8 public benchmark datasets, i.e., UniversalFakeDetect [29], AIGCDetectionBenchmark [52], Synthbuster [3], GenImage [53], DIF [39], DDA-COCO [8], Chameleon [47] and WildRF [5]. The detailed compositions of these datasets are summarized in Table 2. The final test set contains approximately 500,000 images, spanning Difusion Models, GANs, and unknown commercial generators.

## 4.1.2. Baseline Methods

Our proposed method is compared with 8 representative state-of-the-art baselines across diverse detection paradigms, which include the gradient-based [42], reconstruction-based [46, 7], semantic-based [29, 40, 9], and multidomain fusion-based [47] approaches. These methods are briefly introduced as follows:

• LGrad [42] (CVPR’2023) transforms input images into gradient maps via a pre-trained CNN to explicitly isolate underlying generative artifacts from high-level semantic content.

• DIRE [46] (ICCV’2023) utilizes the discrepancy between the original image and its reconstructed counterpart produced by a pre-trained difusion model as a discriminative metric.

• UFD [29] (CVPR’2023) leverages the robust and generalized representation space of a pre-trained CLIP model, combined with a K-Nearest Neighbors (KNN) classifier, to achieve cross-model detection.

• Cozzo2024 [9] (CVPR’2024) constructs semantic-aligned pairs of real/generated images and performs similarity discrimination in the CLIP feature space to achieve efective few-shot forgery detection.

• RINE [21] (ECCV’2024) extracts multi-level representations from the intermediate Transformer blocks of the CLIP encoder to construct a highly discriminative forgery-aware feature space.

• DRCT [7] (ICML’2024) applies contrastive learning on both the original images and their difusion-reconstructed trajectories to explicitly widen the representation margin between the pristine and synthetic domains.

• AIDE [47] (ICLR’2025) constructs a hybrid dual-branch detector by dynamically fusing global semantic features extracted by CLIP with local spatial representations derived from DCT coeficients

• C2P-CLIP [40] (AAAI’2025) injects category-common prompts into the CLIP architecture and fine-tunes the model via contrastive learning to strengthen the real-vs-fake decision boundary.

## 4.1.3. Implementation Details

The proposed RNSIDNet is implemented using PyTorch and trained on a single NVIDIA RTX 3090 GPU. The specific data augmentation and pre-processing strategies for generating the 224 × 224 image inputs strictly follow the protocols detailed in Section 4.1.1. In the dual-branch architecture, the RGB branch employs a frozen CLIP ViT-L/14 backbone to extract global semantics, and the noise branch utilizes 5 × 5 Bayar convolutional kernels to capture highfrequency artifacts. Prior to dynamic fusion, the extracted representations from both branches are mapped and aligned into a unified $d = 5 1 2$ dimensional space.

The network is trained for 1 epochs using the standard Adam optimizer with a batch size of 64. The learning rate is initialized at $1 \times 1 0 ^ { - 4 }$ . Crucially, to implement our HSCL strategy, the balancing coeficient ?? for the joint objective is empirically set to 0.8, and the contrastive temperature ?? is set to 0.07. To efectively penalize highly deceptive samples, the top 10% of negative samples within each batch are dynamically mined as "hard negatives" and assigned an explicit penalty weight of 2.0.

Table 4  
Performance comparison of various models across all evaluation benchmarks. Results are presented in the format of ACC (%) / AUC (%). Due to table width constraints, AIGCDetectionBenchmark [52] and UniversalFakeDetect [29] are abbreviated as AIGCDetect and UFDetect, respectively. The best and second-best results are highlighted in bold and underlined.

<table><tr><td>Method</td><td>GenImage</td><td>Synthbuster</td><td>AIGCDetect</td><td>UFDetect</td><td>DDA-COCO</td><td>DIF</td><td>Chameleon</td><td>WildRF</td><td>AVG</td></tr><tr><td>LGrad [42]</td><td>51.94/57.04</td><td>44.33/39.16</td><td>50.19/53.81</td><td>36.51/35.94</td><td>50.19/52.15</td><td>49.91/53.88</td><td>48.49/47.43</td><td>47.79/60.67</td><td>47.42/50.01</td></tr><tr><td>DIRE [46]</td><td>56.05/61.83</td><td>47.80/46.88</td><td>53.74/56.33</td><td>49.98/48.61</td><td>51.41/52.10</td><td>52.01/53.46</td><td>46.66/51.79</td><td>53.73/54.26</td><td>51.42/53.16</td></tr><tr><td>UFD [29]</td><td>69.87/88.67</td><td>55.36/53.94</td><td>78.43/91.75</td><td>82.93/93.42</td><td>52.75/71.15</td><td>81.44/91.11</td><td>57.36/54.29</td><td>55.60/58.92</td><td>66.72/75.41</td></tr><tr><td>DRCT [7]</td><td>83.46/98.08</td><td>72.99/74.42</td><td>70.67/80.05</td><td>70.90/79.12</td><td>70.23/85.59</td><td>68.49/76.94</td><td>69.90/74.48</td><td>68.14/74.05</td><td>71.85/80.34</td></tr><tr><td>Cozzo2024 [9]</td><td>81.79/94.81</td><td>73.86/91.60</td><td>77.49/90.33</td><td>67.88/80.98</td><td>50.42/61.25</td><td>75.93/86.15</td><td>55.37/52.83</td><td>73.98/89.55</td><td>69.59/80.94</td></tr><tr><td>RINE [21]</td><td>95.15/99.25</td><td>87.49/87.54</td><td>90.10/98.58</td><td>88.34/97.88</td><td>51.36/85.96</td><td>89.54/98.21</td><td>46.97/45.43</td><td>72.20/79.64</td><td>77.64/86.56</td></tr><tr><td>AIDE [47]</td><td>87.14/96.78</td><td>56.26/62.49</td><td>81.94/92.40</td><td>78.65/88.32</td><td>50.12/54.74</td><td>80.08/90.65</td><td>64.20/73.60</td><td>66.04/73.71</td><td>70.55/79.09</td></tr><tr><td>C2P-CLIP [40]</td><td>96.59/99.51</td><td>46.12/54.50</td><td>83.47/90.81</td><td>74.29/86.06</td><td>50.95/66.92</td><td>75.38/86.00</td><td>54.14/59.77</td><td>59.57/67.23</td><td>67.56/76.35</td></tr><tr><td>RNSIDNet</td><td>91.70/98.40</td><td>92.43/97.38</td><td>91.32/97.61</td><td>83.94/94.13</td><td>85.18/98.22</td><td>86.18/95.78</td><td>66.65/76.80</td><td>73.85/83.46</td><td>83.81/92.72</td></tr></table>

Performance comparison of diferent AI-generated image detection models on the AIGCDetectionBenchmark [52] dataset. All benchmark results are reported as ACC (%).

<table><tr><td>Method</td><td>ADM</td><td>DALL-E 2</td><td>Glide</td><td>Midjourney</td><td>VQDM</td><td>BigGAN</td><td>CycleGAN</td><td>GauGAN</td><td>ProGAN</td><td>SD 1.4</td><td>SD 1.5</td><td>StarGAN</td><td>StyleGAN</td><td>StyleGAN 2</td><td>WFR</td><td>WuKong</td><td>AVG</td></tr><tr><td>LGrad [42]</td><td>53.45</td><td>57.60</td><td>60.05</td><td>56.83</td><td>51.40</td><td>48.18</td><td>44.17</td><td>48.10</td><td>49.20</td><td>47.05</td><td>46.75</td><td>48.17</td><td>48.15</td><td>46.04</td><td>48.55</td><td>49.30</td><td>50.19</td></tr><tr><td>DIRE [46]</td><td>57.63</td><td>69.45</td><td>62.92</td><td>58.00</td><td>54.91</td><td>46.42</td><td>50.11</td><td>52.26</td><td>49.98</td><td>51.35</td><td>51.82</td><td>50.15</td><td>50.29</td><td>52.96</td><td>49.50</td><td>52.06</td><td>53.74</td></tr><tr><td>UFD [29]</td><td>66.87</td><td>50.75</td><td>62.46</td><td>56.13</td><td>85.31</td><td>95.08</td><td>98.33</td><td>99.47</td><td>99.81</td><td>63.66</td><td>63.49</td><td>95.75</td><td>84.93</td><td>74.96</td><td>86.90</td><td>70.93</td><td>78.43</td></tr><tr><td>DRCT [7]</td><td>66.38</td><td>77.55</td><td>73.24</td><td>94.33</td><td>76.78</td><td>60.15</td><td>49.77</td><td>50.94</td><td>58.53</td><td>99.24</td><td>99.11</td><td>55.48</td><td>64.40</td><td>55.05</td><td>50.50</td><td>99.22</td><td>70.67</td></tr><tr><td>Cozzo2024 [9]</td><td>66.58</td><td>90.75</td><td>95.93</td><td>68.08</td><td>82.75</td><td>74.10</td><td>87.28</td><td>83.94</td><td>72.96</td><td>85.46</td><td>85.72</td><td>54.95</td><td>70.50</td><td>70.86</td><td>71.55</td><td>78.50</td><td>77.49</td></tr><tr><td>RINE [21]</td><td>95.65</td><td>74.80</td><td>92.07</td><td>81.67</td><td>96.58</td><td>88.88</td><td>94.40</td><td>98.09</td><td>99.49</td><td>96.76</td><td>96.62</td><td>63.33</td><td>87.39</td><td>79.51</td><td>99.60</td><td>96.68</td><td>90.10</td></tr><tr><td>AIDE [47]</td><td>78.47</td><td>95.00</td><td>91.76</td><td>81.42</td><td>80.25</td><td>77.20</td><td>74.38</td><td>64.36</td><td>69.27</td><td>99.74</td><td>99.74</td><td>80.27</td><td>71.32</td><td>72.51</td><td>76.45</td><td>98.85</td><td>81.94</td></tr><tr><td>C2P-CLIP [40]</td><td>90.22</td><td>99.35</td><td>97.81</td><td>97.55</td><td>96.39</td><td>72.12</td><td>97.09</td><td>68.39</td><td>56.49</td><td>98.99</td><td>98.79</td><td>61.31</td><td>61.54</td><td>58.33</td><td>82.45</td><td>98.62</td><td>83.47</td></tr><tr><td>RNSIDNet</td><td>83.73</td><td>87.30</td><td>89.93</td><td>94.32</td><td>93.14</td><td>97.25</td><td>96.03</td><td>97.80</td><td>90.29</td><td>96.92</td><td>96.96</td><td>95.02</td><td>86.93</td><td>86.52</td><td>74.20</td><td>94.79</td><td>91.32</td></tr></table>

Performance comparison of diferent AI-generated image detection models on the AIGCDetectionBenchmark [52] dataset. All benchmark results are reported as AUC (%).

<table><tr><td>Method</td><td>ADM</td><td>DALL-E 2</td><td>Glide</td><td>Midjourney</td><td>VQDM</td><td>BigGAN</td><td>CycleGAN</td><td>GauGAN</td><td>ProGAN</td><td>SD 1.4</td><td>SD 1.5</td><td>StarGAN</td><td>StyleGAN</td><td>StyleGAN 2</td><td>WFR</td><td>WuKong</td><td>AVG</td></tr><tr><td>LGrad [42]</td><td>60.80</td><td>77.11</td><td>71.46</td><td>64.44</td><td>52.22</td><td>46.10</td><td>48.37</td><td>52.91</td><td>50.81</td><td>47.78</td><td>47.68</td><td>46.58</td><td>48.21</td><td>45.09</td><td>52.08</td><td>49.36</td><td>53.81</td></tr><tr><td>DIRE [46]</td><td>63.35</td><td>82.25</td><td>78.66</td><td>64.12</td><td>59.61</td><td>46.00</td><td>47.34</td><td>52.27</td><td>49.64</td><td>53.02</td><td>53.64</td><td>45.15</td><td>51.78</td><td>52.00</td><td>48.16</td><td>54.36</td><td>56.33</td></tr><tr><td>UFD [29]</td><td>87.20</td><td>66.82</td><td>85.24</td><td>75.54</td><td>96.49</td><td>99.19</td><td> $\underline{99.77}$ </td><td> $\underline{99.98}$ </td><td> $\underline{100.00}$ </td><td>87.79</td><td>87.35</td><td>99.32</td><td> $\underline{97.29}$ </td><td> $\underline{97.93}$ </td><td>96.52</td><td>91.59</td><td>91.75</td></tr><tr><td>DRCT [7]</td><td>94.84</td><td>99.69</td><td>97.52</td><td> $\underline{99.33}$ </td><td>96.94</td><td>70.65</td><td> $\underline{62.27}$ </td><td>47.58</td><td>63.89</td><td> $\underline{100.00}$ </td><td>99.98</td><td>59.91</td><td> $\underline{74.32}$ </td><td>59.96</td><td>53.92</td><td> $\underline{99.99}$ </td><td>80.05</td></tr><tr><td>Cozzo2024 [9]</td><td>85.85</td><td> $\underline{99.85}$ </td><td> $\underline{99.35}$ </td><td> $\underline{89.17}$ </td><td>96.22</td><td>81.89</td><td>92.41</td><td>92.07</td><td>92.58</td><td>97.22</td><td>97.30</td><td>81.22</td><td>78.43</td><td>77.65</td><td>89.06</td><td>94.94</td><td>90.33</td></tr><tr><td>RINE [21]</td><td> $\underline{99.25}$ </td><td>99.34</td><td>97.68</td><td>95.01</td><td> $\underline{99.75}$ </td><td> $\underline{99.81}$ </td><td> $\underline{99.91}$ </td><td> $\underline{99.98}$ </td><td> $\underline{100.00}$ </td><td>99.89</td><td>99.85</td><td>99.98</td><td> $\underline{98.45}$ </td><td>88.52</td><td> $\underline{99.99}$ </td><td>99.83</td><td> $\underline{98.58}$ </td></tr><tr><td>AIDE [47]</td><td>91.46</td><td> $\underline{99.92}$ </td><td>98.54</td><td>97.40</td><td>96.00</td><td>85.67</td><td> $\underline{94.55}$ </td><td>72.57</td><td>84.25</td><td> $\underline{99.98}$ </td><td> $\underline{99.99}$ </td><td>91.07</td><td>87.22</td><td>90.37</td><td>89.38</td><td>99.96</td><td>92.40</td></tr><tr><td>C2P-CLIP [40]</td><td> $\underline{98.58}$ </td><td> $\underline{99.85}$ </td><td> $\underline{99.50}$ </td><td> $\underline{99.68}$ </td><td> $\underline{99.40}$ </td><td>83.65</td><td>98.91</td><td>76.48</td><td>83.22</td><td> $\underline{99.97}$ </td><td>99.96</td><td>89.07</td><td>68.03</td><td>65.87</td><td>90.88</td><td> $\underline{99.83}$ </td><td>90.81</td></tr><tr><td>RNSIDNet</td><td>96.03</td><td>99.22</td><td>97.85</td><td>99.18</td><td>98.83</td><td> $\underline{99.71}$ </td><td>99.36</td><td> $\underline{99.89}$ </td><td> $\underline{99.79}$ </td><td>99.67</td><td>99.59</td><td> $\underline{99.99}$ </td><td>96.47</td><td> $\underline{93.73}$ </td><td>83.30</td><td>99.18</td><td> $\underline{97.61}$ </td></tr></table>

## 4.1.4. Evaluation Metrics

Following standard image forensics protocols, we employ Accuracy (ACC) and the Area Under the Receiver Operating Characteristic Curve (AUC) as our primary quantitative metrics. The classification threshold is fixed at 0.5. ACC measures the overall classification correctness, while AUC provides a robust evaluation of the model’s discriminative capability across varying thresholds by integrating the True Positive Rate (TPR) against the False Positive Rate (FPR). Meanwhile, We also report the average (AVG) metric values across the test datasets to obtain summary evaluations.

## 4.2. Comparison to State-of-the-art Models

Extensive evaluations across eight benchmarks (Tables 3–7) demonstrate that RNSIDNet achieves state-of-theart cross-domain generalization, with an average ACC of 83.81% and AUC of 92.72%. To expose common failure modes, we closely examine several representative test sets. AIGCDetectionBenchmark [52] is the largest benchmark dataset in our experiment set with the widest variety of generators, while Synthbuster [3] features high-resolution images synthesized by recent advanced generative models. In our experiments, several strong baselines sufer a catastrophic performance inversion. For instance, the detection ACC of C2P-CLIP [40] falls sharply from 83.47% to 46.12%. Such drastic variance signals that RGB-only or prior-dependent detectors tend to memorize generator-specific superficial cues, leading to severe modality overfitting and a form of representation collapse under distribution shift. In contrast, our method remains remarkably stable across this spectrum, efectively bypassing the semantic camouflage that deceives single-modality detectors.

Performance comparison of diferent AI-generated image detection models on the Synthbuster [3] dataset. All benchmark results are reported as ACC (%) / AUC (%).

<table><tr><td>Method</td><td>Dalle-E 2</td><td>Dalle-E 3</td><td>Firefly</td><td>Glide</td><td>Midjourney</td><td>SD 1.3</td><td>SD 1.4</td><td>SD 2</td><td>SD XL</td><td>AVG</td></tr><tr><td>LGrad [42]</td><td>48.15/47.71</td><td>40.80/37.72</td><td>41.85/37.84</td><td>61.45/64.41</td><td>44.95/47.89</td><td>40.30/29.08</td><td>40.25/28.47</td><td>40.70/23.53</td><td>40.50/35.82</td><td>44.33/39.16</td></tr><tr><td>DIRE [46]</td><td>53.55/56.29</td><td>36.35/32.14</td><td>51.45/48.68</td><td>60.90/69.87</td><td>51.90/51.26</td><td>35.05/29.76</td><td>34.60/29.38</td><td>50.10/48.30</td><td>56.30/56.27</td><td>47.80/46.88</td></tr><tr><td>UFD [29]</td><td>70.95/78.59</td><td>34.80/9.02</td><td>78.15/88.40</td><td>40.70/34.40</td><td>39.90/28.85</td><td>57.85/60.71</td><td>57.35/60.65</td><td>63.10/67.71</td><td>55.45/57.12</td><td>55.36/53.94</td></tr><tr><td>DRCT [7]</td><td>47.55/36.28</td><td>53.40/46.49</td><td>48.10/35.37</td><td>60.20/73.86</td><td>89.30/95.64</td><td>94.10/100.00</td><td>94.10/100.00</td><td>90.80/96.44</td><td>79.40/85.69</td><td>72.99/74.42</td></tr><tr><td>Cozzo2024 [9]</td><td>68.10/87.49</td><td>74.15/93.69</td><td>64.55/88.38</td><td>97.25/99.65</td><td>62.85/83.18</td><td>78.85/94.87</td><td>78.55/94.66</td><td>71.45/91.35</td><td>68.95/91.14</td><td>73.86/91.60</td></tr><tr><td>RINE [21]</td><td>89.80/95.28</td><td>47.20/11.36</td><td>85.25/92.22</td><td>90.05/94.89</td><td>92.45/96.32</td><td>96.45/100.00</td><td>96.45/100.00</td><td>93.50/98.08</td><td>96.30/99.74</td><td>87.49/87.54</td></tr><tr><td>AIDE [47]</td><td>39.25/42.21</td><td>39.00/41.68</td><td>25.75/10.52</td><td>65.90/74.59</td><td>59.70/66.19</td><td>75.15/92.18</td><td>74.90/90.91</td><td>55.95/62.25</td><td>70.70/81.91</td><td>56.26/62.49</td></tr><tr><td>C2P-CLIP [40]</td><td>49.60/57.58</td><td>49.85/65.75</td><td>17.25/2.45</td><td>49.90/36.23</td><td>49.90/62.10</td><td>50.10/79.30</td><td>50.10/78.99</td><td>48.35/51.74</td><td>50.00/56.38</td><td>46.12/54.50</td></tr><tr><td>RNSIDNet</td><td>92.30/98.20</td><td>80.10/90.66</td><td>94.40/98.79</td><td>81.10/89.95</td><td>97.15/99.90</td><td>96.50/99.66</td><td>95.90/99.39</td><td>97.20/99.91</td><td>97.20/99.94</td><td>92.43/97.38</td></tr></table>

Performance comparison of diferent AI-generated image detection models on the DDA-COCO [8] dataset. All benchmark results are reported as ACC (%) / AUC (%).

<table><tr><td>Method</td><td>SD-VAE-FT-EMA</td><td>SD-VAE-FT-MSE</td><td>SD XL-VAE</td><td>SD 2.1</td><td>SD 3.5-Large</td><td>AVG</td></tr><tr><td>LGrad [42]</td><td>50.22/51.08</td><td>50.63/54.88</td><td>50.00/51.17</td><td>50.63/54.87</td><td>49.47/48.73</td><td>50.19/52.15</td></tr><tr><td>DIRE [46]</td><td>50.32/50.47</td><td>52.38/53.55</td><td>51.95/53.14</td><td>52.40/53.55</td><td>50.00/49.79</td><td>51.41/52.10</td></tr><tr><td>UFD [29]</td><td>54.40/76.50</td><td>53.24/73.59</td><td>51.52/66.35</td><td>53.29/73.59</td><td>51.29/65.72</td><td>52.75/71.15</td></tr><tr><td>DRCT [7]</td><td>83.32/95.27</td><td>77.00/93.11</td><td>62.49/81.69</td><td>76.83/93.10</td><td>51.49/64.76</td><td>70.23/85.59</td></tr><tr><td>Cozzo2024 [9]</td><td>50.46/63.99</td><td>50.55/62.97</td><td>50.32/60.92</td><td>50.57/62.93</td><td>50.18/55.43</td><td>50.42/61.25</td></tr><tr><td>RINE [21]</td><td>51.56/88.42</td><td>51.91/88.98</td><td>50.24/80.09</td><td>52.02/88.98</td><td>51.05/83.31</td><td>51.36/85.96</td></tr><tr><td>AIDE [47]</td><td>50.18/56.83</td><td>50.14/55.95</td><td>50.13/53.10</td><td>50.12/55.95</td><td>50.05/51.88</td><td>50.12/54.74</td></tr><tr><td>C2P-CLIP [40]</td><td>51.85/69.16</td><td>51.06/67.68</td><td>49.85/62.38</td><td>51.00/67.66</td><td>50.97/67.70</td><td>50.95/66.92</td></tr><tr><td>RNSIDNet</td><td>84.18/97.95</td><td>89.69/98.91</td><td>86.05/98.48</td><td>89.63/98.91</td><td>76.35/96.84</td><td>85.18/98.22</td></tr></table>

To further test generalization performance against cutting-edge generative architectures and extreme visual disguise, we evaluate on the DDA-COCO [8] dataset. With pixel-level aligned fake images almost indistinguishable from real ones, prior-dependent methods like AIDE [47] degenerate to random guessing, while RNSIDNet maintains robust discrimination (80.94% ACC, 98.00% AUC). Notably, DRCT [7] and RINE [21] also exhibit competitive performance here. Given that both their and our method employ contrastive learning, this shared success underscores the critical role of contrastive constraints in disentangling subtle generation artifacts. Finally, in uncontrolled wild scenarios, RNSIDNet achieves the highest AUC of 76.80% on Chameleon [47], with the second-best ACC and AUC of on WildRF [5], confirming that the intrinsic physical fingerprints captured by our enhanced architecture are highly robust against complex real-world post-processing, bridging the gap between laboratory tests and open-world deployment.

## 4.3. Ablation Studies

To rigorously validate the individual contributions of the core components in RNSIDNet, we conduct a comprehensive ablation study. This includes both leave-one-out experiments, which systematically remove the BAM. FiLM. or HSCL modules, and replacement experiments, which substitute the noise extractor and the contrastive loss function.

As shown in Table 8, removing BAM, FiLM, or HSCL decreases the overall accuracy (92.92%) to 90.33%, 89.79%, and 90.42%, respectively. The most severe decline occurs without FiLM (a 3.13% drop), underscoring its critical role in dynamically fusing heterogeneous modalities. Meanwhile, the performance degradation without BAM and HSCL confirms their necessity in refining semantic features and shaping precise decision boundaries. For the replacemen variants, substituting Bayar convolutions with fixed SRM [12] or Noiseprint++ [14] reduces the ACC to 91.81% and 88.97%. Although Noiseprint++ yields a marginally higher AUC (97.96%), its inferior ACC suggests threshold instability, demonstrating that our learnable Bayar constraints are inherently more robust. Similarly, replacing HSCL with InfoNCE++ [30] or SupCon [20] leads to sub-optimal performance, highlighting the advantage of our hard sample-aware mechanism over standard contrastive objectives.

Table 8  
Ablation study of RNSIDNet framework on the Synthbuster [3] dataset. $\mathrm { \Delta } \mathsf { w } / \mathrm { o } ^ { \prime }$ denotes removing a specific module, while $\mathrm { \dot { w } } / \mathrm { \dot { \Omega } }$ denotes replacing our proposed module with a conventional alternative.

<table><tr><td>Category</td><td>Model Variant</td><td>ACC (%)</td><td>Δ ACC</td><td>AUC (%)</td><td>Δ AUC</td></tr><tr><td>Full Model</td><td>RNSIDNet (Ours)</td><td>92.92</td><td>-</td><td>97.68</td><td>-</td></tr><tr><td rowspan="3">Module Ablation</td><td>w/o BAM</td><td>90.33</td><td>-2.59</td><td>97.21</td><td>-0.47</td></tr><tr><td>w/o FiLM</td><td>89.79</td><td>-3.13</td><td>95.51</td><td>-2.17</td></tr><tr><td>w/o HSCL</td><td>90.42</td><td>-2.50</td><td>96.46</td><td>-1.22</td></tr><tr><td rowspan="2">Noise Extractor</td><td>w/ SRM Filters</td><td>91.81</td><td>-1.11</td><td>97.43</td><td>-0.25</td></tr><tr><td>w/ Noiseprint++</td><td>88.97</td><td>-3.95</td><td>97.96</td><td>+0.28</td></tr><tr><td rowspan="2">Loss Function</td><td>w/ InfoNCE++</td><td>90.41</td><td>-2.51</td><td>96.58</td><td>-1.10</td></tr><tr><td>w/ SupCon</td><td>90.62</td><td>-2.30</td><td>96.72</td><td>-0.96</td></tr></table>

## Table 9

Comparison of model complexity among diferent methods.

<table><tr><td>Method</td><td>Trainable Parameters (M)</td><td>Total Parameters (M)</td><td>FLOPs (G)</td></tr><tr><td>LGrad [42]</td><td>23.51</td><td>23.51</td><td>8.26</td></tr><tr><td>DIRE [46]</td><td>23.51</td><td>23.51</td><td>8.26</td></tr><tr><td>UFD [29]</td><td>0.00077</td><td>427.62</td><td>103.79</td></tr><tr><td>DRCT [7]</td><td>88.62</td><td>88.62</td><td>30.71</td></tr><tr><td>Cozzo2024 [9]</td><td>0.00077</td><td>427.62</td><td>103.79</td></tr><tr><td>RINE [21]</td><td>10.52</td><td>438.14</td><td>104.00</td></tr><tr><td>AIDE [47]</td><td>54.43</td><td>897.83</td><td>451.39</td></tr><tr><td>C2P-CLIP [40]</td><td>2.36</td><td>429.98</td><td>155.64</td></tr><tr><td>RNSIDNet</td><td>13.58</td><td>441.20</td><td>104.31</td></tr></table>

## 4.4. Robustness Analysis

To verify the structural stability of our enhanced RGB-Noise representations, we conduct a comprehensive robustness analysis under various degradation conditions. Specifically, we evaluate the models on two distinct datasets, Synthbuster [3] and AIGCDetect [52] , applying JPEG compression and Gaussian blur across varying levels of intensity.

As illustrated in Figure 3, RNSIDNet consistently maintains a competitive advantage across various degradation conditions and data sources. In contrast, several baseline methods exhibit significant instability and severe performance fluctuations due to diferent datasets. Meanwhile, we observe that the model inevitably experiences a performance drop under Gaussian blur perturbations. This is expected, as Gaussian blur acts as a low-pass filter that severely erases critica high-frequency information in the frequency domain. Nevertheless, the dual-branch architecture of RNSIDNet provides essential structural resilience, successfully preventing a complete performance collapse and sustaining a functiona detection capability even under heavy blur

## 4.5. Scalability and Complexity Analysis

In this section, We evaluate the practical applicability of RNSIDNet from two perspectives: data scalability and computational complexity. To assess data scalability, we trained the model using 20%, 50%, 80%, and 100% of the training data and evaluated it on the Synthbuster [3] dataset. As illustrated in Figure 4, increasing the training data boosts the average accuracy from 89.54% to 92.92%, enabling the network to learn more robust features. Notably, the model demonstrates excellent data eficiency, achieving over 90% accuracy on Midjourney and Stable Difusion with only 20% of the data. However, handling more complex distributions requires more data; for instance, performance on Firefly surges from 75.90% to 90.25% when utilizing the full dataset. Thus, expanding the training set efectively enhances both absolute performance and cross-domain generalization.

![](images/d104c568319a66bd1832a8e721c0adcaa7e8a49fdfabcdd9b53ab9d9acbec01d.jpg)  
(a) Synthbuster - JPEG Compression Quality

![](images/a01febab3e427eae2e9c3dedbf73721a69e25aec3c9de90fb378dd71416df4b2.jpg)  
(b) Synthbuster - Gaussian Blur Radius

![](images/44e4cde12a6a4ff9471894d063e3a2af64a6ac5e598bedb0e9d8851368b112bf.jpg)  
(c) AIGCDetect - JPEG Compression Quality

![](images/3541841c847632f393ba525fd46841d612ea67609d8bf8d1d981446933efce78.jpg)  
(d) AIGCDetect - Gaussian Blur Radius  
Figure 3: Performance evaluation of diferent models under JPEG compression and Gaussian blur on Synthbuster [3] and AIGCDetectionBenchmark [52] datasets.

Beyond data eficiency, Table 9 compares the computational complexity of various methods. RNSIDNet achieves an optimal balance between eficiency and detection accuracy. While lightweight methods (e.g., UFD [29], Cozzo2024 [9]) often lack the representational capacity for diverse distributions, larger models incur prohibitive computational costs. RNSIDNet avoids both extremes by utilizing only 13.58M trainable parameters, ensuring strong generalization with minimal overhead. Computationally, despite its dual-branch architecture, RNSIDNet requires 104.31G FLOPs. This is highly comparable to standard CLIP ViT-L [34] baselines (103G–105G), demonstrating that the introduced noise extraction and modulation components add negligible computational burden. Consequently, RN SIDNet successfully disentangles heterogeneous features without noticeably increasing inference latency, confirming its practicality for real-world applications.

Generalized Synthetic Image Detection with Enhanced RGB-Noise Representation Learning

![](images/6be6c087f0ae8061763af54c4b684d1ed6b3982ae401ee128aa240bde9473081.jpg)  
Figure 4: Performance comparison of the RNSIDNet model trained with diferent sample set sizes on the Synthbuster [3] dataset.

Table 10  
Performance comparison on the AIGCDetectionBenchmark [52] to evaluate the impact of training data sources on crossarchitecture generalization. Results are reported in AUC (%).

<table><tr><td>Method</td><td>ADM</td><td>DALL-E 2</td><td>Glide</td><td>Midjourney</td><td>VQDM</td><td>BigGAN</td><td>CycleGAN</td><td>GauGAN</td><td>ProGAN</td><td>SD 1.4</td><td>SD 1.5</td><td>StarGAN</td><td>StyleGAN</td><td>StyleGAN 2</td><td>WFR</td><td>WuKong</td><td>AVG</td></tr><tr><td>RINE [21]</td><td>99.25</td><td>99.34</td><td>97.68</td><td>95.01</td><td>99.75</td><td>99.81</td><td>99.91</td><td>99.98</td><td>100.00</td><td>99.89</td><td>99.85</td><td>99.98</td><td>98.45</td><td>88.52</td><td>99.99</td><td>99.83</td><td>98.58</td></tr><tr><td>RNSIDNet</td><td>96.03</td><td>99.22</td><td>97.85</td><td>99.18</td><td>98.83</td><td>99.71</td><td>99.36</td><td>99.89</td><td>99.79</td><td>99.67</td><td>99.59</td><td>99.99</td><td>96.47</td><td>93.73</td><td>83.30</td><td>99.18</td><td>97.61</td></tr><tr><td>RNSIDNet (ProGAN)</td><td>93.46</td><td>86.74</td><td>94.93</td><td>89.54</td><td>98.57</td><td>99.97</td><td>99.98</td><td>100.00</td><td>100.00</td><td>97.47</td><td>97.37</td><td>99.98</td><td>99.86</td><td>99.67</td><td>99.78</td><td>98.01</td><td>97.21</td></tr></table>

## 4.6. Efect of Multi-Source Training Strategy

To investigate the impact of training data sources on model generalization, we evaluate our method under two diferent training paradigms in Table 10. Specifically, following the protocol adopted by previous works [29, 21, 47], we train RNSIDNet exclusively on the ProGAN dataset provided by Wang et al. [44], whereas the standard RNSIDNet is trained on the multi-source AMSID dataset.

As shown in Table 10, the ProGAN-trained RNSIDNet generalizes exceptionally well across unseen GANs (near 100% AUC), indicating shared structural artifacts within the same generative family. However, its performance drops significantly on Difusion models (e.g., 86.74% on DALL-E 2), exposing a critical cross-architecture domain gap. In contrast, the standard multi-source RNSIDNet successfully bridges this gap, boosting Difusion model detection to over 99% while maintaining peak performance on GANs. This validates that relying on a single generator inevitably introduces architecture-specific biases, whereas our multi-source strategy forces the network to capture universal, intrinsic generative artifacts for superior cross-model generalization.

## 4.7. Feature Visualization Analysis

To evaluate feature disentanglement, we apply t-SNE on equal samples from Real, GAN, and Difusion categories. As depicted in Fig. 5(a), raw image features exhibit severe semantic entanglement. In contrast, RNSIDNet transforms the feature space into an ideal clustering structure, as seen in Fig. 5(b). A substantial margin clearly separates pristine and synthetic clusters, validating the eficacy of our dual-branch feature fusion mechanism.

Crucially, GAN and Difusion samples are further repelled into distinct clusters within the synthetic manifold. This high separability proves that our HSCL strategy successfully enforces strict intra-class compactness and inter-class separation. By successfully decoupling the unique physical fingerprints of heterogeneous generative architectures, RNSIDNet establishes a solid theoretical foundation for its superior cross-domain generalization.

![](images/36ffe70864d3ee201bf4c9d1a07849979e6f10c7c37dcb6d7679d3088338c6ae.jpg)  
(a) Features of unprocessed raw images

![](images/b64c607df73c665a78b23b98ff370e9e559c85f528d01d832950c0c8b4e11ee0.jpg)  
(b) Features extracted by RNSIDNet  
Figure 5: Feature distribution visualization of real and synthetic images via t-SNE

## 5. Conclusion

In this work, we propose RNSIDNet, an innovative dual-branch framework for robust and generalizable synthetic image detection. By synergizing global RGB semantics extracted from a visual foundation model with high-frequency artifacts captured by Bayar-constrained convolutions, our method efectively constructs a comprehensive representation of image authenticity. Crucially, the introduction of our HSCL strategy successfully forces the model to decouple image semantics from inherent generative traces. Comprehensive evaluations validate that RNSIDNet establishes a new state-of-the-art, demonstrating superior cross-model generalization and exceptional resilience against complex post-processing degradations. Moving forward, applying this content-artifact decoupling paradigm to synthetic video forensics remains a promising direction

## References

[1] Agustsson, E., Timofte, R., 2017. Ntire 2017 challenge on single image super-resolution: Dataset and study, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 126–135.

[2] Bai, J., Lin, M., Cao, G., Lou, Z., 2024. Ai-generated video detection via spatial-temporal anomaly learning, in: Chinese Conference on Pattern Recognition and Computer Vision (PRCV), pp. 460–470.

[3] Bammey, Q., 2023. Synthbuster: Towards detection of difusion model generated images. IEEE Open Journal of Signal Processing 5, 1–9.

[4] Bayar, B., Stamm, M.C., 2016. A deep learning approach to universal image manipulation detection using a new convolutional layer, in: Proceedings of the 4th ACM Workshop on Information Hiding and Multimedia Security, pp. 5–10.

[5] Cavia, B., Horwitz, E., Reiss, T., Hoshen, Y., 2024. Real-time deepfake detection in the real-world. arXiv preprint arXiv:2406.09398 .

[6] Chang, J.. Li. Z.. Lou. J.. Oiu. Z.. Lin. H., 2025. Mmp-2k: A benchmark multi-labeled macro photography image quality assessment database arXiv preprint arXiv:2505.19065 .

[7] Chen, B., Zeng, J., Yang, J., Yang, R., 2024. DRCT: Difusion reconstruction contrastive training towards universal detection of difusion generated images, in: Proceedings of the International Conference on Machine Learning, pp. 7621–7639.

[8] Chen. R.. Xi. J.. Yan. Z.. Zhang, K.Y.. Wu. S.. Xie, J.. Chen. X.. Xu. L.. Guan, I.. Yao, T.. Ding, S.. . Dual data alignment makes ai-generated image detector easier generalizable, in: Advances in Neural Information Processing Systems, pp. 106475–106500.

[9] Cozzolino, D., Poggi, G., Corvi, R., Nießner, M., Verdoliva, L., 2024. Raising the bar of AI-generated image detection with CLIP, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4356–4366.

[10] Dang-Nguyen, D.T., Pasquini, C., Conotter, V., Boato, G., 2015. Raise: A raw images dataset for digital image forensics, in: Proceedings of the ACM Multimedia Systems Conference, pp. 219–224.

[11] Frank, J., Eisenhofer, T., Schönherr, L., Fischer, A., Kolossa, D., Holz, T., 2020. Leveraging frequency analysis for deep fake image recognition, in: Proceedings of the International Conference on Machine Learning, pp. 3247–3258.

[12] Fridrich, J., Kodovsky, J., 2012. Rich models for steganalysis of digital images. IEEE Transactions on information Forensics and Security 7, 868–882.

[13] Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y., 2014. Generative adversaria nets. Advances in neural information processing systems 27.

[14] Guillaro, F., Cozzolino, D., Sud, A., Dufour, N., Verdoliva, L., 2023. Trufor: Leveraging all-round clues for trustworthy image forgery detection and localization, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 20606–20615.

[15] Guillaro, F., Zingarini, G., Usman, B., Sud, A., Cozzolino, D., Verdoliva, L., 2025. A bias-free training paradigm for more general AI-generated image detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18685–18694.

[16] Guo, K., Zhu, H., Cao, G., 2024. Efective image tampering localization via enhanced transformer and co-attention fusion, in: Icassp 2024-2024 ieee international conference on acoustics, speech and signal processing, pp. 4895–4899.

[17] Ho, J., Jain, A., Abbeel, P., 2020. Denoising difusion probabilistic models, in: Advances in Neural Information Processing Systems, pp. 6840–6851.

[18] Ju, Y., Jia, S., Ke, L., Xue, H., Nagano, K., Lyu, S., 2022. Fusing global and local features for generalized ai-synthesized image detection, in: 2022 IEEE International Conference on Image Processing, pp. 3465–3469.

[19] Karras, T., Aittala, M., Aila, T., Laine, S., 2022. Elucidating the design space of difusion-based generative models, in: Advances in Neura Information Processing Systems, pp. 26565–26577.

[20] Khosla, P., Teterwak, P., Wang, C., Sarna, A., Tian, Y., Isola, P., Maschinot, A., Liu, C., Krishnan, D., 2020. Supervised contrastive learning, in: Advances in Neural Information Processing Systems, pp. 18661–18673.

[21] Koutlis, C., Papadopoulos, S., 2024. Leveraging representations from intermediate encoder-blocks for synthetic image detection, in: Proceedings of the European Conference on Computer Vision, pp. 394–411.

[22] Lim, B., Son, S., Kim, H., Nah, S., Mu Lee, K., 2017. Enhanced deep residual networks for single image super-resolution, in: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pp. 136–144.

[23] Lin, T.Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L., 2014. Microsoft COCO: Common objects in context, in: Proceedings of the European Conference on Computer Vision, pp. 740–755.

[24] Liu, B., Yang, F., Bi, X., Xiao, B., Li, W., Gao, X., 2022. Detecting generated images by real images, in: European conference on computer vision, pp. 95–110.

[25] Liu, H., Tan, Z., Tan, C., Wei, Y., Wang, J., Zhao, Y., 2024. Forgery-aware adaptive transformer for generalizable synthetic image detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10770–10780.

[26] Lou, Z., Cao, G., Guo, K., Yu, L., Weng, S., 2025a. Exploring multi-view pixel contrast for general and robust image forgery localization. IEEE Transactions on Information Forensics and Security .

[27] Lou, Z., Cao, G., Lin, M., Yu, L., Weng, S., 2025b. Trusted video inpainting localization via deep attentive noise learning. IEEE Transactions on Dependable and Secure Computing .

[28] Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J., 2022. Dpm-solver: A fast ode solver for difusion probabilistic model sampling in around 10 steps, in: Advances in Neural Information Processing Systems, pp. 5775–5787.

[29] Ojha, U., Li, Y., Lee, Y.J., 2023. Towards universal fake image detectors that generalize across generative models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24480–24489.

[30] Oord, A.v.d., Li, Y., Vinyals, O., 2018. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748 .

[31] Park, J., Woo, S., Lee, J.Y., Kweon, I.S., 2018. Bam: Bottleneck attention module. arXiv preprint arXiv:1807.06514 .

[32] Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A., 2018. Film: Visual reasoning with a general conditioning layer, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 3942–3951.

[33] Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R., 2024. SDXL: Improving latent difusion models for high-resolution image synthesis, in: Proceedings of the International Conference on Learning Representations, pp. 1862–1874.

[34] Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al., 2021. Learning transferable visual models from natural language supervision, in: Proceedings of the International Conference on Machine Learning, pp. 8748–8763.

[35] Rajan, A.S., Ojha, U., Schloesser, J., Lee, Y.J., 2025. Aligned datasets improve detection of latent difusion-generated images, in: Proceedings of the International Conference on Learning Representations.

[36] Robinson, J.D., Chuang, C., Sra, S., Jegelka, S., 2021. Contrastive learning with hard negative samples, in: Proceedings of the International Conference on Learning Representations.

[37] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B., 2022. High-resolution image synthesis with latent difusion models, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695.

[38] Sha, Z., Li, Z., Yu, N., Zhang, Y., 2023. De-fake: Detection and attribution of fake images generated by text-to-image generation models, in: Proceedings of the 2023 ACM SIGSAC conference on computer and communications security, pp. 3418–3432.

[39] Sinitsa, S., Fried, O., 2023. Deep image fingerprint: Accurate and low budget synthetic image detector. arXiv preprint arXiv:2303.10762 1.

[40] Tan, C., Tao, R., Liu, H., Gu, G., Wu, B., Zhao, Y., Wei, Y., 2025. C2p-clip: Injecting category common prompt in clip to enhance generalization in deepfake detection, in: Proceedings of the AAAI Conference on Artificial Intelligence, pp. 7184–7192.

[41] Tan, C., Zhao, Y., Wei, S., Gu, G., Liu, P., Wei, Y., 2024. Rethinking the up-sampling operations in cnn-based generative network for generalizable deepfake detection, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 28130–28139.

[42] Tan, C., Zhao, Y., Wei, S., Gu, G., Wei, Y., 2023. Learning on gradients: Generalized artifacts representation for GAN-generated images detection, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 12105–12114.

[43] Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł., Polosukhin, I., 2017. Attention is all you need, in: Advances in Neural Information Processing Systems.

[44] Wang, S.Y., Wang, O., Zhang, R., Owens, A., Efros, A.A., 2020. CNN-generated images are surprisingly easy to spot... for now, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 8695–8704.

[45] Wang, X., Xie, L., Dong, C., Shan, Y., 2021. Real-ESRGAN: Training real-world blind super-resolution with pure synthetic data, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1905–1914.

[46] Wang, Z., Bao, J., Zhou, W., Wang, W., Hu, H., Chen, H., Li, H., 2023. DIRE for difusion-generated image detection, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 22445–22455.

[47] Yan, S., Li, O., Cai, J., Hao, Y., Jiang, X., Hu, Y., Xie, W., 2025. A sanity check for AI-generated image detection, in: Proceedings of the International Conference on Learning Representations.

[48] Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., Darrell, T., 2020. Bdd100k: A diverse driving dataset for heterogeneous multitask learning, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2636–2645.

[49] Zhang, L., Rao, A., Agrawala, M., 2023. Adding conditional control to text-to-image difusion models, in: Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3836–3847.

[50] Zhang, R., 2019. Making convolutional networks shift-invariant again, in: Proceedings of the International Conference on Machine Learning, pp. 7324–7334.

[51] Zhao, M., Cao, G., Huang, X., Yang, L., 2022. Hybrid transformer-cnn for real image denoising. IEEE Signal Processing Letters 29, 1252– 1256.

[52] Zhong, N., Xu, Y., Li, S., Qian, Z., Zhang, X., 2023. Patchcraft: Exploring texture patch for eficient AI-generated image detection. arXiv preprint arXiv:2311.12397 .

[53] Zhu, M., Chen, H., Yan, Q., Huang, X., Lin, G., Li, W., Tu, Z., Hu, H., Hu, J., Wang, Y., 2023. Genimage: A million-scale benchmark for detecting AI-generated image, in: Advances in Neural Information Processing Systems, pp. 77771–77782.