# Radial Basis Function Networks as Projection Heads in Self-Supervised Learning

Andreas Schliebitz<sup>1,2[0000−0003−0361−7770]</sup>, Heiko Tapken<sup>1[0000−0002−0685−5072]</sup>, and Martin Atzmueller<sup>2,3[0000−0002−2480−6901]</sup>

<sup>1</sup> Osnabrück University of Applied Sciences, Albrechtstr. 30, 49076 Osnabrück, Germany

<sup>2</sup> Osnabrück University, Semantic Information Systems Group, Wachsbleiche 27, 49090 Osnabrück Germany

3 German Research Center for Artificial Intelligence (DFKI), Hamburger Str. 24, 49084 Osnabrück, Germany martin.atzmueller@uni-osnabrueck.de

Abstract. Self-supervised learning (SSL) typically relies on a backbone encoder followed by a small multilayer perceptron (MLP) projection head, which is conventionally discarded after training, while backbone quality is assessed via costly linear probing on labeled data. We argue that this approach including discarding the projector is rather computationally wasteful. Instead, we propose replacing the MLP head with a radial basis function network (RBFN), whose interpretable center and shape parameters can be exploited to judge representation quality without labels or a separate classifier. To this end, we introduce Scale-Normalized Separation (SNS), a novel label-free quality metric derived solely from the kernel centers and shapes learned during training. Across five canonical SSL architectures (MoCo, SimCLR, BYOL, SwAV and SimSiam) and four image classification datasets, we show that RBFN projection heads are competitive drop-in replacements for standard MLP projectors. We recommend constructing them with three RBF layers activated by the Gaussian radial basis function. Moreover, SNS exhibits strong to very strong positive correlation with established logistic regression metrics, demonstrating that a trained RBFN projector can act as a reliable proxy for backbone representation quality. We additionally publish a novel PyTorch compatible image classification dataset based on Google’s Open Images V7 to facilitate reproducible research into representation learning.

Keywords: Nonlinear Projection Heads · Multilayer Perceptrons · Radial Basis Function Networks · Quality Metrics · Model Evaluation · Self-Supervised Learning · Representation Learning

## 1 Introduction

Self-supervised learning (SSL) aims to learn meaningful representations from sparsely labeled data. Common SSL architectures [8,10,12,13,21] consist of two parts, the backbone f(·) and the projection head g(·). Backbones are usually deep convolutional neural networks (CNNs) or vision transformers (ViTs) [9]. Projection heads (also called projectors), implemented as shallow multilayer perceptrons (MLPs), have been shown in training to significantly improve the quality of learned representations [10, 22, 32, 44, 53]. Early experiments demonstrated that nonlinear projection heads, i. e. MLPs with a nonlinear activation function like ReLU, outperform linear projectors and can increase representation quality by up to 10 % [10, 26], illustrating the importance of the projection head in SSL. Furthermore, there is not only a basic architectural consensus within the field of SSL but also on training, testing and inference protocols [52]. Given a suficiently large image dataset subdivided into disjoint train, validation and test splits, the entire network $g ( f ( \cdot ) )$ ), i. e. the backbone and the projector, are trained end-toend on the train split using gradient descent with backpropagation. This classical nonlinear optimization pipeline adjusts the weights in accordance to some loss function applied to the outputs of the projector. After training however, the projector with its trained weights is typically discarded [10,22,52] and the backbone model is evaluated by training a linear classifier [10, 21, 34] on the labeled test split. Once model quality is deemed suficient in testing, a final transfer learning step is conducted to adapt the backbone’s weights to a specific downstream task like image classification or segmentation [52].

This paper is motivated by the observation that unconditionally discarding the projector after training appears wasteful not only from a computational but also information theoretical point of view. We demonstrate, how training a specific type of nonlinear projection head, namely a radial basis function network (RBFN) [7], can be exploited to judge representation quality of the backbone without the need for a labeled test split and by extension a linear classifier. The main contributions of this paper can be summarized as follows:

1. We present an RBFN projection and prediction head as a drop-in replacement for most default MLP heads used in SSL, with full compatibility with the popular LightlySSL framework [46].

2. We propose a new quality metric called SNS (Scale-Normalized Separation) for scoring a backbone’s representation quality based on center and shape parameters learned by the RBFN projector during training. We show through statistical correlation between our novel and current metrics that scoring representation quality is possible solely by interpreting those parameters.

3. Furthermore, we present the eficacy of our RBFN projector and quality metric by conducting an empirical study on five diferent SSL architectures and four image classification datasets paired with a grid search over diferent radial basis functions, number of kernels, projector depths and RBFN normalizations. For that, we also publish a new PyTorch compatible Dataset implementation for representation learning based on subsets from Google’s Open Images V7 dataset.

## 2 Related Work

Our work presented in this paper intersects four lines of research: self-supervised representation learning, the role of projection and prediction heads, radial basis function networks in modern deep learning and label-free evaluation of learned representations. In the following, we review each in turn.

## 2.1 Self-Supervised Visual Representation Learning

Contrastive and non-contrastive self-supervised learning (SSL) methods have rapidly closed the gap to supervised pretraining on standard image benchmarks. Contrastive approaches such as MoCo [12] and SimCLR [10] learn invariances by pulling together augmented views of the same image while pushing apart views of diferent images. Non-contrastive methods avoid explicit negatives: BYOL [21] and SimSiam [13] rely on asymmetric predictor networks and stop-gradient operations, while SwAV [8] enforces consistency between cluster assignments of diferent views. Subsequent work such as Barlow Twins [55] and VICReg [3] replaces explicit pairwise comparisons with feature-decorrelation objectives.

All of these frameworks share a common architectural pattern: a backbone encoder followed by a small multilayer perceptron projection head, which is discarded after pretraining. Our work targets this projection head directly, replacing it with a shallow radial basis function network across all five canonical frameworks.

## 2.2 The Role of Projection and Prediction Heads

Although projection heads were originally introduced as a minor implementation detail in SimCLR [10], subsequent analyses have established that they are crucial for downstream transfer. Chen et al. [10] observed that linear probing on the projection-head input substantially outperforms probing on its output, motivating the practice of discarding the head after pretraining. Gupta et al. [22] provide a theoretical account of this phenomenon, showing that the projection head absorbs invariances induced by the contrastive loss that would otherwise damage class-relevant features in the backbone. Bordes et al. [5] systematically study this “guillotine regularization” efect, demonstrating that the optimal cutof layer depends on the alignment between pretext and downstream tasks. Further work has examined the geometry of representations inside the head [26] and the impact of head depth and width [53].

To the best of our knowledge, however, no prior work has investigated whether non-standard MLP heads with strong inductive biases, such as radial basis functions, can serve as drop-in replacements while preserving or even improving the quality of the underlying backbone.

## 2.3 Radial Basis Function Networks in Deep Learning

Radial basis function networks, introduced by Broomhead and Lowe [7] and analyzed by Park and Sandberg [37] as universal approximators, were a mainstay of classical neural network research but have received comparatively little attention in the deep learning era. Recent revivals include their use as interpretable classifiers [2] and as robust alternatives to softmax heads against adversarial perturbations [54]. Gaussian RBF activations have also re-emerged in the context of kernel methods [51] and as a principled bridge between neural networks and Gaussian processes [29]. In the self-supervised setting, kernel-based perspectives on contrastive learning have been developed by Li et al. [30] and Johnson et al. [27], but these works analyze existing MLP-based losses through a kernel lens rather than introducing explicit RBF components into the architecture. In this paper we present, to the best of our knowledge, the first systematic integration of trainable RBFN heads into mainstream SSL pipelines.

## 2.4 Label-Free Evaluation of Self-Supervised Representations

The standard protocol for evaluating SSL backbones, i. e., linear probing with labeled data [28], is computationally costly and fundamentally dependent on a downstream supervised dataset. This has motivated several label-free or proxy evaluation metrics. Garrido et al. [19] propose RankMe, a measure of the efective rank of learned features that correlates with linear-probe accuracy without requiring labels. Thilak et al. [47] extend this idea with LiDAR, a discriminative variant that incorporates augmentation structure. Other proxies include alignment and uniformity on the hypersphere [50], neural collapse-inspired measures [4], and spectral properties of the embedding covariance [1]. Our proposed metric difers from those mentioned above by deriving its signal from a trained head rather than from the backbone features directly: the learned centers and shapes of the Gaussian RBF encode information about cluster structure in the feature space that, as we show empirically, tracks downstream accuracy as measured by evaluating a logistic regression model on features generated by the trained backbone. In this respect, our work is conceptually related to approaches that exploit the structure of learned classifier weights [18], but transposed to the unsupervised RBF setting.

## 3 Background

## 3.1 Methods for Self-Supervised Learning

Below, we provide a brief overview of the SSL architectures used in Section 4 to conduct our experiments. Most of them, especially the contrastive ones, are siamese in nature. Siamese networks [6] consist of two separate feedforward branches projecting two distinct transformations of the same input sample into some embedding space. In the case of SSL, the final embeddings are created using a projection head, which takes intermediate representations generated by an upstream backbone network and maps them into an often lower dimensional projection space [10]. This space is then nonlinearly optimized using a special loss function which usually rewards feature similarity [35] together with gradient descent and backpropagation. Since optimization is carried out end-to-end, weight updates are not only limited to the projector but also extend to the backbone itself, efectively minimizing the training loss after the projector [52].

MoCo [12] A self-supervised contrastive learning architecture that uses a ResNet backbone to encode images into representations, followed by a nonlinear MLP projection head. MoCo consists of a query encoder and a momentumupdated key encoder. It is trained using Information Noise-Contrastive Estimation (InfoNCE) loss to attract representations of augmented views of the same image (positive pairs) while simultaneously pushing apart a large set of negative samples stored in a queue. Improvements over the first version include the use of a nonlinear MLP projection head over a single linear layer and stronger data augmentations.

SimCLR [10] A contrastive learning framework that uses a shared ResNet backbone to encode strongly augmented views of images into representations, followed by a 2-layer MLP projector which maps features into a lower dimensional projection space. The architecture is trained using Normalized Temperature-scaled Cross Entropy (NT-Xent) loss, a variation of InfoNCE, over large batches without the use of a memory bank or momentum encoder. SimCLR relies heavily on extensive data augmentations and implicit in-batch negatives for learning visual representations.

BYOL [21] A self-supervised learning architecture that uses a ResNet backbone to encode augmented views of images, followed by a nonlinear MLP projection head and an additional prediction head in the online network. BYOL consists of an online encoder and a target encoder updated via an exponential moving average, called momentum. It is trained to regress the target projections without the need for contrastive negatives while preventing model collapse through its asymmetric architecture and momentum update.

SwAV [8] A self-supervised learning framework that uses a ResNet backbone followed by a nonlinear projection head. SwAV introduces online clustering with a swapped prediction target, where cluster codes from one augmented view are used to predict codes of another. The network is trained with a contrastive-like objective function without requiring explicit negative samples by leveraging large batch sizes and multi-crop augmentations.

SimSiam [13] A self-supervised learning architecture using a ResNet backbone with a nonlinear projector and an additional prediction MLP. The architecture consists of two identical encoders without momentum update. Sim-Siam can learn without negative samples by regressing the projection of one view from the prediction of another. Model collapse is prevented through the asymmetry between the predictor and the stop-gradient applied to one branch.

## 3.2 Radial Basis Function Networks

A radial basis function network is an artificial neural network that is activated using radial basis functions (RBFs), a family of mostly nonlinear real valued functions, which map the distance<sup>4</sup> $\| \cdot \| : \mathbb { R } ^ { n } \to [ 0 , \infty )$ of an input vector $\mathbf { x } \in \mathbb { R } ^ { n }$ to a fixed point $\mathbf { c } \in \mathbb { R } ^ { n }$ onto a scalar:

$$
\varphi_ {\mathbf {c}} \colon [ 0, \infty) \to \mathbb {R},   r \mapsto \varphi (\underbrace {\| \mathbf {x} - \mathbf {c} \|} _ {r})\tag{1}
$$

If c is omitted in Equation 1, then the RBF $\varphi$ calculates the distance of x to the origin. In both cases, the value of the RBF will only depend on the distance between the input x and the chosen fixed point. For any RBF using a fixed point other than the origin, $\varphi _ { \mathbf { c } }$ is called a radial kernel centered at $\mathbf { c } \in \mathbb { R } ^ { n }$ Infinitely smooth RBF $( \varphi \in C ^ { \infty } ( \mathbb { R } ) )$ are categorized into being either strictly positive-definite or not. A well known and widely used RBF that is infinitely smooth and strictly positive-definite is the Gaussian RBF, although others exist as discussed in Section 4.2:

$$
\varphi (r) = \exp (- (\varepsilon r) ^ {2})\tag{2}
$$

Note that due to $\sigma = 1 / \sqrt { 2 } \varepsilon$ , increasing the shape parameter $\varepsilon \in \mathbb { R }$ decreases the width of the standard Gaussian $\exp ( - r ^ { 2 } / 2 \sigma ^ { 2 } )$ resulting in a narrower kernel.

RBFNs are usually fully connected feedforward networks, which are commonly known as multilayer perceptrons. As with any other type of MLP, neither their depth (i. e. number of linear layers) nor width (i. e. number of neurons within a layer) are strictly limited. The simplest RBFN is a three layer MLP with a linear input layer, a hidden layer activated by a (nonlinear) RBF and a linear output layer. Furthermore, RBFNs are proven to be universal function approximators [37], making them equivalent to most artificial neural networks in terms of representation ability. The output of an RBFN can be described as a linear combination of radial basis functions applied to some input vector $\mathbf { x } \in \mathbb { R } ^ { n }$

$$
\phi \colon \mathbb {R} ^ {n} \to \mathbb {R}, \mathbf {x} \mapsto \sum_ {i = 1} ^ {N} w _ {i} \varphi_ {\mathbf {c}} (\parallel \mathbf {x} - \mathbf {c} _ {i} \parallel)\tag{3}
$$

where N is the number of neurons in the hidden layer, $\mathbf { c } _ { i } \in \mathbb { R } ^ { n }$ the center fixed point for neuron i and $w _ { i } \in \mathbb { R }$ the i-th output neuron’s weight. If a radial basis function $\varphi _ { \mathbf { c } }$ with a required shape parameter ε is chosen, the i-th neuron will contain its own $\varepsilon _ { i }$ , completing the set $\left\{ \mathbf { w } , \varepsilon _ { i } , \mathbf { c } _ { i } \right\}$ of learnable parameters for that specific neuron.

## 4 Method

## 4.1 Idea

Trained radial basis function networks are known to be highly explainable in comparison to regular neural networks through their use of radial basis functions as nonlinearities [25]. Interpretability results from a special set of learnable parameters which mathematically describe the shape ε and location c (centers) of the fitted RBFs within the network’s projection space [36]. Since selfsupervision is inherently based on feature similarity, methods like MoCo and SimCLR learn and retain clusters of visually similar embeddings in their backbone’s latent space [38]. We hypothesize that the quality of such a clustering can be characterized using the aforementioned RBFN parameters since the backbone is optimized using the outputs of the projector. Assuming this hypothesis holds, representation quality of the backbone could be scored via some quality function calculated on the parameters learned by the RBFN projector during training. We therefore propose the following changes to established SSL architectures and evaluation protocols:

1. Replacement of the generic MLP heads with our custom RBFN projection and prediction heads, which use highly interpretable radial basis functions, such as the Gaussian RBF, as their nonlinearities.

2. Calculation of a quality metric which is solely based the parameters learned by the RBFN projector during self-supervised training. During evaluation, this could replace both the training of a separate logistic regression (LogReg) model as well as the labeled test split.

## 4.2 Contributions

RBFN Projection Head We base our RBFN projection head (see Figure 1) on a PyTorch compatible nn.Module implementation of a radial basis function layer developed by Russo [42]. This RBF layer introduces four new hyperparameters, namely the number of kernels $( K = 2 0 4 8 ) ^ { 5 }$ to use, whether the RBF outputs should be normalized or not (true) with $n \in \{ \mathrm { t r u e } , \mathrm { f a l s e } \}$ , the type of distance function $f _ { d }$ (euclidian) being applied to $\mathbf { x } - \mathbf { c } _ { i }$ and the radial basis function $f _ { b }$ (gaussian) to process the resulting distance r. The number of efective parameters can immediately be reduced to three, since the distance function is almost always chosen to be Euclidean. Our experiments further indicate that normalization can also be disregarded in most cases, reducing the number of additional hyperparameters to only two when compared to generic MLP projectors. For activating the RBF layer within our projector, we provide ready-to-use implementations of the most common radial basis functions like the Gaussian RBF but also quadratic, multiquadric and spline variants. In our experiments, we choose two RBFs which are strictly positive definite (gaussian, inverse multiquadric) and two which are not (multiquadric, thin plate spine (tps)). If the

RBF is strictly positive definite, the resulting kernel matrix is guaranteed to be symmetric positive definite and therefore invertible [33]. This property ensures unique results for the weights of the RBFN while simultaneously stabilizing training by reducing sensitivity to noisy inputs. After training, the $K \times D$ matrix of K kernel centers $\boldsymbol { c } _ { 1 } , \dots , \boldsymbol { c } _ { K } \in \mathbb { R } ^ { D }$ with K associated shape parameters $\varepsilon _ { 1 } , \dots , \varepsilon _ { K } \in \mathbb { R }$ can be extracted from the RBF layer via predefined getter methods.

The RBFN projector does not use ReLU activation; instead, it relies on dedicated RBF layers to achieve nonlinearity while strictly adhering to the design principles of the LightlySSL framework. Our RBFN projector implements the framework’s ProjectionHead interface allowing seamless interoperability with the selection of SSL architectures described in Section 3.1. Note that, in the case of our RBFN projector, a “single layer” always refers to the combination of a linear layer, followed by optional batch normalization and an RBF layer acting as nonlinearity. The RBFN head itself is inspired by SimCLR’s MLP projector [10], hence ofering precise parametrization of input (2048), hidden (2048) and output dimensions (128), variable depth (3) as well as an optional batch normalization (false) step prior to the RBF layer. Since some SSL architectures like BYOL [21] and SimSiam [13] use an additional ReLU activated prediction head on top of the projector, we also provide our own implementation of a batch normalized RBFN prediction head (s. Figure 2) consisting of a single RBF layer with the same default parameters as its projector counterpart.

![](images/82258392ac292a09920ed66afacd4310563141eb0eddeae29e38dba448188cb2.jpg)  
Fig. 1. RBFN projection head with stacked linear and RBF layers featuring RBF nonlinearities, preceded by an optional batch normalization step (disabled by default).

![](images/bbda13cb5a052866f471a1f0e0c7ff79534d232f32f09eb78149101d6161633f.jpg)  
Fig. 2. Batch-normalized RBFN prediction head used by some architectures after the projector, consisting of a single RBF layer sandwiched between two linear layers.

SNS Metric We propose a novel quality metric called Scale-Normalized Separation (SNS) which is calculated using the center and shape parameters learned by our RBFN projection head during self-supervised training. SNS is a simple and direct measure of kernel separation based on Gaussian interaction energy between kernel centers with scale-normalized distances. Let $\{ c _ { k } \} _ { k = 1 } ^ { K } \subset \mathbb { R } ^ { D }$ denote the set of RBF kernel centers and $\{ \varepsilon _ { k } \} _ { k = 1 } ^ { K }$ their associated kernel scales. For each pair of distinct kernels with $i \neq j$ , we first calculate their dimension-corrected scale-normalized distance as

$$
\delta_ {i, j} = \frac {\parallel c _ {i} - c _ {j} \parallel_ {2}}{\sqrt {D} \sqrt {\varepsilon_ {i} ^ {2} + \varepsilon_ {j} ^ {2} + \mu}}\tag{4}
$$

where the factor $\sqrt { D }$ compensates for the growth of Euclidian distances in high-dimensional spaces and $\mu \in \mathbb { R } _ { > 0 }$ is a small constant added for numerical stability. In order to quantify kernel separation based on these pairwise distances, we use the following Gaussian interaction function:

$$
\psi : \mathbb {R} \to \mathbb {R}, \delta_ {i, j} \mapsto \delta_ {i, j} ^ {2} \exp (- \delta_ {i, j} ^ {2})\tag{5}
$$

Note that the energy of $\psi$ is maximized if $\delta _ { i , j } ~ = ~ 1$ but vanishes as $\delta _ { i , j }$ approaches zero or infinity. Finally, the SNS score is defined as the average interaction energy between all unordered pairs of kernel centers represented by their scale-normalized distances:

$$
\mathrm{SNS} = \frac{1}{K(K - 1)}\sum_{\substack{i,j = 1\\ i\neq j}}^{K}\psi (\delta_{i,j})\tag{6}
$$

Our SNS metric is maximized when inter-center distances are well-matched to kernel scales, i. e. $\delta _ { i , j } \approx 1 \forall i \neq j$ . Further properties of SNS include dimension and scale invariance under joint rescaling of centers and shapes, symmetry due to a pairwise relation between kernel centers and full diferentiability.

Open Images V7 Subsets In order to simplify the exploration of model behavior in terms of class count, we publish [43] a new sampling mechanism for Google’s Open Images V7 dataset (OI-V7) [20] and provide a $\mathrm { P y }$ Torch compatible Dataset implementation with five pre-sampled subsets featuring 10, 30, 50, 70 and 100 classes. Our sampling strategy can be used to generate any balanced subset of the Open Images V7 dataset regardless of class count. However, sample counts per class can vary depending on the requested number of classes, since the smallest class within the original dataset automatically limits the size of all other classes in the subset. To reduce the practical downsides of this approach, we ofer a built-in augmentation pipeline that can optionally augment each class to any size using state-of-the-art techniques like AutoAugment [15], AugMix [24] or RandAugment [16].

In addition to sampling flexibility, our Open Images V7 subsets also offer disjointness to the ImageNet-1K [17, 41] dataset, which is often used by popular machine learning libraries like Torchvision [49] to pre-train architectures like ResNet [23]. Since weight initialization is a widespread technique to boost training eficiency, practitioners have to be especially cautious in interpreting a model’s fitting behavior when training with ImageNet-1K or subsets like ImageNet-100 [14, 48]. That said, judging model performance in a purely comparative setting is usually less problematic, since metrics like accuracy will only increase in absolute terms while preserving relative performance. The third and last advantage of our Open Images V7 subsets over ImageNet-1K is mean sample resolution. On average, samples within our 100 class subset are approximately four times larger in pixel area $( 9 6 4 \times 8 0 0 \mathrm { p x } )$ [43] compared to ImageNet-1K $( 4 8 2 \times 4 1 5 \mathrm { p x } )$ [40]. This property can be beneficial for augmentation heavy architectures like SimCLR, where Gaussian blur is often removed from the standard transformation pipeline when training with low resolution datasets like CIFAR100 $( 3 2 \times 3 2 \mathrm { p x } )$ due to it visually distorting samples beyond recognition. Hence, with increasing image resolution, this and other kinds of heavy augmentation are less likely to compromise sample integrity.

## 4.3 Experimental Setup

In this section, we conduct experiments in order to provide empirical evidence for the following assumptions:

1. Replacing the default MLP projection head with a radial basis function network does not hurt model performance in any significant way.

2. Representation quality can be judged using our SNS metric and the parameters learned by our RBFN projector during self-supervised training.

We assume the first hypothesis to hold since suitably activated RBFN are proven [37] to be universal approximators on a compact subset of <sup>Rn</sup>. Hence, replacing a generic MLP projector with an equally capable RBFN should not impact performance in any meaningful way. However, it should be noted that RBFNs introduce a small set of additional hyperparameters which could either worsen or improve performance, depending on their specific values. The second hypothesis is assumed to be true since nonlinear optimization of the backbone is conducted using some loss function acting directly on the feature embeddings generated by the projector. Therefore, fitting explainable radial basis functions to those projections should provide qualitative insight into the much more complex embedding space learned by the backbone.

Establishing a Baseline In order to compare the eficacy of our RBFN projection heads with default MLP projectors, we first conduct baseline training and evaluation runs on the five well-known SSL architectures MoCo (v2) [12], SimCLR [10], BYOL [21], SwAV [8] and SimSiam [13]. Baseline runs are carried out using the respective architecture’s default projection and prediction head in combination with a ResNet-50 [23] backbone pre-trained on ImageNet-1K [41]. Each architecture is trained for 100 epochs on eight NVIDIA A100 80 GB GPUs in distributed data parallel (DDP) mode using default hyperparameters<sup>6</sup>. For our baseline measurements, we vary only the depth of the default 2-layer MLP projector used by SimCLR and MoCo, as the implementations of BYOL, SwAV and SimSiam do not support this modification. The projector’s depth is first increased to three and finally four layers where possible.

Since we conduct a purely comparative study, we use the well-established ImageNet-100 dataset [48] as our main benchmark. It consists of 100 randomly sampled classes from ImageNet-1K, with approximately 1300 samples per class. Model performance on fewer classes is explored via three image classification subsets generated from Google’s Open Images V7 dataset with 10, 30 and 50 classes [43]. Each dataset is subdivided into train, validation and test splits using a 70/10/20 split ratio. All samples are resized to 224 × 224 px resolution before being normalized using the dataset’s global per channel mean and standard deviation values.

After self-supervised training, we follow the standard protocol for model evaluation by fitting a logistic regression model to the feature vectors generated by the backbone on the labeled test split. The regression model is trained for 200 epochs with a batch size of 64 and a learning rate of 0.001. We finally report accuracy (A), precision (P ), recall (R) and $F _ { 1 }$ score as metrics for the resulting linear classifier. In summary, we collect the results of nine training and evaluation runs for each of the four datasets, yielding a total of 36 baseline measurements.

Training RBFN Projectors For the training runs involving our RBFN projection heads, we use the exact same datasets and hyperparameters as with our baseline MLP projectors. The only major diference is the replacement of the default projection and prediction heads within each architecture with our custom RBFN-based implementations. Since our RBFN heads introduce a small set of new hyperparameters into the training process, we extend our baseline grid search to include not only projector depth but also the number of RBF kernels (128, 256, 512) and diferent radial basis functions (gaussian, inverse\_multiquadric, multiquadric, tps), used within either a standard or normalized RBFN architecture. After training, the backbone is evaluated using the same protocol and metrics as outlined in Section 4.3. Due to an increased number of hyperparameters examined in our grid search, we conduct 216 diferent training runs on each dataset resulting in a total of 864 runs with training durations ranging from approximately 1.5 to 2.5 hours depending on the architecture. In Section 5.2, we analyze the evaluation results using SHAP [31] to derive evidence-based recommendations for suitable default values of the new hyperparameters.

Model Evaluation using SNS Since SNS is computed from kernel centers and shapes, evaluation with this metric is only possible when an RBFN projection head is used during self-supervised training. Hence, the baseline runs described in Section 4.3 using a default MLP projector cannot be evaluated using this method. We instead show the eficacy of SNS by evaluating each RBFN run using both the well-known metrics calculated after logistic regression and SNS, followed by a correlation analysis quantifying the relationship between the two. We argue that high correlation is suitable to support our initial hypothesis that examining trained RBFN projectors via some quality function can provide valuable insights on the backbone’s representation quality. After each of our RBFN training runs, we first evaluate the backbone using LogReg with established classification metrics. We additionally store the final model checkpoint, including the trained backbone and projection head, for further analysis. This procedure allows us to develop and test our SNS metric without having to retrain the entire projector from scratch every time we change our implementation.

In order to evaluate a run using SNS, we load the trained checkpoint’s raw state dictionary and extract kernel centers and shapes from the deepest RBF layer within the RBFN projector:

– If $s _ { i , j } \in \mathbb { R } _ { > 0 }$ denotes the SNS metric calculated for the i-th run on the j-th datasets with $1 \leq i \leq 2 1 6$ and $1 \leq j \leq 4$ , then $\mathbf { S } \in \mathbb { R } _ { > 0 } ^ { 2 1 6 \times 4 }$ is the resulting matrix of calculated SNS values for each run and dataset combination.

– Since we calculate four metrics (accuracy, precision, recall and $F _ { 1 }$ score) on the LogReg model instead of only one, the correlation target is now a tensor $\mathbf { L } \stackrel { \textstyle = } { = } ( l _ { i , k , j } ^ { \top } ) \in \mathbb { R } ^ { 2 1 6 \times 4 \times 4 }$ with $l _ { i , k , j } \in [ 0 , 1 ] \forall i , k , j$ , where the second dimension indexes the aforementioned LogReg metrics and the last the four test datasets.

Hence, for each dataset $j \in \{ 1 , \ldots , 4 \}$ and each metric $k \in \{ 1 , \ldots , 4 \}$ , we now consider the vectors

$$
\mathbf {s} _ {j} = (s _ {i, j}, \ldots , s _ {2 1 6, j}) ^ {\top} \in \mathbb {R} _ {> 0} ^ {2 1 6}, \quad \mathbf {l} _ {k, j} = (l _ {1, k, j}, \ldots , l _ {2 1 6, k, j}) ^ {\top} \in \mathbb {R} _ {[ 0, 1 ]} ^ {2 1 6}\tag{7}
$$

where each ${ \bf s } _ { j }$ remains constant while being correlated two times with four different types of LogReg metrics per dataset. This is done by calculating Pearson’s r [39] and Spearman’s $\rho \ [ 4 5 ]$ correlation coeficients between our SNS and the LogReg metrics as

$$
r _ {k, j} = \mathrm{corr} _ {r} (\mathbf {s} _ {j}, \mathbf {l} _ {k, j}), \quad \rho_ {k, j} = \mathrm{corr} _ {\rho} (\mathbf {s} _ {j}, \mathbf {l} _ {k, j})\tag{8}
$$

yielding a pair of $4 \times 4$ correlation matrices, where the entry at position $( k , j )$ indicates the correlation between SNS and the k-th LogReg metric on the $j { \cdot } \mathrm { t h }$ test dataset. In contrast to $r \in [ - 1 , 1 ]$ , the value of $\rho \in [ - 1 , 1 ]$ is a measure for any nonlinear monotonic relationship between two random variables or samples, whereas r only measures the strength of their linear relationship. Additionally, Pearson’s method also yields a p-value denoted $p _ { k , j }$ testing the null hypothesis of no monotonic association between the inputs.

Table 1. Peak backbone performance by accuracy (A) using default MLP and our RBFN projection heads after LogReg evaluation.

<table><tr><td>Dataset</td><td colspan="8">ImageNet-100</td><td colspan="8">OpenImagesV7-50</td></tr><tr><td>Proj. head</td><td colspan="4">default</td><td colspan="4">rbfn</td><td colspan="4">default</td><td colspan="4">rbfn</td></tr><tr><td>Metric</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td></tr><tr><td>MoCo</td><td>0.92</td><td>0.92</td><td>0.92</td><td>0.92</td><td>0.92</td><td>0.92</td><td>0.93</td><td>0.92</td><td>0.55</td><td>0.58</td><td>0.56</td><td>0.58</td><td>0.53</td><td>0.56</td><td>0.55</td><td>0.55</td></tr><tr><td>SimCLR</td><td>0.88</td><td>0.88</td><td>0.89</td><td>0.88</td><td>0.92</td><td>0.92</td><td>0.93</td><td>0.92</td><td>0.52</td><td>0.54</td><td>0.54</td><td>0.54</td><td>0.51</td><td>0.54</td><td>0.54</td><td>0.54</td></tr><tr><td>BYOL</td><td>0.88</td><td>0.89</td><td>0.90</td><td>0.89</td><td>0.92</td><td>0.92</td><td>0.93</td><td>0.92</td><td>0.49</td><td>0.52</td><td>0.55</td><td>0.52</td><td>0.53</td><td>0.55</td><td>0.55</td><td>0.55</td></tr><tr><td>SwAV</td><td>0.88</td><td>0.88</td><td>0.89</td><td>0.88</td><td>0.89</td><td>0.89</td><td>0.90</td><td>0.89</td><td>0.47</td><td>0.51</td><td>0.48</td><td>0.51</td><td>0.44</td><td>0.49</td><td>0.46</td><td>0.49</td></tr><tr><td>SimSiam</td><td>0.92</td><td>0.92</td><td>0.93</td><td>0.92</td><td>0.92</td><td>0.92</td><td>0.93</td><td>0.92</td><td>0.55</td><td>0.57</td><td>0.56</td><td>0.57</td><td>0.52</td><td>0.55</td><td>0.55</td><td>0.55</td></tr><tr><td>Average</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.90</td><td>0.91</td><td>0.92</td><td>0.92</td><td>0.92</td><td>0.52</td><td>0.55</td><td>0.54</td><td>0.55</td><td>0.51</td><td>0.54</td><td>0.53</td><td>0.54</td></tr><tr><td colspan="17"></td></tr><tr><td>Dataset</td><td colspan="8">OpenImagesV7-30</td><td colspan="8">OpenImagesV7-10</td></tr><tr><td>Proj. Head</td><td colspan="4">default</td><td colspan="4">rbfn</td><td colspan="4">default</td><td colspan="4">rbfn</td></tr><tr><td>Metric</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td></tr><tr><td>MoCo</td><td>0.70</td><td>0.71</td><td>0.71</td><td>0.71</td><td>0.65</td><td>0.67</td><td>0.68</td><td>0.67</td><td>0.76</td><td>0.76</td><td>0.76</td><td>0.76</td><td>0.74</td><td>0.74</td><td>0.75</td><td>0.74</td></tr><tr><td>SimCLR</td><td>0.66</td><td>0.67</td><td>0.67</td><td>0.67</td><td>0.64</td><td>0.66</td><td>0.67</td><td>0.66</td><td>0.73</td><td>0.73</td><td>0.73</td><td>0.73</td><td>0.74</td><td>0.74</td><td>0.74</td><td>0.74</td></tr><tr><td>BYOL</td><td>0.62</td><td>0.65</td><td>0.69</td><td>0.65</td><td>0.65</td><td>0.67</td><td>0.68</td><td>0.67</td><td>0.72</td><td>0.72</td><td>0.72</td><td>0.72</td><td>0.74</td><td>0.74</td><td>0.75</td><td>0.74</td></tr><tr><td>SwAV</td><td>0.63</td><td>0.65</td><td>0.66</td><td>0.65</td><td>0.61</td><td>0.63</td><td>0.64</td><td>0.63</td><td>0.70</td><td>0.70</td><td>0.70</td><td>0.70</td><td>0.70</td><td>0.70</td><td>0.70</td><td>0.70</td></tr><tr><td>SimSiam</td><td>0.68</td><td>0.69</td><td>0.68</td><td>0.69</td><td>0.65</td><td>0.67</td><td>0.68</td><td>0.67</td><td>0.73</td><td>0.73</td><td>0.73</td><td>0.73</td><td>0.74</td><td>0.74</td><td>0.74</td><td>0.74</td></tr><tr><td>Average</td><td>0.66</td><td>0.67</td><td>0.68</td><td>0.67</td><td>0.66</td><td>0.64</td><td>0.67</td><td>0.66</td><td>0.73</td><td>0.73</td><td>0.73</td><td>0.73</td><td>0.73</td><td>0.73</td><td>0.73</td><td>0.73</td></tr></table>

## 5 Results

## 5.1 Projection Head Comparison

When comparing each architecture’s default MLP projection head with our RBFN alternative, no significant performance diference can be observed. In Table 1, we deliberately report the best evaluation runs based on accuracy, since all datasets exhibit nearly perfect inter-class balance. Although our RBFN head outperforms all MLP projectors on ImageNet-100 with up to +4 % accuracy on SimCLR and matches their performance on OpenImagesV7-10, it exhibits a modest performance deficit of 1–3 % on OpenImagesV7-50 and OpenImagesV7-30. However, our RBFN projector is particularly efective when used in conjunction with BYOL, where it manages to increase peak backbone performance by about 2-4 % across all datasets. On average, our RBFN projector is only outperformed by about half a percentage point in accuracy across diferent architectures and test datasets, which can be easily attributed to noise within our measurements.

## 5.2 Impact of RBFN Parameters

The results of our SHAP analysis (see Table 2), which quantify the impact of RBFN parameterization as discussed in Section 4.3, indicate that RBFN projection heads should generally be activated using the Gaussian radial basis function. Our experiments also show that increasing the projector’s depth to four layers without normalizing the RBFN’s architecture allows the thin place spine to be used as an alternative activation function without compromising the backbone’s performance. Based on our experiments with SimCLR, BYOL and SwAV, we generally advise against using multiquadric and inverse multiquadric radial basis functions, as these can lead to significant performance losses and even trigger model collapse. In contrast to the radial basis function’s profound impact on model performance, we do not record any significant fluctuations in backbone accuracy when varying the number of kernels fitted within each RBF layer. We therefore recommend adjusting the number of RBF kernels to match the number of compressed output features generated by the projector, which is typically 128.

Table 2. Overall impact of RBFN hyperparameters on backbone accuracy (A) after LogReg evaluation based on SHAP analysis. Let $\overline { { { \phi } _ { i } ^ { + } } }$ denote the mean of the positive SHAP values and $\overline { { { \phi } _ { i } ^ { - } } }$ the mean value of the negative SHAP values for hyperparameter i. If $\overline { { \phi _ { i } ^ { + } } } > | \overline { { \phi _ { i } ^ { - } } }$ |, the hyperparameter exerts a net positive influence on prediction accuracy, while |ϕ<sub>i</sub>| quantifies the strength of its overall efect.

<table><tr><td colspan="2">Dataset</td><td colspan="3">ImageNet100</td><td colspan="3">OI-V7-50</td><td colspan="3">OI-V7-30</td><td colspan="3">OI-V7-10</td></tr><tr><td colspan="2">Mean SHAP</td><td> $\overline{|\phi_i|}$ </td><td> $\phi_i^+$ </td><td> $\overline{\phi_i^-}$ </td><td> $\overline{|\phi_i|}$ </td><td> $\overline{\phi_i^+}$ </td><td> $\overline{\phi_i^-}$ </td><td> $\overline{|\phi_i|}$ </td><td> $\overline{\phi_i^+}$ </td><td> $\overline{\phi_i^-}$ </td><td> $\overline{|\phi_i|}$ </td><td> $\overline{\phi_i^+}$ </td><td> $\overline{\phi_i^-}$ </td></tr><tr><td rowspan="5">RBF layer</td><td>gaussian</td><td>0.11</td><td>0.22</td><td>-0.07</td><td>0.07</td><td>0.15</td><td>-0.04</td><td>0.08</td><td>0.16</td><td>-0.05</td><td>0.07</td><td>0.15</td><td>-0.05</td></tr><tr><td>tps</td><td>0.02</td><td>0.05</td><td>-0.02</td><td>0.02</td><td>0.05</td><td>-0.01</td><td>0.02</td><td>0.05</td><td>-0.01</td><td>0.02</td><td>0.05</td><td>-0.02</td></tr><tr><td>multiquadric</td><td>0.00</td><td>0.00</td><td>-0.00</td><td>0.00</td><td>0.00</td><td>-0.00</td><td>0.00</td><td>0.00</td><td>-0.00</td><td>0.00</td><td>0.00</td><td>-0.00</td></tr><tr><td> $multiquadric^{-1}$ </td><td>0.03</td><td>0.02</td><td>-0.06</td><td>0.00</td><td>0.00</td><td>-0.00</td><td>0.00</td><td>0.00</td><td>-0.01</td><td>0.00</td><td>0.00</td><td>-0.01</td></tr><tr><td>num. kernels</td><td>0.02</td><td>0.02</td><td>-0.02</td><td>0.01</td><td>0.01</td><td>-0.01</td><td>0.02</td><td>0.01</td><td>-0.03</td><td>0.01</td><td>0.01</td><td>-0.01</td></tr><tr><td rowspan="2">RBFN head</td><td>num. layers</td><td>0.04</td><td>0.08</td><td>-0.03</td><td>0.02</td><td>0.03</td><td>-0.01</td><td>0.02</td><td>0.03</td><td>-0.02</td><td>0.03</td><td>0.04</td><td>-0.02</td></tr><tr><td>normalize</td><td>0.04</td><td>0.03</td><td>-0.04</td><td>0.02</td><td>0.02</td><td>-0.02</td><td>0.02</td><td>0.02</td><td>-0.02</td><td>0.02</td><td>0.02</td><td>-0.02</td></tr></table>

With regards to the parameters that afect not only the RBF layers within the projector but the projector itself, our SHAP analysis confirms the generally observed benefit of increased projector depth [11]. We find that slightly deeper projectors with three rather than two layers perform better on average, while even deeper RBFN projectors with four layers yield only marginal gains. Hence, we advise constructing RBFN projection heads with three instead of only two layers. Lastly, our SHAP results indicate, that RBFN normalization does not impact model performance in any significant way. However, as previously illustrated with tps activation, some hyperparameter combinations might still require the RBF network to be normalized to achieve optimal performance. Since we do not find any clear advantages of this hyperparameter, we decide to eliminate it by advising against the use of normalization.

## 5.3 Correlation of Classification Metrics with SNS

After applying our SNS-based model evaluation protocol described in Section 4.3, we compile our correlation results in Table 3. The obtained correlation coeficients indicate a strong (0.60–0.79) to very strong (0.80–1.00) positive linear correlation between our SNS metric and all four established classification metrics computed after logistic regression. Additionally, all $p _ { k , j } .$ -values obtained using Spearman’s rank correlation are close to zero, providing strong evidence against the null hypothesis and supporting a monotonic relationship between SNS and LogReg results.

Table 3. Pearson $( r _ { k , j } )$ and Spearman $( \rho _ { k , j } )$ correlation of our SNS metric with established performance indicators calculated after logistic regression.

<table><tr><td>Dataset (j)</td><td colspan="4">ImageNet-100</td><td colspan="4">OpenImagesV7-50</td></tr><tr><td>Metric (k)</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td></tr><tr><td> $r_{k,j}$ </td><td>0.87</td><td>0.87</td><td>0.85</td><td>0.87</td><td>0.93</td><td>0.92</td><td>0.92</td><td>0.92</td></tr><tr><td> $\rho_{k,j}$ </td><td>0.85</td><td>0.85</td><td>0.85</td><td>0.85</td><td>0.83</td><td>0.83</td><td>0.69</td><td>0.83</td></tr><tr><td> $p_{k,j}$ </td><td> $1 \times 10^{-61}$ </td><td> $8 \times 10^{-61}$ </td><td> $2 \times 10^{-61}$ </td><td> $8 \times 10^{-61}$ </td><td> $1 \times 10^{-55}$ </td><td> $4 \times 10^{-56}$ </td><td> $2 \times 10^{-32}$ </td><td> $4 \times 10^{-56}$ </td></tr><tr><td>Dataset (j)</td><td colspan="4">OpenImagesV7-30</td><td colspan="4">OpenImagesV7-10</td></tr><tr><td>Metric (k)</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td><td> $F_1$ </td><td>A</td><td>P</td><td>R</td></tr><tr><td> $r_{k,j}$ </td><td>0.88</td><td>0.87</td><td>0.87</td><td>0.87</td><td>0.81</td><td>0.81</td><td>0.79</td><td>0.81</td></tr><tr><td> $\rho_{k,j}$ </td><td>0.76</td><td>0.76</td><td>0.72</td><td>0.76</td><td>0.83</td><td>0.83</td><td>0.82</td><td>0.83</td></tr><tr><td> $p_{k,j}$ </td><td> $1 \times 10^{-41}$ </td><td> $1 \times 10^{-41}$ </td><td> $4 \times 10^{-35}$ </td><td> $1 \times 10^{-41}$ </td><td> $1 \times 10^{-56}$ </td><td> $2 \times 10^{-57}$ </td><td> $1 \times 10^{-52}$ </td><td> $2 \times 10^{-57}$ </td></tr></table>

## 6 Discussion

After analyzing the results presented in Table 1, we come to the conclusion that shallow radial basis function networks with two to four RBF layers are a competitive replacement for most generic MLP projection heads used in self-supervised learning. This empirical observation supports our first hypothesis that replacing the default MLP with our RBFN projector does not significantly hurt performance. This outcome was expected, since RBFNs are proven universal approximators. Their hidden neurons correspond to localized kernel centers in the input space, making their decision process more transparent and interpretable than that of conventional MLPs. As a result, RBFNs can be regarded as a highly explainable type of multilayer perceptron that fits kernel centers to inputs using nonlinear RBF activation functions.

Once we have determined RBFNs to be a suitable alternative for MLP projectors, we continue to investigate their construction in terms of hyperparameter choice (see Table 2). SHAP analysis suggests the Gaussian radial basis function to be the preferred method for activating our RBFN heads. Due to SSL being linked to clustering of feature vectors in latent space, our rationale for this finding is primarily based on the parameters learned by the RBFN during endto-end training. We hypothesize that a multivariate normal distribution, being equivalent to a hyperdimensional Gaussian, provides the most flexible geometric shape for fitting clusters in feature space using shape $\left( \varepsilon _ { i } \right)$ and center (c<sub>i</sub>) parameters. In particular, the Gaussian RBF is highly localized due to its exponential decay as the distance r between embeddings increases. The advantage of this property becomes obvious once applied to clustering, since points further away from center $\mathbf { c } _ { i }$ should have reduced afinity and therefore a lower probability of belonging to the same cluster. Additionally, the learned shape parameters $\varepsilon _ { i } ,$ which mathematically control the Gaussian’s width, can be used to introduce the notion of neighborhood, essentially limiting the projections that should be included into the cluster centered at $\mathbf { c } _ { i }$

Further experiments on constructing our RBF layer with 128, 256 and 512 kernels show no significant impact on model performance. We deliberately choose 128 kernels as our lower bound to match the number of output features typically generated by the projection head. After that, we double the number of kernels twice to accommodate architectures such as SimSiam, which produce higherdimensional projections. We do not find any evidence that the optimal number of RBF kernels is tied to the projector’s output dimensions. Instead, we suspect that this parameter has no efect in our experiments, since even the smallest number of 128 kernels could have been suficient to fully saturate the projector. Hence, additional grid searches with a significantly reduced number of kernels are required to verify this hypothesis. Even so, further research is needed to identify the key factors governing the optimal number of kernels in an RBF layer.

Turning to the RBFN projection head itself, we observe a clear benefit from increasing the number of RBF layers from two to three. This phenomenon could be explained by the additional layer increasing the projector’s bufering capacity through better isolation of the objective function from the outputs of the backbone, which are known to be less suitable for loss calculation. Instead, a third layer could allow the projections to distort more favorably in terms of the loss function, without hurting the valuable representations learned by the backbone. Additionally, deeper projectors can have a regularization-like efect on the backbone and could therefore prevent overfitting of the objective function. Lastly, we find normalization of our RBFN architecture to be inefective, which could be explained by the fact that all samples were already normalized prior to training and testing using their associated dataset’s mean and standard deviation values.

Finally, we test our second hypothesis, which states that representation quality can be inferred from learned RBFN parameters. To this end, we evaluate each RBFN training run using our novel SNS metric and established linear classification metrics, followed by a correlation analysis of the resulting scores. Since both Pearson and Spearman correlation coeficients shown in Table 3 indicate a strong to very strong positive linear relationship between our SNS and LogReg metrics, we conclude that our second hypothesis also holds. Hence, a trained RBFN projector can be used as a proxy for judging backbone quality without the need for training a separate logistic regression model with labeled samples. We believe that correlation can be increased even further by either developing more sophisticated alternatives to our SNS metric or additional enhancements to our RBFN projector, improving the fit of kernel centers (c<sub>i</sub>) and shapes (ε<sub>i</sub>) to the representations in projection space.

Acknowledgements. This work was funded by the Lower Saxony Ministry of Science and Culture as well as the Volkswagen Foundation as part of the research project “RLA - KI Reallabor Agrar” under grant number ZN4530.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Agrawal, K.K., Mondal, A.K., Ghosh, A., Richards, B.A.: α-ReQ: Assessing representation quality by measuring eigenspectrum decay. In: Proc. International Conference on Neural Information Processing Systems. NIPS ’22, Curran Associates Inc., Red Hook, NY, USA (2022), https://dl.acm.org/doi/10.5555/3600270. 3601551

2. Amirian, M., Schwenker, F.: Radial Basis Function Networks for Convolutional Neural Networks to Learn Similarity Distance Metric and Improve Interpretability. IEEE Access 8, 123087–123097 (2020). https://doi.org/10.1109/ACCESS.2020. 3007337

3. Bardes, A., Ponce, J., LeCun, Y.: VICReg: Variance-Invariance-Covariance Regularization for Self-Supervised Learning. In: Proc. International Conference on Learning Representations, ICLR. OpenReview.net (2022). https://doi.org/10. 48550/arXiv.2105.04906

4. Ben-Shaul, I., Shwartz-Ziv, R., Galanti, T., Dekel, S., LeCun, Y.: Reverse Engineering Self-Supervised Learning. In: Oh, A., Naumann, T., Globerson, A., Saenko, K., Hardt, M., Levine, S. (eds.) Proc. International Conference on Neural Information Processing Systems (NeurIPS) (2023). https://doi.org/10.48550/arXiv.2305.15614

5. Bordes, F., Balestriero, R., Garrido, Q., Bardes, A., Vincent, P.: Guillotine Regularization: Why removing layers is needed to improve generalization in Self-Supervised Learning. Transactions on Machine Learning Research 2023 (2023). https://doi.org/10.48550/arXiv.2206.13378

6. Bromley, J., Guyon, I., LeCun, Y., Säckinger, E., Shah, R.: Signature Verification using a ”Siamese” Time Delay Neural Network. In: Advances in Neural Information Processing Systems. vol. 6 (1993), https://dl.acm.org/doi/10.5555/2987189. 2987282

7. Broomhead, D.S., Lowe, D.: Multivariable Functional Interpolation and Adaptive Networks. Complex Systems 2(3) (1988), https://www.complex-systems.com/ abstracts/v02\_i03\_a05/, accessed: 2026-06-02

8. Caron, M., Misra, I., Mairal, J., Goyal, P., Bojanowski, P., Joulin, A.: Unsupervised Learning of Visual Features by Contrasting Cluster Assignments. In: Proc. International Conference on Neural Information Processing Systems. NIPS ’20, Curran Associates Inc., Red Hook, NY, USA (2020), https://dl.acm.org/doi/10. 5555/3495724.3496555

9. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging Properties in Self-Supervised Vision Transformers. In: 2021 IEEE/CVF International Conference on Computer Vision, ICCV 2021, Montreal, QC, Canada, October 10-17, 2021. pp. 9630–9640. IEEE (2021). https: //doi.org/10.1109/ICCV48922.2021.00951

10. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A Simple Framework for Contrastive Learning of Visual Representations. In: Proceedings of the 37th International Conference on Machine Learning. ICML’20, JMLR.org (2020), https: //dl.acm.org/doi/10.5555/3524938.3525087

11. Chen, T., Kornblith, S., Swersky, K., Norouzi, M., Hinton, G.: Big Self-Supervised Models are Strong Semi-Supervised Learners. In: Proc. International Conference on Neural Information Processing Systems. NIPS ’20, Curran Associates Inc., Red Hook, NY, USA (2020), https://dl.acm.org/doi/10.5555/3495724.3497589

12. Chen, X., Fan, H., Girshick, R.B., He, K.: Improved Baselines with Momentum Contrastive Learning. CoRR abs/2003.04297 (2020). https://doi.org/10.48550/ arXiv.2003.04297

13. Chen, X., He, K.: Exploring Simple Siamese Representation Learning. In: 2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 15745–15753 (2021). https://doi.org/10.1109/CVPR46437.2021.01549

14. Chun-Hsiao Yeh, Y.C.: IN100pytorch: PyTorch Implementation: Training ResNets on ImageNet-100 (2022), https://github.com/danielchyeh/ImageNet-100-Pytorch, accessed: 2026-06-02

15. Cubuk, E.D., Zoph, B., Mané, D., Vasudevan, V., Le, Q.V.: AutoAugment: Learning Augmentation Strategies From Data. In: Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 113–123 (2019). https: //doi.org/10.1109/CVPR.2019.00020

16. Cubuk, E.D., Zoph, B., Shlens, J., Le, Q.V.: RandAugment: Practical automated data augmentation with a reduced search space. In: 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW). pp. 3008–3017 (2020). https://doi.org/10.1109/CVPRW50498.2020.00359

17. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: ImageNet: A Large-Scale Hierarchical Image Database. In: Proc. IEEE Conference on Computer Vision and Pattern Recognition. pp. 248–255 (2009). https://doi.org/10.1109/CVPR. 2009.5206848

18. Deng, W., Zheng, L.: Are Labels Always Necessary for Classifier Accuracy Evaluation? In: Proc. IEEE Conference on Computer Vision and Pattern Recognition, CVPR. pp. 15069–15078. Computer Vision Foundation / IEEE (2021). https://doi.org/10.1109/CVPR46437.2021.01482

19. Garrido, Q., Balestriero, R., Najman, L., LeCun, Y.: RankMe: Assessing the Downstream Performance of Pretrained Self-supervised Representations by Their Rank. In: Proc. International Conference on Machine Learning. ICML’23, JMLR.org (2023), http://dl.acm.org/doi/10.5555/3618408.3618848

20. Google LLC: Open Images V7 (2022), https://storage.googleapis.com/ openimages/web/factsfigures\_v7.html, accessed: 2026-06-02

21. Grill, J.B., Strub, F., Altché, F., Tallec, C., Richemond, P.H., Buchatskaya, E., Doersch, C., Pires, B.A., Guo, Z.D., Azar, M.G., Piot, B., Kavukcuoglu, K., Munos, R., Valko, M.: Bootstrap Your Own Latent - A New Approach to Self-Supervised Learning. In: Proc. International Conference on Neural Information Processing Systems. NIPS ’20, Curran Associates Inc., Red Hook, NY, USA (2020), http: //dl.acm.org/doi/abs/10.5555/3495724.3497510

22. Gupta, K., Ajanthan, T., van den Hengel, A., Gould, S.: Understanding and Improving the Role of Projection Head in Self-Supervised Learning. CoRR abs/2212.11491 (2022). https://doi.org/10.48550/arXiv.2212.11491

23. He, K., Zhang, X., Ren, S., Sun, J.: Deep Residual Learning for Image Recognition. In: 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR). pp. 770–778 (2016). https://doi.org/10.1109/CVPR.2016.90

24. Hendrycks, D., Mu, N., Cubuk, E.D., Zoph, B., Gilmer, J., Lakshminarayanan, B.: AugMix: A Simple Data Processing Method to Improve Robustness and Uncertainty. In: Proc. International Conference on Learning Representations, ICLR. OpenReview.net (2020). https://doi.org/10.48550/arXiv.1912.02781

25. Jang, J.S., Sun, C.T.: Functional Equivalence Between Radial Basis Function Networks and Fuzzy Inference Systems. IEEE Transactions on Neural Networks 4(1), 156–159 (1993). https://doi.org/10.1109/72.182710

26. Jing, L., Vincent, P., LeCun, Y., Tian, Y.: Understanding Dimensional Collapse in Contrastive Self-supervised Learning. In: Proc. International Conference on Learning Representations, ICLR. OpenReview.net (2022). https://doi.org/10.48550/ arXiv.2110.09348

27. Johnson, D.D., Hanchi, A.E., Maddison, C.J.: Contrastive Learning Can Find An Optimal Basis For Approximately View-Invariant Functions. In: Proc. International Conference on Learning Representations, ICLR. OpenReview.net (2023). https://doi.org/10.48550/arXiv.2210.01883

28. Kornblith, S., Shlens, J., Le, Q.V.: Do Better ImageNet Models Transfer Better? In: 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 2656–2666 (2019). https://doi.org/10.1109/CVPR.2019.00277

29. Lee, J., Bahri, Y., Novak, R., Schoenholz, S.S., Pennington, J., Sohl-Dickstein, J.: Deep Neural Networks as Gaussian Processes. In: Proc. International Conference on Learning Representations, ICLR. OpenReview.net (2018). https://doi.org/10. 48550/arXiv.1711.00165

30. Li, Y., Pogodin, R., Sutherland, D.J., Gretton, A.: Self-Supervised Learning with Kernel Dependence Maximization. In: Proc. International Conference on Neural Information Processing Systems. NIPS ’21, Curran Associates Inc., Red Hook, NY, USA (2021), http://dl.acm.org/doi/10.5555/3540261.3541451

31. Lundberg, S.M., Lee, S.I.: A Unified Approach to Interpreting Model Predictions. In: Proc. International Conference on Neural Information Processing Systems. p. 4768–4777. NIPS’17, Curran Associates Inc., Red Hook, NY, USA (2017), https: //dl.acm.org/doi/10.5555/3295222.3295230

32. Ma, J., Hu, T., Wang, W.: Deciphering the Projection Head: Representation Evaluation Self-supervised Learning. In: Proc. International Joint Conference on Artificial Intelligence. IJCAI ’24 (2024). https://doi.org/10.24963/ijcai.2024/522

33. Micchelli, C.A.: Interpolation of Scattered Data: Distance Matrices and Conditionally Positive Definite Functions. Constructive Approximation 2(1), 11–22 (1986). https://doi.org/10.1007/BF01893414

34. Nandam, S.R., Atito, S., Feng, Z., Kittler, J., Awais, M.: Investigating Self-Supervised Methods for Label-Eficient Learning. Int. J. Comput. Vision 133(7), 4522–4537 (Mar 2025). https://doi.org/10.1007/s11263-025-02397-4

35. van den Oord, A., Li, Y., Vinyals, O.: Representation Learning with Contrastive Predictive Coding. CoRR abs/1807.03748 (2018). https://doi.org/10.48550/ arXiv.1807.03748

36. Orr, M.J.L.: Introduction to Radial Basis Function Networks. Tech. rep., Centre for Cognitive Science, University of Edinburgh (April 1996), https://faculty.cc.gatech. edu/\~isbell/tutorials/rbf-intro.pdf, accessed: 2026-06-05

37. Park, J., Sandberg, I.W.: Universal Approximation Using Radial-Basis-Function Networks. Neural Computation 3(2), 246–257 (06 1991). https://doi.org/10.1162 neco.1991.3.2.246

38. Parulekar, A., Collins, L., Shanmugam, K., Mokhtari, A., Shakkottai, S.: InfoNCE Loss Provably Learns Cluster-Preserving Representations. In: Neu, G., Rosasco, L. (eds.) Proc. International Conference on Learning Theory. Proceedings of Machine Learning Research, vol. 195, pp. 1914–1961. PMLR (12–15 Jul 2023). https://doi. org/10.48550/arXiv.2302.07920

39. Pearson, K.: VII. Note on Regression and Inheritance in the Case of Two Parents. Proceedings of the Royal Society of London 58(347-352), 240–242 (12 1895). https: //doi.org/10.1098/rspl.1895.0041

40. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A.C., Fei-Fei, L.: ImageNet Large Scale Visual Recognition Challenge 2013 (ILSVRC2013) (2013), https: //www.image-net.org/challenges/LSVRC/2013/index, accessed: 2026-06-02

41. Russakovsky, O., Deng, J., Su, H., Krause, J., Satheesh, S., Ma, S., Huang, Z., Karpathy, A., Khosla, A., Bernstein, M., Berg, A.C., Fei-Fei, L.: ImageNet Large Scale Visual Recognition Challenge. International Journal of Computer Vision (IJCV) 115(3), 211–252 (2015). https://doi.org/10.1007/s11263-015-0816-y

42. Russo, A.: Pytorch RBF Layer (2021), https://github.com/rssalessio/ PytorchRBFLayer, accessed: 2026-06-02

43. Schliebitz, A., Tapken, H., Atzmueller, M.: The OpenImagesV7-100 Dataset (Dec 2025), https://github.com/andreas-schliebitz/open-images-v7-100, accessed: 2026-06-02

44. Song, Z., Su, X., Wang, J., Qiang, W., Zheng, C., Sun, F.: Towards the Sparseness of Projection Head in Self-Supervised Learning. CoRR abs/2307.08913 (2023). https://doi.org/10.48550/arXiv.2307.08913

45. Spearman, C.: The Proof and Measurement of Association between Two Things. The American Journal of Psychology 15(1), 72–101 (1904), http://www.jstor.org/ stable/1412159

46. Susmelj, I., Heller, M., Wirth, P., Prescott, J., Ebner, M., et al.: Lightly, https: //github.com/lightly-ai/lightly, accessed: 2026-06-02

47. Thilak, V., Huang, C., Saremi, O., Dinh, L., Goh, H., Nakkiran, P., Susskind, J.M., Littwin, E.: LiDAR: Sensing Linear Probing Performance in Joint Embedding SSL Architectures. In: Proc. International Conference on Learning Representations, ICLR. OpenReview.net (2024). https://doi.org/10.48550/arXiv.2312.04000

48. Tian, Y., Krishnan, D., Isola, P.: Contrastive Multiview Coding. In: Proc European Conference on Computer Vision (ECCV). p. 776–794. Springer-Verlag, Berlin, Heidelberg (2020). https://doi.org/10.1007/978-3-030-58621-8\_45

49. TorchVision maintainers and contributors: TorchVision: PyTorch’s Computer Vision library (Nov 2016), https://github.com/pytorch/vision, accessed: 2026-06-02

50. Wang, T., Isola, P.: Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere. In: Proc. International Conference on Machine Learning. ICML’20, JMLR.org (2020), https://dl.acm.org/doi/ 10.5555/3524938.3525859

51. Wilson, A.G., Hu, Z., Salakhutdinov, R., Xing, E.P.: Deep Kernel Learning. In: Gretton, A., Robert, C.C. (eds.) Proc. International Conference on Artificial Intelligence and Statistics. vol. 51, pp. 370–378. PMLR, Cadiz, Spain (09–11 May 2016). https://doi.org/10.48550/arXiv.1511.02222

52. Wittscher, L.: A survey on design choices for self-supervised learning in computer vision. Artificial Intelligence Review 59(4), 112 (Feb 2026). https://doi.org/10. 1007/s10462-026-11506-9

53. Xue, Y., Gan, E., Ni, J., Joshi, S., Mirzasoleiman, B.: Investigating the Benefits of Projection Head for Representation Learning. In: Proc. International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net (2024). https://doi.org/10.48550/arXiv.2403.11391

54. Zadeh, P.H., Hosseini, R., Sra, S.: Deep-RBF Networks Revisited: Robust Classification with Rejection. CoRR abs/1812.03190 (2018). https://doi.org/10.48550/ arXiv.1812.03190

55. Zbontar, J., Jing, L., Misra, I., LeCun, Y., Deny, S.: Barlow Twins: Self-Supervised Learning via Redundancy Reduction. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 12310–12320. PMLR (18–24 Jul 2021). https: //doi.org/10.48550/arXiv.2103.03230