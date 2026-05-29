# A self-supervised learning approach to deep filter banks for texture recognition

Florindo, Joao B.b, Lyra, Lucas O.a, Fabris, Antonio E.a

aInstitute of Mathematics and Statistics of the University of Sao Paulo, Rua do Matao, 1010, Sao Paulo, 05508-090, Sao Paulo, Brazil bInstitute of Mathematics, Statistics and Scientific Computing of the University of Campinas, Rua Sergio Buarque de Holanda, 651, Campinas, 13083-859, Sao Paulo, Brazil

# Abstract

An important challenge in texture recognition is the limited amount of data for training frequently found in real-world applications. In computer vision in general, a successful strategy to mitigate this issue is the use of a pretraining stage where the neural network learns to identify relations between parts of the data in a self-supervised manner. A well-established framework in this direction is masked autoencoder. Nevertheless, these models usually rely on computationally intensive architectures, such as vision transformers. In the particular case of texture images, most of the relevant information is compacted within a delimited area around each pixel, which suggests that capturing longrange dependence via the attention mechanism may be unnecessary. Based on that assumption, here we propose a framework where the pretraining model is a convolutional autoencoder. To leverage the rich information conveyed by texture patterns, we employ deep filters coupled with Fisher vector pooling. In this way, we improve the performance of texture recognition without adding significant computational burden. Our approach is compared with several stateof-the-art methods in different texture databases, confirming its potential both in terms of classification accuracy and computational complexity.

Keywords: Self-supervised learning, Texture recognition, Autoencoder, Fisher vector, Convolutional neural newtworks

# 1. Introduction

Texture recognition is a fundamental task in computer vision, with applications to diverse areas, such as in medicine [1], agriculture [2], material sciences [3], remote sensing [4], and many others.

Despite the efforts in the literature to improve the performance of deep learning methods in this problem, we still have some challenges in most real-world applications of texture analysis, e.g., the usually reduced amount of data for training, high interclass similarity and intraclass dissimilarity. In this scenario, the use “off-the-shelf” of convolutional neural networks (CNN) does not achieve optimal results. Most texture recognition methodologies rely on extracting more information from the training data than a CNN can typically represent. And a modern strategy that has been successful in situations like that is the use of a self-supervised pretraining stage [5].

Self-supervised algorithms in computer vision can be usually grouped into contrastive, clustering-based, and autoassociative learning. The last category, in particular, is usually more effective in feature extraction, which is a crucial step in texture recognition algorithms. A prototypical representer of autoassociative model is masked auto-encoder (MAE) [6]. Its pretraining task consists of reconstructing images with randomly masked patches. Such reconstruction requires the understanding of long-range relations within the image, which motivates the use of Vision Transforms (ViT) [7] instead of convolutional networks. On the other hand, using ViT adds significant computational burden to the pretraining stage and useful deep representations of texture images are known for a long time to be highly dependent on local characteristics.

Based on that, here we propose a convolutional auto-encoder for the pretraining stage of texture recognition tasks. Two approaches are explored: 1) The input and output of the autoencoder are the same image; 2) The input corresponds to a noisy version of the original image, which should be recovered at the output. In summary, the main contributions of this manuscript are:

• We propose a deep convolutional encoder-decoder architecture that extracts meaningful features from texture images in a self-supervised manner.

• A model for texture recognition is introduced, using a combination of self-supervised CNN and Fisher Vectors.   
• We advance the current state-of-the-art results on the Flickr Material Database (FMD) [29], the Describable Textures Dataset (DTD) [30], and the KTH-TIPS2-b image database [28].   
• The proposed model demonstrates outstanding accuracy in the practical task of identifying Brazilian plant species, substantially surpassing previously reported results in the literature.

Section 2 reviews the literature relevant to our research. In Section 3, we outline the theoretical foundations necessary for understanding our method. Section 4 provides a detailed description of our proposed approach for medical image classification. The experimental setup, including dataset descriptions and implementation specifics, is presented in Section 5. Section 6 presents and analyzes our experimental findings. Finally, Section 7 offers concluding remarks and suggestions for future research directions.

# 2. Related works

In this section, we provide a brief overview of studies relevant to our work. We begin by examining the literature on autoencoders, then discuss orderless encoding methods, and finally review general approaches to texture recognition.

# 2.1. Autoencoders

Despite being an algorithm typically associated with unsupervised learning, we can find examples of the use of autoencoders on image recognition.

In [9], an autoencoder is employed for representation learning. The extracted features provide interesting results in character recognition even when applied to simple classifiers like nearest neighbors. The authors in [10] propose the concept of orthogonal autoencoder regression, where the representation capability of autoencoders is combined with least square regression for image classification. Autoencoders are also frequently used as an auxiliary tool to improve the classification pipeline in general. For example, in [11], a masked autoencoder is used for debluring of ultrasound images.

# 2.2. Orderless Encoding

In visual texture classification, the scarcity of labeled data and the domain gap from ImageNet have motivated many researchers to leverage pre-trained CNN weights. Orderless encoding methods are particularly well-suited for this field and are widely adopted. Cimpoi et al. [12] introduced the use of Fisher Vectors to encode features extracted from the final convolutional layer of a VGG network [13]. Building on this idea, Lyra et al. [8] developed a novel approach for extracting features from multiple convolutional layers across various CNN architectures, achieving improved accuracy compared to using CNNs alone.

Beyond Fisher Vectors, several other orderless encoding techniques have been explored. Chen et al. [14] also extract features from multiple convolutional layers, but concatenate the feature maps by upsampling lower-resolution maps using bilinear interpolation. They then assess statistical self-similarity with a differential box counting method and compute soft histograms, which are concatenated with average pooling to form the final image descriptor. RADAM [15] employs a similar cross-layer concatenation strategy, but generates the image descriptor using a randomized autoencoder. In [16], first- and second-order statistics are extracted from feature maps using a frequency attention mechanism and encoded with bilinear models. Xu et al. [17] propose combining fractal average pooling with global average pooling to create more robust descriptors. Florindo et al. [18] compute descriptors using fuzzy equivalence measures applied to clustered local features.

# 2.3. Texture Recognition

Recent advances in texture recognition have been driven by developments in deep learning architectures, feature extraction strategies, and hybrid modeling approaches.

The introduction of Vision Transformers (ViTs) has brought new perspectives to texture recognition. Scabini et al. [19] conducted an extensive evaluation of 21 ViT variants, including ViT-B with DINO pre-training, BeiT v2, and Swin Transformers. Their findings indicate that ViTs generally surpass traditional CNNs and hand-crafted models, especially for in-the-wild texture tasks. EfficientFormer was identified as a cost-effective alternative, delivering strong performance with lower computational requirements. Scabini et al. [15] also proposed RADAM (Randomized Aggregated Deep Activation Maps), a method that encodes texture representations without fine-tuning the backbone. RADAM uses a Randomized Autoencoder (RAE) trained locally on each image to process outputs from various depths of a pre-trained CNN, producing a 1D texture representation classified by a linear SVM. This approach achieves stateof-the-art results on multiple texture benchmarks, demonstrating the power of pre-trained backbones without additional fine-tuning.

To address CNNs’ limitations in capturing fine local details, Zhu et al. [20] integrated Gabor filters into CNN architectures. Their method introduces a texture branch that extracts multi-frequency features using Gabor filters, complemented by a statistical feature extractor and a gate selection mechanism. This hybrid design enhances recognition of fine-grained categories, achieving state-of-the-art results on datasets such as CUB-200-2011 and GTOS-mobile.

Bera et al. [21] proposed a fusion approach that combines global texture features with local patch-based information for fine-grained image classification. Their method extracts deep features from fixed-size, non-overlapping patches, encodes them with LSTM, and computes image-level textures using Local Binary Patterns (LBP). The integration of these streams yields an efficient feature vector, leading to improved classification accuracy across diverse datasets, including those for human faces and skin lesions.

Goyal et al. [22] investigated transfer learning for texture classification using pre-trained models such as MobileNetV3 and InceptionV3. Their experiments on datasets like Brodatz, Kylberg, and Outex demonstrated high classification accuracy, underscoring the effectiveness of transfer learning in texture analysis.

# 3. Background

In this section, we outline two fundamental principles underlying our methodology. The first one is the autoencoder, which handles feature extraction. The second is Fisher Vector encoding, which generates the feature vectors used for classification.

# 3.1. Autoencoder

The encoder mathematically performs a series of operations:

$$
E _ {i} (x) = \sigma (W _ {i, 2} * \sigma (W _ {i, 1} * E _ {i - 1} (x) + b _ {i, 1}) + b _ {i, 2}) \tag {1}
$$

$$
D _ {i} (x) = P (E _ {i} (x)),
$$

where $E _ { i }$ represents the output of the i-th encoder block, $D _ { i }$ represents the downsampled output after pooling, $W _ { i , j }$ are the convolution kernels $\left( 3 \times 3 \right)$ , $b _ { i , j }$ are bias terms, $\sigma$ is the ReLU activation function, P is the max pooling operation $( 2 \times 2$ with stride 2).

The bottleneck connects the encoder and decoder:

$$
B (x) = \sigma (W _ {b, 2} * \sigma (W _ {b, 1} * D _ {J} (x) + b _ {b, 1}) + b _ {b, 2}), \tag {2}
$$

where J is the final encoder level and B is the bottleneck output.

The decoder performs upsampling followed by convolutions:

$$
U _ {i} (x) = W _ {u, i} * U (D _ {J - i + 1} (x))
$$

$$
C _ {i} (x) = \operatorname{concat} \left(E _ {J - i + 1} (x), U _ {i} (x)\right) \tag {3}
$$

$$
F _ {i} (x) = \sigma (W _ {d, i, 2} * \sigma (W _ {d, i, 1} * C _ {i} (x) + b _ {d, i, 1}) + b _ {d, i, 2}),
$$

where U is the upsampling operation, $W _ { u , i }$ are the transposed convolution weights, $C _ { i }$ is the concatenation of encoder features with upsampled decoder features, and $F _ { i }$ is the output of the i-th decoder block.

The skip connections can be mathematically represented as:

$$
C _ {i} (x) = \operatorname{concat} \left(E _ {J - i + 1} (x), U _ {i} (x)\right). \tag {4}
$$

These connections preserve spatial information that would otherwise be lost during encoding, enhancing the precision of segmentation outputs

# 3.2. Fisher Vector

Let $X = \{ \mathbf { x } _ { t } \in \mathbb { R } ^ { D } \mid t = 1 , \ldots , T \}$ represent a sample containing T observations. Assuming the generation process of X follows a probability density function $u _ { \lambda }$ with parameters $\lambda ,$ we characterize these observations through the gradient vector:

$$
G _ {\lambda} ^ {X} = \nabla_ {\lambda} \log u _ {\lambda} (X). \tag {5}
$$

This gradient vector from Equation 5 can be employed in classification tasks. Following [23], we use the Fisher information matrix $F _ { \lambda }$ defined as:

$$
F _ {\lambda} = \mathbb {E} _ {X} \left[ G _ {\lambda} ^ {X} (G _ {\lambda} ^ {X}) ^ {\top} \right], \tag {6}
$$

where $\mathbb { E } _ { X }$ denotes the expectation over X.

The Fisher Kernel (FK) measuring similarity between two samples X and Y is then given by:

$$
K _ {\mathrm{FK}} (X, Y) = \left(G _ {\lambda} ^ {X}\right) ^ {\top} F _ {\lambda} ^ {- 1} G _ {\lambda} ^ {Y}. \tag {7}
$$

Since $F _ { \lambda } ^ { - 1 }$ is positive semi-definite, we apply Cholesky decomposition $F _ { \lambda } ^ { - 1 } =$ $L _ { \lambda } ^ { \top } L _ { \lambda }$ to reformulate the kernel as:

$$
K _ {\mathrm{FK}} (X, Y) = (\mathcal {G} _ {\lambda} ^ {X}) ^ {\top} \mathcal {G} _ {\lambda} ^ {Y}, \tag {8}
$$

where the Fisher Vector (FV) is defined by:

$$
\mathcal {G} _ {\lambda} ^ {X} = L _ {\lambda} G _ {\lambda} ^ {X}. \tag {9}
$$

Notably, the Fisher Vector $\mathcal { G } _ { \lambda } ^ { X }$ maintains the same dimensionality as $G _ { \lambda } ^ { X }$ [24]. This implies that classification using a linear kernel machine with FVs is equivalent to employing a non-linear kernel machine with $K _ { \mathrm { F K } }$ . For comprehensive details on Fisher Vectors, we refer readers to [24].

# 4. Proposed Method

The proposed methodology can be organized into 4 major modules: autoencoder self-supervised module, CNN feature extractor, GMM training, and predictor. The following sections delineate the overall idea of each such component.

# 4.1. Self-supervised module

The self-supervised module works at two stages of the pipeline. In the pretraining task, the input image is processed by Equations (1)-(4).

Being an autoencoder, we have that both the input and target output correspond to the same training image $\ b { I } \in \mathbb { R } ^ { m \times n }$ . For the reconstruction loss we adopt the classical mean squared error, such that all weights W and biases b in (1)-(4) are learned by minimizing

$$
L = \frac {1}{m n} \sum_ {x = 1} ^ {m} \sum_ {y = 1} ^ {n} \left(I [ x, y ] - F _ {i} (I) [ x, y ]\right) ^ {2},
$$

where $F _ { i }$ is defined in (4).

In the final classification task, the decoder is removed and the output is provided by $B ( I )$ , where B is the bottleneck operator defined by Equation (2).

# 4.2. CNN Feature Extractor

This module consists of a series of convolutional operations. At each layer $\ell ,$ for an input with C channels, the output at position $( i , j )$ and channel k is defined as

$$
x _ {i j k \ell} = \sum_ {c = 1} ^ {C} \sum_ {a = 0} ^ {m - 1} \sum_ {b = 0} ^ {m - 1} w _ {a b c k} \cdot y _ {(i + a) (j + b) c} ^ {\ell - 1},
$$

where m is the kernel size, w represents the (potentially learnable) weights, and

$$
y _ {i j k} ^ {\ell} = \sigma (x _ {i j k} ^ {\ell}),
$$

with σ denoting a nonlinear activation function. This architecture may also include other standard operations, such as pooling, depending on the definition of w.

# 4.3. GMM Training

Fisher vectors are computed from the aggregated set of local features $X =$ $\{ \mathbf { x } _ { t } \in \mathbb { R } ^ { D } \mid t = 1 , 2 , \ldots , T \}$ , where $\begin{array} { r } { T = \sum _ { i = 0 } ^ { k } T _ { n - i } } \end{array}$ and $D = D _ { n - k }$ .

We assume each local feature $\mathbf { x } _ { t }$ is independently generated according to the distribution $u _ { \lambda }$ . Under this assumption, Equation (9) becomes:

$$
\mathcal {G} _ {\lambda} ^ {X} = L _ {\lambda} \frac {1}{T} \sum_ {t = 1} ^ {T} \nabla_ {\lambda} \log u _ {\lambda} (\mathbf {x} _ {t}). \tag {10}
$$

Here, $u _ { \lambda }$ is modeled as a Gaussian Mixture Model (GMM) with K Gaussian components, each representing a visual word in the learned dictionary:

$$
u _ {\lambda} (\mathbf {x}) = \sum_ {i = 1} ^ {K} w _ {i} u _ {i} (\mathbf {x}), \tag {11}
$$

where $\lambda = \{ w _ { i } , \mu _ { i } , \Sigma _ { i } \mid i = 1 , \ldots , K \}$ , and $w _ { i } , \mu _ { i }$ , and $\Sigma _ { i }$ denote the weight, mean, and covariance matrix of the i-th Gaussian component, respectively.

The probability that an observation $\mathbf { x } _ { t }$ is generated by the i-th Gaussian is given by:

$$
\gamma_ {i} (\mathbf {x} _ {t}) = \frac {w _ {i} u _ {i} (\mathbf {x} _ {t})}{\sum_ {j = 1} ^ {K} w _ {j} u _ {j} (\mathbf {x} _ {t})}. \tag {12}
$$

We assume diagonal covariance matrices, as any distribution can be approximated to arbitrary precision by a weighted sum of Gaussians with diagonal covariances [25]. Let $\sigma _ { i } ^ { 2 } = \mathrm { d i a g } ( \Sigma _ { i } )$ . Using the expressions for $L _ { \lambda }$ and $\nabla _ { \lambda } \log u _ { \lambda } ( X )$ from [25], Equation (9) can be rewritten as:

$$
\mathcal {G} _ {w _ {i} ^ {d}} ^ {X} = \frac {1}{T \sqrt {w _ {i}}} \sum_ {t = 1} ^ {T} \left(\gamma_ {i} (\mathbf {x} _ {t}) - w _ {i}\right), \tag {13}
$$

$$
\mathcal {G} _ {\mu_ {i} ^ {d}} ^ {X} = \frac {1}{T \sqrt {w _ {i}}} \sum_ {t = 1} ^ {T} \gamma_ {i} (\mathbf {x} _ {t}) \left(\frac {x _ {t} ^ {d} - \mu_ {i} ^ {d}}{\sigma_ {i} ^ {d}}\right), \tag {14}
$$

$$
\mathcal {G} _ {\sigma_ {i} ^ {d}} ^ {X} = \frac {1}{T \sqrt {2 w _ {i}}} \sum_ {t = 1} ^ {T} \gamma_ {i} (\mathbf {x} _ {t}) \left[ \frac {(x _ {t} ^ {d} - \mu_ {i} ^ {d}) ^ {2}}{(\sigma_ {i} ^ {d}) ^ {2}} - 1 \right]. \tag {15}
$$

These equations (13–15) are used to compute each component of the Fisher Vector (FV) from any set of local features extracted from the CNN’s convolutional layers. Additionally, we extract features from the last fully connected layer by removing the output layer; this feature vector is referred to as FC.

# 4.4. Predictor

For classification, we apply both L2 and power normalization to the FV features, as recommended in [26]. While various normalization techniques exist for different image types, some can be computationally expensive. Therefore, we adopt an efficient normalization strategy to ensure high performance.

We further enhance the feature representation by concatenating the selfsupervised component (SSL) with the normalized FV features, resulting in a combined feature vector denoted as SSL+FV.

Classification is performed using a Support Vector Machine (SVM) with the Bhattacharyya coefficient, as defined in Definition ??, serving as the kernel. For $\mathbf { x } , \mathbf { y } \in \mathbb { R } ^ { N }$ , the Bhattacharyya coefficient is defined as:

$$
K (\mathbf {x}, \mathbf {y}) = \sum_ {i = 1} ^ {N} \mathrm{sign} (x _ {i} y _ {i}) \sqrt {| x _ {i} y _ {i} |}. \tag {16}
$$

This coefficient can be rewritten as:

$$
K (\mathbf {x}, \mathbf {y}) = \phi (\mathbf {x}) ^ {T} \phi (\mathbf {y}), \tag {17}
$$

where ϕ(x) is a vector with components:

$$
\phi (\mathbf {x}) _ {i} = \operatorname{sign} (x _ {i}) \sqrt {| x _ {i} |}. \tag {18}
$$

By applying the transformation in Equation (18) to the feature vectors, classification can be performed using a linear SVM. For multi-class problems, we employ a one-vs-rest strategy.

Figure 1 illustrates the overall architecture of the proposed framework.

![](images/304eefa290fc4043bde5a25d89332ae9122970431d44e6a5b510b43cce46ec01.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Image"] --> B["Encoder"]
    A --> C["Decoder"]
    A --> D["CNN Feature Extraction"]
    B --> E["Latent Representation"]
    C --> E
    D --> F["Concatenation"]
    E --> F
    F --> G["Predictor"]
    H["Fisher Vectors"] --> F
    I["GMM Training"] --> F
```
</details>

Figure 1: Proposed method.

# 5. Experimental setup

In this section, we outline the evaluation protocol for our proposed methodology. Our base model employs the EfficientNet-B5 architecture [27], utilizing

ImageNet pre-trained weights as the feature extractor.

The datasets used for evaluation include KTH-TIPS2-b, FMD, DTD, UIUC, and UMD. For the practical application, we use the 1200Tex dataset. Each of these datasets is described below.

KTH-TIPS2-b [28] contains images of 11 materials, each with 4 samples. Every sample is captured at 9 scales, 3 poses, and 4 lighting conditions, resulting in 108 images per material per sample. For each evaluation round, 3 samples are used for training and 1 for testing.

FMD [29] comprises 10 classes with 100 images each, at a resolution of 512×384 pixels. We conduct 10 rounds of training and testing, randomly splitting the dataset in half for each round.

DTD [30] consists of 5,640 images of varying sizes, divided into 47 categories (120 images per class). Each class is split equally into training, validation, and testing sets. The dataset provides 10 predefined splits; for each, we use the training and validation sets for model adjustment and the test set for evaluation.

UMD [31] includes 25 classes with 40 images each, all sized 1280×960 pixels. We follow the same evaluation protocol as for FMD.

UIUC [32] contains 1,000 images evenly distributed across 25 classes, each with a resolution of 640×480 pixels. The evaluation protocol matches that used for FMD.

1200Tex [33] comprises 1,200 leaf surface images from 20 Brazilian plant species, with 60 samples per class. The same training and testing protocol as FMD is applied.

To ensure fair comparison across datasets, we standardize the image width to yield a similar number of local features. For datasets with varying image sizes, both width and height are set to the same value, with a default width of 320 pixels.

Unless otherwise noted, Fisher Vectors are computed using 16 kernels. Local features are extracted from two convolutional layers: the last layer of block 5 and the last layer of block 6. Since the latter has more channels, dimensionality reduction is applied to its features, typically using PCA.

Our experiments begin by comparing PCA with two alternative dimensionality reduction methods: average pooling and max pooling. Next, we assess how accuracy changes as the number of layers used for local feature extraction increases, including results for a single layer (the last convolutional layer), as in [34]. Layers are selected as the last convolutional layer of each block, starting from the final block; for n layers, the last n blocks are used.

We also investigate how the number of kernels and the number of local features affect the descriptive power of Fisher Vectors. In the third experiment, we vary the number of kernels and evaluate the resulting accuracy. Subsequently, we examine the impact of the number of local features by adjusting the input image width.

Finally, we compare our base model to alternative state-of-the-art methods, using optimized parameters for our approach. The experimental section concludes with a practical application: identifying Brazilian plant species from scanned images of leaf surfaces.

# 6. Results and Discussion

Table 1 lists different classification metrics for the proposed method on the analyzed datasets. Overall, we observe that using only Fisher features provide slightly better metrics. This can be explained by the intrinsic similarity in nature between FC features and AE features, as both are generated by a learnable CNN. In this regard, the statistical procedure supported by GMM is more effective as it takes into account particularities of the distribution of deep features across the dataset.

Table 2 presents the accuracy achieved by the proposed method in comparison with several classical and state-of-the-art approaches in the literature. The overall performances of FVAE and FCFVAE show that our proposals are competitive with the state-of-the-art and represent a promising direction of exploration in texture recognition. Particularly, compared with the baseline backbone (EfficientNet-B5) of our architecture, we notice that FV+AE features contributed significantly, increasing the average accuracy in more than 5% in most databases. Such numbers can be explained by the ability of autoencoders to compress information while preserving its essence. In a self-supervised framework, where the pre-training stage should be able of enriching the CNN with specific internal patterns of the image, this approach represents a straightforward and, at the same time, highly efficient way of obtaining such representation.

Table 1: Classification metrics of the proposed method, in its two variations FCFVAE and FVAE. 

<table><tr><td colspan="5">FMD</td></tr><tr><td>Feature Vector</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td>FV</td><td>90.3 ± 0.5</td><td>90.3 ± 0.5</td><td>90.5 ± 0.5</td><td>90.2 ± 0.5</td></tr><tr><td>FV+FC</td><td>91.7 ± 0.4</td><td>91.7 ± 0.3</td><td>91.8 ± 0.4</td><td>91.7 ± 0.4</td></tr><tr><td>FCFVAE</td><td>92.2 ± 0.2</td><td>92.3 ± 0.2</td><td>92.3 ± 0.2</td><td>92.2 ± 0.2</td></tr><tr><td>FVAE</td><td>92.9 ± 0.4</td><td>93.0 ± 0.4</td><td>93.0 ± 0.5</td><td>92.9 ± 0.4</td></tr><tr><td colspan="5">KTH</td></tr><tr><td>Feature Vector</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1-Score</td></tr><tr><td>FV</td><td>92.8 ± 4.9</td><td>93.9 ± 4.1</td><td>92.8 ± 4.9</td><td>92.2 ± 5.6</td></tr><tr><td>FV+FC</td><td>92.4 ± 4.6</td><td>93.5 ± 4.0</td><td>92.4 ± 4.6</td><td>92.0 ± 5.1</td></tr><tr><td>FCFVAE</td><td>92.3 ± 4.6</td><td>93.4 ± 4.0</td><td>92.3 ± 4.6</td><td>91.8 ± 5.1</td></tr><tr><td>FVAE</td><td>92.7 ± 4.8</td><td>93.7 ± 4.1</td><td>92.7 ± 4.8</td><td>92.2 ± 5.4</td></tr></table>

Table 2: Accuracy comparison with other methods in literature. In the first row, we include the performance of fine-tuning the same CNN architecture we are using. All results shown are obtained directly from the original paper of each method. Non-published results are represented by dashes. 

<table><tr><td>Method</td><td>KTH-TIPS2-b</td><td>FMD</td><td>DTD</td><td>UMD</td><td>UIUC</td><td>GTOS</td></tr><tr><td>EfficientNet-B5</td><td> $87.0 \pm 5.9$ </td><td> $87.4 \pm 0.6$ </td><td> $77.6 \pm 0.4$ </td><td> $99.9 \pm 0.1$ </td><td> $98.4 \pm 0.4$ </td><td> $78.7 \pm 2.0$ </td></tr><tr><td>FV-VGGVD [34]</td><td> $81.8 \pm 2.4$ </td><td> $79.8 \pm 1.8$ </td><td> $72.3 \pm 1.0$ </td><td> $99.9 \pm 0.1$ </td><td> $99.9 \pm 0.1$ </td><td>-</td></tr><tr><td>SIFT-FV [34]</td><td> $81.5 \pm 2.0$ </td><td> $82.2 \pm 1.4$ </td><td> $75.5 \pm 0.8$ </td><td> $99.9 \pm 0.1$ </td><td> $99.9 \pm 0.1$ </td><td>-</td></tr><tr><td>LFV [35]</td><td> $82.6 \pm 2.6$ </td><td> $82.1 \pm 1.9$ </td><td> $73.8 \pm 1.0$ </td><td>-</td><td>-</td><td>-</td></tr><tr><td>DeepTEN [36]</td><td> $82.0 \pm 3.3$ </td><td> $80.2 \pm 0.9$ </td><td>-</td><td>-</td><td>-</td><td> $84.3 \pm 1.9$ </td></tr><tr><td>Xception + SIFT-FV [37]</td><td>-</td><td> $86.1 \pm 1.6$ </td><td> $75.4 \pm 1.0$ </td><td>-</td><td>-</td><td>-</td></tr><tr><td>DSRNet [38]</td><td> $85.9 \pm 1.3$ </td><td> $86.0 \pm 0.8$ </td><td> $77.6 \pm 0.6$ </td><td>-</td><td>-</td><td> $85.3 \pm 2.0$ </td></tr><tr><td>VisGraphNet [39]</td><td>-</td><td>77.3</td><td>-</td><td>98.1</td><td>97.6</td><td>-</td></tr><tr><td>Non-Add Entropy [40]</td><td>84.4</td><td>77.7</td><td>-</td><td>98.8</td><td>98.5</td><td>-</td></tr><tr><td>Residual Pooling [41]</td><td>-</td><td>85.7</td><td>76.6</td><td>-</td><td>-</td><td></td></tr><tr><td>FENet [17]</td><td> $88.2 \pm 0.2$ </td><td> $86.7 \pm 0.1$ </td><td> $74.2 \pm 0.1$ </td><td>-</td><td>-</td><td> $85.7 \pm 0.1$ </td></tr><tr><td>CLASSNet [14]</td><td> $87.7 \pm 1.3$ </td><td> $86.2 \pm 0.9$ </td><td> $74.0 \pm 0.5$ </td><td>-</td><td>-</td><td> $85.6 \pm 2.2$ </td></tr><tr><td>DFAEN [16]</td><td>86.6</td><td>87.6</td><td>76.1</td><td>-</td><td>-</td><td>-</td></tr><tr><td>RADAM [15]</td><td> $90.7 \pm 4.0$ </td><td> $88.7 \pm 0.4$ </td><td> $77.0 \pm 0.7$ </td><td>-</td><td>-</td><td> $84.2 \pm 1.7$ </td></tr><tr><td>Capsule [42]</td><td>71.8</td><td>80.7</td><td>71.0</td><td>-</td><td>99.3</td><td></td></tr><tr><td>FCFVAE (Ours)</td><td> $92.3 \pm 4.6$ </td><td> $92.4 \pm 0.2$ </td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>FVAE (Ours)</td><td> $92.7 \pm 4.8$ </td><td> $92.9 \pm 0.4$ </td><td>-</td><td>-</td><td>-</td><td>-</td></tr></table>

In summary, our results confirm the expectations that there are still many different ways of tackling self-supervised learning in computer vision. In particular, less computationally intensive architectures might be a promising direction to be further explored. Whilst models with large number of parameters, like vision transformers, yield remarkable performance in tasks where the relation among distant regions within the image play important role, this is not necessarily the case of other domains, such as the visual textures. Our study verify that “U-Net-like” architectures can be highly efficient in this scenario, providing rich deep representation of unlabeled images with relatively low computational cost.

# 7. Conclusions

Here we proposed a self-supervised framework for texture recognition. Instead of relying on computationally intensive backbones, such as vision transformers, we adopt a convolutional simplified encoder-decoder module. The learnable features are processed by an advanced pooling operator inspired by Fisher vectors. Finally, the overall architecture is employed for texture recognition.

Our proposal is evaluated on the classification of diverse texture datasets, achieving promising results. The accuracy outperforms several state-of-theart models in the state-of-the-art with a low computational cost of the selfsupervised module. The results suggest the potential still existent for convolutional architectures in self-supervised frameworks. Whereas ViTs are a natural choice in scenarios where the amount of data for pretraining is large, when little data is available, more efficient architectures are recommended.

This study also points to aspects that demand further future investigation. One of such possibilities is how other self-supervised paradigms, like contrastive learning, could be adapted to benefit from efficient CNN backbones. Another one is a study on how other models from the state-of-the-art in texture analysis might be benefited by a self-supervised pretraining stage. In conclusion, we believe that these research lines will not only address current limitations of the literature but also pave the way for innovative applications and methodological advancements in the field.

# Code availability

The code developed in this study is publicly available at https://github. com/lolyra/medical.

# Declaration of competing interest

J. B. F. reports equipment, drugs, or supplies was provided by State of Sao Paulo Research Foundation. J. B. F. reports financial support was provided by National Council for Scientific and Technological Development. L. O. L. reports financial support was provided by Coordination of Higher Education Personnel Improvement.

# Acknowledgements

This study was financed in part by the Coordenação de Aperfeiçoamento de Pessoal de Nível Superior - Brasil (CAPES) - Finance Code 001. J. B. F. gratefully acknowledges the financial support of São Paulo Research Foundation (FAPESP) (Grant #2020/01984-8) and of National Council for Scientific and Technological Development, Brazil (CNPq) (Grant #306981/2022-0).

# References

[1] D. Young, N. Khan, S. R. Hobson, D. Sussman, Diagnosis of placenta accreta spectrum using ultrasound texture feature fusion and machine learning, Computers in Biology and Medicine 178 (2024) 108757.   
[2] S. Barburiceanu, S. Meza, B. Orza, R. Malutan, R. Terebes, Convolutional neural networks for texture feature extraction. applications to leaf disease classification in precision agriculture, IEEE Access 9 (2021) 160085–160103.

[3] J. Si, S. Kim, V-daft: Visual technique for texture image defect recognition with denoising autoencoder and fourier transform, Signal, Image and Video Processing 18 (10) (2024) 7405–7418.   
[4] H. Han, Z. Feng, W. Du, S. Guo, P. Wang, T. Xu, Remote sensing image classification based on multi-spectral cross-sensor super-resolution combined with texture features: A case study in the liaohe planting area, IEEE Access 12 (2024) 16830–16843.   
[5] P. Akiva, M. Purri, M. Leotta, Self-supervised material and texture representation learning for remote sensing tasks, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022, pp. 8203–8215.   
[6] K. He, X. Chen, S. Xie, Y. Li, P. Dollár, R. Girshick, Masked autoencoders are scalable vision learners, in: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 16000–16009.   
[7] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly, et al., An image is worth 16x16 words: Transformers for image recognition at scale, arXiv preprint arXiv:2010.11929 (2020).   
[8] L. O. Lyra, A. E. Fabris, J. B. Florindo, A multilevel pooling scheme in convolutional neural networks for texture image recognition, Applied Soft Computing (2024) 111282doi:https://doi.org/10.1016/j.asoc.2024.111282.   
[9] A. Gogna, A. Majumdar, Discriminative autoencoder for feature extraction: Application to character recognition, Neural Processing Letters 49 (2019) 1723–1735.   
[10] Z. Yang, X. Wu, P. Huang, F. Zhang, M. Wan, Z. Lai, Orthogonal autoencoder regression for image classification, Information Sciences 618 (2022) 400–416.

[11] Q. Kang, J. Gao, K. Li, Q. Lao, Deblurring masked autoencoder is better recipe for ultrasound image recognition, in: International Conference on Medical Image Computing and Computer-Assisted Intervention, Springer, 2023, pp. 352–362.   
[12] M. Cimpoi, S. Maji, A. Vedaldi, Deep filter banks for texture recognition and segmentation, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2015, pp. 3828–3836.   
[13] K. Simonyan, A. Zisserman, Very deep convolutional networks for largescale image recognition, arXiv preprint arXiv:1409.1556 (2014).   
[14] Z. Chen, F. Li, Y. Quan, Y. Xu, H. Ji, Deep texture recognition via exploiting cross-layer statistical self-similarity, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2021, pp. 5231– 5240.   
[15] L. Scabini, K. M. Zielinski, L. C. Ribas, W. N. Gonçalves, B. De Baets, O. M. Bruno, Radam: Texture recognition through randomized aggregated encoding of deep activation maps, Pattern Recognition 143 (2023) 109802. doi:https://doi.org/10.1016/j.patcog.2023.109802. URL https://www.sciencedirect.com/science/article/pii/ S0031320323005009   
[16] Z. Yang, S. Lai, X. Hong, Y. Shi, Y. Cheng, C. Qing, Dfaen: Double-order knowledge fusion and attentional encoding network for texture recognition, Expert Systems with Applications 209 (2022) 118223.   
[17] Y. Xu, F. Li, Z. Chen, J. Liang, Y. Quan, Encoding spatial distribution of convolutional features for texture representation, Advances in Neural Information Processing Systems 34 (2021).   
[18] J. B. Florindo, E. E. Laureano, Boff: A bag of fuzzy deep features for texture recognition, Expert Systems with Applications 219 (2023) 119627.

[19] L. Scabini, A. Sacilotti, K. M. Zielinski, L. C. Ribas, B. De Baets, O. M. Bruno, A comparative survey of vision transformers for feature extraction in texture analysis, arXiv preprint arXiv:2406.06136 (2024).   
[20] L. Zhu, T. Chen, J. Yin, S. See, J. Liu, Learning gabor texture features for fine-grained recognition, in: Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 1621–1631.   
[21] A. Bera, D. Bhattacharjee, M. Nasipuri, Deep neural networks fused with textures for image classification, in: International conference on frontiers in computing and systems, Springer, 2022, pp. 103–111.   
[22] V. Goyal, S. Sharma, Texture classification for visual data using transfer learning, Multimedia Tools and Applications 82 (16) (2023) 24841–24864.   
[23] T. Jaakkola, D. Haussler, Exploiting generative models in discriminative classifiers, Advances in neural information processing systems 11 (1998).   
[24] J. Sánchez, F. Perronnin, T. Mensink, J. Verbeek, Image classification with the fisher vector: Theory and practice, International journal of computer vision 105 (3) (2013) 222–245.   
[25] F. Perronnin, C. Dance, Fisher kernels on visual vocabularies for image categorization, in: 2007 IEEE conference on computer vision and pattern recognition, IEEE, 2007, pp. 1–8.   
[26] F. Perronnin, J. Sánchez, T. Mensink, Improving the fisher kernel for largescale image classification, in: European conference on computer vision, Springer, 2010, pp. 143–156.   
[27] M. Tan, Q. Le, Efficientnet: Rethinking model scaling for convolutional neural networks, in: International Conference on Machine Learning, PMLR, 2019, pp. 6105–6114.   
[28] B. Caputo, E. Hayman, P. Mallikarjuna, Class-specific material categorisation, in: Tenth IEEE International Conference on Computer

Vision (ICCV’05) Volume 1, Vol. 2, 2005, pp. 1597–1604 Vol. 2. doi:10.1109/ICCV.2005.54.   
[29] L. Sharan, R. Rosenholtz, E. H. Adelson, Accuracy and speed of material categorization in real-world images, Journal of Vision 14 (9) (2014) 12–12. doi:10.1167/14.9.12.   
[30] M. Cimpoi, S. Maji, I. Kokkinos, S. Mohamed, A. Vedaldi, Describing textures in the wild, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2014, pp. 3606–3613.   
[31] Y. Xu, H. Ji, C. Fermüller, Viewpoint invariant texture description using fractal analysis, International Journal of Computer Vision 83 (1) (2009) 85–100. doi:10.1007/s11263-009-0220-6.   
[32] S. Lazebnik, C. Schmid, J. Ponce, A sparse texture representation using local affine regions, IEEE Transactions on Pattern Analysis and Machine Intelligence 27 (8) (2005) 1265–1278. doi:10.1109/TPAMI.2005.151.   
[33] D. Casanova, J. J. de Mesquita Sá Junior, O. M. Bruno, Plant leaf identification using gabor wavelets, International Journal of Imaging Systems and Technology 19 (3) (2009) 236–243. doi:10.1002/ima.20201.   
[34] M. Cimpoi, S. Maji, I. Kokkinos, A. Vedaldi, Deep filter banks for texture recognition, description, and segmentation, International Journal of Computer Vision 118 (1) (2016) 65–94.   
[35] Y. Song, F. Zhang, Q. Li, H. Huang, L. J. O’Donnell, W. Cai, Locallytransferred fisher vectors for texture classification, in: Proceedings of the IEEE International Conference on Computer Vision, 2017, pp. 4912–4920.   
[36] H. Zhang, J. Xue, K. Dana, Deep ten: Texture encoding network, in: Proceedings of the IEEE conference on computer vision and pattern recognition, 2017, pp. 708–717.

[37] M. Jbene, A. D. El Maliani, M. El Hassouni, Fusion of convolutional neural network and statistical features for texture classification, in: 2019 International Conference on Wireless Networks and Mobile Communications (WINCOM), IEEE, 2019, pp. 1–4.   
[38] W. Zhai, Y. Cao, Z.-J. Zha, H. Xie, F. Wu, Deep structure-revealed network for texture recognition, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2020, pp. 11010–11019.   
[39] J. B. Florindo, Y.-S. Lee, K. Jun, G. Jeon, M. K. Albertini, Visgraphnet: A complex network interpretation of convolutional neural features, Information Sciences 543 (2021) 296–308.   
[40] J. Florindo, K. Metze, Using non-additive entropy to enhance convolutional neural features for texture recognition, Entropy 23 (2021) 1259. doi:10.3390/e23101259.   
[41] S. Mao, D. Rajan, L. T. Chia, Deep residual pooling network for texture recognition, Pattern Recognition 112 (2021) 107817.   
[42] B. Mamidibathula, S. Amirneni, S. S. Sistla, N. Patnam, Texture classification using capsule networks, in: Pattern Recognition and Image Analysis: 9th Iberian Conference, IbPRIA 2019, Madrid, Spain, July 1–4, 2019, Proceedings, Part I 9, Springer, 2019, pp. 589–599.