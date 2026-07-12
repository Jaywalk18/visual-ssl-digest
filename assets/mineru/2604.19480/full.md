# Deep Sprite-based Image Models: An Analysis

Zeynep Sonat Baltacı

sonat.baltaci@enpc.fr

LIGM, CNRS, Univ Gustave Eifel, ENPC, Institut Polytechnique de Paris, France

Romain Loiseau

romain.loiseau@enpc.fr

LIGM, CNRS, Univ Gustave Eifel, ENPC, Institut Polytechnique de Paris, France

Mathieu Aubry

LIGM, CNRS, Univ Gustave Eifel, ENPC, Institut Polytechnique de Paris, France

mathieu.aubry@enpc.fr

Reviewed on OpenReview: https: // openreview. net/ forum? id= pXuxMLFo9g

## Abstract

While foundation models drive steady progress in image segmentation and difusion algorithms compose always more realistic images, the seemingly simple problem of identifying recurrent patterns in a collection of images remains very much open. In this paper, we focus on sprite-based image decomposition models, which have shown some promise for clustering and image decomposition and are appealing because of their high interpretability. These models come in diferent flavors, need to be tailored to specific datasets, and struggle to scale to images with many objects. We dive into the details of their design, identify their core components, and perform an extensive analysis on clustering benchmarks. We leverage this analysis to propose a deep sprite-based image decomposition method that performs on par with state-of-the-art unsupervised class-aware image segmentation methods on the standard CLEVR benchmark, scales linearly with the number of objects, identifies explicitly object categories, and fully models images in an easily interpretable way.<sup>1</sup>

![](images/23075ab9e6e096adf9e3b0f94b8221ad877e0ce5aaabec849b1ea6a611e2cb38.jpg)  
(a) Sprite-based approaches.

![](images/b45850bd79c6a1fb4433f635b24be84f7a30fb9de95f34dd1bee2ce05581af09.jpg)  
(b) Image clustering.

![](images/48169651a9aeeab13663bf03915c7d88fc6969eb69a4c9488f2cbeec952bb7de.jpg)  
(c) Unsupervised object discovery.

Figure 1: (a) Sprite-based approaches take a set of images as input and learn jointly a family of sprites and how to decompose each image into a sequence of transformed sprites. They can be applied to (b) image clustering and (c) unsupervised object discovery.

![](images/06ead3e5e2f92b6ab5e4d06d6b82f021767497a09c00800b920572f6209d6715.jpg)  
Figure 2: Overview. We decompose all sprite-based models in four main components: (1) a Sprite Generation Module ( ) that outputs K sprites S, (2) a Transformation Module ( ) that takes as input an image I and the sprites S to predict transformed sprites $\bar { S } ^ { I }$ , (3) a Decision Module ( ) that takes the image I and transformed sprites $\hat { S } ^ { \hat { I } }$ as input and outputs a probability distribution $p ^ { I }$ for using the sprites, and (4) a Training Criteria ( ) which consist of a reconstruction loss and potential regularization terms.

## 1 Introduction

Identifying recurring patterns in an image collection is a task in which humans excel. It is also critical for many scientific applications, from historical documents to medical image analysis. Although foundation features or models might be attractive tools for approaching this problem, they come with their black-box efects and the biases of their training data. Instead, we advocate for methods that can be directly optimized on the target image collection, ofer maximal interpretability, and have limited bias.

In this study, we focus more specifically on sprite-based methods (Visser et al., 2019; Monnier et al., 2020; 2021; Smirnov et al., 2021; Loiseau et al., 2024; Siglidis et al., 2024), which are the main type of object-centric approaches to unsupervised object discovery that allow joint categorization and localization (Villa-Vásquez & Pedersoli, 2024) (Fig. 1). Sprite-based methods ofer several other attractive advantages. First, they explicitly model repeated patterns as a finite set of prototypical objects, called sprites. Second, not only do they provide for each analyzed image a layered decomposition, but they also give direct, explicit access to the transformation of the sprites in the image, such as position, scale, and color transformations. Third, their relationship with the standard K-means clustering algorithm (MacQueen, 1967; Bottou & Bengio, 1994) and transformation invariant methods (Frey & Jojic, 1999; 2001; 2003) is well understood (Monnier et al., 2020). However, sprite-based methods have not been fully explored. In particular, the impact of architectural changes and training methodology on their results is poorly understood and diferent approaches have been demonstrated on diferent non-standard datasets. Our goal in this study is to better identify key design choices for sprite-based methods and analyze their efects.

In more detail, we separate sprite-based architectures into their key components, visualized in Fig. 2: Sprite Generation Module, Transformation Module, Decision Module, and Training Criteria. For each components, we identify diferent design choices proposed in the literature, as well as simpler baselines, detailed in Fig. 3. We explain how the training criteria correspond to diferent image composition models and are related to the exponential cost of some sprite-based image decomposition approaches. We show that one can efectively study the impact of most design choices for clustering, where the benchmarks are more realistic and diverse than for image decomposition, where they are mainly synthetic.

Our key insight is that the main challenge of sprite-based approaches lies in jointly learning and selecting the sprites. K-means-style optimization for sprite selection leads to the discovery of more visually coherent, precise, and semantically accurate objects, without the need for complex regularization, as regularization is implicitly enforced through cluster reassignment policies. However, this type of optimization scales exponentially with the number of objects per image. We show that diferent regularization techniques can improve approaches that directly predict sprite selection. While we perform most of our analyses on the more diverse and less computationally demanding clustering benchmarks, this actually enable us to design an approach which we demonstrate can generalize to multi-layer decomposition.

This paper is organized as follows: First, in Section 2, we review the literature on clustering and image decomposition. Second, in Section 3, we present a unified formalization for sprite-based image decomposition models. Third, in Section 4, we perform a comparative analysis of the diferent design choices on clustering and propose our new approach. Finally, in Section 5, we extend and evaluate our approach for multi-layer image decomposition.

Contributions. Our contributions are as follows:

• We perform an exhaustive analysis of sprite-based methods and identify their key components.

• We systematically study their impact on clustering benchmarks.

• We propose a novel sprite-based approach that predicts sprite selection and scales linearly with the number of objects per image.

## 2 Related Work

## 2.1 Image Clustering

We focus on image clustering approaches that are most related to our work and classify them into pixelbased and deep-feature-based clustering. For a broader literature review, we refer the reader to dedicated surveys (Zhou et al., 2024; Ren et al., 2024; Wei et al., 2024).

## 2.1.1 Pixel-based Clustering

Clustering in pixel space is highly challenging since the image content can be associated with diferent backgrounds and can undergo spatial and color transformations that completely change its pixel representation. Traditional clustering methods, such as K-means (MacQueen, 1967), therefore lead to limited results when applied directly on full images. EM-based transformation-invariant clustering algorithms have been proposed to gain invariance to user-defined families of transformation (Frey & Jojic, 1999; 2001; 2003). They operate directly in image space, compare pixel values, and provide prototypical representations of clusters. The idea of transformation invariance was also adopted in congealing-based image alignment models that learn transformations using a data-driven approach (Cox et al., 2008; 2009; Huang et al., 2007; Miller et al., 2000; Annunziata et al., 2019; Learned-Miller, 2006), some with a focus on clustering (Mattar et al., 2012; Liu et al., 2009). Deep Transformation-invariant (DTI) Clustering builds on this idea but optimizes prototypes and transformations in a deep learning framework Monnier et al. (2020). The sprite-based image models we study are very related to DTI-Clustering, which can be seen as a single-layer image model, where sprites correspond to prototypes.

## 2.1.2 Deep Features and Clustering

Many recent deep architectures adopt clustering as an objective for representation learning, e.g., Caron et al. (2018; 2020); Li et al. (2021); Liang et al. (2023), without specifically targeting clustering performance. More relevant to us are those that specifically target clustering, focusing on various technical tools, such as CNNs (Yang et al., 2016; Chang et al., 2017), autoencoders (Xie et al., 2016; Mrabah et al., 2019; Dizaji et al., 2017; Kosiorek et al., 2019; Shaham et al., 2018), mutual information (Hu et al., 2017; Ji et al., 2019), generative models (Jiang et al., 2016; Mukherjee et al., 2018), or instance discrimination (Niu et al., 2022; Van Gansbeke et al., 2020). The crucial and common aspect of these deep clustering approaches is relying on abstract image features. However, relying on deep representations of images and clusters in feature space makes it very hard to interpret the results, performance, and failures, especially in a visually intuitive way.

## 2.2 Image Decomposition

Image decomposition is a broad concept and could encompass broad areas of research from image cosegmentation to layered video representations. In this section, we focus on the approaches that are the most relevant to our work and are often referred to as unsupervised multi-object segmentation approaches or deep object-centric image decomposition methods. We only review single-image methods, and do not dive into the many works that leverage motion, video, or 3D. We follow the taxonomy of Karazija et al. (2021), diferentiating pixel-based, glimpse-based, and sprite-based approaches. Another view of these approaches is presented in Gref et al. (2020), which diferentiates approaches depending on the type of slot they rely on, namely instance slots, sequential slots, spatial slots, and category slots. For a broader review of unsupervised object discovery approaches, we refer the reader to Villa-Vásquez & Pedersoli (2024).

![](images/59d3df63e94ea17ac6eafb4165b805c44710d2b2b05f50c06125e5decc0a8e88.jpg)  
d) Composition Model and Training Criteria  
Figure 3: Possible design choices for the main components identified in Fig. 2. Modules can take as input the input image I, features from the input image $f ( I )$ , the sprites S, the transformed sprites $\bar { S } ^ { I }$ and the predicted sprite probabilities $p ^ { I }$ . (a) The Sprite Generation Module ( ) can learn the sprites directly as learnable parameters (Pixels), generate them from learnable latent variables with a multi-layer perceptron (MLP) or a UNet architecture. (b) The Transformation Module ( ) parameters can be learned with a shared or sprite-specific network, and with diferent curriculum learning strategies. (c) The Decision Module ( ) can select sprites leading to the minimum reconstruction error (Min-Loss), or predict them using the sprites’ latent representations (Weight Prediction), or directly a linear projection (Linear Mapping), with alternative activations. (d) The Composition Model and Training Criteria ( ), where the main loss can either be the sum of the reconstruction errors obtained with all the possible sprites selection weighted by their probability $\left( \mathcal { L } _ { 0 - 1 } \right)$ or the reconstruction error with composite sprites $\left( \mathcal { L } _ { \mathrm { c o m p } } \right)$ . It can also include regularizations $( { \mathcal { L } } _ { \{ { \mathrm { f r e q } } , { \mathrm { b i n } } , { \mathrm { e m p t y } } \} } )$

Pixel-based methods. Pixel-based methods assign each pixel to an image component, typically by performing probabilistic pixel clustering. Early works tackle this clustering problem by developing approaches based on denoising autoencoders (DAE) (Gref et al., 2015; Vincent et al., 2008), Iterative Amortized Grouping (Gref et al., 2016), and Neural Expectation Maximization (Gref et al., 2017). However, these pioneer methods were limited to simple images with a small number of objects.

More recent models following this pixel-based paradigm include MONet (Burgess et al., 2019), IODINE (Gref et al., 2019), eMORL (Emami et al., 2021), and GENESIS (Engelcke et al., 2020; 2021), and demonstrate results on the more challenging synthetic CLEVR dataset of rendered 3D spheres, cubes, and cylinders. They typically output a segmentation mask for each image component as well as a latent code that enables generating an appearance image for each component. The ECON (von Kügelgen et al., 2020) method is built on MONet but is more related to our work because it explicitly models object layers and is designed to completely model occluded objects. However, it has only been demonstrated on very simple synthetic data.

Moving away from probabilistic scene representation and pixel clustering of these so-called scene-mixture models, Locatello et al. (2020) proposed a discriminative approach to scene component identification. Their slot attention mechanism localizes scene components through an iterative clustering-like attention mechanism and leads to a latent representation for each slot, which encodes both its mask and appearance. This approach has been successfully applied to perform object discovery on much more challenging images. DINOSAUR (Seitzer et al., 2023) first demonstrated results on real-world datasets by applying slot attention to DINO features (Caron et al., 2021), instead of pixels. It has also been combined with more complex slot decoders, including auto-regressive (Singh et al., 2022; Kakogeorgiou et al., 2024) and difusion (Jiang et al., 2023; Wu et al., 2023; Singh et al., 2025) ones.

Glimpse-based methods. Glimpse-based methods first extract regions of the image containing objects and then predict object models for each region. This idea was introduced by the Attend, Infer, Repeat approach (AIR) (Eslami et al., 2016), which was the main inspiration for a series of works, such as SQAIR (Kosiorek et al., 2018), SPAIR (Crawford & Pineau, 2019), or SuPAIR (Stelzner et al., 2019). Similar to early pixel-based approaches, these works were developed for very simple synthetic datasets. More recent works, such as SPACE (Lin et al., 2020), GNM (Jiang & Ahn, 2020), and AST (Sauvalle & de La Fortelle, 2023), extend glimpse-based approaches to CLEVR-like datasets. However, they seem to be out-shone by slotattention-based approaches, which brought comparable eficiency to pixel-based approaches.

Sprite-based methods. Sprite-based methods learn a set of object prototypes, referred to as sprites, and how to combine sprites to reconstruct images. These sprites make their image model much more tangible than pixel and glimpse-based approaches, enabling them to discover object categories, not just instance segmentation. StampNet (Visser et al., 2019) can be considered as the first deep sprite-based approach. It learns a latent space to categorize and localize objects, but was only demonstrated on very simple synthetic datasets. Capsule approaches (Kosiorek et al., 2019; Xiang et al., 2021) are similar in spirit and have been categorized as sprite-based methods in Villa-Vásquez & Pedersoli (2024). However, they learn abstract feature-based representations of object parts, they are typically evaluated on clustering benchmarks, and to the best of our knowledge, they have not been demonstrated on standard multi-object datasets. Appealing results on video-game and text images have been demonstrated by MarioNette (Smirnov et al., 2021), which sees sprite discovery as a self-supervised learning problem, and learns to predict sprite occurrence and position. DTI-Sprites (Monnier et al., 2021) models sprite shape and color transformation, enabling it to tackle more complex datasets, including CLEVR data. However, it requires testing many sprite configurations instead of predicting sprite occurrence, and thus does not scale to a large number of objects. Focusing on text line analysis, the Learnable Typewriter (Siglidis et al., 2024) combines ideas from MarioNette and DTI-Sprites for applications in digital humanities. Similar ideas have been applied to model 3D point clouds (Loiseau et al., 2024). Our analysis encompasses all these sprite-based approaches and clarifies their diferences.

## 3 Sprite-based Approaches

In this section, we first present a unified view of sprite-based decomposition methods for clustering and layered image decomposition, summarized in Figure 2. Then, for each key component that we identify, we detail diferent design choices that have been introduced in the literature and that we consider in our study, which are summarized in Figure 3, and are related to the literature in Table 1.

Table 1: Comparison of sprite-based models. Existing sprite-based methods make very diferent choices for several of the four components that we identified, making direct comparison between their performance dificult. Our exhaustive analysis leads to an informed choice of all the components for clustering (Ours-C) and image decomposition (Ours-D).

<table><tr><td>Method</td><td colspan="8">● Sprite Generation ● Curriculum ● Sharing ● Decision ● Activation ● $\mathcal{L}_{\text{rec}}$  ● $\mathcal{L}_{\text{reg}}$ </td></tr><tr><td>StampNet Visser et al. (2019)</td><td>pixels</td><td>all</td><td>sprite-specific</td><td>linear</td><td>hard GS</td><td> $\mathcal{L}_{\text{comp}}$ </td><td>-</td><td>decomposition</td></tr><tr><td>DTI-Clustering Monnier et al. (2020)</td><td>pixels</td><td>one-by-one</td><td>sprite-specific</td><td>Min-Loss</td><td>none</td><td> $\mathcal{L}_{\text{comp}=0-1}$ </td><td>reassignment</td><td>clustering</td></tr><tr><td>DTI-Sprites Monnier et al. (2021)</td><td>pixels</td><td>one-by-one</td><td>sprite-specific</td><td>Min-Loss</td><td>softmax</td><td colspan="2"> $\mathcal{L}_{\text{comp}=0-1}$  reassignment,  $\mathcal{L}_{\text{empty}}$ </td><td>decomposition</td></tr><tr><td>MarioNette Smirnov et al. (2021)</td><td>MLP</td><td>all</td><td>shared</td><td>weight prediction</td><td>none</td><td> $\mathcal{L}_{\text{comp}}$ </td><td> $\mathcal{L}_{\text{bin}}$ </td><td>decomposition</td></tr><tr><td>Learnable Earth Parser Loiseau et al. (2024)</td><td>3D point clouds</td><td>one-by-one</td><td>shared</td><td>linear</td><td>softmax</td><td> $\mathcal{L}_{0-1}$ </td><td> $\mathcal{L}_{\text{freq}}$ </td><td>decomposition</td></tr><tr><td>Learnable Typewriter Siglidis et al. (2024)</td><td>MLP</td><td>all</td><td>shared</td><td>weight prediction</td><td>softmax</td><td> $\mathcal{L}_{\text{comp}}$ </td><td>-</td><td>decomposition</td></tr><tr><td>Ours-C</td><td>MLP</td><td>one-by-one</td><td>sprite-specific</td><td>linear</td><td>soft GS</td><td> $\mathcal{L}_{\text{comp}}$ </td><td> $\mathcal{L}_{\{\text{freq},\text{bin}\}}$ </td><td>clustering</td></tr><tr><td>Ours-D</td><td>MLP</td><td>one-by-one</td><td>sprite-specific</td><td>linear</td><td>soft GS</td><td> $\mathcal{L}_{\text{comp}}$ </td><td> $\mathcal{L}_{\text{empty}}$ </td><td>decomposition</td></tr></table>

## 3.1 Unified View and Formalization

Our key insight is that sprite-based approaches rely on four main components that we present first. We then discuss how these modules can be used for multi-object image decomposition and clustering. The choices made by diferent sprite-based approaches in the literature are summarized in Table 1.

## 3.1.1 Key Components

Sprite-based approaches take as input an image $I \in \mathbb { R } ^ { W \times H \times C }$ , with $C = 1$ for a grayscale image and $C = 3$ for an RGB image, and predict a set of layers associated with this image. As visualized in Figure 2, we identified four key components in sprite-based methods:

• A sprite generation module, G (Section 3.2), which learns K sprites $S _ { 1 } , \cdots , S _ { K }$ , with for all $k \in \mathbf { \bar { \{ 1 , ~ \cdots ~ } , K \} } , S _ { k } \in \mathbb { R } ^ { R \times R \times C ^ { \prime } }$ , where R is the size of the sprites and $C ^ { \prime }$ is the number of channels per sprite. Sprites can be interpreted as prototypical images and can include segmentation, encoded as a transparency mask. Depending on the approach, $C ^ { \prime }$ can be 1 (a grayscale image), 2 (a grayscale image and transparency), 3 (an RGB image), or 4 (an RGB image with a transparency channel).

• A transformation module, T (Section 3.3), which takes as input a target image I and sprites $S _ { 1 } , \cdots , S _ { K }$ , and outputs a set of transformed sprites $\bar { S } ^ { I }$ . Transformation typically includes color and spatial transformations. The transformed sprites are images of the same size as the input image I, with an optional transparency channel. Note that this module can predict several transformations for each sprite, enabling the modeling of images with multiple elements, as we clarify in Section 3.1.2.

• A decision module, P (Section 3.4), which predicts probabilities $p ^ { I }$ for each of the transformed sprites to be used in the reconstruction of the input image.

• A reconstruction loss, L (Section 3.5), which evaluates how well the transformed sprites associated with the predicted probabilities explain the input image, and with which the model is optimized.

These components and the losses correspond to an image formation model, $\mathcal { C } ( \bar { S } , p )$

## 3.1.2 Layered Image Decomposition

For layered image decomposition, one typically assumes a maximum number of layers L. Each sprite $S _ { k }$ for $k \in \{ 1 , \ \cdots , K \}$ is then transformed into L sprites $\bar { S } _ { k , l } ^ { I } \in \mathbb { R } ^ { W \times H \times C ^ { \prime } }$ for $l \in \{ 1 , \cdots , L \}$ , with $C ^ { \prime } = C + 1$ leading to a set of $K \times L$ transformed sprites $\hat { S } ^ { I } = ( \hat { S } _ { 1 , 1 } ^ { I } , \cdot \cdot \cdot , \hat { S } _ { K , L } ^ { I } )$ . Sprites are selected according to $p ^ { I } \in [ 0 , 1 ] ^ { K \times L }$ . Note that one of the sprites can be used as an empty sprite, $i . e .$ , frozen and completely transparent, to allow modeling a variable number of objects. Background can be modeled using one or several specific opaque sprites, possibly with particular constraints (e.g., having a uniform color) and be associated with their own specific transformations. To simplify notation, we do not diferentiate background sprites from the other sprites. In our experiments on layered image decomposition, we always model the background with a single sprite. The image formation model, C, composites the transformed background sprite with the sprites from the following layers. To better handle occlusion, we follow DTI-Sprites (Monnier et al., 2021) and predict a matrix defining the order of the layers.

## 3.1.3 Clustering

In the case of clustering, the simplest scenario (Monnier et al., 2020) is to consider a single-layer image model using only completely opaque sprites. In that case, the set of transformed sprites is $\breve { \bar { S } } ^ { I } = \breve { ( S _ { 1 , 1 } ^ { I } } , \cdots , \bar { S } _ { K , 1 } ^ { I } )$ with for all $k \in \{ 1 , \ \cdot \cdot \ , K \} , \ { \bar { S } } _ { k . 1 } ^ { I } \in \mathbb { R } ^ { W \times H \times C }$ the transformed version of sprite $S _ { k }$ . Note that if there are no transformations, and the $L _ { 2 }$ loss between the input and the transformed sprite that best approximates it is optimized, this model boils down to standard K-means (MacQueen, 1967; Bottou & Bengio, 1994).

Another approach that typically leads to better results for more complex images (Monnier et al., 2021) is to explicitly model the background using a background sprite and the diferent clusters with sprites including a transparency channel, and thus consider a 2-layer model. The image formation model, ${ \mathcal { C } } ,$ composites the transformed background sprite with the other transformed sprites depending on the output $p ^ { I } \doteq [ 0 , 1 ] ^ { K }$ of the selection module.

Both of these approaches can be seen as specific cases of layered image decomposition and leverage the same modules, enabling us to start our analysis by focusing on the simpler clustering scenario.

## 3.2 Sprite Generation Module

The sprites $S _ { 1 } , \cdots , S _ { K }$ are the visual representation of the recurrent patterns identified by the model in the target image collection. They are thus common to all input images I, they are themselves modeled as images – color or grayscale, and associated or not with a transparency mask – and they can be learned with diferent strategies.

## 3.2.1 Learning Pixel Values

Directly learning the sprite, i.e. setting each sprite’s pixel values as learnable parameters, is the simplest choice and has been used in Monnier et al. (2020; 2021)

## 3.2.2 Decoding Learned Latent Variables with a Generator Network (MLP or U-Net)

Motivated by the possibility of using latent variables to link sprite generation and clustering, Smirnov et al. (2021) proposes to learn K latent vectors $z _ { 1 } , \cdots , z _ { K }$ and a generation network G that takes as input those latent vectors, and outputs the corresponding sprite $S _ { k } = G ( z _ { k } )$ . Note that while generated by a network, the sprites still do not depend on the input image I, and that once the network is trained, they could be computed once and for all, without using the generator network. Following Siglidis et al. (2024), we explore the use of a Multi-Layer Perceptron (MLP) or a U-Net architecture (Ronneberger et al., 2015) (U-Net) as the generation network.

## 3.3 Transformation Module

Sprite-based approaches account for variations in the appearance of objects in terms of shape or color by explicitly modeling them. Given an input image I, they predict one (for clustering) or several (for image decomposition) transformations for each sprite. The family of transformations that are available and the way in which they are learned are important hyperparameters, and the optimal choice depends on the target dataset. Transformations typically include (i) spatial transformations, modeled with Spatial Transformer Networks (Jaderberg et al., 2015), and (ii) afine color transformation, where parameters are predicted and applied on the sprite values. They may include more specific transformations, such as morphological transformations to model stroke width for the MNIST dataset (LeCun et al., 2010). There are several key design choices in this transformation learning that we explore.

## 3.3.1 Curriculum Learning

Because transformations could model dramatic changes, curriculum learning is the key to progressively learning meaningful transformations. We explore various curriculum scheduling strategies. To study them, we first decided on a fixed order of transformation by increasing complexity, as visualized in Figure 3b: no transformation, afine color transformation, afine spatial transformation, morphological transformation, Thin Plate Spline (TPS) transformation, and projective transformation. With all transformations initialized as the identity function, we then tested diferent strategies:

• all: optimizing all transformations together from the start,

• id+rest: learning first without any transformation, then optimizing all transformations together,

• id+g1+g2 : grouping transformations into three groups – (id) no transformation, (g1) afine color and spatial transformations, $\left( \mathrm { g 2 } \right)$ other transformations – and adding each group of transformations into the optimization one-by-one, and

• one-by-one: adding each of them into the optimization one-by-one.

Note that for each dataset, we only use transformations relevant to the dataset (see Appendix Table 12).

## 3.3.2 Sprite-Specific vs. Shared Transformations

Another question we explore is the possibility and consequences of sharing the transformations among sprites. Intuitively, one could expect the sprites to be better aligned if the same transformations are applied to all sprites, while an architecture that applies specific transformations to all sprites might be more powerful. Sharing transformations might also be beneficial when modeling a large number of sprites.

## 3.4 Decision Module

A crucial problem of sprite-based approaches is deciding which transformed sprites to use to reconstruct a specific image. We consider two types of solutions.

## 3.4.1 Minimum Loss

A simple approach is to choose the sprites that minimize the loss (Bottou & Bengio, 1994; Monnier et al., 2020). However, this means that (i) during training, only sprites that are selected receive gradients, and thus some might never be used, which requires specific re-assignment strategies, and (ii) when modeling images with multiple objects, the number of possible sprite combinations is exponential in the number of objects, which complicates optimization. Note that this approach can be seen as a deterministic layer predicting one-hot probability vectors $p ^ { I }$ , and we refer to it as Min-Loss.

## 3.4.2 Probability Prediction

Another approach is to use a neural network to predict which transformed sprites should be used for a specific input image I by predicting probability distributions among transformed sprites. While this is much more in line with common deep learning paradigms, we show experimentally that jointly learning the sprites, their transformations, and the selection of the best sprites is a challenging optimization problem, which requires using many regularization functions that make the method more specific and less robust.

The more standard architecture to predict such a probability distribution is a network that takes as input the target image I and finishes with a linear layer and a softmax, which we refer to as linear mapping. However, MarioNette (Smirnov et al., 2021) proposes having a network instead predict classification weights from latent variables, shared with the sprite generation module, which are then compared with the input image features, before applying a softmax. We refer to this approach as weight prediction.

Finally, because what is ultimately needed is a binary selection of the sprites, we experimented with replacing the softmax by Gumbel softmax (Jang et al., 2017; Maddison et al., 2017), similar to StampNet (Visser et al., 2019). However, while StampNet uses Gumbel softmax with binary selection, we use Gumbel softmax with soft selection, which consistently led to better performances.

## 3.5 Composition Model and Training Criteria

We decompose the training loss as a reconstruction loss, ${ \mathcal { L } } _ { \mathrm { r e c } } ,$ and a regularization loss, $\mathcal { L } _ { \mathrm { r e g } } \mathrm { : }$

$$
\mathcal {L} = \mathcal {L} _ {\text { rec }} + \mathcal {L} _ {\text { reg }}.\tag{1}
$$

We study two reconstruction losses, which actually correspond to two diferent composition models $\mathcal { C } ( \bar { S } , p )$ as well as diferent regularization losses.

## 3.5.1 Composition Model and Reconstruction Loss

The composition model, $\mathcal { C } ( \bar { S } ^ { I } , p ^ { I } )$ , composites transformed sprites into an image. The first way to see this model is to consider that it can only select sprites in a binary way, and thus, the loss should be a weighted sum of reconstruction errors of each sprite selection weighted by their probability $( \mathcal { L } _ { 0 - 1 } )$ . The second way to build composite sprites is by weighting transformed sprites according to predicted probabilities $p ,$ then reconstructing images with composite sprites $\left( \mathcal { L } _ { \mathrm { c o m p } } \right)$

More formally, let us define $\mathcal { C } ^ { L }$ the standard alpha-blending composition of L images $( A _ { 1 } , \alpha _ { 1 } ) , \cdot \cdot \cdot , ( A _ { L } , \alpha _ { L } )$ where for $l = 1 , \cdots , L$ the $A _ { l }$ are RGB images and $\alpha _ { l }$ their associates transparency channels:

$$
\mathcal {C} ^ {L} \left(\left(A _ {1}, \alpha_ {1}\right), \dots , \left(A _ {L}, \alpha_ {L}\right)\right) = \sum_ {l = 1} ^ {L} \left(\alpha_ {l} \prod_ {k = l + 1} ^ {L} \left(1 - \alpha_ {k}\right)\right) A _ {l},\tag{2}
$$

where the product is 1 if empty and multiplications are to be understood pixelwise. Let us consider probabilities $p ^ { \bar { I } } \in \mathbb { R } ^ { K \times L }$ and transformed sprites $\bar { S } _ { k , l } ^ { I } \in \mathbb { R } ^ { W \times H \times C }$ for all $k \in \{ 1 , \ \cdots , K \}$ and $l \in \{ 1 , \cdots , L \}$ Then $\mathcal { L } _ { 0 - 1 }$ is defined by:

$$
\mathcal {L} _ {0 - 1} (\bar {S} ^ {I}, p ^ {I}) = \sum_ {(k _ {1}, \dots k _ {L}) \in \{0, \dots , K \} ^ {L}} \left(\prod_ {l = 1} ^ {L} p _ {k _ {l}, l} ^ {I}\right) | | I - \mathcal {C} ^ {L} (\bar {S} _ {k _ {1}, 1} ^ {I}, \dots , \bar {S} _ {k _ {L}, L} ^ {I}) | | _ {2} ^ {2},\tag{3}
$$

and $\mathcal { L } _ { \mathrm { c o m p } }$ is defined by:

$$
\mathcal {L} _ {\mathrm{comp}} (\bar {S} ^ {I}, p ^ {I}) = | | I - \mathcal {C} ^ {L} (\sum_ {k = 1} ^ {K} p _ {k, 1} ^ {I} \bar {S} _ {k, 1} ^ {I}, \dots , \sum_ {k = 1} ^ {K} p _ {k, L} ^ {I} \bar {S} _ {k, L} ^ {I}) | | _ {2} ^ {2}.\tag{4}
$$

As can be seen from the equations, $\mathcal { L } _ { 0 - 1 }$ requires to compute $K ^ { L }$ composite images, which is computationally prohibitive for large numbers of sprites and layers, while $\mathcal { L } _ { \mathrm { c o m p } }$ only requires computing L composed sprites and a single composite images. However, $\mathcal { L } _ { \mathrm { c o m p } }$ corresponds to an image composition model where diferent transformed sprites can be merged, which might lead to undesired optima where objects are represented by overlapping multiple sprites. Note that in the case where $p ^ { I }$ is binary, we have $\mathcal { L } _ { 0 - 1 } = \mathcal { L } _ { \mathrm { c o m p } }$

## 3.5.2 Regularizations

We consider three regularization losses.

First, $\mathcal { L } _ { \mathrm { f r e q } }$ attempts to prevent some sprites from never being used, by penalizing using a sprite with a frequency lower than a scalar value $\epsilon \in [ 0 , 1 ]$ :

$$
\mathcal {L} _ {\mathrm{freq}} = \sum_ {k = 1} ^ {K} \max \left(0, \epsilon - \frac {1}{| D |} \sum_ {I} \sum_ {l = 1} ^ {L} p _ {k, l} ^ {I}\right),\tag{5}
$$

where in practice the loss is computed over a batch of images I. Note that DTI-Sprites (Monnier et al., 2021) has a re-assignment strategy for unused sprites that plays a similar role and has a similar minimum frequency hyperparameter ϵ.

Second, ${ \mathcal { L } } _ { \mathrm { b i n } }$ encourages one-hot probability vectors $p ^ { I }$ , and thus attempts to avoid several transformed sprites being used together to reconstruct an object. Thus, it is particularly meaningful to regularize ${ \mathcal { L } } _ { \mathrm { c o m p } } .$ Following Smirnov et al. (2021), we define ${ \mathcal { L } } _ { \mathrm { b i n } }$ by:

$$
\mathcal {L} _ {\mathrm{bin}} = \frac {1}{K} \sum_ {k = 1} ^ {K} \operatorname{Beta} (2, 2) \left(p _ {k} ^ {I}\right),\tag{6}
$$

where the probability density function of the Beta distribution is given by:

$$
f (x; \alpha , \beta) = \left\{ \begin{array}{l l} \frac {x ^ {\alpha - 1} (1 - x) ^ {\beta - 1}}{B (\alpha , \beta)} & \text { for } 0 <   x <   1 \\ 0 & \text { otherwise } \end{array} \right.
$$

where $\alpha > 0$ and $\beta > 0$ are the shape parameters, and $B ( \alpha , \beta )$ is the Beta function, defined as:

$$
B (\alpha , \beta) = \int_ {0} ^ {1} t ^ {\alpha - 1} (1 - t) ^ {\beta - 1} d t = \frac {\Gamma (\alpha) \Gamma (\beta)}{\Gamma (\alpha + \beta)},
$$

where, $\Gamma ( \cdot )$ is the Gamma function, which generalizes the factorial function $( n - 1 ) !$

Third, following Monnier et al. (2021), $\mathcal { L } _ { \mathrm { e m p t y } }$ encourages the model to use as few sprites as possible to reconstruct an image, and attempts to avoid failure cases like sprites used with a high transparency to better reconstruct details of the images. It penalizes the use of non-empty sprites, and writing e the index of the empty sprite can be defined as:

$$
\mathcal {L} _ {\mathrm{empty}} = \sum_ {l = 1} ^ {L} (1 - p _ {e, l} ^ {I}) .\tag{7}
$$

$\mathcal { L } _ { \mathrm { r e g } }$ is defined as a weighted sum of these three regularization losses:

$$
\mathcal {L} _ {\mathrm{reg}} = \lambda_ {\mathrm{freq}} \mathcal {L} _ {\mathrm{freq}} + \lambda_ {\mathrm{bin}} \mathcal {L} _ {\mathrm{bin}} + \lambda_ {\mathrm{empty}} \mathcal {L} _ {\mathrm{empty}},\tag{8}
$$

with $\lambda _ { \mathrm { f r e q } } , \lambda _ { \mathrm { b i n } }$ and $\lambda _ { \mathrm { e m p t y } }$ scalar hyperparameters.

## 4 Analysis on Clustering

In this section, we analyze single-layer sprite-based approaches on clustering, for which experiments are faster, and datasets are more diverse than for unsupervised image decomposition, and we leverage this analysis to define a new approach for sprite-based clustering. Section 4.1 introduces the details of our experimental setup. Sections 4.2 to 4.4 present comparative analysis of approaches through experiments on Sprite Generation, Transformation, Decision, and Training Criteria (see Fig. 3 for an overview and terminology). Finally, Section 4.5 compares our clustering approach with the state-of-the-art.

## 4.1 Experimental Setup

## 4.1.1 Datasets

We conducted experiments on 8 datasets with diferent characteristics: MNIST (LeCun et al., 2010), ColoredMNIST (Arjovsky et al., 2019), FashionMNIST (Xiao et al., 2017), AfNIST (Tieleman, 2013), USPS (Hull, 1994), FRGC (Phillips et al., 2005), SVHN (Netzer et al., 2011), and GTSRB-8 (Stallkamp et al., 2012) (detailed in the Section .1). Digit datasets (LeCun et al., 2010; Arjovsky et al., 2019; Hull, 1994; Netzer et al., 2011) difer in complexity, ranging from grayscale digits to real-world RGB street number images. The other datasets tackle fashion items (Xiao et al., 2017), faces (Phillips et al., 2005), and trafic signs (Stallkamp et al., 2012), ofering a diversity of challenges.

Table 2: Analysis of the sprite generation module for clustering. We report the performances of sprite generation approaches. We report accuracy (%) and standard error over 10 runs. : one-by-one, sprite-specific transformation, : Min-Loss, $: \mathcal { L } _ { \mathrm { c o m p = 0 - 1 } }$ , reassignment.

<table><tr><td>Module</td><td>Init.</td><td>MNIST</td><td>ColoredMNIST</td><td>FashionMNIST</td><td>AffNIST</td><td>USPS</td><td>FRGC</td><td>SVHN</td><td>GTSRB-8</td><td>Average</td></tr><tr><td colspan="11">Pixel Space</td></tr><tr><td>Pixels</td><td>sample</td><td> $97.2 \pm 0.0$ </td><td> $94.5 \pm 1.5$ </td><td> $\underline{58.3 \pm 0.6}$ </td><td> $93.3 \pm 2.0$ </td><td> $\underline{86.3 \pm 2.0}$ </td><td> $\underline{40.4 \pm 0.8}$ </td><td> $42.8 \pm 2.4$ </td><td> $\underline{51.4 \pm 1.5}$ </td><td>70.5</td></tr><tr><td>Pixels</td><td>random</td><td> $97.2 \pm 0.0$ </td><td> $95.5 \pm 1.3$ </td><td> $57.2 \pm 0.7$ </td><td> $89.5 \pm 2.0$ </td><td> $84.0 \pm 0.5$ </td><td> $\underline{40.8 \pm 0.7}$ </td><td> $42.2 \pm 2.0$ </td><td> $\underline{51.2 \pm 0.6}$ </td><td>69.7</td></tr><tr><td colspan="11">Latent Space</td></tr><tr><td>MLP</td><td rowspan="2">random</td><td> $97.1 \pm 0.0$ </td><td> $94.3 \pm 1.5$ </td><td> $58.9 \pm 0.7$ </td><td> $95.7 \pm 1.3$ </td><td> $85.5 \pm 2.3$ </td><td> $40.3 \pm 0.4$ </td><td> $\underline{45.8 \pm 1.2}$ </td><td> $51.1 \pm 1.6$ </td><td>71.1</td></tr><tr><td>UNet</td><td> $97.1 \pm 0.1$ </td><td> $\underline{94.8 \pm 1.5}$ </td><td> $57.9 \pm 1.3$ </td><td> $\underline{94.5 \pm 1.8}$ </td><td> $\underline{86.6 \pm 1.5}$ </td><td> $33.7 \pm 1.8$ </td><td> $\underline{45.8 \pm 2.4}$ </td><td> $50.3 \pm 1.0$ </td><td>70.1</td></tr></table>

## 4.1.2 Training Details and Evaluation

We report the mean accuracy over all samples for clustering using Hungarian matching (Kuhn, 1955) for cluster-to-class assignments. To evaluate the impact of our regularization losses, we provide a sensitivity analysis in Section .2.2 over a sequential grid search for the regularization weights. This analysis shows that performance variance across the searched ranges is low, establishing that while $\mathcal { L } _ { \mathrm { f r e q } }$ is crucial to prevent cluster collapse, the model remains robust to minor hyperparameter variations. Details of the training setup and complete hyperparameter configurations are provided in Section .2. Unless stated otherwise, we report the mean and standard error over 10 runs for each experiment.

## 4.1.3 Reference Setting

We sequentially evaluate the influence of each of the key component we have identified, starting from the DTI-Clustering setting (Monnier et al., 2020), which demonstrated competitive results for clustering, and which is closest to the K-means baseline. Note that our notion of a sprite encompasses the notion of prototype used in DTI-Clustering. Then, in each section, we define a new reference setting for each of our component, depending on our experimental analysis.

## 4.2 Sprite Generation Module

As detailed in Section 3.2 and Figure 3a, we compare directly learning pixel values and learning sprites through a generator network. When learning pixel values, we compare initializing the sprites randomly or from a random sample, similar to the original DTI-Clustering (Monnier et al., 2020). When learning sprites in latent space, we compare using a two-layer MLP and a UNet. For the MLP, we learn a latent representation of size 128 and use a hidden layer with 128 units. For UNet, we used a latent representation with the same dimension as the sample sprite and the architecture of Siglidis et al. (2024).

![](images/58365af81a56ce069f15b3d0acfd261fc32e4f6477cd70ce64ef89e69f313cd0.jpg)

![](images/27538725c49ca9fbc40d73aea991b39b2299d1ac914cce7c3487942d6f1c96ee.jpg)

![](images/8ecb5633f5e3618738c39e5963ef5b2d5a0d96a1902cf03a4f4d98316d8a648a.jpg)  
Figure 4: Training loss for diferent sprite generation modules. We show the average loss over 10 runs for 3 datasets. For all datasets, learning sprites through a generator network converges faster. Better seen in the digital version.

Our results, reported in Table 2, show that the best performing approach de-

pends on the dataset. Learning sprites with an MLP leads to slightly better results on average, it is within or close to the error margin of the best method on all datasets, and clearly improving on pixel-based approaches on AfNIST and SVHN. Moreover, an analysis of the training loss curves shown in Fig. 4 shows that learning sprites through generator networks leads to clearly faster convergence. We thus adopt learning sprites through an MLP for the rest of our analysis.

## 4.3 Transformation Module

As explained in Section 3.3, we explore diferent constraints on the deformation module, namely diferent curriculum and weight-learning strategies.

## 4.3.1 Curriculum Learning

Table 3: Efect of curriculum learning on the transformation module. We explore diferent curriculum strategies. We report accuracy (%) and standard deviation over 10 runs. : MLP, : spritespecific transformation, : Min-Loss, : $\mathcal { L } _ { \mathrm { c o m p { = } 0 - 1 } }$ , reassignment.

<table><tr><td rowspan="2">Dataset</td><td colspan="4">Curriculum strategy</td></tr><tr><td>all</td><td>id+rest</td><td>id+g1+g2</td><td>one-by-one</td></tr><tr><td>MNIST</td><td>88.1±2.6</td><td>95.8±1.1</td><td>95.8±0.9</td><td>97.1±0.0</td></tr><tr><td>ColoredMNIST</td><td>82.8±2.7</td><td>83.9±2.4</td><td>94.2±1.8</td><td>94.3±1.5</td></tr><tr><td>FashionMNIST</td><td>56.0±1.2</td><td>58.7±1.8</td><td>57.8±0.9</td><td>58.9±0.7</td></tr><tr><td>AffNIST</td><td>83.1±4.0</td><td>81.7±3.0</td><td>95.8±1.4</td><td>95.7±1.3</td></tr><tr><td>USPS</td><td>81.3±2.1</td><td>86.0±1.8</td><td>83.2±1.2</td><td>85.5±2.3</td></tr><tr><td>FRGC</td><td>34.9±0.9</td><td>34.5±0.5</td><td>39.1±0.5</td><td>40.3±0.4</td></tr><tr><td>SVHN</td><td>32.6±2.2</td><td>33.3±0.4</td><td>45.8±1.2</td><td>45.8±1.2</td></tr><tr><td>GTSRB-8</td><td>49.3±1.5</td><td>51.3±1.8</td><td>51.1±1.6</td><td>51.1±1.6</td></tr><tr><td>Average</td><td>63.5</td><td>65.7</td><td>70.4</td><td>71.1</td></tr></table>

In Table 3, we report results using various curriculum strategies to learn the transformations presented in Section 3.3. They show that curriculum is critical for good performance. A 2-step-only curriculum, which can be interpreted as a K-means initialization followed by a full unfreeze of the network, is not suficient, while splitting transformations into two groups already leads to good results. One-by-one curriculum performs best, and we thus continue using one-by-one curriculum for the rest of our analysis.

## 4.3.2 Shared Transformations

Table 4: Efect of sharing transformations among sprites in the transformation module. We report accuracy (%) and standard deviation over 10 runs. : MLP, : one-by-one, : Min-Loss, $: \mathcal { L } _ { \mathrm { c o m p = 0 - 1 } } ,$ reassignment.

<table><tr><td>Dataset</td><td>Shared transfo.</td><td>Sprite-specific transfo.</td></tr><tr><td>MNIST</td><td>91.9±2.2</td><td>97.1±0.0</td></tr><tr><td>ColoredMNIST</td><td>92.6±2.0</td><td>94.3±1.5</td></tr><tr><td>FashionMNIST</td><td>57.0±0.4</td><td>58.9±0.7</td></tr><tr><td>AffNIST</td><td>86.4±2.8</td><td>95.7±1.3</td></tr><tr><td>USPS</td><td>84.4±2.3</td><td>85.5±2.3</td></tr><tr><td>FRGC</td><td>41.1±0.6</td><td>40.3±0.4</td></tr><tr><td>SVHN</td><td>34.3±0.1</td><td>45.8±1.2</td></tr><tr><td>GTSRB-8</td><td>49.2±1.2</td><td>51.1±1.6</td></tr><tr><td>Average</td><td>67.1</td><td>71.1</td></tr></table>

Sharing transformations among sprites would intuitively put them in the same “reference frame” which would be beneficial for qualitative analysis. We visualize this efect in Fig. 5 on the ColoredMNIST and AfNIST datasets. When transformations are not shared, sprites have non-uniformed colors and positions, while they are much more consistent when transformations are shared. Although this qualitative property would b desirable, we found in the quantitative results reported in Table 4 that sharing transformations significant deteriorates quantitative performances. We thus keep sprite-specific transformations for each sprite in the rest of the analysis.

![](images/0e0bd14df5b8eab663f3581258ed7b9716cb9f5baea0b03cd127ec831ab8e384.jpg)

![](images/c5b6aba202b6dff321a9201aac641ba97af7a7cdc85a960cdd289e14870d25e5.jpg)  
Figure 5: Qualitative efect of sharing transformations among sprites in the transformation module. We compare on (a) ColoredMNIST (Arjovsky et al., 2019) and (b) AfNIST (Tieleman, 2013) the sprites learned with sprite-specific transformations (top rows) with the ones learned with shared transformations (bottom rows). Sharing the transformations among sprites encourages them to be more uniform, e.g., have similar (a) colors and (b) spatial location.

## 4.4 Decision Module and Training Criteria

Training criteria and decision modules are closely related. Thus, we first analyze jointly decision module and training criteria, then study the impact of regularizations.

## 4.4.1 Reconstruction Loss and Decision

Table 5: Results of diferent decision modules ( ) and training criteria ( ). We experimented with the training criteria and decision modules, along with Gumbel softmax. We report accuracy (%) and standard deviation over 10 runs. We train all models and the baseline (second row) until convergence, which might mean a diferent number of iterations for diferent models. : MLP, : one-by-one, sprite-specific transformation.

<table><tr><td> $\mathcal{L}_{\text{rec}}$ </td><td> $p_k$ </td><td>MNIST</td><td>ColoredMNIST</td><td>FashionMNIST</td><td>AffNIST</td><td>USPS</td><td>FRGC</td><td>SVHN</td><td>GTSRB-8</td><td>Average</td></tr><tr><td rowspan="2"> $\mathcal{L}_{\text{comp}} = \mathcal{L}_{0-1}$ </td><td>Min-Loss</td><td>92.4±1.6</td><td>71.0±2.1</td><td>58.3±0.6</td><td>89.2±2.2</td><td>82.1±1.7</td><td>30.2±0.7</td><td>47.1±1.8</td><td>47.1±1.1</td><td>64.7</td></tr><tr><td>w/ reassignment</td><td>96.5±0.5</td><td>92.0±2.0</td><td>59.6±0.7</td><td>97.3±0.0</td><td>88.4±2.9</td><td>41.1±0.6</td><td>43.3±2.7</td><td>49.3±1.3</td><td>70.9</td></tr><tr><td rowspan="2"> $\mathcal{L}_{0-1}$ </td><td>weight prediction</td><td>86.6±1.2</td><td>42.0±3.2</td><td>55.3±0.9</td><td>66.6±4.1</td><td>79.9±1.4</td><td>11.2±0.6</td><td>33.1±0.9</td><td>51.7±0.2</td><td>53.3</td></tr><tr><td>linear mapping</td><td>88.8±1.5</td><td>33.0±4.9</td><td>54.7±1.1</td><td>55.5±1.7</td><td>73.7±3.1</td><td>17.9±0.7</td><td>31.1±1.3</td><td>51.6±0.4</td><td>50.8</td></tr><tr><td rowspan="4"> $\mathcal{L}_{\text{comp}}$ </td><td>weight prediction</td><td>72.4±0.9</td><td>50.5±3.6</td><td>35.1±1.3</td><td>72.7±2.2</td><td>54.3±2.1</td><td>40.2±0.9</td><td>19.7±0.3</td><td>38.2±1.0</td><td>47.9</td></tr><tr><td>w/ Gumbel softmax</td><td>93.2±1.6</td><td>47.7±4.5</td><td>60.6±0.4</td><td>75.8±1.4</td><td>82.1±0.2</td><td>38.7±0.8</td><td>34.7±0.7</td><td>50.3±0.1</td><td>60.4</td></tr><tr><td>linear mapping</td><td>72.1±1.4</td><td>47.5±1.6</td><td>36.3±1.1</td><td>66.9±0.9</td><td>54.3±2.0</td><td>40.4±1.1</td><td>19.9±0.5</td><td>38.5±0.1</td><td>47.0</td></tr><tr><td>w/ Gumbel softmax</td><td>96.5±0.1</td><td>53.2±4.3</td><td>60.7±0.8</td><td>75.4±2.3</td><td>82.2±0.4</td><td>39.5±1.3</td><td>33.9±0.5</td><td>50.0±0.2</td><td>61.4</td></tr></table>

In Table 5, we compare the reconstruction losses we introduced – namely $ { \mathcal { L } } _ { 0 - 1 }$ defined in Eq. (3) and $\mathcal { L } _ { \mathrm { c o m p } }$ defined in Eq. (4) – alongside the diferent decision modules. Min-Loss selection, for which both losses are the same, using a cluster re-assignment strategy (Monnier et al., 2020) (Table 5 row 2) shows overall better performance than training the network to predict the sprite selection. This higher performance is largely due to the implicit regularization given by the empty cluster reassignment strategy proposed in (Monnier et al., 2020) (Table 5 rows 1 and 2).

Qualitatively, the main failure case of $\mathcal { L } _ { \mathrm { c o m p } }$ is to compose a layer from several sprites, as can be seen in Fig. 6a for MNIST, where a 9 digit is reconstructed using a circle and a loop, and in Fig. 6b for FRGC, where diferent sprites are combined to model lighting efects. As optimizing reconstruction by composition is not the targeted behavior for clustering, we experimented with replacing softmax activation with Gumbel softmax for this $\mathcal { L } _ { \mathrm { c o m p } }$ , both with linear mapping and weight prediction (Table 5 rows 6 and 8). This resulted in a significant improvement in performances of more than 10% on average. While performances remains lower than with Min-Loss selection with reassignment by almost 10%, this led to the best results with a predicted sprite selection, almost on par with Min-Loss selection without re-assignment regularization. Learning sprite selection is appealing as it does not require to test all selection possibilities, as in Min-Loss selection, which will be prohibitively costly when using multiple layers. Because we obtained slightly better performances with linear mapping than classification weight prediction, and because it is conceptually simpler, we use linear mapping for the rest of the paper, and explore if its performance could be further improved using additional regularization losses.

![](images/f327453b183bb01212ada9f817cf06dd1669bb02884223dcff64e678374ba2a6.jpg)  
(a) MNIST LeCun et al. (2010)

![](images/c0f8ed8d5f4b6ca542a126f8e2a1f10ea56160e754438cfb4994ee2787389633.jpg)  
(b) FRGC Phillips et al. (2005)  
Figure 6: Qualitative results with diferent training criteria. Compared with weighting the reconstruction loss for each sprite $( \mathcal { L } _ { 0 - 1 }$ , top rows), weighting transformed sprites and composing to reconstruct $( \mathcal { L } _ { \mathrm { c o m p } }$ , bottom rows) results in (a) sprites representing parts of the objects instead of the object itself and (b) sprites focusing on the distinct characteristics of a subject and using composition to model shading efects.

Table 6: Efect of regularization. Experiments on regularization losses with two training criteria and Gumbel softmax. We report accuracy (%) and standard deviation over 10 runs. : MLP, : one-by-one, sprite-specific transformation, : linear mapping.

<table><tr><td> $\mathcal{L}_{\text{rec}}$ </td><td> $\mathcal{L}_{\text{freq}}$ </td><td> $\mathcal{L}_{\text{bin}}$ </td><td>MNIST</td><td>ColoredMNIST</td><td>FashionMNIST</td><td>AffNIST</td><td>USPS</td><td>FRGC</td><td>SVHN</td><td>GTSRB-8</td><td>Average</td></tr><tr><td rowspan="2"> $\mathcal{L}_{0-1}$ </td><td>-</td><td>-</td><td>88.8±1.5</td><td>33.0±4.9</td><td>54.7±1.1</td><td>55.5±1.7</td><td>73.7±3.1</td><td>17.9±0.7</td><td>31.1±1.3</td><td>51.6±0.4</td><td>50.8</td></tr><tr><td>√</td><td>-</td><td>98.2±0.0</td><td>93.1±2.1</td><td>57.6±1.5</td><td>97.1±0.1</td><td>82.8±0.1</td><td>41.1±0.6</td><td>38.0±2.0</td><td>57.5±0.2</td><td>70.7</td></tr><tr><td rowspan="8"> $\mathcal{L}_{\text{comp}}$ </td><td colspan="2">w/ softmax</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>-</td><td>-</td><td>72.1±1.4</td><td>47.5±1.6</td><td>36.3±1.1</td><td>66.9±0.9</td><td>54.3±2.0</td><td>40.4±1.1</td><td>19.9±0.5</td><td>38.5±0.1</td><td>47.0</td></tr><tr><td>√</td><td>-</td><td>78.5±1.6</td><td>48.4±3.1</td><td>40.6±1.1</td><td>75.5±0.0</td><td>54.3±2.0</td><td>42.6±1.1</td><td>23.4±0.9</td><td>38.8±0.3</td><td>50.3</td></tr><tr><td>√</td><td>√</td><td>95.3±0.5</td><td>81.7±3.5</td><td>62.0±1.5</td><td>83.1±0.0</td><td>63.1±2.2</td><td>42.6±1.1</td><td>34.4±0.5</td><td>54.5±2.1</td><td>64.6</td></tr><tr><td colspan="2">w/ Gumbel softmax</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>-</td><td>-</td><td>96.5±0.1</td><td>53.2±4.3</td><td>60.7±0.8</td><td>75.4±2.3</td><td>82.2±0.4</td><td>39.5±1.3</td><td>33.9±0.5</td><td>50.0±0.2</td><td>61.4</td></tr><tr><td>√</td><td>-</td><td>96.7±0.0</td><td>95.9±0.1</td><td>60.7±0.8</td><td>94.1±1.9</td><td>82.2±0.4</td><td>44.8±0.8</td><td>35.3±0.4</td><td>53.2±1.2</td><td>70.4</td></tr><tr><td>√</td><td>√</td><td>96.7±0.0</td><td>96.0±0.1</td><td>60.7±0.8</td><td>94.1±1.9</td><td>85.3±1.1</td><td>44.8±0.8</td><td>37.6±0.3</td><td>53.2±1.2</td><td>71.1</td></tr></table>

## 4.4.2 Regularization Losses

We report in Table 6 the results obtained with using $\mathcal { L } _ { \mathrm { { f r e q } } } \left( \mathrm { E q . ~ ( 5 ) } \right)$ ) and ${ \mathcal { L } } _ { \mathrm { b i n } }$ . Note that ${ \mathcal { L } } _ { \mathrm { b i n } }$ is designed to overcome the composition issue associated to ${ \mathcal { L } } _ { \mathrm { c o m p } } .$ we do not test it with $\mathcal { L } _ { 0 - 1 }$ , and $\mathcal { L } _ { \mathrm { e m p t y } }$ does not make sense for clustering, where no empty sprite is used. We selected the regularization loss weights through a grid search for each dataset to optimize performance.

Using $\mathcal { L } _ { \mathrm { f r e q } } \left( \mathrm { E q . ~ ( 5 ) } \right)$ as a regularization significantly increases performance. Both when using $\mathcal { L } _ { 0 - 1 }$ and $\mathcal { L } _ { \mathrm { c o m p } }$ losses coupled with Gumbel softmax, this leads to results on par with Min-Loss selection with reassignment (Table 6). This again validates our claim that the superior performance of Min-Loss selection is largely due to the implicit regularization of the reassignment strategy.

To improve results obtained with $\mathcal { L } _ { \mathrm { c o m p } }$ we evaluated using $\mathcal { L } _ { \mathrm { b i n } } ~ ( \mathrm { E q . ~ ( 6 ) } )$ to encourage binary selection, similar to Smirnov et al. (2021). ${ \mathcal { L } } _ { \mathrm { b i n } }$ significantly improves the results with normal softmax while remaining worse than the best approaches, and provides a small improvement when using Gumbel softmax which already encourages binary selection. We thus propose using $\mathcal { L } _ { \mathbf { c o m p } }$ with Gumbel softmax, and $\mathcal { L } _ { \mathbf { f r e q } }$ and $\mathcal { L } _ { \bf { b i n } }$ regularizations.

## 4.5 Comparison with State-of-the-art

Given the analysis of the previous sections, we use the following design choices, summarized in Table 1, for clustering: using an MLP-based sprite generation module, with sprite-specific transformations, learned oneby-one in a curriculum fashion, a linear decision module with Gumbel softmax, a composite reconstruction loss, and frequency and binning regularization. We compare the performance of this setting with a single opaque layer (Ours-C 1 layer) to a variety of competing clustering methods in Table 7. For SVHN and

Table 7: Comparisons on clustering. We compare our results with methods that cluster over features as well as pixels. We report accuracy (%) and standard deviation for our method over 10 runs.

<table><tr><td>Method</td><td># runs</td><td>MNIST</td><td>ColoredMNIST</td><td>FashionMNIST</td><td>AffNIST</td><td>USPS</td><td>FRGC</td><td>SVHN</td><td>GTSRB-8</td></tr><tr><td colspan="10">Clustering on learned features</td></tr><tr><td>JULE Yang et al. (2016)</td><td>3</td><td>96.4</td><td>-</td><td>56.3</td><td>-</td><td>95.0</td><td>46.1</td><td>-</td><td>-</td></tr><tr><td>DEPICT Dizaji et al. (2017)</td><td>5</td><td>96.5</td><td>-</td><td>39.2</td><td>-</td><td>96.4</td><td>47.0</td><td>-</td><td>-</td></tr><tr><td>DSCDAN Yang et al. (2019)</td><td>10</td><td>97.8</td><td>-</td><td>66.2</td><td>-</td><td>86.9</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="10">+ with data augmentation and/or ad-hoc data representation</td></tr><tr><td>SpectralNet Shaham et al. (2018)</td><td>5</td><td>97.1</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>IMSAT Hu et al. (2017)</td><td>12 (5)</td><td>98.4</td><td>(10.6)</td><td>-</td><td>(18.2)</td><td>-</td><td>-</td><td>57.3</td><td>26.9</td></tr><tr><td>ADC Haeusser et al. (2019)</td><td>20</td><td>98.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>43.7</td><td>38.6</td><td>-</td></tr><tr><td>SCAE Kosiorek et al. (2019)</td><td>5</td><td>98.7</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>55.3</td><td>-</td></tr><tr><td>IIC Ji et al. (2019)</td><td>5</td><td>98.4</td><td>10.6</td><td>-</td><td>57.6</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SCAN Van Gansbeke et al. (2020)</td><td>5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>54.2</td><td>90.4</td></tr><tr><td colspan="10">Clustering on pixels</td></tr><tr><td>K-means</td><td>10</td><td>54.8</td><td>-</td><td>54.1</td><td>-</td><td>65.3</td><td>22.7</td><td>12.2</td><td>-</td></tr><tr><td>DTI-Clustering Monnier et al. (2020)</td><td>10</td><td>97.3</td><td>96.8</td><td>61.2</td><td>95.5</td><td>86.4</td><td>39.6</td><td>44.5</td><td>-</td></tr><tr><td>Ours-C 1 layer + multi-layer</td><td>10</td><td>96.7±0.0</td><td>96.0±0.1</td><td>60.7±0.8</td><td>94.1±1.9</td><td>85.3±1.1</td><td>44.8±0.8</td><td>37.6±0.3</td><td>53.2±1.2</td></tr><tr><td>DTI-Sprites Monnier et al. (2021)</td><td>10</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>63.1</td><td>89.9</td></tr><tr><td>Ours-C 2 layers</td><td>10</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>52.4±0.5</td><td>80.9±1.8</td></tr></table>

GTSRB-8, we also report our approach using a background model (Ours-C 2 layers). Note that most approaches rely on learning and clustering features, which limit the results’ interpretability, and that many leverage specific data-augmentation or representations, such as Gabor filters, which are strong priors and simplify the task. Our results are competitive with state-of-the-art on class-aware metrics, while predicting cluster selection, and learning an explicit cluster prototype and image-specific transformation. While it often performs slightly worse than DTI-Clustering, our setting does not rely on comparing each possible reconstruction to the target to assign clusters, but instead directly learns and predicts cluster selection. Thus, as shown in the next session, our approach can be directly extended into an eficient multi-layer image decomposition model.

## 5 Analysis for Multi-layer Image Decomposition

In this section, we explore how our analysis of sprite-based image models for clustering can be leveraged for multi-layer image decomposition. In the following, we first summarize our experimental setting, including datasets and metrics, and then discuss our results.

## 5.1 Experimental Setup

## 5.1.1 Datasets

We present results on the Tetrominoes (Gref et al., 2019) – images of 3 non-overlapping colored 2D blocks sampled among 19 unique ones, on black background –, the Multi-dSprites (Kabra et al., 2019) – images of up to 5 possibly overlapping colored 2D objects of diferent sizes sampled among 3 unique ones, on gray background –, and the CLEVR6 and CLEVR (Johnson et al., 2017) datasets – images of respectively up to 6 and 10 possibly overlapping colored 3D objects of diferent sizes sampled among 6 unique ones, on a simple background. More details are given in the Section .1. Note that all of these datasets are synthetic and relatively simple, but they are the main ones used in the literature for our task.

## 5.1.2 Training Details

Details of the transformations for the foreground and background are given in the Appendix Table 12. Our empirical results across four distinct datasets show that the regularization weight $\lambda _ { \mathrm { e m p t y } }$ is crucial for multilayer object decomposition, and higher scene complexity generally demands higher $\lambda _ { \mathrm { e m p t y } }$ . Details of training setup and hyperparameters for each dataset are provided in the Section .2.

Table 8: Results for multi-object semantic discovery. ( ) Sprite generation, ( ) decision and activation function, ( ) training criterion and regularization. Mean accuracy (mAcc) and average mean IoU (avg-mIoU) over classes, with standard error over 3 runs. (†): longer training, except Monnier et al. (2021) on CLEVR (in italic) obtained with the training schedule in the paper. Gray entries denote single-run results where initial performance indicated lower performance than the baseline.

<table><tr><td colspan="6">Method</td><td colspan="2">Tetrominoes</td><td colspan="2">Multi-dSprites</td><td colspan="2">CLEVR6</td><td colspan="2">CLEVR</td></tr><tr><td>DTI-Sprites Monnier et al. (2021)†</td><td>Pixels</td><td>Min-Loss</td><td>S</td><td> $\mathcal{L}_{0-1=comp}$ </td><td> $\mathcal{L}_{empty}$ </td><td> $\underline{\text{mAcc}}$ 99.5±0.2</td><td> $\underline{\text{avg-mIoU}}$ 99.2±0.3</td><td> $\underline{\text{mAcc}}$ 91.3±0.9</td><td> $\underline{\text{avg-mIoU}}$ 84.0±1.4</td><td> $\underline{\text{mAcc}}$ 79.3±2.7</td><td> $\underline{\text{avg-mIoU}}$ 64.2±3.1</td><td> $\underline{\text{mAcc}}$ 69.8±5.0</td><td> $\underline{\text{avg-mIoU}}$ 55.7±4.3</td></tr><tr><td rowspan="6">Ours-D</td><td rowspan="6">MLP</td><td rowspan="6">linear mapping</td><td rowspan="6">GS</td><td rowspan="6"> $\mathcal{L}_{comp}$ </td><td>-</td><td>74.3±1.4</td><td>64.4±1.9</td><td>65.6±0.1</td><td>54.4±0.1</td><td>66.8±0.4</td><td>49.6±0.8</td><td>59.3±7.4</td><td>43.0±5.6</td></tr><tr><td>τ</td><td>92.7±3.9</td><td>89.2±5.5</td><td>65.2</td><td>53.8</td><td>56.3</td><td>39.6</td><td>55.3</td><td>42.1</td></tr><tr><td>+  $\mathcal{L}_{freq}$ </td><td>93.9±0.4</td><td>89.9±0.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td> $\mathcal{L}_{empty}$ </td><td>-</td><td>-</td><td>65.9±0.5</td><td>54.7±0.4</td><td>74.7±1.4</td><td>57.8±1.5</td><td>70.6±0.1</td><td>55.3±0.1</td></tr><tr><td>+  $\mathcal{L}_{freq}$ </td><td>-</td><td>-</td><td>66.0±1.0</td><td>54.7±0.9</td><td>72.2±1.2</td><td>54.8±1.3</td><td>70.5±1.2</td><td>53.9±0.5</td></tr><tr><td>+  $\mathcal{L}_{bin}$ </td><td>-</td><td>-</td><td>65.4</td><td>54.2</td><td>65.0</td><td>46.5</td><td>68.0</td><td>53.1</td></tr></table>

## 5.1.3 Metrics

For our analysis in Table 8, we reported two class-aware metrics, mean accuracy (mAcc) and average mean Intersection-over-Union (avg-mIoU). We use Hungarian matching to align the predicted and ground-truth classes. Mean accuracy measures the proportion of correctly-predicted pixels in classification. The average mean IoU computes the IoU, a segmentation accuracy metric, class-wise and averages it across all classes, including background, to reflect class awareness.

To give results comparable with the metrics most frequently reported in the literature, we also report in Table 9 instance mean IoU (mIoU) and the Adjusted Rand Index computed only for the foreground (ARI-FG). Instance mean IoU measures the segmentation accuracy of predicted instances without considering the class of the prediction, but takes the background into account. ARI-FG evaluates how well pixels’ instance assignments align with the ground truth while ignoring the background. These two metrics are adopted by the literature because most existing methods focus solely on predicting instance segmentation without providing the corresponding class labels. For the few approaches that additionally provide class predictions, we also report mean accuracy (mAcc) and mean IoU averaged over classes (avg-mIoU).

## 5.2 Results

## 5.2.1 Regularizations

In Table 8, we analyze the impact of diferent regularizations on the performance of our approach and compare it to DTI-Sprites (Monnier et al., 2021). Indeed, the regularization needs are diferent from the ones in clustering. In particular, ${ \mathcal { L } } _ { \mathrm { b i n } } .$ which we adopted for clustering, prevents multiple sprites from the same layer from being combined to reconstruct diferent parts of the same object, but it does not prevent the same efect with sprites from diferent layers. Thus, in addition to ${ \mathcal { L } } _ { \mathrm { b i n } }$ and $\mathcal { L } _ { \mathrm { f r e q } } ,$ we experimented with $\mathcal { L } _ { \mathrm { e m p t y } }$ (Eq. (7)) which favors the use of an empty sprite, i.e. a sprite with a completely transparent alpha mask. Because we observed early-stage high-confidence class predictions during training, which is likely detrimental to learning, we also explored the impact of learning the Gumbel softmax temperature parameter, τ , which could mitigate this efect.

For Tetrominoes, where the number of objects is constant and equal to the number of layers, $\mathcal { L } _ { \mathrm { e m p t y } }$ and ${ \mathcal { L } } _ { \mathrm { b i n } }$ make little sense, and we only explore learning the Gumbel softmax temperature τ and $\mathcal { L } _ { \mathrm { f r e q } }$ , while we explore all regularizations for the other datasets.

Learning the Gumbel softmax temperature τ gives a huge boost to the results on Tetrominoes. One of the three runs actually matches the almost perfect performance of DTI-Sprites, emphasizing the additional complexity of learning the class prediction. Adding $\mathcal { L } _ { \mathrm { f r e q } }$ to learning τ further improves the average on Tetrominoes, but they remain below the almost perfect results of DTI-Sprites, without any run matching it. In contrast, for Multi-dSprites, CLEVR6, and CLEVR, learning τ leads to the worst results.

For Multi-dSprites, CLEVR6, and CLEVR, we thus discarded learning τ and instead experimented with $\mathcal { L } _ { \mathrm { e m p t y } } .$ , trying to better estimate the number of objects, which is the main challenge for our approach without regularization on CLEVR6 and CLEVR. On Multi-dSprites (Gref et al., 2019), which includes 3 distinct objects (square, ellipsoid, and heart), our method is still clearly outperformed by DTI-Sprites (Monnier et al., 2021), due to the fact that our model fails to discover a distinct representation of the heart shape, instead reconstructing it as a composition of two rotated ellipsoids. $\mathrm { O n }$ CLEVR6 and CLEVR, our performance with $\mathcal { L } _ { \mathrm { e m p t y } }$ is on par with DTI-Sprites, and the results are qualitatively similar. Further adding $\mathcal { L } _ { \mathrm { f r e q } }$ does not significantly change these results, and adding ${ \mathcal { L } } _ { \mathrm { b i n } }$ degrades them. The main diference between our approach and DTI-Sprites on this more challenging CLEVR dataset is the higher scalability of our approach, which we discuss in the next section.

## 5.2.2 Complexity

The major advantage of our model with respect to DTI-Sprites is that it learns to predict which sprite to select for which layer, rather than iteratively trying many sprite selections and combinations, which results in a significant improvement in time complexity, as shown in Fig. 7 on the CLEVR dataset.

The time complexity of our model scales linearly with the maximum number of objects in a scene, while the DTI-Sprites scales exponentially. Note that due to this exponential time complexity, we trained DTI-Sprites in Table 8 and Table 9 with the training schedule in the original paper, i.e. 351,900 iterations, while we could train our model until convergence, for 703,800 iterations, in less time.

## 5.2.3 Comparisons

In 9, we compare our results with the state-of-the-art on the CLEVR dataset. AST-Seg-B3-CT (Sauvalle & de La Fortelle, 2023) clearly dominates in terms of mIoU and ARI-FG, but our results are on par with most baselines for these metrics, which focus on instances and do not consider the class prediction. We thus also compared class aware metrics for methods from which can be extracted. For MarioNette (Smirnov et al., 2021), we match learned sprites in the dictionary to classes in a many-to-one manner with Hungarian matching. Because it is the best-performing instance segmentation method, we also applied Kmeans with $K = 6$ to the object features $\left( z _ { \mathrm { w h a t } } \right)$ of AST-Seg-B3-CT (Sauvalle & de La Fortelle, 2023) and clustered them,

![](images/4129c804d994e76c618111f070de601d5bcdb83cab0f5a892d6fc175d7d57f36.jpg)  
Figure 7: Complexity. The time per iteration of our approach scales linearly with the number of object layers, while that of the only other method with comparable results, DTI-Sprites (Monnier et al., 2021), scales exponentially.

leading to a class-aware adaptation of this method. The only method that achieves similar results to ours for class-aware metrics is DTI-Sprites (Monnier et al., 2021), while the other two baselines lag far behind. This emphasizes another advantage of our approach.

## 5.2.4 Qualitative Results

Qualitative examples of our decomposition with a large number of objects from CLEVR (Johnson et al., 2017) are presented in Fig. 8. They show that our model is able to recover both accurate instances and semantic segmentation with a large number of objects.

![](images/45551ae926b7a8231d2524aecaec03802fc0cf7be447316f6914209935046cb5.jpg)  
Figure 8: Qualitative results for multi-object discovery on CLEVR (Johnson et al., 2017). The three left columns show the sprites’ appearances (Frg.), masks, and combination (Sprite), including the empty sprite, and the background. The other columns show for four diferent examples, the input image, its reconstruction, semantic segmentation (Sem. Seg.), instance segmentation (Ins. Seg.), background (Bkg. Layer), and the diferent transformed sprites (Object Layers).

Table 9: Comparisons for instance segmentation with standard deviation over 3 runs. Sources for † (excluding Monnier et al. (2021)): Karazija et al. (2021) and Sauvalle & de La Fortelle (2023).

<table><tr><td>Method</td><td>class-aware</td><td colspan="4">CLEVR</td></tr><tr><td></td><td></td><td>mIoU†</td><td>ARI-FG†</td><td>mAcc</td><td>avg-mIoU</td></tr><tr><td>MONet Burgess et al. (2019)</td><td></td><td>30.7±14.9</td><td>54.5±11.4</td><td>-</td><td>-</td></tr><tr><td>IODINE Greff et al. (2019)</td><td></td><td>45.1±17.9</td><td>93.8±0.8</td><td>-</td><td>-</td></tr><tr><td>SPAIR Crawford &amp; Pineau (2019)</td><td></td><td>66.0±4.0</td><td>77.1±1.9</td><td>-</td><td>-</td></tr><tr><td>GNM Jiang &amp; Ahn (2020)</td><td></td><td>59.9±3.7</td><td>65.1±4.2</td><td>-</td><td>-</td></tr><tr><td>Slot Attention Locatello et al. (2020)</td><td></td><td>36.6±24.8</td><td>95.9±2.4</td><td>-</td><td>-</td></tr><tr><td>eMORL Emami et al. (2021)</td><td></td><td>50.2±22.6</td><td>93.3±3.2</td><td>-</td><td>-</td></tr><tr><td>Genesis-V2 Engelcke et al. (2021)</td><td></td><td>9.5±0.6</td><td>57.9±20.4</td><td>-</td><td>-</td></tr><tr><td>MarioNette Smirnov et al. (2021)</td><td>√</td><td>72.1±0.6</td><td>56.8±0.4</td><td>16.1±0.2</td><td>7.3±0.4</td></tr><tr><td>AST-Seg-B3-CT Sauvalle &amp; de La Fortelle (2023)</td><td></td><td>90.3±0.2</td><td>98.3±0.1</td><td>20.8±1.2</td><td>12.1±0.2</td></tr><tr><td>DTI-Sprites Monnier et al. (2021)</td><td>√</td><td>54.5±1.2</td><td>93.2±2.0</td><td>69.8±4.5</td><td>55.7±6.0</td></tr><tr><td>Ours-D</td><td>√</td><td>53.8±0.3</td><td>95.1±0.5</td><td>70.6±0.2</td><td>55.3±0.2</td></tr></table>

## 6 Conclusion

In this work, we introduced a unified formalization for sprite-based models, specifying their relationships, and unifying approaches for clustering and multi-layer image decomposition. This analysis clarifies the design space of methods in the literature and enables its exploration on the clustering task, which is less computationally intensive and uses more diverse, realistic datasets. This yields in turn an approach to image decomposition that learns to predict sprite selection, to avoid an exponential complexity in the number of objects per image, and maintains strong performance.

Our study ofers several key insights for layer-based image decomposition: (i) learning sprite representations via an MLP yields slightly better reconstruction, but more importantly significantly accelerates training; (ii) sharing transformation parameters across sprites acts as a structural regularizer, encouraging uniformity in color and spatial alignment, but can slightly hurt performances; (iii) opposite to the exponential cost of minloss optimization, predicting sprite probabilities and using a loss on composite sprites scales linearly with the number of objects per image, enabling to model a larger number of objects per image, but requires explicit regularization and performs slightly worse in simple cases. Moreover, our analysis emphasizes that while the standard CLEVR benchmarks seem saturated in terms of the commonly reported class-agnostic metrics, it is actually far from being the case for class-aware metrics, where our method provides state-of-the-art but imperfect results, encouraging more works in this direction.

One of the main limitations of the approach we propose is the necessity of hyperparameter selection. Although we follow standard protocols in the literature for evaluation and show some robustness to hyperparameter selection, our reliance on ground-truth validation labels to perform a grid search for regularization hyperparameters departs from a fully unsupervised deployment setting where labels are unavailable.

## Acknowledgments

Z. S. Baltacı and M. Aubry supported by the ANR project VHS ANR-21-CE38-0008, and the ERC project DISCOVER funded by the European Union’s Horizon Europe Research and Innovation program under grant agreement No. 101076028. This work was granted access to the HPC resources of IDRIS under the allocation AD011015415R1, AD011015415, and AD011014404 made by GENCI. We would like to thank Ioannis Siglidis for insightful discussions, and Robin Champenois and Ségolène Albouy for their contributions to the codebase.

## References

Roberto Annunziata, Christos Sagonas, and Jacques Calì. Jointly aligning millions of images with deep penalised reconstruction congealing. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.

Martin Arjovsky, Léon Bottou, Ishaan Gulrajani, and David Lopez-Paz. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.

Léon Bottou and Yoshua Bengio. Convergence properties of the K-means algorithms. In Advances in Neural Information Processing Systems, 1994.

Christopher P Burgess, Loic Matthey, Nicholas Watters, Rishabh Kabra, Irina Higgins, Matt Botvinick, and Alexander Lerchner. MoNet: Unsupervised scene decomposition and representation. arXiv preprint arXiv:1901.11390, 2019.

Mathilde Caron, Piotr Bojanowski, Armand Joulin, and Matthijs Douze. Deep clustering for unsupervised learning of visual features. In Proceedings of the IEEE/CVF European Conference on Computer Vision, 2018.

Mathilde Caron, Ishan Misra, Julien Mairal, Priya Goyal, Piotr Bojanowski, and Armand Joulin. Unsupervised learning of visual features by contrasting cluster assignments. In Advances in Neural Information Processing Systems, 2020.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.

Jianlong Chang, Lingfeng Wang, Gaofeng Meng, Shiming Xiang, and Chunhong Pan. Deep adaptive image clustering. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2017.

M. Cox, S. Sridharan, S. Lucey, and J. Cohn. Least-squares congealing for large numbers of images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2009.

Mark Cox, Sridha Sridharan, Simon Lucey, and Jefrey Cohn. Least squares congealing for unsupervised alignment of images. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2008.

Eric Crawford and Joelle Pineau. Spatial invariant unsupervised object detection with convolutional neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, 2019.

Kamran Ghasedi Dizaji, Amirhossein Herandi, Cheng Deng, Weidong (Tom) Cai, and Heng Huang. Deep clustering via joint convolutional autoencoder embedding and relative entropy minimization. Proceedings of the IEEE/CVF International Conference on Computer Vision, 2017.

Patrick Emami, Pan He, Sanjay Ranka, and Anand Rangarajan. Eficient iterative amortized inference for learning symmetric and disentangled multi-object representations. In Proceedings of the International Conference on Machine Learning, 2021.

Martin Engelcke, Adam R. Kosiorek, Oiwi Parker Jones, and Ingmar Posner. GENESIS: Generative scene inference and sampling with object-centric latent representations. In Proceedings of the International Conference on Learning Representations, 2020.

Martin Engelcke, Oiwi Parker Jones, and Ingmar Posner. GENESIS-v2: Inferring unordered object representations without iterative refinement. Advances in Neural Information Processing Systems, 2021.

S. M. Ali Eslami, Nicolas Heess, Theophane Weber, Yuval Tassa, David Szepesvari, Koray Kavukcuoglu, and Geofrey E. Hinton. Attend, Infer, Repeat: Fast scene understanding with generative models. In Advances in Neural Information Processing Systems, 2016.

B.J. Frey and N. Jojic. Estimating mixture models of images and inferring spatial transformations using the EM algorithm. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 1999.

B.J. Frey and N. Jojic. Transformation-invariant clustering using the EM algorithm. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2003.

Brendan J Frey and Nebojsa Jojic. Fast, large-scale transformation-invariant clustering. In Advances in Neural Information Processing Systems. MIT Press, 2001.

Klaus Gref, Rupesh Kumar Srivastava, and Jürgen Schmidhuber. Binding via reconstruction clustering. arXiv preprint arXiv:1511.06418, 2015.

Klaus Gref, Antti Rasmus, Mathias Berglund, Tele Hotloo Hao, Harri Valpola, and Jürgen Schmidhuber. Tagger: Deep unsupervised perceptual grouping. In Advances in Neural Information Processing Systems, 2016.

Klaus Gref, Sjoerd van Steenkiste, and Jürgen Schmidhuber. Neural expectation maximization. In Proceedings of the International Conference on Learning Representations, 2017.

Klaus Gref, Raphaël Lopez Kaufman, Rishabh Kabra, Nick Watters, Chris Burgess, Daniel Zoran, Loic Matthey, Matthew M. Botvinick, and Alexander Lerchner. Multi-object representation learning with iterative variational inference. In Proceedings of the International Conference on Machine Learning, 2019.

Klaus Gref, Sjoerd Van Steenkiste, and Jürgen Schmidhuber. On the binding problem in artificial neural networks. arXiv preprint arXiv:2012.05208, 2020.

Philip Haeusser, Johannes Plapp, Vladimir Golkov, Elie Aljalbout, and Daniel Cremers. Associative Deep Clustering: Training a classification network with no labels. In Thomas Brox, Andrés Bruhn, and Mario Fritz (eds.), Pattern Recognition, 2019.

Weihua Hu, Takeru Miyato, Seiya Tokui, Eiichi Matsumoto, and Masashi Sugiyama. Learning discrete representations via information maximizing self-augmented training. In Proceedings of the International Conference on Machine Learning, 2017.

Gary B. Huang, Vidit Jain, and Erik Learned-Miller. Unsupervised joint alignment of complex images. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2007.

J.J. Hull. A database for handwritten text recognition research. IEEE Transactions on Pattern Analysis and Machine Intelligence, 1994.

Max Jaderberg, Karen Simonyan, Andrew Zisserman, et al. Spatial transformer networks. Advances in Neural Information Processing Systems, 2015.

Eric Jang, Shixiang Gu, and Ben Poole. Categorical reparameterization with Gumbel-softmax. In Proceedings of the International Conference on Learning Representations, 2017.

Xu Ji, João F Henriques, and Andrea Vedaldi. Invariant information clustering for unsupervised image classification and segmentation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2019.

Jindong Jiang and Sungjin Ahn. Generative neurosymbolic machines. Advances in Neural Information Processing Systems, 2020.

Jindong Jiang, Fei Deng, Gautam Singh, and Sungjin Ahn. Object-centric slot difusion. arXiv preprint arXiv:2303.10834, 2023.

Zhuxi Jiang, Yin Zheng, Huachun Tan, Bangsheng Tang, and Hanning Zhou. Variational Deep Embedding: An unsupervised and generative approach to clustering. In International Joint Conference on Artificial Intelligence, 2016.

Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick. CLEVR: A diagnostic dataset for compositional language and elementary visual reasoning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2017.

Rishabh Kabra, Chris Burgess, Loic Matthey, Raphael Lopez Kaufman, Klaus Gref, Malcolm Reynolds, and Alexander Lerchner. Multi-object datasets. https://github.com/deepmind/multi-object-datasets/, 2019.

Ioannis Kakogeorgiou, Spyros Gidaris, Konstantinos Karantzalos, and Nikos Komodakis. SPOT: Self-training with patch-order permutation for object-centric learning with autoregressive transformers. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Laurynas Karazija, Iro Laina, and Christian Rupprecht. ClevrTex: A texture-rich benchmark for unsuper vised multi-object segmentation. In Advances in Neural Information Processing Systems Datasets and Benchmarks Track, 2021.

Adam R. Kosiorek, Hyunjik Kim, Yee Whye Teh, and Ingmar Posner. Sequential Attend, Infer, Repeat: Generative modelling of moving objects. In Advances in Neural Information Processing Systems, 2018.

Adam R. Kosiorek, Sara Sabour, Yee Whye Teh, and Geofrey E. Hinton. Stacked capsule autoencoders. In Advances in Neural Information Processing Systems, 2019.

H. W. Kuhn. The Hungarian method for the assignment problem. Naval Research Logistics Quarterly, 1955.

E.G. Learned-Miller. Data driven image models through continuous joint alignment. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2006.

Yann LeCun, Corinna Cortes, Chris Burges, et al. MNIST handwritten digit database, 2010.

Junnan Li, Pan Zhou, Caiming Xiong, and Steven Hoi. Prototypical contrastive learning of unsupervised representations. In Proceedings of the International Conference on Learning Representations, 2021.

James Chenhao Liang, Yiming Cui, Qifan Wang, Tong Geng, Wenguan Wang, and Dongfang Liu. Cluster-Former: Clustering as a universal visual learner. In Advances in Neural Information Processing Systems, 2023.

Zhixuan Lin, Yi-Fu Wu, Skand Vishwanath Peri, Weihao Sun, Gautam Singh, Fei Deng, Jindong Jiang, and Sungjin Ahn. SPACE: Unsupervised object-oriented scene representation via spatial attention and decomposition. In International Conference on Learning Representations, 2020.

Xiaoming Liu, Yan Tong, and Frederick W. Wheeler. Simultaneous alignment and clustering for an image ensemble. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2009.

Francesco Locatello, Dirk Weissenborn, Thomas Unterthiner, Aravindh Mahendran, Georg Heigold, Jakob Uszkoreit, Alexey Dosovitskiy, and Thomas Kipf. Object-centric learning with slot attention. In Advances in Neural Information Processing Systems, 2020.

Romain Loiseau, Elliot Vincent, Mathieu Aubry, and Loic Landrieu. Learnable Earth Parser: Discovering 3d prototypes in aerial scans. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

James MacQueen. Some methods for classification and analysis of multivariate observations. In Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, Volume 1: Statistics. Uni versity of California press, 1967.

Chris J. Maddison, Andriy Mnih, and Yee Whye Teh. The Concrete Distribution: A continuous relaxation of discrete random variables. In Proceedings of the International Conference on Learning Representations, 2017.

Marwan A. Mattar, Allen R. Hanson, and Erik G. Learned-Miller. Unsupervised joint alignment and clustering using bayesian nonparametrics. In Conference on Uncertainty in Artificial Intelligence, 2012.

E.G. Miller, N.E. Matsakis, and P.A. Viola. Learning from one example through shared densities on transforms. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2000.

Tom Monnier, Thibault Groueix, and Mathieu Aubry. Deep transformation-invariant clustering. In Advances in Neural Information Processing Systems, 2020.

Tom Monnier, Elliot Vincent, Jean Ponce, and Mathieu Aubry. Unsupervised layered image decomposition into object prototypes. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2021.

Nairouz Mrabah, Mohamed Bouguessa, and Riadh Ksantini. Adversarial Deep Embedded Clustering: On a better trade-of between feature randomness and feature drift. IEEE Transactions on Knowledge and Data Engineering, 2019.

Sudipto Mukherjee, Himanshu Asnani, Eugene Lin, and Sreeram Kannan. ClusterGAN: Latent space clustering in generative adversarial networks. In Proceedings of the AAAI Conference on Artificial Intelligence, 2018.

Yuval Netzer, Tao Wang, Adam Coates, Alessandro Bissacco, Bo Wu, and Andrew Y Ng. Reading digits in natural images with unsupervised feature learning. Advances in Neural Information Processing Systems Workshop on Deep Learning and Unsupervised Feature Learning, 2011.

Chuang Niu, Hongming Shan, and Ge Wang. SPICE: Semantic pseudo-labeling for image clustering. IEEE Transactions on Image Processing, 2022.

P.J. Phillips, P.J. Flynn, T. Scruggs, K.W. Bowyer, Jin Chang, K. Hofman, J. Marques, Jaesik Min, and W. Worek. Overview of the face recognition grand challenge. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2005.

Yazhou Ren, Jingyu Pu, Zhimeng Yang, Jie Xu, Guofeng Li, Xiaorong Pu, Philip S. Yu, and Lifang He. Deep Clustering: A comprehensive survey. IEEE Transactions on Neural Networks and Learning Systems, 2024.

Olaf Ronneberger, Philipp Fischer, and Thomas Brox. U-Net: Convolutional networks for biomedical image segmentation. In International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 234–241. Springer, 2015.

Bruno Sauvalle and Arnaud de La Fortelle. Unsupervised multi-object segmentation using attention and soft-argmax. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision 2023.

Maximilian Seitzer, Max Horn, Andrii Zadaianchuk, Dominik Zietlow, Tianjun Xiao, Carl-Johann Simon-Gabriel, Tong He, Zheng Zhang, Bernhard Schölkopf, Thomas Brox, and Francesco Locatello. Bridging the gap to real-world object-centric learning. In Proceedings of the International Conference on Learning Representations, 2023.

Uri Shaham, Kelly Stanton, Henry Li, Ronen Basri, Boaz Nadler, and Yuval Kluger. SpectralNet: Spectral clustering using deep neural networks. In Proceedings of the International Conference on Learning Representations, 2018.

Ioannis Siglidis, Nicolas Gonthier, Julien Gaubil, Tom Monnier, and Mathieu Aubry. The Learnable Type writer: A generative approach to text analysis. In Proceedings of the International Conference on Document Analysis and Recognition, 2024.

Gautam Singh, Fei Deng, and Sungjin Ahn. Illiterate DALL-E learns to compose. In Proceedings of the International Conference on Learning Representations, 2022.

Krishnakant Singh, Simone Schaub-Meyer, and Stefan Roth. GLASS: Guided latent slot difusion for object-centric learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Patter Recognition, 2025.

Dmitriy Smirnov, Michael Gharbi, Matthew Fisher, Vitor Guizilini, Alexei Efros, and Justin M Solomon. MarioNette: Self-supervised sprite learning. Advances in Neural Information Processing Systems, 2021.

J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. Man vs. computer: Benchmarking machine learning algorithms for trafic sign recognition. Neural Networks, 2012.

Karl Stelzner, Robert Peharz, and Kristian Kersting. Faster attend-infer-repeat with tractable probabilistic models. In Proceedings of the International Conference on Machine Learning, 2019.

Tijmen Tieleman. afNIST — cs.toronto.edu. https://www.cs.toronto.edu/\~tijmen/affNIST/, 2013. [Accessed 04-11-2025].

Wouter Van Gansbeke, Simon Vandenhende, Stamatios Georgoulis, Marc Proesmans, and Luc Van Gool. SCAN: Learning to classify images without labels. In Proceedings of the IEEE/CVF European Conference on Computer Vision, 2020.

José-Fabian Villa-Vásquez and Marco Pedersoli. Unsupervised object discovery: A comprehensive survey and unified taxonomy. arXiv preprint arXiv:2411.00868, 2024.

Pascal Vincent, H. Larochelle, Yoshua Bengio, and Pierre-Antoine Manzagol. Extracting and composing robust features with denoising autoencoders. In Proceedings of the International Conference on Machine Learning, 2008.

Joost Visser, Alessandro Corbetta, Vlado Menkovski, and Federico Toschi. StampNet: Unsupervised multiclass object discovery. In Proceedings of the IEEE International Conference on Image Processing, 2019.

J. von Kügelgen, I. Ustyuzhaninov, P. Gehler, M. Bethge, and B. Schölkopf. Towards causal generative scene models via competition of experts. In International Conference on Learning Representations Workshop on Causal Learning for Decision Making, 2020.

Xiuxi Wei, Zhihui Zhang, Huajuan Huang, and Yongquan Zhou. An overview on deep clustering. Neurocomputing, 2024.

Ziyi Wu, Jingyu Hu, Wuyue Lu, Igor Gilitschenski, and Animesh Garg. SlotDifusion: Object-centric generative modeling with difusion models. Advances in Neural Information Processing Systems, 2023.

Canqun Xiang, Zhennan Wang, Wenbin Zou, and Chen Xu. DPR-CAE: capsule autoencoder with dynamic part representation for image parsing. arXiv preprint arXiv:2104.14735, 2021.

Han Xiao, Kashif Rasul, and Roland Vollgraf. Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms. arXiv preprint arXiv:1708.07747, 2017.

Junyuan Xie, Ross Girshick, and Ali Farhadi. Unsupervised deep embedding for clustering analysis. In Proceedings of the International Conference on Machine Learning, 2016.

Jianwei Yang, Devi Parikh, and Dhruv Batra. Joint unsupervised learning of deep representations and image clusters. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2016.

Xu Yang, Cheng Deng, Feng Zheng, Junchi Yan, and Wei Liu. Deep spectral clustering using dual autoencoder network. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2019.

Sheng Zhou, Hongjia Xu, Zhuonan Zheng, Jiawei Chen, Zhao Li, Jiajun Bu, Jia Wu, Xin Wang, Wenwu Zhu, and Martin Ester. A Comprehensive Survey on Deep Clustering: Taxonomy, challenges, and future directions. ACM Computing Surveys, 2024.

## .1 Dataset Descriptions

MNIST (LeCun et al., 2010) MNIST is a widely used dataset of handwritten grayscale digits, containing 60,000 training images and 10,000 testing images.

ColoredMNIST (Arjovsky et al., 2019) Colored MNIST is built from the MNIST dataset by randomly adding color to the foreground and background, resulting in a collection of 70,000 images. Each digit image is transformed into a 3-channel representation, ofering a more complex dataset.

FashionMNIST (Xiao et al., 2017) FashionMNIST is designed as an alternative to MNIST, consisting of 60,000 fashion item images for training and 10,000 for testing. These images are grayscale and categorized into 10 classes.

AfNIST (Tieleman, 2013) Derived from MNIST, the AfNIST dataset enriches the original dataset by applying afine transformations to its digits. We employ the test split of 10,000 images to assess algorithm robustness against various transformations.

USPS (Hull, 1994) The United States Postal Service (USPS) dataset includes handwritten grayscale digit images of envelopes, containing 7,291 training samples and 2,007 testing samples.

FRGC (Phillips et al., 2005) The Face Recognition Grand Challenge (FRGC) dataset is a collection of face images in RGB space, which contains over 50,000 images of various individuals captured under diferent poses, expressions, and lighting conditions.

SVHN (Netzer et al., 2011) The Street View House Numbers (SVHN) dataset includes more than 600,000 RGB images of house numbers captured from Google Street View. It is intended for digit recognition tasks and ofers more challenging variations in terms of font styles, sizes, and cluttered backgrounds compared to MNIST.

GTSRB-8 (Stallkamp et al., 2012) The German Trafic Sign Recognition Benchmark (GTSRB) dataset subset (GTSRB-8) focuses on eight common trafic sign classes and contains more than 25,000 images for training and testing.

Tetrominoes (Gref et al., 2019) Tetrominoes contains around 60,000 images with size 35x35 featuring 3 Tetris-like shapes with diferent color and position from 19 unique shapes. Each image has a black background, and shapes do not occlude each other.

Multi-dSprites (Kabra et al., 2019) Multi-dSprites contains around 60,000 images with multiple oval, heart, or square-shaped objects with a uniform background. Each object has diferent scale, color, and position, and the maximum number of objects in an image is 5.

CLEVR (Johnson et al., 2017) CLEVR dataset contains 6 unique objects with varying scale, color, and position on a uniform background. Although released for visual reasoning tasks, it is commonly used in object discovery. We reported results in 2 versions of CLEVR: CLEVR6 and CLEVR where the maximum numbers of objects in an image are 6 and 10, respectively. CLEVR6 contains around 35,000 and CLEVR contains around 100,000 images.

## .2 Training Details

We adopt the training setup of Monnier et al. (2020) for clustering and Monnier et al. (2021) for multi-object semantic discovery as our baseline. Hyperparameters are provided in Tables 10 and 11. For Table 8, we report the mean and standard error of 3 runs. Due to its computational complexity, we adopt the training schedule reported for CLEVR6 in Monnier et al. (2021) to CLEVR for DTI-Sprites (italic in Table 8). To be comparable with the literature (Karazija et al., 2021), we reported the mean and standard deviation of 3 runs for Table 9. The results for DTI-Sprites and our variation are reported over the whole dataset.

Table 10: Training setup and hyperparameters for clustering.

<table><tr><td>Dataset</td><td>MNIST</td><td>ColoredMNIST</td><td>FashionMNIST</td><td>AffNIST</td><td>USPS</td><td>FRGC</td><td>SVHN</td><td>GTSRB-8</td></tr><tr><td colspan="9">Model &amp; Data</td></tr><tr><td># sprites</td><td>10</td><td>10</td><td>10</td><td>10</td><td>10</td><td>20</td><td>10</td><td>8</td></tr><tr><td>sprite tr.</td><td colspan="8">id, aff, mor, tps id, color, aff, tps id, color, aff, tps id, aff, mor, tps id, color, aff, tps id, color, aff, tps id, color, proj id, color, proj</td></tr><tr><td>sprite tr. curr.</td><td>10, 30, 40</td><td>10, 30, 60</td><td>10, 30, 50</td><td>10, 40, 50</td><td>120, 240, 400</td><td>100, 400, 800</td><td>16, 144</td><td>160, 1440</td></tr><tr><td colspan="9">Training</td></tr><tr><td>batch size</td><td>128</td><td>128</td><td>128</td><td>128</td><td>128</td><td>128</td><td>128</td><td>128</td></tr><tr><td>learning rate</td><td>1e-3</td><td>1e-3</td><td>1e-3</td><td>1e-4</td><td>1e-3</td><td>1e-3</td><td>1e-3</td><td>1e-3</td></tr><tr><td>weight decay</td><td>1e-6</td><td>1e-6</td><td>1e-6</td><td>1e-6</td><td>1e-6</td><td>1e-6</td><td>1e-6</td><td>1e-6</td></tr><tr><td>lr. step</td><td>70</td><td>90</td><td>70</td><td>74</td><td>500</td><td>1300</td><td>240</td><td>2400</td></tr><tr><td># epochs</td><td>80</td><td>100</td><td>80</td><td>90</td><td>640</td><td>1400</td><td>264</td><td>2640</td></tr><tr><td> $\lambda_{freq}$ </td><td>0.01</td><td>0.1</td><td>0</td><td>0.01</td><td>0</td><td>0.01</td><td>0.01</td><td>0.1</td></tr><tr><td> $\lambda_{bin}$ </td><td>0</td><td>0.001</td><td>0</td><td>0</td><td>0.01</td><td>0</td><td>0.001</td><td>0</td></tr></table>

Table 11: Training setup and hyperparameters for multi-object decomposition.

<table><tr><td>Dataset</td><td>Tetrominoes</td><td>Multi-dSprites</td><td>CLEVR6</td><td>CLEVR</td></tr><tr><td colspan="5">Model &amp; Data</td></tr><tr><td># sprites</td><td>19</td><td>3</td><td>6</td><td>6</td></tr><tr><td># bkg</td><td>0</td><td>1</td><td>1</td><td>1</td></tr><tr><td># objects</td><td>3</td><td>5</td><td>6</td><td>10</td></tr><tr><td># channels</td><td>3</td><td>3</td><td>3</td><td>3</td></tr><tr><td>frg., bkg., mask curr.</td><td>600, 0, 1</td><td>0, 0, 20</td><td>0, 0, 80</td><td>0, 0, 80</td></tr><tr><td>sprite/layer init.</td><td>cons, cons, gauss.</td><td>cons, cons, gauss.</td><td>cons, mean, gauss.</td><td>cons, mean, gauss.</td></tr><tr><td>init. values</td><td>0.9, 0.9, 0.</td><td>0.9, 0.5, 0.</td><td>0.9, 0., 0.</td><td>0.9, 0., 0.</td></tr><tr><td>gauss. std.</td><td>5</td><td>7</td><td>10</td><td>10</td></tr><tr><td>sprite tr.</td><td>id</td><td>id, scale+rot.</td><td>id, proj.</td><td>id, proj.</td></tr><tr><td>bkg. tr.</td><td>-</td><td>color</td><td>color</td><td>color</td></tr><tr><td>layer tr.</td><td>color, scale+affine</td><td>color, scale+affine</td><td>color, scale+affine</td><td>color, scale+affine</td></tr><tr><td>sprite tr. curr.</td><td>-</td><td>40</td><td>300</td><td>300</td></tr><tr><td>sprite size</td><td>24, 24</td><td>28, 28</td><td>40, 40</td><td>40, 40</td></tr><tr><td>image size</td><td>35, 35</td><td>35, 35</td><td>128, 128</td><td>128, 128</td></tr><tr><td>occlusion</td><td>-</td><td>√</td><td>√</td><td>√</td></tr><tr><td colspan="5">Training</td></tr><tr><td>avg. pool</td><td>1, 1</td><td>1, 1</td><td>1, 1</td><td>1, 1</td></tr><tr><td>batch size</td><td>32</td><td>32</td><td>32</td><td>32</td></tr><tr><td>learning rate</td><td>1e-4</td><td>1e-4</td><td>1e-4</td><td>1e-4</td></tr><tr><td>lr. step</td><td>1000, 1200</td><td>500, 1000</td><td>500,800</td><td>500, 800</td></tr><tr><td># epochs</td><td>1220</td><td>1020</td><td>900</td><td>900</td></tr><tr><td> $\lambda_{freq}$ </td><td>1e-3</td><td>0</td><td>0</td><td>0</td></tr><tr><td> $\lambda_{bin}$ </td><td>1e-4</td><td>0</td><td>0</td><td>0</td></tr><tr><td> $\lambda_{empty}$ </td><td>-</td><td>1e-4</td><td>1e-3</td><td>1e-2</td></tr></table>

## .2.1 Transformation Module

We follow the transformation setup and order in Table 12 according to Monnier et al. (2020; 2021). Table 12 demonstrates three levels of transformations, applied to the sprites, the background and the layers.

Table 12: Transformation setups of datasets. Transformations are selected and ordered depending on the characteristics of each dataset. Transformations for background and layers are highlighted.

<table><tr><td>Dataset</td><td>id.</td><td>color</td><td>affine</td><td>morpho.</td><td>tps</td><td>proj.</td><td>scale+rot.</td><td>scale+affine</td></tr><tr><td>MNIST</td><td>1</td><td></td><td>2</td><td>3</td><td>4</td><td></td><td></td><td></td></tr><tr><td>ColoredMNIST</td><td>1</td><td>2</td><td>3</td><td></td><td>4</td><td></td><td></td><td></td></tr><tr><td>FashionMNIST</td><td>1</td><td>2</td><td>3</td><td></td><td>4</td><td></td><td></td><td></td></tr><tr><td>affNIST</td><td>1</td><td></td><td>2</td><td>3</td><td>4</td><td></td><td></td><td></td></tr><tr><td>USPS</td><td>1</td><td>2</td><td>3</td><td></td><td>4</td><td></td><td></td><td></td></tr><tr><td>FRGC</td><td>1</td><td>2</td><td>3</td><td></td><td>4</td><td></td><td></td><td></td></tr><tr><td>SVHN</td><td>1</td><td>2</td><td></td><td></td><td></td><td>3</td><td></td><td></td></tr><tr><td>GTSRB-8</td><td>1</td><td>2</td><td></td><td></td><td></td><td>3</td><td></td><td></td></tr><tr><td>Tetrominoes</td><td>1/1</td><td>1</td><td></td><td></td><td></td><td></td><td></td><td>2</td></tr><tr><td>Multi-dSprites</td><td>1</td><td>1/1</td><td></td><td></td><td></td><td></td><td>2</td><td>2</td></tr><tr><td>CLEVR(6)</td><td>1</td><td>1/1</td><td></td><td></td><td></td><td></td><td></td><td>2</td></tr></table>

## .2.2 Analysis on Regularization Hyperparameter Tuning

The weights $\lambda _ { \mathrm { f r e q } }$ and $\lambda _ { \mathrm { b i n } }$ were searched over the range {0, 0.01, 0.1, 1.0} and {0, 0.001, 0.01, 0.1} in clustering, respectively. For the multi-layer setup, we conduct a sequential grid search in range $\{ 0 , 0 . 0 0 0 1 , 0 . 0 0 1 , 0 . 0 1 , 0 . 1 \}$ following the order $\lambda _ { \mathrm { e m p t y } } , \lambda _ { \mathrm { f r e q } }$ , and $\lambda _ { \mathrm { b i n } }$ . We provide the sequential grid search results of $\lambda _ { \mathrm { f r e q } }$ and $\lambda _ { \mathrm { b i n } }$ on four characteristically distinct datasets, ColoredMNIST Arjovsky et al. (2019), FRGC Phillips et al. (2005), SVHN Netzer et al. (2011), and GTSRB-8 Stallkamp et al. (2012), to demonstrate the sensibility of our model to these hyperparameters in Table 13. Our results indicate that $\lambda _ { \mathrm { f r e q } }$ is a critical hyperparameter for preventing cluster collapse. We observed that $\lambda _ { \mathrm { b i n } }$ acts as a regularizer that improves the probabilities to be one-hot, but has a lower impact on overall performance compared to $\lambda _ { \mathrm { f r e q } } .$ Although tuning regularization hyperparameters via ground truth labels allows us to establish a performance upper bound for the proposed architecture, we acknowledge that this protocol departs from a strictly unsupervised setting. We identify the development of robust, fully unsupervised model selection criteria as a significant remaining challenge for the field.

Table 13: Efect of $\lambda _ { \mathbf { f r e q } }$ (left) and $\lambda _ { \mathbf { b i n } }$ (right) on four datasets: ColoredMNIST Arjovsky et al. (2019), FRGC Phillips et al. (2005), SVHN Netzer et al. (2011), and GTSRB-8 Stallkamp et al. (2012). ( ) Gumbel softmax, $( \Rightarrow \lambda _ { \mathrm { b i n } } = 0 ( \mathrm { l e f t } ) , \mathcal { L } _ { \mathrm { r e c } } ^ { s } .$

<table><tr><td rowspan="2">Dataset</td><td colspan="4"> $\lambda_{freq}$ </td></tr><tr><td>1.0</td><td>0.1</td><td>0.01</td><td>0</td></tr><tr><td>ColoredMNIST</td><td>82.7±2.2</td><td>95.9±0.1</td><td>86.4±2.6</td><td>53.2±4.3</td></tr><tr><td>FRGC</td><td>-</td><td>36.1±0.8</td><td>44.8±0.8</td><td>39.5±1.3</td></tr><tr><td>SVHN</td><td>-</td><td>32.8±0.7</td><td>35.3±0.4</td><td>33.9±0.5</td></tr><tr><td>GTSRB-8</td><td>53.0±1.15</td><td>53.2±1.2</td><td>50.5±0.7</td><td>50.0±0.2</td></tr></table>

<table><tr><td rowspan="2">Dataset</td><td rowspan="2"> $\lambda_{freq}$ </td><td colspan="3"> $\lambda_{bin}$ </td></tr><tr><td>0.01</td><td>0.001</td><td>0</td></tr><tr><td>ColoredMNIST</td><td>0.1</td><td>56.6±9.3</td><td>96.0±0.1</td><td>95.9±0.1</td></tr><tr><td>FRGC</td><td>0.01</td><td>42.5±1.0</td><td>41.7±0.6</td><td>44.8±0.8</td></tr><tr><td>SVHN</td><td>0.01</td><td>36.1±0.5</td><td>37.6±0.3</td><td>35.3±0.4</td></tr><tr><td>GTSRB-8</td><td>0.1</td><td>49.7±0.3</td><td>50.0±0.8</td><td>53.2±1.2</td></tr></table>