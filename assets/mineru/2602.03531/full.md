# Robust Representation Learning in Masked Autoencoders

Anika Shrivastava, Renu Rameshan, and Samar Agnihotri

Correspondence Email: samar.agnihotri@gmail.com

## Abstract

Masked Autoencoders (MAEs) achieve impressive performance in image classification tasks, yet the internal representations they learn remain less understood. This work started as an attempt to understand the strong downstream classification performance of MAE. In this process we discover that representations learned with the pretraining and fine-tuning, are quite robust— demonstrating a good classification performance in the presence of degradations, such as blur and occlusions. Through layerwise analysis of token embeddings, we show that pretrained MAE progressively constructs its latent space in a class-aware manner across network depth: embeddings from different classes lie in subspaces that become increasingly separable. We further observe that MAE exhibits early and persistent global attention across encoder layers, in contrast to standard Vision Transformers (ViTs). To quantify feature robustness, we introduce two sensitivity indicators: directional alignment between clean and perturbed embeddings, and head-wise retention of active features under degradations. These studies help establish the robust classification performance of MAEs.

## I. INTRODUCTION

Self-supervised learning (SSL) has been shown [1], [2] to be a powerful approach for visual representation learning - achieving state-of-the-art performance in various downstream tasks, without relying on costly human annotations. Within SSL, Masked Image Modeling (MIM) [3]–[6] has proven particularly effective, where the model intentionally masks patches of the input image and then reconstructs the pixels using a sparse set of visible patches, helping models infer global structure of the input from limited context [3], mirroring the masked-token prediction strategy in NLP [7].

Among MIM approaches, the Masked Autoencoder (MAE) [3] is one of the prominent models that brings masked autoencoding to vision—demonstrating strong classification performance. MAE follows an asymmetric encoder–decoder design in which the encoder operates only on the visible patches and lightweight decoder reconstructs the masked content. As the model receives only a small, random subset of patches (often just 25% of the image), the encoder is forced to learn rich hidden representations. Empirically, the authors of MAE attribute its success to the effectiveness of the pretraining strategy: a Vision Transformer (ViT) [8] trained from scratch requires strong regularizations and long training schedules, whereas MAE pretraining achieves higher classification accuracy (about 84.9%) with minimal fine-tuning [3]. While pretraining underlies MAE’s classification performance, a natural question is how the latent space of pretrained model is organized, particularly with respect to class-wise structure across network depth. Do specific geometric structures arise that aid classification?

While prior work has explored numerous architectural variants and training refinements for MAE, including decoder decomposition [9], mixed input strategies [10], and semantic-part learning [11]–considerably less work has been dedicated to understanding its inner mechanisms [12]–[14] (described in Section II-B). Although, these works provide valuable insights into different aspects of MAE, they do not examine how representations behave internally. In particular, we still lack a clear understanding of how MAE’s encoder organizes class information across network depth, whether any class-specific structure emerges in the pretrained model, despite the absence of labels and how such structure evolves across network depth. While supervised models naturally develop class-separable representations, it is unclear whether similar organization arises in MAE. Moreover, the robustness of the representations obtained after fine tuning, to input perturbations remains underexplored. We believe that MAE’s strong classification performance (compared to ViTs) is fundamentally tied to the structure and behavio of the latent representations formed in the pretrained MAE encoder.

Contributions.

Our main contributions are as follows:

1) We show that pretrained MAE model progressively develops class-separable structure, with distinct class clusters emerging in CLS tokens and also in mean-patch and raw patch token representations across network depth.

2) We provide a geometric characterization of this behavior through a subspace-based analysis, demonstrating that classspecific subspaces gradually diverge with increasing network depth.

3) We evaluate the robustness of fine-tuned MAE under Gaussian blur and attention-guided occlusion and demonstrate that classification performance remains stable across a wide range of perturbation levels.

4) We propose two complementary sensitivity indicators–(i) directional invariance, and (ii) head-wise retention of active features–to quantify the sensitivity of latent representations to perturbations.

5) Our results reveal that MAE’s classification performance is closely linked to the robustness of its latent representations. We use robustness in the sense that the representation does not change much even when the input is perturbed with controlled blur and occlusion.

## II. RELATED WORK

## A. Masked image modelling

Masked Image Modeling (MIM) aims to reconstruct missing regions of an image from a corrupted input. Early approaches, such as Stacked Denoising Autoencoders [15] treated MIM primarily as a denoising problem. With the advent of Vision Transformer [8], MIM evolved into a token-prediction paradigm analogous to BERT [7], in which an image is partitioned into patches and processed as a sequence. Models like BEiT [5], MAE [3], SimMIM [6] adopt this formulation. BEiT uses discrete tokens generated by a dVAE tokenizer [16], while MAE and SimMIM demonstrate that reconstruction from raw pixels alone can produce strong representations. Context Autoencoders [17] further extend this idea by combining masked representation prediction with masked patch reconstruction in the encoded feature space. Finally, iBOT [18] jointly learns a tokenizer via selfdistillation over masked patch tokens and class tokens, allowing the model to acquire semantically meaningful representations. A key insight across these works is that vision benefits from higher masking ratios than language: images exhibit strong spatia redundancy, and heavy masking creates a challenging prediction problem that encourages the model to learn global structure rather than rely on local image statistics [3].

MAE encodes only visible patches in the encoder, introducing mask tokens only at the decoder stage. This design reduces the computational overhead of the encoder and makes MAE sufficiently scalable. Despite heavy masking during pretraining, MAE learns representations that transfer effectively to classification tasks [3], motivating a closer examination of its internal representations.

## B. Understanding Masked Autoencoders

Despite MAE’s effectiveness, comparatively limited work has examined the structure, evolution, and robustness of the representations learned within its encoder layers. Cao et al. [12] present a theoretical perspective on MAE, showing that its patch-based attention mechanism is equivalent to a learnable integral kernel transform. Kong et al. [13] analyze MAE through a hierarchical latent variable model, demonstrating how masking ratio and patch size influence the semantic level of the learned representations. Other theoretical efforts [14], [19], [20] relate MAE to contrastive learning. In particular, [14] shows that MAE implicitly aligns mask-induced positive pairs and introduces a uniformity-enhanced loss to address dimensional collapse. Both [19] and [20] present interpretive frameworks that highlight MAE’s invariance to random masking– the work in [19] does so by reformulating masked image modeling as an equivalent Siamese framework, interpreting MAE as a special case of contrastive learning that reveals occlusion-invariant features, while [20] employs a local contrastive framework to analyze both the reconstructive and contrastive aspects of MAE.

Most work on MAE focuses on modifying its architecture or training strategy to improve performance. The work [9] replaces self-attention layers in the decoder with cross-attention to aggregate encoder output into each input token within the decoder layers, while [10] replaces masked tokens with visible tokens from another image to reconstruct two original images from a single mixed input; the work [11] incorporates semantic-part supervision into MAE training; and [4] integrates convolutiona modules into a multi-scale hierarchical design.

While these works address different aspects of MAE and introduce improvements to its architecture or training, they do not explicitly examine the learned representations that ultimately drive its strong classification performance. Moreover, the behaviour of MAE’s representations under input perturbations remains relatively less explored. This gap motivates our work. We show that the pretrained MAE encoder develops a class-discriminative structure, with representations from different classe becoming increasingly separable in the final encoder layers. This structure provides a strong initialization for fine-tuning and leads to more robust and stable representations for classification.

## III. PROPOSED ANALYSIS PIPELINE

This section describes our analysis pipeline for studying the internal representations learned by MAE through pretraining and fine-tuning. We first analyze the layer-wise token embeddings in the pretrained MAE encoder to examine how class-level structure evolves across network depth, with the aim of assessing whether class-relevant organization emerges in the absence of supervision. We then extend this analysis to the fine-tuned model and study the impact of input perturbations on both classification performance and the robustness of latent representations. Representation robustness is characterized using two complementary indicators: directional alignment between clean and perturbed embeddings, and feature-level robustness within individual attention heads.

## A. Layer-wise structural analysis of the pretrained encoder

## 1) Overview of MAE:

Given an input image $\boldsymbol { x } \in \mathbb { R } ^ { H \times W \times 3 }$ , MAE first divides it into N non-overlapping patches of size $n \times n ,$ forming a sequence $\boldsymbol { x _ { p } } = \{ x _ { p } ^ { 1 } , x _ { p } ^ { 2 } , . . . , x _ { p } ^ { N } \}$ , where each $\boldsymbol { x } _ { p } ^ { i } \in \mathbb { R } ^ { n \times n \times 3 }$ . A random subset of $N _ { v }$ visible patches $x _ { v } \subset x _ { p }$ is selected based on the masking ratio (typically 75%). Each patch is flattened and projected into a D-dimensional embedding by the embedding layer $E _ { p } : \mathbf { \bar { \mathbb { R } } } ^ { n \times n \times 3 } \to \mathbb { R } ^ { \bar { D } }$ . A learnable class (CLS) token $x _ { c l s } \in \mathbb { R } ^ { D }$ is prefixed, and positional embeddings are added to preserve spatial information. The resulting sequence is passed through the encoder $f ( . )$ , producing the latent representation $z = f ( [ x _ { c l s } ; E _ { p } ( x _ { v } ) ] + E _ { p o s } )$ , where $\overline { { E _ { p o s } } } : \bar { \mathbb { R } ^ { \left( N _ { v } + 1 \right) \times D } }$ . These encoded tokens are concatenated with a set of learned mask tokens $e _ { M }$ and fed into a lightweight decoder $g ( . )$ , which reconstructs the masked patches to yield $\hat { x } = g ( z , e _ { M } )$ . We use the MAE ViT-Base architecture [3], consisting of 12 transformer encoder layers with an overall embedding dimension of $D = 7 6 8$ Every layer contains 12 self-attention heads with embedding dimension $d _ { h } = 6 4$ . The output from all heads are concatenated to form the full 768-dimensional embedding at each layer. We follow the original MAE setup, using an input resolution of 224 ˆ 224 and a masking ratio 0.75, resulting in $N _ { v } = 4 9$ visible patches.

## 2) Layer-wise embedding extraction:

To study how MAE organizes class information across network depth, we extract token embeddings from each encoder layer $l \in \{ 1 , \ldots , 1 2 \}$ . For an input image x, the layer-wise output of every encoder layer l is:

$$
z ^ {(l)} = \{z _ {0} ^ {(l)}, z _ {1} ^ {(l)}, \ldots , z _ {N _ {v}} ^ {(l)} \} \in \mathbb {R} ^ {(N _ {v} + 1) \times D},
$$

where $z _ { 0 } ^ { ( l ) }$ denotes the CLS embedding and $z _ { i } ^ { ( l ) }$ for $i \geqslant 1$ correspond to the visible patch embeddings. Using this, we examine three types of embeddings and track how they evolve with network depth: CLS embedding $z _ { 0 } ^ { ( l ) } \in \mathbb { R } ^ { \bar { D } }$ , patch token embeddings $\{ z _ { i } ^ { ( l ) } \} _ { i = 1 } ^ { \bar { N } _ { v } }$ and their mean patch embedding $\begin{array} { r } { \dot { z } ^ { ( l ) } = \frac { 1 } { N _ { v } } \sum _ { i = 1 } ^ { N _ { v } } z _ { i } ^ { ( l ) } } \end{array}$ which aggregates information across all visible patches into a single global descriptor of the image and is known to serve as an effective representation in classification tasks [3]. Motivated by this property, we adopt it in our fine-tuned MAE analysis when evaluating robustness under perturbations.

In Section IV-A, we use t-SNE [21] to visualize the evolution of token embeddings as the input propagates through successive self-attention layers. While these visualizations provide qualitative insight into the emergence of class-related structure, they do not yield a geometry-aware characterization of how representations are organized across classes and network depth. We therefore complement this analysis with a subspace-based geometric study.

## 3) Subspace geometry:

To study the class-level geometric structure of MAE representations, we collect the patch embeddings from all image belonging to the same class and arrange them into a matrix. For class ${ \mathcal { C } } ,$ the layer-l representation matrix is defined as $X _ { \mathcal { C } } ^ { ( l ) } \in \mathbb { R } ^ { N _ { c } \times D }$ , where each row is a patch token embedding of dimension D and $N _ { c }$ is the total number of visible patch tokens aggregated across all images of class C.

To extract the dominant geometric structure of these representations, we apply singular value decomposition (SVD) to $X _ { \mathcal { C } } ^ { ( l ) }$ yielding $X _ { \mathcal { C } } ^ { ( l ) } = U _ { \mathcal { C } } ^ { ( l ) } \Sigma _ { \mathcal { C } } ^ { ( l ) } V _ { \mathcal { C } } ^ { ( l ) ^ { \top } }$ . The right singular vectors $V _ { c } ^ { \left( l \right) ^ { \top } }$ define orthogonal directions in feature space along which the embeddings of class $\mathcal { C }$ exhibit the largest variance. Retaining the top-k singular vectors gives the best possible basis for<sub>\` ˘</sub> representation without redundancy yielding a compact, low-dimensional subspace $S _ { \mathcal { C } } ^ { ( l ) } = \operatorname { s p a n } \bigl ( V _ { \mathcal { C } , k } ^ { ( l ) } \bigr )$ for class $\mathcal { C }$ at depth l.

In Section IV-A, we examine how these class-specific subspaces are oriented relative to one another in the feature space using prinicpal angles [22]. By tracking the principal angles between the subspaces across layers, we obtain a layer-wise view of how class-specific embeddings evolve and are separated with increasing depth in the pretrained MAE encoder.

## B. Fine-tuning robustness

## 1) Perturbations.

We analyze the classification performance of a fine-tuned MAE under two types of perturbations, each applied at multiple severity levels:

1) Gaussian blur: Blur severity is increased by using larger kernel sizes and standard deviations, which lead to a monotonic decrease in PSNR and SSIM. This allows blur severity to be ordered by the resulting image degradation, rather than by parameter choice alone.

2) Attention-guided occlusion: Instead of masking random regions, we use attention rollout [23] to estimate the contribution of each patch to the model’s final prediction. Attention rollout recursively multiplies attention matrices across layers to produce a global importance score for each patch. Patches are then ranked according to these rollout scores, and the top $p \%$ most attended patches–corresponding to different masking levels–are masked to obtain the occluded image $x _ { o }$ . This perturbation explicitly targets regions that the MAE attends to most, providing a challenging setting for assessing the model robustness.

2) Representational robustness indicators:

To study how MAE’s latent representations respond to input perturbations, we examine their robustness from two complementary perspectives:

Directional Robustness:

As an initial analysis, we examine how MAE’s latent representations behave under different random masked versions of the same image. For a fixed image, we generate 100 randomly masked images at the same masking ratio. Across these runs, the resulting latent embeddings remain highly aligned in direction. This behavior is consistently observed across multiple images: while both direction and magnitude vary from image-to-image, they remain remarkably consistent across masked variants of the same image. In particular, the magnitudes for a given image remain tightly concentrated $( \mathrm { e . g . }$ , in the range 92.4 to 95.4 for one example) across all its masked variants. These findings suggests that learned latent representations are largely invariant to random masking, consistent with the findings in [19], [20]. Building on this observation, we next examine whether similar robustness extends to perturbations such as blur and occlusion, where the severity of the perturbation is systematically increased.

We use cosine similarity measure for assessing robustness in terms of direction. Formally, let $\bar { z } _ { ( c l e a n ) } \in \mathbb { R } ^ { D }$ denote the mean patch embedding for the clean input, and let $\bar { z } _ { ( p e r t ) } \in \mathbb { R } ^ { D }$ be the corresponding embedding under blur or occlusion. Directional alignment is quantified as:

$$
\cos (\theta) = \frac {\left<   \bar {z} _ {(c l e a n)} , \bar {z} _ {(p e r t)} \right > }{\| \bar {z} _ {(c l e a n)} \| _ {2} \| \bar {z} _ {(p e r t)} \| _ {2}}.\tag{1}
$$

A value closer to 1 indicates that the encoder projects degraded inputs in a direction similar to that of the clean input in latent space, reflecting strong directional consistency.

Feature-level robustness of attention-heads:

While cosine similarity captures directional alignment between clean and perturbed embeddings, it does not capture featurelevel changes within individual attention heads. To address this limitation, we analyze the set of active features (defined below) retained within each attention head under input perturbations. For a given layer l and attention head h, the head-wise output is given by:

$$
O ^ {(l, h)} = A ^ {(l, h)} V ^ {(l, h)}, \quad O ^ {(l, h)} \in \mathbb {R} ^ {N _ {v} \times d _ {h}},\tag{2}
$$

where $A ^ { ( l , h ) } \in \mathbb { R } ^ { N _ { v } \times N _ { v } }$ is the attention matrix and $V ^ { ( l , h ) } \in \mathbb { R } ^ { N _ { v } \times d _ { h } }$ is the value matrix, with $d _ { h } = 6 4$ for ViT-Base. Each row of $V ^ { ( l , h ) }$ corresponds to the value vector of a visible patch token, i.e., a $d _ { h }$ -dimensional representation, where each dimension corresponds to a feature. Thus, in $V ^ { ( l , h ) }$ , each patch is represented by a $6 4 - D$ feature vector, and each column of V represents the activation of a particular feature across all visible tokens. The attention matrix $A ^ { ( l , h ) }$ determines which patches contribute in updating a given patch’s representation. Specifically, the $i ^ { \mathrm { { t h } } }$ row of $A ^ { ( l , h ) }$ is a probability distribution over visible patch tokens, indicating how strongly each patch contributes to the updated representation of patch i. The head output for token i

$$
O _ {i, k} ^ {(l, h)} = \sum_ {j = 1} ^ {N _ {v}} A _ {i, j} ^ {(l, h)} V _ {j, k} ^ {(l, h)}, \qquad k = 1, \ldots , d _ {h}.\tag{3}
$$

The resulting vector $O _ { i , : } ^ { ( l , h ) } \in \mathbb { R } ^ { 1 \times d _ { h } }$ represents the updated $d _ { h }$ -dimensional representation of the $i ^ { \mathrm { { t h } } }$ patch, obtained as a weighted aggregation of features from the attended patches. In this sense, the head output reflects a combination of importance (via attention) and feature presence (via the value matrix). A feature becomes prominent in $O _ { i , : } ^ { ( l , h ) }$ only if it has high magnitude in the value representations of patches that receive high attention. Conversely, if a feature is absent across the attended patches, its magnitude in the updated representation remains low or zero. We refer to the features with high magnitude in $\stackrel { \bullet } { O } _ { i , : } ^ { ( l , h ) }$ as active features. The outputs of all 12 heads act as a fundamental building blocks as they are concatenated at each encoder layer to form a 768-dimensional layer embedding, that is passed to subsequent layers, ultimately yielding the final representation used for classification.

We take the mean over all rows of $O ^ { ( l , h ) }$ , i.e., calculating mean patch token z¯ to obtain a single 64-dimensional vector per attention head for each image. For a given class C, we collect these vectors across all images in the class and, for each image we identify the top-k active features based on magnitude. A feature is considered common within class C if it appears among the top-k features for at least 60% of the images. For example, for a class containing 50 images and $k = 1 0$ , the common features are those dimensions of z¯ that are present among the top-10 active features in at least 30 images. The number of such features gives the common-feature count $C _ { c l e a n } ^ { ( l , h ) }$ which quantifies how many features remain consistently active within layer l, head h across images of class C. We repeat the same procedure for perturbed inputs to obtain $C _ { p e r t } ^ { ( l , h ) }$ . Importantly, we keep these counts as the cardinality of the intersection between the clean head-level features and the features activated unde perturbation. This directly quantifies how reliably each head preserves its clean feature activations. To make the comparison visually clearer, we also compute the mean drop in feature count: $\Delta C ^ { ( l , h ) } = C _ { \mathrm { c l e a n } } ^ { ( l , h ) } - C _ { \mathrm { p e r t } } ^ { ( l , h ) }$ averaged over all layers and heads for a given perturbation level.

![](images/2b7dad9cacf0a3df09d00813e58c0fba80c5025f8b52736ae42411c76fd6451d.jpg)  
(a) CLS tokens.

![](images/a336cac687dacb3aa59272ad13fe9b9d43d0125fb6ff4a0d7962f16698364955.jpg)  
(b) Mean patch tokens.  
Fig. 1: t-SNE visualizations of token embeddings across encoder layers: (a) CLS tokens and (b) mean patch tokens.

In Section IV-C, we present the empirical findings obtained using the similarity indices define above.

## IV. EXPERIMENTS AND RESULTS

In this section, we present both qualitative and quantitative analyses to trace how MAE’s semantic representations evolve across network depth in a pretrained encoder and how they behave under perturbations after fine-tuning. All experiments are conducted on the Imagenet-1K dataset [24], using the MAE ViT-B model [3] with default architectural settings. Wherever a different dataset is used to support or verify a specific result, it is explicitly mentioned at that point.

## A. Layer-wise analysis of representational structure of MAE pre-trained encoder

## 1) Evolution of token embeddings across depth:

Following Section III-A, we consider a fixed set of ten ImageNet-1K classes [24], selected to include both semantically similar categories (e.g., multiple dog breeds) and diverse categories (e.g., balloon, mosque, speedboat), and draw a total of 100 images uniformly at random (10 per class). We then extract the CLS embeddings $z _ { 0 } ^ { ( l ) }$ , raw patch token embeddings $\{ z _ { i } ^ { ( l ) } \} _ { i = 1 } ^ { N _ { v } }$ and their mean patch embeddings $\bar { z } ^ { ( l ) }$ for each layer l, and visualize them using t-SNE [21].

In the early encoder layers, embeddings from different classes occupy largely overlapping regions with no clear class separation. However, around layers $l = 9 - 1 0$ , distinct class clusters begin to emerge, becoming progressively more separable in the deeper layers. The same trend is consistently observed for the CLS token (Fig. 1a), the mean patch token (Fig. 1b), and, perhaps most interestingly for the raw patch tokens as well. The fact that even individual patches exhibit class-level separation, indicates that, through self-attention, patch representations become increasingly contextualized, allowing class-discriminative information to be distributed across tokens.

We hypothesize that the high masking ratio used during MAE pretraining contributes to the clustering behaviour. With only a small fraction of patches visible, the encoder cannot rely on local neighborhoods and is forced to attend over far away patches from early layers. To quantify this behavior, we compute mean attention distances [25] across 100 images from ten ImageNet-1K classes, for each attention head across the 12 layers. In standard ViTs, attention expands with network depth: initial layers focus on local patterns, while deeper layers gradually incorporate more global context (Fig. 2 (left)). In contrast, for MAE (even with no masking), we find consistently high mean attention distances of approximately 80–120 pixels across all heads and layers (Fig. 2 (right)), indicating that MAE attends globally from the outset. A similar attention-distance pattern is observed on 100 randomly selected images from the SAM [26] dataset, with values exceeding 110-120 pixels across all heads and layers. This persistent long-range attention offers a mechanistic explanation for the emergence of class structure observed in the t-SNE plots.

## 2) Class-wise subspace geometry across network depth:

To complement the qualitative observations from t-SNE, we adopt a geometric viewpoint to quantify how class-specific structure emerges in the latent space of a pretrained MAE. The core idea is to treat the collection of patch-token embedding from a class as points in $\mathbb { R } ^ { 7 6 8 }$ and study the subspaces these points span.

We use the selected ImageNet-1K classes $\mathcal { C } _ { i } , i \in \left. 1 , 2 , \dots , 1 0 \right.$ for this analysis and randomly sample 50 images per class. As described in Section III-A, at each encoder layer l, visible patch embeddings from class $\mathcal { C } _ { i }$ are stacked and SVD is applied to yield a k-dimensional class-specific subspace by retaining the top-k dominant directions of variance. Restricting the analysis to the leading k directions is more for covenience of analysis. We compute principal angles between every pair $( S _ { \mathcal { C } _ { i } } ^ { ( l ) } , \bar { S } _ { \mathcal { C } _ { j } } ^ { \bar { ( l ) } } )$

![](images/b2c84c6fe951a476b9028c95dbee91381bdd7464bc34645cb28b59800c4b399c.jpg)

![](images/a5bf210a6e2306f3c6235e0fc9fdb8d524ca481de65cffaad83c6e2597e9ac8c.jpg)  
Fig. 2: Each dot shows the mean attention distance across images for one head in left: ViT (adapted from [8]) and right: MAE.

![](images/959aa8ad1ea670cde90d90160077c9847b815c5c7c00b6fb88fa815634b7f0b0.jpg)  
Fig. 3: $l e f t { : }$ Layer-wise distribution of principal angles $\theta _ { 1 }$ (in degrees) between classes across layers. right: Layer-wise evolution of the minimum singular value across classes.

The smallest principal angle $\theta _ { 1 }$ serves as a measure of subspace proximity: small angles indicate stronger alignment $( \mathrm { i . e . , }$ overlapping subspaces), while larger angles reflect greater separation. Figure 3 $( l e f t )$ shows the distribution of $\theta _ { 1 }$ over all class pairs, across layers using box plots. In the early layers, the distributions are tightly concentrated at low angles, indicating that subspaces corresponding to different classes remain closely aligned in feature space. With increasing depth, the distribution of $\theta _ { 1 }$ shifts progressively toward larger values, with both the median and the interquartile range increasing across layers. This indicates that class-specific subspaces systematically rotate away from one another and become increasingly well separated in deeper layers. We further observe that singular values also increase with depth, including the minimum singular value (Figure 3 (right)), suggesting that the encoder progressively selects more significant basis directions that contribute meaningfully to the class-specific structure in deeper layers.

Taken together, we show a clear layer-by-layer emergence of class-specific structure: as the network depth increases, embeddings from different classes gradually diverge and occupy increasingly distinct subspaces in the latent space.

## B. Robustness of fine-tuned MAE under input perturbations

Having established that the pretrained MAE encoder develops class-level structure, we now examine the behavior of these representations after supervised fine-tuning. Our goal is to assess the robustness of model’s classification performance under controlled input degradations. We compute mean classification accuracy over the selected ImageNet-1K classes (50 images per class) using predictions based on the mean patch embedding $\bar { z } ^ { ( l ) }$ . We also evaluate the model on Caltech-256 [27] dataset, achieving a top-1 accuracy of 89.47% on clean inputs.

Table I reports the PSNR, SSIM, and top-1 accuracy (mean ˘ standard deviation (σ) across multiple runs) under Gaussian blur for different blur levels. Although image quality drops sharply—PSNR falling from 28.12 dB to 20.06 dB and SSIM from 0.868 to 0.466—the top-1 accuracy remains comparatively stable, remaining above 80% even for the strongest blur. This empirically indicates that even when local details are heavily degraded, the latent representation still retains the information necessary for correct classification.

TABLE I: Top-1 accuracy (mean ˘ σ), PSNR, and SSIM for varying blur levels, characterized by kernel size and standard deviation on ImageNet-1K dataset.

<table><tr><td>Blur level</td><td>Blur setting (k,s)</td><td>Top-1 accuracy (%)</td><td>PSNR (dB)</td><td>SSIM</td></tr><tr><td>I</td><td>k=5, s=1.0</td><td>89.790 ± 0.23</td><td>28.12</td><td>0.868</td></tr><tr><td>II</td><td>k=5, s=2.0</td><td>88.000 ± 0.20</td><td>25.62</td><td>0.774</td></tr><tr><td>III</td><td>k=5, s=4.0</td><td>87.500 ± 0.48</td><td>25.01</td><td>0.743</td></tr><tr><td>IV</td><td>k=5, s=9.0</td><td>87.400 ± 0.62</td><td>24.85</td><td>0.734</td></tr><tr><td>V</td><td>k=7, s=2.0</td><td>82.600 ± 0.72</td><td>24.69</td><td>0.729</td></tr><tr><td>VI</td><td>k=7, s=4.0</td><td>82.400 ± 0.39</td><td>24.21</td><td>0.667</td></tr><tr><td>VII</td><td>k=7, s=13.5</td><td>82.300 ± 0.34</td><td>23.65</td><td>0.646</td></tr><tr><td>VIII</td><td>k=7, s=15.0</td><td>81.202 ± 0.55</td><td>23.00</td><td>0.643</td></tr><tr><td>IX</td><td>k=11, s=2.0</td><td>80.800 ± 0.46</td><td>21.40</td><td>0.601</td></tr><tr><td>X</td><td>k=11, s=5.0</td><td>80.800 ± 0.77</td><td>20.06</td><td>0.466</td></tr></table>

![](images/3d7fa48bba0cbb131805e38b70dda9f0f940ad9f453e1f2af2d5483083be3d91.jpg)  
Fig. 4: Occlusion level vs Mean Accuracy plot on ImageNet-1K dataset.

To probe robustness under more extensive information loss, we use attention-guided occlusion, with results averaged over multiple runs and exhibiting low variability. Despite this adversarial masking strategy, fine-tuned MAE shows high accuracy even when 50% of the most attended patches are removed, as can be seen from Figure 4. Beyond 60% occlusion, accuracy drops sharply but remains at 60.8% even when 90% of the most attended patches are masked. This behaviour indicates that MAE does not rely solely on the most attended patches for classification. Instead, the model is able to form a semantic representation from whichever subset of patches remains visible. This observation is consistent with our pretraining analysis, where raw patch tokens exhibited clear class-level separation, showing that patch embeddings encode class-relevant information.

We further compute results on the ImageNet-C [28] dataset which includes 15 degradation types spanning noise, blur, weather, and digital artifacts, each evaluated at five severity levels to assess MAE’s robustness under a broader set of degradations. MAE maintains stable classification performance (above 75%) for many degradations, including all weather-based variants, defocus blur, motion blur, contrast, pixelation and JPEG compression, with a gradual drop as severity increases. Noise-based degradations lead to a steeper yet progressive decline in accuracy. Overall, these results indicate that MAE exhibits strong robustness to a wide range of algorithmically generated degradations. In contrast, MAE struggles on datasets involving larger distribution shifts, including ImageNet-R [29] which alters texture and local image statistics through artistic renditions, and ImageNet-A [30] which consists of adversarial natural images drawn from a shifted input distribution.

## C. Robustness of latent representations

We now report the empirical observations based on the robustness indicators defined in Section III-B2. We first examine how the direction of latent embeddings changes under perturbations, by taking mean cosine similarity over the selected ImageNet-1K classes $\mathcal { C } _ { i }$ to measure directional alignment of $\bar { z } _ { p e r t }$ with respect to $\bar { z } _ { c l e a n }$ . Despite increasing degradation levels, the similarity remains high under moderate blur and occlusion. Table II(a) shows cosine similarity stays above 0.85 for all blur levels up to $( k = 7 , \sigma = 1 5 . 0 )$ ; at the strongest blur setting $( k = 1 1 , \sigma = 5 . 0 )$ , it drops only to 0.79, consistent with the modest decrease in accuracy (Table I). A similar pattern appears under occlusion (Table II(b)), where similarity remains relatively high upto 60% occlusion and then declines gradually, consistent with the drop observed in accuracy curve (Figure 4).

TABLE II: Mean cosine similarity between clean and perturbed embeddings for (a) Gaussian blur and (b) attention-guided occlusion on ImageNet-1K.  
(a) Gaussian blur

<table><tr><td>Level</td><td>I</td><td>II</td><td>III</td><td>IV</td><td>V</td><td>VI</td><td>VII</td><td>VIII</td><td>IX</td><td>X</td></tr><tr><td>Mean sim.</td><td>0.966</td><td>0.911</td><td>0.908</td><td>0.907</td><td>0.884</td><td>0.866</td><td>0.859</td><td>0.852</td><td>0.793</td><td>0.790</td></tr></table>

(b) Attention-guided occlusion

<table><tr><td>Occlusion (%)</td><td>0</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td></tr><tr><td>Mean sim.</td><td>1.000</td><td>0.954</td><td>0.856</td><td>0.855</td><td>0.855</td><td>0.852</td><td>0.836</td><td>0.805</td><td>0.790</td><td>0.665</td></tr></table>

![](images/5ef5836493bb274714d89cfe6e95a82b23c6d13d1c54240a1863465eeb4d3839.jpg)  
(a) Blur

![](images/05ae1db208f2b8c0fa50f7e3275da7c9a337f11ec22a6520277cb740f86ae2dd.jpg)  
(b) Occlusion  
Fig. 5: Average drop from $C _ { \mathrm { c l e a n } } ^ { ( l , h ) }$ for (a) Gaussian blur and (b) attention-guided occlusion perturbations.

These results show that under moderate perturbations, MAE continues to project degraded inputs to latent embeddings whose directions remain closely aligned with their clean counterparts. However, as severity increases, this alignment progressively drops from above 0.85 to 0.790 for blur and 0.665 for occlusion at highest perturbation setting. Experimentally, we also observe that misclassified images exhibit a much lower cosine similarity with their clean embeddings as compared to the correctly classified ones: For highest blur level, correctly classified cosine is « 0.915 whereas misclassified cosine « 0.289; a similar trend is observed for occlusion as well. These observations suggest that directional alignment in the latent space is closely linked to the classification accuracy of MAE, supporting loss of robustness at higher perturbations, observed in Section IV-B. In addition to directional robustness, we also find that across most blur and occlusion levels, the absolute difference between the norm of the perturbed embedding and that of the clean embedding remains relatively low (e.g., the absolute difference « 1.23–5.56, for one example), indicating the magnitudes of embeddings remain tightly concentrated under perturbations as well.

For examining robustness at a finer scale, we analyze whether features that are consistently active for clean inputs within each attention head continue to remain active when the input is perturbed. For each layer–head pair, we compare the common-feature counts obtained from clean input with those under blur or occlusion. These common features are identified based on their consistent activation across clean images of a class. By checking whether the same features remain active after perturbation, we directly assess how much of the head’s original feature activations is preserved when the input is degraded. For clean inputs, shallow layers exhibit higher common-feature counts than deeper layers, and this behavior is largely preserved unde perturbations as well.

Under blur, feature retention remains largely robust across layers and heads. For most heads, $C _ { p e r t } ^ { ( l , h ) }$ stays close to the clean baseline, with noticeable drops appearing only under the strongest blur. This indicates that moderate blur does not meaningfully disrupt the consistent feature activation structure observed for clean inputs. In contrast, occlusion produces a much sharper effect. Up to 50–60% masking, a large portion of clean features remain active. Beyond this point however, feature retention drops rapidly, especially in deeper layers, where several heads lose most of their common active features. This collapse mirrors the steep accuracy drop observed in Figure 4.

We visualize this behavior by computing the mean drop in feature count averaged over all layers and heads (Figure 5). For moderate blur, ∆C fluctuates across relatively low values, rising sharply at the strongest blur setting (Figure 5a) consistent with the point at which accuracy begins to decline. For occlusion, ∆C increases steadily with masking ratio (Figure 5b), with a pronounced rise beyond 60% occlusion, coinciding with the point at which classification accuracy also begun to collapse.

Overall, the directional and feature-level analyses show consistent trends under increasing severity. Moderate perturbations preserve both embedding direction and head-wise common-feature counts, while extreme perturbations disrupt both, coinciding with the breakdown in classification accuracy.

## V. CONCLUSION

We present a systematic analysis of the representations learned by Masked Autoencoders across both pretraining and finetuning. Our analysis shows that pretrained MAE exhibits a clear layer-wise emergence of class-aware structure despite the absence of labels. Visualization of token embeddings shows that class separation strengthens with network depth, and notably, individual patch tokens also become class-discriminative. Subspace analysis reinforces this observation as embeddings from different classes increasingly diverge and occupy distinct subspaces as the network depth increases. The minimum singular value for each class also increases, indicating that even the weakest directions remain informative. We also note that MAE exhibits global attention right from the outset. After fine-tuning, we show that MAE maintains strong classification performance under Gaussian blur and attention-guided occlusion across a wide range of perturbation levels, with a gradual decline at highe severities. This robust behavior is further supported by our results on ImageNet-C dataset. To better understand this robustness, we examine representation robustness from two complementary viewpoints. Under moderate perturbations, directional alignment analysis shows that perturbed embeddings remain closely aligned with their clean counterparts, and head-wise feature-retention analysis reveals that a substantial portion of active features are retained for perturbed inputs. At extreme perturbation levels, both sensitivity indicators degrade sharply, coinciding with a decline in classification accuracy, suggesting a link between the two.

## REFERENCES

[1] Baevski A., Hsu W. N., Xu Q., Babu A., Gu J. and Auli M.: Data2vec: A general framework for self-supervised learning in speech, vision and language. In: Proc. of ICML pp. 1298-1312. PMLR, (2022).

[2] Tan M., Pang R. and V. Le Q.: Efficientdet: Scalable and efficient object detection. In: Proc. of CVPR, pp. 10781–10790, (2020).

[3] He K., Chen X., Xie S., Li Y., Dollar P. and Girshick R.: Masked autoencoders are scalable vision learners. In: arXiv preprint:2111.06377, (2021).´

[4] Gao P., Ma T., Li H., Lin Z., Dai J. and Y. Q.: Convmae: Masked convolution meets masked autoencoders. In: arXiv preprint: 2205.03892, (2022).

[5] Bao H., Dong L., Piao S. and Wei F.: BEit: BERT pre-training of image transformers. In: Proc. of ICLR, (2022).

[6] Xie Z., Zhang Z., Cao Y., Lin Y., Bao J., Yao Z., Dai Q. and Hu H.: Simmim: A simple framework for masked image modeling. In: Proc. of CVPR, (2022).

[7] Devlin, J., Chang, M.W., Lee, K. and Toutanova, K.: BERT: Pre-training of deep bidirectional transformers for language understanding. In: Proc. of NAACL, vol. 1, pp. 4171–4186, (2019).

[8] Dosovitskiy, A. et. al. : An image is worth 16x16 words: Transformers for image recognition at scale. In: arXiv preprint:2010.11929, (2020).

[9] Fu L., Lian L., Wang R., Shi B., Wang X., Yala A., Darrell T., Efros A. A. and Goldberg, K.: Rethinking patch dependence for masked autoencoders. In: arXiv preprint:2401.14391, (2024).

[10] Liu J., Huang X., Liu Y. and Li H.: Mixmim: Mixed and masked image modeling for efficient visual representation learning. In: arXiv preprint:2205.13137, (2022).

[11] Li G., Zheng H., Liu D., Wang C., Su B. and Zheng C.: Semmae: Semantic-guided masking for learning masked autoencoders. In: Proc. of NeuRIPS, 35:14290–14302, (2022).

[12] Cao S., Xu P. and Clifton D. A.: How to understand masked autoencoders. In: arXiv preprint:2202.03670, (2022).

[13] Kong L., Ma M. Q., Chen G., Xing E. P., Chi Y., Morency L. P. and Zhang K.: Understanding masked autoencoders via hierarchical latent variable models. In: Proc. of CVPR, pp. 7918-7928, (2023).

[14] Zhang Q., Wang Y. and Wang Y.: How mask matters: Towards theoretical understandings of masked autoencoders. In: Proc. of NeuRIPS, 35:27127–27139, (2022).

[15] Vincent P., Larochelle H., Lajoie I., Bengio Y., Manzagol P. and Bottou L.: Stacked denoising autoencoders: Learning useful representations in a deep network with a local denoising criterion. Journal of Machine Learning Research, vol. 11, no. 12, (2010).

[16] Ramesh A., Pavlov M., Goh G., Gray S., Voss C., Radford A., Chen M. and Sutskever I.: Zero-shot text-to-image generation. In: Proc. of ICML, pages 8821–8831. PMLR, (2021).

[17] Chen X., Ding M., Wang X., Xin Y., Mo S., Wang Y., Han S., Luo P., Zeng G. and Wang, J.: Context autoencoder for self-supervised representation learning. IJCV, vol. 132(1), pp. 208-223, (2024).

[18] Zhou J., Wei C., Wang H., Shen W., Xie C., Yuille A. and Kong T.: iBot: Image BERT pre-training with online tokenizer. arXiv preprint:2111.07832, (2021).

[19] Kong X. and Zhang X.: Understanding masked image modeling via learning occlusion invariant feature. In: Proc. of CVPR, pp. 6241–6251, (2023).

[20] Yue X., Bai L., Wei M., Pang J., Liu X., Zhou L. and Ouyang W.: Understanding masked autoencoders from a local contrastive perspective. In: arXiv preprint:2310.01994, (2023).

[21] Maaten L. van der. and Hinton G.: Visualizing data using t-sne. Journal of Machine Learning Research, vol. 9, no. 11, pp. 2579–2605, (2008).

[22] Zhu P. and Knyazev A. V.: Angles between subspaces and their tangents. In: arXiv preprint:1209.0523, (2012).

[23] Abnar S. and Zuidema W.: Quantifying attention flow in transformers. In: arXiv preprint:2005.00928, (2020).

[24] Deng J., Dong W., Socher R., Li L.-J., Li K. and Fei-Fei L.: Imagenet: A large-scale hierarchical image database. In: Proc. of CVPR, Miami, FL, USA, March (2009).

[25] Heo B., Yun S., Han D., Chun S., Choe J. and Oh S. J.: Rethinking spatial dimensions of vision transformers. In: Proc. of ICCV, pp. 11936–11945 (2021).

[26] Kirillov A., Mintun E., Ravi N., Mao H., Rolland C., Gustafson L., Xiao T., Whitehead S., Berg A. C., Lo W., Dollar P. and Girshick R.: Segment´ anything dataset, arXiv:2304.02643, (2023).

[27] Griffin G., Holub A. and Perona P.: Caltech 256 dataset, https://data.caltech.edu/records/nyy15-4j048, (2022) (last accessed: 30 April 2025).

[29] Hendrycks D., Basart S., Mu N., Kadavath S., Wang F., Dorundo E., et. al.: The many faces of robustness: A critical analysis of out-of-distribution generalization. In: Proc. of the IEEE/CVF ICCV, pp. 8340-8349, (2021).

[28] Hendrycks D. and Dietterich T.: Benchmarking neural network robustness to common corruptions and perturbations. arXiv preprint:1903.12261, (2019)

[30] Hendrycks D., Zhao K., Basart S., Steinhardt J. and Song D.: Natural adversarial examples. In: Proc. of CVPR, pp. 15262-15271, (2021).