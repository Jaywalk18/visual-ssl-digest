# Reevaluating the Intra-Modal Misalignment Hypothesis in CLIP

Jonas Herzog

Zhejiang University

jherzog@zju.edu.cn

Yue Wang

Zhejiang University

wangyue@iipc.zju.edu.cn

# Abstract

Recent research suggested that the embeddings produced by CLIP-like contrastive language-image training are suboptimal for image-only tasks. The main theory is that the inter-modal (language-image) alignment loss ignores intramodal (image-image) alignment, leading to poorly calibrated distances between images. In this study, we question this intra-modal misalignment hypothesis. We reexamine its foundational theoretical argument, the indicators used to support it, and the performance metrics affected. For the theoretical argument, we demonstrate that there are no such supposed degrees of freedom for image embedding distances. For the empirical measures, our findings reveal they yield similar results for language-image trained models (CLIP, SigLIP) and image-image trained models (DINO, SigLIP2). This indicates the observed phenomena do not stem from a misalignment specific to the former. Experiments on the commonly studied intra-modal tasks retrieval and few-shot classification confirm that addressing task ambiguity, not supposed misalignment, is key for best results. Project page: https://vision-kek.github.io/Is-CLIP-Really-Misaligned/.

# 1. Introduction

The success of contrastive language-image pretraining exemplified by CLIP [33] has significantly shaped the contemporary era of multimodal, foundational models. The most fundamental and still ubiquitous use-case is to embed images and texts in a shared space, and then compare them via cosine similarity. Following this simple approach, many long-standing computer vision problems from classification to detection [23, 24] to segmentation [19, 49] became addressable in the popular open-vocabulary zero-shot setting.

Nevertheless, recent works have raised concerns that these embeddings may not be ideal for image-only tasks. The so-called intra-modal misalignment hypothesis argues that contrastive language–image training focuses solely on inter-modal (image–text) alignment while neglecting intramodal (image–image) alignment [2, 26, 40, 44]. As a result,

![](images/52103f956d2713ef380a2f9d2a93c0a6527209392dd1a261bd47cf414d39c2cc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["a photo of a dog"] --> B["d≠"]
    C["a photo of a cat"] --> D["d="]
    B --> E["&quot;cat"]
    D --> F["&quot;cat"]
```
</details>

Figure 1. Previous work illustrated an intra-modal misalignment in CLIP space by showing there are cat images closer to a dog (d̸=) than to another cat (d=). We argue $d _ { \neq } < d _ { = }$ is no sign of misalignment. For a labeled downstream dataset, intra-class variance of open vocabulary models is expected and desired to capture semantics and style beyond the narrow dataset-specific labels. Classifying and retrieving with frozen CLIP image embeddings still works well when similarities are measured along the datasetspecific semantic axes. Here the horizontal axis captures dog/cat.

even though the model produces good image-text similarities, the same might not be true for image-image similarities. This effect is often illustrated as in Fig. 1.

The hypothesis relies on the following pillars: (i) a theoretical argument that misalignment emerges from unconstrained degrees of freedom in the embedding space, and (ii) empirical evidence through proposed misaligment indicators such as cosine similarity histograms, modality-gap magnitudes, and retrieval or few-shot classification performance metrics.

In this work, we critically re-examine these arguments.

For the theoretical model, we demonstrate that the degree of freedom argument breaks when considering a sufficiently large set of embeddings. Image–image similarities can in fact be recovered entirely from image–text similarities. This shows that image–image structure is not arbitrary but a consequence of the learned image-text structure.

For the empirical indicators that were proposed to reveal the misalignment, we experiment with these measures and find they are no reliable indicators of intra-modal alignment quality. For example, we argue intra-class variance as shown in Fig. 1 is not a sign of misalignment. Instead, this is normal behavior because without fine-tuning, open-vocabulary models are expected to capture semantics and styles beyond the closed-vocabulary of a narrow downstream dataset. More crucially, the same indicators would also suggest “misalignment” in non-CLIP vision encoders such as DINO [5], which have never been trained with language supervision — revealing that these signals are artifacts of the metrics rather than of the training objective.

Based on our findings, we propose a simple alternative method that measures similarities in class-relevant axes. We conduct evaluations on retrieval and few-shot classification, which have been previously assumed to be affected by the misalignment, and provide a reinterpretation of the results. Finally we discuss the role of the modality gap [20] which is often cited [25, 26, 44] in the context of intra-modal misalignment. In summary:

• We critically re-examine the intra-modal misalignment hypothesis, which directly impacts all methods that use CLIP or SigLIP to compare image embeddings with each other.   
• We point out the limitations of previous arguments and evidences for such misalignment and provide an alternative explanation for their findings.   
• Consequently, we construct a simple alternative method that confirms that the best performance on the previously studied few-shot classification and retrieval tasks can be achieved without assuming a misalignment in CLIP.

# 2. The Intra-Modal Misalignment Hypothesis

We examine the following belief:

CLIP is not optimized for uni-modal scenarios [44]. This leads to intra-modal misalignment [26] / miscalibration [40], its performance in intramodal tasks is not guaranteed [44]. Only exploiting the image encoder of such models is highly suboptimal for intra-modal tasks like image-toimage retrieval [26].

# 2.1. The Supporting Evidence

We review some evidence in favor of the above intra-modal misalignment hypothesis.

The consequence of having only a cross-modal loss term in CLIP has been theoretically explained [40, 44] by degrees of freedom in the embedding space. This idea has been illustrated similar to Fig. 4a-c.

Further, to measure and expose poorly calibrated similarities, several indicators were proposed. The main tool for this purpose are cosine similarity histograms, where the distribution of pairwise similarities between images are recorded. There are two types of similarities that are examined, by class and by modality, shown in Fig. 2. For the similarity distributions by classes [18, 26], the point is straightforward: Two images of the same class should have higher similarity than two images of different classes. As in Fig. 2a, if the two histograms that capture the similarities of same-class and different-class image pairs have a high overlap, it is hypothesized this indicates poor class separation and hence is a sign of misalignment. On the other hand, measuring the similarity distributions by modality [2, 38, 40] as in Fig. 2b reveals that image embeddings have a much higher similarity to each other than to text embeddings. This is considered problematic especially for few-shot classification in the vision-language model adaptation literature [2, 38, 40, 44] because these work combine inter-modal and intra-modal similarity scores, such that it is questioned if the two differently distributed scores can be treated equally.

![](images/18ef94647055c969be603a3e432f7041f13fa9f849bacbf60a8a9160f9add8ba.jpg)

<details>
<summary>area</summary>

| Cosine Similarity | same class | other class |
| ----------------- | ---------- | ----------- |
| 0.6               | 0          | 0           |
| 0.65              | 100        | 500         |
| 0.7               | 200        | 1500        |
| 0.75              | 250        | 2500        |
| 0.8               | 200        | 2000        |
| 0.85              | 100        | 1000        |
| 0.9               | 0          | 0           |
| 0.95              | 0          | 0           |
</details>

![](images/70129d915f2c3ba19785a3222095f26c8393751578304ceae7062298518ac9a6.jpg)

<details>
<summary>histogram</summary>

| Cosine Similarity Range | text-image Sample Density | image-image Sample Density |
| ----------------------- | ------------------------- | -------------------------- |
| 0.1 - 0.2               | 12                        | 0                          |
| 0.2 - 0.3               | 11                        | 0                          |
| 0.3 - 0.4               | 0                         | 0                          |
| 0.4 - 0.5               | 0                         | 0                          |
| 0.5 - 0.6               | 0                         | 0                          |
| 0.6 - 0.7               | 0                         | 1                          |
| 0.7 - 0.8               | 0                         | 6                          |
| 0.8 - 0.9               | 0                         | 5                          |
| 0.9 - 1.0               | 0                         | 0                          |
</details>

Figure 2. Pairwise cosine similarity distributions. Left: Similarities between same class (blue) and opposite class (orange) image feature pairs. A high overlap ratio between the two colors was previously highlighted as an indicator for an intra-modal misalignment issue in CLIP. Right: Similarity distributions of image-text pairs (purple) versus image-image pairs (green). Because CLIP is only supervised on the former, the divergence has previously prompted concerns about whether the latter reflect true similarities. CLIP ViT-B/16. Dataset as in Tab. 1.

This finding of higher intra-modal similarities is a consequence of the modality gap [20], which describes the effect that image embeddings cluster in a manifold clearly separated from the text embedding cluster. Distances across the cluster are therefore naturally higher on average than within. Some work [2, 26] has also attributed the misalignment to the modality gap, but the existing experiments [20, 26] documented that a shifting or fine-tuning approach to narrow the gap tends to worsen performance.

To mitigate the assumed misalignment, prior works then proposed methods for better performance. The most common idea is to avoid measuring image-image similarities in favor of text-image similarities. For few-shot classification, Tip-X [40] proposed to address uncalibrated imageimage embedding distances in few-shot classification by using image-text similarities as a bridge. Instead of directly comparing image features, it constructs “signatures”

![](images/68c59a5b165f6940739a31dc0aceaa2ab7a27a9247242a279c30fe219afd0ecc.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["I₁"] --> B["Image Encoder"]
    C["I₂"] --> B
    B --> D["Image Encoder"]
    D --> E["Text Encoder"]
    E --> F["Image Encoder"]
    F --> G["Text Encoder"]
    G --> H["Feedback Loop"]
    H --> D
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style D fill:#ccf,stroke:#333
    style E fill:#ccf,stroke:#333
    style F fill:#ccf,stroke:#333
    style G fill:#ccf,stroke:#333
```
</details>

Figure 3. Motivated by the intra-modal misalignment hypothesis, previous work [26] posited it is necessary to convert image-image comparison (left) into image-text comparison (right).

for both test and few-shot images by measuring their affinity with text classifier weights. The final affinity between images is then calculated as the KL-divergence between these inter-modal probability distributions. With a similar motivation, Mistretta et al. [26] proposed to tune a text token that represents the image information, an approach illustrated in Fig. 3 and dubbed Optimization-based Textual Inversion (OTI) inspired by [13]. This proved to be more efficient on image-image retrieval than comparing image embeddings directly. In another work, Barbier et al. [2] found that simply normalizing the cosine similarity distributions as in Fig. 2a to equal mean and variance can bring improvements for few-shot object detection with CLIP-like VLMs. Chao et al. [44] also assumed wrong distances between images for intra-modal tasks, and therefore introduced to represent an image by its cross-modal distances to text generated with the help of large language models.

In summary, a substantial body of prior work has interpreted calibrated similarity statistics, modality-wise clustering, and improved performance from text-based surrogates as evidence of intra-modal misalignment.

# 2.2. The Prior Opposing Indicators that Call for Reassessment

While the preceding section reviewed evidence supporting the intra-modal misalignment hypothesis, there also exist several observations that lower the prior probability that the hypothesis is true before considering the evidence presented in our paper.

A first source of skepticism comes from some influential literature that has successfully employed CLIP and related vision–language models for uni-modal tasks such as image classification [14], few-shot adaptation [3, 42, 47], image generation [34], and video synthesis [11, 48]. Further, [25] report that image-to-image retrieval actually outperforms image-to-text retrieval in downstream VLM adaptation tasks. If CLIP embeddings were indeed severely misaligned within the visual modality, it would be difficult to reconcile this with the strong results reported in these works, particularly those experiments that exclusively rely on the image encoder. A second reason to question the mis-Table 1. Repeating the demonstrative experiment in [26] on the simplistic Dogs vs Cats legacy dataset, where near-perfect results are expected. It was suggested in [26] that poor results with CLIP image-image similarities evidences a misalignment in the imageimage space. This hypothesis gets no evidence when swapping model for the uni-modal DINO, a widely acknowledged state-ofthe-art image embedder. CLIP scores highest, suggesting that the observed low metrics are not caused by a model weakness and hence neither by a misalignment. Instead, the performance gap between text-image (T-I) and image-image (I-I) can be attributed to the ambiguity [21] in the way the task is conveyed to the model: Two images with opposite labels might still share enough other concepts to be justifiably similar.

<table><tr><td></td><td colspan="2">Retrieval</td><td colspan="2">Classification</td></tr><tr><td>Model</td><td>T-I</td><td>I-I</td><td>T-I (0-shot)</td><td>I-I (1 | 16-shot)</td></tr><tr><td>CLIP ViT-B/16</td><td>99.3</td><td>87.1</td><td>99.6</td><td>84.2 | 99.7</td></tr><tr><td>DINOv2 ViT-B/14</td><td>-</td><td>81.8</td><td>-</td><td>76.2 | 97.3</td></tr><tr><td>DINOv3 ViT-L/16</td><td>-</td><td>84.3</td><td>-</td><td>80.2 | 97.8</td></tr></table>

alignment interpretation concerns the meaning of intra-class variance. An alternative explanation for the phenomena illustrated in Fig. 1 has been articulated in [53]: embeddings naturally contain both task-specific and task-irrelevant components. From this perspective, high intra-class variance may reflect the coexistence of other semantic factors rather than represent a flaw in the embedding space. The challenge, therefore, is not to “correct” image–image distances, but to identify task-relevant axes (classifier) given the limited samples [12, 21].

Taken together, Sec. 2.1 and Sec. 2.2 highlight a tension: while many works argue for intra-modal misalignment, several existing results in the broader literature are difficult to reconcile with the misalignment view. This inconsistency calls for a careful reassessment of the hypothesis.

# 2.3. A Preliminary Experiment

As a first step toward reassessing the misalignment hypothesis, we revisit a simple experiment in [26] which particularly inspired this work. In this experiment, poor CLIP retrieval performance on a very easy cats and dogs legacy dataset [9] was attributed to the intra-modal misalignment.

However, we consider it unlikely that CLIP fails on such dataset. A simple way to rule out a model weakness is comparison with uni-modal vision models such as DINO [29, 36]. Tab. 1 demonstrates how image-image metrics with DINO lack even more behind the text-image result, prompting the suggestion that the observed measures are not a weakness of CLIP.

![](images/dd65d270632f9ed6aee41a9e1524e015aa3849badc0f1176e8796f9b07e07fdf.jpg)  
Figure 4. (a)-(c): The previous intra-modal degree-of-freedom argument in [40] illustrates that two image embeddings can be either close together (a) or far apart (b) while having the same text-image distance r, concluding that two image embeddings can lie on any two arbitrary points on the circumference (c), leaving a degree of freedom for image-image miscalibration. Our interpretation (d)-(f): The previous line of argumentation overlooks that each image embedding is bound to more than one text anchor (f). Moreover, the two different configurations in (a) and (b) are not arbitrary, but have a good reason to exist: Images in (a,d) and (b,e) have equal distance r to the “cat” text, but the two images in (a,d) are much more similar to each other than those in (b,e). Displayed distance values are real measurements.

# 3. Methodology

Following the structure established by the previous supporting evidence from Sec. 2.1, our methodology to re-evaluate the intra-modal misalignment hypothesis is threefold:

From the theoretical perspective, Sec. 3.1 re-examines if text-image alignment leaves degrees of freedom for imageimage misalignment.

From the empirical perspective, Sec. 3.2 builds on the insight from the preliminary experiment that we can substitute models to allow for controlled study of indicators.

Lastly, to explain performance gains of previous imageto-text augmentation approaches, Sec. 3.3 proposes a simple alternative method.

# 3.1. Analyzing intra-modal degrees of freedom

We begin with re-examining of the degree of freedom argument in [40], where it was hypothesized that intra-modal misalignment emerges from the degrees of freedom that image embeddings have during contrastive language-image training. The idea how this leaves room for poor calibration is illustrated in Fig. 4a-c.

We argue the shortcoming of this hypothesis and its visual demonstration lies in its reduction to a single text embedding. If we extend their model to a larger number of embeddings, then the conclusion no longer holds.

The general acknowledged assumption is that text-image distances are well calibrated. An argument for the misalignment states that given certain text-image distances, there is still room for miscalibrated image-image distances. We can test this argument, formulating the question:

In the pre-training dataset with $n _ { T }$ texts and $n _ { I }$ images, given fixed pair-wise inter-modal cosine similarities expressed through matrix $S _ { i n t e r } \in$ RnT ×nI , is there any room for miscalibrated intra-modal similarities $S _ { i n t r a } \in \mathbb { R } ^ { n _ { I } \times n _ { I } } ?$

If there were a way to express $S _ { i n t r a }$ as an entangled consequence of $S _ { i n t e r }$ , then we could conclude that there is no degree of freedom. The underlying image embeddings $X _ { I } ~ \in ~ \mathbb { R } ^ { n _ { I } \times d }$ are the unknown. Also the text embeddings $X _ { T } ~ \in ~ \mathbb { R } ^ { n _ { T } \times d }$ are in general not necessary to infer $S _ { i n t r a }$ . We can show that in a d-dimensional space, with only d sampled text anchors represented by row indices $\mathcal { I } \subset \{ 1 , \dotsc , n _ { T } \} , | \mathcal { I } | = d .$ , we can set up a linear system that allows for unique recovery of all image embeddings $X _ { I } \in \mathbb { R } ^ { n _ { I } \times d }$ . We know that:

$$
S _ {i n t e r} [ \mathcal {J} ] = X _ {T} [ \mathcal {J} ] \cdot X _ {I} ^ {\top}. \tag {1}
$$

Solving for $X _ { I }$ ,

$$
X _ {I} = (X _ {T} [ J ]) ^ {- 1} \cdot S _ {\text { inter }} [ J ]. \tag {2}
$$

Sampling at least d text embeddings ensures $X _ { T } [ \mathcal { T } ]$ is invertible, given that every row in $X _ { T }$ is a unique embedding unit vector. Typically $n _ { T } , n _ { I } \gg d ,$ so the number of required anchor points is comparatively small. The logic behind Eq. (2) is illustrated for $d = 2$ in Fig. 4f. From Eq. (2) follows that all image-image similarities are welldefined without further degrees of freedom:

$$
S _ {i n t r a} = X _ {I} X _ {I} ^ {\top}. \tag {3}
$$

The dot product equals cosine similarity because the rows in $X _ { I }$ are unit vectors. Eq. (2) produces unit vectors provided that the rows of $X _ { T }$ are normalized likewise.

While in this section we borrow the text anchor assumption from previous work for illustration, it should be noted that such fixed points do not exist in reality. Therefore, in Appx. A we show an alternative way to infer $S _ { i n t r a }$ without sampled anchor embeddings.

# 3.2. Contrasting with non-CLIP

We seek to put the previous proposed indicators for a misalignment under test.

To isolate effects specific to the lack of an intra-modal training objective, we employ the method of comparing models solely trained with inter-modal objectives (CLIP, SigLIP [46]) with models trained with intra-modal objectives (DINO, SigLIP2 [39]).

As in [26] and Fig. 2a, we measure same- vs. opposite class cosine similarity distributions for paired and unpaired samples. Following [2, 40] and Fig. 2b, we further measure same- vs. opposite modality similarity $( S _ { i n t r a }$ vs $S _ { i n t r a } )$ distributions by substituting CLIP models with models that have additional image supervision as in the DINO objective. Finally, we conduct the same CLIP vs. non-CLIP comparison on performance metrics such as retrieval average precision and few-shot classification accuracy.

This way, if an experimental outcome reproduces for both model types, it cannot be attributed to a missing intramodal term in $\mathrm { C L I P ^ { \prime } s }$ loss function.

# 3.3. A simple alternative for more “classness” in similarity measure

When relying on similarity measures between images, many typical classification and retrieval tasks would benefit from only focusing on the image’s single most dominant concept. Ignoring all visual details and background objects would then remove information that does not correlate with the dataset’s labels.

We attribute previous improvements brought by the modality inversion [1] technique from Fig. 3 to this effect. This technique was used in [26] as a means to overcome intra-modal misalignment by converting an image to a text token. This text token $v ^ { * }$ within the “a photo of a $v ^ { \ast } { } ^ { , , }$ prompt is tuned through backpropagation through the text encoder in order to approximate the image embedding.

We argue this is only efficient because it forces the information contained in the image to collapse to a single wordlike token. The resulting text embedding is like “What single word describes the image best?”, which very often correlates strongly with the to-predict class in the test dataset.

To validate our claim, we propose to replace the entire Modality Inversion procedure through simply projecting the image embeddings on the subspace that explains the most variance of $\mathbf { \ddot { a } }$ photo of $x '$ prompts, where x is a wordlike class name. This is similar to [52, 54]. Irrespective of the downstream dataset, we choose ImageNet class names for x. Given n class names, we extract n text embeddings with dimension d. The first $d / 2$ principal components are selected. All test images are projected onto the subspace spanned by the selected components. We then obtain results by measuring image-image cosine similarities in this subspace. We refer to this method as $P C A ^ {  }$ in the experiments.

# 4. Results

Sec. 4.2 evaluates the cosine similarity histogram indicators under the CLIP/non-CLIP substitution methodology from Sec. 3.2. Sec. 4.3 and Sec. 4.4 present results on few-shot classification and image-to-image retrieval, respectively, including the method from Sec. 3.3.

# 4.1. Datasets and Metrics

Following the line of works [47, 50, 51] on few-shot classification with CLIP, we evaluate on 11 image classification datasets. These datasets encompass a variety of image recognition tasks, such as generic object recognition with ImageNet [8] and Caltech101 [41], fine-grained recognition using OxfordPets [30], StanfordCars [17], Flowers102 [28], Food101 [4], and FGVCAircraft [22], satellite image classification with EuroSAT [15], action classification with UCF101 [37], texture classification with DTD [7], and scene recognition with SUN397 [43]. We measure the common classification accuracy metric.

We use these datasets also for image-to-image retrieval, following the work that hypothesized an image-image misalignment [26]. For retrieval, we additionally evaluate on the more commonly used dataset ROxford and RParis [32]. For retrieval evaluations, following [26], we measure mean average precision (mAP) over retrieval queries.

![](images/eaba34f94ed8273af3123cd229d72f0b514c885ed400a28c68bce77f5ee4619c.jpg)

<details>
<summary>histogram</summary>

| Cosine Similarity Range | same class Sample Density | other class Sample Density |
| ----------------------- | -------------------------- | --------------------------- |
| 0.2 - 0.3               | 0                          | 0                           |
| 0.3 - 0.4               | 0                          | 0                           |
| 0.4 - 0.5               | 1                          | 6                           |
| 0.5 - 0.6               | 2                          | 3                           |
| 0.6 - 0.7               | 3                          | 1                           |
| 0.7 - 0.8               | 4                          | 0                           |
| 0.8 - 0.9               | 3                          | 0                           |
| 0.9 - 1.0               | 1                          | 0                           |
</details>

![](images/ac90db6026c2a23ff6398a15692872f6a709a6b1a3847a433c602bf5e2f8419c.jpg)

<details>
<summary>histogram</summary>

| Cosine Similarity Range | Text-Image Count | Image-Image Count |
| ----------------------- | ---------------- | ----------------- |
| -0.1 to -0.05           | 14               | 0                 |
| -0.05 to 0.0            | 12               | 0                 |
| 0.0 to 0.05             | 8                | 6                 |
| 0.05 to 0.1            | 4                | 4                 |
| 0.1 to 0.15            | 2                | 2                 |
| 0.15 to 0.2            | 1                | 1                 |
| 0.2 to 0.25            | 0                | 0                 |
| 0.25 to 0.3            | 0                | 0                 |
| 0.3 to 0.35            | 0                | 0                 |
| 0.35 to 0.4            | 0                | 0                 |
| 0.4 to 0.45            | 0                | 6                 |
| 0.45 to 0.5            | 0                | 6                 |
| 0.5 to 0.55            | 0                | 4                 |
| 0.55 to 0.6            | 0                | 2                 |
| 0.6 to 0.65            | 0                | 1                 |
</details>

![](images/0e9d891f26098940005327ba0aa56b0b01339110e309fd11a9febbeb55e10bae.jpg)

<details>
<summary>histogram</summary>

| Cosine Similarity Range | Sample Density (SigLIP2) | Sample Density (Blue) |
| ----------------------- | ------------------------ | --------------------- |
| 0.3 - 0.4               | 0                        | 0                     |
| 0.4 - 0.5               | 6                        | 0                     |
| 0.5 - 0.6               | 2                        | 1                     |
| 0.6 - 0.7               | 0                        | 3                     |
| 0.7 - 0.8               | 0                        | 4                     |
| 0.8 - 0.9               | 0                        | 3                     |
| 0.9 - 1.0               | 0                        | 1                     |
</details>

![](images/56ef2dbf1ec5f0afe95f87d6fb2801e0541a126cd8400958d70fc3df98a6090a.jpg)

<details>
<summary>histogram</summary>

| Cosine Similarity Bin | Sample Density (SigLIP2) | Sample Density (Other) |
| --------------------- | ------------------------ | ---------------------- |
| 0.00 - 0.01           | 14                       | 0                      |
| 0.01 - 0.02           | 12                       | 0                      |
| 0.02 - 0.03           | 8                        | 0                      |
| 0.03 - 0.04           | 4                        | 0                      |
| 0.04 - 0.05           | 2                        | 0                      |
| 0.05 - 0.06           | 1                        | 0                      |
| 0.06 - 0.07           | 0                        | 0                      |
| 0.07 - 0.08           | 0                        | 0                      |
| 0.08 - 0.09           | 0                        | 0                      |
| 0.09 - 0.10           | 0                        | 0                      |
| 0.10 - 0.11           | 0                        | 0                      |
| 0.11 - 0.12           | 0                        | 0                      |
| 0.12 - 0.13           | 0                        | 0                      |
| 0.13 - 0.14           | 0                        | 0                      |
| 0.14 - 0.15           | 0                        | 0                      |
| 0.15 - 0.16           | 0                        | 0                      |
| 0.16 - 0.17           | 0                        | 0                      |
| 0.17 - 0.18           | 0                        | 0                      |
| 0.18 - 0.19           | 0                        | 0                      |
| 0.19 - 0.20           | 0                        | 0                      |
| 0.20 - 0.21           | 0                        | 0                      |
| 0.21 - 0.22           | 0                        | 0                      |
| 0.22 - 0.23           | 0                        | 0                      |
| 0.23 - 0.24           | 0                        | 0                      |
| 0.24 - 0.25           | 0                        | 0                      |
| 0.25 - 0.26           | 0                        | 0                      |
| 0.26 - 0.27           | 0                        | 0                      |
| 0.27 - 0.28           | 0                        | 0                      |
| 0.28 - 0.29           | 0                        | 0                      |
| 0.29 - 0.30           | 0                        | 0                      |
| 0.30 - 0.31           | 0                        | 1                      |
| 0.31 - 0.32           | 0                        | 1                      |
| 0.32 - 0.33           | 0                        | 1                      |
| 0.33 - 0.34           | 0                        | 1                      |
| 0.34 - 0.35           | 0                        | 1                      |
| 0.35 - 0.36           | 0                        | 1                      |
| 0.36 - 0.37           | 0                        | 1                      |
| 0.37 - 0.38           | 0                        | 1                      |
| 0.38 - 0.39           | 0                        | 1                      |
| 0.39 - 0.40           | 0                        | 1                      |
| Note: The actual sample density values are not provided in the code | so they are left blank in the CSV data. |                         |
</details>

Figure 5. Cosine similarity histograms by class (left) and by modality (right). The distributions are almost identical for purely text-image trained SigLIP (first row) and SigLIP2 (second row) which includes an image-image self-supervised objective as in the DINO line of work. This indicates intra-class variation (left) and the gap between text and image embeddings (right) are no signs of misalignment brought by pure text-image training, but rather normal behavior. All models are ViT-B. Embeddings are sampled from ImageNet validation set.

# 4.2. Cosine Similarity Distributions

Setup. We analyze the two cosine similarity distribution indicators proposed as evidence for intra-modal misalignment (see Fig. 2 for a recap). To test whether these indicators are specific to models lacking an intra-modal objective, we contrast a model trained with only an inter-modal loss (SigLIP) with a model that includes an additional image-image objective (SigLIP2). If the indicators are caused by a missing intra-modal loss, they should be “fixed” in SigLIP2. Results are shown in Fig. 5. The key insight is that both indicators are indistinguishable between the two models, with details

![](images/19122bde1a78567d7f239f1c3fcdb7acf2bbced7aa4af13e79370d96c7f72b0a.jpg)  
Figure 6. Few-shot classification as in training-free VLM adaptation literature. Approaches that attempt to address the intra-modal misalignment issue through avoiding use of image-image similarities (Tip-X, Coder) [40, 44] do not win against feature selection in APE [53] or a basic Linear Discriminant Analysis (LDA) [42] on image embeddings.

discussed below.

Class distances. Fig. 5 (left) shows that both SigLIP and SigLIP2 produce cosine similarity distributions where there are many image pairs from the same class less similar to each other than some image pairs from a different class, mirroring the introductory cat-cat cat-dog example in Fig. 1. The finding that also SigLIP2 manifests the high overlaps in the histograms further supports our alternative hypothesis that this is not a sign of misalignment, but an expected and desirable property of general-purpose pretrained vision encoders. Moreover, the result suggests that SigLIP2’s improvement over SigLIP is not due to a better class separation that can be visualized through the used histogram.

Modality gap. Fig. 5 (right) leads to the same conclusion. The divergence in similarity distributions between inter-modal (image-text) and intra-modal (image-image) pairs is nearly identical for both models. The fact that SigLIP2’s additional image-image training does not close this gap demonstrates that it is a misalignment introduced by a missing image-image objective. We do not see that this divergence is harmful for intra-modal tasks. In fact, the different range can be explained straightforward as a consequence of the modality gap: Because image and text embeddings cluster in two distinct regions in the embedding space, distances measured within the cluster are naturally smaller than between clusters. We discuss this in context of findings from prior work in Sec. 5.

# 4.3. Comparison on the Few-Shot Classification Task

Setup. We study few-shot classification performance in two complementary settings: (1) the standard VLM few-shot adaptation setting [50, 51] where a text prompt provides a zero-shot classifier that is refined using few-shot labeled images; and (2) an image-only few-shot setting with no text prompts, designed to isolate the intrinsic alignment quality of the image embeddings. The former corresponds to Fig. 6, the latter to Tab. 2.

Few-Shot VLM Adaptation. Following prior work, CLIP ResNet-50 results are reported. Fig. 6 summarizes performance across 11 datasets. Methods such as TIP-X and Coder were explicitly designed to circumvent potential intra-modal misalignment by avoiding image–image similarity. However, the results show that these approaches do not outperform simple image-space discriminative baselines such as APE feature selection or Linear Discriminant Analysis (LDA). 1 The reported $P C A ^ {  }$ results simply use the class mean of the projected few-shot samples as classifier. It performs competitively with Tip-X and Coder, supporting the view that classification in the image space is sufficient, without requiring the avoidance of image–image similarity.

Image-Only Few-Shot Classification. We evaluate few-shot classification without any text prompts, more akin to a classical uni-modal few-shot learning setup [6]. Classification is performed measuring image-image cosinesimilarity to prototypes (class means) or by estimating the classifier as in [42]. The two classifiers are tested with original and $P C A ^ {  }$ projected embeddings. CLIP ViT-B/16 is used.

Across 11 datasets, Tab. 2 evidences how SigLIP (intermodal only) outperforms DINOv2 (image-only) in imageimage similarity based classification. This ranking is consistent across classifiers. While comparing models that have been trained on different data does not allow for conclusions about the impact of the training objective (loss function), the experiment still reveals a notable result relevant to the misalignment hypothesis: pure contrastive language-image training can yield image embeddings that are aligned well enough to outperform strong pure vision encoders. In other words, the results lack any evidence of misalignment.

Table 2. Few-shot classification with no use of text prompts. Accuracy is therefore only based on image-to-image similarities. This allows directly to test the intra-modal alignment quality of models with inter-modal loss only (CLIP, SigLIP), both inter- and intra-modal losses (SigLIP2) and image-only models (DINO). Results on $P C A ^ {  }$ -projected features are in (·) parenthesis. Two best models per classifier are underlined. SigLIP scores higher than DINO , indicating well-aligned image embeddings despite inter-modal training only. All models are ViT-B. Reported values are the average over the 11 datasets as in Fig. 6. 

<table><tr><td>Model</td><td>Classifier</td><td>1-shot</td><td>2-shot</td><td>4-shot</td><td>8-shot</td><td>16-shot</td></tr><tr><td rowspan="2">CLIP</td><td>Prototype</td><td>43.4 (50.7)</td><td>55.3 (62.5)</td><td>63.8 (69.7)</td><td>69.6 (74.7)</td><td>73.5 (77.5)</td></tr><tr><td>LDA</td><td>43.4 (50.7)</td><td>60.0 (62.5)</td><td>69.8 (70.8)</td><td>76.1 (76.5)</td><td>79.5 (79.6)</td></tr><tr><td rowspan="2">SigLIP</td><td>Prototype</td><td>57.3 (62.1)</td><td>68.6 (71.9)</td><td>76.3 (78.5)</td><td>80.5 (81.8)</td><td>82.5 (83.7)</td></tr><tr><td>LDA</td><td>57.3 (62.1)</td><td>71.0 (72.4)</td><td>79.0 (79.4)</td><td>83.3 (83.3)</td><td>85.3 (85.3)</td></tr><tr><td rowspan="2">SigLIP2</td><td>Prototype</td><td>58.0 (63.4)</td><td>69.7 (73.2)</td><td>77.0 (79.4)</td><td>80.8 (82.6)</td><td>83.0 (84.5)</td></tr><tr><td>LDA</td><td>58.0 (63.4)</td><td>73.2 (74.5)</td><td>80.5 (81.0)</td><td>84.5 (84.7)</td><td>86.5 (86.5)</td></tr><tr><td rowspan="2">DINOv2</td><td>Prototype</td><td>59.6</td><td>67.1</td><td>71.8</td><td>76.0</td><td>78.2</td></tr><tr><td>LDA</td><td>59.6</td><td>69.2</td><td>75.3</td><td>80.3</td><td>83.3</td></tr></table>

# 4.4. Comparison on the Retrieval Task

Setup. We compare six models that we classify in intermodal and intra-modal, following our contrasting approach. Inter-modal models are those with no image self-supervised objective: CLIP B/32, L/14 and SigLIP. Intra-modal models are SLIP [27], SigLIP2 and DINO, that all have some selfsupervised loss on the images. CLIP, SigLIP and SLIP were also used in the OTI work [26], SigLIP2 and DINO were added by us in order to cover the intra-modal models.

Results. Tab. 3 reports our image-to-image retrieval results. Our proposed $P C A ^ {  }$ alternative consistently outperforms OTI across all datasets and across all models.

We consider this critical as it proves two key points our paper makes. First, it shows comparing image embeddings works well, such that there is no reason to assume a misalignment and thus no need to invert images to pseudotexts as in OTI and Fig. 3. Second, it reveals no experimental outcome is specific to pure text-image training as in CLIP. Specifically, the original models SigLIP and SigLIP2 perform comparable, with only 1.2 mAP gain on SigLIP2, which is an expected improvement given it is an evolution of SigLIP, but no large gap which would point towards a significant problem in SigLIP with pure text-image training. Also the degree to which the $P C A ^ {  }$ projection helps is comparable, with 5.6 and 5.8 mAP gains, respectively.

We include DINOv3 L/16 as a reference upper limit. Interestingly, it does not always score highest. On Stanford Cars, it scores 8.6 mAP lower than our SigLIP B/16 $P C A ^ {  }$ from the inter-modal category. On the other hand, DINO excels on the more standard retrieval datasets ROxford and RParis. We attribute this to dataset curation: ROxford/RParis are designed for unambiguous retrieval, while the other datasets (to our knowledge only used in [26]) suffer from query ambiguity akin to one-shot classification. Their retrieval mAPs we show here are relevant for the study of the misalignment hypothesis, but for the task ambiguity reason, we caution against viewing them as generally meaningful retrieval benchmarks in future work.

Ablation. To test our explanation why $P C A ^ {  }$ can improve performance metrics, we conduct an experiment on a task where we assume it is not helpful to focus on the dominant concept in the image. Classifying the weather and time of the day requires an image representation that captures more than the main class, as background information matters. We therefore measure retrieval metrics on “weather” and “time of day” classification on the BDD100k [45] driving dataset. Unlike on all datasets in Tab. 3, we find a performance drop by 2.3% mAP on weather and 1.0% mAP on time of day compared to original CLIP (ViT-L/14). The information loss through the projection is harmful here. This confirms that $P C A ^ {  }$ only brings gains if dataset labels align with the images’ main semantic concept (Fig. 1), but does not fix some misalignment in general.

Table 3. Image-to-image retrieval results (mAP) following [26]. Rows with ⟨I, I⟩ or $\langle I ^ {  } , I ^ {  } \rangle$ measure intra-modal similarities. We show the performance gain brought by OTI [26] - an image-to-text conversion attempt motivated by the hypothesis that only ⟨T, I⟩ is aligned - can be significantly surpassed through simply measuring $\langle I ^ {  } , I ^ {  } \rangle$ along the main axes of variance of ImageNet class names. This trend is no different between models without intra-modal and with intra-modal training objectives. 

<table><tr><td></td><td>ViT</td><td>Method</td><td> $\langle \cdot, \cdot \rangle$ </td><td>ROxford</td><td>RParis</td><td>Cars</td><td>Pets</td><td>Flowers</td><td>Aircraft</td><td>DTD</td><td>EuroSAT</td><td>Food101</td><td>SUN397</td><td>Caltech</td><td>UCF101</td><td>ImageNet</td><td>Average</td></tr><tr><td rowspan="6">CLIP</td><td rowspan="3">B/32</td><td>Original</td><td> $\langle I, I \rangle$ </td><td>42.4</td><td>74.0</td><td>24.9</td><td>31.2</td><td>62.5</td><td>14.5</td><td>28.3</td><td>49.3</td><td>33.6</td><td>34.6</td><td>77.7</td><td>46.2</td><td>21.6</td><td>41.6</td></tr><tr><td>OTI</td><td> $\langle T, I \rangle$ </td><td>43.0</td><td>70.3</td><td>28.0</td><td>37.5</td><td>62.6</td><td>14.4</td><td>31.9</td><td>47.2</td><td>34.7</td><td>36.3</td><td>79.9</td><td>48.6</td><td>23.8</td><td>42.9</td></tr><tr><td> $PCA^{\leftarrow}$ </td><td> $\langle I^{\leftarrow}, I^{\leftarrow} \rangle$ </td><td>51.4</td><td>80.9</td><td>34.6</td><td>47.7</td><td>70.3</td><td>16.1</td><td>34.0</td><td>53.8</td><td>43.0</td><td>40.0</td><td>83.3</td><td>53.3</td><td>28.6</td><td>49.0</td></tr><tr><td rowspan="3">L/14</td><td>Original</td><td> $\langle I, I \rangle$ </td><td>57.1</td><td>77.8</td><td>43.8</td><td>47.2</td><td>84.2</td><td>25.8</td><td>33.9</td><td>57.8</td><td>55.0</td><td>39.2</td><td>83.8</td><td>59.5</td><td>33.0</td><td>53.7</td></tr><tr><td>OTI</td><td> $\langle T, I \rangle$ </td><td>62.4</td><td>77.1</td><td>50.5</td><td>56.0</td><td>86.0</td><td>27.1</td><td>37.7</td><td>56.3</td><td>55.9</td><td>43.5</td><td>87.3</td><td>62.8</td><td>38.2</td><td>57.0</td></tr><tr><td> $PCA^{\leftarrow}$ </td><td> $\langle I^{\leftarrow}, I^{\leftarrow} \rangle$ </td><td>64.5</td><td>83.0</td><td>57.2</td><td>62.7</td><td>88.9</td><td>28.7</td><td>39.9</td><td>62.8</td><td>64.4</td><td>46.0</td><td>89.5</td><td>66.8</td><td>42.2</td><td>61.3</td></tr><tr><td rowspan="3">Sig LIP</td><td rowspan="3">B/16</td><td>Original</td><td> $\langle I, I \rangle$ </td><td>50.6</td><td>73.1</td><td>65.7</td><td>56.4</td><td>87.5</td><td>37.9</td><td>39.8</td><td>53.3</td><td>56.3</td><td>42.8</td><td>87.2</td><td>56.9</td><td>35.8</td><td>57.2</td></tr><tr><td>OTI</td><td> $\langle T, I \rangle$ </td><td>55.2</td><td>79.1</td><td>71.8</td><td>64.2</td><td>89.7</td><td>37.6</td><td>43.3</td><td>52.9</td><td>59.0</td><td>43.6</td><td>88.9</td><td>54.9</td><td>38.8</td><td>60.0</td></tr><tr><td> $PCA^{\leftarrow}$ </td><td> $\langle I^{\leftarrow}, I^{\leftarrow} \rangle$ </td><td>57.9</td><td>78.4</td><td>77.2</td><td>68.5</td><td>92.0</td><td>40.9</td><td>44.2</td><td>54.2</td><td>61.8</td><td>46.9</td><td>91.2</td><td>60.3</td><td>43.5</td><td>62.8</td></tr><tr><td rowspan="3">SLIP</td><td rowspan="3">B/16</td><td>Original</td><td> $\langle I, I \rangle$ </td><td>36.5</td><td>79.2</td><td>4.9</td><td>17.8</td><td>65.7</td><td>9.1</td><td>29.7</td><td>53.7</td><td>19.5</td><td>26.1</td><td>65.4</td><td>40.2</td><td>15.4</td><td>35.6</td></tr><tr><td>OTI</td><td> $\langle T, I \rangle$ </td><td>36.4</td><td>79.3</td><td>5.0</td><td>19.3</td><td>65.1</td><td>9.0</td><td>30.5</td><td>50.6</td><td>20.0</td><td>26.4</td><td>67.6</td><td>40.6</td><td>14.8</td><td>35.7</td></tr><tr><td> $PCA^{\leftarrow}$ </td><td> $\langle I^{\leftarrow}, I^{\leftarrow} \rangle$ </td><td>43.6</td><td>83.9</td><td>6.2</td><td>23.1</td><td>72.5</td><td>10.3</td><td>32.2</td><td>53.2</td><td>25.1</td><td>30.7</td><td>70.7</td><td>42.4</td><td>19.1</td><td>39.4</td></tr><tr><td rowspan="2">Sig LIP2</td><td rowspan="2">B/16</td><td>Original</td><td> $\langle I, I \rangle$ </td><td>52.5</td><td>75.6</td><td>70.8</td><td>56.6</td><td>89.3</td><td>40.7</td><td>38.6</td><td>49.3</td><td>59.7</td><td>43.0</td><td>89.2</td><td>59.2</td><td>37.9</td><td>58.6</td></tr><tr><td> $PCA^{\leftarrow}$ </td><td> $\langle I^{\leftarrow}, I^{\leftarrow} \rangle$ </td><td>59.4</td><td>78.6</td><td>80.2</td><td>67.8</td><td>93.2</td><td>46.3</td><td>44.1</td><td>51.1</td><td>65.3</td><td>48.9</td><td>93.0</td><td>63.0</td><td>46.5</td><td>64.4</td></tr><tr><td colspan="3">DINOv3 L/16</td><td> $\langle I, I \rangle$ </td><td>91.4</td><td>93.5</td><td>68.6</td><td>82.3</td><td>99.4</td><td>39.8</td><td>44.9</td><td>59.9</td><td>71.0</td><td>50.1</td><td>90.7</td><td>70.6</td><td>57.6</td><td>70.8</td></tr></table>

# 5. Discussion

Is the modality gap actually a problem? Given that previous papers titled “Mind the gap” [20], “Mitigate the Gap” [10], “Cross the gap” [26], “Bridging the Gap” [2], the gap may appear as a problem. The resulting different distributions of cosine similarities (Fig. 2b, Fig. 5b) further made some [2, 40] belief something is intra-modally misaligned. However, we failed to identify any such misalignment that negatively impacts image-only tasks. Attempts of previous work to narrow the gap could also not bring consistent improvement [16, 20, 26]. Eslami et al. [10] suggest adding an intra-modal separation loss reduces the gap while helpful for downstream performance, but this result is confined to their from-scratch trained modified shared text-vision encoder architecture. Qian et al. [31] theoretically demonstrate that the gap cannot be reduced sufficiently by minimizing the contrastive loss in CLIP. Jiang et al. [16] further prove that exact modality alignment is suboptimal in general for downstream prediction tasks. We therefore see it as reasonable, not problematic, that image and text embeddings lie in two separate manifolds.

What about other tasks? E.g. segmentation, depth estimation, VQA? – We only evaluate on retrieval and fewshot classification because these were the tasks that were previously studied in the context of the intra-modal misalignment hypothesis. To our knowledge, literature on other image-image applications of CLIP has not mentioned the misalignment issue. If supposed misalignment turns out to be a non-issue for the previously studied tasks, we do not

expect other tasks to be impacted, either. As for the $P C A ^ {  }$ method, we do not believe it will be useful for these tasks.

Did we disprove the hypothesis? For the experimental results, strictly speaking, no. Tabs. 1 to 3 and Fig. 5 prove intra-modal misalignment was not the cause of the observed trends, but do not prove it is impossible to exist. Similar to [35], the refutation logic is to show that what previously was believed to be evidence for a hypothesis, can actually not serve as such. For the theoretical argument in Sec. 3 and Appx. A, yes it seeks to refute the belief in the underlying cause. But ultimately, the hypotheses in prior work vary to some extent, so we believe it is more useful and appropriate to consider the evidence for the individual arguments than to speak of holistic proof/disproof.

# 6. Conclusion

In this work, we reevaluate the intra-modal misalignment hypothesis in CLIP-style vision-language models. Through theoretical analysis, we demonstrate that intra-modal similarities are not unconstrained with arbitrary degrees of freedom, but determined by cross-modal similarities. Empirically, we reexamine previously proposed indicators and performance metrics, showing they are no reliable diagnostic of intra-modal misalignment. Comparison with a PCA-style projection method further supports our alternative hypothesis that task ambiguity in the few-shot setting, not misalignment, best explains the results. We hope this can guide future work to leverage the pretrained intra-modal geometry rather than circumvent it.

Acknowledgment This work was supported by the National Nature Science Foundation of China under Grant No. 62522317 and 62373322, and by Zhejiang Provincial Natural Science Foundation of China under Grant No. LD25F030001.

# References

[1] Alberto Baldrati, Lorenzo Agnolucci, Marco Bertini, and A. Bimbo. Zero-shot composed image retrieval with textual inversion. In ICCV, 2023. 5   
[2] Clement Barbier, Baptiste Abeloss, and St ´ ephane Herbin. ´ Bridging the modality gap: Training-free adaptation of vision-language models for remote sensing via visual prototypes. In CVPR Workshops, 2025. 1, 2, 3, 5, 8   
[3] Yassir Bendou, Amine Ouasfi, Vincent Gripon, and Adnane Boukhayma. Proker: A kernel perspective on few-shot adaptation of large vision-language models. In CVPR, 2025. 3   
[4] Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In ECCV, 2014. 5   
[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Herve J ´ egou, ´ Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021. 2   
[6] Wei-Yu Chen, Yen-Cheng Liu, Zsolt Kira, Y. Wang, and Jia-Bin Huang. A closer look at few-shot classification. In ICLR, 2019. 7   
[7] Mircea Cimpoi, Subhransu Maji, Iasonas Kokkinos, Sammy Mohamed, and Andrea Vedaldi. Describing textures in the wild. CVPR, 2014. 5   
[8] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In CVPR, 2009. 5   
[9] Jeremy Elson, John R. Douceur, Jon Howell, and Jared Saul. Asirra: a captcha that exploits interest-aligned manual image categorization. In Conference on Computer and Communications Security, 2007. 3   
[10] Sedigheh Eslami and Gerard de Melo. Mitigate the gap: Investigating approaches for improving cross-modal alignment in clip. In ICLR, 2025. 8   
[11] Patrick Esser, Johnathan Chiu, Parmida Atighehchian, Jonathan Granskog, and Anastasis Germanidis. Structure and content-guided video synthesis with diffusion models. In ICCV, 2023. 3   
[12] Matteo Farina, Massimiliano Mancini, Giovanni Iacca, and Elisa Ricci. Rethinking few-shot adaptation of visionlanguage models in two stages. In CVPR, 2025. 3   
[13] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H. Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In ICLR, 2023. 3   
[14] Robert Geirhos, Priyank Jaini, Austin Stone, Sourabh Medapati, Xi Yi, George Toderici, Abhijit Ogale, and Jonathon Shlens. Towards flexible perception with visual memory. In ICML, 2025. 3   
[15] Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning

benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7), 2019. 5   
[16] Qian Jiang, Changyou Chen, Han Zhao, Liqun Chen, Q. Ping, Son Dinh Tran, Yi Xu, Belinda Zeng, and Trishul M. Chilimbi. Understanding and constructing latent modality structures in multi-modal representation learning. In CVPR, 2023. 8   
[17] Jonathan Krause, Michael Stark, Jia Deng, and Li Fei-Fei. 3d object representations for fine-grained categorization. In CVPR workshops, 2013. 5   
[18] Shuo Li, Fang Liu, Zehua Hao, Xinyi Wang, Lingling Li, Xu Liu, Puhua Chen, and Wenping Ma. Logits deconfusion with clip for few-shot learning. In CVPR, 2025. 2   
[19] Feng Liang, Bichen Wu, Xiaoliang Dai, Kunpeng Li, Yinan Zhao, Hang Zhang, Peizhao Zhang, Peter Vajda, and Diana ´ Marculescu. Open-vocabulary semantic segmentation with mask-adapted clip. In CVPR, 2023. 1   
[20] Weixin Liang, Yuhui Zhang, Yongchan Kwon, Serena Yeung, and James Y. Zou. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. In NeurIPS, 2022. 2, 8   
[21] Zhiqiu Lin, Samuel Yu, Zhiyi Kuang, Deepak Pathak, and Deva Ramana. Multimodality helps unimodality: Crossmodal few-shot learning with multimodal models. In CVPR, 2023. 3   
[22] Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013. 5   
[23] Matthias Minderer, Alexey A. Gritsenko, Austin Stone, Maxim Neumann, Dirk Weissenborn, Alexey Dosovitskiy, Aravindh Mahendran, Anurag Arnab, Mostafa Dehghani, Zhuoran Shen, Xiao Wang, Xiaohua Zhai, Thomas Kipf, and Neil Houlsby. Simple open-vocabulary object detection with vision transformers. In ECCV, 2022. 1   
[24] Matthias Minderer, Alexey A. Gritsenko, and Neil Houlsby. Scaling open-vocabulary object detection. In NeurIPS, 2023. 1   
[25] Yifei Ming and Yixuan Li. Understanding retrievalaugmented task adaptation for vision-language models. In ICML, 2024. 2, 3   
[26] Marco Mistretta, Alberto Baldrati1, Lorenzo Agnolucci, Marco Bertini, and Andrew D. Bagdanov1. Cross the gap: Exposing the intra-modal misalignment in clip via modality inversion. In ICLR, 2025. 1, 2, 3, 5, 7, 8   
[27] Norman Mu, Alexander Kirillov, David A. Wagner, and Saining Xie. Slip: Self-supervision meets language-image pre-training. In ECCV, 2022. 7   
[28] Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In Indian conference on computer vision, graphics & image processing, 2008. 5   
[29] Maxime Oquab, Timothee Darcet, Th ´ eo Moutakanni, ´ Huy Q. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, Mahmoud Assran, Nicolas Ballas, Wojciech Galuba, Russ

Howes, Po-Yao (Bernie) Huang, Shang-Wen Li, Ishan Misra, Michael G. Rabbat, Vasu Sharma, Gabriel Synnaeve, Huijiao Xu, Herve J ´ egou, Julien Mairal, Patrick Labatut, Armand ´ Joulin, and Piotr Bojanowski. Dinov2: Learning robust visual features without supervision. ArXiv arXiv:2304.0719, 2023. 3   
[30] Omkar M Parkhi, Andrea Vedaldi, Andrew Zisserman, and CV Jawahar. Cats and dogs. In CVPR, 2012. 5   
[31] Qi Qian, Yuanhong Xu, and Juhua Hu. Intra-modal proxy learning for zero-shot visual categorization with clip. In NeurIPS, 2023. 8   
[32] Filip Radenovic, Ahmet Iscen, Giorgos Tolias, Yannis ´ Avrithis, and Ondˇrej Chum. Revisiting oxford and paris: Large-scale image retrieval benchmarking. In CVPR, 2018. 5   
[33] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. In ICML, 2021. 1   
[34] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In CVPR, 2023. 3   
[35] Rylan Schaeffer, Brando Miranda, and Oluwasanmi Koyejo. Are emergent abilities of large language models a mirage? In NeurIPS, 2023. 8   
[36] Oriane Simeoni, Huy V. Vo, Maximilian Seitzer, Federico´ Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michael Ramamonjisoa, ¨ Francisco Massa, Daniel Haziza, Luca Wehrstedt, Jianyuan Wang, Timothee Darcet, Th´ eo Moutakanni, Leonel Sentana,´ Claire Roberts, Andrea Vedaldi, Jamie Tolan, John Brandt, Camille Couprie, Julien Mairal, Herve J ´ egou, Patrick La- ´ batut, and Piotr Bojanowski. DINOv3. arXiv preprint arXiv:2508.10104, 2025. 3   
[37] Khurram Soomro, Amir Roshan Zamir, and Mubarak Shah. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012. 5   
[38] Christoph Timmermann, Hyunse Lee, and Woojin Lee. Semobridge: Semantic modality bridge for efficient few-shot adaptation of clip. arXiv preprint arXiv:2509.26036, 2025. 2   
[39] Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim M. Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier H’enaff, Jeremiah Harmsen, Andreas Steiner, and Xiao-Qi Zhai. Siglip 2: Multilingual visionlanguage encoders with improved semantic understanding, localization, and dense features. arXiv-preprint arXiv:2502.14786, 2025. 4   
[40] Vishaal Udandarao, Ankush Gupta, and Samuel Albanie. Sus-x: Training-free name-only transfer of vision-language models. In CVPR, 2023. 1, 2, 4, 5, 6, 8   
[41] Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011. 5

[42] Zhengbo Wang, Jian Liang, Lijun Sheng, Ran He, Zilei Wang, and Tieniu Tan. A hard-to-beat baseline for trainingfree clip-based adaptation. In ICLR, 2024. 3, 6, 7   
[43] Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In CVPR, 2010. 5   
[44] Chao Yi, Lu Ren, De-Chuan Zhan, and Han-Jia Ye. Leveraging cross-modal neighbor representation for improved clip classification. In CVPR, 2024. 1, 2, 3, 6   
[45] Fisher Yu, Haofeng Chen, Xin Wang, Wenqi Xian, Yingying Chen, Fangchen Liu, Vashisht Madhavan, and Trevor Darrell. Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In CVPR, 2018. 7   
[46] Xiaohua Zhai, Basil Mustafa, Alexander Kolesnikov, and Lucas Beyer. Sigmoid loss for language image pre-training. In ICCV, 2023. 4   
[47] Renrui Zhang, Rongyao Fang, Peng Gao, Wei Zhang, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free clip-adapter for better visionlanguage modeling. In ECCV, 2022. 3, 5   
[48] Zhixing Zhang, Bichen Wu, Xiaoyan Wang, Yaqiao Luo, Luxin Zhang, Yinan Zhao, Peter Vajda, Dimitris Metaxas, and Licheng Yu. Avid: Any-length video inpainting with diffusion model. In CVPR, 2024. 3   
[49] Chong Zhou, Chen Change Loy, and Bo Dai. Extract free dense labels from clip. In ECCV, 2022. 1   
[50] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for vision-language models. In IJCV, 2022. 5, 6   
[51] Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In CVPR, 2022. 5, 6   
[52] Beier Zhu, Jiequan Cui, Han Chao Zhang, and Chi Zhang. Project-probe-aggregate: Efficient fine-tuning for group robustness. In CVPR, 2025. 5   
[53] Xiangyang Zhu, Renrui Zhang, Bowei He, A-Long Zhou, Dong Wang, Bingyan Zhao, and Peng Gao. Not all features matter: Enhancing few-shot clip with adaptive prior refinement. In ICCV, 2023. 3, 6   
[54] Xingyu Zhu, Beier Zhu, Shuo Wang, Kesen Zhao, and Han Chao Zhang. Enhancing clip robustness via crossmodality alignment. In NeurIPS, 2025. 5

# Reevaluating the Intra-Modal Misalignment Hypothesis in CLIP

# Supplementary Material

# A. On degrees of freedom

In Sec. 3.1 of the main paper, we presented a way to recover intra-modal similarities from inter-modal similarities. This solution assumed a set of text anchors for simplification. Here we show that even without this assumption, unique recovery of image-image similarities is possible:

Start with given inter-modal similarities

$$
S _ {i n t e r} = X _ {T} X _ {I} ^ {\top}, \tag {4}
$$

where the hidden underlying $X _ { T } , X _ { I } \in \mathbb { R } ^ { N \times d }$ have d-dim row vectors normalized to length 1 and $N \gg d$ .

Decompose the $n \times n$ matrix $S _ { i n t e r }$ via SVD:

$$
S _ {i n t e r} = X _ {T} X _ {I} ^ {\top} = U \Sigma V ^ {\top} \tag {5}
$$

where:

• $\Sigma \in \mathbb { R } ^ { d \times d }$ is a diagonal matrix, we do not further use it.   
$\mathbf { \Psi } \bullet \ U , V \in \mathbb { R } ^ { N \times d }$ are orthogonal s.t. $U ^ { \top } U = I = V ^ { \top } V$

Since the columns of V span the same space as the columns of (full-rank) $X _ { I }$ , we can write:

$$
X _ {I} = V C \tag {6}
$$

for some d × d matrix C. Then:

$$
S _ {i n t r a} = X _ {I} X _ {I} ^ {\top} = V C C ^ {\top} V ^ {\top}. \tag {7}
$$

Let $Q = C C ^ { \top }$ . Because of normalized $X _ { I }$ , we know:

$$
\mathrm{diag} (X _ {I} X _ {I} ^ {\top}) = \mathrm{diag} (V Q V ^ {T}) = \mathbf {1} _ {N}. \tag {8}
$$

This gives a linear system for the entries of $Q \in \mathbb { R } ^ { d \times d }$ Specifically, there are N quadratic forms

$$
v _ {i} ^ {\top} Q v _ {i} = \sum_ {j = 1} ^ {d} \sum_ {k = 1} ^ {d} V _ {i j} V _ {i k} Q _ {j k} = 1 \quad \text { for   } i = 1, \dots , N. \tag {9}
$$

Solve by rearranging in a standard $A x = b$ system,

A : set the scalar product $V _ { i j } V _ { i k }$ as the entry of the ith row and $( j d - d + k )$ )th column in coefficient matrix $A \in$ RN ×d2 . $\mathbb { R } ^ { N \times d ^ { 2 } }$

x : set flatten(Q) as the variable vector $x \in \mathbb { R } ^ { d ^ { 2 } }$ .

b : set $b = \mathbf { 1 } _ { N } \in \mathbb { R } ^ { N }$ .

There are $d ( d \mathrm { + } 1 ) / 2$ unknowns in x since Q is a symmetric $d \times d$ matrix. Since $N \gg d ,$ , the linear system $A x = b$ is overdetermined such that there is at most one solution for x and there are no degrees of freedom. In our recovery case, the solution exists; $b = 1 _ { N }$ is in the column space of A. Solving for x, then reshaping x back to $Q ,$ the intra-modal image-image similarities can be obtained via Eq. (7):

$$
S _ {i n t r a} = V Q V ^ {T}. \tag {10}
$$

We provide a demonstration in PyTorch.

# B. On the projection for more “classness”

In Sec. 3.3 of the main paper, we presented a way to reduce the semantics of an image embedding to its dominant concept by projecting onto axes spanned by class names.

Here we show that despite usage of text embeddings, this method, $P C A ^ {  }$ , still can be considered image-image comparison.

Interpretation - A neat way to illustrate this symmetry is by interpreting the projection of an image embedding $x _ { i } ~ \in ~ \mathbb { R } ^ { d }$ as a sequence of three operations: a rotational change of basis $( Q ^ { \top } )$ into the coordinate system defined by the principal components of class names, followed by a scaling (Λ) that preserves or cancels components, followed by the change back to the original basis $( Q )$ :

$$
x _ {i} ^ {\leftarrow} = Q \Lambda Q ^ {\top} x _ {i}. \tag {11}
$$

The resulting $x _ { i } ^ {  } \in \mathbb { R } ^ { d }$ is the projected image embedding.

The columns of $Q \in \mathbb { R } ^ { d \times d }$ contain the sorted eigenvectors (components) obtained by Eigendecomposition (PCA) of the covariance matrix of ImageNet class name text embeddings. The orthogonality of $Q$ brings the isometric property that ensures $Q$ and $Q ^ { \top }$ themselves preserve angles and distances, and hence also similarities.

The diagonal $\Lambda \in \mathbb { R } ^ { d \times d }$ determines the scaling of each basis vector. If $\Lambda = I ,$ then the projection has no effect $( x _ { i } ^ {  } = x _ { i } )$ . We can instead set $\Lambda = \mathrm { d i a g } ( 1 . . . , 1 , 0 , . . . , 0 )$ to eliminate the components that do not explain much variance of class names. After (optional) re-projection to the original space via $Q _ { \ l }$ , the resulting $\boldsymbol { x } _ { i } ^ {  }$ can be interpreted as the original $x _ { i }$ being preserved in selected directions, while cut off in the other.

Visualizations with UMAP, t-SNE and PCA in Fig. 7 demonstrate that $\boldsymbol { x } _ { i } ^ {  }$ is indeed both globally and locally close to $x _ { i }$ . For visualization purposes, mean adjustment can be optionally performed via $\tilde { x _ { i } } ^ {  } = x _ { i } ^ {  } + \delta _ { \mu }$ ,

$$
\delta_ {\mu} = Q (1 - \Lambda) Q ^ {\top} \mu_ {x}, \tag {12}
$$

i.e. adding back the part of the image embedding mean $\mu _ { x }$ that was cut off during projection in Eq. (11). Since this is a translation, orthogonal to all $\boldsymbol { x } _ { i } ^ {  }$ , it has no effect on distances between $\boldsymbol { x } _ { i } ^ {  }$ ; it only shifts them back towards the original center $\mu _ { x }$ (compare Fig. 7 left and right). Besides these plots, Fig. 9 shows how CLIP-conditioned captions can still be generated with projected $\boldsymbol { x } _ { i } ^ {  }$ .

Concluding, it appears valid to interpret comparison of two $\boldsymbol { x } _ { i } ^ {  }$ still as image-image comparison. Decent performance from this $P C A ^ {  }$ comparison then suggests there is no issue with an intra-modal misalignment.

![](images/bcf84ff0537f8e311cfc7a68f3f9bc7fb2523739c8799fa5247bb7d64688e9ff.jpg)

<details>
<summary>scatter</summary>

| UMAP Component 1 | UMAP Component 2 | Category       |
| ---------------- | ---------------- | -------------- |
| -5.0             | 0.0              | Original Img   |
| -4.0             | 1.0              | Original Img   |
| -3.0             | -1.0             | Projected Img  |
| -2.0             | 2.0              | Projected Img  |
| -1.0             | -2.0             | Projected Img  |
| 0.0              | 3.0              | Projected Img  |
| 1.0              | -3.0             | Projected Img  |
| 2.0              | 4.0              | Text           |
| 3.0              | 5.0              | Text           |
| 4.0              | 6.0              | Text           |
| 5.0              | 7.0              | Text           |
| 6.0              | 8.0              | Text           |
| 7.0              | 9.0              | Text           |
| 8.0              | 10.0             | Text           |
| 9.0              | 11.0             | Text           |
| 10.0             | 12.0             | Text           |
| 11.0             | 13.0             | Text           |
| 12.0             | 14.0             | Text           |
| 13.0             | 15.0             | Text           |
| 14.0             | 16.0             | Text           |
| 15.0             | 17.0             | Text           |
| 16.0             | 18.0             | Text           |
| 17.0             | 19.0             | Text           |
| 18.0             | 20.0             | Text           |
| 19.0             | 21.0             | Text           |
| 20.0             | 22.0             | Text           |
| 21.0             | 23.0             | Text           |
| 22.0             | 24.0             | Text           |
| 23.0             | 25.0             | Text           |
| 24.0             | 26.0             | Text           |
| 25.0             | 27.0             | Text           |
| 26.0             | 28.0             | Text           |
| 27.0             | 29.0             | Text           |
| 28.0             | 30.0             | Text           |
| 29.0             | 31.0             | Text           |
| 30.0             | 32.0             | Text           |
| 31.0             | 33.0             | Text           |
| 32.0             | 34.0             | Text           |
| 33.0             | 35.0             | Text           |
| 34.0             | 36.0             | Text           |
| 35.0             | 37.0             | Text           |
| 36.0             | 38.0             | Text           |
| 37.0             | 39.0             | Text           |
| 38.0             | 40.0             | Text           |
| 39.0             | 41.0             | Text           |
| 40.0             | 42.0             | Text           |
| 41.0             | 43.0             | Text           |
| 42.0             | 44.0             | Text           |
| 43.0             | 45.0             | Text           |
| 44.0             | 46.0             | Text           |
| 45.0             | 47.0             | Text           |
| 46.0             | 48.0             | Text           |
| 47.0             | 49.0             | Text           |
| 48.0             | 50.0             | Text           |
| 49.0             | 51.0             | Text           |
| 50.0             | 52.0             | Text           |
| 51.0             | 53.0             | Text           |
| 52.0             | 54.0             | Text           |
| 53.0             | 55.0             | Text           |
| 54.0             | 56.0             | Text           |
| 55.0             | 57.0             | Text           |
| 56.0             | 58.0             | Text           |
| 57.0             | 59.0             | Text           |
| 58.0             | 60.0             | Text           |
| 59.0             | 61.0             | Text           |
| 60.0             | 62.0             | Text           |
| 61.0             | 63.0             | Text           |
| 62.0             | 64.0             | Text           |
| 63.0             | 65.0             | Text           |
| 64.0             | 66.0             | Text           |
| 65.0             | 67.0             | Text           |
| 66.0             | 68.0             | Text           |
| 67.0             | 69.0             | Text           |
| 68.0             | 70.0             | Text           |
| 69.0             | 71.0             | Text           |
| 70.0             | 72.0             | Text           |
| 71.0             | 73.0             | Text           |
| 72.0             | 74.0             | Text           |
| 73.0             | 75.0             | Text           |
| 74.0             | 76.0             | Text           |
| 75.0             | 77.0             | Text           |
| 76.0             | 78.0             | Text           |
| 77.0             | 79.0             | Text           |
| 78.0             | 80.0             | Text           |
| 79.0             | 81.0             | Text           |
| 80.0             | 82.0             | Text           |
| 81.0             | 83.0             | Text           |
| 82.0             | 84.0             | Text           |
| 83.0             | 85.0             | Text           |
| 84.0             | 86.0             | Text           |
| 85.0             | 87.0             | Text           |
| 86.0             | 88.0             | Text           |
| 87.0             | 89.0             | Text           |
| 88.0             | 90.0             | Text           |
| 89.0             | 91.0             | Text           |
| 90.0             | 92.0             | Text           |
| 91.0             | 93.0             | Text           |
| 92.0             | 94.0             | Text           |
| 93.0             | 95.0             | Text           |
| 94.0             | 96.0             | Text           |
| 95.0             | 97.0             | Text           |
| 96.0             | 98.0             | Text           |
| 97.0             | 99.0             | Text           |
| 98.0             | 100.0            | Text           |
| Note: The actual values for the original and projected images are not provided in the code snippet, so they are left blank in the code snippet for all three images in the original image and projected image data respectively in the original image data series.
</details>

![](images/10772ab1cf40030e92b5daadb32a58f6245cc9c5401cb1127c62111c54ac881d.jpg)

<details>
<summary>scatter</summary>

| UMAP Component 1 | UMAP Component 2 | Category        |
| ---------------- | ---------------- | --------------- |
| -5.0             | 10.0             | Text            |
| -8.0             | 7.0              | Text            |
| -6.0             | 6.0              | Text            |
| 10.0             | 9.0              | Projected Img   |
| 12.0             | 3.0              | Projected Img   |
| 14.0             | 4.0              | Projected Img   |
| 15.0             | 6.0              | Projected Img   |
| 16.0             | 5.0              | Projected Img   |
| 8.0              | 2.0              | Original Img    |
| 9.0              | 1.5              | Original Img    |
| 10.0             | 2.5              | Original Img    |
| 11.0             | 3.0              | Original Img    |
| 12.0             | 2.0              | Original Img    |
| 13.0             | 3.5              | Original Img    |
| 14.0             | 4.0              | Original Img    |
| 15.0             | 3.0              | Original Img    |
| 8.0              | 1.0              | Projected Img   |
| 9.0              | 2.0              | Projected Img   |
| 10.0             | 3.0              | Projected Img   |
| 11.0             | 4.0              | Projected Img   |
| 12.0             | 3.5              | Projected Img   |
| 13.0             | 4.5              | Projected Img   |
| 14.0             | 3.0              | Projected Img   |
| 15.0             | 5.0              | Projected Img   |
</details>

![](images/0a49dfc74d70b0c930739f94b56a9bfcb25b62124e5441cf89a92972bb9f7880.jpg)

<details>
<summary>scatter</summary>

| Category         | PC1 (18.64% variance) | PC2 (6.70% variance) |
| ---------------- | --------------------- | -------------------- |
| Original Img     | -0.4                  | 0.2                  |
| Projected Img    | 0.0                   | -0.4                 |
| Text             | 0.4                   | 0.3                  |
</details>

![](images/36b6bdba174ff0264c22709be1358933ccd7ad728877b0944d116f3a180497bb.jpg)

<details>
<summary>scatter</summary>

| PC1 (30.74% variance) | PC2 (3.23% variance) | Category        |
| --------------------- | -------------------- | --------------- |
| -0.4                  | 0.3                  | Original Img    |
| -0.3                  | 0.2                  | Projected Img   |
| 0.6                   | 0.1                  | Text            |
| 0.5                   | -0.1                 | Text            |
| -0.2                  | -0.3                 | Original Img    |
| -0.1                  | -0.2                 | Projected Img   |
| 0.4                   | 0.0                  | Text            |
| 0.6                   | 0.1                  | Text            |
</details>

![](images/e53a9857a405a8afe6975be7881e195f9ba01c92b325c7cbaeb5d765163ef80d.jpg)

<details>
<summary>scatter</summary>

| T-SNE Component 1 | T-SNE Component 2 | Category        |
| ----------------- | ----------------- | --------------- |
| -60               | 30                | Original Img    |
| -40               | 20                | Original Img    |
| -20               | 10                | Original Img    |
| 0                 | 0                 | Original Img    |
| 20                | -10               | Original Img    |
| 40                | -20               | Original Img    |
| 60                | -30               | Original Img    |
| 80                | -40               | Original Img    |
| -60               | -50               | Projected Img   |
| -40               | -40               | Projected Img   |
| -20               | -30               | Projected Img   |
| 0                 | -20               | Projected Img   |
| 20                | -10               | Projected Img   |
| 40                | 0                 | Projected Img   |
| 60                | 10                | Projected Img   |
| 80                | 20                | Projected Img   |
| -60               | 40                | Text            |
| -40               | 30                | Text            |
| -20               | 20                | Text            |
| 0                 | 10                | Text            |
| 20                | 0                 | Text            |
| 40                | -10               | Text            |
| 60                | -20               | Text            |
| 80                | -30               | Text            |
</details>

![](images/2c35c7ac380676e758bee2122b1a68a7d395be0aa8c1af2341275904732be199.jpg)

<details>
<summary>scatter</summary>

| T-SNE Component 1 | T-SNE Component 2 | Category        |
| ----------------- | ----------------- | --------------- |
| -50               | -10               | Original Img    |
| -40               | -30               | Projected Img   |
| 20                | 10                | Text            |
| 60                | -20               | Text            |
| 70                | 5                 | Text            |
| -30               | -40               | Projected Img   |
| -20               | 30                | Projected Img   |
| 10                | -15               | Original Img    |
| 80                | 25                | Text            |
| -60               | -50               | Projected Img   |
| -70               | -25               | Projected Img   |
| 30                | 20                | Text            |
| 90                | -10               | Text            |
</details>

Figure 7. Distribution of CLIP embeddings: text (orange), original image (purple), projected image (blue). UMAP (top), PCA (middle), t-SNE (bottom). Left column: projection via Eq. (11). Right column: mean-adjusted projection via Eq. (12). Interpretation: (i) Like many previous studies noted, text embeddings and image embeddings lie in two different cones, such that we can find them clearly separated in the figure. We argued in the main paper this “modality gap” does not lead to an intra-modal misalignment. (ii) Projecting via $P C A ^ {  }$ and re-projecting introduces a global shift (see left PCA) in $\mathbb { R } ^ { 5 1 2 }$ that is orthogonal to the principal components $\in \mathbb { R } ^ { 2 5 6 }$ . (iii) For comparison in $\mathbf { \hat { \mathbb { R } } ^ { 5 1 2 } }$ , we can compensate for this shift by adding back the mean of the canceled components (right column). ViT-B/16, ImageNet val.

![](images/be015bc22e0cc298a2e0681f13df88db9fa4e6faa4f3d99ee6e222333ea8f404.jpg)

<details>
<summary>histogram</summary>

| Cosine Similarity Range | text-image Sample Density | image-image Sample Density |
| ----------------------- | ------------------------- | -------------------------- |
| 0.10 - 0.12             | 1                         | 0                          |
| 0.12 - 0.14             | 3                         | 0                          |
| 0.14 - 0.16             | 6                         | 0                          |
| 0.16 - 0.18             | 9                         | 0                          |
| 0.18 - 0.20             | 14                        | 0                          |
| 0.20 - 0.22             | 15                        | 0                          |
| 0.22 - 0.24             | 10                        | 0                          |
| 0.24 - 0.26             | 6                         | 0                          |
| 0.26 - 0.28             | 3                         | 0                          |
| 0.28 - 0.30             | 1                         | 0                          |
| 0.30 - 0.32             | 0                         | 1                          |
| 0.32 - 0.34             | 0                         | 2                          |
| 0.34 - 0.36             | 0                         | 3                          |
| 0.36 - 0.38             | 0                         | 4                          |
| 0.38 - 0.40             | 0                         | 5                          |
| 0.40 - 0.42             | 0                         | 6                          |
| 0.42 - 0.44             | 0                         | 7                          |
| 0.44 - 0.46             | 0                         | 8                          |
| 0.46 - 0.48             | 0                         | 9                          |
| 0.48 - 0.50             | 0                         | 10                         |
| 0.50 - 0.52             | 0                         | 11                         |
| 0.52 - 0.54             | 0                         | 12                         |
| 0.54 - 0.56             | 0                         | 13                         |
| 0.56 - 0.58             | 0                         | 14                         |
| 0.58 - 0.60             | 0                         | 15                         |
| 0.60 - 0.62             | 0                         | 14                         |
| 0.62 - 0.64             | 0                         | 13                         |
| 0.64 - 0.66             | 0                         | 12                         |
| 0.66 - 0.68             | 0                         | 11                         |
| 0.68 - 0.70             | 0                         | 10                         |
| 0.70 - 0.72             | 0                         | 9                          |
| 0.72 - 0.74             | 0                         | 8                          |
| 0.74 - 0.76             | 0                         | 7                          |
| 0.76 - 0.78             | 0                         | 6                          |
| 0.78 - 0.80             | 0                         | 5                          |
| 0.80 - 0.82             | 0                         | 4                          |
| 0.82 - 0.84             | 0                         | 3                          |
| 0.84 - 0.86             | 0                         | 2                          |
| 0.86 - 0.88             | 0                         | 1                          |
| 0.88 - 0.90             | 0                         | 1                          |
| 0.90 - 0.92             | 0                         | 1                          |
| 0.92 - 0.94             | 0                         | 1                          |
| 0.94 - 0.96             | 0                         | 1                          |
| 0.96 - 0.98             | 0                         | 1                          |
| 0.98 - 1.00             | 0                         | 1                          |
</details>

![](images/353b39d90e20c510a87d54fb2a2d4195db304cb5d8f04f1369fed9227546e1aa.jpg)

<details>
<summary>area</summary>

| Cosine Similarity | text-image | image-image |
| ----------------- | ---------- | ----------- |
| 0.1               | 0          | 0           |
| 0.15              | 0          | 0           |
| 0.2               | 0          | 0           |
| 0.25              | 4          | 3           |
| 0.3               | 8          | 5           |
| 0.35              | 6          | 4           |
| 0.4               | 2          | 2           |
| 0.45              | 1          | 1           |
| 0.5               | 0          | 0           |
</details>

![](images/101a4ba3199e5b89c719b467f19df31b475b1bed1f63cb60ed8b7fccfa8f905c.jpg)

<details>
<summary>histogram</summary>

| Cosine Similarity Range | text-image Sample Density | image-image Sample Density |
| ------------------------ | ------------------------- | -------------------------- |
| 0.0 - 0.1                | 1                         | 0                          |
| 0.1 - 0.2                | 8                         | 0                          |
| 0.2 - 0.3                | 12                        | 0                          |
| 0.3 - 0.4                | 6                         | 0                          |
| 0.4 - 0.5                | 2                         | 0                          |
| 0.5 - 0.6                | 0                         | 4                          |
| 0.6 - 0.7                | 0                         | 9                          |
| 0.7 - 0.8                | 0                         | 6                          |
| 0.8 - 0.9                | 0                         | 3                          |
| 0.9 - 1.0                | 0                         | 1                          |
</details>

Figure 8. Modality gap: In the original CLIP space, image-image cosine similarities are high (left). With the projection from Eq. (11), these similarities seem to decrease (middle). Applying mean adjustment as in Eq. (12) removes this effect (right). Interpretation: Because mean adjustment preserves Euclidean distances, the observed changes arises solely from normalization: the projection zeros out components, reducing vector norms such that embeddings lie no more on, but inside the unit hypersphere. Normalization back on the hypersphere then squeezes the projected embeddings apart, leading to a lower cosine similarity value range (middle) compared to original CLIP (left). Adding back the canceled mean before normalization avoids this phenomenon (right). At the same time, there is no significant performance change between (middle) and (right), which once more illustrates that such cosine similarity histograms are insufficient indicators of alignment quality. ViT-B/16, ImageNet validation set.

![](images/0abe5ecfa081c64a9a4f60c8682d55cff72a08ea7041cfe4312a0f253d2d7f94.jpg)

<details>
<summary>natural_image</summary>

Outdoor basketball game in progress with players jumping under the hoop (no visible text or signage)
</details>

CLIP :   
A group of people standing around a basketball court.   
CLIP:   
A group of young men playing a game of basketball.

![](images/afd9ce3834da5a3474700822d864e522e1860d9f429bf4ded5169c097da3b112.jpg)

<details>
<summary>natural_image</summary>

Exterior view of a Gothic-style building with tall spires and a green lawn, framed by a window frame (no signage or text visible)
</details>

CLIP :   
A couple of pictures of a building with a castle in the background.   
CLIP:   
A large building with a large clock tower.

![](images/d9e869a32b803092e6862444ea95d0d259c3f7e1c597f39e0b4489075dadefb1.jpg)

<details>
<summary>natural_image</summary>

Two colorful macarads perched on a branch among green foliage (no text or symbols visible)
</details>

CLIP :   
Three parrots sitting on a tree branch next to each other.   
CLIP:   
Two parrots perched on a branch with leaves.

Figure 9. Captions generated with CLIP Prefix Captioning (Mokady et al. 2021), CLIP ViT-B/32, GPT-2. Swapping out the original CLIP image embedding xi for our projected x←i , we can observe the captions still cover the main objects. Samples are from datasets used for retrieval and few-shot classification in the main paper, left to right: SUN, ROxford, ImageNet.   
![](images/99676eff2c97c033ea0fd8f0335f27472048a0d504f296fb1c0fd27b51147942.jpg)

<details>
<summary>text_image</summary>

Night street photo showing a car at night with visible store signboards including 'Vansley' and 'CAR BOSA'
</details>

CLIP :   
A group of vehicles driving down a street.   
CLIP:   
A car is driving down a street at night.

![](images/0226aa972c0172a4a055b9bdd85cd49594348ff2414636b9383f2620e8225633.jpg)

<details>
<summary>text_image</summary>

Night street scene with illuminated buildings, traffic lights, and a visible '100' sign on the road
</details>

CLIP :   
A street scene with people walking and cars driving.   
CLIP:   
A city street at night with a lot of traffic.

![](images/46e5e7e9e1c9bd9a033de54717bc533b0fbdd22a20521cd0c5bffea74e80e49e.jpg)

<details>
<summary>natural_image</summary>

Street view with pedestrians crossing a crosswalk, surrounded by modern buildings and traffic lights (no visible text or signage)
</details>

CLIP :   
A collage of street signs and traffic lights.   
CLIP:   
A man is walking down the street while talking on his cell phone.   
Figure 10. The same experiment as in Fig. 9, but with the street scene dataset BDD100k used in the ablation in Sec. 4.3 of the main paper to validate our interpretation that the projection increases the “classness” of the image embedding, thereby being beneficial for classificationlike tasks, but harmful for tasks such as daytime recognition because some information is lost. In line with the finding of the main paper, here we can see how the captions generated with $P C A ^ {  }$ ignore that it is night (left, middle) and focus on the main objects only (right).