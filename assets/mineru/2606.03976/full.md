# Formalizing the Binding Problem

Lianghuan Huang \* 1 Yihao Li \* 2 Saeed Salehi 3 Yingshan Chang 4 Ansh Soni 5 Konrad P. Kording 6

# Abstract

Representations of the world, arguably, contain information about features (e.g. something is blue, something is a circle) but also information about which features are part of the same object (e.g. the circle is blue), which we call binding information. Any system with the ability to understand scenes with multiple objects must be able to solve the binding problem: it needs to know which features belong together. However, despite work showing that Vision Transformers (ViTs) know which patches belong together, it is not known whether current deep learning models learn to exhibit binding information, i.e., for features. We may believe that there is not much binding information, after all misattributing features to wrong objects is a common failure of ViT-based architectures, especially in scenes with objects sharing features. Here we formalize the binding problem with an information-theoretic approach, and introduce a probing method to measure binding information in model representations. We perform experiments on ViTs, measuring binding from different components of the architecture, such as the image summary token [CLS] or the spatial tokens. We use datasets with different binding challenges, such as feature sharing, occlusion, and natural features, while comparing the performance of several pre-trained ViTs. Overall, our research demonstrates binding as a key ingredient to strong visual recognition and reasoning.1

\*Equal contribution 1Department of Physics and Astronomy, University of Pennsylvania, Philadelphia, PA, USA 2Department of Computer and Information Science, University of Pennsylvania 3Machine Learning Group, Technical University of Berlin, Berlin, Germany 4Language Technology Institute, Carnegie Mellon University, Pittsburgh, PA, USA 5Department of Psychology, University of Pennsylvania 6Department of Neuroscience, University of Pennsylvania, Philadelphia, PA, USA. Correspondence to: Konrad P. Kording <kording@seas.upenn.edu>.

Proceedings of the 43 rd International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

1Code available at: https://github.com/ KordingLab/formalizing-the-binding-problem.

# 1. Introduction

Binding is the ability to tell which features belong to which objects in a multi-object scene. Humans naturally bind features to objects. This ability is so effortless that we rarely recognize binding as a computational problem in its own right. In the visual cortex, distinct neuronal populations are tuned to specific features (e.g., color, orientation, motion), but in multi-object scenes such feature-based encoding alone does not specify which features belong together as parts of the same object. Nevertheless, the brain achieves binding with extraordinary precision, and a range of mechanisms have been proposed to account for this capacity (Treisman & Gelade, 1980; von der Malsburg, 1981; Zhang et al., 2020) (more in Section A), but it remains largely an open problem (Yu & Lau, 2023).

Binding is just as equally important for artificial models. Compositional learning, a hallmark of generalization and reasoning for deep learning models (Uselis et al., 2025; Lake & Baroni, 2018; Hupkes et al., 2020; Wiedemer et al., 2023), relies crucially on the ability to bind the compositionally learned concepts into novel combinations (e.g., having learned the concepts of animals, body parts, and motion, can a model understand a flying penguin?) (Greff et al., 2020).

However, artificial vision and vision-language models perform markedly worse at binding than the brain. In multi-object scenes, especially when objects are cluttered and/or share features, the ability of modern vision and vision–language models to correctly identify objects and their corresponding features drops noticeably (Campbell et al., 2025; Zhang et al., 2024; Lewis et al., 2024; Yuksekgonul et al., 2023; Assouel et al., 2025b). For instance, (Campbell et al., 2025) shows that the ability of a visionlanguage model to accurately describe objects and their features monotonically decreases as the number of featuresharing object triplets increases in a scene (Figure 1), with similar trends observed in object counting, search, and analogy reasoning tasks. As another example, (Zhang et al., 2024) prompts vision-language models to describe objects in a given block of a Raven matrix (Figure 1). The model instead combines elements from adjacent blocks, attributing them to the same object. They further show that segmenting the grid into separate images before passing them in significantly reduces such errors. The failure of artificial vision models in binding as compared to the brain raises a natural question: to what extent do artificial models reliably learn binding information? Is there a framework that can precisely define binding information learned by a model, and is there a formal way of measuring binding as defined by the theory?

Using an information-theoretic approach, we define binding as the information a representation contains about which objects are present or absent among all possible objects. We then introduce several variants of this definition that separate binding information from feature information, or normalize over dataset binding uncertainty priors. We then introduce a probing method to measure binding information defined as such. Finally, we apply this framework to state-of-the-art Vision Transformers, measuring the binding information learned in the image-level summary token [CLS] and the full set of spatial tokens, on datasets with different binding challenges. Overall, our contributions are as follows:

• We introduce an information-theoretic framework that defines binding and a probing method for measuring binding information in model representations.   
• We show that the image-level summary token [CLS] encodes less than half of all binding information in a multi-object scene; for the binding information it does encode, it is largely structured quadratically.   
• We show that binding is encoded almost perfectly in the full set of spatial tokens, decodable using a simplified attention probe.   
• We perform probing experiments on a variety of models and datasets with different binding challenges, and analyze the binding information learned under each scenario.

# 2. Methods

Here we introduce an information-theoretic framework of formalizing binding, and show how binding information can be measured by probes. A scene has a set of features that are present. For the scene to make sense there are also objects, which are characterized by their specific conjunctions of features. And then there is the overall scene which is a collection of potential objects that are present or absent. Binding information is the information about which objects are present and absent.

# 2.1. Defining Binding Information

Definition 2.1 (Features). Let the $f _ { i }$ be discrete features. For example, they could be red, square, smooth, human, running, velocity=3m/s, occluded, top-left, etc.

![](images/34bb020b759e26de48537d3fdaa1993b499276594e46003bf961336a1eb2cd1b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Circle with black circle"] -->|1| B["Green diamond"]
    C["Circle with blue circle"] -->|2| D["Orange X"]
    E["Triangle with green triangle"] -->|3| F["Triangle with yellow triangle"]
    B --> G["Blue circle"]
    D --> H["Orange X"]
    F --> I["Triangle with yellow triangle"]
```
</details>

![](images/6b655694b1c71b28a77556c97b1aad5c1dd0f501e04a23f97ffb2c7f1e2321d9.jpg)

<details>
<summary>text_image</summary>

Grid puzzle with geometric shapes including squares, circles, triangles, and stars, arranged in a 3x3 pattern.
</details>

Figure 1. Binding failures. Left: (Campbell et al., 2025) shows that the ability of a vision-language model to accurately describe objects and their features in a scene degrades monotonically as the number of feature-sharing triplets (or the total number of objects) increases in a scene. Right: (Zhang et al., 2024) prompts visionlanguage models to describe grid patterns in the Raven matrix. For the middle-right block, the model outputs “a triangle with an X inside,” combining elements from the middle-center and the middle-right blocks. They further show that segmenting the grid into separate images before passing them in significantly reduces such errors. Figures reproduced from the papers.

Let $\mathcal { F } = \{ f _ { 1 } , f _ { 2 } , \cdot \cdot \cdot , f _ { n } \}$ be a finite set of all features of interest.

Definition 2.2 (Feature code). One way of looking at a scene is in terms of existence of features. Some of the features may be shared by multiple objects (e.g. a blue bag and a blue hat). But the feature existence vector F is useful to characterize which features are present. We define the feature code of scene as a finite random vector

$$
F = \left[ F _ {1}, F _ {2}, \dots , F _ {n} \right],
$$

where

$$
F _ {i} = \left\{ \begin{array}{l l} 1 & \text { if   feature } f _ {i} \text { is   present   in   the   scene }, \\ 0 & \text { otherwise }. \end{array} \right.
$$

Definition 2.3 (Objects). While F does transmit which features are present, it does not describe how features jointly make up objects. An object can be described by the collection of its features. We posit that each object corresponds to a distinct subset of ${ \mathcal F } .$

$$
o _ {i} \mapsto \left\{f _ {a}, f _ {b}, \dots \right\} \subseteq \mathcal {F}
$$

For example, $o _ { i }$ can be “red, blue, smooth cube,” “human running at velocity=3m/s,” or “top-left occluded triangle.”

Remark 2.4. We posit that no two objects can correspond to the exact same set of features. In practice, location is often a discriminator for objects that may share all other features.

For this reason, we consider the object to feature subset mapping a surjective map. More formally, let O = $\{ o _ { 1 } , o _ { 2 } , \cdots , o _ { n } \}$ be a finite set of all objects of interest. Then there exist a surjective mapping ψ such that

$$
\psi : \mathcal {O} \mapsto \mathcal {P} (\mathcal {F}),
$$

![](images/2576f0c85cd4566710523f16cbf7037f4a9c93580fc9d7efc78d6e17790edb50.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input X"] --> B["Representation Z"]
    B --> C{Probe f0(Z)}
    C --> D["Predicted O"]
    D --> E["Binding information= I(O;Z) = H(O) - H(O|Z)"]
    C --> F["Probe fF(Z)"]
    F --> G["Predicted F"]
    G --> H["Conditional binding information= I(O;Z|F) = H(O|F) - H(O|F,Z)<br>= H(O|F) - (H(O|Z) - H(F|Z))"]
    style A fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style C fill:#cfc,stroke:#333
    style D fill:#ffc,stroke:#333
    style E fill:#fcc,stroke:#333
    style F fill:#cff,stroke:#333
    style G fill:#ffc,stroke:#333
    style H fill:#fcc,stroke:#333
```
</details>

Figure 2. Binding theory and probing framework. We first define the space of features and objects of interest, as shown in features set F and object set O (Definitions 2.1, 2.3). Each object is a combination of features from the feature set (Definition 2.3, Remark 2.4). From there, we define the feature code F and object code O, which are random vectors that denote the presence and absence of each feature and object in a scene X. Binding information $I ( O ; Z )$ is the information of the object code in the representation Z (Definition 2.10). Conditional binding information is the information of the object code in the representation beyond what can be explained by features, hence $I ( O ; Z \mid { \breve { F } } )$ (Definition 2.13). Using information theory, the I terms can be decomposed into entropies H. $H ( O \mid Z )$ and $H ( F \mid Z )$ are the uncertainties of the object and feature codes in the representation, which can be estimated by training object code probe $f _ { O } ( Z )$ and feature code probe $f _ { F } ( Z )$ on the representation with ground truth labels of the object and feature codes (Section 2.2). The figure shows hypothetical probe-predicted logits for each element of the object and feature code. The cross-entropy losses of the object and feature probes are provable estimates of $H ( O \mid Z )$ and $H ( F \mid Z )$ (Theorem 2.20, Lemma 2.21). The remaining H(O) or $H ( O \mid F )$ terms are dataset priors which can be independently calculated from the distribution of the F and O in the dataset (Section 2.3). Combining these terms, we arrive at a probe-estimated value of binding information and conditional binding information.

where P(F) denotes the power set of F.

Definition 2.5 (Object code). An object code of a scene is a random vector

$$
O = [ O _ {1}, O _ {2}, \dots ]
$$

where

$$
O _ {i} = \left\{ \begin{array}{l l} 1 & \text { if   object   o_{i } is   present   in   the   scene}, \\ 0 & \text { otherwise }. \end{array} \right.
$$

Remark 2.6. To bind is to specify the object code of a scene, i.e., which objects exist and which objects do not (Definition 2.10). One potential difficulty of binding is that objects may share some (though not all) features with each other. For example, a scene may contain a red square, a blue circle, a red circle, but not a blue square. Their shared features can lead to binding errors under particular encoding schemes, especially when object encoding is correlated with feature encoding (Campbell et al., 2025; Greff et al., 2020; Treisman & Schmidt, 1982).

Remark 2.7. We note that our definition of object code is fully agnostic of the specific encoding scheme of objects of a model. While we define objects in feature-conjunctive ways (Definition 2.3), they may be encoded in many ways including tensor-product of features (Smolensky, 1990), slots (Locatello et al., 2020), object files (Kahneman et al., 1992), etc. For this reason, learning features well does not necessarily make learning objects easy for a model: in our definition, we do not assume their correlation or causation (although there usually is some correlation, see Definition 2.13).

Remark 2.8. The distinction between features and objects depends on our abstraction level. Objects can be features, and features can be objects, under different feature and object sets F, O. For example, “human” may be a feature among “chimpanzee” and “gorilla,” but it may be an object when “mammal,” “bipedal,” “vertebrate” are features.2

Remark 2.9. Assuming access to the object-to-featuresubset map ψ (Remark 2.4), for any particular scene, F is a deterministic function of O. That is, once we know which objects exist in a scene, we also know which features exist, by virtue of the object-to-feature map.

We consider a model Φ : X → Z which maps scenes to learned representations. Let $X \in { \mathcal { X } }$ be a random variable for the scene, and $Z \in { \mathcal { Z } }$ be a random variable for the representation. We consider F and O to be deterministic functions of X, given an oracle scene observer. The model Φ, however, may lose information of F and O in its representation of the scene X, i.e., they are irrecoverable even with the optimal decoder. We therefore define binding information in a representation Z as follows:

Definition 2.10 (Binding information). Let I(A; B) denote mutual information between random variables A and B.

We define binding information in the representation Z of a scene, relative to object set O (and feature set F ), as

$$
\mathbf {B} _ {\mathcal {O}} (Z) := I (O; Z),
$$

in unit of bits. In plain language, it is the information contained in the representation about the object code, i.e., about which objects are present and absent in a scene.

Remark 2.11. Let $H ( A )$ denote the entropy of the random variable A, and $H ( A \mid B )$ be its conditional entropy on B.

$$
\mathbf {B} _ {\mathcal {O}} (Z) := I (O; Z) = H (O) - H (O \mid Z) \tag {1}
$$

From the lens of entropy, binding information is also the reduction in the uncertainty of object code $O \left( H ( O ) \right)$ due to knowledge of the representation Z.

Definition 2.12 (Feature information). Similarly, we can define feature information in the representation Z as $I ( F ; Z )$ , relative to feature set F (and object set O).

Note that feature and object information are generally not independent within a representation. In particular, a representation that encodes features well can make objects easier to decode. Sometimes, however, we wish to measure the binding information contained in a representation beyond what can be explained by feature information alone. This may be especially desirable when comparing models with different feature learning capabilities,3 or when comparing scene distributions with different feature learning complexities.4 Under these scenarios, we may wish to measure binding information while controlling for decodable feature information. This can be done naturally by conditioning binding information on knowledge of the feature code.

Definition 2.13 (Binding information, conditioned on feature code). We define binding information in a representation Z, conditioned on feature code F , as

$$
\mathbf {B} _ {\mathcal {O}, \mathcal {F}} ^ {*} (Z) := I (O; Z \mid F)
$$

# Theorem 2.14.

$$
\mathbf {B} _ {\mathcal {O}, \mathcal {F}} ^ {*} (Z) = I (O; Z) - I (F; Z).
$$

Proof in Appendix C.

Remark 2.15. We note that conditioning on F does not result in a counterfactual measure of binding information as it would if a feature code were supplied to the model during

inference. The representation Z is fixed, so we are adjusting binding information post-hoc during measurement, for the amount of feature information, as is shown here.

Theorem 2.16. $\mathbf { B } _ { \mathcal { O } , \mathcal { F } } ^ { * } ( Z )$ can be similarly written in entropy form:

$$
\begin{array}{l} \mathbf {B} _ {\mathcal {O}, \mathcal {F}} ^ {*} (Z) = H (O \mid F) - H (O \mid Z, F) \\ \begin{array}{l} \mathcal {L} _ {O, \mathcal {F}} (Z) = H (O \mid Z) - H (O \mid Z, F) \\ = H (O) - H (O \mid Z) - H (F) + H (F \mid Z). \end{array} \tag {2} \\ \end{array}
$$

Proof in Appendix C.

Some scene distributions may have higher prior uncertainties in O, as measured by H(O). Binding information for a distribution is, at best, H(O), since

$$
\mathbf {B} _ {\mathcal {O}} (Z) := I (O; Z) = H (O) - H (O \mid Z) \leq H (O).
$$

When comparing the difficulty of binding across different scene distributions, we may wish to remove the scale effect of this prior H(O). This can be done by normalizing over $H ( O )$ .

Definition 2.17 (Binding information, normalized by prior uncertainty of object code). We define binding information normalized by the uncertainty of O in the scene distribution as

$$
\beta_ {\mathcal {O}} (Z) := \frac {I (O ; Z)}{H (O)} = 1 - \frac {H (O \mid Z)}{H (O)},
$$

without units.

Similarly, a normalized measure for the conditional binding information is

$$
\beta_ {\mathcal {O}, \mathcal {F}} ^ {*} (Z) := \frac {I (O ; Z \mid F)}{H (O \mid F)} = 1 - \frac {H (O \mid Z , F)}{H (O \mid F)}.
$$

Remark 2.18. The normalized binding information can therefore be interpreted as the proportion of object code uncertainty in the scene distribution $( H ( O )$ or $H ( O \mid F ) )$ resolved by knowledge of the representation.

Remark 2.19. Note that removing the scale effect of H(O) or $H ( O \mid F )$ does not remove other influences from the scene distribution, such as occlusion, background clutter, etc., which are often the points of comparison across datasets.

With binding information defined, we now show how it can be conveniently decoded by probes from model representations. Across variants above, there are two key terms: a representation-dependent term $H ( O \mid Z )$ or $H ( O \mid Z , F )$ , and a prior term depending only on the data distribution, $H ( O )$ or $H ( O \mid F )$ . We show how they can be calculated in Sections 2.2 and 2.3, respectively. And we summarize the full probing procedure in Section 2.4.

# 2.2. Probing Object Code on the Representation

We start with $H ( O \mid Z )$ which is defined as

$$
H (O \mid Z) = \mathbb {E} _ {(z, o) \sim p (Z, O)} \left[ - \log p (o \mid z) \right].
$$

A straightforward approach is to train a probe $q _ { \theta } ( o \mid z )$ to approximate $p ( o \mid z )$ . If we use cross-entropy loss

$$
\mathcal {L} _ {\mathrm{CE}} (\theta) = \mathbb {E} _ {(z, o) \sim p (Z, O)} \left[ - \log q _ {\theta} (o \mid z) \right], \tag {3}
$$

we will be directly approximating $H ( O \mid Z )$ , provided that $q _ { \theta } ( o \mid z )$ is a good approximation of $p ( o \mid z )$ . The following theorem captures this intuition:

Theorem 2.20 (Probe approximates binding uncertainty). Let $D ( p ( a \mid b ) \| q ( a \mid b ) )$ be the conditional relative entropy (Kullback–Leibler distance) between the two conditional probabilities $p ( a \mid b )$ and $q ( a \mid b )$ , where $( a , b ) \sim p ( A , B )$ . Then we have the following:

$$
\mathcal {L} _ {\mathrm{CE}} (\theta) = H (O \mid Z) + D (p (o \mid z) \| q _ {\theta} (o \mid z)).
$$

Proof in Appendix C.

Lemma 2.21. Since $D ( p \parallel q _ { \theta } ) \geq 0 ,$ , we have

$$
\mathcal {L} _ {\mathrm{CE}} (\theta) \geq H (O \mid Z).
$$

Remark 2.22. Lemma 2.21 shows that the further we can minimize ${ \mathcal { L } } _ { \mathrm { C E } } ( \theta )$ , the closer we will be to ground truth $H ( O \mid Z )$ . This requires us to train the probes properly until convergence, with the appropriate training schedule, regularization (since we report cross-entropy loss on the test set), and particularly, a good model family to begin with. We therefore compare a range of probe families in our experiments, such as linear, quadratic, deep neural networks, and attention (more in Section 3).

A practical challenge in training a $q _ { \theta } ( o \mid z )$ probe is the exponential space of its labels o (recall from Definition 2.5 that $O = o$ is a binary vector), which would require an exponential size dataset to cover the entire distribution space. We can circumvent this by decomposing the probe as follows:

$$
q _ {\theta} (o \mid z) = \prod_ {k = 1} ^ {K} q _ {\theta} (o _ {k} \mid o _ {<   k}, z), \tag {4}
$$

where $O _ { k }$ denotes the k-th component of the vector $^ { O , }$ and $O _ { < k }$ is the shorthand for $o _ { 1 } , \cdots , o _ { k - 1 } . ^ { 5 }$ This allows each $q _ { \theta } ( o _ { k } \mid o _ { < k } , z )$ to be trained individually with a singledimension label for $O _ { k }$ while incorporating $o _ { < k }$ as prior.

This amounts to probing for each object $O _ { k }$ in the representation, with the “code” for the prior $o _ { < k }$ objects given (Definitions 2.3, 2.5).6 Their cross-entropy loss can then be summed to obtain the overall loss in Eq. (3):

$$
\mathcal {L} _ {\mathrm{CE}} (\theta) = \sum_ {k = 1} ^ {K} \mathbb {E} _ {(z, o) \sim p (Z, O)} \left[ - \log q _ {\theta} (o _ {k} \mid o _ {<   k}, z) \right]. \tag {5}
$$

To approximate $H ( F \mid Z )$ in the conditional definition (Theorem 2.16), we can similarly use probes $q _ { \theta } ( f _ { k } \mid f _ { < k } , z )$ and sum their cross-entropy loss to obtain the ${ \mathcal { L } } _ { \mathrm { C E } } ( \theta )$ for features, which will then be our probe estimate of $H ( F \mid$ $Z )$ .

# 2.3. Estimating Dataset Priors

The prior terms in the binding definitions such as $H ( O )$ and $H ( F )$ can be empirically calculated using their entropy definition:

$$
H (O) = \mathbb {E} _ {o \sim p (O)} \left[ - \log p (o) \right],
$$

where $p ( O )$ is controllable for the synthetically generated datasets, similarly for $H ( F ) . ^ { 7 }$ Appendix F contains examples of calculating these quantities for our datasets.

# 2.4. Summary: Probing Binding on Model Representations

Finally, we summarize the procedure for probing binding information from model representations Z:

1. Find the dataset prior $H ( O )$ (and $H ( F )$ for conditional definition)   
2. Train binding probes $q _ { \theta } ( o _ { k } \mid o _ { < k } , z )$ (and $q _ { \theta } ( f _ { k } \ )$ $f _ { < k } , z )$ for conditional definition)   
3. Sum their cross-entropy loss to obtain a probe estimate of $H ( O \mid Z )$ (and $H ( F \mid Z )$ for conditional definition)   
4. Calculate binding information using Eq. (1) (Eq. (2) for conditional definition), with the option of normalization using Definition 2.17.

# 3. Results

Next, we probe ViT representations with real data. In particular, we are interested in understanding how binding works with different architectural components of the ViT (such as the [CLS] and the spatial tokens), and on different datasets with various challenges for binding.

In Sections 3.1 and 3.2, we ask which type of image representation in the ViT architecture is best for encoding binding: are summary tokens sufficient (e.g., the [CLS] or a global average pooling), or do we need the full set of spatial tokens? Both summary and spatial tokens are commonly used in pre-training objectives and downstream models: pretraining frameworks such as CLIP (Radford et al., 2021), simCLR (Chen et al., 2020), Barlow Twins (Zbontar et al., 2021), and other contrastive methods often use the [CLS] or globally pooled representations as the input to their training objectives; Vision-Language Models (VLMs), on the other hand, utilize the full set of spatial tokens in their crossmodality processing with language (Liu et al., 2023; Bai et al., 2023). Studying how well binding is encoded in these different types of representations can thus guide better model development and evaluation.

In Section 3.3, we study the data aspects of binding. We probe on datasets with different challenges for binding: datasets with an increasing number of features and objects in the space, datasets with different levels of occlusion, and natural datasets with high-level, semantic features.

# 3.1. To what extent does the [CLS] summary token encode binding?

To test binding, we create a synthetic ColorShape dataset with 8 colors and 8 shapes in the feature space, and 64 objects in the object space, each taking one shape and one color. The resulting feature and object codes have $\mathrm { d i m } ( F ) = 1 6 , \mathrm { d i m } ( O ) = 6 4$ . To create a balanced dataset for both the feature and object codes, we sample each image as follows: randomly choose 6 colors and 6 shapes, and out of the 36 possible objects, choose 18 objects randomly that cover all 6 colors and 6 shapes chosen, and place them in the image at random locations without overlap. The resulting baseline accuracy is $6 / 8 = 7 5 \%$ for feature probes $q _ { \theta } ( f _ { k } \mid f _ { < k } , z )$ and $1 - 1 8 / 6 4 = 7 1 . 9 \%$ for object probes $q _ { \theta } ( o _ { k } \mid o _ { < k } , z )$ , where the baseline is a trivial prediction of a constant 1 and 0, respectively.8 We report error reduction, (accuracy – baseline) $/ \left( 1 - \mathrm { b a s e l i n e } \right) \times 1 0 0 \%$ , in lieu of raw accuracy for our probes.

To test whether our probes learn generalizable patterns instead of memorization, we split the dataset into training, validation, and test sets. They each contain a disjoint set of feature codes F and object codes O. The latter occurs naturally by virtue of the large combinatorial space of the object codes $( \sim ~ 1 0 ^ { 1 2 } )$ , so every image almost certainly has a different object code. The feature code split is implemented separately. The resulting training set contains 39,200 samples, validation 9,800 samples, and test 9,800 samples.

We then compute the dataset entropies $H ( F )$ and $H ( O )$ for each split. Since all feature and object codes have equal probability in our setup, we only need the number of all possible feature codes and object codes under the feature code constraint for each split, and then take their logarithm to find the entropies. Both can be calculated with basic combinatorics, which we detail in Appendix F. On the test set for which we report binding information metrics, $H ( O ) = 3 9 . 9$ bits and $H ( F ) = 7 . 0$ bits.

Using this dataset, we probe for binding information from the [CLS] token of the DINOv2-Large ViT (Oquab et al., 2024). We train autoregressive probes $q _ { \theta } ( o _ { k } \mid o _ { < k } , z )$ to approximate the $H ( O \mid Z )$ term in the binding definition (one for each $o _ { k }$ , so 49 in total). To incorporate the conditional prior $O _ { < k } .$ , we directly concatenate $o _ { < k }$ with representation z: $x = [ z | | o _ { < k } ] . ^ { 9 }$ We experiment with three probe families:

• Linear probes: $\ell _ { k } ( x ) = W _ { k } x + b _ { k }$

• Quadratic probes: $\ell _ { k } ( x ) = x ^ { \top } W _ { k } x + b _ { k }$

• Deep neural network (DNN): $\ell _ { k } ( x ) = f _ { \mathrm { D N N } } ( x ) + b _ { k }$ where $f _ { \mathrm { D N N } }$ is 4 layers of 1024 width with GELU activations.

where $q _ { \theta } \bigl ( o _ { k } = 1 \mid o _ { < k } , z \bigr ) = \sigma ( \ell _ { k } ( x ) )$ .

We show the object code and feature code probing results in Table 1. All three probe families generalize well to unseen object and feature codes. In terms of accuracy, feature decoding can reach 90% in error reduction, while object decoding reaches merely around 65% percent. Linear probes perform the worst in both cases. The DNN probe performs the best, although only slightly better than the much smaller quadratic probe. Given that DNNs are known as “universal approximators” (Hornik et al., 1989), the DNN probes (with 3M parameters) reflect our best attempt at approaching the true binding (i.e., object) information in the representation, by closing the gap between the probe model $q _ { \theta } ( o \mid z )$ and the true distribution $p ( o \mid z )$ (Lemma 2.21). Since the quadratic probes achieve only slightly higher loss than the DNN probes, it is fair to assume that most binding (i.e., object) information and feature information in the representation is quadratic, and that higher-order interactions add little to no extra information decodable.10

Table 1. Average performance of object and feature code probes for different probe families when z is the [CLS] token. We report error reduction (ER) relative to the trivial majority-label baseline, which achieves 71.9% accuracy for object codes and 75.0% accuracy for feature codes. Numbers in parentheses denote the number of parameters relative to the corresponding linear probe. 

<table><tr><td>Probe type</td><td>Probe family</td><td>Train loss ↓ / ER ↑</td><td>Val loss ↓ / ER ↑</td><td>Test loss ↓ / ER ↑</td></tr><tr><td rowspan="3"> $q_{\theta}(o_{k} \mid o_{< k}, z)$ </td><td>Linear (×1)</td><td>33.8 / 38.4%</td><td>34.3 / 37.4%</td><td>34.2 / 37.4%</td></tr><tr><td>Quadratic (×64)</td><td>21.4 / 64.4%</td><td>22.0 / 63.3%</td><td>22.0 / 63.0%</td></tr><tr><td>DNN (×2953)</td><td>19.9 / 65.8%</td><td>20.6 / 64.4%</td><td>20.6 / 64.4%</td></tr><tr><td rowspan="3"> $q_{\theta}(f_{k} \mid f_{< k}, z)$ </td><td>Linear (×1)</td><td>4.2 / 73.2%</td><td>4.5 / 69.6%</td><td>4.4 / 70.4%</td></tr><tr><td>Quadratic (×64)</td><td>1.5 / 90.8%</td><td>1.6 / 89.2%</td><td>1.6 / 89.6%</td></tr><tr><td>DNN (×3042)</td><td>1.1 / 92.4%</td><td>1.3 / 90.8%</td><td>1.3 / 91.2%</td></tr></table>

To further understand how this quadratic binding information may be structured, we then devise a version of the quadratic probe where we reuse parameters across $o _ { k }$ probes when the objects $o _ { k }$ share features (e.g., red square probe and red circle probe share the “red” portion of their weights). More formally, we set $W _ { k } = U _ { \mathrm { c o l o r } } ^ { \top } V _ { \mathrm { s h a p e } }$ and the probe becomes

$$
\ell_ {k} (x) = x ^ {\top} U _ {\text { color }} ^ {\top} V _ {\text { shape }} x + b _ {k},
$$

where $U _ { \mathrm { c o l o r } }$ is shared for all objects with the same color, and $V _ { \mathrm { s h a p e } }$ is shared for all objects with the same shape. From Table 2, we observe a small 2.4-bit increase in loss relative to the non-parameter-reusing $o _ { k }$ probes. This suggests the strong presence of quadratic binding (i.e., object) information in the [CLS] token as the dot product between color and shape projections, a conjunctive encoding of objects based on features.

Table 2. Comparison of quadratic object code probes with and without parameter reuse. We report error reduction (ER) relative to the trivial majority-label baseline, which achieves 71.9% accuracy for object codes. Numbers in parentheses denote the number of parameters relative to the linear probe. 

<table><tr><td>Parameter reuse</td><td>Train loss ↓ / ER ↑</td><td>Val loss ↓ / ER ↑</td><td>Test loss ↓ / ER ↑</td></tr><tr><td>Yes (×1/6)</td><td>23.9 / 58.7%</td><td>24.4 / 58.0%</td><td>24.4 / 57.7%</td></tr><tr><td>No (×1)</td><td>21.4 / 64.4%</td><td>22.0 / 63.3%</td><td>22.0 / 63.0%</td></tr><tr><td>Δ (Yes - No)</td><td>+2.5 / -5.7 pp</td><td>+2.4 / -5.3 pp</td><td>+2.4 / -5.3 pp</td></tr></table>

Next we compute the amount of binding information and proportion of dataset binding uncertainty resolved, with or without conditioning on features, as we have defined in Section 2.1. Table 3 shows the results on the test set, using the probe losses tabled above. To be fair across probe types for the feature conditioned results, we use the quadratic feature probe loss for all of their feature terms, varying only the object probe type. The DNN probe, again, decodes the most amount of binding information, and only slightly more than the quadratic probe. Assuming that the DNN probe approaches the true binding information in the representation, our results show that the [CLS] token encodes less than half (48.5%) of the binding information. If we wish to separate binding information from feature information, the [CLS] encodes only 42.4% of the binding information beyond what can be explained by features.

Table 3. Probe-estimated binding information $\mathbf { B } _ { \mathcal { O } } ( Z ) ,$ , conditional binding information $\mathbf { B } _ { O , \mathcal { F } } ^ { * } ( Z )$ , and their normalized forms $\beta _ { \mathcal { O } } ( Z )$ and $\beta _ { O , \mathcal { F } } ^ { * } ( Z )$ , recorded on the test set. $H ( O ) =$ 39.9 bits and $H ( F ) = 7 . 0$ bits on this test set. All conditional results use the loss of the quadratic feature probe for the $H ( F \mid Z )$ term, varying the object code probe type. 

<table><tr><td>Probe type</td><td> $\mathbf{B}_{\mathcal{O}}(Z)$  (bits)  $\uparrow$ </td><td> $\beta_{\mathcal{O}}(Z)$   $\uparrow$ </td><td> $\mathbf{B}_{\mathcal{O},\mathcal{F}}^{*}(Z)$  (bits)  $\uparrow$ </td><td> $\beta_{\mathcal{O},\mathcal{F}}^{*}(Z)$   $\uparrow$ </td></tr><tr><td>Linear</td><td>5.7</td><td>14.3%</td><td>0.3</td><td>0.8%</td></tr><tr><td>Quadratic</td><td>17.9</td><td>44.9%</td><td>12.5</td><td>37.9%</td></tr><tr><td>DNN</td><td>19.4</td><td>48.5%</td><td>13.9</td><td>42.4%</td></tr></table>

# 3.2. To what extent does the full set of spatial tokens encode binding?

If a single summary token performs poorly for binding, would the full set of spatial tokens retain more binding information? Again, we use the same ColorShape dataset and DINOv2-Large ViT to probe for binding. Directly concatenating all spatial tokens, however, would produce a very high-dimensional input to the probe, making the number of parameters explode. So instead, we use a simplified attention probe that queries on the spatial tokens and learns their dynamically weighted mean:

Let $z = \{ s _ { i } \} _ { i = 1 } ^ { N }$ denote the full set of final-layer spatial tokens of the ViT. Each probe $q _ { \theta } ( o _ { k } \mid o _ { < k } , z )$ learns a query vector $q _ { k }$ conditioned on $\scriptstyle O _ { < k } \colon$

$$
q _ {k} = g _ {k} (o _ {<   k}).
$$

The query scores each spatial token by a dot product,

$$
a _ {k, i} = \frac {\exp (q _ {k} ^ {\top} s _ {i})}{\sum_ {j = 1} ^ {N} \exp (q _ {k} ^ {\top} s _ {j})},
$$

and forms a weighted spatial representation

$$
\bar {s} _ {k} = \sum_ {i = 1} ^ {N} a _ {k, i} s _ {i}.
$$

A final quadratic readout layer then predicts

$$
q _ {\theta} (o _ {k} = 1 \mid o _ {<   k}, \{s _ {i} \} _ {i = 1} ^ {N}) = \sigma (s _ {k} ^ {\top} W _ {k} \bar {s} _ {k} + b _ {k}).
$$

This probe is simpler than a typical transformer attention layer in that it uses learned queries, but no learned key or value projections. As shown in Table 4, this simplified attention probe can perform at around 97% in error reduction for object decoding, far outperforming the best DNN probe on the [CLS] token only.

Table 4. Average performance of attention probes with quadratic readout using the full set of spatial tokens, compared to the best-performing [CLS] probe for object and feature code prediction. We report error reduction (ER) relative to the trivial majority-label baselines, which achieve 71.9% accuracy for object codes and 75.0% accuracy for feature codes. 

<table><tr><td>Probe type</td><td>Probe family</td><td>Train loss ↓ / ER ↑</td><td>Val loss ↓ / ER ↑</td><td>Test loss ↓ / ER ↑</td></tr><tr><td rowspan="2"> $q_{\theta}(o_k \mid o < k, z)$ </td><td>Attention + spatial</td><td>2.9 / 97.2%</td><td>3.2 / 96.8%</td><td>3.1 / 96.8%</td></tr><tr><td>DNN + [CLS]</td><td>19.9 / 65.8%</td><td>20.6 / 64.4%</td><td>20.6 / 64.4%</td></tr><tr><td rowspan="2"> $q_{\theta}(f_k \mid f < k, z)$ </td><td>Attention + spatial</td><td>1.0 / 92.8%</td><td>1.2 / 91.2%</td><td>1.2 / 91.6%</td></tr><tr><td>DNN + [CLS]</td><td>1.1 / 92.4%</td><td>1.3 / 90.8%</td><td>1.3 / 91.2%</td></tr></table>

The binding information metrics calculated from these probes are listed in Table 5. As can be seen, the attention probe on the full set of spatial tokens decodes 92.2% of binding information, and for the binding information beyond what can be explained by features, there is 94.1% decoded. Both significantly outperform the best DNN probe on the [CLS] token only.

Table 5. Probe-estimated binding information $\mathbf { B } _ { \mathcal { O } } ( Z )$ , conditional binding information $\mathbf { B } _ { \mathcal { O } , \mathcal { F } } ^ { * } ( Z )$ , proportion of dataset binding information resolved $\beta _ { \mathcal { O } } ( Z )$ , and their normalized forms $\bar { \beta } _ { \mathcal { O } } ( Z )$ and $\beta _ { O , \mathcal { F } } ^ { * } ( Z )$ , recorded on the test set. $H ( O ) =$ 39.9 bits and $H ( F ) = 7 . 0$ bits on this test set. Attention probe’s conditional results use the loss of the attention feature probe, while DNN’s conditional results use the loss of the quadratic feature probe, as in Table 3. 

<table><tr><td>Probe type</td><td> $\mathbf{B}_{\mathcal{O}}(Z)$  (bits)  $\uparrow$ </td><td> $\beta_{\mathcal{O}}(Z) \uparrow$ </td><td> $\mathbf{B}_{\mathcal{O},\mathcal{F}}^{*}(Z)$  (bits)  $\uparrow$ </td><td> $\beta_{\mathcal{O},\mathcal{F}}^{*}(Z) \uparrow$ </td></tr><tr><td>Attention + spatial</td><td>36.8</td><td>92.2%</td><td>31.0</td><td>94.1%</td></tr><tr><td>DNN + [CLS]</td><td>19.4</td><td>48.5%</td><td>13.9</td><td>42.4%</td></tr></table>

Here in this probe, the attention weights ${ a } _ { k , i }$ can be interpreted as selecting which spatial tokens are most useful for decoding $o _ { k }$ . Qualitatively, we find that this routing process is highly accurate in our probes: when the target object $o _ { k }$ exists in the image (with label $o _ { k } = 1 )$ , the highest attention score $a _ { k , i }$ almost always maps to the target object patch.

# 3.3. To what extent can binding be learned on different datasets?

Does our binding framework capture the intrinsic difficulty of binding in the data distribution? We measure binding information on the following datasets with different binding challenges:

• ColorShape with the number of colors and shapes varying from 1 to 7 each, where the growing feature (and object) space can make it increasingly difficult for the model to bind well. We simplify the data distribution for easier comparison.   
• CLEVR with varying degree of occlusion between objects (Johnson et al., 2016), where occlusion can lead to ambiguous boundaries between objects.

• Visual Genome with densely annotated natural features (Krishna et al., 2017) where feature and object learning may be intrinsically more difficult.

Appendix G contains the detailed setups for each dataset. We note here that we construct image representations by concatenating the [CLS] token with the mean-pooled spatial tokens from the final layer of the ViT.

Influence of feature and object space complexity on binding. We run the CLIP ViT-L/14 (224px) model on the ColorShape dataset while varying the number of colors and shapes from 1 to 7. Figure 3 shows the results for all 49 configurations on this dataset. As shown in the figure (right), as the number of feature values (e.g., colors and shapes) increases, the space of possible bindings grows exponentially in the number of feature combinations, leading to a rapid increase in binding uncertainty (e.g., from 1 to $\bar { 2 } ^ { 4 3 }$ possibilities as the numbers of colors and shapes grow from 1 to 7). While the fraction of binding information captured by the model decreases with increasing complexity, it does not decay exponentially (Figure 3 Middle).

Influence of occlusion on binding. We probe the representations of DINOv2-Large on datasets with different levels of occlusion (examples shown in Figure 5), where we vary the level of occlusion (by adjusting the camera elevation and angle of the scene) As shown in Table 6, occlusion has a notable influence on a model’s ability to bind: binding information decodable monotonically decreases with higher levels of occlusion.

Table 6. Effect of occlusion on binding for DINOv2-Large, using CLEVR dataset with 4 colors and 3 shapes. Camera elevation serves as a proxy for occlusion level (higher means less occlusion). $\Delta \beta ( Z )$ is measured relative to the most occluded setting (0.6). 

<table><tr><td>ht.(Camera)</td><td> $\mathbf{B}_{\mathcal{O},\mathcal{F}}^{*}(Z)$  (bits)</td><td> $\beta_{\mathcal{O},\mathcal{F}}^{*}(Z)$ </td><td> $\Delta\beta_{\mathcal{O},\mathcal{F}}^{*}(Z)$  (pp)</td></tr><tr><td>0.6</td><td>5.8</td><td>45.0%</td><td>0.0</td></tr><tr><td>1.2</td><td>6.6</td><td>51.1%</td><td>+6.1</td></tr><tr><td>1.8</td><td>6.9</td><td>53.3%</td><td>+8.3</td></tr><tr><td>2.5</td><td>7.1</td><td>55.4%</td><td>+10.4</td></tr><tr><td>3.2</td><td>7.6</td><td>58.7%</td><td>+13.7</td></tr></table>

Binding on natural datasets. Table 7 shows binding information on the natural Visual Genome dataset, along with previous synthetic datasets across three models. Although natural features and objects are more difficult to learn (often emerging in later layers of a neural network), the models achieve comparable levels of binding with synthetic datasets. Additionally, we note from the table that our binding measure also captures model capability: increasing CLIP’s input resolution from 224px to 336px improves binding on all datasets, suggesting that finer spatial representations better support object-feature binding.

![](images/6bdfa2306a8d1f6f498956080b877bf4fc1cb24a5e8d03746d496b2987c0386a.jpg)

<details>
<summary>text_image</summary>

7 colors, 7 shapes
</details>

? O ? = ? = 47.73 ????

![](images/613503b96e39d83f3a58838a66939ea421fdab317b0745b637a38fae3cf260b2.jpg)

<details>
<summary>heatmap</summary>

| |1 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 6 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 7 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
</details>

|Shapesl

![](images/6ba8b8f17dd261a0296edcb334a8ab90f6bdd0829c3decf82497c042a8f444b4.jpg)  
IShapesl

Figure 3. Binding degrades with increasing feature space complexity in ColorShape. Middle: We run the CLIP ViT-L/14 (224px) model on the Color-Shape dataset with all 49 configurations of numbers of colors and shapes, and report the normalized binding information $\beta _ { O , F } ^ { * } ( Z )$ . The percentage of binding information captured by the model decreases as the dataset becomes more complex, involving more objects with more colors and shapes. Left: With the full feature code $F$ set to 7 colors and 7 shapes (all present), the uncertainty of binding is captured by $H ( O \mid F { \overset { \cdot } { = } } f )$ which is 47.73 bits. Right: The binding uncertainty of the dataset distribution $H ( O | F )$ for each configuration of numbers of colors and shapes.   
![](images/dd19f3903a0f449538445f68511ec4090435c3fb44638eb9779c62c6e5c42468.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| (green, cube, metal, large) | 3.97 bits |
| (cyan, cylinder, metal, small) | 3.91 bits |
| (blue, cylinder, metal, small) | 2.29 bits |
| (green, cylinder, rubber, large) | 0.63 bits |
| cyan, cube, metal, large) | 0.44 bits |
| (cyan, cylinder, rubber, large) | 0.37 bits |
</details>

![](images/24feaba24ad98f86576b5b90ef3dc85cc372414950b24d2346c2eb556d48b591.jpg)

<details>
<summary>bar</summary>

| Category | Value (bits) |
|---|---|
| brown,cylinder,metal,small) | 8.01 |
| brown,cube,metal,small) | 3.77 |
| (green,cube,metal,large) | 1.83 |
| (green,cylinder,rubber,large) | 1.46 |
| (red,cylinder,rubber,small) | 0.25 |
| (gray,cube,metal,large) | 0.24 |
</details>

Figure 4. CLEVR examples of object-code uncertainty revealed by binding probes. Blue bars $( H ( O \mid Z ) )$ ) indicate uncertainty in predicting objects. Left: Color misattribution in binding. Right: Uncertainty on the joint presence of a brown cylinder and a brown cube.

Table 7. Conditional binding information $\beta _ { O , F } ^ { * } ( Z )$ for different models and datasets. The ColorShape dataset here has 7 colors and 7 shapes, with feature and object code distributions detailed in Appendix G. VG:Color and VG:TopAttr are two subsets of Visual Genome. 

<table><tr><td>Model</td><td>ColorShape</td><td>CLEVR</td><td>VG:COLOR</td><td>VG:TopAttr</td></tr><tr><td>DINOv2-Large</td><td>41.8%</td><td>61.0%</td><td>41.8%</td><td>47.0%</td></tr><tr><td>CLIP ViT-L/14 (224px)</td><td>47.7%</td><td>68.1%</td><td>45.8%</td><td>39.9%</td></tr><tr><td>CLIP ViT-L/14 (336px)</td><td>56.4%</td><td>68.5%</td><td>46.4%</td><td>45.6%</td></tr></table>

# Discussion

In our work, we defined binding with an informationtheoretic framework and devised probing methods for measuring binding in model representations. We believe there are many benefits for taking a representational approach to defining binding: first, virtually all models are representation learners. Our definition can thus be widely applied for measurement and for comparison. Second, compared to directly measuring accuracy in binding tasks, the information approach (measuring entropy) takes into account the decision uncertainty of the model. Third, a representation measurement can enable us to understand the structure of the information by using differentially structured probes, as we have shown with the quadratic binding structure of the [CLS]. Finally, a representation-level diagnosis can guide approaches in model pre-training that specifically target better representations, such as the self-supervised pre-training of world models (Assran et al., 2025).

Future work can focus on improving binding performance of the latest models using the binding probe performances as objectives, and exploring potential architectural inductive biases to improve binding.

Our work does have limitations. Our framework relies on predefined discrete feature vocabularies; future work could extend it to continuous features with continuous probes. In addition, our probes measure decodable binding information rather than whether the model causally uses such information during downstream inference, so high probe performance may reflect latent information inaccessible to the model’s native readout mechanisms. Finally, the conditional definition $( B _ { O , F } ^ { * } ( Z ) )$ also assumes that feature code (F ) can be reliably inferred from object code (O), which may not hold in noisy biological or human perception.

# Acknowledgements

The authors would like to thank the anonymous reviewers for their helpful feedback.

# Impact Statement

This paper presents work whose goal is to advance the field of machine learning and neuroscience. There are many potential societal consequences of our work, including those related to improved visual capabilities of artificial intelligence and better understanding of biological vision.

# References

Andreas, J., Rohrbach, M., Darrell, T., and Klein, D. Neural module networks, 2017. URL https://arxiv.org/ abs/1511.02799.   
Assouel, R., Astolfi, P., Bordes, F., Drozdzal, M., and Romero-Soriano, A. Object-centric binding in contrastive language-image pretraining. arXiv preprint arXiv:2502.14113, 2025a.   
Assouel, R., Campbell, D., and Webb, T. Visual symbolic mechanisms: Emergent symbol processing in vision language models, 2025b. URL https://arxiv.org/ abs/2506.15871.   
Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Mojtaba, Komeili, Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Hogan, F. R., Dugas, D., Bojanowski, P., Khalidov, V., Labatut, P., Massa, F., Szafraniec, M., Krishnakumar, K., Li, Y., Ma, X., Chandar, S., Meier, F., LeCun, Y., Rabbat, M., and Ballas, N. V-jepa 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL https://arxiv.org/abs/2506.09985.   
Aydemir, G., Xie, W., and Guney, F. Self-supervised objectcentric learning for videos. Advances in Neural Information Processing Systems, 36:32879–32899, 2023.   
Bai, J., Bai, S., Yang, S., Wang, S., Tan, S., Wang, P., Lin, J., Zhou, C., and Zhou, J. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond, 2023. URL https: //arxiv.org/abs/2308.12966.   
Burgess, C. P., Matthey, L., Watters, N., Kabra, R., Higgins, I., Botvinick, M., and Lerchner, A. Monet: Unsupervised scene decomposition and representation. arXiv preprint arXiv:1901.11390, 2019.   
Campbell, D., Rane, S., Giallanza, T., Sabbata, N. D., Ghods, K., Joshi, A., Ku, A., Frankland, S. M., Griffiths, T. L., Cohen, J. D., and Webb, T. W. Understanding the limits of vision language models through

the lens of the binding problem, 2025. URL https: //arxiv.org/abs/2411.00238.

Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations, 2020. URL https://arxiv.org/abs/ 2002.05709.

Di Lollo, V. The feature-binding problem is an ill-posed problem. Trends in cognitive sciences, 16(6):317–321, 2012.

Feldman, J. The neural binding problem (s). Cognitive neurodynamics, 7(1):1–11, 2013.

Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K. V., Joulin, A., and Misra, I. Imagebind: One embedding space to bind them all. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15180–15190, 2023.

Gopalakrishnan, A., Stanic, A., Schmidhuber, J., and Mozer, ´ M. C. Recurrent complex-weighted autoencoders for unsupervised object discovery. Advances in Neural Information Processing Systems, 37:140787–140811, 2024.

Greff, K., Kaufman, R. L., Kabra, R., Watters, N., Burgess, C., Zoran, D., Matthey, L., Botvinick, M., and Lerchner, A. Multi-object representation learning with iterative variational inference. In International conference on machine learning, pp. 2424–2433. PMLR, 2019.

Greff, K., van Steenkiste, S., and Schmidhuber, J. On the binding problem in artificial neural networks, 2020. URL https://arxiv.org/abs/2012.05208.

He, K., Chen, X., Xie, S., Li, Y., Dollar, P., and Girshick, ´ R. Masked autoencoders are scalable vision learners. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 16000–16009, 2022.

Hornik, K., Stinchcombe, M., and White, H. Multilayer feedforward networks are universal approximators. Neural Networks, 2(5):359–366, 1989. ISSN 0893-6080. doi: https://doi.org/10.1016/0893-6080(89)90020-8. URL https://www.sciencedirect.com/ science/article/pii/0893608089900208.

Hu, T., Li, L., van de Weijer, J., Gao, H., Shahbaz Khan, F., Yang, J., Cheng, M.-M., Wang, K., and Wang, Y. Token merging for training-free semantic binding in textto-image synthesis. Advances in Neural Information Processing Systems, 37:137646–137672, 2024.

Hupkes, D., Dankers, V., Mul, M., and Bruni, E. Compositionality decomposed: how do neural networks generalise?, 2020. URL https://arxiv.org/abs/ 1908.08351.

Johnson, J., Hariharan, B., van der Maaten, L., Fei-Fei, L., Zitnick, C. L., and Girshick, R. Clevr: A diagnostic dataset for compositional language and elementary visual reasoning, 2016. URL https://arxiv.org/abs/ 1612.06890.   
Kahneman, D., Treisman, A., and Gibbs, B. J. The reviewing of object files: Object-specific integration of information. Cognitive Psychology, 24(2):175–219, 1992. ISSN 0010-0285. doi: https://doi.org/10.1016/0010-0285(92)90007-O. URL https://www.sciencedirect.com/ science/article/pii/001002859290007O.   
Karazija, L., Laina, I., and Rupprecht, C. Clevrtex: A texture-rich benchmark for unsupervised multi-object segmentation. arXiv preprint arXiv:2111.10265, 2021.   
Koishigarina, D., Uselis, A., and Oh, S. J. Clip behaves like a bag-of-words model cross-modally but not unimodally, 2025. URL https://arxiv.org/abs/ 2502.03566.   
Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.-J., Shamma, D. A., et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123(1):32–73, 2017.   
Krizhevsky, A., Sutskever, I., and Hinton, G. E. Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems, 25, 2012.   
Lake, B. M. and Baroni, M. Generalization without systematicity: On the compositional skills of sequenceto-sequence recurrent networks, 2018. URL https: //arxiv.org/abs/1711.00350.   
LeCun, Y., Bengio, Y., and Hinton, G. Deep learning. nature, 521(7553):436–444, 2015.   
Lewis, M., Nayak, N. V., Yu, P., Yu, Q., Merullo, J., Bach, S. H., and Pavlick, E. Does clip bind concepts? probing compositionality in large image models, 2024. URL https://arxiv.org/abs/2212.10537.   
Li, Y., Salehi, S., Ungar, L., and Kording, K. P. Does object binding naturally emerge in large pretrained vision transformers?, 2025. URL https://arxiv.org/ abs/2510.24709.   
Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., and Dollar, P. Microsoft coco: Common objects in con-´ text, 2015. URL https://arxiv.org/abs/1405. 0312.

Liu, H., Li, C., Wu, Q., and Lee, Y. J. Visual instruction tuning, 2023. URL https://arxiv.org/abs/2304. 08485.   
Liu, J., Yang, X., Li, W., and Wang, P. Finecops-ref: A new dataset and task for fine-grained compositional referring expression comprehension. arXiv preprint arXiv:2409.14750, 2024.   
Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., and Kipf, T. Object-centric learning with slot attention. Advances in neural information processing systems, 33:11525–11538, 2020.   
Miyato, T., Lowe, S., Geiger, A., and Welling, M. Ar- ¨ tificial kuramoto oscillatory neurons. arXiv preprint arXiv:2410.13821, 2024.   
Okawa, M., Lubana, E. S., Dick, R., and Tanaka, H. Compositional abilities emerge multiplicatively: Exploring diffusion models on a synthetic task. Advances in Neural Information Processing Systems, 36:50173–50195, 2023.   
Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., and Bojanowski, P. Dinov2: Learning robust visual features without supervision, 2024. URL https://arxiv.org/abs/2304.07193.   
Pearson, B., Boulbarss, B., Wray, M., and Lewis, M. Evaluating compositional generalisation in vlms and diffusion models, 2025. URL https://arxiv.org/abs/ 2508.20783.   
Puebla, G. and Bowers, J. S. Visual reasoning in objectcentric deep neural networks: A comparative cognition approach. Neural Networks, pp. 107582, 2025.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision, 2021. URL https://arxiv.org/abs/2103.00020.   
Reynolds, J. H. and Desimone, R. The role of neural mechanisms of attention in solving the binding problem. Neuron, 24(1):19–29, 1999.   
Robertson, L. C. Binding, spatial attention and perceptual awareness. Nature Reviews Neuroscience, 4(2):93–102, 2003.   
Roelfsema, P. R. Solving the binding problem: Assemblies form when neurons enhance their firing rate—they don’t

need to oscillate or synchronize. Neuron, 111(7):1003– 1019, 2023.   
Roelfsema, P. R. and Serre, T. Feature binding in biological and artificial vision. Trends in Cognitive Sciences, 2025.   
Rosenblatt, F. Principles of neurodynamics: Perceptrons and the theory of brain mechanisms, volume 55. Spartan books Washington, DC, 1962.   
Roskies, A. L. The binding problem. Neuron, 24(1):7–9, 1999.   
Rubinstein, A., Prabhu, A., Bethge, M., and Oh, S. J. Are we done with object-centric learning? arXiv preprint arXiv:2504.07092, 2025.   
Sabour, S., Frosst, N., and Hinton, G. E. Dynamic routing between capsules, 2017. URL https://arxiv.org/ abs/1710.09829.   
Salehi, S., Lei, J., Benjamin, A. S., Muller, K.-R., and¨ Kording, K. P. Modeling attention and binding in the brain through bidirectional recurrent gating. BioRxiv, pp. 2024–09, 2024.   
Scholte, H. S. and de Haan, E. H. Beyond binding: from modular to natural vision. Trends in Cognitive Sciences, 2025.   
Seitzer, M., Horn, M., Zadaianchuk, A., Zietlow, D., Xiao, T., Simon-Gabriel, C.-J., He, T., Zhang, Z., Scholkopf, ¨ B., Brox, T., et al. Bridging the gap to real-world objectcentric learning. arXiv preprint arXiv:2209.14860, 2022.   
Seitzer, M., Horn, M., Zadaianchuk, A., Zietlow, D., Xiao, T., Simon-Gabriel, C.-J., He, T., Zhang, Z., Scholkopf, B., ¨ Brox, T., and Locatello, F. Bridging the gap to real-world object-centric learning, 2023. URL https://arxiv. org/abs/2209.14860.   
Seo, H., Bang, J., Lee, H., Lee, J., Lee, B. H., and Chun, S. Y. Geometrical properties of text token embeddings for strong semantic binding in text-to-image generation. arXiv preprint arXiv:2503.23011, 2025.   
Simeoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab,´ M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.   
Singer, W. and Gray, C. M. Visual feature integration and the temporal correlation hypothesis. Annual review of neuroscience, 18(1):555–586, 1995.   
Smolensky, P. Tensor product variable binding and the representation of symbolic structures in connectionist systems. Artif. Intell., 46(1–2):159–216, November 1990. ISSN 0004-3702. doi: 10.1016/0004-3702(90)

90007-M. URL https://doi.org/10.1016/ 0004-3702(90)90007-M.   
Treisman, A. The binding problem. Current opinion in neurobiology, 6(2):171–178, 1996.   
Treisman, A. Solutions to the binding problem: progress through controversy and convergence. Neuron, 24(1): 105–125, 1999.   
Treisman, A. and Schmidt, H. Illusory conjunctions in the perception of objects. Cognitive Psychology, 14(1):107–141, 1982. ISSN 0010-0285. doi: https://doi.org/10.1016/0010-0285(82)90006-8. URL https://www.sciencedirect.com/ science/article/pii/0010028582900068.   
Treisman, A. M. and Gelade, G. A feature-integration theory of attention. Cognitive Psychology, 12 (1):97–136, 1980. ISSN 0010-0285. doi: https://doi.org/10.1016/0010-0285(80)90005-5. URL https://www.sciencedirect.com/ science/article/pii/0010028580900055.   
Uselis, A., Dittadi, A., and Oh, S. J. Does data scaling lead to visual compositional generalization?, 2025. URL https://arxiv.org/abs/2507.07102.   
von der Malsburg, C. The correlation theory of brain function. Technical Report Internal Report 81-2, Max-Planck-Institute for Biophysical Chemistry, Department of Neurobiology, Gottingen, West ¨ Germany, 1981. URL https://web-archive. southampton.ac.uk/cogprints.org/1380/.   
Von der Malsburg, C. The what and why of binding: the modeler’s perspective. Neuron, 24(1):95–104, 1999.   
Wiedemer, T., Mayilvahanan, P., Bethge, M., and Brendel, W. Compositional generalization from first principles, 2023. URL https://arxiv.org/abs/2307. 05596.   
Wolfe, J. M. The binding problem lives on: Comment on di lollo. Trends in cognitive sciences, 16(6):307, 2012.   
Wu, Y., Geiger, A., and Milliere, R. How do transformers \` learn variable binding in symbolic programs? arXiv preprint arXiv:2505.20896, 2025.   
Yu, X. and Lau, E. The binding problem 2.0: Beyond perceptual features. Cognitive Science, 47(2):e13244, 2023. doi: 10.1111/cogs.13244.   
Yuksekgonul, M., Bianchi, F., Kalluri, P., Jurafsky, D., and Zou, J. When and why vision-language models behave like bags-of-words, and what to do about it?, 2023. URL https://arxiv.org/abs/2210.01936.

Zbontar, J., Jing, L., Misra, I., LeCun, Y., and Deny, S. Barlow twins: Self-supervised learning via redundancy reduction, 2021. URL https://arxiv.org/abs/ 2103.03230.   
Zhang, Y., Zhang, Y.-Y., and Fang, F. Neural mechanisms of feature binding. Sci. China Life Sci, 63:926–928, 2020.   
Zhang, Y., Bai, H., Zhang, R., Gu, J., Zhai, S., Susskind, J., and Jaitly, N. How far are we from intelligent visual deductive reasoning?, 2024. URL https://arxiv. org/abs/2403.04732.   
Zhou, J., Wei, C., Wang, H., Shen, W., Xie, C., Yuille, A., and Kong, T. ibot: Image bert pre-training with online tokenizer. arXiv preprint arXiv:2111.07832, 2021.

# A. Related Work

The Binding Problem With its roots in early artificial intelligence (Rosenblatt, 1962), the binding problem has puzzled computer scientists and cognitive neuroscientists for over half a century (Greff et al., 2020; Von der Malsburg, 1999; Feldman, 2013; Roskies, 1999; Treisman, 1996). Since early artificial vision models were in their infancy and lacked the capacity for complex scene decomposition, the initial momentum for understanding feature binding came largely from psychology and neuroscience (Treisman, 1999; von der Malsburg, 1981). Although there is yet no consensus on the exact neural mechanisms of binding in the brain (Robertson, 2003; Wolfe, 2012; Di Lollo, 2012; Scholte & de Haan, 2025; Roelfsema & Serre, 2025), significant progress has been made in identifying some of its core mechanistic components (Treisman & Gelade, 1980; Reynolds & Desimone, 1999; Roelfsema, 2023; Singer & Gray, 1995). These biological insights have inspired architectures designed to explicitly address binding, whether through temporal synchrony (Miyato et al., 2024; Gopalakrishnan et al., 2024) or attention-based mechanisms (Salehi et al., 2024).

Feature Binding in Deep Neural Networks Following the unprecedented success of deep learning in computer vision (Krizhevsky et al., 2012; LeCun et al., 2015), the binding problem has gained renewed interest within the machine learning community. Earlier approaches attempted to address binding through explicit object-centric architectural and training biases (Locatello et al., 2020; Puebla & Bowers, 2025; Assouel et al., 2025a; Aydemir et al., 2023), known as Object-Centric Learning (OCL) (Locatello et al., 2020; Greff et al., 2019). Approaches such as Slot Attention (Locatello et al., 2020) and MONet (Burgess et al., 2019) decompose inputs to permutation-invariant ”slots” via iterative refinement, each representing an object. Although these models excel at grouping spatially coherent regions (object-part binding), they arguably fall short in feature binding, often struggling to correctly associate complex attributes like texture with their respective objects (Karazija et al., 2021). Also, their general applicability is largely surpassed by the performance and scalability of selfsupervised Vision Transformers (ViTs) (Simeoni et al. ´ , 2025; He et al., 2022; Zhou et al., 2021) and multi-modal models (Radford et al., 2021; Girdhar et al., 2023; Rubinstein et al., 2025). State-of-the-art ViTs now excel in both global recognition and dense prediction tasks (e.g., segmentation) (Simeoni et al. ´ , 2025) without requiring explicit architectural components or training process for binding. This has prompted a growing body of research investigating the degree to which binding capabilities naturally emerge in these models (Li et al., 2025), identifying the underlying mechanisms that drive their partial success (Okawa et al., 2023; Seitzer et al., 2022; Wu et al., 2025), and proposing methods to remedy their limitations (Seo et al., 2025; Hu et al., 2024; Koishigarina et al., 2025).

Probing for Binding A rigorous evaluation of feature binding requires both a suitably complex dataset and a precise metric for quantification. An ideal dataset must include scenes with multiple objects to challenge the model’s binding capability, while providing ground-truth annotations that link objects to their specific attributes. Standard benchmarks such as MS-COCO (Lin et al., 2015) and synthetic environments like CLEVR (Johnson et al., 2016) have traditionally served this role, while recent works have introduced specialized datasets to target fine-grained compositional understanding (Liu et al., 2024). In terms of measurement, existing approaches fall into two categories. The ”symptomatic” approaches assess binding indirectly via performance on high-level downstream tasks, such as visual reasoning (Campbell et al., 2025), text-image retrieval errors (Yuksekgonul et al., 2023) and compositional generalization (Lewis et al., 2024; Pearson et al., 2025). Conversely, ”mechanistic” approaches aim to evaluate binding directly within the internal representations, offering a task-agnostic measure of how information is organized and bound at the low level (Li et al., 2025).

Recent studies have shown that vision-language models (e.g., CLIP) behave like bag-of-words and do not reliably bind the correct attributes to their corresponding entity (Koishigarina et al., 2025; Yuksekgonul et al., 2023), but these analyses rely primarily on linear probes. Since binding represents a combinatorial problem where linear separation may require exponentially high dimensionality, we argue that linear probes are insufficient. We therefore train non-linear probes to capture these complex relationships. (Li et al., 2025) propose quadratic probes to evaluate the object-binding property of IsSameObject. However, this approach is not easily generalizable and implicitly assumes that each patch corresponds to a single, well-isolated object. Here, we provide a more formalized framework for binding that, in principle, extends to arbitrary models, representations, and feature definitions.

# B. Data Processing Inequality

Theorem B.1 (Internal processing does not increase binding information). A model typically has several layers, yielding internal representations $Z _ { 1 } , \cdots , Z _ { n } .$ . For a deterministic model where

$$
Z _ {1} = g _ {0} (X), \quad Z _ {i + 1} = g _ {i} (Z _ {i}), \quad i \in [ 1 \dots n ],
$$

there is

$$
\mathbf {B} _ {\mathcal {O}} (X) \geq \mathbf {B} _ {\mathcal {O}} (Z _ {1}) \geq \dots \geq \mathbf {B} _ {\mathcal {O}} (Z _ {n}),
$$

where we define $\mathbf { B } _ { \mathcal { O } } ( X ) : = I ( O ; X \mid F )$ . That is, any representation $Z _ { i }$ cannot increase binding information in the input X, and neither does binding information increase through the internal processing of the model.

Proof. Since $Z _ { i + 1 } \perp \perp Z _ { < i } \mid Z _ { i } ,$ , we have a Markov Chain: $Z _ { 1 } \to \cdots \to Z _ { n }$ . Since X is generated from $p ( X \mid O )$ (where O specifies the feature values of objects), we can add O to the Markov Chain: $O \to X \to Z _ { 1 } \to \cdot \cdot \cdot \to Z _ { n }$ . For simplicity of notation, we denote $X : = Z _ { 0 }$ .

For any $0 \leq i < j$ , we first show that $I ( Z _ { j } ; O \mid Z _ { i } , F ) = 0 . I ( Z _ { j } ; O \mid Z _ { i } , F ) = H ( Z _ { j } \mid Z _ { i } , F ) - H ( Z _ { j } \mid O , Z _ { i } , F )$ . Since $0 \leq H ( Z _ { j } \mid Z _ { i } , F ) \leq H ( Z _ { j } \mid Z _ { i } ) = 0 , 0 \leq H ( Z _ { j } \mid O , Z _ { i } , F ) \leq H ( Z _ { j } \mid Z _ { i } ) = 0$ , we have $H ( Z _ { j } \ | \ Z _ { i } , F ) = H ( Z _ { j } \ |$ $O , Z _ { i } , F ) = 0 .$ . Hence $I ( Z _ { j } ; O \mid Z _ { i } , F ) = 0$ .

By the chain rule of information,

$$
I (Z _ {i}; O \mid F) = I (Z _ {i}; O \mid F) + I (Z _ {j}; O \mid F, Z _ {i}) = I (Z _ {i}, Z _ {j}; O \mid F) = I (Z _ {j}; O \mid F) + I (Z _ {i}; O \mid F, Z _ {j}) \geq I (Z _ {j}; O \mid F).
$$

# C. Proofs

# C.1. Theorem 2.14

Proof. We know that

$$
I (O; Z) = H (O) - H (O \mid Z)
$$

$$
I (F; Z) = H (F) - H (F \mid Z) \tag {6}
$$

$$
\mathbf {B} _ {\mathcal {O}} (Z) = H (O \mid F) - H (O \mid F, Z) \tag {7}
$$

Adding Eqs. 6 and 7, we have $I ( F ; Z ) + \mathbf { B } _ { \mathcal { O } } ( Z ) = H ( O , F ) - H ( O , F \mid Z ) = H ( F ) - H ( O \mid Z ) = I ( O ; Z )$ . The penultimate equality is due to the fact that F is a deterministic function of O. □

# C.2. Theorem 2.16

Proof. By definition,

$$
\mathbf {B} _ {\mathcal {O}, \mathcal {F}} ^ {*} (Z) = I (O; Z \mid F) = H (O \mid F) - H (O \mid Z, F).
$$

Since F is a deterministic function of O, $H ( F \mid O ) = H ( F \mid O , Z ) = 0 .$ . Hence,

$$
H (O \mid F) = H (O, F) - H (F) = H (O) - H (F),
$$

and

$$
H (O \mid Z, F) = H (O, F \mid Z) - H (F \mid Z) = H (O \mid Z) - H (F \mid Z).
$$

Substituting gives

$$
\mathbf {B} _ {\mathcal {O}, \mathcal {F}} ^ {*} (Z) = H (O) - H (O \mid Z) - H (F) + H (F \mid Z).
$$

# C.3. Theorem 2.20

Proof. By definition,

$$
\mathcal {L} _ {\mathrm{CE}} (\theta) = \mathbb {E} _ {(o, z) \sim p (O, Z)} \left[ - \log q _ {\theta} (o \mid z) \right].
$$

Adding and subtracting log p(o | z), we get

$$
\mathcal {L} _ {\mathrm{CE}} (\theta) = \mathbb {E} _ {(o, z)} [ - \log p (o \mid z) ] + \mathbb {E} _ {(o, z)} \left[ \log \frac {p (o \mid z)}{q _ {\theta} (o \mid z)} \right].
$$

The first term is H(O | Z), and the second term is

$$
D (p (o \mid z) \| q _ {\theta} (o \mid z)).
$$

Therefore,

$$
\mathcal {L} _ {\mathrm{CE}} (\theta) = H (O \mid Z) + D (p (o \mid z) \| q _ {\theta} (o \mid z)).
$$

# D. Experimental Setup

We train probes on frozen visual representations. A summary of the default probe architectures and training hyperparameters is provided in Table 8. All experiments are run on NVIDIA RTX 4090 GPUs.

In the dataset comparisons, we construct image representations by concatenating the [CLS] token with the mean-pooled spatial tokens from the final layer of the vision encoder.

Table 8. Default probe architectures and training hyperparameters. 

<table><tr><td>Trainer component</td><td>Setting</td></tr><tr><td>Batch size</td><td>512</td></tr><tr><td>Gradient Accumulation</td><td>2</td></tr><tr><td>Learning rate</td><td> $1 \times 10^{-3}$ </td></tr><tr><td>Weight decay</td><td> $1 \times 10^{-4}$ </td></tr><tr><td>Epochs</td><td>40 epochs (converged for all probes)</td></tr><tr><td>LR scheduler</td><td>StepLR (step size = 8,  $\gamma = 0.2$ )</td></tr></table>

Models. We evaluate feature binding behavior using a diverse set of pretrained Vision Transformer backbones (Table 9). All models are used in a frozen setting.

Table 9. Vision Transformer backbones used for evaluation. 

<table><tr><td>Model family</td><td>HuggingFace identifier</td></tr><tr><td>DINOv2-Large</td><td>facebook/dinov2-large</td></tr><tr><td>CLIP ViT-L/14 (224px)</td><td>openai/clip-vit-large-patch14</td></tr><tr><td>CLIP ViT-L/14 (336px)</td><td>openai/clip-vit-large-patch14-336</td></tr></table>

# E. Ablation on $O _ { < k }$ labels

Table 10 shows that ablating on the conditionals leads to a notable decrease in binding information decoded, suggesting that it is necessary to condition on the $o _ { < k }$ labels as required by the decomposition in Eq. 4.

Table 10. Ablation of conditioning quadratic object code probes on previous object codes $O _ { < k }$ . We report error reduction (ER) relative to the trivial majority-label baseline, which achieves 71.9% accuracy for object codes. 

<table><tr><td>Condition on  $o_{<k}$ </td><td>Train loss ↓ / ER ↑</td><td>Val loss ↓ / ER ↑</td><td>Test loss ↓ / ER ↑</td></tr><tr><td>Yes</td><td>21.4 / 64.4%</td><td>22.0 / 63.3%</td><td>22.0 / 63.0%</td></tr><tr><td>No</td><td>26.6 / 52.3%</td><td>27.3 / 50.9%</td><td>27.3 / 50.9%</td></tr><tr><td> $\Delta$ (Yes – No)</td><td>-5.2 / +12.1 pp</td><td>-5.3 / +12.5 pp</td><td>-5.3 / +12.1 pp</td></tr></table>

# F. Estimating Dataset Priors

Computing $H ( O )$ and $H ( F )$ for ColorShape (8 colors and 8 shapes) To find $H ( O )$ and $H ( F )$ for the ColorShape dataset, we only need to find the size of their support in each split and then take their logarithm, since they are both evenly distributed. Prior to splitting, there are $( \ l _ { 6 } ^ { 8 } ) \times ( \ l _ { 6 } ^ { 8 } ) = 7 8 4$ possibilities of choosing F . The training/validation/test sets each takes 522/131/131 disjoint values of F . Hence $H _ { \mathrm { t r a i n } } ( F ) = \log _ { 2 } ( 5 2 2 ) = 9 . 0 ~ { \mathrm { b i t s } } , H _ { \mathrm { v a l } } ( F ) = H _ { \mathrm { t e s t } } ( F ) = \log _ { 2 } ( 1 3 1 ) = 7 . 0$ bits.

Next we find the support size of O for each split. For each chosen set of 6 colors and 6 shapes, we now find the number of ways to choose 18 objects out of the 36 possibilities while ensuring coverage of all features. We use the inclusive-exclusion formula by first choosing 18 objects out of 36,  3618, and then subtracting that by scenarios where we miss at least one feature $- \binom { 6 } { 1 } \binom { 5 \times 6 } { 1 8 } - \binom { 6 } { 1 } \binom { 6 \times 5 } { 1 8 }$ 1 and adding back scenarios where we miss at least two features ${ \binom { 6 } { 2 } } { \binom { 4 \times 6 } { 1 8 } } + { \binom { 6 } { 1 } } { \binom { 6 } { 1 } } { \binom { 6 \times 4 } { 1 8 } } + { \binom { 6 } { 2 } } { \binom { 5 \times 5 } { 1 8 } }$ , and so on. More compactly, the support size of O, given a chosen set of 6 colors and 6 shapes, is as follows:

$$
\sum_ {i = 0} ^ {6} \sum_ {j = 0} ^ {6} (- 1) ^ {i + j} \binom {6} {i} \binom {6} {j} \binom {(6 - i) (6 - j)} {1 8} = 8, 0 5 8, 5 2 5, 4 4 0.
$$

Now, O is different for different chosen sets of colors and shapes, so the total support size is the above multiplied by the support size of F in each split. Hence $H _ { \mathrm { t r a i n } } ( O ) \ = \ \log _ { 2 } ( 8 , 0 5 8 , 5 2 5 , 4 4 0 \times 5 2 2 ) \ = \ 4 1 . 9$ bits, and similarly $H _ { \mathrm { v a l } } ( O ) = H _ { \mathrm { t e s t } } ( O ) = \log _ { 2 } ( 8 , 0 5 8 , 5 2 5 , 4 4 0 \times 1 3 1 ) = 3 9 . 9 \mathrm { b i t s }$ s.

Computing $H ( O \mid F )$ for ColorShape (1 – 7 colors/shapes) and CLEVR In ColorShape and CLEVR, we generate synthetic data with full control, enforcing a uniform distribution over all valid binding configurations, with the number of objects capped at $k _ { \mathrm { m a x } }$ . Under this assumption,

$$
H (O \mid F = f) = \log_ {2} \left(\sum_ {k \in \mathcal {K} (f)} N _ {k} (a _ {1}, \dots , a _ {G})\right),
$$

where f denotes a fixed feature realization, $\kappa ( f )$ is the set of admissible object counts consistent with $f ,$ and $N _ { k } ( a _ { 1 } , \ldots , a _ { G } )$ counts the number of valid binding configurations with exactly k objects.

Because the distribution of F is also known and tractable in these datasets, we can compute the true conditional binding entropy $H ( O \mid F )$ ) exactly by averaging $H ( O \mid F = f )$ over feature realizations.

Computing $H ( O \mid F )$ for Visual Genome. We again approximate $H ( O \mid F )$ by enumerating binding configurations consistent with $F = f$ . Here, however, we cannot control for O being uniform, so we need to use the empirical dataset distribution of O as an approximation. We assume that each object instance occurs independently: this means the probability of a binding $O = o$ is the product of probabilities of each object instance in that binding assignment. The probability of each object instance (e.g. red car, blue balloon) is easily computed over the dataset. This then gives $H ( O \mid F = f )$ as well as $p ( f )$ , and after averaging over $f \sim p ( f )$ , yields $H ( O \mid F )$ .

# G. Details of datasets in Section 3.3

ColorShape (1 – 7 colors/shapes) We consider a ColorShape dataset with 7 colors and 7 shapes. The maximum number of objects is capped at $\lfloor N _ { 1 } N _ { 2 } / 2 \rfloor$ . We evaluate the model across all possible numbers of color–shape pairs. The number of training samples is scaled proportionally to $N _ { 1 } \cdot N _ { 2 }$ . The dataset consists of 48,000 samples.

Different from the ColorShape with 8 colors and 8 shapes, and to simplify calculation, we generate images X ∼ P (X | O) with O uniform over its allowed values in its space ΩO, i.e., every multi-object scene, determined by the object types in the scene, is equally likely.

CLEVR To get progressively closer to real-world scenarios that include more feature types, we evaluate binding on CLEVR (Johnson et al., 2016), for which we generate images with 3-D objects out of 8 colors, 3 shapes, 2 materials, and 2 sizes, with a total of 96 possible feature combinations. We illustrate the different occlusion levels of the CLEVR dataset in Figure 5, which are controlled by camera elevation and pitch angle. The dataset consists of 24,000 samples.

![](images/e256fad77f096f908d5b3ec41bc4f04cc90cb19b49232f8e5b4d017a82f4add3.jpg)  
Figure 5. The different occlusion levels of the CLEVR dataset, controlled by camera elevation and pitch angle. In left-to-right, top-to-bottom order, the camera elevation and pitch angles are (0.6, 6), (1.2, 12), (1.8, 18), (2.5, 25), (3.2, 32). As can be seen, in the most occluded scenario, the gray cube is almost entirely covered by the red cube, and the positioning of the green cylinder and green cube (as well as the blue cylinder and blue cube) can potentially create difficulty for binding due to their ambiguous boundaries. These issues are progressively resolved when there is less occlusion.

Visual Genome Visual Genome is a large-scale vision dataset with dense annotations of objects, attributes, and relationships. From it, we construct two subsets: VG:Color and VG:TopAttr. In VG:Color, we restrict attributes to color terms; in VG:TopAttr, we retain the most frequent (top) attributes. In both cases, we keep only images containing the selected object classes and attributes, resulting in approximately 50,000 samples. We select the 20 most frequent object classes and pair them with either 20 colors or 20 top attributes. Object classes and attributes are collapsed using WordNet synsets. Attributes that are not annotated for an object are treated as an explicit feature value, in addition to the annotated attributes.

![](images/7ddecd3fe0704c69e0da1c6bd7eceb90c829a29419293f3d14cf9267d2321894.jpg)  
Figure 6. VG:Color examples of object-code uncertainty revealed by binding probes.