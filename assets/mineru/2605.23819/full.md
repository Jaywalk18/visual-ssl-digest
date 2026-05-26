# Not Too Generative, Not Too Discriminative: The Human Alignment Sweet Spot

Jorge Chang Ortega ANITI

Bastien Le Lan ANITI

Thomas Serre∗ Brown University, ANITI

Victor Boutin∗CNRS, ANITI

# Abstract

A central question in computational vision is whether human-like visual representations are better explained by discriminative or generative learning. Existing comparisons, however, often confound the learning objective with architecture, scale, and training data, leaving open whether the objective itself drives alignment. We address this confound using Joint Energy-Based Models (JEMs), which interpolate continuously between discriminative and generative training within a fixed architecture. By varying a single mixing coefficient, we isolate the effect of the learning objective and evaluate the resulting models across six human-alignment benchmarks spanning perceptual similarity, gloss perception, human response uncertainty, robustness, shape–texture cue conflict, and diagnostic feature attribution. Across this diverse suite, human alignment is consistently maximized at intermediate points of the generative–discriminative continuum, rather than at either endpoint. Hybrid JEMs combine the categorical structure induced by discriminative learning with the sensitivity to input structure induced by generative learning, yielding more human-like behavior across multiple levels of vision.

These results suggest that the generative–discriminative dichotomy is the wrong axis for understanding human-aligned vision: alignment emerges not from choosing one objective over the other, but from balancing both.

# 1 Introduction

What computational principle underlies human-like visual representations—discriminative or generative learning? Discriminative models cast vision as a feedforward mapping from inputs to labels Riesenhuber and Poggio [1999], DiCarlo and Cox [2007]. Generative models take the opposite view—vision as inference under an internal world model, shaped by top-down priors Rao and Ballard [1999], Lee and Mumford [2003], Yuille and Kersten [2006], Peters et al. [2024]. In representation learning, this tension runs just as deep: discriminative training pushes representations toward category boundaries, while generative training pushes them toward the structure of the input distribution Ng and Jordan [2001]. Both traditions have claimed the crown at different levels of vision: discriminative training long dominated the modeling of high-level visual cortex Riesenhuber and Poggio [1999], DiCarlo and Cox [2007], while generative training has been argued to better capture mid-level perception, gloss, and shape-based behavior Rao and Ballard [1999], Lee and Mumford [2003], Yuille and Kersten [2006], Peters et al. [2024].

The debate, however, rests on a shaky comparison. Discriminative and generative models rarely differ only in their objective: they also differ in architecture, scale, and training data. Recent work makes this confound concrete: apparent advantages of generative models in shape bias and human alignment can be partly reproduced by simply low-pass filtering the inputs to a discriminative model Wolff et al. [2026]; training objective and dataset matter more than architecture for human similarity judgments Muttenthaler et al. [2023]; and latent diffusion models align with humans no better than standard ImageNet classifiers on odd-one-out tasks Linhardt et al. [2024]. What is still missing is a comparison in which the objective is varied in isolation. Until it is, the debate cannot be resolved.

Joint Energy-Based Models (JEMs) Grathwohl et al. [2020a] offer a principled way to resolve this debate. A JEM assigns an energy to each input–label pair, with lower energy meaning higher compatibility. The same network simultaneously supports discriminative prediction— the conditional $p ( y | x )$ — and generative density modeling—the marginal $p ( x )$ . A single mixing coefficient $\alpha \in [ 0 , 1 ]$ controls the balance between the two objectives, while architecture, scale, and data stay fixed. This gives us a clean axis along which to ask: where does human alignment live?

We train JEMs across this continuum and evaluate them on a battery of human–machine benchmarks. The benchmarks cover low-, mid-, and high-level vision: perceptual similarity, gloss perception, human uncertainty on ambiguous images, generalization under distribution shift, shape–texture cue conflict, and diagnostic feature attribution. Across all benchmarks, the answer is consistent: maximum human alignment is never achieved at the extreme values $\alpha = 0 . 0$ or $\alpha = 1 . 0$ (see Fig. 1). Hybrid JEMs match or

exceed discriminative baselines on perceptual similarity, surpass all baselines on gloss judgments, better reproduce human uncertainty and shape bias, and retain strong generalization under distribution shift. We further show that the energy landscape itself can be exploited at test time — refining inputs toward higher-probability images amplifies shape-consistent responses without any retraining. Taken together, these results reframe the long-standing debate. The question shifts from “which objective is best?” to “how should discriminative and generative objectives be balanced?” Human-aligned representations emerge not from either extreme, but from the regime in between.

![](images/030b529f6599045972999f153cfde03affabe69147303d7a3813123f432ff9e7.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Gloss perception\nRate the glossiness level (from 1 to 6)"] --> B["α = 0\np(y | x)"]
    C["Similarity perception\nWhich patch (left or right) is &quot;closer&quot; to the middle?"] --> D["0.1\n0.3\n0.5\n0.7\n0.9"]
    E["Response dist.\nClassify this among 10 categories"] --> D
    F["Diagnostic features\nWhat are the most important regions?"] --> D
    G["OOD generalization\nClassify under image distortions"] --> D
    H["Shape-texture bias\nIs that a cat or an elephant?"] --> D
    I["Toward best aligned JEM"] --> J["I"]
    K["Training datasets\nImageNet\nSynthetic\nCIFAR-10"] --> L["I"]
    M["α = 1\np(x)"] --> N["I"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
```
</details>

Figure 1. Human alignment peaks in the hybrid regime. JEMs are trained across the generative $( p ( x ) ) -$ （204号 discriminative $( p ( y | x ) )$ ) continuum by varying $\alpha \in [ 0 , 1 ]$ and evaluated on six human–machine comparison benchmarks. Arrows indicate the best-aligned α for each benchmark.

# 2 Related Work

# Human alignment of generative and discriminative models.

Deep networks have become influential models of human and biological vision, from hierarchical recognition models inspired by the ventral visual stream Riesenhuber and Poggio [1999], Kriegeskorte [2015] to supervised networks that predict primate neural responses and aspects of human behavior Khaligh-Razavi and Kriegeskorte [2014], Cadieu et al. [2014], Yamins et al. [2014], Yamins and DiCarlo [2016], Schrimpf et al. [2018]. However, high classification accuracy does not guarantee human-like perception: standard discriminative models can be fooled by imperceptible or unrecognizable patterns Szegedy et al. [2014], Nguyen et al. [2015], rely on local or texture-based cues Baker et al. [2018], Geirhos et al. [2018a], and diverge from humans under cue conflict and distribution shift Geirhos et al. [2020a, 2021], Bowers et al. [2023]. This has motivated benchmarks that compare models to humans beyond accuracy, including perceptual similarity Zhang et al. [2018], human response distributions Peterson et al. [2019], Battleday et al. [2020], Collins et al. [2022], shape– texture bias Geirhos et al. [2018a], Hermann et al. [2020], robustness to distribution shift Hendrycks and Dietterich [2019], Geirhos et al. [2021], gloss perception Storrs et al. [2021], and perceptual illusions Jaini et al. [2024]. Together, this work suggests a mixed picture. Discriminative models capture important aspects of recognition and neural predictability, whereas recent generative models have been reported to better capture some human-like perceptual biases, including gloss judgments, shape bias, human-like errors, and illusion sensitivity Storrs et al. [2021], Jaini et al. [2024]. Yet these comparisons typically contrast models that differ in objective, architecture, training data, scale, and inductive bias. As a result, it remains unclear whether human alignment stems from the learning objective itself or from correlated model properties.

# Hybrid generative–discriminative models.

Hybrid generative–discriminative learning aims to combine density modeling with category prediction Lasserre et al. [2006], Kingma et al. [2014], Maaløe et al. [2016]. However, many hybrid approaches introduce auxiliary inference networks, modify the architecture, or combine separately parameterized generative and discriminative components, making it difficult to isolate the effect of the learning objective. Energy-based models provide a cleaner testbed by defining a single energy function over input–label pairs LeCun et al. [2006], Du and Mordatch [2019]. In particular, Joint Energy-based Models (JEMs) reinterpret a standard classifier as a joint energy model, enabling both conditional classification and density modeling within the same architecture Grathwohl et al. [2020a]. We exploit this property to treat the generative–discriminative balance as an experimental variable, interpolating between the two objectives while holding architecture, scale, and data fixed. Additional related work is discussed in Appendix. A.

# 3 Method

# 3.1 Joint Energy-Based Models (JEMs)

Let $p _ { \mathcal { D } } ( \mathbf { x } , y )$ denote the unknown data distribution over input images $\mathbf { x } \in \mathbb { R } ^ { N }$ and class labels $y \in \{ 1 , \ldots , K \}$ .

Joint Energy-Based Models (JEMs) Grathwohl et al. [2020a] define a parametric model $p _ { \pmb { \theta } } ( \mathbf { x } , y )$ over the joint distribution of inputs and labels, where θ denotes the model parameters. They can therefore be viewed as generative classifiers: rather than modeling only $p _ { \pmb { \theta } } ( \pmb { y } \mid \mathbf { x } )$ , they model the joint distribution $p _ { \pmb { \theta } } ( \mathbf { x } , y )$ . Because the model is joint, its log-probability decomposes as

$$
\log p _ {\boldsymbol {\theta}} (\mathbf {x}, y) = \log p _ {\boldsymbol {\theta}} (y \mid \mathbf {x}) + \log p _ {\boldsymbol {\theta}} (\mathbf {x}) \tag {1}
$$

Eq. 1 reveals two complementary components of the joint log-likelihood: a discriminative term, log $p _ { \theta } ( y \mid \mathbf { x } )$ , that drives correct classification, and a generative term, log $p _ { \theta } ( \mathbf { x } )$ , that captures the structure of the input distribution.

A key property of JEMs is that both terms are implemented using the same classifier-like network. Let $\bar { f } _ { \theta } ( \mathbf { x } ) ^ { \mathbf { \bar { \alpha } } } \in \bar { \mathbb { R } } ^ { K }$ denote the logits produced by a neural network with parameters θ for an input x, and let $f _ { \boldsymbol { \theta } } ( \mathbf { x } ) [ y ]$ denote the logit associated with class y. JEMs reinterpret these logits by defining an energy function $E _ { \pmb { \theta } } ( \mathbf { x } , y )$ over input–label pairs

$$
p _ {\boldsymbol {\theta}} (\mathbf {x}, y) = \frac {\exp (- E _ {\boldsymbol {\theta}} (\mathbf {x} , y))}{Z (\boldsymbol {\theta})} \quad \text { with } \quad E _ {\boldsymbol {\theta}} (\mathbf {x}, y) = - \boldsymbol {f} _ {\boldsymbol {\theta}} (\mathbf {x}) [ y ] \tag {2}
$$

where $\begin{array} { r } { Z ( \pmb \theta ) = \sum _ { \boldsymbol { u } ^ { \prime } } \int \exp ( - E _ { \pmb \theta } ( \mathbf x ^ { \prime } , y ^ { \prime } ) ) d \mathbf x ^ { \prime } } \end{array}$ is the partition function. Because it requires summing over labels and integrating over all possible inputs x in a high-dimensional space, $Z ( \pmb \theta )$ is computationally intractable and cannot be evaluated exactly. Intuitively, lower energy indicates greater compatibility between input x and label y. From this joint distribution, we recover both the conditional distribution over labels and the marginal distribution over inputs:

$$
p _ {\boldsymbol {\theta}} (y \mid \mathbf {x}) = \frac {\exp (\boldsymbol {f} _ {\boldsymbol {\theta}} (\mathbf {x}) [ y ])}{\sum_ {y ^ {\prime}} \exp (\boldsymbol {f} _ {\boldsymbol {\theta}} (\mathbf {x}) [ y ^ {\prime} ])} \quad (3) \quad p _ {\boldsymbol {\theta}} (\mathbf {x}) = \frac {\sum_ {y} \exp (\boldsymbol {f} _ {\boldsymbol {\theta}} (\mathbf {x}) [ y ])}{Z (\boldsymbol {\theta})} \tag {4}
$$

Eq. 3 is exactly the usual softmax classifier, while Eq. 4 defines a density over inputs using the same logits. This is the key advantage of JEMs for our purpose: the same network, and therefore the same learned representation, supports both discriminative prediction and generative modeling. This lets us probe a shared representation shaped by both objectives, rather than comparing separately parameterized generative and discriminative models.

We train $p _ { \pmb { \theta } } ( \mathbf { x } , y )$ by maximizing the expected joint log-likelihood under the data distribution:

$$
\mathbb {E} _ {(\mathbf {x}, y) \sim p _ {\mathcal {D}}} [ \log p _ {\boldsymbol {\theta}} (\mathbf {x}, y) ] = \mathcal {L} _ {D} + \mathcal {L} _ {G}, \quad \left\{ \begin{array}{l} \mathcal {L} _ {D} = \mathbb {E} _ {(\mathbf {x}, y) \sim p _ {\mathcal {D}}} [ \log p _ {\boldsymbol {\theta}} (y \mid \mathbf {x}) ] \\ \mathcal {L} _ {G} = \mathbb {E} _ {\mathbf {x} \sim p _ {\mathcal {D}}} [ \log p _ {\boldsymbol {\theta}} (\mathbf {x}) ] \end{array} \right. \tag {5}
$$

In Eq. 5, $\mathcal { L } _ { D }$ is the discriminative log-likelihood, corresponding to the negative of the standard cross-entropy loss, while $\mathcal { L } _ { G }$ is the generative log-likelihood over inputs. In practice, $\mathcal { L } _ { G }$ is approximated via contrastive divergence Hinton [2002], Woodford [2006], Du and Mordatch [2019] (see Appendix B.1), which requires drawing samples from the marginal $p _ { \pmb { \theta } } ( \mathbf { x } )$ . Because the partition function $Z ( \pmb \theta )$ is intractable, direct sampling is infeasible. However, Eq. 4 reveals that $\begin{array} { r } { p _ { \pmb \theta } ( \mathbf x ) \propto \sum _ { y } \exp ( \pmb f _ { \pmb \theta } ( \mathbf x ) [ y ] ) } \end{array}$ , so the corresponding unnormalized marginal energy takes the simple form of a log-sum-exp over the logits:

$$
E _ {\boldsymbol {\theta}} (\mathbf {x}) = - \log \sum_ {y} \exp (\boldsymbol {f} _ {\boldsymbol {\theta}} (\mathbf {x}) [ y ]) \tag {6}
$$

This quantity is cheap to compute, differentiable w.r.t. both x and θ, and requires no architectural change: the same logits that define the softmax classifier also define the marginal energy. We draw approximate samples from $p _ { \pmb { \theta } } ( \mathbf { x } )$ by running Stochastic Gradient Langevin Dynamics (SGLD) Welling and Teh [2011] on $E _ { \pmb { \theta } } ( \mathbf { x } )$ . Starting from an initial sample $\mathbf { x } _ { \mathrm { 0 } }$ , SGLD iterates:

$$
\mathbf {x} _ {t + 1} = \mathbf {x} _ {t} - \frac {\eta}{2} \nabla_ {\mathbf {x}} E _ {\boldsymbol {\theta}} (\mathbf {x} _ {t}) + \omega , \quad \omega \sim \mathcal {N} (0, \eta \mathbf {I}) \tag {7}
$$

where η is the step size and ω is Gaussian noise, which turns gradient descent on the energy into an approximate sampling procedure rather than pure mode-seeking optimization. After $\bar { L }$ steps, the output $\mathbf { x } _ { L }$ serves as the negative sample for the contrastive-divergence update of $\mathcal { L } _ { G }$ Du and Mordatch [2019]. Beyond training, SGLD also provides a natural refinement mechanism: since the energy landscape encodes the model’s learned image prior, iterating SGLD from an arbitrary input tends to move it toward lower-energy, higher-probability regions under the model. We exploit this property at test time in Section 4.4, where we use it to amplify shape-consistent responses without any retraining.

Finally, the joint decomposition in Eq. 5 provides the basis for our main experimental manipulation: we use the mixing coefficient α to interpolate between discriminative and generative training.

# 3.2 Interpolating the Generative–Discriminative Trade-off

To control the balance between discriminative and generative learning, we maximize the following interpolated objective:

$$
\mathcal {L} = \alpha \cdot \frac {\left\| \nabla_ {\boldsymbol {\theta}} \mathcal {L} _ {D} \right\|}{\left\| \nabla_ {\boldsymbol {\theta}} \mathcal {L} _ {G} \right\|} \cdot \mathcal {L} _ {G} + (1 - \alpha) \cdot \mathcal {L} _ {D} \tag {8}
$$

Here, $\alpha \in [ 0 , 1 ]$ controls the balance between the two terms: $\alpha = 0$ recovers purely discriminative training, while $\alpha = 1$ recovers purely generative training. The generative term is rescaled by the ratio of gradient norms, treated as a stop-gradient scalar, so that both objectives contribute at a comparable scale during training. Without this correction, the raw magnitudes of $\| \nabla _ { \pmb { \theta } } \mathcal { L } _ { D } \|$ and $\| \nabla _ { \pmb { \theta } } \mathcal { L } _ { G } \|$ can differ by orders of magnitude—particularly at intermediate α—causing α to behave non-linearly as an interpolation parameter and destabilizing hybrid training. Throughout this paper, JEMs trained with different values of α are represented by a continuous color gradient, ranging from for fully discriminative JEMs to for fully generative JEMs. We use a Wide ResNet architecture Zagoruyko and Komodakis [2016], following the original JEM implementation Grathwohl et al. [2020a]. Training 11 JEMs across the full continuum is computationally intensive, particularly at ImageNet scale where we operate in the latent space of a pretrained autoencoder to keep hybrid training feasible. Full training details, compute requirements, and hyperparameter choices are provided in Appendix B, and reproducible code is available at https://anonymous.4open.science/r/JEM-0B78/.

# 3.3 Human–Machine Comparison Tasks

To evaluate how the generative–discriminative trade-off affects human alignment, we use a diverse set of benchmarks spanning four complementary aspects of human visual behavior: low- and mid-level perceptual judgments, response distributions under uncertainty, generalization and cue use, and diagnostic feature attribution. This breadth is important because human alignment is a multidimensional property: models may match human perception, uncertainty, and diagnostic features to different degrees, and we ask whether the same generative–discriminative balance improves alignment across all of them. We selected benchmarks with established, publicly available protocols, well-defined human reference points, and deep-learning baselines against which JEMs can be quantitatively compared.

# Low- and mid-level perceptual judgments.

For low-level perception, we use the perceptual similarity benchmark of Zhang et al. [2018], based on the BAPPS dataset of distorted image patches. It includes two tasks. In the two-alternative forced choice (2AFC) task, observers choose which of two distorted patches is more similar to a reference. In the just noticeable difference (JND) task, they judge whether a distorted patch is perceptually distinguishable from the reference. Alignment is measured by 2AFC Test (%) and JND Test (mAP), respectively, with higher values indicating better alignment. Human consistency on the 2AFC task provides an empirical ceiling of 83% Zhang et al. [2018]; the metric maximum of 100% would imply perfect agreement, which is not observed in practice.

For mid-level perception, we use the gloss perception benchmark of Storrs et al. [2021], based on controlled synthetic images in which illumination and surface geometry are systematically varied as nuisance factors, so that surface reflectance is the only reliable cue for a correct gloss judgment. Alignment is measured by the Pearson correlation between human and model responses. Higher values indicate better agreement, with r = 1 as the theoretical upper bound of the metric rather than an empirically measured human ceiling. See Appendix C and Appendix D for more details on the low- and mid-level perceptual benchmarks, respectively.

# Human classification response distributions.

Unlike benchmarks that reduce alignment to a single accuracy score, CIFAR-10H Peterson et al. [2019] captures graded uncertainty across observers, allowing us to test whether models reproduce the full distribution of human responses rather than just the modal one. Alignment is measured by cross-entropy between model and human response distributions, where lower values indicate better alignment. The cross-entropy between independent groups of human observers (0.55 nats) serves as an empirical ceiling, reflecting irreducible inter-observer disagreement. See Appendix E for more details.

# Generalization and representational bias.

We next evaluate whether models generalize and rely on visual cues in a human-like way. For generalization, we use the ImageNet-based benchmark of Geirhos et al. [2018b], which compares human and model classification accuracy under a range of out-of-distribution image transformations. We report error consistency with human responses (Cohen’s κ, chance-corrected), with higher values indicating better alignment and a human inter-rater ceiling of approximately κ = 0.39 Geirhos et al. [2021].

For cue use, we use the shape–texture conflict benchmark of Geirhos et al. [2018a], in which each image combines the global shape of one object category with the local texture of another. A model’s classification reveals which cue dominates its decision. We report shape bias, defined as the proportion of cue-conflict images classified according to shape rather than texture, with values closer to the human reference of approximately 96% indicating more human-like cue use Geirhos et al. [2018a]. Together, these benchmarks test whether the generative–discriminative trade-off affects the model’s learned invariances and visual features. See Appendix F and Appendix G for details.

# Human diagnostic features.

Finally, we use the ClickMe dataset Linsley et al. [2018] to test whether models rely on the same diagnostic features as humans during ImageNet object recognition. Participants clicked on image regions they found informative for recognizing the object, yielding spatial importance maps that reflect human visual strategies. We compare model attribution maps against these human maps using the mean Spearman correlation normalized by human inter-rater agreement Fel et al. [2022a], with higher values indicating better alignment and a normalized score of 1 corresponding to the human ceiling. This benchmark is uniquely diagnostic: rather than measuring what the model predicts, it measures which visual evidence it uses, directly probing the spatial structure of learned representations. See Appendix H for more details.

The JEM family evaluated on each benchmark matches its underlying visual domain. Specifically, we use ImageNet-trained JEMs for perceptual similarity (BAPPS), human-like generalization, shape– texture cue-conflict, and ClickMe; CIFAR-10-trained JEMs for CIFAR-10H; and synthetic-datatrained JEMs for gloss perception. This ensures that comparisons probe the effect of the generative– discriminative trade-off within an appropriate training domain, rather than across mismatched datasets. See Appendix B.2 for full training details.

a) Low-level perceptual similarity   
![](images/2f958913d426a00f3c70f47d16d7ac8770b22f46a2919198db62bdc71ccb9bdd.jpg)

<details>
<summary>scatter</summary>

| Model   | JND test [mAP %] | 2AFC test [score %] |
|---------|------------------|---------------------|
| VAE     | 53               | 71                  |
| BiGAN   | 54.5             | 76.5                |
| Alex    | 55.5             | 75.0                |
| Squeeze | 57.5             | 77                  |
| VGG     | 56.5             | 76.0                |
| 1.0     | 55.5             | 74.5                |
</details>

b) Mid-level gloss perception   
![](images/81efbc3d89fde50c451877360b859c146a7e73fa87148d422cf6084e724ef273.jpg)

c) Response uncertainty in classification   
![](images/c3bcbf0e8dbe883338ae6befbcab0c9fcaacdb75374ebe13fa452f607e161fb8.jpg)

<details>
<summary>scatter</summary>

| Model     | CIFAR-10 accuracy [%] | CIFAR-10H cross-entropy |
|-----------|------------------------|--------------------------|
| ResNet    | 0.85                   | 0.6                      |
| VGG       | 0.9                    | 1.0                      |
| ResNeXt   | 0.9                    | 1.2                      |
| ConvNeXt  | 0.9                    | 1.4                      |
</details>

d) Robustness to transformation   
![](images/64cd5662cdf4e8d1f51693ff5b55219c1872c2ba4e0c7269fd78e21c6e556f5c.jpg)

<details>
<summary>scatter</summary>

| Model  | OOD accuracy [%] | Error consistency (k) |
|--------|------------------|------------------------|
| VAE    | 10               | 0.0                    |
| VAE    | 10               | 0.0                    |
| VAE    | 25               | 0.08                   |
| VAE    | 30               | 0.1                    |
| VAE    | 35               | 0.15                   |
| VAE    | 40               | 0.1                    |
| VAE    | 45               | 0.1                    |
| VAE    | 50               | 0.1                    |
| VAE    | 55               | 0.1                    |
| VAE    | 60               | 0.1                    |
| VAE    | 65               | 0.1                    |
| VAE    | 70               | 0.1                    |
| Alex   | 40               | 0.18                   |
| Alex   | 45               | 0.12                   |
| Alex   | 50               | 0.1                    |
| ResNet | 45               | 0.1                    |
| ResNet | 50               | 0.1                    |
| ResNet | 55               | 0.1                    |
| ResNet | 60               | 0.1                    |
| ResNet | 65               | 0.1                    |
| ResNet | 70               | 0.1                    |
| Imagen | 70               | 0.3                    |
| Imagen | 75               | 0.3                    |
| Imagen | 80               | 0.3                    |
| Imagen | 85               | 0.3                    |
| Imagen | 90               | 0.3                    |
| Imagen | 95               | 0.3                    |
| Imagen | 100              | 0.3                    |
| SD     | 70               | 0.28                   |
| SD     | 75               | 0.28                   |
| SD     | 80               | 0.28                   |
| SD     | 85               | 0.28                   |
| SD     | 90               | 0.28                   |
| SD     | 95               | 0.28                   |
| SD     | 100              | 0.28                   |
</details>

e) Shape-texture cue-conflict   
![](images/bd695c05e692ed0160c226d3f75c572d28aeba4bdfe02c200dc5d87633d6f4f0.jpg)

f) Human diagnostic features   
![](images/79a6892c89aaee246f0d5aab9b416900fab08522d32d1dbeafbc9a4c42b75254.jpg)

<details>
<summary>scatter</summary>

| Model       | ImageNet accuracy [%] | ClickMe alignment score |
|-------------|------------------------|--------------------------|
| VAE         | 5                      | 0.15                     |
| 1.0         | 10                     | 0.07                     |
| VGG         | 45                     | 0.24                     |
| EfficientNet| 60                     | 0.28                     |
| ViT         | 65                     | 0.35                     |
| ResNet      | 65                     | 0.37                     |
</details>

![](images/bec3f42be3b5ad51d3c667d6087e4dd45ecd34b540985a0b9009684200a3d9c1.jpg)

<details>
<summary>text_image</summary>

Generative
baselines ▶ Discriminative
baselines ★ Humans ● JEMs ● Most aligned
JEMs α
0 0.5 1
</details>

Figure 2. Human alignment across the generative–discriminative continuum. JEMs are evaluated across $\alpha \in [ 0 , 1 ]$ , from purely discriminative $( \alpha = 0 )$ to purely generative (α = 1). (a) Low-level perceptual similarity on BAPPS (JND mAP and 2AFC; human ceiling: 83% Zhang et al. [2018]). (b) Mid-level gloss perception (gloss accuracy and Pearson correlation with human judgments; theoretical upper bound: $r = 1 )$ . (c) CIFAR-10H response uncertainty (CIFAR-10 accuracy and cross-entropy with human distributions, lower is better; human ceiling: 0.55 nats). (d) Generalization under out-of-distribution transformations (OOD accuracy and error consistency with humans, Cohen’s $\kappa ;$ human ceiling: $\kappa = 0 . 3 9$ Geirhos et al. [2021]). (e) Shape–texture cue conflict (OOD accuracy and shape bias; human reference: ≈ 96% Geirhos et al. [2018a]). (f) Diagnostic feature alignment on ClickMe (Spearman correlation normalized by human inter-rater agreement Fel et al. [2022a]; human ceiling: 1 by construction). Across all benchmarks, purely discriminative and purely generative JEMs are never the most human-aligned, confirming that hybrid objectives better capture human visual behavior.

# 4 Results

# 4.1 A consistent sweet spot across six benchmarks (Fig. 2)

Fig. 2 plots JEMs across the full continuum of α (ranging from generative to discriminative) against a distinct alignment benchmark. Despite the diversity of tasks — spanning low-level image perception, mid-level material perception, classification uncertainty, robustness, cue use, and feature attribution—a single pattern recurs throughout: the most human-aligned models consistently cluster at intermediate values (often around $\alpha = 0 . 5 )$ , away from both endpoints.

For each benchmark, we also report the established discriminative and generative baselines from the original evaluation protocol, so that JEMs are compared against the relevant reference models rather than a single arbitrary baseline family. Across benchmarks, the best-aligned JEM consistently falls at intermediate α—where alignment is measured as the distance to the human reference in each benchmark’s metric space. The optimal value ranges from $\alpha \approx 0 . 3$ (diagnostic features, see Fig. 2f) to $\alpha \approx 0 . 8$ (response uncertainty, see Fig. 2c), with most tasks clustering near $\alpha \in [ 0 . 5 , 0 . 6 ]$ (see Fig 2a-b-d-e). The remainder of this section examines each benchmark in detail. To rule out the VAE encoder itself as a confound in Imagenet-based benchmarks (Fig 2a-d-e-f), we include a baseline that uses the same frozen pretrained VAE as our JEMs—but without any JEM training on top. This baseline consistently underperforms hybrid JEMs, confirming that the alignment gains reflect the learned objective rather than the latent representation.

# 4.2 Perceptual alignment peaks in hybrid regimes (Fig. 2a-b)

We first examine whether the intermediate optimum holds at the perceptual level, where existing comparisons between generative and discriminative models have produced conflicting conclusions. For low-level perceptual similarity on BAPPS (Fig. 2a), prior work suggested that discriminative models tend to outperform generative ones, especially on the JND task (e.g., SqueezeNet Iandola et al. [2016] and AlexNet Krizhevsky et al. [2012] vs. BiGAN Brock et al. [2019] and VAE Kingma and Welling [2014]). However, the JEM continuum reveals a richer picture — a hybrid model at $\alpha = 0 . 6$ nearly matches the best discriminative baseline on JND (57.5% vs. 58.4%) while simultaneously achieving the highest 2AFC score, a combination no purely discriminative or generative model achieves. This indicates that the joint pressure to both classify and model the input distribution produces low-level features that are more perceptually calibrated than either objective alone.

For mid-level gloss perception (Fig. 2b), the pattern is even more informative because it directly revisits a published conclusion. Storrs et al. [2021] found that an unsupervised PixelVAE Zhao et al. [2017] outperformed a supervised ResNet He et al. [2016] in predicting human gloss judgments, taken as evidence for the superiority of generative learning at this level. Using JEMs to control for architecture, we obtain a different result: the purely discriminative JEM (α = 0) already outperforms the purely generative JEM (α = 1), while adding a small generative contribution improves alignment further. Crucially, the best hybrid JEM $( \alpha = 0 . 5 )$ surpasses both the PixelVAE and ResNet baselines from Storrs et al. [2021]. It suggests that the originally reported advantage of generative models was confounded by architectural differences rather than the learning objective per se. Within a controlled architecture, even a small dose of generative pressure is beneficial—but the optimum remains consistently in hybrid territory.

# 4.3 Hybrid objectives better capture human response uncertainty (Fig. 2c)

Having established the intermediate optimum at low and mid levels, we ask whether it extends to the full distribution of human responses on ambiguous images. Fig. 2c shows CIFAR-10H results, where alignment is measured by cross-entropy between the model’s output distribution and the empirical distribution of human labels. The lowest cross-entropy—reflecting the closest match to human uncertainty—is obtained at $\alpha = 0 . 8$ . Purely discriminative models capture the dominant label but under-represent graded uncertainty, while purely generative models recover some distributional spread at the cost of weakened category structure. Hybrid JEMs strike a balance between these failure modes: they preserve enough categorical structure to agree with the modal human response while remaining sensitive to the ambiguity that leads to disagreement across observers. Notably, this alignment emerges without any direct supervision on human soft labels—unlike Peterson et al.

[2019], who fine-tune on CIFAR-10H soft labels directly—suggesting it is a natural consequence of the right generative–discriminative balance.

# 4.4 Hybrid objectives promote shape-based generalization (Fig. 2d-e and Fig. 3)

We next turn to higher-level visual behavior, where the generative–discriminative trade-off has a mechanistically interpretable effect. Fig. 2d shows that hybrid JEMs improve error consistency with humans on out-of-distribution (OOD) transformations relative to both endpoints of the continuum. However, they remain below large-scale generative baselines such as Imagen Saharia et al. [2022] and Stable Diffusion Rombach et al. [2022], which benefit from broader training data and language conditioning. Within the JEM family, the improvement is consistent and monotonically ordered around the intermediate regime.

The shape–texture cue-conflict benchmark (Fig. 2e) reveals a particularly clean effect. As α increases, shape bias increases monotonically across the JEM continuum: the best hybrid JEMs reach a shape bias close to 0.7, well above AlexNet and ResNet-50, and approach the level of shape use seen in large generative models. We attribute this to an asymmetry between the two objectives: a discriminative model can reduce its loss by relying on locally predictive texture cues, whereas a generative model is pressured to assign high probability to globally coherent images, for which shape structure is often a dominant organizing cue. As α increases, the energy landscape increasingly penalizes globally incoherent inputs, gradually shifting learned representations toward global shape.

![](images/108f060e5e92670e34b3899d35030659c474798274172a8ba445252d5e085270.jpg)

<details>
<summary>text_image</summary>

a)
cue-conflict
α
0.0	0.1	0.3	0.5	0.7	0.9
</details>

![](images/fff8d5ac19e35af2346440d974e2c11ee94f2ef6353df55459f14f087812e44a.jpg)

<details>
<summary>bar</summary>

| α    | Number of SGLD steps 0 | Number of SGLD steps 5 | Number of SGLD steps 10 | Number of SGLD steps 20 |
| ---- | ---------------------- | ---------------------- | ----------------------- | ----------------------- |
| 0.0  | 0.5                    | 0.0                    | 0.0                     | 0.0                     |
| 0.1  | 0.45                   | 0.0                    | 0.0                     | 0.0                     |
| 0.3  | 0.45                   | 0.1                    | 0.0                     | 0.0                     |
| 0.5  | 0.45                   | 0.2                    | 0.1                     | 0.1                     |
| 0.7  | 0.45                   | 0.3                    | 0.2                     | 0.2                     |
| 0.9  | 0.45                   | 0.3                    | 0.2                     | 0.2                     |
</details>

Figure 3. Generative pressure reveals shape bias. a) Visualization of a cue-conflict image under the generative component of JEMs trained with different α values; increasing α shifts the visualization from texture-consistent toward shape-consistent. b) Shape bias across SGLD steps for each α. SGLD increases shape bias in hybrid/generative JEMs, indicating shape-favoring energy landscapes. The $\alpha = 1$ endpoint is omitted because the purely generative model is not trained for categorization, making shape-bias scores unreliable.

This mechanism can be exploited at test time via SGLD refinement (Fig. 3). Applying Langevin dynamics to a cue-conflict image moves it toward lower-energy regions of the model’s energy landscape. For high-α models, this favors the shape-consistent interpretation: local, high-frequency texture cues are attenuated, while globally coherent shape structure is preserved. Fig. 3 shows this effect both qualitatively and quantitatively: visualizations of the generative component shift toward the shape-consistent interpretation as α increases, and SGLD refinement amplifies shape bias for hybrid and generative JEMs $( \alpha \geq 0 . 3 )$ , with negligible effect at $\alpha = 0$ . Thus, SGLD exposes at inference the representational preference induced by the generative objective, without any retraining.

# 4.5 Hybrid objectives align model attention with human diagnostic regions (Fig. 2f and Fig. 4)

Finally, we ask whether the generative–discriminative balance affects not just what the model predicts, but which visual evidence it uses to do so. Using the ClickMe benchmark Linsley et al. [2019], Fel et al. [2022b], we compare model attribution maps against human importance maps collected during object recognition. The best alignment is obtained at $\alpha = 0 . 3$ , with both purely generative and purely discriminative models performing worse (see Fig. 2f). The progression across α in Fig. 4 is visually interpretable. $\mathrm { \bf A t } \alpha = 0$ , attribution maps are diffuse and fragmented; at $\alpha = 0 . 3$ , they sharpen around object-relevant regions that humans also mark as diagnostic. Beyond this hybrid regime, maps lose selectivity, consistent with weakened discriminative pressure in overly generative models. This non-monotonic pattern in Fig. 2f mirrors the shape–texture benchmark: a moderate generative contribution improves alignment, but too much erodes the categorical structure that makes features diagnostic. Thus, the generative objective shapes not only model predictions, but also the spatial structure of the representations that support them.

![](images/71b54b8f87d88c806ab0e4acc967cd0cfc0fdf9d2fac946425b61e1a29265044.jpg)

<details>
<summary>text_image</summary>

Original
image
Click-Me
Maps
α
0.0	0.1	0.3	0.5	0.7	0.9
</details>

Figure 4. Hybrid JEMs align with human saliency. Original images are shown on the left, followed by human ClickMe maps and model attribution maps for JEMs trained across the generative–discriminative continuum. As the generative contribution increases up to intermediate values, attribution maps become more concentrated on object-relevant regions and better resemble human diagnostic regions. Beyond this hybrid regime, maps become less focused, suggesting that excessive generative weighting weakens diagnostic feature selectivity.

Taken together, no benchmark favors either endpoint, suggesting that human-like visual representations depend not on choosing between generative and discriminative learning, but on balancing them.

# 5 Conclusion & Discussion

This paper asked whether human-like visual representations are better explained by generative or discriminative learning. Using JEMs as a controlled interpolation framework, we found that this framing is incomplete: across six qualitatively distinct benchmarks, human alignment is consistently maximized at intermediate values of α.

This result reframes the usual opposition between discriminative and generative accounts of vision. Discriminative learning provides categorical structure, whereas generative learning provides sensitivity to the input distribution. Human-like vision appears to require both. This is consistent with neuroscientific theories proposing that the brain combines bottom-up discriminative recognition with top-down generative prediction—as in predictive coding Rao and Ballard [1999] and analysis-bysynthesis Yuille and Kersten [2006]. If this hybrid computation is central to human vision Peters et al. [2024], our results offer a machine-learning analog: models become more human-like when trained to balance discriminative recognition with generative modeling. A natural next step is to test whether the same hybrid regimes better predict neural responses and dynamics across the visual hierarchy.

Our study has several limitations. JEMs provide a clean testbed, but energy-based training remains computationally demanding and requires careful stabilization. Extending this analysis to modern architectures, larger datasets, and more stable large-scale generative classifiers will be important. Moreover, the optimal value of α varies across benchmarks, suggesting that human alignment is not a single scalar property but a family of constraints spanning perception, uncertainty, generalization, and feature use.

The systematic shift in optimal α across benchmarks is itself informative. We propose that it reflects a gradient from category-level to input-level processing demands: diagnostic feature alignment peaks in a more discriminative regime $( \alpha \approx 0 . 3 )$ because humans selectively attend to features that maximally separate categories, a pressure best captured by the conditional $p ( y \mid x )$ ; response uncertainty peaks in a more generative regime (α ≈ 0.8) because human ambiguity tracks the statistical atypicality of images relative to natural image manifolds, a property encoded by $p ( x )$ ; and mid-level tasks such as gloss perception and perceptual similarity reside near the center, requiring both input-structure preservation and categorical abstraction. These interpretations are post-hoc and await direct empirical test, but they suggest a principled basis for the variation: the relative weight of generative worldmodeling versus discriminative labeling should scale with the degree to which a task depends on input-level rather than category-level information.

Finally, these results have broader implications for human-aligned AI. Hybrid objectives may improve alignment with human perception, uncertainty, and diagnostic feature use, but human-likeness can also reproduce human errors, biases, and overconfident interpretations. Human alignment should therefore be treated as a measurable design objective, not as a guarantee of fairness, safety, or robustness. Progress may require moving beyond the generative-versus-discriminative dichotomy entirely, toward understanding how perceptual systems structure the interaction between recognition and density modeling.

# References

Maximilian Riesenhuber and Tomaso Poggio. Hierarchical models of object recognition in cortex. Nature neuroscience, 2(11):1019–1025, 1999.   
James J. DiCarlo and David D. Cox. Untangling invariant object recognition. Trends in Cognitive Sciences, 11 (8):333–341, 2007. doi: 10.1016/j.tics.2007.06.010.   
Rajesh PN Rao and Dana H Ballard. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature neuroscience, 2(1):79–87, 1999.   
Tai Sing Lee and David Mumford. Hierarchical bayesian inference in the visual cortex. Journal of the Optical Society of America A, 20(7):1434–1448, 2003.   
Alan Yuille and Daniel Kersten. Vision as bayesian inference: analysis by synthesis? Trends in cognitive sciences, 10(7):301–308, 2006.   
Benjamin Peters, James J DiCarlo, Todd Gureckis, Ralf Haefner, Leyla Isik, Joshua Tenenbaum, Talia Konkle, Thomas Naselaris, Kimberly Stachenfeld, Zenna Tavares, et al. How does the primate brain combine generative and discriminative computations in vision? ArXiv, pages arXiv–2401, 2024.   
Andrew Ng and Michael Jordan. On discriminative vs. generative classifiers: A comparison of logistic regression and naive bayes. Advances in neural information processing systems, 14, 2001.   
Max Wolff, Thomas Klein, Evgenia Rusak, Felix Wichmann, and Wieland Brendel. Low-pass filtering improves behavioral alignment of vision models. arXiv preprint arXiv:2602.13859, 2026. doi: 10.48550/arXiv.2602. 13859. URL https://arxiv.org/abs/2602.13859.   
Lukas Muttenthaler, Jonas Dippel, Lorenz Linhardt, Robert A. Vandermeulen, and Simon Kornblith. Human alignment of neural network representations. In International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=ReDQ1OUQR0X.   
Lorenz Linhardt, Marco Morik, Sidney Bender, and Naima Elosegui Borras. An analysis of human alignment of latent diffusion models. In ICLR 2024 Workshop on Representational Alignment, 2024. doi: 10.48550/arXiv. 2403.08469. URL https://arxiv.org/abs/2403.08469.   
Will Grathwohl, Kuan-Chieh Wang, Jörn-Henrik Jacobsen, David Duvenaud, Mohammad Norouzi, and Kevin Swersky. Your classifier is secretly an energy based model and you should treat it like one. ICLR, 2020a.   
Nikolaus Kriegeskorte. Deep neural networks: A new framework for modeling biological vision and brain information processing. Annual Review of Vision Science, 1(1):417–446, 2015. doi: 10.1146/ annurev-vision-082114-035447.   
Seyed-Mahdi Khaligh-Razavi and Nikolaus Kriegeskorte. Deep supervised, but not unsupervised, models may explain it cortical representation. PLoS computational biology, 10(11):e1003915, 2014.   
Charles F Cadieu, Ha Hong, Daniel LK Yamins, Nicolas Pinto, Diego Ardila, Ethan A Solomon, Najib J Majaj, and James J DiCarlo. Deep neural networks rival the representation of primate it cortex for core visual object recognition. PLoS computational biology, 10(12):e1003963, 2014.   
Daniel L. K. Yamins, Ha Hong, Charles F. Cadieu, Ethan A. Solomon, Darren Seibert, and James J. DiCarlo. Performance-optimized hierarchical models predict neural responses in higher visual cortex. Proceedings of the National Academy of Sciences, 111(23):8619–8624, 2014. doi: 10.1073/pnas.1403112111.   
Daniel L. K. Yamins and James J. DiCarlo. Using goal-driven deep learning models to understand sensory cortex. Nature Neuroscience, 19(3):356–365, 2016. doi: 10.1038/nn.4244.   
Martin Schrimpf, Jonas Kubilius, Ha Hong, Najib J. Majaj, Rishi Rajalingham, Elias B. Issa, Kohitij Kar, Pouya Bashivan, Jonathan Prescott-Roy, Kailyn Schmidt, Daniel L. K. Yamins, and James J. DiCarlo. Brain-score: Which artificial neural network for object recognition is most brain-like? bioRxiv, 2018. doi: 10.1101/407007.   
Christian Szegedy, Wojciech Zaremba, Ilya Sutskever, Joan Bruna, Dumitru Erhan, Ian Goodfellow, and Rob Fergus. Intriguing properties of neural networks. In International Conference on Learning Representations, 2014. URL https://arxiv.org/abs/1312.6199.   
Anh Nguyen, Jason Yosinski, and Jeff Clune. Deep neural networks are easily fooled: High confidence predictions for unrecognizable images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 427–436, 2015. doi: 10.1109/CVPR.2015.7298640.

Nicholas Baker, Hongjing Lu, Gennady Erlikhman, and Philip J. Kellman. Deep convolutional networks do not classify based on global object shape. PLOS Computational Biology, 14(12):e1006613, 2018. doi: 10.1371/journal.pcbi.1006613.   
Robert Geirhos, Patricia Rubisch, Claudio Michaelis, Matthias Bethge, Felix A Wichmann, and Wieland Brendel. Imagenet-trained cnns are biased towards texture; increasing shape bias improves accuracy and robustness. In International conference on learning representations, 2018a.   
Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2:665–673, 2020a. doi: 10.1038/s42256-020-00257-z.   
Robert Geirhos, Kantharaju Narayanappa, Benjamin Mitzkus, Tizian Thieringer, Matthias Bethge, Felix A. Wichmann, and Wieland Brendel. Partial success in closing the gap between human and machine vision. In Advances in Neural Information Processing Systems, volume 34, 2021. URL https://openreview.net/ forum?id=QkljT4mrfs.   
Jeffrey S. Bowers, Gaurav Malhotra, Marin Dujmovic, Milton Llera Montero, Christian Tsvetkov, Valerio ´ Biscione, Guillermo Puebla, Federico Adolfi, John E. Hummel, Rachel F. Heaton, Benjamin D. Evans, Jeffrey Mitchell, and Ryan Blything. Deep problems with neural network models of human vision. Behavioral and Brain Sciences, 46:e385, 2023. doi: 10.1017/S0140525X22002813.   
Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable effectiveness of deep features as a perceptual metric. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 586–595, 2018.   
Joshua C Peterson, Ruairidh M Battleday, Thomas L Griffiths, and Olga Russakovsky. Human uncertainty makes classification more robust. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9617–9626, 2019.   
Ruairidh M. Battleday, Joshua C. Peterson, and Thomas L. Griffiths. Capturing human categorization of natural images by combining deep networks and cognitive models. Nature Communications, 11(1):5418, 2020. doi: 10.1038/s41467-020-18946-z.   
Katherine M. Collins, Umang Bhatt, and Adrian Weller. Eliciting and learning with soft labels from every annotator. In Proceedings of the AAAI Conference on Human Computation and Crowdsourcing, volume 10, pages 40–52, 2022.   
Katherine Hermann, Ting Chen, and Simon Kornblith. The origins and prevalence of texture bias in convolutional neural networks. In Advances in Neural Information Processing Systems, volume 33, 2020.   
Dan Hendrycks and Thomas Dietterich. Benchmarking neural network robustness to common corruptions and perturbations. In International Conference on Learning Representations, 2019. URL https://arxiv.org/ abs/1903.12261.   
Katherine R Storrs, Barton L Anderson, and Roland W Fleming. Unsupervised learning predicts human perception and misperception of gloss. Nature human behaviour, 5(10):1402–1417, 2021.   
Priyank Jaini, Kevin Clark, and Robert Geirhos. Intriguing properties of generative classifiers. In International Conference on Learning Representations, 2024.   
Julia A. Lasserre, Christopher M. Bishop, and Tom P. Minka. Principled hybrids of generative and discriminative models. In 2006 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR’06), pages 87–94, 2006. doi: 10.1109/CVPR.2006.227.   
Diederik P. Kingma, Shakir Mohamed, Danilo Jimenez Rezende, and Max Welling. Semi-supervised learning with deep generative models. In Advances in Neural Information Processing Systems, volume 27, 2014.   
Lars Maaløe, Casper Kaae Sønderby, Søren Kaae Sønderby, and Ole Winther. Auxiliary deep generative models. In Proceedings of the 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 1445–1453. PMLR, 2016.   
Yann LeCun, Sumit Chopra, Raia Hadsell, Marc’Aurelio Ranzato, and Fu Jie Huang. A tutorial on energy-based learning. In Gökhan Bakir, Thomas Hofmann, Bernhard Schölkopf, Alexander J. Smola, Ben Taskar, and S. V. N. Vishwanathan, editors, Predicting Structured Data. MIT Press, 2006.   
Yilun Du and Igor Mordatch. Implicit generation and modeling with energy based models. Advances in neural information processing systems, 32, 2019.

Geoffrey E Hinton. Training products of experts by minimizing contrastive divergence. Neural computation, 14 (8):1771–1800, 2002.   
Oliver Woodford. Notes on contrastive divergence. Department of Engineering Science, University of Oxford, Tech. Rep, 4, 2006.   
Max Welling and Yee W Teh. Bayesian learning via stochastic gradient langevin dynamics. In Proceedings of the 28th international conference on machine learning (ICML-11), pages 681–688, 2011.   
Sergey Zagoruyko and Nikos Komodakis. Wide residual networks. arXiv preprint arXiv:1605.07146, 2016.   
Robert Geirhos, Carlos R. Medina Temme, Jonas Rauber, Heiko H. Schütt, Matthias Bethge, and Felix A. Wichmann. Generalisation in humans and deep neural networks. In Advances in Neural Information Processing Systems, volume 31, 2018b.   
Drew Linsley, Dan Shiebler, Sven Eberhardt, and Thomas Serre. Learning what and where to attend. arXiv preprint arXiv:1805.08819, 2018.   
Thomas Fel, Ivan F Rodriguez Rodriguez, Drew Linsley, and Thomas Serre. Harmonizing the object recognition strategies of deep neural networks with humans. Advances in neural information processing systems, 35: 9432–9446, 2022a.   
Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf, William J. Dally, and Kurt Keutzer. Squeezenet: Alexnet-level accuracy with 50x fewer parameters and <0.5mb model size. arXiv preprint arXiv:1602.07360, 2016.   
Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. NeurIPS, 2012.   
Andrew Brock, Jeff Donahue, and Karen Simonyan. Large scale gan training for high fidelity natural image synthesis. In International Conference on Learning Representations, 2019.   
Diederik P. Kingma and Max Welling. Auto-encoding variational bayes. International Conference on Learning Representations, 2014.   
Shengjia Zhao, Jiaming Song, and Stefano Ermon. Towards deeper understanding of variational autoencoding models. arXiv preprint arXiv:1702.08658, 2017.   
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.   
Chitwan Saharia, William Chan, Saurabh Saxena, Lala Li, Jay Whang, Emily L. Denton, Seyed Kamyar Seyed Ghasemipour, Raphael Gontijo Lopes, Burcu Karagol Ayan, Tim Salimans, Jonathan Ho, David J. Fleet, and Mohammad Norouzi. Photorealistic text-to-image diffusion models with deep language understanding. In Advances in Neural Information Processing Systems, volume 35, pages 36479–36494, 2022.   
Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2022.   
D. Linsley, D. Schiebler, S. Eberhardt, and T. Serre. Learning what and where to attend. International Conference on Learning Representations (ICLR), 2019. URL https://openreview.net/forum?id=BJgLg3R9KQ.   
Thomas Fel, Ivan Felipe, Drew Linsley, and Thomas Serre. Harmonizing the object recognition strategies of deep neural networks with humans. International Conference on Learning Representations (ICLR), 2022b. doi: 10.48550/ARXIV.2211.04533.   
Nikolaus Kriegeskorte and Pamela K. Douglas. Cognitive computational neuroscience. Nature Neuroscience, 21 (9):1148–1160, 2018. doi: 10.1038/s41593-018-0210-5.   
Thomas Serre, Aude Oliva, and Tomaso Poggio. A feedforward architecture accounts for rapid categorization. Proceedings of the National Academy of Sciences, 104(15):6424–6429, 2007. doi: 10.1073/pnas.0700622104.   
Rufin VanRullen and Simon J. Thorpe. The time course of visual processing: From early perception to decisionmaking. Journal of Cognitive Neuroscience, 13(4):454–461, 2001. doi: 10.1162/08989290152001880.   
Karl Friston. A theory of cortical responses. Philosophical Transactions of the Royal Society B: Biological Sciences, 360(1456):815–836, 2005. doi: 10.1098/rstb.2005.1622.

Victor Boutin, Angelo Franciosini, Frédéric Chavane, and Laurent U Perrinet. Pooling strategies in v1 can account for the functional and structural diversity across species. PLOS Computational Biology, 18(7): e1010270, 2022a.   
Daniel Kersten, Pascal Mamassian, and Alan Yuille. Object perception as bayesian inference. Annual Review of Psychology, 55:271–304, 2004. doi: 10.1146/annurev.psych.55.090902.142005.   
Gabriel Kreiman and Thomas Serre. Beyond the feedforward sweep: Feedback computations in the visual cortex. Annals of the New York Academy of Sciences, 1464(1):222–241, 2020. doi: 10.1111/nyas.14320.   
Kohitij Kar and James J. DiCarlo. Fast recurrent processing via ventrolateral prefrontal cortex is needed by the primate ventral stream for robust core visual object recognition. Neuron, 109(1):164–176.e5, 2021. doi: 10.1016/j.neuron.2020.09.035.   
Victor Boutin, Lakshya Singhal, Xavier Thomas, and Thomas Serre. Diversity vs. recognizability: Human-like generalization in one-shot generative models. Advances in Neural Information Processing Systems, 35: 20933–20946, 2022b.   
Victor Boutin, Thomas Fel, Lakshya Singhal, Rishav Mukherji, Akash Nagaraj, Julien Colin, and Thomas Serre. Diffusion models as artists: Are we closing the gap between humans and machines? In International Conference on Machine Learning, pages 2953–3002. PMLR, 2023.   
Victor Boutin, Rishav Mukherji, Aditya Agrawal, Sabine Muzellec, Thomas Fel, Thomas Serre, and Rufin VanRullen. Latent representation matters: Human-like sketches in one-shot drawing tasks. Advances in Neural Information Processing Systems, 37:96282–96324, 2024.   
Robert Geirhos, Kristof Meding, and Felix A. Wichmann. Beyond accuracy: Quantifying trial-by-trial behaviour of CNNs and humans by measuring error consistency. Advances in Neural Information Processing Systems, 33, 2020b.   
Tal Golan, Prashant C. Raju, and Nikolaus Kriegeskorte. Controversial stimuli: Pitting neural networks against each other as models of human cognition. Proceedings of the National Academy of Sciences, 117(47): 29330–29337, 2020. doi: 10.1073/pnas.1912334117.   
Rajat Raina, Andrew Y. Ng, and Christopher D. Manning. Classification with hybrid generative/discriminative models. In Advances in Neural Information Processing Systems 16, 2003.   
Guillaume Bouchard and Bill Triggs. The tradeoff between generative and discriminative classifiers. In COMPSTAT 2004, pages 721–728, 2004.   
Victor Boutin, Aimen Zerroug, Minju Jung, and Thomas Serre. Iterative vae as a predictive brain model for out-of-distribution generalization. arXiv preprint arXiv:2012.00557, 2020.   
Gregory Druck, Chris Pal, Andrew McCallum, and Xiaojin Zhu. Semi-supervised classification with hybrid generative/discriminative methods. In Proceedings of the 13th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 280–289, 2007.   
Volodymyr Kuleshov and Stefano Ermon. Hybrid deep discriminative/generative models for semi-supervised learning. In Proceedings of the Thirty-Third Conference on Uncertainty in Artificial Intelligence, 2017.   
Jonathan Gordon and José Miguel Hernández-Lobato. Combining deep generative and discriminative models for bayesian semi-supervised learning. Pattern Recognition, 100:107156, 2020. doi: 10.1016/j.patcog.2019. 107156.   
Hugo Larochelle and Yoshua Bengio. Classification using discriminative restricted boltzmann machines. In Proceedings of the 25th International Conference on Machine Learning, pages 536–543, 2008. doi: 10.1145/1390156.1390224.   
Xiulong Yang and Shihao Ji. JEM++: Improved techniques for training JEM. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 6494–6503, 2021.   
Xiulong Yang, Qing Su, and Shihao Ji. Towards bridging the performance gaps of joint energy-based models. arXiv preprint arXiv:2209.07959, 2022.   
Will Grathwohl, Kuan-Chieh Wang, and Jorn-Henrik Jacobsen. Your Classifier is Secretly an Energy Based Model and You Should Treat it Like One. ICLR, 2020b. URL https://arxiv.org/abs/1912.03263.   
Gaurav Parmar, Richard Zhang, and Jun-Yan Zhu. On aliased resizing and surprising subtleties in gan evaluation. In CVPR, 2022.

L. Béthune, D. Vigouroux, Y. Du, R. VanRullen, and T. Serre. Follow the energy, find the path: Riemannian metrics from energy-based models for trajectory prediction. 2025. doi: 10.48550/arXiv.2505.18230.   
Alex Krizhevsky and Geoffrey Hinton. Learning multiple layers of features from tiny images. Technical report, University of Toronto, 2009.   
Haohan Wang, Songwei Ge, Eric P. Xing, and Zachary C. Lipton. Learning robust global representations by penalizing local predictive power, 2019. URL https://arxiv.org/abs/1905.13549.

# Supplementary Material

# A Extended Related Work

Generative and discriminative theories of vision. A longstanding question in vision science is whether human-like visual representations are better explained by discriminative or generative learning principles. Recent work frames this debate as a contrast between two influential views of vision: one emphasizing direct discriminative inference from sensory data, the other emphasizing perception as inference through the inversion of a generative model of the world Peters et al. [2024], Kriegeskorte and Douglas [2018]. On one side, rapid object recognition has often been modeled as a largely feedforward and discriminative process Riesenhuber and Poggio [1999], Serre et al. [2007], VanRullen and Thorpe [2001], DiCarlo and Cox [2007], Khaligh-Razavi and Kriegeskorte [2014]. On the other, generative accounts emphasize recurrent inference under an internal model, as in predictive coding Rao and Ballard [1999], Friston [2005], Boutin et al. [2022a] and Bayesian frameworks Lee and Mumford [2003], Yuille and Kersten [2006], Kersten et al. [2004]. An emerging view is that biological vision recruits both, with their relative contribution depending on task demands and sensory conditions Kriegeskorte and Douglas [2018], Kreiman and Serre [2020], Kar and DiCarlo [2021]. Feedforward discriminative processing may dominate rapid, simple recognition, whereas recurrent generative inference may become more important under ambiguity, degradation, or occlusion. Rather than resolving this debate at the algorithmic level, we ask a more operational question: which balance between discriminative and generative learning yields the most human-aligned representations?

Human alignment of generative and discriminative models. A growing body of work has compared generative and discriminative models on human-relevant perceptual and behavioral tasks. In recognition, supervised discriminative deep networks capture important aspects of primate IT representations Cadieu et al. [2014], Khaligh-Razavi and Kriegeskorte [2014], Yamins et al. [2014], Yamins and DiCarlo [2016], Schrimpf et al. [2018], whereas generative classifiers have been reported to show more human-like shape–texture bias, stronger alignment with human classification errors, and greater sensitivity to some perceptual illusions than standard discriminative models Jaini et al. [2024]. In perceptual judgment tasks, unsupervised generative models predict human judgments and misperceptions of gloss Storrs et al. [2021], while deep feature spaces align well with human perceptual similarity judgments Zhang et al. [2018]. Related comparisons have also been made in one-shot generation, where generative models differ in how well they reproduce human-like drawing behavior Boutin et al. [2022b, 2023, 2024]. More generally, comparisons between humans and models reveal substantial gaps in generalization, including robustness to image degradations, shape versus texture reliance, and image-level error patterns Geirhos et al. [2018b,a, 2020b], while controversial stimuli provide a sharper way to test which models best capture human judgments Golan et al. [2020]. Together, these results suggest that both generative and discriminative objectives can support aspects of human alignment. However, they do not isolate the role of the learning objective itself, because the compared models often differ simultaneously in architecture, training pipeline, and inductive bias.

Hybrid generative–discriminative models. A related line of work explores models that combine generative and discriminative learning rather than treating them as mutually exclusive Ng and Jordan [2001], Raina et al. [2003], Bouchard and Triggs [2004], Lasserre et al. [2006], Boutin et al. [2020]. This idea has been developed in several directions, including semi-supervised hybrid training Druck et al. [2007], Kuleshov and Ermon [2017], Gordon and Hernández-Lobato [2020] and hybrid energybased classifiers Larochelle and Bengio [2008], Grathwohl et al. [2020a]. These approaches are motivated by the complementary strengths of generative and discriminative objectives, but many introduce additional latent variables, separate modules, or partially distinct parameterizations Kuleshov and Ermon [2017], Gordon and Hernández-Lobato [2020]. For our purposes, Joint Energy-based Models (JEMs) are especially attractive because they preserve a standard classifier-like architecture while enabling a controlled interpolation between discriminative and generative training within a single scoring function Grathwohl et al. [2020a]. Subsequent work has further improved their training stability, speed, and empirical performance, making them more practical on modern image benchmarks Yang and Ji [2021], Yang et al. [2022]. This makes JEMs a particularly clean testbed for isolating the representational effect of the learning objective while minimizing architectural confounds.

Positioning of the present work. Closest to ours, recent work showed that generative classifiers can outperform standard discriminative models on several human-alignment benchmarks Jaini et al. [2024]. Our goal is complementary: rather than comparing distinct model families, we use JEMs to interpolate continuously between generative and discriminative objectives while keeping the architecture fixed. This lets us test how human alignment varies across that spectrum. We find that it peaks at an intermediate regime: the most human-like representations emerge neither from purely generative nor purely discriminative training, but from a balance between the two.

# B EBM training

# B.1 Contrastive divergence

The demonstration below is adapted from Woodford [2006] to fit our notation. Even though this mathematical derivation is not crucial for a good understanding of our work, we include it to make sure our article is self-contained and complete.

We consider an Energy-Based Model (EBM) defining a probability distribution via the Boltzmann form:

$$
p _ {\pmb {\theta}} (\mathbf {x}) = \frac {\exp (- E _ {\pmb {\theta}} (\mathbf {x}))}{Z (\pmb {\theta})} \quad \mathrm{with} \quad Z (\pmb {\theta}) = \int \exp (- E _ {\pmb {\theta}} (\mathbf {x})) d \mathbf {x}.
$$

Our goal is to minimize the negative log-likelihood with respect to the empirical data distribution $p _ { \mathcal { D } } \colon$

$$
\mathcal {L} _ {\mathrm{ML}} (\pmb {\theta}) = \mathbb {E} _ {\mathbf {x} \sim p _ {\mathcal {D}}} [ - \log p _ {\pmb {\theta}} (\mathbf {x}) ].
$$

We first expand the log-probability:

$$
- \log p _ {\boldsymbol {\theta}} (\mathbf {x}) = E _ {\boldsymbol {\theta}} (\mathbf {x}) + \log Z (\boldsymbol {\theta}).
$$

Taking the gradient with respect to $\pmb \theta \colon$

$$
\nabla_ {\boldsymbol {\theta}} \mathcal {L} _ {\mathrm{ML}} = \mathbb {E} _ {\mathbf {x} \sim p _ {\mathcal {D}}} \left[ \nabla_ {\boldsymbol {\theta}} E _ {\boldsymbol {\theta}} (\mathbf {x}) + \nabla_ {\boldsymbol {\theta}} \log Z (\boldsymbol {\theta}) \right].
$$

The derivative of the log-partition function could be simplified:

$$
\begin{array}{l} \nabla_ {\boldsymbol {\theta}} \log Z (\boldsymbol {\theta}) = \frac {1}{Z (\boldsymbol {\theta})} \nabla_ {\boldsymbol {\theta}} Z (\boldsymbol {\theta}) \\ = \frac {1}{Z (\boldsymbol {\theta})} \nabla_ {\boldsymbol {\theta}} \int \exp (- E _ {\boldsymbol {\theta}} (\mathbf {x})) d \mathbf {x} \\ = - \frac {1}{Z (\theta)} \int \exp (- E _ {\theta} (\mathbf {x})) \nabla_ {\pmb {\theta}} E _ {\pmb {\theta}} (\mathbf {x}) d \mathbf {x} \\ = - \int p _ {\pmb {\theta}} (\mathbf {x}) \nabla_ {\pmb {\theta}} E _ {\pmb {\theta}} (\mathbf {x}) d \mathbf {x} \\ = - \mathbb {E} _ {\mathbf {x} \sim p _ {\theta}} \left[ \nabla_ {\theta} E _ {\theta} (\mathbf {x}) \right]. \\ \end{array}
$$

Substituting this back into the gradient of the loss:

$$
\nabla_ {\boldsymbol {\theta}} \mathcal {L} _ {\mathrm{ML}} = \mathbb {E} _ {\mathbf {x} \sim p _ {\mathcal {D}}} \left[ \nabla_ {\boldsymbol {\theta}} E _ {\boldsymbol {\theta}} (\mathbf {x}) \right] - \mathbb {E} _ {\mathbf {x} \sim p _ {\boldsymbol {\theta}}} \left[ \nabla_ {\boldsymbol {\theta}} E _ {\boldsymbol {\theta}} (\mathbf {x}) \right].
$$

In practice, we denote the $\mathbf { x } ^ { + }$ the "positive" samples from the empirical data distribution $p _ { D }$ , and $\mathbf { x } ^ { - }$ the "negative" samples from the model:

$$
\nabla_ {\boldsymbol {\theta}} \mathcal {L} _ {\mathrm{ML}} \approx \mathbb {E} _ {\mathbf {x} ^ {+} \sim p _ {\mathcal {D}}} \left[ \nabla_ {\boldsymbol {\theta}} E _ {\boldsymbol {\theta}} (\mathbf {x} ^ {+}) \right] - \mathbb {E} _ {\mathbf {x} ^ {-} \sim p _ {\boldsymbol {\theta}}} \left[ \nabla_ {\boldsymbol {\theta}} E _ {\boldsymbol {\theta}} (\mathbf {x} ^ {-}) \right].
$$

# B.2 JEM training

Algorithm. To train our Joint Energy-Based Models (JEMs), we follow the general approach of Grathwohl et al. [2020b]. Algorithm 1 summarizes the training procedure used in our experiments.

Algorithm 1: Training Joint Energy Model using Langevin dynamics and gradient-balanced interpolation between generative and discriminative objectives.   
Input: Training dataset D, learning rate η, replay buffer B, initial Langevin step size λ, noise scale σ, number of Langevin steps L, interpolation weight α   
while Training do   
$\mathbf{x}^{+}\sim \mathcal{D}$ # sample from the dataset $(\mathbf{x},y)\sim \mathcal{D}$ # sample a batch for the discriminative term $\mathbf{x}^0\sim \mathcal{B}$ # sample from a replay buffer  
# Refine negative samples using Langevin dynamics  
for $t\gets 0$ to $L - 1$ do $\left|\begin{array}{l}\lambda_t\gets \lambda \left(1 - \frac{t}{L}\right)\quad \# \text{linearly decayed step size}\\ \mathbf{x}^{t + 1}\gets \mathbf{x}^t +\lambda_t\nabla_{\mathbf{x}^t}\log p_\theta (\mathbf{x}^t) + \omega \quad \text{with}\omega \sim \mathcal{N}(0,\sigma)\\ \mathbf{x}^{t + 1}\gets \mathrm{clip}(\mathbf{x}^{t + 1}, - 1,1)\# \text{optional clipping for stabilization}\end{array}\right.$ $\mathbf{x}^{-} = \mathbf{x}^{L}.$ detach() $\ell_G = -\left(\frac{1}{B}\sum_i f_\theta (\mathbf{x}_i^+) - \frac{1}{B}\sum_i f_\theta (\mathbf{x}_i^-)\right)$ # generative loss $\ell_D = -\frac{1}{B}\sum_i\log p_\theta (y_i|\mathbf{x}_i)$ # discriminative loss $g_{G}\gets \| \nabla_{\pmb{\theta}}\ell_{G}\| ,\qquad g_{D}\gets \| \nabla_{\pmb{\theta}}\ell_{D}\|$ $c\gets \frac{g_D}{g_G + \varepsilon}$ # gradient-norm correction $\ell = \alpha c\ell_G + (1 - \alpha)\ell_D$ $\pmb {\theta}\gets \pmb {\theta} - \eta \nabla_{\pmb {\theta}}\ell$ #update parameters by gradient descent $\mathcal{B}\gets \mathcal{B}\cup \mathbf{x}^{-}$ #update replay buffer

In the reported experiments, we sweep α from 0 to 1 in increments of 0.1, yielding 11 models spanning the range from discriminative to generative learning. This provides a controlled continuum that allows us to test whether human alignment and scene-property-related metrics arise from the balance between both, rather than from differences in model class.

Unless otherwise noted, the main training settings are as follows:

• We use L = 20 Langevin steps with a multistep decay schedule for the SGLD step size, which improved stability in all experiments. The initial Langevin step size λ is typically set between 0.8 and 1.0, and the Langevin noise scale σ between $9 \times 1 \dot { 0 } ^ { - 3 }$ and $2 \times 1 0 ^ { - 2 }$ .   
• Within each Langevin chain, we linearly decay the step size from its initial value to zero, which we found helpful for stabilizing training.   
• Model parameters are optimized with Adam, using learning rates between $1 \times 1 0 ^ { - 4 }$ and $5 \times 1 0 ^ { - 5 }$ , since larger learning rates tended to destabilize hybrid objectives.   
• We use a cosine learning-rate schedule with 1000 warmup iterations.   
• We found input noise augmentation with standard deviation around 0.05 to be helpful for stabilizing training, and use dropout rates of at most 0.04, or no dropout at all.   
• When both generative and discriminative objectives are active, we rescale the generative term using the ratio of gradient norms, as described in Eq. 8.   
• We do not use batch normalization in the energy model, as in our experiments it tended to destabilize generative training and often prevented convergence.   
• All JEMs were trained using mixed precision (via PyTorch AMP) and torch.compile to improve training efficiency.   
• For Gloss and CIFAR-10H, the generative model (α = 1.0) was selected at the best-FID epoch using Clean-FID Parmar et al. [2022].

Compute resources. Experiments were run on an internal Slurm cluster. CIFAR-10 experiments used single-GPU jobs on NVIDIA L40S GPUs, with 4 CPU cores and 26 GB RAM per run. Based on the training logs, a typical CIFAR-10 epoch took about 5.5 minutes, and runs were allocated up to 36 hours of wall-clock time.

Gloss and ImageNet experiments used single-GPU jobs on NVIDIA B200 GPUs, with 6 CPU cores and 80 GB RAM per run. For gloss, a typical epoch took about 2.1 minutes, and the slowestconverging settings were allocated up to 13 hours of wall-clock time. For ImageNet, a typical epoch took about 17.9 minutes. ImageNet runs were typically allocated up to 4 days, while the highest-α settings (α = 0.7, 0.8, 0.9, 1.0) were allocated up to 5 days because they took longer to converge.

Full α-sweeps consisted of 11 runs launched in parallel, so the elapsed wall-clock time for a sweep was determined by the slowest job, while total compute scaled with the number of parallel runs. These values are reported as wall-clock budgets for the main runs. The full research project required additional preliminary and failed runs beyond the final experiments reported in the paper.

Existing datasets, codebases, and pretrained models. Our experiments use existing public datasets, codebases, and pretrained models, including CIFAR-10, CIFAR-10H, the gloss datset, the Model-vs-Human benchmark suite, ClickMe-related evaluation assets, texture–shape cue-conflict benchmarks, the JEM codebase, pytorch\_image\_classification models, LPIPS/PerceptualSimilarity, the Harmonization repository, and the stabilityai/sd-vae-ft-mse AutoencoderKL checkpoint. We cite the original papers and repositories for all of these assets throughout the paper and appendix, and report licenses when clearly stated in the original source, including Apache-2.0 for JEM, MIT for pytorch\_image\_classification, Harmonization, and stabilityai/sd-vae-ft-mse, CC BY-NC-SA 4.0 for CIFAR-10H, and CC BY 4.0 for the gloss and texture–shape cue-conflict datasets. For assets whose official public source did not clearly expose a license, we cite the original source and follow stated access and usage conditions.

# B.3 Architecture of the Energy Function

Our implementation is based on the public code released by Grathwohl et al. Grathwohl et al. [2020a], which parameterizes the energy function with a Wide Residual Network (WRN) backbone Zagoruyko and Komodakis [2016]. Relative to the standard WRN formulation, this implementation uses LeakyReLU activations, configurable normalization layers that can be disabled entirely, and optional dropout, all of which we found useful for stabilizing hybrid generative–discriminative training. For smaller-scale datasets such as CIFAR-10 and gloss, we use WRN-28-10 and WRN-22-10, respectively, where the first number denotes network depth and the second the widening factor.

# B.4 ImageNet training

For ImageNet, we use WRN-22-8 to obtain a more computationally manageable model while retaining the same architectural family. Rather than training directly in pixel space, we train the model in the latent space of a frozen pretrained AutoencoderKL Rombach et al. [2022], using the stabilityai/sd-vae-ft-mse checkpoint. Operating in latent space substantially reduces the dimensionality of the input, which makes hybrid and fully generative training at ImageNet scale computationally feasible. Concretely, images are first encoded into latent representations, and these latents are then used as inputs to the energy model (see Béthune et al. [2025]). We use the posterior mean of the encoder, rather than posterior sampling, yielding a deterministic latent representation; in our experiments, this option was more stable for training than sampling from the posterior.

As a sanity check, we trained a purely discriminative WRN with the same hyperparameters directly in pixel space and found that it achieved essentially the same top-1 accuracy as the α = 0 JEM trained in VAE latent space. This suggests that operating in the VAE latent space preserves discriminative performance at a level comparable to the pixel-space baseline.

Qualitative sampling behavior. Figures 5a and 5b illustrate the effect of latent-space SGLD refinement for an ImageNet JEM with α = 0.5. We begin from a random noise image, encode it with the VAE, and use the resulting latent as the starting point for SGLD under the learned energy function. The first column shows the VAE decoding of this initial latent, while subsequent columns show the decoded latent after increasing numbers of SGLD refinement steps. Early in training, this refinement produces only weakly organized image structure, whereas by the best validation epoch, it yields substantially more coherent generations. Most of the qualitative improvement occurs within the first several refinement steps, after which the samples largely stabilize. Figure 6 compares generations obtained from the same initialization across values of α, illustrating how the balance between discriminative and generative training affects the resulting samples. The figures are intended as qualitative illustrations of the sampling dynamics rather than quantitative evaluation metrics.

![](images/51140e5170227dff9997261fc81e72094ca60bbf6dbbd3e23fbe8819c977e1ca.jpg)

<details>
<summary>text_image</summary>

Decoded noise SGLD 1 SGLD 2 SGLD 4 SGLD 8 SGLD 16 SGLD 32 SGLD 50
</details>

(a) First training epoch.

![](images/294f859bd93403a9e26ebd4f4e1369e277ad2d10e1580955bac5c242d6272503.jpg)

<details>
<summary>text_image</summary>

Decoded noise SGLD 1 SGLD 2 SGLD 4 SGLD 8 SGLD 16 SGLD 32 SGLD 50
</details>

(b) Best validation epoch.   
Figure 5. Latent-space sampling trajectories for an ImageNet JEM with $\alpha = 0 . 5$ . Each row starts from the VAE latent of a random noise image, and successive columns show the decoded sample after increasing numbers of SGLD refinement steps. (a) shows trajectories at the first training epoch; (b) shows trajectories at the best validation epoch.

![](images/95d15adb37fecc95bb0913d27fd8fc9f1eac884368a3a61d4fbe19dfff19a583.jpg)

<details>
<summary>text_image</summary>

Decoded noise SGLD 1 SGLD 2 SGLD 4 SGLD 8 SGLD 16 SGLD 32 SGLD 50
α = 0.1
α = 0.2
α = 0.3
α = 0.4
α = 0.5
α = 0.6
α = 0.7
α = 0.8
α = 0.9
α = 1
</details>

Figure 6. Qualitative generations obtained after 50 SGLD steps from the same shared latent initialization across different values of α. The same shared initial latent code and the same sampling seed are used for all models, so that visual differences more directly reflect the effect of α. The first column shows the VAE decoding of the shared initial latent, and the remaining columns show the decoded samples after refinement under each model.

# C Low-level Perceptual Similarity Benchmark

To evaluate alignment with human judgments of low-level perceptual similarity, we used the BAPPS dataset, which contains approximately 484,000 human judgments. As mentioned, its main benchmark is a two-alternative forced choice (2AFC) task in which, given a reference patch and two distorted versions of it, observers indicate which distorted patch is more similar to the reference (Figure 7). The dataset includes a wide range of traditional distortions, CNN-based artifacts, and outputs from real image-processing algorithms, making it a broad test for determining whether learned representations preserve perceptual similarity in a way that agrees with human judgments.

![](images/edea236971952f67a600daae9d531a762929eeb5db20f768321702090eb4982a.jpg)

<details>
<summary>text_image</summary>

Patch 0
Reference
Patch 1
</details>

Figure 7. Example of a reference image and its distorted patches.

BAPPS also includes a just noticeable difference (JND) task to measure sensitivity to small perceptual changes. In the JND task, observers are required to judge whether a distorted path and the reference image appear perceptually the same or different. Models are then evaluated through similarity, by computing a distance measure for each image pair and ranking pairs from most to least similar; a good model assigns small distances to pairs that humans most often judge as the same.

# C.1 Models and evaluation

We evaluated JEMs on BAPPS using the standard 2AFC and JND protocols (Figure 8), following Zhang et al. Zhang et al. [2018]. For 2AFC, performance is reported as agreement with human preference judgments; for JND, we rank image pairs by increasing perceptual distance and report mean average precision (mAP).

To evaluate a JEM, we construct an LPIPS-style perceptual distance from intermediate activations of the WRN backbone, using features extracted from the three main residual stages. We evaluate the raw distance induced by the pretrained representation alone, using equal layer/channel weighting as in the uncalibrated setting of Zhang et al. Zhang et al. [2018]. Thus, our BAPPS results test whether perceptual similarity emerges naturally, rather than from direct supervision on BAPPS itself.

For comparison, we include the pretrained supervised baselines released by Zhang et al. Zhang et al. [2018]. For BiGAN, because the implementation and pretrained checkpoint were not publicly available, we report the values provided in the original paper. We additionally report a VAE baseline from the frozen AutoencoderKL used in our ImageNet experiments, using features from encoder down-blocks 0–3. This baseline provides a reference point for the perceptual quality of the pretrained latent representation itself to determine whether JEM training improves upon the VAE features.

![](images/0dc52966b3338cfb9e41379ddde76e71fa6840b3da8f93e51b44f96aeb87acab.jpg)

<details>
<summary>line</summary>

| α    | 2AFC [%] |
| ---- | -------- |
| 0.0  | 75.1     |
| 0.2  | 75.3     |
| 0.4  | 76.7     |
| 0.6  | 76.8     |
| 0.8  | 75.3     |
| 1.0  | 74.3     |
</details>

![](images/1f1971d5c4b2ab5b1e3f7ec8b47614340e89ae570da690d9f8335d1cd568de2c.jpg)

<details>
<summary>line</summary>

| α    | JND mAP [%] |
| ---- | ----------- |
| 0.0  | 55.2        |
| 0.1  | 54.8        |
| 0.2  | 55.0        |
| 0.3  | 55.3        |
| 0.4  | 55.7        |
| 0.5  | 56.0        |
| 0.6  | 56.9        |
| 0.7  | 56.7        |
| 0.8  | 56.0        |
| 0.9  | 55.3        |
| 1.0  | 55.2        |
</details>

Figure 8. 2AFC accuracy and JND mAP on the BAPPS perceptual similarity benchmark, across JEM α values. Shaded regions indicate the standard error of the mean (SEM) across two seeds.

# D Gloss and depth perceptual benchmark

The gloss perception dataset probes mid-level material perception. This is a challenging task because it requires distinguishing surface reflectance properties while disentangling them from other physical factors, such as illumination and surface geometry, that also shape image appearance. We use the dataset introduced by Storrs et al. Storrs et al. [2021], which includes visual stimuli, class labels, and human ratings. One component of the dataset consists of 9,998 rendered images of bumpy surfaces with varying surface relief, diffuse color, lighting, and specular reflectance properties, together with binary labels indicating high versus low gloss (Figure 9). The authors also provide a second set of 50 rendered surfaces with randomly sampled specular reflectance magnitudes, ranging from almost matte to almost mirror-like, along with ratings from 20 human observers.

![](images/defa422bba9a056cf43c72f075f91a890bf7005684099835c9a74ba816219815.jpg)

<details>
<summary>natural_image</summary>

Microscopic view of spherical particles under low and high conditions, showing varying bubble sizes and surface textures (no text or symbols)
</details>

Figure 9. Examples of images labeled as Low or High gloss.

# D.1 Models and data

In general, and to make direct comparisons possible, we followed Storrs et al. Storrs et al. [2021] as a blueprint for our gloss experiments. We used the rendered bumpy-surface images described in the previous section. Unless otherwise noted, all images were resized to 128 × 128 pixels and normalized to the range [−1, 1].

Because the original dataset did not include a validation split, we introduced one for model selection while keeping the test set held out for final evaluation. Specifically, we partitioned the 9,998 images into 3 subsets: 8,000 training images, 1,000 validation images, and 998 test images.

In the original study, the authors compared an unsupervised PixelVAE model Zhao et al. [2017] with a supervised ResNet-18 model He et al. [2016]. For direct comparison, we trained the same two model classes using the public implementation provided in the Storrs et al. Storrs et al. [2021] repository and evaluated them with the same downstream procedures, including the human-alignment analysis. For PixelVAE, we followed the architecture and hyperparameters reported in the original study, including six residual blocks with three layers of 64 convolutional feature maps. For ResNet-18, we followed the setup consisting of three residual blocks each made up of three layers of 56 convolutional feature maps, with a fully connected latent-code layer, and a 2-unit softmax classification head. In both models, the bottleneck representation was used for the downstream probing analyses. For JEMs, we inserted a small projection head preceded by a concat-ELU nonlinearity, motivated by its use in the PixelVAE architecture from the original study. All reported JEM results use this parameterization.

# D.2 Training protocol

We first reproduced the original findings using their reported hyperparameters and data split, obtaining nearly identical metrics. We then retrained both PixelVAE and ResNet18 on the same data split used for the JEMs to provide fair baselines for comparison. For JEMs, PixelVAE, and ResNet18 alike, we considered latent dimensionalities of 10, 100, 500, and 2000. PixelVAE baselines were trained with 10 random seeds. In contrast, the ResNet18 baselines used one seed, and each of the eleven JEM variants were trained with two seeds per condition. Additionally, we used mild label smoothing of 0.05 for the JEMs, which we found helpful for stabilizing training in the binary classification setting.

The next step was to evaluate each model through a fixed-dimensional latent representation. In their analysis Storrs et al. [2021], the bottleneck layer of the PixelVAE was treated as the model’s latent feature space and used for downstream analyses of gloss, depth, and human alignment. For our experiments, aditionally, we needed to assess how the balance bewteen discriminative and generative learning affected both alignment with human judgements and the recoverability of task-relevant scene properties. To do this and, analog to what was done in the reference material, we extracted a fixed-dimensional representation from each JEM and applied the same downstream probes to that representation.

Following the general analysis protocol of Storrs et al. Storrs et al. [2021], we extracted a fixeddimensional latent representation and used it to assess the recoverability of gloss, illumination, and surface relief. All downstream probes were trained on the non-test portion of the data, (training and validation images), and were evaluated on the test set of images.

# D.3 Evaluation metrics

We report three scene-property metrics: (i) gloss classification accuracy, (ii) six-way illumination classification accuracy, and (iii) surface-relief prediction accuracy measured by $R ^ { 2 }$ .

Gloss accuracy was measured directly from the models’ classifier head. For fully generative models, we instead trained a linear SVM on the extracted representation, again using five-fold cross-validation on the non-test portion of the data to select C. The linear SVM additionally provides decision values, which were used in the human-alignment analysis described ahead.

For light field classification, we trained a linear SVM, and the regularization parameter $C$ was selected by using five-fold cross-validation within the non-test portion of the data.

For surface-relief prediction, we trained a linear regression model and report test-set $R ^ { 2 }$ for predicting bump height.

In all cases, latent features were standardized using a StandardScaler before fitting the downstream model.

# D.4 Human alignment

To evaluate alignment with human perception, we compared model-derived gloss predictions with the human gloss ratings on the 50 surfaces rendered by Storrs et al. Storrs et al. [2021]. These were rated by 20 human observers on a six-point scale from 1 (matte) to 6 (glossy).

As done in our reference study, we derived a continuous gloss prediction for each image from the decision value of a glossy-versus-matte linear SVM, that ${ \mathrm { i s } } ,$ the signed distance of the image representation from the gloss-separating hyperplane. For model families with multiple training instances, predictions were computed separately for each instance and then averaged across instances for each of the 50 stimuli.

We quantified human alignment by computing the correlation between these continuous model predictions and the mean human gloss rating for each of the 50 stimuli. Higher correlation indicates that the learned representation gives rise to gloss predictions that are more consistent with human perceptual judgments.

The same evaluation procedure was applied to JEMs, PixelVAE seeds, and ResNet18 baselines, so that human-alignment comparisons were based on a matched protocol across model families.

# D.5 Further discussion

Human alignment peaks in the hybrid regime. Across latent dimensionalities, intermediate JEMs— especially around α = 0.5—consistently achieve the strongest or near-strongest human correlation, whereas the purely generative endpoint $( \alpha = 1 )$ performs noticeably worse. Because the JEM comparisons keep the architecture fixed, this pattern provides stronger evidence than prior crossmodel comparisons that the learning objective itself plays a central role in shaping human-aligned gloss representations. This conclusion is especially clear in the 500-dimensional setting, where the JEM is approximately matched in parameter count to the PixelVAE (roughly 37M versus 40M parameters). Even in this more comparable regime, the strongest human alignment is obtained by a hybrid JEM rather than by the purely generative baseline (Fig. 10).

PixelVAE nevertheless remains a strong baseline, reinforcing the original insight of Storrs et al. Storrs et al. [2021]: that generative learning captures important structure for gloss perception. However, because the original PixelVAE–ResNet comparison also changed model family and parameterization, it did not isolate the contribution of the learning objective itself. Our results refine that conclusion: once architecture is controlled, the best human alignment does not arise from purely generative training, but from an intermediate balance between generative and discriminative pressure. This hybrid advantage is not limited to human correlation. The same intermediate JEMs also yield the strongest downstream decoding of scene properties, including surface relief and light field, which indicates that the hybrid regime preserves the latent physical variables underlying gloss more effectively than either endpoint of the continuum (Fig. 11).

For completeness, we also report the two-seed results for the 500-dimensional setting (Fig. 12) together with qualitative generations across α (Fig. 13). While increasing α generally improves the visual plausibility of the generated surfaces, the best alignment with human gloss judgments is achieved in the hybrid regime rather than at the purely generative endpoint.

a) 10 dimensions   
![](images/fd7673f0952f4d67bbbfa59146dda409816e2d9825827c323fb5f7da95e88f40.jpg)

<details>
<summary>scatter</summary>

| Model      | Gloss accuracy [%] | Human correlation |
| ---------- | ------------------ | ----------------- |
| PixelVAE   | 98                 | 0.9               |
| ResNet     | 98                 | 0.85              |
| PixelVAE   | 98                 | 0.8               |
| ResNet     | 98                 | 0.8               |
| PixelVAE   | 98                 | 0.85              |
| ResNet     | 98                 | 0.8               |
| PixelVAE   | 98                 | 0.85              |
| ResNet     | 98                 | 0.8               |
| PixelVAE   | 98                 | 0.85              |
| ResNet     | 98                 | 0.8               |
| PixelVAE   | 98                 | 0.9               |
| ResNet     | 98                 | 0.85              |
| PixelVAE   | 98                 | 0.8               |
| ResNet     | 98                 | 0.85              |
| PixelVAE   | 98                 | 0.8               |
| ResNet     | 98                 | 0.85              |
| PixelVAE   | 98                 | 0.8               |
| ResNet     | 98                 | 0.85              |
| PixelVAE   | 98                 | 1.0               |
| ResNet     | 98                 | 1.0               |
| PixelVAE   | 98                 | 1.0               |
| ResNet     | 98                 | 1.0               |
| PixelVAE   | 98                 | 1.0               |
| ResNet     | 98                 | 1.0               |
| PixelVAE   | 98                 | 1.0               |
| ResNet     | 98                 | 1.0               |
</details>

b) 100 dimensions   
![](images/6b63cb974bbe5db492a286e9f30f51f9d32779d1e0fc22a74cc7877456d67590.jpg)

<details>
<summary>scatter</summary>

| Model     | Gloss accuracy [%] | Human correlation |
| --------- | ------------------ | ----------------- |
| PixelVAE  | 99.0               | 0.90              |
| ResNet    | 97.5               | 0.82              |
| ResNet    | 99.5               | 0.85              |
| ResNet    | 99.8               | 0.83              |
| ResNet    | 99.9               | 0.84              |
| ResNet    | 100.0              | 0.81              |
| PixelVAE  | 96.0               | 0.87              |
| PixelVAE  | 99.0               | 0.91              |
| PixelVAE  | 100.0              | 0.90              |
| PixelVAE  | 100.0              | 0.89              |
| PixelVAE  | 100.0              | 0.88              |
| PixelVAE  | 100.0              | 0.87              |
| PixelVAE  | 100.0              | 0.86              |
| PixelVAE  | 100.0              | 0.85              |
| PixelVAE  | 100.0              | 0.84              |
| PixelVAE  | 100.0              | 0.83              |
| PixelVAE  | 100.0              | 0.82              |
| PixelVAE  | 100.0              | 0.81              |
| PixelVAE  | 100.0              | 0.80              |
| PixelVAE  | 100.0              | 0.79              |
| PixelVAE  | 100.0              | 0.78              |
| PixelVAE  | 100.0              | 0.77              |
| PixelVAE  | 100.0              | 0.76              |
| PixelVAE  | 100.0              | 0.75              |
| PixelVAE  | 100.0              | 0.74              |
| PixelVAE  | 100.0              | 0.73              |
| PixelVAE  | 100.0              | 0.72              |
| PixelVAE  | 100.0              | 0.71              |
| PixelVAE  | 100.0              | 0.70              |
| PixelVAE  | 100.0              | 0.69              |
| PixelVAE  | 100.0              | 0.68              |
| PixelVAE  | 100.0              | 0.67              |
| PixelVAE  | 100.0              | 0.66              |
| PixelVAE  | 100.0              | 0.65              |
| PixelVAE  | 100.0              | 0.64              |
| PixelVAE  | 100.0              | 0.63              |
| PixelVAE  | 100.0              | 0.62              |
| PixelVAE  | 100.0              | 0.61              |
| PixelVAE  | 100.0              | 0.60              |
| PixelVAE  | 100.0              | 0.59              |
| PixelVAE  | 100.0              | 0.58              |
| PixelVAE  | 100.0              | 0.57              |
| PixelVAE  | 100.0              | 0.56              |
| PixelVAE  | 100.0              | 0.55              |
| PixelVAE  | 100.0              | 0.54              |
| PixelVAE  | 100.0              | 0.53              |
| PixelVAE  | 100.0              | 0.52              |
| PixelVAE  | 100.0              | 0.51              |
| PixelVAE  | 100.0              | 0.50              |
| PixelVAE  | 100.0              | 0.49              |
| PixelVAE  | 100.0              | 0.48              |
| PixelVAE  | 100.0              | 0.47              |
| PixelVAE  | 100.0              | 0.46              |
| PixelVAE  | 100.0              | 0.45              |
| PixelVAE  | 100.0              | 0.44              |
| PixelVAE  | 100.0              | 0.43              |
| PixelVAE  | 100.0              | 0.42              |
| PixelVAE  | 100.0              | 0.41              |
| PixelVAE  | 100.0              | 0.40              |
| PixelVAE  | 100.0              | 0.39              |
| PixelVAE  | 100.0              | 0.38              |
| PixelVAE  | 100.0              | 0.37              |
| PixelVAE  | 100.0              | 0.36              |
| PixelVAE  | 100.0              | 0.35              |
| PixelVAE  | 100.0              | 0.34              |
| PixelVAE  | 100.0              | 0.33              |
| PixelVAE  | 100.0              | 0.32              |
| PixelVAE  | 100.0              | 0.31              |
| PixelVAE  | 100.0              | 0.30              |
| PixelVAE  | 100.0              | 0.29              |
| PixelVAE  | 100.0              | 0.28              |
| PixelVAE  | 100.0              | 0.27              |
| PixelVAE  | 100.0              | 0.26              |
| PixelVAE  | 100.0              | 0.25              |
| PixelVAE  | 100.0              | 0.24              |
| PixelVAE  | 100.0              | 0.23              |
| PixelVAE  | 100.0              | 0.22              |
| PixelVAE  | 100.0              | 0.21              |
| PixelVAE  | 100.0              | 0.225             |
| PixelVAE   | -                  | -                 |
| ResNet    | -                  | -                 |
| ResNet    | -                  | -                 |
| ResNet    | -                  | -                 |
| ResNet    | -                  | -                 |
| ResNet    | -                  | -                 |
| ResNet    | -                  | -                 |
| ResNet    | -                  | -                 |
| ResNet    | -                  | -                 |
| ResNet    | -                  | -                 |
| ResNet    (approx) - ResNet (approx)      vs -          |
| ResNet (approx) - ResNet (approx)      vs -          |
| ResNet (approx) - ResNet (approx)      vs -          |
| ResNet (approx) - ResNet (approx)      vs -          |
| ResNet (approx) - ResNet (approx)      vs -          |
| ResNet (approx) - ResNet (approx)      vs -          |
| ResNet (approx) - ResNet (approx)      vs -          |
|
| ResNet (approx) - ResNet (approx)      vs -          |
|
| ResNet (approx) - ResNet (approx)      vs -          |
|
|
| ResNet (approx) - ResNet (approx)      vs -          |
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|
|        |

Note: The data points are estimated based on the provided code format as shown in the code format.
</details>

c) 500 dimensions   
![](images/7adf753cf4e74c094a47bbe85fbec8237ec8253831335ee4d1b1420227b956e6.jpg)

d) 2000 dimensions   
![](images/79b07250a3db63ce3719074f57cee677b2bd253d54787911df370defd39bdf9e.jpg)

<details>
<summary>scatter</summary>

| Model     | Gloss accuracy [%] | Human correlation | JEMs |
|-----------|--------------------|-------------------|------|
| PixelVAE  | 97.0               | 0.58              | 1.0  |
| ResNet    | 98.2               | 0.87              | 0.5  |
| ResNet    | 99.2               | 0.81              | 0.5  |
| ResNet    | 100.0              | 0.90              | 0.5  |
</details>

Figure 10. Gloss accuracy vs. human correlation in experiments with different latent sizes.

a) 10 dimensions   
![](images/45f9901b5457d93e9bb910f8c845aa6ae7aba70516ed0aa498476c0ab1f24137.jpg)

<details>
<summary>line</summary>

| α    | Score (Blue Line) | Score (Orange Line) |
| ---- | ----------------- | ------------------- |
| 0.0  | 0.35              | 0.33                |
| 0.1  | 0.37              | 0.35                |
| 0.2  | 0.38              | 0.37                |
| 0.3  | 0.39              | 0.26                |
| 0.4  | 0.41              | 0.23                |
| 0.5  | 0.45              | 0.23                |
| 0.6  | 0.47              | 0.34                |
| 0.7  | 0.49              | 0.38                |
| 0.8  | 0.45              | 0.21                |
| 0.9  | 0.41              | 0.16                |
| 1.0  | 0.43              | 0.20                |
</details>

b) 100 dimensions   
![](images/d037399294f04a321ae1f145b18b0d5277d61cc02f3439e3f5eb9d5608f88487.jpg)

<details>
<summary>line</summary>

| α    | Score (Orange Line) | Score (Blue Line) |
| ---- | ------------------- | ----------------- |
| 0.0  | 0.67                | 0.54              |
| 0.1  | 0.71                | 0.52              |
| 0.2  | 0.75                | 0.53              |
| 0.3  | 0.74                | 0.54              |
| 0.4  | 0.72                | 0.55              |
| 0.5  | 0.73                | 0.59              |
| 0.6  | 0.75                | 0.59              |
| 0.7  | 0.63                | 0.56              |
| 0.8  | 0.53                | 0.51              |
| 0.9  | 0.42                | 0.51              |
| 1.0  | 0.41                | 0.53              |
| PixelVAE | 0.78              | 0.61              |
</details>

c) 500 dimensions   
![](images/1c5792af863a597bf923e12ae58e652a689cf023bfefed8edb157ba191bbb388.jpg)

<details>
<summary>line</summary>

| α    | Score (Orange Line) | Score (Blue Line) |
| ---- | ------------------- | ----------------- |
| 0.0  | 0.70                | 0.65              |
| 0.1  | 0.75                | 0.62              |
| 0.2  | 0.80                | 0.63              |
| 0.3  | 0.81                | 0.64              |
| 0.4  | 0.83                | 0.64              |
| 0.5  | 0.83                | 0.64              |
| 0.6  | 0.81                | 0.65              |
| 0.7  | 0.86                | 0.67              |
| 0.8  | 0.84                | 0.62              |
| 0.9  | 0.63                | 0.57              |
| 1.0  | 0.65                | 0.58              |
| PixelVAE | 0.75              | 0.57              |
</details>

d) 2000 dimensions   
![](images/f9239ec80bc97bcf7d3c8da4cd39a4fd95d1871190d4f4cab6e5e0cc0e96b987.jpg)

<details>
<summary>line</summary>

| α    | Score (Orange Line) | Score (Blue Line) |
| ---- | ------------------- | ----------------- |
| 0.0  | 0.74                | 0.70              |
| 0.1  | 0.80                | 0.66              |
| 0.2  | 0.84                | 0.66              |
| 0.3  | 0.87                | 0.65              |
| 0.4  | 0.86                | 0.70              |
| 0.5  | 0.87                | 0.70              |
| 0.6  | 0.89                | 0.72              |
| 0.7  | 0.88                | 0.71              |
| 0.8  | 0.86                | 0.68              |
| 0.9  | 0.82                | 0.64              |
| 1.0  | 0.82                | 0.68              |
| PixelVAE | 0.73              | 0.52              |
</details>

Surface relief $( R ^ { 2 } )$   
Light field accuracy

Figure 11. Surface relief $( R ^ { 2 } )$ and light field accuracy across JEM α values for different latent dimensionalities. Shaded regions indicate the standard error of the mean (SEM) across two seeds. The disconnected point on the right of each panel shows the corresponding PixelVAE baseline.   
a) Human correlation   
![](images/4dcda485d2beff34b377ccaa25f99fc8a066faadd0e5a94959207550c73ad9e3.jpg)

<details>
<summary>line</summary>

| α    | Score |
| ---- | ----- |
| 0.0  | 0.85  |
| 0.1  | 0.87  |
| 0.2  | 0.75  |
| 0.3  | 0.82  |
| 0.4  | 0.76  |
| 0.5  | 0.94  |
| 0.6  | 0.91  |
| 0.7  | 0.91  |
| 0.8  | 0.90  |
| 0.9  | 0.82  |
| 1.0  | 0.55  |
| PixelVAE | 0.86 |
</details>

b) Gloss accuracy   
![](images/b5a03f81379e74b41c03d3c8ff35a3c64420439bcd48bd40be85a6eb05036845.jpg)

<details>
<summary>line</summary>

| α    | Score |
| ---- | ----- |
| 0    | 1.00  |
| 0.1  | 1.00  |
| 0.2  | 1.00  |
| 0.3  | 1.00  |
| 0.4  | 1.00  |
| 0.5  | 1.00  |
| 0.6  | 1.00  |
| 0.7  | 1.00  |
| 0.8  | 0.99  |
| 0.9  | 0.97  |
| 1    | 0.94  |
| PixelVAE | 0.98 |
</details>

Gloss-human correlation   
Gloss accuracy

Figure 12. Gloss-human correlation and gloss accuracy across JEM α values at 500 latent dimensions. Shaded regions indicate the standard error of the mean (SEM) across two seeds. The disconnected point on the right of each panel shows the corresponding PixelVAE baseline.

![](images/9fd3be74110e2cd02248e45677008f90c25cd1ddc8799f184af3f15f1e1982ba.jpg)

![](images/7b39a218c7926a0f5da3472ac08589e72f46827bac76658316532d5bfe262d0b.jpg)

![](images/1d7f4bdcd58d18e5f3bf60ee180faf49522a91d3103baed88798032b2e422bbd.jpg)

Figure 13. Qualitative gloss generations as a function of α.

# E CIFAR10H

CIFAR-10H extends standard object classification to visual categorization under human uncertainty. It provides full distributions of human labels for the entire 10,000-image CIFAR-10 test set, covering the same ten object categories and based on 511,000+ crowdsourced human judgments, corresponding to 51.1 judgments per image on average (range: 47–63). Rather than reducing each image to a single hard label, CIFAR-10H captures the full distribution of human categorizations, allowing evaluation of whether a model reflects not only the most likely class but also whether the model’s uncertainty across categories is aligned with human categorization behavior.

# E.1 Models and training protocol

In the original CIFAR-10H paper, the authors considered an explicit human-supervision setting in which pretrained models were fine-tuned on CIFAR-10H soft labels. Our goal here is different: we ask whether alignment with human uncertainty arises naturally from the representations learned by $\mathrm { J E M s } ,$ rather than being directly imposed through supervision on CIFAR-10H. Therefore, all models in our study were trained on standard CIFAR-10 Krizhevsky and Hinton [2009] and evaluated on CIFAR-10H only at test time.

We trained three JEM instances with different seeds for each value of α, using the same general procedure described in Appendix B.2. For the discriminative baselines, we trained VGG, ResNet, and ResNeXt models using the pytorch\_image\_classification codebase, matching the repository used by Peterson et al. Peterson et al. [2019]. Additionally, we included a ConvNeXt-Tiny baseline trained using timm, which was not part of the original CIFAR-10H study. For model selection, we held out 5% of the CIFAR-10 training set as a validation split. Moreover, we used random cropping and random horizontal flipping as the only data augmentations. We did not use label smoothing or any fine-tuning on CIFAR-10H in order to keep the comparison with the JEMs as fair as possible.

Across the discriminative baselines, we used Adam with learning rate $1 0 ^ { - 3 }$ . For JEMs, we used a smaller learning rate of 10−4, since larger values tended to destabilize the hybrid objective. Aside from this difference, baseline models were trained until convergence on the same CIFAR-10 setup used for the JEM experiments and were then evaluated on the CIFAR-10H test set using the metrics described below.

# E.2 Evaluation metrics

Following Peterson et al. Peterson et al. [2019], we use cross-entropy with respect to the human label distribution as our primary metric on CIFAR-10H. When the target is a human soft-label distribution $p _ { \mathrm { h u m } } ( y \mid x )$ , cross-entropy provides a principled measure of how well the model prediction $p _ { \theta } ( y \mid x )$ matches human categorization behavior under uncertainty. It is also more informative than top-1 accuracy in this setting, since accuracy alone does not capture how probability mass is distributed across incorrect classes.

We additionally report KL divergence from the human distribution to the model distribution as a complementary diagnostic of distributional mismatch. Because cross-entropy and KL divergence differ only by the entropy of the human label distribution, the two metrics are closely related. We use cross-entropy as the primary metric for consistency with Peterson et al. Peterson et al. [2019], and KL divergence as a secondary summary of alignment with human uncertainty.

For completeness, we also report top-1 accuracy on the same 10,000 test images. This allows us to distinguish conventional classification performance from alignment with human uncertainty: a model may achieve high top-1 accuracy while still assigning probability mass across categories in a way that differs substantially from human judgments. In the corresponding plots (Fig. 14), each point shows the mean across three seeds, and error bars denote standard error of the mean.

![](images/40504fc1bcbf55a5fbe0eb0b754550e3c5c38adcd216a0f738335b902f5a937c.jpg)

<details>
<summary>line</summary>

| α    | Top-1 accuracy (%) |
| ---- | ------------------ |
| 0.0  | 90.0               |
| 0.2  | 90.0               |
| 0.4  | 90.0               |
| 0.6  | 89.5               |
| 0.8  | 87.0               |
| 0.9  | 62.0               |
| 1.0  | 55.0               |
</details>

![](images/db02b89ec12eb45458b5a085fe2ffd87dc5977d92062eb46c87da738aa6adcb7.jpg)

<details>
<summary>line</summary>

| α    | Cross-entropy (Blue Line) | Cross-entropy (Red Line) |
| ---- | ------------------------- | ------------------------ |
| 0.0  | 1.25                      | -                        |
| 0.1  | 1.10                      | -                        |
| 0.2  | 1.10                      | -                        |
| 0.3  | 1.00                      | -                        |
| 0.4  | 0.95                      | -                        |
| 0.5  | 0.85                      | -                        |
| 0.6  | 0.80                      | -                        |
| 0.7  | 0.80                      | -                        |
| 0.8  | 0.65                      | -                        |
| 0.9  | -                         | 1.15                     |
| 1.0  | -                         | 1.30                     |
</details>

![](images/4d3b41347dfa7d146bc1c07750e4803dbfda1a4fcd93b209bf75e35d7a840710.jpg)

<details>
<summary>line</summary>

| α    | KL divergence |
| ---- | ------------- |
| 0.0  | 1.0           |
| 0.1  | 0.9           |
| 0.2  | 0.9           |
| 0.3  | 0.8           |
| 0.4  | 0.75          |
| 0.5  | 0.7           |
| 0.6  | 0.7           |
| 0.7  | 0.7           |
| 0.8  | 0.4           |
| 0.9  | 1.0           |
| 1.0  | 1.2           |
</details>

Figure 14. CIFAR-10 and CIFAR-10H evaluation across α.   
Figure 15. CIFAR-10 classification performance and alignment with CIFAR-10H human soft labels as a function of the generative–discriminative trade-off parameter α. Each point shows the mean across three seeds, and error bars denote the standard error of the mean (SEM).

# F Generalization benchmark

To evaluate model generalization, we use the Model-vs-Human benchmark suite introduced by Geirhos et al. [2021] . This suite consists of 17 out-of-distribution (OOD) datasets designed for ImageNet-trained models and evaluated on 16 categories derived from the same dataset. In addition, the benchmark provides human psychophysical data collected under controlled laboratory conditions. We describe further each dataset in Tables 1, 2, and show examples in Figures 16, 17.

The human dataset comprises 85,120 trials with 90 participants, who were asked to classify the transformed images. Geirhos et al. [2021] also evaluated a broad range of state-of-the-art ImageNet models on these benchmarks, including standard supervised classifiers, self-supervised models, vision transformers, adversarially trained models, and models trained on substantially larger datasets. This makes the benchmark effective for assessing not only OOD accuracy, but also the extent to which different models generalize in a human-like way.

Table 1. Noise generalization benchmarks. Parametric image manipulations from Geirhos et al. [2018b]. 

<table><tr><td>Benchmark</td><td>Levels / description</td></tr><tr><td>Greyscale</td><td>Colour vs. greyscale</td></tr><tr><td>Contrast</td><td>100, 50, 30, 15, 10, 5, 3, 1% contrast</td></tr><tr><td>High-pass</td><td> $\sigma = \infty, 3.0, 1.5, 1.0, 0.7, 0.55, 0.45, 0.4$ </td></tr><tr><td>Low-pass</td><td> $\sigma = 0, 1, 3, 5, 7, 10, 15, 40$ </td></tr><tr><td>Phase scrambling</td><td> $0^{\circ}, 30^{\circ}, 60^{\circ}, 90^{\circ}, 120^{\circ}, 150^{\circ}, 180^{\circ}$  phase noise</td></tr><tr><td>Power equalisation</td><td>Original vs. power-spectrum-equalised images</td></tr><tr><td>False colour</td><td>True-colour vs. opponent-colour images</td></tr><tr><td>Rotation</td><td> $0^{\circ}, 90^{\circ}, 180^{\circ}, 270^{\circ}$  rotations</td></tr><tr><td>Eidolon I</td><td>Coherence 10; reach varies with  $\log_{2}$ reach 0–7</td></tr><tr><td>Eidolon II</td><td>Coherence 3</td></tr><tr><td>Eidolon III</td><td>Coherence 0</td></tr><tr><td>Uniform noise</td><td>Width 0.0, .03, .05, .1, .2, .35, .6, .9</td></tr></table>

![](images/693e17ff088b6617a9cd94602173f279fdff4445e26bb426496477e24a9911ff.jpg)

Figure 16. Parametric transformations used in the Model-vs-Human benchmark: colour, contrast, frequency filtering, phase noise, power equalisation, opponent colour, rotation, eidolons, and uniform noise.   
Table 2. Texture–shape benchmarks. Nonparametric datasets from Geirhos et al. [2018a] and Wang et al. [2019]. 

<table><tr><td>Benchmark</td><td>Levels / description</td></tr><tr><td>Original</td><td>Clean reference photographs</td></tr><tr><td>Greyscale</td><td>Desaturated originals</td></tr><tr><td>Edge</td><td>Canny-edge line drawings</td></tr><tr><td>Silhouette</td><td>Black-on-white object silhouettes</td></tr><tr><td>Texture</td><td>Texture-only patches</td></tr><tr><td>Cue conflict</td><td>Stylized images with conflicting shape and texture cues</td></tr><tr><td>Stylized</td><td>Stylized-ImageNet images</td></tr><tr><td>Sketch</td><td>ImageNet-Sketch hand-drawn sketches</td></tr></table>

![](images/3cec99acd4daa9e2f885a9560b3d3ba82151f81e5c7114e0ea2ecada6571b3fd.jpg)

<details>
<summary>natural_image</summary>

Four-panel illustration showing a cat, a car, a chair, and a cat (no text or symbols)
</details>

Sketch

![](images/b53f997479468f0305e0e00cd94dcdf1b9dffec64969e066fbfa06bebbf2f6f7.jpg)

<details>
<summary>natural_image</summary>

Four-panel collage showing abstract scenes: a cat face, a parked car, a stone structure, and a colorful geometric pattern (no text or symbols)
</details>

Stylized

![](images/914260e81e31e8cc8fb7ac7e13e8e7eeddc77195fe7757c2db446c9d3185a349.jpg)

<details>
<summary>natural_image</summary>

Four-panel line drawing showing a person sitting on a chair, with abstract shapes in the corner (no text or symbols)
</details>

Edge

![](images/657d92bdf3f19918efc2b16b38af2059f69a8bacc9d4f5b022b6464dda9dd14e.jpg)

<details>
<summary>natural_image</summary>

Four-panel black silhouette image showing abstract shapes: a cat, a dog, a chair, and a fish-like shape (no text or symbols)
</details>

Silhouette

![](images/f7cacf1f8dc10cc249910cabe8f3c2e0ef1c42ce59cc31cc81c6fe9198565c73.jpg)

<details>
<summary>natural_image</summary>

Four-panel image showing abstract line drawings and plant-like shapes, no text or symbols present.
</details>

Cueconflict   
Figure 17. Nonparametric transformations used in the Model-vs-Human benchmark: sketch, stylized images, edge maps, silhouettes, and cue-conflict images.

# Evaluation metrics

We evaluated 11 ImageNet-trained JEMs, corresponding to values of α ranging from 0 to 1 in increments of 0.1, i.e., from purely discriminative to purely generative training. Following Geirhos et al. [2021], we assessed these models on the Model-vs-Human benchmark suite and report two summary metrics: OOD accuracy and error consistency (Figs. 18 & 19).

OOD accuracy. For each benchmark condition, we compute the model’s classification accuracy on the corresponding out-of-distribution images. Because the model is not trained on these image manipulations, this value measures OOD accuracy. In the main results, we report OOD accuracy averaged across benchmark conditions and datasets.

Error consistency. In addition to accuracy, we measure whether the model tends to make mistakes on the same images as human observers. We use error consistency, a chance-corrected agreement measure closely related to Cohen’s κ. This correction is important because two decision makers with high accuracy can show high raw agreement simply because both classify most images correctly. Error consistency, instead, asks whether the observed overlap in errors exceeds what would be expected from two independent decision makers with matched accuracies.

Formally, let D denote the set of datasets, $C _ { d }$ the set of conditions for dataset d, $H _ { d }$ the set of human participants evaluated on dataset $d ,$ and $S _ { d , c }$ the set of stimuli in condition c of dataset d. Let $b _ { h , m } ( s ) = 1$ indicate when human observer $h$ and model m give the same correctness outcome on stimulus s (that is, both are correct or both are incorrect), and $\bar { b } _ { h , m } ( s ) = 0$ otherwise. Let $\hat { o } _ { h , m } ( S _ { d , c } )$ denote the expected agreement under independent binomial decision makers with matched accuracies. The average error consistency is then

$$
E (m): \mathbb {R} \rightarrow [ - 1, 1 ], \quad m \mapsto \frac {1}{| D |} \sum_ {d \in D} \frac {1}{| H _ {d} |} \sum_ {h \in H _ {d}} \frac {1}{| C _ {d} |} \sum_ {c \in C _ {d}} \frac {\frac {1}{| S _ {d , c} |} \sum_ {s \in S _ {d , c}} b _ {h , m} (s) - \hat {o} _ {h , m} \left(S _ {d , c}\right)}{1 - \hat {o} _ {h , m} \left(S _ {d , c}\right)} \tag {9}
$$

where higher values indicate that the model’s pattern of errors is more similar to that of human observers than would be expected by chance.

![](images/b70a73a1c651ca878b92d2f32809e1f31ac31245349dd2c21d0b1fdecc74068e.jpg)

<details>
<summary>line</summary>

| α    | OOD accuracy [%] |
| ---- | ---------------- |
| 0.0  | 45.0             |
| 0.2  | 44.5             |
| 0.4  | 44.0             |
| 0.6  | 37.0             |
| 0.8  | 30.0             |
| 1.0  | 8.0              |
</details>

![](images/7a9c33b16c1c6f0462d80472b2f882ab24c7795afd716dfcb07b6e00b155a956.jpg)

<details>
<summary>line</summary>

| α    | Error consistency (k) |
| ---- | --------------------- |
| 0.0  | 0.12                  |
| 0.1  | 0.11                  |
| 0.2  | 0.12                  |
| 0.3  | 0.12                  |
| 0.4  | 0.13                  |
| 0.5  | 0.16                  |
| 0.6  | 0.15                  |
| 0.7  | 0.13                  |
| 0.8  | 0.11                  |
| 0.9  | 0.08                  |
| 1.0  | 0.01                  |
</details>

Figure 18. OOD accuracy and error consistency metrics, across JEM α values. Shaded regions indicate the standard error of the mean (SEM) across two seeds.

![](images/c9e1aeabe58f7f285d8d93326d5542cca30fb1bff0004bb0082274eab4d409f7.jpg)

<details>
<summary>line</summary>

| Colour    | Classification accuracy |
| --------- | ------------------------ |
| colour    | 0.95                     |
| greyscale | 0.85                     |
</details>

(a) Colour vs. greyscale

![](images/10ac48a1cf528666528c6383ef6adb2afb6a4523fac75c21bb5cb4c1f1c4430f.jpg)

<details>
<summary>line</summary>

| Colour    | Error consistency (kappa) |
| --------- | ------------------------- |
| colour    | 0.5                       |
| greyscale | 0.4                       |
</details>

![](images/497279c762c33ac5853293be03faec3f41af62896b13483adafac003923a7b8a.jpg)

<details>
<summary>line</summary>

| Colour   | Classification accuracy |
| -------- | ------------------------ |
| true     | 0.9                      |
| opponent | 0.7                      |
</details>

(b) True vs. false colour

![](images/f05fabaa26dd2681192eb3cff2f087ccec4687b511a3ab58844bab0c37c098dd.jpg)

<details>
<summary>line</summary>

| Colour   | Error consistency (kappa) |
| -------- | ------------------------- |
| true     | 0.4                       |
| opponent | 0.4                       |
</details>

![](images/8d0522e152f809adddb79482f233310fb1dcc0899faa7424e0349e23b2328e71.jpg)

<details>
<summary>line</summary>

| Uniform noise width | Classification accuracy (Line 1) | Classification accuracy (Line 2) | Classification accuracy (Line 3) |
| ------------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| 0.0                 | 0.85                              | 0.75                              | 0.45                              |
| 0.03                | 0.80                              | 0.70                              | 0.40                              |
| 0.05                | 0.75                              | 0.65                              | 0.35                              |
| 0.1                 | 0.70                              | 0.60                              | 0.30                              |
| 0.2                 | 0.65                              | 0.55                              | 0.25                              |
| 0.35                | 0.60                              | 0.50                              | 0.20                              |
| 0.6                 | 0.55                              | 0.45                              | 0.15                              |
| 0.9                 | 0.50                              | 0.40                              | 0.10                              |
</details>

(c) Uniform noise

![](images/04b492d1738280d282c645f4623431e90915c9f239bf5584a8562c0cd957014a.jpg)

<details>
<summary>line</summary>

| Uniform noise width | Error consistency (kappa) |
| ------------------- | ------------------------- |
| 0.0                 | 0.4                       |
| 0.3                 | 0.3                       |
| 0.5                 | 0.2                       |
| 1.0                 | 0.1                       |
| 2.0                 | 0.05                      |
| 3.5                 | 0.0                       |
| 6.0                 | -0.1                      |
| 9.0                 | -0.2                      |
</details>

![](images/ff66a7e20405480de7e782bf1d60ff39d767bb1ad8940afa4006c080be8c713e.jpg)

<details>
<summary>line</summary>

| Filter standard deviation | Classification accuracy (Line 1) | Classification accuracy (Line 2) | Classification accuracy (Line 3) | Classification accuracy (Line 4) |
| ------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| 0                         | 0.9                               | 0.7                               | 0.6                               | 0.3                               |
| 1                         | 0.85                              | 0.65                              | 0.55                              | 0.25                              |
| 5                         | 0.7                               | 0.5                               | 0.4                               | 0.15                              |
| 10                        | 0.6                               | 0.4                               | 0.3                               | 0.1                               |
| 40                        | 0.1                               | 0.1                               | 0.1                               | 0.1                               |
</details>

(d) Low-pass

![](images/416b5c11631423f97e95280fa8f1274ff73ec52ff43efd509a83ba1e56bb0ada.jpg)

<details>
<summary>line</summary>

| Filter standard deviation | Error consistency (kappa) |
| ------------------------- | ------------------------- |
| 0                         | 0.2                       |
| 1                         | 0.55                      |
| 3                         | 0.5                       |
| 5                         | 0.45                      |
| 7                         | 0.4                       |
| 10                        | 0.3                       |
| 15                        | 0.1                       |
| 40                        | 0.05                      |
</details>

![](images/f3a1b5159750599bffcd5aac68ecb03fddcc8d0790f17bf80447bb7de126071c.jpg)

<details>
<summary>line</summary>

| Contrast in percent | Classification accuracy (Line 1) | Classification accuracy (Line 2) | Classification accuracy (Line 3) | Classification accuracy (Line 4) | Classification accuracy (Line 5) | Classification accuracy (Line 6) | Classification accuracy (Line 7) |
| ------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| 100                 | 0.85                              | 0.80                              | 0.75                              | 0.70                              | 0.65                              | 0.60                              | 0.55                              |
| 50                  | 0.80                              | 0.75                              | 0.70                              | 0.65                              | 0.60                              | 0.55                              | 0.50                              |
| 15                  | 0.75                              | 0.70                              | 0.65                              | 0.60                              | 0.55                              | 0.50                              | 0.45                              |
| 5                   | 0.70                              | 0.65                              | 0.60                              | 0.55                              | 0.50                              | 0.45                              | 0.40                              |
| 1                   | 0.65                              | 0.60                              | 0.55                              | 0.50                              | 0.45                              | 0.40                              | 0.35                              |
</details>

(e) Contrast

![](images/0004e764d02f9a7a45e064ac6cac04c1e2758c246de2950ae5ebb93f2f3f3b70.jpg)

<details>
<summary>line</summary>

| Contrast in percent | Error consistency (kappa) |
| ------------------- | ------------------------- |
| 100                 | 0.4                       |
| 50                  | 0.3                       |
| 30                  | 0.4                       |
| 15                  | 0.3                       |
| 10                  | 0.2                       |
| 5                   | 0.4                       |
| 3                   | 0.1                       |
| 1                   | 0.0                       |
</details>

![](images/540b91e5e8ba8382cdaebcb63f8eff045ff7ac6583d2208518fa803c08e20b10.jpg)

<details>
<summary>line</summary>

| Filter standard deviation | Classification accuracy (Line 1) | Classification accuracy (Line 2) | Classification accuracy (Line 3) | Classification accuracy (Line 4) | Classification accuracy (Line 5) |
| ------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| inf                       | 0.8                               | 0.7                               | 0.6                               | 0.5                               | 0.4                               |
| 3.0                       | 0.7                               | 0.6                               | 0.5                               | 0.4                               | 0.3                               |
| 1.5                       | 0.6                               | 0.5                               | 0.4                               | 0.3                               | 0.2                               |
| 1.0                       | 0.5                               | 0.4                               | 0.3                               | 0.2                               | 0.1                               |
| .7                        | 0.4                               | 0.3                               | 0.2                               | 0.1                               | 0.05                              |
| .55                       | 0.3                               | 0.2                               | 0.1                               | 0.05                              | 0.02                              |
| .45                       | 0.2                               | 0.1                               | 0.05                              | 0.02                              | 0.01                              |
| .4                       | 0.1                               | 0.05                              | 0.02                              | 0.01                              | 0.005                             |
</details>

(f) High-pass

![](images/33df332c7b2fc3a4473557f3fe8f629504a1f67d2e919052bc30de920d2a0397.jpg)

<details>
<summary>line</summary>

| Filter standard deviation | Error consistency (kappa) |
| ------------------------- | -------------------------- |
| inf                       | 0.4                        |
| 3.0                       | 0.6                        |
| 1.5                       | 0.6                        |
| 1.0                       | 0.4                        |
| 0.7                       | 0.3                        |
| 0.5                       | 0.2                        |
| 0.4                       | 0.1                        |
</details>

![](images/3e77d560fa32caf715a014eae265de405d608fdf8929eeeeb75125d40ad2a6e7.jpg)

<details>
<summary>line</summary>

| Log₂ of 'reach' parameter | Classification accuracy (Line 1) | Classification accuracy (Line 2) | Classification accuracy (Line 3) | Classification accuracy (Line 4) | Classification accuracy (Line 5) | Classification accuracy (Line 6) | Classification accuracy (Line 7) |
| ------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- | --------------------------------- |
| 0                         | 0.8                               | 0.7                               | 0.6                               | 0.5                               | 0.4                               | 0.3                               | 0.2                               |
| 1                         | 0.9                               | 0.8                               | 0.7                               | 0.6                               | 0.5                               | 0.4                               | 0.3                               |
| 2                         | 0.85                              | 0.75                              | 0.65                              | 0.55                              | 0.45                              | 0.35                              | 0.25                              |
| 3                         | 0.8                               | 0.7                               | 0.6                               | 0.5                               | 0.4                               | 0.3                               | 0.2                               |
| 4                         | 0.7                               | 0.6                               | 0.5                               | 0.4                               | 0.3                               | 0.2                               | 0.15                              |
| 5                         | 0.6                               | 0.5                               | 0.4                               | 0.3                               | 0.2                               | 0.15                              | 0.1                               |
| 6                         | 0.4                               | 0.3                               | 0.2                               | 0.15                              | 0.1                               | 0.05                              | 0.05                              |
| 7                         | 0.1                               | 0.1                               | 0.1                               | 0.1                               | 0.1                               | 0.1                               | 0.1                               |
</details>

(g) Eidolon I

![](images/a3ca6227f722e11bc36051e19261ae5e64c7a78a02f91211c3340359c8314922.jpg)

<details>
<summary>line</summary>

| Log₂ of 'reach' parameter | Error consistency (kappa) |
| ------------------------- | ------------------------- |
| 0                         | 0.4                       |
| 1                         | 0.3                       |
| 2                         | 0.2                       |
| 3                         | 0.5                       |
| 4                         | 0.4                       |
| 5                         | 0.3                       |
| 6                         | 0.4                       |
| 7                         | 0.2                       |
</details>

![](images/a2cb35bff7ef23d826745b96824c8251e9a08527226deadf6fce177e9a19f81f.jpg)

<details>
<summary>line</summary>

| Phase noise width [°] | Classification accuracy (Line 1) | Classification accuracy (Line 2) | Classification accuracy (Line 3) | Classification accuracy (Line 4) | Classification accuracy (Line 5) | Classification accuracy (Line 6) | Classification accuracy (Line 7) |
| --------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- | ---------------------------------- |
| 0                     | 0.9                                | 0.8                                | 0.7                                | 0.6                                | 0.5                                | 0.4                                | 0.3                                |
| 30                    | 0.85                               | 0.75                               | 0.65                               | 0.55                               | 0.45                               | 0.35                               | 0.25                               |
| 60                    | 0.8                                | 0.7                                | 0.6                                | 0.5                                | 0.4                                | 0.3                                | 0.2                                |
| 90                    | 0.7                                | 0.6                                | 0.5                                | 0.4                                | 0.3                                | 0.2                                | 0.15                               |
| 120                   | 0.6                                | 0.5                                | 0.4                                | 0.3                                | 0.2                                | 0.15                               | 0.1                                |
| 150                   | 0.5                                | 0.4                                | 0.3                                | 0.2                                | 0.15                               | 0.1                                | 0.05                               |
| 180                   | 0.4                                | 0.3                                | 0.2                                | 0.15                               | 0.1                                | 0.05                               | 0.0                                |
</details>

(h) Phase noise

![](images/fcf01868adf9633047b3d6ba6ee5d56d4ed774d30e59bbaca6d90314622f26da.jpg)

<details>
<summary>line</summary>

| Phase noise width [°] | Error consistency (kappa) |
| --------------------- | -------------------------- |
| 0                     | 0.4                        |
| 30                    | 0.5                        |
| 60                    | 0.5                        |
| 90                    | 0.4                        |
| 120                   | 0.4                        |
| 150                   | 0.1                        |
| 180                   | 0.2                        |
</details>

![](images/c2f1b8b8bef4b1994862e95cc679ac44790fd7e1cfeac14f7161ef526e4e7e94.jpg)

<details>
<summary>line</summary>

| Log₂ of 'reach' parameter | Classification accuracy |
| ------------------------- | ------------------------ |
| 0                         | 0.8                      |
| 1                         | 0.7                      |
| 2                         | 0.6                      |
| 3                         | 0.5                      |
| 4                         | 0.3                      |
| 5                         | 0.1                      |
| 6                         | 0.05                     |
| 7                         | 0.0                      |
</details>

(i) Eidolon II

![](images/e5d1a096e14a27a8c006e039ce0089401f1fdf3b96e67725940b1759164e6a41.jpg)

<details>
<summary>line</summary>

| Log₂ of 'reach' parameter | Error consistency (kappa) |
| ------------------------- | -------------------------- |
| 0                         | 0.4                        |
| 1                         | 0.6                        |
| 2                         | 0.5                        |
| 3                         | 0.4                        |
| 4                         | 0.3                        |
| 5                         | 0.2                        |
| 6                         | 0.1                        |
| 7                         | 0.0                        |
</details>

![](images/c5f8435473b016fad3b1d5bb283ecb97e616f4b17365d36fa67f4a1f24727fc8.jpg)

<details>
<summary>line</summary>

| Power spectrum | Classification accuracy |
| -------------- | ------------------------ |
| original       | 0.8                      |
| equalised      | 0.3                      |
</details>

(j) Power equalisation

![](images/c58f6da5bd038144deeda9ca8b5f74fd0aa39ee58fc14a8d9ad18259123c58f1.jpg)

<details>
<summary>line</summary>

| Power spectrum | Error consistency (kappa) |
| -------------- | ------------------------- |
| original       | 0.2                       |
| equalised      | 0.1                       |
</details>

1.0  1.0 y  1.0 Figure 19. OOD accuracy and error consistency for each benchmark individually.

# G Shape—texture cue conflict benchmark

Geirhos et al. [2018a] created a cue-conflicting images dataset as shown in Fig.3, where the outline is from one Imagenet category, and the texture is from another one. This allowed us to quantify across humans and models, which was the most important factor in making a classification decision. Traditional CNNs have been shown to be biased toward texture, whereas generative models are much more shape-leaning. Humans evaluated on this task are heavily biased towards shape/contours. We can also clearly see from Fig. 20 that this task is object-dependent and varies significantly across objects. Another thing to note is that the purely generative JEM seems all over the place; this is because the classification head was never trained and is randomly initialized.

![](images/76ead67f3ad419f5ac3679d85d38cd0e1dd1064f34a87616575b25c4b37c0e64.jpg)  
Figure 20. Percentage of shape or texture choice made per category for each model .

One advantage of JEMs is that they permit refinement of the input through MCMC during testing, pushing the image toward regions of higher probability. This can be interpreted as allowing the model to “think longer” and move away from uncertain or ambiguous states. As a result, models with small α refine images toward textures whereas in models with larger α, shape bias becomes even stronger after a few MCMC steps. To illustrate this effect, we visualize the evolution of a representative cue-conflict image at 0, 5, 10, and 20 MCMC steps (Fig. 21).

![](images/2e5a7e1ff2824cbd0afb083e47e63c2b35132c64a06ee21a1b8c5f2b0a4e547e.jpg)

<details>
<summary>text_image</summary>

original
5 steps
α=0.0	α=0.1	α=0.2	α=0.3	α=0.4	α=0.5	α=0.6	α=0.7	α=0.8	α=0.9	α=1.0
10 steps
20 steps
</details>

Figure 21. Evolution of an image dependent on the alpha and the number of MCMC steps.

![](images/524d91235e74973caa0c5f6994ab5d11e7d959bae0c89450ea9001d622ea9600.jpg)

<details>
<summary>line</summary>

| α    | Shape bias |
| ---- | ---------- |
| 0.0  | 0.46       |
| 0.1  | 0.47       |
| 0.2  | 0.46       |
| 0.3  | 0.48       |
| 0.4  | 0.53       |
| 0.5  | 0.63       |
| 0.6  | 0.67       |
| 0.7  | 0.70       |
| 0.8  | 0.73       |
| 0.9  | 0.73       |
| 1.0  | 0.63       |
</details>

Figure 22. Shape bias across JEM α values. Shaded regions indicate the standard error of the mean (SEM) across two seeds.

# H The Click-Me benchmark

Modern CNNs achieve high performance on object-recognition benchmarks, but they are also known to rely on shortcut cues that can diverge from the diagnostic features used by human observers. To assess this aspect of alignment, we use the ClickMe dataset introduced by Linsley et al. [2018], which provides large-scale human feature-importance maps for ImageNet images (Fig. 23). In this task, participants highlight the image regions they consider most informative for determining the object category, yielding a direct measure of the visual evidence humans rely on for recognition.

![](images/a4759ef19d4738f0837e4e39440b3ed8f633f206bbe1a2119ade60f96df20a19.jpg)

<details>
<summary>natural_image</summary>

Collage of 20 diverse images showing various scenes including a person in traditional attire, a cat face with green glow, bird and deer, and a wine bottle (no visible text or symbols)
</details>

Figure 23. Examples of human feature importance maps.

![](images/5949782828fa3cb5c45ca8304374fb93a565a4b134c4eca31aa8a776771cdc6d.jpg)  
Figure 24. Visual strategy of object recognition.

Evaluation metrics . To compare models with humans, we follow the evaluation protocol of Fel et al. [2022b]. For each model, saliency maps are computed on the ClickMe images and compared with the corresponding human feature-importance maps, yielding a quantitative measure of feature alignment (Fig. 24). In our case, however, the classifier is built on top of a pretrained VAE, and we do not want gradients through the VAE to dominate the attribution signal. We therefore adopt the approach of Boutin et al. [2024], which uses the VAE decoder’s Jacobian to propagate saliency from the latent space back to pixel space.

We call $p _ { \psi } ( x | z )$ the decoder of the VAE, and $p _ { \theta } ( e | z )$ the energy density learnt by the JEM. To make the mathematical derivations more concise, we define the following functions:

$$
p _ {\psi}: \mathbb {R} ^ {d} \longrightarrow \mathbb {R} ^ {D} \quad \text { and } \quad p _ {\theta}: \mathbb {R} ^ {d} \longrightarrow \mathbb {R} \tag {10}
$$

$$
z \longmapsto x = \log p _ {\psi} (\cdot | z) \quad z \longmapsto e = \log p _ {\theta} (\cdot | z) \tag {11}
$$

To project each energy value e into the pixel space, we feed them into the decoder. The resulting projection is $x = p _ { \psi , \theta } ( z ) = p _ { \psi } \circ p _ { \theta } ( z )$ .

For each energy value, the importance feature map quantifies how the absolute value of $p _ { \psi , \theta } ( e )$ changes as z varies. $\phi ( x )$ describes the accumulation, of these “local feature maps”:

$$
\phi (x) = \frac {\partial p _ {\psi , \theta} (e)}{\partial x} \tag {12}
$$

$$
= \frac {\partial p _ {\psi} \circ p _ {\theta} (e)}{\partial x} \tag {13}
$$

$$
= \frac {\partial p _ {\psi}}{\partial x} (p _ {\theta} (z)) \frac {\partial p _ {\theta}}{\partial z} (e) \tag {14}
$$

$$
= J _ {p _ {\psi}} (x) \nabla_ {z} p _ {\theta} (z) \tag {15}
$$

with $J _ { p _ { \psi } } ( x )$ being the Jacobian of the function $p _ { \psi }$ w.r.t. x computed in $p _ { \theta } ( z )$ .

![](images/5b7cd6ee0bd16e2ac0d786ca1ac71ed228bf2f707d5bee0900eb3edc8c4da912.jpg)

<details>
<summary>line</summary>

| α    | ClickMe alignment score |
| ---- | ------------------------ |
| 0.0  | 0.32                     |
| 0.1  | 0.31                     |
| 0.2  | 0.32                     |
| 0.3  | 0.33                     |
| 0.4  | 0.32                     |
| 0.5  | 0.31                     |
| 0.6  | 0.27                     |
| 0.7  | 0.24                     |
| 0.8  | 0.21                     |
| 0.9  | 0.15                     |
| 1.0  | 0.08                     |
</details>

Figure 25. ClickMe alignment score across JEM α values. Shaded regions indicate the standard error of the mean (SEM) across two seeds.