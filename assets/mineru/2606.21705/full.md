# Structural Assessment for Understanding and Guiding Dataset Distillation in Discrete Token Space

Yue Cao<sup>1</sup>, Jianyang Gu<sup>2</sup>, Vyacheslav Kungurtsev<sup>3</sup>, Yu Hu<sup>1</sup>, Jozsef Hamari<sup>4</sup>, Zheng Liu<sup>1∗</sup>, and Mohsen Zardadi<sup>4∗</sup>

<sup>1</sup> The University of British Columbia

<sup>2</sup> The Ohio State University

3 Czech Technical University in Prague

4 TerraSense Analytics

{yue.cao, zheng.liu}@ubc.ca, mohsen.zardadi@terrasense.ca

Abstract. Dataset distillation (DD) has proven to reduce training cost while preserving accuracy. While promising, the factors that make one distilled dataset more efective than another remain poorly understood. In this work, we investigate this question through the lens of discrete visual tokenizers. Whereas many prior DD eforts emphasize matching global data distributions, we suggest that the efectiveness depends on which semantic concepts are captured and how they are composed. Discrete visual tokenizers provide a finite vocabulary that enables direct statistical analysis of such compositional structure. Through quantitative analysis of token-level statistics, we introduce the structural score to measure the adequacy of token compositions. We observe that distilled datasets with balanced token composition yield higher validation performance. On the other hand, divergence from the original data does not necessarily harm performance. We further show that samples with high structural scores in the discrete token space can efectively guide difusion-based DD. Our findings highlight the importance of token composition in dataset efectiveness, ofering a principled complement to distributional similarity considerations in DD.

Keywords: Dataset Distillation · Eficient Machine Learning

## 1 Introduction

Dataset distillation aims to replace a large training set with a surrogate while preserving training performance [19, 43, 45]. Despite rapid empirical progress, what makes a good distilled dataset remains poorly understood [23, 24]. Many prior eforts enforce surrogates to match the continuous distributions of real data via feature matching [22,47,52], optimal transport [21], or gradient alignment [1,48]. However, such distributional proximity alone does not indicate semantic structures or discriminate variations informative for learning. A surrogate may be

(a)

![](images/442517d03ae5ea4912b863354062556aa0c9be0aa443457984c3972e4d021144.jpg)

![](images/ff96ce63c2f539115818ae42947a88ee9a92df4606d351701bf291437e21e738.jpg)  
(b)  
Fig. 1: The discrete token distribution of airplane images in standard and remote sensing imagery. (a) The airplanes frequently use similar top tokens across datasets. (b) At the same time, they exhibit a large divergence (JSD=0.47) in their overall token distribution due to structural diferences.

close to real data in embedding space yet still collapse rare concepts, overrepresent trivial structures, or fail to span discriminative variations. In practice, the dataset most similar to the original is not always the most efective for training. This observation raises an open question: what properties of a surrogate, beyond distributional similarity, ultimately govern its training efectiveness?

We investigate this question through the lens of discrete visual tokenizers. Unlike high-dimensional continuous feature spaces, where semantics are heavily entangled, discrete tokenizers map each image into a sequence of indices from a finite vocabulary. This enables direct statistical analysis of how visual primitives are composed within a dataset. As illustrated in Figure 1, we use VQ-VAE [36, 39] to quantize airplane images from ImageNet [7] and remote sensing [5] into visual tokens. Although these two domains difer substantially in viewpoint, background, and style, they share a subset of high-frequency tokens that likely correspond to common airplane-related visual primitives. At the same time, their overall token distributions diverge due to the diferences in context and domain-specific artifacts. This motivates us to move beyond matching global distributions and instead explicitly characterize token structure within a dataset.

To formalize this assessment, we represent each image as a discrete distribution over the shared visual vocabulary and quantify this representation through three complementary metrics: (1) Jensen–Shannon Divergence (JSD) [20] between an image’s token distribution and the average distribution of its corresponding class. Lower JSD indicates samples with more representative compositional patterns. (2) Herfindahl–Hirschman Index (HHI) [15] over the token distribution, where a lower HHI reflects more balanced usage of codebook tokens. (3) Coverage rate (COV) over class-discriminative tokens identified from the real dataset, where higher coverage reflects stronger preservation of class-relevant primitives. Together, these metrics provide a holistic structural signature of each image. We then relate the aggregated statistics of surrogates generated by diverse DD methods to their validation accuracy via regression analysis. Empirically, the combination of these three metrics, named the structural score, provides a solid prediction of validation accuracy. The resulting coeficients reveal that more balanced token composition primarily leads to higher accuracy. Intriguingly, merely minimizing token-level divergence from the original set does not reliably correlate with better performance.

Building on these insights, we move from diagnosis to design and explore how token-level statistics can guide surrogate generation. Unlike prior methods that rely on continuous feature centroids to guide difusion [3, 34], we cluster data in the discrete token space and rank samples based on the structural score to acquire optimal guidance signals. Using the same steering technique as prior methods, the Token-Guided Dataset Distillation (TGDD) method demonstrates more efective information integration across diverse difusion backbones and benchmarks. On ImageWoof, TGDD yields a 5.4% improvement compared to the state-of-the-art baseline [3].

To summarize, our work presents two contributions: (1) We introduce a training-free structural score that reliably assesses the quality of the distilled dataset in various DD methods and domains. (2) We develop a token-guided distillation framework that leverages this score to actively guide surrogate generation, proving the efectiveness of the discrete structural assessment.

## 2 Related Work

## 2.1 Dataset Distillation

Dataset distillation aims to synthesize a surrogate dataset such that training a model on it yields comparable performance to training on the full original dataset [19, 45]. Among previous eforts, optimization-based methods directly update surrogate images to mimic the training dynamics of the original dataset. [1, 16, 38, 48] align the gradient or training trajectory, and [6, 21, 35, 41, 44, 46] align feature or distribution statistics. As optimization-based methods typically rely on a teacher model for supervision, their generalization across architectures is constrained. Another line of work synthesizes datasets using generative priors. For example, GLaD [2], D2M [31], and H-PD [51] leverage pretrained GANs to facilitate the optimization of the dataset. More recent works employ difusion and visual autoregressive models as the backbone for DD, formulating synthesis as a guided generation process [11, 12, 34, 49, 50]. $\mathrm { M G D ^ { 3 } }$ [3] and VLCP [53] improve intra-class diversity through mode guidance and vision–language prototypes. IGD [4] and $\mathrm { C a O _ { 2 } }$ [40] further introduce trajectory-influence and consistencybased guidance. Although representativeness, informativeness, and diversity are widely acknowledged to be important, prior work often relies on heuristic similarity scores as proxies. In this work, we aim to interpret datasets as compositions of visual concepts in certain contexts. Thereby, we turn the assessment of distilled data from distributional similarity to the structural score.

## 2.2 Feature-based Data Selection Criteria

To assess the usefulness of features and data samples, prior work employs a range of statistical and information-theoretic measures. The Herfindahl Hirschman Index (HHI) [15], Shannon entropy [33], and the Gini coeficient [10] are used to characterize the concentration and diversity of feature distributions. To compare probability distributions induced by features or by selected subsets of data, Jensen–Shannon divergence and Kullback–Leibler divergence are widely used [18,20]. For feature specificity, term frequency–inverse document frequency [32] highlights rare yet discriminative patterns. Collectively, these metrics provide quantitative tools for evaluating informativeness, coverage, and redundancy, which motivate the token-level criteria used in our framework.

## 3 Structural Assessment for Dataset Distillation

As motivated in §1, beyond overall proximity to the original data, we seek a more fine-grained understanding of surrogate quality. To this end, we propose explicitly assessing the structure of visual primitives by mapping continuous images into a discrete token space. Statistics over a finite vocabulary enable analyzing how these tokens are used at the dataset level. Building on this framework, we introduce the structural score as a robust signature of dataset quality to quantitatively evaluate the distilled surrogate.

## 3.1 Structural Representation in Token Space

We first employ a discrete visual tokenizer to map continuous images into a sequence of indices from a finite token vocabulary. In recent years, there have been extensive eforts in developing efective visual tokenizers [8, 27, 36]. This tokenizing step does not rely on a specific tokenizer type, and can be broadly applied. Without loss of generality, we first demonstrate an instantiation of the assessment with a multi-scale VQ-VAE [36] model.

Given a VQ-VAE codebook of size V and the encoder with L scales, each surrogate image is embedded with the encoder into discrete token maps $\{ z _ { i } ^ { ( 1 ) } , z _ { i } ^ { ( 2 ) }$ ， $\dots , z _ { i } ^ { ( L ) } \}$ , where $\boldsymbol { z } _ { i } ^ { ( \ell ) }$ contains indices from the codebook at the ℓ-th scale of the total L scales. For each scale, token occurrences are counted and normalized to form a probability vector over the codebook:

$$
\boldsymbol {p} _ {i} ^ {(\ell)} \in \mathbb {R} ^ {V}, \quad \sum_ {k = 1} ^ {V} p _ {i} ^ {(\ell)} (k) = 1,
$$

where $p _ { i } ^ { ( \ell ) } ( k )$ denotes the relative frequency of the k-th token in image $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ at scale ℓ. While VQ-VAE tokens on diferent scales capture distinct visual information, analyzing them separately would yield fragmented scores that are dificult to reconcile. To synthesize these signals into a coherent statistical signature, we aggregate the distributions across all scales. Since the number of tokens grows quadratically with scale, a direct summation causes an inherent density imbalance. Therefore, we fuse the token distributions using a weighted sum:

$$
\boldsymbol {p} _ {i} = \sum_ {\ell = 1} ^ {L} w _ {\ell} \boldsymbol {p} _ {i} ^ {(\ell)},
$$

where $\pmb { p } _ { i }$ represents the fused token prior. We group the $L = 1 0$ scales into low (1–3), mid (4–7), and high (8–10) resolution bands. Intuitively, we assign them relative weights of [3, 1, 0.5]. This strategy efectively upweights the lowresolution bands to compensate for their sparsity. The validation of this weighting scheme is provided in Appendix §C.1.

Similar to words, each token in the codebook may convey a series of semantics. The semantics are not directly interpretable without being incorporated into a composition. We then examine the utilization structure of diferent visual tokens in addition to their presence. Specifically, we look into three properties of the distilled dataset that characterize its token-level composition:

Contextual fit. The surrogate dataset is expected to use a composition of tokens similar to the original dataset to reflect the corresponding contexts. Jensen– Shannon Divergence (JSD) [20] is incorporated to measure the contextual fit:

$$
\operatorname{JSD} \left(\boldsymbol {p} _ {i}, \boldsymbol {\mu} _ {c}\right) = \frac {1}{2} \left(\operatorname{KL} \left(\boldsymbol {p} _ {i} \| \boldsymbol {m}\right) + \operatorname{KL} \left(\boldsymbol {\mu} _ {c} \| \boldsymbol {m}\right)\right),
$$

where $c$ is the class that $\mathbf { \Delta } _ { \mathbf { \mathcal { X } } _ { i } }$ belongs to, and $\pmb { \mu } _ { c }$ is the average token distribution of the corresponding class, $i . e .$ , the class centroid. $\operatorname { K L } ( \cdot \| \cdot )$ calculates the KL divergence between two distributions, and m is the mixture distribution of $\pmb { p } _ { i }$ and $\pmb { \mu } _ { c }$ . Lower JSD scores indicate that the samples have more typical compositional patterns of that class.

Compositional richness. The surrogate should convey meanings by combining a broad set of distinct primitives rather than reusing a few. Herfindahl-Hirschman Index (HHI) [15] is employed for this measurement:

$$
\operatorname{HHI} \left(\boldsymbol {p} _ {i}\right) = \sum_ {k = 1} ^ {V} \left(p _ {i} (k)\right) ^ {2},
$$

where $k \in \{ 1 , \ldots , V \}$ indexes codebook tokens, and $p _ { i } ( k )$ is the k-th entry of the fused per-image token distribution $\pmb { p } _ { i }$ . Smaller HHI values correspond to more balanced token usage and thus richer visual information within the image.

Categorical presence. The usage of tokens should carry important class characteristics. We use the coverage rate of class-related tokens identified by TF-IDF [32] that appear in the original dataset:

$$
\operatorname{COV} \left(\boldsymbol {p} _ {i}\right) = \sum_ {k \in \mathcal {T} _ {c}} p _ {i} (k),
$$

where ${ \mathcal { T } } _ { c } \subseteq \{ 1 , \ldots , V \}$ denotes the set of top TF-IDF tokens for class $c .$ Higher coverage indicates that the sample uses more class-relevant visual primitives.

These three metrics reflect complementary properties of datasets. Note that the structural assessment can also be implemented with other discrete visual tokenizers, such as VQGAN [8] and BEiTv2 [27]. We then investigate the impact of these three properties on the validation performance.

## 3.2 Empirical Results

We collect surrogates for ImageWoof generated by a variety of dataset distillation methods, including both optimization-based (DM [47], SRe<sup>2</sup>L [44]), selectionbased (RDED [35]), and generative-based approaches (MGD<sup>3</sup> [3], Minimax [11], VAR [36], Stable Difusion [28]). For each method except SRe<sup>2</sup>L, we collect multiple surrogates from at least 5 runs to reduce randomness. We compute the token statistics of each surrogate with respect to the original ImageWoof [9] training set, where each metric is computed by averaging the scores of all individual samples within the distilled dataset. We also train a ResNet-10 [13] classifier from scratch on each surrogate to obtain their validation accuracy.

We fit a linear regression model with a Lasso regularization coeficient of 0.5 that maps the token statistics to the resulting validation accuracy. As shown in Figure 2, the predicted validation accuracy exhibits a small mean-squared error (MSE) from the ground-truth accuracy. This strong correlation indicates that the token statistics provide essential information for assessing the quality of surrogates.

![](images/6bc2cfa9dba67f21f0d10ce7be52b26b52936d907b5b0348ca1723dceaad6f42.jpg)  
Fig. 2: Structural score versus ground-truth validation accuracy on ImageWoof across distilled datasets generated by multiple methods. We report the mean-squared error (MSE) over all data points. Note that TGDD (Ours) is only used for test but not for linear regression.

We further ablate this linear regression with diferent groups of metrics, and the results are shown in Figure 3. When only one metric is adopted to fit the validation

accuracy, HHI illustrates a smaller MSE compared with JSD and COV. Combining two metrics yields a smaller MSE compared with each of them alone. The best explanation is produced from the combination of all three metrics, which is our proposed assessment in Figure 2. The acquired coeficients for each metric and the intercept $\beta$ are:

$$
w _ {\mathrm{JSD}} = 2. 9 6, w _ {\mathrm{HHI}} = - 1 0. 9 9, w _ {\mathrm{COV}} = 2. 0 7, \beta = 4 7. 7 5.
$$

While they cannot be evaluated in isolation, the validation performance tends to have a strong correlation with lower HHI values, $i . e .$ , more balanced token compositions. We refer to this combination of statistics as the structural score, which provides more comprehensive assessments of distilled datasets compared with the distributional similarity alone.

![](images/b5b847f6e7d64ec644124cfe01c20d7331abb354e2d21a4bba316f780e7ba452.jpg)  
Fig. 3: The ablation study of diferent combinations of token statistics (HHI, JSD, COV) in predicting model accuracy via linear regression. Each subplot shows the relationship between the adopted token statistics and model accuracy across various distilled datasets, with Mean Squared Error (MSE) reported.

We list the token statistics and validation accuracy of surrogates distilled by Stable Difusion [28], DM [47], RDED [35], and $\mathrm { M G D ^ { 3 } }$ [3] in Table 1 for a more detailed case study. Aligned with the coeficient, HHI has

Table 1: Detailed metric values and validation accuracy of 50-IPC surrogates for ImageWoof.

<table><tr><td>Method</td><td>JSD</td><td>HHI</td><td>COV</td><td>Accuracy</td></tr><tr><td>Stable Diffusion [28]</td><td>0.122</td><td> $4.608 \times 10^{-3}$ </td><td>0.299</td><td>31.2</td></tr><tr><td>DM [47]</td><td>0.186</td><td> $3.334 \times 10^{-3}$ </td><td>0.207</td><td>42.2</td></tr><tr><td>RDED [35]</td><td>0.136</td><td> $2.616 \times 10^{-3}$ </td><td>0.228</td><td>47.2</td></tr><tr><td>MGD $^{3}$ [3]</td><td>0.077</td><td> $1.185 \times 10^{-3}$ </td><td>0.261</td><td>55.8</td></tr></table>

the strongest influence on the validation accuracy, while the other two metrics cannot lead to a strong correlation by themselves. Notably, DM and RDED illustrate higher JSD compared with the images generated by Stable Difusion, but still yield higher accuracy. This result corresponds to our claim that the distributional similarity alone cannot indicate the data efectiveness.

Generalizability of the scoring model. With the initial metric coeficients acquired on ImageWoof, we further conduct the same linear regression with surrogates from ImageNette [9]. The obtained weights for the three metrics are: $w _ { \mathrm { J S D } } = 2 . 4 3 .$ 2 $w _ { \mathrm { H H I } } = - 9 . 5 9$ 2 $w _ { \mathrm { C O V } } = 1 . 4 1$ . While the absolute numbers vary, the relative importance of the metrics remains consistent, with HHI suggesting the most significant influence on the validation performance. This result confirms the efectiveness of the derived conclusions from the structural assessments.

Direct application to other domains. Although the relative relationship stands across datasets, it is infeasible to calculate the coeficients for each new application. Therefore, we demonstrate the direct generalization of the acquired coeficients on remote sensing imagery [14]. We employ Distribution Matching (DM), an optimization-based method, to perform the dataset distillation for

Table 2: Application of the proposed structural score to the EuroSAT dataset, demonstrating that our metric generalizes to visually distinct domains without retraining.

<table><tr><td>Iters</td><td>0</td><td>200</td><td>400</td><td>600</td><td>800</td><td>1000</td><td>1200</td><td>1400</td><td>1600</td><td>1800</td></tr><tr><td>Structural Score</td><td>-76.7</td><td>-34.2</td><td>-22.1</td><td>-15.5</td><td>-13.7</td><td>-11.1</td><td>-9.9</td><td>-4.9</td><td>-1.6</td><td>2.2</td></tr><tr><td>Accuracy (%)</td><td>11.5</td><td>40.6</td><td>44.9</td><td>47.1</td><td>47.8</td><td>51.7</td><td>53.0</td><td>55.8</td><td>57.2</td><td>61.1</td></tr></table>

![](images/43d66a6c37163505ce742a258d0778a25c366cc13615f8604184bb4ba6257612.jpg)  
Fig. 4: Qualitative comparison of images with high and low values of JSD, COV, and HHI metrics. Each row corresponds to one metric, with images on the left exhibiting high values and those on the right showing low values.

EuroSAT. During the distillation process, we collect 10 surrogate datasets at diferent timesteps. The structural score and corresponding validation accuracy of each surrogate are reported in Table 2. As the images are progressively updated, the structural scores increase monotonically alongside the downstream validation accuracy. Due to the large domain gap, the predicted score does not equal the accuracy. However, it is still able to compare the efectiveness of different surrogates based on relative relationships. This trend confirms that our token-level metrics are robust indicators of dataset quality, even when applied to domains significantly diferent from ImageNet.

## 3.3 Example Visualization

We compare representative image groups with high and low metric values to better understand how each token statistics in the structural assessment reflects dataset quality, as illustrated in Figure 4. We select examples generated by MGD<sup>3</sup> [3] and Stable Difusion [28], and analyze how each metric correlates with visual quality. For JSD, both sets of images are generated by MGD<sup>3</sup>, and the JSD is computed by comparing each image’s token distribution with the average token distribution of the corresponding class in the original ImageWoof dataset. Images with higher JSD values tend to exhibit unrealistic features, such as disordered object shapes or incorrect combinations of semantic elements, making them deviate from the original class distribution. For the coverage metric, lower values are often associated with images where the target object occupies a smaller region of the image or is missing entirely. This indicates a weaker representation of class-specific content. Finally, we visualize images based on their HHI values. We use samples generated by MGD<sup>3</sup> for the low-HHI group, which generally exhibit higher diversity. By contrast, samples generated by Stable Diffusion show similar object poses and more repetitive patterns, reflected by higher HHI values.

## 4 Token-Guided Dataset Distillation

The preceding analysis demonstrates that token statistics reliably indicate the quality of distilled datasets. Building on this insight, we further show that these statistics can also be incorporated to guide difusion denoising. Following [3], we cluster each class to identify representative modes, but based on the token distribution instead of continuous features. We then rank and select anchors from each cluster using the proposed structural score, and use the selected anchors to guide a difusion model in generating synthetic samples as in [3].

## 4.1 Anchor Selection

Given a real dataset $\mathcal { T } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N _ { T } }$ , the discrete visual tokenizer is employed to embed each image to capture the underlying distribution of visual concepts $\{ p _ { i } \}$ . For numerical stability, principal component analysis (PCA) is first applied to reduce the token space to d dimensions, and the projected vectors are L2- normalized. This normalization makes Euclidean distances a monotonic transformation of cosine similarities, ensuring that clustering captures diferences in token composition rather than in feature magnitude. k-means is then run on the normalized vectors to obtain $K _ { c }$ clusters, where $K _ { c }$ is set by the IPC target. This step is the same as that in [3], except in the discrete token space, where distances reflect composition diferences over visual concepts. Therefore, each cluster corresponds to a pattern of token use within the class.

Token-level clustering reveals intra-class token modes but leaves many candidates per mode. Instead of using the centroid as in [3], we select a small number of anchors from each cluster based on the token statistics identified in §3.1. Specifically, we use the linear regression model in §3.2 to predict a structure score for each sample. The only diference from the previous JSD calculation is that samples are compared with cluster centers instead of class centers. The top M images with the highest scores are selected as anchors. We set $M = 2 0$ anchors per cluster for $\mathrm { I P C } < 5 0 $ and $M = 1 0$ for higher IPC settings. We summarize the anchor selection procedure in algorithm 1 of Appendix $\ S \mathrm { A }$

## 4.2 Token-Guided Generation

The selected anchors from each cluster are used to guide synthetic data generation. We average the anchor embeddings of one cluster into one latent, representing the corresponding mode. Following the denoising procedure of [3], we guide the denoising process with the acquired modes:

$$
\pmb {z} _ {t - 1} = \pmb {z} _ {t - 1} ^ {\mathrm{base}} + \mathbb {1} _ {\{t > t _ {\mathrm{stop}} \}} \lambda (\bar {\pmb {z}} _ {c, m} - \hat {\pmb {z}} _ {0}),
$$

where λ denotes the mode-guidance strength, $z _ { t }$ is the latent variable at step t, and $\hat { z } _ { 0 }$ is the predicted clean latent. The guidance is applied only for timesteps $t > t _ { \mathrm { s t o p } }$ to maintain sample diversity. We demonstrate that by simply summarizing modes in the discrete token space with the help of token statistics, the acquired guidance can be more efective than that of continuous cluster centroids. Moreover, in §5.3, we demonstrate that the method can be applied to a broad range of discrete tokenizers and difusion architectures to provide more efective guidance than continuous models.

## 5 Experiments

## 5.1 Implementation Details

For the visual tokenizer, we adopt the multi-scale VQ-VAE model developed by [36], while the difusion generation framework employs a pre-trained DiT [26]. Following the literature [3, 26], we use 256×256 image resolution with 50 sampling steps and apply stop guidance at step 25. All experiments are conducted on a single RTX 3080 GPU. Further implementation details are provided in Appendix §B.

Datasets. We evaluate our method on multiple high-resolution benchmarks with 256×256 images to thoroughly validate its performance. The dataset used in this experiment includes ImageNette, ImageWoof [9], ImageIDC [16], ImageNet-100 [37], and ImageNet-1k [7]. ImageNette and ImageIDC each contain 10 classes, and ImageWoof comprises 10 diferent dog breeds, making it particularly challenging due to high inter-class similarity. For evaluation, we adopt the same metrics as [3, 11].

## 5.2 Comparison with State-of-the-art Methods

We compare our method against multiple state-of-the-art baselines using a consistent evaluation protocol. The baselines include both generative-based methods: Minimax Difusion [11], DiT [26], and MGD<sup>3</sup> [3], and optimization-based methods: DM [47], RDED [35], SRe<sup>2</sup>L [44] and IDC [16]. All methods are evaluated across multiple images-per-class (IPC) settings and classification architectures to validate the generalizability.

ImageWoof is selected for initial analysis due to its fine-grained classification structure, where all classes correspond to diferent dog breeds. The high visual similarity among these classes requires models to rely on subtle, localized features for accurate discrimination. In such scenarios, methods that focus solely on global distribution alignment often struggle to capture meaningful intra-class variation, leading to trivial or overlapping representations. TGDD addresses this challenge by leveraging discrete token statistics to identify structurally informative anchors, ofering complementary perspectives on dataset composition. As shown in Table 3, TGDD consistently achieves the best classification accuracy across nearly all configurations. Furthermore, we show in Figure 2 that the tokenlevel statistics of TGDD surrogate accurately predict its validation accuracy via the structural score.

Table 3: Comparison of performance across state-of-the-art methods on ImageWoof. The best results are marked in bold.

<table><tr><td>IPC (Ratio)</td><td>Test Model</td><td>Random</td><td>Herding [42]</td><td>DiT [26]</td><td>DM [47]</td><td>MiniMax [11]</td><td> $\text{MGD}^3$ [3]</td><td>TGDD</td><td>Full</td></tr><tr><td rowspan="3">10 (0.8%)</td><td>ConvNet-6</td><td> $24.3_{\pm 1.1}$ </td><td> $26.7_{\pm 0.5}$ </td><td> $34.2_{\pm 1.1}$ </td><td> $26.9_{\pm 1.2}$ </td><td> $\mathbf{37.0}_{\pm 1.0}$ </td><td> $34.7_{\pm 1.1}$ </td><td> $34.8_{\pm 1.1}$ </td><td> $86.4_{\pm 0.2}$ </td></tr><tr><td>ResNetAP-10</td><td> $29.4_{\pm 0.8}$ </td><td> $32.0_{\pm 0.3}$ </td><td> $34.7_{\pm 0.5}$ </td><td> $30.3_{\pm 1.2}$ </td><td> $39.2_{\pm 1.3}$ </td><td> $40.4_{\pm 1.9}$ </td><td> $\mathbf{41.2}_{\pm 2.6}$ </td><td> $87.5_{\pm 0.5}$ </td></tr><tr><td>ResNet-18</td><td> $27.7_{\pm 0.9}$ </td><td> $30.2_{\pm 1.2}$ </td><td> $34.7_{\pm 0.4}$ </td><td> $33.4_{\pm 0.7}$ </td><td> $37.6_{\pm 0.9}$ </td><td> $38.5_{\pm 2.5}$ </td><td> $\mathbf{38.8}_{\pm 0.9}$ </td><td> $89.3_{\pm 1.2}$ </td></tr><tr><td rowspan="3">20 (1.6%)</td><td>ConvNet-6</td><td> $29.1_{\pm 0.7}$ </td><td> $29.5_{\pm 0.3}$ </td><td> $36.1_{\pm 0.8}$ </td><td> $29.9_{\pm 1.0}$ </td><td> $37.6_{\pm 0.2}$ </td><td> $39.0_{\pm 3.5}$ </td><td> $\mathbf{39.1}_{\pm 0.6}$ </td><td> $86.4_{\pm 0.2}$ </td></tr><tr><td>ResNetAP-10</td><td> $32.7_{\pm 0.4}$ </td><td> $34.9_{\pm 0.1}$ </td><td> $41.1_{\pm 0.8}$ </td><td> $35.2_{\pm 0.6}$ </td><td> $45.8_{\pm 0.5}$ </td><td> $43.6_{\pm 1.6}$ </td><td> $\mathbf{46.3}_{\pm 0.8}$ </td><td> $87.5_{\pm 0.5}$ </td></tr><tr><td>ResNet-18</td><td> $29.7_{\pm 0.5}$ </td><td> $32.2_{\pm 0.6}$ </td><td> $40.5_{\pm 0.5}$ </td><td> $29.8_{\pm 1.7}$ </td><td> $42.5_{\pm 0.6}$ </td><td> $41.9_{\pm 2.1}$ </td><td> $\mathbf{42.7}_{\pm 0.5}$ </td><td> $89.3_{\pm 1.2}$ </td></tr><tr><td rowspan="3">50 (3.8%)</td><td>ConvNet-6</td><td> $41.3_{\pm 0.6}$ </td><td> $40.3_{\pm 0.7}$ </td><td> $46.5_{\pm 0.8}$ </td><td> $44.4_{\pm 1.0}$ </td><td> $53.9_{\pm 0.6}$ </td><td> $54.5_{\pm 1.6}$ </td><td> $\mathbf{54.9}_{\pm 0.7}$ </td><td> $86.4_{\pm 0.2}$ </td></tr><tr><td>ResNetAP-10</td><td> $47.2_{\pm 1.3}$ </td><td> $49.1_{\pm 0.7}$ </td><td> $49.3_{\pm 0.2}$ </td><td> $47.1_{\pm 1.1}$ </td><td> $56.3_{\pm 1.0}$ </td><td> $56.5_{\pm 1.9}$ </td><td> $\mathbf{60.3}_{\pm 0.9}$ </td><td> $87.5_{\pm 0.5}$ </td></tr><tr><td>ResNet-18</td><td> $47.9_{\pm 1.8}$ </td><td> $48.3_{\pm 1.2}$ </td><td> $50.1_{\pm 0.5}$ </td><td> $46.2_{\pm 0.6}$ </td><td> $57.1_{\pm 0.6}$ </td><td> $58.3_{\pm 1.4}$ </td><td> $\mathbf{61.5}_{\pm 0.4}$ </td><td> $89.3_{\pm 1.2}$ </td></tr><tr><td rowspan="3">70 (5.4%)</td><td>ConvNet-6</td><td> $46.3_{\pm 0.6}$ </td><td> $46.2_{\pm 0.6}$ </td><td> $50.1_{\pm 1.2}$ </td><td> $47.5_{\pm 0.8}$ </td><td> $55.7_{\pm 0.9}$ </td><td> $55.1_{\pm 2.5}$ </td><td> $\mathbf{58.1}_{\pm 1.5}$ </td><td> $86.4_{\pm 0.2}$ </td></tr><tr><td>ResNetAP-10</td><td> $50.8_{\pm 0.6}$ </td><td> $53.4_{\pm 1.4}$ </td><td> $53.4_{\pm 0.9}$ </td><td> $51.7_{\pm 0.8}$ </td><td> $58.3_{\pm 0.2}$ </td><td> $60.2_{\pm 2.4}$ </td><td> $\mathbf{63.4}_{\pm 1.0}$ </td><td> $87.5_{\pm 0.5}$ </td></tr><tr><td>ResNet-18</td><td> $52.1_{\pm 1.0}$ </td><td> $49.7_{\pm 0.8}$ </td><td> $51.5_{\pm 1.0}$ </td><td> $51.9_{\pm 0.8}$ </td><td> $58.8_{\pm 0.7}$ </td><td> $59.7_{\pm 2.7}$ </td><td> $\mathbf{65.1}_{\pm 1.2}$ </td><td> $89.3_{\pm 1.2}$ </td></tr><tr><td rowspan="3">100 (7.7%)</td><td>ConvNet-6</td><td> $52.2_{\pm 0.4}$ </td><td> $54.4_{\pm 1.1}$ </td><td> $53.4_{\pm 0.3}$ </td><td> $55.0_{\pm 1.3}$ </td><td> $61.1_{\pm 0.7}$ </td><td> $60.1_{\pm 1.2}$ </td><td> $\mathbf{63.6}_{\pm 1.6}$ </td><td> $86.4_{\pm 0.2}$ </td></tr><tr><td>ResNetAP-10</td><td> $59.4_{\pm 1.0}$ </td><td> $61.7_{\pm 0.9}$ </td><td> $58.3_{\pm 0.8}$ </td><td> $56.4_{\pm 0.8}$ </td><td> $64.5_{\pm 0.2}$ </td><td> $66.5_{\pm 1.0}$ </td><td> $\mathbf{67.3}_{\pm 0.8}$ </td><td> $87.5_{\pm 0.5}$ </td></tr><tr><td>ResNet-18</td><td> $61.5_{\pm 1.3}$ </td><td> $59.3_{\pm 0.7}$ </td><td> $58.9_{\pm 1.3}$ </td><td> $60.2_{\pm 1.0}$ </td><td> $65.7_{\pm 0.4}$ </td><td> $68.8_{\pm 0.7}$ </td><td> $\mathbf{70.1}_{\pm 0.6}$ </td><td> $89.3_{\pm 1.2}$ </td></tr></table>

<table><tr><td>IPC (Ratio)</td><td>Test Model</td><td>Random</td><td>Herding [42]</td><td>IDC-1 [16]</td><td>MiniMax [11]</td><td>MGD $^{3}$  [3]</td><td>TGDD (Ours)</td><td>Full</td></tr><tr><td rowspan="3">10 (0.8%)</td><td>ConvNet-6</td><td> $17.0_{\pm 0.3}$ </td><td> $17.2_{\pm 0.3}$ </td><td> $24.3_{\pm 0.5}$ </td><td> $22.3_{\pm 0.5}$ </td><td> $23.4_{\pm 0.9}$ </td><td> $\textbf{24.8}_{\pm 0.4}$ </td><td> $79.9_{\pm 0.4}$ </td></tr><tr><td>ResNetAP-10</td><td> $19.1_{\pm 0.4}$ </td><td> $19.8_{\pm 0.3}$ </td><td> $25.7_{\pm 0.1}$ </td><td> $24.8_{\pm 0.2}$ </td><td> $25.8_{\pm 0.5}$ </td><td> $\textbf{27.0}_{\pm 0.4}$ </td><td> $80.3_{\pm 0.2}$ </td></tr><tr><td>ResNet-18</td><td> $17.5_{\pm 0.5}$ </td><td> $16.1_{\pm 0.2}$ </td><td> $25.1_{\pm 0.2}$ </td><td> $22.5_{\pm 0.3}$ </td><td> $23.6_{\pm 0.4}$ </td><td> $\textbf{24.7}_{\pm 0.9}$ </td><td> $81.8_{\pm 0.7}$ </td></tr><tr><td rowspan="3">20 (1.6%)</td><td>ConvNet-6</td><td> $24.8_{\pm 0.2}$ </td><td> $24.3_{\pm 0.4}$ </td><td> $28.8_{\pm 0.3}$ </td><td> $29.3_{\pm 0.4}$ </td><td> $30.6_{\pm 0.4}$ </td><td> $\textbf{31.8}_{\pm 0.5}$ </td><td> $79.9_{\pm 0.4}$ </td></tr><tr><td>ResNetAP-10</td><td> $26.7_{\pm 0.5}$ </td><td> $27.6_{\pm 0.1}$ </td><td> $29.9_{\pm 0.2}$ </td><td> $32.3_{\pm 0.1}$ </td><td> $33.9_{\pm 1.1}$ </td><td> $\textbf{35.2}_{\pm 0.3}$ </td><td> $80.3_{\pm 0.2}$ </td></tr><tr><td>ResNet-18</td><td> $25.5_{\pm 0.3}$ </td><td> $24.7_{\pm 0.1}$ </td><td> $30.2_{\pm 0.2}$ </td><td> $31.2_{\pm 0.1}$ </td><td> $32.6_{\pm 0.4}$ </td><td> $\textbf{33.4}_{\pm 0.3}$ </td><td> $81.8_{\pm 0.7}$ </td></tr></table>

Table 4: Comparison of dataset distillation performance on ImageNet-100. The best results are marked in bold.

Extending the evaluation to more challenging benchmarks, Table 4 and Table 5 report distillation results on ImageNet-100 and ImageNet-1k in two IPC settings. TGDD consistently achieves top-tier accuracy across these configurations. These results highlight the scalability of TGDD to a more diverse and complex dataset with greater class variability. In addition, the method performs consistently well across both lightweight and deeper architectures, indicating strong generalization across model architectures.

The efectiveness of TGDD is further evaluated on the ImageNette and ImageIDC subsets across varying IPC settings using a ResNet-10 backbone. These datasets consist of coarser-grained, well-separated classes with lower inter-class ambiguity compared to fine-grained datasets like ImageWoof. Experimental results in Table 6 show that TGDD achieves the highest accuracy across all configurations. In low-IPC settings, limited data makes it harder to recover semantic diversity. TGDD maintains a clear advantage by selectively choosing structurally suficient anchors instead of using all cluster members. This reduces noise and leads to cleaner and more representative cluster centers in the low IPC setting. In contrast, methods such as $\mathrm { M G D ^ { 3 } }$ [3], which include all members of the cluster when computing centers, are more likely to introduce noisy samples, especially under tight data budgets. Beyond quantitative performance, we further provide qualitative comparisons and eficiency analysis in Appendix §C. These additional evaluations confirm that TGDD maintains competitiveness in both sample quality and computational overhead.

Table 5: Comparison of dataset distillation performance on ImageNet-1k. The best results are marked in bold.

<table><tr><td>IPC</td><td>SRe $^{2}$ L [44]</td><td>RDED [35]</td><td>MiniMax [11]</td><td>MGD $^{3}$  [3]</td><td>TGDD</td></tr><tr><td>10</td><td>21.3 $_{\pm 0.6}$ </td><td>42.0 $_{\pm 0.1}$ </td><td>44.3 $_{\pm 0.5}$ </td><td>45.6 $_{\pm 0.1}$ </td><td>45.8 $_{\pm 0.2}$ </td></tr><tr><td>50</td><td>46.8 $_{\pm 0.2}$ </td><td>56.5 $_{\pm 0.1}$ </td><td>58.6 $_{\pm 0.3}$ </td><td>60.2 $_{\pm 0.1}$ </td><td>60.3 $_{\pm 0.1}$ </td></tr></table>

Table 6: Performance on the ImageNet subsets (Nette and IDC) across multiple images-per-class (IPC) settings. All results are obtained on a ResNetAP-10. The best results are marked in bold.

<table><tr><td></td><td>IPC</td><td>Random</td><td>DiT [26]</td><td>MiniMax [11]</td><td>MGD $^3$  [3]</td><td>TGDD</td><td>Full</td></tr><tr><td rowspan="3">Nette</td><td>10</td><td>54.2±1.6</td><td>59.1±0.7</td><td>62.0±0.2</td><td>66.4±2.4</td><td>67.8±0.6</td><td rowspan="3">93.3±0.1</td></tr><tr><td>20</td><td>63.5±0.5</td><td>64.8±1.2</td><td>66.8±0.4</td><td>71.2±0.5</td><td>73.6±0.5</td></tr><tr><td>50</td><td>76.1±1.1</td><td>73.3±0.9</td><td>76.6±0.2</td><td>79.5±1.3</td><td>81.3±0.6</td></tr><tr><td rowspan="3">IDC</td><td>10</td><td>48.1±0.8</td><td>54.1±0.4</td><td>53.1±0.2</td><td>55.9±2.1</td><td>57.1±1.6</td><td rowspan="3">92.1±0.4</td></tr><tr><td>20</td><td>52.5±0.9</td><td>58.9±0.2</td><td>59.0±0.4</td><td>61.9±0.9</td><td>63.4±0.5</td></tr><tr><td>50</td><td>68.1±0.7</td><td>64.3±0.6</td><td>69.6±0.2</td><td>72.1±0.8</td><td>73.1±0.8</td></tr></table>

## 5.3 Ablation Study

Contributions of TGDD components. Table 7 presents an ablation study evaluating the contributions of three key components in the proposed guided difusion pipeline: discrete space, PCA, and anchor selection. Each module contributes to a distinct aspect of the framework’s overall efectiveness. Clustering in the discrete space provides a symbolic representation of image content, which facilitates guidance with more specific visual concepts. PCA reduces the dimensionality of token features, making subsequent computations more robust and highlighting the most informative axes of variation. Anchor selection further enhances performance by filtering out noisy or redundant samples and retaining only structurally suficient examples to guide synthesis. To verify that this gain comes from the structural score itself, we further evaluate two control settings on ImageWoof IPC=20. Replacing score-based selection with random-M sampling within each discrete cluster drops accuracy to 43.1%, even below the centroid baseline of 45.2% in Table 7. Conversely, applying score-based ranking on continuous-feature clustering yields 44.5%, which improves over $\mathrm { M G D ^ { 3 } }$ at 43.6% but remains below the full discrete pipeline at 46.3%. These results confirm that each component contributes incremental gains, that the score identifies meaningfully better anchors, and that discrete clustering and score-based selection are complementary.

Table 7: The ablation study of the proposed token-guided dataset distillation scheme. Results are reported on the ImageWoof dataset with 20 and 50 images per class.

<table><tr><td>Discrete Space</td><td>PCA</td><td>Anchor Selection</td><td>IPC=20</td><td>IPC=50</td></tr><tr><td>-</td><td>-</td><td>-</td><td> $43.6_{\pm 1.6}$ </td><td> $56.5_{\pm 1.9}$ </td></tr><tr><td>√</td><td>-</td><td>-</td><td> $44.3_{\pm 1.2}$ </td><td> $57.4_{\pm 1.3}$ </td></tr><tr><td>√</td><td>√</td><td>-</td><td> $45.2_{\pm 1.3}$ </td><td> $58.9_{\pm 1.0}$ </td></tr><tr><td>√</td><td>√</td><td>√</td><td> $\mathbf{46.3}_{\pm 0.8}$ </td><td> $\mathbf{60.3}_{\pm 0.9}$ </td></tr></table>

Table 8: Ablation study of structural score components in TGDD. Top-1 test accuracies (%) on ImageWoof and ImageNette under $\mathrm { I P C } = 1 0 $ and 50 are reported.

<table><tr><td rowspan="2">JSD</td><td rowspan="2">HHI</td><td rowspan="2">COV</td><td colspan="2">ImageWoof</td><td colspan="2">ImageNette</td></tr><tr><td>IPC=10</td><td>IPC=50</td><td>IPC=10</td><td>IPC=50</td></tr><tr><td>√</td><td>-</td><td>-</td><td>38.2±2.4</td><td>58.7±0.6</td><td>65.1±1.5</td><td>80.3±0.6</td></tr><tr><td>-</td><td>√</td><td>-</td><td>37.8±2.3</td><td>58.7±1.0</td><td>65.9±1.0</td><td>80.5±0.8</td></tr><tr><td>-</td><td>-</td><td>√</td><td>37.1±2.6</td><td>57.8±2.1</td><td>65.6±1.7</td><td>80.0±1.2</td></tr><tr><td>√</td><td>√</td><td>-</td><td>38.7±3.4</td><td>59.3±0.5</td><td>65.9±0.6</td><td>81.2±0.5</td></tr><tr><td>√</td><td>-</td><td>√</td><td>38.9±1.4</td><td>59.0±1.1</td><td>65.8±0.3</td><td>80.7±1.1</td></tr><tr><td>-</td><td>√</td><td>√</td><td>38.9±1.7</td><td>59.5±0.6</td><td>66.0±0.7</td><td>80.8±0.5</td></tr><tr><td>√</td><td>√</td><td>√</td><td>41.2±2.6</td><td>60.3±0.9</td><td>67.8±0.6</td><td>81.3±0.6</td></tr></table>

Structural score component. To analyze how each structural score component in the anchor selection module contributes to dataset distillation, an ablation study is conducted, with results summarized in Table 8. The results indicate that relying on any single metric provides limited benefit, as each captures only a partial aspect of the structural score. JSD measures distributional divergence within each class, but ofers limited benefit when intra-class similarity is high. COV captures class-specific compositional information, which is valuable in low-IPC settings, where preserving class identity is even more critical. HHI reflects information concentration and balance, contributing consistently across diferent configurations. Combining metrics in pairs leads to moderate improvements, suggesting partial complementarity. The full integration of JSD, HHI, and COV yields the best results across all the settings. These findings highlight the complementary nature of the three components and underscore the importance of their joint use for efective anchor selection in token-guided dataset distillation.

Dimensions of PCA. Table 9a investigates the impact of varying the output dimensionality of PCA within the proposed framework. Results are reported on ImageWoof, with ResNetAP-10 trained at IPC 10 and 50. The baseline model without PCA achieves 57.4% accuracy. Applying PCA leads to consistent improvements, with the best performance observed at 512 dimensions. However, further compressing the feature dimensionality to 256-D and 128-D results in a slight performance decline, suggesting that excessive compression can remove important semantic information. These results confirm the efectiveness of PCA as a feature compression mechanism and indicate that a 512-dimensional representation ofers the best trade-of between compactness and task performance.

Table 9: (a) The ablation study on the efectiveness of PCA. Results are reported on ImageWoof with $\mathrm { I P C = 1 0 }$ and 50. (b) Impact of anchor set size on distillation performance. The experiment is conducted on ImageNette using ResNetAP-10 under $\mathrm { I P C } = 1 0 $ and 50.  
(a)  
(b)

<table><tr><td rowspan="2">IPC</td><td colspan="5">PCA Dimension</td></tr><tr><td>None</td><td>1024</td><td>512</td><td>256</td><td>128</td></tr><tr><td>10</td><td> $38.7_{\pm 0.9}$ </td><td> $40.1_{\pm 1.7}$ </td><td> $\mathbf{40.2}_{\pm 0.9}$ </td><td> $39.5_{\pm 1.1}$ </td><td> $38.5_{\pm 0.9}$ </td></tr><tr><td>50</td><td> $57.4_{\pm 1.3}$ </td><td> $57.5_{\pm 1.7}$ </td><td> $\mathbf{58.9}_{\pm 1.0}$ </td><td> $58.7_{\pm 0.4}$ </td><td> $58.5_{\pm 1.9}$ </td></tr></table>

<table><tr><td rowspan="2">IPC</td><td colspan="5">Number of anchors</td></tr><tr><td>1</td><td>5</td><td>10</td><td>20</td><td>30</td></tr><tr><td>10</td><td> $61.3_{\pm 2.4}$ </td><td> $63.8_{\pm 1.8}$ </td><td> $64.2_{\pm 0.9}$ </td><td> $\mathbf{67.8}_{\pm 0.6}$ </td><td> $64.0_{\pm 1.4}$ </td></tr><tr><td>50</td><td> $78.5_{\pm 1.0}$ </td><td> $79.7_{\pm 0.6}$ </td><td> $\mathbf{81.3}_{\pm 0.6}$ </td><td> $79.9_{\pm 0.1}$ </td><td> $80.2_{\pm 1.0}$ </td></tr></table>

Table 10: Performance comparison across various representation spaces and difusion architectures. Using our structural score for anchor selection, discrete tokenizers consistently outperform continuous models in guiding the generation process on ImageWoof.

<table><tr><td rowspan="2">Diffusion Arch.</td><td rowspan="2">No Guidance</td><td colspan="3">Continuous Models</td><td colspan="3">Discrete Tokenizers</td></tr><tr><td>VAE [30]</td><td>CLIP [29]</td><td>DINOv2 [25]</td><td>VQGAN [8]</td><td>BEiTv2 [27]</td><td>VQ-VAE [36]</td></tr><tr><td>DiT [26]</td><td> $49.3_{\pm 0.2}$ </td><td> $56.5_{\pm 1.9}$ </td><td> $57.5_{\pm 0.4}$ </td><td> $58.1_{\pm 0.6}$ </td><td> $60.0_{\pm 0.4}$ </td><td> $59.3_{\pm 0.7}$ </td><td> $60.3_{\pm 0.9}$ </td></tr><tr><td>LDM [30]</td><td> $48.6_{\pm 1.6}$ </td><td> $52.9_{\pm 2.2}$ </td><td> $53.4_{\pm 1.3}$ </td><td> $53.7_{\pm 0.8}$ </td><td> $55.8_{\pm 0.7}$ </td><td> $54.2_{\pm 0.5}$ </td><td> $55.1_{\pm 0.6}$ </td></tr></table>

Efect of anchor quantity on generation. To identify the optimal number of anchors for guiding the distillation process, we study how varying the anchor set size afects generation quality. Results are shown in Table 9b. When IPC = 10, using 20 anchors yields the best performance, while at higher IPC levels, fewer anchors (e.g., 10) are suficient. This trend aligns with expectations: under limited data conditions, using more guiding anchors helps enrich the semantic coverage of the generated samples. However, when excessive anchors are used, they may introduce noise or redundancy, leading to a drop in performance.

Generalization across spaces and difusion architectures. To validate the generalization and efectiveness of the proposed TGDD framework, we conduct a comprehensive cross-architecture evaluation. We evaluate our method using three distinct discrete visual tokenizers: VQ-VAE [36], VQGAN [8], and BEiTv2 [27], and compare them with widely used continuous embedding models including VAE encoder [17] from [30], CLIP [29], and DINOv2 [25]. These representations are paired with two difusion architectures, specifically the transformer-based DiT [26] and the U-Net-based LDM [30]. For continuous models, we follow the standard practice of clustering dense features and selecting samples closest to the centroids as anchors [3]. The results in Table 10 demonstrate the robustness of our token-guided approach. While applying guidance with advanced continuous models like DINOv2 and CLIP improves upon the unguided baselines, they are consistently outperformed by our TGDD framework across all tested discrete tokenizers and generative architectures. This empirical evidence confirms that our structural score metrics actively drive performance gain and provide a robust anchor selection strategy that generalizes across diverse discrete tokenizers and ofers a novel compositional perspective to the dataset distillation task.

## 6 Conclusion

In this work, we revisit dataset distillation through the compositional perspective of discrete visual tokens. By mapping image representations to discrete tokens via visual tokenizers, we reframe the assessment of distilled datasets as a matter of structural score rather than merely distributional similarity. We find that the statistics of visual tokens can provide a reliable predictor of validation performances. Building on this insight, we further leverage token statistics to guide difusion denoising. The proposed Token-Guided Dataset Distillation (TGDD) achieves state-of-the-art performance across multiple benchmarks. These results suggest that discrete token analysis provides principled value for understanding and guiding dataset distillation.

## Acknowledgements

This work was supported by Mitacs through the Mitacs Accelerate Program (Grant No. IT42711), by TerraSense Analytics, and in part by the U.S. National Science Foundation (OAC-2118240, HDR Institute: Imageomics).

## References

1. Cazenavette, G., Wang, T., Torralba, A., Efros, A.A., Zhu, J.Y.: Dataset distillation by matching training trajectories. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 4750–4759 (2022) 1, 3

2. Cazenavette, G., Wang, T., Torralba, A., Efros, A.A., Zhu, J.Y.: Generalizing dataset distillation via deep generative prior. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 3739–3748 (2023) 3

3. Chan Santiago, J.A., Tirupattur, P., Nayak, G.K., Liu, G., Shah, M.: MGD<sup>3</sup>: Mode-guided dataset distillation using difusion models. In: Proceedings of the 42nd International Conference on Machine Learning (ICML) (2025) 3, 6, 7, 8, 9, 10, 11, 12, 14, 20, 21, 22, 23

4. Chen, M., Du, J., Huang, B., Wang, Y., Zhang, X., Wang, W.: Influence-guided difusion for dataset distillation. In: The Thirteenth International Conference on Learning Representations (2025) 3

5. Cheng, G., Han, J., Lu, X.: Remote sensing image scene classification: Benchmark and state of the art. Proceedings of the IEEE 105(10), 1865–1883 (2017) 2

6. Cui, X., Qin, Y., Zhou, W., Li, H., Li, H.: Optical: Leveraging optimal transport for contribution allocation in dataset distillation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15245–15254 (2025) 3

7. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255. Ieee (2009) 2, 10

8. Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 12873–12883 (2021) 4, 6, 14

9. Fastai: fastai/imagenette: A smaller subset of 10 easily classified classes from imagenet, and a little more french. https://github.com/fastai/imagenette (2019) 6, 7, 10

10. Gini, C.: Measurement of inequality of incomes. The economic journal 31(121), 124–125 (1921) 4

11. Gu, J., Vahidian, S., Kungurtsev, V., Wang, H., Jiang, W., You, Y., Chen, Y.: Eficient dataset distillation via minimax difusion. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 15793–15803 (2024) 3, 6, 10, 11, 12, 20, 21

12. Gu, J., Wang, H., Jia, R., Vahidian, S., Kungurtsev, V., Jiang, W., Chen, Y.: Concord: Concept-informed difusion for dataset distillation. arXiv preprint arXiv:2505.18358 (2025) 3

13. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016) 6, 20

14. Helber, P., Bischke, B., Dengel, A., Borth, D.: Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing 12(7), 2217– 2226 (2019) 7

15. Hirschman, A.O.: National power and the structure of foreign trade, vol. 105. Univ of California Press (1980) 2, 4, 5

16. Kim, J.H., Kim, J., Oh, S.J., Yun, S., Song, H., Jeong, J., Ha, J.W., Song, H.O.: Dataset condensation via eficient synthetic-data parameterization. In: International Conference on Machine Learning. pp. 11102–11118. PMLR (2022) 3, 10, 11

17. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013) 14

18. Kullback, S., Leibler, R.A.: On information and suficiency. The annals of mathematical statistics 22(1), 79–86 (1951) 4

19. Lei, S., Tao, D.: A comprehensive survey of dataset distillation. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(1), 17–32 (2023) 1, 3

20. Lin, J.: Divergence measures based on the shannon entropy. IEEE Transactions on Information theory 37(1), 145–151 (1991) 2, 4, 5

21. Liu, H., Li, Y., Xing, T., Dalal, V., Li, L., He, J., Wang, H.: Dataset distillation via the wasserstein metric. arXiv preprint arXiv:2311.18531 (2023) 1, 3

22. Liu, Y., Gu, J., Wang, K., Zhu, Z., Jiang, W., You, Y.: Dream: Eficient dataset distillation by representative matching. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 17314–17324 (2023) 1

23. Lu, Y., Chen, X., Gu, J., Zhang, Y., Xuan, Q., Zhu, Z.: Dataset distillation with pre-trained models: A contrastive approach. Neurocomputing p. 132015 (2025) 1

24. Lu, Y., Chen, X., Zhang, Y., Gu, J., Zhang, T., Zhang, Y., Yang, X., Xuan, Q., Wang, K., You, Y.: Can pre-trained models assist in dataset distillation? arXiv preprint arXiv:2310.03295 (2023) 1

25. Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., et al.: Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193 (2023) 14

26. Peebles, W., Xie, S.: Scalable difusion models with transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 4195–4205 (2023) 10, 11, 12, 14

27. Peng, Z., Dong, L., Bao, H., Ye, Q., Wei, F.: Beit v2: Masked image modeling with vector-quantized visual tokenizers. arXiv preprint arXiv:2208.06366 (2022) 4, 6, 14

28. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent difusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023) 6, 7, 8

29. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. In: International conference on machine learning. pp. 8748–8763. PMLR (2021) 14

30. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022) 14

31. Sajedi, A., Khaki, S., Liu, L.Z., Amjadian, E., Lawryshyn, Y.A., Plataniotis, K.N.: Data-to-model distillation: Data-eficient learning framework. In: European Conference on Computer Vision. pp. 438–457. Springer (2024) 3

32. Sammut, C., Webb, G.I. (eds.): TF–IDF, pp. 986–987. Springer US, Boston, MA (2010). https://doi.org/10.1007/978-0-387-30164-8\_832 4, 5

33. Shannon, C.E.: A mathematical theory of communication. The Bell system technical journal 27(3), 379–423 (1948) 4

34. Su, D., Hou, J., Gao, W., Tian, Y., Tang, B.: D^4m: Dataset distillation via disentangled difusion model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 5809–5818 (June 2024) 3

35. Sun, P., Shi, B., Yu, D., Lin, T.: On the diversity and realism of distilled dataset: An eficient dataset distillation paradigm. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9390–9399 (2024) 3, 6, 7, 10, 12

36. Tian, K., Jiang, Y., Yuan, Z., Peng, B., Wang, L.: Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems 37, 84839–84865 (2024) 2, 4, 6, 10, 14, 20

37. Tian, Y., Krishnan, D., Isola, P.: Contrastive multiview coding. In: European conference on computer vision. pp. 776–794. Springer (2020) 10

38. Vahidian, S., Wang, M., Gu, J., Kungurtsev, V., Jiang, W., Chen, Y.: Group distributionally robust dataset distillation with risk minimization. In: The Thirteenth International Conference on Learning Representations (2025) 3

39. Van Den Oord, A., Vinyals, O., et al.: Neural discrete representation learning. Advances in neural information processing systems 30 (2017) 2

40. Wang, H., Zhao, Z., Wu, J., Shang, Y., Liu, G., Yan, Y.: Cao<sub>2</sub>: Rectifying inconsistencies in difusion-based dataset distillation. arXiv preprint arXiv:2506.22637 (2025) 3

41. Wang, S., Yang, Y., Liu, Z., Sun, C., Hu, X., He, C., Zhang, L.: Dataset distillation with neural characteristic function: A minmax perspective. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 25570–25580 (2025) 3

42. Welling, M.: Herding dynamical weights to learn. In: Proceedings of the 26th annual international conference on machine learning. pp. 1121–1128 (2009) 11

43. Yang, W., Zhu, Y., Deng, Z., Russakovsky, O.: What is dataset distillation learning? arXiv preprint arXiv:2406.04284 (2024) 1

44. Yin, Z., Xing, E., Shen, Z.: Squeeze, recover and relabel: Dataset condensation at imagenet scale from a new perspective. Advances in Neural Information Processing Systems 36, 73582–73603 (2023) 3, 6, 10, 12

45. Yu, R., Liu, S., Wang, X.: Dataset distillation: A comprehensive review. IEEE transactions on pattern analysis and machine intelligence 46(1), 150–170 (2023) 1, 3

46. Zhang, H., Li, S., Lin, F., Wang, W., Qian, Z., Ge, S.: DANCE: Dual-view distribution alignment for dataset condensation. In: Proceedings of the International Joint Conference on Artificial Intelligence (IJCAI) (2024) 3

47. Zhao, B., Bilen, H.: Dataset condensation with distribution matching. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. pp. 6514–6523 (2023) 1, 6, 7, 10, 11

48. Zhao, B., Mopuri, K.R., Bilen, H.: Dataset condensation with gradient matching. In: International Conference on Learning Representations (2021), https: //openreview.net/forum?id=mSAKhLYLSsl 1, 3

49. Zhao, L., Jiang, X., Xiao, X., Fan, Q., Lu, L., Wang, Y., Lin, X., Camps, O., Zhao, P., Gu, J.: Hieramp: Coarse-to-fine autoregressive amplification for generative dataset distillation. arXiv preprint arXiv:2603.06932 (2026) 3

50. Zhao, L., Wu, Y., Jiang, X., Gu, J., Wang, Y., Xu, X., Zhao, P., Lin, X.: Taming difusion for dataset distillation with high representativeness. In: Forty-second International Conference on Machine Learning (2025) 3

51. Zhong, X., Fang, H., Chen, B., Gu, X., Qiu, M., Qi, S., Xia, S.T.: Hierarchical features matter: A deep exploration of progressive parameterization method for dataset distillation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 30462–30471 (2025) 3

52. Zhou, Y., Nezhadarya, E., Ba, J.: Dataset distillation using neural feature regression. Advances in Neural Information Processing Systems 35, 9813–9827 (2022) 1

53. Zou, Y., Li, G., Su, D., Wang, Z., Yu, J., Zhang, C.: Dataset distillation via visionlanguage category prototype. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) (2025) 3

## Appendix

The appendix is organized as follows:

– §A: Pseudo-code for the anchor selection procedure used in TGDD.

– §B: Further implementation details for our experimental setup.

– §C: Additional analysis of the proposed TGDD model, including diferent scale weighting configurations, eficiency analysis, and qualitative comparisons.

## A Anchor Selection Algorithm

The anchor selection procedure is summarized in algorithm 1. The algorithm takes the tokenized dataset as input and selects the images with the top structural scores to guide the difusion generation process. The output consists of the top M anchors for each cluster, where M = 10 for IPC settings greater than 50 and M = 20 for IPC settings below 50.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1: Token-Guided Anchor Selection

Input : Dataset $\mathcal{T} = \{(x_i, y_i)\}$; multi-scale VQ-VAE; codebook size $V$; scale weights $\{w_\ell\}_{\ell=1}^L$; IPC $\Rightarrow$ clusters $K_c$ per class; feature weights $W_{\text{JSD}}, W_{\text{HHI}}, W_{\text{Cov}}$; TF-IDF top-token size $K_{\text{tfidf}}$; PCA dim $d$; Anchor number $M$.

Output: Anchor set $\mathcal{A}$ with $M$ anchors per cluster.

for each class $c = 1, \ldots, C$ do

for each $x_i$ with $y_i = c$ do

Compute per-scale token histograms $\boldsymbol{p}_i^{(\ell)} \in \mathbb{R}^V$; fuse $\boldsymbol{p}_i \leftarrow \sum_{\ell=1}^L w_\ell \boldsymbol{p}_i^{(\ell)}$

Compute per-scale TF, classwise IDF; form fused TF-IDF vector; set $\mathcal{T}_c$ of size $K_{\text{tfidf}}$

Apply PCA to $d$ dims, L2-normalize; run $k$-means into $K_c$ clusters to obtain member sets $\{\mathcal{S}_{c,m}\}_{m=1}^{K_c}$.

for each $m = 1, \ldots, K_c$ do

Compute the cluster prior $\boldsymbol{\mu}_{c,m}$ from members $\mathcal{S}_{c,m}$

For $i \in \mathcal{S}_{c,m}$, set $\mathbf{m} = \frac{1}{2} (\boldsymbol{p}_i + \boldsymbol{\mu}_{c,m})$ and compute
$\mathrm{JS}(\boldsymbol{p}_i, \boldsymbol{\mu}_{c,m}) = \frac{1}{2} (\mathrm{KL}(\boldsymbol{p}_i \| \mathbf{m}) + \mathrm{KL}(\boldsymbol{\mu}_{c,m} \| \mathbf{m}))$

For $i \in \mathcal{S}_{c,m}$, compute $\mathrm{Cov}_i = \sum_{k \in \mathcal{T}_c} p_i(k)$ and $D_i = 1 - \sum_{k=1}^V (p_i(k))^2$;

normalize $\{\mathrm{JS}_i\}, \{\mathrm{Cov}_i\}$ and $\{D_i\}$ within each cluster

Rank by $s_i = w_{\text{Cov}} \widetilde{\text{Cov}}_i + w_{\text{HHI}} \widetilde{D}_i + w_{\text{JSD}} (1 - \widetilde{\text{JS}}_i)$ and add the top $M$ indices to $\mathcal{A}$ for cluster $m$
</div>

## B Additional Implementation Details

Throughout this study, we follow the literature [3,11] and evaluate the efectiveness of the generated datasets using three main architectures: ConvNet-6, a 6-layer convolutional network with 256×256 input size; ResNetAP-10, a 10- layer ResNet [13] with average pooling for downsampling; and ResNet-18, an 18-layer ResNet with instance normalization. For experiments on ImageWoof, ImageNette, ImageIDC, and ImageNet-100, we adopt the hard-label protocol used in [3]. All architectures are trained using SGD with a learning rate of 0.01 and a decay factor of 0.2 applied at 2/3 and 5/6 of the total training epochs. Models are trained for 1500 epochs under IPC values of 20, 50, 70, and 100, and for 2000 epochs under the IPC=10 setting. Random resize-crop and CutMix augmentations are applied during training.

For evaluation on ImageNet-1k, we use the soft-label protocol described in [3, 11]. Training is performed on a ResNet-18 architecture serving as both the teacher and the student for 300 epochs. The AdamW optimizer is used for ImageNet-1K training, with a learning rate of 0.001 and a weight decay of 0.01.

## C Additional Analysis of TGDD

## C.1 Diferent Scale Weighting Configurations

Our TGDD framework employs a multi-scale VQ-VAE [36] as the default visual tokenizer. Due to its pyramidal architecture, the token count grows quadratically at each subsequent scale. Therefore, a direct summation of the distributions causes finer scales to dominate the result. To mitigate this inherent density imbalance, we performed an ablation study on the ImageWoof and ImageNette datasets by grouping the 10 scales into Low, Mid, and High bands. As shown in Table 11, we observe a consistent trend: configurations that prioritize coarse scales (Low > Mid > High) outperform uniform or inverse weighting schemes. Based on these empirical results, we adopted [3, 1, 0.5] as the default configuration to maintain an optimal balance between the information from diferent scales.

Table 11: The results demonstrate the sensitivity of the model to the relative importance assigned to Low, Mid, and High resolution scales.

<table><tr><td rowspan="2">Benchmark</td><td colspan="5">Scale Weighting Configurations</td></tr><tr><td>[5, 1, 0.1]</td><td>[3, 1, 0.5]</td><td>[1, 1, 1]</td><td>[1, 2, 3]</td><td>[0.5, 1, 3]</td></tr><tr><td>ImageWoof</td><td> $59.1_{\pm 0.3}$ </td><td> $\mathbf{60.3}_{\pm 0.9}$ </td><td> $58.8_{\pm 1.4}$ </td><td> $58.9_{\pm 0.4}$ </td><td> $58.2_{\pm 1.7}$ </td></tr><tr><td>ImageNette</td><td> $80.5_{\pm 0.6}$ </td><td> $\mathbf{81.3}_{\pm 0.6}$ </td><td> $80.1_{\pm 0.6}$ </td><td> $79.2_{\pm 0.7}$ </td><td> $79.2_{\pm 0.9}$ </td></tr></table>

## C.2 Eficiency Analysis

We analyze the distillation time of TGDD by decomposing the computation into tokenization, PCA transformation, anchor selection, and difusion-based generation. TGDD uses the same difusion backbone as existing methods $( \mathrm { e . g . }$ Minimax Difusion [11] and $\mathrm { M G D ^ { 3 } \ [ 3 ] ) }$ and operates without any fine-tuning, so the main additional computational cost comes from the pre-processing stages. The total complexity can be written as

$$
\mathcal {C} _ {\mathrm{TGDD}} = \mathcal {C} _ {\mathrm{Diff}} + \mathcal {O} (N T) + \mathcal {O} (d ^ {2} N) + \mathcal {O} (N d) + \mathcal {O} (N \log K),
$$

where each term corresponds to a specific component of TGDD. The difusion sampling cost $\mathcal { C } _ { \mathrm { D i f f } }$ is shared across difusion-based distillation methods and dominates the overall runtime. The tokenization step processes N images, each represented by $T$ discrete visual tokens, leading to the $\mathcal { O } ( N T )$ term. PCA projection is applied to all token histograms, and computing a d-dimensional projection requires multiplying by a $d \times d$ matrix, resulting in $\mathcal O ( d ^ { 2 } N )$ . Anchor selection evaluates structural scores, including JSD, HHI, and COV, each computed over a d-dimensional feature, producing the O(N d) term, followed by a partial ranking step that contributes $\mathcal { O } ( N \log K )$ . Here, K denotes the number of anchors retained per class for guiding the difusion process. These additional operations are lightweight compared with C<sub>Dif</sub>.

To empirically validate the eficiency, Table 12 presents the per-image processing time for each stage of the TGDD and $\mathrm { M G D ^ { 3 } }$ pipelines. As demonstrated in the table, the computational cost of feature extraction and anchor selection is minor, with the difusion-based generation process accounting for the majority of the total runtime. This confirms that the generation phase is the primary computational bottleneck, whereas the proposed token-based assessment introduces limited overhead. Specifically, for the total distillation of the 10-class ImageWoof dataset on a single RTX 3080 GPU, TGDD requires approximately 0.43 hours. This duration is comparable to the highly eficient $\mathrm { M G D ^ { 3 } }$ at 0.32 hours, and is significantly faster than other difusion-based baselines that require training, such as Minimax Difusion which demands 2.02 hours. Considering the substantial performance gain of up to $5 . 4 \%$ on ImageWoof, TGDD delivers stronger performance with only modest extra preprocessing efort.

<table><tr><td>Model</td><td>Feature Extraction</td><td colspan="2">Clustering Generation</td></tr><tr><td>TGDD</td><td>0.0198s</td><td>0.0344s</td><td>1.7s</td></tr><tr><td>MGD $^{3}$ </td><td>0.0212s</td><td>0.0010s</td><td>1.7s</td></tr></table>

Table 12: Comparison of per-image processing time between TGDD and $\mathrm { M G D ^ { 3 } }$

## C.3 Visualization of Synthetic Samples

We visualized the images generated by TGDD on the ImageWoof dataset alongside those produced by MGD<sup>3</sup> [3] and real samples randomly selected from the original dataset, as shown in Figure 5. All results are obtained under the IPC = 10 setting using the same random seed. The proposed TGDD generates highquality samples that closely resemble the real images, demonstrating its ability to preserve both semantic content and structural details. While MGD<sup>3</sup> also produces high-quality synthetic data, some images exhibit artifacts, such as slight distortions in facial regions or incomplete object parts. These diferences likely arise from the anchor selection strategy of TGDD, which selects only the topranked and structurally suficient anchors to guide the generation. This selective guidance reduces the noise introduced by averaging large numbers of candidate images and mitigates the influence of repeated patterns or outliers that may negatively afect sample quality.

![](images/ad0b9f780d126faf1af029f6e847ae6c9d589f689b973c63143263c67f5d3186.jpg)

(a) Rhodesian Ridgeback  
![](images/6a18298bca713249cacf5ecc28b0289ffca92ed0c08f353c6e56eb1cda8fcf48.jpg)

(b) Border Terrier  
![](images/68377cc4680864de3f3b6296759a4bc0312e7eeca301c4f34734d9311acfdf3d.jpg)  
(c) Australian Terrier  
Fig. 5: Visualization of generated samples for three ImageWoof classes (Rhodesian Ridgeback, Border Terrier, and Australian Terrier). For each class, we show real images, images generated by $\mathrm { M G D ^ { 3 } }$ [3], and images generated by TGDD under the $\mathrm { I P C } = 1 0 $ setting with the same random seed.