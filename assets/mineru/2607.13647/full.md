# Beyond Color Geometry: Evaluating Human-Like Color Representations in Vision Models

Ayan Igali, Pakizar Shamoi

School of Information Technology and Engineering

Kazakh–British Technical University

Almaty, Kazakhstan

{ay igali, p.shamoi}@kbtu.kz

Abstract—Do vision models see colors the way humans do? Existing evaluations of color representations usually compare them with geometric spaces such as CIELAB or with discrete color labels. These references capture perceptual distance or category membership, but not the graded way in which people organize colors. We evaluate color grounding against a fuzzy perceptual model with 86 graded categories fitted to human survey data. The framework can be applied to any image encoder and measures three complementary properties: category boundaries, category compactness, and graded alignment beyond what color geometry alone can explain. Across eleven Vision Transformer encoders, the category-level results are broadly similar, whereas graded alignment differs substantially. Masked Autoencoders achieve the strongest beyond-geometry alignment, with confidence intervals that do not overlap those of the other encoders. A layer-wise analysis further shows that masked reconstruction preserves this structure toward the output. On natural images, MAE represents surface color globally, while language-supervised models encode color more strongly in relation to the foreground object. These results show that human-like color grounding has several distinct aspects that should not be reduced to a single score.

## I. INTRODUCTION

Modern vision and vision–language models are widely used as feature extractors across computer vision [1], [2]. For human-centered and explainable AI, it is useful to ask whether their internal color representations resemble human color perception or mainly reflect pixel-level regularities. Human color perception is not organized as a uniform metric continuum. People group colors into overlapping categories with soft boundaries, and a color can belong partly to several named categories such as red, dark red, gray, or black [3].

Most previous studies of color grounding have used geometric color spaces. Abdou et al. [4], for example, showed that text-only language models recover part of the topology of CIELAB [5], and later work reported similar results [6], [7]. Such analyses test whether a representation preserves perceptual distances. They do not directly test whether it reproduces the graded category structure of human color perception. Color-naming datasets such as the World Color Survey (WCS) [8] provide human categories, but these are usually treated as hard labels and therefore lose partial membership near category boundaries.

We address this gap with a fuzzy perceptual reference. In a fuzzy color space, each color is represented by graded membership in several human-named categories rather than by a single label. We use COLIBRI [9], which defines 86 categories over Hue, Saturation, and Intensity using membership functions fitted to a large human survey. This lets us compare model representations with human color organization at both categorical and graded levels.

We apply the evaluation to eleven encoders from the Vision Transformer family. The main contributions of this paper are:

• We introduce a reusable framework for evaluating color representations against the fuzzy human color model. The framework applies to any image encoder and reports complementary measures of categorical and graded color organization.

• We propose a geometry-controlled alignment measure that separates agreement with human color structure from conventional perceptual color distance.

• We provide a systematic comparison of eleven Vision Transformer encoders across multiple training regimes, including layer-wise and natural-image analyses

The remainder of this paper is structured as follows. Section II reviews related work on color perception, color representations in learned models, and existing approaches to human–model alignment. Section III describes the data, the fuzzy perceptual color model, the evaluated vision encoders, and the experimental methodology. Section IV presents the results. Section V discusses the findings in relation to recent studies. Finally, Section VI concludes the paper.

## II. RELATED WORK

Research on color categories has long examined the relationship between perception and language, from accounts of basic color terms [10] to cross-linguistic analyses based on the World Color Survey [8]. Information-theoretic studies further suggest that color naming reflects efficient communication and that different regions of color space are described with varying precision [11]. Across these traditions, human color categories are generally understood as graded and unequal in extent. Colors near a category boundary may partially belong to several categories, while broad categories such as green may occupy substantially more of color space than narrower categories such as yellow. COLIBRI [9] models these properties using overlapping fuzzy membership functions defined over Hue, Saturation, and Intensity.

Several studies have investigated whether learned representations recover perceptual color structure. Abdou et al. [4] used representational similarity analysis (RSA) [12] and linear probing to show that text-only models encode aspects of CIELAB geometry. Related findings have since been reported for other language models [6], [7]. More broadly, research on human–model alignment has emphasized that alignment should be assessed through multiple complementary measures rather than reduced to a single score [13]. Other studies suggest that training data and learning objectives may influence human alignment more strongly than model scale or architectural family [14], [15].

Color has also been evaluated directly in vision and vision-language models. Task-level benchmarks such as Color-Bench [16] assess color perception, reasoning, and robustness through model responses. Such benchmarks measure observable behavior on specific tasks, rather than the organization of color information within the encoder representation. In a different line of work, comparisons between model representations and neural responses indicate that masked autoencoders preserve low-level visual structure while showing weaker correspondence with higher-level neural representations [17].

The studies most closely related to representation-level color evaluation rely either on discrete categories or on continuous perceptual similarity. Akbarinia [18] evaluates categorical color perception in vision networks using an odd-one-out task over World Color Survey chips, but represents perception through a small set of discrete basic-color labels. Wickramanayaka and Oizumi [19] compare sixteen networks with human color-similarity judgments using RSA and unsupervised Gromov–Wasserstein alignment. Their results suggest that CLIP most closely reproduces the fine-grained geometry of human similarity judgments. Fuzzy color naming has also been studied directly [20], primarily as a method for assigning linguistic labels to colors rather than as a reference for evaluating internal model representations.

Existing approaches therefore evaluate learned color representations against geometric color spaces, continuous similarity judgments, discrete perceptual categories, or task-level behavior. These references capture different aspects of color perception, but none simultaneously represents categorical structure, graded membership, and overlap between linguistic color categories at the representation level (Table I).

Our work addresses this gap by evaluating frozen imageencoder representations against COLIBRI, a human-derived fuzzy color model. In contrast to geometric references [4] and continuous similarity judgments [19], COLIBRI describes how colors belong to overlapping linguistic categories with different degrees of membership. At the same time, our partialcorrelation analysis controls for geometric color distance, allowing categorical agreement beyond color geometry to be measured explicitly.

## III. METHODS

We evaluate color grounding by comparing the structure of a model’s embedding space with the human color structure represented by COLIBRI. The framework requires only forward passes through a frozen encoder and reports several complementary measures. Figure 1 summarizes the evaluation pipeline.

TABLE I  
APPROACHES TO EVALUATING COLOR IN LEARNED MODELS.

<table><tr><td>Approach</td><td>Ground truth</td><td>Cat.</td><td>Graded</td><td>Repr.</td></tr><tr><td>Abdou [4]</td><td>CIELAB geometry</td><td>—</td><td>(cont.)</td><td>√</td></tr><tr><td>Wickr. [19]</td><td>similarity judg.</td><td>—</td><td>(cont.)</td><td>√</td></tr><tr><td>Akbarinia [18]</td><td>hard categories</td><td>√</td><td>—</td><td>√</td></tr><tr><td>ColorBench [16]</td><td>task answers</td><td>√</td><td>—</td><td>—</td></tr><tr><td>Benavente [20]</td><td>fuzzy naming</td><td>√</td><td>√</td><td>—</td></tr><tr><td>Ours</td><td>fuzzy human (COLIBRI)</td><td>√</td><td>√</td><td>√</td></tr></table>

## A. Data

We use the 330 Munsell chips from the World Color Survey [8], a standard stimulus set in color-cognition research, together with their CIELAB coordinates under illuminant C.

Uniform color fields differ from the natural images on which the encoders were trained, so we render each chip at 224×224 as a flat patch and a shaded sphere. We use shaded spheres as an intermediate stimulus between uniform color patches and natural images. Compared with flat patches, they add shading, contrast, shape, and spatial variation while keeping the underlying surface color controlled. COLIBRI provides the human reference for both forms. Its 86-dimensional membership vectors are used for continuous analyses, and the dominant categories are used for clustering. CIEDE2000 [21] distances in CIELAB serve only as a geometric control used to test whether model–COLIBRI alignment remains after physical color geometry is accounted for.

We also use the Tiny ImageNet subset of MegaCOIN [22]. Each image is annotated with a human color term for the foreground object and another for the background, which allows us to examine color representations in multi-colored scenes. The foreground probe uses 7,997 images balanced across the eleven color terms; the background probe and the paired object-selectivity gap use the 7,744 of these that also carry a valid background label. Representative examples of all three stimulus formats are shown in Fig. 2.

## B. COLIBRI: A fuzzy perceptual color space

COLIBRI [9] is a fuzzy color model grounded in human color-naming data. Rather than assigning each color to a single label, it represents graded membership in overlapping linguistic categories. An RGB color is first converted to Hue–Saturation–Intensity (HSI), after which membership is computed using triangular or trapezoidal fuzzy sets derived from a survey of 2,496 participants.

As shown in Fig. 3, COLIBRI partitions hue into nine chromatic categories, saturation into four levels, and intensity into three levels. Not all combinations define separate categories, because extreme intensity values map to black or white and very low saturation maps to dark grey, grey, or light grey regardless of hue. The remaining categories combine nine hues, three chromatic saturation levels, and three intensity levels, giving $9 \times 3 \times 3 + 5 = 8 6$ categories.

![](images/bf62180f274081357e988e886f8363792097830f07d35162325a397fb034c669.jpg)  
Fig. 1. Overview of the evaluation pipeline. World Color Survey / Munsell chips are rendered as flat patches and shaded spheres, mapped to COLIBRI fuzzy human color memberships, passed through ViT encoders, and evaluated by comparing model representational structure with fuzzy, geometric, and categorica color structure.

![](images/deb8d4443f9e97dbb9a90d7188d900b91420c81681e26327923b49f3843dae65.jpg)  
Fig. 2. Representative stimuli used in the experiments.

For a color $( h , s , i )$ and category $\boldsymbol { v } ~ = ~ \left( v _ { H } , v _ { S } , v _ { I } \right)$ , the composite membership is $\mu _ { v _ { H } } ( h ) \mu _ { v _ { S } } ( s ) \mu _ { v _ { I } } ( i )$ . The resulting 86-dimensional vector is used in our continuous analyses, while its largest component defines the dominant label used for clustering. This representation preserves category overlap near perceptual boundaries and reflects human color naming rather than geometric distance alone.

## C. Models

We evaluate eleven encoders spanning four training regimes and two model scales:

• Language-supervised models: CLIP ViT-B/32 and ViT-L/14 [23], OpenCLIP ViT-L/14 trained on LAION-2B [24], SigLIP-Large [25], and multilingual SigLIP-2- Large [26].

• Class-supervised models: ViT-B/16 and ViT-L/16 trained on ImageNet-1k [27].

• Reconstruction models: MAE ViT-B/16 and ViT-L/16 [28].

• Self-distillation models: DINOv2 ViT-B/14 and ViT-L/14 [29].

## D. Experimental Settings

All models checkpoints are public and are used without fine-tuning. For image–text models, we use the post-projection image embedding. For MAE, patch masking is disabled and patch tokens are mean-pooled. For supervised ViTs, we use the final class token. Each stimulus produces one embedding, and pairwise dissimilarity is measured with cosine distance.

All models share a ViT backbone but differ in training data, scale, patch size, and recipe, so we report associations with the training regime rather than attributing differences to the objective alone.

1) Chip experiments: We combine RSA with categorylevel metrics. For embeddings $\mathbf { x } _ { i }$ of N stimuli, we construct a representational dissimilarity matrix (RDM) using cosine distance: $D _ { i j } ^ { \mathrm { e m b } } = 1 - \cos ( { \bf x } _ { i } , { \bf x } _ { j } )$

We compare the upper triangle of the embedding RDM with the upper triangle of a reference RDM using Spearman rank correlation. We denote the correlation with the COLIBRI RDM by $\rho _ { \mathrm { f u z z y } }$ . The COLIBRI RDM is computed with cosine distance between the 86-dimensional membership vectors.

COLIBRI and CIELAB are related, so a high model– COLIBRI correlation may partly reflect ordinary color geometry. To isolate the remaining graded structure, we compute a partial rank correlation between the model and COLIBRI RDMs while controlling for the CIELAB/CIEDE2000 RDM:

$$
p _ {\Delta} = \frac {\rho_ {m , f} - \rho_ {m , l} \rho_ {f , l}}{\sqrt {(1 - \rho_ {m , l} ^ {2}) (1 - \rho_ {f , l} ^ {2})}}.\tag{1}
$$

Here $m , f ,$ and l denote the model, fuzzy (COLIBRI), and Lab RDMs, so that $\rho _ { m , f }$ is exactly $\rho _ { \mathrm { f u z z y } }$ from above. Applying the standard first-order partial-correlation formula to the Spearman coefficients gives this beyond-geometry measure.

Because the 330 chips act as dominant examples of only 50 of the 86 categories, and some of those categories are represented by very few chips, the clustering metrics (silhouette and ARI) use only categories with at least four chips. The

![](images/6605d20cc023a51d6361b564419baeb7e0d9198eac3e6d93f4dbdd113ca2da05.jpg)  
Fig. 3. COLIBRI category structure: nine hues, four saturation levels, three intensity levels. At very low saturation a color is named by an achromatic grey level and at extreme intensity by black or white, regardless of hue, giving five achromatic categories.

RSA measures instead compare the full pairwise structure of the 86-dimensional membership vectors, so they use all 330 chips regardless of how often each category is dominant.

For category-level analyses, each chip is assigned its dominant COLIBRI label $c ( i )$ . The silhouette score measures whether embeddings are closer to members of their own category than to members of the nearest alternative category, so higher values indicate more compact categories; it uses the COLIBRI labels directly and involves no clustering. Separately, we run k-means on the embeddings (n\_init = 10) and compare the resulting clusters against the COLIBRI labels with the Adjusted Rand Index (ARI), which is chancecorrected so that random agreement is approximately zero.

Silhouette and ARI describe global category organization. All RDMs use cosine distance except the CIELAB reference, which uses CIEDE2000. We estimate 95% percentile bootstrap confidence intervals by resampling the 330 chips, rather than the individual pairwise distances.

2) Natural-image experiment: We use the balanced Mega-COIN images, extracting embeddings exactly as for the WCS stimuli and training a separate linear probe for the foregroundobject color and for the background color. Each probe uses five-fold cross-validation and is scored by balanced accuracy, which corrects for the imbalance in the background labels. Object selectivity is the gap $\Delta \ = \ \mathrm { a c c _ { f g } \ - \ a c c _ { b g } , }$ with confidence intervals from a paired bootstrap over the shared images.

## IV. RESULTS

## A. Color Representation Analysis

In table II, we report results for shaded spheres, which are closer to the encoders’ training distribution than uniform patches. The category-level metrics vary little across models. ARI values lie in a narrow band with overlapping bootstrap intervals, and silhouette scores overlap as well, so neither category boundaries nor cluster compactness clearly separates the training regimes.

The graded measures show a different pattern. MAE alignment is high and nearly unchanged across rendering formats, whereas alignment for the language-supervised models rises markedly from flat patches to shaded spheres. Both $\rho _ { \mathrm { f u z z y } }$ and $p _ { \Delta }$ place the two MAE models above all other encoders. MAE reaches $\rho _ { \mathrm { f u z z y } } = 0 . 6 7 ~ [ 0 . 6 5 , 0 . 7 0 ]$ and $p _ { \Delta } = 0 . 4 7 ~ [ 0 . 4 3 , 0 . 5 0 ]$ and on each measure its interval clears the best non-MAE encoder, which reaches only 0.58 and 0.28. The same ordering holds on flat patches, where MAE again leads on both $\rho _ { \mathrm { f u z z y } }$ (around 0.66 against 0.57) and $p _ { \Delta }$ (around 0.39 against 0.30), so the advantage survives even after Lab geometry is partialled out and does not rest on the naturalistic rendering. This is not explained by tighter clusters, since MAE has relatively low silhouette scores. It reflects instead a closer match to the graded structure of COLIBRI.

## B. Natural images: global and object-bound color representations

On the MegaCOIN images, the foreground-object color and background color produce different model rankings (Fig. 4). Language-supervised models decode the foreground object’s color most accurately, whereas the two MAE models perform best on background color. This difference is summarized by object selectivity, ∆. It is largest for language-supervised models (+0.12 to +0.14), intermediate for class supervision and self-distillation (+0.07 to +0.11), and close to zero for masked reconstruction (+0.02 and +0.00). The confidence interval for MAE-base includes zero, and neither MAE interval overlaps the interval of any other encoder. Embedding dimension does not explain the pattern: DINOv2-large and MAE-large both have 1,024-dimensional embeddings, yet DINOv2-large has the lowest background-color accuracy.

These results suggest that masked reconstruction retains a more global representation of surface color, whereas language supervision makes color more specific to the foreground object. This fits the WCS results, where a single-color stimulus plays to the strength of a global color representation.

## C. Layer dynamics

The previous analyses use only the final embedding. To trace how graded color structure changes through the network, we mean-pool the token representations of each transformer layer and recompute $\rho _ { \mathrm { f u z z y } }$ for all eleven encoders (Fig. 5). Uniform pooling makes depths comparable within and across models, so these per-layer values are not expected to match the final-embedding scores of Table II exactly; we read the trajectories for their shape, not their absolute level. We report relative depth because the models contain different numbers of layers.

The two MAE models have the highest alignment at nearly every depth and are the only encoders that do not lose alignment toward the output. Language-supervised models begin with high alignment at the input embedding, but it declines steadily through the network. DINOv2 shows a similar downward trend. The class-supervised models differ by scale: ViT-L drops sharply in the middle layers and remains low, whereas ViT-B partially recovers after an initial decline. Thus, graded human color structure is already present in early representations across all models, but masked reconstruction preserves it most consistently through depth.

![](images/367a70cf4347ff811e07fdac5cf5345b9bab7d9c901fd381e58bee0adae48a96.jpg)  
Fig. 4. Natural images from Tiny ImageNet. (a) Balanced probe accuracy for background (◦) and foreground-object (•) color. Segment length shows th object-color selectivity of each encoder. (b) The same effect expressed as $\Delta = \mathrm { a c c _ { f g } - a c c _ { b g } , }$ with paired-bootstrap 95% confidence intervals over 7,744 images.

TABLE II  
ALL ELEVEN ENCODERS, ORDERED BY SPHERE $\rho _ { \mathrm { F U Z Z Y } } .$ . ARI AND SILHOUETTE ARE REPORTED ON SPHERES; THE GRADED MEASURES ρ<sub>FUZZY</sub> AND p<sub>∆</sub> ARE REPORTED ON BOTH FLAT PATCHES (PT) AND SHADED SPHERES (SPH). SPHERE COLUMNS CARRY 95% BOOTSTRAP CONFIDENCE INTERVALS; PATCH COLUMNS GIVE POINT ESTIMATES.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Supervision</td><td rowspan="2">ARI</td><td rowspan="2">Sil.</td><td colspan="2"> $\rho_{\text{fuzzy}}$ </td><td colspan="2"> $p_{\Delta}$ </td></tr><tr><td>pt</td><td>sph</td><td>pt</td><td>sph</td></tr><tr><td>MAE-large</td><td>Reconstruction</td><td>0.358 [.29,.41]</td><td>0.071 [-.04,.12]</td><td>0.665</td><td>0.673 [.65,.70]</td><td>0.401</td><td>0.467 [.43,.50]</td></tr><tr><td>MAE-base</td><td>Reconstruction</td><td>0.318 [.29,.41]</td><td>0.094 [-.01,.13]</td><td>0.661</td><td>0.667 [.64,.69]</td><td>0.384</td><td>0.455 [.42,.49]</td></tr><tr><td>DINOv2-base</td><td>Self-distillation</td><td>0.348 [.26,.38]</td><td>0.081 [.01,.12]</td><td>0.524</td><td>0.576 [.55,.60]</td><td>0.298</td><td>0.248 [.21,.28]</td></tr><tr><td>CLIP-B</td><td>Language</td><td>0.338 [.27,.43]</td><td>0.177 [.10,.22]</td><td>0.400</td><td>0.563 [.54,.59]</td><td>0.236</td><td>0.207 [.18,.24]</td></tr><tr><td>ViT-B (IN1k)</td><td>Class</td><td>0.285 [.26,.40]</td><td>0.138 [.06,.18]</td><td>0.571</td><td>0.560 [.54,.59]</td><td>0.289</td><td>0.237 [.20,.27]</td></tr><tr><td>DINOv2-large</td><td>Self-distillation</td><td>0.283 [.25,.36]</td><td>0.067 [-.00,.10]</td><td>0.444</td><td>0.548 [.52,.58]</td><td>0.171</td><td>0.227 [.19,.26]</td></tr><tr><td>SigLIP-2-L</td><td>Language</td><td>0.331 [.25,.38]</td><td>0.145 [.07,.18]</td><td>0.499</td><td>0.543 [.51,.57]</td><td>0.245</td><td>0.277 [.24,.32]</td></tr><tr><td>CLIP-L</td><td>Language</td><td>0.285 [.24,.36]</td><td>0.141 [.07,.18]</td><td>0.394</td><td>0.533 [.50,.56]</td><td>0.213</td><td>0.193 [.15,.24]</td></tr><tr><td>SigLIP-L</td><td>Language</td><td>0.271 [.23,.35]</td><td>0.103 [.03,.15]</td><td>0.506</td><td>0.476 [.44,.51]</td><td>0.264</td><td>0.171 [.12,.22]</td></tr><tr><td>ViT-L (IN1k)</td><td>Class</td><td>0.328 [.27,.41]</td><td>0.112 [.04,.16]</td><td>0.510</td><td>0.457 [.43,.49]</td><td>0.169</td><td>0.163 [.13,.20]</td></tr><tr><td>OpenCLIP-L</td><td>Language</td><td>0.312 [.24,.37]</td><td>0.118 [.05,.16]</td><td>0.352</td><td>0.444 [.41,.48]</td><td>0.163</td><td>-0.027 [-.08,.02]</td></tr></table>

![](images/84f208461aa90f1a42d630af160e93a5151b8778c94ae9d389ba2d81e9810ef2.jpg)  
Fig. 5. Agreement with graded human color structure, $\rho _ { \mathrm { f u z z y } } ,$ across relative ViT depth for all eleven encoders. Color denotes training regime; dashed lines indicate base variants.

## V. DISCUSSION

To better understand the significance of our findings, we compare them with recent studies on human-like color and visual representations in neural networks.

Across the eleven encoders, category boundaries and cluster compactness look much the same, yet the graded alignment measured after controlling for color geometry pulls them apart, with masked reconstruction standing well above the rest. Since these models also differ in training data, scale, patch size, and recipe, we read this as an association with the training regime rather than a causal effect of the objective on its own. Earlier work points the same way, finding that training data and objectives shape human–model alignment more than architecture does [14], [15].

A plausible reason is that reconstruction has to preserve lowlevel detail. Color is exactly such a property, so an objective that rebuilds pixels may retain it more faithfully than one organized around semantic classes or image–text matching. This fits evidence that MAEs keep low-level visual structure while diverging from models tuned for higher-level neural representations [17]. It stays an interpretation rather than a causal claim, given how many training choices vary at once.

## VI. CONCLUSION

We introduced a framework for evaluating color grounding with a graded fuzzy model of human color categories. The results show why color grounding should not be summarized by a single score. Across eleven ViT encoders, category boundaries and cluster compactness are broadly similar. The models differ more clearly in their agreement with graded human structure after color geometry is controlled. On this measure, the two MAE models achieve the highest scores and are separated from the other encoders by non-overlapping confidence intervals. These findings are important because they show that models can organize color categories similarly while still differing in how well they reflect human color perception

The study has several limitations. The 330 WCS chips cover only 50 of the 86 COLIBRI categories as dominant labels, and the stimuli are not statistically independent. The evaluated models differ in data, scale, patch size, training recipe, and objective, so a controlled comparison is needed to isolate the source of the MAE result. The natural-image experiment uses whole-image embeddings and does not include object masks. Finally, all evaluated models belong to the ViT family. Future work should vary the training objective under controlled conditions, include other architectural families, and evaluate object-segmented natural scenes.

## ACKNOWLEDGMENT

This research has been funded by the Science Committee of the Ministry of Science and Higher Education of the Republic of Kazakhstan (Grant No. AP22786412).

## REFERENCES

[1] K. Han, Y. Wang, H. Chen, X. Chen, J. Guo, Z. Liu, Y. Tang, A. Xiao, C. Xu, Y. Xu, Z. Yang, Y. Zhang, and D. Tao, “A survey on vision transformer,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 45, no. 1, pp. 87–110, 2023.

[2] J. Zhang, J. Huang, S. Jin, and S. Lu, “Vision-language models for vision tasks: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 46, no. 8, pp. 5625–5644, 2024.

[3] K. Siuda-Krzywicka, C. Witzel, E. Chabani, M. Taga, C. Coste, N. Cools, S. Ferrieux, L. Cohen, T. Seidel Malkinson, and P. Bartolomeo, “Color categorization independent of color naming,” Cell Reports, vol. 28, no. 10, pp. 2471–2479.e5, 2019.

[4] M. Abdou, A. Kulmizev, D. Hershcovich, S. Frank, E. Pavlick, and A. Søgaard, “Can language models encode perceptual structure without grounding? a case study in color,” in Proceedings of the 25th Conference on Computational Natural Language Learning, A. Bisazza and O. Abend, Eds. Online: Association for Computational Linguistics, Nov. 2021, pp. 109–132.

[5] M. Muratbekova, N. Toganas, A. Igali, M. Shagyrov, E. Kadyrgali, A. Yerkin, and P. Shamoi, “Color models in image processing: a review and experimental comparison,” Discover Applied Sciences, vol. 8, no. 5, Mar. 2026.

[6] R. Patel and E. Pavlick, “Mapping language models to grounded conceptual spaces,” in International conference on learning representations, 2022.

[7] A. Søgaard, “Grounding the vector space of an octopus: Word meaning from raw text,” Minds and Machines, vol. 33, no. 1, pp. 33–54, 2023.

[8] P. Kay and R. S. Cook, “World color survey,” in Encyclopedia of color science and technology. Springer, 2023, pp. 1601–1607.

[9] P. Shamoi, N. Toganas, M. Muratbekova, E. Kadyrgali, A. Yerkin, A. Igali, M. Ziyada, A. Adilova, A. Karatayev, and Y. Torekhan, “Colibri fuzzy model: Color linguistic-based representation and interpretation,” IEEE Access, vol. 13, pp. 205 932–205 956, 2025.

[10] B. Berlin and P. Kay, Basic Color Terms: Their Universality and Evolution. University of California Press, 1991.

[11] N. Zaslavsky, C. Kemp, T. Regier, and N. Tishby, “Efficient compression in color naming and its evolution,” Proceedings of the National Academy of Sciences, vol. 115, no. 31, pp. 7937–7942, 2018.

[12] N. Kriegeskorte, “Representational similarity analysis – connecting the branches of systems neuroscience,” Frontiers in Systems Neuroscience, 2008.

[13] I. Sucholutsky, L. Muttenthaler, A. Weller, A. Peng, A. Bobu, B. Kim, B. C. Love, E. Grant, I. Groen, J. Achterberg et al., “Getting aligned on representational alignment,” arXiv preprint arXiv:2310.13018, 2023.

[14] L. Muttenthaler, J. Dippel, L. Linhardt, R. A. Vandermeulen, and S. Kornblith, “Human alignment of neural network representations,” arXiv preprint arXiv:2211.01201, 2022.

[15] C. Conwell, J. S. Prince, K. N. Kay, G. A. Alvarez, and T. Konkle, “A large-scale examination of inductive biases shaping high-level visual representation in brains and machines,” Nature communications, vol. 15, no. 1, p. 9383, 2024.

[16] Y. Liang, M. Li, C. Fan, Z. Li, D. Nguyen, K. Cobbina, S. Bhardwaj, J. Chen, F. Liu, and T. Zhou, “Colorbench: Can vlms see and understand the colorful world? a comprehensive benchmark for color perception, reasoning, and robustness,” Advances in Neural Information Processing Systems, vol. 38, 2026.

[17] H. Karimi and S. Anzellotti, “Do masked autoencoders learn a humanlike geometry of neural representation? divergence and convergence across brains and machines during naturalistic vision,” in NeurIPS 2025 Workshop on Symmetry and Geometry in Neural Representations, 2025.

[18] A. Akbarinia, “Exploring the categorical nature of colour perception: Insights from artificial networks,” Neural Networks, vol. 181, p. 106758, 2025.

[19] N. R. Wickramanayaka and M. Oizumi, “Systematic comparison of color representations between humans and deep neural networks: towards predicting human color perception in a vast color space,” bioRxiv, 2025, preprint.

[20] R. Benavente, M. Vanrell, and R. Baldrich, “Parametric fuzzy sets for automatic color naming,” Journal of the Optical Society of America A, vol. 25, no. 10, pp. 2582–2593, 2008.

[21] G. Sharma, W. Wu, and E. N. Dalal, “The ciede2000 color-difference formula: Implementation notes, supplementary test data, and mathematical observations,” Color Research & Application, vol. 30, no. 1, pp. 21–30, 2005.

[22] M.-C. Chiu, S. Wen, P.-Y. Chen, and X. Ma, “Megacoin: enhancing medium-grained color perception for vision-language models,” in Proceedings of the AAAI’26/IAAI’26/EAAI’26. AAAI Press, 2026.

[23] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., “Learning transferable visual models from natural language supervision,” in International conference on machine learning. PmLR, 2021, pp. 8748–8763.

[24] M. Cherti, R. Beaumont, R. Wightman, M. Wortsman, G. Ilharco, C. Gordon, C. Schuhmann, L. Schmidt, and J. Jitsev, “Reproducible scaling laws for contrastive language-image learning,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2023, pp. 2818–2829.

[25] X. Zhai, B. Mustafa, A. Kolesnikov, and L. Beyer, “Sigmoid loss for language image pre-training,” in Proceedings of the IEEE/CVF international conference on computer vision, 2023, pp. 11 975–11 986.

[26] M. Tschannen, A. Gritsenko, X. Wang, M. F. Naeem, I. Alabdulmohsin, N. Parthasarathy, T. Evans, L. Beyer, Y. Xia, B. Mustafa et al., “Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features,” arXiv preprint arXiv:2502.14786, 2025.

[27] A. Dosovitskiy, L. Beyer, A. Kolesnikov, D. Weissenborn, X. Zhai, T. Unterthiner, M. Dehghani, M. Minderer, G. Heigold, S. Gelly et al., “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.

[28] K. He, X. Chen, S. Xie, Y. Li, P. Dollar, and R. Girshick, “Masked au-´ toencoders are scalable vision learners,” in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, 2022, pp. 16 000–16 009.

[29] M. Oquab, T. Darcet, T. Moutakanni, H. Vo, M. Szafraniec, V. Khalidov, P. Fernandez, D. Haziza, F. Massa, A. El-Nouby et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023.