# Evaluating the Interpretability of Sparse Autoencoders with Concept Annotations

Jonas Klotz<sup>1,2</sup> , Cassio F. Dantas<sup>3,5</sup> , Pallavi Jain<sup>4,5</sup> , Diego Marcos<sup>4,5</sup> , and Begüm Demir<sup>1,2</sup>

<sup>1</sup> The Berlin Institute for the Foundations of Learning and Data (BIFOLD)

<sup>2</sup> Technische Universität Berlin, Germany {j.klotz,demir}@tu-berlin.de

<sup>3</sup> INRAE cassio.fraga-dantas@inrae.fr <sup>4</sup> Inria, EVERGREEN {pallavi.jain, diego.marcos}@inria.fr <sup>5</sup> UMR TETIS, Univ Montpellier, France

Abstract. Sparse autoencoders (SAEs) are increasingly used to extract interpretable concepts from vision and vision language models, yet existing evaluation methods largely rely on proxy metrics or qualitative inspection rather than measuring semantic correspondence. We present a human-grounded evaluation framework that quantifies alignment between SAE latents and human-annotated concepts, without requiring user studies, and validate this matching through targeted attribute perturbations. To enable this intervention-style evaluation in vision, we construct synCUB and synCOCO, synthetic benchmarks of paired images that difer in exactly one attribute. We introduce Fully-Binary Matching Pursuit (FBMP), a coalition-based matching procedure that supports many-to-one mappings between SAE latents and annotated concepts, and consistently outperforms one-to-one baselines. For functional validation, we propose a Targeted Attribute Perturbation Alignment Score (TAPAScore), which tests whether matched concepts respond selectively and in the expected direction under targeted image-level attribute perturbations. Under sanity checks, our matching and TAPAScore are the only evaluated metrics that reliably distinguish trained SAEs from untrained ones. Across SAEs trained on CLIP and DINOv2 embeddings, we find that increased overcompleteness can reduce perturbation alignment, indicating a reduction in interpretability. Our evaluation framework suggests that moderate dictionary sizes provide the best trade-of, yielding the most interpretable SAEs. Code and datasets are available at https://github.com/JonasKlotz/sae-concept-eval.

Keywords: Sparse Autoencoders · Interpretability

## 1 Introduction

Large vision and vision-language models such as DINOv2 [35] and CLIP [39] rely on high-dimensional internal representations whose structure is dificult to interpret. Understanding what individual latent dimensions represent is therefore critical for trust, controllability, and scientific analysis of these models [44].

![](images/2958f92a51eb44ac712384d158fbabc0a1a6c9364f9568407f4e0635080a65df.jpg)  
2 Synthetic DatasetsLatent-Concept Matching Targeted Attribute Perturbation Alignment31  
Fig. 1: Human-grounded evaluation of SAE interpretability. The framework comprises three components: (1) latent-concept matching, where binarized SAE latents are aligned to ground truth attribute vectors; (2) synthetic datasets, where paired images are constructed to difer in exactly one concept (e.g., belly pattern changed from solid to striped); and (3) targeted attribute perturbation alignment, where the induced latent change is measured via the signed responses $\delta _ { \mathrm { a d d } }$ and $\delta _ { \mathrm { r e m } }$ . TAPAScore combines these components to quantify the perturbation alignment.

One promising approach is the use of sparse autoencoders (SAEs) to inspect their internal representations, a technique first developed for large language models [5, 21]. By enforcing sparsity, SAEs decompose high-dimensional activations into more localized and disentangled components that can be aligned with human-interpretable concepts. Following their success in language-model interpretability, SAEs are being applied to vision encoders to identify sparse latents associated with object parts, textures, and attributes [3, 13, 28, 41].

However, SAEs are known to exhibit systematic failure modes, such as feature splitting, where a single concept is fragmented across multiple latents [5]; feature absorption, where a general feature develops blind spots covered by specialized latents [10]; and feature composition, where co-occurring concepts merge into a single latent [53]. SAE features are furthermore often organized hierarchically, with both fine-grained and higher-level abstractions distributed across latents [8]. Unless the SAE captures inductive biases that reflect the semantic structure present in the data, there is no guarantee that learned features will align with the concepts that human observers identify as meaningful [14]. Consequently, we claim that the alignment between SAE features and humanunderstandable concepts cannot be assumed from architectural design or reconstruction quality alone, but must be evaluated explicitly. Such explicit evaluation requires metrics that capture semantic correspondence. Existing SAE metrics, developed primarily in the language domain, fall into structural and functional families [46]. Whereas structural metrics, which assess representational properties such as sparsity or reconstruction fidelity, transfer naturally from language to vision, functional evaluation requires controlled interventions that isolate a single semantic change. This condition is particularly dificult to satisfy in vision, where attribute changes in images rarely occur in isolation. As a result, SAE evaluation for vision models remains dominated by qualitative examples and structural proxies, which measure representational organization rather than whether features correspond to semantic concepts.

To address this limitation, we operationalize interpretability as the degree to which learned SAE features align with human-understandable concepts present in the data, as in prior work on representation interpretability [2, 23]. Human evaluation is considered the gold standard of interpretability assessment [11], and human-annotated concepts/attributes provide a practical measure of this standard, as they precisely reflect the factors that human observers consistently identify as meaningful. Figure 1 summarizes our proposed evaluation framework. We first quantify semantic correspondence by matching binarized SAE activations to binary ground truth attribute vectors. We then construct controlled image pairs that difer in exactly one attribute, enabling intervention-style tests where the target change is precisely known. Finally, we introduce TAPAScore, which measures whether the latents previously matched to the perturbed attribute respond in the expected direction. In summary, our contributions are:

– We propose matching metrics between SAE latents and human-annotated attributes, including a coalition-based formulation, fully-binary matching pursuit (FBMP), that supports many-to-one mappings. We find that existing metrics fail basic sanity checks that our proposed metrics pass (Fig. 4) and that FBMP consistently outperforms one-to-one matchings (Figs. 5, 6, top).

– We construct two synthetic datasets, synCUB and synCOCO, consisting of paired images that difer in exactly one attribute or object label, enabling intervention-style evaluation of SAEs in vision.

– We propose a targeted attribute perturbation alignment score (TAPAScore) that tests whether matched latent sets respond selectively and in the right direction under targeted attribute perturbations. We show that increased overcompleteness can degrade perturbation alignment (Figs. 5, 6, bottom), and that matching score and TAPAScore are positively correlated (Fig. 7).

Beyond the framework itself, our evaluation yields consistent empirical guidance for practitioners: increasing overcompleteness improves statistical matching but tends to degrade perturbation alignment; moderate dictionary sizes ofer the best trade-of. Based on these findings, we recommend FBMP $F _ { 0 . 5 }$ matching paired with TAPAScore as the default evaluation protocol.

## 2 Background and Related Work

Sparse Autoencoders (SAEs) [5, 21] are a method to solve sparse dictionary learning, where signals are represented as sparse linear combinations of basis elements (atoms) from an overcomplete dictionary [43, 48]. More recently, SAEs have been used to identify interpretable concepts in neural network activations. This is motivated by evidence that individual neurons are frequently polysemantic and encode multiple unrelated concepts, suggesting that deep networks store information in superposed representations [12]. SAEs disentangle these representations by learning a sparse and overcomplete representation in which the number of learned features exceeds the dimensionality of the activation space. SAEs implement dictionary learning through a neural network consisting of an encoder $\mathbf { W } _ { \mathrm { e n c } } \in \mathbb { R } ^ { D \times L }$ , decoder $\mathbf { W } _ { \mathrm { d e c } } \in \mathbb { R } ^ { L \times D }$ , and shared bias b $\in \mathbb { R } ^ { D }$ . Given an embedding $\mathbf { x } \in \mathbb { R } ^ { D }$ from a pretrained model, the SAE computes sparse activations $\mathbf { z } \in \mathbb { R } ^ { L }$ and the corresponding reconstruction $\hat { \bf x }$ as:

$$
\mathbf {z} = \sigma \big (\mathbf {W} _ {\mathrm{enc}} ^ {\top} (\mathbf {x} - \mathbf {b}) \big), \qquad \hat {\mathbf {x}} = \mathbf {W} _ {\mathrm{dec}} ^ {\top} \mathbf {z} + \mathbf {b}.\tag{1}
$$

The model parameters are learned by minimizing an objective $\mathcal { L } ( \mathbf { x } ) = \mathcal { R } ( \mathbf { x } ) +$ $\lambda S ( \mathbf { x } )$ with a reconstruction and a sparsity-inducing term. SAE architectures primarily difer in how sparsity is enforced. Vanilla SAEs [5] use ReLU activation with $L ^ { 1 }$ penalty and $L ^ { 2 }$ reconstruction. TopK SAEs [16] and BatchTopK SAEs [7] replace soft sparsity penalties with hard constraints, retaining only the K largest activations per sample or batch. JumpReLU SAEs [40] instead learn a per-latent activation threshold under a direct $L ^ { 0 }$ penalty. Matryoshka SAEs [8] learn nested dictionaries with grouped activations.

Evaluation Metrics in Language-Model Interpretability. Evaluation metrics for SAEs are commonly categorized into structural and functional metrics [46]. Structural metrics assess whether the learned representation of the SAE preserves properties of the original embedding space, including reconstruction error, recovery of known features, and ablation-based diagnostics [16,30,47]. The centered kernel nearest neighbor alignment (CKNNA) metric [55] quantifies how well the neighborhood structure of the original embedding space is preserved after transformation. While essential for diagnosing training behavior, these measures do not directly evaluate whether learned features correspond to meaningful semantic concepts. Functional metrics assess whether SAE representations are semantically interpretable and practically useful. Feature Monosemanticity Score (FMS) [18] trains a predictor to relate latents to semantic attributes, while automated interpretability approaches [38] score features by prompting an LLM to generate and test natural language explanations. A complementary direction focuses on intervention-based evaluation, testing whether manipulating identified features produces the intended semantic efect [4,32,45], a paradigm widely applied in mechanistic analysis [20] and activation steering [17,42]. However, applying this paradigm to vision is non-trivial, as it often requires counterfactual data that isolates the influence of individual visual attributes.

SAEs for Vision Models. While SAEs were initially developed and widely applied in natural language processing [10, 46], recent work has transferred this methodology to vision models to recover sparse visual features corresponding to object parts, textures, attributes, or other interpretable concepts [3, 13, 15, 28], enabling applications such as concept discovery, representation probing, and interpretability analysis [34, 41, 51, 55]. Empirical studies further indicate that the internal representations of the CLIP vision encoder organize along interpretable concept directions recoverable through sparse feature learning [3, 41, 51]. Evaluating vision SAEs, however, presents unique challenges: while structural metrics transfer across domains, functional evaluation must be adapted to visual semantics. Pach et al. [36] introduce the MonoSemanticity (MS) score, measuring how consistently a latent is activated by semantically similar inputs, computed as the activation-weighted similarity between images that strongly activate it. Complementary steerability metrics quantify how strongly manipulating a feature alters model output distributions and concept coverage [22].

An approach towards ground-truth evaluation uses synthetic benchmarks with known generative factors. Fel et al. [14] propose a Soft Identifiability Benchmark where images are constructed by collaging distinct objects, evaluating SAE features by whether each object class has a corresponding latent that activates when present. However, the synthetic scenes lack visual richness, which may limit the evaluation of larger SAEs. The SUB [1] dataset ofers more realistic concept-level evaluation via CUB-derived bird images with controlled concept substitutions, but is designed for concept bottleneck models rather than SAEs, leaving a gap for ground-truth evaluation on complex data.

## 3 Human-Grounded Evaluation of SAE Concepts

Evaluating whether SAE features correspond to meaningful concepts requires a concrete operationalization of interpretability. Following prior work [2, 23], we define interpretability as the degree to which SAE features align with humanunderstandable concepts, and use human-annotated concepts as a scalable proxy for human judgment. This forms the basis of two complementary evaluation components: latent-concept matching, which quantifies statistical alignment between SAE latents and annotated concepts, and targeted attribute perturbation alignment, which tests whether matched latents respond selectively and in the correct direction to images difering in exactly one concept.

## 3.1 Latent-Concept Matching

To measure statistical alignment between SAE latents and human-annotated concepts, we compare binary ground-truth attribute annotations with binarized SAE activations. Let $\mathbf { Y } \in \{ 0 , \overset { \vartriangle } { 1 } \} ^ { A \times N }$ denote the attribute annotation matrix for N samples and A attributes, and let $\mathbf { y } _ { a } \in \{ 0 , 1 \} ^ { N }$ be the row corresponding to attribute a. For a latent unit l, let $\mathbf { z } _ { l } \in \mathbb { R } ^ { N }$ denote its activation vector across all samples, where $( { \bf z } _ { l } ) _ { i }$ is the activation produced by the SAE for sample i. We define the binarized activation vector $\bar { \mathbf { z } _ { \mathrm { b i n } } ^ { l } } \in \{ 0 , 1 \} ^ { \bar { N } }$ such that $( \mathbf { z } _ { \mathrm { { b i n } } } ^ { l } ) _ { i } = 1$ if $( { \bf z } _ { l } ) _ { i } > 0$ and 0 otherwise. Alignment between latent unit l and attribute a is then evaluated by comparing $\mathbf { z } _ { \mathrm { b i n } } ^ { l }$ and $\mathbf { y } _ { a }$ using standard binary matching metrics. One-to-one matching. The SAE latent that best aligns with attribute a under the $F _ { 1 }$ score is given by:

$$
l _ {a} ^ {*} = \arg \max _ {l} F _ {1} (\mathbf {y} _ {a}, \mathbf {z} _ {\mathrm{bin}} ^ {l}).\tag{2}
$$

One-to-one matching assumes that each attribute is represented by exactly one SAE latent. However, feature splitting [5] causes a unified concept to fragment across multiple specialized latents. For instance, ‘striped belly’ may be split into one latent responding to stripe pattern and one to belly region, such that neither alone achieves high $F _ { 1 }$ even though together they fully encode the attribute. This motivates a many-to-one matching formulation, described next.

Many-to-one matching (Fully-Binary Matching Pursuit). Rather than selecting the single best-aligned latent, we seek a small subset (coalition) of latents whose combined activations better reconstruct a given attribute annotation. A naïve approach would select the top-k latents by individual binary similarity score $( \mathrm { e . g . , } F _ { 1 } )$ , but this tends to produce redundant coalitions rather than complementary ones. We therefore adopt a sequential selection procedure that greedily builds a more informative coalition. The proposed approach is based on well-established greedy techniques from sparse reconstruction literature, namely Matching Pursuit [31] and its variants [37,54]. At each step, the latent that best complements the current coalition is selected, and its contribution is removed from the residual before the next selection. This ensures that each newly added latent is complementary to the previously selected ones rather than redundant. Although best subset selection is NP-hard [33], greedy approaches of this kind have proven efective in practice and optimal under certain conditions [49, 50].

We propose Fully-Binary Matching Pursuit (FBMP), a variant of Matching Pursuit tailored for the reconstruction of binary signals. While binary matching pursuit has been explored in prior work [27, 54], existing formulations target binary coeficient vectors while leaving the input vector and candidate atoms continuous. In our setting, both the attribute annotation vectors and the latent activations are binary, which precludes the use of standard inner products and vector sums. We therefore adapt the matching pursuit procedure to operate entirely in the binary domain. The resulting approach, described in Algorithm 1, is, to the best of our knowledge, novel. In the proposed procedure: (1) a binary similarity metric $( \mathrm { e . g . } , F _ { \beta }$ score<sup>6</sup>) is used instead of standard inner-product-based correlation for best-atom selection; (2) logical OR (∨) operations are used to sum the contributions of each atom in the reconstruction, instead of simple sums; (3) logical AND (∧) and NOT (¬) operations are used to update residuals instead of standard subtraction operations. A stopping criterion terminates the algorithm as soon as a newly-selected latent no longer improves the $F _ { 1 }$ score of the current approximation. Although the maximum coalition size k may be set to a large value, the algorithm typically returns a smaller subset (see Appendix S2.2).

While FBMP is many-to-one per attribute call, as each attribute selects a subset of latents, the overall matching is many-to-many at the set level: a latent can be selected by multiple attributes simultaneously (see Fig. S9). Binarizing the latent activations discards magnitude information. This is deliberate: latents whose magnitudes encode several distinct concepts [25] are penalized precisely because we treat such magnitude-encoded multiplicity as insuficient disentanglement. FBMP nonetheless outperforms a magnitude-aware non-negative orthogonal matching pursuit [6] baseline in causal alignment at matched sparsity and without thresholding (Sec. S6.3).

Matching Score. Given a set of concepts $ { \boldsymbol { S } } _ { a }$ matched to attribute $^ { a , }$ the corresponding matching score is defined as the $F _ { 1 }$ similarity between the coalition of selected latents and the ground-truth annotations ${ \bf y } _ { a }$ . The overall MATCHScore is then obtained by averaging over attributes:

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 Fully-Binary Matching Pursuit (FBMP)

Require: Set of binarized latent activation vectors  $z_{bin}^{l} \in \{0, 1\}^{N}$ , with  $l \in \{1, \ldots, L\}$ 

Target attribute annotations  $y \in \{0, 1\}^{N}$ 

Max iterations k (max subset size),  $\beta \leq 1$  (for concept selection criterion), Binary similarity metric SIM $_{bin}$  (e.g.,  $F_{\beta}$ ).

Ensure: Latents subset  $S \subset \{1, \ldots, L\}$  that well approximate y

1:  $r^{0} \leftarrow y$    ▷ Initialize residual

2:  $\hat{y}^{0} \leftarrow 0$    ▷ Initialize approximation

3:  $S \leftarrow \emptyset$    ▷ Initialize empty subset of selected concepts

4: for  $\kappa = 0$  to k - 1 do

5:  $l^{*} \leftarrow \arg\max_{l} SIM_{\text{bin}}(\mathbf{r}^{\kappa}, \mathbf{z}_{\text{bin}}^{l})$    ▷ Select most-aligned concept

6:  $r^{\kappa+1} \leftarrow r^{\kappa} \wedge \neg z_{\text{bin}}^{l^{*}}$    ▷ Update residual

7:  $\hat{y}^{\kappa+1} \leftarrow \hat{y}^{\kappa} \vee z_{\text{bin}}^{l^{*}}$    ▷ Update approximation

8: if  $F_{1}(y, \hat{y}^{\kappa+1}) \leq F_{1}(y, \hat{y}^{\kappa})$  then    ▷ Stopping criterion

9: break

10:  $S \leftarrow S \cup \{l^{*}\}$    ▷ Add to subset

11: return S
</div>

$$
\mathrm{MATCHScore} = \frac {1}{A} \sum_ {a = 1} ^ {A} F _ {1} (\mathbf {y} _ {a}, \bigwedge_ {l \in \mathcal {S} _ {a}} \mathbf {z} _ {\mathrm{bin}} ^ {l})\tag{3}
$$

For one-to-one matching, we simply have ${ \cal S } _ { a } = \{ l _ { a } ^ { * } \}$ as defined in Eq. (2). Finally, to enable fair comparisons between SAEs of diferent sizes, we introduce an adjusted matching score. Larger dictionaries provide a larger pool of candidate latent units, which artificially inflates matching performance (Fig. S27). To account for this efect, we subtract $F _ { 1 , \mathrm { r n g } } ,$ the matching score achieved by an untrained SAE (random weights) using the same matching strategy, to compute the ∆MATCHScore, which reflects improvement over a random baseline:

$$
\Delta \text { MATCHScore } = \text { MATCHScore } - F _ {1, \text { rng }}.\tag{4}
$$

## 3.2 Evaluation via Targeted Attribute Perturbation Alignment

A high latent-concept matching score indicates statistical alignment between the SAE’s latent activations and the annotated attributes. However, correlation does not imply that a latent encodes the underlying semantic attribute. A latent may correlate with an attribute due to confounding structure, co-occurrence patterns, or background cues, without encoding the semantic factor itself. If a latent truly encodes an attribute, then changing that attribute while holding all other attributes constant should induce a structured and directionally consistent change in the latent activation. We use targeted perturbations to measure this change, by isolating a single semantic attribute while minimizing confounding variation. By constructing paired inputs $( { \bf x } , \hat { \bf x } )$ that difer in exactly one annotated attribute, we can test whether latents matched to the perturbed attribute respond in the correct direction, increasing for additions and decreasing for removals.

Let x denote the original image embedding produced by a pretrained model $f ,$ and let xˆ denote the perturbed version that difers in exactly one attribute. Let $\mathbf { z } _ { \mathrm { b i n } } , \hat { \mathbf { z } } _ { \mathrm { b i n } } \in \mathbb { R } ^ { L }$ denote the binarized SAE latent representations of x and $\hat { \mathbf { x } } ,$ respectively, with dictionary size L. Using the latent-concept matching defined previously, let $I _ { \mathrm { a d d } } \subset \{ 1 , \ldots , L \}$ denote the set of latents matched to the attribute being added, and $I _ { \mathrm { r e m } }$ those matched to the attribute being removed (see Fig. 1). We aggregate over each matched set via the maximum (equivalently, a logical OR): the concept is considered active if at least one of its latents fires. The signed response for a single image pair is:

$$
\delta_ {\mathrm{add}} = \max _ {i \in I _ {\mathrm{add}}} \hat {\mathbf {z}} _ {\mathrm{bin}} ^ {i} - \max _ {i \in I _ {\mathrm{add}}} \mathbf {z} _ {\mathrm{bin}} ^ {i}, \quad \mathrm{and} \quad \delta_ {\mathrm{rem}} = \max _ {i \in I _ {\mathrm{rem}}} \hat {\mathbf {z}} _ {b i n} ^ {i} - \max _ {i \in I _ {\mathrm{rem}}} \mathbf {z} _ {b i n} ^ {i}.\tag{5}
$$

Averaging over all P image pairs in the dataset yields:

$$
\varDelta_ {\mathrm{add}} = \frac {1}{P} \sum_ {p} \delta_ {\mathrm{add}} ^ {(p)}, \quad \mathrm{and} \quad \varDelta_ {\mathrm{rem}} = \frac {1}{P} \sum_ {p} \delta_ {\mathrm{rem}} ^ {(p)}.\tag{6}
$$

Finally, we define the Targeted $\underline { { \mathbf { A } } } \mathbf { t }$ tribute Perturbation Alignment Score (TAPAScore) as:

$$
\mathrm{TAPAScore} = \varDelta_ {\mathrm{add}} - \varDelta_ {\mathrm{rem}}.\tag{7}
$$

For datasets containing only removal perturbations (e.g., synCOCO), we set $\varDelta _ { \mathrm { a d d } } = 0$ . A positive TAPAScore indicates that matched latents respond selectively and in the correct direction under targeted attribute perturbations.

## 3.3 Synthetic Dataset generation

Evaluating TAPAScore requires image pairs that difer in exactly one semantic attribute while keeping all other factors constant. Since no such benchmark exists for naturalistic images, we construct two synthetic datasets tailored to this requirement: synCUB, which targets fine-grained attribute perturbations in single-object bird images, and synCOCO, which targets object removal in complex multi-object scenes. To the best of our knowledge, this constitutes the first intervention-style benchmark for SAE evaluation in naturalistic images, as prior work has been largely restricted to simple controlled settings [14].

![](images/72c5d0db662cde0a9eba510489aca81e85890e58250f37767705ec09f4c92643.jpg)  
Fig. 2: synCUB pairs: a reference image guides the target attribute (e.g., breast pattern: solid → spotted), while the base image preserves identity, pose, and background. Synthetic CUB (synCUB, attribute perturbations). CUB-200-2011 [52] provides dense, per-instance attribute annotations across 312 attributes covering color, shape, and pattern across bird parts, making it a natural starting point

for attribute-level intervention evaluation. We construct synCUB following the SUB benchmark [1], which introduces attribute variation by generating prototypical class examples with uncommon attributes. However, SUB pairs images across diferent class instances rather than the same bird before and after a single attribute change, making it unsuitable for image-level intervention. We therefore construct synCUB, where each pair (x, xˆ) difers in exactly one target attribute while all other annotated attributes remain unchanged (Fig. 2), restricted to the 33-class subset and 45 attribute concepts of SUB. For each target attribute, a base image and a reference image exhibiting the desired state are passed to Flux2 [26], which edits the base image (preserving identity, pose, and background) while the reference guides the target attribute. We validate edits with an attribute predictor and manually curate failures (Appendix Sec. S3.4).

Removed: Umbrella  
![](images/5292893fbf1a757b211b16cebc4ebcfc80601454f102171010efdd214e110082.jpg)  
Umbrella

![](images/18d7af0d39e4c64f040652b29ade1fad442f6a159fb1d478c8d10f0faf48ba8b.jpg)

![](images/715cde6140b4ab94071663e6d43fdb439f55f479be86b12c678c7acf3bef062b.jpg)  
Car

![](images/3c5a569aec15bdf30e53ec109bfcf8a6e5259560c0fbea64ec12446698745969.jpg)  
Removed: Car  
Fig. 3: synCOCO pairs: the target object (e.g., umbrella, car) is removed while the remaining scene is preserved.

Synthetic COCO (synCOCO, object removal perturbations). While synCUB evaluates attribute-level alignment in a controlled single-object setting, real-world scenes are considerably more complex. MS-COCO [29] provides a natural complement: its images contain multiple objects, diverse backgrounds, and rich compositional structure absent in CUB. As COCO has no attribute annotations, we treat object labels as high-level semantic concepts and construct synCOCO, where each pair (x, xˆ) difers in one object label, with all other objects preserved (Fig. 3). We select the target object by lowest instance count, breaking ties by largest area, and remove all its instances via Flux2 [26] conditioned on the original image and a fixed removal prompt. We verify removal with the same automatic classifier check as synCUB, followed by manual validation of every retained pair. Full details for both datasets are in Appendix Sec. S3.

## 4 Experimental Results

Experimental Settings. We evaluate SAE interpretability on two controlled benchmarks derived from CUB-200-2011 [52] and MS-COCO [29]. For both datasets, we construct synthetic intervention benchmarks (synCUB and syn-COCO) as described in Sec. 3.3. Latent-concept matching is computed on the original CUB and COCO datasets, while perturbation alignment is evaluated on the synthetic benchmarks. For each dataset, we extract image embeddings from two pretrained vision models, CLIP (ViT-L-14 backbone) and DINOv2 (ViT-S-14 backbone), and train SAEs on these embeddings. We compare four SAE variants: JumpReLU, TopK, BatchTopK, and Matryoshka SAEs, across dictionary sizes {128, 256, 512, 1024, 2048, 4096}. Unless stated otherwise, all SAEs except JumpReLU use TopK sparsity with K=32, which is the average occurrence of attributes in CUB; a sweep over the sparsity K in the Appendix (Sec. S6.4) confirms that moderate sparsity levels provide the best trade-of between matching and perturbation alignment. Full training statistics, including reconstruction loss and sparsity metrics, are listed in the Appendix in Sec. S4. For state-of-the-art comparison, we compare functional proxy metrics commonly used in prior work, namely FMS [18], MS [36], and CKNNA [55]. We evaluate monosemanticity-based scores (FMS [18] and MS [36]), aggregated over all latents. While monosemanticity-based scores were not designed as a metric for evaluating a full SAE, we treat the average over latents as a proxy: if an SAE contains more monosemantic latents, its average score should indicate higher interpretability. For our selection criteria, we evaluate latent-concept matching using classical $F _ { 1 }$ (with k=1) and $F _ { \beta }$ -based matching pursuit (FBMP, with k=3) across $\beta \in \{ 0 . 2 5 , 0 . 5 , 1 \}$ ; an analysis of metric sensitivity over k is provided in the Appendix Sec. S6.1. As a supervised upper bound, we additionally train perattribute logistic regression probes on the raw embeddings and evaluate their thresholded outputs within the same pipeline (Figs. 5 and 6).

![](images/779856b6783c1309486fccad820a14a95eef9148aa6358414296aac5525475f1.jpg)

![](images/d46de6102169e8fec487a9a8e15fa1331d1f99d5f24a8aac527cb6139ec0d84e.jpg)  
Fig. 4: Metrics failure mode comparison aggregated over all dictionary sizes for CUB (left) and COCO (right) with CLIP, across three conditions: trained SAEs, an untrained TopK SAE, and random activations. TAPAScore (computed on the synthetic datasets) and MATCHScore with FBMP clearly drop under untrained and random conditions, while FMS and MS show little sensitivity, and CKNNA inflates for the untrained SAE.

Sanity Check Analysis. Evaluating interpretability metrics is inherently challenging due to the absence of reliable ground truth explanations [19,24]. Drawing on the disentanglement literature, where failure modes of common metrics are well-documented [9], we test whether a metric can distinguish meaningful concepts from spurious or random correspondence. Concretely, we compare three conditions: trained SAEs (with scores aggregated over all dictionary sizes and SAE variants), an untrained TopK SAE, and random activations. We report results for CKNNA [55], FMS [18], and MS [36], alongside our MATCHScore with two criteria (FBMP $F _ { 1 }$ with k=3 and $F _ { 1 }$ with k=1) and the TAPAScore for the respective matching. To compare with the untrained baseline, we evaluate the unnormalized matching score as described in Eq. 3. The metrics and MATCHScores are calculated on the respective dataset (CUB or COCO), whereas TAPAS is calculated on the corresponding synthetic version.

![](images/edff49430b805b04e5ae443d748f490ea632d3d69356808b96943ff8c5c7139c.jpg)  
-- Probe upper bound (F1 k=1) FBMP F0.25 (k=3) FBMP F0.5 (k=3)  FBMP F1 (k=3)  F1 (k=1)  
Fig. 5: $F _ { 1 }$ ∆MATCHScore on CUB (top row) and TAPAScore on synCUB (bottom row) as a function of dictionary size for SAEs trained on CLIP embeddings of CUB, across BatchTopK, Matryoshka, TopK and JumpReLU SAE variants (left to right) and diferent matching criteria. The gray dashed line marks the supervised linearprobe upper bound.

From Fig. 4, one can observe that both matching variants exhibit a pronounced drop when replacing trained SAEs with either untrained weights or random activations. For both datasets, the MATCHScores for FBMP with selection criterion $F _ { 1 } \ ( k { = } 3 )$ and $F _ { 1 } \ ( k { = } 1 )$ show clear absolute decreases in matching $F _ { 1 }$ score under the untrained and random conditions. In contrast, MS shows limited sensitivity to whether the representation is trained, untrained, or random across both datasets, while FMS shows this limitation only on CUB but not on COCO. CKNNA behaves inconsistently: while it shows a similar drop for random activations, it exhibits a strong absolute increase for the untrained SAE, indicating that the untrained SAE achieves higher CKNNA scores than the trained models. TAPAScore shows the strongest separation, dropping to near zero for both untrained and random conditions on both datasets. Overall, only our matching variants and TAPAScore pass the intended sanity checks, clearly distinguishing trained SAEs from both untrained and random conditions.

## 4.1 Latent to Concept Matching

We evaluate semantic alignment between SAE latents and human-annotated concepts across dictionary sizes and SAE variants for CLIP. The matching scores of untrained SAEs increase with dictionary size, as larger dictionaries ofer more latents and thus a greater chance of spurious correlations, a trend we validate in the Appendix (Fig. S27). To enable fair comparison across dictionary sizes, we therefore calculate the ∆MATCHScore (Eq. 4) throughout. The raw scores and results for diferent models are provided in the Appendix (Sec. S6.1).

Fig. 5 (top row ) shows the $F _ { 1 } \varDelta$ matching scores across dictionary sizes for BatchTopK, Matryoshka, TopK, and JumpReLU SAEs on CUB with a CLIP model. For each attribute, we find the best matching latent(s) under a given criterion, either one-to-one $( F _ { 1 } , k { = } 1 )$ or diferent FBMP variants with k=3 (FBMP $F _ { 1 }$ , FBMP $F _ { 0 . 5 } ,$ FBMP $F _ { 0 . 2 5 } )$ . The scores are aggregated over all attributes to obtain a mean $F _ { 1 }$ matching score per dictionary size. We observe that FBMP consistently outperforms its one-to-one counterpart across all SAE variants, consistent with the motivation that attributes may be represented by multiple complementary latents rather than a single unit. Among the FBMP variants, FBMP $F _ { 0 . 5 }$ tends to achieve the highest scores. Matching scores are not monotonically increasing with dictionary size: BatchTopK and Matryoshka peak at dictionary sizes 512 and 256, respectively, after which the score decreases. TopK, in contrast, exhibits a more monotonic increase, peaking at 2048, and therefore outperforms the other variants at large dictionary sizes, though a drop is observed at 4096. JumpReLU shows a dip at dictionary size 256 followed by a steady increase up to 4096, while its one-to-one matching slowly decreases with dictionary size. All SAE variants remain well below the linear-probe upper bound, indicating that only part of the attribute information present in the embeddings is recovered as individual latents.

![](images/6a9c86da364bb0da193fcb04d33a998b8ce4c05a07a323f4d087a174d3d92056.jpg)  
-- Probe upper bound (F1 k=1) FBMP F0.25 (k=3) FBMP F0.5 (k=3)  FBMP F1 (k=3) F1 (k=1)  
Fig. 6: $F _ { 1 }$ ∆MATCHScore on COCO (top row) and TAPAScore on synCOCO (bottom row) as a function of dictionary size for SAEs trained on CLIP embeddings of COCO, across BatchTopK, Matryoshka, TopK and JumpReLU SAE variants (left to right) and diferent matching criteria. The gray dashed line marks the supervised linearprobe upper bound.

Fig. 6 (top row ) reports the $F _ { 1 }$ matching scores for BatchTopK, Matryoshka, TopK, and JumpReLU SAEs trained on COCO with CLIP. Similarly to the CUB results, FBMP consistently outperforms the one-to-one baseline across all SAE variants, and FBMP $F _ { 0 . 5 }$ achieves the highest scores overall. Matching scores are notably higher than on CUB, which we attribute to the smaller and higher-level concept set whose categories are more semantically distinct from one another. Unlike in CUB, BatchTopK and Matryoshka do not exhibit a clear peak followed by a decline; instead, their performance remains stable or continues to improve at larger dictionary sizes, suggesting that the greater visual complexity of COCO requires more latents to fully encode its representations. JumpReLU follows the same pattern, increasing monotonically and saturating around dictionary size

2048, although the gap between FBMP and one-to-one matching is smaller than for the other variants.

## 4.2 Functional Validation of Concept Alignment

While a high matching score indicates statistical alignment between SAE latents and annotated attributes, correlation alone does not imply causal encoding. To validate that matched latents genuinely encode the corresponding semantic attributes, we compute TAPAScore on our synthetic datasets. The SAEs are trained on CLIP embeddings of CUB and COCO. The results for DINOv2 are provided in the Appendix (Sec. S6.2). From Fig. 5 (bottom row), we can observe that for the BatchTopK and Matryoshka SAE, the TAPAScore values closely mirror the matching score, suggesting that statistical and causal alignment are consistent for these variants. In contrast, the TopK SAE exhibits a markedly diferent pattern: while matching scores increase with dictionary size, TAPAScore peaks early and then degrades sharply. This dissociation suggests that larger TopK dictionaries achieve better statistical alignment without encoding concepts that are causally valid. The JumpReLU SAE shows no such severe degradation: its FBMP TAPAScores fluctuate around a stable level across dictionary sizes, and only the one-to-one criterion declines beyond dictionary size 512.

From Fig. 6 (bottom row ), we observe a larger divergence between matching and TAPAScore than in synCUB. $F _ { 1 }$ ∆MATCHScore grows with dictionary size for all SAE variants, saturating around sizes 512–1024. TAPAScore follows a similar trend up to this point, but then declines for larger dictionary sizes in BatchTopK, TopK, and JumpReLU, suggesting that overcompleteness degrades perturbation alignment despite improved matching quality; for JumpReLU this decline only sets in at dictionary size 4096 and is the least pronounced. Matryoshka behaves diferently: FBMP $F _ { 0 . 5 } \ ( k { = } 3 )$ continues to improve beyond 1024, reaching its maximum at dictionary size 2048 before plateauing, suggesting that the nested structure of Matryoshka SAEs requires larger dictionaries to fully capture the representations. However, Matryoshka scores are overall lower than those of the other SAE types. In the Appendix (Sec. S6.5) we further verify that TAPAScore is not inflated by leakage: latents matched to untouched attributes remain stable under perturbation.

![](images/16631eb6c0f87ee05f6dfc838c7a42ee7310048cef7109dcdddaab9ecf3695ca.jpg)

![](images/43fd0980c9f7b769b55bcdec4420e986900cae6890c26ce6de5c30332c77748a.jpg)  
Fig. 7: Correlation between matching score and TAPAScore across SAE configurations on synCUB (top) and synCOCO (bottom).

Correlation analysis. To validate the patterns observed in Secs. 4.1 and 4.2, we examine whether latent-concept matching predicts targeted perturbation alignment. We compute the Pearson correlation between matching scores and TAPAScore across all SAE configurations (variant and dictionary size), with each point in Fig. 7 corresponding to one configuration. On synCUB, we observe a consistent positive correlation across all criteria, with FBMP variants outperforming one-to-one matching, and FBMP $F _ { 0 . 5 }$ achieving the highest correlation. On synCOCO, the correlations are weaker overall, reflecting the divergence between matching and TAPAScore at larger dictionary sizes, where matching keeps improving while TAPAScore declines. Here, FBMP variants yield near-zero correlations, while $F _ { 1 } \ k { = } 1$ yields a negative correlation.

Taken together, the results reveal a consistent picture: statistical and causal alignment largely agree, with higher-matching configurations tending to exhibit stronger perturbation alignment, but the correspondence is not universal. TopK on CUB shows that high matching scores do not guarantee causality, and the COCO results show that overcompleteness can reduce perturbation alignment despite higher matching quality. TAPAScore therefore provides a necessary complement that statistical alignment alone cannot replace, and moderate dictionary sizes achieve the best trade-of between the two. We recommend FBMP $F _ { 0 . 5 }$ with k=3 as the default matching criterion, paired with TAPAScore for causal validation.

## 5 Conclusion

We present a human-grounded framework for evaluating SAE interpretability in vision models, operationalizing interpretability as alignment between learned sparse latents and human-annotated semantic concepts. The framework comprises latent-concept matching criteria, including FBMP, a coalition-based formulation robust to common SAE failure modes; two synthetic intervention benchmarks (synCUB and synCOCO), which provide controlled single-attribute perturbations; and TAPAScore, a functional metric that tests whether matched latents respond selectively and in the correct direction under targeted perturbations. We find that increased overcompleteness can reduce perturbation alignment, indicating a loss of interpretability, and that moderate dictionary sizes provide the best trade-of between semantic coverage and functional selectivity.

The framework has a limitation that suggests a direction for future work. The matching quality is directly bounded by annotation quality and granularity, and since TAPAScore relies on matched latent sets, poor annotations lead to unreliable perturbation alignment estimates. Future work could therefore explore approaches that reduce the dependence on manually curated attribute annotations. Beyond evaluation, a natural extension is to use TAPAScore as a steering tool. Instead of measuring whether matched latents respond to attribute perturbations, they could be actively manipulated to control the predictions of an attribute trained classifier, turning the framework into a test for targeted model intervention.

## References

1. Bader, J., Girrbach, L., Alaniz, S., Akata, Z.: Sub: Benchmarking cbm generalization via synthetic attribute substitutions. Proceedings of the IEEE International Conference on Computer Vision (ICCV) (2025)

2. Bau, D., Zhou, B., Khosla, A., Oliva, A., Torralba, A.: Network dissection: Quantifying interpretability of deep visual representations. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2017)

3. Bhalla, U., Oesterling, A., Srinivas, S., Calmon, F.P., Lakkaraju, H.: Interpreting CLIP with sparse linear concept embeddings (SpLiCE). Advances in Neural Information Processing Systems (NeurIPS) (2024)

4. Bhalla, U., Srinivas, S., Ghandeharioun, A., Lakkaraju, H.: Towards unifying interpretability and control: Evaluation via intervention. ArXiv e-print (2024)

5. Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., et al.: Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread (2023)

6. Bruckstein, A.M., Elad, M., Zibulevsky, M.: Sparse non-negative solution of a linear system of equations is unique. In: 2008 3rd International Symposium on Communications, Control and Signal Processing. pp. 762–767. IEEE (2008)

7. Bussmann, B., Leask, P., Nanda, N.: BatchTopK sparse autoencoders. NeurIPS 2024 Workshop on Scientific Methods for Understanding Deep Learning (2024)

8. Bussmann, B., Nabeshima, N., Karvonen, A., Nanda, N.: Learning multi-level features with matryoshka sparse autoencoders. Proceedings of the International Conference on Machine Learning (ICML) (2025)

9. Carbonneau, M.A., Zaidi, J., Boilard, J., Gagnon, G.: Measuring disentanglement: A review of metrics. IEEE transactions on neural networks and learning systems 35(7), 8747–8761 (2022)

10. Chanin, D., Wilken-Smith, J., Dulka, T., Bhatnagar, H., Golechha, S., Bloom, J.: A is for absorption: Studying feature splitting and absorption in sparse autoencoders. Advances in Neural Information Processing Systems (NeurIPS) (2025)

11. Doshi-Velez, F., Kim, B.: Towards a rigorous science of interpretable machine learning. stat (2017)

12. Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., Hatfield-Dodds, Z., Lasenby, R., Drain, D., Chen, C., et al.: Toy models of superposition. ArXiv e-print (2022)

13. Fel, T., Boutin, V., Béthune, L., Cadène, R., Moayeri, M., Andéol, L., Chalvidal, M., Serre, T.: A holistic approach to unifying automatic concept extraction and concept importance estimation. Advances in Neural Information Processing Systems (NeurIPS) (2023)

14. Fel, T., Lubana, E.S., Prince, J.S., Kowal, M., Boutin, V., Papadimitriou, I., Wang, B., Wattenberg, M., Ba, D.E., Konkle, T.: Archetypal SAE: Adaptive and stable dictionary learning for concept extraction in large vision models. Proceedings of the International Conference on Machine Learning (ICML) (2025)

15. Fel, T., Wang, B., Lepori, M.A., Kowal, M., Lee, A., Balestriero, R., Joseph, S., Lubana, E.S., Konkle, T., Ba, D., et al.: Into the rabbit hull: From task-relevant concepts in DINO to minkowski geometry. Proceedings of the International Conference on Learning Representations (ICLR) (2026)

16. Gao, L., la Tour, T.D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., Wu, J.: Scaling and evaluating sparse autoencoders. Proceedings of the International Conference on Learning Representations (ICLR) (2025)

17. Ghandeharioun, A., Yuan, A., Guerard, M., Reif, E., Lepori, M., Dixon, L.: Who’s asking? user personas and the mechanics of latent misalignment. Advances in Neural Information Processing Systems (NeurIPS) (2024)

18. Härle, R., Friedrich, F., Brack, M., Wäldchen, S., Deiseroth, B., Schramowski, P., Kersting, K.: Measuring and guiding monosemanticity. Advances in Neural Information Processing Systems (NeurIPS) (2025)

19. Hedström, A., Bommer, P., Wickstrøm, K.K., Samek, W., Lapuschkin, S., Höhne, M.M.C.: The meta-evaluation problem in explainable AI: identifying reliable estimators with metaquantus. The Journal of Transactions on Machine Learning Research (TMLR) (2023)

20. Hernandez, E., Sharma, A.S., Haklay, T., Meng, K., Wattenberg, M., Andreas, J., Belinkov, Y., Bau, D.: Linearity of relation decoding in transformer language models. Proceedings of the International Conference on Learning Representations (ICLR) (2024)

21. Huben, R., Cunningham, H., Smith, L.R., Ewart, A., Sharkey, L.: Sparse autoencoders find highly interpretable features in language models. Proceedings of the International Conference on Learning Representations (ICLR) (2024)

22. Joseph, S., Suresh, P., Goldfarb, E., Hufe, L., Gandelsman, Y., Graham, R., Bzdok, D., Samek, W., Richards, B.A.: Steering clip’s vision transformer with sparse autoencoders. Mechanistic Interpretability for Vision at CVPR 2025 (Nonproceedings Track) (2025)

23. Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., et al.: Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (TCAV). Proceedings of the International Conference on Machine Learning (ICML) (2018)

24. Klotz, J., Burgert, T., Demir, B.: On the efectiveness of methods and metrics for explainable AI in remote sensing image scene classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (JSTARS) (2025)

25. Kopf, L., Feldhus, N., Bykov, K., Bommer, P.L., Hedström, A., Höhne, M., Eberle, O.: Capturing polysemanticity with prism: A multi-concept feature description framework. Advances in Neural Information Processing Systems (NeurIPS) (2026)

26. Labs, B.F.: FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2 (2025)

27. Li, H., Ying, H., Liu, X.: Binary generalized orthogonal matching pursuit. Japan Journal of Industrial and Applied Mathematics 41(1), 1–12 (2024)

28. Lim, H., Choi, J., Choo, J., Schneider, S.: Sparse autoencoders reveal selective remapping of visual concepts during adaptation. ArXiv e-print (2024)

29. Lin, T.Y., Maire, M., Belongie, S.J., Bourdev, L.D., Girshick, R.B., Hays, J., Perona, P., Ramanan, D., Dollár, P., Zitnick, C.L.: Microsoft coco: Common objects in context. CoRR (2014)

30. Makelov, A., Lange, G., Nanda, N.: Towards principled evaluations of sparse autoencoders for interpretability and control. Proceedings of the International Conference on Learning Representations (ICLR) (2025)

31. Mallat, S.G., Zhang, Z.: Matching pursuits with time-frequency dictionaries. IEEE Transactions on signal processing 41(12), 3397–3415 (1993)

32. Mueller, A., Brinkmann, J., Li, M., Marks, S., Pal, K., Prakash, N., Rager, C., Sankaranarayanan, A., Sharma, A.S., Sun, J., Todd, E., Bau, D., Belinkov, Y.: The quest for the right mediator: Surveying mechanistic interpretability for nlp through the lens of causal mediation analysis. Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL) pp. 1–48 (2026)

33. Natarajan, B.K.: Sparse approximate solutions to linear systems. SIAM Journal on Computing 24(2), 227–234 (1995)

34. Olson, M.L., Hinck, M., Ratzlaf, N., Li, C., Howard, P., Lal, V., Tseng, S.Y.: Analyzing hierarchical structure in vision models with sparse autoencoders. Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (2025)

35. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., HAZIZA, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.Y., Li, S.W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: DINOv2: Learning robust visual features without supervision. The Journal of Transactions on Machine Learning Research (TMLR) (2024)

36. Pach, M., Karthik, S., Bouniot, Q., Belongie, S., Akata, Z.: Sparse autoencoders learn monosemantic features in vision-language models. Advances in Neural Information Processing Systems (NeurIPS) (2025)

37. Pati, Y.C., Rezaiifar, R., Krishnaprasad, P.S.: Orthogonal matching pursuit: Recursive function approximation with applications to wavelet decomposition. Conference Record of The Twenty-Seventh Asilomar Conference on Signals, Systems and Computers (1993)

38. Paulo, G.S., Mallen, A.T., Juang, C., Belrose, N.: Automatically interpreting millions of features in large language models. In: Proceedings of the International Conference on Machine Learning (ICML) (2025)

39. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al.: Learning transferable visual models from natural language supervision. Proceedings of the International Conference on Machine Learning (ICML) (2021)

40. Rajamanoharan, S., Lieberum, T., Sonnerat, N., Conmy, A., Varma, V., Kramár, J., Nanda, N.: Jumping ahead: Improving reconstruction fidelity with jumprelu sparse autoencoders. ArXiv e-print (2024)

41. Rao, S., Mahajan, S., Böhle, M., Schiele, B.: Discover-then-name: Task-agnostic concept bottlenecks via automated concept discovery. Proceedings of the IEEE European Conference on Computer Vision (ECCV) (2024)

42. Rimsky, N., Gabrieli, N., Schulz, J., Tong, M., Hubinger, E., Turner, A.: Steering llama 2 via contrastive activation addition. Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL) (2024)

43. Rubinstein, R., Bruckstein, A.M., Elad, M.: Dictionaries for sparse representation modeling. Proceedings of the IEEE 98(6), 1045–1057 (2010)

44. Saeed, W., Omlin, C.: Explainable AI (XAI): A systematic meta-survey of current challenges and future opportunities. Knowledge-Based Systems 263, 110273 (2023)

45. Saphra, N., Wiegrefe, S.: Mechanistic? In: Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP (2024)

46. Shu, D., Wu, X., Zhao, H., Rai, D., Yao, Z., Liu, N., Du, M.: A survey on sparse autoencoders: Interpreting the internal mechanisms of large language models. Proceedings of the Annual Meeting of the Association for Computational Linguistics (ACL) (2025)

47. Templeton, A.: Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Anthropic (2024)

48. Tošić, I., Frossard, P.: Dictionary learning. IEEE Signal Processing Magazine 28(2), 27–38 (2011)

49. Tropp, J.A.: Greed is good: Algorithmic results for sparse approximation. IEEE Transactions on Information Theory 50(10), 2231–2242 (2004)

50. Tropp, J.A., Gilbert, A.C.: Signal recovery from random measurements via orthogonal matching pursuit. IEEE Transactions on Information Theory 53(12), 4655–4666 (2007)

51. Vielhaben, J., Bareeva, D., Berend, J., Samek, W., Strodthof, N.: Beyond scalars: Concept-based alignment analysis in vision transformers. Advances in Neural Information Processing Systems (NeurIPS) (2025)

52. Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S.: The caltech-ucsd birds-200-2011 dataset. California Institute of Technology Technical Report (2011)

53. Wattenberg, M., Viégas, F.: Relational composition in neural networks: A survey and call to action. ICML 2024 Workshop on Mechanistic Interpretability (2024)

54. Wen, J., Li, H.: Binary sparse signal recovery with binary matching pursuit. Inverse Problems 37(6), 065014 (2021)

55. Zaigrajew, V., Baniecki, H., Biecek, P.: Interpreting CLIP with hierarchical sparse autoencoders. Proceedings of the International Conference on Machine Learning (ICML) (2025)

## Supplementary Materials

S1: Discussions and Broader Impact S2: Latent–Concept Matching Details S3: Synthetic Dataset Construction Details – S4: Sparse Autoencoder Training Configuration – S5: Qualitative Matching Results – S6: Additional Results

## S1 Discussions and Broader Impact

Limitations and Future Work. The primary limitation of our framework is its dependence on the availability, quality and granularity of human-annotated concepts. Since latent-concept matching relies on binary attribute annotations, the matching quality is directly bounded by annotation completeness. Concepts absent from the annotation set cannot be matched, and noisy or coarse annotations may yield unreliable matchings. Because TAPAScore builds on matched latent sets, this limitation propagates to the functional evaluation stage. A natural direction for future work is therefore to reduce reliance on manually curated annotations, for instance by leveraging vision-language models to automatically generate attribute vocabularies.

Beyond evaluation, our framework opens a natural extension toward active model intervention. Rather than measuring whether matched latents respond to attribute perturbations, one could actively steer those latents to control downstream predictions, turning TAPAScore into a diagnostic for targeted concept manipulation. An interesting open question concerns the nature of unmatched latents; we hypothesize that some of these may capture low-level or background concepts that, while not directly interpretable in terms of human-annotated attributes, nonetheless play an important role in shaping the model’s internal representation. Exploring this direction in the context of fine-grained recognition and vision-language alignment represents a promising avenue for future work.

Broader Impact. Our work contributes tools for auditing the internal representations of large vision models. By grounding interpretability evaluation in human-annotated concepts and intervention-style benchmarks, the framework supports more rigorous transparency and accountability in the deployment of such models. In particular, the ability to detect whether SAE features align with semantically meaningful concepts can help identify unintended or spurious representations that might otherwise go undetected. We do not foresee direct negative societal impacts from this work; however, as with any interpretability method, results should be interpreted with care, as apparent alignment with human concepts does not guarantee that a model is free from bias or other failure modes outside the scope of the evaluated attributes.

## S2 Latent–Concept Matching Details

This section provides full details of the latent–concept matching procedure introduced in Section 3.1 of the main manuscript. We describe the binary matching metrics used to quantify alignment between SAE latents and human-annotated concepts, and outline both the one-to-one and many-to-one matching schemes evaluated in our experiments.

## S2.1 One-to-one Matching

In the one-to-one matching scheme, each attribute a is assigned to exactly one SAE latent l that best captures its activation pattern. To identify this match, we compute standard binary classification metrics between the binarized activation vector $\mathbf { z } _ { \mathrm { b i n } } ^ { l }$ and the ground-truth annotation vector ${ \bf y } _ { a } ,$ and select the latent maximizing the F1 score. Formally, for each attribute a and SAE latent l, we compute precision $P _ { a l }$ and recall $R _ { a l }$ as:

$$
P _ {a l} = \frac {T P _ {a l}}{T P _ {a l} + F P _ {a l} + \varepsilon}, \qquad R _ {a l} = \frac {T P _ {a l}}{T P _ {a l} + F N _ {a l} + \varepsilon},\tag{8}
$$

and define the $F 1 _ { a l }$ score as:

$$
F 1 _ {a l} = \frac {2 P _ {a l} R _ {a l}}{P _ {a l} + R _ {a l} + \varepsilon}.\tag{9}
$$

![](images/7cf082b27f598afde58bdf7d4b5c9278b785f6449a64c95e3e845be5edf9f493.jpg)  
Fig. S8: One-to-one matching procedure using an F1 similarity criterion.

The F1-based matching procedure is summarized in Fig. S8. Although each attribute a is matched to a single latent, nothing prevents the same latent l from being selected for multiple attributes (see Fig. S9). Such a case would suggest a polysemantic nature of said latent. While the same phenomenon may arise under a many-to-one matching, a diferent interpretation can be drawn in this more complex setting. Because an attribute can be decomposed into multiple latents, it now accommodates feature composition [53], where co-occurring semantic concepts (i.e., commonalities between attributes) are captured by a common latent. For example, attributes such as “yellow head”, “yellow crown” and “yellow throat” may share a latent representing the concept “yellow”, while their attribute-specific semantics can be expressed through composition with additional, distinct latents. By construction, such compositional structure cannot be recovered under a strict one-to-one assignment.

![](images/637e5ab0213693aa582b0fa5e9a1e20bcc72a11bf11ba6b312983100d2171854.jpg)  
Fig. S9: Graphical representation of the two discussed matching paradigms. Since matching iterates over attributes independently, a latent can be selected by multiple attributes simultaneously.

## S2.2 Fully-Binary Matching Pursuit (FBMP)

The many-to-one matching procedure performed by FBMP is summarized in Fig. S10. It consists of a sequential selection procedure where one latent is selected at each step (based on some similarity criterion), then a residual is computed by removing the latent’s contribution. The following iteration then proceeds on the residual of the previous step. In Fig. S10, a $F _ { 0 . 5 }$ score is used as the selection criterion. Notice that it selects a diferent latent from the one selected by the F1 score in Fig. S8. This allows FBMP, in this particular case, to achieve a perfect reconstruction of the attribute annotations after the selection of a second latent in the coalition.

![](images/dc2da6b0ca2476756cd3ed6d22b38848ccf826c2329f248b15eb6cf4e35a4edd.jpg)  
Fig. S10: Many-to-one matching approach on a sequential residual update procedure.

Figure S11a illustrates the efect of the parameter $\beta$ on the $F _ { \beta }$ -score used for the latent selection step. The matching F1-score averaged over all 312 CUB attributes is displayed for diferent maximum coalition sizes (k). Lower $\beta$ values privilege precision over recall of the selected latent compared to the attribute presence annotations. As a result, bigger coalitions are required to attain a given F1 matching score, but higher overall scores tend to be achieved for suficiently large coalition sizes. On the contrary, $\beta = 1$ leads to a very quick start (quite high matching after the first selection), but tends to stall earlier. As a good compromise between a higher final matching score and smaller coalition sizes, we select $\beta = 0 . 5$ as the default value for FBMP.

Due to its early-stopping criterion, the sequential procedure of FBMP can terminate earlier than the maximum number of iterations defined by the parameter k. In Figure S11b we show a histogram of the actual coalition sizes obtained over the 312 CUB attributes when running FBMP with a high value of $k = 2 0$ It shows that most of the attributes were matched to a small number of SAE latents, with 50% of them using up to 5 latents only. Very few attributes required a coalition of more than 10 latents. Indeed, for such high coalition sizes, interpretability starts to be compromised and we can even infer that such attributes have not been appropriately learned by the underlying SAE.

![](images/d7028318265c37d2f6e6fdc667020d2ecd64191e4d980c95cedf17a41de7c21c.jpg)  
(a) Comparison of β values

![](images/ffb683d400bd611a1745d6f35e0beacbd9f484d2f3edb8c8e54d32f59e1fce66.jpg)  
(b) Histogram of optimal coalition sizes  
Fig. S11: Fully-Binary Matching Pursuit results over a TopK SAE trained on CLIP embeddings of CUB data samples.

## S3 Synthetic Dataset Construction Details

This section provides full details of the synthetic dataset construction pipeline introduced in Section 3.3, covering the generation procedure, prompt design, and curation steps for both synCUB and synCOCO .

## S3.1 synCUB

synCUB is constructed by applying reference-guided attribute perturbations to images from a controlled subset of CUB-200-2011 [52]. We restrict ourselves to the 33-class subset used in SUB [1] and consider their curated set of 45 attribute concepts. CUB attributes are provided with per-instance confidence scores; for each part family (e.g., has leg color, or has bill shape), we select the three most frequent attributes that also exhibit the highest annotation confidence. For each selected target attribute, we identify reference images that guide the appearance of the target attribute during editing. Reference candidates are drawn from the three most attribute-frequent classes within the CUB subset that contain at least three high-confidence examples of the attribute.

Table S1: Statistics of the synthetic perturbation datasets used for TAPAS evaluation. The verification classifier flagged 54% of synCUB and 72% of synCOCO pairs for manual review (Sec. S3.4).

<table><tr><td>Dataset</td><td>Generated pairs</td><td>Attributes manipulated</td><td>Avg. pairs per attribute</td><td>Final curated pairs</td></tr><tr><td>COCO</td><td>9000</td><td>80</td><td>32.08</td><td>2534</td></tr><tr><td>CUB</td><td>3063</td><td>43</td><td>136.42</td><td>2933</td></tr></table>

Table S1 summarizes the resulting dataset statistics. In total, we generated 3063 edited image pairs across 43 manipulated attributes, of which 2933 pairs remained after the automatic and manual filtering stages. Figure S12 (top) shows the distribution of manipulated attributes. The distribution is relatively balanced across attributes, with an average of 136.42 pairs per attribute. While some attributes occur more frequently than others, the overall distribution remains well spread across the attribute space, indicating that the generation and filtering pipeline preserves a diverse set of semantic perturbations.

## S3.2 synCOCO

While synCUB evaluates attribute-level alignment in a controlled single-object setting, real-world scenes are considerably more complex. Editing in such complex scenes is considerably more dificult. In particular, adding objects would frequently lead to unrealistic scenes. For example, removing a table does not imply that a car can plausibly be inserted into the same scene, which would result in out-of-distribution images. We therefore restrict synCOCO to object removal operations. Even object removal remains challenging due to the compositional nature of COCO scenes. For each pair, we select the target object to remove by first choosing the object with the lowest instance count in the scene. Our experiments showed that removing multiple objects simultaneously leads to unstable edits and unreliable verification. If multiple objects have the same instance count, ties are broken by selecting the object with the largest area in the image, ensuring that the perturbation remains visually significant.

Table S1 summarizes the resulting dataset statistics. We generated 9000 candidate image pairs covering all 80 COCO object categories. After automatic filtering and manual verification, 2534 pairs remained in the final curated dataset, covering 79 of the 80 categories; all pairs of one category (mouse) were rejected during curation. Figure S12 (bottom) shows the resulting concept distribution. Compared to synCUB, the distribution is less balanced, with person remaining the most frequent category, as such objects are easier both for the editing

CUB Synthetic Dataset Attribute Distribution (sorted)

![](images/5820990467a3e3400584177ebf8fdb518c6a5d779ac7277f7570a9f15c782b96.jpg)

![](images/896caa8dc6a8dda864b2cc61945a5f09dd64338beabb293135103578f3e4309c.jpg)

![](images/2b5315ebe1f9c4829c6e7ff507cb45dabc763888a7fe1c44e6e9f7c9827bc9af.jpg)  
Fig. S12: Attribute histograms for the synthetic datasets. Top: synCUB, bottom: syn-COCO.

model to remove and for the verification classifier to detect reliably. During the manual review of classifier-flagged pairs we therefore prioritized rare object categories, which improves the balance of the final dataset compared to a purely classifier-based selection.

## S3.3 Prompts

All attribute perturbations are generated using structured natural language prompts designed to enforce a strict single factor intervention constraint. The prompts are instantiated programmatically to ensure consistency across attributes and object classes. For synCUB, we use reference-based editing. Given a base image and a reference image representing the target attribute state, the edited image is generated using Flux2 [26], which allows conditioning on multiple input images. The base image serves as the identity reference, preserving pose, background, and all other attributes, while the attribute-reference image guides the appearance of the target attribute to be added. The prompt explicitly instructs

has primary color: blue has primary color: brown

![](images/5d85fc6ebca20a4de866883814360872c054b81b708a8696d23bacf2323bd6de.jpg)

![](images/b7d58804b192c104b0895585af38879925513d443b29255c1d00b8fbd1bf09fe.jpg)  
has primary color: blue has primary color: green  
has primary color: brown

![](images/ad9e7adad129cdf8d33eaa60e0c87e40f1031f7e331fadb345def10ba46a9262.jpg)

![](images/d69a46650b424614de8d695e637211b532fb95746143a2e6e22708c64b299936.jpg)

![](images/7321a57d3f6b154b7bd53e2f67b7114f4cf2299f8483b2bbc045eca03daab231.jpg)

![](images/a84a3abd342c16f3906337c589faa7f36124c6ac7ad4533b036719f4da781b83.jpg)

![](images/16c035982fd4f91ac5b40415bc7a8cabc9b46c75eea8836153816f9dc3527196.jpg)

![](images/0b214ee83b25e8ca9f7e43cbc5cf0dd75b50a68f8ee0447dd0122eff866f5aeb.jpg)

![](images/ff5cf51e268fe523ca65e9e551f4fdc784e6944af106820bb33c6de67d594c4b.jpg)

![](images/4d68af28c7d207c4dc0625e6886f76f8b9105b9edd78f5ed616fea1c3495404a.jpg)

![](images/4ab0854bad25f9dc06e7d3aa9d8a4d4afcc021f147670597972813575bb3c2c0.jpg)

![](images/ab95028d49556c0147f5b780b3a5ba3cd2a3f1040e73f6d56e304dda58b7cb5d.jpg)

![](images/60db2e86ed4a459d7c32258b3acb3ad07ba996a0a5885987cdf43744d759fba8.jpg)

![](images/de52e29b3e345ad0d02a51d9c07ccfedc89ffa6e1ef13716436607ca3c4616c4.jpg)

![](images/617d0b33249ddc022bda97ee98e12f3970fe973c305c406b5a4ca4a4d08a76ea.jpg)  
Fig. S13: synCUB pairs: Additional example images.

![](images/370c36bd41c1ea1adec9e904d0bb39f3e54f411ac68a524c5782ba173e285e74.jpg)

![](images/03796bd70a3fc7ceacb21b2c86093fab96245367d228440d415204cdc5d39ccc.jpg)

![](images/51c0f174e182b38dc0690b61d7c86239d214308ec2bacd5c75827026d74eab29.jpg)

![](images/ae42b3f2a20f1b51781c26e579095931817626f01615589b54d8fec897f16369.jpg)

![](images/fe29fdfb373cb001115b7a08fcbc53f4ca033ad8dcfeeb663451fd5d38beae2e.jpg)

![](images/0a23e482a3137d5f74eb3d42f88978a93fed597c1f8e22efcb6be27729e586ff.jpg)  
Fig. S14: synCOCO pairs: Additional example images.

![](images/05204c8844c3b585c610ab0443c001b5ad2ac17a68d77863a68204d6b1d7e1b3.jpg)  
Fig. S15: Prompts for the image generation

the model to modify only the specified attribute while preserving identity, pose, and background. For synCOCO, no reference images are used. Since the edits consist solely of removing an object rather than inserting a new one, the model is conditioned only on the original image together with a prompt that instructs the model to remove the specified object while preserving the rest of the scene. The prompt templates below are used verbatim, with placeholders replaced by the corresponding attribute names, class names, and semantic families. The exact prompts are visualized in Fig. S15.

## S3.4 Dataset Curation

Classifier-based selection of samples To ensure label consistency in the generated synthetic datasets, we trained multi-label ResNet-50 classifiers separately on the original CUB (attribute annotations) and COCO datasets. Training was performed for 30 epochs on CUB and 10 epochs on COCO. The classifier was used as a filtering mechanism during dataset construction. For each original–synthetic image pair, we verified whether the classifier predictions were consistent with the expected attribute perturbations. If the classifier misclassified either the original or the synthetic image with respect to the manipulated label indices, the pair was manually inspected to ensure correctness of the perturbation and annotation.

Table S2 reports the final validation performance of the classifiers on original and synthetic data. For both datasets, performance on synthetic images is lower than on original images, reflecting the distribution shift introduced by controlled attribute manipulation. The gap is more pronounced for CUB in terms of macro

Table S2: Validation performance of the ResNet-50 classifiers used for sample filtering.

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Original</td><td colspan="2">Synthetic</td></tr><tr><td>F1 (mi)</td><td>F1 (ma)</td><td>F1 (mi)</td><td>F1 (ma)</td></tr><tr><td>Synthetic COCO</td><td>0.63</td><td>0.62</td><td>0.42</td><td>0.37</td></tr><tr><td>Synthetic CUB</td><td>0.58</td><td>0.42</td><td>0.42</td><td>0.28</td></tr></table>

![](images/bc3539bb4be91c12a2520f193674b7177893c9214484895c9ff10d419b9b51a4.jpg)  
(a) Synthetic CUB

![](images/84989c68a1af305e1e7f4368c5db836f4e97ebf9d65cb60fdf99ef32d0502809.jpg)  
(b) Synthetic COCO  
Fig. S16: Training curves of the ResNet-50 classifiers used for dataset filtering.  
F1, consistent with its fine-grained attribute structure. The training dynamics are shown in Fig. S16. The curves illustrate stable convergence on the original data, with decreasing training loss and increasing validation metrics.

Human validation of samples Image pairs flagged by the filtering classifier described in Section S3.4 were manually curated. Annotation was carried out by four of the authors; each image pair was assigned to one of the categories shown in Fig. S17. The categories describe whether the intended concept perturbation was successfully applied and whether additional unintended changes occurred. For both datasets, the first two categories correspond to valid edits and are retained, while the remaining categories correspond to generation failures and are removed from the dataset. For synCUB, images were annotated as either a correct concept perturbation or as containing additional unintended attribute changes. Edits that produced visual artifacts or otherwise invalid modifications were categorized as invalid edits and filtered out. Flagged synCUB pairs were reviewed in multiple rounds by diferent annotators, and the per-pair scores were averaged; 130 pairs were removed in this way. For synCOCO, images were annotated using the same criteria with one annotator per image pair, and with an additional category capturing incomplete removal of the target object. Here, we extended the manual validation to the entire dataset: all 2504 pairs that passed the classifier check were also manually reviewed, of which 903 (36.1%) were rejected, predominantly due to incomplete removal of the target object. In addition, 1488 classifier-flagged pairs were reviewed, prioritizing rare object categories, of which 933 (62.7%) were accepted. The final synCOCO dataset therefore contains exclusively human-verified pairs.

Validity of the classifier flag. To verify that the automatic flagging carries signal, we performed a blind re-curation experiment: for each dataset, 150 flagged and 150 unflagged pairs were sampled, mixed, and re-curated without knowledge of their flag status. Rejection rates were consistently higher among flagged pairs (synCUB: 20.7% vs. 10.7%, Fisher’s exact test $p = 0 . 0 2 5 ;$ synCOCO: 41.3% vs. 32.7%, $p = 0 . 1 5 )$ , confirming that the flag is informative. For synCOCO, any residual dependence on the classifier is eliminated by the subsequent manual validation of every retained pair; for synCUB, the rejection rate among unflagged pairs bounds the residual error rate at roughly 10%.

## S4 Sparse Autoencoder Training Configuration

We train Sparse Autoencoders (SAEs) on frozen feature representations extracted from pretrained vision backbones. Specifically, we use the normalized class-token representation from the final transformer layer. Experiments are conducted on features from CLIP-ViT-L/14 and DINOv2- ViT-S/14. Training data is obtained from the full training splits of the CUB and COCO datasets. For each configuration, SAEs are trained for 50 epochs with dictionary sizes {128, 256, 512, 1024, 2048, 4096}. We evaluate several SAE training variants: TopK, BatchTopK, GlobalBatchTop-

![](images/87b0735eac6e92dcebfbb10606cd8b0860105bbba830be22999d11238bc8b8b0.jpg)  
Fig. S17: Human validation categories used during dataset curation. Green categories denote edits that satisfy the concept perturbation constraint and are retained, while red categories correspond to generation failures that are filtered from the final dataset.

KMatryoshkaSAE (referred to as Matryoshka), and JumpReLU [40]. As reference baselines, we additionally report results for randomly activated dictionaries and untrained (frozen) autoencoders. For the TopK-based variants, the sparsity constraint is fixed to $K = 3 2$ active features per input. Training follows the implementation and hyperparameters of [16]. Optimization is performed using Adam with learning rate $5 \times 1 0 ^ { - 4 } , \epsilon = 6 . 2 5 \times 1 0 ^ { - 1 0 }$ , and $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 9 9 )$ Gradients are clipped during training and early stopping is applied based on the validation loss.

Training Losses. For the TopK and BatchTopK SAEs, the training objective consists of a reconstruction loss, an $\ell _ { 1 }$ sparsity penalty on the activations, and an auxiliary loss following [16]. Given input features x and reconstruction xˆ, the reconstruction loss and sparsity penalty are defined as:

$$
\mathcal {L} _ {\mathrm{rec}} = \| x - \hat {x} \| _ {2} ^ {2}, \qquad \mathcal {L} _ {\ell_ {1}} = \lambda \| a \| _ {1},\tag{10}
$$

where a denotes the top-k activations and $\lambda$ controls the sparsity strength. The overall training objective is

$$
\mathcal {L} = \mathcal {L} _ {\text { rec }} + \mathcal {L} _ {\ell_ {1}} + \mathcal {L} _ {\text { aux }}.\tag{11}
$$

The auxiliary loss $\mathcal { L } _ { \mathrm { a u x } }$ is computed using inactive (dead) latents as proposed by [16], encouraging them to model the residual reconstruction error.

For the Matryoshka SAE, multiple nested intermediate reconstructions are produced during decoding. Reconstruction losses are computed for each intermediate reconstruction and averaged to form the final reconstruction term. The resulting objective is: $\begin{array} { r c l } { \mathcal { L } } & { = } & { \overline { { \mathcal { L } } } _ { \mathrm { r e c } } + \mathcal { L } _ { \ell _ { 1 } } + } \end{array}$ $\mathcal { L } _ { \mathrm { a u x } } ,$ where ${ \overline { { \mathcal { L } } } } _ { \mathrm { r e c } }$ denotes the mean reconstruction loss across all intermediate reconstruction stages and the final output.

![](images/bfb0628f50acc26ce34e449cfd4faf69f16d6c087202fefd45fb4b062499c67c.jpg)  
Fig. S18: Test reconstruction loss of the TopK, BatchTopK, and Matryoshka SAE variants as a function of dictionary size. Results are shown for features extracted from CLIP-ViT-L/14 and DINOv2 on the CUB and COCO datasets.

For the JumpReLU SAE [40], sparsity is not enforced through a fixed activation budget but through a learned per-latent threshold: the JumpReLU activation zeroes all pre-activations below their threshold. The training objective combines the reconstruction loss with an $\ell _ { 0 }$ sparsity penalty on the activations,

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{rec}} + \lambda \| a \| _ {0},\tag{12}
$$

where the thresholds are optimized using straight-through estimators following [40]. We use a sparsity coeficient of $\lambda = 0 . 0 0 1$ and a kernel bandwidth of 0.001 for the rectangle pseudo-gradients of both the JumpReLU activation and the $\ell _ { 0 }$ step function.

Figure S18 reports the reconstruction loss on the test set as a function of dictionary size, illustrating the scaling behavior of the diferent SAE variants across datasets and backbone models. Figure S18 shows that reconstruction loss decreases consistently with increasing dictionary size on COCO for both CLIP-$\mathrm { V i T - L } / 1 4$ and DINOv2 features, indicating that larger dictionaries improve reconstruction capacity on this more diverse dataset. In contrast, on CUB the reconstruction loss for BatchTopK and Matryoshka increases as the dictionary size grows. This behavior suggests overfitting when the dictionary becomes too large relative to the complexity of the CUB dataset, where additional latent capacity does not translate into improved reconstruction performance.

## S5 Qualitative Matching Results

Each qualitative plot shows the four highest-activating samples for a matched latent–concept pair. More precisely, given a ground-truth concept, we first identify the corresponding latent or latent coalition using either F1 matching or FBMP. For one-to-one matching and FBMP with coalition size $k = 1$ , retrieval is based on the activation of the single matched latent. For FBMP coalitions with $k > 1$ , we rank samples by the maximum activation across all latents in the matched coalition, such that the displayed images correspond to those with the strongest evidence for the matched concept. The green box “GT Attribute Present” indicates that the image was labeled with the corresponding attribute.

COCO Dataset, TopK SAE, Concept: Giraffe Dictionary size = 256 1 Dictionary size = 2048  
![](images/76e2bf8e56c942ee1d6e8ae9591f2e8565537ed67f2d2b18b1117fc66fa7a772.jpg)  
Fig. S19: Qualitative latent–concept matching example for the COCO concept “Girafe”. Columns correspond to SAE dictionary sizes (d = 256 and d = 2048), while rows compare the matching procedures (F1 one-to-one matching and FBMP many-to-one matching).

COCO Dataset, TopK SAE, Concept: Horse  
![](images/b9c79238ee64f654db3230a9535402d137748cb560fde7f8574829f2b5d6ee95.jpg)  
Fig. S20: Qualitative latent–concept matching example for the COCO concept “Horse”. Columns correspond to SAE dictionary sizes (d = 256 and d = 2048), while rows compare the matching procedures (F1 one-to-one matching and FBMP many-to-one matching).

Figure S19 and Figure S20 show qualitative results on COCO concepts. For the concept “girafe”, all four matching configurations correctly identify the top four activating images, suggesting that this concept is robustly and consistently captured across dictionary sizes and matching procedures. For the concept “horse”, rows compare F1 one-to-one matching and FBMP many-to-one matching, while columns correspond to dictionary sizes $d = 2 5 6$ and $d = 2 0 4 8$ . For the larger dictionary size, both matching procedures retrieve very similar samples, with two failure cases: one image containing a bench and one containing a dog, suggesting that the corresponding latent is not fully disentangled and partially captures visually related but semantically distinct content. For the smaller dictionary size, the retrieved samples difer more strongly between F1 and FBMP: F1 retrieves three correct horse images alongside one dog image, whereas FBMP retrieves four correct horse images, demonstrating the benefit of coalition-based matching under limited dictionary capacity.

Figure S21 illustrates a more subtle failure mode for the concept “sports ball”. While two of the four retrieved images clearly contain a tennis ball and one carries the corresponding ground-truth label, the fourth image is neither labeled nor visibly contains the object. Notably, all four images are strongly tennisrelated, suggesting that the matched latent has conflated the “sports ball” concept with the broader “tennis” context. This type of semantic entanglement, where a latent captures a correlated concept rather than the target one, is precisely the kind of error that TAPAScore is designed to detect, as a truly selective latent should respond to the presence of the object itself rather than to its co-occurring context.

![](images/d34db6d59d1dc2946a90e3f18f9f6f09fef9dc958dedacc777c9af92b09c5793.jpg)  
Fig. S21: Qualitative matching results for the COCO concept sports ball using $F _ { 1 }$ matching for a TopK SAE with $d = 2 0 4 8$ . All four retrieved images are tennisrelated.

Figures S22 and S23 show qualitative matching results for four CUB concepts us-

ing the same configuration: a TopK SAE with $d = 2 5 6$ and FBMP $( k = 3 ,$ $F _ { 0 . 5 } )$ . Figure S22 illustrates two successful cases: for has bill shape: cone, all four highest-activating images are correctly retrieved and carry the ground-truth label, demonstrating that even fine-grained morphological concepts are reliably identified; a similar result holds for has upperparts color: brown, confirming that color concepts are also well captured by the learned latent coalitions.

![](images/be06cde7c63f4cb40f9454840bf4c9b1e2bf207622ef140a1bffa4475c92b2a3.jpg)  
Fig. S22: Qualitative matching results for two CUB concepts using a TopK SAE with d = 256 and FBMP $( k = 3 , \ F _ { 0 . 5 } )$ . All four highest-activating images are correctly retrieved and labeled for both has bill shape: cone (left) and has upperparts color: brown (right), demonstrating that FBMP can reliably identify both fine-grained morphological and color concepts.

![](images/1ecb0b128c08a0b21ae16c8ab44706b954f36fbb6bbdfb380ebeaa9f7659d6ea.jpg)  
Fig. S23: Qualitative matching results illustrating limitations caused by spatial specificity and annotation noise in CUB. For has upperparts color: orange (left), all retrieved birds display orange plumage, suggesting that the latents do capture the general color concept but not its precise spatial localization (upperparts vs. underparts). For has breast pattern: striped (right), the retrieved images are visually consistent with the striped pattern, even though the ground-truth annotations are not always correctly aligned with the visual evidence.

Figure S23 highlights two cases where retrieval quality is imperfect. For has upperparts color: orange, three of the four retrieved images display orange plumage but have negative ground-truth labels. Closer inspection reveals that the orange coloration in these images is located on the underparts rather than the upperparts, indicating that the matched latent captures the color concept without encoding its spatial localization. This suggests that the granularity of attribute definitions, distinguishing upperparts from underparts color, poses a challenge for the SAE, which may learn a more general color representation that does not respect part-level distinctions. The ambiguity inherent in such spatially specific annotations can lead to annotation inconsistencies that negatively impacts the matching quality. For has breast pattern: striped, retrieved images are visually correct, but one shows the striped patt ern only partially and another is clearly mislabeled, further illustrating how annotation noise can artificially de- clearly mislabeled, further illustrating how an grade matching scores even when the underl ying latent captures the intended concept. concept.

![](images/3161f9b9b08da814171398f6803675bf08c573d382e4539d9a103cf371b34f06.jpg)  
Fig. S24: Qualitative matching results for the CUB concept has underparts color: purple using a TopK SAE with d=256 and FBMP $( k = 3 , F _ { 0 . 5 } )$

Figure S24 illustrates a failure case for the rare concept has underparts color: purple, where none of the four retrieved images contain the intended attribute. Unlike the annotation noise observed in Figure S23, this failure is related to concept rarity: purple underparts occur in very few CUB samples, providing insuficient training signal for the SAE to learn a dedicated latent. This suggests that matching quality is not only sensitive to annotation noise but also to the frequency of concepts in the underlying data distribution.

## S6 Additional Results

## S6.1 Latent to Concept Matching

Fig. S25 $\left( t o p \ r o w \right)$ and Fig. S26 (top row) show the $F _ { 1 }$ ∆MATCHScore as a function of dictionary size for BatchTopK, Matryoshka, TopK, and JumpReLU SAEs trained on DINOv2 embeddings of CUB and COCO, respectively. Across all four SAE variants on CUB, FBMP consistently outperforms the one-to-one baseline $( F _ { 1 } , k { = } 1 )$ , reinforcing the finding that attributes are better captured by a coalition of complementary latents than by a single unit. The advantage of FBMP is even more pronounced than with CLIP, as one-to-one scores remain low across all dictionary sizes. BatchTopK peaks sharply at dictionary size 512 before declining at larger sizes. Matryoshka exhibits a peak at 256 followed by a sharp decline at 512, after which scores partially recover at larger dictionary sizes without reaching the earlier peak again. TopK exhibits a broadly monotonic increase up to dictionary size 2048, mirroring the pattern observed with CLIP. JumpReLU increases steadily with dictionary size and reaches the highest ∆MATCHScores of all variants at 4096; together with TopK it slightly exceeds the corresponding CLIP peaks, whereas BatchTopK and Matryoshka peak lower than with CLIP. On COCO, results largely mirror the trends observed in the results with CLIP: FBMP consistently outperforms one-to-one matching, and the relative ordering of matching criteria is preserved. Absolute score diferences between SAE variants are smaller, and the overall trends with dictionary size are consistent with the CLIP findings, indicating that matching behavior on COCO is robust to the choice of vision backbone. An exception is JumpReLU, which fails to learn meaningful matchings at the smallest dictionary sizes (∆MATCHScore close to zero at 128 and 256) and only approaches the other variants from dictionary size 512 onwards.

![](images/32d1105406f9446c2eae198d5cfcc95a32f9bd74817c3fa67c93bf084fb153ed.jpg)  
-- Probe upper bound (F1 k=1) FBMP F0.25 (k=3) FBMP F0.5 (k=3)  FBMP F1 (k=3)  F1 (k=1)  
Fig. S25: F<sub>1</sub> ∆MATCHScore on CUB (top row) and TAPAScore on synCUB (bottom row) as a function of dictionary size for SAEs trained on DINOV2 embeddings of CUB, across BatchTopK, Matryoshka, TopK and JumpReLU SAE variants (left to right) and diferent matching criteria. The gray dashed line marks the supervised linear-probe upper bound.

Matching Scores of Untrained SAE. Figure S27 validates the monotonic increase of matching scores with dictionary size for an untrained TopK SAE across all four dataset and backbone combinations (CUB CLIP, CUB DINOv2, COCO CLIP, COCO DINOv2). In all cases, the untrained SAE baseline grows steadily as dictionary size increases from 128 to 4096, confirming that larger dictionaries inflate matching scores irrespective of whether meaningful representations have been learned. This inflation arises because a larger pool of candidate latents increases the probability of spurious statistical alignment between random activations and ground-truth attribute annotations. The efect is consistent across both FBMP variants and one-to-one matching, and is observed on both datasets and both backbones. This motivates the use of ∆MATCHScore, which subtracts the untrained baseline to isolate genuine improvements in semantic alignment from dictionary-size-induced inflation.

![](images/749180dfa3e2bd62cd1f08134b8721b2879c5506bac0de10fcbd3cc14402d326.jpg)  
-- Probe upper bound (F1 k=1) FBMP F0.25 (k=3) FBMP F0.5 (k=3)  FBMP F1 (k=3)  F1 (k=1)

Fig. S26: $F _ { 1 }$ ∆MATCHScore on COCO (top row) and TAPAScore on synCOCO (bottom row) as a function of dictionary size for SAEs trained on DINOV2 embeddings of COCO, across BatchTopK, Matryoshka, TopK and JumpReLU SAE variants (left to right) and diferent matching criteria. The gray dashed line marks the supervised linear-probe upper bound.  
![](images/8446e526f50d4a485197a30f3a126550f96fab419ada495f77acb83a47b36aa8.jpg)

![](images/a441756b73764580e2990a743110d1b3b00a4daa82dfb686a30bedc3a73512b7.jpg)

![](images/2bbc5f4916f8bfaf79a52b4aa5449ba62e07c2b5555d5389d07b0ee700c7bc0a.jpg)

![](images/5349643bbfe40bef95bc6bc18cb88b2579ac35660caf7893c0559c38de0ec196.jpg)  
—- Probe baselineFBMP F0.25 (k=3) FBMP F0.5 (k=3)  FBMP F1 (k=3) F1 (k=1)

Fig. S27: Unnormalized $F _ { 1 }$ matching scores of an untrained TopK SAE as a function of dictionary size, shown for CUB CLIP, CUB DINOv2, COCO CLIP, and COCO DINOv2 (left to right).

Matching over coalition sizes (k). Figure S28 shows ∆MATCHScore as a function of coalition size k for BatchTopK, Matryoshka, TopK, and JumpReLU SAEs trained on CLIP embeddings, at dictionary sizes d=256 (solid) and d=2048 (dashed). Across both datasets and all SAE variants, FBMP matches or outperforms naive top-k matching with $F _ { 1 }$ at every coalition size k. While FBMP improves or remains stable as k increases, the naive $F _ { 1 }$ matching score degrades steadily, as additional latents are added without regard for their relevance. For example, for the Matryoshka SAE trained on COCO embeddings, the naive score drops to 0.08 (d=256) and 0.16 (d=2048) at k=10, while FBMP remains stable at 0.34 and 0.40, respectively. This confirms that the sequential residual selection of FBMP is essential for maintaining matching quality at larger coalition sizes. Based on these results, we further select k=3 as the default coalition size for FBMP, as scores plateau beyond this point.

## S6.2 Functional Validation of Concept Alignment

Fig. S25 (bottom row) and Fig. S26 (bottom row) report TAPAScore on synCUB and synCOCO for SAEs trained on DINOv2 embeddings. From Fig. S25 (bottom row) it can be seen that the TAPAScore results for synCUB difer markedly from the corresponding CLIP results (main paper). Whereas the CLIP results often showed a peak at intermediate dictionary sizes followed by a sharp decline, this degradation is considerably less pronounced for the DINOv2 results. Batch-TopK remains comparatively stable across all dictionary sizes. TopK peaks at dictionary sizes 256–512 and then decreases, qualitatively consistent with CLIP but with a shallower drop for the FBMP criteria, while its one-to-one criterion degrades sharply. JumpReLU similarly peaks at 256–512 before declining moderately. Matryoshka, however, exhibits a notably diferent pattern: TAPAScore continues to improve with dictionary size and achieves its highest values at the largest dictionary sizes, consistent with the recovery of its matching scores in the top row. The TAPAScore results for synCOCO with DINOv2 are more stable across dictionary sizes compared to their CLIP counterparts, with less pronounced divergence between matching and perturbation alignment at larger dictionary sizes. Unlike the CLIP results, where overcompleteness led to a clear TAPAScore decline for BatchTopK and TopK, DINOv2 results exhibit only a mild decline at the largest dictionary sizes, suggesting that the backbone choice influences the degree to which overcompleteness reduces causal alignment.

## S6.3 Binary vs. Magnitude-aware Matching

Binarizing the latent activations prior to matching discards their magnitudes, which raises the question of whether a magnitude-aware matcher could exploit this information to obtain better concept assignments. To investigate this, we replace the binary $F _ { \beta }$ selection criterion of FBMP with non-negative Orthogonal Matching Pursuit (NN-OMP) [6], a standard magnitude-aware greedy algorithm that assigns each attribute to a coalition of latents using inner-product correlations on the raw (non-binarized) activations. We evaluate NN-OMP at coalition sizes k=1 (one-to-one) and k=3 (matched to the default coalition size of FBMP), and compare it against FBMP F0.5 (k=3) across all dictionary sizes (128–4096), both datasets, and the BatchTopK, Matryoshka, TopK and JumpReLU variants. Figures S29 and S30 report the ∆MATCHScore (top rows) and the TAPAScore (bottom rows) for COCO/synCOCO and CUB/synCUB, respectively. On COCO, NN-OMP separates trained from untrained SAEs, as its ∆ MATCHScore (which subtracts the untrained baseline) is clearly positive across all dictionary sizes; it is therefore a legitimate matcher rather than a degenerate baseline, and at small to moderate dictionary sizes NN-OMP (k=3) is even competitive with FBMP, which overtakes it at the largest dictionaries. On CUB, however, the NN-OMP scores remain close to zero — the one-to-one variant even falls below the untrained baseline — so that FBMP attains a markedly higher ∆MATCHScore at every dictionary size. The advantage is unambiguous under the causal metric: on both synCOCO and synCUB, and across all four SAE variants, FBMP F0.5 lies above both NN-OMP curves at essentially every dictionary size, with the most consistent margins observed on synCOCO.

We attribute this behaviour to the binary structure of the matching task. The ground-truth annotations are binary, and SAE sparsity already acts as a hard concept selector, so that matching binary atoms under an $F _ { \beta }$ criterion is more discriminative than inner-product correlation over raw magnitudes. This design choice does, however, place a particular concept of interpretability at the center of the evaluation. A latent whose activation magnitude encodes several distinct concepts is polysemantic in the sense studied by [25], and a magnitude-aware matcher may reward such a latent for aligning with a target attribute. We instead treat magnitude-encoded multiplicity as a form of insuficient disentanglement: rather than crediting such latents, our binary evaluation penalises them, which we regard as the intended behaviour for a metric that aims to assess whether individual latents correspond to single, human-aligned concepts.

## S6.4 Efect of Activation Sparsity

Beyond dictionary size, we study the efect of the activation sparsity by training TopK SAEs with $K \in \{ 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ at a fixed dictionary size of 1024 on CLIP embeddings of both datasets (Fig. S31). On CUB, the ∆MATCHScore peaks at $K \in \{ 1 6 , 3 2 \}$ for all matching criteria and declines sharply for larger K. On COCO, matching scores remain stable for $K \leq 3 2$ before degrading in the same manner. The TAPAScore on synCOCO peaks at K=32 for all criteria. On synCUB, in contrast, TAPAScore is highest at small sparsity levels, peaking at $K { = } 1 6$ for FBMP $F _ { 0 . 2 5 }$ , with the criteria diverging and partially recovering at larger K, while the one-to-one criterion behaves erratically. Overall, moderate sparsity levels $( K \in \{ 1 6 , 3 2 \} )$ provide the best trade-of between statistical and causal alignment. This extends our dictionary-size findings to the sparsity dimension, excessive capacity, whether through overcomplete dictionaries or large activation budgets, degrades concept alignment. This supports the default choice of K=32 used throughout the paper.

## S6.5 Stability of Untouched Attributes (∆stay).

TAPAScore measures whether matched latents respond to the perturbed attribute, but a high score could in principle be inflated by leakage, i.e., latents drifting on attributes that were never touched by the intervention. To rule this out, we compute a complementary score, ∆stay, on the same synthetic pairs. For each pair $( { \bf x } , \hat { \bf x } )$ and each annotated attribute a that is not part of the intervention set, let $I _ { a }$ denote its matched latent set. We define

$$
\delta_ {\mathrm{stay}} ^ {a} = \max _ {i \in I _ {a}} \hat {\mathbf {z}} _ {\mathrm{bin}} ^ {i} - \max _ {i \in I _ {a}} \mathbf {z} _ {\mathrm{bin}} ^ {i},\tag{13}
$$

and report ∆stay as the mean of $| \delta _ { \mathrm { s t a y } } ^ { a } |$ over all untouched-attribute instances. Small values indicate that latents matched to unrelated attributes remain stable under the perturbation, whereas large values indicate leakage. We keep ∆stay separate from TAPAScore, rather than folding it into the score, so that the TAPAScore values reported throughout the paper remain directly comparable.

Table S3 reports ∆stay per SAE variant for FBMP $F _ { 0 . 5 }$ matching with $k { = } 3$ aggregated over dictionary sizes. We obtain 0.12 ± 0.01 on synCOCO and 0.21 ± 0.04 on synCUB, confirming low absolute drift on untouched attributes. The values are highly consistent across SAE variants, particularly on synCOCO, indicating that the high TAPAScores reported in the main paper cannot be explained by unselective latent drift.

Table S3: ∆stay (mean ± std over dictionary sizes) for FBMP $F _ { 0 . 5 }$ matching with k=3. Lower is better.

<table><tr><td>SAE Variant</td><td>synCUB</td><td>synCOCO</td></tr><tr><td>TopK</td><td> $0.18 \pm 0.05$ </td><td> $0.12 \pm 0.01$ </td></tr><tr><td>BatchTopK</td><td> $0.24 \pm 0.03$ </td><td> $0.12 \pm 0.01$ </td></tr><tr><td>Matryoshka</td><td> $0.24 \pm 0.02$ </td><td> $0.12 \pm 0.01$ </td></tr><tr><td>JumpReLU</td><td> $0.20 \pm 0.02$ </td><td> $0.11 \pm 0.01$ </td></tr><tr><td>All</td><td> $0.21 \pm 0.04$ </td><td> $0.12 \pm 0.01$ </td></tr></table>

![](images/2d412d51e892423ce18d5ba3f866533fd2f53deabd864f24644911a82c58508b.jpg)

![](images/bc1f8f0d58cff9deb56e5847d19c61f4ddffd71ea2787942369cd1d2c1b68fc3.jpg)

![](images/df60b3861c24e8e491bc0392eda976dbf67236e945e82c529ecbd2a98ce36b44.jpg)

![](images/45e62336346ee7ceda3bc14a25635535f57c1ab3a9860d0aab5522d401c0750b.jpg)

![](images/7a2b065dafb4a7b4af1c5e753e71de8f3cca75b2129e6b55dc596405f268a358.jpg)

![](images/e056170849646d1f84eced534bfbdaa37013e84bf809ec74c3823163c69966bb.jpg)

![](images/baee124cf286aea733a074ce2cb9255e843c480ce17ff02386a56afe2df183dc.jpg)  
F1 | d=256  FBMP F1 | d=256F1 | d=2048 FBMP F1 | d=2048

![](images/18319766f13a7b52a6f7ae7b34a2382a1745b47ad0dd81fea9e4852c11041cf4.jpg)  
Fig. S28: ∆MATCHScore as a function of coalition size k for all SAE families on CLIP, dict size $d \in \{ 2 5 6 , 2 0 4 8 \}$ . Top row: CUB. Bottom row: COCO.

![](images/f87e48c7552d723573cf7143467eeea79d4175f5a491f9a6c839bb771e63dbac.jpg)

![](images/fd818c9a1d9bec1b57c63e2cf52e13b243ead73b0a399bb1e04b016f99b05343.jpg)

![](images/6e608998405da1f1d36e0335ee3e65b69f288a0d738e3a5c37056e78d7b47c68.jpg)

![](images/782322b03efe97907a0f910f796215ede28563a2fc57d1dc35d9c682dc4bd495.jpg)

![](images/0c0ccf70063c7bcea9b919c66158da2ad0cc23f712c876c4caa8fe37438bf01e.jpg)

![](images/2d356ff04bef18bc616a5997f80037553bba6e1f1d3452066856924f8eafe83e.jpg)

![](images/e0cb7adfaf64bf70ee19e04985817377ae198e73afe1cd356166583c8b140d1e.jpg)

![](images/912d0b27d1225b1a8738606bcb12d12d274ca9fc188507b466f025d1815ce19d.jpg)  
NN-OMP (k=3)FBMP F0.5 (k=3)  NN-OMP (k=1)  
Fig. S29: $F _ { 1 }$ ∆MATCHScore on COCO (top row) and TAPAScore on synCOCO (bottom row) comparing NN-OMP and FBMP F0.5 matching criteria for SAEs trained on CLIP embeddings, across BatchTopK, Matryoshka, TopK and JumpReLU SAE variants $( l e f t$ to right ).

![](images/fdee140e1ba91b1a6dcf870eee00d4bf4eb98f53fe10fd6a2482765003509941.jpg)

![](images/a266b79f9e4572a5c20d6d680561b2460edf0ddd0f8997d48dca2172e96c7a9a.jpg)

![](images/7a5f2ea4a8b20e6a02227e460396fe469de03f1bf867ef9fd32a1d5ba6e1f8c8.jpg)

![](images/e0c59a3f715a45cae20756f264d76005d13303197cff5fad07ef6c18ef5a9767.jpg)

![](images/78b7c62f368b647da87b282ea3cf8e1bf0d32f0a255eea8d8c8d1e2c8cc13e56.jpg)

![](images/24c0c4640de4983003d6a87b2a8ec87c3e4b25a1a74f630ba4bd09399a163ef2.jpg)

![](images/37c090228afc92290cea9e2588c7d90a9dbb1813300ed7da33be6812942036e9.jpg)

![](images/96281ee1755db3dd14fa3439fb20b1627855483bc4e2b2d4030d4171ad907a94.jpg)  
NN-OMP (k=3)FBMP F0.5 (k=3) NN-OMP (k=1)  
Fig. S30: F ∆MATCHScore on CUB (top row) and TAPAScore on synCUB (bottom row) comparing NN-OMP and FBMP F0.5 matching criteria for SAEs trained on CLIP embeddings, across BatchTopK, Matryoshka, TopK and JumpReLU SAE variants (left to right ).

![](images/19a7625f7575812b25d18916d805c784a37b564390425cdf314986936c4fc42e.jpg)

![](images/ee12e85cf492e73a28c2b15e4fd9c2b4f1e9eaa2d0ae92d8aae6b78d92f16e7e.jpg)

![](images/bf209b7c2ac35f6fdf149e5d20803f265a0a8c8a0443d96dc3853a4b0b8d57bc.jpg)  
FBMP F0.25 (k=3)FBMP F0.5 (k=3)FBMP F1 (k=3)F1 (k=1)

![](images/06eff016a1191371bd6e43689917cb4b5e243a6df81fd8b8a41f87ecab72137e.jpg)  
Fig. S31: Efect of activation sparsity K for TopK SAEs (CLIP) at dictionary size d = 1024. From left to right: ∆MATCHScore and TAPAScore on CUB/synCUB, followed by ∆MATCHScore and TAPAScore on COCO/synCOCO.