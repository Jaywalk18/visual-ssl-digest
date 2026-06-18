# RAIGen: Rare Attribute Identification in Text-to-Image Generative Models

Silpa Vadakkeeveetil Sreelatha 1 Dan Wang 2 Serge Belongie 3 Muhammad Awais 1 Anjan Dutta 1

## Abstract

Text-to-image diffusion models achieve impressive generation quality but inherit and amplify training-data biases, skewing coverage of semantic attributes. Prior work addresses this in two ways. Closed-set approaches mitigate biases in predefined fairness categories (e.g., gender, race), assuming socially salient minority attributes are known a priori. Open-set approaches frame the task as bias identification, highlighting majority attributes that dominate outputs. Both overlook a complementary task: uncovering rare or minority features underrepresented in the data distribution (social, cultural, or stylistic) yet still encoded in model representations. We introduce RAIGen, the first framework, to our knowledge, for label-free rare-attribute discovery in diffusion models, requiring no predefined minority categories. RAIGen leverages Matryoshka Sparse Autoencoders and a novel minority metric combining neuron activation frequency with semantic distinctiveness to identify interpretable neurons whose top-activating images reveal underrepresented attributes. Experiments show RAIGen discovers attributes beyond fixed fairness categories in Stable Diffusion, scales to larger models such as SDXL, supports systematic auditing across architectures, and enables targeted amplification of rare attributes during generation. The project page is available at https://vssilpa.github. io/RAIGen\_webpage/.

## 1. Introduction

Text-to-image (T2I) diffusion models such as Stable Diffusion have revolutionized image generation by producing high-fidelity visuals from natural language prompts (Podell et al., 2024; Rombach et al., 2022). However, these models

1University of Surrey 2University of California, San Diego 3University of Copenhagen. Correspondence to: Silpa Vadakkeeveetil Sreelatha <s.vadakkeeveetilsreelatha@surrey.ac.uk>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

not only reflect biases from their training data (Luccioni et al., 2023; Perera & Patel, 2023), but can also amplify them during generation, reinforcing societal stereotypes and inequalities if left unaddressed (Seshadri et al., 2024). For instance, despite near-parity in the LAION-5B dataset for occupations like “teacher”, generated samples remain heavily gender-skewed (Friedrich et al., 2023). Such disparities reduce semantic coverage and raise concerns about fairness and real-world deployment.

Several approaches counteract biases in T2I generative models by rebalancing or diversifying their outputs (Chuang et al., 2023; Ni et al., 2024; Shen et al., 2024; Li et al., 2024). While effective for predefined categories like gender or race, they often overlook forms of underrepresentation, such as physical traits, cultural symbols, or stylistic variations, essential for semantic diversity and faithful generation. Open-set bias detection (D’Inca et al. \` , 2024) broadens auditing but mainly identifies the majority attributes, and the surfaced attributes are largely dictated by inductive biases of external world models. Suppressing such majority features does not amplify the underrepresented ones, overlooking the critical task of identifying minority attributes: semantic factors encoded in the model’s internal representations, but consistently underexpressed. We validate this empirically in Appendix G.1, showing that suppression of dominant attributes reduces their presence but reallocates probability mass unevenly across minority groups. This observation reinforces the need for systematic discovery of rare attributes as a prerequisite for comprehensive auditing and mitigation.

We introduce RAIGen, the first label-free framework for rare attribute identification in diffusion models where no predefined minority labels are required. Rather than identifying all possible underrepresented attributes, RAIGen targets those that are already encoded in the internal representations of the model but are systematically underexpressed during generation. These attributes are not hallucinated or externally defined, but emerge directly from the learned feature space of the model. By shifting the focus from majority identification to the structured discovery of these suppressed features, RAIGen enables more comprehensive auditing and understanding of representational gaps in generative models.

Identifying minority attributes requires access to the internal factors of variation learned by diffusion models. However, these representations are often entangled and uninterpretable. To address this, we require a mechanism that maps entangled internal representations into semantically interpretable features, a role effectively realized by Sparse Autoencoders (SAEs) (Kim et al., 2025). We specifically adopt Matryoshka Sparse Autoencoders (MSAEs), which have shown strong interpretability in vision-language models via hierarchical semantic decomposition (Pach et al., 2025; Zaigrajew et al., 2025). While finer, more overcomplete levels could in principle better isolate minority attributes hidden within broader concepts, in practice they often fragment a single concept into many localized part-features, inflating the search space and yielding brittle or spurious rare attributes that are not stable, human-interpretable factors. We thus focus on the coarsest MSAE level, whose features are typically semantically and spatially coherent, producing more reliable attribute hypotheses.

Given these coarse features, a natural heuristic for minority attribute discovery is the activation frequency of MSAE neurons. We validate this approach in a controlled toy setting with known rare factors, where the least frequently activated neurons consistently align with the injected minority attributes (Section 5.1), establishing frequency as a reliable proxy for rarity. However, in a real-world setting, low activation alone may correspond to non-semantic or noisy neurons. To address this, we incorporate semantic distinctiveness, measuring how far the neuron’s top-activating samples lie from the dataset’s average semantic representation. Our final minority score combines rarity and distinctiveness, prioritizing neurons that are both infrequent and semantically separated. MSAEs further support verification via top-activating samples and spatial heatmaps, enabling direct inspection of neuron-level attributes.

The key contributions of this work are as follows: ❶ To the best of our knowledge, we introduce the first framework for rare attribute identification in diffusion models, extending bias analysis from predefined fairness categories or majority-dominant features to the systematic identification of underrepresented attributes encoded in model representations. ❷ We propose a simple, yet effective, minority metric that combines neuron activation frequency with semantic distinctiveness, forming the basis of RAIGen. ❸ We show that RAIGen reveals attributes beyond fairness categories, enables auditing across multiple diffusion architectures (Stable Diffusion 1.5, 2, XL, FLUX.1-schnell), and supports amplification via lightweight prompt interventions.

## 2. Related Work

T2I generation has significantly advanced generative AI, enabling the creation of highly realistic images from textual prompts (Ho et al., 2020; Ramesh et al., 2022; Rombach et al., 2022), but they also inherit and amplify the biases in their training data (Cho et al., 2023; Luccioni et al., 2023).

Bias Mitigation in Diffusion Models: Several methods mitigate biases in diffusion models such as Stable Diffusion. (Chuang et al., 2023) learn projection matrices on text embeddings aligned with fairness attributes; (Friedrich et al., 2023) and (Parihar et al., 2024) use classifier-free guidance to steer generations without retraining; and (Li et al., 2024; Vadakkeeveetil Sreelatha et al., 2025) introduce learnable modules on bottleneck representations to enforce responsible concepts while preserving semantic alignment. All assume the target attributes are known a priori. In contrast, we seek to uncover minority attributes that are already encoded in the model but systematically underexpressed, beyond any predefined fairness category.

Unknown bias identification: Bias auditing in text-toimage models has shifted from mitigating predefined categories to identifying previously unknown biases. Open-set bias detection emphasizes uncovering such biases without relying on predefined labels. (D’Inca et al. \` , 2024) proposes a framework to automatically identify biases in generative models by leveraging large language models to suggest potential bias attributes, generating synthetic images, and applying visual question answering to rank the prevalence of these biases. However, such methods mainly surface majority attributes that dominate generations, revealing overrepresentation but not underrepresentation. We address this gap by shifting the focus from identifying dominant biases to discovering minority attributes that are suppressed.

Interpretability with Sparse Autoencoders: Sparse autoencoders (SAEs) have become useful tools for interpreting generative models by mapping intermediate activations to human-interpretable concepts, enabling concept steering, suppression, and unlearning (Kim et al., 2025; Surkov et al., 2025; Tinaz et al., 2025; Cywinski & Deja ´ , 2025). Recent work further extends SAEs to hierarchical representations, such as Matryoshka SAEs for coarse-to-fine interpretability in CLIP (Pach et al., 2025). In responsible generation, DiffLens (Shi et al., 2025) and SAeUron (Cywinski & Deja ´ , 2025) intervene on SAE features associated with predefined sensitive attributes or target concepts. In contrast, RAIGen tackles the upstream open-set problem of identifying which attributes are encoded but suppressed, without assuming categories in advance.

## 3. Preliminaries

Diffusion Models: Diffusion models (Sohl-Dickstein et al., 2015; Ho et al., 2020; Song & Ermon, 2019) synthesize data by learning to reverse a forward process that progressively adds Gaussian noise to a clean sample $\mathbf { x } _ { \mathrm { 0 } } \sim p _ { \mathrm { d a t a } }$ according to a variance schedule, yielding $\mathbf { x } _ { T } \sim \mathcal { N } ( \mathbf { 0 } , \mathbf { I } )$ as $t \to T$ . A neural network $\epsilon _ { \theta } ( \mathbf { x } _ { t } , t )$ learns to predict the added noise, defining a denoising transition at each step. At inference, generation starts from noise and recursively applies the learned reverse process to produce a clean sample. T2I models such as Stable Diffusion (Rombach et al., 2022) extend this framework by operating in a compressed latent space where they condition on text embeddings from a language encoder, aligning generated images with language.

Sparse Autoencoders (SAEs): Sparse autoencoders aim to decompose input representations r $\mathbf { \Psi } \in \mathbb { R } ^ { n }$ into a set of latent features $\mathbf { z } = \{ z _ { 1 } , \ldots , z _ { d } \} \in \mathbb { R } ^ { d }$ that are both overcomplete $( d \gg n )$ and sparse, thereby encouraging interpretability and disentanglement of concepts. The encoder-decoder architecture is defined as:

$$
\mathbf {z} = \operatorname{ReLU} \left(W _ {\text { enc }} (\mathbf {r} - \mathbf {b} _ {\text { pre }}) + \mathbf {b} _ {\text { enc }}\right), \quad \hat {\mathbf {r}} = W _ {\text { dec }} \mathbf {z} + \mathbf {b} _ {\text { pre }},
$$

where $W _ { \mathrm { e n c } } \in \mathbb { R } ^ { d \times n } , W _ { \mathrm { d e c } } \in \mathbb { R } ^ { n \times d }$ , and $\mathbf { b } _ { \mathrm { e n c } } \in \mathbb { R } ^ { d } , \mathbf { b } _ { \mathrm { p r e } } \in$ $\mathbb { R } ^ { n }$ are learnable parameters. The model is trained to minimize the reconstruction loss $\mathcal { L } _ { \mathrm { S A E } } = \| \mathbf { r } - \hat { \mathbf { r } } \| _ { 2 } ^ { 2 }$ , while enforcing sparsity on z. Sparsity is imposed either through $\ell _ { 1 }$ penalties on z (Bricken et al., 2023), which can cause activation shrinkage (Rajamanoharan et al., 2024), or by hard selection of the top-k coordinates per input (Gao et al., 2025), which enforces exact sparsity but fixes the number of active units. BatchTopK (Bussmann et al., 2025) modifies this by flattening all activations in a batch into a single vector and retaining the largest $k \times B$ entries (for batch size $B ) _ { \ l }$ , allowing the number of active features to vary across samples while maintaining a global sparsity constraint.

Matryoshka Sparse Autoencoders (MSAEs): MSAEs extend SAEs by training under multiple sparsity constraints at once, following the idea of Matryoshka representation learning (Kusupati et al., 2022). Instead of selecting a single sparsity level $k ,$ the model applies a family of Top-k operators with increasing levels $\{ k _ { 1 } , k _ { 2 } , \ldots , k _ { f } \}$ , where $k _ { 1 } < k _ { 2 } < \cdot \cdot \cdot < k _ { f } = d ,$ forming a nested budget: $k _ { 1 }$ active neurons at the first level, then $\left( k _ { 2 } - k _ { 1 } \right)$ more at the next, and so forth, up to d. For an input r, the encoder produces multiple sparse codes and reconstructions for each level as follows:

$$
\mathbf {z} ^ {(k _ {i})} = \operatorname{ReLU} \left(\operatorname{Top} _ {k _ {i}} \left(W _ {\text { enc }} (\mathbf {r} - \mathbf {b} _ {\text { pre }}) + \mathbf {b} _ {\text { enc }}\right)\right),
$$

$$
\hat {\mathbf {r}} ^ {(k _ {i})} = W _ {\mathrm{dec}} \mathbf {z} ^ {(k _ {i})} + \mathbf {b} _ {\mathrm{pre}}.
$$

The training objective aggregates reconstruction losses across all levels,

$$
\mathcal {L} _ {\mathrm{MSAE}} = \sum_ {i = 1} ^ {f} \alpha_ {i} \| \mathbf {r} - \hat {\mathbf {r}} ^ {(k _ {i})} \| _ {2} ^ {2}, \tag {1}
$$

with coefficients $\alpha _ { i }$ weighting the contribution of each sparsity level. At inference, any $k _ { i }$ can be probed to reveal features at varying granularities. This design produces a hierarchical representation, with coarse levels capturing broad semantics and finer levels encoding detailed attributes.

## 4. Methodology

We propose RAIGen, a framework for rare attribute identification in text-to-image diffusion models. An overview of the framework is illustrated in Figure 1.

## 4.1. Problem Formulation

Let $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } , . . . , a _ { m } \}$ denote the set of semantic attributes that may be expressed in the outputs of a conditional generative model, where $m \geq 2$ . For instance, A could include {male, female, urban background, rural background, dark skin tone, $\cdots \}$ . A conditional generative model is defined as $G : ( \pmb { \xi } , \mathbf c ) \mapsto \mathbf x$ , where ${ \pmb \xi } \sim \mathcal { N } ( { \bf 0 } , { \bf I } )$ is a latent variable, $\mathbf { c } \in { \mathcal { C } }$ is an external condition (for example, a text prompt), and x is the generated output. The model induces a conditional probability distribution over attribute values:

$$
P _ {G} (a _ {i} \mid \mathbf {c}) = \operatorname * {P r} [ A (\mathbf {x}) = a _ {i} \mid \mathbf {c} ], \quad i = 1, \ldots , m.
$$

where $A ( \mathbf { x } )$ denotes the attribute value associated with the generated sample x.

Definition 1 (Generative Bias). A model G exhibits generative bias (Ferrara, 2024; Huang & Huang, 2025) with respect to attribute set A under condition c if there exist $i \neq j$ such that

$$
P _ {G} (a _ {i} \mid \mathbf {c}) \neq P _ {G} (a _ {j} \mid \mathbf {c}).
$$

That is, the model assigns uneven probabilities to attribute values under identical conditions.

Definition 2 (Minority Attribute). For a tolerance parameter $\epsilon > 0 .$ , an attribute $a _ { j } \in { \mathcal { A } }$ is a minority attribute under condition c if

$$
0 <   P _ {G} (a _ {j} \mid \mathbf {c}) \leq \min _ {a _ {i} \in \mathcal {A} \backslash \{a _ {j} \}} P _ {G} (a _ {i} \mid \mathbf {c}) + \epsilon
$$

This definition implies two conditions: (1) attributes with $P _ { G } ( a _ { j } \mid \mathbf { c } ) = 0$ are excluded, as they are not represented in the model’s latent space; (2) minority attributes are defined with respect to the model’s generative distribution rather than the raw training data, identifying features that are internally encoded but suppressed in outputs.

Objective: The discovery of minority attributes requires a mechanism that meets the following criteria: (1) It exposes the set of internal semantic concepts encoded by the generative model G. Formally, let h denote the representations extracted from G. We aim to utilize a feature decomposition operator M that maps the representations into a set of sparse latent features $M ( \mathbf { h } ) = \mathbf { z } = \{ z _ { 1 } , \dots , z _ { d } \} \in \mathbb { R } ^ { d }$ . Throughout, we use the term neuron to refer to individual latent feature $z _ { i }$ in the sparse representation $\mathbf { z } ,$ each of which may capture a distinct semantic feature. (2) It assigns a quantitative score reflecting the degree of underrepresentation of each feature. We define a scoring function s that assigns each latent feature $z _ { i }$ a value $s ( z _ { i } ) \in [ 0 , 1 ]$ , which yields the score vector $s ( \mathbf { z } ) = ( s ( z _ { 1 } ) , \ldots , s ( z _ { d } ) ) \in [ 0 , 1 ] ^ { d }$ . Each $s ( z _ { i } )$ captures both the rarity of feature $z _ { i }$ under condition c and its semantic distinctiveness relative to other features in z.

![](images/991d1e524f7c6fc4f0b4f149226f96d5caf5f05dcbf51aae21cffe4f3ad1deba.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
  A["ξ ~ (0,1)"] --> B["Diff UNet Encoder"]
  B --> C["h"]
  C --> D["Diff UNet Decoder"]
  D --> E["MSAE Decoder"]
  E --> F["z"]
  F --> G["MSAE Decoder"]
  G --> H["ĥ"]
  H --> I["Male doctor in black and white photography s = 0.615"]
  I --> J["Doctor wearing suit s = 0.280"]
  K["Female doctor s = 0.219"] --> L["coarsest level of MSAE neurons"]
  L --> M["color ≡ neuron identity size ∝ activation frequency"]
  M --> N["image Representations"]
    O["Coarsest level of MSAE neurons"] -.-> P["k₁, k₂, ..., k_f"]
  P --> Q["MSAE Decoder"]
  Q --> R["x"]
  R --> S["Doctor wearing suit s = 0.280"]
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
    style F fill:#f9f,stroke:#333
    style G fill:#f9f,stroke:#333
    style H fill:#f9f,stroke:#333
    style I fill:#f9f,stroke:#333
    style J fill:#f9f,stroke:#333
    style K fill:#f9f,stroke:#333
    style L fill:#f9f,stroke:#333
    style M fill:#f9f,stroke:#333
    style N fill:#f9f,stroke:#333
    style O fill:#f9f,stroke:#333
```
</details>

Figure 1. Overview of RAIGen. Diffusion representations (h) are decomposed by MSAE into interpretable features (z). A minority score (s), combining rarity and distinctiveness, ranks neurons or features to reveal minority attributes. Minority concepts are identified at the coarsest MSAE level (e.g., female doctor, doctor in suit), where size reflects activation frequency (smaller size = less frequent) and color denotes neuron identity.

This motivates RAIGen, which employs MSAEs to decompose the representations h into semantically meaningful features z, and introduces a novel minority score $s ( \mathbf { z } )$ that integrates feature rarity with semantic distinctiveness. Together, these components enable the label-free discovery of minority attributes directly from the internal representations of diffusion models.

## 4.2. Rare Attribute Identification

## Feature Decomposition from Diffusion Representations:

To expose the internal concepts encoded by diffusion models, we train MSAE on intermediate representations extracted during reverse sampling. Given a T2I diffusion model $G$ and a prompt c, we extract bottleneck representations $\mathbf { h } _ { t } \in \mathbb { R } ^ { \bar { h } \times w \times n }$ at each denoising step t. These representations are inherently interpretable (Kwon et al., 2023), and (Kim et al., 2025) have shown that SAEs trained on them reveal high-level features. Following (Cywinski ´ & Deja, 2025), we treat each spatial location in ht as an n-dimensional training example, disregarding spatial coordinates. These vectors are then used to train a MSAE using Equation (1), yielding a hierarchy of sparse codes $\mathbf { z } ^ { ( k _ { i } ) }$ that capture semantic structure at varying levels of granularity from broad concepts at coarse levels $( k _ { 1 } )$ to finer details at deeper levels $( k _ { f } )$ .

For minority attribute discovery, we perform inference with MSAE by collecting representation-image pairs $\mathcal { D } _ { c } =$ $\{ ( \mathbf { h } _ { t } ^ { ( j ) } , \mathbf { x } ^ { ( j ) } ) \} _ { j = 1 } ^ { N }$ , where $\mathbf { h } _ { t } ^ { ( j ) } \in \mathbb { R } ^ { h \times w \times n }$ denotes bottleneck representation at a fixed denoising step $t ,$ and $\mathbf { x } ^ { ( j ) }$ is the corresponding generated image for a prompt c. In practice, we use the final timestep, where semantic information is most fully expressed. For simplicity, we omit both the sample index j and the timestep index $t ,$ and write $( \mathbf { h } , \mathbf { x } ) \in \mathcal { D } _ { c }$ for an arbitrary pair. Each tensor h is flattened into $h \times w$ feature vectors, which are individually passed through the MSAE encoder following the training setup. For each MSAE neuron $z _ { i } ,$ , with $i = 1 , \ldots , d$ corresponding to a sparse latent feature, we define its activation on h as $z _ { i } ( \mathbf { h } )$ , obtained by averaging activations across spatial positions. These per-neuron activations form the basis for computing the minority score.

Minority Score: To quantify the degree to which each neuron encodes a minority attribute, we introduce the Minority Score, which balances two complementary signals: rarity of activation and semantic distinctiveness. Let $( \mathbf { h } , \mathbf { x } ) \in \mathcal { D } _ { c }$ be a diffusion representation-image pair, and $z _ { i } ( \mathbf { h } )$ the activation of MSAE neuron $z _ { i }$ as previously defined. We define the activation frequency as the proportion of samples where the neuron is active (i.e., has nonzero activation):

$$
\nu_ {i} = \frac {\left| \left\{(\mathbf {h} , \mathbf {x}) \in \mathcal {D} _ {c} : z _ {i} (\mathbf {h}) > 0 \right\} \right|}{\left| \mathcal {D} _ {c} \right|}. \tag {2}
$$

This metric directly measures how often the neuron participates across the dataset $\mathcal { D } _ { c } ,$ with rarer features corresponding to lower $\nu _ { \mathrm { i } }$ . We empirically validate activation frequency as a rarity signal in a controlled toy experiment (Section 5.1), and find that the least frequently activated neurons are more likely to correspond to rare features. In a real-world setting, however, low $\nu _ { i }$ can be noisy and may reflect uninterpretable activations, so we complement it with semantic distinctiveness.

We evaluate the semantic distinctiveness of each neuron by comparing its activation-weighted CLIP centroid to the global dataset centroid. Let $\mathrm { C L I P } ( \mathbf { x } )$ denote the CLIP embedding of image x. The centroid $\mu _ { i }$ for neuron $z _ { i } ,$ and the global centroid $\mu _ {  { \mathscr { D } } _ { c } }$ , are computed as:

$$
\mu_ {i} = \frac {\sum_ {(\mathbf {h} , \mathbf {x}) \in \mathcal {D} _ {c}} z _ {i} (\mathbf {h}) \operatorname{CLIP} (\mathbf {x})}{\sum_ {(\mathbf {h} , \mathbf {x}) \in \mathcal {D} _ {c}} z _ {i} (\mathbf {h})}, \tag {3}
$$

$$
\mu_ {\mathcal {D} _ {c}} = \frac {1}{| \mathcal {D} _ {c} |} \sum_ {(\mathbf {h}, \mathbf {x}) \in \mathcal {D} _ {c}} \operatorname{CLIP} (\mathbf {x}).
$$

Semantic distinctiveness $d _ { i }$ is then defined as the cosine distance between the two centroids. This metric ensures that the neuron centroid $\mu _ { i }$ is dominated by its top-activating images, since activation strengths directly weight their contribution. In contrast, the global dataset centroid $\mu _ {  { \mathscr { D } } _ { c } }$ represents the average semantics of the entire set of images in $\mathcal { D } _ { c } .$ The resulting distance $d _ { i }$ thus measures how much the neuron’s semantics deviate from the dataset’s average semantic representation. Both $d _ { i }$ and $\nu _ { i }$ are min–max normalized to [0, 1] for comparability. Minority Score is then defined as:

$$
s (\mathbf {z}) = \mathbf {d} \odot (1 - \nu) \tag {4}
$$

where $\mathbf { d } = ( d _ { 1 } , \ldots , d _ { d } ) , \pmb { \nu } = ( \nu _ { 1 } , \ldots , \nu _ { d } )$ . This formulation assigns a high score to neurons that are both rarely active (low $\nu _ { i } )$ and semantically distinct (high di) relative to the dataset’s average semantics. Intuitively, neurons with larger $s ( z _ { i } )$ are more likely to encode minority attributes, since they capture concepts that occur infrequently yet deviate substantially from dominant patterns 1. We demonstrate in Appendix G.10 that combining activation frequency and semantic distinctiveness is necessary to reliably recover minority-associated neurons. Conversely, neurons with the lowest minority scores do not necessarily correspond to dominant attributes, since low values can also arise from noisy or undistinctive activations, as we explain in detail in Appendix G.3. Although the Minority Score can be computed across all MSAE neurons, we focus on the coarsest level $z ( k _ { 1 } )$ , which captures broad, interpretable semantics. By restricting analysis to the top-k1 codes, we prioritize high-level structure over low-level noise, leveraging the hierarchical design of MSAEs to expose global attributes more clearly than standard SAEs.

Minority concepts often appear redundantly across multiple neurons with similar activation patterns. To obtain a compact and diverse set, we use the neuron centroid $\mu _ { i } \left( \operatorname { E q . } 3 \right)$ as a representative of each neuron’s semantics. Redundancy is assessed via pairwise cosine distances between centroids, which quantify similarity between neurons. We then iterate over neurons in descending order of Minority Score, retaining the current neuron in the final set and removing all others within a small fixed distance. This threshold, treated as a hyperparameter, controls semantic redundancy. The resulting set therefore contains distinct, interpretable, minority neurons ranked by underrepresentation. For interpretability, we visualize top-activating images with MSAE heatmaps and provide human-readable annotations via MLLMs.

We refer to RAIGen as label-free rather than fully unsupervised. The discovery pipeline: MSAE training, the Minority Score, and neuron ranking, requires no predefined minority categories or attribute labels, and operates purely on the diffusion model’s internal representations. However, we do rely on a pretrained semantic prior, a CLIP-style image encoder used in Eq. 3 to define semantic distinctiveness, which provides semantic geometry but not attribute supervision.

![](images/30a9ad7b52c9b92eaa6946cd7f895cdd6171e1f7c8e55a5ab4cd77959a38fd83.jpg)

<details>
<summary>bar chart</summary>

| q-value | Least-frequent latents | Random latents |
| ------- | ---------------------- | -------------- |
| q=0.1   | 0.85                   | 0.10           |
| q=0.2   | 0.97                   | 0.20           |
| q=0.3   | 0.99                   | 0.30           |
</details>

Figure 2. Least-active latents preferentially map to rare ground-truth features in the toy setting. The blue bars report P(rare feature | latent ∈ least-active), i.e., the fraction of least-active latents whose matched feature is rare. The orange bars report the same probability for a random baseline, computed by sampling the same number of latents uniformly at random.

## 5. Experiments

We first validate activation frequency as a signal for rarity in a controlled toy experiment. We then qualitatively and quantitatively evaluate RAIGen-discovered minority attributes on Stable Diffusion v1.4 and SDXL, including a user study, and show that these attributes can be systematically amplified during generation. Appendix G.2 demonstrates that RAIGen enables auditing across diffusion architectures, and Appendix 5.5 shows that RAIGen extends to FLUX.1- schnell. Additional experiments are reported in Appendix G.

## 5.1. Activation Frequency as a Proxy for Rarity

In this section, we test the central heuristic underlying our minority metric: latents with low activation frequency correspond to rare underlying factors. To evaluate this claim in a setting with known ground truth, we replicate the hierarchical tree-structured toy data generation procedure of Bussmann et al. (2025), which produces a long-tailed distribution of feature frequencies (Figure 6). We then train a Matryoshka SAE under their toy configuration and align learned latents to ground-truth features via a one-to-one Hungarian assignment computed from activation-similarity statistics, following Bussmann et al. (2025). Using this alignment, we assess (i) the relationship between each latent’s firing rate and the empirical frequency of its matched ground-truth feature, and (ii) whether the least-active Matryoshka latents disproportionately map to the rarest groundtruth features. We repeat all experiments over 20 seeds.

![](images/567f7debdfc7520f350ca6dd7bb59e2e5f61a3eac707e2c074585451391e43b1.jpg)  
Figure 3. WinoBias qualitative examples on SDXL. Top-activating images and MSAE activation heatmaps for minority neurons discovered by RAIGen across three WinoBias profession prompts: Doctor (top), Sheriff (middle), and Writer (bottom). Labels above each group show generated language annotations for the corresponding neuron.

For a quantile level $q \in \{ 0 . 1 , 0 . 2 , 0 . 3 \}$ , we operationalize rarity as follows. After matching latents to ground-truth features, we define (i) least-active latents as the bottom-q fraction of matched latents ranked by firing rate, and (ii) rare features as the bottom-q fraction of matched groundtruth features ranked by their true activation frequency. Figure 2 reports the conditional probability that a latent drawn from the least-active set is matched to a rare feature, P(rare feature | least-active latent), and contrasts it with a random-latent baseline. For all three quantiles, the least-active set exhibits a substantially higher probability of mapping to rare features than random, indicating that low latent firing rate is informative of true feature rarity in this controlled setting. This is further supported by a strong monotonic association between latent firing rates and the frequencies of their matched ground-truth features, with mean Spearman correlation $\rho \approx 0 . 9 9 1$ across seeds. Taken together, these results suggest that activation frequency serves as a reliable proxy for rarity. We view this as a validation of the heuristic under near-ideal feature–latent alignment; the ablations in Appendix G.10 show why frequency alone is insufficient in practice.

## 5.2. Minority Attributes Discovery

Datasets: We perform minority attribute discovery on Wino-Bias (Zhao et al., 2018) and COCO prompts (Lin et al., 2015) to ground our analysis in datasets that capture complementary aspects of bias and diversity. WinoBias provides controlled, occupation-based prompts that are widely used in fairness evaluations, making it well-suited for testing whether RAIGen can surface socially salient minority attributes such as gender imbalances in profession-related generations. In contrast, COCO offers a broad and diverse distribution of everyday scenes, and contexts, allowing us to evaluate whether RAIGen can uncover underrepresented attributes beyond fairness categories.

Experimental setting: We run minority attribute discovery on Stable Diffusion v1.4 (SD v1.4) (Rombach et al., 2022) and SDXL (Podell et al., 2024), a substantially larger and higher-capacity text-to-image model. For each prompt in WinoBias and COCO, we generate images and extract the model’s bottleneck representations. We use these representations to train MSAE and then identify underrepresented attributes using the procedure described in Section 4. For interpretability, we annotate the resulting minority neurons with GPT-5.2. Additional experimental details are provided in Appendix E.

woman with afro-curly textured hair  
![](images/437b4f5df7f80c7d83770ba09501b99debd91ed6c149d9b5c5151c7ecbc46220.jpg)

<details>
<summary>text_image</summary>

front-facing train with large smoke plumes
side-view train with strong motion blur
</details>

woman holding a camera  
Figure 4. COCO qualitative examples on SDXL and SD v1.4. Top-activating images and MSAE activation heatmaps for minority neurons discovered by RAIGen for two COCO prompts: “A woman taking a picture of herself in front of a desktop” using SDXL (top row) and “A train going down a track at full speed” using SD v1.4 (bottom). Labels above each group indicate generated language annotation for the corresponding neuron.

We evaluate RAIGen using an Attribute Presence metric based on VQA-style attribute verification commonly used in text-to-image evaluation (Hu et al., 2023). For each discovered minority attribute, we convert its language annotation (Section 4) into an attribute query and apply it to every generated image. To mitigate evaluator-specific biases, we do not rely on a single vision-language model; instead, we query both Llama 4-Scout (AI, 2024) and Qwen3-VL-8B-Instruct (Bai et al., 2025) for each image–attribute pair and aggregate their binary outputs via majority vote. Attribute Presence is then defined as the fraction of images for which the ensemble predicts the attribute is present (lower values indicate stronger rarity). We compare RAIGen against Open-Bias (D’Inca et al. \` , 2024), which targets majority attributes; while prior open-set approaches reveal what models tend to overproduce, RAIGen complements them by uncovering attributes the model systematically suppresses. Additional details on evaluation is in Appendix F.

Quantitative results: Table 1 reports Attribute Presence for Stable Diffusion v1.4 and SDXL on WinoBias (Prof) and COCO, comparing minority attributes surfaced by RAIGen to majority attributes discovered by OpenBias (D’Inca et al. \` , 2024). Across both datasets, OpenBias attributes occur with high frequency, whereas RAIGen attributes appear substantially less often, confirming that RAIGen isolates features that are encoded but rarely expressed under standard sampling. Notably, RAIGen presence is slightly lower in SDXL than SD v1.4 on both WinoBias and COCO, suggesting that increased model capacity does not necessarily translate into higher expression of rare modes.

<table><tr><td>Model</td><td>Approach</td><td>WinoBias(↓)</td><td>COCO (↓)</td></tr><tr><td rowspan="2">SD v1.4</td><td>OpenBias</td><td>0.941</td><td>0.933</td></tr><tr><td>RAIGen</td><td>0.205</td><td>0.220</td></tr><tr><td rowspan="2">SDXL</td><td>OpenBias</td><td>0.941</td><td>0.933</td></tr><tr><td>RAIGen</td><td>0.194</td><td>0.199</td></tr></table>

Table 1. Attribute Presence for majority (OpenBias) and minority (RAIGen) attributes on WinoBias (Prof) and COCO. Lower values indicate stronger underrepresentation.

Qualitative results: To assess RAIGen qualitatively, we visualize top-activating images for discovered MSAE neurons with activation heatmaps, and the corresponding language annotations. Figure 3 shows WinoBias results on SDXL. RAIGen uncovers both socially salient minority attributes and non-fairness concepts: for example, it surfaces neurons corresponding to female doctor as well as presentation/context cues such as doctor in a framed portrait and doctor with a medical chart in the background.

Figure 4 presents COCO examples for two prompts with SDXL results in the first row and SD v1.4 results in the second row. Across both models, RAIGen surfaces coherent minority attributes that correspond to rarely expressed semantics, including appearance and object-centric cues (e.g., afro-curly textured hair, a woman explicitly holding a camera) as well as distinctive scene modes (e.g., front-facing trains with large smoke plumes or side-view trains with pronounced motion blur). Overall, these examples show that RAIGen uncovers underrepresented attributes beyond fairness categories, spanning fine-grained stylistic, contextual, and compositional modes; importantly, these rare modes are detectable in both SDXL and SD v1.4 generations. Additional qualitative examples appear in Appendix H.

![](images/5c65bacff0d241684ede01b4efb506e570ede33a6f30f2c48ff14dddd5d184e3.jpg)

<details>
<summary>natural_image</summary>

Grid of medical professionals in various settings including face recognition, brain scan, and diagnostic imaging (no visible text or symbols)
</details>

![](images/881ae03855ab45b8026fc41d9d4a298c57a16fe1572a17218cc70e747c725372.jpg)

<details>
<summary>natural_image</summary>

Grid of medical professionals in various settings including face masks, thermal or heatmap overlays, and outdoor scenes (no visible text or symbols)
</details>

Figure 5. RAIGen on FLUX.1-schnell (transformer-based diffusion). Qualitative examples of minority neurons discovered for the prompt “a photo of a doctor”. For each neuron, we show its top-activating generated images and the corresponding activation heatmaps.

## 5.3. User study

Table 2. Human-estimated presence of RAIGen minority attributes (expected count out of 10 images; lower is rarer).

<table><tr><td>Profession</td><td>Avg. Mean Presence (↓)</td><td>95% CI</td></tr><tr><td>Analyst</td><td>1.35</td><td>[1.03, 1.67]</td></tr><tr><td>CEO</td><td>0.70</td><td>[0.44, 0.96]</td></tr><tr><td>Doctor</td><td>1.18</td><td>[0.97, 1.39]</td></tr><tr><td>Salesperson</td><td>1.45</td><td>[0.99, 1.91]</td></tr><tr><td>Sheriff</td><td>2.64</td><td>[2.21, 3.07]</td></tr></table>

To test whether RAIGen’s minority attributes are systematically underexpressed in default generations, we ran a user study with 25 participants. We consider five WinoBias professions (ANALYST, CEO, DOCTOR, SALESPERSON, SHERIFF) and, for each profession, the top-6 minority attributes discovered by RAIGen. For each profession, participants viewed a participant-specific grid of 10 images randomly sampled from SD v1.4 and, for each attribute, answered: “How many of the images in this grid contain this attribute?” (integer response in [0, 10]). For each profession, we compute a participant-level presence score by averaging each participant’s counts across the six attributes. We report the profession-level mean of these scores, with 95% confidence intervals across participants. Table 2 summarizes the results. Although RAIGen identifies attributes from intermediate representations, this study deliberately evaluates them at the image level: the perceptual ground truth is whether the attribute is visibly expressed in the generated outputs. This design directly tests whether RAIGen surfaces attributes that are not only internally encoded, but also meaningfully rare in the model’s standard generations.

Across professions, participants report RAIGen attributes in fewer than 3 out of 10 images on average, providing direct human evidence that these attributes are rarely expressed under standard sampling despite being encoded in the model’s representation space. The scarcity is most pronounced for CEO, where participants rarely observe the discovered attributes. Even for the profession with the highest presence, SHERIFF, the attributes appear only in a small minority of images. Overall, these results show that RAIGen’s discovered attributes are not only interpretable, but also recognized by humans as minority attributes: they are perceptible when present, yet occur infrequently under standard sampling, making RAIGen a practical tool for auditing rare modes beyond the model’s dominant behaviors.

## 5.4. Amplification of discovered minority attributes during generation

Table 3. Amplification on WinoBias via RAIGen-guided prompt revision. The top table reports results for SD v1.4 and the bottom table reports results for SDXL.

<table><tr><td>Prompt used</td><td>NLL (↑)</td><td>Dev. ratio (↓)</td><td>Align. (↑)</td></tr><tr><td>Base prompt</td><td>1.917</td><td>0.50</td><td>20.30</td></tr><tr><td>Ours-Revised</td><td>1.935</td><td>0.22</td><td>19.80</td></tr></table>

<table><tr><td>Prompt used</td><td>NLL (↑)</td><td>Dev. ratio (↓)</td><td>Align. (↑)</td></tr><tr><td>Base prompts</td><td>1.812</td><td>0.49</td><td>27.26</td></tr><tr><td>Ours-Revised</td><td>1.852</td><td>0.23</td><td>26.89</td></tr></table>

We investigate whether minority attributes identified by RAIGen can be used to amplify underrepresented modes in the generative distribution. Using RAIGen’s language annotations, we perform lightweight prompt revision guided by Llama 4-Scout, which injects minority descriptors into the input text. While RAIGen is agnostic to the downstream mitigation strategy (Chuang et al., 2023; Friedrich et al., 2023), prompt revision provides a simple test of whether reintroducing minority concepts changes the model’s generations in the intended direction. We evaluate on COCO and WinoBias by uniformly sampling images from (i) the original prompt and (ii) the revised prompt. For each setting, we report: (i) negative log-likelihood scored under the base prompt, (ii) deviation from a uniform attribute distribution, and (iii) CLIP alignment to the original prompt (see Appendix F.2 for metric details).

Table 3 shows that RAIGen-guided prompt revision substantially reduces attribute-presence deviation for both SD v1.4 and SDXL, indicating that the revised prompts increase coverage of minority attributes and move generations closer to a balanced attribute distribution. RAIGen-guided generations have higher negative log-likelihood under the base prompt, indicating that the amplified samples occupy lowerdensity regions of the original distribution and a small drop in prompt alignment, suggesting that we can meaningfully surface rare modes while largely preserving the original prompt semantics. Additional results on COCO are provided in Appendix G.4. Overall, RAIGen provides a practical mechanism for amplifying underrepresented attributes beyond predefined fairness categories.

## 5.5. Applicability of RAIGen to Transformer-based Diffusion Models

We further evaluate RAIGen on a transformer-based diffusion backbone by applying it to FLUX.1-schnell, whose denoiser follows a DiT-style transformer architecture. For a fixed prompt c (e.g., “a photo of a doctor”), we run the model with 4 denoising steps, cache intermediate hidden states from transformer.transformer blocks.18 following Surkov et al. (2025), and train an MSAE on these activations. We then rank coarse-level features using the RAIGen minority score. On the prompt “a photo of a doc-$\operatorname { t o r } ^ { \prime \mathrm { : } }$ , the discovered attributes achieve Attribute Presence $= 0 . 1 1$ , indicating that RAIGen isolates concepts that are encoded in FLUX representations yet rarely expressed under default sampling. As shown in our qualitative results (Fig. 5), the top-ranked neurons correspond to coherent underrepresented modes visible in both top-activating images and attribution maps, including female doctors with curly/afro-textured hair and masked doctors.

Interestingly, we observe a higher fraction of high-scoring but weakly interpretable neurons in FLUX compared to U-Net–based diffusion models: several high-scoring candidates exhibit diffuse or non-localized heatmaps in the qualitative analysis, making semantic verification less reliable. We attribute this, in part, to architectural differences in representational structure. U-Nets expose an explicit spatial bottleneck whose channels are naturally aligned with localized image regions, making feature attribution and heatmap visualization comparatively well-behaved. In contrast, the chosen FLUX transformer hook point may encode features that are less spatially grounded than U-Net bottlenecks, especially under few-step sampling. This suggests that rare-attribute discovery in transformer diffusion may benefit from more systematic selection of hook points across blocks and submodules (e.g., attention vs. MLP streams), which we leave for future work. Overall, these results indicate that RAIGen generalizes beyond U-Net architectures and provides a viable approach for rare-attribute auditing in transformer-based diffusion models.

## 6. Conclusions

We propose RAIGen, a framework for minority attribute discovery in diffusion models that combines Matryoshka Sparse Autoencoders with a novel minority score to identify features encoded in latent representations but underrepresented during generation. Unlike prior work focused on fixed fairness categories or majority trends, RAIGen uncovers rare, semantically meaningful attributes directly from internal activations. Through quantitative and qualitative analyses, including a user study showing that RAIGen discovers user-aligned attributes that are underexpressed in generations, we demonstrate that RAIGen reveals fairness, stylistic, and cultural minorities across Stable Diffusion variants, and facilitates targeted amplification via prompt revision. By grounding discovery in model representations, RAIGen complements LLM-based bias tools and lays the foundation for hybrid auditing frameworks that bridge external priors with internal model dynamics.

## Impact Statement

This paper introduces RAIGen, a representation-based method for identifying and amplifying underrepresented attributes encoded in text-to-image diffusion models via interpretable neuron representations. The primary intended positive impact is to provide a general tool for uncovering suppressed or rare semantic factors in generative models, improving transparency about what models represent and what they tend to neglect.

RAIGen can only surface attributes that are already encoded in a model’s internal representations. As a result, socially salient minorities that are not learned by the model may not be discovered. This motivates combining representationgrounded discovery with complementary approaches that incorporate external semantic priors, such as LLM-based tools, to obtain a broader view of underrepresentation.

The same capabilities could be misused. Amplifying rare attributes may enable the deliberate production of sensitive, stereotyped, or culturally harmful imagery, or facilitate targeted generation of protected traits in contexts where this is undesirable. More generally, feature-level control could be used to steer models toward propaganda-style messaging or other harmful content. Overall, we view RAIGen as a representational analysis and control tool whose societal impact will depend on the safeguards, governance, and deployment practices surrounding generative models.

## Acknowledgements

Serge Belongie and Silpa Vadakkeeveetil Sreelatha are supported in part by the Pioneer Centre for AI, DNRF grant number P1. Silpa Vadakkeeveetil Sreelatha also thanks the ELLIS PhD Program for support and acknowledges travel support from the ELIAS Mobility Fund and the Turing Mobility Scheme (2024/25) from the UK. The authors acknowledge the use of resources provided by the Isambard-AI National AI Research Resource (AIRR) (McIntosh-Smith et al., 2024). Isambard-AI is operated by the University of Bristol and is funded by the UK Government’s Department for Science, Innovation and Technology (DSIT) via UK Research and Innovation; and the Science and Technology Facilities Council [ST/AIRR/I-A-I/1023]. We also thank Stella Frank for proofreading the paper and providing valuable suggestions.

## References

AI, M. Introducing llama 4: Advancing multimodal intelligence, 2024.  
AlDahoul, N., Rahwan, T., and Zaki, Y. AI-generated faces influence gender stereotypes and racial homogenization. Scientific Reports, 2025.

Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., and Zhu, K. Qwen3-vl technical report. arXiv, 2025.

Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Tamkin, A., Nguyen, K., McLean, B., Burke, J., Hume, T., Carter, S., Henighan, T., and Olah, C. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023.

Bussmann, B., Nabeshima, N., Karvonen, A., and Nanda, N. Learning multi-level features with matryoshka sparse autoencoders. In ICML, 2025.

Cho, J., Zala, A., and Bansal, M. Dall-eval: Probing the reasoning skills and social biases of text-to-image generation models. In ICCV, 2023.

Chuang, C.-Y., Jampani, V., Li, Y., Torralba, A., and Jegelka, S. Debiasing vision-language models via biased prompts. arXiv, 2023.

Cywinski, B. and Deja, K. SAeuron: Interpretable concept ´ unlearning in diffusion models with sparse autoencoders. In ICML, 2025.

D’Inca, M., Peruzzo, E., Mancini, M., Xu, D., Goel, V., Xu, \` X., Wang, Z., Shi, H., and Sebe, N. Openbias: Openset bias detection in text-to-image generative models. In CVPR, 2024.

Ferrara, E. Fairness and bias in artificial intelligence: A brief survey of sources, impacts, and mitigation strategies. Sci, 2024.

Friedrich, F., Brack, M., Struppek, L., Hintersdorf, D., Schramowski, P., Luccioni, S., and Kersting, K. Fair diffusion: Instructing text-to-image generation models on fairness. arXiv, 2023.

Gao, L., la Tour, T. D., Tillman, H., Goh, G., Troll, R., Radford, A., Sutskever, I., Leike, J., and Wu, J. Scaling and evaluating sparse autoencoders. In ICLR, 2025.

Ho, J., Jain, A., and Abbeel, P. Denoising diffusion probabilistic models. NeurIPS, 2020.

Hu, Y., Liu, B., Kasai, J., Wang, Y., Ostendorf, M., Krishna, R., and Smith, N. A. Tifa: Accurate and interpretable textto-image faithfulness evaluation with question answering. In ICCV, 2023.  
Huang, L. T.-L. and Huang, T.-R. Generative bias: widespread, unexpected, and uninterpretable biases in generative models and their implications. AI & Society, 2025.  
Kang, s., Han, W., Ju, D., and Hwang, S. J. Rare text semantics were always there in your diffusion transformer. In NeurIPS, 2025.  
Kappiyath, A., Chaudhuri, A., Jaiswal, A. K., Liu, Z., Li, Y., Zhu, X., and Yin, L. SEBRA : Debiasing through self-guided bias ranking. In ICLR, 2025.  
Kim, D., Thomas, X., and Ghadiyaram, D. Revelio: Interpreting and leveraging semantic information in diffusion models. In ICCV, 2025.  
Kusupati, A., Bhatt, G., Rege, A., Wallingford, M., Sinha, A., Ramanujan, V., Howard-Snyder, W., Chen, K., Kakade, S. M., Jain, P., et al. Matryoshka representation learning. In NeurIPS, 2022.  
Kwon, M., Jeong, J., and Uh, Y. Diffusion models already have a semantic latent space. In ICLR, 2023.  
Li, H., Shen, C., Torr, P., Tresp, V., and Gu, J. Selfdiscovering interpretable diffusion latent directions for responsible text-to-image generation. In CVPR, 2024.  
Li, Z., Evtimov, I., Gordo, A., Hazirbas, C., Hassner, T., Ferrer, C. C., Xu, C., and Ibrahim, M. A whac-a-mole dilemma: Shortcuts come in multiples where mitigating one amplifies others. In CVPR, 2023.  
Lin, T.-Y., Maire, M., Belongie, S., Bourdev, L., Girshick, R., Hays, J., Perona, P., Ramanan, D., Zitnick, C. L., and Dollar, P. Microsoft coco: Common objects in context. ´ arXiv, 2015.  
Luccioni, S., Akiki, C., Mitchell, M., and Jernite, Y. Stable bias: Evaluating societal representations in diffusion models. In NeurIPS, 2023.  
McIntosh-Smith, S., Alam, S. R., and Woods, C. Isambardai: a leadership class supercomputer optimised specifically for artificial intelligence, 2024. URL https: //arxiv.org/abs/2410.11199.  
Ni, M., Wu, C., Wang, X., Yin, S., Wang, L., Liu, Z., and Duan, N. Ores: Open-vocabulary responsible visual synthesis. AAAI, 2024.  
Pach, M., Karthik, S., Bouniot, Q., Belongie, S., and Akata, Z. Sparse autoencoders learn monosemantic features in vision-language models. In NeurIPS, 2025.  
Parihar, R., Bhat, A., Basu, A., Mallick, S., Kundu, J. N., and Babu, R. V. Balancing act: Distribution-guided debiasing in diffusion models. In CVPR, 2024.  
Park, D., Kim, S., Moon, T., Kim, M., Lee, K., and Cho, J. Rare-to-frequent: Unlocking compositional generation power of diffusion models on rare concepts with llm guidance. ICLR, 2025.  
Perera, M. V. and Patel, V. M. Analyzing bias in diffusionbased face generation models. In IJCB, 2023.  
Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Muller, J., Penna, J., and Rombach, R. SDXL: Im-¨ proving latent diffusion models for high-resolution image synthesis. In ICLR, 2024.  
Rajamanoharan, S., Conmy, A., Smith, L., Lieberum, T., Varma, V., Kramar, J., Shah, R., and Nanda, N. Improving ´ dictionary learning with gated sparse autoencoders. arXiv, 2024.  
Ramesh, A., Dhariwal, P., Nichol, A., Chu, C., and Chen, M. Hierarchical text-conditional image generation with clip latents. arXiv, 2022.  
Rombach, R., Blattmann, A., Lorenz, D., Esser, P., and Ommer, B. High-resolution image synthesis with latent diffusion models. In CVPR, 2022.  
Seshadri, P., Singh, S., and Elazar, Y. The bias amplification paradox in text-to-image generation. In ACL, 2024.  
Shen, X., Du, C., Pang, T., Lin, M., Wong, Y., and Kankanhalli, M. Finetuning text-to-image diffusion models for fairness. In ICLR, 2024.  
Shi, Y., Li, C., Wang, Y., Zhao, Y., Pang, A., Yang, S., Yu, J., and Ren, K. Dissecting and mitigating diffusion bias via mechanistic interpretability. In CVPR, 2025.  
Sohl-Dickstein, J., Weiss, E., Maheswaranathan, N., and Ganguli, S. Deep unsupervised learning using nonequilibrium thermodynamics. In ICML, 2015.  
Song, Y. and Ermon, S. Generative modeling by estimating gradients of the data distribution. NeurIPS, 2019.  
Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., and Poole, B. Score-based generative modeling through stochastic differential equations. In ICLR, 2021.  
Surkov, V., Wendler, C., Mari, A., Terekhov, M., Deschenaux, J., West, R., Gulcehre, C., and Bau, D. Onestep is enough: Sparse autoencoders for text-to-image diffusion models. In NeurIPS, 2025.  
Tinaz, B., Fabian, Z., and Soltanolkotabi, M. Emergence and evolution of interpretable concepts in diffusion models. In NeurIPS, 2025.  
Um, S., Lee, S., and Ye, J. C. Don’t play favorites: Minority guidance for diffusion models. In ICLR, 2024.  
Vadakkeeveetil Sreelatha, S., Nag, S., Awais, M., Belongie, S., and Dutta, A. Respodiff: Dual-module bottleneck transformation for responsible & faithful t2i generation. In NeurIPS, 2025.  
Zaigrajew, V., Baniecki, H., and Biecek, P. Interpreting CLIP with hierarchical sparse autoencoders. In ICML, 2025.  
Zhai, X., Mustafa, B., Kolesnikov, A., and Beyer, L. Sigmoid loss for language image pre-training. In ICCV, 2023.  
Zhao, J., Wang, T., Yatskar, M., Ordonez, V., and Chang, K.-W. Gender bias in coreference resolution: Evaluation and debiasing methods. In NACL, 2018.

## A. Appendix

In the primary text of our submission, we introduce RAIGen, a framework for uncovering minority attributes encoded in the internal representations of diffusion models. To preserve clarity and conciseness in the main paper, we provide an extensive appendix that complements the core manuscript. The appendix includes additional experiments, detailed implementation protocols, broader qualitative analyses, and deeper ablations that could not fit within the page limits. Together, these materials extend the discussion in the main text by offering a fuller view of our methodology, empirical validation, and implications.

## B. Discussion

Meaningful discoveries vs. generic long-tail behavior A natural question is whether RAIGen surfaces socially or semantically meaningful rare attributes, or merely identifies low-frequency visual patterns that reflect generic long-tailed generation. We argue that, in the generative setting, this distinction cannot be drawn a priori. Unlike discriminative long-tail recognition, where rare classes are identified by counting labels, generation has no predefined categories over which to measure imbalance: a “class” in the generative tail may be a viewpoint, a species, a hair texture, or a contextual cue, and is itself unknown before discovery. Several RAIGen findings are not predictable from any obvious semantic prior. For the prompt “a photo of an assistant” (Fig. 12, SDXL), RAIGen surfaces animals (cats, dogs) as a suppressed mode, a learned association no semantic prior would anticipate. For “a small yellow bird sitting on a tree branch” (Fig. 13), it reveals species collapse onto a narrow visual prototype. These discoveries would be difficult to anticipate through manual specification of fairness categories. Whether a given suppressed attribute warrants intervention is inherently application-dependent: suppressed species diversity matters for an educational platform but not for a recruitment tool; suppressed hair textures matter for stock photography but not for autonomous driving. RAIGen’s role is to surface the full inventory of suppressed modes, and the practitioner provides the domain-specific judgment over which entries matter.

Practical utility of RAIGen Beyond cross-model auditing (Sec. G.2 of Appendix) and amplification (Sec. 5.4), RAIGen serves a broader role as an upstream discovery layer for any pipeline that needs to know which attributes are encoded but suppressed in a target model. 1. Identifying which attributes require specification. Every unspecified dimension in a prompt is a dimension where bias can operate silently. This is the setting in which all open-set bias auditing operates (D’Inca\` et al., 2024): prompting “a female doctor” resolves gender, but skin tone, hair texture, clothing, and setting remain unspecified, and each subsequent specification opens new unspecified dimensions. The space of possible suppressions is too large for manual inspection. Without a systematic discovery tool, practitioners have no principled way to know which dimensions are biased and which are not. RAIGen surfaces these dimensions explicitly, and its discoveries persist for reuse in amplification, cross-model tracking, or targeted auditing of specific deployment prompts (e.g., “doctor” or “CEO” for recruitment imagery). 2. Targeted rare generation: RAIGen’s discoveries directly address a gap in existing rare-generation pipelines. Methods such as Minority Guidance (Um et al., 2024) amplify low-density regions but do not identify which attributes are rare or worth amplifying. We validate this experimentally in Appendix G.5: for “a photo of an assistant,” Minority Guidance amplifies “animals” due to statistical rarity (attribute presence 0.382 vs. 0.150 for the SDXL baseline), but this attribute is contextually irrelevant for a profession task. Methods such as Rare-to-Frequent (R2F) (Park et al., 2025) and ToRA (Kang et al., 2025) produce high-fidelity rare generations but require prior knowledge of which concepts to target as input. RAIGen complements both lines of work: rather than replacing them, it provides the upstream discovery step they assume, automating the manual specification of what to generate. This positions RAIGen as a diagnostic and auditing tool whose outputs can serve as inputs to dedicated generation methods.

## C. Future work

A promising future direction is the integration of LLM-based and representation-based approaches for minority attribute discovery. While RAIGen surfaces attributes that are internally encoded but suppressed during generation, LLM-based tools offer a complementary perspective by drawing on external world knowledge to hypothesize socially salient or semantically expected minorities.

Since no existing method explicitly targets minority attribute discovery via LLMs, we perform a preliminary study by adapting OpenBias, originally designed for majority (overrepresented) feature identification for this purpose. While not validated for minority detection, we repurpose its language-based pipeline to surface expected but underrepresented attributes (e.g., non-binary cashiers, Hispanic teachers) and compare them against those discovered by RAIGen (e.g., doctors in vintage attire, teachers outdoors). Adapting OpenBias for minority discovery yields attributes with near-zero presence in generated samples (0.021 for WinoBias, 0.058 for COCO), reflecting concepts that the model fails to encode altogether. This highlights the complementarity of the two approaches: LLM-based methods surface expectation-driven minorities aligned with human priors, while RAIGen reveals underexpressed concepts that are internally encoded. Together, they provide a broader view of underrepresentation, motivating future work on unified frameworks that combine external semantic priors with internal representation-grounded discovery.

Our current pipeline incurs additional compute because minority discovery is prompt-specific, requiring training a separate sparse autoencoder (SAE) for each auditing setup. Future work could reduce this overhead by amortizing representation learning across prompts, for example, training a shared SAE on a diverse prompt mixture and using lightweight promptconditioned latent selection.

## D. Limitations

While RAIGen provides a systematic framework for discovering minority attributes in diffusion models, it is not without limitations. First, our approach only captures attributes that are encoded in the model’s internal representations, meaning that socially salient fairness categories entirely absent from the representation space (e.g., non-binary identities, certain cultural groups) will not be surfaced. This limits the scope of fairness auditing to what the model has already learned, rather than what society may expect. Second, since RAIGen relies on sparse autoencoders for interpretability, the quality and granularity of discovered attributes depend heavily on what the autoencoder itself captures. Attributes may be fragmented, merged, or overlooked depending on the sparsity budget and training dynamics, and alternative architectures could yield different results. Third, even among the minority neurons that are identified, some may remain noisy or ambiguous despite our filtering and validation steps. In the absence of ground truth annotations for minority attributes, it is difficult to rigorously quantify coverage and precision of the discovered neurons. These limitations should be considered when applying RAIGen, particularly in fairness auditing and downstream interventions. Nonetheless, they do not diminish the standalone utility of the approach in surfacing underrepresented concepts, but rather highlight opportunities to further strengthen it when combined with complementary tools.

## E. Experimental details

This section provides a detailed account of the experimental setup, including datasets, training procedures, and hyperparameter choices, to ensure that our results can be reliably reproduced.

## E.1. Discovery of minority attributes

We utilize 10 WinoBias professions and 50 COCO prompts to investigate the effectiveness of RAIGen. For WinoBias, prompts are constructed in the form “A photo of a profession”, where the profession is drawn from the benchmark, following prior text-to-image generation frameworks. For COCO, we use the original prompt captions without modification. For each prompt in WinoBias and COCO, we generate 5000 images together with their bottleneck representations of size $1 2 8 0 \times 8 \times 8$ across all timesteps. Each spatial location at each timestep is treated as an independent sample, yielding a 1280-dimensional vector that serves as input to the MSAE. This setup is adopted by (Cywinski & Deja ´ , 2025). We train MSAE using a sparse latent feature space of size $1 2 8 0 \times 1 6$ , employing two sparsity levels: a coarse level with $k _ { 1 } = 2 0 4 8$ neurons, and the remaining neurons allocated to the fine level. We adopt the official implementation provided by (Bussmann et al., 2025) for training, with the following hyperparameters: 5 training epochs, an effective batch size of 4, 096, a learning $3 \times 1 0 ^ { - 4 }$ $\frac { 1 } { 3 2 }$

Following training, we generate 5, 000 images along with their bottleneck representations of size 1280×8×8 at timestep=49. The number of timesteps during sampling from diffusion models is set to 50. To compute semantic distinctiveness, we use the CLIP ViT-L/14 model to extract embeddings for the top-activating images of each neuron. For interpretability, we use GPT-5.2 to annotate the minority neurons. During the pruning, we fix a cosine distance threshold of 0.003. To further ensure semantic relevance, we restrict the set to neurons whose minority scores exceed the 90th percentile. Hyperparameters were selected through sweeps and guided by heuristics, and validated using interpretability checks.

We now provide the prompt that we used to annotate our neurons using GPT-5.2 :

You are a JSON-only generator. You are not allowed to explain anything, write markdown, or comment. Only return a single valid JSON object. You have to analyze a specific

```txt
neuron from a sparse autoencoder trained on profession-related generated images. Each neuron activates in response to specific, consistent visual features that appear across its top-activating images, as confirmed by corresponding heatmaps.

- You are provided with:
  - Top-activating images (first row): strongest activations for this neuron
  - Heatmaps (second row): regions most responsible for the neuron activation

- Your job is:
    1. To carefully observe the top-activating images and heatmaps, and identify visually consistent attributes that correlate with the neuron activation.
    2. To generate a modified version of the base prompt that includes these attributes naturally and precisely.
    3. To output a flat list of non-redundant keywords capturing only the consistent attributes.

- Strict requirements:
    - The identified attributes must be:
    - Clearly and consistently visible across the top-activating images
    - Highlighted or partially supported by the heatmap attention
    - Not already implied by the base prompt
    - Not a core object/tool expected for the profession
    {
    "neuron_id": "{neuron_id}",
    "input prompt": "{base_prompt}",
    "identified_attribute": "<short, precise description of all consistent visual attributes across the images>",
    "suggested_prompt": "<the base prompt modified to include these attributes naturally >",
    "keywords": ["<keyword_1>", "<keyword_2>", "..."]
    }
    }

### Example Outputs

Input Prompt: "a photo of a doctor"

Example 1:
    {
    "neuron_id": "2041",
    "input prompt": "a photo of a doctor",
    "identified_attribute": "female doctor",
    "suggested_prompt": "a photo of a female doctor",
    "keywords": ["female"]
    }
```

## F. Evaluation details

In this section, we provide additional evaluation details that we utilized to investigate the effectiveness of RAIGen in identifying underrepresented attributes.

## F.1. Discovery of Minority attributes

We evaluate the effectiveness of RAIGen at discovering minority attributes using Attribute Presence, which measures how often the discovered attributes are expressed in generated samples. We use the minority-attribute annotations produced by GPT-5.2 (Section 4 of the main paper) to form candidate attribute queries. For each generated image and candidate attribute, we perform VQA-style verification by providing the image and the attribute to a vision-language evaluator and asking: “Is the attribute present in the image?” We record the resulting binary prediction and aggregate over images to obtain an attribute-wise presence score, then aggregate across attributes for a given prompt, and finally average across prompts to report the overall Attribute Presence. Lower values correspond to attributes that are more underrepresented under default

![](images/7451e1005308fa6147f19f7e6c041927cff9cdafe927cceefe09e7a412e3a86d.jpg)

<details>
<summary>bar chart</summary>

| Feature index | Activation count (log scale) |
| ------------- | ---------------------------- |
| 0             | 43000                        |
| 8             | 43000                        |
| 4             | 43000                        |
| 14            | 15000                        |
| 18            | 15000                        |
| 13            | 15000                        |
| 19            | 15000                        |
| 17            | 15000                        |
| 16            | 15000                        |
| 12            | 15000                        |
| 15            | 15000                        |
| 3             | 8000                         |
| 2             | 8000                         |
| 9             | 8000                         |
| 10            | 8000                         |
| 1             | 8000                         |
| 5             | 8000                         |
| 7             | 8000                         |
| 6             | 8000                         |
| 11            | 8000                         |
</details>

Figure 6. Long-tailed feature frequencies in the toy setting. Activation counts for each ground-truth feature over N=300,000 samples, sorted in descending order.

sampling.

To reduce evaluator-specific biases, we use an ensemble of vision-language models rather than relying on a single VQA system. Concretely, we query both Llama 4-Scout and Qwen3-VL-8B-Instruct for each image–attribute pair and combine their outputs via majority vote.

## F.2. Amplification of Minority attributes

We evaluate the effectiveness of amplifying the minority attributes discovered by RAIGen during generation using three metrics: likelihood, discrepancy from a uniform distribution, and CLIP alignment. In the amplification setting, we generate images for each attribute under two conditions: (i) the base prompt, and (ii) a revised prompt augmented with the discovered minority attribute. For each image, one of these two conditions is randomly selected, ensuring a balanced mixture of base and revised generations across the evaluation set.

Negative log-likelihood. For each neuron identified as encoding a minority attribute, we select its top-10 activating images and compute their exact log-likelihoods conditioned on the prompt using the PF-ODE estimator of Song et al. (2021). The likelihoods are aggregated across images for each prompt. For each attribute, we compute the average log-likelihood across all generated images and then aggregate across all attributes under consideration. We report the negative log-likelihood in bits per dimension. Higher values indicate that the amplified samples remain in low-density regions of the distribution, consistent with their underrepresented nature.

Discrepancy. Discrepancy measures how evenly images are distributed between the base and revised prompts. Intuitively, if amplification is successful, the distribution of samples across attributes should approach uniformity. Formally, this metric is defined analogously to fairness discrepancy measures in prior works (Parihar et al., 2024), where smaller values indicate better balance between base and revised generations.

CLIP Alignment. Finally, we measure the semantic alignment between the generated images and their corresponding textual prompts using CLIP embeddings. Higher alignment scores indicate that amplification strengthens the consistency between the intended minority attribute and the visual output, without sacrificing prompt fidelity.

![](images/56267b61d32372b6f792e73f98d803500ed5528df5ea71cf483962f6b919a442.jpg)

<details>
<summary>bar chart</summary>

| Ethnicity | Fraction of generated images (%) in Doctor (%) |
| :--- | :--- |
| East Asian | 2 |
| Black | 4 |
| White | 96 |
</details>

![](images/e8d05b21e6aa1fabcafabc3de7854d2789de19eb0c155bc26889e30770a4cdc3.jpg)

<details>
<summary>bar chart</summary>

| Category | Default (%) | After suppression (%) |
| :--- | :--- | :--- |
| Group 1 | 8 | 20 |
| Group 2 | 6 | 60 |
| Group 3 | 87 | 20 |
</details>

Figure 7. Attribute prevalence under default sampling vs. after naive suppression of the dominant attribute (White) for SDXL generations for Doctor and Manager. Suppressing the majority attribute reduces White prevalence substantially, but the resulting increase is concentrated in some minorities (notably Black) rather than uniformly amplifying all minorities (e.g., East Asian).

For both WinoBias and COCO prompts, we evaluate amplification using the top-5 minority attributes per prompt, generating 100 images for each prompt under this setting.

## G. Additional experiments

In this section, we provide additional experiments that we performed to investigate the effectiveness of RAIGen in identifying underrepresented attributes. The experiments are performed on the representations extracted using SD v1.4.

## G.1. Majority-suppression does not uniformly amplify minority attributes

We illustrate a limitation of majority-attribute suppression using a controlled study on SDXL. For each occupation prompt (“a photo of a doctor” and “a photo of a manager”), we generate 500 images with default sampling and estimate the prevalence of demographic attributes (White, Black, East Asian) using an external attribute annotator. We then apply a naive prompt-based suppression intended to reduce the dominant attribute (White): concretely, we add the negative prompt ‘‘white-race, caucasian, european, blonde’’ and re-sample 500 images under the same settings, which reduces the frequency of White-presenting subjects.

The results are reported in Figure 7. As expected, default sampling is highly skewed toward White subjects (Doctor: 95.6%, Manager: 87.0%), consistent with prior observations (AlDahoul et al., 2025). After naive suppression of the dominant attribute, the prevalence of White subjects drops substantially, but the freed probability mass does not translate into a uniform increase across minority groups. Instead, it shifts disproportionately toward a subset of minorities (e.g., a sharp increase in Black subjects) while others remain comparatively underrepresented (e.g., only a modest increase in East Asian subjects). This pattern is the generative-model analogue of the Whac-A-Mole dilemma identified in discriminative settings by Li et al. (2023); Kappiyath et al. (2025): mitigating one dominant factor does not eliminate the broader imbalance, but redistributes it onto a different sub-attribute. Single-target suppression therefore does not reliably surface or amplify all minority attributes that may be encoded in the model, motivating the need for systematic minority-attribute discovery before any intervention is applied.

## G.2. Cross-Model Analysis of Minority Attributes

In this section, we use RAIGen as a systematic auditing framework to compare minority attribute representations across diffusion architectures. Focusing on the profession of Doctor, we apply RAIGen independently to SD v1.4, v2.1, and XL to identify minority neurons and annotate their semantics. We then collect the unique set of discovered attribute annotations across all models to form a unified attribute vocabulary. For each attribute in this set, we measure its presence in each model by generating 1000 samples from the corresponding model to evaluate attribute presence. For every candidate attribute in the unified vocabulary, the Llama 4-Scout model is provided with the generated image and asked whether the attribute is

![](images/da7ff17e1ec0ee11e252a9e903c93fcd441e04d992e6677b48625dcd94b37565.jpg)

<details>
<summary>bar chart</summary>

| Category   | Green Bar | Orange Bar |
| ---------- | --------- | ---------- |
| beard      | 180       | 180        |
| female     | 100       | 290        |
| black-race | 100       | 300        |
</details>

![](images/4a97c847370c6ea872f48c82b7d1cbb96073f415561b7af6429ccfbdbde217bd.jpg)

<details>
<summary>bar chart</summary>

| Attribute         | Value |
| ----------------- | ----- |
| suit              | 70    |
| outdoors          | 40    |
| grassy area       | 28    |
| vintage clothing  | 14    |
| sepia tone        | 10    |
</details>

![](images/af32f4905467a9834d2b2729d0989e68d7630522f8d03b18669ed16cfcc7b3a8.jpg)

<details>
<summary>bar chart</summary>

| Category           | Attribute Presence |
| ------------------ | ------------------ |
| clinical            | 80                 |
| bookshelves        | 320                |
| hospital corridor  | 90                 |
| surgical mask      | 420                |
| medical charts     | 150                |
| pharmacy setting   | 30                 |
| consulting notes    | 20                 |
| medical diagrams   | 10                 |
| anatomy posters   | 10                 |
</details>

![](images/f6114fc0a0968dae473276da0ba61e002a6fe3413e701c03eed8be49eb39dc14.jpg)

<details>
<summary>bar chart</summary>

| Category       | Green Bar | Orange Bar | Blue Bar |
| -------------- | --------- | ---------- | -------- |
| arms crossed   | 220       | 215        | 170      |
| sitting        | 25        | 90         | 190      |
</details>

SDv1.4 SDv2.1 SDXL

Figure 8. Category-level presence of minority attributes in images of doctors across SD versions. Demographic, stylistic, contextual, and gestural attributes reveal distinct representational shifts.

present using the prompt “Is the attribute present in the image?”, and we record whether the attribute is detected. We repeat this process across all attributes. This produces an occurrence count that allows us to compare how frequently each attribute manifests across models.

The results are summarized in Figure 8. Demographic attributes show the strongest change: SD v1.4 heavily underrepresents female and black doctors, partially corrected in SD v2.1, while SDXL introduces new imbalances, such as overemphasis on traits like beards. Stylistic minorities follow the reverse trend—attributes like sepia tone, vintage clothing, and outdoor settings appear in SD v1.4 but are nearly absent in SDXL, which favors standardized, formal portrayals. Contextual minorities shift most dramatically: diverse backdrops (e.g., bookshelves, hospital corridors) in early models are replaced by a dominant clinical office setting in SDXL, indicating contextual homogenization. In contrast, gestural attributes such as arms crossed or sitting remain stable across versions, suggesting pose features are robustly preserved. By surfacing minority attributes and enabling their tracking across model families, RAIGen reveals how architectural changes and scaling can shift underrepresentation rather than resolve it. Gains in demographic balance may trade off with stylistic or contextual diversity, highlighting the need for auditing of all forms of underrepresentation beyond narrow fairness categories.

## G.3. Why Low Minority Scores Do Not Capture Majority Features

While the Minority Score is well suited to isolate underrepresented features (low frequency, high distinctiveness), we investigate whether neurons with the lowest scores might instead recover dominant attributes. To this end, we utilize RAIGen to identify minority neurons from the intermediate representations of diffusion models for the prompt “A photo of a Doctor”. We then examine the bottom-ranked neurons from our trained MSAE at coarse sparsity levels, where semantic content is most fully expressed. We visualized their activations and corresponding heatmaps, using Figure 9 to illustrate representative cases.

Empirically, we observe that the dominant features appear duplicated, fragmented, or spread across many neurons whereas minority features emerge as sharp, semantically coherent units. This arises because the majority attributes correspond to broad, high-variance regions of the diffusion representation space, which are distributed across multiple correlated directions rather than concentrated along a single axis. As a result, the neurons with the lowest Minority Scores remain diffuse and unreliable for interpretation, often localizing to narrow, fine-grained patches in the heatmaps rather than capturing the attribute holistically. This variation persists even under strong sparsity, forcing the model to distribute reconstruction responsibility across several correlated neurons. By contrast, minority concepts occupy compact, lower-variance regions with little redundancy, enabling the MSAE to assign a single neuron to capture the full feature without incurring large reconstruction penalties. Consequently, low scores do not provide clean access to dominant ones and hence, the framework is expressly tailored for the discovery of minority attributes, and not for the recovery of interpretable majority features.

![](images/399dfde0427456586eec65092eec0966ff4e68d07295e074cf7dc4bb65c5ab0a.jpg)

<details>
<summary>natural_image</summary>

Group of medical professionals in white coats and blue scrubs with stethoscope overlays, showing various facial expressions (no text or symbols)
</details>

Minority score =

![](images/1686c4fc9f8fe253dcd085d1761048b842e6f2f4a3573c383acf6ac4378acab0.jpg)

<details>
<summary>natural_image</summary>

Group of medical professionals in different lab coats and caps, each with a stethoscope and a corresponding thermal heatmap overlay (no text or symbols)
</details>

Minority score =

![](images/29c09be02b484e53288ad4c4b159fca494c630c2239956de5a3579634510f75a.jpg)

<details>
<summary>text_image</summary>

A. DCICARR
D. TOAR
A. DCICARR
D. TOAR
</details>

Minority score=

![](images/6bf948a979a8adfe2b5149b53450adc6a6b7edfac0a918b76d940401182233af.jpg)

<details>
<summary>natural_image</summary>

Group of medical professionals in white coats and stethoscope, alongside a thermal heatmap image of the same individuals (no text or symbols visible)
</details>

Minority score=  
Figure 9. Low-scored (majority) neurons identified by RAIGen for the prompt “A photo of a doctor”. Unlike minority neurons, their activations and heatmaps are diffuse, fragmented, and fail to capture coherent semantic attributes.

Table 4. Amplification on COCO via RAIGen-guided prompt revision. The top table reports results for SD v1.4 and the bottom table reports results for SDXL.

<table><tr><td>Prompt used</td><td>NLL (↑)</td><td>Dev. ratio (↓)</td><td>Align. (↑)</td></tr><tr><td>Base prompt</td><td>1.943</td><td>0.53</td><td>26.93</td></tr><tr><td>Ours-revised</td><td>1.962</td><td>0.25</td><td>25.96</td></tr><tr><td colspan="4"></td></tr><tr><td>Prompt used</td><td>NLL (↑)</td><td>Dev. ratio (↓)</td><td>Align. (↑)</td></tr><tr><td>Base prompt</td><td>1.819</td><td>0.51</td><td>20.52</td></tr><tr><td>Ours-revised</td><td>1.864</td><td>0.28</td><td>19.83</td></tr></table>

## G.4. Additional results on amplification of minority attributes on COCO

To complement our evaluation in Section 5.4 of the main paper, we assess how the minority attributes discovered by RAIGen can be utilized to amplify their generation. We consider the top-5 minority attributes identified by RAIGen for 50 COCO prompts. For each attribute, we generate images with and without revised prompts uniformly and compute the likelihood under the base prompt distribution. We also compute attribute deviation from uniformity and CLIP alignment. As in the main paper, likelihood values are compared against a baseline of 500 randomly sampled images for each COCO prompt. The results are summarized in Table 4.

Table 4 shows that RAIGen-guided prompt revision substantially reduces attribute-presence deviation for both SD v1.4 and SDXL, indicating improved coverage of minority attributes and generations closer to a balanced attribute distribution. At the same time, prompt revision using RAIGen discovered atributes increases NLL when scored under the base prompt for both models, consistent with shifting samples toward lower-density regions of the original distribution (i.e., surfacing underrepresented modes). Prompt alignment decreases modestly, which is expected because alignment is measured against the original prompt while our revised prompts explicitly inject additional minority descriptors. As in the main paper, our goal is not to optimize the mitigation strategy but to demonstrate that RAIGen’s discovered attributes can be used to reliably amplify rare modes in an open-domain setting; more targeted interventions (e.g., controlled decoding or representation steering) could further preserve prompt fidelity while amplifying minority attributes.

![](images/c0ae885755de3e539796773cb3ad28695d531c3ff8ffac3a039764a3ce1ef457.jpg)  
Figure 10. Visualizations of top-activating samples and corresponding heatmaps for top minority neurons identified using a standard SAE for RAIGen. Each pair shows the top-activating images and the neuron’s activations. While the neurons capture meaningful localized features (e.g., facial details or specific contextual elements), they frequently fragment broader concepts across multiple neurons, illustrating reduced interpretability compared to MSAE.

## G.5. Comparison with statistical rarity sampling: Minority Guidance

Sampling-based methods such as Minority Guidance (Um et al., 2024) amplify generations toward low-density regions of the conditional distribution using statistical signals (e.g., classifier-free guidance scaled by an inverse-likelihood term). A natural question is whether such methods make a discovery framework like RAIGen redundant: if low-density samples are already accessible via sampling alone, why train an MSAE and rank neurons?

We argue that Minority Guidance and RAIGen solve different problems. Minority Guidance amplifies any attribute satisfying its statistical-rarity heuristic, regardless of whether that attribute is contextually meaningful or socially relevant. RAIGen instead identifies which attributes are encoded but suppressed, leaving the choice of which to amplify to the practitioner. We validate this empirically on the prompt “a photo of an assistant” (SDXL), for which RAIGen surfaces animals (cats, dogs) as a learned suppressed mode (see Fig. 12, “assistant”). We then generate samples from SDXL with Minority Guidance applied at the same prompt and compute Attribute Presence for “animals.”

Table 5. Attribute Presence for “animals” on the prompt “a photo of an assistant” (SDXL). Minority Guidance amplifies “animals” to 0.382, but “animals” is contextually irrelevant for a profession task. RAIGen’s role is to surface the attribute (so practitioners can decide whether to amplify it), not to amplify all statistically rare attributes indiscriminately.

<table><tr><td>Method</td><td>Attribute Presence (“animals”)</td></tr><tr><td>Base SDXL</td><td>0.150</td></tr><tr><td>Minority Guidance</td><td>0.382</td></tr></table>

As shown in Table 5, Minority Guidance more than doubles the presence of “animals” over the base SDXL distribution. Without prior knowledge that “animals” is the attribute being amplified, a practitioner using Minority Guidance to audit or diversify “assistant” generations would obtain outputs more frequently containing animals, an outcome that is statistically warranted but contextually problematic for an occupation-related task. RAIGen’s contribution here is upstream: it identifies that this association exists in the model, so a practitioner can decide whether to amplify or suppress it. The two approaches are therefore complementary rather than competing: RAIGen identifies what is suppressed, and sampling-based methods like Minority Guidance can then amplify attributes once they are chosen.

## G.6. Comparison with image-level clustering

To test whether neuron-level decomposition is necessary for rare-attribute discovery, we compare RAIGen against a simpler image-level baseline: k-means clustering on CLIP image embeddings of generated samples. For each prompt, we generate 5,000 images on SDXL, compute their CLIP-ViT-L/14 embeddings, and run k-means with k = 50. We treat the ten smallest clusters as minority-attribute candidates (the image-level analogue of selecting rarely activated neurons), annotate each cluster using the same protocol as for RAIGen neurons (Sec. 4), and compute Attribute Presence on the resulting annotations.

Table 6. Mean Attribute Presence (↓) of minority attributes discovered by CLIP k-means vs. RAIGen across six WinoBias professions on SDXL. RAIGen consistently isolates more strongly suppressed attributes.

<table><tr><td>Profession</td><td>CLIP k-means</td><td>RAIGen</td></tr><tr><td>Doctor</td><td>0.30</td><td>0.15</td></tr><tr><td>Sheriff</td><td>0.50</td><td>0.18</td></tr><tr><td>Farmer</td><td>0.69</td><td>0.25</td></tr><tr><td>Attendant</td><td>0.47</td><td>0.13</td></tr><tr><td>Nurse</td><td>0.32</td><td>0.23</td></tr><tr><td>Assistant</td><td>0.35</td><td>0.15</td></tr><tr><td>Average</td><td>0.37</td><td>0.18</td></tr></table>

Table 6 shows that RAIGen achieves lower Attribute Presence than CLIP k-means across all six professions, with an average gap of 0.19 in absolute presence. Qualitatively, image-level clusters frequently mix dominant and minority instances within a single cluster, since CLIP embeddings cluster by overall visual similarity rather than by individual attribute factors. For example, a cluster surfaced for “doctor” often contains a mix of attire, demographics, and contexts, making it difficult to isolate which attribute makes that cluster “rare.” Several small clusters are also uninterpretable at the attribute level where they correspond to low-frequency visual patterns (e.g., specific lighting conditions, framing artifacts) without a coherent semantic attribute. By contrast, RAIGen’s MSAE-based decomposition factorizes generations along axes that align more cleanly with individual semantic concepts, yielding minority neurons that correspond to coherent, interpretable attributes. These results confirm that neuron-level decomposition is necessary for systematic minority-attribute discovery and image-level clustering, while simpler, does not surface the same set of suppressed attributes.

## G.7. RAIGen using vanilla SAE

While our primary experiments utilize MSAEs for hierarchical semantic decomposition, we also conduct a parallel analysis using standard SAEs to evaluate how the choice of decomposition method affects the discovery of minority attributes. We apply RAIGen to intermediate representations of diffusion models for the prompt “A photo of a Doctor”. In this setting, rather than using MSAE, we employ a vanilla SAE to identify neurons associated with underrepresented attributes. We visualize several of the top minority neurons discovered by our approach, with the results summarized in Figure 10.

To evaluate the role of the decomposition method, we repeated our analysis using a standard Sparse Autoencoder (SAE) in place of the Matryoshka SAE. As shown in Figure 10, the SAE was still able to surface minority attributes under our Minority Score, with neurons capturing features such as eyeglasses and contextual backdrops. However, consistent with observations in (Bussmann et al., 2025), SAE representations were considerably more fragmented, often distributing a single concept across multiple neurons. For example, we identified cases where one neuron responded primarily to the right eye and another to the left eye; although both were flagged as minority neurons due to their selective activations, they did not reflect genuine semantic minorities but rather over-fragmented features. This fragmentation reduced semantic coherence, making annotation less straightforward and limiting interpretability. Overall, these results demonstrate that while SAEs can uncover minority attributes, their tendency to fragment concepts across multiple neurons can be misleading, as neurons may appear to represent minorities while in fact capturing only partial or redundant features. In contrast, MSAEs mitigate this issue by providing hierarchical control over granularity and emphasizing global semantic features, enabling more faithful and interpretable discovery of genuinely underrepresented attributes.

## G.8. Robustness to choice of vision-language encoder

The semantic distinctiveness term in the Minority Score $( \mathrm { E q . ~ } 3 )$ is computed against CLIP image embeddings. To test whether RAIGen’s discoveries depend on the specific embedding model, we replace CLIP-ViT-L/14 with SigLIP (Zhai et al., 2023) and re-run the full discovery pipeline on the prompt “a photo of a doctor” (SDXL), keeping all other components (MSAE, activation frequency, ranking, redundancy filtering) unchanged. We then compare the top-10 minority neurons surfaced by each variant.

We observe that 9 of the top-10 neurons are recovered by both encoders, and the single non-overlapping neuron is noninterpretable under both. We attribute this stability to the nature of the distinctiveness term itself. It measures a coarse semantic contrast between a neuron’s activation-weighted centroid and the dataset-wide centroid, a signal that any visionlanguage model trained on large-scale image–text pairs captures reliably. The agreement between two encoders trained with different contrastive objectives (CLIP’s symmetric InfoNCE vs. SigLIP’s sigmoid loss) indicates that RAIGen’s discoveries reflect stable structure in the MSAE feature space rather than embedding-specific artifacts. Exploring more principled distinctiveness formulations (e.g., information-theoretic or contrastive-objective-aware) remains a promising direction for future work.

## G.9. Ablation on MSAE coarse level size for RAIGen

In this section, we perform an ablation on the number of neurons in the coarse sparsity level in MSAE which we utilize for RAIGen to understand its effect on the minority attribute discovery. We ablate the MSAE coarse sparsity level size $k _ { 1 } \in \{ 1 2 8 0 , 2 0 4 8 , 1 0 2 4 0 \}$ while fixing the sparse feature dimension to 20480 (input representation size = 1280, expansion factor = 16), and evaluate on the prompt “A photo of a doctor”. As shown in Table 7, attribute presence increases at larger $k _ { 1 }$ . Qualitatively, we observed that some demographic attributes $( \mathrm { e . g . }$ ., Black skin tone) are missed at $k _ { 1 } { = } 1 2 8 0$ but captured at $k _ { 1 } = 2 0 4 8$ . Balancing coverage against fragmentation, we adopt $k _ { 1 } = 2 0 4 8$ as the default, which reliably recovers key demographics without the over-fragmentation seen at very large $k _ { 1 }$ .

<table><tr><td>Number of neurons in  $k_1$ </td><td>Attribute presence</td></tr><tr><td>1280</td><td>0.15</td></tr><tr><td>2048</td><td>0.15</td></tr><tr><td>10240</td><td>0.20</td></tr></table>

Table 7. Ablation on the number of neurons in coarse sparsity level $k _ { 1 }$ in MSAE for RAIGen. Attribute presence is computed as in our evaluation protocol.

## G.10. Ablation on different components of the Minority Score

This section evaluates the design choice behind our Minority Score, which combines activation frequency and semantic distinctiveness to identify minority-associated neurons. Our goal is to recover neurons that correspond to underrepresented attributes while remaining semantically meaningful. We compare three neuron-ranking strategies under the prompt $^ { * } A$ photo of a doctor” in SDXL: frequency-only ranking using $\nu _ { i } .$ , semantic-distinctiveness based ranking using $d _ { i } ,$ and the combined Minority Score $s ( z _ { i } ) = d _ { i } \cdot ( 1 - \nu _ { i } )$ and consider top 10 neurons discovered in each case. We evaluate each selection using (i) the fraction of neurons labeled No identified attribute by the annotation model which we deem to be uninterpretable and (ii) the mean attribute presence of identified attributes, reported in Table 8.

<table><tr><td>Method</td><td>Uninterpretable ↓</td><td>Interpretable ↑</td><td>Mean attribute presence ↓</td></tr><tr><td>Frequency-only</td><td>0.625</td><td>0.375</td><td>0.080</td></tr><tr><td>Semantic distinctiveness-only</td><td>0.100</td><td>0.900</td><td>0.257</td></tr><tr><td>Minority score (Combined)</td><td>0.200</td><td>0.800</td><td>0.150</td></tr></table>

Table 8. Neurons are selected by frequency-only (decreasing $\nu _ { i } ) ,$ semantic distinctiveness-only (increasing $d _ { i } )$ , or the combined Minority Score $s ( z _ { i } ) = d _ { i } ( 1 - \nu _ { i } )$ (increasing).

![](images/f4896975d56cfc24f0187d72dff055296ce62913538327063deaacbc1cfc71da.jpg)

<details>
<summary>natural_image</summary>

Collage of medical professionals in various settings including stethoscope, hand-drawn surgery, and thermal imaging (no visible text or symbols)
</details>

![](images/c52011a480c04c3ed086809085e68f282de9f7568eab0a98842b2781d2a66af1.jpg)

<details>
<summary>natural_image</summary>

Grid of medical professionals in various lab coats and stethoscope, each with a photo frame and thermal imaging overlay (no visible text or symbols)
</details>

Figure 11. (a) Frequency-only minority neuron identification: Neuron with (Left) Frequency 0.01, and (Right) Frequency 0.06. These are the least frequently activated neurons, but appear noisy and uninterpretable.

Table 8 shows that frequency-only selection yields a high uninterpretable rate, revealing a common failure mode of raritybased ranking in practice: neurons can fire rarely for non-semantic reasons $( \mathrm { e . g . }$ , dead units or idiosyncratic low-level effects). In a controlled setting where the number of factors is small and the SAE is trained sufficiently well that features align nearly one-to-one with attributes, activation frequency $\nu _ { i }$ is a reliable signal of rarity. In realistic settings, we cannot assume such a perfect feature–attribute alignment, and low activation frequency often reflects sparse, noisy, or semantically inconsistent features rather than genuinely underrepresented attributes. Notably, the mean attribute presence for frequency-only selection is very low, indicating that when a low-frequency neuron is interpretable, it typically corresponds to a genuinely minority attribute, consistent with the controlled experiment, yet some of the neurons remain uninterpretable in practice, as also observed in Figure 11.

Next, we consider the case where neurons are ranked by semantic distinctiveness $d _ { i }$ alone. Table 8 shows that semantic distinctiveness-based selection is highly interpretable. Intuitively, $d _ { i }$ favors neurons that consistently activate on a coherent, non-generic concept whose semantics lie far from the dataset’s average representation. When a neuron repeatedly responds to such a specific concept, its top-activating images occupy a stable region in CLIP space, yielding an activation-weighted centroid $\mu _ { i }$ that remains reliably displaced from the dataset centroid $\mu _ { D } .$ c and therefore produces a large $d _ { i }$ . In contrast, noisy neurons tend to activate on a heterogeneous set of images with no consistent semantics; their CLIP embeddings cancel in different directions, causing $\mu _ { i }$ to drift toward the dataset mean and resulting in a much smaller distinctiveness score. However, semantic distinctiveness alone is insufficient for minority discovery, because deviation from the dataset’s average semantics may not always imply underrepresentation. A concept can be coherent and directionally far from the dataset centroid while still being common (i.e., a frequent sub-mode of the distribution). This is reflected in Table 8: although semantic distinctiveness-based selection achieves high interpretability, it yields a higher mean attribute presence, indicating that it can surface semantically coherent yet often prevalent attributes.

These complementary modes motivate the product form $s ( z _ { i } ) = d _ { i } \cdot ( 1 - \nu _ { i } )$ . As observed in Table 8, the combined criterion preserves most of the interpretability gained by distinctiveness selection while substantially reducing attribute prevalence, and it lowers the fraction of uninterpretable neurons compared to frequency-only selection. While each ablation is optimal on a single axis, the combined score yields the best tradeoff by maximizing the share of interpretable neurons with low attribute presence. Conceptually, $d _ { i }$ acts as a semantic-coherence filter that suppresses rare-but-unstructured activations, whereas $( 1 - \nu _ { i } )$ captures underrepresentation and downweights coherent but frequent concepts. Thus, reliably identifying minority-associated neurons requires both signals to avoid selecting either rare noise or common, semantically coherent attributes.

## G.11. Interpretability generalization across prompts

The Minority Score ablation in Appendix G.10 reports interpretability and mean Attribute Presence on a single prompt (“a photo of a doctor,” SDXL). To verify that the high interpretability rate of the combined Minority Score is robust across settings, we extend the analysis to four additional WinoBias professions and five COCO prompts, all evaluated on SDXL using the same protocol.

Table 9. Interpretability and mean Attribute Presence of RAIGen-discovered minority neurons across additional WinoBias professions, on SDXL. Interpretable rates remain comparable to or higher than the 0.80 reported in Table 8 for “doctor.”

<table><tr><td>Profession</td><td>Interpretable ↑</td><td>Mean Attr. Presence ↓</td></tr><tr><td>Farmer</td><td>0.80</td><td>0.251</td></tr><tr><td>Sheriff</td><td>1.00</td><td>0.177</td></tr><tr><td>Nurse</td><td>0.80</td><td>0.229</td></tr><tr><td>Attendant</td><td>0.70</td><td>0.128</td></tr></table>

Table 10. Interpretability and mean Attribute Presence of RAIGen-discovered minority neurons across five COCO prompts, on SDXL.

<table><tr><td>Prompt</td><td>Interpretable ↑</td><td>Mean Attr. Presence ↓</td></tr><tr><td>A large white building with a big clock tower at one corner.</td><td>1.00</td><td>0.083</td></tr><tr><td>A couple of people carrying surf board walk on a beach.</td><td>0.90</td><td>0.194</td></tr><tr><td>A batter swings the bat as the crowd watches attentively.</td><td>0.80</td><td>0.085</td></tr><tr><td>A young skater is boarding inside of an empty pool.</td><td>0.80</td><td>0.156</td></tr><tr><td>A man talks on his cell phone while he surfs his computer.</td><td>1.00</td><td>0.210</td></tr></table>

Across all nine additional settings, the interpretable rate ranges from 0.70 to 1.00, closely matching the 0.80 reported in Table 8 for the original “doctor” prompt and exceeding it in seven of nine cases. Mean Attribute Presence is consistently low (between 0.083 and 0.251), confirming that the discovered attributes remain genuinely underrepresented across both occupation prompts and open-domain COCO scenes. These results indicate that RAIGen’s discovery quality is robust to prompt variations.

## G.12. Sample-complexity study: how many generations are needed?

A practical question for RAIGen is how many generated samples are required before minority-attribute discovery becomes stable. To investigate this, we vary the number of generated samples per prompt $N \in \{ 1 0 0 , 1 0 0 0 , 5 0 0 0 , 1 0 0 0 0 \}$ , train an MSAE on the resulting bottleneck representations, and apply the full RAIGen pipeline. We use the prompt “a photo of a doctor” on SDXL and evaluate two complementary properties: (i) the fraction of top discovered neurons that are interpretable, judged by whether the annotation model identifies a consistent semantic attribute across top-activating images (Interpretable Neurons); and (ii) the mean Attribute Presence of identified attributes, where lower presence indicates that the discovered attributes are genuinely rare in default sampling.

Table 11. Effect of the number of generated samples per prompt on RAIGen discovery, for the prompt “a photo of a doctor” on SDXL. Discovery becomes stable around $\bar { N } = 5 0 0 0$ .

<table><tr><td>Samples (N)</td><td>Interpretable Neurons ↑</td><td>Attr. Presence ↓</td></tr><tr><td>100</td><td>0.1</td><td>—</td></tr><tr><td>1,000</td><td>0.5</td><td>0.41</td></tr><tr><td>5,000</td><td>0.8</td><td>0.15</td></tr><tr><td>10,000</td><td>0.9</td><td>0.14</td></tr></table>

Table 11 summarizes the results. At $N = 1 0 0 ,$ discovery is unreliable since too few neurons are interpretable to compute a stable Attribute Presence estimate. Both metrics improve substantially up to $N = 5 0 0 0$ , after which gains are marginal. Doubling the sample budget from 5,000 to 10,000 improves interpretability from 0.8 to 0.9 and Attribute Presence from 0.15 to 0.14. We therefore adopt $N = 5 0 0 0$ throughout the paper as a cost–quality operating point that captures most of the asymptotic discovery quality at half the compute budget of $N = 1 0 0 0 0$ . Practitioners with tighter compute budgets can use N as low as 1000 at the cost of reduced interpretability and noisier rarity estimates.

## G.13. Computational cost

We report end-to-end wall-clock time for a single prompt on SDXL, measured on a single NVIDIA A100 (40 GB) GPU. Reported times are for 5,000 generated samples per prompt (the operating point selected in Appendix G.12).

Table 12. End-to-end wall-clock time for one prompt on SDXL, single A100 (40 GB).

<table><tr><td>Stage</td><td>Time</td></tr><tr><td>Image generation + bottleneck extraction (5,000 samples)</td><td>55 min</td></tr><tr><td>MSAE training + neuron discovery</td><td>16 min</td></tr><tr><td>LLM annotation of discovered neurons</td><td>44 sec</td></tr><tr><td>Total per prompt</td><td>~72 min</td></tr></table>

As observed in Table 12, the dominant cost is image generation rather than MSAE training or annotation. This cost is not specific to RAIGen. Any open-set auditing framework that operates on generated samples (e.g., OpenBias (D’Inca et al. \` , 2024)) incurs a comparable cost, since large-scale sampling is unavoidable for identifying systematic failure modes.

## H. Qualitative results

In this section, we present qualitative results illustrating the minority neurons identified by RAIGen across different datasets and prompts. Figures 12 and 13 show minority neurons corresponding to different WinoBias professions from the representations of SDXL. Figures 14 and 15 show minority neurons corresponding to different WinoBias professions from the representations of SD v1.4. Figure 16 displays minority neurons obtained for a set of diverse COCO prompts in SD v1.4. Finally, Figures 17 present the minority neurons identified for the prompt “A photo of a Doctor” in SD v2.1. For each neuron, we strongly activating images along with their corresponding activation heatmaps, which highlight the visual features. Since RAIGen surfaces several minority neurons, we present only a representative subset in the figures for clarity and illustration.

Assistant  
![](images/7301cdb8ec10cd3ccccbaee9ee78ab5ec7e5390706205bce9849552004f47576.jpg)

<details>
<summary>natural_image</summary>

Collage of eight dog portraits and corresponding thermal imaging overlays, no text or symbols present
</details>

![](images/2d8bf0f81e3b427b0f738beb368ad1cc9a499ae99c4d935c18259565287baaa2.jpg)

<details>
<summary>natural_image</summary>

Collage of outdoor dog and dog photos showing various activities including patrolling, walking, and relaxing (no text or symbols visible)
</details>

![](images/ac16cd01ff13a1c1dd250ebaf5db447b82e30e2311b0614e4e304715112fd0a8.jpg)

<details>
<summary>natural_image</summary>

Illustration of office scenes with employees at desks, computer workstations, and thermal imaging overlays (no text or symbols)
</details>

![](images/d00eabba8816895bc2cfad56564afc695dbb13457bb6e79638717468dc32b9c4.jpg)

<details>
<summary>natural_image</summary>

Six-panel collage showing a woman and two men in an office, each with a cat and a heatmap overlay (no text or symbols)
</details>

![](images/f0c27ce30f172246292bb2d4a0c415b1f681294232c6663457e59d3f00d89d35.jpg)

<details>
<summary>text_image</summary>

L'code poultrevenue l'été fair
enir d'ab de lauf en casseurer
CENAI CEO
CENAI CEO
</details>

![](images/e1abd32fa848b734876d7326bc34ef1264020f4baee9dfedcd7ac22c6e82dd9e.jpg)

<details>
<summary>natural_image</summary>

Panoramic interior view of a modern office with multiple windows and desks, showing interior layouts and spatial overlays (no visible text or symbols)
</details>

![](images/81d8ba54bd3611f4d536a226ac46576abdd7d682e59e617c9106ad1b5ffccfc6.jpg)

<details>
<summary>natural_image</summary>

Six portrait photos of men in business attire, each with a corresponding thermal map overlay showing head silhouette (no text or symbols)
</details>

![](images/b1dfb0df9601d25e4c16390d5c91687677309e2789538027492be896cdbad9eb.jpg)

<details>
<summary>natural_image</summary>

Group of business professionals in a meeting setting, each with laptop and thermal imaging overlays (no visible text or symbols)
</details>

urse  
![](images/f25513bc9e2cde385394644566d7a0430c7702d7de9711461597fac266024964.jpg)

<details>
<summary>natural_image</summary>

Series of black-and-white photos of nurses in various poses, framed within a filmstrip border (no text or symbols visible)
</details>

![](images/495fa1fa04fbecab9423d1531618c36dbb0ce26ca4ee8bc3f2cd0225227e2d9d.jpg)

<details>
<summary>natural_image</summary>

Group of medical professionals in white coats and stethoscope, each with a corresponding thermal heatmap overlay (no text or symbols)
</details>

![](images/73142542031383b37cd7f48a9c4a53b8c1479226e189da925fa68c28e5035042.jpg)

<details>
<summary>natural_image</summary>

Group of medical professionals in various settings including face masks and thermal imaging overlays (no text or symbols)
</details>

![](images/cd61edefdd99e0e4383579494d310c18adecaf74f770b730e6a6d9187cdbc393.jpg)

<details>
<summary>natural_image</summary>

Grid of nine medical professionals in various headgear and face masks, each with a corresponding thermal or clinical scan image (no text or symbols visible)
</details>

Figure 12. Visualization of 4 minority neurons obtained by RAIGen for different WinoBias professions in SDXL. For each neuron, we show the top five activating images and their corresponding activation heatmaps.

![](images/978f9e768d56762dc508631f10deb4c4a9393871d8670d4e8180102e3d3dec3b.jpg)

<details>
<summary>natural_image</summary>

Collage of fashion photos showing people in pink and purple hats, including phone calls, earphones, and face masks (no visible text or symbols)
</details>

(a) “A woman in a pink hat is on the phone.”  
![](images/c6542d51ee182f15376abdb61c88c9a3c5f1f58ac97d874489287555138ab4c1.jpg)

<details>
<summary>natural_image</summary>

Collage of various male and female professionals in various attire and outdoor scenes, including aircraft, airports, and digital maps (no visible text or symbols)
</details>

(b) “A man in suit and tie is wearing sunglasses.”  
![](images/087cc9951e55ef55c42cd56ae75e690ccc5453f60d9eb5ba5334a81c9dbd46f4.jpg)

<details>
<summary>natural_image</summary>

Collage of 12 diverse scenes showing people using laptops in various settings, with heatmaps and computer screens (no visible text or symbols)
</details>

(c) “A man talks on his cell phone while he surfs his computer.”  
![](images/6abb2bb2c4e63c108ae4e69313a768a8e414bdf6bc18602abd85da28930ff562.jpg)

<details>
<summary>natural_image</summary>

Collage of street scenes showing motorcycles in various locations, including a classical building and commercial building, with thermal or heat map overlays (no visible text or symbols)
</details>

(d) “A motorcycle parked outside the doors of a building.”  
![](images/2b81f3eed26b62903b8502c60fd25ad0378917945e88ef462212ceb09c294a44.jpg)

<details>
<summary>natural_image</summary>

Collage of wedding and banquet photos including formally dressed individuals, grooms, and guests (no visible text or symbols)
</details>

(e) “A young couple in formal evening attire toasts the crowd.”  
Figure 13. Visualization of 2 minority neurons obtained by RAIGen for different COCO prompts in SDXL. For each neuron, we show the top five activating images and their corresponding activation heatmaps.

![](images/f78007e41186771b9512c4f5d4143bd4d116d90dddcfb92d1541962c2f06e733.jpg)  
Figure 14. Visualization of 4 minority neurons obtained by RAIGen for different WinoBias professions. For each neuron, we show the top five activating images and their corresponding activation heatmaps.

![](images/8cfe923bfbfcf1a6439c720ddd44c513a2853aaa125377522ef533170ab728a2.jpg)

<details>
<summary>natural_image</summary>

Collage of construction workers and thermal imaging scenes showing excavators, safety vests, and vehicle displays (no text or symbols)
</details>

![](images/50bf437ec2e56e4ec4463c48bbe3cceea1c0b30948c0ad92a5f8bf9c54dd0bd4.jpg)

<details>
<summary>natural_image</summary>

Collage of construction site photos showing workers in safety vests and helmets, alongside thermal heatmaps of a person's body (no text or symbols)
</details>

![](images/4ae9c8869dfd34af32df4bbc470ea5a0010b86042b162e10bfcf7df68fdcfc21.jpg)

<details>
<summary>natural_image</summary>

Composite image showing construction workers in hard hats alongside thermal or heat map overlays (no text or symbols)
</details>

![](images/b060a7d1a0f8cc23cea41caecdd8862c80720d1a4aa237cf5e06d27ecde171ef.jpg)

<details>
<summary>natural_image</summary>

Collage of construction workers in hard hats and helmets, some working on soil, others inspecting or applying material, with a thermal heatmap overlay (no text or symbols)
</details>

snn  
![](images/3bd58298571328c478fe97a706d63385511fef457c85622c38385504d37a939b.jpg)

<details>
<summary>natural_image</summary>

Group of black-and-white photos showing medical professionals in various settings, with thermal or heat map overlays below (no visible text or symbols)
</details>

![](images/babc5824c0b1061cec917d10423a99aa094b56a60eaad0634d968bc94601b3a8.jpg)

<details>
<summary>natural_image</summary>

Six medical professionals in professional attire displaying thermal imaging overlays (no text or symbols visible)
</details>

![](images/33f2b42f6e1d58f8b74aad5ac0c279b397b23f9231310ae11eb92f9066a0531e.jpg)

<details>
<summary>natural_image</summary>

Medical personnel in scrubs and gowns performing a walking motion alongside thermal imaging of a patient's torso (no text or symbols visible)
</details>

![](images/3340da4aa0377ea048cdf4a2907624836b70e86e5169e6397e0ae967d0abf305.jpg)

<details>
<summary>natural_image</summary>

Grid of nine medical professionals in various angles and head expressions, each with a corresponding thermal or heatmap visualization (no text or symbols)
</details>

![](images/b943759f2b54d71f4316d0252f9e9207996889c3316ccd586c6659554a4841b2.jpg)

<details>
<summary>natural_image</summary>

Group of eight professionals wearing headsets and face masks, with thermal imaging overlays showing heat signatures (no text or symbols)
</details>

![](images/b89a6ea2242a3999b9954236011b3876de3000dca265bbcb6d708b611e59415b.jpg)

<details>
<summary>natural_image</summary>

Six-panel collage showing a woman in professional attire, with thermal or heatmap overlays of the same person's face (no text or symbols visible)
</details>

![](images/f6bcf3061b338ecdd23d0038f7c6634a18837e9eeb06450391080098a095b337.jpg)

<details>
<summary>natural_image</summary>

Office scene with employees working at desks and thermal imaging overlays showing heat signatures (no text or symbols)
</details>

![](images/23b10f83eb90b501560e6628ab43340ecce569f08aa9fcc7bf85ae7f9584e901.jpg)

<details>
<summary>natural_image</summary>

Collage of office staff in various roles and thermal imaging displays, including computer monitors and heatmaps (no visible text or symbols)
</details>

Figure 15. Visualization of 4 minority neurons obtained by RAIGen in SD v1.4 for different WinoBias prompts. For each neuron, we show the top five activating images and their corresponding activation heatmaps.

![](images/275be6b4021efaa5d0b5d6a348b3961602d2d846a97922c0dbdcdf096c681bae.jpg)

<details>
<summary>natural_image</summary>

Sequence of outdoor skateboarding exercises showing various poses and motion patterns (no text or symbols)
</details>

![](images/93b5fe8086bebe45e347e1f5e49c0dd2fd0ff6a8e204e2a761387b59e2c52cb1.jpg)

<details>
<summary>natural_image</summary>

Collage of 12 photos showing people on ice skates with heatmaps, including motion visualization and temperature gradient overlays (no text or symbols)
</details>

(a) “This is a photo of someone standing on the skating board”  
![](images/4d042247f14fb08488aee36bb0dda2e3bfc88ebdafbc974ff9fba40df6616d36.jpg)

<details>
<summary>natural_image</summary>

Collage of beach scenes including surfboards, palm trees, and beach murals under a tropical sky (no text or symbols)
</details>

![](images/1cf2ea59de365ab9195ce5ae7e69567e1281ee16020d6f617f05d07451dff8bc.jpg)

<details>
<summary>natural_image</summary>

Collage of beach scenes including beach rentals, beach swings, and surfboards with thermal or heat map overlays (no text or symbols)
</details>

(b) “The surfing board is on the sand of the beach”  
![](images/3462f406b92afb0a1c3c9053820115aec7a5e760b762635454e1e5cec85993be.jpg)

<details>
<summary>natural_image</summary>

Grid of nine photos showing yellow and blue bird perched on branches, with no visible text or symbols.
</details>

![](images/f7eb17a4ff2d747991b4c14be8fb86bae0f8d80397ee3a4e999b11838ff2b8db.jpg)

<details>
<summary>natural_image</summary>

Six-panel collage showing yellow and gray birds perched on branches, with blue-and-white thermal images of their beaks (no text or symbols)
</details>

(c) “A small yellow bird sitting on a tree branch”  
![](images/d508baa1cfb245fa8beb37d2447302d9e0bfcaed08a3fae52d0069bb858332ed.jpg)

<details>
<summary>natural_image</summary>

Overhead view of various office computers and devices including laptops, laptops with smart glasses, and laptops with heatmaps (no text or symbols visible)
</details>

![](images/a1c175e8a12519a5d6fc150f86d01aefd27196ef768cecf2f8a73b265b40e6c0.jpg)

<details>
<summary>natural_image</summary>

Grid of blue laptops displaying heatmaps on a wooden surface, no text or symbols visible
</details>

(d) “A nice blue laptop on a messy table”  
![](images/2fb755295d328e6b2181cab1958090b453379dffda26f35420105385cd4d83db.jpg)

<details>
<summary>natural_image</summary>

Grid of eight photos showing children playing with devices and a heatmap overlay, no visible text or symbols
</details>

![](images/e19100b620db64a0dfe2bf9e27a79a6feb4b6183b0e94936dc19fd841c4afc89.jpg)

<details>
<summary>natural_image</summary>

Grid of eight photos showing young girls playing with controllers in different settings (no text or symbols visible)
</details>

(e) “A little girl sitting on the couch playing”  
Figure 16. Visualization of 2 minority neurons obtained by RAIGen in SD v1.4 for different COCO prompts. For each neuron, we show the top five activating images and their corresponding activation heatmaps.

![](images/e49c45685f523ff5dae9939493747ee1543f36f7391ae793b857e86f22e942fa.jpg)  
Figure 17. Visualization of 8 minority neurons identified by RAIGen for the prompt “A photo of a Doctor” in SD v2.1. For each neuron, we show the top five activating images and their corresponding activation heatmaps.