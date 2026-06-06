# Diff-CA: Separating Common and Salient Factors with Diffusion Models

Michaël Soumm

INRIA at Univ. Grenoble Alpes michael.soumm@inria.fr

Alexandre Fournier Montgieux

CEA List, Palaiseau alexandre.fourniermontgieux@cea.fr

Yunlong He

Télécom Paris, Institut Polytechnique de Paris yunlong.he@telecom-paris.fr

Pietro Gori

Télécom Paris, Institut Polytechnique de Paris pietro.gori@telecom-paris.fr

Alasdair Newson

Télécom Paris, Institut Polytechnique de Paris alasdair.newson@telecom-paris.fr

# Abstract

Contrastive Analysis aims to separate factors that are common between two data distributions from those that are salient to only one of them. Existing contrastive methods are based on generative models (e.g., VAEs or GANs) that often suffer from limited reconstruction and image quality, which hampers effective latent factor separation and limits their applicability to high-fidelity image generation and edition. We propose a novel conditioning framework for diffusion models that enables contrastive decomposition without compromising generation quality. We first train a prompt-free, image-conditioned diffusion model, and then learn to decompose the conditioning into a common and a salient factor, using weak supervision. We prove that the additive contrastive factorization, commonly assumed in prior work, is identifiable under mild conditions. This factorization enables targeted operations by swapping or interpolating only the salient factor.

# 1 Introduction

Learning common and salient information between data distributions is an important task in many domains of representation learning, such as multi-view [49, 13, 51, 43, 11], multi-modal [12] representation learning, domain adaptation [31], subgroups discovery [35, 38] and disentanglement [46]. In this article, we focus on Contrastive Analysis (CA), which is the problem of separating what is common to two data distributions from what is salient to one of them. In this setting, we observe two unpaired image distributions: a background distribution $p _ { X \mid Y = 0 }$ and a target distribution $p _ { X \mid Y = 1 }$ , where X represents an image and $Y \in \{ 0 , 1 \}$ } indicates which distribution it belongs to (i.e., a weak binary signal). The target distribution contains the additional, salient factor, which is absent in the background.

Using as example Fig. 1, where the two datasets contain face images without glasses $( Y = 0 )$ and with $( Y = 1 )$ glasses, the goal of CA is to estimate a representation that decomposes images into common $C \left( e . g . \right.$ ., identity, gender) and salient factors S (e.g., glasses style and color). Other examples include medical images with or without a pathology or tumor, and images of products with or without defects, such as metallurgical anomalies (e.g. cracks, fissures) in industrial inspection. Binary group membership (healthy/pathological or pass/fail) are labels that are routinely assigned in hospitals or in industry. However, they might be obtained through intrusive tests (e.g., biopsy, blood tests) and/or be based on complex, salient patterns that can be difficult to capture using text prompts or that are not necessarily known in advance. The identification of such patterns could enable diagnosis via noninvasive imaging and a better understanding of the disease. For all these applications, CA provides a principled framework for automatically discovering, isolating, and modeling the salient S and common C factors, enabling targeted operations (swapping, interpolation, synthetic/counterfactual generation, or anomaly quantification) without fine-grained annotations or text prompts.

![](images/93a97c4193d630f3d09a601e06e6e96c0e56ac94bc0dbc648a62c889c08d7fed.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Background (No Glasses)"] --> B["Target (Glasses)"]
    C["unpaired"] --> B
    D["weak binary supervision"] --> E["Common factors"]
    D --> F["Salient factors"]
    G["Swap glasses"] --> H["Swapping salient factors"]
    I["swap pose"] --> H
    J["swap cat/dog"] --> H
    K["swap tumor"] --> H
    L["identity eyes pose"] --> M["common factors"]
    N["gender hair color"] --> O["common factors"]
    P["style shape, color, size,..."] --> Q["salient factors"]
```
</details>

Figure 1: Contrastive Analysis separates common from salient factors between two unpaired data distributions using only weak binary supervision (i.e., dataset-level). The learned common and salient latent spaces can be manipulated for editing or knowledge discovery, for instance.

In this work, we propose a Contrastive Analysis approach based on latent diffusion models. Our method applies CA directly to the conditioning token sequence Z of a latent diffusion architecture. We learn a decomposition $Z \mapsto ( \hat { Z } _ { C } , \hat { Z } _ { S } )$ such that $\hat { Z } _ { C }$ captures the common factor shared across distributions (C), while $\hat { Z } _ { S }$ isolates the salient factor specific to the target distribution (S). This yields an effective CA-aligned intervention mechanism: swap, remove, or scale $\hat { Z } _ { S }$ while keeping $\hat { Z } _ { C }$ fixed. CA thus provides an alternative to text-driven diffusion editors, which often rely on prompt/inversion/attention heuristics [22, 5, 39, 41] that can be sensitive to prompt/guidance choices, introduce unintended changes , or that cannot be applied when salient patterns are difficult to describe or are unknown. We further discuss the limitations of these editors in Appendix B.

Furthermore, we provide a theoretical analysis of the common-salient decomposition based on the additive decomposition: $Z = \hat { Z } _ { C } + \hat { Z } _ { S } ,$ , which is commonly assumed in CA [1, 6, 36, 37, 52, 21]. We also prove its identifiability under mild conditions, showing that the additive decomposition in the token space enables the recovery of the true underlying common and salient factors.

Our contributions are as follows:

• Theoretical Analysis for CA: Section 3 provides a rigorous theoretical framework for Contrastive Analysis in the context of generative models. We derive a lower bound on controllability based on mutual information (Prop. 1) and prove that under an additive structural assumption, the true common and salient factors are uniquely identifiable from weak supervision (Thm. 1).   
• High-fidelity latent space for diffusion models: We construct in Sec. 4.1 a conditioning latent space for diffusion models to match the structural assumptions of Sec 3. By training a Cross-Query encoder with DINOv3 features and a novel Color Token, we obtain a latent space suitable for both high fidelity reconstruction and generated image attribute manipulation.   
• CA on diffusion conditioning tokens: In Sec 4.2, we introduce our separator architecture, Diff-CA, and translate the theoretical constraints into practical training losses applied to the token latent space. As evaluated in Sec. 5 across multiple domains, Diff-CA isolates and edits salient attributes

while effectively preserving common factors (e.g., identity, gender), achieving state-of-the-art reconstruction and swapping performance compared to existing CA baselines.

# 2 Related Works

Contrastive analysis. Classic CA formulations include contrastive PCA [2], and VAE based architectures with different regularizations to enforce a common/salient latent separation [1, 52, 36, 3, 4, 46, 29]. More recent CA work explores adversarial generators to improve sample quality while retaining a common/salient decomposition [15, 6], or representation-level separation without relying on high-fidelity generation [37]. A recurring limitation is that imperfect reconstruction or synthesis prevents latent factors from being effectively separated.

Diffusion conditioning as a controllable interface. Diffusion models provide high-fidelity synthesis [24], and latent diffusion allows for external conditioning through cross-attention over conditioning embeddings/tokens [44]. Several works expand conditioning pathways to improve control, e.g. by adding dedicated conditioning branches or image-derived prompts [56, 54]. In parallel, other lines of work aim to expose more structured latent variables for diffusion to enable semantic manipulation [42, 30, 40, 26]. Our work is complementary: instead of proposing a new editing heuristic or a new global latent, we bring CA to diffusion by decomposing the image-conditioning tokens themselves into common C and salient S under the target/background weak supervision signal.

Diffusion image editing. A large body of diffusion editing methods performs user-specified edits via prompts, attention control, and/or inversion [22, 39, 5, 45]. These approaches can produce impressive edits, but they are not designed to identify and separate dataset-level common and salient generative factors between two distributions, under weak supervision. Instead, they steer generation through prompt- or inversion-dependent mechanisms.

Disentanglement. Conceptually, our goal is also related to disentanglement, which aims to produce changes of semantically meaningful attributes (e.g., gender) by altering a single latent component. Disentanglement is known to be ill-posed without additional constraint or supervisions [23, 7, 34] and its objective is complementary to the one of CA. Indeed, CA deals with the latent separation of the generative factors C and S, which may be further disentangled (i.e., one factor per attribute), using prior information about salient/common attributes (left as future work in this article).

# 3 Theoretical Analysis of CA decomposition

In this section, we introduce the additive structural assumptions of the CA framework. Furthermore, we present theoretical analysis to show that these assumptions lead to an identifiable CA decomposition.

We consider image-label pairs $( x , y ) \sim p _ { X , Y }$ in $\mathcal { X } \times \{ 0 , 1 \}$ , where y is a binary indicator of the absence / presence of the target attributes. We refer to samples from $p _ { X \mid Y = 0 }$ and $p _ { X \mid Y = 1 }$ as negative (or background) and positive (or target) samples, respectively. We assume that X is generated by an unknown, deterministic and invertible process g applied to two unobserved and independent generative factors S and $C ,$ such that $g : \mathcal { S } \times \mathcal { C } \to \mathcal { X }$ , where $S \in { \mathcal { S } } \subset \mathbb { R } ^ { d _ { S } }$ is the salient factor and $C \in \mathcal { C } \subset \mathbb { R } ^ { d _ { C } }$ the common one:

![](images/6b903be7bf2c9e1fa4b83565e50640d1839e51e0d46a48f88a3bcb61b9336f1f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    S["Cloud S,C"] -->|g| X["X"]
    X --> f["f_θ"]
    f --> Z["Z"]
    Z --> E["E_θE"]
    E --> Z
    Z --> Ŝ_S["ˆZ_S"]
    Ŝ_S --> Ω[+]
    Ω --> Ŝ[Ẑ]
    Ŝ --> G["G_ψ"]
    G --> Xhat["ˆX"]
    Xhat -->|g⁻¹| Ŝhat
    ε[ε] --> G
```
</details>

Figure 2: Conceptual view of our decomposition. Factors $( S , C )$ generate an observed image X. We learn a feature extractor $f _ { \theta }$ that projects the images into latents $Z ,$ , and an encoder $E _ { \theta _ { E } }$ that splits $Z ,$ into $( \hat { Z } _ { S } , \hat { Z } _ { C } )$ aimed at representing $( S , C )$ . $( 2 0 , 2 0 )$ sum to $\hat { Z } _ { \cdot }$ , which conditions a diffusion model $G _ { \psi }$ aimed at generating an approximation $\hat { X }$ of X .

$$
X = \left\{ \begin{array}{l l} g (0, C) & \text { if   } Y = 0 \\ g (\underbrace {S} _ {\neq 0}, C) & \text { if   } Y = 1, \end{array} \right. \quad \text { and } \quad C \perp S. \tag {1}
$$

The factor S encodes the target attributes, describing their characteristics such as shape, texture, or style. Since C should fully represent background samples $( Y = 0 )$ , we model the absence of salient information in X by fixing $S = 0$ , as is commonly done in CA [2, 1, 6, 36, 37, 52].

Our goal is thus to learn, from images X and weak binary labels Y alone, a representation $( \hat { Z } _ { S } , \hat { Z } _ { C } )$ that mirrors the $( C , S )$ decomposition, where $\hat { Z } _ { S }$ models all of the variability of the salient factor S (not simply its absence / presence), while $\hat { Z } _ { C }$ models the common factors C.

# 3.1 Additive Structure of the Latent Space and Identifiability

Additive decomposition. Before aiming for a decomposition of salient and common information, we need to learn a conditioning representation $Z = { \dot { \gamma } } _ { \theta } ( X ) \in \mathbb { R } ^ { d }$ that accurately captures both common and salient factors. To make the decomposition of S and C tractable, we postulate that the encoder $f _ { \theta }$ maps the complex non-linear interactions of the pixel space into a structured linear additive latent space, inspired by previous work on CA [52, 36, 37, 6, 34]. Formally, we assume the existence of mappings $\varphi _ { S } : { \mathcal { S } } \dot {  } \mathbb { R } ^ { d }$ and $\varphi _ { C } : \mathcal { C } \to \mathbb { R } ^ { d }$ such that the conditioning $\dot { Z }$ decomposes as

$$
\operatorname{Im} (f _ {\theta}) = \operatorname{Im} (\varphi_ {S}) \oplus \operatorname{Im} (\varphi_ {C}) \quad \text { i.e. } Z = Z _ {S} + Z _ {C}, \tag {2}
$$

where ⊕ denotes a direct sum and we define the encoded factors as $Z _ { S } : = \varphi _ { S } ( S )$ and $Z _ { C } : = \varphi _ { C } ( C )$ . Since $S \perp \perp C$ by definition, $Z _ { S }$ and $Z _ { C }$ are also statistically independent. While high-level semantic linearity is a strong assumption, we empirically validate in Sec. D.1 that our DINOv3-based token representation supports such vector arithmetic. Importantly, weak supervision provides a pinning constraint. In the background distribution $( Y = 0 )$ , the salient factor is absent $( S = 0 )$ . We require the salient mapping to vanish at this reference point: $S = 0 \implies Z _ { S } = 0 .$ .

Identifiability of the Decomposition. The direct sum assumption implies that a unique decomposition of Z exists, but we do not know the subspaces for $Z _ { S }$ and $Z _ { C }$ that span it. Therefore, we cannot simply project Z, we must learn these subspaces. To this end, we introduce a separator encoder network to separate the two: $E _ { \theta _ { E } } ( Z ) : = ( \hat { Z } _ { S } , \hat { Z } _ { C } )$ , aimed at inferring $( Z _ { S } , Z _ { C } )$ given Z. We refer to $( \hat { Z } _ { S } , \hat { Z } _ { C } )$ as latent codes (learned) to distinguish them from true common/salient factors (theoretical).

A key question is under what theoretical conditions we can guarantee the recovery of the true common/salient factors, avoiding spurious decompositions. The following theorem establishes that three structural assumptions are sufficient to ensure identifiability.

Theorem 1 (Identifiability of Additive Factors). Assume the conditioning Z is generated by the direct sum of independent factors $Z _ { S }$ and $Z _ { C } ,$ , where $Z _ { S }$ vanishes for background samples $( Y = 0 )$ . If a learned decomposition $Z \mapsto ( \hat { Z } _ { S } , \hat { Z } _ { C } )$ satisfies:

1. Additive Reconstruction: The learned codes sum to the input: $\hat { Z } _ { S } + \hat { Z } _ { C } = Z$   
2. Pinning: The salient code vanishes only for background samples: $Y = 0 \Longleftrightarrow \hat { Z } _ { S } = 0 .$   
3. Independence: The learned codes are independent: $\hat { Z } _ { S }$ ⊥⊥ $\hat { Z } _ { C }$

Then, the learned codes uniquely identify the true encoded factors:

$$
\hat {Z} _ {S} = Z _ {S} \quad a n d \quad \hat {Z} _ {C} = Z _ {C} \tag {3}
$$

The proof is in App. C.1. This result provides the theoretical justification for our approach. While the true subspaces for $( Z _ { S } , Z _ { C } )$ are unobserved, the theorem guarantees that any decomposition $( \hat { Z } _ { S } , \hat { Z } _ { C } )$ satisfying these three structural constraints will perfectly recover the true underlying factors. We present the practical training objectives used to enforce these constraints in Section 4.2. Importantly, this result requires no assumptions about the intrinsic effective dimension of $\hat { Z } _ { S }$ and $\hat { Z } _ { C }$ .

# 3.2 Control over Generative Models

To translate the learned latent decomposition into explicit semantic control over pixels, we introduce a conditional generative model $G _ { \psi }$ . Given the conditioning representation $Z ,$ it approximates the

data distribution via :

$$
\hat {X} = G _ {\psi} (Z, \varepsilon), \quad \varepsilon \sim \mathcal {N} (0, I), \tag {4}
$$

where ε is independent noise. We denote by $\hat { S } \in \cal S$ and $\hat { C } \in \mathcal { C }$ the latent factors of the generated image Xˆ .

To ensure that varying the $\hat { Z } _ { S }$ component within Z effectively modifies only the salient factor of the generated image (independent of the common information or noise), we quantify control via the mutual information $I ( \hat { S } ; \hat { Z } _ { S } )$ . Maximizing this quantity ensures that the latent code $\hat { Z } _ { S }$ carries significant information about the output $\hat { S }$ (and similarly for $\hat { C }$ and $\hat { Z } _ { C } )$ .

The next result establishes that valid control is guaranteed if the model satisfies three conditions: high fidelity to the real factors S and C, low noise dependance of the generated images and low entanglement between $\hat { Z } _ { C }$ and $\hat { Z } _ { S }$ .

Proposition 1 (Control decomposition). Assume a decomposition $( \hat { Z } _ { S } , \hat { Z } _ { C } )$ satisfies the identifiability conditions of Th. 1, with $Z = \hat { Z } _ { S } + \hat { Z } _ { C }$ Then, control is lower-bounded by:

$$
I (\hat {S}; \hat {Z} _ {S}) \geq \underbrace {I (S ; \hat {S})} _ {\text { Fidelity }} - \underbrace {I (\hat {X} ; \varepsilon | Z)} _ {\text { Noise   dependence }} - \underbrace {I (S ; \hat {Z} _ {C} | \hat {Z} _ {S})} _ {\text { Entanglement }} \tag {5}
$$

and, for the common context:

$$
I (\hat {C}; \hat {Z} _ {C}) \geq I (C; \hat {C}) - I (\hat {X}; \varepsilon | Z) - I (C; \hat {Z} _ {S} | \hat {Z} _ {C}) \tag {6}
$$

The proof can be found in App. C.2. By the Data Processing Inequality, $I ( \hat { S } ; S ) \le I ( \hat { X } , X )$ (and similarly $I ( \hat { C } ; C ) \leq I ( \hat { X } , X ) )$ . This means that to effectively use $\hat { Z } _ { S }$ and $\hat { Z } _ { C }$ to control the factors of a generated image, we need $f _ { \theta }$ and $G _ { \psi }$ to act as an autoencoder $X \mapsto { \hat { X } }$ where: (1) the factors of X need to be preserved in Xˆ (i.e., high Fidelity), (2) the generated image must minimally depend on the noise $\varepsilon \left( i . e . \right.$ , low Noise Dependance) and (3) $\hat { Z } _ { S }$ and $\hat { Z } _ { C }$ should be independent.

# 4 Methodology

To translate the theoretical requirements of Section 3 into a practical framework, we propose a two-stage training procedure. First, to ensure low Noise Dependance, we must construct the feature extractor $f _ { \theta }$ and generative model $G _ { \psi }$ such that the generated attributes are driven entirely by the conditioning $Z = f _ { \theta } ( X )$ rather than the initial sampling noise ε (Stage 1). While standard diffusion and flow-matching models often rely heavily on the starting noise for structural variability[30, 19, 18], we require the common and salient factors of the generated image to be fully actionable through the conditioning alone. Second, once this actionable latent space is established, we train the encoder network $E _ { \theta _ { E } }$ to enforce the structural conditions on $\hat { Z } _ { S }$ and $\hat { Z } _ { C }$ , ensuring high Fidelity and low Entanglement (Stage 2).

# 4.1 Image-Conditioned Generator Architecture

Standard prompt-based methods inject features into frozen text-to-image backbones, often failing to override the base model’s priors and leading to identity loss or style drift (Fig. 8). Consequently, we train a generative model from scratch using image-only conditioning, derived from DINOv3 [48] embeddings, consisting in K tokens. We adopt a Flow Matching objective, which enforces a deterministic mapping between noise and data, ensuring that visual fidelity is entirely determined by the structure of $\bar { Z }$ rather than stochastic sampling paths. Details on the generator and the Cross-Query (CQ and CQC) encoding of the conditioning Z, which allows low noise dependence for $G _ { \psi } ,$ , can be found in App. D.1.1 and in Fig 3 (Left).

# 4.2 Separator Encoder Architecture and Training

Our CA approach, Diff-CA, leverages the formulation from Sec. 3 and the conditioning pipeline from App. D.1. We implement the encoder $E _ { \theta _ { E } }$ as a small 5-block Transformer encoder. A learnable

CLS token is prepended to the sequence Z. After processing, the CLS token state is projected via a linear layer to form the salient code $\hat { Z } _ { S }$ , while the remaining K output tokens form the common information $\hat { Z } _ { C }$ (Fig. 3 (Right)).

Additive structure. We enforce the sum-constraint $Z = \hat { Z } _ { S } + \hat { Z } _ { C }$ using the reconstruction loss:

$$
\mathcal {L} _ {r e c} = \mathbb {E} \left[ | | Z - (\hat {Z} _ {S} + \hat {Z} _ {C}) | | _ {2} ^ {2} \right] \tag {7}
$$

Pinning. To enforce the pinning condition $( Z _ { S } = 0 )$ for the background distribution $( Y = 0 )$ , we apply a norm-based regularization on $\hat { Z } _ { S }$ . This penalty encourages no (salient) information content for background samples $( Y = 0 )$ while preventing collapse for target samples $( Y = 1 )$ :

$$
\mathcal {L} _ {p i n} = \mathbb {E} \left[ \mathbb {1} _ {Y = 0} | | \hat {Z} _ {S} | | _ {2} ^ {2} + \mathbb {1} _ {Y = 1} e ^ {- | | \hat {Z} _ {S} | | _ {2} ^ {2}} \right] \tag {8}
$$

Cycle Consistency. To enforce the independence condition, we require that the learned factors be interchangeable between pairs of samples with a cycle-consistency loss. We construct a mixed latent code $Z ^ { m i x } = \hat { Z } _ { C } ^ { a } + \hat { Z } _ { S } ^ { b }$ using factors from two independent samples a and b. The encoder is trained to recover the original codes from this mixed sample:

$$
\mathcal {L} _ {c y c} = \mathbb {E} \left[ | | \hat {Z} _ {C} ^ {m i x} - \hat {Z} _ {C} ^ {a} | | _ {2} ^ {2} + | | \hat {Z} _ {S} ^ {m i x} - \hat {Z} _ {S} ^ {b} | | _ {2} ^ {2} ] \right] \tag {9}
$$

where $( \hat { Z } _ { S } ^ { m i x } , \hat { Z } _ { C } ^ { m i x } ) = E _ { \theta _ { E } } ( Z ^ { m i x } )$ . This prevents the estimation of the common code from relying on correlations with a specific salient code, effectively pushing the learned distributions toward independence (see App. C.3). Intuitively, we want swapped counterfactuals to be "valid" samples (Zmi $( Z _ { m i x } \stackrel { L a w } { = } Z )$ x Law= Z). We rely on the separator encoder EθE itself to act as a weak, implicit discriminator. $E _ { \theta _ { E } }$ Specifically, we maintain an exponential moving average (EMA) copy of $E _ { \theta _ { E } }$ , which is used to re-encode swapped counterfactuals $Z _ { \mathrm { m i x } }$ . We detail this heuristic regularization strategy in App. F.

Adversarial Training. To further discourage $\hat { Z } _ { C }$ from being predictive of Y (thus encouraging $\hat { Z } _ { C }$ to encode common information), we employ a Gradient Reversal Layer (GRL) adversarial setup [14]. A discriminator $D _ { a d v }$ is trained to predict the domain Y from the common codes $\hat { Z } _ { C }$ by minimizing the binary cross-entropy:

$$
\mathcal {L} _ {a d v} = - \mathbb {E} \left[ y \log \hat {y} + (1 - y) \log (1 - \hat {y}) \right] \tag {10}
$$

where $\hat { y } = D _ { a d v } ( \hat { Z } _ { C } )$ . To optimize this adversarial loss without instability, we introduce a self-tuning mechanism that dynamically scales the GRL strength to maintain a target discriminator accuracy. We detail this training protocol in App. F.

Total Objective. The total loss is defined as:

$$
\mathcal {L} _ {\text { tot }} = \mathcal {L} _ {\text { rec }} + \lambda_ {1} \mathcal {L} _ {\text { pin }} + \lambda_ {2} \mathcal {L} _ {\text { cycle }} + \lambda_ {3} \mathcal {L} _ {\text { adv }} \tag {11}
$$

The optimization is a min-max game where $D _ { a d \ i }$ minimizes classification error, while the encoder maximizes it (via GRL with strength $\lambda _ { a d v } )$ to remove Y -related information:

$$
\min _ {\theta_ {a d v}} \mathcal {L} _ {a d v} \quad \text { and } \quad \min _ {\theta_ {E}} \left(\mathcal {L} _ {t o t} - \lambda_ {a d v} \mathcal {L} _ {a d v}\right) \tag {12}
$$

# 5 Experiments

Implementation Details. Comprehensive details regarding architectures, hyperparameters, and data preprocessing for all stages are provided in App. D and F.

# 5.1 Empirical Validation of the Conditioning Space

Setup and Baselines. We evaluate our conditioning architecture against several established baselines: DiffAE [42], which utilizes a ResNet to produce a single 512-d conditioning vector; EncDiff [53], which trains a CNN to generate $3 2 \times 1 2 8$ tokens; and projected variants of CLIP and DINOv3. For the latter, we utilize all last-layer activations (197 and 261 tokens) and project each to 128-d before conditioning. All models are trained on FFHQ [27] and tested on CelebA-HQ [33]512 × 512.

![](images/0dd24c9be35947389e51bf8c22532b2ea148a79e4e74615c6dddc0bf076e3794.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["x1"] --> B["DINOv3 Encoder"]
    C["ε"] --> D["u_t"]
    D --> E["Denoising U-Net"]
    F["SHallow CNN"] --> G["Color token t_col"]
    G --> H["(c)"]
    H --> I["Patch tokens"]
    I --> J["(a)"]
    J --> K["Cross-Query Extractor"]
    K --> L["(b)"]
    L --> M["Semantic tokens T"]
    M --> N["(d)"]
    N --> O["Conditioning Z"]
    O --> P["Cross-Attention"]
    P --> Q["(d)"]
    Q --> R["VAE"]
    S["ˆu1"] --> T["VAE"]
    U["ˆx1"] --> V["VAE"]
```
</details>

![](images/5990866c588ce68446a9b38d157f8ebf4dcc860e2c5580d4540e1fc75419b59d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["CLS"] --> B["Z"]
    B --> C["SA Block"]
    C --> D["Transformer Encoder"]
    D --> E["CLS"]
    E --> F["Linear proj"]
    F --> G["K × d"]
    G --> H["+"]
    H --> I["Σ"]
    I --> J["L_recon"]
    J --> K["L_cycle"]
    L["D_adv"] --> M["y"]
    M --> N["L_adv"]
    O["L̂_pin"] --> P["CLS"]
    Q["L̂_C"] --> R["CLS"]
    S["L̂_C"] --> T["CLS"]
    U["L̂_C"] --> V["CLS"]
    W["L̂_C"] --> X["CLS"]
    Y["L̂_C"] --> Z["CLS"]
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#ccf,stroke:#333
    style D fill:#cfc,stroke:#333
    style E fill:#fcc,stroke:#333
    style F fill:#cff,stroke:#333
    style G fill:#ffc,stroke:#333
    style H fill:#fcf,stroke:#333
    style I fill:#cff,stroke:#333
    style J fill:#ffc,stroke:#333
    style K fill:#cfc,stroke:#333
    style L fill:#fcc,stroke:#333
    style M fill:#ffc,stroke:#333
    style N fill:#fcc,stroke:#333
    style O fill:#ffc,stroke:#333
    style P fill:#fcc,stroke:#333
    style Q fill:#fcc,stroke:#333
    style R fill:#fcc,stroke:#333
    style S fill:#fcc,stroke:#333
    style T fill:#fcc,stroke:#333
    style U fill:#fcc,stroke:#333
    style V fill:#fcc,stroke:#333
```
</details>

Figure 3: Our 2-Stage training protocol. Left: Conditioning pipeline. An input image $x _ { 1 }$ is mapped to a latent u0 by a VAE encoder and to DINOv3 features (a). A cross-query module (b) produces semantic tokens T , and a small CNN (c) produces a color token $t _ { \mathrm { c o l } }$ . The concatenated tokens Z condition a U-Net diffusion model via cross-attention (d). Right: Training pipeline of our Common-Salient encoder. A CLS token is prepended to the input tokens. The encoder separates the common and salient parts, whose sum reconstructs the input. An anchor loss $\mathcal { L } _ { a n c h o r }$ ensures that the salient part has no information about background samples, while adversarial training purifies the common part. A cycle-consistency protocol ensures that counterfactuals (swaps) produce meaningful outputs.

Reconstruction 

<table><tr><td>Encoder</td><td>Cond. size</td><td>SSIM ↑</td><td>LPIPS ↓</td><td>FID ↓</td></tr><tr><td>DiffAE</td><td>512</td><td>0.530</td><td>0.209</td><td>14.28</td></tr><tr><td>EncDiff</td><td>32 × 128</td><td>0.420</td><td>0.323</td><td>18.43</td></tr><tr><td>CLIP</td><td>197 × 128</td><td>0.490</td><td>0.201</td><td>7.69</td></tr><tr><td>DINOv3</td><td>261 × 128</td><td>0.645</td><td>0.099</td><td>5.66</td></tr><tr><td>CQ (ours)</td><td>32 × 128</td><td>0.589</td><td>0.124</td><td>5.59</td></tr><tr><td>CQC (ours)</td><td>32 × 128</td><td>0.611</td><td>0.114</td><td>5.52</td></tr></table>

Latent Space Smoothness   
DINOv3   
![](images/ce32e3a428ca1b725145f565c18157aed2cb766308f63cc26d9a59a95b855751.jpg)

<details>
<summary>natural_image</summary>

Grid of 12 facial portraits showing different expressions and angles (no text or symbols)
</details>

CQC (ours)   
Figure 4: Left: Comparison of image reconstruction on CelebA-HQ 256×256. This serves as a sanity check to make sure that the images produced by our model are of good quality. Our methods (CQ and CQC) present competitive reconstructions compared to DINOv3 with fewer tokens, while also being able to carry out editing and image control (which DINOv3 cannot). Right: Exploring the Z-space. We linearly interpolate between $\check { Z }$ codes of two images and generate from random noise at each interpolation point. Our method yields smooth changes in identity, attributes, and color without introducing artifacts.

Reconstruction. To validate the fidelity of our generative model, we evaluate the reconstruction performance of our Cross-Query (CQ) and Color (CQC) encoders (Implementation details in App. D.1.1) against established conditioning approaches (Fig. 4, Left). Our conditioning yields highly competitive reconstruction metrics, despite utilizing a significantly smaller token representation than its strongest baseline, DINOv3. Qualitative results can be found in App. D and E.1

Linear structure of the conditioning space. To evaluate the geometric properties of our learned token manifold, we also perform linear interpolation between the conditioning codes (Z) of two distinct images (Fig. 4, Right). When compared with DINOv3, our conditioning yields drastically smoother, artifact-free transitions, indicating that our conditioning provides a more continuous and well-structured representation while maintaining competitive reconstruction performance.

Table 1: Quantitative comparison on FFHQ (Glasses Attribute). We evaluate reconstruction fidelity and the precision of salient attribute swaps. Swapping metrics assess both the success of the transfer (Acc: glasses class accuracy) and the preservation of unrelated common attributes (Gender, Smile, Pose, ID-Sim E.2) to measure latent leakage. Swapping results are macro-averaged over fine-grained classes (NoGlasses, ReadingGlasses, SunGlasses). Detailed results can be found in Table 8. 

<table><tr><td rowspan="2">Method</td><td colspan="3">Reconstruction</td><td colspan="5">Swapping Fine-grained classes (Glasses)</td></tr><tr><td>SSIM ↑</td><td>LPIPS ↓</td><td>ID-Sim ↑</td><td>Acc ↑</td><td>ID-Sim↑</td><td>Gender (acc) ↑</td><td>Smile (acc) ↑</td><td>Pose (MAE) ↓</td></tr><tr><td>MM-cVAE</td><td>.370</td><td>.643</td><td>.307</td><td>11.8</td><td>.302</td><td>62.3</td><td>57.7</td><td>4.62</td></tr><tr><td>SepVAE</td><td>.358</td><td>.642</td><td>.307</td><td>19.6</td><td>.301</td><td>57.7</td><td>56.4</td><td>8.29</td></tr><tr><td>D.InfoGAN</td><td>.326</td><td>.430</td><td>.327</td><td>47.0</td><td>.314</td><td>68.5</td><td>79.2</td><td>7.35</td></tr><tr><td>Diff-CA (Ours)</td><td>.610</td><td>.115</td><td>.522</td><td>94.5</td><td>.474</td><td>92.5</td><td>92.9</td><td>1.87</td></tr></table>

![](images/2021955e2acd418abd6caf4f53bf8dff01cd0a9928275ab419997ce6f95320e9.jpg)

<details>
<summary>natural_image</summary>

Four close-up portraits of cats in different colors and expressions, set against a blurred natural background (no text or symbols)
</details>

![](images/6744e3c690983a23c30e40b0872c9d6a3aac4aec4b6bbd2f7ee630861474a647.jpg)

<details>
<summary>natural_image</summary>

Four grayscale brain scan images showing axial, sagittal, coronal, and lateral views (no text or labels visible)
</details>

![](images/c04696c1e7df5075319415b526b7b6aad33f863379c7ea55a9da23d8a747b446.jpg)

<details>
<summary>natural_image</summary>

Four portrait photos of smiling individuals wearing sunglasses (no text or symbols visible)
</details>

![](images/1d29ae58a2097c65cc4a8295c08af1efc86e71424b5dfee7a78e956dc98bc9e5.jpg)

<details>
<summary>natural_image</summary>

Four portrait photos of women in different settings (no visible text or symbols)
</details>

Figure 5: For each dataset, the first row represents two real images from background and target while the second row shows swapping results with salient : “cat”, “tumor”, “glasses” and “smile".

# 5.2 Contrastive Decomposition and Swapping

Datasets and Attributes. We evaluate our method across three distinct domains to demonstrate its generality. We use FFHQ [27] and target two salient attributes: Glasses (binary: No Glasses vs. Glasses), Smile (Neutral vs. Smiling). For evaluation only, we use fine-grained labels when possible (Reading vs. Sunglasses) to measure latent structure discovery. On AFHQ [8], we treat species as the salient factor, using a binary Cat vs. Dog setup. BraTS 2023 [28] is a medical imaging benchmark composed of brain MRI data where the salient factor S is the presence of a brain tumor. We use 2D slice images with a slice-level binary (tumor) label as Y .

Baselines. We compare our method against three state-of-the-art CA generative models: MM-cVAE [52], SepVAE [36], and DoubleInfoGAN [6] (our method is the first to use diffusion for CA). As discussed in Sec. 2, these models rely on low-capacity latent bottlenecks. We use their default configurations.

Reconstruction and swapping. As shown in Fig. 16, VAE and GAN-based baselines produce blurred reconstructions and lose identity during swaps. By contrast, Diff-CA not only reconstructs faithfully from $\hat { Z } _ { S } + \hat { Z } _ { C }$ , but also allows for the swapping of $\hat { Z } _ { S }$ between images, transferring the target attribute without altering the rest of the image.

The superiority of our approach in Fig. 16 over the others is further confirmed by the quantitative results in Tab. 1, where Diff-CA substantially outperforms all baselines across every metric.

Beyond significantly better reconstruction performances, the most telling evidence of our decomposition quality lies in the swapping results. Diff-CA achieves 94.5% fine-grained glasses classification accuracy after swap, more than doubling the best baseline (47.0% for DoubleInfoGAN), demonstrating that the salient subspace captures not only coarse binary attributes but is furthermore structured to reflect fine-grained stylistic distinctions despite training under binary supervision only.

Preservation of the common factors. Crucially, this transfer is selective: gender, smile, and head pose are preserved with 92.5%, 92.9%, and 1.87◦ MAE respectively, compared to 68.5%, 79.2%, and $7 . 3 5 ^ { \circ }$ for DoubleInfoGAN. The gap confirms that $\hat { Z } _ { S }$ and $\hat { Z } _ { C }$ are well disentangled, with minimal attribute leakage across subspaces.

![](images/3c3f0905171ccfc663c350121cf81c20522d49b8a30249a04b276495d130d65c.jpg)

$$
Z ^ {m i x} = \hat {Z} _ {C} ^ {a} + (1 - \alpha) \hat {Z} _ {S} ^ {a} + \alpha \hat {Z} _ {S} ^ {b}
$$

Figure 6: Salient interpolation: We fix the common $\hat { Z } _ { C }$ of the left image and interpolate the salient $\hat { Z } _ { S }$ with the salient of the right image. Top: The style of the glasses is progressively transferred. Bottom: cat features (ears, whiskers) progressively appear.   
![](images/c53a7715384e82d81dfe117cdbb1ce17b655f600d4eb1cc424ae3cd31ab9c761.jpg)

![](images/383b308c52b14d5dedf9d32f3ea37e867e00b38a335525552c87fffa6648287d.jpg)

<details>
<summary>scatter</summary>

| Sub-Labels       | UMAP Dimension 1 | UMAP Dimension 2 |
| ---------------- | ---------------- | ---------------- |
| No Glasses       | -5.0             | 3.5              |
| Reading glasses  | 8.0              | 7.0              |
| Sunglasses       | 12.0             | 9.0              |
</details>

Figure 7: PCA and UMAP projection of $\hat { Z } _ { S }$ . Even though $\hat { Z } _ { S }$ was only learned using weak binary supervision, fine-grained subclasses are still separated.

Finally, Fig. 5 shows that this capacity for clean salient separation generalises beyond faces, extending to animal species (AFHQ) and pathological regions (BraTS), which underlines the domain-agnostic nature of our contrastive decomposition.

# 5.3 Latent Space Analysis

Salient interpolation. To evaluate the smoothness and semantic coherence of the learned salient latent space, we perform linear interpolation between salient codes while fixing the common factor. Specifically, given two images a and b, we extract their respective decompositions $\hat { Z } _ { S } ^ { a } , \hat { Z } _ { C } ^ { a }$ and $\hat { Z } _ { S } ^ { b } , \hat { Z } _ { C } ^ { b }$ , then generate images along the interpolation path $G ( \hat { Z } _ { C } ^ { a } + \alpha \hat { Z } _ { S } ^ { a } + ( 1 - \alpha ) \hat { Z } _ { S } ^ { b } , \varepsilon )$ for $\alpha \in [ 0 , 1 ]$ . As shown in Fig. 6, this operation produces semantically meaningful intermediate results. For the glasses attribute (top row), the interpolation smoothly transitions between different styles of eyewear while preserving identity and other facial characteristics. For the cat-dog attribute (bottom row), interpolation reveals a progressive emergence of species-specific features (e.g., ear shape, whiskers) while maintaining pose and background. These results demonstrate that our learned $\bar { Z } _ { S }$ captures a continuous representation of the salient factor, enabling fine-grained control beyond the weak binary supervision used during training.

PCA and UMAP projections. To assess whether the learned salient codes $\hat { Z } _ { S }$ capture fine-grained structure beyond the coarse binary supervision, we analyze their geometric organization using dimensionality reduction. Figure 7 shows PCA and UMAP projections of salient vectors extracted from the weak supervision (Glasses vs. No Glasses). We plot the three fine-grained classes, No Glasses, Reading Glasses, and Sunglasses, in different colors. The three subclasses form clearly distinct clusters in the UMAP projection, indicating that $\hat { Z } _ { S }$ spontaneously learns to separate finegrained variations of the salient attribute without explicit supervision.

# 6 Conclusion and Future Work

We propose a rigorous theoretical framework for Contrastive Analysis in the context of generative models. We proved that under an additive structural assumption with weak binary supervision, the true common and salient factors are uniquely identifiable (Th. 1), and derived a lower bound on controllability based on mutual information that establishes the conditions for effective latent manipulation (Prop. 1).

Building on this theory, we introduced Diff-CA, a novel method that leverages both a high-fidelity editable conditioning space and a practical CA training framework. Our conditioning architecture constructs a latent space that satisfies the structural assumptions required by our identifiability results, while enabling precise semantic control. The encoder network, trained with our proposed losses learns to decompose conditioning tokens into common $( \hat { Z _ { C } } )$ and salient $( \hat { Z } _ { S } )$ factors that can be independently manipulated.

Diff-CA addresses a fundamental limitation of existing CA methods: the trade-off between reconstruction quality and editability. While prior CA approaches sacrifice image fidelity to achieve latent separation, our diffusion-based model maintains both. Empirically, Diff-CA achieves state-of-theart reconstruction fidelity across multiple domains while simultaneously delivering state-of-the-art editing performance.

Our current implementation operates on 2D images and is not adapted to fully 3D volumetric data, such as high-resolution medical volumes, which we leave for future work. In addition, we focus on an additive factorization of common and salient components. Exploring alternative factorizations, as well as settings where salient factors appear in both distributions, is an important future research direction.

# Acknowledgments

We acknowledge the support of the French Agence Nationale de la Recherche (ANR) under reference ANR-21-CE23- 0024 IDEGEN, ANR-19-CE40-005 MISTIC and EUR BERTIP (ANR-18-EURE-0002). This work was performed using HPC resources from GENCI–IDRIS (A0160615058).

# References

[1] Abubakar Abid and James Zou. Contrastive variational autoencoder enhances salient features. arXiv preprint arXiv:1902.04601, 2019.   
[2] Abubakar Abid, Martin J Zhang, Vivek K Bagaria, and James Zou. Exploring patterns enriched in a dataset with contrastive principal component analysis. Nature communications, 9(1):2134, 2018.   
[3] Sagie Benaim, Michael Khaitov, Tomer Galanti, and Lior Wolf. Domain intersection and domain difference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 3453–3462, 2019.   
[4] Konstantinos Bousmalis, George Trigeorgis, Nathan Silberman, Dilip Krishnan, and Dumitru Erhan. Domain separation networks. In Advances in Neural Information Processing Systems, 2016.   
[5] Tim Brooks, Aleksander Holynski, and Alexei A Efros. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 18392–18402, 2023.   
[6] Florence Carton, Robin Louiset, and Pietro Gori. Double infogan for contrastive analysis. In International Conference on Artificial Intelligence and Statistics, pages 172–180. PMLR, 2024.

[7] Xi Chen, Yan Duan, Rein Houthooft, John Schulman, Ilya Sutskever, and Pieter Abbeel. Infogan: Interpretable representation learning by information maximizing generative adversarial nets. Advances in neural information processing systems, 29, 2016.   
[8] Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. Stargan v2: Diverse image synthesis for multiple domains. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2020.   
[9] Jiankang Deng, Jia Guo, Niannan Xue, and Stefanos Zafeiriou. Arcface: Additive angular margin loss for deep face recognition. In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4685–4694, 2019. doi: 10.1109/CVPR.2019.00482.   
[10] Jiankang Deng, Jia Guo, Debing Zhang, Yafeng Deng, Xiangju Lu, and Song Shi. Lightweight face recognition challenge. In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pages 2638–2646, 2019. doi: 10.1109/ICCVW.2019.00322.   
[11] Benoit Dufumier, Carlo Alberto Barbano, Robin Louiset, Edouard Duchesnay, and Pietro Gori. Integrating Prior Knowledge in Contrastive Learning with Kernel. In International Conference on Machine Learning (ICML), 2023.   
[12] Benoit Dufumier, Javiera Castillo-Navarro, Devis Tuia, and Jean-Philippe Thiran. What to align in multimodal contrastive learning? In International Conference on Learning Representations (ICLR), 2025.   
[13] Marco Federici, Anjan Dutta, Patrick Forré, Nate Kushman, and Zeynep Akata. Learning robust representations via multi-view information bottleneck. In International Conference on Learning Representations, 2020.   
[14] Yaroslav Ganin and Victor Lempitsky. Unsupervised domain adaptation by backpropagation. In International conference on machine learning, pages 1180–1189. PMLR, 2015.   
[15] Abel Gonzalez-Garcia, Joost van de Weijer, and Yoshua Bengio. Image-to-image translation for crossdomain disentanglement. In Advances in Neural Information Processing Systems, 2018.   
[16] Google DeepMind. Nano banana pro: Advanced visual reasoning and editing with gemini 3 pro image. Technical report, Google, November 2025. URL https://ai.google.dev/gemini-api/docs/ image-generation.   
[17] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.   
[18] Paul Grimal, Michaël Soumm, Hervé Le Borgne, Olivier Ferret, and Akihiro Sugimoto. Saga: Learning signal-aligned distributions for improved text-to-image generation, 2025. URL https://arxiv.org/ abs/2508.13866.   
[19] Xiefan Guo, Jinlin Liu, Miaomiao Cui, Jiankai Li, Hongyu Yang, and Di Huang. Initno: Boosting text-to-image diffusion models via initial noise optimization. In CVPR, 2024.   
[20] Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.   
[21] Yunlong He, Gwilherm Lesné, Ziqian Liu, Michaël Soumm, and Pietro Gori. Learning Common and Salient Generative Factors Between Two Image Datasets, 2025.   
[22] Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, and Daniel Cohen-or. Prompt-toprompt image editing with cross-attention control. In The Eleventh International Conference on Learning Representations, 2023.   
[23] Irina Higgins, Loic Matthey, Arka Pal, Christopher Burgess, Xavier Glorot, Matthew Botvinick, Shakir Mohamed, and Alexander Lerchner. beta-vae: Learning basic visual concepts with a constrained variational framework. In International conference on learning representations, 2017.   
[24] Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising diffusion probabilistic models. Advances in neural information processing systems, 33:6840–6851, 2020.   
[25] Andrew Jaegle, Sebastian Borgeaud, Jean-Baptiste Alayrac, Carl Doersch, Catalin Ionescu, David Ding, Skanda Koppula, Daniel Zoran, Andrew Brock, Evan Shelhamer, et al. Perceiver io: A general architecture for structured inputs & outputs. arXiv preprint arXiv:2107.14795, 2021.

[26] Jaeseok Jeong, Mingi Kwon, and Youngjung Uh. Training-free content injection using h-space in diffusion models. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision, pages 5151–5161, 2024.   
[27] Tero Karras, Samuli Laine, and Timo Aila. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 4401–4410, 2019.   
[28] Anahita Fathi Kazerooni, Nastaran Khalili, Xinyang Liu, Debanjan Haldar, Zhifan Jiang, Syed Muhammed Anwar, Jake Albrecht, Maruf Adewole, Udunna Anazodo, Hannah Anderson, et al. The brain tumor segmentation (brats) challenge 2023: Focus on pediatrics (cbtn-connect-dipgr-asnr-miccai brats-peds). ArXiv, pages arXiv–2305, 2024.   
[29] Michael Kleinman, Alessandro Achille, Stefano Soatto, and Jonathan C. Kao. Gács–körner common information variational autoencoder. In Advances in Neural Information Processing Systems, 2023.   
[30] Mingi Kwon, Jaeseok Jeong, and Youngjung Uh. Diffusion models already have a semantic latent space. In The Eleventh International Conference on Learning Representations, 2023.   
[31] Seunghun Lee, Sunghyun Cho, and Sunghoon Im. Dranet: Disentangling representation and adaptation networks for unsupervised cross-domain adaptation. In CVPR, pages 15252–15261, 2021.   
[32] Yaron Lipman, Ricky TQ Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.   
[33] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. Deep learning face attributes in the wild. In Proceedings of International Conference on Computer Vision (ICCV), 2015.   
[34] Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch, Sylvain Gelly, Bernhard Schölkopf, and Olivier Bachem. Challenging common assumptions in the unsupervised learning of disentangled representations. In international conference on machine learning, pages 4114–4124. PMLR, 2019.   
[35] Robin Louiset, Pietro Gori, Benoit Dufumier, Josselin Houenou, Antoine Grigis, and Edouard Duchesnay. UCSL : A Machine Learning Expectation-Maximization Framework for Unsupervised Clustering Driven by Supervised Learning. In Machine Learning and Knowledge Discovery in Databases. Research Track, 2021.   
[36] Robin Louiset, Edouard Duchesnay, Grigis Antoine, Benoit Dufumier, and Pietro Gori. Sepvae: a contrastive vae to separate pathological patterns from healthy ones. In Ninon Burgos, Caroline Petitjean, Maria Vakalopoulou, Stergios Christodoulidis, Pierrick Coupe, Hervé Delingette, Carole Lartizien, and Diana Mateus, editors, Proceedings of The 7nd International Conference on Medical Imaging with Deep Learning, volume 250 of Proceedings of Machine Learning Research, pages 918–936. PMLR, 03–05 Jul 2024. URL https://proceedings.mlr.press/v250/louiset24a.html.   
[37] Robin Louiset, Edouard Duchesnay, Antoine Grigis, and Pietro Gori. Separating common from salient patterns with contrastive representation learning. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum?id=30N3bNAiw3.   
[38] Robin Louiset, Edouard Duchesnay, Benoit Dufumier, Antoine Grigis, and Pietro Gori. Automatic Discovery of Disease Subgroups by Contrasting with Healthy Controls. Data Mining and Knowledge Discovery, 2026.   
[39] Ron Mokady, Amir Hertz, Kfir Aberman, Yael Pritch, and Daniel Cohen-Or. Null-text inversion for editing real images using guided diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 6038–6047, 2023.   
[40] Yong-Hyun Park, Mingi Kwon, Jaewoong Choi, Junghyo Jo, and Youngjung Uh. Understanding the latent space of diffusion models through the lens of riemannian geometry. Advances in Neural Information Processing Systems, 36:24129–24142, 2023.   
[41] Gaurav Parmar, Krishna Kumar Singh, Richard Zhang, Yijun Li, Jingwan Lu, and Jun-Yan Zhu. Zero-shot image-to-image translation. In ACM SIGGRAPH 2023 conference proceedings, pages 1–11, 2023.   
[42] Konpat Preechakul, Nattanat Chatthee, Suttisak Wizadwongsa, and Supasorn Suwajanakorn. Diffusion autoencoders: Toward a meaningful and decodable representation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10619–10629, 2022.

[43] Hugo Richard, Pierre Ablin, Bertrand Thirion, Alexandre Gramfort, and Aapo Hyvarinen. Shared Independent Component Analysis for Multi-Subject Neuroimaging. In NeurIPS, volume 34, pages 29962–29971, 2021.   
[44] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. High-resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 10684–10695, 2022.   
[45] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. Dreambooth: Fine tuning text-to-image diffusion models for subject-driven generation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 22500–22510, 2023.   
[46] Eduardo Hugo Sánchez, Mathieu Serrurier, and Mathias Ortner. Learning disentangled representations via mutual information estimation. In European Conference on Computer Vision (ECCV), 2020.   
[47] Ka Chun Shum, Binh-Son Hua, Duc Thanh Nguyen, and Sai-Kit Yeung. Color alignment in diffusion. In Proceedings of the Computer Vision and Pattern Recognition Conference, pages 28446–28455, 2025.   
[48] Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.   
[49] Yonglong Tian, Dilip Krishnan, and Phillip Isola. Contrastive multiview coding. In ECCV, pages 776–794, 2020.   
[50] Alexander Tong, Kilian FATRAS, Nikolay Malkin, Guillaume Huguet, Yanlei Zhang, Jarrid Rector-Brooks, Guy Wolf, and Yoshua Bengio. Improving and generalizing flow-based generative models with minibatch optimal transport. Transactions on Machine Learning Research, 2024. ISSN 2835-8856.   
[51] Weiran Wang, Xinchen Yan, Honglak Lee, and Karen Livescu. Deep variational canonical correlation analysis. In International Conference on Learning Representations (ICLR), 2017.   
[52] Ethan Weinberger, Nicasia Beebe-Wang, and Su-In Lee. Moment matching deep contrastive latent variable models. arXiv preprint arXiv:2202.10560, 2022.   
[53] Tao Yang, Cuiling Lan, Yan Lu, and Nanning Zheng. Diffusion model with cross attention as an inductive bias for disentanglement. Advances in Neural Information Processing Systems, 37:82465–82492, 2024.   
[54] Hu Ye, Jun Zhang, Sibo Liu, Xiao Han, and Wei Yang. Ip-adapter: Text compatible image prompt adapter for text-to-image diffusion models. arXiv preprint arXiv:2308.06721, 2023.   
[55] Junyu Zhang, Daochang Liu, Eunbyung Park, Shichao Zhang, and Chang Xu. Anti-exposure bias in diffusion models. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=MtDd7rWok1.   
[56] Lvmin Zhang, Anyi Rao, and Maneesh Agrawala. Adding conditional control to text-to-image diffusion models. In Proceedings of the IEEE/CVF international conference on computer vision, pages 3836–3847, 2023.   
[57] Boyang Zheng, Nanye Ma, Shengbang Tong, and Saining Xie. Diffusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690, 2025.

# Appendix A: Broader Impact

Our method has the potential to significantly impact domains such as medical imaging, where diffusion-based models combined with contrastive analysis could help identify subtle imaging patterns associated with specific pathologies that may not be visible to the naked eye of a clinician. This could lead to improved understanding of disease mechanisms and enable the discovery of novel, non-invasive imaging biomarkers. More broadly, structuring the representation space through contrastive objectives may enhance the trustworthiness and explainability of diffusion-based models for generation and editing by disentangling common and salient factors, thereby facilitating the identification of hidden biases or spurious shortcuts. However, these benefits must be balanced against potential risks: although Diff-CA primarily aims at enhancing explainability, improved generative and editing capabilities may still enable the creation of highly realistic synthetic content, raising concerns about deepfakes and misinformation. Furthermore, if not carefully designed, particularly in dataset construction, contrastive objectives may inadvertently reinforce existing biases or induce misleading feature separations, thereby undermining the intended gains in robustness and interpretability. Finally, AI decisions can affect an uncautious expert’s final decision, which could have dramatic consequences in fields such as medicine. The use of contrastive analysis (CA) and, more broadly, artificial intelligence should be conducted with all necessary critical thinking and should NOT replace, but rather support, the expert’s work or analysis.

# Appendix B: Additional Results and Discussion on T2I Models

![](images/62343c70ccba90dc1ddd47645980e32e92afdb05e051344d2dc88153bd9e0bef.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["FLUX.1 Kontext [dev"]] --> B["Nano Banana Pro"]
    B --> C["tumor removal"]
    C --> D["Nano Banana Pro swap of pose"]
```
</details>

Figure 8: Failure cases of some prompt-based diffusion editing methods. Left: FLUX.1 Kontext adds the same generic sunglasses to everyone. Middle: Nano Banana Pro correctly removes the tumor but fails to generate an anatomically realistic image (the generated brain gyrification is not realistic), and it alters the healthy anatomy (ventricles in the bottom image). Right: Nano Banana Pro fails to swap head position without altering other attributes (e.g., background, hair, colors).

Text-to-image (T2I) methods have seen massive adoption for general-purpose image editing, yet they exhibit critical structural limitations when precise, controlled, or out-of-domain operations are required. We discuss four such limitations that are particularly relevant to the CA setting.

Output randomization. Across repeated runs with identical inputs and prompts, T2I models produce meaningfully different outputs. On medical images, for instance, repeated queries yield varying degrees of lesion removal, inconsistent tissue intensities, and unstable anatomical geometry. In industrial and medical contexts, reproducibility is a hard requirement for validation and regulatory compliance. A method that produces different results across runs cannot be audited or certified regardless of its average visual quality.

Distributional implausibility. Even visually plausible outputs can be clinically or scientifically incorrect, a failure mode that is silent and therefore particularly dangerous. In our MRI experiments, prompted baselines and our own replications produce brain ventricle and gyrification geometries inconsistent with healthy anatomy, yet these outputs appear reasonable to a non-expert. In any domain requiring distributional conformity, outputs that fall outside the target distribution are more problematic than obvious failures. By performing exact latent swapping within a model trained on the target distribution, our approach preserves individual anatomy by construction.

Prompt sensitivity. The quality of T2I outputs is highly sensitive to the precise wording of the prompt, a choice entirely decoupled from the actual content of interest. This uncontrolled variable makes deployment impractical in settings where the salient concept is not easily verbalized, or where prompt standardization across users and institutions cannot be guaranteed. CA requires no such verbalization, which is precisely its motivation.

Single-image guided editing   
Real   
![](images/b6ef408923bc7cb014457d1af00f99688648a9287239c1fa21e1d8420ba26668.jpg)

Generated   
![](images/f5084b30fda97f0274fe4bbc0bddff2f0ad29ce3ec131aefbca81dc55654f48a.jpg)

remove the tumor from the brain MRlimage   
![](images/822e671c6cf9acafec85dcd6f5a51f501ac8d4049c34766c8992e1ce76e2aab6.jpg)

![](images/508973a44e7667636025b477e4731dfbc84d4b576793b363b2adbd328eddf13e.jpg)  
Remove the tumor from the brain MRl image andgenerate an anatomically realistic healthy image,modifying the original image as little as possible.

![](images/dc74b13ff771704c0ddaea903d78169cb8456aee8018ae044054fb1557f36eb5.jpg)

![](images/928a179355e1b6d4844f482fea49d260d594552a65e427a75b74fb8d4126cebd.jpg)  
adda tumorto this image in a realistic way

Multi-image guided editing   
Real   
![](images/c17469b261bd75d44cd35114debc87f9666c1d06e2a8e4940977355c037e39cd.jpg)

![](images/52b82edfd17545ab87f56a25413dd202095051a9df40166caa63444804be9bc4.jpg)  
estimate the salient and common patterns between the two images，exchange only the salient patterns between teh two images and generate the two images   
Generated

![](images/9ac835548927791430eba79d18d468d1e9c7a9cd18defa5ce95c5b3f4d7d1b40.jpg)

![](images/bea6d58a862ec18683d35a77b506d03eac8f2be15b85c16d0ea25400e0bc49e0.jpg)

![](images/b033d78ed342554f9c522eff57dec029903a892f3f85b35c2197909d0a0e560f.jpg)

![](images/1f956d7a6e4d2471897928fec64ce40796abf72f1edadb771b7a6a091b349bd4.jpg)

![](images/aece48a41d45b29361045f158a8028bc51406de5835df9464c054ab20b538af8.jpg)

<details>
<summary>natural_image</summary>

Two axial brain MRI scans showing ventricular structures (no text or labels visible)
</details>

![](images/63b5df6e0db9b19e317fc70a2ddffce8f308480cecbf879e424859cf94febe51.jpg)

![](images/be6b071cfad446721bbc6b93c15764e421f8c5ada5bbb55cd246ecaee3a08dee.jpg)  
Figure 9: Failure cases of Nano Banana 2 (Gemini 3 Flash Image). Left: We use a single real image and a prompt to guide the editing. All generated images are either anatomically unrealistic or show altered anatomy (orange arrows). Right: Two real images (without and with a tumor) are provided as input, along with a prompt to guide the editing. This setup more closely resembles the goal of Contrastive Analysis (CA), although CA is a population-based statistical method rather than an image editing approach based on two input images. The different pairs of generated images correspond to outputs generated sequentially (one after the other) using the same prompt, with the addition of the instruction “the previous images are wrong”. This shows that the results can vary depending on the prompt (low robustness) and that, even with corrective prompts, the model fails to produce a convincing solution. Indeed, all generated images exhibit differences in orientation and anatomical structures compared to the original images (e.g., skull, gyrification, ventricles, shape), and in some cases the tumor (i.e., the salient feature) is not correctly added or removed.

Domain-agnostic safety filters. General-purpose generators apply content filters that can interfere unpredictably with scientific use cases. In our experiments, pathology re-introduction (a natural reversibility check) was refused by the model in several runs. Models trained specifically for a target domain and task, as in our framework, operate without such constraints.

The most widely used commercial models include Nano Banana Pro/2 [16] and ChatGPT, which rely on in-context information and can technically use multiple images alongside a text prompt to perform edits and swapping. In Fig. 8, Fig. 9 and Fig. 10, we illustrate failure cases for both single-image and multi-image guided editing, highlighting the importance of specialized models for critical tasks, such as medical imaging.

"Take the head position of the first image and apply it to the second one. Only change the angle, everything else should stay the same."

![](images/0734472d8d049a5314665eb54cbc9991767bcd5dd6b1c897a62f6eb6ff52d767.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a young child with curly hair, wearing a blue top (no text or symbols visible)
</details>

![](images/40f6be709cc5bfa14ce2ef55c777efcdeaa97483e7567d33b3a05c160758682b.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a smiling woman with dark hair, outdoors with trees and buildings in background (no visible text or symbols)
</details>

Nano → Banana Pro   
![](images/ee50be0556a30239575ad9a0ca11a1a31f9081ecf4b2ea234574389e5320b19f.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a smiling woman with dark hair tied back, outdoors with blurred background (no text or symbols visible)
</details>

"Take the head position of the first image and apply it to the second one."

![](images/c42d7621d5f4e11cb8ac4a8ed7a98f105cec8040c2189480b9c5f239e4e99069.jpg)

![](images/85bec6d2449377223f97f5e5a34577b6f0abace61cb1bb2ab95fbb7f6913471d.jpg)

Nano Banana Pro   
![](images/f6ace2b02aae3b3cbb218d108888445a0b1048acb9277ae7a5a459a0a1f3579f.jpg)

<details>
<summary>natural_image</summary>

Close-up portrait of a smiling woman with brown hair tied back, outdoors with blurred greenery background (no text or symbols visible)
</details>

![](images/ef03a0e4be2e9ca020472b4814ee3630a2cbb93697259f49197532c0cdc197d7.jpg)

![](images/730e1de8fa12208dabccc89abcc4ec235575309ec01c2b749483c6b0317115a7.jpg)

OURS(Swap)  
![](images/33c36d6be6c62ccf2972893c76e2a08a65fa5321ea40e6b4f962094eb9066410.jpg)

![](images/ca0df01eff77aee7105521839c54a22ba24cf8c7d20471334dd229bab7b25973.jpg)  
Figure 10: Failure cases of Nano Banana Pro for a head position transfer for two different prompts, and our swapping model.

# Appendix C: Theory

# C.1 Proof of Theorem 1

We prove the following result:

Theorem 1 (Identifiability of Additive Factors). Assume the conditioning Z is generated by the direct sum of independent factors $Z _ { S }$ and $Z _ { C } ,$ , where $Z _ { S }$ vanishes for background samples $( Y = 0 )$ . If a learned decomposition $Z \mapsto ( \hat { Z } _ { S } , \hat { Z } _ { C } )$ satisfies:

1. Additive Reconstruction: The learned codes sum to the input: $\hat { Z } _ { S } + \hat { Z } _ { C } = Z$   
2. Pinning: The salient code vanishes only for background samples: $Y = 0 \Longleftrightarrow \hat { Z } _ { S } = 0 .$   
3. Independence: The learned codes are independent: $\hat { Z } _ { S } \perp \perp \hat { Z } _ { C } .$

Then, the learned codes uniquely identify the true encoded factors:

$$
\hat {Z} _ {S} = Z _ {S} \quad a n d \quad \hat {Z} _ {C} = Z _ {C} \tag {13}
$$

To provide a rigorous proof, we restate the conditions from the main text as formal hypotheses:

H1 (Latent Direct Sum): $Z = Z _ { S } + Z _ { C }$ , where $Z _ { S } \in { \cal S } , Z _ { C } \in { \mathcal { C } }$ , and $\mathbb { R } ^ { d } = \pmb { S } \oplus \pmb { \mathcal { C } }$ .

H2 (Latent Independence): $Z _ { S } \perp \perp Z _ { C }$ and $\mathbb { P } ( Z _ { S } = 0 ) > 0 .$ .

H3 (Code Decomposition): The latent codes additively reconstruct Z: $Z = \hat { Z } _ { S } + \hat { Z } _ { C }$ .

H4 (Code Independence): The learned factors are independent: $\hat { Z } _ { S } \perp \perp \hat { Z } _ { C }$ .

H5 (Pinning): The absence of the attribute in the signal implies a zero code: $\hat { Z } _ { S } = 0 \iff$ $Z _ { S } = 0 .$ D.

Proof. The proof is in two steps. In the first step, we identify the marginal laws of $\hat { Z } _ { S }$ and $\hat { Z } _ { C }$ as the laws of $Z _ { S }$ and $Z _ { C }$ , respectively. In the second step, we extend the equality in law to equality almost surely, by showing that the leakage variable $L : = \hat { Z } _ { S } - Z _ { S }$ is almost surely zero. Note that by the additive structure (H3), we have $\hat { Z } _ { C } - Z _ { C } = - L { \mathrm { a . s } }$ .

# Step 1: Identification of marginals.

We first establish that the learned factors match the true marginal distributions.

Consider the event $\{ Y = 0 \}$ . By the pinning assumption H5, $\hat { Z } _ { S } = 0$ . Consequently, conditioned on $Y = 0 .$ , we have $Z = \hat { Z } _ { C }$ .

By the data generation assumption, $Y = 0 \implies Z _ { S } = 0$ , so, conditioned on $Y = 0$ , we have $Z = Z _ { C }$ .

Thus, conditioned on $Y = 0$ , we have $\hat { Z } _ { C } = Z _ { C }$ . We now generalize to the unconditional result.

• By ${ \bf H } 4 , \hat { Z } _ { C } \perp \perp \hat { Z } _ { S }$ . Since $Y = \mathbb { 1 } \{ \hat { Z } _ { S } \neq 0 \}$ is a function of $\hat { Z } _ { S } .$ , we have $\hat { Z } _ { C } \perp \perp Y$ .   
• By $\mathbf { H } 2 , Z _ { C } \perp \perp Z _ { S }$ . Since $Y = \mathbb { 1 } \{ S \neq 0 \}$ is function of S, we have $Z _ { C } \perp \perp Y$ .

Since $\hat { Z } _ { C } = Z _ { C } \mathrm { o n } \left\{ Y = 0 \right\}$ , the two previous points allow us to extend the equality in law to all values of Y :

$$
\hat {Z} _ {C} \stackrel {\text { Law }} {=} Z _ {C} \tag {14}
$$

We know show equality of the marginal laws of $Z _ { S }$ and $\hat { Z } _ { S }$ .

We denote by $\phi _ { X } ( t ) = \mathbb { E } [ e ^ { i \langle X ; t \rangle } ]$ the characteristic function of a variable X. Since $Z = \hat { Z } _ { S } + \hat { Z } _ { C } =$ $Z _ { S } + Z _ { C }$ involves sums of independent variables, the characteristic functions satisfy $\phi _ { \hat { Z } _ { S } } \phi _ { \hat { Z } _ { C } } =$ $\phi _ { Z _ { S } } \phi _ { Z _ { C } }$ .

A characteristic function is continuous on R and takes the value 1 at 0. Therefore, all characteristic functions are non-zero around 0. Thus, on a neighborhood U around 0, we have:

$$
\forall t \in U, \phi_ {\hat {Z} _ {S}} (t) = \phi_ {Z _ {S}} (t) \tag {15}
$$

Assuming the encoded attributes possess a moment-generating function (satisfied by the bounded DINO-token space), the characteristic functions are analytic, and the identity extends to $\mathbb { R } ^ { d }$ by the identity theorem for analytic functions. Thus, we have:

$$
\hat {Z} _ {S} \stackrel {\text { Law }} {=} Z _ {S} \tag {16}
$$

Step 2: Vanishing leakage via gradients. We leverage the independence of the learned codes to constrain the leakage variable $L = \hat { Z } _ { S } - Z _ { S }$ . We will first show that it’s zero a.s. conditioned on Z, then show it’s zero a.s. unconditionnaly.

By independence H4, for all $u , v \in \mathbb { R } ^ { d } ;$

$$
\mathbb {E} \left[ e ^ {i \langle u, \hat {Z} _ {S} \rangle + i \langle v, \hat {Z} _ {C} \rangle} \right] = \phi_ {\hat {Z} _ {S}} (u) \phi_ {\hat {Z} _ {C}} (v) \tag {17}
$$

Substituting the relations $\hat { Z } _ { S } = Z _ { S } + L$ and $\hat { Z } _ { C } = Z _ { C } - L$ on the LHS, and using the marginal identification from Step 1 to replace the RHS characteristic functions with those of the true factors, we get:

$$
\mathbb {E} \left[ e ^ {i \langle u, Z _ {S} \rangle + i \langle v, Z _ {C} \rangle} e ^ {i \langle u - v, L \rangle} \right] = \phi_ {Z _ {S}} (u) \phi_ {Z _ {C}} (v) \tag {18}
$$

We differentiate both sides with respect to u. Assuming finite first moments, we apply the gradient $\nabla _ { u }$ under the expectation:

$$
\begin{array}{l} \nabla_ {u} \mathrm{LHS} = \nabla_ {u} \mathbb {E} \left[ e ^ {i \langle u, Z _ {S} \rangle + i \langle v, Z _ {C} \rangle} e ^ {i \langle u - v, L \rangle} \right] \\ = \mathbb {E} \left[ \nabla_ {u} e ^ {i \langle u, Z _ {S} \rangle + i \langle v, Z _ {C} \rangle} e ^ {i \langle u - v, L \rangle} \right] \\ = \mathbb {E} \left[ (i Z _ {S} + i L) e ^ {i \langle u, Z _ {S} \rangle + i \langle v, Z _ {C} \rangle} e ^ {i \langle u - v, L \rangle} \right] \\ = i \mathbb {E} \left[ (Z _ {S} + L) e ^ {i \langle u, Z _ {S} \rangle + i \langle v, Z _ {C} \rangle} e ^ {i \langle u - v, L \rangle} \right] \\ \end{array}
$$

$$
\nabla_ {u} \mathrm{RHS} = \nabla_ {u} \left[ \phi_ {Z _ {S}} (u) \phi_ {Z _ {C}} (v) \right]
$$

$$
= \nabla_ {u} \left[ \phi_ {Z _ {S}} (u) \right] \phi_ {Z _ {C}} (v)
$$

$$
= \mathbb {E} [ i Z _ {S} e ^ {i \langle u, Z _ {S} \rangle} ] \phi_ {Z _ {C}} (v)
$$

$$
= i \mathbb {E} [ i Z _ {S} e ^ {\langle u, Z _ {S} \rangle} ] \phi_ {Z _ {C}} (v)
$$

Dividing by i on both sides we get:

$$
\mathbb {E} \left[ \left(Z _ {S} + L\right) e ^ {i \langle u, Z _ {S} \rangle + i \langle v, Z _ {C} \rangle} e ^ {i \langle u - v, L \rangle} \right] \tag {19}
$$

$$
= \mathbb {E} [ Z _ {S} e ^ {\langle u, Z _ {S} \rangle} ] \phi_ {Z _ {C}} (v) \tag {20}
$$

We evaluate this equation on the hyperplane $u = v = t$ . The leakage term in the exponent vanishes $( e ^ { i \langle 0 , L \rangle } = 1 )$ , yielding:

$$
\mathbb {E} \left[ (Z _ {S} + L) e ^ {i \langle t, Z _ {S} + Z _ {C} \rangle} \right] = \mathbb {E} [ Z _ {S} e ^ {i \langle t, Z _ {S} \rangle} ] \phi_ {Z _ {C}} (t) \tag {21}
$$

Using the independence $Z _ { S } \perp \perp Z _ { C }$ , the RHS can be rewritten as $\mathbb { E } [ Z _ { S } e ^ { i \langle t , Z _ { S } \rangle } e ^ { i \langle t , Z _ { C } \rangle } ]$ . Subtracting the RHS from the LHS in (21), we obtain:

$$
\mathbb {E} \left[ \left(Z _ {S} + L\right) e ^ {i \langle t, Z _ {S} + Z _ {C} \rangle} - Z _ {S} e ^ {i \langle t, Z _ {S} \rangle} e ^ {i \langle t, Z _ {C} \rangle} \right] = 0 \tag {22}
$$

The terms $Z _ { S } e ^ { i \left. t , Z _ { S } \right. } e ^ { i \left. t , Z _ { C } \right. }$ cancel out, leaving:

$$
\forall t \in \mathbb {R} ^ {d}, \quad \mathbb {E} \left[ L e ^ {i \langle t, Z _ {S} + Z _ {C} \rangle} \right] = 0 \tag {23}
$$

$$
\mathbb {E} \left[ L e ^ {i \langle t, Z \rangle} \right] = 0 \tag {24}
$$

$$
\mathbb {E} \left[ \mathbb {E} \left[ L e ^ {i \langle t, Z \rangle} | Z \right] \right] = 0 \tag {25}
$$

$$
\int_ {\mathbb {R} ^ {d}} \mathbb {E} \left[ L e ^ {i \langle t, z \rangle} | Z = z \right] d \mathbb {P} _ {Z} (z) = 0 \tag {26}
$$

In the integral above, $d \mathbb { P } _ { Z } ( z )$ is the probability measure of Z. We define a new signed measure $\mu$ such that $\bar { d \mu } ( z ) = \mathbb { E } [ L \mid Z = z ] d \mathbb { P } _ { Z } ( z )$ .

In this case,

$$
\int e ^ {i \langle t, z \rangle} d \mu (z) = 0 \tag {27}
$$

is the Fourier-Stieltjes transform of the measure $\mu .$ By the uniqueness of the Fourier transform, if the transform is zero for all $t ,$ then the measure $\mu$ must be zero everywhere.

For $\mu$ to be the zero measure, its density with respect to $\mathbb { P } _ { Z }$ must vanish for $\mathbb { P } _ { Z }$ -almost every z:

$$
\mathbb {E} [ L \mid Z ] = 0 \quad \mathbb {P} _ {Z} - \text { almost   surely }. \tag {28}
$$

Because $\hat { Z } _ { S }$ is produced by a deterministic encoder $E _ { \theta _ { E } } ( Z )$ and $Z _ { S }$ is a deterministic projection of $Z$ , the difference $L = \hat { Z } _ { S } - Z _ { S }$ is perfectly known once Z is known (i.e., it is $\sigma ( Z )$ -measurable).

Therefore, $\mathbb { E } [ L \mid Z ] = L { \mathrm { a . s } }$ .

Combined with the Fourier result $\mathbb { E } [ L \mid Z ] = 0 .$ , we conclude L = 0 almost surely. Therefore, we have the equality a.s. between the true factors and the encoded factors. □

# C.2 Proof of Proposition 1

Assume that the generator $G _ { \psi }$ is deterministic given $( Z , \varepsilon )$ , and let $\hat { X } = G _ { \psi } ( Z , \varepsilon )$ with corresponding attributes $\hat { S }$ and $\hat { C }$ . Then for any $( \hat { Z } _ { S } , \hat { Z } _ { C } )$ ) that are functions of Z, the control over the salient attribute is lower-bounded by:

$$
I (\hat {S}; \hat {Z} _ {S}) \geq \underbrace {I (S ; \hat {S})} _ {\text { Fidelity }} - \underbrace {I (\hat {S} ; \varepsilon | Z)} _ {\text { Noise   dependence }} - \underbrace {I (S ; \hat {Z} _ {C} | \hat {Z} _ {S})} _ {\text { Entanglement }} \tag {29}
$$

Similarly, for the common context:

$$
I (\hat {C}; \hat {Z} _ {C}) \geq I (C; \hat {C}) - I (\hat {C}; \varepsilon | Z) - I (C; \hat {Z} _ {S} | \hat {Z} _ {C}) \tag {30}
$$

Proof. We prove the result for $S$ and $Z _ { C }$ . The proof for C and $Z _ { C }$ is analogous.

Step 1: Decomposition. We start with the identity $I ( \hat { S } ; \hat { Z } _ { S } ) = H ( \hat { S } ) - H ( \hat { S } | \hat { Z } _ { S } )$ . We first decompose the conditional entropy term. Using the definition of conditional mutual information, we write:

$$
H (\hat {S} | \hat {Z} _ {S}) = I (\hat {S}; \hat {Z} _ {C} | \hat {Z} _ {S}) + H (\hat {S} | \hat {Z} _ {S}, \hat {Z} _ {C}). \tag {31}
$$

Since the pair $( \hat { Z } _ { S } , \hat { Z } _ { C } )$ uniquely determines the conditioning $Z ,$ we have $H ( \hat { S } | \hat { Z } _ { S } , \hat { Z } _ { C } ) = H ( \hat { S } | Z )$ . Furthermore, because $\hat { S }$ is determined by $Z$ and ε, any remaining entropy in $\hat { S }$ given $Z$ is attributable entirely to the noise ε:

$$
H (\hat {S} | Z) = I (\hat {S}; \varepsilon | Z) + H (\hat {S} | Z, \varepsilon) = I (\hat {S}; \varepsilon | Z), \tag {32}
$$

where $H ( \hat { S } | Z , \varepsilon ) = 0$ due to the determinism assumption. Substituting these back into the entropy identity yields:

$$
I (\hat {S}; \hat {Z} _ {S}) = H (\hat {S}) - I (\hat {S}; \hat {Z} _ {C} | \hat {Z} _ {S}) - I (\hat {S}; \varepsilon | Z). \tag {33}
$$

Step 2: Bounding the Entanglement. We now bound the entanglement term $I ( \hat { S } ; \hat { Z } _ { C } | \hat { Z } _ { S } )$ to introduce the true attribute S. By the chain rule of mutual information, adding a variable cannot decrease the information:

$$
I (\hat {S}; \hat {Z} _ {C} | \hat {Z} _ {S}) \leq I (\hat {S}, S; \hat {Z} _ {C} | \hat {Z} _ {S}). \tag {34}
$$

Expanding the right-hand side via the chain rule:

$$
I (\hat {S}, S; \hat {Z} _ {C} | \hat {Z} _ {S}) = I (S; \hat {Z} _ {C} | \hat {Z} _ {S}) + I (\hat {S}; \hat {Z} _ {C} | \hat {Z} _ {S}, S). \tag {35}
$$

We bound the second term using the property that mutual information is bounded by discrete entropy, $I ( X ; Y | Z ) \leq H ( X | Z )$ :

$$
I (\hat {S}; \hat {Z} _ {C} | \hat {Z} _ {S}, S) \leq H (\hat {S} | \hat {Z} _ {S}, S) \leq H (\hat {S} | S). \tag {36}
$$

Combining these inequalities gives:

$$
I (\hat {S}; \hat {Z} _ {C} | \hat {Z} _ {S}) \leq I (S; \hat {Z} _ {C} | \hat {Z} _ {S}) + H (\hat {S} | S). \tag {37}
$$

Step 3: Rearrangement. Substituting the bound from (37) into (33):

$$
I (\hat {S}; \hat {Z} _ {S}) \geq H (\hat {S}) - \left[ I (S; \hat {Z} _ {C} | \hat {Z} _ {S}) + H (\hat {S} | S) \right] - I (\hat {S}; \varepsilon | Z). \tag {38}
$$

We group the entropy terms. Recalling that $I ( S ; \hat { S } ) = H ( \hat { S } ) - H ( \hat { S } | S )$ , we obtain the final lower bound:

$$
I (\hat {S}; \hat {Z} _ {S}) \geq \underbrace {I (S ; \hat {S})} _ {\text { Fidelity }} - \underbrace {I (\hat {S} ; \varepsilon | Z)} _ {\text { Noise   Dependence }} - \underbrace {I (S ; \hat {Z} _ {C} | \hat {Z} _ {S})} _ {\text { Entanglement }}. \tag {39}
$$

Additionnaly, by the data-processing theorem, since $\hat { S }$ is a function of $\hat { X }$ , we have $I ( \hat { S } ; \varepsilon | Z ) \leq$ $I ( { \hat { X } } ; \varepsilon | Z )$ .

Therefore:

$$
I (\hat {S}; \hat {Z} _ {S}) \geq \underbrace {I (S ; \hat {S})} _ {\text { Fidelity }} - \underbrace {I (\hat {X} ; \varepsilon | Z)} _ {\text { Noise   Dependence }} - \underbrace {I (S ; \hat {Z} _ {C} | \hat {Z} _ {S})} _ {\text { Entanglement }} \tag {40}
$$

□

# C.3 Link Between Cycle consistency and Independence

In Sec. 3, we introduce a cycle consistency loss $\mathcal { L } _ { c y c l e }$ and claim that it helps enforce independence between latent codes $\hat { Z } _ { S }$ and $\hat { Z } _ { C }$ . More precisely, we claim the following:

Proposition 2. Let $\hat { Z } _ { C } ^ { a }$ and $\hat { Z } _ { S } ^ { b }$ be latent codes extracted from independently sampled images a and b with independent generative factors C and S. Define the mixed latent code

$$
Z ^ {\mathrm{mix}} = \hat {Z} _ {C} ^ {a} + \hat {Z} _ {S} ^ {b},
$$

and let

$$
(\hat {Z} _ {S} ^ {\mathrm{mix}}, \hat {Z} _ {C} ^ {\mathrm{mix}}) = E _ {\theta_ {E}} (Z ^ {\mathrm{mix}})
$$

be the separator outputs. Then, minimizing the cycle-consistency loss

$$
\mathcal {L} _ {\mathrm{cyc}} = \mathbb {E} \Big [ \| \hat {Z} _ {C} ^ {\mathrm{mix}} - \hat {Z} _ {C} ^ {a} \| _ {2} ^ {2} + \| \hat {Z} _ {S} ^ {\mathrm{mix}} - \hat {Z} _ {S} ^ {b} \| _ {2} ^ {2} \Big ]
$$

enforces statistical independence between $\hat { Z } _ { S }$ and $\hat { Z } _ { C } .$ .

Proof. Given $Z ^ { \mathrm { m i x } } = \hat { Z } _ { C } ^ { a } + \hat { Z } _ { S } ^ { b }$ , since a and b are independent, it follows that also $\hat { Z } _ { C } ^ { a }$ and $\hat { Z } _ { S } ^ { b }$ are independent: $\hat { Z } _ { C } ^ { a } \perp \perp \hat { Z } _ { S } ^ { b }$ . If the cycle consistency loss is minimized, namely $\hat { Z } _ { S } ^ { m i x } = \hat { Z } _ { S } ^ { b }$ and $\hat { Z } _ { C } ^ { m i x } = \hat { Z } _ { C } ^ { a }$ , then we also obtain $\hat { Z } _ { C } ^ { m i x } \perp \perp \hat { Z } _ { S } ^ { m i x }$ . Finally, assuming that mixed latents follow the same law as real samples $( \hat { Z } ^ { m i x } \overset { L a w } { = } Z )$ , we have $E _ { \theta _ { E } } ( Z _ { m i x } )  { \stackrel { \stackrel { L a w } { = } } { = } } E _ { \theta _ { E } } ( Z )$ . Since $E _ { \theta _ { E } } ( Z _ { m i x } ) =$ $( \hat { Z } _ { S } ^ { m i x } , \hat { Z } _ { C } ^ { m i x } ) = ( \hat { Z } _ { S } ^ { b } , \hat { Z } _ { C } ^ { a } )$ and $E _ { \theta _ { E } } ( Z ) = ( \hat { Z } _ { S } , \hat { Z } _ { C } )$ , we obtain $( \hat { Z } _ { S } , \hat { Z } _ { C } ) \overset { L a w } { = } ( \hat { Z } _ { S } ^ { a } , \hat { Z } _ { C } ^ { b } )$ , which entails that $\hat { Z } _ { S } \perp \perp \hat { Z } _ { C }$ . □

The assumption ${ \hat { Z } } ^ { m i x } \ { \stackrel { L a w } { = } } \ Z$ is not enforced with a loss, but is rather a consequence of our training paradigm for cycle consistency, as explained in App. F.

# Appendix D: Implementation Details for Fine-Grained Control

# D.1 A Diffusion-based Conditioning Architecture

Conditioning as the Primary Signal. To strictly control salient and common factors via the conditioning Z, we must ensure that Z is the sole driver of semantic content, minimizing information leakage into the noise ε used for generation (see Figure 2). Standard prompt-based methods inject features into frozen text-to-image backbones, often failing to override the base model’s priors and leading to identity loss or style drift (Fig. 8). Consequently, as stated in Sec. 4.1, we train a generative model from scratch using image-only conditioning, derived from DINOv3 [48] embeddings, consisting in K tokens. We adopt a Flow Matching objective, which enforces a deterministic mapping between noise and data, ensuring that visual fidelity is entirely determined by the structure of Z rather than stochastic sampling paths.

# D.1.1 Architecture for Low Noise Dependence

The generator $G _ { \psi }$ is built conditioned on a structured token sequence Z (Fig. 3). It operates on VAE latents and is trained using Optimal Transport Conditional Flow Matching (OT-CFM) [32, 50], which encourages straight trajectories in the latent space.

Latent Flow Matching. Let $u _ { 1 } \in \mathbb { R } ^ { d }$ denote the image latents encoded by a frozen VAE. We define a linear probability path interpolating between a source distribution $p _ { 0 }$ (standard Gaussian noise) and the target data distribution $p _ { 1 }$ (image latents):

$$
u _ {t} = (1 - t) u _ {0} + t u _ {1}, \quad t \in [ 0, 1 ] \tag {41}
$$

where $u _ { 0 } \sim \mathcal { N } ( 0 , I )$ and $u _ { 1 }$ is the data latent. We train a U-Net vector field estimator $v _ { \psi }$ to predict the velocity of this path, minimizing the flow matching objective:

$$
\mathcal {L} _ {\mathrm{FM}} = \mathbb {E} _ {t, u _ {0}, u _ {1}} \left[ \left| \left| v _ {\psi} (u _ {t}, t, Z) - (u _ {1} - u _ {0}) \right| \right| ^ {2} \right]. \tag {42}
$$

This formulation yields a deterministic ODE solver at inference time, enabling higher fidelity reconstruction and more stable latent manipulations than stochastic sampling.

Structured Token Conditioning. The conditioning signal $Z \in \mathbb { R } ^ { K \times d _ { c o n d } }$ must capture both highlevel semantics and low-level appearance. We construct Z by concatenating two complementary token streams:

• Semantic Tokens T : We extract features $F _ { \mathrm { d i n o } }$ from a frozen DINOv3 encoder. A lightweight cross-query module with K − 1 learnable queries base on Perceiver-IO [25] compresses these features into a fixed-length sequence $T \in \mathbb { R } ^ { ( \hat { K } - 1 ) \times d _ { \mathrm { c o n d } } }$ . This distills spatial semantics (e.g., “eyes”, “texture”) into a compact representation. This pipeline is illustrated in Fig. 11 with further details in Tab. 3.   
• Color Token $t _ { \mathrm { c o l } } \colon$ Diffusion models have trouble recovring low-frequency information [55, 47]. Additionally, the distilled DINO features can lose color information. We introduce a single token $t _ { \mathrm { c o l } }$ derived from a shallow CNN encoder trained to recover histogram information and mitigate color-shifting. The specification are described in Tab. 4.

![](images/87c9c97dd2792f3c6dc6e09ee9f7eb36e34249001087da47f15e8e05dfe1235b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Patch tokens"] --> B["proj"]
    B --> C["Self Attention"]
    C --> D["Cross Attention"]
    D --> E["proj+norm"]
    F["Learnable Queries"] --> D
    D --> G["×4"]
    G --> H["Output"]
```
</details>

Figure 11: Diagram of our Cross-query extractor module

The final conditioning $Z : = [ T ; t _ { \mathrm { c o l } } ]$ is injected via cross-attention. This design forces the flow model to rely on T for structure and $t _ { \mathrm { c o l } }$ for colorimetric information.

# D.1.2 Empirical Validation of Structural Assumptions

We evaluate our conditioning architecture against several established baselines: DiffAE [42], which utilizes a ResNet to produce a single 512-d conditioning vector; EncDiff [53], which trains a CNN to generate 32 × 128 tokens; and projected variants of CLIP and DINOv3. For the latter, we utilize all last-layer activations (197 and 261 tokens respectively) and project each to 128-d before conditioning. All models are trained on FFHQ [27] and tested on CelebA-HQ [33].

High-Fidelity Reconstruction. As summarized in Tab. 5, our Cross-Query and Color (CQC) conditioning achieves a very good LPIPS and FID, as well as a correct SSIM: this indicates that the model successfully reconstructs the major semantic attributes of our image, and relies on the noise to retrieve high-frequency textural details (e.g. hair). Visual results in Figure 12 confirm that our model accurately preserves identity and fine structural details when sampling from random noise. Importantly, the inclusion of the color token $t _ { c o l }$ mitigates the histogram mismatch that can be observed in vanilla Cross-Query reconstructions. See App. E.2 for exhaustive reconstruction metrics across multiple datasets and App. E.1 for further qualitative analysis.

Latent Manipulation. To assess the geometric and semantic properties of the conditioning space $Z ,$ we perform linear operations within the token manifold.

![](images/b593e6038eb7af5c9bcef165b262bfd8e35699a6ff770bcdefac111ca7345fcf.jpg)

<details>
<summary>text_image</summary>

Original
DiffAE
EncDiff
CLIP
DINOv3
CQ
(ours)
CQC
(ours)
</details>

Figure 12: Reconstruction from random noise for different conditioning encoders. Our conditioning preserves identity, local details, and color. Zoom-in for details. Additional results in App. E.1.

PC 1 (Std: 12.262)   
![](images/a35f65db3c01e47fe0bc46f125080e7a63b935f5d70b7a501b6f6e09cd7733f7.jpg)

<details>
<summary>text_image</summary>

Alpha=-1.50σ
Alpha=-0.75σ
Alpha=0.00σ
Alpha=0.75σ
Alpha=1.50σ
</details>

PC 2 (Std: 11.718)   
![](images/61f49903b5f5d9528e2ded7cd9a777e5dfb3bea342c9b5a0615fa56a6693d05d.jpg)

<details>
<summary>text_image</summary>

Alpha=-1.50σ
Alpha=-0.75σ
Alpha=0.00σ
Alpha=0.75σ
Alpha=1.50σ
</details>

Figure 13: Principal directions of the Z-space. The first one corresponds to head position, the second one to gender.

• Interpolation: Fig. 4 (Right) shows that linear interpolation between Z codes of two identities yields smooth, artifact-free transitions, indicating a continuous and well-structured representation, compared to DINOv3 which displays a "cross-fading" effect.   
• PCA-based Editing: We further probe the semantic linearity of the token space by exploring its principal components. As illustrated in Fig. 13, moving linearly along these directions induces semantically valid changes in the output images, supporting the linear assumption of our conditioning space.

These results illustrate that our token representation supports the linear and additive assumption done in Sec.3.1.

# D.2 Training setup

Datasets. We use FFHQ [27] as our main training experiment dataset. Following recommendation from their creator, we use the first 60k images as a training set and the last 10k images as the testing set. Images are recized to 256×256, and the dataset is processed by the publically available

Table 2: U-Net Architecture Specifications across datasets 

<table><tr><td rowspan="2">Parameter</td><td colspan="3">Dataset</td></tr><tr><td>FFHQ</td><td>AFHQ</td><td>BraTS</td></tr><tr><td> $f$ </td><td>4</td><td>8</td><td>4</td></tr><tr><td> $z$ -shape</td><td>64 × 64 × 3</td><td>32 × 32 × 4</td><td>32 × 32 × 4</td></tr><tr><td> $|Z|$ </td><td>8192</td><td>4096</td><td>4096</td></tr><tr><td>Channels</td><td>128</td><td>128</td><td>128</td></tr><tr><td>Depth</td><td>2</td><td>2</td><td>2</td></tr><tr><td>Channel Multiplier</td><td>[1, 2, 4, 4]</td><td>[1, 2, 4, 4]</td><td>[1, 2, 4, 4]</td></tr><tr><td>Attention resolutions</td><td>[32, 16, 8]</td><td>[32, 16, 8]</td><td>[32, 16, 8]</td></tr><tr><td>Head Channels</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Cross-Attention dim</td><td>128</td><td>128</td><td>128</td></tr><tr><td># Params</td><td>160M</td><td>160M</td><td>160M</td></tr><tr><td>Compute</td><td>90 GFlops</td><td>34 GFlops</td><td>34 GFlops</td></tr><tr><td>Prediction</td><td>v-pred</td><td>v-pred</td><td>v-pred</td></tr><tr><td>Training type</td><td>OT Flow-Matching</td><td>OT Flow-Matching</td><td>OT Flow-Matching</td></tr><tr><td>Batch Size</td><td>64</td><td>64</td><td>64</td></tr><tr><td>Optimizer</td><td>AdamW</td><td>AdamW</td><td>AdamW</td></tr><tr><td>Iterations</td><td>300k</td><td>300k</td><td>300k</td></tr><tr><td>Learning Rate</td><td>1e-4</td><td>1e-4</td><td>1e-4</td></tr><tr><td>EMA decay</td><td>0.9999</td><td>0.9999</td><td>0.9999</td></tr></table>

Table 3: CrossQueryEncoder Architecture Specifications. 

<table><tr><td>Output tokens count</td><td>32</td></tr><tr><td>Output token dim</td><td>128</td></tr><tr><td>Input embedding dim</td><td>768</td></tr><tr><td>Latent dimension (d)</td><td>512</td></tr><tr><td>Depth</td><td>4</td></tr><tr><td>Attention heads</td><td>12</td></tr><tr><td>Head dimension</td><td>64</td></tr><tr><td>Feedforward multiplier</td><td>4</td></tr><tr><td># Params</td><td>15M</td></tr><tr><td>Compute</td><td>2.3 GFlops</td></tr><tr><td>Batch Size</td><td>64</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>1e-4</td></tr></table>

Table 4: ColorEncoder Architecture 

<table><tr><td>Layer</td><td>Operation</td><td>Kernel/Stride</td><td>Output Shape</td></tr><tr><td>Input</td><td>RGB Image</td><td>-</td><td> $3 \times H \times W$ </td></tr><tr><td>Conv1</td><td>Conv2D + ReLU</td><td> $5 \times 5 / 2$ </td><td> $32 \times \frac{H}{2} \times \frac{W}{2}$ </td></tr><tr><td>Conv2</td><td>Conv2D + ReLU</td><td> $5 \times 5 / 2$ </td><td> $128 \times \frac{H}{4} \times \frac{W}{4}$ </td></tr><tr><td>Pool</td><td>AdaptiveAvgPool2D</td><td> $1 \times 1$ </td><td> $128 \times 1 \times 1$ </td></tr><tr><td>Flatten</td><td>Reshape</td><td>-</td><td>128</td></tr><tr><td>FC</td><td>Linear</td><td>-</td><td>128</td></tr></table>

VQ-VAE trained on CelebA- $\cdot \mathrm { H Q } ^ { 2 }$ . Despite FFHQ being notoriously more diverisifed than CelebA, this VQ-VAE model performed better in raw reconstuction than the standard sd-vae-ft-mse.

We use CelebA-HQ [33] as a generalization dataset to assess the reconstruction quality of our first stage training. We use the 30k images as the testing set, and resize them to $2 5 6 \times 2 5 6$ . The dataset is processed using the same VQ-VAE as FFHQ.

# Original

![](images/e896f32a7dda0b2751cb6ef0462b25d5b9a77f8f365b58b3ed4e48ea782b24bb.jpg)

<details>
<summary>natural_image</summary>

Six grayscale brain MRI scans showing axial, sagittal, and coronal views of the human brain (no text or annotations visible)
</details>

Figure 14: Visual inspection of the BraTS 2023 autoencoding using sd-vae-ft-mse.

We use BraTS 2023 [28] as an auxiliary dataset with a different domain. Since we need the segmentation masks to have tumor labels, we decide to re-split the train set (since the validation set does not contain the segmentation labels). To avoid any data leakage, we split patient-wise to get 90/10 train/test splits. We use the T1-normalized MRI volume and apply z-score normalization restricted to non-background voxels using a brain mask. The volume is then sliced axially, and each slice is clipped to a fixed intensity range (-3 to 3), min–max normalized, and rescaled to a range of [0, 255]. Slices that have more than 99.5% of black pixels are discarded, and remaining slices are replicated across three channels and saved as PNG images. Using the segmentation masks, each slice is labeled with a binary tumor-presence indicator. We use the pre-trained sd-vae-ft-mse as the autoencoder for this dataset. A visual inspection (Figure 14) confirms that despite being out-of-domain, this dataset can be easily processed with this autoencoder. The resulting training split consist of 151,824 image of 1,323 patients and the test split consist of 16’830 images of 147 patients. For evaluation, we concentrate on the slices between 85 and 100.

We use AFHQ [8] as another auxiliary dataset with a different domain than faces. The dataset is comprised of 3 main classes: cat; dog; wild. We resize all 16,130 images to 256 × 256, and perform geometrical augmentations (random flip and light random crop) during first stage training to mitigate overfitting. We use the sd-vae-ft-mse as the autoencoder for this dataset.

Optimization and Training. For efficiency, we precompute DINOv3 and VAE features. We jointly train the U-Net, Cross-Query, and Color encoder for 300k steps, using AdamW and a batch size of 32. The learning rate is warmed up from 10−6 to 10−4 for 5000 steps, then kept constant at 10−4 for the rest of the training. All models are trained in mixed precision with bfloat16. Training was performed on a single NVIDIA H100 GPU, equipped with an Intel Xeon Platinum 8468 CPU with 24 active cores (workers). Training takes about 48 hours on this configuration.

Flow Matching. The only loss at play in this stage is $\mathcal { L } _ { F M }$ defined in Eq. 42. We use Minimatch Optimal Transport Flow matching [50], where in each batch, we create data-noise pairs by minimizing the overall OT distance between the two distributions.

Diffusion inference details. We sample our images using a simple Euler integration, with 20 steps (unless otherwise noted). For all methods, the same starting noise is used. A study on the effect of random noise initialization can be found in App. E.1.

# D.3 Baselines

DiffAE. We use the official DiffAE repository3, and use their pretrained weights ffhq\_256\_autoenc. The original DiffAE method heavily relies on noise inversion to perfom the edits. However, we saw in Sec. 3 that to properly control target attributes, we need these attributes to not depend on initial noise. Therefore, we generate the images from random noise. This explains most of the performance metric differences with the original paper.

EncDiff. The EncDiff architecture differs with our work mainly on how the conditioning tokens are constructed. Therefore, for a fair comparison, we re-empleted the exact EncDiff encoder, and trained it in a similar setting than our DINO-based Cross-Query extractor. This ensures that the difference in reconstruction is due to conditioning quality, and not diffusion training differences. We trained EncDiff to produce exactly the same number of tokens than us, resulting in a much tighter bottleneck.

DINOv3 and CLIP. For efficiency, we first pre-compute DINOv3 and CLIP embeddings and store them. For CLIP, images were resized to 224 × 224 for compatibility. All of the last activations were used to condition the diffusion model.

• For DINOv3: this includes the CLS token, the register tokens, and the 256 encoded patch tokens, resulting in K = 261 tokens.   
• for CLIP: this includes the CLS token and the 197 patch tokens, resulting in K = 198 tokens.

All of the tokens were linearly projected in d = 128 for consistency across different backbones (except for pre-trained DiffAE which does not condition using cross-attention, and uses d = 512).

# Appendix E: Additional Results for the Conditioning Tokens.

# E.1 Additional Qualitative Results for Conditioning

Cross-Attention maps. To understand what kind of spatial or semantic information each conditioning token focuses on during the generation process, we analyze the cross-attention maps within the√ U-Net. Specifically, we extract the attention matrices, Softmax $( Q K ^ { T } / \sqrt { d _ { k } } )$ , from the cross-attention layers in the final decoder block of the U-Net where our concept tokens interact with the spatial features of the U-Net. These maps are averaged across all sampling timesteps to provide a stable visualization of each token’s influence.

Visualizations (Figure 15) show that our cross-query concept tokens produce attention maps that highlight broad semantic regions (e.g., "eyes and mouth", "lower-left face", "eyebrows and chin"). In contrast, raw DINOv3 attention maps tend to be highly localized and correspond to specific image patches. For EncDiff, the attention maps appear more diffuse or less structurally coherent, indicating a less specialized encoding by its tokens.

Cross-Attention maps. To understand what kind of spatial or semantic information each conditioning token focuses on during the generation process, we analyze the cross-attention maps within the√ U-Net. Specifically, we extract the attention matrices, Softmax $( Q K ^ { T } / \sqrt { d _ { k } } )$ , from the cross-attention layers in the final decoder block of the U-Net where our concept tokens interact with the spatial features of the U-Net. These maps are averaged across all sampling timesteps to provide a stable visualization of each token’s influence.

Visualizations (Figure 15) show that our cross-query concept tokens produce attention maps that highlight broad semantic regions (e.g., "eyes and mouth", "lower-left face", "eyebrows and chin"). In contrast, raw DINOv3 attention maps tend to be highly localized and correspond to specific image patches. For EncDiff, the attention maps appear more diffuse or less structurally coherent, indicating a less specialized encoding by its tokens.

Dependence on starting noise. Fig. 17 shows compares the dependence on the starting noise of DiffAE vs. our CrossQuery conditioning. Our method achieves better structural reconstruction across various starting noises. This seems to indicate that DiffAE’s reconstruction power comes, in part, from the noise inversion protocol. Therefore, precise structural edits are not possible with DiffAE by acting solely on the conditioning.

EncDiff   
![](images/57f4a31bae59e9230e311a6e87caeedc596e3ae2a2ea157f9a67ee6eaf785451.jpg)

<details>
<summary>natural_image</summary>

Five thermal imaging slices showing a person's face with different lighting and color variations (no text or symbols visible)
</details>

DINOv3   
![](images/430cc1883385d01d145a501b6fc526418576505bb0eb1b0850eadcbec01ca7e2.jpg)

<details>
<summary>natural_image</summary>

Sequence of five portrait photos showing a woman's face with facial features and eye thermal imaging overlays (no text or symbols)
</details>

Cross-Query (ours)   
![](images/80c3ca53d9deb93e498c75c7c4af76ee44f020e07f05fa84a54754491c18d0e9.jpg)

<details>
<summary>natural_image</summary>

Sequence of five facial images showing thermal imaging patterns (no text or symbols visible)
</details>

Figure 15: Cross-attention maps for some conditioning tokens, comparing our conditioning to raw DINOv3 and EncDiff. Our tokens attend to large, semantically coherent regions (e.g., facial parts), whereas DINO tokens are overly local and EncDiff tokens diffuse, supporting the claim that our conditioning yields a more structured, interpretable representation.

Effect of the color token. We display the information learned by the color token in Fig. 18.

# E.2 Additional Quantitative Results for Conditioning

ID-Sim implementation details. The ID-Sim metrics correspond to the cosine similarity in a Face Recognition model’s latent space between the original and the generated images. It aims at verifying that the generated face represents the same person as the original image. We use the latent space of an IResnet 50 [20] trained on MSMV3 [10] with the ArcFace loss [9].

More precisely, we extract $I d _ { o }$ and $I d _ { g } ,$ , the latent code for the original, and generate images which naturally lie on the unit hypersphere $S ^ { \tilde { 5 } 1 2 }$ . We then derive ID-Sim as follows:

$$
\text { ID - Sim } = \cos (I d _ {o}, I d _ {g}) = I d _ {o} ^ {T} I d _ {g} \tag {43}
$$

Reconstruction. We present additional reconstruction metrics of the models trained on FFHQ and tested on CeleA-HQ with various ablations in Tab. 5. As we see, DINOv3 leads to the best reconstruction metrics due to its massive conditioning size, combined with DINO-based training which favors very structured outputs. There are promising existing approaches that use these DINO feature maps as the direct input to diffusion models [57]. We instead focus on using these tokens to condition diffusion through cross-attention.

![](images/964e9bda8cad3dd76ba285cc121405787b193472d2203f34bbcb396e165a941b.jpg)

<details>
<summary>text_image</summary>

Original
Mm-cVAE
SepVAE
Double InfoGAN
Ours
Reconstruction
Mm-cVAE
SepVAE
Double InfoGAN
Ours
Swapping
</details>

Figure 16: Comparing reconstruction and swapping between Diff-CA (ours) and CA baselines. The baselines succeed in swapping the salient factor, but only generate blurry pictures, undermining the fidelity to the original image.

![](images/6a9b38a67f0c3140154ffdee0a49d7d45dd32fe198cfcd4954c7d634982d12ec.jpg)

<details>
<summary>text_image</summary>

samples
Mean Std
DiffAE
CQC
(Ours)
</details>

Figure 17: Dependence of the generated images on different starting noises.

Our cross query extractor module, despite learning to a much smaller conditioning space, still achieves competitive results, particularly in perceptual metrics such as LPIPS and DISTS. This seems to indicate that although very fine-graine textures (such as hair or grass) can be lost through the distilation process, most of the semantic information is still present in the Z-space.

We present reconstruction metrics of our model on the other datasets in Tab 7, and study the effect of sampling steps in Tab. 6.

# Appendix F: Implementation Details for Separating Network

# F.1 Architecture

The separator network $E _ { \theta _ { E } }$ is designed to process the conditioning token sequence Z and perform a structured decomposition into salient $\hat { Z } _ { S }$ and common $\hat { Z } _ { C }$ components.

Transformer Backbone. We implement $E _ { \theta _ { E } }$ as a 5-layer Transformer encoder. Each layer consists of a multi-head self-attention (MHSA) block with 12 heads and a feed-forward network (FFN) with a hidden dimension of 512, following the standard Transformer architecture. The input sequence Z consists of $K = 3 2$ tokens of dimension $d = 1 2 8$ . No positional encoding is needed, since the conditioning DINOv3 tokens were already processed with their own positional encoding.

Common-Salient Head. Following the setup in Fig. 3, we prepend a learnable CLS token to the input sequence Z. After processing through the Transformer blocks, the state of the CLS token is passed through a linear projection layer to produce the salient code $\hat { Z } _ { S } \in \mathbb { R } ^ { d }$ . The remaining K output tokens are treated as the common code $\hat { Z } _ { C } \in \mathbb { R } ^ { K \times d }$ . This design ensures that the salient information is distilled into a compact representation while the common information preserves spatial and semantic context across the original token sequence length.

Original   
![](images/4eca5a6cc3378703d46dc4bec81c96610ef1dd25eb4a080da157112b2dac892e.jpg)

Rec.   
Color token permutation   
Figure 18: Effect of the color token. Top row: original images. Middle row: reconstruction using T and $t _ { c o l }$ . Bottom row: we permute the color tokens $t _ { c o l }$ between the 4 images and reconstruct from this modified conditioning. The color token $t _ { c o l }$ contains color histogram information. 

<table><tr><td>Encoder</td><td>Steps</td><td>Cond. Size</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DISTS↓</td><td>FID↓</td><td>FD-DINO↓</td><td>KID↓</td><td>ID-Sim↑</td></tr><tr><td>DiffAE</td><td>20</td><td>512</td><td>19.09</td><td>0.530</td><td>0.209</td><td>0.173</td><td>14.28</td><td>32.89</td><td>0.013</td><td>0.407</td></tr><tr><td>DiffAE</td><td>200</td><td>512</td><td>18.81</td><td>0.507</td><td>0.196</td><td>0.161</td><td>4.76</td><td>28.70</td><td>0.003</td><td>0.404</td></tr><tr><td>EncDiff</td><td>20</td><td> $32 \times 128$ </td><td>17.03</td><td>0.420</td><td>0.323</td><td>0.210</td><td>18.43</td><td>64.91</td><td>0.014</td><td>0.313</td></tr><tr><td>CLIP</td><td>20</td><td> $198 \times 128$ </td><td>18.23</td><td>0.490</td><td>0.201</td><td>0.169</td><td>7.69</td><td>31.45</td><td>0.006</td><td>0.538</td></tr><tr><td>DINOv3</td><td>20</td><td> $261 \times 128$ </td><td>23.31</td><td>0.645</td><td>0.099</td><td>0.114</td><td>5.66</td><td>20.89</td><td>0.005</td><td>0.609</td></tr><tr><td>CQ</td><td>20</td><td> $32 \times 128$ </td><td>21.76</td><td>0.589</td><td>0.124</td><td>0.131</td><td>5.59</td><td>22.39</td><td>0.005</td><td>0.515</td></tr><tr><td>CQC</td><td>20</td><td> $32 \times 128$ </td><td>22.57</td><td>0.611</td><td>0.114</td><td>0.118</td><td>5.52</td><td>22.28</td><td>0.005</td><td>0.523</td></tr></table>

Table 5: Additional reconstruction results across various setups. Training is done on the train split of FFHQ and evaluated on CelebA-HQ in 256 × 256. Sampling is performed from the same random noise. CQ: Cross-Query without color token; CQC: Cross-Query with Color token.

Additive Constraint. The final conditioning fed to the generator $G _ { \psi }$ is reconstructed as $\hat { Z } =$ $\hat { Z } _ { S } + \hat { Z } _ { C }$ . This additive structure is strictly enforced by the reconstruction loss $\mathcal { L } _ { r e c }$ .

# F.2 Training setup

Weak Supervision and Binarization. Our framework relies on weak binary supervision $Y \in \{ 0 , 1 \}$ to separate common from salient factors. Since many benchmark datasets contain continuous or multi-class labels, we apply the following binarization protocols for training:

• FFHQ (Glasses): We define the background $( Y = 0 )$ as images with the "No Glasses" label and the target $( Y = 1 )$ as any image containing eyewear. For evaluation, we utilize held-out fine-grained labels (Reading Glasses, Sunglasses) to assess latent structure discovery.   
• FFHQ (Headpose): Samples are binarized based on the yaw angle θ. We define $Y = 0$ (Neutral) if $\bar { | \theta | } \overset { - } { \le } 8 ^ { \circ }$ and $\dot { Y } = 1$ (Turned) if $| \theta | > 1 2 ^ { \circ }$ . This gap ensures a clear separation between distributions during training. Evaluation is performed by measuring the Mean Absolute Error (MAE) of yaw predicted by a separate regressor on swapped images.

<table><tr><td>Encoder</td><td>Steps</td><td>Cond. Size</td><td>PSNR↑</td><td>SSIM↑</td><td>LPIPS↓</td><td>DISTS↓</td><td>FID↓</td><td>FD-DINO↓</td><td>KID↓</td></tr><tr><td>CQC</td><td>10</td><td>32 × 128</td><td>22.79</td><td>0.623</td><td>0.120</td><td>0.122</td><td>8.36</td><td>27.85</td><td>0.008</td></tr><tr><td>CQC</td><td>20</td><td>32 × 128</td><td>22.57</td><td>0.611</td><td>0.114</td><td>0.118</td><td>5.52</td><td>22.28</td><td>0.005</td></tr><tr><td>CQC</td><td>50</td><td>32 × 128</td><td>22.41</td><td>0.603</td><td>0.113</td><td>0.118</td><td>4.42</td><td>19.88</td><td>0.003</td></tr></table>

Table 6: Effect of the number of sampling steps on our cross query encoder reconstruction performance.

<table><tr><td>Dataset</td><td>Sampling Steps</td><td>Conditioning Size</td><td>SSIM↑</td><td>LPIPS↓</td><td>DISTS↓</td><td>FID↓</td><td>FD-DINO↓</td></tr><tr><td>FFHQ</td><td>20</td><td>32 × 128</td><td>0.585</td><td>0.120</td><td>0.127</td><td>6.18</td><td>16.18</td></tr><tr><td>AFHQ</td><td>20</td><td>32 × 128</td><td>0.433</td><td>0.155</td><td>0.139</td><td>8.13</td><td>15.91</td></tr><tr><td>BraTS</td><td>20</td><td>32 × 128</td><td>0.649</td><td>0.151</td><td>0.156</td><td>31.54</td><td>35.09</td></tr></table>

Table 7: Additional reconstruction metrics of our CrossQuery+Color Encoder on different datasets.

• FFHQ (Smile): We apply a threshold of 0.5 on the smiling attribute. We acknowledge significant label noise in this attribute and thus restrict our analysis to qualitative results.   
• AFHQ (Species): We utilize a binary Cat vs. Dog setup for training to treat species-specific traits as the salient factor.   
• BraTS 2023 (Tumor): To ensure high-quality training signals in medical data, we restrict our volume to axial slices between indices 84 and 100. A slice is labeled $Y = 1$ (Positive) if the segmentation mask contains > 500 tumorous pixels; otherwise, it is $Y = 0$ .

Preprocessing. We measure reconstruction and swapping accuracies during training. For efficiency, we evaluate the accuracy using a classifier trained on the conditioning space.

Adversarial training. Instead of relying on a classical adversarial setup using GAN-like adversarial losses, we opt for a Gradient Reversal Layer training procedure:

• Forward pass: The encoder $E _ { \theta _ { E } }$ produces samples $( \hat { Z } _ { S } , \hat { Z } _ { C } )$ . A small (2-layer MLP) adversary $D _ { a v d }$ takes $\hat { Z } _ { C }$ and aims at predicting the label yˆ associated with this sample, and a standard binary cross-entropy loss (BCE) is computed between y and $\hat { y }$ .   
• Backward pass: when gradients flow back from $D _ { a d v }$ to $E _ { \theta _ { E } }$ , the gradients are multiplied ${ \tt b y } - \lambda _ { a d v }$

The negative weighting of $\lambda _ { a d v }$ effectively allows the gradients that flow back to the encoder to maximize the BCE loss outputted by $D _ { a d v } ,$ , effectively emulating an adversarial training, with only one scalar $\lambda _ { a d v }$ to tune. Instead of relying on a single value of $\lambda _ { a d v }$ , we opt for a self tuning adversarial mechansim, described thereafter.

Self-tuning adversary. At each training step, a rolling accuracy Acc is computed on the current batch (with a small EMA decay of 0.95). The goal of the adversary is to maximize the accuracy, while the goal of the encoder is to minimize it. During various stages of training, the power of both networks to achieve their respective tasks can vary greatly. Therefore, we establish a target accuracy $A c c _ { t a r g e t }$ . At each step:

• If $A c c > A c c _ { t a r g e t }$ : The adversary is too strong, and we multiply $\lambda _ { a d v }$ by 0.999.   
• If $A c c < A c c _ { t a r g e t } ;$ The encoder is too strong, and we devide $\lambda _ { a d v }$ by 0.999.

This basic controller for $\lambda _ { a d v }$ was sufficient to keep a smooth training for target accuracies $A c c _ { t a r g e t }$ between 0.70 and 0.80, with $\lambda _ { a d \iota }$ varying between 0 and 5. We display an example of the evolution of $\lambda _ { a d v }$ in Fig. 19 and the associated accuracy evolution in Fig. 20.

Swaping Cycle-Consistency. When trained only using reconstruction loss $\mathcal { L } _ { r e c } { . }$ , adversarial loss $\mathcal { L } _ { a d v }$ , and pinning loss $\mathcal { L } _ { p i n }$ , we found that the model had good reconstruction performances, and that the salient $\hat { Z } _ { S }$ had correct classification accuracy. However, swapping often lead to unexpected

![](images/dee98b9d4fd433fc57d14e048f9f575570011f7525660e30c166d853b4d43b0a.jpg)

<details>
<summary>line</summary>

| Step | Adaptive lambda |
| ---- | ---------------- |
| 0    | 0.0              |
| 5k   | 0.7              |
| 10k  | 0.6              |
| 15k  | 0.8              |
| 20k  | 1.0              |
| 25k  | 1.2              |
| 30k  | 1.3              |
| 35k  | 1.2              |
| 40k  | 1.3              |
| 45k  | 1.4              |
| 50k  | 1.5              |
</details>

Figure 19: Training dynamics of the self-tuning GRL parameter $\lambda _ { a d v }$ . Its goal is to dynamically maintain the discriminator accuracy at 0.80.

![](images/58b2b3f0aa291f80eb51ab3b1a9c25a9111ba4bdb0eb318804bb012eede865d7.jpg)

<details>
<summary>line</summary>

| Step | Adaptive lambda |
| ---- | --------------- |
| 0    | 0.4             |
| 5k   | 0.8             |
| 10k  | 0.8             |
| 15k  | 0.8             |
| 20k  | 0.8             |
| 25k  | 0.8             |
| 30k  | 0.8             |
| 35k  | 0.8             |
| 40k  | 0.8             |
| 45k  | 0.8             |
</details>

Figure 20: Training adversarial accuracy using our self-tuning GRL schedule. After a chaotic warmup phase, accuracy stabilizes at 0.80.

Table 8: Detailed Swapping Results on FFHQ (Glasses Attribute). We evaluate three metrics across all six possible attribute transfers between No Glasses (N), Reading Glasses (R), and Sunglasses (S). Models were trained only on binary supervision $\scriptstyle ( \mathrm { N = 0 } ; \mathrm { R = 1 } , \mathrm { S = 1 } )$ with 60,000 samples. Evaluation is performed with 1,000 swapped pairs for each fine-grained class. 

<table><tr><td rowspan="2">Method</td><td rowspan="2">Metric</td><td colspan="2">From No Glasses</td><td colspan="2">From Reading</td><td colspan="2">From Sun</td><td>Aggregate</td></tr><tr><td> $N \rightarrow R$ </td><td> $N \rightarrow S$ </td><td> $R \rightarrow N$ </td><td> $R \rightarrow S$ </td><td> $S \rightarrow N$ </td><td> $S \rightarrow R$ </td><td>Score</td></tr><tr><td rowspan="3">MM-cVAE</td><td>Acc ↑</td><td>15.6</td><td>27.1</td><td>1.00</td><td>26.0</td><td>1.00</td><td>0.00</td><td>11.78</td></tr><tr><td>ID-Simrec↑</td><td>.592</td><td>.394</td><td>.591</td><td>.465</td><td>.390</td><td>.390</td><td>.470</td></tr><tr><td>ID-Simreal↑</td><td>.302</td><td>.298</td><td>.302</td><td>.309</td><td>.300</td><td>.300</td><td>.302</td></tr><tr><td rowspan="2">SepVAE</td><td>Acc ↑</td><td>31.1</td><td>39.4</td><td>1.00</td><td>42.3</td><td>1.00</td><td>0.00</td><td>19.58</td></tr><tr><td>ID-Simreal↑</td><td>.300</td><td>.298</td><td>.300</td><td>.306</td><td>.301</td><td>.301</td><td>.301</td></tr><tr><td rowspan="3">D.InfoGAN</td><td>Acc ↑</td><td>87.8</td><td>47.5</td><td>99.8</td><td>46.1</td><td>1.00</td><td>0.00</td><td>47.03</td></tr><tr><td>ID-Simrec↑</td><td>.412</td><td>.415</td><td>.424</td><td>.425</td><td>.422</td><td>.422</td><td>.413</td></tr><tr><td>ID-Simreal↑</td><td>.311</td><td>.310</td><td>.318</td><td>.320</td><td>.312</td><td>.312</td><td>.314</td></tr><tr><td rowspan="3">Diff-CA(Ours)</td><td>Acc ↑</td><td>91.4</td><td>94.2</td><td>98.8</td><td>91.5</td><td>98.7</td><td>92.9</td><td> $94.48^{(+47.18)}$ </td></tr><tr><td>ID-Simrec↑</td><td>0.507</td><td>.458</td><td>.510</td><td>.492</td><td>.427</td><td>.453</td><td> $.474^{(+.004)}$ </td></tr><tr><td>ID-Simreal↑</td><td>0.468</td><td>.431</td><td>.463</td><td>.452</td><td>.398</td><td>.416</td><td> $.438^{(+.124)}$ </td></tr></table>

Notes: Acc denotes top-1 swapping accuracy of a fine-grained classifier. ID-Sim is the ArcFace cosine similarity between non-swapped and swapped identities: we compute it between the swappings and the non-swapped reconstructions of the model, $\scriptstyle \operatorname { I D - S i m } _ { \operatorname { r e c } }$ ;and between the swapping and the original images, $\scriptstyle \mathrm { I D - S i m } _ { \mathrm { r e a l } } .$ . KID is the Kernel Inception Distance.

![](images/8b5402be47b7525c016fd51c757b43b509ddce8781486c73c2ab7fb12c834130.jpg)

<details>
<summary>text_image</summary>

Reconstruction
Swapping without
Cycle-Consistency
</details>

Figure 21: Example of swapping with a model trained without cycle consistency. Swapping $\hat { Z } _ { S }$ leads to major color changes from the reconstruction, indicating a hidden dependance between $\hat { Z } _ { C }$ and $\hat { Z } _ { C }$

changes in the images, such as color shifts and other attributes inconsistencies (see Fig. 21). We hypothesize this happens because $\mathcal { L } _ { a d v }$ only enforces $\hat { Z } _ { C } ~ \perp \perp ~ Y$ and not $\hat { Z } _ { C } ~ \perp \perp ~ \hat { Z } _ { S } ,$ which a somewhat correct but imperfect approximation. Therefore, in this case conditions for Th. 3.1 are not met, and we observe a "leakage" from context to salient.

Cycle consistency loss aims to correct this. Intuitively, we want swapped counterfactuals (samples of the form $Z _ { m i x } = \hat { Z } _ { S } ^ { a } + \hat { Z } _ { C } ^ { b }$ with $\hat { Z } _ { S } ^ { a }$ and $\hat { Z } _ { C } ^ { b }$ from different images a and b) to be "valid" samples. In other terms, intuitively we want:

$$
Z _ {m i x} \stackrel {\text { Law }} {=} Z \quad \left(\mathrm{H} _ {\text { mix }}\right)
$$

which is the modeling assumption used in Sec. C.3. Importantly, this condition is not enforced by an explicit loss, but rather serves as an intuitive guideline motivating the design of the cycle-consistency objective.

While one could in principle enforce $\mathrm { ( H _ { m i x } ) }$ using an adversarial discriminator (similarly to how $\mathcal { L } _ { a d v }$ enforces $\hat { Z } _ { C } \perp \perp Y )$ , we found such approaches to be unstable in practice. Instead, we rely on the separator $E _ { \theta _ { E } }$ itself to act as a weak, implicit discriminator. Specifically, we maintain an exponential moving average (EMA) copy of $E _ { \theta _ { E } }$ , which is used to re-encode swapped counterfactuals $Z _ { \mathrm { m i x } }$ .

If $Z _ { \mathrm { m i x } }$ does not resemble a valid sample from the true conditioning distribution, the frozen EMA encoder fails to satisfy the cycle-consistency criterion. This mechanism limits co-adaptation between the representation $( \hat { Z } _ { S } , \hat { Z } _ { C } )$ and the separator, and should be understood as a heuristic regularization strategy rather than a formal guarantee of distributional matching or independence. This design is conceptually related to self-distillation methods such as BYOL [17], where an EMA target network prevents collapse toward trivial solutions.

Optimization and Hyperparameters. We jointly optimize the separator and adversarial discriminator for 1000 epochs using the AdamW optimizer with a batch size of 1024. Training was performed on a single NVIDIA A100 GPU, equipped with an AMD EPYC 7543 processor with 16 active cores (workers). Training takes about 3-4 hours on this configuration.

For the pinning loss $\mathcal { L } _ { p i n }$ , we use a weighting of 1 on the negative classes (to force all background salients to colapse at 0), and a weighting of 0.1 on the positive classes. The cycle consistency loss weighting was set at 1.

Table 9: Separator network specifications. 

<table><tr><td>Layers</td><td>5</td></tr><tr><td>Output token dim</td><td>128</td></tr><tr><td>Input embedding dim</td><td>768</td></tr><tr><td>Latent dimension (d)</td><td>512</td></tr><tr><td>Depth</td><td>4</td></tr><tr><td>Attention heads</td><td>12</td></tr><tr><td>Head dimension</td><td>64</td></tr><tr><td>Feedforward multiplier</td><td>4</td></tr><tr><td>Latent initialization</td><td>Learnable queries</td></tr><tr><td># Params</td><td>15M</td></tr><tr><td>Compute</td><td>2.3 GFlops</td></tr><tr><td>Batch Size</td><td>64</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning Rate</td><td>1e-4</td></tr></table>

# Appendix G: Additional Results for Common-Salient Editing

Table 10: Detailed Head Pose Swapping Results. Accuracy measures correct orientation (Left, Straight, Right) identification, while MAE measures precision in degrees. Trained only on binary Neutral $( y = 0 ) \mathrm { v s }$ . Turned $( y = 1 )$ supervision. 

<table><tr><td>Swap Operation</td><td>Accuracy ↑</td><td>Yaw MAE (°) ↓</td></tr><tr><td>Add Left (N → L)</td><td>84.5</td><td>5.5807</td></tr><tr><td>Add Right (N → R)</td><td>84.0</td><td>5.6945</td></tr><tr><td>Remove Left (L → N)</td><td>98.0</td><td>3.0699</td></tr><tr><td>Remove Right (R → N)</td><td>98.2</td><td>2.8752</td></tr><tr><td>Right → Left</td><td>88.7</td><td>5.5692</td></tr><tr><td>Left → Right</td><td>83.5</td><td>5.7251</td></tr><tr><td>Macro-Average</td><td>0.8948</td><td>4.7524</td></tr></table>

Detailed performance for glasses. We present the detailed swapping performance, in terms of swapping accuracy and identity perseveration in Tab. 8. Diff-CA achieves substantial better results in all swappings cases.

Detailed performance on Head Position. We train a small regressor MLP to predict the yaw attribute from our images. This classifier achieves a MAE of 2,55° . We train our separator network on the binary Neutral / Non-Neutral attribute, then evaluate it by performing swaps between the fine-grained labels Neutral / Left /Right. We then evaluate classification accuracy of the swap, both in terms of discrete labels as well as with the predicted yaw value. Results are reported in Tab. 10.

PCA and UMAP of the learned salient factor. Fig. 7 displays projections of the learned $\hat { Z } _ { S } .$ . As we see, dispite being learned solely using binary labels (Glasses vs. No Glasses), the space of $\hat { Z } _ { S }$ is organized according to finer-grained sublabels. This can be seen as an instance of concept discovery using CA.

![](images/e32878778c642c75e23617fc17a7761463cb3952a17b71904b86ea8d86ea573f.jpg)  
Figure 22: Additional swapping results on glasses in FFHQ using Diff-CA.

![](images/539330a50d7bf4a8e5c599ea7fc4d68747ac8aad9788bbdac8212c9df0ce8551.jpg)

Figure 23: Additional swapping results on the head position in FFHQ using Diff-CA.   
![](images/52bddccc24e56a6e49d5efb1e71ac61e52578966803373c75f90d0d4d24b0e1d.jpg)  
Figure 24: Additional swapping results on the cats/togs classes in AFHQ using Diff-CA.

Original   
![](images/c11cfa44626021922b728eb8dc542f479238273b5a98f8caeac51b09e7c452c0.jpg)

<details>
<summary>natural_image</summary>

Eight grayscale brain MRI scans arranged in two rows, showing different anatomical structures (no text or labels visible)
</details>

![](images/7a2c79e1c12508a4a5a1ebd1fd7828b84a725dba4cf53117dda42c4ba2c9c402.jpg)

<details>
<summary>natural_image</summary>

Eight grayscale brain MRI scans arranged in two rows, showing different anatomical views (no text or labels visible)
</details>

Swapped Salient   
Abs. Diff recon/swap   
Tumor 1   
![](images/d942ad375d34f4db1bb40c17b9ac03ffa90c18484cc05224a77b23722a043dbf.jpg)

Interpolating Salient   
![](images/2474a7644a811c35aa0aa14577714332d99b2121f7901b194853dc789abf39a1.jpg)

<details>
<summary>natural_image</summary>

Five grayscale oval brain scan images showing cross-sectional views of the brain (no text or labels visible)
</details>

Tumor 2   
![](images/976cbc4630c9e0b1e5849dc9cc9f5cff4619c6d819ca15b72f28a8e4fcccb6e6.jpg)  
Figure 25: Additional swapping and interpolation results on the BraTS 2023 dataset using Diff-CA.