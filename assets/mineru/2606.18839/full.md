# Semantic Robustness Certification for Vision-Language Models

Peiyu Yang 1 Paul Montague 2 Feng Liu 1 Andrew C. Cullen 1 Amardeep Kaur 2 Christopher Leckie 1 Sarah M. Erfani 1

## Abstract

Vision-language models (VLMs) are now widely used in downstream tasks. However, real-world applications often expose VLMs to distribution shifts induced by semantic variation (e.g., shape, size, and style). Robustness certification determines if a model’s prediction changes when transformations are applied to its input. While most certification frameworks study geometric or pixellevel transformations over inputs, this work proposes a novel framework that enables certifying VLM robustness under semantic-level transformations. Leveraging the open-vocabulary capability of VLMs, we use text prompts as semantic proxies to construct transformations parameterized by an extent that controls the degree of semantic variation. By characterizing the VLM decision boundary in closed form, our framework quantitatively certifies extent intervals for which the predicted class remains unchanged under the semantic transformation. Our framework is the first to certify VLM robustness under semantic-level variations without requiring additional data for each variation, making it practical to apply. Experiments on both synthetic and real-world data show that our framework enables certifying robustness under diverse semantic variations across scenarios. Code is available at https://github.com/ ypeiyu/vlm-semantic-cert.

## 1. Introduction

Vision-language models (VLMs) (Radford et al., 2021; Li et al., 2022; Alayrac et al., 2022) align images and text into a shared embedding space, enabling direct matching between vision and language for open-vocabulary reasoning. This transferable interface has made VLMs a foundation

1School of Computing & Information Systems, University of Melbourne, Australia 2Defence Science and Technology Group, Australia. Correspondence to: Peiyu Yang <peiyu.yang@unimelb.edu.au>.

Preprint. June 18, 2026.

for diverse downstream tasks such as detection, classification, and visual question answering (Du et al., 2022; Miyai et al., 2023; Xiao et al., 2024; Li et al., 2024). Despite their strong performance, VLM predictions can be fragile to visual semantic variations, raising concerns in high-stakes applications (Fang et al., 2022; Crabbe et al. ´ , 2023).

To provide guarantees of a model’s prediction invariance under input variations, robustness certification has been widely studied (Zhang et al., 2018; Cohen et al., 2019). Given allowable transformations $\gamma$ constrained by an extent $\varphi$ that bounds the strength of $\gamma ,$ , it aims to determine the range of $\varphi$ for which the prediction remains unchanged under transformations. Most certificates (Lecuyer et al., 2019; Cohen et al., 2019) focus on pixel-level transformation within an $L _ { p }$ ball, but cannot capture real-world semantic variations. Other works extend certification to closed-form geometrical transforms (e.g., rotations and translations) (Balunovic et al., 2019; Li et al., 2021), but they remain limited to a small set of hand-designed transformations. Recent works perform certification in the latent space of generative models (Mirman et al., 2021; Yuan et al., 2023), enabling advanced semantic transformations (e.g., facial attributes and weather conditions). However, the requirement of substantial training data for each semantic variation limits their practical use (Wang et al., 2023b). Since semantics are highly entangled in the input space, it is challenging for existing works to formulate transformations for these variations.

In this work, we reformulate certification for the VLM embedding space, where semantics are encoded in the geometry induced by cosine similarity, which supports formalizing semantic transformations. Leveraging the open-vocabulary grounding of VLMs, we use a pair of text prompts as semantic proxies to specify the source and target semantics of a variation. We identify that this semantic variation is confined to a two-dimensional subspace spanned by the corresponding textual embeddings. Projecting an image embedding onto this subspace yields a semantic extent that quantifies the strength of the target semantic relative to the source. By varying this extent, we construct a parameterized transformation in embedding space that models the semantic variation. Figure 1 shows that, for an image of $G y -$ oza, text prompts can serve as semantic proxies for diverse variation types (e.g., shape, style, and scene), allowing the construction of semantic transformations for specific target semantics (e.g., triangular, soft, and on a plate).

Input image (gyoza):  
![](images/95aba80536a646339174c7145c1e428f660fc10fcb7a5e08994b19cec222ba8d.jpg)  
Predicted prompt: “A photo ofa gyoza”

Target semantic (shape): triangular Proxy: “A photo of triangular gyoza.” Certificates over semantic extent φ:  
![](images/87489158fc305a8e7150143b33fff684302fb9c5b950595c2c5b0e0595c7a625.jpg)  
Retrieved nearest examples:

![](images/d4f8d06de44b9d69076ba2f5f3029e1117900de8ece78383bca0ddacb6fe0e57.jpg)  
Label: Samosa Similarity:0.92

![](images/feeecba18456f7a911cae75068a1a7bd65203f922e3511711583e0419d5460ce.jpg)  
Label: Samosa Similarity:0.92

![](images/d97902a96fe69239ff6c0315a99a98ce696da1419bb0b738d0f2c6777a32d1cf.jpg)  
Label: Samosa Similarity:0.91

Target semantic (style): soft Proxy: “A soft photo of gyoza.” Certificates over semantic extent φ:  
![](images/19d0e183688c45e5de4d8cdbe5b8b9ba37c76892d7e10b1eb4b776afdd4736a0.jpg)  
Retrieved nearest examples:

![](images/e68efcf21e22fa0d7a4ce05ae17b3e9a92eacada8f753eb3499dfcbc6f0f892c.jpg)  
Label: Dumpling Similarity:0.90

![](images/77e6835c84c2c5c7c9529343c0cdb045d7a2c3ab5f4aee72ac417db985d5805c.jpg)  
Label: Dumpling Similarity:0.88

![](images/816057bacdff0b4bf42f93511bfd46f8d235348757b6fcb1392b129eb0ee8c43.jpg)  
Label: Dumpling Similarity:0.88

Target semantic (scene): on a plate Proxy: “A photo of gyoza on a plate.” Certificates over semantic extent p:  
![](images/9f6e9cfa64064f44ef72964d7e0ff64e7e491ccf0839a5b9c1a784bdb081f415.jpg)  
Retrieved nearest examples:

![](images/be5c3c26e8acb67b0e9c3996f7c8e77d6638bc976445121e3e8fe42b4c9f1a43.jpg)  
Label: Gyoza Similarity:0.91

![](images/3bb9616401b79328fdea67efdfd1af4b4baf8dd2a3f90f56a2e123c8c3e5f897.jpg)  
Label: Gyoza Similarity:0.89

![](images/72252ec5817681f4a1f00c08ec760603174aa96012b3b794ffb227e9ea26d9e4.jpg)  
Label: Gyoza Similarity:0.88  
Figure 1. Illustration of our semantic robustness certificates for VLMs. Each column specifies a target semantic with a text proxy. The certified prediction-invariant intervals are visualized over a normalized semantic extent $\varphi \in [ 0 , 1 ]$ . Nearest images are retrieved from dataset via similarity to the transformed embedding (φ = 1) as visual references, with labels and similarities shown.

With the established transformation, we develop a certification framework over the semantic extent. We characterize the decision boundary of VLM classifiers, where the embedding space is partitioned into Voronoi decision regions. The closed-form decision boundary allows us to analytically determine prediction changes under the transformation. Our framework produces a precise partition of the extent range into prediction-invariant intervals, with each interval annotated by its predicted class. Figure 1 visualizes predictioninvariant intervals certified over the semantic extent φ. For the target semantic triangular, increasing φ strengthens the triangular attribute of the input Gyoza. The certificate shows that the prediction remains Gyoza under this transformation for $\varphi < 0 . 7 7$ and flips to Samosa beyond it. Our framework is evaluated on both generated and real-world images under semantic variation across diverse domains. Results show that our semantic transformations remain consistent with the intended semantic variation and accurately capture prediction changes along the semantic extent for VLMs. Overall, our work is the first to certify semantic robustness without requiring any additional data for each variation. This provides a practical basis for downstream applications to monitor semantic drift, diagnose failure modes under semantic variations, and characterize the evolution of a model’s semantic understanding.

Our main contributions are summarized as follows.

1. We leverage text prompts as semantic proxies to formalize semantic transformations for VLMs.  
2. By characterizing a VLM’s decision boundary, our framework certifies precise prediction-invariant intervals.  
3. Evaluations on both synthetic and real-world data show that our transformations align with the target semantics and that the certificates match prediction changes.

## 2. Related Work

Robustness in VLMs. Vision-language models connect visual inputs with natural-language supervision, supporting zero-shot recognition, open-vocabulary segmentation, and visual reasoning (Radford et al., 2021; Li et al., 2022; Alayrac et al., 2022; Zou et al., 2023). Despite this flexibility, VLM predictions can degrade under out-of-distribution inputs and adversarial perturbations (Schlarmann et al., 2024; Zhang et al., 2024a; Zhu et al., 2025). In response, existing work has studied VLM robustness through distribution-shift adaptation (Ming et al., 2022; Shu et al., 2023), adversarial and visual-security analysis (Zhao et al., 2023; Schlarmann et al., 2024; Li et al., 2025a; Xu et al., 2025), and multimodal optimization or distillation (Zhang et al., 2024b; Li et al., 2025b; Zhou et al., 2025). Related explanation methods connect classifier predictions to human-interpretable concepts or local evidence (Kim et al., 2018; Yang et al., 2023a;b), while recent VLM-based counterfactual methods use embedding-space semantic structure to explain classifier behavior (Kim et al., 2023). Despite known modality gaps (Liang et al., 2022), recent analyses further suggest that VLM representation spaces encode semantic structures useful for interpretation (Bhalla et al., 2024; Sonthalia et al., 2025). These studies improve robustness or interpret model behavior under observed shifts, whereas our work provides closed-form certificates of prediction-invariant intervals over a language-specified semantic extent.

Robustness Certification. Robustness certification aims to certify prediction invariance under a specified set of input transformations. Probabilistic approaches leverage statistical inference to provide robustness guarantees with confidence (Lecuyer et al., 2019; Cohen et al., 2019). Representative methods, including PixelDP (Lecuyer et al., 2019), randomized smoothing (Cohen et al., 2019), and TSS (Li et al., 2021), certify robustness by bounding class probabilities under randomized noise or transformations. In contrast to probabilistic guarantees, deterministic incomplete verifiers, such as AI2 (Gehr et al., 2018), DeepPoly (Singh et al., 2019), CROWN (Zhang et al., 2018), and PRIMA (Muller¨ et al., 2022), employ convex relaxations or abstract domains to provide sound but conservative guarantees. To mitigate the precision loss of relaxations, complete verifiers such as ReluVal (Wang et al., 2018), β-CROWN with branch and bound (Wang et al., 2021), and MN-BaB (Ferrari et al., 2022) employ iterative refinement strategies that systematically partition the search space and guarantee exactness. Building on ExactLine (Sotoudeh & Thakur, 2019), Approx-Line (Mirman et al., 2021) and GCERT (Yuan et al., 2023) certify robustness over semantic variations represented in a generative model’s latent space.

Input Transformations. A certificate is defined with respect to allowable transformations that determine the input variations it captures. Pixel-level certificates model transformations as an $L _ { p }$ ball to capture worst-case pixel perturbations. For explicitly parameterized transformations, methods such as DeepG (Balunovic et al., 2019) and GeoRobust (Wang et al., 2023a) model transformations with closedform geometric parameterizations, enabling certification under affine transforms. For discrete or black-box settings, domain-specific transformations have been explored, including embedding-based transformations in DeepT (Bonaert et al., 2021) for NLP and distributional transformations in CC-Cert (Pautov et al., 2022). However, transformation models based on pixel-level perturbations or explicit geometric parameterizations cannot capture semantic-level variations that lack tractable closed-form descriptions, such as weather conditions or facial features. Consequently, methods such as ApproxLine (Mirman et al., 2021) and GCERT (Yuan et al., 2023) represent semantic transformations in the latent space of generative models, which can encode more complex semantic variations for certification. However, training such generative models requires sufficient in-domain data under the target semantic variation. In contrast, we model semantics in the multimodal embedding space of VLMs, enabling open-vocabulary semantics and supporting broad semantic coverage across domains.

## 3. Problem Statement

In this work, we focus on robustness certification for VLMs. VLMs are typically built on dual encoders that jointly learn visual and textual representations in a shared unit embedding space (Radford et al., 2021; Li et al., 2022; Alayrac et al., 2022). Let $\mathbb { S } ^ { d - 1 } : = \{ e \in \mathbb { R } ^ { d } : \| e \| _ { 2 } = 1 \}$ denote the unit embedding sphere in VLMs. For an image x and a prompt set $\{ t _ { c } : c \in \mathcal { C } \}$ over a label set C, VLMs map x and $t _ { c }$ to a shared embedding space through a visual encoder $f _ { \mathrm { i m g } }$ and a textual encoder ftext, producing embeddings in $\mathbb { S } ^ { d - \bar { 1 } }$ . Denoting the embeddings $z : = f _ { \mathrm { i m g } } ( x )$ and $u _ { c } : = f _ { \mathrm { t e x t } } ( t _ { c } )$ , the VLM classifier $f$ acts on the shared embedding space as

$$
f (z) := \arg \max _ {c \in \mathcal {C}} \langle z, u _ {c} \rangle , \tag {1}
$$

where $\langle \cdot , \cdot \rangle$ denotes the Euclidean inner product, which equals cosine similarity since all embeddings lie on $\mathbb { S } ^ { d - 1 }$ .

Our Objective. For an input x with a source semantic a, we formalize its variation from a to a target semantic $a ^ { \prime }$ in the embedding space using a transformation $\gamma : [ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]  \mathbb { S } ^ { d - 1 }$ parameterized by an extent $\varphi \in \mathbb { R }$ Here, $\varphi _ { a }$ denotes the source extent, and $\varphi _ { a ^ { \prime } }$ specifies the target extent to be certified. For example, in Figure 1, when the semantic variation is the triangular shape of a Gyoza, $\gamma$ formalizes the strength of the triangular attribute, and a larger $\varphi$ corresponds to a more triangular appearance of the input. Our goal is to certify whether the prediction remains invariant for all $\varphi \in [ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ under γ.

Semantic Shift Model. Consider admissible semantic transformations $\gamma ( \varphi ; z )$ of embedding z. We define the prediction for an input to be semantically robust over an extent interval $[ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ if it remains invariant along this range, i.e.,

$$
f (\gamma (\varphi ; z)) = f (z), \quad \forall \varphi \in [ \varphi_ {a}, \varphi_ {a ^ {\prime}} ], \tag {2}
$$

where $\gamma ( \varphi ; z )$ denotes the transformed embedding at extent $\varphi .$ Our framework certifies a labeled partition of the entire extent range $[ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ into subintervals $[ \varphi _ { \ell } , \varphi _ { \ell + 1 } )$ such that $f ( \gamma ( \varphi ; z ) )$ is constant on each subinterval. The resulting labeled partition provides a complete robustness certificate under $\gamma _ { : }$ , in that each subinterval is maximal with a constant predicted label. For a fixed embedding z, we write $\gamma ( \varphi ) : =$ $\gamma ( \varphi ; z )$ for brevity in what follows.

## 4. Methodology

In this section, we develop our robustness certification framework by (i) characterizing semantics in the shared VLM embedding space, (ii) constructing a semantic transformation on input embeddings, and (iii) certifying semantic robustness under semantic variations.

## 4.1. Structured Semantics in the Embedding Space

We begin by characterizing how semantics are represented in the VLM embedding space, and then show that each semantic variation can be confined to an embedding subspace.

## 4.1.1. ADDITIVE SEMANTICS IN EMBEDDING SPACE

A VLM maps an image x and text t to unit embeddings $z , u \in \mathbb { S } ^ { d - 1 }$ in a shared embedding space. For an input visual embedding, VLMs make predictions by comparing similarities to textual embeddings of class labels. Therefore, the prediction rule of VLMs induces a vector representation of semantics in the embedding space. Let $v _ { a } \in \bar { \mathbb { S } } ^ { d - 1 }$ denote the embedding vector corresponding to semantic a. Consistent with the similarity-based prediction rule of VLMs, we adopt the following assumption to quantify semantic strength in the embedding space.

Assumption 4.1 (Similarity-based Semantic Strength). For any semantic a with the embedding $v _ { a } .$ , the semantic strength of a at any query embedding $e ~ \in ~ \mathbb { S } ^ { d - 1 }$ is measured by cosine similarity to $v _ { a }$ .

Under Assumption 4.1, for any query embedding $e \in \mathbb { S } ^ { d - 1 }$ , we define its semantic strength with respect to a as

$$
D _ {a} (e) := \langle e, v _ {a} \rangle \in [ - 1, 1 ]. \tag {3}
$$

This definition specifies the semantics in the embedding space induced by the VLM similarity geometry.

Since semantic strength is measured by cosine similarity, semantic strengths are additive under linear combinations of embeddings. We formalize this property below.

Remark 4.2 (Additive Semantic Strength). For any semantic embedding $v _ { a } ,$ , if a query embedding e admits a linear decomposition over semantics $\begin{array} { r } { \{ v _ { a _ { i } } \} , \mathrm { i } . \mathbf { e } . , e = \sum _ { i } \alpha _ { i } v _ { a _ { i } } } \end{array}$ , then its semantic strength with respect to a decomposes additively as $\begin{array} { r } { D _ { a } ( e ) = \sum _ { i } \alpha _ { i } D _ { a } ( v _ { a _ { i } } ) } \end{array}$ .

This additive property of semantics shows that a semantic can be fully represented and interpreted through the linear decomposition of its embedding in VLMs. Such a property is absent in conventional neural networks whose predictions are not based on a similarity rule.

## 4.1.2. SEMANTIC PLANE

To specify a semantic variation, we consider a pair of semantics $( a , a ^ { \prime } )$ with embeddings $v _ { a }$ and $v _ { a ^ { \prime } }$ in the shared embedding space. Assume that $v _ { a }$ and $v _ { a ^ { \prime } }$ are linearly independent, i.e., $v _ { a } \not \in \operatorname { s p a n } \{ v _ { a ^ { \prime } } \}$ . We define the two-dimensional subspace spanned by $v _ { a }$ and $v _ { a ^ { \prime } }$ as

$$
\mathcal {P} _ {a, a ^ {\prime}} := \operatorname{span} \left\{v _ {a}, v _ {a ^ {\prime}} \right\} \subset \mathbb {R} ^ {d}, \tag {4}
$$

where we refer to $\mathcal { P } _ { a , a ^ { \prime } }$ as the semantic plane of $( a , a ^ { \prime } )$ . The plane $\mathcal { P } _ { a , a ^ { \prime } }$ isolates variations driven by a and $a ^ { \prime } ,$ where semantic variations are captured solely by the semantic strengths to a and $a ^ { \prime }$ . Formally, we have the following remark.

Remark 4.3 (Semantic Plane). Let semantic embeddings $v _ { a } , v _ { a ^ { \prime } } \in \mathbb { S } ^ { d - 1 }$ be linearly independent and let $\mathcal { P } _ { a , a ^ { \prime } } =$ span $\{ v _ { a } , v _ { a ^ { \prime } } \}$ . If an embedding $e \in \mathbb { R } ^ { d }$ varies only in the semantic strengths to $v _ { a }$ and $v _ { a ^ { \prime } }$ , then e necessarily lies in the semantic plane $\mathcal { P } _ { a , a ^ { \prime } }$ .

Remark 4.3 shows that if only the semantic specified by $( a , a ^ { \prime } )$ is varied, as reflected by changes in the strengths to $v _ { a }$ and $v _ { a ^ { \prime } }$ , the variation can be analyzed within the unique twodimensional subspace $\mathcal { P } _ { a , a ^ { \prime } }$ . In other words, controlling semantic strengths to $v _ { a }$ and $v _ { a ^ { \prime } }$ confines the embedding variation to the semantic plane.

## 4.2. Semantic Transformation

In this section, we construct a transformation of the input embedding to model a semantic variation.

## 4.2.1. TEXT PROXY FOR SEMANTICS

Under contrastive training, text embeddings serve as anchors in the embedding space, encouraging matched image embeddings to align with them while separating mismatched embeddings (Radford et al., 2021; Jia et al., 2021). This contrastive objective aligns images and text in a shared embedding space, enabling language as a natural interface for specifying semantics. In contrast to text embeddings, image embeddings summarize the full image content and thus typically entangle multiple semantic factors. This motivates the use of text prompts as a proxy to specify semantics.

To specify a semantic variation, we use a pair of text prompts $( t _ { a } , t _ { a ^ { \prime } } )$ to represent the source semantic a and the target semantic $a ^ { \prime }$ . We denote their embeddings by $u _ { a } : = f _ { \mathrm { t e x t } } ( t _ { a } )$ and $u _ { a ^ { \prime } } : = f _ { \mathrm { t e x t } } ( t _ { a ^ { \prime } } )$ , which serve as text proxies for specifying $v _ { a }$ and $v _ { a ^ { \prime } }$ . We define the semantic plane induced by this pair as $\mathcal { P } _ { a , a ^ { \prime } } : = \operatorname { s p a n } \{ u _ { a } , u _ { a ^ { \prime } } \}$ . By Remark 4.3, if an embedding varies only through its semantic strengths with respect to $u _ { a }$ and $u _ { a ^ { \prime } }$ , then the variation is confined to $\mathcal { P } _ { a , a ^ { \prime } }$ . Therefore, $\mathcal { P } _ { a , a ^ { \prime } }$ captures the degrees of freedom of the variation specified by $( t _ { a } , t _ { a ^ { \prime } } )$ . This reduces the specification of a semantic variation to selecting a prompt pair and analyzing the variation within the spanned plane $\mathcal { P } _ { a , a ^ { \prime } }$ .

## 4.2.2. SEMANTIC TRANSFORMATION

Given the semantic plane $\mathcal { P } _ { a , a ^ { \prime } }$ , we decompose an image embedding z of input x as

$$
z = z _ {\parallel} + z _ {\perp}, \quad z _ {\parallel} \in \mathcal {P} _ {a, a ^ {\prime}}, \quad z _ {\perp} \perp \mathcal {P} _ {a, a ^ {\prime}}, \tag {5}
$$

where $z _ { \parallel }$ is the orthogonal projection of z onto $\mathcal { P } _ { a , a ^ { \prime } }$ and $z _ { \bot }$ is the orthogonal component. Since $u _ { a } , u _ { a ^ { \prime } } \in \mathcal P _ { a , a ^ { \prime } }$ and $z _ { \bot } \ \bot \ \mathcal { P } _ { a , a ^ { \prime } }$ , the semantic strengths with respect to $( u _ { a } , u _ { a ^ { \prime } } )$ depend only on $z _ { \parallel }$ , i.e. $\langle z , u _ { a } \rangle = \langle z _ { \parallel } , u _ { a } \rangle$ and $\langle z , u _ { a ^ { \prime } } \rangle = \langle z _ { \parallel } , u _ { a ^ { \prime } } \rangle$ . In addition, when $u _ { a }$ and $u _ { a ^ { \prime } }$ are linearly independent, each $z _ { | | } \in \mathcal { P } _ { a , a ^ { \prime } }$ admits a representation $z _ { \parallel } = \alpha u _ { a } + \beta u _ { a ^ { \prime } }$ ′ for $( \alpha , \overset { \cdot } { \beta } ) \in \mathbb { R } ^ { 2 }$ .

Equivalently, for any visual embedding $z \in \mathbb { S } ^ { d - 1 }$ , we have

$$
z = \underbrace {\left(\alpha u _ {a} + \beta u _ {a ^ {\prime}}\right)} _ {\text { target   semantic   component   in } \mathcal {P} _ {a, a ^ {\prime}}} + \underbrace {z _ {\perp}} _ {\text { semantics   independent   of } (a, a ^ {\prime})}, \tag {6}
$$

where $( \alpha , \beta )$ are the coordinates of the component in $\mathcal { P } _ { a , a ^ { \prime } }$ and $z _ { \bot } \bot \mathcal { P } _ { a , a ^ { \prime } }$ captures the remaining semantics of z that are independent of $( a , a ^ { \prime } )$ .

To parameterize the semantic transformation, we establish an orthonormal basis of $\mathcal { P } _ { a , a ^ { \prime } }$ from $( u _ { a } , u _ { a ^ { \prime } } )$ as

$$
e _ {1} := u _ {a}, \quad e _ {2} := \frac {u _ {a ^ {\prime}} - \langle u _ {a ^ {\prime}} , u _ {a} \rangle u _ {a}}{\| u _ {a ^ {\prime}} - \langle u _ {a ^ {\prime}} , u _ {a} \rangle u _ {a} \| _ {2}}, \tag {7}
$$

where $( e _ { 1 } , e _ { 2 } )$ forms an orthonormal basis of $\mathcal { P } _ { a , a ^ { \prime } }$ .

![](images/2cf20ac35f961d79f4a2df8c112b26f48214dd219176b9e52f3196d7b102e7eb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  A["Orthogonal component z⊥"] --> B["Image embedding z"]
  B --> C["Target semantic component z∥"]
  C --> D["Semantic transformation γ(φ) := L(cos φe1 + sin φe2) + z⊥"]
  D --> E["Semantic plane P_{a,a}"]
  E --> F["Text embedding u_a'"]
  F --> G["Basis vector e2"]
  G --> H["In-plane transformation L(cos φe1 + sin φe2)"]
  H --> I["Text embedding u_a'/basis vector e1"]
  I --> J["φ^src"]
  J --> K["Semantic transformation γ(φ) := L(cos φe1 + sin φe2) + z⊥"]
```
</details>

Figure 2. Illustration of the semantic transformation in a threedimensional visualization of the VLM embedding space.

With the basis $( e _ { 1 } , e _ { 2 } )$ , embeddings in $\mathcal { P } _ { a , a ^ { \prime } }$ can be parameterized by an extent $\varphi$ that controls the relative strengths with respect to $( u _ { a } , u _ { a ^ { \prime } } )$ . The source semantic extent $\varphi _ { a } \in ( - \pi , \pi ]$ in $\mathcal { P } _ { a , a ^ { \prime } }$ is defined from the orthogonal projection $z _ { \parallel }$ as

$$
\varphi_ {a} := \operatorname{atan2} (\langle z _ {\parallel}, e _ {2} \rangle , \langle z _ {\parallel}, e _ {1} \rangle). \tag {8}
$$

We assume a target extent $\varphi _ { a ^ { \prime } } \in ( - \pi , \pi ]$ that specifies the target semantic of the variation. We define the transformation of semantic variation over extent $\varphi \in [ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ as

$$
\gamma (\varphi) := r \big (\cos \varphi e _ {1} + \sin \varphi e _ {2} \big) + z _ {\perp}, \quad \varphi \in [ \varphi_ {a}, \varphi_ {a ^ {\prime}} ], (9)
$$

where $r : = \| z _ { \parallel } \| _ { 2 }$ is the magnitude of the component of $z$ within $\mathcal { P } _ { a , a ^ { \prime } }$ . The extent $\varphi$ controls this component within the plane, and $z _ { \bot }$ is kept unchanged to preserve the $\mathrm { s e - }$ mantics independent of $( a , a ^ { \prime } )$ . When $\varphi = \varphi _ { a } ,$ we have $\gamma ( \varphi _ { a } ) = z _ { \parallel } + z _ { \perp } = z$ . Moreover, $\gamma ( \varphi ) \in \mathbb { S } ^ { d - 1 }$ for all $\varphi \in [ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ since $r ( \cos \varphi e _ { 1 } + \sin \varphi e _ { 2 } )$ has norm $r$ for all $\varphi$ and is orthogonal $\tan z _ { \perp }$ .

Figure 2 illustrates an example construction of the semantic transformation in the VLM embedding space. For each extent $\varphi , \gamma ( \varphi )$ varies only the component within $\mathcal { P } _ { a , a ^ { \prime } }$ , which determines the semantic strengths with respect to text embeddings $( u _ { a } , u _ { a ^ { \prime } } )$ , while keeping $z _ { \bot }$ unchanged to preserve the remaining semantics that are independent of $( a , a ^ { \prime } )$ . As $\varphi$ varies, the transformation adjusts the relative strengths to a and $a ^ { \prime }$ within $\mathcal { P } _ { a , a ^ { \prime } }$ while preserving $z _ { \bot }$ .

## 4.2.3. SEMANTIC EXTENT

The semantic transformation $\gamma ( \varphi )$ is defined over an extent specified by a target extent $\varphi _ { a ^ { \prime } }$ . However, semantic strength $( \mathrm { e . g . }$ ., how $ { \mathbf { \hat { b } } }  { \mathbf { i } }  { \mathbf { g } } ^ { \flat }$ is “big”) does not admit an objective canonical scale, leaving $\varphi _ { a ^ { \prime } }$ ambiguous.

We therefore consider two practical specifications of $\varphi _ { a ^ { \prime } }$ under the text-specified and image-specified settings. In the text-specified setting, we anchor $\varphi _ { a ^ { \prime } }$ using the target prompt embedding $u _ { a ^ { \prime } }$ as $\varphi _ { a ^ { \prime } } : = \mathrm { a t a n 2 } ( \langle u _ { a ^ { \prime } } , e _ { 2 } \rangle , \langle u _ { a ^ { \prime } } , e _ { 1 } \rangle )$ . This specification treats text as a semantic reference calibrated in the VLM’s similarity geometry used for prediction. In the image-specified setting, we anchor $\varphi _ { a ^ { \prime } }$ using a reference image $x ^ { \prime }$ exhibiting the target semantics. Let $z ^ { \prime } : = f _ { \mathrm { i m g } } ( x ^ { \prime } )$ and $z _ { \parallel } ^ { \prime }$ be its projection onto $\mathcal { P } _ { a , a ^ { \prime } }$ . We set $\varphi _ { a ^ { \prime } } : = \mathrm { a t a n 2 } ( \langle z _ { \parallel } ^ { \prime } , e _ { 2 } \rangle , \langle z _ { \parallel } ^ { \prime } , e _ { 1 } \rangle )$ . This uses direct visual evidence in the same geometry, providing a concrete specification of semantic strength when such a reference is available. These two specifications make the abstract semantic extent explicit and reproducible by yielding a precisely specified target extent, which fixes the domain of $\varphi$ over which the subsequent certification is conditioned.

## 4.3. Semantic Robustness Certification

With the established transformation of semantic variations, we develop a certification framework to determine the prediction changes under the variation.

## 4.3.1. DECISION GEOMETRY

Consider a finite set of textual labels $\{ u _ { c } \} _ { c \in \mathcal { C } } \subset \mathbb { S } ^ { d - 1 }$ . Under the VLM prediction rule, each input embedding $e \in$ $\mathbb { S } ^ { d - 1 }$ is assigned the label $f ( e )$ . This classification rule partitions $\mathbb { S } ^ { d - 1 }$ into Voronoi cells as

$$
\mathcal {V} _ {c} := \left\{e \in \mathbb {S} ^ {d - 1}: \langle e, u _ {c} \rangle \geq \langle e, u _ {c ^ {\prime}} \rangle   \forall c ^ {\prime} \in \mathcal {C} \right\}, \tag {10}
$$

with decision boundaries induced by pairwise bisectors

$$
\mathcal {B} _ {c, c ^ {\prime}} := \left\{e \in \mathbb {S} ^ {d - 1}: \langle e, u _ {c} - u _ {c ^ {\prime}} \rangle = 0 \right\}, \quad c \neq c ^ {\prime}. \tag {11}
$$

Under a semantic transformation $\gamma ( \varphi )$ , class flips can occur only at extents $\varphi$ where $\gamma ( \varphi )$ intersects some $\boldsymbol { B } _ { c , c ^ { \prime } }$ .

Substituting the semantic transformation $\gamma ( \varphi )$ defined in Eq. (9) into the pairwise margin gives

$$
\begin{array}{l} m _ {c, c ^ {\prime}} (\varphi) = \left\langle r (\cos \varphi e _ {1} + \sin \varphi e _ {2}) + z _ {\perp}, u _ {c} - u _ {c ^ {\prime}} \right\rangle \\ = A _ {c, c ^ {\prime}} \cos \varphi + B _ {c, c ^ {\prime}} \sin \varphi + C _ {c, c ^ {\prime}}, \tag {12} \\ \end{array}
$$

where $A _ { c , c ^ { \prime } } = r \langle e _ { 1 } , u _ { c } - u _ { c ^ { \prime } } \rangle , B _ { c , c ^ { \prime } } = r \langle e _ { 2 } , u _ { c } - u _ { c ^ { \prime } } \rangle$ , and $C _ { c , c ^ { \prime } } = \langle z _ { \perp } , u _ { c } - u _ { c ^ { \prime } } \rangle$ . Boundary crossings along $\gamma ( \varphi )$ are therefore given in closed form by solving $m _ { c , c ^ { \prime } } ( \varphi ) = 0$ .

## 4.3.2. CERTIFICATIONS

Our framework certifies prediction invariance over the extent by partitioning $[ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ into subintervals on which $f ( \gamma ( \varphi ) )$ ) remains unchanged. As $\varphi$ varies, label changes can occur only at extents where $m _ { c , c ^ { \prime } } ( \varphi ) = 0$ for some pair of classes $( c , c ^ { \prime } )$ . By Eq. (12), each $m _ { c , c ^ { \prime } } ( \varphi )$ has a closed form. Candidate change extents are therefore obtained by solving $m _ { c , c ^ { \prime } } ( \varphi ) = 0$ for each $( c , c ^ { \prime } )$ and retaining the solutions in $[ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ . Collecting these solutions over all label pairs $\{ \varphi _ { \ell } \} _ { \ell = 0 } ^ { L }$ with $\varphi _ { 0 } = \varphi _ { a }$ and $\varphi _ { L } = \varphi _ { a ^ { \prime } }$ that contains all extents at which a label change can occur. This sequence induces a complete partition of the extent range into open intervals $( \varphi _ { \ell } , \varphi _ { \ell + 1 } ) \subset [ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ on which the prediction $f ( \gamma ( \varphi ) )$ is constant. Denoting this constant label by $y _ { \ell } : = f ( \gamma ( \varphi ) )$ for any $\varphi \in \left( \varphi _ { \ell } , \varphi _ { \ell + 1 } \right)$ , the certified collection of labeled intervals is

$$
\mathcal {S} := \big \{\big ((\varphi_ {\ell}, \varphi_ {\ell + 1}), y _ {\ell} \big): \ell = 0, \dots , L - 1 \big \}. \tag {13}
$$

Each pair $\big ( ( \varphi _ { \ell } , \varphi _ { \ell + 1 } ) , y _ { \ell } \big )$ is a prediction-invariant interval under the target semantic variation, such that $f ( \gamma ( \varphi ) ) = y \ell$ for any $\varphi \in \left( \varphi _ { \ell } , \varphi _ { \ell + 1 } \right)$ .

We further report the prediction invariance probability, which aggregates robustness over the extent range. Fix a prediction label $\hat { y } = f ( \gamma ( \varphi _ { a } ) )$ , and draw $\varphi$ uniformly from $[ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ . The probability that the prediction remains $\hat { y }$ under the semantic transformation is

$$
\mathbb {P} \big [ f (\gamma (\varphi)) = \hat {y} \big ] = \frac {1}{\varphi_ {a ^ {\prime}} - \varphi_ {a}} \sum_ {\ell : y _ {\ell} = \hat {y}} \big (\varphi_ {\ell + 1} - \varphi_ {\ell} \big), \tag {14}
$$

which measures the total fraction of the semantic extent range on which the prediction is invariant.

## 4.3.3. CERTIFICATE BOUNDS UNDER MISALIGNMENT

Our certificates are conditioned on the similarity-based semantic strength specification in Assumption 4.1. Consequently, cross-modal mismatch can introduce uncertainty in the resulting certificates. We model this effect via a bounded misalignment budget δ and derive conditions, parameterized by δ, under which the certificates remain valid.

Assumption 4.4 (Bounded Misalignment). There exists $\delta \geq 0$ such that for any semantically matched pair with embeddings $z , u \in \mathbb { S } ^ { d - 1 }$ under the target semantic variation, we have $\| z - u \| _ { 2 } \leq \delta .$ .

For boundary crossings $m _ { c , c ^ { \prime } } ( \varphi )$ , we consider any embedding e within a δ-neighborhood of $\gamma ( \varphi )$ at each extent $\varphi \colon$

$$
\Gamma_ {\delta} (\varphi) := \big \{e \in \mathbb {S} ^ {d - 1}: \| e - \gamma (\varphi) \| _ {2} \leq \delta \big \}. \tag {15}
$$

For any $e \in \Gamma _ { \delta } ( \varphi )$ , we define $\tilde { m } _ { c , c ^ { \prime } } ( \varphi ; e ) : = \langle e , u _ { c } - u _ { c ^ { \prime } } \rangle$ . For brevity, we write $\tilde { m } _ { c , c ^ { \prime } } ( \varphi )$ for $\tilde { m } _ { c , c ^ { \prime } } ( \varphi ; e )$ . We next quantify how misalignment affects pairwise margins.

Lemma 4.5 (Bounded Margin Gap under Misalignment). For any $\varphi \in [ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ , any $c \neq c ^ { \prime } .$ , and any $e \in \Gamma _ { \delta } ( \varphi )$ ,

$$
\left| \tilde {m} _ {c, c ^ {\prime}} (\varphi) - m _ {c, c ^ {\prime}} (\varphi) \right| \leq \delta \left\| u _ {c} - u _ {c ^ {\prime}} \right\| _ {2}. \tag {16}
$$

Let ${ \hat { y } } ( \varphi ) : = f ( \gamma ( \varphi ) )$ . We derive a stability condition ensuring that $f ( e ) = { \hat { y } } ( \varphi )$ for all $e \in \Gamma _ { \delta } ( \varphi )$ .

Proposition 4.6 (Stability under Misalignment). Under Assumption 4.4, fix any $\varphi \in \bigl [ \varphi _ { a } , \varphi _ { a ^ { \prime } } \bigr ]$ and let $\hat { y } ( \varphi ) : =$ $f ( \gamma ( \varphi ) )$ . $I f m _ { \hat { y } ( \varphi ) , c ^ { \prime } } ( \varphi ) > \delta \| u _ { \hat { y } ( \varphi ) } - u _ { c ^ { \prime } } \| _ { 2 } f o r$ all $c ^ { \prime } \neq$ $\hat { y } ( \varphi )$ , then $f ( e ) = { \hat { y } } ( \varphi )$ holds for all $e \in \Gamma _ { \delta } ( \varphi )$ .

Using the closed-form margin $m _ { c , c ^ { \prime } } ( \varphi )$ , we further localize the extents where a boundary crossing may occur under misalignment by defining the uncertainty set

$$
\mathcal {U} _ {c, c ^ {\prime}} (\delta) := \left\{\varphi \in [ \varphi_ {a}, \varphi_ {a ^ {\prime}} ]: | m _ {c, c ^ {\prime}} (\varphi) | \leq \varepsilon_ {c, c ^ {\prime}} \right\}, \tag {17}
$$

where $\varepsilon _ { c , c ^ { \prime } } : = \delta \| u _ { c } - u _ { c ^ { \prime } } \| _ { 2 }$ is the margin tolerance. We next localize boundary-crossing uncertainty to $\mathcal { U } _ { c , c ^ { \prime } } ( \delta )$ .

Lemma 4.7 (Crossing Localization). For any $\varphi \notin \mathcal { U } _ { c , c ^ { \prime } } ( \delta )$ and any $e \in \Gamma _ { \delta } ( \varphi ) , m _ { c , c ^ { \prime } } ( \varphi )$ and $\tilde { m } _ { c , c ^ { \prime } } ( \varphi )$ have the same sign. Consequently, any extent φ satisfying $\tilde { m } _ { c , c ^ { \prime } } ( \varphi ) = 0$ must lie in $\mathcal { U } _ { c , c ^ { \prime } } ( \delta )$ .

Using the cosine form of $m _ { c , c ^ { \prime } } ( \varphi ) , \mathcal { U } _ { c , c ^ { \prime } } ( \delta )$ is given by

$$
\left| R _ {c, c ^ {\prime}} \cos (\varphi - \psi_ {c, c ^ {\prime}}) + C _ {c, c ^ {\prime}} \right| \leq \varepsilon_ {c, c ^ {\prime}}, \tag {18}
$$

where the amplitude $R _ { c , c ^ { \prime } } : = ( A _ { c , c ^ { \prime } } ^ { 2 } + B _ { c , c ^ { \prime } } ^ { 2 } ) ^ { 1 / 2 }$ and the phase $\psi _ { c , c ^ { \prime } } : = \mathrm { a t a n 2 } ( B _ { c , c ^ { \prime } } , A _ { c , c ^ { \prime } } )$ . The boundaries solve $R _ { c , c ^ { \prime } } \cos ( \varphi - \psi _ { c , c ^ { \prime } } ) = - C _ { c , c ^ { \prime } } \pm \varepsilon _ { c , c ^ { \prime } }$ for $\varphi \in [ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ .

Thus, for a given misalignment budget $\delta ,$ we identify the extents that are guaranteed prediction-invariant within $\Gamma _ { \delta } ( \varphi )$ and localize the remaining uncertainty to $\mathcal { U } _ { c , c ^ { \prime } } ( \delta )$ .

## 5. Experiments

In this section, we evaluate our framework on both controlled synthetic and real-world semantic variations. Experiments are conducted on publicly available CLIP VLMs (Radford et al., 2021). Before presenting the results, we describe the experimental setup below.

Prompts for Semantics. We use text prompts to specify the semantic variations. The prompt pair $( t _ { a } , t _ { a ^ { \prime } } )$ specifies which semantic factors are allowed to vary under the transformation. If the prompts differ beyond the intended attribute, then $\mathcal { P } _ { a , a ^ { \prime } }$ may capture additional semantic factors, and moving within $\mathcal { P } _ { a , a ^ { \prime } }$ can leak into unintended semantic variations. In our experiments, we control this effect by using prompt pairs that share the same template and content words, differing only in the attribute token. For each attribute type, we construct a comprehensive list of attribute descriptors. Multiple attribute types are considered to evaluate the generality of our framework. Figure 4 provides examples of the descriptors used for each attribute type.

Baselines. We evaluate against ExactLine, a complete certification method that certifies predictions over the linear interpolation path between two endpoint images (Sotoudeh & Thakur, 2019). ExactLine identifies all decision boundary crossing points along the interpolation path, yielding a complete partition of the range into prediction-invariant intervals. Our work targets semantic transformations in VLMs without requiring auxiliary inputs. Under this setting, ExactLine is the only existing framework we are aware of that provides complete certification without additional inputs or supervision (Mirman et al., 2021; Yuan et al., 2023), and therefore serves as our primary baseline.

![](images/48633b194e64e188bd0ab07758cb8312bef60f95982b548790b7e94ed646d432.jpg)  
Predicted prompt:  
a photo of a wallflower.

Certificates:  
![](images/8baf849ddbe26e411d85b143a77323ad20f5467f40c520c1b3aa521bf4230ac2.jpg)

<details>
<summary>bar chart</summary>

| Flower Type | Wallflower | Trump, Creeper | Snapdrago |
|-------------|------------|----------------|-----------|
| Photo of a red flower | 0.72 | 1.00 | 0.31 |
| Photo of a spiral flower | 0.31 | 1.00 | 0.00 |
| Photo of a front-view photo of a flower | 0.00 | 1.00 | 0.00 |
</details>

Input image (beagle):  
![](images/51e790f19d1daa4dad4457efb252f4faeaa29c59956be2647c2d59983b51cde0.jpg)  
Predicted prompt:  
A photo of a beagle.

Certificates:  
![](images/0ce1f03ef0655791097ed9693facec5d57812abe7d14549273308de6abef314b.jpg)

<details>
<summary>bar chart</summary>

| Dog Type | Beagle | Basset Hound |
| -------- | ------ | ------------ |
| → a photo of a pointy dog | 0.72 | 1.00 |
| → a photo of a gray dog | 0.11 | 0.74 |
| → a close-up photo of a dog | 0.82 | 1.00 |
</details>

Input image (samosa):  
![](images/19249b93c32f867fb7ab3c99d3e85cbe9b0a852b6e7d5c110dde0ab995e4a51e.jpg)  
Predicted prompt:  
A photo of a samosa.

Certificates:  
![](images/f74586334ec8320da1c5b5958407b83efb7cce13d79c9d16cd5d950ad0041f79.jpg)

<details>
<summary>text_image</summary>

Samosa
Beignets
0.62
1.00
→ a photo of a white food
Samosa
Beignets Gyozo
0.653 0.679
1.00
→ a photo of a round food
Samosa
Beignets on plate food
</details>

Figure 3. Illustration of our VLM robustness certificates. Prediction-invariant intervals are completely certified over a normalized semantic extent $\varphi \in [ 0 , 1 ]$ for diverse semantic variations across domains. Text prompts serve as proxies for specifying different target semantics.

<table><tr><td>Color:</td></tr><tr><td>- red</td></tr><tr><td>- orange</td></tr><tr><td>- yellow</td></tr><tr><td>...</td></tr></table>

<table><tr><td>Shape:</td></tr><tr><td>- round</td></tr><tr><td>- oval</td></tr><tr><td>- square</td></tr><tr><td>...</td></tr></table>

<table><tr><td>Material:</td></tr><tr><td>- wooden</td></tr><tr><td>- leather</td></tr><tr><td>- plastic</td></tr><tr><td>...</td></tr></table>

<table><tr><td>Style:</td></tr><tr><td>- sketch</td></tr><tr><td>- cartoon</td></tr><tr><td>- 3d-render</td></tr><tr><td>...</td></tr></table>

<table><tr><td>Texture:</td></tr><tr><td>- smooth</td></tr><tr><td>- glossy</td></tr><tr><td>- rough</td></tr><tr><td>...</td></tr></table>

<table><tr><td>Background:</td></tr><tr><td>- tabletop</td></tr><tr><td>- street</td></tr><tr><td>- highway</td></tr><tr><td>...</td></tr></table>

<table><tr><td>Viewpoint:</td></tr><tr><td>- front view</td></tr><tr><td>- side view</td></tr><tr><td>- close-up</td></tr><tr><td>...</td></tr></table>

<table><tr><td>Illumination:</td></tr><tr><td>- bright</td></tr><tr><td>- dark</td></tr><tr><td>- cool</td></tr><tr><td>...</td></tr></table>

Figure 4. Illustration of descriptors grouped by attribute type. Using a fixed prompt template (e.g., “a photo of a {attribute} class”), we vary only the descriptor to model semantic variations.

Visual Reference Transformation and Metric. Groundtruth semantic variations are difficult to define and annotate, which makes it challenging to directly evaluate whether a specified transformation follows the intended change. To obtain a visual reference, we assume access to an image sequence $\{ x _ { k } \} _ { k = 0 } ^ { K }$ that exhibits the target semantic variation from a to $a ^ { \prime }$ . Let $z _ { k } : = f _ { \mathrm { i m g } } ( x _ { k } ) \in \mathbb { S } ^ { d - 1 }$ be the corresponding normalized embeddings, with z0 as the starting embedding. We fit a visual reference transformation as a great circle arc on the unit sphere, $\gamma ^ { \mathrm { r e f } }$ : $[ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]  \mathbb { S } ^ { d - 1 }$ , by least squares over the samples, min-$\begin{array} { r } { \sum _ { k = 0 } ^ { K } \| \gamma ^ { \mathrm { r e f } } ( \varphi _ { k } ) - z _ { k } \| _ { 2 } ^ { 2 } } \end{array}$ $\gamma ^ { \mathrm { r e f } } ( \varphi _ { a } ) = z _ { 0 }$ We then quantify alignment between the specified semantic transformation $\gamma ( \varphi )$ and the visual reference transformation $\gamma ^ { \mathrm { r e f } } ( \varphi )$ using the discrepancy in their prediction invariance probabilities as defined in Eq. (14). Concretely, we fix a reference label $\hat { y } = f ( \gamma ( \varphi _ { a } ) )$ and compute the prediction invariance probability for each transformation. For each image $x _ { i }$ , let $\gamma _ { i } ^ { \mathrm { r e f } } ( \varphi )$ be the fitted visual reference transformation and $\gamma _ { i } ( \varphi )$ be the specified semantic transformation, and we report the mean absolute discrepancy over all N images as 1N PNi=1 $\begin{array} { r } { \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left| \mathbb { P } \big [ f ( \gamma _ { i } ^ { \mathrm { r e f } } ( \varphi ) ) = \hat { y } \big ] - \mathbb { P } \big [ f ( \gamma _ { i } ( \varphi ) ) = \hat { y } \big ] \right| } \end{array}$ .

## 5.1. Qualitative Evaluation of Semantic Variations

Since semantic variation lacks a standard quantitative ground truth, qualitative analysis plays a necessary role in validating that the evaluation reflects the intended semantic change. Before presenting quantitative results, we therefore provide a qualitative study. In Figure 3, we model different semantic variations by using text prompts and present certificates of prediction-invariant intervals over the normalized extent. The figure shows that our method can track diverse semantic variations and can identify plausible flipped classes under transformations. For instance, under the target semantic round for the input Samosa, in contrast to the example in Figure 1, the prediction flips back to Gyoza. This suggests qualitative alignment between our semantic transformation and real-world semantic variations.

![](images/9f1e5b5052ca2ecbb94e9d29571d8b22f5d1e643605b9741cf7e78bdaa78f140.jpg)

<details>
<summary>line chart</summary>

| Misalignment budget δ | Stable coverage | Empirical invariance | Conditional invariance |
| --------------------- | --------------- | -------------------- | ---------------------- |
| 0.00                  | 1.0             | 1.0                  | 1.0                    |
| 0.02                  | 0.6             | 1.0                  | 1.0                    |
| 0.04                  | 0.4             | 1.0                  | 1.0                    |
| 0.06                  | 0.2             | 1.0                  | 1.0                    |
| 0.08                  | 0.1             | 1.0                  | 1.0                    |
| 0.10                  | 0.0             | 1.0                  | 1.0                    |
</details>

![](images/d3d1bd08b24515bd9b0a1181c201a067a9a4ed546763303272fe0f3ad2c04814.jpg)

<details>
<summary>line chart</summary>

| Misalignment budget δ | Stability fraction |
| --------------------- | ------------------ |
| 0.00                  | 1.0                |
| 0.02                  | 0.5                |
| 0.04                  | 0.3                |
| 0.06                  | 0.2                |
| 0.08                  | 0.1                |
| 0.10                  | 0.0                |
</details>

![](images/e6c03c53596e577118f949af1dbd9dfaacba40e38b5cb2b1e8bbd0338b6116de.jpg)

<details>
<summary>line chart</summary>

| Misalignment budget δ | Stable coverage | Empirical invariance | Conditional invariance |
| --------------------- | --------------- | -------------------- | ---------------------- |
| 0.00                  | 1.0             | 1.0                  | 1.0                    |
| 0.02                  | 0.5             | 1.0                  | 1.0                    |
| 0.04                  | 0.3             | 1.0                  | 1.0                    |
| 0.06                  | 0.2             | 1.0                  | 1.0                    |
| 0.08                  | 0.1             | 1.0                  | 1.0                    |
| 0.10                  | 0.0             | 1.0                  | 1.0                    |
</details>

![](images/cb458c31b6ed0a2d9f8666e05eabfdecf1fd02c51c6a62d4a432bf8ea6be69b0.jpg)

<details>
<summary>line chart</summary>

| Misalignment budget δ | Stability fraction |
| --------------------- | ------------------ |
| 0.00                  | 1.0                |
| 0.02                  | 0.4                |
| 0.04                  | 0.0                |
| 0.06                  | 0.0                |
| 0.08                  | 0.0                |
| 0.10                  | 0.0                |
</details>

Figure 5. Evaluation of certificate bounds on an ImageNet subset. We sweep the misalignment budget δ for semantic variations driven by color, style, texture, and background. Stable coverage is the fraction of extents that are certified prediction-invariant. Empirical invariance is the observed fraction of invariant extents under sampled perturbations within the δ-neighborhood. Conditional invariance reports empirical invariance restricted to certified extents.

## 5.2. Evaluation of Certificate Bounds

We evaluate the certificate bounds under semantic misalignment during certification. Since estimating real-world misalignment would require additional annotation and calibration beyond the scope of this work, we evaluate userspecified misalignment budgets. For each semantic variation, we sweep the misalignment budget δ and sample perturbations within the corresponding δ-neighborhood at each extent, then measure whether the prediction matches the nominal prediction along the transformation and aggregate over the extent range. Figure 5 reports stable coverage, empirical invariance, and conditional invariance. Across attributes, conditional invariance remains close to one as δ increases, supporting the validity of the bound. As expected, stable coverage decreases with larger δ because certifying invariance under stronger misalignment requires larger pairwise margins. Empirical invariance stays high over the same range, suggesting that the bound is conservative yet reliable.

Table 1. Comparison of mean absolute discrepancy among ExactLine, our text-specified transformation (T-Spec), and our image-specified transformation (I-Spec) under synthetic semantic variations, covering both in-domain (ID) attributes and out-of-domain (OOD) attributes.  
(a) Oxford Pets.

<table><tr><td rowspan="2"></td><td colspan="2">ID</td><td rowspan="2">OODTexture</td></tr><tr><td>Color</td><td>Background</td></tr><tr><td>ExactLine</td><td>12.6%</td><td>10.2%</td><td>18.1%</td></tr><tr><td>T-Spec</td><td>6.9%</td><td>7.3%</td><td>7.4%</td></tr><tr><td>I-Spec</td><td>4.1%</td><td>3.2%</td><td>6.7%</td></tr></table>

(b) Flowers102.

<table><tr><td rowspan="2"></td><td colspan="2">ID</td><td colspan="2">OOD</td></tr><tr><td>Color</td><td>View</td><td>Color</td><td>Shape</td></tr><tr><td>ExactLine</td><td>8.4%</td><td>12.6%</td><td>10.1%</td><td>19.3%</td></tr><tr><td>T-Spec</td><td>6.5%</td><td>8.2%</td><td>9.4%</td><td>8.7%</td></tr><tr><td>I-Spec</td><td>2.3%</td><td>4.0%</td><td>7.5%</td><td>6.6%</td></tr></table>

(c) Food101.

<table><tr><td></td><td>View</td><td>ID Background</td><td>OOD Color</td></tr><tr><td>ExactLine</td><td>14.6%</td><td>10.7%</td><td>9.3%</td></tr><tr><td>T-Spec</td><td>0.0%</td><td>6.2%</td><td>9.6%</td></tr><tr><td>I-Spec</td><td>0.0%</td><td>3.9%</td><td>6.8%</td></tr></table>

Prompt: A photo of a spiky wallflower.  
![](images/58e52c90e0a3925af6fc34c13e5a30788b6ce4de51a3f2690e29b6eddca67c3e.jpg)

<details>
<summary>natural_image</summary>

Collage of six photos showing yellow and green flowers, some shaped like bread slices, with no visible text or symbols.
</details>

Prompt: A photo of a round samosa.  
![](images/329e601f5ba4d41c1d3d6f3775d8e44bec0cdeaf0b7df9c5b1feb396003f623c.jpg)  
(a) Synthetic semantic variations.

Prompt: A rear-view photo of an Acura TL.  
![](images/508be7bd2eab93838c788323b8d9de50881ac8b0032763092dc9c2ad2261fd2f.jpg)

<details>
<summary>text_image</summary>

Prompt: A dark photo of horse riding.
Prompt: A photo of a hamburger on a plate.
</details>

(b) Real-world semantic variations.  
Figure 6. Images of synthetic and real-world semantic variations.

## 5.3. Evaluation on Synthetic Semantic Variations

To evaluate certificates under controlled semantic variations, we use multimodal LLMs (Achiam et al., 2023; Guo et al., 2025) to generate a sequence of images exhibiting gradual semantic variations for three image recognition datasets, OxfordPets (Parkhi et al., 2012), Flowers102 (Nilsback & Zisserman, 2008), and Food101 (Bossard et al., 2014). For each dataset, we sample images across multiple categories as source images for generation. Figure 6(a) shows example image sequences generated under different semantic variations. We consider both in-domain (ID) variations that occur in the dataset and out-of-distribution (OOD) variations that do not occur in the dataset. For ExactLine and our image-based method, we set the last image from the collected images as the reference image to specify the target semantic extent. Table 1 reports the mean absolute discrepancy over the semantic extent for ExactLine and our framework. The results show that our method consistently yields lower discrepancy, indicating that the proposed transformation aligns better with the intended semantic variation. ExactLine models variation by linear interpolation between two endpoint images, which can approximate simple semantic variations such as color. However, for more complex variations, such as viewpoint or shape, it often introduces unintended changes that are not part of the target semantics. In contrast, our transformation leads to more stable alignment across both in-domain and out-of-distribution variations.

## 5.4. Evaluation on Real-World Semantic Variations

To evaluate certificates on real-world semantic variations, we use eight image recognition datasets, including Caltech101 (Fei-Fei et al., 2004), OxfordPets (Parkhi et al., 2012), StanfordCars (Krause et al., 2013), Flowers102 (Nilsback & Zisserman, 2008), Food101 (Bossard et al., 2014), UCF101 (Soomro et al., 2012), DTD (Cimpoi et al., 2014) and FGVCAircraft (Maji et al., 2013). For each dataset, we identify the main attribute types that naturally vary within the dataset. We then construct a real-world image sequence of semantic variations, as illustrated in Figure 6(b). Compared to generated sequences, real-world sequences exhibit greater uncontrolled variation and may not isolate a target variation. To mitigate this problem, we use the VLM to rank images by their similarity to the corresponding prompt within a semantic family, thereby ordering the sequence primarily by the intended semantic. This VLM-induced ordering provides a practical visual reference for evaluating alignment. Table 2 reports the mean absolute discrepancy over the semantic extent, showing that our method consistently outperforms ExactLine across datasets and semantics, indicating alignment with real-world semantic variation.

## 6. Discussion

Beyond the formal certification guarantee, our framework offers practical benefits for analyzing, comparing, and improving VLM robustness. Certifying prediction-invariant intervals along the semantic extent characterizes where the prediction remains stable. This makes the certificate not only a robustness statement, but also a diagnostic description of the model’s semantic decision geometry. Since reference images can be mapped onto the same semantic extent, the certified intervals can be related to concrete visual examples rather than only to abstract embedding coordinates. The closed-form margin further indicates proximity to semantic decision boundaries and quantifies how the certified intervals change under bounded cross-modal misalignment.

Table 2. Comparison of mean absolute discrepancy among ExactLine, our text-specified transformation (T-Spec), and our image-specified transformation (I-Spec) under real-world semantic variations (lower is better).  
(a) DTD.

<table><tr><td></td><td>Color</td><td>Size</td><td>Texture</td><td>Illumination (ILL)</td></tr><tr><td>ExactLine</td><td>29.9%</td><td>18.5%</td><td>8.7%</td><td>11.9%</td></tr><tr><td>T-Spec</td><td>21.3%</td><td>10.1%</td><td>2.4%</td><td>0.0%</td></tr><tr><td>I-Spec</td><td>19.3%</td><td>8.1%</td><td>3.9%</td><td>0.0%</td></tr></table>

(b) FGVCAircraft.

<table><tr><td></td><td>Viewpoint (VP)</td><td>Background (BG)</td></tr><tr><td>ExactLine</td><td>31.9%</td><td>13.0%</td></tr><tr><td>T-Spec</td><td>26.8%</td><td>14.9%</td></tr><tr><td>I-Spec</td><td>27.4%</td><td>11.0%</td></tr></table>

(c) Caltech101.

<table><tr><td></td><td>Color</td><td>VP</td><td>Style</td></tr><tr><td>ExactLine</td><td>2.7%</td><td>0.0%</td><td>10.5%</td></tr><tr><td>T-Spec</td><td>0.0%</td><td>0.0%</td><td>11.9%</td></tr><tr><td>I-Spec</td><td>0.0%</td><td>0.0%</td><td>7.8%</td></tr></table>

(d) StanfordCars.

<table><tr><td></td><td>Color</td><td>VP</td><td>BG</td></tr><tr><td>ExactLine</td><td>32.8%</td><td>31.7%</td><td>21.4%</td></tr><tr><td>T-Spec</td><td>11.3%</td><td>8.2%</td><td>3.0%</td></tr><tr><td>I-Spec</td><td>6.2%</td><td>5.8%</td><td>4.5%</td></tr></table>

(e) Flowers102.

<table><tr><td></td><td>Color</td><td>Shape</td><td>VP</td></tr><tr><td>ExactLine</td><td>14.9%</td><td>23.5%</td><td>0.0%</td></tr><tr><td>T-Spec</td><td>11.0%</td><td>10.3%</td><td>0.0%</td></tr><tr><td>I-Spec</td><td>6.2%</td><td>8.7%</td><td>0.0%</td></tr></table>

(f) OxfordPets.

<table><tr><td></td><td>Texture</td><td>VP</td><td>BG</td></tr><tr><td>ExactLine</td><td>5.2%</td><td>10.3%</td><td>11.5%</td></tr><tr><td>T-Spec</td><td>2.9%</td><td>0.0%</td><td>3.2%</td></tr><tr><td>I-Spec</td><td>0.5%</td><td>0.0%</td><td>0.2%</td></tr></table>

(g) Food101.

<table><tr><td></td><td>Shape</td><td>BG</td><td>ILL</td></tr><tr><td>ExactLine</td><td>14.7%</td><td>11.2%</td><td>7.4%</td></tr><tr><td>T-Spec</td><td>0.0%</td><td>0.0%</td><td>6.8%</td></tr><tr><td>I-Spec</td><td>0.0%</td><td>0.0%</td><td>5.1%</td></tr></table>

(h) UCF101.

<table><tr><td></td><td>VP</td><td>BG</td><td>ILL</td></tr><tr><td>ExactLine</td><td>30.1%</td><td>13.6%</td><td>7.9%</td></tr><tr><td>T-Spec</td><td>14.1%</td><td>0.0%</td><td>7.3%</td></tr><tr><td>I-Spec</td><td>4.0%</td><td>0.0%</td><td>2.9%</td></tr></table>

These properties support practical uses in robustness auditing, prompt learning, and model adaptation. The certified intervals and class transitions reveal which target semantic variations induce prediction changes and support comparisons across datasets and VLMs (Fang et al., 2022; Yang et al., 2026; Jiang et al., 2025; Yu et al., 2026). For prompt engineering, interval length can serve as a certificate-aware criterion that favors stable predictions over target semantic variations, complementing performance-driven objectives (Zhou et al., 2022b;a; Jiang et al., 2024). Since the certified object is the shared image-text scoring mechanism, the same analysis can also inform downstream pipelines that reuse this score, including image-text retrieval, detection, and segmentation (Du et al., 2022; Zou et al., 2023).

The scope of the certificate is subject to two qualifications. First, the framework depends on the quality of the language proxy and the alignment between visual and textual embeddings. Our bounded misalignment analysis makes this dependence explicit and localizes uncertainty due to crossmodal mismatch, but the resulting bound can become conservative when the modality gap is large. Recent work has begun to analyze and reduce such gaps in VLM representation spaces (Liang et al., 2022; Bhalla et al., 2024), while integrating modality-gap correction into semantic certification remains an important future direction. Second, validating semantic transformations remains challenging because semantic variation is difficult to isolate in the input space. The real-world sequences used in our experiments provide practical visual references, but can still contain non-target semantic changes. The synthetic sequences offer more control over the intended progression, but can introduce artifacts from the source identity. In addition, zero discrepancy values can occur when the visual reference path itself does not induce a prediction change. Such cases still test whether a transformation avoids spurious class transitions, but they provide limited evidence about boundary localization. These challenges motivate semantic-variation benchmarks with stronger control over target attributes and evaluation protocols that distinguish transformation alignment from the absence of prediction changes.

## Acknowledgments

The authors would like to thank Neil Marchant for his valuable input during the course of this research project. This work was supported by the Australian Defence Science and Technology (DST) Group via the Advanced Strategic Capabilities Accelerator (ASCA) program.

## Impact Statement

This work develops robustness certificates for visionlanguage models under target semantic variations. By characterizing how predictions evolve along a parameterized semantic extent and identifying prediction-invariant intervals, the framework supports transparent evaluation and monitoring of semantic drift, and helps localize where predictions are reliably stable under the specified semantic change. We certify prediction invariance under semantic variations specified by text prompts in a similarity space, while a bounded misalignment budget models the remaining cross-modal mismatch. Accordingly, the results are intended to complement application-specific testing and should not be interpreted as guarantees under arbitrary realworld transformations or for safety-critical deployment.

## References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.  
Alayrac, J.-B., Donahue, J., Luc, P., Miech, A., Barr, I., Hasson, Y., Lenc, K., Mensch, A., Millican, K., Reynolds, M., et al. Flamingo: a visual language model for few-shot learning. Advances in Neural Information Processing Systems, 35:23716–23736, 2022.  
Balunovic, M., Baader, M., Singh, G., Gehr, T., and Vechev, M. Certifying geometric robustness of neural networks. Advances in Neural Information Processing Systems, 32, 2019.  
Bhalla, U., Oesterling, A., Srinivas, S., Calmon, F. P., and Lakkaraju, H. Interpreting CLIP with sparse linear concept embeddings (SpLiCE). Advances in Neural Information Processing Systems, 37:84298–84328, 2024.  
Bonaert, G., Dimitrov, D. I., Baader, M., and Vechev, M. Fast and precise certification of transformers. In Proceedings of the ACM SIGPLAN International Conference on Programming Language Design and Implementation, pp. 466–481, 2021.  
Bossard, L., Guillaumin, M., and Van Gool, L. Food-101– mining discriminative components with random forests. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part VI 13, pp. 446–461. Springer, 2014.  
Brooks, T., Holynski, A., and Efros, A. A. Instructpix2pix: Learning to follow image editing instructions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18392–18402, 2023.  
Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., and Vedaldi, A. Describing textures in the wild. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3606–3613, 2014.  
Cohen, J., Rosenfeld, E., and Kolter, Z. Certified adversarial robustness via randomized smoothing. In International Conference on Machine Learning, pp. 1310–1320. PMLR, 2019.  
Crabbe, J., Rodr ´ ´ıguez, P., Shankar, V., Zappella, L., and Blaas, A. Interpreting clip: Insights on the robustness to imagenet distribution shifts. arXiv preprint arXiv:2310.13040, 2023.  
Du, Y., Wei, F., Zhang, Z., Shi, M., Gao, Y., and Li, G. Learning to prompt for open-vocabulary object detection with vision-language model. In Proceedings of the

IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 14084–14093, 2022.

Fang, A., Ilharco, G., Wortsman, M., Wan, Y., Shankar, V., Dave, A., and Schmidt, L. Data determines distributional robustness in contrastive language image pre-training (clip). In International Conference on Machine Learning, pp. 6216–6234. PMLR, 2022.

Fei-Fei, L., Fergus, R., and Perona, P. Learning generative visual models from few training examples: An incremental bayesian approach tested on 101 object categories. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshop, pp. 178–178. IEEE, 2004.

Ferrari, C., Muller, M. N., Jovanovic, N., and Vechev, M. Complete verification via multi-neuron relaxation guided branch-and-bound. arXiv preprint arXiv:2205.00263, 2022.

Gehr, T., Mirman, M., Drachsler-Cohen, D., Tsankov, P., Chaudhuri, S., and Vechev, M. Ai2: Safety and robustness certification of neural networks with abstract interpretation. In IEEE Symposium on Security and Privacy, pp. 3–18. IEEE, 2018.

Guo, D., Wu, F., Zhu, F., Leng, F., Shi, G., Chen, H., Fan, H., Wang, J., Jiang, J., Wang, J., et al. Seed1. 5-vl technical report. arXiv preprint arXiv:2505.07062, 2025.

Jia, C., Yang, Y., Xia, Y., Chen, Y.-T., Parekh, Z., Pham, H., Le, Q., Sung, Y.-H., Li, Z., and Duerig, T. Scaling up visual and vision-language representation learning with noisy text supervision. In International Conference on Machine Learning, pp. 4904–4916. PMLR, 2021.

Jiang, J., Wen, Z., Mansoor, A., and Mian, A. Efficient hyperparameter optimization with adaptive fidelity identification. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 26181– 26190, 2024.

Jiang, L., Yu, D., Xu, R., Tang, T., and Wang, G. Uncertainty-aware predict-then-optimize framework for equitable post-disaster power restoration. In International Joint Conference on Artificial Intelligence, 2025.

Karras, T., Laine, S., and Aila, T. A style-based generator architecture for generative adversarial networks. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4401–4410, 2019.

Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In International Conference on Machine Learning, pp. 2668–2677. PMLR, 2018.

Kim, S., Oh, J., Lee, S., Yu, S., Do, J., and Taghavi, T. Grounding counterfactual explanation of image classifiers to textual concept space. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10942–10950, 2023.  
Krause, J., Stark, M., Deng, J., and Fei-Fei, L. 3d object representations for fine-grained categorization. In Proceedings of the IEEE International Conference on Computer Vision Workshops, pp. 554–561, 2013.  
Lecuyer, M., Atlidakis, V., Geambasu, R., Hsu, D., and Jana, S. Certified robustness to adversarial examples with differential privacy. In IEEE Symposium on Security and Privacy, pp. 656–672. IEEE, 2019.  
Li, J., Li, D., Xiong, C., and Hoi, S. Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation. In International Conference on Machine Learning, pp. 12888–12900. PMLR, 2022.  
Li, L., Weber, M., Xu, X., Rimanic, L., Kailkhura, B., Xie, T., Zhang, C., and Li, B. Tss: Transformation-specific smoothing for robustness certification. In Proceedings of the ACM SIGSAC Conference on Computer and Communications Security, pp. 535–557, 2021.  
Li, L., Guan, H., Qiu, J., and Spratling, M. One prompt word is enough to boost adversarial robustness for pretrained vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 24408–24419, 2024.  
Li, Y., Dong, J., Yang, C., Wen, S., Koniusz, P., Huang, T., Tian, Y., and Ong, Y.-S. Mmt-ard: Multimodal multiteacher adversarial distillation for robust vision-language models. arXiv preprint arXiv:2511.17448, 2025a.  
Li, Y., Yang, C., Dong, J., Yao, Z., Xu, H., Dong, Z., Zeng, H., An, Z., and Tian, Y. Ammkd: Adaptive multimodal multi-teacher distillation for lightweight vision-language models. arXiv preprint arXiv:2509.00039, 2025b.  
Liang, V. W., Zhang, Y., Kwon, Y., Yeung, S., and Zou, J. Y. Mind the gap: Understanding the modality gap in multi-modal contrastive representation learning. Advances in Neural Information Processing Systems, 35: 17612–17625, 2022.  
Maji, S., Rahtu, E., Kannala, J., Blaschko, M., and Vedaldi, A. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.  
Ming, Y., Cai, Z., Gu, J., Sun, Y., Li, W., and Li, Y. Delving into out-of-distribution detection with vision-language representations. Advances in Neural Information Processing Systems, 35:35087–35102, 2022.  
Mirman, M., Hagele, A., Bielik, P., Gehr, T., and Vechev, M. ¨ Robustness certification with generative models. In Proceedings of the ACM SIGPLAN International Conference on Programming Language Design and Implementation, pp. 1141–1154, 2021.  
Miyai, A., Yu, Q., Irie, G., and Aizawa, K. Locoop: Fewshot out-of-distribution detection via prompt learning. Advances in Neural Information Processing Systems, 36: 76298–76310, 2023.  
Muller, M. N., Makarchuk, G., Singh, G., P ¨ uschel, M., ¨ and Vechev, M. Prima: general and precise neural network certification via scalable convex hull approximations. Proceedings of the ACM on Programming Languages, 6(POPL):1–33, 2022.  
Nilsback, M.-E. and Zisserman, A. Automated flower classification over a large number of classes. In Indian Conference on Computer Vision, Graphics & Image Processing, pp. 722–729. IEEE, 2008.  
Parkhi, O. M., Vedaldi, A., Zisserman, A., and Jawahar, C. V. Cats and dogs. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3498–3505. IEEE, 2012.  
Pautov, M., Tursynbek, N., Munkhoeva, M., Muravev, N., Petiushko, A., and Oseledets, I. Cc-cert: A probabilistic approach to certify general robustness of neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, 2022.  
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pp. 8748–8763. PMLR, 2021.  
Schlarmann, C., Singh, N. D., Croce, F., and Hein, M. Robust clip: Unsupervised adversarial fine-tuning of vision embeddings for robust large vision-language models. arXiv preprint arXiv:2402.12336, 2024.  
Shu, Y., Guo, X., Wu, J., Wang, X., Wang, J., and Long, M. Clipood: Generalizing clip to out-of-distributions. In International Conference on Machine Learning, pp. 31716–31731. PMLR, 2023.  
Singh, G., Gehr, T., Puschel, M., and Vechev, M. An abstract ¨ domain for certifying neural networks. Proceedings of the ACM on Programming Languages, 3(POPL):1–30, 2019.  
Sonthalia, A., Uselis, A., and Oh, S. J. On the rankability of visual embeddings. Advances in Neural Information Processing Systems, 38:66169–66203, 2025.  
Soomro, K., Zamir, A. R., and Shah, M. Ucf101: A dataset of 101 human actions classes from videos in the wild. arXiv preprint arXiv:1212.0402, 2012.  
Sotoudeh, M. and Thakur, A. V. Computing linear restrictions of neural networks. Advances in Neural Information Processing Systems, 32, 2019.  
Wang, F., Xu, P., Ruan, W., and Huang, X. Towards verifying the geometric robustness of large-scale neural networks. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 15197–15205, 2023a.  
Wang, S., Pei, K., Whitehouse, J., Yang, J., and Jana, S. Formal security analysis of neural networks using symbolic intervals. In USENIX Security Symposium, pp. 1599– 1614, 2018.  
Wang, S., Zhang, H., Xu, K., Lin, X., Jana, S., Hsieh, C.-J., and Kolter, J. Z. Beta-crown: Efficient bound propagation with per-neuron split constraints for neural network robustness verification. Advances in Neural Information Processing Systems, 34:29909–29921, 2021.  
Wang, Z., Jiang, Y., Zheng, H., Wang, P., He, P., Wang, Z., Chen, W., Zhou, M., et al. Patch diffusion: Faster and more data-efficient training of diffusion models. Advances in Neural Information Processing Systems, 36: 72137–72154, 2023b.  
Xiao, Z., Shen, J., Derakhshani, M. M., Liao, S., and Snoek, C. G. Any-shift prompting for generalization over distributions. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 13849– 13860, 2024.  
Xu, Z., Zhang, X., Li, R., Tang, Z., Huang, Q., and Zhang, J. Fakeshield: Explainable image forgery detection and localization via multi-modal large language models. In International Conference on Learning Representations, 2025.  
Yang, P., Akhtar, N., Wen, Z., and Mian, A. Local path integration for attribution. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 37, pp. 3173–3180, 2023a.  
Yang, P., Akhtar, N., Wen, Z., Shah, M., and Mian, A. S. Recalibrating feature attributions for model interpretation. In International Conference on Learning Representations, 2023b.  
Yang, P., Akhtar, N., Jiang, J., and Mian, A. Attributionguided model rectification of unreliable neural network behaviors. arXiv preprint arXiv:2603.15656, 2026.  
Yu, D., Xu, R., Zhuang, D., Bu, Y., Wang, S., and Wang, G. Trustenergy: A unified framework for accurate and  
reliable user-level energy usage prediction. Proceedings of the AAAI Conference on Artificial Intelligence, 40(46): 39558–39566, 2026.  
Yuan, Y., Wang, S., and Su, Z. Precise and generalized robustness certification for neural networks. In USENIX Security Symposium, pp. 4769–4786, 2023.  
Zhang, H., Weng, T.-W., Chen, P.-Y., Hsieh, C.-J., and Daniel, L. Efficient neural network robustness certification with general activation functions. Advances in Neural Information Processing Systems, 31, 2018.  
Zhang, Y., Zhu, W., He, C., and Zhang, L. Lapt: Labeldriven automated prompt tuning for ood detection with vision-language models. In European Conference on Computer Vision, pp. 271–288. Springer, 2024a.  
Zhang, Y., Zhu, W., Tang, H., Ma, Z., Zhou, K., and Zhang, L. Dual memory networks: A versatile adaptation approach for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 28718–28728, 2024b.  
Zhao, Y., Pang, T., Du, C., Yang, X., Li, C., Cheung, N.- M. M., and Lin, M. On evaluating adversarial robustness of large vision-language models. Advances in Neural Information Processing Systems, 36:54111–54138, 2023.  
Zhou, K., Yang, J., Loy, C. C., and Liu, Z. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 16816–16825, 2022a.  
Zhou, K., Yang, J., Loy, C. C., and Liu, Z. Learning to prompt for vision-language models. International Journal of Computer Vision, 130(9):2337–2348, 2022b.  
Zhou, X., Zhang, M., Lee, Z., Hua, Y., Ye, W., Salim, F., Zhang, S., et al. Boosting resilience of large language models through causality-driven robust optimization. Advances in Neural Information Processing Systems, 38: 55821–55845, 2025.  
Zhu, J.-Y., Park, T., Isola, P., and Efros, A. A. Unpaired image-to-image translation using cycle-consistent adversarial networks. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2223–2232, 2017.  
Zhu, W., Zhang, Y., Jin, X., Zeng, W., and Zhang, L. Ants: Adaptive negative textual space shaping for ood detection via test-time mllm understanding and reasoning. arXiv preprint arXiv:2509.03951, 2025.  
Zou, X., Yang, J., Zhang, H., Li, F., Li, L., Wang, J., Wang, L., Gao, J., and Lee, Y. J. Segment everything everywhere all at once. Advances in Neural Information Processing Systems, 36:19769–19782, 2023.

## A. Proof

In this section, we provide the proof of Lemmas 4.5-4.7. We begin with the proof of Lemma 4.5.

Proof of Lemma 4.5. Fix any $\varphi \in [ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ , any $c \neq c ^ { \prime } .$ and any $e \in \Gamma _ { \delta } ( \varphi )$ . Recall that the nominal pairwise margin along the path γ is defined as

$$
m _ {c, c ^ {\prime}} (\varphi) := \big \langle \gamma (\varphi), u _ {c} - u _ {c ^ {\prime}} \big \rangle ,
$$

and the perturbed margin at extent $\varphi$ under an embedding e is

$$
\tilde {m} _ {c, c ^ {\prime}} (\varphi ; e) := \big \langle e, u _ {c} - u _ {c ^ {\prime}} \big \rangle .
$$

Taking the difference and using bilinearity of the inner product yields

$$
\tilde {m} _ {c, c ^ {\prime}} (\varphi ; e) - m _ {c, c ^ {\prime}} (\varphi) = \bigl \langle e - \gamma (\varphi), u _ {c} - u _ {c ^ {\prime}} \bigr \rangle .
$$

Applying the Cauchy–Schwarz inequality gives

$$
\left| \tilde {m} _ {c, c ^ {\prime}} (\varphi ; e) - m _ {c, c ^ {\prime}} (\varphi) \right| \leq \| e - \gamma (\varphi) \| _ {2} \| u _ {c} - u _ {c ^ {\prime}} \| _ {2}.
$$

Finally, since $e \in \Gamma _ { \delta } ( \varphi )$ and

$$
\Gamma_ {\delta} (\varphi) = \bigl \{e \in \mathbb {S} ^ {d - 1}: \| e - \gamma (\varphi) \| _ {2} \leq \delta \bigr \},
$$

we have $\| e - \gamma ( \varphi ) \| _ { 2 } \leq \delta$ . Substituting this bound into the above inequality yields

$$
\left| \tilde {m} _ {c, c ^ {\prime}} (\varphi ; e) - m _ {c, c ^ {\prime}} (\varphi) \right| \leq \delta \left\| u _ {c} - u _ {c ^ {\prime}} \right\| _ {2},
$$

which completes the proof.

We then prove Proposition 4.6.

Proof of Proposition 4.6. Under Assumption 4.4, fix any $\varphi \in \bigl [ \varphi _ { a } , \varphi _ { a ^ { \prime } } \bigr ]$ and let ${ \hat { y } } ( \varphi ) : = f ( \gamma ( \varphi ) )$ ). Take an arbitrary $e \in \Gamma _ { \delta } ( \varphi )$ . For any $c ^ { \prime } \neq \hat { y } ( \varphi )$ , define the perturbed margin

$$
\tilde {m} _ {\hat {y} (\varphi), c ^ {\prime}} (\varphi ; e) := \big \langle e, u _ {\hat {y} (\varphi)} - u _ {c ^ {\prime}} \big \rangle .
$$

By Lemma 4.5 applied to the pair $( \hat { y } ( \varphi ) , c ^ { \prime } )$ ,

$$
\left| \tilde {m} _ {\hat {y} (\varphi), c ^ {\prime}} (\varphi ; e) - m _ {\hat {y} (\varphi), c ^ {\prime}} (\varphi) \right| \leq \delta \left\| u _ {\hat {y} (\varphi)} - u _ {c ^ {\prime}} \right\| _ {2}.
$$

Hence,

$$
\tilde {m} _ {\hat {y} (\varphi), c ^ {\prime}} (\varphi ; e) \geq m _ {\hat {y} (\varphi), c ^ {\prime}} (\varphi) - \delta \left\| u _ {\hat {y} (\varphi)} - u _ {c ^ {\prime}} \right\| _ {2}.
$$

By the condition of the proposition, the right-hand side is strictly positive for every $c ^ { \prime } \neq \hat { y } ( \varphi )$ . Therefore,

$$
\left<   e, u _ {\hat {y} (\varphi)} - u _ {c ^ {\prime}} \right> > 0 \quad \forall c ^ {\prime} \neq \hat {y} (\varphi),
$$

equivalently,

$$
\left\langle e, u _ {\hat {y} (\varphi)} \right\rangle > \left\langle e, u _ {c ^ {\prime}} \right\rangle \quad \forall c ^ {\prime} \neq \hat {y} (\varphi).
$$

Thus $\hat { y } ( \varphi )$ uniquely maximizes $\langle e , u _ { c } \rangle$ over $c \in { \mathcal { C } }$ , and by the definition $f ( e ) : = \arg \operatorname* { m a x } _ { c \in \mathcal { C } } \langle e , u _ { c } \rangle$ in (1), we conclude $f ( e ) = { \hat { y } } ( \varphi )$ . Since $e \in \Gamma _ { \delta } ( \varphi )$ was arbitrary, the claim holds for all $e \in \Gamma _ { \delta } ( \varphi )$ . □

Below, we provide the proof of Lemma 4.7.

Proof of Lemma 4.7. Fix any $c , c ^ { \prime } \in { \mathcal { C } }$ with $c \neq c ^ { \prime } .$ . Take any $\varphi \in [ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ and any $e \in \Gamma _ { \delta } ( \varphi )$ . By definition,

$$
m _ {c, c ^ {\prime}} (\varphi) = \left\langle \gamma (\varphi), u _ {c} - u _ {c ^ {\prime}} \right\rangle , \quad \tilde {m} _ {c, c ^ {\prime}} (\varphi ; e) = \left\langle e, u _ {c} - u _ {c ^ {\prime}} \right\rangle .
$$

Applying Lemma 4.5 yields

$$
\left| \tilde {m} _ {c, c ^ {\prime}} (\varphi ; e) - m _ {c, c ^ {\prime}} (\varphi) \right| \leq \delta \left\| u _ {c} - u _ {c ^ {\prime}} \right\| _ {2} = \varepsilon_ {c, c ^ {\prime}}.
$$

Now assume $\varphi \notin \mathcal { U } _ { c , c ^ { \prime } } ( \delta )$ . By the definition of $\mathcal { U } _ { c , c ^ { \prime } } ( \delta )$ in (17), we have $| m _ { c , c ^ { \prime } } ( \varphi ) | > \varepsilon _ { c , c ^ { \prime } }$ . We show that $m _ { c , c ^ { \prime } } ( \varphi )$ and $\tilde { m } _ { c , c ^ { \prime } } ( \varphi ; e )$ must have the same sign.

If $m _ { c , c ^ { \prime } } ( \varphi ) > \varepsilon _ { c , c ^ { \prime } }$ , then

$$
\tilde {m} _ {c, c ^ {\prime}} (\varphi ; e) \geq m _ {c, c ^ {\prime}} (\varphi) - \left| \tilde {m} _ {c, c ^ {\prime}} (\varphi ; e) - m _ {c, c ^ {\prime}} (\varphi) \right| \geq m _ {c, c ^ {\prime}} (\varphi) - \varepsilon_ {c, c ^ {\prime}} > 0.
$$

If $m _ { c , c ^ { \prime } } ( \varphi ) < - \varepsilon _ { c , c ^ { \prime } }$ , then

$$
\tilde {m} _ {c, c ^ {\prime}} (\varphi ; e) \leq m _ {c, c ^ {\prime}} (\varphi) + \left| \tilde {m} _ {c, c ^ {\prime}} (\varphi ; e) - m _ {c, c ^ {\prime}} (\varphi) \right| \leq m _ {c, c ^ {\prime}} (\varphi) + \varepsilon_ {c, c ^ {\prime}} <   0.
$$

In either case, $\tilde { m } _ { c , c ^ { \prime } } ( \varphi ; e )$ has the same sign as $m _ { c , c ^ { \prime } } ( \varphi )$ . Since $e \in \Gamma _ { \delta } ( \varphi )$ ) was arbitrary, this proves the first claim. For the second claim, suppose there exist $\varphi \in [ \varphi _ { a } , \varphi _ { a ^ { \prime } } ]$ and $e \in \Gamma _ { \delta } ( \varphi )$ ) such that $\tilde { m } _ { c , c ^ { \prime } } ( \varphi ; e ) = 0$ . If $\varphi \notin \mathcal { U } _ { c , c ^ { \prime } } ( \delta )$ , then the first part implies that $\tilde { m } _ { c , c ^ { \prime } } ( \varphi ; e )$ is strictly positive when $m _ { c , c ^ { \prime } } ( \varphi ) > 0$ , and strictly negative when $m _ { c , c ^ { \prime } } ( \varphi ) < 0$ . In either case, $\tilde { m } _ { c , c ^ { \prime } } ( \varphi ; e ) \neq 0 .$ , contradicting the assumption that $\tilde { m } _ { c , c ^ { \prime } } ( \varphi ; e ) = 0$ . Therefore, $\varphi \in \mathcal { U } _ { c , c ^ { \prime } } ( \delta )$ . □

## B. Experimental Setup

In this work, we use the publicly available pretrained CLIP ViT-B/16 model released by OpenAI (Radford et al., 2021). All experiments are conducted using an NVIDIA 3090Ti GPU (24GB), a 16-core 3.9GHz Intel Core i9-12900K CPU, and 128GB RAM.

To evaluate semantic robustness under controllable semantic extents, we construct two complementary evaluation sets: (i) real-world semantic-variation sequences collected from existing datasets, and (ii) synthetic sequences that explicitly instantiate both ID and OOD semantic variations. We emphasize that building a fully controlled semantic-variation benchmark remains practically challenging, as the semantic edits must preserve class identity while avoiding spurious artifacts that can confound robustness analysis.

Synthetic image sequences. Synthetic semantic-variation sequences are harder to construct in a controlled manner when the generator is misaligned with the target domain. In our preliminary attempts, domain-specialized generators (e.g., StyleGAN (Karras et al., 2019) and CycleGAN (Zhu et al., 2017)) often produced unrealistic outputs when asked to enforce semantic shifts on out-of-domain objects, and diffusion-based generators (e.g., InstructPix2Pix (Brooks et al., 2023)) frequently introduced visible artifacts or drifted from the input identity, which injects unintended semantic factors. We therefore use multimodal large language models (MLLM) (e.g., GPT models (Achiam et al., 2023) and Seedream (Guo et al., 2025)) to construct synthetic image sequences. Concretely, for each dataset we choose three representative classes and sample seed images per class. For each seed image, we generate at least one pair of ID and OOD semantic shifts for each semantic. Each shift is instantiated as an ordered image sequence with an explicit semantic progression (e.g., for a food class such as samosa, we generate shape variations from triangular to round as a controlled semantic factor). We curate generated outputs to ensure visual coherence and semantic correctness: samples with severe artifacts, identity mismatch, or off-target edits are discarded. Overall, while these synthetic sequences enable targeted ID/OOD semantic shifts, they also highlight that constructing a fully controlled semantic-variation dataset remains a challenge for robustness evaluation.

Real-world image sequences. For each dataset, we ensure coverage by selecting a subset of classes that accounts for at least 10% of all categories. For each selected class, we embed all images using the CLIP image encoder and perform automatic clustering in the embedding space based on pairwise distances, aiming to expose naturally occurring intra-class modes (e.g., color, shape, texture, background). We then specify a set of attribute prompts per class using prior knowledge of plausible variations (e.g., color for flowers, shape for certain foods), while keeping a shared prompt template $( { \mathrm { e . g . , ~ } } ^ { \cdot \cdot } { \mathrm { a } }$ photo of $\textsf { a } \{ \arctan : { \mathsf { b u t e } } \} \{ \mathsf { c l a s s } \} ^ { \prime \prime } \ \rangle$ to reduce confounding from prompt format. For each prompt, we rank clusters by their proximity to the prompt embedding. Finally, we manually select representative images from the top-ranked clusters to form an ordered image sequence that exhibits a gradual semantic change. This manual step is necessary because purely automatic ranking can still surface outliers or cases where the target attribute is entangled with unintended factors.

## C. Cross-Modal Validity of Language Proxies

![](images/859eade07b5eec9d25cbd6da159c5eb3e3a8a9fb49dc8d5b613403587b580e95.jpg)

<details>
<summary>line chart</summary>

| Prompt rank to the top matching prompt | color | material | style | color+material | color+style | style+material | color+style+material |
| -------------------------------------- | ----- | -------- | ----- | -------------- | ----------- | -------------- | -------------------- |
| 0.0                                    | 0.28  | 0.29     | 0.29  | 0.27           | 0.26        | 0.27           | 0.25                 |
| 0.2                                    | 0.285 | 0.295    | 0.295 | 0.28           | 0.27        | 0.28           | 0.265                |
| 0.4                                    | 0.29  | 0.30     | 0.30  | 0.285          | 0.275       | 0.285          | 0.27                 |
| 0.6                                    | 0.295 | 0.305    | 0.305 | 0.29           | 0.28        | 0.29           | 0.275                |
| 0.8                                    | 0.30  | 0.31     | 0.31  | 0.295          | 0.285       | 0.295          | 0.28                 |
| 1.0                                    | 0.315 | 0.315    | 0.315 | 0.315          | 0.31        | 0.315          | 0.31                 |
</details>

(a) Cosine similarity (image to prompts).

![](images/19911c6044c6b29bafb6f37c4c21a7f9a391f94269b0d78e526faa7baafa43c7.jpg)

<details>
<summary>box plot chart</summary>

| Category           | Spearman rank correlation |
| ------------------ | ------------------------- |
| color              | 0.75                      |
| material           | 0.60                      |
| style              | 0.90                      |
| style+material      | 0.80                      |
| color+style        | 0.70                      |
| color+material     | 0.85                      |
| color+styl.+mat.   | 0.95                      |
</details>

(b) Spearman correlation within concept.  
Figure 7. Cross-modal semantic consistency on an ImageNet subset. For each image, we form a prompt family by fixing a template and varying only the attribute word, $\mathrm { e . g . }$ , “a photo of a [attribute] $[ c l a s s ] ^ { \prime \prime }$ . (a) We select the most similar prompt to the image as the anchor $t ^ { * }$ , rank the remaining prompts by their cosine similarity to $t ^ { * }$ in the prompt embedding space, and plot image-to-prompt cosine similarity over the normalized text-induced rank. (b) Box plots of Spearman $\rho$ measure the agreement between the within-family prompt ranking by prompt-to-prompt cosine similarity to the anchor prompt $t ^ { * }$ and the prompt ranking by image-to-prompt cosine similarity to the image x.

Figure 7 evaluates cross-modal semantic consistency on an ImageNet subset using controlled semantic families (e.g., color, material, and style.) Within each semantic family, we fix a template and vary only the attribute word to form a prompt family, e.g., “a photo of a [color] [class]”. For each image $x ,$ we compute its cosine similarity to every prompt in the family and select the most similar one as the anchor prompt $t ^ { * }$ . We then rank the remaining prompts by their cosine similarity to $t ^ { * }$ . Figure 7(a) plots image-to-prompt cosine similarity over the normalized rank induced by this text-side ordering. Figure 7(b) reports Spearman correlation $\rho$ between the prompt ranking by prompt-to-prompt similarity to the anchor prompt $t ^ { * }$ and the prompt ranking by image-to-prompt cosine similarity to the image x.

Across all concept families, we observe samples with $\rho$ close to 1 in every family, indicating that the prompt ordering induced by prompt-to-prompt similarity to $t ^ { * }$ can be consistent with the ordering induced by image-to-prompt similarity. Higher correlation therefore suggests that relative similarity relations within the prompt family are frequently preserved across modalities, supporting the use of text prompts as semantic proxies in our framework. Compositional prompt families, formed by combining multiple attributes such as color and style, tend to exhibit higher $\rho ,$ suggesting that richer semantic descriptions improve proxy quality under the same controlled setup. A plausible explanation is that additional attributes provide more context for grounding the varied word, reducing ambiguity and making the intended semantic direction more specific. This indicates that prompt engineering (Zhou et al., 2022a;b) can serve as a practical mechanism to strengthen semantic specification when constructing language proxies. While proxy quality can vary across concepts and samples, the overall trend in Figure 7 supports our use of prompt families to parameterize semantic variation and to derive certificates based on the induced embedding geometry.

## D. Alignment to Visual Reference Transformations

In this section, we evaluate whether our constructed semantic transformation $\gamma$ in the embedding space aligns with semantic variation in the input space. Using annotated image sequences that exhibit gradual changes of the target semantic, we build a visual reference transformation $\gamma ^ { \mathrm { r e f } }$ and compare it with our established transformations including the text-specified (T-Spec) and image-specified (I-Spec) methods, as well as the ExactLine baseline. Specifically, we use a uniform extent $\Phi : = \{ 0 , \frac { 1 } { K - 1 } , \ldots , 1 \} \subset [ 0 , 1 ]$ $\gamma ( \varphi )$ $\gamma ^ { \mathrm { r e f } } ( \varphi )$ $\varphi \in \Phi$ average over Φ to obtain the mean cosine similarity, $\begin{array} { r } { \frac { 1 } { | \Phi | } \sum _ { \varphi \in \Phi } \cos \bigl ( \gamma ( \varphi ) , \gamma ^ { \mathrm { r e f } } ( \varphi ) \bigr ) } \end{array}$ .

Figure 8 reports the distribution of mean cosine similarity scores over semantic instances for each dataset. T-Spec and I-Spec consistently achieve higher alignment than ExactLine, demonstrating that our constructed transformations closely track the embedding variation induced by annotated input sequences with gradual semantic changes. In contrast, linear interpolation in the input space can produce embedding paths that deviate substantially from the visual reference transformation. Overall, these results show that our constructed transformations can faithfully follow semantic variation as realized in the input space, providing a concrete input-space grounding for our embedding-space specification and supporting its use for certificate construction.

![](images/bd21f7b040653898f056fa6b8a4fcabcdba2a864a58598949ffefe708151cf11.jpg)

<details>
<summary>box plot</summary>

| Dataset       | T-Spec | I-Spec | ExactLine |
| ------------- | ------ | ------ | --------- |
| Caltech101    | 0.96   | 0.96   | 0.93      |
| DTD           | 0.95   | 0.95   | 0.92      |
| FGVCAircraft  | 0.97   | 0.97   | 0.94      |
| Flowers102    | 0.97   | 0.97   | 0.95      |
| Food101       | 0.95   | 0.95   | 0.92      |
| OxfordPets    | 0.96   | 0.96   | 0.93      |
| StanfordCars  | 0.95   | 0.95   | 0.88      |
| UCF101        | 0.95   | 0.96   | 0.90      |
</details>

Figure 8. Alignment to semantic variation in the input space. For each dataset, we compare our constructed semantic transformation with a visual reference transformation constructed from the annotated image sequence with gradual semantic variations. For each semantic instance, we uniformly sample extents $\varphi \in [ 0 , 1 ]$ , compute cosine similarity between the transformed embedding and the visual reference embedding at each $\varphi ,$ and average over extents to obtain a mean cosine similarity. Boxplots show the distribution of mean cosine similarity across datasets. Both our text-specified transformation (T-Spec) and image-specified transformation (I-Spec) consistently outperform ExactLine, showing strong alignment with semantic variation in the input space.

## E. Stability under Prompt Variations

![](images/ddddd2f77199fed8d9fd91a074ad1f22329e4d11f2ac373ba96361de4c24ea85.jpg)

<details>
<summary>text_image</summary>

Category
Tench: Tinca, Tinca.
Goldfish: Carassius auratus.
Great white shark: White shark, Man-eater.
Tiger shark: Galeocerdo cuvieri'.
Hammerhead: Hammerhead shark.
Electric ray: crampfish, numbfish, torpedo.
Ostrich: Struthio camelus.
...
Template
A photo of a [class]:
A nice [class],
A [class],
A photo of a nice [class].
Color
Red: Bright red, Deep red.
Yellow: Pale yellow, Golden yellow.
Blue: Sky blue, Navy blue.
Pink: Hot pink, Soft pink.
Gray: Light gray, Dark gray.
White: Off white, Ivory white, Cream white.
Black: Jet black, Matte black.
...
Background
On a plate: Served on a plate, On a dish.
In a bowl: Served in a bowl, In a dish.
On a street: On the street, On a road.
On a lawn: On the grass, On a grassy lawn.
On a sofa: On a couch, On the couch.
In the sky: In the air, Up in the sky.
In a gym: At the gym, Inside a gym.
...
Size
Tiny: Very small, Mini.
Small: Little, Compact, Not big.
Medium: Mid-sized, Average.
Large: Big, Full-sized, Oversized.
Huge: Very large, Enormous.
High: Tall, Elevated.
Short: Low, Stubby.
...
Texture
Smooth: Silky, Sleek.
Hard: Rigid, Stiff, Solid.
Fuzzy: Hairy, Fluffy.
Slippery: Slick, Slippery to touch, Oily.
Sticky: Tacky, Gooey.
Grainy: Sandy, Powdery, Gritty.
Dense: Compact, Tight.
...
</details>

Figure 9. Examples of prompt variants on ImageNet. We consider category name variants from ImageNet annotations, template variants for VLM prompting, and attribute synonym sets for semantics such as color, background, size, and texture. Within each type, we form a prompt family by substituting only the corresponding word or phrase while keeping the remaining prompt structure fixed, e.g., “a photo ${ \dot { o f a } } { \dot { T e } } n c h ^ { , , } { \dot { \to } } { \dot { \bar { a } } } _ { a }$ photo of a Tinca”.

We evaluate whether the proxy-specified semantics remain stable under typical prompt variations. We consider three types of prompt variants, as illustrated in Figure 9. These include category name variants from ImageNet annotations, commonly used prompt template variants for VLM prompting, and attribute synonym sets for semantics such as color, background, size, and texture. For each prompt variant, we vary only one prompt component at a time and measure the resulting change in image-to-prompt cosine similarity across variants.

Figure 10 reports prompt-to-prompt cosine similarity to quantify how much a prompt changes under different prompt variation types on an ImageNet subset. For each prompt family, we select a reference prompt and compute the cosine similarity between its prompt embedding and the embeddings of its variants. We aggregate these similarities across prompt families and report the resulting distributions for each variation type. The results show consistently high prompt cosine similarity for template variants and attribute synonym substitutions, indicating that these edits remain close to the reference phrasing in the prompt embedding space. Category name substitutions exhibit a larger spread. This is expected because ImageNet category annotations can include heterogeneous strings, such as scientific names, alternative spellings, and uncommon aliases. For example, a class may be annotated with both a common name and a scientific name (e.g., goldfish vs. Carassius cuvieri), or with alternative aliases (e.g., great white shark vs. man-eater). These substitutions can shift the prompt embedding more substantially than template or attribute edits. Nevertheless, the overall similarities remain high, indicating that category name variants still preserve relative image–prompt alignment for most classes.

![](images/1d6fd78ed6b99a1474fa98c1c308f1236211123bbc2dbeeca4774ebed59f0e5f.jpg)

<details>
<summary>box plot</summary>

| Prompt variation type | Prompt to prompt similarity |
| --------------------- | --------------------------- |
| Categories            | 0.8                         |
| Template              | 0.9                         |
| Color                 | 0.9                         |
| Background            | 0.9                         |
| Size                  | 0.9                         |
| Texture               | 0.9                         |
</details>

Figure 10. Prompt cosine similarity under prompt variations on an ImageNet subset. For each variation type, we compute cosine similarity between a reference prompt (shown in bold) and its variants within the same prompt family, and report the distribution across prompt families.

Overall, these results demonstrate that using language as semantic proxies is robust to small prompt-level variations. This provides empirical support that the resulting transformations are not overly sensitive to the particular wording used to specify a target semantic.

## F. Semantic Strength via Similarity

![](images/8e40a741dfe0c46dc1a3e8b083a5f4aaa6ac07b76fddbe3b93963c4ddc2ec1a8.jpg)

<details>
<summary>line chart</summary>

| Quantile bins of similarity coordinate | DTD (texture) | Flowers102 (color) | SUN397 (background) | UCF101 (illumination) |
| -------------------------------------- | ------------- | ------------------ | ------------------- | --------------------- |
| 0.0                                    | 0.0           | 0.0                | 0.0                 | 0.0                   |
| 0.2                                    | 0.0           | 0.0                | 0.0                 | 0.0                   |
| 0.4                                    | 0.1           | 0.2                | 0.1                 | 0.1                   |
| 0.6                                    | 0.8           | 0.9                | 0.7                 | 1.0                   |
| 0.8                                    | 1.0           | 1.0                | 0.9                 | 1.0                   |
| 1.0                                    | 1.0           | 1.0                | 1.0                 | 1.0                   |
</details>

Figure 11. Semantic strength via similarity. For each dataset, we sample 20 semantic pairs $( a , a ^ { \prime } )$ and randomly split images from the two classes into two disjoint subsets. We compute class mean visual embeddings $\bar { z } _ { a } , \bar { z } _ { a ^ { \prime } }$ on images from one subset and form the semantic direction by $v _ { a , a ^ { \prime } } = \bar { z } _ { a ^ { \prime } } - \bar { z } _ { a }$ . We then score images in the other subset by $t ( x _ { i } ) = \langle z _ { i } , v _ { a , a ^ { \prime } } \rangle$ with $z _ { i } = f _ { \mathrm { i m g } } ( x _ { i } )$ , sort by t(xi), and partition into equal-count quantile bins. The plot shows the fraction of samples with label $y _ { a ^ { \prime } }$ across bins. Solid lines are averages over pairs and shaded bands are 95% normal approximation confidence intervals across pairs.

Our certificates rely on the assumption that semantic strength can be measured by similarity in the embedding space. While we derive bounds that account for semantic misalignment, we also provide empirical evidence that tests whether samples from two semantic instances can be consistently ordered by a single similarity coordinate in VLMs. We consider four datasets that represent common semantic factors. DTD focuses on texture attributes (e.g., dotted, knitted, and woven), making it a natural choice for texture variation. Flowers102 contains flower categories with stable color appearance within each category, making it suitable for testing color variation. SUN397 spans a broad range of scene types, and class differences often manifest through global background context and layout, for example airplane cabin versus desert. UCF101 consists of video frames collected under diverse capture conditions, where lighting and viewpoint changes can be prominent, and we use it as a practical proxy for illumination related variation. Together, these datasets allow us to test similarity-based

semantic strength across heterogeneous semantic factors.

For each dataset, we instantiate semantic variations by sampling 20 semantic pairs $( a , a ^ { \prime } )$ , where a and $a ^ { \prime } .$ , instantiated by two classes in the dataset. For each pair $( a , a ^ { \prime } )$ , we collect images from the two classes and randomly split them into two disjoint subsets. For an image $x _ { i }$ , we denote its visual embedding by $z _ { i } = f _ { \mathrm { i m g } } ( x _ { i } )$ , and denote its label within the pair by $y _ { i } \in \{ y _ { a } , y _ { a ^ { \prime } } \}$ . On one subset, we compute class mean embeddings $\bar { z } _ { a }$ and $\bar { z } _ { a ^ { \prime } }$ by averaging $z _ { i }$ within each class, and define the semantic direction $v _ { a , a ^ { \prime } } = \bar { z } _ { a ^ { \prime } } - \bar { z } _ { a }$ . On the other subset used for evaluation, we compute a similarity coordinate for each image as $t ( x _ { i } ) = \langle z _ { i } , v _ { a , a ^ { \prime } } \rangle$ . We pool the images in the evaluation subset from both classes, sort them by $t ( x _ { i } )$ , and partition them into multiple equal-count quantile bins. For each bin $B ,$ we compute the fraction of samples labeled $y _ { a ^ { \prime } }$ as $\begin{array} { r } { \frac { 1 } { | B | } \sum _ { x _ { i } \in B } \mathbb { I } [ y _ { i } = y _ { a ^ { \prime } } ] } \end{array}$ $y _ { i }$ $x _ { i }$ pairs yields one sequence of fractions per pair. Figure 11 aggregates these results for each dataset by plotting the average across pairs within each bin as the solid line, with a shaded band showing a 95% normal approximation confidence interval for this average based on the empirical variability across pairs.

Figure 11 shows that the fraction of samples with label $y _ { a ^ { \prime } }$ increases across the quantile bins of the similarity coordinate on all four datasets. The transition is more concentrated for DTD and Flowers102 and more gradual for SUN397, which is expected given the larger intra class variation in scene categories. Overall, the results indicate that semantic strength is captured by similarity in the embedding space across different semantic factors.