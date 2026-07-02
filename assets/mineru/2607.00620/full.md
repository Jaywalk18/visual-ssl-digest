# Identifying Latent Concepts and Structures for Generalized Category Discovery

Boyang Dai <sup>1</sup> Chaoqi Chen <sup>2</sup> Yizhou Yu <sup>1</sup>

https://github.com/Michael-McQueen/CPF

## Abstract

Generalized Category Discovery (GCD) aims to recognize known classes while autonomously discovering novel ones in open-world settings. However, current approaches primarily focus on de signing clustering objectives, often overlooking a critical bottleneck: standard vision backbones yield high-rank, entangled token representations that are ill-suited for unsupervised discovery of latent concepts and structures. In this paper, we propose Compositional Primitive Fields (CPF-GCD), a novel representation learning framework that reshapes the feature space to make such la tent structure identifiable by enforcing a low-rank compositional organization. Our core hypothesis is that all categories, whether known or novel, can be expressed as compositions and spatial arrangements of a finite set of learnable visual primitives that capture reusable concepts. CPF instantiates this geometric constraint via a spatial field mech anism. Inserted between the backbone and the head, it rewrites noisy patch tokens through low rank primitive mixtures, effectively decomposing images into reusable atomic parts and their spa tial layouts. By explicitly modeling the spatial distribution of primitives, CPF enables novel cat egories to emerge naturally as new activation pat terns over a shared vocabulary. This shifts the focus of representation from merely partitioning global embeddings to constructing a structured and separable primitive field. Extensive experi ments demonstrate that CPF serves as a generic, plug-and-play module that consistently boosts performance across diverse GCD baselines, validating that identifying and leveraging low-rank compositional structure is a crucial inductive bias for open-world recognition.

![](images/f9e604a4d57473b08cc157ad10bdec8cb4cf0395e6b41763afcbed46b25dd6f3.jpg)  
Figure 1. Why structure tokens before discovery? A compact set of reusable primitives can organize noisy patch tokens before they are consumed by a GCD head, making unknown categories easier to separate as new primitive activation patterns.

## 1. Introduction

Open-world recognition systems are often asked to decide whether an image belongs to a familiar category or to a class that has never been annotated. Generalized category discovery (GCD) (Vaze et al., 2022) formalizes this requirement by training with labeled examples from known classes and unlabeled examples drawn from both known and unknown classes. The desired model must preserve the identity of labeled categories while arranging unlabeled samples from unseen categories into meaningful groups. This setting is deliberately harder than closed-set recognition: the model cannot rely on a fixed label vocabulary, yet it must still produce a representation in which new semantic groups can become visible.

Most GCD methods improve the decision layer attached to a pre-trained visual encoder. Recent systems differ in whether they use clustering objectives (Rastegar et al., 2024b), contrastive criteria (Choi et al., 2024), or prototype-based classifiers (Wen et al., 2023; Cao et al., 2024), but they commonly inherit the same interface: dense visual tokens are collapsed into a global descriptor before a discovery head receives them. When this descriptor is already well organized, such heads can work effectively. When it is not, the head has to separate unknown categories inside a representation that was never built for discovery.

This representation-side bottleneck is particularly visible for modern vision transformers (Park & Kim, 2022). Their global class token is optimized to summarize evidence for known labels, so it can discard the local evidence that distinguishes fine-grained or previously unseen categories. Patch tokens still contain these local cues, but in the raw backbone space, they appear as a noisy high-dimensional cloud: foreground parts, background textures, and incidental correlations may occupy nearby directions (Izmailov et al., 2022). As a result, discovery heads can suffer from two opposite errors: one category may break into several clusters (Zhang et al., 2021), while different categories may merge through shared context or texture (Krishnakumar et al., 2021; Compton et al., 2023). The issue is therefore not only which loss is used after the backbone, but what geometry the backbone exposes to that loss.

As illustrated in Figure 1, we take a compositional tokenorganization view of this problem. Instead of asking a global embedding to carry every visual detail, we first express patch tokens through a small set of learnable primitives (Tang et al., 2025; Li et al., 2023). The primitives act as reusable coordinates for local visual evidence, and the patch-to-primitive assignments form a spatial primitive field over the image. Since the primitive set is intentionally compact, the construction introduces a low-rank bottleneck that suppresses redundant variation while retaining recurring semantic structure. Unknown categories do not require entirely new feature axes; they can appear as new compositional activation patterns and spatial arrangements of primitives shared with known categories.

Based on this idea, we introduce Compositional Primitive Fields (CPF-GCD), a lightweight module inserted between a vision backbone and a GCD head. CPF learns an image-conditioned primitive basis, refines it through token-primitive interactions, and produces a fused patch-toprimitive assignment used to update the token representation. The module does not replace existing GCD heads or losses. Instead, it changes the representation they receive: before classification or clustering, patch tokens are reorganized through a compact low-rank primitive field and then folded back into the global image representation.

Our contributions are:

• A representation-side perspective on GCD. We identify the backbone-to-head interface as a limiting factor in category discovery and argue that token geometry should be structured before applying clustering, contrastive learning, or prototype classification.

• A primitive-based token organizer. We propose CPF, a drop-in token organizer that rewrites patch representations through learnable primitives, token-primitive interactions, and adaptive assignment fusion. This gives existing GCD pipelines a compact intermediate space without modifying their heads or objectives.

• Cross-framework validation and diagnostics. We evaluate CPF with representative prototype-based, clustering-based, and contrastive-based GCD methods. The results show consistent gains on known and novel categories, while additional analyses on rank, entropy, attention, and category-number estimation explain how the low-rank primitive field improves discovery.

Overall, this paper shifts part of the GCD design problem from the output head to the representation interface. By reorganizing patch tokens into low-rank primitive fields before discovery, CPF provides a simple way to make pre-trained visual features more suitable for open-world category formation.

## 2. Related Work

Discovery heads and training signals. Generalized category discovery (Vaze et al., 2022) studies recognition when annotations cover only a subset of the categories that will appear at training time. A common recipe is to keep a strong visual encoder and improve the module that turns its features into labels or clusters. Some methods use learnable classifiers or prototypes (Wen et al., 2023; Cao et al., 2024; Vaze et al., 2023), while others rely on clusteringstyle inference (Chiaroni et al., 2023; Zhao et al., 2023; Choi et al., 2024; Rastegar et al., 2024b). Their supervision signals are also diverse, including contrastive objectives (Chen et al., 2020; He et al., 2020), transport-based matching (Fini et al., 2021), consistency regularization (Tarvainen & Valpola, 2017; Sohn et al., 2020), and promptbased calibration (Zhang et al., 2023; Wang et al., 2024). These designs have substantially advanced the discovery head, but they usually consume the backbone representation as given. CPF is complementary to this line of work: it reorganizes the feature stream before the head sees it, so the same downstream objectives can operate on a cleaner token organization.

What is passed from the backbone. Many recent GCD pipelines inherit representations from self-supervised vision transformers, especially DINO-style encoders (Caron et al., 2021; Park & Kim, 2022). Such backbones provide both a global image descriptor and a set of patch tokens. The global descriptor is convenient for classification, yet it can hide the local cues that separate fine-grained or previously unseen classes. Patch tokens expose more evidence, but they also mix foreground parts, textures, and background context in a high-dimensional space (Izmailov et al., 2022). This creates a mismatch: the discovery loss is asked to form semantic groups from features whose local structure has not been explicitly prepared for grouping. Our method targets this interface between encoder and head. Instead of replacing the classifier, clustering rule, or loss, CPF rewrites patch tokens through the compact low-rank primitive fields and then returns the refined tokens to the ordinary GCD pipeline.

Reusable units for visual evidence. Compositional recognition has long suggested that objects and categories can be described through reusable parts, attributes, or local configurations (Biederman, 1987; Lake et al., 2015). Related ideas appear in slot attention and object-centric learning, where visual content is separated into a small set of entities or factors (Locatello et al., 2020; Dai et al., 2026). Generative models also make heavy use of latent factors and token dictionaries to construct visual content (Van Den Oord et al., 2017; Ramesh et al., 2021). CPF adapts this intuition to discriminative discovery rather than generation: its primitives are not output objects, but reusable units that form a lowrank bottleneck for organizing patch evidence. This view is also consistent with the broader observation that compact, well-conditioned representations can improve discrimination (Yerxa et al., 2023; Papyan et al., 2020; Kothapalli et al., 2023). The key difference is operational: CPF implements the compactness as an image-conditioned token rewriting step that can be inserted into existing GCD systems.

## 3. Problem Formulation

GCD objective. Let $\mathcal { D } _ { L } = \{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n _ { L } }$ be labeled data from the known label set $\mathcal { D } _ { K }$ , and let $\mathcal { D } _ { U } = \{ x _ { j } \} _ { j = 1 } ^ { n _ { U } }$ <sub>1</sub> be unlabeled data whose samples may come from either known or unknown classes. The unknown label set $y _ { U }$ is disjoint from $\mathcal { V } _ { K } ,$ , and labels in $\mathcal { \mathrm { { y } } } _ { U }$ are never observed during training. A GCD model must therefore solve two coupled tasks: assign samples from known classes to their correct labels, and partition samples from unknown classes into coherent groups. Existing pipelines usually attach a discovery head $h _ { \phi }$ to a visual backbone $f _ { \theta }$ and optimize supervised and unsupervised objectives jointly. Our method leaves this head and its loss unchanged, and instead modifies the representation delivered to it.

Backbone-head interface. For an input image x, a transformer backbone produces N patch tokens,

$$
\mathbf {X} = \left[ \mathbf {x} _ {1}, \dots , \mathbf {x} _ {N} \right] ^ {\top} \in \mathbb {R} ^ {N \times D},\tag{1}
$$

Each $\mathbf { x } _ { i }$ stores the D-dimensional feature of one image patch. Standard GCD heads consume a global summary derived from these tokens, such as a class token or pooled feature. This interface is convenient, but it hides how local visual evidence is arranged before discovery. If X contains many redundant or noisy directions, then clustering and contrastive objectives are applied only after the difficult geometry has already been inherited.

Primitive reparameterization. We use a compact token reparameterization before the GCD head. Specifically, CPF introduces a learnable primitive codebook $\mathbf { P } \in \mathbb { R } ^ { M \times D }$ and a token-to-primitive assignment matrix $\mathbf { A } \in \mathbb { R } ^ { N \times M }$ , with $M \ll \operatorname* { m i n } \{ N , D \}$ :

$$
\mathbf {X} \approx \mathbf {A P}.\tag{2}
$$

Here, the rows of P serve as reusable basis elements for local visual evidence, while each row of A records how a patch token is assigned by that basis. Notably, in CPF, the primitive field is instantiated by such assignments, which describe how primitives are spatially activated over patch tokens and further refined with each other, as detailed in Section 4. This view does not require assigning a separate representation subspace to every novel class. Instead, known and novel categories may differ by how they compositionally activate and combine the same compact set of primitives.

Design consequence. The low-rank bottleneck in Eq. 2 changes where discovery begins. Rather than presenting the GCD head with an unconstrained token cloud, we first rewrite the patch representation through primitive assignment fields and decode it back into a refined token set. The primitive codebook reduces redundant variation, while the assignment matrix preserves patch-level organization, which later instantiates the primitive field used for token rewriting. This gives the downstream GCD objective a more regular representation to cluster or classify, without changing the objective itself.

## 4. Methodology

Following the formulation above, CPF implements the primitive reparameterization in Eq. 2 as a compositional token rewriting block. Given the backbone token matrix X, it constructs an image-conditioned primitive codebook, summarizes token evidence into primitive states, refines these states through lightweight interactions, and decodes the refined primitive information back to the patch-token space. The output has the same dimensionality as the original token matrix, so any GCD head can consume it without architectural changes. As summarized in Figure 2, CPF consists of three stages: (i) image-conditioned primitive codebook construction, which builds the codebook and initial tokento-primitive assignment; (ii) primitive evidence refinement, which accumulates and refines token evidence in the primitive space through token-primitive and primitive-primitive interactions; and (iii) assignment fusion and token rewriting, which fuses assignment cues and decodes the refined primitive mixture back to the patch-token space.

## 4.1. Image-Conditioned Primitive Codebook

Let $\mathbf { X } \in \mathbb { R } ^ { N \times D }$ denote the patch-token matrix, where N is the number of patches, and D is the feature dimension. The first part of CPF prepares the two ingredients required by the reparameterization: a small primitive codebook and a provisional token-to-primitive assignment. The codebook supplies the reusable directions, while the assignment specifies which tokens support each direction in the current image.

![](images/8ade077b606e087cd20a13e156e7e593c13bddaeba2593987b8b7dc7524e16bc.jpg)  
Figure 2. Overview of CPF. CPF is placed at the backbone-head interface. It rewrites noisy patch tokens through a compact primitive codebook and adaptive assignment fields, then returns a refined token representation to a standard GCD head.

Dataset-level codebook. We maintain a learnable primitive base

$$
\mathbf {P} _ {\text { base }} \in \mathbb {R} ^ {M \times D},\tag{3}
$$

where M is the number of primitive slots. The rows of $\mathbf { P } _ { \mathrm { b a s e } }$ are shared across the training set and are optimized jointly with the rest of the model. They can capture recurring local evidence such as object parts, textures, or attributes. The value of M controls the capacity of the bottleneck: a larger codebook can represent finer variation, whereas a smaller one forces more aggressive compression. Under the approximation $\mathbf { X } \approx \mathbf { A P }$ , M also upper-bounds the effective rank of the rewritten token representation.

Image-conditioned adaptation. A fixed codebook is too rigid for images with different poses, scales, and backgrounds. We therefore allow the shared base to receive a small image-conditioned offset. We first summarize X with a compact context descriptor using mean and max pooled token statistics, and map the descriptor to an additive codebook update:

$$
\Delta \mathbf {P} = \Phi (\operatorname{MeanMax} (\mathbf {X})) \in \mathbb {R} ^ {M \times D},\tag{4}
$$

where Φ is a learned linear map followed by reshaping. The codebook used for the current image is

$$
\mathbf {P} = \mathbf {P} _ {\text { base }} + \Delta \mathbf {P}.\tag{5}
$$

This keeps the primitive vocabulary shared, but lets each image slightly adjust the slot descriptors before token assignment.

Initial token-to-primitive assignment. Given the adapted codebook, CPF computes a first assignment by comparing tokens with primitives. We project tokens into the primitive space,

$$
\hat {\mathbf {X}} = \mathbf {X} W \in \mathbb {R} ^ {N \times D},\tag{6}
$$

where $W \in \mathbb { R } ^ { D \times D }$ is trainable, and score token-primitive compatibility by

$$
S = \frac {\hat {\mathbf {X}} \mathbf {P} ^ {\top}}{\sqrt {D}} \in \mathbb {R} ^ {N \times M}.\tag{7}
$$

The initial assignment is

$$
\mathbf {A} _ {\text { prior }} (i, m) = \frac {\exp (S _ {i , m})}{\sum_ {j = 1} ^ {N} \exp (S _ {j , m})}, \quad \mathbf {A} _ {\text { prior }} \in \mathbb {R} ^ {N \times M}.\tag{8}
$$

We normalize over tokens for each primitive, so every primitive slot distributes a unit amount of support across the image. This column-wise normalization makes tokens compete for each primitive slot. As a consequence, primitives tend to concentrate on informative regions instead of spreading uniformly over all patches. We use $\mathbf { A } _ { \mathrm { p r i o r } }$ as the prior assignment for the next step.

## 4.2. Primitive Evidence Refinement

The codebook stage provides P and $\mathbf { A } _ { \mathrm { p r i o r } }$ . Together they give a first approximation $\mathbf { X } \approx \mathbf { A } _ { \mathrm { p r i o r } } \mathbf { \hat { P } }$ . This approximation is useful but still shallow: it is obtained from pairwise affinity and does not yet let primitives collect broader token evidence or coordinate with other primitives. CPF therefore performs relational refinement in the primitive space before decoding back to tokens.

Token evidence pooling. We first gather token evidence into M primitive states using the prior assignment:

$$
\mathbf {P} ^ {0} = \mathbf {A} _ {\text { prior }} ^ {\top} \mathbf {X} \in \mathbb {R} ^ {M \times D}.\tag{9}
$$

The m-th row of $\mathbf { P } ^ { 0 }$ is the token summary assigned to primitive $m .$ . This converts the potentially large token set into a compact collection of primitive states, which is cheaper and more stable to reason over.

Token-conditioned update. The pooled states are then allowed to query the original tokens once more. We project primitive states and tokens into attention subspaces:

$$
Q _ {\mathrm{TP}} = \mathbf {P} ^ {0} W _ {q}, \quad K _ {\mathrm{TP}} = \mathbf {X} W _ {k}, \quad V _ {\mathrm{TP}} = \mathbf {X} W _ {v}.\tag{10}
$$

The primitive-to-token interaction is

$$
\mathbf {W} _ {\mathrm{TP}} = \mathrm{softmax} \bigg (\frac {Q _ {\mathrm{TP}} K _ {\mathrm{TP}} ^ {\top}}{\sqrt {d _ {h}}} \bigg),\tag{11}
$$

and the primitive states are updated as

$$
\mathbf {P} ^ {1} = \mathbf {P} ^ {0} + \gamma \mathbf {W} _ {\mathrm{TP}} V _ {\mathrm{TP}}.\tag{12}
$$

This second pass lets each primitive recover useful evidence that may have been under-weighted by the initial assignment. The residual scale $\gamma$ controls the strength of this tokenconditioned correction.

Primitive-state consolidation. Finally, primitives exchange information with each other to remove redundancy and make the codebook states more coherent. Starting from $\mathbf { P } ^ { 1 }$ , we compute

$$
Q _ {\mathrm{PP}} = \mathbf {P} ^ {1} W _ {q} ^ {\prime}, \quad K _ {\mathrm{PP}} = \mathbf {P} ^ {1} W _ {k} ^ {\prime}, \quad V _ {\mathrm{PP}} = \mathbf {P} ^ {1} W _ {v} ^ {\prime}.\tag{13}
$$

The primitive-to-primitive kernel is

$$
\mathbf {W} _ {\mathrm{PP}} = \mathrm{softmax} \left(\frac {Q _ {\mathrm{PP}} K _ {\mathrm{PP}} ^ {\top}}{\sqrt {d _ {h}}}\right),\tag{14}
$$

which gives the final refined primitive states

$$
\mathbf {P} ^ {2} = \mathbf {W} _ {\mathrm{PP}} V _ {\mathrm{PP}}.\tag{15}
$$

In practice, both primitive-to-token and primitive-toprimitive interactions use H interaction heads. For compact notation, Eqs. 11–15 omit the head index and are written as the concatenated and linearly projected multi-head messages in $\mathbb { R } ^ { M \times D }$

Therefore, the full refinement path is

$$
\mathbf {X} \xrightarrow {\text { aggregation }} \mathbf {P} ^ {0} \xrightarrow {\text { refinement }} \mathbf {P} ^ {1} \xrightarrow {\text { contraction }} \mathbf {P} ^ {2}\tag{16}
$$

which turns raw token evidence into a compact set of refined primitive states.

![](images/da764fbe80721dd443d0b227bcb16d6ee5e31473eaeff7e38acaa6b63d1c15be.jpg)  
Figure 3. Assignment fusion and token rewriting in Section 4.3.

## 4.3. Assignment Fusion and Token Rewriting

The refinement stage produces $\mathbf { P } ^ { 2 } \in \mathbb { R } ^ { M \times D }$ . To update the original patch tokens, CPF also needs a final tokento-primitive assignment. As illustrated in Figure 3, we derive one assignment from the initial compatibility scores and another from the refinement dynamics, fuse them, and decode primitive messages back to tokens.

Readout from refinement dynamics. The interaction in Eq. 11 produces multi-head weights $W _ { \mathrm { T P } } ^ { ( h ) } \in \mathbb { R } ^ { M \times N }$ . Although these weights are used to update primitives from tokens, they also indicate which tokens actually contributed to each primitive during refinement. We convert this evidence into a token-to-primitive assignment by transposing the interaction map, averaging over heads, and normalizing over primitives:

$$
\mathbf {A} _ {\text { data }} ^ {(i, m)} = \operatorname{Softmax} _ {m} \left(\frac {1}{H} \sum_ {h = 1} ^ {H} W _ {\mathrm{TP}} ^ {(h, m, i)}\right),\tag{17}
$$

Here, data assignment $\mathbf { A } _ { \mathrm { d a t a } } \in \mathbb { R } ^ { N \times M }$ records how strongly token i is linked to primitive m after the refinement step. Compared with $\mathbf { A } _ { \mathrm { p r i o r } } .$ , it is less tied to the initial score and more tied to the measured token-primitive exchange.

Confidence-weighted fusion. The two assignments play different roles. $\mathbf { A } _ { \mathrm { p r i o r } }$ is stable because it comes directly from the adapted codebook and local token scores; $\mathbf { A } _ { \mathrm { d a t a } }$ is adaptive because it comes from the refinement trajectory. CPF combines them in log-potential space using two learnable scalar weights:

$$
\mathbf {Z} = \alpha_ {\text { prior }} \log \mathbf {A} _ {\text { prior }} + \alpha_ {\text { data }} \log \mathbf {A} _ {\text { data }}.\tag{18}
$$

After exponentiation and row normalization, the final assignment is

$$
\mathbf {A} (i, m) = \frac {\exp (\mathbf {Z} (i , m))}{\sum_ {m ^ {\prime} = 1} ^ {M} \exp (\mathbf {Z} (i , m ^ {\prime}))}.\tag{19}
$$

Primitive-to-token decoding. We transform the refined primitives with a lightweight MLP

$$
\mathbf {P} _ {\text { refined }} = \operatorname{MLP} (\mathbf {P} ^ {2}) \in \mathbb {R} ^ {M \times D},\tag{20}
$$

and use A to decode primitive messages to the patch-token space:

$$
\mathbf {X} _ {\text { refined }} = \mathbf {X} + \operatorname{MLP} (\mathbf {A} \cdot \mathbf {P} _ {\text { refined }}).\tag{21}
$$

The residual connection keeps the original backbone signal, while the decoded primitive mixture injects the compact, denoised structure learned by CPF. The output $\bf { X } _ { \mathrm { { r e f i n e d } } }$ remains an $N \times D$ token matrix, so it can replace X wherever a standard GCD pipeline expects patch tokens.

Integration into GCD pipelines. For image-level discovery heads, we update the class token by pooling the refined patch tokens:

$$
[ \mathbf {C L S} ] ^ {*} = [ \mathbf {C L S} ] + \operatorname{Mean} (\mathbf {X} _ {\text {refined}}).\tag{22}
$$

The enriched representation is then passed to the original GCD head, whether it is prototype-based, clustering-based, or contrastive. Thus CPF changes only the representation interface and does not require modifying the downstream loss or head design.

## 5. Experiments

Through comprehensive experiments, we seek to answer three core questions: (1) Effectiveness. Does CPF-GCD consistently improve discovery performance across diverse benchmarks? (2) Generality. Is our framework a truly generic, plug-and-play module that boosts various existing GCD baselines? (3) Mechanism. What drives the performance gains? Is it the assignments, or their dynamic fusion?

## 5.1. Experimental Setup

Datasets and Evaluation Protocols. We evaluate CPF-GCD on both coarse- and fine-grained benchmarks, including fine-grained domains (CUB-200 (Wah et al., 2011), Stanford Cars (Krause et al., 2013), FGVC Aircraft (Maji et al., 2013)) and coarse-grained generic classification (CIFAR-10 (Krizhevsky et al., 2009), CIFAR-100, ImageNet-100 (Deng et al., 2009)). In experiments, we strictly follow the splits and protocols established in (Vaze et al., 2021; 2022). We report clustering accuracy (ACC) on known/old, novel/new, and all classes, aligning predictions via the Hungarian algorithm (Kuhn, 1955).

Implementation Details. We implement the proposed CPF-GCD as a lightweight, plug-and-play module designed to seamlessly integrate with diverse GCD architectures. Consistent with previous works (Vaze et al., 2022; Zhang et al., 2023; Pu et al., 2023), we employ a frozen DINO ViT-B/16 (Caron et al., 2021) pre-trained on ImageNet-1K (Deng et al., 2009) as the feature extractor, utilizing the sequence of patch tokens as input to our model. For

CPF, we set the number of learnable primitives M and interaction heads H to 12 on fine-grained datasets, and to 16 on coarse-grained datasets. To rigorously evaluate the generality of our approach, we strictly adhere to the original training hyperparameters and optimization schedules of each host baseline, introducing no method-specific tuning.

Baselines. We select mainstream GCD paradigms as the plug-and-play target schemes for CPF, including contrastive learning-based methods (CMS (Choi et al., 2024), SelEx (Rastegar et al., 2024b)) and prototype learning-based methods (SimGCD (Wen et al., 2023), LegoGCD (Cao et al., 2024)). For comprehensive comparison, we further include GCD (Vaze et al., 2022), µGCD (Vaze et al., 2024), PromptCAL (Zhang et al., 2023), DCCL (Pu et al., 2023), InfoSieve (Rastegar et al., 2024a), AMEND (Banerjee et al., 2024), PIM (Chiaroni et al., 2023), ProtoGCD (Ma et al., 2025), APL (Dai et al., 2025) with SimGCD and Con-GCD (Tang et al., 2025) with SPTNet (Wang et al., 2024) in our evaluation.

## 5.2. Results

Results on fine-grained datasets. Table 1 validates the efficacy of CPF-GCD across three fine-grained benchmarks. As a plug-and-play module, CPF-GCD consistently outperforms four baselines. Notably, it boosts average accuracy by 4.75% on Stanford-Cars, with a 6.00% gain on novel classes across all baselines. Significant gains in novel class discovery reach 8.1% with LegoGCD, and even with the strong SelEx baseline, CPF-GCD achieves a peak accuracy of 79.8% on CUB-200. These results confirm that modeling the low-rank compositional primitives effectively resolves fine-grained differences that global features struggle with.

Results on coarse-grained datasets. To ensure CPF-GCD’s effectiveness on global tasks, we evaluate it on CIFAR-10, CIFAR-100, and ImageNet-100 (Table 2). Unlike specialized fine-grained models, CPF-GCD remains strong across coarse-grained benchmarks. On ImageNet-100, it boosts novel class discovery by 2.03% on average, with a peak of 3.1% with SelEx. While slight fluctuations in known class accuracy (e.g., with LegoGCD) occur, this suggests that CPF reallocates part of the representation capacity from known-class specialization to novel-cluster formation. These results show that CPF-GCD enriches the feature space with structural details, complementing global semantics for generic object classification.

## 5.3. Ablation Study

Ablation on the number of primitives. As shown in Figure 4, we observe a consistent performance plateau across fine-grained benchmarks. Our method achieves optimal and stable results within the range of $M \in [ 1 2 , 1 6 ]$ , revealing that CPF-GCD is robust to changes in primitive capacity and does not require dataset-specific fine-tuning, validating its generalizability for diverse open-world scenarios. A core design philosophy of CPF-GCD is to minimize the dependency on sensitive hyperparameters, ensuring its utility as a practical, plug-and-play solution. To this end, we couple the primitive capacity M with the number of attention heads H (setting M = H) in practice.

Table 1. Performance comparison on the fine-grained semantic shift benchmark. The best and runner-up results are marked in bold black text and underlined, respectively. For △, positive growth is indicated by bold green text, while negative growth is shown in standard red text, explicitly marked with a minus sign.

<table><tr><td rowspan="2">Method</td><td colspan="3">CUB-200</td><td colspan="3">FGVC-Aircraft</td><td colspan="3">Stanford-Cars</td><td colspan="3">Average</td></tr><tr><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td>GCD</td><td>51.3</td><td>56.6</td><td>48.7</td><td>45.0</td><td>41.1</td><td>46.9</td><td>39.0</td><td>57.6</td><td>29.9</td><td>45.1</td><td>51.8</td><td>41.8</td></tr><tr><td>PromptCAL</td><td>62.9</td><td>64.4</td><td>62.1</td><td>52.2</td><td>52.2</td><td>52.3</td><td>50.2</td><td>70.1</td><td>40.6</td><td>55.1</td><td>62.2</td><td>51.7</td></tr><tr><td>AMEND</td><td>64.9</td><td>75.6</td><td>59.6</td><td>52.8</td><td>61.8</td><td>48.3</td><td>56.4</td><td>73.3</td><td>48.2</td><td>58.0</td><td>70.2</td><td>52.0</td></tr><tr><td>μGCD</td><td>65.7</td><td>68.0</td><td>64.6</td><td>53.8</td><td>55.4</td><td>53.0</td><td>56.5</td><td>68.1</td><td>50.9</td><td>58.7</td><td>63.8</td><td>56.2</td></tr><tr><td>ProtoGCD</td><td>63.2</td><td>68.5</td><td>60.5</td><td>56.8</td><td>62.5</td><td>53.9</td><td>53.8</td><td>73.7</td><td>44.2</td><td>57.9</td><td>68.2</td><td>52.9</td></tr><tr><td>InfoSieve</td><td>69.4</td><td>77.9</td><td>65.2</td><td>56.3</td><td>63.7</td><td>52.5</td><td>55.7</td><td>74.8</td><td>46.4</td><td>60.5</td><td>72.1</td><td>54.7</td></tr><tr><td>APL</td><td>64.5</td><td>68.1</td><td>62.1</td><td>56.6</td><td>60.2</td><td>54.8</td><td>60.1</td><td>77.6</td><td>51.2</td><td>60.4</td><td>68.6</td><td>56.0</td></tr><tr><td>ConGCD</td><td>68.1</td><td>68.5</td><td>67.8</td><td>59.7</td><td>61.3</td><td>59.2</td><td>59.1</td><td>79.0</td><td>49.8</td><td>62.3</td><td>69.6</td><td>58.9</td></tr><tr><td>SimGCD</td><td>61.2</td><td>65.8</td><td>58.9</td><td>54.5</td><td>59.3</td><td>52.1</td><td>54.6</td><td>72.8</td><td>45.7</td><td>56.8</td><td>66.0</td><td>52.2</td></tr><tr><td>+ CPF-GCD</td><td>63.9</td><td>65.5</td><td>63.1</td><td>55.8</td><td>59.1</td><td>54.2</td><td>57.6</td><td>73.7</td><td>49.8</td><td>59.1</td><td>66.1</td><td>55.7</td></tr><tr><td></td><td>2.7</td><td>-0.3</td><td>4.2</td><td>1.3</td><td>-0.2</td><td>2.1</td><td>3.0</td><td>0.9</td><td>4.1</td><td>2.3</td><td>0.1</td><td>3.5</td></tr><tr><td>LegoGCD</td><td>61.9</td><td>71.9</td><td>56.9</td><td>54.6</td><td>62.7</td><td>50.6</td><td>53.7</td><td>72.2</td><td>44.9</td><td>56.7</td><td>68.9</td><td>50.8</td></tr><tr><td>+ CPF-GCD</td><td>66.0</td><td>73.5</td><td>62.0</td><td>56.3</td><td>62.3</td><td>53.2</td><td>59.7</td><td>73.6</td><td>53.0</td><td>60.7</td><td>69.8</td><td>56.1</td></tr><tr><td></td><td>4.1</td><td>1.6</td><td>5.1</td><td>1.7</td><td>-0.4</td><td>2.6</td><td>6.0</td><td>1.4</td><td>8.1</td><td>4.0</td><td>0.9</td><td>5.3</td></tr><tr><td>CMS</td><td>65.7</td><td>75.8</td><td>60.7</td><td>50.8</td><td>62.1</td><td>45.1</td><td>51.3</td><td>73.5</td><td>40.6</td><td>55.9</td><td>70.5</td><td>48.8</td></tr><tr><td>+ CPF-GCD</td><td>66.8</td><td>74.8</td><td>62.9</td><td>51.5</td><td>60.4</td><td>47.0</td><td>57.9</td><td>78.2</td><td>48.0</td><td>58.7</td><td>71.1</td><td>52.6</td></tr><tr><td></td><td>1.1</td><td>-1.0</td><td>2.2</td><td>0.7</td><td>-1.7</td><td>1.9</td><td>6.6</td><td>4.7</td><td>7.4</td><td>2.8</td><td>0.6</td><td>3.8</td></tr><tr><td>SelEx</td><td>75.6</td><td>77.3</td><td>74.7</td><td>61.1</td><td>68.7</td><td>57.3</td><td>55.5</td><td>77.6</td><td>44.8</td><td>64.1</td><td>74.5</td><td>58.9</td></tr><tr><td>+ CPF-GCD</td><td>79.8</td><td>80.5</td><td>79.4</td><td>61.8</td><td>69.0</td><td>58.1</td><td>58.9</td><td>79.0</td><td>49.2</td><td>66.8</td><td>76.2</td><td>62.2</td></tr><tr><td></td><td>4.2</td><td>3.2</td><td>4.7</td><td>0.7</td><td>0.3</td><td>0.8</td><td>3.4</td><td>1.4</td><td>4.4</td><td>2.7</td><td>1.7</td><td>3.3</td></tr><tr><td>Avg. △</td><td>3.03</td><td>0.88</td><td>4.05</td><td>1.10</td><td>-0.50</td><td>1.85</td><td>4.75</td><td>2.10</td><td>6.00</td><td>2.95</td><td>0.83</td><td>3.98</td></tr></table>

![](images/479f4365100d7da7617946b9676171bec8085d5e8508299157c597e0601dce86.jpg)  
(a) CUB

![](images/698bed00846a0905fd7051fcfca971a8a3614cefa8b51e10bd0bdee6a69f3091.jpg)  
(b) Stanford Cars

![](images/73e3f008c3e0b9b78d35b7010bbbfba8410937a7849fdfb79781fe5988e60a08.jpg)  
(c) FGVC Aircraft  
Figure 4. Hyperparameter sensitivity of the number of primitives (M ) and interaction heads (H).

Ablation on components. We conduct an ablation study (Table 3) to evaluate our method’s key components. First, we replace the refined aggregation $\mathbf { M e a n } ( \mathbf { X } _ { \mathrm { r e f i n e d } } )$ with a global average Mean(X) (denoted as w/ patches), which shows marginal improvement over SimGCD but underperforms CPF-GCD, emphasizing the importance of primitive field token rewriting. Removing the data-assignment $\left( \mathbf { A } _ { \mathrm { d a t a } } \right)$ causes the sharpest drop on CUB-200 (from 63.1% to 60.2%), while excluding the prior-assignment $( \mathbf { A } _ { \mathrm { p r i o r } } )$ degrades novel class accuracy on Stanford Cars to 48.9%.

Finally, omitting the learnable fusion gates α results in suboptimal performance, confirming the need for dynamic arbitration between prior stability and data-driven specificity.

## 6. Further Empirical Analysis

We analyze CPF from three perspectives: geometric properties, semantic interpretability, and computational efficiency.

CPF Decreases Von Neumann Entropy. We use von Neumann Entropy (VNE) (Boes et al., 2019) to examine the geometric transformation induced by CPF-GCD. VNE is calculated from the autocorrelation matrix R of the feature space, and it measures the uniformity of spectral energy distribution. High entropy typically indicates isotropic distribution or unstructured redundancy, which can hinder unsupervised learning in GCD by entangling semantic cues with high-frequency noise. As shown in Figure 5, CPF reduces both VNE and effective rank compared to SelEx, leading to better discovery performance. This suggests that CPF filters out redundant dimensions and reorganizes high-rank token features into a compact, structured primitive field. We interpret this entropy reduction as a spectral purification process that refines the representation into the semantic evidence essential for distinguishing categories.

CPF Offers Precise Representation Distribution Estimation. Table 4 shows that CPF-GCD provides much more accurate estimates of unseen category cardinality compared to CMS. CPF-GCD reduces estimation errors by half on CIFAR-100 and cuts the error margin from 18% to 11.5% on CUB-200. This improvement highlights how grounding the representation space in a low-rank compositional field mitigates geometric confusion and prevents semantic clusters from merging incorrectly.

Table 2. Performance comparison on the coarse-grained classification benchmark.

<table><tr><td rowspan="2">Method</td><td colspan="3">CIFAR-10</td><td colspan="3">CIFAR-100</td><td colspan="3">ImageNet-100</td><td colspan="3">Average</td></tr><tr><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td>GCD</td><td>91.5</td><td>97.9</td><td>88.2</td><td>73.0</td><td>76.2</td><td>66.5</td><td>74.1</td><td>89.8</td><td>66.3</td><td>79.5</td><td>88.0</td><td>73.7</td></tr><tr><td>PIM</td><td>94.7</td><td>97.4</td><td>93.3</td><td>78.3</td><td>84.2</td><td>66.5</td><td>83.1</td><td>95.3</td><td>77.0</td><td>85.4</td><td>92.3</td><td>78.9</td></tr><tr><td>PromptCAL</td><td>97.9</td><td>96.6</td><td>98.5</td><td>81.2</td><td>84.2</td><td>75.3</td><td>83.1</td><td>92.7</td><td>78.3</td><td>87.4</td><td>91.2</td><td>84.0</td></tr><tr><td>DCCL</td><td>96.3</td><td>96.5</td><td>96.9</td><td>75.3</td><td>76.8</td><td>70.2</td><td>80.5</td><td>90.5</td><td>76.2</td><td>84.0</td><td>87.9</td><td>81.1</td></tr><tr><td>ProtoGCD</td><td>97.3</td><td>95.3</td><td>98.2</td><td>81.9</td><td>82.9</td><td>80.0</td><td>84.0</td><td>92.2</td><td>79.9</td><td>87.7</td><td>90.1</td><td>86.0</td></tr><tr><td>InfoSieve</td><td>94.8</td><td>97.7</td><td>93.4</td><td>78.3</td><td>82.2</td><td>70.5</td><td>80.5</td><td>93.8</td><td>73.8</td><td>84.5</td><td>91.2</td><td>79.2</td></tr><tr><td>APL</td><td>97.1</td><td>94.9</td><td>98.2</td><td>80.9</td><td>81.6</td><td>79.5</td><td>83.2</td><td>92.6</td><td>78.5</td><td>87.1</td><td>89.7</td><td>85.4</td></tr><tr><td>ConGCD</td><td>97.4</td><td>95.2</td><td>98.5</td><td>82.5</td><td>85.9</td><td>77.3</td><td>85.9</td><td>93.4</td><td>82.5</td><td>88.6</td><td>91.5</td><td>86.1</td></tr><tr><td>SimGCD</td><td>96.5</td><td>96.1</td><td>96.7</td><td>80.1</td><td>82.1</td><td>76.0</td><td>83.2</td><td>94.0</td><td>77.8</td><td>86.6</td><td>90.7</td><td>83.5</td></tr><tr><td>+ CPF-GCD</td><td>97.4</td><td>95.5</td><td>98.3</td><td>80.9</td><td>82.0</td><td>78.8</td><td>84.2</td><td>92.7</td><td>80.0</td><td>87.5</td><td>90.1</td><td>85.7</td></tr><tr><td></td><td>0.9</td><td>-0.6</td><td>1.6</td><td>0.8</td><td>-0.1</td><td>2.8</td><td>1.0</td><td>-1.3</td><td>2.2</td><td>0.9</td><td>-0.6</td><td>2.2</td></tr><tr><td>LegoGCD</td><td>96.8</td><td>96.1</td><td>97.2</td><td>81.9</td><td>83.5</td><td>78.7</td><td>86.3</td><td>94.5</td><td>82.1</td><td>88.3</td><td>91.4</td><td>86.0</td></tr><tr><td>+ CPF-GCD</td><td>97.5</td><td>95.8</td><td>98.3</td><td>82.8</td><td>83.3</td><td>81.9</td><td>86.1</td><td>93.1</td><td>82.6</td><td>88.8</td><td>90.7</td><td>87.6</td></tr><tr><td></td><td>0.7</td><td>-0.3</td><td>1.1</td><td>0.9</td><td>-0.2</td><td>3.2</td><td>-0.2</td><td>-1.4</td><td>0.5</td><td>0.5</td><td>-0.7</td><td>1.6</td></tr><tr><td>CMS</td><td>95.2</td><td>96.9</td><td>94.4</td><td>82.4</td><td>86.0</td><td>75.3</td><td>83.1</td><td>94.1</td><td>77.6</td><td>86.9</td><td>92.3</td><td>82.4</td></tr><tr><td>+ CPF-GCD</td><td>96.2</td><td>97.0</td><td>95.7</td><td>83.2</td><td>85.8</td><td>78.1</td><td>85.1</td><td>95.4</td><td>79.9</td><td>88.2</td><td>92.7</td><td>84.6</td></tr><tr><td></td><td>1.0</td><td>0.1</td><td>1.3</td><td>0.8</td><td>-0.2</td><td>2.8</td><td>2.0</td><td>1.3</td><td>2.3</td><td>1.3</td><td>0.4</td><td>2.2</td></tr><tr><td>SelEx</td><td>95.5</td><td>97.1</td><td>94.6</td><td>82.1</td><td>85.1</td><td>76.2</td><td>83.3</td><td>94.4</td><td>77.7</td><td>87.0</td><td>92.2</td><td>82.8</td></tr><tr><td>+ CPF-GCD</td><td>96.8</td><td>97.7</td><td>96.3</td><td>82.0</td><td>84.4</td><td>77.3</td><td>85.5</td><td>94.7</td><td>80.8</td><td>88.1</td><td>92.3</td><td>84.8</td></tr><tr><td></td><td>1.3</td><td>0.6</td><td>1.7</td><td>-0.1</td><td>-0.7</td><td>1.1</td><td>2.2</td><td>0.3</td><td>3.1</td><td>1.1</td><td>0.1</td><td>2.0</td></tr><tr><td>Avg. △</td><td>0.98</td><td>-0.05</td><td>1.43</td><td>0.60</td><td>-0.30</td><td>2.48</td><td>1.25</td><td>-0.28</td><td>2.03</td><td>0.95</td><td>-0.20</td><td>2.00</td></tr></table>

Table 3. Ablations on components.

<table><tr><td rowspan="2">Components</td><td colspan="3">CUB-200</td><td colspan="3">Aircraft</td><td colspan="3">S-Cars</td></tr><tr><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td><td>All</td><td>Old</td><td>New</td></tr><tr><td>SimGCD</td><td>61.2</td><td>65.8</td><td>58.9</td><td>54.5</td><td>59.3</td><td>52.1</td><td>54.6</td><td>72.8</td><td>45.7</td></tr><tr><td>w/ patches</td><td>62.1</td><td>65.8</td><td>60.3</td><td>54.8</td><td>59.3</td><td>52.6</td><td>55.4</td><td>70.9</td><td>49.1</td></tr><tr><td>+ CPF-GCD</td><td>63.9</td><td>65.5</td><td>63.1</td><td>55.8</td><td>59.1</td><td>54.2</td><td>57.6</td><td>73.7</td><td>49.8</td></tr><tr><td>w/o  $\alpha$ </td><td>63.2</td><td>64.3</td><td>62.7</td><td>55.0</td><td>58.2</td><td>53.5</td><td>56.6</td><td>74.1</td><td>48.1</td></tr><tr><td>w/o  $A_{prior}$ </td><td>62.6</td><td>62.3</td><td>62.7</td><td>54.9</td><td>57.9</td><td>53.4</td><td>56.7</td><td>72.6</td><td>48.9</td></tr><tr><td>w/o  $A_{data}$ </td><td>62.2</td><td>66.2</td><td>60.2</td><td>55.1</td><td>58.0</td><td>53.7</td><td>57.1</td><td>73.2</td><td>49.3</td></tr></table>

![](images/36460a5f054d96a32e81f2de7d30ab35385abf976f60c16869ecb77793b4a5be.jpg)  
(a) rank<sub>99</sub>(R) of fine-grained datasets

![](images/c1bc8eb1764854676953939e34a7690b4513cce3952802be817450ca1757657b.jpg)  
(b) H(R) of fine-grained datasets  
Figure 5. Comparison between rank<sub>99</sub>(R) and H(R). Here, rank<sub>99</sub>(R) is the count of the largest eigenvalues needed to account for 99% of the total eigenvalue energy.

CPF Reshapes the Model’s Attention. By constraining patch tokens to a compact set of primitives, CPF-GCD transforms the attention mechanism from raw pixel correlations to interactions between primitive tokens. As shown in Figure 6, the baseline exhibits diffuse attention patterns, failing to separate the object of interest from irrelevant context. In contrast, CPF-GCD yields attention maps that better cover the main foreground objects, such as car bodies and aircraft fuselages or wings, while suppressing irrelevant background regions. This suggests that the learned primitive field encourages patch tokens to align with reusable object-level structures, thereby improving the signal-to-noise ratio for more robust category discovery.

![](images/809512bfb1885940d28ad3e1915d997e91dc1f5f44ac60e4bceafee24dc43e28.jpg)  
Figure 6. Visualization of attention maps.

Table 4. Estimated number and error rate.

<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR-100</td><td colspan="2">IN-100</td><td colspan="2">CUB-200</td><td colspan="2">S-Cars</td></tr><tr><td> $|\mathcal{Y}_{u}|$ </td><td>Er(%)</td><td> $|\mathcal{Y}_{u}|$ </td><td>Er(%)</td><td> $|\mathcal{Y}_{u}|$ </td><td>Er(%)</td><td> $|\mathcal{Y}_{u}|$ </td><td>Er(%)</td></tr><tr><td>Ground Truth</td><td>100</td><td>-</td><td>100</td><td>-</td><td>200</td><td>-</td><td>196</td><td>-</td></tr><tr><td>GCD</td><td>100</td><td>0</td><td>109</td><td>9</td><td>231</td><td>15.5</td><td>230</td><td>17.3</td></tr><tr><td>DCCL</td><td>146</td><td>46</td><td>129</td><td>29</td><td>172</td><td>14</td><td>192</td><td>2.04</td></tr><tr><td>CMS</td><td>92</td><td>8</td><td>105</td><td>5</td><td>164</td><td>18</td><td>153</td><td>21.9</td></tr><tr><td>+ CPF-GCD</td><td>96</td><td>4</td><td>97</td><td>3</td><td>177</td><td>11.5</td><td>160</td><td>18.4</td></tr></table>

![](images/f8ff6cb8b1f676f203dbefd711f84724ce27857423447fe445c29ac500b24827.jpg)  
Figure 7. Visualization of the primitives.

Table 5. Computational overhead.

<table><tr><td>Models</td><td>Params (M)</td><td>Training Time (s)</td><td>Inference Time (s)</td></tr><tr><td>SimGCD</td><td>92.10</td><td>27.87</td><td>10.61</td></tr><tr><td>+ CPF-GCD</td><td>99.62</td><td>30.10</td><td>11.25</td></tr><tr><td> $\triangle$ </td><td>+ 7.5</td><td>+ 8%</td><td>+ 6%</td></tr></table>

Primitives are Consistent Across Instances. To validate the semantic consistency of the learned primitive field, we visualize the spatial activation of primitives on ImageNet-100. As shown in Figure 7, the primitives consistently capture cross-category semantics. For example, Primitive #1 focuses on head regions, Primitive #5 attends to the main torso, and Primitive #10 associates with limbs and tails. These patterns indicate that primitives act as a visual alphabet of reusable parts, enabling the model to recognize novel categories as combinations of familiar structural elements, enhancing generalization in the open world.

Minimal Computational Overhead. As shown in Table 5, CPF adds approximately 7.5M of parameters to the overall model, with minimal impact on performance. Training time increases by only 8%, and inference time remains competitive. This efficiency comes from performing token organization through a compact low-rank primitive field, where interactions are mediated by the primitives rather than relying only on the original high-rank patch-token space. CPF-GCD offers consistent accuracy gains while being a lightweight and practical module.

## 7. Conclusion

In this work, we identified the unstructured, high-rank geometry of standard backbone representations as a fundamental bottleneck hindering Generalized Category Discovery and proposed Compositional Primitive Fields (CPF-GCD) to fundamentally reshape this latent space. By explicitly modeling visual data with a compact primitive codebook and spatial token-to-primitive assignment fields, CPF-GCD effectively filters out spurious noise, enabling novel categories to emerge naturally as distinct activation patterns and spatial configurations over a shared primitive vocabulary. Extensive experiments across diverse benchmarks confirm that CPF serves as a potent, plug-and-play inductive bias, delivering consistent performance gains, accurate cardinality estima tion, and minimal computational overhead. Our findings demonstrate that enforcing structural compositionality is a critical missing link in open-world recognition, providing a new direction for future research on dynamic primitive field construction for ever-expanding category spaces.

## References

Banerjee, A., Kallooriyakath, L. S., and Biswas, S. Amend: Adaptive margin and expanded neighborhood for efficient generalized category discovery. In Proceedings of the IEEE/CVF winter conference on applications of computer vision, pp. 2101–2110, 2024.

Biederman, I. Recognition-by-components: a theory of human image understanding. Psychological review, 94 (2):115, 1987.

Boes, P., Eisert, J., Gallego, R., Muller, M. P., and Wilming,¨ H. Von neumann entropy from unitarity. Physical review letters, 122(21):210402, 2019.

Cao, X., Zheng, X., Wang, G., Yu, W., Shen, Y., Li, K., Lu, Y., and Tian, Y. Solving the catastrophic forgetting problem in generalized category discovery. In CVPR, 2024.

Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J.,´ Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021.

Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In ICML, 2020.

Chiaroni, F., Dolz, J., Masud, Z. I., Mitiche, A., and Ben Ayed, I. Parametric information maximization for generalized category discovery. In ICCV, 2023.

Choi, S., Kang, D., and Cho, M. Contrastive mean-shift learning for generalized category discovery. In CVPR, 2024.

Compton, R., Zhang, L., Puli, A., and Ranganath, R. When more is less: Incorporating additional datasets can hurt performance by introducing spurious correlations. In Machine learning for healthcare conference, pp. 110–127. PMLR, 2023.

Dai, B., Fan, Z., Qi, Z., Lou, M., and Yu, Y. CGSA: Classguided slot-aware adaptation for source-free object de-

tection. In The Fourteenth International Conference on Learning Representations, 2026.

Dai, Q., Huang, H., Wu, Y., and Yang, S. Adaptive part learning for fine-grained generalized category discovery: A plug-and-play enhancement. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 25444–25453, 2025.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition. Ieee, 2009.

Fini, E., Sangineto, E., Lathuiliere, S., Zhong, Z., Nabi, M.,\` and Ricci, E. A unified objective for novel class discovery. In ICCV, 2021.

He, K., Fan, H., Wu, Y., Xie, S., and Girshick, R. Momentum contrast for unsupervised visual representation learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020.

Izmailov, P., Kirichenko, P., Gruver, N., and Wilson, A. G. On feature learning in the presence of spurious correlations. Advances in Neural Information Processing Systems, 35:38516–38532, 2022.

Kothapalli, V., Tirer, T., and Bruna, J. A neural collapse perspective on feature evolution in graph neural networks. Advances in Neural Information Processing Systems, 36: 14134–14191, 2023.

Krause, J., Stark, M., Deng, J., and Fei-Fei, L. 3d object representations for fine-grained categorization. In CVPRW, 2013.

Krishnakumar, A., Prabhu, V., Sudhakar, S., and Hoffman, J. Udis: Unsupervised discovery of bias in deep visual recognition models. In BMVC, volume 1, pp. 2, 2021.

Krizhevsky, A., Hinton, G., et al. Learning multiple layers of features from tiny images. 2009.

Kuhn, H. W. The hungarian method for the assignment problem. Naval research logistics quarterly, 2(1-2):83– 97, 1955.

Lake, B. M., Salakhutdinov, R., and Tenenbaum, J. B. Human-level concept learning through probabilistic program induction. Science, 350(6266):1332–1338, 2015.

Li, C., Li, Z., Jing, C., Jia, Y., and Wu, Y. Exploring the effect of primitives for compositional generalization in vision-and-language. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 19092–19101, 2023.

Locatello, F., Weissenborn, D., Unterthiner, T., Mahendran, A., Heigold, G., Uszkoreit, J., Dosovitskiy, A., and Kipf, T. Object-centric learning with slot attention. Advances in neural information processing systems, 33:11525–11538, 2020.

Ma, S., Zhu, F., Zhang, X.-Y., and Liu, C.-L. Protogcd: Unified and unbiased prototype learning for generalized category discovery. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2025.

Maji, S., Rahtu, E., Kannala, J., Blaschko, M., and Vedaldi, A. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.

Papyan, V., Han, X., and Donoho, D. L. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, 2020.

Park, N. and Kim, S. How do vision transformers work? arXiv preprint arXiv:2202.06709, 2022.

Pu, N., Zhong, Z., and Sebe, N. Dynamic conceptional contrastive learning for generalized category discovery. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 7579–7588, 2023.

Ramesh, A., Pavlov, M., Goh, G., Gray, S., Voss, C., Radford, A., Chen, M., and Sutskever, I. Zero-shot text-toimage generation. In International conference on machine learning, pp. 8821–8831. Pmlr, 2021.

Rastegar, S., Doughty, H., and Snoek, C. Learn to categorize or categorize to learn? self-coding for generalized category discovery. Advances in Neural Information Processing Systems, 36, 2024a.

Rastegar, S., Salehi, M., Asano, Y. M., Doughty, H., and Snoek, C. G. Selex: Self-expertise in finegrained generalized category discovery. arXiv preprint arXiv:2408.14371, 2024b.

Sohn, K., Berthelot, D., Carlini, N., Zhang, Z., Zhang, H., Raffel, C. A., Cubuk, E. D., Kurakin, A., and Li, C.-L. Fixmatch: Simplifying semi-supervised learning with consistency and confidence. Advances in neural information processing systems, 33:596–608, 2020.

Tang, L., Huang, K., Chen, C., Yuan, Y., Li, C., Tu, X., Ding, X., and Huang, Y. Dissecting generalized category discovery: Multiplex consensus under self-deconstruction. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 297–307, 2025.

Tarvainen, A. and Valpola, H. Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results. Advances in neural information processing systems, 30, 2017.

Van Den Oord, A., Vinyals, O., et al. Neural discrete representation learning. Advances in neural information processing systems, 30, 2017.

Vaze, S., Han, K., Vedaldi, A., and Zisserman, A. Open-set recognition: A good closed-set classifier is all you need? 2021.

Vaze, S., Han, K., Vedaldi, A., and Zisserman, A. Generalized category discovery. In CVPR, 2022.

Vaze, S., Vedaldi, A., and Zisserman, A. No representation rules them all in category discovery. Advances in Neural Information Processing Systems, 36:19962–19989, 2023.

Vaze, S., Vedaldi, A., and Zisserman, A. No representation rules them all in category discovery. Advances in Neural Information Processing Systems, 36, 2024.

Wah, C., Branson, S., Welinder, P., Perona, P., and Belongie, S. The caltech-ucsd birds-200-2011 dataset. 2011.

Wang, H., Vaze, S., and Han, K. Sptnet: An efficient alternative framework for generalized category discovery with spatial prompt tuning. In ICLR, 2024.

Wen, X., Zhao, B., and Qi, X. Parametric classification for generalized category discovery: A baseline study. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16590–16600, 2023.

Yerxa, T., Kuang, Y., Simoncelli, E., and Chung, S. Learning efficient coding of natural images with maximum manifold capacity representations. Advances in Neural Information Processing Systems, 36:24103–24128, 2023.

Zhang, H., Zhan, T., Basu, S., and Davidson, I. A framework for deep constrained clustering. Data Mining and Knowledge Discovery, 35(2):593–620, 2021.

Zhang, S., Khan, S., Shen, Z., Naseer, M., Chen, G., and Khan, F. S. Promptcal: Contrastive affinity learning via auxiliary prompts for generalized novel category discovery. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3479– 3488, 2023.

Zhao, B., Wen, X., and Han, K. Learning semi-supervised gaussian mixture models for generalized category discovery. In ICCV, 2023.