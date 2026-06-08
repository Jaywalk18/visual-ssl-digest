# Inside the Visual Mind: Neuroscience-Motivated Concept Circuits for Interpreting and Steering Vision Transformers

Tang Li 1 Yanlin Chen 1 Mengmeng Ma 2 Xi Peng 2

# Abstract

Despite high accuracy, Vision Transformer (ViT) predictions can be driven by spurious cues, raising the need to understand their inner workings before safe deployment. Sparse autoencoders (SAEs) provide a promising lens for decomposing model representations into human-interpretable concepts, yet adapting SAE-based interpretation to ViTs remains challenging due to limited control over concept coverage and subjective, nonscalable feature interpretation. To fill the gaps, motivated by neuroscience-inspired principles, we propose ViSAE, a mechanistic interpretability toolbox for understanding ViT inner workings through concept circuits. ViSAE consists of three components: (1) A probing suite with 64K images and a 16K visually grounded concept vocabulary, improving concept coverage efficiency by 20× over ImageNet and interpretation accuracy by 28.7% over existing concept sets. (2) Top-down concept reading and Bottomup circuit tracing algorithms that automatically recover ViT inner workings via concept circuits. (3) Applications for auditing and steering ViT behavior. Through concept editing, ViSAE improves the worst-group accuracy on WaterBirds by 48.2%, outperforming existing methods by 23.8%. Our data and code: https://github. com/deep-real/ViSAE.

# 1. Introduction

Machine learning (ML) models, such as Vision Transformers (ViTs) (Dosovitskiy et al., 2020), have become ubiquitous

1Department of Computer & Information Science, University of Delaware, Newark, DE, USA 2Department of Computer Science, University of Virginia, Charlottesville, VA, USA. Correspondence to: Tang Li <tangli@udel.edu>, Xi Peng <naq5rd@virginia.edu>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

![](images/faca579fbfbf03c86c2a0374ecf0aba29f192e7f64d3f59b686ae95f5b90239f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Existing IML"] --> B["Input"]
    B --> C["ML Model"]
    C --> D["School Bus"]
    D --> E["Our Concept Circuit"]
    E --> F["Interactions"]
    E --> G["Semantic Concepts"]
    style A fill:#f9f,stroke:#333
    style E fill:#bbf,stroke:#333
    style F fill:#dfd,stroke:#333
    style G fill:#dfd,stroke:#333
```
</details>

Figure 1. Existing interpretable machine learning methods (IML) mainly identify where the evidence is, while our concept circuits reveal how concepts interact across layers to support a prediction.

foundations of high-impact systems. Despite their strong empirical performance, their internal mechanisms remain opaque to users, revealing little about how information is represented, transformed, and used inside the network. This opacity makes it difficult to diagnose failures, identify unsafe reasoning patterns, or intervene when a model relies on brittle or spurious features (Arjovsky et al., 2019; Sagawa et al., 2019). Conventional interpretable machine learning (IML) methods (Fig. 1) often explain ML models by attributing predictions to input features (Selvaraju et al., 2017; Lundberg & Lee, 2017) or internal neurons (Ghorbani & Zou, 2020; Bau et al., 2017). However, attribution-based correlations provide limited insight into the model’s underlying computation and decision process (Olah et al., 2020). A natural question arises: Can we “read the mind” of a large vision model? Or, can we understand the inner workings of vision models (layer by layer, neuron by neuron, from input to output) using human-understandable concepts?

Recent advances in Mechanistic Interpretability (MI) provide a promising foundation for this goal, especially in language Transformers. A typical approach is to train Sparse Autoencoders (SAEs), which decompose polysemantic, internal representations into more monosemantic, humaninterpretable concept features (Bricken et al., 2023; Huben et al., 2023; Zou et al., 2023). However, directly porting SAE-based workflows from language to vision runs into two practical bottlenecks. (1) Concept coverage lacks control. Existing practice (Rao et al., 2024; Stevens et al., 2025; Thasarathan et al., 2025) often trains SAEs using large generic datasets such as ImageNet (Deng et al., 2009). This imposes limited control over which concepts are covered and how densely each abstraction level is sampled. As a result, the concepts learned by SAEs are often biased toward dominant dataset content, such as objects (Tab. 1), narrowing their interpretability to deep layer semantics and leaving low- and mid-layer concepts underrepresented. (2) Interpretation is hard to scale. Unlike language features, the vision features learned by SAEs are often not naturally interpretable to humans. A common workflow retrieves the top-activating images for each SAE feature and summarizes them into concept labels (Lim et al., 2025; Pach et al., 2026). However, such summarizations can be subjective, summarizer-dependent, and hard to scale across tens of thousands of SAE features.

To address these gaps and enable holistic analysis of ViT inner workings, we propose ViSAE, a comprehensive mechanistic interpretability toolbox. ViSAE integrates probing data for SAE training and interpretation, algorithms for concept reading and circuit tracing, and practical applications for model auditing and steering. Critically, its design is motivated by neuroscience-inspired principles for studying biological vision and neural computation. Specifically, our toolbox answers three research questions:

(1) Data: How can we improve concept coverage for SAE training and interpretation? To interpret ViTs layer by layer, SAE training requires probing examples that cover the full spectrum of visual processing. Motivated by the hierarchical organization of the human visual system (Goodale & Milner, 1992; Carandini et al., 2005), we organize visual concepts into four abstraction levels: Primitive, Intermediate, Object, and Scene. To train SAEs, we curate 64K probing examples from seven vision datasets, selected so that each example’s primary content aligns with one of these levels. To interpret SAEs, we construct a 16K vocabulary grounded in our images and spanning the same hierarchy (e.g., “stripes”, “skimming”). Our design improves concept coverage efficiency by 20× compared with the standard ImageNet baseline. Moreover, when using our concept set to interpret the same SAEs, we achieve a 28.7% gain in interpretation accuracy over existing concept sets.

(2) Algorithm: How can we reveal the inner workings of ViTs? While SAEs extract discrete concepts from internal representations, ViT mechanisms depend not only on which concepts are present, but also on how concepts interact across layers (Fig. 1). This aligns with two complementary views in cognitive science for understanding biological intelligence: the Hopfieldian view emphasizes transformations in representational spaces, while the Sherringtonian view emphasizes connections among neural units

and structured pathways (Barack & Krakauer, 2021). Motivated by these views, we propose a two-stage tracing algorithm. (i) Top-down concept reading: trains an SAE at each Transformer layer and maps learned features to our concept vocabulary through the vision-language embedding space of CLIP (Radford et al., 2021), avoiding manual summarization. (ii) Bottom-up circuit tracing: estimates cross-layer causal effects through counterfactual interventions, producing a directed interaction graph of concepts. Our interpretations faithfully capture model behavior, outperforming existing counterparts by 23.8% in downstream steering tasks.

(3) Applications: How can concept circuits improve trust in ViTs? Our toolbox enables diagnostic auditing and corrective steering of ViTs. For auditing, it enables users to trace internal decision-making processes of the model, localize the visual evidence of concepts on pixels, and diagnose and summarize model failure modes. More importantly, for steering, it offers a set of conceptual “knobs” to precisely control model behavior by editing concepts within representations. For example, by turning down spuriously correlated concepts (e.g., land backgrounds), our method improves the worst-group accuracy on the WaterBirds (Sagawa et al., 2019) dataset by 48.2%.

Our Contributions: Unlike most existing efforts that propose another SAE variant, our ViSAE toolbox takes a datacentric perspective, providing the missing infrastructure needed to train and interpret SAEs for holistic analysis of ViT inner workings. Our toolbox consists of: (1) A neuroscience-motivated probing suite (64K images, 16K concepts) for SAE training and auto-interpretation. (2) A two-stage causal tracing algorithm for layer-wise discovery of concept circuits within ViTs. (3) Extensive empirical validation, including SAE benchmarking and applications in representation auditing and steering.

# 2. Method

In this section, we first review the basics of SAEs (Sec. 2.1), then introduce our probing suite (Sec. 2.2), followed by how it is used to train SAEs and to trace concept circuits (Sec. 2.3), and finally how to use the toolbox to audit and steer ViTs (Sec. 2.4). See overview in Fig. 2.

# 2.1. Preliminary

Sparse Autoencoders (SAEs) (Ng et al., 2011; Bricken et al., 2023) were proposed to interpret polysemantic neurons by formulating it as a sparse dictionary learning problem (Olshausen & Field, 1997). The objective is to learn an overcomplete set of sparse, disentangled basis features (i.e., concepts) that can reconstruct the input data through linear combination (Thasarathan et al., 2025). Concretely, an

![](images/a29b76a9bf91e008b8579b44fcf0e9d43c13a7da232a0f996682bba3d4604d7a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Probing Image Set"] --> B["ViT"]
    C["Concept Set"] --> D["Encoder"]
    D --> E["Dec"]
    E --> F["Reconstructed Representation"]
    B --> G["Representation"]
    D --> H["Sparse Autoencoder (SAE)"]
    H --> I["Reconstructed Representation"]
    style A fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style B fill:#ccf,stroke:#333
    style D fill:#cfc,stroke:#333
    style E fill:#fcc,stroke:#333
    style F fill:#ffc,stroke:#333
```
</details>

![](images/39927501e43468cf07f38c189ca6ee1dae6dad1cce2eec234936f937122ed28f.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Input Image"] --> B["ViT"]
    B --> C["Prediction Green Traffic Light"]
    C --> D["Applications (Sec.2.4)"]
    D --> E["Auditing Concept Localization"]
    E --> F["Failure Mode Analysis"]
    F --> G["Steering"]
    
    subgraph Input
        B1["Layer 0"] --> B2["Layer 1"] --> B3["Layer 2"] --> B4["..."]
        B5["Layer 11"] --> B6["Layer 10"] --> B7["light"]
        B7 --> B8["bus"]
        B8 --> B9["street"]
        B9 --> B10["child"]
        B10 --> B11["green"]
        B11 --> B12["green"]
        B12 --> B13["blue"]
        B13 --> B14["green"]
        B14 --> B15["green"]
        B15 --> B16["green"]
        B16 --> B17["green"]
        B17 --> B18["green"]
        B18 --> B19["green"]
        B19 --> B20["green"]
        B20 --> B21["green"]
        B21 --> B22["green"]
        B22 --> B23["green"]
        B23 --> B24["green"]
        B24 --> B25["green"]
        B25 --> B26["green"]
        B26 --> B27["green"]
        B27 --> B28["green"]
        B28 --> B29["green"]
        B29 --> B30["green"]
        B30 --> B31["green"]
        B31 --> B32["green"]
        B32 --> B33["green"]
        B33 --> B34["green"]
        B34 --> B35["green"]
        B35 --> B36["green"]
        B36 --> B37["green"]
        B37 --> B38["green"]
        B38 --> B39["green"]
        B39 --> B40["green"]
        B40 --> B41["green"]
        B41 --> B42["green"]
        B42 --> B43["green"]
        B43 --> B44["green"]
        B44 --> B45["green"]
        B45 --> B46["green"]
        B46 --> B47["green"]
        B47 --> B48["green"]
        B48 --> B49["green"]
        B49 --> B50["green"]
        B50 --> B51["green"]
        B51 --> B52["green"]
        B52 --> B53["green"]
        B53 --> B54["green"]
        B54 --> B55["green"]
        B55 --> B56["green"]
        B56 --> B57["green"]
        B57 --> B58["green"]
        B58 --> B59["green"]
        B59 --> B60["green"]
        B60 --> B61["green"]
        B61 --> B62["green"]
        B62 --> B63["green"]
        B63 --> B64["green"]
        B64 --> B65["green"]
        B65 --> B66["green"]
        style Input fill:#f9f,stroke:#333
    style Predicted
    style Learning
    style Steering
    style Learning
    style Steering
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Learning
    style Student
    subgraph Learning
        L1["Layer 2"] --> L2["Layer 1"]
        L2 --> L3["Layer 1"]
        L3 --> L4["Layer 1"]
        L4 --> L5["Layer 1"]
        L5 --> L6["Layer 1"]
        L6 --> L7["Layer 1"]
        L7 --> L8["Layer 1"]
        L8 --> L9["Layer 1"]
        L9 --> L10["Layer 1"]
        L10 --> L11["Layer 1"]
        L11 --> L12["Layer 1"]
        L12 --> L13["Layer 1"]
        L13 --> L14["Layer 1"]
        L14 --> L15["Layer 1"]
        L15 --> L16["Layer 1"]
        L16 --> L17["Layer 1"]
        L17 --> L18["Layer 1"]
        L18 --> L19["Layer 1"]
        L19 --> L20["Layer 1"]
    end
    
    subgraph Learning
        M1["Layer 3"] --> M2["Layer 4"]
        M2 --> M3["Layer 4"]
        M3 --> M4["Layer 4"]
        M4 --> M5["Layer 4"]
        M5 --> M6["Layer 4"]
        M6 --> M7["Layer 4"]
        M7 --> M8["Layer 4"]
        M8 --> M9["Layer 4"]
        M9 --> M10["Layer 4"]
        M10 --> M11["Layer 4"]
        M11 --> M12["Layer 4"]
        M12 --> M13["Layer 4"]
        M13 --> M14["Layer 4"]
        M14 --> M15["Layer 4"]
        M15 --> M16["Layer 4"]
        M16 --> M17["Layer 4"]
        M17 --> M18["Layer 4"]
        M18 --> M19["Layer 4"]
        M19 --> M20["Layer 4"]
    end
    
    subgraph Learning
        N1["Layer 2"] --> N2["Layer 2"]
        N2 --> N3["Layer 2"]
        N3 --> N4["Layer 2"]
        N4 --> N5["Layer 2"]
        N5 --> N6["Layer 2"]
        N6 --> N7["Layer 2"]
        N7 --> N8["Layer 2"]
        N8 --> N9["Layer 2"]
        N9 --> N10["Layer 2"]
        N10 --> N11["Layer 2"]
        N11 --> N12["Layer 2"]
        N12 --> N13["Layer 2"]
        N13 --> N14["Layer 2"]
        N14 --> N15["Layer 2"]
        N15 --> N16["Layer 2"]
        N16 --> N17["Layer 2"]
        N17 --> N18["Layer 2"]
        N18 --> N19["Layer 2"]
    end
    
    subgraph Learning
        O1["Layer 7"] --> O2["Layer 7"]
        O2 --> O3["Layer 7"]
        O3 --> O4["Layer 7"]
        O4 --> O5["Layer 7"]
        O5 --> O6["Layer 7"]
        O6 --> O7["Layer 7"]
    end
    
    subgraph Learning
        P1["Layer 8"] --> P2["Layer 8"]
        P2 --> P3["Layer 8"]
        P3 --> P4["Layer 8"]
        P4 --> P5["Layer 8"]
    end
    
    subgraph Learning
    Q1["Layer 9"] --> Q2["Layer 9"]
    Q2 --> Q3["Layer 9"]
    Q3 --> Q4["Layer 9"]
    Q4 --> Q5["Layer 9"]
    Q5 --> Q6["Layer 9"]
    Q6 --> Q7["Layer 9"]
    Q7 --> Q8["Layer 9"]
    Q8 --> Q9["Layer 9"]
    Q9 --> Q10["Layer 9"]
    Q10 --> Q11["Layer 9"]
    Q11 --> Q12["Layer 9"]
    Q12 --> Q13["Layer 9"]
    Q13 --> Q14["Layer 9"]
    Q14 --> Q15["Layer 9"]
    Q15 --> Q16["Layer 9"]
    Q16 --> Q17["Layer 9"]
    Q17 --> Q18["Layer 9"]
    Q18 --> Q19["Layer 9"]
    Q19 --> Q20["Layer 9"]
    
    subgraph Learning
    R1["Layer 3"] -.-> S
    S -.-> T
    T -.-> U
    U -.-> V
    V -.-> W
    W -.-> X
    X -.-> Y
    Y -.-> Z
```
</details>

Figure 2. Overview of our ViSAE toolbox for interpreting ViT inner workings. Left: Motivated by the human visual cortex hierarchy, we construct a probing suite (64K images + 16K concepts) for SAE training and interpretation. Middle: Our top-down concept reading and bottom-up concept circuit tracing algorithms. Right: Our mechanistic view of ViT inner workings enables various downstream applications, such as concept localization, failure mode analysis, and model steering.

SAE consists of an encoder that expands the input dimensionality, namely $f : \mathbb { R } ^ { d }  \mathbb { R } ^ { m } , m > d ,$ , and a decoder $g : \mathbb { R } ^ { m }  \mathbb { R } ^ { d }$ . A vanilla ReLU-SAE is given by:

$$
\mathbf {h} = f (\mathbf {x}) = \operatorname{ReLU} \left(\mathbf {W} _ {\text { enc }} \mathbf {x} + \mathbf {b} _ {\text { enc }}\right),
$$

$$
\hat {\mathbf {x}} = g (\mathbf {h}) = \mathbf {W} _ {\mathrm{dec}} \mathbf {h} + \mathbf {b} _ {\mathrm{dec}}, \tag {1}
$$

where $\mathbf { W } _ { \mathrm { e n c } } , \mathbf { W } _ { \mathrm { d e c } } ^ { \top } \in \mathbb { R } ^ { m \times d }$ and $\mathbf { b } _ { \mathrm { e n c } } , \mathbf { b } _ { \mathrm { d e c } } \in \mathbb { R } ^ { m }$ . Note that in the decoder parameter matrix $\mathbf { W } _ { \mathrm { d e c } }$ , each column wi represents a learned basis feature, namely $\hat { \textbf { x } } =$ $\begin{array} { r } { \sum _ { i = 0 } ^ { m - 1 } h _ { i } \mathbf { w } _ { i } + \mathbf { b } _ { \mathrm { d e c } } } \end{array}$ . Its training objective minimizes the reconstruction error while enforcing sparsity in latent code:

$$
\mathcal {L} (\mathbf {x}) = \left| \left| \mathbf {x} - \hat {\mathbf {x}} \right| \right| _ {2} ^ {2} + \lambda \left| \left| \mathbf {h} \right| \right| _ {1}. \tag {2}
$$

Given the SAE, two factors largely affect interpretation quality. (1) Quality of the input representation x. The data distribution (e.g., images) that produces x governs what features are even learnable. As noted in Sec. 1, objectcentric datasets bias learning toward specific abstraction levels (Stevens et al., 2025; Thasarathan et al., 2025). (2) Interpretation method of SAE features. Existing works typically label a feature wi by inspecting its top-activating images (Thasarathan et al., 2025; Pach et al., 2026), but this process is subjective and hard to scale. We address these challenges in the following sections with a new probing suite and an automated interpretation method.

# 2.2. Neuroscience-Motivated Probing Suite

As discussed in Sec. 1, for SAE training, object-centric datasets often provide limited concept coverage for the full spectrum of visual processing. To fill this gap, we construct a probing suite that offers broad concept coverage motivated by the hierarchical organization of the human visual cortex from neuroscience (Goodale & Milner, 1992; Carandini et al., 2005; DiCarlo et al., 2012).

![](images/0a9ee0e897fb39e53c390f8e84d50dcf5ba3e19a68cd50e88694b8cd31e600ca.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Primary Visual Cortex"] --> B["Primitive-level: Colors, Edges, Curves, ..."]
    A --> C["Secondary Visual Cortex"]
    C --> D["Intermediate-level: Textures, Shapes, ..."]
    A --> E["Parietal Lobe"]
    E --> F["Scene-level: Motions, Relations, ..."]
    E --> G["Temporal Lobe"]
    G --> H["Object-level: Objects, Parts, ..."]
```
</details>

Figure 3. Our design mirrors the hierarchy of human visual cortex.

Background: visual cortex hierarchy. The human visual system processes information along abstraction levels. As shown in Fig. 3: (1) At the primitive level, the primary visual cortex encodes basic visual primitives, e.g., colors, edges, and curves. (2) At the intermediate level, secondary visual cortex integrates these primitives into more complex patterns e.g., textures, materials, and geometric shapes. (3) At the object level, these patterns are combined into identifiable entities like tables, airplanes, or animals, supporting object recognition in the temporal lobe. (4) Finally, at the scene level, higher-order regions represent actions, spatial relations, and interactions, enabling reasoning about context and events, often associated with the parietal lobe.

Construct probing image set. To maximize the concept coverage of SAE, we first collect probing images from seven vision datasets mirroring the hierarchy above. Specifically, (1) at the primitive level, we collect images from the DTD (Cimpoi et al., 2014) and Broden (Bau et al., 2017); (2) at the intermediate level, we collect images from Broden and ShapeNet (Chang et al., 2015); (3) at the object level, we collect images from ImageNet (Deng et al., 2009) and

Table 1. Comparison of concept coverage (%) across probing image sets. We obtain the concept coverage of images by calculating the CLIP embedding similarity between images and our ground truth concepts (details in Appendix B). As shown, our probing image set demonstrates superior concept coverage across all levels of visual abstraction, with over 20× higher coverage efficiency. 

<table><tr><td rowspan="2">Probing Image Set</td><td rowspan="2">Data Source</td><td rowspan="2"># of Images</td><td colspan="5">Concepts Covered by Images (%)</td><td rowspan="2">Coverage Efficiency ↑ (%/1K Images)</td></tr><tr><td>Primitive</td><td>Intermediate</td><td>Object</td><td>Scene</td><td>Avg.</td></tr><tr><td>ImageNet</td><td>ImageNet</td><td>1,281K</td><td>81.0</td><td>78.2</td><td>97.7</td><td>59.0</td><td>78.9</td><td>0.06</td></tr><tr><td>MSCOCO</td><td>MSCOCO</td><td>118K</td><td>69.6</td><td>65.4</td><td>80.4</td><td>63.1</td><td>69.6</td><td>0.59</td></tr><tr><td>Ours</td><td>Primitive Level: DTD, Broden; Intermediate Level: Broden, ShapeNet; Object Level: ImageNet, VisualGenome; Scene Level: Place365, MSCOCO;</td><td>64K</td><td>87.1</td><td>80.6</td><td>92.6</td><td>61.7</td><td>80.5</td><td>1.26</td></tr></table>

Visual Genome (Krishna et al., 2017); and (4) at the scene level, we collect images from Places365 (Zhou et al., 2017) and MSCOCO (Lin et al., 2014). However, naively aggregating images reduces SAE training efficiency while offering minimal concept coverage benefits. This is because repeated views cause SAEs to waste limited model capacity on the same high-frequency concepts, biasing the learned features. To address this issue, we prune the initial pool (121K raw image candidates) by removing one image from every pair with a cosine similarity greater than 0.85 in the CLIP-ViT-B-32 embedding space. As a result, our final set contains 64K probing images. Tab. 1 shows our superior concept coverage, outperforming the popular ImageNet baseline by 20× in coverage efficiency.

Construct concept set. To enable the proposed automatic interpretation (Sec. 2.3) of SAE features, a candidate pool of concepts with strong visual grounding is required. Existing vocabularies are typically mined from text, e.g., frequent n-grams from Google Books (Oikarinen & Weng, 2022) or LAION captions (Bhalla et al., 2024), which are typically skewed toward linguistically frequent terms and drift from the images. To reduce this bias, we generate concepts from the images themselves. Concretely, for each image in our probing set, we use GPT-5 (OpenAI, 2025) to annotate present concepts under the same four-level hierarchy. The resulting concept set contains 16K unique one- and two-gram concepts (Tab. 2). Compared with existing concept sets, our concepts are 20.6% less redundant and 26.2% more visually grounded (Tab. 3), and it outperforms existing concept counterparts in interpretation accuracy (Tab. 4; See details in Sec. 3.2).

Human Evaluations. To ensure the quality of the concept annotations, we further conduct human evaluations. We evaluate two metrics: Faithfulness, which measures whether concepts are visually grounded in the image, and Comprehensiveness, which measures whether they cover all visual abstraction levels. We randomly sample three groups of images (3×50) paired with our concept annotations and hire graduate students to evaluate all different groups on a 0-5 Likert scale. As shown in Tab. 2, our concepts are of high quality and the annotation process is consistent across

Table 2. Concept set statistics and human evaluation results. 

<table><tr><td>Level</td><td># of Concepts</td><td>Evaluator</td><td>Faithfulness</td><td>Completeness</td></tr><tr><td>Primitive</td><td>1,073</td><td>Human_A</td><td> $4.65 \pm 0.06$ </td><td> $4.72 \pm 0.04$ </td></tr><tr><td>Intermediate</td><td>1,723</td><td>Human_B</td><td> $4.83 \pm 0.20$ </td><td> $4.73 \pm 0.06$ </td></tr><tr><td>Object</td><td>10,534</td><td>Human_C</td><td> $4.90 \pm 0.05$ </td><td> $4.78 \pm 0.18$ </td></tr><tr><td>Scene</td><td>2,720</td><td></td><td></td><td></td></tr><tr><td>Total</td><td>16,050</td><td>Avg.</td><td>4.79</td><td>4.74</td></tr></table>

Table 3. Comparison of concept set quality. Redundancy: the proportion of concept pairs with CLIP text-embedding cosine similarity >0.8. Visually grounded: the proportion of conceptimage pairs with CLIP cosine similarity >0.3. 

<table><tr><td>Concept Set</td><td># of Concepts</td><td>Redundancy (↓)</td><td>Visually Grounded (↑)</td></tr><tr><td>LAION-freq</td><td>15K</td><td>0.419</td><td>0.602</td></tr><tr><td>Google-freq</td><td>20K</td><td>0.654</td><td>0.432</td></tr><tr><td>LaBo</td><td>10K</td><td>0.584</td><td>0.565</td></tr><tr><td>Ours</td><td>16K</td><td>0.213</td><td>0.864</td></tr></table>

images, as evidenced by an average rating above 4.7/5.

# 2.3. Concept Circuit Tracing Algorithm

Top-down concept reading. Although SAEs decompose polysemantic representations into disentangled features, interpreting the semantics of these features remains challenging. Existing methods typically retrieve top-activating samples and summarize them into a specific concept (Bills et al., 2023; Pach et al., 2026). However, for vision models, this process relies on subjective human review and does not scale. To address this issue, we introduce an automated method to “read” concepts directly from representations. Specifically, by leveraging the aligned embedding space of vision-language models, we map each SAE feature to the most semantically aligned textual concept in our concept set. Let $\mathbf { W } _ { \mathrm { d e c } }$ be the decoder weight matrix of a trained SAE, where each column wi is a basis feature. Using our probing image set ${ \mathcal { D } } _ { \mathrm { p r o b e } } = \{ x _ { 1 } , . . . , x _ { N } \}$ , we extract the feature activation vector $q _ { i } = \left[ h _ { i } ( \boldsymbol { x } _ { 1 } ) , h _ { i } ( \boldsymbol { x } _ { 2 } ) , \ldots , h _ { i } ( \boldsymbol { x } _ { N } ) \right] ^ { \top } \in$ $\mathbb { R } ^ { N }$ for neuron i over all images. For the concept set $\mathcal { D } _ { \mathrm { c o n c e p t } } = \{ c _ { 1 } , . . . , c _ { M } \}$ , we compute a concept activation matrix $P \in \mathbb { R } ^ { N \times M }$ using a VLM, e.g., CLIP (Radford et al., 2021), where $P _ { n m }$ is the embedding similarity between image $x _ { n }$ and concept $c _ { m }$ . We associate SAE feature i with concept $c _ { m }$ using the Soft Weighted Point-wise Mutual Information (Soft-WPMI) (Oikarinen & Weng, 2022):

$$
\operatorname{Sim} (i, c _ {m}) = \log \mathbb {E} _ {x \sim \mathcal {D} _ {\text { probe }}} [ \alpha_ {i} (x) \cdot P _ {x m} ] - \lambda \log p (c _ {m}), \tag {3}
$$

where $\begin{array} { r } { \alpha _ { i } ( x _ { n } ) = \frac { \exp ( q _ { i } [ n ] ) } { \sum _ { j = 1 } ^ { N } \exp ( q _ { i } [ j ] ) } } \end{array}$ is the softmax-normalized activation of neuron i, $\begin{array} { r } { p ( c _ { m } ) ~ = ~ \frac { 1 } { N } \sum _ { n = 1 } ^ { N } P _ { n m } } \end{array}$ is the marginal prevalence of concept $c _ { m } .$ , and $\lambda > 0$ controls the penalty for overly frequent concepts. The final concept label for SAE feature i is determined by:

$$
c ^ {*} (i) = \arg \max _ {c _ {m} \in \mathcal {D} _ {\text { concept }}} \operatorname{Sim} (i, c _ {m}). \tag {4}
$$

Bottom-up circuit tracing. Although we have mapped SAE features to concepts, how these discrete concepts relate to one another and how they compose into the final decision remains unclear. To address this, we trace the causal influence among concepts and toward the prediction from bottom up. Specifically, inspired by activation patching methods (Meng et al., 2022; Conmy et al., 2023), we define edges by quantifying the causal importance via counterfactual interventions. Let $\alpha _ { j } ^ { t }$ be the activation of a target concept $c _ { j } ^ { t }$ in a downstream target layer t. Consider a concept $c _ { i } ^ { s }$ extracted from source layer s via SAE, with activation $\alpha _ { i } ^ { s }$ . To measure its influence on a target concept $c _ { j } ^ { t }$ , we construct two layer s representations of the same input x: the original representation $r _ { \mathrm { c l e a n } }$ and a patched representation $r _ { \mathrm { p a t c h } } .$ , in which $c _ { i } ^ { s }$ is ablated by setting its activation $\alpha _ { i } ^ { s }$ to zero and reconstructing the representation via the SAE decoder. In this case, we define the causal influence of $c _ { i } ^ { s }$ on $c _ { j } ^ { t }$ by measuring the interventional effect (IE) (Pearl, 2001):

$$
\begin{array}{l} \mathrm{IE} _ {i \rightarrow j} ^ {s \rightarrow t} = \mathrm{IE} \left(\alpha_ {j} ^ {t}; c _ {i} ^ {s}; r _ {\text { clean }}, r _ {\text { patch }}\right) \\ = \alpha_ {j} ^ {t} (r _ {\text { clean }}) - \alpha_ {j} ^ {t} (r _ {\text { clean }} \mid \mathbf {d o} (\alpha_ {i} ^ {s} = \alpha_ {i} ^ {s} (r _ {\text { patch }})))  . \tag {5} \\ \end{array}
$$

We can also obtain the contribution of concept $c _ { i } ^ { s }$ to the final prediction $y$ by:

$$
\begin{array}{l} \mathrm{IE} _ {i \rightarrow y} ^ {s} = \mathrm{IE} (y; c _ {i} ^ {s}; r _ {\text { clean }}, r _ {\text { patch }}) \\ = y (x _ {\text { clean }}) - y (r _ {\text { clean }} \mid \mathrm{do} (\alpha_ {i} ^ {s} = \alpha_ {i} ^ {s} (r _ {\text { patch }}))). \\ \end{array}
$$

By repeating this procedure for all concepts over all layers, we obtain a directed graph where nodes represent concepts and edges are weighted by the causal importance of the target node regarding the final prediction. Concretely, the weight of an edge from $c _ { i } ^ { s }$ to $c _ { j } ^ { t }$ is given by $\mathrm { I E } _ { i  j } ^ { s  t } \cdot \mathrm { \bar { I E } } _ { j  y } ^ { t } .$ We model the ViT forward pass as a deterministic structural causal model (SCM), and our concept nodes refine this into a more fine-grained SCM on top of it. Consequently, all edges are directed and acyclic, $i . e . , s \to t$ only when $t > s ,$ . This concept circuit reveals how primitive features are progressively composed into intermediate patterns and, ultimately, high-level semantics that drive the model’s predictions.

# 2.4. Applications

Building upon the concept circuits, our ViSAE toolbox offers a practical toolkit for model analysis and intervention.

Auditing. (1) Trace information flow: Users can visualize the concept circuit for an arbitrary image, revealing the pathways of causal influence from low-level primitives to the final prediction (Fig. 5). (2) Localize concepts on pixels: Concretely, to localize $c _ { i }$ on images, we calculate the cosine similarities between $\mathbf { w } _ { i }$ and each image token $t _ { j } \in \mathbb { R } ^ { d }$ from the corresponding transformer layer. This generates a saliency map $\begin{array} { r } { h _ { i } = \frac { \left. \mathbf { w } _ { i } , t _ { j } \right. } { \| \mathbf { w } _ { i } \| _ { 2 } \| t _ { j } \| _ { 2 } } } \end{array}$ that highlights the regions where the concept $c _ { i }$ is most strongly activated (Fig. 6). (3) Diagnose failure modes: By comparing concept circuits from correct and incorrect predictions, users can systematically diagnose failure modes and determine whether errors arise from spurious cues or missing critical factors (Fig. 7).

Steering. Our ViSAE enables targeted control of model behavior by concept editing. (1) Suppress spurious concepts: Mitigate shortcut learning by setting the activation of an undesired concept to zero, effectively removing its influence from the computation graph. (2) Amplify robust concepts: Enhance the effect of desirable features by manually increasing their activation (Tab. 7).

# 3. Experiments

# 3.1. Benchmarking SAEs on Image Data

We systematically evaluate SAE variants on disentangling and reconstructing vision representations.

Settings: We evaluate representative SAEs, i.e., ReLU-SAE (Bricken et al., 2023), BatchTopK-SAE (Bussmann et al., 2024), Matryoshka-SAE (Bussmann et al., 2025), Gated-SAE (Rajamanoharan et al., 2024a), JumpReLU-SAE (Rajamanoharan et al., 2024b). For each SAE architecture, we train multiple variants with five expansion factors (2×, 4×, 8×, 16× and 32×) and five average $L _ { 0 }$ sparsities (8, 16, 32, 64 and 128), resulting in 25 instances in total. The expansion factor refers to the expansion ratio between the SAE latent dimension and input dimension. We train SAEs on the representations (CLS and image token separately, 2×12 SAEs in total) extracted from the residual stream of each layer in ViT (CLIP-ViT-B-32 unless noted), using our probing image set as input. For each layer, we train SAEs separately on cls tokens and image tokens. Details of these SAE architectures are in Appendix D.

Metrics: (1) $L _ { 0 }$ Sparsity: the average number of nonzero activations. (2) Reconstruction Error (RE): the mean squared error (MSE) between the input representation and the SAE reconstructed representation. (3) Decoder Orthogonality (DO) (Zaigrajew et al., 2025): the mean cosine similarity between each pair of decoder columns. This metric measures whether an SAE encodes concepts with distinct semantic meanings. (4) Dead Neuron (DN): the proportion of SAE basis features remaining consistently inactive (zero activation) across the whole training dataset. (5) Monosemanticity (MS) (Pach et al., 2026): whether each basis feature of an SAE consistently activates on images of the same semantics. Details of the metrics are in Appendix E.

![](images/6acfe2a8721015caf1940127c5de9c2c27d4dfd7a7cca1fc1b6b6e0e12bc3792.jpg)  
ReLU Gated JumpReLU BatchTopK Matryoshka

Figure 4. Benchmark results for expansion factor (8×). As shown, BatchTopK-SAE strikes a better trade-off across all metrics. Therefore, in subsequent experiments we use the BatchTopK-SAE with expansion factor = 8× and L0 Sparsity = 128. Full tables in Appendix G.   
Table 4. Comparison of Interpretation Accuracy. 

<table><tr><td rowspan="2">Probing Image Set</td><td rowspan="2">Concept Set</td><td colspan="3">Interpretation Accuracy (%)</td></tr><tr><td>Top-10</td><td>Top-20</td><td>Top-30</td></tr><tr><td>Broden</td><td>Broden</td><td>16.8</td><td>23.9</td><td>26.8</td></tr><tr><td rowspan="4">ImageNet</td><td>LaBo-ImageNet</td><td>15.9</td><td>18.4</td><td>20.3</td></tr><tr><td>Google-20K</td><td>7.3</td><td>10.2</td><td>12.0</td></tr><tr><td>LAION-15K</td><td>16.5</td><td>21.7</td><td>24.0</td></tr><tr><td>Ours-16K</td><td>32.2</td><td>43.5</td><td>50.7</td></tr><tr><td rowspan="3">MSCOCO</td><td>Google-20K</td><td>10.0</td><td>13.3</td><td>15.3</td></tr><tr><td>LAION-15K</td><td>16.3</td><td>20.4</td><td>23.7</td></tr><tr><td>Ours-16K</td><td>34.9</td><td>45.0</td><td>51.2</td></tr><tr><td rowspan="3">Ours-64K</td><td>Google-20K</td><td>10.7</td><td>14.4</td><td>16.7</td></tr><tr><td>LAION-15K</td><td>17.3</td><td>22.2</td><td>25.4</td></tr><tr><td>Ours-16K</td><td>36.6</td><td>47.6</td><td>54.1</td></tr></table>

Results: We observe that all SAE architectures share similar trends with respect to $L _ { 0 }$ sparsity on all five evaluation metrics. To balance the reconstruction quality and monosemanticity, we set the sparsity to 128 and the expansion factors to 8× in practice. Fig. 4 shows the benchmark results for expansion factor 8×. Implementation details and full tables are in Appendix F & G.

# 3.2. Evaluations of Interpretation Accuracy

We develop a new metric based on our probing suite to evaluate the interpretation accuracy of SAE training data and concept sets for auto-interpretation.

Table 5. Comparison with MLLM-based Summarization 

<table><tr><td>Auto-Interp. Method</td><td>Interp. Acc. (IoU)</td><td>Run Time</td></tr><tr><td>Qwen3.5-VL-9B</td><td>0.438 ± 0.249</td><td>1h 40min</td></tr><tr><td>Ours</td><td>0.432 ± 0.215</td><td>5 min</td></tr></table>

Settings: Similar to Sec. 3.1, we train SAEs on the representations (CLS and image tokens separately) extracted from the residual stream of each layer in CLIP-ViT-B-32. (1) We compare different training data (downsampled to 60K), including our probing images, ImageNet (Deng et al., 2009), and MSCOCO (Lin et al., 2014). (2) We compare using different concept sets to interpret the same SAEs, including our concept set (∼16K), LAION frequent words (15K) (Bhalla et al., 2024), and Google Books common English words (20K) (Oikarinen & Weng, 2022). (3) We compare using existing fine-grained interpretability datasets to train and interpret SAEs with our probing suite, including Broden (Bau et al., 2017) and LaBo (Yang et al., 2023). Note that LaBo only provides concept sets, so here we use ImageNet to train the SAEs and use LaBo’s concepts for ImageNet to interpret the SAE features (Sec. 2.3).

Metrics: We split the probing images into train and test sets (60K/4K). Since ground-truth concepts are available for our test images (Sec. 2.2), we calculate interpretation accuracy by measuring the fraction of ground-truth concepts covered within the top-K concepts read by SAEs from all layers. We use CLIP-based semantic match rather than a raw string match to mitigate vocabulary circularity.

Results: As shown in Tab. 4, in the top-30 extracted concepts, SAEs trained on our probing images consistently outperform existing datasets by 2.9% and 3.4%. Moreover, for the same SAEs, using our concept set consistently outperforms existing concept sets by 28.7% and 37.4%. Our probing suite outperforms existing fine-grained interpretability datasets by 27.3%.

![](images/53abc118936387471abb2c6535663a7a16a063bbb0574fa2d0f5498c1a131b2d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Layer 0
        direction TB
        L1["L1.&quot;orange"] --> L4["L4.&quot;orange surface"]
        L2["L2.&quot;white"] --> L7["L7.&quot;face"]
        L7["L7.&quot;paw"] --> L8["L8.&quot;dog face"]
        L1["L1.&quot;mable"] --> L7["L7.&quot;face"]
        L9["L9.&quot;cat"] --> L11["L11.&quot;cat"]
        L11["L11.&quot;cat"] --> L11a["L11.&quot;bark"]
        L11a["L11.&quot;bark"] --> Dog["Dog"]
        L11a["L11.&quot;dog"] --> Dog
        L11a["L11.&quot;bark"] --> Dog
        L11a["L11.&quot;cat"] --> Dog
    end

    subgraph Layer 11
        direction TB
        L11["L11.&quot;bark"] --> L7["L7.&quot;paw"]
        L11a["L11.&quot;dog"] --> L9["L9.&quot;cat"]
        L11a["L11.&quot;cat"] --> Cat["Cat"]
        L10["L10.&quot;cartoon dog"] --> Cat
        L7["L7.&quot;paw"] --> Cat
        L7a["L7.&quot;lying down"] --> Cat
        L6["L6.&quot;dotted fur"] --> Cat
    end

    subgraph Layer 6
        direction TB
        L4["L4.&quot;orange surface"] --> Cat
        L1["L1.&quot;orange"] --> Cat
    end

    subgraph Top-3 Activated Images
        ImageInput["Input"] --> ImageOutput["Top-3 Activated Images"]
        ImageOutput --> ImageOutput1["Image Output"]
        ImageOutput2["Image Output"]
        ImageOutput3["Image Output"]
        ImageOutput4["Image Output"]
        ImageOutput5["Image Output"]
        ImageOutput6["Image Output"]
        ImageOutput7["Image Output"]
        ImageOutput8["Image Output"]
        ImageOutput9["Image Output"]
        ImageOutput10["Image Output"]
        ImageOutput11["Image Output"]
        ImageOutput12["Image Output"]
        ImageOutput13["Image Output"]
        ImageOutput14["Image Output"]
        ImageOutput15["Image Output"]
        ImageOutput16["Image Output"]
        ImageOutput17["Image Output"]
        ImageOutput18["Image Output"]
        ImageOutput19["Image Output"]
        ImageOutput20["Image Output"]
    end

    subgraph Top-3 Activated Images
        ImageInput --> ImageOutput --> ImageOutput1 --> ImageOutput2 --> ImageOutput3 --> ImageOutput4 --> ImageOutput5 --> ImageOutput6 --> ImageOutput7 --> ImageOutput8 --> ImageOutput9 --> ImageOutput10 --> ImageOutput11 --> ImageOutput12 --> ImageOutput13 --> ImageOutput14 --> ImageOutput15
    end

    subgraph Top-3 Activated Images
        ImageInput --> ImageOutput --> ImageOutput2 --> ImageOutput3 --> ImageOutput4 --> ImageOutput5 --> ImageOutput6 --> ImageOutput7 --> ImageOutput8 --> ImageOutput9 --> ImageOutput10 --> ImageOutput11 --> ImageOutput12 --> ImageOutput13
    end

    subgraph Top-3 Activated Images
        ImageInput --> ImageOutput --> ImageOutput3 --> ImageOutput4 --> ImageOutput5 --> ImageOutput6 --> ImageOutput7 --> ImageOutput8 --> ImageOutput9 --> ImageOutput10 --> ImageOutput11 --> ImageOutput12 --> ImageOutput13
    end

    subgraph Top-3 Activated Images
        ImageInput --> ImageOutput --> ImageOutput3 --> ImageOutput4 --> ImageOutput5 --> ImageOutput6 --> ImageOutput7 --> ImageOutput8 --> ImageOutput9 --> ImageOutput0
    end

    subgraph Top-3 Activated Images
        ImageInput --> ImageOutput --> ImageOutput3 --> ImageOutput4 --> ImageOutput5 --> ImageOutput6 --> ImageOutput7 --> ImageOutput8 --> ImageOutput9 --> ImageOutput0
    end

    subgraph Top-3 Activated Images
        ImageInput --> ImageOutput --> ImageOutput3 --> ImageOutput4 --> ImageOutput5 --> ImageOutput6 --> ImageOutput7 --> ImageOutput8 --> ImageOutput0
    end

    subgraph Top-3 Activated Images
        ImageInput --> ImageOutput --> ImageOutput3 --> ImageOutput4 --> ImageOutput5 --> ImageOutput6 --> ImageOutput7 --> ImageOutput8 --> ImageOutput9
    end

    subgraph Top-3 Activated Images
        ImageInput --> ImageOutput --> ImageOutput3 --> ImageOutput4 --> ImageOutput5 --> ImageOutput6 --> ImageOutput7 --> ImageOutput8 --> ImageOutput9
    end

    subgraph Top-3 Activated Images
        ImageInput --> ImageOutput --> ImageOutput3 --> ImageOutput4 --> ImageOutput5 --> ImageOutput6 --> ImageOutput7 --> ImageOutput8 --> ImageOutput9
    end
```
</details>

Figure 5. Visualization of concept circuits. For an input image containing both a dog and a cat, our method traces the unique causal pathways leading to each prediction. The circuit for “dog” composes primitive and intermediate concepts (e.g., “orange” and “marble”) into high-level semantics (e.g., “bark” and “dog”). In contrast, the circuit for “cat” relies on a different set of concepts (e.g., “dotted fur”). Our method can faithfully audit the inner workings of ViT and highlight responsible concepts.

![](images/949d3827de077357493e340660b7ad81aa3559a1f9381d0ee893ac942a4a7226.jpg)

<details>
<summary>heatmap</summary>

| Depth Level | Category | Count |
|-------------|----------|-------|
| Primitive   | Original  | 5831  |
| Primitive   | Intermediate | 1509  |
| Primitive   | Object    | 1761  |
| Primitive   | Scene     | 1274  |
| Object     | Original  | 2114  |
| Object     | Intermediate | 135   |
| Object     | Object    | 2893  |
| Object     | Scene     | 2554  |
| Scene      | Original  | -     |
| Scene      | Intermediate | -     |
| Scene      | Object    | -     |
| Scene      | Scene     | -     |
</details>

Figure 6. Localize concepts in the pixel space. Notably, our method can even localize highly abstract semantics, such as “looking at”, by highlighting both the subject (i.e., the person) and the object involved (i.e., the paper).

Compared with MLLM-based summarization: We compare our auto-interpretation method against an MLLMbased baseline (Zhang et al., 2025). We implement it by using Qwen3.5-VL-9B to explain SAE features based on top-activating images. In their evaluation protocol, each interpreted concept label of the SAE feature will be grounded by GroundingDINO-SAM (Liu et al., 2024) on ImageNetval images, and its spatial agreement with the SAE feature activation map will be evaluated using IoU. As shown in Tab. 5, our interpretation achieves similar accuracy and is more stable, while requiring 20× less runtime per layer.

# 3.3. Auditing

In this section, we demonstrate how to use our toolbox to audit model behavior.

Trace decision-making processes. Fig. 5 shows the concept circuit examples traced by our method. Beyond faithfully identifying decision pathways, the circuits reveal a layer-wise progression that is similar to the human visual system: early layers detect low-level primitives (colors, textures), while deeper layers compose these cues into higherlevel semantics (objects, relations/motion).

Localize concepts on pixels. Qualitatively, Figs. 5 & 6 show examples of our concept localizations. Our method can accurately localize concepts across visual abstraction levels. Note that we do not manually choose the layer; the SAE activations determine it. For example, if an image strongly activates an SAE feature labeled “wooden texture” in its layer-3 representation, we use that layer-3 feature for localization. Quantitatively, as shown in Tab. 6, our heatmaps on the Quantus (Hedstrom et al.¨ , 2023) benchmark improve over the existing attribution-based method (Chefer et al., 2021) by 3.7% using the VOC2007 dataset in terms

![](images/9635f9333bc2a6283fef1b65db297bce4283ae82a66304a8f81137074963f962.jpg)

<details>
<summary>other</summary>

| Failure Images | Difference in SAE Concepts |
| --- | --- |
| Label: canoe | [“holding”, “people”, “mountain”, “water”, “shade”, _]
| Pred: paddle | [“gymnast”, “flying”, “bar”, “downward”, “indoor”, _]
| Label: parallel bar | [“gymnast”, “flying”, “bar”, “downward”, “indoor”, _]
| Pred: high bar (single) | [“gymnast”, “flying”, “bar”, “downward”, “indoor”, _]
Right vs. Wrong
Difference in SAE Concepts
Failure Modes Statistics (%) 
Part-Whole Confusion 15.8
Missing Key Concept 15.8
Spurious Co-occurrence 13.2
Misleading Motion 7.9
Age Bias 5.3
Background Shortcut 18.4
Color Bias 23.7
</details>

Figure 7. Failure mode analysis. We identify seven failure modes of CLIP on the ImageNet-val set. For example, CLIP tends to misclassify parallel bar images as high bars when the gymnast is “flying” or oriented “downward”.

Table 6. Comparison of heatmap localization accuracy. 

<table><tr><td>Method</td><td>Point Game</td><td>Attribution Localization</td></tr><tr><td>Chefer et al.</td><td>41.9</td><td>32.3</td></tr><tr><td>Ours</td><td>45.0</td><td>36.0</td></tr></table>

of localization accuracy.

Diagnose failure modes. Beyond instance-level auditing, understanding the failure patterns of the model at a global level is essential for improving its robustness. To this end, we demonstrate how to leverage our ViSAE to diagnose the failure modes of the CLIP-ViT-B-32 model on the ImageNet validation set. Specifically, we identify all 38 classes for which more than 40% of the images are consistently misclassified into the same incorrect class. For each such class, we use our trained SAEs to extract the most frequently activated concepts from both correctly and incorrectly classified images. By comparing these two concept groups, we summarize key semantic differences, eventually organizing them into seven distinct failure modes (Fig. 7). Building on these analyses, our ViSAE can be extended to support data quality control by identifying mislabeled, ambiguous, or systematically biased samples.

# 3.4. Steering

In this section, we demonstrate the capability of our ViSAE in steering model behavior.

Settings. We use the WaterBirds dataset (Sagawa et al., 2019) to evaluate robustness against spurious correlations between bird species (land/water) and their backgrounds. The training set amplifies this spurious correlation (e.g., waterbirds on water), while a 5% “worst-group” in the test set breaks it (e.g., waterbirds on land). A linear classifier trained on CLIP-ViT-B-32’s final-layer CLS tokens achieves only 50.3% accuracy on this worst group, confirming heavy reliance on background cues. We compare with Concept Bottleneck Models (CBM) (Koh et al., 2020), SpLiCE (Bhalla et al., 2024), DN-CBM (Rao et al., 2024), PCBM (Yuksek-

Table 7. Model steering accuracy on the WaterBird dataset. The Worst Group refers to “Waterbird on Land”. 

<table><tr><td>Method</td><td>Steer Spuri. Corr.</td><td>Overall Acc. (%)</td><td>Worst Group Acc. (%)</td><td> $\Delta$ </td></tr><tr><td rowspan="2">CBM</td><td>None</td><td>-</td><td>37.3</td><td>-</td></tr><tr><td>Remove</td><td>-</td><td>51.8</td><td>+ 14.5</td></tr><tr><td rowspan="2">SpLiCE</td><td>None</td><td>-</td><td>48.0</td><td>-</td></tr><tr><td>Remove</td><td>-</td><td>60.0</td><td>+ 12.0</td></tr><tr><td rowspan="2">DN-CBM</td><td>None</td><td>-</td><td>57.5</td><td>-</td></tr><tr><td>Remove</td><td>-</td><td>71.3</td><td>+ 13.8</td></tr><tr><td rowspan="2">PCBM</td><td>None</td><td>-</td><td>50.3</td><td>-</td></tr><tr><td>Remove</td><td>-</td><td>74.7</td><td>+ 24.4</td></tr><tr><td rowspan="2">Joseph et al.</td><td>None</td><td>68.8</td><td>22.4</td><td>-</td></tr><tr><td>Remove</td><td>71.9</td><td>24.6</td><td>+ 2.2</td></tr><tr><td rowspan="2">COAR</td><td>None</td><td>88.0</td><td>64.0</td><td>-</td></tr><tr><td>Remove</td><td>91.0</td><td>83.0</td><td>+ 19.0</td></tr><tr><td rowspan="3">Ours</td><td>None</td><td>79.7</td><td>50.3</td><td>-</td></tr><tr><td>Enhance</td><td>74.5</td><td>5.3</td><td>- 45.0</td></tr><tr><td>Remove</td><td>85.2</td><td>98.5</td><td>+ 48.2</td></tr></table>

Table 8. Ablation study on the impact of probing image sets for model steering. 

<table><tr><td>SAE Probing Set</td><td>Steer Spuri. Corr.</td><td>Worst Group Acc. (%)</td></tr><tr><td>None</td><td>None</td><td>50.3</td></tr><tr><td>ImageNet</td><td>Remove</td><td>63.9</td></tr><tr><td>MSCOCO</td><td>Remove</td><td>95.4</td></tr><tr><td>Ours</td><td>Remove</td><td>98.5</td></tr></table>

gonul et al., 2022), Joseph et al. (Joseph et al., 2025), and COAR (Shah et al., 2024) that can steer the model.

Edit spurious concepts. We mitigate spurious correlations by using our SAE to ablate background-related concepts (e.g., “grass”, “land”) in the worst-group samples. This is done by setting their activations to zero, reconstructing new CLS tokens, and re-classifying. As shown in Tab. 7, this intervention boosts worst-group accuracy by 48.2%. Conversely, enhancing these concepts degrades performance by 45.0%, demonstrating precise bidirectional control via our concept-level “knobs”.

Ablation study on different probing image sets. We further conduct ablation studies to evaluate the influence of the probing image set for training SAEs on steering performance. To ensure a fair comparison, we keep the SAE architecture and all training hyperparameters fixed, and train SAEs using ImageNet, MSCOCO, and our curated probing image set, respectively. As shown in Tab. 8, the SAEs trained on our probing image set significantly outperform their counterparts, even though those also achieve improvements over existing methods.

# 4. Related Work

In this section, we will discuss the most related works. A more detailed version is in Appendix A.

Interpretable Machine Learning (IML). Existing IML methods are broadly categorized into two paradigms. Posthoc methods, such as GradCAM (Selvaraju et al., 2017), LIME (Ribeiro et al., 2016), and SHAP (Lundberg & Lee, 2017), typically provide a saliency map of input pixels; Intrinsic methods, such as ProtoPNet (Chen et al., 2019) and explanation-guided learning (Ross et al., 2017), incorporate interpretability directly into the model architecture. However, the former is limited to input-output correlations, and the latter relies on custom architectures that are not easily generalizable across tasks (Rudin, 2019; Wang et al., 2025). Although there are existing attempts using interpretations as guidance to provide additional supervision for the model training (Li et al., 2023; 2024b;a; Ma et al., 2025), the inner workings of the model remain a black box. Differently, our method interprets internal representations post hoc, without requiring architectural changes to the explainee model.

Concept-based interpretability. Concept-based methods address the limited intelligibility of saliency maps by explaining predictions via human-understandable concepts. Techniques include TCAV (Kim et al., 2018) , Network Dissection (Bau et al., 2017) , and Concept Bottleneck Models (Koh et al., 2020). There are also existing attempts to use predefined concept bottlenecks for failure detection (Nguyen et al., 2025). However, these approaches rely on predefined concept sets or annotations, limiting their scalability in open-world settings (Yuksekgonul et al., 2022; Margeloiu et al., 2021). Different from existing works, we use SAEs to “read” concepts directly from model representations without supervised concept labels.

Mechanistic Interpretation (MI). MI methods aim to reverse-engineer the internal mechanisms of deep models (Nanda et al., 2023; Bereska & Gavves, 2024). Bottomup approaches, such as the Circuits framework (Olah et al., 2020; Conmy et al., 2023), dissect neural connectivity but yield low-level graphs that lack human interpretability (Marks et al., 2024; Peng et al., 2026). Top-down methods like SAEs (Olshausen & Field, 1997; Ng et al., 2011) learn disentangled features to address superposition. In language models, SAEs are trained on large corpora and interpreted via LLM summarization (Bills et al., 2023). For vision, however, SAEs face two key challenges: biased datasets with limited concept coverage, and feature interpretation remains subjective (Thasarathan et al., 2025; Pach et al., 2026). Our toolbox bridges these gaps by curating data and auto-interpretation.

# 5. Conclusion

Different from existing works that primarily focus on improving the architecture design of SAE, we improve the interpretability from a data perspective. Motivated by neuroscience, we develop a compact, diagnostic toolbox, ViSAE, that enables SAEs to efficiently interpret the inner workings of ViTs. Specifically, our toolbox (data and algorithm) enables automatic SAE feature interpretation at scale and faithful concept circuit tracing. We show that ViSAE is not only a powerful auditing tool for identifying spurious correlations and failure modes but also enables effective model steering through concept-level interventions.

Limitations. ViSAE inherits several limitations from SAEbased mechanistic interpretation. First, although our probing suite improves concept coverage and automatic interpretation, SAE features may still exhibit feature absorption or feature composition: a single SAE feature can absorb multiple correlated visual concepts, while a high-level semantic concept may be compositionally represented by multiple SAE features across layers. Therefore, the resulting concept labels and circuit edges should be interpreted as useful approximations of model mechanisms, rather than a guaranteed one-to-one mapping between features and human concepts. Second, our automatic concept reading depends on the coverage of the concept vocabulary and the visionlanguage embedding space used for matching, which may miss rare, abstract, or domain-specific semantics. Third, our empirical evaluation focuses primarily on CLIP-style ViTs and natural images. Extending ViSAE to other architectures, larger vision-language models, and specialized domains remains an important direction for future work.

# Acknowledgement

This work is supported by the National Science Foundation under grant numbers CAREER 2340074, SLES 2416937, and III CORE 2412675, the National Institutes of Health under grant number R21CA301093, and the Department of Defense under grant number AFOSR FA9550-23-1-0494. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the authors and do not reflect the views of the supporting entities.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Achiam, J., Adler, S., Agarwal, S., Ahmad, L., Akkaya, I., Aleman, F. L., Almeida, D., Altenschmidt, J., Altman, S., Anadkat, S., et al. Gpt-4 technical report. arXiv preprint arXiv:2303.08774, 2023.   
Adebayo, J., Gilmer, J., Muelly, M., Goodfellow, I., Hardt, M., and Kim, B. Sanity checks for saliency maps. Advances in neural information processing systems, 31, 2018.   
Arjovsky, M., Bottou, L., Gulrajani, I., and Lopez-Paz, D. Invariant risk minimization. arXiv preprint arXiv:1907.02893, 2019.   
Barack, D. L. and Krakauer, J. W. Two views on the cognitive brain. Nature Reviews Neuroscience, 22(6):359–371, 2021.   
Bau, D., Zhou, B., Khosla, A., Oliva, A., and Torralba, A. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6541–6549, 2017.   
Bereska, L. and Gavves, S. Mechanistic interpretability for ai safety-a review. Transactions on Machine Learning Research, 2024.   
Bhalla, U., Oesterling, A., Srinivas, S., Calmon, F., and Lakkaraju, H. Interpreting clip with sparse linear concept embeddings (splice). Advances in Neural Information Processing Systems, 37:84298–84328, 2024.   
Bills, S., Cammarata, N., Mossing, D., Tillman, H., Gao, L., Goh, G., Sutskever, I., Leike, J., Wu, J., and Saunders, W. Language models can explain neurons in language models. https: //openaipublic.blob.core.windows.net/ neuron-explainer/paper/index.html, 2023.   
Bricken, T., Templeton, A., Batson, J., Chen, B., Jermyn, A., Conerly, T., Turner, N., Anil, C., Denison, C., Askell, A., Lasenby, R., Wu, Y., Kravec, S., Schiefer, N., Maxwell, T., Joseph, N., Hatfield-Dodds, Z., Tamkin, A., Nguyen, K., McLean, B., Burke, J. E., Hume, T., Carter, S., Henighan, T., and Olah, C. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. URL https://transformer-circuits. pub/2023/monosemantic-features.   
Bussmann, B., Leask, P., and Nanda, N. Batchtopk sparse autoencoders. In NeurIPS 2024 Workshop on Scientific Methods for Understanding Deep Learning, 2024.

Bussmann, B., Nabeshima, N., Karvonen, A., and Nanda, N. Learning multi-level features with matryoshka sparse autoencoders. In International Conference on Machine Learning, pp. 6077–6101. PMLR, 2025.

Carandini, M., Demb, J. B., Mante, V., Tolhurst, D. J., Dan, Y., Olshausen, B. A., Gallant, J. L., and Rust, N. C. Do we know what the early visual system does? Journal of Neuroscience, 25(46):10577–10597, 2005.

Chang, A. X., Funkhouser, T., Guibas, L., Hanrahan, P., Huang, Q., Li, Z., Savarese, S., Savva, M., Song, S., Su, H., et al. Shapenet: An information-rich 3d model repository. arXiv preprint arXiv:1512.03012, 2015.

Chefer, H., Gur, S., and Wolf, L. Generic attention-model explainability for interpreting bi-modal and encoderdecoder transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 397– 406, 2021.

Chen, C., Li, O., Tao, D., Barnett, A., Rudin, C., and Su, J. K. This looks like that: deep learning for interpretable image recognition. Advances in neural information processing systems, 32, 2019.

Cimpoi, M., Maji, S., Kokkinos, I., Mohamed, S., , and Vedaldi, A. Describing textures in the wild. In Proceedings of the IEEE Conf. on Computer Vision and Pattern Recognition (CVPR), 2014.

Conmy, A., Mavor-Parker, A., Lynch, A., Heimersheim, S., and Garriga-Alonso, A. Towards automated circuit discovery for mechanistic interpretability. Advances in Neural Information Processing Systems, 36:16318–16352, 2023.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

DiCarlo, J. J., Zoccolan, D., and Rust, N. C. How does the brain solve visual object recognition? Neuron, 73(3): 415–434, 2012.

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al. An image is worth 16x16 words: Transformers for image recognition at scale. In International Conference on Learning Representations, 2020.

Gao, L., Biderman, S., Black, S., Golding, L., Hoppe, T., Foster, C., Phang, J., He, H., Thite, A., Nabeshima, N., Presser, S., and Leahy, C. The Pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.

Ghorbani, A. and Zou, J. Y. Neuron shapley: Discovering the responsible neurons. Advances in neural information processing systems, 33:5922–5932, 2020.   
Goodale, M. A. and Milner, A. D. Separate visual pathways for perception and action. Trends in neurosciences, 15(1): 20–25, 1992.   
Hedstrom, A., Weber, L., Krakowczyk, D., Bareeva, D., ¨ Motzkus, F., Samek, W., Lapuschkin, S., and Hohne, ¨ M. M.-C. Quantus: An explainable ai toolkit for responsible evaluation of neural network explanations and beyond. Journal of Machine Learning Research, 24(34): 1–11, 2023.   
Huben, R., Cunningham, H., Smith, L. R., Ewart, A., and Sharkey, L. Sparse autoencoders find highly interpretable features in language models. In The Twelfth International Conference on Learning Representations, 2023.   
Joseph, S., Suresh, P., Goldfarb, E., Hufe, L., Gandelsman, Y., Graham, R., Bzdok, D., Samek, W., and Richards, B. A. Steering clip’s vision transformer with sparse autoencoders. In Mechanistic Interpretability for Vision at CVPR 2025 (Non-proceedings Track), 2025.   
Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., et al. Interpretability beyond feature attribution: Quantitative testing with concept activation vectors (tcav). In International conference on machine learning, pp. 2668–2677. PMLR, 2018.   
Kindermans, P.-J., Hooker, S., Adebayo, J., Alber, M., Schutt, K. T., D ¨ ahne, S., Erhan, D., and Kim, B. The ¨ (un) reliability of saliency methods. Explainable AI: Interpreting, explaining and visualizing deep learning, pp. 267–280, 2019.   
Koh, P. W., Nguyen, T., Tang, Y. S., Mussmann, S., Pierson, E., Kim, B., and Liang, P. Concept bottleneck models. In International conference on machine learning, pp. 5338– 5348. PMLR, 2020.   
Krishna, R., Zhu, Y., Groth, O., Johnson, J., Hata, K., Kravitz, J., Chen, S., Kalantidis, Y., Li, L.-J., Shamma, D. A., et al. Visual genome: Connecting language and vision using crowdsourced dense image annotations. International journal of computer vision, 123:32–73, 2017.   
Li, T., Qiao, F., Ma, M., and Peng, X. Are data-driven explanations robust against out-of-distribution data? In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3821–3831, 2023.   
Li, T., Ma, M., and Peng, X. Beyond accuracy: ensuring correct predictions with correct rationales. Advances in Neural Information Processing Systems, 37:43164– 43188, 2024a.

Li, T., Ma, M., and Peng, X. Deal: Disentangle and localize concept-level explanations for vlms. In European Conference on Computer Vision, pp. 383–401. Springer, 2024b.   
Lieberum, T., Rajamanoharan, S., Conmy, A., Smith, L., Sonnerat, N., Varma, V., Kramar, J., Dragan, A., Shah, R., ´ and Nanda, N. Gemma scope: Open sparse autoencoders everywhere all at once on gemma 2. In Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP, pp. 278–300, 2024.   
Lim, H., Choi, J., Choo, J., and Schneider, S. Sparse autoencoders reveal selective remapping of visual concepts during adaptation. In International Conference on Learning Representations, volume 2025, pp. 24444–24469, 2025.   
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft coco: ´ Common objects in context. In Computer vision–ECCV 2014: 13th European conference, zurich, Switzerland, September 6-12, 2014, proceedings, part v 13, pp. 740– 755. Springer, 2014.   
Liu, S., Zeng, Z., Ren, T., Li, F., Zhang, H., Yang, J., Jiang, Q., Li, C., Yang, J., Su, H., et al. Grounding dino: Marrying dino with grounded pre-training for open-set object detection. In European conference on computer vision, pp. 38–55. Springer, 2024.   
Lundberg, S. M. and Lee, S.-I. A unified approach to interpreting model predictions. Advances in neural information processing systems, 30, 2017.   
Ma, M., Li, T., Peng, Y., Lin, L., Beylergil, V., Zhao, B., Akin, O., and Peng, X. “why is there a tumor?”: Tell me the reason, show me the evidence. Proceedings of machine learning research, 267:41992, 2025.   
Margeloiu, A., Ashman, M., Bhatt, U., Chen, Y., Jamnik, M., and Weller, A. Do concept bottleneck models learn as intended? arXiv preprint arXiv:2105.04289, 2021.   
Marks, S., Rager, C., Michaud, E. J., Belinkov, Y., Bau, D., and Mueller, A. Sparse feature circuits: Discovering and editing interpretable causal graphs in language models. arXiv preprint arXiv:2403.19647, 2024.   
Meng, K., Bau, D., Andonian, A., and Belinkov, Y. Locating and editing factual associations in gpt. Advances in neural information processing systems, 35:17359–17372, 2022.   
Nanda, N., Chan, L., Lieberum, T., Smith, J., and Steinhardt, J. Progress measures for grokking via mechanistic interpretability. In The Eleventh International Conference on Learning Representations, 2023.

Ng, A. et al. Sparse autoencoder. CS294A Lecture notes, 72 (2011):1–19, 2011.   
Nguyen, K. X., Li, T., and Peng, X. Interpretable failure detection with human-level concepts. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 26326–26334, 2025.   
Oikarinen, T. and Weng, T.-W. Clip-dissect: Automatic description of neuron representations in deep vision networks. In ICLR 2022 Workshop on PAIR {\textasciicircum} 2Struct: Privacy, Accountability, Interpretability, Robustness, Reasoning on Structured Data, 2022.   
Olah, C., Cammarata, N., Schubert, L., Goh, G., Petrov, M., and Carter, S. Zoom in: An introduction to circuits. Distill, 2020. doi: 10.23915/distill.00024.001. https://distill.pub/2020/circuits/zoom-in.   
Olshausen, B. A. and Field, D. J. Sparse coding with an overcomplete basis set: A strategy employed by v1? Vision research, 37(23):3311–3325, 1997.   
OpenAI. Gpt-5 system card. https://cdn.openai. com/gpt-5-system-card.pdf, August 2025.   
Pach, M., Karthik, S., Bouniot, Q., Belongie, S., and Akata, Z. Sparse autoencoders learn monosemantic features in vision-language models. Advances in Neural Information Processing Systems, 38:95706–95742, 2026.   
Pearl, J. Direct and indirect effects. In Probabilistic and causal inference: the works of Judea Pearl, pp. 373–392. 2001.   
Peng, Y., Ma, M., Yao, Z., and Peng, X. Inside-out: Measuring generalization in vision transformers through inner workings. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2026.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PmLR, 2021.   
Rajamanoharan, S., Conmy, A., Smith, L., Lieberum, T., Varma, V., Kramar, J., Shah, R., and Nanda, N. Improving ´ dictionary learning with gated sparse autoencoders. arXiv preprint arXiv:2404.16014, 2024a.   
Rajamanoharan, S., Lieberum, T., Sonnerat, N., Conmy, A., Varma, V., Kramar, J., and Nanda, N. Jumping ahead: ´ Improving reconstruction fidelity with jumprelu sparse autoencoders. arXiv preprint arXiv:2407.14435, 2024b.

Rao, S., Mahajan, S., Bohle, M., and Schiele, B. Discover- ¨ then-name: Task-agnostic concept bottlenecks via automated concept discovery. In European Conference on Computer Vision, pp. 444–461. Springer, 2024.   
Ribeiro, M. T., Singh, S., and Guestrin, C. ” why should i trust you?” explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining, pp. 1135–1144, 2016.   
Ross, A. S., Hughes, M. C., and Doshi-Velez, F. Right for the right reasons: Training differentiable models by constraining their explanations. In Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, pp. 2662–2670. International Joint Conferences on Artificial Intelligence Organization, 2017.   
Rudin, C. Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead. Nature machine intelligence, 1(5):206– 215, 2019.   
Sagawa, S., Koh, P. W., Hashimoto, T. B., and Liang, P. Distributionally robust neural networks. In International Conference on Learning Representations, 2019.   
Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., and Batra, D. Grad-cam: Visual explanations from deep networks via gradient-based localization. In Proceedings of the IEEE international conference on computer vision, pp. 618–626, 2017.   
Shah, H., Ilyas, A., and Madry, A. Decomposing and editing predictions by modeling model computation. In Proceedings of the 41st International Conference on Machine Learning, pp. 44244–44292, 2024.   
Stevens, S., Chao, W.-L., Berger-Wolf, T., and Su, Y. Sparse autoencoders for scientifically rigorous interpretation of vision models. arXiv preprint arXiv:2502.06755, 2025.   
Team, G., Mesnard, T., Hardin, C., Dadashi, R., Bhupatiraju, S., Pathak, S., Sifre, L., Riviere, M., Kale, M. S., Love, \` J., et al. Gemma: Open models based on gemini research and technology. arXiv preprint arXiv:2403.08295, 2024.   
Thasarathan, H., Forsyth, J., Fel, T., Kowal, M., and Derpanis, K. G. Universal sparse autoencoders: Interpretable cross-model concept alignment. In Forty-second International Conference on Machine Learning, 2025.   
Wang, Q., Li, T., Nguyen, K. X., and Peng, X. Beyond accuracy: On the effects of fine-tuning towards visionlanguage model’s prediction rationality. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 39, pp. 21225–21233, 2025.

Yang, Y., Panagopoulou, A., Zhou, S., Jin, D., Callison-Burch, C., and Yatskar, M. Language in a bottle: Language model guided concept bottlenecks for interpretable image classification. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 19187–19197, 2023.   
Yuksekgonul, M., Wang, M., and Zou, J. Post-hoc concept bottleneck models. In The Eleventh International Conference on Learning Representations, 2022.   
Zaigrajew, V., Baniecki, H., and Biecek, P. Interpreting clip with hierarchical sparse autoencoders. In International Conference on Machine Learning, pp. 73918– 73956. PMLR, 2025.   
Zhang, K., Shen, Y., Li, B., and Liu, Z. Large multi-modal models can interpret features in large multi-modal models. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 3650–3661, 2025.   
Zhou, B., Lapedriza, A., Khosla, A., Oliva, A., and Torralba, A. Places: A 10 million image database for scene recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.   
Zou, A., Phan, L., Chen, S., Campbell, J., Guo, P., Ren, R., Pan, A., Yin, X., Mazeika, M., Dombrowski, A.-K., et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint arXiv:2310.01405, 2023.

# Appendix

# A. Related Work

Interpretable Machine Learning (IML). IML methods aim to uncover the reasons behind model predictions and are typically categorized as: Post-hoc methods, such as GradCAM (Selvaraju et al., 2017), LIME (Ribeiro et al., 2016), and SHAP (Lundberg & Lee, 2017), provide explanations by attributing predictions to input features; Intrinsic methods, such as ProtoPNet (Chen et al., 2019) and explanation-guided learning (Ross et al., 2017), incorporate interpretability directly into the model architecture by design. However, the former is often limited to surface-level input-output relationships, while the latter depends on custom architectures that are not easily generalizable across tasks (Adebayo et al., 2018; Rudin, 2019). In contrast, our method decouples interpretation from prediction and instead analyzes internal representations post hoc, without requiring architectural changes to the explainee model.

Concept-based interpretability. Concept-based methods emerged as a response to the limitations of attribution methods, where saliency maps often fail to provide human-interpretable explanations (Adebayo et al., 2018; Kindermans et al., 2019). TCAV (Kim et al., 2018) uses curated probing sets to evaluate a model’s sensitivity to predefined concept directions. Network Dissection (Bau et al., 2017) assigns semantics to individual neurons using human-annotated labels. ACE automatically discovers salient concept clusters in latent space. Concept Bottleneck Models (CBMs) (Koh et al., 2020) enforce a humandefined concept layer within the network, enabling transparency and intervention. However, these methods typically require concept annotations or assume a closed-world setting with a fixed concept vocabulary, making them struggle to scale up to open-world concept discovery that does not assume the set of concepts is a known prior (Yuksekgonul et al., 2022; Margeloiu et al., 2021). Our method differs by directly extracting concepts from pretrained models without requiring concept supervision or architecture changes.

Mechanistic Interpretation (MI). MI methods (Nanda et al., 2023; Bereska & Gavves, 2024) aim to uncover the internal computational mechanisms of deep models and have shown promising progress, particularly in language models. Bottom-up approaches, such as the Circuits framework (Olah et al., 2020; Conmy et al., 2023), dissect individual neurons and their connectivity to reveal functional subcomputations. However, the resulting units (e.g., neurons) are often not interpretable to humans (Marks et al., 2024). Top-down approaches, including representation engineering (Zou et al., 2023) and Sparse Autoencoders (SAEs) (Olshausen & Field, 1997; Ng et al., 2011), address this by learning disentangled, monosemantic features that map more naturally to human-understandable concepts, mitigating the feature superposition issue. While SAEs decompose polysemantic representations into monosemantic features, ensuring comprehensive coverage and interpreting their semantic meanings remains challenging. For language models, existing approaches (Huben et al., 2023; Lieberum et al., 2024) typically train SAEs on massive text corpora, such as the Pile (∼7M) (Gao et al., 2020) or Gemma (∼3T) (Team et al., 2024), to broaden concept coverage, and interpret features by prompting LLMs (e.g., GPT-4 (Achiam et al., 2023)) to summarize the semantics of top-activating examples (Bills et al., 2023). For vision models, however, available datasets are usually biased toward object-level concepts (e.g., ImageNet (Deng et al., 2009)), might not cover the full spectrum of visual processing. Furthermore, the top-activating images of SAE features often show ambiguous semantics, making their interpretation subjective (Thasarathan et al., 2025; Pach et al., 2026).

# B. Concept Coverage Calculation

To measure concept coverage in a dataset-agnostic manner, we leverage a Top Percentile Method that evaluates how well each concept is represented by the most similar images in the dataset. For each concept $c _ { i } .$ , we compute the cosine similarity between the concept’s text embedding and all image embeddings (CLIP-ViT-B-32) in the dataset, yielding a similarity vector $\mathbf { s } _ { i } \in \mathbb { R } ^ { N }$ where N is the dataset size. Rather than relying on maximum similarity (which can be noisy) or overall mean similarity (which may be dominated by irrelevant images), we calculate the mean similarity of the top k most similar images, where k = max $( 1 , [ N \cdot p / 1 0 0 ] )$ and p is a small percentile (typically 0.005%), namely:

$$
\text { Coverage   Score } \left(c _ {i}\right) = \frac {1}{k} \sum_ {j = 1} ^ {k} s _ {i, \text { top } - j}, \tag {7}
$$

where $s _ { i , \mathrm { t o p } - j }$ represents the $j \cdot$ th highest similarity score between concept $c _ { i }$ and the images in the dataset. A concept is considered “covered” at threshold τ if Coverage Score $( c _ { i } ) \geq \tau$ . In practice, we set $\tau = 0 . 2 5$ as a meaningful similarity between modalities.

This approach is robust to dataset size variations and provides a stable measure of concept representation quality by focusing on the images that most strongly exhibit each concept, while avoiding the influence of outliers or the vast majority of irrelevant images.

# C. Concept Count Calculation

To understand the semantic distribution of existing concept sets across different abstraction levels, we perform a match analysis that assigns each of their concepts to its best-matching abstraction level in our ground truth taxonomy. Given an existing concept set $\mathcal { C } = \{ c _ { 1 } , c _ { 2 } , . . . , c _ { N } \}$ and ground truth concepts organized by abstraction levels $\mathcal { G } _ { \mathrm { g t } } = \{ \mathcal { G } _ { \mathrm { p r i m i t i v e } } , \mathcal { G } _ { \mathrm { i n t e r m e d i a t e } } , \mathcal { G } _ { \mathrm { o b j e c t } } , \mathcal { G } _ { \mathrm { s c e n e } } \}$ , we first compute the semantic similarity between each new concept and all ground truth concepts sim $( c _ { i } , g _ { i } )$ using CLIP text embeddings (CLIP-ViT-B-32). We then identify the best-matching ground truth concept for each new concept:

$$
g _ {i} ^ {*} = \underset {g _ {j} \in \bigcup_ {l} g _ {l}} {\arg \max} \text { sim } (c _ {i}, g _ {j}) \tag {8}
$$

A concept $c _ { i }$ is considered well-matched if its maximum similarity exceeds a threshold τ (0.9 in practice). Each well-matched concept is then assigned to the abstraction level of its best-matching ground truth concept.

This analysis reveals the semantic composition of existing concept sets, showing how many concepts align with each abstraction level (primitive, intermediate, object, scene) and identifying concepts that may not represent visual concepts.

# D. Details on SAE Architectures

In this section, we provide details for the SAE variants used in our study. These variants differ primarily in how they induce sparsity in the hidden representation h.

ReLU SAE (Bricken et al., 2023). This is the standard sparse autoencoder using ReLU nonlinearity followed by an $L _ { 1 }$ sparsity penalty. Given the input representation $\mathbf { x } \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ , encoder, decoder weights $\mathbf { W } _ { \mathrm { e n c } } , \mathbf { W } _ { \mathrm { d e c } } ^ { \top } \in \dot { \mathbb { R } } ^ { d _ { \mathrm { h i d } } \times d _ { \mathrm { i n } } }$ and biases $\mathbf { b } _ { \mathrm { e n c } } , \mathbf { b } _ { \mathrm { d e c } } \in \mathbb { R } ^ { d _ { \mathrm { h i d } } }$ , the hidden representation h $\in \mathbb { R } ^ { d _ { \mathrm { h i d } } }$ is given by:

$$
\mathbf {h} = \operatorname{ReLU} (\mathbf {W} _ {\text { enc }} \mathbf {x} + \mathbf {b} _ {\text { enc }}), \tag {9}
$$

and the reconstruction is:

$$
\hat {\mathbf {x}} = \mathbf {W} _ {\text { dec }} \mathbf {h} + \mathbf {b} _ {\text { dec }}. \tag {10}
$$

The ReLU SAE is trained to minimize a loss function:

$$
\mathcal {L} = \mathcal {L} _ {\text { reconstruction }} + \mathcal {L} _ {\text { sparsity }} = \| \mathbf {x} - \hat {\mathbf {x}} \| _ {2} ^ {2} + \lambda \| \mathbf {h} \| _ {1}, \tag {11}
$$

where λ is the regularization coefficient to control sparsity. This variant is simple and effective, but sparsity is indirectly controlled by λ.

BatchTopK SAE (Bussmann et al., 2024). Instead of applying a soft sparsity penalty, this variant enforces hard sparsity by retaining only the top-k activations across a batch of n samples. Specifically, it retains the $n \times k$ largest activations across the batch and zeros out all others:

$$
\mathbf {h} = \text { BatchTopK } (\mathbf {W} _ {\text { enc }} \mathbf {x} + \mathbf {b} _ {\text { enc }}). \tag {12}
$$

The loss becomes:

$$
\mathcal {L} = \left\| \mathbf {x} - \mathbf {W} _ {\mathrm{dec}} \mathbf {h} + \mathbf {b} _ {\mathrm{dec}} \right\| _ {2} ^ {2}. \tag {13}
$$

This provides deterministic sparsity and is particularly suitable for interpretability-focused applications.

Matryoshka SAE (Bussmann et al., 2025). Matryoshka SAE is inspired by the idea of nested sparsity levels. It produces multiple nested representations $\{ \mathbf { h } ^ { ( 1 ) } , \ldots , \mathbf { h } ^ { ( K ) } \}$ such that each $\mathbf { h } ^ { ( \bar { k } ) }$ satisfies $\mathbf { h } ^ { ( 1 ) } \subseteq \dots \subseteq \bar { \mathbf { h } } ^ { ( K ) }$ , where $\mathbf { h } ^ { ( K ) } = \mathrm { B a t c h T o p K } ( \mathbf { W } _ { \mathrm { e n c } } \mathbf { x } + \mathbf { b } _ { \mathrm { e n c } } )$ . The reconstruction loss is computed over all levels:

$$
\mathcal {L} = \sum_ {k = 1} ^ {K} \| \mathbf {x} - \mathbf {W} _ {\mathrm{dec}} \mathbf {h} ^ {(k)} + \mathbf {b} _ {\mathrm{dec}} \| _ {2} ^ {2}. \tag {14}
$$

This encourages a hierarchical structure in the learned features and facilitates interpretability at multiple granularity levels.

JumpReLU SAE (Rajamanoharan et al., 2024b). JumpReLU replaces the standard ReLU with a modified activation function that enforces a minimum activation threshold:

$$
\operatorname{JumpReLU} (z) = \left\{ \begin{array}{l l} z, & z > \tau \\ 0, & \text { otherwise } \end{array} , \right. \tag {15}
$$

where τ is a fixed threshold $( \mathrm { e } . \mathrm { g } . , \tau = 0 . 0 0 1 )$ . This nonlinearity encourages fewer active units by cutting off low activations more aggressively than ReLU, resulting in sparser codes even without explicit sparsity penalties.

Gated SAE (Rajamanoharan et al., 2024a). Gated SAE uses multiplicative gating to modulate activations. Each hidden unit has a learned gate $g _ { i } \in [ 0 , 1 ]$ , typically computed via a sigmoid:

$$
\mathbf {h} _ {i} = \sigma (\mathbf {a} _ {i} ^ {\top} \mathbf {x}) \cdot \mathrm{ReLU} (\mathbf {w} _ {i} ^ {\top} \mathbf {x} + b _ {i}). \tag {16}
$$

This allows the model to selectively suppress irrelevant features and provides an adaptive mechanism to control sparsity, potentially improving both interpretability and flexibility.

# E. Evaluation Metrics

Reconstruction Error. This metric quantifies how well the autoencoder reconstructs the input data from the sparse code. Formally, given input $\mathbf { x } \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ and reconstructed output $\hat { \mathbf { x } } = \mathbf { W } _ { \mathrm { d e c } } \mathbf { h } + \mathbf { b } _ { \mathrm { d e c } }$ , the Reconstruction Error is defined as:

$$
\text { Reconstruction   Error } = \mathbb {E} _ {\mathbf {x} \sim \mathcal {D}} \left[ \| \mathbf {x} - \hat {\mathbf {x}} \| _ {2} ^ {2} \right]. \tag {17}
$$

Lower Reconstruction Error indicates that the SAE preserves more input information. However, extremely low reconstruction error may come at the cost of losing sparsity or interpretability.

Monosemanticity (Pach et al., 2026). Monosemanticity is a measure of how consistently a neuron responds to semantically similar inputs. Intuitively, a neuron is considered monosemantic if its highest activations occur for a group of inputs that are semantically coherent (e.g., images depicting the same object or concept). Specifically, it measures the visual similarity between the top-k activated inputs for each neuron.

Formally, let $f ( \mathbf { x } ) = \mathbf { h } \in \mathbb { R } ^ { d _ { \mathrm { h i d } } }$ be the encoder output for input x, and let $h _ { i }$ denote the activation of the i-th neuron. For each neuron $i \in \{ 1 , \ldots , d _ { \mathrm { h i d } } \}$ , we identify the top-k inputs from the dataset D that elicit the highest activations:

$$
\mathcal {T} _ {i} = \operatorname{TopK} \left(\left\{\left(\mathbf {x}, h _ {i} (\mathbf {x})\right) \mid \mathbf {x} \in \mathcal {D} \right\}\right). \tag {18}
$$

We then compute the average pairwise cosine similarity between the embeddings of the top-k input images. Let $\phi ( \mathbf { x } ) \in \mathbb { R } ^ { d }$ be a feature embedding of image x obtained from the CLIP ViT-B/32 model. The monosemanticity score for neuron i is:

$$
\operatorname{Mono} (i) = \frac {2}{k (k - 1)} \sum_ {1 \leq p <   q \leq k} \frac {\phi (\mathbf {x} _ {p}) ^ {\top} \phi (\mathbf {x} _ {q})}{\| \phi (\mathbf {x} _ {p}) \| _ {2} \cdot \| \phi (\mathbf {x} _ {q}) \| _ {2}}, \quad \text { where } \{\mathbf {x} _ {1}, \dots , \mathbf {x} _ {k} \} = \mathcal {T} _ {i}. \tag {19}
$$

Finally, the overall monosemanticity score for the autoencoder is obtained by averaging over all neurons:

$$
\text { Monosemanticity } = \frac {1}{d _ {\mathrm{hid}}} \sum_ {i = 1} ^ {d _ {\mathrm{hid}}} \operatorname{Mono} (i). \tag {20}
$$

Higher values indicate that neurons respond selectively to visually similar inputs, which supports more interpretable and disentangled representations.

Decoder Orthogonality (Zaigrajew et al., 2025). To enhance interpretability, it is desirable that the learned basis features (i.e., decoder columns) are disentangled. One way to promote this is to encourage orthogonality among decoder vectors. Let $\mathbf { W } _ { \mathrm { d e c } } = [ \mathbf { w } _ { 1 } , \hdots , \mathbf { w } _ { d _ { \mathrm { h i d } } } ] \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times d _ { \mathrm { h i d } } }$ denote the decoder weight matrix. The Decoder Orthogonality is defined as the mean pair-wise cosine similarity between each pair of decoder columns:

$$
\text { Decoder   Orthogonality } = \frac {2}{d _ {\mathrm{hid}} (d _ {\mathrm{hid}} - 1)} \sum_ {1 \leq i <   j \leq d _ {\mathrm{hid}}} \mathbf {w} _ {i} ^ {\top} \mathbf {w} _ {j}. \tag {21}
$$

A smaller value implies greater orthogonality and lower redundancy among the learned features. Perfect orthogonality occurs when all decoder vectors are mutually orthogonal unit vectors.

Dead Neuron. This metric measures the fraction of hidden units that are never activated across a dataset. A hidden neuron is considered ”dead“ if its activation is zero for all inputs in a dataset D. Let $\mathbf { h } ( \mathbf { x } )$ be the hidden representation on input x and $h _ { i } ( \mathbf { x } )$ the i-th dimension. Define the dead neuron set:

$$
\mathcal {D} _ {\text { dead }} = \left\{i \in \{1, \dots , d _ {\text { hid }} \}   \middle |   \sum_ {\mathbf {x} \sim \mathcal {D}} h _ {i} (\mathbf {x}) = 0 \right\}. \tag {22}
$$

The Dead Neuron ratio is:

$$
\text { Dead   Neuron } = \frac {\left| \mathcal {D} _ {\text { dead }} \right|}{d _ {\text { hid }}}. \tag {23}
$$

A high dead neuron ratio indicates underutilization of the model capacity, which may suggest over-regularization or poor feature allocation. On the other hand, a moderate level of dead neurons may naturally emerge in highly sparse encoders.

# F. Implementation Details

Training Details. We train all Sparse Autoencoders (SAEs) on our probing image set using the cls tokens and image tokens from the residual stream of each layer in the CLIP ViT-B/32 model (each layer has two SAEs). Each SAE consists of an overcomplete linear encoder and a sparse decoder, with the decoder columns constrained to unit $\ell _ { 2 }$ norm. For benchmark experiment, we vary the expansion factor $e f \in \{ 2 , 4 , 8 , 1 6 , 3 2 \}$ , defined as $d _ { \mathrm { h i d } } = e f \cdot d _ { \mathrm { i n } } .$ , and $L _ { 0 }$ sparsity $L _ { 0 } \in \{ 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ for all SAE architectures.

Optimization and Scheduler. We train all models using a modified Adam optimizer that enforces unit-norm constraints on decoder columns. We use a fixed batch size of 4096, learning rate $\eta = 3 \times 1 0 ^ { - 4 }$ , and train for 100 epochs. No learning rate decay or warmup is applied.

Hyperparameter Choice. We control sparsity in BatchTopK and Matryoshka SAEs by directly setting the top-k values $k \in { 8 , 1 6 , 3 2 , 6 4 , 1 2 8 }$ . For ReLU, JumpReLU, and Gated SAEs, we perform a sweep over sparsity regularization strength λ to approximate the target $L _ { 0 }$ sparsity levels. In Matryoshka SAEs, we use nested hidden representations with cumulative fractions $\textstyle { \frac { 1 } { 3 2 } } , { \frac { 1 } { 1 6 } } , { \frac { 1 } { 8 } } , { \frac { 1 } { 4 } } , { \frac { 1 } { 2 } }$ 1 , 1. For JumpReLU SAEs, we fix the jump threshold $\tau = 0 . 0 0 1$ throughout all experiments. For the Monosemanticity metric, we compute pair-wise similarity between $k = 9$ top activated images for each feature basis, for the Interpretation Accuracy metric, we retrieve k = 3 concepts from the concept set for each feature basis.

# G. Additional Experimental Results

We provide full benchmark results in Figs. 8& 9.

![](images/454dd51a1ec44c63540ddac9761f4db904d88b114d3dfbf389577a39526eaf6d.jpg)  
ReLU Gated JumpReLU + BatchTopK Matryoshka

Figure 8. Full Benchmark results for cls tokens.

![](images/c9dc32859a9c5377696f4d14abe673a8d993a7f145010547838b3e355af72266.jpg)  
ReLU Gated JumpReLU + BatchTopK Matryoshka

Figure 9. Full Benchmark results for image tokens.