# Neuron Populations Exhibit Divergent Selectivity with Scale

Amil Dravid1 Yasaman Bahri1 Alexei A. Efros1 Yossi Gandelsman2

1UC Berkeley 2TTIC

# Abstract

We investigate whether neuron populations within neural networks evolve predictably with scale, extending scaling laws beyond macroscopic observables such as loss. To probe this question, we study Rosetta Neurons, a previously characterized class of neurons whose activation patterns are similar across independently trained models (Dravid et al., 2023). In separate analyses of language models up to 30B parameters and vision models up to 5B parameters, we observe that the population of Rosetta Neurons follows a sublinear power law in model size, growing in absolute number but occupying a shrinking fraction of the total neuron count. We further observe a Neuron Polarization Effect: Rosetta Neurons become more selective and increasingly monosemantic with scale, separating from a growing non-Rosetta population that remains less selective. An analytical model balancing feature utility against limited neuron capacity explains the sublinear power-law scaling and this polarization effect. Finally, we find that Rosetta Neurons become more domain-specialized with scale and illustrate their selectivity through a targeted data-filtering case study for continued pretraining. Our results point to a scaling law for interpretable, shared neuron-level structure, linking model size to systematic changes in neuron universality, selectivity, and specialization.1

# 1 Introduction

A central question in both deep learning and neuroscience is how neurons encode structure in the world. In biological systems, this question has motivated longstanding debates about whether representations are localized in single units (Hubel & Wiesel, 1962) or distributed across populations (Haxby et al., 2001), as well as studies of how such structure might recur across subjects (Hasson et al., 2004). An analogous question arises for artificial neural networks. Despite a growing body of work probing neurons, such as those encoding sentiment in language models (Radford et al., 2017) or object segments in vision models (Bau et al., 2017), most neurons in today’s large-scale models are not easily interpretable (Bills et al., 2023).

One reason for this difficulty is that neuron-level representations may vary in how cleanly they isolate individual features. Some neurons appear relatively monosemantic, responding selectively to coherent semantic concepts, while superposition predicts that others may be polysemantic, encoding multiple unrelated features when models represent more features than available dimensions (Elhage et al., 2022). This makes monosemanticity and polysemanticity natural questions at the neural-population level: as models scale, which neurons become selective, which remain in superposition, and how do these properties relate to shared internal function across independently trained models?

These questions become especially salient in the scaling regime, where models change not only in performance but also in representational capacity. Neural scaling laws (Kaplan et al., 2020; Hoffmann et al., 2022; Hestness et al., 2017) have revealed striking regularities in external model behavior, with quantities such as loss following precise power-law relationships with compute, data, and model size. Yet these laws say little about how representations are organized inside the model. We therefore apply a similar scaling analysis at the neuron level, asking whether cross-model recurring neuron populations evolve predictably with scale: how their size changes, whether they become more selective and monosemantic, and how their features specialize with respect to the data distribution.

![](images/97a3748ee0e09b520bcc7d6190f4b8beb42a2282b874dcfaa653968d838a7d10.jpg)

<details>
<summary>bar</summary>

| Data Patterns | Frequency |
| ------------- | --------- |
| common features | 100 |
| rare features | 5 |
</details>

B   
![](images/b06a078a8f8cb6b26a09ce89084eaa47ae6b3826237a455eb12f1fac8e27450e.jpg)

<details>
<summary>text_image</summary>

Universality
Pythia-2.8B
two years and eight months after trial
published a mere three days after
Qwen2.5-3B
two years and eight months after trial
published a mere three days after
Shared activation pattern
</details>

![](images/a7519f80adba5e2b995c867f5e04e1f9d70915002cd52fc1deffff2575f3360b.jpg)

<details>
<summary>text_image</summary>

Selectivity
Monosemantic
scientist who challenged Aristotle
Plato did what he knew would be done
Polysemantic
stir until the yeast is dissolved
inside the west end of the building
One concept vs. multiple
</details>

D   
![](images/7b447ac8e7ef30cfc86e7f5607db56ab870e356feb07ae3d72e93ec13b92db3b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Plural pronouns"] --> B["Plural pronouns: They treat us with more care and awe than we do towards"]
    C["Historical periods"] --> D["Gold Rush that built California in the 19th century"]
    E["Organic chemistry"] --> F["polyphenylene ethers dissolve readily in methylene chloride"]
    B --> G["Specialized concepts"]
    D --> G
    F --> G
```
</details>

Figure 1: Neuron populations across scale. To study how neuron populations scale, we use Rosetta Neurons: units that recur across different models. (A) Features compete for representation in a finite set of neurons, leaving them isolated, mixed, or unrepresented at a given scale. This picture guides our analysis of universality, selectivity, and specialization. In panels B–D, each column shows top-activating contexts from a single neuron. (B) Universality: how does the recurring Rosetta Neuron population scale? (C) Selectivity: do recurring neurons become increasingly monosemantic relative to polysemantic neurons? (D) Specialization: which features of the data distribution are encoded by Rosetta Neurons at different scales? Across language and vision models, Rosetta Neurons grow in number but shrink as a fraction of all neurons, while becoming more selective and specialized.

To make these questions measurable, we adopt the notion of Rosetta Neurons from Dravid et al. (2023): neurons whose activation patterns recur across independently trained models. We use this recurring population to study three neuron-level properties across scale: universality, selectivity, and specialization (Figure 1). Prior work suggests that cross-model recurrence can reflect stable features of the data distribution rather than idiosyncrasies of any one model, but leaves open whether such recurring neurons follow systematic laws across scale and model families in both language and vision.

In this paper, we treat Rosetta Neurons as a scaling observable, identifying recurring cross-model units in language and vision models spanning 80M to 30B parameters (Section 3). In Section 4, we apply a scaling-law analysis to this population, showing that its size grows predictably with model size according to a sublinear power law in neuron count: larger models contain more shared units, yet these units occupy a shrinking fraction of the total neuron population.

We then develop a phenomenological model linking this sublinear scaling to limited neuron capacity (Section 4.3). As networks scale, more features become monosemantically represented in individual neurons and thus more likely to recur across models. However, this set grows more slowly than the broader pool of new features represented in network-specific patterns of superposition (Figure 1A). This theory predicts a Neuron Polarization Effect: Rosetta Neurons become increasingly selective with scale, separating from the polysemantic non-Rosetta population. We validate this prediction across language and vision models (Section 5) and further show that Rosetta Neurons become increasingly specialized, shifting toward domains such as code and mathematics.

Finally, we demonstrate this specialization functionally in Section 5.3: Rosetta Neuron activations can filter data from a specific code domain with near-oracle accuracy, yielding continued-pretraining performance that matches training on ground-truth domain data. Together, our analysis provides a scalable way to identify shared, interpretable, and predictable structure within large models, revealing a neuron-level population whose size, selectivity, and specialization evolve systematically with scale.

# 2 Related Work

Neural scaling laws. A large body of work has shown that neural network performance follows predictable scaling behavior with respect to model scale, dataset size, and training compute, often well described by power laws of the form ${ \mathcal { L } } ( x ) = { \mathcal { L } } _ { \infty } + A x ^ { - \alpha }$ (Hestness et al., 2017; Kaplan et al., 2020). Subsequent work refined these observations by identifying compute-optimal training regimes (Hoffmann et al., 2022) and demonstrating similar scaling behavior across modalities and architectures (Zhai et al., 2022). A number of works have sought to explain these laws from various perspectives, including (but not limited to) the alignment of tasks to models, the structure of the data distribution, the intrinsic dimensionality of the data manifold, the geometry of learned features, and the learning dynamics of rare tasks under capacity constraints (Bordelon et al., 2020; Michaud et al., 2023; Bahri et al., 2024; Liu et al., 2025; Cagnetta et al., 2026; Huang et al., 2026). We extend this perspective to internal representations, using Rosetta Neurons to study whether a form of shared neuron-level structure also scales predictably.

Structured representations and superposition. Understanding how neural networks develop structured representations has been a central question in deep learning. Interpretability work suggests that learned representations can exhibit approximately linear organization (Mikolov et al., 2013; Arora et al., 2018; Park et al., 2024). Mechanistic studies show that limited-capacity models can represent features in superposition, with toy-model analyses framing this as a capacity-allocation tradeoff between ignoring, isolating, and superposing features (Elhage et al., 2022; Scherlis et al., 2022). Motivated by this structured but superposed organization, sparse decomposition approaches seek to recover interpretable features that are more selective and monosemantic (Bricken et al., 2023b). We build on these ideas at the neuron level, showing that scaling polarizes neurons into a more selective, interpretable Rosetta population against a polysemantic background.

Representational similarity and universality. A complementary line of work studies the extent to which learned representations are shared across models and training runs. Various similarity measures have been proposed to quantify alignment between neural representations (Raghu et al., 2017; Morcos et al., 2018; Kornblith et al., 2019), while alternative approaches such as model stitching probe functional equivalence (Bansal et al., 2021; Lenc & Vedaldi, 2015). These ideas build on earlier notions of representational similarity from neuroscience (Kriegeskorte et al., 2008a; Edelman, 1998), where shared geometric structure is used to compare representations across systems (Kriegeskorte et al., 2008b; Haxby et al., 2001). Together, these perspectives have supported growing evidence for convergent structure in learned representations (Sorscher et al., 2022; Huh et al., 2024; Li et al., 2016). At the neuron level, Dravid et al. (2023) identified shared “Rosetta Neurons” across diverse vision models, while Gurnee et al. (2024) studied shared neurons in small-scale GPT-2 models trained from different random seeds. Beyond these settings, we study Rosetta Neurons within both language and vision models spanning heterogeneous architectures and datasets, and characterize their scaling laws.

# 3 Identifying Rosetta Neurons

We now describe how we identify Rosetta Neurons – common neurons with similar responses across models. We define MLP neurons and the token-wise activations used to compare them, then describe how to measure pairwise similarity and filter these similarities into reliable shared neuron pairs.

# 3.1 MLP Neurons in Transformer Models

We study both language and vision models built on the Transformer architecture (Vaswani et al., 2017). Given an input sequence of text tokens or image patches, the model maps each token to an embedding, producing a sequence $e _ { 1 } , \ldots , e _ { T } \in \mathbb { R } ^ { d }$ . These are then processed through a series of blocks that alternate between multi-head self-attention and multilayer perceptron (MLP) layers. We focus exclusively on the MLP layers. Let $h _ { t } ^ { ( \ell ) }$ denote the hidden state at token position t in layer ℓ. The MLP first applies an affine transformation followed by an element-wise nonlinearity ϕ:

$$
m _ {t} ^ {(\ell)} = \phi \left(W _ {\text { in }} ^ {(\ell)} h _ {t} ^ {(\ell)} + b _ {\text { in }} ^ {(\ell)}\right) \in \mathbb {R} ^ {d _ {\mathrm{mlp}}}, \tag {1}
$$

It then projects this activation back to the residual stream as $\tilde { h } _ { t } ^ { ( \ell ) } = W _ { \mathrm { o u t } } ^ { ( \ell ) } m _ { t } ^ { ( \ell ) } + b _ { \mathrm { o u t } } ^ { ( \ell ) }$ . Each coordinate $m _ { t } ^ { ( \ell , c ) }$ t out t out of the intermediate activation vector is a neuron, indexed by layer ℓ and channel c.

![](images/ae55a16144f42dcd6fe5ebc8a5100aea6130ec52bdf136541ed7ab233a641ced.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Model A
        A1["Win"] -->|Neuron (fα,Cα)| M1
        A2["Wout"] --> M1
        M1 --> M2
    end
    subgraph Model B
        B1["Win"] -->|Neuron (fβ,Cβ)| M2
        B2["Wout"] --> M2
        M2 --> M3
    end
    M1 -->|Mutual nearest neighbors| M2
    M2 --> M3
```
</details>

![](images/071b37a2ebd4a9a976d3043d267a65ec54d5eca80e8efffc21090f1ab88d2d22.jpg)

<details>
<summary>text_image</summary>

Language
Pythia-12B
#1
the convergence of SGD in terms of minimizing
#2
the gradient norm is relatively less well-understood
Qwen2.5-14B
the convergence of SGD in terms of minimizing
the gradient norm is relatively less well-understood
Rosetta Neuron:
gradient-based optimization
</details>

![](images/a781220321c64cadd4d09acc69c143a6a842e7a98e0b4c7b74192c6462ff2f65.jpg)

<details>
<summary>text_image</summary>

Vision
OpenCLIP
ViT-L
#1
#2
DINOv2
ViT-L
Rosetta Neuron:
animal noses
</details>

Figure 2: Identifying Rosetta Neurons. We compare MLP neuron activations across independently trained models on the same inputs and identify mutual nearest-neighbor pairs under Pearson correlation. The language and vision examples show individual matched neuron pairs firing on the same high-activating inputs, revealing similar activation patterns and coherent shared concepts.

# 3.2 Quantifying Pairwise Neuron Similarity

We now describe how to compare MLP neurons between two models. Let model A and model B be two Transformers of the same modality. We run the same dataset of inputs $\mathcal { X } = \{ x _ { i } \} _ { i = 1 } ^ { n }$ through both models, where each $x _ { i }$ is either an image or a sequence of text. The Transformer processes each input jointly to produce token-wise activations, which are then aligned across models. In vision models, activations are aligned to a common spatial grid, while in language models they are aligned to a shared sequence of text positions. Details of this alignment procedure are given in Section B.

Consider a neuron $u = ( \ell _ { A } , c _ { A } )$ in model A and a neuron $\boldsymbol { v } = ( \ell _ { B } , c _ { B } )$ in model B. Let $m _ { t } ^ { u } ( x _ { i } )$ and $m _ { t } ^ { v } ( x _ { i } )$ denote their activations at aligned token t for input $x _ { i }$ . We compare neurons u and v by computing the Pearson correlation between these activations on all aligned tokens across the dataset:

$$
\operatorname{sim} (u, v) = \frac {1}{N} \sum_ {i, t} \frac {m _ {t} ^ {u} (x _ {i}) - \mu_ {u}}{\sigma_ {u}} \cdot \frac {m _ {t} ^ {v} (x _ {i}) - \mu_ {v}}{\sigma_ {v}}, \tag {2}
$$

where $\mu _ { u }$ and $\sigma _ { u } ^ { 2 }$ are the empirical mean and variance for neuron u over the dataset of N total aligned tokens, and $\mu _ { v }$ and $\sigma _ { v } ^ { 2 }$ are defined analogously for neuron v. We compute this similarity for all neuron pairs between models A and B, producing a large table of pairwise similarities.

# 3.3 Filtering Pairwise Neuron Correspondences

To detect reliable matches between neurons, we retain only those pairs that are nearest neighbors of one another under our similarity metric. Specifically, for neuron $\iota \in { \mathcal { N } } ( A )$ and $v \in \mathcal { N } ( \bar { B } )$ , we include the match $( u , v )$ in $\mathcal { R } ( A , B )$ iff $v \in \mathsf { N N } _ { k } ( u ; \mathsf { \bar { B } } )$ and $u \in \mathrm { N N } _ { k } ( v ; A )$ , where $\mathrm { N N } _ { k } ( \boldsymbol { u } ; B )$ denotes the top-k neurons in model B most similar to $u ,$ and analogously for $\mathrm { N N } _ { k } ( v ; A )$ . Unless otherwise stated, all experiments use the default setting $k = 1 ;$ we ablate this choice in Section C.2. Mutual nearest-neighbor matching provides a simple way to retain robust correspondences while filtering asymmetric or noisy nearest-neighbor matches (Dekel et al., 2015). We therefore interpret the pairs of neurons in $\mathcal { R } ( A , \dot { B } )$ as common units across models, and refer to them as Rosetta Neurons. Figure 2 visualizes how these neurons are identified and what the resulting matches look like.

# 4 Scaling Laws for Rosetta Neurons

In this section, we study how the number of Rosetta Neurons scales with model size in both language and vision models. We first describe the experimental setting, including the data and model families. We then analyze the resulting scaling trends in both domains and compare them against a null baseline. We conclude with a phenomenological model to explain the observed scaling behavior.

# 4.1 Experimental Setup

Data. In the language setting, each pair of models used for matching is evaluated on a shared set of approximately 10 million tokens formed by sampling sequences i.i.d. from The Pile (Gao et al., 2020). For vision models, we follow Dravid et al. (2023) and match a generative model with a discriminative model. We sample 50,000 class-balanced images from a diffusion model by conditioning on ImageNet-1k labels (Deng et al., 2009), and then pass these images through the discriminative model. Ablations on the data are provided in Section G.

![](images/fa800a0d4c283cc249b248ea47151964f0a8b04534163016d00f6cc1e470e997.jpg)

<details>
<summary>line</summary>

| Model | Model Size (# Neurons) | # Rosetta Neurons |
|-------|------------------------|-------------------|
| OPT - Qwen2.5 | 10^6 | 12.43 |
| Pythia - GPT2 | 10^6 | 2.20 |
| Pythia - OPT | 10^6 | 0.68 |
| OPT - Pythia - GPT2 | 10^6 | 1.99 |
</details>

(a) Rosetta Neuron scaling laws in language models.

![](images/ed28baf2476da8088ebae5ec6ce1687d840b5e14b9e510696abd02a13d79b1db.jpg)

<details>
<summary>line</summary>

| Model Size (# Neurons) | Diffusion - DINOv2 | Diffusion - OpenCLIP | Diffusion - Pixio | Diffusion - OpenCLIP | DINOv2 - Pixio | Pixio - Diffusion |
| ---------------------- | ------------------ | -------------------- | ----------------- | -------------------- | -------------- | ----------------- |
| 1e5                    | 5.25x^0.55         | 8.16x^0.49           | 1.13x^0.63        | 0.23x^0.69          | 0.17x^0.69     | 0.55x^0.55        |
</details>

(b) Rosetta Neuron scaling laws in vision models.   
Figure 3: Scaling laws for Rosetta Neurons in language and vision models. We plot the number of discovered Rosetta Neurons for various model families at different scales. Dashed lines show power-law fits in log-log space. Across all family comparisons, the fitted exponents are sublinear, and the corresponding fits achieve $R ^ { 2 }$ values around 0.99. Further details are provided in Section C.

Model Families. In the language domain, we consider the Pythia, GPT-2, OPT, and Qwen-2.5 model families, spanning roughly 100 million to 30 billion parameters (Radford et al., 2019; Biderman et al., 2023; Zhang et al., 2022; Yang et al., 2024). For vision, we use discriminative models from the OpenCLIP, DINOv2, and Pixio families, spanning scales from approximately 80 million to 5 billion parameters (Cherti et al., 2023; Radford et al., 2021; Oquab et al., 2024; Yang et al., 2025). For the generative model, we leverage one-step diffusion models built on the Diffusion Transformer architecture (Peebles & Xie, 2023). We provide further details on the model families in Section C.

Forming the scaling curves. Given two model families, we form a point on the scaling curve by selecting one model from each family at approximately matched scale. We apply the matching procedure from Section 3 and record the number of discovered Rosetta Neurons. Repeating this across increasingly larger models yields a scaling curve for that family pair. When two or more curves share a common model, we collapse them into a single trajectory by intersecting the Rosetta Neurons identified within the shared model across the comparisons. This produces a stricter notion of universality by keeping neurons shared across more than two model families, allowing us to test whether the same scaling trend holds under a more selective definition of Rosetta Neurons.

Power-law functional form. We model the number of discovered Rosetta Neurons as a power law in model size, $| \mathcal { R } | = c x ^ { \alpha }$ , where |R| denotes the number of discovered neuron correspondences. Because the matched models are selected to be at approximately the same scale, we use their average total neuron count x to represent the scale of the matched models. In this parameterization, the exponent α is the primary quantity of interest, since it governs how rapidly the number of shared neurons grows with scale, while the constant c depends more strongly on the particular model families and experimental setting. We estimate the fitted curve using ordinary least squares in log-log space.

# 4.2 Rosetta Neurons Follow Power-Law Scaling

We establish that the Rosetta Neuron population follows a sublinear power law in both language and vision models. We then show that this trend is absent in untrained networks, indicating that the scaling law is a property of learned representations rather than an artifact of the matching procedure.

Power-law scaling in language models. Across the language model families we study, the number of Rosetta Neurons increases predictably with model size as shown in Figure 3a. The resulting curves are well fit by power laws, with $R ^ { 2 }$ values around 0.99. Notably, the fitted exponents lie in a narrow and consistently sublinear range, between approximately $0 . 5 \stackrel { - } { - } 0 . 7 .$ This indicates that the number of shared neurons grows predictably with model size, but slower than the rate at which models are scaling. These scaling laws span model sizes from roughly 100 million to 30 billion parameters, corresponding to about 40 thousand to 2 million neurons. The same qualitative behavior holds for the collapsed curves (Pythia-OPT-Qwen2.5, OPT-Pythia-GPT2), suggesting that the scaling trends remain even under a stricter notion of universality.

Power-law scaling in vision models. We observe that a similar scaling trend for Rosetta Neurons emerges in vision models trained with distinct objectives. The scaling curves are well described by power laws, as shown in Figure 3b, with $R ^ { 2 }$ values around 0.99. The fitted exponents fall in a sublinear range between approximately $0 . 5 - 0 . 7$ , indicating a stable Rosetta Neuron scaling regime across model families. These scaling laws span model sizes from roughly 80 million to 5 billion parameters, or 40 thousand to 600 thousand neurons. Importantly, this trend holds across models trained with contrastive image-text supervision, self-distillation, masked autoencoding, and diffusion or flow-based generative modeling. Collapsing curves across shared models reveals a similar scaling pattern, suggesting that the observed scaling law for Rosetta Neurons is robust across multiple models.

Power-law scaling is absent in untrained networks. To test whether our previously observed scaling laws could be induced by the matching procedure itself, we apply the same pipeline to untrained networks initialized according to their architecture-specific random initialization schemes. We report the results across three random seeds in Figure 4. In both language and vision, this yields a marginal number of Rosetta Neuron matches, with no systematic trend as model size increases. Thus, the increasing count of Rosetta Neurons cannot be explained trivially by a larger selection pool. This suggests that our discovery algorithm does not by itself induce the observed scaling behavior. A complementary trained-network is also absent when input alignment is corrupted.

![](images/8ede467a1cf2641542f6603571c807ede24e600e0074b866d7112a61ac1b8c40.jpg)

<details>
<summary>line</summary>

| Model Size (# Neurons) | Language | Vision |
| ---------------------- | -------- | ------ |
| 5 × 10⁴                | 22       | 4      |
| 1 × 10⁵                | 29       | 3      |
| 5 × 10⁵                | 25       | 2      |
| 1 × 10⁶                | 22       | 1      |
</details>

Figure 4: Rosetta Neuron counts in untrained networks lack systematic scaling.

null in Section C.3 shows that the scaling trend

# 4.3 An Analytical Model of Rosetta Neuron Scaling

The empirical scaling law in Figure 3 raises a question: why do Rosetta Neurons grow as a sublinear power law, occupying a shrinking fraction of the total neuron population? We adopt a capacityallocation view of superposition (Elhage et al., 2022; Scherlis et al., 2022): many features compete for representation in a limited number of neuron coordinates. High-importance features, which contribute more to reducing downstream loss, are worth representing more cleanly, while lower-importance features remain mixed in superposition. This gives a natural model of Rosetta Neurons: neurons mostly explained by a single shared feature are reproducible across independently trained networks, whereas neurons that mix many features can do so in network-specific ways and are harder to match. The analytical model below formalizes this intuition; derivations and simulations are in Section D.

Feature-isolation setup. We consider the setting where a network with N neuron coordinates remains in a superposition regime as it scales (Liu et al., 2025): it represents $A ( N ) > N$ latent features with $\hat { A ( N ) } = \Omega ( N )$ , so not every feature can receive a dedicated coordinate. This setting could arise through several non-exclusive mechanisms, which we do not attempt to distinguish here. It could come from effectively infinite data relative to finite model size (Elhage et al., 2022), or from regularization pressures (Liu et al., 2025; Bricken et al., 2023a). We assume a power-law featureimportance spectrum, motivated by prior scaling-law theories in which learnable structure is organized into ranked modes (Michaud et al., 2023; Bordelon et al., 2020; Bahri et al., 2024). Specifically, features have importance $w _ { r } \propto r ^ { - \beta }$ with $\beta > 1$ , where smaller r denotes a more important feature and $\beta > 1$ ensures finite total importance. Intuitively, importance measures a feature’s value for reducing loss, reflecting factors such as frequency, predictiveness, and task relevance.

In a superposition regime, a single neuron may contain signal from many features, and a single feature may be distributed across many neurons. We summarize how cleanly feature r is captured by any individual neuron with an isolation score $s _ { r } \geq 0$ . Informally, $s _ { r }$ compares how much of the activation variance in the neuron most aligned with feature r comes from feature r, versus from other features mixed into that neuron. Thus, $s _ { r }$ is large when one neuron is dominated by feature r, and small when every neuron containing feature r is strongly mixed with other features. We formalize this quantity in Section D, where $s _ { r }$ is derived as a signal-to-interference ratio.

![](images/e0305cde31ede6b1f1338749b79ddf2033dd25b2eb78a797c7f36fef00dbffd0.jpg)

<details>
<summary>bar</summary>

| Ordered Feature Rank r | Importance W_r^α r^-β | Isolation S_r |
| ---------------------- | --------------------- | ------------ |
| r_τ(N)                 | ~1.5                  | ~0.8         |
| r_0(N)                 | ~0.3                  | ~0.2         |
| A(N)                   | ~0.1                  | ~0.05        |
</details>

Figure 5: Feature-isolation frontiers. Features are ordered by decreasing importance $w _ { r } \propto r ^ { - \beta }$ . The optimal allocation partitions the spectrum into Rosetta-detectable features with $s _ { r } \geq \tau$ , partially isolated features with $0 < s _ { r } < \tau .$ , strongly superposed features with $s _ { r } = 0$ , and features beyond the represented set $A ( N )$ . The frontiers $r _ { \tau } ( N )$ and $r _ { 0 } ( N )$ scale as $\Theta ( N ^ { 1 / \beta } )$ , yielding the sublinear Rosetta Neuron count $R _ { \tau } ( N ) = \Theta ( N ^ { 1 / \beta } )$ .

We now use this isolation score to relate superposition to Rosetta Neuron matching across models. Assuming matched networks trained on the same data distribution share a common feature spectrum, Rosetta Neurons can be analyzed through whether a feature is sufficiently isolated in a representative single model. Intuitively, high-isolation features produce reproducible matches because the same shared feature dominates the neuron, whereas low-isolation features are obscured by model-specific mixtures of other tail features. We formalize this intuition in Section D.6 by relating $s _ { r }$ to expected cross-model activation correlation under model-specific interference. We call a feature Rosettadetectable if $s _ { r } \geq \tau ;$ its shared signal is strong enough relative to model-specific interference to produce a reproducible single-neuron match. We take $\tau > 1$ , which ensures in the signal-tointerference model that each detectable feature has a distinct dedicated neuron.

Capacity-allocation objective. Given this setup, we model the network as solving an allocation problem with two ingredients. First, detecting a feature from a noisy single-neuron activation has diminishing returns: making a poorly represented feature cleaner is valuable, but further purifying an already-clean feature helps less. Second, the total budget for isolating features in single neurons scales linearly with the number of coordinates N . This yields:

$$
\max _ {s _ {r} \geq 0} \sum_ {r} w _ {r} \log (1 + s _ {r}) \quad \text {   s.t.   } \quad \sum_ {r} s _ {r} \leq \kappa N. \tag {3}
$$

In Section D, we derive the logarithmic utility from a Gaussian channel model in which the neuron most aligned with feature r provides a noisy estimate of that feature’s activation. Increasing $s _ { r }$ reduces the optimal negative log-likelihood by an amount proportional to $\log ( 1 + s _ { r } )$ . The linear budget follows from a bounded-activation-energy condition in the underlying linear superposition model. Solving this allocation problem in the continuum limit yields a simple frontier structure:

$$
s ^ {\star} (r; N) = \left[ \left(\frac {r _ {0} (N)}{r}\right) ^ {\beta} - 1 \right] _ {+}, \quad [ u ] _ {+} = \max (u, 0). \tag {4}
$$

so features below a cutoff rank $r _ { 0 } ( N )$ are more cleanly isolated in individual neurons, while features beyond it remain in superposition with $s _ { r } = 0$ . This cutoff is set by the total budget:

$$
\kappa N = \int_ {1} ^ {r _ {0}} \left[ \left(\frac {r _ {0}}{r}\right) ^ {\beta} - 1 \right] d r = \Theta (r _ {0} ^ {\beta}), \tag {5}
$$

and hence $r _ { 0 } ( N ) = \Theta ( N ^ { 1 / \beta } )$ . Let $R _ { \tau } ( N )$ count threshold-crossing features, which correspond to Rosetta Neurons in this idealized model. Solving $s ^ { \star } ( r _ { \tau } ; N ) = \tau$ for rτ yields

$$
r _ {\tau} (N) = r _ {0} (N) (1 + \tau) ^ {- 1 / \beta}, \quad R _ {\tau} (N) = \Theta (r _ {\tau} (N)) = \Theta (N ^ {1 / \beta}). \tag {6}
$$

Since $\beta > 1$ , the Rosetta count grows as a sublinear power law. Figure 5 summarizes this frontier structure: as model size N increases, the Rosetta-detectability frontier moves outward, while a long tail of lower-importance features remains either partially isolated or strongly superposed with $s _ { r } = 0$ .

![](images/5623dd061433f0e904752981ddada45b708db011f4beaf2d2bd24f65027cdef7.jpg)

<details>
<summary>line</summary>

| Model Size (# Neurons) | Rosetta Neurons | Non-Rosetta Neurons |
| ---------------------- | --------------- | ------------------- |
| 5 × 10⁴                | 3.0             | 3.0                 |
| 1 × 10⁵                | 3.5             | 0.5                 |
| 5 × 10⁵                | 10.0            | 0.8                 |
| >5 × 10⁵               | 22.0            | 1.0                 |
</details>

(a) Vocabulary-space neuron selectivity in Pythia.

![](images/e7719ed9e95b2cfd6dee64748bd3672b8fedbfa10f0e6779f76ad188bc75b6c6.jpg)

<details>
<summary>line</summary>

| Model Size (# Neurons) | Rosetta Neurons Monosemolarity Rate | Non-Rosetta Neurons Monosemolarity Rate |
| ----------------------- | ------------------------------------- | ----------------------------------------- |
| 5 × 10⁴                 | 0.32                                  | 0.22                                      |
| 1 × 10⁵                 | 0.36                                  | 0.17                                      |
| 5 × 10⁵                 | 0.52                                  | 0.18                                      |
| >5 × 10⁵                | 0.64                                  | 0.10                                      |
</details>

(b) VLM-judged monosemanticity rate in OpenCLIP.   
Figure 6: The Neuron Polarization Effect in language and vision models. (a) In language models, Rosetta Neurons show increasing mean excess kurtosis of vocabulary-space projections with scale. Non-Rosetta neurons remain near zero, indicating weak selectivity. (b) In vision models, VLM-judged monosemanticity increases with scale for Rosetta Neurons and decreases for non-Rosetta neurons.

Prediction: Neuron Polarization. As network scale increases, the Rosetta frontier moves outward in rank, but the features already inside the frontier become cleaner. Averaging the optimal allocation over the Rosetta set and the non-Rosetta tail, respectively, gives

$$
\bar {s} _ {\text { Rosetta }} (N) = \Theta \left(N ^ {(\beta - 1) / \beta}\right), \quad \bar {s} _ {\text { non - Rosetta }} (N) = O \left(\frac {N ^ {1 / \beta}}{A (N)}\right)\rightarrow 0. \tag {7}
$$

Thus, Rosetta Neurons become increasingly clean and selective, while the unresolved feature tail remains packed in a polysemantic background. We validate this prediction in the next section.

# 5 Properties of Rosetta Neurons

Having established Rosetta Neuron scaling laws, we next investigate how their properties change with scale. We test the predicted Neuron Polarization Effect across modalities, examine semantic specialization in language models, and conclude with a case study demonstrating that an individual Rosetta Neuron can be both selective and specialized enough to support domain-specific data filtering. Qualitative examples in Section A.2 show the same pattern, with Rosetta Neurons firing on more interpretable and coherent concepts compared to non-Rosetta neurons.

# 5.1 The Neuron Polarization Effect in Large-Scale Models

We next test the predicted Neuron Polarization Effect: scaling separates increasingly monosemantic Rosetta Neurons from a comparatively polysemantic non-Rosetta background. We observe this trend using heuristic neuron selectivity measures in both language and vision models.

Language models. We measure language-model neuron selectivity through vocabulary-space projections (Gurnee et al., 2024). For each neuron $u = ( \ell , c )$ , we compare its output weight u u $\dot { W } _ { \mathrm { o u } } ^ { u }$ to each token’s unembedding vector $W _ { U } [ v ]$ via cosine similarity: $s _ { v } ^ { u } = \dot { \mathrm { c o s } } ( W _ { \mathrm { o u t } } ^ { u } , \mathbf { \dot { W } } _ { U } [ v ] )$ . We use the excess kurtosis of these alignments as the neuron’s output-side selectivity metric, where higher values indicate concentration on a small set of tokens. In Pythia, we compute the mean of this metric separately over Rosetta and non-Rosetta neurons, defining the Rosetta set at each scale using the OPT–Pythia matches. We plot this metric as a function of model size in Figure 6a, and find that the Rosetta mean increases with scale, indicating greater concentration on a small set of tokens and more coherent, monosemantic functionality (Avrahamy et al., 2026). In contrast, the non-Rosetta mean decreases and remains near zero, consistent with weak selectivity and a more polysemantic population. We observe the same qualitative trend in other language model families (Section E).

Vision models. Since vision models do not admit the same natural vocabulary-space selectivity measure as language models, we use a VLM-as-a-judge proxy for monosemanticity, building on the established practice of using multimodal models to describe neurons (Oikarinen & Weng, 2023;

Rosetta Neuron Firing by Document Type   
![](images/63b55c7c55a2093136d91a10d7897f9fc929b90b8d4a942af7de5e16a24f7620.jpg)

<details>
<summary>bar</summary>

| Text Source Category | 160M | 410M | 1.4B | 2.8B | 6.9B | 12B |
|----------------------|------|------|------|------|------|-----|
| Code                 | 0.4x | 0.5x | 0.7x | 1.0x | 1.1x | 1.1x |
| Math                 | 0.7x | 0.8x | 1.3x | 1.2x | 1.3x | 1.4x |
| Formal / Science      | 0.8x | 0.8x | 0.8x | 0.8x | 0.9x | 0.9x |
| General Prose         | 1.4x | 1.4x | 1.4x | 1.3x | 1.2x | 1.0x |
| Conversational        | 1.5x | 1.5x | 1.4x | 1.3x | 1.3x | 1.2x |
</details>

Figure 7: Rosetta Neuron document-type firing in Pythia. For each Pythia model size, we plot how often top-activating Rosetta Neuron contexts fall into a document category, normalized by that category’s token frequency in the validation set. The dashed line marks the corpus baseline. With scale, Rosetta Neuron firing shifts toward specialized categories such as code and math.

Shaham et al., 2024). At each scale, we evaluate the OpenCLIP neurons identified in the Diffusion– OpenCLIP Rosetta matches. For each neuron, we present GPT-5.4 with its top 20 activating images, activation maps, and overlays, and ask whether the responses reflect a single coherent visual feature. We evaluate five disjoint subsets of 100 Rosetta Neurons and five matched subsets of non-Rosetta neurons. In Figure 6b, we plot the fraction of neurons judged monosemantic as OpenCLIP model size scales. This fraction increases with scale for Rosetta Neurons and decreases for non-Rosetta neurons, consistent with the Neuron Polarization Effect observed in language models. Additional results on other models, methodological details, and a metric reliability study are provided in Section H.

# 5.2 Rosetta Neurons Become More Specialized with Scale in Language Models

We now investigate whether Rosetta Neurons become more specialized with scale by measuring their top-activating document types across Pythia models of different sizes. Our analytical model predicts that as the Rosetta frontier expands, lower-ranked features can become sufficiently clean and reproducible to be shared across independently trained models. As rarer data patterns lie deeper in the feature importance spectrum, larger models should increasingly contain Rosetta Neurons selective for specialized domains, appearing as a scale-dependent shift in document-type firing.

Measuring Rosetta Neuron firing by document type. We measure how Rosetta Neuron firing patterns vary across text categories with model size. At each Pythia scale, we use Rosetta Neurons from the OPT–Pythia scaling runs and collect each neuron’s top-20 activating contexts from the Pile validation set. We assign each context to one of five categories: code, math, formal/scientific text, general prose, or conversational text. For each category, we compare its share of Rosetta Neuron top activations with its token share in the validation cache. A value of 1 corresponds to the corpus baseline, where top Rosetta activations fall in a category at the same fraction as that category’s token share in the dataset. Additional details on the category construction are in Section E.

Rosetta Neurons shift toward specialized domains with scale. Figure 7 shows that Rosetta Neuron firing becomes increasingly concentrated on specialized text domains with model scale. Code and math become increasingly represented among top activations relative to their dataset frequency, while broader categories such as general prose and conversational text decline. Since all Pythia models were trained on the same data mixture and token budget, this shift cannot be explained by greater exposure to specialized data. Instead, scale appears to change how models organize internal representations: larger models do not just add shared neurons for basic features, but expand the Rosetta population toward increasingly specialized domains. Although our results are aggregated over many Rosetta Neurons, it is consistent with qualitative examples in Section A, where individual Rosetta Neurons become increasingly domain-selective rather than firing broadly across text types. In contrast, a non-Rosetta neuron baseline in Section E shows fairly stable document-type preferences across scale, with no corresponding shift toward specialized domains. In the next section, we move from population-level specialization to a single-neuron case study, testing whether an individual Rosetta Neuron is both selective and specialized enough to support domain-specific data filtering.

# 5.3 Testing Rosetta Neuron Selectivity with Data Filtering

We next study specialization at the single-neuron level, testing whether an individual Rosetta Neuron can identify a coherent, specialized domain well enough to support data filtering. In a controlled codedomain setting, we use ground-truth labels to evaluate domain recovery and continued pretraining to test downstream utility.

Data-filtering setup. We use CodeSearchNet (Husain et al., 2019), a multilingual code corpus with source-language labels. We take JavaScript as a representative target domain, whose training split contains roughly 58K parsed code functions totaling 16M tokens. Each filtering method selects functions from the multilingual pool under a 16M token budget, and we evaluate recovery by F1 score against the ground-truth JavaScript subset. We then test how well the recovered data supports learning the target domain. We continue pretraining GPT2-1.5B, which has limited code exposure, on the selected data and report JavaScript test-set perplexity.

Under this matched token-budget setup, we compare four filtering methods. The Rosetta Neuron filter uses a single JavaScript-selective Rosetta Neuron from Pythia-6.9B, discovered in the Pythia–OPT matching runs on the Pile. We score CodeSearchNet training functions by this neuron’s activations and select the highest-scoring functions up to the 16M-token budget. We apply the same procedure to a non-Rosetta neuron that activates on JavaScript text, identified on the same Pile cache. We compare these neuron-based filters against two baselines: a uniform random sample from the multilingual pool and the oracle JavaScript subset. Additional experimental details are in Section F.

Rosetta Neuron selectivity predicts useful target-domain data. Table 1 reports each filter’s F1 recovery of the groundtruth JavaScript subset and test-set perplexity over three training runs. All filters improve over the base model, consistent with GPT2-1.5B’s limited code exposure. Notably, the Rosetta filter recovers nearly the entire JavaScript subset, achieving 0.98 F1. This selectivity translates into downstream utility: continued pretraining on Rosetta-filtered data reduces test perplexity from 3.59 for random filtering to 3.02, nearly matching the oracle at 3.01. In contrast, the non-Rosetta JavaScript neuron improves over random filtering but is less selective and yields weaker downstream gains. Qualitative examples in Section F suggest that the non-Rosetta neuron is less specialized, firing on broader web-programming syntax and JavaScript-adjacent languages

<table><tr><td>Method</td><td>F1</td><td>Test PPL</td></tr><tr><td>Base Model</td><td>-</td><td>6.73</td></tr><tr><td>Random</td><td>0.06</td><td>3.59 ± 0.07</td></tr><tr><td>Non-Rosetta</td><td>0.09</td><td>3.23 ± 0.07</td></tr><tr><td>Rosetta</td><td>0.98</td><td>3.02 ± 0.05</td></tr><tr><td>Oracle</td><td>1.00</td><td>3.01 ± 0.04</td></tr></table>

Table 1: JavaScript filtering. Matched-budget filters on Code-SearchNet; PPL reports mean ± 95% CI over three runs.

rather than specifically JavaScript. While this single-neuron experiment serves as a controlled case study of downstream data filtering, qualitative examples in Section A suggest that other Rosetta Neurons exhibit similarly interpretable, selective firing.

# 6 Discussion, Limitations, and Future Work

We studied whether the internal organization of neural networks evolves predictably with scale. Using Rosetta Neurons as a probe, we found sublinear power-law growth of shared single-neuron structure in language and vision models. Together, our analytical model and empirical measurements suggest that scaling polarizes neurons: a shared Rosetta population grows more selective, while the remaining neurons form a larger, superposed background. We further showed that Rosetta Neurons become more specialized with scale, demonstrating this specialization in a controlled data-filtering study for continued pretraining. We close by discussing the limitations and scope of the analysis, along with future directions; additional discussion and failure cases are in Section I.

Our analysis targets a specific form of neuron-level structure: Rosetta Neurons. This is only a subset of the structure inside neural networks. Some computations may be carried by circuits (Elhage et al., 2021), attention heads (Olsson et al., 2022), or subspaces that don’t appear as neuron correspondences (Wang et al., 2018; Bricken et al., 2023a). Thus, the Rosetta population should not be viewed as a map of all shared computation. We instead view Rosetta Neurons as a tractable observable: a shared neuron-level population for studying structure that is not specific to any single training run, architecture, or model family. Combining universality with scale makes Rosetta Neurons a powerful discovery tool: cross-model recurrence identifies candidate interpretable units without target concepts or auxiliary training, while comparisons across model size expose predictable changes in this structure. However, the resulting trends should be interpreted within the scope of our measurements. For instance, our monosemanticity measurements rely on modality-specific proxies, and should be viewed as relative population-level trends.

More broadly, our results point toward a bridge between macroscopic scaling behavior and the microscopic organization of learned representations. Standard scaling laws describe external quantities such as loss as a function of scale, while Rosetta Neurons provide a complementary lens on how shared structure can evolve predictably at the level of individual units. This opens several directions: discovering other internal observables whose structure changes predictably with scale, studying how universal neurons emerge during training, how they are transformed by post-training, and how they relate to distributed forms of shared computation. Our analytical model offers a simple account of Rosetta Neuron scaling and polarization, but deriving these phenomena from gradient-based training dynamics remains an important next step. Such an account could explain how universal structure emerges from optimization, and may inform objectives or regularizers that shape not only loss scaling but also the internal organization and adaptability of learned representations.

Acknowledgments. We thank members of Berkeley AI Research and the Redwood Center for Theoretical Neuroscience for helpful discussions. We are particularly grateful to Mason Kamb, Phillip Isola, David Bau, Yizhou Liu, Sophie Wang, Grace Luo, Stephanie Fu, Jasmine Shone, Tamar Rott Shaham, and Tyler Bonnen for their thoughtful feedback. YB is a visiting scholar at UC Berkeley and member of the Simons Collaboration on the Physics of Learning & Neural Computation. AD is supported by the US Department of Energy Computational Science Graduate Fellowship. Additional support came from ONR MURI, NSF IIS-2403305, and the Google-BAIR Commons Program.

# References

Sanjeev Arora, Yuanzhi Li, Yingyu Liang, Tengyu Ma, and Andrej Risteski. Linear algebraic structure of word senses, with applications to polysemy. Transactions of the Association for Computational Linguistics, 6:483–495, 2018.   
Asaf Avrahamy, Yoav Gur-Arieh, and Mor Geva. Disentangling mlp neuron weights in vocabulary space. arXiv preprint arXiv:2604.06005, 2026.   
Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.   
Yasaman Bahri, Ethan Dyer, Jared Kaplan, Jaehoon Lee, and Utkarsh Sharma. Explaining neural scaling laws. Proceedings of the National Academy of Sciences, 121(27):e2311878121, 2024.   
Yamini Bansal, Preetum Nakkiran, and Boaz Barak. Revisiting model stitching to compare neural representations. Advances in neural information processing systems, 34:225–236, 2021.   
David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 6541–6549, 2017.   
Stella Biderman, Hailey Schoelkopf, Quentin Gregory Anthony, Herbie Bradley, Kyle O’Brien, Eric Hallahan, Mohammad Aflah Khan, Shivanshu Purohit, USVSN Sai Prashanth, Edward Raff, et al. Pythia: A suite for analyzing large language models across training and scaling. In International conference on machine learning, pp. 2397–2430. PMLR, 2023.   
Steven Bills, Nick Cammarata, Dan Mossing, Henk Tillman, Leo Gao, Gabriel Goh, Ilya Sutskever, Jan Leike, Jeff Wu, and William Saunders. Language models can explain neurons in language models. https://openaipublic.blob.core.windows.net/neuron-explainer/ paper/index.html, 2023.   
Black Forest Labs. FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2, 2025.   
Blake Bordelon, Abdulkadir Canatar, and Cengiz Pehlevan. Spectrum dependent learning curves in kernel regression and wide neural networks. In International Conference on Machine Learning, pp. 1024–1034. PMLR, 2020.

Trenton Bricken, Rylan Schaeffer, Bruno Olshausen, and Gabriel Kreiman. Emergence of sparse representations from noise. In Proceedings of the 40th International Conference on Machine Learning, pp. 3148–3191, 2023a.   
Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023b. https://transformer-circuits.pub/2023/monosemanticfeatures/index.html.   
Francesco Cagnetta, Allan Raventós, Surya Ganguli, and Matthieu Wyart. Deriving neural scaling laws from the statistics of natural language. arXiv preprint arXiv:2602.07488, 2026.   
Junsong Chen, Shuchen Xue, Yuyang Zhao, Jincheng Yu, Sayak Paul, Junyu Chen, Han Cai, Song Han, and Enze Xie. Sana-sprint: One-step diffusion with continuous-time consistency distillation. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16185–16195, 2025.   
Mehdi Cherti, Romain Beaumont, Ross Wightman, Mitchell Wortsman, Gabriel Ilharco, Cade Gordon, Christoph Schuhmann, Ludwig Schmidt, and Jenia Jitsev. Reproducible scaling laws for contrastive language-image learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2818–2829, 2023.   
Tali Dekel, Shaul Oron, Michael Rubinstein, Shai Avidan, and William T Freeman. Best-buddies similarity for robust template matching. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2021–2029, 2015.   
Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.   
Amil Dravid, Yossi Gandelsman, Alexei A Efros, and Assaf Shocher. Rosetta neurons: Mining the common units in a model zoo. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 1934–1943, 2023.   
Shimon Edelman. Representation is representation of similarities. Behavioral and brain sciences, 21 (4):449–467, 1998.   
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021.   
Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. arXiv preprint arXiv:2209.10652, 2022.   
Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.   
Mor Geva, Roei Schuster, Jonathan Berant, and Omer Levy. Transformer feed-forward layers are key-value memories. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pp. 5484–5495, 2021.   
Wes Gurnee, Theo Horsley, Zifan Carl Guo, Tara Rezaei Kheirkhah, Qinyi Sun, Will Hathaway, Neel Nanda, and Dimitris Bertsimas. Universal neurons in gpt2 language models. Transactions on Machine Learning Research, 2024.   
Uri Hasson, Yuval Nir, Ifat Levy, Galit Fuhrmann, and Rafael Malach. Intersubject synchronization of cortical activity during natural vision. Science, 303(5664):1634–1640, 2004. doi: 10.1126/ science.1089506. URL https://www.science.org/doi/abs/10.1126/science.1089506.

James V Haxby, M Ida Gobbini, Maura L Furey, Alumit Ishai, Jennifer L Schouten, and Pietro Pietrini. Distributed and overlapping representations of faces and objects in ventral temporal cortex. Science, 293(5539):2425–2430, 2001.   
Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Delving deep into rectifiers: Surpassing human-level performance on imagenet classification. In Proceedings of the IEEE international conference on computer vision, pp. 1026–1034, 2015.   
Joel Hestness, Sharan Narang, Newsha Ardalani, Gregory Diamos, Heewoo Jun, Hassan Kianinejad, Md Mostofa Ali Patwary, Yang Yang, and Yanqi Zhou. Deep learning scaling is predictable, empirically. arXiv preprint arXiv:1712.00409, 2017.   
Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, DDL Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al. Training compute-optimal large language models. arXiv preprint arXiv:2203.15556, 10, 2022.   
Edward J Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Liang Wang, Weizhu Chen, et al. Lora: Low-rank adaptation of large language models. Iclr, 1(2):3, 2022.   
Jing Huang, Daniel Wurgaft, Rachit Bansal, Laura Ruis, Naomi Saphra, David Alvarez-Melis, Andrew Kyle Lampinen, Christopher Potts, and Ekdeep Singh Lubana. Why larger models learn more: Effects of capacity, interference, and rare-task retention. arXiv preprint arXiv:2605.29548, 2026.   
DH Hubel and TN Wiesel. Receptive fields, binocular interaction and functional architecture in the cat’s visual cortex. Journal of Physiology (London), 1962.   
Minyoung Huh, Brian Cheung, Tongzhou Wang, and Phillip Isola. Position: The platonic representation hypothesis. In Forty-first International Conference on Machine Learning, 2024.   
Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. Codesearchnet challenge: Evaluating the state of semantic code search. arXiv preprint arXiv:1909.09436, 2019.   
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.   
Simon Kornblith, Mohammad Norouzi, Honglak Lee, and Geoffrey Hinton. Similarity of neural network representations revisited. In International conference on machine learning, pp. 3519–3529. PMLR, 2019.   
Nikolaus Kriegeskorte, Marieke Mur, and Peter A Bandettini. Representational similarity analysisconnecting the branches of systems neuroscience. Frontiers in systems neuroscience, 2:249, 2008a.   
Nikolaus Kriegeskorte, Marieke Mur, Douglas A Ruff, Roozbeh Kiani, Jerzy Bodurka, Hossein Esteky, Keiji Tanaka, and Peter A Bandettini. Matching categorical object representations in inferior temporal cortex of man and monkey. Neuron, 60(6):1126–1141, 2008b.   
Karel Lenc and Andrea Vedaldi. Understanding image representations by measuring their equivariance and equivalence. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 991–999, 2015.   
Yixuan Li, Jason Yosinski, Jeff Clune, Hod Lipson, and John E. Hopcroft. Convergent learning: Do different neural networks learn the same representations? In ICLR, 2016. URL http: //arxiv.org/abs/1511.07543.   
Yizhou Liu, Ziming Liu, and Jeff Gore. Superposition yields robust neural scaling. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview. net/forum?id=knPz7gtjPW.   
Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019. URL https://openreview.net/forum?id= Bkg6RiCqY7.

Yiyang Lu, Susie Lu, Qiao Sun, Hanhong Zhao, Zhicheng Jiang, Xianbang Wang, Tianhong Li, Zhengyang Geng, and Kaiming He. One-step latent-free image generation with pixel mean flows. arXiv preprint arXiv:2601.22158, 2026.   
Eric Michaud, Ziming Liu, Uzay Girit, and Max Tegmark. The quantization model of neural scaling. Advances in Neural Information Processing Systems, 36:28699–28722, 2023.   
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. Advances in neural information processing systems, 26, 2013.   
Ari Morcos, Maithra Raghu, and Samy Bengio. Insights on representational similarity in neural networks with canonical correlation. Advances in neural information processing systems, 31, 2018.   
Tuomas Oikarinen and Tsui-Wei Weng. Clip-dissect: Automatic description of neuron representations in deep vision networks. In The Eleventh International Conference on Learning Representations, 2023.   
Catherine Olsson, Nelson Elhage, Neel Nanda, Nicholas Joseph, Nova DasSarma, Tom Henighan, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, et al. In-context learning and induction heads. arXiv preprint arXiv:2209.11895, 2022.   
Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy V. Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel HAZIZA, Francisco Massa, Alaaeldin El-Nouby, Mido Assran, Nicolas Ballas, Wojciech Galuba, Russell Howes, Po-Yao Huang, Shang-Wen Li, Ishan Misra, Michael Rabbat, Vasu Sharma, Gabriel Synnaeve, Hu Xu, Herve Jegou, Julien Mairal, Patrick Labatut, Armand Joulin, and Piotr Bojanowski. DINOv2: Learning robust visual features without supervision. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. URL https://openreview.net/forum?id=a68SUt6zFt. Featured Certification.   
Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In Proceedings of the 41st International Conference on Machine Learning, pp. 39643–39666, 2024.   
William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4195–4205, 2023.   
Alec Radford, Rafal Jozefowicz, and Ilya Sutskever. Learning to generate reviews and discovering sentiment. arXiv preprint arXiv:1704.01444, 2017.   
Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al. Language models are unsupervised multitask learners. OpenAI blog, 1(8):9, 2019.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pp. 8748–8763. PMLR, 2021.   
Maithra Raghu, Justin Gilmer, Jason Yosinski, and Jascha Sohl-Dickstein. Svcca: Singular vector canonical correlation analysis for deep learning dynamics and interpretability. Advances in neural information processing systems, 30, 2017.   
Adam Scherlis, Kshitij Sachan, Adam S Jermyn, Joe Benton, and Buck Shlegeris. Polysemanticity and capacity in neural networks. arXiv preprint arXiv:2210.01892, 2022.   
Tamar Rott Shaham, Sarah Schwettmann, Franklin Wang, Achyuta Rajaram, Evan Hernandez, Jacob Andreas, and Antonio Torralba. A multimodal automated interpretability agent. In Forty-first International Conference on Machine Learning, 2024.   
Oriane Siméoni, Huy V Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov, Marc Szafraniec, Seungeun Yi, Michaël Ramamonjisoa, et al. Dinov3. arXiv preprint arXiv:2508.10104, 2025.

Ben Sorscher, Surya Ganguli, and Haim Sompolinsky. Neural representational geometry underlies few-shot concept learning. Proceedings of the National Academy of Sciences, 119(43): e2200800119, 2022.   
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.   
Liwei Wang, Lunjia Hu, Jiayuan Gu, Zhiqiang Hu, Yue Wu, Kun He, and John Hopcroft. Towards understanding learning representations: To what extent do different neural networks learn the same representation. Advances in neural information processing systems, 31, 2018.   
An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, et al. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.   
Ge Yang, Edward Hu, Igor Babuschkin, Szymon Sidor, Xiaodong Liu, David Farhi, Nick Ryder, Jakub Pachocki, Weizhu Chen, and Jianfeng Gao. Tuning large neural networks via zero-shot hyperparameter transfer. Advances in Neural Information Processing Systems, 34:17084–17097, 2021.   
Lihe Yang, Shang-Wen Li, Yang Li, Xinjie Lei, Dong Wang, Abdelrahman Mohamed, Hengshuang Zhao, and Hu Xu. In pursuit of pixel supervision for visual pre-training. arXiv preprint arXiv:2512.15715, 2025.   
Xiaohua Zhai, Alexander Kolesnikov, Neil Houlsby, and Lucas Beyer. Scaling vision transformers. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 12104–12113, 2022.   
Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068, 2022.

# Appendix

A Rosetta Neuron Visualizations 17

A.1 Rosetta Neurons across Model Scales 17   
A.2 Qualitative Comparison of Rosetta and Non-Rosetta Neurons . . 22

B Aligning Token-Wise Activations Across Models . . . . 24

C Further Details on Rosetta-Neuron Scaling . . . . 26

C.1 Model Families 26   
C.2 Robustness to the Mutual Top-k Criterion 27   
C.3 Input-Permutation Null 27

D Detailed Derivation of the Rosetta Neuron Scaling Model . 28

D.1 A Minimal Capacity-Allocation Model . . 28   
D.2 Solving the Allocation Problem 29   
D.3 Prediction: Neuron Polarization 30   
D.4 Deriving the Isolation Score from Superposition 31   
D.5 Deriving the Logarithmic Utility and Linear Isolation Budget . . . . . . . . 33   
D.6 Connection to Empirical Rosetta Matching . . . . . . . 34   
D.7 Synthetic Validation of the Analytical Model 36

E Additional Results on Rosetta Neuron Properties . 38

E.1 Additional Vocabulary-Space Selectivity Results . . . . 38   
E.2 Document-Type Firing Analysis . . . . . 38   
E.3 Depth-Wise Distribution of Rosetta Neurons . 40

F Data Filtering Experimental Details 41

G Dataset Ablations . 45

G.1 Ablation on the Number of Tokens Used for Language Model Matching . . . . . . 45   
G.2 Ablation on the Number of Images Used for Vision Model Matching . 45   
G.3 Ablation on the Image Distribution Used for Vision Model Matching . . . . . 45

H Additional Details on VLM-as-a-Judge . . . 47

H.1 Detailed Experimental Setup 47   
H.2 Sensitivity to the Number of Top-k Activating Images . . . 48   
H.3 Validation of VLM-as-a-Judge as a Predictive Metric 48   
H.4 Results for Neuron Selectivity in Other Vision Models . . . . 49

I Further Discussion and Limitations 49

I.1 DINOv3 as an Informative Failure Case . 49   
I.2 Operationalizing Monosemanticity and Polysemanticity . . . . 50

J Compute Resources . . . 50

# A Rosetta Neuron Visualizations

# A.1 Rosetta Neurons across Model Scales

We provide qualitative examples of Rosetta Neurons across model scales in both language and vision. For language models, we show LLM annotations for a subset of Rosetta Neurons across matched model families along with representative top-activating input spans (Figs. 8–13). The system prompt used for the LLM annotator is provided below. For vision models, we visualize the top three activating images and corresponding activation maps for a subset of Rosetta Neurons at different scales (Figs. 14–18). We report neuron identities in the captions using the format L{layer}/U{unit}, together with the coarse model names. Detailed model names are provided in Section C.1.

You are evaluating what a neuron in a language model responds to. Below are the top 20 sequences that most strongly activate this neuron that is shared across multiple language models. Each box shows a text sequence with a heatmap overlay indicating where the corresponding neuron in each model fires (bright = strong activation). Give an overall summary of the concept the neuron responds to in less than 10 words and give up to three examples of what sort of tokens it would fire on.

LLM Annotation: English first names ('Matthew', ‘Michael',‘Emily') 

<table><tr><td>Pythia-160M</td><td>General</td><td>Matthew</td><td>Entsminger summed up the new standard in</td></tr><tr><td>GPT2-124M</td><td>General</td><td>Matthew</td><td>Entsminger summed up the new standard in</td></tr><tr><td>OPT-125M</td><td>General</td><td>Matthew</td><td>Entsminger summed up the new standard in</td></tr></table>

(a) Pythia-160M: L5/U1767. GPT2-124M: L7/U1625. OPT-125M: L1/U2325.

LLM Annotation: Titles of authority('Professor', ‘Dr.','Commissioner") 

<table><tr><td>Pythia-160M</td><td>Commissioner</td><td>Neil Kerch, a defense attorney, declined to</td></tr><tr><td>GPT2-124M</td><td>Commissioner</td><td>Neil Kerch, a defense attorney, declined to</td></tr><tr><td>OPT-125M</td><td>Commissioner</td><td>Neil Kerch, a defense attorney, declined to</td></tr></table>

(b) Pythia-160M: L5/U1898. GPT2-124M: L8/U557. OPT-125M: L8/U2115.

LLM Annotation: URL Domain Extensions ('.com', ‘.org',‘.net') 

<table><tr><td>Pythia-160M</td><td>\nFacebook: https://www.facebook.com/paul.j.watson.71</td></tr><tr><td>GPT2-124M</td><td>\nFacebook: https://www.facebook.com/paul.j.watson.71</td></tr><tr><td>OPT-125M</td><td>\nFacebook: https://www.facebook.com/paul.j.watson.71</td></tr></table>

(c) Pythia-160M: L10/U2431. GPT2-124M: L11/U424. OPT-125M: L8/U2837.

LLM Annotation: Abbreviated months (Nov', 'Mar','Oct') 

<table><tr><td>Pythia-160M</td><td>2018\n\n18</td><td>A5J4DX on Nov 13,</td><td>2018\n\n19</td><td>Anonymous on Mar 13</td></tr><tr><td>GPT2-124M</td><td>2018\n\n18</td><td>A5J4DX on Nov 13,</td><td>2018\n\n19</td><td>Anonymous on Mar 13</td></tr><tr><td>OPT-125M</td><td>2018\n\n18</td><td>A5J4DX on Nov 13,</td><td>2018\n\n19</td><td>Anonymous on Mar 13</td></tr></table>

(d) Pythia-160M: L0/U2626. GPT2-124M: L9/U649. OPT-125M: L0/U888.

Figure 8: LLM annotations for Rosetta Neurons. Comparison between Pythia-160M, GPT2-124M, OPT-125M.   
LLM Annotation: Words associated with auditory perception (heard',‘sound','listen") 

<table><tr><td>Pythia-410M</td><td>Sano heard what sounded like the front door. Privacy was</td></tr><tr><td>GPT2-355M</td><td>Sano heard what sounded like the front door. Privacy was</td></tr><tr><td>OPT-350M</td><td>Sano heard what sounded like the front door. Privacy was</td></tr></table>

(a) Pythia-410M: L10/U3047. L19/U1698. OPT-350M: L12/U927.

LLM Annotation: Countries and nationalities ('American',‘Switzerland',‘German') 

<table><tr><td>Pythia-410M</td><td>&quot;Switzerland.&quot; &quot;Hi.&quot; &quot;you&#x27;re American?&quot; &quot;So are you?&quot; *ANNIE:</td></tr><tr><td>GPT2-355M</td><td>&quot;Switzerland.&quot; &quot;Hi.&quot; &quot;you&#x27;re American?&quot; &quot;So are you?&quot; *ANNIE:</td></tr><tr><td>OPT-350M</td><td>&quot;Switzerland.&quot; &quot;Hi.&quot; &quot;you&#x27;re American?&quot; &quot;So are you?&quot; *ANNIE:</td></tr></table>

(b) Pythia-410M: L9/U1693. GPT2-355M: L15/U1724.OPT-350M: L13/U846.

LLM Annotation: Words associated with female ('woman','spokeswoman','sister") 

<table><tr><td>Pythia-410M</td><td>EPA limit,spokeswoman</td><td>Danielle Oller said.\n\nIn Security</td></tr><tr><td>GPT2-355M</td><td>EPA limit,spokeswoman</td><td>Danielle Oller said.\n\nIn Security</td></tr><tr><td>OPT-350M</td><td>EPA limit,spokeswoman</td><td>Danielle Oller said.\n\nIn Security</td></tr></table>

(c) Pythia-410M: L23/U39. GPT2-355M: L23/U149. OPT-350M: L23/U2921.

LLM Annotation: Christian denominations and church traditions (church',‘Catholic','saint") 

<table><tr><td>Pythia-410M</td><td>Saturday in the case of those churches practicing seventh-day</td></tr><tr><td>GPT2-355M</td><td>Saturday in the case of those churches practicing seventh-day</td></tr><tr><td>OPT-350M</td><td>Saturday in the case of those churches practicing seventh-day</td></tr></table>

(d) Pythia-410M: L23/U3863. GPT2-355M: L23/U1026. OPT-350M: L22/U1841.

Figure 9: LLM annotations for Rosetta Neurons. Comparison between Pythia-410M, GPT2-355M, OPT-350M.

LLM Annotation: Symbolic identifiers in math expressions ('a',‘x',‘y') 

<table><tr><td>Pythia-1.4B</td><td>Let a be p[i]. Let o = a - 18779728. What is o</td></tr><tr><td>GPT2-1.5B</td><td>Let a be p[i]. Let o = a - 18779728. What is o</td></tr><tr><td>OPT-1.3B</td><td>Let a be p[i]. Let o = a - 18779728. What is o</td></tr></table>

(a) Pythia-1.4B: L2/U3564. GPT2-1.5B: L12/U1978. OPT-1.3B: L3/U652.

LLM Annotation: Written-out numbers ('one', 'two','six') 

<table><tr><td>Pythia-1.4B</td><td>642. What is w rounded to the nearest one hundred?\n-800</td></tr><tr><td>GPT2-1.5B</td><td>642. What is w rounded to the nearest one hundred?\n-800</td></tr><tr><td>OPT-1.3B</td><td>642. What is w rounded to the nearest one hundred?\n-800</td></tr></table>

(b) Pythia-1.4B: L9/U835. GPT2-1.5B: L19/U2508. OPT-1.3B: L17/U7443.

LLM Annotation: Spanish text ('conocer', ‘compromiso',‘identificado') 

<table><tr><td>Pythia-1.4B</td><td>que pacten los agentes sociales -incluida CEOE y aseguro</td></tr><tr><td>GPT2-1.5B</td><td>que pacten los agentes sociales -incluida CEOE y aseguro</td></tr><tr><td>OPT-1.3B</td><td>que pacten los agentes sociales -incluida CEOE y aseguro</td></tr></table>

(c) Pythia-1.4B: L15/U2969. GPT2-1.5B: L34/U491. OPT-1.3B: L17/U3726.

LLM Annotation: Encyclopedia-style historical prose('dynasty',‘reign','century') 

<table><tr><td>Pythia-1.4B</td><td>the Chinese dynasty Tang Dynasty, serving as chancellor</td></tr><tr><td>GPT2-1.5B</td><td>the Chinese dynasty Tang Dynasty, serving as chancellor</td></tr><tr><td>OPT-1.3B</td><td>the Chinese dynasty Tang Dynasty, serving as chancellor</td></tr></table>

(d) Pythia-1.4B: L15/U189. GPT2-1.5B: L35/U1327. OPT-1.3B: L18/U4373.

Figure 10: LLM annotations for Rosetta Neurons. Comparison between Pythia-1.4B, GPT2-1.5B, OPT-1.3B.   
LLM Annotation: Dutch language text (wat',‘het', ‘Rotterdam') 

<table><tr><td>Pythia-2.8B</td><td>emde specialisten en zij bevestigden wat de indieners van</td></tr><tr><td>Qwen2.5-3B</td><td>emde specialisten en zij bevestigden wat de indieners van</td></tr><tr><td>OPT-2.7B</td><td>emde specialisten en zij bevestigden wat de indieners van</td></tr></table>

(a) Pythia-2.8B: L20/U9199. Qwen2.5-3B: L31/U5812. OPT-2.7B: L25/U7525.

LLM Annotation: Journalism and media outlets ('newspaper',‘station',‘correspondent') 

<table><tr><td>Pythia-2.8B</td><td>correspondent for the Daily Telegraph and speaks fluent</td></tr><tr><td>Qwen2.5-3B</td><td>correspondent for the Daily Telegraph and speaks fluent</td></tr><tr><td>OPT-2.7B</td><td>correspondent for the Daily Telegraph and speaks fluent</td></tr></table>

(b) Pythia-2.8B: L22/U6182. Qwen2.5-3B: L31/U1823. OPT-2.7B: L27/U3916.

LLM Annotation: Student life and academic status ('student','major','semester') 

<table><tr><td>Pythia-2.8B</td><td>decision that we simply cannot put our student athletes,</td></tr><tr><td>Qwen2.5-3B</td><td>decision that we simply cannot put our student athletes,</td></tr><tr><td>OPT-2.7B</td><td>decision that we simply cannot put our student athletes,</td></tr></table>

(c) Pythia-2.8B: L12/U5719. Qwen2.5-3B: L26/U2175. OPT-2.7B: L21/U7710.

LLM Annotation: Words associated with pathology ('infectious',‘coronavirus',‘bacterial') 

<table><tr><td>Pythia-2.8B</td><td>[animal coronaviruses, influenza viruses], or non-enveloped</td></tr><tr><td>Qwen2.5-3B</td><td>[animal coronaviruses, influenza viruses], or non-enveloped</td></tr><tr><td>OPT-2.7B</td><td>[animal coronaviruses, influenza viruses], or non-enveloped</td></tr></table>

(d) Pythia-2.8B: L28/U10099. Qwen2.5-3B: L31/U6319. OPT-2.7B: L31/U2355.

Figure 11: LLM annotations for Rosetta Neurons. Comparison between Pythia-2.8B, Qwen2.5-3B, OPT-2.7B.

LLM Annotation: Spelled-out round numbers ('twenty','thirty','forty')   
![](images/0b8894759cd0b7923d5d8ca4cf11d28acbdb09bc2d853d565da622d907b1dff3.jpg)

<details>
<summary>text_image</summary>

Pythia-6.9B this manuscript includes the twenty participants only [Table]
Qwen2.5-7B this manuscript includes the twenty participants only [Table]
OPT-6.7B this manuscript includes the twenty participants only [Table]
</details>

(a) Pythia-6.9B: L1/U1161. Qwen2.5-7B: L22/U12072. OPT-6.7B: L2/U7258.

LLM Annotation: Courtroom and appeals terminology('arraignment','pleaded',‘appellant')   
![](images/495138bb5602ac0290f8cda42c33271afb65abf82426ee51b97a8ca4bb4e35e5.jpg)

<details>
<summary>text_image</summary>

Pythia-6.9B simultaneous joint jury and severed bench trial with three
Qwen2.5-7B simultaneous joint jury and severed bench trial with three
OPT-6.7B simultaneous joint jury and severed bench trial with three
</details>

(b) Pythia-6.9B: L14/U15135. Qwen2.5-7B: L23/U5673. OPT-6.7B: L26/U14602.

LLM Annotation: JavaScript source code syntax ('function',‘window',‘const')   
![](images/9c059c0378cace531a4428a65df2abe2d4aee047b005ee485ff8538ae27a07be.jpg)

<details>
<summary>text_image</summary>

Pythia-6.9B [window. innerHeight || document. documentElement. clientHeight
Qwen2.5-7B [window. innerHeight || document. documentElement. clientHeight
OPT-6.7B [window. innerHeight || document. documentElement. clientHeight
</details>

(c) Pythia-6.9B: L16/U11168. Qwen2.5-7B: L22/U14665. OPT-6.7B: L26/U2660.

LLM Annotation: Assumptions in mathematical proofs('suppose','let','assume')   
![](images/4adb504fe6db3a13298550bddd2db61d6d887f4138ade2d0e8b464eec4b99554.jpg)  
(d) Pythia-6.9B: L27/U826. Qwen2.5-7B: L26/U5796. OPT-6.7B: L29/U14520.

Figure 12: LLM annotations for Rosetta Neurons. Comparison between Pythia-6.9B, Qwen2.5-7B, OPT-6.7B.   
LLM Annotation: Words associated with gradientbased optimization ('gradient',‘SGD',‘descent')   
![](images/57fbffdfc8a4c4545e1228384517e44b678ec08777ad30fde86c7abdbed2f479.jpg)

<details>
<summary>text_image</summary>

Pythia-12B The primal-dual gradient-based method is one of such methods
Qwen2.5-14B The primal-dual gradient-based method is one of such methods
OPT-13B The primal-dual gradient-based method is one of such methods
</details>

(a) Pythia-12B: L1/U17927. Qwen2.5-14B: L0/U3786. OPT-13B: L0/U4028.

LLM Annotation: Clinical brain terminology ('neuro',‘brain',‘cognitive")   
![](images/46eeae18a720e5785315e94e098634da43ab4df008bed8683372a029009ebae5.jpg)

<details>
<summary>text_image</summary>

Pythia-12B cognitive testing. To enlarge the cohort with imaging and
Qwen2.5-14B cognitive testing. To enlarge the cohort with imaging and
OPT-13B cognitive testing. To enlarge the cohort with imaging and
</details>

(b) Pythia-12B: L1/U19682. Qwen2.5-14B: L3/U1521. OPT-13B: L0/U2590.

LLM Annotation: Electoral politics (Democrats', ‘Republicans','seats')   
![](images/cd45bdc75b7db844d832a7c839faa5729614b335d6001daeb0b258db32482178.jpg)

<details>
<summary>text_image</summary>

Pythia-12B conservative Democrats to switch parties, which many
Qwen2.5-14B conservative Democrats to switch parties, which many
OPT-13B conservative Democrats to switch parties, which many
</details>

(c) Pythia-12B: L24/U10513. L33/U11681. OPT-13B: L29/U14.

LLM Annotation: Ul button elements in code ('UIButton',‘JButton',‘MouseButton')   
![](images/d59f26e0aed131d914e46244cb8f882596b086fe8f5f38e7e589908659313098.jpg)

<details>
<summary>text_image</summary>

Pythia-12B */\npublic class SmallButton extends JButton { \n Qwen2.5-14B */\npublic class SmallButton extends JButton { \n OPT-13B */\npublic class SmallButton extends JButton { \n
</details>

(d) Pythia-12B: L1/U4031. Qwen2.5-14B: L0/U6188. OPT-13B: L3/U11416.

Figure 13: LLM annotations for Rosetta Neurons. Comparison between Pythia-12B, Qwen2.5-14B, OPT-13B.

![](images/a9b34ac64224b43dec9b5c92ea24e78806c701396aa0adcfc31cc922622b2dc5.jpg)

<details>
<summary>text_image</summary>

DiT-B/16
OpenCLIP ViT-B/16
Image	Activation Map	Overlay	Activation Map	Overlay
</details>

(a) DiT-B/16: L7/U879. OpenCLIP ViT-B/16: L7/U1879.

![](images/9c514b8f775563ed993a35a409d46369eaac46bdfa19b1829e8e81e502f16cfa.jpg)

<details>
<summary>text_image</summary>

DiT-B/16
OpenCLIP ViT-B/16
Image Activation Map Overlay Activation Map Overlay
</details>

(b) DiT-B/16: L11/U610. OpenCLIP ViT-B/16: L4/U415.

![](images/a94caf771f62d702ba2138668b637bde2c3ef104dd19c77a399083acaf101d7f.jpg)

<details>
<summary>text_image</summary>

DiT-B/16
OpenCLIP ViT-B/16
Image Activation Map Overlay Activation Map Overlay
</details>

(c) DiT-B/16: L11/U1641. OpenCLIP ViT-B/16: L4/U730.

![](images/771e2fcf49f9160206c412e39f5487798744bc75f356feecfaff57a3a98ccf2c.jpg)

<details>
<summary>text_image</summary>

DiT-B/16
OpenCLIP ViT-B/16
Image	Activation Map	Overlay	Activation Map	Overlay
</details>

(d) DiT-B/16: L10/U1516. OpenCLIP ViT-B/16: L4/U689.

Figure 14: Top-activating images for Rosetta Neurons. Comparison between DiT-B/16 and OpenCLIP ViT-B/16.   
![](images/10ec47e8d40a27124ab7946e1ed8ab60dbb7eb508479e4b3ebb82cf49bdb1ca5.jpg)

<details>
<summary>text_image</summary>

DiT-L/16
OpenCLIP ViT-L/14
Image Activation Map Overlay Activation Map Overlay
</details>

(a) DiT-L/16: L0/U1228. OpenCLIP ViT-L/14: L1/U1669.

![](images/fd5f097e5a77b71870545416466123a5f1061bfc72102b86289162ab7db1c208.jpg)

<details>
<summary>text_image</summary>

DiT-L/16
OpenCLIP ViT-L/14
Image Activation Map Overlay Activation Map Overlay
</details>

(b) DiT-L/16: L31/U15. OpenCLIP ViT-L/14: L0/U1207.

![](images/edbf709a6c7a7d2e4e158016b1bef9e87d5f8bd48fa424cfade3023c68571d62.jpg)

<details>
<summary>text_image</summary>

DiT-L/16
OpenCLIP ViT-L/14
Image Activation Map Overlay Activation Map Overlay
正朗
正朗
正朗
</details>

(c) DiT-L/16: L31/U1752. OpenCLIP ViT-L/14: L0/U3894.

![](images/6cbc8674d4875a72d3035337748f205ed7cdeee635fc2c58e211d2ee2d765f4a.jpg)

<details>
<summary>text_image</summary>

DiT-L/16
OpenCLIP ViT-L/14
Image Activation Map Overlay Activation Map Overlay
DiT-L/16
OpenCLIP ViT-L/14
</details>

(d) DiT-L/16: L31/U816. OpenCLIP ViT-L/14: L0/U2394.   
Figure 15: Top-activating images for Rosetta Neurons. Comparison between DiT-L/16 and OpenCLIP ViT-L/14.

![](images/f70f92296e22c5ce11545eb20e9a63e598688320f25535cace164a1a904327cc.jpg)

<details>
<summary>text_image</summary>

DiT-H/16
OpenCLIP ViT-H/14
Image	Activation Map	Overlay	Activation Map	Overlay
</details>

(a) DiT-H/16: L45/U1346. OpenCLIP ViT-H/14: L1/U3383.

![](images/29722d1ac0956fe8e41094ea11b5f4215360e8c865819bf901d8aefffc57c2cb.jpg)

<details>
<summary>text_image</summary>

DiT-H/16
OpenCLIP ViT-H/14
Image Activation Map Overlay Activation Map Overlay
</details>

(b) DiT-H/16: L47/U1097. OpenCLIP ViT-H/14: L1/U4324.

![](images/e5944ce7a2934fac9f23f82eaa0729b6a1a5321728fadc614f6e5b2038754a9b.jpg)

<details>
<summary>text_image</summary>

DiT-H/16
OpenCLIP ViT-H/14
Image Activation Map Overlay Activation Map Overlay
</details>

(c) DiT-H/16: L47/U1230. OpenCLIP ViT-H/14: L0/U3495.

![](images/429dc093895dc5e8fcdd371c8eaad583966af197e34f335f26891c6ee907d499.jpg)

<details>
<summary>text_image</summary>

DiT-H/16
OpenCLIP ViT-H/14
Image Activation Map Overlay Activation Map Overlay
</details>

(d) DiT-H/16: L47/U300. OpenCLIP ViT-H/14: L0/U477.

Figure 16: Top-activating images for Rosetta Neurons. Comparison between DiT-H/16 and OpenCLIP ViT-H/14.   
![](images/d32567a7c11a202ff463c3ff704659d9029de7c2fdcba13481d5a41aa6363f02.jpg)

<details>
<summary>text_image</summary>

DiT-1.6B
OpenCLIP ViT-bigG/14
Image	Activation Map	Overlay	Activation Map	Overlay
</details>

(a) DiT-1.6B: L44/U2447. OpenCLIP ViTbigG/14: L29/U2762.

![](images/4cfcccec21e4b3eb8276303fa4b220c6f3541bada6f618bdf1d27391a6a76a5a.jpg)

<details>
<summary>text_image</summary>

DiT-1.6B
OpenCLIP ViT-bigG/14
Image	Activation Map	Overlay	Activation Map	Overlay
</details>

(b) DiT-1.6B: L12/U2668. OpenCLIP ViTbigG/14: L20/U203.

![](images/b3c2a7a6ec08ea6bc84294a2d72158eea3f0f5439db315048f9a1050fcabfa11.jpg)

<details>
<summary>text_image</summary>

DiT-1.6B
OpenCLIP ViT-bigG/14
Image	Activation Map	Overlay	Activation Map	Overlay
</details>

(c) DiT-1.6B: L6/U3382. OpenCLIP ViTbigG/14: L39/U7045.

![](images/16fc4e55cb0f4a682495bd64e46c616a218c3fcb1613577165c09b7c48f5e7ba.jpg)

<details>
<summary>text_image</summary>

DiT-1.6B
OpenCLIP ViT-bigG/14
Image	Activation Map	Overlay	Activation Map	Overlay
</details>

(d) DiT-1.6B: L9/U2940. OpenCLIP ViTbigG/14: L28/U2265.   
Figure 17: Top-activating images for Rosetta Neurons. Comparison between DiT-1.6B and OpenCLIP ViT-bigG/14.

![](images/37c2b82c08ec5776f821c1f90f969a86071688087eb0fe21313f75020c4baed3.jpg)

<details>
<summary>text_image</summary>

DiT-4B
OpenCLIP ViT-4.4B
Image Activation Map Overlay Activation Map Overlay
289 289 289
149 149 149
1013 1013 1013
</details>

(a) DiT-4B: L8/U4935. OpenCLIP ViT-4.4B: L41/U7020.

![](images/6ba47a134c900c0dcff9ab57ee34f381bb83ca56a79adf0d9a7c9258a81dd604.jpg)

<details>
<summary>text_image</summary>

DiT-4B
OpenCLIP ViT-4.4B
Image	Activation Map	Overlay	Activation Map	Overlay
</details>

(b) DiT-4B: L2/U4833. OpenCLIP ViT-4.4B: L36/U14930.

![](images/8608d33296adcd538ad42aea79e999a941ea8108c8dc9cbc598bb8d1df97be13.jpg)

<details>
<summary>text_image</summary>

DiT-4B
OpenCLIP ViT-4.4B
Image	Activation Map	Overlay	Activation Map	Overlay
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
</details>

(c) DiT-4B: L9/U3974. OpenCLIP ViT-4.4B: L45/U10399.

![](images/f8241be2f52ec6cc6e7f7cc48559135614146349d5d2ac6da6c2f9aa70535253.jpg)

<details>
<summary>text_image</summary>

DiT-4B
OpenCLIP ViT-4.4B
Image Activation Map Overlay Activation Map Overlay
</details>

(d) DiT-4B: L9/U8380. OpenCLIP ViT-4.4B: L41/U5498.   
Figure 18: Top-activating images for Rosetta Neurons. Comparison between DiT-4B and Open-CLIP ViT-4.4B.

# A.2 Qualitative Comparison of Rosetta and Non-Rosetta Neurons

We compare Rosetta and non-Rosetta neurons by visualizing top-activating examples. In language, we use randomly selected Pythia-6.9B Rosetta Neurons and non-Rosetta neurons from the same layers (Figures 19 and 20); in vision, we conduct a similar comparison for OpenCLIP ViT-L/14 (Figures 21 and 22). Rosetta Neurons appear more coherent and selective, whereas non-Rosetta neurons often respond to a broader mixture of features. This small sample is not standalone evidence, but it is consistent with the quantitative selectivity trends in Section 5.

Pythia-6.9B Rosetta Neuron (L1/U15046)   
![](images/cd6c1597d5f12f3511d61a1554fd40b5313872e4ce93dc1a0ad534c5d96faac3.jpg)

<details>
<summary>text_image</summary>

#1 attracts given a unable or international book \n\nPlease
#2 go-do-it will create seen to your studio lab. It may discuss
#3 publish books for every reader.\n\nUS edition Book ISBN
#4 About the Author\n\Books by Duncan Lay\n\Copyright\n\n# Map
#5 the electronic edition copyright © 2010 by Rosetta Books LLCvn
</details>

Pythia-6.9B Rosetta Neuron (L1/U13378) 

<table><tr><td>#1</td><td>The 912 census which had shown that Christians were 54% of</td></tr><tr><td>#2</td><td>Director of Topf and Sons from 1933 until 1945. \n\n**HARTMU</td></tr><tr><td>#3</td><td>at AEG. In 1933 Hitler came to power in Germany and wider</td></tr><tr><td>#4</td><td>in the early 1920s and there he remained until 1933</td></tr><tr><td>#5</td><td>saxophonist\n L. Perry Curtis [born 1932], American historian</td></tr></table>

Pythia-6.9B Rosetta Neuron (L19/U415)   
![](images/b16f07c22399780b21098c5b35da2aa95bafdccbc41e288a832c206bd3fc2ddd.jpg)

<details>
<summary>text_image</summary>

#1 underlying chronic liver disease must be considered in any
#2 chronic respiratory tract infections even among children.
#3 and obesity-related diseases such as diabetes and cardiovascular
#4 was diagnosed with a rare vascular cancer nearly three years
#5 had been diagnosed at the time with stomach cancer, and she'd
</details>

Pythia-6.9B Rosetta Neuron (L3/U6729)   
![](images/6c1ab6ca5cf60a7739e6b1145d67e4f1965dee6ad34214f8fea67a25a760a2ee.jpg)

<details>
<summary>text_image</summary>

#1
units.\n\nA source in Jinan [漢濟南] said that so far it
#2
time, relocated to Shan Prefecture [漢州，in modern Sanmenxia,
#3
Lu Yi's grandfather Lu Shide [漢師師國南] was an imperial
#4
units.\n\nA source in Jinan [漢齊南] said that so far it
#5
just about to do something. "\nDoes something like 民帝城"
</details>

Figure 19: Top-5 activating sequences for Pythia-6.9B Rosetta Neurons. Rosetta Neurons demonstrate selective firing for coherent concepts.

Pythia-6.9B Non-Rosetta Neuron (L1/U1731)   
![](images/ba87d0a989c9d9bb590c642bdab14ce7faacd122b7bc3f5a3b011b9ab26bb5d8.jpg)

<details>
<summary>text_image</summary>

#1 All her secrets and all her little." "She wouldn't tell me anything.
#2 right, wrap it up." "Friends." "I was thinking as we
#3 Jesus." "She go back to sleep?" "Ate like a pig." "I
#4 thinner, it's lighter and features much more survivors than ever
#5 I've become a horrid woman." "I can't talk now
</details>

Pythia-6.9B Non-Rosetta Neuron (L19/U4564)   
![](images/2479c3302de87b1b5dd658726e593090600b3e0d7c16a451ad63a2621b14e4d8.jpg)

<details>
<summary>text_image</summary>

#1 us informed of the imnster's innermost thoughts as well as develop
#2 this is a nature lovely dream! With 2 bedrooms and 2 baths
#3 destruction for people to get down to foods backs, but it seems
#4 reflect Fed's message of stronger economy WASHINGTON [AP] – The
#5 to the Mets.\nInTRAINER\ ROOM\nMets. 38 David
</details>

Pythia-6.9B Non-Rosetta Neuron (L1/U2353)   
![](images/d6c46049abcbd37f31c96b17806fdbd5b34ed2d84259348763e01652ee3ae23e.jpg)

<details>
<summary>text_image</summary>

#1 lit a cigar and began to smoke, letting the snow drag idly from
#2 carried her out in the rain, covered in black blisters. * His
#3 reading the newspaper. The day was Sunday; the paper was a day old
#4 Yes, he's living out of cardboard boxes. * 'Well, then, you're
#5 Charleton removed his glasses and slowly cleaned them. Then once
</details>

Pythia-6.9B Non-Rosetta Neuron (L3/U435)   
![](images/661409d65dcf6c59f58781e6c792e65e774ad78045e66d7d429d32abf44edab2.jpg)

<details>
<summary>text_image</summary>

#1 positively proportional to the RC time constant. In other words, the
#2 hr, die ich natürlich unterstige wichtige Teile der Verord
#3 image obtained by a linear image sensor, thus inputting a two
#4 alle sollten sie in dieser Arbeit unterstigen damit die richtigen
#5 its front edge to its back edge. In other words, a doubly folded
</details>

Figure 20: Top-5 activating sequences for Pythia-6.9B non-Rosetta neurons. Neurons are randomly selected from the same layers as those in Figure 19. These neurons appear polysemantic, firing for a variety of unrelated concepts.

OpenCLIP ViT-L Rosetta Neuron (L14/U1675)   
![](images/5d6fdf583f8c5a132396ac3f0c05b0df50a9bfe0cb72bc733bf23e46721fe34e.jpg)

<details>
<summary>natural_image</summary>

Six photos of a dog on a gravel surface, showing varied poses and expressions (no text or symbols visible)
</details>

OpenCLIP ViT-L Rosetta Neuron (L8/U1585)   
![](images/f350ecce6670b90d5a8a92a8805c5895122717942379fc90bc38b009092ad8d6.jpg)

<details>
<summary>natural_image</summary>

Six-panel collage showing hands holding mobile phones, a digital phone with 1:30 display, and a decorated bottle with face design (no visible text or symbols)
</details>

OpenCLIP ViT-L Rosetta Neuron (L0/U617)   
![](images/999c94c63a26a46405af443a5af7a469d13dd17ba72cb75177b49d302d8da1c0.jpg)

<details>
<summary>natural_image</summary>

Six-panel collage showing industrial equipment and control panels, including a filing cabinet, camera, and electronic device (no readable text or symbols)
</details>

OpenCLIP ViT-L Rosetta Neuron (L19/U17)   
![](images/ba6c376a02f9cc9c8cdbb78f1f0646822ed214e691f5826d6d948a5f434d6135.jpg)

<details>
<summary>natural_image</summary>

Six-panel collage showing broccoli and a gift box with cartoon character (no text or symbols visible)
</details>

Figure 21: Top-5 activating images for OpenCLIP ViT-L/14 Rosetta Neurons. Rosetta Neurons demonstrate selective firing for coherent concepts.

OpenCLIP ViT-L Non-Rosetta Neuron (L14/U2181)   
![](images/9212d563deff4fc16d0cb7b72bb7ac6f378bde51393c94cb003f6e717b597de5.jpg)

![](images/e45f3446dc714dfc6903c3f373eac6da49f2ce0b5b7d4c3fa10d7113e8e86187.jpg)

![](images/f40f0eaa0a53acab52cce9ea176cb6c006e36cfd4ce3b5ce7c2b8f9736c1c94e.jpg)

![](images/aa41990dcb7f602c6b0e8b34db994807ba7068f3d355bffe804a3076a3db6f99.jpg)

![](images/079abe8a7d449e640c53c153aecb07ead951cb6b978d7a23d88af33f5329b99b.jpg)

![](images/b79d568c1414d5716febe25ba9f5dfea334351e019c71b608cbf8f9be59aa563.jpg)

OpenCLIP ViT-L Non-Rosetta Neuron (L8/U232)   
![](images/5fea0cc442719b50906506c6d513f70cfef921d70e604dcaab1e65d714d58dd7.jpg)

![](images/e4d2cfddb492140c30d25028c8477b286484832b44cc68465b4ad86da0a0622d.jpg)

![](images/5482f304da77f2718a31582b7d92f6b2cd059f3773eb5c9a1eccdc00306fbf17.jpg)

![](images/6801f638fa001c341f80d00427e27cd5d9fb6bddc28dff1906c634f3826399fe.jpg)

![](images/372cee84e1c7ed942bd97176880507bd5a955632c9c45d96b97f2802a2ac0be5.jpg)

![](images/da9f8d6ff70d938d0e78f3a43da37944c7c0f4a9d485139dc0e13a389a2a50d4.jpg)

OpenCLIP ViT-L Non-Rosetta Neuron (L0/U2089)   
![](images/9de2cbc66735df3ea821a2af9cec2324d213a78c1325012277eeba9f488f82ed.jpg)

![](images/bb3fe29311303971c951fb827c9c743028fb81a83dd45f0abf4f1785f55c716e.jpg)

![](images/fdcb6808b6d6557d9ba850d27f9f247bfc25ddc8d44252e6a86483b8bff6ba9e.jpg)

![](images/445bde02446b3a46a5ef93c421f03974d9a61596f8020325dc5faf151259bb07.jpg)

![](images/daa06c3e1c30362dccf4ea5c425313929780305996d76acfca8db1404d724025.jpg)

![](images/d1a1dcf8ad5d79322a8723d6942c53a81562423b95af654974790b71ee6bd477.jpg)

OpenCLIP ViT-L Non-Rosetta Neuron (L19/U0837)   
![](images/b13fe230b83de09af20a2ef534aff1b05755b9dd44e2fba5986842bbfaa74d5a.jpg)

![](images/57c92db9bb9171e725636bcf6e787cc5fec905c425c7613be11451d8244553a5.jpg)

![](images/866e7e6929a39ea33b4a98dc4aaca4d1287fb08ea65cec66e2318da4fcd26cd1.jpg)

![](images/607ae18c9fb2b7aa348faaf4c008e9165bf16e8657312e208246c3fab0017f20.jpg)

![](images/67ea0dc54183d03b09807973d3577c840a1a1d4e54692295f8bf930567a4bc72.jpg)

![](images/471a473aa3c4cf164cccfe00adb0b47e3be86db08bd035d12054a84badd053e4.jpg)  
Figure 22: Top-5 activating images for OpenCLIP ViT-L/14 non-Rosetta neurons. Neurons are randomly selected from the same layers as those in Figure 21. These neurons appear polysemantic, firing for a variety of unrelated concepts.

# B Aligning Token-Wise Activations Across Models

In this section, we describe how activations are mapped to a shared set of aligned positions. This enables direct comparison of neuron activations across models, even when they use different tokenizers or patch grids.

Language models. The same text may be tokenized differently across language models. To obtain a tokenizer-independent alignment, we represent the input text in $\mathrm { U T F } { \cdot } \bar { 8 }$ byte space. For a text sequence $x ,$ let the tokenizers for models A and B split this into tokens $a _ { 1 } , \dots , a _ { T _ { A } }$ and $b _ { 1 } , \dots , b _ { T _ { B } } ,$ respectively. Each token is associated with a contiguous byte span in the original text. We then define a canonical sequence of byte spans by taking the shared byte boundaries induced by the two tokenizations. These are represented by the dotted red lines in Figure 23. This procedure results in aligned text positions $t = 1 , \ldots , T ^ { * }$ that are independent of either tokenizer. For each canonical span t, we mean pool the activations of any tokens overlapping that span, yielding one activation value $m _ { t } ^ { u } ( x )$ per neuron u and aligned position t. These pooled activations are then used for comparison according to the neuron similarity metric defined in Equation (2).

Vision models. Vision Transformers may use different patch sizes or input resolutions, producing activations on different spatial grids. For an image x, let model A produce patch-token activations on a grid of size $H _ { A } \times W _ { A }$ , and model B produce activations on a grid of size $H _ { B } \times W _ { B }$ . We choose one of these grids as a canonical grid of size $H ^ { * } \times W ^ { * }$ . If a model’s native grid differs from the canonical grid, we reshape its patch-token activations into a spatial feature map and resample that map to resolution $H ^ { * } \times W ^ { * }$ using bilinear interpolation as illustrated in Figure 24. This yields an aligned activation map $m _ { p , q } ^ { u } ( x )$ for each neuron u, defined on the same spatial cells $( p , q )$ . We then flatten the canonical grid into a sequence of aligned spatial positions $t = 1 , \dots , H ^ { * } W ^ { * }$ , and use the resulting activations for neuron comparison according to Equation (2). In practice, we remove any non-spatial prefix tokens such as class or register tokens, and then bilinearly interpolate the activation maps to have the same spatial dimensions according to the maximum of the two map sizes.

![](images/d9933cfde7a5a34e5ae89c43b2d974714212f6f35698a7d56b18175dbc9fb3f7.jpg)

<details>
<summary>other</summary>

| Input Text | Sadio | Mane's | goal | settled | this | season's | first | Merseyside | derby |
|---|---|---|---|---|---|---|---|---|---|
| Model A Tokens | Sad io | Mane's | goal | settled | this | season 's | first | Merseyside | derby |
| Model B Tokens | Sad io | Mane's | goal | settled | this | season 's | first | Merseyside | derby |
| Aligned Spans | Sad io | Mane's | goal | settled | this | season 's | first | Merseyside | derby |
</details>

Figure 23: Aligning text tokens via shared byte boundaries. We align the tokens from Model A and Model B by keeping only the byte boundaries that both tokenizers share (red dashed lines). By finding the tokens that live in these new Aligned Spans, we can pool the activations on these tokens, creating a shared set of positions that we can then use to compare neuron responses.

![](images/21f64c475c11570ea9d2d2ba811d1aacaf6cf6505e92971be5ae1f1f0b31b7de.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Image"] --> B["Patch Activations"]
    B --> C["Aligned Activations"]
    C --> D["Shared Canonical Grid (H* × W*)"]
    
    subgraph Input Image
        A
        B
    end
    
    subgraph Patch Activations
        C
        D
    end
    
    style A fill:#f9f,stroke:#333
    style B fill:#bbf,stroke:#333
    style C fill:#bfb,stroke:#333
    style D fill:#ffb,stroke:#333
```
</details>

Figure 24: Aligning spatial grids for neuron comparison. We align the different patch grids from Model A and Model B by choosing a single target resolution (the canonical grid defined by Model B in this case). Any mismatched native grids are resampled using bilinear interpolation to fit this shared grid. This results in aligned activation maps that can be used for measuring neuron similarity.

# C Further Details on Rosetta-Neuron Scaling

We provide additional details on the model families used in the Rosetta Neuron scaling experiments from Section 4. We then present robustness checks for the matching procedure, including an ablation of the mutual-k nearest-neighbor criterion and a dataset-permutation null.

# C.1 Model Families

Language Model Families. For language models, we only consider pretrained models that have not undergone post-training. We conduct the neuron matching using models from the Pythia, GPT-2, OPT, and Qwen-2.5 model families, spanning roughly 100 million to 30 billion parameters (Radford et al., 2019; Biderman et al., 2023; Zhang et al., 2022; Yang et al., 2024). All models are included in Table 2.

Vision Model Families. We analyze discriminative vision models from the OpenCLIP, DINOv2, and Pixio families, spanning scales from approximately 80 million to 5 billion parameters (Cherti et al., 2023; Radford et al., 2021; Oquab et al., 2024; Yang et al., 2025). For the generative model, we use one-step diffusion models built on the Diffusion Transformer architecture (Peebles & Xie, 2023). We use models from the Pixel Mean Flow (pMF), Sana, and Flux families for diffusion (Lu et al., 2026; Chen et al., 2025; Black Forest Labs, 2025). All models are included in Table 3.

<table><tr><td>Run #</td><td>Models</td><td>Run #</td><td>Models</td></tr><tr><td>1</td><td>OPT-1.3B, Qwen2.5-1.5B</td><td>12</td><td>Pythia-160M, GPT2-124M</td></tr><tr><td>2</td><td>OPT-2.7B, Qwen2.5-3B</td><td>13</td><td>Pythia-410M, GPT2-355M</td></tr><tr><td>3</td><td>OPT-6.7B, Qwen2.5-7B</td><td>14</td><td>Pythia-1.4B, GPT2-1.5B</td></tr><tr><td>4</td><td>OPT-13B, Qwen2.5-14B</td><td>15</td><td>OPT-125M, Pythia-160M, GPT2-124M</td></tr><tr><td>5</td><td>OPT-30B, Qwen2.5-32B</td><td>16</td><td>OPT-350M, Pythia-410M, GPT2-355M</td></tr><tr><td>6</td><td>Pythia-160M, OPT-125M</td><td>17</td><td>OPT-1.3B, Pythia-1.4B, GPT2-1.5B</td></tr><tr><td>7</td><td>Pythia-410M, OPT-350M</td><td>18</td><td>Pythia-1.4B, OPT-1.3B, Qwen2.5-1.5B</td></tr><tr><td>8</td><td>Pythia-1.4B, OPT-1.3B</td><td>19</td><td>Pythia-2.8B, OPT-2.7B, Qwen2.5-3B</td></tr><tr><td>9</td><td>Pythia-2.8B, OPT-2.7B</td><td>20</td><td>Pythia-6.9B, OPT-6.7B, Qwen2.5-7B</td></tr><tr><td>10</td><td>Pythia-6.9B, OPT-6.7B</td><td>21</td><td>Pythia-12B, OPT-13B, Qwen2.5-14B</td></tr><tr><td>11</td><td>Pythia-12B, OPT-13B</td><td></td><td></td></tr></table>

Table 2: Model combinations used in the Rosetta Neuron scaling laws in Figure 3a.

<table><tr><td>Run #</td><td>Models</td><td>Run #</td><td>Models</td></tr><tr><td>1</td><td>pMF DiT-B/16, DINOv2 ViT-B/14</td><td>12</td><td>Sana DiT-1.6B, OpenCLIP ViT-bigG/14</td></tr><tr><td>2</td><td>pMF DiT-L/16, DINOv2 ViT-L/14</td><td>13</td><td>Flux.2-Klein-DiT-4B, OpenCLIP EVA-02-CLIP-E/14</td></tr><tr><td>3</td><td>pMF DiT-H/16, DINOv2 ViT-g/14</td><td>14</td><td>pMF DiT-B/16, Pixio ViT-B/16, OpenCLIP ViT-B/16</td></tr><tr><td>4</td><td>pMF DiT-B/16, Pixio ViT-B/16</td><td>15</td><td>pMF DiT-L/16, Pixio ViT-L/16, OpenCLIP ViT-L/14</td></tr><tr><td>5</td><td>pMF DiT-L/16, Pixio ViT-L/16</td><td>16</td><td>pMF DiT-H/16, Pixio ViT-H/16, OpenCLIP ViT-H/14</td></tr><tr><td>6</td><td>pMF DiT-H/16, Pixio ViT-H/16</td><td>17</td><td>Sana DiT-1.6B, Pixio ViT-1B/16, OpenCLIP ViT-bigG/14</td></tr><tr><td>7</td><td>Sana DiT-1.6B, Pixio ViT-1B/16</td><td>18</td><td>Flux.2-DiT-4B, Pixio ViT-5B/16, OpenCLIP EVA-02-E/14</td></tr><tr><td>8</td><td>Flux.2-Klein-DiT-4B, Pixio ViT-5B/16</td><td>19</td><td>pMF DiT-B/16, DINOv2 ViT-B/14, OpenCLIP ViT-B/16</td></tr><tr><td>9</td><td>pMF DiT-B/16, OpenCLIP ViT-B/16</td><td>20</td><td>pMF DiT-L/16, DINOv2 ViT-L/14, OpenCLIP ViT-L/14</td></tr><tr><td>10</td><td>pMF DiT-L/16, OpenCLIP ViT-L/14</td><td>21</td><td>pMF DiT-H/16, DINOv2 ViT-g/14, OpenCLIP ViT-H/14</td></tr><tr><td>11</td><td>pMF DiT-H/16, OpenCLIP ViT-H/14</td><td></td><td></td></tr></table>

Table 3: Model combinations used in the Rosetta Neuron scaling laws in Figure 3b.

![](images/5030d3fe21015e9047bfcdd31fc1cd23edf808be455f3c5726d241daf3805862.jpg)

![](images/56bbf0cac082caaca830b3fde2204845621467d4e4da3202dbfd8d5de32fe952.jpg)  
Figure 25: Robustness to the mutual top-k matching criterion. We repeat the scaling analysis for Pythia–OPT and Diffusion–OpenCLIP for different values of k in the nearest neighbor criterion. Increasing k results in more discovered neuron pairs, but the fitted power-law exponents remain within a narrow sublinear range.

# C.2 Robustness to the Mutual Top-k Criterion

Our main scaling experiments in Section 4 identify Rosetta Neurons using mutual top-1 nearestneighbor matching. To test whether the observed scaling behavior depends on this particular choice, we repeat the scaling analysis while varying the mutual top-k parameter on one representative scaling trajectory per modality: Pythia–OPT for language models and Diffusion–OpenCLIP for vision models. As seen in Figure 25, increasing k makes the matching rule more permissive and increases the absolute number of discovered neuron pairs. However, across all tested values of k, the fitted power-law exponents remain within a narrow sublinear range in both modalities. This suggests that the Rosetta Neuron scaling trend is not an artifact of the default top-k matching rule. It is also consistent with the detectability-threshold view in our analytical model (Section D): varying the permissiveness of the matching criterion changes which features are counted as detectable and changes the prefactor of the scaling law, but should not change the underlying scaling exponent.

# C.3 Input-Permutation Null

In Section 4.2, we applied the neuron matching pipeline to untrained random networks of different sizes to test whether Rosetta Neuron scaling laws could arise from the matching procedure. That baseline suggests the trend is not simply due to architecture, initialization, or the larger number of candidate neurons in bigger networks. Here, we test whether the observed scaling could still arise from activation marginals or high-dimensional nearest-neighbor effects even when input alignment is corrupted.

Power-law scaling is absent under dataset permutation. For each model pair, we first compute tokenlevel activations on the same dataset used in the main experiments. Before computing cross-model correlations, we randomly permute the flattened activation positions for one model so that correlations are computed between mismatched input positions across the two models. This preserves each neuron’s marginal activation distribution while corrupting input-wise alignment. We then apply the same mutual nearestneighbor matching procedure with k = 1 as in the main experiments. We run this null over three random permutations at each model scale and report means with 95% confidence intervals in Figure 26.

Under this null, the number of discovered Rosetta Neurons collapses to roughly 20–100 matches. Moreover, these counts no longer exhibit the systematic sublinear power-law trend observed with aligned activations. This suggests that Rosetta Neuron scaling depends on shared responses to the aligned inputs, rather than being induced by the matching procedure or activation statistics alone.

![](images/97fb6969c3ba72283d4cbdb74f5f74d98d0f77ccc9392d290563485e1bb42c45.jpg)

<details>
<summary>line</summary>

| Model Size (# Neurons) | Language | Vision |
| ---------------------- | -------- | ------ |
| 5 × 10⁴                | 55       | 25     |
| 1 × 10⁵                | 45       | 35     |
| 5 × 10⁵                | 60       | 20     |
| 10 × 10⁵               | 85       | 25     |
</details>

Figure 26: Rosetta Neuron counts under input permutation lack systematic scaling.

# D Detailed Derivation of the Rosetta Neuron Scaling Model

This appendix gives the formal version of the analytical model for Rosetta Neuron scaling in Section 4.3. The model provides an explanation for the empirical observation that the number of Rosetta Neurons grows as a sublinear power law with scale. It further predicts that scaling separates an increasingly selective Rosetta population from a more crowded non-Rosetta background, a phenomenon we call the Neuron Polarization Effect. We organize this section into four conceptual parts. We first state and solve a minimal allocation problem over latent features. Second, we motivate the assumptions of this allocation problem by interpreting scalar neuron readouts as Gaussian information channels in a superposition model. Third, we connect the idealized Rosetta Neuron detectability criterion to our empirical cross-model neuron matching. Finally, we use simulations to verify that the theory reproduces the predicted scaling and polarization trends when paired with our empirical matching procedure.

# D.1 A Minimal Capacity-Allocation Model

We begin with a minimal capacity-allocation model that abstracts away architectural and optimization details while capturing the central constraint: a growing set of latent features must share a finite budget of clean neuron capacity.

Feature spectrum. Let N denote the number of MLP neuron coordinates under consideration.2 Let A(N ) be the number of active latent features representing patterns from a data distribution D. A network of size N allocates its neuron capacity across this feature set. We index these features by decreasing importance:

$$
r \in \{1, 2, \dots , A (N) \}.
$$

Since modern scaling regimes increase both model size and effective data complexity (Hoffmann et al., 2022), we assume that the feature set does not saturate with model size. Specifically, we take

$$
A (N) = \Omega (N), \quad A (N) > N \tag {8}
$$

for sufficiently large N . Thus the model remains in a superposition regime asymptotically (Liu et al., 2025): there are more latent features than neuron coordinates. Following prior spectral scaling theories (Michaud et al., 2023; Bordelon et al., 2020; Bahri et al., 2024), we assume a power-law feature-importance spectrum for these features:

$$
w _ {r} = C r ^ {- \beta}, \quad \beta > 1, \quad C > 0. \tag {9}
$$

Here, wr measures the importance, or downstream value of making feature r cleanly readable, reflecting factors such as frequency, predictiveness, or task relevance.

Feature isolation and Rosetta detectability. Let $s _ { r } \geq 0$ measure how cleanly feature r is isolated in its most dedicated neuron coordinate. Larger values of $s _ { r }$ indicate that this coordinate is dominated by feature r and is more monosemantic, rather than reflecting a mixture of unrelated features. At this stage, sr is an abstract isolation score; in Section D.4, we derive sr as a signal-to-interference ratio in a linear superposition model.

We assume that independently trained models of similar scale and modality are trained on comparable data distributions D, and therefore share approximately the same coarse latent feature spectrum induced by D. Under this shared-spectrum assumption, we analyze a representative network in isolation: features that are cleanly isolated in one model are expected to be similarly isolated in another matched-scale model, while model-specific interference averages out in cross-model correlations over aligned inputs. We make this connection explicit in Section D.6. We therefore model Rosetta detectability by thresholding the isolation score. For a fixed detectability threshold τ > 1, we define the idealized Rosetta count as

$$
R _ {\tau} (N) = \sum_ {r = 1} ^ {A (N)} \mathbf {1} \{s _ {r} \geq \tau \}. \tag {10}
$$

The thresholded features are those cleanly isolated enough to be reproducible and detectable across independently trained models. For $\tau > 1$ , this feature count corresponds to distinct dedicated neuron coordinates in the superposition model introduced in Section D.4, so we interpret $R _ { \tau } ( N )$ as an idealized Rosetta Neuron count.

Utility and budget. We now specify the allocation rule used in our analytical model. We state the allocation objective and constraint in Equation (11), and derive them in Section D.5 from a simple superposition picture: neuron coordinates mix many latent features, and each coordinate provides a noisy scalar readout of any one feature. Under this view, the logarithmic utility $w _ { r } \log ( 1 + s _ { r } )$ comes from the importance-weighted information gain from improving a noisy readout; the linear budget $\begin{array} { r } { \sum _ { r = 1 } ^ { A ( N ) } s _ { r } \le \kappa N } \end{array}$ , for a constant $\kappa > 0$ , comes from the bounded total variance of an N-coordinate activation vector. These ingredients give the following capacity-allocation problem:

$$
\max _ {s _ {r} \geq 0} \sum_ {r = 1} ^ {A (N)} w _ {r} \log (1 + s _ {r}) \quad \text { subject   to } \quad \sum_ {r = 1} ^ {A (N)} s _ {r} \leq \kappa N. \tag {11}
$$

The objective captures a coverage–purity tradeoff. Because the utility is logarithmic, increasing the isolation of a noisy feature initially gives substantial gain, but further purifying an already clean feature gives diminishing returns. The optimum therefore allocates high isolation to the most important features while also extending partial isolation into the lower-importance tail.

# D.2 Solving the Allocation Problem

Continuum approximation. To analyze the asymptotic scaling behavior, we approximate the ranked feature spectrum by a continuous variable $\dot { r } \in \left[ 1 , A ( N ) \right]$ ]. The continuum version of Eq. (11) yields the following variational problem:

$$
\max _ {s (r) \geq 0} \int_ {1} ^ {A (N)} C r ^ {- \beta} \log (1 + s (r)) d r \quad \text { subject   to } \quad \int_ {1} ^ {A (N)} s (r) d r \leq \kappa N. \tag {12}
$$

The corresponding effective Rosetta Neuron count is

$$
R _ {\tau} (N) = \int_ {1} ^ {A (N)} \mathbf {1} \{s (r; N) \geq \tau \} d r. \tag {13}
$$

Proposition 1: Sublinear Rosetta Neuron scaling. Assume $\beta > 1$ and $A ( N ) = \Omega ( N )$ , with $A ( \bar { N } ) > N$ for sufficiently large N . Then the solution to the allocation problem in Eq. (12) is

$$
s ^ {\star} (r; N) = \left[ \left(\frac {r _ {0} (N)}{r}\right) ^ {\beta} - 1 \right] _ {+}, \quad [ u ] _ {+} := \max \{u, 0 \}, \tag {14}
$$

where $r _ { 0 } ( N ) = \Theta ( N ^ { 1 / \beta } )$ is the largest feature rank receiving positive isolation. For every fixed threshold $\tau > 1$ , the effective Rosetta Neuron count satisfies

$$
R _ {\tau} (N) = \Theta \left(N ^ {1 / \beta}\right). \tag {15}
$$

Since $\beta > 1$ , the Rosetta Neuron count grows sublinearly.

Proof. Introduce a Lagrange multiplier λ for the budget constraint. For any feature rank with positive isolation, $s ( r ) > 0$ , the first-order optimality condition gives

$$
\frac {C r ^ {- \beta}}{1 + s (r)} = \lambda . \tag {16}
$$

Solving for $s ( r )$ gives $s ( r ) = C r ^ { - \beta } / \lambda - 1$ . Let $r _ { 0 } ( N )$ denote the boundary of the positive-support region, so that $s ( r _ { 0 } ( N ) ) = 0$ . Then $\lambda = C r _ { 0 } ( N ) ^ { - \beta }$ , and therefore

$$
s ^ {\star} (r; N) = \left[ \left(\frac {r _ {0} (N)}{r}\right) ^ {\beta} - 1 \right] _ {+}. \tag {17}
$$

Because log $( 1 + s ( r ) )$ is strictly increasing, the optimum uses the full available isolation budget. Substituting the solution $s ^ { \star } ( r ; N )$ into the budget constraint gives

$$
\kappa N = \int_ {1} ^ {r _ {0} (N)} \left[ \left(\frac {r _ {0} (N)}{r}\right) ^ {\beta} - 1 \right] d r \tag {18}
$$

$$
= r _ {0} (N) ^ {\beta} \int_ {1} ^ {r _ {0} (N)} r ^ {- \beta} d r - \left(r _ {0} (N) - 1\right) \tag {19}
$$

$$
= \frac {r _ {0} (N) ^ {\beta}}{\beta - 1} - \frac {\beta r _ {0} (N)}{\beta - 1} + 1 = \Theta \left(r _ {0} (N) ^ {\beta}\right). \tag {20}
$$

Hence $r _ { 0 } ( N ) = \Theta ( N ^ { 1 / \beta } )$ . Since $\beta > 1$ , this cutoff grows sublinearly in N , while $A ( N ) = \Omega ( N )$ grows at least linearly. Therefore $r _ { 0 } ( N ) = o ( A ( N ) )$ , so for sufficiently large $N .$ , the positively isolated features form an initial segment $1 \le r \le r _ { 0 } ( \dot { N } )$ within the full feature range $[ 1 , A ( N ) ]$ .

Let $r _ { \tau } ( N )$ denote the Rosetta frontier, defined by $s ^ { \star } ( r _ { \tau } ( N ) ; N ) = \tau$ . Using the optimal allocation profile, this frontier satisfies

$$
\left(\frac {r _ {0} (N)}{r _ {\tau} (N)}\right) ^ {\beta} - 1 = \tau , \tag {21}
$$

and therefore

$$
r _ {\tau} (N) = r _ {0} (N) (1 + \tau) ^ {- 1 / \beta} = \Theta (N ^ {1 / \beta}). \tag {22}
$$

Thus, counting the features above the Rosetta threshold amounts to counting feature ranks up to order $r _ { \tau } ( N )$ , so

$$
R _ {\tau} (N) = \int_ {1} ^ {A (N)} \mathbf {1} \{s ^ {\star} (r; N) \geq \tau \} d r = \Theta (r _ {\tau} (N)) = \Theta (N ^ {1 / \beta}). \tag {23}
$$

Since $\beta > 1$ , this growth is sublinear.

![](images/64fbf4fd91044448ec820d5e290d412ef5ec65c348abcd30e22b23ca82259b9f.jpg)

The optimal allocation partitions the spectrum into Rosetta-detectable features with $s _ { r } \geq \tau$ , partially isolated features with $0 < s _ { r } < \tau .$ , strongly superposed features with $s _ { r } = 0$ , and features beyond the represented set $A ( N )$ . The frontiers $r _ { \tau } ( N )$ and $r _ { 0 } ( N )$ scale as $\Theta ( N ^ { 1 / \beta } )$ , yielding the sublinear Rosetta Neuron count $\dot { R _ { \tau } } ( N ) = \Theta ( N ^ { 1 / \beta } )$ . This is illustrated in Figure 5.

# D.3 Prediction: Neuron Polarization

The same allocation profile predicts a polarization effect. As N grows, the number of detectable Rosetta Neurons increases, and their average isolation also increases. At the same time, a growing tail of latent features remains weakly isolated, corresponding to a more crowded non-Rosetta background.

Rosetta purification. Consider any fixed top-ranked feature $r _ { \star } ~ ( \mathrm { i } . \mathrm { e } . , \mathrm { a }$ feature with small rank index and high importance). Its isolation in the dedicated neuron coordinate scales as

$$
s ^ {\star} (r _ {\star}; N) = \left(\frac {r _ {0} (N)}{r _ {\star}}\right) ^ {\beta} - 1 = \Theta \left(\frac {N}{r _ {\star} ^ {\beta}}\right) - 1 \rightarrow \infty \quad \text { as } N \rightarrow \infty . \tag {24}
$$

Thus, for any fixed high-importance feature, its isolation grows on the order of N , so its dedicated neuron coordinate becomes increasingly dominated by that feature. Intuitively, this increasing isolation can be interpreted as greater selectivity and monosemanticity.

The Rosetta-detectable features are those with ranks $1 \leq r \leq r _ { \tau } ( N )$ . We therefore define their average isolation as

$$
\bar {s} _ {\text { Rosetta }} (N) := \frac {1}{r _ {\tau} (N)} \int_ {1} ^ {r _ {\tau} (N)} \left[ \left(\frac {r _ {0} (N)}{r}\right) ^ {\beta} - 1 \right] d r \tag {25}
$$

$$
= \Theta \left(\frac {r _ {0} (N) ^ {\beta}}{r _ {\tau} (N)}\right) = \Theta \left(r _ {0} (N) ^ {\beta - 1}\right) = \Theta \left(N ^ {(\beta - 1) / \beta}\right), \tag {26}
$$

where we used $r _ { \tau } ( N ) = r _ { 0 } ( N ) ( 1 + \tau ) ^ { - 1 / \beta }$ . Since $\beta > 1$ , this average isolation increases with scale, corresponding to a Rosetta population that becomes more selective and monosemantic on average.

Non-Rosetta crowding. We now analyze the isolation trend in the non-Rosetta feature tail. For Rosetta-detectable features, the threshold $\tau >$ 1 lets us associate each feature with a distinct dedicated neuron coordinate. Below the threshold, this one-to-one interpretation no longer applies: non-Rosetta features may instead be distributed across multiple shared, superposed coordinates. So, we first analyze crowding on the feature side by measuring the average clean isolation assigned to features outside the Rosetta set:

$$
\bar {s} _ {\text { tail }} (N) := \frac {1}{A (N) - R _ {\tau} (N)} \int_ {r _ {\tau} (N)} ^ {A (N)} s ^ {\star} (r; N) d r. \tag {27}
$$

Only the partially isolated non-Rosetta region $r \in [ r _ { \tau } ( N ) , r _ { 0 } ( N ) ]$ contributes to the integral, since features beyond $r _ { 0 } ( N )$ have zero isolation. Thus,

$$
\int_ {r _ {\tau} (N)} ^ {A (N)} s ^ {\star} (r; N) d r = \int_ {r _ {\tau} (N)} ^ {r _ {0} (N)} \left[ \left(\frac {r _ {0} (N)}{r}\right) ^ {\beta} - 1 \right] d r \tag {28}
$$

$$
= r _ {0} (N) \int_ {(1 + \tau) ^ {- 1 / \beta}} ^ {1} \left(u ^ {- \beta} - 1\right) d u \tag {29}
$$

$$
= \Theta (r _ {0} (N)) = \Theta (N ^ {1 / \beta}), \tag {30}
$$

using the change of variables $u = r / r _ { 0 } ( N )$ . Since $A ( N ) = \Omega ( N ) , R _ { \tau } ( N ) = \Theta ( N ^ { 1 / \beta } )$ , and $\beta > 1$ , the measure of the non-Rosetta feature tail, $A ( N ) - R _ { \tau } ( N )$ , grows asymptotically faster than $N ^ { 1 / \beta }$ . Therefore,

$$
\bar {s} _ {\text { tail }} (N) = O \left(\frac {N ^ {1 / \beta}}{A (N) - R _ {\tau} (N)}\right) = O \left(\frac {N ^ {1 / \beta}}{A (N)}\right)\rightarrow 0. \tag {31}
$$

Equation (31) shows feature-side crowding: the average clean isolation assigned to an unresolved tail feature vanishes. We can also translate this into a neuron-level interpretation. The $R _ { \tau } ( N )$ Rosetta-detectable features correspond to distinct dedicated neurons leaving $N - R _ { \tau } ( N ) = \Theta \dot { (} N )$ non-Rosetta coordinates. The total clean isolation assigned to the unresolved tail is $\Theta ( N ^ { 1 / \beta } )$ , so the average tail-isolation mass per non-Rosetta neuron scales as

$$
O \left(\frac {N ^ {1 / \beta}}{N}\right) = O \left(N ^ {(1 - \beta) / \beta}\right)\rightarrow 0, \tag {32}
$$

since $\beta > 1$ . Thus, on average, non-Rosetta coordinates receive vanishing clean tail-isolation mass. By definition, $A ( N )$ counts the active latent features that the network allocates capacity to represent, so the unresolved tail features are still part of the represented feature population. Since they are not cleanly isolated into dedicated coordinates, they must be represented in a distributed or superposed form. This gives the neuron-level interpretation: the non-Rosetta population forms a crowded, polysemantic background of weakly isolated features.

# D.4 Deriving the Isolation Score from Superposition

The minimal model above treats $s _ { r }$ as an abstract isolation score. We now derive this score from a simple linear superposition model of neuron activations.

Linear feature decomposition. Let x denote an aligned input position, such as a text token or image patch, and let $\mathbf { h } ( \mathbf { \bar { \boldsymbol { x } } } ) \in \mathbb { R } ^ { N }$ be the centered vector of MLP neuron activations at that position, with centering taken over the data distribution $D .$ For each latent feature $r \in \{ 1 , \ldots , A ( \mathbf { \bar { \it { N } } } ) \}$ , let $\tilde { z } _ { r } ( x ) \in \mathbb { R }$ be the centered, input-dependent scalar response of feature r. Let $\mathbf { v } _ { r } \in \mathbb { R } ^ { N }$ describe how that feature response is distributed across neuron coordinates. We model the activation vector as a linear superposition of feature responses,

$$
\mathbf {h} (x) = \sum_ {r = 1} ^ {A (N)} \tilde {z} _ {r} (x) \mathbf {v} _ {r}. \tag {33}
$$

We normalize the feature directions as $\| \mathbf { v } _ { r } \| _ { 2 } ^ { 2 } = 1$ , so $\mathbf { v } _ { r }$ specifies the relative distribution of feature r across coordinates, while the overall strength of the feature response on input x is carried by the scale of $\tilde { z } _ { r } ( x )$ .

For the isolation-score calculation that follows, we first standardize each feature activation over the data distribution. Since $\tilde { z } _ { r } ( x )$ is centered, define

$$
g _ {r} ^ {2} := \mathbb {E} _ {x \sim D} [ \tilde {z} _ {r} (x) ^ {2} ], \quad z _ {r} (x) := \tilde {z} _ {r} (x) / g _ {r}, \tag {34}
$$

for represented features with $g _ { r } > 0$ . Then $z _ { r } ( x )$ has mean zero and unit variance over D, while $g _ { r }$ carries the overall activation scale of feature r. For tractability, we use a diagonal second-moment approximation, keeping only the variances of the standardized feature activations and neglecting cross-feature covariances:

$$
\mathbb {E} _ {x \sim D} [ z _ {r} (x) ] = 0, \quad \mathbb {E} _ {x \sim D} [ z _ {r} (x) ^ {2} ] = 1, \quad \mathbb {E} _ {x \sim D} [ z _ {r} (x) z _ {s} (x) ] = 0 \quad (r \neq s). \tag {35}
$$

The representation can be written as

$$
\mathbf {h} (x) = \sum_ {r = 1} ^ {A (N)} \tilde {z} _ {r} (x) \mathbf {v} _ {r} = \sum_ {r = 1} ^ {A (N)} g _ {r} z _ {r} (x) \mathbf {v} _ {r}. \tag {36}
$$

This normalization will let us define the variance contribution of each feature to each neuron coordinate in the next step.

Define the effective loading of feature r on neuron coordinate j by

$$
W _ {j r} := g _ {r} \mathbf {e} _ {j} ^ {\top} \mathbf {v} _ {r}, \quad a _ {j r} := W _ {j r} ^ {2}. \tag {37}
$$

Note that $W _ { j \imath }$ is an effective feature loading onto neuron coordinates in the superposition model, not a learned MLP weight matrix.

Because $\| \mathbf { v } _ { r } \| _ { 2 } ^ { 2 } = 1$

$$
\sum_ {j = 1} ^ {N} a _ {j r} = g _ {r} ^ {2}. \tag {38}
$$

The activation of coordinate j is

$$
h _ {j} (x) = \sum_ {r = 1} ^ {A (N)} W _ {j r} z _ {r} (x). \tag {39}
$$

Using the decorrelation approximation in Eq. (35),

$$
\mathrm{Var} _ {x} (h _ {j}) = \sum_ {r = 1} ^ {A (N)} W _ {j r} ^ {2} = \sum_ {r = 1} ^ {A (N)} a _ {j r}. \tag {40}
$$

Thus $a _ { j r }$ is the variance contribution of feature r to neuron coordinate j. We next use this to bound the total variance across coordinates. We assume the normalized activation vector has bounded average variance per coordinate:

$$
\sum_ {j = 1} ^ {N} \mathrm{Var} _ {x} (h _ {j}) = \sum_ {j = 1} ^ {N} \sum_ {r = 1} ^ {A (N)} W _ {j r} ^ {2} = \sum_ {r = 1} ^ {A (N)} g _ {r} ^ {2} \leq C _ {h} N, \tag {41}
$$

for a constant $C _ { h }$ independent of N. This says that an N-coordinate normalized activation vector carries $O ( N )$ total variance, consistent with standard scale-preserving initialization and normalization schemes (He et al., 2015; Yang et al., 2021; Ba et al., 2016).

Signal-to-interference isolation score. A single MLP neuron coordinate is a scalar readout of a richer, potentially high-dimensional latent feature state. When using coordinate j to read out feature r, we decompose the coordinate into the desired contribution from feature r and a residual term:

$$
y _ {j} = W _ {j r} z _ {r} + \eta_ {j r}, \tag {42}
$$

where $\eta _ { j r }$ contains all variation in coordinate j not explained by feature r. We model this residual as having two sources: interference from other modeled features that also contribute to coordinate $j ,$ and a fixed finite-resolution noise floor $\sigma _ { \infty } ^ { 2 } > 0$ that captures unmodeled variation. Under the diagonal second-moment approximation above, the effective residual variance is

$$
\nu_ {j r} = \sigma_ {\infty} ^ {2} + \sum_ {k \neq r} W _ {j k} ^ {2}. \tag {43}
$$

We define the signal-to-interference ratio for reading feature r from coordinate j as

$$
s _ {j r} := \frac {W _ {j r} ^ {2}}{\sigma_ {\infty} ^ {2} + \sum_ {k \neq r} W _ {j k} ^ {2}}. \tag {44}
$$

The most dedicated neuron coordinate for feature r is the coordinate with maximal signal-tointerference ratio,

$$
c (r) = \arg \max _ {j} s _ {j r}, \tag {45}
$$

and the feature-level isolation score is

$$
s _ {r} := s _ {c (r), r} = \frac {W _ {c (r) , r} ^ {2}}{\sigma_ {\infty} ^ {2} + \sum_ {k \neq r} W _ {c (r) , k} ^ {2}}. \tag {46}
$$

Large sr means that the most dedicated coordinate for feature r is dominated by that feature and more monosemantic; small $s _ { r }$ means that the feature is read out through a more polysemantic mixture.

This definition also explains why the threshold $\tau > 1$ in Eq. (10) gives a one-to-one feature-tocoordinate count in the idealized model. Suppose two distinct features r and $r ^ { \prime }$ shared the same threshold-passing most dedicated coordinate j. Since $s _ { r } > 1$ ,

$$
W _ {j r} ^ {2} > \sigma_ {\infty} ^ {2} + \sum_ {k \neq r} W _ {j k} ^ {2} \geq W _ {j r ^ {\prime}} ^ {2}.
$$

Similarly, $s _ { r ^ { \prime } } > 1$ implies $W _ { j r ^ { \prime } } ^ { 2 } > W _ { j r } ^ { 2 }$ , a contradiction. Therefore, each threshold-passing feature has its own dedicated neuron coordinate.

# D.5 Deriving the Logarithmic Utility and Linear Isolation Budget

We now derive the logarithmic utility and linear isolation budget used in Eq. (11) based on the signal-to-interference model above.

Logarithmic utility from predictive loss reduction. Recall that $z _ { r }$ has been standardized so that $\mathbb { E } _ { x \sim D } [ z _ { r } ( x ) ] = 0$ and $\mathrm { V a r } _ { x \sim D } ( z _ { r } ( x ) ) = 1$ . For tractability, we use a Gaussian approximation to obtain a closed-form relationship between readout quality and predictive uncertainty. Specifically, we model $z _ { r }$ as Gaussian and treat the residual interference $\eta$ in the most dedicated coordinate as independent Gaussian noise. After choosing the most dedicated coordinate $c ( r )$ , its readout is

$$
y _ {c (r)} = W _ {c (r), r} z _ {r} + \eta_ {c (r), r}. \tag {47}
$$

We divide by the feature coefficient $W _ { c ( r ) , r }$ and obtain the normalized observation model

$$
\tilde {y} _ {r} := \frac {y _ {c (r)}}{W _ {c (r) , r}} = z _ {r} + \epsilon_ {r}, \quad z _ {r} \sim \mathcal {N} (0, 1), \quad \epsilon_ {r} \sim \mathcal {N} (0, 1 / s _ {r}). \tag {48}
$$

For this Gaussian model, the posterior variance is

$$
\operatorname{Var} (z _ {r} \mid \tilde {y} _ {r}) = \frac {1}{1 + s _ {r}}. \tag {49}
$$

Thus the optimal Gaussian negative log-likelihood, or equivalently the conditional entropy, is

$$
H (z _ {r} \mid \tilde {y} _ {r}) = \frac {1}{2} \log \left(2 \pi e \frac {1}{1 + s _ {r}}\right). \tag {50}
$$

Relative to having no informative readout, the reduction in optimal predictive loss is

$$
H (z _ {r}) - H (z _ {r} \mid \tilde {y} _ {r}) = \frac {1}{2} \log (1 + s _ {r}). \tag {51}
$$

We therefore model a feature with downstream importance $w _ { r }$ contributing utility proportional to $w _ { r } \log ( 1 + s _ { r } )$ , with the factor $1 / 2$ absorbed into the overall scale.3

Linear Isolation Budget. From the definition of the feature-level isolation score in Eq. (46),

$$
s _ {r} \leq \frac {W _ {c (r) , r} ^ {2}}{\sigma_ {\infty} ^ {2}}. \tag {52}
$$

Summing over features and using the activation-energy bound in Eq. (41),

$$
\sum_ {r = 1} ^ {A (N)} s _ {r} \leq \sigma_ {\infty} ^ {- 2} \sum_ {r = 1} ^ {A (N)} W _ {c (r), r} ^ {2} \leq \sigma_ {\infty} ^ {- 2} \sum_ {j = 1} ^ {N} \sum_ {r = 1} ^ {A (N)} W _ {j r} ^ {2} \leq \frac {C _ {h}}{\sigma_ {\infty} ^ {2}} N. \tag {53}
$$

Therefore

$$
\sum_ {r = 1} ^ {A (N)} s _ {r} \leq \kappa N, \quad \kappa := \frac {C _ {h}}{\sigma_ {\infty} ^ {2}}. \tag {54}
$$

The linear budget thus follows from normalized activation energy and an irreducible finite-resolution floor for scalar neuron readouts.

# D.6 Connection to Empirical Rosetta Matching

The theory defines Rosetta-detectable features by thresholding the feature-level isolation score $s _ { r }$ . In the experiments, however, Rosetta Neurons are identified by cross-model activation matching rather than by directly observing $s _ { r }$ . In this section, we relate these two notions by showing that, in an idealized setting, the same isolation score that defines Rosetta detectability also controls the cross-model activation correlation used for empirical matching. A high-dimensional randompacking approximation captures the effect of model-specific superposition: independent models mix unresolved tail features in nearly orthogonal ways, so reliable cross-model neuron matches require sufficiently isolated common features.

Superposition as model-specific interference. Consider two independently trained matched-scale models M and $M ^ { \prime } .$ , trained on comparable data distributions D. As in the allocation model, we assume these models share approximately the same coarse latent feature spectrum induced by $D ,$ although their model-specific interference patterns may differ because low-isolation tail features can be packed in many approximately equivalent ways. Let $c _ { M } ( r )$ and $c _ { M ^ { \prime } } ( r )$ denote their most dedicated neuron coordinates for feature $r ,$ with feature-level isolation scores $s _ { r } ^ { M }$ and s M ′r . $s _ { r } ^ { M ^ { \prime } }$

Since empirical matching uses Pearson correlation, arbitrary rescalings of a neuron activation do not affect the match score. For feature $r ,$ the relevant neurons are the most dedicated coordinates $c _ { M } ( r )$ and $c _ { M ^ { \prime } } ( r )$ , whose raw readouts are

$$
y _ {c _ {M} (r)} ^ {M} = W _ {c _ {M} (r), r} ^ {M} z _ {r} + \eta_ {c _ {M} (r), r} ^ {M}, \quad y _ {c _ {M ^ {\prime}} (r)} ^ {M ^ {\prime}} = W _ {c _ {M ^ {\prime}} (r), r} ^ {M ^ {\prime}} z _ {r} + \eta_ {c _ {M ^ {\prime}} (r), r} ^ {M ^ {\prime}}. \tag {55}
$$

We divide each readout by the corresponding feature loading, $W _ { c _ { M } ( r ) , r } ^ { M } \ \mathrm { o r } \ W _ { c _ { M ^ { \prime } } ( r ) , r } ^ { M ^ { \prime } }$ or W M ′ cM′ (r),r, which removes arbitrary activation scale that does not affect Pearson correlation. This leaves normalized readouts consisting of the shared feature signal plus rescaled model-specific interference:

$$
\tilde {y} _ {c _ {M} (r)} ^ {M} = z _ {r} + \epsilon_ {M}, \quad \tilde {y} _ {c _ {M ^ {\prime}} (r)} ^ {M ^ {\prime}} = z _ {r} + \epsilon_ {M ^ {\prime}}. \tag {56}
$$

Here $z _ { r }$ is the shared feature signal, while $\epsilon _ { M }$ and $\epsilon _ { M ^ { \prime } }$ are model-specific interference terms. Recall that the shared feature $z _ { r }$ has unit variance, and that the rescaled interference variances are determined by the feature-level isolation scores:

$$
\mathrm{Var} (z _ {r}) = 1, \qquad \mathrm{Var} (\epsilon_ {M}) = 1 / s _ {r} ^ {M}, \qquad \mathrm{Var} (\epsilon_ {M ^ {\prime}}) = 1 / s _ {r} ^ {M ^ {\prime}}.
$$

Tail leakage and cross-model covariance. We separate the feature-dependent part of the interference by writing it as a mixture over unresolved tail features:

$$
\epsilon_ {M} ^ {\text { feat }} = \sum_ {k \in T _ {r} ^ {M}} b _ {c _ {M} (r), k} ^ {M} z _ {k}, \quad \epsilon_ {M ^ {\prime}} ^ {\text { feat }} = \sum_ {k \in T _ {r} ^ {M ^ {\prime}}} b _ {c _ {M ^ {\prime}} (r), k} ^ {M ^ {\prime}} z _ {k}, \tag {57}
$$

Here $T _ { r } ^ { M }$ and $T _ { r } ^ { M ^ { \prime } }$ denote the non-target features that leak into the scalar readout of feature r from the selected coordinates $c _ { M } ( r )$ and $c _ { M ^ { \prime } } ( r )$ . The coefficients $b _ { c _ { M } ( r ) , k } ^ { M }$ bMcM (r),k and bM′cM′ (r),k are the corresponding $b _ { c _ { M ^ { \prime } } ( r ) , k } ^ { M ^ { \prime } }$ normalized leakage coefficients, obtained after rescaling by the target-feature loadings $W _ { c _ { M } ( r ) , r } ^ { M }$ cM (r),r and WM' $W _ { c _ { M ^ { \prime } } ( r ) , r } ^ { M ^ { \prime } }$ W M′cM′ (r),r. This tail-packing calculation focuses on the feature-dependent part of the interference; the residual noise-floor variation is treated as independent across models and therefore does not contribute to cross-model covariance. Since the total normalized interference variances are $1 / s _ { r } ^ { M }$ and $1 / s _ { r } ^ { M ^ { \prime } }$ , the feature-leakage coefficients satisfy

$$
\sum_ {k \in T _ {r} ^ {M}} \left(b _ {c _ {M} (r), k} ^ {M}\right) ^ {2} \leq 1 / s _ {r} ^ {M}, \quad \sum_ {k \in T _ {r} ^ {M ^ {\prime}}} \left(b _ {c _ {M ^ {\prime}} (r), k} ^ {M ^ {\prime}}\right) ^ {2} \leq 1 / s _ {r} ^ {M ^ {\prime}}. \tag {58}
$$

To compute $\mathrm { C o v } ( \epsilon _ { M } ^ { \mathrm { f e a t } } , \epsilon _ { M ^ { \prime } } ^ { \mathrm { f e a t } } )$ , we expand the two residual mixtures, which gives a double sum over pairs of leaked features. Under the diagonal feature approximation, $\mathrm { C o v } ( z _ { k } , z _ { \ell } ) = 0$ for $k \neq \ell$ and $\bar { \mathrm { V a r } } ( z _ { k } ) = 1$ . Therefore, only the same latent feature appearing in both residual mixtures contributes to cross-model covariance:

$$
\operatorname{Cov} (\epsilon_ {M} ^ {\text { feat }}, \epsilon_ {M ^ {\prime}} ^ {\text { feat }}) = \sum_ {k \in T _ {r} ^ {M} \cap T _ {r} ^ {M ^ {\prime}}} b _ {c _ {M} (r), k} ^ {M} b _ {c _ {M ^ {\prime}} (r), k} ^ {M ^ {\prime}}. \tag {59}
$$

Random tail packing. This term is small in a simple random tail-packing approximation. Suppose the tail available to the readout of feature r has effective size $L _ { r } ,$ where $L _ { r }$ is the effective number of unresolved non-target features that could contribute interference to this scalar readout. We view each set of leakage coefficients as a vector over this Lr-dimensional tail, with zero entries for features absent from the mixture. If the two models choose tail-mixing directions approximately independently and isotropically in this space, then

$$
\mathbb {E} \left[ \operatorname{Cov} (\epsilon_ {M} ^ {\text { feat }}, \epsilon_ {M ^ {\prime}} ^ {\text { feat }}) \right] = 0, \tag {60}
$$

and

$$
\operatorname{Var} \left[ \operatorname{Cov} (\epsilon_ {M} ^ {\mathrm{feat}}, \epsilon_ {M ^ {\prime}} ^ {\mathrm{feat}}) \right] = O \left(\frac {\| b _ {c _ {M} (r)} ^ {M} \| _ {2} ^ {2} \| b _ {c _ {M ^ {\prime}} (r)} ^ {M ^ {\prime}} \| _ {2} ^ {2}}{L _ {r}}\right) \leq O \left(\frac {1}{s _ {r} ^ {M} s _ {r} ^ {M ^ {\prime}} L _ {r}}\right). \tag {61}
$$

Isolation controls matching. Thus, as the unresolved-tail dimension $L _ { r }$ grows, independently packed feature-dependent interference becomes asymptotically uncorrelated across models. Any remaining noise-floor variation is also treated as independent across models. We therefore model $\epsilon _ { M }$ and $\epsilon _ { M ^ { \prime } }$ as uncorrelated across models and uncorrelated with the shared feature signal. Under this approximation, we obtain

$$
\operatorname{Corr} (\tilde {y} _ {c _ {M} (r)} ^ {M}, \tilde {y} _ {c _ {M ^ {\prime}} (r)} ^ {M ^ {\prime}}) = \frac {1}{\sqrt {(1 + 1 / s _ {r} ^ {M}) (1 + 1 / s _ {r} ^ {M ^ {\prime}})}}. \tag {62}
$$

Cross-model activation correlation is monotonically increasing in the feature-level isolation score of each model. It follows that a high-isolation feature is more likely to produce a high-correlation cross-model neuron score. We next connect this score-level relationship to the notion of Rosetta Neuron matching.

For two models and a fixed detectability threshold $\tau > 1$ , the corresponding idealized two-model Rosetta Neuron count is

$$
R _ {\tau} ^ {M, M ^ {\prime}} (N) = \sum_ {r = 1} ^ {A (N)} \mathbf {1} \{\min (s _ {r} ^ {M}, s _ {r} ^ {M ^ {\prime}}) \geq \tau \}. \tag {63}
$$

The condition $\tau > 1$ gives the idealized count a one-feature, one-coordinate interpretation: thresholdpassing features have distinct dedicated neuron coordinates. This mirrors the empirical matching rule, where we use mutual nearest-neighbor matching with $k = 1$ under Pearson correlation. This enforces one-to-one cross-model neuron matches. By Eq. (62), higher isolation scores imply higher cross-model correlation. So, mutual nearest-neighbor matching can be interpreted as an empirical detectability filter: it prefers to keep features whose shared signal dominates model-specific interference in both models.

# D.7 Synthetic Validation of the Analytical Model

We perform a synthetic feature-packing experiment to test whether our phenomenological model produces sublinear Rosetta Neuron scaling and neuron polarization. We embed the theory’s predicted isolation profile into synthetic activations for two independent networks. We then recover Rosetta Neurons using the same Pearson-correlation mutual-nearest-neighbor procedure used in our main experiments and compare the recovered scaling and polarization trends to the theory’s predictions.

Experimental setup. For each neuron budget $N$ , we instantiate $A = \gamma N$ latent features with importance weights $\bar { w } _ { r } \propto r ^ { - \beta }$ . We then solve the isolation-allocation problem with budget $\textstyle \sum _ { r } s _ { r } \leq$ $\kappa N$ to obtain predicted isolation scores $s _ { r }$ . For each synthetic input $x ,$ we sample latent feature responses $z _ { r } ( x )$ for all features r. We assign every feature with positive isolation, $s _ { r } \ > \ 0$ , to a synthetic neuron coordinate. Features with $s _ { r } = 0$ are not assigned as target features, but may still appear as background interference. For a neuron $j$ assigned to feature r, we sample a background set $B _ { j }$ of non-target features and write the activation as

$$
h _ {j} (x) = W _ {j, r} z _ {r} (x) + \sum_ {k \in \mathcal {B} _ {j}} W _ {j, k} z _ {k} (x) + \xi_ {j} (x), \tag {64}
$$

where $\xi _ { j } ( x )$ is an irreducible noise-floor term with variance $\sigma _ { \infty } ^ { 2 }$ . We draw the off-target loadings $W _ { j , k }$ for $k \in B _ { j }$ as a random isotropic Gaussian direction and normalize them to a fixed interference norm. We then choose the target-feature loading $W _ { j , \prime }$ so that the resulting signal-to-interference ratio matches the predicted isolation score:

$$
\frac {W _ {j , r} ^ {2}}{\sum_ {k \in \mathcal {B} _ {j}} W _ {j , k} ^ {2} + \sigma_ {\infty} ^ {2}} \approx s _ {r}. \tag {65}
$$

The background mixtures and noise are sampled independently across the two synthetic models, while the latent feature responses $z _ { r } ( x )$ are shared.

Evaluation protocol. We run the simulation across neuron budgets $N \in \{ 2 ^ { 1 0 } , \dots , 2 ^ { 1 7 } \}$ and featurespectrum exponents $\beta \in \{ 1 . 5 , 1 . 7 5 , 2 , 2 . 2 5 , 2 . 5 , 3 \}$ . For each $( \bar { N } , \beta )$ setting, we report the mean over 10 independent runs; error bars are omitted because the standard errors are smaller than the plotted markers. For each run, we sample M synthetic inputs by drawing a shared latent feature-response matrix $Z \in \mathbb { R } ^ { M \times A }$ with i.i.d. Gaussian entries, then standardize each feature column across inputs. The same $Z$ is used for both synthetic models, while the neuron-level feature mixtures and noise are sampled independently across models. We then apply Pearson-correlation mutual-nearest-neighbor matching between the two synthetic models. We compare the recovered Rosetta count against the theoretical scaling prediction $R _ { \tau } ( N ) = \Theta ( N ^ { 1 / \beta } )$ . To test the polarization effect, we separately track the mean isolation of recovered Rosetta neurons and non-Rosetta neurons as N increases.

Rosetta Neuron matching recovers theoretical predictions. The simulations closely match the analytical predictions as seen in Figure 27. Across the sweep of feature-spectrum exponents $\beta _ { ; }$ the recovered Rosetta Neuron count follows the predicted scaling $R _ { \tau } ( N ) = \Theta ( N ^ { 1 / \beta } )$ : the fitted empirical slopes align with the theoretical exponents $1 / \beta$ (Figure 27a). Since the theory predicts the scaling exponent but not the prefactor, we plot theoretical reference curves with slope $1 / \beta$ and choose the intercept so that each curve passes through the first empirical point. This agreement suggests that Pearson-correlation mutual-nearest-neighbor matching recovers the theory-predicted Rosetta detectability threshold.

The simulation also recovers the predicted polarization behavior. Our analytical model predicts that the mean isolation of Rosetta neurons grows as $\Theta \big ( N ^ { ( \beta - 1 ) / \beta } \big )$ , while the average clean isolation associated with the non-Rosetta neuron population decays as $\dot { O } \big ( N ^ { ( 1 - \beta ) / \beta } \big )$ . Consistent with these predictions, Rosetta Neuron matches have increasing assigned-feature isolation with slope close to $( \beta - 1 ) / \beta$ (Figure 27b). The unmatched non-Rosetta neurons have decreasing average isolation with slope close to the theoretical $- ( \beta - 1 ) / \beta$ (Figure 27c).

Additional Details. For the reported results, we used $\kappa = 1 . 0 , \gamma = 4 . 0 , M = 1 0 , 0 0 0 .$ . We additionally verified that the exponent recovery is robust to these simulation hyperparameters. Across $\kappa \in \{ 2 ^ { - 1 } , 2 ^ { 0 } , \ldots , 2 ^ { 5 } \} , \gamma \in \{ \dot { 2 } ^ { - 1 } , 2 ^ { 0 } , \ldots , 2 ^ { 5 } \}$ , and $M \in \{ 1 0 ^ { 3 } , 1 0 ^ { 4 } , 1 0 ^ { 5 } \}$ , all the fitted exponents remained within $\epsilon \approx 0 . 0 5$ of the corresponding theoretical predictions.

Rosetta Neurons Follow Sublinear Scaling   
![](images/e354075aa9eff50329b4f7bdcdf2523d63237772c57706edc37f3b136415f095.jpg)

<details>
<summary>line</summary>

| Total Neurons | β = 1.5 Theory | β = 1.5 Simulation | β = 1.75 Theory | β = 1.75 Simulation | β = 2.25 Theory | β = 2.25 Simulation | β = 2.5 Theory | β = 2.5 Simulation | β = 3 Theory | β = 3 Simulation |
| ------------- | -------------- | ------------------ | --------------- | ------------------- | --------------- | ------------------- | -------------- | ------------------ | ----------- | ---------------- |
| 1000          | ~40            | ~30                | ~25             | ~20                 | ~15             | ~10                 | ~10            | ~8                 | ~8          | ~6               |
| 10000         | ~150           | ~100               | ~80             | ~60                 | ~40             | ~30                 | ~25            | ~20                | ~15         | ~10              |
| 100000        | ~500           | ~300               | ~200            | ~150                | ~80             | ~60                 | ~40            | ~30                | ~25         | ~15              |
</details>

(a) Rosetta Neuron counts scale as N 1/β. $N ^ { 1 / \beta }$

![](images/93e4d7def26fdf12a0efd0f16d242aeed4c838fd007a95c63ab6dfabd90ca194.jpg)

<details>
<summary>line</summary>

| Total Neurons | β = 1.5 Theory | β = 1.5 Simulation | β = 1.75 Theory | β = 1.75 Simulation | β = 2 | β = 2 | β = 2.25 Theory | β = 2.25 Simulation | β = 2.5 | β = 2.5 Simulation | β = 3 | β = 3 Simulation |
| ------------- | -------------- | ------------------ | --------------- | ------------------- | ----- | ----- | --------------- | ------------------- | ------- | ------------------- | ----- | ----------------- |
| 1000          | ~50            | ~40                | ~80             | ~70                 | ~60   | ~90   | ~120            | ~110                | ~150    | ~140                | ~200  | ~190              |
| 10000         | ~100           | ~80                | ~180            | ~160                | ~140  | ~240  | ~360            | ~340                | ~480    | ~460                | ~700  | ~680              |
| 100000        | ~200           | ~160               | ~360            | ~320                | ~300  | ~640  | ~1280           | ~1240               | ~2560   | ~2480               | ~1280 | ~1240             |
</details>

(b) Rosetta Neuron isolation scales as N (β−1)/β. $N ^ { ( \beta - 1 ) / \beta }$

![](images/e2f9e85e0a6a2ab50447d9076091e72ecfd61d6d0725ef4c875d547eb3932a01.jpg)

<details>
<summary>line</summary>

| Total Neurons | β = 1.5 Theory | β = 1.5 Simulation | β = 1.75 Theory | β = 1.75 Simulation | β = 2.25 Theory | β = 2.25 Simulation | β = 2.5 Theory | β = 2.5 Simulation | β = 3 Theory | β = 3 Simulation |
| ------------- | -------------- | ------------------ | --------------- | ------------------- | --------------- | ------------------- | -------------- | ------------------ | ----------- | ---------------- |
| 1000          | 0.01           | 0.01               | 0.005           | 0.005               | 0.002           | 0.002               | 0.001          | 0.001              | 0.0005      | 0.0005           |
| 10000         | 0.002          | 0.002              | 0.001           | 0.001               | 0.0005          | 0.0005              | 0.0002         | 0.0002             | 0.0001      | 0.0001           |
| 100000        | 0.001          | 0.001              | 0.0005          | 0.0005              | 0.0002          | 0.0002              | 0.0001         | 0.0001             | 0.00005     | 0.00005          |
</details>

(c) Non-Rosetta neuron isolation scales as N (1−β)/β. $N ^ { ( 1 - \beta ) / \beta }$   
Figure 27: Scaling behavior of Rosetta and non-Rosetta neurons in simulation. Top: the number of Rosetta Neurons follows the predicted scaling law according to our analytical model. Middle: Rosetta Neurons become more isolated with scale. Bottom: Non-Rosetta neurons become less isolated with scale, as predicted by our theory.

# E Additional Results on Rosetta Neuron Properties

In this section, we provide additional analyses supporting the neuron-level trends described in Section 5. We first extend the language-model selectivity results from Section 5.1 to additional model families. We then provide further details and results for the document-type firing analysis introduced in Section 5.2. Finally, we analyze where Rosetta Neurons appear across network depth in both language and vision models.

# E.1 Additional Vocabulary-Space Selectivity Results

We measure output-side selectivity using the same vocabulary-space excess-kurtosis metric from Section 5.1. As previously shown in Figure 6, a polarization occurs: the mean excess kurtosis of the Rosetta population increases with scale, while non-Rosetta neurons remain near zero, suggesting weak selectivity. We observe the same qualitative trend for OPT and Qwen-2.5 in Figure 28. Across both families, Rosetta Neurons become increasingly selective with scale, while non-Rosetta neurons remain close to the floor. This supports the interpretation that the Neuron Polarization Effect is not tied to a single model family, but reflects a more general trend across language models.

![](images/9a42a7e6241001cb406a664443c530586fdf2a0b033b7b6ae0fdb5bda663e6c9.jpg)

<details>
<summary>line</summary>

| Model Size (# Neurons) | Rosetta Neurons | Non-Rosetta Neurons |
| ---------------------- | --------------- | ------------------- |
| 2 × 10⁵                | 7.5             | 1.5                 |
| 3 × 10⁵                | 9.5             | 1.5                 |
| 5 × 10⁵                | 12.5            | 1.8                 |
| 8 × 10⁵                | 13.5            | 1.5                 |
| 1 × 10⁶                | 19.5            | 1.2                 |
</details>

(a) Vocabulary-space neuron selectivity in OPT.

![](images/622d0cecf4e81962b5b13a62ce8a4a52302b0d6ef5f5ce8fd16f40dd3c44b393.jpg)

<details>
<summary>line</summary>

| Model Size (# Neurons) | Rosetta Neurons | Non-Rosetta Neurons |
| ----------------------- | --------------- | ------------------- |
| 2 × 10⁵                 | 4.0             | 1.2                 |
| 5 × 10⁵                 | 6.5             | 1.5                 |
| 1 × 10⁶                 | 8.0             | 3.5                 |
| >1 × 10⁶                | 13.5            | 2.0                 |
</details>

(b) Vocabulary-space neuron selectivity in Qwen2.5.   
Figure 28: The Neuron Polarization Effect in Language Models. Rosetta Neurons exhibit increasing mean excess kurtosis of vocabulary-space projections with scale, suggestive of monosemantic function. Non-Rosetta neurons remain near zero, consistent with weaker vocabulary-level selectivity under this metric.

# E.2 Document-Type Firing Analysis

We provide additional details and results for the document-type firing analysis in Section 5.2. For each Rosetta Neuron in a given model, we retrieve its top 20 activating contexts from the validation set of The Pile. Each context is assigned the Pile subset label of its source document (e.g., “GitHub”). We aggregate the original Pile subsets into five general source categories: code, math, formal/scientific text, general prose, and conversational text. The mapping from Pile subsets to categories, together with each category’s token share of the validation cache, is shown in Table 4.

For each category c, we compute the fraction of Rosetta Neuron top activations assigned to that category, and then normalize by the category’s token frequency in the validation cache:

$$
\text { NormalizedFire } (c) = \frac {\# \{\text { top   activations   from   category } c \} / \# \{\text { all   top   activations } \}}{\# \{\text { tokens   from   category } c \} / \# \{\text { all   tokens } \}}.
$$

A value of one corresponds to the corpus baseline, meaning that top activations fall in category c at the same rate as expected from its token frequency. Values above one indicate over-representation among top activations, while values below one indicate under-representation.

As a control, for each model family and scale, we randomly sample 1000 non-Rosetta neurons from the corresponding model and compute the same normalized document-type firing statistic. We repeat this procedure three times and report the mean and 95% confidence interval across random non-Rosetta neuron samples. This baseline tests whether the observed document-type trends reflect a general property of neurons at a given scale, rather than a property specific to the Rosetta population. We apply this analysis on Pythia and Qwen-2.5 and report the results in Figures 29 and 30. Rosetta Neuron firing becomes increasingly concentrated on specialized categories such as code and math as model size increases for both model families. In contrast, random neurons may exhibit categoryspecific biases, but they do not show the same systematic shift in firing preference; their category-level firing patterns remain comparatively stable across scale. Since the models within each family are trained on the same data mixture, this shift is unlikely to be explained solely by changing exposure to specialized documents. It suggests that scale may affect which domain-specific features are represented among Rosetta Neurons.

<table><tr><td>Category</td><td>Corpus Share</td><td>Pile Subsets</td></tr><tr><td>Code</td><td>0.1195</td><td>GitHub</td></tr><tr><td>Math</td><td>0.0317</td><td>DM Mathematics</td></tr><tr><td>Formal/scientific</td><td>0.4644</td><td>ArXiv, PubMed Central, PubMed Abstracts, FreeLaw, USPTO Back-grounds, NIH ExPorter, StackExchange, PhilPapers</td></tr><tr><td>General prose</td><td>0.3430</td><td>Pile-CC, OpenWebText2, Wikipedia (en), Books3, Gutenberg (PG-19), EuroParl, BookCorpus2</td></tr><tr><td>Conversational</td><td>0.0415</td><td>OpenSubtitles, Ubuntu IRC, HackerNews, Enron Emails, YouTubeSubtitles</td></tr></table>

Table 4: Pile source categories used for document-type firing analysis. We aggregate the 22 Pile subsets into five coarse source categories. The corpus share denotes each category’s token frequency in the validation cache and is used to normalize top-activation frequencies.

Rosetta Neuron Firing by Document Type   
![](images/958655bcd57d398204ace822e71993604a3bb3df38b607bd176863b6c023c886.jpg)

<details>
<summary>bar</summary>

| Text Source Category | 160M | 1.4B | 6.9B | 410M | 2.8B | 12B |
|----------------------|------|------|------|------|------|-----|
| Code                 | 0.45 | 0.50 | 0.70 | 0.75 | 1.05 | 1.10 |
| Math                 | 0.70 | 0.80 | 1.25 | 1.20 | 1.25 | 1.40 |
| Formal / Science      | 0.80 | 0.80 | 0.80 | 0.80 | 0.85 | 0.95 |
| General Prose        | 1.45 | 1.45 | 1.35 | 1.45 | 1.35 | 1.15 |
| Conversational       | 1.45 | 1.40 | 1.35 | 1.50 | 1.35 | 1.15 |
</details>

(a) Pythia Rosetta Neurons.

Random Neuron Firing by Document Type   
![](images/d0c8e9f593ca87aeae907fdff8865d3069c6520efdde67ea36c40f67c7f9aefe.jpg)

<details>
<summary>bar</summary>

| Text Source Category | 160M | 410M | 1.4B | 2.8B | 6.9B | 12B |
|----------------------|------|------|------|------|------|-----|
| Code                 | 0.9  | 0.9  | 1.0  | 1.0  | 0.9  | 0.9 |
| Math                 | 1.2  | 1.2  | 1.35 | 1.35 | 1.4  | 1.3 |
| Formal / Science     | 0.9  | 0.9  | 0.9  | 0.9  | 0.9  | 0.9 |
| General Prose        | 1.15 | 1.15 | 1.1  | 1.1  | 1.15 | 1.2 |
| Conversational       | 1.1  | 1.05 | 1.1  | 1.2  | 1.35 | 1.25 |
Corpus Baseline    |
</details>

(b) Pythia random non-Rosetta neurons.   
Figure 29: Rosetta Neuron document-type firing in Pythia. For each Pythia model size, each bar shows how often top-activating Rosetta Neuron contexts fall into a document category, normalized by that category’s token frequency in the validation cache. The dashed line marks the corpus baseline. With scale, Rosetta Neuron firing shifts toward specialized categories such as code and math.

Rosetta Neuron Firing by Document Type   
![](images/e663f707f63bd75033d94785cbea201f1311d371f2ed7a7aa144d1bd2dae6b74.jpg)

<details>
<summary>bar</summary>

| Text Source Category | 1.5B   | 3B     | 7B     | 14B    |
| -------------------- | ------ | ------ | ------ | ------ |
| Code                 | 0.55x  | 0.60x  | 0.95x  | 1.05x  |
| Math                 | 1.30x  | 1.45x  | 1.35x  | 1.60x  |
| Formal / Science     | 0.80x  | 0.80x  | 0.85x  | 0.90x  |
| General Prose        | 1.35x  | 1.35x  | 1.20x  | 1.05x  |
| Conversational       | 1.75x  | 1.65x  | 1.45x  | 1.35x  |
</details>

(a) Qwen2.5 Rosetta Neurons.

Random Neuron Firing by Document Type   
![](images/76becda6a33e21b34f4d569fa6593bbbae2e2eb724fd56b8220ea01a0600ca3a.jpg)

<details>
<summary>bar</summary>

| Text Source Category | 1.5B | 3B | 7B | 14B | 32B |
| --- | --- | --- | --- | --- | --- |
| Code | 1.35 | 1.30 | 1.30 | 1.40 | 1.30 |
| Math | 0.65 | 0.95 | 0.65 | 0.45 | 0.60 |
| Formal / Science | 0.75 | 0.75 | 0.70 | 0.70 | 0.80 |
| General Prose | 1.20 | 1.15 | 1.25 | 1.30 | 1.20 |
| Conversational | 0.95 | 0.85 | 0.95 | 0.85 | 1.00 |
</details>

(b) Qwen2.5 random non-Rosetta neurons.   
Figure 30: Document-type firing in Qwen2.5. We use the same normalized document-type firing statistic as in Figure 29. Rosetta Neurons show an increasing shift toward specialized categories such as code and math with scale. Random non-Rosetta neurons may exhibit category-specific biases, but do not exhibit the same consistent scale-dependent shift toward specialized domains.

# E.3 Depth-Wise Distribution of Rosetta Neurons

We analyze where Rosetta Neurons appear across network depth as models scale. For language models, we use Rosetta Neurons discovered from the Pythia–OPT matching runs. For vision models, we use Rosetta Neurons discovered from the Diffusion–OpenCLIP matching runs. Because models at different scales have different numbers of layers, we compare layers using normalized depth. For each model, we map layer index ℓ to normalized depth ℓ/(L − 1), where L is the total number of transformer blocks, and divide the interval [0, 1] into 12 equally spaced bins. We then compute the fraction of discovered Rosetta Neurons that fall into each depth bin.

As shown in Figures 31 and 32, Rosetta Neurons are distributed across multiple depth bins in both Pythia and OpenCLIP, rather than being confined to a single depth range across scale. The distribution is not uniform: some models show stronger concentration in early layers, such as Pythia-6.9B in language and OpenCLIP ViT-L/14 (300M) in vision. Overall, however, the depth-wise analysis does not reveal a single consistent depth profile shared across all scales and modalities. We therefore interpret this analysis primarily as a non-degeneracy check: the Rosetta Neuron population is not explained solely by matches from one fixed layer region, and the observed scaling trends are not driven by a single depth profile.

Pythia Rosetta-Neuron Depth Distribution   
![](images/e94deb687da306040afcb28ba758d4c348635fd20c237932dc57c111ff729c88.jpg)

<details>
<summary>heatmap</summary>

| Model Size | 0.00-0.06 | 0.08-0.17 | 0.17-0.25 | 0.25-0.33 | 0.33-0.42 | 0.42-0.50 | 0.50-0.58 | 0.58-0.67 | 0.67-0.75 | 0.75-0.83 | 0.83-0.92 | 0.92-1.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 160M | 0.09 | 0.03 | 0.04 | 0.04 | 0.09 | 0.10 | 0.13 | 0.10 | 0.14 | 0.10 | 0.09 | 0.07 |
| 410M | 0.08 | 0.03 | 0.04 | 0.13 | 0.13 | 0.07 | 0.09 | 0.11 | 0.11 | 0.08 | 0.05 | 0.08 |
| 1.4B | 0.10 | 0.04 | 0.08 | 0.12 | 0.11 | 0.11 | 0.11 | 0.10 | 0.09 | 0.04 | 0.05 | 0.06 |
| 2.8B | 0.13 | 0.06 | 0.12 | 0.14 | 0.07 | 0.11 | 0.11 | 0.06 | 0.06 | 0.03 | 0.05 | 0.07 |
| 6.9B | 0.31 | 0.06 | 0.07 | 0.10 | 0.05 | 0.07 | 0.08 | 0.07 | 0.08 | 0.03 | 0.04 | 0.05 |
| 12B | 0.23 | 0.03 | 0.08 | 0.10 | 0.05 | 0.07 | 0.08 | 0.09 | 0.08 | 0.07 | 0.05 | 0.08 |
</details>

Figure 31: Depth-wise distribution of Rosetta Neurons in Pythia across scale. Rosetta Neurons discovered from the Pythia–OPT matching runs.

OpenCLIP Rosetta-Neuron Depth Distribution   
![](images/1df1efe17f3a9f13dcf189331e5d40b613a455224df6c3d09cad3af6f3bef65e.jpg)

<details>
<summary>heatmap</summary>

| Model Size | 0.00-0.08 | 0.08-0.17 | 0.17-0.25 | 0.25-0.33 | 0.33-0.42 | 0.42-0.50 | 0.50-0.58 | 0.58-0.67 | 0.67-0.75 | 0.75-0.83 | 0.83-0.92 | 0.92-1.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 86M | 0.09 | 0.07 | 0.14 | 0.12 | 0.15 | 0.17 | 0.11 | 0.07 | 0.05 | 0.01 | 0.01 | 0.00 |
| 300M | 0.44 | 0.26 | 0.11 | 0.07 | 0.04 | 0.03 | 0.02 | 0.01 | 0.00 | 0.00 | 0.01 | 0.00 |
| 986M | 0.46 | 0.22 | 0.05 | 0.10 | 0.08 | 0.05 | 0.02 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 |
| 1.8B | 0.06 | 0.07 | 0.09 | 0.07 | 0.08 | 0.09 | 0.08 | 0.04 | 0.07 | 0.15 | 0.11 | 0.08 |
| 4.4B | 0.22 | 0.11 | 0.09 | 0.06 | 0.06 | 0.03 | 0.05 | 0.17 | 0.14 | 0.05 | 0.03 | 0.00 |
</details>

Figure 32: Depth-wise distribution of Rosetta Neurons in OpenCLIP across scale. Rosetta Neurons discovered from the Diffusion–OpenCLIP matching runs.

# F Data Filtering Experimental Details

Data. We use CodeSearchNet (Husain et al., 2019), a function-level code corpus extracted from publicly available GitHub repositories. This dataset spans six programming languages: Python, JavaScript, Java, Go, Ruby, PHP. Each example in the dataset is a single parsed function from a real-world repository. We use JavaScript as a representative target domain. Its training split contains 58,025 functions, corresponding to roughly 16M tokens under the GPT-2 tokenizer, and accounts for approximately 6% of the total tokens in the full multilingual training set. The corresponding JavaScript test split contains 3291 functions and 1.11M tokens.

Filtering. Each filtering method selects functions from the full multilingual CodeSearchNet training set under a matched-token setup. The target budget is set to the size of the JavaScript training split, approximately 16M GPT-2 tokens. We compare four filtering methods: a Rosetta Neuron filter, a non-Rosetta neuron filter, an i.i.d. random sample from the full multilingual training set, and the oracle JavaScript training split. For the neuron-based filters, we score each function by the selected neuron’s mean activation over tokens in the function, and select the highest-scoring functions until reaching the matched token budget.

The neurons used for filtering are selected prior to the CodeSearchNet experiment using a different data distribution. We identify candidate JavaScript-selective neurons in Pythia-6.9B using the LLMbased neuron labeling procedure from Section A, which infers neuron function from top-activating contexts in The Pile. Among Rosetta Neurons whose top-activating Pile contexts are annotated as JavaScript-related, we select the single neuron with the highest mean normalized activation over its top-20 Pile contexts. We apply the same rule to non-Rosetta neurons in the same layer whose top activations are annotated as JavaScript-related. This selection is performed using only The Pile activations and LLM annotations before any CodeSearchNet filtering or downstream evaluation. To evaluate domain recovery, we compute F1 between the selected training functions and the oracle

JavaScript training split (Table 1). Since all methods use the same token budget, precision and recall are nearly identical, so F1 compactly summarizes ground truth recovery.

We present qualitative examples of dataset recovery below. The Rosetta Neuron, Layer 16 Unit 11168 in Pythia-6.9B, retrieves JavaScript functions with high selectivity. In contrast, the non-Rosetta neuron, Layer 16 Unit 12873, shows weaker JavaScript selectivity and fires on a broader range of code, especially web-oriented languages such as PHP.

Training details. To evaluate the downstream utility of the filters, we conduct continued pretraining of GPT2-1.5B using rank-1 LoRA in the attention and MLP layers (Hu et al., 2022). All runs use AdamW (Loshchilov & Hutter, 2019) with peak learning rate 10−3 and a 3% linear warmup followed by cosine decay to 0. We train for one epoch at sequence length 1024 using an effective batch size of 32 sequences. All filtering variants use the same model, hyperparameters, token budget, and number of optimizer steps; only the composition of the training data varies. We report perplexity in Table 1 with 95% confidence intervals calculated over three runs.

Rosetta Neuron-Filtered Function #1: Label=JavaScript   
```javascript
function (firstStart) {
    this._startTime = (firstStart) ? Aria._start : new Date();
    if (firstStart) {
    this._logs = [{
    classpath : "Aria",
    msg : "Framework initialization",
    start : Aria._start
    }, {
    classpath : "Aria",
    stop : (new Date()).getTime()
    }];
    this._nbLogs = 2;
    } else {
    this._logs = [];
    this._nbLogs = 0;
    }

    // map function on ItsObject prototype
    ariaCoreJsObject.prototype.$logTimestamp = function (msg, classpath) {
    classpath = classpath ? classpath : this.$classpath;
    aria.utils.Profiling.logTimestamp(classpath, msg);
    };
    ariaCoreJsObject.prototype.$startMeasure = function (msg, classpath) {
    classpath = classpath ? classpath : this.$classpath;
    return aria.utils.Profiling.startMeasure(classpath, msg);
    };
    ariaCoreJsObject.prototype.$stopMeasure = function (id, classpath) {
    classpath = classpath ? classpath : this.$classpath;
    aria.utils.Profiling.stopMeasure(classpath, id);
    };
} 
```

Rosetta Neuron-Filtered Function #2: Label=JavaScript   
```javascript
function(fn, mOptions) {
    // Functionality taken from lodash open source library and adapted as needed
    mOptions = Object.assign({
    wait: 0,
    leading: true
    }, mOptions);
    mOptions.maxWait = mOptions.wait;
    mOptions.trailing = true;
    mOptions.requestAnimationFrame = false;

    return TableUtils.debounce(fn, mOptions);
} 
```

Rosetta Neuron-Filtered Function #3: Label=JavaScript   
```javascript
function ForEach(f, arr) {
    if (NATIVE_ARRAY_FOREACH) {
    if (arr) {
    NATIVE_ARRAY_FOREACH.call(arr, f);
    }
    return;
    }
    for (var i = 0; i < arr.length; i++) {
    f(arr[i], i, arr);
    }
} 
```

Non-Rosetta Neuron-Filtered Function #1: Label=PHP   
```txt
protected static function parseSequence($sequence, &$i = 0)
{
    $output = array();
    $len = strlen($sequence);
    $i += 1;

    // [foo, bar, ...]
    while ($i < $len) {
    switch ($sequence[$i]) {
    case '[':
    // nested sequence
    $output[] = self::parseSequence($sequence, $i);
    break;
    case '{':
    // nested mapping
    $output[] = self::parseMapping($sequence, $i);
    break;
    case '']':
    return $output;
    case ',':
    case '':
    break;
    default:
    $isQuoted = in_array($sequence[$i], array('"'', "''"));
    $value = self::parseScalar($sequence, array(', ], '']'), array('"'', "'''), $i);

    if (!$isQuoted && false !== strpos($value, ': ')) {
    // embedded mapping?
    try {
    $value = self::parseMapping vo{'.$value.'}');
    } catch (InvalldArgumentException $e) {
    // no, it's not
    }
    }

    $output[] = $value;

    --$i;
    }

    ++$i;
}

throw new InvalldArgumentException(sprintf('Malformed inline YAML string %s', $sequence));
} 
```

Non-Rosetta Neuron-Filtered Function #2: Label=PHP   
```perl
private function parsePhpValue($key, $value, array &$result)
{
    $node = & $result;
    $keyBuffer = '';
    for ($i = 0, $t = strlen($key); $i < $t; $i++) {
    switch ($key[$i]) {
    case '['':
    if ($keyBuffer) {
    $this->prepareNode($node, $keyBuffer);
    $node = & $node[$keyBuffer];
    $keyBuffer = '';
    }
    break;
    case '']':
    $k = $this->cleanKey($node, $keyBuffer);
    $this->prepareNode($node, $k);
    $node = & $node[$k];
    $keyBuffer = '';
    break;
    default:
    $keyBuffer .= $key[$i];
    break;
    }
    }

    if (isset($node)) {
    $this->duplicates = true;
    $node[] = $value;
    } else {
    $node = $value;
    }
} 
```

Non-Rosetta Neuron-Filtered Function #3: Label=JavaScript   
```javascript
function generateUniqueKey(index, initiallyKey) {
    var currentCandidate = InitialKey;

    var counter = 0;
    while (index[currentCandidate]) {
    var numberAtEndOfKeyMatches = currentCandidate.match(
    NUMBER_AT_END_OF_KEY_REGEX
    );
    if (numberAtEndOfKeyMatches != null) {
    var nextNumber = parseInt(numberAtEndOfKeyMatches[1], 10) + 1;

    currentCandidate = currentCandidate.replace(
    NUMBER_AT_END_OF_KEY_REGEX,
    "(" + nextNumber + ")
    );
    } else {
    currentCandidate += " (1)";
    }

    // This loop should always find something eventually, but because it's a bit dangerous looping endlessly...
    counter++;
    if (counter >= 100000) {
    throw new DevellerError(
    "Was not able to find a unique key for " +
    initialKey +
    " after 100000 iterations." +
    " This is probably because the regex for matching keys was somehow unable to work for that key."
    );
    }
    }

    return currentCandidate;
} 
```

# G Dataset Ablations

# G.1 Ablation on the Number of Tokens Used for Language Model Matching

We ablate the number of tokens for matching neurons between language models, taking the Pythia-6.9B–OPT-6.7B model pair as a representative run. Specifically, we sample i.i.d. token sequences from the validation split of The Pile and apply the procedure described in Section 3 to identify Rosetta Neurons. We sweep the total number of tokens used for matching from 103 to 108. At each data scale, we identify the set of Rosetta Neurons and report the total count in Figure 33 (left). We also compute the overlap between the Rosetta Neuron set identified at each scale and the set identified at the immediately preceding scale, shown in Figure 33 (right). We find that the discovered Rosetta Neuron set becomes increasingly stable as the number of tokens used for matching increases. We use 10M tokens for the main language-model scaling runs, as this provides a practical balance between stability and computational cost.

# G.2 Ablation on the Number of Images Used for Vision Model Matching

We ablate the number of images used to match neurons between a diffusion model and a discriminative model. Specifically, we take pMF DiT-B/16 and OpenCLIP ViT-B/16 as a representative pairing and follow the procedure described in Section 3 to identify Rosetta Neurons. We sweep the number of images generated by the diffusion model from 1 to 50,000. At each data scale, we identify the set of Rosetta Neurons and report the total count in Figure 34 (left). We also measure the overlap between the set of Rosetta Neurons identified at each data scale and the set identified at the immediately preceding data scale, shown in Figure 34 (right). We find that the discovered set of Rosetta Neurons stabilizes by 50,000 images: the total count plateaus, and the overlap between successive data scales approaches 1.

# G.3 Ablation on the Image Distribution Used for Vision Model Matching

Our vision experiments match neurons between a generative model and a discriminative model, following the GAN-based setup of (Dravid et al., 2023). For modern diffusion-based generators, this requires generated images, since activations from the generative model are only available along the generation trajectory. A natural concern is that this introduces a distribution shift for the discriminative model, which may affect which Rosetta Neurons are identified. As a proxy for this distribution-shift concern, we compare matching under real and generated image distributions using two discriminative models, OpenCLIP ViT-B/16 and DINOv2 ViT-B/14. We run the Rosetta Neurons procedure at data scales from 1 to 50,000 images, once with real images and once with diffusion-generated images. As shown in Figure 35, at larger data scales of 25,000 to 50,000 images, generated images recover approximately 93% of the Rosetta Neuron count found with real images, with an overlap of roughly 0.8. Matching on generated images recovers most of the same correspondences, suggesting that distribution shift has a modest effect.

![](images/774770d6b9e58d44d129ac4ed908451305b63408fe05f72601226adcc1107ebc.jpg)

<details>
<summary>line</summary>

| # Tokens | # Rosetta Neurons |
| -------- | ----------------- |
| 1000     | 20000             |
| 10000    | 17000             |
| 100000   | 16000             |
| 1000000  | 14500             |
| 10000000 | 13500             |
| 100000000| 13000             |
</details>

![](images/365471b10dc9e754a3981934d990713089d82ce06d1569ca751faa82f1120660.jpg)

<details>
<summary>line</summary>

| # Tokens | Jaccard Index |
| -------- | ------------- |
| 10^4     | 0.1           |
| 10^5     | 0.4           |
| 10^6     | 0.5           |
| 10^7     | 0.8           |
| 10^8     | 0.9           |
</details>

Figure 33: Language matching stability as a function of dataset size. We vary the number of tokens used to match neurons between Pythia-6.9B and OPT-6.7B. Left: number of Rosetta Neurons discovered at each data scale. Right: overlap with the Rosetta Neuron set from the previous data scale. The discovered Rosetta Neuron set becomes increasingly stable as the token budget grows.

![](images/a3c2fd3f7bb8fb50981620a3e5cbd096cae6eb5bba37cefc5137d2452600919e.jpg)

<details>
<summary>line</summary>

| # Images | # Rosetta Neurons |
| -------- | ----------------- |
| 1        | 1600              |
| 10       | 1000              |
| 100      | 600               |
| 1000     | 650               |
| 10000    | 700               |
| 100000   | 750               |
</details>

![](images/d574a4ebb546a7b5a89e80a52cc529fe50567739caef7a79312d50ce94f84bd7.jpg)

<details>
<summary>line</summary>

| # Images | Jaccard Index |
| -------- | ------------- |
| 10^1     | 0.05          |
| 10^2     | 0.12          |
| 10^3     | 0.23          |
| 10^4     | 0.65          |
| 10^5     | 0.88          |
| 10^6     | 0.98          |
</details>

Figure 34: Diffusion-to-discriminative matching stability as a function of dataset size. We vary the number of generated images used to match neurons between pMF DiT-B/16 and OpenCLIP ViT-B/16. Left: number of Rosetta Neurons discovered at each data scale. Right: overlap with the Rosetta Neuron set from the previous data scale. Stability improves as the number of images used for neuron matching approaches 50,000.

![](images/acf035888c0f92e76e925f594fadac03c481bfcd6db32405315b1d0bb613315c.jpg)

<details>
<summary>line</summary>

| # Images | ImageNet Val Set | Diffusion-Generated Set |
| -------- | ---------------- | ----------------------- |
| 1        | 2500             | 2200                    |
| 10       | 2800             | 2700                    |
| 100      | 4200             | 3700                    |
| 1000     | 4700             | 4200                    |
| 10000    | 4800             | 4400                    |
| 100000   | 4800             | 4500                    |
</details>

![](images/c292473342a3c9826b5aed89c44de8adacfb5f61d5f5407858197501a8c3139e.jpg)

<details>
<summary>line</summary>

| # Images | Jaccard Index |
| -------- | ------------- |
| 1        | 0.1           |
| 10       | 0.2           |
| 100      | 0.4           |
| 1000     | 0.55          |
| 10000    | 0.65          |
| 100000   | 0.8           |
| 1000000  | 0.82          |
</details>

Figure 35: Effect of image distribution on vision model matching. We compare Rosetta Neuron matching between OpenCLIP ViT-B/16 and DINOv2 ViT-B/14 using real and diffusion-generated images. Left: number of Rosetta Neurons identified at each data scale. Right: Jaccard index between the Rosetta Neuron sets obtained from the two image distributions. At larger data scales, the two distributions yield similar numbers of matches and substantial overlap.

# H Additional Details on VLM-as-a-Judge

This section provides additional details on our VLM-as-a-judge setup for measuring neuron selectivity in vision models. We examine sensitivity to the number of top-activating examples shown to the VLM, validate the approach with a held-out prediction task, and finally show that the same qualitative selectivity trends hold for diffusion models and DINOv2.

# H.1 Detailed Experimental Setup

Collecting activations and top-k images for Rosetta neurons. Given a generative–discriminative model pair (e.g., DiT-B/16 and OpenCLIP ViT-B/16), we generate 50,000 images with the generative model and pass them through the discriminative model. We randomly sample 100 Rosetta Neuron pairs and evaluate each neuron in the pair independently. For the discriminative model, we record each sampled neuron’s patch-level activations over the 50,000 images. For each neuron and image, we compute a scalar activation score by averaging the neuron’s activation over all spatial patches. We rank images by this score and retrieve the top 20 activating images for VLM evaluation. We apply the same procedure to the corresponding generative-model neurons, using activations cached during image generation.

Collecting activations and top-k images for random neurons. For the non-Rosetta neuron baseline, we apply the same procedure independently within each model rather than using matched neuron pairs. In each model, we randomly sample 100 non-Rosetta neurons and record their patch-level activations over the 50,000 images. For each neuron and image, we compute a scalar activation score by averaging the neuron’s activation over all spatial patches. We rank images by this score and retrieve the top 20 activating images for VLM evaluation.

VLM judging. For each selected Rosetta Neuron, we form a grid of the top 20 activating images together with the activation heatmaps from one model. The composite image is provided to GPT-5.4 along with a prompt asking it to determine whether the neuron responds to a single coherent visual feature. We provide this prompt below. For each neuron, the VLM returns a binary monosemanticity judgment together with a short natural-language description of the inferred feature or set of features. We follow a similar procedure for the non-Rosetta neuron baseline. We report results for both Rosetta Neurons and the baseline in a model over five independent trials. For the Rosetta Neuron analysis, each trial consists of a disjoint random subset of 100 Rosetta Neurons. For the non-Rosetta baseline, each trial consists of disjoint random subsets of 100 neurons from the non-Rosetta set. We calculate 95% confidence intervals for the percentage identified as monosemantic in each setting.

You are evaluating whether a neuron in a vision model is monosemantic (responds to one visual feature) or polysemantic (responds to multiple unrelated visual features). Below are the top 20 images that most strongly activate this neuron. Each row shows: the original image, then the activation heatmap from a model and its heatmap overlay over the image. These maps show where the neuron fires (bright = strong activation). Focus on the heatmap regions — what visual feature (texture, shape, color, pattern, object part, object category, etc.) is consistently highlighted across images? Even if the objects in the images differ, the neuron may still be monosemantic if it consistently responds to the same low-level or mid-level visual feature (e.g., "striped patterns", "curved edges", "glossy surfaces"). It could be an abstract, high level concept, or it could be a common visual feature such as texture, which may not be easily nameable but is still a single unifying pattern.

Is there a single visual feature or concept that explains why this neuron fires in all of these images? Respond in JSON: { "is\_monosemantic": true/false, "description": "the shared visual feature if monosemantic, or why not if polysemantic" }

# H.2 Sensitivity to the Number of Top-k Activating Images

We ablate the number of top-activating images shown to the VLM judge. Specifically, we repeat the monosemanticity evaluation with k ∈ {2, 5, 10, 15, 20, 50}, constructing each composite from the top-k images and their corresponding activation maps and overlays. As shown in Figure 36, the monosemanticity rates decrease as more examples are shown, but the gap between Rosetta and non-Rosetta neurons remains stable across all values of k. The estimates begin to stabilize around k = 20, while using substantially fewer examples provides less evidence for judging whether a single coherent feature explains the neuron. We therefore use k = 20 in the main experiments.

Top-k Sensitivity of Monosemanticity Judgments   
![](images/c211f68a5522aa242735bae67735519810dde765cc9d4eddaad9d3d944a5df9f.jpg)

<details>
<summary>line</summary>

| # Top-Activating Images Shown to the VLM | Rosetta Neurons | Non-Rosetta Neurons |
| ---------------------------------------- | --------------- | ------------------- |
| 2                                        | 0.65            | 0.45                |
| 5                                        | 0.52            | 0.30                |
| 10                                       | 0.40            | 0.28                |
| 15                                       | 0.35            | 0.22                |
| 20                                       | 0.30            | 0.20                |
| 50                                       | 0.28            | 0.18                |
</details>

Figure 36: Effect of the number of topactivating images shown to the VLM judge.

# H.3 Validation of VLM-as-a-Judge as a Predictive Metric

Experimental setup. To validate our VLM-as-a-judge setup as a reliable metric, we design a prediction task that tests whether the VLM can recover and generalize the visual features that neurons fire for, rather than relying on spurious cues. We sample 100 random neurons from OpenCLIP ViT-B/16. For each neuron, the VLM is shown the top 20 most-activating images together with their activation maps in a composite grid. We first provide the system prompt described in Section H.1 to the model. We then construct a held-out test set consisting of the next 5 highest-activating images and 5 random images drawn from the bottom 50% of activations. These images are shuffled and presented as a labeled grid containing only the raw images. The VLM must predict which images would activate the neuron based only on the training composite. To do this, we provide the prior conversation based on the 20-image training composite in context, and then issue a test prompt asking the model to predict on the 10 test images. We provide this prompt below.

```jsonl
IMAGE 2 (TEST): The second composite image is a grid of 10 test images labeled A through J. Given the previous composite image you saw of top activating images and the corresponding activation maps, predict which of these ten test images will activate this neuron.
Respond in JSON:
{
"neuron_description": "brief description of what visual feature(s) the neuron responds to",
"predicted_activating": ["A", "C", ...],
"reasoning": "brief explanation of why you chose these images and rejected the others"
} 
```

VLM-as-a-Judge is predictive of neuron selectivity. Performance is evaluated over 5 independent trials, each using a disjoint random subset of 100 neurons. We report 95% confidence intervals for accuracy, precision, recall, and F1 in Table 5. VLM-as-a-judge performs meaningfully above a random baseline. The baseline chance is a random predictor that independently marks each test image as activating with probability 0.5. In this 5-positive/5-negative setup, its expected accuracy and precision are 0.5, while its expected Recall and F1 are 0.4995 and 0.4865, respectively.

<table><tr><td>Model</td><td>Accuracy</td><td>Precision</td><td>Recall</td><td>F1</td></tr><tr><td>VLM-as-a-Judge</td><td>0.792 ± 0.032</td><td>0.865 ± 0.042</td><td>0.698 ± 0.038</td><td>0.764 ± 0.037</td></tr><tr><td>Chance (expected)</td><td>0.500</td><td>0.500</td><td>0.500</td><td>0.487</td></tr></table>

Table 5: Validation of GPT-5.4 as a neuron selectivity judge. For each sampled OpenCLIP ViT-B/16 neuron, GPT-5.4 is shown the top 20 activating images with activation maps. It is then asked to identify the activating images in a shuffled set of 10 unseen raw images containing 5 positives and 5 negatives. We report 95% confidence intervals across 5 independent evaluation runs, with each run sampling 100 distinct neurons. We also report the expected metrics for a random predictor.

![](images/f4deedf252a99819c8d7b749a8120c0990807fe207d91a70dcc11832bfc0da76.jpg)

<details>
<summary>line</summary>

| Model Size (# Neurons) | Rosetta Neurons | Non-Rosetta Neurons |
| ----------------------- | --------------- | ------------------- |
| 5 × 10⁴                 | 0.38            | 0.30                |
| 1 × 10⁵                 | 0.40            | 0.22                |
| >1 × 10⁵                | 0.52            | 0.16                |
</details>

![](images/5842f13b4c068910ed18a8106c8406d723a50fae15d7e421ba1b27cb9a55dcc7.jpg)

<details>
<summary>line</summary>

| Model Size (# Neurons) | Blue Line Value | Red Line Value |
| ---------------------- | --------------- | -------------- |
| 5 × 10⁴                | ~0.8            | ~0.6           |
| 1 × 10⁵                | ~0.9            | ~0.4           |
</details>

Figure 37: VLM-judged monosemanticity rate in DINOv2 and diffusion models. Rosetta Neurons discovered from DINOv2–Diffusion matching runs exhibit increasing monosemanticity with scale, while non-Rosetta neurons become polysemantic according to the metric. This provides additional evidence for the Neuron Polarization Effect across vision model families.

# H.4 Results for Neuron Selectivity in Other Vision Models

In Section 5.1, we showed evidence for the Neuron Polarization Effect in OpenCLIP: Rosetta Neurons become more selective with scale, while the non-Rosetta population remains comparatively less selective. In this section, we apply the same VLM-as-a-judge setup to measure neuron monosemanticity in DINOv2 and diffusion models as they scale. We use the Rosetta Neurons discovered from the DINOv2–Diffusion matching runs. In Figure 37, we plot the fraction of neurons judged monosemantic as a function of model size. The monosemanticity fraction increases with model scale for Rosetta Neurons and decreases for non-Rosetta neurons in both families, consistent with the trend observed in OpenCLIP and language models. The monosemanticity rates for Rosetta Neurons in DINOv2 and diffusion are similar, as these neurons are identified from the same cross-family matching runs. In contrast, the non-Rosetta populations exhibit model-specific levels of polysemanticity.

# I Further Discussion and Limitations

In this section, we further contextualize the Rosetta Neuron scaling picture. We begin with a DINOv3 failure case, which helps clarify the conditions under which Rosetta Neuron scaling arises. We then discuss the challenge of operationalizing monosemanticity and polysemanticity, and relate our selectivity measures to these concepts.

# I.1 DINOv3 as an Informative Failure Case

A notable exception to the scaling behavior observed in our main experiments from Section 4 is DINOv3 (Siméoni et al., 2025), which does not exhibit a clear Rosetta Neuron scaling law. We repeat the discriminative-to-diffusion matching procedure from Section 4 with DINOv3, and report the resulting Rosetta Neuron counts alongside DINOv2 results in Figure 38. In contrast to DINOv2, which follows the trend observed in other vision model families, DI-NOv3 Rosetta Neuron counts do not follow a monotonic scaling trend. This deviation is consistent with the fact that DINOv3 modifies the DINOv2 training setup with additional constraints on intermediate representations, encouraging them to match statistics from earlier in training. We hypothesize that these constraints alter how neuron-level features are organized. More broadly, the absence of a clean trend in DINOv3 suggests that Rosetta Neuron scaling depends on the interaction between data, architecture, optimization, and training objective.

![](images/3318873b8b82b09f9385554f023023d555ba1c22309c372ddb4b10c823249d7b.jpg)

<details>
<summary>line</summary>

| Model Size (# Neurons) | DINOv2 | DINOv3 |
| ----------------------- | ------ | ------ |
| 5 × 10⁴                 | 1700   | 1600   |
| 1 × 10⁵                 | 2300   | 2300   |
| 2 × 10⁵                 | 4000   | 600    |
| 5 × 10⁵                 | -      | 2100   |
</details>

Figure 38: DINOv3 does not exhibit a clear Rosetta Neuron scaling law.

# I.2 Operationalizing Monosemanticity and Polysemanticity

To interpret neuron selectivity, we need a working definition of monosemanticity and polysemanticity (Bricken et al., 2023b). These are general semantic descriptions of representations: a unit is considered monosemantic if it responds to one coherent feature, and polysemantic if it responds to multiple unrelated features. These terms are related to, but distinct from the more formal notion of superposition (Elhage et al., 2022), which describes feature interference in activation space. Since the underlying feature directions are not directly available in real models, prior work has relied on proxies such as top-activating examples, output-space projections (Geva et al., 2021), activation maps (Bau et al., 2017), and VLM-based evaluations (Shaham et al., 2024). These judgments therefore depend on the input distribution, the evaluation procedure, and the level of semantic granularity. For example, a unit that responds to both cats and dogs may be viewed as monosemantic at the level of mammals, but polysemantic at the level of animal species.

Given this ambiguity, we do not treat monosemanticity and polysemanticity as fixed formal definitions. Our goal is instead to understand population-level trends for selectivity as models scale. To do this, we use modality-specific proxies such as vocabulary-space excess kurtosis for language models and VLM judgments for vision models. Additionally, qualitative visualizations of top-activating examples and activation maps show patterns consistent with our measurements. This evaluation does not settle the general problem of defining monosemanticity in real models, but it is well suited to the relative prediction studied here: Rosetta Neurons should become increasingly selective with scale compared with the non-Rosetta background. Developing more principled operational definitions remains an important direction for future work.

# J Compute Resources

All experiments were run on NVIDIA A100 80GB GPUs. Neuron matching experiments and related ablations used 8 GPUs per model pair. Matching between two models in the ∼100M parameter family takes around 30 minutes, while matching models in the ∼30B parameter family takes about 24 hours. This corresponds to roughly 4–192 A100 GPU-hours per model-pair matching run depending on the model scale. The continued pretraining of GPT2-1.5B takes around 30 minutes on 8 GPUs, corresponding to roughly 4 A100 GPU-hours per run. The rest of the experiments were run using a single GPU. We did not train any of the pretrained language or vision models used in the scaling-law experiments. These experiments use existing checkpoints.