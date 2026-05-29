# Beyond Interpretability: When, Why, and How Sparse Autoencoders Enable Label-Free Visual Steering

Gerasimos Chatzoudis1∗ Zhuowei Li1 Gemma E. Moran2 Hao Wang1 Dimitris N. Metaxas1

1Department of Computer Science, Rutgers University

2Department of Statistics, Rutgers University

{gc745, zl502, hw488, dnm}@cs.rutgers.edu gm845@stat.rutgers.edu

Sparse Autoencoders (SAEs) are increasingly used to interpret foundation models, but their role as an actionable intervention space remains less understood, especially in vision. We study whether sparse visual features can be used not only for post-hoc analysis, but also to steer frozen visionlanguage models. We introduce Visual Sparse Steering (VS2), a label-free method that trains a top-k SAE on unlabeled activations from a frozen CLIP image encoder and, at test time, constructs an interpretable steering vector by amplifying the input’s active sparse features and decoding the induced change. We show that this procedure admits a closed-form decomposition as centroid-deviation steering: each input is moved along its deviation from the SAE-learned centroid. The residual term is controlled exactly by the SAE’s per-sample reconstruction error, measured by FVU, yielding an FVU-based residual bound and motivating a reliability gate that falls back to zero-shot CLIP when SAE reconstruction is unreliable. With target-domain SAEs trained on unlabeled CLIP image-encoder activations, VS2 improves zero-shot accuracy across nine image-classification datasets, achieving gains up to +4.12% with less than 0.1% additional inference compute. Finally, a controlled upperbound study, VS2++, shows that selective amplification of sparse features can yield gains up to +21.44%, exposing a reconstruction-vs-task saliency gap: features salient for reconstruction need not align with features useful for downstream prediction.

# 1 Introduction

Sparse Autoencoders (SAEs) have become a central tool in mechanistic interpretability, where they decompose model activations into sparse, human-interpretable features [Bricken et al., 2023, Gao et al., 2024a]. Most work treats SAE latents as objects of analysis: they help explain what a model represents and which features activate on a given input. A key open question is whether these sparse features can move beyond post-hoc interpretation and serve as an actionable intervention space for downstream tasks.

This question is particularly important in vision-language models. Adapting frozen models such as CLIP often requires labeled data, prompt tuning, test-time optimization, or carefully constructed contrastive examples. Test-time adaptation methods can be effective, but typically rely on backpropagation through large encoders [Shu et al., 2022, Feng et al., 2023, Yoon et al., 2024]; steering vectors are cheaper, but usually require positive and negative anchors or other contrastive supervision [Turner et al., 2023, Zou et al., 2023, Liu et al., 2024]. In contrast, SAE features are learned from unlabeled activations and expose interpretable sparse directions in representation space. However, it remains unclear when such features are effective, reliable, or task-relevant for downstream vision tasks.

We study whether SAE sparse features can be used for label-free downstream steering. We focus on zero-shot image classification with frozen CLIP models and introduce Visual Sparse Steering (VS2), which trains a top-k SAE on unlabeled image activations and uses its active sparse features to construct an instance-specific, test-time steering vector. Given an input image, VS2 amplifies its active SAE features and decodes the induced change, producing a direction aligned with the SAE’s most salient reconstruction features for that input. This requires no labels, contrastive anchors, CLIP weight updates, or test-time backpropagation.

![](images/b1524a0d06c367f96b269d6f4f046b11d46cc731989a4d7edb1360f2a89af0af.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Image"] --> B["Input Image"]
    B --> C["VLued Output"]
    C --> D["x"]
    D --> E["SAE encoder"]
    D --> F["SAE decoder"]
    D --> G["SME encoder"]
    D --> H["SME decoder"]
    E --> I["× γ"]
    F --> I
    G --> J["× x̄"]
    H --> J
    I --> K["v = x̄' - x̄"]
    J --> K
    K --> L["Contrastive-SAE"]
    L --> M["x̄ = x + λv"]
    M --> N["Output"]
```
</details>

(a) VS2: Visual Sparse Steering w/o Contrastive Data

![](images/f278e7f16e06b57d3c4bd6808349dfc193d39f72269ca464169999cb142f455b.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    A["Input Image"] --> B["Feature Extraction"]
    B --> C["x"]
    B --> D["x_n"]
    C --> E["Conv-SAE"]
    D --> F["Conv-SAE"]
    E --> G["+"]
    F --> H["+"]
    G --> I["v = v̄_n - v̄_n"]
    H --> J["v̄_n"]
    I --> K["+"]
    J --> L["v̄_n"]
    K --> M["Output x̄ = x + λv"]
    L --> M
```
</details>

(b) ${ \mathrm { V } } { \mathrm { S } } 2 ^ { + + }$ : Selective Visual Sparse Steering   
Figure 1: Overview of VS2 and ${ \mathrm { V } } { \mathrm { S } } 2 ^ { + + }$ : VS2 constructs a steering vector by uniformly amplifying active sparse features, while ${ \mathrm { V } } { \mathrm { S } } 2 ^ { + + }$ selectively amplifies sparse features given external corpus.

Across nine datasets and two CLIP backbones, VS2 consistently improves zero-shot classification and matches or exceeds prompt-based test-time adaptation baselines while adding less than 0.1% inference compute. Beyond empirical gains, we show that VS2 admits an interpretable closed-form decomposition as centroid-deviation steering: each embedding is moved away from the SAE-learned population centroid, suppressing shared structure and amplifying what makes the image distinctive. This provides a geometric explanation for why a label-free intervention can improve classification.

A central challenge is reliability. When the SAE poorly reconstructs the target distribution, sparse features can become unreliable and steering can harm performance. We formalize this failure mode through fraction of variance unexplained (FVU), showing that the residual magnitude in the steering direction is controlled by the SAE’s per-sample reconstruction error. This motivates a label-free FVU-gated fallback that disables steering and reverts to zero-shot CLIP when reconstruction is unreliable.

Finally, we identify a deeper limitation of using SAEs for downstream tasks: features salient for reconstruction need not be the features most useful for prediction. We call this the reconstructionvs-task saliency gap. To quantify the headroom from selecting task-relevant features, we introduce VS2++ as a controlled selective-amplification study using retrieved neighbors and CLIP pseudolabels. The gap between VS2 and oracle VS2++ shows that SAE-based downstream steering is limited less by the absence of useful sparse features than by the difficulty of selecting the task-relevant subset reliably.

Contributions. We make five contributions: (i) we introduce VS2, a label-free SAE-based steering method for frozen vision-language models; (ii) we show consistent zero-shot classification gains across nine datasets and two CLIP backbones with negligible inference overhead; (iii) we derive a closed-form centroid-deviation interpretation with an FVU-based faithfulness guarantee; (iv) we use FVU as a per-sample reliability gate under distribution shift; and (v) we identify the reconstructionvs-task saliency gap through VS2++ and prototype-aligned SAE training.

# 2 Related Work

Mechanistic Interpretability and Sparse Autoencoders. Several traditional approaches exist for interpretability in vision models, including feature visualization [Simonyan et al., 2014, Zeiler and Fergus, 2014, Olah et al., 2017] and network dissection [Bau et al., 2017, Oikarinen and Weng, 2022]. Mechanistic interpretability seeks to systematically analyze and understand neural networks [Elhage et al., 2021, Olah et al., 2020], but it faces challenges due to polysemantic neurons i.e. units that activate in response to multiple, seemingly unrelated inputs [Elhage et al., 2022]. This phenomenon arises from superposition, where networks encode more features than the available dimensions allow, forcing different concepts to share the same activations [Elhage et al., 2022]. Sparse Autoencoders (SAEs) have been explored to mitigate superposition by applying sparse dictionary learning to model internals [Bricken et al., 2023]. Recent efforts have leveraged SAEs to uncover interpretable features within LLMs [Templeton, 2024, Cunningham et al., 2023, Gao et al., 2024a]. Joshi et al. [2025] train SAEs on embedding differences to disentangle multiple concept shifts, enabling precise interventions in model activations without requiring direct supervision. While these methods focus on language models, our work extends sparse steering to the vision domain, where applications of SAEs to vision models remain comparatively underexplored.

While SAEs have been explored for feature analysis, generative modeling, and concept disentanglement in the visual domain [Bhalla et al., 2024, Stevens et al., 2025, Surkov et al., 2024, Fel et al., 2025, Thasarathan et al., 2025], these works focus primarily on interpretability rather than downstream performance. In contrast, our method leverages SAEs to actively steer vision models in a label-free, test-time setting to improve classification. Related efforts like Joseph et al. [2025] study CLIP’s steerability on typographic attacks, while Patch-SAE [Lim et al., 2024] improves classification accuracy via class-conditioned latent masking. Our approach, on the other hand, requires no class labels and avoids class-based activation aggregation during training, enabling broader applicability without reliance on external supervision or gradient updates.

Steering vectors. Steering Vector (SV) methods [Turner et al., 2023, Park et al., 2023, Hernandez et al., 2024, Mikolov et al., 2013], also known as representation engineering [Zou et al., 2023], construct a directional task vector and apply it in the latent space to change the target model’s behavior at inference time. In LLMs/MLLMs, SVs are used to enhance security [Liu et al., 2024], truthfulness [Li et al., 2023], reduce hallucinations [Li et al., 2025a], and improve efficiency [Li et al., 2025b]. Interestingly, prior work has shown that in VLMs, visual cues are influenced by language, and that biases in the model’s response can be steered through simple natural language prompts [Gavrikov et al., 2025]. In contrast, we focus on steering latent representations in vision models without any language input. Recent work has demonstrated that sparse representations can improve interpretability and disentanglement in steering directions [Bayat et al., 2025, Makelov, 2024]. Unlike these methods, which rely on supervised contrastive examples or training data, our approach discovers meaningful sparse directions in vision models without requiring labeled positive/negative concept pairs, making it more adaptable to general visual representations.

Test-Time Adaptation. Recent advances in improving the generalization of vision-language models have introduced a variety of prompt-tuning approaches, such as CoOp [Zhou et al., 2022b], Co-CoOp [Zhou et al., 2022a], and MaPLe [Khattak et al., 2023], along with adapter-based techniques like Tip-Adapter [Zhang et al., 2021] and CLIP-Adapter [Gao et al., 2024b]. Despite their effectiveness, these methods typically assume access to a small number of labeled target-domain examples (often in a few-shot setting). In contrast, Test-Time Adaptation methods aim to improve robustness under distribution shift by adapting a pre-trained model using only unlabeled test samples at inference time [Liu et al., 2021, Sun et al., 2020, Wang et al., 2020, Gao et al., 2022]. Specifically, Test-Time Prompt Tuning (TPT) [Shu et al., 2022] introduced entropy-minimizing prompt optimization over augmented views, inspiring subsequent methods such as DiffTPT [Feng et al., 2023], which leverages diffusion-based augmentations, and C-TPT [Yoon et al., 2024], which incorporates calibration-aware objectives. Although effective, these strategies typically require optimization loops, multiple augmentations, and backpropagation through large encoders, resulting in substantial computational and memory overhead at inference. In contrast, VS2 performs a single forward pass with sparse SAE-guided steering, avoiding test-time optimization while still providing measurable gains.

# 3 Method

Preliminaries (top-k SAE). Let $p _ { \mathrm { t r a i n } }$ denote the empirical distribution over CLS-token activations $\pmb { x } \in \mathbb { R } ^ { d }$ of a frozen vision encoder, and let $\mu : = \mathbb { E } _ { p _ { \mathrm { t r a i n } } } [ \pmb { x } ]$ be its mean. A top-k Sparse Autoencoder (SAE) with latent width n encodes x into sparse features $\boldsymbol { c } \in \mathbb { R } ^ { n }$ and reconstructs $\widetilde { \pmb { x } } \in \mathbb { R } ^ { d }$ via

$$
\boldsymbol {c} = \operatorname{TopK} _ {k} \left(\boldsymbol {W} _ {\text {enc}} (\boldsymbol {x} - \boldsymbol {b} _ {\text {pre}})\right), \tag {1}
$$

$$
\tilde {\boldsymbol {x}} = D _ {c} (\boldsymbol {c}) := \boldsymbol {W} _ {\text {dec}} \boldsymbol {c} + \boldsymbol {b} _ {\text {pre}},
$$

where $\mathrm { T o p K } _ { k } ( \cdot )$ retains the k entries of largest absolute value and zeros the rest. Following Gao et al. [2024a], the SAE is trained to minimize $\| \bar { \boldsymbol { \mathbf { x } } } - \tilde { \boldsymbol { \mathbf { x } } } \| _ { 2 } ^ { 2 }$ over $p _ { \mathrm { t r a i n } }$ with $ { b _ { \mathrm { p r e } } }$ initialized to the empirical mean of x. We write $\pmb { d } _ { i } \in \mathbb { R } ^ { d }$ for the i-th column of $W _ { \mathrm { d e c } }$ and $\mathbb { S } _ { k } ( \pmb { x } ) : = \{ i : c _ { i } \neq 0 \}$ for the active set, so $| \mathbb { S } _ { k } ( \pmb { x } ) | = k$ and $\begin{array} { r } { W _ { \mathrm { d e c } } \pmb { c } = \sum _ { i \in \mathbb { S } _ { k } ( \pmb { x } ) } c _ { i } \pmb { d } _ { i } } \end{array}$ .

# 3.1 Visual Sparse Steering (VS2)

Classical steering-vector methods specify a direction by contrasting curated positive and negative anchor examples [Turner et al., 2023, Zou et al., 2023]. VS2 instead derives a per-instance steering direction directly from the vision encoder’s own internal activations: it amplifies the SAE’s active sparse features, decodes, and subtracts the un-amplified reconstruction. The construction has an exact closed-form interpretation as centroid-deviation steering and a quantitative faithfulness guarantee tied to the SAE’s reconstruction error (Proposition 3.1). We present VS2 as the first systematic study of SAE features as an actionable, label-free control mechanism for vision foundation models, going beyond their established role as interpretability tools.

Construction. Given a frozen vision encoder and an SAE trained on its CLS-token activations, VS2 applies three steps for amplification $\gamma > 1$ and steering scale $\lambda > 0 \mathrm { : }$

$$
\boldsymbol {c} = \operatorname{TopK} _ {k} \left(\boldsymbol {W} _ {\text {enc}} (\boldsymbol {x} - \boldsymbol {b} _ {\text {pre}})\right),
$$

$$
\boldsymbol {v} = D _ {c} (\gamma \boldsymbol {c}) - D _ {c} (\boldsymbol {c}), \tag {2}
$$

$$
\hat {\boldsymbol {x}} = \left(\boldsymbol {x} + \lambda \boldsymbol {v}\right) \frac {\| \boldsymbol {x} \| _ {2}}{\| \boldsymbol {x} + \lambda \boldsymbol {v} \| _ {2}}.
$$

The first line encodes the embedding into its sparse features. The second steers: it amplifies the active sparse features by γ, decodes, and subtracts the un-amplified reconstruction; the difference cancels the SAE’s centering bias and isolates the contribution of the active features. The third rescales to preserve the embedding norm, so the steered point remains comparable to zero-shot CLIP at classification time. The intervention space is the sparse decoder basis $\{ d _ { i } \} _ { i \in \mathbb { S } _ { k } ( { \pmb x } ) }$ , k directions extracted from the encoder’s own activation geometry, many of which empirically align with coherent visual concepts (Section A). Because $\mathbb { S } _ { k } ( \overline { { \boldsymbol { x } } } )$ is input-dependent and the top-k support already identifies features the encoder finds most informative for reconstructing $\mathbf { \Delta } x - b _ { \mathrm { p r e } }$ , the steering direction is per-instance and label-free, with no contrastive anchors required. Figure 1 (left) shows the schematic.

# 3.2 A Geometric View of VS2

We rewrite equation 2 in closed form. Let $\pmb { \epsilon } ( \pmb { x } ) : = \tilde { \pmb { x } } - \pmb { x }$ be the per-sample reconstruction residual and $\mathrm { F V U } ( { \pmb x } ) \dot { : } = | | { \pmb \epsilon } ( { \pmb x } ) | | _ { 2 } ^ { 2 } / | | { \pmb x } - { \pmb \mu } | | _ { 2 } ^ { 2 }$ the per-sample fraction of variance unexplained.

Proposition 3.1 (Centroid-deviation steering with faithfulness guarantee). For $\gamma > 1 , \lambda > 0 ,$ , and $\alpha : = \lambda ( \gamma - 1 )$ ,

$$
\boxed {\boldsymbol {x} + \lambda \boldsymbol {v} = \boldsymbol {x} + \alpha (\boldsymbol {x} - \boldsymbol {b} _ {\text { pre }}) + \alpha \epsilon (\boldsymbol {x}),} \tag {3}
$$

with residual magnitude $\| \alpha \epsilon ( \pmb { x } ) \| _ { 2 } = \alpha \sqrt { \mathrm { F V U } ( \pmb { x } ) } \| \pmb { x } - \pmb { \mu } \| _ { 2 }$ .

Proof The decoder is affine, so ${ \pmb v } = D _ { c } ( \gamma { \pmb c } ) - D _ { c } ( { \pmb c } ) = ( \gamma - 1 ) { \pmb W } _ { \mathrm { d e c } } { \pmb c }$ (the bias cancels). Substituting $W _ { \mathrm { d e c } } { \boldsymbol { c } } = \left( { \pmb x } - { \pmb b } _ { \mathrm { p r e } } \right) + { \boldsymbol \epsilon } ( { \pmb x } )$ from equation 1 and scaling by λ gives equation 3. The residual magnitude is direct from the definition of FVU.

The steered embedding is the input plus an amplification of its deviation from the SAE-learned centroid $\begin{array} { r } { b _ { \mathrm { p r e } } \approx \pmb { \mu } _ { \mathrm { ~ } } } \end{array}$ plus a residual of magnitude $\alpha { \sqrt { \operatorname { F V U } ( \pmb { x } ) } } \| \pmb { x } - \pmb { \mu } \| _ { 2 }$ . We highlight three observations.

• Centroid-deviation steering. Population-mean structure (common backgrounds, mean texture, dataset-wide directions) is absorbed into $b _ { \mathrm { p r e } }$ and subtracted; what remains amplifies what makes the input distinctive.   
• Non-uniform direction from a scalar amplification. Although $\gamma$ is scalar, $\begin{array} { r l } { v } & { { } = } \end{array}$ $\begin{array} { r } { ( \gamma - 1 ) \sum _ { i \in \mathbb { S } _ { k } ( { \pmb x } ) } c _ { i } { \pmb d } _ { i } } \end{array}$ weights each active feature by its own activation; top-k has already filtered features with smaller pre-activations.   
• Faithfulness from FVU. VS2 preserves x and admits the residual only at scale $\alpha ,$ , in contrast to using the SAE reconstruction ${ \tilde { \pmb { x } } } = { \pmb { x } } + { \pmb { \epsilon } }$ directly, which would ingest the full residual. The signal-to-noise ratio $\| \pmb { x } - \pmb { b } _ { \mathrm { p r e } } \| _ { 2 } / \| \pmb { \epsilon } ( \pmb { x } ) \| _ { 2 } \approx 1 / \sqrt { \mathrm { F V U } ( \pmb { x } ) }$ (under faithful centering) makes per-sample FVU the natural reliability gate (Section 4.3).

Remark 3.1 (Toward selective amplification). Replacing the scalar $\gamma$ by a vector $\gamma \in \mathbb { R } ^ { n }$ via $\mathbf { { c } } ^ { \prime } = \gamma \odot { \boldsymbol { c } }$ yields $\begin{array} { r } { \pmb { v } = \sum _ { i \in \mathbb { S } _ { k } ( \pmb { x } ) } ( \gamma _ { i } - 1 ) c _ { i } \pmb { d } _ { i } } \end{array}$ , which spans the full $| \mathbb { S } _ { k } ( \pmb { x } )$ |-dimensional decoder subspace rather than a single direction. ${ \mathrm { V } } { \mathrm { S } } 2 ^ { + + }$ realizes a related extension by replacing c itself with contrastive sparse features aggregated from retrieved neighbors.

Table 1: Benchmarking Visual Sparse Steering. Left: zero-shot top-1 accuracy (%) with and without retrieval on CIFAR-100, CUB-200, and Tiny-ImageNet using ViT-B/32 and ViT-B/16. Right: TTA comparison and inference-time compute for ViT-B/32.   
(a) Main accuracy results 

<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR-100</td><td colspan="2">CUB-200</td><td colspan="2">Tiny-IN</td></tr><tr><td>ViT-B/32</td><td>ViT-B/16</td><td>ViT-B/32</td><td>ViT-B/16</td><td>ViT-B/32</td><td>ViT-B/16</td></tr><tr><td colspan="7">Label-free, unsupervised steering</td></tr><tr><td> $CLIP_{ZS}$ </td><td>61.07 (0)</td><td>63.96 (0)</td><td>51.76 (0)</td><td>55.06 (0)</td><td>56.64 (0)</td><td>61.08 (0)</td></tr><tr><td> $SAE_{REC}^A$ </td><td>58.01 (-3.06)</td><td>64.05 (+0.09)</td><td>47.45 (-4.31)</td><td>51.81 (-3.25)</td><td>30.56 (-26.08)</td><td>52.96 (-8.12)</td></tr><tr><td> $SAE_{REC}^F$ </td><td>58.22 (-2.85)</td><td>63.42 (-0.54)</td><td>48.08 (-3.68)</td><td>51.43 (-3.63)</td><td>36.33 (-20.31)</td><td>54.84 (-6.24)</td></tr><tr><td> $SAE_{+ + \gamma}^{++\gamma}$ </td><td>62.69 (+1.62)</td><td>66.81 (+2.85)</td><td>49.41 (-2.35)</td><td>53.28 (-1.78)</td><td>39.49 (-17.15)</td><td>58.81 (-2.27)</td></tr><tr><td> $VS2^{+}$ </td><td>64.52 (+3.45)</td><td>68.08 (+4.12)</td><td>52.69 (+0.93)</td><td>56.14 (+1.08)</td><td>58.14 (+1.50)</td><td>62.92 (+1.84)</td></tr><tr><td colspan="7">RAG-enhanced, oracle neighbor labels</td></tr><tr><td>Weighted RAG</td><td>76.43 (+15.36)</td><td>69.78 (+5.82)</td><td>58.32 (+6.56)</td><td>60.65 (+5.59)</td><td>72.53 (+15.89)</td><td>67.84 (+6.75)</td></tr><tr><td>CLIP Steering</td><td>81.85 (+20.78)</td><td>84.12 (+20.16)</td><td>56.42 (+4.66)</td><td>61.51 (+6.45)</td><td>80.38 (+23.74)</td><td>84.07 (+22.99)</td></tr><tr><td> $VS2^{++}$ </td><td>81.95 (+20.88)</td><td>85.40 (+21.44)</td><td>58.84 (+7.08)</td><td>61.91 (+6.85)</td><td>73.27 (+16.63)</td><td>81.55 (+20.47)</td></tr><tr><td colspan="7">RAG-enhanced, CLIP pseudo-labels</td></tr><tr><td>CLIP Steering</td><td>77.22 (+16.15)</td><td>78.97 (+15.01)</td><td>47.89 (-3.87)</td><td>53.95 (-1.11)</td><td>74.06 (+17.42)</td><td>76.79 (+15.71)</td></tr><tr><td> $VS2^{++}$ </td><td>77.11 (+16.04)</td><td>79.14 (+15.18)</td><td>52.81 (+1.05)</td><td>57.02 (+1.96)</td><td>72.12 (+15.48)</td><td>76.92 (+15.84)</td></tr></table>

(b) TTA comparison 

<table><tr><td>Method</td><td>CIFAR</td><td>CUB</td><td>Tiny</td></tr><tr><td>ZS</td><td>61.07</td><td>51.76</td><td>56.64</td></tr><tr><td>Ens</td><td>63.66</td><td>51.54</td><td>61.39</td></tr><tr><td>TPT</td><td>64.09</td><td>51.83</td><td>62.77</td></tr><tr><td>C-TPT</td><td>64.86</td><td>52.54</td><td>63.20</td></tr><tr><td>Diff-TPT</td><td>63.04</td><td>52.80</td><td>61.60</td></tr><tr><td>VS2</td><td>64.52</td><td>52.69</td><td>58.14</td></tr><tr><td>VS2 +Ens</td><td>65.48</td><td>52.47</td><td>62.79</td></tr><tr><td>VS2 +Ens+Aug</td><td>65.80</td><td>51.54</td><td>62.59</td></tr></table>

(c) Inference compute 

<table><tr><td>Method</td><td>GFLOPs</td><td>GMACs</td><td>Params</td></tr><tr><td>CLIP</td><td>8.7295</td><td>4.3623</td><td>87.46M</td></tr><tr><td>LoRA</td><td>8.7885</td><td>4.3918</td><td>88.05M</td></tr><tr><td>VS2</td><td>8.7342</td><td>4.3647</td><td>92.18M</td></tr></table>

# 3.3 Headroom Analysis via Selective Amplification $( \mathbf { V } \mathbf { S } 2 ^ { + + } )$

Remark 3.1 noted that scalar amplification collapses the steering direction to a single direction within the decoder subspace. To measure the accuracy headroom available to selective, per-feature amplification, we introduce ${ \mathrm { V } } { \mathrm { S } } 2 ^ { + + }$ as a controlled study: it constructs a contrastive direction from retrieved neighbors using CLIP pseudo-labels (and, as an upper bound, ground-truth labels). VS2++ is not a competing deployment-ready method but an instrument for quantifying this headroom.

Pseudo-labeled neighbor groupretrieve the N nearest neighbors n a query embedding in cosine similarity $\scriptstyle { \pmb { x } } _ { q }$ and an unlabeled corpus  assign each a CLIP pseu $\{ { \pmb x } _ { i } \} _ { i = 1 } ^ { M } .$ $\mathbb { N } ( \pmb { x } _ { q } )$ ${ \hat { y } } _ { i } : = \arg \operatorname* { m a x } _ { y } P ( y \mid \mathbf { x } _ { i } ) { \bar { } }$ . Let ${ \hat { y } } _ { q } : = \operatorname { a r g m a x } _ { y } P ( y \mid \mathbf { x } _ { q } )$ be the query’s own pseudo-label, and partition $\mathbb { N } ( \pmb { x } _ { q } )$ by pseudo-label agreement with $\textstyle { \pmb { x } } _ { q } .$

$$
\mathbb {S} ^ {+} := \{\boldsymbol {x} _ {i} \in \mathbb {N} (\boldsymbol {x} _ {q}): \hat {y} _ {i} = \hat {y} _ {q} \}, \quad \mathbb {S} ^ {-} := \mathbb {N} (\boldsymbol {x} _ {q}) \setminus \mathbb {S} ^ {+}. \tag {4}
$$

Contrastive steering direction. For each $\pmb { x } _ { i } \in \mathbb { N } ( \pmb { x } _ { q } )$ , let $c _ { i }$ denote its sparse features and ${ \mathbf { } } v _ { i }$ its VS2 direction. Define the group-averaged sparse features $\begin{array} { r } { \bar { \pmb { c } } ^ { \pm } : = \frac { 1 } { | \mathbb { S } ^ { \pm } | } \sum _ { \pmb { x } _ { i } \in \mathbb { S } ^ { \pm } } \pmb { c } _ { i } \in \mathbb { R } ^ { n } } \end{array}$ . The ${ \mathrm { V } } { \mathrm { S } } 2 ^ { + + }$ contrastive steering vector is

$$
\bar {\boldsymbol {v}} = \frac {1}{| \mathbb {S} ^ {+} |} \sum_ {\boldsymbol {x} _ {i} \in \mathbb {S} ^ {+}} \boldsymbol {v} _ {i} - \frac {1}{| \mathbb {S} ^ {-} |} \sum_ {\boldsymbol {x} _ {j} \in \mathbb {S} ^ {-}} \boldsymbol {v} _ {j} \tag {5}
$$

$$
= (\gamma - 1) \boldsymbol {W} _ {\mathrm{dec}} \left(\bar {\boldsymbol {c}} ^ {+} - \bar {\boldsymbol {c}} ^ {-}\right),
$$

where the second equality uses $\pmb { v } _ { i } = ( \gamma - 1 ) \pmb { W } _ { \mathrm { d e c } } \pmb { c } _ { i }$ from Proposition 3.1 and linearity of $W _ { \mathrm { d e c } }$ . The query embedding is steered by replacing v with v¯ in step (3) of equation 2.

Selective amplification. ${ \mathrm { V } } { \mathrm { S } } 2 ^ { + + }$ thus anchors steering on features consistently activated within $\mathbb { S } ^ { + }$ while suppressing those highlighted by hard negatives from the same local neighborhood. Figure 1 (right) shows the schematic. The gap between VS2’s uniform amplification and $\mathrm { V S } 2 ^ { + + } \mathrm { s }$ selective amplification quantifies the reconstruction-vs-task saliency gap, which we identify as the central open problem for SAE-based downstream methods.

# 4 Experiments

We evaluate VS2 across nine image-classification datasets and two CLIP backbones. Our goal is not only to measure accuracy, but to understand when, why, and how SAE features can serve as a downstream steering space. We therefore organize the experiments around four questions: (i) can SAE sparse features improve zero-shot classification without labels, contrastive anchors, or test-time optimization? (ii) does the gain come from additive sparse steering rather than reconstruction alone? (iii) how much headroom remains if task-relevant sparse features can be selected? and (iv) can SAE reconstruction error predict when steering is unreliable under distribution shift?

We evaluate VS2 on CIFAR-100 [Krizhevsky et al., 2009], Tiny-ImageNet [Le and Yang, 2015], CUB-200 [Wah et al., 2011], and six additional out-of-distribution datasets used for reliability analysis: Aircraft [Maji et al., 2013], Food101 [Bossard et al., 2014], Flower102 [Nilsback and Zisserman, 2008], Caltech101 [Fei-Fei, 2004], SUN [Xiao et al., 2010], and EuroSAT [Helber et al., 2019]. For the vision-language model, we use CLIP [Radford et al., 2021] with ViT-B/32 and ViT-B/16 backbones. Unless otherwise specified, we train top-k SAEs on unlabeled in-domain training-split activations and apply steering to the final-layer [CLS] token. Training details are provided in Appendix C.

![](images/e55207ee2dc0335a04e6c1c0f8e9857cf4b3c4e16829abde700b9996d6c09dd0.jpg)

<details>
<summary>text_image</summary>

f155
rocket
f2490
telephone
f2732
apple
f2930
clock
</details>

(a) Interpretable SAE features.

![](images/f4e9cc3793a28cc70b3c50c316b784afa379164903f8cded194f956135452af0.jpg)

<details>
<summary>heatmap</summary>

| Dataset | Feature | Value (x) |
| :--- | :--- | :--- |
| CLIP: pine tree VS2: sea | f2865 | 0.97 |
| Cloud | f207 | 0.94 |
| Cloud | f620 | 0.88 |
| Cloud | f908 | 0.68 |
| Cloud | f1169 | 0.67 |
| Cloud | f2950 | 0.67 |
| pickup truck VS2: pickup truck | f1764 | 0.87 |
| Pickup truck VS2: pickup truck | f752 | 0.86 |
| Pickup truck VS2: pickup truck | f2862 | 0.80 |
| Pickup truck VS2: pickup truck | f205 | 0.76 |
| Pickup truck VS2: pickup truck | f2117 | 0.71 |
+ ...
</details>

(b) VS2-corrected examples using weighted interpretable features.   
Figure 2: Qualitative evidence for SAE-based steering. (a) Some SAE features group coherent visual concepts. (b) In CIFAR-100 errors corrected by VS2, top active latents often align with the recovered class, illustrating coarse semantic alignment rather than perfect monosemanticity.

# 4.1 VS2: Visual Sparse Steering

We first evaluate whether SAE sparse features can improve zero-shot image classification without labels, contrastive anchors, or test-time optimization. This experiment also isolates the mechanism of VS2: if SAEs are used naively as reconstruction modules, performance may degrade due to reconstruction error; if sparse features are only amplified and reconstructed, the intervention is not consistently beneficial. VS2 differs from these alternatives by using the SAE only to define an additive steering direction, while preserving the original CLIP embedding.

Baselines. We compare VS2 against baselines that isolate reconstruction and latent-amplification effects. We report: (1) the zero-shot CLIP model (CLIPZS); (2) ${ \mathrm { S A E } } _ { \mathrm { R E C } } ^ { \mathrm { F } }$ , which replaces the finallayer [CLS] token with its SAE reconstruction; and (3) ${ \mathrm { S A E _ { R E C } ^ { A } } }$ , which reconstructs [CLS] tokens from all layers. These baselines test whether SAE reconstruction alone improves classification. We additionally include $\mathrm { S A E } _ { \mathrm { R E C } } ^ { \mathrm { F + } \gamma }$ , which scales the top-k latent activations by a fixed factor $( \gamma = 1 . 5 )$ before reconstructing the final-layer [CLS] token. This baseline tests whether latent amplification alone is sufficient, without the additive steering formulation of VS2. Appendix G provides pseudocode for these variants. We also report comparisons to Splice [Bhalla et al., 2024] in Appendix I, which defines latents using an external vocabulary rather than learning an unsupervised SAE dictionary. Unless stated otherwise, VS2 applies steering to the final-layer [CLS] token; Appendix E analyzes steering across layers.

Results. Table 1 shows that SAE reconstruction alone usually hurts zero-shot performance, consistent with reconstruction error being directly injected into the CLIP embedding [Engels et al., 2025]. Amplifying sparse features before reconstruction can help in some cases, such as CIFAR-100, where $\dot { \mathrm { S A E } } _ { \mathrm { R E C } } ^ { \mathrm { F + } \gamma }$ improves over zero-shot CLIP by 1.62% and 2.85% for ViT-B/32 and ViT-B/16, respectively. However, the same baseline underperforms on CUB-200 and Tiny-ImageNet, suggesting that amplified reconstruction is not a reliable intervention by itself. In contrast, VS2 consistently improves over zero-shot CLIP across all three datasets and both backbones. It yields gains of 3.45% and 4.12% on CIFAR-100, 0.93% and 1.08% on CUB-200, and 1.50% and 1.84% on Tiny-ImageNet for ViT-B/32 and ViT-B/16, respectively. These results show that the benefit comes not from replacing CLIP embeddings with SAE reconstructions, but from using sparse features to define an additive steering direction. Finally, Figure 3 and Appendix B analyze sensitivity to the steering hyperparameters γ and λ. Across datasets, VS2 exhibits a broad region of near-optimal performance, indicating that the method is not overly sensitive to precise hyperparameter choices.

Qualitative feature evidence. Although our main goal is downstream steering rather than interpretability alone, we inspect whether the sparse features used by VS2 correspond to recognizable visual factors. Figure 2 shows that individual SAE latents can group coherent visual concepts and that, in examples corrected by VS2, the active latents often align with coarse visual evidence for the recovered class. Additional qualitative examples and analyses are provided in Appendix A.

![](images/24c594c1ef50c8a479dd8177ac1d64e4ff73fb3474be56eb7137d218abb76990.jpg)

<details>
<summary>heatmap</summary>

| Steering Vector Magnitude λ | Upweighting Factor of Sparse Features γ |
| ---------------------------- | -------------------------------------- |
| 0.0                          | 3.0                                    |
| 0.5                          | 2.9                                    |
| 1.0                          | 2.8                                    |
| 1.5                          | 2.7                                    |
| 2.0                          | 2.6                                    |
| 2.5                          | 2.5                                    |
| 3.0                          | 2.4                                    |
| 3.5                          | 2.3                                    |
| 4.0                          | 2.2                                    |
| 4.5                          | 2.1                                    |
| 5.0                          | 2.0                                    |
| 5.5                          | 1.9                                    |
| 6.0                          | 1.8                                    |
| 6.5                          | 1.7                                    |
| 7.0                          | 1.6                                    |
| 7.5                          | 1.5                                    |
| 8.0                          | 1.4                                    |
| 8.5                          | 1.3                                    |
| 9.0                          | 1.2                                    |
| 9.5                          | 1.1                                    |
| 10.0                         | 1.0                                    |
</details>

![](images/6ccc86010a91554c664bf9327f57036fca74961225c9e8b644d53898a2e8ca0d.jpg)

<details>
<summary>heatmap</summary>

VS2 Top-1 Accuracy Heatmap on CUB-200
|  | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 | 3.00 |
| 1.50 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 | 2.99 |
| 2.00 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 | 2.98 |
| 2.50 | 2.97 | 2.97 | 2.97 | 2.97 | 2.97 | 2.97 | 2.97 | 2.97 | 2.97 | 2.97 | 2.97 | 2.97 | 2.97 | 2.97 | 2.97 |
| 3.00 | 2.96 | 2.96 | 2.96 | 2.96 | 2.96 | 2.96 | 2.96 | 2.96 | 2.96 | 2.96 | 2.96 | 2.96 | 2.96 | 2.96 | 2.96 |
| 3.50 | 2.95 | 2.95 | 2.95 | 2.95 | 2.95 | 2.95 | 2.95 | 2.95 | 2.95 | 2.95 | 2.95 | 2.95 | 2.95 | 2.95 | 2.95 |
| 4.00 | 2.94 | 2.94 | 2.94 | 2.94 | 2.94 | 2.94 | 2.94 | 2.94 | 2.94 | 2.94 | 2.94 | 2.94 | 2.94 | 2.94 | 2.94 |
| ... (The image contains only a grid of values) with no explicit numerical labels provided in the code.
</details>

![](images/3f4534ee02663bcd99f6611e326080e507c17ef3e99464fa1f383bb028729a8a.jpg)  
Figure 3: Sensitivity of VS2 to sparse amplification γ and steering magnitude λ. Across datasets, VS2 exhibits a broad near-optimal region, typically when $\lambda \gamma \in [ 2 , 3 ]$ .

# 4.2 ${ \bf V } { \bf S } 2 ^ { + + }$ : Upper-Bound Analysis via Selective Amplification

The previous section shows that uniformly amplifying an input’s active SAE features can improve zero-shot classification. However, this raises a natural question: are all active sparse features equally useful for the downstream task? Since the SAE is trained for reconstruction rather than classification, its active features may include both task-relevant evidence and nuisance factors such as background, texture, or pose. This creates a potential mismatch between reconstruction saliency and task saliency. To quantify the headroom available if task-relevant sparse features could be selected reliably, we introduce $\mathrm { \dot { V } S 2 ^ { + + } }$ as a controlled upper-bound analysis based on selective amplification.

Concretely, for each test image we retrieve the top-50 nearest neighbors from the dataset’s unlabeled training split using DINOv2 [Oquab et al., 2023]. We then form pseudo-positive and pseudo-negative groups using CLIP predictions: neighbors with the most frequent pseudo-label constitute positives, while the remaining neighbors are treated as negatives. We construct a contrastive steering direction in SAE feature space by subtracting the average negative sparse activations from the average positive sparse activations. This corresponds to the non-oracle setting in Table 1. For comparison, we also report an oracle upper bound in which positives and negatives are defined using the ground-truth labels of the retrieved neighbors. We emphasize that ${ \mathrm { V } } { \mathrm { S } } { \bar { 2 } } ^ { + + }$ is not intended as a deployment-ready replacement for VS2. Instead, it is an analysis tool for measuring how much accuracy is left unrealized by uniform amplification. As baselines, we include a weighted Retrieval Augmented Generation (RAG) pipeline [Lewis et al., 2020], which combines the query embedding with a weighted average of retrieved CLIP embeddings (Appendix D), and a non-SAE contrastive steering baseline that computes the difference between the mean CLIP embeddings of the positive and negative groups. These comparisons distinguish the effect of retrieval alone, CLIP-space contrastive steering, and SAE-space selective amplification.

Results. Table 1 reports ${ \mathrm { V } } { \mathrm { S } } 2 ^ { + + }$ under oracle and non-oracle positive/negative sets. In the oracle setting, ${ \mathrm { V } } { \mathrm { S } } 2 ^ { + + }$ achieves gains of up to 21.44% on CIFAR-100, 7.08% on CUB-200, and 20.47% on Tiny-ImageNet. These large gains show that sparse feature selection has substantial headroom beyond uniform amplification, and support our claim that the main limitation of SAE-based downstream steering is not the absence of useful sparse features, but the difficulty of selecting the task-relevant subset. Weighted RAG is generally weaker than contrastive steering on CIFAR-100 and Tiny-ImageNet while remaining competitive on CUB-200, suggesting that directional offsets are often more informative than proximity alone. ${ \mathrm { V } } { \mathrm { S } } 2 ^ { + + }$ is competitive with CLIP-space steering, although it is not uniformly best, for example on Tiny-ImageNet with ViT-B/32. In the non-oracle setting, performance drops due to pseudo-label noise, but $\bar { \mathrm { V } } \bar { \mathrm { S } } 2 ^ { + + }$ still outperforms the CLIP steering-vector baseline in four of six configurations and avoids negative transfer on CUB-200. Overall, these results identify the reconstruction-vs-task saliency gap as a central challenge for SAE-based downstream interventions and motivate future work on robust sparse-feature selection under weak supervision. Appendix J ablates the number of retrieved neighbors.

Fine-grained class gains. Beyond aggregate accuracy, we analyze which classes benefit most from sparse steering. We compute per-class top-1 accuracy on CIFAR-100 using a ViT-B/32 backbone and report the largest gains in Table 2. The largest improvements occur on visually confusable categories: VS2 improves accuracy by up to 25% absolute on classes such as tractor, forest, and man, where the dominant zero-shot CLIP errors involve nearby visual or semantic alternatives such as lawn mower, pine tree, and boy. With high-quality retrieved neighbors, VS2++ reaches gains of up to 38% absolute on similarly confusable cases such as spider, tiger, and flatfish. This pattern suggests that sparse steering does not merely shift predictions uniformly across classes. Instead, it appears most helpful when the zero-shot decision is close to a visually related class and the intervention can amplify image-specific evidence that separates the correct class from its nearest confounder. This is consistent with the centroid-deviation view of VS2: steering away from shared population structure can emphasize distinctive visual attributes, while selective amplification in $\bar { \mathrm { V } } \bar { \mathrm { S } } \bar { 2 } ^ { + + }$ can further improve performance when task-relevant sparse features are identified reliably. A full per-class breakdown is provided in Table 15 in Appendix K.

Table 2: Top-5 class gains on CIFAR-100. Top-1 accuracy with absolute gain over zero-shot CLIP. “Mislabel” denotes the most common zero-shot CLIP confusion corrected by each method. 

<table><tr><td colspan="5">VS2</td><td colspan="5">VS2++</td></tr><tr><td>Class</td><td>ZS</td><td>VS2</td><td>Mislabel</td><td>Visual confusion</td><td>Class</td><td>ZS</td><td>VS2++</td><td>Mislabel</td><td>Visual confusion</td></tr><tr><td rowspan="2">tractor</td><td rowspan="2">55.0</td><td rowspan="2">80.0↑(+25.0)</td><td rowspan="2">lawn mower</td><td rowspan="2">wheeled farm machines</td><td>spider</td><td>53.0</td><td>91.0↑(+38.0)</td><td>bee</td><td rowspan="3">small dark silhouette stripes vs. mane unclear</td></tr><tr><td>tiger</td><td>48.0</td><td>84.0↑(+36.0)</td><td>lion</td></tr><tr><td>forest</td><td>35.0</td><td>58.0↑(+23.0)</td><td>pine tree</td><td>dense conifer canopy</td><td></td><td></td><td></td><td></td></tr><tr><td>man</td><td>53.0</td><td>74.0↑(+21.0)</td><td>boy</td><td>weak age cue at 32×32</td><td>flatfish</td><td>49.0</td><td>85.0↑(+36.0)</td><td>whale</td><td>flat marine shape on blue</td></tr><tr><td>bus</td><td>51.0</td><td>68.0↑(+17.0)</td><td>pickup</td><td>boxy vehicle silhouette</td><td>possum</td><td>30.0</td><td>65.0↑(+35.0)</td><td>hamster</td><td rowspan="2">small brown mammal legs often invisible</td></tr><tr><td>snake</td><td>61.0</td><td>76.0↑(+15.0)</td><td>worm</td><td>elongated legless body</td><td>lizard</td><td>39.0</td><td>74.0↑(+35.0)</td><td>snake</td></tr></table>

Table 3: FVU-gated fallback under distribution shift. Each cell reports top-1 accuracy with coverage, i.e., the fraction of samples for which steering is applied, in parentheses. The gate disables steering when per-sample FVU exceeds a threshold calibrated without labels. 

<table><tr><td rowspan="2">Method</td><td colspan="3">In Distribution</td><td colspan="6">Out of Distribution</td></tr><tr><td>CIFAR-100</td><td>CUB-200</td><td>Tiny-IN</td><td>Aircraft</td><td>Food101</td><td>Flower102</td><td>Caltech101</td><td>SUN</td><td>EuroSAT</td></tr><tr><td>Baseline (CLIP $_{ZS}$ )</td><td>61.07</td><td>51.76</td><td>56.64</td><td>24.87</td><td>80.72</td><td>63.70</td><td>78.53</td><td>51.79</td><td>31.61</td></tr><tr><td>VS2 (Target-domain SAE, ungated)</td><td>64.52</td><td>52.69</td><td>58.14</td><td>27.69</td><td>81.20</td><td>64.81</td><td>80.95</td><td>54.80</td><td>34.65</td></tr><tr><td>VS2 (Generalized SAE, ungated)</td><td>64.63</td><td>48.81</td><td>59.73</td><td>18.57</td><td>81.06</td><td>64.11</td><td>79.72</td><td>45.78</td><td>30.76</td></tr><tr><td>VS2 + FVU gate ( $q=0.90$ )</td><td>64.27 (99.93%)</td><td>51.76 (0.0%)</td><td>59.29 (85.0%)</td><td>24.87 (0.00%)</td><td>80.72 (0.00%)</td><td>63.70 (0.11%)</td><td>78.44 (2.30%)</td><td>51.80 (0.11%)</td><td>30.39 (54.20%)</td></tr><tr><td>VS2 + FVU gate ( $q=0.95$ )</td><td>64.62 (99.97%)</td><td>51.76 (0.04%)</td><td>58.44 (92.16%)</td><td>24.87 (0.00%)</td><td>80.71 (0.02%)</td><td>63.72 (1.32%)</td><td>78.52 (5.51%)</td><td>51.80 (0.49%)</td><td>31.39 (63.15%)</td></tr><tr><td>VS2 + FVU gate ( $q=0.99$ )</td><td>64.63 (100%)</td><td>51.79 (0.62%)</td><td>58.65 (98.63%)</td><td>24.87 (0.09%)</td><td>80.70 (0.89%)</td><td>64.34 (22.72%)</td><td>78.76 (23.82%)</td><td>51.92 (6.97%)</td><td>31.57 (85.52%)</td></tr><tr><td>VS2 + FVU gate ( $q=0.995$ )</td><td>64.63 (100%)</td><td>51.83 (1.54%)</td><td>58.64 (99.34%)</td><td>24.87 (0.21%)</td><td>80.69 (3.10%)</td><td>64.21 (40.62%)</td><td>78.78 (35.26%)</td><td>52.11 (15.28%)</td><td>31.65 (92.09%)</td></tr></table>

# 4.3 SAEs as a Reliability Diagnostic for Safe Visual Sparse Steering

The previous experiments show that SAE features can improve classification when the SAE faithfully represents the target distribution. However, this condition is not guaranteed in deployment: a fixed SAE may be applied to samples whose activations differ from those seen during SAE training. In this case, the sparse features may no longer provide a faithful basis for steering, and the same intervention that improves accuracy in-domain can become harmful. We therefore ask whether the SAE’s own reconstruction error can serve as a label-free diagnostic for deciding when to apply steering.

Relaxation: a generalized SAE. Our main experiments use SAEs trained on unlabeled in-domain training-split activations for each dataset. This isolates the effect of sparse steering under favorable reconstruction conditions. To approximate a more realistic deployment setting, we train a single generalized SAE on the union of the unlabeled training splits of CIFAR-100, CUB-200, and Tiny-ImageNet, and then apply this same SAE across datasets. We measure reconstruction fidelity using Fraction of Variance Unexplained (FVU), where larger values indicate poorer reconstruction and FVU > 1 means the SAE reconstructs worse than a mean predictor. With the generalized SAE, VS2 improves on CIFAR-100 by +3.56% and +4.26% for ViT-B/32 and ViT-B/16, respectively, where FVU is low (≈ 0.21). It also improves on Tiny-ImageNet by +3.09% and +2.75%, where FVU is moderate (≈ 0.44). In contrast, VS2 degrades on CUB-200 by -2.95% and -5.84%, where FVU is high (≈ 1.93). This supports the prediction from Proposition 3.1: high FVU corresponds to a larger residual magnitude in the steering direction and less reliable steering.

Out-of-distribution steering with FVU-gated fallback. We next evaluate whether FVU can prevent harmful steering under distribution shift. We apply the generalized SAE to six unseen datasets: Aircraft [Maji et al., 2013], Food101 [Bossard et al., 2014], Flower102 [Nilsback and Zisserman,

2008], Caltech101 [Fei-Fei, 2004], SUN [Xiao et al., 2010], and EuroSAT [Helber et al., 2019]. As an upper reference, Table 3 also reports VS2 with an SAE trained on each dataset’s unlabeled training split. In this target-domain SAE setting, VS2 improves over zero-shot CLIP across all six datasets, suggesting that sparse steering remains effective when reconstruction is faithful.

By contrast, ungated generalized steering can substantially degrade performance on some shifted datasets, such as Aircraft and SUN, where the generalized SAE does not reconstruct the target activations reliably. We therefore use per-sample FVU as a label-free reliability diagnostic: if a test sample’s FVU exceeds a threshold τ , we skip steering and fall back to the original CLIP embedding; otherwise, we apply VS2. To set τ without labels, we calibrate a single threshold on the unlabeled CIFAR-100+Tiny-ImageNet training union by taking the q-quantile of per-sample FVU scores, and then apply the same threshold across all datasets. For FVU-gated rows in Table 3, we report both top-1 accuracy and coverage, the fraction of samples for which steering is applied. FVU gating acts as a conservative safety mechanism: when generalized steering is harmful, coverage drops and performance returns toward the zero-shot baseline; when reconstruction is reliable, coverage remains high and most of the gains are retained. This gives VS2 a model-internal reliability signal that standard steering-vector and test-time adaptation pipelines typically lack.

# 5 Limitations and Future Work

VS2 shows that SAE latents can serve not only as tools for post-hoc interpretation, but also as an actionable intervention space for downstream visual steering. However, the method has a genuine offline prerequisite: a top-k SAE must be trained on activations representative of the target distribution. When the SAE generalizes poorly to a sample, Proposition 3.1 shows that the steering residual is controlled by the per-sample FVU. Our FVU gate therefore trades coverage for safety by conservatively reverting to zero-shot CLIP when reconstruction is unreliable. This reduces harmful steering under distribution shift, but it also means that VS2 may occasionally forgo potential gains. Future work could improve SAE fidelity at deployment through lightweight per-instance self-reconstruction or adaptive sparse coding before steering.

A second limitation is the mismatch between reconstruction and task saliency. SAEs are trained to reconstruct activations, so they surface features that are salient for reproducing the embedding rather than necessarily features that are optimal for classification. This reconstruction-vs-task saliency gap limits uniform sparse-feature amplification and motivates task-aligned sparse steering: selecting and amplifying the subset of SAE features that most directly supports the target decision. Appendix L takes an initial step in this direction with Prototype-Alignment Sparse Steering, inspired by prototype theory [Rosch, 1973], suggesting that weak or indirect task structure can better align sparse features with downstream needs.

Empirically, we evaluate VS2 on CLIP ViT-B/32 and ViT-B/16 across nine datasets. ImageNetscale benchmarks and non-CLIP vision-language models remain important directions for future evaluation. Finally, VS2 inherits biases from the underlying vision-language model and amplifies features selected by the SAE. If these features are misaligned with human-relevant or safety-critical attributes, this may be difficult to detect in advance. Safety-critical use should therefore pair VS2 with the FVU gate, per-class auditing, and task-specific validation.

# 6 Conclusion

We introduced Visual Sparse Steering (VS2), a lightweight, label-free method for using Sparse Autoencoder (SAE) latents as an actionable intervention space in frozen vision-language models. VS2 constructs per-instance steering directions from active sparse features, improving zero-shot image classification without labels, contrastive anchors, weight updates, or test-time optimization. Across 9 datasets, VS2 consistently improves over zero-shot CLIP while adding less than 0.1% inference compute. Beyond empirical gains, we showed that VS2 implements centroid-deviation steering, and offers an FVU-based reliability gate that disables steering when the SAE representation is unreliable, reducing harmful interventions under distribution shift. Finally, our selective-amplification study with VS2++ reveals substantial headroom when task-relevant sparse features can be identified, exposing the reconstruction-vs-task saliency gap as a central challenge for future SAE-based downstream methods. Overall, our results suggest that SAEs can move beyond interpretability and provide a practical, analyzable, and reliability-aware basis for label-free visual steering.

# References

David Bau, Bolei Zhou, Aditya Khosla, Aude Oliva, and Antonio Torralba. Network dissection: Quantifying interpretability of deep visual representations. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 6541–6549, 2017.   
Reza Bayat, Ali Rahimi-Kalahroudi, Mohammad Pezeshki, Sarath Chandar, and Pascal Vincent. Steering large language model activations in sparse spaces. arXiv preprint arXiv:2503.00177, 2025.   
Usha Bhalla, Alex Oesterling, Suraj Srinivas, Flavio Calmon, and Himabindu Lakkaraju. Interpreting clip with sparse linear concept embeddings (splice). Advances in Neural Information Processing Systems, 37:84298–84328, 2024.   
Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool. Food-101–mining discriminative components with random forests. In European conference on computer vision, pages 446–461. Springer, 2014.   
Trenton Bricken, Adly Templeton, Joshua Batson, Brian Chen, Adam Jermyn, Tom Conerly, Nick Turner, Cem Anil, Carson Denison, Amanda Askell, Robert Lasenby, Yifan Wu, Shauna Kravec, Nicholas Schiefer, Tim Maxwell, Nicholas Joseph, Zac Hatfield-Dodds, Alex Tamkin, Karina Nguyen, Brayden McLean, Josiah E Burke, Tristan Hume, Shan Carter, Tom Henighan, and Christopher Olah. Towards monosemanticity: Decomposing language models with dictionary learning. Transformer Circuits Thread, 2023. https://transformer-circuits.pub/2023/monosemanticfeatures/index.html.   
David Chanin, James Wilken-Smith, Tomáš Dulka, Hardik Bhatnagar, and Joseph Bloom. A is for absorption: Studying feature splitting and absorption in sparse autoencoders, 2024. URL https://arxiv.org/abs/2409.14507.   
Hoagy Cunningham, Aidan Ewart, Logan Riggs, Robert Huben, and Lee Sharkey. Sparse autoencoders find highly interpretable features in language models. arXiv preprint arXiv:2309.08600, 2023.   
EleutherAI. Sparsify: Sparse autoencoder library. https://github.com/EleutherAI/sparsify, 2024.   
Nelson Elhage, Neel Nanda, Catherine Olsson, Tom Henighan, Nicholas Joseph, Ben Mann, Amanda Askell, Yuntao Bai, Anna Chen, Tom Conerly, et al. A mathematical framework for transformer circuits. Transformer Circuits Thread, 1(1):12, 2021.   
Nelson Elhage, Tristan Hume, Catherine Olsson, Nicholas Schiefer, Tom Henighan, Shauna Kravec, Zac Hatfield-Dodds, Robert Lasenby, Dawn Drain, Carol Chen, et al. Toy models of superposition. arXiv preprint arXiv:2209.10652, 2022.   
Joshua Engels, Logan Riggs Smith, and Max Tegmark. Decomposing the dark matter of sparse autoencoders. Transactions on Machine Learning Research, 2025. ISSN 2835-8856. URL https://openreview.net/forum?id=sXq3Wb3vef.   
Li Fei-Fei. Learning generative visual models from few training examples. In Workshop on Generative-Model Based Vision, IEEE Proc. CVPR, 2004, 2004.   
Thomas Fel, Ekdeep Singh Lubana, Jacob S Prince, Matthew Kowal, Victor Boutin, Isabel Papadimitriou, Binxu Wang, Martin Wattenberg, Demba Ba, and Talia Konkle. Archetypal sae: Adaptive and stable dictionary learning for concept extraction in large vision models. arXiv preprint arXiv:2502.12892, 2025.   
Chun-Mei Feng, Kai Yu, Yong Liu, Salman Khan, and Wangmeng Zuo. Diverse data augmentation with diffusions for effective test-time prompt tuning. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 2704–2714, 2023.   
Leo Gao, Tom Dupré la Tour, Henk Tillman, Gabriel Goh, Rajan Troll, Alec Radford, Ilya Sutskever, Jan Leike, and Jeffrey Wu. Scaling and evaluating sparse autoencoders. arXiv preprint arXiv:2406.04093, 2024a.

Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao. Clip-adapter: Better vision-language models with feature adapters. International Journal of Computer Vision, 132(2):581–595, 2024b.   
Yunhe Gao, Xingjian Shi, Yi Zhu, Hao Wang, Zhiqiang Tang, Xiong Zhou, Mu Li, and Dimitris N Metaxas. Visual prompt tuning for test-time domain adaptation. arXiv preprint arXiv:2210.04831, 2022.   
Paul Gavrikov, Jovita Lukasik, Steffen Jung, Robert Geirhos, Muhammad Jehanzeb Mirza, Margret Keuper, and Janis Keuper. Can we talk models into seeing the world differently? In Thirteenth International Conference on Learning Representations. OpenReview. net, 2025.   
Patrick Helber, Benjamin Bischke, Andreas Dengel, and Damian Borth. Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 12(7):2217–2226, 2019.   
Evan Hernandez, Arnab Sen Sharma, Tal Haklay, Kevin Meng, Martin Wattenberg, Jacob Andreas, Yonatan Belinkov, and David Bau. Linearity of relation decoding in transformer language models. In The Twelfth International Conference on Learning Representations, 2024. URL https:// openreview.net/forum?id=w7LU2s14kE.   
Sonia Joseph, Praneet Suresh, Ethan Goldfarb, Lorenz Hufe, Yossi Gandelsman, Robert Graham, Danilo Bzdok, Wojciech Samek, and Blake Aaron Richards. Steering clip’s vision transformer with sparse autoencoders. arXiv preprint arXiv:2504.08729, 2025.   
Shruti Joshi, Andrea Dittadi, Sébastien Lachapelle, and Dhanya Sridhar. Identifiable steering via sparse autoencoding of multi-concept shifts. arXiv preprint arXiv:2502.12179, 2025.   
Muhammad Uzair Khattak, Hanoona Rasheed, Muhammad Maaz, Salman Khan, and Fahad Shahbaz Khan. Maple: Multi-modal prompt learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 19113–19122, 2023.   
Alex Krizhevsky, Geoffrey Hinton, et al. Learning multiple layers of features from tiny images. 2009.   
Yann Le and Xuan Yang. Tiny imagenet visual recognition challenge. CS 231N, 7(7):3, 2015.   
Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al. Retrieval-augmented generation for knowledge-intensive nlp tasks. Advances in neural information processing systems, 33: 9459–9474, 2020.   
Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, and Martin Wattenberg. Inference-time intervention: Eliciting truthful answers from a language model. Advances in Neural Information Processing Systems, 36:41451–41530, 2023.   
Zhuowei Li, Haizhou Shi, Yunhe Gao, Di Liu, Zhenting Wang, Yuxiao Chen, Ting Liu, Long Zhao, Hao Wang, and Dimitris N. Metaxas. The hidden life of tokens: Reducing hallucination of large vision-language models via visual information steering, 2025a. URL https://arxiv.org/abs/ 2502.03628.   
Zhuowei Li, Zihao Xu, Ligong Han, Yunhe Gao, Song Wen, Di Liu, Hao Wang, and Dimitris N. Metaxas. Implicit in-context learning. In The Thirteenth International Conference on Learning Representations, 2025b. URL https://openreview.net/forum?id=G7u4ue6ncT.   
Hyesu Lim, Jinho Choi, Jaegul Choo, and Steffen Schneider. Sparse autoencoders reveal selective remapping of visual concepts during adaptation. arXiv preprint arXiv:2412.05276, 2024.   
Sheng Liu, Haotian Ye, Lei Xing, and James Zou. In-context vectors: Making in context learning more effective and controllable through latent space steering, 2024.   
Yuejiang Liu, Parth Kothari, Bastien Van Delft, Baptiste Bellot-Gurlet, Taylor Mordan, and Alexandre Alahi. Ttt++: When does self-supervised test-time training fail or thrive? Advances in Neural Information Processing Systems, 34:21808–21820, 2021.

Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew Blaschko, and Andrea Vedaldi. Fine-grained visual classification of aircraft. arXiv preprint arXiv:1306.5151, 2013.   
Aleksandar Makelov. Sparse autoencoders match supervised features for model steering on the ioi task. In ICML 2024 Workshop on Mechanistic Interpretability, 2024.   
Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. Distributed representations of words and phrases and their compositionality. In C.J. Burges, L. Bottou, M. Welling, Z. Ghahramani, and K.Q. Weinberger, editors, Advances in Neural Information Processing Systems, volume 26. Curran Associates, Inc., 2013. URL https://proceedings.neurips.cc/paper\_ files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf.   
Maria-Elena Nilsback and Andrew Zisserman. Automated flower classification over a large number of classes. In 2008 Sixth Indian conference on computer vision, graphics & image processing, pages 722–729. IEEE, 2008.   
Tuomas Oikarinen and Tsui-Wei Weng. Clip-dissect: Automatic description of neuron representations in deep vision networks. arXiv preprint arXiv:2204.10965, 2022.   
Chris Olah, Alexander Mordvintsev, and Ludwig Schubert. Feature visualization. Distill, 2(11):e7, 2017.   
Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, and Shan Carter. Zoom in: An introduction to circuits. Distill, 5(3):e00024–001, 2020.   
Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.   
Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In Causal Representation Learning Workshop at NeurIPS 2023, 2023. URL https://openreview.net/forum?id=T0PoOJg8cK.   
Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International conference on machine learning, pages 8748–8763. PmLR, 2021.   
Eleanor H Rosch. Natural categories. Cognitive psychology, 4(3):328–350, 1973.   
Manli Shu, Weili Nie, De-An Huang, Zhiding Yu, Tom Goldstein, Anima Anandkumar, and Chaowei Xiao. Test-time prompt tuning for zero-shot generalization in vision-language models. Advances in Neural Information Processing Systems, 35:14274–14289, 2022.   
Karen Simonyan, Andrea Vedaldi, and Andrew Zisserman. Visualising image classification models and saliency maps. Deep Inside Convolutional Networks, 2(2), 2014.   
Samuel Stevens, Wei-Lun Chao, Tanya Berger-Wolf, and Yu Su. Sparse autoencoders for scientifically rigorous interpretation of vision models. arXiv preprint arXiv:2502.06755, 2025.   
Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei Efros, and Moritz Hardt. Test-time training with self-supervision for generalization under distribution shifts. In International conference on machine learning, pages 9229–9248. PMLR, 2020.   
Viacheslav Surkov, Chris Wendler, Antonio Mari, Mikhail Terekhov, Justin Deschenaux, Robert West, Caglar Gulcehre, and David Bau. One-step is enough: Sparse autoencoders for text-to-image diffusion models. arXiv preprint arXiv:2410.22366, 2024.   
Adly Templeton. Scaling monosemanticity: Extracting interpretable features from claude 3 sonnet. Anthropic, 2024.   
Harrish Thasarathan, Julian Forsyth, Thomas Fel, Matthew Kowal, and Konstantinos Derpanis. Universal sparse autoencoders: Interpretable cross-model concept alignment. arXiv preprint arXiv:2502.03714, 2025.

Alexander Matt Turner, Lisa Thiergart, David Udell, Gavin Leech, Ulisse Mini, and Monte MacDiarmid. Activation addition: Steering language models without optimization, 2023.   
Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie. The caltech-ucsd birds-200-2011 dataset. 2011.   
Dequan Wang, Evan Shelhamer, Shaoteng Liu, Bruno Olshausen, and Trevor Darrell. Tent: Fully test-time adaptation by entropy minimization. arXiv preprint arXiv:2006.10726, 2020.   
Jianxiong Xiao, James Hays, Krista A Ehinger, Aude Oliva, and Antonio Torralba. Sun database: Large-scale scene recognition from abbey to zoo. In 2010 IEEE computer society conference on computer vision and pattern recognition, pages 3485–3492. IEEE, 2010.   
Hee Suk Yoon, Eunseop Yoon, Joshua Tian Jin Tee, Mark Hasegawa-Johnson, Yingzhen Li, and Chang D Yoo. C-tpt: Calibrated test-time prompt tuning for vision-language models via text feature dispersion. arXiv preprint arXiv:2403.14119, 2024.   
Matthew D Zeiler and Rob Fergus. Visualizing and understanding convolutional networks. In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13, pages 818–833. Springer, 2014.   
Renrui Zhang, Rongyao Fang, Wei Zhang, Peng Gao, Kunchang Li, Jifeng Dai, Yu Qiao, and Hongsheng Li. Tip-adapter: Training-free clip-adapter for better vision-language modeling. arXiv preprint arXiv:2111.03930, 2021.   
Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Conditional prompt learning for vision-language models. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 16816–16825, 2022a.   
Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu. Learning to prompt for visionlanguage models. International Journal of Computer Vision, 130(9):2337–2348, 2022b.   
Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, Shashwat Goel, Nathaniel Li, Michael J. Byun, Zifan Wang, Alex Mallen, Steven Basart, Sanmi Koyejo, Dawn Song, Matt Fredrikson, J. Zico Kolter, and Dan Hendrycks. Representation engineering: A top-down approach to ai transparency, 2023.

# Appendix

# A Decoding the Sparse Latent Space: Insights from SAEs

Our proposed methods achieve significant classification performance gains, primarily due to the contribution of Sparse Autoencoders (SAEs). We hypothesize that SAEs identify meaningful sparse features, which in turn guide the steering mechanisms. To validate this assumption, we conduct both quantitative and qualitative evaluations to assess the significance of the learned features.

# A.1 Quantitative Evaluation of Feature Significance

To evaluate the role of sparse features, we manipulate the top-k most active latent features extracted via Sparse Autoencoders. This experiment examines whether these features are critical for classification and how model predictions change under different modifications. We explore the following manipulation settings:

1. Zeroing-Out $( \gamma = 0 )$ : We set the top-k most active features to zero before applying the steering vector. This removes their influence while preserving the remaining latent structure.   
2. Negation $( \gamma = - 1 )$ : We invert the sign of the top-k features before applying the steering vector, effectively pushing the representation in the opposite direction. This tests whether these dimensions encode class-discriminative information.

Table 4 reports the zero-shot classification accuracy of $\mathrm { S A E } _ { \mathrm { R E C } } ^ { \mathrm { F } + \gamma }$ using ViT-B/32 after applying these transformations to the Sparse Steering vector intervention. We observe that negating or zeroing the top-k most important features consistently degrades performance across all datasets. This result confirms that the features learned by the Sparse Autoencoders are essential for classification.

Table 4: Effect of manipulating top-k sparse codes. Zero-shot accuracy (%) drops sharply when dominant sparse features are zeroed $( \gamma = 0 )$ or negated $( \gamma = - 1 )$ ), confirming their importance. 

<table><tr><td>Modification</td><td>CIFAR-100</td><td>Tiny-IN</td></tr><tr><td> $\text{CLIP}_{\text{ZS}}$ </td><td>61.07</td><td>56.64</td></tr><tr><td> $\text{SAE}_{\text{REC}}^{\text{F}+\gamma}$ </td><td>62.69</td><td>39.49</td></tr><tr><td>+ Zero-out ( $\gamma = 0$ )</td><td>1.71</td><td>16.11</td></tr><tr><td>+ Negate ( $\gamma = -1$ )</td><td>0.06</td><td>0.82</td></tr></table>

# A.2 Qualitative Evaluation of Features Significance

We qualitatively investigate the features learned in the sparse latent representations of the top-k activations in the Sparse Autoencoder (SAE). Specifically, we assess the learned features by analyzing feature activations for each input and identifying the inputs that exhibit the highest activations for a given feature. Unlike mechanistic interpretability in the language domain, where an LLM can be used to assign semantic labels to a feature by summarizing its highly activated inputs, the vision domain lacks an equivalent automated labeling process.

To avoid reliance on human qualitative evaluation, we leverage annotated datasets where each image is associated with predefined attributes. For the qualitative evaluation of feature significance, we use the CUB dataset, which provides rich concept annotations for each image, enabling a structured assessment of the learned representations. Specifically, we investigate whether we can identify specific latent features with the highest concept coverage among their top-k most activated images. Concept coverage refers to how consistently a specific interpretable concept (e.g., an identifiable object category, attribute, or semantic idea) appears across a set of highly activated examples for a given SAE dimension. The intuition is that if a specific concept frequently emerges among the top-activating images for a particular feature, that feature is strongly associated with that concept.

For feature 511 as shown in Figure 4 (left), the activated images exhibit consistent semantic characteristics, including a gray upper part and a white underpart. Notably, this feature predominantly activates for images from similar but different classes, specifically different types of Gulls, such as

Western, California, Herring, and Slaty-Backed Gulls. We observe that the top-activating images for these classes share all concepts in common which is not the case with dimension 3067. For feature 3067, as shown in Figure 4 (right) we observe that the top-activating images share common visual attributes, such as a white-colored throat and black eyes. Through human qualitative evaluation, we find that features in the sparse latent space capture meaningful visual concepts, grouping semantically similar images together, either from the same class or across different classes, as long as they share underlying conceptual similarities.

![](images/10eb945daa6eb0946937dd5f42444686ee18ea396e1ba1d74fe28884f07be34e.jpg)

<details>
<summary>bar</summary>

Dimension 511 - Best Coverage Concept: 'has_upperparts_color:grey' in 9/9 images
| Category | Value |
|---|---|
| has_upperparts_color:grey | 0 |
| has_underparts_color: white | 0 |
| has_breadparts_color: white | 0 |
| has_thread_color: white | 0 |
| has_length_color: white | 0 |
| has_belly_color: white | 0 |
| has_primary_color: white | 0 |
| has_crown_color: white | 0 |
| Has_upparts_color: grey | 0 |
| Has_underparts_color: white | 0 |
| Has_breadparts_color: white | 0 |
| Has_thread_color: white | 0 |
| Has_lengthyan_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_upparts_crown: grey | 0 |
| Has_underparts_crown: white | 0 |
| Has_breadparts_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
|
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: black | 0 |
| Has_bready_crown: black | 0 |
| Has_bready_crown: black | 0 |
| Has_bready_crown: black | 0 |
| Has_bready_crown: black | 0 |
| Has_bready_crown: black | 0 |
| Has_bready_crown: black | 0 |
| Has_bready_crown: black | 0 |
| Has_bread y_crown: white | 0 |
| Has_bread y_crown: white | 0 |
| Has_bread y_crown: white | 0 |
| Has_bread y_crown: white | 0 |
| Has_bread y_crown: white | 0 |
| Has_bread y_crown: white | 0 |
| Has_bread y_crown: white | 0 |
| Has_bread y_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
| Has_bready_crown: white | 0 |
</details>

![](images/45ba50c0adaa70cafd11ad1945c558da6de4415c9d47f5b8c9850c0d5a5ffa9f.jpg)

<details>
<summary>bar</summary>

Dimension 3067 - Best Coverage Concept: 'has throat_color:white' in 9/9 images
| Category | Count |
|---|---|
| Label: 101 Droset Englishbird | 8 |
| Label: 45200000000000000000000000000000000000000000000000000000000000000000000000000000 |
| Label: 151 Black, 3m | 7 |
| Label: 151 Black, 3m (Image) | 6 |
| Label: 151 Black, 3m (Image) | 6 |
| Label: 151 Black, 3m (Image) | 6 |
| Label: 151 Black, 3m (Image) | 6 |
| Label: 151 Black, 3m (Image) | 6 |
| Label: 151 Black, 3m (Image) | 6 |
| Label: "has throat_color-white" | 9 |
| Label: "has_yee_color-black" | 8 |
| Label: "has_repe_color-white" | 7 |
| Label: "has_pice_small_(r_1)_1/n" | 6 |
| Label: "has_wing_pattern-solid" | 6 |
| Label: "has_uppeppers_color-gray" | 6 |
| Label: "has_wing_color-gray" | 6 |
| Label: "has_back_gain-solid" | 5 |
| Label: "has_bill_color-black" | 5 |
The chart displays a bar chart with two sets of data points for each category. The x-axis represents the dimension (in meters), and the y-axis represents the count of images. The top row labels are the dimensions of the concept (e.g., 'has throat_color-white', 'has_yee_color-black') and the bottom row lists the feature names. The chart is grouped into three rows based on the legend entries.
</details>

Figure 4: Concept coverage analysis of learned sparse latent features in the Sparse Autoencoder (SAE). Each subfigure illustrates the top-activating images for two different SAE dimensions, highlighting the consistency of shared visual concepts among the highest-activated examples. The analysis demonstrates how sparse features capture meaningful semantic attributes, grouping semantically similar images either within or across classes. Left: Top-activating images for feature 511. The images predominantly belong to different classes of Gulls (e.g., Western, California, Herring, and Slaty-Backed Gulls), yet they share consistent visual characteristics such as a gray upper part and a white underpart. This suggests that feature 511 captures a semantically meaningful concept spanning multiple related categories. Right: Top-activating images for feature 3067. The images share distinct visual attributes, including a white-colored throat and black eyes. However, unlike feature 511, these images belong to more diverse categories, indicating that this latent dimension captures a broader concept that generalizes across different classes.

Table 5: Top-10 most activated classes for SAE Feature 511 (left) and Feature 3067 (right). Feature 511 strongly aligns with gull-like classes, while Feature 3067 captures a cross-class head/throat attribute. 

<table><tr><td>Class (511)</td><td>Activation</td><td>Class (3067)</td><td>Activation</td></tr><tr><td>066.Western_Gull</td><td>96.7% (29/30)</td><td>082.Ringed_Kingfisher</td><td>60.0% (18/30)</td></tr><tr><td>060.Glaucous_winged_Gull</td><td>96.6% (28/29)</td><td>083.White_breasted_Kingfisher</td><td>40.0% (12/30)</td></tr><tr><td>061.Heermann_Gull</td><td>93.3% (28/30)</td><td>079.Belted_Kingfisher</td><td>30.0% (9/30)</td></tr><tr><td>059.California_Gull</td><td>90.0% (27/30)</td><td>041.Scissor_tailed_Flycatcher</td><td>23.3% (7/30)</td></tr><tr><td>062.Herring_Gull</td><td>83.3% (25/30)</td><td>080.Green_Kingfisher</td><td>16.7% (5/30)</td></tr><tr><td>063.Ivory_Gull</td><td>76.7% (23/30)</td><td>002.Laysan_Albatross</td><td>13.3% (4/30)</td></tr><tr><td>087.Mallard</td><td>76.7% (23/30)</td><td>025.Pelagic_Cormorant</td><td>13.3% (4/30)</td></tr><tr><td>147.Least_Tern</td><td>76.7% (23/30)</td><td>025.Pelagic_Cormorant</td><td>13.3% (4/30)</td></tr><tr><td>064.Ring_billed_Gull</td><td>70.0% (21/30)</td><td>044.Frigatebird</td><td>10.0% (3/30)</td></tr><tr><td>084.Red_legged_Kittiwake</td><td>73.9% (17/23)</td><td>049.Boat_tailed_Grackle</td><td>10.0% (3/30)</td></tr></table>

Class-Conditional Feature Activation. To provide additional evidence that SAE dimensions capture discriminative visual concepts, we report the class-conditional activation frequencies of two example features: 511 and 3067. Specifically, Table 5 shows the percentage of samples in the top-10 activating classes where each feature is present.

What we observe is that feature 511 fires very often in gull-like classes. It also fires Larus-type seabirds, i.e., terns, kittiwakes. We observe that it also fires moderately on similar seabirds (Horned Puffin: 53.3%, White Pelican: 40%, Brown Pelican: 16.7%, Red-breasted Merganser: 10%). Additionally, we observe that Feature 511 activates near 0% on unrelated classes such as buntings, warblers, and hummingbirds. In contrast, Feature 3067 captures a cross-category visual attribute rather than a taxonomic grouping. Specifically, it consistently activates on species that share a distinctive head-and-throat pattern (e.g., white throat with darker eye region), even when those species are taxonomically unrelated. This indicates that the SAE possibly learns attribute-level discriminative concepts, complementing the class-specific features exemplified by Feature 511.

![](images/3eaab7279e4eb772200e72a495305503fc0110122b1df4d09495b266a7045207.jpg)

<details>
<summary>heatmap</summary>

| Steering Vector Magnitude λ | Upweighting Factor of Sparse Features γ | Accuracy |
| --------------------------- | -------------------------------------- | -------- |
| 0.0                         | 3.0                                    | 64.5     |
| 0.5                         | 2.9                                    | 64.0     |
| 1.0                         | 2.8                                    | 63.5     |
| 1.5                         | 2.7                                    | 63.0     |
| 2.0                         | 2.6                                    | 62.5     |
| 2.5                         | 2.5                                    | 62.0     |
| 3.0                         | 2.4                                    | 61.5     |
| 3.5                         | 2.3                                    | 61.5     |
| 4.0                         | 2.2                                    | 61.5     |
| 4.5                         | 2.1                                    | 61.5     |
| 5.0                         | 2.0                                    | 61.5     |
| 5.5                         | 1.9                                    | 61.5     |
| 6.0                         | 1.8                                    | 61.5     |
| 6.5                         | 1.7                                    | 61.5     |
| 7.0                         | 1.6                                    | 61.5     |
| 7.5                         | 1.5                                    | 61.5     |
| 8.0                         | 1.4                                    | 61.5     |
| 8.5                         | 1.3                                    | 61.5     |
| 9.0                         | 1.2                                    | 61.5     |
| 9.5                         | 1.1                                    | 61.5     |
| 10.0                        | 1.0                                    | 61.5     |
</details>

(a) CIFAR-100

![](images/9832117a4265ea8e00fd5dd400af08040f037ff379f3571180744036338e876d.jpg)

<details>
<summary>heatmap</summary>

| Steering Vector Magnitude λ | Upweighting Factor of Sparse Features γ | Accuracy |
| --------------------------- | -------------------------------------- | -------- |
| 0.0                         | 3.0                                    | 52       |
| 0.5                         | 2.8                                    | 51       |
| 1.0                         | 2.6                                    | 50       |
| 1.5                         | 2.4                                    | 49       |
| 2.0                         | 2.2                                    | 48       |
| 2.5                         | 2.0                                    | 47       |
| 3.0                         | 1.8                                    | 46       |
| 3.5                         | 1.6                                    | 45       |
| 4.0                         | 1.4                                    | 45       |
| 4.5                         | 1.2                                    | 45       |
| 5.0                         | 1.0                                    | 45       |
| 5.5                         | 0.8                                    | 45       |
| 6.0                         | 0.6                                    | 45       |
| 6.5                         | 0.4                                    | 45       |
| 7.0                         | 0.2                                    | 45       |
| 7.5                         | 0.0                                    | 45       |
| 8.0                         | -0.2                                   | 45       |
| 8.5                         | -0.4                                   | 45       |
| 9.0                         | -0.6                                   | 45       |
| 9.5                         | -0.8                                   | 45       |
| 10.0                        | -1.0                                   | 45       |
| 10.5                        | -1.2                                   | 45       |
| 11.0                        | -1.4                                   | 45       |
| 11.5                        | -1.6                                   | 45       |
| 12.0                        | -1.8                                   | 45       |
| 12.5                        | -2.0                                   | 45       |
| 13.0                        | -2.2                                   | 45       |
| 13.5                        | -2.4                                   | 45       |
| 14.0                        | -2.6                                   | 45       |
| 14.5                        | -2.8                                   | 45       |
| 15.0                        | -3.0                                   | 45       |
| 15.5                        | -3.2                                   | 45       |
| 16.0                        | -3.4                                   | 45       |
| 16.5                        | -3.6                                   | 45       |
| 17.0                        | -3.8                                   | 45       |
| 17.5                        | -4.0                                   | 45       |
| 18.0                        | -4.2                                   | 45       |
| 18.5                        | -4.4                                   | 45       |
| 19.0                        | -4.6                                   | 45       |
| 19.5                        | -4.8                                   | 45       |
| 20.0                        | -5.0                                   | 45       |
| 20.5                        | -5.2                                   | 45       |
| 21.0                        | -5.4                                   | 45       |
| 21.5                        | -5.6                                   | 45       |
| 22.0                        | -5.8                                   | 45       |
| 22.5                        | -6.0                                   | 45       |
| 23.0                        | -6.2                                   | 45       |
| 23.5                        | -6.4                                   | 45       |
| 24.0                        | -6.6                                   | 45       |
| 24.5                        | -6.8                                   | 45       |
| 25.0                        | -7.0                                   | 45       |
| 25.5                        | -7.2                                   | 45       |
| 26.0                        | -7.4                                   | 45       |
| 26.5                        | -7.6                                   | 45       |
| 27.0                        | -7.8                                   | 45       |
| 27.5                        | -8.0                                   | 45       |
| 28.0                        | -8.2                                   | 45       |
| 28.5                        | -8.4                                   | 45       |
| 29.0                        | -8.6                                   | 45       |
| 29.5                        | -8.8                                   | 45       |
| 30.0                        | -9.0                                   | 45       |
| ...                         |                                         | ...      |
| ...                         |                                        | ...      |
| ...                         |                                        | ...      |
| ...                         |                                        | ...      |
| ...                         |                                        | ...      |
| ...                         |                                        | ...      |
| ...                         |                                        | ...      |
| ...                         |                                        | ...      |
| ...                         |                                        | ...      |
| ...                         |                                        | ...      |
| ...                         |                                        | ...      |
| ...                         |                                        | ...      |
| ...                         |                                        = [value]                            | ...      |
| ...                         | ...                                  | ...      |
The data is a heatmap of the upweighting factor of sparse features γ for each steering vector magnitude λ for each upweighting factor of the remaining scalar features γ for each upweighting factor of the remaining scalar features γ for each upweighting factor of the remaining scalar features γ for each upweighting factor of the remaining scalar features γ for each upweighting factor of the remaining scalar features γ for each upweighting factor of the remaining scalar features γ for each upweighting factor of the remaining scalar features γ for each upweighting factor of the remaining scalar features γ for each upweighting factor of the remaining scalar features γ for each up weighting factor of the remaining scalar features γ for each up weighting factor of the remaining scalar features γ for each up weighting factor of the remaining scalar features γ for each up weighting factor of the remaining scalar features γ for each up weighting factor of the remaining scalar features γ for each up weighting factor of the remaining scalar features γ for each up weighting factor of the remaining scalar features γ for each up weighting factor of the remaining scalar features γ for each up weighting factor of it is also shown in the heatmap.
</details>

(b) CUB-200

![](images/712820989522feac17e975407dbc6f73198d7b9f447ddef47c33aeb2413d4832.jpg)  
(c) Tiny-ImageNet   
Figure 5: Sensitivity of VS2 to sparse amplification γ and steering magnitude λ. All three datasets show a range of near-optimal combinations (warm colours), typically when $\lambda \cdot \gamma \in [ 2 , 3 ]$ . Accuracy degrades if either parameter becomes too large.

Discussion. We note that our goal is not to claim human interpretability of SAE features. As shown in recent mechanistic interpretability work [Chanin et al.], human-aligned features are often split across multiple SAE components (feature absorption), a known behavior in sparse dictionary learning. Nonetheless, our contribution lies in demonstrating that SAEs can go beyond interpretability, serving as effective mechanisms for performance improvement via sparse steering. While some features are interpretable, our main result is that sparse features enable controllable improvements in downstream classification tasks.

# B Sensitivity to Sparse Amplification (γ) and Steering Magnitude (λ)

We analyze how the two key hyperparameters in VS2 (i) the sparse-feature amplification $\gamma ,$ and (ii) the steering vector scale λ affect downstream accuracy. In the absence of contrastive supervision, these parameters govern how strongly we amplify sparse activations and how far the embedding is shifted in feature space. We sweep both values across a grid on three datasets: CIFAR-100, CUB-200, and Tiny-ImageNet using ViT-B/32 backbone.

Figure 5 shows that all tested combinations of $( \lambda , \gamma )$ outperform the zero-shot baseline, though to varying degrees. Each dataset exhibits a diagonal band of near-optimal settings where $\lambda \cdot \gamma \in [ 2 , 3 ]$ tends to yield peak accuracy. For example, CIFAR-100 peaks at $\lambda ^ { * } = 2 . 1$ and $\gamma ^ { * } = 1 . 5$ . Beyond this band, increasing λ or γ causes performance to degrade likely due to over-amplification of sparse features and/or embedding distortion. The consistent contour patterns across datasets suggest that VS2 is robust to moderate variation in its hyperparameters and that good settings generalize well across domains.

# C Implementation, Training, and Compute Details

We follow CLIP [Radford et al., 2021] with a ViT-B/32 and ViT-B/16 vision backbones, intercepting the output of each encoder layer for the CLS token. Specifically, we train top-k SAEs on the CLS embeddings for each chosen layer. We use $k = 6 4$ and $k = 2 \bar { 5 } 6$ as the maximum active features within the latent space for ViT-B/32 and ViT-B/16 respectively, and we set a “dead feature” threshold of 100 i.e., any feature seldom activated is pruned. We also use an expansion factor of 4 relative to the input embedding dimension, resulting in 3,072 latent units. Training largely follows Gao et al. [2024a] and uses EleutherAI [2024], with a linear learning-rate schedule and warmup from $5 \times 1 0 ^ { - 4 }$ .

We monitor reconstruction quality with the fraction of variance unexplained (FVU; lower is better), defined as

$$
\mathrm{FVU} = \frac {\left\| \mathbf {X} - \hat {\mathbf {X}} \right\| _ {F} ^ {2}}{\left\| \mathbf {X} - \bar {\mathbf {X}} \right\| _ {F} ^ {2}}, \tag {6}
$$

where $\mathbf { X } \in \mathbb { R } ^ { B \times d }$ is a batch of CLS embeddings, $\hat { \bf X }$ denotes their SAE reconstructions, and X¯ is the batch mean (so the denominator equals the total variance). FVU is the complement of the coefficient of determination $( 1 - R ^ { 2 } ) ;$ an FVU of 0 indicates perfect reconstruction, while an FVU of 1 corresponds to predicting only the mean. In Table 6, we report FVU results for ViT-B/32 with $k = 6 4$ and ViT-B/16 with $k = 2 5 6$ across all three datasets.

Table 6: Fraction of variance unexplained (FVU; lower is better). Each SAE has 4× expansion; sparsity k = 64 for ViT-B/32 and $k = 2 5 6$ for ViT-B/16. 

<table><tr><td>Dataset</td><td>ViT-B/32 (k=64)</td><td>ViT-B/16 (k=256)</td></tr><tr><td>CIFAR-100</td><td>0.2812</td><td>0.1166</td></tr><tr><td>CUB-200</td><td>0.2487</td><td>0.1653</td></tr><tr><td>Tiny-ImageNet</td><td>0.5060</td><td>0.3018</td></tr></table>

Compute resources. Experiments were run on an internal GPU cluster node with 8 NVIDIA Quadro RTX 8000 GPUs, each with 46,080 MiB of memory, using CUDA 12.6. SAE training and VS2 evaluation were run on a single GPU per experiment unless otherwise specified. VS2 inference adds only one SAE encode/decode pass on top of CLIP inference; Table 1 and Appendix N report the inference-time and training-time compute, respectively.

Existing assets and licenses. We use publicly available pretrained models and image-classification datasets for standard research evaluation. We cite the original sources for all assets used in the paper, including CLIP, DINOv2, CIFAR-100, CUB-200, Tiny-ImageNet, Aircraft, Food101, Flower102, Caltech101, SUN, and EuroSAT. We use these assets only for their intended research and benchmark evaluation purposes and do not redistribute dataset contents or pretrained model weights.

# D Sensitivity in Retrieval-Augmented Generation (RAG)

When an external image corpus is available, VS2 can be extended using a Retrieval-Augmented Generation (RAG) pipeline. We use DINOv2 Oquab et al. [2023] to retrieve top-k images most similar to the input query and compute an enhanced embedding by combining the query with the retrieved set:

$$
\mathbf {E} = \alpha \mathbf {q} + (1 - \alpha) \sum_ {j = 1} ^ {k} w _ {j} \mathbf {r} _ {j},
$$

where q is the query embedding, $\mathbf { r } _ { j }$ the $j \cdot$ -th retrieved embedding, and $w _ { j }$ the normalized similarity weight:

$$
w _ {j} = \frac {s _ {j}}{\sum_ {i = 1} ^ {k} s _ {i}}, \quad \text { where } s _ {j} = \text { sim } (\mathbf {q}, \mathbf {r} _ {j}).
$$

The parameter $\alpha \in [ 0 , 1 ]$ controls the trade-off between using the original query and the retrieved set. We sweep over values of α and k to assess their impact on zero-shot classification performance. Figure 6 shows results for CIFAR-100, and Tiny-ImageNet. CIFAR-100 and Tiny-ImageNet display a similar trend: larger α (i.e., more reliance on the query) typically degrades performance. On Tiny-ImageNet, setting $\alpha = 0 ,$ , completely ignoring the input query and relying purely on retrieved features, yields the best result. Across all datasets, large k eventually hurts performance, confirming that RAG benefits from focused rather than broad context. The trade-off parameter α and the retrieval depth k play dataset-dependent roles in RAG-enhanced pipelines. For fine-grained domains, fewer, high-confidence neighbors and a low α might work best; for noisy domains, more reliance on the retrieval images might be more beneficial.

# E Steering Across Layers of the CLS Token

Our default method reconstructs and applies steering to the CLS token embedding at the final layer of the vision encoder. To study depth effects, we evaluate steering the CLS token across multiple final layers of the transformer. We use ViT-B/16 on CIFAR-100 with expansion factor 4 and 128 sparse activations. Table 7 reports top-1 accuracy when steering the last 1, 2, or 3 layers. Steering only the final layer yields the best performance. Adding earlier layers causes a progressive drop in accuracy, falling below the zero-shot baseline when steering the last three layers.

![](images/b5cc690a4b34b1abcfa92a56c06e1a970d9ea331576bb92a1fdcf3c57093f362.jpg)

<details>
<summary>line</summary>

| Alpha | Top-1 Accuracy (N=10) | Top-1 Accuracy (N=20) | Top-1 Accuracy (N=50) | Top-1 Accuracy (N=100) | Top-1 Accuracy (N=300) | Top-1 Accuracy (N=500) |
|-------|------------------------|------------------------|------------------------|-------------------------|-------------------------|-------------------------|
| 0.00  | 0.76                   | 0.76                   | 0.76                   | 0.74                    | 0.74                    | 0.74                    |
| 0.25  | 0.76                   | 0.76                   | 0.76                   | 0.74                    | 0.74                    | 0.74                    |
| 0.50  | 0.74                   | 0.74                   | 0.74                   | 0.72                    | 0.72                    | 0.72                    |
| 0.75  | 0.68                   | 0.68                   | 0.68                   | 0.66                    | 0.66                    | 0.66                    |
| 1.00  | 0.82                   | 0.82                   | 0.82                   | 0.82                    | 0.82                    | 0.82                    |
</details>

(a) CIFAR-100

![](images/26fedc38be652968d3b349b2a827b50a1aa35521b14470b630a04200ad243643.jpg)

<details>
<summary>line</summary>

| Alpha | Top-1 Accuracy (α=101) | Top-1 Accuracy (α=201) | Top-1 Accuracy (α=501) | Top-1 Accuracy (α=1000) | Top-1 Accuracy (α=5000) |
|-------|------------------------|------------------------|------------------------|-------------------------|-------------------------|
| 0.00  | 0.72                   | 0.72                   | 0.72                   | 0.72                    | 0.72                    |
| 0.25  | 0.71                   | 0.71                   | 0.71                   | 0.71                    | 0.71                    |
| 0.50  | 0.69                   | 0.69                   | 0.69                   | 0.69                    | 0.69                    |
| 0.75  | 0.64                   | 0.64                   | 0.64                   | 0.64                    | 0.64                    |
| 1.00  | 0.34                   | 0.34                   | 0.34                   | 0.34                    | 0.34                    |
</details>

(b) Tiny-ImageNet   
Figure 6: RAG sensitivity to α and top-k. Accuracy varies with the weight α on the original query and the number k of retrieved images. Larger k often introduces noise; smaller α performs better on cluttered datasets.

Table 7: Effect of steering depth on CLS. Top-1 accuracy (%) on CIFAR-100 (ViT-B/16) as the number of steered final layers increases. 

<table><tr><td>Steered Layers</td><td>Accuracy (%)</td></tr><tr><td>12</td><td>68.08</td></tr><tr><td>11 + 12</td><td>65.72</td></tr><tr><td>10 + 11 + 12</td><td>59.36</td></tr></table>

Applying steering at multiple layers likely introduces compounding perturbations that propagate forward, making later representations harder to align with the classifier. CLS steering is most effective at the final layer. Future work could explore coordinated multi-layer steering to avoid error accumulation.

# F Ablation Study of Expansion factor and top-k

In Table 8, we present the downstream task accuracy of CLIP ViT-B/32 using various values of expansion factor and k. Across a 4 × range in width and an 8 × range in sparsity, top-1 accuracy fluctuates by less than one percentage point evidence that VS2’s performance is largely insensitive to the precise SAE capacity–sparsity trade-off. Additionally, in Table 16, we present the average cosine similarities of different steering vectors coming from various configurations of SAEs in terms of expansion factor and top-k.

# G Visual Sparse Steering Pseudocode

For reproducibility purposes, in Algorithm 1, we provide the pseudocode for the baselines and VS2 used in the analysis of 4.1.

Table 8: VS2 accuracy as a function of SAE width (expansion factor) and sparsity (top-k). Numbers are top-1 / top-5 (%). The best result is boldfaced; every other configuration is within of the optimum, highlighting the method’s robustness to architectural choices. 

<table><tr><td>SAE configuration</td><td>Top-1 ↑</td><td>Top-5 ↑</td></tr><tr><td>4 ×, k=128</td><td>64.61</td><td>87.95</td></tr><tr><td>8 ×, k=128</td><td>64.56</td><td>87.76</td></tr><tr><td>4 ×, k=64</td><td>64.54</td><td>87.79</td></tr><tr><td>16 ×, k=64</td><td>64.54</td><td>87.78</td></tr><tr><td>8 ×, k=64</td><td>64.52</td><td>87.96</td></tr><tr><td>10 ×, k=128</td><td>64.43</td><td>87.81</td></tr><tr><td>16 ×, k=512</td><td>64.42</td><td>87.71</td></tr><tr><td>8 ×, k=512</td><td>64.40</td><td>87.91</td></tr><tr><td>4 ×, k=256</td><td>64.34</td><td>87.62</td></tr><tr><td>8 ×, k=256</td><td>64.29</td><td>87.80</td></tr><tr><td>16 ×, k=256</td><td>64.28</td><td>87.75</td></tr><tr><td>4 ×, k=512</td><td>64.12</td><td>87.87</td></tr><tr><td>16 ×, k=128</td><td>64.10</td><td>87.79</td></tr></table>

Table 9: Scaling VS2 to CLIP ViT-L/14. We report zero-shot Top-1 accuracy on the three main datasets. Each row uses a dataset-specific SAE trained on layer 23 with expansion factor E=10 and k=128, with steering applied only at the last layer (last\_n=1). 

<table><tr><td>Dataset</td><td>Eval split (n)</td><td>CLIP ZS</td><td>VS2</td><td>Δ</td></tr><tr><td>CUB-200-2011</td><td>test (5 794)</td><td>62.50</td><td>64.26</td><td>+1.76</td></tr><tr><td>CIFAR-100</td><td>test (10 000)</td><td>72.48</td><td>74.12</td><td>+1.64</td></tr><tr><td>Tiny-ImageNet</td><td>val (10 000)</td><td>72.58</td><td>73.52</td><td>+0.94</td></tr></table>

Algorithm 1 SAE\_STEERING – SAE-based CLS token modification   
1: function SAE_STEERING(h, SAE, k, γ, λ $_{Δz}$ , mode) ▷ h: CLS token; SAE: sparse autoencoder; k: sparsity level
2:    a, idx ← SAE.select_topk(SAE.pre_acts(h), k)
3:    z $_{base}$ ← SAE.decode(a, idx)
4:    z $_{boost}$ ← SAE.decode(γ · a, idx)
5:    if mode = "reconstruction" then
6:    return z $_{base}$ ▷ Variant: CLIP $^{F}_{REC}$ 7:    else if mode = "amplified" then
8:    return z $_{boost}$ ▷ Variant: CLIP $^{F+γ}_{REC}$ 9:    else if mode = "steering" then
10:    return h + λ $_{Δz}$ · (z $_{boost}$ - z $_{base}$ )    ▷ VS2
11:    end if
12: end function

# H Scaling to ViT-L/14

To test whether VS2 transfers beyond the ViT-B/32 and ViT-B/16 backbones used in the main experiments, we apply the same steering pipeline to the larger CLIP ViT-L/14 model (openai/clip-vit-large-patch14; vision tower depth 24, hidden size 1024). For each dataset, we train a dataset-specific top-k SAE on layer 23, corresponding to the final vision block, with expansion factor E=10 and k=128 active latents. Steering is applied only at the last layer (last\_n=1).

Table 9 reports zero-shot Top-1 accuracy on the three datasets used in the main paper. VS2 improves Top-1 accuracy on all three datasets, suggesting that the proposed steering procedure extends to larger CLIP vision backbones. The gains are smaller than those observed for ViT-B/16, which is expected because ViT-L/14 starts from a stronger zero-shot baseline; nevertheless, the consistent positive deltas indicate that SAE-based steering is not specific to the ViT-B family.

![](images/0eca6d5b2594d9c375667c1598bb332a5108cd51d21f9d7f57fc392e82a74761.jpg)  
Figure 7: Top-1 and Top-5 accuracy as a function of number of retrieved neighbors N , using ViT-B/16 (Patch-16) across datasets. Larger N improves general classification but may degrade performance in fine-grained settings.

# I Comparison with other Baseline Methods

We compare VS2 against SpLiCE [Bhalla et al., 2024], using the official implementation provided by the authors. For all three datasets, we report performance using SpLiCE with an external vocabulary of 10,000 LAION-based concepts and an $\ell _ { 1 }$ regularization weight of 0.25, following their best reported configuration. Despite relying on no external vocabulary, VS2 consistently outperforms SpLiCE across all benchmarks highlighting the strength of sparse concept steering even in the absence of external lexical resources.

Table 10: Zero-shot top-1 accuracy (%) on CIFAR-100. 

<table><tr><td>Method</td><td>ViT-B/32</td><td>ViT-B/16</td></tr><tr><td> $CLIP_{ZS}$ </td><td>61.07 (0)</td><td>63.96 (0)</td></tr><tr><td> $SAE_{REC}^{A}$ </td><td>58.01 (-3.06)</td><td>64.05 (+0.09)</td></tr><tr><td> $SAE_{REC}^{F}$ </td><td>58.22 (-2.85)</td><td>63.42 (-0.54)</td></tr><tr><td> $SAE_{REC}^{F+\gamma}$ </td><td>62.69 (+1.62)</td><td>66.81 (+2.85)</td></tr><tr><td>SpLiCE [Bhalla et al., 2024]</td><td>55.57 (-5.50)</td><td>58.29 (-5.67)</td></tr><tr><td>VS2 (ours)</td><td>64.52 (+3.45)</td><td>68.08 (+4.12)</td></tr></table>

# J Effect of Top-N Retrieved Neighbors

To assess the influence of retrieval size on performance, we conduct an ablation over the number of retrieved neighbors N used for latent aggregation. We use a fixed ViT-B/16 (Patch-16) backbone and vary N ∈ {10, 25, 50, 100} across three datasets: CIFAR-100, CUB-200, and Tiny-ImageNet. As shown in Figure 7, both top-1 and top-5 classification accuracy steadily increase with N on CIFAR-100 and Tiny-ImageNet, with diminishing returns beyond N = 50. In contrast, performance on CUB-200 remains largely flat or slightly degrades, suggesting that retrieving more neighbors in fine-grained datasets can introduce noise due to overly similar but semantically irrelevant examples. These results indicate that the utility of neighbor-based aggregation is dataset-dependent: general object recognition tasks may benefit from larger N, while fine-grained classification may require more careful control of retrieval scope.

# K Per-class Performance Analysis on CIFAR-100

In Table 15, we present results for the classes most affected by steering CLIP ViT-B/32 with VS2 and VS2++. We highlight both positive and negative shifts in accuracy to identify the most impacted categories. Unlike the average performance reported in the main experiments, this analysis shows that the top-5 class-level gains are generally larger in magnitude than the corresponding losses. Although the accuracy drops for misclassified categories are smaller in magnitude, they are non-negligible. As future work, it would be valuable to explore when steering should be applied or withheld to further maximize overall performance gains.

Table 11: Benchmarking Visual Sparse Steering: Zero-Shot Accuracy (%) with and Without External Data on CIFAR-100, CUB-200, and Tiny-ImageNet using ViT-B/32 and ViT-B/16. 

<table><tr><td rowspan="2">Method</td><td colspan="2">CIFAR-100</td><td colspan="2">CUB-200</td><td colspan="2">Tiny-IN</td></tr><tr><td>ViT-B/32</td><td>ViT-B/16</td><td>ViT-B/32</td><td>ViT-B/16</td><td>ViT-B/32</td><td>ViT-B/16</td></tr><tr><td colspan="7">Zero-shot (no retrieval)</td></tr><tr><td>CLIP $_{ZS}$ </td><td>61.07 (0)</td><td>63.96 (0)</td><td>51.76 (0)</td><td>55.06 (0)</td><td>56.64 (0)</td><td>61.08 (0)</td></tr><tr><td>SAE $_{REC}^{A}$ </td><td>58.01 (-3.06)</td><td>64.05 (+0.09)</td><td>47.45 (-4.31)</td><td>51.81 (-3.25)</td><td>30.56 (-26.08)</td><td>52.96 (-8.12)</td></tr><tr><td>SAE $_{REC}^{F}$ </td><td>58.22 (-2.85)</td><td>63.42 (-0.54)</td><td>48.08 (-3.68)</td><td>51.43 (-3.63)</td><td>36.33 (-20.31)</td><td>54.84 (-6.24)</td></tr><tr><td>SAE $_{REC}^{F+γ}$ </td><td>62.69 (+1.62)</td><td>66.81 (+2.85)</td><td>49.41 (-2.35)</td><td>53.28 (-1.78)</td><td>39.49 (-17.15)</td><td>58.81 (-2.27)</td></tr><tr><td>VS2 (ours)</td><td>64.52 (+3.45)</td><td>68.08 (+4.12)</td><td>52.69 (+0.93)</td><td>56.14 (+1.08)</td><td>58.14 (+1.50)</td><td>62.92 (+1.84)</td></tr><tr><td>VS2 + PASS (ours)</td><td>70.64 (+9.57)</td><td>68.23 (+4.27)</td><td>52.97 (+1.21)</td><td>56.63 (+1.57)</td><td>57.94 (+1.30)</td><td>62.98 (+1.90)</td></tr></table>

# L Prototype-Aware Sparse Steering Vectors

Our core methods, VS2 and its retrieval-augmented variant ${ \mathrm { V } } { \mathrm { S } } 2 ^ { + + }$ , enhance zero-shot classification by steering CLIP embeddings along directions identified by a top-k Sparse Autoencoder (SAE). These directions correspond to latent features that, ideally, align with class-discriminative concepts. Steering in these directions upweights what the model has learned to be important during reconstruction. This raises a central hypothesis: the reconstruction task itself is to some extent sufficient to uncover features that are also relevant for classification. In other words, there is a meaningful overlap between features that are important for reconstructing the CLS token and those that are predictive for the downstream task. In this section, drawing inspiration from prototype theory [Rosch, 1973], we investigate whether incorporating prototype information during SAE training can better align the features important for reconstruction with those that are critical for downstream classification.

The limited improvements observed on fine-grained datasets like CUB-200 and Tiny-ImageNet suggest that the challenge lies not just in identifying sparse features, but in uncovering the correct ones. This shifts the central question from “What are the most important sparse features to select?” to a deeper inquiry: “Can the reconstruction objective alone reliably capture features that are most useful for classification and if not, how can task-relevant information be effectively incorporated?”

Oracle steering with known prototypes. Following prototype theory, we collect for every class the ten images that ViT-B/32 CLIP classifies with the highest confidence; these serve as oracle prototypes. Given the true class label y, we build a prototype steering vector by averaging their latent sparse features. VS2 using oracle prototypes can lift CLIP to 97.5% on CIFAR-100, 91.04% on CUB-200, and 90.1% on Tiny-ImageNet. This confirms the existence of discriminative sparse directions. In Appendix M we further examine the discriminative ability (measured by orthogonality) of these steering vectors. Generally, these prototype vectors have a low cosine similarity, yet a non-negligible tail reveals strongly overlapping directions between visually or semantically close categories.

Prototype-aligned SAE (PASS). Above, we constructed steering vectors using the pretrained SAE features of oracle prototypes. Now, we consider whether prototypes can be used to learn new, more informative SAE features. We assume during SAE training that we have access to class labels. Then, to the SAE loss we add a regularization term which encourages SAE features to be close to their class mean. That is, for a training sample i with latent sparse feature $\mathbf { z } _ { i }$ and class mean $\bar { \mathbf { z } } _ { \mathrm { c l a s s } ( i ) }$ we minimize

$$
\mathcal {L} = \mathcal {L} _ {\text { recon }} + w _ {\text { aux }} \left\| \mathbf {z} _ {i} - \bar {\mathbf {z}} _ {\text { class } (i)} \right\| _ {2} ^ {2}, \tag {7}
$$

where $w _ { \mathrm { a u x } }$ controls the strength of the prototype-alignment term relative to the reconstruction loss and is set to 0.8. We refer to the resulting steering method as PASS (Prototype-Aligned Sparse Steering). Although PASS uses class labels during SAE training, it remains fully test-time unsupervised. Empirically, in Table 11, we observe that PASS outperforms VS2 across all datasets, with particularly substantial gains on CIFAR-100. However, this improvement comes at the cost of requiring labels for each training sample during SAE training. Gains are modest on CUB-200 and Tiny-ImageNet, and we thus hypothesize that classes which share many features require richer or multi-prototype guidance which is an intriguing avenue for future work.

Table 12: Trade-off between reconstruction fidelity and prototype alignment. Increasing $w _ { \mathrm { a u x } }$ improves classification accuracy but degrades FVU reconstruction. 

<table><tr><td> $w_{aux}$ </td><td>FVU ↓</td><td>Accuracy (%) ↑</td></tr><tr><td>0.1</td><td>0.3437</td><td>68.80</td></tr><tr><td>0.5</td><td>0.4887</td><td>70.40</td></tr><tr><td>1.0</td><td>0.5393</td><td>70.61</td></tr><tr><td>2.0</td><td>0.5702</td><td>70.76</td></tr></table>

Reconstruction vs. Alignment Trade-off Sparse Autoencoders (SAEs) trained for reconstruction can also be optimized to align their latent features with class-level prototypes. However, this introduces a trade-off between two competing objectives: fidelity of reconstruction and discriminative alignment.

To investigate this trade-off, we introduce a weighting coefficient $w _ { \mathrm { a u x } }$ that controls the strength of prototype alignment relative to the reconstruction loss. As $w _ { \mathrm { a u x } }$ increases, alignment is encouraged more strongly. Table 12 reports the resulting changes in reconstruction loss (measured by FVU) and top-1 classification accuracy on CIFAR-100 using ViT-B/16, with 128 sparse latents and expansion factor 4.

We observe that increasing $w _ { \mathrm { a u x } }$ consistently improves classification performance, up to a point, even though it introduces more reconstruction error. This aligns with our hypothesis: while exact input reconstruction encourages general feature coverage, alignment with class prototypes promotes discriminative feature extraction. These results highlight the flexibility of SAE-based steering to balance interpretability and performance depending on downstream objectives.

# M Are exemplar-derived directions really distinct?

A desirable property of class–specific steering vectors is orthogonality: pushing an embedding toward class A should not simultaneously raise its score for class B. Using the oracle prototypes as described in Appendix L, we compute for every CIFAR-100 class a prototype steering vector and measure the pair-wise cosine similarity at layer 11 of the ViT-B/32 encoder. Most pairs have low similarity (mean=0.23), yet a non-negligible tail reveals strongly overlapping directions. Table 13 lists the ten highest-overlap pairs.

Table 13: Top-10 most overlapping prototype steering directions on CIFAR-100. High cosine similarity indicates that the two classes share visual attributes that the SAE encodes along nearly the same latent axis. 

<table><tr><td>Rank</td><td>Class 1</td><td>Class 2</td><td>Cosine ↑</td></tr><tr><td>1</td><td>beetle</td><td>cockroach</td><td>0.91</td></tr><tr><td>2</td><td>mouse</td><td>shrew</td><td>0.89</td></tr><tr><td>3</td><td>dolphin</td><td>shark</td><td>0.84</td></tr><tr><td>4</td><td>otter</td><td>seal</td><td>0.84</td></tr><tr><td>5</td><td>dolphin</td><td>whale</td><td>0.84</td></tr><tr><td>6</td><td>possum</td><td>raccoon</td><td>0.84</td></tr><tr><td>7</td><td>snake</td><td>worm</td><td>0.83</td></tr><tr><td>8</td><td>oak tree</td><td>willow tree</td><td>0.83</td></tr><tr><td>9</td><td>ray</td><td>shark</td><td>0.81</td></tr><tr><td>10</td><td>bowl</td><td>cup</td><td>0.80</td></tr></table>

These high-overlap pairs are semantically plausible confusions (e.g. beetle vs. cockroach or dolphin vs. whale), confirming that exemplar steering directions tend to align for visually or taxonomically proximate classes. In downstream applications, a simple orthogonalization step may help reduce feature overlap between sparse directions. Investigating principled ways to encourage orthogonality during SAE training is a promising direction for future work.

Table 14: Training-time compute and parameter cost per sample for adaptation strategies on the CLIP ViT-B/32 vision encoder. 

<table><tr><td>Method</td><td>Train FLOPs / sample</td><td>GFLOPs</td><td>Trainable Params</td></tr><tr><td>SAE (cached activations)</td><td> $1.42 \times 10^{7}$ </td><td>0.014</td><td>4.72M</td></tr><tr><td>SAE (non-cached; incl. extraction)</td><td> $8.74 \times 10^{9}$ </td><td>8.744</td><td>4.72M</td></tr><tr><td>LoRA (rank = 16)</td><td> $2.64 \times 10^{10}$ </td><td>26.365</td><td>0.59M</td></tr><tr><td>Full fine-tuning</td><td> $2.62 \times 10^{10}$ </td><td>26.188</td><td>87.46M</td></tr></table>

Table 15: Top-5 class gains and losses on CIFAR-100. Green = absolute gain; Red = absolute loss relative to ZS baseline.

(a) VS2 – Top-5 Gains 

<table><tr><td>Class</td><td>ZS</td><td>VS2 ↑(Δ)</td></tr><tr><td>tractor</td><td>0.55</td><td>0.80 ↑(+0.25)</td></tr><tr><td>forest</td><td>0.35</td><td>0.58 ↑(+0.23)</td></tr><tr><td>man</td><td>0.53</td><td>0.74 ↑(+0.21)</td></tr><tr><td>bus</td><td>0.51</td><td>0.68 ↑(+0.17)</td></tr><tr><td>snake</td><td>0.61</td><td>0.76 ↑(+0.15)</td></tr></table>

(c) VS2++ – Top-5 Gains

<table><tr><td>Class</td><td>ZS</td><td>VS2++ ↑(Δ)</td></tr><tr><td>spider</td><td>0.48</td><td>0.91 ↑(+0.43)</td></tr><tr><td>caterpillar</td><td>0.27</td><td>0.68 ↑(+0.41)</td></tr><tr><td>possum</td><td>0.24</td><td>0.65 ↑(+0.41)</td></tr><tr><td>tractor</td><td>0.55</td><td>0.96 ↑(+0.41)</td></tr><tr><td>tiger</td><td>0.45</td><td>0.84 ↑(+0.39)</td></tr></table>

(b) VS2 – Top-5 Losses 

<table><tr><td>Class</td><td>ZS</td><td>VS2 (Δ)</td></tr><tr><td>aquarium_fish</td><td>0.82</td><td>0.61 (-0.21)</td></tr><tr><td>beetle</td><td>0.64</td><td>0.49 (-0.15)</td></tr><tr><td>sweet_pepper</td><td>0.70</td><td>0.58 (-0.12)</td></tr><tr><td>tulip</td><td>0.78</td><td>0.71 (-0.07)</td></tr><tr><td>maple_tree</td><td>0.57</td><td>0.50 (-0.07)</td></tr></table>

(d) VS2++ – Top-5 Losses

<table><tr><td>Class</td><td>ZS</td><td>VS2++ (Δ)</td></tr><tr><td>girl</td><td>0.72</td><td>0.65 (-0.07)</td></tr><tr><td>maple_tree</td><td>0.57</td><td>0.51 (-0.06)</td></tr><tr><td>porcupine</td><td>0.18</td><td>0.13 (-0.05)</td></tr><tr><td>ray</td><td>0.06</td><td>0.02 (-0.04)</td></tr><tr><td>mouse</td><td>0.18</td><td>0.15 (-0.03)</td></tr></table>

# N Computational Overhead: Additional Details

Training-time FLOPs. To quantify the cost of training the SAE versus alternative adaptation methods, Table 14 reports per-sample training FLOPs (forward + backward) and the number of trainable parameters for the CLIP ViT-B/32 vision encoder. Even when activation extraction is included, SAE training remains more than 3× cheaper than LoRA or full fine-tuning. With cached activations, as in our VS2 setup, the cost drops to just 0.014 GFLOPs per sample.

TPT FLOP accounting. For completeness, we outline how the 16.6× test-time overhead factor for TPT [Shu et al., 2022] is obtained. A single CLIP ViT-B/32 forward pass on a 224 × 224 image with 100 text prompts costs 8.7 GFLOPs for the vision encoder and 581.8 GFLOPs for the text encoder, for a total of approximately 0.59 TFLOPs. In our implementation, TPT performs 63 augmented views per image and runs 4 optimization steps; each step requires a full CLIP forward pass over all views and a backward pass through the text encoder. This yields about 9.217 TFLOPs of adaptation compute per test image, followed by one final forward-only evaluation of 0.591 TFLOPs, for a total of roughly 9.807 TFLOPs. Dividing by the 0.59 TFLOPs of plain CLIP gives the reported ≈ 16.6× test-time compute overhead.

Table 16: Cosine similarity (↑) between steering vectors learned by different SAE capacities. Rows/columns are ordered by expansion factor and sparsity (e × expansion, k active). 

<table><tr><td>SAE cfg.</td><td>16x164</td><td>16x512</td><td>16x128</td><td>16x256</td><td>8x128</td><td>8x512</td><td>8x256</td><td>4x256</td><td>4x512</td><td>4x128</td><td>10x128</td><td>4x64</td><td>8x64</td></tr><tr><td>16x/64</td><td>1.000</td><td>0.497</td><td>0.577</td><td>0.705</td><td>0.726</td><td>0.635</td><td>0.680</td><td>0.640</td><td>0.591</td><td>0.681</td><td>0.536</td><td>0.701</td><td>0.716</td></tr><tr><td>16x/512</td><td>0.497</td><td>1.000</td><td>0.161</td><td>0.399</td><td>0.358</td><td>0.817</td><td>0.634</td><td>0.618</td><td>0.611</td><td>0.584</td><td>0.136</td><td>0.316</td><td>0.282</td></tr><tr><td>16x/128</td><td>0.577</td><td>0.161</td><td>1.000</td><td>0.900</td><td>0.865</td><td>0.273</td><td>0.298</td><td>0.277</td><td>0.265</td><td>0.303</td><td>0.952</td><td>0.887</td><td>0.678</td></tr><tr><td>16x/256</td><td>0.705</td><td>0.399</td><td>0.900</td><td>1.000</td><td>0.956</td><td>0.527</td><td>0.572</td><td>0.495</td><td>0.410</td><td>0.542</td><td>0.920</td><td>0.936</td><td>0.744</td></tr><tr><td>8x/128</td><td>0.726</td><td>0.358</td><td>0.865</td><td>0.956</td><td>1.000</td><td>0.543</td><td>0.650</td><td>0.575</td><td>0.483</td><td>0.634</td><td>0.896</td><td>0.959</td><td>0.829</td></tr><tr><td>8x/512</td><td>0.635</td><td>0.817</td><td>0.273</td><td>0.527</td><td>0.543</td><td>1.000</td><td>0.861</td><td>0.811</td><td>0.771</td><td>0.796</td><td>0.243</td><td>0.483</td><td>0.506</td></tr><tr><td>8x/256</td><td>0.680</td><td>0.634</td><td>0.298</td><td>0.572</td><td>0.650</td><td>0.861</td><td>1.000</td><td>0.889</td><td>0.771</td><td>0.912</td><td>0.289</td><td>0.557</td><td>0.624</td></tr><tr><td>4x/256</td><td>0.640</td><td>0.618</td><td>0.277</td><td>0.495</td><td>0.575</td><td>0.811</td><td>0.889</td><td>1.000</td><td>0.873</td><td>0.941</td><td>0.240</td><td>0.532</td><td>0.600</td></tr><tr><td>4x/512</td><td>0.591</td><td>0.611</td><td>0.265</td><td>0.410</td><td>0.483</td><td>0.771</td><td>0.771</td><td>0.873</td><td>1.000</td><td>0.821</td><td>0.191</td><td>0.461</td><td>0.549</td></tr><tr><td>4x/128</td><td>0.681</td><td>0.584</td><td>0.303</td><td>0.542</td><td>0.634</td><td>0.796</td><td>0.912</td><td>0.941</td><td>0.821</td><td>1.000</td><td>0.284</td><td>0.594</td><td>0.655</td></tr><tr><td>10x/128</td><td>0.536</td><td>0.136</td><td>0.952</td><td>0.920</td><td>0.896</td><td>0.243</td><td>0.289</td><td>0.240</td><td>0.191</td><td>0.284</td><td>1.000</td><td>0.910</td><td>0.678</td></tr><tr><td>4x/64</td><td>0.701</td><td>0.316</td><td>0.887</td><td>0.936</td><td>0.959</td><td>0.483</td><td>0.557</td><td>0.532</td><td>0.461</td><td>0.594</td><td>0.910</td><td>1.000</td><td>0.814</td></tr><tr><td>8x/64</td><td>0.716</td><td>0.282</td><td>0.678</td><td>0.744</td><td>0.829</td><td>0.506</td><td>0.624</td><td>0.600</td><td>0.549</td><td>0.655</td><td>0.678</td><td>0.814</td><td>1.000</td></tr></table>