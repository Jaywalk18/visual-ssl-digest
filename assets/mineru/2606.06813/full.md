# Breaking the Lock-in: Diversifying Text-to-Image Generation via Representation Modulation

Dahee Kwon 1 Haeun Lee 1 † Jaesik Choi 1 2

# Abstract

Recent text-to-image models built on large-scale Transformer backbones and flow-based objectives deliver strong text–image alignment and high visual quality, yet often produce overly similar samples under a fixed prompt. Existing diversityenhancement methods alleviate this, but typically require expensive sampling or auxiliary optimization, incurring non-trivial overhead. To investigate the root cause of this homogeneity, we examine intermediate Transformer features and observe that the zero-frequency spatial average (DC) component rapidly converges across seeds early in generation, causing early trajectory lock-in that limits downstream variation. Building on this, we propose DC Attenuation for diVersity Enhancement (DAVE), a training-free representation-level intervention that selectively attenuates this component in the early regime. DAVE preserves the sampling pipeline with negligible overhead, improving prompt-consistent diversity while maintaining competitive image quality.

# 1. Introduction

Recent advancements in text-to-image (T2I) generative models—particularly those leveraging scalable Transformer architectures and flow-based objectives—have established these models as a dominant paradigm in generative modeling, featuring remarkable text–image alignment and photorealism. These models enable users to reliably produce high-quality visual content, driving widespread adoption across diverse domains (Croitoru et al., 2023; Esser et al., 2024; Chen et al., 2025).

†Work done during an undergraduate internship at KAIST. 1Korea Advanced Institute of Science and Technology (KAIST) 2INEEJI. Correspondence to: Jaesik Choi <jaesik.choi@kaist.ac.kr>.

Proceedings of the $\ 4 \mathcal { 3 } ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

However, this reliability in quality is often accompanied by a reduction in generation diversity, defined as variation in under-specified visual factors (e.g., layout, style) while preserving prompt-conditioned semantics. When generating multiple images from a fixed prompt, outputs often exhibit limited variation, converging to similar compositions or stylistic patterns (Mukhopadhyay et al., 2023; Astolfi et al., 2024; Albuquerque et al., 2025). Insufficient diversity hampers users’ ability to explore broad candidates, discover rare configurations, and construct synthetic datasets with wide distributional coverage, ultimately reducing downstream performance gains. Consequently, diversity is not merely an optional add-on, but a core performance axis that determines the usability and scalability of generative models.

A growing body of research has recently turned its attention to this issue, with many approaches introducing diversity control directly at sampling time. By tuning guidance schedules (Um & Ye, 2025; Sadat et al., 2023), injecting stochastic perturbations (Harrington et al., 2025), or modifying the sampling process (Morshed & Boddeti, 2025; Corso et al., 2023), these training-free methods can steer generation behavior to recover diversity while preserving visual quality.

However, existing approaches face two practical limitations. First, they often incur substantial computational and memory overhead. Enhancing diversity typically requires extra sampling steps, auxiliary optimization, or parallel candidate generation across multiple seeds—all of which exacerbate memory consumption and decoding costs. As modern generative models scale, even modest overheads become a significant bottleneck. Second, the fundamental causes of limited diversity remain insufficiently understood. While prior studies analyze diversity through inference-time controls, a detailed mechanistic account of exactly where and why this collapse occurs within the model’s internal representations remains largely unexplored.

From this perspective, we approach diversity control through the lens of representation-level interventions. By analyzing intermediate Transformer representations, we observe that the zero-frequency spatial average (DC) component exhibits strong trajectory lock-in in the early denoising steps across seeds—a phase that coincides with the formation of global layouts and coarse semantic structures. Motivated by this observation, we propose DC Attenuation for diVersity Enhancement (DAVE), which selectively attenuates this component at inference time via a lightweight internal operation. Requiring neither retraining nor changes to the sampling pipeline, this simple adjustment incurs virtually no computational or memory overhead and imposes no batch size constraints. Through extensive experiments, we demonstrate that DAVE achieves competitive performance against a range of state-of-the-art methods while incurring substantially lower overhead.

![](images/419ca504290db0075648aa82c3e25a4103b83c87a78370e6a9f1f01f557c9b06.jpg)  
Figure 1. Overview of DAVE. Compared to the original flow-based generative model, DAVE consistently produces more diverse images with minimal computation.

# 2. Related Work

Diversity Enhancement Recent efforts to mitigate diversity collapse in T2I models broadly fall into two categories: batch-wise diversification and individual trajectory manipulation. Most approaches adopt the former, explicitly promoting diversity within a batch by encouraging jointly generated samples to diverge. For instance, Particle Guidance (PG) (Corso et al., 2023) adds a pairwise gradient potential to push samples apart, while DiverseFlow (Morshed & Boddeti, 2025) optimizes a batch-wide diversity objective along the trajectory. To refine this repulsion, SPELL (Kirchhof et al., 2024) introduces sparse repellency terms that activate only when trajectories risk collapsing onto one another. Similarly, OSCAR (Wu et al., 2025) maximizes the feature-space volume of a batch and injects stochasticity projected orthogonally to the flow, spreading trajectories without harming fidelity. Extending this concept beyond a single batch, SPARKE (Jalali et al., 2026) guides sampling with a prompt-aware Renyi kernel entropy score to ´ scale efficiently across large prompt sets. However, these joint-evaluation methods incur non-trivial computational and memory overheads, which become increasingly burdensome as generative models scale up.

An alternative line of approach manipulates each trajectory directly rather than comparing samples. For example, CADS (Sadat et al., 2023) injects scheduled noise into the text condition during early inference to prevent over-reliance on the prompt and recover diversity suppressed at high guidance scales. Yet, these methods still operate primarily as sampling-level heuristics. They offer little insight into the internal mechanisms that drive generations toward similar outputs in the first place, leaving the root cause of diversity collapse largely unexamined.

Internal Representation Analysis A complementary line of research examines the internal representations of T2I models to improve their interpretability and controllability (Kim et al., 2025; Shin et al., 2025; Li et al., 2026; Yang et al., 2025; Dalva et al., 2024; Si et al., 2024; Han et al., 2025). Several works manipulate intermediate features to steer diffusion or flow dynamics at inference time, such as attention-based editing (Cao et al., 2023; Voynov et al., 2023; Hertz et al., 2022) and architectural rebalancing (Si et al., 2024). Beyond controllable editing, modulating internal features can also foster generation creativity, as seen in C3 (Han et al., 2025). Yet, even such diversity-oriented approaches overlook the fundamental mechanism of diversity collapse. They rarely investigate where and when the trajectory locks in. In contrast, we analyze the internal feature dynamics of Transformer backbones, pinpointing early DC convergence as a major representational bottleneck for seed-level variation. Consequently, we propose DAVE, a direct representation modulation approach that enhances diversity without additional sampling constraints.

# 3. Method

In this section, we first investigate the representation-level mechanisms driving diversity collapse, motivating our proposed method: DC Attenuation for diVersity Enhancement (DAVE). By selectively attenuating the internal DC component during the early stages of generation, DAVE broadens the model’s generative scope with minimal complexity.

# 3.1. Preliminary

Text-conditioned flow matching generates samples by transporting a source distribution $X _ { 0 }$ to the data distribution $X _ { 1 }$ . In practice, we take $p _ { 0 } = \mathcal { N } ( 0 , I )$ , sample $x _ { 0 } \sim p _ { 0 }$ , and evolve it over time via a learned vector field. The resulting trajectory $x _ { t }$ follows the Ordinary Differential Equation:

$$
\frac {d x _ {t}}{d t} = v _ {\theta} (x _ {t}, t; c), \quad t \in [ 0, 1 ], \tag {1}
$$

where c denotes the conditioning signal $( \mathrm { e . g . }$ ., a text embedding) and $v _ { \theta } ( x _ { t } , t ; c )$ specifies the instantaneous direction of evolution (velocity) at time t.

Sampling is performed by discretizing Eq. (1) into K steps. Starting from $x _ { 0 } \sim p _ { 0 }$ , we iteratively update the state according to a numerical solver:

$$
x _ {k + 1} = x _ {k} + \Delta t v _ {\theta} (x _ {k}, t _ {k}; c), \quad k = 0, \dots , K - 1, \tag {2}
$$

where $t _ { k }$ and $\Delta t$ denote the discretized time and step size, respectively. The final generated sample is $x _ { K }$ ≈ $x _ { 1 }$ .

In modern text-to-image architectures, the vector field $v _ { \theta }$ generally adopts a Transformer backbone. Let $h _ { t } ^ { ( \ell ) }$ denote the hidden representation at block ℓ. A single Transformer block updates the representation as:

$$
h _ {t} ^ {(\ell + 1)} = \text { Block } ^ {(\ell + 1)} (h _ {t} ^ {(\ell)}, t, c). \tag {3}
$$

The velocity prediction $v _ { \theta } ( x _ { t } , t ; c )$ at step t is computed from the last-block representation.

# 3.2. Lock-in on DC component

To investigate the mechanisms underlying this diversity degradation, we analyze the internal representations of transformer blocks. This investigation reveals a notable pattern: a pervasive global bias that manifests as a systematic drift in hidden states during generation. Specifically, a comparison of hidden representations $h _ { t } ^ { ( \ell ) }$ across various noise seeds shows that the zero-frequency component—namely, the DC component—becomes remarkably aligned across samples. Using SD3 (Esser et al., 2024), we analyze representations from Transformer block 5 over 100 random seeds per prompt. The DC component shows high consistency across these seeds, exhibiting high pairwise cosine similarity and a low coefficient of variation (Figure 2). Accounting for 51.2% of the total energy, this component constitutes a dominant global signal rather than a minor artifact—a phenomenon we term early DC drift.

![](images/8fb3dfb395f1bb1ba2bf313ead31dfae8be17da2667cba53ca20674b87112230.jpg)

<details>
<summary>bar</summary>

| Category | Coef. of Variation | Cosine Similarity |
| :--- | :--- | :--- |
| Non-DC | 0.777 | 0.105 |
| DC | 0.001 | 0.998 |
</details>

(a)

![](images/8e9e1e78804e2766cea7b92c62780d843bf8e75fbb58d1104c4caf9032be3d18.jpg)

<details>
<summary>line</summary>

| Energy Band | Energy Ratio |
| ----------- | ------------ |
| Top Point   | 51.2%        |
</details>

(b)   
Figure 2. (left) DC variation across seeds; “non-DC” denotes representations with the DC component removed. (right) Band energy ratios (band/total), where the leftmost bin corresponds to the DC (lowest-frequency) component.

We posit that early DC drift is a major bottleneck of reduced sample diversity, arising from a synergy between architectural bias and the training objective. Since the DC component represents the zero-frequency spatial average, its early dominance is directly explained by the spectral bias of neural networks (Rahaman et al., 2019; Wang & Pehlevan, 2026)—their tendency to prioritize low-frequency signals over high-frequency details. This bias directs the model’s focus toward the DC component at the onset of sampling, consistent with the early emergence of global structures in the generative process (Choi et al., 2021; Liu et al., 2025; Esser et al., 2024).

This tendency is further exacerbated by MSE-based flowmatching objectives under high uncertainty. In early sampling steps characterized by a low Signal-to-Noise Ratio (SNR), the objective is mathematically minimized when the model predicts the text-conditioned expectation of the data. While high-frequency, sample-specific textures average out across seeds, the DC component remains a robust cue of global statistics. Driven by this dual bias, the DC component is strongly pulled toward the conditional mean, establishing a seed-invariant anchor that dictates the generative trajectory. Consequently, the global layout tends to lock in so early that it severely restricts the ability of the initial noise seed to induce sufficient structural variation. A step-wise analysis of the denoising trajectory confirms that this cross-seed DC alignment is indeed concentrated in the earliest steps and decays toward the end of generation (See Table 3). Appendix E formalizes how early DC-dominated alignment can limit trajectory separation and diversity.

![](images/6094e6ce228cc95c4816e4fee0b0cd734df12540a2dcf0b6378e602fb5d8ebed.jpg)

<details>
<summary>bar</summary>

| Dataset | Metric Value |
|---|---|
| sd3 | 0.75 |
| sd3.5 | 0.78 |
| FLUX | 0.71 |
| SANA | 0.76 |
</details>

Figure 3. (Left) Outputs with DC attenuation for “A small bird standing on a rocky beach.” and “A beautiful girl with long brown hair.” (Right) Perceptual changes and text–image alignment under DC attenuation; CLIP (Orig) is the unmanipulated CLIP score.

# 3.3. DC Attenuation for Diversity Enhancement

Given that early DC drift acts as a restrictive global anchor, we hypothesize that directly weakening its influence can unlock the generative trajectory from its premature structural commitment. To test this, we selectively attenuate the DC component during early sampling steps to suppress this dominant global bias. As shown in Figure 3, this targeted intervention substantially changes the final image layouts without compromising semantic alignment. These observations confirm that modulating the early-stage DC component is a highly effective strategy for diversity enhancement.

Building on these empirical insights, we propose DC Attenuation for diVersity Enhancement (DAVE). DAVE is designed to broaden the model’s generative scope by strategically dampening the dominant DC signal during early generation stages. By dismantling this seed-invariant anchor, the method empowers the stochasticity of the initial noise to drive meaningful structural variations, breaking the homogenizing effect of the conditional mean.

For a hidden representation h(t $h _ { t } ^ { ( \ell ) } \in \mathbb { R } ^ { D \times C }$ , where D is the number of visual tokens and $C$ is the channel dimension, the DC component $\mu _ { t } ^ { ( \ell ) }$ is straightforwardly obtained as the spatial mean, bypassing the need for explicit frequencydomain transforms:

$$
\mu_ {t} ^ {(\ell)} = \frac {1}{D} \sum_ {d = 1} ^ {D} h _ {t} ^ {(\ell)} [ d,: ] \in \mathbb {R} ^ {1 \times C}. \tag {4}
$$

Since our focus is on spatial structures, we isolate and manipulate only the representations containing visual information, particularly in multi-modal architectures like MMDiT. DAVE intervenes on the output of Transformer blocks $\ell \in { \mathcal { L } }$ during the early generative phase $( t < \tau )$ :

$$
\hat {h} _ {t} ^ {(\ell)} = \alpha \cdot \mu_ {t} ^ {(\ell)} + \left(h _ {t} ^ {(\ell)} - \mu_ {t} ^ {(\ell)}\right). \tag {5}
$$

where $\alpha \in ( 0 , 1 )$ controls the attenuation strength, and $\mu _ { t } ^ { ( \ell ) }$ is broadcast across all D tokens when subtracted from (and added to) $h _ { t } ^ { ( \ell ) }$ . We substitute the original hidden representation h(ℓ)t w $h _ { t } ^ { ( \ell ) }$ ith the modified $\hat { h } _ { t } ^ { ( \ell ) }$ before it is passed to the subsequent Transformer block $( \ell + 1 )$ . As a surgical, trainingfree intervention, DAVE integrates seamlessly into existing pre-trained models without requiring any architectural modifications or additional optimizations. By attenuating the early-stage DC component, DAVE prevents premature anchoring to a common structural baseline, thereby amplifying the relative influence of seed-specific spatial residuals.

Parameter Selection. DAVE is modulated by three key parameters: the attenuation strength α, the target block pool ${ \mathcal { L } } ,$ and the temporal cutoff τ . We outline practical guidance for selecting each parameter below, with a comprehensive empirical analysis of these configurations in Section 4.2.

Temporal Cutoff (τ ): Following the principle of early-stage intervention, we apply DC attenuation during the early generative regime, i.e., for timesteps satisfying $t < \tau$ . Based on the trade-off between output variation and perceptual quality observed in our ablations, we find that intervening within the first 15–20% of the generative process yields stable diversity gains across the evaluated architectures.

Attenuation Strength (α): The coefficient $\alpha \in ( 0 , 1 )$ regulates the magnitude of DC suppression. While a smaller $\alpha$ induces more pronounced structural deviations, excessive attenuation may lead to perceptual artifacts or loss of semantic coherence. To balance structural diversification with text-alignment preservation, we recommend setting $\alpha \in [ 0 . 2 , 0 . 5 ]$ . This enables adjustable intervention intensity based on the desired level of variation.

Target block pool (L): We define L as the set of Transformer blocks whose DC component shows strong cross-seed convergence, aligning across noise realizations. Empirically, DAVE is robust to block choice within this pool; interventions on blocks in $\mathcal { L }$ yield consistent diversity gains. Although the specific indices of L vary across architectures, these variations reflect structural differences rather than adhoc, per-model heuristics. Moreover, the identified blocks largely coincide with structurally significant modules reported in prior literature (Li et al., 2026; Yang et al., 2025), further supporting the architectural grounding of our selection process.

Table 1. Quantitative results for diverse generation in the independently sampled generation setting, where samples are generated without explicit in-batch interaction. Bold indicates the best result for each metric, and underlining indicates the second-best. 

<table><tr><td>Dataset</td><td>Model</td><td>Method</td><td>FID ↓</td><td>Prec ↑</td><td>Rec ↑</td><td>Cov ↑</td><td>Dens ↑</td><td>Vendi ↑</td><td>CLIP ↑</td></tr><tr><td rowspan="18">ImageNet</td><td rowspan="6">SD3.5</td><td>Orig</td><td>22.23</td><td>0.8604</td><td>0.2589</td><td>0.7757</td><td>1.0071</td><td>1.71</td><td>0.2952</td></tr><tr><td>Orig ( $\omega_{CFG} = 2$ )</td><td>17.13</td><td>0.8144</td><td>0.5106</td><td>0.8032</td><td>0.9311</td><td>2.05</td><td>0.2861</td></tr><tr><td>CADS</td><td>17.91</td><td>0.7856</td><td>0.5698</td><td>0.7936</td><td>0.8386</td><td>2.09</td><td>0.2907</td></tr><tr><td>SPARKE</td><td>22.27</td><td>0.8686</td><td>0.3136</td><td>0.4004</td><td>1.2289</td><td>1.77</td><td>0.3016</td></tr><tr><td>Ours</td><td>20.74</td><td>0.8090</td><td>0.6489</td><td>0.7895</td><td>0.7810</td><td>2.33</td><td>0.2897</td></tr><tr><td>Ours Random</td><td>17.57</td><td>0.8591</td><td>0.5422</td><td>0.8371</td><td>1.0114</td><td>2.50</td><td>0.2939</td></tr><tr><td rowspan="6">Flux.1-dev</td><td>Orig</td><td>24.01</td><td>0.8219</td><td>0.3149</td><td>0.7555</td><td>0.9371</td><td>1.88</td><td>0.2902</td></tr><tr><td>Orig ( $\omega_{CFG} = 2$ )</td><td>21.00</td><td>0.8220</td><td>0.4064</td><td>0.7726</td><td>0.9475</td><td>2.09</td><td>0.2852</td></tr><tr><td>CADS</td><td>39.97</td><td>0.5914</td><td>0.5172</td><td>0.5913</td><td>0.5045</td><td>2.69</td><td>0.2689</td></tr><tr><td>SPARKE</td><td>22.65</td><td>0.8298</td><td>0.3561</td><td>0.3935</td><td>1.1112</td><td>2.04</td><td>0.2921</td></tr><tr><td>Ours</td><td>25.31</td><td>0.7112</td><td>0.5102</td><td>0.6963</td><td>0.7189</td><td>2.26</td><td>0.2858</td></tr><tr><td>Ours Random</td><td>24.36</td><td>0.6927</td><td>0.6032</td><td>0.7295</td><td>0.6869</td><td>2.44</td><td>0.2818</td></tr><tr><td rowspan="6">SANA1.5</td><td>Orig</td><td>27.85</td><td>0.7860</td><td>0.1623</td><td>0.6620</td><td>0.8506</td><td>1.59</td><td>0.2918</td></tr><tr><td>Orig ( $\omega_{CFG} = 2$ )</td><td>22.44</td><td>0.7986</td><td>0.2846</td><td>0.7182</td><td>0.8963</td><td>1.80</td><td>0.2827</td></tr><tr><td>CADS</td><td>68.13</td><td>0.3952</td><td>0.5774</td><td>0.4525</td><td>0.3062</td><td>2.32</td><td>0.2618</td></tr><tr><td>SPARKE</td><td>27.87</td><td>0.7774</td><td>0.2220</td><td>0.3013</td><td>0.9739</td><td>1.67</td><td>0.2935</td></tr><tr><td>Ours</td><td>25.16</td><td>0.7588</td><td>0.5254</td><td>0.6435</td><td>0.6422</td><td>2.20</td><td>0.2885</td></tr><tr><td>Ours Random</td><td>22.41</td><td>0.7419</td><td>0.4911</td><td>0.6990</td><td>0.7935</td><td>2.15</td><td>0.2879</td></tr><tr><td rowspan="18">MSCOCO</td><td rowspan="6">SD3.5</td><td>Orig</td><td>36.38</td><td>0.8208</td><td>0.2546</td><td>0.7928</td><td>1.1345</td><td>2.29</td><td>0.3129</td></tr><tr><td>Orig ( $\omega_{CFG} = 2$ )</td><td>28.48</td><td>0.7864</td><td>0.4282</td><td>0.8222</td><td>0.9846</td><td>1.83</td><td>0.3047</td></tr><tr><td>CADS</td><td>28.97</td><td>0.7138</td><td>0.4604</td><td>0.7896</td><td>0.8232</td><td>2.30</td><td>0.3065</td></tr><tr><td>SPARKE</td><td>35.39</td><td>0.7938</td><td>0.2730</td><td>0.7792</td><td>1.0606</td><td>1.76</td><td>0.3174</td></tr><tr><td>Ours</td><td>29.56</td><td>0.6918</td><td>0.4916</td><td>0.8586</td><td>0.7368</td><td>2.55</td><td>0.3055</td></tr><tr><td>Ours Random</td><td>29.75</td><td>0.7302</td><td>0.4406</td><td>0.8835</td><td>0.7866</td><td>2.37</td><td>0.3083</td></tr><tr><td rowspan="6">Flux.1-dev</td><td>Orig</td><td>40.22</td><td>0.8086</td><td>0.2212</td><td>0.7678</td><td>1.0844</td><td>1.68</td><td>0.3042</td></tr><tr><td>Orig ( $\omega_{CFG} = 2$ )</td><td>35.89</td><td>0.8292</td><td>0.2934</td><td>0.8114</td><td>1.1465</td><td>1.73</td><td>0.3014</td></tr><tr><td>CADS</td><td>44.83</td><td>0.7321</td><td>0.3326</td><td>0.7192</td><td>0.9499</td><td>2.43</td><td>0.2656</td></tr><tr><td>SPARKE</td><td>37.75</td><td>0.8066</td><td>0.2578</td><td>0.7816</td><td>1.050</td><td>1.82</td><td>0.3067</td></tr><tr><td>Ours</td><td>34.93</td><td>0.8082</td><td>0.2941</td><td>0.7998</td><td>1.1356</td><td>1.87</td><td>0.3025</td></tr><tr><td>Ours Random</td><td>35.00</td><td>0.7854</td><td>0.3894</td><td>0.7992</td><td>1.0006</td><td>2.20</td><td>0.3005</td></tr><tr><td rowspan="6">SANA1.5</td><td>Orig</td><td>51.53</td><td>0.7060</td><td>0.2040</td><td>0.5700</td><td>0.7350</td><td>1.60</td><td>0.3105</td></tr><tr><td>Orig ( $\omega_{CFG} = 2$ )</td><td>53.40</td><td>0.7286</td><td>0.3150</td><td>0.6467</td><td>0.8328</td><td>1.81</td><td>0.3039</td></tr><tr><td>CADS</td><td>51.29</td><td>0.6992</td><td>0.3410</td><td>0.6104</td><td>0.5330</td><td>1.86</td><td>0.3064</td></tr><tr><td>SPARKE</td><td>42.72</td><td>0.6890</td><td>0.2490</td><td>0.6350</td><td>0.7365</td><td>1.67</td><td>0.3148</td></tr><tr><td>Ours</td><td>50.15</td><td>0.6943</td><td>0.4553</td><td>0.6726</td><td>0.5255</td><td>2.08</td><td>0.3076</td></tr><tr><td>Ours Random</td><td>47.20</td><td>0.6742</td><td>0.3784</td><td>0.6906</td><td>0.7207</td><td>1.94</td><td>0.3084</td></tr></table>

# 4. Experiments

We validate the effectiveness of DAVE on representative models, including Stable Diffusion 3.5 (Esser et al., 2024), FLUX.1-dev, and SANA1.5 (Xie et al., 2025), comparing our results against state-of-the-art diversity-enhancing samplers such as CADS (Sadat et al., 2023), Particle Guidance (PG) (Corso et al., 2023), SPARKE (Jalali et al., 2026), SPELL (Kirchhof et al., 2024), DiverseFlow (Morshed & Boddeti, 2025), and Oscar (Wu et al., 2025). Using prompts sampled from ImageNet (Deng et al., 2009) and MS-COCO (Lin et al., 2014), we quantify performance across three primary dimensions: visual quality (FID, Precision, Density), generation diversity (Recall, Coverage, Vendi Score), and text-image alignment (CLIP score) (Heusel et al., 2017; Kynka¨anniemi et al. ¨ , 2019; Naeem et al., 2020;

![](images/9f80b0f0e0be04c428ae876d9fec5547440429394dccc5d67f2b2633f448a1df.jpg)

<details>
<summary>text_image</summary>

A blue shelving unit has a vase and metal cups on it
A close up of a piece of cake on a plate
Original
PG
CADS
DAVE (Ours)
</details>

Figure 4. Qualitative comparison of in-batch diversity methods on Stable Diffusion 3. DAVE produces diverse samples with varied layouts and object appearances while preserving prompt consistency and visual fidelity.

Friedman & Dieng, 2022). Detailed implementation settings are presented in Appendix A.

# 4.1. Main Results

The results in Table 1 highlight the effectiveness of our approach. We quantitatively compare DAVE against (i) the original baseline, (ii) a low-CFG baseline $\begin{array} { r } { ( \omega _ { \mathrm { { C F G } } } = 2 , } \end{array}$ , a standard heuristic for trading fidelity for diversity), and (iii) prior diversity-enhancement methods CADS and SPARKE (selected for their independence from in-batch settings). For evaluation, we generate 10 samples per prompt for 1,000 ImageNet class labels and 500 MS-COCO prompts. To demonstrate DAVE’s robustness to block selection, we report results under both fixed-block and random-block settings. The fixed-block setting applies the intervention to a single predetermined layer, whereas the random-block setting dynamically samples from the identified pool L. Although the random setting occasionally yields higher diversity, we adopt the fixed-block approach as our default in subsequent experiments to ensure strict reproducibility.

Our results demonstrate that DAVE consistently improves key diversity metrics across all three foundation models, outperforming the original sampler while remaining highly competitive with strong baselines. Crucially, these gains hold under both fixed and randomized block settings, confirming that our method is robust to the exact choice of intervention layer. Notably, while CADS achieves high diversity scores on ImageNet with Flux.1-dev and SANA1.5, it exhibits a severe degradation in precision and CLIP alignment. This suggests that its conditioning-perturbation strategy may be brittle when applied to certain flow-based architectures. In contrast, DAVE delivers substantial diversity improvements while reliably preserving text alignment and image quality, ultimately yielding a favorable diversity–fidelity trade-off (See Appendix C, Figure 11).

Table 2. Quantitative results for in-batch diverse generation on Stable Diffusion 3.5. 

<table><tr><td>Method</td><td>FID</td><td>Prec</td><td>Rec</td><td>Vendi</td><td>CLIP</td></tr><tr><td>Orig</td><td>36.37</td><td>0.821</td><td>0.255</td><td>2.294</td><td>0.313</td></tr><tr><td>PG</td><td>45.69</td><td>0.833</td><td>0.204</td><td>1.946</td><td>0.314</td></tr><tr><td>SPELL</td><td>37.13</td><td>0.816</td><td>0.235</td><td>1.699</td><td>0.318</td></tr><tr><td>DiverseFlow</td><td>39.78</td><td>0.818</td><td>0.209</td><td>1.557</td><td>0.312</td></tr><tr><td>Oscar</td><td>35.87</td><td>0.827</td><td>0.245</td><td>1.698</td><td>0.312</td></tr><tr><td>Ours</td><td>29.75</td><td>0.780</td><td>0.441</td><td>2.370</td><td>0.308</td></tr></table>

![](images/b2f694bff61cf68c5e90092a9b7df18c316b50a02ff4008af044dbf0d1444b08.jpg)

<details>
<summary>bar</summary>

|        | CLIP (↓) | DINO (↓) | LPIPS (↑) |
| ------ | -------- | -------- | --------- |
| Orig   | 0.89     | 0.75     | 0.60      |
| CADS   | 0.87     | 0.70     | 0.65      |
| PG     | 0.88     | 0.74     | 0.61      |
| Ours   | 0.82     | 0.60     | 0.67      |
</details>

Figure 5. In-batch similarity comparison (batch size = 4). “Orig” denotes results from the vanilla Stable Diffusion 3.

We further evaluate the effectiveness of our method in the explicit in-batch generation setting. For this evaluation (Table 2), we compare DAVE against various recent baselines specifically designed for within-batch diversity: SPELL, DiverseFlow, OSCAR, and Particle Guidance (PG). Evaluated on Stable Diffusion 3 and 3.5 (batch size = 4) using MS-COCO prompts, DAVE consistently outperforms these baselines across all diversity metrics while incurring only marginal drops in image quality. Furthermore, as analyzed in Figure 5, DAVE achieves substantially lower featurelevel similarity (CLIP/DINO) and higher perceptual distance (LPIPS), confirming its robust capability to maximize intra-batch diversity. Finally, qualitative results (Figure 4) corroborate these findings: while CADS offers competitive visual variation but frequently compromises prompt adherence, DAVE consistently generates diverse, high-fidelity images that strictly respect the text condition.

![](images/20db0f2b6052cd2c30a89addc7d74d05f09be06ecf47422c2250a096bcfb8b6d.jpg)

<details>
<summary>line</summary>

| Timestep | LPIPS (Ours) | Aesthetic Score (Original) |
| -------- | ------------ | -------------------------- |
| 0        | 0.6          | 0.4                        |
| 9        | 0.1          | 0.8                        |
| 18       | 0.05         | 0.85                       |
| 27       | 0.0          | 0.85                       |
</details>

![](images/c04fd795a0893f20b9ccc89d52d462ab8a28f13356995f457e0876aa00e506c9.jpg)

<details>
<summary>bar</summary>

| α | LPIPS |
|---|---|
| 0 | 0.78 |
| 0.25 | 0.74 |
| 0.5 | 0.70 |
| 0.8 | 0.42 |
| 1.5 | 0.12 |
| 2 | 0.16 |

| Metric | Value |
| :--- | :--- |
| Precision | 0.80 |
| Recall | 0.45 |
</details>

Figure 6. Ablations on hyperparameters. (Left) Effect of the temporal cutoff τ on the magnitude of image change and quality. (Middle) Image-change magnitude as a function of the attenuation strength α. (Right) Robustness to block selection within the block pool L.

# 4.2. Ablation Studies

In this section, we conduct a series of ablation studies to validate our parameter selections and provide deeper insights into the dynamics of DC attenuation.

Temporal Cutoff τ . We evaluate the sensitivity of DAVE to the temporal cutoff τ , which restricts the intervention to the initial ⌊τ · T ⌋ steps (where T denotes the total number of timesteps). Consistent with our hypothesis, the results indicate that trajectory lock-in is indeed concentrated within the earliest stages of the generative process (Figure 6-Left). To further investigate this, we conducted a timestep-wise analysis by partitioning the denoising trajectory into early, middle, and late phases (5 steps each), as summarized in Table 3. Our findings show that cross-seed DC similarity exhibits a clear decreasing trend toward later timesteps. Additionally, the DC energy ratio is highest in the early regime, where our intervention triggers the most significant structural divergence. While accumulating interventions over more timesteps increases structural variation, it eventually introduces a trade-off with image quality. Notably, at the recommended threshold of $\tau = 0 . 1 5$ (approximately the first 4 steps in Stable Diffusion 3), the model maintains its structural integrity and even shows a slight improvement in aesthetic scores compared to the unperturbed baseline. This suggests that early-stage DC attenuation successfully promotes diversity without compromising the subsequent refinement of local details.

Attenuation Strength α. The parameter α regulates the intensity of structural manipulation. As shown in Figure $^ { 6 , }$ lower α values yield higher LPIPS scores, indicating greater structural deviation from the baseline generation. While more aggressive attenuation enhances diversity, settings below $\alpha \approx 0 . 2$ can compromise structural integrity. Interestingly, we observe an asymmetric response: amplifying the DC component (α > 1) results in negligible structural changes. This validates our interpretation of the DC mode as a structural anchor—its suppression releases the model from trajectory lock-in, whereas further amplification imposes no additional constraints on the already established generative pathways. Further sensitivity analyses are detailed in Appendix B.

Table 3. Step-wise analysis of DC characteristics and perceptual impact on SD3. CS-Sim, ER, and LPIPS denote Cross-seed DC similarity, DC energy ratio, and LPIPS distance, respectively. 

<table><tr><td>Steps</td><td>CS-Sim</td><td>ER</td><td>LPIPS</td></tr><tr><td>Early (0–4)</td><td>0.975</td><td>0.512</td><td>0.717</td></tr><tr><td>Mid (12–16)</td><td>0.893</td><td>0.304</td><td>0.113</td></tr><tr><td>Late (24–27)</td><td>0.756</td><td>0.298</td><td>0.095</td></tr></table>

Target Block Pool L. We identify the target pool L by isolating Transformer blocks that exhibit a high degree of cross-seed DC invariance. This phenomenon indicates a deterministic bottleneck, where internal DC components remain nearly identical despite differing initial noise realizations. To objectively quantify this alignment, we apply a ttest with Benjamini-Hochberg FDR correction (Benjamini & Hochberg, 1995) $( p < 0 . 0 5 )$ to blocks maintaining a crossseed cosine similarity $\mathbf { o f } \geq 0 . 9 9$ . In Stable Diffusion 3, this criterion yields a specific subset: $\mathcal { L } = \{ 0 , 2 , 4 , 5 , \hdots , 1 7 \}$ . Empirically, applying DAVE to any candidate block within L yields a consistent boost in Recall with only a marginal reduction in Precision (Figure 6-Right). These findings confirm that early-stage DC alignment is a critical prerequisite for the intervention to successfully break trajectory lock-in. By explicitly targeting this representational property, DAVE establishes a stable framework for diversity enhancement that is highly robust to block selection.

Table 4. Cross-dataset robustness of the L selection strategy for diversity enhancement. 

<table><tr><td>Method</td><td>Vendi (↑)</td><td>CLIP (↑)</td></tr><tr><td>Original</td><td>1.4541</td><td>0.3088</td></tr><tr><td>DAVE</td><td>2.0275</td><td>0.3061</td></tr><tr><td>Cross-dataset</td><td>1.8840</td><td>0.3061</td></tr></table>

We further examined the robustness of the target block pool L with respect to the calibration prompt set. To test this, we compared pools independently constructed from 20 randomly sampled MS-COCO captions and 20 ImageNet labels. In Stable Diffusion 3.5, these independently derived pools exhibited near-complete overlap, achieving an average pairwise Jaccard similarity of 0.98. We also conducted a cross-dataset evaluation by applying the pool L identified from one source to the other; the diversity gains remained robust, showing only a marginal performance gap compared to the in-dataset setting (Table 4). Together, these results demonstrate the intrinsic robustness of our L selection strategy. Consequently, this unified rule—defining the target pool via early-stage cross-seed DC alignment—is inherently dataset-agnostic, enabling seamless adaptation across diverse domains with a single, one-time calibration rather than exhaustive per-dataset tuning.

# 5. Discussion

# 5.1. Trajectory Analysis

We empirically verify that DAVE mitigates trajectory lockin, allowing generative trajectories to evolve along more diverse paths. To evaluate sample-to-sample separation, we compare the pairwise cosine distances along the generative trajectory between baseline SD3 samples and those generated with DAVE. As shown in Figure 7 (Right), DAVE leads to a progressive increase in the average inter-sample distance as the generation proceeds. This indicates that our early-stage intervention effectively expands the accessible state space, facilitating richer structural branching in later steps.

![](images/7c72fee2518ad70e91e48a9ea9235c35548f49958a14fb34c8ff07107f8e0d3a.jpg)

<details>
<summary>line</summary>

| Time Step | Original | Block 12 | Block 14 | Block 0 |
| --------- | -------- | -------- | -------- | ------- |
| 0         | 1.0      | 1.0      | 1.0      | 1.0     |
| 6         | 1.0      | 1.0      | 1.0      | 1.0     |
| 12        | 1.0      | 1.0      | 1.0      | 1.0     |
| 18        | 0.9      | 0.95     | 0.95     | 0.95    |
| 24        | 0.5      | 0.7      | 0.6      | 0.4     |
</details>

Figure 7. (Left) Latent trajectory visualization with generated outputs. (Right) Pairwise latent similarities across steps.

Furthermore, DAVE provides structural flexibility through selective intervention across Transformer blocks $\ell \in { \mathcal { L } } .$ , enabling entirely diverse outcomes from identical initial noise. As qualitatively confirmed by the visualized trajectories and resulting images in Figure 7 (Left), DAVE effectively prevents premature convergence to a single deterministic path, fostering a much broader and more expansive trajectory evolution.

Table 5. Quantitative results with DAVE in SD3.5-Large-Turbo. 

<table><tr><td>Method</td><td>Precision</td><td>Recall</td><td>Vendi</td><td>CLIP</td></tr><tr><td>Orig</td><td>0.816</td><td>0.384</td><td>1.383</td><td>0.205</td></tr><tr><td>CADS</td><td>0.785</td><td>0.423</td><td>1.565</td><td>0.206</td></tr><tr><td>PG</td><td>0.788</td><td>0.418</td><td>1.422</td><td>0.211</td></tr><tr><td>Ours</td><td>0.798</td><td>0.431</td><td>1.643</td><td>0.205</td></tr></table>

# 5.2. DAVE on Distilled Models

A notable advantage of DAVE is its inherent compatibility with distilled models. Because the method intervenes directly at the level of internal representations rather than modifying the scheduler or sampling rules, it is straightforward to apply even when the sampling procedure is heavily streamlined by distillation. We validate this capability on the distilled model SD3.5-Large-Turbo (Esser et al., 2024), where DAVE consistently improves image diversity, as demonstrated by both quantitative metrics (Table 5) and qualitative visualizations (Figure 8).

![](images/8a36c9acab6cfe4c31d682e2f809ca8d00d402ba3d3c8ce14d3b9bcb8aa3d134.jpg)

<details>
<summary>text_image</summary>

A park bench
that has a teddy bear on it
Original
A man standing on top of
a beach holding a surfboard
DAVE
</details>

Figure 8. Qualitative results with DAVE in SD3.5-Large-Turbo.

# 5.3. Block-wise Analysis

In further analysis of DAVE, we examine block-wise manipulation and find a consistent tendency: some blocks repeatedly induce diversity along specific attribute directions. We quantify these shifts by scoring changes relative to the original images across three attributes—Color, Size, Texture—using Gemini 3 Flash. On Stable Diffusion 3.5, structural divergence (DAVE’s primary objective) is reliably achieved across blocks, yet attribute changes are blockdependent: Blocks 1–3 predominantly vary color, Block 0 often alters subject scale/size, and Block 14 tends to produce coarser texture. Additional model cases are deferred to Appendix F.1.

![](images/81c6dc0a3ef8ea325f05a6386fed3abfb127329ac8ed1cfdbb5d39d0a585eebf.jpg)

<details>
<summary>text_image</summary>

A sink next to a large white door
A green unicorn in a snowy forest
1 2
Color
13
Texture
14
Transformer Block
Size
0
0.8
0.6
0.4
0.2
0.0
0.8
0.6
0.4
0.2
0.0
0.8
0.6
0.4
0.2
0.0
0.8
0.6
0.4
0.2
0.0
titi monkey
coffee mug
lesser panda
sweatshirt
</details>

Figure 9. Block-wise change direction analysis for DAVE in Stable Diffusion 3. (Left) VLM-based attribute scores quantifying the change direction for each block. (Right) Representative output examples showing the block-specific changes.

These findings suggest that DC attenuation may interact with internal blocks in a block-dependent manner, offering a mechanistic perspective on how diversity emerges from localized representation-level interventions. Although inducing attribute-specific diversity is not a primary objective of our method, the observed block signatures indicate a potential route to controllable diversity when paired with deeper architectural insight.

Table 6. Comparison between Original baseline and DAVE across different CFG scales on SD3. (default ωCF G = 7) 

<table><tr><td>CFG</td><td>Method</td><td>Prec</td><td>Rec</td><td>CLIP</td><td>Vendi</td></tr><tr><td rowspan="2">3</td><td>Orig</td><td>0.860</td><td>0.052</td><td>0.313</td><td>2.540</td></tr><tr><td>Ours</td><td>0.752</td><td>0.165</td><td>0.310</td><td>2.307</td></tr><tr><td rowspan="2">5</td><td>Orig</td><td>0.883</td><td>0.016</td><td>0.314</td><td>2.975</td></tr><tr><td>Ours</td><td>0.795</td><td>0.133</td><td>0.312</td><td>2.995</td></tr><tr><td rowspan="2">7</td><td>Orig</td><td>0.871</td><td>0.015</td><td>0.316</td><td>3.385</td></tr><tr><td>Ours</td><td>0.735</td><td>0.132</td><td>0.313</td><td>3.828</td></tr><tr><td rowspan="2">10</td><td>Orig</td><td>0.854</td><td>0.035</td><td>0.318</td><td>4.189</td></tr><tr><td>Ours</td><td>0.674</td><td>0.162</td><td>0.307</td><td>5.370</td></tr></table>

# 5.4. Compatibility with Classifier-Free Guidance

DAVE is compatible with CFG and can be naturally combined with it to further enhance performance. Across all CFG scales, DAVE substantially improves Recall, and at moderate-to-high CFG scales it also improves Vendi, while keeping CLIP nearly unchanged. This is practically important because higher CFG typically improves fidelity at the cost of diversity, whereas DAVE helps recover diversity under such settings without harming text alignment. Note that our main experiments already follow the default CFG setting of each base model (Table 8), showing that DAVE works under standard practical configurations.

# 5.5. DAVE with Highly Constrained Prompts

We evaluated DAVE on PartiPrompts (Yu et al., 2022), a benchmark categorized by varying complexity levels. Although complex prompts naturally allow less variation than simpler ones, DAVE consistently improves Vendi-score over the original model across all prompt complexity levels. Notably, CLIP-score changes remain minimal (absolute gap $\leq 0 . 0 1 )$ , demonstrating that DAVE enhances meaningful diversity without compromising prompt alignment, even under high structural constraints.

Table 7. Evaluation across prompt complexities. Bold indicates the best performance in each metric. 

<table><tr><td>Metric</td><td>Method</td><td>Basic</td><td>Simple</td><td>Fine</td><td>Comp.</td></tr><tr><td rowspan="2">Vendi</td><td>Original</td><td>2.051</td><td>1.575</td><td>1.524</td><td>1.511</td></tr><tr><td>DAVE</td><td>2.466</td><td>2.023</td><td>2.009</td><td>2.006</td></tr><tr><td rowspan="2">CLIP</td><td>Original</td><td>0.288</td><td>0.344</td><td>0.331</td><td>0.321</td></tr><tr><td>DAVE</td><td>0.298</td><td>0.337</td><td>0.329</td><td>0.311</td></tr></table>

# 5.6. Computational Efficiency of DAVE

DAVE uses minimal computational overhead while significantly improving generation diversity. Unlike prior approaches that rely on additional optimization, feature-bank maintenance, or explicit inter-sample interactions during sampling, DAVE performs lightweight representation-level modulation directly on intermediate representations during the early generation stage. Detailed complexity analyses are provided in Appendix G.

# 6. Conclusion

In this paper, we addressed the critical issue of limited generation diversity in flow-based text-to-image models, identifying the rapid homogenization of the DC component as a primary bottleneck. To mitigate this, we introduced DC Attenuation for diVersity Enhancement (DAVE), a streamlined representation-level intervention. Our extensive experiments demonstrate that DAVE effectively broadens the generative range, yielding diverse images without compromising prompt alignment, thereby maintaining a highly favorable diversity–fidelity trade-off. As a training-free method with negligible overhead, DAVE offers a practical and scalable framework for unlocking generative models from premature structural lock-in.

# Acknowledgements

This work was supported by the Institute for Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (RS-2019-II190075, Artificial Intelligence Graduate School Support Program (KAIST); RS-2024-00457882, AI Research Hub Project; RS-2022-II220984, Development of Artificial Intelligence Technology for Personalized Plug-and-Play Explanation and Verification of Explanation) and by the InnoCORE program of the Ministry of Science and ICT (N10250156).

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Albuquerque, I., Ktena, I., Wiles, O., Kajic, I., Rannen-Triki, ´ A., Vasconcelos, C., and Nematzadeh, A. Benchmarking diversity in image generation via attribute-conditional human evaluation. arXiv preprint arXiv:2511.10547, 2025.   
Astolfi, P., Careil, M., Hall, M., Manas, O., Muck-˜ ley, M., Verbeek, J., Soriano, A. R., and Drozdzal, M. Consistency-diversity-realism pareto fronts of conditional image generative models. arXiv preprint arXiv:2406.10429, 2024.   
Benjamini, Y. and Hochberg, Y. Controlling the false discovery rate: a practical and powerful approach to multiple testing. Journal of the Royal Statistical Society: series B (Methodological), 57(1):289–300, 1995.   
Bishop, C. M. Pattern recognition and machine learning. Springer, 2006.   
Cao, M., Wang, X., Qi, Z., Shan, Y., Qie, X., and Zheng, Y. Masactrl: Tuning-free mutual self-attention control for consistent image synthesis and editing. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 22560–22570, 2023.   
Chen, L., Song, Y., Guo, J., Sun, L., Childs, P., and Yin, Y. How generative ai supports human in conceptual design. Design Science, 11:e9, 2025.   
Choi, J., Kim, S., Jeong, Y., Gwon, Y., and Yoon, S. Ilvr: Conditioning method for denoising diffusion probabilistic models. arXiv preprint arXiv:2108.02938, 2021.

Corso, G., Xu, Y., De Bortoli, V., Barzilay, R., and Jaakkola, T. Particle guidance: non-iid diverse sampling with diffusion models. arXiv preprint arXiv:2310.13102, 2023.

Croitoru, F.-A., Hondru, V., Ionescu, R. T., and Shah, M. Diffusion models in vision: A survey. IEEE transactions on pattern analysis and machine intelligence, 45(9): 10850–10869, 2023.

Dalva, Y., Venkatesh, K., and Yanardag, P. Fluxspace: Disentangled semantic editing in rectified flow transformers. URL https://arxiv. org/abs/2412.09611, 2024.

Deng, J., Dong, W., Socher, R., Li, L.-J., Li, K., and Fei-Fei, L. Imagenet: A large-scale hierarchical image database. In 2009 IEEE conference on computer vision and pattern recognition, pp. 248–255. Ieee, 2009.

Esser, P., Kulal, S., Blattmann, A., Entezari, R., Muller, J., ¨ Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al. Scaling rectified flow transformers for high-resolution image synthesis. In Forty-first international conference on machine learning, 2024.

Friedman, D. and Dieng, A. B. The vendi score: A diversity evaluation metric for machine learning. arXiv preprint arXiv:2210.02410, 2022.

Han, J., Kwon, D., Lee, G., Kim, J., and Choi, J. Enhancing creative generation on stable diffusion-based models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 28609–28618, 2025.

Harrington, A., Koepke, A., Karthik, S., Darrell, T., and Efros, A. A. It’s never too late: Noise optimization for collapse recovery in trained diffusion models. arXiv preprint arXiv:2601.00090, 2025.

Hertz, A., Mokady, R., Tenenbaum, J., Aberman, K., Pritch, Y., and Cohen-Or, D. Prompt-to-prompt image editing with cross attention control. arXiv preprint arXiv:2208.01626, 2022.

Hessel, J., Holtzman, A., Forbes, M., Le Bras, R., and Choi, Y. Clipscore: A reference-free evaluation metric for image captioning. In Proceedings of the 2021 conference on empirical methods in natural language processing, pp. 7514–7528, 2021.

Heusel, M., Ramsauer, H., Unterthiner, T., Nessler, B., and Hochreiter, S. Gans trained by a two time-scale update rule converge to a local nash equilibrium. Advances in neural information processing systems, 30, 2017.

Jalali, M., Lei, H., Gohari, A., and Farnia, F. Sparke: Scalable prompt-aware diversity and novelty guidance in diffusion models via rke score. Advances in Neural Information Processing Systems, 38:119943–119980, 2026.   
Kim, J., Park, J., Song, Y., Kwak, N., and Rhee, W. Reflex: Text-guided editing of real images in rectified flow via mid-step feature extraction and attention adaptation. arXiv preprint arXiv:2507.01496, 2025.   
Kirchhof, M., Thornton, J., Bethune, L., Ablin, P., Ndi-´ aye, E., and Cuturi, M. Shielded diffusion: Generating novel and diverse images using sparse repellency. arXiv preprint arXiv:2410.06025, 2024.   
Kynka¨anniemi, T., Karras, T., Laine, S., Lehtinen, J., and ¨ Aila, T. Improved precision and recall metric for assessing generative models. Advances in neural information processing systems, 32, 2019.   
Li, B., Yang, M., Tan, Z., Zhang, J., and Li, H. Unraveling mmdit blocks: Training-free analysis and enhancement of text-conditioned diffusion. arXiv preprint arXiv:2601.02211, 2026.   
Lin, T.-Y., Maire, M., Belongie, S., Hays, J., Perona, P., Ramanan, D., Dollar, P., and Zitnick, C. L. Microsoft coco: ´ Common objects in context. In European conference on computer vision, pp. 740–755. Springer, 2014.   
Lipman, Y., Chen, R. T., Ben-Hamu, H., Nickel, M., and Le, M. Flow matching for generative modeling. arXiv preprint arXiv:2210.02747, 2022.   
Liu, H., Liu, J., Li, Y., Bai, L., Ji, Y., Guo, Y., Wan, S., and Wen, H. From navigation to refinement: Revealing the two-stage nature of flow-based diffusion models through oracle velocity. arXiv preprint arXiv:2512.02826, 2025.   
Liu, X., Gong, C., and Liu, Q. Flow straight and fast: Learning to generate images with rectified flow. arXiv preprint arXiv:2209.03003, 2022.   
Miyato, T., Kataoka, T., Koyama, M., and Yoshida, Y. Spectral normalization for generative adversarial networks. In International Conference on Learning Representations, 2018.   
Morshed, M. M. and Boddeti, V. Diverseflow: Sampleefficient diverse mode coverage in flows. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 23303–23312, 2025.   
Mukhopadhyay, S., Gwilliam, M., Agarwal, V., Padmanabhan, N., Swaminathan, A., Hegde, S., Zhou, T., and Shrivastava, A. Diffusion models beat gans on image classification. arXiv preprint arXiv:2307.08702, 2023.

Naeem, M. F., Oh, S. J., Uh, Y., Choi, Y., and Yoo, J. Reliable fidelity and diversity metrics for generative models. In International conference on machine learning, pp. 7176–7185. PMLR, 2020.   
Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., and Sutskever, I. Learning transferable visual models from natural language supervision. arXiv preprint arXiv:2103.00020, 2021.   
Rahaman, N., Baratin, A., Arpit, D., Draxler, F., Lin, M., Hamprecht, F., Bengio, Y., and Courville, A. On the spectral bias of neural networks. In International conference on machine learning, pp. 5301–5310. PMLR, 2019.   
Sadat, S., Buhmann, J., Bradley, D., Hilliges, O., and Weber, R. M. Cads: Unleashing the diversity of diffusion models through condition-annealed sampling. arXiv preprint arXiv:2310.17347, 2023.   
Shin, J., Hwang, A., Kim, Y., Kim, D., and Park, J. Exploring multimodal diffusion transformers for enhanced prompt-based image editing. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 19492–19502, 2025.   
Si, C., Huang, Z., Jiang, Y., and Liu, Z. Freeu: Free lunch in diffusion u-net. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 4733–4743, 2024.   
Um, S. and Ye, J. C. Minority-focused text-to-image generation via prompt optimization. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 20926–20936, 2025.   
Voynov, A., Chu, Q., Cohen-Or, D., and Aberman, K. p+: Extended textual conditioning in text-to-image generation. arXiv preprint arXiv:2303.09522, 2023.   
Wang, B. and Pehlevan, C. An analytical theory of spectral bias in the learning dynamics of diffusion models. Advances in Neural Information Processing Systems, 38: 95865–95963, 2026.   
Wu, J., Wan, Z., Yu, X., Yang, Y., An, B., and Tsang, I. Oscar: Orthogonal stochastic control for alignmentrespecting diversity in flow matching. arXiv preprint arXiv:2510.09060, 2025.   
Xie, E., Chen, J., Zhao, Y., Yu, J., Zhu, L., Wu, C., Lin, Y., Zhang, Z., Li, M., Chen, J., et al. Sana 1.5: Efficient scaling of training-time and inference-time compute in linear diffusion transformer. arXiv preprint arXiv:2501.18427, 2025.

Yang, Y., Wang, Y., Wang, C., Zhang, Y., Chen, Z., and He, S. Splitflux: Learning to decouple content and style from a single image. arXiv preprint arXiv:2511.15258, 2025. Yu, J., Xu, Y., Koh, J. Y., Luong, T., Baid, G., Wang, Z., Vasudevan, V., Ku, A., Yang, Y., Ayan, B. K., et al. Scaling autoregressive models for content-rich text-to-image generation. arXiv preprint arXiv:2206.10789, 2(3):5, 2022.

# A. Implementation Details

Models and Datasets. We conduct experiments on four representative flow-based text-to-image generation models: Stable Diffusion 3 (SD3), Stable Diffusion 3.5 (SD3.5), FLUX.1-dev, and SANA1.5. All methods are evaluated on ImageNet and MS-COCO benchmarks. For ImageNet, we generate 10 samples for each of the 1,000 class-label prompts, resulting in 10K images per method. For MS-COCO, we randomly sample 500 captions from the val2017 split and generate 10 images per prompt, yielding a total of 5K images.

Evaluation Metrics. We evaluate performance along three dimensions: visual quality, diversity, and text–image alignment. Visual quality and diversity is measured using FID, Precision, Recall, Coverage, and Density, computed in a shared Inception feature space. For ImageNet, $N _ { \mathrm { f a k e } } = 1 0 \mathrm { K }$ generated images are compared against $N _ { \mathrm { r e a l } } = 1 0 \mathrm { K }$ real images from the corresponding validation split. For MS-COCO, we similarly use $N _ { \mathrm { f a k e } } = 5 \mathrm { K }$ generated images and $N _ { \mathrm { r e a l } } = 5 \mathrm { K }$ real images.

Diversity is assessed using the Vendi Score (Friedman & Dieng, 2022), computed on the cosine-similarity Gram matrix of L2-normalized CLIP ViT-B/32 image embeddings (Radford et al., 2021). Text–image alignment is evaluated using the CLIP Score (Hessel et al., 2021), computed as the cosine similarity between CLIP ViT-B/32 image and text embeddings. Both metrics are computed independently for each prompt and reported as averages across all prompts.

# A.1. Experimental Settings

A.1.1. BLOCK SELECTION   
![](images/256195b433f5f60f772d58947e85abc15f92c45937748a38cddbb62818e52642.jpg)  
Figure 10. Mean cosine similarity of DC components of hidden representations $h _ { t } ^ { ( \ell ) }$ across 100 random noise seeds, measured at each Transformer block. Higher similarity indicates stronger cross-seed convergence.

To identify suitable intervention blocks for DAVE, we measure the mean cosine similarity of DC components across 100 different noise seeds at each Transformer block. As shown in Figure 10, early and intermediate blocks exhibit consistently high cosine similarity, indicating strong cross-seed alignment and pronounced representational lock-in.

Block Selection Rationale. Based on this analysis, we select intervention blocks from regions where the DC component becomes nearly seed-invariant. For SD3, blocks in the set $\mathcal { L } \in \{ 0 , 2 , 4 , 5 , \hdots , 1 7 \}$ exhibit high cosine similarity and are used as candidate blocks in the random-block setting. For SD3.5, a similar trend is observed for $\mathcal { L } \in \{ 0 , 5 , \dots , 1 5 \}$ . In the fixed-block configuration for both models, we select a representative block $( \mathcal { L } = 5 )$ within this high-alignment regime.

For FLUX.1-dev and SANA1.5, high DC cosine similarity persists over a broader range of blocks, with peak alignment occurring at later blocks compared to SD-based models. Accordingly, we select $\mathscr { L } = 3 0$ for FLUX.1-dev and $\mathcal { L } = 1 3$ for SANA1.5 in the fixed-block setting, and define model-specific candidate pools $( \mathcal { L } \in [ 1 9 , 4 0 ]$ ] for FLUX.1-dev and $\mathcal { L } \in [ 4 , 1 6 ]$ for SANA1.5) for the random-block configuration.

Statistical Criterion. To quantify cross-seed invariance of DC components, we compute pairwise cosine similarity across 100 noise seeds at each block. Candidate blocks are identified using a two-sided t-test with Benjamini–Hochberg FDR correction (Benjamini & Hochberg, 1995) $( p < 0 . 0 5 )$ , and we further require the mean cosine similarity to exceed 0.99. Although this criterion identifies a broad set of high-alignment blocks, we empirically observe that interventions applied to final-stage blocks have limited impact on generation diversity. These late blocks are therefore excluded from the candidate pools.

Table 8. DAVE parameter settings for quantitative evaluation. “ωCF G” denotes the default classifier-free guidance scale of each base model, used in all main experiments unless stated otherwise. 

<table><tr><td>Model</td><td>Setting</td><td> $\omega_{CFG}$ </td><td> $\tau$ </td><td> $\mathcal{L}$ </td><td> $\alpha$ </td></tr><tr><td rowspan="2">SD3</td><td>Fixed</td><td>7.0</td><td>0.15</td><td>5</td><td>0.5</td></tr><tr><td>Random</td><td>7.0</td><td>0.15</td><td>0,2,4–17</td><td>0.5</td></tr><tr><td rowspan="2">SD3.5</td><td>Fixed</td><td>7.0</td><td>0.15</td><td>5</td><td>0.5</td></tr><tr><td>Random</td><td>7.0</td><td>0.15</td><td>0,5–15</td><td>0.5</td></tr><tr><td rowspan="2">FLUX.1-dev</td><td>Fixed</td><td>3.5</td><td>0.2</td><td>30</td><td>0.2</td></tr><tr><td>Random</td><td>3.5</td><td>0.2</td><td>19–40</td><td>0.2</td></tr><tr><td rowspan="2">SANA1.5</td><td>Fixed</td><td>4.5</td><td>0.2</td><td>13</td><td>0.2</td></tr><tr><td>Random</td><td>4.5</td><td>0.2</td><td>4–16</td><td>0.2</td></tr></table>

# A.1.2. EVALUATION SETTINGS

Quantitative Evaluation Settings For quantitative evaluation, we consider two configurations: a fixed-block setting and a random-block setting. In the fixed-block setting, DC attenuation is applied to a single representative block selected from the high-alignment regime. In the random-block setting, intervention blocks are uniformly sampled from model-specific candidate pools identified by the block selection analysis. This configuration is designed to demonstrate that DAVE is not tied to a specific block, but remains effective across diverse blocks satisfying the proposed criteria. The complete parameter settings are summarized in Table 8.

In-batch Diversity Evaluation Settings For in-batch diversity evaluation, we use 200 randomly sampled captions from the MS-COCO val2017 split. For each prompt, we generate images using 5 fixed random seeds, and for each seed we produce 4 samples, resulting in a total of 4K generated images. To ensure fair comparison, the same set of random seeds is used for all methods. We compare the original sampler, CADS, Particle Guidance (PG), SPELL, DiverseFlow, OSCAR, and our method. In-batch diversity is measured using CLIP similarity, DINO feature similarity, and LPIPS, where lower similarity values indicate higher diversity within a batch.

# B. Sensitivity Analysis

Table 9. Sensitivity analysis of DAVE across hyperparameters α (intervention strength) and τ (intervention window). Metrics include Precision (Prec), Recall (Rec), Vendi score (Vendi), and CLIP score (CLIP). 

<table><tr><td colspan="5">Effect of Intervention Strength (α)</td><td colspan="5">Effect of Intervention Window (τ)</td></tr><tr><td>α</td><td>Prec</td><td>Rec</td><td>Vendi</td><td>CLIP</td><td>τ</td><td>Prec</td><td>Rec</td><td>Vendi</td><td>CLIP</td></tr><tr><td>0.3</td><td>0.664</td><td>0.561</td><td>2.42</td><td>0.302</td><td>0.00</td><td>0.821</td><td>0.254</td><td>2.29</td><td>0.313</td></tr><tr><td>0.4</td><td>0.676</td><td>0.554</td><td>2.62</td><td>0.302</td><td>0.10</td><td>0.698</td><td>0.456</td><td>2.53</td><td>0.307</td></tr><tr><td>0.5</td><td>0.692</td><td>0.492</td><td>2.55</td><td>0.306</td><td>0.15</td><td>0.692</td><td>0.492</td><td>2.55</td><td>0.306</td></tr><tr><td>0.6</td><td>0.723</td><td>0.461</td><td>2.41</td><td>0.308</td><td>0.20</td><td>0.622</td><td>0.562</td><td>2.61</td><td>0.303</td></tr><tr><td>0.7</td><td>0.783</td><td>0.407</td><td>2.36</td><td>0.312</td><td>0.25</td><td>0.588</td><td>0.583</td><td>2.63</td><td>0.299</td></tr></table>

DAVE operates in a stable regime with an explicit diversity–fidelity trade-off, rather than as a brittle single-point intervention. The added ablations show that nearby tested settings around our practical defaults (α=0.5, τ =0.15) do not trigger abrupt quality collapse across any evaluated metrics, but instead produce a gradual trade-off between Precision/CLIP and Recall/Vendi. We therefore view α and τ as coarse intervention knobs controlling the strength and duration of early DC attenuation, rather than fragile hyperparameters that require fine-grained tuning.

# C. Trade-off Analysis

Figure 11 presents a Pareto analysis between diversity and fidelity metrics across different diversity-enhancement methods on SD3 and SD3.5. We compare the trade-off trajectories of the original sampler, CADS, SPARKE, and DAVE by jointly visualizing diversity (Vendi score) against perceptual quality (Aesthetic score) and semantic alignment (CLIP score). Each trajectory is obtained by sweeping the diversity-control parameter of each method, where points correspond to different attenuation steps in DAVE, corruption strengths in CADS, and guidance frequencies in SPARKE.

As shown in the figure, existing methods often improve diversity at the cost of image fidelity or text–image alignment, leading to unfavorable trade-offs. In contrast, DAVE achieves substantially higher diversity while preserving competitive aesthetic quality and semantic consistency. Moreover, DAVE exhibits smooth and stable trajectory transitions across attenuation strengths, indicating controllable diversity enhancement without severe structural degradation.

![](images/60a73243e98137a9e68fa55540e6736b9ef433d37e223955f902f1cc95ef2fd3.jpg)

![](images/8bd9b65404e027eabc2af4c7a79db59c3943aa80dfcd3a0c74d212f63d3c1108.jpg)  
CADS

![](images/d51616ffc6e36ecd9f775c06b2b1c0ab04ec3095d91b23c1ac5e9f97d1d01d64.jpg)  
SPARKE

![](images/f006c3bf4cf36c13687aac586b215d27ca47977853fa0bbb2f9eaa5c9fd83f6a.jpg)  
Ours

SD3   
![](images/af2f6d09904c98b513a310353ade182ddca8cfbebf63fdf9b8f20c4604c1c280.jpg)

<details>
<summary>line</summary>

| Vendi (↑) | Aesthetic (↑) |
| --------- | ------------- |
| 1.8       | 32.0          |
| 1.9       | 31.8          |
| 2.0       | 31.5          |
| 2.1       | 32.0          |
| 2.2       | 31.5          |
| 2.3       | 31.0          |
| 2.4       | 30.0          |
| 2.5       | 29.5          |
</details>

SD3.5   
![](images/945a21bfcd3d2044d59dbd6031ef6c97b0ced68b11d5f77335936eec5cd4bea6.jpg)

<details>
<summary>line</summary>

| Vendii (↑) | Aesthetic (↑) |
| ---------- | ------------- |
| 2.1        | 30.0          |
| 2.2        | 29.5          |
| 2.3        | 29.0          |
| 2.4        | 28.5          |
| 2.5        | 27.0          |
| 2.6        | 26.0          |
| 2.7        | 24.0          |
</details>

SD3   
![](images/94135bdfe881aa9b477bad8be0991b8c2f6edf1361b5826757dc3fa2e9da23cf.jpg)

<details>
<summary>line</summary>

| Vendi (↑) | CLIP Score (↑) |
| --------- | -------------- |
| 1.8       | 0.302          |
| 2.0       | 0.301          |
| 2.4       | 0.292          |
</details>

SD3.5   
![](images/944f2db3d1bc446264318282240b38eae450facdc8bff32d0671d8b27f241b16.jpg)

<details>
<summary>line</summary>

| Vendi (↑) | CLIP Score (↑) |
| --------- | -------------- |
| 2.2       | 0.2995         |
| 2.3       | 0.2995         |
| 2.4       | 0.2995         |
| 2.5       | 0.2985         |
| 2.6       | 0.2960         |
| 2.7       | 0.2850         |
</details>

Figure 11. Pareto analysis between diversity and fidelity metrics. Compared to prior methods, DAVE achieves a more favorable trade-off frontier, improving diversity while preserving image quality and semantic alignment across SD3 and SD3.5.

# D. Pseudo Code

The pseudocode for DAVE is provided in Algorithm 1. The official implementation is available at https://github. com/daheekwon/DAVE.

Algorithm 1 DAVE: DC component Attenuation for diVersity Enhancement 

<table><tr><td colspan="3">Input: Hidden states H ∈ R×C, Current timestep t, Block index l</td></tr><tr><td colspan="3">Input: Hyperparameters: Cutoff τ, Target block pool L, Strength α</td></tr><tr><td colspan="3">1: H ← TransformerBlockl(H)</td></tr><tr><td colspan="3">2: if (t &lt; τ) and (l ∈ L) then</td></tr><tr><td>3: μ ← 1/D ∑d=1D Hd,:</td><td colspan="2">▷ Compute spatial mean (DC component)</td></tr><tr><td colspan="3">4: for d = 1 to D do</td></tr><tr><td>5: Hd,: ← Hd,: + (α - 1) · μ</td><td colspan="2">▷Attenuate DC component</td></tr><tr><td colspan="3">6: end for</td></tr><tr><td colspan="3">7: end if</td></tr><tr><td colspan="3">Output: Enhanced hidden states H</td></tr></table>

# E. Theoretical Motivation for Early DC Lock-in

We provide a formal motivation for the intuition behind our method for a Transformer-based Flow Matching model. We divide our analysis into two parts: first, we characterize a mechanism by which the DC component of intermediate representations can become seed-invariant in the early high-noise sampling regime (t ≈ 0); second, we formalize why early contraction can limit later trajectory separation under locally Lipschitz dynamics.

# E.1. Problem Setup and Definitions

Definition E.1 (Spaces and Mapping). Let X be the image space and $\mathcal { H } = \mathbb { R } ^ { D \times C }$ be the latent space, where D is the number of tokens and C is the channel dimension. We define an abstract representation map induced by a fixed block of the pre-trained generator, $\mathcal { E } : \mathcal { X }  \mathcal { H }$ , mapping a sample image $X \in { \mathcal { X } }$ to its block-level hidden representation $h \in \mathcal H$ .

Here X is the state $x _ { t }$ evolved by the ODE in Eq 1, and E to a fixed block $\ell \in \mathcal { L } \left( \mathrm { E q } 3 \right)$ , so that $h = \mathcal { E } ( x _ { t } )$ is precisely the block-level representation $h _ { t } ^ { ( \bar { \ell } ) }$ analyzed in Figure 2 and Table 3. We suppress ℓ and write $h _ { t } : = \mathcal { E } ( x _ { t } )$ ; the separation bounds below thus concern these representations across seeds.

Definition E.2 (Distributions and Coupling). We consider the Flow Matching objective defined over two distributions:

• Data Distribution $( X _ { 1 } ) \colon X _ { 1 } \sim p _ { \mathrm { d a t a } } ( X | c )$ , conditioned on text c. The corresponding latent target is $h _ { 1 } = \mathcal { E } ( X _ { 1 } )$ .   
• Noise Distribution $( X _ { 0 } ) \colon X _ { 0 } \sim p _ { \mathrm { n o i s e } } ( X )$ . The corresponding latent noise is $h _ { 0 }$ .

We adopt the 1-Rectified Flow framework (Liu et al., 2022), which constructs a straight-line probability path between distributions. In the standard training setup (prior to any reflow steps), the coupling between the source and target is independent, meaning the joint distribution factorizes as:

$$
\pi (X _ {0}, X _ {1}) = p _ {\text { noise }} (X _ {0}) p _ {\text { data }} (X _ {1} | c) \tag {6}
$$

This independence implies that the initial noise sample $X _ { 0 }$ contains no information about the target data sample $X _ { 1 }$ $( X _ { 0 } \perp X _ { 1 } )$ .

Definition E.3 (Lipschitz Continuity of Target Estimation). Let $\Psi ( h ) = \mathbb { E } _ { X _ { 1 } } [ \mathcal { E } ( X _ { 1 } ) \mid h , c ]$ be the conditional expectation. We assume that Ψ is K-Lipschitz continuous with respect to the latent metric. That is, there exists a constant $K < \infty$ such that for any $h , h ^ { \prime } \in { \mathcal { H } } \colon$

$$
\left\| \Psi (h) - \Psi \left(h ^ {\prime}\right) \right\| \leq K \| h - h ^ {\prime} \| \tag {7}
$$

This is a standard assumption in generative modeling to ensure the well-posedness of the induced ODE flow (Liu et al., 2022) and is often enforced in deep neural networks via regularization techniques (Miyato et al., 2018).

Definition E.4 (Spectral Decomposition in Latent Space). For any latent state $h \in { \mathcal { H } } .$ , we decompose it into a DC component and a AC component:

$$
h = h _ {D C} + h _ {A C} \tag {8}
$$

where $\begin{array} { r } { h _ { D C } \triangleq \frac { 1 } { D } \mathbf { 1 } _ { D } \mathbf { 1 } _ { D } ^ { \top } h } \end{array}$ is the spatial mean, and $h _ { A C } \triangleq h - h _ { D C }$ is the spatial residual. Equivalently, $h _ { D C } = \mathbf { 1 } _ { D } \mu _ { t }$ where $\mu _ { t } \in \mathbb { R } ^ { 1 \times C }$ is the spatial-mean (DC) vector of Eq 4; thus $h _ { D C }$ is the broadcast of the main-text DC vector across all D tokens, and $h _ { A C } = h - \mathbf { 1 } _ { D } \mu _ { t }$ .

# E.2. Mechanism of Early Spectral Collapse

In this section, we analyze why the learned vector field collapses to the spatial mean at $t \approx 0$ .

Proposition E.5 (Dominance of Ensemble Mean in High-Noise Regime). For a sufficiently small time t ≈ 0 (high-noise regime), the optimal vector field v∗ is dominated by the drift towards the Conditional Ensemble Mean. The deviation caused by instance-specific details is strictly bounded by O(t).

Proof. Let the objective be the standard MSE loss. It is a fundamental property of the $L _ { 2 }$ risk that its global minimizer $v ^ { * }$ is the conditional expectation of the target vector field (Bishop, 2006; Lipman et al., 2022):

$$
v ^ {*} (h _ {t}, t; c) = \mathbb {E} _ {X _ {0}, X _ {1}} [ \mathcal {E} (X _ {1}) - \mathcal {E} (X _ {0}) \mid h _ {t}; c ]. \tag {9}
$$

We decompose the optimal field into its two conditional expectations,

$$
v ^ {*} (h _ {t}, t; c) = \Psi (h _ {t}) - \Phi (h _ {t}), \quad \Psi (h _ {t}) = \mathbb {E} [ \mathcal {E} (X _ {1}) \mid h _ {t}, c ], \quad \Phi (h _ {t}) = \mathbb {E} [ \mathcal {E} (X _ {0}) \mid h _ {t}, c ]. \tag {10}
$$

We analyze the target term Ψ below; the source term Φ is treated after the collapse argument, as it accounts for the residual seed-specific (AC) content.

In the early phase, the interpolated state is $h _ { t } = ( 1 - t ) h _ { 0 } + t h _ { 1 }$ . We compare the estimation at time t with the estimation at time 0 using the Lipschitz assumption:

$$
\| \Psi (h _ {t}) - \Psi (h _ {0}) \| \leq K \| h _ {t} - h _ {0} \| = K \| t (h _ {1} - h _ {0}) \|. \tag {11}
$$

At $t = 0$ , due to independent coupling $( h _ { 0 } \perp h _ { 1 } )$ , the input $h _ { 0 }$ provides no information about $X _ { 1 }$ , so $\Psi ( h _ { 0 } ) = \mathbb { E } [ \mathcal { E } ( X _ { 1 } )$ | $c \bigr ] \triangleq \mu _ { c } ^ { H }$ . Thus, we can write:

$$
\Psi (h _ {t}) = \mu_ {c} ^ {H} + R (t), \quad \text { where } \mathbb {E} [ \| R (t) \| ] \leq t \cdot K \mathbb {E} [ \| h _ {1} - h _ {0} \| ]. \tag {12}
$$

Thus, the expected magnitude of the residual is $\mathcal { O } ( t )$ . The residual $R ( t )$ captures the seed- or instance-dependent correction required for recovering sample-specific structure. Since the conditional ensemble mean $( \mu _ { c } ^ { H } )$ remains an $\mathcal { O } ( 1 )$ term as $t \to 0$ , the early target estimate is dominated by this common component. □

Proposition E.6 (DC Dominance of the Conditional Ensemble Mean under Under-specified Prompts). The Latent Ensemble Mean $\mu _ { c } ^ { H }$ is spectrally sparse, dominated by the DC component $( h _ { D C } ) _ { : }$ , as the spatial variations (AC components) cancel out across the data distribution.

Proof. We apply the expectation operator to the spectral decomposition of the encoded target $\mathcal { E } ( X _ { 1 } )$ :

$$
\mu_ {c} ^ {H} = \mathbb {E} _ {X _ {1}} [ \mathcal {E} (X _ {1}) _ {D C} \mid c ] + \mathbb {E} _ {X _ {1}} [ \mathcal {E} (X _ {1}) _ {A C} \mid c ]. \tag {13}
$$

1. Stability of DC: The DC component $\mathcal { E } ( X _ { 1 } ) _ { D C }$ encodes global semantic information strongly correlated with the text c, aligning coherently across samples. Thus, $\| \mathbb { E } [ \mathcal { E } ( X _ { 1 } ) _ { D C } \mid c ] \| \gg 0$ .

2. Cancellation of AC: The AC component $\mathcal { E } ( X _ { 1 } ) _ { A C }$ encodes high-frequency, spatially localized detail. We assume that, conditioned on c, the phase (spatial placement) of these details is not fully determined by the prompt but varies across the data distribution—e.g., a prompt fixes what objects appear and their coarse global statistics, but leaves their precise position, pose, and fine texture under-specified. Under this assumption, the AC components are not phase-aligned across samples, so they interfere destructively in expectation:

$$
\mathbb {E} [ \mathcal {E} (X _ {1}) _ {A C} \mid c ] \approx 0. \tag {14}
$$

Consequently, the learning target collapses to the spatial mean: $\mu _ { c } ^ { H } \approx \mathbb { E } [ \mathcal { E } ( X _ { 1 } ) _ { D C } \mid c ]$

Summary of Collapse. Together, Propositions E.5 and E.6 suggest that, in the early high-noise regime, the target term Ψ is strongly biased toward a common DC-dominated direction $\mu _ { c } ^ { \bar { H } }$ , while its seed-specific AC correction $R ( t )$ remains $\mathcal { O } ( t )$ . The source term Φ, by contrast, stays seed-dependent: at $t \approx 0$ we have $h _ { t } \approx h _ { 0 } ,$ , so Φ reduces to the sample’s own noise representation and retains its instance-specific (AC) content. Hence the early collapse is confined to the DC subspace of the target estimate, while AC variation is preserved—consistent with the empirically observed coexistence of high cross-seed DC similarity and low AC similarity in Figure 2. This provides a formal motivation for the observation that early trajectories become closely aligned in the DC subspace. The next result analyzes what happens once such early DC alignment has reduced the cross-seed DC separation to an ϵ-neighborhood.

# E.3. Bounded Recovery after Early Lock-in

Having motivated why early DC alignment can reduce seed-specific separation in the DC subspace, we now show that any later recovery of this separation under locally Lipschitz ODE dynamics is bounded by the residual that remains after the early phase. Since the DC component encodes global layout and coarse structure—the under-specified factors that govern perceptual diversity—while the AC component carries localized texture, we treat the cross-seed separation in the DC subspace as the quantity controlling sample diversity.

Let $\begin{array} { r } { P _ { D C } = \frac { 1 } { D } \mathbf { 1 } _ { D } \mathbf { 1 } _ { D } ^ { \top } } \end{array}$ denote the (linear, time-invariant) projection onto the DC subspace, so that $h _ { D C } = P _ { D C } h$ . We strengthen Definition E.3 to the DC subspace: the learned field is assumed locally Lipschitz on the DC subspace, i.e. there exists $L < \infty$ such that for all states on the compact trajectory domain,

$$
\left\| P _ {D C} \big (v _ {\theta} (x, t; c) - v _ {\theta} (y, t; c) \big) \right\| \leq L \| P _ {D C} (x - y) \|. \tag {15}
$$

Theorem E.7 (Bounded Diversity Recovery under ODE Dynamics). Consider the ODE flow $\dot { h } _ { t } = v _ { \theta } ( h _ { t } , t ; c )$ generated by the Transformer network, with vθ locally Lipschitz on the DC subspace with constant $L < \infty . ^ { 1 }$ If the early spectral collapse (Section E.2) constrains the cross-seed DC separation to an ϵ-neighborhood at an early time t∗, i.e. $\| P _ { D C } ( h _ { t ^ { * } } ^ { ( i ) } - h _ { t ^ { * } } ^ { ( j ) } ) \| \overset { - } { \leq } \epsilon$ , then the DC separation at the final time t = 1 is bounded by

$$
\left\| P _ {D C} (h _ {1} ^ {(i)} - h _ {1} ^ {(j)}) \right\| \leq \epsilon \cdot \exp \bigl (L (1 - t ^ {*}) \bigr). \tag {16}
$$

Proof. Let $h _ { t } ^ { ( i ) }$ and $h _ { t } ^ { ( j ) }$ be two trajectories starting from distinct noise samples whose DC separation is reduced to at most ϵ at an early time t∗ due to the mechanisms in Propositions E.5– E.6. Define the DC difference vector

$$
\delta_ {D C} (t) = P _ {D C} \big (h _ {t} ^ {(i)} - h _ {t} ^ {(j)} \big). \tag {17}
$$

For $t \geq t ^ { * }$ , the time derivative of the squared distance is

$$
\frac {d}{d t} \| \delta_ {D C} (t) \| ^ {2} = 2 \langle \delta_ {D C} (t), \dot {\delta} _ {D C} (t) \rangle . \tag {18}
$$

Since both trajectories evolve under the same vector field and text condition c, and $P _ { D C }$ is linear and time-invariant,

$$
\dot {\delta} _ {D C} (t) = P _ {D C} \left(v _ {\theta} \left(h _ {t} ^ {(i)}, t; c\right) - v _ {\theta} \left(h _ {t} ^ {(j)}, t; c\right)\right). \tag {19}
$$

Using the Cauchy–Schwarz inequality and the DC-subspace Lipschitz condition,

$$
\frac {d}{d t} \| \delta_ {D C} (t) \| ^ {2} \leq 2 \| \delta_ {D C} (t) \| \left\| P _ {D C} (v _ {\theta} (h _ {t} ^ {(i)}, t; c) - v _ {\theta} (h _ {t} ^ {(j)}, t; c)) \right\| \leq 2 L \| \delta_ {D C} (t) \| ^ {2}. \tag {20}
$$

Applying Gronwall’s inequality yields ¨

$$
\left\| \delta_ {D C} (1) \right\| ^ {2} \leq \left\| \delta_ {D C} \left(t ^ {*}\right) \right\| ^ {2} \exp \left(2 L \left(1 - t ^ {*}\right)\right). \tag {21}
$$

Taking the square root and using $\| \delta _ { D C } ( t ^ { * } ) \| \leq \epsilon ,$ , we obtain

$$
\left\| P _ {D C} (h _ {1} ^ {(i)} - h _ {1} ^ {(j)}) \right\| = \left\| \delta_ {D C} (1) \right\| \leq \epsilon \exp \bigl (L (1 - t ^ {*}) \bigr). \tag {22}
$$

This gives the desired bound.

Consequently, if the early collapse drives ϵ → 0, the right-hand side vanishes, so the final-time DC separation is upperbounded by a quantity that tends to zero. This formalizes the lock-in intuition: once early dynamics align the global-structure (DC) subspace across seeds, later refinement can only amplify the residual DC differences that remain, rather than reconstructing the suppressed structural variation. We emphasize that this is a capacity bound—it shows that early DC alignment caps the recoverable diversity, which motivates intervening on the DC component early; the resulting diversity gains of DAVE are then established empirically.

# F. Additional examples

# F.1. Block-wise analysis in Flux

Here, we provide a complementary case study on Flux.1-dev to examine whether the block-dependent behaviors observed in Stable Diffusion extend to a different architecture. Beyond confirming that DC attenuation produces consistent seed-level divergence in the intended regime, we observe that the resulting attribute profile is again block-dependent, but the specific block–attribute associations are model-specific. Flux.1-dev exhibits systematic preferences toward certain attribute shifts, but these patterns do not map one-to-one to the block-index trends in Stable Diffusion 3.5, consistent with architectural and training differences. Figure 12 visualizes these Flux-specific tendencies—most prominently in texture and color—supporting the view that DC attenuation interacts with internal blocks in a structured yet model-dependent manner and may serve as a basis for controllable diversity when paired with architectural insight.

![](images/fcf8d42975f50b57bb92ecdb6b118e750eaae0208bbedb51620c494737e3e618.jpg)

<details>
<summary>line</summary>

| Item           | Color Score | Transformer Block |
| -------------- | ----------- | ----------------- |
| sweatshirt      | 0.8         | 44                |
| beer bottle    | 0.8         | 46                |
| taxi           | 0.8         | 30                |
| lesser panda   | 0.8         | 30                |
</details>

Figure 12. Block-wise Analysis on Flux.1-dev.

# F.2. Qualitative Examples

Following the results presented in the main text, we provide additional qualitative visualizations for each model in this section. To ensure a rigorous and consistent comparison, all images were generated using the same seeds as the baseline samples produced without our internal manipulations. This allows for a direct, side-by-side observation of how our method influences the generative process across various architectures while maintaining the fundamental structural characteristics of the original latent trajectories.

# G. Computational Cost

We analyze the computational costs of DAVE and compare it with existing diversity-enhancement methods. Let T denote the number of sampling steps and B the batch size.

Baseline diffusion or flow-based sampling requires one model evaluation per step, resulting in linear complexity with respect to both the number of sampling steps and the batch size: O(T B).

Lightweight perturbation-based methods, such as CADS, preserve linear computational scaling by applying lightweight conditioning perturbations during sampling without explicit inter-sample interactions. Similarly, SPARKE maintains linear asymptotic scaling with respect to the batch size, although additional feature-statistics computations may introduce nontrivial practical memory overhead during sampling.

In contrast, interaction-based diversity-enhancement methods, including Particle Guidance (PG), DiverseFlow, OSCAR, and SPELL, require explicit or implicit interactions across samples during sampling. Particle Guidance and SPELL rely on inter-sample repulsion terms, DiverseFlow introduces kernel-based coupling across trajectories, and OSCAR performs batch-wise diversity optimization using feature-space interactions. Consequently, these methods may incur computational and memory overhead that scales quadratically with respect to the batch size due to batch-wise inter-sample interactions, resulting in overall complexity of O(T B2).

DAVE performs lightweight representation-level modulation only during the early denoising stage. By directly operating on intermediate representations without requiring feature-bank maintenance or batch-wise inter-sample interactions, DAVE preserves linear batch scaling and maintains near-identical practical runtime and memory usage compared to the baseline.

Table 10 summarizes the computational complexity of existing diversity-enhancement approaches. Lightweight perturbationbased methods preserve linear scaling with relatively small additional overhead, whereas interaction-based methods incur substantially larger computational and memory costs due to batch-wise diversity interactions during sampling. In contrast, DAVE maintains linear scaling with respect to the batch size through lightweight representation-level modulation without requiring feature-bank maintenance, inter-sample interactions, or additional optimization procedures.

As shown in Table 11, DAVE preserves near-identical runtime and reserved GPU memory usage to the original SD3 sampler while substantially outperforming interaction-based approaches in practical efficiency. In particular, DAVE avoids the significant computational and memory overhead associated with costly batch-wise inter-sample interactions during sampling. All measurements are averaged over 100 runs under identical sampling settings on a single NVIDIA H100 GPU.

Table 10. Computational complexity of diversity-enhancement methods. 

<table><tr><td>Method</td><td>Asymptotic time</td><td>Additional memory</td><td>Batch scaling</td></tr><tr><td>Baseline</td><td> $O(TB)$ </td><td> $O(B)$ </td><td>Linear</td></tr><tr><td>CADS</td><td> $O(TB)$ </td><td> $O(B)$ </td><td>Linear</td></tr><tr><td>SPARKE</td><td> $O(TB)$ </td><td> $O(B)$ </td><td>Linear</td></tr><tr><td>Particle Guidance</td><td> $O(TB^{2})$ </td><td> $O(B^{2})$ </td><td>Quadratic</td></tr><tr><td>DiverseFlow</td><td> $O(TB^{2})$ </td><td> $O(B^{2})$ </td><td>Quadratic</td></tr><tr><td>OSCAR</td><td> $O(TB^{2})$ </td><td> $O(B^{2})$ </td><td>Quadratic</td></tr><tr><td>SPELL</td><td> $O(TB^{2})$ </td><td> $O(B^{2})$ </td><td>Quadratic</td></tr><tr><td>DAVE (Ours)</td><td> $\mathbf{O}(TB)$ </td><td> $\mathbf{O}(B)$ </td><td>Linear</td></tr></table>

Table 11. Runtime and memory comparison of diversity-enhancement methods on SD3. 

<table><tr><td>Method</td><td>ms/img ↓</td><td>img/s ↑</td><td>Reserved Mem ↓</td></tr><tr><td colspan="4">Batch Size = 1</td></tr><tr><td>Baseline</td><td>2197.06 ± 25.93</td><td>0.455</td><td>21.0G</td></tr><tr><td>CADS</td><td>2217.12 ± 22.10</td><td>0.451</td><td>21.0G</td></tr><tr><td>SPARKE</td><td>2330.86 ± 46.11</td><td>0.429</td><td>37.7G</td></tr><tr><td>DAVE (Ours)</td><td>2201.55 ± 42.20</td><td>0.454</td><td>21.0G</td></tr><tr><td colspan="4">Batch Size = 4</td></tr><tr><td>Baseline</td><td>1956.97 ± 4.75</td><td>0.511</td><td>33.4G</td></tr><tr><td>SPELL</td><td>2791.49 ± 2.48</td><td>0.358</td><td>34.3G</td></tr><tr><td>Particle Guidance</td><td>9400.93 ± 36.32</td><td>0.106</td><td>49.4G</td></tr><tr><td>DAVE (Ours)</td><td>1956.98 ± 3.25</td><td>0.511</td><td>33.4G</td></tr></table>

![](images/2cab64a7a11c710563c8842be671a86cbd26c240e449aff5bdafacb960f741d4.jpg)  
Figure 13. Qualitative results across different models.