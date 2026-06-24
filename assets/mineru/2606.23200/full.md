# NGPS: Structure-Preserving Self-Supervised Denoising via Neighbor-Guided Patch Sampling

Jaehyun Cho<sup>1</sup> and YoungJoon Yoo<sup>1,2</sup>

<sup>1</sup> Department of Artificial Intelligence, Chung-Ang University, 84 Heukseok-ro,

Dongjak-gu, Seoul, Republic of Korea

{q7011805,yjyoo3312}@cau.ac.kr

2 SNUAILAB, 1 Gwanak-ro, Gwanak-gu, Seoul 08826, Korea yjyoo3312@snuailab.ai

Abstract. Neighboring-slice self-supervised denoising is attractive for volumetric medical imaging, yet inter-slice misalignment breaks anatomical correspondence and often yields ghosting and blurred margins when adjacent slices are used naïvely as targets. We propose Neighbor-Guided Patch Sampling (NGPS), a lightweight framework that constructs neighboring supervision under local inter-slice misalignment. To avoid learning from misleading targets, prior methods commonly mask discrepant regions, but this stabilizes training at the cost of leaving a non-trivial portion of neighboring evidence unexploited, particularly around highfrequency anatomical boundaries. NGPS addresses this by decoupling structure matching from signal retrieval: for each masked location, it searches a local neighborhood for structurally similar candidate patches using a simple guide image (e.g., fast bilateral filtering), while retrieving the supervision signal directly from the raw noisy neighbor at the matched coordinates. By matching on a noise-attenuated guide while retrieving raw values from neighboring slices, NGPS constructs local pseudo targets without dense deformation-field estimation or spatial resampling. Across the evaluated CT and synthetic-Rician MRI settings, NGPS improves fidelity and structure-sensitive metrics. Code is available at https://github.com/cv-cho/NGPS.

Keywords: Self-supervised denoising · Medical image restoration · Lowdose CT · Structure preservation · Self-supervised training

## 1 Introduction

Medical imaging enables non-invasive diagnosis and longitudinal monitoring [28]. However, X-ray-based modalities such as CT and fluoroscopy expose patients and clinicians to ionizing radiation, motivating low-dose acquisition [19]. Dose reduction induces photon starvation, yielding quantum noise and reconstruction artifacts that obscure fine anatomical structures, especially in high-attenuation regions [1, 6, 24]. This motivates denoising methods that restore structural and signal fidelity while preserving clinically meaningful content [8, 13, 20].

Slice n + 1

![](images/653fed6c55300a059ab3fc270c2b1c8976bcbee62a98700fd887714f4039c31e.jpg)

![](images/ed8fcb595bb0a6f850fc0261e65274c99f78b78302bc06034cd36d6f451d3609.jpg)

Difference Map (<sub>j</sub>n <sub>¡</sub> (n + 1)<sub>j</sub>)  
![](images/fe5d21d6064bb780bfe439bf304a755c6130bdda01ccf7fe38db1d186911d709.jpg)

Discarded Info (¿ > 0:1)  
![](images/7bb4c009d5f46809a1692f1293d7f2ced69532b6290ce608a2ce52694426af57.jpg)  
(a) Qualitative visualization of discrepancy-based masking

Distribution of Discarded Information in ROIs (¿ = 0:1)  
![](images/2f67afea5417af642956d3ab16b05d7c915403b2c28ea06e7351e011af59966a.jpg)  
(b) Distribution of discarded information

![](images/3fd22f2ead053bba102f0e7875168e9a7fb37fa831a1e844d1d9ed7680823a6f.jpg)  
(c) High-frequency distribution of pixels  
Fig. 1: Illustration of inter-slice anatomical misalignment and the resulting critical information loss from conventional masking strategies $( \tau = 0 . 1 )$ in 2.5 mm LIDC-IDRI data [2]. (a) The absolute diference map $( | n - ( n + 1 ) | )$ ) and the masked pixels (highlighted in red). (b) The histogram revealing the severity of this spatial information loss within critical Regions of Interest (ROIs) across the validation set. (c) The split violin plot showing that the discarded (masked) pixels are heavily concentrated in the high-frequency spectrum (high gradient magnitude) compared to the retained pixels.

Prior approaches include classical priors (e.g., NLM, BM3D, TV) [5,7,10,22] and supervised deep networks [27,39], but paired clean targets are often infeasible and motion or anatomical variability further complicates alignment [17, 24, 34]. Self-supervised learning (SSL) alleviates this constraint by learning directly from noisy observations [3,16,18,35]. For volumetric data, neighboring-slice SSL uses adjacent slices as supervision [25, 36, 41], yet inter-slice misalignment breaks the same-coordinate correspondence, so relevant anatomical evidence is often displaced rather than missing. Fig. 1 highlights that discrepancy-based masking can exclude a substantial portion of pixels within clinically relevant ROIs, removing on average about 20% of the available neighboring supervision. Moreover, the excluded regions are concentrated around high-gradient anatomical boundaries. This creates a persistent dilemma: masking-based remedies stabilize training by avoiding misleading targets, but systematically withhold supervision in structurally challenging regions. Registration-based alternatives explicitly estimate correspondences and resample neighboring slices. Under severe noise or larger inter-slice gaps, the estimated alignment can become less reliable, while spatial resampling may smooth edges and add alignment overhead. These trade-ofs motivate local target retrieval without dense warping. Therefore, a key challenge is to retrieve displaced anatomical evidence from a local neighborhood rather than treating it as unavailable.

Motivated by this, we propose Neighbor-Guided Patch Sampling (NGPS), a misalignment-aware neighboring-slice SSL framework that is designed to (i) recover displaced supervision within a local neighborhood without dense deformation field estimation or spatial warping, (ii) remain compatible with Noise2Noisestyle [18] training, and (iii) preserve high-frequency anatomical margins with minimal overhead. NGPS reframes misalignment handling as a training-time supervision construction problem, rather than an inference-time non-local denoising prior. Specifically, NGPS constructs a neighboring pixel bank by matching pre-filtered guide patches across adjacent slices and retrieving target values from the corresponding raw noisy neighbors. The primary contributions are:

– Misalignment as displaced supervision: We identify a structural supervision gap in neighboring-slice SSL caused by displacement, clarifying the masking versus registration trade-of.

– Neighbor-Guided Patch Sampling (NGPS): We introduce a lightweight patch-based supervision recovery mechanism using guide-feature search and raw-target retrieval for misalignment-aware SSL.

– Comprehensive validation: We demonstrate consistent gains across lowdose CT and MRI benchmarks (AAPM-Mayo [24], LIDC-IDRI [2], IXI [4]), with improved preservation of fine anatomical margins.

## 2 Related Work

Self-supervised Learning for Denoising. To overcome the dificulty of obtaining paired noisy-clean images [24,34], self-supervised learning (SSL) has become a practical alternative. Early image-domain methods, such as Blind-Spot Networks [3, 16], construct supervision without directly observing the target pixel, typically through masking or blind-spot/downsampling designs. While efective in broad settings, these schemes restrict the directly available context and can introduce smoothing or checkerboard artifacts [33]. Recent image-domain SSL methods broaden target construction through masking, downsampling, noise injection, or generative pseudo targets. DifDenoise [11], for example, uses a conditional-difusion pipeline to generate pseudo-clean supervision. These methods difer in how supervision is synthesized or withheld. In the neighboring-slice setting studied here, Fig. 1 shows that discrepancy masking removes supervision concentrated near high-gradient anatomical boundaries.

Volumetric SSL and Misalignment Handling. Volumetric medical SSL can exploit redundancy along diferent axes. Patch2Self [12] exploits difusion-MRI qspace redundancy through held-out-volume regression. Our setting instead uses spatial redundancy across adjacent anatomical slices, which provide alternative noisy observations under the standard slice-independent-noise assumption. This neighboring-slice formulation implicitly assumes anatomical correspondence; in practice, inter-slice displacement can make same-coordinate supervision blur shifted structures. Existing remedies largely follow registration- or maskingbased strategies. Registration-based methods such as MSR2AU-Net [15] and Deformed2Self [36] estimate correspondences and warp neighboring slices. Such alignment can become less reliable under severe degradation or larger interslice gaps, and spatial resampling may alter local signal and noise statistics. Masking-based methods such as Noise2Sim [25] and NS-N2N [41] instead exclude discrepant pixels from the reconstruction loss. This avoids misleading samecoordinate targets but can reduce supervision near high-gradient boundaries, as quantified in Fig. 1. NGPS addresses this displaced-supervision regime through local retrieval rather than dense warping or exclusion.

Patch-based Sampling and Neighbor-Aware Processing. Patch-based self-similarity provides a complementary direction for constructing supervision without paired targets. Pixel2Pixel (P2P) [21] constructs a “PixelBank” by searching for similar patches within a single noisy image and sampling replacement targets from non-local neighbors. Extending this idea to volumetric SSL is non-trivial: exhaustive image-wide search can be costly, and same-image target selection can couple the retrieval process to the same noisy realization. Neighboring slices provide an alternative target source under the standard slice-independent-noise assumption, provided displaced correspondence can be resolved. At a broader conceptual level, Graph Flow Matching [29] introduces neighbor-aware aggregation in a flow-based image-generation setting rather than for denoising-target construction. NGPS difers from both settings by applying local guide-based patch search at locations flagged by inter-slice discrepancy and retrieving raw adjacent-slice values as training targets.

## 3 Methodology

We propose Neighbor-Guided Patch Sampling (NGPS), a misalignment-aware neighboring-slice SSL framework that constructs pseudo targets for misaligned regions via three steps, as illustrated in Fig. 2: (i) fast guide generation with conservative masking, (ii) decoupled structural matching on the guide, and (iii) raw target retrieval with a Top-K ensemble. We then train a 2D slice denoiser $f _ { \theta }$ with a hybrid objective that uses standard same-coordinate supervision in static regions (blue in Fig. 2) and switches to NGPS-recovered targets in misaligned regions (purple). Furthermore, we incorporate an output-level regional consistency regularizer to reduce slice-to-slice flicker in non-masked areas.

## 3.1 Problem Setup and Motivation

Let $x , y \in \mathbb { R } ^ { H \times W \times D }$ denote a clean volumetric image and its noisy observation, modeled as $y _ { z } ( p ) = x _ { z } ( p ) + n _ { z } ( p )$ , where $z \in \{ 1 , \ldots , D \}$ is the slice index and $p$ is the spatial coordinate. Assuming zero-mean $( \mathbb { E } [ n _ { z } ( p ) ] = 0 )$ and slice-independent noise, neighboring-slice SSL learns a denoiser using adjacent slices as pseudo targets [18]. Here, slice independence concerns noise across the input and target slices; it does not require pixel-wise i.i.d. noise within each slice. However, inter-slice misalignment breaks this same-coordinate correspondence. Under an inter-slice displacement δ, a local anatomical structure in the clean signal $x _ { z } ( p )$ appears shifted in the adjacent slice as $x _ { z ^ { \prime } } ( p ) \approx x _ { z } ( p - \delta )$ . Naive same-coordinate supervision can encourage averaging between these shifted structures, yielding a superposition ${ \scriptstyle { \frac { 1 } { 2 } } } \big ( x _ { z } ( p ) + x _ { z } ( p - \delta ) \big )$  that can produce ghosting and blurred anatomical boundaries.

![](images/e4796bfc6abf9c1d5f0cc0f38957678a7925d18810beae579ffe81227e3f6ec3.jpg)  
Fig. 2: Overall NGPS pipeline. A noisy slice triplet is low-pass filtered to produce noise-attenuated guide (“Pseudo-Clean”) slices, whose pairwise diferences yield direction-aware threshold masks with reduced sensitivity to raw noise. For each masked location, NGPS searches a local window in the adjacent slice for the Top-K similar patches (green); the matched coordinates are then used to retrieve and average the corresponding center-pixel values from the original noisy adjacent slice, forming directionaware retrieved targets. Unmasked static pixels use same-coordinate neighboring supervision to compute $\mathcal { L } _ { N 2 N }$ (blue), whereas masked pixels use the retrieved targets to compute L<sub>NGP</sub> <sub>S</sub> (purple). The two reconstruction terms are summed as L<sub>recon</sub>; regional consistency is applied separately to static regions as described in Sec. 3.4.

## 3.2 From Misalignment to Design Principles

To overcome this dilemma, NGPS reframes misalignment as local displacement rather than absence. Instead of discarding misaligned regions, we actively retrieve the corresponding anatomical evidence from a nearby coordinate q in the adjacent slice. This perspective establishes three core design requirements for our framework: (i) Recover displaced supervision via explicit local retrieval rather than passive exclusion, (ii) Retain raw neighboring values as training targets while using filtered guides only for correspondence search, and (iii) Remain lightweight without dense deformation-field estimation or spatial resampling.

We use selection bias to denote noise-driven coordinate selection that induces correlation between input noise and the residual noise of a retrieved target. NGPS performs discrete local correspondence search. Unlike dense registration, it estimates no dense deformation field and performs no spatial warping or interpolation of neighboring slices.

## 3.3 Neighbor-Guided Patch Sampling (NGPS)

For each pixel flagged as misaligned, NGPS constructs a pseudo target $t _ { z  z ^ { \prime } } ( p )$ via (i) fast guide generation with conservative masking, (ii) decoupled patch matching on the guide, and (iii) raw-value retrieval with Top-K aggregation.

Guide generation and misalignment mask. We generate an edge-preserving guide volume $\tilde { y }$ by sequentially applying a 2D bilateral filter (BF) and a median filter (MF) to suppress quantum noise while preserving structural boundaries:

$$
\tilde {y} _ {z} (p) = \operatorname{MF} \left(\operatorname{BF} \left(y _ {z} (p); \sigma_ {s}, \sigma_ {r}\right); \kappa\right),\tag{1}
$$

where $\sigma _ { s }$ and $\sigma _ { r }$ are the spatial and range parameters of the BF, and $\kappa$ is the kernel size of the MF. Let $\mathcal { N } ( z ) = \{ z - 1 , z + 1 \}$ denote the set of available neighboring slice indices (applying reflection padding only at the volume ends). Instead of a single global mask, we define a direction-aware misalignment mask for each neighbor $z ^ { \prime } \in \mathcal { N } ( z )$ based on the same-coordinate guide disagreement:

$$
\mathcal {M} _ {z \rightarrow z ^ {\prime}} (p) = \mathbb {1} \left(\left| \tilde {y} _ {z} (p) - \tilde {y} _ {z ^ {\prime}} (p) \right| > \tau\right),\tag{2}
$$

where $\tau$ is used to flag locations whose same-coordinate guide discrepancy exceeds the selected threshold. For pixels not flagged by $\mathcal { M } _ { z  z ^ { \prime } }$ , we use standard same-coordinate neighboring supervision. For flagged pixels, we avoid samecoordinate targets and invoke local NGPS retrieval.

Decoupled matching and raw retrieval. For each pixel $p \in \mathbb Z ^ { 2 }$ flagged as misaligned with respect to a specific neighboring slice $z ^ { \prime } \in \mathcal { N } ( z )$ , we search for structurally similar candidate patches within a local spatial window $\itOmega _ { p } \Psi \Psi ( e . g .$ $1 5 \times 1 5 )$ in slice $z ^ { \prime }$ . While the similarity search is performed at the patch level to ensure structural context, the supervision signal is retrieved purely as a single scalar value from the center pixel of the matched location. Let $\mathcal { P } _ { k } ( I , p ) \in \mathbb { R } ^ { k \times }$ k extract a $k \times k$ patch centered at $p$ from image I. We compute the guide-based sum of squared diferences (SSD) costs:

$$
\mathcal {D} (p, q; z ^ {\prime}) = \| \mathcal {P} _ {k} (\tilde {y} _ {z}, p) - \mathcal {P} _ {k} (\tilde {y} _ {z ^ {\prime}}, q) \| _ {2} ^ {2}, \quad q \in \Omega_ {p},\tag{3}
$$

and subsequently retrieve the candidate target scalar from the raw center pixel, $y _ { z ^ { \prime } } ( q ) \in \mathbb { R }$ . This separates correspondence search from regression-target retrieval, avoiding direct raw-patch matching while keeping the target value unfiltered.

The guide is computed from noisy observations and therefore does not make the selected coordinates strictly independent of noise. We accordingly treat guide-based matching as an empirical design choice intended to reduce noisedriven target selection, while raw-value retrieval keeps the regression target unfiltered. Supplementary Table S5 compares NGPS with the closest same-slice and adjacent-slice PixelBank-style alternatives.

Top-K ensemble target. To reduce the variance inherent to a single best match, we aggregate the $\mathrm { T o p } { - } K$ candidate pixels for each neighboring direction $z ^ { \prime } .$ . For a given neighbor $z ^ { \prime } .$ , we rank the candidate coordinates in $\varOmega _ { p }$ and select the set $\{ \check { q ^ { ( 1 ) } } , \ldots , q ^ { \check { ( K ) } } \}$ yielding the smallest $\mathcal { D } ( p , q ; z ^ { \prime } )$ . We define the directionaware target $t _ { z  z ^ { \prime } } ( p )$ as the average of these K retrieved center pixels:

$$
t _ {z \rightarrow z ^ {\prime}} (p) = \frac {1}{K} \sum_ {k = 1} ^ {K} y _ {z ^ {\prime}} \big (q ^ {(k)} \big).\tag{4}
$$

This is inspired by the PixelBank philosophy [21] but specialized to volumetric SSL; the sampling is applied directionally (only for misaligned pixels) and the scalar targets are drawn exclusively from the adjacent slice $z ^ { \prime }$ to construct $t _ { z  z ^ { \prime } }$

## 3.4 Training objective

We optimize $f _ { \theta }$ using a hybrid reconstruction loss and regional consistency term:

$$
\mathcal {L} _ {t o t a l} = \mathcal {L} _ {r e c o n} + \lambda \mathcal {L} _ {R C},\tag{5}
$$

where λ is a hyperparameter that controls the strength of the regularization.

Hybrid reconstruction loss Guided by the directional misalignment mask $\mathcal { M } _ { z  z ^ { \prime } }$ , we adaptively switch the supervision signal. Specifically, we apply standard same-coordinate neighboring targets for static regions, and provide the retrieved NGPS targets $t _ { z  z ^ { \prime } }$ for misaligned regions. The hybrid reconstruction loss is thus formulated as:

$$
\mathcal {L} _ {r e c o n} = \mathcal {L} _ {N 2 N} + \mathcal {L} _ {N G P S},\tag{6}
$$

$$
\mathcal {L} _ {N 2 N} = \frac {1}{| \mathcal {N} (z) |} \sum_ {z ^ {\prime} \in \mathcal {N} (z)} \frac {\sum_ {p} \left(1 - \mathcal {M} _ {z \rightarrow z ^ {\prime}} (p)\right)\left(f _ {\theta} (y _ {z}) (p) - y _ {z ^ {\prime}} (p)\right) ^ {2}}{\sum_ {p} \left(1 - \mathcal {M} _ {z \rightarrow z ^ {\prime}} (p)\right) + \epsilon},\tag{7}
$$

$$
\mathcal {L} _ {N G P S} = \frac {1}{| \mathcal {N} (z) |} \sum_ {z ^ {\prime} \in \mathcal {N} (z)} \frac {\sum_ {p} \mathcal {M} _ {z \to z ^ {\prime}} (p) \big (f _ {\theta} (y _ {z}) (p) - t _ {z \to z ^ {\prime}} (p) \big) ^ {2}}{\sum_ {p} \mathcal {M} _ {z \to z ^ {\prime}} (p) + \epsilon},\tag{8}
$$

Table 1: Summary of datasets and noise simulation protocols used in our experiments.

<table><tr><td>Dataset</td><td>Modality</td><td>Anatomy</td><td>Spacing</td><td>Noise Simulation Type</td><td>Train/Test</td></tr><tr><td>AAPM-Mayo [24]</td><td>CT</td><td>Abdomen</td><td>1.0 mm</td><td>Realistic Quarter-Dose (Poisson injection in sinogram)</td><td>8 / 2</td></tr><tr><td>LIDC-IDRI [2]</td><td>CT</td><td>Thorax</td><td>1.25, 2.5 mm</td><td>Simulated Ultra-Low-Dose (Radon Transform, P = 12.5K)</td><td>66 / 6</td></tr><tr><td>IXI-T1 [4]</td><td>MRI</td><td>Brain</td><td>1.2 mm</td><td>Synthetic Rician ( $\sigma \in \{5\%, 7\%, 9\%\}$ )</td><td>52 / 5</td></tr></table>

where $\mathcal { M } _ { z  z ^ { \prime } }$ and $t _ { z  z ^ { \prime } }$ are spatial maps matching the dimensions of the prediction $f _ { \theta } ( y _ { z } )$ , and $\epsilon > 0$ is a small numerical constant preventing division by zero. Each directional loss is normalized by the number of pixels assigned to its corresponding supervision region.

Regional consistency regularization Equation 7 optimizes each slice independently and does not explicitly enforce volumetric coherence. In regions identified as static, adjacent clean signals are expected to be locally similar rather than identical. Following volumetric SSL practice [41], we therefore penalize prediction diferences only on $( 1 - \mathcal { M } _ { z  z ^ { \prime } } )$ , encouraging inter-slice consistency without constraining regions flagged as displaced:

$$
\mathcal {L} _ {R C} = \frac {1}{| \mathcal {N} (z) |} \sum_ {z ^ {\prime} \in \mathcal {N} (z)} \left\| (1 - \mathcal {M} _ {z \to z ^ {\prime}}) \odot \left(f _ {\theta} (y _ {z}) - f _ {\theta} (y _ {z ^ {\prime}})\right) \right\| _ {2} ^ {2}.\tag{9}
$$

## 4 Experiments

## 4.1 Experimental Setup

Datasets and Noise Simulation. To evaluate the proposed NGPS, we use three public medical volumetric datasets. The detailed configurations for each dataset are summarized in Table 1.

AAPM-Mayo (CT) [24]: As a realistic benchmark for low-dose CT (LDCT) denoising, we use abdominal CT scans from the AAPM Low-Dose CT Grand Challenge. Unlike simple additive noise, the Quarter-Dose (QD) images were generated by injecting Poisson noise directly into the projection data (sinograms) of Normal-Dose scans to simulate 25% of the full dose. This dataset serves as a gold standard for evaluating robustness against realistic, spatially correlated CT noise textures. For our experiments, we utilize the 1.0 mm slice reconstructions.

LIDC-IDRI (CT) [2]: To evaluate robustness under ultra-low-dose conditions with severe streak artifacts, we specifically curated a subset of thoracic CT scans acquired with GE Medical Systems scanners. The selected data consists of highresolution and standard scans with slice thicknesses of exactly 1.25 mm and 2.5 mm. We simulate ULD projections via the Radon Transform with a photon count of 12,500, followed by Filtered Back Projection (FBP). This generates realistic streak artifacts and non-stationary noise distributions that pose significant challenges to conventional SSL methods. Unless otherwise noted, aggregate LIDC-IDRI results pool the 1.25-mm and 2.5-mm test subsets; spacing-specific results are reported separately.

Table 2: Quantitative evaluation on the AAPM-Mayo [24] (Quarter-Dose) and LIDC-IDRI [2] (Ultra-Low-Dose) datasets. The best results are highlighted in bold, and the second-best are underlined. ↑ indicates higher is better, while ↓ indicates lower is better.

<table><tr><td rowspan="2">Method</td><td colspan="5">AAPM-Mayo (Quarter-Dose CT)</td><td colspan="5">LIDC-IDRI (Simulated ULD CT)</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>FSIM ↑</td><td>HFEN ↓</td><td>GMSD ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>FSIM ↑</td><td>HFEN ↓</td><td>GMSD ↓</td></tr><tr><td>Baseline (Noisy)</td><td>30.30</td><td>0.7222</td><td>0.8772</td><td>0.3363</td><td>0.0874</td><td>22.14</td><td>0.4213</td><td>0.5920</td><td>0.8429</td><td>0.1959</td></tr><tr><td>BM3D [10]</td><td>34.67</td><td>0.7764</td><td>0.8547</td><td>0.3258</td><td>0.0900</td><td>25.07</td><td>0.5825</td><td>0.7768</td><td>0.6010</td><td>0.1449</td></tr><tr><td>DIP [31]</td><td>35.36</td><td>0.7895</td><td>0.8804</td><td>0.2766</td><td>0.0426</td><td>26.77</td><td>0.6235</td><td>0.8074</td><td>0.5910</td><td>0.1337</td></tr><tr><td>NAC [37]</td><td>34.61</td><td>0.8171</td><td>0.9062</td><td>0.2344</td><td>0.0664</td><td>24.83</td><td>0.5690</td><td>0.7500</td><td>0.6096</td><td>0.1439</td></tr><tr><td>ZS-N2N [23]</td><td>33.68</td><td>0.8129</td><td>0.9107</td><td>0.2911</td><td>0.0631</td><td>26.63</td><td>0.5979</td><td>0.8080</td><td>0.5545</td><td>0.1259</td></tr><tr><td>Pixel2Pixel [21]</td><td>33.71</td><td>0.8357</td><td>0.9117</td><td>0.3183</td><td>0.0739</td><td>25.58</td><td>0.5078</td><td>0.7277</td><td>0.6196</td><td>0.1237</td></tr><tr><td>Noise2Void [16]</td><td>32.57</td><td>0.7786</td><td>0.8977</td><td>0.3286</td><td>0.0739</td><td>26.10</td><td>0.5995</td><td>0.8295</td><td>0.4948</td><td>0.1078</td></tr><tr><td>NB2NB [14]</td><td>33.21</td><td>0.7922</td><td>0.9037</td><td>0.3187</td><td>0.0715</td><td>27.23</td><td>0.6063</td><td>0.8321</td><td>0.4687</td><td>0.1003</td></tr><tr><td>Filter2Noise [30]</td><td>35.23</td><td>0.8334</td><td>0.9265</td><td>0.2895</td><td>0.0625</td><td>28.81</td><td>0.7181</td><td>0.8716</td><td>0.5188</td><td>0.1036</td></tr><tr><td>Deformed2Self [36]</td><td>35.85</td><td>0.8662</td><td>0.9144</td><td>0.2299</td><td>0.0377</td><td>28.94</td><td>0.6703</td><td>0.8170</td><td>0.5134</td><td>0.0808</td></tr><tr><td>Noise2Sim [25]</td><td>35.49</td><td>0.8639</td><td>0.9420</td><td>0.2406</td><td>0.0390</td><td>28.81</td><td>0.7817</td><td>0.8749</td><td>0.5003</td><td>0.1040</td></tr><tr><td>NS-N2N [41]</td><td>35.91</td><td>0.8584</td><td>0.9235</td><td>0.2325</td><td>0.0396</td><td>30.62</td><td>0.8080</td><td>0.8944</td><td>0.4406</td><td>0.0777</td></tr><tr><td>Ours</td><td>36.68</td><td>0.8986</td><td>0.9470</td><td>0.2056</td><td>0.0362</td><td>31.03</td><td>0.8102</td><td>0.9168</td><td>0.4161</td><td>0.0788</td></tr></table>

IXI-T1 (MRI) $I 4 J \colon$ For a controlled cross-modality evaluation, we use brain scans from the IXI dataset with a slice thickness of 1.2 mm. We add synthetic Rician noise at three intensity levels $( \sigma \in \{ 5 \% , 7 \% , 9 \% \} )$ ) to the clean volumetric data.

Evaluation Metrics. We employ five quantitative metrics to assess diferent aspects of image quality. Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity (SSIM) [32] are used to measure general signal fidelity. To evaluate perceptual quality and low-level feature preservation, we utilize Feature Similarity (FSIM) [40]. Furthermore, given the importance of preserving fine anatomical details in medical imaging, we employ High-Frequency Error Norm (HFEN) [26] and Gradient Magnitude Similarity Deviation (GMSD) [38] to specifically measure the restoration quality of edges and textures (lower values indicate better performance for HFEN and GMSD).

Implementation Details. Our method is implemented in PyTorch and evaluated on a single CPU/GPU workstation (AMD Ryzen 9 9950X, NVIDIA RTX 5090). Baselines with public implementations follow released/recommended settings; NS-N2N [41] is reimplemented from the paper using the same NAFNet [9] backbone as NGPS. NGPS uses fixed default hyperparameters across datasets (patch size $p { = } 7 .$ , window size $W { = } 1 5$ , Top-K $K { = } 4$ , masking threshold $\tau { = } 0 . 0 5 )$ , and we train with AdamW $( 2 \times 1 0 ^ { - 4 }$ learning rate, $1 0 ^ { - 5 }$ weight decay) for 10 epochs with batch size 4, using $\lambda { = } 0 . 5$ . Additional NGPS architecture and training details are provided in Supplementary Sec. S1.

## 4.2 Quantitative Comparison

Results on Low-Dose and Ultra-Low-Dose CT Datasets. Table 2 compares quarter-dose AAPM-Mayo [24] and simulated-ULD LIDC-IDRI [2]. NGPS leads all five AAPM metrics. On LIDC-IDRI, it improves PSNR, FSIM, and HFEN over NS-N2N [41] by 0.41 dB, 0.0224, and 0.0245, respectively, whereas the SSIM margin is small (0.0022) and NS-N2N has a slightly lower GMSD (0.0777 vs. 0.0788). Volume-level paired 95% confidence intervals exclude zero for PSNR and FSIM, but include zero for SSIM, HFEN, and GMSD (Supplementary Table S3). We therefore interpret the LIDC-IDRI result as improved fidelity and boundary-sensitive restoration rather than uniform dominance across all metrics. In the evaluated ULD and thicker-slice settings, Deformed2Self [36] shows larger degradation, while masking-based methods omit discrepant regions from their reconstruction losses. NGPS instead retrieves local supervision in these regions without dense warping.

![](images/4336cca205b3620823f9efd4b568fdc84d1360c5662629aa376401f1326820e8.jpg)  
Fig. 3: Quantitative comparison on IXI dataset [4] (Simulated Rician noise). We add diferent levels of Rician noise (5%, 7%, and 9%) for corruption.

Results on the IXI MRI Dataset with Rician Noise. We evaluate robustness on the IXI dataset [4] under increasing synthetic Rician noise (5%, 7%, and 9%), summarizing performance trends in Fig. 3 (full values in Supplementary Table S1). All methods degrade as the corruption level increases, but NGPS maintains the highest PSNR and SSIM across the tested levels. Among the baselines, Deformed2Self [36] ranks second overall, which is consistent with the usefulness of inter-slice context in this setting. The relative gaps among Deformed2Self, Noise2Sim [25], NS-N2N [41], and the single-image baselines vary across corruption levels. Nevertheless, NGPS remains highest in both metrics throughout the tested range, demonstrating consistent performance under the evaluated synthetic Rician corruption.

## 4.3 Qualitative Comparison

Fig. 4 compares quarter-dose AAPM-Mayo [24] and ULD LIDC-IDRI [2] CTs. The baselines exhibit clear visual trade-ofs in the highlighted ROIs: single-image methods such as BM3D [10], Noisy-As-Clean [37], Filter2Noise [30], Pixel2Pixel [21], and ZS-N2N [23] either retain more residual noise or produce softer anatomical details. The warping-based baseline Deformed2Self [36] shows local artifacts in some regions, consistent with the dificulty of estimating correspondence under strong degradation. Masking-based SSL methods such as Noise2Sim [25] and

![](images/d5d0c89e164ded2318e7e7581fba287f90d9adf17b395ff4cf7a900fc6ddffbd.jpg)

(a) AAPM-Mayo [24] (Quarter-Dose CT). Pronounced quantum noise and over-smoothing artifacts.  
![](images/3cf6fc6c09572b87a2f4d2cb20e890a9bcba2db3dff7078b524c79f9ed901d0f.jpg)  
(b) LIDC-IDRI [2] (Simulated ULD CT, P =12.5K). Severe, spatially correlated streak artifacts.  
Fig. 4: Qualitative comparisons on quarter-dose and ultra-low-dose CT. The top-left shows the noisy input with a red ROI, while the rest display magnified ROIs.

NS-N2N [41] produce smoother boundaries in the displayed ROIs, which is consistent with omitting discrepant pixels from the reconstruction loss. In these examples, NGPS reduces residual corruption while retaining sharper anatomical margins through local raw-target retrieval.

Under 9% synthetic Rician noise on IXI MRI [4] (Fig. 5), similar tendencies are visible. Single-image methods retain more residual corruption or oversmoothing, the warping-based baseline shows local degradation under the tested corruption, and masking-based outputs show softer cortical boundaries in the highlighted area. In the highlighted example, NGPS retains sharper local structures while reducing residual corruption, consistent with the quantitative trends.

## 4.4 Discussion

Robustness to Slice Thickness and Masking Threshold. Table 3 isolates spacing changes using independently trained 1.25 mm and 2.5 mm models. NGPS decreases by 0.16 dB, compared with 0.63 dB for NS-N2N [41], 1.88 dB for Deformed2Self [36], and 2.83 dB for Noise2Sim [25]. The threshold sweep in Fig. 6a further shows that NGPS is comparatively stable across τ , whereas NS-N2N degrades under both strict and loose masking. These results support robustness within the evaluated spacing and threshold ranges; controlled larger-gap failures and confidence-based rejection are analyzed in Supplementary Sec. S4.7.

![](images/89ab2ba7bddcf4deb78257a66dd8be14999e2ab6eddcc6517f1803e3d051159b.jpg)  
Fig. 5: Qualitative results on IXI [4] under 9% synthetic Rician noise. In the highlighted region, several baselines show more residual corruption or softer cortical boundaries. The warping- and masking-based baselines exhibit local structure loss in this example, whereas NGPS retains sharper anatomical details.

Table 3: Sensitivity to slice thickness on LIDC-IDRI [2]. ∆ PSNR reports the change from 1.25 mm to 2.5 mm; the best results are highlighted in bold.

<table><tr><td rowspan="2">Method</td><td colspan="2">1.25 mm (Thin Slice)</td><td colspan="2">2.5 mm (Thick Slice)</td><td rowspan="2">Δ PSNR ↓</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>Noise2Sim [25]</td><td>30.53</td><td>0.7963</td><td>27.70</td><td>0.7738</td><td>-2.83 dB</td></tr><tr><td>Deformed2Self [36]</td><td>30.52</td><td>0.7553</td><td>28.64</td><td>0.6550</td><td>-1.88 dB</td></tr><tr><td>NS-N2N [41]</td><td>30.83</td><td>0.8092</td><td>30.20</td><td>0.7877</td><td>-0.63 dB</td></tr><tr><td>Ours (NGPS)</td><td>31.08</td><td>0.8103</td><td>30.92</td><td>0.7994</td><td>-0.16 dB</td></tr></table>

Multi-geometry Search-window Selection. We estimate match-ofset CDFs using a 41×41 reference search across AAPM [24] 1.0 mm, IXI [4] 1.2 mm, and LIDC-IDRI [2] 1.25/2.5 mm, with p=7 and K=4. A 15×15 window covers over 80% of the selected ofsets for every geometry and 98% on LIDC-IDRI 1.25 mm (Fig. 6b). Because larger windows add quadratic candidate cost with limited coverage gain, we use W =15 as a common default. Supplementary Table S7 shows that W =15 is within 0.02 dB of the best 1.25 mm result and performs best at 2.5 mm; we therefore use it as a practical common default rather than a universal optimum.

Efect of Top-K Ensemble Size. In NGPS, K controls the number of retrieved candidate patches used to form the Top-K ensemble target. Fig. 7 shows a clear trade-of on the LIDC dataset [2]: performance improves from small ensembles to a moderate K, then declines when K becomes too large. The best results are achieved at K = 4 (PSNR 31.03 dB, SSIM 0.8102). With K = 1 or 2, the ensemble is too small to suficiently reduce residual noise. With larger K (8 or 16), lower-quality or misaligned matches are increasingly included, which blurs fine structures. Thus, K = 4 provides the best tested noise-detail balance.

![](images/6d42f3c32c13829cc10938e5d8866e46b195f3d9505adf501829489d537bd2ca.jpg)  
(a) Ablation of Masking threshold τ.

![](images/5b540f0f67891ae1cce167b90e81364c75fa05eec782599332d55031a4d58064.jpg)  
(b) CDF of inter-slice displacement (window selection).

Fig. 6: Ablations for mask sensitivity and search-window design. Left: PSNR trends under varying τ. Right: displacement CDF used to set the NGPS search radius.  
![](images/454ec460b40f9ff7aaacfaab482a4b608f8407e5a00d575805f7a35f7a86c60a.jpg)  
Fig. 7: Ablation study on the Top-K ensemble size in the NGPS module.

Table 4: Comparison of preprocessing time overhead for misalignment handling. Time is measured for a standard medical volume consisting of 100 slices (512×512 resolution).

<table><tr><td>Method</td><td>Target Preparation Strategy</td><td>Compute Device</td><td>Time (per 100 slices) ↓</td></tr><tr><td>Pixel2Pixel [21]</td><td>Patch-based Pixel Bank Creation</td><td>GPU</td><td> $\sim 5.38$  s</td></tr><tr><td>Noise2Sim [25]</td><td>Mean Filter Masking</td><td>GPU</td><td> $\sim 0.005$  s</td></tr><tr><td>NS-N2N [41]</td><td>NLM Denoising + Median Masking</td><td>CPU + GPU</td><td> $\sim 13.36$  s</td></tr><tr><td>Ours (NGPS)</td><td>LPF Masking + Vectorized Patch Search</td><td>CPU + GPU</td><td> $\sim 0.72$  s</td></tr></table>

Computational Eficiency. Table 4 compares target-preparation cost for a 100-slice volume. NGPS requires approximately 0.72 s, about 19× faster than NS-N2N [41] and 7.5× faster than Pixel2Pixel [21]; Noise2Sim [25] is faster but attains lower restoration metrics in Table 2. The eficiency of NGPS comes from restricting vectorized patch search to masked pixels and using the lightweight bilateral-plus-median guide.

## 5 Conclusion

We presented NGPS, a lightweight framework for constructing neighboring-slice supervision under inter-slice misalignment. NGPS forms noise-attenuated guides and direction-specific discrepancy masks, performs local Top-K patch search at flagged locations, and retrieves the corresponding raw adjacent-slice values as training targets. Its hybrid objective combines same-coordinate neighboring supervision at unflagged locations with NGPS-retrieved targets at flagged locations, together with regional consistency, without dense warping or a learnable alignment module. Across a realistic quarter-dose CT benchmark, simulated ultra-low-dose CT, and synthetic-Rician MRI, NGPS consistently improves fidelity and structure-sensitive restoration: it leads all AAPM [24] metrics, improves PSNR, FSIM, and HFEN over NS-N2N [41] on simulated ULD CT, and achieves the highest PSNR and SSIM across the evaluated MRI noise levels. These results support local supervision retrieval as an efective and practical strategy for structure-preserving volumetric self-supervised denoising.

Limitations and Future Work. NGPS assumes that a locally corresponding structure with suficiently consistent appearance exists within the search window. This assumption can fail under large through-plane gaps, abrupt anatomical changes, staining variation, signal dropout, or other non-geometric inter-slice appearance changes. The optional match-cost gate mitigates degradation in the tested largegap setting, but does not remove this assumption. The Noise2Noise [18] formulation does not require pixel-wise i.i.d. noise within each slice, but assumes zeromean noise and negligible cross-slice correlation between input- and target-slice noise; cross-slice correlated artifacts remain outside the current validation scope. The fixed p, W , K, and τ values are practical defaults rather than universal optima. Finally, the IXI [4] experiments provide a controlled synthetic-noise evaluation; future work should prioritize acquisition-realistic low-field MRI, noisy 3D microscopy, and downstream clinical validation.

## Acknowledgements

This work was supported by the SNUAILAB, the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) [RS-2021-II211341, Artificial Intelligence Graduate School Program (Chung-Ang University) and RS-2022-II220124, Development of Artificial Intelligence Technology for Self-Improving Competency-Aware Learning Capabilities]. Also, this research was supported by the "Regional Innovation System & Education (RISE)" through the Seoul RISE Center, funded by the Ministry of Education (MOE) and the Seoul Metropolitan Government. (2026-RISE-01-024-04)

## References

1. Andreozzi, E., Pirozzi, M.A., Fratini, A., Cesarelli, G., Cesarelli, M., Bifulco, P.: A novel image quality assessment index for edgeaware noise reduction in low-dose fluoroscopy: Preliminary results. In: Int. Conf. e-Health Bioeng. (EHB). pp. 1–5 (2020)

2. Armato III, S.G., McLennan, G., Bidaut, L., McNitt-Gray, M.F., Meyer, C.R., Reeves, A.P., Clarke, L.P.: The lung image database consortium (lidc) and image database resource initiative (idri): A completed reference database of lung nodules on ct scans. Medical Physics 38(2), 915–931 (2011)

3. Batson, J., Royer, L.: Noise2self: Blind denoising by self-supervision. In: ICML. pp. 524–533 (2019)

4. Biomedical Image Analysis Group: IXI dataset. https://brain- development. org/ixi-dataset/ (2018)

5. Buades, A., Coll, B., Morel, J.M.: A non-local algorithm for image denoising. In: CVPR. vol. 2, pp. 60–65 (2005)

6. Cesarelli, M., Bifulco, P., Cerciello, T., Romano, M., Paura, L.: X-ray fluoroscopy noise modeling for filter design. Int. J. Comput. Assist. Radiol. Surg. 8(2), 269–278 (2013)

7. Chambolle, A.: An algorithm for total variation minimization and applications. J. Math. Imaging Vis. 20(1), 89–97 (2004)

8. Chen, H., Zhang, Y., Kalra, M.K., Lin, F., Chen, Y., Liao, P., Wang, G.: Low-dose ct with a residual encoder-decoder convolutional neural network. IEEE Trans. Med. Imaging 36(12), 2524–2535 (2017)

9. Chen, L., Chu, X., Zhang, X., Sun, J.: Simple baselines for image restoration. In: European Conference on Computer Vision. pp. 17–33. Springer (2022)

10. Dabov, K., Foi, A., Katkovnik, V., Egiazarian, K.: Image denoising by sparse 3-d transform-domain collaborative filtering. IEEE TIP 16(8), 2080–2095 (2007)

11. Demir, B., Liu, Y., Chen, X., Chen, E.Z., Zhao, L., Mailhe, B., Sun, S.: Difdenoise: self-supervised medical image denoising with conditional difusion models. arXiv preprint arXiv:2504.00264 (2025)

12. Fadnavis, S., Batson, J., Garyfallidis, E.: Patch2self: Denoising difusion mri with self-supervised learning. Advances in neural information processing systems 33, 16293–16303 (2020)

13. Fourati, W., Kammoun, F., Bouhlel, M.S.: Medical image denoising using wavelet thresholding. J. Test. Eval. 33(5), 364–369 (2005)

14. Huang, T., Li, S., Jia, X., Lu, H., Liu, J.: Neighbor2neighbor: Self-supervised denoising from single noisy images. In: CVPR. pp. 14781–14790 (2021)

15. Jeon, S.Y., Wang, S., Wang, A.S., Gold, G.E., Choi, J.H.: Unsupervised training of a dynamic context-aware deep denoising framework for low-dose fluoroscopic imaging. IEEE Trans. Instrum. Meas. (2025)

16. Krull, A., Buchholz, T.O., Jug, F.: Noise2void-learning denoising from single noisy images. In: CVPR. pp. 2129–2137 (2019)

17. Lee, M.S., Park, S.W., Lee, S.Y., Kang, M.G.: Motion-adaptive 3d nonlocal means filter based on stochastic distance for low-dose x-ray fluoroscopy. Biomedical Signal Processing and Control 38, 74–85 (2017)

18. Lehtinen, J., Munkberg, J., Hasselgren, J., Laine, S., Karras, T., Aittala, M., Aila, T.: Noise2Noise: Learning image restoration without clean data. In: ICML. pp. 2971–2980 (2018)

19. Li, M., Hsu, W., Xie, X., Cong, J., Gao, W.: Sacnn: Self-attention convolutional neural network for low-dose ct denoising with self-supervised perceptual loss network. IEEE Trans. Med. Imaging 39(7), 2289–2301 (2020)

20. Luo, Y., Majoe, S., Kui, J., Qi, H., Pushparajah, K., Rhode, K.: Ultra-dense denoising network: application to cardiac catheter-based x-ray procedures. IEEE Trans. Biomed. Eng. 68(9), 2626–2636 (2020)

21. Ma, Q., Jiang, J., Zhou, X., Liang, P., Liu, X., Ma, J.: Pixel2Pixel: A pixelwise approach for zero-shot single image denoising. IEEE TPAMI 47(6), 4614–4629 (2025)

22. Manjon, J.V., Carbonell-Caballero, J., Lull, J.J., Garcia-Marti, G., Marti-Bonmati, L., Robles, M.: Mri denoising using non-local means. Med. Image Anal. 12(4), 514– 523 (2008)

23. Mansour, Y., Heckel, R.: Zero-Shot Noise2Noise: Eficient image denoising without any data. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 14018–14027 (2023)

24. McCollough, C.H., Bartley, A.C., Carter, R.E., Chen, B., Drees, T.A., Edwards, P., Fletcher, J.G.: Low-dose ct for the detection and classification of metastatic liver lesions: Results of the 2016 low dose ct grand challenge. Medical Physics 44(10), e339–e352 (2017)

25. Niu, C., Li, M., Fan, F., Wu, W., Guo, X., Lyu, Q., Wang, G.: Noise suppression with similarity-based self-supervised deep learning. IEEE Trans. Med. Imaging 42(6), 1590–1602 (2023)

26. Ravishankar, S., Bresler, Y.: Mr image reconstruction from highly undersampled k-space data by dictionary learning. IEEE Trans. Med. Imaging 30(5), 1028–1041 (2011)

27. Ronneberger, O., Fischer, P., Brox, T.: U-net: Convolutional networks for biomedical image segmentation. In: MICCAI. pp. 234–241 (2015)

28. Sagheer, S.V.M., George, S.N.: A review on medical image denoising algorithms. Biomedical Signal Processing and Control 61, 102036 (2020)

29. Siddiqui, M.S.R., Eliasof, M., Haber, E.: Graph flow matching: Enhancing image generation with neighbor-aware flow fields. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 40, pp. 25463–25471 (March 2026)

30. Sun, Y., Schneider, L.S., Mei, S., Wang, J., Hu, G., Gu, M., Ye, C., Wagner, F., Song, L., Bayer, S., Maier, A.: Filter2Noise: Interpretable self-supervised singleimage denoising for low-dose ct with attention-guided bilateral filtering. arXiv preprint arXiv:2504.13519 (2025)

31. Ulyanov, D., Vedaldi, A., Lempitsky, V.: Deep image prior. In: CVPR. pp. 9446– 9454 (2018)

32. Wang, Z., Bovik, A.C., Sheikh, H.R., Simoncelli, E.P.: Image quality assessment: From error visibility to structural similarity. IEEE TIP 13(4), 600–612 (2004)

33. Wang, Z., Liu, J., Li, G., Han, H.: Blind2unblind: Self-supervised image denoising with visible blind spots. In: CVPR. pp. 2027–2036 (2022)

34. Weigert, M., Schmidt, U., Boothe, T., Müller, A., Dibrov, A., Jain, A., Wilhelm, B., Schmidt, D., Broaddus, C., Culley, S., Rocha-Martins, M., Segovia-Miranda, F., Norden, C., Henriques, R., Zerial, M., Solimena, M., Rink, J., Tomancak, P., Royer, L., Jug, F., Myers, E.W.: Content-aware image restoration: pushing the limits of fluorescence microscopy. Nature Methods 15(12), 1090–1097 (2018)

35. Xie, Y., Wang, Z., Ji, S.: Noise2same: Optimizing a self-supervised bound for image denoising. In: Advances in Neural Information Processing Systems (NeurIPS). vol. 33, pp. 20320–20330 (2020)

36. Xu, J., Adalsteinsson, E.: Deformed2Self: Self-supervised denoising for dynamic medical imaging. In: MICCAI. pp. 25–35 (2021)

37. Xu, J., Huang, Y., Cheng, M.M., Liu, L., Zhu, F., Xu, Z., Shao, L.: Noisy-as-clean: Learning self-supervised denoising from corrupted image. IEEE TIP 29, 9316–9329 (2020)

38. Xue, W., Zhang, L., Mou, X., Bovik, A.C.: Gradient magnitude similarity deviation: A highly eficient perceptual image quality index. IEEE TIP 23(2), 684–695 (2014)

39. Zhang, K., Zuo, W., Chen, Y., Meng, D., Zhang, L.: Beyond a gaussian denoiser: Residual learning of deep cnn for image denoising. IEEE TIP 26(7), 3142–3155 (2017)

40. Zhang, L., Zhang, L., Mou, X., Zhang, D.: FSIM: A feature similarity index for image quality assessment. IEEE TIP 20(8), 2378–2386 (2011)

41. Zhou, L., Zhou, Z., Huang, X., Wang, H., Zhang, X., Li, G.: Neighboring Slice Noise2Noise: Self-supervised medical image denoising from single noisy image volume. arXiv preprint arXiv:2411.10831 (2024)

## S1 Detailed Implementation Details

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm S1 Neighbor-Guided Patch Sampling (NGPS)

Input: Noisy volume  $y \in R^{H \times W \times D}$ , Denoiser  $f_{\theta}$ 

Parameters: Mask threshold  $\tau$ , search window  $\Omega$ , ensemble size K, RC weight  $\lambda$ , learning rate  $\eta$ 

1: % Phase 1: Fast Guide Generation &amp; Masking

2:  $\tilde{y} \leftarrow \text{MF(BF}(y))$  {Edge-preserving filtering}

3: for each slice z and adjacent  $z' \in \{z - 1, z + 1\}$  do

4:  $\mathcal{M}_{z \to z'} \leftarrow \mathbf{1}\big(|\tilde{y}_z - \tilde{y}_{z'}| &gt; \tau\big)$  {1: Misaligned, 0: Static}

5: end for

6: % Phase 2: Decoupled Matching &amp; Retrieval

7: for each slice z and adjacent  $z'$  do

8: for each misaligned pixel p where  $\mathcal{M}_{z \to z'}(p) = 1$  do

9: {Match on guide  $\tilde{y}$ }

10:  $\{q^{(k)}\}_{k=1}^K \leftarrow \arg \text{Top-} K_{q \in \Omega_p} \| \mathcal{P}(\tilde{y}_z, p) - \mathcal{P}(\tilde{y}_{z'}, q) \|_2^2$ 

11: {Retrieve from raw y}

12:  $t_{z \to z'}(p) \leftarrow \frac{1}{K} \sum_{k=1}^K y_{z'}(q^{(k)})$ 

13: end for

14: end for

15: % Phase 3: Hybrid Objective Training

16: for  $e = 1, \ldots, 10$  do

17: for each mini-batch of slice triplets  $\{y_z, y_{z-1}, y_{z+1}\}$  do

18: Predict  $\hat{x}_s \leftarrow f_\theta(y_s), s \in \{z, z - 1, z + 1\}$ 

19: Calculate  $L_{N2N}, L_{NGPS}$ , and  $L_{RC}$ 

20: Update  $\theta \leftarrow \theta - \eta \nabla_\theta(\mathcal{L}_{N2N} + \mathcal{L}_{NGPS} + \lambda \mathcal{L}_{RC})$ 

21: end for

22: end for

23: return Optimized weights  $\theta^*$
</div>

To provide a clear, step-by-step mathematical overview of our proposed framework, Algorithm S1 summarizes the entire training pipeline. The algorithm is logically divided into three phases: edge-preserving guide generation, direction-aware target retrieval, and hybrid objective training. The specific hyperparameters utilized within this algorithm are detailed in the following subsections.

## S1.1 Network Architecture

Because our contribution concerns supervision construction rather than network architecture, we use NAFNet [9] as the denoising backbone. The base width is 32, with encoder blocks [2, 2, 4, 8], eight middle blocks, and decoder blocks [2, 2, 2, 2].

## S1.2 Optimization and Training Configuration

The framework was implemented using PyTorch and trained on a workstation equipped with an AMD Ryzen 9 9950X CPU and a single NVIDIA RTX 5090 GPU. The model was optimized using the AdamW optimizer with an initial learning rate of $2 \times 1 0 ^ { - 4 }$ and a weight decay of $1 0 ^ { - 5 }$ . During training, spatial dimensions were randomly cropped to 256 × 256 to serve as network inputs. We trained the network for 10 epochs with a batch size of 4 across all datasets. The regional-consistency weight was set to $\lambda _ { r c } = 0 . 5$ in all experiments.

## S1.3 NGPS Module and Guide Generation

We use $p = 7 ,$ $W = 1 5$ $K = 4 .$ and $\tau = 0 . 0 5$ as common defaults across the 7 evaluated datasets. Geometry-specific window sensitivity is reported in Table S7.

To compute the structural discrepancies and flag misaligned pixels, we generate a noise-attenuated guide volume using a sequential combination of a Bilateral filter and a Median filter. For the Bilateral filter, we set the spatial window size to $d = 5$ , color sigma to $\sigma _ { c o l o r } = 3 5$ , and space sigma to $\sigma _ { s p a c e } = 5 0$ . Specifically, to ensure a fair and direct comparison with the passive masking baseline, the kernel size of the Median filter was set to $5 \times 5$ . The $5 \times 5$ median kernel follows the NS-N2N configuration [41]; the preceding bilateral stage is the lightweight guide choice evaluated for NGPS. Following this lightweight filtering, the discrepancy masking threshold was fixed at $\tau = 0 . 0 5$

## S2 Additional Quantitative Results

## S2.1 Detailed Results on the IXI Dataset

In the main manuscript, we summarized the denoising performance on the IXI MRI dataset [4] under simulated Rician noise using bar-chart histograms to illustrate the robustness trends. For completeness and precise numerical comparison, Table S1 provides the quantitative measurements (PSNR and SSIM) for all evaluated methods across the noise intensity levels (5%, 7%, and 9%).

Across the evaluated 5%, 7%, and 9% synthetic Rician settings, NGPS attains the highest PSNR and SSIM, with Deformed2Self [36] ranking second overall. These results characterize robustness to the evaluated synthetic corruption model.

## S2.2 Robustness Analysis Across Random Seeds

Self-supervised and zero-shot methods can depend on random initialization and stochastic masking or sampling. We therefore repeat the LIDC-IDRI [2] evaluation with three seeds (Table S2). NGPS attains the strongest mean PSNR, SSIM, FSIM, and HFEN among the evaluated methods. Compared with NS-N2N [41], it also exhibits lower seed variance across all five metrics.

Table S1: Detailed quantitative evaluation on the IXI dataset [4]. The best results are highlighted in bold, and the second-best are underlined. ↑ indicates higher is better.

<table><tr><td rowspan="2">Method</td><td colspan="2">5% Noise</td><td colspan="2">7% Noise</td><td colspan="2">9% Noise</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>Baseline (Rician)</td><td>25.93</td><td>0.5276</td><td>22.71</td><td>0.4070</td><td>20.29</td><td>0.3255</td></tr><tr><td>BM3D [10]</td><td>28.77</td><td>0.8503</td><td>25.53</td><td>0.7789</td><td>23.05</td><td>0.7154</td></tr><tr><td>DIP [31]</td><td>29.51</td><td>0.8118</td><td>25.25</td><td>0.6390</td><td>22.15</td><td>0.4967</td></tr><tr><td>NAC [37]</td><td>29.39</td><td>0.8107</td><td>25.65</td><td>0.6568</td><td>23.13</td><td>0.5889</td></tr><tr><td>ZS-N2N [23]</td><td>29.20</td><td>0.8381</td><td>25.74</td><td>0.7464</td><td>23.21</td><td>0.6723</td></tr><tr><td>Pixel2Pixel [21]</td><td>29.18</td><td>0.8444</td><td>25.99</td><td>0.7634</td><td>23.48</td><td>0.6953</td></tr><tr><td>Noise2Void [16]</td><td>29.66</td><td>0.8609</td><td>25.86</td><td>0.7717</td><td>22.82</td><td>0.6903</td></tr><tr><td>NB2NB [14]</td><td>29.35</td><td>0.8400</td><td>26.42</td><td>0.7663</td><td>23.82</td><td>0.6939</td></tr><tr><td>Filter2Noise [30]</td><td>29.93</td><td>0.8502</td><td>26.53</td><td>0.7850</td><td>23.84</td><td>0.7086</td></tr><tr><td>Deformed2Self [36]</td><td>30.28</td><td>0.8831</td><td>26.73</td><td>0.8011</td><td>24.11</td><td>0.7392</td></tr><tr><td>Noise2Sim [25]</td><td>29.52</td><td>0.8579</td><td>25.06</td><td>0.7589</td><td>24.08</td><td>0.7200</td></tr><tr><td>NS-N2N [41]</td><td>29.86</td><td>0.8478</td><td>25.42</td><td>0.7215</td><td>23.74</td><td>0.6735</td></tr><tr><td>Ours</td><td>30.83</td><td>0.8879</td><td>27.04</td><td>0.8135</td><td>24.21</td><td>0.7459</td></tr></table>

## S2.3 Volume-Level Paired Confidence Intervals on LIDC-IDRI

To assess stability across test subjects, we compute paired metric diferences on the six LIDC-IDRI [2] test volumes. For each volume, the diference is oriented so that a positive value favors NGPS: ∆ = NGPS − NS-N2N for PSNR, SSIM, and FSIM, and $\varDelta = \mathrm { N S - N 2 N - N G P S }$ for HFEN and GMSD. The 95% confidence interval is ${ \bar { \varDelta } } \pm t _ { 0 . 9 7 5 , 5 } s _ { \varDelta } / { \sqrt { 6 } }$

The intervals exclude zero for PSNR and FSIM. The SSIM, HFEN, and GMSD intervals include zero; the mean GMSD marginally favors NS-N2N [41]. We therefore interpret the LIDC-IDRI results as stable PSNR/FSIM gains and a positive mean HFEN trend rather than uniform dominance across all metrics.

## S3 Additional Qualitative Results

## S3.1 Visual Comparisons on the IXI Dataset

Figure S1 compares the tested methods under 5%, 7%, and 9% synthetic Rician corruption. At the stronger corruption levels, several baselines retain more residual noise or produce softer boundaries in the highlighted regions. NGPS retains sharper local structures in these examples. These observations are interpreted together with the quantitative results.

Rician 5%  
BM3D  
Noisy-As-Clean  
Filter2Noise  
![](images/869c780881eb6423910e120a25f63c7c164412c956e2d420688388f63e17b3c0.jpg)  
Pixel2Pixel  
ZS-N2N  
(a) 5% Rician Noise

![](images/ad94a8022f4f5f2388fe1334fd091754996e9a324f7212e3432ff04190adb5c6.jpg)  
(b) 7% Rician Noise

![](images/d9180214e0e854422dc7762fe62465269ebbe2bbccf6f5b452166e6adc1eba67.jpg)  
(c) 9% Rician Noise  
Fig. S1: Qualitative comparisons on IXI [4] under 5%, 7%, and 9% synthetic Rician noise. At stronger corruption levels, several baselines retain more residual corruption or show softer boundaries in the highlighted regions, whereas NGPS retains sharper local structures in these examples.

Table S2: Robustness analysis of diferent denoising methods on the LIDC-IDRI dataset [2] across three random seeds. Results are reported as Mean ± Standard Deviation. BM3D is deterministic and thus has no variance. The best mean results are highlighted in bold, and the second-best are underlined.

<table><tr><td>Method</td><td>PSNR ↑</td><td>SSIM ↑</td><td>FSIM ↑</td><td>HFEN ↓</td><td>GMSD ↓</td></tr><tr><td>BM3D [10]</td><td>25.07</td><td>0.5825</td><td>0.7768</td><td>0.6010</td><td>0.1449</td></tr><tr><td>DIP [31]</td><td>26.77 ± 0.0390</td><td>0.6235 ± 0.0016</td><td>0.8074 ± 0.0022</td><td>0.5910 ± 0.0013</td><td>0.1337 ± 0.0005</td></tr><tr><td>NAC [37]</td><td>24.83 ± 0.0605</td><td>0.5690 ± 0.0082</td><td>0.7500 ± 0.0018</td><td>0.6096 ± 0.0039</td><td>0.1439 ± 0.0010</td></tr><tr><td>ZS-N2N [23]</td><td>26.63 ± 0.0129</td><td>0.5979 ± 0.0006</td><td>0.8080 ± 0.0006</td><td>0.5545 ± 0.0009</td><td>0.1259 ± 0.0003</td></tr><tr><td>Pixel2Pixel [21]</td><td>25.58 ± 0.0213</td><td>0.5078 ± 0.0017</td><td>0.7277 ± 0.0003</td><td>0.6196 ± 0.0005</td><td>0.1237 ± 0.0001</td></tr><tr><td>Noise2Void [16]</td><td>26.10 ± 0.2462</td><td>0.5995 ± 0.0767</td><td>0.8295 ± 0.0046</td><td>0.4948 ± 0.0089</td><td>0.1078 ± 0.0040</td></tr><tr><td>NB2NB [14]</td><td>27.23 ± 0.2045</td><td>0.6063 ± 0.0276</td><td>0.8321 ± 0.0039</td><td>0.4687 ± 0.0073</td><td>0.1003 ± 0.0030</td></tr><tr><td>Filter2Noise [30]</td><td>28.81 ± 0.1260</td><td>0.7181 ± 0.0045</td><td>0.8716 ± 0.0041</td><td>0.5188 ± 0.0208</td><td>0.1036 ± 0.0025</td></tr><tr><td>Deformed2Self [36]</td><td>28.94 ± 0.0171</td><td>0.6703 ± 0.0015</td><td>0.8170 ± 0.0005</td><td>0.5134 ± 0.0003</td><td>0.0808 ± 0.0001</td></tr><tr><td>Noise2Sim [25]</td><td>28.81 ± 0.0952</td><td>0.7817 ± 0.0046</td><td>0.8749 ± 0.0010</td><td>0.5003 ± 0.0008</td><td>0.1040 ± 0.0020</td></tr><tr><td>NS-N2N [41]</td><td>30.62 ± 0.1957</td><td>0.8080 ± 0.0078</td><td>0.8944 ± 0.0041</td><td>0.4406 ± 0.0260</td><td>0.0777 ± 0.0037</td></tr><tr><td>Ours</td><td>31.03 ± 0.0519</td><td>0.8102 ± 0.0027</td><td>0.9168 ± 0.0007</td><td>0.4161 ± 0.0016</td><td>0.0788 ± 0.0002</td></tr></table>

Table S3: Volume-level paired diferences between NGPS and NS-N2N [41] on the six LIDC-IDRI [2] test volumes. Positive ∆ favors NGPS.

<table><tr><td>Metric</td><td>Mean Δ</td><td>95% CI</td></tr><tr><td>PSNR ↑</td><td>+0.4100</td><td>[0.2257, 0.5943]</td></tr><tr><td>SSIM ↑</td><td>+0.0022</td><td>[-0.0050, 0.0094]</td></tr><tr><td>FSIM ↑</td><td>+0.0224</td><td>[0.0184, 0.0264]</td></tr><tr><td>HFEN ↓</td><td>+0.0245</td><td>[-0.0020, 0.0510]</td></tr><tr><td>GMSD ↓</td><td>-0.0011</td><td>[-0.0049, 0.0027]</td></tr></table>

## S4 Ablation Study

This section evaluates the loss components, PixelBank-style [21] counterfactuals, guide construction, spatial hyperparameters, and controlled failure regime of NGPS.

## S4.1 Loss-Component Ablation

Table S4 compares reconstruction, regional-consistency $( \mathcal { L } _ { R C } )$ , and inter-slicecontinuity $( \mathcal { L } _ { I C } )$ terms on AAPM [24]. For NS-N2N [41], $\mathcal { L } _ { I C }$ provides a surrogate constraint in regions omitted by the masked reconstruction loss. For NGPS, which supplies retrieved targets in these regions, adding $\mathcal { L } _ { I C }$ is associated with a 0.10 dB decrease.

## S4.2 PixelBank-Style Counterfactual

Table S5 compares NGPS with two simplified PixelBank-style [21] target constructions. The same-slice variant tests intra-image target retrieval, whereas the adjacent-slice raw-patch bank tests the direct extension of raw patch search across slices.

Table S4: Loss-component ablation on AAPM-Mayo [24]. We compare reconstruction, regional consistency (RC), and inter-slice continuity (IC). IC supplies a surrogate constraint for masked NS-N2N regions, whereas adding it to NGPS is associated with a 0.10 dB decrease.

<table><tr><td>Method</td><td colspan="3">Recon. RC (λ = 0.5) IC (λ = 1.0)</td><td colspan="2">PSNR (dB) ↑ SSIM ↑</td></tr><tr><td rowspan="4">NS-N2N [41]</td><td>✓</td><td></td><td></td><td>35.43</td><td>0.8481</td></tr><tr><td>✓</td><td>✓</td><td></td><td>35.88</td><td>0.8578</td></tr><tr><td>✓</td><td></td><td>✓</td><td>35.44</td><td>0.8473</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>35.91</td><td>0.8584</td></tr><tr><td rowspan="4">NGPS (Ours)</td><td>✓</td><td></td><td></td><td>36.50</td><td>0.8894</td></tr><tr><td>✓</td><td>✓</td><td></td><td>36.68</td><td>0.8986</td></tr><tr><td>✓</td><td></td><td>✓</td><td>36.39</td><td>0.8938</td></tr><tr><td>✓</td><td>✓</td><td>✓</td><td>36.58</td><td>0.8981</td></tr></table>

Table S5: PixelBank-style counterfactual training results. Entries are PSNR / SSIM.

<table><tr><td>Method</td><td>AAPM</td><td>LIDC 1.25 mm</td><td>LIDC 2.5 mm</td></tr><tr><td>Same-slice PixelBank</td><td>35.16 / 0.842</td><td>30.23 / 0.786</td><td>29.94 / 0.733</td></tr><tr><td>Adjacent-slice raw-patch bank</td><td>36.09 / 0.860</td><td>30.67 / 0.792</td><td>30.18 / 0.760</td></tr><tr><td>NGPS</td><td>36.68 / 0.899</td><td>31.08 / 0.810</td><td>30.92 / 0.799</td></tr></table>

NGPS outperforms the adjacent-slice raw-patch bank by 0.41–0.74 dB and also outperforms the same-slice variant on all three settings. These comparisons show that neither simplified alternative reproduces the performance of the full NGPS pipeline; they do not isolate the efect of the discrepancy mask or attribute the margin to any single component.

## S4.3 Why NS-N2N Uses $\scriptstyle { \mathcal { L } } _ { I C }$

In NS-N2N [41], misaligned regions are omitted from the masked reconstruction loss. The inter-slice continuity term therefore supplies a surrogate constraint by encouraging local linearity between averaged inputs and outputs:

$$
\mathcal {L} _ {I C} = \left\| f _ {\theta} \left(\frac {y _ {z} + y _ {z + 1}}{2}\right) - \frac {f _ {\theta} (y _ {z}) + f _ {\theta} (y _ {z + 1})}{2} \right\| _ {2} ^ {2}.\tag{S1}
$$

The term follows a first-order Taylor approximation under $f _ { \theta } ^ { \prime } ( y _ { z } ) \approx f _ { \theta } ^ { \prime } ( y _ { z + 1 } )$ [41]. Adding RC and IC to NS-N2N improves PSNR from 35.43 to 35.91 dB in Table S4.

Table S6: Ablation study on the guide generation filter. We evaluate the impact of diferent filtering strategies on both the AAPM-Mayo and LIDC-IDRI datasets [2,24]. Time is measured in seconds required to process a triplet of adjacent slices (3 slices) on a CPU. The best results are highlighted in bold, and the second-best are underlined.

<table><tr><td rowspan="2">Filter Type</td><td rowspan="2">Threshold (τ)</td><td colspan="2">AAPM-Mayo</td><td colspan="2">LIDC-IDRI</td><td rowspan="2">Time (3 slices, s) ↓</td></tr><tr><td>PSNR ↑</td><td>SSIM ↑</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>None (Raw Noisy)</td><td>0.05</td><td>35.27</td><td>0.8825</td><td>28.94</td><td>0.7355</td><td>-</td></tr><tr><td>Gaussian</td><td>0.05</td><td>36.59</td><td>0.8935</td><td>30.67</td><td>0.7835</td><td>0.0084</td></tr><tr><td>Median</td><td>0.05</td><td>36.43</td><td>0.8939</td><td>30.38</td><td>0.7619</td><td>0.1050</td></tr><tr><td>Bilateral</td><td>0.05</td><td>36.55</td><td>0.8962</td><td>30.97</td><td>0.8047</td><td>0.0498</td></tr><tr><td>Bilateral + Median (Ours)</td><td>0.05</td><td>36.68</td><td>0.8986</td><td>31.03</td><td>0.8102</td><td>0.1422</td></tr><tr><td>NLM</td><td>0.015</td><td>36.62</td><td>0.8996</td><td>31.03</td><td>0.8117</td><td>12.6936</td></tr><tr><td>NLM + Median [41]</td><td>0.015</td><td>36.74</td><td>0.8988</td><td>31.08</td><td>0.8097</td><td>12.7077</td></tr></table>

## S4.4 Interaction Between $\scriptstyle { \mathcal { L } } _ { I C }$ and NGPS

NGPS supplies retrieved targets in the regions omitted by masking. Its best tested configuration uses reconstruction and RC without IC (36.68 dB). Adding IC yields 36.58 dB, indicating empirical objective tension between the retrievedtarget reconstruction term and the local-linearity regularizer. Accordingly, the final NGPS objective uses RC but omits IC.

## S4.5 Impact of Guide Generation Filters

Table S6 compares guide filters and CPU guide-generation time for a threeslice triplet. NLM variants use $\tau = 0 . 0 1 5$ following NS-N2N [41], whereas the lightweight filters use $\tau = 0 . 0 5$ to accommodate their larger residual noise. The threshold sensitivity of NGPS is shown in Fig. 6a of the main paper.

The BF+MF guide provides a practical quality–eficiency trade-of among the tested filters. Gaussian filtering is faster and competitive, whereas the NLMbased [5] guides obtain similar or marginally higher scores but require substantially more preprocessing time under our implementation.

The triplet timings are standalone CPU measurements of guide filtering and are not directly comparable with the vectorized end-to-end 100-slice targetpreparation measurements in Table 4.

## S4.6 Sensitivity to Spatial Hyperparameters

Recovering displaced supervision depends on matching-patch and search-window sizes. Figure S2 reports the LIDC-IDRI [2] sweep. $\mathrm { ~ A ~ } 7 \times 7$ patch and $1 5 \times 1 5$ window are the best tested settings; performance remains within 0.18 dB across the evaluated ranges.

Spacing-Specific Search-Window Sensitivity To test whether the common search window is specific to thin slices, we vary only $W \in \{ 7 , 1 1 , 1 5 , 1 9 , 2 3 \}$ on LIDC-IDRI 1.25 and 2.5 mm while fixing $p = 7 , K = 4$ , and $\tau = 0 . 0 5$

![](images/0995f0c8fa303551b7a81003e9e1d657eb2687030ea7baf4cba5b70d00a09b5f.jpg)

![](images/51c468029eb2419fdbffdbe7f4df8eaac418bbe4be59282b1d51fcc8fdfc2ae3.jpg)  
(a) Patch-size ablation. $\mathrm { ~ A ~ } 7 \times 7$ patch is the best tested setting.

PSNR vs. Search Window Size  
![](images/e5dd3abacf3f9ed57472f2c556b5305934ff360a892ad611f01d1348e4764e23.jpg)

SSIM vs. Search Window Size  
![](images/37a96c5d2268fe23619b94cbbd567fb8fbee86cb49d68325db071b43ef297cc2.jpg)  
(b) Search-window ablation. A 15 × 15 window achieves the highest PSNR in this sweep.  
Fig. S2: Spatial-hyperparameter ablations on LIDC-IDRI [2]. Performance varies moderately over the tested patch- and search-window ranges; the selected defaults are marked in the two panels.

Table S7: Search-window sensitivity across LIDC-IDRI [2] slice spacings. Entries are PSNR / SSIM. Relative candidate cost is $W ^ { 2 } / 1 5 ^ { 2 }$

<table><tr><td colspan="2">W Relative cost</td><td>1.25 mm</td><td>2.5 mm</td></tr><tr><td>7</td><td>0.22×</td><td>31.01 / 0.8085</td><td>30.82 / 0.7938</td></tr><tr><td>11</td><td>0.54×</td><td>31.10 / 0.8118</td><td>30.88 / 0.7965</td></tr><tr><td>15</td><td>1.00×</td><td>31.08 / 0.8103</td><td>30.92 / 0.7994</td></tr><tr><td>19</td><td>1.60×</td><td>31.02 / 0.8062</td><td>30.85 / 0.7941</td></tr><tr><td>23</td><td>2.35×</td><td>30.93 / 0.8015</td><td>30.76 / 0.7892</td></tr></table>

The 15 × 15 default is within 0.02 dB of the best 1.25 mm result and is best at 2.5 mm. Larger windows increase candidate cost and reduce performance. We therefore treat $W = 1 5$ as a practical common default over the evaluated geometries, not a universal optimum.

## S4.7 Controlled Through-Plane Gap Stress

We retrain each method with farther supervisory slices $z \pm k$ , while keeping all hyperparameters fixed to their native-gap settings. The tested gaps are 1–5 mm for AAPM [24] and 1.25–6.25 mm for the LIDC-IDRI [2] 1.25 mm subset.

Optional calibrated match-cost gate (CMG). CMG is evaluated only in this stress test and is not part of the base NGPS method. It suppresses retrieved targets whose guide-patch matching cost is large relative to costs observed in nominally static regions. For clarity, we omit the direction index $z  z ^ { \prime }$ from $M , F , g ,$ , and t below.

Table S8: PSNR (dB) under increasing through-plane gap and optional CMG behavior. Bold indicates the best base method in each column. $\begin{array} { r } { \varDelta _ { \mathrm { C M G } } = \mathrm { P S N R } _ { N G P S + C M G } - } \end{array}$ $\mathrm { P S N R } _ { N G P S } ;$ rejection is the percentage of flagged targets assigned zero gate weight.

<table><tr><td rowspan="2">Method / quantity</td><td colspan="5">AAPM gap (mm)</td><td colspan="5">LIDC 1.25 mm gap (mm)</td></tr><tr><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>1.25</td><td>2.5</td><td>3.75</td><td>5.0</td><td>6.25</td></tr><tr><td>Same-coordinate N2N</td><td>35.62</td><td>34.96</td><td>33.94</td><td>31.72</td><td>30.55</td><td>30.45</td><td>29.78</td><td>27.80</td><td>27.15</td><td>25.36</td></tr><tr><td>Noise2Sim</td><td>35.49</td><td>35.38</td><td>34.96</td><td>33.82</td><td>33.27</td><td>30.53</td><td>30.22</td><td>29.72</td><td>28.30</td><td>27.42</td></tr><tr><td>NS-N2N</td><td>35.91</td><td>35.86</td><td>35.61</td><td>35.37</td><td>35.12</td><td>30.83</td><td>30.71</td><td>30.39</td><td>30.28</td><td>29.91</td></tr><tr><td>NGPS</td><td>36.68</td><td>36.51</td><td>36.10</td><td>35.29</td><td>34.48</td><td>31.08</td><td>30.96</td><td>30.34</td><td>29.52</td><td>28.68</td></tr><tr><td> $\Delta_{CMG}$  (dB)</td><td>-.06</td><td>+.02</td><td>+.12</td><td>+.27</td><td>+.72</td><td>-.06</td><td>-.02</td><td>+.22</td><td>+.60</td><td>+.96</td></tr><tr><td>CMG rejection (%)</td><td>17.5</td><td>18.2</td><td>23.4</td><td>31.8</td><td>35.2</td><td>38.3</td><td>43.6</td><td>48.2</td><td>58.7</td><td>61.4</td></tr></table>

For each location used for calibration or gating, we compute the normalized mean Top-K matching cost

$$
c (p) = \frac {1}{K s ^ {2}} \sum_ {j = 1} ^ {K} \left\| \mathcal {P} _ {s} (\tilde {y} _ {z}, p) - \mathcal {P} _ {s} (\tilde {y} _ {z ^ {\prime}}, q ^ {(j)}) \right\| _ {2} ^ {2}, \qquad s = 7, \quad K = 4,\tag{S2}
$$

where $q ^ { ( j ) }$ denotes the $j \mathrm { t h }$ selected match within the $1 5 \times 1 5$ search window.

For each slice direction, the calibration set C is the first set containing at least 128 pixels in

$$
\{M = 0, F = 1 \} \rightarrow \{M = 0 \} \rightarrow \{F = 1 \} \rightarrow \text { all   pixels },
$$

where

$$
M (p) = \mathbf {1} \left[ | \tilde {y} _ {z} (p) - \tilde {y} _ {z ^ {\prime}} (p) | > 0. 0 5 \right], \qquad F (p) = \mathbf {1} [ \tilde {y} _ {z} (p) > 0. 0 1 ].
$$

The direction-specific threshold is the 95th percentile of the selected calibration costs:

$$
\gamma = Q _ {0. 9 5} \big (\{c (p): p \in \mathcal {C} \} \big), \qquad g (p) = \mathbf {1} [ c (p) \leq \gamma ].
$$

Rejected targets are not replaced by another pseudo target and receive zero weight in the dynamic reconstruction term:

$$
\mathcal {L} _ {N G P S} ^ {C M G} = \frac {\sum_ {p} M (p) g (p) \left(f _ {\theta} (y _ {z}) (p) - t (p)\right) ^ {2}}{\sum_ {p} M (p) + \epsilon}.\tag{S3}
$$

The denominator remains the full flagged-pixel count, rather than the number of accepted targets. Thus, rejection reduces the contribution of uncertain dynamic targets instead of re-normalizing the loss over the accepted subset. The static N2N and regional-consistency terms remain unchanged.

Table S8 shows that NGPS leads the base methods at native and small gaps, but falls below NS-N2N [41] at 4–5 mm on AAPM and 3.75–6.25 mm on LIDC-IDRI, exposing the fixed-window local-homology limit. CMG is nearly neutral at native gaps but becomes more beneficial as the gap increases, reaching +0.72 and +0.96 dB at the largest AAPM and LIDC-IDRI gaps, respectively.