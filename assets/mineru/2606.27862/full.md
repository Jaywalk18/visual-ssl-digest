# ScaLe-INR: Scale and Learn Implicit Neural Representations

Buwaneka Epakanda, Athulya Ratnayake, Pandula Thennakoon, Mario De Silva & Avishka Ranasinghe {e19101,e19328,e18359,e19463,e18280}@eng.pdn.ac.lk

Roshan Godaliyadda & Parakrama Ekanayake roshang@eng.pdn.ac.lk & mpb.ekanayake@ee.pdn.ac.lk

## Abstract

Implicit Neural Representations (INRs) parameterized by multilayer perceptrons excel at modeling continuous signals. However, a key challenge persists as INRs fundamentally suffer from spectral bias and information cross-talk. When a single network attempts to capture multi-scale phenomena, high-frequency weight updates destructively interfere with the underlying low-frequency structural approximation. We introduce Scale and Learn INR (ScaLe-INR), a novel multi-branch architec ture that resolves these limitations by explicitly matching the signal’s frequency spectrum with the optimal operating region of the INR. Drawing upon the Fourier inverse scaling theorem we demonstrate that applying directional coordinate scal ing expands a network’s representational bandwidth along specific spatial axes. To mathematically enforce functional disentanglement and minimize task-specific information leakage between branches, we propose a Directional Edge Guidance Loss, a spatially-conditioned sparsity prior derived from ground-truth gradients. By constraining the high-frequency branches to act as strict, localized edge-filters, ScaLe-INR eliminates spectral cross-talk, accelerates convergence, and achieves high-fidelity signal reconstruction on complex multi-scale topologies. We eval uate ScaLe-INR across diverse reconstruction and inverse tasks, demonstrating substantial performance gains over existing state-of-the-art (SOTA) methods. The proposed architecture improves upon the nearest baselines by +5.16 dB in image reconstruction and +0.65 dB in image denoising. Furthermore, it achieve an impressive figure of 50.02 dB on audio reconstruction and 0.999 IOU(Intersection Over Union) on 3D reconstruction which beats the all SOTA models.

## 1 Introduction

Implicit Neural Representations (INRs) represent signals as continuous functions parameterised by neural networks, learning a mapping from input coordinates to signal values rather than storing signals explicitly on a discrete grid [Essakine et al., 2025, Jayasundara and Patel, 2026]. For images, this corresponds to mapping spatial coordinates to RGB intensities, while for audio, temporal coordinates are mapped to amplitudes. This continuous formulation enables resolution-independent signal modelling and has made INRs effective for image reconstruction, compression, super-resolution, novel view synthesis, and scientific signal representation.

Despite this flexibility, INRs exhibit spectral bias, where low-frequency components are learned more easily and earlier than high-frequency components [Rahaman et al., 2019]. In image representation, this often leads to accurate reconstruction of smooth colour variations and coarse structures, while sharp edges, textures, corners, and fine local details remain difficult to capture. Since these highfrequency components are essential for perceptual and structural fidelity, improving the ability of INRs to represent them remains a central challenge in continuous signal representation.

Existing approaches address this limitation by modifying the INR itself, for example through activations, positional encodings, architectural changes, or training strategies [Essakine et al., 2025]. These methods aim to expand the frequency range that the network can directly model. In this work, we take a complementary perspective: instead of only adapting the INR to the signal spectrum, we investigate whether the signal spectrum can be transformed into a range that is more accessible to the INR. Motivated by the Fourier inverse scaling property, where scaling a signal in the signal domain inversely scales its frequency spectrum, we propose to use coordinate scaling to align relevant spectral components with the effective operating range of the INR.

We first study this idea using one-dimensional signals, where scaling offers a controlled way to examine how spectral content can be shifted into a more learnable range. These experiments show that scaling is beneficial only when appropriately matched: moderate scaling improves the representation of useful frequencies, while excessive scaling can over-compress the spectrum and degrade signal structure. Extending this principle to images, we apply scaling along the x direction, the y direction, or both, leading to a multi-branch INR architecture in which each branch receives a differently scaled coordinate representation. The unscaled branch captures low-frequency image structure, while the scaled branches focus on directional and joint high-frequency variations, enabling richer reconstruction than a standalone INR.

Since different branches may be more reliable in different regions depending on local frequency content and structural complexity, we introduce a confidence-based fusion strategy. This allows the network to adaptively weight branch contributions at each spatial location rather than treating all branches equally. To further encourage meaningful specialization, we introduce an edge-guided regularization term that weakly guides high-frequency branches toward edges and fine details, while allowing smoother regions to be represented primarily by low-frequency components. This regularization is gradually introduced during training so that it supports, rather than dominates, the primary reconstruction objective.

The main contributions of this work can be summarized as follows.

1. We present a Fourier inverse scaling perspective for improving overall spectral representation in INRs. We show that signal-domain scaling can transform selected high-frequency components into a range that is more accessible to a SIREN-based INR, while also highlighting the need for controlled scaling.

2. We propose a multi-branch INR framework for frequency-specialized signal representation. The framework is motivated by the observation that different scaled versions of a signal can make different frequency components more learnable for the INR by matching the relevant signal spectral content to the optimal operating region of the INR. For higher dimensional signals, this leads to multiple branches, which capture and model such specialized frequency components in a complementary manner. We further generalize this formulation to Ndimensional signals.

3. We introduce confidence-based fusion and safe edge-guided regularization to support meaningful branch specialization. The fusion mechanism allows the model to adaptively combine the outputs of different frequency-specialized branches at a pixel level, enabling for detailed spectral information capture, that allows the proposed algorithm to significantly outperform the state-of-the-art.

## 2 Related Work

## 2.1 INRs and Spectral Bias

Spectral bias is widely recognized as a key limitation of implicit neural representations (INRs), where neural networks tend to learn smooth, low-frequency structures more easily than rapidly varying details. Early coordinate-based models employing ReLU activations [Hanin and Rolnick, 2019] demonstrated the feasibility of representing signals continuously, but their limited spectral expressivity reduced their ability to reconstruct fine textures and high-frequency components. This tendency was systematically studied by Rahaman et al. [2019], who showed that standard neural networks inherently favor low-frequency functions during optimization. To improve the frequency representation capability of INRs, several works introduced alternative activation mechanisms with richer spectral properties. Among these, SIREN [Sitzmann et al., 2020] employed periodic activations of the form $\phi ( x ) = \sin ( \omega _ { 0 } x )$ enabling improved modeling of oscillatory patterns and stable gradient propagation. Subsequent approaches explored localized [Saragadam et al., 2023] [Ramasinghe and Lucey, 2022] or adaptive activation [Kazerouni et al., 2024] designs to further enhance spectral coverage and reduce reconstruction artifacts. Following these different approaches, novel INRs are developed [Thennakoon et al., 2025], [Liu et al., 2024] , [Tancik et al., 2020] , [Serrano et al., 2024] , [Vemuri et al., 2025].These developments collectively suggest that the activation function plays a critical role in determining the spectral characteristics and representation capacity of INRs.

## 2.2 Multi-Scale and Frequency-Partitioned Architectures

To address the limitations of single-branch INRs on complex topographies, recent literature has explored multi-scale partitioning strategies. One dominant approach relies on spatial partitioning, where the signal domain is divided into discrete hierarchical grids or trees [Martel et al., 2021], [Müller et al., 2022]. While these methods achieve rapid convergence by storing local features in trainable hash tables or octrees, they fundamentally sacrifice the memory efficiency and infinite continuous differentiability of pure MLP-based INRs, often introducing boundary artifacts at grid intersections.

An alternative approach focuses on frequency partitioning within purely continuous networks. Lindell et al. [2022] introduces a band-limited coordinate network that analytically models the signal at distinct frequency scales, while [Saragadam et al., 2022]MINER routes coordinates through a Laplacian pyramid structure to capture multi-scale residuals. Further, parallel MLPs have been used in INR applications such as [Ashkenazi and Treister, 2024]

Furthermore, the effect of scaling on the input coordinates is used in [Sitzmann et al., 2020] for audio signals. However, the theoretical explanation of the scaling is not explored. Zheng et al. [2025] has explored different kernel transformations on the input coordinates and suggested that the linear transformation is better than non-linear kernels. However, that does not explain how the signal spectrum is changed for the learning process of INRs.

## 3 Methods

## 3.1 Fourier Scaling Perspective for INR Bandwidth Matching

INRs are only capable of modeling a limited range of frequency components, as shown by Rahaman et al. [2019]. Therefore, when the effective bandwidth of the signal exceeds the representational bandwidth of the INR, the model is unable to represent high frequency components.

Our major hypothesis is that this limitation can be reduced by transforming the high frequency components of the signal into a frequency range that is more compatible with the INR. If such components are scaled into the learnable bandwidth of the network, then the INR should be able to approximate them more effectively.

if we enforce a single MLP INR to learn the signal, this will lead to suboptimal representation of both ends of the frequency spectrum. This is because unlike CNNs that use localized filters, MLPs are global function approximators and every parameter in the network has some influence over the entire spatial domain. therefore, we deploy multiple parallel sub-MLPs that specialize on sub-frequency bands of the signal.

We use the duality property of the Fourier transform to control the spectral distribution of a signal through scaling in the signal domain. $\operatorname { L e t } x ( t )$ be a one dimensional signal with Fourier transform $X ( \omega )$ , such that $x ( t ) \longleftrightarrow X ( \omega )$ If we scale the signal in the time domain as $\textstyle x { \bigl ( } { \frac { t } { a } } { \bigr ) }$ , where $a \neq 0 .$ , its Fourier transform becomes $| a | X ( a \omega )$

This relationship shows that scaling in the signal domain produces an inverse scaling in the frequency domain. When the signal is stretched in the time domain, its frequency spectrum is compressed, causing higher frequency components to move towards the lower frequency region. Conversely, when the signal is compressed in the time domain, its frequency spectrum expands towards higher frequencies.

![](images/2f14420d671264afba4255bf0588105a938745c9165564832e566b06d134520c.jpg)  
Figure 1: Effect of signal scaling on SIREN reconstruction of a chirp signal. Moderate scaling improves the match between the reconstructed and ground-truth spectra by shifting high-frequency content into a more learnable range, while excessive scaling degrades reconstruction by over-compressing the spectrum.

![](images/b5cf00c48c456c0278d9c1cd35856bf197b8bc414e82766788b3614d5bcc361f.jpg)  
Figure 2: ScaLe-INR architecture for image-related tasks. There are 4 parallel branches (LL,HL,LH and HH) that scale in different directions.

This property provides a simple and effective mechanism for modifying the spectral content observed by an INR. In particular, we can use signal domain scaling to map high frequency components into a lower frequency range that lies within the effective learning bandwidth of the network. This directly supports our hypothesis, since the transformed signal becomes easier for the INR to approximate while still preserving information from the original high frequency content.

## 3.2 One-Dimensional Analysis of Coordinate Scaling

To examine frequency compression, we train a standard SIREN-based INR to reconstruct a onedimensional chirp signal, whose time-varying frequency content makes it useful for evaluating reconstruction across different spectral regions. As shown in Fig. 1, we apply different scaling factors and compare the ground truth and reconstructed signals in both the time and frequency domains.

Fig. 1 shows that increasing the scaling factor initially improves the reconstruction of high-frequency components, both in the time domain and in the corresponding frequency spectrum. However, this improvement saturates and eventually degrades when the scaling factor becomes too large, as excessive stretching over-compresses the spectrum and makes important spectral structure less distinguishable. These results support our hypothesis that scaling can make high-frequency content more accessible to the INR, while also demonstrating the need for controlled scaling before extending the approach to higher-dimensional signals such as images.

## 3.3 Scale-Conditioned Multi-Branch INR Architecture

Let an image be represented as a continuous function $I ( x , y ) \in \mathbb { R } ^ { 3 }$ , where (x, y) denotes the spatial coordinate and $I ( x , y )$ gives the RGB value at that coordinate. Since an image has frequency content along both horizontal and vertical directions, we use a four-branch architecture to separately model different frequency regions. The four branches are denoted by LL, HL, LH, and HH. Here L denotes a low frequency component and H denotes a high frequency component.The branches can be interpreted as follows:

• LL: low frequency in both x and y

• HL:high frequency in x and low frequency in y

• LH:low frequency in x and high frequency in y

• HH:high frequency in both x and y

The justification for the existence of the branch that attempts to capture simultaneous high frequency variations along both x and y directions, in spite there being separate branches over the x and y directions separately is to leverage the non-linear behavior of the INR to optimally capture edge information that is resultant of simultaneous variation along x and y directions.

$$
f _ {\theta} (x \odot y) \neq f _ {\theta} (x) \odot f _ {\theta} (y)\tag{1}
$$

The inherent non-linearity of the INR as would be discussed in Section 4 enables more intricate edge information to be captured by the HH branch that are not obtainable through a linear operation of the LH and HL branches.

The LL branch is expected to capture slowly varying image information, such as global colour structure, smooth intensity variations, and coarse textures. The HL and LH branches capture directional high frequency information along the horizontal and vertical spatial directions, respectively. The HH branch captures components that vary rapidly along both directions, such as corners, fine structures, and highly localized details.

For each spatial coordinate (x, y) each branch produces two outputs: an RGB prediction and a confidence logit. Let the RGB prediction of branch b be $I ( x , y ) \in \mathring { \mathbb { R } } ^ { 3 }$ , and let its confidence logit be $c _ { b } ( x , y )$ , where $b \in \{ L L , H L , ^ { * } L H , H H \}$

The confidence logits from all branches are concatenated and passed through a softmax operation to obtain normalized confidence weights:

$$
P _ {b} (x, y) = \frac {\exp (c _ {b} (x , y))}{\sum_ {j \in \{L L , H L , L H , H H \}} \exp (c _ {j} (x , y))}\tag{2}
$$

This allows the model to assign a spatially varying contribution to each branch. In regions where low frequency information is dominant, the LL branch can receive a larger weight. In regions containing edges, fine textures, or directional variations, the high frequency branches can contribute more strongly.

We define the contribution of each branch as the product of its RGB prediction and its confidence weight:, and the final reconstructed RGB value is then obtained by adding the confidence weighted contributions from all branches:

$$
\hat {\boldsymbol {I}} (x, y) = \sum_ {b \in \{L L, H L, L H, H H \}} P _ {b} (x, y) \hat {\boldsymbol {I}} _ {b} (x, y)\tag{3}
$$

We refer to this mechanism as confidence based fusion. Instead of treating all branch outputs equally, the model learns how much each branch should contribute at each pixel location. As a result, the architecture combines frequency specialized representations while allowing the final prediction to be adaptively formed according to the local structure of the image. The overall architecture is given in Fig. 2

## 3.4 Training Objective and Edge-Guided Regularisation

We train the proposed model using a reconstruction objective together with a safe edge-guidance term. The total loss used in the main mini-batch loop is defined as

$$
\mathcal {L} _ {\mathrm{Total}} = \mathcal {L} _ {\mathrm{recon}} + \mathcal {L} _ {\mathrm{edge,safe}}\tag{4}
$$

where $\mathcal { L } _ { \mathrm { e d g e , s a f e } } = \mathrm { s a f e } \left( \lambda _ { \mathrm { e d g e } } ( t ) \mathcal { L } _ { \mathrm { e d g e } } \right)$

Here, $\mathcal { L } _ { \mathrm { r e c o n } }$ denotes the reconstruction loss, and $\mathcal { L } _ { \mathrm { e d g e } }$ denotes the directional edge-guidance loss. The scalar $\lambda _ { \mathrm { e d g e } } ( t )$ is a scheduled weighting factor that controls the strength of the edge-guidance term at training step t. The function safe(·) limits the weighted edge term so that it remains an auxiliary regularizer and does not dominate the reconstruction objective.

The reconstruction loss is defined as

$$
\mathcal {L} _ {\mathrm{recon}} = \mathbb {E} _ {(x, y) \sim \mathcal {U} (\Omega_ {d})} \left[ \left\| \hat {\boldsymbol {I}} (x, y) - \boldsymbol {I} (x, y) \right\| _ {2} ^ {2} \right]\tag{5}
$$

where $I ( x , y ) \in \mathbb { R } ^ { 3 }$ is the ground-truth RGB value and $\pmb { \hat { I } } ( x , y ) \in \mathbb { R } ^ { 3 }$ is the predicted RGB value at coordinate $( x , y )$ . The expectation is taken over coordinates sampled uniformly from the discrete image domain $\Omega _ { d } .$ , and in practice is estimated using the sampled mini-batch of pixel coordinates. Since reconstruction quality is evaluated using PSNR, which is a monotonic function of the mean squared error, minimizing $\mathcal { L } _ { \mathrm { r e c o n } }$ is directly aligned with the main optimization goal.

The edge-guidance loss is introduced to encourage meaningful branch specialization. The highfrequency branches should contribute more strongly near image structures such as edges and fine details, while their influence should be reduced in smooth regions. Therefore, we penalize the high-frequency branch contributions in regions where the corresponding directional edge masks are weak:

$$
\mathcal {L} _ {\mathrm{edge}} = \mathbb {E} \left[ (1 - M _ {x}) | C _ {H L} | \right] + \mathbb {E} \left[ (1 - M _ {y}) | C _ {L H} | \right] + \mathbb {E} \left[ (1 - M _ {a l l}) | C _ {H H} | \right] + \beta_ {\mathrm{conf}} \mathcal {L} _ {\mathrm{conf}}\tag{6}
$$

Here, $M _ { x } , M _ { y } ,$ , and $M _ { a l l }$ are the directional edge masks computed from Sobel gradients of the ground truth. The terms $C _ { H L } , C _ { L H }$ , and $C _ { H H }$ denote the confidence-weighted contributions of the corresponding high-frequency branches:

$$
C _ {b} (x, y) = P _ {b} (x, y) \hat {\boldsymbol {I}} _ {b} (x, y), \qquad b \in \{H L, L H, H H \}\tag{7}
$$

This means that the edge loss is applied to the actual contribution of each branch to the final reconstruction, rather than to the raw branch output alone. This is consistent with the confidence fusion mechanism, since a branch affects the final prediction only through its weighted contribution.

The confidence regularization term is given by

$$
\mathcal {L} _ {\text { conf }} = \mathbb {E} \left[ (1 - M _ {x}) P _ {H L} \right] + \mathbb {E} \left[ (1 - M _ {y}) P _ {L H} \right] + \mathbb {E} \left[ (1 - M _ {a l l}) P _ {H H} \right],\tag{8}
$$

where $\beta _ { \mathrm { c o n f } }$ controls the strength of this regularization. This term gently discourages the model from assigning high confidence to high-frequency branches in regions where strong edge structures are not present.

$\mathcal { L } _ { \mathrm { e d g e } }$ provides a mechanism for encouraging pixel-level branch specialization by aligning each branch’s confidence with the local spectral structure of the signal. In particular, it promotes higher confidence for branches whose frequency specialization matches the dominant spatial variation at a given location, while discouraging branches from contributing in regions outside their intended domain of specialization. As a result, the confidence-based fusion mechanism can adaptively assign greater weight to the most relevant branch at each pixel, leading to a more structured and frequency aware reconstruction.

The scheduled weighting factor $\lambda _ { \mathrm { e d g e } } ( t )$ allows the model to first learn a stable image reconstruction before the edge-guidance term becomes active. After the scheduled warm-up period, the edge term is gradually introduced. To ensure that this auxiliary term does not overpower the reconstruction loss, the safe scaling operation constrains its effective contribution as $\lambda _ { \mathrm { e d g e } } ( t ) \mathcal { L } _ { \mathrm { e d g e } } \leq \alpha \mathcal { L } _ { \mathrm { r e c o n } }$ , where α is the edge cap ratio. Thus, the total objective preserves the reconstruction loss as the dominant term, while using the edge-guidance loss to organize the branch contributions in a structurally meaningful way.

The overall architecture generalized for an N-dimensional signal is given in the Appendix A

## 4 Experiments & Results

We implement all experiments in PyTorch and evaluate on an NVIDIA RTX 6000 Ada GPU with 48 GB memory using the Adam optimizer. All tasks share a unified pipeline except 3D occupancy and audio reconstruction due to different input output formats, where occupancy uses 3D coordinates to a single output and audio follows a one to one mapping.

Each model uses 3-layer MLPs with hidden dimension 256 and four parallel branches, except occupancy reconstruction which uses eight branches. The high frequency multiplier is set to 4.0 based on ablation studies and reduced for low frequency tasks. Fusion temperature starts at 1.0 to encourage exploration in confidence based softmax weighting and is reduced during training for more deterministic selection. The edge cap ratio is fixed at 0.05 to limit edge loss contribution to at most 5 percent of reconstruction loss.

## 4.1 Image Representation

Data. We conducted our experiments on the Kodak [Franzen] dataset, comprising of 24 lossless images. The images were trained at their native resolution. Reconstruction performance was evaluated using PSNR as a metric to quantify the error against the ground truth image.

Results. The performance of ScaLe-INR in comparison with state-of-the-art (SOTA) methods on the Kodak dataset is presented in Fig. 3(a) The results demonstrate that ScaLe-INR consistently achieves superior reconstruction performance, attaining an average PSNR of 46.4 dB and outperforming the nearest competing method, COSMO-INR [Thennakoon et al., 2025], by 5.16 dB.

![](images/59672033bc31e6c20957d89cab653391087d94a87433a5e0987a0a945fbbcd22.jpg)  
(a) PSNR plot for Kodak Images

![](images/41e568954c5b48183f597f29f995861938781d9e8ef48d4470a3b87abc5e83c0.jpg)

![](images/8488defb4cdba94beef70e7ae812906861a349b4e74e6090a64a662d69c7759f.jpg)  
Figure 3: ScaLe-INR performance analysis on Kodak image reconstruction

The performance statistics for each model are presented in Fig. 3(b). Furthermore, the loss curves shown in Fig. 3(c) demonstrate that ScaLe-INR converges significantly faster than the other models while maintaining stable optimization behavior throughout training.

## 4.2 Image Denoising

Data. To evaluate robustness against noisy input signals, we utilize the Parrot image from the DIV2K dataset [Timofte et al., 2018]. Photon noise is simulated by applying independent Poisson random variables to the ground truth pixel intensities, followed by an additional additive noise component to increase corruption complexity.

Implementation Details. The original image is downscaled by a factor of $1 / 2$ prior to training. Noise levels are configured using a photon noise factor of $4 \times 1 0 ^ { 1 }$ and an additive SNR noise component of 2, resulting in a degraded input with PSNR of 16.73 dB. Models are trained for 500 epochs using a batch size of $2 5 6 \times 2 5 6 .$ , a learning rate of $1 . 5 \times 1 0 ^ { - 4 }$ , and a decay factor of 0.1. The training requires approximately 3700 MB of GPU memory. In accordance with the duality property in Section 3, a frequency scaling factor of 0.5 is used.

Results. As shown in Fig. 4, ScaLe-INR achieves a PSNR of 30.90 dB, outperforming COSMO-RC while maintaining superior structural fidelity. Notably, ScaLe-INR preserves fine anatomical details in challenging regions such as the beak, where competing methods tend to oversmooth or introduce artifacts. This indicates improved robustness to high-frequency noise and better edge preservation under severe corruption.

![](images/f6ad03c9304e660ecc38ef539b565246c77081f674cc19556d746825060ec456.jpg)  
Figure 4: Qualitative denoising results comparing ScaLe-INR against existing approaches.

## 4.3 Image Super-Resolution

Data. We use a high-resolution image from the DIV2K dataset [Timofte et al., 2018] with resolution $1 3 5 6 \times 2 0 4 0 \times 3$ . Low-resolution inputs are generated using downsampling factors of $1 / 2 , 1 / 4$ , and 1/6, corresponding to 2×, 4×, and 6× super-resolution tasks.

Implementation Details. Models are trained for 500 epochs using a batch size of 256 × 256, learning rate $9 \times 1 0 ^ { - 4 }$ , and decay factor 0.1. The architecture requires approximately 2250 MB GPU memory. Following Section 3, frequency scaling factors of 0.3, 0.5, and 0.9 are used for 2×, 4×, and 6× settings respectively.

Results. Table 1 shows that ScaLe-INR consistently achieves the best performance across all scales. The largest gain is observed at 6×, where improved SSIM (0.85) indicates better structural preservation under severe downsampling. Furthermore, ScaLe-INR improves over COSMO-RC at all scales, with a notable gain of 0.40 dB at $2 \times$ , demonstrating the effectiveness of frequency-aware modeling.

<table><tr><td rowspan="2">Methods</td><td colspan="2">2×</td><td colspan="2">4×</td><td colspan="2">6×</td></tr><tr><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td><td>PSNR</td><td>SSIM</td></tr><tr><td>ReLU+PE</td><td>32.80</td><td>0.91</td><td>28.89</td><td>0.87</td><td>26.29</td><td>0.83</td></tr><tr><td>SIREN</td><td>32.26</td><td>0.90</td><td>29.62</td><td>0.87</td><td>27.31</td><td>0.81</td></tr><tr><td>INCODE</td><td>32.83</td><td>0.90</td><td>29.96</td><td>0.85</td><td>26.63</td><td>0.78</td></tr><tr><td>FINER</td><td>32.94</td><td>0.91</td><td>29.75</td><td>0.84</td><td>27.02</td><td>0.80</td></tr><tr><td>COSMO-RC</td><td>34.03</td><td>0.96</td><td>30.42</td><td>0.95</td><td>27.66</td><td>0.83</td></tr><tr><td>ScaLe-INR</td><td>34.43</td><td>0.97</td><td>30.48</td><td>0.95</td><td>27.68</td><td>0.85</td></tr></table>

Table 1: Comparison of super-resolution performance against existing SOTA methods.

## 4.4 3D Occupancy

Data. We use the Lucy dataset and convert meshes into a $5 1 2 ^ { 3 }$ occupancy grid with binary labels indicating occupied and empty voxels.

Implementation Details. Training is conducted for 200 epochs using $1 0 ^ { 6 }$ coordinate samples per batch, learning rate $1 \times 1 0 ^ { - 4 }$ , and decay factor 0.2. Due to volumetric complexity, training requires approximately 20 GB GPU memory. Eight frequency pipelines are used to model full {H, L} combinations across three spatial axes.

Results. ScaLe-INR achieves the highest IoU (0.999), outperforming COSMO-RC (0.995), FINER (0.994), INCODE (0.993), and SIREN (0.992). The results demonstrate improved geometric fidelity and sharper boundary reconstruction, confirming the effectiveness of frequency scaling for 3D structure modeling.

![](images/994bbdbe7083d548648f12bca1a1c4c4f902b12d491ff03e7ae956053d99992f.jpg)  
Figure 5: 3D occupancy reconstruction results obtained using ScaLe-INR.

## 4.5 Audio Reconstruction

Data. We use the first 7 seconds of Bach’s Cello Suite No. 1 to evaluate audio reconstruction performance in terms of PSNR.

Implementation Details. ScaLe-INR employs two parallel MLP branches modeling low and high frequency components. Frequency scaling is set to 100 for low-frequency and 400 for high-frequency components following prior work. The model is trained for 1000 epochs for fair comparison with INCODE.

Results. ScaLe-INR achieves the best performance with a PSNR of 50.02 dB, outperforming INCODE (49.10 dB), SIREN (37.92 dB), Gaussian features (38.50 dB), and ReLU+PE (22.99 dB). This demonstrates strong effectiveness in modeling high-fidelity temporal signals.

## 5 Conclusion

We presented Scale and Learn Implicit Neural Representations (ScaLe-INR), a novel multi-branch architecture designed to overcome the spectral bias and information cross-talk inherent in standard INRs. By conceptualizing coordinate networks as a multi-resolution continuous filter bank, we demonstrated that directional coordinate scaling inversely scale the representational bandwidth of the network along specific spatial axes. To enforce strict functional disentanglement, we introduced the Directional Edge Guidance Loss, which imposes a spatially-conditioned sparsity prior derived from ground-truth gradients. This structural prior ensures that the high-frequency branches act exclusively as localized directional edge filters, preventing them from corrupting the global low-frequency manifold. Our results confirm that by explicitly partitioning the signal’s frequency spectrum and guiding the network with edge guidance loss, ScaLe-INR effectively eliminates information cross-talk, significantly accelerates convergence, and achieves high-fidelity signal reconstruction on complex multi-scale topologies. We believe our work marks a major milestone in this research domain and future researchers will benefit from our findings. In future, we aim to extend this directional scaling paradigm to higher-dimensional tasks, including dynamic scene reconstruction and neural radiance fields.

## References

Maor Ashkenazi and Eran Treister. Towards croppable implicit neural representations, 2024. URL https://arxiv.org/abs/2409.19472.

Amer Essakine, Yanqi Cheng, Chun-Wun Cheng, Lipei Zhang, Zhongying Deng, Lei Zhu, Carola-Bibiane Schönlieb, and Angelica I Aviles-Rivero. Where do we stand with implicit neural representations? a technical and performance survey, 2025. URL https://arxiv.org/abs/ 2411.03688.

Richard W. Franzen. URL https://r0k.us/graphics/kodak/.

Boris Hanin and David Rolnick. Deep relu networks have surprisingly few activation patterns. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/file/ 9766527f2b5d3e95d4a733fcfb77bd7e-Paper.pdf.

Dhananjaya Jayasundara and Vishal M. Patel. Implicit neural representations: A signal processing perspective, 2026. URL https://arxiv.org/abs/2604.15047.

Amirhossein Kazerouni, Reza Azad, Alireza Hosseini, Dorit Merhof, and Ulas Bagci. Incode: Implicit neural conditioning with prior knowledge embeddings. In 2024 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 1287–1296, 2024. doi: 10.1109/WACV57701. 2024.00133.

David B. Lindell, Dave Van Veen, Jeong Joon Park, and Gordon Wetzstein. Bacon: Band-limited coordinate networks for multiscale scene representation, 2022. URL https://arxiv.org/abs/ 2112.04645.

Zhen Liu, Hao Zhu, Qi Zhang, Jingde Fu, Weibing Deng, Zhan Ma, Yanwen Guo, and Xun Cao. Finer: Flexible spectral-bias tuning in implicit neural representation by variableperiodic activation functions. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 2713–2722, 2024. doi: 10.1109/CVPR52733.2024.00262.

Julien N. P. Martel, David B. Lindell, Connor Z. Lin, Eric R. Chan, Marco Monteiro, and Gordon Wetzstein. Acorn: Adaptive coordinate networks for neural scene representation, 2021. URL https://arxiv.org/abs/2105.02788.

Thomas Müller, Alex Evans, Christoph Schied, and Alexander Keller. Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics, 41(4):1–15, 2022. ISSN 1557-7368. doi: 10.1145/3528223.3530127. URL http://dx.doi.org/10.1145/ 3528223.3530127.

Nasim Rahaman, Aristide Baratin, Devansh Arpit, Felix Draxler, Min Lin, Fred Hamprecht, Yoshua Bengio, and Aaron Courville. On the spectral bias of neural networks. In Kamalika Chaudhuri and Ruslan Salakhutdinov, editors, Proceedings of the 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, pages 5301–5310. PMLR, 09–15 Jun 2019. URL https://proceedings.mlr.press/v97/rahaman19a.html.

Sameera Ramasinghe and Simon Lucey. Beyond periodicity: Towards a unifying framework for acti vations in coordinate-mlps. In Shai Avidan, Gabriel Brostow, Moustapha Cissé, Giovanni Mariacosmo Farinella, and Tal Hassner, editors, Computer Vision – ECCV 2022, pages 142–158, Cham, 2022. Springer Nature Switzerland. ISBN 978-3-031-19827-4.

Vishwanath Saragadam, Jasper Tan, Guha Balakrishnan, Richard Baraniuk, and Ashok Veeraraghavan. Miner: Multiscale implicit neural representations. In European Conf. Computer Vision, 2022.

Vishwanath Saragadam, Daniel LeJeune, Jasper Tan, Guha Balakrishnan, Ashok Veeraraghavan, and Richard G. Baraniuk. Wire: Wavelet implicit neural representations. In 2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 18507–18516, 2023. doi: 10.1109/CVPR52729.2023.01775.

Danzel Serrano, Jakub Szymkowiak, and Przemyslaw Musialski. Hosc: A periodic activation function for preserving sharp features in implicit neural representations, 2024. URL https: //arxiv.org/abs/2401.10967.

Vincent Sitzmann, Julien Martel, Alexander Bergman, David Lindell, and Gordon Wetzstein. Implicit neural representations with periodic activation functions. In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin, editors, Advances in Neural Information Processing Systems, volume 33, pages 7462–7473. Curran Associates, Inc., 2020. URL https://proceedings.neurips.cc/ paper\_files/paper/2020/file/53c04118df112c13a8c34b38343b9c10-Paper.pdf.

Matthew Tancik, Pratul P. Srinivasan, Ben Mildenhall, Sara Fridovich-Keil, Nithin Raghavan, Utkarsh Singhal, Ravi Ramamoorthi, Jonathan T. Barron, and Ren Ng. Fourier features let networks learn high frequency functions in low dimensional domains. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS ’20, Red Hook, NY, USA, 2020. Curran Associates Inc. ISBN 9781713829546.

Pandula Thennakoon, Avishka Ranasinghe, Mario De Silva, Buwaneka Epakanda, Roshan Godaliyadda, Parakrama Ekanayake, and Vijitha Herath. Cosmo-inr: Complex sinusoidal modulation for implicit neural representations, 2025. URL https://arxiv.org/abs/2505.11640.

Radu Timofte, Shuhang Gu, Jiqing Wu, Luc Van Gool, Lei Zhang, Ming-Hsuan Yang, Muhammad Haris, et al. Ntire 2018 challenge on single image super-resolution: Methods and results. In The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, June 2018.

Sai Karthikeya Vemuri, Tim Büchner, and Joachim Denzler. F-inr: Functional tensor decomposition for implicit neural representations, 2025. URL https://arxiv.org/abs/2503.21507.

Sheng Zheng, Chaoning Zhang, Dongshen Han, Fachrina Dewi Puspitasari, Xinhong Hao, Yang Yang, and Heng Tao Shen. Exploring kernel transformations for implicit neural representations. IEEE Transactions on Multimedia, 27:5936–5945, 2025. doi: 10.1109/TMM.2025.3565979.

## A Generalisation of the Proposed Method to N-Dimensional Signals

This appendix shows that the multiresolution implicit neural representation proposed in Section 3 can be extended naturally from a two-dimensional image domain to a general N-dimensional signal domain. The construction preserves the core components of the proposed method: axis-wise frequency-scaled branches, confidence-weighted fusion, reconstruction supervision, directional edge guidance, and local gradient consistency.

## A.1 Two-Dimensional Formulation

In the main implementation, the input is a two-dimensional RGB signal

$$
s: \Omega \subset \mathbb {R} ^ {2} \longrightarrow \mathbb {R} ^ {3},\tag{9}
$$

where $\mathbf { x } = ( x _ { 1 } , x _ { 2 } ) \in \Omega$ denotes a spatial coordinate and $s ( \mathbf { x } ) \in \mathbb { R } ^ { 3 }$ denotes the RGB value at that coordinate.

The model learns an approximation

$$
\hat {s} _ {\theta} (\mathbf {x}) \approx s (\mathbf {x}).\tag{10}
$$

In the two-dimensional case, the branch set is $\{ L L , H L , L H , H H \}$ , where each symbol indicates whether the corresponding coordinate axis is treated as low-frequency or high-frequency.

## A.2 General N-Dimensional Signal Domain

Let

$$
s: \Omega \subset \mathbb {R} ^ {N} \longrightarrow \mathbb {R} ^ {c}\tag{11}
$$

be an N-dimensional signal with c channels. Here,

$$
\mathbf {x} = (x _ {1}, x _ {2}, \ldots , x _ {N}) \in \Omega\tag{12}
$$

denotes an N-dimensional coordinate, and $s ( \mathbf { x } ) \in \mathbb { R } ^ { c }$ denotes the corresponding signal value.

The implicit neural representation is written as

$$
\hat {s} _ {\theta}: \Omega \subset \mathbb {R} ^ {N} \longrightarrow \mathbb {R} ^ {c}.\tag{13}
$$

Although the neural network can be evaluated on $\mathbb { R } ^ { N }$ , it is trained and evaluated on the signal domain Ω or on a finite sampled subset of it.

## A.3 Generalised Branch Structure

For an N-dimensional signal, define the branch set as

$$
\mathcal {B} _ {N} = \{L, H \} ^ {N}.\tag{14}
$$

Thus, each branch

$$
b = \left(b _ {1}, b _ {2}, \dots , b _ {N}\right) \in \mathcal {B} _ {N}\tag{15}
$$

is an N -tuple with

$$
b _ {i} \in \{L, H \}, \qquad i = 1, 2, \dots , N.\tag{16}
$$

The two-dimensional case is recovered as

$$
\mathcal {B} _ {2} = \{L, H \} ^ {2} = \{L L, H L, L H, H H \}.\tag{17}
$$

Since each of the N coordinate axes has two possible frequency states, the number of branches is

$$
| \mathcal {B} _ {N} | = 2 ^ {N}.\tag{18}
$$

This establishes that the proposed branch construction extends directly from four branches in two dimensions to $2 ^ { N }$ branches in N dimensions.

## A.4 Axis-Wise Frequency Scaling

For each branch $b \in B _ { N }$ , define an axis-wise frequency-scaling vector

$$
\boldsymbol {\alpha} _ {b} = (\alpha_ {b _ {1}}, \alpha_ {b _ {2}}, \dots , \alpha_ {b _ {N}}),\tag{19}
$$

where

$$
\alpha_ {b _ {i}} \in \{\alpha_ {L}, \alpha_ {H} \}, \qquad 0 <   \alpha_ {L} <   \alpha_ {H}.\tag{20}
$$

Here, $\alpha _ { L }$ denotes the low-frequency coordinate scaling factor and $\alpha _ { H }$ denotes the high-frequency coordinate scaling factor. For example, in a three-dimensional signal,

$$
\boldsymbol {\alpha} _ {L H L} = (\alpha_ {L}, \alpha_ {H}, \alpha_ {L}).\tag{21}
$$

The input received by branch b is therefore

$$
\boldsymbol {\alpha} _ {b} \odot \mathbf {x},\tag{22}
$$

where ⊙ denotes element-wise multiplication. This axis-wise scaling allows different branches to specialise in different frequency patterns across the coordinate axes.

## A.5 Branch Output and Confidence Logit

Each branch $b \in B _ { N }$ is represented by a coordinate-based neural function

$$
f _ {b}: \mathbb {R} ^ {N} \longrightarrow \mathbb {R} ^ {c + 1}.\tag{23}
$$

For a coordinate x, the branch output is written as

$$
f _ {b} (\boldsymbol {\alpha} _ {b} \odot \mathbf {x}) = [ \mathbf {y} _ {b} (\mathbf {x}), c _ {b} (\mathbf {x}) ],\tag{24}
$$

where

$$
\mathbf {y} _ {b} (\mathbf {x}) \in \mathbb {R} ^ {c}\tag{25}
$$

is the branch-specific signal prediction, and

$$
c _ {b} (\mathbf {x}) \in \mathbb {R}\tag{26}
$$

is the corresponding confidence logit.

Thus, each branch produces both a candidate reconstruction and a scalar confidence logit for that coordinate.

## A.6 Confidence-Weighted Fusion

The confidence logits are normalised across all branches using a softmax function. For branch $b \in B _ { N }$ , the confidence weight is defined as

$$
P _ {b} (\mathbf {x}) = \frac {\exp (c _ {b} (\mathbf {x}) / \tau)}{\sum_ {b ^ {\prime} \in \mathcal {B} _ {N}} \exp (c _ {b ^ {\prime}} (\mathbf {x}) / \tau)},\tag{27}
$$

where $\tau > 0$ is the softmax temperature.

The weights satisfy

$$
P _ {b} (\mathbf {x}) \geq 0, \quad \sum_ {b \in \mathcal {B} _ {N}} P _ {b} (\mathbf {x}) = 1.\tag{28}
$$

The final reconstruction is then given by the convex combination

$$
\hat {s} _ {\theta} (\mathbf {x}) = \sum_ {b \in \mathcal {B} _ {N}} P _ {b} (\mathbf {x}) \mathbf {y} _ {b} (\mathbf {x}).\tag{29}
$$

Therefore, the model adaptively combines the outputs of all frequency-scaled branches at each coordinate.

## A.7 Generalised Loss Functions

## A.7.1 Reconstruction Loss

For a discretely sampled signal defined on a finite coordinate set

$$
\Omega_ {d} = \{\mathbf {x} _ {j} \} _ {j = 1} ^ {| \Omega_ {d} |} \subset \Omega ,\tag{30}
$$

the reconstruction loss is defined as

$$
\mathcal {L} _ {\mathrm{recon}} = \frac {1}{| \Omega_ {d} |} \sum_ {\mathbf {x} \in \Omega_ {d}} \| \hat {s} _ {\theta} (\mathbf {x}) - s (\mathbf {x}) \| _ {2} ^ {2}.\tag{31}
$$

This is the natural N-dimensional extension of the pixel-wise mean squared error used in the twodimensional implementation.

For a continuous signal domain, the analogous objective may be written as

$$
\mathcal {L} _ {\mathrm{recon}} = \frac {1}{| \Omega |} \int_ {\Omega} \| \hat {s} _ {\theta} (\mathbf {x}) - s (\mathbf {x}) \| _ {2} ^ {2} d \mathbf {x},\tag{32}
$$

provided that the integral is well defined.

## A.7.2 Directional Edge Masks

To extend the directional edge-guidance mechanism to N dimensions, we define one directional edge mask per coordinate axis. For a differentiable continuous signal, the edge strength along the i-th coordinate axis may be defined as

$$
M _ {i} (\mathbf {x}) \propto \left\| \frac {\partial s (\mathbf {x})}{\partial x _ {i}} \right\| _ {2}, \qquad i = 1, 2, \ldots , N.\tag{33}
$$

Since $s ( \mathbf { x } ) \in \mathbb { R } ^ { c }$ , the partial derivative $\partial s ( \mathbf { x } ) / \partial x _ { i }$ is channel-valued, and its Euclidean norm gives a scalar directional edge magnitude.

For discrete signals, these masks may be approximated using finite-difference operators, derivative filters, or Sobel-type filters when such filters are appropriate for the signal dimension. After computing the directional magnitudes, a normalisation and optional softening operation may be applied to obtain masks with values in [0, 1].

## A.7.3 Branch-Specific Edge Masks

For a branch $\boldsymbol { b } = ( b _ { 1 } , b _ { 2 } , \dots , b _ { N } )$ , define the set of high-frequency axes as

$$
H (b) = \{i \in \{1, 2, \dots , N \}: b _ {i} = H \}.\tag{34}
$$

For example,

$$
H (L H H) = \{2, 3 \}, \quad H (L L L) = \emptyset .\tag{35}
$$

For branches with $H ( b ) \neq \emptyset$ , define the branch-specific edge mask as

$$
M _ {b} (\mathbf {x}) = \text { Norm } \left(\sqrt {\sum_ {i \in H (b)} \left\| \frac {\partial s (\mathbf {x})}{\partial x _ {i}} \right\| _ {2} ^ {2}}\right),\tag{36}
$$

where Norm(·) denotes a normalisation operation that maps the edge magnitude to the interval [0, 1]. This definition assigns each high-frequency branch an edge mask corresponding to the directions in which that branch applies high-frequency coordinate scaling. The all-low-frequency branch, for which $H ( b ) = \varnothing$ , is excluded from the edge-guidance penalty.

## A.7.4 Directional Edge-Guidance Loss

The directional edge-guidance loss penalises high-frequency branch contributions in non-edge regions. For the N-dimensional case, it is defined as

$$
\mathcal{L}_{\text{edge}} = \sum_{\substack{b\in \mathcal{B}_{N}\\ H(b)\neq \emptyset}}\mathbb{E}_{\mathbf{x}\in \Omega_{d}}\left[ (1 - M_{b}(\mathbf{x}))\left\| P_{b}(\mathbf{x})\mathbf{y}_{b}(\mathbf{x})\right\|_{1}\right].\tag{37}
$$

The term $w _ { b } ( \mathbf { x } ) \mathbf { y } _ { b } ( \mathbf { x } )$ is the actual weighted contribution of branch b to the final reconstruction. Therefore, the loss penalizes high-frequency contributions only where the corresponding directional edge evidence is weak.

## A.7.5 Confidence Regularization

In addition to penalizing the high-frequency contribution itself, we may also discourage the model from assigning high confidence to high-frequency branches in non-edge regions. This gives the confidence regularization term

$$
\mathcal{L}_{\text{conf}} = \sum_{\substack{b\in \mathcal{B}_{N}\\ H(b)\neq \emptyset}}\mathbb{E}_{\mathbf{x}\in \Omega_{d}}\left[ \left(1 - M_{b}(\mathbf{x})\right)P_{b}(\mathbf{x})\right].\tag{38}
$$

The full directional edge-guidance loss is then

$$
\mathcal {L} _ {\mathrm{DEGL}} = \mathcal {L} _ {\text { edge }} + \beta_ {\text { conf }} \mathcal {L} _ {\text { conf }},\tag{39}
$$

where $\beta _ { \mathrm { c o n f } } \geq 0$ controls the strength of the confidence regularization term.

## A.7.6 Overall Objective

The full N -dimensional training objective is

$$
\mathcal {L} _ {\mathrm{total}} = \mathcal {L} _ {\mathrm{recon}} + \lambda_ {\mathrm{edge}} \mathcal {L} _ {\mathrm{DEGL}},\tag{40}
$$

where $\lambda _ { \mathrm { e d g e } } \ge 0$ are the weighting coefficients for the directional edge-guidance loss and the gradient-pair loss, respectively.

## A.8 Recovery of the Two-Dimensional Case

When $N = 2 .$ , the general branch set becomes

$$
\mathcal {B} _ {2} = \{L, H \} ^ {2} = \{L L, H L, L H, H H \}.\tag{41}
$$

The corresponding scaling vectors are

$$
\boldsymbol {\alpha} _ {L L} = (\alpha_ {L}, \alpha_ {L}),\tag{42}
$$

$$
\pmb {\alpha} _ {H L} = (\alpha_ {H}, \alpha_ {L}),\tag{43}
$$

$$
\pmb {\alpha} _ {L H} = (\alpha_ {L}, \alpha_ {H}),
$$

$$
\pmb {\alpha} _ {H H} = (\alpha_ {H}, \alpha_ {H}).\tag{44}
$$

(45)

These are precisely the four branch types used in the two-dimensional image formulation: one low-frequency approximation branch, two axis-oriented high-frequency branches, and one joint high-frequency branch. The confidence-weighted reconstruction reduces to

$$
\hat {s} _ {\theta} (\mathbf {x}) = P _ {L L} (\mathbf {x}) \mathbf {y} _ {L L} (\mathbf {x}) + P _ {H L} (\mathbf {x}) \mathbf {y} _ {H L} (\mathbf {x}) + P _ {L H} (\mathbf {x}) \mathbf {y} _ {L H} (\mathbf {x}) + P _ {H H} (\mathbf {x}) \mathbf {y} _ {H H} (\mathbf {x}).\tag{46}
$$

Thus, the proposed N-dimensional formulation is a direct extension of the implemented twodimensional model.

## B Remark on Computational Scalability

The construction above is mathematically direct, but the number of branches grows exponentially with the signal dimension, since $| B _ { N } | = \dot { 2 } { } ^ { N }$ . Therefore, while the formulation is valid for arbitrary $N ,$ practical implementations for high-dimensional signals may require branch sharing, sparse branch selection, grouped frequency patterns, or other parameter-efficient approximations.

<table><tr><td>Method</td><td>P(K)</td><td>GFLOPs</td><td>Train(s)</td><td>Infer(s)</td><td>Thpt</td><td>PSNR</td></tr><tr><td>SIREN</td><td>199</td><td>25.9</td><td>0.222</td><td>0.074</td><td>350</td><td>32.9</td></tr><tr><td>FINER</td><td>199</td><td>25.9</td><td>0.270</td><td>0.090</td><td>288</td><td>36.4</td></tr><tr><td>INCODE</td><td>437</td><td>38.7</td><td>0.435</td><td>0.145</td><td>267</td><td>36.2</td></tr><tr><td>WIRE</td><td>100</td><td>13.0</td><td>0.645</td><td>0.215</td><td>60</td><td>32.5</td></tr><tr><td>COSMO RC</td><td>437</td><td>38.7</td><td>3.500</td><td>1.100</td><td>33.2</td><td>45.1</td></tr><tr><td>FR INR</td><td>6294</td><td>29.2</td><td>0.229</td><td>0.082</td><td>356.1</td><td>36.2</td></tr><tr><td>ScaLe INR</td><td>534</td><td>69.5</td><td>0.820</td><td>0.291</td><td>238.8</td><td>46.8</td></tr></table>

Table 2: Comparison of super-resolution performance against existing SOTA methods.

## B.1 Performance and Efficiency Ablation Study

Explanation of metrics. P(K) is the number of parameters (in thousands), indicating model size. GFLOPs measures computational cost per forward pass. Train(s/it) and Infer(s/it) denote training and inference time per iteration, respectively. Thpt measures throughput in GFLOPs/s, reflecting efficiency. +PSNR evaluates reconstruction quality, where higher values indicate better fidelity.

Discussion. The results show a clear trade-off between efficiency and reconstruction quality. Lightweight models are computationally efficient but achieve lower PSNR, while heavier methods improve quality at higher cost. ScaLe INR achieves the best PSNR overall, outperforming all baselines while maintaining a reasonable efficiency balance compared to high-cost alternatives such as COSMO RC.

## C Image Inpainting

Data. We use the Celtic Spiral Knots image with resolution 572 × 582 × 3. A random mask is applied such that only 20% of pixel coordinates are observed during training. Evaluation is performed by reconstructing the full image grid.

Implementation Details. Training uses 500 epochs with batch size 256 × 256, learning rate 1.5 × 10<sup>−4</sup>, and decay factor 0.25. The model requires approximately 700 MB GPU memory. A frequency scaling factor of 0.5 is used as described in Section 3.

Results. ScaLe-INR achieves a reconstruction quality of 22.07 dB PSNR, outperforming COSMO-RC (21.88 dB), INCODE (21.85 dB), SIREN (20.69 dB), FINER (21.83 dB), and ReLU+PE (21.68 dB). This consistent improvement demonstrates stronger pixel-level recovery and better structural continuity under extreme sparsity.

![](images/19e880197e58d4c605a7929afd8ab548a9436f9c5eab8b14ec9c5a5d8c8d348d.jpg)  
Figure 6: Qualitative image inpainting results produced by ScaLe-INR.