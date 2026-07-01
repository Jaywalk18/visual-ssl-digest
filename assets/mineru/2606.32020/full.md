![](images/a81f57e329c6ea3056343e2ab31427b8b1c56bf738990662f2d540466c579d9c.jpg)  
Fig. 1: When Latents Don’t Match. (A) Existing distribution-based distillation methods rely on a Shared-Space constraint, assuming Teacher and Student share the same latent resolution and VAE. This prevents transferring knowledge from highresolution teachers (e.g., 1024<sup>2</sup>) to compact students $( e . g . , 5 1 2 ^ { 2 } )$ , as their latent tensors are inherently incompatible. (B) We formalize this setting as Cross-Space Distillation and introduce Bridge (B<sub>ϕ</sub>), a lightweight module that maps student latents z<sub>S</sub> to teacher-compatible latents $\hat { z } _ { T } = B _ { \phi } ( z _ { S } )$ , enabling standard one-step distillation under Cross-Resolution and Cross-VAE mismatch without modifying the student backbone.

# Cross-Space Distillation: Teaching One-Step Students with Modern Difusion Teachers

Anh Nguyen<sup>1\*</sup>, Ngan Nguyen<sup>1\*</sup>, Duc Vu<sup>1\*</sup> Trung Dao<sup>2</sup>, Viet Nguyen<sup>3</sup>, Quan Dao<sup>4</sup> Kien Nguyen<sup>1</sup>, Chi Tran<sup>1</sup>, Phong Nguyen<sup>1</sup>, Khoi Nguyen<sup>1</sup>, Cuong Pham<sup>1</sup> Dimitris Metaxas<sup>4</sup>, Vishal M. Patel<sup>3</sup>, and Anh Tran<sup>1</sup>

<sup>1</sup>Qualcomm AI Research<sup>†</sup> <sup>2</sup>University of Wisconsin–Madison <sup>3</sup>Johns Hopkins University <sup>4</sup>Rutgers University

Abstract. Modern one-step difusion models achieve impressive quality through distribution-based timestep distillation. Yet, they rely on a critical assumption: Teacher and Student must inhabit the same latent space. This Shared-Space constraint prevents knowledge transfer from modern high-capacity Teachers (e.g., SD 3.5 and Flux) into compact, deployment-friendly Students such as SD 1.5, whose latent resolution and VAE parameterization difer from the Teacher. We formalize this overlooked regime as Cross-Space Distillation, where Teacher and Student difer in both latent resolution and VAE space. To enable distillation under this mismatch, we introduce the Bridge (B<sub>ϕ</sub>), a lightweight latent interface that maps Student latents into the Teacher space without modifying the Student backbone. Bridge combines a frozen Student VAE decoder as a spatial prior with a compact learnable projector, and is trained with latent reconstruction and attention fidelity objectives for stable Teacher-space alignment. Across diverse modern Teachers, Bridge enables substantial gains for compact one-step Students; for example, it improves SD 1.5 from 5.4 to 9.4 HPSv3 while preserving one-step inference, low latency, and broad ecosystem compatibility. These results show that heterogeneous large Teachers can be distilled into eficient, deployable backbones through a lightweight latent-space interface.

Keywords: Difusion Distillation · One-Step Generation

## 1 Introduction

State-of-the-art text-to-image difusion models, such as Stable Difusion 3.5 [8] and Flux [16,17], achieve remarkable visual fidelity at high resolutions. However, their large backbones and multi-step sampling impose substantial computational and memory costs, limiting deployment on consumer hardware.

To reduce sampling overhead, recent works distill multi-step teachers into single-step generators via latent consistency distillation [19, 54, 62], variational score distillation [6, 31, 33], or adversarial distillation [39, 40]. Among these approaches, the latter two, which are both grounded in distribution-based distillation, have become central to modern one and few-step high-fidelity models. However, they generally preserve the teacher’s latent structure and assume a shared latent space, causing the distilled models to inherit large parameter counts or high latent resolutions of the teacher, which limits on-device practicality.

Another line of work compresses large generators through network pruning, structural compression, or lightweight architectures [1, 2, 13, 20, 61, 64]. These methods face three major challenges: pruning often hits a hard sparsity threshold beyond which quality drops, designing compact architectures requires extensive retraining and manual tuning specific to a teacher.

Rather than building new backbones from scratch, we propose reusing compact, widely supported models like Stable Difusion 1.5 and transferring the advanced generative quality of modern teachers via distillation. This approach is compact, easily integrates with existing ecosystems, is model-agnostic, and benefits from pretrained weights to stabilize and accelerate optimization.

Transferring knowledge to such students introduces four key challenges: Cross-Resolution (from a teacher at 1024×1024 to a student at 512×512), Cross-VAE (diferent latent spaces), Cross-Architecture (transformers vs. UNets), and Cross-Mechanism (flow matching vs. standard difusion). Empirically, the architecture and mechanism gaps are comparatively minor; the main barriers are resolution and latent-space mismatch, which prevent standard distillation objectives from being applied directly.

We address these two constraints via Cross-Space Distillation, introducing Bridge, a lightweight module that maps the student latent into the teacher latent space. This enables standard distillation objectives without modifying the student architecture. Bridge uses a compact single-stage design with the student’s VAE frozen for spatial alignment, incorporating an Attention Fidelity loss to preserve fine structure and high-quality reconstruction.

Cross-Space Distillation. We study distillation where the Teacher and Student do not share the same latent space:

1. Cross-Resolution: The latent grids have diferent spatial shapes, $e . g .$ $z _ { S } \in \mathbb { R } ^ { h \times w \times C _ { S } }$ and $z _ { T } \in \mathbb { R } ^ { H \times W \times C _ { T } }$ with $( h , w ) \neq ( H , W )$

2. Cross-VAE: The autoencoders difer, so latent spaces are not aligned between $\mathcal { Z } _ { \mathcal { S } }$ and $\mathcal { Z } _ { T }$

Consequence: Standard distillation objectives are not directly applicable between z<sub>S</sub> and $z _ { T }$ without an explicit alignment mapping.

This design preserves backbone compatibility while unlocking high-fidelity outputs. Experiments show that Bridge substantially improves SD 1.5 through Cross-Resolution and Cross-VAE distillation, efectively transferring the teacher’s visual priors. Beyond distillation, Bridge is modular, enabling seamless highresolution synthesis upgrades for existing architectures.

In short, our contributions are as follows:

1. Cross-Space Distillation. We formalize a new distillation setting where Teacher and Student difer simultaneously in latent resolution and VAE space, enabling knowledge transfer across mismatched latent spaces.

2. Bridge. We introduce the Bridge $( \boldsymbol { B } _ { \phi } )$ , a lightweight module that maps Student latents into the Teacher space. It combines (i) an architectural prior that leverages the Student’s spatial decoder as a scafold for a compact Teacherspace projector, and (ii) an Attention Fidelity objective preserving structural and fine-grained detail via a reverse-KL on attention distributions.

3. Practical Efectiveness. Experiments across modern teacher–compact student pairs show substantial improvements in generation quality, pushing compact students close to teacher-level performance while keeping the student backbone unchanged and significantly lighter than the teacher.

## 2 Preliminary

## 2.1 Difusion and Flow

Difusion models [12, 35, 45] and flow matching models [5, 23, 24, 47] provide closely related continuous-time formulations for generative modeling. Both define a forward corruption process that progressively transforms data into Gaussianlike noise and learn a reverse-time generative procedure that maps noise back to the data distribution. During inference, samples are generated by numerically solving the learned reverse dynamics with a discretized solver, typically requiring multiple network evaluations [15, 25, 44].

Difusion. Difusion models define a fixed forward Gaussian noising process that gradually perturbs clean samples $x _ { 0 } \sim p _ { \mathrm { d a t a } }$ over $T$ time steps, characterized by

$$
x _ {t} = \alpha_ {t} x _ {0} + \sigma_ {t} \epsilon , \qquad \epsilon \sim \mathcal {N} (0, I),\tag{1}
$$

where the coeficients $\alpha _ { t }$ and $\sigma _ { t }$ are predefined noise schedules [12]. A neural network, parametrized by $\theta ,$ is trained to approximate the reverse-time dynamics by gradually removing noise from $x _ { t }$ . Common parameterizations of the reverse process include ϵ-prediction, x<sub>0</sub>-prediction, and v-prediction [38].

Flow matching trains a continuous normalizing flow (CNF) by regressing the conditional velocity field of a chosen probability path [23]. A common Gaussian probability path uses linear interpolation between data and Gaussian noise,

$$
x _ {t} = (1 - t) x _ {0} + t \epsilon , \qquad t \in (0, 1).\tag{2}
$$

Instead of learning a reverse-time stochastic process, flow matching directly trains a neural network to predict the conditional velocity field $v _ { \theta } ( x _ { t } , t )$ that transports samples along this path [23]. Related formulations include rectified flow [24] and conditional/OT flow matching objectives [47].

Connection. Difusion and flow matching can be interpreted within a unified continuous-time view where a Gaussian probability path transports data samples to a Gaussian distribution [23,45]. At endpoints, diferent parameterizations are related under closed-form transformations at Gaussian endpoints, allowing flow matching velocities to be expressed in terms of difusion-model outputs [38, 43]. In our work, we convert outputs to a consistent x<sub>0</sub>-estimate in a shared latent space, enabling Teacher supervision after applying our alignment module.

## 2.2 One-Step Distribution-Based Distillation

To reduce inference latency to a single network function evaluation, one-step distillation trains a fast generator $\scriptstyle { \mathcal { S } } _ { \theta }$ to match the sample distribution of a multistep Teacher [56]. In this work, we focus on distribution-based objectives that align the Student distribution with a Teacher-implied target distribution, rather than trajectory-based distillation methods that supervise intermediate sampling states or enforce step-to-step consistency, such as consistency distillation [18,26].

Variational score distillation (VSD) is a distribution-based recipe that updates the Student by regressing the diference between Student and Teacher scores evaluated on the same noised sample [31, 50, 56–58]. Let $x = S _ { \theta } ( \xi , c )$ denote a sample generated by the student model $\scriptstyle { S _ { \theta } }$ conditioned on c from Gaussian noise $\xi .$ We denote $x _ { t }$ as noised sample of x at time t obtained by the forward difusion process with $\epsilon \sim \mathcal { N } ( 0 , I )$ . The generator update takes the form

$$
\nabla_ {\theta} \mathbb {E} _ {t, \xi , \epsilon} \left[ \left(s _ {\psi} (x _ {t}, t, c) - s _ {\eta} (x _ {t}, t, c)\right) \nabla_ {\theta} \mathcal {S} _ {\theta} (\xi , c) \right],\tag{3}
$$

where $s _ { \psi }$ is the frozen Teacher score model and $s _ { \eta }$ approximates the intractable Student score. In practice, $s _ { \eta }$ is commonly implemented by an auxiliary score network trained online on Student samples, with alternating updates between $\scriptstyle { S _ { \theta } }$ and the score estimator [56]. Besides, this distillation often operates entirely in latent space, assuming the Teacher and Student share the same space.

Adversarial distillation. Several works introduce a discriminator to align fewstep Student outputs with those of a multi-step Teacher by providing an additional distribution-level training signal [39, 41]. This objective is often combined with VSD-style distribution matching to further reduce the gap between a onestep Student and a multi-step Teacher [39, 56].

In our work. These objectives require Teacher and Student to have comparable representations, such as the input to the Teacher score model or any discriminator. Our method satisfies this under Cross-Resolution and Cross-VAE mismatch by aligning Student states into a Teacher-compatible latent space, enabling standard VSD and adversarial objectives with minimal modification.

## 3 Distillation Enabled by the Bridge

## 3.1 Cross-Space Distillation

One-step distribution-based distillation methods [6, 31, 33, 39, 40, 57, 58] reduce the temporal cost of difusion generation by decreasing the number of sampling steps. However, these methods typically assume that the teacher and student share the same latent space, thereby requiring supervision to be applied directly within that common representation.

This Shared-Space constraint becomes restrictive in practical large-scale distillation scenarios. Modern state-of-the-art teacher models often operate at higher latent resolution and channels and rely on diferent VAE architectures, whereas compact student models typically employ lower latent resolutions and distinct VAE designs. Under such representational mismatch, standard distillation objectives cannot be directly applied to the teacher and student latent variables without additional alignment.

We refer to this regime as Cross-Space Distillation, in which supervision must be conducted across latent spaces that difer in both spatial resolution and underlying VAE representations. Addressing these discrepancies requires learning an explicit alignment mapping to enable distillation between teacher and student representations.

## 3.2 Problem Setting and Notation

We denote the frozen Teacher denoiser by $\tau$ and the Student denoiser by ${ \mathcal { S } } .$ When referring to autoencoders, we use $( \mathcal { E } _ { T } , \mathcal { D } _ { T } )$ and $( \mathcal { E } _ { S } , \mathcal { D } _ { S } )$ for the Teacher and Student VAE encoder/decoder, respectively. In this paper, we use the term space to refer to a model’s latent representation space induced by its VAE parameterization and latent grid resolution.

$$
\hat {z} _ {T} = \mathcal {B} _ {\phi} (z _ {S}), \qquad \hat {z} _ {T} \in \mathcal {Z} _ {\mathcal {T}}.\tag{4}
$$

![](images/fac71ef8e55db5cd367750eb61bf482fc4be5c0448db186ad445e1852cece252.jpg)  
Fig. 2: Bridge for Cross-Space Distillation. Left. Bridge Training: given an image $x ,$ we encode it with both VAEs to obtain paired latents $( z _ { S } , z _ { T } )$ . The Bridge $B _ { \phi }$ maps $z _ { S }  \hat { z } _ { T }$ using a frozen spatial prior from a frozen prefix of the Student decoder ${ \mathcal { D } } _ { S } ^ { ( n ) }$ to expand the latent grid, followed by a learnable projector $g _ { \phi }$ that outputs a Teacher-compatible latent. We train $B _ { \phi }$ with latent reconstruction $\mathcal { L } _ { r e c }$ and attention fidelity $\mathcal { L } _ { a t t n }$ computed from a frozen Teacher denoiser by matching attention responses induced by $\hat { z } _ { T }$ and z<sub>T</sub> . Top-right. Distillation Enabled by Bridge: after training, $B _ { \phi }$ converts Student outputs into the Teacher latent space so standard onestep distillation losses such as $\angle { _ { V S D } }$ and $\mathcal { L } _ { G A N }$ could apply, bypassing Cross-Resolution and Cross-VAE mismatch. Bottom-right. Inference with Resolution Upgrade: a low-resolution Student sample can be mapped by $B _ { \phi }$ and decoded with the Teacher decoder to synthesize at the Teacher resolution without changing the Student denoiser.

A useful Bridge should meet three criteria: (1) High precision in reconstructing the Teacher latent $\hat { z } _ { T }$ , (2) Eficiency in memory and computation, adding negligible overhead to the Student network, and (3) Distillation-friendliness, enabling efective knowledge transfer using standard distribution-based distillation methods. To satisfy these requirements, we introduce new designs across Bridge architecture (Sec. 3.3), training objectives (Sec. 3.4), and describe how the trained Bridge is used for distillation and inference in Sec. 3.4.

## 3.3 Architecture with Rich Priors

The key challenge in distillation under latent mismatch is that $z _ { S }$ and $z _ { T }$ difer in spatial shape and latent parameterization. We therefore factor $B _ { \phi }$ into two stages. The first stage aligns spatial resolution from $h \times w$ to $H \times W$ . The second stage aligns features into the Teacher latent representation.

Spatial prior. For spatial alignment, one could learn a latent upsampler jointly with the Bridge, but this requires the model to learn spatial upsampling behavior from scratch. Instead, we reuse the early decoding blocks of the Student VAE decoder, which already implement latent upsampling and are trained with a massive amount of data. Let $\mathcal { D } _ { S } ^ { ( n ) }$ denote the first n decoding blocks of the Student VAE decoder. We keep them frozen and choose n such that their output has spatial resolution $H \times W$ , matching the Teacher’s latent resolution. Given

$z _ { S } \in \mathbb { R } ^ { h \times w \times C _ { S } }$ , we compute:

$$
f _ {\mathrm{prior}} = \mathcal {D} _ {S} ^ {(n)} (z _ {S}), \qquad f _ {\mathrm{prior}} \in \mathbb {R} ^ {H \times W \times C _ {\mathrm{prior}}},\tag{5}
$$

where $C _ { \mathrm { p r i o r } }$ is the channel dimension of this decoder feature. This stage performs the spatial expansion without introducing additional trainable parameters and provides a strong scafold for the next stage.

Projection head. After spatial alignment, $f _ { \mathrm { p r i o r } }$ has the correct spatial resolution but remains in the Student decoder feature space rather than the Teacher latent representation. We therefore learn a compact projection head $g _ { \phi }$ that maps $f _ { \mathrm { p r i o r } }$ to a Teacher compatible latent:

$$
\hat {z} _ {T} = \mathcal {B} _ {\phi} (z _ {S}) = g _ {\phi} (f _ {\mathrm{prior}}), \qquad \hat {z} _ {T} \in \mathbb {R} ^ {H \times W \times C _ {T}}.\tag{6}
$$

Overall, our Bridge is: $z _ { S } \xrightarrow { \mathcal { D } _ { S } ^ { ( n ) } } f _ { \mathrm { p r i o r } } \xrightarrow { g _ { \phi } } \hat { z } _ { T }$ , which isolates spatial alignment in the frozen decoder prefix while using the trainable module $g _ { \phi }$ for channel and semantic alignment. We optimize $g _ { \phi }$ with the objectives in Sec. 3.4.

## 3.4 Training Objectives

We optimize the Bridge using two complementary objectives: the first term provides a regression signal in the Teacher latent space, while the second term adds Teacher-based supervision by matching internal attention responses.

Latent Reconstruction $( \mathcal { L } _ { r e c } )$ . Given paired latents $( z _ { S } , z _ { T } )$ extracted from the Student and Teacher encoders, the Bridge maps the Student latent to the Teacher latent space as $\hat { z } _ { T } = B _ { \phi } ( z _ { S } )$ . We supervise the Bridge with an $\ell _ { 1 }$ reconstruction loss:

$$
\mathcal {L} _ {r e c} = \left\| z _ {T} - \hat {z} _ {T} \right\| _ {1}.\tag{7}
$$

While $\ell _ { 2 }$ is a common choice, we observe that it is often ineficient and less efective for latent-space reconstruction in practice, as noted in recent distillation/acceleration works [22,29,60]. In our setting, the $\ell _ { 1 }$ objective provides better training stability and helps preserve salient structure in the Teacher latent space.

Attention Fidelity $\left( \mathcal { L } _ { a t t r } \right)$ . While latent reconstruction $\mathcal { L } _ { r e c }$ is necessary for alignment, it is often insuficient: small latent discrepancies can produce perceptible semantic errors after decoding. Pixel-wise losses may tolerate slight misalignments, thus losing fine-grained structure. Adding adversarial losses can sharpen details but is sensitive to hyperparameters and reduces training stability. We therefore seek a supervision signal that is (i) stable, (ii) globally aware, and (iii) defined directly in Teacher space.

To this end, inspired by [27], we align the Teacher denoiser’s self-attention distributions. Attention captures long-range dependencies, so mismatches between $\hat { z } _ { T }$ and $z _ { T }$ are amplified in the Teacher’s internal responses; a compatible latent should induce similar attention patterns under identical difusion conditions. We extract attention maps $P ^ { l } ( z ; t , c ) \in [ 0 , 1 ] ^ { N \times N }$ from a fixed set of Teacher layers l ∈ Layers, where N is the number of tokens. Each $P ^ { l }$ is obtained via row-wise softmax over key positions from the Teacher’s query and key activations. Following MiniLLM [11], we use reverse KL divergence, which focuses on the dominant attention mass and is empirically more stable than forward KL. We minimize:

$$
\mathcal {L} _ {a t t n} = \sum_ {l \in \mathrm{Layers}} \tau^ {2} \mathrm{KL} \big (P ^ {l} (\hat {z} _ {T}; t, c) \parallel P ^ {l} (z _ {T}; t, c) \big),\tag{8}
$$

where $\tau > 0$ is a temperature for numerical stability, and we normalize attention logits before applying the softmax.

Final Objective $( \mathcal { L } _ { f i n a l } )$ . We train the Bridge by minimizing a weighted sum of latent reconstruction and attention fidelity:

$$
\mathcal {L} _ {f i n a l} = \alpha \mathcal {L} _ {r e c} + \beta \mathcal {L} _ {a t t n},\tag{9}
$$

where $\alpha , \beta$ are hyperparameters and set to $\alpha = \beta = 1$ in our experiments.

Bridge Usage. After training the Bridge, we attach the frozen module to the learnable Student to form an augmented Student that supports one-step distillation and one-step inference from the Teacher. This procedure follows the prior work summarized in Sec. 2.2 and is illustrated in Fig. 2, in the top-right and bottom-right panels.

## 4 Experiments

## 4.1 Experimental Setup

Student Models (S). We initialize the Student from one-step difusion variants: SD 1.5 backbone from DMD2 [57, 58] and SD 2.1 backbone from SiD-LSG [63]. These models serve as representative legacy generators operating at $5 1 2 \times 5 1 2$ resolution.

Teacher Models (T ). We evaluate our method under distillation settings using five state-of-the-art teachers operating at 1024×1024 resolution, including SDXL [36], Kolors [46], PixArt-Σ [3], FLUX.2 [klein] 4B [17], and SD 3.5 Medium [8]. These models span diferent backbone architectures (U-Net, DiT, and MMDiT), and employ diferent latent configurations with varying VAE designs and channel dimensions (4, 16, and 32 channels).

Bridge $( \boldsymbol { B } _ { \phi } )$ . Our Bridge module $B _ { \phi }$ is a lightweight alignment network with approximately 5M trainable parameters. As described in Sec. 3.3, $B _ { \phi }$ is factorized into a frozen spatial prior and a learnable projection head. The spatial prior is implemented by the first n decoding blocks of the Student VAE decoder, denoted $\mathcal { D } _ { S } ^ { ( n ) }$ , which expands the Student latent grid to the Teacher spatial resolution. In all experiments, we set n = 1 and freeze ${ \mathcal D } _ { S } ^ { ( 1 ) }$ . The learnable projection head $g _ { \phi }$ is implemented with SwinIR [21] and maps the intermediate decoder features to a Teacher-compatible latent $\hat { z } _ { T }$ . We train $B _ { \phi }$ with the objectives in Sec. 3.4. For

Table 1: Quantitative results. A: modern multi-step Teachers. B: compact onestep Students enabled by Cross-Space Distillation. Despite large gaps in resolution, VAE space, architecture, and generative mechanism, Bridge transfers knowledge from diverse modern Teachers into the same lightweight Student backbones while preserving one-step inference. In B, the second column lists the Teacher used for each individual distillation run; initialized and merged rows summarize the corresponding Student family. Paras are parameter counts in billions (B), and NFE denotes inference-time network evaluations. Pink marks initialized Students, pale green marks individual Bridge-distilled Students, and stronger green marks checkpoints obtained by averaging all five distilled Students in the same family. Across teacher families and metrics, Bridge consistently improves the initialized Students, and the merged checkpoints provide a simple post-training route to combine knowledge from multiple Teacher sources. Higher values are better for all metrics.

<table><tr><td colspan="2">Teacher</td><td rowspan="2">Paras (B)</td><td rowspan="2">NFE</td><td colspan="5">Metrics</td></tr><tr><td>Architect.</td><td>Model</td><td>HPSv3</td><td>HPSv2</td><td>IR</td><td>MPS</td><td>DPG</td></tr><tr><td colspan="9">A. Multi-Step Teachers</td></tr><tr><td rowspan="2">Large U-Net</td><td>SDXL [36]</td><td>2.57</td><td>50</td><td>9.25</td><td>28.36</td><td>0.66</td><td>13.72</td><td>74.00</td></tr><tr><td>Kolors [46]</td><td>2.57</td><td>50</td><td>10.59</td><td>30.94</td><td>0.88</td><td>14.12</td><td>76.52</td></tr><tr><td>DiT</td><td>PixArt- $\Sigma$ -1024 [3]</td><td>0.61</td><td>20</td><td>9.62</td><td>30.39</td><td>0.92</td><td>14.14</td><td>80.00</td></tr><tr><td rowspan="2">MM-DiT</td><td>FLUX.2-klein-4B [17]</td><td>4.00</td><td>50</td><td>10.05</td><td>28.90</td><td>0.80</td><td>14.00</td><td>83.20</td></tr><tr><td>SD 3.5 Medium [8]</td><td>2.50</td><td>50</td><td>10.86</td><td>30.02</td><td>0.96</td><td>14.13</td><td>84.50</td></tr><tr><td colspan="9">B. Cross-Space Distillation</td></tr><tr><td colspan="2">SD 1.5 Student, DMD2 [57] (init.)</td><td>0.86</td><td>1</td><td>5.37</td><td>21.90</td><td>-0.29</td><td>10.42</td><td>59.85</td></tr><tr><td rowspan="2">Large U-Net</td><td>SDXL [36]</td><td>0.86</td><td>1</td><td>9.04</td><td>27.37</td><td>0.30</td><td>11.93</td><td>63.00</td></tr><tr><td>Kolors [46]</td><td>0.86</td><td>1</td><td>9.33</td><td>27.38</td><td>0.34</td><td>11.66</td><td>64.46</td></tr><tr><td>DiT</td><td>PixArt- $\Sigma$ -1024 [3]</td><td>0.86</td><td>1</td><td>8.65</td><td>26.43</td><td>0.30</td><td>11.94</td><td>63.43</td></tr><tr><td rowspan="2">MM-DiT</td><td>FLUX.2-klein-4B [17]</td><td>0.86</td><td>1</td><td>9.49</td><td>26.64</td><td>0.40</td><td>12.35</td><td>64.06</td></tr><tr><td>SD 3.5 Medium [8]</td><td>0.86</td><td>1</td><td>9.42</td><td>28.30</td><td>0.62</td><td>12.96</td><td>65.75</td></tr><tr><td colspan="2">Merged (All 5 Teachers)</td><td>0.86</td><td>1</td><td>10.53</td><td>29.07</td><td>0.65</td><td>12.62</td><td>66.67</td></tr><tr><td colspan="2">SD 2.1 Student, SiD [63] (init.)</td><td>0.86</td><td>1</td><td>6.42</td><td>23.74</td><td>0.10</td><td>11.29</td><td>61.33</td></tr><tr><td rowspan="2">Large U-Net</td><td>SDXL [36]</td><td>0.86</td><td>1</td><td>8.40</td><td>26.73</td><td>0.23</td><td>11.88</td><td>67.03</td></tr><tr><td>Kolors [46]</td><td>0.86</td><td>1</td><td>8.52</td><td>26.57</td><td>0.20</td><td>11.91</td><td>68.34</td></tr><tr><td>DiT</td><td>PixArt- $\Sigma$ -1024 [3]</td><td>0.86</td><td>1</td><td>8.42</td><td>28.29</td><td>0.44</td><td>12.20</td><td>68.51</td></tr><tr><td rowspan="2">MM-DiT</td><td>FLUX.2-klein-4B [17]</td><td>0.86</td><td>1</td><td>8.74</td><td>28.31</td><td>0.33</td><td>12.03</td><td>66.00</td></tr><tr><td>SD 3.5 Medium [8]</td><td>0.86</td><td>1</td><td>8.64</td><td>29.11</td><td>0.45</td><td>12.92</td><td>68.52</td></tr><tr><td colspan="2">Merged (All 5 Teachers)</td><td>0.86</td><td>1</td><td>9.75</td><td>30.00</td><td>0.74</td><td>12.60</td><td>68.50</td></tr></table>

Attention Fidelity, we use noising level t = 1 and temperature τ = 3.0. Ablations on architecture and objectives are provided in the supplementary material.

Distillation Details. We follow the distillation protocol of DMD2 [57]. All models are trained for 20 hours on 8× NVIDIA H100 GPUs with 80GB memory using the AdamW optimizer. Additional implementation details are provided in the supplementary material.

Training Data. To maximize training eficiency and data accessibility, we avoid relying on large-scale real-image datasets. Instead, we construct a static synthetic dataset of approximately 2M images by sampling text prompts from JourneyDB and LAION [42] and generating images with the corresponding teacher model. We use this synthetic dataset to train $B _ { \phi }$ during space-bridge training. During distillation, we recycle the same dataset to compute an auxiliary adversarial objective, following the setup of DMD2.

Metrics. We evaluate Cross-Space Distillation using five widely adopted metrics: HPSv3 [28], HPSv2 [51], ImageReward [55], MPS [59], and DPG Bench [14]. HPSv3 and HPSv2 assess human preference alignment on 12,000 and 3,200 prompts, respectively, while ImageReward is computed on 100 prompts. We further report MPS on the HPSv2 and ImageReward prompt sets to measure multi-dimensional scores. DPG Bench evaluates text-image alignment on 1,065 prompts. For all metrics, higher scores indicate better performance.

## 4.2 Main Results

Cross-Architecture and Cross-Mechanism Distillation. As demonstrated in Tab. 1, our framework successfully facilitates distillation across architectures and difusion frameworks, such as from DiT-based, flow matching models (e.g., SD 3.5 and FLUX.2) to UNet-based, noise-prediction models $( e . g . , \mathrm { { S D ~ 1 . 5 } } )$ This versatility stems from two key design choices. First, our proposed Bridge explicitly neutralizes the Cross-Resolution and Cross-VAE constraints. Second, we adopt a mechanism-agnostic formulation by reparameterizing the student’s prediction back to $x _ { 0 }$ before sending it to the teacher, thereby eliminating discretization inconsistencies between the difusion and flow matching formulations. Furthermore, we deliberately avoid layer-wise matching; supervision is imposed only at the prediction level, thereby eliminating structural constraints between the DiT and UNet backbones. Notably, even without architecture-based alignment, our method still achieves strong performance, demonstrating that once resolution and VAE mismatches are properly addressed, efective Cross-Space distillation becomes not only feasible but robust.

Quantitative Results. Tab. 1 summarizes performance under five complementary metrics. Across all teacher models, Cross-Space Distillation improves the one-step Students on these metrics, suggesting that the gains are not confined to a single evaluator. For instance, using SD 3.5 Medium as the Teacher increases the SD 1.5 Student from 5.37 to 9.42 on HPSv3 and from −0.29 to 0.62 on ImageReward, with matching improvements on HPSv2, MPS, and DPG Bench. Overall, these results suggest higher perceived quality and stronger prompt adherence while preserving the same one-step inference budget.

Qualitative Results. We report qualitative examples in Fig. 3. Across a diverse set of prompts, our approach produces images with clearer textures, stronger structural consistency, and more faithful local details than the initialized baseline. In contrast, the baseline more frequently exhibits softness, broken fine structure, and occasional anatomical artifacts. These examples visually support the quantitative gains by showing that Bridge enabled distillation improves perceptual sharpness and coherence while preserving one-step generation.

![](images/77a6c1e07ddc347e8a97bae947a203609e3d720b87dc46b9406c97a6706de4af.jpg)  
Fig. 3: Qualitative Comparison of Cross-Space Distillation. We show samples from the initialized one-step student SD 1.5 DMD2 at 512×512 and from one-step students distilled with our method using diferent 1024×1024 teachers. Columns labeled +Teacher denote the student distilled from that teacher.

## 4.3 Post-Training Improvement via Model Merging

As an optional post-training refinement, we investigate model merging across students distilled from diferent teachers. This experiment is not part of the core Bridge design; rather, it serves as a practical extension that can further improve the final student without changing the model architecture, parameter count, or inference cost. Since all distilled checkpoints within each student family share the same backbone and parameterization, they can be combined through direct parameter averaging.

Given a set of distilled one-step student checkpoints $\mathcal { S } = \{ \theta _ { 1 } , \theta _ { 2 } , \dots , \theta _ { | S | } \}$ , we form a merged checkpoint by taking the element-wise mean of the parameters:

$$
\theta_ {\mathrm{merged}} = \frac {1}{| \mathcal {S} |} \sum_ {i = 1} ^ {| \mathcal {S} |} \theta_ {i}.\tag{10}
$$

Table 2: Pruning versus Bridge enabled distillation. We prune SD 3.5 Medium to 30% sparsity with OBS-Dif [64] and use it to initialize a one-step Student, then distill from the full SD 3.5 Medium Teacher using the same setup as Sec. 4.1. Bridge enabled distillation yields higher preference scores and cleaner details, while pruning based initialization more often produces blur and artifacts.

<table><tr><td>Setting</td><td>Params (B)</td><td>HPSv3</td><td>HPSv2</td><td>IR</td><td>MPS</td><td>DPG</td></tr><tr><td>Prune (30%)</td><td>1.5</td><td>2.76</td><td>21.94</td><td>-0.36</td><td>9.54</td><td>55.24</td></tr><tr><td>Ours</td><td>0.86</td><td>9.42</td><td>28.30</td><td>0.62</td><td>12.96</td><td>65.75</td></tr></table>

In our experiments, we merge the five students obtained by distilling from the five diferent teacher models in the main paper. This simple procedure consolidates multiple teacher-specific distilled checkpoints into a single student checkpoint while preserving the same inference-time NFE and model size.

Tab. 1 reports the merged results as highlighted rows under both SD 1.5 and SD 2.1 student families. Parameter averaging yields a stronger overall metric profile than the initialized student and is often competitive with, or better than, the individual distilled checkpoints on HPSv3, HPSv2, and ImageReward. For example, the merged SD 1.5 checkpoint improves HPSv3 from 5.37 to 10.53 and HPSv2 from 21.90 to 29.07 , while the merged SD 2.1 checkpoint improves HPSv3 from 6.42 to 9.75 and HPSv2 from 23.74 to 30.00. These results suggest that model merging can serve as a lightweight post-training enhancement for Cross-Space Distillation.

This model merging presents an unique benefit of our technique. Normally, it is hard to combine the image-generation capabilities of state-of-the-art models like Flux, SD 3.5, or Kolors, given their diferent network architectures and latent spaces. However, our technique distills them into student models with the same backbone and latent space, and combining knowledge across these student models is trivial. It provides a pathway to unify generation capabilities from multiple model sources. With more teacher models and/or stronger distillation techniques beyond DMD2, we expect an even stronger merged student model that can outperform every single teacher model.

## 4.4 Additional Analyses

Student Initialization. We compare two Student initializations: one-step SD 1.5 DMD2 checkpoint and standard multi-step SD 1.5 Teacher weights. With SD 3.5 Medium as the Teacher (Tab. 3), DMD2-init is slightly better in HPSv2, while SD 1.5 is slightly better on other metrics. This suggests initialization has minimal influence on performance in our framework. To maximize training speed and eficiency, we opt to initialize DMD2 as students.

Comparison with Network Pruning. To benchmark against parameter reduction, we compare the Bridge with network pruning. Using OBS-Dif [64], we prune SD 3.5 Medium to 30% sparsity (2.5B → 1.5B) and use it to initialize the one-step Student S, then distill from the full SD 3.5 Medium Teacher following Sec. 4. As shown in Tab. 2 and Fig. 4, with the same NFE=1 budget, our approach yields notably sharper outputs and stronger preferences, while pruned-model distillation tends to introduce blur and artifacts.

Table 3: Student initialization. We compare initializing the Student from a onestep SD 1.5 DMD2 checkpoint versus standard multi-step SD 1.5 weights, with both distilled from the same SD 3.5 Medium Teacher. Results are similar across metrics, so we use DMD2 initialization for subsequent experiments due to better training eficiency.

<table><tr><td>Setting</td><td>Iteration</td><td>HPSv3</td><td>HPSv2</td><td>IR</td><td>MPS</td><td>DPG</td></tr><tr><td>SD 1.5 Init.</td><td>9k</td><td>9.22</td><td>28.44</td><td>0.66</td><td>13.08</td><td>66.89</td></tr><tr><td>SD 1.5 DMD2 Init.</td><td>3k</td><td>9.42</td><td>28.30</td><td>0.62</td><td>12.96</td><td>65.75</td></tr></table>

![](images/45dc6f91910456c8bdb8c93410f3e4945eec43f2e5a9f4f42ed9002c014b6493.jpg)  
Fig. 4: Qualitative comparisons. (Left) Pruning vs. Bridge distillation. We compare a pruned baseline (1.5B) against our Bridge-enabled compact student (0.6B), generated with identical prompts and NFE = 1. (Right) Resolution upgrade at inference time. A 512×512 sample from the same one-step student is mapped by Bridge into the teacher latent space and decoded with the teacher decoder to produce a 1024×1024 output.

Resolution Upgrade with Bridge. Once trained, our Bridge $B _ { \phi }$ can also serve as a plug-and-play inference module that upgrades a low-resolution generator to higher-resolution outputs without retraining the generator itself. As shown in the right pane of Fig. 4, the Bridge synthesizes semantically consistent high-frequency details from low-resolution latents. This demonstrates that $B _ { \phi }$ generalizes to unseen inputs and provides a robust, cross-resolution mapping.

## 5 Related Work

Difusion and Flow. Difusion models have become the dominant paradigm for high-quality image generation. Early works such as DDPM [12] and scorebased generative models [45] introduced a stochastic forward corruption process and a learned reverse denoising process for generative modeling. Subsequent improvements have significantly accelerated sampling through more eficient solvers [15, 25, 44] and improved training formulations [38].

Recently, flow matching models [23,24,47] provide an alternative continuoustime formulation that directly learns the velocity field transporting samples from noise to data. Modern text-to-image systems such as SD 3.5 [8] and Flux [16,17] build on these advances with large-scale backbones and high-resolution latent representations, achieving strong visual fidelity and prompt alignment. However, these models typically require multiple network evaluations during inference, resulting in substantial computational overhead.

Distillation for Fast Difusion Sampling. To reduce inference cost, a large body of work distills multi-step difusion models into fast generators with significantly fewer sampling steps. Distribution-based distillation methods train a Student generator to match the sample distribution implied by a pretrained Teacher model. Representative approaches include variational score distillation and related methods [31,33,50,56–58], which use score diferences between Teacher and Student predictions as training signals. Other approaches explore consistencystyle distillation [18, 26] or trajectory-based methods that directly approximate multi-step sampling dynamics.

Several works further combine distribution matching with adversarial supervision to improve perceptual quality and reduce the gap between fast generators and their multi-step Teachers [39, 40]. While these approaches substantially reduce the number of inference steps, they typically assume that the Teacher and Student operate within the same latent representation and spatial resolution. Consequently, the distilled model often inherits the architectural scale or latent structure of the Teacher, limiting its suitability for lightweight deployment.

In this paper, we study Cross-Space Distillation, introducing a lightweight alignment module that maps Student representations into the Teacher latent space, enabling standard distillation objectives without modifying the Student architecture. This allows compact backbones such as Stable Difusion 1.5 [37] to inherit the generative capability of modern high-capacity models while maintaining ecosystem compatibility.

## 6 Conclusion

We identify a common but restrictive assumption in distribution-based one-step distillation: Teacher and Student are expected to share the same latent representation. We relax this assumption through Cross-Space Distillation and introduce Bridge $( \boldsymbol { B } _ { \phi } )$ , a lightweight latent interface that maps Student latents into the Teacher space, making standard one-step distillation objectives applicable under latent-resolution and VAE mismatch. Across modern Teachers and compact Students, Bridge yields strong preference gains while preserving onestep inference, the original Student backbone, and compatibility with widely used deployment ecosystems. Beyond training, Bridge can also be reused at inference time to map low-resolution Student samples onto the Teacher latent grid for higher-resolution decoding. We hope this work motivates broader exploration of distillation and model reuse across heterogeneous latent representations, and inspires future research on alignment interfaces that unlock knowledge transfer across mismatched latent spaces.

## References

1. Cai, F., Guo, Y., Li, J., Li, W., Chen, J., Fang, X.: Fastflux: Pruning flux with block-wise replacement and sandwich training. arXiv preprint arXiv:2506.10035 (2025)

2. Chen, J., Hu, D., Huang, X., Coskun, H., Sahni, A., Gupta, A., Goyal, A., Lahiri, D., Singh, R., Idelbayev, Y., et al.: Snapgen: Taming high-resolution text-to-image models for mobile devices with eficient architectures and training. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 7997–8008 (2025)

3. Chen, J., Ge, C., Xie, E., Wu, Y., Yao, L., Ren, X., Wang, Z., Luo, P., Lu, H., Li, Z.: Pixart-σ: Weak-to-strong training of difusion transformer for 4k text-toimage generation. In: European Conference on Computer Vision (ECCV) (2024), https://arxiv.org/abs/2403.04692

4. Chen, J., Xue, S., Zhao, Y., Yu, J., Paul, S., Chen, J., Cai, H., Han, S., Xie, E.: Sana-sprint: One-step difusion with continuous-time consistency distillation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 16185–16195 (2025)

5. Dao, Q., Metaxas, D.: Mpdit: Multi-patch global-to-local transformer architecture for eficient flow matching and difusion model. arXiv preprint arXiv:2603.26357 (2026)

6. Dao, T., Nguyen, T.H., Le, T., Vu, D., Nguyen, K., Pham, C., Tran, A.: Swiftbrush v2: Make your one-step difusion model better than its teacher. In: European Conference on Computer Vision. pp. 176–192. Springer (2024)

7. Dao, T.T., Vu, D.H., Pham, C., Tran, A.: Efhq: Multi-purpose extremepose-facehq dataset. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 22605–22615 (2024)

8. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning

9. Gandikota, R., Materzynska, J., Fiotto-Kaufman, J., Bau, D.: Erasing concepts from difusion models. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 2426–2436 (2023)

10. Gu, Y., Dong, L., Wei, F., Huang, M.: Minillm: Knowledge distillation of large language models. In: The Twelfth International Conference on Learning Representations

11. Gu, Y., Dong, L., Wei, F., Huang, M.: Minillm: Knowledge distillation of large language models. arXiv preprint arXiv:2306.08543 (2023)

12. Ho, J., Jain, A., Abbeel, P.: Denoising difusion probabilistic models. Advances in neural information processing systems 33, 6840–6851 (2020)

13. Hu, D., Gupta, A., Gabidolla, M., Sahni, A., Coskun, H., Li, Y., Idelbayev, Y., Mahmood, A., Lebedev, A., Lahiri, D., et al.: Snapgen++: Unleashing difusion transformers for eficient high-fidelity image generation on edge devices. arXiv preprint arXiv:2601.08303 (2026)

14. Hu, X., Wang, R., Fang, Y., et al.: Ella: Equip difusion models with llm for enhanced semantic alignment. arXiv preprint arXiv:2403.05135 (2024)

15. Karras, T., Aittala, M., Aila, T., Laine, S.: Elucidating the design space of difusionbased generative models. In: Advances in Neural Information Processing Systems (NeurIPS) (2022), https://arxiv.org/abs/2206.00364

16. Labs, B.F.: Announcing FLUX.1. https://blackforestlabs.ai/announcingflux-1 (2024), accessed: 2026-03-04

17. Labs, B.F.: FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2 (2025), accessed: 2026-03-04

18. Lee, S., Xu, Y., Gefner, T., Fanti, G., Kreis, K., Vahdat, A., Nie, W.: Truncated consistency models. In: International Conference on Learning Representations (ICLR) (2025), https : / / openreview . net / pdf / bb8f3dceac43037618899ff56c90995c5e08e978.pdf

19. Li, J., Feng, W., Chen, W., Wang, W.Y.: Reward guided latent consistency distillation. arXiv preprint arXiv:2403.11027 (2024). https://doi.org/10.48550/ arXiv.2403.11027

20. Li, Y., Wang, H., Jin, Q., Hu, J., Chemerys, P., Fu, Y., Wang, Y., Tulyakov, S., Ren, J.: Snapfusion: Text-to-image difusion model on mobile devices within two seconds. Advances in Neural Information Processing Systems 36, 20662–20678 (2023)

21. Liang, J., Cao, J., Sun, G., Zhang, K., Van Gool, L., Timofte, R.: Swinir: Image restoration using swin transformer. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1833–1844 (2021)

22. Lin, S., Wang, A., Yang, X.: Sdxl-lightning: Progressive adversarial difusion distillation. arXiv preprint arXiv:2402.13929 (2024). https://doi.org/10.48550/ arXiv.2402.13929, https://arxiv.org/abs/2402.13929

23. Lipman, Y., Chen, R.T.Q., Ben-Hamu, H., Nickel, M., Le, M.: Flow matching for generative modeling. In: International Conference on Learning Representations (ICLR) (2023), https://arxiv.org/abs/2210.02747

24. Liu, X., Gong, C., Liu, Q.: Flow straight and fast: Learning to generate and transfer data with rectified flow. arXiv preprint arXiv:2209.03003 (2022), https://arxiv. org/abs/2209.03003

25. Lu, C., Zhou, Y., Bao, F., Chen, J., Li, C., Zhu, J.: Dpm-solver: A fast ode solver for difusion probabilistic model sampling in around 10 steps. In: Advances in Neural Information Processing Systems (NeurIPS) (2022), https://arxiv.org/ abs/2206.00927

26. Luo, S., Tan, Y., Huang, L., Li, J., Zhao, H.: Latent consistency models: Synthesizing high-resolution images with few-step inference. arXiv preprint arXiv:2310.04378 (2023), https://arxiv.org/abs/2310.04378

27. Ma, J., Peng, Q., Guo, X., Chen, C., Lu, H., Yang, Z.: X2i: Seamless integration of multimodal understanding into difusion transformer via attention distillation. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 16733–16744 (October 2025)

28. Ma, Y., Wu, X., Sun, K., Li, H.: Hpsv3: Towards wide-spectrum human preference score. arXiv preprint arXiv:2508.03789 (2025)

29. Nguyen, A., Nguyen, V., Vu, D., Dao, T., Tran, C., Tran, T., Tran, A.: Improved training technique for shortcut models. arXiv preprint arXiv:2510.21250 (2025). https://doi.org/10.48550/arXiv.2510.21250, https://arxiv.org/abs/2510. 21250, accepted at NeurIPS 2025

30. Nguyen, K., Tran, A., Pham, C.: Suma: A subspace mapping approach for robust and efective concept erasure in text-to-image difusion models. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 19587–19596 (2025)

31. Nguyen, T.H., Tran, A.: Swiftbrush: One-step text-to-image difusion model with variational score distillation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 7807–7816 (2024)

32. Nguyen, T.T., Nguyen, D.A., Tran, A., Pham, C.: Flexedit: Flexible and controllable difusion-based object-centric image editing. arXiv preprint arXiv:2403.18605 (2024)

33. Nguyen, V., Nguyen, A., Dao, T., Nguyen, K., Pham, C., Tran, T., Tran, A.: Supercharged one-step text-to-image difusion models with negative prompts. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 18004–18013 (2025)

34. Nguyen, V., Patel, V.M.: Cgce: Classifier-guided concept erasure in generative models. arXiv preprint arXiv:2511.05865 (2025)

35. Nguyen, V., Vu, G., Thanh, T.N., Than, K., Tran, T.: On inference stability for difusion models. In: Proceedings of the AAAI Conference on Artificial Intelligence. vol. 38, pp. 14449–14456 (2024)

36. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent difusion models for high-resolution image synthesis. In: The Twelfth International Conference on Learning Representations (2024), https://openreview.net/forum?id=di52zR8xgf

37. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10684– 10695 (June 2022)

38. Salimans, T., Ho, J.: Progressive distillation for fast sampling of difusion models. arXiv preprint arXiv:2202.00512 (2022), https://arxiv.org/abs/2202.00512

39. Sauer, A., Boesel, F., Dockhorn, T., Blattmann, A., Esser, P., Rombach, R.: Fast high-resolution image synthesis with latent adversarial difusion distillation. In: SIGGRAPH Asia 2024 Conference Papers. pp. 1–11 (2024)

40. Sauer, A., Lorenz, D., Blattmann, A., Rombach, R.: Adversarial difusion distillation. In: European Conference on Computer Vision. pp. 87–103. Springer (2024)

41. Sauer, A., Lorenz, D., Blattmann, A., Rombach, R.: Adversarial difusion distillation. In: European Conference on Computer Vision. pp. 87–103. Springer (2024)

42. Schuhmann, C., Beaumont, R., Vencu, R., Gordon, C., Wightman, R., Cherti, M., Coombes, T., Katta, A., Mullis, C., Wortsman, M., et al.: Laion-5b: An open largescale dataset for training next generation image-text models. Advances in neural information processing systems 35, 25278–25294 (2022)

43. Schusterbauer, J., Gui, M., Fundel, F., Ommer, B.: Dif2flow: Training flow matching models via difusion model alignment. arXiv preprint arXiv:2506.02221 (2025), https://arxiv.org/abs/2506.02221

44. Song, J., Meng, C., Ermon, S.: Denoising difusion implicit models. arXiv preprint arXiv:2010.02502 (2020), https://arxiv.org/abs/2010.02502

45. Song, Y., Sohl-Dickstein, J., Kingma, D.P., Kumar, A., Ermon, S., Poole, B.: Scorebased generative modeling through stochastic diferential equations. In: International Conference on Learning Representations (ICLR) (2021), https://arxiv. org/abs/2011.13456

46. Team, K.: Kolors: Efective training of difusion model for photorealistic textto-image synthesis. arXiv preprint (2024), https://github.com/Kwai-Kolors/ Kolors

47. Tong, A., Fatras, K., Malkin, N., Huguet, G., Zhang, Y., Rector-Brooks, J., Wolf, G., Bengio, Y.: Improving and generalizing flow-based generative models with minibatch optimal transport. arXiv preprint arXiv:2302.00482 (2023), https://arxiv.org/abs/2302.00482

48. Vu, D., Nguyen, A., Tran, C., Tran, A.: Anti-i2v: Safeguarding your photos from malicious image-to-video generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 37621–37631 (2026)

49. Vu, D., Nguyen, K., Nguyen, T.T., Nguyen, N., Nguyen, P., Nguyen, K., Pham, C., Tran, A.: Inverfill: One-step inversion for enhanced few-step difusion inpainting. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 25677–25687 (2026)

50. Wang, Z., Lu, C., Wang, Y., Bao, F., Li, C., Su, H., Zhu, J.: Prolificdreamer: High-fidelity and diverse text-to-3d generation with variational score distillation. In: Advances in Neural Information Processing Systems (NeurIPS) (2023), https: //arxiv.org/abs/2305.16213

51. Wu, X., Hao, Y., Sun, K., Chen, Y., Zhu, F., Zhao, R., Li, H.: Human preference score v2: A solid benchmark for evaluating human preferences of text-to-image synthesis. arXiv preprint arXiv:2306.09341 (2023)

52. Xie, E., Chen, J., Chen, J., Cai, H., Tang, H., Lin, Y., Zhang, Z., Li, M., Zhu, L., Lu, Y., et al.: Sana: Eficient high-resolution image synthesis with linear difusion transformers. arXiv preprint arXiv:2410.10629 (2024)

53. Xie, E., Chen, J., Zhao, Y., Yu, J., Zhu, L., Lin, Y., Zhang, Z., Li, M., Chen, J., Cai, H., et al.: Sana 1.5: Eficient scaling of training-time and inference-time compute in linear difusion transformer. In: International Conference on Machine Learning. pp. 68578–68598. PMLR (2025)

54. Xie, Q., Liao, Z., Deng, Z., Chen, C., Lu, H.: Tlcm: Training-eficient latent consistency model for image generation with 2-8 steps. arXiv preprint arXiv:2406.05768 (2024). https://doi.org/10.48550/arXiv.2406.05768

55. Xu, J., Liu, X., Wu, Y., Tong, Y., Li, Q., Ding, M., Tang, J., Dong, Y.: Imagereward: Learning and evaluating human preferences for text-to-image generation. In: Advances in Neural Information Processing Systems (NeurIPS) (2023)

56. Xu, Y., Nie, W., Vahdat, A.: One-step difusion models with f-divergence distribution matching. arXiv preprint arXiv:2502.15681 (2025), https://arxiv.org/abs/ 2502.15681

57. Yin, T., Gharbi, M., Park, T., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T.: Improved distribution matching distillation for fast image synthesis. In: NeurIPS (2024)

58. Yin, T., Gharbi, M., Zhang, R., Shechtman, E., Durand, F., Freeman, W.T., Park, T.: One-step difusion with distribution matching distillation. In: CVPR (2024)

59. Zhang, .: Learning multi-dimensional human preference for text-to-image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2024)

60. Zhang, Y., Hooi, B.: Hipa: Enabling one-step text-to-image difusion models via high-frequency-promoting adaptation. arXiv preprint arXiv:2311.18158 (2023). https://doi.org/10.48550/arXiv.2311.18158, https://arxiv.org/abs/2311. 18158

61. Zhao, Y., Xu, Y., Xiao, Z., Jia, H., Hou, T.: Mobiledifusion: Instant text-to-image generation on mobile devices. In: European Conference on Computer Vision. pp. 225–242. Springer (2024)

62. Zheng, J., Hu, M., Fan, Z., Wang, C., Ding, C., Tao, D., Cham, T.J.: Trajectory consistency distillation: Improved latent consistency distillation by semi-linear consistency function with trajectory mapping. arXiv preprint arXiv:2402.19159 (2024). https://doi.org/10.48550/arXiv.2402.19159

63. Zhou, M., Wang, Z., Zheng, H., Huang, H.: Guided score identity distillation for data-free one-step text-to-image generation (2024), https://arxiv.org/abs/ 2406.01561

64. Zhu, J., Wang, H., Su, M., Wang, Z., Wang, H.: Obs-dif: Accurate pruning for difusion models in one-shot. arXiv preprint arXiv:2510.06751 (2025), https:// arxiv.org/abs/2510.06751

Cross-Space Distillation: Teaching One-Step Students with Modern Difusion Teachers – Supplementary Materials –

Overview. This supplementary material expands the main paper with additional implementation details, Bridge design ablations, eficiency analyses, and extended qualitative results. Beyond supporting reproducibility, these results further illustrate the practical strengths of Cross-Space Distillation: Bridge is lightweight, reusable across diverse teachers, eficient to train as a post-training module, and efective in transferring strong visual priors into compact one-step students. We first report implementation details for Bridge training and downstream distillation, then provide additional details on Bridge design, eficiency, usage, and ablations, and finally include extended qualitative comparisons. We conclude by discussing the current scope of the method and promising directions beyond the main paper.

## A Implementation and Hyperparameter Details

We summarize the implementation details for both Bridge training and downstream one-step distillation in Tab. A1. Across all teacher models, we use the same Bridge architecture and modify only the final projection layer to match the latent dimensionality of the target teacher. The Bridge is trained from scratch, without pretrained initialization. In our setup, both stages fit on a single 8×H100 (80GB) node, making the overall pipeline practical as a post-training recipe rather than a full model pretraining procedure. Bridge training optimizes only the lightweight alignment module, whereas the subsequent student distillation follows a standard one-step training setup with minor batch-size adjustments across teachers.

## B Bridge Design, Eficiency, and Usage Details

## B.1 Architectural Design

The ablations in Fig. A3 compare Bridge architectures under the same training budget, with 5M trainable parameters and 10K training iterations. The results confirm the intended ordering of the architectural variants: MLP performs worst, UNet provides a stronger baseline, SwinIR improves further over UNet, and

Table A1: Implementation details, hyperparameters, and compute settings for Bridge training and downstream one-step distillation of students from diferent teachers. Across all teachers, Bridge uses the same architecture and changes only its output dimensionality to match the target teacher latent space. Both stages fit on a single 8×H100 (80GB) node, making the overall pipeline practical as a reusable post-training recipe rather than a full model pretraining procedure. Reported training time excludes text-encoding overhead. We use FLUX.2 and SD 3.5 as abbreviations for FLUX.2- Klein-4B and SD 3.5 Medium, respectively.

<table><tr><td colspan="6">Bridge Settings</td></tr><tr><td>Architecture</td><td></td><td></td><td>SwinIR</td><td></td><td></td></tr><tr><td>Number of synthesized images</td><td></td><td></td><td>2M</td><td></td><td></td></tr><tr><td>Trainable parameters</td><td></td><td></td><td>5M</td><td></td><td></td></tr><tr><td>Loss weight α</td><td></td><td></td><td>1.0</td><td></td><td></td></tr><tr><td>Loss weight β</td><td></td><td></td><td>1.0</td><td></td><td></td></tr><tr><td>Learning rate for B</td><td></td><td></td><td>1e-4</td><td></td><td></td></tr><tr><td>Optimizer</td><td></td><td colspan="4">Adam ( $\beta_1 = 0, \beta_2 = 0.999, \epsilon = 10^{-8}$ )</td></tr><tr><td>Compute</td><td></td><td colspan="4">8×H100 (80GB)</td></tr><tr><td>Batch size per GPU</td><td></td><td colspan="4">32</td></tr><tr><td>Training time (hours)</td><td></td><td colspan="4">8</td></tr><tr><td colspan="6">Distillation Settings</td></tr><tr><td>Teacher model</td><td>SDXL</td><td>Kolors</td><td>PixArt-σ</td><td>FLUX.2</td><td>SD 3.5</td></tr><tr><td>Generative paradigm</td><td>Diffusion</td><td>Diffusion</td><td>Diffusion</td><td>Flow</td><td>Flow</td></tr><tr><td>Teacher latent channels</td><td>4</td><td>4</td><td>4</td><td>32</td><td>16</td></tr><tr><td>Student learning rate</td><td></td><td></td><td>1e-6</td><td></td><td></td></tr><tr><td>Auxiliary score model learning rate</td><td></td><td></td><td>5e-4</td><td></td><td></td></tr><tr><td>Discriminator learning rate</td><td></td><td></td><td>5e-7</td><td></td><td></td></tr><tr><td>Adversarial loss weight wGAN</td><td></td><td></td><td>0.5</td><td></td><td></td></tr><tr><td>Distribution-matching loss weight wVSD</td><td></td><td></td><td>1.0</td><td></td><td></td></tr><tr><td>Optimizer</td><td></td><td colspan="4">Adam ( $\beta_1 = 0, \beta_2 = 0.999, \epsilon = 10^{-8}$ )</td></tr><tr><td>Compute</td><td></td><td colspan="4">8×H100 (80GB)</td></tr><tr><td>Batch size per GPU</td><td>32</td><td>32</td><td>32</td><td>16</td><td>32</td></tr><tr><td>Training time (hours)</td><td>20</td><td>20</td><td>20</td><td>20</td><td>20</td></tr></table>

SwinIR with the proposed Spatial Prior achieves the best overall reconstruction fidelity. These results support our design choice: a stronger image-restorationstyle backbone is more efective for latent-space alignment, and the frozen Spatial Prior provides an additional gain beyond backbone choice alone.

## B.2 Training Objectives

Fig. A1 provides a qualitative comparison of the training objectives used for learning B. Using only the $\ell _ { 1 }$ reconstruction loss preserves the coarse object layout, but the reconstructed image remains visibly over-smoothed, with weakened local contrast and softened facial structure. Adding E-LatentLPIPS yields only limited perceptual improvement over $\ell _ { 1 }$ alone. In contrast, incorporating the proposed Attention Fidelity loss produces substantially sharper and more faithful reconstructions, especially in fine-grained regions such as the eyes, nose contour, and surrounding fur. These visual diferences are consistent with the quantitative ablation in the main paper, where Attention Fidelity improves reconstruction quality over both $\ell _ { 1 }$ and E-LatentLPIPS.

L1  
Groundtruth  
![](images/2ac3b5f8d4879bc43707f5c169d3cbe455163b55ee1c78f7aed70c2cd03ee417.jpg)

![](images/c066eaafa2e37483fb703fd5e0b09e8b1ae009f9124fe289caefa03e906085d4.jpg)

![](images/e0c31c31779ce8ded254f0490f54057daf74450036c6e9a768fd900ad50b6b34.jpg)

![](images/957cdcfdea546f75014244ce2dadc97c08826fa5bd8d22ce828c63e1ff5b48e1.jpg)  
Fig. A1: Qualitative comparison of training objectives for B. From left to right: reconstructions trained with $\ell _ { 1 }$ only, $\ell _ { 1 } +$ E-LatentLPIPS, and $\ell _ { 1 } +$ Attention Fidelity, followed by the ground-truth reconstruction. All variants use the SwinIR backbone and are trained to map SD 2.1 latents to the SDXL latent space. While $\ell _ { 1 }$ preserves the overall structure, it produces noticeably smooth reconstructions, and E-LatentLPIPS provides only marginal visual improvement. The proposed Attention Fidelity loss better preserves local structure and fine facial details, yielding reconstructions that are visually closest to the ground truth. Please zoom in for closer inspection.

Table A2: Runtime and peak memory overhead introduced by Bridge over the base SD 1.5 pipeline. Measurements are averaged over 500 iterations on a single H100 80GB GPU with batch size 1. The results show that Bridge adds only modest inference overhead while enabling alignment to higher-dimensional teacher latent spaces.

<table><tr><td>Setting</td><td>Latent ch.</td><td>Runtime (ms)</td><td>Peak GPU Memory (GB)</td></tr><tr><td>Base SD 1.5 (No  $\mathcal{B}$ )</td><td>4</td><td>26.48</td><td>3.96</td></tr><tr><td>+ $\mathcal{B}$  to SDXL</td><td>4</td><td>32.49 (+6.01)</td><td>4.50 (+0.54)</td></tr><tr><td>+ $\mathcal{B}$  to SD 3.5 Medium</td><td>16</td><td>33.93 (+7.45)</td><td>4.51 (+0.55)</td></tr><tr><td>+ $\mathcal{B}$  to FLUX.2-klein-4B</td><td>32</td><td>44.49 (+18.01)</td><td>4.74 (+0.78)</td></tr></table>

## B.3 Eficiency

Tab. A2 reports the average runtime and peak memory overhead introduced by Bridge during inference. All measurements are averaged over 500 iterations on a single H100 80GB GPU with batch size 1. We report results for the three Bridge variants that map the SD 1.5 latent space to the latent spaces of SDXL, SD 3.5 Medium, and FLUX.2-klein-4B; Kolors and PixArt-σ share the same VAE as SDXL and therefore use the same Bridge configuration.

Overall, Bridge introduces only a modest increase in runtime and memory over the base SD 1.5 pipeline. The overhead grows with the target latent dimensionality, as expected, but remains manageable even for the 32-channel FLUX latent space. These measurements support the practical use of Bridge as a lightweight latent-space interface rather than a heavy second-stage generator.

![](images/d9e2d1b0714cba0f3358f0f7f1cf2458deb7493aec11a797f851a7bea1e2b124.jpg)  
Fig. A2: Qualitative examples of inference-time resolution upgrade with Bridge for the SD 1.5 Student distilled from the SD 3.5 Medium Teacher. Given low-resolution inputs at 512 × 512, Bridge predicts teacher-compatible high-resolution latents at 1024 × 1024 that closely match the target reconstructions, while preserving fine structures, textures, and global consistency across diverse scenes. This experiment evaluates Bridge in the same 512 → 1024 setting studied throughout the paper.

## B.4 Usage During Distillation

Unless otherwise stated, B is kept frozen throughout downstream Cross-Space distillation. We additionally explored jointly updating B together with the Student during distillation, but observed unstable optimization and, in some cases, training collapse. We therefore freeze B in all downstream distillation experiments. This design stabilizes training, reduces memory usage, and preserves B as a pretrained latent-space interface rather than a second trainable generator component.

## B.5 Inference-Time Resolution Upgrade

We provide additional qualitative examples of inference-time resolution upgrade using Bridge in Fig. A2. Starting from low-resolution inputs, Bridge predicts teacher-compatible high-resolution latents that can be decoded at 1024 × 1024 while preserving structure, texture, and global coherence. This shows that Bridge can be reused at inference time as a lightweight latent-space interface, beyond its role in downstream distillation.

This inference setting should be interpreted in the same scope as the main paper: Bridge maps a low-resolution Student latent onto the Teacher latent grid under Cross-Resolution and Cross-VAE mismatch. Our focus is therefore the practical 512 → 1024 setting studied throughout the paper, which matches the Student–Teacher pairs used in Cross-Space Distillation. Larger output scales and arbitrary output resolutions are beyond the scope of this work.

<table><tr><td>Framework</td><td>L1 ↓</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>Baseline</td><td>0.048</td><td>22.6</td><td>0.68</td></tr><tr><td>+ Attention Fidelity</td><td>0.037</td><td>24.39</td><td>0.73</td></tr><tr><td>+ Spatial Prior</td><td>0.035</td><td>24.97</td><td>0.75</td></tr></table>

(a) Components of proposed framework

<table><tr><td>Objectives</td><td>L1 ↓</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>L1</td><td>0.041</td><td>23.58</td><td>0.70</td></tr><tr><td>E-LatentLPIPS</td><td>0.042</td><td>23.62</td><td>0.71</td></tr><tr><td>Attention Fidelity</td><td>0.035</td><td>24.97</td><td>0.75</td></tr></table>

![](images/b3b74eb9223dc5e4cb1a01f54a9400b794955343a59d9a4d558560f60a81e9a1.jpg)  
Convergence across objectives.  
(d)

(b) Objectives

<table><tr><td>Architectures</td><td>L1 ↓</td><td>PSNR ↑</td><td>SSIM ↑</td></tr><tr><td>MLP</td><td>0.075</td><td>19.46</td><td>0.60</td></tr><tr><td>UNet</td><td>0.040</td><td>23.61</td><td>0.68</td></tr><tr><td>SwinIR [21]</td><td>0.037</td><td>24.39</td><td>0.73</td></tr><tr><td>+ Spatial Prior</td><td>0.035</td><td>24.97</td><td>0.75</td></tr></table>

(c) Architectures

![](images/b1e38651b277d3e99694b720c95fd7d9df85645353409e45546616cc330d0716.jpg)  
Convergence across architectures.  
Fig. A3: Bridge design ablations and convergence. Left: Ablations of Bridge training, design components, objectives, and alignment architectures using decodedimage fidelity metrics. Attention Fidelity provides a large gain, and the full Bridge with Spatial Prior plus Attention Fidelity attains the best fidelity. Right: SSIM over training iterations. Attention Fidelity speeds up convergence and increases the final SSIM. Under matched settings, the Spatial Prior alignment architecture converges to higher SSIM than alternatives. L1 reconstruction is used in all experiments.

## B.6 Bridge Ablation Studies

Fig. A3 summarizes our Bridge design ablations and convergence behavior. We report reconstruction metrics L1, PSNR, and SSIM in image space, computed between the decoded outputs $\hat { x } = \mathcal { D } _ { T } ( \hat { z } _ { T } )$ and $x = \mathcal { D } _ { T } ( z _ { T } )$ . All variants are trained on the same data with the same optimization schedule.

Architectures. We compare three backbones: MLP, UNet, and SwinIR. For SwinIR, we additionally evaluate our frozen spatial prior, implemented as a fixed prefix of the Student decoder that expands z<sub>S</sub> to the Teacher grid before projection. The MLP underperforms, while UNet and SwinIR provide stronger spatial modeling. Adding the spatial prior to SwinIR gives a clear gain, indicating that reusing a pretrained upsampling scafold helps the learnable projector focus on feature and semantic alignment rather than relearning spatial expansion.

Objectives. We ablate objectives for learning $B _ { \phi }$ . Using only $\mathcal { L } _ { r e c }$ yields limited fidelity, and E-LatentLPIPS provides marginal improvement. In contrast, Attention Fidelity consistently enhances reconstruction and achieves the same SSIM in fewer steps, indicating that matching the Teacher denoiser’s attention ofers a stronger and more stable supervision in Teacher space.

## C Limitations

Our current study focuses on the most practically important form of heterogeneity in modern distillation pipelines: mismatch in latent resolution and VAE space. In this sense, Bridge is intentionally designed as a latent-space interface: it aligns Student and Teacher representations after conditioning has been formed, while leaving the Student backbone unchanged. This design keeps the method lightweight, modular, and easy to integrate with existing one-step backbones, but it also means that diferences in the conditioning stack, such as the text encoders used by modern foundation models (e.g., FLUX or SD 3.5) versus the CLIPbased conditioning used in compact students such as SD 1.5, are not explicitly modeled in the current formulation. Extending the same interface principle from latent alignment to conditioning-space alignment is a promising next step.

Our experiments also target a deliberate operating point: compact one-step image generation. Rather than reproducing the full capacity of large multi-step Teachers, the goal of this paper is to transfer as much Teacher knowledge as possible into eficient, ecosystem-compatible Students through a minimal additional module. Under this constraint, Bridge substantially narrows the gap to much larger Teachers while preserving the deployment advantages of compact backbones. We view this as a strength of the current formulation: it isolates the representation-alignment problem without requiring student redesign or largescale retraining from scratch.

More broadly, we believe the scope of Cross-Space Distillation extends beyond the specific text-to-image setting studied here. Many modern generative systems operate over heterogeneous latent spaces, including latent video models, image editing pipelines, and other multimodal generators. The formulation introduced in this paper therefore suggests a broader research direction: treating alignment across heterogeneous latent representations as a reusable interface problem. Extending Bridge to video generation, editing models, and richer conditional pipelines is a natural next step, and one that could further increase the practical impact of compact generative models.

## D Additional Qualitative Results

We provide additional uncurated samples generated by our distilled one-step SD 1.5 and SD 2.1 in Fig. A4 and Fig. A5, respectively.

## E Extended Related Work

## E.1 Difusion and Flow.

Difusion models have become the dominant framework for high-quality image generation. Early works such as DDPM [12] and score-based SDE models [45] established the standard formulation of learning a reverse denoising process from progressively corrupted data. Subsequent works improved the eficiency of this framework. DDIM [44] introduced a deterministic sampling path that substantially reduces the number of denoising steps, while DPM-Solver [25] and EDM [15] further improved fast sampling through better numerical solvers, parameterization, and noise design.

In parallel, continuous-time transport formulations provide another important view of generative modeling. Flow Matching [23] learns the velocity field of a probability path between noise and data, ofering a simple alternative to difusion-style objectives. Rectified Flow [24] further studies straighter transport paths to simplify generation trajectories, and OT-based conditional flow matching [47] improves path quality through optimal transport based couplings. These formulations have recently become increasingly relevant for large-scale image generation.

Modern text-to-image systems build on these advances with stronger backbones, larger training data, and higher-capacity latent representations. Stable Difusion 3.5 [8] and Flux [16,17] are representative examples that achieve strong visual fidelity and prompt alignment. However, these gains typically come with higher inference cost and larger model size, which makes direct deployment difficult in practical resource-constrained settings.

## E.2 Distillation for Fast Difusion Sampling.

To reduce inference cost, many works study distilling multi-step difusion models into one-step or few-step generators. Progressive Distillation [38] is an early and influential approach that progressively shortens the sampling trajectory while preserving the behavior of the original model. Consistency-based methods follow a related goal. Latent Consistency Models [26] train a latent generator that supports high-quality few-step inference, and Truncated Consistency Models [18] further improve eficiency by learning from shortened trajectories.

Another important line is distribution-based distillation. SwiftBrush [50] and SwiftBrush v2 [31] show that one-step generators can be trained efectively by matching Teacher and Student predictions on noised samples. Related methods such as one-step difusion with f-divergence distribution matching [56], SNOOPI [33], and more recent distribution matching approaches [57, 58] further improve this idea by directly optimizing the Student toward the Teacher-induced data distribution. These methods have significantly narrowed the quality gap between fast Students and their multi-step Teachers.

Several works also combine distillation with adversarial supervision. ADD [40] introduces a discriminator to provide stronger perceptual guidance during one-step distillation, and LADD [39] extends this idea to latent high-resolution image synthesis. Such methods are efective for improving sharpness and realism, especially when standard distillation losses alone are not suficient. However, most of these approaches assume that Teacher and Student operate in the same latent space, or at least under compatible latent resolution and VAE parameterization. This assumption is natural for within-family distillation, but becomes restrictive when the Teacher and Student come from diferent model families.

Eficient and Mobile Difusion Models. Another line of research aims to reduce the physical size and computational footprint of difusion models to enable deployment on resource-constrained devices. Approaches such as FastFlux [1], SnapFusion [20], MobileDifusion [61], and SnapGen [2, 13] explore techniques including network pruning, structural compression, and lightweight architecture design. These methods demonstrate that difusion models can be significantly compressed while maintaining acceptable image quality.

However, such approaches face several practical challenges. Network pruning often encounters a sparsity threshold beyond which further compression leads to noticeable quality degradation. Lightweight architectures typically require substantial redesign and retraining tailored to specific Teacher models.

Our goal is diferent from these approaches. Instead of modifying the Student backbone itself, we keep the compact Student architecture unchanged and study how to transfer knowledge from a stronger Teacher even when the two models use diferent latent spaces. In this sense, our setting is complementary to eficientmodel design and model compression.

## E.3 Representation Alignment and Internal Distillation.

Our work is also related to distillation methods that align internal representations instead of only matching final outputs. In language modeling, MiniLLM [10] shows that reverse-KL based distillation can better preserve the dominant behavior of a strong Teacher. In difusion models, X2I [27] shows that attention distillation can serve as an efective supervision signal for transferring useful internal structure across models.

These observations are closely related to our design. Rather than treating the Student output alone as the target of distillation, we also consider how internal representations can be aligned when Teacher and Student are mismatched. Diferent from prior works, however, our focus is specifically on the case where the gap comes from both latent resolution and latent parameterization. Our Cross-Space Distillation addresses this problem with a lightweight Bridge that maps Student features into the Teacher latent space, so that standard distillation objectives can still be applied without changing the Student architecture.

## E.4 Comparison to Deep Compression Autoencoder Approaches.

Our technique achieves eficient image generation by distilling from a big teacher operating at resolution 1024×1024 to a compact student operating at resolution $5 1 2 \times 5 1 2$ . Some recent methods achieve similar eficiency using deep compression autoencoders, such as the SANA family [4, 52, 53]. However, our method ofers several advantages: (1) Eficient training. SANA and SANA 1.5 need to be trained from scratch, using a large amount of training data and at least 100K training iterations. Even the distillation process to produce SANA-Sprint requires 25000 iterations on 32 A100 GPUs, using the same training data. In contrast, our method relies on score distillation to a student model that is well initialized with existing pretrained weights. Hence, our training is highly eficient, requiring only 2M synthetic images and around 10K training iterations. (2) Simple architecture. SANA models are based on the DiT structure. To achieve high inference speed, they need to employ advanced techniques such as Linear Attention and Triton-accelerated modules. Our method instead can reuse the standard UNet backbones in Stable Difusion models, (3) Onboardfriendliness. Since SANA models are based on tailored architectures, onboarding them to devices requires significant efort. In contrast, our distilled students can be easily deployed on edge devices and are compatible with a broad range of existing applications.

## F Societal Impacts

Our work aims to enable a highly eficient framework for fast, accessible, and high-quality image generation. At the same time, we recognize that advanced image manipulation methods may be misused to create deceptive content [7, 32, 49]. To address these concerns, we emphasize the importance of developing robust detection approaches for AI-generated or manipulated media [9,30,34,48], alongside promoting the responsible deployment of such technologies.

![](images/4952d0739fcfb3481dad92790c3f9413727ed11503e667a7e14918a20dbf769b.jpg)  
Fig. A4: Additional Qualitative Results for our SD 1.5 Merged Model.

![](images/5853cf61952f1f1b1f986c94193c3ea438edcc954c74eb413f7a90c9d65fad67.jpg)  
Fig. A5: Additional Qualitative Results for our SD 2.1 Student Merged Model.