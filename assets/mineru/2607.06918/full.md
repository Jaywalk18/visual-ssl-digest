# LoCA: Spatially-Aware Low-Rank Convolutional Adaptation of Vision Foundation Models

Sojung An<sup>1\*</sup>, Junha Lee<sup>2,3\*</sup>, Sujeong You<sup>2</sup>, Nam Ik Cho<sup>3</sup>, Donghyun Kim<sup>1†</sup>

<sup>1</sup> Korea University, Seoul, Republic of Korea <sup>2</sup> Korea Institute of Industrial Technology, Ansan, Republic of Korea <sup>3</sup> Seoul National University, Seoul, Republic of Korea

<sup>∗</sup>Equal contribution <sup>†</sup>Corresponding author: d\_kim@korea.ac.kr https://github.com/ssojungan/loca

Abstract. Pre-trained Vision Foundation Models (VFMs) provide strong visual representations for diverse downstream tasks. The key challenge of VFM adaptation stems from the prohibitive costs of full fine-tuning and catastrophic forgetting. To address this, Low-Rank Adaptation (LoRA) has emerged as the prevailing paradigm for Parameter-Eficient Fine-Tuning (PEFT). However, LoRA is typically designed for transformer self-attention layers parameterized by 2D matrices. Since convolutional kernels inherently couple spatial and channel information within a 4D tensor, forcing them into a monolithic 2D matrix disrupts the inherent spatial topology. In this paper, we propose Low-Rank Convolutional Adaptation (LoCA), a convolution-aware PEFT framework that addresses spatial-channel entanglement by decoupling channel and spatial adaptation. LoCA introduces a low-rank channel adaptation for dense crosschannel mixing and refines spatial bases extracted from pre-trained kernels via Singular Value Decomposition (SVD). Experimental results show that LoCA preserves pre-trained spatial priors and achieves competitive or state-of-the-art performance across fine-grained classification, domaingeneralized semantic segmentation, and generative benchmarks.

Keywords: Parameter-Eficient Fine-Tuning · Low-Rank Adaptation · Spatial Inductive Bias · Convolutional Networks

## 1 Introduction

Vision Foundation Models (VFMs) enable a wide array of general vision tasks [12, 28,38]. These models learn rich multi-scale representations from large-scale data and transfer efectively to downstream problems through fine-tuning or frozen feature extraction [24,27]. However, full fine-tuning of large-scale VFMs not only incurs prohibitive computational costs but also leads to catastrophic forgetting of pre-trained knowledge. Parameter-Eficient Fine-Tuning (PEFT) addresses this by optimizing only a small subset of parameters while freezing the original weights [16, 44]. Among these, Low-Rank Adaptation (LoRA) has emerged as the de facto standard, ofering high representational capacity with minimal trainable parameters [11, 21, 47, 50].

![](images/f25768bd61d782a0623ea2c9213c7fb2acf6be5d2ac12f35934cd4f1a33281ec.jpg)  
(a) Radial Spread (↑)

![](images/0adcb2a8eefe9bac4283ea8577c142d375cd765077a390e4e8c516bf04083dec.jpg)  
(b) Center Energy Ratio (↓)

![](images/61c116e88ce88870c9a028f83607ade8721284a0a0f3813e9a79030ebc649dd5.jpg)  
(c) Low-Freq Energy Ratio (↑)  
Fig. A: The training dynamics of Efective Receptive Field (ERF) across convolutional layers (dwconv) for FFT, LoRA, FSF, and LoCA. (a) Radial Spread: Average distance of gradients from the map center. (b) Center Energy Ratio: Gradient ratio in the 3 × 3 center. (c) Low-Freq Energy: Fourier low-frequency power ratio. LoRA and FSF exhibit transient ERF growth followed by a reconvergence toward localized patterns, which limit sustained global context. Spatial reconvergence necessitates a structurepreserving design to achieve an expansive receptive field comparable to FFT.

While LoRA is typically designed for linear projections in transformers, convolutional operators in VFM backbones remain relatively underexplored. This gap is critical because convolutional operators remain fundamental across modern VFM backbones. ConvNeXt is a modern convolutional backbone that emphasizes convolution as a sliding-window, weight-sharing strategy that encodes spatial inductive biases [43]. Self-attention in ViT [12] enables global tokento-token interactions but provides limited built-in spatial inductive bias. This prompts hybrid designs that incorporate convolutional components to encode spatial priors [6]. Mamba-based State Space Model (SSM) vision backbones have gained recent attention [14, 17]. Some architectures retain convolutional blocks in early high-resolution stages for local feature extraction. In addition to visual perception, convolutional networks are widely used in generative models such as Stable Difusion [36] in the U-Net backbone. Given that convolutional operators remain fundamental across modern VFM backbones, extending LoRA beyond Transformer linear layers to convolutional layers becomes an important problem in PEFT.

However, directly applying LoRA’s low-rank updates to convolution layers is suboptimal, as they operate over spatial regions. Naive LoRA flattens convolutional kernels (4D tensor) into a two-dimensional matrix for low-rank updates [10]. Flattening the convolutional kernel collapses the inherent spatial topology by enforcing cross-channel mixing within a single low-rank parameterization. This structural mismatch limits the preservation of spatial priors, including locality and directionality, and reduces adaptation gains reported in prior work [5,52]. Recent filter subspace approaches address spatial–channel entanglement by decomposing convolutional filters into spatial bases and channel coeficients [5]. Sparse coding approximates pre-trained kernels within this decomposed subspace. Such approximation inevitably modifies pre-trained representations prior to fine-tuning. Freezing cross-channel mixing coeficients limits the adaptation of inter-channel correlations in new domains.

![](images/30b3113c3f82c8e3138ef13f034bc8064adc8e6fe5c45a2a62aa89865c38b20a.jpg)

![](images/8380b80ba21467954c2edb28977053522bea3ae778485dd43f6aa63359583fd8.jpg)  
(b) Generalization performance of PEFT methods across diverse vision benchmarks

(a) Parameter-eficient scaling behavior. The x-axis is sorted by GFLOPs.  
![](images/f7b10d6da3ca5470765706e10912b65b43b0044108decb47afa1bf18b7588f0f.jpg)  
(c) Trainable parameter comparison across convolution kernel sizes (K). Detailed analysis is provided in Appendix A.  
Fig. B: Performance comparison of PEFT-based methods on downstream tasks

As shown in Fig. A, we analyze the Efective Receptive Field (ERF) [18] and low-frequency retention during fine-tuning. Higher Radial Spread with lower Center Energy Ratio indicates outward gradient propagation and receptive field growth, while higher Low-Freq Energy reflects stronger retention of informative low-frequency structures. FFT and LoCA show consistent increases in spatial coverage and low-frequency energy throughout training. In contrast, LoRA [21] and Filter Subspace Fine-Tuning (FSF) [5] exhibit only transient spatial expansion, followed by reconvergence toward localized update patterns and unstable frequency behavior. These observations suggest that preserving convolutional spatial structure is crucial for maintaining broad receptive fields, motivating our spatial–channel disentanglement approach with joint adaptation.

To this end, we propose Low-Rank Convolutional Adaptation (LoCA), a structured low-rank reparameterization that adapts convolutional layers by decoupling channel mixing from spatial basis refinement. We first introduce a lowrank channel adaptation process that captures dense cross-channel mixing while mitigating spatial–channel entanglement. Such channel adaptation prevents the topological collapse induced by naive kernel flattening and the structural inaccuracies of conventional weight decomposition. We then design a spatial adaptation mechanism that preserves pre-trained priors through structural bases derived from Singular Value Decomposition (SVD). Additionally, we introduce a hierarchical rank scheduling for convolutional foundation models. LoCA preserves the spatial inductive bias of pre-trained representations and achieves robust performance across diverse downstream tasks.

We summarize the contributions of this work as follows:

– We propose the LoCA framework to address spatial-channel entanglement by decoupling channel and spatial adaptation.

– We introduce SVD-based spatial basis refinement to preserve pre-trained spatial inductive biases efectively.

– We propose a hierarchical rank scheduling tailored to convolutional foundation models.

– Extensive experiments demonstrate that LoCA achieves competitive or stateof-the-art performance across fine-grained classification, domain-generalized segmentation, and generative benchmarks (see Fig. B).

## 2 Related Work

## 2.1 Parameter-Eficient Fine-Tuning

PEFT has emerged as a practical paradigm for adapting large-scale VFMs to downstream tasks by updating only a small subset of parameters while freezing pre-trained backbones [21, 22]. Existing approaches include adapter-based methods [4, 20], which insert lightweight trainable modules into network blocks; prompt-based methods [22,26], which add learnable tokens to the input sequence; selective fine-tuning methods [2, 15], which update specific components such as bias terms; and reparameterization-based methods [11, 21, 47, 50], which optimize implicit low-rank structures for seamless merging into the original weights at inference.

Most PEFT techniques are formulated for Transformer architectures that operate on token sequences with multi-head self-attention [33]. Extending these paradigms to vision tasks such as detection and segmentation remains challenging because pixel-level prediction relies on spatial inductive biases and multiscale hierarchies.

## 2.2 Parameter-Eficient Fine-Tuning for Convolutional Layers

Convolution preserves 2D image structure by leveraging inherent spatial inductive biases [28, 30]. Early convolution-specific PEFT methods such as Conv-Adapter [4] introduce trainable modules into convolutional blocks and increase inference-time computation. Flattening-based extensions convert 2D convolutional kernels parameterized as 4D tensors into 2D matrices to reuse linear LoRA formulations [10]. This dimensional collapse entangles spatial topology with cross-channel mixing, compromising locality and weight sharing [5, 52].

Filter subspace methods constrain updates by decomposing pre-trained kernels into spatial atoms and mixing coeficients [5]. This decomposition reconstructs the pre-trained weights only approximately while introducing accumulation error. Reconstructing weights from these decomposed atoms can also restrict cross-channel flexibility by freezing mixing coeficients. These limitations motivate a convolution-aware PEFT that preserves the original weights while enabling spatially structured low-rank updates.

## 2.3 Low-Rank Adaptation and Singular Value Decomposition

LoRA represents weight adaptation using low-rank factors. To maximize parameter eficiency, subsequent methods have explored rank adaptation. For example, AdaLoRA dynamically adjusts the rank budget across diferent layers based on importance scores [50]. Other works leverage SVD to replace random initialization with informed low-rank initialization. PiSSA [31], SoMA [47], and SoRA [11] decompose pre-trained weights and use principal or minor components to initialize low-rank factors, improving convergence and knowledge retention. These works suggest that singular components capture reusable structure in pre-trained weights. Building upon this idea, we employ SVD to extract spatial bases from convolutional kernels.

## 3 Preliminaries

LoRA. LoRA freezes pre-trained weights and approximates updates using lowrank matrices. Motivated by the hypothesis that weight changes during model adaptation possess a low intrinsic rank, LoRA parameterizes the incremental update via the product of two low-rank matrices [21]. For a pre-trained weight matrix $W _ { 0 } \in \mathbb { R } ^ { d _ { o u t } \times d _ { i n } }$ , where $d _ { i n }$ and $d _ { o u t }$ denote the input and output dimensions respectively, LoRA decomposes the update $\varDelta W \in \mathbb { R } ^ { d _ { o u t } \times d _ { i n } }$ into $B A .$ , where $B \in \mathbb { R } ^ { d _ { o u t } \times r }$ and $A \in \mathbb { R } ^ { r \times d _ { i n } }$ are low-rank matrices with rank r ≪ min $( d _ { o u t } , d _ { i n } )$ and α is a constant scaling factor. Consequently, the fine-tuned weight $W ^ { \prime }$ is formulated as:

$$
W ^ {\prime} = W _ {0} + \varDelta W = W _ {0} + \frac {\alpha}{r} B A\tag{1}
$$

where $W _ { 0 }$ remains frozen during training and α denotes a constant scaling factor. One factor is zero-initialized, yielding $\varDelta W = 0$ at initialization.

A 2D convolutional layer is parameterized by a 4D weight tensor $W _ { 0 } \in$ $\mathbb { R } ^ { C _ { o u t } \times C _ { i n } \times k _ { h } \times k _ { u } }$ with $C _ { o u t }$ output channels, $C _ { i n }$ input channels, and spatial kernel size $k _ { h } \times k _ { w }$ . LoRA is formulated for matrix multiplication, so a naive extension to convolution reshapes the kernel tensor into a matrix. This reshaping merges the spatial dimensions into the input-channel axis and produces the flattened weight $\mathbf { \bar { \boldsymbol { W } } } _ { 0 } ^ { \flat } \in \mathbb { R } ^ { C _ { o u t } \times ( C _ { i n } k _ { h } k _ { w } ) }$ . The low-rank update is computed in the flattened space by applying Eq. (1) to $W _ { 0 } ^ { \flat }$

$$
W ^ {\prime b} = W _ {0} ^ {b} + \frac {\alpha}{r} B A,\tag{2}
$$

where $\boldsymbol { B } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times r }$ and $A \in \mathbb { R } ^ { r \times ( C _ { i n } k _ { h } k _ { w } ) }$

Filter Subspace Adaptation. Filter Subspace Fine-tuning (FSF) was proposed to represent each convolution filter as a linear combination of spatial elements referred to as filter atoms. Formally, Chen et al. [5] decompose a pre-trained convolutional layer $W _ { 0 } \in \mathbb { R } ^ { C _ { o u t } \times C _ { i n } \times k _ { h } \times k _ { w } }$ into a filter atom layer $\mathbf { \bar { D } } \in \mathbb { R } ^ { m \times k _ { h } \times k _ { w } }$ and an atom coeficient layer $\pmb { \alpha } \in \mathbb { R } ^ { C _ { o u t } \times C _ { i n } \times m }$ :

$$
W _ {0} = \boldsymbol {\alpha} \times \mathbf {D}.\tag{3}
$$

![](images/63081f342898dc2f891e1cfe9fb7fd82baf117bc9e6b62b02da6e130d045ca05.jpg)  
Fig. C: Convolutional kernel adaptation architectures. (a) LoRA: Decomposes weights into a frozen $W _ { 0 }$ and a low-rank update $\varDelta W = B A$ . (b) FSF: Decomposes $W _ { 0 }$ into spatial atom bases $( D _ { 1 } )$ , intra-channel mixing components (β), and cross-channel atom coeficients (α). (c) LoCA (ours): Learns only ∆W while freezing $W _ { 0 }$ , integrating spatial information via learned basis S for stable channel–spatial structural adaptation.

This indicates that each filter slice $W _ { 0 } ^ { i , j } \in \mathbb { R } ^ { k _ { h } \times k _ { w } }$ is constructed by a linear combination of the filter atoms: $\begin{array} { r } { W _ { 0 } ^ { i , j } = \sum _ { l = 1 } ^ { m } \alpha ^ { i , j , l } \mathbf { d } _ { l } . \ \mathbf { D } = \{ \mathbf { d } _ { l } \} _ { l = 1 } ^ { m } } \end{array}$ denotes a set of m filter atoms for spatial convolution and α controls spatially invariant cross-channel mixing. Sparse coding initializes these components by minimizing reconstruction error on the pre-trained weights. Each filter atom $\mathbf { d } _ { l }$ can be recursively decomposed into an overcomplete atom set $\mathbf { D } _ { 1 } \in \mathbb { R } ^ { ( m \cdot m _ { 1 } ) \times k _ { h } \times k _ { w } }$ using intra-channel mixing coeficients $\beta \in \bar { \mathbb { R } } ^ { ( C _ { i n } \cdot m ) \times m _ { 1 } }$ . FSF updates only the spatial atoms (D or $\mathbf { D } _ { 1 } )$ while freezing α to retain pre-trained generalization.

## 4 Low-Rank Convolutional Adaptation

This section introduces LoCA as a framework that preserves spatial inductive bias during convolutional layer adaptation. LoCA decouples the adaptation into low-rank channel adaptation (Sec. 4.1) and spatial basis refinement (Sec. 4.2). The low-rank channel adaptation isolates dense cross-channel mixing to resolve spatial–channel entanglement, while spatial basis refinement optimizes SVDderived structural bases to preserve the inherent spatial topology. The independent channel and spatial paths are then composed to form the decoupled LoCA design (Sec. 4.3). Finally, we introduce hierarchical rank scheduling for convolutional vision backbones (Sec. 4.4).

## 4.1 Low-Rank Channel Adaptation

To explicitly resolve the spatial-channel entanglement, we establish a dedicated low-rank channel adaptation mechanism that isolates dense cross-channel dependencies from spatial topology. Building upon the flattened formulation in Eq. (2), we define the low-rank channel adaptation term. Let $r _ { \mathrm { c h } }$ denote the channel rank and α the scaling factor. The update is defined as:

$$
\varDelta W _ {\mathrm{ch}} ^ {\flat} = \frac {\alpha}{r _ {\mathrm{ch}}} B _ {c} A _ {c},\tag{4}
$$

where $B _ { c } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times r _ { \mathrm { c h } } }$ and $A _ { c } \in \mathbb { R } ^ { r _ { \mathrm { c h } } \times ( C _ { \mathrm { i n } } k _ { h } k _ { w } ) }$ . We then reshape $\varDelta W _ { \mathrm { c h } } ^ { \flat }$ back to its original 4D spatial structure, denoted as $\varDelta W _ { \mathrm { c h } } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times k _ { h } \times k _ { w } }$ . To guarantee functional equivalence to the pre-trained model at initialization, $A _ { c }$ is initialized using a Kaiming uniform distribution, and $B _ { c }$ is initialized to zero. This zero-initialization ensures $\varDelta W _ { \mathrm { c h } } = 0$ at the start of training, preserving the original pre-trained representations without requiring kernel replacement.

## 4.2 SVD-based Spatial Basis Refinement

Naive flattening collapses the $k _ { h } \times k _ { w }$ spatial structure, making it dificult to isolate spatial modulation from dense channel transformations. We instead parameterize spatial adaptation with a compact set of pre-trained spatial bases. In practice, we reshape each pre-trained kernel slice into a leng $\cdot \mathrm { h } { - } k _ { h } k _ { w }$ vector and apply zero-mean and unit-variance standardization to obtain $W _ { \mathrm { n o r m } } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } C _ { \mathrm { i n } } \times \bar { k _ { h } } \bar { k } _ { w } }$ We then form a spatial covariance matrix $C _ { \mathrm { s p } } = W _ { \mathrm { n o r m } } ^ { \top } W _ { \mathrm { n o r m } } \in \mathbb { R } ^ { k _ { h } k _ { w } \times k _ { h } k _ { w } }$ Specifically, we compute the SVD of the spatial covariance induced by the pretrained kernel, set the spatial rank as $r _ { \mathrm { s p } } = k _ { h } k _ { w }$ , and obtain an initial basis tensor $S \in \mathbb { R } ^ { r _ { \mathrm { s p } } \times k _ { h } \times k _ { w } }$ . The deterministic basis S is initialized from this SVD and treated as a learnable parameter for refinement. Each slice $S _ { m } \in \mathbb { R } ^ { k _ { h } \times k _ { w } }$ corresponds to the m-th principal spatial pattern, such as edges or textures. Channelspecific coeficients are learned with $U _ { \mathrm { d w } } \in \mathbb { R } ^ { D \times r _ { \mathrm { s p } } }$ where $D = \operatorname* { m i n } ( C _ { \mathrm { o u t } } , C _ { \mathrm { i n } } )$ The spatial update $\varDelta W _ { \mathrm { s p } } ^ { \mathrm { d i a g } } [ i ] \in \mathbb { R } ^ { k _ { h } \times k _ { w } }$ for channel i is defined by a basis expansion:

$$
\varDelta W _ {\mathrm{sp}} ^ {\mathrm{diag}} [ i ] = \sum_ {m = 1} ^ {r _ {\mathrm{sp}}} U _ {\mathrm{dw}} [ i, m ] \cdot \mathcal {S} _ {m}.\tag{5}
$$

The spatial tensor is defined on the depthwise diagonal with the Kronecker delta $\delta _ { i j } \ ( \mathrm { i . e . , } \ \delta _ { i j } = 1 \ \mathrm { i f } \ i = j$ and 0 otherwise). Since $i = j$ implicitly guarantees $i \leq \mathrm { m i n } ( C _ { \mathrm { o u t } } , C _ { \mathrm { i n } } ) = D$ , the index is safely bounded:

$$
\varDelta W _ {\mathrm{sp}} [ i, j,:,: ] = \delta_ {i j} \varDelta W _ {\mathrm{sp}} ^ {\mathrm{diag}} [ i ].\tag{6}
$$

Diagonal parameterization separates spatial refinement from cross-channel mixing.

## 4.3 Channel-Spatial Composition

We compose the independent channel and spatial paths to synthesize our decoupled design into a unified adaptation framework without distorting pre-trained knowledge. The combined update tensor $\varDelta W _ { c s } \in \mathbb { R } ^ { C _ { \mathrm { o u t } } \times C _ { \mathrm { i n } } \times k _ { h } \times k _ { w } }$ adds the spatial update on the depthwise diagonal and uses $\varDelta W _ { \mathrm { c h } }$ for cross-channel mixing:

$$
\varDelta W _ {c s} [ i, j,:,: ] = \varDelta W _ {\mathrm{ch}} [ i, j,:,: ] + \delta_ {i j} \varDelta W _ {\mathrm{sp}} ^ {\mathrm{diag}} [ i ]\tag{7}
$$

Initializing $U _ { \mathrm { d w } }$ and $B _ { c }$ to zero ensures $\varDelta W _ { c s } = 0 $ , thereby preserving exact functional equivalence to the frozen pre-trained model. The final adapted weight $W ^ { \prime }$ is defined as follows:

$$
W ^ {\prime} = W _ {0} + \varDelta W _ {c s}.\tag{8}
$$

## 4.4 Hierarchical Rank Scheduling

Convolution-based VFMs adopt a hierarchical architecture that encodes features through progressively increasing channel dimensions across stages [36,43]. Early stages extract local features with narrower channels, while deeper stages encode global semantics with wider channels. A fixed rank applied uniformly across stages limits the ability to capture this hierarchical diversity. To address this, we introduce hierarchical rank scheduling to determine the channel rank $r _ { \mathrm { c h } } ^ { ( s ) }$ based on the stage-specific width $C _ { \mathrm { o u t } } ^ { ( s ) }$ . We compute a base rank as $\lfloor R \cdot \frac { C _ { \mathrm { o u t } } ^ { ( s ) } } { \sum _ { i } C _ { \mathrm { o u t } } ^ { ( i ) } } \rfloor ,$ where R is the global rank budget. By aligning adaptation capacity with layer width, this strategy ensures that the narrower stem receives smaller ranks while deeper layers are allocated larger ranks to model complex semantic features. While the channel rank is scaled across stages, the spatial rank $r _ { \mathrm { s p } }$ (as defined in Sec. 4.2) remains fixed to the localized spatial kernel size $\left( k _ { h } k _ { w } \right)$ . This strategy improves parameter eficiency without sacrificing adaptation performance. Empirical validation is provided in Sec. 5.4.

## 5 Experiment

In this section, we evaluate LoCA’s efectiveness through experiments across three distinct tasks: (1) fine-grained visual adaptation (Sec. 5.1) using the VTAB-1k and FGVC datasets; (2) generative generalization performance (Sec. 5.2) via the DreamBooth dataset; and (3) domain generalized semantic segmentation (DGSS).

## 5.1 Fine-grained Adaptation on VTAB-1k and FGVC

Experimental Setup. Our evaluation utilizes the FGVC and VTAB-1k benchmarks. The FGVC comprises four fine-grained recognition tasks: CUB-200-2011 [41], Stanford Dogs [23], Stanford Cars [25], and NABirds [40]. The VTAB-1k benchmark partitions downstream tasks into Natural, Specialized, and Structured semantic domains.

Results. We compare the proposed LoCA with existing PEFT methods on VTAB-1k and FGVC benchmarks. As shown in Tab. A, we report LoCA results on ConvNeXt-B and ResNet-50. On VTAB-1k, rank-16 (r16) generally achieves the highest accuracy, while ConvNeXt-B exhibits only marginal gains from increasing the rank. The marginal gains on ConvNeXt-B stem from the smaller dimensionality of convolutional kernels compared with linear layers. LoCA achieves the best VTAB-1k average accuracy with rank-4 on ConvNeXt-B and rank-16 on ResNet-50. On ConvNeXt-B, LoCA reaches this performance with only 0.97 M trainable parameters. In Tab. B, LoCA with rank-16 outperforms the other PEFT methods on FGVC using both ConvNeXt-B and ResNet-50. LoRA applies low-rank approximation after kernel flattening, which increases the number of additional parameters to 17.58 M on ConvNeXt-B while yielding lower accuracy. Due to architectural diferences between ConvNeXt (Conv. Seq.) and ResNet (Res. Par.), we use sequential CA insertion for ConvNeXt and parallel $k \times k$ CA adaptation for ResNet, following the best-performing placements reported in Conv-Adapter [4].

Table A: Performance comparison on the VTAB-1k visual classification benchmark using ConvNeXt-B and ResNet-50 backbones. We compare LoCA (in blue) with full Fine-Tuning (FFT) (in gray), Linear Probing (LP), Partial [45], MLP [20], Bias Tuning (Bias) [2], Visual Prompt Tuning (VPT) [22], Conv-Adapter [4], and Filter Subspace Fine-Tuning (FSF) [5]. Param. represents the trainable parameters (M).

<table><tr><td rowspan="2">Tuning</td><td rowspan="2">Param.</td><td colspan="4">ConvNeXt-B</td><td rowspan="2">Param.</td><td colspan="4">ResNet-50</td></tr><tr><td colspan="4">Natural Specialized Structured Average</td><td colspan="4">Natural Specialized Structured Average</td></tr><tr><td># Tasks</td><td>-</td><td>7</td><td>4</td><td>8</td><td>19</td><td>-</td><td>7</td><td>4</td><td>8</td><td>19</td></tr><tr><td>FFT</td><td>87.62</td><td>80.52</td><td>87.54</td><td>63.85</td><td>74.98</td><td>23.61</td><td>65.58</td><td>82.0</td><td>52.32</td><td>63.45</td></tr><tr><td>LP</td><td>1.68</td><td>74.48</td><td>81.50</td><td>34.76</td><td>59.23</td><td>0.48</td><td>63.75</td><td>77.60</td><td>30.96</td><td>52.89</td></tr><tr><td>Partial-1 [45]</td><td>4.72</td><td>73.76</td><td>81.64</td><td>39.55</td><td>61.01</td><td>2.10</td><td>64.34</td><td>78.64</td><td>45.78</td><td>59.51</td></tr><tr><td>MLP-3 [20]</td><td>2.45</td><td>73.78</td><td>81.36</td><td>35.68</td><td>59.33</td><td>3.51</td><td>61.79</td><td>70.77</td><td>33.97</td><td>51.97</td></tr><tr><td>Bias [2]</td><td>1.76</td><td>69.07</td><td>72.81</td><td>25.29</td><td>51.42</td><td>0.49</td><td>63.51</td><td>77.22</td><td>33.39</td><td>53.85</td></tr><tr><td>VPT [22]</td><td>1.75</td><td>78.48</td><td>83.00</td><td>44.64</td><td>65.18</td><td>0.49</td><td>66.25</td><td>77.32</td><td>37.52</td><td>56.09</td></tr><tr><td>LoRA [21]</td><td>17.32</td><td>80.89</td><td>86.80</td><td>62.46</td><td>74.36</td><td>1.90</td><td>65.06</td><td>82.53</td><td>56.21</td><td>65.01</td></tr><tr><td>CA [4]</td><td>6.83</td><td>80.62</td><td>86.29</td><td>64.88</td><td>75.18</td><td>1.37</td><td>64.20</td><td>81.33</td><td>52.74</td><td>64.78</td></tr><tr><td>CoLoRA [35]</td><td>4.57</td><td>76.1</td><td>83.1</td><td>58.2</td><td>70.0</td><td>1.40</td><td>66.6</td><td>82.6</td><td>51.9</td><td>63.8</td></tr><tr><td>FSF [5]</td><td>1.11</td><td>82.96</td><td>85.53</td><td>59.59</td><td>73.59</td><td>0.72</td><td>62.64</td><td>80.25</td><td>36.50</td><td>56.91</td></tr><tr><td>LoCA (r4)</td><td>0.97</td><td>82.22</td><td>87.54</td><td>64.74</td><td>75.98</td><td>0.47</td><td>66.08</td><td>82.41</td><td>52.44</td><td>63.77</td></tr><tr><td>LoCA (r16)</td><td>5.01</td><td>81.81</td><td>87.51</td><td>65.01</td><td>75.93</td><td>1.51</td><td>67.21</td><td>83.61</td><td>54.52</td><td>65.32</td></tr></table>

Table B: Average Top-1 accuracy (%) on FGVC datasets

<table><tr><td rowspan="2">Tuning</td><td colspan="2">ConvNeXt-B</td><td colspan="2">ResNet-50</td></tr><tr><td># Param.</td><td>Average</td><td># Param.</td><td>Average</td></tr><tr><td>FFT</td><td>87.87</td><td>79.73</td><td>24.14</td><td>75.73</td></tr><tr><td>LP</td><td>0.31</td><td>77.55</td><td>0.62</td><td>45.39</td></tr><tr><td>Bias [2]</td><td>0.44</td><td>64.98</td><td>0.67</td><td>48.85</td></tr><tr><td>LoRA [21]</td><td>17.58</td><td>88.18</td><td>2.43</td><td>80.96</td></tr><tr><td>CA [4]</td><td>6.13</td><td>89.28</td><td>2.23</td><td>83.48</td></tr><tr><td>CoLoRA [35]</td><td>4.57</td><td>86.11</td><td>0.99</td><td>76.44</td></tr><tr><td>FSF [5]</td><td>1.11</td><td>88.04</td><td>2.52</td><td>81.07</td></tr><tr><td>LoCA (r16)</td><td>3.70</td><td>90.06</td><td>1.94</td><td>83.67</td></tr></table>

Table C: Quantitative comparison of subject alignment

<table><tr><td>Methods</td><td>DINO↑</td><td>CLIP-I↑</td><td>CLIP-T↑</td></tr><tr><td>Pretrained</td><td>0.320</td><td>0.643</td><td>0.267</td></tr><tr><td>Real Images</td><td>0.711</td><td>0.857</td><td>-</td></tr><tr><td>Textual Inversion [13]</td><td>0.564</td><td>0.739</td><td>0.213</td></tr><tr><td>DreamBooth [37]</td><td>0.642</td><td>0.794</td><td>0.236</td></tr><tr><td>LoRA [21]</td><td>0.637</td><td>0.792</td><td>0.239</td></tr><tr><td> $\leftrightarrow$  w/ Conv</td><td>0.707</td><td>0.801</td><td>0.278</td></tr><tr><td>FSF [5]</td><td>0.572</td><td>0.715</td><td>0.313</td></tr><tr><td>LoCA (r16)</td><td>0.709</td><td>0.801</td><td>0.280</td></tr></table>

Generalization across Backbones. The performance of LoCA is evaluated

against FFT on convolutionbased VFM backbones, including ResNet, ConvNeXt, MambaVision, EficientNet, and MobileMamba. All experiments are conducted using base models. Fig. D shows that LoCA outperforms FFT across diverse

![](images/4d044a7f312015ca9a8634d2def9de9e1a1629c3266447246eb8ba318e142905.jpg)  
Fig. D: Performance comparison with FFT across various backbone architectures

backbones. On MobileMamba, LoCA improves Top-1 accuracy over FFT, increasing it from 70.6% to 78.9%. The results demonstrate consistent gains across architectures that incorporate convolution operations. Detailed results are provided in Appendix B.

![](images/d9f6d93d900896d8b9c81bb4d2805e10d658ff987b20409d4f65cc534262385b.jpg)  
Fig. E: Qualitative results of subject-driven task. We visualize the results to compare PEFT methods: LoRA [21], FSF [5], and LoCA.

## 5.2 Generative Generalization with DreamBooth

Experimental Setup. For the generative task, comparative experiments involving DreamBooth [37], LoRA [21], and FSF [5] all utilize Stable Difusion v1.4 [36] under identical training configurations. Performance analysis follows the DreamBooth evaluation protocol by using images generated from 25 prompts. Quantitative assessment focuses on text alignment, where DINO [3] and CLIP-I [34] measure subject fidelity while CLIP-T [34] evaluates text prompt fidelity. CLIP-I and DINO calculate the average cosine similarity between the embeddings of generated and ground truth images based on CLIP and ViT-S/16-based DINO, respectively. Similarly, CLIP-T computes the average cosine similarity between the text prompt and the generated image embeddings.

Results. We evaluate LoCA against PEFT approaches on subject-driven generation. Tab. C shows that LoCA achieves strong subject alignment and competitive text alignment. LoCA outperforms LoRA by achieving the best DINO score and tying for the best CLIP-I score, while maintaining a competitive CLIP-T score. Convolution-based adaptation incorporates spatial inductive biases for subject alignment, whereas LoRA applies low-rank updates to linear projections.

Visualization. Fig. E shows that LoCA better retains the structural identity of the reference subject. For qualitative evaluation, we visualize results from LoRA [21], FSF [5], and LoCA across four classes. LoRA preserves subject identity but demonstrates limited reflection of textual attributes (e.g., ‘chef outfit’ or ‘city in the background’). FSF [5] captures textual attributes but struggles to preserve subject identity and capture fine-grained visual details. For the vase-class prompt ‘a [V] floating on top of water’, FSF generates a bulky bottle-like object rather than the intended subject. In contrast, LoCA preserves subject identity while better reflecting textual attributes. Detailed qualitative comparisons appear in Appendix C.

Table D: Performance comparison on synthetic-to-real DGSS using various backbones and model sizes. Models are trained on GTAV and evaluated on Cityscapes, BDD100K, and Mapillary.

<table><tr><td>Method</td><td>Backbone</td><td colspan="3">Param. Trainable Param. GFLOPs</td><td>Citys.</td><td>BDD</td><td>Map.</td><td>Avg.</td></tr><tr><td colspan="9">DINO Pre-trained</td></tr><tr><td>FFT</td><td>ViT-B</td><td>86.5M</td><td>86.5M</td><td>216</td><td>60.84</td><td>52.98</td><td>62.12</td><td>58.65</td></tr><tr><td>SoMA</td><td>ViT-B</td><td>86.5M</td><td>2.3M</td><td>216</td><td>66.71</td><td>57.48</td><td>67.34</td><td>63.84</td></tr><tr><td>FFT</td><td>ConvNeXt-B</td><td>87.56M</td><td>87.56M</td><td>81</td><td>62.18</td><td>57.01</td><td>65.00</td><td>61.40</td></tr><tr><td>LoRA (Linear)</td><td>ConvNeXt-B</td><td>87.56M</td><td>2.9M</td><td>81</td><td>63.90</td><td>57.87</td><td>65.53</td><td>62.43</td></tr><tr><td>LoRA</td><td>ConvNeXt-B</td><td>87.56M</td><td>17.2M</td><td>81</td><td>64.17</td><td>56.98</td><td>65.74</td><td>62.30</td></tr><tr><td>SoMA</td><td>ConvNeXt-B</td><td>87.56M</td><td>2.9M</td><td>81</td><td>64.99</td><td>57.65</td><td>65.67</td><td>62.77</td></tr><tr><td>CA</td><td>ConvNeXt-B</td><td>87.56M</td><td>2.3M</td><td>81</td><td>59.94</td><td>56.47</td><td>63.48</td><td>59.96</td></tr><tr><td>CoLoRA</td><td>ConvNeXt-B</td><td>87.56M</td><td>2.2M</td><td>81</td><td>61.87</td><td>55.70</td><td>64.04</td><td>60.53</td></tr><tr><td>FSF</td><td>ConvNeXt-B</td><td>87.56M</td><td>0.6M</td><td>81</td><td>60.15</td><td>56.98</td><td>63.70</td><td>60.27</td></tr><tr><td>LoCA</td><td>ConvNeXt-B</td><td>87.56M</td><td>3.4M</td><td>81</td><td>66.46</td><td>58.53</td><td>66.29</td><td>63.76</td></tr><tr><td> $LoCA^‡$ </td><td>ConvNeXt-B</td><td>87.56M</td><td>3.4M</td><td>81</td><td>65.56</td><td>58.02</td><td>66.39</td><td>63.32</td></tr><tr><td>FFT</td><td>ConvNeXt-L</td><td>196.2M</td><td>196.2M</td><td>152</td><td>65.50</td><td>59.10</td><td>67.01</td><td>63.87</td></tr><tr><td>LoRA (Linear)</td><td>ConvNeXt-L</td><td>196.2M</td><td>4.3M</td><td>152</td><td>66.95</td><td>60.46</td><td>68.45</td><td>65.29</td></tr><tr><td>LoRA</td><td>ConvNeXt-L</td><td>196.2M</td><td>25.9M</td><td>152</td><td>65.74</td><td>60.50</td><td>68.62</td><td>64.95</td></tr><tr><td>SoMA</td><td>ConvNeXt-L</td><td>196.2M</td><td>4.3M</td><td>152</td><td>68.85</td><td>60.27</td><td>69.26</td><td>66.13</td></tr><tr><td>CA</td><td>ConvNeXt-L</td><td>196.2M</td><td>4.5M</td><td>152</td><td>63.16</td><td>58.41</td><td>67.32</td><td>62.96</td></tr><tr><td>CoLoRA</td><td>ConvNeXt-L</td><td>196.2M</td><td>3.6M</td><td>152</td><td>64.51</td><td>59.56</td><td>67.20</td><td>63.59</td></tr><tr><td>FSF</td><td>ConvNeXt-L</td><td>196.2M</td><td>0.8M</td><td>152</td><td>66.57</td><td>58.52</td><td>66.96</td><td>64.02</td></tr><tr><td>LoCA</td><td>ConvNeXt-L</td><td>196.2M</td><td>5.0M</td><td>152</td><td>68.54</td><td>61.60</td><td>69.33</td><td>66.49</td></tr><tr><td> $LoCA^‡$ </td><td>ConvNeXt-L</td><td>196.2M</td><td>5.0M</td><td>152</td><td>69.73</td><td>62.03</td><td>70.62</td><td>67.46</td></tr><tr><td colspan="9">ImageNet21k Pre-trained</td></tr><tr><td>FFT</td><td>ResNet101</td><td>42.3M</td><td>42.3M</td><td>42</td><td>41.29</td><td>44.29</td><td>48.79</td><td>44.79</td></tr><tr><td> $SoMA^†$ </td><td>ResNet101</td><td>42.3M</td><td>2.5M</td><td>42</td><td>41.23</td><td>45.57</td><td>49.71</td><td>45.50</td></tr><tr><td>LoCA</td><td>ResNet101</td><td>42.3M</td><td>2.8M</td><td>42</td><td>44.82</td><td>46.13</td><td>49.21</td><td>46.72</td></tr><tr><td>FFT</td><td>MambaVision-B</td><td>96.7M</td><td>96.7M</td><td>211</td><td>36.05</td><td>30.13</td><td>31.39</td><td>32.52</td></tr><tr><td>LoCA</td><td>MambaVision-B</td><td>96.7M</td><td>2.5M</td><td>211</td><td>45.21</td><td>41.68</td><td>45.70</td><td>44.20</td></tr><tr><td>FFT</td><td>MambaVision-L3</td><td>737.5M</td><td>737.5M</td><td>1,556</td><td>52.88</td><td>45.87</td><td>56.07</td><td>51.61</td></tr><tr><td>LoCA</td><td>MambaVision-L3</td><td>737.5M</td><td>9.0M</td><td>1,556</td><td>59.94</td><td>50.61</td><td>61.39</td><td>57.31</td></tr></table>

<sup>†</sup> PEFT via linearization of patch-level convolutions and weights.  
<sup>‡</sup> Channel mixing path initialization follows the SVD-based approach of SoMA [47].

## 5.3 Domain Generalization for Semantic Segmentation

Experimental Setup. Experiments utilize convolution-based backbones, specifically ResNet [19], and ConvNeXt [43]. Evaluation further includes recent vision foundation models with hybrid transformer–convolution architectures, including DINOv3-ConvNeXt [38] and MambaVision [17]. All models employ a consistent Mask2Former [7] decoder for DGSS and DGOD tasks. A fixed rank of 16 across all experiments ensures a fair comparison. For DGSS and DGOD tasks, models are trained on the GTAV dataset and evaluated on three real-world benchmarks: Cityscapes [9], BDD100K [46], and Mapillary [32]. Appendix D presents detailed experimental settings and performance analysis for diferent backbones.

![](images/9760c34273f4b3dc57fb9739332bcd3c2de299c62795f6898e837c07ef659d03.jpg)  
Fig. F: Qualitative domain adaptation results (GTAV → Cityscapes)

Results. As shown in Tab. D, we evaluate the generalization capability of LoCA on DGSS by training on GTAV and testing on real-world out-of-distribution (OOD) benchmarks. LoCA with DINO-pretrained ConvNeXt backbones [38] demonstrates competitive performance compared to other PEFT methods on OOD datasets. LoRA-based adaptation flattens convolution kernels and requires learning a large number of parameters. LoCA achieves strong performance with substantially fewer trainable parameters than LoRA. Initializing low-rank components with the principal singular components following SoMA boosts performance by approximately +1.0 on average across three datasets in ConvNeXt-L. Conversely, zero initialization outperforms this method for ConvNeXt-B. The gain observed in ConvNeXt-L suggests that larger models retain more salient eigenvalue structures from pre-trained weights. Benchmarking against SoMA results on ViT-B [12], we evaluated ConvNeXt-B with a comparable parameter count. ConvNeXt-B achieves performance parity with ViT-B at 81 GFLOPs, representing a 2.6× reduction from the 216 GFLOPs required by ViT-B. The ConvNeXt-B-based LoCA maintains high accuracy while mitigating computational overhead. For DGSS, models are trained on GTAV and evaluated on Cityscapes, BDD100K, and Mapillary. Detailed DGOD results are provided in Appendix D.

Visualization. Qualitative evaluations utilize ConvNeXt-B to compare LoRA [21], FSF [5], and LoCA on night clear, foggy, and rainy Cityscapes scenes [9]. These challenging environments serve as a benchmark for robustness. As shown in Fig. F, our method generates fine-grained segmentation maps in both rainy and foggy scenes. LoCA preserves sharp object boundaries and captures fine-grained details in the boxed regions. LoCA exhibits stronger robustness to weather-induced noise than LoRA and FSF.

Generalization across Backbones. Recent vision foundation models adopt convolutional architectures such as Mamba-based designs [14]. We adapt convolutional components during fine-tuning of MambaVision [17], ResNet [19], and

![](images/8a17c023dcdcb01d9776ec7f6ec3582b22d5d1361f29f393505e9c83f7c209c7.jpg)  
Fig. G: Singular value evolution of ConvNeXt-B depthwise convolution weights across three PEFT methods: LoRA, FSF, and LoCA. The y-axis denotes singular values. Columns 1–4: channel-wise SVD of depthwise weights; columns 5–8: spatial SVD. Stable distributions prevent spectral tail growth and ensure controlled efective rank behavior.

ConvNeXt [28]. On MambaVision-B, LoCA improves the average score from 32.52 to 44.20, corresponding to +11.68 points or a 35.9% relative gain over FFT. Tab. D further shows that LoCA achieves competitive performance while using only about 2.5% of the parameters.

## 5.4 Ablation Studies

Evolution of Representational Capacity. The evolution of singular value spectra during training reflects representational capacity during adaptation. A distributed singular value spectrum can indicate efective representation learning, with information distributed across multiple components rather than concentrated in a few dominant ones [48]. To analyze this spectral evolution, we track the singular value spectra of a ConvNeXt-B depthwise convolution weight matrix (dwconv) on VTAB-1k Oxford Pets. As shown in Fig. G, LoRA and FSF exhibit limited variation in their basis components across training epochs. In contrast, LoCA exhibits progressively diverse basis components with smooth singular value growth across both leading and trailing components. This spectral evolution reflects kernel importance identified through covariance analysis and enables joint channel–spatial adaptation.

Spatial Representation Analysis. We validate the efectiveness of LoCA in convolutional adaptation by visualizing the orthogonality and spatial coverage of rank components. Fig. H shows the cosine similarity among rank components and their spatial coverage. LoRA rank components exhibit higher similarity and tend to collapse into redundant feature directions. In contrast, LoCA produces more orthogonal rank components, which encourages diverse subspace representations and improves convolutional expressivity. The increased diversity of rank components becomes more evident at the patch level. The highly activated patches in LoCA are distributed across diverse spatial regions rather than localized to a single spatial cue. LoRA focuses on localized regions, whereas LoCA learns orthogonal rank components with broader spatial coverage.

![](images/24101ca00c49c0296d8ef80532de926f118eee5b81e49dcdb5db14fbb6eb2ac4.jpg)  
Fig. H: Absolute pairwise cosine similarity and top-activated localized patches [49] of trained rank components across stages (s0–3). Localized patches denote image regions with the strongest activation responses for each rank component.

Ablation Analysis of Proposed Methods. We analyze the contribution of each proposed method relative to a LoRA baseline. The analysis compares Table E: Ablation study of proposed methods un-incremental variants, includ- incremental variants, includder DGSS benchmarksing standard convolutional ing standard convolutional

LoRA, channel mixing, spatial basis refinement, and hierarchical rank scheduling. As shown in Tab. E, applying the standard convolutional LoRA formulation decreases the av-

<table><tr><td>Tuning Method</td><td>Citys.</td><td>BDD</td><td>Map.</td><td>Avg.</td></tr><tr><td>LoRA Linear</td><td>66.95</td><td>60.46</td><td>68.45</td><td>65.29</td></tr><tr><td> $\sqcup$  LoRA Convolution</td><td>65.74  $\nabla 1.21$ </td><td>60.50  $\triangle 0.04$ </td><td>68.62  $\triangle 0.17$ </td><td>64.95  $\nabla 0.34$ </td></tr><tr><td> $\sqcup$  Channel Mixing</td><td>68.22  $\triangle 1.27$ </td><td>61.12  $\triangle 0.66$ </td><td>69.13  $\triangle 0.68$ </td><td>66.16  $\triangle 0.87$ </td></tr><tr><td> $\sqcup$  Spatial Basis</td><td>69.44  $\triangle 2.49$ </td><td>61.39  $\triangle 0.93$ </td><td>70.26  $\triangle 1.81$ </td><td>67.03  $\triangle 1.74$ </td></tr><tr><td> $\sqcup$  Hierarchical Rank</td><td>69.73  $\triangle 2.78$ </td><td>62.03  $\triangle 1.57$ </td><td>70.62  $\triangle 2.17$ </td><td>67.46  $\triangle 2.17$ </td></tr></table>

erage score by 0.34 points. Naively applying LoRA to convolutional kernels does not improve adaptation performance and may degrade OOD performance. In contrast, the proposed channel mixing increases the average score to 66.16, yielding a +0.87 gain over LoRA Linear. Adding spatial basis refinement further improves the average score to 67.03, yielding a +1.74 gain over LoRA Linear. Hierarchical rank scheduling achieves the best average score of 67.46, corresponding to a +2.17 gain over LoRA Linear. Convolutional architectures encode coarse-to-fine information across stages. Hierarchical rank scheduling aligns the adaptation capacity with this structural property.

Analysis of Covariance-SVD Initialization. We compare covariance-SVD initialization with other initialization strategies on VTAB-1k and DGSS data. Across diverse tasks, covariance-SVD initialization consistently outperforms other initialization strategies. The Table F: Ablation study on initialization strategies consistent gains suggest that Initialization Zero Flatten SVD Uniform Covariance preserving the directional struc- VTAB-1k Avg. 75.7 73.0 75.7 75.9 ture of convolutional feature DGSS Avg. 65.6 66.2 66.3 66.5 subspaces benefits representation learning. Covariance-SVD preserves pretrained spatial priors because SVD is applied to spatial covariance rather than a flattened convolution tensor. Covariance-SVD performs best on both VTAB-1k and DGSS (Tab. F).

## 6 Conclusions

Although LoRA is the dominant PEFT approach, convolutional adaptation remains underexplored despite the centrality of spatial information to visual adaptation. To this end, we present Low-Rank Convolutional Adaptation (LoCA), a convolution-aware PEFT framework for adapting vision foundation models. LoCA decouples the adaptation into low-rank channel adaptation and spatial basis refinement. This convolution-aware PEFT framework addresses spatialchannel entanglement by decoupling channel and spatial adaptation while preserving pre-trained spatial priors. Furthermore, our hierarchical rank scheduling aligns adaptation capacity with the backbone’s hierarchical feature extraction. Experimental results show that LoCA achieves strong performance across tasks and backbones, outperforming existing methods on several benchmarks and remaining competitive on others.

## Acknowledgements

This research was supported by the Institute of Information & Communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. RS-2019-II190079, Artificial Intelligence Graduate School Program (Korea University), 1%; No. RS-2025-25439490, 40%), Culture, Sports and Tourism R&D Program through the Korea Creative Content Agency grant funded by the Ministry of Culture, Sports and Tourism in 2024 (No. RS-2024- 00345025, International Collaborative Research and Global Talent Development for the Development of Copyright Management and Protection Technologies for Generative AI, 10%), the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (No. RS-2024-00341514, 39%), the Industrial Technology Innovation Program (No. RS-2025-25448266, Development of Humanoid Robots Specialized in Display Manufacturing Processes Based on AI Foundation Models, 10%) grant funded by the Korea government (MOTIE).

## References

1. Biderman, D., Portes, J., Ortiz, J.J.G., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., Cunningham, J.P.: LoRA learns less and forgets less. Transactions on Machine Learning Research (2024), https://openreview.net/forum?id=aloEru2qCG, featured Certification

2. Cai, H., Gan, C., Zhu, L., Han, S.: Tinytl: Reduce memory, not parameters for eficient on-device learning. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 11285–11297. Curran Associates, Inc. (2020), https://proceedings.neurips.cc/ paper\_files/paper/2020/file/81f7acabd411274fcf65ce2070ed568a-Paper.pdf

3. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)

4. Chen, H., Tao, R., Zhang, H., Wang, Y., Li, X., Ye, W., Wang, J., Hu, G., Savvides, M.: Conv-adapter: Exploring parameter eficient transfer learning for convnets. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops. pp. 1551–1561 (June 2024)

5. Chen, W., Miao, Z., Qiu, Q.: Large convolutional model tuning via filter subspace. In: The Thirteenth International Conference on Learning Representations (2025), https://openreview.net/forum?id=E5YmIBvOqV

6. Chen, Z., Duan, Y., Wang, W., He, J., Lu, T., Dai, J., Qiao, Y.: Vision transformer adapter for dense predictions. In: The Eleventh International Conference on Learning Representations (2023), https://openreview.net/forum?id=plKu2GByCNW

7. Cheng, B., Misra, I., Schwing, A.G., Kirillov, A., Girdhar, R.: Masked-attention mask transformer for universal image segmentation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 1290–1299 (June 2022)

8. Contributors, M.: MMSegmentation: Openmmlab semantic segmentation toolbox and benchmark. https://github.com/open-mmlab/mmsegmentation (2020)

9. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., Schiele, B.: The cityscapes dataset for semantic urban scene understanding. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2016)

10. Ding, C., Cao, X., Xie, J., Fan, L., Wang, S., Lu, Z.: Lora-c: Parameter-eficient fine-tuning of robust cnn for iot devices. arXiv preprint arXiv:2410.16954 (2024)

11. Ding, N., Lv, X., Wang, Q., Chen, Y., Zhou, B., Liu, Z., Sun, M.: Sparse lowrank adaptation of pre-trained language models. In: Bouamor, H., Pino, J., Bali, K. (eds.) Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing. pp. 4133–4145. Association for Computational Linguistics, Singapore (Dec 2023). https://doi.org/10.18653/v1/2023.emnlp-main.252, https://aclanthology.org/2023.emnlp-main.252/

12. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021), https://openreview. net/forum?id=YicbFdNTTy

13. Gal, R., Alaluf, Y., Atzmon, Y., Patashnik, O., Bermano, A.H., Chechik, G., Cohen-Or, D.: An image is worth one word: Personalizing text-to-image generation using textual inversion (2022). https://doi.org/10.48550/ARXIV.2208.01618, https://arxiv.org/abs/2208.01618

14. Gu, A., Dao, T.: Mamba: Linear-time sequence modeling with selective state spaces. In: First Conference on Language Modeling (2024), https://openreview. net/forum?id=tEYskw1VY2

15. Guo, D., Rush, A.M., Kim, Y.: Parameter-eficient transfer learning with dif pruning. In: Proceedings of the 59th annual meeting of the association for computational linguistics and the 11th international joint conference on natural language processing (volume 1: Long papers). pp. 4884–4896 (2021)

16. Han, Z., Gao, C., Liu, J., Zhang, J., Zhang, S.Q.: Parameter-eficient fine-tuning for large models: A comprehensive survey. Transactions on Machine Learning Research (2024), https://openreview.net/forum?id=lIsCS8b6zj

17. Hatamizadeh, A., Kautz, J.: Mambavision: A hybrid mamba-transformer vision backbone. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 25261–25270 (2025)

18. He, H., Zhang, J., Cai, Y., Chen, H., Hu, X., Gan, Z., Wang, Y., Wang, C., Wu, Y., Xie, L.: Mobilemamba: Lightweight multi-receptive visual mamba network. In: Proceedings of the computer vision and pattern recognition conference. pp. 4497– 4507 (2025)

19. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2016)

20. Houlsby, N., Giurgiu, A., Jastrzebski, S., Morrone, B., De Laroussilhe, Q., Gesmundo, A., Attariyan, M., Gelly, S.: Parameter-eficient transfer learning for NLP. In: Proceedings of the 36th International Conference on Machine Learning (2019)

21. Hu, E.J., yelong shen, Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., Chen, W.: LoRA: Low-rank adaptation of large language models. In: International Conference on Learning Representations (2022), https://openreview.net/forum?id= nZeVKeeFYf9

22. Jia, M., Tang, L., Chen, B.C., Cardie, C., Belongie, S., Hariharan, B., Lim, S.N.: Visual prompt tuning. In: Computer Vision – ECCV 2022. pp. 709–727. Springer, Springer Nature Switzerland, Cham (2022)

23. Khosla, A., Jayadevaprakash, N., Yao, B., Fei-Fei, L.: Novel dataset for fine-grained image categorization. In: First Workshop on Fine-Grained Visual Categorization, IEEE Conference on Computer Vision and Pattern Recognition. Colorado Springs, CO (June 2011)

24. Kim, D., Wang, K., Sclarof, S., Saenko, K.: A broad study of pre-training for domain generalization and adaptation. In: Computer Vision – ECCV 2022. pp. 621–638. Springer, Springer Nature Switzerland, Cham (2022)

25. Krause, J., Stark, M., Deng, J., Fei-Fei, L.: 3d object representations for finegrained categorization. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV) Workshops (June 2013)

26. Li, X.L., Liang, P.: Prefix-tuning: Optimizing continuous prompts for generation. In: Zong, C., Xia, F., Li, W., Navigli, R. (eds.) Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). pp. 4582–4597. Association for Computational Linguistics, Online (Aug 2021). https://doi.org/10.18653/v1/2021.acl-long.353, https://aclanthology.org/ 2021.acl-long.353/

27. Li, Y., Mao, H., Girshick, R., He, K.: Exploring plain vision transformer backbones for object detection. In: Avidan, S., Brostow, G., Cissé, M., Farinella, G.M., Hassner, T. (eds.) Computer Vision – ECCV 2022. pp. 280–296. Springer Nature Switzerland, Cham (2022)

28. Liu, Z., Mao, H., Wu, C.Y., Feichtenhofer, C., Darrell, T., Xie, S.: A convnet for the 2020s. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 11976–11986 (June 2022)

29. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019), https://openreview.net/forum? id=Bkg6RiCqY7

30. Luo, W., Li, Y., Urtasun, R., Zemel, R.: Understanding the efective receptive field in deep convolutional neural networks. In: Lee, D., Sugiyama, M., Luxburg, U., Guyon, I., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 29. Curran Associates, Inc. (2016), https://proceedings.neurips.cc/paper\_ files/paper/2016/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf

31. Meng, F., Wang, Z., Zhang, M.: Pissa: Principal singular values and singular vectors adaptation of large language models. In: Globerson, A., Mackey, L., Belgrave, D., Fan, A., Paquet, U., Tomczak, J., Zhang, C. (eds.) Advances in Neural Information Processing Systems. vol. 37, pp. 121038–121072. Curran Associates, Inc. (2024). https://doi.org/10.52202/079017- 3846, https://proceedings. neurips.cc/paper\_files/paper/2024/file/db36f4d603cc9e3a2a5e10b93e6428f2- Paper-Conference.pdf

32. Neuhold, G., Ollmann, T., Rota Bulo, S., Kontschieder, P.: The mapillary vistas dataset for semantic understanding of street scenes. In: Proceedings of the IEEE International Conference on Computer Vision (ICCV) (Oct 2017)

33. Prottasha, N.J., Chowdhury, U.R., Mohanto, S., Nuzhat, T., Sami, A.A., Ali, M.S., Sobuj, M.S.I., Raman, H., Kowsher, M., Garibay, O.O.: Peft a2z: parametereficient fine-tuning survey for large language and vision models. arXiv preprint arXiv:2504.14117 (2025)

34. Radford, A., Kim, J.W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., Sutskever, I.: Learning transferable visual models from natural language supervision. In: Meila, M., Zhang, T. (eds.) Proceedings of the 38th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 139, pp. 8748–8763. PMLR (18–24 Jul 2021), https://proceedings.mlr.press/v139/radford21a.html

35. Ran, W., Zhang, W., Pang, S., Zhu, Q., Liu, J., Liu, J., Cao, X., Li, Q., Yan, Y., Ma, C.: Correlated low-rank adaptation for convnets. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2026), https:// openreview.net/forum?id=3pF7rt9fQM

36. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10684– 10695 (June 2022)

37. Ruiz, N., Li, Y., Jampani, V., Pritch, Y., Rubinstein, M., Aberman, K.: Dreambooth: Fine tuning text-to-image difusion models for subject-driven generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 22500–22510 (June 2023)

38. Siméoni, O., Vo, H.V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., Bojanowski, P.: DINOv3 (2025), https://arxiv.org/abs/2508.10104

39. Szegedy, C., Liu, W., Jia, Y., Sermanet, P., Reed, S., Anguelov, D., Erhan, D., Vanhoucke, V., Rabinovich, A.: Going deeper with convolutions. In: Proceedings

of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2015)

40. Van Horn, G., Branson, S., Farrell, R., Haber, S., Barry, J., Ipeirotis, P., Perona, P., Belongie, S.: Building a bird recognition app and large scale dataset with citizen scientists: The fine print in fine-grained dataset collection. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 595–604 (2015)

41. Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S.: The caltech-ucsd birds-200-2011 dataset. Tech. rep., California Institute of Technology (2011)

42. Wei, Z., Chen, L., Jin, Y., Ma, X., Liu, T., Ling, P., Wang, B., Chen, H., Zheng, J.: Stronger fewer & superior: Harnessing vision foundation models for domain generalized semantic segmentation. In: CVPR (2024)

43. Woo, S., Debnath, S., Hu, R., Chen, X., Liu, Z., Kweon, I.S., Xie, S.: Convnext v2: Co-designing and scaling convnets with masked autoencoders. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 16133–16142 (June 2023)

44. Xu, L., Xie, H., Qin, S.J., Tao, X., Wang, F.L.: Parameter-eficient fine-tuning methods for pretrained language models: A critical review and assessment. IEEE Transactions on Pattern Analysis and Machine Intelligence pp. 1–20 (2026). https: //doi.org/10.1109/TPAMI.2026.3657354

45. Yosinski, J., Clune, J., Bengio, Y., Lipson, H.: How transferable are features in deep neural networks? In: Ghahramani, Z., Welling, M., Cortes, C., Lawrence, N., Weinberger, K. (eds.) Advances in Neural Information Processing Systems. vol. 27. Curran Associates, Inc. (2014), https://proceedings.neurips.cc/paper\_ files/paper/2014/file/532a2f85b6977104bc93f8580abbb330-Paper.pdf

46. Yu, F., Chen, H., Wang, X., Xian, W., Chen, Y., Liu, F., Madhavan, V., Darrell, T.: Bdd100k: A diverse driving dataset for heterogeneous multitask learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (June 2020)

47. Yun, S., Chae, S., Lee, D., Ro, Y.: Soma: Singular value decomposed minor components adaptation for domain generalizable representation learning. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 25602–25612 (June 2025)

48. Yunis, D., Patel, K.K., Wheeler, S., Savarese, P.H.P., Vardi, G., Frankle, J., Livescu, K., Maire, M., Walter, M.: Rank minimization, alignment and weight decay in neural networks. In: High-dimensional Learning Dynamics 2024: The Emergence of Structure and Reasoning (2024), https://openreview.net/forum? id=u3sssLLu4y

49. Zeiler, M.D., Fergus, R.: Visualizing and understanding convolutional networks. In: Fleet, D., Pajdla, T., Schiele, B., Tuytelaars, T. (eds.) Computer Vision – ECCV 2014. pp. 818–833. Springer International Publishing, Cham (2014)

50. Zhang, Q., Chen, M., Bukharin, A., He, P., Cheng, Y., Chen, W., Zhao, T.: Adaptive budget allocation for parameter-eficient fine-tuning. In: The Eleventh International Conference on Learning Representations (2023), https://openreview.net/ forum?id=lq62uWRJjiY

51. Zhang, Y., Zhou, K., Liu, Z.: Neural prompt search. IEEE Transactions on Pattern Analysis and Machine Intelligence 47(7), 5268–5280 (2025). https://doi.org/10. 1109/TPAMI.2024.3435939

52. Zhong, Z., Tang, Z., He, T., Fang, H., Yuan, C.: Convolution meets loRA: Parameter eficient finetuning for segment anything model. In: The Twelfth International Conference on Learning Representations (2024), https://openreview.net/forum? id=ezscMer8L0

# Supplementary Material LoCA: Spatially-Aware Low-Rank Convolutional Adaptation of Vision Foundation Models

Sojung An<sup>1\*</sup>, Junha Lee<sup>2,3\*</sup>, Sujeong You<sup>2</sup>, Nam Ik Cho<sup>3</sup>, Donghyun Kim<sup>1†</sup>

<sup>1</sup> Korea University, Seoul, Republic of Korea

2 Korea Institute of Industrial Technology, Ansan, Republic of Korea 3 Seoul National University, Seoul, Republic of Korea

<sup>∗</sup>Equal contribution <sup>†</sup>Corresponding author: d\_kim@korea.ac.kr https://github.com/ssojungan/loca

This supplementary material is organized as follows:

Sec. A provides the computational cost of LoRA and LoCA for diferent convolutional operator types and input resolutions.

– Sec. B presents comprehensive VTAB-1k and FGVC results, including training settings omitted from Sec. 5.1.

– Sec. C specifies the DreamBooth training protocol underlying the generative experiments in Sec. 5.2.

– Sec. D describes the settings and benchmark configurations for the domain generalization experiments in Sec. 5.3.

Sec. E ofers additional analyses of representational capacity through singular value evolution.

– Sec. F provides additional experiments for analyzing the sensitivity of LoCA.

– Sec. G clarifies the formulations in Sec. 4 through simplified code implementations of LoRA and LoCA.

## A Computational Cost

This section analyzes the computational cost of LoRA and LoCA. Parameter-Eficient Fine-Tuning (PEFT) aims to achieve strong performance with a small number of adaptation parameters. We therefore report the parameter complexity and FLOPs.

Computational Cost Formulation w.r.t. Kernel Size. The parameter complexity (P ) of LoRA is approximated as $P _ { \mathrm { L o R A } } \approx r \cdot k ^ { 2 } \cdot C .$ , where $r , k ,$ and C denote the adaptation rank, the kernel size, and the channel dimension, respectively. LoCA decomposes the adaptation into channel mixing and spatial basis components, $P _ { \mathrm { L o C A } } \approx r \cdot C + \mathrm { c o n s t } ( k ^ { 4 } + k ^ { 2 } )$ . The first term represents channel mixing parameters, and the remaining terms correspond to spatial basis parameters. The channel-dependent term scales with $k ^ { 2 } C$ in LoRA and C in LoCA. Implementation details are provided in Sec. G.

Parameter Complexity Analysis. Fig. A illustrates parameter complexity relative to kernel size. For 1×1 kernels, LoCA incurs a marginal parameter overhead (≈ 1.0×) compared to LoRA due to the spatial basis. However, increasing kernel sizes amplifies the parameter eficiency of LoCA. LoRA exhibits quadratic complexity growth (k<sup>2</sup>) with the kernel size due to its reliance on flattened-kernel decomposition. In contrast, LoCA decouples channel mixing and spatial basis components, isolating channel-dependent parameters from the kernel area. This eficiency gain is particularly evident in the depthwise convolution of ConvNeXt (Fig. Ad). In this configuration, LoCA constructs a shared spatial basis $s _ { b a s i s }$ defined solely by the kernel size k, with individual channels utilizing specific coeficients. While LoRA’s complexity involves a multiplicative r · $k ^ { 2 }$ · C factor, LoCA confines k-dependent terms to the spatial basis $( k ^ { 4 }$ and $k ^ { 2 } )$ , maintaining a channel-dependent term of only $r \cdot C .$ . Sharing spatial parameters across channels ensures superior parameter scaling in large-kernel operators, such as the $7 \times 7$ depthwise convolution.

FLOPs. Both LoRA and LoCA utilize an identical convolution operator during inference. Therefore, inference FLOPs remain equivalent, with the distinction residing solely in parameter complexity.

![](images/884bbe4020dccd01352d9a2db66cb4da5cecd7724d633d1fda4de715554f32e5.jpg)  
(a) K=1

![](images/67c03590f2d98603eddb5d9a32b2d5da8d1f50548bd507b5dcaf635700da9de9.jpg)  
(b) K=3

![](images/4f6f3be848d16ddb6f8a73ba5340ced694434f84dbe44ccb339f927be155bed7.jpg)  
(c) K=7

![](images/c8a8a9aba1e73b9a4044d8fb5680ac471dd3ba79f86c203dc33f5a84561b56ee.jpg)  
(d) K=7 (Depthwise)  
Fig. A: Computational cost with respect to kernel size and channel dimension

## B Fine-Grained Classification

We provide an extensive experimental analysis in Sec. 5.1, reporting task-wise results on VTAB-1k tasks and benchmarks on four FGVC datasets. VTAB-1k contains 19 tasks that cover a broad spectrum of domains and semantics. These are grouped into three sets: <sup>natural</sup> (Caltech101, CIFAR-100, DTD, Flowers102, Pets, Sun397, SVHN), <sup>specialized</sup> (EuroSAT, Resisc45, Patch Camelyon, Retinopathy), and <sup>structured</sup> (Clevr/count, Clevr/distance, dSprites/location, dSprites/orientation, SmallNORB/azimuth, SmallNORB/elevation, DM-Lab, KITTI/distance). The FGVC benchmark contains four specialized datasets aimed at fine-grained classification: CUB200, Stanford Dogs, Stanford Cars, and NABird.

## B.1 Setup Details

For a fair comparison, we follow the preprocessing protocol of CA [4] and use Center Crop, following [51]. For FGVC, we additionally apply RandomResizedCrop with a minimum scale of 0.2 and Horizontal Flip [39]. For few-shot classification, we use the same augmentation policy as in FGVC and adopt the hyperparameter search range listed in Tab. A.

Table A: Hyperparameters for image classification tasks of FGVC and VTAB-1k

<table><tr><td></td><td>All Backbones</td></tr><tr><td>Optimizer</td><td>AdamW [29]</td></tr><tr><td>Learning rate</td><td>1e-3</td></tr><tr><td>Weight decay</td><td>1e-4</td></tr><tr><td>LR schedule</td><td>cosine</td></tr><tr><td>Total Epochs</td><td>100</td></tr><tr><td>Warmup</td><td>10</td></tr></table>

## B.2 Experimental Results

In Tab. B and Tab. C, our task-wise analysis shows that LoCA consistently performs well across VTAB-1k tasks. We highlight the top-1, top-2, and top-3 entries in each column with progressively darker shades. For the <sup>natural</sup> group, where the target distribution is closest to ImageNet pre-training, LoCA is competitive with other PEFT baselines while achieving top results on several tasks. This indicates that LoCA can improve performance on <sup>natural</sup> tasks where PEFT baselines tend to lag (e.g., DTD/SVHN), without sacrificing overall competitiveness under limited domain shift. Notably, LoCA achieves the highest performance across all tasks within the <sup>specialized</sup> group. Because this group possesses domain-specific characteristics distinct from those of natural images, these results demonstrate that our approach efectively adapts to fine-grained, specialized domains. For the <sup>structured</sup> group, LoCA outperforms LoRA. We observe high performance variance in the <sup>structured</sup> group when the oficial LoRA implementation jointly tunes convolutional and linear layers. This variance likely occurs because these tasks rely heavily on encoding spatial and shape information. While some classes show gains over the linear-only configuration, others experience significant performance degradation.

In Tab. D and Tab. E, LoCA achieves the best mean accuracy on both backbones while remaining parameter-eficient. Using ConvNeXt-B pre-trained on ImageNet-21K, LoCA reaches the highest average accuracy, outperforming LoRA with significantly fewer trainable parameters (3.7 M vs. 17.6 M). The performance gains on NABird and CUB200 demonstrate our method’s ability to capture subtle inter-class diferences in fine-grained tasks. FSF performs best on Stanford Dogs. LoCA achieves the best results on CUB200 and NABird, matches the top accuracy on Stanford Cars, and obtains the best overall mean with 3.7M trainable parameters. Under the ResNet-50 architecture pre-trained on ImageNet-1K, LoCA yields the highest mean accuracy with only 1.9 M trainable parameters, surpassing both LoRA and FSF. Notably, LoCA improves CUB200 and NABird, indicating robust adaptation even with a smaller backbone. We also observe that full fine-tuning underperforms PEFT methods under the low-data regime, consistent with the tendency to overfit when updating all parameters.

Table B: Performance comparisons on the VTAB-1k benchmark with ConvNeXt-B models pre-trained on ImageNet-21K

<table><tr><td></td><td colspan="7">Natural</td><td colspan="4">Specialized</td><td colspan="7">Structured</td><td colspan="2"></td><td></td><td></td></tr><tr><td></td><td>Caltech101</td><td>CIFAR100</td><td>DTD</td><td>Flowers102</td><td>Pets</td><td>SVHN</td><td>Sun397</td><td>Patch</td><td>Camelyon</td><td>EuroSAT</td><td>Resisc45</td><td>Retinopathy</td><td>Clevr/count</td><td>Clevr/distance</td><td>DMLab</td><td>KITTI</td><td>dSprites/loc</td><td>dSprites/ori</td><td>SmallNORB/azi</td><td>SmallNORB/ele</td><td>Mean</td><td>Params. (M)</td></tr><tr><td>Full fine-tuning</td><td>91.0</td><td>66.2</td><td>74.8</td><td>99.6</td><td>92.0</td><td>88.5</td><td>51.5</td><td>86.8</td><td>95.9</td><td>88.7</td><td>78.8</td><td>81.6</td><td>53.6</td><td>55.3</td><td>82.4</td><td>95.0</td><td>70.3</td><td>37.2</td><td>35.4</td><td>73.7</td><td>87.7</td><td></td></tr><tr><td>LoRA</td><td>91.0</td><td>66.0</td><td>75.2</td><td>99.6</td><td>92.2</td><td>89.5</td><td>52.7</td><td>85.7</td><td>95.4</td><td>87.8</td><td>78.4</td><td>81.0</td><td>49.4</td><td>55.3</td><td>79.5</td><td>96.9</td><td>69.4</td><td>37.4</td><td>30.7</td><td>74.4</td><td>17.3</td><td></td></tr><tr><td>CA</td><td>90.9</td><td>66.0</td><td>74.9</td><td>98.8</td><td>92.4</td><td>52.9</td><td>88.4</td><td>86.0</td><td>95.6</td><td>85.7</td><td>77.9</td><td>86.5</td><td>59.5</td><td>55.0</td><td>93.7</td><td>67.1</td><td>83.5</td><td>39.0</td><td>34.7</td><td>75.2</td><td>6.8</td><td></td></tr><tr><td>CoLoRA</td><td>89.9</td><td>58.2</td><td>71.7</td><td>98.9</td><td>91.5</td><td>83.3</td><td>39.1</td><td>85.1</td><td>93.7</td><td>78.7</td><td>75.1</td><td>83.9</td><td>66.1</td><td>49.1</td><td>80.0</td><td>76.2</td><td>43.7</td><td>22.6</td><td>43.9</td><td>70.0</td><td>4.6</td><td></td></tr><tr><td>FSF</td><td>94.8</td><td>71.7</td><td>76.9</td><td>99.6</td><td>93.1</td><td>87.1</td><td>57.5</td><td>85.1</td><td>94.6</td><td>87.6</td><td>74.8</td><td>70.9</td><td>62.8</td><td>50.3</td><td>82.7</td><td>89.4</td><td>60.4</td><td>31.2</td><td>29.0</td><td>73.6</td><td>1.1</td><td></td></tr><tr><td>LoCA</td><td>90.8</td><td>69.5</td><td>77.1</td><td>99.7</td><td>93.6</td><td>86.7</td><td>55.3</td><td>86.6</td><td>96.1</td><td>88.8</td><td>78.5</td><td>92.9</td><td>54.7</td><td>53.7</td><td>83.2</td><td>89.7</td><td>67.4</td><td>37.1</td><td>41.3</td><td>75.9</td><td>5.0</td><td></td></tr></table>

Table C: Performance comparisons on the VTAB-1k benchmark with ResNet-50 models pre-trained on ImageNet-1K

<table><tr><td></td><td colspan="7">Natural</td><td colspan="4">Specialized</td><td colspan="8">Structured</td><td colspan="2"></td><td></td></tr><tr><td></td><td>Caltech101</td><td>CIFAR100</td><td>DTD</td><td>Flowers102</td><td>Pets</td><td>SVHN</td><td>Sun397</td><td>Patch</td><td>Camelyon</td><td>EuroSAT</td><td>Resisc45</td><td>Retinopathy</td><td>Clevr/count</td><td>Clevr/distance</td><td>DMLab</td><td>KITTI</td><td>dSprites/loc</td><td>dSprites/ori</td><td>SmallINORB/azi</td><td>SmallINORB/ele</td><td>Mean</td><td>Params. (M)</td></tr><tr><td>Full fine-tuning</td><td>83.9</td><td>25.9</td><td>63.4</td><td>90.4</td><td>91.0</td><td>33.1</td><td>71.4</td><td>79.3</td><td>91.6</td><td>82.0</td><td>75.1</td><td>60.4</td><td>50.6</td><td>45.8</td><td>56.4</td><td>61.9</td><td>78.6</td><td>27.3</td><td>37.6</td><td>61.0</td><td>23.6</td><td></td></tr><tr><td>LoRA</td><td>83.88</td><td>26.41</td><td>62.61</td><td>89.15</td><td>90.41</td><td>31.82</td><td>71.14</td><td>81.55</td><td>91.65</td><td>81.32</td><td>75.62</td><td>67.71</td><td>49.69</td><td>45.36</td><td>73.12</td><td>64.56</td><td>76.37</td><td>28.73</td><td>44.10</td><td>65.01</td><td>19.0</td><td></td></tr><tr><td>CA</td><td>86.98</td><td>27.22</td><td>64.40</td><td>82.21</td><td>88.98</td><td>32.67</td><td>51.31</td><td>78.59</td><td>88.20</td><td>75.29</td><td>73.80</td><td>35.94</td><td>44.98</td><td>35.40</td><td>41.50</td><td>15.29</td><td>69.95</td><td>14.72</td><td>38.21</td><td>55.03</td><td>1.4</td><td></td></tr><tr><td>CoLoRA</td><td>89.1</td><td>30.0</td><td>62.8</td><td>87.5</td><td>88.9</td><td>32.4</td><td>75.3</td><td>82.6</td><td>92.2</td><td>81.1</td><td>74.3</td><td>54.0</td><td>54.0</td><td>42.1</td><td>76.4</td><td>37.2</td><td>80.0</td><td>22.6</td><td>48.8</td><td>63.7</td><td>1.4</td><td></td></tr><tr><td>FSF</td><td>87.0</td><td>22.0</td><td>61.0</td><td>88.5</td><td>93.0</td><td>33.0</td><td>54.0</td><td>83.5</td><td>87.5</td><td>73.5</td><td>76.5</td><td>40.5</td><td>48.5</td><td>42.5</td><td>12.0</td><td>18.0</td><td>67.5</td><td>14.5</td><td>26.0</td><td>57.0</td><td>0.7</td><td></td></tr><tr><td>LoCA</td><td>84.4</td><td>30.8</td><td>65.6</td><td>90.6</td><td>91.0</td><td>34.5</td><td>73.6</td><td>83.5</td><td>92.4</td><td>82.1</td><td>76.5</td><td>76.6</td><td>53.2</td><td>48.2</td><td>50.6</td><td>59.5</td><td>77.0</td><td>27.6</td><td>43.4</td><td>65.3</td><td>1.5</td><td></td></tr></table>

## C Subject-Driven Text-to-Image Generation

We further evaluate LoCA on DreamBooth, a subject-driven text-to-image personalization task that adapts a pretrained difusion model from only a few reference images. We provide side-by-side comparisons across methods to analyze identity preservation and prompt consistency across diverse prompts.

## C.1 Setup Details

To evaluate LoCA on DreamBooth, we follow the experiment settings of [36] using Stable Difusion v1.4. The benchmark comprises 30 classes with 25 prompts per class. Prompts follow the DreamBooth [37] template “a photo of [V] [C]”, where [V] is the identifier “sks” and [C] denotes the class name. Optimization utilizes the AdamW optimizer [29] for 1,000 epochs. The learning rate is selected from 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, with 2e-4 as the default. For FSF [5], we report results using 1e-4, which exhibited superior performance in our comparative analysis.

Table D: FGVC benchmark with ConvNeXt-B (ImageNet-21K)

<table><tr><td></td><td>CUB200</td><td>Dogs</td><td>Cars</td><td>NABird</td><td>Mean</td><td>Params.</td></tr><tr><td>Full FT</td><td>73.3</td><td>73.6</td><td>88.3</td><td>67.7</td><td>75.7</td><td>79.7</td></tr><tr><td>LoRA</td><td>89.0</td><td>86.4</td><td>92.9</td><td>84.4</td><td>88.4</td><td>17.6</td></tr><tr><td>CA</td><td>73.6</td><td>88.5</td><td>79.5</td><td>67.1</td><td>77.2</td><td>6.8</td></tr><tr><td>CoLoRA</td><td>87.5</td><td>88.5</td><td>83.8</td><td>84.6</td><td>86.1</td><td>4.6</td></tr><tr><td>FSF</td><td>85.5</td><td>96.1</td><td>86.3</td><td>84.3</td><td>88.0</td><td>1.1</td></tr><tr><td>Ours</td><td>90.5</td><td>89.1</td><td>92.9</td><td>87.8</td><td>90.1</td><td>3.7</td></tr></table>

Table E: FGVC benchmark with ResNet-50 (ImageNet-1K)

<table><tr><td></td><td>CUB200</td><td>Dogs</td><td>Cars</td><td>NABird</td><td>Mean</td><td>Params.</td></tr><tr><td>Full FT</td><td>73.3</td><td>73.6</td><td>88.3</td><td>67.7</td><td>75.7</td><td>24.1</td></tr><tr><td>LoRA</td><td>78.3</td><td>85.8</td><td>87.8</td><td>71.9</td><td>81.0</td><td>2.4</td></tr><tr><td>CA</td><td>-</td><td>-</td><td>-</td><td>-</td><td>83.5</td><td>1.9</td></tr><tr><td>CoLoRA</td><td>74.2</td><td>81.7</td><td>84.1</td><td>65.7</td><td>76.4</td><td>1.0</td></tr><tr><td>FSF</td><td>76.7</td><td>90.6</td><td>79.1</td><td>77.9</td><td>81.1</td><td>2.5</td></tr><tr><td>Ours</td><td>80.8</td><td>88.7</td><td>88.3</td><td>76.9</td><td>83.7</td><td>1.9</td></tr></table>

## C.2 Experimental Results

In Fig. B, the dificulty of subject-driven generation depends heavily on both the prompt type and the input style. The top row presents relatively simple spatial and background-composition prompts, where all three methods successfully produce recognizable subject-centric images with marginal qualitative diferences. Performance gaps emerge clearly when prompts demand explicit scene cues (e.g., the Eifel Tower) rather than basic object placement. The lower-left introduces the more complex challenges of geometry modification and attribute binding, accentuating the qualitative diferences between methods. For the prompt “a cube shaped [V],” LoCA successfully captures the requested geometric transformation; the baselines incorrectly retain the original animal form. The prompt “a [V] wearing a black top hat and a monocle” further exposes varying degrees of success in rendering the specified attributes across the methods. The lower-right panel poses the most challenging setting due to the non-photographic, cartoonlike input style. The models process the simpler prompt “a [V] on a cobblestone street” reasonably well, but struggle against the complex prompt “a [V] on top of green grass with sunflowers.” Overwhelmed by the detailed background request, all three methods over-emphasize the surrounding scene and noticeably degrade the original subject’s identity.

## D Domain Generalized Semantic Segmentation

We provide additional experimental details for the domain generalization benchmarks studied in Sec. 5.3, including domain generalized semantic segmentation (DGSS) and domain generalized object detection (DGOD). These benchmarks assess adaptation to semantic segmentation and object detection under crossdomain distribution shift.

Input images  
![](images/b5aeddd53746d30415520b44f6dd7a76067aeca5a5ef5b4baffa7d71e9e94e1a.jpg)

![](images/7147de648b5ea42d49a9c19ce5d6239e83071808642d30727b14b104f0ce0967.jpg)

100 Iter  
Fig. B: Qualitative results of subject-driven task. We visualize the results to compare PEFT methods: LoRA [21], FSF [5], and LoCA.  
300 Iter  
500 Iter  
700 Iter  
900 Iter  
![](images/25a782ef33e67d8a5efc3ec64124bf8b4f8e78184ded0c9732e95a24129d973e.jpg)  
Fig. C: Qualitative results across training iterations. Both LoRA and LoCA show efective adaptation from around 300 iterations. LoRA shows a slight artifact around 900 iterations.

## D.1 Setup Details

Our implementation uses the MMSegmentation [8] codebases for Domain Generalized Semantic Segmentation (DGSS) and Domain Generalized Object Detection (DGOD), respectively, and Hugging Face scripts for personalization experiments. For DGSS, we follow the configurations in SoMA [47]. Mask2Former [7] serves as the default decode head using basic data augmentation from Rein [42], and EMA is employed to ensure training stability. A consistent configuration is maintained by fixing both the rank and alpha at 16 across all experiments, including backbone adaptation. Optimization is performed using AdamW [29] (lr = 0.001, weight decay = 0.05). Learning-rate multipliers of 0.5, 0.1, and 0.0 are applied to the backbone, $u _ { d w }$ , and $s _ { b a s i s } ,$ respectively. Notably, unlike the original SoMA, weight decay is disabled for these spatial SVD parameters $( u _ { d w }$ and $s _ { b a s i s } )$ . We specifically freeze $s _ { b a s i s }$ with a multiplier of 0.0 to prevent basis collapse, as the adaptation relies primarily on the basis direction rather than its magnitude.

Table F: Efect of the proposed components under GTAV → Mapillary DGSS setting. We highlight the best for each column.

<table><tr><td>Methods</td><td>Params.</td><td>road side.</td><td>build.</td><td>wall</td><td>fence</td><td>pole</td><td>light</td><td>sign</td><td>vege.</td><td>terr.</td><td>sky</td><td>pers.</td><td>rider</td><td>car</td><td>truck</td><td>bus</td><td>train</td><td>motor.</td><td>bicy.</td><td>mIoU</td><td></td></tr><tr><td>Full fine-tuning (baseline)</td><td>87.56M</td><td>90.4</td><td>65.0</td><td>85.5</td><td>41.7</td><td>51.9</td><td>55.2</td><td>67.1</td><td>52.8</td><td>79.7</td><td>52.4</td><td>94.4</td><td>79.5</td><td>58.5</td><td>90.5</td><td>66.0</td><td>76.0</td><td>36.4</td><td>65.1</td><td>46.2</td><td>67.01</td></tr><tr><td> $\sqsubseteq +$  Channel Mixing</td><td>5.0M</td><td>92.6</td><td>66.6</td><td>87.9</td><td>49.5</td><td>56.1</td><td>58.3</td><td>68.0</td><td>55.8</td><td>82.3</td><td>51.1</td><td>95.3</td><td>78.6</td><td>52.3</td><td>89.5</td><td>62.7</td><td>80.3</td><td>51.0</td><td>71.7</td><td>62.4</td><td>69.13</td></tr><tr><td> $\sqsubseteq +$  Spatial Basis</td><td>5.0M</td><td>92.7</td><td>67.4</td><td>88.5</td><td>51.4</td><td>55.7</td><td>58.4</td><td>68.9</td><td>57.9</td><td>83.0</td><td>48.1</td><td>95.7</td><td>79.2</td><td>52.9</td><td>89.4</td><td>59.5</td><td>81.9</td><td>72.8</td><td>72.1</td><td>59.7</td><td>70.26</td></tr><tr><td> $\sqsubseteq +$  Hierarchical Rank</td><td>5.0M</td><td>92.8</td><td>68.3</td><td>87.9</td><td>49.8</td><td>54.4</td><td>57.7</td><td>68.2</td><td>57.1</td><td>84.3</td><td>51.6</td><td>96.0</td><td>78.9</td><td>53.3</td><td>90.2</td><td>64.7</td><td>84.6</td><td>71.0</td><td>71.5</td><td>59.6</td><td>70.62</td></tr></table>

## D.2 Experimental Results

In Tab. F, we report class-wise ablation results on the Mapillary dataset. Full fine-tuning achieves strong performance on several object-centric categories. Spatial basis modeling improves performance on classes with structured spatial patterns (e.g., train and sign). Hierarchical rank provides additional gains for categories that require broader spatial context, including road and sidewalk. These results indicate that diferent components contribute complementary benefits across category types.

## E Evolution of Representational Capacity

We visualize the Singular Value (SV) expansion results for ResNet-50 (Fig. D) and MobileMamba (Fig. E) to complement the analysis presented in the main paper. The SV distributions across layers provide a qualitative view of adaptation behavior.

ResNet. For ResNet-50 (Fig. D), FSF exhibits slow growth during the early training stage. The SVs gradually expand as training progresses. This trend indicates that adaptation occurs but proceeds ineficiently. In addition, the spatial components show limited SV expansion. LoRA and LoCA show faster SV expansion in the ResNet architecture. LoRA produces stronger SV expansion in the early layers. LoCA shows a more consistent SV distribution across all layers. LoRA tends to concentrate adaptation on early features. LoCA reflects features more uniformly across the network. The stacked convolutional structure of ResNet is associated with relatively moderate SV expansion compared to other architectures.

MobileMamba. In the case of MobileMamba-T2 (Fig. E), the architecture employs 3×3 convolution kernels for local learning and wavelet-domain convolutions to capture spatial global information in each block. The wavelet kernels are defined with diferent sizes (3, 5, and 7) across blocks. This hierarchical design facilitates multi-scale representation learning. LoRA exhibits rapid SV expansion during early training. The SV values in the later components fluctuate noticeably. The degree of SV expansion varies across blocks. FSF also exhibits fluctuations and inconsistent SV expansion between blocks. LoCA shows a different pattern. Some fluctuations appear in the local components. The global components remain stable. The SV values adjust smoothly across kernels. Both channel and spatial information are updated consistently.

![](images/7eb573bb9105c1dac83717d2166ba39fe921da66c6fd231533743ea9adc27f71.jpg)

Fig. D: Singular value evolution of ResNet-50 weights across PEFT methods (y-axis: singular values). Columns 1–4: channel-wise SVD of depthwise weights; columns 5–8: spatial SVD. L: layer.  
![](images/6fe592d0c244b777db3af9db8d4ef44feee3c1f54da722bd4a2d929f10b4823c.jpg)  
Fig. E: Singular value evolution of MobileMamba-T2 weights across PEFT methods (yaxis: singular values). Columns 1–4: channel-wise SVD of depthwise weights; columns 5–8: spatial SVD. B: block. Local: depthwise convolution. Global: wavelet-domain convolution.

## F Additional Sensitivity Analyses

## F.1 How sensitive is adaptation performance to the choice of convolution modules with diferent kernel structures?

To analyze performance gains across target modules, ResNet-50 and MobileMamba-T2 are evaluated with all ranks set to 16. ResNet consists of 1×1 and 3×3 convolutions with downsample layers. The conv1 and conv2 modules correspond to the two 3×3 convolutions, while conv3 corresponds to the 1×1 convolution.

![](images/f37d5e25164e61f7e31288a1c5287ba29518b2802a96be2ab22790247bed664d.jpg)

![](images/717be5c09dd3314176ddcdd1668b1e3696f014ff14f9a51373cd08d4269da945.jpg)

![](images/edd077f40c236b5231f479c1f4ff5b24e7b9537f92a68aec588b75cca0e02528.jpg)

![](images/68e2764d6d91b844c23e8bdfbce14bd1f0c1604baf03e0869585b1ea4c0ba430.jpg)  
Fig. F: SV expansion of ConvNeXt-B on VTAB-1K. Shaded regions denote standard deviation across datasets. LoCA shows consistent adaptation expressivity across datasets.

![](images/5fdd7db1ef72ab7b7cea3ea10f2d498c822b2559eaaf63aa2d09e921f37c5a5e.jpg)  
(a) ResNet-50 performance

![](images/755b3d0b63634dc01ff4353aacffe3ebc3857cd5458147ed5a8d847c6d748694.jpg)  
(b) MobileMamba-T2 performance  
Fig. G: Performance across target modules on FGVC datasets. Each bubble’s area is proportional to the number of parameters.

Fig. G shows that adapting only conv3 achieves performance comparable to other configurations while requiring fewer parameters. MobileMamba adopts a hybrid architecture combining convolution and linear operations. The modules are divided into three components: depthwise convolution (dw), feed-forward network (fn), and mixer. The dw captures local spatial patterns through channel-wise filtering, the fn performs channel mixing through pointwise convolutions, and the mixer models long-range dependencies using an SSM-based operator. Adapting the mixer yields the largest performance improvement, whereas adapting only dw captures mainly local information and exhibits performance variance. We observe that the adaptation performs consistently across convolution modules, indicating stable behavior across layers.

## F.2 Sensitivity Analysis of the Scaling Factor

Since LoRA has been reported to be sensitive to the scaling factor α [1], we evaluate $\alpha \in { 4 , 8 , 1 6 , 3 2 }$ across three architectures. As shown in Fig. H, LoCA maintains consistent accuracy across ResNet-50, ConvNeXt-B, and MobileMamba-T2, indicating that LoCA is robust to the choice of α.

![](images/4b963b78029e0fea711d0bb8498d1edbfd9194d163534e7c1ff1e7316baed01b.jpg)

![](images/25477c96ec363ca2dd79137ca848c46a24904bb2259c04e80df96a170e734f36.jpg)  
Fig. H: Sensitivity analysis of the scaling factor α

![](images/4015c2b8802555b089d206187aee9de193fec7dfd14527e0880c73e68dcf452c.jpg)

## G Code

In Sec. G, we provide a reference implementation for the LoRA and LoCA formulations discussed in Sec. 4. Code 2.1<sup>4</sup> shows the flattened convolutional LoRA baseline. Code 2.2 shows how LoCA composes a channel low-rank update with SVD-initialized spatial basis refinement on top of a pretrained convolution kernel.

Code 2.1: Baseline convolutional LoRA implementation  
```python
class ConvLoRA(nn.Module):
    def __init__(self, conv_module, in_channels, out_channels, kernel_size, r=0, lora_alpha=1):

    super().__init__()
    self.conv = conv_module(in_channels, out_channels, kernel_size)

    # LoRA performs low-rank decomposition on the flattened convolution weight
    # Assume square kernel (kh = kw)
    # lora_a: [r * kh, in_channels * kh]
    # lora_b: [out_channels * kh, r * kh]
    self.lora_a = nn.Parameter(
    torch.zeros(r * kernel_size, in_channels * kernel_size)
    )
    self.lora_b = nn.Parameter(
    torch.zeros(out_channels * kernel_size, r * kernel_size)
    )
    self.scaling = lora_alpha / r

def forward(self, x):
    delta_w = (self.lora_b @ self.lora_a)
    delta_w = delta_w.view(self.conv.weight.shape)

    return self.conv._conv_forward(
    x,
    self.conv.weight + delta_w * self.scaling,
    self.conv.bias
    )
    return self.conv(x)
```  
Code 2.2: Proposed LoCA implementation

<sup>4</sup> We use the oficial implementation of LoRA available at https://github.com/ microsoft/LoRA/blob/main/loralib/layers.py. All LoRA experiments are conducted using this implementation.

```python
class LoCAConv2d(nn.Module):
    def __init__(self, conv, lora_rank=16, lora_alpha=16.0):
    super().__init__()
    
    self.weight_shape = conv.weight.shape
    out_c, in_c, kh, kw = self.weight_shape
    
    self.weight = nn.Parameter(conv.weight.data.clone(),
    requires_grad=False)
    self.stride = conv.stride
    self.padding = conv.padding
    self.dilation = conv.dilation
    self.groups = conv.groups
    
    self.r = min(lora_rank, out_c)
    self.scaling = (lora_alpha / self.r)

    # Sec. 4.1 Low-Rank Channel Adaptation
    # Channel mixing low-rank update for convolution kernel
    # lora_a: [r, in_channels * kh * kw]
    # lora_b: [out_channels, r]
    self.lora_a = nn.Parameter(torch.empty(self.r, in_c * kh * kw))
    self.lora_b = nn.Parameter(torch.zeros(out_c, self.r))

    # Spatial basis extracted from pretrained conv kernels
    # s_basis: [kh*kw, kh, kw]
    self.spatial_rank = kh * kw
    self.s_basis = nn.Parameter(init_spatial_basis_from_svd(conv.weight))

    # Spatial update coefficients for diagonal channel pairs
    # u_dw: [diag (min(out_c, in_c)), spatial_rank]
    self.diag = min(out_c, in_c)
    self.u_dw = nn.Parameter(torch.zeros(self.diag, self.spatial_rank))

    # Sec. 4.2 SVD-based Spatial Basis Refinement
    # Spatial diagonal update
    def _get_delta_weight(self):
    out_c, in_c, kh, kw = self.weight_shape

    dW_ch = (self.lora_b @ self.lora_a) * self.scaling

    dW_sp_diag = self.u_dw @ self.s_basis.view(self.spatial_rank, -1)

    dW = dW_ch.view(out_c, in_c, kh * kw)
    dW.diagonal(dim1=0, dim2=1).add_(dW_sp_diag.T)
```

```python
return dW.reshape(out_c, in_c, kh, kw)
def forward(self, x):
    W_eff = self.weight + self._get_delta_weight()
    return F.conv2d(
    x, W_eff, self.bias,
    stride=self.stride,
    padding=self.padding,
    dilation=self.dilation,
    groups=self.groups
)
```

## H Limitation and Future Work

The proposed framework implements a PEFT approach for convolutional operators by decoupling channel mixing and spatial basis components. This formulation facilitates eficient adaptation and improves the parameter–performance trade-of compared to LoRA. The current limitation stems from the fixed spatial basis design across varying kernel sizes, which constrains the spatial representation capacity of 1x1 kernels. Nevertheless, empirical evaluations demonstrate robust performance across diverse kernel scales. Future research aims to investigate adaptive construction of spatial bases conditioned on specific kernel dimensions.