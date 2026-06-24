# SPAR: Semantic-Pixel Self-Alignment and Adaptive Routing for Unified Multimodal Models

Hongxiang Li<sup>1\*</sup>, Hongxu Chen<sup>1\*</sup>, Chenyang Zhu<sup>1</sup>, Xiaoshuang Huang<sup>2</sup>, Jiayin Cai<sup>2</sup>, Xiaolong Jiang<sup>2</sup>, Yao Hu<sup>2</sup>, and Long Chen<sup>1†</sup>

<sup>1</sup> The Hong Kong University of Science and Technology <sup>2</sup> Xiaohongshu Inc.

hlihg@connect.ust.hk

Project page: https://hkust-longgroup.github.io/SPAR

Abstract. Multimodal Large Language Models (MLLMs) have achieved remarkable success in visual understanding but remain constrained in visual generation due to the fundamental feature discrepancy between semantic perception and pixel-level reconstruction. Bridging this gap requires overcoming two core challenges: endowing semantic encoders with high-fidelity reconstruction capabilities, and efectively aligning generative models with semantic spaces without relying on external teachers. To this end, we propose a novel unified multimodal framework featuring Semantic-Pixel self-alignment and Adaptive Routing (SPAR). First, to reconcile semantic perception with pixel-level reconstruction, we introduce an asymmetric dual-stream unified tokenizer. A lightweight semantic stream anchors discriminative features, while a Transformeraugmented pixel stream recovers fine-grained visual details into a unified compact latent space. Second, to eliminate external dependencies, we propose a self-aligned generation paradigm that natively leverages this optimized tokenizer as an internal alignment teacher for the difusion model. Furthermore, to facilitate flexible multimodal interaction within this unified space, we introduce Dynamic Token Routing, which enables each token to adaptively aggregate multi-layer MLLM features based on its distinct semantic demands. Extensive experiments demonstrate that SPAR establishes the state-of-the-art for unified architectures, achieving exceptional generation and reconstruction quality while preserving foundational visual understanding capabilities.

Keywords: Unified Model · Visual Generation · Visual Tokenizer

## 1 Introduction

Multimodal Large Language Models (MLLMs) [1,25,50] have demonstrated exceptional performance across a wide range of perception tasks, such as image captioning, visual question answering, and multimodal reasoning, by aligning powerful visual semantic encoders with Large Language Models (LLMs) [48,61]. While

\* Equal contribution.

<sup>†</sup> Corresponding author.

![](images/685a49ed554225349d06fb2844ebf51b6141cd011dd05978566d2a384562934a.jpg)  
Fig. 1: (a) Image Reconstruction: When modeling directly within the semantic representation space, existing methods sufer from lossy compression and struggle to preserve high-frequency details. In contrast, our method efectively recovers these crucial pixel-level details. (b) Representation Alignment Paradigm: Unlike existing approaches that rely on external semantic encoders to guide the generative model, our framework natively employs the unified tokenizer itself as an alignment teacher.

these models have become pervasive paradigms in the understanding domain, they have yet to dominate the field of visual generation. Meanwhile, current state-of-the-art image generation systems [19,20] still rely primarily on low-level, compact latent spaces constructed by Variational Auto-Encoders (VAEs) [17], upon which difusion processes [11, 36] or autoregressive modeling [45, 65] are performed. This leads to a discrepancy in feature patterns between perception and generation models.

Recent studies [62, 66, 74] have explored incorporating structured semantic representations (e.g., DINOv2 [32]) into generative models. By either aligning difusion processes with these rich semantic priors or developing generative models directly upon them, these approaches have yielded significant gains in generation quality and convergence speed. The growing eficacy of these semantic spaces in generation reveals a promising path toward unifying visual perception and generation. This raises a natural and critical question: Rather than relying on latent spaces, can we seamlessly empower powerful MLLMs with image generation capabilities by shifting the generative modeling space directly to the semantic representation space?

However, realizing this paradigm migration is far from straightforward, as an inherent contradiction between semantic perception and pixel reconstruction needs to be overcome. Although recent approaches [62, 74] have successfully implemented difusion models directly within the semantic representation space, they sufer from severe limitations in image reconstruction quality. As illustrated in Figure 1(a), RAE [74] struggles to preserve high-frequency details, often yielding blurry artifacts and distorted structures. This occurs because semantic encoders optimized through representation learning have feature spaces that are highly converged toward discriminative tasks. This process is essentially a lossy compression of the input image, which tends to discard texture and structural details that are irrelevant to perception. But these details may be crucial for pixel reconstruction. Consequently, this inherently weak reconstruction capability severely hinders the model’s synthesis quality when scaled to text-to-image generation and complex instruction-based image editing [71]. Furthermore, completely fine-tuning the semantic encoder to recover these lost pixel-level details would inevitably trigger catastrophic forgetting, destroying its original understanding abilities. Therefore, how to endow the semantic encoder with high-quality pixel reconstruction capability without compromising its original understanding ability remains the core bottleneck for migrating the generative modeling space to the semantic representation space.

Moreover, another critical challenge arises when efectively aligning representations to the generative model during unified model training. As shown in Figure 1(b), existing representation-aligned generation methods [62,66] typically bridge this gap by enforcing the difusion model to align with external visual encoders [32] While this forced external alignment provides structured guidance, recent studies [3] suggest that as data scale grows, relying on an isolated external teacher becomes sub-optimal. It risks manifold mismatches between the multimodal understanding and visual generation spaces. Consequently, how to natively align the generative model directly with the unified semantic space, and eliminate the dependency on external representation learners is a key challenge for efectively training unified understanding and generation models.

To address the above challenges, we propose Semantic-Pixel self-alignment and Adaptive Routing (SPAR) for unified multimodal models. To overcome the inherent contradiction between semantic perception and pixel reconstruction, we introduce an asymmetric dual-stream self-alignment architecture into the unified tokenizer. The lightweight semantic stream acts as an anchor to preserve the encoder’s original discriminative features, preventing catastrophic forgetting. Concurrently, the Transformer-augmented pixel stream is dedicated to mapping the high-dimensional semantic space into a compact latent space, efectively recovering the high-frequency spatial textures discarded by the encoder’s lossy compression. The two streams are fused in a compact latent space, explicitly decoupling semantic preservation from pixel reconstruction. Notably, the dual-stream-optimized tokenizer itself can serve as the representation alignment teacher for the difusion model, seamlessly transferring the accumulated semantic and pixel knowledge to the generation stage without relying on any disjointed external feature model. Furthermore, we introduce Dynamic Token Routing (DTR) to facilitate flexible multimodal interaction within this unified space. It empowers each token to adaptively aggregate multi-layer MLLM features based on its distinct semantic role, dynamically satisfying the diferentiated hierarchical feature demands of varying tokens.

In summary, our contributions are threefold:

– We propose a semantic-pixel dual-stream self-alignment architecture that explicitly decouples semantic preservation and pixel reconstruction through an asymmetric design, achieving high-fidelity reconstruction without degrading the encoder’s understanding capability.

– We introduce dynamic token routing, which enables each token to adaptively determine its own multi-layer fusion weights from the MLLM to optimally harness multimodal representations for generative guidance.

We present a self-alignment generation paradigm that leverages the unified tokenizer as an alignment teacher, removing the dependency on external feature models while achieving competitive results across multiple benchmarks.

## 2 Related Work

Unified Visual Tokenizer. Early approaches [49,75] relied on reconstructionfocused generative models, which often lack the high-level semantics required for understanding tasks. To address this, recent works have explored unified tokenizers from various perspectives. Some attempt to discretize semantic features directly [29, 34, 56], but sufer from information loss inherent in quantization. Another recent methods [37, 41, 43, 69, 74] attempt to endow pre-trained semantic encoders with pixel reconstruction capabilities via self-distillation and continuous feature alignment. However, forcing a single representation stream to simultaneously accommodate abstract semantics and fine-grained spatial details inevitably leads to capacity bottlenecks and implicit performance trade-ofs. To resolve this tension, we explicitly decouple these competing objectives through an unified tokenizer, coordinating visual understanding and generation.

Representation-Guided Generation. Pre-trained visual representations are increasingly integrated into difusion models to enhance semantic and structural fidelity. Recent methods typically achieve this by either explicitly aligning the generative latent space with external discriminative features [32, 66], or directly performing difusion modeling within the semantic space [74]. However, these approaches face inherent limitations. As highly abstract semantic spaces naturally discard pixel details, such models often sufer from degraded reconstruction quality and of-manifold artifacts [71]. Furthermore, extracting guidance features from external networks forces the generative process to rely on a representation space that is entirely isolated from the model’s native visual encoder. To address this, we introduce a self-aligned generation paradigm. By natively utilizing our explicitly decoupled tokenizer as an internal alignment objective.

Unified Multimodal Understanding and Generation. The integration of visual understanding and generation within a single LLM has garnered significant attention. Early unified models [44, 63] typically formulated both tasks as next-token prediction within a purely autoregressive framework. To leverage the superior synthesis quality of difusion models, recent approaches propose attaching difusion modules directly to the LLM backbone [42, 59]. To bridge the modality gap, these architectures employ various connection mechanisms. For instance, some methods utilize learnable Q-Formers or linear projectors to align representations [6, 54], while others adopt Mixture-of-Experts (MoE) or specialized visual queries to manage task routing [10,23,43]. Despite diverse connector designs, a commonality among these models is their reliance on static, token-agnostic feature extraction $( e . g .$ , using only the final layer or fixed fusion weights). This rigid paradigm fails to fully exploit the rich hierarchical representations within MLLMs. To address this, SPAR introduces dynamic token routing to adaptively aggregate multi-layer features for optimal generative guidance.

![](images/67a037a17f2e32caa90ae2db926a72faa03b7c46b6bfe4ec1055c692062cb97d.jpg)  
Fig. 2: Architecture of the semantic-pixel self-aligned unified tokenizer. Our tokenizer design explicitly decouples semantic preservation from pixel reconstruction. A lightweight semantic stream maps features into a compact latent space, strictly anchored by the frozen encoder to prevent catastrophic forgetting $( \mathcal { L } _ { s } )$ . Concurrently, a Transformer-augmented pixel stream bridges the dimensional gap by aligning with the native pixel latent space $( \mathcal { L } _ { p } )$ to recover high-frequency spatial details. Both streams are fused and decoded to reconstruct the image $( { \mathcal { L } } _ { r } ) _ { : }$ endowing the model with highfidelity generation capabilities without compromising its discriminative understanding.

## 3 Method

## 3.1 Semantic-Pixel Self-Aligned Unified Tokenizer

The unified tokenizer aims to construct a visual encoding space that preserves both discriminative semantic information and pixel-level reconstruction capability. Consequently, it can serve as the visual encoder of the MLLM for understanding tasks while simultaneously providing a compact latent space for difusion modeling. Training pixel decoders directly on the frozen semantic encoder only yields blurry reconstruction results, while unlocking the encoder for reconstruction training triggers catastrophic forgetting.

Semantic information has already been suficiently encoded in the output features of the pre-trained encoder, and applying excessive transformations would instead perturb the original feature distribution. In contrast, the pixel-level details discarded by the encoder during representation learning require additional computational capacity for recovery. Based on this asymmetry, we explicitly decouple the two competing objectives of semantic preservation and pixel reconstruction as shown in Fig. 2. For the input image $\bar { I } \in \mathbb { R } ^ { H \times W \times 3 }$ , we construct two diferent streams on top of the encoder output feature $f _ { v } = E _ { s } ( I ) \in \mathbb { R } ^ { H ^ { ' } \times W ^ { ' } \times D _ { s } }$ where $E _ { s }$ is semantic encoder and $D _ { s }$ is the feature dimension.

Semantic Stream. The semantic stream employs a lightweight projector $\mathcal { P } _ { s }$ consisting of multiple residual blocks and an MLP projection layer, which maps the encoder features to the compact latent space with minimal transformation: $\mathbf { z } _ { \mathrm { s } } = \mathcal { P } _ { s } ( f _ { v } ) \in \mathbb { R } ^ { H ^ { ' } \times W ^ { ' } \times D _ { p } }$ , where $D _ { p } \ll D _ { s }$ is the latent space dimension. The residual connections ensures the output of the semantic stream to preserve the original discriminative structure of $f _ { v }$ as much as possible, serving as an anchor for the encoder’s semantic features in the latent space.

Pixel Stream. A fundamental gap exists between the feature space of the semantic encoder and the compact latent space operated on by the pixel decoder. The semantic space is high-dimensional $( D _ { s } )$ and optimized for discriminative objectives, whereas the pixel latent space is low-dimensional $( D _ { p } \ll D _ { s } )$ and tailored for reconstruction. For the semantic encoder’s output to be correctly decoded by the pixel decoder, an efective mapping from the semantic feature space to the pixel latent space must first be established. The pixel stream is designed as a learnable bridge for this purpose. It transfers high-dimensional semantic features into the compact pixel latent space, thus allowing the finegrained details discarded by the encoder during representation learning to be recovered within the pixel decoder’s native space. Specifically, the pixel stream is equipped with an independent Transformer encoder ${ \mathcal { T } } _ { p } ,$ which models longrange dependencies along the spatial dimension through global self-attention and transforms the semantic features into intermediate representations suitable for pixel reconstruction:

$$
f _ {p} = \mathcal {T} _ {p} (f _ {v}) \in \mathbb {R} ^ {H ^ {'} \times W ^ {'} \times D _ {s}}.
$$

The Transformer output $f _ { p }$ is then mapped to a compact latent space compatible with the pixel decoder through residual blocks and an MLP projection layer:

$$
\mathbf {z} _ {\mathrm{p}} = \mathcal {P} _ {p} (f _ {p}) \in \mathbb {R} ^ {H ^ {'} \times W ^ {'} \times D _ {p}}.
$$

Fusion & Decoding. The compact latent variables produced by the two streams are merged into a unified representation in the fusion layer ${ \mathcal F } .$ . The semantic latent $\mathbf { z } _ { \mathrm { s } }$ and the pixel latent $\mathbf { z } _ { \mathrm { p } }$ are concatenated along the channel dimension and then mapped back to the latent space dimension by the fusion layer:

$$
\mathbf {z} _ {f} = \mathcal {F} \big ([ \mathbf {z} _ {\mathrm{s}}; \mathbf {z} _ {\mathrm{p}} ] \big) \in \mathbb {R} ^ {H ^ {'} \times W ^ {'} \times D _ {p}},
$$

where $[ \cdot ; \cdot ]$ denotes channel concatenation. The fused $\mathbf { z } _ { f }$ is fed into the pretrained pixel decoder $\mathcal { D } _ { p }$ for image reconstruction. This design allows semantic and pixel information to evolve along their respective independent optimization paths before converging in the compact latent space.

## 3.2 Progressive Three-Stage Training for Unified Tokenizer

The training of the unified tokenizer follows a progressive three-stage strategy that gradually releases model capacity, with targeted alignment losses introduced at each stage to guide the optimization direction.

Stage I: Dual-Stream Initialization. The vision encoder and pixel decoder are frozen, and only the dual-stream modules are trained. The core objective of this stage is to enable the pixel stream to learn the transfer mapping from the semantic feature space to the pixel latent space. However, the semantic feature space and the pixel decoder’s latent space difer fundamentally in dimensionality, distribution, and optimization objectives, making it dificult for the pixel stream to discover the correct mapping direction without explicit supervision. To this end, we introduce the pixel alignment loss that uses the output $\mathbf { z } _ { p } ^ { \prime } = E _ { p } ^ { ' } ( I )$ of the frozen pixel encoder $E _ { p }$ as an explicit optimization anchor and aligns the pixel stream’s latent variables with it:

$$
\mathcal {L} _ {p} = \| \mathbf {z} _ {p} - \mathbf {\Sigma} _ {p} ^ {\prime} \| _ {2} ^ {2}.
$$

This loss first guides the pixel stream to transfer the semantic feature space onto the latent manifold of the pixel encoder, aligning its output distribution with the pixel decoder’s native encoding. $\mathrm { O n }$ this basis, the pixel stream can then efectively recover the pixel-level details discarded by the semantic encoder’s lossy compression within that space. The complete training loss for this stage is:

$$
\mathcal {L} _ {\mathrm{I}} = \mathcal {L} _ {\mathrm{MSE}} + \mathcal {L} _ {\mathrm{LPIPS}} + \lambda_ {p} \mathcal {L} _ {p},
$$

where $\mathcal { L } _ { \mathrm { M S E } }$ represents pixel-wise reconstruction loss, and $\mathcal { L } _ { \mathrm { L P I P S } }$ represents the perceptual loss computed using the LPIPS metric.

Stage II: Joint Decoder Training. The vision encoder remains frozen while the pixel decoder $\mathcal { D } _ { p }$ is unfrozen for joint training. Only pixel reconstruction losses are used in this stage:

$$
\mathcal {L} _ {\mathrm{II}} = \mathcal {L} _ {\mathrm{MSE}} + \mathcal {L} _ {\mathrm{LPIPS}}.
$$

The decoder adapts to the latent distribution produced by the dual-stream fusion, maximizing reconstruction quality with a fixed encoder.

Stage III: Encoder Fine-Tuning with Self-Distillation. The vision encoder is unfrozen for end-to-end training to further enhance pixel representation capability. However, unfreezing the encoder risks catastrophic forgetting: the gradients from reconstruction training may corrupt the discriminative semantic features accumulated during contrastive learning. To address this, we introduce a semantic alignment loss that uses the features $f _ { v } ^ { \prime } = E _ { s } ^ { ' } ( I )$ of a frozen encoder to constrain the fine-tuned encoder features to maintain semantic consistency:

$$
\mathcal {L} _ {s} = \| f _ {v} - f _ {v} ^ {\prime} \| _ {2} ^ {2}.
$$

The encoder learning rate is simultaneously decayed to 0.1× the global learning rate, working together with $\mathcal { L } _ { \varepsilon }$ to form a dual constraint that ensures the encoder does not lose its original semantic understanding performance while acquiring stronger pixel representation capability. Additionally, a GAN discriminator is activated after a certain number of training steps to enhance the perceptual quality of reconstructed images. The complete training loss for this stage is:

$$
\mathcal {L} _ {\mathrm{III}} = \mathcal {L} _ {\mathrm{MSE}} + \mathcal {L} _ {\mathrm{LPIPS}} + \lambda_ {s} \mathcal {L} _ {s} + \lambda_ {\mathrm{GAN}} \mathcal {L} _ {\mathrm{GAN}}.
$$

![](images/ecb49ad6c2359cd30b65077c62a5a4ea45da658f8b4962d3d833fe8c78ed5667.jpg)  
Fig. 3: Overview of the unified multimodal model. The frozen MLLM processes multimodal inputs and learnable queries. The DTR adaptively aggregates multi-layer MLLM hidden states based on distinct token semantics to condition the DiT. Furthermore, the optimized tokenizer serves as an internal alignment teacher, establishing a self-alignment paradigm that eliminates reliance on external learners.

## 3.3 Unified Multimodal Model

Conditional Signal Construction. To efectively bridge the multimodal reasoning capability of the MLLM with the difusion-based generation process, we introduce a set of learnable query embeddings as implicit placeholders for the target image representation. During training, specific image generation positions are reserved in the input token sequence, and the query embeddings are inserted at these positions by replacing the corresponding token embeddings as shown in Figure 3. The resulting composite sequence, which contains text tokens, optional reference image tokens, and query embeddings, is then processed uniformly by the MLLM’s causal self-attention mechanism. The query positions located toward the end of the sequence naturally aggregate contextual information from the text instructions and visual references, thereby forming conditional representations that fuse multimodal semantics within the MLLM’s hidden states.

Dynamic Token Routing. Multimodal representations within MLLMs are inherently hierarchical. Existing unified models typically extract only the last-layer hidden states or employ a token-agnostic static concatenation, completely overlooking the diferentiated demands of individual tokens on hierarchical features. To optimally harness these representations for generative guidance, we introduce Dynamic Token Routing (DTR), a mechanism that grants each token the ability to adaptively select its own multi-layer feature fusion weights according to its semantic role. Specifically, the L-layer Transformer of the MLLM outputs all L + 1 hidden states (including the embedding layer output). DTR uniformly samples K layers from them with sampling interval $s = \lfloor L / K \rfloor$ and collects the corresponding hidden states $\{ \mathbf { H } ^ { ( k ) } \} _ { k = 1 } ^ { K }$ , where $\mathbf { H } ^ { ( k ) } \in \bar { \mathbb { R } ^ { B \times N \times D } }$ , stacked as $\mathbf { H } \in \mathbb { R } ^ { B \times \hat { N } \times K \times D }$ . The routing network uses the deepest-layer features $\mathbf { H } ^ { ( K ) }$ as queries, since the top-layer features contain the most complete semantic context and are best suited for determining the role of each token. It computes the routing weights for the i-th token over all layers as:

$$
\mathbf {w} _ {i} = \sigma \left(\frac {g (\mathbf {H} _ {i} ^ {(K)})}{\tau}\right) \in \mathbb {R} ^ {K},
$$

where $\sigma ( \cdot )$ denotes the softmax function, $g ( \cdot )$ is a lightweight routing network, and $\tau$ is a temperature coeficient. The fusion process incorporates learnable per-layer scaling parameters $\alpha \in \mathbb { R } ^ { K }$ :

$$
\hat {\mathbf {H}} _ {i} = \mathbf {W} _ {p} \left(\sum_ {k = 1} ^ {K} w _ {i} ^ {(k)} \cdot \alpha^ {(k)} \cdot \mathbf {H} _ {i} ^ {(k)}\right),
$$

where $\mathbf { W } _ { p }$ is a linear projection. The per-token routing weights $\mathbf { w } _ { i }$ produced by DTR also ofer a novel interpretability perspective: by visualizing the layer preference distributions of diferent types of tokens (image edges, texture regions, text instructions), one can intuitively reveal the model’s internal decision-making mechanism during generation

Self-Aligned Generation. The dual-stream-optimized unified tokenizer possesses both semantic understanding and pixel reconstruction capabilities. We leverage it directly as the representation alignment teacher for the DiT, eliminating the dependence on external pre-trained feature networks [32]. Specifically, we capture the features $f _ { \mathrm { d i t } }$ at a designated intermediate layer of the DiT. These features are mapped through an alignment projection head $\phi _ { a }$ (MLP) and aligned with the projected features $f _ { v } ^ { \prime }$ of the tokenizer’s encoder:

$$
\mathcal {L} _ {\text { align }} = - \frac {1}{N} \sum_ {i = 1} ^ {N} \cos \left(\phi_ {a} (f _ {\text { dit }, i}), f _ {v, i} ^ {\prime}\right),
$$

where $\cos ( \cdot , \cdot )$ denotes cosine similarity. This self-alignment paradigm seamlessly transfers the semantic and pixel knowledge accumulated during tokenizer training to the generation stage, avoiding the distribution bias that external alignment may introduce while ensuring a consistent semantic feature space shared between the understanding and generation.

Training Strategy. The training of the unified model has three stages:

Stage I: Connector Pre-training. The MLLM and DiT are frozen, and only the connector, conditional projection layer, query embeddings, and DTR routing network are trained. Generation data is used with the flow matching loss ${ \mathcal L } _ { \mathrm { f m } }$ as the training objective. The goal of this stage is to enable the connector to learn the mapping from the MLLM’s multimodal output to the DiT’s condition space, while allowing the DTR to learn multi-layer fusion weight distribution.

Table 1: Comparisons of reconstruction quality on the 256 × 256 ImageNet 50k.

<table><tr><td>Model</td><td colspan="2">Ratio rFID↓</td><td>PSNR↑</td><td>SSIM↑</td></tr><tr><td colspan="5">Generative Only Tokenizer</td></tr><tr><td>LlamaGen [40]</td><td>16</td><td>2.19</td><td>20.79</td><td>0.675</td></tr><tr><td>VAR [45]</td><td>16</td><td>1.00</td><td>22.63</td><td>0.755</td></tr><tr><td>Open-MAGVIT2 [28]</td><td>16</td><td>1.67</td><td>22.70</td><td>0.640</td></tr><tr><td>RAE [74]</td><td>16</td><td>0.49</td><td>19.23</td><td>0.620</td></tr><tr><td>SD-VAE [36]</td><td>16</td><td>2.64</td><td>22.13</td><td>0.590</td></tr><tr><td>DC-AE [8]</td><td>32</td><td>0.69</td><td>23.85</td><td>0.660</td></tr><tr><td>VA-VAE [62]</td><td>16</td><td>0.28</td><td>27.96</td><td>0.790</td></tr><tr><td colspan="5">Unified Tokenizer</td></tr><tr><td>VILA-U [56]</td><td>16</td><td>1.80</td><td>-</td><td>-</td></tr><tr><td>Tokenflow [34]</td><td>16</td><td>1.37</td><td>21.41</td><td>0.687</td></tr><tr><td>DualViTok [16]</td><td>16</td><td>1.37</td><td>22.53</td><td>0.741</td></tr><tr><td>DualToken [38]</td><td>16</td><td>0.54</td><td>23.56</td><td>0.742</td></tr><tr><td>EMU2 [41]</td><td>14</td><td>3.27</td><td>13.49</td><td>0.420</td></tr><tr><td>UniLIP [43]</td><td>32</td><td>0.79</td><td>22.99</td><td>0.747</td></tr><tr><td>SPAR</td><td>32</td><td>0.27</td><td>26.65</td><td>0.856</td></tr></table>

Stage II: Joint Pre-training. The MLLM remains frozen while the connector and DiT are jointly trained. The self-alignment loss is added on top of the flow matching loss: ${ \mathcal { L } } _ { \mathrm { f m } } + \lambda _ { a } { \mathcal { L } } _ { \mathrm { a l i g n } }$ . A mixture of generation and editing data is used for training. The DiT releases its parameters in this stage and is co-optimized with the connector, while $\mathcal { L } _ { \mathrm { a l i g n } }$ transfers the tokenizer’s semantic-pixel knowledge to the DiT’s intermediate representation space.

Stage III: Supervised Fine-Tuning. High-quality instruction tuning datasets are used to further improve generation quality and instruction-following capability. The training configuration remains the same as in Stage II.

## 4 Experiments

## 4.1 Experimental Setups

Implementation Details. We implemented two model variants: SPAR-1B and SPAR-3B. SPAR-1B employs InternVL3-1B [76] as the MLLM backbone, which comprises InternViT as the vision encoder and Qwen2.5-0.5B [35] as the language model, paired with SANA-0.6B [58] as the DiT. SPAR-3B adopts InternVL3-2B as the MLLM backbone with a Qwen2.5-1.5B language model and SANA-1.6B DiT. Both variants reuse the InternViT from InternVL3 as the vision encoder and employ the decoder from DC-AE [8] as the pixel decoder. In the asymmetric dual-stream module, the pixel stream uses a 6-layer Transformer encoder, and the semantic stream uses 3 residual blocks. DTR samples 4 layers by default with temperature coeficient τ = 1.0.

Training Data. For generation tasks, we used the combination of a 27M recaptioned data publicly released by BLIP3o [6], a 5M subset of CC12M [5], and 4M synthetic images from JourneyDB [39]. For editing tasks, we used the GPT-Image-Edit [52] dataset. In the SFT stage, we employ the BLIP3o-60K and ShareGPT-4o-Image [7] high-quality datasets. Since we froze the LLM throughout training, data for understanding tasks is not required.

Table 2: Comparison with state-of-the-arts on visual understanding benchmarks.

<table><tr><td>Model</td><td>LLM Params</td><td>MME-P</td><td>MMB</td><td>MMMU</td><td>MM-Vet</td><td>SEED</td><td>MMVP</td></tr><tr><td colspan="8">Und. Only</td></tr><tr><td>LLaVA-OV [21]</td><td>1B</td><td>1238</td><td>52.1</td><td>31.4</td><td>29.1</td><td>65.5</td><td>-</td></tr><tr><td>InternVL3-1B [76]</td><td>1B</td><td>1492</td><td>72.6</td><td>43.4</td><td>59.5</td><td>71.1</td><td>67.3</td></tr><tr><td>InternVL3-2B [76]</td><td>2B</td><td>1633</td><td>80.6</td><td>48.2</td><td>62.2</td><td>75.0</td><td>72.7</td></tr><tr><td>Qwen2.5-VL-3B [2]</td><td>3B</td><td>-</td><td>79.1</td><td>53.1</td><td>61.8</td><td>-</td><td>-</td></tr><tr><td>Emu3-Chat-8B [51]</td><td>8B</td><td>1244</td><td>58.5</td><td>31.6</td><td>37.2</td><td>68.2</td><td>36.6</td></tr><tr><td colspan="8">Und. and Gen.</td></tr><tr><td>Chameleon-7B [44]</td><td>7B</td><td>-</td><td>35.7</td><td>28.4</td><td>8.3</td><td>-</td><td>0.0</td></tr><tr><td>VILA-U-7B [56]</td><td>7B</td><td>1336</td><td>66.6</td><td>32.2</td><td>27.7</td><td>56.3</td><td>22.0</td></tr><tr><td>MetaMorph-8B [46]</td><td>8B</td><td>-</td><td>75.2</td><td>41.8</td><td>-</td><td>-</td><td>48.3</td></tr><tr><td>SEED-X-13B [13]</td><td>13B</td><td>1457</td><td>70.1</td><td>35.6</td><td>43.0</td><td>66.5</td><td>-</td></tr><tr><td>Show-O-1.3B [59]</td><td>1.3B</td><td>1097</td><td>-</td><td>26.7</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Janus-Pro-7B [9]</td><td>7B</td><td>1567</td><td>79.2</td><td>41.0</td><td>50.0</td><td>72.1</td><td>-</td></tr><tr><td>Harmon-1.5B [55]</td><td>1.5B</td><td>1155</td><td>65.5</td><td>38.9</td><td>-</td><td>67.1</td><td>-</td></tr><tr><td>BAGEL-7B [10]</td><td>3B</td><td>1610</td><td>79.2</td><td>43.2</td><td>48.2</td><td>-</td><td>54.7</td></tr><tr><td>BLIP3-o-4B [6]</td><td>4B</td><td>1528</td><td>78.6</td><td>46.6</td><td>60.1</td><td>73.8</td><td>-</td></tr><tr><td>TokLIP-7B [24]</td><td>7B</td><td>1410</td><td>-</td><td>42.1</td><td>-</td><td>65.2</td><td>-</td></tr><tr><td>Tar-7B [15]</td><td>7B</td><td>1571</td><td>74.4</td><td>39.0</td><td>-</td><td>73.0</td><td>-</td></tr><tr><td>SPAR-1B</td><td>1B</td><td>1500</td><td>73.0</td><td>43.2</td><td>59.8</td><td>71.5</td><td>68.9</td></tr><tr><td>SPAR-3B</td><td>2B</td><td>1638</td><td>80.7</td><td>48.7</td><td>62.2</td><td>75.1</td><td>73.3</td></tr></table>

## 4.2 Comparisons with SOTA Methods

Image Reconstruction. Table 1 presents the comparison of image reconstruction quality on the ImageNet 50k validation set at 256 × 256 resolution. Compared to existing unified tokenizers, SPAR achieves state-of-the-art performance. Specifically, our model attains an rFID of 0.27, a PSNR of 26.65, and an SSIM of 0.856, significantly outperforming previous unified methods across all three metrics. Furthermore, compared with generative-only tokenizers, our model still demonstrates highly competitive reconstruction quality, outperforming strong generative baselines such as VA-VAE in SSIM (0.856 vs. 0.790). This indicates that although our tokenizer simultaneously supports both understanding and generation within a unified framework, its reconstruction capability remains competitive with tokenizers designed exclusively for generation. Consequently, it provides a solid foundation for subsequent unified generation models built upon this representation space.

Multimodal Understanding. Table 2 presents the comparison of our model with recent advanced methods across various [12, 22, 27, 47, 67, 68] visual understanding benchmarks. In our implementation, we replace the original Vision Encoder of InternVL3 with the ViT from our SPAR tokenizer. Compared to the pure understanding baseline, InternVL3, our model demonstrates superior performance across multiple benchmarks. Notably, although we unfreeze the visual encoder during training, SPAR does not sufer from any performance degradation. On the contrary, benefiting from our dual-stream self-alignment mechanism, the model achieves a noticeable performance enhancement in these understanding tasks. Furthermore, in comparison with contemporary unified models, SPAR consistently achieves superior results across all evaluated metrics.

Table 3: Evaluation of text-to-image generation on GenEval and WISE benchmark.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Params</td><td colspan="3">GenEval</td><td colspan="3">WISE</td></tr><tr><td>Counting</td><td>Position</td><td>Overall</td><td>Cultural</td><td>Biology</td><td>Overall</td></tr><tr><td colspan="8">Gen. Only</td></tr><tr><td>SDXL [33]</td><td>2.6B</td><td>0.39</td><td>0.15</td><td>0.55</td><td>0.43</td><td>0.44</td><td>0.43</td></tr><tr><td>FLUX.1-dev [18]</td><td>12B</td><td>0.75</td><td>0.68</td><td>0.82</td><td>0.48</td><td>0.42</td><td>0.50</td></tr><tr><td>Emu3-Gen [51]</td><td>8B</td><td>0.34</td><td>0.17</td><td>0.54</td><td>0.34</td><td>0.41</td><td>0.39</td></tr><tr><td>SD3-Medium [11]</td><td>2B</td><td>0.72</td><td>0.33</td><td>0.74</td><td>0.42</td><td>0.39</td><td>0.42</td></tr><tr><td>Sana-1.6B [58]</td><td>1.6B</td><td>0.62</td><td>0.21</td><td>0.66</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="8">Und. and Gen.</td></tr><tr><td>VILA-U [56]</td><td>7B</td><td>-</td><td>-</td><td>-</td><td>0.26</td><td>0.35</td><td>0.31</td></tr><tr><td>TokenFlow-XL [34]</td><td>14B</td><td>0.41</td><td>0.16</td><td>0.55</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Janus-Pro [9]</td><td>7B</td><td>0.59</td><td>0.79</td><td>0.80</td><td>0.30</td><td>0.36</td><td>0.35</td></tr><tr><td>Harmon [55]</td><td>3B</td><td>0.66</td><td>0.74</td><td>0.76</td><td>0.38</td><td>0.37</td><td>0.41</td></tr><tr><td>BLIP3-o-8B [6]</td><td>7B</td><td>-</td><td>-</td><td>0.84</td><td>-</td><td>-</td><td>0.62</td></tr><tr><td>BAGEL [10]</td><td>7B</td><td>0.81</td><td>0.64</td><td>0.82</td><td>0.44</td><td>0.44</td><td>0.52</td></tr><tr><td>OpenUni-B [54]</td><td>1B</td><td>0.74</td><td>0.77</td><td>0.84</td><td>0.37</td><td>0.39</td><td>0.43</td></tr><tr><td>OpenUni-L [54]</td><td>3B</td><td>0.77</td><td>0.75</td><td>0.85</td><td>0.51</td><td>0.48</td><td>0.52</td></tr><tr><td>Show-o2 [60]</td><td>7B</td><td>0.58</td><td>0.52</td><td>0.76</td><td>0.33</td><td>0.39</td><td>0.39</td></tr><tr><td>Tar [15]</td><td>7B</td><td>0.83</td><td>0.80</td><td>0.84</td><td>-</td><td>-</td><td>-</td></tr><tr><td>SPAR-1B</td><td>1B</td><td>0.83</td><td>0.85</td><td>0.89</td><td>0.54</td><td>0.51</td><td>0.57</td></tr><tr><td>SPAR-3B</td><td>3B</td><td>0.84</td><td>0.87</td><td>0.91</td><td>0.67</td><td>0.61</td><td>0.64</td></tr></table>

Image Generation. Table 3 reports the text-to-image generation results on the GenEval [14] and WISE [30] benchmarks, and Figure 4 illustrates the qualitative results of our method. We compare SPAR with two categories of methods: generation-only models and unified models that support both understanding and generation. SPAR-3B achieves an overall score of 0.91 on GenEval, ranking first among all compared methods, with the best scores of 0.84 in Counting and 0.87 in Position. Notably, as a unified model, SPAR surpasses even the much larger generation-only model, demonstrating that SPAR’s representation space efectively accommodates both understanding and generation without performance degradation. On the knowledge-intensive WISE benchmark, SPAR-3B attains an overall score of 0.64, outperforming all compared methods, further validating that the rich semantic information preserved by the SPAR tokenizer benefits generation quality.

Image Editing. We evaluate image editing capabilities on ImgEdit-Bench as shown in Table 4. SPAR-3B achieves an overall score of 4.01, the highest among all open-source methods, closely approaching the proprietary GPT-4o (4.20).

![](images/b005c543d5486f605099d770810fc07af51c0de662f289f73740fe383d477b06.jpg)  
Fig. 4: Qualitative results of image generation.

Compared with the second-best OmniGen2 (3.44), SPAR-3B yields an absolute improvement of +0.58. Across specific editing categories, SPAR-3B achieves the best performance on various subtypes, with particularly pronounced advantages on semantically demanding tasks, indicating that the rich semantic representations within the SPAR tokenizer efectively enhance the editing model’s ability to precisely comprehend and execute editing instructions.

## 4.3 Ablation Studies

Unified Tokenizer. Table 5 investigates the contribution of each component in the unified tokenizer. Removing the pixel stream (Row b) leads to a clear reconstruction degradation (PSNR drops by 2.03, SSIM by 0.068), while the understanding metrics remain nearly unchanged, confirming that the semantic encoder alone cannot recover suficient pixel-level details and the pixel stream is essential for bridging this gap. Removing the semantic stream (Row c) yields better reconstruction than Row b, yet causes a catastrophic collapse in understanding (MMB drops from 73.0 to 18.4, MME-P from 1500 to 709), demonstrating that without the lightweight semantic anchor, end-to-end pixel-oriented optimization destroys the encoder’s discriminative structure. The contrast between Rows b and c validates the asymmetric design: the semantic stream preserves understanding at minimal cost, while the heavier pixel stream shoulders the reconstruction burden. Finally, freezing the encoder throughout training (Row d) preserves understanding well but results in severely degraded reconstruction (rFID 6.14, PSNR 16.26), verifying that Stage III encoder fine-tuning with semantic self-distillation is critical for endowing the encoder with pixel representation capability.

Table 4: Evaluation of image editing ability on ImgEdit benchmark.

<table><tr><td>Model</td><td>Add</td><td>Adj.</td><td>Ext.</td><td>Repl.</td><td>Rmv.</td><td>Bkg.</td><td>Style</td><td>Hyb.</td><td>Act.</td><td>Overall</td></tr><tr><td>GPT-4o [31]</td><td>4.61</td><td>4.33</td><td>2.9</td><td>4.35</td><td>3.66</td><td>4.57</td><td>4.93</td><td>3.96</td><td>4.89</td><td>4.20</td></tr><tr><td>MagicBrush [70]</td><td>2.84</td><td>1.58</td><td>1.51</td><td>1.97</td><td>1.58</td><td>1.75</td><td>2.38</td><td>1.62</td><td>1.22</td><td>1.90</td></tr><tr><td>Instruct-P2P [4]</td><td>2.45</td><td>1.83</td><td>1.44</td><td>2.01</td><td>1.50</td><td>1.44</td><td>3.55</td><td>1.20</td><td>1.46</td><td>1.88</td></tr><tr><td>AnyEdit [64]</td><td>3.18</td><td>2.95</td><td>1.88</td><td>2.47</td><td>2.23</td><td>2.24</td><td>2.85</td><td>1.56</td><td>2.65</td><td>2.45</td></tr><tr><td>UltraEdit [73]</td><td>3.44</td><td>2.81</td><td>2.13</td><td>2.96</td><td>1.45</td><td>2.83</td><td>3.76</td><td>1.91</td><td>2.98</td><td>2.70</td></tr><tr><td>OmniGen [57]</td><td>3.47</td><td>3.04</td><td>1.71</td><td>2.94</td><td>2.43</td><td>3.21</td><td>4.19</td><td>2.24</td><td>3.38</td><td>2.96</td></tr><tr><td>Step1X-Edit [26]</td><td>3.88</td><td>3.14</td><td>1.76</td><td>3.40</td><td>2.41</td><td>3.16</td><td>4.63</td><td>2.64</td><td>2.52</td><td>3.06</td></tr><tr><td>ICEdit [72]</td><td>3.58</td><td>3.39</td><td>1.73</td><td>3.15</td><td>2.93</td><td>3.08</td><td>3.84</td><td>2.04</td><td>3.68</td><td>3.05</td></tr><tr><td>BAGEL [10]</td><td>3.56</td><td>3.31</td><td>1.70</td><td>3.30</td><td>2.62</td><td>3.24</td><td>4.49</td><td>2.38</td><td>4.17</td><td>3.20</td></tr><tr><td>UniWorld-V1 [23]</td><td>3.82</td><td>3.64</td><td>2.27</td><td>3.47</td><td>3.24</td><td>2.99</td><td>4.21</td><td>2.96</td><td>2.74</td><td>3.26</td></tr><tr><td>OmniGen2 [53]</td><td>3.57</td><td>3.06</td><td>1.77</td><td>3.74</td><td>3.20</td><td>3.57</td><td>4.81</td><td>2.52</td><td>4.68</td><td>3.44</td></tr><tr><td>SPAR-3B</td><td>4.31</td><td>3.93</td><td>2.32</td><td>4.52</td><td>4.15</td><td>4.20</td><td>4.87</td><td>3.12</td><td>4.69</td><td>4.01</td></tr></table>

Table 5: Ablation on unified tokenizer. Evaluated on ImageNet 50k (256 × 256) for reconstruction and multimodal benchmarks for understanding.

<table><tr><td rowspan="2" colspan="2">Row Setting</td><td colspan="3">Reconstruction</td><td colspan="3">Understanding</td></tr><tr><td>rFID↓</td><td>PSNR↑</td><td>SSIM↑</td><td>MME-P</td><td>MMB</td><td>MMVP</td></tr><tr><td>(a)</td><td>Full Model</td><td>0.27</td><td>26.65</td><td>0.856</td><td>1500</td><td>73.0</td><td>68.9</td></tr><tr><td>(b)</td><td>w/o Pixel Stream</td><td>0.31</td><td>24.62</td><td>0.788</td><td>1499</td><td>72.6</td><td>68.7</td></tr><tr><td>(c)</td><td>w/o Semantic Stream</td><td>0.29</td><td>25.28</td><td>0.804</td><td>709</td><td>18.4</td><td>50.0</td></tr><tr><td>(d)</td><td>Frozen Encoder</td><td>6.14</td><td>16.26</td><td>0.572</td><td>1492</td><td>72.6</td><td>67.3</td></tr></table>

Unified Model. Table 6 evaluates the two key components of the unified model. Replacing DTR with last-layer-only extraction (Row b) causes the largest overall drop, indicating that adaptively aggregating multi-layer MLLM features provides richer structural and semantic cues than relying solely on the top-layer representation. Removing the self-alignment loss $\mathcal { L } _ { \mathrm { a l i g n } }$ (Row c) also degrades all three metrics, confirming that leveraging the dual-stream tokenizer as an alignment teacher efectively transfers the accumulated semantic-pixel knowledge to the DiT and improves generation quality without any external feature network.

Table 6: Ablation on unified model training.

<table><tr><td colspan="2">Row Setting</td><td>GenEval↑</td><td>WISE↑</td><td>ImgEdit↑</td></tr><tr><td>(a)</td><td>Full Model</td><td>0.89</td><td>0.57</td><td>3.85</td></tr><tr><td>(b)</td><td>w/o DTR</td><td>0.86</td><td>0.54</td><td>3.73</td></tr><tr><td>(c)</td><td>w/o Self-Alignment  $\mathcal{L}_{align}$ </td><td>0.86</td><td>0.53</td><td>3.78</td></tr></table>

## 5 Conclusion

In this paper, we presented SPAR, a unified multimodal framework that addresses the fundamental feature discrepancy between semantic perception and pixel-level generation. To overcome the detail loss inherent in semantic spaces, we introduced an asymmetric dual-stream tokenizer that explicitly decouples semantic preservation from high-quality pixel reconstruction. Furthermore, we proposed dynamic token routing to adaptively harness multi-layer MLLM representations, along with a self- lignment paradigm that eliminates the reliance on external representation teachers. Extensive experiments demonstrate that SPAR achieves state-of-the-art performance across diverse benchmarks, delivering highfidelity image generation and complex editing without degrading the pre-trained understanding capabilities.

## References

1. Bai, S., Cai, Y., Chen, R., Chen, K., Chen, X., Cheng, Z., Deng, L., Ding, W., Gao, C., Ge, C., Ge, W., Guo, Z., Huang, Q., Huang, J., Huang, F., Hui, B., Jiang, S., Li, Z., Li, M., Li, M., Li, K., Lin, Z., Lin, J., Liu, X., Liu, J., Liu, C., Liu, Y., Liu, D., Liu, S., Lu, D., Luo, R., Lv, C., Men, R., Meng, L., Ren, X., Ren, X., Song, S., Sun, Y., Tang, J., Tu, J., Wan, J., Wang, P., Wang, P., Wang, Q., Wang, Y., Xie, T., Xu, Y., Xu, H., Xu, J., Yang, Z., Yang, M., Yang, J., Yang, A., Yu, B., Zhang, F., Zhang, H., Zhang, X., Zheng, B., Zhong, H., Zhou, J., Zhou, F., Zhou, J., Zhu, Y., Zhu, K.: Qwen3-vl technical report. arXiv preprint arXiv:2511.21631 (2025) 1

2. Bai, S., Chen, K., Liu, X., Wang, J., Ge, W., Song, S., Dang, K., Wang, P., Wang, S., Tang, J., Zhong, H., Zhu, Y., Yang, M., Li, Z., Wan, J., Wang, P., Ding, W., Fu, Z., Xu, Y., Ye, J., Zhang, X., Xie, T., Cheng, Z., Zhang, H., Yang, Z., Xu, H., Lin, J.: Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923 (2025) 11

3. Black Forest Labs: FLUX.2: Analyzing and enhancing the latent space of FLUX – representation comparison (2025), https://bfl.ai/research/representationcomparison 3

4. Brooks, T., Holynski, A., Efros, A.A.: Instructpix2pix: Learning to follow image editing instructions. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 18392–18402 (2023) 14

5. Changpinyo, S., Sharma, P., Ding, N., Soricut, R.: Conceptual 12m: Pushing webscale image-text pre-training to recognize long-tail visual concepts. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3558–3568 (2021) 10

6. Chen, J., Xu, Z., Pan, X., Hu, Y., Qin, C., Goldstein, T., Huang, L., Zhou, T., Xie, S., Savarese, S., et al.: Blip3-o: A family of fully open unified multimodal

models-architecture, training and dataset. arXiv preprint arXiv:2505.09568 (2025) 4, 10, 11, 12

7. Chen, J., Cai, Z., Chen, P., Chen, S., Ji, K., Wang, X., Yang, Y., Wang, B.: Sharegpt-4o-image: Aligning multimodal models with gpt-4o-level image generation (2025), https://arxiv.org/abs/2506.18095 11

8. Chen, J., Cai, H., Chen, J., Xie, E., Yang, S., Tang, H., Li, M., Lu, Y., Han, S.: Deep compression autoencoder for eficient high-resolution difusion models. arXiv preprint arXiv:2410.10733 (2024) 10

9. Chen, X., Wu, Z., Liu, X., Pan, Z., Liu, W., Xie, Z., Yu, X., Ruan, C.: Janus-pro: Unified multimodal understanding and generation with data and model scaling. arXiv preprint arXiv:2501.17811 (2025) 11, 12

10. Deng, C., Zhu, D., Li, K., Gou, C., Li, F., Wang, Z., Zhong, S., Yu, W., Nie, X., Song, Z., Shi, G., Fan, H.: Emerging properties in unified multimodal pretraining. arXiv preprint arXiv:2505.14683 (2025) 4, 11, 12, 14

11. Esser, P., Kulal, S., Blattmann, A., Entezari, R., Müller, J., Saini, H., Levi, Y., Lorenz, D., Sauer, A., Boesel, F., et al.: Scaling rectified flow transformers for high-resolution image synthesis. In: Forty-first international conference on machine learning (2024) 2, 12

12. Fu, C., Chen, P., Shen, Y., Qin, Y., Zhang, M., Lin, X., Yang, J., Zheng, X., Li, K., Sun, X., et al.: Mme: A comprehensive evaluation benchmark for multimodal large language models. arXiv preprint arXiv:2306.13394 (2023) 11

13. Ge, Y., Zhao, S., Zhu, J., Ge, Y., Yi, K., Song, L., Li, C., Ding, X., Shan, Y.: Seed-x: Multimodal models with unified multi-granularity comprehension and generation. arXiv preprint arXiv:2404.14396 (2024) 11

14. Ghosh, D., Hajishirzi, H., Schmidt, L.: Geneval: An object-focused framework for evaluating text-to-image alignment. Advances in Neural Information Processing Systems 36, 52132–52152 (2023) 12

15. Han, J., Chen, H., Zhao, Y., Wang, H., Zhao, Q., Yang, Z., He, H., Yue, X., Jiang, L.: Vision as a dialect: Unifying visual understanding and generation via text-aligned representations. arXiv preprint arXiv:2506.18898 (2025) 11, 12

16. Huang, R., Wang, C., Yang, J., Lu, G., Yuan, Y., Han, J., Hou, L., Zhang, W., Hong, L., Zhao, H., et al.: Illume+: Illuminating unified mllm with dual visual tokenization and difusion refinement. arXiv preprint arXiv:2504.01934 (2025) 10

17. Kingma, D.P., Welling, M.: Auto-encoding variational bayes. arXiv preprint arXiv:1312.6114 (2013) 2

18. Labs, B.F.: Flux. https://github.com/black-forest-labs/flux (2024) 12

19. Labs, B.F.: FLUX.2: Frontier Visual Intelligence. https://bfl.ai/blog/flux-2 (2025) 2

20. Labs, B.F., Batifol, S., Blattmann, A., Boesel, F., Consul, S., Diagne, C., Dockhorn, T., English, J., English, Z., Esser, P., Kulal, S., Lacey, K., Levi, Y., Li, C., Lorenz, D., Müller, J., Podell, D., Rombach, R., Saini, H., Sauer, A., Smith, L.: Flux.1 kontext: Flow matching for in-context image generation and editing in latent space (2025), https://arxiv.org/abs/2506.15742 2

21. Li, B., Zhang, Y., Guo, D., Zhang, R., Li, F., Zhang, H., Zhang, K., Zhang, P., Li, Y., Liu, Z., et al.: Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326 (2024) 11

22. Li, B., Ge, Y., Ge, Y., Wang, G., Wang, R., Zhang, R., Shan, Y.: Seedbench: Benchmarking multimodal large language models. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13299– 13308 (2024) 11

23. Lin, B., Li, Z., Cheng, X., Niu, Y., Ye, Y., He, X., Yuan, S., Yu, W., Wang, S., Ge, Y., et al.: Uniworld-v1: High-resolution semantic encoders for unified visual understanding and generation. arXiv preprint arXiv:2506.03147 (2025) 4, 14

24. Lin, H., Wang, T., Ge, Y., Ge, Y., Lu, Z., Wei, Y., Zhang, Q., Sun, Z., Shan, Y.: Toklip: Marry visual tokens to clip for multimodal comprehension and generation. arXiv preprint arXiv:2505.05422 (2025) 11

25. Liu, H., Li, C., Wu, Q., Lee, Y.J.: Visual instruction tuning. Advances in neural information processing systems 36, 34892–34916 (2023) 1

26. Liu, S., Han, Y., Xing, P., Yin, F., Wang, R., Cheng, W., Liao, J., Wang, Y., Fu, H., Han, C., et al.: Step1x-edit: A practical framework for general image editing. arXiv preprint arXiv:2504.17761 (2025) 14

27. Liu, Y., Duan, H., Zhang, Y., Li, B., Zhang, S., Zhao, W., Yuan, Y., Wang, J., He, C., Liu, Z., et al.: Mmbench: Is your multi-modal model an all-around player? In: European conference on computer vision. pp. 216–233. Springer (2024) 11

28. Luo, Z., Shi, F., Ge, Y., Yang, Y., Wang, L., Shan, Y.: Open-magvit2: An open-source project toward democratizing auto-regressive visual generation (2025), https://arxiv.org/abs/2409.04410 10

29. Ma, C., Jiang, Y., Wu, J., Yang, J., Yu, X., Yuan, Z., Peng, B., Qi, X.: Unitok: A unified tokenizer for visual generation and understanding. arXiv preprint arXiv:2502.20321 (2025) 4

30. Niu, Y., Ning, M., Zheng, M., Jin, W., Lin, B., Jin, P., Liao, J., Ning, K., Feng, C., Zhu, B., Yuan, L.: Wise: A world knowledge-informed semantic evaluation for text-to-image generation. arXiv preprint arXiv:2503.07265 (2025) 12

31. OpenAI: Introducing 4o Image Generation (2025), https://openai.com/index/ introducing-4o-image-generation/ 14

32. Oquab, M., Darcet, T., Moutakanni, T., Vo, H.V., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Howes, R., Huang, P.Y., Xu, H., Sharma, V., Li, S.W., Galuba, W., Rabbat, M., Assran, M., Ballas, N., Synnaeve, G., Misra, I., Jegou, H., Mairal, J., Labatut, P., Joulin, A., Bojanowski, P.: Dinov2: Learning robust visual features without supervision (2023) 2, 3, 4, 9

33. Podell, D., English, Z., Lacey, K., Blattmann, A., Dockhorn, T., Müller, J., Penna, J., Rombach, R.: Sdxl: Improving latent difusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952 (2023) 12

34. Qu, L., Zhang, H., Liu, Y., Wang, X., Jiang, Y., Gao, Y., Ye, H., Du, D.K., Yuan, Z., Wu, X.: Tokenflow: Unified image tokenizer for multimodal understanding and generation. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 2545–2555 (2025) 4, 10, 12

35. Qwen, :, Yang, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Li, C., Liu, D., Huang, F., Wei, H., Lin, H., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Lin, J., Dang, K., Lu, K., Bao, K., Yang, K., Yu, L., Li, M., Xue, M., Zhang, P., Zhu, Q., Men, R., Lin, R., Li, T., Tang, T., Xia, T., Ren, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Wan, Y., Liu, Y., Cui, Z., Zhang, Z., Qiu, Z.: Qwen2.5 technical report (2025), https://arxiv.org/abs/2412.15115 10

36. Rombach, R., Blattmann, A., Lorenz, D., Esser, P., Ommer, B.: High-resolution image synthesis with latent difusion models. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 10684–10695 (2022) 2, 10

37. Shi, M., Wang, H., Zheng, W., Yuan, Z., Wu, X., Wang, X., Wan, P., Zhou, J., Lu, J.: Latent difusion model without variational autoencoder (2025), https: //arxiv.org/abs/2510.15301 4

38. Song, W., Wang, Y., Song, Z., Li, Y., Sun, H., Chen, W., Zhou, Z., Xu, J., Wang, J., Yu, K.: Dualtoken: Towards unifying visual understanding and generation with dual visual vocabularies. arXiv preprint arXiv:2503.14324 (2025) 10

39. Sun, K., Pan, J., Ge, Y., Li, H., Duan, H., Wu, X., Zhang, R., Zhou, A., Qin, Z., Wang, Y., et al.: Journeydb: A benchmark for generative image understanding. Advances in neural information processing systems 36, 49659–49678 (2023) 11

40. Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., Yuan, Z.: Autoregressive model beats difusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525 (2024) 10

41. Sun, Q., Cui, Y., Zhang, X., Zhang, F., Yu, Q., Wang, Y., Rao, Y., Liu, J., Huang, T., Wang, X.: Generative multimodal models are in-context learners. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 14398–14409 (2024) 4, 10

42. Sun, Q., Yu, Q., Cui, Y., Zhang, F., Zhang, X., Wang, Y., Gao, H., Liu, J., Huang, T., Wang, X.: Emu: Generative pretraining in multimodality. arXiv preprint arXiv:2307.05222 (2023) 4

43. Tang, H., Xie, C., Bao, X., Weng, T., Li, P., Zheng, Y., Wang, L.: Unilip: Adapting clip for unified multimodal understanding, generation and editing. arXiv preprint arXiv:2507.23278 (2025) 4, 10

44. Team, C.: Chameleon: Mixed-modal early-fusion foundation models. arXiv preprint arXiv:2405.09818 (2024) 4, 11

45. Tian, K., Jiang, Y., Yuan, Z., Peng, B., Wang, L.: Visual autoregressive modeling: Scalable image generation via next-scale prediction. Advances in neural information processing systems 37, 84839–84865 (2024) 2, 10

46. Tong, S., Fan, D., Zhu, J., Xiong, Y., Chen, X., Sinha, K., Rabbat, M., LeCun, Y., Xie, S., Liu, Z.: Metamorph: Multimodal understanding and generation via instruction tuning. arXiv preprint arXiv:2412.14164 (2024) 11

47. Tong, S., Liu, Z., Zhai, Y., Ma, Y., LeCun, Y., Xie, S.: Eyes wide shut? exploring the visual shortcomings of multimodal llms. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9568–9578 (2024) 11

48. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., et al.: Llama: Open and eficient foundation language models. arXiv preprint arXiv:2302.13971 (2023) 1

49. Van Den Oord, A., Vinyals, O., et al.: Neural discrete representation learning. Advances in neural information processing systems 30 (2017) 4

50. Wang, W., Gao, Z., Gu, L., Pu, H., Cui, L., Wei, X., Liu, Z., Jing, L., Ye, S., Shao, J., et al.: Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and eficiency. arXiv preprint arXiv:2508.18265 (2025) 1

51. Wang, X., Zhang, X., Luo, Z., Sun, Q., Cui, Y., Wang, J., Zhang, F., Wang, Y., Li, Z., Yu, Q., et al.: Emu3: Next-token prediction is all you need. arXiv preprint arXiv:2409.18869 (2024) 11, 12

52. Wang, Y., Yang, S., Zhao, B., Zhang, L., Liu, Q., Zhou, Y., Xie, C.: Gptimage-edit-1.5 m: A million-scale, gpt-generated image dataset. arXiv preprint arXiv:2507.21033 (2025) 11

53. Wu, C., Zheng, P., Yan, R., Xiao, S., Luo, X., Wang, Y., Li, W., Jiang, X., Liu, Y., Zhou, J., et al.: Omnigen2: Exploration to advanced multimodal generation. arXiv preprint arXiv:2506.18871 (2025) 14

54. Wu, S., Wu, Z., Gong, Z., Tao, Q., Jin, S., Li, Q., Li, W., Loy, C.C.: Openuni: A simple baseline for unified multimodal understanding and generation. arXiv preprint arXiv:2505.23661 (2025) 4, 12

55. Wu, S., Zhang, W., Xu, L., Jin, S., Wu, Z., Tao, Q., Liu, W., Li, W., Loy, C.C.: Harmonizing visual representations for unified multimodal understanding and generation (2025), https://arxiv.org/abs/2503.21979 11, 12

56. Wu, Y., Zhang, Z., Chen, J., Tang, H., Li, D., Fang, Y., Zhu, L., Xie, E., Yin, H., Yi, L., et al.: Vila-u: a unified foundation model integrating visual understanding and generation. arXiv preprint arXiv:2409.04429 (2024) 4, 10, 11, 12

57. Xiao, S., Wang, Y., Zhou, J., Yuan, H., Xing, X., Yan, R., Li, C., Wang, S., Huang, T., Liu, Z.: Omnigen: Unified image generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 13294–13304 (2025) 14

58. Xie, E., Chen, J., Chen, J., Cai, H., Tang, H., Lin, Y., Zhang, Z., Li, M., Zhu, L., Lu, Y., Han, S.: Sana: Eficient high-resolution image synthesis with linear difusion transformer (2024), https://arxiv.org/abs/2410.10629 10, 12

59. Xie, J., Mao, W., Bai, Z., Zhang, D.J., Wang, W., Lin, K.Q., Gu, Y., Chen, Z., Yang, Z., Shou, M.Z.: Show-o: One single transformer to unify multimodal understanding and generation. arXiv preprint arXiv:2408.12528 (2024) 4, 11

60. Xie, J., Yang, Z., Shou, M.Z.: Show-o2: Improved native unified multimodal models. arXiv preprint arXiv:2506.15564 (2025) 12

61. Yang, A., Li, A., Yang, B., Zhang, B., Hui, B., Zheng, B., Yu, B., Gao, C., Huang, C., Lv, C., Zheng, C., Liu, D., Zhou, F., Huang, F., Hu, F., Ge, H., Wei, H., Lin, H., Tang, J., Yang, J., Tu, J., Zhang, J., Yang, J., Yang, J., Zhou, J., Zhou, J., Lin, J., Dang, K., Bao, K., Yang, K., Yu, L., Deng, L., Li, M., Xue, M., Li, M., Zhang, P., Wang, P., Zhu, Q., Men, R., Gao, R., Liu, S., Luo, S., Li, T., Tang, T., Yin, W., Ren, X., Wang, X., Zhang, X., Ren, X., Fan, Y., Su, Y., Zhang, Y., Zhang, Y., Wan, Y., Liu, Y., Wang, Z., Cui, Z., Zhang, Z., Zhou, Z., Qiu, Z.: Qwen3 technical report. arXiv preprint arXiv:2505.09388 (2025) 1

62. Yao, J., Yang, B., Wang, X.: Reconstruction vs. generation: Taming optimization dilemma in latent difusion models. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 15703–15712 (2025) 2, 3, 10

63. Yu, L., Shi, B., Pasunuru, R., Muller, B., Golovneva, O., Wang, T., Babu, A., Tang, B., Karrer, B., Sheynin, S., Ross, C., Polyak, A., Howes, R., Sharma, V., Xu, P., Tamoyan, H., Ashual, O., Singer, U., Li, S.W., Zhang, S., James, R., Ghosh, G., Taigman, Y., Fazel-Zarandi, M., Celikyilmaz, A., Zettlemoyer, L., Aghajanyan, A.: Scaling autoregressive multi-modal models: Pretraining and instruction tuning (2023), https://arxiv.org/abs/2309.02591 4

64. Yu, Q., Chow, W., Yue, Z., Pan, K., Wu, Y., Wan, X., Li, J., Tang, S., Zhang, H., Zhuang, Y.: Anyedit: Mastering unified high-quality image editing for any idea. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 26125–26135 (2025) 14

65. Yu, Q., Weber, M., Deng, X., Shen, X., Cremers, D., Chen, L.C.: An image is worth 32 tokens for reconstruction and generation. Advances in Neural Information Processing Systems 37, 128940–128966 (2024) 2

66. Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., Xie, S.: Representation alignment for generation: Training difusion transformers is easier than you think. In: International Conference on Learning Representations (2025) 2, 3, 4

67. Yu, W., Yang, Z., Li, L., Wang, J., Lin, K., Liu, Z., Wang, X., Wang, L.: Mmvet: Evaluating large multimodal models for integrated capabilities. arXiv preprint arXiv:2308.02490 (2023) 11

68. Yue, X., Ni, Y., Zhang, K., Zheng, T., Liu, R., Zhang, G., Stevens, S., Jiang, D., Ren, W., Sun, Y., et al.: Mmmu: A massive multi-discipline multimodal

understanding and reasoning benchmark for expert agi. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 9556–9567 (2024) 11

69. Yue, Z., Zhang, H., Zeng, X., Chen, B., Wang, C., Zhuang, S., Dong, L., Du, K., Wang, Y., Wang, L., et al.: Uniflow: A unified pixel flow tokenizer for visual understanding and generation. arXiv preprint arXiv:2510.10575 (2025) 4

70. Zhang, K., Mo, L., Chen, W., Sun, H., Su, Y.: Magicbrush: A manually annotated dataset for instruction-guided image editing. Advances in Neural Information Processing Systems 36, 31428–31449 (2023) 14

71. Zhang, S., Zhang, H., Zhang, Z., Ge, C., Xue, S., Liu, S., Ren, M., Kim, S.Y., Zhou, Y., Liu, Q., et al.: Both semantics and reconstruction matter: Making representation encoders ready for text-to-image generation and editing. arXiv preprint arXiv:2512.17909 (2025) 3, 4

72. Zhang, Z., Xie, J., Lu, Y., Yang, Z., Yang, Y.: Enabling instructional image editing with in-context generation in large scale difusion transformer. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2025) 14

73. Zhao, H., Ma, X.S., Chen, L., Si, S., Wu, R., An, K., Yu, P., Zhang, M., Li, Q., Chang, B.: Ultraedit: Instruction-based fine-grained image editing at scale. Advances in Neural Information Processing Systems 37, 3058–3093 (2024) 14

74. Zheng, B., Ma, N., Tong, S., Xie, S.: Difusion transformers with representation autoencoders. arXiv preprint arXiv:2510.11690 (2025) 2, 4, 10

75. Zhou, C., Yu, L., Babu, A., Tirumala, K., Yasunaga, M., Shamis, L., Kahn, J., Ma, X., Zettlemoyer, L., Levy, O.: Transfusion: Predict the next token and difuse images with one multi-modal model. arXiv preprint arXiv:2408.11039 (2024) 4

76. Zhu, J., Wang, W., Chen, Z., Liu, Z., Ye, S., Gu, L., Tian, H., Duan, Y., Su, W., Shao, J., et al.: Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479 (2025) 10, 11