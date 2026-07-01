# AVTok: 1D Unified Tokenization for Holistic Audio-Video Generation

Kien T. Pham<sup>1</sup> , I Chieh Chen<sup>1</sup> , Qifeng Chen<sup>1</sup> , and Long Chen<sup>1⋆</sup>

The Hong Kong University of Science and Technology {tkpham,icchen}@connect.ust.hk, {cqf,longchen}@ust.hk Project Page: https://hkust-longgroup.github.io/AVTok/

![](images/91edfdf1c95cb8c6c79f87e37f91c7dc7125cac4ba1ef40854ede0f76d004199.jpg)  
Fig. 1: Highlights. (a) We propose AVTok, a novel unified tokenizer with dual-stream transformer-based architecture, capable of jointly encoding an audio-video pair into a single compact 1D latent representation. (b) AVTok achieves competitive performance compared to state-of-the-art unimodal 1D video tokenizers (top) and audio codecs (bottom). (c) From left to right, AVTok can be integrated into AR generative models to achieve audio-to-video, video-to-audio, and joint audio-video generation.

Abstract. Audio-video generation has recently gained unprecedented research attention, aiming to synthesize high-quality sounding video content with fine-grained synchronization and semantic alignment between the auditory and visual components. The preceding methods predominantly adopt a dual-branch design with separate tokenization and generation modules per modality, neglecting the representation gap while necessitating intensive computational resources for proper training. Inspired by recent advancements in one-dimensional visual tokenization, we present AVTok, a novel unified tokenizer designated for holistic audio-video generation. AVTok features a dual-stream transformer-based architecture with shared encoder-decoder and modal-specific learnable queries to eficiently and efectively encode an audio-video pair into a compact one-dimensional latent representation with a unified codebook. To cope with the heterogeneous information imbalance that hinders AV-Tok from exploiting aligned audio-visual information, we devise a hierarchical training strategy to progressively realize reconstruction capabilities for each modality. Extensive experiments demonstrate that AVTok excels both in audio-video reconstruction and when integrated into downstream pipelines for audio-to-video, video-to-audio, and class-conditional joint audio-video generation. AVTok paves the way for the challenge of joint audio-video tokenization and provides a potential direction to build unified large multimodal models for audio-video generation.

Keywords: Unified Audio-Video Tokenization · 1D Latent Representation · Holistic Audio-Video Generation

## 1 Introduction

Audio-Visual (AV) content creation has undergone a remarkable transformation in recent years, catalyzing the emergence of innovative creative tasks that were once considered unattainable. This evolution has been largely driven by the development of powerful generative models, which are capable of Video-to-Audio (V2A) [5,20,31,55,73], Audio-to-Video (A2V) [43,48,63,65,68], and particularly Joint Audio-Video generation (JAVG) [9, 39, 41, 47, 56, 71, 74]. However, their impressive performance comes with a great price. These AV models typically adopt a heavy-weighted dual-branch architecture in which each processes one specific modality separately. In addition, extra auxiliary modules are injected and intertwined for cross-modal interaction. Such a design incurs an intensive computational cost that poses significant challenges to its scalability and accessibility for training and deployment.

Akin to single-modal predecessors [6,11,18,28,29,67,75], various audio-video generation pipelines [39, 41, 46, 71] tend to employ one pretrained tokenization model per modality to compress their respective input into the compact latent representation, partially alleviating the computational burden. However, such a simple integration neglects the intrinsic representation diference between the embedding spaces of the two modalities learned by those distinctively trained tokenizers, as depicted in Fig. 2. Inherently, their synthesized products often exhibit semantic misalignment between auditory and visual elements. To this end, an intuitive question arises: Q: Is it possible to jointly encode both audio and video components into a shared embedding space instead? We hypothesize that by constructing such a shared tokenization space, not only can it avoid the mentioned representation gap to mitigate the audio-visual semantic discrepancy, but also eliminates the need to maintain an expensive dual-branch architecture for the full generation modeling. This motivates us to design a unified tokenizer for both modalities that is capable of efectively and eficiently encoding a sounding video sample into a single latent representation, holistically capturing audio-visual information for decent reconstruction and downstream AV generation tasks.

![](images/23757b45f927b90a84a88a2c6c55440e8846fd788f9ca18cac635c81200618fe.jpg)  
Fig. 2: Motivation. Left: Previous audio-video generation models typically adopt a separate pretrained tokenizer per modality and omit the representation gap between their learned embedding spaces. Right: We aim to design a unified tokenizer that jointly encodes both modalities into a shared token space instead. Here, video and audio embeddings are colored by their respective classes.

To achieve our goal, the first critical challenge that emerges is to determine: C1: Which embedding representation is appropriate to unify and encapsulate auditory and visual information? On the one hand, as raw video inputs have three-dimensional (3D) formation, the majority of prevailing video tokenizers [37, 38, 50, 64] inherently employ 3D spatio-temporal latent as representation for compression. On the other hand, audio signals have a one-dimensional (1D) wave structure, hence many previous audio tokenizers, the so-called audio codecs [7, 22, 23, 30], typically encode an audio into the respective 1D temporal embedding. Meanwhile, some methods [32, 34] extract two-dimensional (2D) mel-spectrogram features as intermediate targets for compression and leverage neural vocoders [27, 33] to reconstruct raw signals more eficiently. Nevertheless, the diference in token organization (3D vs. 1D/2D) still makes it nontrivial to decide which representation is appropriate. Fortunately, some recent works [36, 58] have demonstrated the potential of 1D video tokenization in constructing a causal-friendly discrete latent space that facilitates autoregressive (AR) video generation, conceptually bridging with audio’s native representation. We therefore select 1D discrete latent to be the desired compact representation for audio-video encoding unification, as shown in Fig. 1(a).

Considering the objective 1D latent representation, the final and most important challenge to tackle is C2: How to design a suitable architecture to actualize 1D unified audio-video tokenization? Based on a current state-of-theart 1D video tokenizer [58], we propose AVTok, which is a novel attempt on this novel challenge. Drawing inspiration from [1, 14] that tackle AV pretraining tasks, we transform the baseline [58] into a dual-stream query-based transformer with shared encoder-decoder and modal-specific queries. This model design has several characteristics: (1) Unlike [1,14] which utilize patch-wise local-constraint information, AVTok leverages a holistic tokenization scheme with learned queries to capture higher-level, holistic AV information; (2) Dual-stream forward passes allow AVTok to harmoniously exploit auditory and visual specific elements while fusing their information implicitly to enhance reconstruction, maintaining both eficacy and eficiency; (3) AVTok inherits the AR-friendliness of the baseline [58] that is beneficial for downstream AR-based AV generation tasks.

Despite possessing the above-mentioned architectural advantages, training AVTok properly is challenging due to several reasons. First, visual data exhibit significantly diferent information density from their corresponding auditory companions, causing the model to suppress the learning and deteriorate the performance of one or both modalities. Secondly, the implicitness of information fusing via shared model parameters may lead to insuficient cross-modal interaction that hinders alignment learning. Therefore, we introduce a hierarchical training strategy: Video-First-Audio-Later (VFAL), to realize respective reconstruction capability for each individual modality in a progressive manner. Additionally, inspired by [17,70], we leverage the features extracted from audio-visual foundational models [1,14] with rich semantic correspondence to enhance model learning via a representation alignment objective. The experimental results highlighted in Fig. 1(b,c) show that AVTok achieves outstanding performance not only in AV reconstruction but also in downstream generation tasks, including audio-to-video, video-to-audio, and class-conditional joint AV generation.

Overall, our contributions are summarized as follows:

– We propose a novel task of unified audio-video (AV) tokenization, which aims at jointly encoding both auditory and visual components into a single latent representation, facilitating eficient and efective AV reconstruction and downstream generation

– We present AVTok, a 1D unified AV tokenizer attempting to fulfill the task by leveraging a multi-stream transformer-based architecture with shared encoder-decoder and modal-specific queries.

– We introduce VFAL, a hierarchical training paradigm equipped with a representation alignment learning objective to progressively incorporate video then audio encoding and reconstruction capabilities into AVTok.

Extensive experiments highlight that AVTok excels in not only unified AV reconstruction but also downstream tasks, including audio-to-video (A2V), video-to-audio (V2A), and class-conditional joint AV generation (cJAVG).

## 2 Related Work

## 2.1 1D Visual Tokenization

With the Multimodal Large Language Model (MLLM) for understanding and generation tasks gaining growing popularity in recent years, 1D visual tokenization has emerged as an indispensable component. Not only does it bridge the vision-language representation gap, but it also reduces the computational burden incurred when processing visual data, enabling efortless and eficient integration of visual input into well-established LLMs. Early studies mainly focused on the image domain starting with TiTok [69], a transformer-based tokenizer with learnable queries that can encode a $2 5 6 \times 2 5 6 \times 3$ image using as few as 32 discrete tokens. TA-TiTok [25] then uses rich semantic information from textual input to complement visual features and improve the decoding stage. Subsequent works [2, 57, 62] enforce causality relationships among resulting tokens, making their models autoregressive (AR)-friendly for better adaptation into MLLMs.

Recent advances have started to be explored in the video domain. LARP [58] is the pioneer that employs a query-based transformer architecture with a holistic tokenization scheme and an autoregressive prior model to tokenize videos into a 1D latent representation with optimal token order for downstream AR generation tasks. It is then followed by Adaptok [36], which attempts to induce an adaptive temporal causality within latent space and dynamically manipulate token allocation for flexible tokenization, and DeRA [17], which decouples spatial-temporal representation learning to achieve more eficient and efective training. Inspired by these works and their insights, our work aims to extend the concept of 1D unified tokenization for audio and video together.

## 2.2 Audio Tokenization

Unlike image and video domains that inherently involve 2D and 3D spatial structures, audio is naturally a 1D time-varying signal representing the sound wave’s amplitude over time. Audio tokenization, a.k.a neural audio coding, has been a long-standing challenge, aiming to balance high-fidelity reconstruction with lowbitrate discrete representation that facilitates incorporation into LLMs. Some early codecs include EnCodec [7] and DAC [30] that utilize residual vector quantization (RVQ) within a fully convolutional encoder-decoder architecture. Recently, UniCodec [23] focuses on reducing the redundancy inherent in multicodebook RVQ systems by constructing a unified codebook for universal sound domains. Meanwhile, SpecVQGAN [20], Spectral Codec [32], and MelTok [34] also improve eficiency but alternatively by compressing mel-spectrograms instead of raw waveforms. With the aligned 1D discrete representation, we aim to replicate their audio tokenization capability in our unified model.

## 2.3 Audio-Video Generation

Generative tasks involving audio and video modalities, such as audio-to-video (A2V), video-to-audio (V2A), and joint audio-video generation (JAVG) have attracted a lot of research attention in recent years, leading to a proliferation of many models with impressive synthesizing abilities. Some of the representative works for A2V generation include TempoTokens [68] that adapts a pretrained text-to-video difusion model to support audio conditioning and achieve better synchronization, Seeing-and-Hearing [65] introduces a difusion latent aligner to enhance cross-modal semantic coherence, and SpA2V [43] harnesses spatial auditory cues to realize spatial alignment in synthesized videos.

Regarding the V2A generation, SpecVQGAN [20] is one of the early studies to train a transformer to sample spectrograms conditioning on video features from a pretrained codebook obtained by a VQGAN-variant tokenizer. Later, V-AURA [55] introduces an autoregressive model with an audio-visual feature fusion strategy to enhance temporal alignment. Recently, FoleyCrafter [73], Vin-TAGe [31], and MMAudio [5] leverage difusion and flow matching generative models to achieve better audio synthesis fidelity and diversity.

By unifying the A2V and V2A goals, the JAVG task enables the joint synthesis of high-fidelity video and audio, prioritizing individual modal quality with seamless cross-modal synchronization and semantic alignment. The latest approaches [39,41,60,71] primarily adopt a dual-branch architecture with separate variational autoencoder (VAE) and difusion transformer (DiT) as tokenization and generation modules, respectively, per modality. Despite showing impressive results, such a design is heavy-weighted and necessitates intensive computing resources for adequate training. Besides, using distinct tokenizers also neglects the representation gap between auditory and visual elements, hence they are prone to producing results with semantic misalignment. To address this problem, in this work, we introduce a unified tokenizer to jointly encode both audio and video into a single latent representation.

## 3 Method

## 3.1 Preliminary

Query-based 1D Video Tokenization. As discussed in Sec. 1, the prevailing video tokenizers predominantly adopt a 3D patch-wise tokenization scheme of which latent tokens are encoded from the 3D spatio-temporal patches of the input video, limiting them to low-level patch features and hindering the exploitation for higher-level information. To break this local constraint and enable 1D tokenization, LARP [58] and following works [17,36] adapt the philosophy of [3,35] to leverage a set of fixed learnable queries to capture holistic information in the video. Given a video input $\mathbf { V } \in \mathbb { R } ^ { T \times H \times W \times 3 }$ , it is first processed as:

$$
\mathbf {P} ^ {v} = \mathcal {P} (\mathbf {V}), \quad \mathbf {E} ^ {v} = \mathcal {F} (\mathbf {P} ^ {v}),
$$

where P and $\mathcal { F }$ are linear patchify and flatten operations, $\mathbf { P } ^ { v } \in \mathbb { R } ^ { \frac { T } { f _ { T } } \times \frac { H } { f _ { H } } \times \frac { W } { f _ { W } } }$ ×d and $\mathbf { E } ^ { v } \in \mathbb { R } ^ { m \times d }$ represent the spatiotemporal patches projected onto d dimensions and their flattened embeddings. Here, $f _ { T } , f _ { H } , f _ { W }$ correspond to the downsampling factors for dimensions $T , H , W$ respectively, and $\begin{array} { r } { m \doteq \frac { T } { f _ { T } } \times \frac { H } { f _ { H } } \times \frac { W } { f _ { W } } } \end{array}$ is the total number of tokens. Subsequently, a set of n learnable holistic query embedding $\mathbf { Q } _ { L } ^ { v } \in \mathbb { R } ^ { n \times d }$ is introduced to encode and quantize the patch embeddings $\mathbf { E } ^ { v }$ as follows:

$$
\mathbf {Z} ^ {v} = \mathcal {E} (\mathbf {Q} _ {L} ^ {v} \| \mathbf {E} ^ {v}), \quad \mathbf {x} ^ {v} = \mathcal {Q} (\mathbf {Z} _ {1: n} ^ {v}),
$$

in which E and Q are the encoder and quantizer, ∥ denotes the concatenation operation, and $\mathbf { Z } ^ { v }$ is the latent embeddings of length $( n { + } m )$ . Note that only $\mathbf { Z } _ { 1 : n } ^ { v } ,$ $i . e . ,$ , the first n ones corresponding to the query embeddings $\mathbf { Q } _ { L } ^ { v }$ are quantized into $\mathbf { x } ^ { v } = ( x _ { 1 } ^ { v } , \ldots , x _ { n } ^ { v } )$ discrete tokens, ensuring each $x _ { v } ^ { i }$ can represent any video patch equally. Eventually, during the decoding stage, another m learnable patch query embeddings $\mathbf { Q } _ { P } ^ { v } \in \mathbb { R } ^ { m \times d }$ are utilized to reconstruct the video as:

$$
\hat {\mathbf {Z}} ^ {v} = \mathcal {Q} ^ {- 1} (\mathbf {x} ^ {v}), \quad \hat {\mathbf {E}} ^ {v} = \mathcal {D} (\mathbf {Q} _ {P} ^ {v} \| \hat {\mathbf {Z}} ^ {v}), \quad \hat {\mathbf {V}} = \mathcal {R} (\hat {\mathbf {E}} _ {1: m} ^ {v}),
$$

where ${ \mathcal { Q } } ^ { - 1 }$ denotes de-quantization operation that maps discrete tokens $\mathbf { x } ^ { v }$ back to the continuous latent embedding $\hat { \mathbf { Z } } ^ { v } \in \mathbb { R } ^ { n \times d }$ . Subsequently, they are concatenated with $\mathbf { Q } _ { P } ^ { v }$ and go through the decoder D to decode $\hat { \mathbf { E } } ^ { v }$ , of which only the first m vectors are reshaped via the R operator to reconstruct $\hat { \mathbf { V } } \in \mathbb { R } ^ { T \times H \times \mathrm { \bar { \boldsymbol { W } } \times 3 } }$

Autoregressive Generative Prior. Although the 1D latent tokens $\mathbf { x } ^ { v }$ obtained with the aforementioned query-based tokenizer are now holistic and discrete, there is no specific flattening order enforced. This is because of the unordered nature of the holistic query set and the parallel processing property of the transformer encoder. To make such a latent space compatible with AR generative models, LARP [58] incorporates a lightweight AR transformer with adjusted input and output layers as prior model $\mathcal { M } _ { P }$ to provide gradients for structure optimization. It is jointly trained with the tokenizer in an end-to-end manner using negative log-likelihood (NLL) loss $\mathcal { L } _ { p r i o r }$ for next token prediction objective (NTP) in synergy with reconstruction loss $\mathcal { L } _ { r e c } ^ { v }$ as:

$$
\mathcal {L} = \mathcal {L} _ {r e c} ^ {v} + \alpha \mathcal {L} _ {p r i o r},
$$

where α is the loss weight. Notably, this prior model serves the sole purpose of promoting an AR-friendly discrete latent space during training. It is discarded during inference and thus afects neither the speed nor the memory footprint.

## 3.2 Holistic Audio-Video Tokenization

Patchify. AVTok employs the architecture described in Sec. 3.1 as its video stream of which the patchification remains unchanged that transforms a video input $\mathbf { V } \in \mathbb { R } ^ { T \times H \times W \times 3 }$ into a flattened d-dimensional embedding $E ^ { v } \in \mathbb { R } ^ { m \times d }$ For audio stream, instead of $\mathbf { A } _ { r a w } \in \mathbb { R } ^ { N }$ which is 1D-long continuous data, we opt to use its normalized mel-spectrogram $\mathbf { A } _ { m e l } \in \mathbb { R } ^ { M \times L }$ as input. Here, M and L denote the number of frequency bins and time frames, respectively. Not only does $\mathbf { A } _ { m e l }$ reduce computation complexity, but it can also be interpreted as a gray-scale image that can be patchified similarly as in video stream. Notably, it can be converted back to the raw waveform with lossless quality using ofthe-shelf vocoders [27, 33]. Given $\mathbf { A } _ { m e l }$ , herein referred to as A for brevity, we process it as:

$$
\mathbf {P} ^ {a} = \mathcal {P} (\mathbf {A}), \quad \mathbf {E} ^ {a} = \mathcal {F} (\mathbf {P} ^ {a}),
$$

![](images/020af434a3121cd7f9306176c287b900ba20b4c09c88c15e3dd43f9f20e9f895.jpg)  
Fig. 3: Method illustration. Cubes , squares <sup>□</sup>, and circles respectively represent input patches or patch-wise tokens, holistic discrete tokens, and continuous query embeddings. (a) AVTok features a dual-stream transformer-based architecture, of which each stream’s forward pass is demonstrated in (b), to jointly learns video (Blue stream) and audio (Green stream) reconstructions in a unified holistic scheme. It leverages separate sets of learnable queries and normalization layers to gather modal-specific information, while sharing remaining parameters to enable implicit cross-modal interaction, achieving both eficiency and eficacy. In addition to the standard reconstruction training objectives $\mathcal { L } _ { r e ( } ^ { v }$ and $\mathcal { L } _ { r e c } ^ { a } ,$ we align AVTok’s patch-wise continuous tokens with an audio-visual foundation model $\mathcal { M } _ { F }$ via $\mathcal { L } _ { \boldsymbol { r } \boldsymbol { e p } }$ to better capture synergistic features between auditory and visual elements. Lastly, an AR prior model $\mathcal { M } _ { P }$ is also equipped to encourage an AR-friendly discrete latent space via $\mathcal { L } _ { p r i o r : }$ , facilitating downstream AR generative tasks including (c) audio-to-video, (d) video-to-audio, and (e) class-conditional joint audio-video generation.

where $\mathbf { P } ^ { a } \in \mathbb { R } ^ { \frac { M } { f _ { M } } \times \frac { L } { f _ { L } } \times d }$ and $\mathbf { E } ^ { a } \in \mathbb { R } ^ { p \times d }$ represent the audio patches projected onto d dimensions and their flattened embeddings. Here, $f _ { M } , f _ { L }$ correspond to downsampling factors for dimension M, L accordingly, and $\begin{array} { r } { p = \frac { M } { f _ { M } } \times \frac { \bar { L } } { f _ { L } } } \end{array}$ is the total number of audio tokens.

Dual-stream Transformer. Although transformer-based design has been applied to the context of 1D tokenization for both audio and video modality individually referring to Sec. 2, it has never been explored for the unified setting involving both modalities simultaneously. As the initial attempt for this work, we extend the query-based design in [58] from the video-only modality to audiovideo multi-modality and build a single-stream vanilla version of our AVTok tokenizer. Given the patchified $\mathbf { E } ^ { a }$ and $\mathbf { E } ^ { v }$ embeddings obtained above, we concatenate them and construct a joint embedding ${ \bf E } ^ { a v } = ( { \bf E } ^ { v } \Vert { \bf E } ^ { a } ) \in \mathbb { R } ^ { ( m + p ) \times d }$ . It will then be encoded, quantized, and decoded similarly following Sec. 3.1 as:

$$
\begin{array}{r l} \mathbf {x} ^ {a v} = \mathcal {Q} (\mathcal {E} (\mathbf {Q} _ {L} ^ {a v} \| \mathbf {E} ^ {a v}) _ {1: n}), & \hat {\mathbf {E}} ^ {a v} = \mathcal {D} (\mathbf {Q} _ {P} ^ {a v} \| \mathcal {Q} ^ {- 1} (\mathbf {x} ^ {a v})), \\ \hat {\mathbf {V}} = \mathcal {R} (\hat {\mathbf {E}} _ {1: m} ^ {a v}), & \hat {\mathbf {A}} = \mathcal {R} (\hat {\mathbf {E}} _ {m: m + p} ^ {a v}), \end{array}
$$

where $\mathbf { Q } _ { L } ^ { a v } \in \mathbb { R } ^ { n \times d }$ and $\mathbf { Q } _ { P } ^ { a v } \in \mathbb { R } ^ { ( m + p ) \times d }$ denotes learnable holistic and patch query embeddings respectively. This simple design features cross-modal modeling that may help the model to exploit audio-visual correlation to reconstruct one modality based on the information of the other. However, without explicitly considering the modal-specific features, their significant diference in nature often causes the vanilla model to train inadequately in which the learning of one modality harms that of the other, eventually yielding subpar performance.

To alleviate this problem, we adapt the philosophy of [1, 14] to bootstrap the vanilla design into a dual-stream architecture with shared encoder-decoder but separate sets of learnable holistic and patch query embeddings as well as normalization layers for our finalized AVTok tokenizer. Specifically, we input audio and video patch embeddings $\mathbf { E } ^ { a }$ and $\mathbf { E } ^ { v }$ in two diferent forward passes to the encoder $\mathcal { E } ( \cdot ; L N _ { 1 } , L N _ { 2 } )$ then decoder $\mathcal { D } ( \cdot ; L N _ { 1 } , L N _ { 2 } )$ with each stream leveraging a separate set of normalization layers $( L N _ { 1 } ^ { \{ a , v \} } , L N _ { 2 } ^ { \{ a , v \} } )$ as follows:

$$
\mathbf {Z} ^ {i} = \mathcal {E} (\mathbf {Q} _ {L} ^ {i} \| \mathbf {E} ^ {i}; L N _ {1} ^ {i}, L N _ {2} ^ {i}), \mathbf {x} ^ {i} = \mathcal {Q} (\mathbf {Z} _ {1: j} ^ {i}), \hat {\mathbf {E}} ^ {i} = \mathcal {D} (\mathbf {Q} _ {P} ^ {i} \| \mathcal {Q} ^ {- 1} (\mathbf {x} ^ {i}); L N _ {1} ^ {i}, L N _ {2} ^ {i}),
$$

$$
\hat {\mathbf {V}} = \mathcal {R} (\hat {\mathbf {E}} _ {1: m} ^ {v}), \quad \hat {\mathbf {A}} = \mathcal {R} (\hat {\mathbf {E}} _ {1: p} ^ {a}), \quad (i, j) \in \{(v, n), (a, q) \},
$$

where $\mathbf { Q } _ { L } ^ { v } \in \mathbb { R } ^ { n \times d } , \mathbf { Q } _ { P } ^ { v } \in \mathbb { R } ^ { m \times d } , \mathbf { Q } _ { L } ^ { a } \in \mathbb { R } ^ { q \times d } , \mathbf { Q } _ { P } ^ { a } \in \mathbb { R } ^ { p \times d }$ respectively represent the learnable holistic and patch query embeddings of video and audio modality. This design facilitates harnessing modal-specific information by using distinctive learnable components per modality, while still allowing for implicit audio-visual fusion via sharing remaining parameters, thereby achieving both eficiency and efectiveness for reconstruction and downstream generation tasks. The detailed illustration of AVTok is shown in Fig. $3 ( \mathrm { a } , \mathrm { b } )$

Reconstruction Objective. Following the composition in [58], the reconstructive training loss for the video stream of AVTok, i.e. $\mathcal { L } _ { r e c } ^ { v } ,$ is constituted by $L _ { 1 }$ reconstruction loss, LPIPS perceptual loss [72], GAN adversarial loss [15], and SVQ quantization loss [58]. Meanwhile, for $\mathcal { L } _ { r e c } ^ { a } ,$ since the audio stream reconstruction process involves pretrained vocoders [27, 33], we follow them to adopt Multi-Scale Mel-Spectrogram Loss [30] as reconstruction loss, use Multi-Scale Sub-Band CQT Discriminator [16] and Multi-Period Discriminator [27] for adversarial components, and reuse $\mathrm { S V Q }$ quantization loss from the video stream.

## 3.3 Hierarchical Training Paradigm

Video-First-Audio-Later (VFAL) Strategy. Despite having several architectural advantages, our experiments reveal that simply training AVTok from scratch is non-ideal. This is primarily because of the fact that visual information is abundant, which dominates auditory information, causing the learning of the video stream to suppress that of the audio stream. To accommodate this issue, we design the VFAL hierarchical training strategy for optimal and eficient training of AVTok. Specifically, we start with the training of the more challenging modality, i.e., video stream, while discarding the audio stream in Stage 1, aiming to realize reconstruction ability for visual elements and establish a strong latent token representation space. Subsequently, in Stage 2, we reattach and train only the modules specialized for the audio stream while freezing those of the video stream together with the shared ones, realizing audio reconstruction capability. This is intuitively possible considering that the input mel-spectrogram can be treated as a gray-scale image, as mentioned in Sec. 3.2. Finally, in the last stage, we finetune the decoding modules to attain unified audio-video reconstruction with refined quality. By imposing this explicit training path, VFAL encourages AVTok to optimize the learning of each stream progressively.

Representation Alignment Learning. During experiments, we also observed an issue in which AVTok does not fully exploit audio-visual correspondent features to improve the final reconstruction. We hypothesize that this might be because the cross-modal interaction via shared model parameters is implicit, and hence it hinders audio-visual alignment learning. Drawing inspiration from [17, 70], we leverage a pretrained audio-visual foundation model $\mathcal { M } _ { F } \left[ 1 \right]$ that learned an embedding space with rich semantics and strong correspondence between visual and auditory information as the intermediate aligning module to enhance cross-modal alignment between the two streams of AVTok. This can be achieved by incorporating into the training the representation alignment objective $\mathcal { L } _ { r e p } ,$ which can be computed as follows:

$$
\begin{array}{r l} & {\mathbf {Z} _ {F} ^ {v} = \mathcal {M} _ {F} (\mathbf {V}), \quad \mathbf {Z} _ {F} ^ {a} = \mathcal {M} _ {F} (\mathbf {A}), \quad \tilde {\mathbf {Z}} ^ {v} = \mathbf {Z} _ {n: m + n} ^ {v}, \quad \tilde {\mathbf {Z}} ^ {a} = \mathbf {Z} _ {q: p + q} ^ {a},} \\ & {\qquad \mathcal {L} _ {r e p} = - \mathbb {E} \Big [ \sum_ {i \in \{a, v \}} \frac {1}{N _ {i}} \sum_ {k = 1} ^ {N _ {i}} \mathrm{sim} (\mathbf {Z} _ {F} ^ {i} [ k ], h _ {\phi} (\mathrm{interp} (\tilde {\mathbf {Z}} ^ {i}) [ k ])) \Big ],} \end{array}
$$

where ${ \bf Z } _ { F } ^ { v } , { \bf Z } _ { F } ^ { a } , \tilde { { \bf Z } } ^ { v } , \tilde { { \bf Z } } ^ { a }$ denote the video and audio patch embeddings of length $N _ { v } , N _ { a } , m , p$ extracted by $\mathcal { M } _ { F }$ and our AVTok’s encoder, k is a patch index, sim(·, ·) is a pre-defined similarity function, and $h _ { \phi }$ represents a multilayer perceptron (MLP). Similarly to [17, 70], we linearly interpolate $\tilde { \mathbf { Z } } ^ { v } , \tilde { \mathbf { Z } } ^ { a }$ to the same length of ${ \bf Z } _ { F } ^ { v } , { \bf Z } _ { F } ^ { a }$ via interp(·) operator for computational compatibility.

Cross-modal AR Generative Prior. To facilitate the downstream audio-tovideo, video-to-audio generation, and class-conditional joint audio-video generation tasks simultaneously, we adapt the autoregressive generative prior of [58] mentioned in Sec. 3.1 by simply computing the NTP objective loss for two token orders $\mathbf { x } ^ { v } \Vert \mathbf { x } ^ { a }$ and $\mathbf { x } ^ { a } \Vert \mathbf { x } ^ { v }$ to compose $\mathcal { L } _ { p r i o r }$ . Finally, the overall training objective

Table 1: Quantitative comparison of reconstruction results. Results are categorized into video-only (VO), audio-only (AO), and joint audio-video (AV) tokenization functionality. W/M denote resolution of waveform/mel-spectrogram used as audio input. The best and second best results are in bold and underlined.

<table><tr><td rowspan="2">Type</td><td rowspan="2">Method</td><td colspan="2">Configuration</td><td colspan="3">Video Metrics</td><td colspan="3">Audio Metrics</td></tr><tr><td>Resolution</td><td>#Tokens</td><td>PSNR↑</td><td>rFVD↓</td><td>LPIPS↓</td><td>SI-SDR↓</td><td>rFAD↓</td><td>MR-STFT↓</td></tr><tr><td rowspan="3">VO</td><td>OmniTokenizer [59]</td><td>17 × 128 × 128</td><td>1280</td><td>23.84</td><td>90.99</td><td>0.203</td><td>-</td><td>-</td><td>-</td></tr><tr><td>AdapTok [36]</td><td>16 × 128 × 128</td><td>2048</td><td>23.87</td><td>22.23</td><td>0.180</td><td>-</td><td>-</td><td>-</td></tr><tr><td>LARP [58]</td><td>16 × 128 × 128</td><td>1024</td><td>24.53</td><td>14.24</td><td>0.137</td><td>-</td><td>-</td><td>-</td></tr><tr><td rowspan="3">AO</td><td>WavTokenizer [22]</td><td>98304 × 1 (W)</td><td>164</td><td>-</td><td>-</td><td>-</td><td>24.27</td><td>6.82</td><td>1.589</td></tr><tr><td>UniCodec [23]</td><td>98304 × 1 (W)</td><td>308</td><td>-</td><td>-</td><td>-</td><td>18.25</td><td>6.73</td><td>1.508</td></tr><tr><td>SpectralCodec [32]</td><td>80 × 384 (M)</td><td>384</td><td>-</td><td>-</td><td>-</td><td>29.30</td><td>5.56</td><td>1.514</td></tr><tr><td rowspan="2">AV(Ours)</td><td>Vanilla</td><td>16 × 128 × 128</td><td rowspan="2">1152</td><td>24.50</td><td>14.87</td><td>0.140</td><td>35.45</td><td>10.26</td><td>2.114</td></tr><tr><td>AVTok</td><td>80 × 384 (M)</td><td>25.62</td><td>12.80</td><td>0.126</td><td>23.09</td><td>5.93</td><td>1.523</td></tr></table>

![](images/3ef32e4a71aaed1a6056c1ac5810ce33b0e18d4373a6631217d5494d0d3251cf.jpg)  
Fig. 4: Qualitative comparison of reconstruction results.

for AVTok is formulated as:

$$
\mathcal {L} = \lambda_ {1} \mathcal {L} _ {r e c} ^ {v} + \lambda_ {2} \mathcal {L} _ {r e c} ^ {a} + \lambda_ {3} \mathcal {L} _ {r e p} + \lambda_ {4} \mathcal {L} _ {p r i o r},
$$

with $\lambda _ { 1 , 2 , 3 , 4 }$ are the loss weights for each component.

## 4 Experiments

## 4.1 Setup

Dataset. We conduct our experiments on TAVGBench [42] and VGGSound [4] datasets. Both are used in the reconstruction whilst only the latter is used in the downstream generation tasks deliberately for demonstration purposes due to time and resource constraints. The test set of VGGSound is used for all assessments. By default, we use 16-frame sounding video clips with spatial resolution 128 × 128, frame rate of 3.6fps, single-channel audio with waveform resolution 98304×1, 22kHz sampling rate, and mel-spectrogram resolution 80×384 in both training and evaluation.

Table 2: Comparison of generation results. Results are grouped by tasks including audio-to-video (A2V), video-to-audio (V2A), and class-conditional joint audio-video generation (cJAVG). Dif, AR, and FM respectively denote difusion, autoregressive, and flow matching generative paradigms. The best and second best results are in bold and underlined.

<table><tr><td>Task</td><td>Method</td><td>Gen Type</td><td colspan="2">#Param Tokenizer Generator</td><td>gFVD↓</td><td>gFAD↓</td><td>DeSync↓</td><td>IB-Score↑</td></tr><tr><td rowspan="2">A2V</td><td>TempoTokens [68]</td><td>Diff</td><td>83.7M</td><td>1.9B</td><td>786.61</td><td>-</td><td>1.359</td><td>0.132</td></tr><tr><td>AVTok-A2V (Ours)</td><td>AR</td><td>208.4M</td><td>632.0M</td><td>150.26</td><td>-</td><td>1.317</td><td>0.143</td></tr><tr><td rowspan="5">V2A</td><td>MMAudio [5]</td><td>FM</td><td>298.5M</td><td>1.3B</td><td>-</td><td>17.09</td><td>0.813</td><td>0.291</td></tr><tr><td>VinTAGe [31]</td><td>FM</td><td>110.6M</td><td>1.5B</td><td>-</td><td>80.06</td><td>1.294</td><td>0.044</td></tr><tr><td>V-AURA [55]</td><td>AR</td><td>76.7M</td><td>816.9M</td><td>-</td><td>126.92</td><td>0.967</td><td>0.231</td></tr><tr><td>SpecVQGAN [20]</td><td>AR</td><td>76.4M</td><td>332.4M</td><td>-</td><td>210.07</td><td>1.291</td><td>0.100</td></tr><tr><td>AVTok-V2A (Ours)</td><td>AR</td><td>208.4M</td><td>632.0M</td><td>-</td><td>49.47</td><td>1.239</td><td>0.249</td></tr><tr><td rowspan="3">cJAVG</td><td>JavisDiT [39]</td><td>FM</td><td>448.7M</td><td>8.9B</td><td>1040.28</td><td>268.51</td><td>1.330</td><td>0.195</td></tr><tr><td>Ovi [41]</td><td>FM</td><td>988.6M</td><td>17.3B</td><td>972.65</td><td>129.02</td><td>0.814</td><td>0.172</td></tr><tr><td>AVTok-cJAVG (Ours)</td><td>AR</td><td>208.4M</td><td>632.4M</td><td>138.80</td><td>56.58</td><td>1.319</td><td>0.206</td></tr></table>

Implementation Details. For patchification, we first follow [13, 58] to split the input video and audio mel-spectrogram into continuous visual and auditory patch embeddings using $( f _ { T } , f _ { H } , f _ { W } ) = ( 4 , 8 , 8 )$ and $( f _ { M } , f _ { L } ) = ( 1 6 , 1 6 )$ respectively. We then utilize a set of $n = 1 0 2 4 , q = 1 2 8$ learnable holistic queries to obtain 1152 holistic discrete tokens. For decoding, another set of $m = 1 0 2 4 , q = 1 2 0$ learnable patch queries are leveraged to reconstruct their corresponding modality. Besides, we use HiFi-GAN [27] to convert the output mel-spectrograms back to waveforms. For the remaining, unless otherwise specified, we maintain the same configuration as [58] by default. Regarding downstream generation tasks, we also follow [58] to adopt Llama-like transformer [49, 51] to be our AR generative model. As shown in Fig. 3(c-e), one class token [CLS] is used in the class-conditional joint audio-video generation task, while one separator token [SEP] is employed for cross-modal generation tasks.

## 4.2 Reconstruction Evaluation

Baselines & Metrics. Due to the novelty of the unified audio-video tokenization task, there is no open-source baseline available for direct comparison. Therefore, in addition to our vanilla model, we select some state-of-the-art unimodal methods from each side as representatives for comparisons including OmniTokenizer [59], AdapTok [36], LARP [58] as video-only baselines, and WavTokenizer [22], UniCodec [23], SpectralCodec [32] as audio-only baselines. Regarding metrics, we adopt PSNR [61], FVD [53], LPIPS [72] to assess video reconstruction, and employ SI-SDR [45], FAD [24], MR-STFT [66] to evaluate audio side.

![](images/1f2864b7b556694afb9a1f89516349ba4526a22a81bd2b2853ea4efb7af6cfc0.jpg)  
Fig. 5: Qualitative results for downstream generation tasks. Note that class conditions are only inputted for (a) class-conditional joint audio-video generation, and displayed here for illustration purposes of (b) audio-to-video and (c) video-to-audio generation.

Main Results. As shown in Tab. 1 and Fig. 4, AVTok consistently outperforms the vanilla design and selected unimodal baselines in video reconstruction while maintaining competitive performance on the audio side. They not only indicate the feasibility of the unified audio-video tokenization task and the efectiveness of our approach but also infer that leveraging cross-modal information can boost the performance of each modality.

## 4.3 Generation Evaluation

Baselines & Metrics. For downstream generation tasks, we select and compare our AR generative models with several representative baselines, including: (1) TempoTokens [68] for audio-to-video generation (A2V); (2) MMAudio [5], VinTAGe [31], V-AURA [55], SpecVQGAN [20] for video-to-audio generation (V2A); and (3) JavisDiT [39], Ovi [41] for class-conditional joint audio-video generation (cJAVG). Since some of them require textual caption as the condition to control the synthesis, we bypass it by utilizing class labels as an alternative. Regarding metrics, we use FVD [53] and FAD [24] to assess the quality of the video and audio samples generated, DeSync [21] to measure their temporal synchronization, and ImageBind [12] (IB) Score to evaluate semantic alignment.

Main Results. As demonstrated in Tab. 2, our AR generative models incorporated with the proposed AVTok tokenizer achieve outstanding results in downstream tasks that surpass the majority of selected baselines whilst having eficient designs. This can be attributed to the learned unified discrete latent space of AVTok illustrated in Fig. 2 which is suitable for AR audio-video generation, enabling the synthesis of high-fidelity samples as displayed in Fig. 5.

Table 3: Ablation study on the impact of each component.

<table><tr><td rowspan="2">Configuration</td><td colspan="2">AV Reconstruction</td><td>A2V</td><td>V2A</td><td colspan="2">cJAVG</td></tr><tr><td>rFVD↓</td><td>rFAD↓</td><td>gFVD↓</td><td>gFAD↓</td><td>gFVD↓</td><td>gFAD↓</td></tr><tr><td>Vanilla</td><td>14.87</td><td>10.26</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>AVTok</td><td>12.80</td><td>5.93</td><td>150.26</td><td>49.47</td><td>138.80</td><td>56.58</td></tr><tr><td>Without VFAL</td><td>13.19</td><td>9.38</td><td>209.33</td><td>61.02</td><td>193.28</td><td>80.78</td></tr><tr><td>Without  $\mathcal{L}_{rep}$ </td><td>12.90</td><td>8.48</td><td>182.15</td><td>54.16</td><td>184.20</td><td>75.09</td></tr><tr><td>Without  $\mathcal{L}_{prior}$ </td><td>10.63</td><td>3.47</td><td>266.82</td><td>67.84</td><td>249.47</td><td>90.11</td></tr></table>

## 4.4 Ablation Study

To evaluate the impact of architecture design and each training component proposed in Sec. 3, we conduct an ablation study, of which the results are shown in Tab. 3.

Architecture Design. We first observe that AVTok’s dual-stream architecture attains superior performance compared to the single-stream vanilla version across all tasks. This can be attributed to its capability to harness modal-specific information via distinctive learnable queries per modality while allowing for implicit cross-modal interaction via remaining shared parameters. Notably, such a design of AVTok facilitates efortless integration into AR generative models for diferent downstream generation tasks whereas the vanilla one does not.

Training Components. It is demonstrated that the hierarchical VFAL strategy and the representation alignment training objective $\mathcal { L } _ { \boldsymbol { r } \boldsymbol { e p } }$ contribute significantly to the final performance in reconstruction, which eventually benefits downstream generation. This highlights their efectiveness in encouraging AVTok to optimize the learning of each stream progressively and enhance their semantic alignment. Finally, similar to [58], we find that ablating cross-modal AR generative prior $\mathcal { L } _ { p r i o r }$ yields the best reconstruction but the worst synthesis results, further validating the advantage of leveraging AR prior to train an AR-friendly tokenizer for downstream generation tasks.

## 5 Conclusion

We have presented AVTok, a novel unified audio-video tokenizer capable of jointly encoding an audio-video pair into a single compact one-dimensional latent representation with a unified codebook. AVTok features a dual-stream transformer-based architecture with shared encoder-decoder and modal-specific learnable holistic queries to harmoniously exploit auditory and visual specific elements while fusing their information implicitly for eficient and efective reconstruction. To train AVTok properly, we devise Video-First-Audio-Later (VFAL), a hierarchical strategy that encourages the model to progressively develop reconstruction capability for each individual modality. Additionally, we incorporate an audio-visual foundation model to enhance cross-modal correspondence learning of AVTok via representation alignment loss, eventually improving the learning of each stream. The experimental results demonstrate not only the feasibility of the proposed unified tokenization goal but also the superiority of our model in both reconstruction and downstream generation tasks. We hope that this work will encourage further exploration in this direction to build unified large multimodal models for audio-video generation in future.

Acknowledgement. This work was supported by National Natural Science Foundation of China (NSFC) Young Scientists Fund Category B (62522216), National Natural Science Foundation of China (NSFC) Young Scientists Fund Category C (62402408), Hong Kong SAR Research Grants Council (RGC) Early Career Scheme (26208924), Hong Kong SAR Research Grants Council (RGC) General Research Fund (16219025), and HKUST (WEB25EG01).

## References

1. Araujo, E., Rouditchenko, A., Gong, Y., Bhati, S., Thomas, S., Kingsbury, B., Karlinsky, L., Feris, R., Glass, J.R., Kuehne, H.: Cav-mae sync: Improving contrastive audio-visual mask autoencoders via fine-grained alignment. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 18794–18803 (June 2025)

2. Bachmann, R., Allardice, J., Mizrahi, D., Fini, E., Kar, O.F., Amirloo, E., El-Nouby, A., Zamir, A., Dehghan, A.: Flextok: Resampling images into 1d token sequences of flexible length. In: Forty-second International Conference on Machine Learning (2025)

3. Carion, N., Massa, F., Synnaeve, G., Usunier, N., Kirillov, A., Zagoruyko, S.: Endto-end object detection with transformers. In: Computer Vision – ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I. p. 213–229. Springer-Verlag, Berlin, Heidelberg (2020). https://doi.org/10. 1007/978-3-030-58452-8\_13

4. Chen, H., Xie, W., Vedaldi, A., Zisserman, A.: Vggsound: A large-scale audio-visual dataset. In: International Conference on Acoustics, Speech, and Signal Processing (ICASSP) (2020)

5. Cheng, H.K., Ishii, M., Hayakawa, A., Shibuya, T., Schwing, A., Mitsufuji, Y.: Mmaudio: Taming multimodal joint training for high-quality video-to-audio synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 28901–28911 (June 2025)

6. Copet, J., Kreuk, F., Gat, I., Remez, T., Kant, D., Synnaeve, G., Adi, Y., Défossez, A.: Simple and controllable music generation. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)

7. Défossez, A., Copet, J., Synnaeve, G., Adi, Y.: High fidelity neural audio compression. Transactions on Machine Learning Research (2023), featured Certification, Reproducibility Certification

8. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., Houlsby, N.: An image is worth 16x16 words: Transformers for image recognition at scale. In: International Conference on Learning Representations (2021)

9. Ergasti, A., Tarollo, G.G., Botti, F., Fontanini, T., Ferrari, C., Bertozzi, M., Prati, A.: <sup>r</sup>flav: Rolling flow matching for infinite audio video generation (2025)

10. Esser, P., Rombach, R., Ommer, B.: Taming transformers for high-resolution image synthesis. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 12873–12883 (June 2021)

11. Evans, Z., Parker, J.D., Carr, C., Zukowski, Z., Taylor, J., Pons, J.: Stable audio open (2024)

12. Girdhar, R., El-Nouby, A., Liu, Z., Singh, M., Alwala, K.V., Joulin, A., Misra, I.: Imagebind: One embedding space to bind them all. In: CVPR (2023)

13. Gong, Y., Chung, Y.A., Glass, J.: AST: Audio Spectrogram Transformer. In: Proc. Interspeech 2021. pp. 571–575 (2021). https://doi.org/10.21437/Interspeech. 2021-698

14. Gong, Y., Rouditchenko, A., Liu, A.H., Harwath, D., Karlinsky, L., Kuehne, H., Glass, J.R.: Contrastive audio-visual masked autoencoder. In: The Eleventh International Conference on Learning Representations (2023)

15. Goodfellow, I.J., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., Bengio, Y.: Generative adversarial nets. In: Proceedings of the 28th International Conference on Neural Information Processing Systems - Volume 2. p. 2672–2680. NIPS’14, MIT Press, Cambridge, MA, USA (2014)

16. Gu, Y., Zhang, X., Xue, L., Wu, Z.: Multi-scale sub-band constant-q transform discriminator for high-fidelity vocoder (2023)

17. Guo, P., Wang, J., Xing, Z., Liu, C., Dong, D., Qian, X., Wu, Z.: Dera: Decoupled representation alignment for video tokenization (2025)

18. HaCohen, Y., Chiprut, N., Brazowski, B., Shalem, D., Moshe, D., Richardson, E., Levin, E., Shiran, G., Zabari, N., Gordon, O., Panet, P., Weissbuch, S., Kulikov, V., Bitterman, Y., Melumian, Z., Bibi, O.: Ltx-video: Realtime video latent difusion. arXiv preprint arXiv:2501.00103 (2024)

19. Ho, J., Salimans, T.: Classifier-free difusion guidance (2022)

20. Iashin, V., Rahtu, E.: Taming visually guided sound generation. In: British Machine Vision Conference (BMVC) (2021)

21. Iashin, V., Xie, W., Rahtu, E., Zisserman, A.: Synchformer: Eficient synchronization from sparse cues. In: ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). pp. 5325–5329 (2024). https://doi.org/10.1109/ICASSP48485.2024.10448489

22. Ji, S., Jiang, Z., Wang, W., Chen, Y., Fang, M., Zuo, J., Yang, Q., Cheng, X., Wang, Z., Li, R., et al.: Wavtokenizer: an eficient acoustic discrete codec tokenizer for audio language modeling. arXiv preprint arXiv:2408.16532 (2024)

23. Jiang, Y., Chen, Q., Ji, S., Xi, Y., Wang, W., Zhang, C., Yue, X., Zhang, S., Li, H.: UniCodec: Unified audio codec with single domain-adaptive codebook. In: Che, W., Nabende, J., Shutova, E., Pilehvar, M.T. (eds.) Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). pp. 19112–19124. Association for Computational Linguistics, Vienna, Austria (Jul 2025). https://doi.org/10.18653/v1/2025.acl-long.937

24. Kilgour, K., Zuluaga, M., Roblek, D., Sharifi, M.: Fréchet audio distance: A metric for evaluating music enhancement algorithms (2019)

25. Kim, D., He, J., Yu, Q., Yang, C., Shen, X., Kwak, S., Chen, L.C.: Democratizing text-to-image masked generative models with compact text-aware one-dimensional tokens. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 18442–18452 (October 2025)

26. Kingma, D.P., Ba, J.: Adam: A method for stochastic optimization. In: International Conference on Learning Representations (ICLR) (2015)

27. Kong, J., Kim, J., Bae, J.: Hifi-gan: Generative adversarial networks for eficient and high fidelity speech synthesis. In: Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., Lin, H. (eds.) Advances in Neural Information Processing Systems. vol. 33, pp. 17022–17033. Curran Associates, Inc. (2020)

28. Kong, W., Tian, Q., Zhang, Z., Min, R., Dai, Z., Zhou, J., Xiong, J., Li, X., Wu, B., Zhang, J., et al.: Hunyuanvideo: A systematic framework for large video generative models. arXiv preprint arXiv:2412.03603 (2024)

29. Kreuk, F., Synnaeve, G., Polyak, A., Singer, U., Défossez, A., Copet, J., Parikh, D., Taigman, Y., Adi, Y.: Audiogen: Textually guided audio generation. In: The Eleventh International Conference on Learning Representations (2023)

30. Kumar, R., Seetharaman, P., Luebs, A., Kumar, I., Kumar, K.: High-fidelity audio compression with improved RVQGAN. In: Thirty-seventh Conference on Neural Information Processing Systems (2023)

31. Kushwaha, S.S., Tian, Y.: Vintage: Joint video and text conditioning for holistic audio generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 13529–13539 (June 2025)

32. Langman, R., Jukić, A., Dhawan, K., Koluguri, N.R., Li, J.: Spectral codecs: Improving non-autoregressive speech synthesis with spectrogram-based audio codecs (2025)

33. gil Lee, S., Ping, W., Ginsburg, B., Catanzaro, B., Yoon, S.: BigVGAN: A universal neural vocoder with large-scale training. In: The Eleventh International Conference on Learning Representations (2023)

34. Li, J., Zhao, Z., Zhang, Z., Liu, Y., Lin, L., Zhu, Y., Wu, J., Kong, Q., Li, Y.: Meltok: 2d tokenization for single-codebook audio compression (2025)

35. Li, J., Li, D., Savarese, S., Hoi, S.: BLIP-2: Bootstrapping language-image pretraining with frozen image encoders and large language models. In: Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., Scarlett, J. (eds.) Proceedings of the 40th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 202, pp. 19730–19742. PMLR (23–29 Jul 2023)

36. Li, Y., Tian, C., Xia, R., Liao, N., Guo, W., Yan, J., Li, H., Dai, J., Li, H., Yang, X.: Learning adaptive and temporally causal video tokenization in a 1d latent space (2025)

37. Li, Z., Lin, B., Ye, Y., Chen, L., Cheng, X., Yuan, S., Yuan, L.: Wf-vae: Enhancing video vae by wavelet-driven energy flow for latent video difusion model. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 17778–17788 (June 2025)

38. Lin, B., Ge, Y., Cheng, X., Li, Z., Zhu, B., Wang, S., He, X., Ye, Y., Yuan, S., Chen, L., et al.: Open-sora plan: Open-source large video generation model. arXiv preprint arXiv:2412.00131 (2024)

39. Liu, K., Li, W., Chen, L., Wu, S., Zheng, Y., Ji, J., Zhou, F., Luo, J., Liu, Z., Fei, H., Chua, T.S.: Javisdit: Joint audio-video difusion transformer with hierarchical spatio-temporal prior synchronization. In: The Fourteenth International Conference on Learning Representations (2026)

40. Loshchilov, I., Hutter, F.: Decoupled weight decay regularization. In: International Conference on Learning Representations (2019)

41. Low, C., Wang, W., Katyal, C.: Ovi: Twin backbone cross-modal fusion for audiovideo generation (2025)

42. Mao, Y., Shen, X., Zhang, J., Qin, Z., Zhou, J., Xiang, M., Zhong, Y., Dai, Y.: TAVGBench: Benchmarking text to audible-video generation. In: ACM Multimedia 2024 (2024)

43. Pham, K.T., He, Y., Xing, Y., Chen, Q., Chen, L.: Spa2v: Harnessing spatial auditory cues for audio-driven spatially-aware video generation. In: Proceedings of the 33rd ACM International Conference on Multimedia. p. 10476–10485. MM ’25, Association for Computing Machinery, New York, NY, USA (2025). https: //doi.org/10.1145/3746027.3755705

44. Radford, A., Wu, J., Child, R., Luan, D., Amodei, D., Sutskever, I.: Language models are unsupervised multitask learners (2019)

45. Roux, J.L., Wisdom, S., Erdogan, H., Hershey, J.R.: Sdr - half-baked or well done? (2018)

46. Ruan, L., Ma, Y., Yang, H., He, H., Liu, B., Fu, J., Yuan, N.J., Jin, Q., Guo, B.: Mm-difusion: Learning multi-modal difusion models for joint audio and video generation. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 10219–10228 (June 2023)

47. Seedance, T.: Seedance 1.5 pro: A native audio-visual joint generation foundation model (2025)

48. Song, J., Kwon, M., Jeong, J., Uh, Y.: Syncphony: Synchronized audio-to-video generation with difusion transformers. In: The Fourteenth International Conference on Learning Representations (2026)

49. Sun, P., Jiang, Y., Chen, S., Zhang, S., Peng, B., Luo, P., Yuan, Z.: Autoregressive model beats difusion: Llama for scalable image generation. arXiv preprint arXiv:2406.06525 (2024)

50. Tang, A., He, T., Guo, J., Cheng, X., Song, L., Bian, J.: Vidtok: A versatile and open-source video tokenizer. arXiv preprint arXiv:2412.13061 (2024)

51. Touvron, H., Lavril, T., Izacard, G., Martinet, X., Lachaux, M.A., Lacroix, T., Rozière, B., Goyal, N., Hambro, E., Azhar, F., Rodriguez, A., Joulin, A., Grave, E., Lample, G.: Llama: Open and eficient foundation language models (2023)

52. Tseng, H.Y., Jiang, L., Liu, C., Yang, M.H., Yang, W.: Regularizing generative adversarial networks under limited data. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 7921–7931 (June 2021)

53. Unterthiner, T., van Steenkiste, S., Kurach, K., Marinier, R., Michalski, M., Gelly, S.: Towards accurate generative models of video: A new metric & challenges (2019)

54. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L.u., Polosukhin, I.: Attention is all you need. In: Guyon, I., Luxburg, U.V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 30. Curran Associates, Inc. (2017)

55. Viertola, I., Iashin, V., Rahtu, E.: Temporally aligned audio for video with autoregression. In: ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE (2025)

56. Wan, T.: Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314 (2025)

57. Wang, B., Yue, Z., Zhang, F., Chen, S., Bi, L., Zhang, J., Song, X., Chan, K.Y., Pan, J., Wu, W., Zhou, M., Lin, W., Pan, K., Zhang, S., Jia, L., Hu, W., Zhao,

W., Zhang, H.: Selftok: Discrete visual tokens of autoregression, by difusion, and for reasoning (2025)

58. Wang, H., Suri, S., Ren, Y., Chen, H., Shrivastava, A.: LARP: Tokenizing videos with a learned autoregressive generative prior. In: The Thirteenth International Conference on Learning Representations (2025)

59. Wang, J., Jiang, Y., Yuan, Z., PENG, B., Wu, Z., Jiang, Y.G.: Omnitokenizer: A joint image-video tokenizer for visual generation. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024)

60. Wang, K., Deng, S., Shi, J., Hatzinakos, D., Tian, Y.: Av-dit: Taming image diffusion transformers for eficient joint audio and video generation. In: Proceedings of the 33rd ACM International Conference on Multimedia. p. 10486–10495. MM ’25, Association for Computing Machinery, New York, NY, USA (2025). https://doi.org/10.1145/3746027.3755713

61. Wang, Z., Bovik, A., Sheikh, H., Simoncelli, E.: Image quality assessment: from error visibility to structural similarity. IEEE Transactions on Image Processing 13(4), 600–612 (2004). https://doi.org/10.1109/TIP.2003.819861

62. Wen, X., Zhao, B., Elezi, I., Deng, J., Qi, X.: ”principal components” enable a new language of images. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 16641–16651 (October 2025)

63. Weng, S., Zheng, H., Chang, Z., Li, S., Shi, B., Wang, X.: Audio-sync video generation with multi-stream temporal control. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2026)

64. Xing, Y., Fei, Y., He, Y., Chen, J., Xie, J., Chi, X., Chen, Q.: Videovae+: Large motion video autoencoding with cross-modal video vae. In: Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV). pp. 17951– 17960 (October 2025)

65. Xing, Y., He, Y., Tian, Z., Wang, X., Chen, Q.: Seeing and hearing: Opendomain visual-audio generation with difusion latent aligners. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). pp. 7151–7161 (June 2024)

66. Yamamoto, R., Song, E., Kim, J.M.: Parallel wavegan: A fast waveform generation model based on generative adversarial networks with multi-resolution spectrogram (2020)

67. Yang, Z., Teng, J., Zheng, W., Ding, M., Huang, S., Xu, J., Yang, Y., Hong, W., Zhang, X., Feng, G., Yin, D., Yuxuan.Zhang, Wang, W., Cheng, Y., Xu, B., Gu, X., Dong, Y., Tang, J.: Cogvideox: Text-to-video difusion models with an expert transformer. In: The Thirteenth International Conference on Learning Representations (2025)

68. Yariv, G., Gat, I., Benaim, S., Wolf, L., Schwartz, I., Adi, Y.: Diverse and aligned audio-to-video generation via text-to-video model adaptation. Proceedings of the AAAI Conference on Artificial Intelligence 38(7), 6639–6647 (Mar 2024). https: //doi.org/10.1609/aaai.v38i7.28486

69. Yu, Q., Weber, M., Deng, X., Shen, X., Cremers, D., Chen, L.C.: An image is worth 32 tokens for reconstruction and generation. In: The Thirty-eighth Annual Conference on Neural Information Processing Systems (2024)

70. Yu, S., Kwak, S., Jang, H., Jeong, J., Huang, J., Shin, J., Xie, S.: Representation alignment for generation: Training difusion transformers is easier than you think. In: International Conference on Learning Representations (2025)

71. Zhang, G., Zhou, Z., Hu, T., Peng, Z., Zhang, Y., Chen, Y., Zhou, Y., Lu, Q., Wang, L.: Uniavgen: Unified audio and video generation with asymmetric crossmodal interactions (2025)

72. Zhang, R., Isola, P., Efros, A.A., Shechtman, E., Wang, O.: The unreasonable efectiveness of deep features as a perceptual metric. In: Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (June 2018)

73. Zhang, Y., Gu, Y., Zeng, Y., Xing, Z., Wang, Y., Wu, Z., Liu, B., Chen, K.: Foleycrafter: Bring silent videos to life with lifelike and synchronized sounds. Int. J. Comput. Vision 134(1) (Jan 2026). https://doi.org/10.1007/s11263-025- 02649-3

74. Zheng, J., Pan, S., Yao, Y., Wang, Z., Wang, D., Liu, T.: Aligning what matters: Masked latent adaptation for text-to-audio-video generation. In: The Thirty-ninth Annual Conference on Neural Information Processing Systems (2026)

75. Zheng, Z., Peng, X., Yang, T., Shen, C., Li, S., Liu, H., Zhou, Y., Li, T., You, Y.: Open-sora: Democratizing eficient video production for all. arXiv preprint arXiv:2412.20404 (2024)

## A Additional Experiment Details

## A.1 Datasets

Statistics. We conduct our experiments on VGGSound [4] and TAVGBench [42] datasets. VGGSound consists of more than 210K sounding video clips spanning across 310 diferent classes and is commonly used in various audio-visual understanding and generation tasks. Due to data corruption, only approximately 200K audio-video pairs are available for our usage, of which the train split contains 180K samples and the remaining samples belong to the test split. Meanwhile, TAVGBench is a larger-scale dataset containing 1.7M samples with better alignment between auditory and visual elements compared to VGGSound. However, we only utilize a subset of 460K high-quality samples filtered by [39] through a series of filtering strategies to accommodate resource constraints.

Composition. Eventually, a total of 640K data from both TAVGBench and the train split of VGGSound are used to train the AVTok tokenizer for the reconstruction task, while only the 180K VGGSound ones are used to train AR generative models for the downstream generation tasks. The test set of VG-GSound is used for all evaluations. All audio-video pairs are preprocessed following the adopted neural vocoder HiFi-GAN [27] and the baseline 1D video tokenizer LARP [58], respectively, to the default input resolutions mentioned in the main text, while ensuring that they are synchronized in the temporal dimension with the duration deliberately set at around 4 seconds. This facilitates that the audio component is long enough to provide suficient and meaningful auditory information for model training while maintaining eficiency and compatibility with the adopted pretrained audio-visual foundation model [1].

## A.2 Model Implementation

AVTok Tokenizer. We follow LARP [58] to adopt fixed sin-cos positional encoding [54] in both the encoder and decoder of AVTok. In the encoder, fixed 3D and 2D positional encodings are applied to each video and audio patch, while in the decoder, fixed 1D positional encodings are added to each holistic video and audio token. Notably, since the patch queries and holistic queries for both modalities are position-wise learnable parameters, they do not necessitate additional positional encodings.

The encoder and decoder of AVTok adopt the standard transformer design [54] in which each layer consists of multi-headed self-attention (M SA), layer normalization $\left( L N _ { 1 } , L N _ { 2 } \right)$ , and multilayer perceptron (MLP ) blocks with residual connections. Specifically, the forward pass of each layer is as follows:

$$
\mathbf {x} ^ {\prime} = M S A (L N _ {1} (\mathbf {x})) + \mathbf {x}, \quad \mathbf {y} = M L P (L N _ {2} (\mathbf {x} ^ {\prime})) + \mathbf {x} ^ {\prime},
$$

where x represents the concatenation of learnable holistic queries/patch queries with input patches/holistic tokens described in the main text. We then adapt the philosophy of [1,14] to use separate sets of $\left( L N _ { 1 } ^ { a } , L N _ { 2 } ^ { a } \right)$ and $\left( L N _ { 1 } ^ { v } , L N _ { 2 } ^ { v } \right)$ for audio and video streams to eficiently formulate the final dual-stream architecture.

We employ HiFi-GAN [27], CAV-MAE Sync $[ 1 ] ,$ , and GPT-2 [44] to be our neural vocoder, audio-visual foundational model $\mathcal { M } _ { F }$ , and cross-modal AR generative prior model $\mathcal { M } _ { P }$ , respectively, with the objectives detailed in the main text. During training, only $\mathcal { M } _ { P }$ and the small MLP projector $h _ { \phi }$ associated with $\mathcal { M } _ { F }$ are trained while the others are kept frozen. During inference, both the foundational and prior models are discarded.

AR Generative Models. We adopt Llama-like transformers [49,51] as our AR generative models. Following LARP [58], we leverage absolute learned positional encodings. During training, a dropout rate of 0.1 is applied to token sequences, residual connections, and feedforward layers. Furthermore, the SVQ quantizer of AVTok is configured to be deterministic during the training of AR generative models to encourage a more accurate latent representation learning.

## A.3 Training Details

Reconstruction. During the training of the AVTok tokenizer, $\mathcal { L } _ { r e c } ^ { a }$ and $\mathcal { L } _ { r e c } ^ { v }$ are the two primary objectives for the learning of audio and video streams. For video, $\mathcal { L } _ { r e c } ^ { v }$ comprises $L _ { 1 }$ reconstruction term, LPIPS term [72] for perceptual enhancement, and GAN adversarial term [15] for improved sharpness and finegrained textual details, with corresponding weights of (1.0, 1.0, 0.3) following [58]. Similarly for audio, $\mathcal { L } _ { r e c } ^ { a }$ is a combination of Multi-Scale Mel-Spectrogram reconstruction term [30], deep feature matching term [27], and GAN adversarial term [33], with respective weights of (15.0, 2.0, 1.0) according to [33].

Notably, a ViT-based Discriminator [8] is adopted to compute the GAN component of the video stream, while Multi-Scale Sub-Band CQT Discriminator [16] and Multi-Period Discriminator [27] are employed to compute the GAN and feature matching components of the audio stream. These discriminators are updated once per five training iterations of the AVTok tokenizer with a 70% lower learning rate and LeCam regularization [52] applied for training stability. Besides, SVQ quantization loss with total weight of 0.1 is also added, in which we follow [10] to use a commitment and codebook loss weights of (0.25, 1.0).

With the final training objective and the component weights $\lambda _ { 1 , 2 , 3 , 4 }$ defined in the main text, we then conduct training for AVTok using the proposed VFAL hierarchical strategy. It is decomposed into three progressive stages with the primary target modules set as: (1) Video Reconstruction, when the encoder and decoder with video-specific normalization layers, denoted as $\mathcal { E } ( \cdot ; L N _ { 1 } ^ { v } , L N _ { 2 } ^ { v } )$ and $\mathcal { D } ( \cdot ; L N _ { 1 } ^ { v } , L N _ { 2 } ^ { v } )$ , and learnable queries $( \mathbf { Q } _ { L } ^ { v } , \mathbf { Q } _ { P } ^ { v } )$ are trained for 75 epochs; (2) Audio Reconstruction, when $\mathcal { E } , \mathcal { D }$ are frozen and shared between two streams except for audio-specific learnable queries $( \mathbf { Q } _ { L } ^ { a } , \mathbf { Q } _ { P } ^ { a } )$ and normalization layers $( L N _ { 1 } ^ { a } , L N _ { 2 } ^ { a } ) _ { \{ \varepsilon , { \cal D } \} }$ , which are trained for 35 epochs; (3) Refinement, when only the decoder with both streams $\mathcal { D } ( \cdot ; L N _ { 1 } ^ { \{ a , v \} } , L N _ { 2 } ^ { \{ a , v \} } )$ is further finetuned for 10 epochs. A batch size of 112 and the Adam optimizer [26] with a base $l r = 0 . 0 0 0 1$ ， $( \beta _ { 1 } , \beta _ { 2 } ) \ : = \ : ( 0 . 9 , 0 . 9 5 )$ , and a warm-up cosine schedule are used for all stages. Additional details on other modules and training settings can be found in Tab. 4 and Fig. 6. Note that in Stage 1, $\mathcal { L } _ { p r i o r }$ is computed with holistic video tokens $\mathbf { x } ^ { v } \ { \mathrm { o n l y } } .$

Table 4: Detailed training settings for the three stages of VFAL.

<table><tr><td>Setting</td><td>Stage 1</td><td>Stage 2</td><td>Stage 3</td></tr><tr><td>training purpose</td><td>Video Reconstruction</td><td>Audio Reconstruction</td><td>Refinement</td></tr><tr><td>trainable modules</td><td> $\mathcal{E}(\cdot;LN_1^v, LN_2^v),$  $\mathcal{D}(\cdot;LN_1^v, LN_2^v)$  $\mathbf{Q}_L^v, \mathbf{Q}_P^v, \mathcal{Q}, \mathcal{M}_P$ </td><td> $(LN_1^a, LN_2^a)_{\{\mathcal{E}, \mathcal{D}\}}$  $\mathbf{Q}_L^a, \mathbf{Q}_P^a, \mathcal{M}_P, h_\phi$ </td><td> $\mathcal{D}(\cdot;LN_1^{\{a,v\}}, LN_2^{\{a,v\}})$ </td></tr><tr><td>base learning rate</td><td>0.0001</td><td>0.0001</td><td>0.0001</td></tr><tr><td>scheduler</td><td>cosine</td><td>cosine</td><td>cosine</td></tr><tr><td> $\beta_1, \beta_2$ </td><td>0.9, 0.95</td><td>0.9, 0.95</td><td>0.9, 0.95</td></tr><tr><td>warm-up epochs</td><td>8</td><td>3</td><td>1</td></tr><tr><td>total epochs</td><td>75</td><td>35</td><td>10</td></tr><tr><td>batch size</td><td>112</td><td>112</td><td>112</td></tr><tr><td> $\lambda_1, \lambda_2, \lambda_3, \lambda_4$ </td><td>1.0, 0.0, 0.0, 0.06</td><td>0.1, 1.0, 0.5, 0.06</td><td>1.0, 0.01, 0.5, 0.06</td></tr></table>

![](images/92429ecfb8b8f8d7b85a8877cf8f784578885d8bbd0c142e475177d506a75c3e.jpg)  
Fig. 6: Illustration of trainable parameters at diferent stages of VFAL.

Downstream Generation. We train an AR generative model for each downstream generation task, including audio-to-video (A2V), video-to-audio (V2A), and class-conditional joint audio-video generation (cJAVG). They are all trained on the train split of VGGSound for 75 epochs with a batch size of 128. The AdamW optimizer [40] is used with $( \beta _ { 1 } , \beta _ { 2 } ) = ( 0 . 9 , 0 . 9 5 )$ , a weight decay of 0.05, and a base learning rate of 0.0006, following a warm-up cosine learning rate schedule with 4 warm-up epochs. When generating samples, we follow LARP [58] to apply a small Classifier-Free Guidance (CFG) scale of 1.25 [19] for cJAVG while excluding it for V2A and A2V tasks, and do not use top-k or top-p sampling methods. Besides, the conditioning video/audio for V2A/A2V is input to the corresponding streaming of AVTok to efortlessly produce the conditioning video/audio holistic tokens, while the vanilla model may struggle. At inference time, the AR models predict the holistic tokens for the respective modality outcome, conditioned on the holistic tokens obtained.

## A.4 Evaluation Details

Representative Baselines. For reconstruction, there is no open-source baseline available for direct comparison with our AVTok tokenizer due to the novelty of the proposed unified audio-video tokenization task. Therefore, we select several state-of-the-art unimodal 1D tokenizers of each modality that are closely relevant to AVTok for reasonable comparisons, as included in the main text. Regarding downstream generation, the baselines for each task are selected under a similar relevance consideration to ensure that the comparisons are as fair as possible. Since some baselines, such as Ovi [41], JavisDiT [39], or MMAudio [5], require textual captions as conditions to control the generation process, we bypass them by utilizing class labels available in VGGSound as an alternative.

Metrics. To evaluate accuracy, realism, and perceptual quality of the reconstructed videos/audios with respect to the ground-truths, we respectively adopt PSNR [61]/SI-SDR [45], FVD [53]/FAD [24], and LPIPS [72]/MR-STFT [66]. For downstream generation, we again use FVD [53]/FAD [24] with additions of DeSync [21] and ImageBind [12] (IB) Score to assess realism, temporal synchronization, and semantic alignment accordingly. Notably, for A2V and V2A tasks, FVD/FAD are computed between the generated results and the ground-truths, while the rest are computed between the generated results and input conditions. For cJAVG, FAD and FVD are calculated similarly, whereas DeSync and IB Score are computed between the generated audio-video pairs.

## B Additional Results

## B.1 Ablation Study

Generation Latency. In addition to model capacity measured by the number of parameters, we evaluate the eficiency of our complete generation pipeline comprising AVTok tokenizer and an AR generative model compared to the other baselines. Specifically, we measure the TFLOPs and average latency per sample of all methods in generating 100 samples with a batch size of 1 in the same environment. For other settings of the baselines, we use their default configuration as deemed necessary. The results shown in Tab. 5 highlight the eficiency of our pipelines, complementing their efectiveness to generate high-fidelity samples as demonstrated in the main text.

Table 5: Comparison of generation eficiency.

<table><tr><td rowspan="2">Task</td><td rowspan="2">Method</td><td rowspan="2">Gen Type</td><td colspan="2">#Param</td><td rowspan="2">Latency↓ (sec)</td><td rowspan="2">TFLOPs↓</td></tr><tr><td>Tokenizer</td><td>Generator</td></tr><tr><td rowspan="2">A2V</td><td>TempoTokens [68]</td><td>Diff</td><td>83.7M</td><td>1.9B</td><td>21.103</td><td>1.62K</td></tr><tr><td>AVTok-A2V (Ours)</td><td>AR</td><td>208.4M</td><td>632.0M</td><td>11.058</td><td>1.82</td></tr><tr><td rowspan="5">V2A</td><td>MMAudio [5]</td><td>FM</td><td>298.5M</td><td>1.3B</td><td>1.304</td><td>31.77</td></tr><tr><td>VinTAGe [31]</td><td>FM</td><td>110.6M</td><td>1.5B</td><td>23.423</td><td>474.69</td></tr><tr><td>V-AURA [55]</td><td>AR</td><td>76.7M</td><td>816.9M</td><td>11.290</td><td>191.05</td></tr><tr><td>SpecVQGAN [20]</td><td>AR</td><td>76.4M</td><td>332.4M</td><td>1.307</td><td>17.72</td></tr><tr><td>AVTok-V2A (Ours)</td><td>AR</td><td>208.4M</td><td>632.0M</td><td>1.395</td><td>1.82</td></tr><tr><td rowspan="3">cJAVG</td><td>JavisDiT [39]</td><td>FM</td><td>448.7M</td><td>8.9B</td><td>32.240</td><td>2.60K</td></tr><tr><td>Ovi [41]</td><td>FM</td><td>988.6M</td><td>17.3B</td><td>87.282</td><td>14.99K</td></tr><tr><td>AVTok-cJAVG (Ours)</td><td>AR</td><td>208.4M</td><td>632.4M</td><td>12.755</td><td>3.48</td></tr></table>

Table 6: Comparison of diferent model scale.

<table><tr><td rowspan="2">Model</td><td colspan="3">Configuration</td><td colspan="3">Video Reconstruction</td><td colspan="3">Audio Reconstruction</td></tr><tr><td>Hidden Size</td><td>Depth</td><td>Num Heads</td><td>PSNR↑</td><td>rFVD↓</td><td>LPIPS↓</td><td>SI-SDR↓</td><td>rFAD↓</td><td>MR-STFT↓</td></tr><tr><td>AVTok</td><td>768</td><td>12</td><td>12</td><td>25.62</td><td>12.80</td><td>0.126</td><td>23.09</td><td>5.93</td><td>1.523</td></tr><tr><td>AVTok-B</td><td>768</td><td>8</td><td>12</td><td>24.65</td><td>12.94</td><td>0.148</td><td>24.99</td><td>6.01</td><td>1.794</td></tr><tr><td>AVTok-S</td><td>768</td><td>6</td><td>8</td><td>23.39</td><td>19.12</td><td>0.193</td><td>25.29</td><td>8.86</td><td>2.333</td></tr></table>

Tokenizer Scalability. To explore the efect of scaling our AVTok tokenizer, we adjust the model size while maintaining the same number of latent tokens to construct another two smaller variants AVTok-B and AVTok-S, and conduct training for them under identical settings as the default. As shown in Tab. 6, the performance changes most significantly when scaling from AVTok-S up to AVTok-B with 6.18 FVD and 2.85 FAD lower results, but saturates with minor improvements when scaling up to the largest default. Given this indication, we opt not to scale the model further and use the current largest variant as the default choice.

In addition, we also examine the model performance when adjusting the number of holistic tokens necessary to encode and reconstruct input audio and video. In particular, we alternately halve the default number of tokens for one modality while keeping that of the other unchanged to quantify the efect they induce. Intuitively, the use of fewer tokens enables a faster AR generation process but trades of with degradation in reconstruction quality, which is reflected in Tab. 7. Interestingly, we find that a decreasing number of video tokens significantly affects audio reconstruction, while a marginal impact is observed conversely. This suggests that the model is more susceptible to changes in the video stream, where the information is denser and richer than the audio stream.

External Models. We also ablate on diferent selections of external models, including the vocoders and audio-visual foundation models M . First, we replace CAV-MAE Sync [1] with its predecessor CAV-MAE [14] as our foundational model, which results in performance degradation despite having similar model size. Second, we adopt BigVGAN [33], a more robust neural vocoder, as an alternative to HiFi-GAN [27], which only yields slight improvements and has a significantly larger model size. Therefore, we opt to use CAV-MAE Sync [1] and BigVGAN [33] by default, considering the balance of eficiency and efectiveness.

Table 7: Comparison of diferent holistic token counts.

<table><tr><td rowspan="2">Model</td><td colspan="2">Configuration</td><td colspan="3">Video Reconstruction</td><td colspan="3">Audio Reconstruction</td></tr><tr><td>#Video Tokens</td><td>#Audio Tokens</td><td>PSNR↑</td><td>rFVD↓</td><td>LPIPS↓</td><td>SI-SDR↓</td><td>rFAD↓</td><td>MR-STFT↓</td></tr><tr><td>AVTok</td><td>1024</td><td>128</td><td>25.62</td><td>12.80</td><td>0.126</td><td>23.09</td><td>5.93</td><td>1.523</td></tr><tr><td>AVTok-a64</td><td>1024</td><td>64</td><td>25.37</td><td>12.75</td><td>0.128</td><td>25.35</td><td>12.77</td><td>2.263</td></tr><tr><td>AVTok-v512</td><td>512</td><td>128</td><td>23.94</td><td>23.85</td><td>0.172</td><td>26.10</td><td>14.90</td><td>2.491</td></tr></table>

Table 8: Comparison of using diferent foundational models and vocoders.

<table><tr><td rowspan="2">Model</td><td colspan="2">Configuration</td><td colspan="3">Video Reconstruction</td><td colspan="3">Audio Reconstruction</td></tr><tr><td> $\mathcal{M}_{F}$ </td><td>Vocoder</td><td>PSNR↑</td><td>rFVD↓</td><td>LPIPS↓</td><td>SI-SDR↓</td><td>rFAD↓</td><td>MR-STFT↓</td></tr><tr><td>AVTok</td><td>CAV-MAE Sync [1]</td><td>HiFi-GAN [27]</td><td>25.62</td><td>12.80</td><td>0.126</td><td>23.09</td><td>5.93</td><td>1.523</td></tr><tr><td>AVTok-F</td><td>CAV-MAE [14]</td><td>HiFi-GAN [27]</td><td>25.50</td><td>12.84</td><td>0.128</td><td>24.19</td><td>6.40</td><td>1.622</td></tr><tr><td>AVTok-V</td><td>CAV-MAE Sync [1]</td><td>BigVGAN [33]</td><td>25.61</td><td>12.59</td><td>0.125</td><td>22.78</td><td>5.72</td><td>1.511</td></tr></table>

## B.2 Visualization

We provide additional qualitative results for reconstruction, audio-to-video, videoto-audio, and class-conditional joint audio-video generation tasks in Fig. 7, 8, 9, 10, respectively. These results consistently highlight that the AVTok tokenizer excels in both reconstruction and when incorporated into the downstream AR generative models for generation tasks. Besides, generated sounding video files in MP4 format are also included for subjective inspection.

## C Discussion

## C.1 Potential Limitations

Our proposed AVTok unified tokenizer demonstrates outstanding performance in audio-video tokenization, and excels when integrated into our AR generative models for downstream generation tasks. However, certain limitations remain, which open promising directions for future exploration.

Data Scale and Complexity. Our AVTok tokenizer and AR generative models are trained on approximately 640K and 180K data entries only. Despite maintaining eficacy and eficiency, this may inevitably constrain scalability compared to larger proprietary models and systems. In addition, due to the inherent simplicity of the scenes included in the datasets, artifacts may appear in the generated samples when scenes are particularly complex. We believe that larger-scale training with more diverse and high-quality audio-video datasets could enhance the robustness and generalizability of the models.

![](images/052bbe2a9237c821459e02aa7de8b9494346d196dde05aad53ae337c00bb549a.jpg)  
Fig. 7: Additional qualitative reconstruction results.

Model Design and Training Resource. Similar to other transformer-based unimodal tokenizers, AVTok inherently performs best with fixed-resolution audios and videos due to positional encoding constraints. Besides, amid limited training resources, we could only train and evaluate our models’ capabilities on 16-frame short clips with 128×128×3 low resolution and roughly 4-second audios with a sampling rate of 22kHz. We contemplate that with suficient resources, scaling up AVTok and the AR generative models can enable reconstructing and synthesizing audio-video pairs with larger resolution, longer duration, and higher quality, meeting user demands nowadays.

Synchronization Modeling. The current architecture and training setups of AVTok tokenizer and AR generative models only partially exploit synchronization between auditory and visual elements in an implicit manner by: (1) using synchronized sounding video samples as input; and (2) enabling cross-modal interaction with shared parameters, AR prior and foundation models for AV-Tok, and causal self-attention mechanism in AR generative models. Therefore, the generated audios and videos may lack temporal alignment. We think that modeling synchronization between the two modalities more explicitly can help mitigate this issue and improve the final performance.

![](images/a33b4a35de09777688f634e59f62e7c9972d3ff1e01b5f5c17fbed0cf154e03f.jpg)  
Fig. 8: Additional qualitative results for audio-to-video generation task.

End-to-end Training. The training of the AVTok tokenizer relies on the proposed VFAL hierarchical strategy. Although efective, it requires complicated stage-wise tuning and is prone to cascading errors, which may eventually lead to suboptimal performance. Conversely, a single-stage end-to-end alternative could alleviate these issues by streamlining the training process with a unified objective, albeit with higher optimization sensitivity and computational cost.

## C.2 Statements

Ethics. All datasets and models used in this work are publicly accessible online and contain no private or sensitive information.

![](images/8fa15ac258f33ec6375fa1f943b2c5628404208a57f3180234c7479c04c0d88d.jpg)  
Fig. 9: Additional qualitative results for video-to-audio generation task.

Reproducibility. To ensure full reproducibility, we detail our model’s design, training, and evaluation in the main text and appendix, and will publicly release all code, checkpoints, and datasets.

LLM Usage. Large Language Models (LLMs) were utilized solely as writing aids to polish the language and refine the presentation. They played no role in developing the core concepts or research design.

![](images/a87c15c301cffe444bffb0553bc00d64c630ac0863aa3033984ed414d7aa05cb.jpg)  
Fig. 10: Additional qualitative results for class-conditional joint generation task.