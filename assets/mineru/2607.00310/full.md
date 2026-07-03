# RetailSMV: Exocentric vs. Egocentric Adaptation of Foundation Video World Models in Retail

DreamVu<sup>1</sup>

Foundation video difusion models are increasingly viewed as world simulators for embodied agents, yet their pretraining on internet-scale generic video leaves them poorly aligned with real-world deployment domains. We study parameter-eficient adaptation of a pretrained foundation video world model to retail scenes: when synchronized egocentric and exocentric video of the same activity are available, which viewpoint of training data produces the strongest adapted model?

We introduce RetailSMV (Retail Synchronized Multi-View), a corpus of 32,105 captioned retail clips from five supermarkets with synchronized ego/exo capture from the store-staf perspective (stocking, arranging, weighing, managing supply carts, scanning at checkout), rather than the customer-centric framing of prior retail video corpora, and train three matched Low-Rank Adaptation (LoRA) configurations of Cosmos3-Nano (egocentric-only, exocentric-only, combined) under identical hyperparameters. On a 200-clip held-out test set evaluated with seven complementary metrics under a strict paired statistical protocol, exocentric-only adaptation matches or exceeds combined adaptation on six of seven point estimates and is significantly better on LPIPS, PSNR, and DreamSim, despite training on only 15,985 exocentric clips (versus 32,105 for combined). A symmetric paired comparison further shows that adding exocentric data to egocentric-only training helps while adding egocentric data to exocentric-only training hurts. The absolute adaptation gap is largest at the shortest rollout time, identifying the near-horizon prediction window as the regime in which adaptation is most beneficial.

Date: July 2, 2026

Base vs. RetailSMV-adapted video world model The same retail prompts are continued by a pretrained foundation model and by our adapted model.  
![](images/a5234bc0ac7ce22b9e1127419a09f1e7efc87175fcfa814837bd77049ad8e1b9.jpg)  
RetailSMV adaptation preserves scene layout, object permanence, and action grounding where the pretrained model drifts.  
Figure 1 Base vs. RetailSMV-adapted video world model. The same retail prompts are continued by the pretrained Cosmos3-Nano foundation model (top, red) and by our RetailSMV-adapted LoRA (bottom, green) under identical inference settings. RetailSMV adaptation preserves scene layout (hand-of watermelon), action grounding (weigh tomato crate), and physical geometry (open fridge) where the pretrained baseline drifts.

## 1 Introduction

A video world model takes an observation history (typically a text description, an image, or a short video clip) and predicts a plausible video continuation. Modern video difusion systems such as Sora OpenAI (2024), Stable Video Difusion Blattmann et al. (2023), Movie Gen Meta GenAI (2024), and NVIDIA Cosmos3-Nano and Cosmos-Predict 2.5 NVIDIA (2025) now produce coherent multi-second video from text prompts and have been framed as world simulators for embodied agents Ha and Schmidhuber (2018); Gao et al. (2025). The promise is that a faithful world model could let an embodied agent plan, evaluate counterfactuals, generate training data, and verify policies before acting in the physical world. We discuss the broader landscape of video difusion, world models, and embodied deployment in section 2.

That promise depends on domain alignment. Foundation video models are pretrained on internet-scale generic video (entertainment content, vlogs, dashcam footage, and robot demonstrations) which may underrepresent the structured visual vocabulary of specific deployment domains. Retail environments are a clear example: dense product shelving, narrow parallel aisles, repetitive geometry at multiple scales, distinctive signage and end-caps, and multi-person dynamics around carts and checkout counters. A retail world model must render scenes built from this vocabulary and continue shopping behaviors consistent with the physical and behavioral regularities of supermarket environments. Without explicit adaptation, generated retail scenes drift toward generic interiors, and predicted human motion drifts away from realistic shopping within a few seconds of rollout.

This paper studies parameter-eficient adaptation of a pretrained foundation video world model to retail scenes via Low-Rank Adaptation (LoRA) Hu et al. (2022), and introduces the dataset that makes such a study possible. We use Cosmos3-Nano NVIDIA (2025) as the base model, a 16-billion-parameter text-andimage-conditioned video difusion transformer that is, to our knowledge, one of the largest openly released foundation video models with a documented post-training recipe. As training data we introduce RetailSMV (Retail Synchronized Multi-View), a retail video corpus of 32,105 captioned clips collected across five real-world supermarkets. RetailSMV provides synchronized ego/exo capture of store-staf operational work—stocking, arranging, weighing and labelling produce, carrying crates, pushing supply carts, scanning items at the checkout, and assisting customers—rather than the customer-side behaviour that prior retail video corpora capture. Every work episode is recorded simultaneously by a head-mounted egocentric camera worn by the staf member and a fixed exocentric scene camera that observes the same activity from a third-person perspective. To our knowledge no prior dataset combines a real-world retail deployment domain with synchronized ego/exo capture of the same activity at the scale required to fine-tune a foundation video difusion model; this combination is what enables the controlled view-stratified study reported here.

We organize the study around a single experimental question (see figure 2 for the full pipeline):

Which viewpoint of training data (egocentric, exocentric, or their combination) produces the best adapted video world model for retail scenes?

To answer this we train three LoRA configurations of Cosmos3-Nano under matched hyperparameters and identical optimization budget: (i) egocentric-only $( n { = } 1 6 , 1 2 0 \ \mathrm { c l i p s } )$ , (ii) exocentric-only $( n { = } 1 5 , 9 8 5 \ \mathrm { c l i p s } )$ , and (iii) combined ego+exo $( n { = } 3 2 , 1 0 5 \ \mathrm { c l i p s } )$ . To control for seed variance we additionally train a second seed of the combined configuration. We evaluate every configuration on the same held-out test set of 200 stratified retail clips balanced across both views (n=100 egocentric and n=100 exocentric), generating one video per (configuration, clip) pair under identical inference settings. The balanced stratification reflects both natural deployment viewpoints (the head-mounted egocentric camera and the fixed exocentric scene camera) at equal weight, and the egocentric subset enables a cross-view transfer check (section B.2). The test set is disjoint from both the training data and the adapter-selection validation pool, eliminating selection leakage (section 4).

Findings. Our analysis surfaces three robust empirical findings:

1. Adaptation succeeds uniformly. All three LoRA configurations reduce the validation difusion loss by a factor of approximately 2.8 (from 1.006 to a tight range of 0.355−0.363). Every one of 200 paired evaluation samples improves over the pretrained baseline under each LoRA configuration, with both parametric and non-parametric paired tests at $p \ll 0 . 0 0 1$ (table 4; exact values in section B.1). Seed-induced variation on the combined configuration is smaller than the configuration-to-configuration spread, supporting the interpretation that configuration rankings reflect properties of the training data rather than optimization noise.

2. Exocentric-only adaptation matches or exceeds combined adaptation on six of seven point estimates and is significantly better on LPIPS, PSNR, and DreamSim. Despite training on only the 15,985 exocentric clips (versus the 32,105 clips available to the combined configuration), the exocentric-only adapter achieves the lowest R3D-Fréchet (35.14 vs. base 42.69, a 17.7% relative reduction; combined 36.20, a 15.2% reduction), the lowest JEDi (0.829 vs. base 1.246, a 33.5% relative reduction; combined 0.892, a 28.5% reduction), the largest LPIPS reduction of 0.058 at 100% paired win-rate $( p < 1 0 ^ { - 3 4 } )$ , the largest PSNR improvement of +0.575 dB at 85% win-rate, and the largest DreamSim reduction of 0.062 at 92.5% win-rate $( p < 1 0 ^ { - 3 1 } )$ . The combined configuration matches exo on SSIM (+0.022) and CLIPScore (+0.017 in the Hessel formulation) but is significantly worse than exo on the perceptual and pixel-fidelity metrics in direct paired tests (LPIPS $p { < } 1 0 ^ { - 7 }$ , PSNR $\scriptstyle { p = 0 . 0 0 3 }$ , DreamSim $p { < } 1 0 ^ { - 9 } )$ . The two distributional metrics agree on the ranking exo < combined $< \mathrm { e g o } <$ base, so the conclusion does not rely on either single feature backbone. This result suggests that combined ego+exo training does not improve over exo-only training on this corpus, and tha the in-distribution exocentric subset is what carries the adaptation signal.

3. The adaptation gap is concentrated in the near-horizon prediction window. The absolute LPIPS gap to the pretrained base is largest at the shortest rollout time we sample (t=0.5 s, base mean LPIPS 0.542 vs. exocentric-only 0.460, a gap of 0.082) and narrows steadily as t grows. Restricting the LPIPS analysis to t=1.0 second of rollout widens the paired gap from the full-clip averages (+0.041, +0.058, +0.049 for ego, exo, combined) to +0.061, +0.075, and +0.068 respectively, at 82−89% paired win-rate $( p < 1 0 ^ { - 2 0 } )$ . The near-horizon window is precisely the regime that world-model literature targets for embodied use Wayve (2025).

Contributions. We make four contributions to the study of pretrained video world model adaptation:

1. We introduce RetailSMV (Retail Synchronized Multi-View), a retail video corpus of 32,105 captioned clips collected across five real-world supermarkets, with synchronized egocentric and exocentric capture of store-staf operational work (stocking, arranging, weighing, managing supply carts, scanning at checkout) rather than the customer-side behaviour captured by prior retail video corpora, dense paragraph-level captions, and pre-defined train, validation, and test splits. To our knowledge no prior dataset combines a real-world retail deployment domain with synchronized staf-perspective ego/exo capture of the same activity at this scale.

2. Using RetailSMV, we present, to our knowledge, the first controlled comparison of egocentric-only, exocentric-only, and combined LoRA configurations of a foundation video world model, isolating trainingdata viewpoint as the variable of interest while holding all other factors constant.

3. We document a reproducible headline finding: exocentric-only adaptation matches or exceeds combined adaptation on six of seven point estimates and is significantly better on LPIPS, PSNR, and DreamSim, despite using only the 15,985 exocentric training clips while combined uses the full 32,105. The combined configuration edges exo on SSIM by a small margin, and the two are statistically indistinguishable on CLIPScore. We additionally characterize the temporal structure of the adaptation gap and show that the absolute LPIPS gap to the pretrained base is largest at the shortest rollout time and narrows steadily as t grows.

4. We provide a fully paired statistical protocol for video world model evaluation and report all metrics under both parametric (t-test) and non-parametric (Wilcoxon signed-rank) tests, addressing the documented absence of paired statistical reporting in current video-generation literature.

The remainder of the paper is organized as follows. Section 2 reviews related work on foundation video world models, video generation evaluation, and parameter-eficient adaptation. Section 3 describes the multi-view retail data, the three view-stratified LoRA configurations, and the training and inference protocols. Section 4 defines the test set, the metric suite, and the paired statistical tests. Section 5 presents the main results, the temporal analysis, and metric ablations. Section 6 concludes with discussion and limitations.

## 2 Related Work

Foundation video generation models. Text-to-video and image-to-video difusion models have advanced rapidly. Make-A-Video Singer et al. (2023) and AnimateDif Guo et al. (2024) demonstrated that motion priors can be learned without paired text-video data; DynamiCrafter Xing et al. (2024) and VideoCrafter2 Chen et al. (2024) extended these to open-domain image-to-video conditioning; CogVideoX Yang et al. (2025), Hunyuan-Video Kong et al. (2024), Lumiere Bar-Tal et al. (2024), Movie Gen Meta GenAI (2024), Open-Sora Zheng et al. (2024), Stable Video Difusion Blattmann et al. (2023), Sora OpenAI (2024), and VideoPoet Kondratyuk et al. (2024) pushed scale, fidelity, and duration. The NVIDIA Cosmos family of world foundation models NVIDIA (2025) targets embodied AI specifically and is the basis of our adaptation experiments. The underlying training objectives include denoising difusion Ho et al. (2020); Song et al. (2021), flow matching Lipman et al. (2023), and rectified flow Liu et al. (2023), with UniPC Zhao et al. (2023) now a common deterministic sampler at inference. Almost all of these models are pretrained on internet-scale generic video and report no domain-specific evaluation in retail environments.

World models for embodied simulation. The framing of pretrained video models as world simulators for embodied agents traces to the world-models lineage of Ha and Schmidhuber Ha and Schmidhuber (2018) and is now exemplified by DreamerV3 Hafner et al. (2023), Genie Bruce et al. (2024), and the learningan-interactive-simulator framework of UniSim Yang et al. (2024), as well as more recent surveys Gao et al. (2025). Domain-specialized driving world models add explicit action conditioning: GAIA-1 Hu et al. (2023), GAIA-2 Wayve (2025), Vista Gao et al. (2024), and DriveDreamer Wang et al. (2024a). Embodied robotics has produced action-conditioned generative policies that consume or emit video: RT-2 Brohan et al. (2023), PaLM-E Driess et al. (2023), GR00T NVIDIA (2025c), π Black et al. (2024), AgiBot Contributors (2025), and the Gemini Robotics family Gemini-Robotics-ER-1.5 (2025), supported by retail-adjacent benchmarks such as RoboVQA Sermanet et al. (2023). Outside driving, however, systematic studies of adapting a foundation video model to a specific deployment environment are scarce.

Parameter-efficient adaptation of diffusion models. Adapting a large pretrained difusion model to a new domain is typically done through parameter-eficient fine-tuning. The dominant family of methods comprises Low-Rank Adaptation Hu et al. (2022) and its variants Zhang et al. (2023); Dettmers et al. (2023); subject-driven personalization with DreamBooth Ruiz et al. (2023); word-level personalization via textual inversion Gal et al. (2023); and multi-concept composition Kumari et al. (2023). For video specifically, motion-customization methods have started to appear: MotionDirector Zhao et al. (2024), MotionCtrl Wang et al. (2024b), and AnimateDif Guo et al. (2024) all adopt LoRA-style adapters to inject motion or domain priors into a frozen image- or video-difusion backbone. NVIDIA’s Cosmos-Predict 2.5 cookbook NVIDIA (2025) provides documented LoRA post-training recipes. Most prior video LoRA work treats viewpoint as a nuisance variable or focuses on a single viewpoint; to our knowledge there is no prior systematic comparison of egocentric, exocentric, and combined LoRA configurations on a synchronized multi-view video corpus.

Video generation evaluation. The standard suite is built around the Fréchet video distance (FVD) Unterthiner et al. (2018) on I3D Kinetics features Carreira and Zisserman (2017), per-frame LPIPS Zhang et al. (2018) with AlexNet features Radford et al. (2021), PSNR and SSIM Wang et al. (2004), and CLIPScore Hessel et al. (2021) computed from CLIP image-text embeddings Radford et al. (2021). VBench Huang et al. (2024) and VBench-2 Huang et al. (2025) provide curated capability dimensions for generic video generation. These benchmarks were designed assuming large evaluation sets; FVD is documented to require large samples for stable covariance estimation Luo et al. (2024), a regime rarely available in domain-specific applications. Modern alternatives respond to this gap. DreamSim Fu et al. (2023) combines CLIP, OpenCLIP, and DINO Caron et al. (2021) features with human-similarity fine-tuning, producing a perceptual metric that outperforms LPIPS at semantic structure and object identity. JEDi Luo et al. (2024) replaces I3D-FVD with an MMD on V-JEPA features Bardes et al. (2024), reaching a stable estimate at far smaller sample counts. We adopt DreamSim and JEDi alongside the classical suite, use the Hessel formulation of CLIPScore Hessel et al. (2021), and use an R3D-Fréchet variant of the Kinetics-based Fréchet distance with the ResNet-3D-18 backbone Tran et al. (2018) described in section 4.2.

Egocentric and exocentric video understanding. Egocentric video research is dominated by Ego4D Grauman et al. (2022), Epic-Kitchens Damen et al. (2018), HOI4D Liu et al. (2022), and Aria Meta Reality Labs (2024), all of which skew toward kitchens, outdoor activities, and daily living scenes. EgoSchema Mangalam et al. (2023) provides a long-form egocentric video understanding benchmark. Charades-Ego Sigurdsson et al. (2018) and Ego-Exo4D Grauman et al. (2024) introduce paired ego/exo recordings and demonstrate that joint training improves cross-view action recognition and retrieval. The pretext-task literature has used arrow-of-time prediction Wei et al. (2019) and spatio-temporal jigsaw solving Doersch et al. (2015) to learn temporal structure from unlabelled video. These studies focus on recognition and representation learning; we focus on generation, where the question is qualitatively diferent: does adding ego data to exo data improve generation of new retail scenes, or does it dilute the learning signal? Our experiments show that on retail video the empirical answer is that combined ego+exo training does not improve distributional or semantic metrics over exo-only training: exocentric-only adaptation matches or beats combined adaptation on R3D-Fréchet and DreamSim. Table 1 summarizes the gap in publicly available foundation video models with respect to retail coverage and synchronized multi-view data.

Table 1 Domain coverage of recent foundation video and world models. Existing models are pretrained on entertainment video, driving footage, or generic web video, with no documented retail-specific training or evaluation and no documented use of synchronized egocentric/exocentric capture of the same activity in their disclosed pretraining recipes. Pretraining corpora are not always fully disclosed; the table reflects what is documented in each model’s public technical report. We adapt Cosmos3-Nano on RetailSMV (section 3.1), which provides 32,105 captioned clips with synchronized paired views, enabling the controlled view-stratified study reported in this paper.

<table><tr><td>Model</td><td>Type</td><td>Pretraining domain</td><td>Retail</td><td>Ego+Exo pairs</td></tr><tr><td>Sora OpenAI (2024)</td><td>Text-to-video diffusion</td><td>Internet video, entertainment</td><td>×</td><td>×</td></tr><tr><td>Stable Video Diffusion Blattmann et al. (2023)</td><td>Image-to-video diffusion</td><td>LAION+curated video</td><td>×</td><td>×</td></tr><tr><td>Open-Sora Zheng et al. (2024)</td><td>Text-to-video diffusion</td><td>Open web video</td><td>×</td><td>×</td></tr><tr><td>Movie Gen Meta GenAI (2024)</td><td>Text-to-video diffusion</td><td>Licensed entertainment video</td><td>×</td><td>×</td></tr><tr><td>GAIA-1 Hu et al. (2023)</td><td>Action-conditioned WM</td><td>4,700 h driving</td><td>×</td><td>×</td></tr><tr><td>GAIA-2 Wayve (2025)</td><td>Action-conditioned WM, MV</td><td>16,000 h driving, multi-camera</td><td>×</td><td>×</td></tr><tr><td>Vista Gao et al. (2024)</td><td>Action-conditioned WM</td><td>nuScenes, Waymo</td><td>×</td><td>×</td></tr><tr><td>Cosmos3-Nano NVIDIA (2025)</td><td>Foundation video DiT (16B)</td><td>Internet+robot manipulation</td><td>×</td><td>×</td></tr><tr><td>Cosmos-Predict 2.5 NVIDIA (2025)</td><td>Foundation video DiT (2B/14B)</td><td>Internet+robot manipulation</td><td>×</td><td>×</td></tr><tr><td>This work, RetailSMV (section 3.1)</td><td>(corpus)</td><td>5 supermarkets, real store-staff operations</td><td>√</td><td>√</td></tr></table>

Retail-specific video datasets and benchmarks. Retail and shopping environments have been studied as a deployment domain across recognition, simulation, and language-grounded tasks. MERL Shopping Singh et al. (2016) provides surveillance-style exocentric clips of retail customer interactions; the retail-action benchmark Mazzini et al. (2025) and retail-vision RetailVision Organizers (2020–2025) survey extend this to fine-grained action recognition; SariBench Gajo et al. (2025) provides a simulated retail environment for embodied agents; PRISM Rouhi et al. (2026) is a recent multi-view multi-capability retail video dataset built for embodied vision-language models, with chain-of-thought supervision and an SFT-ready format. These works focus on recognition, question answering, or simulation, not on video generation; in particular none of them has been used to fine-tune or evaluate a foundation video difusion model. They also focus on the customer side of the retail scene (browsing, picking, paying). RetailSMV (this paper) is, to our knowledge, the first retail video corpus with synchronized ego/exo capture of the store-staf operational viewpoint—stocking, arranging, weighing, managing supply carts, and scanning at the checkout—at a scale that supports finetuning of a foundation video world model (table 2). RetailSMV is distinct from PRISM: PRISM targets embodied vision-language supervision, whereas RetailSMV is organized for video-world-model adaptation with paired ego/exo clips, generation captions, adaptation splits, and generated-video evaluation protocols.

Statistical reporting in video generation. A practical observation in recent video-generation literature Huang et al. (2024); Wayve (2025); Fu et al. (2023) is that paired statistical tests, win-rates, and confidence intervals are rarely reported, even though metric standard deviations are often comparable in magnitude to the diferences between methods. The general statistical-comparison framework of Demšar Demšar (2006) predates the difusion-model era but its core advice (report paired tests with both parametric and non-parametric variants) remains directly applicable. We adopt this protocol throughout: every per-clip metric is reported with mean improvement, paired win-rate, and both parametric (t-test) and non-parametric (Wilcoxon signed-rank) p-values.

## 3 Method

We describe the RetailSMV corpus introduced with this paper (section 3.1), the parameter-eficient adaptation recipe applied to the pretrained foundation model (section 3.2), the three view-stratified configurations whose comparison forms the core of our study (section 3.3), and the inference protocol used to generate evaluation videos (section 3.4).

Table 2 Positioning of RetailSMV against related embodied video datasets. RetailSMV is, to our knowledge, the only embodied video dataset that combines a real-world retail deployment domain, synchronized egocentric and exocentric capture of the same activity, and dense paragraph-level captioning at a scale that supports finetuning of a foundation video difusion model. “Sync ego+exo” indicates that ego and exo recordings of the same activity at the same wall-clock time are available and identified at the clip level (not merely that both modalities exist in the corpus). “Paragraph caption” indicates a single dense natural-language paragraph per clip suitable as text conditioning for a text-and-image-to-video model. <sup>†</sup>Ego-Exo4D pairs ego and exo views but is concentrated on skilled-activity domains (sports, music, cooking) rather than retail.

<table><tr><td>Dataset</td><td>Domain</td><td>Viewpoints</td><td>Sync ego+exo</td><td>Retail</td><td>Paragraph caption</td><td>Scale (clips/hours)</td></tr><tr><td>Ego4D Grauman et al. (2022)</td><td>Daily living, general</td><td>Ego</td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td><td>3,670 h</td></tr><tr><td>Ego-Exo4D Grauman et al. (2024)</td><td>Skilled activities $^{\dagger}$ </td><td>Ego + Exo</td><td> $\checkmark$ </td><td> $\times$ </td><td> $\times$ </td><td>1,286 h</td></tr><tr><td>Charades-Ego Sigurdsson et al. (2018)</td><td>Daily living</td><td>Ego + Exo</td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td><td>68.8 h</td></tr><tr><td>Epic-Kitchens Damen et al. (2018)</td><td>Kitchen</td><td>Ego</td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td><td>100 h</td></tr><tr><td>Aria Everyday Activities Meta Reality Labs (2024)</td><td>Daily living</td><td>Ego</td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td><td>—</td></tr><tr><td>RoboVQA Sermanet et al. (2023)</td><td>Office robot manipulation</td><td>Exo</td><td> $\times$ </td><td> $\times$ </td><td> $\times$ </td><td>829,000 pairs</td></tr><tr><td>SariBench Gajo et al. (2025)</td><td>Retail (simulator)</td><td>Ego (simulated)</td><td> $\times$ </td><td> $\checkmark$ </td><td> $\times$ </td><td>100 demos</td></tr><tr><td>RetailSMV (ours)</td><td>Real retail (5 supermarkets)</td><td>Ego + fixed Exo</td><td> $\checkmark$ </td><td> $\checkmark$ </td><td> $\checkmark$ </td><td>32,105 clips</td></tr></table>

## 3.1 The RetailSMV Corpus

We introduce RetailSMV (Retail Synchronized Multi-View), a retail video corpus of 32,105 captioned clips with synchronized egocentric and exocentric capture of the same store-staf activities across five real-world supermarkets. To our knowledge no prior dataset combines a real-world retail deployment domain with synchronized ego/exo capture of the same activity at the scale required to fine-tune a foundation video world model. Table 2 positions RetailSMV against related embodied video datasets.

Capture protocol. The corpus is collected in five operating supermarkets that span distinct store layouts, lighting conditions, aisle configurations, and product category distributions. Unlike prior retail video datasets which focus on customer-side behaviour, RetailSMV is captured from the perspective of store staf performing their day-to-day operational work: stocking shelves, arranging produce, weighing and labelling items, carrying crates, pushing supply carts, measuring shelf space, scanning items at the checkout, and assisting customers. Each work episode is recorded simultaneously by two complementary sensors. The egocentric sensor is a head-mounted camera worn by the staf member while they carry out these tasks. The exocentric sensor is a fixed scene-level camera that observes the same activity from a stable third-person perspective. Synchronized capture means that for every ego clip there exists an exo clip of the identical activity at the same wall-clock time, with the same person, products, and scene context, captured from the complementary viewpoint. This pairing is what enables the controlled view ablations reported in this paper.

Clip-level selection and captioning. From the raw recordings we extract 32,105 short clips (1–8 s each) that satisfy three conditions: (i) the clip captures a complete atomic store-staf action, (ii) the synchronized view from the paired sensor is available for the same time interval, and (iii) a dense paragraph-level caption is available. Each caption is a single paragraph describing the visible store and section, the person’s high-level task and low-level action, hand states and body pose, visible products and signage, and a frame-by-frame motion summary. The same caption is used as the text-conditioning input for every configuration of every training run, eliminating prompt variance as a source of cross-configuration diference. The full clip pool decomposes into 16,120 egocentric and 15,985 exocentric clips, an essentially balanced split that lets us match clip count exactly across single-view configurations.

Privacy. All faces are de-identified through Gaussian blurring at the source. This is applied uniformly to both ego and exo views and to the train, validation, and test splits; we do not introduce or remove any face obfuscation in this work.

Train, validation, and test splits. We partition RetailSMV into three pairwise disjoint splits by per-clip unique identifier. The training set consists of 32,105 clips and is used for LoRA adaptation. The validation set consists of 1,388 clips and is used for adapter selection: the rectified-flow validation loss is computed every 100 training steps on n=32 paired samples drawn from it. The test set consists of 200 clips drawn via stratified sampling from the held-out evaluation pool, balanced across both egocentric (n=100) and exocentric (n=100) viewpoints so that both deployment scenarios are represented at equal weight in the per-clip paired statistics. The egocentric-only model is evaluated on its in-distribution view as well as on the exocentric view, exposing cross-view transfer behavior. Table 3 summarizes the splits.

![](images/92b7ebf342b0738b757f7338126078f914432b6035baabf811c70685e30ed8b8.jpg)  
Figure 2 Pipeline overview. The RetailSMV corpus (section 3.1) provides synchronized egocentric and exocentric video across five real-world supermarkets, yielding 16,120 ego and 15,985 exo clips. Three matched LoRA configurations of the pretrained Cosmos3-Nano foundation video model NVIDIA (2025) (egocentric-only, exocentric-only, and combined) are trained under identical hyperparameters and optimization budget, isolating training-data viewpoint as the variable of interest. Every configuration is evaluated on the same 200-clip stratified held-out test set under a paired statistical protocol.

Table 3 Splits used in this study. Training, validation (used for adapter selection), and test (used for final evaluation) are pairwise disjoint by per-clip unique identifier.

<table><tr><td>Split</td><td>Ego</td><td>Exo</td><td>Total</td></tr><tr><td>Training</td><td>16,120</td><td>15,985</td><td>32,105</td></tr><tr><td>Validation (adapter selection)</td><td>727</td><td>661</td><td>1,388</td></tr><tr><td>Test (final evaluation)</td><td>100</td><td>100</td><td>200</td></tr></table>

## 3.2 LoRA Adaptation of the Foundation Model

We adapt the open-source Cosmos3-Nano video difusion model NVIDIA (2025), a 16-billion-parameter Mixture-of-Transformers Difusion Transformer with a Wan2.2 video VAE and the Cosmos-Reason NVIDIA (2025a,b) text encoder, using Low-Rank Adaptation Hu et al. (2022).

LoRA configuration. We attach LoRA adapters Hu et al. (2022) of rank r=32 and scaling α=64 to the cross-attention projections and the Mixture-of-Experts MLPs in every generative-modality transformer block, training ∼ 0.55% of the base model’s parameters. This follows the standard adapter-pluggable convention now widely used for difusion-model personalization Ruiz et al. (2023); Gal et al. (2023); Kumari et al. (2023); Zhang et al. (2023) and motion customization Zhao et al. (2024); Wang et al. (2024b); Guo et al. (2024). Standard zero-initialization of B ensures the model at step zero reproduces the pretrained base. Full layer-wise target list, exact parameter count, and AdamW Loshchilov and Hutter (2019) settings are in table 9.

Training objective. We train under the rectified-flow Liu et al. (2023) velocity-matching objective used by Cosmos3-Nano during its own pretraining, a stochastic-interpolant family closely related to denoising difusion Ho et al. (2020); Song et al. (2021) and flow matching Lipman et al. (2023). For each training step we encode the input clip to its latent representation $z _ { 0 } ,$ , draw a noise scale $\sigma \sim \mathrm { S i g m o i d } ( { \cal N } ( 0 , 1 ) )$ , sample a Gaussian noise tensor $\epsilon \sim \mathcal { N } ( 0 , \bf { I } )$ , form the perturbed latent $z _ { \sigma } = \sigma \epsilon + ( 1 { - } \sigma ) z _ { 0 }$ , and target the velocity $v _ { \sigma } = \epsilon - z _ { 0 }$ . The training loss is the mean squared error $\mathcal { L } = \| \hat { v } _ { \sigma } - v _ { \sigma } \| _ { 2 } ^ { 2 }$ averaged over the spatio-temporal latent grid.

Training protocol. We train at 480×832 spatial resolution, 81 frames per clip, for 3,000 optimization step under a cosine learning-rate schedule with a peak rate of $3 \times 1 0 ^ { - 4 }$ . Validation difusion loss is computed every 100 steps on n=32 paired samples drawn from the 1,388-clip validation set, and we select the iteration-1500 adapter (lowest validation loss) for all subsequent generation experiments. The full training environment (precision, video decoder, hardware, efective batch size, and other infrastructure details) is in section A.2

and table 9.

## 3.3 View-Stratified Configurations

To isolate the efect of training-data viewpoint while holding all other variables constant, we train three matched configurations under identical hyperparameters and optimization budget:

Combined (ego + exo). The reference configuration, trained on the full 32,105-clip mixture. This corresponds to the natural default of using all available data.

Egocentric-only. Trained on the 16,120 egocentric clips alone. This configuration can specialize to first-person motion patterns and hand-object interaction, at the cost of losing exposure to wide-field scene structure.

Exocentric-only. Trained on the 15,985 exocentric clips alone. This configuration can specialize to scene layout, multi-actor dynamics, and the wide field of view of the fixed scene camera, at the cost of losing exposure to fine-grained hand interaction.

Combined, second seed. An additional combined configuration trained with a diferent random seed (137 instead of 42), used to bound the seed-induced variance on the validation metrics and to confirm that configuration-to-configuration diferences exceed this variance.

## 3.4 Inference Protocol

For evaluation we generate one video per (configuration, test clip) pair under image-to-video conditioning: the first frame of the ground-truth test clip serves as the conditioning image, and the dense caption for that clip serves as the text prompt. We generate 189 frames at 480×832, the native maximum length of Cosmos3-Nano (∼ 7.9 s at 24 FPS), so that wall-clock time-aligned metric sampling is well defined for the full duration of every clip. The deterministic UniPC multistep sampler Zhao et al. (2023) is used at inference. Scheduler, inference steps, guidance scale, negative prompt, and seed are listed in section A.3. The trained adapter is merged into the base weights at inference; we report a numerical merge verification in section C.3.

## 4 Evaluation Protocol

This section defines the test set (section 4.1), the metric suite (section 4.2), the paired statistical protocol (section 4.3), and the near-horizon analysis used to characterize the temporal structure of the adaptation gap (section 4.4).

## 4.1 Test Set

Due to the high inference cost of video generation (on the order of minutes per clip on contemporary hardware), we evaluate on a small held-out test set, the standard mode of evaluation in current video-generation literature. We use the same test set across all four configurations and the pretrained baseline, generated under identical inference settings (section 3.4).

The test set consists of 200 retail clips drawn from a held-out evaluation pool via stratified sampling, balanced across both egocentric (n=100) and exocentric (n=100) viewpoints. The draw is deterministic and uses rejection sampling against the validation set used for adapter selection. We verify by per-clip unique-identifier intersection that the test set is disjoint from both the training data and the adapter-selection validation set.

Asset pipeline. For each test clip we extract the full-length ground-truth video, the first frame as a JPEG image, and the dense caption. We then generate exactly one video per (configuration, test clip) pair, for a total of 4×200 = 800 generated videos. All generations use the deterministic settings of section 3.4.

## 4.2 Metric Suite

We report a deliberately wide panel of seven metrics to avoid over-interpreting any single number. All metrics are computed on the same generated videos. The panel comprises per-frame perceptual metrics (LPIPS, PSNR, SSIM), two distributional metrics (a classical Kinetics-feature Fréchet distance and a recent V-JEPA-based

MMD), a semantic alignment metric (Hessel CLIPScore), and a modern human-aligned perceptual metric (DreamSim). We report JEDi as a recent distributional video-quality metric complementary to R3D-Fréchet; we do not rely on JEDi alone, and the same ranking is supported by R3D-Fréchet and DreamSim.

LPIPS. Zhang et al. (2018) measures perceptual distance between matched generated/ground-truth frame pairs using AlexNet features. We sample 16 evenly-spaced wall-clock timestamps in [0, min(gen\_dur, gt\_dur)], dropping t=0 to remove the trivial first-frame match introduced by image-to-video conditioning. Lower is better. The rationale for wall-clock sampling (avoiding a frame-index pitfall between 24 FPS generations and lower-FPS ground truths) is detailed in section C.2.

PSNR and SSIM. Wang et al. (2004) are computed per frame on the same time-aligned samples used for LPIPS and averaged over the clip. Higher is better.

Hessel CLIPScore. Hessel et al. (2021) measures caption-video alignment. For each of 8 evenly-spaced frames of the generated video, we compute 2.5 · max(0, cos(img\_emb, txt\_emb)) with CLIP ViT-B/32 (Hessel et al.’s convention), and average across frames. Higher is better; typical values fall in the 0.6–0.85 range.

R3D-Fréchet video distance. We compute a Fréchet distance on features from the ResNet-3D-18 video classifier (torchvision.models.video.r3d\_18 with KINETICS400\_V1 weights) on 16 frames per clip, used in place of the canonical I3D Unterthiner et al. (2018); Carreira and Zisserman (2017). We call the resulting metric R3D-Fréchet to make the backbone substitution explicit; absolute values are not directly comparable to I3D-FVD values in the broader literature. Lower is better. See section C for the rationale and a recipe for recomputing with the canonical I3D backbone

DreamSim. Fu et al. (2023) is a perceptual distance built from an ensemble of CLIP, OpenCLIP, and DINO features fine-tuned on human similarity triplets. It is documented to outperform LPIPS, raw CLIP, and DINO on human similarity benchmarks. We compute it on the same 16 time-aligned frames used for LPIPS, averaged over the clip. Lower is better.

JEDi. Luo et al. (2024) is a recent distributional video-quality metric that replaces the I3D-FVD covariance estimate with a maximum mean discrepancy on V-JEPA Bardes et al. (2024) ViT-H/16 features. It is documented to reach a stable estimate at far smaller sample counts than I3D-FVD. We compute JEDi on 16 uniformly-sampled frames per clip at 224×224, using the oficial videojedi package with the V-JEPA ViT-H/16 backbone, with the gold (ground-truth) loader and the generated loader containing the same set of clips. Lower is better. The reported JEDi value is a biased empirical polynomial-kernel MMD estimator and its absolute scale is sample-set dependent (we verified this directly by recomputing JEDi on disjoint sub-pools of our test set); the relative reduction vs. the pretrained base, and the configuration ranking, are the reportable quantities. We report JEDi as a complementary distributional signal to R3D-Fréchet; we do not rely on it alone, and the configuration ranking is consistent with R3D-Fréchet and DreamSim.

Validation diffusion loss. For each configuration we additionally compute the rectified-flow training loss on a paired-noise evaluation set of n=200 random samples from the validation set used for adapter selection (not from the held-out test set). To ensure exact pairing across configurations, the noise tensor and noise scale for sample i are derived from a fixed seed 10000+i shared by all configurations and the base. This produces per-sample loss values that are directly comparable across configurations and forms the basis of table 4.

## 4.3 Paired Statistical Protocol

For every per-clip metric and every LoRA configuration c we compute, for each test clip i, the signed paired diference

$$
\delta_ {i} (c) = \left\{ \begin{array}{l l} m _ {i} (\text {base}) - m _ {i} (c), & m \text {lower - is - better} \\ m _ {i} (c) - m _ {i} (\text {base}), & m \text {higher - is - better} \end{array} \right.
$$

so that $\delta _ { i } ( c ) > 0$ means c beats the base on clip i. We then report (i) the mean paired diference $\bar { \delta } ( c )$ , (ii) the paired win-rate $| \{ i : \delta _ { i } ( c ) > 0 \} | / n$ , (iii) a parametric two-sided paired t-test p-value, and (iv) a non-parametric Wilcoxon signed-rank p-value. We treat $p { < } 1 0 ^ { - 4 }$ as highly significant. For the distributional R3D-Fréchet metric we report the per-configuration scalar score and the relative improvement, since it operates on the set of videos as a whole rather than on paired samples.

Table 4 Unified validation difusion loss on 200 paired samples drawn from the validation set used for adapter selection (no overlap with the held-out test set). The noise tensor and noise scale for sample i are identical across all rows, so the per-sample values are directly comparable. $^ { 6 6 } \mathrm { W i n } ^ { \mathfrak { Y } }$ is the paired win-rate over the pretrained base.

<table><tr><td>Configuration</td><td>Loss ↓</td><td>Δ vs. base</td><td>Win</td><td>p (Wilcoxon)</td></tr><tr><td>Pretrained base</td><td>1.006 ± 0.197</td><td>—</td><td>—</td><td>—</td></tr><tr><td>Egocentric-only</td><td>0.359 ± 0.144</td><td>-0.647</td><td>100%</td><td> $\ll 10^{-4}$ </td></tr><tr><td>Exocentric-only</td><td>0.363 ± 0.150</td><td>-0.643</td><td>100%</td><td> $\ll 10^{-4}$ </td></tr><tr><td>Combined (s=42)</td><td>0.355 ± 0.145</td><td>-0.651</td><td>100%</td><td> $\ll 10^{-4}$ </td></tr><tr><td>Combined (s=137)</td><td>0.357 ± 0.145</td><td>-0.649</td><td>100%</td><td> $\ll 10^{-4}$ </td></tr></table>

## 4.4 Near-Horizon LPIPS Analysis

World models are predominantly used to predict the immediate future of a scene, typically the next one to three seconds Wayve (2025). The mean LPIPS curves (figure 13) rise monotonically with t for every configuration: the absolute gap to the pretrained base is widest at the shortest rollout time we sample $\left( t { = } 0 . 5 \ : \mathrm { s } \right)$ and narrows steadily as rollout time grows, since all configurations drift away from the single ground-truth trajectory in the same way. To characterize the temporal structure of the adaptation gap, we additionally report LPIPS at fixed time stamps $t \in \{ 1 . 0 , 2 . 0 \}$ seconds on the same test clips. This is a reporting convention that we recommend for future video world model evaluations, since it isolates the regime in which the model’s prediction quality matters most for downstream embodied use.

## 5 Results and Analysis

We present the results in five subsections. Section 5.1 establishes that all three view-stratified configurations improve uniformly over the pretrained baseline on the unified validation difusion loss. Section 5.2 reports the main paired metrics on the held-out test set, with direct paired comparisons between the adapted configurations (combined vs. exocentric-only and combined vs. egocentric-only) and the asymmetric datacontribution result. Section 5.3 walks through ten representative video examples to ground the metric-leve diferences in concrete failure modes. Section 5.4 examines the temporal structure of the adaptation gap and identifies the near-horizon prediction window as the regime in which the LoRA-adapted models difer most from the pretrained base. Section 5.5 reports time-aligned-vs-index-based sampling, adapter-merge verification, and modern-metric sensitivity. All p-values are computed under the paired protocol of section 4.3.

## 5.1 Adaptation Improves Validation Loss Uniformly

Table 4 reports the rectified-flow validation loss on 200 paired samples drawn from the validation set used for adapter selection. By construction, the noise tensor and noise scale for sample i are identical across all configurations, so the per-sample loss values are directly comparable.

Three observations follow. First, adaptation reduces validation loss by a factor of approximately 2.8, and every one of the 200 paired evaluation samples improves over the pretrained base under each configuration (perfect paired ordering under the Wilcoxon signed-rank test; exact p-values are reported in section B.1). Second, the seed-induced variation between the two combined seeds is 0.002, smaller than the configuration-to-configuration spread of 0.008 across the four LoRA rows; configuration-to-configuration ordering therefore reflects properties of the training data rather than optimization noise. Third, the three view-stratified configurations are within 0.008 of each other on this metric, which means validation loss alone does not discriminate between view-stratified data choices. Diferences across configurations emerge when generation quality is evaluated directly.

## 5.2 Main Paired Metrics

Table 5 reports the seven primary metrics on the held-out 200-clip test set. The exocentric-only configuration achieves the lowest R3D-Fréchet (35.14 vs. base 42.69, a 17.7% relative reduction), the lowest JEDi (0.829 vs. base 1.246, a 33.5% relative reduction), the largest LPIPS reduction of 0.058 at 100% paired win-rate $( p < 1 0 ^ { - 3 4 } )$ , the largest PSNR improvement of +0.575 dB at 85% paired win-rate, and the largest DreamSim reduction of 0.062 at 92.5% paired win-rate $( p { < } 1 0 ^ { - 3 1 } )$ . The combined configuration is tied with exo on CLIPScore and edges exo on SSIM by 0.0012. Hessel CLIPScore improves under all three configurations with very strong paired significance $( p < 1 0 ^ { - 1 0 } )$ . We report JEDi as a recent distributional video-quality metric complementary to R3D-Fréchet: we do not rely on JEDi alone, and the same ranking is supported by R3D-Fréchet and DreamSim.

Table 5 Main results on the held-out 200-clip test set. For lower-is-better metrics (LPIPS, DreamSim, R3D-Fréchet, JEDi) the best value in each column is in bold; for higher-is-better metrics (PSNR, SSIM, CLIPScore) the largest value is in bold. CLIPScore is in the Hessel 2.5 · max(0, cos) formulation. Paired p-values are Wilcoxon signed-rank against the pretrained base. R3D-Fréchet and JEDi are set-level distributional metrics and admit no paired test; their relative improvement vs. base is reported parenthetically.

<table><tr><td>Configuration</td><td>LPIPS ↓</td><td>PSNR ↑</td><td>SSIM ↑</td><td>CLIPScore ↑</td><td>DreamSim ↓</td><td>R3D-Fr. ↓</td><td>JEDi ↓</td></tr><tr><td colspan="8">Per-clip mean (where applicable)</td></tr><tr><td>Pretrained base</td><td>0.668</td><td>10.68</td><td>0.215</td><td>0.815</td><td>0.263</td><td>42.69</td><td>1.246</td></tr><tr><td>Egocentric-only</td><td>0.626</td><td>11.01</td><td>0.226</td><td>0.832</td><td>0.220</td><td>37.65 (-11.8%)</td><td>0.960 (-22.9%)</td></tr><tr><td>Exocentric-only</td><td>0.610</td><td>11.26</td><td>0.235</td><td>0.834</td><td>0.201</td><td>35.14 (-17.7%)</td><td>0.829 (-33.5%)</td></tr><tr><td>Combined</td><td>0.619</td><td>11.18</td><td>0.236</td><td>0.832</td><td>0.216</td><td>36.20 (-15.2%)</td><td>0.892 (-28.5%)</td></tr><tr><td colspan="8">Paired vs. base: mean improvement / win-rate / Wilcoxon p</td></tr><tr><td>Egocentric-only</td><td>+0.041/90.5%/&lt; $10^{-27}$ </td><td>+0.33/72.0%/&lt; $10^{-12}$ </td><td>+0.011/67.0%/&lt; $10^{-8}$ </td><td>+0.017/72.5%/&lt; $10^{-10}$ </td><td>+0.043/78.0%/&lt; $10^{-18}$ </td><td>—</td><td>—</td></tr><tr><td>Exocentric-only</td><td>+0.058/100%/&lt; $10^{-34}$ </td><td>+0.58/85.0%/&lt; $10^{-25}$ </td><td>+0.021/76.0%/&lt; $10^{-17}$ </td><td>+0.018/75.0%/&lt; $10^{-13}$ </td><td>+0.062/92.5%/&lt; $10^{-31}$ </td><td>—</td><td>—</td></tr><tr><td>Combined</td><td>+0.049/94.5%/&lt; $10^{-31}$ </td><td>+0.50/78.5%/&lt; $10^{-18}$ </td><td>+0.022/80.0%/&lt; $10^{-20}$ </td><td>+0.017/71.0%/&lt; $10^{-10}$ </td><td>+0.046/82.0%/&lt; $10^{-23}$ </td><td>—</td><td>—</td></tr></table>

Headline finding: exocentric-only adaptation matches or exceeds combined adaptation on six of seven point estimates and is significantly better on LPIPS, PSNR, and DreamSim. The exocentric-only adapter, trained on 15,985 exocentric clips, achieves an R3D-Fréchet of 35.14, a 17.7% relative reduction from the base, and a JEDi of 0.829, a 33.5% relative reduction (the largest distributional gap of any configuration); the combined adapter, with access to all 32,105 clips, achieves 36.20 R3D-Fr. (−15.2%) and 0.892 JEDi (−28.5%). On LPIPS the exocentric-only adapter produces the largest paired improvement of +0.058 at 100% paired win-rate (every one of 200 test clips); combined is at +0.049 at 94.5%. On PSNR exo improves by +0.58 dB at 85.0% win-rate versus combined’s +0.50 dB at 78.5%. On DreamSim exo achieves a reduction of 0.062 at 92.5% win-rate $\left( p < 1 0 ^ { - 3 1 } \right)$ versus combined’s reduction of 0.046 at 82.0%. The exocentric-only point estimate matches or exceeds the combined point estimate on six of the seven metrics we report (LPIPS, PSNR, CLIPScore, DreamSim, R3D-Fréchet, and JEDi); combined edges exo on SSIM only, by 0.0012. Of the five per-clip metrics for which a paired test is defined, exo is significantly better than combined on LPIPS, PSNR, and DreamSim (table 6), and the two configurations are statistically indistinguishable on SSIM and CLIPScore. The two distributional metrics agree on the ranking (exo < combined < ego < base), supporting the conclusion under either feature backbone.

Direct paired comparison: combined vs. exocentric-only. The paired Wilcoxon tests in table 5 are against the pretrained base. To support the headline claim that combined ego+exo training does not improve over exo-only, we additionally compute paired tests directly between the two adapted configurations on the same 200-clip test set. Table 6 reports the mean paired diference (combined − exocentric-only), 95% bootstrap confidence interval over clips, combined win-rate, and Wilcoxon p-value for every per-clip metric. Exo is significantly better than combined on LPIPS, PSNR, and DreamSim. Combined and exo are statistically indistinguishable on SSIM and CLIPScore (the 95% CI for SSIM straddles zero, and the CLIPScore CI is symmetric around zero). For the set-level R3D-Fréchet metric we cannot run a paired test (it operates on whole feature sets), but the point estimate is also in favor of exocentric-only (35.14 vs. combined 36.20).

We find no statistically significant metric on which combined adaptation outperforms exocentric-only adaptation. The picture from table 6 is consistent across metrics: on three of the five per-clip metrics (LPIPS, PSNR, DreamSim) combined is statistically worse than exo at $p \leq 0 . 0 0 3$ , with combined-wins fractions of 32–40% (below chance), and the DreamSim gap is significant at $p { < } 1 0 ^ { - 9 }$ . On SSIM and CLIPScore the two are statistically indistinguishable (the 95% CI for SSIM straddles zero and the CLIPScore CI is symmetric around zero). The data-rich combined configuration does not deliver any statistically significant per-clip-metric advantage over the exo-only adapter; the in-distribution exocentric subset carries the adaptation signal.

Direct paired comparison: combined vs. egocentric-only. For a symmetric view of the asymmetry, we additionally compute paired tests directly between the combined and egocentric-only adapters on the same

Table 6 Direct paired comparison between the combined and exocentric-only configurations on the 200-clip test set. ∆ is the per-clip mean of (combined − exocentric-only); 95% CI is by clip-level bootstrap with B=5000 replicates; the win-rate is the fraction of clips on which combined beats exo. Combined is significantly worse than exo on LPIPS, PSNR, and DreamSim; the SSIM and CLIPScore diferences are not statistically distinguishable from zero.

<table><tr><td>Metric</td><td> $\Delta$ </td><td>95% CI</td><td>Win-rate</td><td>Wilcoxon p</td></tr><tr><td>LPIPS ↓</td><td>+0.0086</td><td>[+0.0054, +0.0118]</td><td>33.0%</td><td>&lt; $10^{-7}$ </td></tr><tr><td>PSNR ↑</td><td>-0.075</td><td>[-0.126, -0.025]</td><td>40.0%</td><td>0.003</td></tr><tr><td>SSIM ↑</td><td>+0.0012</td><td>[-0.0007, +0.0031]</td><td>54.0%</td><td>0.18</td></tr><tr><td>CLIPScore ↑</td><td>-0.0015</td><td>[-0.0049, +0.0019]</td><td>44.5%</td><td>0.20</td></tr><tr><td>DreamSim ↓</td><td>+0.0154</td><td>[+0.0105, +0.0204]</td><td>32.0%</td><td>&lt; $10^{-9}$ </td></tr></table>

Table 7 Direct paired comparison between the combined and egocentric-only configurations on the 200-clip test set. ∆ is the per-clip mean of (combined − egocentric-only); 95% CI is by clip-level bootstrap with B=5000 replicates; the win-rate is the fraction of clips on which combined beats egocentric-only. Combined is significantly better than egocentric-only on LPIPS, PSNR, and SSIM; the CLIPScore and DreamSim diferences are not statistically distinguishable from zero.

<table><tr><td>Metric</td><td> $\Delta$ </td><td>95% CI</td><td>Win-rate</td><td>Wilcoxon p</td></tr><tr><td>LPIPS ↓</td><td>-0.0075</td><td>[-0.0122, -0.0032]</td><td>60.0%</td><td>0.0018</td></tr><tr><td>PSNR ↑</td><td>+0.173</td><td>[+0.103, +0.249]</td><td>67.0%</td><td>&lt; $10^{-6}$ </td></tr><tr><td>SSIM ↑</td><td>+0.0105</td><td>[+0.0078, +0.0134]</td><td>77.5%</td><td>&lt; $10^{-14}$ </td></tr><tr><td>CLIPScore ↑</td><td>+0.0001</td><td>[-0.0035, +0.0038]</td><td>51.0%</td><td>0.83</td></tr><tr><td>DreamSim ↓</td><td>-0.0033</td><td>[-0.0090, +0.0020]</td><td>51.0%</td><td>0.53</td></tr></table>

200-clip test set. Table 7 reports the mean paired diference (combined − egocentric-only), 95% bootstrap confidence interval over clips, combined win-rate, and Wilcoxon p-value for every per-clip metric. Combined is significantly better than egocentric-only on LPIPS $\scriptstyle \left( p = 0 . 0 0 2 \right)$ , PSNR $\left( p < 1 0 ^ { - 6 } \right)$ , and SSIM $( p < 1 0 ^ { - 1 4 } )$ ; the two configurations are statistically indistinguishable on CLIPScore and DreamSim. For the set-level distributional metrics, combined is also in front of egocentric-only (R3D-Fr. 36.20 vs. 37.65; JEDi 0.892 vs. 0.960).

Asymmetric data contribution: adding exocentric data to ego helps; adding egocentric data to exo hurts. Read alongside table 6, table 7 exposes a clear asymmetry. Starting from the egocentric-only adapter, adding the exocentric subset (i.e., training combined) produces statistically significant improvements on LPIPS, PSNR, and SSIM at $p \leq 0 . 0 0 1 8$ and improves the set-level distributional scores (R3D-Fréchet and JEDi). Starting from the exocentric-only adapter, however, adding the egocentric subset (again, training combined) produces statistically significant regressions on LPIPS, PSNR, and DreamSim a $p \leq 0 . 0 0 3$ . The two adapted configurations are statistically tied on CLIPScore in both directions. Combined therefore improves over egocentric-only on several metrics, but we find no statistically significant metric on which combined outperforms exocentric-only on this corpus, indicating that the adaptation signal is asymmetrically carried by the exocentric subset.

Egocentric-only adaptation transfers but ranks third. The egocentric-only configuration improves over the base on every metric at $\scriptstyle { p < 1 0 ^ { - 8 } }$ , but it ranks behind both exo-only and combined on every column of table 5. The egocentric adapter does transfer non-trivially to the mixed test viewpoint (LPIPS paired win-rate is 90.5% at $p { < } 1 0 ^ { - 2 7 } )$ but the transfer is consistently weaker than the in-distribution exocentric-only adapter, indicating that paired ego+exo capture does not eliminate the need for view-matched training data.

## 5.3 Qualitative Analysis

The metric-level diferences above translate cleanly into failure-mode patterns that are visible in the generated videos themselves. We walk through ten representative examples drawn from the held-out test set. Every figure in this subsection uses the same five-row layout: the top row is the ground-truth video (GT), and the four rows beneath show how the base, egocentric-only, exocentric-only, and combined configurations extend the shared conditioning frame over the next seven seconds. The column at t=0 is the shared conditioning frame seen by all four generated configurations, which is the same frame as the first GT frame. The first column (t=0) is therefore identical across all five rows by construction. The cell highlighted in red marks, for each example, the base configuration’s frame at which the failure is most visible (typically t=1, t=3, t=5, or t=7 s).

![](images/e126350f911a242aa982d86cb13ea521fbc5f194f44613a4bf8bf7e2fc21cab5.jpg)  
Figure 3 Hand-off watermelon. A produce-area hand-of scene. Base drifts away from the multi-person produce display and ends at t=7 s with the person holding a watermelon in a single hand in the wrong section of the store. All three adapted configurations preserve the produce context and a two-handed grip consistent with the prompt.

Hand-off watermelon. Figure 3 shows the conditioning frame of two people at a produce display, where the person is set up to receive a watermelon and hold it with both hands. The base configuration drifts away from the produce area within a few seconds and ends, at t=7 s, with a single person holding the watermelon awkwardly in one hand in what looks like an unrelated chocolate aisle. The egocentric-only, exocentric-only, and combined configurations all preserve the multi-person produce context and a two-handed grip on the watermelon consistent with the prompt. The contrast is consistent with the metric-level finding that base lacks the in-distribution retail priors that the adapted configurations acquire.

Weighing the tomato crate. Figure 4 shows a weighing-station scene where the person is supposed to lift a blue crate of tomatoes and place it onto a floor scale. Base never produces the bend-and-place motion required by the prompt and ends with the person positioned on (or directly above) the scale rather than the crate. The egocentric-only and exocentric-only configurations both execute a recognizable squat-to-scale motion, and the combined configuration produces the most explicit final frame in which the crate is in contact with the scale. The base failure here is one of action grounding rather than scene quality, and is exactly the type of error that the per-pixel reconstruction metrics fail to penalize but that the semantic and distributional metrics in tables 5 and 6 pick up.

Opening the fridge. Figure 5 shows a refrigerator-side scene where the prompt asks the person to open the door and pick a vegetable from inside. Base produces an explicit geometry violation at t=5 s in which the fridge door appears to pass through the person’s body. The adapted configurations preserve the open-door geometry and a plausible reach into the fridge. This is a low-level structural error in the base model that the adapted models avoid; it is the kind of mistake that disqualifies a world model for downstream embodied use, regardless of pixel-level similarity.

Open the freezer, take ice cream. Figure 6 shows a freezer-aisle scene where the person opens a freezer cabinet to take out an ice-cream tub. Base distorts the freezer structure as the rollout progresses, producing a final frame in which the freezer has lost its rectilinear geometry. The adapted configurations preserve the freezer cabinet structure throughout the rollout, and exo-only produces the most natural-looking reach into the freezer. This example also illustrates the headline finding numerically reported in table 6: exo-only and combined produce similar-quality outputs, but the small SSIM advantage of combined does not capture the additional semantic plausibility of exo-only’s reach geometry.

![](images/e36722d0061540a997dbcded5834eea3c686d347ece4fff605009c889c97a34b.jpg)  
Figure 4 Weighing a tomato crate. The prompt directs the person to place a blue tomato crate onto a floor weighing scale. Base never executes the bend-and-place action and ends with the person’s body, rather than the crate, positioned on the scale. All three adapted configurations cleanly perform the squat-to-scale motion with the crate as the object that lands on the scale.

Carry the crate. Figure 7 shows the person walking down an aisle carrying a yellow crate. Base produces a hand–object interpenetration at t=5 s in which the person’s hands appear to pass through the crate while walking. All three adapted configurations preserve a plausible two-hand carry posture in which the hands remain on the crate handles. As in the fridge example above, this is a hand–object physics failure that the per-pixel reconstruction metrics fail to penalize but that is unmistakable to a human observer.

Pick up the metal scoop. Figure 8 shows the person standing at the bulk-grain bins, ready to take a metal scoop and use it to fill a plastic bag with grain. Base produces a hard agent–object grasp failure at t=1 s in which the scoop floats upward on its own with no hand around it. The egocentric-only, exocentric-only, and combined configurations all correctly close the person’s hand around the scoop and lift it as a single rigid body. This is a hand–object physics failure in the earliest stage of the rollout that disqualifies the base output for downstream embodied use regardless of pixel-level similarity, and is exactly the type of error that the per-pixel reconstruction metrics under-penalize.

Push the cart down the aisle. Figure 9 shows a cart-navigation scene in which the person pushes a shopping cart loaded with small bottles down a backend aisle and then reaches into the cart with the right hand. The base configuration inserts a spurious opening in the back wall of the aisle at t=3 s, exposing an unrelated section of the store that was not visible in the conditioning frame; this is a scene-structure insertion failure rather than a pure pixel-level error. The three adapted configurations all preserve the original aisle geometry and the product layout on the side shelves throughout the rollout. The exocentric-only configuration produces the most stable scene up to t=7 s, consistent with the headline finding that exocentric-only adaptation specialises to wide-field scene structure.

![](images/0defaff72c71da1d89a973bbd48c4cefcf78d8fb208ac2544a2fd7ec59c6f9af.jpg)  
Figure 5 Opening a fridge to pick a vegetable. Base produces a geometry-violating frame at t=5 s in which the fridge door appears to pass through the person’s body. The three adapted configurations preserve the open-door geometry and a plausible reach for the vegetable inside the fridge.

Reach for the back-row bottle. Figure 10 shows a beverage-cooler scene where the person reaches for a black bottle at the back of the shelf, picks it up, and places it in the front row. Base produces a hand-pose failure at t=3 s in which the hand wraps the bottle in an unnatural configuration with no clear contact between fingers and bottle body. The egocentric-only, exocentric-only, and combined configurations all produce a recognisable back-row reach-and-grasp consistent with the prompt. This is a fine-grained hand-pose failure that is invisible to scene-level distributional metrics but immediately apparent to a human observer.

Carry the basket to the dairy section. Figure 11 shows the person walking through the store aisles carrying a blue shopping basket with both hands, and approaching the dairy-section refrigerators. Base produces a hand–object interpenetration failure at t=3 s and again at t=7 s: the person’s fingers appear to enter the inside volume of the basket rather than wrapping around the handles. The egocentric-only, exocentric-only, and combined configurations all preserve a plausible two-hand basket carry posture throughout the rollout. This is the same class of physics failure as the crate example above, replicated on a diferent agent–object pair, and it is consistently absent from all three adapted configurations.

Inspect the cleaning-supplies bottle. Figure 12 shows an inspection scene in which the person pushes a cart down the aisle, stops, and reaches with the right hand to inspect a red bottle of cleaning supplies on the shelf. Base produces a contact-physics artifact at t=7 s in which the reaching hand penetrates the shelf guard rail rather than respecting it as a solid surface. The egocentric-only, exocentric-only, and combined configurations all preserve correct hand–shelf contact and produce a recognisable inspection pose. Together with the crate, basket, and fridge examples, this completes a family of four hand–surface and hand–object contact-physics failures in the base model, all of which the adapted configurations consistently avoid.

## 5.4 Temporal Structure of the Adaptation Gap

Figure 13 plots LPIPS as a function of rollout time on the test set. The absolute gap between the LoRAadapted models and the pretrained base is widest at the shortest rollout time we sample (t=0.5 s, base mean LPIPS 0.542 vs. exocentric-only 0.460, a gap of 0.082) and narrows steadily as t grows, since all configurations drift away from the single ground-truth trajectory in the same way. Table 8 reports the paired LPIPS gap at

![](images/5b1703a129808a84bb644ffb25746bedb81986c17b3508f55dee7b0ba3f95103.jpg)  
Figure 6 Opening a freezer to take an ice cream. Base distorts the freezer structure as the rollout progresses; the adapted configurations preserve the freezer geometry, and exo-only produces the most physically sensible reach for the ice-cream tub.

fixed time stamps within this near-horizon regime.

The near-horizon LPIPS gap to the pretrained base is largest at the shortest rollout time we sample $( t { = } 0 . 5 \mathrm { s } ,$ see figure 13 in the appendix) and remains substantial throughout the $t \in [ 1 . 0 , 2 . 0 ]$ s window. At t=1.0 s the exocentric-only configuration reaches a paired LPIPS gap of +0.075 at 89.0% paired win-rate $\left( p < 1 0 ^ { - 2 8 } \right)$ and the combined configuration +0.068 at 83.5% $( p < 1 0 ^ { - 2 4 } )$ . At t=2.0 s the gap remains substantial (+0.064 to +0.070 at 87.5−96.0% paired win-rate). At longer rollout horizons all configurations drift away from the single ground-truth trajectory in the same way, and the absolute LPIPS gap narrows; we omit those time stamps from the table for clarity and report the full curve in the appendix.

The near-horizon window has a clear interpretive meaning: it is the regime in which the world model’s prediction quality matters most for embodied use, since downstream planning and policy verification typically rely on rollouts of one to three seconds Wayve (2025). Our results indicate that LoRA adaptation is most beneficial precisely in this regime.

## 5.5 Ablations and Sanity Checks

Time-aligned vs. frame-index-based sampling. A naive LPIPS implementation would compare generated and ground-truth frames by index, mis-aligning the generated 24 FPS output against ground-truth at 4 FPS. Switching to the wall-clock time-aligned protocol of section 4.2 increases the paired LPIPS gap on our test set by approximately 3.9× over the index-based comparison. We recommend the time-aligned protocol for any video evaluation in which the generated and reference frame rates difer.

Adapter merge verification. Across all 36 transformer blocks the post-merge weight difers from the pretrained weight in 98.83% of its entries, with per-tensor maximum delta magnitude of 17.8% of the base tensor magnitude. This excludes a class of failure modes in which the adapter loads silently as a zero increment due to a Difusers PEFT name mismatch.

Seed variance. The seed-induced spread on the combined configuration (table 4) is at least an order of magnitude smaller than the gap between any LoRA configuration and the base on every paired metric, supporting the interpretation that observed configuration-to-configuration ordering is not driven by optimization noise.

![](images/daec97b209c1930eae0201020bb48884c18dc4bff355d833ab7c21a05485a831.jpg)  
Figure 7 Carrying a crate through an aisle. Base produces a hand–crate interpenetration at t=5 s: the person’s hands appear to pass through the yellow crate while walking. The adapted configurations preserve a plausible two-hand carry posture.

Modern metric sensitivity. On our test set, the DreamSim paired improvement for the exocentric-only configuration corresponds to a relative reduction of 23.6% from the base mean, approximately 2.7× the LPIPS relative reduction of 8.7%. On the distributional side, JEDi shows a 33.5% relative reduction for exocentric-only versus 17.7% for R3D-Fréchet, a 1.9× amplification. In both pairings the modern metric agrees with the classical metric on the configuration ranking but provides a more sensitive measurement of the adaptation efect, consistent with the design of DreamSim Fu et al. (2023) and JEDi Luo et al. (2024) to capture aspects of similarity that the classical metrics miss.

## 6 Conclusion

We introduced RetailSMV (Retail Synchronized Multi-View), a retail video corpus of 32,105 captioned clips with synchronized egocentric and exocentric capture of the same store-staf activities across five real-world supermarkets, and used it to conduct a controlled study of view-stratified Low-Rank Adaptation for foundation video world models. By training three matched LoRA configurations of Cosmos3-Nano on RetailSMV under identical hyperparameters and optimization budget, and by evaluating them on a held-out test set with a seven-metric suite under a strict paired statistical protocol, we have arrived at several conclusions of practical interest for the world-model adaptation community.

Adaptation is statistically uniform. The reduction in validation difusion loss is large (∼2.8×), uniform across configurations, and achieves perfect paired ordering $( p \ll 0 . 0 0 1 )$ . On this aggregate signal alone there is no ambiguity that the pretrained foundation video model benefits substantially from domain-specific LoRA adaptation. Validation loss alone, however, does not discriminate between view-stratified data choices; the configuration diferences emerge when generation quality is measured directly.

Exocentric-only adaptation matches or exceeds combined adaptation, with significant gains on LPIPS, PSNR, and DreamSim. Our headline finding is that exocentric-only training, using only the 15,985 exocentric clips of the synchronized ego+exo corpus, matches or exceeds the combined configuration on six of the seven point estimates we report. The exocentric-only adapter wins on R3D-Fréchet (35.14 vs. combined 36.20 vs. base 42.69), on JEDi (0.829 vs. combined 0.892 vs. base 1.246, a 33.5% relative reduction), on LPIPS (a paired reduction of 0.058 at 100% paired win-rate vs. combined’s 0.049 at 94.5%), on PSNR (an improvement $\mathrm { o f \ + 0 . 5 7 5 \ : d B \ v s . \ + 0 . 5 0 0 \ d B ) }$ , on Hessel CLIPScore (+0.018 vs. +0.017), and on DreamSim (a reduction of 0.062 vs. 0.046), even though the combined configuration also sees the 16,120 egocentric clips. The combined configuration edges exo only on SSIM, by 0.0012. Direct paired tests between the two adapted configurations (table 6) confirm that combined is significantly worse than exo on LPIPS, PSNR, and DreamSim, and the two distributional metrics (R3D-Fréchet and JEDi) agree on the ranking under either feature backbone. The corresponding combined-vs-egocentric tests (table 7) show the asymmetric counterpart: combined is significantly better than ego-only on LPIPS (p=0.002), PSNR $\left( p < 1 0 ^ { - 6 } \right)$ , and SSIM $( p < 1 0 ^ { - 1 4 } )$ . Adding the exocentric subset to the egocentric subset therefore helps, whereas adding the egocentric subset to the exocentric subset hurts. When synchronized ego/exo capture is available, single-view exocentric training is therefore a strong default; combined ego+exo training does not improve over exo-only training on this corpus. We hypothesize that this reflects the wider field of view and stable reference frame of the exocentric camera, which align well with the inductive prior of a difusion video model that was pretrained primarily on stable third-person footage. What matters is in-distribution coverage of the deployment viewpoint, not raw clip count.

![](images/0ac3c1efe9ab718176857057af710c59fc44fd5f179ebc869d322de01cd50736.jpg)  
Figure 8 Picking up a metal scoop at the bulk-grain bins. The conditioning frame shows the person standing at a row of open bulk-grain bins, ready to grasp a metal scoop. Base produces a frame at t=1 s in which the scoop drifts upward in mid-air with no hand around it (the agent–object grasp is lost); the three adapted configurations close the actor’s hand on the scoop and lift it as a single rigid body.

Egocentric data transfers, but ranks third. The egocentric-only adapter trained on 16,120 egocentric clips improves over the pretrained base on every metric at $p { < } 1 0 ^ { - 8 }$ , including LPIPS (+0.041 at 90.5% paired win-rate), CLIPScore (+0.017 at 72.5%), and DreamSim $\mathrm { ( + 0 . 0 4 3 }$ at 78.0%). The transfer is substantive but consistently weaker than the in-distribution exocentric-only adapter. This pattern may inform data-collection priorities for future embodied world models in retail-like deployment domains.

![](images/a156d4b08169d5981f8c65854bae1cef122f094bd07bfc97b36625b5f851a01d.jpg)  
Figure 9 Pushing a cart through the aisle. The conditioning frame shows the person pushing a cart with bottle down an aisle whose back wall is closed. Base produces a frame at t=3 s in which a spurious opening appears in the back wall of the aisle, exposing an unrelated store section that is not present in the conditioning frame. The three adapted configurations preserve the original aisle wall and product layout for the full rollout, with exocentric-only producing the most stable scene geometry.

The adaptation gap is concentrated in the near-horizon prediction window. The absolute LPIPS gap to the pretrained base is largest at the shortest rollout time we sample (t=0.5 s, base mean LPIPS 0.542 vs. exocentric-only 0.460, a gap of 0.082) and narrows steadily as rollout time grows, since all configurations drift away from the single ground-truth trajectory in the same way. In the t∈[1.0, 2.0] s window the paired gap is still substantial: at t=1.0 s the exocentric-only configuration reaches a paired LPIPS improvement of +0.075 at 89% paired win-rate, approximately 1.3× the full-clip average; the combined and egocentric-only configurations exhibit the same near-horizon amplification at +0.068 and +0.061 respectively. The near-horizon window is precisely the regime in which world models are most directly used for embodied control Wayve (2025). We recommend reporting the LPIPS gap at fixed near-horizon time stamps in addition to the full-clip average as a standard reporting convention.

Modern metrics are more sensitive. The DreamSim paired improvement for the exocentric-only configuration is approximately 2.7× the LPIPS paired improvement in relative terms (23.6% vs. 8.7%), while the two metrics produce the same configuration ranking; on the distributional side, JEDi’s 33.5% relative reduction is roughly 1.9× the corresponding R3D-Fréchet reduction (17.7%), again with identical configuration ordering. This is consistent with the design of DreamSim and JEDi to capture mid-level semantic and feature-distribution similarity that the classical metrics miss Fu et al. (2023); Luo et al. (2024). We report JEDi as a recent distributional video-quality metric complementary to R3D-Fréchet; the conclusions in this paper do not rely on JEDi alone and are supported by R3D-Fréchet and DreamSim as well. We recommend that domain-specific video evaluations report DreamSim alongside (or in place of) LPIPS, and a V-JEPA-based MMD alongside an I3D/R3D Fréchet distance.

Table 8 Near-horizon LPIPS analysis on the held-out 200-clip test set. The paired LPIPS gap to the pretrained base widens from the unrestricted full-clip average (+0.058 for exocentric-only, +0.049 for combined, +0.041 for egocentric-only) to higher values in the near-horizon t∈[1.0, 2.0] second window; figure 13 in the appendix shows that the gap is largest at the shortest rollout time we sample (t=0.5 s) and narrows monotonically thereafter.

<table><tr><td>Time window</td><td>Configuration</td><td>LPIPS Δ</td><td>Win</td><td>p</td></tr><tr><td>Unrestricted average</td><td>Egocentric-only</td><td>+0.041</td><td>90.5%</td><td> $<10^{-27}$ </td></tr><tr><td>Unrestricted average</td><td>Exocentric-only</td><td>+0.058</td><td>100%</td><td> $<10^{-34}$ </td></tr><tr><td>Unrestricted average</td><td>Combined</td><td>+0.049</td><td>94.5%</td><td> $<10^{-31}$ </td></tr><tr><td>t=1.0 second</td><td>Egocentric-only</td><td>+0.061</td><td>82.0%</td><td> $<10^{-20}$ </td></tr><tr><td>t=1.0 second</td><td>Exocentric-only</td><td>+0.075</td><td>89.0%</td><td> $<10^{-28}$ </td></tr><tr><td>t=1.0 second</td><td>Combined</td><td>+0.068</td><td>83.5%</td><td> $<10^{-24}$ </td></tr><tr><td>t=2.0 seconds</td><td>Egocentric-only</td><td>+0.056</td><td>87.5%</td><td> $<10^{-25}$ </td></tr><tr><td>t=2.0 seconds</td><td>Exocentric-only</td><td>+0.070</td><td>96.0%</td><td> $<10^{-32}$ </td></tr><tr><td>t=2.0 seconds</td><td>Combined</td><td>+0.064</td><td>88.5%</td><td> $<10^{-27}$ </td></tr></table>

## Limitations

Our study has four limitations we believe deserve explicit mention.

First, we adapt one base model family (Cosmos3-Nano). A natural extension is to evaluate whether the qualitative findings transfer to other foundation video models such as Cosmos-Predict 2.5, Sora, and Stable Video Difusion. A parallel adaptation efort to Cosmos-Predict 2.5 is ongoing.

Second, our Fréchet-style metric uses a ResNet-3D Kinetics backbone (R3D-Fréchet) rather than the canonical I3D backbone, so absolute values are not directly comparable to I3D-FVD values in the broader literature; see section C for details and a recipe for recomputing with the canonical I3D backbone.

Third, we do not conduct a human evaluation. The metrics community has documented that human judgment remains the gold standard for video generation quality Fu et al. (2023); Luo et al. (2024). Our paired statistical protocol provides a transparent basis for any subsequent human-versus-automatic alignment study, and the modern metric we adopt (DreamSim) is designed precisely to narrow the gap to human judgment, but a direct human comparison would strengthen the conclusions.

Fourth, we use image-and-text conditioning. The natural use case for a world model in embodied control is action conditioning, where the model rolls out a counterfactual based on a proposed action sequence. Cosmos3- Nano does not natively support action conditioning; adding it is a promising direction for downstream evaluation as a true simulator for embodied control.

Beyond these specific limitations, the broader takeaway is that view-stratified studies, paired statistics, and modern perceptual metrics together support reproducible, well-controlled studies of parameter-eficient video world model adaptation. We hope the protocol and findings presented here are useful starting points for the next generation of domain-specialized generative simulators.

![](images/b587b623c623506d0ab5b299f4573267be61305e992aeb5c9922efc63bab59cd.jpg)  
Figure 10 Reaching for a back-row bottle on the shelf. The conditioning frame shows the person reaching for a black bottle at the back of a beverage shelf. Base produces an implausible grip pose around the bottle at t=3 s with no clear contact configuration; the three adapted configurations produce a recognisable reach-and-grasp consistent with picking up a back-row bottle.

![](images/a4e13138664dbad6c1ec94e44e098ed45530d3656b6e067d58638e2af25d0374.jpg)  
Figure 11 Carrying a basket through the aisles to the dairy section. The conditioning frame shows the person walking with a blue shopping basket held with both hands. Base produces frames at t=3 s and t=7 s in which the person’s hands enter the interior volume of the basket rather than gripping the handles; the three adapted configurations maintain a plausible two-hand basket carry posture throughout the rollout.

![](images/7d4cb38fe07b0fc0289f4ac93e5fc958df1c1e2f38fdb91a6f2926ecce608246.jpg)  
Figure 12 Inspecting a bottle of cleaning supplies. The conditioning frame shows the person pushing a cart down a cleaning-supplies aisle. Base produces a frame at t=7 s in which the reaching hand penetrates the shelf guard rail; the three adapted configurations preserve a correct hand–shelf contact and a recognisable inspection pose.

## References

Omer Bar-Tal, Hila Chefer, Omer Tov, Tim Brooks, Amir Hertz, Tali Dekel, and Inbar Mosseri. Lumiere: A space-time difusion model for video generation. In ACM SIGGRAPH Asia, 2024.

Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. Revisiting feature prediction for learning visual representations from video. TMLR, 2024.

Kevin Black et al. π<sub>0</sub>: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

Andreas Blattmann, Tim Dockhorn, Sumith Kulal, et al. Stable video difusion: Scaling latent video difusion models to large datasets. arXiv preprint arXiv:2311.15127, 2023.

Anthony Brohan et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), 2023.

Jake Bruce, Michael D Dennis, Ashley Edwards, Jack Parker-Holder, Yuge Shi, Edward Hughes, et al. Genie: Generative interactive environments. In ICML, 2024.

Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. Emerging properties in self-supervised vision transformers. In ICCV, 2021.

João Carreira and Andrew Zisserman. Quo vadis, action recognition? a new model and the Kinetics dataset. In CVPR, 2017.

Haoxin Chen, Yong Zhang, Xiaodong Cun, et al. VideoCrafter2: Overcoming data limitations for high-quality video difusion models. In CVPR, 2024.

AgiBot World Contributors. Agibot world colosseo: A large-scale manipulation platform for scalable and intelligent embodied systems. arXiv preprint arXiv:2503.06669, 2025.

Dima Damen et al. Scaling egocentric vision: The EPIC-KITCHENS dataset. In ECCV, 2018.

Janez Demšar. Statistical comparisons of classifiers over multiple data sets. JMLR, 2006.

Tim Dettmers et al. QLoRA: Eficient finetuning of quantized language models. arXiv preprint arXiv:2305.14314, 2023.

Carl Doersch, Abhinav Gupta, and Alexei A Efros. Unsupervised visual representation learning by context prediction. In ICCV, 2015.

Danny Driess et al. PaLM-E: An embodied multimodal language model. arXiv preprint arXiv:2303.03378, 2023.

Stephanie Fu, Netanel Tamir, Shobhita Sundaram, et al. DreamSim: Learning new dimensions of human visual similarity using synthetic data. In NeurIPS, 2023.

Justine Gajo, Aldrin Merales, Timothy Escarcha, Marco Molina, Antonio Nartea, Andrei Maminta, Joshua Roldan, and Rowel Atienza. Sari Sandbox: A virtual retail store environment for embodied AI agents. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), RetailVision, 2025. arXiv:2508.00400.

Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. An image is worth one word: Personalizing text-to-image generation using textual inversion. In ICLR, 2023.

Shenyuan Gao, Jiazhi Yang, and Li Chen. Survey of generative world models for embodied ai. arXiv preprint arXiv:2502.00060, 2025.

Shenyuan Gao et al. Vista: A generalizable driving world model with high fidelity and versatile controllability. arXiv preprint arXiv:2405.17398, 2024.

Gemini-Robotics-ER-1.5. Google gemini er 1.5, 2025. URL https://ai.google.dev/gemini-api/docs/ robotics-overview.

Kristen Grauman et al. Ego4D: Around the world in 3,000 hours of egocentric video. In CVPR, 2022.

Kristen Grauman et al. Ego-Exo4D: Understanding skilled human activity from first- and third-person perspectives. arXiv preprint arXiv:2311.18259, 2024.

Yuwei Guo, Ceyuan Yang, Anyi Rao, et al. AnimateDif: Animate your personalized text-to-image difusion models without specific tuning. In ICLR, 2024.

David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv:2301.04104, 2023.

Jack Hessel, Ari Holtzman, Maxwell Forbes, et al. CLIPScore: A reference-free evaluation metric for image captioning. arXiv preprint arXiv:2104.08718, 2021.

Jonathan Ho, Ajay Jain, and Pieter Abbeel. Denoising difusion probabilistic models. In NeurIPS, 2020.

Anthony Hu, Lloyd Russell, Hudson Yeo, et al. GAIA-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.

Edward J Hu et al. LoRA: Low-rank adaptation of large language models. In ICLR, 2022.

Ziqi Huang, Yinan He, Jiashuo Yu, et al. VBench: Comprehensive benchmark suite for video generative models. In CVPR, 2024.

Ziqi Huang et al. VBench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness. arXiv preprint arXiv:2503.21755, 2025.

Dan Kondratyuk, Lijun Yu, Xiuye Gu, et al. VideoPoet: A large language model for zero-shot video generation. In ICML, 2024.

Weijie Kong, Qi Tian, Zijian Zhang, et al. HunyuanVideo: A systematic framework for large video generative models. arXiv:2412.03603, 2024.

Nupur Kumari, Bingliang Zhang, Richard Zhang, Eli Shechtman, and Jun-Yan Zhu. Multi-concept customization of text-to-image difusion. In CVPR, 2023.

Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling. In ICLR, 2023.

Xingchao Liu, Chengyue Gong, and Qiang Liu. Flow straight and fast: Learning to generate and transfer data with rectified flow. In ICLR, 2023.

Yunze Liu, Yun Liu, Che Jiang, et al. HOI4D: A 4d egocentric dataset for category-level human-object interaction. In CVPR, 2022.

Ilya Loshchilov and Frank Hutter. Decoupled weight decay regularization. In ICLR, 2019. URL https://arxiv.org/ abs/1711.05101.

Ge Ya Luo, Gian Mario Favero, Zhi-Hao Luo, et al. Beyond FVD: Enhanced evaluation metrics for video generation quality. In NeurIPS, 2024.

Karttikeya Mangalam, Raiymbek Akshulakov, and Jitendra Malik. EgoSchema: A diagnostic benchmark for very long-form video language understanding. In NeurIPS, 2023.

Davide Mazzini et al. RetailAction: Dataset for multi-view spatio-temporal localization of human–object interactions in retail environments. In Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops (ICCVW), RetailVision, 2025.

Meta GenAI. Movie gen: A cast of media foundation models. arXiv preprint arXiv:2410.13720, 2024.

Meta Reality Labs. Aria everyday activities dataset. In CVPR, 2024.

NVIDIA. Cosmos World Foundation Model Platform for Physical AI. arXiv preprint arXiv:2501.03575, 2025.

NVIDIA. Cosmos-reason1: From physical common sense to embodied reasoning. arXiv preprint, 2025a.

NVIDIA. Cosmos-reason1 and cosmos-reason2: Reasoning foundation models for physical common sense. arXiv preprint, 2025b.

NVIDIA. GR00T n1: An open foundation model for generalist humanoid robots. arXiv preprint, 2025c.

OpenAI. Video generation models as world simulators. openai.com/sora, 2024.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. Learning transferable visual models from natural language supervision. 2021.

RetailVision Organizers. RetailVision workshop series. https://retailvisionworkshop.github.io, 2020–2025. Annual workshop at CVPR/ICCV, 2020–2025.

Amirreza Rouhi, Parikshit Sakurikar, Satya Sai Reddy, Narsimha Menga, Anirudh Govil, Sri Harsha Chittajallu, Rajat Aggarwal, Anoop Namboodiri, and Sashi Reddi. PRISM: A multi-view multi-capability retail video dataset for embodied vision-language models. arXiv preprint arXiv:2603.29281, 2026.

Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. DreamBooth: Fine tuning text-to-image difusion models for subject-driven generation. In CVPR, 2023.

Pierre Sermanet, Tianli Ding, Jefrey Zhao, Fei Xia, Debidatta Dwibedi, Keerthana Gopalakrishnan, Christine Chan, Gabriel Dulac-Arnold, Sharath Maddineni, Nikhil Jain, Peng Xu, Yunfei Yuan, et al. RoboVQA: Multimodal long-horizon reasoning for robotics. arXiv preprint arXiv:2311.00899, 2023.

Gunnar A Sigurdsson, Abhinav Gupta, Cordelia Schmid, et al. Actor and observer: Joint modeling of first and third-person videos. In CVPR, 2018.

Uriel Singer, Adam Polyak, Thomas Hayes, et al. Make-A-Video: Text-to-video generation without text-video data. In ICLR, 2023.

Bharat Singh, Tim K Marks, Michael Jones, Oncel Tuzel, and Ming Shao. A multi-stream bi-directional recurrent neural network for fine-grained action detection. In CVPR, 2016.

Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising difusion implicit models. In ICLR, 2021.

Du Tran, Heng Wang, Lorenzo Torresani, Jamie Ray, Yann LeCun, and Manohar Paluri. A closer look at spatiotemporal convolutions for action recognition. In CVPR, 2018.

Thomas Unterthiner et al. Towards accurate generative models of video: A new metric & challenges. arXiv preprint arXiv:1812.01717, 2018.

Xiaofeng Wang et al. DriveDreamer: Towards real-world-driven world models for autonomous driving. In ECCV, 2024a.

Zhou Wang, Alan C Bovik, Hamid R Sheikh, and Eero P Simoncelli. Image quality assessment: from error visibility to structural similarity. IEEE TIP, 2004.

Zhouxia Wang, Ziyang Yuan, Xintao Wang, et al. MotionCtrl: A unified and flexible motion controller for video generation. In ACM SIGGRAPH, 2024b.

Wayve. GAIA-2: A controllable multi-view generative world model for autonomous driving. arXiv preprint arXiv:2503.20523, 2025.

Dongling Wei et al. Arrow of time and its reversal on the IBM quantum computer. In Scientific Reports, 2019. Original concept: Pickup et al., Arrow of Time in Videos, BMVC 2014.

Jinbo Xing, Menghan Xia, Yong Zhang, et al. DynamiCrafter: Animating open-domain images with video difusion priors. In ECCV, 2024.

Mengjiao Yang, Yilun Du, Kamyar Ghasemipour, Joshua B Tenenbaum, Dale Schuurmans, and Pieter Abbeel. Learning interactive real-world simulators. In ICLR, 2024.

Zhuoyi Yang, Jiayan Teng, Wendi Zheng, et al. CogVideoX: Text-to-video difusion model with an expert transformer. In ICLR, 2025.

Qingru Zhang, Minshuo Chen, Alexander Bukharin, Pengcheng He, Yu Cheng, Weizhu Chen, and Tuo Zhao. AdaLoRA: Adaptive budget allocation for parameter-eficient fine-tuning. In ICLR, 2023.

Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. The unreasonable efectiveness of deep features as a perceptual metric. In CVPR, 2018.

Rui Zhao, Yuchao Gu, Jay Zhangjie Wu, et al. MotionDirector: Motion customization of text-to-video difusion models. In ECCV, 2024.

Wenliang Zhao, Lujia Bai, Yongming Rao, Jie Zhou, and Jiwen Lu. UniPC: A unified predictor-corrector framework for fast sampling of difusion models. In NeurIPS, 2023.

Zangwei Zheng et al. Open-sora: Democratizing eficient video production for all. github.com/hpcaitech/Open-Sora, 2024.

## Appendix

## A Reproducibility Details

## A.1 Hardware and Software

All training and evaluation experiments run on four NVIDIA RTX PRO 6000 Blackwell (96 GB HBM) GPUs in a single host. The software stack is Python 3.12 with CUDA 12.8 and PyTorch 2.11 (cu128 build, including the sm\_120 device kernels required by the Blackwell architecture), together with Difusers 0.39, PEFT 0.19, and Accelerate 1.13. The two modern metrics that require dedicated environments are isolated. DreamSim is run in a virtual environment with transformers<4.50 (required by the oficial DreamSim distribution). JEDi is run in a separate environment containing the oficial videojedi package together with the V-JEPA ViT-H/16 feature extractor checkpoint released by the JEDi authors; this environment pins the V-JEPA preprocessing utilities to the versions used in the original “Beyond FVD” paper.

## A.2 Hyperparameters

Table 9 lists the hyperparameters shared by the three view-stratified configurations and the second-seed control. All four configurations use identical hyperparameters apart from the training data subset and (for the second-seed control) the random seed.

Table 9 Adapter training hyperparameters. All four configurations share these values; only the training data subset and (for the seed-137 control) the random seed difer.

<table><tr><td>Field</td><td>Value</td></tr><tr><td>Base model</td><td>nvidia/Cosmos3-Nano (16B params)</td></tr><tr><td>LoRA rank r</td><td>32</td></tr><tr><td>LoRA scaling α</td><td>64</td></tr><tr><td>Target modules</td><td>cross-attention + MoE MLPs (gen-modality)</td></tr><tr><td>Trainable parameters</td><td> $87.29 \times 10^{6}$  (~ 0.55% of base)</td></tr><tr><td>Optimizer</td><td>AdamW ( $\beta_{1}$ =0.9,  $\beta_{2}$ =0.999, wd 0.01)</td></tr><tr><td>Peak learning rate</td><td> $3 \times 10^{-4}$ </td></tr><tr><td>LR schedule</td><td>cosine, 100 warm-up steps</td></tr><tr><td>Optimization steps</td><td>3,000</td></tr><tr><td>Validation cadence</td><td>every 100 steps, n=32 samples</td></tr><tr><td>Checkpoint cadence</td><td>every 250 steps</td></tr><tr><td>Selected adapter</td><td>iter 1500 (lowest val. loss)</td></tr><tr><td>Effective batch size</td><td>8 (B=1, grad acc =2, 4 GPUs DDP)</td></tr><tr><td>Precision</td><td>BF16 mixed</td></tr><tr><td>Activation checkpointing</td><td>enabled</td></tr><tr><td>Video decoder</td><td>decord</td></tr><tr><td>Spatial resolution</td><td>480×832</td></tr><tr><td>Frames per clip</td><td>81</td></tr><tr><td>Source FPS</td><td>24</td></tr><tr><td>Seed</td><td>42 (and 137 for the combined seed control)</td></tr></table>

## A.3 Inference Settings

For every (configuration, test clip) pair we run image-to-video inference with: 189 frames (Cosmos3-Nano’s native maximum, ∼ 7.9 seconds at 24 FPS), spatial resolution 480×832, the UniPC multistep scheduler with $\sigma _ { \mathrm { s h i f t } } { = } 1 0$ , 20 inference steps, classifier-free guidance scale 6, negative prompt “low quality, blurry, distorted, glitches, watermark”, deterministic seed 42, output FPS 24.

## A.4 Split Construction

We hold out clips from training and partition the held-out pool into a validation set used for adapter selection (1,388 clips, from which we sample n=32 paired clips for the rectified-flow validation loss every 100 training steps) and a test set used for final evaluation (200 clips, balanced across both egocentric $\scriptstyle ( n = 1 0 0 )$ and exocentric (n=100) views with rejection sampling for zero overlap with the validation set). The training set (32,105 clips), the validation set (1,388 clips), and the test set (200 clips) are pairwise disjoint by per-clip unique identifier.

## B Extended Results

## B.1 Exact Validation-Loss p-Values

All four LoRA configurations reduce validation difusion loss on every one of the n=200 paired evaluation samples. Under the Wilcoxon signed-rank test the per-row p-values are $\sim 1 0 ^ { - 1 0 5 }$ for egocentric-only, $\sim 1 0 ^ { - 1 0 4 }$ for exocentric-only, $\sim 1 0 ^ { - 1 0 4 }$ for combined seed-42, and $\sim 1 0 ^ { - 1 0 5 }$ for combined seed-137. These numbers reflect the nominal p associated with perfect paired ordering on n=200 samples and should be interpreted as “every paired sample improves over the base”; we report $p \ll 0 . 0 0 1$ in the main text accordingly.

## B.2 Cross-View Egocentric Transfer Check

The main test set is a stratified 200-clip pool covering both views. To isolate the cross-view behaviour of the three adapters, we report paired LPIPS on the egocentric-view subset of the main test set (n=100 ego clips), so that the same generations used for the main metric panel are reused. Table 10 shows that al three adapted configurations significantly improve egocentric LPIPS on this subset, with the exocentric-only adapter producing the largest paired improvement (+0.045 at 100% paired win-rate) – i.e., the exocentric-only adapter does transfer to the egocentric viewpoint when measured on these clips, and slightly outperforms the egocentric-only adapter (+0.039 at 96% paired win-rate). The combined configuration is comparable to ego-only (+0.038 at 98%). This pattern is consistent with the headline result: under matched optimization budget, the exocentric subset is doing the heavy lifting of adaptation, including at cross-view test time.

Table 10 Cross-view check on the 100-clip egocentric subset of the main test set. All three adapted configurations significantly improve egocentric LPIPS on this subset; the exocentric-only adapter delivers the largest improvement, supporting the headline finding that exocentric-only training is a strong default in this corpus.

<table><tr><td>Configuration</td><td>LPIPS Δ</td><td>Win</td><td>p (Wilcoxon)</td></tr><tr><td>Egocentric-only</td><td>+0.039</td><td>96.0%</td><td> $< 10^{-13}$ </td></tr><tr><td>Exocentric-only</td><td>+0.045</td><td>100%</td><td> $< 10^{-14}$ </td></tr><tr><td>Combined</td><td>+0.038</td><td>98.0%</td><td> $< 10^{-13}$ </td></tr></table>

## B.3 Rollout-Horizon Curve

Figure 13 plots mean LPIPS as a function of rollout time on the held-out 200-clip test set, one curve per configuration. All four curves rise monotonically with t: at t=0.5 s the base mean LPIPS is 0.542 versus 0.460 for the exocentric-only configuration (a gap of 0.082); at t=4.0 s the base reaches 0.685 versus exo 0.627 (a smaller gap of 0.058). The absolute gap to the pretrained base is widest at the smallest rollout times and narrows steadily as t grows, since all configurations drift away from the single ground-truth trajectory in the same way. The exocentric-only configuration retains the smallest LPIPS at every time stamp.

![](images/c0084ac3167191722c3537140b0763d6b15511455411ee260834e49222e05d15.jpg)  
Figure 13 Mean LPIPS as a function of rollout time on the held-out 200-clip test set, one curve per configuration. The exocentric-only configuration retains the lowest LPIPS at every time stamp. The absolute gap to the pretrained base is widest at the smallest rollout times and narrows as t grows, since all configurations drift away from the single ground-truth trajectory.

## C Practical Notes

## C.1 R3D-Fréchet vs. Canonical I3D-FVD

The canonical Fréchet video distance is computed on features extracted by the Inflated 3D ConvNet (I3D) of Carreira and Zisserman (2017). A turnkey I3D feature extractor with Kinetics-400 weights is not currently packaged with mainstream deep-learning frameworks; for this paper we use the closely related ResNet-3D-18 video classifier (torchvision.models.video.r3d\_18 with KINETICS400\_V1 weights) and refer to the resulting metric as R3D-Fréchet to make the deviation explicit. Both feature extractors are pretrained for Kinetics-400 action classification at the same input clip length, so the Fréchet distance retains its intended interpretation as a measure of distributional similarity in a Kinetics-aware embedding. Absolute values, however, are not directly comparable to I3D-FVD numbers from the broader literature. Recomputing the Fréchet-style metric with the canonical I3D backbone is straightforward but requires installing the I3D feature extractor outside the standard framework distribution; we recommend doing so for any direct comparison with prior work.

## C.2 Time-Aligned Sampling

A frame-index-based implementation of LPIPS would mis-align Cosmos3-Nano’s 24 FPS generated output against the 4 FPS ground-truth clips, yielding pairings of frames separated by up to two seconds of wall-clock time. Wall-clock time-aligned sampling (16 evenly-spaced timestamps in [0, min(gen\_dur, gt\_dur)], dropping t=0 to remove the trivial first-frame match) increases the paired LPIPS gap on our test set by approximately 3.9× over the index-based comparison. We recommend the time-aligned protocol for any video evaluation where the generated and reference frame rates difer.

## C.3 Adapter-Merge Verification

For inference we explicitly merge the trained adapter $\Delta W = \left( \alpha / r \right)$ BA into each corresponding base weight matrix and verify the merge numerically. Across all 36 transformer blocks the merged weight difers from the base in 98.83% of its entries, and the per-tensor maximum magnitude of ∆W reaches 17.8% of the base weight magnitude. This sanity check rules out a class of silent-failure modes (adapter loading as a zero increment) that would otherwise produce numerically valid but functionally untrained outputs.

## D Release Policy and Per-Clip Data Availability

Release. The release accompanying this paper will include a representative subset of RetailSMV suficient to reproduce the reported metrics, the full set of evaluation scripts (data preparation, generation, metric computation, paired statistical tests), the generated videos used in the main results, and all per-clip metric files. Full corpus access is controlled: requests will be served on a case-by-case basis through the same channel as the released subset, subject to the privacy and licensing terms under which the corpus was collected (section 3.1).

Per-clip data. For each (configuration, test clip) pair we provide per-clip time-aligned LPIPS, PSNR, SSIM, Hessel CLIPScore, and DreamSim, together with the per-clip wall-clock LPIPS series at t ∈ {0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0} seconds. We also provide the 800 generated videos (four configurations × 200 test clips), the precomputed R3D feature arrays used to compute R3D-Fréchet, and the V-JEPA ViT-H/16 feature arrays used to compute JEDi (one (B,D) array per configuration plus the ground-truth feature array).

## E Contributions and Acknowledgments

Core Contributors. Amirreza Rouhi, Rajat Aggarwal, Parikshit Sakurikar, Anoop M. Namboodiri, Sashi P. Reddi.

Contact. Correspondence and corpus-access requests can be directed to the core contributors at the following addresses:

• amir@dreamvu.ai

• rajat@dreamvu.ai

• parikshit@dreamvu.ai

• anoop@dreamvu.ai

• sashi@dreamvu.ai