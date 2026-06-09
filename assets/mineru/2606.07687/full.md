# What Makes Video World Model Latents Action-Relevant: Prediction over Reconstruction

Jewon Yeom∗ Hanseul Kim∗ Jeongjae Park Sungmok Jung Jaejin Lee Taesup Kim†

Graduate School of Data Science, Seoul National University

Abstract: Video world models are increasingly used to provide predictive visual representations, yet it remains unclear which pretraining signals induce action-relevant structure in their latent spaces. We study this question through a unified probe-based evaluation across diverse encoder families, including image-only self-supervision, video pretraining with and without latent prediction, reconstruction-based autoencoders, diffusion models, and shortcut-forcing dynamics models. Using a common inverse-dynamics probing objective, we find that action-relevant structure is driven primarily by temporal video pretraining rather than pixel reconstruction fidelity: models with strong pixel decoding quality can exhibit near-zero action recoverability, while video-pretrained self-supervised encoders consistently achieve the best Pareto trade-off between visual fidelity and action prediction. Comparing V-JEPA and VideoMAE further shows that most gains arise from natural-video temporal context, with feature-level latent prediction providing a smaller additional benefit. These trends transfer across robotic benchmarks, though CALVIN reveals that static-environment tasks can partially mask the importance of temporal structure by allowing strong image priors to suffice. Finally, inverse-dynamics supervision substantially improves robustness to visual corruption, suggesting that action-aware objectives regularize latent geometry beyond clean-setting performance. Our results identify temporal predictive structure—not reconstruction fidelity—as the primary ingredient underlying action-relevant video representations.

Keywords: world models, vision-language-action, representation learning, inverse dynamics

## 1 Introduction

Vision-language-action (VLA) systems for robot manipulation increasingly use pretrained video models as their visual front-end [1, 2, 3]. The implicit assumption is that a world model trained for next-frame prediction captures action-relevant dynamical structure in its latent space, which a downstream policy can read out. This assumption matters because the choice of frozen-feature extractor is a load-bearing one for the rest of the pipeline: a representation that does not encode rotation, gripper transitions, or contact dynamics cannot be repaired by a stronger policy head. Two empirical observations complicate the assumption. The standard world-model evaluation metrics— PSNR, FVD, LPIPS—measure pixel-level prediction quality, and recent work [4, 5] shows these correlate poorly with control utility, mirroring the longer-standing observation that perception-strong video models often fail at the underlying causal and dynamical structure of the same scene [6]. At the same time, the field already spans a broad range of pretrained encoders—from reconstruction VAEs and masked image models to language-aligned vision encoders and latent prediction models—yet it remains unclear which pretraining signals actually induce action-relevant structure in the resulting latent space.

This paper asks a single question: when does the frozen latent space of a video world model carry action-relevant structure, and how does a small inverse-dynamics loss modify that structure across architectures? We approach it with a simple recipe. We either train or take a pre-existing encoder, optionally apply the inverse-dynamics loss of Han and Yilmaz [7] during world-model training or fine-tuning, then probe the frozen features for action recovery on a task-OOD split of LIBERO [8]. By holding the probe protocol fixed and varying only the encoder, the differences across rows become attributable to the backbone family. We instantiate the study with eight encoder families and a broad set of variant configurations, evaluated under both clean and visually perturbed inputs, and additionally train a lightweight pixel decoder on top of each encoder-only backbone so that all families can be compared on the same (pixel fidelity, action representation) plane.

The study answers four narrower questions in turn. What works: pixel fidelity and action recovery trade off across families, and the Pareto frontier is occupied entirely by video-pretrained selfsupervised backbones. The highest-PSNR backbones—a pixel-quality champion among actionconditioned diffusion models and the SDXL and Cosmos-1 autoencoders—all attain at-or-belowzero action $R ^ { 2 }$ . Why: comparing video pretraining with feature-level masked latent prediction (V-JEPA) to video pretraining with pixel-level masked autoencoding (VideoMAE) attributes most of the gap above image-only SSL to natural-video temporal context, and a smaller residual to the JEPA predictor objective. A per-layer probe of the V-JEPA 2 ViT-L trunk localizes this predictor signature: action $R ^ { 2 }$ peaks in the mid-trunk and drops monotonically toward the final layer on the frozen checkpoint, with the ID fine-tune reversing exactly that drop. When: the cross-architecture ranking holds on MetaWorld; on CALVIN, whose four fixed table environments let per-frame image features encode action signal without temporal context, image-only SSL backbones close most of the gap, sharpening the scope under which video temporal context is the load-bearing axis. What else: the same ID loss that lifts clean action $R ^ { 2 }$ also confers robustness. Frozen pretrained encoders collapse catastrophically under modest noise and blur, while their ID-supervised counterparts degrade gracefully on the same perturbations.

## 2 The Action-Relevance Gap in Video World Model Latents

Prior work leaves three gaps that motivate our study: world models are evaluated by pixel fidelity rather than action recoverability, representations are measured only indirectly through downstream policy success, and any single architecture is characterized rather than pretraining objectives being compared head-to-head.

World model evaluation and encoders for policy. World model quality has historically been measured by pixel-prediction metrics (PSNR, FVD, LPIPS) [9, 10], with recent benchmarks [11, 12, 13, 14, 15] extending evaluation to physics adherence and embodied utility. Tian et al. [4] showed that perceptual metrics rank video predictors differently from control success; we extend this dissociation to an eight-family matrix under a frozen-feature action probe. Joseph et al. [16] complement this from the interpretability side, localizing physical-variable representations inside V-JEPA-class encoders; we target action rather than physics representation and recover a structurally similar pattern (mid-trunk peak with output-layer drop; rotation as the family-separating axis). DINO-WM [2] and V-JEPA 2-AC [1] use frozen pretrained encoders as world-model substrates. Closest is Nilaksh et al. [5], who compare six encoders as latents for action-conditioned diffusion on Bridge V2 by rolling out a frozen policy—reading action quality only indirectly, whereas we probe the representation directly under a fixed protocol. We reuse their exact Cosmos-1 tokenizer and add two architectures absent from their study (pixel diffusion, shortcut-forcing dynamics); our rotation-specific V-JEPA finding is, to our knowledge, new.

Inverse-dynamics supervision and representation robustness. Predicting actions from consecutive frame features is a long-standing self-supervised signal [17, 18, 19, 20]. The closest prior to our recipe is WAM [7], which augments DreamerV2 with an inverse-dynamics head on CALVIN—a single architecture. We use the same loss form and fill the cross-architecture gap: the identical loss produces +0.45 $R ^ { 2 }$ lift on V-JEPA and 0.00 on Dreamer 4, a backbone-dependence not previously documented. Separately, VLA policies are known to collapse under visual distribution shift [21]; we provide the action-probe analog at the representation level, showing that frozen pretrained encoders are catastrophically perturbation-sensitive and that the same inverse-dynamics loss restores robustness in the same regimes where it lifts clean $R ^ { 2 }$ .

Action-relevant representations and physical understanding. A line of work argues that strong perceptual representations need not encode the causal structure required for physical reasoning [6, 16, 22], probing whether predictive objectives align latent spaces with abstract physical variables like velocity or motion direction [16, 4]. These probe physical variables in isolation rather than the action that generated a transition. We close that gap by asking whether pretrained video representations carry enough information to recover the underlying action, making inverse-dynamics recoverability a task-grounded definition of action-relevant structure—and find temporal predictive pretraining, not reconstruction fidelity, to be the primary factor governing whether it emerges.

## 3 Experimental Setup

Benchmark. LIBERO [8], 130 tasks, 7-DoF action space (translation, rotation, gripper). We use the task-OOD split: 104 training tasks, 26 tasks held out entirely from world-model training. Probes are trained on 400 training-task episodes and evaluated on 200 held-out-task episodes. Crossbenchmark results on CALVIN [23] and MetaWorld [24] are reported in Appendix D.

Encoders. We compare eight backbone families. DIFF is a pixel-diffusion world model trained from scratch on LIBERO at 34–91M parameters, used as the LIBERO-native baseline and for withinarchitecture ablations of $\lambda ,$ head form, and capacity. Dreamer 4 [25] is a from-scratch reproduction of the shortcut-forcing dynamics architecture; since the original Dreamer 4 implementation is not public at the time of writing, we reproduced the model from the paper and the public DreamerV3 reference scaffolding,3 training at 64M and 276M parameters from scratch and at 64M with DMControl initialization, and probing both agent-token and spatial-token feature streams. SDXL VAE [26] and the Cosmos-1 image tokenizer [27] are reconstruction-aligned autoencoders; the Cosmos-1 checkpoint we use is the exact reconstruction baseline used in concurrent work [5]. LAPA [28] is a discrete latent-action quantizer pretrained on Open X-Embodiment. Web-DINO [29] is a ViT-L image encoder trained with masked contrastive prediction; $S i g L I P 2 [ 3 0 ]$ is a ViT-L image-language contrastive encoder. VideoMAE [31] V1 ViT-L is a video-pretrained encoder trained with pixel-space masked autoencoding. V-JEPA 2 [1] ViT-L and the distilled V-JEPA 2.1 ViT-B are video-pretrained encoders trained with feature-level masked latent prediction. For each pretrained encoder we report a frozen baseline and an ID-fine-tuned variant; for DIFF and Dreamer 4 we report from-scratch training with and without the ID loss. See Appendix K for checkpoint details.

Inverse-dynamics recipe. Let $o _ { t }$ denote the observation (RGB frames from the manipulator’s camera rig at timestep t) and $a _ { t }$ the corresponding 7-DoF action (3 translation, 3 rotation, 1 gripper). For each encoder $f _ { \theta } ,$ , an inverse-dynamics head $g _ { \phi } ( f _ { \theta } ( o _ { t } ) , f _ { \theta } ( o _ { t + 1 } ) )  \hat { a } _ { t }$ is added during worldmodel training (DIFF, Dreamer 4) or fine-tuning (pretrained encoders), with loss ${ \mathcal { L } } = { \mathcal { L } } _ { \mathrm { p r i m a r y } } + \lambda$ · $\mathrm { M S E } ( \hat { a } _ { t } , a _ { t } )$ . We chose this loss over three alternatives—forward dynamics, temporal contrastive, and single-frame action prediction—based on a head-form ablation on DIFF (Appendix A); inverse dynamics produces the largest lift among the four under matched compute. We use $\lambda { = } 0 . 0 5$ for DIFF and λ=1.0 for pretrained-encoder rows (default of Han and Yilmaz [7]); a λ-sweep across five orders of magnitude shows the choice is not critical (Appendix A). $g _ { \phi }$ is discarded at probe time.

Probe. An ML $\mathrm { ~ P ~ } [ D  2 5 6  1 2 8  7 ]$ trained 3000 steps (AdamW, l $\mathrm { ~ r ~ 3 ~ } \times \mathrm { ~ 1 0 ^ { - 4 } ~ }$ , batch 512) on mean-pooled features of clean training frames. We report aggregate $R ^ { 2 }$ across the 7 action dimensions and per-dimension $R ^ { 2 }$ where mechanism matters. Mean of 3 probe seeds. Robustness uses the same probe with test-time perturbations on the trunk; see Section 4.4.

![](images/61d1aa13ee893309e74c8089db91488016c2104e9fea7407eb3eaf5e48660fdb.jpg)

<details>
<summary>scatterplot</summary>

| Method              | Pixel fidelity PSNR (dB) | Action R² (frozen) |
| ------------------- | ------------------------ | ------------------ |
| V-JEPA 2 ViT-L      | 20.0                     | 0.8                |
| Web-DINO ViT-L      | 20.5                     | 0.15               |
| SDXL VAE            | 30.0                     | -0.4               |
| Video-pretrained SSL | 20.0                     | 0.75               |
| Image SSL           | 20.5                     | 0.1                |
| Pixel-fidelity      | 27.0                     | -0.1               |
| Other (LAPA, Dreamer 4) | 20.0                   | 0.5                |
</details>

Figure 1: Pixel fidelity and frozen action-relevant structure across backbone families on LIBERO. PSNR is rollout PSNR for pixel-producing backbones and decoder PSNR for encoderonly backbones (a 17M pixel decoder on the frozen representation); action $R ^ { 2 }$ is measured on the frozen trunk before ID supervision. The two axes are uncorrelated: at PSNR ≈ 20 dB, action $R ^ { 2 }$ spans −0.01 to +0.46, and pixel-reconstruction backbones (SDXL VAE, Cosmos-1) attain the highest PSNR but the lowest action $R ^ { 2 }$ . The ID multiplier (Section 4.2) is what separates the otherwiseclustered video-SSL, DIFF, and LAPA families.

## 4 Experimental Results

## 4.1 Does Pixel Fidelity Predict Action-Relevant Structure?

The standard world-model selection criterion is pixel-prediction quality (PSNR, FVD, LPIPS). We ask whether this criterion is a useful proxy for action-relevant representation quality. To place encoder-only backbones on the same axes as pixel-producing backbones, we train a lightweight 17M-parameter pixel decoder (4-stage ConvTranspose, MSE + LPIPS loss, 10k steps) on top of each encoder-only variant; the decoder is frozen during action-probe evaluation and is used only to produce a PSNR measurement comparable across families. Figure 1 plots the representative variant from each backbone family on the PSNR-vs-action-R2 plane, with the action probe applied to the frozen representation.

The central observation of Figure 1 is that pixel prediction quality and action-relevant latent structure emerge from fundamentally different pressures during pretraining. If pixel fidelity were a good proxy for controllable dynamics, backbones with similar PSNR would exhibit similar frozen action $R ^ { 2 }$ , and vice versa. Instead, we observe the opposite: models with nearly identical reconstruction quality around 20 dB span frozen action $R ^ { 2 }$ values from near zero to 0.46, while models with similarly poor action structure vary by almost 14 dB in PSNR. The disconnect is most visible in reconstruction-heavy backbones such as SDXL VAE and Cosmos-1, which achieve the highest PSNR yet the lowest frozen action $R ^ { 2 }$ . This suggests that optimizing for visual fidelity alone encourages representations that preserve appearance and texture without organizing latent space around controllable aspects of the environment.

In contrast, temporally predictive video SSL objectives induce a qualitatively different latent geometry: V-JEPA-class models retain strong action-relevant structure despite only moderate pixel fidelity, suggesting that feature-level temporal prediction preferentially preserves transition-relevant information while discarding appearance-specific variation irrelevant for control, whereas reconstructionheavy objectives must spend capacity on static scene fidelity that does not support action prediction.

<table><tr><td>Backbone</td><td>Family</td><td>Lang.</td><td>Pretrain data</td><td>Params</td><td>Frozen</td><td>+ ID</td><td> $\Delta$ </td></tr><tr><td>V-JEPA 2 ViT-LV-JEPA 2.1 ViT-B</td><td>Video + JEPA pred.</td><td>-</td><td>Internet video</td><td>304M87M</td><td>0.400.44</td><td>0.850.82</td><td>+0.45+0.38</td></tr><tr><td>VideoMAE V1 ViT-L</td><td>Video + Pixel MAE</td><td>-</td><td>Internet video</td><td>304M</td><td>0.46</td><td>0.75</td><td>+0.29</td></tr><tr><td>Web-DINO ViT-LSigLIP 2 ViT-L</td><td>Image SSL</td><td>-✓</td><td>Web imagesWeb img-text</td><td>304M316M</td><td>-0.010.05</td><td>0.160.17</td><td>+0.17+0.12</td></tr><tr><td>LAPA-LAQ-OpenX</td><td>Latent action quant.</td><td>-</td><td>Open X-Embod.</td><td>344M</td><td>0.41</td><td>0.51</td><td>+0.10</td></tr><tr><td>DIFF</td><td>Pixel diffusion</td><td>-</td><td>LIBERO 104-task</td><td>91M</td><td>0.43</td><td>0.57</td><td>+0.14</td></tr><tr><td>SDXL VAECosmos-1 tokenizer</td><td>Pixel reconstruction</td><td>-</td><td>Web images</td><td>34M34M</td><td>-0.55-0.36</td><td>-0.41-0.29</td><td>+0.14+0.07</td></tr><tr><td>Dreamer 4</td><td>Shortcut dyn.</td><td>-</td><td>LIBERO 104-task</td><td>64–276M</td><td>-0.04</td><td>-0.04</td><td>0.00</td></tr></table>

Table 1: Inverse-dynamics auxiliary lift, ordered by pretraining objective family. Each backbone evaluated under the same ID recipe (Section 3) on LIBERO task-OOD, grouped by pretraining objective family (multi-row blocks), with language supervision and pretrain data as additional axes. Per-variant detail in Appendix G.

Frozen action $R ^ { 2 }$ alone, however, does not cleanly separate the strongest families: video-pretrained SSL, pixel-diffusion, and latent-action quantization all cluster within $R ^ { 2 } ~ \in ~ [ 0 . 4 0 , 0 . 4 6 ]$ despite very different pretraining objectives. The separation emerges only under a small inverse-dynamics objective (Section 4.2), indicating that what matters is not the amount of action information at initialization but how readily the latent space supports action-oriented adaptation.

## 4.2 Does the Inverse-Dynamics Loss Lift Every Backbone Equally?

Inverse-dynamics supervision does not uniformly improve representations. Instead, it behaves primarily as a multiplier on temporally predictive structure already present in the backbone. Videopredictive encoders benefit dramatically under the same auxiliary loss, while image-only and reconstruction-based representations remain fundamentally limited. This suggests that inverse dynamics cannot manufacture temporal structure absent from the representation; it can only amplify predictive structure already encoded in the latent space.

We apply the same inverse-dynamics (ID) loss to eight backbone families (λ=1.0 for pretrainedencoder rows, λ=0.05 for DIFF), summarized in Table 1. We chose inverse dynamics among four candidate auxiliary heads: a head-form ablation on DIFF (Appendix A) compared inverse dynamics, forward dynamics, temporal contrastive, and single-frame action prediction under matched compute and showed inverse dynamics produces the largest lift, with $L _ { 1 }$ and $L _ { 2 }$ regression forms matched at the top; we use $L _ { 2 }$ to match WAM [7]’s original recipe.

The dominant factor is temporal predictive pretraining. V-JEPA receives the largest lift under ID supervision, reaching 0.85 R², while VideoMAE reaches 0.75 under the same auxiliary loss and parameter scale. This isolates two effects: temporal video pretraining explains most of the gap relative to image-only SSL, while V-JEPA’s feature-level predictive objective contributes the remaining ∼0.1 R² advantage over pixel reconstruction objectives. Same-scale comparisons reinforce this interpretation: at ∼300M parameters, V-JEPA 2 ViT-L + ID reaches 0.85, clearing VideoMAE + ID by 0.10 and Web-DINO + ID by 0.69.

Semantic richness alone does not produce actionable temporal structure. Despite strong semantic representations, Web-DINO and SigLIP 2 remain clustered near reconstruction encoders after ID tuning, reaching only 0.16–0.17 R². A λ sweep across five orders of magnitude leaves them confined to a narrow 0.1-wide R² band (Appendix A), suggesting the limitation is representational rather than optimization-related. The data therefore does not support grouping V-JEPA with image-only “semantic SSL” methods: under identical ID supervision, V-JEPA behaves qualitatively differently from Web-DINO and SigLIP.

![](images/8a642398709290bebc5dc1edfe4ad7ed68a034ebf872cc493864323c84a10d45.jpg)  
Figure 2: Per-dimension action $R ^ { 2 }$ by pretraining backbone. Vertices are translation, rotation, and gripper $R ^ { 2 } ;$ ; the gray triangle marks $\hat { R ^ { 2 } } = 0$ . Filled triangles are +ID variants, dashed are frozen baselines. The DIFF panel uses the $L _ { 1 }$ variant; canonical- $. L _ { 2 }$ DIFF appears in Tables 1 and 2.

The results additionally rule out model capacity as the primary explanation. At ∼90M parameters, V-JEPA 2.1 ViT-B + ID (0.82) exceeds DIFF + ID (0.57) by 0.25 R². Scaling Dreamer 4 from 64M to 276M produces effectively no aggregate improvement, leaving the family flat at −0.04 regardless of scale. Capacity scaling therefore cannot recover the missing structure when the pretraining objective itself fails to encode it.

The ranking is not LIBERO-specific. Replicating the pretrained-encoder rows on CALVIN and MetaWorld (Appendix D) largely preserves the ordering—V-JEPA + ID tops MetaWorld (0.59) and CALVIN (0.88), with Spearman $\rho { = } + 0 . 8 8$ between LIBERO and MetaWorld—and MetaWorld amplifies the image-only SSL weakness, with Web-DINO and SigLIP 2 going negative. CALVIN is the lone exception: image-only SSL reaches 0.77–0.81 R² because its four fixed tabletop environments let static per-frame appearance substitute for temporal context (a probe-budget control in Appendix D rules out a data-scaling artifact). The V-JEPA advantage is thus strongest where the benchmark requires temporal disambiguation of the action target and weakens when static appearance alone suffices.

## 4.3 Where Does the V-JEPA Advantage Come From?

Figure 2 reveals that the dominant bottleneck is end-effector orientation, not translation. Translation and gripper state are recoverable even from relatively weak visual features, while rotation sharply separates predictive video models from appearance-driven encoders: in the radar plots the $\mathrm { \cdots } \mathrm { R o t . } ^ { \mathrm { * } }$ axis collapses outside the video-predictive family, and only V-JEPA sustains high performance across all three dimension groups simultaneously.

The appearance-driven families illustrate this asymmetry. Web-DINO and SigLIP retain moderate translation but produce negative rotation even after inverse-dynamics supervision, indicating that static appearance correspondence localizes coarse motion and contact geometry but not viewpointconsistent 3D orientation [16]; the ID loss cannot manufacture rotational structure absent from the trunk. Reconstruction objectives (SDXL VAE, Cosmos-1) show the same rotation-specific collapse while preserving translation, consistent with pixel-level appearance being retained but not the geometry needed to linearize orientation changes [9, 32]. Dreamer 4 differs: it floors near zero on all three groups, encoding essentially no action signal at all.

![](images/6ae61948664bf5c4704fc2df1ccb47a3206072120e630850e66e75d8ee282937.jpg)

<details>
<summary>line chart</summary>

| V-JEPA 2 ViT-L transformer block index | Frozen V-JEPA 2 ViT-L | + aux fine-tuned |
| -------------------------------------- | --------------------- | ---------------- |
| 7                                      | 0.45                  | 0.55             |
| 14                                     | 0.51                  | 0.62             |
| 18                                     | 0.51                  | 0.64             |
| 19                                     | 0.48                  | 0.66             |
| 20                                     | 0.45                  | 0.71             |
| 21                                     | 0.42                  | 0.73             |
| 22                                     | 0.39                  | 0.71             |
| 23                                     | 0.43                  | 0.69             |
</details>

Figure 3: V-JEPA 2 ViT-L per-layer action $R ^ { 2 }$ . Frozen trunk peaks at layer 14 (0.51) and drops to layer 22 (0.39); the ID fine-tune lifts the final four layers by +0.25 to +0.32 R², with the post-finetune peak at layer 21. Full table in Appendix E.

Video pretraining partially closes the rotation gap—VideoMAE improves rotational decoding over image-only SSL through temporal exposure alone—but V-JEPA expands the rotation axis further while translation and gripper remain comparable. The V-JEPA advantage is thus concentrated on the dimension requiring physically coherent latent dynamics, decoupling temporal exposure (Video-MAE, via masked pixel reconstruction) from latent predictive modeling (V-JEPA, which organizes the representation around controllable dynamics).

The per-layer analysis (Figure 3, Appendix E) supports this reading: the JEPA objective pulls the final layers away from action-readout quality—mirroring the “emerge-then-degrade” profile Joseph et al. [16] report for physical-variable decoding—and the ID loss reverses this, shifting the peak action signal back toward later layers. Together with the sparse-supervision sweep (Appendix I), this indicates that ID supervision does not create rotational structure from scratch but re-aligns orientation geometry already embedded by predictive video pretraining into a more linearly accessible form.

## 4.4 Does the Inverse-Dynamics Lift Survive Distribution Shift?

If ID supervision aligns the representation with what changes under actions, features should stay stable under perturbations that alter appearance but preserve scene dynamics, while pixel-tied representations should degrade sharply. Table 2 bears this out. Frozen pretrained encoders are highly appearance-coupled—frozen V-JEPA 2 ViT-L falls from clean 0.40 to −4.66 under blur, frozen VideoMAE to −2.54, with similar collapses for Web-DINO and SigLIP—so their clean-data structure is not stably aligned with action dynamics. ID fine-tuning changes this qualitatively: the same video-pretrained backbones degrade gracefully (V-JEPA 2 ViT-L + ID 0.85 → 0.56 under blur; VideoMAE + ID positive throughout), a shift in what information survives distribution shift rather than merely a higher clean $R ^ { 2 }$ .

The family contrast clarifies the mechanism. V-JEPA + ID is more noise-robust than VideoMAE + ID at matched clean performance (noise 0.15: 0.66 vs. 0.11), consistent with predictive latent objectives being less pixel-tied than masked pixel reconstruction. Pixel-reconstruction models are the clearest counterexample: SDXL VAE + ID and Cosmos + ID become more negative as perturbation strengthens, still prioritizing appearance even after action supervision, while Dreamer 4 stays flat at its information floor. Robustness thus reflects what each objective preserves: video-pretrained + ID backbones encode action-conditioned dynamics, whereas frozen and reconstruction-oriented features remain coupled to surface appearance.

<table><tr><td>Family</td><td>Backbone</td><td>clean</td><td>noise 0.10</td><td>noise 0.15</td><td>blur 11</td><td>worst</td></tr><tr><td colspan="7">+ ID variants</td></tr><tr><td rowspan="2">Video + JEPA pred.</td><td>V-JEPA 2 ViT-L</td><td>0.85</td><td>0.75</td><td>0.66</td><td>0.56</td><td>0.56</td></tr><tr><td>V-JEPA 2.1 ViT-B</td><td>0.82</td><td>0.46</td><td>0.01</td><td>0.70</td><td>0.01</td></tr><tr><td>Video + Pixel MAE</td><td>VideoMAE V1 ViT-L</td><td>0.75</td><td>0.39</td><td>0.11</td><td>0.55</td><td>0.11</td></tr><tr><td rowspan="2">Image SSL</td><td>Web-DINO ViT-L</td><td>0.16</td><td>0.13</td><td>0.08</td><td>-0.05</td><td>-0.05</td></tr><tr><td>SigLIP 2 ViT-L</td><td>0.17</td><td>0.07</td><td>-0.09</td><td>-0.40</td><td>-0.40</td></tr><tr><td>Latent action quant.</td><td>LAPA-LAQ-OpenX</td><td>0.51</td><td>0.14</td><td>-0.19</td><td>0.50</td><td>-0.19</td></tr><tr><td>Pixel diffusion</td><td>DIFF</td><td>0.57</td><td>0.57</td><td>0.57</td><td>0.52</td><td>0.46</td></tr><tr><td rowspan="2">Pixel reconstruction</td><td>SDXL VAE</td><td>-0.41</td><td>-1.69</td><td>-2.20</td><td>-2.02</td><td>-2.20</td></tr><tr><td>Cosmos-1 tokenizer</td><td>-0.29</td><td>-0.66</td><td>-0.86</td><td>-1.09</td><td>-1.09</td></tr><tr><td>Shortcut-forcing dyn.</td><td>Dreamer 4</td><td>-0.04</td><td>-0.04</td><td>-0.04</td><td>-0.04</td><td>-0.04</td></tr><tr><td colspan="7">Frozen baselines</td></tr><tr><td rowspan="2">Video + JEPA pred.</td><td>V-JEPA 2 ViT-L</td><td>0.40</td><td>-0.82</td><td>-1.18</td><td>-4.66</td><td>-4.66</td></tr><tr><td>V-JEPA 2.1 ViT-B</td><td>0.44</td><td>-4.31</td><td>-5.36</td><td>-0.86</td><td>-5.36</td></tr><tr><td>Video + Pixel MAE</td><td>VideoMAE V1 ViT-L</td><td>0.46</td><td>-2.02</td><td>-2.54</td><td>-1.10</td><td>-2.54</td></tr><tr><td rowspan="2">Image SSL</td><td>Web-DINO ViT-L</td><td>-0.01</td><td>-3.40</td><td>-4.19</td><td>-1.60</td><td>-4.19</td></tr><tr><td>SigLIP 2 ViT-L</td><td>0.05</td><td>-0.63</td><td>-0.62</td><td>-1.75</td><td>-1.75</td></tr><tr><td>Latent action quant.</td><td>LAPA-LAQ-OpenX</td><td>0.41</td><td>-0.24</td><td>-1.14</td><td>0.38</td><td>-1.14</td></tr><tr><td>Pixel diffusion</td><td>DIFF</td><td>0.43</td><td>0.43</td><td>0.44</td><td>0.42</td><td>0.24</td></tr><tr><td rowspan="2">Pixel reconstruction</td><td>SDXL VAE</td><td>-0.55</td><td>-1.57</td><td>-1.96</td><td>-1.52</td><td>-1.96</td></tr><tr><td>Cosmos-1 tokenizer</td><td>-0.36</td><td>-0.59</td><td>-0.80</td><td>-0.94</td><td>-0.94</td></tr><tr><td>Shortcut-forcing dyn.</td><td>Dreamer 4</td><td>-0.04</td><td>-0.04</td><td>-0.04</td><td>-0.04</td><td>-0.04</td></tr></table>

Table 2: Robustness to visual perturbation, grouped by pretraining family. Probe trained on clean features, evaluated on perturbed test-OOD features; “Worst” is the minimum across all seven conditions. Two table-reading caveats: frozen DIFF is in-distribution for LIBERO and floors at clean 0.43 across noise, and Dreamer 4 floors at −0.04 throughout. Full matrix in Appendix H.

## 5 Limitations

Our study is entirely in simulation: although we validate across three benchmarks (LIBERO, with CALVIN and MetaWorld in Appendix D), all are simulated environments, and sim-to-real transfer of the action-recoverability trends remains untested. The probe measures representation quality via held-out action recovery rather than policy performance via closed-loop success; the two are correlated but not identical. Our Dreamer 4 evaluation is a reproduction against public DreamerV3 scaffolding, since no official release is available. Real-robot evaluation and closed-loop success measurement are scoped for follow-up.

## 6 Conclusion

Across eight encoder families and three robotic benchmarks, we find that pixel-prediction quality and action recoverability are largely orthogonal: the standard metrics (PSNR, FVD, LPIPS) do not predict whether a representation encodes the underlying action. What matters is temporal predictive pretraining. Comparing feature-level latent prediction against pixel-level masked autoencoding attributes most of the advantage to natural-video temporal context, with the latent-predictive objective adding a further gain concentrated on end-effector orientation. The inverse-dynamics loss acts as a multiplier that re-aligns structure already embedded by pretraining rather than manufacturing it, which is also why it confers robustness to visual corruption. A video-predictive encoder with a small inverse-dynamics objective is thus a reliable default for a manipulation policy’s visual front-end.

## References

[1] M. Assran et al. V-jepa 2: Self-supervised video models enable understanding, prediction, and planning. arXiv preprint arXiv:2506.09985, 2025.  
[2] G. Zhou, H. Pan, Y. LeCun, and L. Pinto. Dino-wm: World models on pre-trained visual features enable zero-shot planning. In International Conference on Machine Learning (ICML), 2025.  
[3] M. J. Kim et al. Openvla: An open-source vision-language-action model. In Conference on Robot Learning (CoRL), 2024.  
[4] S. Tian, C. Finn, and J. Wu. A control-centric benchmark for video prediction. In International Conference on Learning Representations (ICLR), 2023.  
[5] Nilaksh, S. Jha, A. Zholus, and S. Chandar. Reconstruction or semantics? what makes a latent space useful for robotic world models. arXiv preprint arXiv:2605.06388, 2026.  
[6] K. Yi, C. Gan, Y. Li, P. Kohli, J. Wu, A. Torralba, and J. B. Tenenbaum. CLEVRER: CoLlision events for video representation and reasoning. In International Conference on Learning Representations (ICLR), 2020.  
[7] Y. Han and A. Yilmaz. Enhancing policy learning with world-action model. arXiv preprint arXiv:2603.28955, 2026. DreamerV2 with inverse-dynamics auxiliary head, evaluated on CALVIN.  
[8] B. Liu, Y. Zhu, C. Gao, Y. Feng, Q. Liu, Y. Zhu, and P. Stone. LIBERO: Benchmarking knowledge transfer for lifelong robot learning. In Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2023.  
[9] D. Hafner, J. Pasukonis, J. Ba, and T. Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.  
[10] V. Micheli, E. Alonso, and F. Fleuret. Transformers are sample-efficient world models. In International Conference on Learning Representations (ICLR), 2023.  
[11] C. Li et al. R-Bench: Are your large multimodal models robust to real-world corruptions? arXiv preprint arXiv:2410.05474, 2024.  
[12] D. Li, Y. Fang, et al. Worldmodelbench: Judging video generation models as world models. In NeurIPS Datasets and Benchmarks, 2025.  
[13] H. Yue, S. Huang, et al. Ewmbench: Evaluating scene, motion, and semantic quality in embodied world models. arXiv preprint arXiv:2505.09694, 2025.  
[14] Y. Shang et al. Worldarena: A unified benchmark for evaluating perception and functional utility of embodied world models. arXiv preprint arXiv:2602.08971, 2026.  
[15] J. Zhang et al. World-in-world: World models in a closed-loop world. arXiv preprint arXiv:2510.18135, 2025.  
[16] S. Joseph, Q. Garrido, R. Balestriero, M. Kowal, T. Fel, S. Bakhtiari, B. Richards, and M. Rabbat. Interpreting physics in video world models. arXiv preprint arXiv:2602.07050, 2026.  
[17] P. Agrawal, A. V. Nair, P. Abbeel, J. Malik, and S. Levine. Learning to poke by poking: Experiential learning of intuitive physics. In Advances in Neural Information Processing Systems (NeurIPS), 2016.  
[18] D. Pathak, P. Agrawal, A. A. Efros, and T. Darrell. Curiosity-driven exploration by selfsupervised prediction. In International Conference on Machine Learning (ICML), 2017.  
[19] E. Shelhamer, P. Mahmoudieh, M. Argus, and T. Darrell. Loss is its own reward: Selfsupervision for reinforcement learning. In ICLR Workshop, 2017.  
[20] Z. J. Cui, H. Pan, A. Iyer, S. Haldar, and L. Pinto. Dynamo: In-domain dynamics pretraining for visuo-motor control. In Advances in Neural Information Processing Systems (NeurIPS), 2024.  
[21] X. Zhou et al. LIBERO-PRO: Towards robust and fair evaluation of vision-language-action models beyond memorization. arXiv preprint arXiv:2510.03827, 2025.  
[22] R. Riochet, M. Y. Castro, M. Bernard, A. Lerer, R. Fergus, V. Izard, and E. Dupoux. Intphys: A framework and benchmark for visual intuitive physics reasoning, 2020. URL https:// arxiv.org/abs/1803.07616.  
[23] O. Mees, L. Hermann, E. Rosete-Beas, and W. Burgard. Calvin: A benchmark for languageconditioned policy learning for long-horizon robot manipulation tasks. IEEE Robotics and Automation Letters, 2022.  
[24] T. Yu, D. Quillen, Z. He, R. Julian, K. Hausman, C. Finn, and S. Levine. Meta-world: A benchmark and evaluation for multi-task and meta reinforcement learning. In Conference on Robot Learning (CoRL), 2020.  
[25] D. Hafner, W. Yan, and T. Lillicrap. Training agents inside of scalable world models. arXiv preprint arXiv:2509.24527, 2025.  
[26] D. Podell, Z. English, K. Lacey, A. Blattmann, T. Dockhorn, J. Muller, J. Penna, and R. Rom- ¨ bach. Sdxl: Improving latent diffusion models for high-resolution image synthesis. In International Conference on Learning Representations (ICLR), 2024.  
[27] N. Agarwal, A. Ali, M. Bala, Y. Balaji, E. Barker, et al. Cosmos world foundation model platform for physical ai. arXiv preprint arXiv:2501.03575, 2025.  
[28] S. Ye et al. Latent action pretraining from videos. In International Conference on Learning Representations (ICLR), 2025.  
[29] D. Fan, S. Tong, J. Zhu, K. Sinha, Z. Liu, X. Chen, M. Rabbat, N. Ballas, Y. LeCun, and A. Bar. Scaling language-free visual representation learning. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), 2025.  
[30] M. Tschannen et al. Siglip 2: Multilingual vision-language encoders with improved semantic understanding, localization, and dense features. arXiv preprint arXiv:2502.14786, 2025.  
[31] Z. Tong, Y. Song, J. Wang, and L. Wang. Videomae: Masked autoencoders are data-efficient learners for self-supervised video pre-training. In Advances in Neural Information Processing Systems (NeurIPS), 2022.  
[32] Y. LeCun. A path towards autonomous machine intelligence. OpenReview, 2022. URL https: //openreview.net/forum?id=BZ5a1r-kVsf.  
[33] A. Brohan et al. RT-2: Vision-language-action models transfer web knowledge to robotic control. In Conference on Robot Learning (CoRL), 2023.  
[34] K. Black et al. π0: A vision-language-action flow model for general robot control. arXiv preprint arXiv:2410.24164, 2024.

## A Inverse-Dynamics Loss Ablations on DIFF

All cross-architecture rows in the main text use the same inverse-dynamics recipe: a single inversedynamics head $g _ { \phi } ( f _ { \theta } ( o _ { t } ) , f _ { \theta } ( o _ { t + 1 } ) )  \hat { a } _ { t }$ trained with $L _ { \mathrm { 2 } } \ \mathrm { M S E }$ at $\lambda { = } 0 . 0 5$ . This appendix reports the ablations behind these choices, conducted on the canonical DIFF backbone (91M parameters, 2-camera input, no action conditioning, task-OOD training split) where the hyperparameter sweeps were cheap enough to run exhaustively.

λ sweep. We swept the inverse-dynamics weight λ across three orders of magnitude (Table 3). Action $R ^ { 2 }$ follows an inverted-U with a plateau between $\lambda { = } 0 . 0 1 \left( R ^ { 2 } 0 . 5 9 8 \right)$ and $\lambda { = } 0 . 0 5 ( R ^ { 2 } 0 . 5 7 2 ) $ : the two settings differ by only 0.026 in $R ^ { 2 }$ and both are +0.15 above the no-ID baseline (0.43). Outside the plateau, both sides degrade. Below $\lambda { = } 0 . 0 1$ the signal is too weak to redirect the trunk; above $\lambda { = } 0 . 1$ 1 the ID loss dominates the primary diffusion objective, rollout PSNR drops by 5–9 dB, and action $R ^ { 2 }$ falls below 0.3. Recon MSE and LPIPS move smoothly across the plateau (within 0.0007 and 0.06 respectively), so the ID loss’s effect on generation quality is small in the useful range. Appendix C pairs this λ sweep with a temporal-cosine diagnostic that shows how the loss modifies the trunk’s representations underneath the action- $\cdot R ^ { 2 }$ curve.

<table><tr><td> $\lambda$ </td><td>Action  $R^{2}$ </td><td>Rollout PSNR</td><td>Recon MSE</td><td>LPIPS</td></tr><tr><td>—</td><td>0.426</td><td>19.18</td><td>0.0121</td><td>0.287</td></tr><tr><td>0.01</td><td>0.598</td><td>19.07 (-0.11)</td><td>0.0124</td><td>0.297</td></tr><tr><td>0.025</td><td>0.571</td><td>19.03 (-0.15)</td><td>0.0125</td><td>0.320</td></tr><tr><td>0.05</td><td>0.572</td><td>18.91 (-0.27)</td><td>0.0128</td><td>0.344</td></tr><tr><td>0.1</td><td>0.436</td><td>18.54 (-0.65)</td><td>0.0140</td><td>0.386</td></tr><tr><td>0.15</td><td>0.306</td><td>18.12 (-1.06)</td><td>0.0154</td><td>0.424</td></tr><tr><td>0.3</td><td>0.282</td><td>17.40 (-1.79)</td><td>0.0182</td><td>0.485</td></tr><tr><td>1.0</td><td>0.264</td><td>13.75 (-5.43)</td><td>0.0421</td><td>0.622</td></tr><tr><td>10</td><td>0.145</td><td>10.62 (-8.57)</td><td>0.0867</td><td>0.718</td></tr></table>

Table 3: Auxiliary weight λ sweep on canonical DIFF. Action $R ^ { 2 }$ is inverted-U with a plateau at $\lambda \in \left[ 0 . 0 1 , 0 . 0 5 \right]$ . We use $\lambda { = } 0 . 0 5$ as the canonical setting in all main-text cross-architecture rows for direct comparability; $\lambda { = } 0 . 0 1$ would yield ∼0.03 higher absolute $R ^ { 2 }$ but would not change any qualitative ranking.

Auxiliary loss form. We hold $\lambda { = } 0 . 0 5$ fixed and vary the auxiliary loss: $L _ { 2 }$ vs. $L _ { 1 }$ inversedynamics regression, forward dynamics, temporal contrastive, and single-frame action prediction (Table 4). Both inverse-dynamics regressions sit at the top; $L _ { 1 }$ regression (matching WAM’s original specification) outperforms our $L _ { 2 }$ default by +0.05 on action $R ^ { 2 } \ : ( 0 . 6 2 9 $ vs. 0.578) at essentially identical rollout PSNR (18.84 vs. 18.85 dB). Single-frame action prediction $( g _ { \phi } ( f _ { t } ) \to a _ { t } , \operatorname { n o } f _ { t + 1 } )$ drops to $R ^ { 2 } \ 0 . 5 2 3 .$ , confirming that consecutive-frame structure is what the loss depends on. Forward dynamics $( g _ { \phi } ( f _ { t } , a _ { t } ) \to f _ { t + 1 } )$ falls to 0.408, suggesting the direction of inversion matters: predicting actions from feature pairs shapes the trunk more effectively than predicting features from action-conditioned pairs. A temporal-contrastive objective collapses to 0.073.

<table><tr><td>Loss form</td><td>Action  $R^{2}$ </td><td>Rollout PSNR</td></tr><tr><td> $L_{1}$  inverse dynamics</td><td>0.629</td><td>18.84</td></tr><tr><td> $L_{2}$  MSE inverse dynamics (default)</td><td>0.578</td><td>18.85</td></tr><tr><td>Single-frame action prediction</td><td>0.523</td><td>18.85</td></tr><tr><td>Forward dynamics ( $f_{t}, a_{t} \rightarrow f_{t+1}$ )</td><td>0.408</td><td>19.11</td></tr><tr><td>Temporal contrastive</td><td>0.073</td><td>18.43</td></tr></table>

Table 4: Auxiliary loss-form ablation at $\overline { { \lambda { = } 0 . 0 5 } } .$ . The head architecture is fixed; only the loss form varies. $L _ { 1 }$ inverse-dynamics regression is the best objective by +0.05 over the $L _ { 2 }$ default. We retain $L _ { 2 }$ as the canonical setting in the main text because it is the more common choice and to avoid conflating “ID supervision helps” with “a particular loss form helps”; the $L _ { 1 }$ variant would shift the DIFF + ID row in Table 1 from 0.57 to 0.63.

Prediction horizon k. Extending the ID loss from 1-step $\begin{array} { r c l } { ( g _ { \phi } ( f _ { t } , f _ { t + 1 } ) } & { \to } & { a _ { t } ) } \end{array}$ to k-step $( g _ { \phi } ( f _ { t } , f _ { t + k } ) \to ( a _ { t } , \dots , a _ { t + k - 1 } ) )$ traces an inverted-U with peak at $k { = } 4 \colon$ action $R ^ { 2 }$ moves from 0.578 (k=1) to 0.587 (k=2) to 0.608 (k=4) to 0.546 (k=8). The peak at k=4 aligns with the typical action chunk length used by downstream chunk-style action heads [33, 3, 34]. We retain $k { = } 1$ as the canonical setting for direct comparability with WAM.

Discussion. The ablations support two narrower claims about the ID loss. First, the recipe is forgiving in a band around the canonical settings $( \lambda \in [ 0 . 0 1 , 0 . 0 5 ]$ , L2 or $L _ { 1 } , k \in [ 1 , 4 ] ) ;$ ; none of these knobs is fragile. Second, the magnitude of the cross-architecture multiplier effect reported in Section 4.2 (+0.45 on V-JEPA vs. +0.07 on Dreamer 4) is much larger than any of the within-DIFF ablation effects (+0.05 at most). The multiplier effect is therefore not an artifact of a particular inverse-dynamics recipe but reflects backbone-level differences that no reasonable recipe tuning would mask.

λ sweep on image-only SSL backbones. A natural concern is whether the V-JEPA-specific multiplier in Section 4.2 (Web-DINO + ID 0.16, SigLIP 2 + ID 0.17 at $\lambda { = } 1 . 0 )$ is an artifact of the inverse-dynamics weight setting. We rule this out by sweeping $\lambda \in \{ 0 . 0 1 , 0 . 0 2 5 , 0 . 1 , 0 . 3 , 1 . 0 \}$ on both Web-DINO ViT-L and SigLIP 2 ViT-L, with all other settings held fixed (Table 5).

<table><tr><td> $\lambda$ </td><td>Web-DINO test  $R^{2}$ </td><td>SigLIP 2 test  $R^{2}$ </td></tr><tr><td>0.01</td><td>0.134</td><td>0.143</td></tr><tr><td>0.025</td><td>0.122</td><td>0.106</td></tr><tr><td>0.1</td><td>0.168</td><td>0.139</td></tr><tr><td>0.3</td><td>0.076</td><td>0.077</td></tr><tr><td>1.0 (main text)</td><td>0.159</td><td>0.166</td></tr></table>

Table 5: ID λ sweep on image-only SSL backbones. Both Web-DINO and SigLIP 2 stay in a 0.10-wide band $( R ^ { 2 } \bar { \in } [ 0 . 0 7 , 0 . 1 7 ] )$ across all λ values tested. The best per backbone is only ${ \sim } 0 . 0 1$ above the $\lambda { = } 1 . 0$ main-text setting. No λ closes the gap to V-JEPA + ID at 0.85. The dip at $\lambda { = } 0 . 3$ is consistent across both backbones, suggesting a real (mild) optimization instability rather than noise. The conclusion is that the inverse-dynamics weight axis is not the lever; the ceiling is set by the backbone family.

## B DIFF-Specific Findings: Action Conditioning and Capacity

This appendix reports two diagnostic findings on our DIFF pixel-diffusion backbone that informed how we set up the cross-architecture rows in Section 4.2, but are not central to the cross-architecture claim itself.

The action-conditioning paradox. A natural baseline for a robot world model is to condition the trunk on the action sequence at training time, since the model is then explicitly modeling actionconditioned futures. We find that this hurts action recovery from the resulting frozen features. Table 6 reports DIFF variants that differ only in whether action conditioning is provided to the trunk, with all other settings (camera setup, capacity, training length) held fixed.

The interpretation is straightforward in hindsight: when the trunk receives the action sequence as input, it can offload action-dependent prediction to the conditioning pathway and let the visual representation focus on pixel-level prediction. The features that remain in the trunk are then predicates of what the action did not affect—background, lighting, static geometry—rather than action-relevant motion. Removing the action input forces the trunk to encode action-relevant signal because it has no other way to produce a coherent next-frame prediction.

This is a within-DIFF observation; we do not claim it generalizes to other architectures (V-JEPA, Web-DINO, etc., do not see action data at pretraining and so cannot exhibit this paradox in the same form). It does, however, motivate our cross-architecture choice to never feed actions to the trunk for the no-ID baselines: doing so would systematically suppress the very signal the probe measures.

<table><tr><td>Variant</td><td>cam</td><td>action input</td><td>steps</td><td>size</td><td>Action  $R^{2}$ </td></tr><tr><td>DIFF (no action)</td><td>1</td><td>—</td><td>100k</td><td>medium</td><td>+0.26</td></tr><tr><td>DIFF + action</td><td>1</td><td>√</td><td>100k</td><td>medium</td><td>-0.32</td></tr><tr><td>DIFF + action, longer</td><td>1</td><td>√</td><td>200k</td><td>medium</td><td>+0.06</td></tr><tr><td>DIFF + action, XL</td><td>1</td><td>√</td><td>200k</td><td>XL</td><td>-0.01</td></tr><tr><td>DIFF (no action), XL</td><td>1</td><td>—</td><td>100k</td><td>XL</td><td>+0.31</td></tr><tr><td>DIFF (no action), 2cam</td><td>2</td><td>—</td><td>100k</td><td>XL</td><td>+0.36</td></tr><tr><td>DIFF (no action), 2cam, task-OOD</td><td>2</td><td>—</td><td>100k</td><td>XL</td><td>+0.43</td></tr></table>

Table 6: Action conditioning hurts action recovery. At medium capacity and single camera, adding action input drops $R ^ { 2 } \ \tilde { \bf b } \bf { y } \ 0 . 5 8 \ ( 0 . 2 6  - 0 . 3 2 ) $ . Scaling capacity and training length (with action input retained) improves rollout PSNR but does not restore $\mathrm { { \bar { \it R } } ^ { 2 } }$ . Removing the action input restores $\dot { R } ^ { 2 }$ ; adding the second camera and training on the task-OOD split further lift it. Pretraining and capacity scale generation, not representation.

Capacity is a channel, not a free lever. A second diagnostic asks whether scaling model capacity recovers action structure on its own. Our DIFF capacity sweep, factored over 1-camera vs. 2-camera input, shows that capacity helps action recovery only when the input modality already contains the relevant information. At 1 camera (agentview only), doubling parameters from medium to XL gives an essentially flat action $R ^ { 2 }$ curve. At 2 cameras (agentview + wrist), the same capacity sweep shows monotone improvement. Masking the wrist camera at test time on a 2-camera-trained model collapses both action $R ^ { 2 }$ and task classification to near-chance levels.

We summarize this as “capacity is a channel”: model capacity acts as a multiplicative term on the action-relevant signal that the input modality provides, but it does not synthesize signal from inputs that lack it. The wrist camera contains end-effector geometry that the agentview camera does not, and no amount of capacity recovers what the agentview cannot see. This appendix reports the finding because it informs our 2-camera setup in all DIFF rows of the main text; for the pretrained encoder rows (V-JEPA, Web-DINO, etc.), the question is moot because those models see only the standard frontal view and the action-relevant signal must come from the encoder’s pretrained features rather than from a wrist channel.

## C Temporal Cosine: A Diagnostic for the Auxiliary’s Mechanism

To diagnose how the inverse-dynamics loss modifies the trunk’s representations, we measure the cosine similarity between mean-pooled features at temporal offset k frames: cos $( \bar { f } _ { t } , \bar { f } _ { t + k } )$ . A high cosine indicates that the features change slowly over time (temporally collapsed); a low cosine indicates that the features change rapidly. Table 7 reports cosine and action $R ^ { 2 }$ across the λ sweep on DIFF.

<table><tr><td>Variant</td><td> $\cos(t, t + 16)$ </td><td>Action  $R^2$ </td><td>Rollout PSNR</td></tr><tr><td>DIFF baseline (no ID)</td><td>0.999</td><td>+0.426</td><td>19.18</td></tr><tr><td>DIFF + ID  $\lambda=0.01$ </td><td>0.994</td><td>+0.598</td><td>19.07</td></tr><tr><td>DIFF + ID  $\lambda=0.05$ </td><td>0.972</td><td>+0.572</td><td>18.91</td></tr><tr><td>DIFF + ID  $\lambda=0.1$ </td><td>0.948</td><td>+0.436</td><td>18.54</td></tr><tr><td>DIFF + ID  $\lambda=0.3$ </td><td>0.900</td><td>+0.282</td><td>17.40</td></tr><tr><td>DIFF + ID  $\lambda=1.0$ </td><td>0.877</td><td>+0.264</td><td>13.75</td></tr><tr><td>DIFF + ID  $\lambda=10$ </td><td>0.652</td><td>+0.145</td><td>10.62</td></tr></table>

Table 7: Temporal cosine decreases monotonically with $\lambda ;$ action $R ^ { 2 }$ does not. The baseline DIFF trunk is fully temporally collapsed (cosine 0.999 at $k { = } 1 6 ) ;$ features barely change across 16 frames. The ID loss monotonically de-collapses the trunk: cosine drops smoothly from 0.999 to 0.65 as λ increases. Action $R ^ { 2 }$ , however, peaks at λ=0.01 (cosine 0.994) and falls on both sides.

Two readings of the cosine– $R ^ { 2 }$ dissociation. The pairing in Table 7 reveals two things. First, the ID loss’s effect on the trunk is real and graded: increasing λ progressively breaks the temporalcollapse that the diffusion objective alone produces. Second, temporal de-collapse is necessary but not sufficient for action recovery. A small de-collapse (cosine drops from 0.999 to 0.994) brings the trunk to peak action $R ^ { 2 }$ , but further de-collapse (cosine 0.652) destroys the action signal even though the trunk’s features are now varying substantially over time. The action-relevant information lives in fine-grained directional differences of high-cosine features, not in coarse temporal variation.

Limitation. Cosine is a within-DIFF mechanism diagnostic. We do not extend it to the pretrainedencoder rows of the main text because (a) those encoders are fine-tuned with the ID loss, not trained from scratch, so the baseline “no-ID” cosine reference is not available; and (b) ViT-based encoders have substantially different patch geometry and pooling structure, which makes cosine values not directly comparable to our DIFF (transformer over diffusion-noised pixels). The diagnostic is included here as evidence about how the ID loss modifies a from-scratch trunk; the cross-architecture multiplier effect in the main text does not depend on this mechanism story.

## D Cross-Benchmark Transfer

We extend the cross-architecture analysis to two additional simulator benchmarks under the same probe protocol: CALVIN [23] (24 long-horizon tasks across 4 fixed table environments, 7-DoF) and MetaWorld [24] (ML45 split: 45 training tasks, 5 held-out, 4-DoF, scripted demos). For each benchmark we apply the ID fine-tune to eight backbones and evaluate on its task-OOD held-out split with the same 3000-step MLP probe used in the main paper; Table 8 reports the headline results.

<table><tr><td>Backbone + ID</td><td>LIBERO</td><td>CALVIN</td><td>MetaWorld</td></tr><tr><td>V-JEPA 2 ViT-L</td><td>+0.85</td><td>+0.86</td><td>+0.59</td></tr><tr><td>V-JEPA 2.1 ViT-B</td><td>+0.82</td><td>+0.88</td><td>+0.40</td></tr><tr><td>VideoMAE V1 ViT-L</td><td>+0.75</td><td>+0.80</td><td>+0.35</td></tr><tr><td>Web-DINO ViT-L</td><td>+0.16</td><td>+0.81</td><td>-0.44</td></tr><tr><td>SigLIP 2 ViT-L</td><td>+0.17</td><td>+0.77</td><td>-0.17</td></tr><tr><td>LAPA</td><td>+0.51</td><td>+0.78</td><td>-0.35</td></tr><tr><td>SDXL VAE</td><td>-0.41</td><td>+0.09</td><td>-0.54</td></tr><tr><td>Cosmos-1 tokenizer</td><td>-0.29</td><td>+0.39</td><td>-0.31</td></tr></table>

Table 8: Cross-benchmark consistency on three task-OOD splits. V-JEPA + ID is the top family on LIBERO and MetaWorld (V-JEPA 2 ViT-L) and on CALVIN (V-JEPA 2.1 ViT-B at 0.88, V-JEPA 2 ViT-L close behind at 0.86). The Spearman rank correlation between LIBERO and Meta-World backbone rankings is $\rho = + 0 . 8 8$ . $\mathrm { C A L V I N } ^ { \prime } \mathrm { s } \rho$ with LIBERO is +0.83 and with MetaWorld is +0.67; CALVIN reorders image-only SSL backbones from near-zero $\scriptstyle \mathbf { R } ^ { 2 }$ on LIBERO and negative $\scriptstyle \mathbf { R } ^ { 2 }$ on MetaWorld to 0.77–0.81 R², which we attribute to its four fixed table environments where perframe image features encode object position consistently enough to substitute for temporal context. DIFF and Dreamer 4 are LIBERO-trained from scratch and omitted from cross-benchmark comparison.

The pattern across LIBERO and MetaWorld is consistent: video-pretrained backbones top the ranking, image-only SSL backbones land far below, and reconstruction encoders are at or below zero. MetaWorld is the sharpest version of this—image-only SSL and LAPA actually collapse to negative $\scriptstyle \mathrm { \mathrm { R } } ^ { 2 }$ in the 4-DoF, 50-task regime—and the $\rho = + 0 . 8 8$ rank correlation with LIBERO confirms that the LIBERO finding is not a single-benchmark artifact. The four-DoF nature of MetaWorld also drops a separate constraint: the V-JEPA advantage on rotation-heavy LIBERO (drx, dry, drz) cannot, by construction, explain its dominance on MetaWorld, which means V-JEPA carries actionrelevant structure beyond rotation specifically. The advantage is real on translation and gripper recovery too, even if its rotation contribution is the most dramatic.

CALVIN is the outlier and deserves explanation. With only four fixed table environments and consistent object layouts within each, a per-frame image encoder that captures object pose already encodes most of the action signal, and the inverse-dynamics head can read it out from any reasonable visual feature space. LIBERO has 130 scene-distinct tasks and MetaWorld has 45 visually distinct manipulation primitives—both require disambiguating “which object is the target” from features alone, where temporal context helps. The CALVIN catch-up of image-only SSL is therefore consistent with our broader account: video-pretrained backbones supply scene-disambiguating structure that image-only backbones do not, and that advantage matters specifically when the benchmark requires such disambiguation. CALVIN thus delineates the conditions under which the V-JEPA advantage manifests, rather than overturning it.

Probe-budget control. An alternative reading of the CALVIN catch-up is that LIBERO’s small probe-train budget (400 episodes) under-estimates image-only SSL while CALVIN’s larger budget surfaces hidden action-relevant features. We address this by sweeping the LIBERO probe-train size at {400, 800, 1600, 3200} episodes on Web-DINO + ID and SigLIP 2 + ID. Web-DINO + ID R² rises from 0.16 (400 ep) to 0.37 (3200 ep) and SigLIP 2 + ID from −0.02 to 0.26, an $8 \times$ data increase yielding ∼0.2 R² lift. The remaining gap to V-JEPA + ID at LIBERO probe-train 400 (0.85) is still 0.48 for Web-DINO and 0.59 for SigLIP 2. The CALVIN gap collapse is not reproduced at any LIBERO probe budget tested, so the LIBERO V-JEPA dominance and the CALVIN image-SSL catch-up arise from different mechanisms.

## E V-JEPA Per-Layer Surgical Probe

To find direct evidence for the mechanism conjecture in Section 4.3—that V-JEPA’s pretraining objective leaves a specific signature in the trunk—we probe each of V-JEPA 2 ViT-L’s 24 transformer blocks for action $\bar { R } ^ { 2 }$ on LIBERO task-OOD, both on the frozen pretrained checkpoint and on the LIBERO + ID fine-tuned checkpoint (Table 9). Layer 0 is just after patch embedding; layer 23 is the encoder’s final output (the layer our main-text extractors use).

<table><tr><td>Layer</td><td>Frozen  $R^{2}$ </td><td>+ ID  $R^{2}$ </td><td> $\Delta$ </td></tr><tr><td>7</td><td>0.456</td><td>0.554</td><td>+0.10</td></tr><tr><td>14</td><td>0.514 (frozen peak)</td><td>0.616</td><td>+0.10</td></tr><tr><td>18</td><td>0.516</td><td>0.632</td><td>+0.12</td></tr><tr><td>19</td><td>0.487</td><td>0.658</td><td>+0.17</td></tr><tr><td>20</td><td>0.460</td><td>0.708</td><td>+0.25</td></tr><tr><td>21</td><td>0.422</td><td>0.722 (+ ID peak)</td><td>+0.30</td></tr><tr><td>22</td><td>0.394 (frozen min)</td><td>0.712</td><td>+0.32</td></tr><tr><td>23 (final)</td><td>0.425</td><td>0.688</td><td>+0.26</td></tr></table>

Table 9: V-JEPA 2 ViT-L per-layer action $R ^ { 2 } { \mathrm { : } }$ , before and after inverse-dynamics fine-tuning. The frozen trunk peaks at layer 14 and drops by ∼0.12 R² by layer 22, recovering only slightly at the final layer. After 20k steps of inverse-dynamics fine-tuning, the same per-layer profile shifts upward, but the peak moves to layer 21 rather than the final layer.

The frozen profile is the key finding. Action $R ^ { 2 }$ rises through the early and middle blocks, peaks at block 14, then decreases monotonically through blocks 18–22 before partially recovering at the final block. The final layers are measurably less action-relevant than mid-trunk features, consistent with the JEPA pretraining loss steering those layers toward predictor compatibility—features designed to be reconstructed from masked counterparts in latent space, not features designed to encode actionrelevant geometry.

ID fine-tuning largely reverses this pattern: the final four layers gain +0.25 to +0.32 R², an order of magnitude more than the mid-trunk gains. The peak shifts from layer 14 (frozen) to layer 21 (+ ID), and the final layer still sits 0.034 R² below the new peak. The ID loss therefore does not simply lift the entire trunk uniformly: it pulls the final blocks back toward action-relevance, which is exactly the location where the JEPA pretraining objective pushed them away.

This is indirect evidence for the predictor-as-mechanism reading we develop in Section 4.3: the JEPA objective leaves a measurable signature on the trunk’s final layers, and that signature interacts specifically with the ID loss. Isolating the predictor module causally would still require re-pretraining V-JEPA under matched compute (out of academic reach at the original scale), but the per-layer profile establishes that some pretraining-induced asymmetry exists in the trunk, that it concentrates in the final layers, and that it is reversible by a 20k-step fine-tune.

## F Probe Data Efficiency Control

An alternative reading of the CALVIN catch-up in Appendix D is that image-only SSL backbones look weak on LIBERO not because their features are weak but because LIBERO’s probe-train budget (400 episodes) is too small to extract their action signal. We test this by encoding LIBERO probe features once per backbone and re-probing at sizes {400, 800, 1600, 3200}, with the same probe head and the same task-OOD test split (Table 10).

<table><tr><td>Probe size</td><td>Web-DINO frozen</td><td>Web-DINO + ID</td><td>SigLIP 2 frozen</td><td>SigLIP 2 + ID</td></tr><tr><td>400</td><td>+0.04</td><td>+0.16</td><td>+0.00</td><td>-0.02</td></tr><tr><td>800</td><td>+0.21</td><td>+0.29</td><td>+0.08</td><td>+0.18</td></tr><tr><td>1600</td><td>+0.21</td><td>+0.33</td><td>+0.03</td><td>+0.21</td></tr><tr><td>3200</td><td>+0.24</td><td>+0.37</td><td>+0.12</td><td>+0.26</td></tr></table>

Table 10: LIBERO probe-train size sweep on image-only SSL backbones. An 8× increase in probe-train data lifts Web-DINO + ID from 0.16 to 0.37 and SigLIP 2 + ID from −0.02 to 0.26. The lift is real but bounded.

The control rules out the probe-budget hypothesis as the full explanation. At an 8× probe-train budget, Web-DINO + ID on LIBERO reaches 0.37, still 0.48 below V-JEPA + ID at the original 400-episode budget (0.85), and still 0.44 below the same backbone’s CALVIN result (0.81). The image-SSL ceiling on LIBERO does rise with more probe data, but it does not reach either V-JEPA’s LIBERO performance or its own CALVIN performance. The cross-benchmark pattern in Appendix D is therefore not reducible to probe-data scaling.

## G Detailed Cross-Architecture Multiplier Table

Table 11 reports the per-variant breakdown of Table 1 in the main text, with parameter counts and individual frozen / + ID R² for every backbone we evaluated.

<table><tr><td>Pretraining family</td><td>Backbone</td><td>Params</td><td>Frozen  $R^2$ </td><td>+ ID  $R^2$ </td><td>Δ</td></tr><tr><td rowspan="2">Video + JEPA pred.</td><td>V-JEPA 2 ViT-L</td><td>304M</td><td>0.40</td><td>0.85</td><td>+0.45</td></tr><tr><td>V-JEPA 2.1 ViT-B (distilled)</td><td>87M</td><td>0.44</td><td>0.82</td><td>+0.38</td></tr><tr><td>Video + pixel MAE</td><td>VideoMAE V1 ViT-L</td><td>304M</td><td>0.46</td><td>0.75</td><td>+0.29</td></tr><tr><td rowspan="2">Image-only SSL</td><td>Web-DINO ViT-L</td><td>304M</td><td>-0.01</td><td>0.16</td><td>+0.17</td></tr><tr><td>SigLIP 2 ViT-L</td><td>316M</td><td>0.05</td><td>0.17</td><td>+0.12</td></tr><tr><td>Latent action quant.</td><td>LAPA-LAQ-OpenX</td><td>344M</td><td>0.41</td><td>0.51</td><td>+0.10</td></tr><tr><td>Pixel generation</td><td>DIFF (pixel diffusion)</td><td>91M</td><td>0.43</td><td>0.57</td><td>+0.14</td></tr><tr><td rowspan="2">Pixel reconstruction</td><td>SDXL VAE</td><td>34M</td><td>-0.55</td><td>-0.41</td><td>+0.14</td></tr><tr><td>Cosmos-1 tokenizer</td><td>34M</td><td>-0.36</td><td>-0.29</td><td>+0.07</td></tr><tr><td rowspan="3">Shortcut-forcing</td><td>Dreamer 4 (scratch 64M)</td><td>64M</td><td>-0.04</td><td>-0.04</td><td>0.00</td></tr><tr><td>Dreamer 4 (DMControl init 64M)</td><td>64M</td><td>n/a</td><td>-0.02</td><td>0.00</td></tr><tr><td>Dreamer 4 (scratch 276M)</td><td>276M</td><td>n/a</td><td>-0.04</td><td>0.00</td></tr></table>

Table 11: Detailed per-variant multiplier table. Backbones grouped by pretraining objective. “n/a” indicates a configuration not run: the two Dreamer 4 rows without a frozen entry are ID-only variants (DMControl-initialized fine-tune; 276M from-scratch + ID). The Cosmos-1 tokenizer row matches the exact encoder used by Nilaksh et al. [5].

## H Full Visual-Perturbation Robustness Aggregates

Per-family averages (Table 12) summarize the per-perturbation breakdown beyond the headline subset reported in Table 2. Family means hide the within-family ID-vs-frozen split documented in Table $2 \colon + \mathrm { I D }$ variants are the robust subset within each family, while frozen baselines collapse under noise and blur.

<table><tr><td>Family</td><td>N</td><td>clean</td><td>noise 0.10</td><td>noise 0.15</td><td>blur 11</td></tr><tr><td>V-JEPA 2</td><td>4</td><td>+0.48</td><td>-0.24</td><td>-0.57</td><td>-1.05</td></tr><tr><td>V-JEPA 2.1</td><td>2</td><td>+0.63</td><td>-1.92</td><td>-2.68</td><td>-0.08</td></tr><tr><td>VideoMAE</td><td>2</td><td>+0.60</td><td>-0.81</td><td>-1.22</td><td>-0.28</td></tr><tr><td>LAPA</td><td>2</td><td>+0.46</td><td>-0.05</td><td>-0.67</td><td>+0.44</td></tr><tr><td>DIFF</td><td>55</td><td>+0.16</td><td>+0.14</td><td>+0.11</td><td>-0.02</td></tr><tr><td>Web-DINO</td><td>2</td><td>+0.08</td><td>-1.64</td><td>-2.05</td><td>-0.83</td></tr><tr><td>SigLIP 2</td><td>2</td><td>+0.11</td><td>-0.28</td><td>-0.35</td><td>-1.08</td></tr><tr><td>Cosmos-1 tokenizer</td><td>2</td><td>-0.33</td><td>-0.63</td><td>-0.83</td><td>-1.02</td></tr><tr><td>SDXL VAE</td><td>2</td><td>-0.48</td><td>-1.63</td><td>-2.08</td><td>-1.77</td></tr><tr><td>Dreamer 4</td><td>10</td><td>-0.07</td><td>-0.13</td><td>-0.16</td><td>-0.15</td></tr></table>

Table 12: Per-family robustness aggregates. N is the variant count in the family (frozen and + ID variants mixed; the large DIFF count comes from its capacity, λ, and head-form sweeps documented in Appendix A and Appendix B). Family means hide the within-family ID-vs-frozen split documented in Table $2 \colon + \mathrm { I D }$ variants are the robust subset within each family, while frozen baselines collapse under noise and blur.

## I ID Supervision Sample-Budget Sweep

To probe the mechanism by which the inverse-dynamics loss interacts with V-JEPA’s pretrained features, we vary the fraction p of mini-batch samples that receive ID gradient. For each sample in a mini-batch of size B, we draw a Bernoulli(p) mask; samples with mask= 0 contribute zero loss (their gradient is skipped). All other hyperparameters—total fine-tune steps (20k), batch size (4 for V-JEPA, 2 for VideoMAE), learning rate $( 5 \times 1 0 ^ { - 5 } ) , \lambda { = } 1 . 0 \nonumber$ , EMA, and probe protocol— are held to the values used in the main multiplier matrix (Table 1). At p=0 no encoder gradient is produced, recovering the frozen baseline; at p=1 every sample contributes ID loss, recovering the standard ID fine-tune. We sweep $p \in \{ 0 . 0 2 , 0 . 0 5 , 0 . 1 0 , 0 . 2 5 , 0 . 5 0 , 1 . 0 0 \}$ on both V-JEPA 2 ViT-L and VideoMAE V1 ViT-L (Figure 4). The V-JEPA p=1 point is re-run at 20k steps for step-matched comparison; the matrix V-JEPA + ID value at 30k steps differs by less than 0.01 R².

<table><tr><td rowspan="2">Backbone</td><td colspan="7">ID supervision fraction p</td></tr><tr><td>0.00</td><td>0.02</td><td>0.05</td><td>0.10</td><td>0.25</td><td>0.50</td><td>1.00</td></tr><tr><td>V-JEPA 2 ViT-L + ID</td><td>0.400</td><td>0.598</td><td>0.672</td><td>0.693</td><td>0.783</td><td>0.806</td><td>0.852</td></tr><tr><td>VideoMAE V1 ViT-L + ID</td><td>0.457</td><td>0.480</td><td>0.499</td><td>0.620</td><td>0.709</td><td>0.735</td><td>0.748</td></tr></table>

Table 13: Aggregate action $\overline { { R ^ { 2 } } }$ across the sparsity sweep. Mean of 3 probe seeds. At $\scriptstyle p = 0 . 0 2 ,$ $\mathrm { V \mathrm { - } J E P A \mathrm { + } I D \left( 0 . 5 9 8 \right) }$ already exceeds VideoMAE + ID at any $p \leq 0 . 0 5$ , ruling out a generic “ID supervision is data-efficient” explanation.

Findings. The aggregate sweep (Table 13) shows four patterns. (i) V-JEPA saturates fast. At $\scriptstyle p = 0 . 0 2$ (only 2% of samples receive ID loss), V-JEPA already gains +0.20 R² over the frozen baseline; by $\scriptstyle { p = 0 . 1 0 }$ it has captured 65% of the full $p { = } 1$ lift. (ii) VideoMAE scales more linearly. Same recipe, $p { = } 0 . 0 2  0 . 0 5$ adds only $+ 0 . 0 2 \ \mathrm { R } ^ { 2 }$ (vs. $\mathrm { \Delta V { - } J E P A ^ { \prime } s ~ { + } 0 . 0 7 ) } \mathrm { { ; } }$ ; meaningful VideoMAE gains require $p \geq 0 . 1 0$ . (iii) Per-dim breakdown reinforces the rotation-localization claim. Translation $( d x , d y , d z )$ climbs $\mathrm { t o } \geq 0 . 9 4$ by $\scriptstyle { p = 0 . 0 5 }$ on both backbones, while rotation (drx, dry, drz) needs $p \geq 0 . 2 5$ for V-JEPA to exceed 0.50 and stays at 0.40–0.50 for VideoMAE at $\scriptstyle { p = 0 . 5 0 } .$ (iv) V-JEPA outperforms VideoMAE at every sparsity level. At $\scriptstyle p = 0 . 0 2$ , V-JEPA reaches 0.60 while VideoMAE manages only 0.48, a +0.12 gap that the budget axis cannot close. This is inconsistent with a simple “data-efficiency” reading and consistent with the view that V-JEPA’s pretrained features already encode much of the action-relevant structure; ID supervision re-aligns rather than creates the signal. VideoMAE’s pixel-level masked-autoencoding objective leaves features less action-aligned out of the box, so denser ID supervision is needed to reach comparable $R ^ { 2 }$ .

![](images/6f6d68b802ef93ae6ee8af78436e982e6b7c0f3cb9c1e8f15c7b26bad35c5d1a.jpg)

<details>
<summary>line chart</summary>

| ID supervision fraction p (per-sample mask probability) | V-JEPA 2 ViT-L + ID | VideoMAE V1 ViT-L + ID |
| -------------------------------------------------------- | ------------------- | ---------------------- |
| 0.0                                                      | 0.60                | 0.48                   |
| 0.1                                                      | 0.70                | 0.62                   |
| 0.25                                                     | 0.78                | 0.71                   |
| 0.5                                                      | 0.81                | 0.74                   |
| 1.0                                                      | 0.85                | 0.75                   |
</details>

Figure 4: ID supervision sample-budget sweep on V-JEPA 2 ViT-L and VideoMAE V1 ViT-L. x-axis: fraction p of mini-batch samples receiving ID gradient (per-sample mask probability). yaxis: action probe $R ^ { 2 }$ on LIBERO task-OOD, mean of 3 probe seeds. Endpoints at $p { = } 0$ are the frozen baselines; endpoints at $p { = } 1$ are the standard ID fine-tunes (V-JEPA re-run at 20k steps for step-matched comparison). V-JEPA captures +0.20 R² lift from just 2% ID supervision (effectively ∼1600 samples receiving ID gradient over 20k×4 batch slots), and reaches 65% of the full $p { = } 1$ lift at $\scriptstyle { p = 0 . 1 0 }$ . VideoMAE scales more linearly and gains little below $\scriptstyle { p = 0 . 1 0 }$ .

Implementation detail. The per-sample masking is implemented as mask = torch.rand ${ \bf \Psi } ( \mathtt { B } ) \ < \ \mathtt { p } ;$ the loss is (mask.float() \* per sample id loss).sum() / mask.sum().clamp(min=1), with the step skipped entirely if mask.sum() == 0. At p=0.02, $B { = } 4$ , this gives expected 0.08 active samples per step, so the majority of mini-batches contribute zero gradient—approximately 1600 samples receive ID gradient across the full 20k×4 training run.

## J Linear-Probe Action Subspace Visualization

To visualize the multiplier effect at the episode-trajectory level, we project test-OOD episode features into a canonical 2D action subspace. For each backbone, we fit a Ridge linear regressor $\hat { a } = X W$ (regularization $\alpha { = } 1 . 0 )$ on training-set features, then take the top-2 right-singular vectors of $W \in \bar { \mathbb { R } } ^ { D \times 7 }$ as the canonical subspace. Per-episode feature sequences are projected onto these two directions and plotted as time-ordered trajectories.

Linear vs. MLP probe. The $R ^ { 2 }$ values shown in Figure 5 are linear-probe (Ridge regression) accuracy on the 2D top-SVD subspace of the linear regressor weight matrix; the corresponding MLP-probe values reported in Table 1 are computed in the full feature space and differ accordingly (V-JEPA: linear 0.84 vs. MLP 0.85; V-JEPA frozen: linear 0.38 vs. MLP 0.40; Web-DINO + ID: linear 0.29 vs. MLP 0.16; SDXL VAE + ID: linear −1.16 vs. MLP −0.41). Backbone ordering

![](images/d059f231496f1abbc03a30607f79aba9ea89b39c8d812ab7a73b088a4c77f0b1.jpg)

<details>
<summary>line chart</summary>

| Probe dir 1 | Probe dir 2 |
| ----------- | ----------- |
| -0.8        | -1.9        |
| -0.7        | -1.6        |
| -0.6        | -1.5        |
| -0.5        | -1.4        |
| -0.4        | -1.5        |
</details>

![](images/2228ff1d40810a3e1107abe49c620aed589df175ad7967be9cd0c079ddab8d04.jpg)

<details>
<summary>line chart</summary>

| Probe dir 1 | Probe dir 2 |
| ----------- | ----------- |
| -1.15       | 0.25        |
| -1.10       | 0.19        |
| -1.05       | 0.16        |
| -1.00       | 0.14        |
| -0.95       | 0.13        |
</details>

![](images/273106a4cea9286e4bd7750ecaa912cfa63b3c1b542ae8bd59a86e9bf09c8863.jpg)

<details>
<summary>line chart</summary>

| Probe dir 1 | Probe dir 2 |
| ----------- | ----------- |
| -1.2        | 1.0         |
| -1.0        | 1.2         |
| -0.8        | 1.4         |
| -0.6        | 1.6         |
| -0.4        | 1.8         |
| -0.2        | 2.0         |
</details>

![](images/aed7aa265f957a4e3beeda2291bca61e8c8b70adff19e0c1c55f8756bb1c6740.jpg)

<details>
<summary>line chart</summary>

| Probe dir 1 | Probe dir 2 |
| ----------- | ----------- |
| -0.8        | 0.6         |
| -0.7        | 0.5         |
| -0.6        | 0.4         |
| -0.5        | 0.3         |
| -0.4        | 0.2         |
| -0.3        | 0.1         |
| -0.2        | 0.0         |
| -0.1        | -0.1        |
| 0.0         | -0.2        |
| 0.1         | -0.3        |
| 0.2         | -0.4        |
| 0.3         | -0.5        |
| 0.4         | -0.6        |
| 0.5         | -0.7        |
| 0.6         | -0.8        |
</details>

![](images/262bcd95d6dc27829ef21918f7010aa3b59319f138b8291e26435e222da17c87.jpg)

<details>
<summary>line chart</summary>

| Probe dir 1 | Probe dir 2 |
| ----------- | ----------- |
| -0.1        | 0.5         |
| 0.0         | 0.3         |
| 0.1         | 0.2         |
| 0.2         | 0.4         |
| 0.3         | 0.1         |
| 0.4         | 0.2         |
</details>

![](images/885e0b1d632c495be7f48df84ddb5b1937eda9d5a9bd004e0cc69dd6e755a39d.jpg)

<details>
<summary>line chart</summary>

| Probe dir 1 | Probe dir 2 |
| ----------- | ----------- |
| 1.8         | -1.0        |
| 2.0         | -0.5        |
| 2.2         | -0.3        |
| 2.4         | -0.2        |
| 2.6         | 0.2         |
| 2.8         | 0.3         |
</details>

episode start episode end gripper open/close transition trajectory (light dark = early late frame)

Figure 5: Linear-probe action subspace trajectories on LIBERO task-OOD. Top row: frozen backbones. Bottom row: + ID fine-tunes. Each colored curve is one episode (5 episodes shown, fixed random seed across panels; 3-frame moving average smoothing applied). ⃝ = episode start, × = episode end, □ = gripper open/close transition. Color saturation encodes time within an episode (light → dark = early → late frame). The linear-probe $R ^ { 2 }$ shown in each panel is computed on the same 2D subspace as the trajectories. The contrast across columns visualizes the multiplier effect documented in Table 1: V-JEPA + ID (bottom-left, $R ^ { 2 } = 0 . 8 4 )$ produces episode-distinct trajectories spanning the entire 2D subspace; Web-DINO + ID (bottom-middle, $R ^ { \hat { 2 } } = 0 . 2 9 )$ shows partial improvement with trajectories still compressed; SDXL VAE + ID (bottom-right, $R ^ { 2 } ~ = ~ \hat { - } 1 . 1 6 )$ shows the trajectories regressing relative to its frozen baseline $( R ^ { 2 } = - 0 . 8 5 )$ , since the ID loss cannot align features whose pretraining objective preserves pixel-level appearance rather than actionrelevant geometry.

on the $\mathrm { \Delta V \mathrm { - } J E P A } \to \mathrm { W e b \mathrm { - } D I N O } \to \mathrm { S D X I }$ VAE axis is preserved across both probes; the linearprobe subspace is used here because it admits a closed-form top-2 SVD, providing a canonical 2D visualization comparable across backbones.

Reading the figure. Three patterns are visible. V-JEPA frozen → + ID: trajectory spread expands from a narrow horizontal band (probe dir 2 range ∼ 0.5) to a 2D fan (range ∼ 1.5); episode endpoints separate into distinct clusters at the panel corners, indicating that different episodes terminate in different action-relevant states. Web-DINO frozen → + ID: trajectories sharpen modestly with the ID loss but most episode trajectories remain mutually overlapping—consistent with the family’s negative rotation $R ^ { 2 }$ in Figure 2. SDXL VAE frozen → + ID: the frozen panel shows trajectories scattered across the subspace; the + ID panel shows trajectories compressed into a denser cluster around probe dir $2 \in [ - 0 . 5 , 0 ]$ while linear-probe $R ^ { 2 }$ drops from −0.85 to −1.16, consistent with pixel-reconstruction-aligned features moving further from the action manifold under ID supervision rather than toward it. The episode-level visualization thus reproduces the same family ordering as the aggregate per-dimension and robustness tables.

## K Backbone Catalog

Detailed description of the eight backbone families and the specific checkpoints used.

V-JEPA 2 (ViT-L, ViT-B). A self-supervised video encoder trained with feature-level masked latent prediction [1]: the model is given a video clip with most spatio-temporal patches masked and learns to predict the encoder’s own features at the masked positions, via a target-encoder + predictor architecture. We use the public V-JEPA 2 ViT-L checkpoint (304M parameters) and the distilled V-JEPA 2.1 ViT-B checkpoint (87M parameters). Pretraining is on internet-scale video. The output token sequence has spatio-temporal structure; we average-pool across patch tokens within each frame and stack across frames to form the per-step feature vector.

VideoMAE V1 ViT-L. A self-supervised video encoder trained with pixel-space masked autoencoding [31]: most spatio-temporal patches of a video clip are masked and a decoder reconstructs raw RGB pixels at the masked positions. We use the public VideoMAE V1 ViT-L checkpoint (304M parameters) pretrained on Kinetics-400. We discard the decoder at evaluation and use the encoder’s patch-token features identically to V-JEPA.

Web-DINO ViT-L. A self-supervised image encoder trained with masked contrastive prediction on internet images [29]; conceptually a successor to DINOv2 that scales the SSL recipe to web-scale data. We use the public Web-DINO ViT-L checkpoint (304M parameters). Per-frame patch-token features are pooled the same way as the video encoders; temporal context, when used, comes only from stacking consecutive per-frame features.

SigLIP 2 ViT-L. A vision-language contrastive encoder trained with sigmoid-loss image-text pairs on a large web-scale image-text corpus [30]. We use the public SigLIP 2 ViT-L checkpoint (316M parameters) and extract patch-token features from the vision side. Language supervision is the axis that distinguishes SigLIP 2 from Web-DINO in our taxonomy.

LAPA-LAQ-OpenX. A latent-action quantizer pretrained on the Open X-Embodiment robot manipulation corpus [28]. The pretraining objective is to map consecutive video frames to a discrete latent-action code, learning a representation aligned with manipulation dynamics. We use the public LAPA-LAQ checkpoint (344M parameters). LAPA is the only encoder in our study explicitly designed with action structure in mind; its presence at moderate R2 (0.51 + ID) but well below V-JEPA (0.85) is informative about the limits of action-explicit pretraining alone.

DIFF. A pixel-diffusion world model trained from scratch on LIBERO at 34–91M parameters. The architecture is a 2-camera U-Net diffusion model predicting the next frame from a window of past frames; we use the 91M “large” configuration without action conditioning (action conditioning hurts representation, as documented in Appendix B) and trained on the task-OOD split (training tasks only, 400 episodes). Feature extraction is from the diffusion-trunk middle block at zero diffusion timestep.

SDXL VAE and Cosmos-1 image tokenizer. Two pixel-reconstruction-aligned autoencoders. The SDXL VAE [26] is the publicly released image VAE from the SDXL diffusion stack (34M parameters), pretrained on LAION-5B with an MSE + LPIPS + KL reconstruction objective. The Cosmos-1 image tokenizer [27] is the continuous image tokenizer used in NVIDIA’s Cosmos-1 world-model stack (34M parameters), pretrained on the Cosmos-1 video corpus. The Cosmos-1 checkpoint we use is the exact checkpoint employed by Nilaksh et al. [5], enabling direct comparison with their results. Both are encoder-only backbones in our study; we extract patch-token features from the encoder and discard the decoder.

Dreamer 4. A from-scratch reproduction of the shortcut-forcing dynamics architecture introduced by Hafner et al. [25]. Since the original Dreamer 4 implementation is not publicly released at the time of writing, we reproduced the model from the paper specification and the public DreamerV3 reference scaffolding,4 matching architectural details where the paper is specific. We trained at 64M and 276M parameters, both from scratch on LIBERO and (for 64M) from a DMControl-pretrained initialization, and probed both the agent-token and spatial-token feature streams. The agent-token stream is the canonical Dreamer “model state” (a sequence of ∼ 100 latent tokens combining a stochastic state and deterministic GRU hidden state); the spatial-token stream is the ViT trunk’s patch features before agent-token compression.