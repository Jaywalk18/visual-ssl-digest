# DECODABLE IS NOT GROUNDED: A VISION-ABLATION ARBITER FOR VLM SPATIAL REASONING

Chih-Ting Liao<sup>∗</sup> University of New South Wales mill.liao@unsw.edu.au

Xin Cao University of New South Wales

Fei Shen<sup>†</sup> National University of Singapore shenfei29@nus.edu.sg

Tat-Seng Chua National University of Singapore

## ABSTRACT

The standard way to read latent knowledge out of a model, a linear probe confirmed by a steering recovery, can systematically overstate what a vision–language model (VLM) actually grounds in the image. We show this on spatial reasoning, where the error is invisible to both probing and steering yet exposed by a one-line causal control: replacing the image with a gray blank. Probes decode the within-axis answer at 73–97% across axes, and a training-free projection lifts a near-chance axis from 59% to 79%, exactly the signature of unlocking latent knowledge. The blank-image arbiter refutes it, revealing three grounding regimes that probing conflates: an axis can be grounded (vision-dependent, correct), a prior (vision-independent, with its decode and its apparent recovery a directional default rather than perception), or, surprisingly, inverted: decodable, causally controllable, but deployed with the wrong sign, so the model scores below chance and the error requires looking. The taxonomy holds across the studied VLMs: in fourteen models spanning six language-model families and 2B–27B, horizontal is grounded, vertical is a prior, and depth is inverted, with the inversion emerging at scale within families. The decode̸=deploy inversion replicates on seven of eight models across five families, and the minimal edit that re-deploys it varies with geometry: a training-free rotation matches a trained edit on the cleanest model, while distributed inversions need a trained low-rank edit, tracing a per-model correction-complexity spectrum. The cheap, self-calibrating arbiter cleanly separates grounded perception, inverted perception, and prior substitution; we argue it should be a default control for latent-knowledge and steering claims in VLMs.

## 1 INTRODUCTION

A vision–language model (VLM) can answer a spatial question below chance, and blinding it (replacing the image with a same-size gray blank) makes it more accurate: the model is not failing to use the image, it is using it backwards. This sits inside a more familiar pattern. A linear probe reads a target attribute out of activations even when the model answers questions about it poorly [13, 18, 24], and the standard reading is that the model represents the attribute but fails to deploy it, motivating training-free interventions (steering, projection, representation engineering) that unlock the latent capability without gradient updates [32, 45, 18]. This paper is a cautionary study of that inference: we follow the recipe to its natural conclusion on one model, Qwen2.5-VL-7B [3], and one controlled axis (egocentric directional judgement on ViewSpatial-Bench [20]), then show the conclusion is wrong and that the corrected picture holds across fourteen models and six LM families (§5).

![](images/99253d0fca1838a2c19f2d68626670c4008452de57973f5a57db99896531d439.jpg)  
Figure 1: Decodability does not predict whether behavior depends on the image. Each point is a (model, axis) pair across 8 VLMs: x is probe decoding accuracy, y is vision-dependence (realimage − gray-image behavior). Probes decode every axis well (x>70%), but behavior splits into three regimes probing cannot see: horizontal grounded (high), vertical a prior (zero), depth inverted (below zero). The arbiter recovers this in one forward pass. Percentages.

The seductive result. Camera-relative direction decomposes into horizontal (left/right), vertical (above/below), and depth (front/back). The model is strong on horizontal, near chance on vertical, and below chance on depth (§4), yet a linear probe decodes all three axes from the residual stream, vertical the most (94% vs. horizontal’s 85%). A training-free projection along the vertical direction then lifts vertical accuracy from 59% to 79% while leaving horizontal untouched (§4.1), precisely the result one would publish as latent spatial knowledge recovered without training.

The arbiter. We then ask what the probe-and-steer pipeline never asks: does the recovered behavior depend on the image? A one-line control (replace the image with a same-size gray blank, leaving question and options unchanged; Fig. 1) fractures the single not-deployed story into three grounding regimes. Horizontal is grounded and correct (accuracy collapses 82% → 49% when blinded). Depth is grounded and inverted (below chance with the image, 31%; reverts to chance without it, 45%, so the error requires looking). Vertical is not grounded: real and blank accuracy are identical (59% ≈ 60%), and the projection recovery gains the same with no image (+19 blind vs. +21 sighted), so the 94% decode reads a directional default-to-above prior, not vision.

Why this matters. Vertical is the cautionary case: the most decodable axis, the cleanest trainingfree recovery, and the one axis that does not use vision, exactly the headline a decode-and-steer pipeline would report. Depth is the surprise: a model that scores below chance because it reads a real visual cue with the wrong sign. The vision-ablation arbiter catches both, is essentially free (one forward pass with pixels zeroed), and is discriminating rather than deflationary. Our contributions: (i) a three-regime grounding taxonomy (grounded-correct, grounded-inverted, prior-driven) and the finding that linear decodability and behavioral accuracy each fail to identify which regime an axis is in, the most decodable and most recoverable axis being the least grounded (§4–4.2); (ii) a cheap, discriminating vision-ablation arbiter, robust across five ablations including a real-but-mismatched in-distribution image, which we argue should be a default control for latent-knowledge and steering claims in VLMs (§4.2, §7); (iii) architecture-generality and a task-type boundary across fourteen VLMs, six LM families, and 2B–27B: across capable models horizontal is grounded, vertical a prior, depth inverted (emerging with scale within families), and the same models that invert cameraegocentric depth are grounded-correct on near-field and metric depth (§5–5.1); and (iv) a correction that reads out the geometry: the inversion replicates on seven of eight models across five families, and the minimal re-deploying edit (training-free rotation on the cleanest model, matching a trained edit; a trained low-rank edit on distributed ones) traces a per-model correction-complexity spectrum (§6).

## 2 RELATED WORK

Spatial reasoning in VLMs. Early benchmarks test two-object relations from canonical viewpoints (What’sUp [17], VSR [22]), now largely saturated [6, 9]; the frontier targets egocentric and multiperspective settings: VSI-Bench [42], MMSI-Bench [43], OmniSpatial [16], and ViewSpatial-Bench [20], the source of our task. A large training-based literature closes these gaps with synthetic 3D supervision or RL [7, 23, 8]; our study is diagnostic and training-free, orthogonal to those remedies.

Linear probing, spatial IDs, and probe validity. Linear world models for space have been read out of LLMs [13, 37] and VLMs; most relevantly, Kang et al. [18] identify linear spatial IDs that are causally steerable across VLMs. This establishes that spatial structure is linearly tractable and steerable; we ask the complementary question, whether a decodable, steerable direction is the one the model uses to look, and find the most decodable axis is the one that ignores the image. This connects to a long-standing caution: high decoding accuracy certifies linear presence, not use, and control tasks are needed to calibrate it [14, 5]. We provide a behavioral, causal control (vision ablation) rather than a probe-side one.

Latent spatial knowledge: present, but used how? A fast-growing line argues VLMs do represent space and that failures are an access problem. Closest to us, Gao et al. [10] show a latent 3D-topology map exists but is overshadowed by semantics and surface it with a subspace + Dirichlet regularizer; Qi et al. [33] blame vision-token norms suppressing positional cues and restore it by normalizing norms; Savietto et al. [36] explain multi-object errors as concept-vector interference and validate by steering; and concurrently Min et al. [28] find a vertical–distance entanglement that strengthens with scale. These share a template (locate a representation that is absent, overshadowed, entangled, or steerable, then surface or steer it) and assume the obstacle is access. Our phenomenon differs in kind: depth is present, correctly decodable, and causally controllable yet deployed with the wrong sign (decode̸=deploy), while vertical is decodable but behaviorally a prior. Our discriminator is a behavioral vision-ablation arbiter, and our remedy is a measured correction-complexity spectrum. A capability matrix (Appendix B, Table 5) places these works on common axes; the inversion regime and the cross-model correction ladder are, to our knowledge, unique here.

Steering and the actionability gap. Contrastive activation addition [32, 31] and representation engineering [45], following the activation-editing tradition [27, 39], inject a direction to change behavior. Basu et al. [4] give a pointed negative result: probes reach near-perfect AUROC while interventions correct only ∼20% of errors. We sharpen this from another angle: an intervention that moves accuracy can still act on a non-visual prior, so a behavioral success is not grounded recovery without a causal control. Relatedly, the VQA literature showed models exploit question priors and score well without the image [12, 1]. Our arbiter shares that intuition but is applied per axis and reads the signed real−ablated difference, so it separates not just grounded from ungrounded but a regime a whole-question blind baseline cannot express: an inverted axis, image-driven yet deployed wrong-sign (below chance with the image, at chance without). Liu et al. [24] document a related attention–correctness disconnect on spatial QA.

## 3 SETUP

Task and axes. Our probe is one controlled task that cleanly separates three spatial axes. We use the Camera-Relative-Direction split of ViewSpatial-Bench [20]: egocentric four-option questions “where is X relative to Y from the camera’s perspective” (1773 items). We decompose each item by the physical axis its gold answer lies on (horizontal left/right, vertical above/below, depth front/back, and composite, diagnostic only) and run all causal experiments on the 1047 pure-axis items (peraxis counts in Table 2). We report native 18-way accuracy (chance 25%) and a binary within-axis accuracy (the two poles, order randomized; chance 50%), balanced wherever a pole split is skewed. The anchor is Qwen2.5-VL-7B-Instruct [3]; implementation details are in Appendix B.

Probing, steering, and the arbiter. For decodability we fit a five-fold CV logistic probe (PCA-50) on the decision-token residual stream per axis and layer, reporting accuracy over a shuffledlabel floor. For causal tests we extract a per-axis direction (diff-of-means or back-projected probe weight) and add it at the decision token scaled in residual-stream standard deviations; the trainingfree projection amplifies the localized direction over the top-k neurons in the L20–24 band with a single global scalar γ and no gradients or labels (details in Appendix B). The vision-ablation arbiter re-evaluates any condition with the image replaced by a same-size gray blank, prompt and decoding unchanged, and classifies each axis by the signed real−gray gap: grounded if real exceeds gray and gray falls to chance, inverted if real is below chance and gray reverts toward it, and prior if real and gray coincide within bootstrap CIs. We apply it to both baseline accuracy and the projection gain, and guard against input-degeneracy with five ablations (gray, black, noise, patch-scramble, and a real-but-mismatched same-axis image) required to agree (§4.2). Intervention hyperparameters (γ, θ, layer band, top-k) are chosen on validation and frozen for the reported test numbers.

Generality battery. The deep mechanism is characterized on the anchor; for generality we run an independent battery (per-axis behavior, arbiter, per-layer probe) on thirteen further VLMs spanning six LM families and 2B–27B (full roster in §5), with per-model decode-peak layers re-found. We further run the behavioral battery on two external benchmarks (What’sUp [17], 3DSRBench [25]) and a nine-method depth correction battery (§6).

## 4 THE DECODE̸=DEPLOY DISSOCIATION AND A VISION-ABLATION ARBITER

Behavior: a single grounded axis. The native task accuracy is 45.4% on the full 1773 items, matching prior ViewSpatial numbers and validating our evaluation. Decomposed by axis (Table 2), the model is strong on horizontal (74.9% 18-way, 81.3% binary), near chance on vertical (26.9% / 58.5%), and below chance on depth (16.3% / 30.7%; binary depth is below chance at $\begin{array} { l l l } { p } & { < } & { 1 0 ^ { - 5 } } \end{array}$ , binomial). Controls rule out shortcuts (a same-dimension distractor tracks full accuracy; the composite class scores above chance only with a left/right component, 52.1% vs. 9.2%; Appendix B.6): the model is competent on horizontal and defaults elsewhere. Depth below chance on a randomized two-choice question rules out position bias and points to a systematic inversion, not a prompt-wording artifact: rephrasing front/back as closer/farther from the camera leaves accuracy below chance (Table 1).

Decodability: every axis, vertical most. A linear probe tells a very different story (Fig. 7). Decoding the within-axis pole from the decision-token residual stream rises sharply at layers 16–24 for all three axes and peaks well above the shuffledlabel floor: horizontal 84.9, depth 77.2, and vertical 94.1, the highest of the three. The imagepooled locus stays low everywhere (59–67), so the direction is assembled in the residual stream at the decision/text locus in mid-to-late layers rather than read directly off image tokens.

Table 1: Not a prompt-wording artifact. Qwen2.5-VL-7B, depth n=153, chance 0.5. Both phrasings stay below chance, so unambiguous camera-relative wording does not recover depth.

<table><tr><td>Phrasing</td><td>acc.</td><td>vs. ch.</td></tr><tr><td>“front / back”</td><td>28.1</td><td>-21.9</td></tr><tr><td>“closer / farther”</td><td>32.7</td><td>-17.3</td></tr></table>

The dissociation. Placing behavior next to decodability (Table 2) gives the gap that drives the rest of the paper. Horizontal is decoded and deployed (gap +4). Vertical is the most decodable axis yet behaves near chance (gap +36). Depth is decodable while behavior runs below chance (gap +47, anti-correlated). Taken on its own, this is the canonical knows-but-does-not-deploy picture: a strong latent representation (vertical 94%) that the model fails to act on, plus an inverted readout for depth. It is exactly the configuration that invites a training-free recovery method, which we build next.

## 4.1 AN INTERVENTION THAT APPEARS TO WORK

If vertical is decodable at 94% but behaves at chance, the natural move is to inject the decoded direction and watch behavior follow. It does.

Causal steering. Adding the per-axis diff-of-means direction at the peak layer cleanly and monotonically moves the positive-pole probability for all three axes (Appendix B.3, Fig. 11; depth moves the wrong way, confirming an inverted readout), establishing causal control. A corrective oracle raises vertical from 59% to 74%, and projecting the direction out collapses it to chance, a necessity signature. A multi-layer band around the decode peak (L20–24) recovers far more than any single layer, locating the effect on a distributed path (Appendix B.6, Table 11); the decoding-optimal probe direction is not the intervention-optimal one (Table 12).

Table 2: Per-axis behavior vs. linear decodability (Qwen2.5-VL-7B, ViewSpatial Camera-Relative-Direction). 18-way native accuracy (chance 25); binary within-axis poles (chance 50); decode peak probe accuracy (shuffle floor in parentheses). The gap (decode−binary) is small for horizontal, large for vertical, largest for depth where behavior is below chance. Accuracies in %.

<table><tr><td>Axis</td><td>n</td><td>18-way</td><td>binary</td><td>decode (shuffle)</td><td>gap</td><td>behavioral verdict</td></tr><tr><td>Horizontal (L/R)</td><td>518</td><td>74.9</td><td>81.3</td><td>84.9 (57)</td><td>+4</td><td>deployed, correct</td></tr><tr><td>Vertical (up/dn)</td><td>376</td><td>26.9</td><td>58.5</td><td>94.1 (63)</td><td>+36</td><td>near chance</td></tr><tr><td>Depth (front/back)</td><td>153</td><td>16.3</td><td>30.7</td><td>77.2 (55)</td><td>+47</td><td>below chance</td></tr></table>

![](images/ed67f5cc25ff487191807983cf381866784b1865d6fca79c99cdc34ec319e31f.jpg)  
Figure 2: The recovery is prior amplification, not visual deployment. Training-free projection along the localized vertical direction lifts vertical from 58.5 to 79 at γ = 2 with horizontal intact. But the same projection with a gray-blank image (dashed) gains just as much (+19.1 vs. +20.5): the lever amplifies a non-visual prior. An off-band projection (L2–6, dotted) does nothing. Percentages.

Training-free projection. The strongest version uses no gradients or labels: we amplify the localized vertical signal with one global scalar γ (§3). Vertical rises from 59% to 79% at γ=2 (+21) with horizontal essentially unchanged (82% → 81%), positive across a broad range, while an offband projection does nothing (Appendix B.3, Fig. 2), so the lever is real and localized to the decode band. Depth does not lift cleanly, consistent with an inverted/entangled mode rather than an underdeployed one.

The result we could have stopped at. Vertical is the most decodable axis (94%), behaves at chance (59%), and is recovered to 79% by a training-free, label-free projection that leaves horizontal intact, while depth fails to recover, a clean dissociation. Reported as is, this reads as latent vertical spatial knowledge recovered without training. It would be wrong.

## 4.2 THE VISION-ABLATION ARBITER

The recovery in §4.1 never checked whether the behavior it moved depends on the image. We run that check by re-evaluating every condition with the image replaced by a same-size gray blank (§3). Numbers here are for Qwen2.5-VL-7B; §5 shows the regimes recur across fourteen models.

The arbiter is discriminating, not deflationary. The three axes respond to blinding in three predictable ways (Table 3). Horizontal collapses from 82% to chance (51% balanced): its correct answers were image-driven, so the arbiter is a valid positive control. Depth sits below chance with the image (31%) but rises to chance when blinded (45%): the inversion itself is caused by the image; remove the input and the systematic error disappears. Vertical barely moves (59% → 60% raw, 70% ≈ 70% balanced): the model is not using the image for vertical at all. Four further ablations (black, noise, patch-scramble, and a real but mismatched in-distribution image) give the same verdict (Appendix B.6, Table 4), so the taxonomy is a property of how the model uses the image, not an artifact of the blank.

Table 3: The vision-ablation arbiter and the three grounding regimes (Qwen2.5-VL-7B). Per axis: accuracy with the real image and a gray blank (raw and balanced), and the training-free projection gain with the real vs. gray image. The regime is read off the real−gray baseline gap and whether the gain needs the image. Colors: grounded+correct , grounded+inverted , prior . Accuracies in %.

<table><tr><td rowspan="2">Axis</td><td colspan="2">baseline (raw)</td><td colspan="2">baseline (balanced)</td><td colspan="2">proj. gain</td><td rowspan="2">grounding regime</td></tr><tr><td>real</td><td>gray</td><td>real</td><td>gray</td><td>real</td><td>gray</td></tr><tr><td>Horizontal</td><td>82.4</td><td>49.4</td><td>82.6</td><td>50.7</td><td>+3</td><td>~0</td><td>vision-dep. &amp; correct</td></tr><tr><td>Depth</td><td>31.4</td><td>45.1</td><td>31.6</td><td>46.3</td><td>≤0</td><td>~0</td><td>vision-dep. &amp; inverted</td></tr><tr><td>Vertical</td><td>58.5</td><td>59.6</td><td>69.8</td><td>70.3</td><td>+21</td><td>+19</td><td>vision-indep. (prior)</td></tr></table>

Reading. Horizontal: removing the image collapses accuracy to chance (82 → 49), and the projection gain vanishes with the gray image, so the signal is genuinely visual. Depth: real accuracy is below chance but the gray image reverts to chance (31 < 45): the inversion is caused by looking at the image. Vertical: real ≈ gray under both raw and balanced scoring, and the projection gain is the same with or without the image, so the model is not using vision and the recovery amplifies a linguistic prior. Back-solving the per-class recalls on the 31/69 pole split gives recall(above) ≈ 99, recall(below) ≈ 40: a directional default-to-above response bias, not noise.

Table 4: The arbiter is robust to the input-degeneracy objection (Qwen2.5-VL-7B, balanced accuracy, full n, Run 12). Five vision ablations, including mismatch: a real-but-wrong same-axis image (fully in-distribution). All five agree with the gray-blank verdict. All values are balanced accuracy in %.

<table><tr><td>axis</td><td>real</td><td>gray</td><td>black</td><td>noise</td><td>scramble</td><td>mismatch</td><td>verdict</td></tr><tr><td>Horizontal</td><td>82.6</td><td>50.7</td><td>49.6</td><td>47.4</td><td>54.1</td><td>50.7</td><td>all → chance (grounded, correct)</td></tr><tr><td>Vertical</td><td>69.8</td><td>70.3</td><td>70.1</td><td>66.5</td><td>60.5</td><td>68.6</td><td>all ≈ real (prior)</td></tr><tr><td>Depth</td><td>31.6</td><td>46.3</td><td>44.2</td><td>42.7</td><td>48.9</td><td>45.7</td><td>all → chance from below (inverted)</td></tr></table>

The mismatch column is the decisive in-distribution control: a correct, real, same-axis image that simply does not match the question. Horizontal still collapses (50.7), vertical is unchanged (68.6 ≈ 69.8, so it ignores image content, not just image presence), and depth’s inversion relaxes to chance (45.7). The taxonomy is not an artifact of feeding the model a degenerate blank.

The recovery amplifies a prior. The decisive test is whether the projection gain of §4.1 needs the image. It does not: with the gray blank the vertical projection gains +19 at γ=2, indistinguishable from the +21 it gains with the real image (balanced: +12 blind vs. +14 sighted). The lever is a direction that pushes the output distribution, and that distribution was never conditioned on the image. The horizontal control behaves oppositely: its small already-deployed gain vanishes under the blank, as there is no prior to amplify. What vertical encodes is a directional default: balanced accuracy (70%) exceeds raw (59%) on the 31/69 above/below split, with recall(above) ≈ 99% and recall(below) ≈ 40%. The probe decodes vertical at 94% because this default-to-above is linearly readable, a language-side prior, the residual-stream analogue of a blind-VQA prior [12, 1] that probe accuracy cannot distinguish from grounded knowledge [14, 5].

Consequence. The headline of §4.1 does not survive: vertical is a prior, not under-deployed visual knowledge, and depth is inverted rather than absent. Depth also resists a clean training-free flip: sweeping a signed projection at both signs, linear edits only neutralize the inversion toward chance (best ≈ 47% vs. the ≈ 68% a global sign-flip would give), revealing an entangled mode rather than a single free sign; a minimal trained edit at the causally localized layer does re-deploy it (§6). On this model, probing and steering overstate grounded spatial knowledge, and one cheap control says by how much and where.

![](images/79b17722ddb58e604d84e0bda01f899901e309f88233d1ca67a29d650e44188a.jpg)

![](images/ae5acdd49ee2fe7a01d30803b4a09f68cdbfb7fd5f68cd58325873950660fb64.jpg)  
Figure 3: The grounding taxonomy as a cross-model card. Real-image accuracy on the three ViewSpatial axes for 14 VLMs (six LM families, 2B–27B), each cell colored by its arbiter regime. At a glance: horizontal grounded, vertical a prior, depth inverted in every capable model, with depth inversion scale-emergent within families (InternVL3 38 → 31 → 28; Gemma non-monotone 53→43 →46); SmolVLM-2.2B is a capability floor. Percentages.

## 5 THE TAXONOMY IS ARCHITECTURE-GENERAL AND TASK-SPECIFIC

The single-model arc could be a Qwen2.5-VL-7B idiosyncrasy. We run the independent battery (per-axis behavior, the vision arbiter, and a per-layer probe) on thirteen further VLMs, for fourteen models across six LM families (Qwen2/2.5/3, Mistral, SmolLM, Gemma) and 2B–27B, including three within-family scale ladders (full roster in Appendix B.4, Table 7; matrix in Fig. 3).

The three regimes recur across LM families. In every capable model horizontal is visiondependent and correct (real ≫ blank → chance), vertical is prior-dominated (real ≈ blank), and depth is at or below chance. This holds for Pixtral-12B (Mistral LM) and Gemma-3 (a different family with a SigLIP rather than Qwen-derived vision tower), so the taxonomy is a property of current VLMs, not a Qwen or InternViT artifact. SmolVLM-2.2B is the one all-axis exception, near chance even on horizontal (47%): a capability floor below which no axis is grounded, exactly as expected.

Decodability overstates knowledge in every model. The probe decodes vertical at 87–97% and depth at 73–86% in every capable model while behavior on those axes is prior or inverted. This gap between what the probe reads and what the model grounds is systematic across architectures, and the cautionary claim of §4.2 is the paper’s most robust result.

Depth inversion is scale-emergent within families. Three within-family ladders, varying only LM scale, show the inversion emerging with size (Appendix B.4, Fig. 14): InternVL3 (38% → 31% → 28%) and Qwen2.5-VL (48% → 32%) deepen monotonically, and in Gemma-3 (53% → 43% → 46%) it emerges by 12B with horizontal grounding also emerging (50% → 67%). Onset and severity are family-modulated (InternVL3 is inverted already at 2B, Qwen2.5-VL-3B and Gemma-4B not yet, and Gemma-3 partially recovers at 27B), so the law is that inversion emerges with LM scale within a family, with degree and onset family-specific.

The LM backbone sets the fingerprint. LLaVA-OneVision-7B and InternVL3-8B share the Qwen2-7B LM with different vision towers (SigLIP vs. InternViT) and are qualitatively identical on all three axes, differing only in the magnitude of the weak vertical vision signal. The grounding fingerprint is thus driven by the language backbone with the vision tower and scale tuning the degree, consistent with the locus analysis (Appendix B.2, Fig. 7), where the axis signal is assembled LM-side.

## 5.1 EXTERNAL BENCHMARKS: A TASK-TYPE BOUNDARY

A single-benchmark taxonomy invites the objection that ViewSpatial is simply broken on the nonhorizontal axes. We test this by running the behavioral battery and arbiter on two external benchmarks (What’sUp [17], tabletop on/under and front/behind; and 3DSRBench [25], real-scene height and closer-to-camera), mapping their categories onto our axes Table 6. The result replicates the taxonomy and sharpens it into a task-type boundary.

Depth inversion is camera-egocentric, not a general depth failure. The decisive result is a dissociation within the same model (Appendix B.4, Table 6): Qwen2.5-VL-7B inverts depth on ViewSpatial’s camera-relative front/back yet is grounded-correct on What’sUp near-field front/behind and 3DSRBench closer-to-camera, with mismatch controls confirming both are vision-driven; four further models show the same pattern, and horizontal replicates cleanly everywhere as a positive control. The inversion is thus a reproducible property of camera-egocentric front/back framing, a task type rather than a missing capability or a benchmark artifact, which pre-empts the artifact objection by exhibiting the contrast in one model. Vertical is likewise task-type-dependent (grounded on tabletop on/under, a prior on ViewSpatial above/below and 3DSRBench height): grounding is organized by task type, with simple near-field contact relations grounded while abstract camera-perspective and real-scene relations fall back to priors or invert, and the arbiter transfers unchanged to make this legible.

## 6 CORRECTING THE INVERSION DIAGNOSES ITS GEOMETRY

If the depth axis is decodable and causally controllable (§4.1) but read out with the wrong sign, what is the minimal edit that re-deploys it? We run a battery of nine corrections (scalar recalibration, logit-lens reflection, a norm-preserving angular rotation [38], selective multi-layer rotation, activation edits SEA [34] and SADI [40], per-head reflection ITI [21], SAE feature steering [30], and a trained low-rank edit LoRA [15] / ReFT [41]) on four models, then test generalization on four more (Appendix B.5). Two findings emerge: the decode̸=deploy inversion is a population phenomenon, and the minimal correction is a per-model complexity spectrum.

The inversion is a population phenomenon. A signed-slope diagnostic (the sign of the steering response of decodability versus behavior along depth) finds the inverted configuration (decodeslope > 0, deploy-slope < 0) on seven of eight inverted-capable models across five LM families (Fig. 4a, Table 9); the only exception is the not-yet-inverted scale floor Qwen2.5-VL-3B. Decodability rises with the steer everywhere while behavior moves the opposite way, ruling out a single-model artifact.

The minimal correction is a per-model complexity spectrum. Which correction re-deploys the axis reads out each model’s geometry. On Qwen3-VL-8B, the cleanest case, a training-free, normpreserving rotation by θ ≈ π nearly triples depth accuracy (21% → 77%) and, across three seeds, matches a trained low-rank edit (+49±8 vs. +55±2): the predicted un-inversion with no training. Across the population, which rung suffices is itself the diagnostic: a clean rotation on Qwen3-VL-8B, a partial one on LLaVA-OV (+14) and InternVL3-8B (+8), recalibration on InternVL3-14B, a trained edit on the distributed Qwen2.5-VL-7B (+43), and suppress-only (decodable but not installable) on InternVL3-2B and Pixtral-12B. The minimal sufficient correction ladders with the geometry (recalibration → rotation → trained low-rank → none), and the winning rung is a per-model fingerprint (Fig. 4b; full battery in Appendix B.5, Table 8).

The causal locus is mid-late, one-sided, and shared across the population. Localizing the inverted channel layer-by-layer in seven models (Fig. 5), the causal-controllability peak sits mid-late (L19–36), at or downstream of the decode peak and never early, and is strongly asymmetric (one steer sign recovers correct depth, the opposite does almost nothing) in 6/7 models; its magnitude tracks correction-complexity (0.99 for Qwen3-VL-8B down to ≤ 0.56 for trained-only models), so localization and correction cross-validate. On Qwen3-VL-8B (Appendix B.5, Table 10) the correctable channel is low-dimensional yet off the probe axis (a rank-1 ReFT +58 and a 4-component ICA reflection +60 recover it; the single probe direction and a GEVD subspace fail, +0); the training-free=trained tie is seed-stable. This closes the arc from diagnosis through mechanism to

![](images/7673cf3c8006e3d50d74cd3dba699389568555c9311b6612164bf737f4cd9053.jpg)  
(a) Population: $a _ { \mathrm { d e c o d e } } { > } 0$ but $a _ { \mathrm { d e p l o y } } { < } 0$ (inverted) in 7/8 models.

![](images/16793cf6196f57126cb22face9ec47670e34daaf44d2557ad9985233f29ab07a.jpg)  
(b) Correction spectrum: training-free rotation matches trained LoRA only on Qwen3-VL-8B.

Figure 4: The decode̸=deploy inversion is a population phenomenon (left); which minimal edit re-deploys it reads out each model’s geometry (right). Left: signed steering slope of decodability $( a _ { \mathrm { d e c o d e } } , \mathsf { b l u e } )$ and behaviour $( a _ { \mathrm { d e p l o y } }$ , vermillion) along depth; decodability rises everywhere while behaviour moves the opposite way in 7/8 models across five LM families (exception: the not-yetinverted Qwen2.5-VL-3B). Right: depth-accuracy lift over baseline. On the clean-rotation Qwen3- VL-8B a training-free rotation matches a trained LoRA across three seeds $( + 4 9 { \pm } 8 \mathrm { v s . + 5 5 { \pm } 2 } ) \colon$ ; on distributed (Qwen2.5-VL-7B) and not-installable (Pixtral-12B) inversions it does not, so the winning rung is a per-model fingerprint. Percentages.

![](images/d75849a41e539734347077f5ab173102e4db39473d33edbe1957fbf0178040a9.jpg)

![](images/a5e29556ac6c28ad757055a4c8a413abd5b78ed0c6138b53063979f0b0602a38.jpg)  
Figure 5: The inverted depth channel is causal and localized across the population, with a one-sided directional signature. Seven inverted models. Left: the probe-decode-peak (blue) and causal-controllability-peak (vermillion) layers both sit mid-late, never early (shaded). Right: causal controllability is strongly asymmetric, the ctrl(−) steer (bar) recovers correct depth while ctrl(+) (tick) does almost nothing, in 6/7 models; magnitude tracks card-C fixability. Controllability in %; left axis is layer index.

a geometry-diagnostic correction: a shared inversion whose minimal fix, per model, reads out its geometry.

## 7 DISCUSSION AND CONCLUSION

Decodability is not grounding, and steering recovery is not either. Both halves of the standard latent-knowledge pipeline failed to flag the vertical prior: the probe called it the most present axis (94%) and the projection the most recoverable, yet neither measures whether behavior is conditioned on the image, so their agreement is not corroboration. This sharpens the actionability-gap caution of Basu et al. [4]: a behavioral gain is not grounded recovery without a control.

The arbiter as a default control, and why three regimes. The vision ablation is cheap, assumption-light, and self-calibrating: it drives the genuinely visual horizontal axis to chance, so a null on another axis is informative rather than an input artifact. A two-way grounded-vs-not split would lump depth with the failures; the arbiter separates depth (vision-dependent, wrong sign) from vertical (vision-independent). We recommend reporting real-vs-ablated accuracy and treating a gain that survives ablation as prior amplification. Probing and steering overstate grounded knowledge because they are blind to image-dependence; the arbiter is not.

Why the inversion arises. Four convergent findings point to a learned frame-convention conflict rather than random damage. The inversion is construct-bound (grounded-correct on external depth, inverted only on ViewSpatial camera-relative front/back; Table 6), scale-emergent (deepens with LM scale, so learned), image-driven (not a fixed text prior; Appendix B.6), and a geometric mis-rotation (a norm-preserving rotation re-deploys it). The convergence indicates a depth readout learned against ViewSpatial’s camera-relative convention; tracing it to training data is future work.

Limitations. Depth-inversion emerges with LM scale but not monotonically; SmolVLM-2.2B bounds the horizontal-universal claim to capable models; and our causal evidence is representationlevel, as a clean pixel-level minimal pair for camera-relative front/back is unattainable (a mirror is identity for front/back, with no same-image pairs).

## REPRODUCIBILITY STATEMENT

All experiments use publicly available open-weight VLMs and the public ViewSpatial-Bench, What’sUp, and 3DSRBench datasets. The task/axis decomposition, behavioral protocols (18-way and binary), probe (five-fold CV logistic on PCA-50 decision-token features with a shuffled-label floor), the training-free projection, the vision-ablation arbiter and its five image controls, the crossmodel battery, and the nine-method correction battery are specified in §3 and Appendix B; per-model layers, hyperparameters, seeds, and the multi-seed protocol for the rotation/trained-edit comparison are reported with the corresponding results (§6, Appendix B.5). Limitations of single-seed and single-anchor analyses are stated explicitly in §7.

## REFERENCES

[1] Aishwarya Agrawal, Dhruv Batra, Devi Parikh, and Aniruddha Kembhavi. Don’t just assume; look and answer: Overcoming priors for visual question answering. In CVPR, 2018.

[2] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, et al. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631, 2025.

[3] Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, Jiabo Ye, Xi Zhang, Tianbao Xie, Zesen Cheng, Hang Zhang, Zhibo Yang, Haiyang Xu, and Junyang Lin. Qwen2.5-vl technical report. arXiv preprint arXiv:2502.13923, 2025.

[4] Sanjay Basu et al. Interpretability without actionability: Mechanistic methods cannot correct language model errors despite near-perfect internal representations. arXiv preprint arXiv:2603.18353, 2026.

[5] Yonatan Belinkov. Probing classifiers: Promises, shortcomings, and advances. Computational Linguistics, 48(1):207–219, 2022.

[6] Wenxiao Cai et al. Spatialbot: Precise spatial understanding with vision language models. arXiv preprint arXiv:2406.13642, 2024.

[7] Boyuan Chen et al. Spatialvlm: Endowing vision-language models with spatial reasoning capabilities. In CVPR, 2024.

[8] Erik Daxberger et al. Mm-spatial: Exploring 3d spatial understanding in multimodal llms. In ICCV, 2025. arXiv:2503.13111.

[9] Mengfei Du et al. Embspatial-bench: Benchmarking spatial understanding for embodied tasks with large vision-language models. arXiv preprint arXiv:2406.05756, 2024.

[10] Wei Gao et al. Uncovering and shaping the latent representation of 3d scene topology in vision-language models. arXiv preprint arXiv:2605.07148, 2026.

[11] Gemma Team, Aishwarya Kamath, Johan Ferret, Shreya Pathak, Nino Vieillard, et al. Gemma 3 technical report. arXiv preprint arXiv:2503.19786, 2025.

[12] Yash Goyal, Tejas Khot, Douglas Summers-Stay, Dhruv Batra, and Devi Parikh. Making the V in VQA matter: Elevating the role of image understanding in visual question answering. In CVPR, 2017.

[13] Wes Gurnee and Max Tegmark. Language models represent space and time. In ICLR, 2024.

[14] John Hewitt and Percy Liang. Designing and interpreting probes with control tasks. In EMNLP, 2019.

[15] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations (ICLR), 2022. arXiv:2106.09685.

[16] Mengdi Jia, Zekun Qi, Shaochen Zhang, Wenyao Zhang, Xinqiang Yu, Jiawei He, He Wang, and Li Yi. Omnispatial: Towards comprehensive spatial reasoning benchmark for vision language models. In ICLR, 2026. arXiv:2506.03135.

[17] Amita Kamath, Jack Hessel, and Kai-Wei Chang. What’s “up” with vision-language models? investigating their struggle with spatial reasoning. In EMNLP, 2023. arXiv:2310.19785; benchmark commonly cited as “What’sUp”.

[18] Raphi Kang, Hongqiao Chen, Georgia Gkioxari, and Pietro Perona. Linear mechanisms for spatiotemporal reasoning in vision language models. arXiv preprint arXiv:2601.12626, 2026.

[19] Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. Llava-onevision: Easy visual task transfer. arXiv preprint arXiv:2408.03326, 2024.

[20] Dingming Li, Hongxing Li, Zixuan Wang, Yuchen Yan, Hang Zhang, Siqi Chen, Guiyang Hou, Shengpei Jiang, Wenqi Zhang, Yongliang Shen, Weiming Lu, and Yueting Zhuang. Viewspatial-bench: Evaluating multi-perspective spatial localization in vision-language models. arXiv preprint arXiv:2505.21500, 2025.

[21] Kenneth Li, Oam Patel, Fernanda Viegas, Hanspeter Pfister, and Martin Wattenberg. Inference-´ time intervention: Eliciting truthful answers from a language model. In Advances in Neural Information Processing Systems (NeurIPS), 2023. arXiv:2306.03341.

[22] Fangyu Liu, Guy Emerson, and Nigel Collier. Visual spatial reasoning. TACL, 2023.

[23] Yuhong Liu, Beichen Zhang, Yuhang Zang, Yuhang Cao, Long Xing, Xiaoyi Dong, Haodong Duan, Dahua Lin, and Jiaqi Wang. Spatial-ssrl: Enhancing spatial understanding via selfsupervised reinforcement learning. arXiv preprint arXiv:2510.27606, 2025. CVPR 2026 accepted.

[24] Zhining Liu, Ziyi Chen, Hui Liu, Chen Luo, Xianfeng Tang, Suhang Wang, Joy Zeng, Zhenwei Dai, Zhan Shi, Tianxin Wei, Benoit Dumoulin, and Hanghang Tong. Seeing but not believing: Probing the disconnect between visual attention and answer correctness in vlms. arXiv preprint arXiv:2510.17771, 2025.

[25] Wufei Ma, Haoyu Chen, Guofeng Zhang, Celso M. de Melo, Alan Yuille, and Jieneng Chen. 3DSRBench: A comprehensive 3D spatial reasoning benchmark. In arXiv preprint arXiv:2412.07825, 2024.

[26] Andres Marafioti et al. Smolvlm: Redefining small and efficient multimodal models. ´ arXiv preprint arXiv:2504.05299, 2025.

[27] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. Locating and editing factual associations in gpt. In NeurIPS, 2022.

[28] Cheolhong Min, Jaeyun Jung, Daeun Lee, Hyeonseong Jeon, Yu Su, Jonathan Tremblay, Chan Hee Song, and Jaesik Park. Why far looks up: Probing spatial representation in visionlanguage models. arXiv preprint arXiv:2605.30161, 2026.

[29] Mistral AI. Pixtral 12b. arXiv preprint arXiv:2410.07073, 2024.

[30] Mateusz Pach, Shyamgopal Karthik, Quentin Bouniot, Serge Belongie, and Zeynep Akata. Sparse autoencoders learn monosemantic features in vision-language models. In CVPR, 2025. arXiv:2504.02821.

[31] Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering Llama 2 via contrastive activation addition. In ACL, 2024. arXiv:2312.06681.

[32] Nina Panickssery, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, and Alexander Matt Turner. Steering llama 2 via contrastive activation addition. In ACL, 2024. Alias for panickssery2024; preserved for citations.

[33] Jianing Qi, Jiawei Liu, Hao Tang, and Zhigang Zhu. Beyond semantics: Rediscovering spatial awareness in vision-language models. arXiv preprint arXiv:2503.17349, 2025.

[34] Yifu Qiu, Zheng Zhao, Yftah Ziser, Anna Korhonen, Edoardo M. Ponti, and Shay B. Cohen. Spectral editing of activations for large language model alignment. In Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2405.09719.

[35] RemyxAI. SpaceQwen2.5-VL and SpaceOm: Spatial-reasoning fine-tunes of Qwen2.5-VL. Hugging Face model cards, https://huggingface.co/remyxai, 2025. VQASynthdistilled spatial-reasoning VLMs, building on SpatialVLM [7].

[36] Daniele Savietto, Declan Campbell, Andre Panisson, Marco Nurisso, Giovanni Petri,´ Jonathan D. Cohen, and Alan Perotti. The geometry of representational failures in vision language models. arXiv preprint arXiv:2602.07025, 2026.

[37] Matthieu Tehenan, Christian Bolivar Moya, Tenghai Long, and Guang Lin. Linear spatial world models emerge in large language models. arXiv preprint arXiv:2506.02996, 2025.

[38] Vu and Nguyen. Angular steering: Behavior control via rotation in activation space. In Advances in Neural Information Processing Systems (NeurIPS), 2025. arXiv:2510.26243.

[39] Kevin Wang et al. Interpretability in the wild: a circuit for indirect object identification in gpt-2 small. In ICLR, 2023.

[40] Weixuan Wang, Jingyuan Yang, and Wei Peng. Semantics-adaptive activation intervention for LLMs via dynamic steering vectors. In International Conference on Learning Representations (ICLR), 2025. arXiv:2410.12299.

[41] Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, and Christopher Potts. ReFT: Representation finetuning for language models. In NeurIPS, 2024.

[42] Jihan Yang, Shusheng Yang, Anjali W. Gupta, Rilyn Han, Li Fei-Fei, and Saining Xie. Thinking in space: How multimodal large language models see, remember and recall spaces. In CVPR, 2025. VSI-Bench.

[43] Sihan Yang, Runsen Xu, Yiman Xie, Sizhe Yang, Mo Li, Jingli Lin, Chenming Zhu, Xiaochen Chen, Haodong Duan, Xiangyu Yue, Dahua Lin, Tai Wang, and Jiangmiao Pang. Mmsi-bench: A benchmark for multi-image spatial intelligence. In ICLR, 2026. arXiv:2505.23764.

[44] Jinguo Zhu, Weiyun Wang, Zhe Chen, Zhaoyang Liu, Shenglong Ye, Lixin Gu, Yuchen Duan, Hao Tian, Weijie Su, Jie Shao, et al. Internvl3: Exploring advanced training and test-time recipes for open-source multimodal models. arXiv preprint arXiv:2504.10479, 2025.

[45] Andy Zou et al. Representation engineering: A top-down approach to ai transparency. arXiv preprint, 2023.

## A IMPLEMENTATION DETAILS

Model and decoding. Qwen2.5-VL-7B-Instruct [3], loaded in 4-bit nf4 on a single RTX 4090, attn implementation="sdpa" (flash-attention not installed), greedy decoding, max new tokens=8. The language tower has 28 decoder layers, hidden size 3584; the image token id is 151655. We index hidden states[i] as the output of decoder layer i−1. For Qwen2.5-VL under transformers 4.57 the decoder path is model.model.language model.layers; we use a multi-path fallback to remain robust to version changes.

Answer reading. Letters are matched robustly over the variants "A"/" A" to avoid tokenizationinduced baseline drift. In the binary protocol the positive-pole letter is randomized per item; all scoring, steering targets, and projections use the per-item positive-pole map, and we verify the evaluation reproduces canonical baselines (V 0.585 / H 0.824 / D 0.314) before trusting any intervention number.

Probing. Decision-token residual-stream activations are cached for all layers (L=29 including the embedding, d=3584) over the 1047 pure-axis items. Per axis and layer we fit a five-fold crossvalidated logistic regression on PCA-50 features; the shuffled-label probe gives the chance floor reported in brackets. We also report a light |d<sup>′</sup>| sparse-neuron localizer on decision-token activations; cross-axis top-20 neuron overlap is near zero, indicating distinct codes.

Steering scale. Raw unit-vector steering $( \alpha \hat { d } , \alpha \leq 8 )$ is negligible against residual-stream norms of 170–245 and produces null effects. We therefore scale by the standard deviation of the projection of activations onto $\hat { d } _ { l } ,$ so α counts standard deviations; we cap $| \alpha | \le 4$ because larger values over-steer (at $| \alpha | { = } 6$ the injected norm reaches ∼65% of ∥h∥). Causal-control claims rest on the monotonicity of $P ( + { \mathrm { p o l e } } )$ , not on accuracy at extreme α.

Training-free projection. On the k=64 highest-|d<sup>′</sup>| neurons in the localized band we add $\begin{array} { r } { \gamma \left( \sum _ { i } \mathrm { s i g n } ( d _ { i } ^ { \prime } ) a _ { i } ( x ) \right) } \end{array}$ <sup>ˆ</sup>d<sub>l</sub> to the down-projection output, with a single global $\gamma$ and no gradient updates. The band is L20–24 (vertical L21–25, depth L19–23); the off-band control uses L2–6. We note that γ is currently swept on the evaluation set; a held-out γ selection is the correct protocol and we report the broad positive range rather than a single value to mitigate this.

Arbiter. The gray-blank image is constructed at the same resolution as the real image and passed through the identical preprocessing path (make prompt blank), so the only change is pixel content. Baseline and projection are re-evaluated under this input with all else fixed.

## B ADDITIONAL RESULTS

This appendix contains the supporting figures and tables referenced from the main text: the layerwise analyses (§B.2), the full cross-model and external-benchmark results (§B.4), the complete correction battery and population mechanism (§B.5), and robustness controls (§B.6).

## B.1 THE DISSOCIATION, IN DETAIL

Figure 6 plots behavior against decodability for the anchor model, the visual form of the main-text Table 2.

## B.2 LAYER-WISE ANALYSES

Complementing the main-text locus figure (Fig. 7), the per-layer decision-token decode peaks at L20–24 (Fig. 8), and the causal locus coincides with this decode onset (Fig. 9). On the second model, the inverted depth channel is causal, sharply localized, and one-sided across the population (Fig. 5), with a layer-wise decode/controllability profile shown in Fig. 10.

Table 5: Capability matrix against interpretability studies of VLM spatial failure. Columns: representation treated as present; grounding tested behaviorally (vision ablation) vs. another probe; a sign-inversion identified; a 3-way grounded/prior/inverted taxonomy; a cross-model correction spectrum; training-free. <sup>✓</sup>/<sup>✗</sup>/<sup>●</sup>=yes/no/partial.

<table><tr><td></td><td>repr. present</td><td><img src="images/c62924ad8dfb7e6c9e3f1ee2a9e29080175edf1c124e7d1bec5a640eb9b654a4.jpg"/></td><td><img src="images/aa939f44dfda5729a1f8dbc1faf6d7bed739bfe9c2bcf33dd636cdaea7b43d8c.jpg"/></td><td><img src="images/babfa7bf034aa45f28f30431306cdb0d7f4fc13a8e91814deba7055aa1e74c26.jpg"/></td><td><img src="images/ed4a521e4f0e54c611908878dcec46ea5251ddd3414f52741354db8408e20a4c.jpg"/></td><td>training-free</td></tr><tr><td>Beyond Semantics [33]</td><td>√</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>3D Scene Topology [10]</td><td>√</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td></tr><tr><td>Geometry of Failures [36]</td><td>√</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>Why Far Looks Up [28]</td><td>√</td><td>●</td><td>✕</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>Spatial IDs [18]</td><td>√</td><td>✕</td><td>✕</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>Actionability gap [4]</td><td>√</td><td>●</td><td>✕</td><td>✕</td><td>✕</td><td>√</td></tr><tr><td>Ours</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td><td>●</td></tr></table>

<sup>●</sup> (behavioral): relate representation to behavior but not via a per-axis causal vision-ablation control. <sup>●</sup> (Ours, training-free): rotation is training-free on the clean-rotation model; distributed inversions still need a trained edit.

![](images/6cce7a463607b69a8db710ab3df6c5e3b6403eaa518b285c3a856226b8193f12.jpg)  
Figure 6: The decodability–behavior dissociation that invites a wrong conclusion. Peak probedecode accuracy (blue) vs. actual binary behavior (teal). Horizontal is decoded and deployed (gap +4). Vertical is the most decodable axis yet sits near chance behaviorally (gap +36). Depth is decodable but behavior is below chance (gap +47, anti-correlated). Read alone, this looks like “knowledge present, not deployed” and motivates a recovery method — the trap we spring in §4.2. Values shown as percentages.

## B.3 INTERVENTION GEOMETRY

The causal sweep (Fig. 11) and the diff-of-means projection (Fig. 2) establish causal control; the un-inversion is a near-π rotation of the readout (Fig. 12).

## B.4 FULL CROSS-MODEL AND EXTERNAL RESULTS

Table 7 is the complete fourteen-model roster behind the matrix figure (Fig. 3); Fig. 13 summarizes per-axis behavior and Fig. 14 the within-family scale ladders. On external benchmarks the inversion becomes a task-type boundary (Fig. 15, Table 6): the same models that invert ViewSpatial cameraegocentric front/back are grounded-correct on near-field front/behind and metric closer-to-camera depth.

Table 6: External-benchmark grounding (Tier-3). Real vs. gray/mismatch balanced accuracy (chance 50; grounded-correct when real ≫ mismatch). The same models that invert depth on ViewSpatial [20] (Table 2) are grounded-correct on What’sUp [17] and 3DSRBench [25] depth, a depth-type boundary; four further models replicate the external grounding (Appendix B.4).

<table><tr><td>benchmark</td><td>axis (task)</td><td>real</td><td>gray</td><td>mismatch</td><td>verdict</td></tr><tr><td colspan="6">Qwen2.5-VL-7B anchor</td></tr><tr><td>What&#x27;sUp-A</td><td>horizontal (left/right)</td><td>99</td><td>51.5</td><td>51.5</td><td>grounded</td></tr><tr><td>What&#x27;sUp-A</td><td>vertical (on/under)</td><td>99.5</td><td>—</td><td>47.6</td><td>grounded</td></tr><tr><td>What&#x27;sUp-B</td><td>horizontal</td><td>100</td><td>—</td><td>47.1</td><td>grounded</td></tr><tr><td>What&#x27;sUp-B</td><td>depth (front/behind)</td><td>99.5</td><td>50</td><td>48.5</td><td>grounded-correct</td></tr><tr><td>3DSRBench</td><td>vertical (height)</td><td>56.1</td><td>56.2</td><td>54.1</td><td>prior</td></tr><tr><td>3DSRBench</td><td>depth (closer-to-cam)</td><td>72.8</td><td>56.6</td><td>49.4</td><td>grounded-correct</td></tr></table>

![](images/f0b11ad93020ea75f5fbe542f73bcc964f8e5b5a2f7cd5ec6748cc7b75e00b6a.jpg)  
Figure 7: The axis signal is assembled LM-side, not in raw image tokens (Qwen2.5-VL-7B, fivefold CV probe). Per-layer decode accuracy of the within-axis pole from three loci: decision token, mean-pooled text tokens, and mean-pooled image tokens (grey). On all three axes the image-pool trace stays flat near its shuffle floor (dashed) while the decision/text traces rise sharply at L16–24: the direction is composed in the residual stream LM-side, not read off image patches. Percentages.

## B.5 CORRECTION BATTERY, POPULATION, AND SECOND-MODEL MECHANISM

Table 8 is the full nine-correction battery behind the main-text Fig. 4b; Table 9 is the eight-model population behind the main-text forest plot (Fig. 4a); Table 10 reproduces the deep mechanism on a second model.

## B.6 ROBUSTNESS CONTROLS AND AUXILIARY ANALYSES

Why does it invert? A negative answer (not a prior). Replacing the image with a blank/gray input and measuring the lopsidedness of the answer (Table 13) rules out the simplest explanation— a fixed text-convention prior (“front means farther”). A pure prior would floor depth accuracy at chance, but all three clearly-inverted models score below chance with the real image, which a mild blank-image bias cannot produce. The inversion is therefore image-driven: the model perceives the depth cue and deploys it with a flipped sign. This is a negative “why”; a positive egocentric-vsallocentric frame-flip account is left to future work.

Composite class and the L/R crutch. The composite class (n=726) scores 39.9% overall, but split by whether the gold answer carries a left/right component it is 52.1% (with L/R, n=520) vs. 9.2% (without L/R, n=206). A confusion analysis shows horizontal stays on-axis (79%) while vertical and depth escape to composite (45.7% / 49.7%), coherent with horizontal carrying the only grounded directional signal.

decoder layer  
![](images/572ab83dc164a6c933bbc1a846abf481fb5387701f4ffe5f43e3dc819196731a.jpg)  
Figure 8: All three axes are linearly decodable; vertical most of all. Five-fold cross-validated logistic-probe accuracy for the two within-axis poles, read from the decision-token residual stream at each layer (PCA-50 features). Decoding rises sharply at L16–24 for every axis. Vertical peaks highest (94 at L23), above horizontal (85); depth reaches 77. The dotted band is the per-axis shuffledlabel floor (55–63). On a probe-only reading, all three axes “contain” the answer — a conclusion Fig. 1 overturns for vertical. Values shown as percentages.

![](images/52365e098e4f23078c4b5bb67f451ab08db2154cc9c81ea05e108afc89f74c2d.jpg)

![](images/c499fef574fff6164403e3fe1ac2042fd71d2e560dd205b05440acc61a7e4ad5.jpg)  
Figure 9: The deployment locus is a distributed L20–24 path, and vertical necessity is localized there (S4). Left: corrective-injection gain peaks at L21–22 for all axes, matching the decode onset (Fig. 7). Right: project-out (necessity) dips negative for vertical at L21–24 (peak −6.7 at L21), whereas horizontal stays flat (redundant code) and depth is positive (removing the inverted signal helps). A five-layer band beats any single layer (Table 11). n=150 subsample, so deltas are subsample-level; the shape and locus are robust. Values shown as percentages.

## C NEGATIVE RESULTS AND A METHODOLOGICAL NOTE

A masked-LoRA tuning ladder that we discarded. Before the training-free projection we ran a neuron-targeted masked-LoRA fine-tune (300 steps) on the vertical axis, comparing d<sup>′</sup>-localized neurons against random in-band neurons and against tuning the whole band. The d<sup>′</sup>-localized run recovered vertical strongly while preserving horizontal, which looked like evidence that the localized neurons specifically “deploy” vertical. We do not report it as a result for two reasons. First, a random-in-band control recovered vertical almost identically, exposing the obvious confound: 300 steps of supervised fine-tuning on ∼300 same-distribution items fit the binary task regardless of which sufficient parameter set is tuned. The only defensible signal there was specificity (the localized run damaged horizontal less), which is a weaker claim than we wanted. Second, and more importantly, any SFT-based “recovery” is exactly the kind of result the arbiter (§4.2) later showed to be untrustworthy for a grounding claim: it moves accuracy without establishing that the moved behavior uses the image. This is why the paper’s recovery analysis uses the label-free projection (which cannot fit the task) and is then audited under vision ablation.

![](images/7cb9476014ebcf6c30a6820ea2395f8b791822fc0ff96622130faa4e5f252ae6.jpg)  
Figure 10: Layer-wise decode and causal controllability (Qwen3-VL-8B depth). Probe-decoding accuracy (blue) climbs from chance in the early layers to a peak at L26, while one-sided causal controllability (vermillion, β=4) is flat and near-zero early, then spikes to 98.7% at L24 before falling back to ∼35–39% in the later layers. The narrow controllability spike at the decode-onset region—strongly asymmetric, recovering correct depth from a single-sign steer—is the clean causal locus of the inversion; the best trained edit instead prefers the decode-peak L30 (+57.8 vs. +38.3 at L24, §6). Early decode values (<L20) are approximate; L20–L36 points and the controllability curve are measured. Values shown as percentages.

![](images/029ab9d8056433006e5a593f58abaa2bfc935fb9e17271b06203217cf08db34d.jpg)  
steering strength α (std along d<sup>ˆ</sup>, injected at decision token)  
Figure 11: Causal control is clean and monotone on all three axes (S3). Probability of the positive pole as the per-axis diff-of-means direction is added at the decision token (full n=1047, Qwen2.5- VL-7B). Horizontal and vertical increase with α (auto-detected readout sign +1); depth decreases (sign −1), the steering signature of an inverted readout. Each axis is causally steerable—which, like decodability (Fig. 8), is necessary but not sufficient for grounding: the arbiter (§4.2) shows the vertical lever moves a prior. Values shown as percentages.

An evaluation bug, and the lesson. An early version of the tuning eval omitted the per-item positive-pole map, which silently collapsed the baselines to chance (horizontal 0.54 vs. the true 0.82) and mismapped the supervision target on roughly half the items. The numbers were discarded once the eval was made to reproduce the canonical baselines. The standing practice we adopt as a result—verify that any re-implemented evaluation reproduces a known baseline before trusting any delta on top of it—is the within-pipeline analogue of the arbiter: cheap controls, applied early, that catch a confound before it becomes a claim.

![](images/c009cca49dfaa0255e17ffd8a2e200187d8cd559c6f4f0e286a91b46a2092ceb.jpg)  
Figure 12: The depth inversion is a clean low-dimensional rotation in Qwen3-VL-8B. Schematic of the depth readout plane. Left: the model’s readout places true-“closer” items on the “farther” side of the decision boundary (and vice-versa)—a systematic sign inversion of an otherwise wellseparated representation (accuracy 21.4, well below chance). Right: a single training-free, normpreserving rotation by θ≈π re-aligns the clusters with the correct labels (accuracy 77.1, matching a trained low-rank edit). When the inversion is a clean rotation, the un-inversion is too—this is the geometry that lets a zero-training fix re-deploy the axis (§6). Values shown as percentages.

![](images/242ba56bd6cfde1e7569c8ca4d33bc106f136ca48e400e6e4e381cd269cfe93f.jpg)  
Figure 13: The three-regime taxonomy is architecture-general (10 of 14 models shown; full set in Table 7). Real balanced accuracy per axis. Horizontal (teal) is high and vision-dependent in every capable model; vertical (slate) is above chance but prior-driven everywhere (real ≈ gray ≈ mismatch; Table 7) and is the most probe-decodable axis; depth (vermillion) is at or below chance in all and inverted in the larger models. The lone all-axis exception is SmolVLM-2.2B (horizontal ≈ chance), a capability floor rather than a counterexample. The newest model (Qwen3-VL-8B) is the most depth-inverted; the non-Qwen Pixtral-12B (Mistral) still shows the full taxonomy. Values shown as percentages.

Table 7: Cross-model grounding matrix: 14 VLMs, six LM families. Real balanced accuracy per axis (chance 50); H grounded-correct , V prior , D inverted . The taxonomy holds in every capable model, including the non-Qwen Pixtral (Mistral) and Gemma-3 (SigLIP vision). Shaded rows are the three within-family scale ladders. The probe still decodes vertical at 87–97 and depth at 73–86 in every capable model, so decodability overstates usable knowledge universally. All values are balanced accuracy in %.

<table><tr><td>model</td><td>LM family</td><td>H</td><td>V</td><td>D</td><td>note</td></tr><tr><td>InternVL3-2B [44]</td><td>Qwen2-based</td><td>65.8</td><td>57.5</td><td>38.1</td><td>ladder: already inverted at 2B</td></tr><tr><td>InternVL3-8B</td><td>Qwen2-7B</td><td>86</td><td>89.5</td><td>30.9</td><td>ladder (monotone)</td></tr><tr><td>InternVL3-14B</td><td>Qwen2.5-14B</td><td>88</td><td>90</td><td>28.2</td><td>ladder: deepest</td></tr><tr><td>Gemma-3-4B [11]</td><td>Gemma/SigLIP</td><td>49.5</td><td>—</td><td>52.6</td><td>ladder: H &amp; D not yet (small)</td></tr><tr><td>Gemma-3-12B</td><td>Gemma/SigLIP</td><td>67.3</td><td>75.9</td><td>43</td><td>ladder: deepest</td></tr><tr><td>Gemma-3-27B</td><td>Gemma/SigLIP</td><td>69.8</td><td>78.4</td><td>45.6</td><td>ladder: partial recovery (non-mono.)</td></tr><tr><td>Qwen2.5-VL-3B [3]</td><td>Qwen2.5</td><td>70.7</td><td>65.9</td><td>47.9</td><td>depth not yet inverted</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Qwen2.5</td><td>82.6</td><td>69.8</td><td>31.6</td><td>primary (deep mechanism)</td></tr><tr><td>LLaVA-OV-7B [19]</td><td>Qwen2-7B</td><td>75.3</td><td>66.6</td><td>37.8</td><td> $\approx$  InternVL3-8B (shared LM)</td></tr><tr><td>Qwen3-VL-8B [2]</td><td>Qwen3</td><td>87.7</td><td>94.6</td><td>22.4</td><td>newest; most inverted</td></tr><tr><td>Pixtral-12B [29]</td><td>Mistral</td><td>70.8</td><td>74.5</td><td>42.2</td><td>non-Qwen; taxonomy holds</td></tr><tr><td>SpaceOm [35]</td><td>Qwen2.5 (sp-FT)</td><td>75.6</td><td>65.3</td><td>47.3</td><td>spatial-FT did not fix V/D $^{\dagger}$ </td></tr><tr><td>SpaceQwen-3B [35]</td><td>Qwen2.5 (sp-FT)</td><td>50.2</td><td>56.2</td><td>49.9</td><td>near-chance; not a donor $^{\dagger}$ </td></tr><tr><td>SmolVLM-2.2B [26]</td><td>SmolLM</td><td>47.4</td><td>57.4</td><td>47.8</td><td>capability floor (even H  $\approx$  chance)</td></tr></table>

Shaded = within-family scale ladders (InternVL3 and Qwen2.5 monotone; Gemma non-monotone, 27B partially recovers; Fig. 14). <sup>†</sup>The two spatial-fine-tuned models (SpaceOm, SpaceQwen) are grounded-correct on no failing axis, so neither is a clean “donor”; SpaceOm is a thinking model whose first-token eval likely understates it. SmolVLM-2.2B is near-chance on all axes, bounding “horizontal universal” to capable models.

![](images/23487715bf026b37e37971cc92e11a21aab9a4c4361143ef35e8be07d676a31e.jpg)  
Figure 14: Depth inversion emerges with LM scale within families, but the size–severity relation is family-specific and not strictly monotone. InternVL3 (2B 38.1 → 8B 30.9 → 14B 28.2) and Qwen2.5-VL (3B 47.9 → 7B 31.6) deepen monotonically with size, whereas Gemma-3 $( 4 B 5 2 . 6 \to 1 2 B 4 3 \to 2 7 B 4 5 . 6 )$ emerges then partially recovers at 27B—a clean within-family non-monotonicity. Onset differs by family (InternVL3 already inverted at 2B; Qwen2.5-VL-3B and Gemma-4B not yet), and across families severity is not size-monotonic (Pixtral-12B milder than Qwen2.5-VL-7B; newest Qwen3-VL-8B the most inverted). The honest summary is “inversion emerges with scale within families; degree and monotonicity are family-specific.” Values shown as percentages.

![](images/4f31ee3b64148c6b73b1fc0cb283e05e7fa24bb891a32352473e80fd232eefe9.jpg)  
Figure 15: Grounding is task-type-specific: the depth inversion is ViewSpatial-cameraegocentric, not a general depth failure (Qwen2.5-VL-7B; depth on What’sUp-B front/behind and 3DSRBench closer-to-camera, vertical on What’sUp-A on/under and 3DSRBench height). The same model that inverts depth on ViewSpatial’s camera-relative front/back (30.7) is grounded-correct on near-field front/behind (99.5) and metric closer-to-camera (72.8); mismatch controls confirm these are vision-driven (48.5/49.4). Vertical is grounded on simple tabletop on/under (99.5) but a prior on both ViewSpatial and 3DSRBench real-scene height. Horizontal is grounded everywhere externally (99–100). This pre-empts the “ViewSpatial artifact” objection: the inversion is a specific, reproducible property of camera-egocentric front/back framing, not a broken benchmark. Values shown as percentages.

Table 8: The minimal correction that re-deploys depth diagnoses the inversion’s geometry (testacc lift over baseline; ViewSpatial [20] depth, n≈150; baselines qwen 26.2 / qwen3-vl 21.4 / pixtral 39.8 / 3B 48.1). Across nine corrections, a training-free norm-preserving rotation re-deploys the axis where the inversion is a clean low-dimensional flip (Qwen3-VL-8B [2]), matching a trained edit; a distributed or decodable-but-not-installable inversion needs the trained edit; and the un-inverted 3B (not inverted) admits no correction. Lifts in percentage points over baseline.

<table><tr><td>correction</td><td>qwen-7B</td><td>qwen3-vl-8B</td><td>pixtral-12B</td><td>3B</td></tr><tr><td>recalibrate (1-D scalar)</td><td>+11.2</td><td>+0</td><td>+1</td><td>+0</td></tr><tr><td>logit-lens reflect-at-onset</td><td>n/a</td><td>+0</td><td>+0.7</td><td>-2.7</td></tr><tr><td>angular rotate ( $\theta \approx \pi$ ) [38]</td><td>+8.8</td><td>+55.7</td><td>+0.7</td><td>+8.8</td></tr><tr><td>selective multi-layer rotate [38]</td><td>+15.5</td><td>+54.8</td><td>+2.4</td><td>+1.9</td></tr><tr><td>SEA [34] / SADI [40] edits</td><td>+10</td><td>+6</td><td>+1</td><td>+1.4</td></tr><tr><td>ITI per-head reflect [21]</td><td>+5.5</td><td>+1.4</td><td>+0</td><td>+0.2</td></tr><tr><td>SAE feature steer $^{\ddagger}$  [30]</td><td>+2.9</td><td>+1.2</td><td>+0.2</td><td>-1.2</td></tr><tr><td>LoRA [15] / ReFT [41] (trained)</td><td>+42.9</td><td>+57.1</td><td>+32.5</td><td>-2.6</td></tr><tr><td>card-C rung</td><td>reflect (1-D)</td><td>rotate (2-D)</td><td>none (trained)</td><td>none (not inv.)</td></tr><tr><td>inversion geometry</td><td>distributed</td><td>clean rotation</td><td>not-installable</td><td>flat (not inv.)</td></tr></table>

Shaded: training-free rotation wins trained edit needed . On Qwen3-VL-8B the training-free rotation (+55.7, 21.4 → 77.1, test > val: not overfit) matches trained LoRA (+57.1). <sup>‡</sup>The SAE feature diagnostic is exploratory and is not used as evidence. The training-free≈trained match on Qwen3-VL-8B holds over three seeds (§6), and the population picture is the decode̸=deploy forest of §6. The card-C rung is the minimal-complexity fixed correction within margin of the best lift (a per-model effective-dimensionality fingerprint); the four models shown are clear cases; borderline models (e.g. InternVL3-8B/14B) are reported separately.

Table 9: The inversion replicates across families; the minimal correction is per-model. Depth, n≈150, single seed; lifts over the no-edit baseline. Four newly added models (top) plus the three correction-battery models (bottom) all show inverted deployment $( a _ { \mathrm { d e c o d e } } { > } 0 , \ a _ { \mathrm { d e p l o y } } { < } 0 ) ;$ the training-free angular rotation that re-deploys depth on Qwen3-VL-8B [2] does not generalize cleanly, confirming that the rotation tie is model-specific and that the population claim is the universal inversion plus a per-model correction-complexity spectrum (§6). Lift values shown as percentages.

<table><tr><td>model</td><td>family</td><td>base</td><td> $a_{dec}$ </td><td> $a_{dep}$ </td><td>rotate*</td><td>recalib</td><td>correctability</td></tr><tr><td>InternVL3-2B [44]</td><td>InternVL3</td><td>38.3</td><td>+.74</td><td>-.11</td><td>-2.1</td><td>+1.4</td><td>suppress-only</td></tr><tr><td>InternVL3-8B</td><td>InternVL3</td><td>26.4</td><td>+.30</td><td>-.07</td><td>+8.1</td><td>+2.6</td><td>mild rotation</td></tr><tr><td>InternVL3-14B</td><td>InternVL3</td><td>31.4</td><td>+.50</td><td>-.19</td><td>-1.0</td><td>+6.2</td><td>recalib/reflect</td></tr><tr><td>LLaVA-OV-7B [19]</td><td>LLaVA</td><td>36.2</td><td>+.17</td><td>-.13</td><td>+13.8</td><td>+7.9</td><td>moderate rotation</td></tr><tr><td>Qwen3-VL-8B</td><td>Qwen3</td><td>21.4</td><td>+.08</td><td>-.15</td><td>+55.7</td><td>+6.0</td><td>clean rotation = trained</td></tr><tr><td>Qwen2.5-VL-7B</td><td>Qwen2.5</td><td>26.2</td><td>+.23</td><td>-.08</td><td>+15.5</td><td>+11.2</td><td>distributed (trained)</td></tr><tr><td>Pixtral-12B [29]</td><td>Mistral</td><td>39.8</td><td>+1.59</td><td>-.25</td><td>+2.4</td><td>+1.0</td><td>not-installable (trained)</td></tr><tr><td>Qwen2.5-VL-3B [3]</td><td>Qwen2.5</td><td>48.1</td><td>+.30</td><td>+.01</td><td>+8.8</td><td>+0.0</td><td>aligned-flat (not inverted)</td></tr></table>

$ { a _ { \mathrm { d e c } } } /  { a _ { \mathrm { d e p } } }  { : }$ signed steering slopes for probe-decodability and behaviour (both > 0 in the only non-inverted model). 7 of 8 models across 5 families are inverted; only the 3B scale-floor is not. rotate<sup>∗</sup> is the best norm-preserving angular rotation (training-free); recalib is the best 1-D scalar grid lift.

Table 10: The mechanism replicates on a second model (Qwen3-VL-8B [2] depth, base 21.4%). The deep localization-and-correction story established on the Qwen2.5-VL-7B [3] anchor reproduces qualitatively: a trained rank-1 edit recovers the axis (matching the anchor’s trained lift), and a 4-component ICA-subspace reflection also works—while the single probe direction and a closedform GEVD subspace both fail. The correctable channel is low-dimensional yet off the probe axis. Lift values shown as percentages.

<table><tr><td>correction</td><td>best setting</td><td>test</td><td>lift</td><td>reading</td></tr><tr><td>ReFT [41] (trained)</td><td>rank 1, L30</td><td>79.2</td><td>+57.8</td><td>low-rank trainable (val≈test)</td></tr><tr><td>ICA-subspace reflect</td><td>k=4, L30</td><td>81.0</td><td>+59.5</td><td>multi-component, off probe axis</td></tr><tr><td>probe-direction reflect</td><td>k=1...8</td><td>21.4</td><td>+0.0</td><td>single probe axis fails</td></tr><tr><td>GEVD subspace (closed-form)</td><td>k=1, L30</td><td>21.7</td><td>+0.2</td><td>closed-form basis fails</td></tr></table>

The trained ReFT lift (+57.8) matches this model’s correction-battery LoRA (+57.1, Table 8, LoRA [15]). That ICA (k=4) succeeds where the single probe direction does not echoes the cheap-lens vs. probe contrast and locates the readout error in a small, non-probe-aligned subspace.

Table 11: Causal localization: multi-layer band, full n (Qwen2.5-VL-7B [3], T16b). Correctiveinjection (recover, GT-signed) and project-out (ablate) at a 5-layer band around each axis’s decode peak. Recover measures whether the readout can use the direction if it is written in (causal usability, not a deployable accuracy gain); ablate measures necessity. Deltas vs. baseline in parentheses. Accuracies in %; deltas in percentage points.

<table><tr><td>Axis</td><td>band</td><td>baseline</td><td>recover</td><td>ablate</td><td>reading</td></tr><tr><td>Horizontal</td><td>L19–23</td><td>82.4</td><td>98.1 (+15.6)</td><td>82.2 (−0.2)</td><td>deployed; ablate flat (redundant)</td></tr><tr><td>Vertical</td><td>L21–25</td><td>58.5</td><td>70.2 (+11.7)</td><td>49.2 (−9.3)</td><td>recoverable &amp; necessary (ablate→chance)</td></tr><tr><td>Depth</td><td>L19–23</td><td>31.4</td><td>62.7 (+31.4)</td><td>36.6 (+5.2)</td><td>sign-flip crosses chance; ablate helps</td></tr></table>

Recover uses a GT-signed corrective injection: it shows the direction is causally usable, not that test accuracy improves. The arbiter (Table 3) shows the vertical lever operates on a prior rather than on vision.

Table 12: Decoding-optimal ̸= intervention-optimal (Qwen2.5-VL-7B [3], T14, full $n ) .$ . The probe direction (which maximizes decode, e.g. vertical 94) steers ∼3× weaker than diff-of-means and is not even necessary (vertical probe-ablate 57.7 ≈ baseline 58.5 ≈ random 58.6). bias = $P ( + { \mathrm { p o l e } } )$ span over the α sweep; oracle = corrective-oracle peak; ablate = project-out (random in parentheses). Bias ranges are $P \mathrm { ( + p o l e ) }$ in $\% ;$ oracle/ablate in % accuracy.

<table><tr><td>Axis</td><td>direction</td><td>bias range</td><td>oracle</td><td>ablate (rand)</td></tr><tr><td rowspan="2">Horizontal</td><td>diff-of-means</td><td>0.29 → 0.72</td><td>0.950</td><td>0.820 (0.824)</td></tr><tr><td>probe</td><td>47 → 57</td><td>87.8</td><td>82.8 (82.4)</td></tr><tr><td rowspan="2">Vertical</td><td>diff-of-means</td><td>0.55 → 0.85</td><td>0.739</td><td>0.500 (0.586)</td></tr><tr><td>probe</td><td>64 → 77</td><td>65.4</td><td>57.7 (58.6)</td></tr><tr><td rowspan="2">Depth</td><td>diff-of-means</td><td>0.57 → 0.28</td><td>0.464</td><td>0.366 (0.314)</td></tr><tr><td>probe</td><td>48 → 43</td><td>34.6</td><td>31.4 (31.4)</td></tr></table>

Table 13: The depth inversion is image-driven, not a fixed text-convention prior (Run 20, depth $n { = } 1 5 3 )$ . For the three clearly-inverted models we substitute a blank/gray image and measure the lopsidedness of the resulting answer (prior bias) and whether errors follow the prior pole (alignon-err). A fixed text prior would floor depth accuracy at ∼chance; the observed below-chance accuracies cannot be produced by the mild blank-image bias, so the model perceives the depth cue and deploys it with a flipped sign. All three verdicts are not-prior-dominated; a grounded horizontal control behaves the same way. Accuracies and biases in %.

<table><tr><td>model</td><td>depth acc</td><td>prior bias</td><td>align-on-err</td><td>verdict</td></tr><tr><td>Qwen2.5-VL-7B [3]</td><td>32.7</td><td>45.1</td><td>66.0</td><td>not-prior-dominated</td></tr><tr><td>Qwen3-VL-8B [2]</td><td>23.5</td><td>8.5</td><td>62.4</td><td>not-prior-dominated</td></tr><tr><td>Pixtral-12B [29]</td><td>41.2</td><td>8.5</td><td>71.1</td><td>not-prior-dominated</td></tr></table>

The verdict is driven by the model-specific below-chance accuracy (which differs across models, 23.5–41.2), so it is robust; we do not rely on the precise prior-bias magnitudes (the blank-image eval reported a suspiciously identical bias for two models and is flagged for re-verification). This establishes a negative answer to “why” (not a prior); a positive egocentric-vs-allocentric frame-flip test is left to future work.