# The Seriality Gap in Video Diffusion Models

Jorge Diaz Chao<sup>∗</sup> Konpat Preechakul<sup>∗</sup> Yuxi Liu Yutong Bai

UC Berkeley {jdiazchao,konpat,yuxi\_liu,yutongbai}@berkeley.edu

![](images/c59410ae06bbb813fce3824f55c8db1d0a86c319e2f0fe43a9d19dbd3c79578c.jpg)  
Can diffusion models predict long chains of dependent events?  
Figure 1: Dependent-event prediction exposes the seriality gap. (a) Hard-sphere dynamics separate non-serial from serial video prediction. (i) In the single-ball control, any future state can be computed directly from the initial state, without resolving intermediate states. (ii) With multiple balls, each ball–ball collision changes the state governing later collisions, creating dependent-event chains that must be resolved in temporal order. (b) Given initial frames, can a diffusion model remain accurate as longer prediction horizons demand more serial computation?

## Abstract

When one ball strikes another, then another, video models should predict the consequences of each bounce. In controlled experiments on multi-ball hard-sphere dynamics, we find that the performance of standard bidirectional video diffusion degrades as the causal chain lengthens, even when provided more denoising steps. In a length-matched single-ball control, where ball–ball interactions are absent, the degradation largely disappears, isolating dependent-event structure rather than video length as the cause. Across intervention studies, methods that increase effective serial computation improve performance disproportionately, including autoregressive/blockwise generation and architectural depth. We identify this pattern as the seriality gap: a mismatch between tasks requiring growing serial computation and video diffusion models whose denoising loop does not provide scalable serial compute. We then prove that, for deterministic video prediction, denoising steps do not add serial computation beyond the backbone, indicating a structural obstacle for video diffusion on serial reasoning and simulation tasks.

![](images/40bfd6d32e700317a7ba9a750c34aaf70426a6978f93e970a6a955566107e433.jpg)

## 1 Introduction

When one ball strikes a second, which then strikes a third, predicting the chain of consequences seems like a minimal competence for any video model. Yet, even as video diffusion models generate compelling clips and are increasingly positioned as world simulators and visual reasoners [1, 2], persistent failures in physical consistency [3–9] and reasoning [2, 10] remain. What if this is due not only to a lack of data coverage but also to a limitation of the diffusion model itself? We therefore ask the following:

## Can video diffusion models predict ever-longer chains of dependent events?

This simple question is consequential precisely because the negative answer would be surprising. Because diffusion generates videos through repeated denoising, failure on this task would call into question what computation those iterations actually provide—an issue that has recently received substantial attention but remains unresolved. Some studies attribute reasoning to computation across denoising steps [11]; others report erosion of motion structure over later steps [12], early plan commitment and the need for chained generations for longer solutions [10], or rapidly saturating gains from additional steps [13]. Together, these findings motivate characterizing whether denoising supplies computation that scales with dependency-chain length (Section 6).

We study this question with a deliberately minimal video-prediction task: hard-sphere dynamics. Identical balls move frictionlessly in a box and undergo elastic collisions; given the initial frames of a trajectory, the model must predict the future video (Figure 1b). Here, visual complexity is minimal, the dynamics are deterministic, and prediction errors are unambiguous. Our setup varies the length of the dependent-event chain: with n > 1 balls, longer f-frame rollouts contain longer chains of ball–ball interactions, whereas n = 1 acts as a control with no such interactions and admits a closed-form solution. This lets us vary serial complexity while holding visual content essentially fixed, a control difficult to obtain in natural video (Section 2).

The task’s simplicity is deliberate. Natural video adds appearance complexity, uncertainty, and ambiguity, but none removes the need to propagate dependent events. Failure in this controlled setting therefore identifies a bottleneck that may persist in richer settings; conversely, success would address that bottleneck, not solve video modeling as a whole.

![](images/542f0afc02af01402ba943c4068716c709ee5ffd0d6ff34da2de305f584b2ecc.jpg)

![](images/9d6c112b78cab69d3364e13d26cf16906cbaeb736c37155a7e8596756366733a.jpg)

![](images/f292cea01021e11a857d529efa32b9d5486f884ebd39a3edb48371649690e8c1.jpg)

![](images/1f809b40ac6767e7ce02a30b025bf9a81568a11a5b12bc2f15bd4995522047c6.jpg)

![](images/e957e6c3331dd0d89b62b1dbe6c4c5908af43809d8c5a97b5bb838423f4ded65.jpg)  
Figure 2: Serial scaling. (a) Bidirectional diffusion jointly denoises future frames, but denoising does not supply scalable serial compute (Section 4). Consistent with our theory, performance improves with (b) autoregressive/blockwise inference factorization or (c) a deeper backbone.

The seriality gap (Section 3) In controlled multi-ball (n = 5) experiments, standard bidirectional video diffusion degrades as causal chains lengthen. This degradation largely disappears in the length-matched single-ball (n = 1) control, showing that the gap is driven by more difficult event dependencies rather than video length. Increasing denoising steps provides little benefit and does not close the gap. By contrast, at comparable total compute, interventions that add effective serial computation help disproportionately: increasing depth is more effective than increasing width, and serializing generation yields monotonic improvements—from bidirectional generation through progressively smaller temporal blocks to full autoregression.

Why denoising steps do not close the gap (Section 4) The result is counterintuitive for two reasons. First, longer rollouts already give bidirectional video DiTs [14] far more total compute than even a simulator would require: temporal self-attention scales as $\bar { \mathcal { O } } ( \bar { f } ^ { 2 } )$ ), while a physical rollout needs only $\mathcal { O } ( f )$ . Second, diffusion denoising is itself iterative, suggesting that more steps could close the seriality gap. Both intuitions fail. The issue is not total computation but scalable serial computation.

For a closed, deterministic video process, the future is uniquely determined by the initial frames. If the score network computes the exact score, a single evaluation contains enough information to recover that future (Proposition 4.1). Repeated denoising therefore does not provide additional serial computation beyond what the backbone has already computed. When predicting dependent events requires serial computation beyond the capacity of a fixed-depth backbone, the required computation must instead come from greater backbone depth or an inference factorization—such as autoregression—aligned with the temporal dependency graph (Figure 2). This explains why depth and temporal factorization help to predict dependent events, while additional denoising steps do not.

Beyond hard-sphere dynamics Under accurate score matching, denoising is not an independent source of scalable serial computation beyond the backbone. This limitation is not specific to hardsphere dynamics: it can arise whenever diffusion models face tasks whose serial demands grow with problem size, including maze and puzzle solving. In such tasks, however, the dependency structure need not follow temporal order, so frame-wise autoregression is not a universal remedy. The needed serial computation must instead come from greater backbone depth, a task-aligned inference factorization, or potentially a new class of non-score-matching generative models.

## Contributions

(i) We introduce a video-prediction testbed based on hard-sphere dynamics for studying dependentevent prediction. The testbed varies serial complexity while holding visual content essentially fixed (Section 2).

(ii) We identify the seriality gap: the performance of bidirectional video diffusion degrades as chains of dependent events lengthen, despite increasing total compute. This degradation is not explained by video length or closed by additional denoising steps, but is reduced by interventions that add effective serial computation (Section 3).

(iii) We show theoretically that, for deterministic video prediction, denoising steps do not add scalable serial computation beyond what is already computable by the backbone (Section 4).

## 2 Hard-Sphere Dynamics as a Testbed for Dependent-Event Prediction

In order to study serial temporal prediction, we design a setting where dependency length can vary while visual complexity and ambiguity remain controlled. We use hard-sphere dynamics (Figure 1a): n identical balls move frictionlessly in a square box and undergo perfectly elastic collisions with walls and with one another. The rendering is simple, the dynamics are deterministic, and each initial condition has a single ground-truth rollout. This deliberately minimal setting is diagnostic—if a video model fails here, the failure is unlikely to be explained by visual complexity or ambiguous futures.

In this setting, serial dependence is determined by the number of ball–ball collisions. This lets us compare multi-ball $( n > 1 )$ rollouts, where collision chains grow with the prediction horizon, against single-ball $( n = 1 )$ rollouts, which can be equally long but have closed-form, non-serial futures. This contrast separates difficulty due to video length from difficulty due to serial dependence.

## 2.1 Experimental Setting

Dataset For each dataset configuration, we synthesize f-frame hard-sphere simulation videos at $1 2 8 \times 1 2 8$ resolution. Because the simulator is deterministic, each initial condition has a unique ground-truth rollout, so prediction error can be measured directly and unambiguously. Appendix G documents the simulator, initial-condition sampling, and filtering criteria.

We construct two dataset families. In the serial setting $( n = 5 )$ , our filtering procedure enforces a minimum post-conditioning ball–ball collision density, so the minimum number of collision events grows with the $f .$ -frame prediction horizon. Since each collision can change the state that determines subsequent collisions, these events must be resolved in temporal order, making f a proxy for serial complexity. In the length-matched control $( n = 1 )$ , ball–ball collisions are absent, so any future state can be computed in closed form from the initial conditions without simulating intermediate states.

As $f$ increases, bidirectional video diffusion models process larger spatiotemporal tensors and therefore receive more total computation [15]. If this computation were sufficient for tracking dependent events, local physical accuracy in the multi-ball setting should not degrade substantially more with $f$ than in the single-ball control.

Models We train DiT-style [14] video diffusion models adapted from Wan2.1 [15], a standard bidirectional video diffusion architecture with competitive performance. Every model operates in the Wan2.1 VAE latent space, which downsamples videos by $8 \times$ along each spatial axis and 4 temporally. The patch embedder groups $2 \times 2$ spatial latent patches without additional temporal downsampling, so one temporal latent frame corresponds to four rendered video frames.

We compare standard bidirectional denoising with variants that allocate computation more serially, as summarized in Figure 2. Autoregressive and block-autoregressive variants use the same base architecture but replace full temporal attention with FlexAttention [16] block-causal masks over temporal latent frames. We write Block-k for generation in blocks of k temporal latent frames; Appendix H shows the training and inference attention masks. Unless otherwise specified, evaluation uses $T = 5 0$ denoising steps. To separate inference factorization from architecture, we also vary backbone width $d _ { \mathrm { m o d e l } }$ and depth $n _ { \mathrm { l a y e r s } }$

Training Models are trained for 200,000 iterations with a batch size of 64 using AdamW with a weight decay of 0.01 and a constant learning rate of $2 \times 1 0 ^ { - 4 }$ . We maintain an exponential moving average (EMA) of the weights with decay 0.9999 and use the EMA weights for evaluation. The reported training runs require approximately 300 A100-80GB GPU-days.

Evaluation We evaluate predictions along three axes.

1. Global error $\Delta x ^ { \mathrm { ( G T ) } }$ measures the average Euclidean distance between predicted and groundtruth ball centers across all balls and evaluated frames. This is the most direct notion of prediction accuracy, but it is sensitive to accumulated long-horizon drift.

2. Local error $\Delta x ^ { ( k ) }$ measures local physical consistency. For each generated frame, we go back k frames, extract ball centers and velocities from the generated pixels, roll the simulator forward for k steps, and compare the result to the generated frame. This avoids penalizing accumulated drift caused by state-estimation errors. By default, we report $\Delta x ^ { ( 5 ) }$

3. Visual quality (IoU) measures shape fidelity independently of position error. For each ball, we center-align its generated silhouette with an ideal circular mask of the correct radius and compute their intersection over union.

Let $\mathcal { T } = \{ f _ { 0 } + 1 , \ldots , f \}$ denote the predicted frames used for evaluation, with $f _ { 0 } = 5$ conditioning frames in all experiments. We define the global and local accuracy metrics as follows:

$$
\Delta x ^ {\mathrm{(GT)}} = \frac {1}{| \mathcal {I} | n} \sum_ {t \in \mathcal {I}} \sum_ {i = 1} ^ {n} \left\| \hat {x} _ {t, i} - x _ {t, i} ^ {\mathrm{GT}} \right\| _ {2},
$$

$$
\Delta x ^ {(k)} = \frac {1}{| \mathcal {I} | n} \sum_ {t \in \mathcal {I}} \sum_ {i = 1} ^ {n} \left\| \hat {x} _ {t, i} - \tilde {x} _ {t, i} ^ {(k)} \right\| _ {2},
$$

where $\hat { x } _ { t , i }$ denotes the predicted position of ball i at time $t , x _ { t , i } ^ { \mathrm { G T } }$ the ground-truth position, and $\tilde { x } _ { t , i } ^ { ( k ) }$ the position obtained by extracting the generated state at time $t - k$ and rolling the simulator forward k steps. A human evaluation in Appendix F confirms that large gaps in these metrics are salient.

Since models predict pixels rather than coordinates, we recover $\hat { x } _ { t , i }$ by thresholding the known color range of ball i in each generated frame and taking the centroid of the segmented mask; $x _ { t , i } ^ { \mathrm { G T } }$ is obtained from the simulator.

## 3 The Seriality Gap in Hard-Sphere Dynamics

We summarize the empirical findings as five observations. Appendix B gives full results; Appendix E rules out optimization and initialization confounds.

![](images/506947827583514ea8a4549444f9760a6b9cf1f70e44da13ca144c171d039714.jpg)

![](images/34f8d8d466e2296eea2bd8cbd90a1d8118be0f8cebf0dfe12560abf29b4f6b23.jpg)

![](images/3c138f0913a6ca3b3f9f8105475d1994f5bb3eee22705a613b9f0e9ffbe51584.jpg)

![](images/b983725565bf8c4c2d0c570c88333ec09d90cf39c831f96b71518bc24f77345f.jpg)  
AR Steps / Out frames ( serial compute)  
Figure 3: Empirical evidence for the seriality gap. (Top left) Bidirectional diffusion degrades as collision chains lengthen, while autoregressive generation remains more accurate. (Top right) Increasing the number of denoising steps does not close the gap. (Bottom left) Changing backbone depth and width probes whether architectural serial computation helps. (Bottom right) Performance improves as inference becomes more temporally serial, interpolating from bidirectional to blockwise to autoregressive generation. All plotted errors are $\Delta x ^ { ( 5 ) }$ (lower is better).

Observation 1 Bidirectional diffusion degrades as videos get longer. In the $n = 5$ setting, our filtering criterion makes the minimum number of post-conditioning ball–ball collisions grow with video length f, thereby increasing serial complexity (Appendix G). Although both bidirectional and autoregressive models receive more inference compute at longer horizons, with temporal self-attention scaling roughly as $\mathcal { O } ( f ^ { 2 } )$ , autoregressive generation maintains lower and nearly flat local physical error $\Delta x ^ { ( 5 ) }$ . Bidirectional diffusion instead worsens with $f ,$ as shown in Figure 3 (top left).

Observation 2 The gap disappears without ball–ball collisions. As a length-matched control, we set $n = 1$ , removing ball–ball collisions and leaving trajectories that can be computed in closed form from the initial state. In this non-serial setting, bidirectional and autoregressive models have nearly identical local physical error $\Delta x ^ { ( 5 ) }$ , and both remain stable as f increases, as shown in Figure 3 (top left). Together with the $n = 5$ result, this control shows that video length alone cannot explain the degradation and supports serial complexity as its key driver.

Observation 3 Serializing inference improves performance. We interpolate from bidirectional to autoregressive diffusion using block-causal masks over temporal latent frames, keeping the backbone fixed. Larger blocks denoise more frames jointly, while smaller blocks impose a more sequential generation order. Local physical error improves as the factorization becomes more serial, following bidirectional (Block-11)  Block-3  Block-2  autoregressive (Block-1) across model sizes, as shown in Figure 3 (bottom right). Since model capacity is unchanged, these gains reflect temporal compute allocation rather than a larger backbone.

Observation 4 More denoising steps do not close the gap. A natural explanation is that bidirectional diffusion may simply be under-computed at inference time. If denoising iterations supplied the serial computation needed to predict longer collision chains, increasing T should especially improve long-horizon rollouts. We therefore evaluate bidirectional diffusion with $T \in \{ 1 0 , \bar { 2 } 0 , 5 0 , \bar { 1 } 0 0 , \bar { 2 } 0 0 \}$ across all video lengths. Performance improves up to roughly $T = 5 0 ;$ beyond this point, additional denoising steps provide no consistent benefit and sometimes increase error, while the long-horizon degradation remains. Increasing T therefore does not close the gap to autoregressive generation, as shown in Figure 3 (top right).

Observation 5 Depth scaling is more effective than width scaling. We next ask whether the missing serial computation can be supplied by the backbone itself. We sweep bidirectional models over width $d _ { \mathrm { m o d e l } }$ and depth $n _ { \mathrm { l a y e r s } } .$ , varying how capacity and compute are allocated. Bidirectional models benefit more strongly from increasing depth than increasing width up to about 30 layers. At comparable budgets, deeper, narrower models outperform shallower, wider ones, and the gain from $n _ { \mathrm { l a y e r s } } = 1 0$ to 30 layers exceeds the improvements from widening a 30-layer model from $d _ { \mathrm { m o d e l } } = \mathrm { { 6 4 0 } }$ to 1536, as shown in Figure 3 (bottom left). Beyond 30 layers, additional depth gives diminishing returns, possibly because deeper models become harder to optimize [17].

![](images/88730c78b38f4094fb7446d981afb9c467f557a1d47699e766c532ed2f9fe34a.jpg)  
(a)

![](images/94fe371899aa33475578046ddf23762112373f96daa3d81a740baf033e8879c8.jpg)  
(b)

![](images/bc24f5f7b6bb89539c65dd4a9529d6d262aed6b0cfadf588260a36ec27247f20.jpg)  
(c)  
Figure 4: Illustrative failure modes in bidirectional generations. We show common qualitative failures produced by the bidirectional diffusion model trained on $f = 4 9 , n = 5 , d _ { \mathrm { m o d e l } } = 1 5 3 6$ and $n _ { \mathrm { l a y e r s } } = 3 0$ . (a) The dark-blue ball splits into two, resulting in a ball-count violation. (b) The dark-blue and light-blue balls exchange colors, while the green and yellow balls merge into one; both failures violate state consistency. (c) The dark-blue ball should bounce off the light-blue ball, but instead tunnels through it. Additional qualitative examples, including side-by-side comparisons with autoregressive generations, are shown in Appendix C.

Qualitative failure modes We observe three recurring state-consistency failures, illustrated in Figure 4: tunneling, ball-count violations, and color-identity violations. In tunneling, a ball passes through another object or disappears and reappears elsewhere without a physically valid intermediate trajectory. In ball-count violations, two balls merge into one or one ball splits into multiple balls. In color-identity violations, a ball changes color, duplicates another ball’s color, or swaps its color identity with another ball. These failures can occur in both bidirectional and autoregressive generations, but they are much more frequent in bidirectional models, especially for high-velocity balls and long multi-ball rollouts.

In the backbone depth-width ablation, all model sizes produce visually plausible videos, but smaller and especially shallower backbones exhibit these state-consistency failures more often.

## 4 Why Denoising Steps Do Not Provide Scalable Serial Computation

The observations above are puzzling. As videos get longer, bidirectional diffusion models already receive roughly $\mathcal { O } ( f ^ { 2 } )$ inference compute from self-attention, while the ground-truth simulator requires only $\mathcal { O } ( f )$ computation. Moreover, because our main metric is local accuracy, the degradation cannot be explained merely by long-horizon drift. The issue is therefore not simply a lack of total compute, but a failure to convert that compute into scalable serial computation for predicting dependent events.

Through a complexity-theoretic lens, this is not so puzzling. Video diffusion models are asked to solve an inherently serial problem, but their denoising steps do not provide scalable serial computation. To make this precise, we first introduce the relevant complexity classes.

At the parallelizable end, the class ${ \mathsf { T C } } ^ { 0 }$ captures polynomial-size, constant-depth threshold circuits; informally, these are computations closer to a closed-form shortcut than to a step-by-step simulation.

At the serial end, P-complete problems capture polynomial-time computations believed not to be efficiently parallelizable.<sup>1</sup>

In our case, the multi-ball task shares the elasticcollision primitive with classical P-complete billiard-ball computation [19, 18]. Meanwhile, fixed-depth transformer backbones are boundeddepth parallel computations under standard precision assumptions [20]. Thus, a fixed-depth backbone cannot be the source of serial computation that grows with the task. The natural hope is that repeated denoising steps supply the missing serial computation; we now show why this does not happen for deterministic video prediction.

Deterministic video prediction If we assume that the physical process is deterministic, then, conditional on the initial state of the system, the set of valid physical trajectories is a single point in the space of all possible trajectories, because there is only one correct trajectory starting from that point, no matter how long the trajectory is.

Given a single initial frame of a video, we may not know the exact initial state of the system. For example, we may only observe the positions but not the velocities of bouncing balls. However, in that case, we need to observe just 2 frames to determine both the position and the velocity of each ball, and thus determine the state of the system. In general, if we can determine the exact state of the system after observing $\mathcal { O } ( 1 )$ frames, there is only one valid video trajectory from then on. This condition is known as full observability.

Under deterministic $x _ { i + 1 } = \Phi ( x _ { i } )$ 2

![](images/321e501d93bd693d5902f12569b5bcadc9242d97692a683c7981c3440b76f1ec.jpg)

![](images/006f1257faf6a6cd47d05f34243569fa9f439a3d2eb027071a9172df3d5e9fb0.jpg)

$$
w _ {0} ^ {'} = \underbrace {\frac {w _ {t} + \sigma (t) ^ {2} s (w _ {t} \mid u , t)}{\sqrt {1 - \sigma (t) ^ {2}}}} _ {\mathrm{known}}
$$

Figure 5: Theory overview. Deterministic dynamics map an initial state $s _ { 0 }$ to a unique future trajectory through repeated transitions $s _ { i + 1 } =$ $\bar { \Phi ( s _ { i } ) }$ Given $s _ { 0 } ,$ the conditional distribution over futures is therefore a point mass. Under Gaussian noising, the exact conditional score at any fixed noise level points back to this unique future, so one score-network evaluation can recover it (eq. (3)). Because the clean future is recovered from a single score-network evaluation, denoising steps do not add scalable serial computation beyond what the backbone already computes.

Under these assumptions, the video prediction problem reduces to computing a deterministic function:

$$
\mathcal {O} (1) \text {   initial   frames   } \xrightarrow {f} \text {   future   frames.   }
$$

We will formally set up two problems, one about video prediction and one about state prediction. We then argue:

1. If the video prediction problem is solvable by a diffusion model with a ${ \mathsf { T C } } ^ { 0 }$ backbone, then its underlying state prediction problem is ${ \mathsf { T } } { \bar { \mathsf { C } } } ^ { 0 }$

2. Bouncing balls and, indeed, most state prediction problems are P-complete and $\mathbf { S O } ,$ , under standard assumptions in computational complexity theory, are not in ${ \mathsf { T C } } ^ { 0 }$

3. Thus, most video prediction problems are not solvable by a diffusion model with $\mathsf { a } \mathsf { T } \mathsf { C } ^ { 0 }$ backbone.

From this perspective, the deficiency of a diffusion model is that it converges in $\mathcal { O } ( 1 )$ serial steps, despite expending $\mathcal { O } ( f ^ { 2 } )$ parallel compute per serial step. In contrast, an autoregressive model spends $\mathcal { O } ( \dot { \boldsymbol { f } } )$ serial steps, sufficient to solve P-complete problems.

## 4.1 Problem Formalism

Suppose we are given a deterministic process. Let the state space be $X ,$ , and let the deterministic time evolution be $\Phi : X \to X$ . We have an observation function $F : X \to Y$ , which takes a state $x \in X$ and converts it into an image $F ( x ) \in Y$ . Here, $Y$ is the space of possible images. For example, if we consider an image to be an $H \times { \dot { W } }$ array of pixels, then $\dot { \boldsymbol { Y } } = \mathbb { R } ^ { \dot { \boldsymbol { H } } \times \boldsymbol { W } }$

Suppose that the physical process is fully observable. That is, we assume that there exists a constant number $f _ { 0 }$ such that, given any two trajectories

$$
x _ {0}, x _ {1} = \Phi (x _ {0}), x _ {2} = \Phi (x _ {1}), \ldots ; x _ {0} ^ {\prime}, x _ {1} ^ {\prime} = \Phi (x _ {0} ^ {\prime}), x _ {2} ^ {\prime} = \Phi (x _ {1} ^ {\prime}), \ldots ,
$$

$$
F (x _ {0}) = F \left(x _ {0} ^ {\prime}\right), F \left(x _ {1}\right) = F \left(x _ {1} ^ {\prime}\right), \dots F \left(x _ {f _ {0}}\right) = F \left(x _ {f _ {0}} ^ {\prime}\right), \text {then} x _ {0} = x _ {0} ^ {\prime}.
$$

In other words, we can fully determine $x _ { 0 }$ if we observe $f _ { 0 } ~ + ~ 1$ consecutive frames $F ( x _ { 0 } ) , F ( x _ { 1 } ) , \hdots , F ( x _ { f _ { 0 } } )$

We also assume that observability is computable in constant time. That is, we have a constant-time algorithm that, given $F ( x _ { 0 : f _ { 0 } } )$ , returns $x _ { 0 : f _ { 0 } }$ . In our case of bouncing balls, this is clear: given two frames, we can find the velocity of the balls by taking their difference.

The key point is simple. For deterministic video prediction, once the conditioning frames determine the state, there is only one correct future video. After Gaussian noising, the conditional distribution remains centered on that unique future, so the exact score at any fixed noise level contains enough information to recover it in one step. Repeated denoising therefore does not add scalable serial computation beyond what the backbone already provides, as summarized in Figure 5.

Given this setup, we can define two formal problems:

1. Video segment prediction: given an initial video segment $F ( x _ { 0 : f _ { 0 } } )$ of a physical trajectory and a number $f ,$ compute its future video segment $\bar { F } ( x _ { f _ { 0 } + 1 : f } )$

2. State prediction: given an initial physical trajectory $x _ { 0 : f _ { 0 } }$ and a number $f ,$ compute $x _ { f }$

Suppose further that we have a diffusion model that solves the following problem: given an initial video segment $y _ { 0 : f _ { 0 } } : = F ( x _ { 0 : f _ { 0 } } )$ of a physical trajectory and a number $f _ { i }$ compute its future video trajectory $y _ { f _ { 0 } + 1 : f } : = F ( x _ { f _ { 0 } + 1 : f } )$

## 4.2 Diffusion Modeling

Now, we specify what diffusion modeling means. We use the noise-conditional score-matching network (NCSN) formalism. Diffusion models in other formalisms can be converted to an equivalent NCSN diffusion model [21]. The full formalism is in Appendix A.

Diffusion modeling is a technique to sample from a conditional probability distribution. Let w be an object we want to sample, and let $u$ be a conditioning input. w is an element of a sample space, here taken to be $\mathbb { R } ^ { d }$ , where d is some positive integer.<sup>2</sup> For example, in our case of video modeling, the condition is $y _ { 0 : f _ { 0 } }$ , the initial video segment, and the object we want to sample is $y _ { f _ { 0 } + 1 : f }$ , the rest of the video, within the sample space $\mathbb { R } ^ { \mathbf { \breve { H } } \times W \times ( f - f _ { 0 } ) }$

Now, we define a forward diffusion process. We use the standard Gaussian diffusion process. The process is defined by a noise rate function $\beta : [ 0 , \infty )  ( 0 , \infty )$ ), with the condition that $\begin{array} { r } { \int _ { 0 } ^ { \infty } \bar { \beta } ( t ) d t = \infty } \end{array}$ . We first sample a starting point $\boldsymbol { w } _ { 0 } \sim \rho _ { 0 } ( \boldsymbol { w } | \boldsymbol { u } )$ and then incrementally add white noise at a rate of $\beta ( t )$ . The accumulated effect is that, at t, the noised sample has the following distribution:

$$
w _ {t} = \sqrt {1 - \sigma (t) ^ {2}} w _ {0} + \sigma (t) z, \quad z \sim \mathcal {N} (0, I _ {d}),\tag{1}
$$

where $\sigma ( t ) ^ { 2 } = 1 - e ^ { - \int _ { 0 } ^ { t } \beta ( \tau ) d \tau }$ is the variance due to accumulated noise.

In other words, the probability distribution of $w _ { t }$ , conditional on u, is the convolution of a scaled version of $\rho _ { 0 } ( \cdot | u )$ with a Gaussian function. The probability distributions $\rho ( \cdot | u , t )$ are increasingly blurry versions of the scaled original $\rho _ { 0 } ( \cdot | u )$ , starting at $\rho ( \cdot | u , 0 ) = \rho _ { 0 } ( \cdot | u )$ and ending with lim $1 _ { t  \infty } \rho ( \cdot | u , \infty ) = \rho _ {  { N } ( 0 , I _ { d } ) }$ , where we lose all traces of the original and obtain the standard Gaussian distribution.

Define the score function

$$
s (w _ {t} | u, t) := \nabla_ {w _ {t}} \log \rho (w _ {t} | u, t).\tag{2}
$$

The score function can be used to define a backward stochastic process, starting from $\mathcal { N } ( 0 , I _ { d } )$ and ending with the original $\rho _ { 0 } ( \cdot | u )$ . In the NCSN formalism, we use a neural network $s _ { \theta }$ to approximate s. This network is the titular “noise conditional score network” (NCSN). We then approximate the backward process by numerical stochastic integration.

The exact score function s has no closed-form solution in general, because $\rho _ { 0 } ( w _ { 0 } | u )$ is too generic to be integrated. However, if $w _ { 0 }$ is a deterministic function of $u ,$ then the score function has a closed-form solution, which allows us to solve for $w _ { 0 } ( u )$ in one step:

$$
w _ {0} (u) = \frac {\sigma (t) ^ {2} s (w _ {t} | u , t) + w _ {t}}{\sqrt {1 - \sigma (t) ^ {2}}}.\tag{3}
$$

In particular, this means that if we have a neural network that computes the score function, we can use it to compute $w _ { 0 } ( u )$ for any choice of $t > 0$ and any choice of $w _ { t }$ . Intuitively, this is possible because the flows in the corresponding flow-matching problem are just straight lines and can therefore be solved in one step without evaluating any integrals (Figure 5).

Now we can frame a video diffusion model in this formalism. Its backbone network takes as inputs an initial video segment $y _ { 0 : f _ { 0 } } .$ , a positive integer $f > f _ { 0 } .$ , a noise level t, and a noisy future video segment $\hat { y } _ { f _ { 0 } + 1 : f }$ . It outputs the noise-conditional score:

$$
s (\hat {y} _ {f _ {0} + 1: f} | y _ {0: f _ {0}}, f, t) = \nabla_ {\hat {y} _ {f _ {0} + 1: f}} \log \rho (\hat {y} _ {f _ {0} + 1: f} | y _ {0: f _ {0}}, t).\tag{4}
$$

Proposition 4.1. If a diffusion model solves the video prediction problem and its NCSN s<sub>θ</sub> is a ${ \mathsf { T C } } ^ { 0 }$ network computing the exact score (2), then the state prediction problem is in ${ \mathsf { T C } } ^ { 0 }$

Proof. We solve the state prediction problem by an algorithm that calls the NCSN just once. The entire algorithm runs in ${ \mathsf { T C } } ^ { 0 }$ . The details are in Appendix A. □

Comment. Is it unrealistic to assume that the score function is matched exactly? We use exact score matching to present the core argument as clearly as possible, but the conclusion does not require exact equality. Even when $s _ { \boldsymbol { \theta } } \approx s ,$ , we can still prove that the state prediction problem is in ${ \mathsf { T C } } ^ { 0 }$ given a sufficiently strong upper bound on $| s _ { \theta } - s | ;$ see [22, App. F].

The diffusion models considered here are trained to approximate the score, so large score errors reflect optimization failure or insufficient model capacity. Scaling model size and training compute should improve score estimation. This makes our proposition more relevant at scale.

On a positive note, our result points toward a different class of iterative generative models. Rather than training each iteration only to approximate the score, such models could train their intermediate states to perform and preserve useful computation across iterations. Developing objectives that enable this behavior is an open direction.

## 5 Discussion and Limitations

The testbed isolates dependent-event prediction Hard-sphere dynamics is a diagnostic abstraction: by removing visual realism, semantic ambiguity, stochasticity, and open-world uncertainty, it isolates a single question: can a video diffusion model track a growing chain of dependent events? The seriality gap is therefore not about bouncing balls or physics specifically, but about a broader class of tasks whose predictions require serial chains of deterministic transitions through intermediate states.

Autoregression is not a universal fix Autoregression helps in our setting because the dependencies are mostly temporal: each short future block depends on the recent past, so serial generation matches the causal order. Other visual reasoning tasks need not factor this way. Visual-reasoning studies instead find early global planning or progressive refinement across denoising steps [2, 10, 11]; such tasks need not factor into frame-aligned transitions. Autoregression helps when its factorization aligns with task dependency structure; otherwise, the needed serial computation must come from depth, search, planning, or another mechanism.

Moreover, even if more serial computation is supplied, it does not guarantee that a model will use it to track event dependencies. In natural video, finite model capacity must also represent appearance and stochasticity. These competing demands may dominate learning, allowing models–autoregressive or bidirectional–to favor visually plausible outputs over faithful dependent-event prediction. Our controlled testbed removes these confounds and therefore establishes a clear computational advantage, not a guarantee of success on unconstrained real-world video.

Limitations Our theory applies to deterministic video prediction and is one-way: it does not characterize every setting in which denoising steps fail to provide scalable serial computation. Stochasticity or noise can break the assumptions, but this does not imply diffusion models acquire scalable serial computation. Indeed, our vibrating-wall hard-sphere dynamics variant in Appendix D shows the same qualitative pattern: even when the deterministic assumption is violated, autoregressive generation remains more locally consistent than bidirectional denoising. This suggests that violating the theorem’s assumptions does not necessarily remove the seriality gap.

## 6 Related Work

Video models as simulators and reasoners Video generators are increasingly framed as more than synthesis systems: they are proposed as world simulators, physical reasoners, and visual planning models [1, 15, 2, 23, 24]. This has led to benchmarks that test physical consistency, temporal coherence, and physical commonsense, including PhyGenBench, T2VPhysBench, Physics-IQ, Morpheus, VBench, and VBench2 [3–9]. Recent maze-solving studies push the same view toward visual planning and reasoning [2, 10]. Synthetic physics environments provide a complementary way to remove visual realism as a confound: Kang et al. [25] study video diffusion models in the PHYRE physical simulator [26], showing that models can struggle with physical laws even in visually simple settings. Our hard-sphere dataset follows this controlled-evaluation philosophy, but focuses on the question: can video diffusion models track a chain of dependent events when the ground truth is unambiguous and visual realism is not a confound?

Video diffusion architectures and inference paradigms Modern video generators commonly apply latent diffusion to spacetime tensors using transformer or U-Net backbones with temporal attention [27–29, 15]. The dominant inference paradigm is bidirectional denoising: all frames are refined jointly by a bounded-depth backbone with access to both past and future frames. Recent work explores autoregressive and sliding-window diffusion for streaming and long-horizon generation [30–32]. We treat these inference choices not only as engineering tradeoffs, but as different ways of allocating serial computation across time.

Other temporal bottlenecks in video diffusion Prior work targets temporal bottlenecks through noise rescheduling [33, 34], joint appearance–motion representations [35], temporal regularization [36], or preservation of few-step motion priors during refinement [12]. These methods improve temporal structure; our focus is different: better priors may improve coherence but do not address bidirectional denoising’s lack of scalable serial computation for long dependency chains.

Our explanation connects these empirical failures to prior work on transformer parallelism, depth– width tradeoffs, and serial scaling, which argues that model depth, width, and denoising steps are not interchangeable computational resources [20, 37, 38, 22].

## Acknowledgments and Disclosure of Funding

We thank Alexei Efros for his constant efforts to debate with us. We thank Nicolas Dufour for suggestions for improving experimental rigor. We thank Amil Dravid for his suggestions to add “more dots” to the experiments section. We thank other members of Berkeley AI Research for helpful discussions. KP and YB are supported by ONR MURI.

## References

[1] Tim Brooks, Bill Peebles, Connor Holmes, Will DePue, Yufei Guo, Li Jing, David Schnurr, Joe Taylor, Troy Luhman, Eric Luhman, Clarence Ng, Ricky Wang, and Aditya Ramesh. Video generation models as world simulators, 2024. URL https://openai.com/research/ video-generation-models-as-world-simulators.

[2] Thaddäus Wiedemer, Yuxuan Li, Paul Vicol, Shixiang Shane Gu, Nick Matarese, Kevin Swersky, Been Kim, Priyank Jaini, and Robert Geirhos. Video models are zero-shot learners and reasoners, 2025. URL https://arxiv.org/abs/2509.20328.

[3] Fanqing Meng, Jiaqi Liao, Xinyu Tan, Wenqi Shao, Quanfeng Lu, Kaipeng Zhang, Yu Cheng, Dianqi Li, Yu Qiao, and Ping Luo. Towards world simulator: Crafting physical commonsense-based benchmark for video generation, 2024. URL https://arxiv.org/abs/2410.05363.

[4] Xuyang Guo, Jiayan Huo, Zhenmei Shi, Zhao Song, Jiahao Zhang, and Jiale Zhao. T2vphysbench: A first-principles benchmark for physical consistency in text-to-video generation, 2025. URL https: //arxiv.org/abs/2505.00337.

[5] Saman Motamed, Laura Culp, Kevin Swersky, Priyank Jaini, and Robert Geirhos. Do generative video models understand physical principles?, 2025. URL https://arxiv.org/abs/2501.09038.

[6] Tim Rädsch, Yuki M. Asano, Hilde Kuehne, Stefan Bauer, Priyank Jaini, Robert Geirhos, and Carsten T. Lüth. Physics-iq verified, 2026. URL https://arxiv.org/abs/2606.18943.

[7] Antonios Tragoudaras, Chenyu Zhang, Daniil Cherniavskii, Antonios Vozikis, Thijmen Nijdam, Derck W. E. Prinzhorn, Mark Bodracska, Nicu Sebe, Andrii Zadaianchuk, and Efstratios Gavves. Evaluating newtonian mechanics in video generative models with real physical systems, 2026. URL https://arxiv. org/abs/2504.02918.

[8] Ziqi Huang, Yinan He, Jiashuo Yu, Fan Zhang, Chenyang Si, Yuming Jiang, Yuanhan Zhang, Tianxing Wu, Qingyang Jin, Nattapol Chanpaisit, Yaohui Wang, Xinyuan Chen, Limin Wang, Dahua Lin, Yu Qiao, and Ziwei Liu. Vbench: Comprehensive benchmark suite for video generative models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21807–21818, June 2024.

[9] Dian Zheng, Ziqi Huang, Hongbo Liu, Kai Zou, Yinan He, Fan Zhang, Lulu Gu, Yuanhan Zhang, Jingwen He, Wei-Shi Zheng, Yu Qiao, and Ziwei Liu. Vbench-2.0: Advancing video generation benchmark suite for intrinsic faithfulness, 2025. URL https://arxiv.org/abs/2503.21755.

[10] Kaleb Newman, Tyler Zhu, and Olga Russakovsky. Video models reason early: Exploiting plan commitment for maze solving, 2026. URL https://arxiv.org/abs/2603.30043.

[11] Ruisi Wang, Zhongang Cai, Fanyi Pu, Junxiang Xu, Wanqi Yin, Maijunxian Wang, Ran Ji, Chenyang Gu, Bo Li, Ziqi Huang, Hokin Deng, Dahua Lin, Ziwei Liu, and Lei Yang. Demystifying video reasoning. In Proceedings of the European Conference on Computer Vision, 2026. URL https://arxiv.org/abs/ 2603.16870.

[12] Woojung Han, Seil Kang, Youngjun Jun, Min-Hung Chen, Fu-En Yang, and Seong Jae Hwang. Physics in 2-steps: Locking motion priors before visual refinement erases them. In Proceedings of the 43rd International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2606.06361.

[13] Nanye Ma, Shangyuan Tong, Haolin Jia, Hexiang Hu, Yu-Chuan Su, Mingda Zhang, Xuan Yang, Yandong Li, Tommi Jaakkola, Xuhui Jia, and Saining Xie. Inference-time scaling for diffusion models beyond scaling denoising steps, 2025. URL https://arxiv.org/abs/2501.09732.

[14] William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pages 4195–4205, October 2023.

[15] Team Wan, Ang Wang, Baole Ai, Bin Wen, Chaojie Mao, Chen-Wei Xie, Di Chen, Feiwu Yu, Haiming Zhao, Jianxiao Yang, Jianyuan Zeng, Jiayu Wang, Jingfeng Zhang, Jingren Zhou, Jinkai Wang, Jixuan Chen, Kai Zhu, Kang Zhao, Keyu Yan, Lianghua Huang, Mengyang Feng, Ningyi Zhang, Pandeng Li, Pingyu Wu, Ruihang Chu, Ruili Feng, Shiwei Zhang, Siyang Sun, Tao Fang, Tianxing Wang, Tianyi Gui, Tingyu Weng, Tong Shen, Wei Lin, Wei Wang, Wei Wang, Wenmeng Zhou, Wente Wang, Wenting Shen,

Wenyuan Yu, Xianzhong Shi, Xiaoming Huang, Xin Xu, Yan Kou, Yangyu Lv, Yifei Li, Yijing Liu, Yiming Wang, Yingya Zhang, Yitong Huang, Yong Li, You Wu, Yu Liu, Yulin Pan, Yun Zheng, Yuntao Hong, Yupeng Shi, Yutong Feng, Zeyinzi Jiang, Zhen Han, Zhi-Fan Wu, and Ziyu Liu. Wan: Open and advanced large-scale video generative models, 2025. URL https://arxiv.org/abs/2503.20314.

[16] Juechu Dong, Boyuan Feng, Driss Guessous, Yanbo Liang, and Horace He. Flex attention: A programming model for generating optimized attention kernels. arXiv preprint arXiv:2412.05496, 2(3):4, 2024.

[17] Rupesh K Srivastava, Klaus Greff, and Jürgen Schmidhuber. Training very deep networks. In C. Cortes, N. Lawrence, D. Lee, M. Sugiyama, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc., 2015. URL https://proceedings.neurips.cc/paper\_ files/paper/2015/file/215a71a12769b056c3c32e7299f1c5ed-Paper.pdf.

[18] Raymond Greenlaw, H. James Hoover, and Walter L. Ruzzo. Limits to Parallel Computation: P-Completeness Theory. Oxford University Press, June 1995. ISBN 9780197560518. doi: 10.1093/oso/ 9780195085914.001.0001. URL http://dx.doi.org/10.1093/oso/9780195085914.001.0001.

[19] Edward Fredkin and Tommaso Toffoli. Conservative logic. International Journal of Theoretical Physics, 21(3-4):219–253, April 1982. ISSN 1572-9575. doi: 10.1007/bf01857727. URL http://dx.doi.org/ 10.1007/BF01857727.

[20] William Merrill and Ashish Sabharwal. The parallelism tradeoff: Limitations of log-precision transformers. Transactions of the Association for Computational Linguistics, 11:531–545, 2023. ISSN 2307-387X. doi: 10.1162/tacl\_a\_00562. URL http://dx.doi.org/10.1162/tacl\_a\_00562.

[21] Calvin Luo. Understanding diffusion models: A unified perspective, 2022. URL https://arxiv.org/ abs/2208.11970.

[22] Yuxi Liu, Konpat Preechakul, Kananart Kuwaranancharoen, and Yutong Bai. The serial scaling hypothesis, 2026. URL https://arxiv.org/abs/2507.12549.

[23] Maijunxian Wang, Ruisi Wang, Juyi Lin, Ran Ji, Thaddäus Wiedemer, Qingying Gao, Dezhi Luo, Yaoyao Qian, Lianyu Huang, Zelong Hong, Jiahui Ge, Qianli Ma, Hang He, Yifan Zhou, Lingzi Guo, Lantao Mei, Jiachen Li, Hanwen Xing, Tianqi Zhao, Fengyuan Yu, Weihang Xiao, Yizheng Jiao, Jianheng Hou, Danyang Zhang, Pengcheng Xu, Boyang Zhong, Zehong Zhao, Gaoyun Fang, John Kitaoka, Yile Xu, Hua Xu, Kenton Blacutt, Tin Nguyen, Siyuan Song, Haoran Sun, Shaoyue Wen, Linyang He, Runming Wang, Yanzhi Wang, Mengyue Yang, Ziqiao Ma, Raphaël Millière, Freda Shi, Nuno Vasconcelos, Daniel Khashabi, Alan Yuille, Yilun Du, Ziming Liu, Bo Li, Dahua Lin, Ziwei Liu, Vikash Kumar, Yijiang Li, Lei Yang, Zhongang Cai, and Hokin Deng. A very big video reasoning suite. In Proceedings of the 43rd International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2602.20159.

[24] Yutong Bai, Danny Tran, Amir Bar, Yann LeCun, Trevor Darrell, and Jitendra Malik. Whole-body conditioned egocentric video prediction. In D. Belgrave, C. Zhang, H. Lin, R. Pascanu, P. Koniusz, M. Ghassemi, and N. Chen, editors, Advances in Neural Information Processing Systems, volume 38, pages 164375–164418. Curran Associates, Inc., 2025. URL https://proceedings.neurips.cc/paper\_ files/paper/2025/file/f0552f14388d95b19740dee809f5cad1-Paper-Conference.pdf.

[25] Bingyi Kang, Yang Yue, Rui Lu, Zhijie Lin, Yang Zhao, Kaixin Wang, Gao Huang, and Jiashi Feng. How far is video generation from world model: A physical law perspective. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 28991–29017. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/kang25g.html.

[26] Anton Bakhtin, Laurens van der Maaten, Justin Johnson, Laura Gustafson, and Ross Girshick. Phyre: A new benchmark for physical reasoning. In H. Wallach, H. Larochelle, A. Beygelzimer, F. d'Alché-Buc, E. Fox, and R. Garnett, editors, Advances in Neural Information Processing Systems, volume 32. Curran Associates, Inc., 2019. URL https://proceedings.neurips.cc/paper\_files/paper/2019/ file/4191ef5f6c1576762869ac49281130c9-Paper.pdf.

[27] Jonathan Ho, Tim Salimans, Alexey Gritsenko, William Chan, Mohammad Norouzi, and David Fleet. Video diffusion models. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 8633–8646. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ 39235c56aef13fb05a6adc95eb9d8d66-Paper-Conference.pdf.

[28] Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, Varun Jampani, and Robin Rombach. Stable video diffusion: Scaling latent video diffusion models to large datasets, 2023. URL https://arxiv.org/abs/ 2311.15127.

[29] Omer Bar-Tal, Hila Chefer, Omer Tov, Charles Herrmann, Roni Paiss, Shiran Zada, Ariel Ephrat, Junhwa Hur, Guanghui Liu, Amit Raj, Yuanzhen Li, Michael Rubinstein, Tomer Michaeli, Oliver Wang, Deqing Sun, Tali Dekel, and Inbar Mosseri. Lumiere: A space-time diffusion model for video generation. In SIGGRAPH Asia 2024 Conference Papers, SA ’24, pages 1–11. ACM, December 2024. doi: 10.1145/ 3680528.3687614. URL http://dx.doi.org/10.1145/3680528.3687614.

[30] Wenming Weng, Ruoyu Feng, Yanhui Wang, Qi Dai, Chunyu Wang, Dacheng Yin, Zhiyuan Zhao, Kai Qiu, Jianmin Bao, Yuhui Yuan, Chong Luo, Yueyi Zhang, and Zhiwei Xiong. Art•v: Auto-regressive text-to-video generation with diffusion models. In 2024 IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW), pages 7395–7405. IEEE, June 2024. doi: 10.1109/cvprw63382. 2024.00735. URL http://dx.doi.org/10.1109/CVPRW63382.2024.00735.

[31] Desai Xie, Zhan Xu, Yicong Hong, Hao Tan, Difan Liu, Feng Liu, Arie Kaufman, and Yang Zhou. Progressive autoregressive video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, pages 6376–6386, June 2025.

[32] Tianwei Yin, Qiang Zhang, Richard Zhang, William T. Freeman, Fredo Durand, Eli Shechtman, and Xun Huang. From slow bidirectional to fast autoregressive video diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 22963–22974, June 2025.

[33] Agrim Gupta, Lijun Yu, Kihyuk Sohn, Xiuye Gu, Meera Hahn, Fei-Fei Li, Irfan Essa, Lu Jiang, and José Lezama. Photorealistic Video Generation with Diffusion Models, pages 393–411. Springer Nature Switzerland, November 2024. ISBN 9783031729867. doi: 10.1007/978-3-031-72986-7\_23. URL http://dx.doi.org/10.1007/978-3-031-72986-7\_23.

[34] Haonan Qiu, Menghan Xia, Yong Zhang, Yingqing He, Xintao Wang, Ying Shan, and Ziwei Liu. Freenoise: Tuning-free longer video diffusion via noise rescheduling, 2024. URL https://arxiv.org/abs/2310. 15169.

[35] Hila Chefer, Uriel Singer, Amit Zohar, Yuval Kirstain, Adam Polyak, Yaniv Taigman, Lior Wolf, and Shelly Sheynin. VideoJAM: Joint appearance-motion representations for enhanced motion generation in video models. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 7595–7616. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/chefer25a.html.

[36] Harold Haodong Chen, Haojian Huang, Xianfeng Wu, Yexin Liu, Yajing Bai, Wen-Jie Shu, Harry Yang, and Ser-Nam Lim. Temporal regularization makes your video generator stronger, 2025. URL https://arxiv.org/abs/2503.15417.

[37] Matus Telgarsky. benefits of depth in neural networks. In Vitaly Feldman, Alexander Rakhlin, and Ohad Shamir, editors, 29th Annual Conference on Learning Theory, volume 49 of Proceedings of Machine Learning Research, pages 1517–1539, Columbia University, New York, New York, USA, 23–26 Jun 2016. PMLR. URL https://proceedings.mlr.press/v49/telgarsky16.html.

[38] Lijie Chen, Binghui Peng, and Hongxun Wu. Theoretical limitations of multi-layer transformer, 2024. URL https://arxiv.org/abs/2412.02975.

[39] Brian D.O. Anderson. Reverse-time diffusion equation models. Stochastic Processes and their Applications, 12(3):313–326, May 1982. ISSN 0304-4149. doi: 10.1016/0304-4149(82)90051-5. URL http://dx. doi.org/10.1016/0304-4149(82)90051-5.

[40] Hao Li, Yang Zou, Ying Wang, Orchid Majumder, Yusheng Xie, R. Manmatha, Ashwin Swaminathan, Zhuowen Tu, Stefano Ermon, and Stefano Soatto. On the scalability of diffusion-based text-to-image generation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9400–9409, June 2024.

[41] Nicolas Dufour, Alexei A. Efros, and Patrick Pérez. The fid lottery: Quantifying hidden randomness in generative-model evaluation, 2026. URL https://arxiv.org/abs/2606.20536.

[42] Charles R. Harris, K. Jarrod Millman, Stéfan J. van der Walt, Ralf Gommers, Pauli Virtanen, David Cournapeau, Eric Wieser, Julian Taylor, Sebastian Berg, Nathaniel J. Smith, Robert Kern, Matti Picus, Stephan Hoyer, Marten H. van Kerkwijk, Matthew Brett, Allan Haldane, Jaime Fernández del Río, Mark Wiebe, Pearu Peterson, Pierre Gérard-Marchant, Kevin Sheppard, Tyler Reddy, Warren Weckesser, Hameer Abbasi, Christoph Gohlke, and Travis E. Oliphant. Array programming with numpy. Nature, 585 (7825):357–362, September 2020. ISSN 1476-4687. doi: 10.1038/s41586-020-2649-2. URL http: //dx.doi.org/10.1038/s41586-020-2649-2.

## A Theoretical Details

As before, let w be an object we want to sample, and let u be a conditioning input. w is an element of a sample space, here taken to be $\mathbb { R } ^ { d }$ , where d is some positive integer.

The forward diffusion process is defined by a noise rate function $\beta : [ 0 , \infty )  ( 0 , \infty )$ , with the condition that $\textstyle \int _ { 0 } ^ { \infty } \beta ( t ) d t = \infty$ . We begin by sampling a starting point $w _ { 0 } \sim \rho _ { 0 } ( w | u )$ and then define the forward stochastic process:

$$
w _ {t + d t} = w _ {t} - \underbrace {\frac {1}{2} \beta (t) w _ {t} d t} _ {\text { drift }} + \underbrace {\sqrt {\beta (t)} d W _ {t}} _ {\text { noise }},
$$

for all $t \in [ 0 , \infty )$ , where $W _ { t }$ is a standard Brownian motion in $\mathbb { R } ^ { d }$ . The $W$ stands for $\mathrm { ^ { * } W i e n e r ^ { * } }$ , since this is also called the Wiener process. It can be schematically thought of as $d W _ { t } \sim { \mathcal { N } } ( 0 , d t I _ { d } )$ ).

This is a stochastic differential equation with a closed-form solution:

$$
w _ {t} = \sqrt {1 - \sigma (t) ^ {2}} w _ {0} + \sigma (t) z, \quad z \sim \mathcal {N} (0, I _ {d}),
$$

where $\sigma ( t ) ^ { 2 } = 1 - e ^ { - \int _ { 0 } ^ { t } \beta ( \tau ) d \tau }$ is the variance due to accumulated noise.

In other words, the probability distribution of $w _ { t } .$ , conditional on u, is the convolution of a scaled version of $\rho _ { 0 } ( \cdot | u )$ with a Gaussian function. The probability distributions $\rho ( \cdot | u , t )$ are increasingly blurry versions of the original $\rho _ { 0 } ( \cdot | u )$ , starting at $\rho ( \cdot | u , 0 ) ~ = ~ \rho _ { 0 } ( \cdot | u )$ and ending with $\begin{array} { r } { \operatorname* { l i m } _ { t \to \infty } \rho ( \cdot | u , \infty ) = \rho _ { \mathcal { N } ( 0 , I _ { d } ) } } \end{array}$ , where we lose all traces of the original and obtain the standard Gaussian distribution.

Define the score function

$$
s (w _ {t} | u, t) := \nabla_ {w _ {t}} \log \rho (w _ {t} | u, t).
$$

With the score function, the same process can be expressed in the time-reversed form [39]:

$$
w _ {t - d t} = w _ {t} + \frac {1}{2} \beta (t) w _ {t} d t + \beta (t) \underbrace {s (w _ {t} | u , t)} _ {\text {score function}} d t + \sqrt {\beta (t)} d W _ {t}.
$$

In the NCSN formalism, we have a neural network $s _ { \boldsymbol { \theta } } \approx s _ { \boldsymbol { \vartheta } }$ . We then solve the backward process by numerical stochastic integration:

$$
\hat {w} _ {t - \Delta t} = \hat {w} _ {t} + \frac {1}{2} \beta (t) \hat {w} _ {t} \Delta t + \beta (t) s _ {\theta} (\hat {w} _ {t} | u, t) \Delta t + \sqrt {\beta (t)} \Delta W _ {t}.
$$

The exact score function s has no closed-form solution in general, because $\rho _ { 0 } ( w _ { 0 } | u )$ is too generic to be integrated. However, if $w _ { 0 }$ is a deterministic function of u, then

$$
\begin{array}{c} \log \rho (w _ {t} | u, t) = - \frac {1}{2 \sigma (t) ^ {2}} \left\| w _ {t} - \sqrt {1 - \sigma (t) ^ {2}} w (u) \right\| ^ {2} + \mathrm{Const} \\ s (w _ {t} | u, t) = \frac {\sqrt {1 - \sigma (t) ^ {2}} w _ {0} (u) - w _ {t}}{\sigma (t) ^ {2}}. \end{array}
$$

Thus, we are actually able to use the score function to solve for $w _ { 0 } ( u )$ in one step:

$$
w _ {0} (u) = \frac {\sigma (t) ^ {2} s (w _ {t} | u , t) + w _ {t}}{\sqrt {1 - \sigma (t) ^ {2}}}.
$$

Proof of proposition 4.1.

1. Assign t to be any constant. Any constant will be fine. For example, $t : = 1$ will do.

2. Assign $\hat { y } _ { f _ { 0 } + 1 : f , t }$ to be any constant. Any constant will be fine. For example, this will do:

$$
\hat {y} _ {f _ {0} + 1: f, t} = (0, 0, \dots , 0),
$$

where 0 stands for a blank image.

3. Compute the score

$$
s _ {t} := s (\hat {y} _ {f _ {0} + 1: f} | y _ {0: f _ {0}}, f, t)
$$

using the NCSN backbone.

4. Solve for the future video segment

$$
y _ {f _ {0} + 1: f} := \frac {\sigma (t) ^ {2} s _ {t} + \hat {y} _ {f _ {0} + 1 : f , t}}{\sqrt {1 - \sigma (t) ^ {2}}}.
$$

5. Invert the last segment of the video trajectory to its state:

$$
x _ {f - f _ {0}: f} = F ^ {- 1} (y _ {f - f _ {0}: f}).
$$

6. Output $x _ { f } .$ .

By assumption, the NCSN backbone is a ${ \mathsf { T C } } ^ { 0 }$ network, and every other step also runs in $\mathcal { O } ( 1 )$ serial time. Thus, the algorithm is in ${ \mathsf { T C } } ^ { 0 }$ , and therefore the state prediction problem is also in ${ \mathsf { T C } } ^ { 0 }$ □

## B Numerical Results

Table 1: Results under different video lengths and inference factorizations. Physical accuracy and visual fidelity for bidirectional, blockwise, and autoregressive diffusion models on $n = 5$ hard-sphere dynamics videos spanning a range of lengths $f .$ Lower $\Delta x ^ { \mathrm { ( G T ) } } , \Delta x ^ { \mathrm { ( 1 ) } }$ , and $\Delta x ^ { ( 5 ) }$ indicate higher physical accuracy; higher IoU indicates higher visual fidelity. The results support Observations 1 and 3: bidirectional models degrade with longer chains of dependent events, while temporal serialization improves local physical accuracy.

<table><tr><td>Type</td><td> $d_{model}$ </td><td> $n_{layers}$ </td><td> $n_{params}$ </td><td>TFLOPs</td><td> $\Delta x^{(GT)} \downarrow$ </td><td> $\Delta x^{(1)} \downarrow$ </td><td> $\Delta x^{(5)} \downarrow$ </td><td>IoU ↑</td></tr><tr><td colspan="9">f=25 frames, n=5 balls</td></tr><tr><td>Bidir</td><td>768</td><td>30</td><td>217M</td><td>10.4</td><td>1.441</td><td>0.197</td><td>0.799</td><td>0.703</td></tr><tr><td>Bidir</td><td>1536</td><td>30</td><td>867M</td><td>39.9</td><td>1.265</td><td>0.160</td><td>0.670</td><td>0.726</td></tr><tr><td>AR</td><td>768</td><td>30</td><td>217M</td><td>7.3</td><td>1.179</td><td>0.166</td><td>0.664</td><td>0.714</td></tr><tr><td>AR</td><td>1536</td><td>30</td><td>867M</td><td>28.1</td><td>1.042</td><td>0.148</td><td>0.613</td><td>0.730</td></tr><tr><td colspan="9">f=33 frames, n=5 balls</td></tr><tr><td>Bidir</td><td>768</td><td>30</td><td>217M</td><td>13.8</td><td>2.047</td><td>0.210</td><td>0.873</td><td>0.693</td></tr><tr><td>Bidir</td><td>1536</td><td>30</td><td>867M</td><td>52.0</td><td>1.944</td><td>0.164</td><td>0.718</td><td>0.725</td></tr><tr><td>AR</td><td>768</td><td>30</td><td>217M</td><td>10.3</td><td>1.905</td><td>0.175</td><td>0.712</td><td>0.705</td></tr><tr><td>AR</td><td>1536</td><td>30</td><td>867M</td><td>39.6</td><td>1.645</td><td>0.142</td><td>0.604</td><td>0.735</td></tr><tr><td colspan="9">f=41 frames, n=5 balls</td></tr><tr><td>Bidir</td><td>768</td><td>30</td><td>217M</td><td>17.2</td><td>2.581</td><td>0.227</td><td>0.952</td><td>0.696</td></tr><tr><td>Bidir</td><td>1536</td><td>30</td><td>867M</td><td>64.4</td><td>2.402</td><td>0.200</td><td>0.856</td><td>0.698</td></tr><tr><td>AR</td><td>768</td><td>30</td><td>217M</td><td>13.4</td><td>2.423</td><td>0.167</td><td>0.695</td><td>0.712</td></tr><tr><td>AR</td><td>1536</td><td>30</td><td>867M</td><td>51.3</td><td>2.165</td><td>0.142</td><td>0.606</td><td>0.732</td></tr><tr><td colspan="9">f=49 frames, n=5 balls</td></tr><tr><td>Bidir</td><td>768</td><td>30</td><td>217M</td><td>20.9</td><td>2.923</td><td>0.261</td><td>1.055</td><td>0.691</td></tr><tr><td>Bidir</td><td>1536</td><td>30</td><td>867M</td><td>77.0</td><td>2.868</td><td>0.205</td><td>0.883</td><td>0.704</td></tr><tr><td>Block-3</td><td>768</td><td>30</td><td>217M</td><td>16.9</td><td>2.749</td><td>0.182</td><td>0.762</td><td>0.713</td></tr><tr><td>Block-3</td><td>1536</td><td>30</td><td>867M</td><td>63.7</td><td>2.675</td><td>0.174</td><td>0.716</td><td>0.705</td></tr><tr><td>Block-2</td><td>768</td><td>30</td><td>217M</td><td>16.7</td><td>2.717</td><td>0.169</td><td>0.708</td><td>0.715</td></tr><tr><td>Block-2</td><td>1536</td><td>30</td><td>867M</td><td>63.4</td><td>2.578</td><td>0.145</td><td>0.620</td><td>0.731</td></tr><tr><td>AR</td><td>768</td><td>30</td><td>217M</td><td>16.6</td><td>2.798</td><td>0.178</td><td>0.739</td><td>0.703</td></tr><tr><td>AR</td><td>1536</td><td>30</td><td>867M</td><td>63.1</td><td>2.548</td><td>0.142</td><td>0.607</td><td>0.733</td></tr></table>

Table 2: Single-ball control as a non-serial setting. Physical accuracy and visual fidelity for bidirectional and autoregressive diffusion models on $n = 1$ hard-sphere dynamics videos of different lengths f . Unlike the multi-ball setting, longer single-ball videos do not introduce longer chains of dependent events. The results support Observation 2: the bidirectional–autoregressive gap largely disappears without serial dependence.

<table><tr><td>Type</td><td> $d_{model}$ </td><td> $n_{layers}$ </td><td> $n_{params}$ </td><td>TFLOPs</td><td> $\Delta x^{(GT)} \downarrow n = 1$ </td><td> $\Delta x^{(5)} \downarrow n = 1$ </td><td> $\Delta x^{(5)} \downarrow n = 5$ </td><td>IoU↑ $n = 1$ </td></tr><tr><td colspan="9"> $f = 25$  frames</td></tr><tr><td>Bidir</td><td>768</td><td>30</td><td>217M</td><td>10.4</td><td>0.310</td><td>0.220</td><td>0.799</td><td>0.823</td></tr><tr><td>AR</td><td>768</td><td>30</td><td>217M</td><td>7.3</td><td>0.162</td><td>0.201</td><td>0.664</td><td>0.819</td></tr><tr><td colspan="9"> $f = 49$  frames</td></tr><tr><td>Bidir</td><td>768</td><td>30</td><td>217M</td><td>20.9</td><td>1.417</td><td>0.246</td><td>1.055</td><td>0.810</td></tr><tr><td>AR</td><td>768</td><td>30</td><td>217M</td><td>16.6</td><td>0.471</td><td>0.220</td><td>0.739</td><td>0.815</td></tr></table>

Table 3: More denoising steps do not close the seriality gap. Local physical accuracy for bidirectional diffusion models with $T \in \{ 1 0 , 2 0 , 5 0 , 1 0 0 , 2 0 0 \}$ denoising steps on $n = 5$ hard-sphere dynamics videos of different lengths f. Performance plateaus after a moderate number of steps, while longer videos remain more difficult. The results support Observation 4: denoising depth does not provide scalable serial computation for longer chains of dependent events.

<table><tr><td>Type</td><td> $d_{model}$ </td><td> $n_{layers}$ </td><td>TFLOPs</td><td> $\Delta x^{(5)} \downarrow T = 10$ </td><td> $\Delta x^{(5)} \downarrow T = 20$ </td><td> $\Delta x^{(5)} \downarrow T = 50$ </td><td> $\Delta x^{(5)} \downarrow T = 100$ </td><td> $\Delta x^{(5)} \downarrow T = 200$ </td></tr><tr><td> $f = 25$ </td><td colspan="8">frames,  $n = 5$  balls</td></tr><tr><td>Bidir</td><td>1536</td><td>30</td><td>39.9</td><td>0.690</td><td>0.667</td><td>0.670</td><td>0.679</td><td>0.689</td></tr><tr><td> $f = 33$ </td><td colspan="8">frames,  $n = 5$  balls</td></tr><tr><td>Bidir</td><td>1536</td><td>30</td><td>52.0</td><td>0.746</td><td>0.719</td><td>0.718</td><td>0.724</td><td>0.740</td></tr><tr><td> $f = 41$ </td><td colspan="8">frames,  $n = 5$  balls</td></tr><tr><td>Bidir</td><td>1536</td><td>30</td><td>64.4</td><td>0.946</td><td>0.886</td><td>0.856</td><td>0.847</td><td>0.848</td></tr><tr><td> $f = 49$ </td><td colspan="8">frames,  $n = 5$  balls</td></tr><tr><td>Bidir</td><td>1536</td><td>30</td><td>77.0</td><td>0.991</td><td>0.931</td><td>0.883</td><td>0.889</td><td>0.916</td></tr></table>

Table 4: Depth scaling is more effective than width scaling. Physical accuracy and visual fidelity for bidirectional diffusion models with different widths $d _ { \mathrm { m o d e l } }$ and depths $n _ { \mathrm { l a y e r s } }$ on $n = 5$ hard-sphere dynamics videos. Depth improves physical accuracy more strongly than width up to about 30 layers, beyond which further improvements are unclear. The results support Observation 5: useful serial computation is supplied more effectively by backbone depth than by width alone.

<table><tr><td>Type</td><td> $d_{model}$ </td><td> $n_{layers}$ </td><td> $n_{params}$ </td><td>TFLOPs</td><td> $\Delta x^{(GT)} \downarrow$ </td><td> $\Delta x^{(1)} \downarrow$ </td><td> $\Delta x^{(5)} \downarrow$ </td><td>IoU ↑</td></tr><tr><td colspan="9">f = 49 frames, n = 5 balls</td></tr><tr><td>Bidir</td><td>640</td><td>30</td><td>151M</td><td>14.9</td><td>2.973</td><td>0.307</td><td>1.172</td><td>0.665</td></tr><tr><td>Bidir</td><td>768</td><td>20</td><td>146M</td><td>14.0</td><td>3.038</td><td>0.403</td><td>1.420</td><td>0.640</td></tr><tr><td>Bidir</td><td>768</td><td>30</td><td>217M</td><td>20.9</td><td>2.923</td><td>0.261</td><td>1.055</td><td>0.691</td></tr><tr><td>Bidir</td><td>768</td><td>60</td><td>430M</td><td>41.7</td><td>2.839</td><td>0.233</td><td>0.950</td><td>0.696</td></tr><tr><td>Bidir</td><td>768</td><td>90</td><td>643M</td><td>62.6</td><td>2.788</td><td>0.210</td><td>0.881</td><td>0.700</td></tr><tr><td>Bidir</td><td>1152</td><td>10</td><td>169M</td><td>14.8</td><td>3.076</td><td>0.416</td><td>1.466</td><td>0.653</td></tr><tr><td>Bidir</td><td>1152</td><td>20</td><td>329M</td><td>29.7</td><td>2.941</td><td>0.261</td><td>1.060</td><td>0.690</td></tr><tr><td>Bidir</td><td>1152</td><td>30</td><td>488M</td><td>44.5</td><td>2.822</td><td>0.227</td><td>0.951</td><td>0.694</td></tr><tr><td>Bidir</td><td>1152</td><td>50</td><td>807M</td><td>74.2</td><td>2.739</td><td>0.251</td><td>1.000</td><td>0.676</td></tr><tr><td>Bidir</td><td>1536</td><td>10</td><td>301M</td><td>25.7</td><td>2.939</td><td>0.324</td><td>1.230</td><td>0.680</td></tr><tr><td>Bidir</td><td>1536</td><td>30</td><td>867M</td><td>77.0</td><td>2.868</td><td>0.205</td><td>0.883</td><td>0.704</td></tr></table>

## C Qualitative Results

![](images/4ae12c65117ae4e3def8cfeb18e3ff089b2b8795707ac4ddfaceb488f67b1d70.jpg)

![](images/c5f03afec074f2b175d90c5dd5b675b4e6aecdafa5b77b61b32e70cbd24ea1aa.jpg)

![](images/a80c83cc2abaaa0ba55246b82a0608414efa6cdaa38f31a73b92b9c899dbd008.jpg)

![](images/ba971716e1aad12fcd9648542f3995d23a6e69c61f572b5526cc8efd70ef5263.jpg)

![](images/9bbd41d2debbd6dc905258e04c3da3c29ac144950d225df7d0b498ff4cc71a3a.jpg)  
(a)  
Figure 6: Qualitative failures. We compare bidirectional and autoregressive outputs from models with $d _ { \mathrm { m o d e l } } = 1 5 3 6$ and $n _ { \mathrm { l a y e r s } } = 3 0$ . The left column shows the shared conditioning frames with motion trails; the remaining columns show 8 output frames, with frame indices below. Thick borders highlight frames where failures appear. Bidirectional outputs exhibit state-consistency failures such as tunneling, ball-count violations, and color-identity violations, while the autoregressive counterparts remain more physically consistent.

![](images/c391d3a3f61451be0bac6bea09268d6181f9ad24ef5a2b971cefc9cc200a745c.jpg)  
(b)  
Figure 6: Qualitative failures, continued. Illustrations follow the same format as Figure 6a.

## D Vibrating-Wall Hard-Sphere Dynamics

In Section 4, we formulate our theory for deterministic video prediction: conditioned on a fully observed initial state, the future trajectory is unique. Natural videos, however, often involve stochas ticity, partial observability, or unmodeled perturbations. To test whether the seriality gap persists beyond the deterministic setting, we conduct a preliminary experiment on a stochastic variant of the hard-sphere dynamics dataset.

In this vibrating-wall variant, wall collisions are no longer perfectly deterministic. Whenever a ball bounces off a wall, we perturb its speed by a random amount

$$
s _ {\mathrm{after}} = | s _ {\mathrm{before}} + \delta s |, \qquad \delta s \sim \mathrm{Unif} [ - 6, 6 ],
$$

while keeping the direction of the post-collision velocity unchanged. Thus, even with the same initial state, the future trajectory is no longer unique: different wall vibrations induce different valid rollouts.

The vibrating-wall setting breaks the point-mass assumption used in the deterministic theory, but it does not eliminate all serial structure. Between vibrating-wall collisions, the motion remains deterministic and follows the same local physical laws as in the original hard-sphere system. We therefore evaluate local physical consistency only on deterministic segments. Concretely, we compute the rollout-5 error $\Delta x ^ { ( 5 ) }$ after excluding any frame for which the five-step rollout window contains a ball-to-wall collision. This isolates whether the generated trajectory is locally consistent within intervals where the dynamics are deterministic.

The results in fig. 7 are preliminary, and we do not claim that this simple vibrating-wall perturbation fully captures the ambiguity of natural video. Nevertheless, the same qualitative trend persists: bidirectional denoising remains worse on local physical consistency than inference schemes with more effective serial computation. This suggests that the deterministic theory identifies one clean mechanism behind the seriality gap, but the phenomenon may extend beyond the exact assumptions of the theorem.

![](images/089c07dd1e8e357d81cc9bb56202154497bd67011a97f3ab62a0b7fe9172d31d.jpg)  
Figure 7: Preliminary results on vibrating-wall hard-sphere dynamics. We introduce stochasticity by perturbing the speed of a ball after each wall collision, while leaving the velocity direction unchanged. Evaluation reports rollout-5 error only on deterministic segments, excluding windows that contain a wall collision. Although preliminary, the results show the same qualitative pattern as the deterministic setting, suggesting that the seriality gap is not merely an artifact of globally deterministic rollouts.

Table 5: Model trends are consistent under vibrating-wall dynamics. Physical accuracy and visual quality for bidirectional and autoregressive diffusion models on $n = 5$ vibrating-wall hard-sphere videos of different lengths f .

<table><tr><td>Type</td><td> $d_{model}$ </td><td> $n_{layers}$ </td><td> $n_{params}$ </td><td>TFLOPs</td><td> $\Delta x^{(GT)} \downarrow$ </td><td> $\Delta x^{(1)} \downarrow$ </td><td> $\Delta x^{(5)} \downarrow$ </td><td>IoU↑</td></tr><tr><td colspan="9">f=25 frames, n=5 balls, vibrating-wall</td></tr><tr><td>Bidir</td><td>768</td><td>30</td><td>217M</td><td>10.4</td><td>1.941</td><td>0.312</td><td>1.135</td><td>0.670</td></tr><tr><td>AR</td><td>768</td><td>30</td><td>217M</td><td>7.3</td><td>1.726</td><td>0.197</td><td>0.826</td><td>0.718</td></tr><tr><td colspan="9">f=33 frames, n=5 balls, vibrating-wall</td></tr><tr><td>Bidir</td><td>768</td><td>30</td><td>217M</td><td>13.8</td><td>2.454</td><td>0.354</td><td>1.242</td><td>0.660</td></tr><tr><td>AR</td><td>768</td><td>30</td><td>217M</td><td>10.3</td><td>2.281</td><td>0.177</td><td>0.768</td><td>0.728</td></tr><tr><td colspan="9">f=41 frames, n=5 balls, vibrating-wall</td></tr><tr><td>Bidir</td><td>768</td><td>30</td><td>217M</td><td>17.2</td><td>2.895</td><td>0.397</td><td>1.383</td><td>0.660</td></tr><tr><td>AR</td><td>768</td><td>30</td><td>217M</td><td>13.4</td><td>2.720</td><td>0.177</td><td>0.762</td><td>0.722</td></tr><tr><td colspan="9">f=49 frames, n=5 balls, vibrating-wall</td></tr><tr><td>Bidir</td><td>768</td><td>30</td><td>217M</td><td>20.9</td><td>3.163</td><td>0.421</td><td>1.447</td><td>0.662</td></tr><tr><td>AR</td><td>768</td><td>30</td><td>217M</td><td>16.6</td><td>3.017</td><td>0.167</td><td>0.727</td><td>0.726</td></tr></table>

## E Ruling Out Alternative Explanations

We test whether the observed advantage of autoregressive generation could be explained by optimization or initialization rather than inference factorization. Specifically, we vary training batch size, repeat training across independent seeds, and fine-tune models initialized from a pretrained Wan checkpoint. In each case, the qualitative autoregressive–bidirectional gap persists.

## E.1 Batch Size

Diffusion training can be sensitive to batch size, with prior scaling experiments reporting improved generation metrics at larger batch sizes [40]. Because computational constraints limit our main experiments to a batch size $b = 6 4$ , we repeat the autoregressive–bidirectional comparison with $b = 2 5 6$ . We keep all other settings fixed, including the learning rate of $2 \times 1 0 ^ { - 4 }$ , which we do not retune for the larger batch. The results are shown in Figure 8. Although the $b = 2 5 6$ models have higher absolute error, potentially because the learning rate was not retuned, the seriality gap persists at both batch sizes.

![](images/b8958991c4c02a5d5b91fbb87869366fc6558bfc01960eb65154d0cc6898adf9.jpg)  
Figure 8: Batch-size ablation. Rollout-5 error as a function of video length f for bidirectional and autoregressive models trained with $b \in \{ 6 4 , 2 5 6 \}$ . The $b = 2 5 6$ models have slightly higher error, possibly because the learning rate was not retuned. At both batch sizes, bidirectional error grows more rapidly with f than autoregressive error, preserving the seriality gap.

![](images/bf6c3e7a4852b8a9cd3504a061247d7fcdecdc7eeaaf2120b1be6e63e3bd79c2.jpg)

![](images/e2285b5547d187b837c5d05f1c450ba0ceb73c74b7526f9ed476f3077818e19a.jpg)

![](images/78f46c408e77b94c36767ceb9ed1551c658eb838e45be28d9e8fe04f48b2961b.jpg)  
Figure 9: Robustness to training-seed variation. Mean global trajectory error $\Delta x ^ { \mathrm { ( G T ) } }$ , rollout-5 error $\Delta x ^ { ( 5 ) }$ , and IoU for autoregressive (AR) and bidirectional (Bidir) models across three independent training runs. Error bars and shaded regions denote 1 standard deviation. AR has lower errors and higher IoU, with non-overlapping intervals.

GT Error ( <sub>#</sub> )  
![](images/3ff288781dd547792ecff92746dbedd72b80577d50b3f64ed24104972d270221.jpg)

![](images/bd6248cd633d425dff00e6a566a0f74f0a2bd771611e694bd1e8972d250b9694.jpg)

![](images/3eb64744877a16b4ebb43dbad0c9172e93416f863d6458332ee83bb8c8376478.jpg)  
Figure 10: Wan initialization preserves the model ordering but lowers absolute performance. Global trajectory error $\bar { \Delta } x ^ { \mathrm { ( G T ) } }$ , rollout-5 error $\Delta x ^ { ( 5 ) }$ , and IoU for models fine-tuned from Wan2.1-T2V-1.3B. Circles show the fine-tuned models, and dashed lines show the corresponding models trained from scratch. AR retains lower errors and higher IoU than Bidir, but both fine-tuned models underperform their trained-from-scratch counterparts, possibly because of a resolution mismatch and the out-of-distribution hard-sphere task.

## E.2 Training-Seed Variation

Independent diffusion training runs can yield different evaluation outcomes due to parameter initialization, minibatch ordering, and sampled training noise [41]. We therefore train each model with three independent seeds at $d _ { \mathrm { m o d e l } } = 7 6 8$ and $n _ { \mathrm { l a y e r s } } = 3 0 .$ , using $f = 4 9$ frames and n = 5 balls, and evaluate each run on 1,024 examples. The results are shown in Figure 9. Across all metrics, the error bars are non-overlapping and well separated, indicating that the autoregressive advantage is robust to training-seed variation rather than the result of a favorable seed lottery.

## E.3 Fine-Tuning from a Pretrained Checkpoint

To test whether pretraining changes the model ordering, we compare our models trained from scratch with variants initialized from the Wan-AI/Wan2.1-T2V-1.3B checkpoint [15] and fine-tuned on the hard-sphere dataset. We use the same training configuration, changing only the learning rate from $2 \times 1 0 ^ { - 4 } t o 2 \times 1 0 ^ { - 5 }$ for fine-tuning. The results are shown in Figure 10.

## F Human Evaluation of Perceptual Plausibility

To validate whether our rollout-k error metric reflects perceptual plausibility, we conduct a lightweight human preference study. Evaluators used a blind A/B interface, available online<sup>3</sup>. The interface displays two videos side by side and asks which one looks more physically plausible, with instructions to focus on motion and artifacts rather than sharpness. We compiled judgments from 21 volunteer evaluators. No compensation was provided, and the interface did not request personal or demographic information. Each evaluator rated 7 pairs for each of three comparison groups: AR vs. Bidir, AR vs. Block-3, and Block-3 vs. Bidir, for 21 pairwise judgments per evaluator and 441 total judgments. Video pairs were sampled from examples where the rollout metric indicated a large separation between the two models.

![](images/b9d1e3247e425c87a1151ffc594014c49c867aca53f04faa8229554209f02049.jpg)

![](images/4c03d8f3b1841da1037d6c201ea7f6361d15d6e0a69a8186ab0a5f6fe03033fd.jpg)  
Figure 11: Human preference study. Left: aggregate preference rates by model show that AR and Block-3 generations are preferred substantially more often than bidirectional generations. Right: pairwise preferences show strong human preference for AR over Bidir and Block-3 over Bidir, while AR is only moderately preferred over Block-3. This matches the rollout-error ranking, suggesting that large differences in $\Delta x ^ { ( 5 ) }$ correspond to perceptually salient differences in physical plausibility.

The human judgments align with the rollout metric. When $\Delta x ^ { ( 5 ) }$ indicates a large separation, that separation is also perceptible to human evaluators: AR and Block-3 are both strongly preferred over bidirectional generation, whereas the perceptual gap between AR and Block-3 is much smaller. This supports the use of rollout error as a proxy for physically salient generation failures.

## G Data Synthesis

Videos are synthesized with a custom NumPy [42] event-driven simulator of frictionless balls in a closed square box. For ball–ball interactions, the simulator computes the next collision time analytically. For a pair of balls $( i , j )$ with relative displacement $\Delta x = x _ { j } - x _ { i }$ , relative velocity $\Delta { } v = v _ { j } - v _ { i }$ , and radii $r _ { i } , r _ { j }$ , the collision time t is the smallest nonnegative root of

$$
\| \Delta x + t \Delta v \| _ {2} ^ {2} = (r _ {i} + r _ {j}) ^ {2},
$$

which yields a quadratic equation in t. Once t is found, the collision positions follow directly from $x _ { i } ( t ) = x _ { i } + v _ { i } t$ and $x _ { j } ( t ) = x _ { j } + v _ { j } t$

For each rollout, we sample initial ball positions uniformly at random such that no two balls overlap at the first frame, then sample each ball’s initial direction uniformly from $[ 0 , 2 \pi )$ and speed uniformly from [8, 10]. We then simulate a longer trajectory and extract fixed-length clips of length f frames with a sliding window, retaining windows with an overlap of at most $\bar { \boldsymbol { f } } / 2$ to avoid highly similar neighboring clips and increase diversity. Each dataset configuration fixes the number of balls, frame length, box size, ball radius, and rendering resolution.

In the reported experiments, we use $n \in \{ 1 , 5 \}$ balls, video lengths $f \in \{ 2 5 , 3 3 , 4 1 , 4 9 \}$ , ball radius 0.7, world size $1 0 \times 1 0$ , rendering resolution $1 2 8 \times 1 2 8 .$ , and 5 conditioning frames. Training sets contain 20,000 videos. For evaluation, we generate 1,024 base videos of $5 7$ frames and derive the shorter sets by retaining only the first f frames of each video. This keeps the conditioning frames, and therefore the initial conditions, matched across frame lengths for the same sample, making cross-length comparisons fairer. To avoid trivial trajectories, we retain only windows whose post-conditioning ball–ball collision density,

## number of ball–ball collisions

## active frames number of balls

exceeds 0.035. For $n = 5 .$ , this corresponds to at least 5 collisions when $f = 3 3$ and at least 8 when $f = 4 9$ . For the base 57-frame evaluation set, we also try to spread samples more evenly over ball–ball collision counts, although the resulting distribution is not perfectly uniform; the shorter trimmed evaluation sets inherit this coverage. Training and evaluation use disjoint random seeds but the same simulator and parameter settings, so evaluation remains in-distribution.

For evaluation, we recover ball positions from generated frames using color-based tracking. Each ball is rendered with a distinct RGB color; for each frame and ball, we form a binary mask of pixels whose RGB values lie within a fixed tolerance of that ball’s color and compute the mask centroid via image moments. The resulting coordinates are used to compute $\Delta x ^ { \mathrm { ( G T ) } }$ and $\Delta x ^ { ( k ) }$

The rollout metric $\Delta x ^ { ( k ) }$ additionally requires per-ball velocities to seed the simulator. We estimate the velocity at frame n by finite-differencing the recovered centroids, $\hat { v } _ { n } = \hat { x } _ { n } - \hat { x } _ { n - 1 }$ . When a collision occurs within the interval $( n - 1 , n )$ , this finite difference straddles the collision and no longer reflects the instantaneous post-collision velocity; in that case, we instead estimate the precollision velocity from the preceding interval, $\hat { x } _ { n - 1 } - \hat { x } _ { n - 2 }$ , and set $\hat { v } _ { n }$ to the analytic post-collision velocity implied by the hard-sphere collision rule.

## H Attention Masks

![](images/7ff9f1adf2b7195689647cf44347a584873733459c3ef3337240a0c4deeb7732.jpg)  
Figure 12: Attention masks for blockwise generation. The masks are illustrated for the $f = 2 5$ setting. Rows denote query positions and columns denote key/value positions; colored cells indicate permitted attention entries and white cells are masked. The top row shows inference-time masks over condition and output frames, while the bottom row shows the corresponding training-time masks over clean and noisy token copies. Columns compare autoregressive Block-1 masking, intermediate block-causal masks, and bidirectional masking. Thick lines mark condition/output and clean/noisy boundaries; dashed lines show individual frame or token cells.