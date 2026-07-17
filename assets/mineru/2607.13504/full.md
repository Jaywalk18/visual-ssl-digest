# DP-BOA: Dirichlet-Process Birth-or-Assign for On-the-Fly Category Discovery

Peiyan Gu<sup>1</sup> , Zixin Teng<sup>1</sup> , and Xuming He<sup>1,2⋆</sup>

ShanghaiTech University, Shanghai, China

2 Shanghai Engineering Research Center of Intelligent Vision and Imaging peiyangu@outlook.com, {tengzx2025,hexm}@shanghaitech.edu.cn

Abstract. On-the-fly category discovery requires deciding for each incoming test sample whether to assign it to an existing category or spawn a new one. Existing methods typically implement this decision through matching-based heuristics, such as radius- or hash-based rules. While effective in practice, these methods usually treat category birth implicitly as a fallback when no existing category matches confidently, rather than as an explicit alternative supported by its own statistical evidence. To address this, we propose DP-BOA, a posterior-predictive decision framework based on an online Dirichlet-process Gaussian mixture model with a Normal–Inverse–Wishart prior. During training, we use labeled data to calibrate a shared NIW prior over category Gaussians and warm-start the known-category posteriors. At test time, for each incoming sample, DP-BOA compares the posterior predictive evidence for assignment to existing categories against the evidence for spawning a new category induced by the DP prior, and then updates category statistics online after the decision. The method captures anisotropic category geometry and naturally adapts decision confidence as evidence accumulates. Across standard OCD benchmarks, DP-BOA consistently outperforms strong baselines and delivers particularly strong novel-class discovery performance while maintaining competitive known-class accuracy. The project page is available at DP-BOA.

Keywords: On-the-Fly Category Discovery· Generalized Category Discovery· Bayesian Nonparametrics

## 1 Introduction

Deep learning has achieved remarkable progress in visual recognition [10,20,26], but most models still assume a closed world, i.e., all test categories are seen during training. This assumption rarely holds in dynamic real-world environments. Open-world settings such as Novel Category Discovery (NCD) [18] and Generalized Category Discovery (GCD) [46] partially address this issue by using labeled known classes to organize and discover novel ones in an unlabeled set.

However, they typically assume ofline access to the full unlabeled dataset, which is impractical in many applications (e.g., autonomous driving, robotics, live content moderation) where data arrive continuously and decisions must be made immediately. To address this gap, On-the-fly Category Discovery (OCD) [11] considers a strict streaming setting: samples arrive one by one, and the system must instantly decide whether each sample belongs to an existing category or should start a new one. This is particularly challenging because early mistakes can propagate and degrade later decisions.

A key challenge in OCD is the assign-versus-birth decision. It involves two coupled questions: how to define “new”, and how to represent “old” under a stream. For the birth, the system needs a reliable criterion for deciding when a sample should start a new category, and labeled known categories are the main available source of prior guidance for this decision. For the assignment, category representations should evolve with the stream: they should be updated with newly assigned samples, remain uncertain when evidence is scarce, and become more confident as support accumulates. These suggest that an efective OCD method should combine an explicit birth criterion with online category representations whose uncertainty adapts to the amount of observed evidence.

Most existing OCD methods make online decisions via distance- or similaritybased matching rules [3,11,29,60]: a sample is assigned to the nearest prototype if suficiently close, otherwise it spawns a new one. While this design is simple and eficient, it treats category birth implicitly as a fallback from failed assignment, rather than as an explicit alternative supported by its own evidence. It also brings two limitations. First, such matching rules often induce prototypecentered regions of fixed shape that ignore direction-dependent covariance (e.g., Hamming balls [11, 60], Euclidean balls [29], or angular caps under cosine similarity [3]), which can be misaligned with heteroscedastic and anisotropic feature distributions (see Appendix M for empirical geometry statistics). Second, threshold control in these methods is typically fixed or heuristic and does not explicitly model how predictive uncertainty should contract as category evidence accumulates. As a result, it becomes harder to distinguish newly formed categories from stable ones, increasing the risk of error propagation after early false births.

To address these limitations, we propose DP-BOA, a probabilistic OCD framework that formulates assign-versus-birth as a Bayesian evidence comparison. For each incoming test sample, DP-BOA compares the prior-weighted predictive evidence for assignment to existing categories against that for spawning a new category, so that birth is no longer treated merely as rejection by existing categories but is explicitly evaluated within the same decision framework. We instantiate this idea with an online Dirichlet Process Gaussian Mixture Model (DP-GMM) [38, 49] and a Normal–Inverse–Wishart (NIW) prior [4]. Statistics from labeled known classes are used to initialize known-category posteriors and to estimate a shared prior over category Gaussians. Full-covariance category modeling captures anisotropic geometry, posterior-predictive updates allow predictive uncertainty to contract as evidence accumulates, and the DP prior enables open-ended category growth.

We validate these design choices through streaming experiments on standard OCD benchmarks, consistently outperforming baselines. Ablations further show that support-set prior calibration, the size- and concentration-dependent DP category prior, full-covariance modeling, and evidence-adaptive posterior updates each contribute meaningfully to the overall gains.

In summary, our contributions are:

We reformulate OCD as a Bayesian evidence comparison between assigning a sample to an existing category and spawning a new one, making birth an explicit alternative rather than an implicit fallback from failed assignment. We instantiate this formulation with an online DP-GMM and an NIW prior, yielding an online posterior-predictive algorithm that leverages known-class statistics for prior initialization, captures anisotropic category geometry, and adapts predictive uncertainty as evidence accumulates.

– We demonstrate strong performance on standard OCD benchmarks, particularly for novel classes.

## 2 Related Work

## 2.1 Generalized / Novel Category Discovery

Novel Class Discovery (NCD) [18] aims to cluster novel classes in an unlabeled set by leveraging a labeled known-class set. Early work focused on pairwise similarity prediction for clustering [21, 22], while subsequent methods improved similarity estimation [17, 59], feature learning [30, 52, 61, 62], and clustering objectives [14,56,57]. Generalized Category Discovery (GCD) [46] extends NCD to unlabeled data containing both known and novel classes. Early GCD methods established strong pool-based baselines with contrastive representation learning and semi-supervised clustering [46, 47, 54, 58]. More recent work has explored several distinct directions, including clustering-oriented representation learning via mean-shift updates [7], eficient adaptation with spatial prompt tuning [51], robustness under domain shifts [50], debiased parametric learning with distribution guidance [31], and hierarchy-aware geometry [32]. Beyond centralized visual GCD, recent studies have also extended the problem to decentralized and multimodal settings. Fed-GCD [37] studies GCD under federated learning, where data are distributed across clients with heterogeneous label spaces, while languageor multimodal-assisted methods exploit vision-language models, LLM feedback, or multimodal alignment to improve category discovery [2, 34, 42]. Unlike these ofline, pool-based methods, we target a single-pass streaming setting.

## 2.2 On-the-Fly Category Discovery

On-the-Fly Category Discovery (OCD) studies a single-pass test stream in which each incoming sample must be immediately assigned to an existing category or trigger a new one [11]. SMILE [11] introduces the task and uses instance-level hash codes with a Hamming-radius rule for online birth-or-assign decisions. PHE [60] improves this line with prototype-guided hash encoding to better preserve category structure in Hamming space. More recent variants incorporate auxiliary priors: DifGRE [29] leverages synthesized novel samples through a generate– refine–encode pipeline, while Sync [3] augments OCD with language-assisted representations and lightweight active querying. In contrast, we focus on the standard OCD setting without such auxiliary priors.

## 2.3 Bayesian Non-parametrics for Mixtures

Our approach builds on Dirichlet-process (DP) mixture models. Blackwell and MacQueen [5] showed that marginalizing a DP yields the Chinese Restaurant Process, whose concentration parameter governs new-component creation, and Sethuraman’s stick-breaking construction provides a constructive representation [41]. Rasmussen’s Infinite Gaussian Mixture Model [38] popularized DP-GMMs with Normal–Inverse–Wishart (NIW) priors and closed-form Student-t predictive densities under conjugacy. Subsequent work studied more scalable or streaming inference for non-parametric mixtures and clustering [9, 40, 49], while adjacent non-parametric deep clustering and open-world recognition methods explored related ideas in learned embedding spaces [39, 55]. Unlike these methods, which primarily address unsupervised density modeling or clustering, OCD requires single-pass online decisions with known class support-set supervision.

## 3 Method

## 3.1 Problem Formulation and Preliminaries

We follow the On-the-Fly Category Discovery (OCD) setting [11]. Let $\mathcal { D } _ { S } =$ $\{ ( { \bf x } _ { i } , y _ { i } ) \} _ { i = 1 } ^ { M } \subset \mathcal { X } \times \mathcal { Y } _ { S }$ denote the labeled support set used for training, where $y _ { S }$ is the set of known classes. For evaluation, the test stream is $\mathcal { D } _ { Q } = \{ ( \mathbf { x } _ { j } ^ { q } , y _ { j } ^ { q } ) \} _ { j = 1 } ^ { N } \subset$ $\mathcal { X } \times \mathcal { Y } _ { Q }$ , whose label space satisfies $\mathcal { V } _ { Q } = \mathcal { V } _ { S } \cup \mathcal { V } _ { N }$ with $\mathcal { V } _ { S } \cap \mathcal { V } _ { N } \doteq \check { \mathcal { O } }$ , where $\mathcal { D } _ { N }$ denotes the novel classes. Following the standard OCD protocol [11, 60], only $\mathcal { D } _ { S }$ is available during training; at test time, query samples are revealed sequentially, and their ground-truth labels are used only for evaluation. For each arriving sample $\mathbf { x } _ { t }$ , the model must decide whether to assign it to one of the existing categories or to declare the birth of a new one.

Feature Space. Our method operates in a feature space learned from the labeled support set. Specifically, a feature encoder $f _ { \theta }$ maps each sample $\mathbf { x } _ { t }$ to a d-dimensional representation $\mathbf z _ { t } = f _ { \theta } ( \mathbf x _ { t } )$ . We train $f _ { \theta }$ ofline on the labeled support set $D _ { S }$ using the standard supervised cross-entropy loss, and freeze it during online inference. We deliberately adopt this simple and generic feature-learning setup to keep the focus of the paper on the birth-or-assign decision mechanism, rather than on task-specific representation objectives. All probabilistic decisions are then performed in this fixed feature space. Details are provided in Sec. 4.1.

## 3.2 A Probabilistic Framework for OCD

We cast the core assign-versus-birth decision in OCD as a Bayesian evidence comparison. At time $t ,$ let $K _ { t - 1 }$ be the number of existing categories after processing the first t−1 query samples, and let $\mathcal { D } _ { t - 1 }$ denote the current online state.

![](images/4608714930e99e203bebc5097add723e1b07be9a1cb7fb2204d2f636e3eb75a7.jpg)  
Fig. 1: Overview of our proposed DP-BOA framework. Our method consists of two phases. (Top) Ofline Initialization: We first train a feature encoder $f _ { \theta }$ on the labeled set $\mathcal { D } _ { S }$ . We then use the extracted features to compute the suficient statistics $\left( n _ { k } , \bar { \mathbf { z } } _ { k } , \mathbf { S } _ { k } \right)$ for all $K _ { S }$ known classes. These statistics are used to initialize our DP-GMM model (i.e., calibrate the global NIW hyperparameters from support-set statistics (Sec. 3.4)). (Bottom) Online Inference: At test time, a new sample x<sub>t</sub> is passed through the frozen $f _ { \theta }$ . Our model performs a posterior-predictive birth-or-assign decision $\left( \mathrm { E q . ~ } \left( 1 2 \right) \right)$ , comparing the posterior probability of assigning z<sub>t</sub> to an existing category $( { \cal P } ( c _ { t } = k | . . . ) )$ versus birthing a new category $( P ( c _ { t } = \mathrm { n e w } | . . . ) )$ . Based on this decision, the statistics of the corresponding category are updated online.

For exposition, we write this state as a history of past observations, although in implementation it is represented only through per-category suficient statistics. For an arriving feature vector $\mathbf { z } _ { t }$ , we choose a decision ${ \mathfrak { c } } _ { t } \in \{ 1 , \dots , K _ { t - 1 } , \operatorname { n e w } \}$ by maximizing the posterior:

$$
\hat {c} _ {t} = \arg \max _ {k \in \{1, \dots , K _ {t - 1}, \text { new } \}} P (c _ {t} = k \mid \mathbf {z} _ {t}, \mathcal {D} _ {t - 1}).\tag{1}
$$

By Bayes’ rule, $P ( c _ { t } = k \mid \mathbf { z } _ { t } , \mathcal { D } _ { t - 1 } ) \propto P ( c _ { t } = k \mid \mathcal { D } _ { t - 1 } ) p ( \mathbf { z } _ { t } \mid c _ { t } = k , \mathcal { D } _ { t - 1 } )$ To obtain a tractable online rule, we make two practical modeling approximations. For an existing category $k ,$ once $c _ { t } = k$ is assumed, the predictive density depends only on that category’s own history, giving $p ( \mathbf { z } _ { t } \mid c _ { t } = k , \mathcal { D } _ { t - 1 } ) = p ( \mathbf { z } _ { t } \mid$ $\mathcal { D } _ { t - 1 } ^ { k } )$ . For a new category, we use the prior predictive induced by a shared global prior estimated ofline, so $p ( \mathbf { z } _ { t } \mid c _ { t } = \operatorname { n e w } , \mathcal { D } _ { t - 1 } ) = p ( \mathbf { z } _ { t } \mid c _ { t } = \operatorname { n e w } )$ . This yields

$$
\begin{array}{r l} & {\hat {c} _ {t} = \arg \max \Big \{\max _ {k \in \{1, \ldots , K _ {t - 1} \}} \big [ P (c _ {t} = k \mid \mathcal {D} _ {t - 1}) p (\mathbf {z} _ {t} \mid \mathcal {D} _ {t - 1} ^ {k}) \big ],} \\ & {\qquad \big [ P (c _ {t} = \mathrm{new} \mid \mathcal {D} _ {t - 1}) p (\mathbf {z} _ {t} \mid c _ {t} = \mathrm{new}) \big ] \Big \}.} \end{array}\tag{2}
$$

The remaining task is to specify the two probabilistic components in $\operatorname { E q . } \ ( 2 )$ : the category priors $\begin{array} { r l } { P ( c _ { t } } & { { } | \ D _ { t - 1 } ) } \end{array}$ and the predictive densities $p ( \mathbf { z } _ { t } \mid \mathbf { \theta } \cdot \mathbf { \alpha } )$ . The following section details our specifications for each.

## 3.3 Probabilistic Modeling via Adapted DP-GMM

As shown in Fig. 1, we instantiate the framework in Eq. (2) with a Dirichlet– Process Gaussian Mixture Model (DP-GMM) [38] and a Normal–Inverse–Wishart (NIW) prior. In contrast to standard DP-GMM usage, OCD provides a labeled support set $\mathcal { D } _ { S }$ and requires single-pass online birth-or-assign decisions without revisiting past samples. We therefore adapt DP-GMM to this setting by using $\mathcal { D } _ { S }$ to estimate a shared prior and initialize known categories. We briefly introduce DP-GMM in our method below.

Predictive Densities (Gaussian + NIW). To define the continuous density terms in Eq. (2) and model anisotropic category geometry, we represent each category k by a full-covariance Gaussian $\mathcal { N } ( \textbf { z } | ~ \mu _ { k } , \Sigma _ { k } )$ . We adopt a Bayesian formulation and place a conjugate Normal–Inverse–Wishart (NIW) prior over the Gaussian parameters $( \mu , \Sigma )$ , parameterized by global hyperparameters $\theta _ { 0 } \ =$ $( \mu _ { 0 } , \Psi _ { 0 } , \nu _ { 0 } , \kappa _ { 0 } )$ . Here, $\pmb { \mu } _ { 0 }$ is the prior mean, $\Psi _ { 0 }$ is the scale matrix for covariance, $\nu _ { 0 }$ is the degrees-of-freedom parameter, and $\kappa _ { 0 }$ controls the strength of the prior on the mean. A key property of this conjugate prior is that, after observing data, the posterior over $( \mu , \Sigma )$ remains NIW. For an existing category k with data $\mathcal { D } _ { t - 1 } ^ { k }$ , the posterior NIW hyperparameters are $\pmb { \theta } _ { k } = ( \pmb { \mu } _ { k } , \pmb { \Psi } _ { k } , \pmb { \nu } _ { k } , \kappa _ { k } )$ . These can be computed in closed form from the prior $\pmb { \theta } _ { 0 }$ and the suficient statistics of category k: the count $n _ { k } = | \mathcal { D } _ { t - 1 } ^ { k } |$ , the sample mean $\bar { \mathbf { z } } _ { k }$ , and the scatter matrix $\begin{array} { r } { { \bf S } _ { k } = \sum _ { { \bf z } \in \mathcal { D } _ { + - 1 } ^ { k } } ( { \bf z } - \bar { \bf z } _ { k } ) ( { \bf z } - \bar { \bf z } _ { k } ) ^ { \top } } \end{array}$ . The exact NIW update equations are given in Appendix C. Integrating out $( \mu , \Sigma )$ yields closed-form multivariate Student-t predictive densities $t _ { d } \colon$

$$
p (\mathbf {z} _ {t} \mid \mathrm{new}) = t _ {d} \Bigl (\mathbf {z} _ {t} \left| \mu_ {0}, \frac {\kappa_ {0} + 1}{\kappa_ {0} (\nu_ {0} - d + 1)} \Psi_ {0}, \nu_ {0} - d + 1\right) \Bigr)\tag{3}
$$

$$
p (\mathbf {z} _ {t} \mid \mathcal {D} _ {t - 1} ^ {k}) = t _ {d} \Bigl (\mathbf {z} _ {t} \Big | \boldsymbol {\mu} _ {k}, \frac {\kappa_ {k} + 1}{\kappa_ {k} (\nu_ {k} - d + 1)} \boldsymbol {\Psi} _ {k}, \nu_ {k} - d + 1 \Bigr)\tag{4}
$$

Because the posterior hyperparameters are updated with accumulated category evidence, the predictive density for category k becomes progressively sharper as $n _ { k }$ increases. This allows category uncertainty to contract naturally with support, rather than being handled only indirectly through heuristic matching rules. Appendix D provides the derivation and an illustration of how the predictive density evolves as evidence accumulates.

Category Priors (Dirichlet Process). To define the discrete prior probability terms in Eq. (2), we assume a Dirichlet Process (DP) prior [38], which nonparametrically allows for dynamically expandable category growth.

A standard result from the marginalization of the DP (the Blackwell Mac-Queen Polya Urn scheme [5, 13]) provides the exact prior probabilities for the decision $c _ { t }$ (see Appendix E for details):

$$
P (c _ {t} = k | \mathcal {D} _ {t - 1}) = \frac {n _ {k}}{\alpha + N _ {t - 1}}\tag{5}
$$

$$
P (c _ {t} = \mathrm{new} | \mathcal {D} _ {t - 1}) = \frac {\alpha}{\alpha + N _ {t - 1}}\tag{6}
$$

where $n _ { k }$ is the number of samples in category k, $\begin{array} { r } { N _ { t - 1 } = \sum _ { k } n _ { k } } \end{array}$ is the total number of samples processed so far, and $\alpha > 0$ is the $\mathrm { D P s }$ concentration parameter, acting as a principled “innovation rate” for new categories.

## 3.4 Initialization from Known Categories

Unlike standard DP-GMM formulations, which are fully unsupervised, OCD provides labels for a subset of categories through the support set $\mathcal { D } _ { S }$ . We therefore use $\mathcal { D } _ { S }$ to calibrate an EB-style NIW prior and to initialize posterior NIW states for known categories. This initialization plays two roles: (1) it uses $\mathcal { D } _ { S }$ to calibrate a robust global prior $\pmb { \theta } _ { 0 }$ (for the “Birth”), and (2) it initializes the known categories with their posterior NIW statistics (for the “Assign”).

Estimating global priors. Our global prior is defined by $\pmb { \theta } _ { 0 } = ( \pmb { \mu } _ { 0 } , \pmb { \Psi } _ { 0 } , \pmb { \nu } _ { 0 } , \kappa _ { 0 } )$ and the DP concentration α. We use the $K _ { S } = | \mathcal { D } _ { S } |$ known classes in $\mathcal { D } _ { S }$ to estimate them. For each known category $k ,$ , let $\mathcal { D } _ { k }$ denote the set of support-set features belonging to that category. First, we define the necessary base statistics:

$$
\bar {\mathbf {z}} _ {k} = \frac {1}{n _ {k}} \sum_ {\mathbf {z} \in \mathcal {D} _ {k}} \mathbf {z} \quad (\mathrm{categorymean})
$$

$$
\mathbf {S} _ {k} = \sum_ {\mathbf {z} \in \mathcal {D} _ {k}} (\mathbf {z} - \bar {\mathbf {z}} _ {k}) (\mathbf {z} - \bar {\mathbf {z}} _ {k}) ^ {\top} \quad (\text { category   scatter })
$$

$$
\bar {\mathbf {z}} = \frac {1}{M} \sum_ {k = 1} ^ {K _ {S}} n _ {k} \bar {\mathbf {z}} _ {k} \quad (\mathrm{globalmean})
$$

$$
\mathbf {\Sigma} _ {\mathrm{within}} = \frac {1}{M - K _ {S}} \sum_ {k = 1} ^ {K _ {S}} \mathbf {S} _ {k} \quad (\mathrm{pooledwithin-categorycovariance})
$$

$$
\pmb {\Sigma} _ {\mathrm{means}} = \frac {1}{K _ {S} - 1} \sum_ {k = 1} ^ {K _ {S}} (\bar {\mathbf {z}} _ {k} - \bar {\mathbf {z}}) (\bar {\mathbf {z}} _ {k} - \bar {\mathbf {z}}) ^ {\top} \quad (\text {covariance of category means})
$$

$$
\overline {{n ^ {- 1}}} = \frac {1}{K _ {S}} \sum_ {k = 1} ^ {K _ {S}} \frac {1}{n _ {k}} \quad (\text { average   inverse   category   size }).
$$

We use these statistics to estimate the hyperparameters via an empirical-Bayes method [33](see Appendix F for derivations):

– Prior mean $( \mu _ { 0 } ) { : }$ : We set the prior mean to the global mean:

$$
\boldsymbol {\mu} _ {0} = \bar {\mathbf {z}}\tag{7}
$$

– Covariance scale $\left( \Psi _ { 0 } \right)$ : We choose the expected value of the prior covariance <sup>E</sup>[Σ] to match the pooled within-category covariance:

$$
\mathbb {E} [ \boldsymbol {\Sigma} ] = \boldsymbol {\Sigma} _ {\text { within }} \quad \Rightarrow \quad \boldsymbol {\Psi} _ {0} = (\nu_ {0} - d - 1)   \boldsymbol {\Sigma} _ {\text { within }}\tag{8}
$$

– Mean strength $\left( \kappa _ { 0 } \right)$ : We estimate $\kappa _ { 0 }$ by matching the trace of the observed covariance of category means $\pmb { \Sigma } _ { \mathrm { m e a n s } }$ to its theoretical expectation under the NIW prior:

$$
\pmb {\Sigma} _ {\mathrm{means}} \approx \pmb {\Sigma} _ {\mathrm{within}} \left(\overline {{n ^ {- 1}}} + \frac {1}{\kappa_ {0}}\right)\tag{9}
$$

We use the trace approximation to solve this equation.

– Tunable priors $( \nu _ { 0 } , \alpha )$ : While our EB-style calibration sets most NIW parameters $\left( \mu _ { 0 } , \kappa _ { 0 } , \Psi _ { 0 } \right)$ from $\mathcal { D } _ { S }$ , two scalars must still be specified: the NIW degrees of freedom $\nu _ { 0 }$ and the $\mathrm { D P }$ concentration α. We first focus on $\nu _ { 0 }$ , which we reparameterize as $n _ { 0 } = \nu _ { 0 } - d - 1$ for interpretability. This $n _ { 0 }$ primarily controls the prior predictive density $p ( \mathbf { z } \mid \mathrm { n e w } )$ for the “Birth” hypothesis. Given $\Psi _ { 0 } = n _ { 0 } \sum _ { \mathrm { w i t h i n } }$ , the prior-predictive scale $\mathbf { V } _ { 0 }$ in $\operatorname { E q } .$ . (3) simplifies to:

$$
\mathbf {V} _ {0} = \frac {\kappa_ {0} + 1}{\kappa_ {0} (\nu_ {0} - d + 1)} \mathbf {\Psi} _ {0} = \frac {\kappa_ {0} + 1}{\kappa_ {0}} \cdot \frac {n _ {0}}{n _ {0} + 2} \mathbf {\Sigma} _ {\mathrm{within}}\tag{10}
$$

and the predictive degrees of freedom in Eq. (3) become $\nu _ { 0 } ^ { \prime } = \nu _ { 0 } - d + 1 =$ $n _ { 0 } + 2$ . Thus, $n _ { 0 }$ jointly tunes the “peakiness” (via $\mathbf { V } _ { 0 } )$ and “tail thickness” (via $\nu _ { 0 } ^ { \prime } )$ of the evidence required to birth a new category. Since a single fixed constant $( \mathrm { e . g . } , n _ { 0 } = 1 0 )$ fails to generalize across datasets with diferent scales (see Tab. 4), we propose a data-driven heuristic:

$$
n _ {0} = \min \bigl (\frac {1}{2} \bar {n}, n _ {\mathrm{cap}} \bigr), \qquad \bar {n} = \frac {1}{K _ {S}} \sum_ {k = 1} ^ {K _ {S}} n _ {k},\tag{11}
$$

where $n _ { \mathrm { c a p } }$ is a fixed hyperparameter that limits the maximum prior strength. This rule is guided by two principles: (1) tying $n _ { 0 }$ to n¯ interprets it as a pseudo-sample size, calibrating the prior’s strength to the scale of the observed data; and (2) the cap at $n _ { \mathrm { c a p } }$ prevents an over-confident prior on datasets with large n¯. This is particularly important in OCD, where an overly strong prior derived from $\mathcal { D } _ { S }$ could suppress the discovery of novel categories with diferent covariance structures. Empirically, Eq. (11) yields consistently strong performance and obviates per-dataset tuning.

For the DP concentration α, we find that model performance is highly robust to its value. We therefore set a single constant across all datasets; detailed sensitivity analyses of $\nu _ { 0 }$ and α are provided in Sec. 4.3.

Initializing known-category posteriors. Finally, we initialize the first $K _ { S } = | \mathcal { D } _ { S } |$ known categories. Instead of starting them as empty components, we use their suficient statistics $\left( n _ { k } , \bar { \mathbf { z } } _ { k } , \mathbf { S } _ { k } \right)$ computed from $\mathcal { D } _ { S }$ to derive their posterior NIW parameters $\left( \mu _ { k } , \Psi _ { k } , \nu _ { k } , \kappa _ { k } \right)$ using the update rules in Appendix C. These $K _ { S }$ initialized categories form the initial set for the online assignment process.

This “warm start” ensures not only that the known “Assign” categories are modeled with high confidence from the outset, but also that $\mathrm { t h e } \tilde { \cdot } \mathrm { { s i r t h } } ^ { \bar { \cdot } }$ hypothesis is governed by a meaningful, data-driven standard (the estimated $\pmb { \theta } _ { 0 } )$ .

## 3.5 Online Inference and Update

With the $K _ { S }$ known categories and the global prior initialized, the model begins processing the query stream $\mathcal { D } _ { Q }$ . For each arriving sample $\mathbf { z } _ { t }$ , which could belong to either an existing category or a novel one, the model must perform the twostage process: (1) Inference and (2) Update.

Decision Rule. First, the model executes the Bayesian evidence comparison $\left( \operatorname { E q . } \left( 2 \right) \right)$ to determine if $\mathbf { z } _ { t }$ belongs to an existing category (either one of the $K _ { S }$ known ones or a previously discovered novel one) or represents a new, unseen category. We substitute the components from Sec. 3.3, yielding the decision $\hat { c } _ { t } \colon$

$$
\begin{array}{l} \arg \max \Big \{\big [ \alpha \cdot t _ {d} (\mathbf {z} _ {t} | \boldsymbol {\mu} _ {0}, \frac {\kappa_ {0} + 1}{\kappa_ {0} (\nu_ {0} - d + 1)} \boldsymbol {\Psi} _ {0}, \nu_ {0} - d + 1) \big ], \\ \underset {k \in \{1 \ldots K _ {t - 1} \}} {\max} n _ {k} t _ {d} (\mathbf {z} _ {t} | \boldsymbol {\mu} _ {k}, \frac {\kappa_ {k} + 1}{\kappa_ {k} (\nu_ {k} - d + 1)} \boldsymbol {\Psi} _ {k}, \nu_ {k} - d + 1) \Big ] \Big \} \end{array}\tag{12}
$$

Online Update. Second, based on the decision $\hat { c } _ { t } ^ { \phantom { \dagger } } ,$ the model updates its state online. This update step is critical for the OCD setting, as it allows the model to learn from and refine its understanding of novel categories as they emerge. Due to the conjugacy of the NIW prior, we only need to update the relevant statistics for the chosen category.

If Assign $\left( \hat { c } _ { t } = k \right)$ : We update the suficient statistics $\left( n _ { k } , \bar { \mathbf { z } } _ { k } , \mathbf { S } _ { k } \right)$ of the chosen category k using standard online update rules [53] (see Appendix G):

$$
n _ {k} ^ {\mathrm{new}} = n _ {k} ^ {\mathrm{old}} + 1\tag{13}
$$

$$
\bar {\mathbf {z}} _ {k} ^ {\mathrm{new}} = \bar {\mathbf {z}} _ {k} ^ {\mathrm{old}} + \frac {1}{n _ {k} ^ {\mathrm{new}}} (\mathbf {z} _ {t} - \bar {\mathbf {z}} _ {k} ^ {\mathrm{old}})\tag{14}
$$

$$
\mathbf {S} _ {k} ^ {\mathrm{new}} = \mathbf {S} _ {k} ^ {\mathrm{old}} + (\mathbf {z} _ {t} - \bar {\mathbf {z}} _ {k} ^ {\mathrm{old}}) (\mathbf {z} _ {t} - \bar {\mathbf {z}} _ {k} ^ {\mathrm{new}}) ^ {\top}\tag{15}
$$

If Birth $\left( \hat { c } _ { t } = \mathbf { n e w } \right)$ : We increment the category count $K _ { t } = K _ { t - 1 } + 1$ , and initialize this new category $k ^ { \prime } = K _ { t }$ with its own suficient statistics:

$$
n _ {k ^ {\prime}} = 1, \quad \bar {\mathbf {z}} _ {k ^ {\prime}} = \mathbf {z} _ {t}, \quad \mathbf {S} _ {k ^ {\prime}} = \mathbf {0}\tag{16}
$$

These updated statistics $( n _ { \hat { c } _ { t } } , \bar { \mathbf { z } } _ { \hat { c } _ { t } } , \mathbf { S } _ { \hat { c } _ { t } } )$ are then used to compute the category’s posterior parameters $( \mu _ { \hat { c } _ { t } } , \Psi _ { \hat { c } _ { t } } , \nu _ { \hat { c } _ { t } } , \kappa _ { \hat { c } _ { t } } )$ via the NIW update rules in $\mathrm { A p \mathrm { - } }$ pendix C, which in turn define the category’s predictive density $p ( \mathbf { z } _ { t + 1 } | \mathcal { D } _ { \hat { c } _ { t } } )$ in subsequent decisions, thereby fulfilling the “evidence-adaptive” requirement.

Complexity. Since DP-BOA replaces lightweight Hamming/radius checks with full-covariance Student-t predictive densities and DP-style updates, its online head incurs additional overhead. Let d be the feature dimension and K the number of active categories. Because the backbone cost is the same as in prior OCD methods, we focus on the online head. For each incoming feature, evaluating full-covariance Student-t predictive densities for all existing categories plus the “new” costs $O ( K d ^ { 2 } )$ time, and updating the suficient statistics and NIW posterior for the selected category has worst-case cost $O ( d ^ { 3 } )$ ). Thus, DP-BOA has per-sample complexity $O ( \bar { d ^ { 3 } } + K d ^ { 2 } )$ time with $O ( K d ^ { 2 } )$ memory. In practice, we cache covariance inverses and log-determinants, so only the afected category is recomputed after each update. To improve scalability, we further consider DP-BOA-L, a lightweight variant that replaces full covariance with a rank-r approximation $( r \ll d )$ maintained by Frequent Directions [28]:

$$
\Sigma_ {k} \approx \sigma_ {k} ^ {2} I + U _ {k} \mathrm{diag} (\lambda_ {k}) U _ {k} ^ {\top}, \quad U _ {k} \in \mathbb {R} ^ {d \times r}, r \ll d.\tag{17}
$$

Here, the columns of $U _ { k }$ are the r approximate dominant covariance directions, $\lambda _ { k }$ contains the corresponding approximate eigenvalues, and $\sigma _ { k } ^ { 2 }$ is the scalar residual variance used to summarize the covariance outside the low-rank subspace. With fixed small $r ,$ its per-sample complexity becomes $O ( K d r + d r ^ { 2 } )$ time with O(Kdr) memory, reducing the dependence on representation dimensionality from quadratic to linear. This preserves the same probabilistic birth-or-assign principle while bringing DP-BOA-L to the same linear order in both representation dimensionality and the number of categories as recent OCD heads [3,29,60]. DP-BOA-L thus serves as a scalability-oriented approximation to full DP-BOA: it preserves the same online birth-or-assign decision rule, while the low-rank covariance representation substantially reduces latency and memory at the cost of only a small accuracy drop. Appendix K provides the implementation and detailed complexity analysis of DP-BOA-L.

## 4 Experiments

## 4.1 Setup

Datasets. Following [11, 60], we conduct experiments on multiple datasets, including two generic datasets—CIFAR100 [25] and ImageNet100 [8]—as well as eight fine-grained datasets: CUB [48], Stanford Cars [24], Herbarium19 [43], Oxford-IIIT Pet [35], and four super-categories from iNaturalist [45], including Fungi, Arachnida, Animalia, and Mollusca. Following OCD [11], the categories of each dataset are split into subsets of known and novel classes. Specifically, 50% of the samples from the known classes are used to form the labeled set $\mathcal { D } _ { S }$ for training, while the remainder forms the query stream $\mathcal { D } _ { Q }$ for on-the-fly testing. The details of the datasets are provided in Appendix A.

Evaluation Protocol. We follow the protocol [46,60] and evaluate using clustering accuracy on unlabeled datasets. This metric is calculated by finding an optimal one-to-one mapping $\mathcal { P }$ between predicted cluster indices $\hat { y } _ { i }$ and ground-truth labels $y _ { i }$ with the Hungarian algorithm [27]: $\begin{array} { r } { \mathrm { A C C } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } 1 \left\{ y _ { i } = \mathcal { P } \left( \hat { y } _ { i } \right) \right\} } \end{array}$ . We report model performance separately for known, novel, and all classes.

Implementation Details. We implement our method in PyTorch and run all experiments on NVIDIA TITAN RTX GPUs. Following OCD [11,29,60], we use a DINO-pretrained ViT-B/16 backbone [6] and fine-tune only the last transformer block for 100 epochs. We train with SGD [1] (momentum 0.9) using a cosine learning-rate schedule from 1.0 down to $1 0 ^ { - 4 }$ and a batch size of 256. Unlike prior OCD methods [3,11,29,60], we do not rely on sophisticated representation learning. We simply train the encoder with standard supervised cross-entropy on the labeled support set and keep it frozen during online inference, so that any performance gain mainly comes from our probabilistic birth-or-assign head. After training, we fix the backbone as the encoder $f _ { \theta }$ with feature dimension $d = 7 6 8$ , and set $\alpha = 1 0 ^ { - 9 }$ and $n _ { \mathrm { c a p } } = 5 0$ . All results are averaged over 3 runs with diferent seeds.

Table 1: Comparison with State-of-the-Art methods on the first five datasets (Part I). We report accuracy for All, Known, and Novel classes. Bold indicates the best performance. Rows highlighted in gray (marked with <sup>†</sup>) use external knowledge.

<table><tr><td rowspan="2">Method</td><td colspan="3">Animalia</td><td colspan="3">Arachnida</td><td colspan="3">CIFAR100</td><td colspan="3">CUB</td><td colspan="3">Fungi</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>SLC [19]</td><td>32.4</td><td>61.9</td><td>19.3</td><td>25.4</td><td>44.6</td><td>11.4</td><td>44.4</td><td>59.0</td><td>15.1</td><td>28.6</td><td>44.0</td><td>20.9</td><td>27.7</td><td>60.0</td><td>13.4</td></tr><tr><td>RankStat [17]</td><td>31.4</td><td>54.9</td><td>21.6</td><td>26.6</td><td>51.0</td><td>10.0</td><td>35.0</td><td>44.0</td><td>17.0</td><td>21.2</td><td>26.9</td><td>18.4</td><td>23.8</td><td>50.5</td><td>12.0</td></tr><tr><td>WTA [23]</td><td>33.4</td><td>59.8</td><td>22.4</td><td>28.1</td><td>55.5</td><td>10.9</td><td>40.8</td><td>52.9</td><td>16.7</td><td>21.9</td><td>26.9</td><td>19.4</td><td>27.5</td><td>65.5</td><td>12.0</td></tr><tr><td>SMILE [11]</td><td>35.9</td><td>49.4</td><td>30.3</td><td>29.9</td><td>57.9</td><td>12.2</td><td>51.6</td><td>61.5</td><td>31.7</td><td>32.2</td><td>50.9</td><td>22.9</td><td>29.3</td><td>64.6</td><td>13.6</td></tr><tr><td>PHE [60]</td><td>40.3</td><td>55.7</td><td>31.8</td><td>37.0</td><td>75.7</td><td>12.6</td><td>57.4</td><td>72.1</td><td>27.9</td><td>36.4</td><td>55.8</td><td>27.0</td><td>31.4</td><td>67.9</td><td>15.2</td></tr><tr><td> $\text{DiffGRE}^{\dagger }$  [29]</td><td>43.5</td><td>63.2</td><td>35.3</td><td>47.7</td><td>76.6</td><td>29.4</td><td>-</td><td>-</td><td>-</td><td>42.5</td><td>54.4</td><td>36.5</td><td>-</td><td>-</td><td>-</td></tr><tr><td> $\text{Sync}^{\dagger }$  [3]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>56.1</td><td>68.4</td><td>31.5</td><td>45.3</td><td>54.3</td><td>40.9</td><td>-</td><td>-</td><td>-</td></tr><tr><td>DP-BOA</td><td>50.7</td><td>67.6</td><td>43.7</td><td>50.6</td><td>53.4</td><td>49.4</td><td>60.7</td><td>75.0</td><td>32.0</td><td>53.4</td><td>57.2</td><td>51.6</td><td>51.8</td><td>65.0</td><td>46.0</td></tr></table>

Table 2: Comparison with SOTA methods on the remaining five datasets (Part II).

<table><tr><td rowspan="2">Method</td><td colspan="3">Herbarium19</td><td colspan="3">ImageNet100</td><td colspan="3">Mollusca</td><td colspan="3">OxfordPets</td><td colspan="3">StanfordCars</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>SLC [19]</td><td>14.9</td><td>27.4</td><td>8.1</td><td>32.9</td><td>86.5</td><td>5.2</td><td>31.1</td><td>59.8</td><td>15.0</td><td>35.5</td><td>41.3</td><td>33.1</td><td>14.0</td><td>23.0</td><td>9.7</td></tr><tr><td>RankStat [17]</td><td>13.8</td><td>20.6</td><td>10.2</td><td>31.1</td><td>73.3</td><td>9.8</td><td>29.3</td><td>55.2</td><td>15.5</td><td>33.2</td><td>42.3</td><td>28.4</td><td>14.8</td><td>19.9</td><td>12.3</td></tr><tr><td>WTA [23]</td><td>14.6</td><td>21.2</td><td>11.1</td><td>30.8</td><td>72.9</td><td>19.4</td><td>30.3</td><td>55.4</td><td>17.0</td><td>35.2</td><td>46.3</td><td>29.3</td><td>17.1</td><td>24.4</td><td>13.6</td></tr><tr><td>SMILE [11]</td><td>22.9</td><td>39.3</td><td>14.1</td><td>33.8</td><td>74.2</td><td>13.4</td><td>33.3</td><td>44.5</td><td>27.2</td><td>41.2</td><td>42.1</td><td>40.7</td><td>26.2</td><td>46.6</td><td>16.2</td></tr><tr><td>PHE [60]</td><td>22.6</td><td>40.5</td><td>12.9</td><td>34.0</td><td>80.2</td><td>10.9</td><td>39.9</td><td>65.0</td><td>26.5</td><td>48.3</td><td>53.8</td><td>45.4</td><td>31.3</td><td>61.9</td><td>16.8</td></tr><tr><td> $\text{DiffGRE}^{\dagger }$  [29]</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>42.6</td><td>62.0</td><td>32.3</td><td>49.6</td><td>50.1</td><td>49.3</td><td>27.7</td><td>48.1</td><td>17.8</td></tr><tr><td> $\text{Sync}^{\dagger }$  [3]</td><td>-</td><td>-</td><td>-</td><td>44.0</td><td>86.2</td><td>22.8</td><td>-</td><td>-</td><td>-</td><td>61.6</td><td>69.5</td><td>57.5</td><td>24.6</td><td>34.8</td><td>19.5</td></tr><tr><td>DP-BOA</td><td>30.2</td><td>46.4</td><td>21.5</td><td>33.8</td><td>75.8</td><td>12.7</td><td>48.9</td><td>53.9</td><td>46.3</td><td>59.0</td><td>63.6</td><td>56.6</td><td>32.4</td><td>58.8</td><td>19.7</td></tr></table>

## 4.2 Comparison with the State of the Art

Compared Methods. In Tabs. 1 and 2, we compare DP-BOA against representative methods on ten benchmarks: three from adjacent settings that are standard OCD baselines (SLC [19], RankStat [17], WTA [23]) and four OCD-specific methods (SMILE [11], PHE [60], DifGRE<sup>†</sup> [29], Sync<sup>†</sup> [3]). Methods marked with <sup>†</sup> use external knowledge (e.g., difusion models or CLIP+text), making the comparison conservative for DP-BOA. For DifGRE, we report its strongest variant —PHE + DifGRE with online clustering inference—which achieves the highest reported average. “–” denotes metrics not reported in the original works.

Results. Without using any external knowledge, DP-BOA attains the best All score on 8/10 datasets. It is particularly strong on novel classes: on CUB, DP-BOA reaches 51.6% Novel accuracy, a +10.7 point gain over the next-best method (Sync), and on Mollusca it achieves 46.3% vs 32.3% for DifGRE (+14.0). This indicates that our probabilistic framework is highly efective for the core “on-the-fly discovery” task. Sync obtains higher All accuracy on ImageNet100 and OxfordPets, but relies on CLIP and text. Among methods that do not use external knowledge, DP-BOA (33.8%) is competitive with PHE (34.0%) on ImageNet100 and clearly outperforms PHE on OxfordPets (59.0% vs 48.3%). Overall, DP-BOA sets a new state-of-the-art among OCD methods that operate purely on the visual stream, ofering stronger and more balanced novel-category discovery without external knowledge bases.

Table 3: Ablation on probabilistic design choices in DP-BOA. Each row disables one component while keeping others fixed: w/o µ<sub>0</sub>: zero prior mean (no data–driven centering); w/o $\varPsi _ { 0 } \colon$ identity prior covariance (no scale matching); w/o $\kappa _ { 0 } \colon$ set $\kappa _ { 0 } { = } 1$ (no EB estimation of $\kappa _ { \left. \right)} $ ; w/o $D P$ prior: remove the DP category prior terms and score hypotheses using predictive densities only; $w / o$ adapt: freeze NIW posteriors at test time (no evidence-adaptive updating); Spherical: use an isotropic covariance instead of a full matrix. Best in bold.

<table><tr><td rowspan="2">Method</td><td colspan="3">Animalia</td><td colspan="3">CUB</td><td colspan="3">OxfordPets</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>w/o  $\mu_0$ </td><td>50.1</td><td>65.4</td><td>43.7</td><td>52.9</td><td>51.1</td><td>53.8</td><td>58.1</td><td>63.0</td><td>55.5</td></tr><tr><td>w/o  $\Psi_0$ </td><td>47.9</td><td>70.6</td><td>38.4</td><td>40.8</td><td>39.8</td><td>41.3</td><td>56.0</td><td>52.0</td><td>58.2</td></tr><tr><td>w/o  $\kappa_0$ </td><td>45.2</td><td>59.1</td><td>39.4</td><td>52.2</td><td>51.7</td><td>52.4</td><td>57.5</td><td>60.1</td><td>56.2</td></tr><tr><td>w/o DP prior</td><td>48.0</td><td>71.3</td><td>38.4</td><td>52.3</td><td>57.3</td><td>49.8</td><td>57.1</td><td>62.7</td><td>54.2</td></tr><tr><td>w/o adapt</td><td>40.2</td><td>63.7</td><td>30.5</td><td>35.9</td><td>53.6</td><td>27.0</td><td>54.7</td><td>55.8</td><td>54.2</td></tr><tr><td>Spherical</td><td>41.7</td><td>61.7</td><td>33.5</td><td>40.4</td><td>49.0</td><td>36.1</td><td>47.1</td><td>45.5</td><td>47.9</td></tr><tr><td>DP-BOA</td><td>50.7</td><td>67.6</td><td>43.7</td><td>53.4</td><td>57.2</td><td>51.6</td><td>59.0</td><td>63.6</td><td>56.6</td></tr></table>

## 4.3 Ablation and Analysis

Ablation on probabilistic design choices. We ablate our probabilistic design by disabling one component at a time while keeping all others fixed (Tab. 3).

Prior initialization. The first three rows remove individual support-set calibration components of the birth prior. Overall, the full calibrated initialization (DP-BOA) attains the best All accuracy on all three datasets. Removing the data-driven prior mean $\mu _ { 0 }$ has only a mild efect, but discarding the learned covariance scale $\varPsi _ { 0 }$ or mean strength $\kappa _ { 0 }$ is clearly harmful. In particular, w/o $\varPsi _ { 0 }$ produces the largest drops (e.g., CUB All 53.4→40.8), and $w / o \ \kappa _ { 0 }$ consistently reduces All by 5.5/1.2/1.5 points, indicating that EB scale matching and mean shrinkage are key to a well-calibrated prior.

Dirichlet–process prior. Removing the DP prior in Eqs. (5) and (6) (w/o DP prior ) and scoring categories purely by predictive density—treating all existing clusters as equally likely a priori—degrades performance on all datasets. Novel accuracy drops from 43.7 to 38.4 on Animalia and from 56.6 to 54.2 on OxfordPets, suggesting that the DP size- and concentration-dependent terms help stabilize the birth rate and avoid over-fragmenting categories.

Evidence-adaptive updates. Freezing NIW posteriors at test time (w/o adapt) is particularly damaging for novel-class discovery. Novel accuracy drops from 43.7→30.5 on Animalia and 51.6→27.0 on CUB, while Known accuracy remains relatively high, showing that continuously updating cluster posteriors with incoming evidence is crucial.

Covariance geometry. Finally, we compare our full-covariance model with the Spherical variant that enforces isotropy for both prior and posteriors. Concretely, we (i) match the prior scale to the full model by setting $\bar { \psi } _ { 0 } ^ { \mathrm { s p h } } = ( \nu _ { 0 } -$ $d - 1 ) \bar { \sigma } ^ { 2 } \mathbf { I }$ , where $\begin{array} { r } { \bar { \sigma } ^ { 2 } = \frac { 1 } { d } \mathrm { t r } ( \sum _ { \mathrm { w i t h i n } } ) } \end{array}$ , and (ii) after each NIW update, project the posterior to spherical by $\begin{array} { r } { \varPsi _ { k } ^ { \mathrm { s p h } } \gets \big ( \frac { 1 } { d } \mathrm { t r } ( \varPsi _ { k } ) \big ) \mathbf { I } . } \end{array}$ Even under this calibrated construction, Spherical lags far behind Full: All drops by 9.0/13.0/11.9 points and Novel by $1 0 . 2 / 1 5 . 5 / 8 . 7$ on Animalia/CUB/OxfordPets. This highlights that modeling anisotropic category shapes is essential for accurate posterior predictives and reliable birth-or-assign decisions.

Table 4: Sensitivity to $n _ { 0 }$ . Bold = best per column, underlined = second-best. Row $\bar { n } / 2 ^ { * }$ is our default rule $n _ { 0 } = \operatorname* { m i n } ( \bar { n } / 2 , n _ { \mathrm { c a p } } )$

<table><tr><td rowspan="2"> $n_0$ </td><td colspan="3">Animalia ( $\bar{n}/2=19$ )</td><td colspan="3">CUB ( $\bar{n}/2=7$ )</td><td colspan="3">OxfordPets ( $\bar{n}/2=24$ )</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>5</td><td>40.7</td><td>60.7</td><td>32.5</td><td>55.4</td><td>60.0</td><td>53.2</td><td>58.5</td><td>62.0</td><td>56.6</td></tr><tr><td>10</td><td>46.5</td><td>63.3</td><td>39.5</td><td>50.7</td><td>52.3</td><td>49.9</td><td>57.8</td><td>55.6</td><td>58.9</td></tr><tr><td>15</td><td>48.0</td><td>69.7</td><td>39.0</td><td>47.6</td><td>47.4</td><td>46.1</td><td>58.4</td><td>55.4</td><td>60.0</td></tr><tr><td>20</td><td>49.9</td><td>66.6</td><td>42.9</td><td>46.2</td><td>43.0</td><td>48.8</td><td>58.7</td><td>61.2</td><td>57.5</td></tr><tr><td>25</td><td>51.4</td><td>63.5</td><td>46.4</td><td>42.6</td><td>41.7</td><td>43.0</td><td>60.8</td><td>60.9</td><td>61.2</td></tr><tr><td>50</td><td>46.7</td><td>58.9</td><td>41.6</td><td>39.9</td><td>38.3</td><td>40.7</td><td>54.4</td><td>63.2</td><td>49.8</td></tr><tr><td> $\bar{n}/2^*$ </td><td>50.7</td><td>67.6</td><td>43.7</td><td>53.4</td><td>57.2</td><td>51.6</td><td>59.0</td><td>63.6</td><td>56.6</td></tr></table>

Table 5: Sensitivity to α. Bold = best per column, underlined = second-best. The row marked <sup>∗</sup> is our default.

<table><tr><td rowspan="2">log α</td><td colspan="3">Animalia</td><td colspan="3">CUB</td><td colspan="3">OxfordPets</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>-6</td><td>48.8</td><td>65.4</td><td>41.9</td><td>52.4</td><td>53.8</td><td>51.7</td><td>61.0</td><td>60.9</td><td>61.1</td></tr><tr><td>-7</td><td>49.5</td><td>67.1</td><td>42.3</td><td>53.0</td><td>53.3</td><td>52.9</td><td>61.4</td><td>62.9</td><td>60.6</td></tr><tr><td>-8</td><td>50.2</td><td>68.5</td><td>42.7</td><td>52.8</td><td>55.0</td><td>51.6</td><td>59.7</td><td>63.2</td><td>57.9</td></tr><tr><td>-9*</td><td>50.7</td><td>67.6</td><td>43.7</td><td>53.4</td><td>57.2</td><td>51.6</td><td>59.0</td><td>63.6</td><td>56.6</td></tr><tr><td>-10</td><td>51.2</td><td>66.8</td><td>44.7</td><td>52.9</td><td>55.1</td><td>51.7</td><td>58.9</td><td>64.8</td><td>56.2</td></tr><tr><td>-11</td><td>49.0</td><td>64.0</td><td>42.9</td><td>53.2</td><td>56.8</td><td>51.4</td><td>59.0</td><td>65.0</td><td>55.9</td></tr><tr><td>-12</td><td>50.6</td><td>66.7</td><td>43.9</td><td>53.5</td><td>56.1</td><td>52.2</td><td>58.3</td><td>65.1</td><td>54.7</td></tr></table>

Sensitivity to $n _ { 0 }$ and α. The reparameterization $n _ { 0 } = \nu _ { 0 } - d - 1$ controls the tail heaviness of the Student-t prior predictive, while the DP concentration α sets the prior odds of birthing a new category. Very small $n _ { 0 }$ makes the birth predictive difuse, while large $n _ { 0 }$ concentrates it around the global prior; similarly, too small α suppresses novel-class creation, whereas too large α can over-fragment the stream. In Tab. 4, the data-scaled rule $n _ { 0 } = \operatorname* { m i n } ( \bar { n } / 2 , n _ { \mathrm { c a p } } )$ remains close to the best All accuracy, within 0.7, 2.0, and 1.8 points on Animalia, CUB, and OxfordPets. Table 5 shows similar stability for α: the default log $\alpha = - 9$ is within 0.5, 0.1, and 2.4 All points of the best value on the same datasets. Although more sophisticated dataset- or stream-adaptive schedules may exist, we adopt these rules for their simplicity, stability, and near-optimal performance.

Cluster Growth and Estimated Number of Classes. We analyze the evolution of the number of discovered clusters along the OCD stream. Fig. 2 plots $K ( t )$ on CUB, showing that DP-BOA exhibits smooth, controlled growth that broadly tracks the ground-truth trend while remaining slightly conservative. Consistently, Tab. 6 reports $K _ { \mathrm { f i n a l } } { = } 1 7 5 / 1 7 7$ on CUB/StanfordCars versus 200/196 ground-truth classes, substantially closer than other baselines, indicating a more calibrated birth rate without severe stream fragmentation.

![](images/69b2a500bc0b2a6b823037c9c4add3c716a58851a6c19c87f9087e4af53c7908.jpg)  
Fig. 2: Cluster growth on CUB.

Table 6: Final cluster count.

<table><tr><td>Method</td><td>CUB</td><td>StanfordCars</td></tr><tr><td>SMILE-16bit</td><td>924</td><td>896</td></tr><tr><td>PHE-16bit</td><td>318</td><td>709</td></tr><tr><td>DiffGRE</td><td>116</td><td>171</td></tr><tr><td>Sync-AL</td><td>255</td><td>447</td></tr><tr><td>DP-BOA</td><td>175</td><td>177</td></tr><tr><td>Ground Truth</td><td>200</td><td>196</td></tr></table>

Table 7: Runtime and accuracy–eficiency trade-of. Accuracy, maximum persample latency (Lat.) in milliseconds, and head memory (Mem.) in MB. Numbers in parentheses denote the number of categories.

<table><tr><td rowspan="2">Method</td><td colspan="5">Animalia (77)</td><td colspan="5">CUB (200)</td><td colspan="5">StanfordCars (196)</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>Lat.</td><td>Mem.</td><td>All</td><td>Known</td><td>Novel</td><td>Lat.</td><td>Mem.</td><td>All</td><td>Known</td><td>Novel</td><td>Lat.</td><td>Mem.</td></tr><tr><td>DP-BOA</td><td>50.7</td><td>67.6</td><td>43.7</td><td>46.1</td><td>299.2</td><td>53.4</td><td>57.2</td><td>51.6</td><td>81.7</td><td>619.2</td><td>32.4</td><td>58.8</td><td>19.7</td><td>82.9</td><td>626.7</td></tr><tr><td>DP-BOA-L</td><td>49.2</td><td>64.2</td><td>42.9</td><td>34.0</td><td>40.5</td><td>52.2</td><td>54.6</td><td>50.9</td><td>50.3</td><td>51.0</td><td>31.5</td><td>54.7</td><td>20.2</td><td>44.2</td><td>49.7</td></tr></table>

Eficiency. To complement the theoretical complexity in Sec. 3.5, we report the wall-clock cost and head memory footprint of DP-BOA and its low-rank variant in Tab. 7. Although full DP-BOA uses full-covariance posterior predictive and therefore incurs a quadratic memory cost in the feature dimension, its absolute overhead remains moderate in the OCD setting: the maximum per-sample latency is below 83 ms on the reported benchmarks, and the head memory remains below 0.7 GB. Overall, DP-BOA trades additional computation and memory for accuracy, but the resulting overhead remains moderate and practically acceptable in the OCD setting.

To further mitigate the cost of full-covariance modeling, we evaluate DP-BOA-L. As shown in Tab. 7, DP-BOA-L preserves most of the accuracy of full DP-BOA while substantially reducing both latency and head memory. Its All accuracy drops by only 1.5, 1.2, and 0.9 points on Animalia, CUB, and Stanford-Cars, respectively, while head memory is reduced from 299.2/619.2/626.7 MB to 40.5/51.0/49.7 MB and maximum latency decreases from 46.1/81.7/82.9 ms to 34.0/50.3/44.2 ms. The reduction is especially clear on CUB and StanfordCars, where the larger number of categories makes the full-covariance head more expensive. Overall, DP-BOA-L ofers a favorable accuracy-eficiency trade-of and provides a practical approximation for larger-K or resource-constrained settings.

Robustness to stream order. We use the standard OCD protocol following prior work [60] and evaluate all methods on the same query stream for controlled comparison. To test whether DP-BOA relies on this ordering, we construct burstycorrelated streams by shufling samples within each ground-truth class, grouping them into chunks of up to B = 10 samples, and then randomly shufling these chunks, while keeping the same support/query split, query set, and evaluation metric. This induces stronger local temporal correlation than the standard protocol. As shown in Tab. 8, DP-BOA outperforms SMILE and PHE in All and

Table 8: Robustness to bursty-correlated stream orders. We use local class bursts of size B = 10 to induce strong temporal correlation. Best results are in bold.

<table><tr><td rowspan="2">Method</td><td colspan="3">Pets</td><td colspan="3">StanfordCars</td><td colspan="3">CUB</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>SMILE</td><td>40.0</td><td>46.2</td><td>36.9</td><td>25.1</td><td>42.0</td><td>16.9</td><td>30.9</td><td>50.0</td><td>21.2</td></tr><tr><td>PHE</td><td>47.0</td><td>45.5</td><td>47.9</td><td>30.5</td><td>62.0</td><td>15.3</td><td>33.8</td><td>45.9</td><td>27.8</td></tr><tr><td>DP-BOA</td><td>58.3</td><td>57.7</td><td>58.5</td><td>35.0</td><td>57.3</td><td>24.2</td><td>50.5</td><td>56.8</td><td>47.3</td></tr></table>

Novel accuracy on all three datasets, with a particularly large gain on CUB $( + 1 6 . 7 / + 1 9 . 5 )$ points in All/Novel accuracy over PHE, suggesting that its gains do not rely on a favorable random stream order.

## 5 Conclusion

We revisited on-the-fly category discovery from the perspective of probabilistic decision-making. Instead of relying on heuristic matching rules, DP-BOA formulates the assign-versus-birth choice as a Bayesian evidence comparison, combining a Dirichlet-process prior for open-ended category growth with NIW posterior predictives for geometry-aware and evidence-adaptive online decisions. By leveraging labeled known classes to initialize both the global prior and knowncategory posteriors, DP-BOA provides a new framework for streaming category discovery and achieves strong performance across standard OCD benchmarks.

Limitations and future work. DP-BOA is designed for the conventional OCD setting, where online decisions must be made eficiently without revisiting past samples. First, given the limited auxiliary information in this protocol, we use a small set of hyperparameters and simple fixed defaults, a common practice in Bayesian nonparametric mixture models [12, 15]; these defaults already yield strong empirical performance. In richer scenarios, extra signals (e.g., prior knowledge about class granularity or semantic side information) could potentially enable more adaptive hyperparameter calibration. Second, for eficiency, each category in our method is modeled by a single elliptical component, which provides a strong accuracy–eficiency trade-of on diverse benchmarks. When streams exhibit more complex intra-class structure, such as multi-modality or non-elliptical geometry induced by domain shift, more complex and expressive category models (e.g., multi-component or non-Gaussian likelihoods) may further improve robustness [4, 38]. However, such expressiveness typically increases latency and memory due to additional state maintenance, and may also introduce extra parameters. A future direction is to incorporate richer category models into our probabilistic framework while reducing these overheads.

## Acknowledgements

This work was supported by the MoE Key Lab of Intelligent Perception and Human-Machine Collaboration (ShanghaiTech University), and the Shanghai Key Technology R&D Program No. 25JC3200500.

## References

1. Amari, S.i.: Backpropagation and stochastic gradient descent method. Neurocomputing 5(4-5), 185–196 (1993)

2. An, W., Shi, W., Tian, F., Lin, H., Wang, Q., Wu, Y., Cai, M., Wang, L., Chen, Y., Zhu, H., et al.: Generalized category discovery with large language models in the loop. In: Findings of the Association for Computational Linguistics: ACL 2024. pp. 8653–8665 (2024)

3. Banerjee, A., Biswas, S.: Language-assisted feature representation and lightweight active learning for on-the-fly category discovery. Transactions on Machine Learning Research (2025)

4. Bishop, C.M., Nasrabadi, N.M.: Pattern recognition and machine learning, vol. 4. Springer (2006)

5. Blackwell, D., MacQueen, J.B.: Ferguson distributions via pólya urn schemes. The annals of statistics 1(2), 353–355 (1973)

6. Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., Joulin, A.: Emerging properties in self-supervised vision transformers. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 9650–9660 (2021)

7. Choi, S., Kang, D., Cho, M.: Contrastive mean-shift learning for generalized category discovery. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 23094–23104 (2024)

8. Deng, J., Dong, W., Socher, R., Li, L.J., Li, K., Fei-Fei, L.: Imagenet: A largescale hierarchical image database. In: 2009 IEEE conference on computer vision and pattern recognition. pp. 248–255. Ieee (2009)

9. Dinari, O., Freifeld, O.: Sampling in dirichlet process mixture models for clustering streaming data. In: International Conference on Artificial Intelligence and Statistics. pp. 818–835. PMLR (2022)

10. Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., Minderer, M., Heigold, G., Gelly, S., et al.: An image is worth 16x16 words: Transformers for image recognition at scale. arXiv preprint arXiv:2010.11929 (2020)

11. Du, R., Chang, D., Liang, K., Hospedales, T., Song, Y.Z., Ma, Z.: On-the-fly category discovery. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 11691–11700 (2023)

12. Escobar, M.D., West, M.: Bayesian density estimation and inference using mixtures. Journal of the American Statistical Association 90(430), 577–588 (1995)

13. Ferguson, T.S.: A bayesian analysis of some nonparametric problems. The annals of statistics pp. 209–230 (1973)

14. Fini, E., Sangineto, E., Lathuilière, S., Zhong, Z., Nabi, M., Ricci, E.: A unified objective for novel class discovery. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 9284–9292 (2021)

15. Gershman, S.J., Blei, D.M.: A Tutorial on Bayesian Nonparametric Models. Journal of Mathematical Psychology 56(1), 1–12 (2012)

16. Ghashami, M., Liberty, E., Phillips, J.M., Woodruf, D.P.: Frequent directions: Simple and deterministic matrix sketching. SIAM Journal on Computing 45(5), 1762–1792 (2016)

17. Han, K., Rebufi, S.A., Ehrhardt, S., Vedaldi, A., Zisserman, A.: Autonovel: Automatically discovering and learning novel visual categories. IEEE Transactions on Pattern Analysis and Machine Intelligence (2021)

18. Han, K., Vedaldi, A., Zisserman, A.: Learning to discover novel visual categories via deep transfer clustering. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 8401–8409 (2019)

19. Hartigan, J.A.: Clustering algorithms. John Wiley & Sons, Inc. (1975)

20. He, K., Zhang, X., Ren, S., Sun, J.: Deep residual learning for image recognition. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 770–778 (2016)

21. Hsu, Y.C., Lv, Z., Kira, Z.: Learning to cluster in order to transfer across domains and tasks. In: International Conference on Learning Representations (2018)

22. Hsu, Y.C., Lv, Z., Schlosser, J., Odom, P., Kira, Z.: Multi-class classification without multi-class labels. In: International Conference on Learning Representations (2018)

23. Jia, X., Han, K., Zhu, Y., Green, B.: Joint representation learning and novel category discovery on single-and multi-modal data. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 610–619 (2021)

24. Krause, J., Stark, M., Deng, J., Fei-Fei, L.: 3d object representations for finegrained categorization. In: Proceedings of the IEEE international conference on computer vision workshops. pp. 554–561 (2013)

25. Krizhevsky, A., Hinton, G., et al.: Learning multiple layers of features from tiny images (2009)

26. Krizhevsky, A., Sutskever, I., Hinton, G.E.: Imagenet classification with deep convolutional neural networks. Advances in neural information processing systems 25 (2012)

27. Kuhn, H.W.: The hungarian method for the assignment problem. Naval research logistics quarterly 2(1-2), 83–97 (1955)

28. Liberty, E.: Simple and deterministic matrix sketching. In: Proceedings of the 19th ACM SIGKDD international conference on Knowledge discovery and data mining. pp. 581–588 (2013)

29. Liu, X., Pu, N., Zheng, H., Li, W., Sebe, N., Zhong, Z.: Generate, refine, and encode: Leveraging synthesized novel samples for on-the-fly fine-grained category discovery. In: Proceedings of the IEEE/CVF International Conference on Computer Vision. pp. 1078–1087 (2025)

30. Liu, Y., Cai, Y., Jia, Q., Qiu, B., Wang, W., Pu, N.: Novel class discovery for ultrafine-grained visual categorization. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 17679–17688 (2024)

31. Liu, Y., Han, K.: Debgcd: Debiased learning with distribution guidance for generalized category discovery. arXiv preprint arXiv:2504.04804 (2025)

32. Liu, Y., He, Z., Han, K.: Hyperbolic category discovery. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 9891–9900 (2025)

33. Murphy, K.P.: Machine learning: a probabilistic perspective. MIT press (2012)

34. Ouldnoughi, R., Kuo, C.W., Kira, Z.: Clip-gcd: Simple language guided generalized category discovery. arXiv preprint arXiv:2305.10420 (2023)

35. Parkhi, O.M., Vedaldi, A., Zisserman, A., Jawahar, C.: Cats and dogs. In: 2012 IEEE conference on computer vision and pattern recognition. pp. 3498–3505. IEEE (2012)

36. Peng, X., Bai, Q., Xia, X., Huang, Z., Saenko, K., Wang, B.: Moment matching for multi-source domain adaptation. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 1406–1415 (2019)

37. Pu, N., Li, W., Ji, X., Qin, Y., Sebe, N., Zhong, Z.: Federated generalized category discovery. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 28741–28750 (2024)

38. Rasmussen, C.E.: The infinite gaussian mixture model. In: Solla, S.A., Leen, T.K., Müller, K.R. (eds.) Advances in Neural Information Processing Systems 12. pp. 554–560. MIT Press, Cambridge, MA, USA (2000)

39. Ronen, M., Finder, S.E., Freifeld, O.: Deepdpm: Deep clustering with an unknown number of clusters. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9861–9870 (2022)

40. Schaefer, R., Liu, G.K.M., Du, Y., Linderman, S., Fiete, I.R.: Streaming inference for infinite non-stationary clustering. In: Conference on Lifelong Learning Agents. pp. 310–326. PMLR (2022)

41. Sethuraman, J.: A constructive definition of dirichlet priors. Statistica sinica pp. 639–650 (1994)

42. Su, Y., Zhou, R., Huang, S., Li, X., Wang, T., Wang, Z., Xu, M.: Multimodal generalized category discovery. In: Proceedings of the Computer Vision and Pattern Recognition Conference. pp. 1634–1643 (2025)

43. Tan, K.C., Liu, Y., Ambrose, B., Tulig, M., Belongie, S.: The herbarium challenge 2019 dataset. arXiv preprint arXiv:1906.05372 (2019)

44. Teh, Y.W., Jordan, M.I., Beal, M.J., Blei, D.M.: Hierarchical dirichlet processes. Journal of the american statistical association 101(476), 1566–1581 (2006)

45. Van Horn, G., Mac Aodha, O., Song, Y., Cui, Y., Sun, C., Shepard, A., Adam, H., Perona, P., Belongie, S.: The inaturalist species classification and detection dataset. In: Proceedings of the IEEE conference on computer vision and pattern recognition. pp. 8769–8778 (2018)

46. Vaze, S., Han, K., Vedaldi, A., Zisserman, A.: Generalized category discovery. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 7492–7501 (2022)

47. Vaze, S., Vedaldi, A., Zisserman, A.: No representation rules them all in category discovery. Advances in Neural Information Processing Systems 36, 19962–19989 (2023)

48. Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S.: The caltech-ucsd birds-200-2011 dataset (2011)

49. Wang, C., Paisley, J., Blei, D.M.: Online variational inference for the hierarchical dirichlet process. In: Proceedings of the fourteenth international conference on artificial intelligence and statistics. pp. 752–760. JMLR Workshop and Conference Proceedings (2011)

50. Wang, H., Vaze, S., Han, K.: Hilo: A learning framework for generalized category discovery robust to domain shifts. arXiv preprint arXiv:2408.04591 (2024)

51. Wang, H., Vaze, S., Han, K.: Sptnet: An eficient alternative framework for generalized category discovery with spatial prompt tuning. arXiv preprint arXiv:2403.13684 (2024)

52. Wang, Y., Chen, Z., Yang, D., Sun, Y., Qi, L.: Self-cooperation knowledge distillation for novel class discovery. arXiv preprint arXiv:2407.01930 (2024)

53. Welford, B.P.: Note on a method for calculating corrected sums of squares and products. Technometrics 4(3), 419–420 (1962)

54. Wen, X., Zhao, B., Qi, X.: Parametric classification for generalized category discovery: A baseline study. In: Proceedings of the IEEE/CVF international conference on computer vision. pp. 16590–16600 (2023)

55. Willes, J., Harrison, J., Harakeh, A., Finn, C., Pavone, M., Waslander, S.L.: Bayesian embeddings for few-shot open world recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence 46(3), 1513–1529 (2022)

56. Xu, R., Zhang, C., Ren, H., He, X.: Dual-level adaptive self-labeling for novel class discovery in point cloud segmentation. arXiv preprint arXiv:2407.12489 (2024)

57. Zhang, C., Xu, R., He, X.: Novel class discovery for long-tailed recognition. Transactions on Machine Learning Research (2023)

58. Zhang, S., Khan, S., Shen, Z., Naseer, M., Chen, G., Khan, F.S.: Promptcal: Contrastive afinity learning via auxiliary prompts for generalized novel category discovery. In: Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. pp. 3479–3488 (2023)

59. Zhao, B., Han, K.: Novel visual category discovery with dual ranking statistics and mutual knowledge distillation. Advances in Neural Information Processing Systems 34 (2021)

60. Zheng, H., Pu, N., Li, W., Sebe, N., Zhong, Z.: Prototypical hash encoding for onthe-fly fine-grained category discovery. Advances in Neural Information Processing Systems 37, 101428–101455 (2024)

61. Zhong, Z., Fini, E., Roy, S., Luo, Z., Ricci, E., Sebe, N.: Neighborhood contrastive learning for novel class discovery. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 10867–10875 (2021)

62. Zhong, Z., Zhu, L., Luo, Z., Li, S., Yang, Y., Sebe, N.: Openmix: Reviving known knowledge for discovering novel visual categories in an open world. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. pp. 9462–9470 (2021)

## Contents of the Appendix

– Appendix A: Implementation Details Per-dataset statistics, known/novel splits, support/query construction, and baseline fairness.

– Appendix B: Standard Deviations Standard deviations over three runs for all benchmarks and evaluation metrics.

– Appendix C: NIW Update Equations Definition of the Normal–Inverse–Wishart prior and derivation of the closedform posterior updates.

– Appendix D: Student-t Predictive Densities Derivation of the multivariate Student-t predictives for existing and new categories, plus a 1D illustration of their behavior as evidence grows.

– Appendix E: Dirichlet–Process Predictive Rule Derivation of the DP class-prior over category indices and the corresponding birth probability used in DP-BOA.

– Appendix F: Estimation of NIW Hyperparameters Empirical-Bayes-inspired moment matching for estimating \mu \_0 , \Psi \_0 , and \kap \_0 from labeled support data.

– Appendix G: Online Update of Suficient Statistics Welford-style streaming updates for per-category counts, means, and scatters (n\_k,\bar {mthf z} S) .

– Appendix H: Stronger Encoder Analysis Additional stronger-encoder evaluation with supervised contrastive learning while keeping the DP-BOA online head unchanged.

– Appendix I: Category Decision Boundary Visualization 2D PCA visualizations of Student-t decision ellipses for known and novel DP-BOA categories.

– Appendix K: Complexity Analysis Time and memory complexity of the full DP-BOA head, memory footprint analysis, and the low-rank DP-BOA-L variant for improved scalability.

– Appendix L: More Hyperparameters Analysis Ablations and sensitivity analysis for the NIW mean-strength \kap \_0 and the degrees-of-freedom cap n\_{\mathr cp} . – Appendix M: Feature Geometry Analysis Empirical category-geometry statistics, analysis of the Gaussian approximation, and support–novel mismatch stress tests.

– Appendix N: Comparison with Simpler Probabilistic Heads Comparison between DP-BOA and simpler heads under identical frozen features, including posterior-threshold and Mahalanobis-threshold variants.

– Appendix O: Temporal Diagnostics Stream-level diagnostics of DP-BOA, including false-birth rates, clustermean drift, and evidence margin.

Table 9: Dataset statistics. $| y _ { S } | ~ / ~ | y _ { Q } |$ denote the numbers of known $/$ all classes, and $| \mathcal { D } s | \ / \ | \mathcal { D } _ { Q } |$ denote the sizes of the labeled support set / unlabeled query stream.

<table><tr><td></td><td colspan="2">CIFAR100 ImageNet100</td><td colspan="8">CUB Cars Herb19 Pets Fungi Arachnida Animalia Mollusca</td></tr><tr><td> $|\mathcal{Y}_S|$ </td><td>80</td><td>50</td><td>100</td><td>98</td><td>341</td><td>19</td><td>61</td><td>28</td><td>39</td><td>47</td></tr><tr><td> $|\mathcal{Y}_Q|$ </td><td>100</td><td>100</td><td>200</td><td>196</td><td>683</td><td>38</td><td>121</td><td>56</td><td>77</td><td>93</td></tr><tr><td> $|\mathcal{D}_S|$ </td><td>20.0K</td><td>31.9K</td><td>1.5K</td><td>2.0K</td><td>8.9K</td><td>0.9K</td><td>1.8K</td><td>1.7K</td><td>1.5K</td><td>2.4K</td></tr><tr><td> $|\mathcal{D}_Q|$ </td><td>30.0K</td><td>95.3K</td><td>4.5K</td><td>6.1K</td><td>25.4K</td><td>2.7K</td><td>5.8K</td><td>4.3K</td><td>5.1K</td><td>7.0K</td></tr></table>

## A Implementation Details

## A.1 Dataset Details

We evaluate DP-BOA on ten OCD benchmarks. For CIFAR100, ImageNet100, and Herbarium19, we adopt the oficial splits from On-the-Fly Category Discovery [11]. For the remaining seven fine-grained benchmarks—CUB, Stanford Cars, Oxford-IIIT Pet, and four iNaturalist 2017 [45] super-categories, namely Fungi, Arachnida, Animalia, and Mollusca—we follow the class splits used by PHE [60].

Following the standard OCD protocol [11, 60], for each known class $c \in \mathcal { V } _ { S }$ 2 we use 50\% of its images as the labeled support set $\mathcal { D } _ { S }$ , and place the remaining 50\% into the query stream $\mathcal { D } _ { Q }$ . All images from novel classes $y _ { Q } \backslash y _ { S }$ are used only in $\mathcal { D } _ { Q }$ . Tab. 9 summarizes the dataset statistics.

## A.2 Baseline Fairness

For a controlled comparison, we follow the standard evaluation setting used in prior OCD methods. All compared methods are evaluated with the same DINO backbone family, except Sync [3], which uses CLIP features and language information. Our encoder is fine-tuned only with standard supervised cross-entropy on the labeled support set, whereas several baselines employ additional method-specific representation-learning objectives, such as hashing regularization in SMILE [11].

We also compare against prior methods in their standard forms. Many existing OCD heads are based on hash codes, prototype matching, or thresholded similarity rules, and are not designed to maintain full-covariance category statistics online. Retrofitting these methods with full-covariance predictive modeling would require non-trivial algorithmic changes and would no longer correspond to their original implementations. We therefore report their standard results and isolate the contribution of our probabilistic head through additional same-feature comparisons in Sec. N.

## B Standard Deviations

In the main paper, we report the mean accuracy over three runs with diferent random seeds. For completeness, Tab. 10 reports the corresponding standard deviations of DP-BOA on all benchmarks. All values are measured in percentage points. The standard deviations are generally small for All accuracy, indicating that the overall performance is stable across runs.

Table 10: Standard deviations over three runs. We report the standard deviations of DP-BOA for $A l l ,$ Known, and Novel accuracy on all benchmarks. Values are measured in percentage points.

<table><tr><td>Dataset</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>Animalia</td><td>1.41</td><td>0.24</td><td>2.04</td></tr><tr><td>Arachnida</td><td>0.72</td><td>5.21</td><td>2.13</td></tr><tr><td>CIFAR100</td><td>0.19</td><td>1.24</td><td>2.11</td></tr><tr><td>CUB</td><td>1.40</td><td>1.16</td><td>2.63</td></tr><tr><td>Fungi</td><td>1.09</td><td>2.00</td><td>1.57</td></tr><tr><td>Herbarium19</td><td>0.38</td><td>0.99</td><td>0.22</td></tr><tr><td>ImageNet100</td><td>0.89</td><td>0.57</td><td>1.05</td></tr><tr><td>Mollusca</td><td>0.60</td><td>0.73</td><td>0.54</td></tr><tr><td>OxfordPets</td><td>2.17</td><td>2.60</td><td>3.28</td></tr><tr><td>StanfordCars</td><td>0.54</td><td>1.66</td><td>1.17</td></tr></table>

## C NIW Update Equations

In this section, we first introduce the Normal–Inverse–Wishart (NIW) distribution and its parameters. We then derive step by step that the NIW prior is conjugate to a multivariate Gaussian with unknown mean and covariance, and obtain the closed-form update equations used in DP-BOA.

## C.1 The Normal–Inverse–Wishart Distribution

The Normal–Inverse–Wishart (NIW) distribution is a four-parameter family that serves as the conjugate prior for a multivariate Gaussian $\mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ with unknown mean $\mu$ and covariance $\Sigma .$ It is defined as the product of a Normal distribution over the mean and an Inverse-Wishart distribution over the covariance:

$$
\mathrm{NIW} (\mu , \Sigma \mid \mu_ {0}, \kappa_ {0}, \Psi_ {0}, \nu_ {0}) \triangleq \mathcal {N} \Bigl (\mu \mid \mu_ {0}, \frac {1}{\kappa_ {0}} \Sigma \Bigr) \times \mathcal {W} ^ {- 1} (\Sigma \mid \Psi_ {0}, \nu_ {0})\tag{18}
$$

where we use $\mathcal { W } ^ { - 1 } ( \cdot \mid \varPsi _ { 0 } , \nu _ { 0 } )$ to denote the Inverse-Wishart distribution with scale matrix $\varPsi _ { 0 }$ and degrees of freedom $\nu _ { 0 }$ . Its density is

$$
\mathcal {W} ^ {- 1} (\Sigma \mid \varPsi_ {0}, \nu_ {0}) = \frac {| \varPsi_ {0} | ^ {\frac {\nu_ {0}}{2}}}{2 ^ {\frac {\nu_ {0} d}{2}} \varGamma_ {d} (\frac {\nu_ {0}}{2})} | \Sigma | ^ {- \frac {\nu_ {0} + d + 1}{2}} \exp \left(- \frac {1}{2} \mathrm{tr} (\varPsi_ {0} \varSigma^ {- 1})\right)\tag{19}
$$

where d is the feature dimension and $\textstyle { \cal T } _ { d } ( \cdot )$ is the multivariate gamma function. Combining Eq. (18) and Eq. (19), and dropping normalization constants that do not depend on $( \mu , \Sigma )$ , the prior density $p ( \mu , \Sigma )$ can be written as

$$
p (\mu , \Sigma) \propto | \Sigma | ^ {- \frac {\nu_ {0} + d + 2}{2}} \exp \left(- \frac {1}{2} \operatorname{tr} \left(\Psi_ {0} \Sigma^ {- 1}\right)\right) \exp \left(- \frac {\kappa_ {0}}{2} \left(\mu - \mu_ {0}\right) ^ {\top} \Sigma^ {- 1} \left(\mu - \mu_ {0}\right)\right)\tag{20}
$$

The four hyperparameters have intuitive interpretations:

$\mu _ { 0 } \in \mathbb { R } ^ { d } ;$ : prior mean (the expected center);

$\kappa _ { 0 } > 0 :$ prior strength (efective pseudo-count);

$\varPsi _ { 0 } \in \mathbb { R } ^ { d \times d }$ : prior scale matrix;

$\nu _ { 0 } > d - 1$ : degrees of freedom of the Inverse-Wishart.

## C.2 Estimation detail of Posterior Updates

Consider a category k with $n _ { k }$ observed feature vectors $\mathcal { D } _ { k } = \{ z _ { 1 } , . . . , z _ { n _ { k } } \}$ . As described in Sec. 3.3, we assume $z _ { i } \sim \mathcal { N } ( \boldsymbol { \mu } , \boldsymbol { \Sigma } )$ . The likelihood of the data (up to a constant) is

$$
p (\mathcal {D} _ {k} \mid \mu , \Sigma) \propto | \Sigma | ^ {- \frac {n _ {k}}{2}} \exp \left(- \frac {1}{2} \sum_ {i = 1} ^ {n _ {k}} (z _ {i} - \mu) ^ {\top} \Sigma^ {- 1} (z _ {i} - \mu)\right)\tag{21}
$$

To facilitate combination with the prior, we rewrite the quadratic form using the sample mean $\begin{array} { r } { \bar { z } _ { k } = \frac { 1 } { n _ { k } } \sum _ { i } z _ { i } } \end{array}$ and the scatter matrix $\begin{array} { r } { S _ { k } = \sum _ { i } ( z _ { i } - \bar { z } _ { k } ) ( z _ { i } - } \end{array}$ F $\bar { z } _ { k } ) ^ { \top } : _ { n _ { k } }$ k

$$
\sum_ {i = 1} ^ {n} (z _ {i} - \mu) ^ {\top} \varSigma^ {- 1} (z _ {i} - \mu) = \mathrm{tr} (S _ {k} \varSigma^ {- 1}) + n _ {k} (\bar {z} _ {k} - \mu) ^ {\top} \varSigma^ {- 1} (\bar {z} _ {k} - \mu)\tag{22}
$$

By Bayes’ rule, the posterior (again up to a constant) is

$$
p (\mu , \Sigma \mid \mathcal {D} _ {k}) = \frac {p (\mathcal {D} _ {k} \mid \mu , \Sigma) p (\mu , \Sigma)}{p (\mathcal {D} _ {k})} \propto p (\mathcal {D} _ {k} \mid \mu , \Sigma) p (\mu , \Sigma)\tag{23}
$$

Multiplying the likelihood and prior, and using Eq. (22), we obtain

$$
p (\mu , \Sigma \mid \mathcal {D} _ {k}) \propto | \Sigma | ^ {- \frac {\left(\nu_ {0} + n _ {k}\right) + d + 2}{2}} \exp \left(- \frac {1}{2} \operatorname{tr} \left((\Psi_ {0} + S _ {k}) \Sigma^ {- 1}\right)\right)
$$

$$
\times \exp \left(- \frac {1}{2} \left[ \kappa_ {0} \| \mu - \mu_ {0} \| _ {\Sigma^ {- 1}} ^ {2} + n _ {k} \| \mu - \bar {z} _ {k} \| _ {\Sigma^ {- 1}} ^ {2} \right]\right)\tag{24}
$$

where we use the shorthand $\| \boldsymbol { x } \| _ { \mathcal { X } ^ { - 1 } } ^ { 2 } = \boldsymbol { x } ^ { \top } \Sigma ^ { - 1 } \boldsymbol { x } .$

The quadratic term in $\mu$ can be simplified by completing the square:

$$
\kappa_ {0} \| \mu - \mu_ {0} \| _ {\Sigma^ {- 1}} ^ {2} + n _ {k} \| \mu - \bar {z} _ {k} \| _ {\Sigma^ {- 1}} ^ {2} =
$$

$$
\left(\kappa_ {0} + n _ {k}\right) \left\| \mu - \frac {\kappa_ {0} \mu_ {0} + n _ {k} \bar {z} _ {k}}{\kappa_ {0} + n _ {k}} \right\| _ {\Sigma^ {- 1}} ^ {2} + \frac {\kappa_ {0} n _ {k}}{\kappa_ {0} + n _ {k}} \| \bar {z} _ {k} - \mu_ {0} \| _ {\Sigma^ {- 1}} ^ {2}\tag{25}
$$

Substituting Eq. (25) back into the posterior and collecting terms, we can recognize the form of another NIW distribution with updated parameters.

## C.3 Final Update Equations

Matching the posterior to the NIW form, we obtain the update rules used in the main paper. For a category k with suficient statistics $\{ n _ { k } , \bar { z } _ { k } , S _ { k } \}$ , the posterior $( \mu , \Sigma ) \mid \mathcal { D } _ { k } \sim \mathrm { N I W } ( \mu _ { k } , \kappa _ { k } , \varPsi _ { k } , \nu _ { k } )$ has hyperparameters

$$
\kappa_ {k} = \kappa_ {0} + n _ {k},\tag{26}
$$

$$
\nu_ {k} = \nu_ {0} + n _ {k},\tag{27}
$$

$$
\mu_ {k} = \frac {\kappa_ {0} \mu_ {0} + n _ {k} \bar {z} _ {k}}{\kappa_ {0} + n _ {k}},\tag{28}
$$

$$
\varPsi_ {k} = \varPsi_ {0} + S _ {k} + \frac {\kappa_ {0} n _ {k}}{\kappa_ {0} + n _ {k}} (\bar {z} _ {k} - \mu_ {0}) (\bar {z} _ {k} - \mu_ {0}) ^ {\top}.\tag{29}
$$

These closed-form updates allow DP-BOA to maintain the exact NIW posterior for each category using only $\left( n _ { k } , \bar { z } _ { k } , S _ { k } \right)$

## D Student-t Predictive Densities

In this section, we derive the closed-form predictive distributions obtained by marginalizing the Gaussian parameters $( \mu , \Sigma )$ under the Normal–Inverse–Wishart (NIW) prior/posterior. As shown in Appendix C, conditioning on a category k with suficient statistics $\{ n _ { k } , \bar { z } _ { k } , S _ { k } \}$ yields the NIW posterior

$$
(\mu , \Sigma) \mid \mathcal {D} _ {k} \sim \mathrm{NIW} (\mu_ {k}, \kappa_ {k}, \Psi_ {k}, \nu_ {k}),\tag{30}
$$

where the updated hyperparameters $\left( \mu _ { k } , \kappa _ { k } , \varPsi _ { k } , \nu _ { k } \right)$ are given in Appendix C. We show that integrating out $( \mu , \Sigma )$ leads to multivariate Student-t predictive densities, which correspond to the category-wise likelihood terms used in Sec. 3.3.

## D.1 Multivariate Student-t Distribution

We briefly recall the formulation of the multivariate Student-t distribution. For a d-dimensional random vector $z \in \mathbb { R } ^ { d }$ , the Student-t density with location $\mu \in \mathbb { R } ^ { d }$ 2 positive-definite scale matrix $\boldsymbol { \varLambda } \in \mathbb { R } ^ { d \times d }$ , and degrees of freedom $\nu > 0$ is

$$
t _ {d} (z \mid \mu , \Lambda , \nu) = \frac {\Gamma \left(\frac {\nu + d}{2}\right)}{\Gamma \left(\frac {\nu}{2}\right) (\nu \pi) ^ {d / 2} | \Lambda | ^ {1 / 2}} \left[ 1 + \frac {1}{\nu} (z - \mu) ^ {\top} \Lambda^ {- 1} (z - \mu) \right] ^ {- \frac {\nu + d}{2}},\tag{31}
$$

where Γ (·) is the gamma function.

## D.2 Posterior Predictive for an Existing Category

Consider a category k with NIW posterior $( \mu , \varSigma ) \mid \mathcal { D } _ { k } \sim \mathrm { N I W } ( \mu _ { k } , \kappa _ { k } , \varPsi _ { k } , \nu _ { k } )$ We are interested in the predictive density of a new feature $z \in \mathbb { R } ^ { d }$ given $\mathcal { D } _ { k } \mathrm { : }$ :

$$
p (z \mid \mathcal {D} _ {k}) = \iint p (z \mid \mu , \Sigma) p (\mu , \Sigma \mid \mathcal {D} _ {k}) \mathrm{d} \mu \mathrm{d} \Sigma ,\tag{32}
$$

where $p ( z \mid \mu , \varSigma ) = \mathcal { N } ( z \mid \mu , \varSigma )$ . Since z is conditionally independent of $\mathcal { D } _ { k }$ given $( \mu , \Sigma )$ , we write $p ( z \mid \mu , \Sigma )$ instead of $p ( z \mid \mu , \varSigma , \mathcal { D } _ { k } )$ , and keep the dependence on $\mathcal { D } _ { k }$ only in $p ( \boldsymbol { \mu } , \boldsymbol { \Sigma } \mid \mathcal { D } _ { k } )$

Step 1: Integrating out $\mu .$ . Using the hierarchical form of the NIW posterior,

$$
\Sigma \mid \mathcal {D} _ {k} \sim \mathcal {W} ^ {- 1} (\varPsi_ {k}, \nu_ {k}),\tag{33}
$$

$$
\mu \mid \Sigma , \mathcal {D} _ {k} \sim \mathcal {N} \left(\mu_ {k}, \frac {1}{\kappa_ {k}} \Sigma\right),\tag{34}
$$

the conditional predictive density given Σ is

$$
p (z \mid \Sigma , \mathcal {D} _ {k}) = \int \mathcal {N} (z \mid \mu , \Sigma) \mathcal {N} \left(\mu \mid \mu_ {k}, \frac {1}{\kappa_ {k}} \Sigma\right) d \mu .\tag{35}
$$

This is a convolution of two Gaussians with shared covariance structure, which yields another Gaussian:

$$
p (z \mid \Sigma , \mathcal {D} _ {k}) = \mathcal {N} \left(z \mid \mu_ {k}, \left(1 + \frac {1}{\kappa_ {k}}\right) \Sigma\right).\tag{36}
$$

Step 2: Integrating out Σ. We now integrate out $\varSigma \sim \mathcal { W } ^ { - 1 } ( \varPsi _ { k } , \nu _ { k } )$

$$
p (z \mid \mathcal {D} _ {k}) = \int \mathcal {N} \Bigl (z \mid \mu_ {k}, \left(1 + \frac {1}{\kappa_ {k}}\right) \varSigma \Bigr) \mathcal {W} ^ {- 1} (\varSigma \mid \varPsi_ {k}, \nu_ {k}) \mathrm{d} \varSigma .\tag{37}
$$

It is helpful to recall the general relationship between Gaussian–Inverse-Wishart mixtures and the multivariate Student-t distribution. Let

$$
\begin{array}{r} \Sigma \sim \mathcal {W} ^ {- 1} (\varPsi , \nu), \\ z \mid \Sigma \sim \mathcal {N} (z \mid \mu , c \Sigma), \end{array}\tag{38}
$$

for some scalar $c > 0$ . A standard calculation (see, e.g., [4, 33]) shows that the marginal distribution of z is multivariate Student-t:

$$
z \sim t _ {d} \left(\mu , \frac {c}{\nu - d + 1} \Psi , \nu - d + 1\right).\tag{39}
$$

Intuitively, the Inverse-Wishart prior over Σ plays the role of a scale mixture over Gaussian covariances, and integrating out Σ yields heavy-tailed Student-t marginals.

Comparing Eq. (36) with the general form Eq. (38), we identify

$$
c = 1 + \frac {1}{\kappa_ {k}}, \quad \mu = \mu_ {k}, \quad \Psi = \Psi_ {k}, \quad \nu = \nu_ {k}.
$$

Substituting these into Eq. (39), and rewriting the scale matrix in the form used in Eq. (31), we obtain

$$
p (z \mid \mathcal {D} _ {k}) = t _ {d} \left(z \mid \mu_ {k}, \frac {\kappa_ {k} + 1}{\kappa_ {k} (\nu_ {k} - d + 1)} \Psi_ {k}, \nu_ {k} - d + 1\right).\tag{40}
$$

That is, the predictive distribution for category k is a Student-t with location µ , degrees of freedom $\nu _ { k } - d + 1$ , and scale matrix $\frac { \kappa _ { k } + 1 } { \kappa _ { k } \left( \nu _ { k } - d + 1 \right) } \varPsi _ { k }$

## D.3 Prior Predictive for a New Category

For the “birth” hypothesis, we require a predictive density for a new category whose parameters have not yet been updated by any data. In DP-GMM, such a category is represented only by the global NIW hyperparameters $\left( \mu _ { 0 } , \kappa _ { 0 } , \varPsi _ { 0 } , \nu _ { 0 } \right)$ , which encode our prior belief about a generic category before seeing any of its samples. Equivalently, a hypothetical new category with zero observations $( n _ { k } = 0 )$ has posterior equal to this prior:

$$
(\mu , \Sigma) \mid \text { new } \sim \text { NIW } (\mu_ {0}, \kappa_ {0}, \Psi_ {0}, \nu_ {0}).
$$

The predictive density for a feature z under this new category is therefore the NIW prior predictive, obtained by marginalizing out $( \mu , \Sigma )$ under this global prior:

$$
p (z \mid \text { new }) = \iint p (z \mid \mu , \Sigma)   p (\mu , \Sigma \mid \mu_ {0}, \kappa_ {0}, \Psi_ {0}, \nu_ {0})   \mathrm{d} \mu   \mathrm{d} \Sigma ,\tag{41}
$$

where $p ( z \mid \mu , \varSigma ) = \mathcal { N } ( z \mid \mu , \varSigma )$ and $( \mu , \Sigma ) \sim \mathrm { N I W } ( \mu _ { 0 } , \kappa _ { 0 } , \varPsi _ { 0 } , \nu _ { 0 } )$ . In other words, the $\mathrm { ^ { 6 6 } n e w } ^ { \mathrm { 5 } }$ category reuses exactly the same Gaussian–NIW hierarchy as in the posterior case, but with the hyperparameters $\left( \mu _ { k } , \kappa _ { k } , \varPsi _ { k } , \nu _ { k } \right)$ replaced by the global prior $\left( \mu _ { 0 } , \kappa _ { 0 } , \varPsi _ { 0 } , \nu _ { 0 } \right)$

Repeating the same two-step marginalization as above but now with the prior parameters, we obtain the prior predictive density

$$
p (z \mid \text { new }) = t _ {d} \left(z \mid \mu_ {0}, \frac {\kappa_ {0} + 1}{\kappa_ {0} (\nu_ {0} - d + 1)} \Psi_ {0}, \nu_ {0} - d + 1\right).\tag{42}
$$

In other words, the $\mathrm { ^ { 6 } b i r t h } ^ { \prime \ }$ likelihood is also a multivariate Student-t with location $\mu _ { 0 }$ , degrees of freedom $\nu _ { 0 } - d + 1$ , and scale matrix $\frac { \kappa _ { 0 } + 1 } { \kappa _ { 0 } ( \nu _ { 0 } - d + 1 ) } \varPsi _ { 0 }$

## D.4 Qualitative Behavior of Student-t Predictives

The multivariate Student-t distribution $t _ { d } ( z \mid \mu , \lambda , \nu )$ is controlled by three parameters: the location $\mu ,$ the positive-definite scale matrix \Lambd , and the degrees of freedom $\nu .$ In this appendix we focus on how \nu alone shapes the density when $\mu$ and $\varLambda$ are fixed. For small $\nu ,$ the distribution has heavy, power-law tails and a relatively low central peak; as \nu increases, probability mass moves towards the center, the tails decay faster, and the Student-t gradually approaches the corresponding Gaussian $\mathcal { N } ( \mu , \varLambda )$ . In the limit $\nu \to \infty$ the two coincide.

To visualize this efect, we consider the one-dimensional case $d { = } 1$ and fix $\mu { = } 0$ and $\varLambda { = } 1 . \ \mathrm { F i g . \ 3 }$ plots $t _ { 1 } ( z \mid 0 , 1 , \nu )$ for $\nu \in \{ 0 . 1 , 1 , 2 , 5 , 1 0 0 \}$ , together with the standard normal $\mathcal { N } ( 0 , 1 )$ as the Gaussian limit. When $\nu$ is very small $( \mathrm { e . g . }$ $\scriptstyle \nu = 0 . 1 )$ , the density has a flatter, less peaked center and extremely heavy tails. As \nu increases to 1, 2, and $5 ,$ the peak at the center becomes higher and more concentrated while the tails become lighter. By $\nu { = } 1 0 0$ , the Student-t curve is almost indistinguishable from $\mathcal { N } ( 0 , 1 )$ , illustrating how increasing the degrees of freedom removes heavy tails and yields a more Gaussian-like predictive.

Importantly, the predictive densities used in DP-BOA have exactly this Student-t: for category k we have a predictive with degrees of freedom $\nu _ { k } ^ { \prime } =$ $\nu _ { k } - d + 1$ , where $\nu _ { k } = \nu _ { 0 } + n _ { k }$ grows with the cluster size $n _ { k }$ . Thus, as more samples are assigned to a category, $\nu _ { k } ^ { \prime }$ increases, its predictive becomes more concentrated around $\mu _ { k }$ with lighter tails, and the corresponding decision region of DP-BOA becomes sharper and more confident.

![](images/0c8f48a7c154352f1193106176036f19a923f3ad1d32e08e4ad05ff3386c67e3.jpg)  
Fig. 3: One-dimensional Student-t densities $t _ { 1 } ( z \mid \mu , \lambda , \nu )$ with $\mu { = } 0$ and $\varLambda { = } 1$ for diferent degrees of freedom $\nu \in \{ 0 . 1 , 1 , 2 , 5 , 1 0 0 \}$ , together with the standard normal $\mathcal { N } ( 0 , 1 )$ (dashed) corresponding to the limit $\nu \to \infty$ . As \nu increases, the distribution becomes more concentrated around $\mu$ and its tails become lighter, eventually matching the Gaussian shape.

## E Dirichlet–Process Predictive Rule

In this appendix, we derive the Dirichlet–process (DP) predictive rule that gives the class–prior terms $\boldsymbol { P } ( \boldsymbol { c } _ { t } \mid \mathcal { D } _ { t - 1 } )$ used in the main text. We start from the definition of a DP and then derive the predictive distribution step by step.

## E.1 Dirichlet Process: Definition

Let $\Theta$ be the parameter space for component parameters $( \mathrm { e . g . }$ , Gaussian/NIW parameters in our mixture model). A Dirichlet process $\mathrm { D P } ( \alpha , G _ { 0 } )$ is a distribution over random probability measures $G$ on $\Theta$ such that for any finite measurable partition $( A _ { 1 } , \dotsc , A _ { M } )$ of $\Theta$ we have

$$
\left(G (A _ {1}), \dots , G (A _ {M})\right) \sim \operatorname{Dir} \left(\alpha G _ {0} (A _ {1}), \dots , \alpha G _ {0} (A _ {M})\right),\tag{43}
$$

where $\alpha > 0$ is the concentration parameter and $G _ { 0 }$ is a fixed base measure on Θ. The Dirichlet distribution has density

$$
\operatorname{Dir} (x _ {1}, \dots , x _ {M} \mid \beta_ {1}, \dots , \beta_ {M}) \propto \prod_ {m = 1} ^ {M} x _ {m} ^ {\beta_ {m} - 1}, \qquad x _ {m} \geq 0, \sum_ {m} x _ {m} = 1\tag{44}
$$

with parameters $\beta _ { m } > 0$

A useful fact is that if

$$
(X _ {1}, \dots , X _ {M}) \sim \operatorname{Dir} (\beta_ {1}, \dots , \beta_ {M}),
$$

then the expectation is

$$
\mathbb {E} [ X _ {m} ] = \frac {\beta_ {m}}{\sum_ {j = 1} ^ {M} \beta_ {j}}.\tag{45}
$$

## E.2 DP Mixture and Posterior over G

In a DP mixture model, we draw a random measure $G$ and latent component parameters $\theta _ { t }$ for each sample:

$$
G \sim \mathrm{DP} (\alpha , G _ {0}),\tag{46}
$$

$$
\theta_ {t} \mid G \sim G,\tag{47}
$$

$$
z _ {t} \mid \theta_ {t} \sim p (z \mid \theta_ {t}),\tag{48}
$$

where $z _ { t }$ is the observed feature and $\theta _ { t } \in \Theta$ is the parameter of the component (category) that generated $z _ { t }$

Suppose we have observed $t - 1$ latent parameters $\theta _ { 1 : t - 1 } = ( \theta _ { 1 } , \ldots , \theta _ { t - 1 } )$ We are interested in the posterior over $G$ and the predictive distribution for $\theta _ { t } .$ . To make the DP conjugacy explicit, consider an arbitrary finite partition $( A _ { 1 } , \dotsc , A _ { M } )$ of Θ. Define the counts

$$
n _ {m} = \sum_ {i = 1} ^ {t - 1} \mathbf {1} \left\{\theta_ {i} \in A _ {m} \right\}, \quad N _ {t - 1} = \sum_ {m = 1} ^ {M} n _ {m} = t - 1.\tag{49}
$$

By the definition in $\operatorname { E q } .$ . (43), under the prior we have

$$
\left(G \left(A _ {1}\right), \dots , G \left(A _ {M}\right)\right) \sim \operatorname{Dir} \left(\alpha G _ {0} \left(A _ {1}\right), \dots , \alpha G _ {0} \left(A _ {M}\right)\right).
$$

Conditioning on $\theta _ { 1 : t - 1 }$ , the likelihood only depends on G through the vector $( G ( A _ { 1 } ) , \dots , G ( A _ { M } ) )$ ). Each $\theta _ { i }$ that falls in $A _ { m }$ contributes a factor $G ( A _ { m } )$ to the likelihood, so the posterior over ${ \bigl ( } G ( A _ { 1 } ) , \ldots , G ( A _ { M } ) { \bigr ) }$ given $\theta _ { 1 : t - 1 }$ is

$$
\begin{array}{l} \big (G (A _ {1}), \ldots , G (A _ {M}) \big) \mid \theta_ {1: t - 1} \\ \sim \operatorname{Dir} \big (\alpha G _ {0} (A _ {1}) + n _ {1}, \ldots , \alpha G _ {0} (A _ {M}) + n _ {M} \big). \end{array}\tag{50}
$$

by Dirichlet–multinomial conjugacy.

Comparing Eq. (50) with the defining property in Eq. (43), we see that the posterior over G is still a Dirichlet process:

$$
G \mid \theta_ {1: t - 1} \sim \mathrm{DP} (\alpha + N _ {t - 1}, G _ {0} ^ {\prime}),\tag{51}
$$

where the updated base measure $G _ { 0 } ^ { \prime }$ satisfies

$$
G _ {0} ^ {\prime} (A _ {m}) = \frac {\alpha G _ {0} (A _ {m}) + n _ {m}}{\alpha + N _ {t - 1}} \quad \text { for   all } m.\tag{52}
$$

Eq. (52) extends uniquely from the partition to all measurable sets $A ,$ and can be written compactly as

$$
G _ {0} ^ {\prime} = \frac {\alpha}{\alpha + N _ {t - 1}} G _ {0} + \frac {1}{\alpha + N _ {t - 1}} \sum_ {i = 1} ^ {t - 1} \delta_ {\theta_ {i}},\tag{53}
$$

where $\delta _ { \theta }$ is a point mass at $\theta .$

## E.3 Predictive Distribution for $\theta _ { t }$

We now derive the predictive distribution for the next component parameter $\theta _ { t }$ given $\theta _ { 1 : t - 1 }$ . By the generative model,

$$
p (\theta_ {t} \in A \mid \theta_ {1: t - 1}) = \int G (A) p (G \mid \theta_ {1: t - 1}) \mathrm{d} G.\tag{54}
$$

The right-hand side is the posterior expectation of $G ( A )$ , which we can compute using the finite-dimensional Dirichlet representation. For the partition $( A , A ^ { c } )$ , the prior is

$$
\big (G (A), G (A ^ {c}) \big) \sim \operatorname{Dir} \big (\alpha G _ {0} (A), \alpha G _ {0} (A ^ {c}) \big),
$$

and the posterior over $( G ( A ) , G ( A ^ { c } ) )$ is

$$
\begin{array}{l} \big (G (A), G (A ^ {c}) \big) \mid \theta_ {1: t - 1} \\ \sim \mathrm{Dir} \big (\alpha G _ {0} (A) + n _ {A},   \alpha G _ {0} (A ^ {c}) + N _ {t - 1} - n _ {A} \big). \end{array}\tag{55}
$$

where $\begin{array} { r } { n _ { A } = \sum _ { i = 1 } ^ { t - 1 } { \bf 1 } \{ \theta _ { i } \in A \} } \end{array}$ . By the expectation formula Eq. (45),

$$
p (\theta_ {t} \in A \mid \theta_ {1: t - 1}) = \mathbb {E} [ G (A) \mid \theta_ {1: t - 1} ] = \frac {\alpha G _ {0} (A) + n _ {A}}{\alpha + N _ {t - 1}}.\tag{56}
$$

This is the Dirichlet–process predictive rule: the next draw $\theta _ { t }$ lands in A with probability proportional to a weighted sum of the prior mass $\alpha G _ { 0 } ( A )$ and the number of previous draws that fell in A.

## E.4 Predictive Rule for Category Indices

To connect Eq. (56) with the category index $c _ { t }$ used in the main text, let $\{ \phi _ { 1 } , \ldots , \phi _ { K _ { t - 1 } } \}$ be the $K _ { t - 1 }$ distinct values among $\{ \theta _ { 1 } , \ldots , \theta _ { t - 1 } \}$ , corresponding to the existing categories at time $t - 1$ . Let $n _ { k }$ be the number of times $\theta _ { i }$ takes the value $\phi _ { k } .$ :

$$
n _ {k} = \sum_ {i = 1} ^ {t - 1} \mathbf {1} \{\theta_ {i} = \phi_ {k} \}, \qquad N _ {t - 1} = \sum_ {k = 1} ^ {K _ {t - 1}} n _ {k}.\tag{57}
$$

Consider the measurable sets $A _ { k } = \left\{ \phi _ { k } \right\}$ for $k = 1 , \dots , K _ { t - 1 }$ . For each such singleton, Eq. (56) gives

$$
p (\theta_ {t} = \phi_ {k} \mid \theta_ {1: t - 1}) = \frac {\alpha G _ {0} (\{\phi_ {k} \}) + n _ {k}}{\alpha + N _ {t - 1}}.\tag{58}
$$

In our $\mathrm { D P }$ mixture, the base measure $G _ { 0 }$ is a continuous prior over component parameters (it puts no point mass on any specific value), so $G _ { 0 } ( \{ \phi _ { k } \} ) = 0$ for every previously seen atom $\phi _ { k }$ . In this case, Eq. (58) simplifies to

$$
p (\theta_ {t} = \phi_ {k} \mid \theta_ {1: t - 1}) = \frac {n _ {k}}{\alpha + N _ {t - 1}}.\tag{59}
$$

We now define the category index $c _ { t }$ by

$$
c _ {t} = \left\{ \begin{array}{l l} k, & \text {if \theta_{t} = \phi_{k} for some k}, \\ \text {new,} & \text {if \theta_{t} is a new draw from G_{0}}. \end{array} \right.
$$

The probability of assigning $z _ { t }$ to an existing category k is exactly the probability that $\theta _ { t }$ equals the corresponding atom $\phi _ { k } \colon$

$$
P (c _ {t} = k \mid \theta_ {1: t - 1}) = p (\theta_ {t} = \phi_ {k} \mid \theta_ {1: t - 1}) = \frac {n _ {k}}{\alpha + N _ {t - 1}}.\tag{60}
$$

The remaining probability mass corresponds to drawing a new atom from $G _ { 0 } ;$ this is the “birth” event: Kt-1

$$
\begin{array}{l} \text {in event:} \\ P (c _ {t} = \text {new} \mid \theta_ {1: t - 1}) = 1 - \sum_ {k = 1} ^ {K _ {t - 1}} P (c _ {t} = k \mid \theta_ {1: t - 1}) \\ \qquad = 1 - \sum_ {k = 1} ^ {K _ {t - 1}} \frac {n _ {k}}{\alpha + N _ {t - 1}} \\ \qquad = \frac {\alpha}{\alpha + N _ {t - 1}}, \end{array}\tag{61}
$$

where we used $\begin{array} { r } { \sum _ { k } n _ { k } = N _ { t - 1 } } \end{array}$ . Since the history $\mathcal { D } _ { t - 1 }$ determines the category counts $\{ n _ { k } \}$ (and hence the empirical partition of $\theta _ { 1 : t - 1 } )$ , the same predictive rule holds when conditioning on $\mathcal { D } _ { t - 1 }$ :

$$
P (c _ {t} = k \mid \mathcal {D} _ {t - 1}) = \frac {n _ {k}}{\alpha + N _ {t - 1}},\tag{62}
$$

$$
P (c _ {t} = \mathrm{new} | \mathcal {D} _ {t - 1}) = \frac {\alpha}{\alpha + N _ {t - 1}}.\tag{63}
$$

## F Estimation of NIW Hyperparameters

In this appendix, we provide the moment-matching estimation details for the Normal–Inverse–Wishart (NIW) hyperparameters $\left( \mu _ { 0 } , \kappa _ { 0 } , \varPsi _ { 0 } , \nu _ { 0 } \right)$ used in Sec. 3.4. We refer to this procedure as empirical-Bayes-inspired because it uses labeled support-set statistics to calibrate the global prior, while also making practical approximations for stability and simplicity. In particular, the NIW–Gaussian hierarchy below should be viewed as a working calibration model rather than a claim that all known and future novel categories are exactly generated from this hierarchy.

The main idea is to match empirical moments computed from the labeled known classes in $\mathcal { D } _ { S }$ to the corresponding moments implied by the working NIW–Gaussian model. This yields data-adaptive estimates for $\mu _ { 0 } , \psi _ { 0 }$ , and $\kappa _ { 0 }$ which provide a practical initialization for the DP-BOA birth prior and knowncategory posteriors.

## F.1 Setup and Empirical Statistics

Let $\mathcal { { D } } _ { S }$ denote the set of $K _ { S }$ known classes, and let $\mathcal { D } _ { k } \subset \mathcal { D } _ { S }$ be the support-set features for class k. We define the per-class and pooled statistics (repeating the notation from Sec. 3.4 for completeness):

$$
\bar {\mathbf {z}} _ {k} = \frac {1}{n _ {k}} \sum_ {\mathbf {z} \in \mathcal {D} _ {k}} \mathbf {z} \quad (\mathrm{classmean})
$$

$$
\mathbf {S} _ {k} = \sum_ {\mathbf {z} \in \mathcal {D} _ {k}} (\mathbf {z} - \bar {\mathbf {z}} _ {k}) (\mathbf {z} - \bar {\mathbf {z}} _ {k}) ^ {\top} \quad (\text { class   scatter })
$$

$$
\bar {\mathbf {z}} = \frac {1}{M} \sum_ {k = 1} ^ {K _ {S}} n _ {k} \bar {\mathbf {z}} _ {k} \quad (\mathrm{globalmean})
$$

$$
\pmb {\Sigma} _ {\mathrm{within}} = \frac {1}{M - K _ {S}} \sum_ {k = 1} ^ {K _ {S}} \mathbf {S} _ {k} \quad (\mathrm{pooledwithin-classcovariance})
$$

$$
\pmb {\Sigma} _ {\mathrm{means}} = \frac {1}{K _ {S} - 1} \sum_ {k = 1} ^ {K _ {S}} (\bar {\mathbf {z}} _ {k} - \bar {\mathbf {z}}) (\bar {\mathbf {z}} _ {k} - \bar {\mathbf {z}}) ^ {\top} \quad (\mathrm{covarianceofclassmeans})
$$

$$
\overline {{n ^ {- 1}}} = \frac {1}{K _ {S}} \sum_ {k = 1} ^ {K _ {S}} \frac {1}{n _ {k}} \quad (\text { average   inverse   class   size }).
$$

Here $M = | \mathcal { D } _ { S } |$ is the total number of labeled support samples and d is the feature dimension.

## F.2 NIW Working Model and Basic Moments

We use the following NIW–Gaussian hierarchy as a working model for prior calibration. For each known class $k ,$ we associate a class-level Gaussian with parameters $\left( \mu _ { k } , \Sigma _ { k } \right)$ , and place a shared NIW prior over these parameters:

$$
(\mu_ {k}, \Sigma_ {k}) \sim \mathrm{NIW} (\mu_ {0}, \kappa_ {0}, \Psi_ {0}, \nu_ {0}),\tag{64}
$$

$$
z \mid \mu_ {k}, \Sigma_ {k} \sim \mathcal {N} (z \mid \mu_ {k}, \Sigma_ {k}).\tag{65}
$$

The support features in $\mathcal { D } _ { k }$ are then treated as samples from this class-level Gaussian for the purpose of estimating the global prior statistics. This modeling view is used only to obtain a stable support-set calibration; during online inference, category posteriors are updated sequentially from the stream evidence.

The NIW hierarchy can be written in the standard conditional form (Appendix C):

$$
\Sigma_ {k} \sim \mathcal {W} ^ {- 1} (\varPsi_ {0}, \nu_ {0}),\tag{66}
$$

$$
\mu_ {k} \mid \Sigma_ {k} \sim \mathcal {N} \left(\mu_ {0}, \frac {1}{\kappa_ {0}} \Sigma_ {k}\right).\tag{67}
$$

From this we obtain the basic prior moments:

$$
\mathbb {E} [ \mu_ {k} ] = \mu_ {0},\tag{68}
$$

$$
\operatorname{Cov} \left(\mu_ {k}\right) = \mathbb {E} \left[ \operatorname{Cov} \left(\mu_ {k} \mid \Sigma_ {k}\right) \right] + \operatorname{Cov} \left(\mathbb {E} \left[ \mu_ {k} \mid \Sigma_ {k} \right]\right)\tag{69}
$$

$$
= \mathbb {E} \Big [ \frac {1}{\kappa_ {0}} \Sigma_ {k} \Big ] + 0 = \frac {1}{\kappa_ {0}} \mathbb {E} [ \Sigma_ {k} ].\tag{70}
$$

For the Inverse–Wishart part, a standard property (see, e.g., [4, 33]) is that if $\varSigma \sim \mathcal { W } ^ { - 1 } ( \varPsi _ { 0 } , \nu _ { 0 } )$ with $\nu _ { 0 } > d + 1$ , then

$$
\mathbb {E} [ \Sigma ] = \frac {\Psi_ {0}}{\nu_ {0} - d - 1}.\tag{71}
$$

We will use Eqs. (68), (70) and (71) to construct empirical–Bayes estimates.

## F.3 Prior Mean µ<sub>0</sub>

We justify the choice of $\mu _ { 0 }$ as the empirical global mean z¯ by computing the unconditional mean of a randomly drawn support feature under the NIW–Gaussian hierarchy.

Consider drawing a labeled feature z according to the generative process in Appendix C and Eq. (64). For each sample we can write

$$
z \sim p (z) = \sum_ {k \in \mathcal {Y} _ {S}} P (y = k)   p (z \mid y = k),
$$

where $( \mu _ { k } , \varSigma _ { k } ) \sim \mathrm { N I W } ( \mu _ { 0 } , \kappa _ { 0 } , \varPsi _ { 0 } , \nu _ { 0 } )$ and $z ~ \mid ~ y = k , \mu _ { k } , \varSigma _ { k } \sim \mathcal { N } ( z ~ | ~ \mu _ { k } , \varSigma _ { k } )$ Using the law of total expectation,

$$
\mathbb {E} [ z ] = \mathbb {E} _ {y} \mathbb {E} _ {\mu_ {k}, \Sigma_ {k}} \left[ \mathbb {E} [ z | y = k, \mu_ {k}, \Sigma_ {k} ] \right]\tag{72}
$$

$$
= \mathbb {E} _ {y} \mathbb {E} _ {\mu_ {k}, \Sigma_ {k}} [ \mu_ {k} ] = \mathbb {E} _ {y} [ \mu_ {0} ] = \mu_ {0},\tag{73}
$$

where we used $\mathbb { E } [ \mu _ { k } ] = \mu _ { 0 }$ from Eq. (68). Thus $\mu _ { 0 }$ is the population mean of the marginal feature distribution obtained by integrating out both the class label and the class parameters.

Given the labeled support set $\mathcal { D } _ { S } = \{ z _ { i } \} _ { i = 1 } ^ { M }$ , the empirical mean

$$
\bar {\mathbf {z}} = \frac {1}{M} \sum_ {i = 1} ^ {M} z _ {i} = \frac {1}{M} \sum_ {k = 1} ^ {K _ {S}} n _ {k} \bar {\mathbf {z}} _ {k}\tag{74}
$$

is a consistent, approximately unbiased estimator of $\mathbb { E } [ z ] = \mu _ { 0 }$ . In the empirical– Bayes spirit, we therefore set

$$
\mu_ {0} = \bar {\mathbf {z}},\tag{75}
$$

so that the global mean implied by the NIW prior matches the empirical mean of the labeled features.

## F.4 Covariance Scale $\pmb { { \psi } _ { 0 } }$

We now derive the initialization of the covariance scale $\varPsi _ { 0 }$ under the NIW hierarchy.

Prior mean of class covariances. From Appendix C, the covariance of class k obeys

$$
\Sigma_ {k} \sim \mathcal {W} ^ {- 1} (\Psi_ {0}, \nu_ {0}),\tag{76}
$$

and Eq. (71) gives its prior mean

$$
\mathbb {E} [ \Sigma_ {k} ] = \frac {\varPsi_ {0}}{\nu_ {0} - d - 1}.\tag{77}
$$

Since all classes share the same NIW hyperparameters, $\mathbb { E } [ \Sigma _ { k } ]$ is identical across $k .$

Pooled within-class covariance as an estimator of $\mathbb { E } [ \Sigma _ { k } ]$ . For class k with $n _ { k }$ support features $\{ z _ { i } \} _ { i = 1 } ^ { n _ { k } } \subset \mathcal { D } _ { k }$ drawn i.i.d. from $\mathcal { N } ( \mu _ { k } , \Sigma _ { k } )$ , define the sample covariance as

$$
C _ {k} = \frac {\mathbf {S} _ {k}}{n _ {k} - 1}.\tag{78}
$$

Standard Gaussian results imply

$$
\mathbb {E} [ C _ {k} \mid \mu_ {k}, \Sigma_ {k} ] = \Sigma_ {k}.\tag{79}
$$

Taking expectation over the NIW prior gives

$$
\mathbb {E} [ C _ {k} ] = \mathbb {E} [ \Sigma_ {k} ].\tag{80}
$$

The pooled within-class covariance is

$$
\mathbf {\Sigma} _ {\mathrm{within}} = \frac {1}{M - K _ {S}} \sum_ {k = 1} ^ {K _ {S}} \mathbf {S} _ {k} = \sum_ {k = 1} ^ {K _ {S}} w _ {k} C _ {k},\tag{81}
$$

where $\begin{array} { r } { w _ { k } = \frac { n _ { k } - 1 } { M - K _ { S } } } \end{array}$ and $\textstyle \sum _ { k } w _ { k } = 1$ . Using linearity of expectation and $\operatorname { E q }$ . (80),

$$
\begin{array}{c} \mathbb {E} [ \boldsymbol {\Sigma} _ {\mathrm{within}} ] = \mathbb {E} \Big [ \sum_ {k = 1} ^ {K _ {S}} w _ {k}   C _ {k} \Big ] = \sum_ {k = 1} ^ {K _ {S}} w _ {k}   \mathbb {E} [ C _ {k} ] \\ = \sum_ {k = 1} ^ {K _ {S}} w _ {k}   \mathbb {E} [ \Sigma_ {k} ] = \mathbb {E} [ \Sigma_ {k} ]. \end{array}\tag{82}
$$

Thus $\pmb { \Sigma } _ { \mathrm { w i t h i n } }$ is an unbiased, sample-size–weighted estimator of the prior mean $\mathbb { E } [ \Sigma _ { k } ]$

Moment matching $f o r \varPsi _ { 0 }$ . In the empirical–Bayes framework, we match the NIW prior mean of the class covariances to the pooled within-class covariance:

$$
\mathbb {E} [ \Sigma_ {k} ] \approx \pmb {\Sigma} _ {\mathrm{within}}.\tag{83}
$$

Combining Eqs. (77) and (82) yields

$$
\pmb {\Sigma} _ {\mathrm{within}} \approx \frac {\varPsi_ {0}}{\nu_ {0} - d - 1} \quad \Longrightarrow \quad \varPsi_ {0} = (\nu_ {0} - d - 1) \pmb {\Sigma} _ {\mathrm{within}}.\tag{84}
$$

This gives the covariance-scale initialization used in the main text.

## F.5 Mean Strength $\kappa _ { 0 }$

We now derive the empirical–Bayes estimate for the mean-strength parameter $\kappa _ { 0 }$ by matching the covariance of class means.

Class-mean covariance under the NIW hierarchy. Let $\bar { Z } _ { k }$ denote the class mean corresponding to class $k ,$ obtained from $n _ { k }$ i.i.d. samples $z _ { i } \sim \mathcal { N } ( \mu _ { k } , \Sigma _ { k } )$ under the NIW prior $\operatorname { E q }$ . (64). Conditionally on $\left( \mu _ { k } , \Sigma _ { k } \right)$ 2

$$
\bar {Z} _ {k} \mid \mu_ {k}, \Sigma_ {k} \sim \mathcal {N} \bigg (\mu_ {k}, \frac {1}{n _ {k}} \Sigma_ {k} \bigg),
$$

so

$$
\operatorname{Cov} \left(\bar {Z} _ {k} \mid \mu_ {k}, \Sigma_ {k}\right) = \frac {1}{n _ {k}} \Sigma_ {k}, \quad \mathbb {E} \left[ \bar {Z} _ {k} \mid \mu_ {k}, \Sigma_ {k} \right] = \mu_ {k}.
$$

Applying the law of total covariance,

$$
\begin{array}{l} \operatorname{Cov} (\bar {Z} _ {k}) = \mathbb {E} \big [ \operatorname{Cov} (\bar {Z} _ {k} \mid \mu_ {k}, \Sigma_ {k}) \big ] + \operatorname{Cov} \big (\mathbb {E} [ \bar {Z} _ {k} \mid \mu_ {k}, \Sigma_ {k} ] \big) \\ = \frac {1}{n _ {k}}   \mathbb {E} [ \Sigma_ {k} ] + \operatorname{Cov} (\mu_ {k}). \end{array}\tag{85}
$$

Using Eq. (70), this becomes

$$
\operatorname{Cov} \left(\bar {Z} _ {k}\right) = \left(\frac {1}{n _ {k}} + \frac {1}{\kappa_ {0}}\right) \mathbb {E} \left[ \Sigma_ {k} \right].\tag{86}
$$

Matching to empirical class-mean covariance. On the empirical side, the observed class means $\{ \bar { \mathbf { z } } _ { k } \} _ { k = 1 } ^ { K _ { S } }$ with global mean z¯ define the sample covariance of class means $\pmb { \Sigma } _ { \mathrm { m e a n s } }$ introduced in the setup above. This is a natural estimator of the model-side covariance Cov $( \hat { Z } _ { k } )$ in Eq. (86), i.e.,

$$
\mathrm{Cov} (\bar {Z} _ {k}) \approx \pmb {\Sigma} _ {\mathrm{means}}.
$$

By $\operatorname { E q . }$ (83), the pooled within-class covariance $\pmb { \Sigma } _ { \mathrm { w i t h i n } }$ provides an empirical estimate of the prior mean of the class covariances,

$$
\mathbb {E} [ \Sigma_ {k} ] \approx \pmb {\Sigma} _ {\mathrm{within}}.
$$

Substituting these plug-in estimates into Eq. (86) and averaging over classes (so that $\mathbb { E } [ 1 / n _ { k } ]$ is approximated by $\overline { { n ^ { - 1 } } } )$ yields the matrix-valued momentmatching relation

$$
\pmb {\Sigma} _ {\mathrm{means}} \approx \left(\overline {{n ^ {- 1}}} + \frac {1}{\kappa_ {0}}\right) \pmb {\Sigma} _ {\mathrm{within}},\tag{87}
$$

which we use to estimate $\kappa _ { 0 }$ .

Trace-based scalar approximation. The matrix relation in $\operatorname { E q } .$ (87) cannot in general be matched exactly with a single scalar $\kappa _ { 0 } ,$ especially when the empirical covariance of class means and the pooled within-class covariance have diferent eigenspaces. We therefore use a trace-based moment-matching approximation, which matches the average variance across dimensions:

$$
\mathrm{tr} (\pmb {\Sigma} _ {\mathrm{means}}) \approx \left(\overline {{n ^ {- 1}}} + \frac {1}{\kappa_ {0}}\right) \mathrm{tr} (\pmb {\Sigma} _ {\mathrm{within}}).\tag{88}
$$

Solving for $1 / \kappa _ { 0 }$ gives

$$
\frac {1}{\kappa_ {0}} \approx \frac {\mathrm{tr} (\pmb {\Sigma} _ {\mathrm{means}})}{\mathrm{tr} (\pmb {\Sigma} _ {\mathrm{within}})} - \overline {{n ^ {- 1}}},\tag{89}
$$

and hence

$$
\kappa_ {0} \approx \left(\frac {\mathrm{tr} (\boldsymbol {\Sigma} _ {\mathrm{means}})}{\mathrm{tr} (\boldsymbol {\Sigma} _ {\mathrm{within}})} - \overline {{n ^ {- 1}}}\right) ^ {- 1}.\tag{90}
$$

In practice, we use this trace-based approximation as a stable scalar calibration of the NIW mean strength, while retaining the full covariance structure in $\pmb { \Sigma } _ { \mathrm { w i t h i n } }$ 1 and the subsequent NIW posterior updates.

## G Online Update of Suficient Statistics

In this part, following [53], we derive the online update rules for the per-category suficient statistics $( n _ { k } , \bar { \mathbf { z } } _ { k } , \mathbf { S } _ { k } )$ used by DP-BOA. These rules allow us to update the statistics of the assigned category for each incoming feature $\mathbf { z } _ { t }$ in constant memory, without storing past samples.

## G.1 Suficient Statistics and Invariants

For a given category $k ,$ let $\mathcal { D } _ { k }$ denote the current set of features assigned to this category, and let $n _ { k } = | \mathcal { D } _ { k } |$ be its cardinality. We maintain the following suficient statistics (Sec. 3.5 and Appendix F):

$$
\bar {\mathbf {z}} _ {k} = \frac {1}{n _ {k}} \sum_ {\mathbf {z} \in \mathcal {D} _ {k}} \mathbf {z},\tag{91}
$$

$$
\mathbf {S} _ {k} = \sum_ {\mathbf {z} \in \mathcal {D} _ {k}} (\mathbf {z} - \bar {\mathbf {z}} _ {k}) (\mathbf {z} - \bar {\mathbf {z}} _ {k}) ^ {\top}.\tag{92}
$$

Here $n _ { k }$ is the category size, $\bar { \mathbf { z } } _ { k }$ is the sample mean, and $\mathbf { S } _ { k }$ is the within-category scatter matrix. Note that Eq. (92) is defined with respect to the current mean $\bar { \mathbf { z } } _ { k }$ , so both $\bar { \mathbf { z } } _ { k }$ and $\mathbf { S } _ { k }$ must be updated when a new sample is added.

In what follows we drop the index k and the time superscript for readability, and write $( n , { \bar { \mathbf { z } } } , \mathbf { S } )$ for the pre-update statistics of the category receiving $\mathbf { z } _ { t }$ .

## G.2 Online Update of the Mean

Suppose the current category has statistics $( n , { \bar { \mathbf { z } } } )$ , and a new feature $\mathbf { z } _ { t }$ is assigned to this category. The updated count is

$$
n ^ {+} = n + 1.\tag{93}
$$

By the definition of the sample mean, the updated mean is

$$
\bar {\mathbf {z}} ^ {+} = \frac {1}{n ^ {+}} \left(\sum_ {\mathbf {z} \in \mathcal {D} _ {k}} \mathbf {z} + \mathbf {z} _ {t}\right) = \frac {n \bar {\mathbf {z}} + \mathbf {z} _ {t}}{n ^ {+}}.\tag{94}
$$

It is convenient to write this in incremental form. Define the deviation

$$
\pmb {\delta} = \mathbf {z} _ {t} - \bar {\mathbf {z}},\tag{95}
$$

then Eq. (94) becomes

$$
\bar {\mathbf {z}} ^ {+} = \bar {\mathbf {z}} + \frac {1}{n ^ {+}} \pmb {\delta}.\tag{96}
$$

## G.3 Online Update of the Scatter

We now derive the corresponding online update for the scatter matrix S in Eq. (92). Let $\mathcal { D } _ { k } ^ { + } = \mathcal { D } _ { k } \cup \{ { \mathbf z } _ { t } \}$ denote the updated sample set, and let $( n ^ { + } , \bar { { \bf z } } ^ { + } , { \bf S } ^ { + } )$ 1 be the updated statistics. By definition,

$$
\mathbf {S} ^ {+} = \sum_ {\mathbf {z} \in \mathcal {D} _ {k} ^ {+}} (\mathbf {z} - \bar {\mathbf {z}} ^ {+}) (\mathbf {z} - \bar {\mathbf {z}} ^ {+}) ^ {\top}.\tag{97}
$$

We decompose the sum in Eq. (97) into the contribution from the old samples and the new sample:

$$
\begin{array}{c} \mathbf {S} ^ {+} = \sum_ {\mathbf {z} \in \mathcal {D} _ {k}} (\mathbf {z} - \bar {\mathbf {z}} ^ {+}) (\mathbf {z} - \bar {\mathbf {z}} ^ {+}) ^ {\top} \\ + (\mathbf {z} _ {t} - \bar {\mathbf {z}} ^ {+}) (\mathbf {z} _ {t} - \bar {\mathbf {z}} ^ {+}) ^ {\top}. \end{array}\tag{98}
$$

For the first term, write ${ \bf z } - \bar { \bf z } ^ { + } = ( { \bf z } - \bar { \bf z } ) + ( \bar { \bf z } - \bar { \bf z } ^ { + } )$ . Using $\begin{array} { r } { \sum _ { \mathbf { z } \in \mathcal { D } _ { k } } ( \mathbf { z } - \bar { \mathbf { z } } ) = \mathbf { 0 } } \end{array}$ 2 we obtain

$$
\begin{array}{l} \sum_ {\mathbf {z} \in \mathcal {D} _ {k}} (\mathbf {z} - \bar {\mathbf {z}} ^ {+}) (\mathbf {z} - \bar {\mathbf {z}} ^ {+}) ^ {\top} \\ = \sum_ {\mathbf {z} \in \mathcal {D} _ {k}} \left[ (\mathbf {z} - \bar {\mathbf {z}}) (\mathbf {z} - \bar {\mathbf {z}}) ^ {\top} + (\bar {\mathbf {z}} - \bar {\mathbf {z}} ^ {+}) (\bar {\mathbf {z}} - \bar {\mathbf {z}} ^ {+}) ^ {\top} \right] \\ = \mathbf {S} + n (\bar {\mathbf {z}} - \bar {\mathbf {z}} ^ {+}) (\bar {\mathbf {z}} - \bar {\mathbf {z}} ^ {+}) ^ {\top}. \end{array}\tag{99}
$$

Substituting Eq. (99) into Eq. (98) gives

$$
\begin{array}{c} \mathbf {S} ^ {+} = \mathbf {S} + n \left(\bar {\mathbf {z}} - \bar {\mathbf {z}} ^ {+}\right) \bigl (\bar {\mathbf {z}} - \bar {\mathbf {z}} ^ {+} \bigr) ^ {\top} \\ + \bigl (\mathbf {z} _ {t} - \bar {\mathbf {z}} ^ {+} \bigr) \bigl (\mathbf {z} _ {t} - \bar {\mathbf {z}} ^ {+} \bigr) ^ {\top}. \end{array}\tag{100}
$$

Next we express all terms in Eq. (100) in terms of the deviation $\pmb { \delta } = \mathbf { z } _ { t } - \bar { \mathbf { z } }$ and the updated mean Eq. (96). From Eq. (96) we have

$$
\bar {\bf {z}} ^ {+} - \bar {\bf {z}} = \frac {1}{n ^ {+}} \delta ,\tag{101}
$$

$$
\bar {\bf {z}} - \bar {\bf {z}} ^ {+} = - \frac {1}{n ^ {+}} \delta ,\tag{102}
$$

$$
\mathbf {z} _ {t} - \bar {\mathbf {z}} ^ {+} = \pmb {\delta} - (\bar {\mathbf {z}} ^ {+} - \bar {\mathbf {z}}) = \pmb {\delta} - \frac {1}{n ^ {+}} \pmb {\delta} = \frac {n}{n ^ {+}} \pmb {\delta}.\tag{103}
$$

Substituting into Eq. (100) yields

$$
\begin{array}{l} \mathbf {S} ^ {+} = \mathbf {S} + n \left(- \frac {1}{n ^ {+}} \boldsymbol {\delta}\right) \left(- \frac {1}{n ^ {+}} \boldsymbol {\delta}\right) ^ {\top} \\ \qquad + \left(\frac {n}{n ^ {+}} \boldsymbol {\delta}\right) \left(\frac {n}{n ^ {+}} \boldsymbol {\delta}\right) ^ {\top} \\ \qquad = \mathbf {S} + \frac {n}{(n ^ {+}) ^ {2}} \boldsymbol {\delta} \boldsymbol {\delta} ^ {\top} + \frac {n ^ {2}}{(n ^ {+}) ^ {2}} \boldsymbol {\delta} \boldsymbol {\delta} ^ {\top} \\ \qquad = \mathbf {S} + \frac {n (n + 1)}{(n + 1) ^ {2}} \boldsymbol {\delta} \boldsymbol {\delta} ^ {\top} \\ \qquad = \mathbf {S} + \frac {n}{n ^ {+}}   \boldsymbol {\delta} \boldsymbol {\delta} ^ {\top}. \end{array}\tag{104}
$$

Therefore, the scatter update can be written as

$$
\mathbf {S} ^ {+} = \mathbf {S} + \frac {n}{n + 1} (\mathbf {z} _ {t} - \bar {\mathbf {z}}) (\mathbf {z} _ {t} - \bar {\mathbf {z}}) ^ {\top}.\tag{105}
$$

In practice, a numerically stable and commonly used equivalent form is

$$
\mathbf {S} ^ {+} = \mathbf {S} + (\mathbf {z} _ {t} - \bar {\mathbf {z}}) (\mathbf {z} _ {t} - \bar {\mathbf {z}} ^ {+}) ^ {\top},\tag{106}
$$

where $\bar { \mathbf { z } } ^ { + }$ is given by Eq. (96). Eqs. (105) and (106) are algebraically equivalent: substituting $\begin{array} { r } { { \bf z } _ { t } - \bar { \bf z } ^ { + } = \frac { n } { n ^ { + } } ( { \bf z } _ { t } - \bar { \bf z } ) } \end{array}$ into Eq. (106) recovers Eq. (105).

## G.4 Initialization for a New Category

When a new category is born at time t (i.e., it has no previously assigned samples), we initialize its statistics using the current feature $\mathbf { z } _ { t }$ :

$$
n ^ {+} = 1,\tag{107}
$$

$$
\bar {\mathbf {z}} ^ {+} = \mathbf {z} _ {t},\tag{108}
$$

$$
\mathbf {S} ^ {+} = \mathbf {0}.\tag{109}
$$

Subsequently, the category statistics $( n , { \bar { \mathbf { z } } } , \mathbf { S } )$ are updated online using Eqs. (96) and (106). Combined with the NIW posterior updates in Appendix C, these online updates ensure that DP-BOA only needs to maintain $( n _ { k } , \bar { \mathbf { z } } _ { k } , \mathbf { S } _ { k } )$ per category, without storing any past samples.

## H Stronger Encoder Analysis

The main results use our default encoder training protocol: a DINO-pretrained ViT-B/16 backbone is fine-tuned on the labeled support set with standard supervised cross-entropy (CE), and the encoder is then frozen during online inference. To examine whether DP-BOA can benefit from stronger representations, we additionally evaluate an encoder fine-tuned with a supervised contrastive auxiliary loss (SupCon). This experiment changes only the ofline encoder fine-tuning objective; the DP-BOA online head, empirical prior initialization, and test-time update rules are kept unchanged.

Table 11: Stronger encoder analysis. CE denotes the default encoder trained with cross-entropy. CE+SupCon adds a supervised contrastive auxiliary loss with $\lambda = 0 . 1$

<table><tr><td rowspan="2">Loss</td><td colspan="3">CUB</td><td colspan="3">ImageNet100</td><td colspan="3">Animalia</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>CE</td><td>53.4</td><td>57.2</td><td>51.6</td><td>33.8</td><td>75.8</td><td>12.7</td><td>50.7</td><td>67.6</td><td>43.7</td></tr><tr><td>CE+SupCon</td><td>55.8</td><td>62.4</td><td>52.5</td><td>47.1</td><td>84.7</td><td>28.2</td><td>51.5</td><td>68.6</td><td>44.5</td></tr></table>

Specifically, we optimize

$$
\mathcal {L} _ {\mathrm{enc}} = \mathcal {L} _ {\mathrm{CE}} + \lambda \mathcal {L} _ {\mathrm{supcon}}, \qquad \lambda = 0. 1,\tag{110}
$$

where $\mathcal { L } _ { \mathrm { C E } }$ is the standard cross-entropy loss on the labeled support set. We use the standard supervised contrastive loss

$$
\mathcal {L} _ {\mathrm{supcon}} = \frac {1}{| \mathcal {B} |} \sum_ {i \in \mathcal {B}} \frac {- 1}{| \mathcal {P} (i) |} \sum_ {p \in \mathcal {P} (i)} \log \frac {\exp (\mathrm{sim} (\mathbf {h} _ {i} , \mathbf {h} _ {p}) / \tau)}{\sum_ {a \in \mathcal {B} \setminus \{i \}} \exp (\mathrm{sim} (\mathbf {h} _ {i} , \mathbf {h} _ {a}) / \tau)},\tag{111}
$$

where $\mathbf { h } _ { i }$ denotes the normalized feature of sample i, sim $( \cdot , \cdot )$ is cosine similarity, ${ \mathcal { P } } ( i ) = \{ p \in B \setminus \{ i \} : y _ { p } = y _ { i } \}$ is the set of positive samples in the mini-batch $B ,$ and τ is the temperature parameter.

Tab. 11 compares the default CE encoder with the stronger CE+SupCon encoder. DP-BOA consistently benefits from the stronger representation, with especially clear gains on ImageNet100. This indicates that better feature geometry is complementary to our probabilistic decision framework: representation learning can improve the frozen feature space, while DP-BOA remains responsible for online category birth and assignment.

## I Category Decision Boundary Visualization

To qualitatively illustrate (i) the anisotropic geometry modeled by DP-BOA and (ii) how its posterior–predictive high-density regions (and hence the induced decision regions) adapt as evidence accumulates, we visualize individual DP-BOA categories in a fixed 2D feature subspace. We run OCD on the Oxford-IIIT Pet dataset with a frozen encoder. Throughout this part, a “category” refers to a cluster indexed by DP-BOA. For known categories, each such cluster is initialized from a single labeled class; for novel categories, the clusters are discovered online from unlabeled query samples.

For a chosen DP-BOA category k, we collect all features that were ever assigned to it: for known categories this includes both labeled support examples and their subsequent query assignments, while for novel categories it consists only of the query samples assigned after the category is born. Let $\{ \mathbf { x } _ { i } \} _ { i = 1 } ^ { N _ { k } } \subset \mathbb { R } ^ { d }$ denote these assigned features. We construct a category-specific 2D projection by PCA on $\{ { \bf { x } } _ { i } \}$ . Writing $\bar { \mathbf { x } } _ { k }$ for the empirical mean and $\mathbf { C } _ { k }$ for the empirical covariance, we take the top two eigenvectors $\mathbf { v } _ { k , 1 } , \mathbf { v } _ { k , 2 } \ \in \ \mathbb { R } ^ { d }$ and form $\mathbf { P } _ { k } = [ \mathbf { v } _ { k , 1 } , \mathbf { v } _ { k , 2 } ] \in \mathbb { R } ^ { d \times 2 }$ . Each feature is mapped to $\mathbf { z } _ { i } = \mathbf { P } _ { k } ^ { \top } ( \mathbf { x } _ { i } - \bar { \mathbf { x } } _ { k } ) \in \mathbb { R } ^ { 2 }$ ， and we treat this 2D PCA subspace as the working space for visualization. For consistency with the full-dimensional NIW prior, we project the pooled withinclass covariance into this subspace to obtain a 2D NIW prior, and then run the same online NIW/Student-t updates as in the main algorithm, now on the 2D points $\mathbf { z } _ { i }$ (so the Student-t formula in Appendix D is used with d=2).

![](images/e05e034d8e4451cc0b768f238c552b544267d2c134517f8848580258ba9e2278.jpg)  
Fig. 4: Evidence-adaptive decision boundaries for three known DP-BOA categories on the Oxford-IIIT Pet dataset. Each row corresponds to one category that is initialized from a labeled class, and columns show checkpoints where 2%, 5%, 10%, 25%, 50%, 75%, and 100% of the category’s assigned query samples have been observed (left to right). Light gray points indicate all PCA-projected features from the dataset, colored points highlight the labeled support and the query samples currently assigned to this category, and the red ellipse is a fixed Mahalanobis-radius level set of the 2D Student-t posterior–predictive distribution. The ellipses are anisotropic and largely stable over time, reflecting well-estimated covariances and high confidence for known categories.

Along each category’s query stream, we record several representative checkpoints at diferent stages of online evidence accumulation (2%, 5%, 10%, 25%, 50%, 75%, and 100%). At each checkpoint, we take the current NIW posterior $\left( \mu _ { k } , \kappa _ { k } , \varPsi _ { k } , \nu _ { k } \right)$ estimated from the samples observed so far, convert it to the corresponding Student-t predictive distribution as in Appendix D, and visualize an elliptical level set with a constant Mahalanobis radius, namely a fixed $^ { 6 } n _ { \mathrm { s t d } } \sigma ^ { \prime }$ contour with respect to the predictive scale matrix. This allows us to inspect how the posterior-predictive uncertainty and category geometry evolve as more samples are assigned to the category. For a fixed category $k ,$ we keep the PCA projection $\mathbf { P } _ { k }$ fixed across all checkpoints, so that changes in the ellipse directly reflect the evolution of the posterior-predictive geometry, rather than being caused by axis rescaling or changes in the visualization coordinate system.

Known categories. Fig. 4 shows three known DP-BOA categories on Oxford-IIIT Pet, each initialized from a distinct labeled class. Each row corresponds to one such category, and columns show snapshots at increasing fractions of assigned query samples (from 2% to 100%, left to right).

![](images/880d4bab51f1a2660f2c79550c59b3ffad7143cfc45636609dd6a9e711d9669f.jpg)  
Fig. 5: Evidence-adaptive decision boundaries for three novel DP-BOA categories on the Oxford-IIIT Pet dataset. These clusters contain only unlabeled query samples and have no ground-truth labels. Early contours are influenced by the global NIW prior and a small, noisy empirical scatter, so they can expand or rotate as $S _ { k }$ rapidly reshapes the posterior covariance $\varPsi _ { k }$ . As more points are assigned, $S _ { k }$ stabilizes, the NIW parameters $\left( \kappa _ { k } , \nu _ { k } \right)$ grow, and the Student-t predictive contours become smaller and more stable, yielding tight, anisotropic decision regions around each discovered novel category.

The induced high-density regions are clearly anisotropic: diferent known categories exhibit diferent orientations and aspect ratios that are aligned with their empirical scatter in the PCA subspace.

In the 2D subspace, the predictive scale used to draw the ellipse has the form

$$
\Sigma_ {\mathrm{pred}, k} = \frac {\kappa_ {k} + 1}{\kappa_ {k} (\nu_ {k} - d + 1)} \Psi_ {k},\tag{112}
$$

where $d { = } 2$ here and $\varPsi _ { k }$ is the NIW scale matrix updated from the scatter $S _ { k }$ of the category. For i.i.d. data with (finite) population covariance $\varSigma ^ { \star }$ in this subspace, the scatter satisfies $\mathbb { E } [ S _ { k } ] = \left( n _ { k } - 1 \right) \Sigma ^ { \star }$ , and by the law of large numbers $S _ { k } / n _ { k } \to \Sigma ^ { \star }$ as $n _ { k } \to \infty$ . Thus $S _ { k }$ (and hence $\varPsi _ { k }$ through the NIW update) grows approximately linearly with the number of assigned samples $n _ { k }$ for moderate to large $n _ { k }$ . At the same time, $\kappa _ { k } = \kappa _ { 0 } + n _ { k }$ ， $\nu _ { k } = \nu _ { 0 } + n _ { k }$ , so in the large-n<sub>k</sub> regime

$$
\frac {\kappa_ {k} + 1}{\kappa_ {k} (\nu_ {k} - d + 1)} \approx \frac {1}{n _ {k}}.\tag{113}
$$

As a result, the linear growth of $\varPsi _ { k }$ with $n _ { k }$ and the roughly $1 / n _ { k }$ decay of the scalar prefactor largely cancels, and $\Sigma _ { \mathrm { p r e d } , k }$ converges to a finite, anisotropic covariance close to the underlying within-class covariance in this subspace. For known classes, we already start with dozens of labeled examples, so this largesample regime is reached quickly: the empirical scatter changes slowly, the eigenvectors and eigenvalue ratios of $\varPsi _ { k }$ are stable, and the ellipses in ${ \mathrm { F i g . } }$ 4 preserve their orientation and aspect ratio with only mild adjustments in overall size.

In addition, as $n _ { k }$ grows, the degrees of freedom $\nu _ { k } - d + 1$ increase, so the Student-t predictive becomes less heavy-tailed and closer to a Gaussian. Intuitively, this concentrates probability mass closer to the mean $\mu _ { k }$ and makes high-density regions slightly tighter. Together, these efects produce the intended behavior: for well-supported known categories, DP-BOA maintains a stable, anisotropic posterior–predictive region (and thus a stable local decision region) whose scale and tail behavior reflect high confidence in the underlying category.

Novel categories. Fig. 5 shows three automatically discovered novel DP-BOA categories on Oxford-IIIT Pet, visualized with the same protocol. These clusters are formed purely from unlabeled query samples and do not correspond to any ground-truth label. When such a category is first born, its initial NIW posterior is obtained by combining the global prior $\left( \mu _ { 0 } , \kappa _ { 0 } , \varPsi _ { 0 } , \nu _ { 0 } \right)$ with the first birth sample. The prior covariance $\varPsi _ { 0 }$ encodes only an average “typical” scale across known classes; in the local 2D subspace of a specific novel cluster, the true covariance can difer substantially.

In the very early snapshots, $n _ { k }$ is small and the empirical scatter $S _ { k }$ is based on only a few points, so the NIW update $\varPsi _ { k } = \varPsi _ { 0 } + S _ { k } +$ (rank-one mean term) is still a compromise between the global prior $\varPsi _ { 0 }$ and a noisy local estimate. The resulting $\varPsi _ { k }$ may not yet be aligned with the principal direction of the emerging cluster, and the degrees of freedom $\nu _ { k } - d + 1$ are still close to their prior value, so the Student-t predictive remains relatively heavy-tailed. As more samples are assigned, $S _ { k }$ quickly grows and dominates the update of $\varPsi _ { k }$ , causing the ellipse to expand or rotate to align with the principal direction of the novel cluster (left to middle columns of Fig. 5). Once a novel category has accumulated enough samples, its empirical covariance $S _ { k } / ( n _ { k } - 1 )$ stabilizes and closely matches a local population covariance $\varSigma ^ { \star }$ aligned with the discovered cluster. Beyond this point, the same compensation mechanism as for known categories takes efect: $S _ { k }$ grows roughly linearly with $n _ { k }$ , while the scalar factor $\frac { \kappa _ { k } + 1 } { \kappa _ { k } ( \nu _ { k } - d + 1 ) }$ decays approximately like $1 / n _ { k }$ , so the predictive covariance $\Sigma _ { \mathrm { p r e d } , k }$ approaches a stable anisotropic limit rather than shrinking indefinitely. Meanwhile, the increasing degrees of freedom $\nu _ { k } - d + 1$ make the Student-t predictive less heavy-tailed, further tightening the high-density region around $\mu _ { k } .$ . Accordingly, the later checkpoints (rightmost columns in Fig. 5) show ellipses that have settled into a tight, stable shape around each novel cluster, with only minor further changes as more evidence arrives. Taken together, Fig. 4 and Fig. 5 demonstrate that DP-BOA (i) models category-specific anisotropic geometry and (ii) adapts each posterior–predictive (and hence decision) region from a prior-dominated shape to a data-driven, stable covariance as evidence accumulates, both for known categories aligned with labeled classes and for purely discovered novel clusters.

## J Pseudocode for DP-BOA

We summarize DP-BOA as a single procedure containing an ofline initialization phase and an online birth-or-assign phase. Notation and all closed-form NIW / Student-t expressions follow the main text.

<div class="mineru-algorithm" style="white-space: pre-wrap; font-family:monospace;">
Algorithm 1 DP-BOA (Offline Initialization + Online Inference)

1: Input: support set $\mathcal{D}_S$, query stream $\mathcal{D}_Q$.
2: Output: assignments $\{\hat{c}_t\}_{t=1}^N$

3: Offline: supervised encoder + NIW initialization
4: Train encoder $f_\theta$ and a linear classifier on $\mathcal{D}_S$ with cross-entropy
5: Discard the classifier and freeze $f_\theta$
6: Extract features $z_i = f_\theta(x_i)$ for $(x_i, y_i) \in \mathcal{D}_S$ and group by label
7: For each class $k$, compute $n_k$, class mean $\bar{z}_k$, and scatter $S_k$
8: From $\{(n_k, \bar{z}_k, S_k)\}$, estimate NIW prior ($\mu_0, \Psi_0, \kappa_0$)
by the Empirical–Bayes procedure and set $\nu_0$ via the heuristic in Eq. (11)
9: For each known class $k$, apply the NIW posterior update to obtain
$\theta_k = (\mu_k, \Psi_k, \nu_k, \kappa_k)$ and initialize a cluster $C_k$

10: Online: DP-BOA birth-or-assign loop
11: Let $\mathcal{C}$ be the list of current clusters (initialized by known classes)
12: for each query sample $x_t$ in $\mathcal{D}_Q$ do
13: Compute feature $z_t = f_\theta(x_t)$
14: Compute “new”-cluster Student-$t$ predictive $t_d(z_t \mid \theta_0)$
using Eq. (3)
15: Birth log-score: $\ell_{\text{new}} \leftarrow \log \alpha + \log t_d(z_t \mid \theta_0)$
16: for each existing cluster $C_k \in \mathcal{C}$ do
17: From $\theta_k$, compute Student-$t$ predictive $t_d(z_t \mid \theta_k)$
using Eq. (4)
18: Assign log-score: $\ell_k \leftarrow \log n_k + \log t_d(z_t \mid \theta_k)$
19: end for
20: Birth-or-assign decision:
21: if $\ell_{\text{new}} &gt; \max_k \ell_k$ then
22: Spawn a new cluster $C_{\text{new}}$ with $n = 1$, mean $z_t$, scatter $S = 0$
23: Compute its NIW posterior $\theta_{\text{new}}$ from $\theta_0$
24: Append $C_{\text{new}}$ to $\mathcal{C}$, set $\hat{c}_t \leftarrow |\mathcal{C}|$
25: else
26: Let $k^\star = \arg \max_k \ell_k$ and set $\hat{c}_t \leftarrow k^\star$
27: Update $(n_{k^\star}, \bar{z}_{k^\star}, S_{k^\star})$ with a Welford step
28: Recompute $\theta_{k^\star}$ from $(n_{k^\star}, \bar{z}_{k^\star}, S_{k^\star})$
29: end if
30: end for
</div>

## K Complexity Analysis

In the main text, we analyzed the complexity of the DP-BOA head and showed that its additional cost comes from full-covariance posterior-predictive scoring and online Bayesian updates. Let d be the feature dimension and K the current number of active categories. As summarized in Appendix J, each online step for an incoming feature $\mathbf { z } _ { t }$ consists of three parts: (i) computing posterior-predictive Student-t scores for all existing categories together with the “new” hypothesis; (ii)

Table 12: Runtime of DP-BOA. Mean and max latency (ms / sample) over the full OCD stream. Numbers in parentheses denote the total number of categories (#Cls).

<table><tr><td></td><td>Animalia (77)</td><td>CUB (200)</td><td>OxfordPets (38)</td></tr><tr><td>Mean / Max latency (ms)</td><td>26.6 / 46.1</td><td>58.7 / 81.7</td><td>17.4 / 37.0</td></tr></table>

Table 13: DP-BOA memory (MB). “E2E” denotes end-to-end.

<table><tr><td rowspan="2">Method</td><td colspan="3">Animalia</td><td colspan="3">CUB</td><td colspan="3">StanfordCars</td></tr><tr><td>Head</td><td>E2E</td><td>#Clusters</td><td>Head</td><td>E2E</td><td>#Clusters</td><td>Head</td><td>E2E</td><td>#Clusters</td></tr><tr><td>DP-BOA</td><td>299.2</td><td>779.8</td><td>81</td><td>619.2</td><td>1108.8</td><td>175</td><td>626.7</td><td>1120.4</td><td>177</td></tr></table>

making the birth-or-assign decision by comparing these scores; and (iii) updating the statistics $\left( n _ { k } , \bar { \mathbf { z } } _ { k } , \mathbf { S } _ { k } \right)$ and the NIW posterior of the selected category.

With full covariance matrices, the dominant cost in step (i) is evaluating one quadratic form per category, which gives $O ( K d ^ { 2 } )$ time. In step (iii), updating the count, mean, and scatter matrix via the online Welford-style rules costs $O ( d ^ { 2 } )$ ), while recomputing the predictive covariance factors for the updated category costs $O ( d ^ { 3 } )$ through a Cholesky factorization. In practice, we cache the inverse and log-determinant of each category’s predictive covariance, so only the selected category needs to be refactorized after an update. Therefore, the per-sample complexity of the DP-BOA head is $O ( K d ^ { 2 } + d ^ { 3 } )$ time and $O ( K d ^ { 2 } )$ memory, consistent with the statement in the main text.

## K.1 Memory Footprint and Runtime of DP-BOA

To complement the theoretical complexity in Sec. 3.5, we report more detailed wall-clock latency of DP-BOA in Tab. 12. Across all benchmarks, the mean testtime cost is in the tens of milliseconds per sample on a single GPU (NVIDIA TITAN RTX), with worst-case latency still under 90 ms. CUB, which has the largest number of categories, exhibits the highest latency (58.7/81.7 ms), while Animalia and OxfordPets are noticeably faster, consistent with the $O ( d ^ { 3 } + K d ^ { 2 } )$ dependence of the head. Overall, DP-BOA trades additional computation and memory for accuracy, but the resulting overhead remains moderate and practically acceptable in the OCD setting.

We also report both head-only and end-to-end GPU memory in Tab. 13. As expected from the $O ( K d ^ { 2 } )$ storage of full-covariance statistics, the head memory grows approximately linearly with the number of active categories: 299.2 MB at 81 categories on Animalia, 619.2 MB at 175 categories on CUB, and 626.7 MB at 177 categories on StanfordCars. This corresponds to an empirical cost of roughly 3.6–3.8 MB per category. Extrapolating this trend suggests that even at $K \approx 1 0 0 0$ , the DP-BOA head would require only about 4 GB of memory.

Therefore, although it is heavier than lightweight hashing- or radius-based OCD heads, it remains feasible on modern 8–16 GB GPUs.

## K.2 A Low-Rank Variant for Large-K Settings

To further improve scalability in higher-dimensional or larger-K regimes, we also evaluate a lightweight variant, DP-BOA-L. Importantly, DP-BOA-L keeps the same Bayesian birth-or-assign framework as DP-BOA: it still compares priorweighted posterior-predictive evidence for assigning a sample to an existing category versus spawning a new one. The only change is how each category covariance is represented and updated.

Instead of storing a full $d \times d$ predictive covariance for every category, DP-BOA-L approximates it by a low-rank anisotropic term plus an isotropic residual,

$$
\pmb {\Sigma} _ {k} \approx \sigma_ {k} ^ {2} \mathbf {I} + \mathbf {U} _ {k} \mathrm{diag} (\pmb {\lambda} _ {k}) \mathbf {U} _ {k} ^ {\top},\tag{114}
$$

where $\mathbf { U } _ { k } \in \mathbb { R } ^ { d \times r }$ contains the top-r directions, $\boldsymbol { \lambda } _ { k } \in \mathbb { R } ^ { r }$ stores their corresponding spectral strengths, and $\sigma _ { k } ^ { 2 }$ captures the average residual variance outside the low-rank subspace, with $r \ll d .$ This form preserves the dominant anisotropic geometry while reducing the per-category degrees of freedom from $O ( d ^ { 2 } )$ to $O ( d r )$

In implementation, DP-BOA-L still maintains the count $n _ { k }$ and mean $\pmb { \mu } _ { k }$ for each category, but it no longer stores a full second-order matrix explicitly. Instead, each category keeps a fixed-size covariance sketch $\mathbf { B } _ { k } \in \mathbb { R } ^ { \ell \times d }$ , with $\ell = O ( r )$ , and updates it online using Frequent Directions [16]. Each new centered residual is inserted into the sketch, and a small spectral shrinkage step is applied when needed. The resulting sketch provides an approximate low-rank decomposition from which $\mathbf { U } _ { k } , \lambda _ { k } .$ , and $\sigma _ { k } ^ { 2 }$ can be recovered. With this representation, evaluating a posterior-predictive score only requires projecting $\left( \mathbf { z } _ { t } - \mu _ { k } \right)$ onto the rank-r subspace, together with a few scalar operations for the residual term, so the per-category scoring cost becomes $O ( d r )$ rather than $O ( d ^ { 2 } )$ Traversing all K candidate categories therefore costs about $O ( K d r )$ , and updating the sketch introduces an additional low-rank maintenance cost that is approximately $O ( d r ^ { 2 } )$ . Overall, the head complexity is reduced from $O ( K d ^ { 2 } + \bar { d ^ { 3 } } )$ time and $O ( K d ^ { 2 } )$ memory to approximately $O ( K d r + d r ^ { 2 } )$ time and $O ( K d r )$ memory. In our implementation, we use $r = \ell = 3 2$

## L More Hyperparameters Analysis

We briefly analyze two hyperparameters of DP-BOA: (i) the Empirical–Bayes estimate of the NIW mean-strength $\kappa _ { 0 }$ from Appendix F, and (ii) the degreesof-freedom cap $n _ { \mathrm { c a p } }$ used in the heuristic for n<sub>0</sub> (Eq. (11) in the main text).

## L.1 Efect of the Empirical–Bayes κ<sub>0</sub>

Tab. 14 compares several fixed $\kappa _ { 0 } \in \{ 1 , 1 0 ^ { - 1 } , . . . , 1 0 ^ { - 6 } \}$ with our Empirical– Bayes value $\bar { \kappa } _ { 0 } ^ { \mathrm { E B } }$ on three fine-grained benchmarks.

Table 14: Ablation of the NIW mean-strength $\kappa _ { 0 }$ on three fine-grained benchmarks. “Ours” uses the Empirical–Bayes estimate $\kappa _ { 0 } ^ { \mathrm { E B } }$ from Appendix F. The estimated values are $\kappa _ { 0 } ^ { \mathrm { E B } } \approx 5 . 6 \times 1 0 ^ { - \frac { 5 } { 3 } }$ (Animalia), $2 . 0 \times 1 0 ^ { - 2 }$ (CUB), and $4 . 4 \times 1 0 ^ { - 3 }$ (Oxford-IIIT Pet).

<table><tr><td rowspan="2"> $\kappa_0$ </td><td colspan="3">Animalia</td><td colspan="3">CUB</td><td colspan="3">Oxford-IIIT Pet</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>1</td><td>45.2</td><td>59.1</td><td>39.4</td><td>52.2</td><td>51.7</td><td>52.4</td><td>57.5</td><td>60.1</td><td>56.2</td></tr><tr><td> $10^{-1}$ </td><td>42.6</td><td>63.8</td><td>33.9</td><td>52.8</td><td>59.1</td><td>49.6</td><td>59.6</td><td>57.9</td><td>60.5</td></tr><tr><td> $10^{-2}$ </td><td>48.0</td><td>66.6</td><td>40.3</td><td>49.3</td><td>49.0</td><td>49.5</td><td>58.8</td><td>61.5</td><td>57.3</td></tr><tr><td> $10^{-3}$ </td><td>43.0</td><td>55.1</td><td>38.0</td><td>38.8</td><td>27.9</td><td>44.2</td><td>53.2</td><td>51.9</td><td>53.9</td></tr><tr><td> $10^{-4}$ </td><td>37.9</td><td>45.2</td><td>34.9</td><td>38.8</td><td>26.1</td><td>45.1</td><td>47.8</td><td>33.8</td><td>55.1</td></tr><tr><td> $10^{-5}$ </td><td>38.1</td><td>45.3</td><td>35.2</td><td>40.6</td><td>25.0</td><td>48.4</td><td>48.1</td><td>33.9</td><td>55.5</td></tr><tr><td> $10^{-6}$ </td><td>37.2</td><td>46.2</td><td>33.4</td><td>40.6</td><td>24.9</td><td>48.4</td><td>47.9</td><td>28.2</td><td>58.2</td></tr><tr><td>Ours</td><td>50.7</td><td>67.6</td><td>43.7</td><td>53.4</td><td>57.2</td><td>51.6</td><td>59.0</td><td>63.6</td><td>56.6</td></tr></table>

Across all three datasets, $\kappa _ { 0 } ^ { \mathrm { E B } }$ lies in the same order of magnitude as the best grid value and yields the strongest or near-strongest “All” accuracy, while clearly outperforming commonly used baselines such as $\kappa _ { 0 } { = } 1$ . Very large or very small $\kappa _ { 0 }$ cause noticeable degradation, confirming that the data-driven estimate provides a good default without extra tuning.

## L.2 Sensitivity of $\mathbf { \Delta } ^ { n _ { 0 } }$ to $\mathbf { \delta } n _ { \mathbf { c a p } }$

We choose $n _ { 0 }$ via

$$
n _ {0} = \min \left(\frac {1}{2} \bar {n}, n _ {\mathrm{cap}}\right), \qquad \bar {n} = \frac {1}{K _ {S}} \sum_ {k = 1} ^ {K _ {S}} n _ {k},\tag{115}
$$

where $n _ { \mathrm { c a p } }$ caps the efective degrees of freedom. This cap is mainly needed on large-scale datasets, where n¯ can be very large; without it, $n _ { 0 }$ (and hence the Student-t degrees of freedom) would grow with dataset size, making the predictive almost Gaussian and overly concentrated.

Tab. 15 studies $n _ { \mathrm { c a p } }$ on CIFAR-100 and ImageNet-100. Very small $n _ { \mathrm { c a p } } \ ( \mathrm { e . g . }$ 10) keep the predictive distributions too heavy-tailed and yield poor overall accuracy, especially on novel classes. As $n _ { \mathrm { c a p } }$ increases, performance improves substantially and is strongest around 50, with a fairly broad plateau: on CIFAR-100, the best results lie in the range 50–65, while on ImageNet-100, 45–55 already gives very similar performance. This suggests that the method is not overly sensitive to the exact choice of $n _ { \mathrm { c a p } }$ as long as it is set in a moderate range near 50. Larger values (e.g., 100 or $\bar { n } / 2 )$ further improve known-class accuracy but sharply degrade novel-class accuracy, as overly large degrees of freedom make the Student-t predictive almost Gaussian and discourage the birth of new categories. We therefore fix $n _ { \mathrm { c a p } } { = } 5 0$ in all experiments as a simple, robust choice near the center of this stable high-performing region.

Table 15: Ablation of the cap $n _ { \mathrm { c a p } }$ in $\operatorname { E q . }$ (115) on CIFAR-100 and ImageNet-100. $^ { \mathfrak { s o } } \bar { n } / 2 ^ { \mathfrak { s } }$ corresponds to using $n _ { 0 } = \bar { n } / 2$ without the cap. Here $\bar { n } / 2 \approx 2 5 0$ (CIFAR-100) and $\bar { n } / 2 { \approx } 6 3 8$ (ImageNet-100). The row marked <sup>∗</sup> is our default.

<table><tr><td rowspan="2"> $n_{\text{cap}}$ </td><td colspan="3">CIFAR-100</td><td colspan="3">ImageNet-100</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>10</td><td>48.8</td><td>65.0</td><td>16.2</td><td>29.6</td><td>57.3</td><td>15.7</td></tr><tr><td>25</td><td>58.2</td><td>74.3</td><td>26.0</td><td>28.9</td><td>61.4</td><td>12.6</td></tr><tr><td>35</td><td>59.8</td><td>74.0</td><td>31.6</td><td>31.6</td><td>70.1</td><td>12.4</td></tr><tr><td>45</td><td>60.6</td><td>74.9</td><td>31.9</td><td>33.5</td><td>74.9</td><td>12.8</td></tr><tr><td>50*</td><td>60.7</td><td>75.0</td><td>32.0</td><td>33.8</td><td>75.8</td><td>12.7</td></tr><tr><td>55</td><td>60.7</td><td>74.8</td><td>32.6</td><td>33.7</td><td>77.0</td><td>12.1</td></tr><tr><td>65</td><td>61.0</td><td>74.6</td><td>33.7</td><td>33.0</td><td>77.1</td><td>10.9</td></tr><tr><td>100</td><td>59.5</td><td>75.8</td><td>26.8</td><td>33.1</td><td>83.7</td><td>7.6</td></tr><tr><td> $\bar{n}/2$ </td><td>56.1</td><td>72.7</td><td>22.8</td><td>33.1</td><td>92.8</td><td>3.2</td></tr></table>

## M Feature Geometry Analysis

## M.1 Category Geometry Is Heteroscedastic and Anisotropic

We first examine the geometry of categories appearing in the unlabeled query stream, since these are the distributions that the online discovery head must model at test time. On CUB, we extract query features using the trained encoder and group unlabeled samples by their ground-truth categories only for post-hoc analysis. For each category $c ,$ we compute its empirical covariance $\Sigma _ { c }$ from the corresponding query features.

As shown in Fig. 6(a), the per-category scale $s ( \Sigma _ { c } ) = \mathrm { t r } ( \Sigma _ { c } ) / d$ varies substantially across query categories, indicating heteroscedasticity rather than a shared variance across categories. Fig. 6(b) further shows a pronounced right tail in the anisotropy ratio $\kappa ( \varSigma _ { c } ) = \lambda _ { \mathrm { m a x } } / \lambda _ { \mathrm { m i n } }$ , suggesting that many categories occupy elongated, direction-dependent regions rather than spherical neighborhoods. These observations motivate geometry-aware second-order modeling in the online head. In particular, they explain why fixed-shape regions such as Euclidean balls, Hamming balls, or angular caps may be insuficient to describe the category geometry encountered during online discovery.

## M.2 Gaussian Approximation to Category Geometry

We next examine whether category-conditioned features are reasonably compatible with a single full-covariance Gaussian approximation. This analysis should not be interpreted as claiming that every category is exactly Gaussian. Rather, our goal is to assess whether a category-specific mean and covariance can capture the dominant location, scale, and anisotropy of the observed feature clouds.

For visualization, we use all samples of each category and project each category to its own 2D PCA plane, following the protocol in Appendix I. A Gaussian

![](images/8f3f416d40aac7910909041099107052edc04e0471c05c7ea67d5825dd120284.jpg)  
(a) Per-category scale.

![](images/c83acdebef3c7cf3b66f6c50c7c9cd8a239c9e72769ea6698ec90311c7069a68.jpg)

$$
\kappa (\Sigma_ {c}) = \lambda_ {\max} / \lambda_ {\min}
$$

(b) Per-category anisotropy.  
Fig. 6: Online category geometry on CUB. We analyze the feature distributions of categories in the unlabeled query split. In (a), we report the per-category scale $s ( \Sigma _ { c } ) = \mathrm { t r } ( \Sigma _ { c } ) / d$ , where d is the feature dimension. In (b), we report the per-category anisotropy $\kappa ( \varSigma _ { c } ) = \lambda _ { \mathrm { m a x } } / \lambda _ { \mathrm { m i n } }$ . The distributions show substantial variation in scale and clear deviations from sphericity.  
![](images/af87f14c7c162902ebf0e3b524b1be8982f2510e432c4b707ab22bbce25b3e67.jpg)

![](images/c540a8fcac3ee1c4bd4b8e59366c672a38269e58cd057650edcff7bf8f0cabb5.jpg)

![](images/c42e46ea2372ef8ce547abdf8bc77bfbeeb0ada4370cfa7de05a9b24f9f0e026.jpg)  
Fig. 7: Category visualizations on CIFAR100. Each panel shows one category in its own 2D PCA plane. Blue points are the category features, and the solid / dashed curves denote the fitted Gaussian 1\sigma / 2\sigma contours, respectively.

is then fitted to the projected features, and its 1\sigma and 2\sigma contours are overlaid on the scatter plot.

As shown in Figs. 7 and 8, many categories exhibit a dominant central region whose spread is direction-dependent but broadly elliptical. In several examples, the fitted 1\sigma contour aligns with the densest part of the feature cloud, while the 2\sigma contour covers much of the broader support. The fit is not perfect, and some categories show non-elliptical tails or local substructure. Nevertheless, the visualizations suggest that a unimodal full-covariance Gaussian often provides a useful first-order approximation to the dominant category geometry.

Together with the heteroscedasticity and anisotropy observed above, this supports the modeling choice in DP-BOA as a practical approximation: the model does not require categories to be spherical, but instead represents each category by an adaptive full-covariance posterior predictive distribution. At the same time, the single-Gaussian assumption remains an approximation, and we analyze its behavior under a more challenging mismatch setting below.

![](images/841f1f54496ea58191a4e8466dca3d99407ee44a136ee426a250f5840abf4e4e.jpg)  
Fig. 8: Category visualizations on Oxford-IIIT Pet. Each panel shows one category in its own 2D PCA plane. Blue points are the category features, and the solid / dashed curves denote the fitted Gaussian 1\sigma / 2\sigma contours, respectively.

## M.3 Beyond Single-Gaussian Modeling

A single full-covariance Gaussian is a tractable and efective default for categoryconditioned features in our standard OCD benchmarks, but it is not exact in all regimes. In particular, under domain shift, samples from the same semantic category may occupy diferent sub-regions of the frozen feature space. This can induce a non-elliptical structure that is less well matched by a single Gaussian component. We therefore use source-target domain shift as a stress test for the single-Gaussian approximation and for the support-set prior calibration.

To probe this setting, we conduct a stress test on DomainNet [36], following the known–novel split protocol introduced in HiLo [50]. The training data contains a single source domain, while the online query stream mixes source-domain and target-domain samples. This creates a mismatch between the support-set statistics used to initialize DP-BOA and the geometry of future query samples, especially in the target domain.

The results in Tab. 16 show that DP-BOA remains competitive under this mismatch. It achieves the best All accuracy in three of the four domain slices and the best Novel accuracy in all four. This suggests that the proposed posteriorpredictive birth-or-assign rule remains efective even when category geometry is distorted by cross-domain mismatch. Importantly, DP-BOA does not require the support-set prior to perfectly represent all future novel classes. The support statistics provide an initial calibration, while the category posterior states are updated online after each decision and increasingly reflect accumulated stream evidence.

At the same time, the source-target gap remains clear, and PHE is still stronger on some target-domain Known metrics. This indicates that the current frozen-feature formulation is reasonably robust compared with existing OCD methods, but severe target-domain shift remains challenging, especially for preserving known-category assignments in the target domain.

Two natural extensions may further improve performance in this regime. The first is to enrich the category model itself. Because DP-BOA makes decisions through a modular comparison of prior-weighted predictive evidence, the current single-Gaussian predictive density can, in principle, be replaced by a richer model, such as a multi-component or hierarchical nonparametric mixture, while preserving the same birth-or-assign framework [44]. Such models may better capture domain-induced sub-modes or non-elliptical structure within a category. However, they would also enlarge the latent state per category, introduce additional components and hyperparameters, and increase time and memory complexity.

Table 16: Model performance on DomainNet under source-target domain shift. Best results in each column are in bold.

<table><tr><td rowspan="2">Method</td><td colspan="3">Real (Source)</td><td colspan="3">Clipart (Target)</td><td colspan="3">Real (Source)</td><td colspan="3">Painting (Target)</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>SMILE</td><td>18.3</td><td>18.5</td><td>18.3</td><td>8.6</td><td>11.0</td><td>6.2</td><td>19.1</td><td>23.5</td><td>18.2</td><td>13.1</td><td>16.7</td><td>9.5</td></tr><tr><td>PHE</td><td>27.6</td><td>37.9</td><td>25.7</td><td>19.1</td><td>31.6</td><td>6.3</td><td>26.1</td><td>41.0</td><td>23.3</td><td>22.4</td><td>36.3</td><td>8.2</td></tr><tr><td>DP-BOA</td><td>37.9</td><td>51.0</td><td>35.4</td><td>18.9</td><td>23.7</td><td>14.1</td><td>35.6</td><td>31.4</td><td>36.4</td><td>24.9</td><td>38.5</td><td>11.0</td></tr></table>

The second direction is to adapt the feature space at test time so that samples from diferent domains become better aligned. This could reduce the sourcetarget gap observed above. However, integrating test-time adaptation into OCD is non-trivial. The stream contains both known and novel categories, so objectives that simply encourage confident assignment may bias the model toward absorbing truly novel samples into existing categories, thereby suppressing category birth. Online backbone updates also introduce extra backpropagation cost, optimizer state, and the risk of drift or forgetting under non-stationary streams.

Overall, these results suggest that the current single-Gaussian DP-BOA is a strong and robust default under standard OCD benchmarks and remains competitive under domain shift. Future gains may come from richer category models or carefully controlled online feature adaptation, but both directions introduce additional computational and stability challenges.

## N Comparison with Simpler Probabilistic Heads

We further compare DP-BOA with simpler online probabilistic heads under identical frozen features. These variants use the same encoder and support/query split as DP-BOA, but replace the explicit DP-weighted birth hypothesis with a fixed rejection mechanism. This comparison helps isolate the contribution of the proposed birth-or-assign rule from representation quality or support-set calibration alone.

Posterior-threshold variant. The first variant uses the same posterior probability as DP-BOA for existing categories, but replaces the explicit birth hypothesis with a threshold rule. Specifically, for an incoming sample $z _ { t } ,$ the variant predicts

$$
k ^ {*} = \arg \max _ {k} P (c _ {t} = k \mid z _ {t}, \mathcal {D} _ {t - 1}),\tag{116}
$$

Table 17: Comparison with simpler probabilistic heads. Maha uses a supportcalibrated Mahalanobis metric with a fixed rejection threshold. Post.-thr. uses the same posterior probability as DP-BOA for existing categories but replaces the explicit birth hypothesis with a fixed posterior threshold. DP-BOA keeps the full DP-weighted birthor-assign comparison. All methods use identical frozen features. Best results in each column are in bold.

<table><tr><td rowspan="2">Method</td><td colspan="3">Pets</td><td colspan="3">CUB</td><td colspan="3">Animalia</td></tr><tr><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td><td>All</td><td>Known</td><td>Novel</td></tr><tr><td>Maha</td><td>45.2</td><td>44.3</td><td>45.6</td><td>36.8</td><td>45.2</td><td>32.6</td><td>44.4</td><td>73.0</td><td>32.5</td></tr><tr><td>Post.-thr.</td><td>57.8</td><td>65.0</td><td>54.0</td><td>50.8</td><td>54.9</td><td>48.8</td><td>48.6</td><td>65.6</td><td>41.5</td></tr><tr><td>DP-BOA</td><td>59.0</td><td>63.6</td><td>56.6</td><td>53.4</td><td>57.2</td><td>51.6</td><td>50.7</td><td>67.6</td><td>43.7</td></tr></table>

and spawns a new category if

$$
\max _ {k} P (c _ {t} = k \mid z _ {t}, \mathcal {D} _ {t - 1}) <   p _ {0};\tag{117}
$$

otherwise, it assigns the sample to $k ^ { * }$ . For numerical stability, the threshold p<sub>0</sub> is tuned in log-space on a small held-out validation split carved out from the training data. All other settings, including the encoder and online posterior updates, are kept unchanged.

Mahalanobis-threshold variant. The second variant uses a support-calibrated Mahalanobis metric with a fixed rejection threshold. Let $\scriptstyle \sum _ { \mathrm { w i t h i n } }$ denote the pooled within-class covariance estimated from the labeled support set. For each existing category $k ,$ we compute

$$
d _ {k} ^ {\mathrm{Maha}} (z _ {t}) = (z _ {t} - \bar {z} _ {k}) ^ {\top} \Sigma_ {\mathrm{within}} ^ {- 1} (z _ {t} - \bar {z} _ {k}),\tag{118}
$$

and select the nearest category

$$
k ^ {*} = \arg \min _ {k} d _ {k} ^ {\text { Maha }} (z _ {t}).\tag{119}
$$

The sample is assigned to $k ^ { * }$ if min $\cdot d _ { k } ^ { \mathrm { M a h a } } ( z _ { t } ) \leq \tau _ { \mathrm { M a h a } }$ , and otherwise spawns a new category. The threshold $\tau _ { \mathrm { M a h a } }$ is tuned on the same held-out validation split. This variant uses the same support-set statistics for calibration, but does not maintain category-specific posterior uncertainty or compare assignment with an explicit DP-induced birth score.

The results are reported in Tab. 17. The Mahalanobis-threshold variant can be competitive on known-class accuracy, but its novel-class performance is substantially weaker, suggesting that a fixed global rejection radius is insuficient for reliable online discovery. The posterior-threshold variant improves over the Mahalanobis head by using NIW posterior predictives, but still underperforms DP-BOA. Overall, DP-BOA achieves the best All and Novel accuracy on all three datasets, showing that its advantage does not come merely from posterior scoring or support-set covariance calibration. Instead, explicitly comparing assignment evidence with DP-weighted birth evidence provides a more calibrated online decision rule.

## O Temporal Diagnostics

We further analyze the temporal behavior of DP-BOA along the query stream. The goal is to examine whether the online posterior state becomes more stable as evidence accumulates, and whether the proposed birth-or-assign rule reduces unnecessary category fragmentation. We partition the query stream into five equal temporal bins, corresponding to 0.2T , 0.4T , 0.6T , 0.8T , and T , where T denotes the stream length.

False-birth rate. A birth decision is counted as false if the model spawns a new category for a sample whose ground-truth class has already been represented by an existing category in the current stream state. This includes known-class samples, whose categories are initialized from the labeled support set, and novelclass samples whose ground-truth class has appeared earlier in the query stream. For a temporal bin B, we compute

\mathr {FBR}(cl ) = f 1|  su \_ in   b !e [   w ;dg  y x    p o     ]  0%.  q: $\langle B \rangle = { \frac { 1 } { | B | } } \sum _ { t \in B } \mathbf { 1 } [ { \hat { c } } _ { t } = \operatorname { n e w } \ \wedge \ y _ { t }$ has been represented before time $t ] \times 1 0 0 \%$

(120)

A lower false-birth rate indicates that the method is less likely to over-fragment existing categories as the stream progresses. As shown in Fig. 9a, DP-BOA maintains a much lower false-birth rate than PHE across all temporal bins on Animalia. Specifically, the false-birth rate of DP-BOA remains below 1.3% throughout the stream and decreases to 0.5% at the end, whereas PHE stays substantially higher, decreasing from 8.9% to 4.4%. This suggests that the explicit birth hypothesis in DP-BOA does not lead to excessive category creation and is consistent with reduced category fragmentation.

Cluster-mean drift. To quantify how much the posterior state changes after online updates, we measure the drift of the selected category mean. Let $\mu _ { \hat { c } _ { t } } ^ { ( t - 1 ) }$ and $\mu _ { \hat { c } _ { t } } ^ { ( t ) }$ denote the posterior mean of the selected category before and after processing $z _ { t } ,$ respectively. We define

$$
\varDelta \mu_ {t} = \left\| \mu_ {\hat {c} _ {t}} ^ {(t)} - \mu_ {\hat {c} _ {t}} ^ {(t - 1)} \right\| _ {2}.\tag{121}
$$

We report the average $\varDelta \mu _ { t }$ in each temporal bin, normalized by the value in the first bin for readability. A decreasing trend indicates that category posteriors become more stable after early stream adaptation. As shown in Fig. 9b, the normalized posterior cluster-mean drift of DP-BOA first increases slightly from 1.0 to 1.2, remains close to this level at 0.6T with a value of 1.2, and then decreases to 0.9 and 0.6 in the later bins. This pattern suggests that the posterior state undergoes an initial adaptation phase and then becomes progressively more stable as additional evidence is accumulated.

Evidence margin. To examine decision ambiguity along the stream, we measure the evidence margin for each incoming sample before the online update.

![](images/6be0d79bf14eee472c3f8f5012c210a87f88c04640537e2c792ce92ad57e3a32.jpg)  
(a) False-birth rate.

![](images/7a767bc9a7dca7db1451363ffbfb715d524c2a6b12da6e17dfaf2693182fee17.jpg)  
(b) Cluster-mean drift.

![](images/5b072b4c7baea5f6c3294f5fc1bc16f1e86bcee0db55961954c230d5c22c3f6f.jpg)  
(c) Evidence margin.  
Fig. 9: Temporal diagnostics on Animalia. From left to right, we show the falsebirth rate, the normalized posterior cluster-mean drift, and the normalized evidence margin over five temporal bins along the query stream.

Specifically, we compute the log evidence of all existing-category assignment hypotheses together with the explicit birth hypothesis, and define the margin as the gap between the largest and second-largest log evidence:

$$
m _ {t} = \log p _ {t} ^ {(1)} - \log p _ {t} ^ {(2)}.
$$

We report the average m in each temporal bin, normalized by the value in the first bin for readability. A larger margin means that the selected birth-or-assign hypothesis is more clearly separated from its closest competitor, while a smaller margin indicates stronger ambiguity among competing hypotheses. As shown in Fig. 9c, the normalized evidence margin of DP-BOA decreases from 1.0 to 0.7 and then to 0.4 in the middle of the stream, before partially recovering to 0.5 and 0.6 in the later bins. This non-monotonic trend is reasonable in the online discovery setting, because two competing efects act simultaneously: as more samples are observed, posterior statistics become better adapted and can sharpen decisions; at the same time, newly created categories enlarge the hypothesis set and intensify competition among nearby categories.

(122)