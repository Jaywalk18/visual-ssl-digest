# Understanding Self-Supervised Learning via Latent Distribution Matching

# Fabian A Mikulasch 1 Friedemann Zenke 1 2

# Abstract

Self-supervised learning (SSL) excels at finding general-purpose latent representations from complex data, yet lacks a unifying theoretical framework that explains the diverse existing methods and guides the design of new ones. We cast SSL as latent distribution matching (LDM): learning representations that maximize their logprobability under an assumed latent model (alignment), while maximizing latent entropy to prevent collapse (uniformity). This view unifies independent component analysis with contrastive, noncontrastive, and predictive SSL methods, including stop gradient (stopgrad) approaches. Leveraging LDM, we derive a nonlinear, sampling-free Bayesian filtering model with a Kalman-based predictor for high-dimensional timeseries. We further prove that predictive LDM yields identifiable latent representations under mild assumptions, even with nonlinear predictors. Overall, LDM clarifies the assumptions behind established SSL methods and provides principled guidance for developing new approaches.

# 1. Introduction

Self-supervised learning (SSL) has become a central paradigm for representation learning, enabling models to extract useful structure from unlabeled data across vision, language, and audio domains (Noroozi & Favaro, 2016; Oord et al., 2018; Dawid & LeCun, 2024; Gui et al., 2024). By learning from relationships between multiple views of the same data, such as augmentations, temporal neighbors, or masked context, modern SSL methods often provide more useful representations for downstream tasks than traditional likelihood-based approaches like autoencoders, which emphasize low-level reconstruction and can overfit to superficial details.

1Friedrich Miescher Institute for Biomedical Research, 4056 Basel, Switzerland 2Faculty of Science, University of Basel, 4033 Basel, Switzerland. Correspondence to: Fabian A Mikulasch <fabian.mikulasch+ldm@fmi.ch>, Friedemann Zenke <friedemann.zenke+ldm@fmi.ch>.

Preprint. May 28, 2026.

![](images/c5fee89f4b824d202eb88b640d7d7cc4e5b96fadaf8d98906998ccdb1a6ee946.jpg)

<details>
<summary>text_image</summary>

R(z,z') = \langle R(z|x)R(z'|x')\rangle_{Pdata(x,x')} \nR(z|x) \nR(z'|x') \nPdata(x^-)
</details>

$$
\begin{array}{l} \mathcal {F} = - D _ {K L} [ R (z, z ^ {\prime}) \parallel P _ {\theta} (z, z ^ {\prime}) ] \\ = \left\langle \log P _ {\theta} (z ^ {\prime} | z) + \log P _ {\theta} (z) \right\rangle_ {R (z, z ^ {\prime})} + H _ {R} [ z, z ^ {\prime} ] \\ \end{array}
$$

Figure 1. We formulate SSL as a distribution matching problem in which the transformed data distribution $R ( z , z ^ { \prime } )$ is matched to the latent model $P _ { \theta } \left( z , z ^ { \prime } \right)$ . The transformation is deterministic $R ( z | x ) = \delta ( z - f ( x ) )$ ), where $f ( x )$ is a deep network. The model likelihood log $P _ { \theta }$ and latent entropy $H _ { R }$ correspond to alignment and uniformity terms in the loss function (Wang & Isola, 2020).

Among SSL approaches, latent predictive models have recently achieved state-of-the-art performance on temporal and sequential data (Oord et al., 2018; Bardes et al., 2024; Assran et al., 2025). Here, instead of predicting upcoming data points as in temporal generative models, the goal is to predict future latent representations, while preventing representational collapse. Despite the empirical success of predictive SSL, these models typically rely on heuristic objectives or regularization strategies, while our understanding of why they work, and how to design them systematically, remains incomplete (Reizinger et al., 2025).

A common way to interpret SSL is through a geometric alignment perspective (Wang & Isola, 2020): representations of paired views are encouraged to be close in latent space, while additional “uniformity” or repulsion terms in the loss prevent representational collapse. Although this view offers intuition and has influenced practical designs, it does not constitute a formal statistical foundation for SSL. In particular, it provides limited guidance for models that do not explicitly rely on contrastive or repulsive terms, such as BYOL (Grill et al., 2020) or SimSiam (Chen et al., 2020).

Another influential line of work frames SSL as mutual information (MI) maximization between paired views to preserve shared information while discarding noise (Oord et al.,

2018; Tian et al., 2020; Shwartz-Ziv et al., 2023; Galvez´ et al., 2023; Shwartz Ziv & LeCun, 2024). While appealing, MI-based interpretations face a fundamental limitation: MI is invariant under arbitrary invertible transformations ϕ and ψ, i.e.,

$$
I [ x, y ] = I [ \phi (x), \psi (y) ] \tag {1}
$$

and, therefore, does not by itself promote semantically meaningful or identifiable representations. Empirically, MI maximization has proven neither necessary nor sufficient for successful SSL (Tschannen et al., 2020), leaving its precise role elusive.

In contrast, formal guarantees for successful representations in SSL stem from the latent distribution matching (LDM) principle (Zimmermann et al., 2021; Khemakhem et al., 2020b). If a learned latent distribution matches the distribution of the “true” underlying latent variables, then—under assumptions that depend on the model—the representations can recover these variables up to trivial equivalences, such as permutations or affine transformations.

Motivated by these results, we ask whether LDM can serve as a unifying objective for SSL rather than an implicit or auxiliary principle. Our main contributions are:

A unified distribution-matching framework for SSL. We formulate SSL as matching a data distribution to an assumed latent model (Fig. 1) and show how a wide range of existing SSL algorithms can be recovered as instances of LDM (Table 1), with differences between methods arising naturally from distinct choices of latent models or entropy estimators.

Clarifying the role of MI maximization. We show that in LDM, MI maximization contributes little to representation quality beyond entropy maximization because it is invariant to arbitrary invertible transformations. Instead, approximate MI maximization implicitly performs distribution matching.

Derivation of new SSL objectives. We show how LDM naturally leads to new SSL variants with beneficial properties. As an example, we derive a predictive SSL model with Kalman-based latent dynamics combined with a deep encoder, thereby enabling principled uncertainty quantification over latent states.

Identifiability guarantees for predictive SSL. We provide identifiability results for predictive SSL models trained via LDM, showing that they can recover underlying latent variables up to affine transformations, thereby providing a formal explanation for their strong empirical performance.

# 2. Prior work

We organize the prior work according to theoretical approaches for understanding SSL, notable identifiability results on SSL, and previous use of LDM in SSL.

Theoretical perspective on SSL. There has been significant interest in understanding SSL through theoretical analysis (e.g., Saunshi et al., 2019; Wang & Isola, 2020; Ben-Shaul et al., 2023). A line of work closely related to this article aims to connect contrastive SSL (e.g., CPC) to latent variable models (Von Kugelgen et al. ¨ , 2021; Zimmermann et al., 2021; Aitchison & Ganev, 2024; Nakamura et al., 2023; Bizeul et al., 2024). Others proposed information-theoretic interpretations of regularization-based SSL (e.g., VICReg) (Shwartz-Ziv et al., 2023) and clustering/contrastive SSL (Galvez et al. ´ , 2023). Finally, stopgrad-based SSL was analyzed using several approaches (Tian et al., 2021; Halvagal et al., 2023; Nakamura et al., 2023), yet a unified treatment is lacking.

Identifiability in SSL and ICA. Identifiability is a core concept in linear independent component analysis (ICA) (Hyvarinen & Oja ¨ , 1999), which guarantees the recovery of ”true” underlying variables up to trivial transformations. The concept was later extended to nonlinear ICA using assumptions such as temporal structure or auxiliary variables (Sprekeler et al., 2014; Khemakhem et al., 2020a; Hyvarinen et al., 2019; Roeder et al., 2021). More recently, these insights also enabled proving identifiability for SSL models (Von Kugelgen et al. ¨ , 2021; Zimmermann et al., 2021; Daunhawer et al., 2023; Reizinger et al., 2024; Laiz et al., 2025). However, most of these results were derived for specific algorithms, emphasizing the need for a more general framework.

Distribution matching approaches in SSL. Zimmermann et al. (2021) implicitly showed that contrastive SSL can be understood as conditional LDM via the reverse KL-Divergence (Appendix A.2.1). Other LDM approaches to SSL have been proposed from the perspective of optimal transport (Chen et al., 2024; Jiao et al., 2025) or rate reduction (Ma et al., 2022). Balestriero & LeCun (2025) proposed single-variable LDM based on an isotropic Gaussian as basis for SSL. In this nonlinear setting, single-variable LDM does not provide identifiability guarantees and must be augmented with additional loss terms. Still, how one can use LDM to design identifiable SSL algorithms has not been explored.

# 3. Distribution matching framework

To motivate the latent distribution matching (LDM) framework, we first revisit the maximum likelihood objective, which is also used, e.g., in linear ICA (Hyvarinen & Oja ¨ , 1999) or normalizing flow networks (Papamakarios et al., 2021). In these approaches, models often compute the datalikelihood in latent, instead of data space, which is possible for an invertible data generation process $g \equiv f ^ { - 1 }$ (see Appendix A.4.1 and Papamakarios et al., 2021; Cardoso, 2002, see Appendix A.1 for a summary of our notation)

A   
Linear ICA as distribution matching   
![](images/f064ad882a7d342ee5502fbc4a5802473f31e243ca2f88bbf145d338c5929436.jpg)

<details>
<summary>text_image</summary>

Pdata(x1)
x2
x1
Pdata(x)
W
R(z1)
Pθ(z1)
z2
z1
R(z)
F = -DKL[R(z) || Pθ(z)]
= ⟨log Pθ(z)⟩R(z) + HR[z]
</details>

B   
![](images/a0a482d3965551239033f255864de3b58dd56299f537d1005ba82186f2388ec3.jpg)

<details>
<summary>histogram</summary>

| Image Type | Pixel Intensity Range | Probability Density |
|------------|------------------------|---------------------|
| Original   | -2.5 to 0.0            | Low                 |
| Mixed      | -2.5 to 0.0            | High                |
| Rec        | -2.5 to 0.0            | Medium              |
| Gen. Gauss.| -2.5 to 0.0           | High                |
</details>

C   
![](images/af3cb4f466d273cc153187895c7dd5133a427adc3ad737ecd17893ac48514c0f.jpg)

<details>
<summary>line</summary>

| Time | OU proc. | mixed proc. | recovered proc. |
|------|----------|-------------|------------------|
| 0    | 0        | 0           | 0                |
| 50   | 0        | 0           | 0                |
| 100  | 0        | 0           | 0                |
</details>

Figure 2. Source recovery with LDM in linear ICA. A Linear ICA assumes that the data distribution has independent factors, that can be recovered by aligning them with the correct underlying independent distribution (Cardoso, 2002). B Distributions of pixel intensities in natural images are non-Gaussian (Hyvarinen & Oja ¨ , 1999). In contrast, mixed images are closer to Gaussian, as expected from the central limit theorem. Disentanglement proceeds by learning W to recover an assumed short-tailed distribution (red dashed line). C Also Gaussian sources can be disentangled, which, however, requires more assumptions on the data generating process. Here we recover the outputs of two Ornstein-Uhlenbeck processes assuming known variances and that W is volume preserving (determinant of $| W | = 1 )$ .

$$
\begin{array}{l} \left\langle \log P _ {\theta} (x) \right\rangle_ {P _ {\text { data }} (x)} \propto \underbrace {\left\langle \log P _ {\theta} (f (x)) \right\rangle_ {P _ {\text { data }} (x)}} _ {\text { Likelihood }} + \underbrace {H _ {P _ {\text { data }}} [ f (x) ]} _ {\text { Entropy }} \\ = - D _ {K L} \left[ P _ {\text { data }} (f (x)) \| P _ {\theta} (f (x)) \right], \tag {2} \\ \end{array}
$$

where $P _ { \theta }$ is the model distribution with parameters θ. Importantly, this does not require the encoder f to be invertible in general but only on the data manifold (Appendix A.4.1), which is the same assumption underlying the identifiability results for SSL (Section 7). Thus, for an encoder that is invertible on the data manifold, distribution matching in latent space is equivalent to maximum likelihood learning.

# 3.1. Linear ICA as latent distribution matching

To build intuition, we briefly review how linear ICA can be understood as LDM (see Choi et al., 2005; Hyvarinen & ¨ Oja, 1999, for more thorough reviews on ICA). Several linear ICA variants, including InfoMax and likelihood-based methods, assume that the data can be disentangled with an unmixing matrix W to match an assumed latent distribution $\begin{array} { r } { P _ { \theta } ( z ) = \prod _ { i } P _ { \theta } ( z _ { i } ) } \end{array}$ that factorizes over latent factors (Hyvarinen & Oja ¨ , 1999; Cardoso, 2002). This unmixing corresponds to an LDM goal (Fig. 2A). If the data is linearly transformed as $z = W x$ , the transformed data distribution is $R ( z ) = \langle \delta ( z - W x ) \rangle _ { P _ { \mathrm { d a t a } } ( x ) }$ , which is matched to $P _ { \theta } ( z )$ via (Cardoso, 2002)

$$
\begin{array}{l} \mathcal {F} _ {\mathrm{ICA}} = - D _ {K L} [ R (z) \parallel P _ {\theta} (z) ] = \langle \log P _ {\theta} (z) \rangle_ {R (z)} + H _ {R} [ z ] \\ = \left\langle \log P _ {\theta} (z) \right\rangle_ {R (z)} + \frac {1}{2} \log | W W ^ {T} | + H _ {P _ {\text { data }}} [ x ], \tag {3} \\ \end{array}
$$

where we used the formula for the change of variables in the entropy. Since the data entropy term is constant it can be omitted, and maximizing the likelihood and log-determinant terms leads to disentangled factors (Fig. 2).

The assumption of known source distributions might seem limiting. However, on the one hand, the assumed latent distribution need not match precisely for successful recovery; it is typically sufficient if the general shape matches (e.g., subor super-gaussian, Hyvarinen & Oja ¨ , 1999). On the other hand, since we have a well-defined log-likelihood function, we can optimize the distribution parameters θ to better fit the data (Hyvarinen & Oja ¨ , 1999). This last insight will be especially relevant to understand predictive SSL (Section 6).

# 4. SSL from latent distribution matching

The central limitation of single-variable ICA, as formulated above, is that, for a nonlinear unmixing function $z = f ( x )$ , the problem becomes ill-defined (Hyvarinen et al. ¨ , 2023). However, it is possible to formulate a well-defined source recovery problem by specifying additional conditional dependencies in the latent variable distribution (Sprekeler et al., 2014; Hyvarinen et al., 2019; Zimmermann et al., 2021; Khemakhem et al., 2020a; Hyvarinen et al.¨ , 2023). The basic intuition is that these additional conditional constraints in latent space provide enough information to determine a unique solution to the nonlinear unmixing problem, i.e., up to trivial transformations.

Inspired by this we define a joint distribution matching goal (cf. Fig. 1; general formulation in Appendix A.3). We first define the reshaped latent distribution via a recognition density $R ( z | x )$ and the data distribution

$$
R (z, z ^ {\prime}) := \left\langle R (z | x) R (z ^ {\prime} | x ^ {\prime}) \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})}. \tag {4}
$$

In the following, we assume a deterministic recognition function $R ( z | x ) = \delta ( z - f ( x ) )$ ), where $f ( x )$ is parametrized by a deep neural network. We explore nondeterministic encoders in Appendix A.8. With these definitions, we can formulate the LDM SSL goal function as follows

$$
\begin{array}{l} \mathcal {F} _ {\mathrm{LDM}} = - D _ {K L} \left[ R \left(z, z ^ {\prime}\right) \| P _ {\theta} \left(z, z ^ {\prime}\right) \right] \\ = \underbrace {\left\langle \log P _ {\theta} (z , z ^ {\prime}) \right\rangle_ {R (z , z ^ {\prime})}} _ {\text { Alignment }} + \underbrace {H _ {R} [ z , z ^ {\prime} ]} _ {\text { Uniformity }}. \tag {5} \\ \end{array}
$$

Two observations are worth highlighting. First, the likelihood term promotes alignment, whereas the entropy term rewards uniformity (cf. Fig. 1), as discussed previously by Wang & Isola (2020). Importantly, the uniformity term encourages invertibility of the encoder on the data manifold, by preventing different data points from collapsing to the same point in latent space (Appendix A.4.2). Second, a well-known property of the KL divergence is that it is nonnegative, and zero only if the distributions match. Hence, if $\mathcal { F } _ { \mathrm { L D M } }$ is maximized and R and $P _ { \theta }$ have the same support, $R ( z , z ^ { \prime } ) = P _ { \theta } ( z , z ^ { \prime } )$ , and in particular $R ( z | z ^ { \prime } ) = P _ { \theta } ( z | z ^ { \prime } )$ . As outlined above, this conditional LDM has been shown to recover the underlying latent variables up to trivial transformations when the data-generating process is invertible (Hyvarinen et al., 2019; Zimmermann et al., 2021; Roeder et al., 2021). We will use this property for proving the theorems in Section 7.

# 4.1. Distribution matching and MI maximization

LDM and MI maximization are closely related, which can be shown through a goal function previously proposed by Aitchison & Ganev (2024). Specifically, based on variational inference principles, they arrived at

$$
\begin{array}{l} \mathcal {F} _ {\mathrm{MI}} = - D _ {K L} \left[ R \left(z, z ^ {\prime}\right) \| P _ {\theta} \left(z, z ^ {\prime}\right) \right] + I _ {R} \left[ z, z ^ {\prime} \right] \tag {6} \\ = \left\langle \log P _ {\theta} (z, z ^ {\prime}) \right\rangle_ {R (z, z ^ {\prime})} + 2 H _ {R} [ z ], \\ \end{array}
$$

which differs from $\mathcal { F } _ { \mathrm { L D M } }$ only through the MI term. For the second line we assumed that $R ( z )$ equals $R ( z ^ { \prime } )$ and thus $H _ { R } [ z ] = H _ { R } [ z ^ { \prime } ]$ to simplify notation. This expression constitutes a well-known variational bound on the MI of z and $z ^ { \prime }$ (Barber & Agakov, 2004; Poole et al., 2019, Appendix A.2.3). To see this property, note that Equation (6) equals $I _ { R } [ z , z ^ { \prime } ]$ if $P _ { \theta } \left( z , z ^ { \prime } \right)$ is learned to match $R ( z , z ^ { \prime } )$ . 1 The SSL formulation by Aitchison et al. is therefore more closely related to many previous SSL approaches, which are often motivated through MI maximization (Shwartz-Ziv et al., 2023; Oord et al., 2018).

The key link between $\mathcal { F } _ { \mathrm { L D M } }$ and $\mathcal { F } _ { \mathrm { M I } }$ is that, for an encoder that is invertible on the data manifold, MI is already maximized (Appendix A.4.3). Because entropy maximization encourages invertibility, we expect the additional MI maximization in Equation (6) to contribute little to the resulting latent representation. In the following, we show how popular SSL algorithms can be derived from $\mathcal { F } _ { \mathrm { M I } }$ and then demonstrate in simulations that they yield virtually identical representations to those of matching SSL algorithms derived from $\mathcal { F } _ { \mathrm { L D M } }$ . To this end, we first discuss briefly how different entropy estimators relate to different SSL approaches.

# 4.2. Entropy estimation

While for categorical latent distributions it is sometimes possible to compute the entropy directly, for continuous variables it has to be approximated.2 This problem is challenging and has no unique best solution (Paninski, 2003). However, empirically, we find that relatively basic estimators yield good results as long as they are robust and efficiently optimized.

Depending on the approximation, it is possible to recover contrastive, non-contrastive, or other SSL approaches (for details, see Appendix A.5). Entropy estimation based on kernel density estimation (KDE) leads to contrastive SSL such as SimCLR. In contrast, parametric estimation can be related to non-contrastive SSL such as VICReg. We further show that conditional entropy estimation with a predictor is implicitly implemented by SSL models with a predictor and stop gradient (stopgrad) (Section 6). In addition, we propose that entropy estimation for SSL can be implemented through a K-nearest neighbors (kNN) estimator, which is closely related to the contrastive approach.

# 5. Contrastive and non-contrastive learning

VICReg is a popular non-contrastive SSL approach that leverages variance and covariance regularization of latent representations (Bardes et al., 2021; Halvagal & Zenke, 2023). To derive a similar algorithm we make the choices

$$
\begin{array}{c} P _ {\theta} (z) = \text {Flat}, \\ P _ {0} (z) = A f (z, \dots , F _ {0} - ^ {2} z). \end{array} \tag {7}
$$

$$
P _ {\theta} (z ^ {\prime} | z) = \mathcal {N} (z ^ {\prime}; \mu = z, \Sigma = \sigma^ {2} I).
$$

This model assumes that two related observations z and $z ^ { \prime }$ are much closer than expected under the improper flat prior. The goal function of VICReg can be derived from $\mathcal { F } _ { \mathrm { M I } }$ by employing the parametric entropy estimator under a normal distribution (Appendix A.6.1)

Table 1. Summary of model mapping to the LDM framework. MI max. denotes whether $\mathcal { F } _ { \mathrm { L D M } }$ (◦) or $\mathcal { F } _ { \mathrm { M I } }$ (•) is employed. 

<table><tr><td>SSL Method</td><td>Latent distribution  $P_{\theta}$  (prior + cond.)</td><td>Entropy estimator</td><td>MI max.</td></tr><tr><td>VICReg (Bardes et al., 2021)</td><td>flat + cond. Gaussian</td><td>parametric (Gaussian entropy)</td><td>●</td></tr><tr><td>SimCLR (Chen et al., 2020)</td><td>uniform + cond. von Mises-Fisher</td><td>kernel density</td><td>●</td></tr><tr><td>CPC (Oord et al., 2018)</td><td>empirical + predictive cond. von Mises-Fisher</td><td>see Aitchison &amp; Ganev (2024)</td><td>●</td></tr><tr><td>BYOL/SimSiam (Grill et al., 2020; Chen &amp; He, 2021)</td><td>empirical + cond. von Mises-Fisher</td><td>conditional entropy plugin</td><td>○</td></tr><tr><td>JEPA (e.g., Bardes et al., 2024; Mohammadi et al., 2025)</td><td>predictive cond. Gaussian</td><td>conditional entropy plugin</td><td>○</td></tr></table>

$$
\mathcal {F} _ {\mathrm{MI}} = - \frac {1}{2 \sigma^ {2}} \left\langle \| f (x) - f (x ^ {\prime}) \| ^ {2} \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})} + \log | \Sigma_ {z} |, \tag {8}
$$

where $\Sigma _ { z }$ is the covariance matrix of $z ,$ and $\sigma ^ { 2 }$ trades off between the push and pull terms. As discussed by Shwartz-Ziv et al. (2023), the variance-covariance regularization of VICReg can be understood to approximately maximize the single-variable covariance log-determinant log $\left| \Sigma _ { z } \right|$ . This relation can be seen by performing a Taylor expansion of the log-determinant log |Σz | ≈ Pi log Σii − 12 Pj̸=i Σij ΣjiΣiiΣjj , $\begin{array} { r } { \left| \dot { \sum _ { z } } \right| \approx \sum _ { i } \bar { \log \sum _ { i i } } - \frac { 1 } { 2 } \dot { \sum } _ { j \neq i } \frac { \Sigma _ { i j } \Sigma _ { j i } } { \Sigma _ { i i } \Sigma _ { j i } } } \end{array}$ which is similar to the regularizer in VICReg. Thus, $\scriptstyle { \dot { \mathrm { V I } } } -$ CReg can be understood as LDM to a conditional normal distribution, with additional MI maximization between latent representations.

If, instead, we employ the proposed goal function $\mathcal { F } _ { \mathrm { L D M } }$ , the following goal function can be derived:

$$
\begin{array}{l} \mathcal {F} _ {\mathrm{LDM}} = - \frac {1}{2 \sigma^ {2}} \left\langle \| f (x) - f (x ^ {\prime}) \| ^ {2} \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})} \tag {9} \\ + \frac {1}{2} \log | \Sigma_ {(z, z ^ {\prime})} |, \\ \end{array}
$$

where $\Sigma _ { \left( z , z ^ { \prime } \right) }$ is the covariance matrix of the concatenated vector $( z , z ^ { \prime } )$ .

SimCLR is a simple contrastive learning algorithm, in which latents z are lying on the unit sphere (Chen et al., 2020). To derive it, we assume

$$
\begin{array}{l} P _ {\theta} (z) = \frac {1}{Z}, \\ P _ {\theta} (z ^ {\prime} | z) = \frac {1}{Z} \exp (\beta z ^ {T} z ^ {\prime}) . \tag {10} \\ \end{array}
$$

Here, Z is the normalization from integrating over the sphere, and related latents are assumed to be locally close based on the von Mises-Fisher (vMF) distribution, i.e., the spherical normal distribution. If the entropy is approximated via KDE with bandwidth $1 / \beta$ then based on $\mathcal { F } _ { \mathrm { M I } }$ we recover the original goal function (Appendix A.6.1)

$$
\begin{array}{l} \mathcal {F} _ {\mathrm{MI}} = \left\langle \beta f (x) ^ {T} f (x ^ {\prime}) \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})} - \\ 2 \left\langle \log \left\langle \exp \left\{\beta f (x) ^ {T} f \left(x ^ {-}\right) \right\} \right\rangle_ {P _ {\text { data }} \left(x ^ {-}\right)} \right\rangle_ {P _ {\text { data }} (x)}. \tag {11} \\ \end{array}
$$

Thus, SimCLR can be understood as LDM to a conditional vMF distribution with added MI maximization.

# 5.1. Empirical comparison

To compare both SSL formulations via $\mathcal { F } _ { \mathrm { L D M } }$ and $\mathcal { F } _ { \mathrm { M I } }$ , we tested how MI maximization and the choice of entropy estimator affect representation learning on natural image datasets. We implemented SSL algorithms based on all combinations of assumed latent space (sphere or plane), entropy estimators (KDE, kNN, parametric with LogDet), and whether MI is maximized $( \mathcal { F } _ { \mathrm { M I } } )$ or not $( \mathcal { F } _ { \mathrm { L D M } } )$ . Note that although LogDet technically is not a valid entropy estimator for spherical distributions, we included it here for completeness. We then assessed representational quality using linear probing (Table 2). We found no systematic performance difference between models with and without MI maximization, as also reflected in network gradients $( \mathsf { A p - }$ pendix Fig. A2). Instead, the combination of latent space and entropy estimator had a clear and systematic impact on validation performance. Also, when analyzing the dimensionality of latent representations, we found no apparent difference between $\mathcal { F } _ { \mathrm { M I } }$ and $\mathcal { F } _ { \mathrm { L D M } }$ (Fig. 3). Instead, the assumed latent space, and especially the entropy estimator, had a more pronounced effect.

Table 2. Validation accuracy of distinct SSL variants for linear probing on different datasets. We used a standard SSL image pretraining setup as described by da Costa et al. (2022). MI denotes whether $\hat { \mathcal { F } } _ { \mathrm { I } }$ DM (◦) or FMI (•) was used. Plane-LogDet-• corresponds to VICReg, Sphere-Contr.-• corresponds to SimCLR. SimCLR can also be derived via the reverse KL divergence (Zimmermann et al., 2021), potentially explaining the difference in performance. See Table A1 for extended results, including SVHN. 

<table><tr><td rowspan="2">Space</td><td colspan="2">Model</td><td rowspan="2">CIFAR-10Top-1 Acc.</td><td rowspan="2">CIFAR-100Top-1 Acc.</td><td rowspan="2">Imagenet-100Top-1 Acc.</td></tr><tr><td>Entropy est.</td><td>MI</td></tr><tr><td>Plane</td><td>Contr.</td><td>○</td><td> $91.9 \pm 0.1$ </td><td> $65.3 \pm 0.3$ </td><td>72.6</td></tr><tr><td>Plane</td><td>Contr.</td><td>●</td><td> $92.1 \pm 0.1$ </td><td> $65.3 \pm 0.2$ </td><td>72.7</td></tr><tr><td>Plane</td><td>kNN</td><td>○</td><td> $92.1 \pm 0.2$ </td><td> $65.6 \pm 0.4$ </td><td>74.3</td></tr><tr><td>Plane</td><td>kNN</td><td>●</td><td> $91.9 \pm 0.1$ </td><td> $65.8 \pm 0.4$ </td><td>73.7</td></tr><tr><td>Plane</td><td>LogDet</td><td>○</td><td> $92.1 \pm 0.1$ </td><td> $69.5 \pm 0.2$ </td><td>75.9</td></tr><tr><td>Plane</td><td>LogDet</td><td>●</td><td> $91.9 \pm 0.1$ </td><td> $68.6 \pm 0.1$ </td><td>74.7</td></tr><tr><td>Sphere</td><td>Contr.</td><td>○</td><td> $90.9 \pm 0.1$ </td><td> $64.8 \pm 0.2$ </td><td>72.1</td></tr><tr><td>Sphere</td><td>Contr.</td><td>●</td><td> $91.4 \pm 0.2$ </td><td> $66.0 \pm 0.2$ </td><td>73.1</td></tr><tr><td>Sphere</td><td>kNN</td><td>○</td><td> $90.2 \pm 0.0$ </td><td> $64.3 \pm 0.1$ </td><td>72.6</td></tr><tr><td>Sphere</td><td>kNN</td><td>●</td><td> $90.0 \pm 0.2$ </td><td> $64.5 \pm 0.2$ </td><td>73.3</td></tr><tr><td>Sphere</td><td>LogDet</td><td>○</td><td> $91.4 \pm 0.2$ </td><td> $65.4 \pm 0.2$ </td><td>73.0</td></tr><tr><td>Sphere</td><td>LogDet</td><td>●</td><td> $91.2 \pm 0.1$ </td><td> $65.4 \pm 0.2$ </td><td>72.3</td></tr></table>

# 6. Predictive learning of dynamical systems

An appealing feature of the LDM framework is that it also allows us to treat temporal models. To this end, we first generalize the goal function to the temporal case

A   
![](images/3f3f2bcd63420ded04fd4d3b8ae76d118de1052ee20234af24f8b2a96c8a7f35.jpg)

<details>
<summary>line</summary>

| Index | Plane Contr. | Plane Contr. + MI | Plane kNN | Plane kNN + MI | Plane LogDet | Plane LogDet + MI | Sphere Contr. | Sphere Contr. + MI | Sphere kNN | Sphere kNN + MI | Sphere LogDet | Sphere LogDet + MI |
|-------|--------------|-------------------|-----------|----------------|--------------|-------------------|---------------|--------------------|------------|-----------------|---------------|--------------------|
| 10^0  | ~10^3        | ~10^3             | ~10^3     | ~10^3          | ~10^3        | ~10^3             | ~10^-1        | ~10^-1             | ~10^-1     | ~10^-1          | ~10^-1        | ~10^-1             |
| 10^1  | ~10^-5       | ~10^-5            | ~10^-5    | ~10^-5         | ~10^-5       | ~10^-5            | ~10^-1        | ~10^-1             | ~10^-1     | ~10^-1          | ~10^-1        | ~10^-1             |
| 10^2  | ~10^-9       | ~10^-9            | ~10^-9    | ~10^-9         | ~10^-9       | ~10^-9            | ~10^-5        | ~10^-5             | ~10^-5     | ~10^-5          | ~10^-5        | ~10^-5             |
</details>

B   
![](images/c023551c1238cfdfba5b7ac667205ae03063fe6467208165f73976669752003e.jpg)

<details>
<summary>scatter</summary>

| Dataset  | Method     | Contrastive Entropy | kNN Entropy | LogDet Entropy |
|----------|------------|---------------------|-------------|----------------|
| Plane    | LDM        | Varies              | Varies      | Varies         |
| Plane    | LDM + MI   | Varies              | Varies      | Varies         |
| Sphere   | LDM        | Varies              | Varies      | Varies         |
| Sphere   | LDM + MI   | Varies              | Varies      | Varies         |
</details>

Figure 3. Comparison of learned image representations on CIFAR-10. A The eigenspectrum of the learned representations generally decays more slowly for parametric entropy estimators, both on the plane (solid) and the sphere (dashed). Whether or not MI was maximized (+ MI) had little impact on the spectrum. The observed cutoff at low double digits is consistent with previous estimates of intrinsic dimensionality of CIFAR-10 (Pope et al., 2021). B T-SNE embeddings (Maaten & Hinton, 2008) of representations paint a similar picture in which MI maximization has little impact. Color denotes label.

$$
\begin{array}{l} \mathcal {F} _ {\mathrm{LDM}} = - D _ {K L} [ R (\boldsymbol {z}) \parallel P _ {\theta} (\boldsymbol {z}) ] \\ = \sum_ {t} \left\langle \log P _ {\theta} (z _ {t} | z _ {: t}) \right\rangle_ {R (z _ {t}, z _ {: t})} + H _ {R} [ z _ {t} | z _ {: t} ] , \tag {12} \\ \end{array}
$$

where $z _ { : t } = \{ z _ { s } : 0 \leq s < t \}$ . For additional MI maximization we find

$$
\begin{array}{l} \mathcal {F} _ {\mathrm{MI}} = - D _ {K L} [ R (\boldsymbol {z}) \parallel P _ {\theta} (\boldsymbol {z}) ] + \sum_ {t} I _ {R} [ z _ {t}, z _ {: t} ] \\ = \sum_ {t} \left\langle \log P _ {\theta} \left(z _ {t} \mid z _ {: t}\right) \right\rangle_ {R \left(z _ {t}, z _ {: t}\right)} + H _ {R} \left[ z _ {t} \right]. \tag {13} \\ \end{array}
$$

Depending on the choice of predictor $P _ { \theta } ( z _ { t } | z _ { : t } )$ and entropy estimator, we can relate $\mathcal { F } _ { \mathrm { L D M } }$ and $\mathcal { F } _ { \mathrm { M I } }$ to previously proposed models. The key difference from the image models in Section 5 is that these temporal models also optimize the latent models’ parameters θ.

Predictive SSL with stopgrad. The LDM goal $\mathcal { F } _ { \mathrm { { I } } }$ DM requires to either maximize the total entropy $H _ { R } [ z ]$ or the conditional entropies $H _ { R } [ z _ { t } | z _ { : t } ]$ . For long time series, the latter is much easier to achieve, since z becomes high-dimensional. In fact, predictive SSL approaches with stopgrad (Grill et al., 2020; Bardes et al., 2024; Mohammadi et al., 2025) implicitly implement approximate conditional entropy maximization, which we show in Appendix A.6.2. These approaches approximate the goal function as

$$
\mathcal {F} _ {\mathrm{LDM}} \approx \sum_ {t} \left\langle \log P _ {\theta} (S G [ z _ {t} ] | z _ {: t}) \right\rangle_ {R (z _ {t}, z _ {: t})}. \tag {14}
$$

In particular, if the predictor $P _ { \theta } ( z _ { t } | z _ { : t } )$ is a conditional Gaussian distribution with mean $p ( \boldsymbol { z } _ { : t } )$ , we recover the loss functions of previous predictive SSL approaches as proportional to $\| S \dot { G } [ z _ { t } ] - p \dot { ( } z _ { : t } ) \| ^ { 2 }$ .

Contrastive predictive coding (CPC). For $\mathcal { F } _ { \mathrm { M I } }$ , singletimestep entropy can easily be maximized through either KDE, kNN, or parametric estimators. Note that InfoNCE has been derived before from $\mathcal { F } _ { \mathrm { M I } }$ under a slightly different model (Aitchison & Ganev, 2024).

# 6.1. Nonlinear Bayesian filtering with Kalman predictor

To demonstrate how LDM enables deriving new SSL algorithms, we now consider a model in which the predictor is given by a Kalman filter. This enables to analytically quantify uncertainty in latent representations in a nonlinear dynamical system with linear latent dynamics. We summarize the assumptions in Fig. 4B, which lead us to the following model log-likelihood:

$$
\begin{array}{l} \log P _ {\theta} (z _ {t} | z _ {: t}) = \\ - \frac {1}{2} (z _ {t} - A h _ {t}) ^ {\top} \Sigma_ {e} ^ {- 1} (z _ {t} - A h _ {t}) - \frac {1}{2} \log | \Sigma_ {e} |, \tag {15} \\ \end{array}
$$

where $h _ { t }$ is the estimated hidden latent state of the filter and $\Sigma _ { e }$ the estimated prediction error covariance. $h _ { t }$ and $\Sigma _ { e }$ can be computed analytically from past latent states $z _ { t }$ through the well-known Kalman filter equations.

To understand the differences between representation learning with CPC and predictive stopgrad, we devised a nonlinear filtering problem based on synthetic videos. Since we employed a Kalman-based latent state predictor, we generated linear hidden dynamics that were then nonlinearly transformed into image space (Fig. 4A, Appendix Fig. A3). After learning, both CPC (with MI maximization) and predictive stopgrad SSL (without MI maximization) recovered the underlying latent variables equally well (Fig. 4C) with an $R ^ { 2 } = 0 . 9 9$ for both, as quantified by linear probing. Since we did not observe a discernible difference between maximizing MI or not, we wondered whether the network gradients of the single variable entropy $H _ { R } [ z _ { t } ]$ and conditional entropy $H _ { R } [ z _ { t } | z _ { : t } ]$ , respectively, were different. We therefore computed the cosine similarity between the network gradients with respect to the kNN single-variable entropy estimator (Appendix A.5.2), and the stopgrad conditional entropy estimator (Appendix A.5.4), and found that they consistently showed good alignment (Fig. 4D).

A   
![](images/618c76c14880318a6f90a68d8230cb4df0cd646e2214418050f21e1579859822.jpg)

<details>
<summary>text_image</summary>

Time
Ground Truth Pos.
Est. Pos. Stopgrad
Est. Pos. KNN Entropy
</details>

B   
![](images/bf9fc7ceb1a45e783dedade0a2ad13ba312a0d523eb11e6aebb022f1be032e69.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["..."] --> B["h_{t-1}"]
    B --> C["h_t"]
    C --> D["z_t"]
    D --> E["x_{t-1}"]
    B --> F["z_t-1"]
    C --> G["z_t"]
    F --> H["x_t"]
    G --> I["z_t"]
    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#ccf,stroke:#333
    style E fill:#cfc,stroke:#333
    style F fill:#fcc,stroke:#333
    style G fill:#fcc,stroke:#333
    style H fill:#ffc,stroke:#333
    style I fill:#ffc,stroke:#333
```
</details>

C   
![](images/07762930e6bd99a72aefb55156235deb4efa3fb9226e3b56ab2ca6ee0362fdb1.jpg)

<details>
<summary>line</summary>

| Time | x Position | y Position |
|------|------------|------------|
| 0    | 1.5        | 0.5        |
| 10   | -1.0       | -0.5       |
| 20   | 0.5        | 0.0        |
| 30   | -1.0       | -0.5       |
| 40   | 0.5        | 0.0        |
| 50   | -1.0       | -0.5       |
</details>

D   
![](images/3966e1462d10dae723744e1bac742792ebaf27e4556ac9bc24fa1ad38df02847.jpg)

<details>
<summary>line</summary>

| Epoch | Trained with kNN entropy | Trained with SG |
|-------|--------------------------|-----------------|
| 0     | 0.5                      | 0.3             |
| 5     | 0.7                      | 0.5             |
| 10    | 0.8                      | 0.6             |
| 15    | 0.75                     | 0.55            |
| 20    | 0.7                      | 0.6             |
</details>

E   
![](images/351619c1dfff8eb71610c540b2c34222ec3203de40d20072b15eb9db0d985f28.jpg)

F   
![](images/cc20aa7e82c419edc7fc8c535fd3add2eb7b9e37008f3fa469439e6c828e8d66.jpg)

<details>
<summary>scatter</summary>

| PCA 1 | PCA 2 |
|-------|-------|
| (various) | (various) |
</details>

G   
![](images/2f5d8f7262aab12cd472be5b9cfad2e84397437e6580e7db95f9a1c25b025cd5.jpg)

<details>
<summary>line</summary>

| Time (s) | True Position [m] | Est. kNN Position [m] |
| -------- | ----------------- | --------------------- |
| 0        | 1                 | 1                     |
| 5        | 2                 | 2                     |
| 10       | 1                 | 1                     |
| 15       | 2                 | 2                     |
| 20       | 1                 | 1                     |
| 25       | 2                 | 2                     |
| 30       | 1                 | 1                     |
| 35       | 2                 | 2                     |
| 40       | 1                 | 1                     |
| 45       | 2                 | 2                     |
| 50       | 1                 | 1                     |
| 55       | 2                 | 2                     |
| 60       | 1                 | 1                     |
| 65       | 2                 | 2                     |
| 70       | 1                 | 1                     |
| 75       | 2                 | 2                     |
| 80       | 1                 | 1                     |
| 85       | 2                 | 2                     |
| 90       | 1                 | 1                     |
| 95       | 2                 | 2                     |
| 100      | 1                 | 1                     |
| 105      | 2                 | 2                     |
| 110      | 1                 | 1                     |
| 115      | 2                 | 2                     |
| 120      | 1                 | 1                     |
| 125      | 2                 | 2                     |
| 130      | 1                 | 1                     |
| 135      | 2                 | 2                     |
| 140      | 1                 | 1                     |
| 145      | 2                 | 2                     |
| 150      | 1                 | 1                     |
| 155      | 2                 | 2                     |
| 160      | 1                 | 1                     |
| 165      | 2                 | 2                     |
| 170      | 1                 | 1                     |
| 175      | 2                 | 2                     |
| 180      | 1                 | 1                     |
| 185      | 2                 | 2                     |
| 190      | 1                 | 1                     |
| 195      | 2                 | 2                     |
| 200      | 1                 | 1                     |
| 205      | 2                 | 2                     |
| 210      | 1                 | 1                     |
| 215      | 2                 | 2                     |
| 220      | 1                 | 1                     |
| 225      | 2                 | 2                     |
| 230      | 1                 | 1                     |
| 235      | 2                 | 2                     |
| 240      | 1                 | 1                     |
| 245      | 2                 | 2                     |
| 250      | 1                 | 1                     |
| 255      | 2                 | 2                     |
| 260      | 1                 | 1                     |
| 265      | 2                 | 2                     |
| 270      | 1                 | 1                     |
| 275      | 2                 | 2                     |
| 280      | 1                 | 1                     |
| 285      | 2                 | 2                     |
| 290      | 1                 | 1                     |
| 295      | 2                 | 2                     |
| 300      | 1                 | 1                     |
| 305      | 2                 | 2                     |
| 310      | 1                 | 1                     |
| 315      | 2                 | 2                     |
| 320      | 1                 | 1                     |
| 325      | 2                 | 2                     |
| 330      | 1                 | 1                     |
| 335      | 2                 | 2                     |
| 340      | 1                 | 1                     |
| 345      | 2                 | 2                     |
| 350      | 1                 | 1                     |
| 355      | 2                 | 2                     |
| 360      | 1                 | 1                     |
| 365      | 2                 | 2                     |
| 370      | 1                 | 1                     |
| 375      | 2                 | 2                     |
| 380      | 1                 | 1                     |
| 385      | 2                 | 2                     |
| 390      | 1                 | 1                     |
| 395      | 2                 | 2                     |
| 400      | 1                 | 1                     |
| Note: The actual values for Neurons and Position are not provided in the code. I have used a placeholder to indicate the position values for each neuron. The actual values would be the true values from the input data. There is no label for the position values. The output data is estimated based on the estimated kNN values. The time scale is labeled as "Time [m]" (seconds).
</details>

Figure 4. Predictive distribution matching in latent space using a nonlinear Bayesian filtering model with Kalman-based predictor. A Example frames of synthetic dataset of a high dimensional noisy observable with linear latent dynamics. The red line denotes ground truth position. See Appendix, Fig. A3 for a more nonlinear task. B We use a Kalman filter backbone for the predictor $P _ { \theta } \big ( z _ { t } | z _ { : t } \big )$ with hidden states $h _ { t }$ and latent “observations” zt. C Estimated position and ground truth over time. After learning, we can linearly decode the true position and uncertainty from the Kalman hidden states $h _ { t }$ . Training with MI maximization (kNN entropy estimator) or without (stopgrad) results in approximately equal performance. D The cosine similarity of gradients of network weights w.r.t. entropy estimators also shows that both approaches lead to similar optimization. Similarity increases with training when the predictor becomes more accurate and so does the stopgrad entropy estimator. E Experimental setup in Grosmark & Buzsaki ´ (2016). F The learned latent states $h _ { t }$ are arranged in a circle according to the position and direction of the rat. One example trajectory over time is displayed in gray. G Example neural spiking timeseries and the estimated position based on the model. Shaded areas denote 95% confidence intervals based on model covariances $\bar { \Sigma } _ { h _ { t } }$ (see Appendix B).

To showcase the benefits of the Kalman-based predictor’s uncertainty-aware state, which is purely based on latent-state prediction, we extended the framework to allow for input-dependent predictor parameters $\theta ( x _ { t } )$ (Appendix, Fig. A4). Doing so allows noise-aware filtering that dynamically tunes the predictor to the current input noise level. We demonstrate this benefit using an in-vivo dataset of rat hippocampal spike trains, recorded while rats ran on a linear track (Fig. 4E). Since rat traversals are periodic, Kalman-based SSL learns to arrange encoded spike trains on a circle, where trajectories approximately follow a linear system (Fig 4F). The estimated uncertainty in the Kalman filter can be directly translated into an uncertainty estimate on predictions of rat position (Fig. 4G, Appendix, Fig. A4C), which previous predictive SSL approaches do not allow.

# 7. Identifiability in predictive models

A central question for predictive SSL is whether the learned latent state spaces correspond to meaningful underlying variables, or whether they remain arbitrary up to complex transformations. In this section, we show that LDM admits identifiable latent representations under standard assumptions on the generative process and noise model.

We consider predictive models that factorize the joint latent distribution as $\begin{array} { r } { P _ { \theta } ( z ) = \prod _ { t } P _ { \theta } ( z _ { t } | z _ { : t } ) } \end{array}$ , and assume that learning proceeds by matching this distribution and the empirical latent distribution $R ( z )$ induced by encoder $f .$

Theorem 1 (Identifiability of predictive distribution matching under a Gaussian predictor). Assume that:

(i) the model and true predictive distributions are Gaussian of the form:

$$
P (z _ {t} | z _ {: t}) = \mathcal {N} \bigl (z _ {t}; p (z _ {: t}), \Sigma \bigr) \quad ,
$$

with (potentially nonlinear) prediction function p(·) and non-degenerate covariance Σ;

(ii) the encoder is invertible on the data manifold;

(iii) the predictor covers the latent space.

Then, at the optimum of the LDM objective, the learned representation recovers the true latent variables up to an affine transformation.

The theorem shows that predictive SSL models trained via LDM identify the actual latent state spaces, without explicit observation-level likelihoods or contrastive objectives. Identifiability arises from the local noise structure of the prediction errors: the assumed Gaussian form of the predictive residuals constrains admissible transformations of the latent space, ruling out arbitrary nonlinear reparameterizations (Fig. 5A). Importantly, the result does not require strong assumptions on the expressiveness or architecture of the predictor itself beyond coverage of the latent space.

![](images/3f42ec34ae3619898abb242ebee34ebbd3aebdecda23889e8752e85b250f107c.jpg)

<details>
<summary>text_image</summary>

A
True latent space
Recov. latent space
vary \tilde{z}_1
P^*(\tilde{z}_t|\tilde{z}_{:t})
\tilde{z}_t
\tilde{z}_1
\tilde{z}_N
\tilde{z}_4
\tilde{z}_2
\tilde{z}_3
B
Noise-free latent trajectory
\tilde{z}_1
\tilde{z}_2
Data
Time
</details>

![](images/77a7ded5a73e6dfb5b5fb2f001d3684582b3e68e29a0de1570154db86e52e5c1.jpg)

<details>
<summary>text_image</summary>

C
True latent space
Rec. lat. sp. before learning
Rec. lat. sp. after learning
</details>

Figure 5. System identification through predictive LDM. A Forcing prediction errors into a Gaussian form leads to local linearization of the relation between true and recovered latent variables. B Schematic of nonlinear prediction task. Trajectory noise in the true latent space is Gaussian to enable identification. C Visualizations of the actual (left) and recovered latent space before (middle) and after (right) learning. Predictive LDM recovers the true space up to affine transformations.

The proof is given in Appendix A.7.1 and is closely related to previous identifiability proofs (Hyvarinen et al., 2019; Zimmermann et al., 2021; Laiz et al., 2025).

We can derive a similar identifiability result for von Mises-Fisher predictive models

Theorem 2. Under assumptions (ii) and (iii) and assuming a von Mises-Fisher predictive model the LDM objective recovers the true latent space up to an affine transformation.

See Appendix A.7.1 for the proof.

# 7.1. Empirical validation

To demonstrate identifiability in simulations we set up a nonlinear prediction task based on the simple dynamical system (Fig. 5B)

$$
\left\{ \begin{array}{l} \dot {x} = a k \cos (\theta) \cos (k (\theta - \theta_ {0})) - r \sin (\theta) + \nu_ {x} \\ \dot {y} = a k \sin (\theta) \cos (k (\theta - \theta_ {0})) + r \cos (\theta) + \nu_ {y} \end{array} \right. \tag {16}
$$

where $r = \sqrt { x ^ { 2 } + y ^ { 2 } } , \theta = \arctan ( y , x ) , \theta _ { 0 }$ is the angle of the system, and $k = 4$ . Further, $\nu _ { x }$ and $\nu _ { y }$ are Gaussian distributed random variables, which are essential to satisfy Assumption (i) of Theorem 1—although the model is robust to limited deviations from this condition (Appendix, Fig. A6). As before, we converted positions to images of a dot, while applying an additional nonlinear swirl transformation.

To model this data, we trained MLP inference and RNN + MLP prediction networks based on $\mathcal { F } _ { \mathrm { M I } }$ with a Gaussian predictor and kNN entropy estimator. This satisfies assumption (i)—note that we do not enforce assumptions (ii) and (iii), which we expected to emerge from entropy maximization. We chose a two-dimensional latent space to match the true and recovered latent dimensionality. The simulation gave two key results. First, the model linearly encodes the true position $( R ^ { 2 } = 0 . 9 4 )$ and thus can recover the true latent space up to affine transformations (Fig. 5C). Second, even though the initial inference function clearly is not injective, training via entropy maximization in LDM leads to an injective encoder after learning (Appendix A.4.2).

Note that stopgrad-based models like V-JEPA make similar model assumptions as in this section, but employ a heuristic entropy estimator that assumes the conditional distribution to be Gaussian. While this was enough to identify the simpler system in Section 6, where the predictor was additionally constrained to be linear, in this nonlinear scenario, we only found approximate identification of the true system (Appendix Fig. A6).

Finally, we verified that identifiably can also be achieved with higher dimensional latent distributions, although learning becomes more challenging (Appendix, Fig. A7).

# 8. Discussion

We showed that several established approaches to SSL can be understood under the unifying principle of distribution matching in latent space. We motivated this approach with insights from linear ICA and manifold normalizing flows: For encoders that are invertible on the data manifold, maximizing the data log-likelihood is equivalent to LDM. However, instead of maximizing the log-determinant of the Jacobian, as in normalizing flows, we maximized the entropy of the latent representations. We further showed that latent entropy maximization in LDM encourages invertibility on the data manifold. This insight explains the close connection between LDM with or without MI maximization, since in the case of an invertible encoder, the mutual information is maximized $I [ f ( x ) , f ( x ^ { \prime } ) ] = I [ x , x ^ { \prime } ]$ , and thus $\mathcal { F } _ { \mathrm { M I } } \propto \mathcal { F } _ { \mathrm { L D M } }$ . This explains why MI-based approaches have empirically been found to enable SSL despite the invariance of MI: they systematically misestimate MI, based on constrained (e.g., Gaussian) predictors, resulting in an LDM goal.

Our analysis highlights an important conceptual point. The assumption that the encoder is approximately invertible may appear to conflict with a common intuition in SSL: that its strength lies in discarding “irrelevant” information and retaining only high-level semantic or conceptual information. The LDM framework suggests a different interpretation: SSL representations may, in fact, discard little details about the data manifold. Indeed, latent spaces in SSL models frequently span hundreds or thousands of dimensions, while the intrinsic dimensionality of image data is typically estimated to be only a few tens (Pope et al., 2021). This disparity indicates that SSL does not primarily operate through lossy compression. Rather, its advantage lies in the ability to exert fine-grained control over how the data manifold is geometrically reparameterized by specifying the model likelihood in latent, rather than data, space. By embedding appropriate inductive biases into the latent model, the resulting alignment objective can promote manifold geometries in which abstract, generalizable features are untangled and readily accessible to the predictor, while deemphasizing local, pixel-level structure that generative models prioritize to preserve.

One of the main practical challenges in the LDM approach is to design accurate, sample efficient, and easily optimizable entropy estimators. Especially the requirement of optimizability necessitates compromises when choosing between different entropy approximation approaches. For example, choosing a fixed bandwidth for KDE is not optimal for precise entropy estimation, but significantly reduces the complexity of the estimator and increases its robustness. Given that entropy estimation significantly affects representational geometry and downstream performance, designing new entropy estimators with improved performance might be one of the most pressing problems for advancing SSL. This observation aligns with recent work proposing uniformity objectives grounded in statistical test theory (Balestriero & LeCun, 2025). Viewed through the lens of LDM, such approaches can be systematically analyzed and extended in the future using tools from statistical modeling.

# Acknowledgments

We thank Steffen Schneider, Rodrigo Gonzalez Laiz, Tobias ´ Schmidt, Manu Halvagal, Atena Mohammadi, and all Zenke Lab members for their input and discussions. This project was supported by the Swiss National Science Foundation (Grant Number PCEFP3 202981) and the Novartis Research Foundation.

# Impact Statement

This paper presents work whose goal is to advance the field of machine learning. There are many potential societal consequences of our work, none of which we feel must be specifically highlighted here.

# References

Ahmad, I. and Lin, P.-E. A nonparametric estimation of the entropy for absolutely continuous distributions (corresp.). IEEE Transactions on Information Theory, 22(3):372– 375, 1976.   
Aitchison, L. and Ganev, S. K. InfoNCE is variational inference in a recognition parameterised model. Transactions on Machine Learning Research, 2024. ISSN 2835-8856.   
Alshammari, S., Hershey, J., Feldmann, A., Freeman, W. T., and Hamilton, M. I-con: A unifying framework for representation learning. arXiv preprint arXiv:2504.16929, 2025.   
Assran, M., Balestriero, R., Duval, Q., Bordes, F., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., and Ballas, N. The hidden uniform cluster prior in self-supervised learning. In The Eleventh International Conference on Learning Representations, 2023.   
Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., et al. V-jepa 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025.   
Balestriero, R. and LeCun, Y. Lejepa: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025.   
Balestriero, R., Ballas, N., Rabbat, M., and LeCun, Y. Gaussian embeddings: How jepas secretly learn your data density. arXiv preprint arXiv:2510.05949, 2025.   
Barber, D. and Agakov, F. The im algorithm: a variational approach to information maximization. Advances in neural information processing systems, 16(320):201, 2004.   
Bardes, A., Ponce, J., and LeCun, Y. Vicreg: Varianceinvariance-covariance regularization for self-supervised learning. arXiv preprint arXiv:2105.04906, 2021.   
Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., and Ballas, N. Revisiting feature prediction for learning visual representations from video. arXiv preprint arXiv:2404.08471, 2024.   
Ben-Shaul, I., Shwartz-Ziv, R., Galanti, T., Dekel, S., and LeCun, Y. Reverse engineering self-supervised learning. Advances in Neural Information Processing Systems, 36: 58324–58345, 2023.   
Bizeul, A., Scholkopf, B., and Allen, C. A probabilistic ¨ model behind self- supervised learning. Transactions on Machine Learning Research, 2024. ISSN 2835-8856.

Brehmer, J. and Cranmer, K. Flows for simultaneous manifold learning and density estimation. Advances in neural information processing systems, 33:442–453, 2020.   
Cardoso, J.-F. Infomax and maximum likelihood for blind source separation. IEEE Signal processing letters, 4(4): 112–114, 2002.   
Caron, M., Touvron, H., Misra, I., Jegou, H., Mairal, J., ´ Bojanowski, P., and Joulin, A. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 9650–9660, 2021.   
Chen, T., Kornblith, S., Norouzi, M., and Hinton, G. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pp. 1597–1607. PmLR, 2020.   
Chen, X. and He, K. Exploring simple siamese representation learning. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 15750–15758, 2021.   
Chen, Z., Lin, C.-H., Liu, R., Xiao, J., and Dyer, E. Your contrastive learning problem is secretly a distribution alignment problem. Advances in Neural Information Processing Systems, 37:91597–91617, 2024.   
Choi, S., Cichocki, A., Park, H.-M., and Lee, S.-Y. Blind source separation and independent component analysis: A review. Neural Information Processing-Letters and Reviews, 6(1):1–57, 2005.   
da Costa, V. G. T., Fini, E., Nabi, M., Sebe, N., and Ricci, E. solo-learn: A library of self-supervised methods for visual representation learning. Journal of Machine Learning Research, 23(56):1–6, 2022.   
Daunhawer, I., Bizeul, A., Palumbo, E., Marx, A., and Vogt, J. E. Identifiability results for multimodal contrastive learning. arXiv preprint arXiv:2303.09166, 2023.   
Dawid, A. and LeCun, Y. Introduction to latent variable energy-based models: a path toward autonomous machine intelligence. Journal of Statistical Mechanics: Theory and Experiment, 2024(10):104011, 2024.   
Galvez, B. R., Blaas, A., Rodr ´ ´ıguez, P., Golinski, A., Suau, X., Ramapuram, J., Busbridge, D., and Zappella, L. The role of entropy and reconstruction in multi-view selfsupervised learning. In International Conference on Machine Learning, pp. 29143–29160. PMLR, 2023.   
Geiger, B. C. and Kubin, G. On the information loss in memoryless systems: the multivariate case. In 22th International Zurich Seminar on Communications (IZS). Eidgenossische Technische Hochschule Z ¨ urich, 2012. ¨

Grill, J.-B., Strub, F., Altche, F., Tallec, C., Richemond, P.,´ Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.   
Grosmark, A. D. and Buzsaki, G. Diversity in neural firing ´ dynamics supports both rigid and learned hippocampal sequences. Science, 351(6280):1440–1443, 2016.   
Gui, J., Chen, T., Zhang, J., Cao, Q., Sun, Z., Luo, H., and Tao, D. A survey on self-supervised learning: Algorithms, applications, and future trends. IEEE Transactions on Pattern Analysis and Machine Intelligence, 46(12):9052– 9071, 2024.   
Halvagal, M. S. and Zenke, F. The combination of hebbian and predictive plasticity learns invariant object representations in deep sensory networks. Nature neuroscience, 26(11):1906–1915, 2023.   
Halvagal, M. S., Laborieux, A., and Zenke, F. Implicit variance regularization in non-contrastive ssl. Advances in Neural Information Processing Systems, 36:63409– 63436, 2023.   
Hromadka, S., Biegun, K., Fox, L., Heald, J., and Sahani, M. Maximum likelihood learning of latent dynamics without reconstruction. arXiv preprint arXiv:2505.23569, 2025.   
Hyvarinen, A. and Oja, E. Independent component analysis: ¨ A tutorial. Neural Networks, 1:1–30, 1999.   
Hyvarinen, A., Sasaki, H., and Turner, R. Nonlinear ica using auxiliary variables and generalized contrastive learning. In The 22nd international conference on artificial intelligence and statistics, pp. 859–868. PMLR, 2019.   
Hyvarinen, A., Khemakhem, I., and Morioka, H. Nonlinear ¨ independent component analysis for principled disentanglement in unsupervised deep learning. Patterns, 4(10), 2023.   
Jiao, Y., Ma, W., Sun, D., Wang, H., and Wang, Y. Distribution matching for self-supervised transfer learning. arXiv preprint arXiv:2502.14424, 2025.   
Khemakhem, I., Kingma, D., Monti, R., and Hyvarinen, A. Variational autoencoders and nonlinear ica: A unifying framework. In International conference on artificial intelligence and statistics, pp. 2207–2217. PMLR, 2020a.   
Khemakhem, I., Monti, R., Kingma, D., and Hyvarinen, A. Ice-beem: Identifiable conditional energy-based deep models based on nonlinear ica. Advances in Neural Information Processing Systems, 33:12768–12778, 2020b.

Kirchhof, M., Kasneci, E., and Oh, S. J. Probabilistic contrastive learning recovers the correct aleatoric uncertainty of ambiguous inputs. In International Conference on Machine Learning, pp. 17085–17104. PMLR, 2023.   
Kozachenko, L. Sample estimate of the entropy of a random vector. Probl. Pered. Inform., 23:9, 1987.   
Krantz, S. and Parks, H. Geometric integration theory. Springer, 2008.   
Krantz, S. G. and Parks, H. R. The implicit function theorem: history, theory, and applications. Springer Science & Business Media, 2002.   
Laiz, R. G., Schmidt, T., and Schneider, S. Self-supervised contrastive learning performs non-linear system identification. In The Thirteenth International Conference on Learning Representations, 2025.   
Lombardi, D. and Pant, S. Nonparametric k-nearestneighbor entropy estimator. Physical Review E, 93(1): 013310, 2016.   
Ma, Y., Tsao, D., and Shum, H.-Y. On the principles of parsimony and self-consistency for the emergence of intelligence. Frontiers of Information Technology & Electronic Engineering, 23(9):1298–1323, 2022.   
Maaten, L. v. d. and Hinton, G. Visualizing data using t-sne. Journal of machine learning research, 9(Nov): 2579–2605, 2008.   
Mazur, S. and Ulam, S. Sur les transformations isometriques ´ d’espaces vectoriels normes.´ CR Acad. Sci. Paris, 194 (946-948):116, 1932.   
Mohammadi, A. G., Halvagal, M. S., and Zenke, F. Understanding cortical computation through the lens of jointembedding predictive architectures. bioRxiv, pp. 2025– 11, 2025.   
Nakamura, H., Okada, M., and Taniguchi, T. Representation uncertainty in self-supervised learning as variational inference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 16484–16493, 2023.   
Noroozi, M. and Favaro, P. Unsupervised Learning of Visual Representations by Solving Jigsaw Puzzles. In Leibe, B., Matas, J., Sebe, N., and Welling, M. (eds.), Computer Vision – ECCV 2016, Lecture Notes in Computer Science, pp. 69–84, Cham, 2016. Springer International Publishing. ISBN 978-3-319-46466-4. doi: 10.1007/978-3-319-46466-4 5.   
Oord, A. v. d., Li, Y., and Vinyals, O. Representation learning with contrastive predictive coding. arXiv preprint arXiv:1807.03748, 2018.

Paninski, L. Estimation of entropy and mutual information. Neural computation, 15(6):1191–1253, 2003.   
Papamakarios, G., Nalisnick, E., Rezende, D. J., Mohamed, S., and Lakshminarayanan, B. Normalizing flows for probabilistic modeling and inference. Journal of Machine Learning Research, 22(57):1–64, 2021.   
Poole, B., Ozair, S., Van Den Oord, A., Alemi, A., and Tucker, G. On variational bounds of mutual information. In International conference on machine learning, pp. 5171–5180. PMLR, 2019.   
Pope, P., Zhu, C., Abdelkader, A., Goldblum, M., and Goldstein, T. The intrinsic dimension of images and its impact on learning. In International Conference on Learning Representations, 2021.   
Reizinger, P., Bizeul, A., Juhos, A., Vogt, J. E., Balestriero, R., Brendel, W., and Klindt, D. Cross-entropy is all you need to invert the data generating process. arXiv preprint arXiv:2410.21869, 2024.   
Reizinger, P., Balestriero, R., Klindt, D., and Brendel, W. Position: An empirically grounded identifiability theory will accelerate self-supervised learning research. arXiv preprint arXiv:2504.13101, 2025.   
Roeder, G., Metz, L., and Kingma, D. On linear identifiability of learned representations. In International Conference on Machine Learning, pp. 9030–9039. PMLR, 2021.   
Ruelle, D. Positivity of entropy production in nonequilibrium statistical mechanics. Journal of Statistical Physics, 85(1):1–23, 1996.   
Saunshi, N., Plevrakis, O., Arora, S., Khodak, M., and Khandeparkar, H. A theoretical analysis of contrastive unsupervised representation learning. In International conference on machine learning, pp. 5628–5637. PMLR, 2019.   
Schneider, S., Lee, J. H., and Mathis, M. W. Learnable latent embeddings for joint behavioural and neural analysis. Nature, 617(7960):360–368, 2023.   
Shwartz Ziv, R. and LeCun, Y. To compress or not to compress—self-supervised learning and information theory: A review. Entropy, 26(3):252, 2024.   
Shwartz-Ziv, R., Balestriero, R., Kawaguchi, K., Rudner, T. G., and LeCun, Y. An information theory perspective on variance-invariance-covariance regularization. Advances in Neural Information Processing Systems, 36: 33965–33998, 2023.

Sorrenson, P., Draxler, F., Rousselot, A., Hummerich, S., Zimmermann, L., and Kothe, U. Lifting archi-¨ tectural constraints of injective flows. arXiv preprint arXiv:2306.01843, 2023.   
Sprekeler, H., Zito, T., and Wiskott, L. An extension of slow feature analysis for nonlinear blind source separation. The Journal of Machine Learning Research, 15(1):921–947, 2014.   
Tian, Y., Krishnan, D., and Isola, P. Contrastive multiview coding. In European conference on computer vision, pp. 776–794. Springer, 2020.   
Tian, Y., Chen, X., and Ganguli, S. Understanding selfsupervised learning dynamics without contrastive pairs. In International Conference on Machine Learning, pp. 10268–10278. PMLR, 2021.   
Tschannen, M., Djolonga, J., Rubenstein, P. K., Gelly, S., and Lucic, M. On mutual information maximization for representation learning. In International Conference on Learning Representations, 2020.   
Von Kugelgen, J., Sharma, Y., Gresele, L., Brendel, W., ¨ Scholkopf, B., Besserve, M., and Locatello, F. Self-¨ supervised learning with data augmentations provably isolates content from style. Advances in neural information processing systems, 34:16451–16467, 2021.   
Walker, W. I., Soulat, H., Yu, C., and Sahani, M. Unsupervised representation learning with recognitionparametrised probabilistic models. In International Conference on Artificial Intelligence and Statistics, pp. 4209– 4230. PMLR, 2023.   
Wang, T. and Isola, P. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In International conference on machine learning, pp. 9929–9939. PMLR, 2020.   
Zhouyin, Z. and Liu, D. Understanding neural networks with logarithm determinant entropy estimator. Neurocomputing, pp. 131520, 2025.   
Zimmermann, R. S., Sharma, Y., Schneider, S., Bethge, M., and Brendel, W. Contrastive learning inverts the data generating process. In International conference on machine learning, pp. 12979–12990. PMLR, 2021.

# Appendix Contents

# A Mathematical appendix 13

A.1 Notation 13   
A.2 Related frameworks 13   
A.3 Latent distribution matching framework overview 14   
A.4 Motivation of latent distribution matching framework 15   
A.5 Entropy estimators 17   
A.6 Goal function derivations 18   
A.7 Proofs 22   
A.8 Investigation of categorical model with probabilistic encoder 26

# B Simulation details 28

# C Supplementary figures and tables 29

# A. Mathematical appendix

# A.1. Notation

We make use of the following mathematical notation

• $\begin{array} { r } { \langle f ( x ) \rangle _ { P ( x ) } = \int f ( x ) P ( x ) } \end{array}$ dx denotes the average of $f ( x )$ w.r.t. $P ( x )$ .   
• $\delta ( x )$ is the Dirac delta function.   
• $\begin{array} { r } { H _ { P } [ x ] = - \int P ( x ) \log P ( x ) } \end{array}$ dx is the entropy of $P ( x )$   
• $\begin{array} { r } { H _ { P } [ x | y ] = - \int P ( x , y ) } \end{array}$ log $P ( x | y )$ dx is the conditional entropy of $P ( x | y )$ .   
• $I _ { P } [ x , y ] = H _ { P } [ x ] + H _ { P } [ y ] - H _ { P } [ x , y ]$ is the mutual information between x and $y .$   
• $\begin{array} { r } { D _ { K L } [ P ( x ) \parallel Q ( x ) ] = \int P ( x ) \log ( P ( x ) / Q ( x ) ) } \end{array}$ dx is the Kullback-Leibler divergence of $P ( x )$ and $Q ( x )$   
• $| M |$ denotes the determinant of the matrix M .   
• $\begin{array} { r } { J _ { f } ( x ) = \left[ { \frac { \partial f ( x ) } { \partial x _ { 1 } } } \ldots { \frac { \partial f ( x ) } { \partial x _ { n } } } \right] } \end{array}$ is the Jacobian matrix of $f$ evaluated at $x .$ ∂xn   
• $\operatorname { S G } [ \cdot ]$ is the stop-gradient operator, which prevents optimization gradients from flowing through this node

# A.2. Related frameworks

# A.2.1. REVERSE KL DIVERGENCE

A formalism closely related to ours was proposed by Zimmermann et al. (2021). In their theory, the latent distribution is defined through the conditional $\begin{array} { r } { R ( z ^ { \prime } | z ) = \frac { 1 } { Z } \exp ( - E ( f ( x ) , f ( x ^ { \prime } ) ) ) } \end{array}$ ), where $E ( \cdot )$ is an energy function and $f$ encodes x to $z .$ The goal function is then (implicitly) defined as

$$
\mathcal {F} _ {\mathrm{MDL}} = - D _ {K L} \left[ P _ {\theta} (z ^ {\prime} | z) \parallel R (z ^ {\prime} | z) \right] \propto \langle \log R (z ^ {\prime} | z) \rangle_ {P _ {\theta} (z ^ {\prime} | z)} , \tag {17}
$$

where the proportionality holds for fixed θ. This is also called the reverse KL divergence (Papamakarios et al., 2021). While here it is not necessary to compute an entropy term, the difficulty with this approach is that, instead of averaging over the data distribution, it requires to average over the latent distribution $P _ { \theta } \left( z , z ^ { \prime } \right)$ . This average can only be taken in specific circumstances, $\mathrm { e . g . }$ , for uniform distributions on the sphere, which hinders a more general applicability of the framework, and renders relating it to the diversity of SSL approaches difficult.

A   
![](images/5025ca609d241d9ef44a836d013ac6f5642cce3cfcf5fe941ae9fb5a9d4464f2.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["z"] --> B["z'"]
    C["x"] --> A
    D["x'"] --> B
    E["Pθ(z'|z)"] --> A
    F["Pdata(x,x')"] --> C
    F --> D
```
</details>

B   
![](images/d193965eae044b234607cc1b0ee32c354e6a2075acc132fecf652bdb49a15a30.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    Rz["R(z)"] --> z1["z1"]
    Rz --> z2["z2"]
    Rz --> z3["z3"]
    Rz --> zi["xi"]
    z1 --> xi["x1"]
    z2 --> xi2["x2"]
    z3 --> xi3["x3"]
    zi --> xi4["x4"]
    zi --> fi["fi(xi)"]
    z1 --> Pθ[Pθ(zi|PA(zi))]
    z2 --> Pθ
    z3 --> Pθ
    zi --> Pθ
    zi --> Pθ
    zi --> Pθ
```
</details>

Figure A1. Examples of LDM models. A In a simple model with two inputs the goal is to predict one latent variable from the other. B In general, any loop-free graphical model can form the predictor on any number of inputs, possibly with distinct encoders $f _ { i } ( x _ { i } )$ .

# A.2.2. I-CON

Another similar loss has been proposed by Alshammari et al. (2025). I-Con defines two conditional neighborhood distributions over data points $i , j \in \mathcal { X } \colon$ a supervisory distribution $p _ { \theta } ( j | i )$ encoding assumed relationships between samples (e.g., augmentation pairs, class labels, k-nearest neighbors), and a learned distribution $q _ { \phi } ( j | i )$ derived from the representations. The I-Con loss then takes the form

$$
\mathcal {L} _ {\mathrm{I-Con}} = \int_ {i \in \mathcal {X}} D _ {\mathrm{KL}} \left(p _ {\theta} (\cdot | i) \| q _ {\phi} (\cdot | i)\right). \tag {18}
$$

A key structural difference to LDM is that I-Con treats the supervisory signal $p _ { \theta } ( j | i )$ as given (or separately constructed), and optimizes the learned representation to match it. LDM instead specifies a generative latent model $P _ { \theta } ( \mathbf { z } ^ { \prime } | \mathbf { z } ) P _ { \theta } ( \mathbf { z } )$ and jointly optimizes both the encoder and the model parameters θ to minimize the divergence in latent space. An unresolved question is wether I-Con can also establish identifiability guarantees, similar to the forward or reverse KL divergence approaches.

# A.2.3. VARIATIONAL BOUND ON MUTUAL INFORMATION

Here we clarify the relation of the mutual information bound used in the main paper, and the bound most previous work employs (Barber & Agakov, 2004; Poole et al., 2019; Wang & Isola, 2020; Shwartz-Ziv et al., 2023; Galvez et al. ´ , 2023). This second bound takes the form

$$
I _ {R} [ z, z ^ {\prime} ] \geq \hat {I} _ {R} [ z, z ^ {\prime} ] = \left\langle \log P _ {\theta} (z | z ^ {\prime}) \right\rangle_ {R (z, z ^ {\prime})} + H _ {R} [ z ], \tag {19}
$$

which seems to differ from $\mathcal { F } _ { \mathrm { M I } }$ . However, these two MI bounds are easy to relate. To show this, we assume that the latent model employs an empirical prior, and thus takes the form $P _ { \theta } ( z , z ^ { \prime } ) = P _ { \theta } ( z | z ^ { \prime } ) R ( z ^ { \prime } )$ . Under this model $\mathcal { F } _ { \mathrm { M I } }$ simplifies to

$$
\begin{array}{l} \mathcal {F} _ {\mathrm{MI}} = - D _ {K L} \left[ R \left(z, z ^ {\prime}\right) \| P _ {\theta} \left(z, z ^ {\prime}\right) \right] + I _ {R} \left[ z, z ^ {\prime} \right] \\ = \left\langle \log P _ {\theta} (z, z ^ {\prime}) \right\rangle_ {R (z, z ^ {\prime})} + H _ {R} [ z ] + H _ {R} [ z ^ {\prime} ] \\ = \left\langle \log P _ {\theta} (z | z ^ {\prime}) + \log R (z ^ {\prime}) \right\rangle_ {R (z, z ^ {\prime})} + H _ {R} [ z ] + H _ {R} [ z ^ {\prime} ] \tag {20} \\ = \left\langle \log P _ {\theta} (z | z ^ {\prime}) \right\rangle_ {R (z, z ^ {\prime})} + H _ {R} [ z ]. \\ \end{array}
$$

# A.3. Latent distribution matching framework overview

In its most general form the goal function can be written as

$$
\begin{array}{l} \mathcal {F} _ {\mathrm{LDM}} = - D _ {K L} [ R (\boldsymbol {z}) \| P _ {\theta} (\boldsymbol {z}) ] \\ = \underbrace {\left\langle \sum_ {i} \log P _ {\theta} \left(z _ {i} \mid \mathrm{PA} \left(z _ {i}\right)\right) \right\rangle_ {R (\boldsymbol {z})}} _ {\text {Likelihood}} + \underbrace {H _ {R} [ \boldsymbol {z} ]} _ {\text {Entropy}} \end{array} \tag {21}
$$

where $z = \{ z _ { i } : i \in [ 1 \ldots N ] \}$ and $\mathrm { P A } \left( z _ { i } \right)$ are the parents of $z _ { i }$ in the probabilistic model. The parents of $z _ { i }$ can also include dependencies on additional conditioning variables that are independent of other observations, which we did not consider in this paper. These can be, for example, control signals, time indices, etc. While the joint entropy $H _ { R } [ z ]$ can be difficult to maximize, given the arguments in this paper (Section A.4.3), it might be replaced by the sum of individual entropies $\textstyle \sum _ { i } H _ { R _ { i } } [ z _ { i } ]$ with indistinguishable results.

The empirical latent distribution induced by the data distribution and the encoder is defined by

$$
\boxed {R (\boldsymbol {z}) = \left\langle \prod_ {i} R _ {i} (z _ {i} | x _ {i}) \right\rangle_ {P _ {\mathrm{data}} (\boldsymbol {x})}} \tag {22}
$$

This formulation also subsumes models with multimodal inputs and multiple encoders, or temporal models (Figure A1). In the following we only employ deterministic encoders $R _ { i } ( z _ { i } | x _ { i } ) = \delta ( z _ { i } - f _ { i } ( x _ { i } ) )$ , as motivated by the discussion in Section A.4.1, while probabilistic encoders are discussed in Section A.8.

# A.4. Motivation of latent distribution matching framework

# A.4.1. RELATION BETWEEN MAXIMUM LIKELIHOOD LEARNING AND DISTRIBUTION MATCHING

We can motivate the LDM framework through the maximum likelihood objective, mirroring the approach employed in ICA (Hyvarinen & Oja ¨ , 1999) or normalizing flow networks (Papamakarios et al., 2021). Here, models often compute the datalikelihood in latent, instead of data space, by applying the change of variables formula. Specifically, if the transformation g from latent variables z to data x is invertible

$$
\begin{array}{l} \mathcal {L} (\theta , g) = \left\langle \log P _ {\theta} (x) \right\rangle_ {P _ {\text { data }} (x)} \\ = \left\langle \log P _ {\theta} \left(g ^ {- 1} (x)\right) + \log \left| J _ {g ^ {- 1}} (x) \right| \right\rangle_ {P _ {\text {data}} (x)}, \tag {23} \\ \end{array}
$$

where $P _ { \theta }$ is the model distribution with parameters θ and $J _ { g ^ { - 1 } } ( x )$ is the Jacobian of the inverse transformation (see also Papamakarios et al., 2021). Here and in the following we assume that g (and $f )$ is a smooth and continuous function. The change of variables as above requires data and latent space to be of the same dimensionality and thus cannot be applied to functions that change the number of dimensions, as employed here. However, for the following argument it is sufficient if the encoder is invertible on the data manifold ${ \mathcal { M } } _ { \mathrm { d a t a } } ,$ , which is also possible if the data manifold is embedded into a higher dimensional space. This can be shown via a generalized change of variables formula that has been previously used, e.g., in manifold normalizing flows (Brehmer & Cranmer, 2020), and takes the form

$$
\begin{array}{l} \mathcal {L} (\theta , g) = \left\langle \log P _ {\theta} (x) \right\rangle_ {P _ {\text { data }} (x)} \\ = \left\langle \log P _ {\theta} (g ^ {- 1} (x)) + \frac {1}{2} \log | J _ {g ^ {- 1}} (x) J _ {g ^ {- 1}} (x) ^ {T} | \right\rangle_ {P _ {\mathrm{data}} (x)}. \tag {24} \\ \end{array}
$$

We can relate the maximum likelihood objective to a LDM goal by noticing that for any transformation f that is invertible on the data manifold, the differential entropy changes as

$$
H _ {P} [ x ] = H _ {P} [ f (x) ] - \left\langle \frac {1}{2} \log | J _ {f} (x) J _ {f} (x) ^ {T} | \right\rangle_ {P (x)}, \tag {25}
$$

which is also a consequence of the change of variables formula for manifolds. Thus, if we define the encoder $f ( x ) = g ^ { - 1 } ( x )$ we can rewrite the log-likelihood in an exclusively latent-space form, by adding the constant data entropy (Papamakarios et al., 2021; Cardoso, 2002)

$$
\begin{array}{l} \mathcal {L} (\theta , f) \propto \left\langle \log P _ {\theta} (f (x)) + \frac {1}{2} \log | J _ {f} (x) J _ {f} (x) ^ {T} | \right\rangle_ {P _ {\text { data }} (x)} + H _ {P _ {\text { data }}} [ x ] \\ = \left\langle \log P _ {\theta} (f (x)) \right\rangle_ {P _ {\text {data}} (x)} + H _ {P _ {\text {data}}} [ f (x) ] \tag {26} \\ = - D _ {K L} \left[ P _ {\text { data }} (f (x)) \| P _ {\theta} (f (x)) \right]. \\ \end{array}
$$

In summary, for an invertible data generation process, maximum likelihood learning is equivalent to LDM in latent space. This also means that, in principle, we can estimate the log-likelihood of a datapoint based on the change of variables formula above; we do not explore this here, but see Balestriero et al. (2025). The other observation we want to highlight is that, while manifold normalizing flows is concerned with maximizing or regularizing the Jacobian term log $| J _ { f } \bar { ( } x ) \bar { J } _ { f } ( x ) ^ { T } |$ |, it is equivalent to maximize the latent entropy instead. Notably, while commonly manifold normalizing flow networks are constrained to be injective on the data manifold, we do not constrain the networks, but instead fully rely on the regularization through entropy maximization (Section A.4.2). This is closely related to approaches in manifold normalizing flow that aim to drop architectural constraints to improve expressibility (Sorrenson et al., 2023).

# A.4.2. APPROXIMATE INVERTIBILITY THROUGH ENTROPY MAXIMIZATION

In the proposed goal function the encoder is encouraged to be both locally and globally invertible on the data manifold through the entropy term. To see how it encourages local invertibility, we assume the data lies on a manifold $\mathcal { M } _ { \mathrm { d a t a } }$ of intrinsic dimension matching the latent space. Here we define the manifold as the support of the data distribution $\mathcal { M } _ { \mathrm { d a t a } } = \operatorname { s u p p } ( P _ { \mathrm { d a t a } } )$ . For the encoder to be a local diffeomorphism (invertible on the manifold), the Jacobian restricted to the tangent space must have full rank (Krantz & Parks, 2008, p. 125). This is satisfied if for all $x \in \mathcal { M } _ { \mathrm { d a t a } }$ (Inverse function theorem):

$$
\begin{array}{l} \sqrt {\left| J _ {f} (x) J _ {f} (x) ^ {T} \right|} > 0 \\ \Leftrightarrow \frac {1}{2} \log \left| J _ {f} (x) J _ {f} (x) ^ {T} \right| > - \infty \end{array} \tag {27}
$$

On the other hand, through a generalized change of variables formula for manifolds, the latent entropy has the upper bound (Geiger & Kubin, 2012):

$$
H _ {P} [ f (x) ] \leq H _ {P} [ x ] + \left\langle \frac {1}{2} \log | J _ {f} (x) J _ {f} (x) ^ {T} | \right\rangle_ {P (x)}, \tag {28}
$$

which is an equality for injective (non-folding) transformations. Consequently, if the encoder is locally invertible over the data distribution upon initialization, latent entropy maximization encourages it to remain locally invertible during training by penalizing singular Jacobians, i.e., for which the log-determinant would approach −∞ (see also Fig. A2A).

Global invertibility on the data manifold of f requires additional assumptions next to non-singular Jacobians, for example that f is a proper map as assumed in Hadamard’s global inverse function theorem (Krantz & Parks, 2002, p. 127). However, entropy maximization also provides an inductive bias towards global invertibility. Consider a mapping that is locally invertible but globally non-injective (e.g., a ’folded’ or ’wrapped’ manifold). Such a mapping inevitably superimposes probability mass from distinct data regions onto the same latent locations. If the amount the distribution can be ’stretched’ by f (the log-determinant of the Jacobian) is limited, which typically is a result of simultaneously maximizing the model log-likelihood, this creates regions of higher probability density. Since differential entropy is maximized by spreading probability mass as uniformly as possible over the available volume, any such superposition (folding) results in a suboptimal entropy compared to an unfolded, globally injective representation.

To be more precise, how much information is lost by such a folding of the manifold is quantified by the missing term in the previous inequality (Equation 28), which is also termed the folding entropy $H _ { P } [ x | f ( x ) ]$ ]. This leads to the expression of the latent entropy, which—for finite Jacobian—is maximized for non folding manifolds with $H _ { P } [ x | f ( x ) ] = 0$ (Geiger & Kubin, 2012; Ruelle, 1996)

$$
H _ {P} [ f (x) ] = H _ {P} [ x ] + \left\langle \frac {1}{2} \log | J _ {f} (x) J _ {f} (x) ^ {T} | \right\rangle_ {P (x)} - H _ {P} [ x | f (x) ] \tag {29}
$$

This equation holds for piecewise invertible f, i.e., most non-degenerate neural networks—see also the derivation in Section A.7.2. Thus, the objective function naturally penalizes topological defects and encourages the encoder to effectively ’unfold’ the data manifold into the latent space. Note that LDM might still lead to non-injective encoders under a misspecification of the model, e.g., if the prior $P _ { \theta } ( z )$ is chosen too narrowly, or the conditional $P _ { \theta } ( z ^ { \prime } | z )$ is too loose and permits extreme stretching of the manifold—or, of course, if entropy estimation is inaccurate.

Previously, invertibility of the encoder on the data manifold has been proven to result from contrastive learning under different assumptions by Schneider et al. (2023); Reizinger et al. (2024).

# A.4.3. MAXIMIZATION OF MUTUAL INFORMATION THROUGH ENTROPY MAXIMIZATION

Given the argument in A.4.2, we can link MI maximization between related inputs to latent entropy maximization. First, we note that through the data processing inequality we know that for any encoder f MI can only decrease

$$
I [ x, x ^ {\prime} ] \geq I [ f (x), f (x ^ {\prime}) ]. \tag {30}
$$

As discussed in the introduction, a fundamental fact about MI is that if the encoder $f$ is invertible on the data manifold, then MI is preserved

$$
I [ x, x ^ {\prime} ] = I [ f (x), f (x ^ {\prime}) ]. \tag {31}
$$

Clearly, this is the maximum MI achievable. As discussed in A.4.2, invertibility can be encouraged simply by maximization of the latent entropy (while restricting the stretching of the latent manifold), which thus also implies approximate maximization of MI.

# A.5. Entropy estimators

Before discussing the derivations of specific goal functions, we briefly introduce the entropy estimators used to approximate the uniformity terms.

# A.5.1. KERNEL DENSITY ESTIMATION (KDE)

KDE first approximates the continuous probability density function (PDF) R(z) through probability kernels κ around samples $z _ { j } \sim R ( z )$ , and based on this approximates the entropy (Ahmad & Lin, 1976). The kernel approximation of the pdf is given by $\begin{array} { r } { \hat { R } ( z ) = \frac { 1 } { n h } \sum _ { j } \kappa \left( \frac { d ( x , z _ { j } ) } { h } \right) } \end{array}$  d(x,zj )h , where d(·, ·) is some distance measure, n the number of samples, and h is the $d ( \cdot , \cdot )$ bandwidth of the kernel. The entropy estimate is then given by

$$
\hat {H} _ {\mathrm{KDE}} [ z ] = - \frac {1}{n} \sum_ {i} \log \hat {R} (z _ {i}) \propto - \frac {1}{n} \sum_ {i} \log \sum_ {j} \kappa \left(\frac {d (z _ {i} , z _ {j})}{h}\right). \tag {32}
$$

Note that we immediately can identify the comparison of $z _ { i }$ and $z _ { j }$ with the contrastive principle of SSL.

# A.5.2. K-NEAREST NEIGHBOR (KNN) ESTIMATION

Another robust estimator for the entropy of high-dimensional distributions is the Kozachenko and Leonenko nearest neighbor estimator (Kozachenko, 1987), which later has been generalized to kNN estimators (Lombardi & Pant, 2016). The estimator constructs a ball around every sample $z _ { i } ,$ , with radius given by the p-norm distance $\epsilon _ { i } k = | z _ { i } - \mathrm { k N N } ( z _ { i } ) | _ { p }$ between $z _ { i }$ and its kNN. Similar to the KDE, the sum of balls then approximate the PDF under the assumption of uniform distribution within the balls. This gives rise to the entropy estimator

$$
\hat {H} _ {\mathrm{kNN}} [ z ] \propto \frac {D}{n} \sum_ {i} \log \epsilon_ {i} k. \tag {33}
$$

One advantage of this estimator over KDE is that its parameters $p$ and k are more straightforward to choose than the bandwidth h.

# A.5.3. PARAMETRIC ESTIMATION

Entropy can also be estimated via a parametric approach, by assuming that the data is distributed according to a distribution with a closed form entropy expression $\hat { R } _ { \phi } ( z )$ , and estimating the parameters ϕ of this distribution from the data $z _ { j } \sim R ( z )$ . In practice, to the best of our knowledge, the only non-trivial distribution that allows to feasibly compute the entropy in high dimensions is the multidimensional normal and derived distributions (e.g., log-normal). For a D-dimensional normal $\hat { R } _ { \phi } ( z ) = \mathcal { N } ( z ; \mu , \Sigma )$ we can estimate $\begin{array} { r } { \mu = \frac { 1 } { N } \sum _ { j } z _ { j } } \end{array}$ and $\begin{array} { r } { \Sigma = \frac { 1 } { N - 1 } \sum _ { j } ( z _ { j } - \bar { \mu } ) ( \bar { z _ { j } } - \mu ) ^ { T } } \end{array}$ , and the entropy is given by the log-determinant of the covariance matrix

$$
\hat {H} _ {\text { LogDet }} [ z ] = \frac {D}{2} (1 + \log (2 \pi)) + \frac {1}{2} \log | \Sigma |. \tag {34}
$$

In practice, for high dimensional distributions, the determinant is numerically complex to compute, and additional approximations are needed (Zhouyin & Liu, 2025). To provide additional intuition, a very simple approximation can be obtained by assuming that off-diagonal terms are small, $\Sigma = D + A$ , with D diagonal and A small. Performing a low-order Taylor expansion yields log |Σ| = Pi log Σii − 12 Pj̸=  ii jj $\begin{array} { r } { | \Sigma | = \sum _ { i } \log \Sigma _ { i i } - \frac { 1 } { 2 } \sum _ { j \neq i } \frac { \Sigma _ { i j } \Sigma _ { j i } } { \Sigma _ { i i } \Sigma _ { j i } } + \mathcal { O } ( A ^ { 3 } ) } \end{array}$ i ΣijΣjiΣ Σ + O(A3). From this it becomes clear that entropy is maximized for maximum variance and minimum covariance terms.

# A.5.4. CONDITIONAL ENTROPY ESTIMATION WITH PREDICTOR

The conditional entropy of $R ( z ^ { \prime } | z )$ is given by $H _ { R } [ z ^ { \prime } | z ] = - \langle \log R ( z ^ { \prime } | z ) \rangle _ { R ( z ^ { \prime } , z ) }$ . Here we will evaluate the average $\langle \cdot \rangle _ { R ( z ^ { \prime } , z ) }$ via samples, and the main question regards how the conditional distribution is approximated. The most straightforward approach is to learn a predictor $\hat { R } _ { \theta } ( z ^ { \prime } | z )$ by minimizing the cross entropy $- \langle \log \hat { R } _ { \theta } ( z ^ { \prime } | z ) \rangle _ { R ( z ^ { \prime } , z ) }$ with respect to the parameters θ. The entropy can then be approximated and maximized with the plug-in estimator

$$
\hat {H} _ {\text { pred }} [ z ^ {\prime} | z ] = - \frac {1}{N} \sum_ {(z ^ {\prime}, z)} \log \hat {R} _ {\theta} (z ^ {\prime} | z). \tag {35}
$$

Intriguingly, the cross entropy to train the predictor is exactly equal to the entropy estimator of the conditional entropy. This means that when we train a predictor $\hat { R } _ { \theta } ( z ^ { \prime } | z )$ via cross-entropy, we are simultaneously obtaining an entropy estimate for free, without needing a separate estimator. We will discuss how this observation is leveraged implicitly by SSL models with predictor and stopgrad in Section A.6.2.

# A.6. Goal function derivations

# A.6.1. MODELS WITH PAIRS OF INPUTS

We will first treat models with pairs of inputs, and later discuss temporal models. Specifically, the goal is to derive the likelihood and entropy terms of the goal functions, which are special cases of the general goal function (Eq 21)

$$
\mathcal {F} _ {\mathrm{LDM}} = \left\langle \log P _ {\theta} \left(z, z ^ {\prime}\right) \right\rangle_ {R \left(z, z ^ {\prime}\right)} + H _ {R} \left[ z, z ^ {\prime} \right] \tag {36}
$$

$$
\mathcal {F} _ {\mathrm{MI}} = \left\langle \log P _ {\theta} \left(z, z ^ {\prime}\right) \right\rangle_ {R \left(z, z ^ {\prime}\right)} + 2 H _ {R} [ z ] \tag {37}
$$

Given the definition of the induced latent distribution (Eq 22) and deterministic encoders, the average likelihood becomes

$$
\begin{array}{l} \langle \log P _ {\theta} (z, z ^ {\prime}) \rangle_ {R (z, z ^ {\prime})} = \int (\log P _ {\theta} (z, z ^ {\prime}) \left\langle R (z | x) R (z ^ {\prime} | x ^ {\prime}) \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})} d z d z ^ {\prime} \\ = \left\langle \int \left(\log P _ {\theta} (z, z ^ {\prime})\right) R (z | x) R (z ^ {\prime} | x ^ {\prime}) d z d z ^ {\prime} \right\rangle_ {P _ {\text { data }} (x, x ^ {\prime})} \tag {38} \\ = \left\langle \log P _ {\theta} (f (x), f (x ^ {\prime})) \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})} \\ = \left\langle \log P _ {\theta} (f (x ^ {\prime}) | f (x)) + \log P _ {\theta} (f (x)) \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})}, \\ \end{array}
$$

where in the third step we used $R ( z | x ) = \delta ( z - f ( x ) )$ . In the last step we used $P _ { \theta } ( z , z ^ { \prime } ) = P _ { \theta } ( z ^ { \prime } | z ) P _ { \theta } ( z )$ .

The following derivations will first specify assumptions for $P _ { \theta } ( z ^ { \prime } | z )$ and $P _ { \theta } ( z )$ , and then approximations for the entropy, based on the estimators outlined in detail in Section A.5. See also the derivations in Wang & Isola (2020); Shwartz-Ziv et al. (2023); Galvez et al. ´ (2023), which employed the MI estimator derived from $\mathcal { F } _ { \mathrm { M I } }$ in A.2.3.

# VICReg We make the choices

$$
\begin{array}{l} P _ {\theta} (z) = F l a t, \\ \begin{array}{l} P _ {\theta} (z) = \text {P.} \\ P _ {\theta} \left(z ^ {\prime} \mid z\right) = \mathcal {N} \left(z ^ {\prime}; \mu = z, \Sigma = \sigma^ {2} I\right). \end{array} \tag {39} \\ \end{array}
$$

The likelihood term simplifies to

$$
\begin{array}{l} \left. \left\langle \log P _ {\theta} \left(f \left(x ^ {\prime}\right) \mid f (x)\right) + \log P _ {\theta} (f (x)) \right\rangle_ {P _ {\text { data }} \left(x, x ^ {\prime}\right)} \right. \\ = \left\langle \log \mathcal {N} \left(f \left(x ^ {\prime}\right); \mu = f (x), \Sigma = \sigma^ {2} I\right) \right\rangle_ {P _ {\text {data}} \left(x, x ^ {\prime}\right)} \tag {40} \\ \propto - \left\langle \frac {1}{2 \sigma^ {2}} \parallel f (x) - f (x ^ {\prime}) \parallel^ {2} \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})}. \\ \end{array}
$$

To estimate the entropy, we approximate the latent distribution with a Normal distribution $R ( z , z ^ { \prime } ) \approx \mathcal { N } ( [ z , z ^ { \prime } ] ; \mu _ { z , z ^ { \prime } } , \Sigma _ { z , z ^ { \prime } } )$ or $R ( z ) \approx \mathcal { N } ( z ; \mu _ { z } , \Sigma _ { z } )$ with $\mu$ the empirical mean and Σ the empirical covariance matrix. This leads to the closed form entropy approximation

$$
H _ {R} \left[ z, z ^ {\prime} \right] \propto \frac {1}{2} \log \left| \Sigma_ {z, z ^ {\prime}} \right| \tag {41}
$$

$$
H _ {R} [ z ] \propto \frac {1}{2} \log | \Sigma_ {z} |.
$$

While there is no closed form for the determinant of the covariance matrix that can be optimized easily, it is clear that a maximum entropy Gaussian is achieved with maximal variance and minimal off-diagonal covariance terms. One potential approach to show this is to assume that the covariance is close to diagonal (D diagonal and A small) and expand the log-determinant

$$
\log | \Sigma | = \log | D + A | \approx \sum_ {i} \log \Sigma_ {i i} - \sum_ {i \neq j} \mathrm{C} _ {i j}, \tag {42}
$$

where the first term promotes large variance and the second small covariance $C _ { i j } = \Sigma _ { i j } ^ { 2 } / ( \Sigma _ { i i } \Sigma _ { j j } )$ terms. Overall, VICReg can be understood to implement an approximation to the goal of minimizing

$$
\mathcal {F} _ {\mathrm{MI}} = - \frac {1}{2 \sigma^ {2}} \left\langle \| f (x) - f (x ^ {\prime}) \| ^ {2} \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})} + \log | \Sigma_ {z} |. \tag {43}
$$

See also Shwartz-Ziv et al. (2023), which finds a similar goal function based on principles of MI maximization.

SimCLR We make the choices, with latents z lying on the unit sphere

$$
P _ {\theta} (z) = \frac {1}{Z},
$$

$$
P _ {\theta} (z ^ {\prime} | z) = \frac {1}{Z} \exp (\beta z ^ {T} z ^ {\prime}) . \tag {44}
$$

The likelihood term then becomes

$$
\left\langle \int_ {S} (\log P _ {\theta} (z, z ^ {\prime})) R (z | x) R (z ^ {\prime} | x ^ {\prime}) d z d z ^ {\prime} \right\rangle_ {P _ {\text { data }} (x, x ^ {\prime})}
$$

$$
\propto \left\langle \int_ {S} \beta z ^ {T} z ^ {\prime} \delta (z - f (x)) \delta (z ^ {\prime} - f (x ^ {\prime}))   d z   d z ^ {\prime} \right\rangle_ {P _ {\text { data }} (x, x ')} \tag {45}
$$

$$
= \left\langle \beta f (x) ^ {T} f (x ^ {\prime}) \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})}.
$$

The Entropy term becomes, using the KDE approach with von Mises-Fisher kernels with bandwidth $\gamma$

$$
\hat {H} _ {\mathrm{KDE}} [ z ] \propto - \frac {1}{N} \sum_ {i} \log \left[ \sum_ {j} \kappa (f (x _ {i}), f (x _ {j}), \gamma) \right]
$$

$$
= - \frac {1}{N} \sum_ {i} \log \left[ \sum_ {j} 1 / Z \exp (\gamma^ {- 1} f (x _ {i}) ^ {T} f (x _ {j})) \right] \tag {46}
$$

$$
\propto - \frac {1}{N} \sum_ {i} \log \left[ \sum_ {j} \exp (\gamma^ {- 1} f (x _ {i}) ^ {T} f (x _ {j})) \right].
$$

If the entropy is approximated via KDE with bandwidth $\gamma = 1 / \beta$ then based on $\mathcal { F } _ { \mathrm { L D M } }$ we find the goal function

$$
\mathcal {F} = \left\langle \beta f (x) ^ {T} f \left(x ^ {\prime}\right) \right\rangle_ {P _ {\text {data}} \left(x, x ^ {\prime}\right)} - \left\langle \log \left\langle \exp \left\{\beta \left(f (x) ^ {T} f \left(x ^ {-}\right) + f \left(x ^ {\prime}\right) ^ {T} f \left(x ^ {\prime -}\right)\right) \right\} \right\rangle_ {P _ {\text {data}} \left(x ^ {-}, x ^ {\prime -}\right)} \right\rangle_ {P _ {\text {data}} \left(x, x ^ {\prime}\right)}. \tag {47}
$$

Similar to VICReg, the original goal function can be recovered from $\mathcal { F } _ { \mathrm { M I } }$ instead as

$$
\mathcal {F} = \left\langle \beta f (x) ^ {T} f \left(x ^ {\prime}\right) \right\rangle_ {P _ {\text {data}} \left(x, x ^ {\prime}\right)} - 2 \left\langle \log \left\langle \exp \left\{\beta f (x) ^ {T} f \left(x ^ {-}\right) \right\} \right\rangle_ {P _ {\text {data}} \left(x ^ {-}\right)} \right\rangle_ {P _ {\text {data}} (x)}. \tag {48}
$$

Technically speaking, the SimCLR goal differs from this goal function in that in SimCLR the negative samples $x ^ { - }$ include $x ^ { \prime } ,$ , the positive sample, while in LDM all samples in the entropy are i.i.d.—in practice, however, we did not observe a significant difference between the two.

SSL on the simplex Next to latent variables on the plane and sphere, we can also consider variables on the simplex $\Delta ^ { D - 1 } = \{ z \in \mathbb { R } : \sum _ { i } z _ { i } = 1 \}$ , which can be obtained with a softmax output layer. Often, points on the simplex are understood as probability distributions over categorical variables, but we will treat them here simply as points, allowing us to understand encoders as deterministic functions, as before. This leads to loss functions related to those used in the DINO family (Caron et al., 2021).

We make the choices, with latents z lying on the simplex

$$
P _ {\theta} (z) = \frac {1}{Z},
$$

$$
P _ {\theta} (z ^ {\prime} | z) = \mathrm{Dir} (z ^ {\prime}, \tau z + 1) = \frac {1}{B (\tau z + 1)} \prod_ {k} z _ {k} ^ {\prime \tau z _ {k}}. \tag {49}
$$

where here and in the following k indexes the components of z, $\begin{array} { r } { , \mathrm { D i r } ( p , \alpha ) = \frac { 1 } { B ( \alpha ) } \prod _ { k } z _ { k } ^ { \alpha _ { k } - 1 } } \end{array}$ is the Dirichlet distribution, and dist $\begin{array} { r } { B ( \alpha ) = \frac { \prod _ { k } \Gamma \left( \alpha _ { k } \right) } { \Gamma \left( \sum _ { k } \alpha _ { k } \right) } } \end{array}$ . In this model, τ can be understood as a p the mean of the conditional distribution is eter, similar to pre, but the mode is a Gaussian, justifying $\begin{array} { r } { \frac { \alpha } { \sum _ { k } \alpha _ { k } } = \frac { \tau z + 1 } { \tau + K } } \end{array}$ P α−1k αk−K = z $\textstyle { \frac { \alpha - 1 } { \sum _ { k } \alpha _ { k } - K } } = z .$ this choice of model.

The likelihood term for this model (dropping constant terms) is

$$
\left\langle \log P _ {\theta} (z, z ^ {\prime}) \right\rangle_ {R (z, z ^ {\prime})} \propto \left\langle \tau z ^ {T} \log (z ^ {\prime}) - \sum_ {k} \log \Gamma (\tau z _ {k} + 1) \right\rangle_ {R (z, z ^ {\prime})}, \tag {50}
$$

With Dirichlet-based KDE entropy, we can derive

$$
\hat {H} _ {\mathrm{KDE}} [ z ] \propto - \frac {1}{N} \sum_ {i} \log \left[ \sum_ {j} \operatorname{Dir} \left(z _ {i}, \tau z _ {j}\right) \right] \tag {51}
$$

$$
\propto - \frac {1}{N} \sum_ {i} \log \left[ \sum_ {j} \exp \left(\tau z _ {j} ^ {T} \log (z _ {i}) - \sum_ {k} \log \Gamma (\tau z _ {j k} + 1)\right) \right].
$$

Taken together we arrive at the DINO alignment term $\tau \sum _ { k } z _ { k } \log ( z _ { k } ^ { \prime } )$ , and an effective collapse prevention term that depends on contrastive samples and the log Gamma term from the normalization. The DINO models, in contrast, rely on more heuristic collapse prevention methods, that might not directly be related to any entropy estimator.

Empirically, we find that the other entropy estimators (kNN, LogDet) are not suited for variables on the simplex and do not converge. Results for the contrastive approach are summarized in Table A1.

# A.6.2. PREDICTIVE STOPGRAD MODELS

We can also understand predictive models with stopgrad (Grill et al., 2020; Chen et al., 2020; Bardes et al., 2024; Assran et al., 2025; Mohammadi et al., 2025) as approximately maximizing entropy. These models have the general goal function

$$
\mathcal {F} = \sum_ {t} \left\langle \log P _ {\theta} (S G [ z _ {t} ] | z _ {: t}) \right\rangle_ {R (z _ {t}, z _ {: t})}, \tag {52}
$$

where commonly, but not necessarily, $P _ { \theta }$ is a Normal distribution with either fixed or variable variance. To recall, in the proposed framework of LDM, the goal for a temporal model is the same as before, with the general goal function

$$
- D _ {K L} [ R (\boldsymbol {z}) \| P _ {\theta} (\boldsymbol {z}) ] = \sum_ {t} \left\langle \log P _ {\theta} \left(z _ {t} \mid z: t\right) \right\rangle_ {R \left(z _ {t}, z: t\right)} + H _ {R} \left[ z _ {t} \mid z: t \right]. \tag {53}
$$

To show the relation to predictive stopgrad losses, we start with the basic observation that

$$
\left. \right.\left\langle \log P _ {\theta} \left(S G \left[ z _ {t} \right] \mid z _ {: t}\right)\right\rangle_ {R \left(z _ {t}, z _ {: t}\right)} \stackrel {{\partial}} {{=}} \left\langle \log P _ {\theta} \left(z _ {t} \mid z _ {: t}\right)\right\rangle_ {R \left(z _ {t}, z _ {: t}\right)} - \left\langle \log P _ {S G [ \theta ]} \left(z _ {t} \mid S G [ z _ {: t} ]\right)\right\rangle_ {R \left(z _ {t}, z _ {: t}\right)}. \tag {54}
$$

where $\underline { { \underline { { \partial } } } }$ denotes equal in derivative. Thus, both goal functions match in derivative if the second term approximates the (derivative of the) conditional entropy of $R \big ( z _ { t } | z _ { : t } \big )$ . First, we can understand $P _ { S G [ \theta ] } ( \cdot | \cdot ) = \hat { R } ( \cdot | \cdot )$ as a fixed estimator of the conditional distribution $R \big ( z _ { t } | z _ { : t } \big )$ , since it is only trained via the first term which attains a minimum if the conditional distributions match $P _ { S G [ \theta ] } ( \cdot | \cdot ) = R ( \cdot | \cdot )$ . Note, that this requires to train $P _ { \theta }$ on a faster timescale than the encoders. Some stopgrad based approaches ensure this heuristically by employing exponentially moving average encoder targets in the goal function (Grill et al., 2020; Assran et al., 2023). Note also that there is no stopgrad on the average, since R is not parametrized directly, but is defined through sampling x and mapping from x to z. We thus see that the second term is a sampling-based estimator of the conditional entropy (see also Section A.5.4)

$$
\left\langle - \log P _ {S G [ \theta ]} (z _ {t} | S G [ z _ {: t} ]) \right\rangle_ {R (z _ {t}, z _ {: t})} = \left\langle - \log \hat {R} (z _ {t} | S G [ z _ {: t} ]) \right\rangle_ {R (z _ {t}, z _ {: t})} \approx H _ {R} [ z _ {t} | S G [ z _ {: t} ] ]. \tag {55}
$$

The stopgrad approach can therefore be seen to approximately maximize conditional entropy via $z _ { t }$ . We can gain more insight by rewriting $H _ { R } [ z _ { t } | S G [ z _ { : t } ] ] = H _ { R } [ z _ { t } ] - I _ { R } [ z _ { t } , S G [ z _ { : t } ] ]$ . This shows that, in fact, the single timestep entropy is maximized fully, while the term that is only partially minimized is the (as we have argued, typically inconsequential) MI term. This suggests that in principle it would also be feasible to train a separate (ideally, less constrained) predictor $\hat { R } _ { \phi } ( { \boldsymbol { z } } _ { t } | { \boldsymbol { z } } _ { : t } )$ with parameters $\phi$ on a faster timescale and use this fixed predictor to estimate the conditional entropy. We leave this to be explored by future research.

BYOL/SimSiam We first apply this idea to models in the nontemporal setting. To derive the goal function of BYOL (Grill et al., 2020) and SimSiam (Chen et al., 2020) we make the choices

$$
\begin{array}{l} P _ {\theta} (z) = R (z), \\ P _ {\theta} (z ^ {\prime} | z) = \frac {1}{Z} \exp (\tau p (z) ^ {T} z ^ {\prime}) , \tag {56} \\ \end{array}
$$

where $p ( \cdot )$ is the predictor, and latent variables are normalized to lie on the sphere. Here an empirical prior on latent variables is assumed, similar to the model in Aitchison & Ganev (2024). Future work might explore models with a proper prior to reduce the need for additional model regularization (Grill et al., 2020). With the empirical prior $\mathcal { F } _ { \mathrm { I } }$ LDM simplifies to

$$
\mathcal {F} _ {\mathrm{LDM}} = - D _ {K L} \left[ R \left(z ^ {\prime} | z\right) R (z) \| P _ {\theta} \left(z ^ {\prime} | z\right) R (z) \right]
$$

$$
= - \left\langle D _ {K L} \left[ R \left(z ^ {\prime} \mid z\right) \| P _ {\theta} \left(z ^ {\prime} \mid z\right) \right] \right\rangle_ {R (z)} \tag {57}
$$

$$
= \langle \log P _ {\theta} (z ^ {\prime} | z) \rangle_ {R (z, z ^ {\prime})} + H _ {R} [ z ^ {\prime} | z ].
$$

Here we employ the same trick as in A.6.2 and replace the conditional entropy with the stopgrad entropy estimator. This recovers the original goal function

$$
\mathcal {F} _ {\mathrm{LDM}} \approx \langle \log P _ {\theta} (z ^ {\prime} | z) \rangle_ {R (z, z ^ {\prime})} - \langle \log P _ {S G [ \theta ]} (z ^ {\prime} | S G [ z ]) \rangle_ {R (z, z ^ {\prime})}
$$

$$
\stackrel {\partial} {=} \left\langle \log P _ {\theta} (S G [ z ^ {\prime} ] | z) \right\rangle_ {R (z, z ^ {\prime})} \tag {58}
$$

$$
\propto \left\langle \tau p (z) ^ {T} S G [ z ^ {\prime} ] \right\rangle_ {R (z, z ^ {\prime})}.
$$

Temporal Gaussian models The same argument can be made for temporal models, or models where the predictor is conditioned on multiple variables $z _ { : t }$ (Bardes et al., 2024; Assran et al., 2025; Mohammadi et al., 2025). To derive these methods we make the choices

$$
\begin{array}{l} P _ {\theta} (z _ {0}) = R (z _ {0}), \\ \begin{array}{l} P _ {\theta} \left(z _ {0}\right) = N \left(z _ {0}\right), \\ P _ {\theta} \left(z _ {t} \mid z _ {: t}\right) = \mathcal {N} \left(z _ {t}; \mu = p \left(z _ {: t}\right), \Sigma = \sigma^ {2} I\right), \end{array} \tag {59} \\ \end{array}
$$

where $p ( \cdot )$ now is a temporal/multi-input predictor, specified, e.g., by an RNN. With the same argument as above we find the goal function

$$
\begin{array}{l} \mathcal {F} _ {\mathrm{LDM}} = \sum_ {t} \left\langle \log P _ {\theta} (z _ {t} | z _ {: t}) \right\rangle_ {R (z _ {t}, z _ {: t})} + H _ {R} [ z _ {t} | z _ {: t} ] \\ \approx \left\langle \sum_ {t} \log P _ {\theta} (z _ {t} | z _ {: t}) \rangle_ {R (z, z ^ {\prime})} - \log P _ {S G [ \theta ]} (z _ {t} | S G [ z _ {: t} ]) \right\rangle_ {R (\boldsymbol {z})} \\ \stackrel {\partial} {=} \left\langle \sum_ {t} \log P _ {\theta} (S G [ z _ {t} ] | z _ {: t}) \right\rangle_ {R (\boldsymbol {z})} \tag {60} \\ \propto \left\langle - \sum_ {t} \frac {1}{2 \sigma^ {2}} \left\| S G [ z _ {t} ] - p (z _ {: t}) \right\| ^ {2} \right\rangle_ {R (\pmb {z})}. \\ \end{array}
$$

# A.7. Proofs

# A.7.1. IDENTIFIABILITY OF PREDICTIVE MODEL

Our goal is to show that under a predictive Gaussian model LDM recovers the original latent variables up to trivial transformations.

Definition 1 (Data generation process). We start by defining the true latent variable distribution

$$
P (\tilde {z}) = \prod_ {t} P \left(\tilde {z} _ {t} \mid \tilde {z} _ {: t}\right) \tag {61}
$$

$$
P (\tilde {z} _ {t} | \tilde {z} _ {: t}) = \mathcal {N} (\tilde {z} _ {t}; p (\tilde {z} _ {: t}), \Sigma).
$$

Note that here p and Σ can be time dependent, which we do not write down explicitly for compactness. We assume an injective data generation process g, which together with the latent distribution defines the data distribution.

Definition 2. The model is defined by the encoder f and the data distribution. This leads to the transformation between true and recovered latent variables $h = f \circ g .$ . Further, by assumption of a sufficiently flexible encoder $f ,$ after LDM to the form of (61) the latent distribution has the form

$$
R (\boldsymbol {z}) = \prod_ {t} R (z _ {t} | z _ {: t}) \tag {62}
$$

$$
R (z _ {t} | z _ {: t}) = \mathcal {N} (z _ {t}; p _ {R} (z _ {: t}), \Sigma_ {R}).
$$

Note that, technically speaking, $p _ { R }$ is the predictor that is induced by f . This is different from the predictor that would be learned in the target distribution of the model $P _ { \theta } ( z )$ , but for a sufficiently flexible learned predictor they will be equal after LDM, and we don’t make a difference here.

# Theorem 1 (restated). Assuming the following assumptions hold:

(i) (Data generation process) Data is generated by the predictive latent process Eq. (61) with invertible covariance Σ and a differentiable and injective data generation function $g : \mathbb { R } ^ { n }  \mathbb { R } ^ { m }$ .   
(ii) (Distribution matching) The model is defined through a sufficiently flexible encoder $f : \mathbb { R } ^ { m }  \mathbb { R } ^ { n }$ to achieve perfect LDM (i.e., reach form 62), and f is differentiable and invertible on the image of g, that is, $h = f \circ g$ is invertible.   
(iii) (Predictor covers the latent space) The Jacobian $J _ { p _ { R } } ( h ( z _ { : t } ) )$ has full row rank, equal to the latent dimension n (and so does $J _ { p } ( { \boldsymbol { z } } _ { : t } ) )$ .

Then, distribution matching recovers the true latent variables up to affine transformations, that is, if the form of R matches the form of $P ,$ the transforming function between latent variables is affine $h ( z _ { t } ) = A z _ { t } + c .$

Note 1. We only require h to be invertible, but not f in general. This is important, since typically the data dimension is larger than the latent dimension $m > n ,$ , making it impossible for f to be invertible on the whole set $\mathbb { R } ^ { m }$ .

Note 2. Assumption (iii) is required to prevent unconstrained dimensions in latent space in the last step of the proof. It is related, but not equivalent to the predictor being invertible—e.g., if the predictor depends only on the last step $p _ { R } ( z _ { : t } ) = p _ { R } ( z _ { t - 1 } )$ , i.e., $J _ { p _ { R } } ( h ( z _ { : t } ) )$ is square, then (3) implies $p _ { R }$ is invertible, but for general predictors this assumption is a weaker constraint.

Proof. Since R is completely defined by $h = f \circ g$ and P , and h is invertible and differentiable (Assumptions i & ii), we can write R based on the change of variables formula (note that we switch from z˜ to z to simplify notation)

$$
\log R (h (\boldsymbol {z})) + \log | J _ {h} (\boldsymbol {z}) | = \log P (\boldsymbol {z})
$$

$$
\log R (h (\boldsymbol {z})) + \sum_ {t} \log | J _ {h} (z _ {t}) | = \log P (\boldsymbol {z}) \tag {63}
$$

$$
\sum_ {t} \log R (h (z _ {t}) | h (z _ {: t})) + \log | J _ {h} (z _ {t}) | = \sum_ {t} \log P (z _ {t} | z _ {: t}),
$$

where the second line follows since $h ( \cdot )$ is applied to the timesteps individually. We notice that the terms have to match per individual timestep. One way to see this is that, since the last representation $z _ { T }$ only occurs once, the terms for $t = T$ have to match on both sides, and thus by induction (removing matching terms) all terms before. From this, and by using the form of the transformed latent distribution (62) we find

$$
\log R (h (z _ {t}) | h (z _ {: t})) + \log | J _ {h} (z _ {t}) | = \log P (z _ {t} | z _ {: t})
$$

$$
- \log Z _ {R} - \frac {1}{2} \| h (z _ {t}) - p _ {R} (h (z _ {: t})) \| _ {\Sigma_ {R} ^ {- 1}} ^ {2} + \log | J _ {h} (z _ {t}) | = - \log Z _ {P} - \frac {1}{2} \| z _ {t} - p (z _ {: t}) \| _ {\Sigma^ {- 1}} ^ {2}, \tag {64}
$$

where $Z$ are the normalization terms and we are using the weighted norm $\left\| \boldsymbol { x } \right\| _ { A } ^ { 2 } = \boldsymbol { x } ^ { T }$ Ax. Finally, taking derivatives with respect to present and past latent variables, constant and single variable terms drop out and we find

$$
\nabla_ {z _ {t}} \nabla_ {z _ {: t}} \left[ \log R (h (z _ {t}) | h (z _ {: t})) + \log | J _ {h} (z _ {t}) | \right] = \nabla_ {z _ {t}} \nabla_ {z _ {: t}} \log P (z _ {t} | z _ {: t})
$$

$$
\nabla_ {z _ {t}} \nabla_ {z _ {: t}} \| h (z _ {t}) - p _ {R} (h (z _ {: t})) \| _ {\Sigma_ {R} ^ {- 1}} ^ {2} = \nabla_ {z _ {t}} \nabla_ {z _ {: t}} \| z _ {t} - p (z _ {: t}) \| _ {\Sigma^ {- 1}} ^ {2}
$$

$$
\nabla_ {z _ {t}} \nabla_ {z: t} \left[ (h (z _ {t}) - p _ {R} (h (z: t))) ^ {T} \Sigma_ {R} ^ {- 1} (h (z _ {t}) - p _ {R} (h (z: t))) \right] = \nabla_ {z _ {t}} \nabla_ {z: t} \left[ (z _ {t} - p (z: t)) ^ {T} \Sigma^ {- 1} (z _ {t} - p (z: t)) \right] \tag {65}
$$

$$
\nabla_ {z _ {t}} \left[ - (J _ {p _ {R}} (h (z _ {: t})) J _ {h} (z _ {: t})) ^ {T} \Sigma_ {R} ^ {- 1} (h (z _ {t}) - p _ {R} (h (z _ {: t}))) \right] = \nabla_ {z _ {t}} \left[ - J _ {p} (z _ {: t}) ^ {T} \Sigma^ {- 1} (z _ {t} - p (z _ {: t})) \right]
$$

$$
- (J _ {p _ {R}} (h (z _ {: t})) J _ {h} (z _ {: t})) ^ {T} \Sigma_ {R} ^ {- 1} J _ {h} (z _ {t}) = - J _ {p} (z _ {: t}) ^ {T} \Sigma^ {- 1} = \text { const.   w.r.t. } z _ {t} .
$$

In the last line, since $J _ { p _ { R } } ( h ( z _ { : t } ) )$ has full row rank (Assumption iii), and so have $J _ { h } ( z _ { : t } )$ (h is invertible) and $\Sigma ^ { - 1 }$ , it follows that $J _ { h } ( z _ { t } )$ has to be constant w.r.t. $z _ { t }$ . A function with a constant Jacobian is necessarily affine, and we thus conclude that the transformation between true and recovered latent variables is an affine transformation $h ( z _ { t } ) = A z _ { t } + c .$ . □

Note 3. In equation $^ { 6 3 }$ we assume that the in the first timestep $t = 0$ the uncoditional distributions match. Following the approach for conditional distributions, this could be ensured by assuming a Gaussian marginal distribution of $z _ { \mathrm { 0 } }$ and learning the marginal. In practice it is sufficient to assume an empirical prior $P _ { \theta } ( z _ { 0 } ) = R ( z _ { 0 } )$ and hence ignore the first term, since recovery arises only from the conditional distributions.

The result is closely related to the Mazur-Ulam Theorem, which states that any surjective isometry (measure preserving map) between normed spaces is affine (Mazur & Ulam, 1932). Importantly, linearity results entirely from the noise structure of the Gaussian, which is equivalent to the norm of the space in Mazur-Ulam. This argument does not directly extend to variable covariance models—in the Kalman filter with input-dependent covariance, recovery likely stems from the constrained linear predictor.

The result generalizes the results of Zimmermann et al. (2021) and (most closely related) Laiz et al. (2025) under weaker assumptions. It does not only hold for a particular goal function, but for any algorithm that performs LDM while forcing the encoder to be injective. This could in principle also be performed via the reverse KL divergence, with the caveats mentioned in A.2.1, or other divergences.

Finally, the affinity of h leads to a straightforward followup observation.

Corollary 1. Under the assumptions of Theorem 1, and since the transformation function between latent variables is affine $h ( z _ { t } ) = A z _ { t } + c _ { \mathrm { ~ } }$ , it follows that the learned predictor $p _ { R }$ is an affine function of the true predictor $p ,$ i.e., $p _ { R } ( h ( z _ { : t } ) ) = A p ( z _ { : t } ) + c = h ( p ( z _ { : t } ) )$ .

Proof. The true conditional distribution is $\boldsymbol { z } _ { t } \sim \mathcal { N } ( \boldsymbol { p } ( \boldsymbol { z } _ { : t } ) , \Sigma )$ . Applying the affine transformation property of Gaussian random variables, the distribution of the transformed variable h(zt) is

$$
h (z _ {t}) \sim \mathcal {N} (A p (z _ {: t}) + c, A \Sigma A ^ {T}). \tag {66}
$$

By definition (62), the model parametrizes this distribution as $\mathcal { N } ( p _ { R } ( h ( \boldsymbol { z } _ { : t } ) ) , \Sigma _ { R } )$ . Matching the means of these two equivalent distributions yields

$$
p _ {R} (h (z _ {: t})) = A p (z _ {: t}) + c = h (p (z _ {: t})). \tag {67}
$$

Remark 1. While these proofs concern the relation of latent variables $\tilde { z } _ { t } = f ( x _ { t } )$ to true latent variables $z _ { t } ,$ , they do not tell us about the relation between the variables within the predictor. These are, for example, the ‘hidden’ latent variables $h _ { t }$ of the Kalman filter (cf. Fig. 4), or the hidden variables in an RNN that are commonly used for downstream tasks (Oord et al., 2018). Here we remark that with a (partially) linear relation between ‘hidden’ and ‘observed’ latent variables $( h _ { t }$ and $z _ { t }$ respectively), and if the transformation function h between true and recovered ‘observed’ latent variables is affine, then there exists a (partially) affine relation between true and recovered ‘hidden’ variables.

Note 4. Partially linear means that if we train the model based on a linear predictor M from, $\mathrm { e . g . , } h _ { t - 1 }$ to $z _ { t }$ , then only the row space of M in $h _ { t - 1 }$ is guaranteed to have a linear relation to $z _ { t }$ .

Proof. The composition of affine functions is affine.

Another straightforward insight is that theorem 1 can be extended to include models similar to CPC under the assumption that the MI term can be ignored (although even then CPC with the InfoNCE loss does not fully satisfy the assumptions, Aitchison & Ganev, 2024) with latent variables on the sphere.

Theorem 2 (restated). (Affine identifiability of vMF predictive model) Under the assumptions (i)–(iii) of Theorem 1, and a von Mises-Fisher latent predictive distribution $\begin{array} { r } { P ( z _ { t } | z _ { : t } ) = \frac { 1 } { Z _ { P } } \exp ( \beta _ { P } z _ { t } ^ { T } p ( z _ { : t } ) ) } \end{array}$ ) and a matching model, the transforming function h between latent variables is affine.

Proof. The proof remains the same until Equation 64, where we now find

$$
\log R (h (z _ {t}) | h (z _ {: t})) + \log | J _ {h} (z _ {t}) | = \log P (z _ {t} | z _ {: t})
$$

$$
- \log Z _ {R} + \beta_ {R} h (z _ {t}) ^ {T} p _ {R} (h (z _ {: t})) + \log | J _ {h} (z _ {t}) | = - \log Z _ {P} - \beta_ {P} z _ {t} ^ {T} p (z _ {: t}). \tag {68}
$$

After differentiation with respect to $z _ { t }$ and $z _ { : t }$ we are left with

$$
\beta_ {R} \left(J _ {p _ {R}} \left(h \left(z: t\right)\right) J _ {h} \left(z: t\right)\right) ^ {T} J _ {h} \left(z _ {t}\right) = \beta_ {P} J _ {p} \left(z: t\right) ^ {T} = \text { const.   w.r.t. } z _ {t} \tag {69}
$$

and the same conclusion follows.

# A.7.2. FOLDING ENTROPY

In this section we outline a formal proof of the folding entropy formula, in order to provide additional intuition. The formula holds under the following condition:

Definition 3 (Piecewise invertible). An encoder is piecewise invertible on M if

$$
f (x) = \left\{ \begin{array}{l l} f _ {1} (x) & \text { if } x \in \mathcal {X} _ {1} \\ f _ {2} (x) & \text { if } x \in \mathcal {X} _ {2} \\ & \vdots \end{array} \right. \tag {70}
$$

and for all $\mathcal { X } _ { i } \subset \mathcal { M } , x \in \mathcal { X } _ { i } : f _ { i } ( x )$ invertible.

Note 5. For a differentiable (Lipschitz) encoder with nonzero singular values piecewise invertibility automatically holds. Since nonzero singular values are also encouraged by entropy maximization, and functions defined by neural networks are (mostly) differentiable, we can assume that during learning networks in LDM are piecewise invertible on the data manifold.

Definition 4 (Folding entropy). For an encoder $f$ and distribution P on M, the folding entropy is

$$
H _ {P} [ x | f (x) ] := - \left\langle \log P (x | f (x)) \right\rangle_ {P (x)}, \tag {71}
$$

where $P ( x | f ( x ) = z )$ is the conditional distribution over preimages. The folding entropy satisfies $H _ { P } [ x | f ( x ) ] \geq 0$ , with equality if and only if $f$ is injective on supp(P ).

Remark 2. The folding entropy quantifies information lost due to non-injectivity: when $f$ maps distinct points to the same latent representation, uncertainty about the original point given its image increases. For injective $f ,$ the conditional satisfies $P ( x | f ( x ) ) = 1$ everywhere, so $H _ { P } [ x | f ( x ) ] = 0$ and the formula reduces to the standard manifold change of variables. Inversely, if the folding entropy is nonzero $H _ { P } [ x | f ( x ) ] > 0$ it captures information lost when $f$ collapses distinct points.

While the following formula has been proven previously by Geiger & Kubin (2012), we here give a different proof to provide intuition.

Lemma 1 (Folding entropy decomposition). Let $f : \mathcal { M } \to \mathbb { R } ^ { n }$ be a differentiable encoder with nonzero singular values, hence the Jacobian determinant of f on M is non-zero $| J _ { f } ( x ) J _ { f } ( x ) ^ { T } | ^ { 1 / 2 } > 0$ (required to avoid infinities). Let $R ( z ) = \langle \delta ( z - f ( x ) ) \rangle _ { P ( x ) }$ be the pushforward of P under $f .$ . Then

$$
H _ {R} [ z ] = H _ {P} [ x ] + \left\langle \frac {1}{2} \log | J _ {f} (x) J _ {f} (x) ^ {T} | \right\rangle_ {P (x)} - H _ {P} [ x | f (x) ]. \tag {72}
$$

Proof. We first find an expression of the latent distribution $R ( f ( x ) )$ . Define the local volume-adjusted density $r ( x ) : =$ $P ( x ) / | J _ { f } ( x ) J _ { f } ( x ) ^ { T } | ^ { 1 / 2 }$ , which accounts for the change of volume element under $f$ (Krantz & Parks, 2008, p. 125). Given the assumption of piecewise invertible $f ,$ the pushforward density at $z \in f ( { \mathcal { M } } )$ is

$$
R (z) = \sum_ {x \in f ^ {- 1} (z)} r (x), \tag {73}
$$

where the sum runs over all preimages of z. The conditional distribution over preimages is then the local density normalized by the contributions of all preimages

$$
P (x \mid f (x)) = \frac {r (x)}{Z} = \frac {r (x)}{\sum_ {x ^ {\prime} \in f ^ {- 1} (f (x))} r \left(x ^ {\prime}\right)} = \frac {r (x)}{R (f (x))}. \tag {74}
$$

Rearranging gives $R ( f ( x ) ) = r ( x ) / P ( x | f ( x ) )$ , and substituting the definition of $r ( x )$ we get the desired expression

$$
R (f (x)) = \frac {P (x)}{\left| J _ {f} (x) J _ {f} (x) ^ {T} \right| ^ {1 / 2} P (x | f (x))}. \tag {75}
$$

We can compute the latent entropy by first taking logarithms and then taking the average with respect to x

$$
\log R (f (x)) = \log P (x) - \frac {1}{2} \log | J _ {f} (x) J _ {f} (x) ^ {T} | - \log P (x | f (x)), \tag {76}
$$

leading to

$$
\begin{array}{l} H _ {R} [ z ] = - \left\langle \log R (f (x)) \right\rangle_ {P (x)} (77) \\ = - \left\langle \log P (x) \right\rangle_ {P (x)} + \left\langle \frac {1}{2} \log | J _ {f} (x) J _ {f} (x) ^ {T} | \right\rangle_ {P (x)} + \left\langle \log P (x | f (x)) \right\rangle_ {P (x)} \\ = H _ {P} [ x ] + \left\langle \frac {1}{2} \log | J _ {f} (x) J _ {f} (x) ^ {T} | \right\rangle_ {P (x)} - H _ {P} [ x | f (x) ], (78) \\ \end{array}
$$

where we identified $- \langle \log P ( x ) \rangle _ { P ( x ) } = H _ { P } [ x ] { \mathrm { ~ a n d } } - \langle \log P ( x | f ( x ) ) \rangle _ { P ( x ) } = H _ { P } [ x | f ( x ) ] .$ .

![](images/f3621e8237b2a5483192d84583c77913c18b7f1b339beb5ea7c957ce893a38ee.jpg)

# A.8. Investigation of categorical model with probabilistic encoder

We also wanted to understand if LDM in latent space allows for uncertainty in representations of single observations. To this end we used the same non-temporal goal functions as before, but defined a recognition distribution $R ( z | x )$ that is not a simple delta peak.

Categorical model To enable analytical tractability, we defined a simple categorical model that assigns individual probabilities to n categories via a softmax encoder We make the choices (with n categories k)

$$
R (z = k | x) = p _ {k} (x) = \frac {1}{Z (x)} \exp (f _ {k} (x))
$$

$$
P _ {\theta} (z = k) = \frac {1}{N} \tag {79}
$$

$$
P _ {\theta} (z ^ {\prime} = k ^ {\prime} | z = k) = \frac {1}{Z} \exp (\beta \mathbb {I} [ k ^ {\prime} = k ]).
$$

This model assumes that if z and $z ^ { \prime }$ are encodings of related samples, they are assigned to the same category with probability $p _ { \theta } = \exp ( \beta ) / ( \exp ( \beta ) + n - 1 )$ ). The likelihood term becomes

$$
\begin{array}{l} \left\langle \sum_ {k = 1} ^ {n} \sum_ {k ^ {\prime} = 1} ^ {n} \log P _ {\theta} (z = k | z ^ {\prime} = k ^ {\prime}) R (z = k | x) R (z ^ {\prime} = k ^ {\prime} | x ^ {\prime}) \right\rangle_ {P _ {\text { data }} (x, x ^ {\prime})} \\ \propto \left\langle \sum_ {k = 1} ^ {n} \sum_ {k ^ {\prime} = 1} ^ {n} \beta \mathbb {I} [ k ^ {\prime} = k ] \frac {1}{Z (x)} \exp (f _ {k} (x)) \frac {1}{Z (x ^ {\prime})} \exp (f _ {k ^ {\prime}} (x ^ {\prime})) \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})} \tag {80} \\ = \beta \left\langle \sum_ {k = 1} ^ {n} p _ {k} (x) p _ {k} (x ^ {\prime}) \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})}. \\ \end{array}
$$

The entropy term can be computed analytically

$$
\begin{array}{l} H _ {R} [ z ] = - \sum_ {k = 1} ^ {n} R (z = k) \log R (z = k) \\ \approx - \sum_ {i} \sum_ {k = 1} ^ {n} \frac {1}{Z (x _ {i})} \exp (f _ {k} (x _ {i})) \log \left[ \sum_ {j} \frac {1}{Z (x _ {j})} \exp (f _ {k} (x _ {j})) \right]. \tag {81} \\ \end{array}
$$

This results in the goal function

$$
\mathcal {F} = \beta \sum_ {k = 1} ^ {n} \left\langle R (z = k | x) R (z ^ {\prime} = k | x ^ {\prime}) \right\rangle_ {P _ {\mathrm{data}} (x, x ^ {\prime})} + H _ {R} [ z, z ^ {\prime} ] . \tag {82}
$$

Simulations We tested the ability of the model to learn probabilistic representations on MNIST, by providing two different images of the same digit as input to the SSL algorithm. We assumed $n = 1 0$ categories, and that two presented digits can be detected to be in the same category with probability $p _ { \theta } = 0 . 9 9$ by choosing $\beta$ in the model accordingly (Fig. A8A). To prevent model collapse in the beginning of learning, we annealed $p _ { \theta }$ over epochs to the final value, starting from $p _ { \theta } = 0 . 8$ After learning and sorting learned categories, the algorithm recovered the digit identity near perfectly with an accuracy of 0.99 (Fig. A8B). In contrast, with additional MI maximization the model would collapse categories under the correct latent model. Instead, we had to assume a misspecified latent model with $p _ { \theta } = 0 . 8$ that effectively counteracts MI maximization to achieve reliable representation learning of digit identities.

We also found that representations with $\mathcal { F } _ { \mathrm { L D M } }$ had meaningful quantification of uncertainty: Digits that were assigned to a single category (low entropy encodings) were consistently highly stereotypical, while digits with uncertain encoding (high entropy) were outliers without clear identity (Fig. A8C,D). In comparison, with additional MI maximization, the algorithm predominantly forms extremely low-entropy representations (Fig. A9). Evidently, here MI is not maximized through the presence of an invertible encoder, and thus plays a non-neglegible role in shaping representations.

LDM is able to produce probabilistic encodings that respect epistemic uncertainty, which has previously been demonstrated by Kirchhof et al. (2023). Notably, while Kirchhof et al. (2023) employed the reverse KL divergence, which requires reparametrization sampling to evaluate the goal function, the approach here is simpler and does not require sampling. Crucially, however, the motivation through the data likelihood is based on the assumption of invertible encoders, which clearly conflicts with probabilistic encoders. Indeed, in preliminary experiments we found that it is not easily possible to optimize model parameters θ in the same way as it is for deterministic encoders, which likely is caused by this mismatch. A theoretical framework for how flexible probabilistic models based on latent space prediction are related to LDM might be provided via more complex theoretical approaches in the future, e.g., recognition parametrized models (Walker et al., 2023; Hromadka et al., 2025).

# B. Simulation details

Details on parameters and precise implementation can be found in the simulation code at https://github.com/fmi-basel/latent distribution matching.

Learning of image representations. For experiments we extended the solo-learn library provided by da Costa et al. (2022), which uses ResNet-18 as standard encoder and 1000/400 epochs of training for CIFAR/Imagenet-100. Note that for Imagenet-100 we used the classes as used by Tian et al. (2020), which were different than those used by da Costa et al. (2022). For latents on the plane we used a Gaussian conditional model as described in A.6.1; for latents on the sphere we used a vMF conditional model as described in A.6.1. We implemented the entropy estimators as described in A.5, with the following details. For contrastive entropy estimation via KDE (A.5.1) we used vMF kernels on the sphere, leading to the same loss as in Chen et al. (2020), and Gaussian kernels on the plane. For non-contrastive entropy estimation (A.5.3) on the plane, for comparability, we used the variance-covariance regularization as originally proposed in Bardes et al. (2021). On the sphere we used the log-det expansion from A.5.3. For kNN entropy estimation (A.5.2) we used the euclidean metric (p = 2), chose k = 3 and discarded the upper 10% of kNN, which we considered outliers and already well separated.

Learning of a simple dynamical system with Kalman-based SSL. We randomly sampled 800 training sequences consisting of trajectories and noise and trained for 20 epochs—note that only the simulations in Fig. A4 used noise in the trajectories. Synthetic videos of 10 by 10 pixel resolution were then created according to the process outlined in the main text. We specified the encoder through a simple MLP with ReLU activation and one hidden layer with 100 units. We specified the Kalman filter with diagonal covariances $\Sigma _ { A }$ and $\Sigma _ { D }$ . D was specified, without loss of generality, as a fixed diagonal matrix, with or without zeros on the diagonal, depending on whether ’hidden’ latent states were desired or not. For both models using stopgrad (A.6.2) and kNN (A.5.2) entropy estimation, we learned the Kalman filter on faster timescales than the encoders. For kNN entropy estimation, again, we used the euclidean metric (p = 2), chose k = 3 and discarded the upper 10% of kNN, which we considered outliers and already well separated.

Modeling hippocampal activity dynamics with Kalman-based SSL. We modeled the data of the hc-11 dataset (Grosmark & Buzsaki ´ , 2016) based on the pre-processing as described in Schneider et al. (2023), i.e., model inputs were spike-counts of 120 neurons in 25 ms bins. For one epoch we sampled 100000 windows of length ∼10 s from the full spike-train (about 45 min), training for 16 epochs. For the Kalman filter we chose an 8D observation and 16D latent space. For the encoders we used the same setup as before. We used kNN entropy estimation, with the same parameters as before. For a linear position predictor the prediction distribution can be computed analytically. To find 95% confidence intervals for MLP position prediction, for each timestep we produced 1000 samples of the latent state distribution $\mathcal { N } _ { h _ { t } } ( \mu _ { h _ { t } } , \Sigma _ { h _ { t } } )$ as given by the Kalman filter backbone. We then computed the predicted positions for these samples through the MLP (trained by predicting position from latent mean $\mu _ { h _ { t } } )$ , and found the 95% sampling based confidence intervals.

Nonlinear systems identification with predictive distribution matching. We randomly sampled 10000 sequences and trained for 300 epochs (although convergence was observed much earlier). The encoder was specified as an MLP with ReLU activation, 10 hidden layers, and 200 hidden units per layer. The predictor, mapping from all previous 2 dimensional encodings to the next encoding, was an LSTM RNN with an MLP predictor head. The LSTM had 1 layer with 10 hidden units. The predictor head was an MLP with ReLU activation, 2 hidden layers, and 20 hidden units. In the main text we used kNN entropy estimation, with the same parameters as before. In the Appendix Fig. A6 we used the stopgrad entropy estimator approximation as defined before, the LogDet entropy estimator with the slogdet function in pytorch, and the KDE estimator with bandwidth 1. For the second experiment (Fig. A7) we used the same setup, training on 100000 samples for 20 epochs.

Probabilistic representations in categorical SSL. We used ResNet-18 as encoder, with an additional soft-max output layer, training for 30 epochs.

# C. Supplementary figures and tables

Table A1. Validation accuracy for linear probing on CIFAR-10, CIFAR-100, SVHN, and Imagenet-100. We used a standard SSL image pretraining setup as described by da Costa et al. (2022). MI max. denotes whether $\mathcal { F } _ { \mathrm { L D M } }$ (◦) or $\mathcal { F } _ { \mathrm { M I } }$ (•) was used. Standard deviation was calculated based on three runs. Note that Imagenet-100 classes were selected according to Tian et al. (2020), which were different than those used by da Costa et al. (2022), accounting for the difference in performance. The implementation of VICReg (Plane-LogDet-•) is the same as in da Costa et al. (2022) and serves as a reference. 

<table><tr><td rowspan="2">Space</td><td colspan="2">Model</td><td colspan="2">CIFAR-10 Acc.</td><td colspan="2">CIFAR-100 Acc.</td><td colspan="2">SVHN Acc.</td><td colspan="2">Imagenet-100 Acc.</td></tr><tr><td>Entropy est.</td><td>MI max.</td><td>Top-1</td><td>Top-5</td><td>Top-1</td><td>Top-5</td><td>Top-1</td><td>Top-5</td><td>Top-1</td><td>Top-5</td></tr><tr><td>Plane</td><td>Contr.</td><td>○</td><td>91.87±0.06</td><td>99.73±0.00</td><td>65.28±0.28</td><td>89.06±0.24</td><td>92.15±0.17</td><td>99.07±0.13</td><td>72.60</td><td>92.36</td></tr><tr><td>Plane</td><td>Contr.</td><td>●</td><td>92.09±0.08</td><td>99.76±0.04</td><td>65.29±0.15</td><td>88.96±0.25</td><td>92.20±0.11</td><td>99.11±0.03</td><td>72.68</td><td>92.04</td></tr><tr><td>Plane</td><td>kNN</td><td>○</td><td>92.07±0.17</td><td>99.68±0.05</td><td>65.63±0.36</td><td>88.93±0.06</td><td>92.75±0.13</td><td>99.20±0.04</td><td>74.32</td><td>93.36</td></tr><tr><td>Plane</td><td>kNN</td><td>●</td><td>91.89±0.05</td><td>99.69±0.03</td><td>65.85±0.38</td><td>88.94±0.17</td><td>92.85±0.11</td><td>99.19±0.04</td><td>73.68</td><td>92.84</td></tr><tr><td>Plane</td><td>LogDet</td><td>○</td><td>92.06±0.05</td><td>99.74±0.02</td><td>69.47±0.21</td><td>91.27±0.15</td><td>92.79±0.13</td><td>99.20±0.04</td><td>75.92</td><td>93.32</td></tr><tr><td>Plane</td><td>LogDet</td><td>●</td><td>91.94±0.05</td><td>99.74±0.02</td><td>68.62±0.06</td><td>90.77±0.19</td><td>92.72±0.09</td><td>99.13±0.04</td><td>74.72</td><td>93.40</td></tr><tr><td>Sphere</td><td>Contr.</td><td>○</td><td>90.93±0.13</td><td>99.74±0.03</td><td>64.83±0.19</td><td>88.63±0.21</td><td>91.59±0.05</td><td>98.92±0.05</td><td>72.12</td><td>92.12</td></tr><tr><td>Sphere</td><td>Contr.</td><td>●</td><td>91.38±0.22</td><td>99.73±0.03</td><td>66.01±0.24</td><td>89.47±0.14</td><td>92.33±0.12</td><td>99.05±0.04</td><td>73.08</td><td>92.72</td></tr><tr><td>Sphere</td><td>kNN</td><td>○</td><td>90.22±0.04</td><td>99.68±0.02</td><td>64.31±0.07</td><td>88.00±0.09</td><td>92.27±0.12</td><td>98.99±0.04</td><td>72.64</td><td>92.16</td></tr><tr><td>Sphere</td><td>kNN</td><td>●</td><td>89.96±0.16</td><td>99.63±0.04</td><td>64.50±0.18</td><td>87.82±0.12</td><td>92.16±0.03</td><td>99.05±0.02</td><td>73.28</td><td>92.28</td></tr><tr><td>Sphere</td><td>LogDet</td><td>○</td><td>91.41±0.21</td><td>99.77±0.02</td><td>65.40±0.18</td><td>89.31±0.11</td><td>92.57±0.02</td><td>99.09±0.03</td><td>73.00</td><td>93.00</td></tr><tr><td>Sphere</td><td>LogDet</td><td>●</td><td>91.20±0.11</td><td>99.73±0.01</td><td>65.37±0.16</td><td>89.34±0.22</td><td>92.54±0.04</td><td>99.09±0.01</td><td>72.28</td><td>92.40</td></tr><tr><td>Simplex</td><td>Contr.</td><td>●</td><td>90.23±0.13</td><td>99.69±0.03</td><td>62.65±0.22</td><td>86.98±0.12</td><td>92.21±0.07</td><td>99.03±0.05</td><td>70.50</td><td>91.97</td></tr></table>

A   
![](images/649e6700008ccb995bf5e6134b2ee5b3fbf322529b414889098bfb8c8587ada4.jpg)

<details>
<summary>line</summary>

| Epoch | Rank of Jacobian |
|-------|------------------|
| 0     | ~500             |
| 100   | ~350             |
| 200   | ~300             |
| 300   | ~280             |
| 400   | ~270             |
| 500   | ~260             |
</details>

B   
![](images/a171fce22b690db73f839880ccbf1a929f460b0ca6132036bc0ac1b9c2b76c4e.jpg)

<details>
<summary>line</summary>

| Epoch | Grad. Cos. Sim. |
|-------|-----------------|
| 0     | 1.0             |
| 400   | 0.5             |
| 800   | 0.2             |
</details>

![](images/a2e9e89f03b9777bf84eac806b73a6f93b7b9c6c805f73eb158cc2e8d441db4c.jpg)

![](images/7ced3028c706d2dcba3b086a1d553c6dc1b57e93ef7ee8deda9284ddf43dd546.jpg)  
Figure A2. A The rank of encoder Jacobian reaches similar levels for all approaches (here, averaged over data-points on CIFAR-10). The rank is regularized by the entropy term, since without the entropy it collapses to zero. Note that even if the final rank exceeds the dimensionality of the data manifold (in this case estimated to be less than 100 dimensions) this does not imply local invertibility. However, we tested for local invertibility on a toy dataset with known data generation process (detailed in Fig. A7), by computing the rank of the combined generator-encoder function. This function was consistently full rank, implying local invertibility. B Similarity of single and joint entropy gradients of $\mathcal { F } _ { \mathrm { M I } }$ and $\mathcal { F } _ { \mathrm { L D M } }$ , respectively, for CIFAR-100. Single and joint entropy gradients point into consistent directions throughout learning, with the exception of the LogDet estimator on the plane for $\mathcal { F } _ { \mathrm { L D M } }$ . That is, when training with $\hat { H } [ z , z ^ { \prime } ]$ in $\mathcal { F } _ { \mathrm { L D M } } .$ , the estimated entropy $\hat { H } [ z , z ^ { \prime } ]$ has slightly opposing gradient to ${ \hat { H } } [ z ]$ towards the end of the training (but not the other way around when training with $\mathcal { F } _ { \mathrm { M I } } )$ . The likely reason are the strong assumptions about the shape of the latent distribution by the LogDet estimator.

![](images/590bf5007b9f63bf8fc38359a0e0c9fdfb9e9bea7e406b3add4192291ef60b1f.jpg)

<details>
<summary>line</summary>

| Time | h_t  | Rollout | z_t  |
|------|------|---------|------|
| 0    | 1.0  | 1.0     | 1.0  |
| 5    | 0.8  | 0.9     | 0.7  |
| 10   | -0.2 | -0.3    | -0.4 |
| 15   | 0.6  | 0.7     | 0.5  |
| 20   | 1.0  | 1.1     | 1.0  |
| 25   | 0.8  | 0.9     | 0.7  |
| 30   | -0.2 | -0.3    | -0.4 |
| 35   | 0.6  | 0.7     | 0.5  |
| 40   | 1.0  | 1.1     | 1.0  |
| 45   | 0.8  | 0.9     | 0.7  |
| 50   | -0.2 | -0.3    | -0.4 |
</details>

![](images/60220ea266ad636492af4ece7aff5d91638c064c166b035b722d5cb953ace93d.jpg)

<details>
<summary>line</summary>

| Time | Ground Truth Pos. | Est. Pos. KNN Entropy | Est. Pos. Stopgrad |
|------|-------------------|------------------------|--------------------|
| 0    | ~1.5              | ~1.5                   | ~1.5               |
| 10   | ~-1.5             | ~-1.5                  | ~-1.5              |
| 20   | ~1.5              | ~1.5                   | ~1.5               |
| 30   | ~-1.5             | ~-1.5                  | ~-1.5              |
| 40   | ~1.5              | ~1.5                   | ~1.5               |
| 50   | ~-1.5             | ~-1.5                  | ~-1.5              |
</details>

Figure A3. Square movement task. A A dot moving on a square trajectory is learned to be mapped to a linear dynamical system. The learned system captures the inferred dynamics, demonstrated by simulating a latent trajectory based on the latent state at $i = 2 5$ . We also show the reconstructed input (MLP decoder) and trajectory (linear decoder). Linear decoding does not enable decoding the correct trajectory. B To estimate decoding performance and uncertainty based on a linear decoder, we provide a surrogate true position, which is a sinusoidal with the frequency of the moving dot. Both contrastive and stopgrad predictive SSL find good encodings $\dot { R } ^ { 2 } = 0 . 9 7$ .

A   
![](images/4ce1e5944ebca52aa99a41d2df9ffe9e54449b61bb3253f7c331e4bb121a97a9.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["h_{t-1}"] --> B["h_t"]
    B --> C["z_t"]
    C --> D["x_t"]
    D --> E["Σ_D = f(x_t)"]
    style A fill:#f9f,stroke:#333
    style B fill:#f9f,stroke:#333
    style C fill:#f9f,stroke:#333
    style D fill:#f9f,stroke:#333
    style E fill:#f9f,stroke:#333
```
</details>

B   
![](images/f7e1ce4f734210cf948d3d4db3a1b9ad61e5b2b0ef843f61ad863edd039dc88e.jpg)

![](images/686b2f2a41be0eb33c11f943b8045c6c4ef714b4a8bd28ec75bc972a17ac080c.jpg)

![](images/49d1ef8574672ad91a6f15756e7bad086ad84451dcd5624ad53f12c525e02d89.jpg)

C   
![](images/7e28f1dca65e097b7611199d492a8bee185fecf19b58cd2336db25d5d2a65069.jpg)

<details>
<summary>line</summary>

| Time | h1    | h2    |
|------|-------|-------|
| 0    | ~0.5  | ~0.5  |
| 10   | ~-0.5 | ~-0.5 |
| 20   | ~0.5  | ~0.5  |
| 30   | ~-0.5 | ~-0.5 |
| 40   | ~0.5  | ~0.5  |
| 50   | ~-0.5 | ~-0.5 |
</details>

D   
![](images/0a25917024fd931837f61c131e96d3fc291d0023a143041acfe5e0ab30532212.jpg)

<details>
<summary>line</summary>

| Time | Ground Truth Pos. | Est. SG |
|------|-------------------|---------|
| 0    | 1.0               | 1.0     |
| 10   | -1.0              | -0.5    |
| 20   | 0.5               | 0.8     |
| 30   | -0.5              | -0.2    |
| 40   | 1.0               | 0.7     |
| 50   | -0.5              | -0.3    |
</details>

Figure A4. Kalman SSL with input dependent observation noise. A Observation noise (diagonal $\Sigma _ { D } )$ is estimated with an MLP with softplus head. B Data has randomly chosen noise level every timestep. C The model correctly infers the changing noise level and D finds a good estimate of the latent trajectory in the hidden variables $h _ { t } \ ( \hat { R ^ { 2 } } = 0 . 9 6 )$ .

![](images/a2aa33e5c6da4cbef99f20122a23bfb3e3e137737d1c480c0f39f052b458336a.jpg)

<details>
<summary>line</summary>

| Time [s] | SSL MLP | Est. kNN | Direct dec. |
| -------- | ------- | -------- | ----------- |
| 0        | ~1.5    | ~1.5     | ~1.5        |
| 5        | ~1.5    | ~1.5     | ~1.5        |
| 10       | ~1.5    | ~1.5     | ~1.5        |
| 15       | ~1.5    | ~1.5     | ~1.5        |
| 20       | ~1.5    | ~1.5     | ~1.5        |
| 25       | ~1.5    | ~1.5     | ~1.5        |
</details>

Figure A5. Comparison of position prediction through Kalman-based SSL or directly from the data. Kalman-based SSL makes highly accurate predictions with MLP decoder $( R ^ { 2 } = 0 . 9 8 )$ . With a linear decoder prediction becomes worse since at the turning points real position dynamics are not well approximated by a linear model $( R ^ { 2 } = 0 . 8 8$ , similar problem as in Fig. A3). For direct prediction we use the spikes binned in 25ms time windows, which leads to inferior performance of MLP $( R ^ { 2 } = 0 . 8 8 )$ and Linear $( R ^ { 2 } = ) . 5 7 )$ predictors.

![](images/f4ab7874fa20625eb1b727b7875092537584237f925dbefe82e6dae6b95e32b6.jpg)

<details>
<summary>surface_3d</summary>

| β    | Method                  | z1   | z2   |
|------|-------------------------|------|------|
| 8    | Predictive Stopgrad SSL   | 0.5  | 0.6  |
| 8    | Predictive SSL + LogDet entropy | 0.4  | 0.5  |
| 8    | Predictive SSL + KDE entropy | 0.3  | 0.4  |
| 4    | Generalized Gaussian noise (kNN) | 0.7  | 0.8  |
| 4    | Generalized Gaussian noise (kNN) | 0.6  | 0.7  |
| 4    | Generalized Gaussian noise (kNN) | 0.5  | 0.6  |
| 2    | Generalized Gaussian noise (kNN) | 0.9  | 1.0  |
| 2    | Generalized Gaussian noise (kNN) | 0.8  | 0.9  |
| 2    | Generalized Gaussian noise (kNN) | 0.7  | 0.8  |
| Long Tail | Generalized Gaussian noise (kNN) | 1.1  | 1.2  |
| Long Tail | Generalized Gaussian noise (kNN) | 1.0  | 1.1  |
| Long Tail | Generalized Gaussian noise (kNN) | 0.9  | 1.0  |
| Short Tail | Generalized Gaussian noise (kNN) | 0.8  | 0.9  |
| Short Tail | Generalized Gaussian noise (kNN) | 0.7  | 0.8  |
| Short Tail | Generalized Gaussian noise (kNN) | 0.6  | 0.7  |
| Short Tail | Generalized Gaussian noise (kNN) | 0.5  | 0.6  |
| Short Tail | Generalized Gaussian noise (kNN) | 0.4  | 0.5  |
| Short Tail | Generalized Gaussian noise (kNN) | 0.3  | 0.4  |
| Short Tail | Generalized Gaussian noise (kNN) | 0.2  | 0.3  |
| Short Tail | Generalized Gaussian noise (kNN) | 0.1  | 0.2  |
| Short Tail | Generalized Gaussian noise (kNN) | 0.05 | 0.15 |
The image contains multiple colored regions corresponding to the x and y axes, but no explicit numerical labels are provided in the image.
</details>

Figure A6. Top stopgrad based predictive SSL leads to nonlinear identification of the underlying dynamics, which we exemplify with three learning outcomes (cf. Fig. 5C). The resulting data representations are approximately locally linearlized on the latent plane, and learned dynamics are predictive of future representations (not shown), but, since stopgrad based predictive SSL employs a heuristic entropy estimator (which already assumes the conditional distribution to be Gaussian), they retain nonlinear distortions compared to the true variables. Also, when using the LogDet entropy estimator (instead of kNN in the main paper) affine identification is only approximate. The reason is that for the underlying variables of this particular data the Gaussian assumption of the LogDet estimator does not hold, which thus leads to a biased entropy estimate. Such biased entropy estimates, such as in $\scriptstyle { \hat { \mathrm { V I C R e g } } }$ (Bardes et al., 2021) or SICReg (Balestriero & LeCun, 2025), could also be a problem in practice, even when exact identification is not required, since it can lead to a mismatch between empirical and model conditional distributions (e.g., the model makes Gaussian-distributed predictions, while the empirical conditional latent distribution has a different shape). For KDE entropy estimation we find similarly good results as for the kNN estimator. Bottom Identifiability is robust to limited deviations from Gaussian noise in the conditional distribution of the data generation process. Here we replaced Gaussian noise with Generalized Gaussian noise with long or short tails. Approximate identification persisted for non-isotropic noise as displayed here (notice also the alignment of the recovered manifold with the axes). For isotropic noise, or for multi-modal conditional distributions, we did not observe the same robustness in general (not shown), suggesting that more complex latent models are needed in such cases.

A   
![](images/3e7c0eb6e2185e7e27eeabe2626b72d668029965b13b6e55443bcc8609d8d4c6.jpg)

<details>
<summary>line</summary>

| N | R2 score for position (Line 1) | R2 score for position (Line 2) | R2 score for position (Line 3) | R2 score for position (Line 4) | R2 score for position (Line 5) |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.0 | 1.0 | 0.6 | 0.5 | 0.45 |
| 4 | 0.95 | 0.95 | 0.75 | 0.65 | 0.48 |
| 6 | 0.85 | 0.85 | 0.8 | 0.75 | 0.45 |
| 8 | 0.8 | 0.8 | 0.85 | 0.75 | 0.42 |
| 10 | 0.75 | 0.75 | 0.8 | 0.7 | 0.4 |
</details>

B

![](images/c6b5c55b496dc088857a39420c589fdd24ee5c3006cf195742d4a58c19cd7641.jpg)

<details>
<summary>line</summary>

| N | knn | kde | logdet | stopgrad | random |
| --- | --- | --- | --- | --- | --- |
| 2 | 1.0 | 0.85 | 1.0 | 0.85 | 0.8 |
| 4 | 0.95 | 0.9 | 0.95 | 0.95 | 0.7 |
| 6 | 0.85 | 0.85 | 0.85 | 0.9 | 0.6 |
| 8 | 0.85 | 0.85 | 0.85 | 0.9 | 0.6 |
| 10 | 0.8 | 0.8 | 0.8 | 0.85 | 0.55 |
</details>

Figure A7. Higher-dimensional nonlinear source recovery task. We follow standard practice in identifiability literature (e.g. Khemakhem et al., 2020b), with a nonlinear latent relation and an invertible DNN based generation function. We sample two latent variables $z , z ^ { \prime } ,$ , where $z \sim \mathcal { N } ( 0 , \mathbb { 1 } )$ and $z ^ { \prime } \sim \mathcal { N } ( \mu ( z ) , \alpha \mathbb { 1 } )$ , $\mu ( z ) = R z + 0 . 1$ tanh(Rz), R is a random rotation matrix. 100-dimensional observations are generated from z and $z ^ { \prime }$ via an invertible (LeakyReLU) random 3-layer MLP. A Recovery generally works well in low dimensions N (except for stopgrad models) and deteriorates for $N > 4$ . We suspect two reasons: 1) Higher N require exponentially more samples to cover the latent space (for intuition, the $\mathrm { 1 0 ^ { 5 } }$ samples in $N = 1 0$ dimensions correspond to roughly 3 points per axis in a grid); 2) entropy maximization becomes more challenging, since ’unfolding’ the latent distribution requires pushing the ’creases’ through the distribution itself. This process slows down when there are more creases and dimensions. Still, recovery consistently outperforms randomly initialized models. B To test the second hypothesis, we trained overcomplete models with latent dimensionality $M = 2 { \cdot } N .$ , which allows probability mass to ’slip past’ itself more easily. We find improved recovery and more reliable training for all models, even though now representations could in principle be nonlinearly related to true latents due to the mismatch in dimensionality. Shaded regions denote 95% confidence intervals based on standard error for 5 runs.

A   
![](images/ac2ccf6140df37b6cd0b9fe3a514b0538d866893713299b2a008aeef282f760f.jpg)

B   
![](images/361190e41296a613890cb3bc1f23427a8b0aa5ee744690580c01a5ce5ab43a95.jpg)

<details>
<summary>heatmap</summary>

DM, pθ = 0.99
| | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| True label | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| Prediction | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
</details>

DM+MI, p = 0.99   
![](images/3d76a4520ba83a516ab39c107ba8802a9380e0dd3d269b2858ea5668e54966d2.jpg)

<details>
<summary>heatmap</summary>

| Prediction | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 | 1 | 1 | 1 | 0 | 0 | 0 |
| 7 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 8 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| 9 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
The image displays a heatmap with color intensity (ranging from light beige to dark blue) representing the value of each cell in the matrix. The x-axis is labeled 'Prediction' and the y-axis is labeled 'Value'. There are no labels or additional data series present.
</details>

DM+MI, p = 0.8   
![](images/167f798288bcf23b1e505b73e2283ad50b446ed5fed14caaee716ce464c44a9e.jpg)

<details>
<summary>heatmap</summary>

| Prediction | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| 5 | 0 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 | 0 |
| 6 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 | 0 |
| 7 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| 8 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 0 |
| 9 | 0 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
</details>

MI   
![](images/c002bf27c5947725e7f09ce47a8f28e4df9705226af79690b54f5f301aef55da.jpg)

<details>
<summary>heatmap</summary>

| Prediction | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 1 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 2 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 3 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 4 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 5 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 6 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 7 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 | 1000 |
| 8 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 | 1555 |
| 9 | 875 | 875 | 875 | 875 | 875 | 875 | 875 | 875 | 875 | 875 |
The image contains a heatmap with a color scale ranging from red (low) to blue (high). The x-axis labels are 'Prediction' and the y-axis labels are 'Sample' (e.g., 'Number of Samples'). The color bar indicates the number of samples in each cell.
</details>

C C   
Minimum entropy representations   
![](images/aa3231536bd9755fb74905118cddc6c2cf246aac69aedfc621f33deb7d4537ed.jpg)

<details>
<summary>bar</summary>

| Category | Probability |
|---|---|
| 1 | 0123456789 |
| 7 | 0123456789 |
| 0 | 0123456789 |
| 7 | 0123456789 |
| 7 | 0123456789 |
| 0 | 0123456789 |
</details>

D   
Maximum entropy representations   
![](images/1769eb9a64f7b6a463702bb305953bac760ad95ffc24099ece38ea8dc7b45cb7.jpg)

<details>
<summary>bar</summary>

| Category | Value |
|---|---|
| 0123456789 | 1 |
| 0123456789 | 2 |
| 0123456789 | 3 |
| 0123456789 | 4 |
| 0123456789 | 5 |
| 0123456789 | 6 |
| 0123456789 | 7 |
| 0123456789 | 8 |
| 0123456789 | 9 |
| 0123456789 | 10 |
</details>

Figure A8. SSL based probabilistic representations of MNIST digits. A After digits are encoded with softmax encoders $R ( z | x )$ , the algorithm encourages them to be in the same category with probability pθ. B LDM recovers digit identities after sorting of categories with an accuracy of 0.99. With MI maximization the model has to be mis-specified (low matching probability of $p _ { \theta } = 0 . 8 )$ to prevent collapse of digit identities. MI as an optimization goal alone shows poor results. C Most certain digit encodings. D Most uncertain digit encodings.

![](images/fc7786be5349c086cf09d3d08652f5d0c8bc6cc7134ec4591dc80c4665764ed6.jpg)

<details>
<summary>histogram</summary>

| Log Entropy Bin | Count     |
| --------------- | --------- |
| -10 to -8       | 1         |
| -8 to -6        | 10        |
| -6 to -4        | 100       |
| -4 to -2        | 1000      |
| -2 to 0         | 10000     |
</details>

![](images/3af38a0fc614f122e6ef076db87ecfc1c1d481da4b9f796331b301aaaf656dd3.jpg)

<details>
<summary>histogram</summary>

| Log Entropy Bin | Frequency |
| --------------- | --------- |
| -15 to -12      | 10^4      |
| -12 to -9       | 10^3      |
| -9 to -6        | 10^2      |
| -6 to -3        | 10^2      |
| -3 to 0         | 10^1      |
</details>

![](images/17fce08d6ec41881ed6dbfa3f7b91eebdea17a49c92bd8f61d3fa9a7a0039b14.jpg)

<details>
<summary>bar</summary>

| Log Entropy | Frequency |
| ----------- | --------- |
| -15         | 10000     |
| -10         | 100       |
| -5          | 10        |
| 0           | 100       |
</details>

![](images/06ee9d361ba926e017b5d9512d0e1fd8a89722f5ffc6b1e840c920db327e92a0.jpg)

<details>
<summary>area</summary>

MI
| Log Entropy | Value |
| :--- | :--- |
| -12 | 3000 |
| -10 | 2500 |
| -8 | 2000 |
| -6 | 1500 |
| -4 | 1000 |
| -2 | 800 |
| 0 | 900 |
</details>

Figure A9. Histogram of entropies in digit encodings. LDM alone leads to moderate to high uncertainty in digit encodings, while (additional) MI maximization results in extremely low estimated uncertainties.