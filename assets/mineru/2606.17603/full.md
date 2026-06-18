# Expanding SPHERE-JEPA: A Family of Statistical Regularizers for the Hypersphere

Léo Nicollier

Université Paris-Saclay, CNRS, ENS Paris-Saclay, Centre Borelli, France Advanced Track and Trace

leo.nicollier@gmail.com

Enric Meinhardt-Llopis

Université Paris-Saclay, CNRS, ENS Paris-Saclay, Centre Borelli, France

Max Dunitz

Advanced Track and Trace

Marc Pic

Advanced Track and Trace

Pablo Musé

IIE, Facultad de Ingeniería, Universidad de la Republica, Uruguay Université Paris-Saclay, CNRS, ENS Paris-Saclay, Centre Borelli, France

Gabriele Facciolo

Université Paris-Saclay, CNRS, ENS Paris-Saclay, Centre Borelli, France Institut Universitaire de France

## Abstract

In Self-Supervised Learning (SSL), preventing representation collapse by explicitly enforcing a uniform distribution on the unit hypersphere has proven to be effective. However, current frameworks typically rely on sliced statistical regularizers such as SIGReg (used in LeJEPA) and SUSReg (used in SPHERE-JEPA), which approximate this continuous objective via Monte Carlo sampling along random 1D directions. This stochasticity injects projection variance into the training gradients, destabilizing optimization, and hindering convergence. In this work, we first show that analytically integrating out these random projections natively yields a deterministic Maximum Mean Discrepancy (MMD), bypassing the variance of sliced methods. Motivated by this equivalence, we formulate full-dimensional objectives for MMD, Kernel Stein Discrepancy (KSD), and Kullback–Leibler (KL) divergence directly on the sphere to enforce a uniform distribution. To prevent spatial bias, we equip these tests with rotationally invariant kernels constructed via spectral theory, systematically evaluating two canonical families: smooth exponential decay (Heat) and strict frequency cutoff (Bandlimited) filters. Empirically, removing projection-induced noise results in more stable optimization, faster convergence, and consistent improvements over stochastic sliced regularizers on ImageNet and Galaxy10. Furthermore, we reveal that the choice of the statistical test shapes the geometry of the learned latent space: MMD and KSD favor locally clustered organization suitable for object-centric domains, whereas the continuous KDE-based KL divergence promotes fine-grained instance separation, yielding the strongest results on unclustered procedural texture retrieval.

## 1 Introduction

A central paradigm in modern Self-Supervised Learning (SSL) is the multi-view formulation, where an encoder learns representations by minimizing an invariance objective across augmented views of the same input (Chen et al., 2020; Grill et al., 2020). To prevent representation collapse, modern architectures employ explicit global regularization (Balestriero and LeCun, 2025). Recent findings demonstrate that enforcing a uniform distribution on the unit hypersphere $\mathbb { S } ^ { d - 1 }$ is an optimal target, minimizing worst-case errors for downstream evaluations (Nicollier et al., 2026).

To enforce hyperspherical uniformity, frameworks such as SPHERE-JEPA (Nicollier et al., 2026) (via SUS-Reg) rely on sliced statistical regularizers, which project representations onto random 1D directions and compute the Epps-Pulley (Epps and Pulley, 1983) (EP) test. However, relying on this Monte Carlo approximation naturally injects “projection variance” into the training gradients, destabilizing optimization and slowing convergence. Concurrently, KerJEPA (Zimmermann et al., 2025) demonstrated that analogous sliced regularizers enforcing Gaussian embeddings can be analytically integrated to entirely bypass this variance, resulting in stable, closed-form Maximum Mean Discrepancy (MMD).

In this work, we extend this insight to the uniform distribution on the sphere. Building upon the equivalence established by KerJEPA, we demonstrate that analytically integrating out the random projections over $\mathbb { S } ^ { d - 1 }$ similarly yields a deterministic, closed-form Maximum Mean Discrepancy (MMD). Motivated by the strict suboptimality of 1D proxies, we focus only on exact multivariate statistics. While KerJEPA successfully employed MMD and Kernel Stein Discrepancy (KSD) (Liu et al., 2016) to enforce unconstrained Gaussian priors in Euclidean spaces, we adapt these exact multivariate statistical tests—alongside the Kullback-Leibler (KL) divergence (Kullback and Leibler, 1951)—to operate directly on the manifold, thereby enforcing hyperspherical uniformity.

Because standard empirical distribution tests do not generalize beyond 1D, exact high-dimensional statistical testing intrinsically relies on kernel methods. However, to ensure that the kernel does not privilege any specific region of the hypersphere $\mathbb { S } ^ { d - 1 }$ , we restrict our focus to rotationally invariant kernels. Any such kernel is characterized by the Laplacian’s eigenvalues. Leveraging this spectral foundation, we systematically evaluate three distinct kernels: the exact kernel implicitly induced by the analytical integration of SUSReg (Nicollier et al., 2026), alongside two canonical frequency filters adapted to the sphere: the Heat kernel (acting as a smooth exponential decay) and the Bandlimited kernel (enforcing a strict frequency cutoff).

In summary, our main contributions are as follows:

• Deterministic Hyperspherical Regularization: We derive a deterministic formulation of the SUSReg objective by analytically integrating the random 1D projections. This results in a closedform Maximum Mean Discrepancy (MMD) objective directly defined on $\mathbb { S } ^ { d - 1 }$ , removing Monte Carlo projection sampling during training.  
• Hyperspherical Statistical Regularizers: We adapt multivariate statistical objectives— Maximum Mean Discrepancy (MMD), Kernel Stein Discrepancy (KSD), and Kullback–Leibler (KL) divergence—to enforce uniform priors directly on the hyperspherical manifold.  
• Kernel Choice from Spectral Characterization: Guided by the spectral characterization of rotationally invariant kernels on the sphere, we study two kernel families on the hypersphere: Heat and Bandlimited kernels.  
• Improved Representation Quality: Across MMD and KSD variants, our deterministic objectives consistently outperform SUSReg, yielding gains of up to +1.3/ + 1.6 points (linear/k-NN) on ImageNet-100 and +5.1/ + 4.1 points on Galaxy10.

The remainder of this paper is organized as follows. Section 2 reviews the standard multi-view SSL framework on the hypersphere. In Section 3, we demonstrate the suboptimality of sliced proxies and establish their equivalence to a closed-form MMD. Section 4 introduces the exact multivariate statistical tests, and Section 5 details the design of our rotationally invariant spectral kernels. We derive explicit closed-form training objectives in Section 6. Finally, Section 7 presents our empirical evaluation across multiple benchmarks, and Section 8 concludes the work. Detailed proofs and derivations are deferred to the Appendices.

## 2 Preliminaries: Self-Supervised Learning on the Hypersphere

A central paradigm in modern Self-Supervised Learning (SSL) is the multi-view formulation, in which different augmented views of the same sample are used as supervisory signals (Chen et al., 2020; Grill et al., 2020). Formally, we consider a dataset of N independent samples. Each sample is processed through a stochastic data augmentation pipeline to generate $V _ { a }$ views $x _ { n , v } \in \mathbb { R } ^ { D }$ , where $n \in \{ 1 , \ldots , N \}$ and $v \in \{ 1 , \ldots , V _ { a } \}$ . Among these, we distinguish $V _ { g } \leq V _ { a }$ global views—which typically correspond to large-scale crops—from the remaining local views.

An encoder network $f _ { \theta }$ maps each view to a representation $z _ { n , v } = f _ { \theta } ( x _ { n , v } ) \in \mathbb { R } ^ { d }$ . Following the exact formulation introduced by SPHERE-JEPA (Nicollier et al., 2026), these embeddings are subsequently $\ell _ { 2 ^ { - } }$ normalized to lie on the unit hypersphere:

$$
\tilde {z} _ {n, v} = \frac {z _ {n , v}}{\| z _ {n , v} \|} \in \mathbb {S} ^ {d - 1}.
$$

To enforce representation consistency across augmentations, the model minimizes an alignment objective. This multi-view invariance loss can be formulated as:

$$
\mathcal {L} _ {\mathrm{inv}} = \frac {1}{V _ {a}} \sum_ {v = 1} ^ {V _ {a}} \| \tilde {z} _ {n, v} - \mu_ {n} \| _ {2} ^ {2}, \quad \mu_ {n} = \frac {1}{V _ {g}} \sum_ {v ^ {\prime} = 1} ^ {V _ {g}} \tilde {z} _ {n, v ^ {\prime}}. \tag {1}
$$

However, optimizing ${ \mathcal { L } } _ { \mathrm { i n v } }$ alone is degenerate, as the network can easily achieve zero loss by mapping all inputs to a single constant vector (Grill et al., 2020) (representation collapse). To prevent such trivial solutions and strictly control the global geometry of the representation space, SSL frameworks introduce an explicit regularization term (Caron et al., 2021; Balestriero and LeCun, 2025), leading to the general training objective:

$$
\mathcal {L} = (1 - \lambda) \mathcal {L} _ {\text { inv }} + \lambda \mathcal {L} _ {\text { reg }}, \tag {2}
$$

where $\lambda \in ( 0 , 1 )$ is a hyperparameter that controls the trade-off between the two objectives. In this formulation, while ${ \mathcal { L } } _ { \mathrm { i n v } }$ attracts views of the same instance, $\mathcal { L } _ { \mathrm { r e g } }$ constrains the global distribution of the embeddings. Based on spherical uniformity optimality (Nicollier et al., 2026), our goal is to design $\mathcal { L } _ { \mathrm { r e g } }$ such that it explicitly enforces the uniform distribution on the hypersphere, $q = \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ . The critical question then becomes: how should we practically compute the discrepancy between the empirical embeddings and this continuous uniform target?

## 3 Beyond Sliced Methods: Full-Dimensional Tests

To regularize high-dimensional representation spaces, recent frameworks (Balestriero and LeCun, 2025; Nicollier et al., 2026) rely on sliced methods. Justified by the Cramér-Wold theorem (Cuesta-Albertos et al., 2007), these approaches project the embeddings onto random 1D directions, effectively reducing the complex multi-dimensional distribution matching problem into a series of tractable 1D statistical tests. However, approximating this continuous objective via Monte Carlo sampling inherently injects an artificial projection variance into the training gradients. Fortunately, this stochasticity is unnecessary.

Equivalence to a Closed-Form MMD. By leveraging Bochner’s (Rahimi and Recht, 2007) and Fubini’s theorems, we can analytically integrate out all possible random projection directions $a \in \mathbb { S } ^ { d - 1 }$ . As detailed in Appendix A, we apply the analytical integration framework from KerJEPA (Zimmermann et al., 2025)— which has recently established this for the Gaussian distribution—to the uniform distribution on $\mathbb { S } ^ { d - 1 }$ . Specifically, let X and Y be random variables on $\mathbb { S } ^ { d - 1 }$ (representing, for instance, the empirical embedding distribution and the uniform target), and let $\operatorname { E P } ( \cdot , \cdot )$ denote the 1D Epps-Pulley discrepancy (Balestriero and LeCun, 2025). This exact integration reveals that the expected sliced Epps-Pulley discrepancy natively yields a deterministic Maximum Mean Discrepancy (Gretton et al., 2012) (MMD) with an induced kernel $\bar { k } { : }$

$$
\mathbb {E} _ {a \sim \operatorname{Unif} (\mathbb {S} ^ {d - 1})} \left[ \mathrm{EP} (a ^ {\top} X, a ^ {\top} Y) \right] = \mathrm{MMD} _ {k} ^ {2} (X, Y). \tag {3}
$$

![](images/b1a4e30f82884ffec44f87ad2f9440a665f584d6e56b5c1e87353072034553ff.jpg)

<details>
<summary>line chart</summary>

| Estimator: n=256 | 0 | ~10^-7 | ~10^-8 |
| --- | --- | --- | --- |
| Estimator: n=256 | 250 | ~10^-7 | ~10^-8 |
| Estimator: n=256 | 500 | ~10^-7 | ~10^-8 |
| Estimator: n=256 | 750 | ~10^-7 | ~10^-8 |
| Estimator: n=256 | 1000 | ~10^-7 | ~10^-8 |
| Estimator: n=256 | 1250 | ~10^-7 | ~10^-8 |
| Estimator: n=256 | 1500 | ~10^-7 | ~10^-8 |
| Estimator: n=256 | 1750 | ~10^-7 | ~10^-8 |
| Estimator: n=256 | 2000 | ~10^-7 | ~10^-8 |
| SUSReg-induced: n=256 | 0 | ~10^-7 | ~10^-8 |
| SUSReg-induced: n=256 | 250 | ~10^-7 | ~10^-8 |
| SUSReg-induced: n=256 | 500 | ~10^-7 | ~10^-8 |
| SUSReg-induced: n=256 | 750 | ~10^-7 | ~10^-8 |
| SUSReg-induced: n=256 | 1000 | ~10^-7 | ~10^-8 |
| SUSReg-induced: n=256 | 1250 | ~10^-7 | ~10^-8 |
| SUSReg-induced: n=256 | 1500 | ~10^-7 | ~10^-8 |
| SUSReg-induced: n=256 | 1750 | ~10^-7 | ~10^-8 |
| SUSReg-induced: n=256 | 2000 | ~10^-7 | ~10^-8 |
| SUSReg: n=512 | 0 | ~10^-7 | ~10^-8 |
| SUSReg: n=512 | 250 | ~10^-7 | ~10^-8 |
| SUSReg: n=512 | 500 | ~10^-7 | ~10^-8 |
| SUSReg: n=512 | 750 | ~10^-7 | ~10^-8 |
| SUSReg: n=512 | 1000 | ~10^-7 | ~10^-8 |
| SUSReg: n=512 | 1250 | ~10^-7 | ~10^-8 |
| SUSReg: n=512 | 1500 | ~10^-7 | ~10^-8 |
| SUSReg: n=512 | 1750 | ~10^-7 | ~10^-8 |
| SUSReg: n=512 | 2000 | ~10^-7 | ~10^-8 |
| SUSReg: n=1024 | 0 | ~10^-7 | ~10^-8 |
| SUSReg: n=1024 | 250 | ~10^-7.5 | ~10^-9 |
| SUSReg: n=1024 | 500 | ~10^-7.5 | ~10^-9 |
| SUSReg: n=1024 | 750 | ~10^-7.5 | ~10^-9 |
| SUSReg: n=1024 | 1000 | ~10^-7.5 | ~10^-9 |
| SUSReg: n=1024 | 1250 | ~10^-7.5 | ~10^-9 |
| SUSReg: n=1024 | 1500 | ~10^-7.5 | ~10^-9 |
| SUSReg: n=1024 | 1750 | ~10^-7.5 | ~10^-9 |
| SUSReg: n=1024 | 2000 | ~10^-7.5 | ~10^-9 |
</details>

Figure 1: Projection Variance. Variance comparison between standard SUSReg (dashed) and its induced MMD estimator (solid) on $\dot { \mathbb { S } } ^ { 2 5 5 }$ .

![](images/aac90534013dc76d9bc3e309cebf58178d08e4be5e8131c701dbcce14a499514.jpg)

<details>
<summary>line chart</summary>

| Epoch | SUSReg (stochastic) | MMD induced SUSReg k̄ |
|-------|---------------------|----------------------|
| 0     | 0.1                 | 0.1                  |
| 25    | 0.4                 | 0.5                  |
| 50    | 0.55                | 0.6                  |
| 75    | 0.6                 | 0.65                 |
| 100   | 0.65                | 0.68                 |
| 125   | 0.68                | 0.7                  |
| 150   | 0.7                 | 0.71                 |
| 175   | 0.7                 | 0.72                 |
| 200   | 0.7                 | 0.72                 |
</details>

Figure 2: Training Dynamics. The closed-form MMD (solid) entirely eliminates projection noise, achieving notably faster convergence.

Crucially, because the projection directions are distributed uniformly, the induced kernel $\bar { k }$ is rotationally invariant and admits the following explicit 1D integral representation:

$$
\bar {k} (x, y) = \int_ {- 1} ^ {1} \exp \left(- (1 - x ^ {\top} y) t ^ {2}\right) \rho_ {d} (t) d t, \tag {4}
$$

where $\rho _ { d } ( t ) \propto ( 1 - t ^ { 2 } ) ^ { \frac { d - 3 } { 2 } }$ is the marginal density of the hypersphere. In practice, we compute this exact kernel using Gauss-Jacobi quadrature. This numerical scheme naturally absorbs the geometric density $\rho _ { d } ( t )$ into its precomputed weights, transforming the analytical integral into a simple, deterministic dot product.

The Cost of Projection Variance. Compared to our exact formulation, evaluating the sliced objective via standard Monte Carlo sampling inherently suffers from projection noise (Figure 1). This artificial variance destabilizes the training gradients and bounds optimization efficiency, converging to the true MMD only in the infinite-projection limit. By eliminating this stochasticity, our deterministic closed-form MMD achieves notably faster convergence and superior training dynamics, as demonstrated by the accelerated training convergence on ImageNet-100 (Figure 2). Motivated by the strict empirical and theoretical suboptimality of 1D proxies, we discard sliced approximations to focus on exact multivariate discrepancies natively on the manifold.

## 4 Statistical Tests for Hyperspherical Uniformity

As established in Section 2, our objective is to match the empirical distribution of representations with the uniform target $q \ : = \ : \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ . For each view $v ,$ let $\hat { p } _ { v }$ denote the empirical distribution of normalized embeddings $\{ \tilde { z } _ { n , v } \} _ { n = 1 } ^ { N }$ . To enforce uniformity across views, we minimize the general regularization loss

$$
\mathcal {L} _ {\text { reg }} = \frac {1}{V _ {a}} \sum_ {v = 1} ^ {V _ {a}} D (\hat {p} _ {v}, q), \tag {5}
$$

where D is a statistical discrepancy measure between probability distributions. The choice of this discrepancy is not a detail; it dictates how the algorithm behaves and shapes the geometry of the learned latent space. Because different statistical tests penalize deviations from uniformity in distinct ways, they implicitly favor different topological arrangements—such as locally clustered spaces suited for object-centric classification, or unclustered, mutually repelled spaces optimal for continuous textures.

For notational simplicity, we drop the view index v when the context is clear, denoting the empirical distribution simply as ${ \hat { p } } .$

Motivated by the exact high-dimensional equivalence demonstrated in Section 3, we bypass stochastic 1D approximations to evaluate $D ( \hat { p } , q )$ directly through full-dimensional statistical tests on the manifold. All the following discrepancies rely on a positive definite base kernel k $\mathrm { : ~ } \mathbb { S } ^ { d - 1 } \times \mathbb { S } ^ { d - 1 }  \mathbb { R }$ . We introduce and formulate objectives for the following three distinct statistical tests:

Maximum Mean Discrepancy (MMD). MMD quantifies the distance between distributions by comparing their kernel mean embeddings (Gretton et al., 2012). In our framework, the squared MMD between the empirical batch distribution $\hat { p }$ and the continuous uniform target $q ,$ given a positive definite base kernel $k ,$ expands as:

$$
D _ {\mathrm{MMD} ^ {2}} (\hat {p}, q) = \mathbb {E} _ {x, x ^ {\prime} \sim \hat {p}} [ k (x, x ^ {\prime}) ] + \mathbb {E} _ {y, y ^ {\prime} \sim q} [ k (y, y ^ {\prime}) ] - 2 \mathbb {E} _ {x \sim \hat {p}, y \sim q} [ k (x, y) ]. \tag {6}
$$

As detailed in Appendix C, because the target $q$ is uniform and the kernel is rotationally invariant, all expectations with respect to q reduce to analytic constants. As a result, the training objective $D _ { \mathrm { M M D } } ( \hat { p } , q )$ simplifies: it relies solely on pairwise kernel similarities within the empirical batch $\hat { p } ,$ eliminating the need for target sampling.

Kernel Stein Discrepancy (KSD). KSD quantifies distribution mismatch by evaluating how well empirical samples satisfy Stein’s identity for a target distribution (Liu et al., 2016). Formally, the squared KSD for our empirical batch $\hat { p }$ against the uniform target $q$ is defined as the expectation of a Stein kernel $k _ { q } \mathrm { : }$

$$
D _ {\mathrm{KSD} ^ {2}} (\hat {p}, q) = \mathbb {E} _ {x, x ^ {\prime} \sim \hat {p}} [ k _ {q} (x, x ^ {\prime}) ]. \tag {7}
$$

The full construction of this Stein kernel, detailed in Appendix B, relies on the target distribution’s score function, $s _ { q } ( x ) = \nabla _ { x } \log q ( x )$ , and a base reproducing kernel k. Because the uniform target q has a constant density on the hypersphere $\dot { \mathbb S } ^ { d - 1 }$ , its score function vanishes. Consequently, the objective $D _ { \mathrm { K S D } } ( \hat { p } , q )$ can be evaluated in closed form using only the base kernel’s first and second derivatives with respect to the pairwise cosine similarity.

Kullback-Leibler (KL) Divergence via Kernel Density Estimation. The standard KL divergence measures the relative entropy between distributions. To match our empirical batch $\hat { p }$ with the uniform target $q ,$ , the objective takes the form:

$$
\mathrm{KL} (\hat {p} \| q) = \mathbb {E} _ {x \sim \hat {p}} [ \log \hat {p} (x) - \log q (x) ]. \tag {8}
$$

However, this standard formulation cannot directly compare the discrete empirical distribution $\hat { p }$ against the continuous target $q .$ To resolve this domain gap, we construct a continuous surrogate for the batch using a Kernel Density Estimator (KDE), $\tilde { p } ( x ) = \mathbb { E } _ { x ^ { \prime } \sim \hat { p } } [ k ( x , x ^ { \prime } ) ]$ ]. Assuming this smooth surrogate faithfully captures the underlying data distribution, substituting it into the equation above yields our tractable sample-based objective:

$$
D _ {\mathrm{KL}} (\hat {p}, q) \approx \mathbb {E} _ {x \sim \hat {p}} [ \log \tilde {p} (x) - \log q (x) ]. \tag {9}
$$

## 5 Spectral Kernel Design on the Hypersphere

The effectiveness of MMD, KSD, and KL-based discrepancies depends on the properties of the kernel k. To prevent the regularization objective from biasing the representation space toward any preferred direction, we constrain the kernel to be rotationally invariant (zonal): $k ( x , y ) = \varphi ( x ^ { \top } y )$ . We systematically normalize it so that $\varphi ( 1 ) = 1$ .

Viewed through the lens of spectral theory, the hypersphere is equipped with the Laplace-Beltrami operator, whose eigenfunctions are the spherical harmonics. By Schoenberg’s theorem (Schoenberg, 1942), any valid positive-definite zonal kernel on $\mathbb { S } ^ { d - 1 }$ admits a spectral expansion over normalized Gegenbauer polynomials $\tilde { C } _ { \ell } ^ { ( \alpha ) }$ 1981), we can explicitly control the smoothness of the Reproducing Kernel Hilbert Space (RKHS) in which we measure the discrepancies by defining these coefficients as weights $w ( \lambda _ { \ell } )$ that depend on the smoothness of the corresponding mode:

$$
\varphi (c) = \frac {1}{Z} \sum_ {\ell = 0} ^ {\infty} w (\lambda_ {\ell}) \tilde {C} _ {\ell} ^ {(\alpha)} (c), \tag {10}
$$

![](images/7ec244cdc9d79f714a0537022c5c60255f8a50bfb51e4b2d6f69a703954ccba2.jpg)

<details>
<summary>line chart</summary>

| ⟨x, y⟩ | Heat Kernel — t = 4/d | Heat Kernel — t = 5/d | Heat Kernel — t = 6/d | Bandlimited Kernel | SUSReg induced kernel |
| ------ | --------------------- | --------------------- | --------------------- | ------------------ | --------------------- |
| -1.00  | 0.0                   | 0.03                  | 0.27                  | 1.0                | 0.45                  |
| -0.75  | 0.0                   | 0.05                  | 0.32                  | 0.8                | 0.47                  |
| -0.50  | 0.0                   | 0.07                  | 0.37                  | 0.6                | 0.49                  |
| -0.25  | 0.0                   | 0.10                  | 0.42                  | 0.4                | 0.51                  |
| 0.00   | 0.0                   | 0.15                  | 0.47                  | 0.2                | 0.53                  |
| 0.25   | 0.05                  | 0.22                  | 0.52                  | 0.1                | 0.55                  |
| 0.50   | 0.15                  | 0.32                  | 0.60                  | 0.3                | 0.57                  |
| 0.75   | 0.35                  | 0.48                  | 0.72                  | 0.6                | 0.61                  |
| 1.00   | 1.0                   | 1.0                   | 1.0                   | 1.0                | 1.0                   |
</details>

Figure 3: Profiles of normalized zonal kernels on the hypersphere $\mathbb { S } ^ { d - 1 } ~ ( d = 2 5 6 )$ . We compare the heat kernel at different scale parameters $( t \in \{ 4 / d , 5 / d , 6 / d \} )$ and the Bandlimited kernel against the implicit kernel induced by the SUSReg objective. All kernels are evaluated as a function of the pairwise cosine similarity $c = x ^ { \top } y \in [ - 1 , 1 ]$ .

where $c = x ^ { \top } y , \alpha = ( d - 2 ) / 2$ , and $\lambda _ { \ell } = \ell ( \ell + d - 2 )$ are the Laplacian eigenvalues. The normalization constant $\begin{array} { r } { Z = \sum _ { \ell = 0 } ^ { \infty } w ( \lambda _ { \ell } ) \tilde { C } _ { \ell } ^ { ( \alpha ) } ( 1 ) } \end{array}$ enforces $\varphi ( 1 ) = 1$ .

Importantly, each eigenvalue $\lambda _ { \ell }$ equals the Dirichlet energy of its corresponding eigenfunction. Because these kernels heavily penalize high-energy (unsmooth) functions, statistical tests derived from this general expansion are broadly referred to as “Sobolev tests” in the statistics literature (Giné, 1975; Jupp, 2008).

In practice, the spectral weights $w ( \lambda _ { \ell } )$ act as frequency filters. Low values of ℓ correspond to smooth, global geometric variations, while higher values capture increasingly oscillatory spatial patterns. This framework naturally motivates two canonical filter choices natively adapted to the sphere’s geometry (illustrated in Figure 3):

• Heat Kernel (Smooth Decay): Applying an exponential decay $w ( \lambda _ { \ell } ) = e ^ { - t \lambda _ { \ell } }$ , where $t > 0$ is a scale parameter, yields the heat kernel (Zhao and Song, 2018). This progressive damping of highly oscillatory patterns results in a stable, multi-scale similarity measure.  
• Bandlimited Kernel (Hard Cutoff): Alternatively, applying a strict low-pass filter $w ( \lambda _ { \ell } ) = \mathbf { 1 } _ { \ell \leq L }$ yields a bandlimited kernel. This acts as an exact projection onto the first L eigenmodes, capturing global geometry while rigorously filtering out fine-grained spatial noise.

Certain weight functions that lead to kernels with known closed form include the chordal distance kernel of the Giné $S _ { n }$ test, the Poisson kernel, and (in terms of sums of special functions) the thin-plate spline kernel of order one Beatson et al. (2018); Jupp (2008).

Having established these rigorous kernel families, the final step is to integrate them into our general discrepancy measures. Because these specific filters yield well-behaved scalar functions of the pairwise cosine similarity $\varphi ( c )$ , they allow us to transform the abstract statistical tests introduced in Section 4 into exact, computationally tractable training objectives, as we derive in the next section.

## 6 Explicit Forms of the Statistical Tests

Using the rotationally invariant zonal kernels $\varphi ( c )$ evaluated on the pairwise cosine similarity $c = x ^ { \top } y$ , the general discrepancies introduced in Section 4 evaluate analytically. This yields the following deterministic, closed-form objectives (complete derivations are deferred to Appendices B, C and D):

$$
D _ {\mathrm{MMD}} = \frac {1}{C _ {\text {norm / MMD}}} \left(\mathbb {E} _ {x, y \sim \hat {p}} [ \varphi (c) ] - C _ {\text {bias / MMD}}\right), \tag {11}
$$

$$
D _ {\mathrm{KSD}} = \frac {1}{C _ {\text {norm / KSD}}} \mathbb {E} _ {x, y \sim \hat {p}} \left[ \frac {1}{2} \left((c ^ {2} - 1) \varphi^ {\prime \prime} (c) + c (d - 1) \varphi^ {\prime} (c)\right) \right], \tag {12}
$$

$$
D _ {\mathrm{KL}} = \frac {1}{C _ {\text { norm } / \mathrm{KL}}} \left(\mathbb {E} _ {x \sim \hat {p}} \left[ \log \mathbb {E} _ {y \sim \hat {p} _ {- x}} [ \varphi (c) ] \right] - C _ {\text { bias } / \mathrm{KL}}\right). \tag {13}
$$

For MMD and KSD, expectations over the empirical distribution $\hat { p }$ are computed using standard Vestimators. For the KL divergence, $\hat { p } _ { - x }$ denotes the leave-one-out empirical estimator to prevent trivial self-similarity singularities.

To ensure strict comparability across objectives, each discrepancy is offset and scaled by analytic constants such that a total representation collapse to a Dirac delta distribution $( \mathrm { i . e . , ~ } c = 1 $ for all pairs) yields a worst-case loss of exactly 1. Assuming a normalized base kernel $\varphi ( 1 ) = 1$ , these constants are defined as follows:

• MMD: $\begin{array} { r } { C _ { \mathrm { b i a s / M M D } } = \int _ { - 1 } ^ { 1 } \varphi ( v ) \rho _ { d } ( v ) d v } \end{array}$ is evaluated numerically via highly efficient Gauss-Jacobi quadrature utilizing the hyperspherical marginal density $\rho _ { d } ( v ) \propto ( 1 - v ^ { 2 } ) ^ { \frac { d - 3 } { 2 } }$ , and $C _ { \mathrm { n o r m / M M D } } =$ $1 - C _ { \mathrm { b i a s / M M D } }$ .  
$\begin{array} { r } { C _ { \mathrm { n o r m / K S D } } = \frac { d - 1 } { 2 } \varphi ^ { \prime } ( 1 ) } \end{array}$ uniform target).  
$\bullet \mathrm { \bf ~ K L } \colon C _ { \mathrm { b i a s / K L } } = \log | \mathbb { S } ^ { d - 1 } | \mathrm { \ a n d \ } C _ { \mathrm { n o r m / K L } } = - C _ { \mathrm { b i a s / K L } } .$

In summary, Equations (11)–(13) provide the exact, closed-form objectives that replace stochastic 1D approximations. Computationally, for a batch size $B ,$ our exact formulation scales as $\mathcal { O } ( B ^ { 2 } )$ , whereas sliced methods scale as $\mathcal { O } ( B | \boldsymbol { A } | )$ , where |A| is the number of random projections. As a result, exact tests are strictly cheaper to compute when the batch size satisfies $B < | { \mathcal { A } } |$ . Because standard sliced implementations default to $| \mathcal { A } | = 1 0 2 4$ projections (Nicollier et al., 2026), our exact objectives require less compute time for reasonable batch sizes, but become more expensive when scaling B beyond this threshold.

## 7 Experimental Evaluation

We empirically validate our exact, full-dimensional statistical tests across a diverse suite of benchmarks. Through these experiments, we aim to answer three practical questions: (1) Does analytically bypassing projection variance (via the induced kernel) consistently improve representation quality over stochastic baselines like $\mathrm { S U S R e g ? }$ (2) How do the different spectral kernels (Induced SUSReg, Heat, and Bandlimited) compare empirically? (3) How do the different exact tests (MMD, KSD, and KL-based) behave depending on the underlying topology of the dataset (e.g., clustered object classes versus unclustered continuous textures)?

Unless otherwise specified, all models employ a projection head outputting to $\mathbb { S } ^ { 2 5 5 }$ . To ensure fair comparisons, we maintain consistent hyperparameter configurations across methods. Specifically, we set the regularization weight to $\lambda = 0 . 0 5$ for MMD and KSD, strictly matching the established baseline in SPHERE-JEPA (Nicollier et al., 2026). For the KL divergence, we set $\lambda = 0 . 5 ;$ ; this value is motivated by its structural similarity to the InfoNCE objective, which, when explicitly decomposed into an invariance loss and a repulsive contrastive term, naturally induces an effective regularization weight of 0.5. Furthermore, for all experiments utilizing the heat kernel, we set the temperature to $t = 5 / d$ for MMD and KSD, and $t = 2 / d$ for the KDE-based KL divergence; an ablation justifying these specific thermal operating points is provided in Appendix E.

![](images/5715deec4f1ce08f06be613ad2ea4677227afdd54a3d1ea4ca143a472205dcf9.jpg)  
Figure 4: Examples of images from the datasets used in our experiments. The first row shows samples from ImageNet100, and the second row shows samples from Galaxy10.

## 7.1 Standard Pretraining (ImageNet-100 & Galaxy10)

We first evaluate the quality of the representations by pretraining a ResNet-18 (He et al., 2016) on ImageNet-100 (Deng et al., 2009) for 200 epochs and a ResNet-50 (He et al., 2016) on the Galaxy10 dataset (Leung and Bovy, 2018) for 200 epochs (see Figure 4 for dataset samples). Because our exact regularizers estimate the hyperspherical distribution directly from the empirical batch, maintaining consistent statistics is critical; we therefore strictly fix the global batch size to 256 across all runs. We subsequently assess the semantic quality of the frozen features using standard linear probing and k-NN classification.

As shown in Table 1, the exact deterministic regularizers consistently outperform the stochastic SUSReg baseline across both datasets. On ImageNet-100, MMD and KSD variants improve linear probing accuracy by up to 1.3% and k-NN accuracy by up to 1.6%. On Galaxy10, the performance gains reach up to 5.1% in linear probing and 4.1% in k-NN classification.

The direct comparison between SUSReg and MMD equipped with the analytically induced kernel ¯k isolates the impact of projection noise, as both objectives share the same continuous limit. Replacing the stochastic 1D projections with the exact evaluation yields a performance increase of 1.0% on ImageNet-100 and 4.7% on Galaxy10 for linear probing. Among the spectral kernels, the Bandlimited filter (L = 2) slightly outperforms the smooth Heat kernel and the Induced kernel on these multi-class tasks. Conversely, the KDE-based KL divergence underperforms on both datasets, scoring below the baseline on ImageNet-100. This indicates that the continuous repulsion enforced by the KL objective penalizes the local macroscopic clustering necessary for category-level semantic classification.

## 7.2 Instance-Level Texture Retrieval

Given that the KL divergence underperformed on class-heavy datasets like ImageNet, we hypothesize that its continuous surrogate formulation might be optimally suited for purely unclustered distributions. To verify this, we investigate instance-level discrimination via nearest-neighbor retrieval on four procedural texture datasets (Cloud, Disk, Flake, Wood) introduced in SPHERE-JEPA (Nicollier et al., 2026) (see Figure 5 for dataset samples). Following their exact evaluation protocol, this benchmark provides a strictly continuous visual domain where discrete object classes do not exist. Detailed per-dataset results are provided in Appendix F.

Remarkably, as detailed in Table 2, the KDE-based KL divergence strongly dominates its MMD and KSD counterparts in this scenario. KL (Heat) achieves a top-1 retrieval accuracy (Recall@1) of 95.3%, representing a substantial +6.6 point improvement over the stochastic SUSReg baseline (88.7%) and outperforming the best alternative continuous metric, KSD (Heat), by +3.3 points. Similar margins are observed across the mean Average Precision (mAP) metrics, demonstrating robust improvements in both the projection head and the underlying embeddings.

Table 1: Linear probing and k-NN classification accuracy (%) for standard pretraining on ImageNet-100 and Galaxy10. Values are reported as mean ± sample standard deviation over seeds. Bold values indicate the best mean performance within each dataset and protocol.

<table><tr><td rowspan="2">Method</td><td colspan="2">ImageNet-100</td><td colspan="2">Galaxy10</td></tr><tr><td>Linear</td><td>k-NN</td><td>Linear</td><td>k-NN</td></tr><tr><td>SUSReg (Stochastic Baseline)</td><td>71.22 ± 0.58</td><td>65.67 ± 0.56</td><td>72.13 ± 1.39</td><td>68.67 ± 1.04</td></tr><tr><td>MMD (Induced SUSReg k̄)</td><td>72.17 ± 0.40</td><td>67.02 ± 0.23</td><td>76.78 ± 0.31</td><td>71.67 ± 0.89</td></tr><tr><td>MMD (Heat, t = 5/d)</td><td>72.19 ± 0.26</td><td>67.21 ± 0.22</td><td>76.31 ± 0.56</td><td>71.91 ± 0.62</td></tr><tr><td>MMD (Bandlimited, L = 2)</td><td>72.26 ± 0.44</td><td>67.25 ± 0.43</td><td>76.21 ± 0.49</td><td>72.73 ± 1.00</td></tr><tr><td>KSD (Heat, t = 5/d)</td><td>72.55 ± 0.29</td><td>66.71 ± 0.31</td><td>76.76 ± 0.40</td><td>72.03 ± 0.57</td></tr><tr><td>KSD (Bandlimited, L = 2)</td><td>72.19 ± 0.38</td><td>66.91 ± 0.14</td><td>77.21 ± 0.55</td><td>72.70 ± 0.77</td></tr><tr><td>KL (Heat)</td><td>67.72 ± 0.21</td><td>62.09 ± 0.42</td><td>75.76 ± 0.99</td><td>70.29 ± 0.85</td></tr></table>

Table 2: Average procedural texture retrieval performance across four datasets. The continuous KL divergence strongly dominates on these unclustered domains.

<table><tr><td>Method</td><td>Recall@1</td><td>Recall@3</td><td>Recall@5</td><td>mAP</td><td>mAP (emb)</td></tr><tr><td>SUSReg (Stochastic Baseline)</td><td>88.7</td><td>96.1</td><td>97.7</td><td>92.6</td><td>91.4</td></tr><tr><td>MMD (Induced SUSReg  $\bar{k}$ )</td><td>89.1</td><td>96.1</td><td>97.8</td><td>92.9</td><td>91.1</td></tr><tr><td>MMD (Heat,  $t = 5/d$ )</td><td>91.4</td><td>97.0</td><td>98.3</td><td>94.4</td><td>93.2</td></tr><tr><td>MMD (Bandlimited,  $L = 2$ )</td><td>91.5</td><td>96.6</td><td>97.7</td><td>94.3</td><td>93.7</td></tr><tr><td>KSD (Heat,  $t = 5/d$ )</td><td>92.0</td><td>97.1</td><td>98.2</td><td>94.8</td><td>93.5</td></tr><tr><td>KSD (Bandlimited,  $L = 2$ )</td><td>91.6</td><td>96.6</td><td>97.7</td><td>94.3</td><td>93.6</td></tr><tr><td>KL (Heat)</td><td>95.3</td><td>97.6</td><td>98.3</td><td>96.7</td><td>96.3</td></tr></table>

This performance gap directly validates our geometric intuition. Because procedural textures lack semantic element clusters, the continuous KL divergence optimally regularizes the manifold by uniformly repelling all instances individually, leading to exact hyperspherical uniformity. Conversely, point-based integral metrics like MMD and KSD inherently accommodate and even encourage macroscopic local clustering. While this behavior is highly beneficial for multi-class discrimination (e.g., ImageNet) where semantic grouping is desirable, it proves fundamentally suboptimal for continuous, instance-level texture spaces.

## 8 Conclusion

In this work, we first demonstrated that sliced statistical regularizers for hyperspherical uniformity are strictly suboptimal compared to their analytically integrated Maximum Mean Discrepancy (MMD) equivalent. Building on this insight, we introduced a broader family of exact, full-dimensional regularizers based on MMD, Kernel Stein Discrepancy (KSD), and the Kullback-Leibler (KL) divergence. To avoid spatial bias, we grounded these tests in spectral theory, explicitly motivating the use of two canonical kernels: the Heat and Bandlimited filters.

Empirically, our exact formulations confirmed that completely bypassing Monte Carlo projection noise systematically improves representation quality across all evaluated datasets. On object-centric domains, MMD and KSD equipped with our canonical spectral kernels achieved similar state-of-the-art performance, outperforming both the standard stochastic baseline and its exact SUSReg-induced kernel. Finally, we revealed how the choice of statistical discrepancy fundamentally shapes the learned topology: while MMD and KSD inherently accommodate the local clustering necessary for distinct classes, the continuous KDE-based KL divergence optimally forces fine-grained instance repulsion, yielding vastly superior performance on unclustered procedural textures.

Despite these theoretical and empirical advantages, it is important to contextualize the computational tradeoffs of exact manifold regularizers. Because our full-dimensional tests rely on exact pairwise kernel evaluations, their computational complexity scales quadratically with the global batch size, $\mathcal { O } ( B ^ { 2 } )$ . In contrast, stochastic sliced methods like SUSReg compute 1D statistics and scale linearly with respect to the batch size, O(B|A|), where |A| is the number of random projections. Consequently, while our exact regularizers strictly dominate in standard training regimes, projection-based approaches remain a highly practical and necessary alternative when scaling SSL architectures to massive batch sizes where quadratic complexity becomes computationally prohibitive.

## References

R. Balestriero and Y. LeCun. Lejepa: Provable and scalable self-supervised learning without the heuristics, 2025. URL https://arxiv.org/abs/2511.08544.  
A. Barp, C. J. Oates, E. Porcu, and M. Girolami. A Riemann–Stein kernel method. Bernoulli, 28(4):2181 – 2208, 2022. doi: 10.3150/21-BEJ1415. URL https://doi.org/10.3150/21-BEJ1415.  
R. K. Beatson, W. Zu Castell, et al. Thinplate splines on the sphere. SIGMA. Symmetry, Integrability and Geometry: Methods and Applications, 14:083, 2018.  
M. Caron, H. Touvron, I. Misra, H. Jégou, J. Mairal, P. Bojanowski, and A. Joulin. Emerging properties in self-supervised vision transformers. In Proceedings of the IEEE/CVF international conference on computer vision, pages 9650–9660, 2021.  
T. Chen, S. Kornblith, M. Norouzi, and G. Hinton. A simple framework for contrastive learning of visual representations. In International conference on machine learning, pages 1597–1607. PmLR, 2020.  
K. Chwialkowski, H. Strathmann, and A. Gretton. A kernel test of goodness of fit. In M. F. Balcan and K. Q. Weinberger, editors, Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pages 2606–2615, New York, New York, USA, 20–22 Jun 2016. PMLR. URL https://proceedings.mlr.press/v48/chwialkowski16.html.  
J. A. Cuesta-Albertos, R. Fraiman, and T. Ransford. A sharp form of the Cramér–Wold theorem. Journal of Theoretical Probability, 2007.  
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2009.  
T. W. Epps and L. B. Pulley. A test for normality based on the empirical characteristic function. Biometrika, 70(3):723–726, 1983. ISSN 00063444. URL http://www.jstor.org/stable/2336512.  
E. Giné. Invariant tests for uniformity on compact riemannian manifolds based on sobolev norms. The Annals of statistics, 3(6):1243–1266, 1975.  
A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola. A kernel two-sample test. J. Mach. Learn. Res., 13(null):723–773, Mar. 2012. ISSN 1532-4435.  
J.-B. Grill, F. Strub, F. Altché, C. Tallec, P. Richemond, E. Buchatskaya, C. Doersch, B. Avila Pires, Z. Guo, M. Gheshlaghi Azar, et al. Bootstrap your own latent-a new approach to self-supervised learning. Advances in neural information processing systems, 33:21271–21284, 2020.  
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 770–778, 2016.  
P. E. Jupp. Data-driven Sobolev tests of uniformity on compact Riemannian manifolds. The Annals of Statistics, 36(3):1246 – 1260, 2008. doi: 10.1214/009053607000000541. URL https://doi.org/10.1214/ 009053607000000541.  
S. Kolouri, K. Nadjahi, S. Shahrampour, and U. Şimşekli. Generalized sliced probability metrics. In ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), pages 4513–4517. IEEE, 2022.  
S. Kullback and R. A. Leibler. On information and sufficiency. The Annals of Mathematical Statistics, 22 (1):79–86, 1951. ISSN 00034851. URL http://www.jstor.org/stable/2236703.  
H. W. Leung and J. Bovy. Deep learning of multi-element abundances from high-resolution spectroscopic data. Monthly Notices of the Royal Astronomical Society, Nov. 2018. ISSN 1365-2966. doi: 10.1093/ mnras/sty3217. URL http://dx.doi.org/10.1093/mnras/sty3217.  
Q. Liu, J. Lee, and M. Jordan. A kernelized stein discrepancy for goodness-of-fit tests. In International conference on machine learning, pages 276–284. PMLR, 2016.  
K. Nadjahi, A. Durmus, L. Chizat, S. Kolouri, S. Shahrampour, and U. Simsekli. Statistical and topological properties of sliced probability divergences. Advances in Neural Information Processing Systems, 33: 20802–20812, 2020.  
L. Nicollier, M. Dunitz, M. Pic, P. Musé, E. Meinhardt-Llopis, and G. Facciolo. Sphere-jepa: Spherical prediction with homogeneous embeddings, 2026. URL https://arxiv.org/abs/2605.26900.  
X. Qu and B. C. Vemuri. Theory and applications of kernel stein’s method on riemannian manifolds, 2025. URL https://arxiv.org/abs/2501.00695.  
A. Rahimi and B. Recht. Random features for large-scale kernel machines. In J. Platt, D. Koller, Y. Singer, and S. Roweis, editors, Advances in Neural Information Processing Systems, volume 20. Curran Associates, Inc., 2007. URL https://proceedings.neurips.cc/paper\_files/paper/2007/file/ 013a006f03dbc5392effeb8f18fda755-Paper.pdf.  
R. M. Rustamov. Closed-form expressions for maximum mean discrepancy with applications to wasserstein auto-encoders. Stat, 10(1):e329, 2021.  
A. Sablayrolles, M. Douze, C. Schmid, and H. Jégou. Spreading vectors for similarity search, 2019. URL https://arxiv.org/abs/1806.03198.  
I. J. Schoenberg. Positive definite functions on spheres. Duke Mathematical Journal, 9(1):96 – 108, 1942. doi: 10.1215/S0012-7094-42-00908-6. URL https://doi.org/10.1215/S0012-7094-42-00908-6.  
G. Wahba. Spline interpolation and smoothing on the sphere. SIAM Journal on Scientific and Statistical Computing, 2(1):5–16, 1981.  
W. Xu and T. Matsuda. A stein goodness-of-fit test for directional distributions. In International Conference on Artificial Intelligence and Statistics, pages 320–329. PMLR, 2020.  
C. Zhao and J. S. Song. Exact heat kernel on a hypersphere and its applications in kernel svm. Frontiers in Applied Mathematics and Statistics, 4, Jan. 2018. ISSN 2297-4687. doi: 10.3389/fams.2018.00001. URL http://dx.doi.org/10.3389/fams.2018.00001.  
E. Zimmermann, H. Wiltzer, J. Szeto, D. Alvarez-Melis, and L. Mackey. Kerjepa: Kernel discrepancies for euclidean self-supervised learning, 2025. URL https://arxiv.org/abs/2512.19605.

## A Formal Equivalence and Analytical Integration of Sliced Discrepancies

In this section, we provide the formal proofs supporting the theoretical claims made in Section 3. Specifically, we establish that the expected value of the sliced Epps-Pulley (EP) test—which serves as the core statistical regularizer in stochastic methods like SUSReg—is equivalent to a full-dimensional Maximum Mean Discrepancy (MMD) natively operating on $\mathbb { S } ^ { d - 1 }$ .

We proceed in three logical steps: (1) we show via Bochner’s theorem that the 1D EP test is equivalent to a 1D MMD; (2) we prove via Fubini’s theorem that integrating this 1D MMD over all uniform random projections yields a valid full-dimensional MMD; and (3) we geometrically derive the explicit 1D integral form of the induced kernel ¯k introduced in the main text.

Lemma 1 (The 1D Epps-Pulley Test is a 1D MMD [p. 4 (Rustamov, 2021)]). Let U and V be scalar random variables with characteristic functions $\varphi _ { U }$ and $\varphi _ { V }$ . The Epps-Pulley discrepancy is defined as

$$
\mathrm{EP} (U, V) = \int_ {\mathbb {R}} \left| \varphi_ {U} (t) - \varphi_ {V} (t) \right| ^ {2} w (t) d t, \tag {14}
$$

where $\begin{array} { r } { w ( t ) = \frac { 1 } { \sqrt { 2 \pi } } \exp \left( - \frac { t ^ { 2 } } { 2 } \right) } \end{array}$ . This formulation is equivalent to a Maximum Mean Discrepancy:

$$
\mathrm{EP} (U, V) = \mathrm{MMD} _ {k _ {1}} ^ {2} (U, V), \tag {15}
$$

where $\begin{array} { r } { k _ { 1 } ( u , v ) = \exp \left( - \frac { ( u - v ) ^ { 2 } } { 2 } \right) } \end{array}$ is the standard Gaussian base kernel.

Proof. By Bochner’s theorem (Rahimi and Recht, 2007), a continuous, translation-invariant positive definite kernel $k _ { 1 } ( u , v ) = \psi ( u - v )$ admits a spectral representation via the inverse Fourier transform of a density $w ( t )$ :

$$
\psi (\tau) = \int_ {\mathbb {R}} e ^ {i t \tau} w (t) d t.
$$

Specifically, the standard Gaussian density $\begin{array} { r } { w ( t ) = \frac { 1 } { \sqrt { 2 \pi } } \exp ( - t ^ { 2 } / 2 ) } \end{array}$ of the standard Gaussian base kernel $\begin{array} { r } { k _ { 1 } ( u , v ) = \exp \left( - \frac { ( u - v ) ^ { 2 } } { 2 } \right) } \end{array}$ . For such translation-invariant kernels, the squared MMD between the distributions of $U$ and V can be evaluated directly in the frequency domain as the weighted $L ^ { 2 }$ distance between their characteristic functions:

$$
\mathrm{MMD} _ {k _ {1}} ^ {2} (U, V) = \int_ {\mathbb {R}} \bigl | \varphi_ {U} (t) - \varphi_ {V} (t) \bigr | ^ {2} w (t) d t.
$$

This frequency-domain formulation natively matches the exact definition of the EP test, concluding the proof. □

Lemma 2 (Expected Sliced MMD is a Full-Dimensional MMD [p. 4, Eq. (4)-(6)(Kolouri et al., 2022)). Let X and Y be random variables in $\mathbb { R } ^ { d }$ , and let $k _ { 1 }$ be a continuous positive definite kernel on R. Define the integrated kernel on $\mathbb { R } ^ { d } \times \mathbb { R } ^ { d }$ as:

$$
\bar {k} (x, y) = \mathbb {E} _ {a \sim \mathrm{Unif} (\mathbb {S} ^ {d - 1})} \big [ k _ {1} (a ^ {\top} x, a ^ {\top} y) \big ].
$$

Then $\bar { k }$ is a valid positive definite kernel on $\mathbb { R } ^ { d }$ , and the expected 1D MMD over random uniform projections $i s$ identically equal to the MMD induced by ¯k:

$$
\mathbb {E} _ {a \sim \mathrm{Unif} (\mathbb {S} ^ {d - 1})} \left[ \mathrm{MMD} _ {k _ {1}} ^ {2} (a ^ {\top} X, a ^ {\top} Y) \right] = \mathrm{MMD} _ {\bar {k}} ^ {2} (X, Y).
$$

Remark 1. It is worth noting that the integrated kernel ¯k on $\mathbb { R } ^ { d }$ inherits not just the positive-definiteness of the base kernel $k _ { 1 }$ , but also its characteristic property. It is an easy matter to verify that ¯k is characteristic when $k _ { 1 }$ is directly, using the Cramér-Wold theorem; it follows also as a special case of $/ p .$ 4, Proposition 1 (Nadjahi et al., 2020)]. Thus, the induced full-dimensional MMD divergence remains a (definite) metric on probability measures $i f k _ { 1 }$ is.

Proof. Since $k _ { 1 }$ is positive definite on R, for any fixed direction $a ~ \in ~ \mathbb { S } ^ { d - 1 }$ , the function $k _ { a } ( x , y ) \ : =$ $k _ { 1 } ( a ^ { \top } x , a ^ { \top } y )$ is a positive definite kernel on $\mathbb { R } ^ { d }$ . Indeed, for any $x _ { 1 } , \ldots , x _ { n } \in \mathbb { R } ^ { d }$ and coefficients $c _ { 1 } , \ldots , c _ { n } \in \mathbb { R }$ ,

$$
\sum_ {i, j} c _ {i} c _ {j} k _ {a} (x _ {i}, x _ {j}) = \sum_ {i, j} c _ {i} c _ {j} k _ {1} (a ^ {\top} x _ {i}, a ^ {\top} x _ {j}) \geq 0.
$$

Because the expectation of positive definite kernels remains positive definite, the integrated kernel $\bar { k } ( x , y ) =$ $\mathbb { E } _ { a } [ k _ { a } ( x , y ) ]$ is a valid positive definite kernel on $\mathbb { R } ^ { d }$ .

To prove the equality of the discrepancies, we expand the expected one-dimensional MMD:

$$
\mathbb {E} _ {a} \left[ \mathrm{MMD} _ {k _ {1}} ^ {2} (a ^ {\top} X, a ^ {\top} Y) \right] = \mathbb {E} _ {a} \left[ \mathbb {E} _ {X, X ^ {\prime}} [ k _ {1} (a ^ {\top} X, a ^ {\top} X ^ {\prime}) ] + \mathbb {E} _ {Y, Y ^ {\prime}} [ k _ {1} (a ^ {\top} Y, a ^ {\top} Y ^ {\prime}) ] \right.
$$

$$
\left. - 2 \mathbb {E} _ {X, Y} [ k _ {1} (a ^ {\top} X, a ^ {\top} Y) ] \right].
$$

By Fubini’s theorem, we can safely exchange the expectations over the random directions a and the random variables $X , Y \colon$ :

$$
\mathbb {E} _ {a} \big [ \mathrm{MMD} _ {k _ {1}} ^ {2} (a ^ {\top} X, a ^ {\top} Y) \big ] = \mathbb {E} _ {X, X ^ {\prime}} \big [ \mathbb {E} _ {a} [ k _ {1} (a ^ {\top} X, a ^ {\top} X ^ {\prime}) ] \big ] + \mathbb {E} _ {Y, Y ^ {\prime}} \big [ \mathbb {E} _ {a} [ k _ {1} (a ^ {\top} Y, a ^ {\top} Y ^ {\prime}) ] \big ]
$$

$$
- 2 \mathbb {E} _ {X, Y} \left[ \mathbb {E} _ {a} [ k _ {1} (a ^ {\top} X, a ^ {\top} Y) ] \right].
$$

Substituting the definition of the integrated kernel $\bar { k }$ into the expectations yields exactly the full-dimensional formulation $\mathrm { M M D } _ { \bar { k } } ^ { 2 } ( X , Y )$ . □

Derivation of the Explicit Integral Form. By sequentially applying Lemma 1 and Lemma 2, we have established that the expected sliced objective resolves exactly to $\bar { \mathbb { E } _ { a } } [ \mathrm { E P } ( a ^ { \top } X , a ^ { \top } Y ) ] = \mathrm { M M D } _ { k } ^ { 2 } ( X , Y )$ . To compute this deterministically in practice, we must derive the explicit form of the induced kernel $\bar { k } ( x , y )$ .

For unit vectors $x , y \in \mathbb { S } ^ { d - 1 }$ , the standard Gaussian base kernel evaluates to:

$$
k _ {1} (a ^ {\top} x, a ^ {\top} y) = \exp \left(- \frac {(a ^ {\top} (x - y)) ^ {2}}{2}\right).
$$

Let $c = x ^ { \top } y$ denote the cosine similarity. On the unit sphere, the distance between embeddings is $\| x - y \| =$ p2(1 − c). We can therefore factorize the random projection along the unit direction u = x−y∥x−y∥ $\sqrt { 2 ( 1 - c ) }$ $\begin{array} { r } { u = \frac { x - y } { \Vert x - y \Vert } } \end{array}$ as:

$$
a ^ {\top} (x - y) = \| x - y \| (a ^ {\top} u) = \sqrt {2 (1 - c)} \cdot t,
$$

where $t = a ^ { \top } u$ . Since the vector a is drawn uniformly from $\mathbb { S } ^ { d - 1 }$ and $u$ is fixed, the scalar projection $t \in [ - 1 , 1 ]$ natively follows the hyperspherical marginal distribution, whose probability density is:

$$
\rho_ {d} (t) = \frac {\Gamma (d / 2)}{\sqrt {\pi} \Gamma ((d - 1) / 2)} (1 - t ^ {2}) ^ {\frac {d - 3}{2}}.
$$

Substituting this geometric projection back into the kernel expectation, the multidimensional integration over the random directions a simplifies into a single 1D integral. The exponent simplifies cleanly since $\frac { 1 } { 2 } ( \sqrt { 2 ( 1 - c ) } t ) ^ { 2 } = ( 1 - c ) t ^ { 2 }$ , yielding:

$$
\bar {k} (x, y) = \int_ {- 1} ^ {1} \exp \left(- (1 - c) t ^ {2}\right) \rho_ {d} (t) d t.
$$

This derivation confirms that the induced kernel is strictly rotationally invariant—depending solely on the cosine similarity c—and matches the explicit equation presented in Section 3. Furthermore, this form naturally justifies the use of Gauss-Jacobi quadrature for fast, deterministic, and fully differentiable evaluations.

## B Kernel Stein Discrepancy on the Hypersphere

In this appendix, we derive a closed-form expression of the Kernel Stein Discrepancy (KSD) for the uniform distribution on the hypersphere $\mathbb { S } ^ { d - 1 }$ , yielding the explicit objective $D _ { \mathrm { K S D } }$ presented in Section 6.

While the theoretical properties and consistency of this generalized Riemannian KSD have been established in Qu and Vemuri (2025), we provide here a self-contained derivation explicitly tailored to our uniformity test.

The Riemann-Stein Framework. Building upon the geometric Stein kernel framework (Barp et al., 2022) and its directional formulation (Xu and Matsuda, 2020), let M be a Riemannian manifold and q a target density. The main idea behind the Stein method is to find an operator A such that $\mathbb { E } _ { x \sim q } [ { \cal A } f ( x ) ] = 0$ for any suitable test function $f .$ By Stokes theorem, on a compact manifold without boundary, the integral of the divergence of any vector field is zero. To construct our operator $\mathcal { A } _ { X }$ , we consider the density-weighted vector field $Y = f \cdot q \cdot X$ , where $f$ is a scalar test function and $X$ is a fixed vector field. Expanding its divergence via the standard product rule yields:

$$
\begin{array}{l} \operatorname{div} (f q X) = \langle \nabla (f q), X \rangle + f q \operatorname{div} (X) \\ = q \langle \nabla f, X \rangle + f \langle \nabla q, X \rangle + f q \operatorname{div} (X). \\ \end{array}
$$

Factoring out q and using the identity $\nabla q = q \nabla$ log q, we obtain div $\ b ( f q X ) = q \mathcal { A } _ { X } f$ , which defines the general Stein operator acting on $f$ along X:

$$
\mathcal {A} _ {X} f (x) = \left\langle X (x), \nabla f (x) \right\rangle + \left(\operatorname{div} (X) (x) + \left\langle X (x), \nabla \log q (x) \right\rangle\right) f (x). \tag {16}
$$

Since $\begin{array} { r } { \int _ { \mathcal { M } } \mathrm { d i v } ( f q X ) d \mu = 0 } \end{array}$ , it immediately follows that the expectation under q is zero: $\mathbb { E } _ { q } [ \mathcal { A } _ { X } f ] = 0 .$ .

Strategic Choice of Vector Field. The general Stein operator defined in (16) is valid for any smooth vector field X. However, evaluating this operator depends on computing the divergence div(X), which often leads to numerically unstable expressions (such as Jacobian derivatives in local spherical coordinates).

To circumvent this issue, we will choose X to be a divergence-free vector field, more specifically a Killing vector field or infinitessimal isometry.

For the hypersphere $\mathbb { S } ^ { d - 1 }$ , these isometries are generated by the Lie algebra ${ \mathfrak { s o } } ( d )$ consisting of all skewsymmetric matrices of size $d \times d .$ . We adopt its canonical orthonormal basis, given by the matrices $E _ { i j }$ for $1 \leq i < j \leq d \colon$

$$
E _ {i j} = \frac {1}{\sqrt {2}} \left(e _ {i} e _ {j} ^ {\top} - e _ {j} e _ {i} ^ {\top}\right) \in \mathfrak {s o} (d),
$$

/here $e _ { i }$ is the i-th canonical basis vector of $\mathbb { R } ^ { d } .$ . Applying these matrices to a point $x \in \mathbb { S } ^ { d - 1 }$ yields a basis of Killing vector fields. By setting our generic field to $X ( x ) = E _ { i j } x$ , the divergence term in (16) vanishes.

Furthermore, since our target distribution q is the uniform density on $\mathbb { S } ^ { d - 1 }$ , its log-derivative vanishes $\left( \nabla \log q \equiv 0 \right)$ . Equation (16) therefore reduces to:

$$
\mathcal {A} _ {i j} f (x) = \left(E _ {i j} x\right) ^ {\top} \nabla_ {x} f (x). \tag {17}
$$

Constructing the Stein Kernel. The mechanism of the Kernel Stein Discrepancy established by Chwialkowski et al. (2016), evaluates the discrepancy by taking the supremum over a unit ball in a Reproducing Kernel Hilbert Space (RKHS). Thanks to the reproducing property, this supremum evaluates in closed form to the expectation of a symmetric Stein kernel $k _ { q } ( x , y )$ .

This Stein kernel is constructed by applying the Stein operator to a base scalar reproducing kernel $k ( x , y )$ on both variables. Because this directional approach utilizes a set of independent operators—one for each skew-symmetric matrix in the Lie algebra ${ \mathfrak { s o } } ( d )$ —the overall Stein kernel on the hypersphere is formed by applying the double operator along each valid direction and summing the results over the entire basis:

$$
k _ {q} (x, y) = \sum_ {1 \leq i <   j \leq d} \left(\mathcal {A} _ {i j} ^ {x} \mathcal {A} _ {i j} ^ {y} k\right) (x, y). \tag {18}
$$

Let us expand the inner term. First, applying the operator with respect to y yields a row vector:

$$
\mathcal {A} _ {i j} ^ {y} k (x, y) = \left(E _ {i j} y\right) ^ {\top} \nabla_ {y} k (x, y) = \nabla_ {y} ^ {\top} k (x, y) \left(E _ {i j} y\right).
$$

Next, applying the operator with respect to x to this result requires the Hessian of the kernel, obtained via standard multivariable calculus:

$$
\left(\mathcal {A} _ {i j} ^ {x} \mathcal {A} _ {i j} ^ {y} k\right) (x, y) = \left(E _ {i j} x\right) ^ {\top} \left[ \nabla_ {x} \left(\nabla_ {y} ^ {\top} k (x, y)\right) \right] \left(E _ {i j} y\right). \tag {19}
$$

Specialization to Radial Kernels. We now specialize to radial (zonal) kernels on the sphere, which only depend on the inner product $c : = x ^ { \top } y$ . Let $k ( x , y ) = \phi ( c )$ for $x , y \in \mathbb { S } ^ { d - 1 }$ , where $\phi : [ - 1 , 1 ] $ R is twice differentiable.

Using this notation, we have $\phi ^ { \prime } ( c ) = \nabla _ { y } k ( x , y ) x$ .

Differentiating with respect to x yields the Hessian:

$$
\nabla_ {x} \nabla_ {y} ^ {\top} k (x, y) = \phi^ {\prime \prime} (c) y x ^ {\top} + \phi^ {\prime} (c) I. \tag {20}
$$

Evaluating the Sum over the Lie Algebra. Substituting (20) back into (19), the term for a single basis vector becomes:

$$
\begin{array}{l} \big (\mathcal {A} _ {i j} ^ {x} \mathcal {A} _ {i j} ^ {y} k \big) (x, y) = \big (E _ {i j} x \big) ^ {\top} \big (\phi^ {\prime \prime} (c) y x ^ {\top} + \phi^ {\prime} (c) I \big) (E _ {i j} y) \\ = \phi^ {\prime \prime} (c) \big ((E _ {i j} x) ^ {\top} y \big) \big (x ^ {\top} E _ {i j} y \big) + \phi^ {\prime} (c) \big ((E _ {i j} x) ^ {\top} (E _ {i j} y) \big). \tag {21} \\ \end{array}
$$

To compute the sum over $1 \leq i < j \leq d ,$ , an expansion using linear algebra yields the matrix M :

$$
M (x, y) := \sum_ {1 \leq i <   j \leq d} (E _ {i j} x) (E _ {i j} y) ^ {\top} = \frac {1}{2} \left(c   I - y x ^ {\top}\right). \tag {22}
$$

We can now evaluate the two terms in (21) using M:

The $\phi ^ { \prime \prime } ( c )$ term (Quadratic Form): Since $E _ { i j }$ is skew-symmetric $( E _ { i j } ^ { \top } = - E _ { i j } )$ , we can manipulate the transposes: $( E _ { i j } x ) ^ { \top } y = - x ^ { \top } E _ { i j } y = y ^ { \top } E _ { i j } x$ . Moreover, the scalar $x ^ { \top } E _ { i j } y$ equals its transpose $- y ^ { \top } E _ { i j } x$ . The product thus becomes:

$$
\big ((E _ {i j} x) ^ {\top} y \big) \big (x ^ {\top} E _ {i j} y \big) = (y ^ {\top} E _ {i j} x) (- y ^ {\top} E _ {i j} x) = - (y ^ {\top} E _ {i j} x) ^ {2}.
$$

Summing this quadratic expression is equivalent to evaluating the form $y ^ { \top } M x$ on the matrix M defined in (22), because $\begin{array} { r } { \sum _ { i < j } ( y ^ { \top } E _ { i j } \bar { x } ) ( y ^ { \top } E _ { i j } ^ { \top } x ) = y ^ { \top } } \end{array}$ M x. Evaluating this explicitly yields:

$$
y ^ {\top} M x = y ^ {\top} \left(\frac {1}{2} (c I - y x ^ {\top})\right) x = \frac {1}{2} \big (c (y ^ {\top} x) - (y ^ {\top} y) (x ^ {\top} x) \big) = \frac {1}{2} (c ^ {2} - 1). \tag {23}
$$

The $\phi ^ { \prime } ( c )$ term (Trace): The dot product of two vectors $u ^ { \top } v$ is equal to the trace of their outer product $\operatorname { T r } ( v u ^ { \top } )$ . Therefore:

$$
\sum_ {1 \leq i <   j \leq d} (E _ {i j} x) ^ {\top} (E _ {i j} y) = \sum_ {1 \leq i <   j \leq d} \operatorname{Tr} \bigl ((E _ {i j} y) (E _ {i j} x) ^ {\top} \bigr) = \operatorname{Tr} (M).
$$

Taking the trace of M from (22), $\operatorname { T r } ( I ) = d$ and $\mathrm { T r } ( y x ^ { \top } ) = x ^ { \top } y = c ,$ yields:

$$
\operatorname{Tr} (M) = \operatorname{Tr} \left(\frac {1}{2} (c I - y x ^ {\top})\right) = \frac {1}{2} (c d - c) = \frac {c (d - 1)}{2}. \tag {24}
$$

Closed Form. Combining these two evaluated sums, we obtain our remarkably simple closed form for the Stein kernel of the uniform distribution on $\mathbb { S } ^ { d - 1 }$ , depending on the first two derivatives of the radial kernel ϕ:

$$
k _ {q} (x, y) = \frac {1}{2} \left[ \left(c ^ {2} - 1\right) \phi^ {\prime \prime} (c) + c (d - 1) \phi^ {\prime} (c) \right], \quad c = x ^ {\top} y. \tag {25}
$$

Normalization for Representation Learning. Substituting this closed-form Stein kernel back into the expectation over the empirical distribution $\hat { p }$ yields our raw KSD objective. As established in Section 6, we scale this discrepancy such that a worst-case representation collapse evaluates exactly to 1. In the event of collapse $( x = y$ , yielding $c = 1 )$ , the term $( c ^ { 2 } - 1 ) \phi ^ { \prime \prime } ( c )$ vanishes, and the expectation evaluates to exactly $\textstyle { \frac { d - 1 } { 2 } } \bar { \phi ^ { \prime } } ( 1 )$ .

By defining the normalization constant $\begin{array} { r } { C _ { \mathrm { n o r m / K S D } } = \frac { d - 1 } { 2 } \phi ^ { \prime } ( 1 ) } \end{array}$ , we scale the objective such that a complete point collapse $( c = 1 )$ evaluates to exactly 1, yielding the final explicit form $D _ { \mathrm { K S D } }$ presented in Section 6:

$$
D _ {\mathrm{KSD}} = \frac {1}{C _ {\text {norm} / \mathrm{KSD}}} \mathbb {E} _ {x, y \sim \hat {p}} \left[ \frac {1}{2} \left((c ^ {2} - 1) \phi^ {\prime \prime} (c) + c (d - 1) \phi^ {\prime} (c)\right) \right]. \tag {26}
$$

## C Maximum Mean Discrepancy on the Hypersphere

In this appendix, we detail the derivation of the Maximum Mean Discrepancy (MMD) (Gretton et al., 2012) on $\mathbb { S } ^ { d - 1 }$ and specialize it to the rotationally invariant zonal kernels introduced in Section 5, yielding the explicit objective presented in Section 6.

Maximum Mean Discrepancy on a Manifold. Let $\mu$ be the Haar measure on $\mathbb { S } ^ { d - 1 }$ , and let

$$
k: \mathbb {S} ^ {d - 1} \times \mathbb {S} ^ {d - 1} \to \mathbb {R}
$$

be a positive definite kernel with an associated reproducing kernel Hilbert space (RKHS) H. For two probability densities $p$ and $q$ on $\mathbb { S } ^ { d - 1 }$ (with respect to $\mu )$ , the squared Maximum Mean Discrepancy is defined as the distance between their mean embeddings:

$$
\mathrm{MMD} ^ {2} (p, q) = \| \mu_ {p} - \mu_ {q} \| _ {\mathcal {H}} ^ {2},
$$

where the mean embedding of a distribution r is given by $\begin{array} { r } { \mu _ { r } = \int _ { \mathbb { S } ^ { d - 1 } } k ( \cdot , x ) r ( x ) \mu ( d x ) } \end{array}$

Expanding the RKHS norm yields the classical expression:

$$
\mathrm{MMD} ^ {2} (p, q) = \mathbb {E} _ {x, x ^ {\prime} \sim p} [ k (x, x ^ {\prime}) ] + \mathbb {E} _ {y, y ^ {\prime} \sim q} [ k (y, y ^ {\prime}) ] - 2 \mathbb {E} _ {x \sim p, y \sim q} [ k (x, y) ]. \tag {27}
$$

Zonal Kernels and the Uniform Target. Following Section 4, we evaluate the discrepancy between the empirical distribution of the embeddings, $p = { \hat { p } } .$ , and the uniform target distribution, $q = \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ . Furthermore, as established in Section 5, we restrict our focus to rotationally invariant zonal kernels of the form:

$$
k (x, y) = \varphi (c), \qquad \text {where} \quad c := x ^ {\top} y \in [ - 1, 1 ].
$$

Substituting these into Equation (27), the empirical MMD becomes:

$$
\mathrm{MMD} ^ {2} (\hat {p}, q) = \mathbb {E} _ {x, x ^ {\prime} \sim \hat {p}} \left[ \varphi \left(x ^ {\top} x ^ {\prime}\right) \right] + \mathbb {E} _ {y, y ^ {\prime} \sim q} \left[ \varphi \left(y ^ {\top} y ^ {\prime}\right) \right] - 2 \mathbb {E} _ {x \sim \hat {p}, y \sim q} \left[ \varphi \left(x ^ {\top} y\right) \right]. \tag {28}
$$

Because $q$ is uniform, by rotational invariance, if $y \sim q$ and $x \in \mathbb { S } ^ { d - 1 }$ is a fixed vector, the inner product $v = x ^ { \top } y$ follows a known distribution with density:

$$
\rho_ {d} (v) = \frac {(1 - v ^ {2}) ^ {\frac {d - 3}{2}}}{B \left(\frac {1}{2} , \frac {d - 1}{2}\right)}, \qquad v \in [ - 1, 1 ],
$$

where $B ( \cdot , \cdot )$ denotes the Beta function. Consequently, expectations involving the uniform target reduce to a tractable one-dimensional integral over $[ - 1 , 1 ]$ . We define this analytic bias constant as:

$$
C _ {\mathrm{bias/MMD}} := \int_ {- 1} ^ {1} \varphi (v) \rho_ {d} (v) d v.
$$

Because both the target-target expectation and the cross-term expectation integrate over the uniform measure $q ,$ they simplify identically to this constant:

$$
\mathbb {E} _ {y, y ^ {\prime} \sim q} \big [ \varphi (y ^ {\top} y ^ {\prime}) \big ] = C _ {\mathrm{bias/MMD}},
$$

$$
\mathbb {E} _ {x \sim \hat {p}, y \sim q} [ \varphi (x ^ {\top} y) ] = C _ {\mathrm{bias/MMD}}.
$$

Substituting these identities back into Equation (28) yields the unnormalized objective:

$$
\mathrm{MMD} ^ {2} (\hat {p}, q) = \mathbb {E} _ {x, x ^ {\prime} \sim \hat {p}} \bigl [ \varphi (x ^ {\top} x ^ {\prime}) \bigr ] - C _ {\text {bias / MMD}}. \tag {29}
$$

Normalization for Representation Learning. To ensure comparable gradients and bounded objectives across different kernels and dimensionalities, we scale this raw discrepancy to obtain a worst-case collapse value of exactly 1. Total representation collapse occurs when all embeddings are mapped to the same point on the hypersphere $( { \mathrm { i . e . , ~ } } x = x ^ { \prime }$ , yielding $x ^ { \top } x ^ { \prime } = 1 )$ .

Since our kernels are systematically normalized such that $\varphi ( 1 ) = 1$ , the maximum possible value of the unnormalized MMD is $1 - C _ { \mathrm { b i a s / M M D } }$ . By defining the normalization constant $C _ { \mathrm { n o r m / M M D } } = 1 - C _ { \mathrm { b i a s / M M D } }$ , we obtain the final explicit regularization objective $D _ { \mathrm { M M D } }$ presented in Section 6:

$$
D _ {\mathrm{MMD}} = \frac {1}{C _ {\mathrm{norm/MMD}}} \left(\mathbb {E} _ {x, x ^ {\prime} \sim \hat {p}} [ \varphi (c) ] - C _ {\mathrm{bias/MMD}}\right). \tag {30}
$$

## D Kullback–Leibler Divergence on the Hypersphere

In this appendix, we detail the derivation of the Kullback–Leibler (KL) divergence on the hypersphere $\mathbb { S } ^ { d - 1 }$ and show how a Kernel Density Estimation (KDE) approach leads to the explicit, closed-form objective $D _ { \mathrm { K I } }$ L presented in Section 6.

Kullback–Leibler Divergence on a Manifold. Let $\mu$ be the Haar measure on $\mathbb { S } ^ { d - 1 }$ . For two probability densities $p$ and $q$ on $\mathbb { S } ^ { d - 1 }$ (with respect to $\mu )$ , the Kullback–Leibler divergence is defined as:

$$
\mathrm{KL} (p \| q) = \int_ {\mathbb {S} ^ {d - 1}} p (x) \log \frac {p (x)}{q (x)} \mu (d x). \tag {31}
$$

Equivalently, writing the expectation with respect to $p ,$ we have:

$$
\mathrm{KL} (p \| q) = \mathbb {E} _ {x \sim p} [ \log p (x) - \log q (x) ]. \tag {32}
$$

Uniform Reference Distribution. We now specialize to the case where the target reference distribution is the uniform measure on the hypersphere, $q = \mathrm { U n i f } ( \mathbb { S } ^ { d - 1 } )$ . The density of the uniform distribution with respect to the Haar measure is constant and equal to $q ( x ) = 1 / | \mathbb { S } ^ { d - 1 } |$ , where

$$
| \mathbb {S} ^ {d - 1} | = \frac {2 \pi^ {d / 2}}{\Gamma (d / 2)}
$$

denotes the surface area of the hypersphere. Substituting this constant density into Equation (32) yields:

$$
\mathrm{KL} (p \| q) = \mathbb {E} _ {x \sim p} [ \log p (x) ] + \log | \mathbb {S} ^ {d - 1} |. \tag {33}
$$

This establishes that minimizing the KL divergence to the uniform distribution is fundamentally equivalent to maximizing the differential entropy of $p .$ The geometric constant governing this shift is defined in our formulation as $C _ { \mathrm { b i a s / K L } } = \log | \mathbb { S } ^ { d - 1 } |$ .

Sample-Based Approximations and Gradient Stability. In practical representation learning, we only have access to a discrete empirical distribution $\hat { p }$ derived from a finite minibatch of embeddings. Because the exact differential entropy $\mathbb { E } _ { { x } \sim { \hat { p } } } [ \log { \hat { p } ( x ) } ]$ diverges for a discrete sum of Dirac deltas, we must rely on continuous, sample-based approximations.

While some existing methods (such as the KoLeo estimator (Sablayrolles et al., 2019)) approximate this entropy using nearest-neighbor distances, we explicitly discard this approach. The reliance on a hard min operator to identify the closest neighbor inherently induces highly non-smooth, unstable gradients during backpropagation, severely destabilizing optimization.

Instead, to obtain a robust, smoothly differentiable, and full-dimensional statistical test that seamlessly integrates with our rotationally invariant kernels $k ( x , x ^ { \prime } ) = \varphi ( c )$ , we construct a continuous surrogate for the empirical density using Kernel Density Estimation (KDE). To prevent trivial self-similarity singularities—where the density estimate at x is entirely dominated by the kernel evaluating its own embedding $( x ^ { \top } x = 1 )$ —we strictly evaluate this density using a leave-one-out estimator, denoted $\hat { p } _ { - x } \colon$

$$
\tilde {p} (x) = \mathbb {E} _ {x ^ {\prime} \sim \hat {p} _ {- x}} \big [ \varphi (x ^ {\top} x ^ {\prime}) \big ].
$$

Replacing the exact density $p ( x )$ in Equation (33) with this leave-one-out surrogate yields our unnormalized KDE-based objective.

Normalization for Representation Learning. As established in Section 6, we systematically scale our discrepancies such that a worst-case representation collapse evaluates exactly to 1. In the event of total collapse, all normalized embeddings map to the exact same coordinate $( x = x ^ { \prime }$ , yielding $c = 1 )$ . Because our zonal kernels are calibrated such that $\varphi ( 1 ) = 1$ , the leave-one-out density estimate $\tilde { p } ( x )$ equals 1, and its logarithm evaluates to exactly 0.

To map this boundary condition consistently across all tests, we apply the scaling constant $C _ { \mathrm { n o r m / K L } }$ alongside the analytic bias $C _ { \mathrm { b i a s / K I } }$ defined above. This yields the final explicit, normalized objective $D _ { \mathrm { K L } }$ as implemented in Section 6:

$$
D _ {\mathrm{KL}} = \frac {1}{C _ {\mathrm{norm} / \mathrm{KL}}} \left(\mathbb {E} _ {x \sim \hat {p}} \left[ \log \mathbb {E} _ {x ^ {\prime} \sim \hat {p} _ {- x}} [ \varphi (c) ] \right] - C _ {\mathrm{bias} / \mathrm{KL}}\right). \tag {34}
$$

## E Ablation of the Temperature for the Heat Kernel

The regularizing behavior of the heat kernel inherently depends on its scale parameter $t ,$ which dictates the exponential decay of the spectral weights $w ( \lambda _ { \ell } ) = e ^ { - t \lambda _ { \ell } }$ . On the hypersphere $\mathbb { S } ^ { d - 1 }$ , the natural baseline for this time step scales inversely with the dimension, establishing $1 / d$ as a unit.

To determine a good smoothing factor, we focus our ablation on the critical region where the kernel’s spatial profile exhibits the most significant structural variations. As visualized in Figure 3, this corresponds to the temperature range of $t \in \{ 4 / d , 5 / d , 6 / d \}$ . It is worth noting that for density-based evaluations—such as the KDE approximation used in the KL divergence—the requirements are different. In such cases, a much smaller temperature (strictly evaluated at $2 / d )$ is necessary to ensure the density approximation remains theoretically valid.

The empirical sensitivity of the MMD and KSD objectives to the temperature parameter is detailed in Table 3 for ImageNet-100. An observation is the optimization instability of the KSD objective at $t = 4 / d ,$ , which can lead to representation collapse in some runs. This collapse explains the large variance across random seeds (yielding a standard deviation of ±35.20% for linear probing) observed on ImageNet-100.

## F Texture Retrieval Evaluation

Following the experimental framework established by SPHERE-JEPA (Nicollier et al., 2026), we evaluate our method on their nonparametric texture retrieval task. This task is specifically designed to assess the geometry of learned representations: given a query image, the objective is to retrieve another view of the same texture instance among visually similar candidate samples.

Table 3: ImageNet-100 heat-kernel temperature sensitivity. Linear probing and k-NN classification accuracy (%) are reported as mean ± sample standard deviation over seeds.

<table><tr><td rowspan="2">Temperature</td><td colspan="2">MMD (Heat)</td><td colspan="2">KSD (Heat)</td></tr><tr><td>Linear</td><td>k-NN</td><td>Linear</td><td>k-NN</td></tr><tr><td>4/d</td><td>72.25 ± 0.70</td><td>66.48 ± 0.21</td><td>53.80 ± 35.20</td><td>49.93 ± 32.62</td></tr><tr><td>5/d</td><td>72.55 ± 0.29</td><td>66.71 ± 0.31</td><td>72.55 ± 0.29</td><td>66.71 ± 0.31</td></tr><tr><td>6/d</td><td>71.73 ± 0.49</td><td>66.64 ± 0.60</td><td>72.44 ± 0.30</td><td>67.27 ± 0.42</td></tr></table>

We adopt their exact evaluation protocol, procedural datasets, and data augmentation pipeline, which we briefly summarize below.

## F.1 Datasets

We use the four procedural texture datasets introduced in Nicollier et al. (2026): Disk, Cloud, Flake, and Wood. As originally described by the authors, each dataset is generated from a distinct stochastic process (e.g., heavy-tailed noise, Brownian motion, Perlin noise).

As illustrated in Figure 5, samples within a given dataset are generated using different random seeds. This yields images that share a similar statistical structure while remaining visually distinct. Following their standard setup, each texture family consists of 10,000 training, 500 validation, and 10,000 test samples.

## F.2 Data Augmentation and View Generation

To generate multiple views from each texture image $( V _ { g } = 2$ views per image by default), we directly apply the stochastic spatial and photometric augmentation pipeline proposed in Nicollier et al. (2026).

As visualized in Figure 6, each view is generated using a random affine transformation (incorporating rotation, translation, scaling, and shear), followed by photometric augmentations such as brightness/contrast adjustments and random erasing. These transformations introduce significant spatial variability while preserving the underlying texture statistics. The exact same augmentation pipeline is applied at test time to ensure consistency.

## F.3 Quantitative Results

Evaluation is performed within mini-batches of size B = 100. For each query, similarity is computed against the other samples in the batch, and retrieval performance is measured based on the ranking of these candidates. We report nearest-neighbor retrieval performance using Recall@K $( K \in \{ 1 , 3 , 5 \} )$ ) and mean Average Precision (mAP). To ensure the regularizers effectively structure the underlying feature space rather than just the projection space, we evaluate both on the projection head outputs (mAP) and directly on the frozen backbone embeddings (mAP emb).

Table 2 summarizes the average performance across all four texture domains. Tables 4, 5, 6, and 7 provide the detailed breakdown for each individual dataset.

Superiority of Exact Hyperspherical Uniformity. As shown in Table 2, KL (Heat) consistently and significantly outperforms all other methods on average. The performance gains are particularly striking in the top-1 retrieval metric, where KL (Heat) achieves an average Recall@1 of 95.3%, representing a substantial +6.6 absolute points improvement over the stochastic SUSReg baseline (88.7%). This dominant trend is consistent across all individual datasets, where KL (Heat) even nearly saturates performance on the easier domains (e.g., 99.4% Recall@1 on Wood).

![](images/4d679708e77edff9331f2b35e7b81f8e27eb4c3d6ed80c29be0bdab357b361a8.jpg)

<details>
<summary>natural_image</summary>

Abstract grayscale texture with no discernible objects, text, or symbols
</details>

![](images/af662f165b47c463d43fe5fd649a0ad76a8d351182abdbc128fb5cb8d91aaf3b.jpg)

<details>
<summary>natural_image</summary>

Abstract grayscale texture with no discernible text, symbols, or structured elements
</details>

![](images/18e153a7d40ad4e1fe656f6a91e9ab812fca788048ee63523abb5fe897decc8b.jpg)

<details>
<summary>natural_image</summary>

Grayscale abstract texture with no discernible text, symbols, or structured elements
</details>

![](images/55efda412d7ba491d57c93711673a61a734c82a5390d80ba8136baf3c09d1399.jpg)

<details>
<summary>natural_image</summary>

Abstract pattern of black and white dots on a gray background (no text or symbols)
</details>

![](images/d1b2449496a52f8351b45493f5ef292c45e9e1a552129d2b874af6ba05a91940.jpg)

<details>
<summary>natural_image</summary>

Abstract pattern of scattered black and white dots on a gray background (no text or symbols)
</details>

![](images/eb7311f0113a84218bb1b806f3fcf350d1025a58b6e2c6fa63078c110158cd62.jpg)

<details>
<summary>natural_image</summary>

Abstract pattern of scattered black and white dots on a gray background (no text or symbols)
</details>

![](images/0df34ddfd35724d584b6eeebfffac6f7231f293d02922b476a00329be6be337a.jpg)

<details>
<summary>natural_image</summary>

Abstract grayscale pattern with scattered star-like and starburst shapes (no text or symbols)
</details>

![](images/d8469526fc793bd6cb11fc520e414e73f28b481d4cbf59964b246575b244943f.jpg)

<details>
<summary>natural_image</summary>

Abstract grayscale pattern with star-like and radiating shapes (no text or symbols)
</details>

![](images/9f4b6356a3b36f014ee8401b7b1ad83e75218465170b703cc09c28a279b7fb1f.jpg)

<details>
<summary>natural_image</summary>

Abstract grayscale pattern with scattered star-like shapes and dots (no text or symbols)
</details>

![](images/1db04904e01f76cd131a06498c8156d222af7616e16ae6b8287fed832257bf34.jpg)

<details>
<summary>natural_image</summary>

Abstract pattern of wavy black and white lines on a light background (no text or symbols)
</details>

![](images/e204cd1cc2f72baaf05039546a73a1b9a94055f5ed546052195b9d1bf0d5e00e.jpg)

<details>
<summary>natural_image</summary>

Abstract pattern of wavy black lines on white background (no text or symbols)
</details>

![](images/ae41618fa29e2624ffaac9ecc1b718d21b42ff26fdbcfd499e36b4994dedbe86.jpg)

<details>
<summary>natural_image</summary>

Abstract pattern of black wavy lines on white background, no text or symbols present
</details>

Figure 5: Examples of procedural textures used for retrieval evaluation, as introduced in Nicollier et al. (2026). Each row corresponds to a texture family (top to bottom: cloud, disk, flake, wood). Images within a row are generated from the same stochastic process with different random seeds, resulting in strong statistical similarity but distinct visual realizations. 20 20

![](images/a04a4ae5fa48217f6fee80782235e1164bea7a65b71209f0ba2628b1f9e95c55.jpg)

<details>
<summary>natural_image</summary>

Abstract geometric composition with overlapping colored polygons (blue, yellow, red, green) on a starry background with scattered white dots (no text or symbols)
</details>

![](images/c8e8d8b9bb9c4e5455593b71efc848e0a4b3dbaf1679532737ee0c25c5dc6a00.jpg)

<details>
<summary>natural_image</summary>

Black-and-white image showing scattered bright white dots against a dark background, resembling stars or particles (no text or symbols)
</details>

![](images/558b15420568e37b8bc600850839368fbdb6632d2d2de79c37d74d37d6323666.jpg)

<details>
<summary>natural_image</summary>

Dark field image showing scattered bright spots with a black rectangle in the center (no text or symbols)
</details>

![](images/f61a4481a275f478c3ee932b88479071a3d6c21264064440c2f7ef7480a6e457.jpg)

<details>
<summary>natural_image</summary>

Microscopic view of scattered bright spots on a gray background with a black square in the bottom right corner (no text or symbols)
</details>

![](images/4af768d052d811eb3eb998ada88aeaf74f8f4f1f7c1e0b5d871eef0b1df8aa1f.jpg)

<details>
<summary>natural_image</summary>

Microscopic view of scattered white particles on a gray background with a black rectangular region in the top-right corner (no text or symbols)
</details>

Figure 6: Left: effective source regions induced by independently sampled random affine transformations. Right: corresponding augmented views obtained after applying the complete affine and photometric transformation pipeline from Nicollier et al. (2026).

Continuous vs. Stochastic Regularization. The aggregated results clearly validate our theoretical analysis regarding representation geometry in nonparametric settings. The continuous, full-dimensional regularizers—specifically MMD and KSD utilizing Heat or Bandlimited kernels—consistently yield better instance-level discrimination than both the stochastic baseline (SUSReg) and the induced MMD approach. For instance, replacing the induced MMD with a continuous Heat kernel MMD improves the average Recall@1 by +2.3 points (from 89.1% to 91.4%). This confirms that continuous structural constraints pave the way for the substantial jump achieved by the exact KL penalty.

Backbone vs. Projection Head. Finally, comparing the average mAP evaluated on the projection head (96.7% for KL Heat) to the mAP on the frozen backbone embeddings (96.3% for KL Heat) reveals that the structural improvements transfer successfully to the core representations. While there is a standard, slight drop when evaluating the embeddings directly across all methods, models regularized with continuous metrics maintain excellent embedding performance. This confirms that the uniformly distributed geometry is genuinely learned by the backbone and not merely overfitted within the projection head.

Table 4: Disk Texture

<table><tr><td>Method</td><td>Recall@1</td><td>Recall@3</td><td>Recall@5</td><td>mAP</td><td>mAP (emb)</td></tr><tr><td>SUSReg (Stochastic Baseline)</td><td>88.7</td><td>95.5</td><td>97.2</td><td>92.4</td><td>90.1</td></tr><tr><td>MMD (Induced SUSReg  $\bar{k}$ )</td><td>87.9</td><td>95.1</td><td>97.1</td><td>91.9</td><td>88.8</td></tr><tr><td>MMD (Heat,  $t = 5/d$ )</td><td>91.7</td><td>96.4</td><td>97.8</td><td>94.4</td><td>93.1</td></tr><tr><td>MMD (Bandlimited,  $L = 2$ )</td><td>90.5</td><td>95.0</td><td>96.2</td><td>93.1</td><td>92.9</td></tr><tr><td>KSD (Heat,  $t = 5/d$ )</td><td>91.9</td><td>96.5</td><td>97.7</td><td>94.5</td><td>92.5</td></tr><tr><td>KSD (Bandlimited,  $L = 2$ )</td><td>91.1</td><td>95.7</td><td>96.7</td><td>93.7</td><td>93.3</td></tr><tr><td>KL (Heat)</td><td>97.0</td><td>98.7</td><td>99.1</td><td>97.9</td><td>97.2</td></tr></table>

Table 5: Cloud Texture

<table><tr><td>Method</td><td>Recall@1</td><td>Recall@3</td><td>Recall@5</td><td>mAP</td><td>mAP (emb)</td></tr><tr><td>SUSReg (Stochastic Baseline)</td><td>89.7</td><td>95.4</td><td>96.8</td><td>92.9</td><td>91.4</td></tr><tr><td>MMD (Induced SUSReg  $\bar{k}$ )</td><td>90.2</td><td>95.9</td><td>97.6</td><td>93.4</td><td>91.4</td></tr><tr><td>MMD (Heat,  $t = 5/d$ ) 93.1</td><td>97.0</td><td>98.1</td><td>95.3</td><td>94.1</td><td></td></tr><tr><td>MMD (Bandlimited,  $L = 2$ ) 93.3</td><td>97.0</td><td>97.7</td><td>95.3</td><td>95.0</td><td></td></tr><tr><td>KSD (Heat,  $t = 5/d$ )</td><td>93.4</td><td>97.0</td><td>97.9</td><td>95.4</td><td>94.6</td></tr><tr><td>KSD (Bandlimited,  $L = 2$ )</td><td>91.5</td><td>95.9</td><td>96.9</td><td>94.0</td><td>93.1</td></tr><tr><td>KL (Heat)</td><td>96.3</td><td>98.0</td><td>98.5</td><td>97.3</td><td>96.2</td></tr></table>

Table 6: Wood Texture

<table><tr><td>Method</td><td>Recall@1<