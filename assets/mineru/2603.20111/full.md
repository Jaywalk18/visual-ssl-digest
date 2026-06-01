# Var-JEPA: A Variational Formulation of the Joint-Embedding Predictive Architecture – Bridging Predictive and Generative Self-Supervised Learning

Moritz Gogl ¨ 1 Christopher Yau 1 2

# Abstract

The Joint-Embedding Predictive Architecture (JEPA) is often seen as a non-generative alternative to likelihood-based self-supervised learning, emphasizing prediction in representation space rather than reconstruction in observation space. We argue that the resulting separation from probabilistic generative modeling is largely rhetorical rather than structural: the canonical JEPA design–coupled encoders with a context-to-target predictor–mirrors the variational posteriors and learned conditional priors obtained when variational inference is applied to a particular class of coupled latent-variable models, and standard JEPA can be viewed as a deterministic specialization in which regularization is imposed via architectural and training heuristics rather than an explicit likelihood. Building on this view, we derive the Variational JEPA (Var-JEPA), which makes the latent generative structure explicit by optimizing a single Evidence Lower Bound (ELBO). This yields meaningful representations without ad-hoc anti-collapse regularizers and allows principled uncertainty quantification in the latent space. We instantiate the framework for tabular data (Var-T-JEPA) and achieve strong representation learning and downstream performance, consistently improving over T-JEPA while remaining competitive with strong raw-feature baselines.

# 1. Introduction

A main goal in machine learning (ML) is to build models that truly understand and predict the world from data. Self-supervised learning approaches address this challenge by training models to predict one part of an input from another, thereby allowing to learn rich representations without requiring human labels. In this context, the Joint-Embedding Predictive Architecture (JEPA) was introduced by LeCun (2022) as a step toward creating a “world model”, 1University of Oxford, UK. 2Health Data Research UK. Correspondence to: Moritz Gogl ¨ <moritz.gogl@keble.ox.ac.uk>.

Preprint. March 23, 2026.

aiming to learn abstract representations of the world for intelligent decision-making. Unlike regular generative models that predict raw inputs such as pixels (van den Oord et al., 2016b;a; Parmar et al., 2018), the idea behind JEPA is to work entirely in the latent space by predicting the embedding of a target view y from the embedding of a related context signal x.

This is implemented using a context encoder $f ^ { \mathrm { c t x } }$ , a target encoder $f ^ { \mathrm { t r g } }$ , and a predictor g. The model creates a target representation $\hat { s } _ { y }$ from a context representation $s _ { x }$ and an optional auxiliary variable z, formulated as:

$$
s _ {x} = f _ {\mathrm{ctx}} (x), \quad s _ {y} = f _ {\mathrm{trg}} (y), \quad \hat {s} _ {y} = g (s _ {x}, z). \tag {1}
$$

The training objective minimizes the distance between the predicted and actual target representations, $\mathcal { L } _ { \mathrm { J E P A } } ~ =$ $\mathbb { E } \left[ \lVert \hat { s } _ { y } - \mathrm { s g } ( s _ { y } ) \rVert _ { 2 } ^ { 2 } \right]$ , where sg(·) is a stop-gradient operation (Grill et al., 2020). This intentional blocking of gradients into the target encoder is crucial in order to prevent the model from learning trivial predictive features.

While powerful, this design has two consequences. First, it is susceptible to representational collapse, where the encoders learn to output constant vectors, trivially minimizing the loss. This necessitates ad-hoc solutions, such as auxiliary costs $C ( \hat { s } _ { y } )$ (Drozdov et al., 2024; Balestriero & LeCun, 2025) or other custom regularization mechanisms such as Exponential Moving Average (EMA) (Assran et al., 2023; Bardes et al., 2024; Thimonier et al., 2025) to enforce representational diversity. Second, JEPA is typically framed as a fundamentally non-generative alternative to likelihoodbased self-supervised learning, which suggests a separation from probabilistic latent-variable modeling (LeCun, 2022).

This paper provides a novel perspective on this separation. We argue that it is mostly rhetorical rather than structural: the general JEPA design which includes coupled encoders and a predictor from context to target aligns closely with the variational posteriors and a learned conditional prior in a coupled variational autoencoder (VAE) (Kingma & Welling, 2013; Hao & Shafto, 2023). Concretely, we reverseengineer a probabilistic model whose ELBO yields the JEPA predictor as a learned latent-space conditional prior, and argue that standard JEPA is naturally interpreted as a deterministic specialization in which regularization is enforced by architectural and training heuristics rather than an explicit likelihood. While the original motivation for JEPA’s nongenerative framing was to avoid modeling observation noise, our view makes precise when reconstruction is a useful regularizer: it forces latents to encode predictive information while retaining a principled uncertainty semantics. This new framing not only connects predictive and generative selfsupervision, but also suggests an inherent route to avoiding collapse and enables latent-space uncertainty quantification.

Specifically, our contributions are: (1) We establish a novel formal link between JEPA and variational inference, reframing it as a deterministic latent variable model. (2) We introduce Var-JEPA, a generative JEPA derived from this link, with a corresponding ELBO objective. (3) We show how this ELBO naturally prevents representational collapse without needing the surrogate losses common in JEPA and provide a connection to the recently proposed SIGReg regularization in LeJEPA (Balestriero & LeCun, 2025). (4) We develop a practical Var-T-JEPA implementation for heterogeneous tabular data. (5) We empirically validate our approach in a controlled simulation study and across multiple downstream datasets, showing consistent gains over standard JEPA and strong baselines trained on raw features.

# 2. Related Work

JEPA-style predictive representation learning. JEPA was proposed as a general framework for learning predictive world models by matching representations (LeCun, 2022). Notable implementations include I-JEPA (Assran et al., 2023), which demonstrates that this approach allows to learn strong image representations without handcrafted data augmentations by predicting embeddings of masked regions from context regions. V-JEPA extends the approach to video, learning spatiotemporal representations that support understanding and prediction (Bardes et al., 2024). T-JEPA (Thimonier et al., 2025) adapts JEPA to feature-level masking and transformer tokenization for heterogeneous tabular data, showing competitive tabular representation learning without the heavy reliance on typical data augmentations.

Avoiding collapse via distributional regularization. Common JEPA implementations use EMA to stabilize training by slowly updating the target encoder (Assran et al., 2023; Bardes et al., 2024), but this heuristic approach does not directly control the distributional properties of the learned embeddings. In contrast, Drozdov et al. (2024) prevent collapse by explicitly regularizing embedding distributions via variance–covariance penalties that encourage feature-wise variance and discourage correlated dimensions. LeJEPA (Balestriero & LeCun, 2025) provides a complementary view: it argues that an (approximately) isotropic Gaussian embedding distribution is minimax-favorable for downstream probing under broad probe families, and it proposes Sketched Isotropic Gaussian Regularization (SIGReg) as a scalable way to match the aggregated embedding distribution to $\mathcal { N } ( 0 , I )$ via random one-dimensional projections. Formally, SIGReg applies a univariate statistical test T (typically the Epps-Pulley test (Epps & Pulley, 1983)) to projections of embeddings along random unit-norm directions $\mathbb { A } = \{ a _ { 1 } , \ldots , a _ { M } \}$ :

$$
\operatorname{SIGReg} \left(\mathbb {A}, \left\{e _ {n} \right\} _ {n = 1} ^ {N}\right) = \frac {1}{| \mathbb {A} |} \sum_ {a \in \mathbb {A}} T \left(\left\{a ^ {\top} e _ {n} \right\} _ {n = 1} ^ {N}\right), \tag {2}
$$

where $\{ e _ { n } \} _ { n = 1 } ^ { N }$ denotes a batch of embeddings and T measures the discrepancy between the empirical distribution of projected embeddings and the standard normal distribution. In our work, the ELBO provides per-sample regularization toward fixed priors for $s _ { x }$ and z, while we study SIGReg as an explicit aggregated-distribution regularizer (especially for $s _ { y } )$ in the simulation study.

Generative world models and latent-space modeling. World-model research has emphasized learning latent dynamics for prediction and planning (Ha & Schmidhuber, 2018). Compared to generative latent models and VAEs (Kingma & Welling, 2013; Bowman et al., 2016), JEPA emphasizes prediction in representation space and often avoids explicit likelihood modeling. Our main contribution is to show that a JEPA-like architecture arises naturally from a variational latent-variable model with a learned conditional prior. While concurrent work by Huang (2026) also explores a probabilistic formulation of JEPA for uncertainty-aware latent prediction, Var-JEPA instead formulates JEPA as a coupled latent-variable generative model with a unified ELBO, thereby bridging predictive joint-embedding learning and generative modeling in a single objective. This shows that Var-JEPA is not an incremental regularizer over JEPA, but a rigorous variational formulation that exposes the latent generative structure implicit in the predictor-encoder pattern.

# 3. The Variational Perspective on JEPA

# 3.1. Problem Formulation

We began by asking a simple structural question: if the predictive embedding steps in JEPA were interpreted as variational posteriors of a coupled VAE, what generative model would give rise to those posteriors?

This viewpoint naturally leads us to a novel reinterpretation of JEPA within a probabilistic latent-variable framework. Concretely, we replace JEPA’s deterministic encoders and predictor with conditional distributions and interpret the predictive pathway as a learned latent-space conditional prior. This creates a clear generative process over context, target, and auxiliary latent variables, from which a single variational objective (ELBO) emerges. Our formulation unifies JEPA-style predictive learning with reconstruction and conditional generation, while providing a rigorous mechanism for avoiding collapse through latent regularization.

Structure. Like JEPA, our model operates on context observations $\boldsymbol { x } \in \mathbb { R } ^ { D }$ and target observations $\boldsymbol { y } \in \mathbb { R } ^ { D }$ , learning latent representations $s _ { x } \in \mathbb { R } ^ { d }$ (context), $s _ { y } \in \mathbb { R } ^ { d }$ (target), and $z \in \mathbb { R } ^ { d _ { z } }$ (auxiliary predictive variable to capture variability in $s _ { y }$ that $s _ { x }$ cannot explain). The directed acyclic graph (DAG) follows the underlying generative process:

![](images/12101ae57efdf3cb10b3c6c20390eed36283c62454789c3dadf67bb23558d01a.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph LR
    x --> s_x
    s_x --> s_y
    s_y --> y
    z --> s_y
    s_y --> y
    style x fill:#fff,stroke:#000
    style s_x fill:#fff,stroke:#000
    style s_y fill:#fff,stroke:#000
    style y fill:#fff,stroke:#000
    note1["inference (JEPA)"]
    note2["inference (JEPA)"]
    s_x -.-> note1
    s_y -.-> note2
```
</details>

While a standard JEPA primarily operates with unidirectional mappings from observations to representations $( x $ $s _ { x } , y \to s _ { y } )$ , our variational framework requires bidirectional relationships to model both the generative and inference processes. For both the context and target, we learn two directions: encoding (from observations to latents) and reconstruction (from latents back to observations). A single ELBO objective ties these directions together and trains encoders and decoders jointly.

Objective and Factorization. Our main objective is to learn a generative model that maximizes the marginal loglikelihood of observed data pairs $( x , y )$ :

$$
\max _ {\theta} \mathbb {E} _ {(x, y) \sim \text { data }} [ \log p _ {\theta} (x, y) ] = \tag {3}
$$

$$
\max _ {\theta} \mathbb {E} _ {(x, y) \sim \mathrm{data}} \Big [ \log \iiint p _ {\theta} (x, y, s _ {x}, z, s _ {y}) d s _ {x} d z d s _ {y} \Big ].
$$

The log-likelihood log $p _ { \theta } ( x , y )$ measures how well our parameterized model (with parameters θ) explains the observed data. Following the DAG, the joint distribution $p _ { \theta } ( x , y , s _ { x } , z , s _ { y } )$ factorizes as

$$
p _ {\theta} (x, y, s _ {x}, z, s _ {y}) = p \left(s _ {x}\right) \cdot p (z) \cdot p _ {\theta} (x \mid s _ {x}) \tag {4}
$$

$$
\cdot p _ {\theta} (s _ {y} \mid s _ {x}, z) \cdot p _ {\theta} (y \mid s _ {y}).
$$

Generative Model Parameterization. We implement distributions in the generative model as Gaussians. The priors over latent variables are standard Gaussians, while the conditional distributions are parameterized by neural networks:

$$
p (s _ {x}) = \mathcal {N} (s _ {x}; 0, I), \tag {5a}
$$

$$
p (z) = \mathcal {N} (z; 0, I), \tag {5b}
$$

$$
p _ {\theta} (x \mid s _ {x}) = \mathcal {N} \big (x; U _ {\theta} ^ {x} (s _ {x}), \sigma_ {x} ^ {2} I \big), \tag {5c}
$$

$$
p _ {\theta} (s _ {y} \mid s _ {x}, z) = \mathcal {N} \big (s _ {y}; \mu_ {\theta} ^ {s _ {y}} (s _ {x}, z), \Sigma_ {\theta} ^ {s _ {y}} (s _ {x}, z) \big), \tag {5d}
$$

$$
p _ {\theta} (y \mid s _ {y}) = \mathcal {N} \big (y; U _ {\theta} ^ {y} (s _ {y}), \sigma_ {y} ^ {2} I \big). \tag {5e}
$$

The decoder networks $U _ { \theta } ^ { x } \colon \mathbb { R } ^ { D _ { \mathrm { h i d } } } \to \mathbb { R } ^ { D _ { \mathrm { o b s } } }$ and $U _ { \theta } ^ { y } \colon \mathbb { R } ^ { D _ { \mathrm { h i d } } } \to$ $\mathbb { R } ^ { D _ { \mathrm { o b s } } }$ reconstruct observations from their latent representations, where $D _ { \mathrm { o b s } }$ represents the observation dimension. The predictive network outputs $\mu _ { \theta } ^ { s _ { y } } ( s _ { x } , z )$ and $\Sigma _ { \theta } ^ { s _ { y } } \left( s _ { x } , z \right)$ are computed by neural networks with parameters θ, generating distributional parameters for the target latent $s _ { y }$ conditioned on context $s _ { x }$ and auxiliary variable z. The reconstruction noise parameters $\sigma _ { x } ^ { 2 }$ and $\sigma _ { y } ^ { 2 }$ may be set to 1 or learned globally to model observation uncertainty.

# 3.2. Variational Posterior

Since the integral in Eq. 4 is intractable under the given factorization and parameterization, due to complex neural network functions and high-dimensional latent spaces, we cannot optimize log $p _ { \theta } ( x , y )$ directly. Instead, we employ a variational inference approach by introducing a tractable variational posterior $q _ { \phi } ( s _ { x } , z , s _ { y } \mid x , y )$ as an approximation of the true posterior $p ( s _ { x } , z , s _ { y } \mid x , y )$ , and parameterized by neural networks with parameters $\phi$ . We factorize the variational posterior as:

$$
q _ {\phi} \left(s _ {x}, z, s _ {y} \mid x, y\right) = q _ {\phi} \left(s _ {x} \mid x\right) \cdot q _ {\phi} (z \mid s _ {x}) \tag {6}
$$

$$
\cdot q _ {\phi} (s _ {y} \mid s _ {x}, z, y).
$$

We adopt this factorization for the following reasons:

• The context latent $q _ { \phi } ( s _ { x } \mid x )$ depends only on the context observation $x ,$ ensuring that context representations are learned independently of target information.

• The auxiliary latent z depends only on the context representation $s _ { x }$ to prevent information leakage from targets during training.

• The target posterior $q _ { \phi } ( s _ { y } | s _ { x } , z , y )$ depends on both $s _ { x }$ and z as well as the target observation $y ,$ as we will use a reconstruction term to regularize the learning of $s _ { y }$ to encode meaningful target information, ensuring that the dependence on context is learned through proper predictive relationships rather than trivial shortcuts.

Variational Posterior Parameterization. Each component of the variational posterior is parameterized as a Gaussian distribution with learnable mean and covariance:

$$
q _ {\phi} (s _ {x} \mid x) = \mathcal {N} \big (s _ {x}; \mu_ {\phi} ^ {s _ {x}} (x), \Sigma_ {\phi} ^ {s _ {x}} (x) \big), \tag {7a}
$$

$$
q _ {\phi} (z \mid s _ {x}) = \mathcal {N} \big (z; \mu_ {\phi} ^ {z} (s _ {x}), \Sigma_ {\phi} ^ {z} (s _ {x}) \big), \tag {7b}
$$

$$
q _ {\phi} (s _ {y} \mid s _ {x}, z, y) = \mathcal {N} \big (s _ {y}; \mu_ {\phi} ^ {s _ {y}} (s _ {x}, z, y), \Sigma_ {\phi} ^ {s _ {y}} (s _ {x}, z, y) \big). \tag {7c}
$$

The inference networks $\mu _ { \phi } ^ { s _ { x } } , \Sigma _ { \phi } ^ { s _ { x } } , \mu _ { \phi } ^ { z } , \Sigma _ { \phi } ^ { z } , \mu _ { \phi } ^ { s _ { y } }$ sx Σ ϕ , µ ϕ sy , and Σsyϕ $\Sigma _ { \phi } ^ { s _ { y } }$ are implemented as neural networks that output the distributional parameters given their respective inputs.

# 3.3. Var-JEPA: Evidence Lower Bound

Having established the variational posterior, we derive the ELBO, a computable variational lower bound on the marginal log-likelihood via Jensen’s inequality:

$$
\log p _ {\theta} (x, y) \geq \mathbb {E} _ {q _ {\phi}} \left[ \log p _ {\theta} (x, y, s _ {x}, z, s _ {y}) - \right. \tag {8}
$$

$$
\left. \log q _ {\phi} (s _ {x}, z, s _ {y} \mid x, y) \right] \equiv \operatorname{ELBO} (x, y; \theta , \phi).
$$

More specifically, we achieve our goal of maximizing $\log p _ { \theta } ( x , y )$ by maximizing its tractable lower bound. Substituting the factorizations of $p _ { \theta }$ and $q _ { \phi }$ , we derive:

$$
\begin{array}{l} \operatorname{ELBO} (x, y; \theta , \phi) = \\ = \mathbb {E} _ {q _ {\phi}} \left[ \log p (s _ {x}) + \log p (z) + \log p _ {\theta} (x \mid s _ {x}) + \log p _ {\theta} (s _ {y} | s _ {x}, z) + \right. \\ \left. \log p _ {\theta} (y | s _ {y}) - \log q _ {\phi} (s _ {x} | x) - \log q _ {\phi} (z | s _ {x}) - \log q _ {\phi} (s _ {y} | s _ {x}, z, y) \right] \\ = \mathbb {E} _ {q _ {\phi} (s _ {x} | x)} \left[ \log p _ {\theta} (x | s _ {x}) \right] + \mathbb {E} _ {q _ {\phi} (s _ {x}, z, s _ {y} | x, y)} \left[ \log p _ {\theta} (y | s _ {y}) \right] + \\ \mathbb {E} _ {q _ {\phi} (s _ {x} | x)} \left[ \log p (s _ {x}) - \log q _ {\phi} (s _ {x} | x) \right] + \mathbb {E} _ {q _ {\phi} (s _ {x}, z | x)} \left[ \log p (z) - \right. \\ \left. \log q _ {\phi} (z \mid s _ {x}) \right] + \underbrace {\mathbb {E} _ {q _ {\phi} \left(s _ {x} , z , s _ {y} \mid x , y\right)} \left[ \log p _ {\theta} \left(s _ {y} \mid s _ {x} , z\right) \right]} _ {(a)} + \\ \underbrace {\mathbb {E} _ {q _ {\phi} (s _ {x} , z , s _ {y} | x , y)} \left[ - \log q _ {\phi} (s _ {y} \mid s _ {x} , z , y) \right]} _ {(b)}. \\ \end{array}
$$

We recognize that $\begin{array} { r } { \mathbb { E } _ { q _ { \phi } } [ \log p ( \cdot ) - \log q _ { \phi } ( \cdot ) ] = - \mathrm { K L } ( q _ { \phi } \| p ) } \end{array}$ and that the last two terms (a) and (b) can be combined into a single KL divergence term: $- \mathrm { K L } \big ( q _ { \phi } ( s _ { y } | s _ { x } , z , y ) \big | | p _ { \theta } ( s _ { y } |$ $s _ { x } , z ) \big )$ . This gives us the final expression for the ELBO:

$$
\operatorname{ELBO} (x, y; \theta , \phi) = \mathbb {E} _ {q _ {\phi} (s _ {x} | x)} [ \log p _ {\theta} (x \mid s _ {x}) ] + \tag {9}
$$

$$
\mathbb {E} _ {q _ {\phi} (s _ {x}, z, s _ {y} | x, y)} \left[ \log p _ {\theta} (y \mid s _ {y}) \right] - \mathrm{KL} \left(q _ {\phi} (s _ {x} \mid x) \| p (s _ {x})\right) -
$$

$$
\mathrm{KL} \big (q _ {\phi} (z | s _ {x}) \| p (z) \big) - \mathrm{KL} \big (q _ {\phi} (s _ {y} | s _ {x}, z, y) \| p _ {\theta} (s _ {y} | s _ {x}, z) \big)
$$

This objective can be viewed as a coupled VAE: $q _ { \phi } ( s _ { x } \mid$ | $x )$ , $q _ { \phi } ( z \mid s _ { x } )$ , and $q _ { \phi } ( s _ { y } \mid s _ { x } , z , y )$ are the amortized posteriors; $p _ { \theta } ( x \mid s _ { x } )$ and $p _ { \theta } ( y \mid s _ { y } )$ are the decoders; and $p _ { \theta } ( s _ { y } \mid s _ { x } , z )$ acts as a conditional prior that couples the two latent spaces, recovering the JEPA predictor as a generative component.

Our objective is to maximize the ELBO: $\begin{array} { r l } { \operatorname* { m a x } _ { \phi , \theta } } & { { } \mathbb { E } ( x , y ) \sim \mathrm { d a t a } \big [ \mathrm { E L B O } ( x , y ; \theta , \phi ) \big ] } \end{array}$ . In practice, we convert this to a minimization problem by defining a loss function as a weighted combination of the negative ELBO terms. Specifically, we minimize $\mathcal { L } _ { \mathrm { E L B O } } ( x , y ; \theta , \phi )$ , which decomposes the ELBO into 5 interpretable terms, each with its own scalar weight:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{ELBO}} (x, y; \theta , \phi) = \alpha^ {\text { rec }} \left(\underbrace {- \mathbb {E} _ {q _ {\phi} \left(s _ {x} \mid x\right)} \left[ \log p _ {\theta} \left(x \mid s _ {x}\right) \right]} _ {\mathcal {L} ^ {\text { rec }}}\right) \\ + \alpha^ {\text { gen }} \big (\underbrace {- \mathbb {E} _ {q _ {\phi} (s _ {x} , z , s _ {y} | x , y)} \left[ \log p _ {\theta} (y \mid s _ {y}) \right]} _ {\mathcal {L} ^ {\text { gen }}} \big) \\ + \alpha_ {s _ {x}} ^ {\mathrm{KL}} \underbrace {\mathrm{KL} \big (q _ {\phi} (s _ {x} \mid x) \| p (s _ {x}) \big)} _ {\mathcal {L} _ {s _ {x}} ^ {\mathrm{KL}}} + \alpha_ {z} ^ {\mathrm{KL}} \underbrace {\mathrm{KL} \big (q _ {\phi} (z \mid s _ {x}) \| p (z) \big)} _ {\mathcal {L} _ {z} ^ {\mathrm{KL}}} \\ + \alpha_ {s _ {y}} ^ {\mathrm{KL}} \underbrace {\mathrm{KL} \left(q _ {\phi} (s _ {y} \mid s _ {x} , z , y) \| p _ {\theta} (s _ {y} \mid s _ {x} , z)\right)} _ {\mathcal {L} _ {s _ {y}} ^ {\mathrm{KL}} \equiv - [ (\mathrm{a}) + (\mathrm{b}) ]} \tag {10} \\ \end{array}
$$

The loss terms have the following practical interpretation:

${ \mathcal { L } } ^ { \mathrm { r e c } }$ (Context Reconstruction): Measures the reconstruction quality–how accurately we can recover the original context observation x from its latent representation $s _ { x }$ .

$\mathcal { L } ^ { \mathrm { g e n } }$ (Target Generation): Measures the generation quality–how accurately the target latent $s _ { y }$ can reconstruct the actual target observation $y .$ .

$\mathcal { L } _ { s _ { x } } ^ { \mathrm { K L } }$ (KL on $s _ { x } ) \colon$ Regularizes the context latent distribution $s _ { x }$ toward the prior $\mathcal { N } ( 0 , I )$ , maintaining a wellbehaved latent space for $s _ { x }$ .

$\mathcal { L } _ { z } ^ { \mathrm { K L } }$ (KL on z): Regularizes the auxiliary latent distribution z toward the prior $\mathcal { N } ( 0 , I )$ , ensuring z doesn’t become overly complex or specialized.

$\mathcal { L } _ { s _ { y } } ^ { \mathrm { K L } }$ (KL on $\begin{array} { r } { s _ { y } ) \colon } \end{array}$ Regularizes the target latent posterior toward the generative model prediction.

(a) (Prediction): Measures the predictive accuracy–how well $s _ { x }$ and auxiliary variable z can jointly forecast the target latent representation $s _ { y } .$

(b) (Entropy): Maintains distributional diversity of the target posterior $q _ { \phi } ( s _ { y } \vert s _ { x } , z , y )$ by encouraging it to remain sufficiently uncertain, therefore preventing collapse.

# 3.4. JEPA vs. Var-JEPA Architectures

Fig. 1 illustrates the relationship between standard JEPA and our Var-JEPA. Both approaches share the same core prediction structure with context (x) and target (y) observations, their latent representations $s _ { x } , s _ { y } ,$ and auxiliary latent variable z. JEPA relies on inference and prediction networks, which requires surrogate costs $C ( \cdot )$ to prevent the collapse of representations. Var-JEPA adds generative networks (i.e., decoders) so the model can be trained with a variational objective which naturally prevents collapse.

# 3.5. Reparameterization Trick and Sampling Implementation

To backpropagate through stochastic latents, we apply the reparameterization trick (Kingma & Welling, 2013), expressing samples as deterministic functions of noise so gradients are well-defined. The latent variables are sampled using the following reparameterized forms:

$$
s _ {x} = \mu_ {\phi} ^ {s _ {x}} (x) + \left[ \Sigma_ {\phi} ^ {s _ {x}} (x) \right] ^ {1 / 2} \epsilon_ {s _ {x}},
$$

$$
z = \mu_ {\phi} ^ {z} (s _ {x}) + \left[ \Sigma_ {\phi} ^ {z} (s _ {x}) \right] ^ {1 / 2} \epsilon_ {z},
$$

$$
s _ {y} = \mu_ {\phi} ^ {s _ {y}} (s _ {x}, z, y) + \left[ \Sigma_ {\phi} ^ {s _ {y}} (s _ {x}, z, y) \right] ^ {1 / 2} \epsilon_ {s _ {y}},
$$

where $\epsilon _ { s _ { x } } , \epsilon _ { z } , \epsilon _ { s _ { y } } ~ \sim ~ \mathcal { N } ( 0 , I )$ are independent standard Gaussian noise vectors, and $[ \Sigma _ { \phi } ^ { ( \cdot ) } ( \cdot ) ] ^ { 1 / 2 }$ denotes the matrix square root of the covariance. In practice, we estimate the gradients by sampling a single realization of each latent variable per forward pass. The gradient estimator becomes:

$$
\nabla_ {\theta , \phi} \mathcal {L} \approx \nabla_ {\theta , \phi} \bigl [ \log p _ {\theta} (x, y, s _ {x}, z, s _ {y}) - \log q _ {\phi} (s _ {x}, z, s _ {y} | x, y) \bigr ],
$$

where each training step uses new samples $s _ { x } , z , s _ { y }$ drawn from the reparameterized distributions.

![](images/d102b85b031507cff61251349277de9cc49b7e2a23ead93f46960547fd7a982d.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph JEPA
        x --> fφ(x)
        fφ(x) --> sx
        sx --> gθ(sx,z)
        gθ(sx,z) --> ŝy
        z --> C(ŝy)
        y --> fφ(y)
        fφ(y) --> sy
        sy --> Lpred
    end

    subgraph Variational JEPA
        x --> qφ(sx|x)
        qφ(sx|x) --> sx
        sx --> p(z)
        p(z) --> KLzKL
        KLzKL --> qφ(z|sx)
        qφ(z|sx) --> Z
        Z --> KLzKL
        KLzKL --> pθ(y|sy)
        pθ(y|sy) --> Lgen
    end

    style JEPA fill:#f9f,stroke:#333
    style Variational JEPA fill:#bbf,stroke:#333
```
</details>

Figure 1. Comparison between JEPA (left) and Var-JEPA (right). Var-JEPA extends JEPA by replacing deterministic encoders and predictor with probabilistic distributions, and adding generative networks (decoders) to enable variational learning under a unified ELBO.

# 3.6. Theoretical Generative Inference

When Var-JEPA is interpreted as a truly generative model, it can generate target observations from context by sampling through the generative pathway. The generative inference procedure, applicable when targets are unavailable, is:

$$
s _ {x} \sim q _ {\phi} (s _ {x} \mid x) = \mathcal {N} \big (s _ {x}; \mu_ {\phi} ^ {s _ {x}} (x), \Sigma_ {\phi} ^ {s _ {x}} (x) \big),
$$

$$
z \sim p (z) = \mathcal {N} (z; 0, I),
$$

$$
s _ {y} \sim p _ {\theta} (s _ {y} \mid s _ {x}, z) = \mathcal {N} \big (s _ {y}; \mu_ {\theta} ^ {s _ {y}} (s _ {x}, z), \Sigma_ {\theta} ^ {s _ {y}} (s _ {x}, z) \big),
$$

$$
y \sim p _ {\theta} (y \mid s _ {y}) = \mathcal {N} \big (y; D _ {\theta} ^ {y} (s _ {y}), \sigma_ {y} ^ {2} I \big).
$$

This differs from training, where z is inferred via $q _ { \phi } ( z \mid s _ { x } )$ and the target posterior $q _ { \phi } ( s _ { y } \mid s _ { x } , z , y )$ has access to ground truth targets. For downstream representation learning tasks, where the goal is predictive rather than generative, we use deterministic embeddings by reporting posterior means rather than sampling (see Appendix D.5).

# 3.7. Relationship to LeJEPA

LeJEPA (Balestriero & LeCun, 2025) motivates isotropic Gaussian embeddings as minimax-optimal for downstream prediction under a broad class of probes. It enforces this distributional structure using SIGReg, which matches the aggregated embedding distribution to an isotropic Gaussian via random one-dimensional projections.

Var-JEPA relates to this picture through its variational regularization terms. For latent variables with a fixed standardnormal prior $( s _ { x }$ and z), the ELBO contains per-sample KL terms E $\mathrm { \Delta K L } ( q _ { \phi } ( \cdot ) \parallel \mathcal { N } ( 0 , I ) )$ . These admit the standard “ELBO surgery” decomposition (Hoffman & Johnson, 2016) into an aggregated posterior mismatch and an informationbottleneck term:

$$
\begin{array}{l} \mathbb {E} _ {x} \mathrm{KL} \left(q _ {\phi} (s _ {x} \mid x) \| \mathcal {N} (0, I)\right) = \\ \mathrm{KL} \big (q _ {\phi} (s _ {x}) \| \mathcal {N} (0, I) \big) + I _ {q _ {\phi}} (x; s _ {x}), \\ \end{array}
$$

with an analogous identity for z. In contrast, the targetlatent term in Var-JEPA is $\mathrm { K L } ( q _ { \phi } ( s _ { y } \mid s _ { x } , z , y ) \parallel p _ { \theta } ( s _ { y }$ | $s _ { x } , z ) )$ , i.e., regularization toward a learned conditional prior rather than $\mathcal { N } ( 0 , I )$ , and therefore does not decompose into an aggregated KL to a fixed reference distribution in the same way. This motivates studying SIGReg as an explicit aggregated-distribution regularizer, particularly for $s _ { y } ,$ and reporting LeJEPA-link diagnostics in the simulation study.

# 4. Experiments

# 4.1. Simulation Study

We evaluate Var-JEPA in a controlled synthetic setting designed to isolate the effects of latent regularisation, isotropy, and information bottlenecks while retaining a known ground-truth generative process. This setting allows us to analyse how the variational objective shapes the aggregated latent distribution and how this, in turn, affects downstream probe performance.

Data generation. We generate observations $( x _ { i } , y _ { i } )$ from a latent process with a mixture-structured context latent $s _ { x _ { i } }$ , a target latent $s _ { y _ { i } }$ that is correlated with $s _ { x _ { i } }$ , and an auxiliary factor $z _ { i }$ that influences $s _ { y _ { i } }$ and therefore $y _ { i }$ . Observations are obtained by passing these latents through nonlinear maps $h _ { x }$ and $h _ { y }$ and adding independent Gaussian noise:

$$
z _ {i} \sim \mathcal {N} (0, \Sigma_ {z}), s _ {x _ {i}} \sim \frac {1}{2} \mathcal {N} (0, \Sigma_ {x}) + \frac {1}{2} \mathcal {N} (\delta , \Sigma_ {x}),
$$

$$
s _ {y _ {i}} \mid s _ {x _ {i}}, z _ {i} \sim \mathcal {N} (s _ {x _ {i}} + A z _ {i}, \Sigma_ {y}),
$$

$$
x _ {i} = h _ {x} (s _ {x _ {i}}) + \epsilon_ {x}, \epsilon_ {x} \sim \mathcal {N} (0, \tau_ {x} ^ {2}),
$$

$$
y _ {i} = h _ {y} (s _ {y _ {i}}) + \epsilon_ {y}, \epsilon_ {y} \sim \mathcal {N} (0, \tau_ {y} ^ {2}).
$$

ELBO decomposition and interpretation. Although Var-JEPA does not explicitly optimize aggregated-posterior KL terms, the expected KL terms for latents with fixed priors (here $s _ { x }$ and z) admit the standard decomposition in Eq. 11. For the target latent, the ELBO instead contains a conditional-prior term that does not decompose into an aggregated KL to a fixed reference distribution in the same way. This highlights that Var-JEPA couples distributional regularization of latent spaces with fixed priors with information bottlenecks that penalize excessive dependence between inputs and latents.

Experimental variants. To study the relationship between Var-JEPA and LeJEPA, we perform a systematic ablation study comparing ten variants (labeled A–J in Table 1). Unless explicitly set to zero, all loss weight parameters use default values: $\dot { \alpha } ^ { \mathrm { r e c } } = \alpha ^ { \mathrm { g e n } } = \alpha _ { s _ { x } } ^ { \mathrm { K L } } = \dot { \alpha } _ { z } ^ { \mathrm { K L } } \dot { = } \alpha _ { s _ { u } } ^ { \mathrm { K L } } = 1$ s x αz sy for ELBO terms, and $\lambda _ { s _ { x } } = \lambda _ { s _ { y } } = 1 0$ for SIGReg regularization.

We compare the following variants: (A) the full ELBO objective (standard Var-JEPA); (B–D) ELBO augmented with SIGReg, where SIGReg is applied to $s _ { x }$ only $( \mathbf { B } , \lambda _ { s _ { y } } =$ 0), $s _ { y }$ only $( \mathbf { C } , \lambda _ { s _ { x } } = 0 )$ , or both (D). The general form of ELBO+SIGReg is:

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{ELBO} + \mathrm{SIGReg}} = \mathcal {L} _ {\mathrm{ELBO}} + \lambda_ {s _ {x}} \text { SIGReg } (\{s _ {x, i} \} _ {i = 1} ^ {B}, \mathcal {N} (0, I)) \\ + \lambda_ {s _ {y}} \text { SIGReg } (\{s _ {y, i} \} _ {i = 1} ^ {B}, \mathcal {N} (0, I)). \\ \end{array}
$$

We also consider (E–F) ELBO with some KL terms removed $( \alpha _ { s _ { x } } ^ { \mathrm { K L } } = 0$ sx and $\alpha _ { s _ { y } } ^ { \mathrm { K L } } = 0$ sy , respectively); (G) ELBO with reconstruction and generation terms removed $( \alpha ^ { \mathrm { r e c } } = \alpha ^ { \mathrm { g e n } } =$ 0), leaving only KL regularization; (H) the same as (G) but with SIGReg added; (I) ELBO with all KL terms removed $( \alpha _ { s _ { x } } ^ { \mathrm { K L } } = \alpha _ { z } ^ { \mathrm { K L } } = \alpha _ { s _ { u } } ^ { \mathrm { K L } } = 0 )$ αz = α , leaving only reconstruction sy and generation; and (J) the same as (I) but with SIGReg added, which corresponds to setting all KL terms to zero in $\mathcal { L } _ { \mathrm { E L B O + S I G R e g } }$ , resulting in a surrogate objective where persample KL terms are replaced by distribution-level SIGReg regularization. These variants allow us to disentangle the effects of per-sample variational regularization from direct control of the aggregated latent distributions.

Simulation study results. We evaluate a controlled comparison with identical architecture and training schedule, varying only the presence/scope of SIGReg and whether KL terms are included. Results are shown in Table 1, which reports distribution diagnostics: aggregated KL divergence to $\mathcal { N } ( 0 , I )$ , SIGReg-MSE (discrepancy via SIGReg), Frobenius norm of covariance deviation from identity $\| \mathrm { C o v } ( s ) -$ $I \| _ { F } ,$ , and mean norm $\| \mathbb { E } [ s ] \| _ { 2 }$ . Figure 2 shows how these diagnostics evolve during training for selected ablations.

The key finding is that the KL divergence terms in the ELBO achieve distributional properties for $s _ { x }$ that are comparable to those obtained with explicit aggregated-distribution regularization via SIGReg. This demonstrates that per-sample KL regularization toward fixed priors naturally enforces aggregated-distribution isotropy, without requiring additional regularization mechanisms. For $s _ { y }$ , the ELBO regularizes toward a learned conditional prior rather than $\mathcal { N } ( 0 , I )$ , which is the theoretically correct objective for the target latent; consequently, the aggregated distribution deviates from isotropic Gaussian, as expected.

Removing reconstruction terms (variant G) causes representational collapse, evidenced by probe accuracy dropping to near chance-level. Removing KL terms on $s _ { x }$ (variant E) leads to severe distributional collapse, with aggregated KL divergences and SIGReg-MSE values increasing dramatically. Removing all KL terms (variant I) causes even more severe collapse across all distributional metrics, demonstrating that KL regularization is essential for maintaining well-behaved latent distributions. When SIGReg is added to the no-KL variant (variant J), it partially compensates for the missing KL terms, but the full ELBO remains optimal. The training dynamics in Figure 2 reveal that these distributional properties emerge gradually during training, with KL terms providing stable regularization throughout.

# 4.2. Downstream Evaluation on Tabular Data

# 4.2.1. EXPERIMENTAL DETAILS

Var-T-JEPA. We evaluate our tabular implementation, Var-T-JEPA, which combines feature-level masking with the unified variational objective from Eq. 10. The model is inspired by the deterministic T-JEPA (Thimonier et al., 2025), but instantiates the Var-JEPA framework for tabular data by learning Gaussian latent embeddings and training the prediction and reconstruction pathways jointly under the ELBO. Var-T-JEPA tokenizes heterogeneous numerical and categorical features into a transformer sequence, infers Gaussian latent embeddings $s _ { x }$ and $s _ { y } ,$ , and trains a latentspace predictor via a coupled reconstruction–prediction objective that directly instantiates the Var-JEPA framework for heterogeneous tabular data. This yields both deterministic embeddings (via posterior means) and per-sample uncertainty estimates from the learned latent distributions. We defer a full conceptual description to Appendix A.

Datasets. We evaluate learned representations on five realworld tabular datasets: Adult (AD), Covertype (CO), Electricity (EL), Credit Card (CC), and Bank Marketing (BM), as well as MNIST (treated as tabular features with controllable input corruption) and a fully-synthetic simulation dataset (SIM); more details are provided in Appendix C.1.1.

Downstream and baseline predictors. For each dataset, we compare strong raw-feature baselines to the same predictor architectures trained on embeddings produced by Var-T-JEPA and T-JEPA (details in Appendix D.5). Following Thimonier et al. (2025), we consider a range of strong and widely-used tabular predictors: MLP, DCNv2 (Wang et al., 2021), ResNet (He et al., 2016), AutoInt (Song et al., 2019), FT-Transformer (Gorishniy et al., 2021), and XGBoost (Chen & Guestrin, 2016).

Selective evaluation via uncertainty. Var-T-JEPA provides a per-sample uncertainty estimate; we report selectiveevaluation where the most uncertain 10%, 20%, or 50% of samples are discarded before computing accuracy (“Var-T-JEPA (10%)”, etc.), illustrating the coverage–accuracy trade-off when abstaining on low-confidence samples.

Table 1. Simulation study: Var-JEPA and SIGReg ablations (mean ± std over 5 runs). We report linear-probe accuracy (predicting the mixture component) and LeJEPA-link diagnostics for the aggregated latent distributions of $s _ { x }$ and $s _ { y } .$ For $s _ { y }$ we additionally report the mean conditional-prior coupling term E KL ${ \mathrm { \Omega } } _ { \cdot } ( q _ { \phi } ( s _ { y } \vert s _ { x } , z , y ) \parallel p _ { \theta } ( s _ { y } \vert s _ { x } , z ) )$ ). 

<table><tr><td>Exp.</td><td>Objective</td><td>Acc(s)</td><td> $\text{KL}_{\text{agg}}(q(s)\|\mathcal{N}(0,I))$ </td><td>SIGReg-MSE(s)</td><td> $\| \text{Cov}(s)-I \|_F$ </td><td> $\| \mathbb{E}[s] \|_2$ </td></tr><tr><td colspan="7">Context latent diagnostics ( $s_x$ )</td></tr><tr><td>(A)</td><td>ELBO (Var-JEPA)</td><td>0.996 ± 0.002</td><td>0.113 ± 0.037</td><td>4.0e-4 ± 1.0e-4</td><td>0.649 ± 0.127</td><td>0.082 ± 0.009</td></tr><tr><td>(B)</td><td>ELBO+SIGReg ( $\lambda_{sx}=10, \lambda_{sy}=0$ )</td><td>0.996 ± 0.002</td><td>0.108 ± 0.032</td><td>4.0e-4 ± 1.0e-4</td><td>0.639 ± 0.113</td><td>0.077 ± 0.018</td></tr><tr><td>(C)</td><td>ELBO+SIGReg ( $\lambda_{sx}=0, \lambda_{sy}=10$ )</td><td>0.996 ± 0.002</td><td>0.110 ± 0.033</td><td>4.0e-4 ± 1.0e-4</td><td>0.647 ± 0.118</td><td>0.077 ± 0.018</td></tr><tr><td>(D)</td><td>ELBO+SIGReg</td><td>0.996 ± 0.002</td><td>0.107 ± 0.030</td><td>3.9e-4 ± 1.0e-4</td><td>0.634 ± 0.108</td><td>0.078 ± 0.018</td></tr><tr><td>(E)</td><td>ELBO ( $\alpha_{s}^{\text{KL}}=0$ )</td><td>0.996 ± 0.002</td><td>8.374 ± 1.944</td><td>5.1e-2 ± 1.9e-2</td><td>7.197 ± 1.522</td><td>2.550 ± 0.607</td></tr><tr><td>(F)</td><td>ELBO ( $\alpha_{sy}^{\text{KL}}=0$ )</td><td>0.996 ± 0.002</td><td>0.098 ± 0.038</td><td>3.0e-4 ± 1.0e-4</td><td>0.597 ± 0.124</td><td>0.083 ± 0.010</td></tr><tr><td>(G)</td><td>ELBO ( $\alpha^{\text{rec}}=\alpha^{\text{gen}}=0$ )</td><td>0.571 ± 0.010</td><td>0.015 ± 0.002</td><td>1.6e-4 ± 1.1e-5</td><td>0.229 ± 0.016</td><td>0.055 ± 0.005</td></tr><tr><td>(H)</td><td>ELBO ( $\alpha^{\text{rec}}=\alpha^{\text{gen}}=0$ )+SIGReg</td><td>0.834 ± 0.044</td><td>0.015 ± 0.001</td><td>1.7e-4 ± 1.4e-5</td><td>0.225 ± 0.008</td><td>0.065 ± 0.006</td></tr><tr><td>(I)</td><td>ELBO ( $\alpha_{s}^{\text{KL}}=\alpha_{z}^{\text{KL}}=\alpha_{sy}^{\text{KL}}=0$ )</td><td>0.995 ± 0.002</td><td>10.465 ± 2.207</td><td>5.7e-2 ± 1.9e-2</td><td>8.807 ± 1.208</td><td>3.100 ± 0.594</td></tr><tr><td>(J)</td><td>ELBO ( $\alpha_{s}^{\text{KL}}=\alpha_{z}^{\text{KL}}=\alpha_{sy}^{\text{KL}}=0$ ) + SIGReg</td><td>0.996 ± 0.002</td><td>2.115 ± 0.093</td><td>3.3e-3 ± 5.4e-4</td><td>2.594 ± 0.077</td><td>0.205 ± 0.032</td></tr></table>

Target latent diagnostics $( s _ { y } )$ 

<table><tr><td>(A)</td><td>ELBO (Var-JEPA)</td><td>0.993 ± 0.001</td><td>3.530 ± 0.377</td><td>1.7e-2 ± 1.7e-3</td><td>4.727 ± 0.528</td><td>1.320 ± 0.180</td></tr><tr><td>(B)</td><td>ELBO+SIGReg ( $\lambda_{sx} = 10, \lambda_{sy} = 0$ )</td><td>0.992 ± 0.002</td><td>3.511 ± 0.449</td><td>1.4e-2 ± 3.0e-3</td><td>4.748 ± 0.651</td><td>1.272 ± 0.198</td></tr><tr><td>(C)</td><td>ELBO+SIGReg ( $\lambda_{sx} = 0, \lambda_{sy} = 10$ )</td><td>0.993 ± 0.002</td><td>1.999 ± 0.058</td><td>4.6e-3 ± 4.0e-4</td><td>3.423 ± 0.215</td><td>0.189 ± 0.030</td></tr><tr><td>(D)</td><td>ELBO+SIGReg</td><td>0.992 ± 0.002</td><td>1.998 ± 0.057</td><td>4.6e-3 ± 4.0e-4</td><td>3.423 ± 0.216</td><td>0.189 ± 0.030</td></tr><tr><td>(E)</td><td>ELBO ( $\alpha_{sx}^{\text{KL}} = 0$ )</td><td>0.993 ± 0.001</td><td>4.051 ± 0.517</td><td>2.1e-2 ± 3.4e-3</td><td>4.985 ± 0.591</td><td>1.589 ± 0.201</td></tr><tr><td>(F)</td><td>ELBO ( $\alpha_{sy}^{\text{KL}} = 0$ )</td><td>0.983 ± 0.003</td><td>6.201 ± 1.319</td><td>2.8e-2 ± 5.5e-3</td><td>6.288 ± 1.636</td><td>1.924 ± 0.348</td></tr><tr><td>(G)</td><td>ELBO ( $\alpha^{\text{rec}} = \alpha^{\text{gen}} = 0$ )</td><td>0.543 ± 0.033</td><td>0.055 ± 0.006</td><td>4.2e-4 ± 3.2e-5</td><td>0.428 ± 0.021</td><td>0.164 ± 0.019</td></tr><tr><td>(H)</td><td>ELBO ( $\alpha^{\text{rec}} = \alpha^{\text{gen}} = 0$ )+SIGReg</td><td>0.821 ± 0.067</td><td>0.017 ± 0.001</td><td>1.6e-4 ± 1.4e-5</td><td>0.239 ± 0.018</td><td>0.067 ± 0.014</td></tr><tr><td>(I)</td><td>ELBO ( $\alpha_{sx}^{\text{KL}} = \alpha_{z}^{\text{KL}} = \alpha_{sy}^{\text{KL}} = 0$ )</td><td>0.984 ± 0.004</td><td>20.276 ± 11.331</td><td>8.4e-2 ± 4.3e-2</td><td>11.342 ± 5.062</td><td>4.871 ± 2.150</td></tr><tr><td>(J)</td><td>ELBO ( $\alpha_{sx}^{\text{KL}} = \alpha_{z}^{\text{KL}} = \alpha_{sy}^{\text{KL}} = 0$ )+SIGReg</td><td>0.983 ± 0.001</td><td>2.215 ± 0.249</td><td>3.4e-3 ± 7.8e-4</td><td>2.825 ± 0.271</td><td>0.248 ± 0.068</td></tr></table>

![](images/8819b74e84a419c39ea958327e20295c9076d6c31be7b0b5b4d3fcd4545e8fcc.jpg)

![](images/ea018f8732fd6f5c944c1b50be87a927ffd5c71db9300b9f58e50ef4d63e7316.jpg)

![](images/d4bda51ddcc577b7862cdbe11cfbaa8167e06ff401a4e950254c9ff763a04876.jpg)

![](images/ff5b02d5513b2113ff19566ee46121de1fe066eeb71456402b264b7ff475403f.jpg)

<details>
<summary>line</summary>

| epoch | SIGReg-MSE(s_y) |
| ----- | --------------- |
| 0     | 0.04            |
| 10    | 0.10            |
| 20    | 0.08            |
| 30    | 0.09            |
| 40    | 0.08            |
</details>

![](images/c38fcae15bfc4b6810f696f1e6819929416999365f31597c0afa229062c45936.jpg)

<details>
<summary>line</summary>

| epoch | Line 1 | Line 2 | Line 3 | Line 4 | Line 5 |
|-------|--------|--------|--------|--------|--------|
| 0     | 12.0   | 10.0   | 8.0    | 6.0    | 4.0    |
| 20    | 11.0   | 7.0    | 5.5    | 4.5    | 3.5    |
| 40    | 11.5   | 6.5    | 5.0    | 4.0    | 3.0    |
</details>

![](images/bee6f6a68ffc94f573fdc1039b0bbd38018db45577539a24ae63c1764a5373a3.jpg)

![](images/dd3e8f7ec84c66befabdaf18fc5b0637490e64e5f7cc32352ae093021688eea1.jpg)  
Figure 2. Epoch-wise distribution diagnostics for selected experiments. We show how aggregated KL divergences, SIGReg-MSE, isotropy metrics, and conditional-prior coupling evolve during training.

# 4.2.2. RESULTS

Table 2 summarizes downstream test accuracy across datasets and predictor families. Across the real-world tabular datasets (AD, CO, EL, CC, BM), Var-T-JEPA yields competitive embeddings for downstream classifiers, and selective evaluation exhibits a clear coverage–accuracy tradeoff: discarding the most uncertain test samples improves performance on the retained subset across model families. In contrast, the deterministic T-JEPA baseline can suffer from representation collapse on some datasets, leading to poor downstream performance and reduced robustness.

For the (semi-)synthetic MNIST and SIM datasets, Var-T-JEPA additionally produces uncertainty signals that are consistent with the underlying simulated corruption/ambiguity structure: Figure 3 visualizes (left) risk–coverage curves induced by abstaining on samples with highest latent uncertainty, (middle) a positive association between standardized latent uncertainty and the simulated uncertainty score, and (right) ROC curves showing that latent uncertainty can identify high-ambiguity samples (defined by a high-quantile threshold on the simulated uncertainty score). These diagnostics complement the selective-evaluation results in Table 2 by showing that the learned latent uncertainty is not only useful for abstention, but also aligned with the known uncertainty signal in these controlled settings. Additional experimental results, including sensitivity analyses, are provided in Appendix B.

# 5. Discussion

We provide a novel reinterpretation of the JEPA design pattern as variational inference in a coupled latent-variable model: the predictor is a learned conditional prior $p _ { \theta } ( s _ { y }$ | $s _ { x } , z )$ , and ad hoc costs are indirect forms of distributional control that the ELBO regularizes. This clarifies that the common “JEPA vs. generative modeling” dichotomy is largely a matter of framing: JEPA occupies a particular (im-

Table 2. Downstream performance comparison across tabular datasets. Results (mean ± std over 5 downstream model runs; XGBoost single deterministic run) show test accuracy. AD=Adult, CO=Covertype, EL=Electricity, CC=Credit Card Default, BM=Bank Marketing. Light blue rows indicate selective evaluation (reduced coverage). 

<table><tr><td>Method</td><td>AD ↑</td><td>CO ↑</td><td>EL ↑</td><td>CC ↑</td><td>BM ↑</td><td>MNIST ↑</td><td>SIM ↑</td></tr><tr><td>MLP</td><td> $0.849±1.4e^{-3}$ </td><td> $0.750±3.7e^{-3}$ </td><td> $0.781±3.6e^{-3}$ </td><td> $0.816±2.5e^{-3}$ </td><td> $0.898±1.4e^{-3}$ </td><td> $0.822±1.1e^{-2}$ </td><td> $0.823±7.5e^{-3}$ </td></tr><tr><td>+T-JEPA</td><td> $0.849±4.4e^{-3}$ </td><td> $0.408±2.3e^{-1}$ </td><td> $0.627±4.1e^{-2}$ </td><td> $0.774±0e^{-0}$ </td><td> $0.616±3.7e^{-1}$ </td><td> $0.113±6.2e^{-3}$ </td><td> $0.692±3.5e^{-2}$ </td></tr><tr><td>+Var-T-JEPA</td><td> $0.852±2.1e^{-3}$ </td><td> $0.679±2.9e^{-2}$ </td><td> $0.790±3.3e^{-3}$ </td><td> $0.818±1.5e^{-3}$ </td><td> $0.900±2.0e^{-3}$ </td><td> $0.822±1.8e^{-2}$ </td><td> $0.695±1.4e^{-2}$ </td></tr><tr><td>+Var-T-JEPA (10%)</td><td> $0.865±2.0e^{-3}$ </td><td> $0.696±2.8e^{-2}$ </td><td> $0.793±2.9e^{-3}$ </td><td> $0.827±1.4e^{-3}$ </td><td> $0.905±2.2e^{-3}$ </td><td> $0.838±1.6e^{-2}$ </td><td> $0.705±1.3e^{-2}$ </td></tr><tr><td>+Var-T-JEPA (20%)</td><td> $0.883±1.7e^{-3}$ </td><td> $0.697±2.7e^{-2}$ </td><td> $0.795±2.8e^{-3}$ </td><td> $0.835±1.2e^{-3}$ </td><td> $0.908±2.5e^{-3}$ </td><td> $0.856±1.9e^{-2}$ </td><td> $0.706±1.5e^{-2}$ </td></tr><tr><td>+Var-T-JEPA (50%)</td><td> $0.921±2.1e^{-3}$ </td><td> $0.706±1.4e^{-2}$ </td><td> $0.804±3.9e^{-3}$ </td><td> $0.841±7.0e^{-4}$ </td><td> $0.921±1.4e^{-3}$ </td><td> $0.901±2.1e^{-2}$ </td><td> $0.729±1.5e^{-2}$ </td></tr><tr><td>DCNv2</td><td> $0.762±6.2e^{-4}$ </td><td> $0.757±6.2e^{-3}$ </td><td> $0.825±7.3e^{-4}$ </td><td> $0.814±1.1e^{-3}$ </td><td> $0.896±1.1e^{-3}$ </td><td> $0.875±4.5e^{-3}$ </td><td> $0.714±1.8e^{-3}$ </td></tr><tr><td>+T-JEPA</td><td> $0.851±1.5e^{-3}$ </td><td> $0.546±8.9e^{-3}$ </td><td> $0.769±2.3e^{-3}$ </td><td> $0.792±1.7e^{-2}$ </td><td> $0.895±3.1e^{-4}$ </td><td> $0.841±4.6e^{-2}$ </td><td> $0.772±3.7e^{-3}$ </td></tr><tr><td>+Var-T-JEPA</td><td> $0.851±1.8e^{-3}$ </td><td> $0.778±6.1e^{-3}$ </td><td> $0.830±1.9e^{-3}$ </td><td> $0.814±2.4e^{-3}$ </td><td> $0.896±6.9e^{-4}$ </td><td> $0.885±4.8e^{-3}$ </td><td> $0.720±7.8e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (10%)</td><td> $0.864±1.7e^{-3}$ </td><td> $0.785±6.8e^{-3}$ </td><td> $0.831±2.6e^{-3}$ </td><td> $0.823±2.7e^{-3}$ </td><td> $0.903±7.9e^{-4}$ </td><td> $0.898±5.2e^{-3}$ </td><td> $0.728±7.2e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (20%)</td><td> $0.882±1.6e^{-3}$ </td><td> $0.783±6.8e^{-3}$ </td><td> $0.831±2.1e^{-3}$ </td><td> $0.833±2.5e^{-3}$ </td><td> $0.907±1.2e^{-3}$ </td><td> $0.909±5.0e^{-3}$ </td><td> $0.726±7.5e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (50%)</td><td> $0.920±1.3e^{-3}$ </td><td> $0.782±5.8e^{-3}$ </td><td> $0.831±2.7e^{-3}$ </td><td> $0.840±1.5e^{-3}$ </td><td> $0.918±1.6e^{-3}$ </td><td> $0.936±3.0e^{-3}$ </td><td> $0.757±5.4e^{-3}$ </td></tr><tr><td>ResNet</td><td> $0.849±1.9e^{-3}$ </td><td> $0.776±7.0e^{-3}$ </td><td> $0.823±2.5e^{-3}$ </td><td> $0.813±2.0e^{-3}$ </td><td> $0.897±2.3e^{-3}$ </td><td> $0.860±7.0e^{-3}$ </td><td> $0.878±5.9e^{-3}$ </td></tr><tr><td>+T-JEPA</td><td> $0.852±2.5e^{-3}$ </td><td> $0.540±6.3e^{-2}$ </td><td> $0.808±7.4e^{-3}$ </td><td> $0.817±3.7e^{-3}$ </td><td> $0.897±4.7e^{-3}$ </td><td> $0.854±3.0e^{-2}$ </td><td> $0.693±6.6e^{-2}$ </td></tr><tr><td>+Var-T-JEPA</td><td> $0.854±1.4e^{-3}$ </td><td> $0.779±1.1e^{-2}$ </td><td> $0.820±7.6e^{-3}$ </td><td> $0.813±5.0e^{-3}$ </td><td> $0.900±1.2e^{-3}$ </td><td> $0.871±1.3e^{-2}$ </td><td> $0.564±1.1e^{-1}$ </td></tr><tr><td>+Var-T-JEPA (10%)</td><td> $0.868±9.2e^{-4}$ </td><td> $0.785±1.1e^{-2}$ </td><td> $0.822±7.0e^{-3}$ </td><td> $0.824±2.9e^{-3}$ </td><td> $0.906±1.1e^{-3}$ </td><td> $0.887±9.7e^{-3}$ </td><td> $0.573±1.1e^{-1}$ </td></tr><tr><td>+Var-T-JEPA (20%)</td><td> $0.886±8.2e^{-4}$ </td><td> $0.782±1.3e^{-2}$ </td><td> $0.823±6.8e^{-3}$ </td><td> $0.833±2.5e^{-3}$ </td><td> $0.910±9.2e^{-4}$ </td><td> $0.900±8.7e^{-3}$ </td><td> $0.583±1.1e^{-1}$ </td></tr><tr><td>+Var-T-JEPA (50%)</td><td> $0.924±6.6e^{-4}$ </td><td> $0.787±1.4e^{-2}$ </td><td> $0.826±7.0e^{-3}$ </td><td> $0.841±2.2e^{-3}$ </td><td> $0.920±2.0e^{-3}$ </td><td> $0.936±5.2e^{-3}$ </td><td> $0.612±1.2e^{-1}$ </td></tr><tr><td>AutoInt</td><td> $0.761±7.8e^{-4}$ </td><td> $0.753±1.6e^{-2}$ </td><td> $0.754±1.1e^{-2}$ </td><td> $0.817±1.4e^{-3}$ </td><td> $0.901±1.5e^{-3}$ </td><td> $0.809±1.6e^{-2}$ </td><td> $0.897±1.9e^{-2}$ </td></tr><tr><td>+T-JEPA</td><td> $0.854±2.6e^{-3}$ </td><td> $0.448±1.3e^{-1}$ </td><td> $0.756±2.0e^{-2}$ </td><td> $0.804±3.2e^{-4}$ </td><td> $0.893±3.7e^{-3}$ </td><td> $0.817±7.4e^{-3}$ </td><td> $0.775±7.9e^{-3}$ </td></tr><tr><td>+Var-T-JEPA</td><td> $0.854±2.5e^{-3}$ </td><td> $0.752±4.5e^{-3}$ </td><td> $0.822±2.7e^{-3}$ </td><td> $0.816±1.3e^{-3}$ </td><td> $0.900±1.5e^{-3}$ </td><td> $0.810±4.9e^{-3}$ </td><td> $0.723±7.4e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (10%)</td><td> $0.868±3.0e^{-3}$ </td><td> $0.756±4.2e^{-3}$ </td><td> $0.823±3.6e^{-3}$ </td><td> $0.824±1.5e^{-3}$ </td><td> $0.906±1.5e^{-3}$ </td><td> $0.825±2.0e^{-3}$ </td><td> $0.730±6.7e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (20%)</td><td> $0.885±2.2e^{-3}$ </td><td> $0.754±4.7e^{-3}$ </td><td> $0.822±3.9e^{-3}$ </td><td> $0.834±1.1e^{-3}$ </td><td> $0.909±1.1e^{-3}$ </td><td> $0.841±1.9e^{-3}$ </td><td> $0.732±6.2e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (50%)</td><td> $0.923±1.5e^{-3}$ </td><td> $0.749±6.2e^{-3}$ </td><td> $0.823±4.3e^{-3}$ </td><td> $0.840±9.6e^{-4}$ </td><td> $0.920±1.2e^{-3}$ </td><td> $0.880±4.2e^{-3}$ </td><td> $0.757±7.0e^{-3}$ </td></tr><tr><td>FT-Trans</td><td> $0.761±4.1e^{-4}$ </td><td> $0.747±9.1e^{-3}$ </td><td> $0.814±3.9e^{-3}$ </td><td> $0.820±1.0e^{-3}$ </td><td> $0.901±6.7e^{-4}$ </td><td> $0.877±2.4e^{-2}$ </td><td> $0.962±1.4e^{-3}$ </td></tr><tr><td>+T-JEPA</td><td> $0.854±1.0e^{-3}$ </td><td> $0.522±1.3e^{-2}$ </td><td> $0.705±4.7e^{-2}$ </td><td> $0.802±2.1e^{-3}$ </td><td> $0.886±3.7e^{-4}$ </td><td> $0.317±2.2e^{-1}$ </td><td> $0.723±1.2e^{-2}$ </td></tr><tr><td>+Var-T-JEPA</td><td> $0.852±1.2e^{-3}$ </td><td> $0.748±5.6e^{-3}$ </td><td> $0.798±4.2e^{-3}$ </td><td> $0.818±3.6e^{-4}$ </td><td> $0.900±4.1e^{-4}$ </td><td> $0.864±9.3e^{-3}$ </td><td> $0.716±7.7e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (10%)</td><td> $0.867±1.4e^{-3}$ </td><td> $0.756±5.6e^{-3}$ </td><td> $0.800±3.7e^{-3}$ </td><td> $0.827±4.7e^{-4}$ </td><td> $0.905±6.3e^{-4}$ </td><td> $0.879±9.4e^{-3}$ </td><td> $0.726±8.2e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (20%)</td><td> $0.885±1.4e^{-3}$ </td><td> $0.753±5.8e^{-3}$ </td><td> $0.802±3.5e^{-3}$ </td><td> $0.836±7.0e^{-4}$ </td><td> $0.909±6.6e^{-4}$ </td><td> $0.891±9.8e^{-3}$ </td><td> $0.727±6.5e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (50%)</td><td> $0.923±1.2e^{-3}$ </td><td> $0.746±7.9e^{-3}$ </td><td> $0.809±2.1e^{-3}$ </td><td> $0.842±6.3e^{-4}$ </td><td> $0.922±8.8e^{-4}$ </td><td> $0.918±6.5e^{-3}$ </td><td> $0.748±2.5e^{-3}$ </td></tr><tr><td>XGBoost</td><td>0.864</td><td>0.807</td><td>0.917</td><td>0.811</td><td>0.900</td><td>0.881</td><td>0.949</td></tr><tr><td>+T-JEPA</td><td>0.854</td><td>0.807</td><td>0.860</td><td>0.801</td><td>0.898</td><td>0.871</td><td>0.851</td></tr><tr><td>+Var-T-JEPA</td><td>0.855</td><td>0.809</td><td>0.888</td><td>0.806</td><td>0.904</td><td>0.872</td><td>0.945</td></tr><tr><td>+Var-T-JEPA (10%)</td><td>0.874</td><td>0.818</td><td>0.890</td><td>0.817</td><td>0.910</td><td>0.883</td><td>0.948</td></tr><tr><td>+Var-T-JEPA (20%)</td><td>0.893</td><td>0.818</td><td>0.891</td><td>0.823</td><td>0.912</td><td>0.889</td><td>0.951</td></tr><tr><td>+Var-T-JEPA (50%)</td><td>0.928</td><td>0.816</td><td>0.891</td><td>0.832</td><td>0.921</td><td>0.913</td><td>0.955</td></tr></table>

![](images/405e04e785781cef67b090695545d0e12523b142259126edfca3816c130a489b.jpg)  
Figure 3. Uncertainty quantification on the MNIST (A) and SIM (B) datasets. Left: risk–coverage curve induced by abstaining on samples with highest latent uncertainty. Middle: standardized latent uncertainty versus simulated uncertainty. Right: ROC curve for detecting high-ambiguity samples from latent uncertainty.

plicit, often deterministic) point in the design space of latentvariable generative models. By making this latent generative structure explicit, we unify predictive and generative selfsupervised learning within a single principled framework. Empirically, ELBO training yields usable representations without heuristic anti-collapse objectives (e.g., auxiliary regularizers such as EMA or distribution-matching penalties) and enables rigorous uncertainty estimates from posterior covariances. We introduce Var-T-JEPA as a concrete implementation for tabular data, and in downstream tabular evaluation it produces competitive embeddings across various datasets and supports selective prediction based on estimated latent uncertainty. More broadly, we observe that per-sample KL terms to fixed priors drive aggregated distributional behavior comparable to explicit distributional regularizers such as SIGReg, while leaving the target latent governed by its learned conditional prior. Future work includes scaling to vision and video, and extending to settings where target observations are absent at test time, making conditional generation central.

# Acknowledgements

MG is supported by the EPSRC Centre for Doctoral Training in Health Data Science (EP/S02428X/1). CY is supported by a UKRI Turing AI Acceleration Fellowship (EP/V023233/2).

# References

Assran, M., Duval, Q., Misra, I., Bojanowski, P., Vincent, P., Rabbat, M., LeCun, Y., and Ballas, N. Self-supervised learning from images with a joint-embedding predictive architecture, 2023.   
Balestriero, R. and LeCun, Y. Lejepa: Provable and scalable self-supervised learning without the heuristics, 2025.   
Bardes, A., Garrido, Q., Ponce, J., Chen, X., Rabbat, M., LeCun, Y., Assran, M., and Ballas, N. Revisiting feature prediction for learning visual representations from video, 2024.   
Blackard, J. Covertype. UCI Machine Learning Repository, 1998. doi: 10.24432/C50K5N.   
Bowman, S. R., Vilnis, L., Vinyals, O., Dai, A. M., Jozefowicz, R., and Bengio, S. Generating sentences from a continuous space. In 20th SIGNLL Conference on Computational Natural Language Learning, CoNLL 2016, pp. 10–21. Association for Computational Linguistics (ACL), 2016.   
Chen, T. and Guestrin, C. Xgboost: A scalable tree boosting system. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD ’16, pp. 785–794. ACM, August 2016. doi: 10.1145/2939672.2939785.   
Drozdov, K., Shwartz-Ziv, R., and LeCun, Y. Video representation learning with joint-embedding predictive architectures, 2024.   
Epps, T. W. and Pulley, L. B. A test for normality based on the empirical characteristic function. Biometrika, 70(3): 723–726, 1983. ISSN 1464-3510. doi: 10.1093/biomet/ 70.3.723.   
Gorishniy, Y., Rubachev, I., Khrulkov, V., and Babenko, A. Revisiting deep learning models for tabular data. In Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P., and Vaughan, J. W. (eds.), Advances in Neural Information Processing Systems, volume 34, pp. 18932–18943. Curran Associates, Inc., 2021.   
Grill, J.-B., Strub, F., Altche, F., Tallec, C., Richemond, P., ´ Buchatskaya, E., Doersch, C., Avila Pires, B., Guo, Z., Gheshlaghi Azar, M., Piot, B., kavukcuoglu, k., Munos, R., and Valko, M. Bootstrap your own latent - a new

approach to self-supervised learning. In Larochelle, H., Ranzato, M., Hadsell, R., Balcan, M., and Lin, H. (eds.), Advances in Neural Information Processing Systems, volume 33, pp. 21271–21284. Curran Associates, Inc., 2020.   
Ha, D. and Schmidhuber, J. World models. 2018. doi: 10.48550/ARXIV.1803.10122.   
Hao, X. and Shafto, P. Coupled variational autoencoder. In Krause, A., Brunskill, E., Cho, K., Engelhardt, B., Sabato, S., and Scarlett, J. (eds.), Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pp. 12546–12555. PMLR, 23–29 Jul 2023.   
Harries, M. Splice-2 comparative evaluation: Electricity pricing. Technical report, The University of New South Wales, 1999.   
He, K., Zhang, X., Ren, S., and Sun, J. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pp. 770–778. IEEE, June 2016. doi: 10.1109/cvpr.2016.90.   
Hoffman, M. D. and Johnson, M. J. Elbo surgery: yet another way to carve up the variational evidence lower bound. In Advances in Approximate Bayesian Inference, 2016. NIPS 2016 Workshop.   
Huang, Y. Vjepa: Variational joint embedding predictive architectures as probabilistic world models, 2026.   
Kingma, D. P. and Welling, M. Auto-encoding variational bayes, 2013.   
Kohavi, R. Scaling up the accuracy of naive-bayes classifiers: A decision-tree hybrid. AAAI Press, Menlo Park, CA (United States), 12 1996.   
LeCun, Y. A path towards autonomous machine intelligence, 2022. White paper.   
LeCun, Y., Bottou, L., Bengio, Y., and Haffner, P. Gradientbased learning applied to document recognition. Proceedings of the IEEE, 86(11):2278–2324, 1998. ISSN 0018-9219. doi: 10.1109/5.726791.   
Moro, S., Rita, P., and Cortez, P. Bank Marketing. UCI Machine Learning Repository, 2014. doi: 10.24432/C5K306.   
Murphy, K. P. Probabilistic Machine Learning: An introduction. MIT Press, 2022.   
Parmar, N., Vaswani, A., Uszkoreit, J., Kaiser, L., Shazeer, N., Ku, A., and Tran, D. Image Transformer. In Proceedings of the 35th International Conference on Machine Learning, pp. 4052–4061. PMLR, 2018.

Song, W., Shi, C., Xiao, Z., Duan, Z., Xu, Y., Zhang, M., and Tang, J. Autoint: Automatic feature interaction learning via self-attentive neural networks. In Proceedings of the 28th ACM International Conference on Information and Knowledge Management, CIKM ’19, pp. 1161–1170. ACM, November 2019. doi: 10.1145/3357384.3357925.   
Thimonier, H., Costa, J. L. D. M., Popineau, F., Rimmel, A., and DOAN, B.-L. T-JEPA: Augmentation-free selfsupervised learning for tabular data. In The Thirteenth International Conference on Learning Representations, 2025.   
van den Oord, A., Kalchbrenner, N., Espeholt, L., kavukcuoglu, k., Vinyals, O., and Graves, A. Conditional image generation with pixelcnn decoders. In Lee, D., Sugiyama, M., Luxburg, U., Guyon, I., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 29. Curran Associates, Inc., 2016a.   
van den Oord, A., Kalchbrenner, N., and Kavukcuoglu, K. Pixel recurrent neural networks. In Balcan, M. F. and Weinberger, K. Q. (eds.), Proceedings of The 33rd International Conference on Machine Learning, volume 48 of Proceedings of Machine Learning Research, pp. 1747– 1756, New York, New York, USA, 20–22 Jun 2016b. PMLR.   
Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, L. u., and Polosukhin, I. Attention is all you need. In Guyon, I., Luxburg, U. V., Bengio, S., Wallach, H., Fergus, R., Vishwanathan, S., and Garnett, R. (eds.), Advances in Neural Information Processing Systems, volume 30. Curran Associates, Inc., 2017.   
Wang, R., Shivanna, R., Cheng, D., Jain, S., Lin, D., Hong, L., and Chi, E. Dcn v2: Improved deep & cross network and practical lessons for web-scale learning to rank systems. In Proceedings of the Web Conference 2021, WWW ’21, pp. 1785–1797. ACM, April 2021. doi: 10.1145/3442381.3450078.   
Yeh, I.-C. Default of Credit Card Clients. UCI Machine Learning Repository, 2009. doi: 10.24432/C55S3H.

# A. Var-T-JEPA: Variational Joint Embedding Predictive Architecture for Tabular Data

# A.1. Overview: Neural Architecture and Modeling Strategies

The following provides a full conceptual overview of Var-T-JEPA. Detailed implementation specifications are deferred to Appendix D.2.

Tabular data differs fundamentally from image or text modalities often encountered in JEPA applications in several aspects: (1) features are inherently heterogeneous, mixing numerical and categorical variables with different scales and distributions; (2) the notion of spatial or temporal locality is absent, requiring alternative masking strategies; (3) feature interactions are often complex and non-obvious, making the predictive task particularly suitable for representation learning approaches like JEPA.

Our tabular Var-T-JEPA pipeline broadly builds on modeling strategies established by Thimonier et al. (2025), who introduced T-JEPA as a standard (non-variational) JEPA implementation for tabular data. The model architecture of our tabular Var-T-JEPA model is depicted in Fig. 4, which translates the theoretical framework from Section 3 into a practical neural architecture tailored for tabular data.

![](images/7599f223156efcd706e7aef723cbe173a21103e45a174517a9698947a3050b91.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Original Sample"] --> B["Context Mask"]
    B --> C["Context"]
    C --> D["Context Encoder"]
    D --> E["Target Mask"]
    E --> F["Target Posterior"]
    F --> G["Target Rep."]
    G --> H["Predictor"]
    H --> I["Σθk = 1/K Mtrg * Σθk - Σθmtrg KL(qφ(Sy_k^j)|x_z,w;m_trg^k)"]
    I --> J["Target Decoder"]
    J --> K["Downstream Model"]
    K --> L["Testing"]
    L --> M["Target Mask"]
    M --> N["Target Rep."]
    N --> O["Σθk = μθ^z + Σθw ⊕ εθ^z"]
    O --> P["Target Decoder"]
    P --> Q["Downstream Model"]
    Q --> R["Testing"]
    R --> S["Target Mask"]
    S --> T["Target Rep."]
    T --> U["Σθk = μθ^z + Σθw ⊕ εθ^z"]
    U --> V["Target Decoder"]
    V --> W["Downstream Model"]
    W --> X["Testing"]
    X --> Y["Target Mask"]
    Y --> Z["Target Rep."]
    Z --> AA["Σθk = μθ^z + Σθw ⊕ εθ^z"]
    AA --> AB["Target Decoder"]
    AB --> AC["Downstream Model"]
    AC --> AD["Testing"]
    AD --> AE["Target Mask"]
    AE --> AF["Target Rep."]
    AF --> AG["Σθk = μθ^z + Σθw ⊕ εθ^z"]
    AG --> AH["Target Decoder"]
    AH --> AI["Downstream Model"]
    AI --> AJ["Testing"]
    AJ --> AK["Target Mask"]
    AK --> AL["Target Rep."]
    AL --> AM["Σθk = μθ^z + Σθw ⊕ εθ^z"]
    AM --> AN["Target Decoder"]
    AN --> AO["Downstream Model"]
    AO --> AP["Testing"]
    AP --> AQ["Target Mask"]
    AQ --> AR["Target Rep."]
    AR --> AS["Σθk = μθ^z + Σθw ⊕ εθ^z"]
    AS --> AT["Target Decoder"]
    AT --> AU["Downstream Model"]
    AU --> AV["Testing"]
    AV --> AW["Target Mask"]
```
</details>

Figure 4. Architectural overview of Var-T-JEPA for tabular data. The model implements the theoretical framework through specialized components: (1) the context encoder processes masked tabular features to produce variational context representations $q _ { \phi } ( s _ { x } ^ { ( j ) } | x ) ;$ ; (2) the auxiliary encoder infers predictive latents $q _ { \phi } ( z | s _ { x } ) ; ( 3 )$ the target posterior prepends context and auxiliary latents as special tokens before feature processing $q _ { \phi } \big ( s _ { y _ { k } } ^ { ( j ) } \big | s _ { x } , z , w ; m _ { \mathrm { t r g } } ^ { ( k ) } \big )$ n(rk) ; (4) the predictor generates target representations $p _ { \theta } \big ( s _ { y _ { k } } ^ { ( j ) } \big | s _ { x } , z ; m _ { \mathrm { t r g } } ^ { ( k ) } \big )$ , and (5) the decoders reconstruct original tabular features from latent representations.

# A.2. Tabular Feature Masking Strategy

Following the T-JEPA masking framework, we partition the complete feature vector $w \in \mathbb { R } ^ { D _ { \mathrm { f e a t } } }$ into context and target subsets at the feature level; in this formulation w is the full (unmasked) target view, while $y _ { k }$ are masked target sub-views and x the context view derived from w. Given $D _ { \mathrm { f e a t } }$ features, we create context masks $m _ { \mathrm { c t x } } \in \{ 0 , 1 \} ^ { D _ { \mathrm { f e a t } } }$ and target masks $m _ { \mathrm { t r g } } ^ { ( i ) } \in \{ 0 , 1 \} ^ { D _ { \mathrm { f e a t } } }$ mtrg where:

$$
x = m _ {\mathrm{ctx}} \odot w, \quad \| m _ {\mathrm{ctx}} \| _ {0} = M _ {\mathrm{ctx}} \tag {12}
$$

$$
y _ {k} = m _ {\mathrm{trg}} ^ {(k)} \odot w, \quad \| m _ {\mathrm{trg}} ^ {(k)} \| _ {0} = M _ {\mathrm{trg}}, \quad k = 1, \dots , K \tag {13}
$$

The mask sizes are sampled uniformly for each batch-optimization step: $M _ { \mathrm { c t x } } \sim$ Uniform $[ \lfloor D _ { \mathrm { f e a t } } \cdot r _ { \mathrm { c t x } } ^ { \mathrm { m i n } } \rfloor , \lfloor D _ { \mathrm { f e a t } } \cdot r _ { \mathrm { c t x } } ^ { \mathrm { m a x } } \rfloor ]$ and $M _ { \mathrm { t r g } } \sim$ Uniform $\big [ D _ { \mathrm { f e a t } } \cdot r _ { \mathrm { t r g } } ^ { \mathrm { m i n } } \big ] , \lfloor D _ { \mathrm { f e a t } } \cdot r _ { \mathrm { t r g } } ^ { \mathrm { m a x } } \rfloor \big ]$ , where the ratio parameters define the minimum and maximum shares of features to mask. During training, we typically generate one context view and K target predictions per sample, ensuring that $M _ { \mathrm { c t x } } + M _ { \mathrm { t r g } } \leq D _ { \mathrm { f e a t } }$ to maintain non-overlapping masks.

# A.3. Variational Network Components

Our tabular Var-T-JEPA architecture consists of six key variational network components that implement the theoretical framework for heterogeneous tabular data. Each feature is indexed by $j \in \{ 1 , 2 , . . . , D _ { \mathrm { f e a t } } \}$ where $D _ { \mathrm { f e a t } }$ denotes the total number of features in the dataset. Features are categorized as either numerical $( j \in \mathbb { Z } _ { \mathrm { n u m } } )$ or categorical $( j \in \mathcal { T } _ { \mathrm { c a t } } )$ . More detailed implementation specifications are provided in Appendix D.2.

(I) Context Encoder. Following Thimonier et al. (2025), $q _ { \phi } ( s _ { x } ^ { ( j ) } | x )$ processes masked tabular features through feature tokenization, transformer encoding, and variational output heads. Each feature is embedded with type and positional information, then processed by a transformer with prepended CLS tokens, yielding contextualized representations that parameterize the context latent distribution

$$
h _ {\mathrm{ctx}} = f _ {\phi} ^ {\mathrm{ctx}} (x), \tag {14}
$$

$$
q _ {\phi} (s _ {x} ^ {(j)} | x) = \mathcal {N} (s _ {x} ^ {(j)}; \mu_ {\phi} ^ {s _ {x}} (h _ {\mathrm{ctx}}) ^ {(j)}, \Sigma_ {\phi} ^ {s _ {x}} (h _ {\mathrm{ctx}}) ^ {(j)}). \tag {15}
$$

Here, $f _ { \phi } ^ { \mathrm { c t x } } ( x )$ is the transformer-based encoder and $\mu _ { \phi } ^ { s _ { x } }$ and $\Sigma _ { \phi } ^ { s _ { x } }$ are linear projections applied to the encoder output.

(II) Auxiliary Encoder. $q _ { \phi } ( z | s _ { x } )$ infers auxiliary predictive latents from (pooled) context representations $s _ { x }$ through a compact MLP, maintaining the JEPA principle of preventing target information leakage:

$$
q _ {\phi} (z | s _ {x}) = \mathcal {N} (z; \mu_ {\phi} ^ {z} (s _ {x}), \Sigma_ {\phi} ^ {z} (s _ {x})) \tag {16}
$$

(III) Target Posterior. $q _ { \phi } \big ( s _ { y _ { k } } ^ { ( j ) } \big | s _ { x } , z , w ; m _ { \mathrm { t r g } } ^ { ( k ) } \big )$ conditions on context latents, auxiliary latents, and complete raw features using a similar encoder design as the context encoder, but extended to incorporate latent tokens to condition on $s _ { x }$ and $z .$ .

$$
h _ {\mathrm{trg}} = f _ {\phi} ^ {\mathrm{trg}} (s _ {x}, z, w), \tag {17}
$$

$$
q _ {\phi} (s _ {y _ {k}} ^ {(j)} | s _ {x}, z, w; m _ {\mathrm{trg}} ^ {(k)}) = \mathcal {N} (s _ {w} ^ {(j)}; \mu_ {\phi} ^ {s _ {w}} (h _ {\mathrm{trg}}) ^ {(j)}, \Sigma_ {\phi} ^ {s _ {w}} (h _ {\mathrm{trg}}) ^ {(j)}) \text {   for   } j \in m _ {\mathrm{trg}} ^ {(k)}. \tag {18}
$$

Here, $f _ { \phi } ^ { \mathrm { t r g } } ( s _ { x } , z , w )$ is the transformer-based encoder, and $\mu _ { \phi } ^ { s _ { w } }$ and $\Sigma _ { \phi } ^ { s _ { u } }$ are linear projections applied to the encoder output. $s _ { y _ { k } } ^ { ( j ) } = s _ { w } ^ { ( j ) } : j \in m _ { \mathrm { t r g } } ^ { ( k ) }$ are the masked target representations.

(IV) Predictive Model. $p _ { \theta } ( s _ { y _ { k } } ^ { ( j ) } | s _ { x } , z ; m _ { \mathrm { t r g } } ^ { ( k ) } )$ generates target latent distributions without access to target observations:

$$
h _ {\text { pred }} = g _ {\theta} (s _ {x}, z, m _ {\text { trg }} ^ {(k)}), \tag {19}
$$

$$
p _ {\theta} (s _ {y _ {k}} ^ {(j)} | s _ {x}, z; m _ {\mathrm{trg}} ^ {(k)}) = \mathcal {N} (s _ {y _ {k}} ^ {(j)}; \mu_ {\theta} ^ {s _ {y}} (h _ {\mathrm{pred}}) ^ {(j)}, \Sigma_ {\theta} ^ {s _ {y}} (h _ {\mathrm{pred}}) ^ {(j)}). \tag {20}
$$

gθ(sx, z, m(k)trg ) Here, positi $g _ { \theta } ( s _ { x } , z , m _ { \mathrm { t r g } } ^ { ( k ) } )$ involves another transformer architecture which is conditioned on the target mask gs and learnable mask tokens. Its output is used to generate distributional parameter $m _ { \mathrm { t r g } } ^ { ( k ) }$ throughhe target latent $s _ { y }$ via linear projections.

(V) Context and (VI) Target Decoders. The decoders reconstruct masked features using feature-specific MLPs, $u _ { \theta } ^ { ( j ) }$ Let $\pi \in \{ x , w \}$ denote any observation (context or full target) with corresponding latent representation $s _ { \pi }$ . Each decoder reconstructs a masked feature independently according to the unified formulation:

$$
p _ {\theta} \left(\pi^ {(j)} \mid s _ {\pi} ^ {(j)}\right) = \left\{ \begin{array}{l l} \mathcal {N} \left(\pi^ {(j)}; u _ {\theta} ^ {(j)} \left(s _ {\pi} ^ {(j)}\right), \sigma_ {\pi} ^ {2}\right) & \text { if } j \in \mathcal {I} _ {\text { num }} \\ \text { Categorical } \left(\pi^ {(j)}; \operatorname{softmax} \left(u _ {\theta} ^ {(j)} \left(s _ {\pi} ^ {(j)}\right)\right)\right) & \text { if } j \in \mathcal {I} _ {\text { cat }} \end{array} \right. \tag {21}
$$

The full reconstruction distribution factorizes as $\begin{array} { r } { p _ { \theta } ( \pi | s _ { \pi } ) = \prod _ { j \in m _ { \pi } } p _ { \theta } ( \pi ^ { ( j ) } | s _ { \pi } ^ { ( j ) } ) } \end{array}$ , where $m _ { \pi }$ denotes the feature index set for observation $\pi \ ( \mathrm { i . e . , } m _ { \mathrm { c t x } }$ for context $x ,$ and all feature indices for w under full-target reconstruction). For numerical features, the reconstruction noise $\sigma _ { \pi } ^ { 2 }$ is one of two globally learned parameters: $\sigma _ { x } ^ { 2 }$ for the context path $( \pi = x )$ , and a shared $\sigma _ { y } ^ { 2 }$ for the target path $( \pi = w )$ ).

# A.3.1. LOSS COMPUTATION

Adapting Var-JEPA’s general loss function 10 to feature-level masking strategies introduced by Thimonier et al. (2025), we obtain the following sample-level loss terms for Var-T-JEPA:

$$
\mathcal {L} ^ {\text { rec }} = \frac {1}{M _ {\mathrm{ctx}}} \sum_ {j \in m _ {\mathrm{ctx}}} - \log p _ {\theta} (x ^ {(j)} | s _ {x} ^ {(j)}) \tag {22}
$$

$$
\mathcal {L} ^ {\text { gen }} = \frac {1}{D _ {\text { feat }}} \sum_ {j = 1} ^ {D _ {\text { feat }}} - \log p _ {\theta} (w ^ {(j)} \mid s _ {w} ^ {(j)}) \tag {23}
$$

$$
\mathcal {L} _ {s _ {x}} ^ {\mathrm{KL}} = \frac {1}{M _ {\mathrm{ctx}}} \sum_ {j \in m _ {\mathrm{ctx}}} \mathrm{KL} \left(q _ {\phi} \left(s _ {x} ^ {(j)} | x\right) \| \mathcal {N} (0, I)\right) \tag {24}
$$

$$
\mathcal {L} _ {z} ^ {\mathrm{KL}} = \mathrm{KL} \left(q _ {\phi} (z | s _ {x}) \| \mathcal {N} (0, I)\right) \tag {25}
$$

$$
\mathcal {L} _ {s _ {y}} ^ {\mathrm{KL}} = \frac {1}{K \cdot M _ {\mathrm{trg}}} \sum_ {k = 1} ^ {K} \sum_ {j \in m _ {\mathrm{trg}} ^ {(k)}} \mathrm{KL} \left(q _ {\phi} (s _ {y _ {k}} ^ {(j)} | s _ {x}, z, w; m _ {\mathrm{trg}} ^ {(k)}) \| p _ {\theta} (s _ {y _ {k}} ^ {(j)} | s _ {x}, z; m _ {\mathrm{trg}} ^ {(k)})\right) \tag {26}
$$

# A.4. Outlook: Extension to Vision and Video

We implemented Var-JEPA as Var-T-JEPA to provide a compute-efficient proof-of-concept in a setting where careful ablations and uncertainty analyses are tractable. However, the formulation is not specific to tabular data and can be transferred directly to visual modalities in close analogy to I-JEPA (Assran et al., 2023). Concretely, one can (i) tokenize images/videos into patch or tubelet sequences, (ii) define a masked context view x and masked target view y, (iii) parameterize $q _ { \phi } ( s _ { x } \mid x )$ and $q _ { \phi } ( s _ { y } \mid s _ { x } , z , y )$ with ViT-based encoders and $p _ { \theta } ( s _ { y } \mid s _ { x } , z )$ with a lightweight predictor (as in I-JEPA), and (iv) add a decoder $p _ { \theta } ( y \mid s _ { y } )$ (e.g., Gaussian pixels or discrete token likelihoods) to obtain the ELBO objective. This yields a generative, uncertainty-aware analogue of JEPA-style representation prediction for vision/video, while retaining the same conditional-prior interpretation of the predictor.

# B. Additional Experimental Results

Our experimental evaluation intentionally focuses on validating our conceptual contribution (bridging JEPA and variational inference) within the JEPA framework–emphasizing collapse prevention, uncertainty quantification, and the relationship between per-sample and aggregated regularization–rather than comprehensive benchmarking against the broader selfsupervised learning landscape. To complement the accuracy results reported in the main text, we provide F1-score evaluations and sensitivity analyses below.

# B.1. Additional Downstream Metrics: F1 Score

Table 3 reports downstream macro F1-scores (mean ± std over 5 downstream model runs) for the same set of tabular datasets and predictor families as Table 2. As in the main table, we report results on raw features, on T-JEPA embeddings, and on Var-T-JEPA embeddings (including selective evaluation where the most uncertain fraction of test samples is discarded).

# B.2. Sensitivity Analysis

# B.2.1. SENSITIVITY ANALYSIS ON α

We perform a sensitivity study on Adult by varying the end weights of the KL terms for $s _ { x }$ and $s _ { y } \ ( \mathrm { i . e . , } \alpha _ { s _ { x } } ^ { \mathrm { K L } }$ and $\alpha _ { s _ { y } } ^ { \mathrm { K L } } )$ and the reconstruction/generation weights $( \mathrm { i } . \mathrm { e } . , \alpha ^ { \mathrm { r e c } }$ and $\alpha ^ { \mathrm { g e n } } )$ around the baseline setting. Table 4 shows that downstream performance is broadly robust across these ablations (i.e., not strongly sensitive to precise tuning), and that selective evaluation consistently improves accuracy across all settings as more uncertain samples are filtered out.

# B.2.2. SENSITIVITY ANALYSIS ON OTHER MODEL PARAMETERS

We additionally study sensitivity to model capacity and optimization hyperparameters by varying the transformer hidden dimension d (token/embedding width), number of layers L, predictor feedforward dimension $\mathrm { f } _ { p }$ (i.e., the MLP width in the predictor block), learning rate η, and batch size B. Table 5 shows that downstream accuracy remains broadly stable for most settings, while some changes (e.g., substantially deeper encoders or smaller batch size) can reduce performance. As in the α sensitivity study, selective evaluation improves accuracy consistently across all ablations as more uncertain samples are filtered out.

Table 3. Downstream performance comparison across tabular datasets using macro F1-score. Results (mean ± std over 5 downstream model runs; XGBoost single deterministic run) show test macro F1-score. AD=Adult, CO=Covertype, EL=Electricity, CC=Credit Card, BM=Bank Marketing. Light blue rows indicate selective evaluation (reduced coverage). 

<table><tr><td>Method</td><td>AD ↑</td><td>CO ↑</td><td>EL ↑</td><td>CC ↑</td><td>BM ↑</td><td>MNIST ↑</td><td>SIM ↑</td></tr><tr><td>MLP</td><td> $0.845±3.7e^{-3}$ </td><td> $0.747±5.0e^{-3}$ </td><td> $0.780±3.5e^{-3}$ </td><td> $0.792±6.9e^{-3}$ </td><td> $0.887±4.8e^{-3}$ </td><td> $0.822±1.1e^{-2}$ </td><td> $0.821±9.0e^{-3}$ </td></tr><tr><td>+T-JEPA</td><td> $0.842±9.0e^{-3}$ </td><td> $0.274±1.5e^{-1}$ </td><td> $0.545±7.4e^{-2}$ </td><td> $0.675±0e^{-0}$ </td><td> $0.583±3.9e^{-1}$ </td><td> $0.023±2.3e^{-3}$ </td><td> $0.662±3.3e^{-2}$ </td></tr><tr><td>+Var-T-JEPA</td><td> $0.845±3.3e^{-3}$ </td><td> $0.657±3.8e^{-2}$ </td><td> $0.789±2.2e^{-3}$ </td><td> $0.797±4.1e^{-3}$ </td><td> $0.889±3.1e^{-3}$ </td><td> $0.823±1.7e^{-2}$ </td><td> $0.659±3.3e^{-2}$ </td></tr><tr><td>+Var-T-JEPA (10%)</td><td> $0.859±3.0e^{-3}$ </td><td> $0.678±3.4e^{-2}$ </td><td> $0.792±2.0e^{-3}$ </td><td> $0.792±5.6e^{-3}$ </td><td> $0.893±3.5e^{-3}$ </td><td> $0.840±1.5e^{-2}$ </td><td> $0.671±3.1e^{-2}$ </td></tr><tr><td>+Var-T-JEPA (20%)</td><td> $0.877±2.4e^{-3}$ </td><td> $0.680±3.0e^{-2}$ </td><td> $0.794±2.3e^{-3}$ </td><td> $0.792±4.3e^{-3}$ </td><td> $0.895±4.0e^{-3}$ </td><td> $0.857±1.8e^{-2}$ </td><td> $0.671±3.2e^{-2}$ </td></tr><tr><td>+Var-T-JEPA (50%)</td><td> $0.915±2.6e^{-3}$ </td><td> $0.692±1.7e^{-2}$ </td><td> $0.803±3.4e^{-3}$ </td><td> $0.780±1.9e^{-3}$ </td><td> $0.907±3.9e^{-3}$ </td><td> $0.901±2.0e^{-2}$ </td><td> $0.691±3.7e^{-2}$ </td></tr><tr><td>DCNv2</td><td> $0.684±4.5e^{-3}$ </td><td> $0.749±6.2e^{-3}$ </td><td> $0.824±7.3e^{-4}$ </td><td> $0.794±3.0e^{-3}$ </td><td> $0.890±2.1e^{-3}$ </td><td> $0.875±4.3e^{-3}$ </td><td> $0.673±5.0e^{-3}$ </td></tr><tr><td>+T-JEPA</td><td> $0.846±1.7e^{-3}$ </td><td> $0.443±5.3e^{-2}$ </td><td> $0.764±1.9e^{-3}$ </td><td> $0.737±5.7e^{-2}$ </td><td> $0.878±1.4e^{-3}$ </td><td> $0.840±4.7e^{-2}$ </td><td> $0.759±4.2e^{-3}$ </td></tr><tr><td>+Var-T-JEPA</td><td> $0.848±1.5e^{-3}$ </td><td> $0.775±6.9e^{-3}$ </td><td> $0.830±1.9e^{-3}$ </td><td> $0.794±1.3e^{-3}$ </td><td> $0.891±5.9e^{-4}$ </td><td> $0.884±4.8e^{-3}$ </td><td> $0.703±8.8e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (10%)</td><td> $0.861±1.2e^{-3}$ </td><td> $0.783±7.0e^{-3}$ </td><td> $0.831±2.6e^{-3}$ </td><td> $0.790±1.4e^{-3}$ </td><td> $0.896±6.6e^{-4}$ </td><td> $0.897±5.2e^{-3}$ </td><td> $0.711±8.3e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (20%)</td><td> $0.878±1.3e^{-3}$ </td><td> $0.780±7.1e^{-3}$ </td><td> $0.831±2.1e^{-3}$ </td><td> $0.791±1.9e^{-3}$ </td><td> $0.900±1.1e^{-3}$ </td><td> $0.908±5.0e^{-3}$ </td><td> $0.707±8.6e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (50%)</td><td> $0.916±7.9e^{-4}$ </td><td> $0.778±5.7e^{-3}$ </td><td> $0.831±2.6e^{-3}$ </td><td> $0.780±1.6e^{-3}$ </td><td> $0.909±1.8e^{-3}$ </td><td> $0.936±3.0e^{-3}$ </td><td> $0.735±6.1e^{-3}$ </td></tr><tr><td>ResNet</td><td> $0.844±1.8e^{-3}$ </td><td> $0.776±7.3e^{-3}$ </td><td> $0.823±2.6e^{-3}$ </td><td> $0.794±1.5e^{-3}$ </td><td> $0.890±4.0e^{-3}$ </td><td> $0.860±7.1e^{-3}$ </td><td> $0.877±5.9e^{-3}$ </td></tr><tr><td>+T-JEPA</td><td> $0.845±4.8e^{-3}$ </td><td> $0.499±8.7e^{-2}$ </td><td> $0.807±8.8e^{-3}$ </td><td> $0.794±7.9e^{-3}$ </td><td> $0.893±3.8e^{-3}$ </td><td> $0.856±2.9e^{-2}$ </td><td> $0.689±5.8e^{-2}$ </td></tr><tr><td>+Var-T-JEPA</td><td> $0.849±2.4e^{-3}$ </td><td> $0.778±1.2e^{-2}$ </td><td> $0.820±7.2e^{-3}$ </td><td> $0.791±1.0e^{-2}$ </td><td> $0.894±2.6e^{-3}$ </td><td> $0.872±1.2e^{-2}$ </td><td> $0.549±1.2e^{-1}$ </td></tr><tr><td>+Var-T-JEPA (10%)</td><td> $0.864±1.6e^{-3}$ </td><td> $0.784±1.2e^{-2}$ </td><td> $0.822±6.8e^{-3}$ </td><td> $0.789±7.0e^{-3}$ </td><td> $0.899±2.0e^{-3}$ </td><td> $0.888±9.2e^{-3}$ </td><td> $0.560±1.1e^{-1}$ </td></tr><tr><td>+Var-T-JEPA (20%)</td><td> $0.881±1.2e^{-3}$ </td><td> $0.781±1.3e^{-2}$ </td><td> $0.823±6.7e^{-3}$ </td><td> $0.789±5.1e^{-3}$ </td><td> $0.903±1.9e^{-3}$ </td><td> $0.900±8.5e^{-3}$ </td><td> $0.570±1.1e^{-1}$ </td></tr><tr><td>+Var-T-JEPA (50%)</td><td> $0.920±8.5e^{-4}$ </td><td> $0.784±1.3e^{-2}$ </td><td> $0.826±7.5e^{-3}$ </td><td> $0.780±2.1e^{-3}$ </td><td> $0.912±1.0e^{-3}$ </td><td> $0.936±5.3e^{-3}$ </td><td> $0.602±1.1e^{-1}$ </td></tr><tr><td>AutoInt</td><td> $0.687±6.3e^{-3}$ </td><td> $0.752±1.7e^{-2}$ </td><td> $0.755±1.1e^{-2}$ </td><td> $0.798±1.2e^{-3}$ </td><td> $0.896±2.5e^{-3}$ </td><td> $0.809±1.6e^{-2}$ </td><td> $0.897±1.9e^{-2}$ </td></tr><tr><td>+T-JEPA</td><td> $0.849±2.9e^{-3}$ </td><td> $0.399±1.6e^{-1}$ </td><td> $0.745±3.1e^{-2}$ </td><td> $0.777±7.8e^{-4}$ </td><td> $0.871±1.2e^{-2}$ </td><td> $0.816±7.9e^{-3}$ </td><td> $0.768±8.4e^{-3}$ </td></tr><tr><td>+Var-T-JEPA</td><td> $0.848±3.1e^{-3}$ </td><td> $0.750±4.4e^{-3}$ </td><td> $0.821±2.8e^{-3}$ </td><td> $0.796±2.9e^{-3}$ </td><td> $0.894±1.3e^{-3}$ </td><td> $0.810±5.5e^{-3}$ </td><td> $0.711±9.3e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (10%)</td><td> $0.863±3.6e^{-3}$ </td><td> $0.755±4.2e^{-3}$ </td><td> $0.822±3.8e^{-3}$ </td><td> $0.791±3.1e^{-3}$ </td><td> $0.898±1.4e^{-3}$ </td><td> $0.824±2.0e^{-3}$ </td><td> $0.719±8.3e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (20%)</td><td> $0.880±2.8e^{-3}$ </td><td> $0.752±4.3e^{-3}$ </td><td> $0.822±4.1e^{-3}$ </td><td> $0.793±2.0e^{-3}$ </td><td> $0.902±1.4e^{-3}$ </td><td> $0.840±1.7e^{-3}$ </td><td> $0.720±7.9e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (50%)</td><td> $0.918±2.2e^{-3}$ </td><td> $0.745±6.3e^{-3}$ </td><td> $0.823±4.5e^{-3}$ </td><td> $0.781±1.6e^{-3}$ </td><td> $0.912±1.1e^{-3}$ </td><td> $0.879±3.8e^{-3}$ </td><td> $0.741±1.0e^{-2}$ </td></tr><tr><td>FT-Trans</td><td> $0.687±4.9e^{-3}$ </td><td> $0.741±1.0e^{-2}$ </td><td> $0.813±4.0e^{-3}$ </td><td> $0.799±1.0e^{-3}$ </td><td> $0.894±1.3e^{-3}$ </td><td> $0.877±2.3e^{-2}$ </td><td> $0.962±1.4e^{-3}$ </td></tr><tr><td>+T-JEPA</td><td> $0.847±1.1e^{-3}$ </td><td> $0.417±8.5e^{-2}$ </td><td> $0.700±4.7e^{-2}$ </td><td> $0.777±2.0e^{-3}$ </td><td> $0.851±1.0e^{-3}$ </td><td> $0.269±2.7e^{-1}$ </td><td> $0.696±2.0e^{-2}$ </td></tr><tr><td>+Var-T-JEPA</td><td> $0.847±2.2e^{-3}$ </td><td> $0.743±6.5e^{-3}$ </td><td> $0.797±4.6e^{-3}$ </td><td> $0.798±1.3e^{-3}$ </td><td> $0.893±9.2e^{-4}$ </td><td> $0.864±9.4e^{-3}$ </td><td> $0.701±8.0e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (10%)</td><td> $0.862±2.3e^{-3}$ </td><td> $0.752±5.6e^{-3}$ </td><td> $0.799±4.0e^{-3}$ </td><td> $0.793±9.1e^{-4}$ </td><td> $0.897±9.1e^{-4}$ </td><td> $0.879±9.6e^{-3}$ </td><td> $0.712±8.5e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (20%)</td><td> $0.879±2.0e^{-3}$ </td><td> $0.750±5.7e^{-3}$ </td><td> $0.801±3.8e^{-3}$ </td><td> $0.794±1.6e^{-3}$ </td><td> $0.900±1.3e^{-3}$ </td><td> $0.891±9.9e^{-3}$ </td><td> $0.712±6.8e^{-3}$ </td></tr><tr><td>+Var-T-JEPA (50%)</td><td> $0.917±2.0e^{-3}$ </td><td> $0.743±7.9e^{-3}$ </td><td> $0.809±2.2e^{-3}$ </td><td> $0.782±1.9e^{-3}$ </td><td> $0.912±1.3e^{-3}$ </td><td> $0.918±6.7e^{-3}$ </td><td> $0.732±3.6e^{-3}$ </td></tr><tr><td>XGBoost</td><td>0.860</td><td>0.805</td><td>0.917</td><td>0.791</td><td>0.894</td><td>0.881</td><td>0.948</td></tr><tr><td>+T-JEPA</td><td>0.850</td><td>0.804</td><td>0.860</td><td>0.783</td><td>0.889</td><td>0.870</td><td>0.849</td></tr><tr><td>+Var-T-JEPA</td><td>0.850</td><td>0.807</td><td>0.888</td><td>0.787</td><td>0.897</td><td>0.871</td><td>0.945</td></tr><tr><td>+Var-T-JEPA (10%)</td><td>0.870</td><td>0.816</td><td>0.890</td><td>0.788</td><td>0.901</td><td>0.883</td><td>0.948</td></tr><tr><td>+Var-T-JEPA (20%)</td><td>0.888</td><td>0.815</td><td>0.891</td><td>0.788</td><td>0.903</td><td>0.889</td><td>0.951</td></tr><tr><td>+Var-T-JEPA (50%)</td><td>0.923</td><td>0.813</td><td>0.891</td><td>0.784</td><td>0.911</td><td>0.913</td><td>0.955</td></tr></table>

Table 4. Adult sensitivity analysis for Var-T-JEPA end-weight settings. Columns correspond to ablations varying $( \alpha _ { s _ { x } } ^ { \mathrm { K L } } , \alpha _ { s _ { y } } ^ { \mathrm { K L } } , \alpha ^ { \mathrm { r e c } } , \alpha ^ { \mathrm { g e n } } )$ ; we report downstream test accuracy (mean ± std over 5 downstream runs) for MLP and ResNet probes, including selective evaluation. 

<table><tr><td></td><td>Base</td><td>Low KL</td><td>High KL</td><td>Low recon.</td><td>High recon.</td><td>High KL + low recon.</td><td>Low KL + high recon.</td></tr><tr><td> $\alpha_{sx}^{KL}$ </td><td> $10^{-4}$ </td><td> $10^{-5}$ </td><td> $10^{-3}$ </td><td> $10^{-4}$ </td><td> $10^{-4}$ </td><td> $10^{-3}$ </td><td> $10^{-5}$ </td></tr><tr><td> $\alpha_{sy}^{KL}$ </td><td> $10^{-5}$ </td><td> $10^{-6}$ </td><td> $10^{-4}$ </td><td> $10^{-5}$ </td><td> $10^{-5}$ </td><td> $10^{-4}$ </td><td> $10^{-6}$ </td></tr><tr><td> $\alpha_{rec}^{sc}$ </td><td> $10^{-1}$ </td><td> $10^{-1}$ </td><td> $10^{-1}$ </td><td> $10^{-2}$ </td><td>1</td><td> $10^{-2}$ </td><td>1</td></tr><tr><td> $\alpha_{gen}^{scn}$ </td><td>1</td><td>1</td><td>1</td><td> $10^{-1}$ </td><td>10</td><td> $10^{-1}$ </td><td>10</td></tr><tr><td>MLP + Var-T-JEPA</td><td> $0.852±2.1e^{-3}$ </td><td> $0.850±1.2e^{-3}$ </td><td> $0.826±2.7e^{-3}$ </td><td> $0.834±2.7e^{-3}$ </td><td> $0.852±2.3e^{-3}$ </td><td> $0.849±1.1e^{-3}$ </td><td> $0.851±1.2e^{-3}$ </td></tr><tr><td>MLP + Var-T-JEPA (10%)</td><td> $0.865±2.0e^{-3}$ </td><td> $0.864±9.6e^{-4}$ </td><td> $0.839±2.6e^{-3}$ </td><td> $0.850±2.0e^{-3}$ </td><td> $0.868±1.7e^{-3}$ </td><td> $0.863±1.3e^{-3}$ </td><td> $0.860±1.1e^{-3}$ </td></tr><tr><td>MLP + Var-T-JEPA (20%)</td><td> $0.883±1.7e^{-3}$ </td><td> $0.880±1.4e^{-3}$ </td><td> $0.852±2.4e^{-3}$ </td><td> $0.868±2.2e^{-3}$ </td><td> $0.879±1.9e^{-3}$ </td><td> $0.873±1.2e^{-3}$ </td><td> $0.878±8.5e^{-4}$ </td></tr><tr><td>MLP + Var-T-JEPA (50%)</td><td> $0.921±2.1e^{-3}$ </td><td> $0.920±8.7e^{-4}$ </td><td> $0.890±2.8e^{-3}$ </td><td> $0.888±3.0e^{-3}$ </td><td> $0.906±2.5e^{-3}$ </td><td> $0.888±1.4e^{-3}$ </td><td> $0.928±7.9e^{-4}$ </td></tr><tr><td>ResNet + Var-T-JEPA</td><td> $0.854±1.4e^{-3}$ </td><td> $0.852±1.8e^{-3}$ </td><td> $0.827±8.5e^{-4}$ </td><td> $0.836±3.1e^{-3}$ </td><td> $0.852±1.1e^{-3}$ </td><td> $0.853±1.6e^{-3}$ </td><td> $0.853±1.4e^{-3}$ </td></tr><tr><td>ResNet + Var-T-JEPA (10%)</td><td> $0.868±9.2e^{-4}$ </td><td> $0.867±1.4e^{-3}$ </td><td> $0.839±1.3e^{-3}$ </td><td> $0.851±3.1e^{-3}$ </td><td> $0.868±1.1e^{-3}$ </td><td> $0.867±1.8e^{-3}$ </td><td> $0.863±1.6e^{-3}$ </td></tr><tr><td>ResNet + Var-T-JEPA (20%)</td><td> $0.886±8.2e^{-4}$ </td><td> $0.883±1.9e^{-3}$ </td><td> $0.852±1.5e^{-3}$ </td><td> $0.869±3.2e^{-3}$ </td><td> $0.878±1.3e^{-3}$ </td><td> $0.876±1.1e^{-3}$ </td><td> $0.881±1.1e^{-3}$ </td></tr><tr><td>ResNet + Var-T-JEPA (50%)</td><td> $0.924±6.6e^{-4}$ </td><td> $0.921±8.6e^{-4}$ </td><td> $0.890±2.5e^{-3}$ </td><td> $0.888±4.3e^{-3}$ </td><td> $0.906±1.3e^{-3}$ </td><td> $0.891±1.4e^{-3}$ </td><td> $0.929±1.4e^{-3}$ </td></tr></table>

Table 5. Adult sensitivity analysis for Var-T-JEPA architecture and optimization settings. Columns correspond to model/optimizer ablations varying $( d , L , \dot { \mathrm { f f } } _ { p } , \eta , \dot { B } )$ . We report downstream test accuracy (mean ± std over 5 downstream runs) for MLP and ResNet probes, including selective evaluation (light blue rows). 

<table><tr><td></td><td>Shallow</td><td>Low pred.-FF</td><td>Narrow</td><td>High pred.-FF</td><td>Deep</td><td>Low LR</td><td>Batch size 256</td></tr><tr><td>d (hidden dim.)</td><td>64</td><td>64</td><td>32</td><td>64</td><td>64</td><td>64</td><td>64</td></tr><tr><td>L (layers)</td><td>4</td><td>8</td><td>8</td><td>8</td><td>16</td><td>8</td><td>8</td></tr><tr><td> $\text{ff}_p$ (pred.)</td><td>256</td><td>128</td><td>128</td><td>512</td><td>256</td><td>256</td><td>256</td></tr><tr><td>η (LR)</td><td> $10^{-3}$ </td><td> $10^{-3}$ </td><td> $10^{-3}$ </td><td> $10^{-3}$ </td><td> $10^{-3}$ </td><td> $5 \times 10^{-4}$ </td><td> $10^{-3}$ </td></tr><tr><td>B (batch)</td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td><td>512</td><td>256</td></tr><tr><td>MLP + Var-T-JEPA</td><td> $0.850 \pm 2.1e^{-3}$ </td><td> $0.852 \pm 2.1e^{-3}$ </td><td> $0.850 \pm 1.6e^{-3}$ </td><td> $0.852 \pm 2.1e^{-3}$ </td><td> $0.832 \pm 1.7e^{-3}$ </td><td> $0.849 \pm 3.2e^{-3}$ </td><td> $0.851 \pm 2.7e^{-3}$ </td></tr><tr><td>MLP + Var-T-JEPA (10%)</td><td> $0.864 \pm 1.1e^{-3}$ </td><td> $0.865 \pm 2.0e^{-3}$ </td><td> $0.867 \pm 1.6e^{-3}$ </td><td> $0.865 \pm 2.0e^{-3}$ </td><td> $0.843 \pm 1.7e^{-3}$ </td><td> $0.866 \pm 3.6e^{-3}$ </td><td> $0.854 \pm 2.9e^{-3}$ </td></tr><tr><td>MLP + Var-T-JEPA (20%)</td><td> $0.877 \pm 8.3e^{-4}$ </td><td> $0.883 \pm 1.7e^{-3}$ </td><td> $0.883 \pm 1.4e^{-3}$ </td><td> $0.883 \pm 1.7e^{-3}$ </td><td> $0.864 \pm 1.5e^{-3}$ </td><td> $0.881 \pm 3.7e^{-3}$ </td><td> $0.856 \pm 2.9e^{-3}$ </td></tr><tr><td>MLP + Var-T-JEPA (50%)</td><td> $0.905 \pm 1.1e^{-3}$ </td><td> $0.921 \pm 2.1e^{-3}$ </td><td> $0.902 \pm 9.6e^{-4}$ </td><td> $0.921 \pm 2.1e^{-3}$ </td><td> $0.908 \pm 1.5e^{-3}$ </td><td> $0.907 \pm 3.7e^{-3}$ </td><td> $0.871 \pm 3.0e^{-3}$ </td></tr><tr><td>ResNet + Var-T-JEPA</td><td> $0.852 \pm 1.2e^{-3}$ </td><td> $0.854 \pm 1.4e^{-3}$ </td><td> $0.851 \pm 6.5e^{-4}$ </td><td> $0.854 \pm 1.4e^{-3}$ </td><td> $0.833 \pm 1.5e^{-3}$ </td><td> $0.853 \pm 8.7e^{-4}$ </td><td> $0.852 \pm 2.0e^{-3}$ </td></tr><tr><td>ResNet + Var-T-JEPA (10%)</td><td> $0.864 \pm 1.1e^{-3}$ </td><td> $0.868 \pm 9.2e^{-4}$ </td><td> $0.870 \pm 9.4e^{-4}$ </td><td> $0.868 \pm 9.2e^{-4}$ </td><td> $0.845 \pm 1.1e^{-3}$ </td><td> $0.869 \pm 1.1e^{-3}$ </td><td> $0.856 \pm 1.9e^{-3}$ </td></tr><tr><td>ResNet + Var-T-JEPA (20%)</td><td> $0.877 \pm 1.4e^{-3}$ </td><td> $0.886 \pm 8.2e^{-4}$ </td><td> $0.885 \pm 9.9e^{-4}$ </td><td> $0.886 \pm 8.2e^{-4}$ </td><td> $0.865 \pm 1.2e^{-3}$ </td><td> $0.884 \pm 2.0e^{-3}$ </td><td> $0.858 \pm 1.6e^{-3}$ </td></tr><tr><td>ResNet + Var-T-JEPA (50%)</td><td> $0.904 \pm 1.6e^{-3}$ </td><td> $0.924 \pm 6.6e^{-4}$ </td><td> $0.900 \pm 1.7e^{-3}$ </td><td> $0.924 \pm 6.6e^{-4}$ </td><td> $0.909 \pm 1.9e^{-3}$ </td><td> $0.907 \pm 2.4e^{-3}$ </td><td> $0.873 \pm 1.5e^{-3}$ </td></tr></table>

# C. Dataset Descriptions

# C.1. Dataset Descriptions for Downstream Experiments with Var-T-JEPA

# C.1.1. DATASET SUMMARY

Our experimental evaluation of Var-T-JEPA uses five real-world tabular datasets (Adult, Covertype, Electricity, Credit Card Default, Bank Marketing), a vision dataset (MNIST), where we treat every pixel as an individual feature column, and a fully synthetic dataset (SIM). The latter two datasets were adapted to exhibit controllable uncertainty (noise) in the original feature space. Table 6 summarizes their basic characteristics. For Covertype, we randomly subsample 10,000 examples for our experiments. For supervised downstream model training and evaluation we create train/validation/test splits (70/10/20) from the embeddings learned on the full feature dataset. For baseline models trained on raw features, we apply the same splits to the original datasets.

Table 6. Dataset characteristics used in experimental evaluation. 

<table><tr><td>Dataset</td><td>Samples</td><td>Features</td><td>(num/cat)</td><td>Classes</td><td>Task</td><td>Domain</td></tr><tr><td>Adult (AD) (Kohavi, 1996)</td><td>48,842</td><td>14</td><td>(6/8)</td><td>2</td><td>Classification</td><td>Census</td></tr><tr><td>Covertype (CO) (Blackard, 1998)</td><td>10,000</td><td>54</td><td>(54/0)</td><td>7</td><td>Classification</td><td>Forestry</td></tr><tr><td>Electricity (EL) (Harries, 1999)</td><td>45,312</td><td>8</td><td>(8/0)</td><td>2</td><td>Classification</td><td>Time series</td></tr><tr><td>Credit Card Default (CC) (Yeh, 2009)</td><td>30,000</td><td>23</td><td>(23/0)</td><td>2</td><td>Classification</td><td>Finance</td></tr><tr><td>Bank Marketing (BM) (Moro et al., 2014)</td><td>45,211</td><td>16</td><td>(16/0)</td><td>2</td><td>Classification</td><td>Marketing</td></tr><tr><td>MNIST (LeCun et al., 1998)</td><td>10,000</td><td>784</td><td>(784/0)</td><td>10</td><td>Classification</td><td>Vision</td></tr><tr><td>Simulated (SIM)</td><td>10,000</td><td>32</td><td>(28/4)</td><td>3</td><td>Classification</td><td>Synthetic</td></tr></table>

# C.1.2. (SEMI-)SYNTHETIC DATA GENERATION

Fully synthetic dataset (SIM). Our simulated dataset is designed to produce well-separated class structure for most samples, while inducing controlled uncertainty in the original feature space for a subset of samples through prototype mixing. Concretely, we generate three class-conditional Gaussian clusters in a 28-dimensional numeric space with shared isotropic variance. We then draw a per-sample ambiguity score $( u _ { i } \sim \mathrm { U n i f o r m } ( 0 , 1 ) ) ,$ and map it to an amplified uncertainty/ambiguity strength $( u _ { i } ^ { \mathrm { ( a m b ) } } = u _ { i } ^ { \gamma } )$ with $( \gamma = 2 )$ . In doing so, most samples are easy and only a fraction become highly ambiguous. For each sample (i) with true class (c), we select an alternative class $\left( c ^ { \prime } \right)$ and blend the numeric features toward the alternative class prototype with strength $( \alpha _ { i } \propto u _ { i } ^ { ( \mathrm { a m b } ) } )$ . We further (i) mix a random subset of numeric features toward the alternative prototype, and (ii) inject additional ambiguity-driven noise in a small set of informative dimensions. Finally, we derive four in categorical variables by quantile-binning selected numeric dimensions and flip categorical values with probability increasing $u _ { i } ^ { \mathrm { ( a m b ) } }$ . The output includes 28 numeric features, 4 categorical features, and 3 classes.

MNIST. We represent MNIST as a tabular dataset by flattening each $2 8 \times 2 8$ grayscale image to a 784-dimensional feature vector and using the digit label as class to be predicted. To create a controllable uncertainty signal, we incorporate a per-sample score $( u _ { i } \sim \mathrm { U n i f o r m } ( 0 , 1 ) )$ and corrupt the input by interpolating between the original image $( x _ { i } )$ and an independent random image $( r _ { i } \sim \mathrm { U n i f o r m } ( 0 , 1 ) ^ { 7 8 4 } )$ :

$$
\alpha_ {i} = \operatorname{clip} (\lambda u _ {i}, 0, 1), \quad x _ {i} ^ {\prime} = (1 - \alpha_ {i}) x _ {i} + \alpha_ {i} r _ {i},
$$

with a fixed noise scale $( \lambda = 0 . 7 5 )$ . As illustrated in Figure 5, different values of $u _ { i }$ produce varying levels of corruption, from clean images $( u _ { i } = 0 )$ to almost random noise $( u _ { i } = 0 . 9 9 )$ . We use a random subset of $1 0 { , } 0 0 0$ samples for the MNIST experiments.

![](images/df4ef79a4b6ef49cefba3a2dc116766580da5194be42d2a6f9bbfbc6ac1d0196.jpg)

<details>
<summary>text_image</summary>

u = 0.00
u = 0.25
u = 0.50
u = 0.75
u = 0.99
2 2 2 2
</details>

Figure 5. Visualization of MNIST image corruption under different values of the uncertainty score $u _ { i } .$ .

# C.2. Data Generation for Simulation Study

We generate paired observations $( x _ { i } , y _ { i } ) \in \mathbb { R } ^ { D _ { \mathrm { o b s } } } \times \mathbb { R } ^ { D _ { \mathrm { o b s } } }$ from latent variables $s _ { x _ { i } } , s _ { y _ { i } } \in \mathbb { R } ^ { d _ { s } }$ and $z _ { i } \in \mathbb { R } ^ { d _ { z } }$ . In all simulation experiments we use $D _ { \mathrm { o b s } } = 3 2 , d _ { s } = 1 6 , d _ { z } = 8$ , diagonal covariances $\Sigma _ { x } = \overset {  } { \sigma _ { x } ^ { 2 } } I _ { d _ { s } } , \Sigma _ { y } = \sigma _ { y } ^ { 2 } I _ { d _ { s } } , \Sigma _ { z } = \sigma _ { z } ^ { 2 } I _ { d _ { \iota } }$ z with $( \sigma _ { x } , \sigma _ { y } , \sigma _ { z } ) = ( 1 . 0 , 0 . 5 , 1 . 0 )$ , a mean-shift $\delta = \delta _ { \mathrm { s c a l e } } \mathbf { 1 }$ with $\delta _ { \mathrm { s c a l e } } = 2 . 0$ , and observation noise $( \tau _ { x } , \tau _ { y } ) = ( 0 . 3 , 0 . 3 )$ .

$$
z _ {i} \sim \mathcal {N} (0, \Sigma_ {z}), \qquad s _ {x _ {i}} \sim \frac {1}{2} \mathcal {N} (0, \Sigma_ {x}) + \frac {1}{2} \mathcal {N} (\delta , \Sigma_ {x}), \qquad s _ {y _ {i}} \mid s _ {x _ {i}}, z _ {i} \sim \mathcal {N} (s _ {x _ {i}} + A z _ {i}, \Sigma_ {y}),
$$

$$
x _ {i} = h _ {x} (s _ {x _ {i}}) + \epsilon_ {x _ {i}}, \epsilon_ {x _ {i}} \sim \mathcal {N} (0, \tau_ {x} ^ {2} I _ {D _ {\mathrm{obs}}}), \qquad y _ {i} = h _ {y} (s _ {y _ {i}}) + \epsilon_ {y _ {i}}, \epsilon_ {y _ {i}} \sim \mathcal {N} (0, \tau_ {y} ^ {2} I _ {D _ {\mathrm{obs}}}).
$$

Equivalently, we introduce a binary mixture label $c _ { i } \sim$ Bernoulli(1/2) and sample $s _ { x _ { i } } \mid c _ { i } \sim \mathcal { N } ( c _ { i } \delta , \Sigma _ { x } )$ . For each run, we draw $\dot { A } \in \mathbb { R } ^ { d _ { s } \times d _ { z } }$ once with entries $A _ { j k } \sim \mathcal { N } ( 0 , 1 / d _ { z } )$ and keep it fixed. The nonlinear maps $h _ { x } , h _ { y }$ are also drawn once per run as independent two-layer MLPs with tanh activation (hidden width 64) and then frozen. Different runs use different draws of $( A , h _ { x } , h _ { y } )$ and therefore different simulated datasets.

# D. Implementation Details

# D.1. Var-JEPA Implementation for Simulation Study

The simulation study implements a minimal Var-JEPA variant to isolate the effects of per-sample KL regularization and aggregated-distribution regularization (SIGReg) under a ground-truth latent-variable data generator. We implement an MLPbased Var-JEPA with amortized Gaussian posteriors $q _ { \phi } ( s _ { x } \mid x ) , q _ { \phi } ( z \mid s _ { x } ) , q _ { \phi } ( s _ { y } \mid s _ { x } , z , y )$ and Gaussian decoders/conditional priors.

# D.2. Var-T-JEPA Implementation for Tabular Data

We provide implementation details for each module of Var-T-JEPA below, adopting standard components from T-JEPA (Thimonier et al., 2025) where appropriate.

# D.2.1. CONTEXT ENCODER IMPLEMENTATION

The context encoder $q _ { \phi } ( s _ { x } ^ { ( j ) } | x )$ processes masked tabular input through three main stages:

Feature tokenization. Each feature $x ^ { ( j ) }$ from the masked input is embedded into a $D _ { \mathrm { h i d } }$ -dimensional token representation (in Var-T-JEPA we set $d = d _ { z } = D _ { \mathrm { h i d } } )$ . For numerical features, we use a shared linear projection with feature-specific bias terms $b ^ { ( j ) } \in \mathbb { R } ^ { D _ { \mathrm { h i d } } }$ that account for different feature scales and distributions:

$$
t _ {\mathrm{ctx}} ^ {(j)} = \left\{ \begin{array}{l l} W _ {\text { num }} \cdot x ^ {(j)} + b ^ {(j)} + e _ {\text { type }} ^ {\text { num }} + e _ {\text { pos }} ^ {(j)} & \text { if   } j \in \mathcal {I} _ {\text { num }} \\ \operatorname{Embed} _ {L _ {j}} (x ^ {(j)}) + e _ {\text { type }} ^ {\text { cat }} + e _ {\text { pos }} ^ {(j)} & \text { if   } j \in \mathcal {I} _ {\text { cat }} \end{array} \right. \tag {27}
$$

where $W _ { \mathrm { n u m } } \in \mathbb { R } ^ { D _ { \mathrm { h i d } } \times 1 }$ is a shared linear projection for numerical features, Embed $\mathbf { \Phi } _ { L _ { j } } : \{ 0 , 1 , \dots , L _ { j } - 1 \} \to \mathbb { R } ^ { D _ { \mathrm { h i d } } }$ are learned embedding tables for categorical features with $L _ { j }$ categories, $e _ { \mathrm { t y p e } }$ denotes feature type embeddings distinguishing numerical from categorical features, and $e _ { \mathrm { p o s } } ^ { ( j ) }$ denotes positional embeddings encoding feature position.

Transformer processing. The tokenized features are processed by a standard transformer encoder architecture (Vaswani et al., 2017) with multi-head self-attention and feed-forward layers. Following T-JEPA conventions, we prepend $N _ { \mathrm { C L S } }$ learnable classification tokens to the sequence, yielding contextualized representations:

$$
h _ {\mathrm{ctx}} = \text { Transformer } ([ \{t _ {\mathrm{CLS}} ^ {(i)} \} _ {i = 1} ^ {N _ {\mathrm{CLS}}}, \{t _ {\mathrm{ctx}} ^ {(j)} \} _ {j \in m _ {\mathrm{ctx}}} ]) \in \mathbb {R} ^ {(N _ {\mathrm{CLS}} + M _ {\mathrm{ctx}}) \times D _ {\mathrm{hid}}} \tag {28}
$$

Variational output. The context encoder outputs distributional parameters for the variational posterior:

$$
\mu_ {\phi} ^ {s _ {x}} (h _ {\mathrm{ctx}}) = \text { Linear } (h _ {\mathrm{ctx}}) \tag {29}
$$

$$
\log \Sigma_ {\phi} ^ {s _ {x}} (h _ {\mathrm{ctx}}) = \text { Linear } (h _ {\mathrm{ctx}}) \tag {30}
$$

$$
q _ {\phi} (s _ {x} ^ {(j)} | x) = \mathcal {N} (s _ {x} ^ {(j)}; \mu_ {\phi} ^ {s _ {x}} (h _ {\mathrm{ctx}}) ^ {(j)}, \Sigma_ {\phi} ^ {s _ {x}} (h _ {\mathrm{ctx}}) ^ {(j)}) \tag {31}
$$

# D.2.2. AUXILIARY ENCODER IMPLEMENTATION

$s _ { x }$ oth the auxiliary encoder a to a fixed set of pooled c e targett tokens $\overline { { s } } _ { x } = \{ \bar { s } _ { x } ^ { ( i ) } \} _ { i = 1 } ^ { L _ { \mathrm { p o o l } } }$ ention-based pooling to map the variable-length context sequence. This mechanism is order-invariant and yields a fixed number of pooled tokens regardless of $M _ { \mathrm { c t x } }$ .

The auxiliary encoder $q _ { \phi } ( z | s _ { x } )$ takes the pooled context representation $\bar { s } _ { x }$ and processes it through a compact MLP. Consistent with the JEPA constraint, the auxiliary encoder conditions only on the context representation $s _ { x } ,$ , not on target information:

$$
h _ {\text { aux }} ^ {(l)} = \operatorname{GELU} (\text { LayerNorm } (\text { Linear } (h _ {\text { aux }} ^ {(l - 1)}))) \tag {32}
$$

$$
\mu_ {\phi} ^ {z}, \log \sigma_ {\phi} ^ {z} = \mathrm{MLP} _ {\phi} (\bar {s} _ {x}) \tag {33}
$$

$$
q _ {\phi} (z \mid s _ {x}) = \mathcal {N} \left(z; \mu_ {\phi} ^ {z}, \operatorname{diag} \left(\sigma_ {\phi} ^ {z}\right) ^ {2}\right) \tag {34}
$$

The auxiliary encoder uses fewer layers than the main encoders to maintain z as a lightweight predictive hint rather than a complex feature representation.

# D.2.3. TARGET POSTERIOR IMPLEMENTATION

ithin the tto obtain posterior pooled $q _ { \phi } \big ( s _ { y _ { k } } ^ { ( j ) } \big | s _ { x } , z , w ; m _ { \mathrm { t r g } } ^ { ( k ) } \big )$ ), we apply attention-based pooling to the variable-length context sequenceoncatenating with z and all feature tokens. This approach maintains proper $s _ { x }$ $L _ { \mathrm { p o o l } }$ conditioning on context latents $s _ { x }$ and auxiliary latent z while processing complete raw data w.

Pooled token preparation. The pooled context and auxiliary latent are projected into the token embedding space with type embeddings:

$$
t _ {\bar {s} _ {x}} ^ {(i)} = W _ {s _ {x}} \cdot \bar {s} _ {x} ^ {(i)} + e _ {\text { type }} ^ {\text { pooled - ctx }} \quad \text { for   } i = 1, \dots , L _ {\text { pool }} \tag {35}
$$

$$
t _ {z} = W _ {z} \cdot z + e _ {\text {type}} ^ {\text {aux - latent}} \tag {36}
$$

$W _ { s _ { x } } , W _ { z } \in \mathbb { R } ^ { D _ { \mathrm { h i d } } \times D _ { \mathrm { h i d } } }$ are learned projection matrices, and $e _ { \mathrm { t y p e } } ^ { \mathrm { p o o l e d - c t x } } , e _ { \mathrm { t y p e } } ^ { \mathrm { a u x - l a t e n t } } \in \mathbb { R } ^ { D _ { \mathrm { h i d } } }$

Feature tokenization. Each feature $w ^ { ( j ) }$ from the complete data is embedded using the same tokenization scheme as the context encoder:

$$
t _ {\text { feat }} ^ {(j)} = \left\{ \begin{array}{l l} W _ {\text { num }} \cdot w ^ {(j)} + b ^ {(j)} + e _ {\text { type }} ^ {\text { num }} + e _ {\text { pos }} ^ {(j)} & \text { if   } j \in \mathcal {I} _ {\text { num }} \\ \operatorname{Embed} _ {L _ {j}} (w ^ {(j)}) + e _ {\text { type }} ^ {\text { cat }} + e _ {\text { pos }} ^ {(j)} & \text { if   } j \in \mathcal {I} _ {\text { cat }} \end{array} \right. \tag {37}
$$

Fixed-size augmented sequence processing. The target posterior constructs a fixed-length augmented token sequence by concatenating CLS, pooled context, auxiliary, and feature tokens:

$$
\operatorname{seq} _ {\text { aug }} = \left[ \left\{t _ {\mathrm{CLS}} ^ {(i)} \right\} _ {i = 1} ^ {N _ {\mathrm{CLS}}}, \left\{t _ {\bar {s} _ {x}} ^ {(i)} \right\} _ {i = 1} ^ {L _ {\text { pool }}}, t _ {z}, \left\{t _ {\text { feat }} ^ {(j)} \right\} _ {j = 1} ^ {D _ {\text { feat }}} \right] \tag {38}
$$

$$
h _ {\text { aug }} = \text { Transformer } (\text { seq } _ {\text { aug }}) \tag {39}
$$

$$
\mu_ {\phi} ^ {s _ {w}} (h _ {\text { aug }}), \log \Sigma_ {\phi} ^ {s _ {w}} (h _ {\text { aug }}) = \text { Linear } (h _ {\text { aug }}), \text { Linear } (h _ {\text { aug }}) \tag {40}
$$

The target latent distributions are extracted from the feature portion of the augmented sequence:

$$
q _ {\phi} (s _ {y _ {k}} ^ {(j)} | s _ {x}, z, w; m _ {\mathrm{trg}} ^ {(k)}) = \mathcal {N} (s _ {y _ {k}} ^ {(j)}; \mu_ {\phi} ^ {s _ {w}} (h _ {\mathrm{feat}}) ^ {(j)}, \Sigma_ {\phi} ^ {s _ {w}} (h _ {\mathrm{feat}}) ^ {(j)}) \text {   for   } j \in m _ {\mathrm{trg}} ^ {(k)} \tag {41}
$$

where $h _ { \mathrm { f e a t } }$ denotes the feature token representations extracted from $h _ { \mathrm { a u g } }$ .

# D.2.4. PREDICTIVE MODEL IMPLEMENTATION

The predictive model $p _ { \theta } \big ( s _ { y _ { k } } ^ { ( j ) } \vert s _ { x } , z ; m _ { \mathrm { t r g } } ^ { ( k ) } \ $ |sx, z; m(k)trg ) operates entirely in latent space using learnable mask tokens. Unlike the target posterior, it has no access to raw target features and must predict latent representations from context and auxiliary information:

$$
h _ {\mathrm{ctx}} = \operatorname{Linear} \left(s _ {x}\right) + \operatorname{Linear} (z) + e _ {\text { pos }} ^ {\mathrm{ctx}} \tag {42}
$$

$$
h _ {\text { mask }} = \text { MaskToken } + \text { Linear } (z) + e _ {\text { pos }} ^ {\text { trg }} \tag {43}
$$

$$
h _ {\text { pred }} = \text { Transformer } (\text { concat } (h _ {\text { ctx }}, h _ {\text { mask }})) \tag {44}
$$

$$
\mu_ {\theta} ^ {s _ {y}} \left(h _ {\text { pred }}\right), \Sigma_ {\theta} ^ {s _ {y}} \left(h _ {\text { pred }}\right) = \operatorname{Linear} \left(h _ {\text { pred }}\right), \operatorname{Linear} \left(h _ {\text { pred }}\right) \tag {45}
$$

$$
p _ {\theta} (s _ {y _ {k}} ^ {(j)} | s _ {x}, z; m _ {\mathrm{trg}} ^ {(k)}) = \mathcal {N} (s _ {y _ {k}} ^ {(j)}; \mu_ {\theta} ^ {s _ {y}} (h _ {\mathrm{pred}}) ^ {(j)}, \Sigma_ {\theta} ^ {s _ {y}} (h _ {\mathrm{pred}}) ^ {(j)}) \tag {46}
$$

The predictive model does not directly distinguish between categorical and numerical features since it works purely in the latent representation space.

The predictor generates sequential target predictions for each target mask, enabling the model to handle multiple prediction tasks simultaneously during training. Crucially, this component has no access to target observations, maintaining the JEPA principle of predictive learning.

# D.2.5. CONTEXT AND TARGET DECODER IMPLEMENTATION

The context and target decoders reconstruct original tabular features from their latent representations using feature-specific decoder heads. Both decoders share identical architectures but operate on different latent sequences.

Feature-specific decoder architecture. Each decoder maintains separate neural network heads for every original feature in the dataset:

$$
\text { NumericDecoders } = \{u _ {\theta} ^ {(j)}: \mathbb {R} ^ {D _ {\mathrm{hid}}} \to \mathbb {R} ^ {1} \mid j \in \mathcal {I} _ {\mathrm{num}} \} \tag {47}
$$

$$
\text { CategoricalDecoders } = \{u _ {\theta} ^ {(j)}: \mathbb {R} ^ {D _ {\mathrm{hid}}} \to \mathbb {R} ^ {C _ {j}} \mid j \in \mathcal {I} _ {\mathrm{cat}} \} \tag {48}
$$

where $C _ { j }$ denotes the cardinality of categorical feature j. Each decoder head is implemented as a linear layer specific to the feature type and index.

Context decoder processing. Given a context latent sequence $s _ { x } \in \mathbb { R } ^ { M _ { \mathrm { c t x . } } \times D _ { \mathrm { h i d } } }$ and context feature indices $m _ { \mathrm { { c u x } } } \in \mathbb { R } ^ { M _ { \mathrm { { c u x } } } }$ , the context decoder processes each latent token:

$$
\text { for } t = 1, \dots , M _ {\mathrm{ctx}}: \quad j = m _ {\mathrm{ctx}} [ t ], \quad \hat {x} ^ {(j)} = u _ {\theta} ^ {(j)} (s _ {x} ^ {(t)}) \tag {49}
$$

Target decoder processing. The target decoder follows identical processing and operates directly on target latent sequences:

$$
\hat {w} ^ {(j)} = u _ {\theta} ^ {(j)} (s _ {w} ^ {(j)}) \tag {50}
$$

Reconstruction distributions. The decoder outputs are interpreted as distributional parameters for any observation $\pi \in \{ x , w \}$ with corresponding latent representation $s _ { \pi }$ :

$$
p _ {\theta} \left(\pi^ {(j)} \mid s _ {\pi} ^ {(j)}\right) = \left\{ \begin{array}{l l} \mathcal {N} \left(\pi^ {(j)}; u _ {\theta} ^ {(j)} \left(s _ {\pi} ^ {(j)}\right), \sigma^ {2}\right) & \text { if } j \in \mathcal {I} _ {\text { num }} \\ \operatorname{Categorical} \left(\pi^ {(j)}; \operatorname{softmax} \left(u _ {\theta} ^ {(j)} \left(s _ {\pi} ^ {(j)}\right)\right)\right) & \text { if } j \in \mathcal {I} _ {\text { cat }} \end{array} \right. \tag {51}
$$

where $\sigma ^ { 2 }$ is a learnable global noise parameter for numerical reconstruction uncertainty. The full reconstruction distribution factorizes as $\begin{array} { r } { p _ { \theta } ( \pi | s _ { \pi } ) = \prod _ { j \in m _ { \pi } } p _ { \theta } ( \pi ^ { ( j ) } | s _ { \pi } ^ { ( j ) } ) } \end{array}$ , where $m _ { \pi }$ denotes the mask for observation π.

# D.2.6. KL DIVERGENCES

We compute the closed form KL divergences between Gaussians, which is given by (Murphy, 2022)

$$
\mathrm{KL} \left(\mathcal {N} \left(\mu_ {q}, \Sigma_ {q}\right) \| \mathcal {N} \left(\mu_ {p}, \Sigma_ {p}\right)\right) = \frac {1}{2} \left[ + \operatorname{tr} \left(\Sigma_ {p} ^ {- 1} \Sigma_ {q}\right) + \left(\mu_ {p} - \mu_ {q}\right) ^ {\top} \Sigma_ {p} ^ {- 1} \left(\mu_ {p} - \mu_ {q}\right) + \log \frac {\det \Sigma_ {p}}{\det \Sigma_ {q}} - d \right]. \tag {52}
$$

In our implementation, all latent variables use diagonal covariance matrices $\Sigma = \mathrm { d i a g } ( \Sigma _ { 1 } , \dots , \Sigma _ { D _ { \mathrm { h i d } } } ) $ where each $\Sigma _ { d }$ represents the variance of dimension d. As our priors have $\mu _ { p } = 0 , \Sigma _ { p } = I$ for $s _ { x }$ and z, Eq. 52 simplifies to:

$$
\mathrm{KL} \big (q _ {\phi} (s _ {x} ^ {(j)} | x) \| \mathcal {N} (0, I) \big) = \frac {1}{2} \sum_ {d = 1} ^ {D _ {\mathrm{hid}}} \Big [ (\Sigma_ {\phi} ^ {s _ {x} ^ {(j, d)}}) + (\mu_ {\phi} ^ {s _ {x} ^ {(j, d)}}) ^ {2} - \log (\Sigma_ {\phi} ^ {s _ {x} ^ {(j, d)}}) - 1 \Big ]
$$

$$
\mathrm{KL} \big (q _ {\phi} (z | s _ {x}) \| \mathcal {N} (0, I) \big) = \frac {1}{2} \sum_ {d = 1} ^ {D _ {\mathrm{hid}}} \Big [ (\Sigma_ {\phi} ^ {z ^ {(d)}}) + (\mu_ {\phi} ^ {z ^ {(d)}}) ^ {2} - \log (\Sigma_ {\phi} ^ {z ^ {(d)}}) - 1 \Big ]
$$

The third KL term between two diagonal Gaussians becomes:

$$
\mathrm{KL} \big (q _ {\phi} (s _ {y _ {k}} ^ {(j)} | s _ {x}, z, w; m _ {\mathrm{trg}} ^ {(k)}) \| p _ {\theta} (s _ {y _ {k}} ^ {(j)} | s _ {x}, z; m _ {\mathrm{trg}} ^ {(k)}) \big) = \frac {1}{2} \sum_ {d = 1} ^ {D _ {\mathrm{hid}}} \Big [ \frac {(\Sigma_ {\phi} ^ {s _ {y _ {k}} ^ {(j , d)}})}{(\Sigma_ {\theta} ^ {s _ {y _ {k}} ^ {(j , d)}})} + \frac {(\mu_ {\theta} ^ {s _ {y _ {k}} ^ {(j , d)}} - \mu_ {\phi} ^ {s _ {y _ {k}} ^ {(j , d)}}) ^ {2}}{(\Sigma_ {\theta} ^ {s _ {y _ {k}} ^ {(j , d)}})} + \log \frac {(\Sigma_ {\theta} ^ {s _ {y _ {k}} ^ {(j , d)}})}{(\Sigma_ {\phi} ^ {s _ {y _ {k}} ^ {(j , d)}})} - 1 \Big ]
$$

where µ yϕ $\mu _ { \phi } ^ { s _ { y _ { k } } ^ { ( j , d ) } } = \mu _ { \phi } ^ { s _ { w } ^ { ( j , d ) } } : j \in m _ { \mathrm { t r g } } ^ { ( k ) }$ ∈ m (k)trg and Σs(j,d)ykϕ = $\Sigma _ { \phi } ^ { s _ { y _ { k } } ^ { ( j , d ) } } = \Sigma _ { \phi } ^ { s _ { w } ^ { ( j , d ) } } : j \in m _ { \mathrm { t r g } } ^ { ( k ) }$ Σs(j,d)w : j ∈ m trg are the masked means and covariances for target $y _ { k }$

# D.3. Training and Testing Procedure

This section provides detailed implementation specifics for training the Var-T-JEPA model, including KL annealing schedules and the training algorithm.

# D.3.1. KL DIVERGENCE ANNEALING

The KL divergence terms in the ELBO are gradually introduced during training through separate annealing schedules for each latent variable type. This prevents posterior collapse and ensures stable training dynamics, a common technique in training variational autoencoders (Bowman et al., 2016). We implement linear annealing schedules with configurable start times and durations:

$$
\alpha_ {s _ {x}} ^ {\mathrm{KL}} (t) = \min \left(\frac {t}{T _ {s _ {x}} ^ {\text { anneal }}}, 1\right) \cdot \alpha_ {s _ {x}, \text { final }} ^ {\mathrm{KL}} \tag {53}
$$

$$
\alpha_ {z} ^ {\mathrm{KL}} (t) = \min \left(\frac {t}{T _ {z} ^ {\text { anneal }}}, 1\right) \cdot \alpha_ {z, \text { final }} ^ {\mathrm{KL}} \tag {54}
$$

$$
\alpha_ {s _ {y}} ^ {\mathrm{KL}} (t) = \min \left(\frac {t}{T _ {s _ {y}} ^ {\text { anneal }}}, 1\right) \cdot \alpha_ {s _ {y}, \text { final }} ^ {\mathrm{KL}} \tag {55}
$$

where t denotes the current training step, T anneal· specifies the annealing duration, and $\alpha _ { \cdot , \mathrm { f i n a l } } ^ { \mathrm { K L } }$ sets the final weight values.

# D.3.2. TRAINING ALGORITHM

Algorithm 1 outlines the complete training procedure for our Var-T-JEPA for tabular data. Note that, while we used a per-sample notation in the main paper, the algorithm presents the batch-level implementation where loss terms represent averages over mini-batches and variables are indexed by sample position within the batch.

Algorithm 1: Var-T-JEPA Training Procedure   
Input: Dataset D, mask parameters ( $r_{ctx}^{min}, r_{ctx}^{max}, r_{trg}^{min}, r_{trg}^{max}$ ), number of target views K (and typically 1 context view)
Input: KL annealing schedules over optimizer steps $\{\alpha_{s_x}^{\mathrm{KL}}(t), \alpha_z^{\mathrm{KL}}(t), \alpha_{s_y}^{\mathrm{KL}}(t)\}$ while not converged and epoch < max epochs do

for each batch in dataloader do

Sample batch feature vectors $\{w^{(n)}\}_{n=1}^{B} \sim D;$ Generate masks (MaskCollator):

Sample mask sizes $M_{ctx}, M_{trg}$ uniformly from the configured ranges, then generate context masks $\{m_{ctx}^{(n)}\}_{n=1}^{B}$ and K target masks $\{m_{trg}^{(k,n)}\}_{k=1}^{K}$ per sample;

Get current KL weights from schedules at global optimizer-step t: $\alpha_{s_x}^{\mathrm{KL}}(t), \alpha_z^{\mathrm{KL}}(t), \alpha_{s_y}^{\mathrm{KL}}(t);$ Forward Pass (vectorized over target masks):

Sample $\{s_x^{(n)} \sim q_\phi(s_x | w^{(n)}; m_{ctx}^{(n)})\}_{n=1}^{B}$ using reparameterization trick;

Sample $\{z^{(n)} \sim q_\phi(z | s_x^{(n)})\}_{n=1}^{B}$ using a pooled context representation;

Concatenate target masks over k to form an effective target batch of size $B \cdot K$ ; repeat $s_x, z,$ and w to match this effective batch;

Sample target posterior latents for masked target features: $s_{y_k}^{(n)} \sim q_\phi(s_y | s_x^{(n)}, z^{(n)}, w^{(n)}; m_{trg}^{(k,n)});$ Predict target latents for the same masked features: $p_\theta(s_{y_k}^{(n)} | s_x^{(n)}, z^{(n)}; m_{trg}^{(k,n)});$ Compute Losses: $L^{\mathrm{rec}} = \frac{1}{B} \sum_{n=1}^{B} \frac{1}{M_{ctx}} \sum_{j \in m_{ctx}^{(n)}} - \log p_\theta(x^{(j,n)}| s_x^{(j,n)});$ $L^{\mathrm{gen}} = \frac{1}{B \cdot K} \sum_{n=1}^{B} \sum_{k=1}^{K} \frac{1}{D_{\mathrm{feat}}} \sum_{j=1}^{D_{\mathrm{feat}}} - \log p_\theta(w^{(j,n)} | s_w^{(j,n)});$ $L_{s_x}^{\mathrm{KL}} = \frac{1}{B} \sum_{n=1}^{B} \frac{1}{M_{ctx}} \sum_{j \in m_{ctx}^{(n)}} \mathrm{KL}(q_\phi(s_x^{(j,n)}| x^{(n)} || \mathcal{N}(0, I));$ $L_{z}^{\mathrm{KL}} = \frac{1}{B} \sum_{n=1}^{B} \mathrm{KL}(q_\phi(z^{(n)}| s_x^{(n)} || \mathcal{N}(0, I));$ $L_{s_y}^{\mathrm{KL}} = \frac{1}{B \cdot K} \sum_{n=1}^{B} \sum_{k=1}^{K} \frac{1}{M_{trg}} \sum_{j \in m_{trg}^{(k,n)}} \mathrm{KL}(q_\phi(s_{y_k}^{(j,n)}| s_x^{(n)}, z^{(n)}, w^{(n)}; m_{trg}^{(k,n)} || p_\theta(s_{y_k}^{(j,n)}| s_x^{(n)}, z^{(n)}; m_{trg}^{(k,n)}));$ $L = \alpha^{\mathrm{rec}} L^{\mathrm{rec}} + \alpha^{\mathrm{gen}} L^{\mathrm{gen}} + \alpha_{s_x}^{\mathrm{KL}}(t) L_{s_x}^{\mathrm{KL}} + \alpha_z^{\mathrm{KL}}(t) L_z^{\mathrm{KL}} + \alpha_{s_y}^{\mathrm{KL}}(t) L_{s_y}^{\mathrm{KL}};$ Compute gradients: $\nabla_\theta,_\phi L;$ Update parameters using optimizer (e.g., AdamW);

Increment KL annealing step: $t \leftarrow t + 1;$ Optionally run downstream probe every C epochs and update best checkpoint based on validation score;

if no downstream improvement for P probe evaluations then

Load best checkpoint and terminate training;

# D.4. T-JEPA Implementation for Tabular Data

T-JEPA (Thimonier et al., 2025) employs a deterministic joint-embedding architecture with three main components: context encoder $f _ { \theta } ^ { \mathrm { c t x } }$ , target encoder $f _ { \theta } ^ { \mathrm { t r g } }$ , and predictor $g _ { \theta }$ , as illustrated in Figure 6.

Context and target encoders. Both encoders $f _ { \theta } ^ { \mathrm { c t x } }$ and $f _ { \theta } ^ { \mathrm { t r g } }$ share identical architectures but operate on different masked views of the input. T-JEPA uses the same tokenization scheme as described in Eq. 27 and Eq. 37, processing tabular features through feature-specific embeddings with type and positional encodings. The tokenized features are processed by a transformer encoder to produce deterministic representations $s _ { x } = f _ { \theta } ^ { \mathrm { c t x } } ( x )$ and $s _ { y } = f _ { \theta } ^ { \mathrm { t r g } } ( w ; m _ { \mathrm { t r g } } )$ for context and target views respectively.

Predictor. As shown in Figure 6, the predictor $g _ { \theta }$ learns to forecast target representations from context representations. The predictor is implemented using a similar transformer architecture as our Var-T-JEPA, including learnable mask tokens with positional embeddings, used to predict target features $\hat { s } _ { y } = g _ { \theta } ( s _ { x } ; m _ { \mathrm { t r g } } )$ .

![](images/ab2b0df17bc7262935d5da83083e7683e4913cfb77bc3a4523378be8028549f4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Original Sample"] --> B["Context Mask"]
    B --> C["Context"]
    C --> D["Context Encoder"]
    D --> E["Context Rep."]
    E --> F["Predictor"]
    F --> G["Target Pred."]
    
    subgraph Target Masks
        H1["\mathcal{M}={m_{\text{img}}^{(1)} m_{\text{img}}^{(2)} m_{\text{my}}^{(k)}}\n\ny_k = m_{\text{img}}^{(k)} \odot w \in \mathbb{R}^{\text{M}_{\text{img}}} \n}"]
    end
    
    subgraph Target Encoder
        I1["\mathcal{M}={m_{\text{img}}^{(1)} m_{\text{img}}^{(2)} m_{\text{my}}^{(k)}}\n\ny_k = m_{\text{img}}^{(k)} \odot w \in \mathbb{R}^{\text{M}_{\text{img}}} \n}"]
    end
    
    subgraph Target Rep.
        J1["\mathcal{S}_{y_k} = f_\phi(w; m_{\text{img}}^{(k)})\n\ns_{y_1}\n\ns_{y_2}\n\ns_{y_k}\n\ns_{y_k} \in \mathbb{R}^{\text{M}_{\text{img}} \times \text{D}_{\text{hid}}} \n\nTesting"]
        K1["\Downstream Model"]
    end
    
    L["Original Sample"] --> M["m_{ctx}"]
    M --> N["x = m_{ctx} ⊙ w"]
    N --> O["R^M_{ctx}"]
    
    subgraph Testing
        P1["\mathcal{S}_{w}"] --> Q["\mathcal{S}_{y_k}"]
        Q --> R["\mathcal{S}_{y_1}"]
        Q --> S["\mathcal{S}_{y_2}"]
        Q --> T["\mathcal{S}_{y_k}"]
        Q --> U["\mathcal{S}_{y_k} \in \mathbb{R}^{\text{M}_{\text{img}} \times \text{D}_{\text{hid}}} \n\nEMA"]
    end
    
    subgraph Target Mask
        V1["\mathcal{M}={m_{\text{img}}^{(1)} m_{\text{img}}^{(2)} m_{\text{my}}^{(k)}}\n\ny_k = m_{\text{img}}^{(k)} \odot w \in \mathbb{R}^{\text{M}_{\text{img}}} \n}"]
    end
    
    subgraph Target Rep.
        W1["\mathcal{S}_{y_k}"] --> X["\mathcal{S}_{y_1}\n\ns_{y_2}\n\ns_{y_k}\n\ns_{y_k} \in \mathbb{R}^{\text{M}_{\text{img}} \times \text{D}_{\text{hid}}} \n\nTesting"]
    end
    
    style Target Masks fill:#f9f,stroke:#333
    style Target Encoder fill:#ccf,stroke:#333
    style Target Rep fill:#cfc,stroke:#333
```
</details>

Figure 6. Architectural overview of the T-JEPA pipeline by Thimonier et al. (2025) extended by learnable auxiliary tokens.

Training objective. T-JEPA minimizes the averaged mean squared error between predicted and actual target representations over all target masks and features:

$$
\mathcal {L} _ {\mathrm{T-JEPA}} = \frac {1}{K \cdot M _ {\mathrm{trg}}} \sum_ {k = 1} ^ {K} \sum_ {j \in m _ {\mathrm{trg}} ^ {(k)}} \| \hat {s} _ {y _ {k}} ^ {(j)} - s _ {y _ {k}} ^ {(j)} \| ^ {2} \tag {56}
$$

The target encoder parameters are updated via exponential moving average (EMA) of the context encoder parameters to ensure stable training, while the predictor is trained end-to-end with gradient descent.

Learnable auxiliary tokens. The original T-JEPA framework by Thimonier et al. (2025) does not incorporate the auxiliary variable z in its predictor architecture. To match our theoretical formulation where predictions depend on both context representations $s _ { x }$ and auxiliary variables z, we introduce $N _ { \mathrm { a u x } }$ learnable auxiliary tokens AuxToken $\mathbf { \tau } \in \mathbb { R } ^ { N _ { \mathrm { { a u x } } } \times D _ { \mathrm { { h i d } } } }$ as trainable parameters that serve as dedicated slots for auxiliary predictive information. These tokens are augmented with distinct positional embeddings $e _ { \mathrm { p o s } } ^ { \mathrm { a u x } } \in \mathbb { R } ^ { N _ { \mathrm { a u x } } \times D _ { \mathrm { h i d } } }$ and concatenated with context and target tokens in the predictor input sequence:

$$
\operatorname{Concat} \left(s _ {x}, \text {AuxToken} + e _ {\text {pos}} ^ {\text {aux}}, \text {MaskToken} + e _ {\text {pos}} ^ {\text {trg}}\right)\rightarrow \text {Transformer} \rightarrow \hat {s} _ {y _ {k}} \tag {57}
$$

where MaskToken $+ e _ { \mathrm { p o s } } ^ { \mathrm { t r g } }$ denotes the masked target tokens with their positional embeddings, and $\hat { s } _ { y _ { k } }$ represents the predicted target latent representation. The auxiliary tokens provide learnable parameters that can encode predictive hints analogous to sampling from $q _ { \phi } ( z | s _ { x } )$ in our variational framework, enabling the deterministic T-JEPA predictor to access auxiliary predictive capacity while maintaining end-to-end differentiability.

# D.5. Embedding Generation and Uncertainty Estimates for Downstream Evaluation

For downstream evaluation, our tabular Var-T-JEPA deviates from the general test-time procedure described in Section 3.6 in two key aspects. First, the target posterior $q _ { \phi } \big ( s _ { y _ { k } } ^ { ( j ) } \big | s _ { x } , z , w ; m _ { \mathrm { t r g } } ^ { ( k ) } \big )$ ) conditions on the feature vector w since tabular features are accessible during embedding generation, unlike the general Var-JEPA framework where target observations may generally be unavailable. Second, the auxiliary latent is generated via the learned auxiliary encoder $q _ { \phi } ( z | s _ { x } )$ rather than sampling from the standard normal prior $p ( z ) = \mathcal { N } ( 0 , I )$ , maintaining consistency with the training regime where the auxiliary encoder learns meaningful context-to-auxiliary mappings for downstream representation quality. Following standard practice in representation learning, final embeddings for downstream analysis were generated using distributional means rather than sampling to ensure deterministic representations: $s _ { x } = \mu _ { \phi } ^ { s _ { x } } , z = \mu _ { \phi } ^ { z }$ , and $s _ { w } = \mu _ { \phi } ^ { s _ { w } }$ . For our downstream evaluation experiments of both our tabular Var-T-JEPA and T-JEPA, we use the target posterior representations $s _ { w }$ as the final embeddings, which capture the joint information from context, auxiliary latents, and complete feature observations. Uncertainty estimates for selective evaluation are computed by aggregating the standard deviations of the target latent posterior distribution: for real-world datasets (AD, CO, EL, CC, BM) we use mean aggregation, while for (semi-)synthetic datasets (MNIST, SIM) with known ground-truth uncertainty we use the 90th percentile to better capture tail behavior. For embedding and uncertainty extraction, checkpoints are selected based on online linear-probe validation performance: for T-JEPA we use the best unfiltered validation accuracy, while for Var-T-JEPA we use the best filtered validation accuracy with the most-uncertain 20% of validation samples discarded.

# D.6. Hyperparameter Settings

# D.6.1. COMPUTE RESOURCES

Var-T-JEPA and T-JEPA were trained on a computer cluster with one NVIDIA A100 GPU (40GB or 80GB VRAM), two CPU cores, and 20 GB system RAM per task. Var-T-JEPA training time varied by dataset and configuration, and was typically between under 2 hours and less than 24 hours per run across our experiments.

# D.6.2. VAR-T-JEPA HYPERPARAMETERS

Table 7 summarizes the hyperparameters used for the tabular Var-T-JEPA experiments. Notation: ‘batch‘: batch size; ‘lr‘: learning rate; ‘warm‘: warmup epochs; ‘ctx‘/‘ctxM‘: min/max context mask share; ‘trg‘/‘trgM‘: min/max target mask share; ‘K‘: number of target predictions; ‘d‘: hidden dimension; ‘L‘: number of layers; ‘H‘: number of heads; ‘ff‘: feedforward on; ‘dropp‘: predictor dro: KL annealing epochs for $^ { \cdot } \alpha _ { s _ { x } } ^ { \mathrm { K L } \cdot } , ^ { \cdot } \alpha _ { z } ^ { \mathrm { K L } \cdot } , ^ { \cdot } \alpha _ { s _ { y } } ^ { \mathrm { K L } \cdot } \mathrm { ~ : ~ }$ final KL weights for n weights for x, y. $s _ { x } ,$ $s _ { y } ; \ { ' } T _ { s _ { x } } ^ { \mathrm { a n n e a l } * } , \ { ' } T _ { z } ^ { \mathrm { a n n e a l } * } , \ { ' } T _ { s _ { y } } ^ { \mathrm { a n n e a l } * }$ $s _ { x } , z , s _ { y } ; \ { } ^ { \circ } \alpha ^ { \mathrm { r e c } } , \ { } ^ { \circ } \alpha ^ { \mathrm { g e n } } { } ^ { \circ }$

Table 7. Var-T-JEPA hyperparameters used for downstream experiments. 

<table><tr><td>Dataset</td><td>batch</td><td>lr</td><td>warm</td><td>ctx</td><td>ctxM</td><td>trg</td><td>trgM</td><td>K</td><td>d</td><td>L</td><td>H</td><td>ff</td><td>drop</td><td>Lp</td><td>Hp</td><td>ffp</td><td>dropp</td><td> $\alpha_{sx}^{\text{KL}}$ </td><td> $\alpha_z^{\text{KL}}$ </td><td> $\alpha_{sy}^{\text{KL}}$ </td><td> $T_{sx}^{\text{anneal}}$ </td><td> $T_z^{\text{anneal}}$ </td><td> $T_{sy}^{\text{anneal}}$ </td><td> $\alpha^{\text{rec}}$ </td><td> $\alpha^{\text{gen}}$ </td></tr><tr><td>AD</td><td>512</td><td>0.001</td><td>10</td><td>0.1</td><td>0.3</td><td>0.1</td><td>0.6</td><td>4</td><td>64</td><td>8</td><td>4</td><td>256</td><td>0.001</td><td>16</td><td>4</td><td>256</td><td>0.002</td><td>1e-04</td><td>1e-06</td><td>1e-05</td><td>15</td><td>15</td><td>15</td><td>0.1</td><td>1</td></tr><tr><td>CO</td><td>512</td><td>5e-04</td><td>12</td><td>0.15</td><td>0.35</td><td>0.15</td><td>0.6</td><td>4</td><td>64</td><td>8</td><td>8</td><td>64</td><td>0.0015</td><td>16</td><td>4</td><td>256</td><td>0.002</td><td>1e-06</td><td>1e-05</td><td>1e-06</td><td>60</td><td>60</td><td>20</td><td>0.001</td><td>0.5</td></tr><tr><td>EL</td><td>512</td><td>5e-04</td><td>10</td><td>0.15</td><td>0.6</td><td>0.15</td><td>0.9</td><td>4</td><td>64</td><td>6</td><td>4</td><td>64</td><td>0.001</td><td>16</td><td>4</td><td>128</td><td>0.002</td><td>1e-06</td><td>1e-05</td><td>1e-06</td><td>40</td><td>40</td><td>15</td><td>0.001</td><td>0.25</td></tr><tr><td>CC</td><td>512</td><td>0.001</td><td>10</td><td>0.1</td><td>0.3</td><td>0.1</td><td>0.6</td><td>4</td><td>64</td><td>8</td><td>4</td><td>256</td><td>0.001</td><td>16</td><td>4</td><td>256</td><td>0.002</td><td>1e-04</td><td>1e-06</td><td>1e-05</td><td>15</td><td>15</td><td>15</td><td>0.1</td><td>1</td></tr><tr><td>BM</td><td>512</td><td>0.001</td><td>10</td><td>0.1</td><td>0.3</td><td>0.1</td><td>0.6</td><td>4</td><td>64</td><td>8</td><td>4</td><td>256</td><td>0.001</td><td>16</td><td>4</td><td>256</td><td>0.002</td><td>1e-04</td><td>1e-06</td><td>1e-05</td><td>15</td><td>15</td><td>15</td><td>0.1</td><td>1</td></tr><tr><td>MNIST</td><td>256</td><td>0.001</td><td>10</td><td>0.15</td><td>0.5</td><td>0.15</td><td>0.8</td><td>4</td><td>64</td><td>8</td><td>4</td><td>128</td><td>0.002</td><td>32</td><td>4</td><td>256</td><td>0.002</td><td>1e-06</td><td>1e-06</td><td>1e-06</td><td>100</td><td>100</td><td>100</td><td>0.001</td><td>0.1</td></tr><tr><td>SIM</td><td>512</td><td>5e-04</td><td>10</td><td>0.15</td><td>0.5</td><td>0.15</td><td>0.8</td><td>4</td><td>64</td><td>16</td><td>2</td><td>64</td><td>0.002</td><td>16</td><td>4</td><td>256</td><td>0.002</td><td>1e-06</td><td>1e-06</td><td>1e-05</td><td>50</td><td>50</td><td>50</td><td>0.001</td><td>0.1</td></tr></table>

# D.6.3. T-JEPA HYPERPARAMETERS

Table 8 summarizes the hyperparameters used for the tabular T-JEPA experiments. For any hyperparameter that has a direct counterpart in Var-T-JEPA same setting as in Var-T-JEPA. Notation matches Var-T-JEPA: ‘batch‘: batch size; ‘lr‘: learning rate; ‘warm‘: warmup epochs; ‘ctx‘/‘ctxM‘: min/max context mask share; ‘trg‘/‘trgM‘: min/max target mask share; $^ { \mathfrak { \bullet } } \mathbf { K } ^ { \mathfrak { \bullet } }$ : number of target predictions; ‘d‘: hidden dimension; ‘L‘: number of layers; ‘H‘: number of heads; $\cdot _ { \mathrm { f f } ^ { 6 } ) }$ feedforward dimension; ‘drop‘: dropout probability; ‘Lp‘: predictor embedding dimension; $\cdot _ { \mathrm { H p } } \cdot _ { \mathrm { \ell } }$ : predictor number of heads; ‘ffp‘: predictor feedforward dimension; ‘dropp‘: predictor dropout probability; ‘ema‘: EMA decay rate.

Table 8. T-JEPA hyperparameters used for downstream experiments. 

<table><tr><td>Dataset</td><td>batch</td><td>lr</td><td>warm</td><td>ctx</td><td>ctxM</td><td>trg</td><td>trgM</td><td>K</td><td>d</td><td>L</td><td>H</td><td>ff</td><td>drop</td><td>Lp</td><td>Hp</td><td>ffp</td><td>dropp</td><td>ema</td></tr><tr><td>AD</td><td>512</td><td>0.001</td><td>10</td><td>0.1</td><td>0.3</td><td>0.1</td><td>0.6</td><td>4</td><td>64</td><td>8</td><td>4</td><td>256</td><td>0.001</td><td>16</td><td>4</td><td>256</td><td>0.002</td><td>0.996</td></tr><tr><td>CO</td><td>512</td><td>5e-04</td><td>12</td><td>0.15</td><td>0.35</td><td>0.15</td><td>0.6</td><td>4</td><td>64</td><td>8</td><td>8</td><td>64</td><td>0.0015</td><td>16</td><td>4</td><td>256</td><td>0.002</td><td>0.996</td></tr><tr><td>EL</td><td>512</td><td>5e-04</td><td>10</td><td>0.15</td><td>0.6</td><td>0.15</td><td>0.9</td><td>4</td><td>64</td><td>6</td><td>4</td><td>64</td><td>0.001</td><td>16</td><td>4</td><td>128</td><td>0.002</td><td>0.996</td></tr><tr><td>CC</td><td>512</td><td>0.001</td><td>10</td><td>0.1</td><td>0.3</td><td>0.1</td><td>0.6</td><td>4</td><td>64</td><td>8</td><td>4</td><td>256</td><td>0.001</td><td>16</td><td>4</td><td>256</td><td>0.002</td><td>0.996</td></tr><tr><td>BM</td><td>512</td><td>0.001</td><td>10</td><td>0.1</td><td>0.3</td><td>0.1</td><td>0.6</td><td>4</td><td>64</td><td>8</td><td>4</td><td>256</td><td>0.001</td><td>16</td><td>4</td><td>256</td><td>0.002</td><td>0.996</td></tr><tr><td>MNIST</td><td>256</td><td>0.001</td><td>10</td><td>0.15</td><td>0.5</td><td>0.15</td><td>0.8</td><td>4</td><td>64</td><td>8</td><td>4</td><td>128</td><td>0.002</td><td>32</td><td>4</td><td>256</td><td>0.002</td><td>0.996</td></tr><tr><td>SIM</td><td>512</td><td>5e-04</td><td>10</td><td>0.15</td><td>0.5</td><td>0.15</td><td>0.8</td><td>4</td><td>64</td><td>16</td><td>2</td><td>64</td><td>0.002</td><td>16</td><td>4</td><td>256</td><td>0.002</td><td>0.996</td></tr></table>

# D.6.4. DOWNSTREAM AND BASELINE MODEL HYPERPARAMETERS

Downstream predictors were trained either on raw features or on learned embeddings. For embedding-based evaluation, embeddings are flattened for neural models. For raw-feature baselines, standard tabular encoders are used. All neural models (MLP, DCNv2, ResNet, AutoInt, FT-Transformer) share common hyperparameters: learning rate 0.001, weight decay 0.01, batch size 128, 50 training epochs with exponential patience of 16. Model-specific parameters include: MLP (2 hidden layers, hidden dimension 128, dropout 0.1); DCNv2 (2 cross layers); ResNet (3 blocks, output dimension 128, block dimension 128, hidden dimension 256, dropout 0.1); AutoInt (token dimension 128, 2 attention heads, 3 layers); FT-Transformer (standard transformer architecture). XGBoost uses a tree-based configuration with 100 estimators, max depth 3, learning rate 0.1, and does not use embedding flattening. We additionally use an online linear probe for downstream evaluation during Var-T-JEPA training and model selection, trained with learning rate $1 0 ^ { - 3 }$ , weight decay $2 \times 1 0 ^ { - 5 }$ , batch size 128, and 50 epochs. For the risk–coverage curves in Figure 3, we train a linear probe on standardized embeddings for 20 epochs with learning rate $1 0 ^ { - 2 }$ and batch size 8192.

# D.6.5. SIMULATION STUDY (VAR-JEPA) HYPERPARAMETERS

We use batch size 512. Training runs for 40 epochs with learning rate $1 0 ^ { - 3 }$ and weight decay $1 0 ^ { - 6 }$ . The Var-JEPA MLP uses hidden dimension 128 and depth 2. Default loss weights are directions, 64 frequencies, maximum frequency 5.0, with strengt $\alpha ^ { \mathrm { r e c } } = \alpha ^ { \mathrm { g e n } } = \alpha _ { s _ { x } } ^ { \mathrm { K L } } = \alpha _ { z } ^ { \mathrm { K L } } = \alpha _ { s _ { u } } ^ { \mathrm { K L } } = 1$ . SIGReg uses 64bes are trained for $\lambda _ { s _ { x } } = 1 0$ $\lambda _ { s _ { y } } = 1 0$ 50 epochs with learning rate $3 \times 1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , batch size 512, and hidden dimension 64.