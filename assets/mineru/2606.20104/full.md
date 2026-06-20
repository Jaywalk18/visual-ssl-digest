# Sensorimotor World Models: Perception for Action via Inverse Dynamics

Petr Ivashkov∗1 Randall Balestriero 2 Bernhard Schölkopf 1,3,4

1Max Planck Institute for Intelligent Systems, Tübingen, Germany 2Department of Computer Science, Brown University, Providence, RI, USA 3ELLIS Institute, Tübingen, Germany 4ETH Zürich, Zürich, Switzerland

![](images/67946e1013216ab948f55d0f6c2e04e9d19b3e4c35aea2628df1428a4d65fd47.jpg)

Webpage

![](images/06b4cb3bad00096239125c2971824541b2c9e72a620f3b49c13ca15afbd67a23.jpg)

Code

## Abstract

Perception for action suggests that representations of the world should be shaped not by visual fidelity alone, but by their relevance for actions. At the same time, latent JEPA-style world models advocate learning compact predictive states from high-dimensional observations to facilitate the prediction of future states, but endto-end training of these models is nontrivial because representations may collapse if our only goal is to construct a latent state that is easy to predict. We introduce a sensorimotor world model (SMWM): a latent world model trained end-to-end with inverse dynamics regularization. This single regularizer addresses both issues: it prevents representation collapse and induces action-aligned representations. By forcing latent states to preserve information about the action underlying a transition, it biases the model toward the controllable degrees of freedom of the environment while discarding uncontrollable distractors. This yields stable latent world models trained from offline, reward-free trajectories, without frozen encoders, exponential moving averages, or complex latent regularizers. Empirically, SMWM learns compact, interpretable latent spaces and enables competitive planning performance across simple 2D and 3D control tasks.

## 1 Introduction

World models that predict future states given actions are central to intelligent agents [1–3]. Learning such models directly in pixel space is difficult because observations are high-dimensional and pixellevel reconstruction encourages the model to capture background details that are often irrelevant for control. Latent world models avoid these issues by learning dynamics in a representation space rather than in observation space [2, 4–6], and more recently, Joint Embedding Predictive Architectures (JEPAs) [7–9] dispense with reconstruction altogether and predict directly in embedding space.

JEPA-style world models from pixels are appealing but susceptible to representation collapse: when the encoder and the latent dynamics model are trained jointly to make correct predictions in embedding space, the encoder may learn to map all observations to a constant embedding, making the dynamics prediction task trivial while rendering the learned model useless for other tasks [10, 11]. Recent work approaches this problem from several angles. DINO-WM [12] sidesteps the problem by freezing a pretrained encoder. PLDM [13] adds a multi-term variance–covariance regularizer on the embeddings. LeWorldModel [14, 15] matches the embedding distribution to an isotropic Gaussian via SIGReg. V-JEPA 2 [16] uses stop-gradient targets from an exponential-moving-average target encoder.

While representation collapse is an important practical problem, it also points to the basic question of what a useful world model representation should retain. It has been argued that this connects to causality: a causal representation [17] should not merely encode predictive correlations, but also actions and their effects, to provide an interventional world model [3]. In cognitive psychology and neuroscience, the term perception for action captures the view that perception ultimately subserves action [18]. Common coding proposes that perceived events and planned actions share a unified representational domain, with actions represented partly in terms of their anticipated perceptual effects [19], rendering the content of a representation dependent on the organism’s action repertoire.2 Representations should thus not only be judged by how much information they preserve about observations, but by whether they preserve the distinctions that matter for the actions available to the agent and for the consequences of those actions.

![](images/bf4aaa6d71476dab49f15697325c0fa3dd69b1188c6da42a7bb2d60475e80beb.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Offline dataset
  O_t["O_t"] --> Encoder1["Encoder"]
  O_t+1["O_{t+1}"] --> Encoder2["Encoder"]
    end
    
    subgraph Offline dataset
  A["a_t"] --> Encoder3["Encoder"]
  A --> Encoder4["Encoder"]
    end
    
    subgraph Inverse dataset
  Z_t1["Z_{t+1}"] --> ForwardModel["Forward model"]
  Z_t2["Z_{t+1}"] --> ForwardModel
  Z_t3["Z_{t+1}"] --> InverseModel["Inverse model"]
  Z_t4["Z_{t+1}"] --> InverseModel
  Z_t5["Z_{t+1}"] --> InverseModel
    
  Z_t1 -->|at| ForwardModel
  Z_t2 -->|at| ForwardModel
  Z_t3 -->|at| ForwardModel
  Z_t4 -->|at| ForwardModel
  Z_t5 -->|at| InverseModel
  Z_t6["Z_{t+1}"] -->|MSE forward loss| ForwardModel
    
  Z_t7["â_t"] --> InverseModel
    
  InverseModel -->|prevents collapse| Z_t1
  InverseModel -->|prevents collapse| Z_t2
  InverseModel -->|prevents collapse| Z_t3
  InverseModel -->|prevents collapse| Z_t4
  InverseModel -->|prevents collapse| Z_t5
    
    style Offline dataset fill:#f9f,stroke:#333
    style Inverse dataset fill:#ccf,stroke:#333
```
</details>

Figure 1: Method overview. We train an encoder $f _ { \theta } { } _ { ; }$ , a forward dynamics model $g _ { \phi }$ , and an inverse dynamics model $h _ { \psi }$ jointly from an offline dataset of transitions $\left( o _ { t } , a _ { t } , o _ { t + 1 } \right)$ . The encoder maps each observation to a compact embedding, $z _ { t } = f _ { \theta } ( o _ { t } )$ and $z _ { t + 1 } = f _ { \theta } ( o _ { t + 1 } )$ . The forward model predicts the next embedding from the current embedding and action, $\hat { z } _ { t + 1 } = g _ { \phi } ( z _ { t } , a _ { t } )$ , and is supervised by the mean-squared forward loss $\mathcal { L } _ { \mathrm { f w d } }$ between $\hat { z } _ { t + 1 }$ and $z _ { t + 1 }$ . In parallel, the inverse model predicts the executed action from the two embeddings, $\hat { a } _ { t } = h _ { \psi } ( z _ { t } , z _ { t + 1 } )$ , supervised by the mean-squared inverse loss ${ \mathcal { L } } _ { \mathrm { i n v } }$ between $\hat { a } _ { t }$ and $a _ { t } .$ . Both losses backpropagate into the encoder through the joint objective $\begin{array} { r } { \mathcal { L } = \mathcal { L } _ { \mathrm { f w d } } + \lambda \mathcal { L } _ { \mathrm { i n v } } . } \end{array}$ . The inverse loss acts as an anti-collapse mechanism: to enable action recovery, $f _ { \theta }$ must preserve the action-relevant information contained in $\left( o _ { t } , o _ { t + 1 } \right)$ .

Enactive perception is the theory that perception is not a passive reception of sensory input, but an active bodily exploration of the world [22]. A recent instantiation of this view focuses on sensorimotor contingencies as the lawful relationships between an agent’s actions and the resulting changes in sensory input [23]. With this in mind, we return to our practical problem, i.e., collapse: can we train a latent world model whose representation respects these contingencies by retaining information sufficient to recover the agent’s actions? We call the resulting object a sensorimotor world model (SMWM): a predictive latent model whose state is shaped by the coupling between motor commands and sensory change.

We use a simple and principled mechanism to instantiate this idea: inverse dynamics regularization. We add a single-step inverse dynamics head that predicts the action taken between two consecutive observations from their embeddings, and we propagate gradients from this head into the encoder, as illustrated in Fig. 1. The intuition is direct: if two consecutive embeddings contain enough information to recover the action that produced the transition, they must preserve features of the observations that are relevant for controllable dynamics. This encourages the encoder to retain action-relevant information while discarding the rest. Unlike distributional regularizers that prescribe the geometry of the embedding space, inverse dynamics anchors the representation to a task-grounded quantity—the action.

Setting and contributions. We consider offline datasets of trajectories consisting of video frames and corresponding continuous actions, without rewards, task labels, or knowledge of the datacollecting policy. This setting broadly covers demonstration data, action-annotated video datasets, and logged interaction data. Our contributions are:

1. SMWM, a simple sensorimotor world model trained end-to-end from pixels, using inverse dynamics regularization as the sole anti-collapse mechanism with one additional hyperparameter,  
2. evidence that the learned representations are not merely non-collapsed, but track the controllable degrees of freedom, preserve spatial geometry, and filter out uncontrollable distractors,  
3. competitive planning performance against SIGReg-regularized baselines across 2D and 3D control tasks, showing that this structural advantage carries over to downstream control.

## 2 Background and related work

Latent world models. Learning dynamics models in learned latent spaces has a rich history. The Dreamer family [2, 4, 24] and PlaNet [25] learn latent dynamics with a pixel decoder under a variational objective. TD-MPC [5, 6] drops the decoder and trains a joint encoder and latent dynamics model end-to-end with a temporal-difference objective. JEPAs [7–9] push this further, dispensing with both reconstruction and reward signal and predicting directly in embedding space. Our work belongs to the JEPA family and operates in the offline, reward-free regime.

Representation collapse. The self-supervised learning literature has charted a design space of remedies for representation collapse: contrastive losses with negative samples [26, 27], asymmetric architectures with stop-gradients [10, 28], and regularizers on the embedding distribution [11, 29]. Recent JEPA world models inherit choices from this design space: DINO-WM [12] freezes a pretrained encoder, PLDM [13] adds multiple auxiliary loss terms including variance–covariance regularization, temporal smoothness, and inverse dynamics, LeWorldModel [14, 15] matches embeddings to an isotropic Gaussian, and V-JEPA 2 [16] uses stop-gradient and exponential moving average target networks. We share the JEPA training framework and the offline pixel-action setting with these works but use inverse-dynamics as the standalone anti-collapse mechanism, as described in Sec. 3.

Inverse dynamics. The use of inverse dynamics for representation learning is not new. Pathak et al. [30] use it to learn features that capture controllable aspects of the environment for intrinsic-reward computation. A line of theoretical work shows that, in finite rich-observation settings with exogenous distractors, multi-step inverse objectives can recover control-relevant latent state information under certain assumptions [31–34]. While none of these results transfers to our continuous-state, continuousaction setting with a single-step inverse, we treat them as motivation for using action prediction as an inductive bias toward controllable state structure. Concurrent work has begun to combine inverse dynamics with world-model objectives in related but distinct settings [35–37]; we discuss our differences from each of these works in Sec. H.

Other directions. Several adjacent lines share architectural ingredients but address different problems. Action-free or latent-action world models infer pseudo-actions from video and use them to supervise dynamics: LAPO [38] and DynaMo [39] use a learned inverse dynamics model to extract latent actions from action-free demonstrations, and APV [40] pretrains an action-free latent video predictor for downstream action-conditional world-model learning. Visual RL methods share encoders between auxiliary objectives and policies but do not learn explicit world models [41–43]. Bisimulation metrics offer a complementary lens on which states should map to similar embeddings [44, 45], and causal abstraction [46] formalizes a related consistency between structural models via a commutative diagram on states and interventions. Sec. H provides further detail.

## 3 Method

Dataset and setting. We assume access to an offline dataset $\mathcal { D } = \{ ( o _ { t } , a _ { t } , o _ { t + 1 } ) \}$ of transitions, where $o _ { t } , o _ { t + 1 } \in \mathcal { O }$ are consecutive observations (e.g., video frames) and $a _ { t } \in \mathcal { A } \subseteq \mathbb { R } ^ { m }$ is a continuous action taken in between. We do not require a behavior policy or a reward function. Our goal is to learn an encoder $f _ { \theta } : \mathcal { O }  \mathcal { Z }$ that maps observations to a compact embedding space $\mathcal { Z } \subseteq \mathbb { R } ^ { d }$ , together with a forward dynamics model $g _ { \phi } : \mathcal { Z } \times \mathcal { A } \to \mathcal { Z }$ that predicts the next embedding from the current embedding and action.

Forward dynamics in embedding space. Given a transition $\left( o _ { t } , a _ { t } , o _ { t + 1 } \right)$ , we encode both observations and train the dynamics model to match the embedding of the next observation,

$$
z _ {t} = f _ {\theta} \left(o _ {t}\right), \quad z _ {t + 1} = f _ {\theta} \left(o _ {t + 1}\right), \quad \hat {z} _ {t + 1} = g _ {\phi} \left(z _ {t}, a _ {t}\right), \tag {1}
$$

by minimizing the mean-squared forward loss

$$
\mathcal {L} _ {\mathrm{fwd}} = \mathbb {E} _ {(o _ {t}, a _ {t}, o _ {t + 1}) \sim \mathcal {D}} \left[ \| \hat {z} _ {t + 1} - z _ {t + 1} \| _ {2} ^ {2} \right]. \tag {2}
$$

This objective alone admits a trivial solution: if $f _ { \theta }$ maps every observation to a constant embedding $z ^ { \star }$ and $g _ { \phi }$ outputs $z ^ { \star }$ regardless of its inputs, then ${ \mathcal { L } } _ { \mathrm { f w d } } = 0$ . Such a collapsed representation carries no information about the observation and is useless for downstream planning or control.

Inverse dynamics regularization. To prevent this collapse, we introduce an inverse dynamics model $h _ { \psi } \overset { \cdot } { : } \mathcal { Z } \times \mathcal { Z } \overset { \cdot } {  } \mathcal { A }$ that predicts the action taken between two consecutive observations from their embeddings, $\hat { a } _ { t } = h _ { \psi } ( z _ { t } , z _ { t + 1 } )$ , trained with the mean-squared action loss

$$
\mathcal {L} _ {\text { inv }} = \mathbb {E} _ {(o _ {t}, a _ {t}, o _ {t + 1}) \sim \mathcal {D}} \left[ \| \hat {a} _ {t} - a _ {t} \| _ {2} ^ {2} \right]. \tag {3}
$$

Under squared loss, a collapsed encoder would give the inverse model identical inputs for every transition, so the best inverse predictor is a constant action, with risk $\mathbb { E } \Vert a _ { t } - \mathbb { E } [ a _ { t } ] \Vert _ { 2 } ^ { 2 }$ . Any reduction below this constant-predictor risk requires $\left( z _ { t } , z _ { t + 1 } \right)$ to preserve information that is predictive of $a _ { t }$ Such a nontrivial inverse prediction thus rules out a fully collapsed representation. In contrast to distributional priors such as SIGReg’s isotropic-Gaussian regularizer [14], this mechanism does not prescribe the geometry of the embedding space: it only requires that the encoder preserve whatever information about $a _ { t }$ is contained in the pair $\left( o _ { t } , o _ { t + 1 } \right)$ .

Joint training objective. We jointly optimize the encoder, forward model, and inverse model by minimizing

$$
\mathcal {L} = \mathcal {L} _ {\mathrm{fwd}} + \lambda \mathcal {L} _ {\mathrm{inv}}, \tag {4}
$$

where $\lambda > 0$ controls the strength of the inverse dynamics regularization. The forward loss updates $f _ { \theta }$ and $g _ { \phi } ;$ the inverse loss updates $f _ { \theta }$ and $h _ { \psi }$ . The encoder, therefore, receives gradients from both objectives, producing embeddings that are simultaneously predictive under forward dynamics and action-informative under inverse dynamics.

## 4 Learned latent structure

To build intuition for what SMWM learns and to assess which structure of the world the resulting representations reflect, we instantiate the method on a minimal testbed with known ground-truth state, intrinsic dimension, and action geometry.

A controlled testbed. We consider a dot world: a single-dot environment whose state is fully described by the 2D position $( x , y )$ of a colored dot rendered on a 64×64 white canvas; the observation $o _ { t } \in \mathbb { R } ^ { 3 \times 6 ^ { \bullet } \times 6 4 }$ is the rendered image. Actions $a _ { t } = ( \Delta x , \Delta y ) \in \mathcal { A } \subset \mathbb { R } ^ { 2 }$ displace the dot, so the dynamics are $s _ { t + 1 } = s _ { t } + a _ { t }$ in the state space. The dataset D is constructed by sampling, for each transition, a position and a random displacement. The encoder $f _ { \theta }$ is a small CNN with the latent dimension $d = 6 4$ , and the forward $g _ { \phi }$ and inverse $h _ { \psi }$ models are two-layer MLPs. The exact architecture, dataset, and training details are deferred to Sec. E.

Recovered intrinsic dimension. First, we show that $f _ { \theta }$ identifies the true 2D structure of the world from pixels and actions alone. Fig. 2 reports the PCA spectrum of the embeddings $\{ f _ { \boldsymbol { \theta } } ( \boldsymbol { o } ) \}$ over a uniform grid of dot positions. Two principal components carry essentially all the variance in the embedding space, and the spectrum drops sharply at the true intrinsic dimension $d _ { \mathrm { t r u e } } = 2 ;$ the remaining 62 directions are effectively collapsed. Projecting the embeddings onto the top two principal components reveals that the latent geometry is not only low-dimensional but also spatially faithful: the grid of world states maps to a near-square grid in latent space. A mild deformation occurs only near the boundary, which corresponds to out-of-distribution states. As a result, spatial neighbors in the observation space remain neighbors in the latent space. The encoder thus recovers an approximately 2-dimensional, topology-preserving representation of the state space without any ground-truth state supervision.

![](images/aac2c41228d09085a2749cda39237169d4b6fb86ac89e9d476d9de5018ce4023.jpg)

<details>
<summary>bar chart</summary>

| Principal component | Explained variance ratio |
| ------------------- | ------------------------ |
| 1                   | 0.45                     |
| 2                   | 0.45                     |
| 3                   | 0.02                     |
| 4                   | 0.01                     |
| 5                   | 0.01                     |
| 6                   | 0.005                    |
| 7                   | 0.002                    |
| 8                   | 0.001                    |
| 9                   | 0.0005                   |
| 10                  | 0.0002                   |
| 11                  | 0.0001                   |
| 12                  | 0.00005                  |
| 13                  | 0.00002                  |
| 14                  | 0.00001                  |
| 15                  | 0.000005                 |
</details>

![](images/6979f482b069a564715514d13ac18b8d38f5483cdd4b867cc16a1d54e202f456.jpg)

<details>
<summary>scatter plot</summary>

| x  | y  |
|----|----|
| 12 | 12 |
| 12 | 24 |
| 12 | 36 |
| 12 | 48 |
| 24 | 12 |
| 24 | 24 |
| 24 | 36 |
| 24 | 48 |
| 36 | 12 |
| 36 | 24 |
| 36 | 36 |
| 36 | 48 |
| 48 | 12 |
| 48 | 24 |
| 48 | 36 |
| 48 | 48 |
</details>

![](images/e55d307636ae509de4fbeb8a1800c3267721b8fbad157b857efafe4627913df6.jpg)

<details>
<summary>scatter plot</summary>

| PC1 | PC2 |
| --- | --- |
| -2.0 | 1.0 |
| -1.5 | 1.5 |
| -1.0 | 2.0 |
| -0.5 | 1.5 |
| 0.0 | 1.0 |
| 0.5 | 0.5 |
| 1.0 | 0.0 |
| 1.5 | -0.5 |
| 2.0 | -1.0 |
</details>

Figure 2: Dot world latent geometry. Left: PCA spectrum of the learned embeddings; the explainedvariance ratio drops sharply past the true intrinsic dimension $d _ { \mathrm { t r u e } } = 2$ (red dashed line). Center: grid of probe world states $( x , y )$ , color-coded by position. Right: the same probes embedded by $f _ { \theta }$ and projected onto the top two principal components. Despite no state supervision, the encoder recovers an effectively 2-dimensional representation inside a 64-dimensional ambient latent space. Moreover, the recovered representation is spatially faithful to the true state space. The latent grid appears rotated relative to $( x , y )$ , as expected: PC axes are identifiable only up to rotation and sign.

Forward model. The encoder and forward model should jointly satisfy a consistency condition: encoding the observation reached after taking action a should agree with first encoding the current observation and then applying $g _ { \phi } ( \cdot , a )$ . For ease of notation, we drop the indices and write $g _ { a } ( z ) : =$ $g _ { \phi } ( z , a )$ and $f$ for $f _ { \theta } ,$ , rendering the consistency condition

$$
f (a (o)) \stackrel {!} {=} g _ {a} (f (o)). \tag {5}
$$

In other words, the diagram on the left of Fig. 3 must commute [46, 47]. To test it, we encode an initial observation $o _ { 1 }$ , unroll g autoregressively along a 5-step action sequence $a _ { 1 } , \ldots a _ { 5 }$ , and compare the predicted embeddings against the encoded ground-truth embeddings $z _ { t } = f ( o _ { t } )$ . The predicted and encoded trajectories agree closely along the entire rollout (Fig. 3, right), illustrating that Eq. (5) approximately holds on the trained model. Each $a _ { t } \in$ A induces a corresponding displacement $\rho ( a _ { t } )$ in the relevant subspace of the latent space (with $\rho ( a )$ approximately independent of $z )$ , so the forward model behaves as $g _ { a } ( z ) \approx z + \rho ( a )$ . The trained model thus delivers, in addition to the state representation $f ,$ a representation of actions as latent translations $g _ { a }$ , without any term in $\mathcal { L }$ enforcing such a structure.

Equivariance and latent action composition. The above approximate equivariance condition is encouraged by the forward prediction loss. Denote the effect of action a on the observation o by $o \mapsto a ( o ) .$ .3 The forward loss minimizes, in our simplified notation,

$$
\left\| f (a (o)) - g _ {a} (f (o)) \right\| _ {2} ^ {2}.
$$

On the support of the training distribution and up to approximation error, this encourages

$$
f (a (o)) \approx g _ {a} (f (o)).
$$

This means that encoding after applying the intervention approximately agrees with first encoding and then applying the learned latent intervention $g _ { a }$ .4

This one-step equivariance has a compositional consequence. If actions form a semigroup under composition and equivariance holds exactly, then physical action composition must be represented by composition of the learned latent interventions, at least on the encoded data manifold:

![](images/f0f2608d58a7ab1f5bf1943ab44294eac4184c652583dcb9d699dadf6b72efbd.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
  O -->|a| O
  O -->|fθ| Z
  O -->|fθ| Z
  Z -->|gφ| Z
```
</details>

![](images/c0cb9098b36298e723e388f294244ecd27595ecabeb4f30340d223d054bc0ab3.jpg)

<details>
<summary>line chart</summary>

| Point | x  | y  |
|-------|----|----|
| o1    | 16 | 34 |
| o2    | 16 | 28 |
| o3    | 16 | 20 |
| o4    | 32 | 17 |
| o5    | 48 | 30 |
| o6    | 48 | 34 |
</details>

![](images/5481c2f7b43233616cf5e61e57513b221f7d38f8d4b03c5001b5eb438a8969b4.jpg)

<details>
<summary>scatter plot</summary>

| PC1  | PC2  | Type        |
|------|------|-------------|
| -1.0 | 0.0  | true z      |
| -1.0 | -1.0 | predicted z |
| -1.0 | 0.5  | true z      |
| -1.0 | 1.0  | predicted z |
| -1.0 | -1.0 | true z      |
| 0.5  | 1.0  | predicted z |
| 1.0  | 0.5  | true z      |
| 1.0  | -0.5 | true z      |
| 1.0  | 0.0  | predicted z |
| -1.0 | -1.0 | true z      |
| -1.0 | -1.0 | predicted z |
</details>

Figure 3: Encoder and forward model commute. Left: Equivariance that should be satisfied by the learned representation: $f \circ a = g _ { a } \circ f$ . Center: a 5-step trajectory in world-state space with actions $a _ { 1 } , \ldots , a _ { 5 }$ . Right: the corresponding rollout in latent space; predictions $\hat { z } _ { t }$ (filled red) obtained by autoregressive application of $g$ track the encoded ground-truth embeddings $z _ { t } = f ( o _ { t } )$ (open blue) along the entire rollout. Therefore, the encoder and forward model approximately commute, and world actions act as translations in latent space.

Theorem 1. Let A be a semigroup. $_ { I f f }$ is equivariant, $i . e . , f ( a ( o ) ) = g _ { a } ( f ( o ) )$ for all $a \in { \mathcal { A } }$ and $o \in { \mathcal { O } } ,$ , then $a \mapsto g _ { a }$ is a homomorphism on $f ( \mathcal O )$ . That is, for all $a _ { 1 } , a _ { 2 } \in \mathcal { A } a n d z \in f ( \mathcal { O } )$ ,

$$
g _ {a _ {2} \circ a _ {1}} (z) = g _ {a _ {2}} (g _ {a _ {1}} (z)). \tag {6}
$$

We prove theorem 1 in Sec. G. In the learned model, this identity should be interpreted approximately.

Equivariance alone, however, admits degenerate solutions, including collapse. Useful representations, therefore, additionally require some form of faithfulness of g on $f ( \mathcal { O } )$ , i.e., every $a \in { \mathcal { A } }$ that acts nontrivially on $\mathcal { O }$ must also act nontrivially on $f ( { \mathcal { O } } )$ . More strongly, we can require that the latent action is identifiable from observations: for every pair $( z , z ^ { \prime } ) \in \bar { f ( \mathcal { O } ) } \times f ( \mathcal { O } )$ with $z ^ { \prime } = g _ { a } ( z )$ , the action a is recoverable from $( z , z ^ { \prime } )$ . We achieve this by training an inverse model that predicts a from $( z , z ^ { \prime } )$ .

Beyond a homomorphism into latent transformations under composition, the experiments in Fig. 3 suggest that in our case the homomorphism may (approximately) ${ \bf g 0 }$ into the additive group, i.e., the learned interventions appear to act approximately as translations,

$$
g _ {a} (z) \approx z + \rho (a),
$$

with $\rho ( a )$ approximately independent of z. It is intriguing to ask where it may stem from. The inverse-dynamics loss provides a plausible bias toward this translation-like regime: a simple way to make $a _ { t }$ recoverable from $\left( z _ { t } , z _ { t + 1 } \right)$ is to encode the action in the displacement vector

$$
z _ {t + 1} - z _ {t} \approx \rho (a _ {t}). \tag {7}
$$

If $\rho$ is injective on the action support, then the inverse h can decode the action by applying an approximate inverse of $\rho$ to this displacement. This mechanism does not enforce additivity, but it renders an additive, translation-like representation a low-complexity solution.

Controllable degrees of freedom. The 2D dot world establishes that the learned representation has the right intrinsic dimension when it equals the action dimension. To stress-test this property, we vary the controllable degrees of freedom and add uncontrollable distractors. Fig. 4 reports four configurations. Independent contains two dots controlled by independent displacements, giving a 4D state and a 4D action. Coupled contains two dots that move together under a single 2D displacement. Distractor contains one controlled dot together with a second dot that moves randomly on its own and is not controlled by the action. Combined merges all three: two independent dots, one coupled pair, and one distractor, with a 6D action vector.

In every case, the number of significant principal components matches the controllable dimension of the system, with the remaining directions effectively collapsed. One may notice that the spectra within the controllable subspace are not exactly uniform, even though the state space is uniformly covered under uniform coordinate sampling. However, this is expected: the encoder is free to rescale individual dimensions and concentrate variance unevenly, so what matters is the rank of the embedding, not the uniformity of its spectrum. Of the four configurations, the distractor case is particularly informative: even though the random dot’s position varies in the observation, the encoder ignores it. This is the expected outcome of Eq. (4): a distractor’s state is neither needed to recover the action from a transition (it is independent of $a _ { t } )$ nor predictable from past latent state under $g _ { \phi }$ (since it evolves stochastically), so encoding it would only inflate ${ \mathcal L } _ { \mathrm { f w d } }$ without reducing ${ \mathcal { L } } _ { \mathrm { i n v } }$ .

![](images/99a25a3f0916dabee3610953b7bd66189fa80011c457a983e6ea7b213b694502.jpg)  
Figure 4: Effective latent dimension tracks controllable degrees of freedom. Top row: four dot-world configurations with controllable dimensions 4, 2, 2, and 6; in Distractor and Combined, the wavy-arrowed dot moves randomly and is not controlled by the action. Bottom row: PCA spectra of the corresponding learned embeddings, with the true intrinsic dimension marked by the red dashed line. The encoder allocates significant variance to exactly the controllable degrees of freedom and filters out uncontrollable distractors, correctly recovering a minimal necessary subspace inside the higher-dimensional ambient latent space.

Visualizing control-dependent perception. As a complementary visualization, we ask whether the model “sees” the same object differently depending on what is controllable. To make this explicit, we introduce an additional toy environment in Sec. F: an asymmetric 2D triangular sprite whose pose is $( x , y , \theta )$ . The sprite can translate and rotate, but we vary the action repertoire by exposing only some of the pose changes as actions, while the remaining degrees of freedom continue to move randomly. After training, we freeze the encoder and train a post-hoc decoder from $z _ { t }$ to $o _ { t }$ . The reconstructions show that the representation preserves the control-relevant pose variables and averages out the uncontrolled ones: for example, when agent rotations are uncontrolled, the decoded triangle becomes an orientation-averaged blob, whereas full $x / y / \theta$ control yields a sharp oriented reconstruction.

Together, these experiments show that SMWM learns representations that are low-dimensional, spatially faithful, and structure-preserving with respect to the action-induced dynamics.

## 5 Planning in latent space

The latent structure recovered above is only useful insofar as it carries over to richer environments and supports actionable downstream control.

Setup. We perform goal-conditioned trajectory optimization directly in the learned latent space, following the planning setup of LeWM [14]. Given a current observation $o _ { 1 }$ and a goal observation $o _ { g } .$ , we encode both with the frozen encoder, $z _ { 1 } = f _ { \theta } ( o _ { 1 } )$ and $z _ { g } = f _ { \theta } ( o _ { g } )$ , and roll the forward model out autoregressively from $\hat { z } _ { 1 } = z _ { 1 } , \hat { z } _ { t + 1 } = g _ { \phi } ( \hat { z } _ { t } , a _ { t } )$ ), to predict the latent trajectory induced by a candidate action sequence $a _ { 1 : H }$ over a planning horizon $\dot { H }$ . Each candidate is scored by the

![](images/a038620d071186922798e6746020b00e0ddb437cc3f0756da6393ad37246a686.jpg)  
Figure 5: Planning success across environments. Top: the four evaluation environments— TwoRoom (2D navigation), Reacher (continuous control), Push-T (2D contact-rich manipulation), and OGBench-Cube (3D tabletop manipulation). Bottom: goal-conditioned planning success rate (mean and standard error over five seeds) under a fixed budget of 50 environment steps and a goal placed 25 steps ahead of the initial state. SMWM matches the SIGReg baseline on the three 2D tasks and clearly outperforms it on OGBench-Cube. Both regularized variants dominate the Forward-only ablation and the Random-action baseline, confirming that an effective anti-collapse mechanism is necessary for usable planning.

terminal goal-matching cost

$$
\mathcal {C} (a _ {1: H}) = \| \hat {z} _ {H + 1} - z _ {g} \| _ {2} ^ {2}, \tag {8}
$$

which we minimize with the Cross-Entropy Method (CEM) [48]: at each iteration, CEM samples action sequences from a Gaussian, scores them via latent rollouts under $g _ { \phi } .$ , and refits the sampling distribution to the top-scoring fraction. Because autoregressive rollouts accumulate prediction error as H grows, we wrap CEM in a receding-horizon model-predictive control (MPC) loop, executing only the first $K \leq H$ optimized actions before re-encoding the resulting observation and replanning. Throughout, fθ and $g _ { \phi }$ remain frozen; only the action sequence is optimized.

Environments. We evaluate on four tasks shown in Fig. 5: TwoRoom (2D navigation), Reacher (continuous control), Push-T (2D contact-rich manipulation), and OGBench-Cube (3D tabletop manipulation). Datasets and environment details are deferred to Sec. B.

Baselines. We compare against three baselines. SIGReg [14] is our main head-to-head: it shares the same encoder and predictor but replaces ${ \mathcal { L } } _ { \mathrm { i n v } }$ with a Gaussian-matching regularizer on the latent embeddings. Forward-only (λ = 0) ablates the regularizer entirely, isolating the contribution of our anti-collapse mechanism. Random samples actions uniformly from the action space and serves as a floor reference. Full implementation details are in the appendix.

Architecture. We adopt LeWM’s encoder and predictor [14]: the encoder is a ViT-Tiny (∼5M parameters) whose final CLS token is projected to $z \in \mathbb { R } ^ { d }$ with $d = 1 9 2$ , and $g _ { \phi }$ is a small transformer (∼10M parameters) that predicts $\hat { z } _ { t + 1 }$ from a single $( z _ { t } , a _ { t } )$ pair, with $a _ { t }$ injected via AdaLN-zero. Our only architectural addition is the inverse model $h _ { \psi } \colon$ a 2-layer MLP of width 256 mapping $[ z _ { t } ; z _ { t + 1 } ] \in \mathbb { R } ^ { 2 d }$ to $\hat { a } _ { t } \in \mathbb { R } ^ { m }$ . Details are provided in Sec. A.

Planning performance. All methods are evaluated with a fixed budget of 50 environment steps per episode and goals placed 25 steps ahead of the initial state. Both regularized variants substantially outperform Forward-only and Random across all four environments (Fig. 5). SMWM approximately matches SIGReg on the three 2D tasks—TwoRoom, Reacher, and Push-T—and clearly outperforms it on OGBench-Cube, the only environment with a 3D contact-rich manipulation, where SMWM retains 84% success while SIGReg drops to 59%.

Table 1: Probing the physical state. Held-out $R ^ { 2 }$ for linear and two-layer MLP probes regressing ground-truth physical quantities from frozen embeddings; the best score per row and probe type is in bold. Both regularized methods (SMWM and LeWM/SIGReg) recover the underlying physical state near-perfectly under MLP probes, with the exception of Push-T block pose.

<table><tr><td rowspan="2">Environment</td><td rowspan="2">Quantity</td><td colspan="3">Linear probe</td><td colspan="3">MLP probe</td></tr><tr><td>SMWM</td><td>SIGReg</td><td>Fwd.-only</td><td>SMWM</td><td>SIGReg</td><td>Fwd.-only</td></tr><tr><td>TwoRoom</td><td>agent position</td><td>1.000</td><td>0.996</td><td>0.832</td><td>1.000</td><td>1.000</td><td>0.991</td></tr><tr><td>Reacher</td><td>joint angles</td><td>0.946</td><td>0.999</td><td>0.647</td><td>0.995</td><td>1.000</td><td>0.962</td></tr><tr><td rowspan="3">Push-T</td><td>agent position</td><td>0.993</td><td>0.954</td><td>0.293</td><td>0.988</td><td>0.986</td><td>0.433</td></tr><tr><td>block position</td><td>0.946</td><td>0.973</td><td>0.352</td><td>0.981</td><td>0.996</td><td>0.623</td></tr><tr><td>block orientation</td><td>0.664</td><td>0.931</td><td>0.202</td><td>0.882</td><td>0.987</td><td>0.491</td></tr><tr><td rowspan="3">Cube</td><td>cube position</td><td>0.998</td><td>0.975</td><td>0.706</td><td>0.998</td><td>0.993</td><td>0.854</td></tr><tr><td>gripper position</td><td>0.999</td><td>0.986</td><td>0.902</td><td>0.999</td><td>0.995</td><td>0.979</td></tr><tr><td>gripper opening</td><td>0.989</td><td>0.961</td><td>0.269</td><td>0.992</td><td>0.983</td><td>0.427</td></tr></table>

Probing physical state. To assess what physical content the embeddings retain, we train linear and two-layer MLP probes on frozen embeddings to regress the ground-truth state of each environment; held-out $R ^ { 2 }$ values are reported in Tab. 1.

Both regularized methods recover the underlying state near-perfectly under MLP probes, and their linear-probe scores stay near saturation across most quantities, with SMWM holding a consistent linear advantage on the Push-T agent position and on all three Cube quantities, and SIGReg holding a corresponding advantage on the Reacher joint angles and the Push-T block pose. The Forward-only ablation is markedly poorer, particularly under linear probes, but is rarely fully degenerate—some residual variance survives even in a partially collapsed encoder. Both regularized representations therefore make the underlying physical state recoverable.

Latent geometry. We next turn to the geometry that organizes them. Fig. 6 reports, for each environment, the PCA spectrum of held-out embeddings (top), the distribution of a representative ground-truth quantity in physical state space (middle), and the embeddings projected onto a 2- or 3-dimensional PC subspace (bottom). In every environment, the spectrum drops sharply, confirming that the embeddings concentrate on an effectively low-dimensional subspace of the 192-dimensional ambient latent space. The corresponding SIGReg visualizations are provided in Sec. D; while SIGReg also prevents collapse, its latent geometry appears less consistently organized by the controllable coordinates.

In TwoRoom, the encoder behaves much as in the dot world: $\mathrm { P C _ { 1 } }$ and $\mathrm { P C _ { 2 } }$ recover a spatially faithful Cartesian map of the agent, and the wall—a region unreachable to the agent—leaves an empty band in latent space, providing further evidence that the recovered representation is spatially faithful. In Push-T, the agent position is again encoded in $\left( \mathrm { P C } _ { 1 } , \mathrm { P C } _ { 2 } \right)$ , but the spectrum has heavier tails: the embedding must additionally accommodate the position and orientation of the manipulable block (cf. Tab. 1). Reacher reveals a richer geometry. Its state space is the torus $\mathbb { T } ^ { 2 }$ of two periodic angles $( \theta _ { 1 } , \theta _ { 2 } )$ , which cannot be continuously encoded into $\mathbb { R } ^ { 2 ^ { \bullet } }$ due to a discontinuity at $\bar { \theta } = 0 \sim 2 \pi$ . A natural representation is the product-of-circles embedding $( \theta _ { 1 } , \theta _ { 2 } ) \mapsto ( \cos { \theta _ { 1 } } , \sin { \theta _ { 1 } } , \cos { \theta _ { 2 } } , \sin { \theta _ { 2 } } )$ , which lifts $\mathbb { T } ^ { 2 }$ into $\mathbb { R } ^ { 4 }$ . Its coordinate 3D projections are circular cylinders, which is exactly the signature we observe in the encoded embeddings: the spectrum exposes four appreciable components, and every 3D PC subspace in Fig. 6 (bottom) shows a clear cylindrical shape. The two angles, however, spread across all four components, which is consistent with arbitrary linear mixing of the product-of-circles coordinates. Finally, in Cube the gripper position is recovered in $\left( \mathrm { P C } _ { 1 } , \mathrm { P C } _ { 3 } \right)$ ; $\mathrm { \bar { P } C _ { 2 } }$ and the heavier tail of the spectrum carry the cube pose and gripper opening (cf. Tab. 1).

![](images/9a09aa6d435b50c1d2acccc7a3f3b7c86890e8052aedc3961fbff9bdc699b9ab.jpg)  
Figure 6: Latent geometry of SMWM embeddings. For each environment we show the PCA spectrum of held-out embeddings (top), the distribution of a representative ground-truth quantity in physical state space (middle), and the embeddings projected onto a 2- or 3-dimensional PC subspace, color-coded by the same physical quantity (bottom). The dashed red lines mark the action dimension. Across all four environments, the embeddings concentrate on a low-dimensional subspace whose geometry mirrors the controllable state space: Cartesian coordinates (TwoRoom and Push-T agent, Cube gripper) appear as approximately linear directions, while the periodic joint angles in Reacher are encoded as a flat 2-torus, visible as a cylindrical structure in any 3D PC projection.

Takeaway. Across all four environments, SMWM recovers compact latent spaces whose dimension and geometry mirror controllable structure: Cartesian degrees of freedom form approximately linear directions, while periodic variables form circles. The structure carries over to 2D and 3D control, with the largest planning gain over SIGReg appearing on the 3D environment.

## 6 Discussion

We showed that inverse dynamics regularization addresses two coupled issues simultaneously. First, as an anti-collapse mechanism, a single-step inverse dynamics head suffices to stabilize end-to-end training of a JEPA-style latent world model from pixels, while introducing only one hyperparameter and imposing no distributional prior on the embedding space. Second, the same mechanism biases the encoder toward a compact, action-sufficient representation due to the joint pressure from the forward and inverse dynamics loss terms. Empirically, this pressure yields representations that track the controllable degrees of freedom of the environment, preserve both Cartesian and periodic state geometry, and filter out uncontrollable distractors, although no term in the objective explicitly enforces this structure. The learned representation further supports latent planning that matches or outperforms a strong SIGReg-regularized baseline across four control tasks, with the largest margin on the most complex 3D manipulation environment with a five-dimensional action space. Taken together, these results support the view of the sensorimotor world model as an effective and simple instantiation of a “perception for action” bias in latent world-model learning.

Limitations and future work. Our analysis assumes that actions are recoverable from consecutive observations, which may fail when distinct actions induce the same visible change. Relatedly, because the encoder currently maps individual frames to latent states, it will not capture quantities that are not identifiable from a single frame, such as velocities, even when these quantities would improve forward prediction; incorporating short observation histories into the encoder or predictor is a natural extension. Furthermore, the action-aligned bias, while desirable, may discard information needed for downstream tasks that depend on aspects of the environment irrelevant to the training actions (we consider this a feature rather than a bug). In addition, biased behavior policies may create action-correlated but uncontrollable distractors, which could be captured by the encoder. Finally, our planning experiments inherit the usual limitations of offline world models: limited dataset coverage constrains the reachable plans, long-horizon rollouts remain vulnerable to compounding model error, and our empirical validation is restricted to moderate-scale simulated control tasks. Future work should study more robust data regimes, multi-frame, and multi-step inverse objectives. While we use inverse loss only as a regularizer, inverse models themselves could be integrated into latent or hierarchical planners to reduce reliance on long autoregressive forward rollouts.

## References

[1] David Ha and Jürgen Schmidhuber. World models. arXiv preprint arXiv:1803.10122, 2018.  
[2] Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to control: Learning behaviors by latent imagination. In ICLR, 2020.  
[3] Bernhard Schölkopf. Causality for machine learning. 2019. URL http://arxiv.org/abs/ 1911.10500. Published in: Probabilistic and Causal Inference: The Works of Judea Pearl.  
[4] Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse domains through world models. arXiv preprint arXiv:2301.04104, 2023.  
[5] Nicklas Hansen, Xiaolong Wang, and Hao Su. Temporal difference learning for model predictive control. In ICML, 2022.  
[6] Nicklas Hansen, Hao Su, and Xiaolong Wang. Td-mpc2: Scalable, robust world models for continuous control. ICLR, 2024.  
[7] Yann LeCun. A path towards autonomous machine intelligence. OpenReview, 2022.  
[8] Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, and Nicolas Ballas. Self-supervised learning from images with a jointembedding predictive architecture. In CVPR, 2023.  
[9] Adrien Bardes, Quentin Garrido, Jean Ponce, Xinlei Chen, Michael Rabbat, Yann LeCun, Mahmoud Assran, and Nicolas Ballas. V-jepa: Latent video prediction for visual representation learning. arXiv preprint arXiv:2402.04252, 2024.  
[10] Jean-Bastien Grill, Florian Strub, Florent Altché, Corentin Tallec, Pierre Richemond, Elena Buchatskaya, Carl Doersch, Bernardo Avila Pires, Zhaohan Guo, Mohammad Gheshlaghi Azar, et al. Bootstrap your own latent: A new approach to self-supervised learning. In NeurIPS, 2020.  
[11] Adrien Bardes, Jean Ponce, and Yann LeCun. Vicreg: Variance-invariance-covariance regularization for self-supervised learning. In ICLR, 2022.  
[12] Gaoyue Zhou, Hengkai Pan, Yann LeCun, and Lerrel Pinto. DINO-WM: World models on pre-trained visual features enable zero-shot planning. In Proceedings of the 42nd International Conference on Machine Learning (ICML 2025), volume 267 of Proceedings of Machine Learning Research, pages 79115–79135. PMLR, 2025. URL https://proceedings.mlr. press/v267/zhou25t.html.  
[13] Vlad Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, and Yann LeCun. Learning from reward-free offline data: A case for planning with latent dynamics models. In Advances in Neural Information Processing Systems 38 (NeurIPS 2025), 2025. URL https://neurips.cc/virtual/2025/poster/116649.  
[14] Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, and Randall Balestriero. LeWorld-Model: Stable end-to-end joint-embedding predictive architecture from pixels. arXiv preprint arXiv:2603.19312, 2026. URL https://arxiv.org/abs/2603.19312.  
[15] Randall Balestriero and Yann LeCun. LeJEPA: Provable and scalable self-supervised learning without the heuristics. arXiv preprint arXiv:2511.08544, 2025. URL https://arxiv.org/ abs/2511.08544.  
[16] Mahmoud Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning. arXiv preprint arXiv:2506.09985, 2025. URL https://arxiv.org/abs/2506.09985.  
[17] B. Schölkopf, F. Locatello, S. Bauer, N. R. Ke, N. Kalchbrenner, A. Goyal, and Y. Bengio. Toward causal representation learning. Proceedings of the IEEE, 109(5):612–634, 2021.  
[18] Melvyn A. Goodale and A. David Milner. Separate visual pathways for perception and action. Trends in Neurosciences, 15(1):20–25, 1992.  
[19] Wolfgang Prinz. A common coding approach to perception and action. In Odmar Neumann and Wolfgang Prinz, editors, Relationships Between Perception and Action: Current Approaches, pages 167–201. Springer, Berlin, 1990.  
[20] James J. Gibson. The Ecological Approach to Visual Perception. Houghton Mifflin, 1979.  
[21] Jakob von Uexküll. Streifzüge durch die Umwelten von Tieren und Menschen: Ein Bilderbuch unsichtbarer Welten, volume 21 of Verständliche Wissenschaft. Verlag von Julius Springer, Berlin, 1934.  
[22] Francisco J. Varela, Eleanor Rosch, and Evan Thompson. The Embodied Mind: Cognitive Science and Human Experience. MIT Press, Cambridge, MA, 1991.  
[23] J. Kevin O’Regan and Alva Noë. A sensorimotor account of vision and visual consciousness. Behavioral and Brain Sciences, 24(5):939–1031, 2001.  
[24] Danijar Hafner, Timothy Lillicrap, Mohammad Norouzi, and Jimmy Ba. Mastering Atari with discrete world models. In ICLR, 2021.  
[25] Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning latent dynamics for planning from pixels. In ICML, 2019.  
[26] Aaron van den Oord, Yazhe Li, and Oriol Vinyals. Representation learning with contrastive predictive coding. In arXiv preprint arXiv:1807.03748, 2018.  
[27] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. A simple framework for contrastive learning of visual representations. In ICML, 2020.  
[28] Xinlei Chen and Kaiming He. Exploring simple siamese representation learning. In CVPR, 2021.  
[29] Jure Zbontar, Li Jing, Ishan Misra, Yann LeCun, and Stéphane Deny. Barlow twins: Selfsupervised learning via redundancy reduction. In ICML, 2021.  
[30] Deepak Pathak, Pulkit Agrawal, Alexei A Efros, and Trevor Darrell. Curiosity-driven exploration by self-supervised prediction. In ICML, 2017.  
[31] Yonathan Efroni, Dipendra Misra, Akshay Krishnamurthy, Alekh Agarwal, and John Langford. Provably filtering exogenous distractors using multistep inverse dynamics. In International Conference on Learning Representations (ICLR 2022), 2022. URL https://openreview. net/forum?id=RQLLzMCefQu.  
[32] Alex Lamb, Riashat Islam, Yonathan Efroni, Aniket Rajiv Didolkar, Dipendra Misra, Dylan J Foster, Lekan P Molu, Rajan Chari, Akshay Krishnamurthy, and John Langford. Guaranteed discovery of control-endogenous latent states with multi-step inverse models. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/ forum?id=TNocbXm5MZ.  
[33] Riashat Islam, Manan Tomar, Alex Lamb, Yonathan Efroni, Hongyu Zang, Aniket Didolkar, Dipendra Misra, Xin Li, Harm van Seijen, Remi Tachet des Combes, and John Langford. Agent-controller representations: Principled offline RL with rich exogenous information. arXiv preprint arXiv:2211.00164, 2022.  
[34] Zakaria Mhammedi, Dylan J. Foster, and Alexander Rakhlin. Representation learning with multi-step inverse kinematics: An efficient and optimal approach to rich-observation RL. In Proceedings of the 40th International Conference on Machine Learning (ICML 2023), volume 202 of Proceedings of Machine Learning Research, pages 24659–24700. PMLR, 2023. URL https://proceedings.mlr.press/v202/mhammedi23a.html. Oral presentation.  
[35] Yuci Han and Alper Yilmaz. Enhancing policy learning with world-action model. arXiv preprint arXiv:2603.28955, 2026. URL https://arxiv.org/abs/2603.28955.  
[36] Basile Terver, Randall Balestriero, Megi Dervishi, David Fan, Quentin Garrido, Tushar Nagarajan, Koustuv Sinha, Wancong Zhang, Mike Rabbat, Yann LeCun, and Amir Bar. A lightweight library for energy-based joint-embedding predictive architectures. arXiv preprint arXiv:2602.03604, 2026. URL https://arxiv.org/abs/2602.03604.  
[37] Jiacan Yu, Siyi Chen, Mingrui Liu, Nono Horiuchi, Vladimir Braverman, Zicheng Xu, Dan Haramati, and Randall Balestriero. Why and how auxiliary tasks improve JEPA representations. In UniReps: 3rd Edition of the Workshop on Unifying Representations in Neural Models, 2025. URL https://openreview.net/forum?id=ZVx4SdKhlc.  
[38] Dominik Schmidt and Minqi Jiang. Learning to act without actions. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=rvUq3cxpDF.  
[39] Zichen Jeff Cui, Hengkai Pan, Aadhithya Iyer, Siddhant Haldar, and Lerrel Pinto. Dynamo: Indomain dynamics pretraining for visuo-motor control. In The Thirty-eighth Annual Conference on Neural Information Processing Systems, 2024. URL https://openreview.net/forum? id=vUrOuc6NR3.  
[40] Younggyo Seo, Kimin Lee, Stephen L. James, and Pieter Abbeel. Reinforcement learning with action-free pre-training from videos. In Proceedings of the 39th International Conference on Machine Learning (ICML 2022), volume 162 of Proceedings of Machine Learning Research, pages 19561–19579. PMLR, 2022. URL https://proceedings.mlr.press/v162/ seo22a.html.  
[41] Michael Laskin, Aravind Srinivas, and Pieter Abbeel. Curl: Contrastive unsupervised representations for reinforcement learning. In International conference on machine learning, pages 5639–5650. PMLR, 2020.  
[42] Ilya Kostrikov, Denis Yarats, and Rob Fergus. Image augmentation is all you need: Regularizing deep reinforcement learning from pixels. arXiv preprint arXiv:2004.13649, 2020.  
[43] Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. Reinforcement learning with prototypical representations. In International Conference on Machine Learning, pages 11920–11931. PMLR, 2021.  
[44] Norm Ferns, Prakash Panangaden, and Doina Precup. Metrics for finite Markov decision processes. In UAI, 2004.  
[45] Pablo Samuel Castro. Scalable methods for computing state similarity in deterministic Markov decision processes. In AAAI, 2020.  
[46] P. K. Rubenstein\*, S. Weichwald\*, S. Bongers, J. M. Mooij, D. Janzing, M. Grosse-Wentrup, and B. Schölkopf. Causal consistency of structural equation models. In Proceedings of the Thirty-Third Conference on Uncertainty in Artificial Intelligence (UAI), 2017. URL http: //auai.org/uai2017/proceedings/papers/11.pdf.  
[47] Heinrich Hertz. The Principles of Mechanics Presented in a New Form. Macmillan, London, 1899.  
[48] Reuven Y Rubinstein. Optimization of computer simulation models with rare events. European Journal of Operational Research, 99(1):89–112, 1997.  
[49] Lucas Maes, Quentin Le Lidec, Dan Haramati, Nassim Massaudi, Damien Scieur, Yann LeCun, and Randall Balestriero. stable-worldmodel-v1: Reproducible world modeling research and evaluation. arXiv preprint arXiv:2602.08968, 2026.  
[50] Vlad Sobal, Wancong Zhang, Kyunghyun Cho, Randall Balestriero, Tim G. J. Rudner, and Yann LeCun. Stress-testing offline reward-free reinforcement learning: A case for planning with latent dynamics models. In 7th Robot Learning Workshop: Towards Robots with Human-Level Abilities, 2025. URL https://openreview.net/forum?id=jON7H6A9UU.  
[51] Seohong Park, Kevin Frans, Benjamin Eysenbach, and Sergey Levine. OGBench: Benchmarking offline goal-conditioned RL. In The Thirteenth International Conference on Learning Representations, 2025. URL https://openreview.net/forum?id=M992mjgKzI.  
[52] Yuval Tassa, Yotam Doron, Alistair Muldal, Tom Erez, Yazhe Li, Diego de Las Casas, David Budden, Abbas Abdolmaleki, Josh Merel, Andrew Lefrancq, et al. Deepmind control suite. arXiv preprint arXiv:1801.00690, 2018.  
[53] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, and Sergey Levine. Soft actor-critic: Offpolicy maximum entropy deep reinforcement learning with a stochastic actor. In International conference on machine learning, pages 1861–1870. PMLR, 2018.  
[54] Balaraman Ravindran and Andrew G. Barto. Smdp homomorphisms: An algebraic approach to abstraction in semi-markov decision processes. In International Joint Conference on Artificial Intelligence (IJCAI), pages 1011–1016, 2003.  
[55] H. Keurti, H.-R. Pan, M. Besserve, B. F. Grewe, and B. Schölkopf. Homomorphism AutoEncoder — learning group structured representations from observed transitions. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 16190–16215. PMLR, 2023. URL https: //proceedings.mlr.press/v202/keurti23a.html.  
[56] Kiho Park, Yo Joong Choe, and Victor Veitch. The linear representation hypothesis and the geometry of large language models. In Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 39643–39666. PMLR, 2024.  
[57] Alexander Levine, Peter Stone, and Amy Zhang. Multistep inverse is not all you need. Reinforcement Learning Journal, 2:884–925, 2024. URL https://rlj.cs.umass.edu/2024/ papers/Paper117.html. Presented at the Reinforcement Learning Conference (RLC 2024).  
[58] David Brandfonbrener, Ofir Nachum, and Joan Bruna. Inverse dynamics pretraining learns good representations for multitask imitation. In Advances in Neural Information Processing Systems 36 (NeurIPS 2023), 2023. URL https://proceedings.neurips.cc/paper\_files/ paper/2023/hash/d36dfcdb14473a8526111c221660f2ab-Abstract-Conference. html.  
[59] Max Schwarzer, Ankesh Anand, Rishab Goel, R Devon Hjelm, Aaron Courville, and Philip Bachman. Data-efficient reinforcement learning with self-predictive representations. In ICLR, 2021.

## A Implementation details

## A.1 Training objective

Alg. 1 gives PyTorch-style pseudocode for the mini-batch objective used to train SMWM. The encoder receives gradients from both the forward prediction loss and the inverse dynamics loss, which is what prevents representation collapse.

Algorithm 1 Mini-batch loss for SMWM training.  
```python
def loss(o_t, a_t, o_tp1, lambda_inv=10.0):
    """
    o_t, o_tp1: (B, C, H, W) consecutive pixel observations
    a_t:    (B, A) action taken between them
    lambda_inv: (float) inverse dynamics loss weight
    """
    z_t = encoder(o_t)    # (B, D)
    z_tp1 = encoder(o_tp1)    # (B, D)

    z_hat = fwd_model(z_t, a_t)    # predict next embedding
    a_hat = inv_model(z_t, z_tp1)    # predict intervening action

    loss_fwd = F.mse_loss(z_hat, z_tp1)    # forward dynamics loss
    loss_inv = F.mse_loss(a_hat, a_t)    # inverse dynamics loss

    return loss_fwd + lambda_inv * loss_inv
```

## A.2 Architecture

We adopt the encoder and forward model of LeWorldModel [14] essentially unchanged, and add a small inverse head $h _ { \psi }$ . Inputs are $2 2 4 \times 2 2 4$ RGB images.

Encoder $f _ { \theta } .$ A Vision Transformer Tiny (ViT-Tiny) with patch size 14, instantiated from the Hugging Face library. The final CLS token is projected $\boldsymbol { \mathrm { t o ~ } } z \in \mathbb { R } ^ { d }$ with $d = 1 9 2$ ; the encoder has ≈ 5M parameters.

Forward model $g _ { \phi } .$ . A ViT-S backbone with learned positional embeddings and causal masking over the observation history, matching the LeWM predictor; the action $a _ { t }$ is injected via AdaLN-zero. However, in contrast to $\mathrm { L e W M } .$ , we set the history length to 1 on every environment, so $g _ { \phi }$ predicts $\hat { z } _ { t + 1 }$ from a single $\left( \boldsymbol { z } _ { t } , \boldsymbol { a } _ { t } \right)$ pair, as described in Sec. 3. LeWM instead uses history 3 on Push-T and OGBench-Cube. The forward model has ≈ 10M parameters.

Inverse model $h _ { \psi }$ . Our only architectural addition. A 2-layer MLP of width 256 with ReLU activations, mapping $[ z _ { t } ; z _ { t + 1 } ] \in \mathbb { R } ^ { 2 d } : 0 \hat { a } _ { t } \in \mathbb { R } ^ { m }$ , without output activation. Approximately 0.1M parameters.

## A.3 Optimization details

All world models are trained end-to-end with the objective in Eq. (4). We use the same optimization settings for SMWM, the forward-only ablation, and the SIGReg baseline; only the regularization term is changed. For SMWM, the inverse-dynamics weight λ is environment-specific. The values are listed in Tab. 2.

The remaining training hyperparameters are shared across environments unless explicitly noted in Tab. 3. We train on two-frame snippets with a single-step forward target. Since datasets are loaded with frameskip 5, the action encoder and inverse head receive the concatenated action over the skipped interval. The SIGReg comparison uses the same optimizer and architecture, with $\lambda = 0$ and SIGReg weight 0.09.

Table 2: Inverse-dynamics weights. Values of λ in Lfwd + λLinv.

<table><tr><td>Environment</td><td>λ</td></tr><tr><td>TwoRoom</td><td>0.1</td></tr><tr><td>Reacher</td><td>5</td></tr><tr><td>Push-T</td><td>30</td></tr><tr><td>OGBench-Cube</td><td>1</td></tr></table>

Table 3: Optimization hyperparameters.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $10^{-4}$ </td></tr><tr><td>Weight decay</td><td> $10^{-3}$ </td></tr><tr><td>Batch size</td><td>256</td></tr><tr><td>Training epochs</td><td>10</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr></table>

Choosing λ. Fig. 7 shows that λ requires some environment-level tuning. For example, $\lambda = 0 . 1$ is optimal for TwoRoom but causes Reacher to collapse to near-random planning performance. The final values in Tab. 2 were chosen from this broad sweep. The sweep contains one trained model per λ value, so the shaded bands show binomial standard error over the 100 evaluation episodes rather than variation across training seeds.

![](images/8d56ef0f2cd1ede3bfe6117a8d28d24c517218a1b8a09a9328e1e8fdb41701d1.jpg)

<details>
<summary>line chart</summary>

| x    | Success rate |
| ---- | ------------ |
| 0    | 0.3          |
| 0.1  | 1.0          |
| 1    | 0.85         |
| 10   | 0.9          |
| 100  | 0.85         |
</details>

![](images/889e8429c247563d28d60ea3c694629fc8701d59a84dc0b39769a5a1b5d01d06.jpg)

<details>
<summary>line chart</summary>

| x    | y     |
| ---- | ----- |
| 0    | 0.0   |
| 0.1  | 0.0   |
| 1    | 0.0   |
| 10   | 0.0   |
| 100  | 0.0   |
</details>

![](images/12516e80c709acc20317d431f0c201311926e0c4759c1fff6bcb0f377f3ab4fd.jpg)

<details>
<summary>line chart</summary>

| x    | y     |
| ---- | ----- |
| 0    | 0.0   |
| 0.1  | 0.5   |
| 1    | 0.7   |
| 10   | 0.9   |
| 100  | 0.8   |
</details>

![](images/d30fd16dfbeac5ad0749c00218ac349bb7c08b16bb4060f21a3e4ac23aaad7ad.jpg)

<details>
<summary>line chart</summary>

| x    | y     |
| ---- | ----- |
| 0    | 0.5   |
| 0.1  | 1.0   |
| 1    | 1.0   |
| 10   | 1.0   |
| 100  | 1.0   |
</details>

Inverse weight λ  
Figure 7: Sensitivity to inverse-dynamics weight. Goal-conditioned planning success rate as a function of the inverse-dynamics loss weight λ at goal offset 25. Each panel corresponds to one environment, and the red dotted line marks the value used for the main paper experiments.

## A.4 Planning protocol

For each evaluation episode, the policy receives the current observation $o _ { t }$ and a goal observation $o _ { g } ,$ encodes them as $z _ { t } = f _ { \theta } ( o _ { t } )$ and $z _ { g } = f _ { \theta } ( o _ { g } )$ , and optimizes candidate action sequences with CEM under the terminal cost in Eq. (8). Candidate sequences are rolled out using $g _ { \phi }$ for H latent model steps, where each model-step action is a block of five primitive environment actions.

Planning uses the same MPC and CEM hyperparameters in all four environments (Tab. 4). We sample 100 start states without replacement from the held-out evaluation dataset, requiring that the same episode contains a goal frame 25 steps later. The environment is reset to the sampled start state, the target is set from the goal row, and success is measured over a budget of 50 primitive environment steps. The sampled tasks are fixed across methods and seeds within each environment.

## A.5 Probing protocol.

For Tab. 1, we freeze $f _ { \theta } ,$ , encode embeddings $z ~ = ~ f _ { \theta } ( o )$ , and train probes on 25,000 training embeddings per environment and method, reporting held-out $R ^ { 2 }$ on a disjoint 5,000-sample validation split. The linear probe is ridge regression with $\alpha = 1 0 ^ { - 3 }$ . The nonlinear probe is a two-layer MLP with widths 256, 128 and ReLU activations. For vector-valued targets, including sine–cosine angle encodings, we report uniformly averaged $R ^ { 2 }$ over target coordinates.

Table 4: Planning hyperparameters.

<table><tr><td>Hyperparameter</td><td>Value</td></tr><tr><td>Planning horizon H</td><td>5 model steps</td></tr><tr><td>Executed steps K</td><td>5 model steps</td></tr><tr><td>Action block</td><td>5 primitive actions</td></tr><tr><td>CEM population</td><td>300</td></tr><tr><td>CEM iterations</td><td>30</td></tr><tr><td>CEM elites</td><td>30</td></tr><tr><td>Initial CEM variance scale</td><td>1.0</td></tr><tr><td>Evaluation tasks</td><td>100</td></tr><tr><td>Goal offset</td><td>25 primitive steps</td></tr><tr><td>Evaluation budget</td><td>50 primitive steps</td></tr></table>

## A.6 Compute.

All training and planning runs used a single NVIDIA H100 GPU. Each world model was trained in under 4 hours.

## B Environments and datasets

Our experiments use the four environments and offline datasets adopted by Maes et al. [14] (Fig. 8). All models are trained and evaluated through the stable-worldmodel codebase [49], which standardizes the environment interfaces, dataset loading, and downstream planning procedure.

For every environment, we use an episode-level 90/10 train/validation split: $f _ { \theta } , g _ { \phi } .$ , and $h _ { \psi }$ are trained on the training episodes, while all reported metrics—planning (Figs. 5 and 9), probes (Tab. 1), and latent-geometry analyses—are computed on the held-out validation split.

(a) TwoRoom. TwoRoom [13, 50] is a 2D point-navigation task with a simple constrained geometry: the agent starts in one room and the goal lies in the other, with a single doorway as the only passage between them. The script moves the agent first toward the doorway and then toward the goal after crossing the wall. The dataset contains 10,000 noisy scripted rollouts, averaging 92 steps.  
(b) Push-T. Push-T is a 2D contact-rich manipulation task in which a dot-shaped agent must push a T-shaped object into a target pose. We use the DINO-WM expert demonstrations [12], consisting of 20,000 episodes with mean length 196.  
(c) OGBench-Cube. OGBench-Cube [51] is a tabletop manipulation task where a robot gripper must move a cube to a specified target pose. We use the single-cube setting and the benchmark’s scripted dataset: 10,000 trajectories, each of length 200.  
(d) Reacher. Reacher is the two-link arm task from the DeepMind Control Suite [52]. We follow the DINO-WM variant, where success is defined by matching the target joint configuration, not just by end-effector proximity. The offline dataset contains 10,000 SAC-generated episodes [53], each 200 steps long.

## C Planning performance on extended goal offsets

Fig. 9 evaluates whether the learned latent dynamics remain useful when the goal is placed farther from the initial state. For each goal offset, the start and goal observations are sampled from the held-out evaluation split, and the planner is given a budget of 2× the offset in primitive environment steps. Thus the evaluation becomes harder because the model must support longer latent rollouts. The same trained models and MPC/CEM planner from Fig. 5 are used throughout.

![](images/5dbc724dcd9320246a6f5425ad21afe7c7ec301111abaa1410af172e739b3f10.jpg)

<details>
<summary>text_image</summary>

TwoRoom
</details>

![](images/a5cd8175951cd7cf200c351887a3bb95d9a196a9cee9bb03f4f6a1dd99c8fa74.jpg)

<details>
<summary>text_image</summary>

Reacher
</details>

![](images/c6180c8cb897a05d4136bbea9d2cb3042c555fc3d5f8b6e31f7a51883190830f.jpg)

<details>
<summary>natural_image</summary>

Simple 3D diagram of a push button with blue and green directional arrows (no text or symbols)
</details>

![](images/6fc2b2cdfd70c4d973120a625a1c65d07b469d94800a34b25f13a77c450a3048.jpg)

<details>
<summary>natural_image</summary>

3D rendering of a robotic arm with purple base and black body, labeled 'OGBench-Cube' at top (no other text or symbols)
</details>

Figure 8: Environments. Four evaluation environments—TwoRoom (2D navigation), Reacher (continuous control), Push-T (2D contact-rich manipulation), and OGBench-Cube (3D tabletop manipulation).

SMWM is stable across longer horizons on TwoRoom and OGBench-Cube, where SIGReg either degrades sharply or remains consistently lower. On Reacher, the inverse and SIGReg curves stay close over the tested offsets. Push-T is the main failure case: both regularized methods drop as the offset increases.

![](images/f5f3e15e901965937e48d96c4637aab86f65f5b1ce0186554ceb85b22570ad99.jpg)

<details>
<summary>line chart</summary>

| Environment | Method | Goal offset | Success rate (%) |
| --- | --- | --- | --- |
| TwoRoom | Our | 25 | 100 |
| TwoRoom | Our | 40 | 95 |
| TwoRoom | Our | 55 | 90 |
| TwoRoom | Our | 70 | 85 |
| TwoRoom | Our | 85 | 80 |
| TwoRoom | Our | 100 | 75 |
| TwoRoom | SIGReg | 25 | 90 |
| TwoRoom | SIGReg | 40 | 85 |
| TwoRoom | SIGReg | 55 | 80 |
| TwoRoom | SIGReg | 70 | 75 |
| TwoRoom | SIGReg | 85 | 70 |
| TwoRoom | SIGReg | 100 | 65 |
| TwoRoom | Random | 25 | 30 |
| TwoRoom | Random | 40 | 25 |
| TwoRoom | Random | 55 | 20 |
| TwoRoom | Random | 70 | 15 |
| TwoRoom | Random | 85 | 10 |
| TwoRoom | Random | 100 | 5 |
| TwoRoom | Forward only | 25 | 40 |
| TwoRoom | Forward only | 40 | 35 |
| TwoRoom | Forward only | 55 | 30 |
| TwoRoom | Forward only | 70 | 25 |
| TwoRoom | Forward only | 85 | 20 |
| TwoRoom | Forward only | 100 | 15 |
| Reacher | Our | 25 | 70 |
| Reacher | Our | 40 | 85 |
| Reacher | Our | 55 | 90 |
| Reacher | Our | 70 | 95 |
| Reacher | Our | 85 | 90 |
| Reacher | Our | 100 | 85 |
| Reacher | SIGReg | 25 | 80 |
| Reacher | SIGReg | 40 | 90 |
| Reacher | SIGReg | 55 | 95 |
| Reacher | SIGReg | 70 | 90 |
| Reacher | SIGReg | 85 | 85 |
| Reacher | SIGReg | 100 | 80 |
| Reacher | Random | 25 | 20 |
| Reacher | Random | 40 | 15 |
| Reacher | Random | 55 | 10 |
| Reacher | Random | 70 | 5 |
| Reacher | Random | 85 | 5 |
| Reacher | Random | 100 | 5 |
| Push-T | Our | 25 | 90 |
| Push-T | Our | 40 | 60 |
| Push-T | Our | 55 | 30 |
| Push-T | Our | 70 | 15 |
| Push-T | Our | 85 | 10 |
| Push-T | Our | 100 | 10 |
| Push-T | SIGReg | 25 | 90 |
| Push-T | SIGReg | 40 | 70 |
| Push-T | SIGReg | 55 | 40 |
| Push-T | SIGReg | 70 | 25 |
| Push-T | SIGReg | 85 | 20 |
| Push-T | SIGReg | 100 | 25 |
| Push-T | Random | 25 | 10 |
| Push-T | Random | 40 | 5 |
| Push-T | Random | 55 | 2 |
| Push-T | Random | 70 | 1 |
| Push-T | Random | 85 | 1 |
| Push-T | Random | 100 | 1 |
| OGBench-Cube | Our | 25 | 85 |
| OGBench-Cube | Our | 40 | 75 |
| OGBench-Cube | Our | 55 | 60 |
| OGBench-Cube | Our | 70 | 45 |
| OGBench-Cube | Our | 85 | 40 |
| OGBench-Cube | Our | 100 | 45 |
| OGBench-Cube | SIGReg | 25 | 70 |
| OGBench-Cube | SIGReg | 40 | 65 |
| OGBench-Cube | SIGReg | 55 | 45 |
| OGBench-Cube | SIGReg | 70 | 35 |
| OGBench-Cube | SIGReg | 85 | 30 |
| OGBench-Cube | SIGReg | 100 | 35 |
| OGBench-Cube | Random | 25 | 60 |
| OGBench-Cube | Random | 40 | 45 |
| OGBench-Cube | Random | 55 | 30 |
| OGBench-Cube | Random | 70 | 25 |
| OGBench-Cube | Random | 85 | 20 |
| OGBench-Cube | Random | 100 | 25 |
| OGBench-Cube | Forward only | 25 | 60 |
| OGBench-Cube | Forward only | 40 | 55 |
| OGBench-Cube | Forward only | 55 | 35 |
| OGBench-Cube | Forward only | 70 | 30 |
| OGBench-Cube | Forward only | 85 | 35 |
| OGBench-Cube | Forward only | 100 | 40 |
</details>

Figure 9: Robustness to planning horizon. Goal-conditioned planning success rate as a function of goal $o f f s e t ,$ the number of environment steps between the initial and goal observations; the planner’s evaluation budget is fixed at $2 \times$ the goal offset. Methods and environments match Fig. 5. SMWM maintains near-flat success across horizons on TwoRoom and OGBench-Cube, while SIGReg either degrades sharply (TwoRoom) or trails consistently (Cube). The two regularized methods are comparable on Reacher and Push-T.

## D SIGReg latent geometry

Fig. 10 repeats the representation summary of Fig. 6 for the SIGReg baseline. The same heldout embeddings and PCA protocol are used, with the bottom row showing $\left( \mathrm { P C } _ { 1 } , \mathrm { P C } _ { 2 } \right)$ for each environment. Unlike SMWM embeddings, the SIGReg spectra do not exhibit a clear elbow. This is expected: the SIGReg regularizer explicitly encourages the embedding distribution to match an isotropic Gaussian, spreading variance across many principal components rather than concentrating it in a compact task-relevant subspace. As a result, although state information remains recoverable by probes, the resulting representation is not low-dimensional, and is less directly interpretable than the SMWM representation.

## E Dot world details

## E.1 Environment

Each observation is a 64×64 RGB image rendered on a white canvas: each dot is drawn as a filled disc of radius $r = 2$ pixels in a fixed RGB color. Integer dot positions are sampled uniformly in $[ m , 6 4 - 1 - m ] ^ { 2 }$ with a margin of $m = r + \Delta _ { \mathrm { m a x } } = 1 8$ , and integer per-step displacements in $[ - \Delta _ { \mathrm { m a x } } , \Delta _ { \mathrm { m a x } } ] ^ { 2 }$ with $\Delta _ { \mathrm { m a x } } = 1 6$ , so every sampled transition stays inside the canvas. Rejection sampling enforces a minimum center-to-center separation of $2 r + 1 = 5$ pixels at both endpoints. Each $\left( o _ { t } , a _ { t } , o _ { t + 1 } \right)$ is generated lazily from a seeded RNG, making the dataset deterministic and reproducible.

![](images/f59835a8a78d41ecb22ed8bbe162631d3ed3bb9743af29b5ce7d3998e1069f13.jpg)  
Figure 10: Latent geometry of SIGReg embeddings. For each environment we show the PCA spectrum of held-out embeddings (top), the distribution of a representative ground-truth quantity in physical state space (middle), and the first two principal components of the encoded embeddings, color-coded by the same physical quantity (bottom). The dashed red lines mark the action dimension. Compared with SMWM embeddings in Fig. 6, SIGReg produces less compact and less structured latent geometry: the spectra lack a sharp elbow, and the state-dependent structure is spread across many components rather than appearing in a small interpretable PC subspace.

The four configurations of Fig. 4 combine three motion types: Independent dots have their own $( \Delta x , \Delta y )$ in the action vector, coupled dot pairs share a single $( \Delta x , \Delta y )$ , and random dots move at every step without contributing to the action vector. Concretely, Independent = 2 independent dots $( | a | = 4 )$ ; Coupled = 1 coupled pair $( | a | = 2 )$ ; Distractor = 1 independent dot + 1 random dot $( | a | = 2 ) ;$ Combined = 1 coupled pair + 2 independent dots + 1 random dot $( | a | = 6 )$ .

## E.2 Architecture and training

The encoder $f _ { \theta }$ stacks three stride-2 convolutions (3→32→64→128 channels, $3 { \times } 3$ kernels) interleaved with ReLU, followed by a $2 \times 2$ max-pool, a flatten, a linear projection to $\mathbb { R } ^ { d }$ , and a final tanh that bounds embeddings in $[ - \dot { 1 } , 1 ] ^ { d }$ with $d = 6 4$ . Both $g _ { \phi }$ and $h _ { \psi }$ are 2-hidden-layer MLPs of width 256 with ReLU activations; $g _ { \phi }$ ends in tanh to match the encoder range, while $h _ { \psi }$ has no output activation. Approximate parameter counts: $f _ { \theta } \approx 2 2 0 \mathrm { k \Omega }$ , $g _ { \phi } \approx 1 0 0 \mathbf { k } , h _ { \psi }$ ≈ 100k.

All three networks are trained jointly with Adam $( \mathrm { l r } = 5 { \times } 1 0 ^ { - 4 }$ , default $\beta ,$ no weight decay, no LR schedule) for 100 epochs at batch size 256 on 250,000 transitions, with seed 42 and $\lambda = 1 0$ . Actions are normalized by $\Delta _ { \mathrm { m a x } }$ before being passed to $g _ { \phi }$ and predicted by $h _ { \psi }$ , so all action coordinates lie in $[ - 1 , 1 ] ;$ we report $\mathcal { L } _ { \mathrm { i n v } }$ on these normalized actions. All metrics are computed on a held-out evaluation split of 1,000 transitions sampled with a disjoint seed.

## E.3 Collapse without the inverse loss

To verify that the inverse loss is the active anti-collapse mechanism on dot world, we rerun the singledot configuration with $\lambda = 0$ and otherwise identical hyperparameters. Fig. 11 reports the resulting representation using the same probe grid as Fig. 2. Without the inverse loss, the encoder collapses all probe observations to essentially the same embedding. Consequently, there is no state-dependent variation for PCA to expose. The forward loss alone can therefore be minimized by a constant, trivially predictable representation.

![](images/ce962a098baa877ed6cd75aefd381073d04a200bd2710df6e453bea3780e3e4e.jpg)

<details>
<summary>bar chart</summary>

| Principal component | Expilated variance ratio |
| ------------------- | ------------------------ |
| 1                   | 0.10                     |
| 2                   | 0.01                     |
| 3                   | 0.01                     |
| 4                   | 0.01                     |
| 5                   | 0.01                     |
| 6                   | 0.01                     |
| 7                   | 0.01                     |
| 8                   | 0.01                     |
| 9                   | 0.01                     |
| 10                  | 0.01                     |
| 11                  | 0.01                     |
| 12                  | 0.01                     |
| 13                  | 0.01                     |
| 14                  | 0.01                     |
| 15                  | 0.01                     |
</details>

![](images/537ff041372fc644b162529573016551a917103b5a63b5d8f9d43f9082d3170d.jpg)

<details>
<summary>scatter plot</summary>

| x  | y  |
|----|----|
| 12 | 12 |
| 12 | 24 |
| 12 | 36 |
| 12 | 48 |
| 24 | 12 |
| 24 | 24 |
| 24 | 36 |
| 24 | 48 |
| 36 | 12 |
| 36 | 24 |
| 36 | 36 |
| 36 | 48 |
| 48 | 12 |
| 48 | 24 |
| 48 | 36 |
| 48 | 48 |
</details>

![](images/989ee3832bd29166dc0de005f076014577b35f03887981f22417047fbca951f2.jpg)

<details>
<summary>scatter plot</summary>

| PC1 | PC2 |
| --- | --- |
| 0   | 0   |
</details>

Figure 11: Forward-only collapse on dot world. The single-dot model is trained with $\lambda = 0$ and evaluated with the same probe grid and visualization protocol as Fig. 2. Removing the inverse loss maps the probe states to the same embedding, so the PC projection collapses to a single point at the origin.

## F Sprite world reconstruction experiment

## F.1 Environment and control settings

To make the control-dependent nature of the learned representation visually explicit, we introduce a second toy environment with a single triangular agent moving in a 2D plane. The agent is rendered as an asymmetric sprite on a 64×64 RGB canvas. Its state is its pose,

$$
s = (x, y, \theta),
$$

where x, y denote position and θ denotes orientation. Because the triangle is asymmetric, its orientation is visible from a single frame.

The key intervention in this experiment is to vary the action repertoire while keeping the underlying visual world fixed. In all settings, the agent can move in all three degrees of freedom: $x , y ,$ , and ${ \bar { \theta } } .$ . The difference is only which changes are exposed to the model as actions. Uncontrolled degrees of freedom are not held fixed; they continue to move randomly.

We consider the following control settings:

• No control: the agent translates and rotates randomly, and the action vector is empty.  
• $x / y$ control: the action is $a _ { t } = ( \Delta x , \Delta y )$ , while rotations remain random and uncontrolled.  
• $x / y / \theta$ control: the action is $a _ { t } = ( \Delta x , \Delta y , \Delta \theta )$ , so both translation and rotation are controlled.

Thus the same triangular agent is observed under different action interfaces. This lets us ask whether the representation changes depending on which aspects of the agent’s pose are controllable.

## F.2 Training and reconstruction probe

For each control setting, we train the same inverse-dynamics-regularized world model as in the dot-world experiments: a CNN encoder $f _ { \theta } ,$ a forward model $g _ { \phi }$ , and an inverse model $h _ { \psi }$ , optimized with the objective in Eq. (4). The architecture and training procedure match Sec. E.2. In the no-control setting, the action dimension is zero, so there is no inverse-dynamics signal and the model reduces to the forward-only objective.

After training, we freeze the encoder and train a decoder as a post-hoc probe,

$$
\hat {o} _ {t} = d _ {\xi} (f _ {\theta} (o _ {t})).
$$

The decoder is trained only to visualize what information is already present in the frozen embedding $z _ { t } = f _ { \theta } ( o _ { t } )$ ; it is not used to train the representation.

![](images/f0dd8394e31f4df49ec3bb7f3a79abb03fd2ad79a339c3a8b3d7895463ccb803.jpg)

<details>
<summary>heatmap</summary>

| Condition | Time Point | Red Triangle Shape | Red Circle Shape |
|---|---|---|---|
| Ground truth | t | ▲ | ▼ |
| Ground truth | t+1 | ▼ | ▼ |
| Ground truth | t+2 | ▼ | ▼ |
| Ground truth | t+3 | ▼ | ▲ |
| Ground truth | t+4 | ▲ | ▼ |
| Ground truth | t+5 | ▼ | ▼ |
| No control | t | ■ | ■ |
| No control | t+1 | ■ | ■ |
| No control | t+2 | ■ | ■ |
| No control | t+3 | ■ | ■ |
| No control | t+4 | ■ | ■ |
| No control | t+5 | ■ | ■ |
| x/y control | t | ● | ● |
| x/y control | t+1 | ● | ● |
| x/y control | t+2 | ● | ● |
| x/y control | t+3 | ● | ● |
| x/y control | t+4 | ● | ● |
| x/y control | t+5 | ● | ● |
| x/y/angle | t | ▲ | ▼ |
| x/y/angle | t+1 | ▼ | ▼ |
| x/y/angle | t+2 | ▼ | ▼ |
| x/y/angle | t+3 | ▼ | ▼ |
| x/y/angle | t+4 | ▲ | ▼ |
| x/y/angle | t+5 | ▼ | ▼ |
</details>

Figure 12: Control-dependent reconstruction of a triangular agent. The top row shows the ground-truth trajectory of an asymmetric triangular agent with pose (x, y, θ). The remaining rows show reconstructions from frozen embeddings learned under different action interfaces. With no control, the representation collapses and the decoder outputs an average occupancy pattern. With $x / y$ control, the representation preserves position but averages over the uncontrolled orientation, producing a blurred, approximately symmetric blob at the correct location. With full $x / y / \theta$ control, the representation preserves both position and orientation, yielding sharp oriented reconstructions.

## F.3 Interpretation

The experiment shows that the model “sees” the same visual object differently depending on what the agent can control. When there are no actions, there is no inverse signal that could force the encoder to preserve state information, so the representation collapses. When only translation is controlled, the encoder preserves where the agent is, because this is needed to recover and predict translational effects, but it discards orientation, because rotation is random and action-irrelevant. The decoder therefore averages over possible orientations. When rotation is also controlled, orientation becomes action-relevant, and the encoder preserves the full pose.

This provides a direct visual counterpart to the PCA results in Fig. 4: the learned representation retains controllable degrees of freedom and averages out uncontrolled ones.

## G Equivariance and latent action composition

This appendix gives the formal statement behind the compositional claim in Sec. 4. Denote the effect of action a on the observation o by $o \mapsto a ( o )$ , and write $g _ { a } ( z ) : = g _ { \phi } ( z , a )$ . In the main text, the forward prediction loss encourages the approximate equivariance condition

$$
f (a (o)) \approx g _ {a} (f (o)), \tag {9}
$$

on the support of the training distribution. Here, for clarity, we consider the exact version

$$
f (a (o)) = g _ {a} (f (o)) \tag {10}
$$

for all $a \in { \mathcal { A } }$ and $o \in { \mathcal { O } }$ . The encoder thus intertwines the physical intervention with the learned latent intervention, making this a statement of equivariance. In physics terminology one might also say that the latent state transforms covariantly with the intervention.

We consider sequences of actions, assuming that the actions form a semigroup under composition; $\mathrm { i . e . }$ ., for $a _ { 1 } , a _ { 2 } \in { \mathcal { A } }$ , the composition $a _ { 2 } \circ a _ { 1 }$ is again in A, and composition is associative. We say that $a \mapsto g _ { a }$ is a homomorphism on the latent manifold $f ( \mathcal O )$ if, for every $a _ { 1 } , a _ { 2 } \in { \mathcal { A } } ,$ , we have

$$
g _ {a _ {2} \circ a _ {1}} = g _ {a _ {2}} \circ g _ {a _ {1}} \quad \text { on } f (\mathcal {O}), \tag {11}
$$

i.e., for every $z \in f ( \mathcal { O } )$ ,

$$
g _ {a _ {2} \circ a _ {1}} (z) = g _ {a _ {2}} (g _ {a _ {1}} (z)). \tag {12}
$$

Proof of theorem 1. Choose any $a _ { 1 } , a _ { 2 } \in { \mathcal { A } }$ and $z \in f ( \mathcal { O } )$ . Choose $o \in \mathcal { O }$ with $f ( o ) = z$ . By closure, $a _ { 3 } : = a _ { 2 } \circ a _ { 1 } \in \mathcal { A }$ .

Applying equivariance three times,

$$
\begin{array}{l} g _ {a _ {2} \circ a _ {1}} (z) = g _ {a _ {3}} (f (o)) \\ = f \left(a _ {3} (o)\right) \quad [ \text { equivariance   at } (a _ {3}, o) ] \\ = f \left(a _ {2} \left(a _ {1} (o)\right)\right) \\ = g _ {a _ {2}} \left(f \left(a _ {1} (o)\right)\right) \quad [ \text { equivariance   at } \left(a _ {2}, a _ {1} (o)\right) ] \\ = g _ {a _ {2}} \left(g _ {a _ {1}} (f (o))\right) \quad [ \text { equivariance   at } (a _ {1}, o) ] \\ = g _ {a _ {2}} \left(g _ {a _ {1}} (z)\right). \\ \end{array}
$$

Thus $g _ { a _ { 2 } \circ a _ { 1 } } ( z ) = g _ { a _ { 2 } } ( g _ { a _ { 1 } } ( z ) )$ for every $z \in f ( \mathcal { O } )$ , so $a \mapsto g _ { a }$ is a homomorphism on $f ( \mathcal { O } )$ .

![](images/a18d12ced2606ca500c96cf0538959a1b6bd139df4c0e132b2cf6b80d5bcea97.jpg)

One-step equivariance thus forces physical action composition to be represented by composition of the learned latent interventions, at least on the encoded data manifold. In the learned model, this identity should be interpreted approximately and on-support. Note that the assumption of semigroup closure is nontrivial: in practice, it could happen that a multi-step action sequence is not equivalent to any single primitive action.56

Equivariance alone admits a wide range of solutions, including two trivial extremes:

1. No encoding. Take $\mathcal { Z } = \mathcal { O } , f = \mathrm { i d }$ , and $g _ { a } = a$ . Then both sides of Eq. (10) equal $a ( o )$ . This solution satisfies equivariance but achieves no compression.

2. Collapse. Take $\mathcal { Z } = \{ z \} , f \equiv z$ , and $g _ { a } = \mathrm { i d }$ . Then both sides of Eq. (10) equal z. This solution satisfies equivariance but discards all information about a.

Useful representations therefore need more than equivariance: the latent dynamics should remain faithful to the physical action. In particular, if an action a $, \in { \mathcal { A } }$ changes observations nontrivially in O, it should also induce a nontrivial transformation on $f ( \mathcal { O } )$ . A stronger requirement is latent action identifiability: whenever $z ^ { \prime } = g _ { a } ( z )$ for $( z , z ^ { \prime } ) \in f ( \mathcal { O } ) \stackrel { \cdot } { \times } \hat { f } ( \mathcal { O } )$ ), the action a should be recoverable from the latent transition $( z , z ^ { \prime } )$ . This is exactly the constraint imposed by the inverse model, which predicts a from $( z , z ^ { \prime } )$ .

Beyond a homomorphism into latent transformations under composition, the experiments in Fig. 3 suggest that in our case the homomorphism may approximately go into the additive group, i.e., the learned interventions appear to act approximately as translations,

$$
g _ {a} (z) \approx z + \rho (a), \tag {13}
$$

with $\rho ( a )$ approximately independent of z. Here $\rho ( a )$ is the latent displacement, or latent effect vector, associated with action a. Substituting this into the composition identity yields

$$
z + \rho (a _ {2} \circ a _ {1}) \approx g _ {a _ {2}} (g _ {a _ {1}} (z)) \approx z + \rho (a _ {1}) + \rho (a _ {2}), \tag {14}
$$

and therefore

$$
\rho (a _ {2} \circ a _ {1}) \approx \rho (a _ {1}) + \rho (a _ {2}). \tag {15}
$$

Table 5: Comparison of related methods. “Inv.” and “Fwd.” denote inverse and forward dynamics components; “True action” indicates supervision with ground-truth actions; “WM” marks methods that deliver a world model usable for planning. The last column lists the primary anti-collapse mechanism. SMWM is the only entry that combines offline reward-free world-model training with inverse dynamics as the sole anti-collapse mechanism.

<table><tr><td>Method</td><td>Inv.</td><td>Fwd.</td><td>True action</td><td>Reward</td><td>Offline</td><td>WM</td><td>Anti-collapse</td></tr><tr><td>ICM [30]</td><td>√</td><td>√</td><td>√</td><td>×</td><td>×</td><td>×</td><td>Inv. dynamics on  $\phi_{\theta}$ </td></tr><tr><td>CURL [41]</td><td>×</td><td>×</td><td>√</td><td>√</td><td>×</td><td>×</td><td>Contrastive</td></tr><tr><td>Dreamer [4]</td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>Recon. + KL</td></tr><tr><td>TD-MPC [5]</td><td>×</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>EMA latent target</td></tr><tr><td>DynaMo [39]</td><td>√</td><td>√</td><td>×</td><td>×</td><td>√</td><td>√</td><td>Joint inv. + fwd. dyn.</td></tr><tr><td>PLDM [13]</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>VICReg + inv. dyn.</td></tr><tr><td>DINO-WM [12]</td><td>×</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>Frozen DINOv2</td></tr><tr><td>LeWorldModel [14]</td><td>×</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>SIGReg</td></tr><tr><td>WAM [35]</td><td>√</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>Recon. + KL + inv. dyn.</td></tr><tr><td>EB-JEPA [36]</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>VICReg + inv. dyn.</td></tr><tr><td>SMWM</td><td>√</td><td>√</td><td>√</td><td>×</td><td>√</td><td>√</td><td>Inv. dynamics alone</td></tr></table>

For pure displacement actions, where composition is itself approximately additive away from boundaries, this means that the model has learned an approximately additive latent representation of interventions. This stronger property is empirical rather than imposed by the objective.7

It is intriguing to ask where this translation-like structure may stem from. The inverse-dynamics loss provides a plausible bias toward this regime. Equivariance alone admits degenerate solutions, including collapse; useful representations additionally require the latent action to be identifiable from the transition. The inverse head enforces this by requiring $a _ { t }$ to be recoverable from $\left( z _ { t } , z _ { t + 1 } \right)$ , where $z _ { t } = f ( o _ { t } )$ and $z _ { t + 1 } = f ( o _ { t + 1 } )$ . A simple way to satisfy this constraint is to encode the action in the displacement vector

$$
z _ {t + 1} - z _ {t} \approx \rho (a _ {t}). \tag {16}
$$

If $\rho$ is injective on the action support, then the inverse $h$ can decode the action by applying an approximate inverse of $\rho$ to this displacement. This mechanism does not enforce additivity, but it renders an additive, translation-like representation a low-complexity solution.

This perspective connects the empirical latent geometry to an interventional version of the linear representation hypothesis [56]. The claim is stronger than ordinary linear decodability: it is not merely that actions can be decoded from latent states, but that applying an intervention corresponds approximately to moving along an action-dependent direction in latent space,

$$
f (a (o)) - f (o) \approx \rho (a). \tag {17}
$$

Thus the learned representation appears to linearize controllable interventions: the encoder is approximately equivariant to action-induced transformations, and in simple displacement domains the learned latent interventions further specialize to additive translations.

## H Extended related work

This appendix expands the discussion of Sec. 2. Tab. 5 summarizes the discussed methods along the dimensions most relevant to our contribution.

## H.1 Inverse dynamics for representation learning

## H.1.1 Intrinsic Curiosity Module (ICM)

ICM [30] is the canonical inverse-dynamics-as-feature-learner construction we extend. Two architectural differences matter for our claim. First, in ICM the encoder receives gradients only from the inverse loss; the forward loss treats $\phi _ { \theta } \mathopen { } \mathclose \bgroup \left( o _ { t + 1 } \aftergroup \egroup \right)$ as a stop-gradient target and trains only $g _ { \omega }$ . In our setup the encoder receives gradients from both losses, because the forward predictor is part of the world model we deliver. Second, ICM uses a separate A3C policy with its own encoder; the ICM encoder is used only to compute a scalar intrinsic reward. Our encoder is the world-model encoder used downstream. Conceptually, ICM motivates inverse dynamics as a way to ignore action-irrelevant variation, whereas we frame it as the anti-collapse mechanism that allows joint training of an end-to-end JEPA world model without reconstruction, EMA, or distributional regularizers.

## H.1.2 Multi-step inverse identifiability theory

A line of theoretical work establishes asymptotic recovery of the control-endogenous latent state for multi-step inverse models in finite Exogenous Block MDPs [31–34]. We summarize their assumptions precisely here.

AC-State [32]. AC-State is proved correct under: (i) finite endogenous state space $S ; ( \mathrm { i i } )$ finite action space A; (iii) deterministic endogenous transition $T ( s ^ { \prime } \mid ^ { - } s , a ) ;$ ; (iv) action-independent exogenous noise; (v) bounded endogenous diameter D and a behavior policy that mixes well; (vi) a multi-step inverse loss with horizon $K \geq D$ , jointly with an information bottleneck that selects, among loss-minimizers, the encoder of minimum output cardinality. The conclusion is asymptotic identification of the partition induced by $f ^ { * }$ . The authors explicitly note that the finite-state assumption “rules out continuous control problems”. Their Appendix B exhibits a discrete Ex-BMDP in which a single-step inverse model cannot distinguish two distinct endogenous states; Efroni et al. [31] and Islam et al. [33] provide further such counterexamples.

Refinements and known failure modes. Levine et al. [57] show that even AC-State can fail in deterministic Ex-BMDPs when the witness distance between two states exceeds D, and fails for any finite horizon K under periodic endogenous dynamics; they propose adding a latent forward-dynamics term to restore guarantees.

Implication for our work. Our setting differs from these results on every dimension that matters: continuous (rather than finite) state and action spaces, single-step (rather than multi-step) inverse, possibly stochastic endogenous dynamics, and no information bottleneck on the encoder’s output cardinality. We therefore do not claim that any of these guarantees transfer to our inverse-dynamics objective. We cite this body of work as conceptual motivation: a substantial theoretical literature confirms that inverse-dynamics objectives of various forms learn features tied to controllable structure under some conditions, and this informs our design choice. The single-step positive result of Brandfonbrener et al. [58] for a linear-Gaussian multitask imitation model is the closest theoretical endorsement of single-step inverse dynamics that we know of, but its scope (deconfounding latent task variables in imitation) does not match ours.

## H.2 Inverse-dynamics-regularized world models

DynaMo [39]. DynaMo is architecturally the closest neighbor to ours: joint latent inverse and forward dynamics without reconstruction. The decisive difference is action supervision: DynaMo treats actions as unobserved latents inferred jointly with the encoder, while we supervise the inverse head with the executed action $a _ { t }$ . Ground-truth labels turn the inverse term into a hard anti-collapse signal that a co-learned action representation cannot provide.

LAPO and APV. Inverse-dynamics ideas also underlie work that infers pseudo-actions from action-free video. LAPO [38] uses inverse dynamics model to discover latent actions and trains a forward model conditioned on these. APV [40] pretrains an action-free latent video predictor and fine-tunes for control. These methods address the complementary regime where ground-truth actions are unavailable; with actions in hand, we use them directly to supervise inverse-dynamics prediction.

World-Action Model [35]. WAM augments DreamerV2 with an inverse dynamics head and reports that this regularizes the latent state toward action-relevant structure. The construction is conceptually closest to ours among reconstruction-based methods. Two differences are decisive: WAM operates within Dreamer’s pixel-reconstruction ELBO and requires reward supervision; we are reconstructionfree, JEPA-style, and reward-free. WAM treats the inverse dynamics head as one auxiliary among many; we identify it as the sufficient anti-collapse mechanism for end-to-end JEPA training.

EB-JEPA [36]. EB-JEPA is an action-conditioned JEPA with a multi-term loss combining variance, covariance, temporal-similarity, and inverse-dynamics regularizers; their ablations report that all four terms contribute to anti-collapse. Our claim is more specific: the inverse-dynamics term alone suffices, removing the need for variance and covariance terms and yielding a substantially simpler training objective.

Auxiliary-task JEPA theory [37]. Concurrent theoretical work proves a no-unhealthy-collapse theorem for JEPAs with an auxiliary regression head: in deterministic MDPs, if both the latent prediction loss and the auxiliary regression loss are zero, then observations with different transition dynamics or different auxiliary values must map to distinct embeddings. Our IDR objective is the action-prediction instantiation of this template. Our contribution complementary to theirs is empirical: we instantiate the template on continuous pixel-action data and show that single-step IDR is sufficient in practice.

## H.3 Alternative anti-collapse mechanisms in latent world models

## H.3.1 Reconstruction-based

The Dreamer family [2, 4, 24] and PlaNet [25] train recurrent state-space models with pixelreconstruction and reward-prediction heads under a variational objective:

$$
\mathcal {L} _ {\text { Dreamer }} = \mathbb {E} \left[ \sum_ {t} - \log p _ {\theta} (o _ {t} | h _ {t}, z _ {t}) - \log p _ {\theta} (r _ {t} | h _ {t}, z _ {t}) + \beta \operatorname{KL} \bigl (q _ {\theta} (z _ {t} | \cdot) \| p _ {\theta} (z _ {t} | h _ {t}) \bigr) \right]. \tag {18}
$$

DreamerV3 retains this objective family but adds engineering refinements (symlog targets, KL balancing, two-hot value targets) for stability across domains. Reconstruction provides an implicit anti-collapse signal but commits modelling capacity to pixel-level details that are often irrelevant for control. Our setting differs from Dreamer’s in two ways: we train without reward labels, and we predict in embedding space rather than reconstructing observations.

## H.3.2 EMA latent targets

TD-MPC [5, 6] learns a latent encoder $z _ { t } = \phi _ { \theta } ( o _ { t } )$ and a latent transition $g _ { \omega }$ together with reward and value heads, all trained end-to-end via TD learning. The objective combines reward, value, and a latent-consistency term that regresses $g _ { \omega } ( z _ { t } , a _ { t } )$ onto $\mathbf { s g } ( \phi _ { \theta ^ { - } } ( o _ { t + 1 } ) )$ , where $\theta ^ { - }$ is an EMA copy of θ. SPR [59] applies the same idea to data-efficient Atari, predicting k-step latent rollouts against a BYOL-style [10] momentum target encoder. Both methods rely on the EMA target as the primary defense against collapse: a slowly-moving target makes constant solutions a moving target. TD-MPC additionally requires reward supervision. Our work avoids EMA targets and reward labels, replacing both with the inverse-dynamics signal.

## H.3.3 Augmentation and contrastive methods

CURL [41] adds an InfoNCE [26] contrastive auxiliary loss between augmented views of the same observation, sharing the encoder with the policy and Q-function. DrQ [42] shows that random image-shift augmentation alone, without an explicit auxiliary loss, suffices for state-of-the-art visual RL. Proto-RL [43] uses prototypical clustering of embeddings as both an exploration bonus and a representation learner. None of these methods learns a world model in our sense: they learn encoders shared with model-free policies. Their anti-collapse mechanisms (negatives, augmentation, prototype balance) are orthogonal to inverse-dynamics regularization and could in principle be combined with it.

## H.3.4 JEPA-style regularizers

JEPAs predict in embedding space, with the target embedding produced by an encoder applied to the future or masked observation, typically tied to the online encoder via stop-gradient and EMA [7–9]. In the action-conditioned regime relevant to control, several recent methods stand out. Zhou et al. [12] (DINO-WM) freeze a pretrained DINOv2 encoder and learn only a latent dynamics model on top. Sobal et al. [13] (PLDM) train end-to-end with a multi-term objective combining VICReg-style regularizers [11] with an inverse-dynamics term. Maes et al. [14] (LeWorldModel)

build on LeJEPA [15] and regularize embeddings toward isotropic Gaussianity using SIGReg. Assran et al. [16] (V-JEPA 2) scale to large internet-video corpora and rely on EMA targets and a separate action-conditioning post-training stage. Our inverse-dynamics objective adds a distinct point on this design axis: a task-grounded action-prediction signal anchoring the embedding to controllable structure.