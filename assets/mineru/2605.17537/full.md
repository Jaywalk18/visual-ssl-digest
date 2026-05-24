# Self-supervised Hierarchical Visual Reasoning with World Model

Yuanfei Xu 1 Lin Liu 1 Wengang Zhou 1 2 Mingxiao Feng 2 Houqiang Li 1 2

# Abstract

3D open-world environments with adversarial opponents remain a core challenge for reinforcement learning due to their vast state spaces. Effective reasoning representations are essential in such settings. While existing self-supervised visual foresight reasoning approaches often suffer from multi-step error accumulation, many recent studies resort to injecting domain-specific knowledge for more stable guidance. Our key insight is that the photorealistic fidelity of visual reasoning representations is secondary; what truly matters is providing informative, task-relevant signals. To this end, we propose ResDreamer, a hierarchical world model in which each higher-level layer is trained to reconstruct the residuals of the layer below. This design enables progressive abstraction of increasingly sophisticated world dynamics and fosters the emergence of richer latent representations. Drawing inspiration from the “Bitter Lesson,” ResDreamer trains its reasoning representations in a purely self-supervised manner. The higher-level residual representations are used to modulate lower-level predictions, allowing the world model to scale effectively with only linearly increasing cross-layer communication costs. Experiments show that ResDreamer achieves stateof-the-art sample efficiency and parameter efficiency. This scalable hierarchical visual foresight reasoning architecture paves the way for more capable online RL agents in open-ended, dynamic environments. The code is accessible at https://github.com/XuYuanFei01/ ResDreamer.

1Department of Electronic Engineering and Information Science, University of Science and Technology of China, Hefei, Anhui, China 2Institute of Artificial Intelligence, Hefei Comprehensive National Science Center, Hefei, Anhui, China. Correspondence to: Mingxiao Feng <fengmx@iai.ustc.edu.cn>, Houqiang Li <lihq@ustc.edu.cn>.

Proceedings of the $\it 4 3 ^ { r d }$ International Conference on Machine Learning, Seoul, South Korea. PMLR 306, 2026. Copyright 2026 by the author(s).

# 1. introduction

In interaction or combat scenarios, task objectives are relevant with dynamic elements or proactive agent. This introduces a highly dynamic and uncertain environment evolution. The vast state space of an 3D open-ended environment exacerbates this challenge. The agent must construct an internal world representation based on partial information and make decisions accordingly.

World models have significantly advanced the frontiers of reinforcement learning (RL) (Schrittwieser et al., 2020; Robine et al., 2023; Zhang et al., 2023; Alonso et al., 2024). DreamerV3 (Hafner et al., 2025) achieves strong generalization across over 150 diverse tasks using a single set of hyperparameters. Robotic World Model (Li et al., 2025a) enables stable prediction of 100+ proprioceptive observation steps. BOOM (Zhan et al., 2025) integrates world models with sampling-based online planning, attaining state-of-theart (SOTA) sample efficiency on high-dimensional locomotion tasks. However, existing model-based RL (MBRL) methods primarily rely on latent-space look-ahead search or model predictive control. Pixel-level reconstruction is rarely exploited as a reasoning representation, largely because it provides no additional information beyond latent representations and suffers from compounding error accumulation.

Reasoning representations are critical for long horizon tasks. Large Language Models (LLMs) can provide advanced and interpretable reasoning representations, such as decomposed sub-task prompts and policy-as-code formulations. Recent LLM-powered embodied agents tailored to the Minecraft environment include JARVIS-1 (Wang et al., 2023b), MC-Planner (Wang et al., 2023c), and RL-GPT (Liu et al., 2024). However, despite recent efforts to mitigate the latency of embodied Chain-of-Thought (CoT) reasoning (Zheng et al., 2025; Duan et al., 2025), language-grounded CoT remains poorly suited to high-frequency, dynamic environments due to its inherent inference overhead and lower temporal resolution.

Embodied visual reasoning representations—such as affordance maps (Chu et al., 2025), visual grounding (Li et al., 2024b; Chen et al., 2025b), and goal images (Zhao et al., 2025)—have promised superior task performance. However, they typically rely on domain-specific priors and environment-anchored annotations. In contrast, large-scale domain-general video generation models trained in an unsupervised manner, such as Genie (Bruce et al., 2024), WAN (Wan et al., 2025) and Cosmos (NVIDIA et al., 2025), generally comprise over 10B parameters. Their inference latency and deployment costs remain high for embodied online visual reasoning.

Neuroscience evidence suggests that the biological neural signals encode prediction error rather than raw sensory input (Rao & Ballard, 1999; Hosoya et al., 2005). Visual neurons employ a dynamic predictive coding strategy to filter out predictable components from the visual stream, transmitting only unexpected surprise or ”report valuable” stimulis (Kok & de Lange, 2015).

Building on these insights, we introduce ResDreamer, a hierarchical world model that employs residually connected visual planning representations. By modeling visual reconstruction residuals, each higher-level layer not only builds a richer, more comprehensive internal world representation but also refines the foresight of lower layers through residual reasoning, thereby providing more informative reasoning representation.

We adopt a highly lightweight parameter regime: Res-Dreamer operates with only 50–200 million parameters yet successfully solves long-horizon combat tasks in Minecraft open-world environments that feature complex battle mechanics. Our investigation centers on the novel residually connected visual reasoning representation we propose; accordingly, we incorporate no language-conditioned modules. We assess the method’s effectiveness via online RL success rates on these combat tasks.

In summary, the major contributions of this work are:

• We introduce a hierarchical world model architecture for representation learning, in which upper layers learn progressively more advanced world dynamics from the residuals of lower layers. This design paves the way for the ”ResNet era” of world models where residual connections enable deeper hierarchies.   
• We propose a residual-enhanced visual reasoning representation. By modulating visual foresight with normalized upper-layer residual predictions, we deliberately forgo photorealistic fidelity and instead emphasize unexpected visual stimulis, thereby delivering highly informative reasoning signals   
• ResDreamer achieves state-of-the-art sample efficiency and parameter efficiency in online RL settings, and it is the only method to exhibit non-near-zero success rates on high-difficulty tasks such as hunting Shulkers. Ablation studies further isolate the contributions of foresight rollouts and residual modulation.

# 2. Related Work

Imagination-driven MBRL. Recurrent world dynamic models facilitate representation, simulation and policy improvement in MBRL (Ha & Schmidhuber, 2018). MuZero (Schrittwieser et al., 2020) conducts Monte Carlo tree search in the latent space by the learned state space model. DreamerV3 (Hafner et al., 2025) outperformed expert models tuned for specific domains and, for the first time, successfully collected diamonds from scratch in Minecraft. LS-Imagine (Li et al., 2024a) breaks the limitations of singlestep reasoning and uses the affordance map to trigger the cross-step jump prediction. It simulates jumping to the vicinity of high return targets in the future by magnifying specific areas in the observed image. In visual MBRL, transformerbased architectures (Micheli et al., 2022; Robine et al., 2023; Zhang et al., 2023) and diffusion models (Alonso et al., 2024) have emerged as particularly effective paradigms for world modeling, offering enhanced expressivity and sample efficiency. Although these methods support look-ahead search via Monte Carlo Tree Search (MCTS), none of them are designed to provide reasoning representations that supply additional visual guidance. ResDreamer follows the imagination-driven MBRL paradigm, in which the actorcritic is trained purely on imagined trajectories. This design completely avoids policy distributional divergence. Consequently, we can leverage a massive replay buffer as a rich self-supervised signal for world model representation learning, while all policy and value updates remain strictly online. What sets our approach apart is that it naturally constructs and utilizes a hierarchical reasoning representation built directly upon the residuals of sensory signals.

Embodied Reasoning Representation. Embodied reasoning is transitioning from high-latency, high-performance paradigms toward compact and real-time approaches. For instance, Fast-ThinkAct (Huang et al., 2026) introduces verbalizable latent planning, where a compact latent CoT is distilled from a teacher model via teacher-student distillation, achieving up to 89.3% reduction in inference latency. In highly compositional dynamic scenarios, objectcentric reasoning representations have proven effective in improving prediction accuracy for dynamic entities and enhancing agents’ ability to interact intelligently with them. Meta AI’s Vision-Language World Model (VLWM) (Chen et al., 2025a) exemplifies an alternative reasoning pathway that avoids pixel-level prediction altogether: it uses natural language as an abstract representation of world evolution, achieving extreme semantic compression through structures such as the Tree of Captions. Beyond latent-space reasoning, leveraging domain knowledge to infer goal states can also be beneficial. Puppeteer (Hansen et al., 2025a) directly employs expert trajectories from human motion capture data to train high-level goal synthesis for low-level whole-body humanoid controllers. COVR (Xia et al., 2026) proposes a bidirectional learning framework that accelerates RL training by using prior guidance from vision-language models (VLMs), while high-quality trajectories collected by the RL agent during environment interaction are used to fine-tune the VLM. In contrast, our proposed ResDreamer adopts a reasoning representation that is both highly efficient for real-time inference. Its latent rollout and pixel-space foresight facilitates efficient reasoning and feature extraction. Moreover, pixel-space signals offer interpretability and the additive property of residuals. Within a lightweight 50- 200M parameter scale, we demonstrate that even blurry yet task-relevant visual foresight provides substantial benefits.

![](images/a0dc7502024fce90d274bf56b1d2a65bd8aa32135e22d9fba5c9e8d60b8561c4.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph_Lower_Residual_Observations["Lower residual observation"]
        Z0["z₀ᵗ"] --> A["Δ"]
        H0["h₀ᵗ"] --> A
        A --> C["-"]
        C --> D["Image"]
    end

    subgraph_Imaginary_Hint_Observations["Imaginary hint observation"]
        H1["h₁ᵗ"] --> I["Open-loop imagination"]
        H0["h₀ᵗ"] --> J["Open-loop imagination"]
        H0 --> K["Image"]
    end

    subgraph_Raw_World_Observation["Raw world observation"]
        O["World"] --> P["Image"]
    end

    subgraph PBB^k["PPB^k"]
        Lrec["L_rec"] --> LrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrecLrec
    end

    subgraph_Encoder_Estimation["Reconstruction estimation of the corresponding signal"]
        Dyn_LDyn["L_dyn"] --> Rep_Lrep["L_rep"]
        Zt_Zt_k["z_t^k"] --> Lrep
        Zt_Zt_k --> Rep
        Zt_Zt_k --> Lrep
        Zt_Zt_k --> Rep
        Zt_Zt_k --> Lrep
        Zt_Zt_k --> Rep
        Zt_Zt_k --> Lrep
        Zt_Zt_k --> Rep
        Zt_Zt_k --> Lrep
        Zt_Zt_k --> Rep
        Zt_Zt_k --> Lrep
        Zt_Zt_k --> Rep
        Zt_Zt_k --> Lrep
    end

    subgraph_Stop_Gradient["Stop gradient"]
        E["Decoder"] --> F["Encoder"]
        G["Stop gradient"] --> H["Encoder"]
    end

    style Lower Residual Observations fill:#f9f,stroke:#333
    style Imaginary Hint Observations fill:#ccf,stroke:#333
    style Raw World Observation fill:#fff,stroke:#333
```
</details>

Figure 1. Overview of ResDreamer a model base RL algorithm based on hierarchical world model. The left side shows the structure of enhanced visual observations. Adjacent world model layers communicate by residual and predictive signal within the enhanced observation. The right side shows the modules and training process of the k-th layer world model. The Encoder reads enhanced visual observations and gives the posterior $z _ { t . } ^ { k } .$ The dynamic predictor learns to estimate $z _ { t } ^ { k }$ with $\hat { z } _ { t } ^ { k }$ without accessing the observation. The sequence model updates internal state $h _ { t } ^ { k }$ by $z _ { t } ^ { k }$ . The Decoder reconstructs the observation signal which generates reconstruction loss and residual visual signal for upper layer.

Compared to Hierarchical RL. Hierarchical reinforcement learning (HRL) is widely regarded as a promising approach for mitigating exploration stagnation induced by sparse rewards in long-horizon tasks. A central focus in HRL is sub-goal discovery. Classic goal-conditioned policies obtain sub-goals either by extracting them from successful trajectories in the replay buffer (e.g., HER) or by treating sub-goals as high-level policies in a broader sense (e.g., HIRO). Another key direction involves learning world dynamics models at multiple temporal scales. For example, THICK (Gumbsch et al., 2024) adaptively discovers larger temporal abstractions by guiding lower-level world models to sparsely update their partial latent states. Recent work has explored more efficient sub-goal discovery algorithms and representations (Hafner et al., 2022; Hamed et al., 2024; Hansen et al., 2025b). The motivation of ResDreamer differs fundamentally from these prior HRL approaches. ResDreamer primarily focuses on layered hierarchical representation learning of world dynamics and leverages the residual-enhanced reasoning representations to boost performance of RL tasks.

# 3. Method

In this section, we present the details of ResDreamer. We introduce ResDreamer from the perspectives of representation learning and behavior learning. First, we describe the basic module of each layer in our Hierarchical Recurrent State-Space Model (HRSSM). Next, we present our primary innovation in representation learning architecture, namely the enhanced observation through residual connection. Finally, we formalize the loss functions and the overall training algorithm.

# 3.1. Hierarchical World Model

We implement the HRSSM based on Predictive Processing Blocks (PPBs). Predictive Processing or Predictive coding is a paradigm to explain hierarchical reciprocally connected organization of the cortex (Huang & Rao, 2011).

![](images/95307a2067c8432ba36a0e594eb6acb12829c76b4fb0fd2f53ac8152713c36a1.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    subgraph Prediction feedforward channel
        A["+"] --> B["htk+1"]
        C["+"] --> D["htk"]
        E["+"] --> F["htk"]
        G["+"] --> H["htk"]
        I["+"] --> J["htk"]
        K["+"] --> L["htk"]
        M["+"] --> N["htk"]
        O["+"] --> P["htk"]
        Q["+"] --> R["htk"]
        S["+"] --> T["htk"]
        U["+"] --> V["htk"]
        W["+"] --> X["htk"]
        Y["+"] --> Z["htk"]
        AA["+"] --> AB["htk"]
        AC["+"] --> AD["htk"]
        AE["+"] --> AF["htk"]
    end

    subgraph Error feedforward channel
        AG["htk+1"] --> AH["ztk+1"]
        AI["htk"] --> AJ["ztk+1"]
        AK["htk"] --> AL["ztk+1"]
        AM["htk"] --> AN["ztk+1"]
        AO["htk"] --> AP["ztk+1"]
        AQ["htk"] --> AR["ztk+1"]
        AS["htk"] --> AT["ztk+1"]
        AU["htk"] --> AV["ztk+1"]
        AW["htk"] --> AX["ztk+1"]
        AY["htk+1"] --> AZ["hkt+1"]
        BA["htk"] --> BB["hkt+1"]
        BC["htk"] --> BD["hkt+1"]
        BE["htk"] --> BF["hkt+1"]
        BG["htk"] --> BH["hkt+1"]
        BI["htk"] --> BJ["hkt+1"]
        BK["htk"] --> BL["hkt+1"]
        BM["htk"] --> BN["hkt+1"]
        BO["htk"] --> BP["hkt+1"]
        BQ["htk"] --> BR["hkt+1"]
        BS["htk"] --> BT["hkt+1"]
        BU["htk"] --> BV["hkt+1"]
        BW["htk"] --> BX["hkt+1"]
        BY["htk"] --> BZ["hkt+1"]
        CA["htk"] --> CB["hkt+1"]
        CC["htk"] --> CD["hkt+1"]
        DD["hkt+1"] --> DP["hkt+1"]
        DP --> DP2["hkt+1"]
        DP2 --> DP3["hkt+1"]
        DP3 --> DP4["hkt+1"]
        DP4 --> DP5["hkt+1"]
        DP5 --> DP6["hkt+1"]
        DP6 --> DP7["hkt+1"]
    end

    subgraph Current state or imagination entry
        BQ["hkt+1"] --> BZ
        BZ --> BZ2["hkt+1"]
        BZ2 --> BZ3["hkt+1"]
        BZ3 --> BZ4["hkt+1"]
        BZ4 --> BZ5["hkt+1"]
        BZ5 --> BZ6["hkt+1"]
        BZ6 --> BZ7["hkt+1"]
    end

    subgraph Open-loop imagination
        DQ["hkt+1"] --> DZ
        DZ --> DZ2["hkt+1"]
        DZ2 --> DZ3["hkt+1"]
        DZ3 --> DZ4["hkt+1"]
        DZ4 --> DZ5["hkt+1"]
        DZ5 --> DZ6["hkt+1"]
    end

    subgraph Open-loop imagination
        EQ["hkt+1"] --> EQ
        EQ --> EQ2["hkt+1"]
        EQ2 --> EQ3["hkt+1"]
        EQ3 --> EQ4["hkt+1"]
        EQ4 --> EQ5["hkt+1"]
    end

    style Prediction feedforward channel fill:#f9f,stroke:#333
    style Error feedforward channel fill:#ccf,stroke:#333
    style Open-loop imagination fill:#cfc,stroke:#333
```
</details>

Figure 2. The information channel between world model layers is bidirectional. Only reconstruction error and modulated foresight images are transmitted between layers, with no gradients being passed. On one hand, each layer of the PPB generates predictions about the external world and transmits visual planning representations to lower layers. On the other hand, the PPB treats low-level residuals as self-supervised learning signals to obtain a more complete inner representation.

In the k-th layer block $\mathrm { P P B } ^ { k }$ , recurrent state contains the deterministic state $h _ { t } ^ { k }$ and the stochastic state $z _ { t } ^ { k }$ . The sequence model is used to represent the state transitions conditioned by action taken. The Encoder extracts useful information from the new input observations to guide the recurrent state update, while the Predictor attempts to predict the stochastic state without accessing the observations.

$$
\mathrm{PPB} ^ {k} \left\{ \begin{array}{l l} \text {Sequence model:} & h _ {t} ^ {k} = S _ {\phi} \left(z _ {t - 1} ^ {k}, h _ {t - 1} ^ {k}, a _ {t - 1}\right) \\ \text {Encoder:} & z _ {t} ^ {k} \sim q _ {\phi} \left(z _ {t} ^ {k} \mid h _ {t} ^ {k}, o _ {t} ^ {k}\right) \\ \text {Predictor:} & \hat {z} _ {t} ^ {k} \sim p _ {\phi} \left(\hat {z} _ {t} ^ {k} \mid h _ {t} ^ {k}\right) \\ \text {Decoder:} & \hat {o} _ {t} ^ {k} \sim D _ {\phi} \left(\hat {o} _ {t} ^ {k} \mid h _ {t} ^ {k}, z _ {t} ^ {k}\right). \end{array} \right. \tag {1}
$$

where $\hat { z } _ { t } ^ { k }$ is the predicted stochastic state, $o _ { t } ^ { k }$ and $\hat { o } _ { t } ^ { k }$ are true and reconstructed observations. Layer index $k \mathbf { \Sigma } =$ $0 , 1 , \cdots L - 1$ and L is the number of HRSSM layers.

# 3.2. Visual Hint Structure and Residual Modeling

Figure 1 gives an overview of ResDreamer, a hierarchical world model in which layers communicate through error feedback and predictive visual hints. This section elaborates on the forms of information exchange between the world model layers. ResDreamer is characterized by progressive residual learning of sensory signals and image foresight corrected by residual prediction.

Sensory reconstruction error is fed into the higher-level world model for residual learning. The lower residual observation is given by

$$
o _ {\text {res}} ^ {1} = \operatorname{Norm} ^ {1} \left(o _ {\text {raw}} - \hat {o} _ {\text {raw}}\right), \tag {2}
$$

$$
o _ {\mathrm{res}} ^ {k} = \mathrm{Norm} ^ {k} \left(o _ {\mathrm{res}} ^ {k - 1} - \hat {o} _ {\mathrm{res}} ^ {k - 1}\right),
$$

where $k = 2 , 3 , \cdots , L - 1$ , the omitted time indices are all $t ,$ and the same applies hereafter. Layer 0 only has environmental observations $O _ { \mathrm { { r a w } } }$ and provides its reconstruction $\hat { O } _ { \mathrm { r a w } } ,$ , and has no residual observations. $\mathrm { N o r m } ^ { k } ( \cdot )$ computes the mean and variance across the pixel dimension and updates them with exponential moving average.

It is worth noting that any layer of the well-trained PPB can rollout latent trajectories by replacing the posterior with the prior. This means that as long as PPB is trained to model the lower residual, it can perform open-loop reasoning and correct the visual reasoning representations at the lower level. The imaginary hint observation is given by

$$
o _ {\mathrm{imag}} ^ {0} = \left\{\hat {o} _ {\mathrm{raw}} + \hat {o} _ {\mathrm{res}} ^ {1} \right\} _ {t: t + H},
$$

$$
o _ {\text { imag }} ^ {k} = \left\{\hat {o} _ {\text { res }} ^ {k} + \hat {o} _ {\text { res }} ^ {k + 1} \right\} _ {t: t + H}, \tag {3}
$$

$$
o _ {\mathrm{imag}} ^ {L - 1} = \left\{\hat {o} _ {\mathrm{res}} ^ {L - 1} \right\} _ {t: t + H}.
$$

where $k = 1 , 2 , \cdots , L - 2$ , the subscript $\{ \cdot \} _ { t : t + H }$ stands for concatenation of reconstructed open-loop imagination for future time $t , t + 1 , \cdots , t + H - 1$ . Specifically, if the raw image shape is $( h , w , 3 )$ , then $\{ \hat { o } _ { \mathrm { r e s } } ^ { k } \} _ { t : t + H }$ and $\{ \bar { o } _ { \mathrm { i m a g } } ^ { k } \} _ { t : t + H }$ decoded from a H steps latent trajectory both have the shape of $( H , h , w , 3 )$ . They are added by residual connections and concatenated along the channel axis into shape $( h , w , 3 H )$ .

![](images/42b90d35e72bd44899c3c6e044476b7088aa02528200ff97c8330d3077d714f5.jpg)

<details>
<summary>flowchart</summary>

```mermaid
graph TD
    A["Environment step"] --> B["Top Level Encoder"]
    B --> C["Bottom Level Encoder"]
    C --> D["Top Level Encoder"]
    D --> E["Bottom Level Encoder"]
    E --> F["Top Level Encoder"]
    F --> G["Bottom Level Encoder"]
    G --> H["Top Level Encoder"]
    H --> I["Bottom Level Encoder"]
    I --> J["Top Level Encoder"]
    J --> K["Bottom Level Encoder"]
    K --> L["Top Level Encoder"]
    L --> M["Bottom Level Encoder"]
    M --> N["Top Level Encoder"]
    N --> O["Bottom Level Encoder"]
    O --> P["Top Level Encoder"]
    P --> Q["Bottom Level Encoder"]
    Q --> R["Top Level Encoder"]
    R --> S["Bottom Level Encoder"]
    S --> T["Top Level Encoder"]
    T --> U["Bottom Level Encoder"]
    U --> V["Top Level Encoder"]
    V --> W["Bottom Level Encoder"]
    W --> X["Top Level Encoder"]
    X --> Y["Bottom Level Encoder"]
    Y --> Z["Top Level Encoder"]
    Z --> AA["Bottom Level Encoder"]
    AA --> AB["Top Level Encoder"]
    AB --> AC["Bottom Level Encoder"]
    AC --> AD["Top Level Encoder"]
    AD --> AE["Bottom Level Encoder"]
    AE --> AF["Top Level Encoder"]
    AF --> AG["Bottom Level Encoder"]
    AG --> AH["Top Level Encoder"]
    AH --> AI["Bottom Level Encoder"]
    AI --> AJ["Top Level Encoder"]
    AJ --> AK["Bottom Level Encoder"]
    AK --> AL["Top Level Encoder"]
    AL --> AM["Bottom Level Encoder"]
    AM --> AN["Top Level Encoder"]
    AN --> AO["Bottom Level Encoder"]
    AO --> AP["Top Level Encoder"]
    AP --> AQ["Bottom Level Encoder"]
    AQ --> AR["Top Level Encoder"]
    AR --> AS["Bottom Level Encoder"]
    AS --> AT["Top Level Encoder"]
    AT --> AU["Bottom Level Encoder"]
    AU --> AV["Top Level Encoder"]
    AV --> AW["Bottom Level Encoder"]
    AW --> AX["Top Level Encoder"]
    AX --> AY["Bottom Level Encoder"]
    AY --> AZ["Top Level Encoder"]
    AZ --> BA["Bottom Level Encoder"]
    BA --> BB["Top Level Encoder"]
    BB --> BC["Bottom Level Encoder"]
    BC --> BD["Top Level Encoder"]
    BD --> BE["Bottom Level Encoder"]
    BE --> BF["Top Level Encoder"]
```
</details>

Figure 3. Visualization of residual-modulation mechanism of a two-layer world model example. The trajectory clip shows a ghast switching to attack mode (red body) and launching a bomb. The agent is able to anticipate the bomb before it appears through visual reasoning. As the bomb fell, the agent continues to retreat.

The imaginary hint observation utilities the hierarchical structure of the world model to build visual reasoning representation. The foresight is blurry but informative and expressive. See Figure 3 which shows the raw observation and imaginary hint observation on world model bottom layer while the agent is combating a ghast.

The observation expanded by lower-level residuals and upper-level predictions is the key to our hierarchical world model:

$$
o _ {t} ^ {k} = \operatorname{sg} \left(\left\{o _ {\text { imag }} ^ {k}, o _ {\text { raw }}, o _ {\text { res }} ^ {k} \right\} _ {t}\right). \tag {4}
$$

where $\operatorname { s g } ( \cdot )$ is stop gradient operation. In our implementation, all the observations are concatenated along the channel axis. Eventually, a complete enhanced observation tensor is formed with the shape of $( h , w , 3 H + 6 )$ .

The imaginary hint observation is merely additional guidance or enhancement for the encoder. Only the lower residual observation and environmental input are reconstructed by the decoder and provide reconstruction loss during the training $\hat { o } _ { t } ^ { k } = \mathrm { s g } \left( \bar { \left\{ { { \hat { o } } _ { \mathrm { r a w } } , { { \hat { o } } _ { \mathrm { r e s } } ^ { k } } } \right\} } _ { t } \right)$ . The complete process of constructing enhanced observations from the bottom layer to the top layer and updating the recursive state in sequence is shown in Algorithm 1.

At this point, we have established the feedforward and feedback information channels of the hierarchical world model based on the enhanced observation (see Figure 2). This architecture combines the bandwidth advantage of inter-layer communication and the computational efficiency advantage within layers.

The visual hint does incur necessary computational cost, but from the perspective of parameter scale, the above architecture introduces almost no overhead. Within each layer, although the number of image channels has significantly increased due to the addition of video hint, the distribution of the visual hint is highly matched with the original image distribution benefits from the residual modeling, thus allowing for the sharing of major convolutional features. Therefore, in practice, we have not expanded the depth of the encoder and dimensions of stochastic state compared to DreamerV3 (Hafner et al., 2025).

# 3.3. World Model and Behavior Learning

In the context of RL, the final goal is to improve the policy. The actor-critic method is employed for policy optimization and state value learning.

$$
\begin{array}{l l} \text { Actor: } & a _ {t} \sim \pi_ {\theta} (a _ {t} \mid s _ {t}) \\ \text { C. } & \text { (1) } \end{array} \tag {5}
$$

$$
\mathrm{Critic:} v _ {t} \sim v _ {\psi} (v _ {t} | s _ {t})
$$

The world environment generously provides continuous stream of sensory signals. Reconstructing sensory inputs serves as a critical training signal for world models. This drives the model to encode as much environmental information as possible in deterministic state.

$$
\mathcal {L} _ {\mathrm{rec}} ^ {k} (\phi) = - \ln p _ {\phi} \left(o _ {\mathrm{raw}} \mid s _ {t} ^ {k}\right) - \ln p _ {\phi} \left(o _ {\mathrm{res}} ^ {k} \mid s _ {t} ^ {k}\right). \tag {6}
$$

The stochastic state serves as the information channel between observation and latent state representation. The enforced sparsity makes the stochastic state more predictable, while the representation loss ensures that the posterior distribution converges to a more predictable distribution.

$$
\begin{array}{l} \mathcal {L} _ {\mathrm{dyn}} ^ {k} (\phi) = \mathrm{KL} \left[ \mathrm{sg} \left(q _ {\phi} \left(z _ {t} ^ {k} \mid h _ {t} ^ {k}, o _ {t} ^ {k}\right)\right) | | p _ {\phi} \left(z _ {t} ^ {k} \mid h _ {t} ^ {k},\right) \right] \\ \mathcal {L} _ {\text { rep }} ^ {k} (\phi) = \mathrm{KL} \left[ q _ {\phi} \left(z _ {t} ^ {k} \mid h _ {t} ^ {k}, o _ {t} ^ {k}\right) | | \operatorname{sg} \left(p _ {\phi} \left(z _ {t} ^ {k} \mid h _ {t} ^ {k}\right)\right) \right] \tag {7} \\ \end{array}
$$

Additional prediction heads perform reward modeling $\hat { r } _ { t } \sim$ $p _ { \phi } \left( \hat { r } _ { t } \mid s _ { t } \right)$ and episode-continuation flag $\hat { c } _ { t } \sim p _ { \phi } \left( \hat { c } _ { t } ^ { k } \mid s _ { t } \right)$ are also trained in a self-supervised manner:

$$
\mathcal {L} _ {\text { heads }} (\phi) = - \ln p _ {\phi} (r _ {t} \mid s _ {t}) - \ln p _ {\phi} (c _ {t} \mid s _ {t}) \tag {8}
$$

The actor-critic is trained purely on imagined trajectories. We compute the bootstrapped λ-return $R _ { t } ^ { \lambda }$ to train the critic. $R _ { t } ^ { \lambda }$ accounts for $r _ { t }$ within the trajectory horizon T and incorporates the critic’s expected value for returns beyond the horizon.

$$
\mathcal {L} (\psi) = - \sum_ {t = 1} ^ {T} \ln p _ {\psi} \left(R _ {t} ^ {\lambda} \mid s _ {t}\right)
$$

$$
R _ {t} ^ {\lambda} = \left\{ \begin{array}{l l} r _ {t} + \gamma c _ {t} ((1 - \lambda) v _ {t} + \lambda R _ {t + 1} ^ {\lambda}), & t <   T \\ \mathbb {E} [ v _ {\psi} (\cdot | s _ {t}) ], & t = T \end{array} \right. \tag {9}
$$

The actor learns to maximize returns with entropy regularizer:

$$
\mathcal {L} (\theta) = - \sum_ {t = 1} ^ {T} \frac {R _ {t} ^ {\lambda} - \operatorname{sg} \left(v _ {\psi} \left(s _ {t}\right)\right)}{\max (1 , S)} \log \pi_ {\theta} \left(a _ {t} \mid s _ {t}\right) \tag {10}
$$

$$
+ \eta \mathrm{H} \left[ \pi_ {\theta} \left(a _ {t} \mid s _ {t}\right) \right]
$$

where $S \ = \ \mathrm { E M A } \left( \mathrm { P e r } \left( R _ { t } ^ { \lambda } , 9 5 \right) - \mathrm { P e r } \left( R _ { t } ^ { \lambda } , 5 \right) \right)$ is the scale from the 5% quantile to the 95% quantile maintained by the exponential moving average. The scale factor S is used to normalize the return and is clipped to be no less than 1 to prevent noise amplification. Overall, we follow the official DreamerV3 Hafner et al. (2025) hyperparameters and implementation details. Benefits from DreamerV3’s robust generalization capabilities, ResDreamer can be trained across diverse tasks without hyperparameter tuning.

Residual Enhanced Visual Observation is the main innovation of this work. To visually demonstrate the structure of this visual foresight and the planning information it provides, we visualize the bottom layer observation in Figure 3.

# 4. Experiments

Engaging in combat within open-ended 3D worlds poses substantial challenges, requiring robust terrain understanding, effective use of weapons and defensive tools, and realtime anticipation of adversarial enemy movements and behaviors. We evaluate ResDreamer on five combat tasks from the MineDojo benchmark (Fan et al., 2022) detailed in Table 3. These tasks span a range of difficulty levels and involve fighting various hostile entities under diverse initial inventories and lighting setups.

The only non-trivial hyperparameter introduced by Res-Dreamer is the rollout horizon H for image foresight predictions. To assess its impact, along with the complementary time stride D (the interval between predicted frames), we conduct a sensitivity analysis on a suite of visual continuous control tasks from the DeepMind Control Suite (DMC) (Ortiz et al., 2024) with pixel observations

# 4.1. Main Comparison

We evaluate all methods using success rates during training. The training curves in Figure 4 and bar charts in Figure 8 demonstrate that ResDreamer achieves superior sample efficiency as a hierarchical model-based RL method. To enable a fair assessment of parameter efficiency, we test ResDreamer under two configurations:

• ResDreamer (100M×2): a two-layer hierarchical model with approximately 100 million parameters per layer (total 200M). This variant exhibits the strongest sample efficiency and fastest convergence across all evaluated tasks.   
• ResDreamer (50M×2): a lighter two-layer model. It surpassed the average performance of DreamerV3 with only 84% of the parameter size.

The detailed parameter size, hyper parameter and compute budget is shown in Table 1. All the baselines use the default configuration of the official implementation. Further details are provided in Appendix A.

![](images/a9ee6b8d0dc9ab29b69a0d5241aee98f8a82402884fb56319119d129ac4cfaa8.jpg)

<details>
<summary>text_image</summary>

- - Steve1 • DreamerV3 (100M) • PTGM • ResDreamer (50Mx2) • ResDreamer (100Mx2)
</details>

![](images/c76a9a821a5b9aa09b8406afef3ff07022eded37c4ba7e9baacc30811460e878.jpg)

<details>
<summary>line</summary>

| Step   | Blue Line | Red Line | Yellow Line | Green Line |
| ------ | --------- | -------- | ----------- | ---------- |
| 0      | 0.2       | 0.2      | 0.2         | 0.1        |
| 250K   | 0.6       | 0.7      | 0.7         | 0.1        |
| 500K   | 0.7       | 0.7      | 0.7         | 0.1        |
| 750K   | 0.7       | 0.8      | 0.8         | 0.3        |
| 1.0M   | 0.8       | 0.8      | 0.8         | 0.25       |
</details>

![](images/add873201614c1bc22c9f1b75fb268f14b8ea67b9145665198438062427edcb4.jpg)

<details>
<summary>line</summary>

| Step   | Combat shulker |
| ------ | -------------- |
| 0      | 0.0            |
| 250K   | 0.0            |
| 500K   | 0.1            |
| 750K   | 0.2            |
| 1.0M   | 0.3            |
</details>

![](images/33555709e7203d6c1f014eef216f72b8ab66c4b0a03d094a1ba249b74b0ed1f6.jpg)

<details>
<summary>line</summary>

| Step   | Red Line | Yellow Line | Green Line | Blue Line |
| ------ | -------- | ----------- | ---------- | --------- |
| 0      | 0.05     | 0.05        | 0.05       | 0.05      |
| 250K   | 0.10     | 0.10        | 0.10       | 0.05      |
| 500K   | 0.15     | 0.15        | 0.15       | 0.05      |
| 750K   | 0.20     | 0.20        | 0.20       | 0.10      |
| 1.0M   | 0.25     | 0.25        | 0.25       | 0.15      |
</details>

![](images/269c39647ed5f7e0a132017076e76c9468002194882dbb3861a6ae7c484bb9ba.jpg)

<details>
<summary>line</summary>

| Environment Steps | Success Rate (Blue) | Success Rate (Red) | Success Rate (Green) |
| ----------------- | ------------------- | ------------------ | -------------------- |
| 0                 | 0.2                 | 0.2                | 0.1                  |
| 250K              | 0.8                 | 0.8                | 0.6                  |
| 500K              | 0.85                | 0.75               | 0.65                 |
| 750K              | 0.88                | 0.85               | 0.6                  |
| 1.0M              | 0.87                | 0.85               | 0.6                  |
</details>

![](images/5f6a0ec8895915452e057e494a2bd43823b0e7b655e3cdd2927c3d7dd61a35e6.jpg)

<details>
<summary>line</summary>

| Environment Steps | Red Line | Yellow Line | Blue Line | Green Line |
| ----------------- | -------- | ----------- | --------- | ---------- |
| 0                 | 0.0      | 0.0         | 0.0       | 0.0        |
| 250K              | 0.2      | 0.3         | 0.1       | 0.0        |
| 500K              | 0.5      | 0.6         | 0.3       | 0.0        |
| 750K              | 0.6      | 0.7         | 0.5       | 0.0        |
| 1.0M              | 0.7      | 0.7         | 0.7       | 0.0        |
</details>

![](images/2196cc6a3bd76dfdeb7ec2daaa507a275beedc0580cdbb98126d8e004b3ac62e.jpg)

<details>
<summary>line</summary>

| Environment Steps | Yellow Line | Red Line | Blue Line | Green Line |
| ----------------- | ----------- | -------- | --------- | ---------- |
| 0                 | 0.15        | 0.15     | 0.15      | 0.10       |
| 250K              | 0.35        | 0.30     | 0.28      | 0.20       |
| 500K              | 0.45        | 0.40     | 0.35      | 0.22       |
| 750K              | 0.55        | 0.50     | 0.45      | 0.23       |
| 1.0M              | 0.60        | 0.55     | 0.52      | 0.22       |
</details>

Figure 4. Comparison of ResDreamer against Steve-1 (Lifshitz et al., 2023), DreamerV3 (Hafner et al., 2025), PTGM (Yuan et al., 2024). We introduce the compared models in Appendix D. Results of transformer based MBRL method IRIS (Micheli et al., 2022) is presented in Appendix C.1

ResDreamer (100M×2) is the only method among the evaluated baselines that successfully solves the high-difficulty Shulker combat task within $1 \times 1 0 ^ { 6 }$ environment steps. The hostile mob Shulker launches a guided projectile that causes prolonged levitation and causes fall damage afterwards. This poses complex and challenging dynamic interaction mechanisms for the agent. Our analysis shows that the residual-modulated visual reasoning representation in ResDreamer provides critical assistance in mastering this interaction. The reasoning representation supplies modulated foresight images that enrich the observation stream with predictive signals about impending changes (e.g., incoming projectiles, see also Figure 3), making the input more informative for decision-making.

# 4.2. Model Analysis

The foresight rollout and residual connection mechanism are the key components of ResDreamer. We present ablation study to isolate the contribution from each design.

# 4.2.1. ABLATION STUDY

The residual connections in ResDreamer enable the flow of both predictive signals and reconstruction errors across hierarchical layers, forming a key mechanism for enhanced visual reasoning. Figure 5 presents ablation results comparing the standard ResDreamer against several alternative configurations.

ResDreamer 50Mx3. We extend the main two-layer model (50M×2) to three layers while maintaining a similar perlayer parameter budget. The three-layer variant achieves a higher mean task success rate than the two-layer version. This demonstrates that ResDreamer’s residual hierarchy provides an effective and scalable pathway for world models.

ResDreamer Only Residual.In this setup, imaginary hint observations consist exclusively of upper-layer residual signals, excluding the current layer’s own predictive reconstruction. Although the current layer’s latent state already encompasses complete information from open-loop predictions, we find that blending predictive reconstruction with residual signals yields superior performance.

ResDreamer W/O Residual. We remove residual connections entirely, so actor-critic and prediction heads only access upper layer latent state for additional information. Task performance drops markedly, underscoring that residualmodulated visual foresight is the core component that improves performance.

![](images/1ae4fd5e7c49a3758e5c5a2556ebb8c6483064e3c1b91cbee3fcca2ee99f31b0.jpg)

<details>
<summary>text_image</summary>

ResDreamer 50Mx3
ResDreamer 50Mx2
ResDreamer Only Residual
DreamerV3 + rollout
ResDreamer W/O Residual
ResDreamer Stacked State
</details>

![](images/fa960fc3e9f97e86627a878003feac44627286df0b4a26df960d0fdb87eec1b8.jpg)

<details>
<summary>line</summary>

| Environment Steps | Success Rate (Line 1) | Success Rate (Line 2) | Success Rate (Line 3) | Success Rate (Line 4) | Success Rate (Line 5) |
| ----------------- | --------------------- | --------------------- | --------------------- | --------------------- | --------------------- |
| 0                 | 0.1                   | 0.1                   | 0.1                   | 0.1                   | 0.1                   |
| 250K              | 0.4                   | 0.4                   | 0.3                   | 0.3                   | 0.2                   |
| 500K              | 0.6                   | 0.6                   | 0.5                   | 0.5                   | 0.4                   |
| 750K              | 0.7                   | 0.7                   | 0.6                   | 0.6                   | 0.5                   |
| 1.0M              | 0.8                   | 0.8                   | 0.7                   | 0.7                   | 0.6                   |
</details>

Figure 5. ResDreamer ablation study results.

ResDreamer Stacked State. The actor, critic, and prediction heads in ResDreamer are conditioned on the stacked latent states of all layers. Despite the stacked states theoretically containing richer recursive information, performance declines under 1M environment steps. We assume that this degradation arises because the distribution of lower-layer residuals shifts during training, causing instability in upperlayer representations before full convergence. Future work could explore longer training regimes or adaptive normalization to mitigate this effect.

DreamerV3 + rollout. We augment a standard single-layer DreamerV3 baseline by adding online-computed visual foresight rollouts and reconstructions as additional grounded reasoning observations. Despite this enhancement, performance remains below ResDreamer, further validating the synergistic benefits of the hierarchical architecture combined with residual connections.

These ablations collectively isolate the contributions of residual modulation and hierarchical depth, confirming that both elements are essential for ResDreamer’s superior performance.

# 4.2.2. FORESIGHT HORIZON SENSITIVITY ANALYSIS

We evaluate ResDreamer using foresight horizons H = 4, 8, 16 and strides D = 1, 2, 4 respectively, keeping the total number of predicted frames fixed at 4. Experiments are conducted on visual ontinuous control suite DMC Vision.

As shown in Figure 6, longer horizon and larger stride eventually converge to higher average performance. However, excessively long horizons can slow early-stage convergence on some tasks due to initially noisy predictions.

ResDreamer consistently outperforms DreamerV3 under different horizon configurations. In DMC Hopper Hop, H = 16 achieves the best results, as 16 steps align closely with a full hopping cycle, enabling better anticipation of balance-critical dynamics.

Overall, longer and coarser foresight proves more informative for complex dynamics after sufficient training, though shorter horizons may accelerate early-stage learning.

# 5. Conclusion

In this paper, we present ResDreamer, a hierarchical world model that employs residually connected visual planning representations. By modeling reconstruction residuals, each layer passes only the novel, unexpected sensory signals upward, creating an bandwidth efficient information channel between layers. Residual rollouts from upper layers modulate the visual foresight at lower levels, enriching the predictive reasoning representation.

Extensive comparisons and ablation studies demonstrate that ResDreamer achieves superior sample efficiency and parameter efficiency in challenging online visual RL tasks. Critically, the synergy of hierarchical structure, residual modulation, and modulated image foresight proves substantially more effective than any subset of these components

The primary limitation of the current approach is the fixed foresight horizon length. Longer static horizons incur higher computational cost, while overly short horizons may fail to capture sufficient long-range dynamics. Developing adaptive or learnable horizon mechanisms remains an important direction for future work.

Overall, ResDreamer’s task-agnostic reasoning representation readily adapts to any visual RL scenario, establishing a powerful and broadly applicable baseline for imaginationdriven MBRL across discrete and continuous action spaces. It defines the frontier of online RL agents in 3D open-ended dynamic interactive environments.

![](images/a4ba11ea313a09d025c8ff595bbdac34419a794babdb09846b2ed018b0afc60f.jpg)  
. ResDreamer (H=4, D=1) ResDreamer (H=8, D=2) · ResDreamer (H=16, D=4) dreamerv3

Figure 6. Comparison of ResDreamer with different foresight horizon on DMC Vision (Ortiz et al., 2024) continuous control suite.

# Impact Statement

This paper presents work whose goal is to advance the field of Machine Learning. There are many potential societal consequences of our work, none which we feel must be specifically highlighted here.

In this work we adhere to the code of ethics. This work does not involve human subjects, personal data, or sensitive information.

# References

Alonso, E., Jelley, A., Micheli, V., Kanervisto, A., Storkey, A. J., Pearce, T., and Fleuret, F. Diffusion for world modeling: Visual details matter in atari. Advances in Neural Information Processing Systems, 37:58757–58791, 2024.   
Bruce, J., Dennis, M. D., Edwards, A., Parker-Holder, J., Shi, Y., Hughes, E., Lai, M., Mavalankar, A., Steigerwald, R., Apps, C., et al. Genie: Generative interactive environments. In Forty-first International Conference on Machine Learning, 2024.   
Cai, S., Mu, Z., Liu, A., and Liang, Y. Rocket-2: Steering

visuomotor policy via cross-view goal alignment. arXiv preprint arXiv:2503.02505, 2025a.

Cai, S., Mu, Z., Xia, H., Zhang, B., Liu, A., and Liang, Y. Scalable multi-task reinforcement learning for generalizable spatial intelligence in visuomotor agents, 2025b. URL https://arxiv.org/abs/2507.23698.

Chen, D., Moutakanni, T., Chung, W., Bang, Y., Ji, Z., Bolourchi, A., and Fung, P. Planning with reasoning using vision language world model. arXiv preprint arXiv:2509.02722, 2025a.

Chen, H., Zhao, M., Yang, R., Ma, Q., Yang, K., Yao, J., Wang, K., Bai, H., Wang, Z., Pan, R., et al. Era: Transforming vlms into embodied agents via embodied prior learning and online reinforcement learning. arXiv preprint arXiv:2510.12693, 2025b.

Chu, H., Deng, X., Lv, Q., Chen, X., Li, Y., Hao, J., and Nie, L. 3d-affordancellm: Harnessing large language models for open-vocabulary affordance detection in 3d worlds. arXiv preprint arXiv:2502.20041, 2025.

Deng, J., Wang, Z., Cai, S., Liu, A., and Liang, Y. Openworld skill discovery from unsegmented demonstrations. arXiv preprint arXiv:2503.10684, 2025.

Duan, Z., Zhang, Y., Geng, S., Liu, G., Boedecker, J., and Lu, C. X. Fast ecot: Efficient embodied chain-of-thought via thoughts reuse. arXiv preprint arXiv:2506.07639, 2025.   
Fan, L., Wang, G., Jiang, Y., Mandlekar, A., Yang, Y., Zhu, H., Tang, A., Huang, D.-A., Zhu, Y., and Anandkumar, A. Minedojo: Building open-ended embodied agents with internet-scale knowledge. In Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2022. URL https: //openreview.net/forum?id=rc8o\_j8I8PX.   
Gumbsch, C., Sajid, N., Martius, G., and Butz, M. V. Learning hierarchical world models with adaptive temporal abstractions from discrete latent dynamics. In The Twelfth International Conference on Learning Representations, 2024. URL https://openreview.net/forum? id=TjCDNssXKU.   
Ha, D. and Schmidhuber, J. Recurrent world models facilitate policy evolution. Advances in neural information processing systems, 31, 2018.   
Hafner, D., Lee, K.-H., Fischer, I., and Abbeel, P. Deep hierarchical planning from pixels. Advances in Neural Information Processing Systems, 35:26091–26104, 2022.   
Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. Mastering diverse control tasks through world models. Nature, pp. 1–7, 2025.   
Hamed, H., Kim, S., Kim, D., Yoon, J., and Ahn, S. Dr. strategy: Model-based generalist agents with strategic dreaming. In International Conference on Machine Learning, 2024.   
Hansen, N., SV, J., Sobal, V., LeCun, Y., Wang, X., and Su, H. Hierarchical world models as visual wholebody humanoid controllers. In International Conference on Learning Representations, volume 2025, pp. 62175– 62195, 2025a.   
Hansen, N., SV, J., Sobal, V., LeCun, Y., Wang, X., and Su, H. Hierarchical world models as visual whole-body humanoid controllers, 2025b.   
Hosoya, T., Baccus, S. A., and Meister, M. Dynamic predictive coding by the retina. Nature, 436(7047):71–77, 2005.   
Huang, C.-P., Man, Y., Yu, Z., Chen, M.-H., Kautz, J., Wang, Y.-C. F., and Yang, F.-E. Fast-thinkact: Efficient vision-language-action reasoning via verbalizable latent planning. arXiv preprint arXiv:2601.09708, 2026.   
Huang, Y. and Rao, R. P. Predictive coding. Wiley Interdisciplinary Reviews: Cognitive Science, 2(5):580–593, 2011.

Kirillov, A., Mintun, E., Ravi, N., Mao, H., Rolland, C., Gustafson, L., Xiao, T., Whitehead, S., Berg, A. C., Lo, W.-Y., et al. Segment anything. In Proceedings of the IEEE/CVF international conference on computer vision, pp. 4015–4026, 2023.   
Kok, P. and de Lange, F. P. Predictive coding in sensory cortex. In An introduction to model-based cognitive neuroscience, pp. 221–244. Springer, 2015.   
Li, C., Krause, A., and Hutter, M. Robotic world model: A neural network simulator for robust policy optimization in robotics. arXiv preprint arXiv:2501.10100, 2025a.   
Li, J., Wang, Q., Wang, Y., Jin, X., Li, Y., Zeng, W., and Yang, X. Open-world reinforcement learning over long short-term imagination. arXiv preprint arXiv:2410.03618, 2024a.   
Li, M., Wang, Z., He, K., Ma, X., and Liang, Y. Jarvisvla: Post-training large-scale vision language models to play visual games with keyboards and mouse. arXiv:2503.16365, 2025b.   
Li, X., Zhang, M., Geng, Y., Geng, H., Long, Y., Shen, Y., Zhang, R., Liu, J., and Dong, H. Manipllm: Embodied multimodal large language model for object-centric robotic manipulation. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18061–18070, 2024b.   
Lifshitz, S., Paster, K., Chan, H., Ba, J., and McIlraith, S. Steve-1: A generative model for text-to-behavior in minecraft. Advances in Neural Information Processing Systems, 36:69900–69929, 2023.   
Liu, S., Yuan, H., Hu, M., Li, Y., Chen, Y., Liu, S., Lu, Z., and Jia, J. Rl-gpt: Integrating reinforcement learning and code-as-policy. Advances in Neural Information Processing Systems, 37:28430–28459, 2024.   
Micheli, V., Alonso, E., and Fleuret, F. Transformers are sample-efficient world models. arXiv preprint arXiv:2209.00588, 2022.   
NVIDIA, Ali, A., Bai, J., Bala, M., Balaji, Y., Blakeman, A., Cai, T., Cao, J., Cao, T., Cha, E., Chao, Y.-W., Chattopadhyay, P., Chen, M., Chen, Y., Chen, Y., Cheng, S., Cui, Y., Diamond, J., Ding, Y., Fan, J., Fan, L., Feng, L., Ferroni, F., Fidler, S., Fu, X., Gao, R., Ge, Y., Gu, J., Gupta, A., Gururani, S., El Hanafi, I., Hassani, A., Hao, Z., Huffman, J., Jang, J., Jannaty, P., Kautz, J., Lam, G., Li, X., Li, Z., Liao, M., Lin, C.-H., Lin, T.-Y., Lin, Y.-C., Ling, H., Liu, M.-Y., Liu, X., Lu, Y., Luo, A., Ma, Q., Mao, H., Mo, K., Nah, S., Narang, Y., Panaskar, A., Pavao, L., Pham, T., Ramezanali, M., Reda, F., Reed, S., Ren, X., Shao, H., Shen, Y., Shi, S., Song, S., Stefaniak, B., Sun, S., Tang, S., Tasmeen, S., Tchapmi, L.,

Tseng, W.-C., Varghese, J., Wang, A. Z., Wang, H., Wang, H., Wang, H., Wang, T.-C., Wei, F., Xu, J., Yang, D., Yang, X., Ye, H., Ye, S., Zeng, X., Zhang, J., Zhang, Q., Zheng, K., Zhu, A., and Zhu, Y. World simulation with video foundation models for physical ai, 2025. URL https://arxiv.org/abs/2511.00062.   
Ortiz, J., Dedieu, A., Lehrach, W., Guntupalli, J. S., Wendelken, C., Humayun, A., Swaminathan, S., Zhou, G., Lazaro-Gredilla, M., and Murphy, K. P. Dmc-vb: A ´ benchmark for representation learning for control with visual distractors. Advances in Neural Information Processing Systems, 37:6574–6602, 2024.   
Rao, R. P. and Ballard, D. H. Predictive coding in the visual cortex: a functional interpretation of some extra-classical receptive-field effects. Nature neuroscience, 2(1):79–87, 1999.   
Robine, J., Hoftmann, M., Uelwer, T., and Harmeling, S.¨ Transformer-based world models are happy with 100k interactions. arXiv preprint arXiv:2303.07109, 2023.   
Schrittwieser, J., Antonoglou, I., Hubert, T., Simonyan, K., Sifre, L., Schmitt, S., Guez, A., Lockhart, E., Hassabis, D., Graepel, T., et al. Mastering atari, go, chess and shogi by planning with a learned model. Nature, 588(7839): 604–609, 2020.   
Wan, T., Wang, A., Ai, B., Wen, B., Mao, C., Xie, C.-W., Chen, D., Yu, F., Zhao, H., Yang, J., et al. Wan: Open and advanced large-scale video generative models. arXiv preprint arXiv:2503.20314, 2025.   
Wang, G., Xie, Y., Jiang, Y., Mandlekar, A., Xiao, C., Zhu, Y., Fan, L., and Anandkumar, A. Voyager: An openended embodied agent with large language models. arXiv preprint arXiv: Arxiv-2305.16291, 2023a.   
Wang, Z., Cai, S., Liu, A., Jin, Y., Hou, J., Zhang, B., Lin, H., He, Z., Zheng, Z., Yang, Y., Ma, X., and Liang, Y. Jarvis-1: Open-world multi-task agents with memoryaugmented multimodal language models. arXiv preprint arXiv: 2311.05997, 2023b.   
Wang, Z., Cai, S., Liu, A., Ma, X., and Liang, Y. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. arXiv preprint arXiv:2302.01560, 2023c.   
Xia, C., Peng, P., Tan, G., Su, Z., Xu, H., Liu, Z., and Li, L. Covr: Collaborative optimization of vlms and rl agent for visual-based control. arXiv preprint arXiv:2601.06122, 2026.   
Yuan, H., Mu, Z., Xie, F., and Lu, Z. Pre-training goalbased models for sample-efficient reinforcement learning. In The Twelfth International Conference on Learning Representations, 2024.

Zhan, G., Wang, L., Zhang, X., Gao, J., Tomizuka, M., and Li, S. E. Bootstrap off-policy with world model. arXiv preprint arXiv:2511.00423, 2025.   
Zhang, W., Wang, G., Sun, J., Yuan, Y., and Huang, G. Storm: Efficient stochastic transformer based world models for reinforcement learning. Advances in Neural Information Processing Systems, 36:27147–27166, 2023.   
Zhao, Q., Lu, Y., Kim, M. J., Fu, Z., Zhang, Z., Wu, Y., Li, Z., Ma, Q., Han, S., Finn, C., et al. Cot-vla: Visual chain-of-thought reasoning for vision-language-action models. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 1702–1713, 2025.   
Zheng, Y., Mao, S., Zhang, D., and Cai, W. Llm-enhanced rapid-reflex async-reflect embodied agent for real-time decision-making in dynamically changing environments. arXiv preprint arXiv:2506.07223, 2025.

# A. Model Details

Table 1. ResDreamer and DreamerV3 world model settings. 

<table><tr><td>Configurations</td><td>DreamerV3</td><td>ResDreamer (50Mx2)</td><td>ResDreamer (100Mx2)</td></tr><tr><td>Foresight horizon</td><td>0</td><td>4</td><td>4</td></tr><tr><td>Recurrent  $h_t$  size</td><td>6144</td><td>4096</td><td>6144</td></tr><tr><td>Recurrent  $z_t$  size</td><td>32 × 48</td><td>32 × 32</td><td>32 × 48</td></tr><tr><td>Hidden size</td><td>768</td><td>512</td><td>768</td></tr><tr><td>Encoder CNN channels</td><td>48</td><td>32</td><td>48</td></tr><tr><td>Decoder CNN channels</td><td>32</td><td>32</td><td>32</td></tr><tr><td>hierarchies</td><td>1</td><td>2</td><td>2</td></tr><tr><td>Total parameters</td><td>109.5M</td><td>92.0M</td><td>192.7M</td></tr><tr><td>Total training hours</td><td>6.2</td><td>12.3</td><td>14.5</td></tr></table>

Table 2. Hyperparameter settings. 

<table><tr><td>Hyperparameter</td><td>Value</td><td>Hyperparameter</td><td>Value</td></tr><tr><td colspan="4">Training</td></tr><tr><td>Learning rate</td><td> $4 \times 10^{-5}$ </td><td>Slow value network update rate</td><td>0.02</td></tr><tr><td>Environment samples</td><td>4</td><td>Value horizon</td><td>333</td></tr><tr><td>Replay batch size</td><td>16</td><td>Discount factor ( $\gamma$ )</td><td>1-1/333</td></tr><tr><td>Replay batch length</td><td>64</td><td>Optimizer</td><td>Adam</td></tr><tr><td>Buffer size</td><td> $1 \times 10^{6}$ </td><td>Train ratio</td><td>48</td></tr><tr><td colspan="4">World Model</td></tr><tr><td>Imagination length</td><td>15</td><td>Return normalization</td><td>Moving (5%, 95%)</td></tr><tr><td>Reward loss scale</td><td>1</td><td>Dynamics loss scale</td><td>1</td></tr><tr><td>Value loss scale</td><td>1</td><td>Representation loss scale</td><td>0.1</td></tr><tr><td>Policy loss scale</td><td>1</td><td>Replay value loss scale</td><td>0.3</td></tr><tr><td>Reconstruction loss scale</td><td>1</td><td></td><td></td></tr><tr><td colspan="4">Actor-Critic</td></tr><tr><td>bootstrap  $\lambda$ -return  $\lambda$  factor</td><td>0.95</td><td>Slow value regularization scale</td><td>1</td></tr><tr><td>Entropy coefficient ( $\alpha$ )</td><td> $3 \times 10^{-4}$ </td><td>Value distribution</td><td>symexp twohot</td></tr></table>

Figure 2 illustrates the data flow diagram of open-loop imagination and how it constructs enhanced visual observations during training. Our hierarchical model extends the process of updating the internal recurrent state based on observations. See Algorithm 1 for details. The sequence of environmental interactions stored in the replay buffer is utilized only for training the representational learning of the world model, while policy improvement relies exclusively on imagined trajectories. Consequently, the training pipeline and the environment interaction are entirely asynchronous. For a detailed description of the training pipeline, refer to Algorithm 2.

Algorithm 1 Update the recurrent state of ResDreamer upon observation   
Input: recurrent state $s_{t}$ , raw observation $o_{raw}$ .
Output: recurrent state $s_{t+1}$ , world model losses $\mathcal{L}_{\mathrm{dyn}}(\phi), \mathcal{L}_{\mathrm{rep}}(\phi), \mathcal{L}_{\mathrm{rec}}(\phi)$ .
1: Open-loop rollout imaginary state-action trajectory $\left\{\hat{s}^{0:L-1}, a\right\}_{t+1:t+H}$ 2: initiate $o_{res}^{k}$ with empty set.
3: for each $k = 0, 1, \cdots, L - 1$ do
4: Compute $o_{imag}^{k}$ with Eq. 3.
5: Compute $o_{t}^{k}$ with Eq. 4.
6: $z_{t}^{k} \leftarrow \text{sample}\left[q_{\phi}\left(z_{t}^{k} \mid h_{t}^{k}, o_{t}^{k}\right)\right]$ . {Encoder}
7: $\hat{z}_{t}^{k} \leftarrow \text{sample}\left[p_{\phi}\left(z_{t}^{k} \mid h_{t}^{k}\right)\right]$ . {Predictor}
8: Compute prediction loss $\mathcal{L}_{\mathrm{dyn}}^{k}(\phi)$ and representation loss $\mathcal{L}_{\mathrm{rep}}^{k}(\phi)$ with Eq. 7.
9: $h_{t+1}^{k} \leftarrow S_{\phi}\left(z_{t}^{k}, h_{t}^{k}, a_{t}^{k}\right)$ . {Sequence model}
10: Compute sensory signal reconstruction $\hat{o}_{t}^{k}=\left\{\hat{o}_{\mathrm{raw}}^{k},\hat{o}_{\mathrm{res}}^{k}\right\}_{t}$ . {Decoder}
11: Compute reconstruction loss $\mathcal{L}_{\mathrm{rec}}^{k}(\phi)$ with Eq. 6.
12: Compute $o_{res}^{k}$ with Eq. 2.
13: end for
return $s_{t+1}, \mathcal{L}_{\mathrm{dyn}}(\phi), \mathcal{L}_{\mathrm{rep}}(\phi), \mathcal{L}_{\mathrm{rec}}(\phi)$ .

Algorithm 2 The training pipeline of ResDreamer   
1: initiate parameters $\phi, \theta, \psi$ .
2: initiate carried state $s_{carry}$ .
3: while not converged do
4: {World model representation learning}
5: Sample a environmental interaction sequence $\{o_{raw}, a\}_{0:T-1}$ from replay buffer.
6: for each $t = 0, 1, \cdots, T - 1$ do
7: Update the $s_{carry}$ upon $\{o_{raw}\}_{t}$ with Algorithm 1.
8: Store trajectory feature $\left\{h_{t}^{0:L-1}, z_{t}^{0:L-1}\right\}$ and losses $\mathcal{L}_{\mathrm{dyn}}(\phi), \mathcal{L}_{\mathrm{rep}}(\phi), \mathcal{L}_{\mathrm{rec}}(\phi)$ .
9: end for
10: {Actor-critic learning}
11: Stack feature sequence $F \leftarrow \left\{ h_{0:T-1}^{0:L-1}, z_{0:T-1}^{0:L-1} \right\}$ .
12: Compute the bootstrapped $\lambda$ -return $R_{t}^{\lambda}$ and critic loss $\mathcal{L}(\theta)$ with Eq. 9.
13: View F as a batch of entry points sized T.
14: Open-loop rollout imaginary state-action trajectory of B time-steps starting at entry points batch F.
15: for each imaginary trajectory $\{\hat{s}_{0:B-1}, a_{0:B-1}\}$ do
16: Compute the normalized return and actor loss $\mathcal{L}(\psi)$ with Eq. 10.
17: end for
18: Back propagate losses $\mathcal{L}_{\mathrm{dyn}}(\phi), \mathcal{L}_{\mathrm{rep}}(\phi), \mathcal{L}_{\mathrm{rec}}(\phi), \mathcal{L}(\theta), \mathcal{L}(\psi)$ .
19: Optimize parameters $\phi, \theta, \psi$ .
20: end while

# B. Environment Details

MineDojo agent’s initial inventory includes a iron sword, shield, and a full suite of iron armors across all tasks. The maximum number of time-steps for one episode is 1000. For other specifications, see Table 3.

As shown in Table 3, the five Mobs each possess distinct characteristics. Each episode terminates upon timeout or when the agent’s health reaches zero, which implies that the agent must not only explore and approach enemies but also learn to evade attacks or defend with a shield. The rich interaction mechanisms thoroughly test the generalization capabilities of RL algorithms.

In MineDojo tasks, the agent is equipped with iron armors and iron sword shield at initialization. We adopt sparse reward from MineDojo at episode termination and dense reward from MineCLIP (Fan et al., 2022). Each MineCLIP reward is computed of video segment of 16 time-steps, with calculations taking place every 8 frames. In addition, the agent is rewarded at any valid attack and punished for losing health points. The agent is trained for $1 \times 1 0 ^ { 6 }$ environment steps. The image input of the MineCLIP model is 160 × 256 pixels, while ResDreamer observers 2x down-sampled images. All experiments can be reproduced with VRAM less than 29 GB.

Table 3. MineDojo tasks specifications. 

<table><tr><td>Mobs</td><td>Biome</td><td>Mob Features</td><td>MineClip prompt</td></tr><tr><td>Spider</td><td>extreme hills</td><td>Fast movement</td><td>combat a spider in night extreme hills with a iron sword, shield, and a full suite of iron armors</td></tr><tr><td>Shulker</td><td>end</td><td>Shoots guided bullets which causes floating</td><td>combat a shulker in the end with a iron sword, shield, and a full suite of iron armors</td></tr><tr><td>Wolf</td><td>taiga</td><td>More agile, group attacks</td><td>combat a wolf in taiga with a iron sword, shield, and a full suite of iron armors</td></tr><tr><td>Skeleton</td><td>extreme hills</td><td>Accurate ranged attacks with arrows</td><td>combat a skeleton in night extreme hills with a iron sword, shield, and a full suite of iron armors</td></tr><tr><td>Ghast</td><td>nether</td><td>Flying, ranged attacks with explosive fireball, terrain destruction</td><td>combat a ghast in nether with a iron sword, shield, and a full suite of iron armors</td></tr></table>

# C. Additional Results

# C.1. IRIS Baseline

We reproduce IRIS on Minedojo and run a experiment using three random seeds and the official default hyperparameters. The only necessary changes were:

• Adding support for MineDojo’s MultiDiscrete action space (instead of the original flat Discrete used on Atari).   
• Setting MineDojo observations to 64×64 (instead of our usual 80×128) to match IRIS’s hard-coded input resolution. This might weaken the model’s capabilities, but is required for compatibility.

We have trained IRIS (Micheli et al., 2022) under default configuration for 7 days. So far, IRIS has failed to achieve meaningful success even on the easiest Combat Spider task throughout 500K environment steps. Our current conclusion regarding the reproduction of IRIS on Minedojo is that the default configuration is unable to complete the Minedojo task. Further hyperparameter tuning is still required.

![](images/56a50d7b49df7d7b41c80946ec98d1cf5869361e9b65961c2d9ba86d47d0ec5b.jpg)

<details>
<summary>text_image</summary>

Steve1
• DreamerV3 (100M)
• IRIS
• PTGM
• ResDreamer (50Mx2)
• ResDreamer (100Mx2)
</details>

![](images/ad9f66a0a775f6705a0b05059249fb94abc9667c28e2e48efc9d5c289e6e3f43.jpg)

<details>
<summary>line</summary>

| Step   | Purple Line | Blue Line | Yellow Line | Red Line | Green Line |
| ------ | ----------- | --------- | ----------- | -------- | ---------- |
| 0      | 0.1         | 0.1       | 0.1         | 0.0      | 0.0        |
| 150K   | 0.9         | 0.9       | 0.8         | 0.6      | 0.1        |
| 300K   | 0.95        | 0.85      | 0.85        | 0.7      | 0.15       |
| 450K   | 0.98        | 0.9       | 0.8         | 0.75     | 0.05       |
| 500K   | 0.97        | 0.9       | 0.75        | 0.65     | 0.05       |
</details>

Figure 7. Result of IRIS baseline.

C.2. Main Comparison Bar-chart and Ablation Numerical Result   
![](images/d9e2ddb2dddc36ba22a5f16e3f848afc2a0e706e5a6cc6e896edaba1f2cfae28.jpg)

<details>
<summary>bar</summary>

| Category         | DreamerV3 (100M) | ResDreamer (50Mx2) | ResDreamer (100Mx2) |
| ---------------- | ---------------- | ------------------ | ------------------- |
| Combat ghast     | 0.75             | 0.85               | 0.80                |
| Combat shulker   | 0.05             | 0.05               | 0.30                |
| Combat skeleton  | 0.05             | 0.20               | 0.25                |
| Combat spider    | 0.85             | 0.85               | 0.95                |
| Combat wolf      | 0.65             | 0.70               | 0.65                |
| Combat ghast     | 85.0             | 88.0               | 88.0                |
| Combat shulker   | 30.0             | 30.0               | 48.0                |
| Combat skeleton  | 30.0             | 48.0               | 48.0                |
| Combat spider    | 95.0             | 95.0               | 98.0                |
| Combat wolf      | 85.0             | 82.0               | 82.0                |
| Combat ghast     | 25.0             | 20.0               | 22.0                |
| Combat shulker   | 95.0             | 95.0               | 68.0                |
| Combat skeleton  | 85.0             | 82.0               | 75.0                |
| Combat spider    | 28.0             | 25.0               | 18.0                |
| Combat wolf      | 50.0             | 45.0               | 48.0                |
</details>

Figure 8. Comparisons of success rate (↑), episode score (↑) and episode length (↓) across tasks. It can be seen that ResDreamer achieves higher scores and success rates with fewer steps. Although the ResDreamer (50Mx2) has slightly fewer total parameters than DreamerV3 (100M), it performs better in almost all tasks.

Table 4. Ablation result 

<table><tr><td>Configurations</td><td>hierarchy</td><td>Rollout Hint</td><td>Residual Connection</td><td>Success Rate</td></tr><tr><td>ResDreamer (50Mx3)</td><td>3</td><td>√</td><td>√</td><td>0.776</td></tr><tr><td>ResDreamer (50Mx2)</td><td>2</td><td>√</td><td>√</td><td>0.727</td></tr><tr><td>ResDreamer (Only residual hints)</td><td>2</td><td></td><td>√</td><td>0.563</td></tr><tr><td>Dreamer (With roll-out foresight)</td><td>1</td><td>√</td><td></td><td>0.559</td></tr><tr><td>ResDreamer (Only rollout hints)</td><td>2</td><td>√</td><td></td><td>0.400</td></tr><tr><td>ResDreamer (Heads conditioned on all)</td><td>2</td><td>√</td><td>√</td><td>0.377</td></tr></table>

# D. Baseline Introduction

# D.1. Selected Methods

We compare ResDreamer with strong Minecraft RL algorithms, including:

DreamerV3 (Hafner et al., 2025): A model-based RL foundation model. DreamerV3 is trained from scratch without demonstrations and domain knowledge. It generates future latent states recurrently with a non-hierarchical world model.

STEVE-1 (Lifshitz et al., 2023): An finetuned Video Pretraining (VPT) model for open-ended text and visual instructions following. It is post trained through self-supervised behavioral cloning. We test its zero-shoot text instructions following performance in MineDojo tasks.

PTGM (Yuan et al., 2024): A hierarchical approach integrating a high-level task goal generation strategy and a low-level goal-conditioned RL strategy. The high-level goal strategy is pretrained on large-scale, task-agnostic datasets, while the low-level strategy is learned online through RL. We utilize the open-source upper-layer strategy parameters of PTGM and evaluate its online training performance on MineDojo tasks using the default configuration of PTGM code-base.

# D.2. Unselected Methods

We provide introductions of other strong Minecraft agents and the reasons we do not compare ResDreamer with them.

LS-Imagine (Li et al., 2024a): An MBRL method that achieves arbitrary time-span reasoning through dual-branch prediction. It is based on DreamerV3, but it supports long-term prediction by simulating jumping to the vicinity of navigation targets through cropping observation. However, combat missions are different from navigation and exploration. Factors such as terrain, enemy reactions, etc. have a significant impact on the expected return, and cutting the images disrupts the data distribution. For instance, it is not reasonable to jump to flying enemies like ghasts by cropping the image.

Voyager (Wang et al., 2023a), JARVIS-1 (Wang et al., 2023b), MC-Planner (Wang et al., 2023c), RL-GPT (Liu et al., 2024): Open-Ended embodied agents that integrates RL with LLM. They adopt heterogeneous hierarchical models, leveraging the prior knowledge of LLMs to achieve task decomposition, long-term planning, code as strategy, and lifelong skill accumulation. Their focus lies in the integration and interaction methods between LLMs and RL, emphasizing the evaluation of an agent’s efficiency in accumulating atomic skills and activating technological milestones. Our proposed ResDreamer is a model-based RL foundation model, focusing on evaluating the data efficiency, scalability, and interpretability. ResDreamer can work together with all kinds of upper layer LLMs as a more powerful RL algorithm.

ROCKET-2 (Cai et al., 2025a), ROCKET-3 (Cai et al., 2025b) SkillDiscovery (Deng et al., 2025), JarvisVLA (Li et al., 2025b): Open-world VLA agents powered by imitation learning (IL) and prior knowledge of visual foundation model such as SAM (Kirillov et al., 2023). VLA agents focus on following open instructions within a broader range of atomic skills and their combinations. However, ResDreamer is a MBRL foundation model trained without any prior knowledge. ResDreamer focuses on developing a task-agnostic and domain general hierarchical world model method.