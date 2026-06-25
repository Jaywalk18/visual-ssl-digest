# Semantic Allocation in Ordered Bottlenecks: Predictive Residual Inference for Visual Representation Learning

Erik Ayari<sup>[0009−0003−9408−9762]</sup>, Manuel Traub<sup>[0000−0003−0897−1701]</sup>, and Martin V. Butz<sup>[0000−0002−8120−8537]</sup>

Neuro-Cognitive Modeling Group, University of Tübingen, Tübingen, Germany erik.ayari@uni-tuebingen.de

manuel.traub@uni-tuebingen.de, martin.butz@uni-tuebingen.de

Abstract. Ordered bottlenecks aim to provide utility at flexible budgets by assigning coarse information to early tokens and task-relevant detail to later ones. Prior work, including tail dropping (TD), typically enforces ordering by means of a masking-based ordering pressure (MBOP): Late tokens are masked more frequently than early tokens and are therefore encouraged to store less essential fine details. We introduce predictive residual inference for ordered representations (PRIOR), a framework designed to address inherent weaknesses of MBOP. MBOP is prone to weak late-token utility because it lacks an explicit refinement objective and uses gradient exposure as a proxy for importance. Furthermore, representations may become particularly brittle in optimization-sensitive settings, such as when using discrete or quantized token representations. PRIOR replaces activation-rate control with log -scaled levels and levelwise predictors. These predictors separate already explained from unexplained information, focusing each level on residual error. We compare PRIOR against MBOP-TD and independent tail-biased dropout (MBOP-ITD) in contrastive learning and image reconstruction tasks. Unlike the baselines, PRIOR learns well-ordered representations across experiments: low budgets provide coarse descriptors, while high budgets add refinements. Simultaneously, full-budget performance with PRIOR is higher in all but one experimental setting, where performance remains comparable. MBOP baselines are severely limited in discrete and quantized settings, while PRIOR approaches the performance of continuous counterparts. Taken together, these findings establish PRIOR as an effective framework for ordered representation learning.

Keywords: predictive residual inference · ordered representations · tail dropping · contrastive learning · coarse-to-fine processing

## 1 Introduction

Most deep learning architectures treat latent representations as fixed-size codes: During training and inference, the full code is produced, transmitted, and consumed by downstream components. This assumption becomes restrictive when computational, storage, or communication resources vary. With limited bandwidth, a mobile system may initially transmit only coarse scene information, and load additional details only after a user expresses interest. Similarly, an embodied agent may first obtain rough object estimates, select task-relevant objects, and then request finer details only for those. To support such adaptive use cases, representations should be hierarchically structured such that early tokens provide compact descriptors, while later tokens add complementary information when richer inference is needed.

Common methods for learning ordered representations impose order through masking or truncation during training, thereby inducing a masking-based ordering pressure (MBOP). For example, nested dropout stochastically removes contiguous sets of hidden units, making earlier dimensions more important for reconstruction and retrieval [13]. Related rateless auto-encoding approaches use gradually varying dropout rates to support flexible representation sizes [8]. More recently, quality-controllable image tokenizers have used tail dropping (TD) to concentrate important information near the beginning of a discrete token sequence [9]. Together, these methods show that masking and truncation can turn an otherwise flat latent sequence into a useful prefix code.

Despite their efectiveness, MBOP methods have notable weaknesses. First, the ordering pressure is tied directly to gradient exposure: later positions are masked more often and therefore receive weaker training signals. As a result, increasing the representation budget may yield diminishing returns, undermining the goal of budget-controllable representations. Second, reduced gradient exposure may be especially problematic for representations with biased or highvariance gradient estimates, such as categorical one-hot or quantized vectors. Finally, MBOP encourages ordering only implicitly, without an explicit objective that trains later tokens to refine the content already provided by earlier ones.

Models of human cognition ofer a useful complementary perspective. Perception is not only a bottom-up accumulation of local features. Instead, global structure can influence perception from early on. Accordingly, the visual system can be viewed as a hierarchical processing system, in which coarse prior knowledge about scenes or objects activates top-down expectations that facilitate perception [2,4,10]. Predictive coding models make this idea computationally explicit: higher levels predict lower-level activity in a top-down manner, while feedforward signals transmit residual prediction errors from lower to higher levels [11]. These findings suggest a representational principle: priors predict coarse structural outlines, while additional capacity should encode residual information, that is, details the coarser representation cannot explain.

We introduce PRIOR, a framework for learning hierarchical representations with flexible operating points. PRIOR separates ordering pressure from gradient exposure by learning a log -scaled hierarchy of token sequences. From coarse to fine, self-predictive modules double the latent resolution at each level and use residual errors to explicitly recruit the added capacity for information not yet explained by coarser levels.

We evaluate PRIOR across training objectives, downstream tasks, and architectural choices, comparing it against two MBOP baselines: standard tail dropping (TD) and independent tail-biased dropout (ITD). The first experimental track uses instance-level temporal contrastive learning and evaluates class readouts at multiple hierarchy levels. The second track studies reconstruction quality in an autoencoding task.

## 2 Methods

For all bottlenecks, we denote the latent representation as $\mathbf { Z } ,$ an arrangement of N tokens with channel size C. MBOP methods use a flat token sequence, i.e., an $[ N , C ]$ tensor; PRIOR arranges tokens into a pyramidal hierarchy. Fig. 1 summarizes the architectural distinction between MBOP-TD, MBOP-ITD, and PRIOR.

## 2.1 MBOP

We consider the forward pass of a single training sample and let i index tokens in Z. The probability of token i being masked is denoted by $m _ { i }$ . TD samples a single cutof c from a log-uniform distribution:

$$
u \sim \mathcal {U} (0, \log (N + 1)), \qquad c = \lfloor \exp (u) \rfloor , \qquad m _ {i} = \mathbf {1} [ i \geq c ].\tag{1}
$$

ITD instead samples masks independently for each token:

$$
m _ {i} \sim \text { Bernoulli } \left(\frac {\log (i + 1)}{\log (N + 1)}\right).\tag{2}
$$

Besides serving as an additional baseline, ITD allows us to test specific hypotheses within the MBOP framework. With TD, late tokens face a double disadvantage: they are updated less often, and their contribution to the loss may remain small because downstream components rely on earlier, more strongly trained tokens. ITD may improve attribution by creating low-probability events in which late tokens have increased downstream influence. At the same time, ITD may inadvertently increase redundancy, because conditioning on preceding tokens is weakened.

## 2.2 PRIOR

In PRIOR, Z is decomposed into L levels with doubling sequence lengths:

$$
\sum_ {\ell = 0} ^ {L - 1} 2 ^ {\ell} = N.\tag{3}
$$

Let $\mathbf { y } _ { \ell } .$ , a tensor of shape $\lceil 2 ^ { \ell } , C \rceil$ , be the unprocessed encoded input to PRIOR at level ℓ. From coarser to finer levels, predictor modules $f _ { \ell - 1 }$ refine the latent approximation $\hat { \mathbf { y } } _ { \ell - 1 }$ from level $\ell - 1$ . The residual error $\mathbf { r } _ { \ell }$ is compressed through a token bottleneck $\mathcal { Q } _ { \ell }$ . The compressed result is then added to the prediction $\hat { \mathbf { y } } _ { \ell } ^ { \mathrm { p r e d } }$ , yielding $\begin{array} { r } { \hat { \bf y } _ { \ell } , } \end{array}$ which serves as both the level output and the input to the next refinement stage:

![](images/be81ff41aac86293bae0ba72a784c093947b0adc4a35c32e66f9715b15bd70bb.jpg)  
Fig. 1. Visual summary of the three bottleneck architectures.

$$
\hat {\mathbf {y}} _ {\ell} ^ {\mathrm{pred}} = f _ {\ell - 1} (\hat {\mathbf {y}} _ {\ell - 1}), \qquad \mathbf {r} _ {\ell} = \mathbf {y} _ {\ell} - \hat {\mathbf {y}} _ {\ell} ^ {\mathrm{pred}}, \qquad \hat {\mathbf {y}} _ {\ell} = \hat {\mathbf {y}} _ {\ell} ^ {\mathrm{pred}} + \mathcal {Q} _ {\ell} (\mathbf {r} _ {\ell}).\tag{4}
$$

For $\ell = 0$ , the predictor output is zero. Refinement predictions are trained with

$$
\mathcal {R} _ {\mathrm{pred}} = \frac {1}{L - 1} \sum_ {\ell = 1} ^ {L - 1} \left\| \hat {\mathbf {y}} _ {\ell} ^ {\mathrm{pred}} - \mathrm{sg} [ \mathbf {y} _ {\ell} ] \right\| _ {2} ^ {2},\tag{5}
$$

where sg (stop-gradient) prevents the encoder from adjusting to the predictor.

Each level output $\hat { \mathbf { y } } _ { \ell }$ serves a separate downstream head with objective $\mathcal { I } ^ { ( \ell ) }$ Normalized geometric weighting yields the scalar loss:

$$
\mathcal {J} _ {\text { PRIOR }} = \frac {1}{Z} \sum_ {\ell = 0} ^ {L - 1} 2 ^ {- (L - 1 - \ell)} \mathcal {J} ^ {(\ell)}, \quad Z = \sum_ {k = 0} ^ {L - 1} 2 ^ {- k}.\tag{6}
$$

This preserves the training signal at all levels, while assigning exponentially larger loss weights to deeper levels, which contain an exponentially larger number of tokens. In our experiments, all levels use the same objective.

## 2.3 Token Parametrizations

Three variants of tokens are considered in this study: Gaussians, Categoricals, and EMA-VQ codebook vectors. In the contrastive learning task, tokens are additionally regularized to encourage compression and code usage, as this was found to stabilize performance (cf. Table 1). For PRIOR, parametrized tokens encode the residuals at each level with an added continuous predictor component. We still consider PRIOR to be a token-bottlenecked model, as the predictor component does not transmit additional sample-specific information beyond the retained residual tokens.

Gaussians are sampled via the reparameterization trick, Categoricals and EMA-VQ use straight-through gradient estimates. EMA-VQ vectors may select the null state.

Table 1. Token parametrizations, masked states, and token-specific regularization terms. H denotes categorical entropy.

<table><tr><td>Token Type</td><td>Masked State</td><td>Token Regularization</td></tr><tr><td>Gaussian</td><td> $\mathcal{N}(\mathbf{0},\mathbf{I})$ </td><td>Free-bits KL [7]:  $\mathbb{E}_{j}$  [max(0,  $D_{\text{KL}} - \kappa$ )]</td></tr><tr><td>Categorical</td><td> $\frac{1}{C} \cdot \mathbf{1}$ </td><td>Sample entropy:  $\mathbb{E}$  [H(q)]Negative batch entropy:  $-H$  (  $\mathbb{E}$  [q])</td></tr><tr><td>EMA-VQ</td><td> $\mathbf{0}$ </td><td>Commitment:  $\mathbb{E}$  [  $\|\hat{x} - \text{sg}[x^{\text{EMA}}]\|_{2}^{2}$  ]</td></tr></table>

Efective Information For each retained set of tokens $S ,$ , we use the validation set to estimate efective information from a token-wise bottleneck usage b:

$$
I (S) = \sum_ {u \in S} b (u)\tag{7}
$$

Across token types, $b ( u )$ estimates the information gained from token u relative to its marginal or inactive state: mutual information for discrete tokens, and the Gaussian variational-rate estimate via KL to the prior [1]:

$$
b (u) = \left\{ \begin{array}{l l} \mathbb {E} _ {x} [ D _ {\mathrm{KL}} (q _ {u} (z \mid x) \| p (z)) ] & \text {for Gaussians} \\ \max (0, H (\mathbb {E} _ {x} [ q _ {u} (z \mid x) ]) - \mathbb {E} _ {x} [ H (q _ {u} (z \mid x)) ]) & \text {for Categoricals} \\ H (\mathbb {E} _ {x} [ \mathrm{onehot} (a _ {u} (x)) ]) & \text {for EMA - VQ} \end{array} \right.\tag{8}
$$

## 2.4 Contrastive Learning

Training data are sampled from MVImgNet2 [16] object tracks, where separate frames of the same object instance constitute a positive pair. We use the standard NT-Xent objective [3,14]:

$$
\ell_ {i} = - \log \frac {\exp \big (s (\mathbf {h} _ {i} , \mathbf {h} _ {p (i)}) / \tau \big)}{\sum_ {j = 1} ^ {2 B} \mathbf {1} _ {j \neq i} \exp (s (\mathbf {h} _ {i} , \mathbf {h} _ {j}) / \tau)}, \qquad \mathcal {L} _ {\mathrm{NTX}} = \frac {1}{2 B} \sum_ {i = 1} ^ {2 B} \ell_ {i}.\tag{9}
$$

Here, $p ( i )$ denotes the positive partner of view i, s is cosine similarity, $\tau$ is a temperature parameter, and B is the number of pairs in the batch. The numerator rewards similarity of encoded positive pairs; the denominator avoids model collapse by penalizing similarities of negative pairs. As in [3], we use projection heads, $\mathrm { i . e . , }$ small MLPs producing $\mathbf { h } _ { i }$ as a projected, lower-dimensional descriptor of view i. We adapt this idea to PRIOR by means of level-wise projection heads.

For the encoder, we use FLIP [15], a transformer-based, object-centric vision model with foveal multi-scale patching to selectively process individual objects. Regularizer weights were selected by a two-stage Bayesian optimization procedure. Broad stage-1 samples were first evaluated in short-horizon runs;

a Gaussian-process surrogate [12] then guided stage-2 refinement using strong anchors, posterior-mean candidates, and expected improvement [5].

MVImgNet2 provides category labels at four taxonomy levels, from abstract to nuanced, referred to here as Coarse-, Mid-, Fine-, and Class-level. Readouts are evaluated by means of linear probes trained on frozen representations. Table 2 reports the resulting contrastive model sizes, separating shared encoders, projection heads, EMA codebooks, and PRIOR self-predictors.

Table 2. Model parameters in millions by architecture, token type and model component. EMA codebooks are accounted for separately, as they are not directly optimized but track network activity over time and become a component of the representation. Self-Pred. refers to the parameters of PRIOR’s refinement modules. Total accounts for parameters used in downstream tasks, i.e., without projection heads, with bracketed numbers including them.

<table><tr><td>Architecture</td><td>Token Type</td><td>FLIP</td><td>Projection</td><td>EMA</td><td>Self-Pred.</td><td>Total</td></tr><tr><td rowspan="3">PRIOR</td><td>Gaussian</td><td>3.042</td><td>0.929</td><td>-</td><td>3.778</td><td>6.819 [7.748]</td></tr><tr><td>Categorical</td><td>2.926</td><td>0.929</td><td>-</td><td>3.778</td><td>6.704 [7.633]</td></tr><tr><td>EMA-VQ</td><td>2.811</td><td>0.929</td><td>0.259</td><td>3.778</td><td>6.847 [7.776]</td></tr><tr><td rowspan="3">MBOP</td><td>Gaussian</td><td>6.996</td><td>0.133</td><td>-</td><td>-</td><td>6.996 [7.129]</td></tr><tr><td>Categorical</td><td>4.899</td><td>0.133</td><td>-</td><td>-</td><td>4.899 [5.032]</td></tr><tr><td>EMA-VQ</td><td>2.802</td><td>0.133</td><td>4.145</td><td>-</td><td>6.948 [7.080]</td></tr></table>

## 2.5 Reconstruction

A wavelet-space vision transformer is trained on FFHQ autoencoding [6]. As in contrastive learning, reconstruction with PRIOR uses level-wise decoders. The parameter count for full-budget reconstruction was approximately 11.479 million for MBOP models and 11.761 million for PRIOR.

## 3 Results

We evaluate ordered bottlenecks according to two criteria: whether they preserve strong peak performance, and whether additional tokens provide complementary, fine-grained information.

## 3.1 Contrastive Learning of Ordered Object Codes

Before analyzing frozen linear probes, Table 3 reports the validation NT-Xent losses of the trained contrastive models. PRIOR reaches the lowest validation loss in all token families; among MBOP models, ITD is lower for Categorical and EMA-VQ, and TD for Gaussian.

Table 3. NT-Xent validation loss of the trained models. Bold marks the lowest loss within the token family. PRIOR models report the loss value from the head of the final level.

<table><tr><td>Token type</td><td>PRIOR opt.</td><td>MBOP-ITD opt. (match)</td><td>MBOP-TD opt. (match)</td></tr><tr><td>Gaussian</td><td>1.244</td><td>1.337 (1.391)</td><td>1.288 (1.424)</td></tr><tr><td>Categorical</td><td>1.400</td><td>1.844 (2.137)</td><td>2.881 (2.664)</td></tr><tr><td>EMA-VQ</td><td>1.350</td><td>1.426 (1.994)</td><td>2.183 (2.376)</td></tr></table>

![](images/3e7092a293d7c6e7ce65aec2d3999f82e29ed54647ef4c82b43ecec59903decc.jpg)

![](images/dc9588b8599b8d4f18ae7f6673024ced632127e4d95f16f0057eef48352d67b0.jpg)  
Fig. 2. Budget-wise linear-probe accuracy for class and coarse taxonomy levels. Shading decomposes readout quality by utilized tokens up to the highest-performing prefix of the respective model; for PRIOR, a prefix corresponds to the first k concatenated levels.

Fig. 2 summarizes budget-wise linear-probe accuracy for class and coarse taxonomy readouts; complete results are given in Table 4. Diferences between MBOP-ITD and MBOP-TD tend to be small, with a tendency for higher accuracies in MBOP-ITD. EMA-VQ models are the exception; ITD strongly improves over TD $( \mathrm { e . g . }$ , peak class accuracy of 26.37% vs. 15.9%). As these models were tuned individually to generate maximally strong baselines against PRIOR with respect to their training objective, ITD–TD results on their own do not grant clear attribution to masking schemes. Therefore, we conducted a control experiment in which each ITD–TD pair was tuned to optimize the combined, average performance. Within the control, MBOP diferences were similar with respect to directionality and magnitude (cf. Fig. 3).

Conversely, PRIOR and MBOP baselines difer strongly, both with respect to peak accuracies and in relation to ordering. Highest accuracies per model and token parametrizations are predominantly observed with PRIOR. The exception is Gaussian class accuracy, where MBOP-ITD peaks at 81.75%, followed by PRIOR at 79.74% and MBOP-TD at 77.95%. Coarse, mid, and fine readouts are all maximized by PRIOR, followed by MBOP-ITD. For instance, within the Gaussian family at the mid level, PRIOR reaches 75.46%, MBOP-ITD 68.78%, and MBOP-TD 66.25%.

Table 4. Frozen linear-probe accuracy across architectures, tokens, budgets, and taxonomy levels. Budgets describe (level-) prefixes; bold and underlining indicate the best and second-best model within each family, budget, and taxonomy level. Star and plus mark the best and second-best model within each family and taxonomy level, across budgets.

<table><tr><td rowspan="2">Budget</td><td colspan="4">PRIOR</td><td colspan="4">MBOP-ITD</td><td colspan="4">MBOP-TD</td></tr><tr><td>Coarse</td><td>Mid</td><td>Fine</td><td>Class</td><td>Coarse</td><td>Mid</td><td>Fine</td><td>Class</td><td>Coarse</td><td>Mid</td><td>Fine</td><td>Class</td></tr><tr><td colspan="13">Gaussian</td></tr><tr><td>1</td><td>70.11</td><td>60.46</td><td>44.60</td><td>49.23</td><td>73.35</td><td>63.95</td><td>51.68</td><td>73.49</td><td>74.45</td><td>65.23</td><td>53.10</td><td>76.91</td></tr><tr><td>3</td><td>71.22</td><td>61.46</td><td>46.05</td><td>56.45</td><td>73.75</td><td>64.67</td><td>52.27</td><td>76.12</td><td>74.62</td><td>66.25</td><td>53.97</td><td>77.60</td></tr><tr><td>7</td><td>72.46</td><td>63.19</td><td>48.67</td><td>62.91</td><td>74.20</td><td>65.56</td><td>53.32</td><td>77.72</td><td>74.94</td><td>65.95</td><td>54.16</td><td>77.64</td></tr><tr><td>15</td><td>74.32</td><td>65.33</td><td>51.88</td><td>68.71</td><td>74.60</td><td>66.09</td><td>54.05</td><td>79.70</td><td>74.71</td><td>66.00</td><td>54.18</td><td>77.75</td></tr><tr><td>31</td><td>77.55</td><td>68.82</td><td>56.17</td><td>73.01</td><td>75.07</td><td>66.64</td><td>55.26</td><td>80.58</td><td>74.76</td><td>66.06</td><td>54.28</td><td>77.75</td></tr><tr><td>63</td><td>82.70</td><td>73.52</td><td>63.65</td><td>78.95</td><td>75.75</td><td>67.57</td><td>56.72</td><td>81.75*</td><td>74.80</td><td>65.53</td><td>54.12</td><td>77.95</td></tr><tr><td>127</td><td>82.94*</td><td>75.46*</td><td>64.97*</td><td>79.74+</td><td>76.82+</td><td>68.78+</td><td>58.57+</td><td>80.30</td><td>74.80</td><td>66.23</td><td>54.53</td><td>77.35</td></tr><tr><td colspan="13">Categorical</td></tr><tr><td>1</td><td>75.37</td><td>65.42</td><td>48.32</td><td>24.16</td><td>63.66</td><td>54.03</td><td>36.49</td><td>7.96</td><td>63.22</td><td>53.68</td><td>36.38</td><td>6.48</td></tr><tr><td>3</td><td>80.27</td><td>71.05</td><td>56.14</td><td>48.17</td><td>68.70</td><td>59.02</td><td>42.25</td><td>22.11</td><td>67.36</td><td>57.67</td><td>40.99</td><td>16.28</td></tr><tr><td>7</td><td>81.14</td><td>72.44</td><td>58.45</td><td>59.11</td><td>72.35</td><td>62.85</td><td>47.10</td><td>35.91</td><td>70.66</td><td>61.16</td><td>44.42</td><td>24.34</td></tr><tr><td>15</td><td>82.87</td><td>74.41</td><td>61.67</td><td>66.63</td><td>74.92</td><td>65.58</td><td>51.08</td><td>45.41</td><td>73.49</td><td>64.05</td><td>48.01</td><td>35.61</td></tr><tr><td>31</td><td>83.85</td><td>75.95</td><td>64.88</td><td>71.94</td><td>76.83</td><td>67.56</td><td>53.64</td><td>50.92</td><td>77.42</td><td>68.16</td><td>54.33</td><td>49.50</td></tr><tr><td>63</td><td>86.07</td><td>79.51</td><td>68.43</td><td>77.17*</td><td>78.89</td><td>70.45</td><td>56.59</td><td>54.91</td><td>79.45</td><td>70.81</td><td>57.88</td><td>58.02</td></tr><tr><td>127</td><td>88.01*</td><td>80.45*</td><td>70.37*</td><td>75.81</td><td>81.73+</td><td>73.40+</td><td>60.97+</td><td>61.87</td><td>79.73</td><td>71.32</td><td>58.51</td><td>63.67+</td></tr><tr><td colspan="13">EMA-VQ</td></tr><tr><td>1</td><td>63.28</td><td>52.93</td><td>35.03</td><td>4.55</td><td>62.83</td><td>52.68</td><td>34.74</td><td>4.37</td><td>62.50</td><td>51.68</td><td>27.87</td><td>9.74</td></tr><tr><td>3</td><td>68.44</td><td>58.68</td><td>40.78</td><td>20.50</td><td>62.79</td><td>52.51</td><td>34.75</td><td>6.18</td><td>65.50</td><td>52.25</td><td>33.28</td><td>14.40</td></tr><tr><td>7</td><td>71.85</td><td>61.80</td><td>45.82</td><td>40.71</td><td>65.87</td><td>56.17</td><td>39.08</td><td>17.75</td><td>67.42</td><td>56.15</td><td>31.61</td><td>15.35</td></tr><tr><td>15</td><td>76.21</td><td>67.16</td><td>52.68</td><td>54.16</td><td>66.52</td><td>56.45</td><td>39.22</td><td>18.06</td><td>67.59</td><td>55.82</td><td>34.82</td><td>15.90</td></tr><tr><td>31</td><td>78.16</td><td>69.83</td><td>56.01</td><td>60.03</td><td>66.48</td><td>56.86</td><td>39.43</td><td>17.30</td><td>67.01</td><td>57.00</td><td>35.54</td><td>15.00</td></tr><tr><td>63</td><td>80.50</td><td>70.98</td><td>60.01</td><td>64.63</td><td>69.02+</td><td>58.56+</td><td>40.73+</td><td>26.37+</td><td>67.21</td><td>54.57</td><td>35.63</td><td>14.91</td></tr><tr><td>127</td><td>81.57*</td><td>73.31*</td><td>61.86*</td><td>69.42*</td><td>67.33</td><td>56.72</td><td>37.05</td><td>24.97</td><td>66.75</td><td>57.25</td><td>35.63</td><td>13.77</td></tr></table>

Particularly strong diferences are observed in the Categorical and EMA-VQ families. For example, Categorical-PRIOR reaches a class accuracy of 77.17% versus 63.67% for MBOP-TD, and EMA-VQ-PRIOR peaks at 69.42% compared to MBOP-ITD at only 26.37%.

Considering peak accuracies, PRIOR generally improves over MBOP within token families. More importantly, peak accuracies fall into a tighter range, lifting discrete and quantized tokens much closer to the Gaussian ceiling. As representational capacity across models difers significantly, Fig. 3 controls for efective information I. Accuracy diferences in the discrete and quantized settings remain high, making mere bandwidth diferences unlikely as a viable explanation.

PRIOR shows robust ordering of fine-grained semantic information as measured by readouts. Across token families, class-level accuracies strongly improve with increased budgets. Gaussian PRIOR, for instance, improves from 49.23% at one token to 79.74% at full budget; EMA-VQ PRIOR does so from 4.55% to 69.42%. Qualitatively, budget-dependent diferences are more evenly distributed and stepwise in PRIOR models (cf. Fig. 2, Fig. 3).

Adjusting for their weaker ceiling performance, discrete and quantized MBOP models show similarly strong class-level ordering efects. However, their Gaussian ITD counterpart is only weakly ordered (73.49% 81.75%), and TD is virtually flat (76.91%  77.95%). The control experiments show that lower regularization can improve ordering in Gaussian MBOP. However, the contrastive training objective is suboptimal in these cases and, more importantly, efective information is exponentially higher (cf. Fig. 3, Table 3).

![](images/d0f38b848f750831c3f93f588a272727f3fb1c1c1a5f2ad53fa79d5af0ba4b90.jpg)  
Fig. 3. Class accuracy versus log-scaled efective information for retained prefix budgets across tuned architectures and regularizer-matched controls. Arrows indicate directional changes of the respective weights in the control setting. Single arrows indicate both variants, two arrows indicate ITD/TD changes.

Because of their strong but quickly saturating performance, Gaussian MBOP models tend to outperform Gaussian PRIOR within the first budget quantile (up to 31 tokens), after which PRIOR converts additional budget into stronger readouts. Within the Categorical and EMA-VQ families, PRIOR yields stronger readouts across nearly all budgets; the one exception is EMA-VQ class readout with a single token.

The contrastive experiments can therefore be summarized as follows: Compared against baselines, PRIOR yields strong, coarse-to-fine ordered representations without exception. It most clearly improves discrete and quantized representations where MBOP models are comparatively weak, especially with EMA-VQ. With Gaussian tokens, accuracy diferences are small and inconsistent; PRIOR improves mainly the budget profiles.

## 3.2 Ordered Variational Autoencoding

Fig. 4 and Table 5 show a complementary pattern for FFHQ reconstruction. MBOP-TD is strongest at the smallest budgets, while MBOP-ITD is weakest. From 15 tokens onward, MBOP-ITD overtakes MBOP-TD, but both variants saturate early. In contrast, PRIOR improves steadily as additional levels become active, surpasses both MBOP variants from 31 tokens onward and continues improving up to the full budget. Representative examples in Fig. 5 are consistent with this result: TD provides plausible low-budget reconstruction but changes little later; ITD fails at the one-token extreme; and PRIOR adds visible detail across successive levels.

![](images/f487b30658824e834c8312ee5b3c62bd72a336363f726165bb1a9eade26a924c.jpg)

Fig. 4. Mean validation SSIM (↑) across cumulative token budgets.  
![](images/074310cfffcd0c1cf4106498f1cca5f6ccd586f4c318e1868f9acca24a0c5098.jpg)  
Fig. 5. Representative ground-truth (GT) images and reconstructions across architectures and token budgets.

## 4 Discussion

The results suggest that PRIOR is a promising framework for ordered representation learning. Across several experimental settings, we compared PRIOR against strong baselines. In these experiments, PRIOR was the only model that consistently approached the main goal of ordered representations: learning coarse-tofine codes while maintaining strong peak performance.

For MBOP, our results indicate a tension between prefix performance and overall representation quality. At high capacity, MBOP can use early tokens very eficiently, but it does not produce a clearly ordered semantic allocation. Settings that improve ordering tend to reduce peak performance. The results suggest that the forms of ordering induced by MBOP constrain capacity or lead to underfitting, rather than inducing robust structural organization.

Independent masking does not fully resolve this tension in our experiments. MBOP-ITD sometimes improves over MBOP-TD, especially at moderate to high budgets, but the gains are comparatively small and less consistent than the diferences between PRIOR and MBOP.

Table 5. FFHQ reconstruction validation means across token budgets. Best values are bold; second-best values are underlined.

<table><tr><td>Metric</td><td>Setup</td><td>1</td><td>15</td><td>31</td><td>127</td></tr><tr><td rowspan="3">SSIM (↑)</td><td>PRIOR</td><td>0.5370</td><td>0.6777</td><td>0.7232</td><td>0.7639</td></tr><tr><td>MBOP-ITD</td><td>0.4403</td><td>0.6851</td><td>0.6968</td><td>0.6990</td></tr><tr><td>MBOP-TD</td><td>0.5778</td><td>0.6734</td><td>0.6849</td><td>0.6860</td></tr><tr><td rowspan="3">L1 (↓)</td><td>PRIOR</td><td>0.0833</td><td>0.0420</td><td>0.0348</td><td>0.0301</td></tr><tr><td>MBOP-ITD</td><td>0.1716</td><td>0.0411</td><td>0.0385</td><td>0.0379</td></tr><tr><td>MBOP-TD</td><td>0.0725</td><td>0.0440</td><td>0.0419</td><td>0.0419</td></tr><tr><td rowspan="3">L2 (↓)</td><td>PRIOR</td><td>0.0143</td><td>0.0044</td><td>0.0031</td><td>0.0023</td></tr><tr><td>MBOP-ITD</td><td>0.0585</td><td>0.0043</td><td>0.0038</td><td>0.0037</td></tr><tr><td>MBOP-TD</td><td>0.0116</td><td>0.0047</td><td>0.0043</td><td>0.0043</td></tr><tr><td rowspan="3">MS-SSIM (↑)</td><td>PRIOR</td><td>0.7156</td><td>0.9184</td><td>0.9505</td><td>0.9699</td></tr><tr><td>MBOP-ITD</td><td>0.4882</td><td>0.9227</td><td>0.9348</td><td>0.9366</td></tr><tr><td>MBOP-TD</td><td>0.7669</td><td>0.9127</td><td>0.9247</td><td>0.9250</td></tr></table>

Overall, the results support the main hypotheses that motivated PRIOR. Strong coarse readouts from early levels, together with consistent late-level improvements in detailed readouts, suggest that self-predictive residual refinement provides an efective ordering mechanism. The pronounced utility of late tokens in discrete and quantized models further supports the view that restricted gradient exposure is a limiting factor for MBOP, but not for PRIOR. The analysis of efective information is consistent with this interpretation, indicating that PRIOR-induced orderings can transmit information more eficiently.

Taken together, these findings point to several directions for future work. Further evaluations should assess the breadth of PRIOR’s applicability, the robustness of learned hierarchical encodings, and the semantic nature of these encodings. PRIOR also ofers additional design choices that were not exhaustively studied here, such as alternative weighting schemes or diferent per-level objectives. Most importantly, we intend to embed PRIOR into larger architectures and broader problem settings. PRIOR’s emergent hierarchical predictive structure enables refinement “on demand”. We believe that this ability could be especially useful for developing hierarchical world models that can be flexibly recruited for task-adaptive perception and planning. Such models may be particularly powerful in environments where relevant objects and interactions change depending on the task and environmental context.

Acknowledgments. We acknowledge funding by the Deutsche Forschungsgemeinschaft (DFG, German Research Foundation) – 381713393, 467045002, 564829065. Martin Butz is a member of the Machine Learning Cluster of Excellence – 390727645. The authors thank the International Max Planck Research School for Intelligent Systems (IMPRS-IS) for supporting Manuel Traub.

Disclosure of Interests. The authors have no competing interests to declare that are relevant to the content of this article.

## References

1. Alemi, A.A., Fischer, I., Dillon, J.V., Murphy, K.: Deep variational information bottleneck. In: International Conference on Learning Representations (2017), https://arxiv.org/abs/1612.00410

2. Bar, M., Kassam, K.S., Ghuman, A.S., Boshyan, J., Schmid, A.M., Dale, A.M., et al.: Top-down facilitation of visual recognition. Proceedings of the National Academy of Sciences 103(2), 449–454 (2006). https://doi.org/10.1073/pnas. 0507062103

3. Chen, T., Kornblith, S., Norouzi, M., Hinton, G.: A simple framework for contrastive learning of visual representations. In: Proceedings of the 37th International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 119, pp. 1597–1607. PMLR (2020)

4. Hochstein, S., Ahissar, M.: View from the top: Hierarchies and reverse hierarchies in the visual system. Neuron 36(5), 791–804 (2002). https://doi.org/10.1016/ S0896-6273(02)01091-7

5. Jones, D.R., Schonlau, M., Welch, W.J.: Eficient global optimization of expensive black-box functions. Journal of Global Optimization 13(4), 455–492 (1998). https: //doi.org/10.1023/A:1008306431147

6. Karras, T., Laine, S., Aila, T.: A style-based generator architecture for generative adversarial networks (2019), https://arxiv.org/abs/1812.04948

7. Kingma, D.P., Salimans, T., Jozefowicz, R., Chen, X., Sutskever, I., Welling, M.: Improving variational inference with inverse autoregressive flow (2017), https: //arxiv.org/abs/1606.04934

8. Koike-Akino, T., Wang, Y.: Stochastic bottleneck: Rateless auto-encoder for flexible dimensionality reduction. In: 2020 IEEE International Symposium on Information Theory (ISIT). pp. 2735–2740. IEEE (2020). https://doi.org/10.1109/ ISIT44484.2020.9174302

9. Miwa, K., Sasaki, K., Arai, H., Takahashi, T., Yamaguchi, Y.: One-d-piece: Image tokenizer meets quality-controllable compression. arXiv preprint arXiv:2501.10064 (2025)

10. Navon, D.: Forest before trees: The precedence of global features in visual perception. Cognitive Psychology 9(3), 353–383 (1977). https://doi.org/10.1016/ 0010-0285(77)90012-3

11. Rao, R.P.N., Ballard, D.H.: Predictive coding in the visual cortex: A functional interpretation of some extra-classical receptive-field efects. Nature Neuroscience 2(1), 79–87 (1999). https://doi.org/10.1038/4580

12. Rasmussen, C.E., Williams, C.K.I.: Gaussian Processes for Machine Learning. MIT Press (2006)

13. Rippel, O., Gelbart, M., Adams, R.: Learning ordered representations with nested dropout. In: Proceedings of the 31st International Conference on Machine Learning. Proceedings of Machine Learning Research, vol. 32, pp. 1746–1754. PMLR (2014)

14. Sohn, K.: Improved deep metric learning with multi-class n-pair loss objective. In: Lee, D., Sugiyama, M., Luxburg, U., Guyon, I., Garnett, R. (eds.) Advances in Neural Information Processing Systems. vol. 29. Curran Associates, Inc. (2016)

15. Traub, M., Butz, M.V.: Looking locally: Object-centric vision transformers as foundation models for eficient segmentation (2025), https://arxiv.org/abs/2502. 02763

16. Yu, X., Xu, M., Zhang, Y., Liu, H., Ye, C., Wu, Y., et al.: Mvimgnet: A large-scale dataset of multi-view images (2023)