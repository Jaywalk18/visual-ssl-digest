# DIT-REWARD: GENERATIVE REPRESENTATIONS FOR TEXT-TO-IMAGE REWARD MODELING

Yuanming Yang<sup>1,2</sup> Guoqing Ma<sup>1</sup> Bo Wang<sup>1,3</sup> Yuan Zhang<sup>1</sup> Wei Tang<sup>1</sup> Chenyi Li<sup>1,4</sup> Haoyang Huang<sup>1</sup> Nan Duan<sup>1</sup>

<sup>1</sup>JD Explore Academy, JD.com, Beijing, China <sup>2</sup>Tsinghua University, Beijing, China <sup>3</sup>Beijing Institute of Technology, Beijing, China <sup>4</sup>Peking University, Beijing, China

## ABSTRACT

Can representations learned for image generation also support the evaluation of generated images? We study text-to-image reward prediction as a downstream task of generative representation learning. To this end, we introduce DiT-Reward, which converts a pretrained text-to-image Diffusion Transformer into a reward model by processing near-clean image latents and aggregating text-conditioned image representations across transformer layers. Under the same training data mixture as HPSv3, DiT-Reward outperforms HPSv3 on all four evaluated preference benchmarks, reaching 85.6% on HPDv2 and 77.6% on HPDv3. When the generative backbone is frozen, a lightweight learned head can still extract meaningful preference predictions from its representations. Probing across depth further reveals that downstream reward performance is strongest in the middle-tolate layers and benefits from combining representations across different stages. We also observe consistent positive scaling with generative backbone capacity. Finally, when used to optimize Stable Diffusion 3.5 Large with Flow-GRPO, DiT-Reward outperforms HPSv3 along the matched training trajectory, with particularly clear gains in realism. Direct latent scoring also achieves a 1.65× inference speedup over HPSv3 with comparable peak memory. These results show that pretrained generative DiTs provide transferable representations for reward modeling and policy optimization.

## 1 INTRODUCTION

“What I cannot create, I do not understand.” This maxim, attributed to Feynman, suggests a tight connection between generation and understanding. Modern text-to-image diffusion transformers appear to raise the same question from the opposite direction: they can generate increasingly realistic, complex, and semantically rich images, yet the representations learned by these models are rarely used to understand or evaluate the generated results, or to align generation with human preferences. This raises a direct question: can the internal representations learned by a generator be turned into a reward signal for evaluating the outputs of that same generator?

Reward models have become an important interface for evaluating and optimizing generative models, supporting automatic evaluation, reranking, reinforcement learning, and preference optimization (Xu et al., 2023b; Kirstain et al., 2023; Wu et al., 2023b; Ma et al., 2025; Black et al., 2024; Fan et al., 2023; Wallace et al., 2024; Liu et al., 2025). However, existing text-to-image reward models are typically built on separately pretrained discriminative vision–language encoders or general-purpose VLMs, such as CLIP, BLIP, and their successors (Radford et al., 2021; Li et al., 2022). Modern DiT and MMDiT policies jointly process text and image tokens in a latent generative space, whereas reward evaluation typically requires the generated outputs to be decoded into pixels and then mapped into a separate vision–language representation space (Peebles & Xie, 2023; Esser et al., 2024; Stability AI, 2024; Black Forest Labs, 2024). Evaluation across these separate spaces not only introduces additional image decoding and visual encoding, but also separates the reward signal from the internal generative representations of the policy. This raises a direct question: can reward modeling operate in the native representation space of the generator?

To this end, we view reward prediction as a downstream task of generative representation learning and introduce DiT-Reward, a method that directly converts a pretrained text-to-image DiT into a reward model. Because a DiT is originally trained on noisy latents rather than completed images, we first encode an input image into the latent space used by the generator and apply a small near-clean perturbation to better match the DiT input distribution. We then extract text-conditioned image token representations from multiple MMDiT layers and map their pooled features to a scalar reward with a lightweight MLP. Under the same training data mixture as HPSv3, DiT-Reward outperforms HPSv3 on all four evaluated benchmarks and obtains the best results on HPDv2 and HPDv3. When the SD3.5-Large generative backbone is frozen and only a lightweight reward head is trained, the resulting text-conditioned image representations can already support human preference prediction without adapting the generative backbone. Increasing the generative backbone size further yields consistent gains across all four benchmarks.

Many technical reports on language model alignment adopt a common design in which the policy and reward model are derived from the same or closely related pretrained language models and then trained separately for generation and reward prediction (Ouyang et al., 2022; Touvron et al., 2023; Bai et al., 2023; Yang et al., 2023). Although they do not share their resulting weights, this common initialization allows the reward model to inherit semantic representations, pretrained knowledge, and a modeling space closely related to those of the policy. Text-to-image reinforcement learning has not widely adopted this design and instead typically obtains rewards from an independently pretrained CLIP, BLIP, or VLM (Black et al., 2024; Fan et al., 2023; Wallace et al., 2024; Liu et al., 2025). DiT-Reward brings the same idea to text-to-image reinforcement learning: in our experiments, both DiT-Reward and the policy are derived from Stable Diffusion 3.5 Large and subsequently optimized for reward modeling and generation, respectively. With Flow-GRPO, external evaluations by GPT-5 and Gemini-3-Flash show that the policy trained with DiT-Reward outperforms its HPSv3-trained counterpart along the matched training trajectory; dimension-specific evaluations and qualitative comparisons further show clear improvements in realism and structural accuracy. Direct policy latent scoring is also 1.65× faster than HPSv3 with comparable peak memory.

Our contributions are threefold. First, we propose DiT-Reward and show that a pretrained text-to-image DiT can be directly repurposed as an effective reward model, achieving leading or competitive results across major human preference benchmarks. Second, through probing with a frozen backbone, representation analysis across layers, and backbone scaling, we evaluate the transferability of pretrained generative representations to reward modeling and study how network depth and model capacity affect downstream reward prediction. Third, we validate DiT-Reward for text-to-image reinforcement learning when the reward model and policy share a common pretrained model origin, and demonstrate the efficiency advantage of direct reward evaluation in latent space.

## 2 RELATED WORK

Human preference modeling and text-to-image evaluation. Human preference modeling is widely used for evaluating and aligning text-to-image models. Existing methods learn image-text preference scorers from CLIP, BLIP, or stronger VLM backbones for evaluation and reranking (Xu et al., 2023b; Kirstain et al., 2023; Wu et al., 2023a; Ma et al., 2025). Recent work further decomposes preference into fine-grained dimensions or adopts generative VLMs to improve interpretability, scalability, and robustness (Xu et al., 2026; He et al., 2024; Wu et al., 2025b). In contrast, DiT-Reward reuses the text-to-image generative model itself and tests whether its internal representations are sufficient for human preference prediction.

Feedback learning for diffusion and flow models. Another line of work optimizes diffusion or flow generators with preference and reward signals, including reward-gradient guidance, policygradient formulations, KL-regularized online RL, and diffusion variants of DPO (Prabhudesai et al., 2023; Black et al., 2024; Fan et al., 2023; Wallace et al., 2024). Recent methods extend this direction to flow matching and rectified-flow models, such as Flow-GRPO and DiffusionNFT (Liu et al., 2025; Zheng et al., 2026). These works mainly ask how to optimize a generator given a reward. Our focus is complementary: we study how the reward model itself should be constructed.

Generative backbones as transferable representation learners. A growing body of work shows that pretrained generative models are also transferable visual representation learners. Diffusion features have been used for semantic correspondence, segmentation, depth estimation, and related dense prediction tasks, with quality depending on layer, timestep, noise level, and conditioning (Tang et al., 2023; Luo et al., 2023; Xu et al., 2023a; Zhao et al., 2023; Stracke et al., 2025). Meanwhile, text-to-image backbones have scaled from LDMs and DiTs to large multimodal diffusion and flow transformers (Rombach et al., 2022; Peebles & Xie, 2023; Podell et al., 2023; Esser et al., 2024; Black Forest Labs, 2024; Wu et al., 2025a; Cao et al., 2025). DiT-Reward connects these threads by using the generative backbone as the representation source for reward modeling.

## 3 METHOD

## 3.1 PROBLEM FORMULATION

We consider the problem of learning a reward model for text-to-image generation. Given a text prompt $p$ and a pair of generated images $( x ^ { + } , x ^ { - } )$ , where $x ^ { + }$ is preferred over $x ^ { - }$ according to human judgments, our goal is to learn a reward function $r _ { \theta } ( x , p )$ that assigns higher scores to preferred images. Formally, we seek to learn $r _ { \theta }$ such that $r _ { \theta } ( x ^ { + } , p ) > r _ { \theta } ( x ^ { - } , p )$ for preference pairs in the training data. The learned reward model can then be applied to image ranking or as a reward signal for downstream reinforcement learning optimization of text-to-image models.

## 3.2 GENERATOR-NATIVE REWARD MODELING WITH DIT

Most existing text-to-image reward models are built on discriminative vision-language encoders, such as CLIP-style or VLM backbones. DiT-Reward instead uses the pretrained text-to-image generator itself as the reward backbone. The motivation is that a DiT-based generator must internally model visual quality, prompt-image alignment, composition, and generation feasibility in order to synthesize images. Rather than introducing a separate vision-language encoder, we reuse these generator-native representations for reward prediction.

Given a prompt-image pair $( p , x )$ , the prompt is encoded by the frozen text encoders of the textto-image model, and the image is encoded into the VAE latent space. Since the DiT backbone is trained to process noisy latent states rather than perfectly clean image latents, directly feeding the clean latent would introduce an input-distribution mismatch. We therefore construct a near-clean latent by adding a small amount of Gaussian noise:

$$
z _ {\tau} = (1 - \tau) z _ {0} + \tau \epsilon , \quad \epsilon \sim \mathcal {N} (0, I), \quad 0 <   \tau \ll 1,\tag{1}
$$

where $z _ { \mathrm { 0 } }$ denotes the VAE latent of x and τ is a small constant. This perturbation places the image latent in a flow-time state that is more compatible with the pretrained DiT. The noise level must remain small: large noise would corrupt the image content and make the reward model score a substantially perturbed image rather than the original candidate.

The near-clean image latent and the text embeddings are then processed jointly by the pretrained MMDiT backbone. Let $\boldsymbol { S } = \{ \ell _ { 1 } , \ell _ { 2 } , \dots , \ell _ { K } \}$ denote the selected DiT layers. For each layer $\ell \in S$ we extract the image-token hidden states:

$$
H _ {\ell} (x, p) \in \mathbb {R} ^ {N \times d},\tag{2}
$$

where $N$ is the number of image tokens (spatial positions in the latent space) and d is the hidden dimension of the MMDiT.

We aggregate each layer’s image-token representation by mean pooling:

$$
f _ {\ell} (x, p) = \frac {1}{N} \sum_ {i = 1} ^ {N} H _ {\ell} ^ {(i)} (x, p),\tag{3}
$$

and concatenate the pooled features across layers:

$$
f (x, p) = \operatorname{Concat} \left(f _ {\ell_ {1}} (x, p), f _ {\ell_ {2}} (x, p), \dots , f _ {\ell_ {K}} (x, p)\right) \in \mathbb {R} ^ {K \cdot d}.\tag{4}
$$

This multi-layer aggregation exposes reward-relevant information from different depths of the generative model. The concatenated feature is then passed to a lightweight MLP reward head, which progressively projects the high-dimensional DiT representation into a scalar reward:

$$
r _ {\theta} (x, p) = \operatorname{MLP} (f (x, p)) \in \mathbb {R}.\tag{5}
$$

![](images/540a1efff962701cd1eece905b1d7ba0b09399a3f38ad5b17737b07a91923355.jpg)

![](images/84fa971f353dcd2a65ddc657f387db073d6db082dc4231482eb8f530d316c352.jpg)

![](images/9c8a700d6a6162f0424c38c74a67166be602ddec9c2247e95d5a3d715adcba07.jpg)  
Figure 1: Model architecture, preference training, and reinforcement learning interface of DiT-Reward. (a) Given a prompt and image pair, the VAE maps the image into latent space, and a small amount of noise produces a nearly clean latent. A pretrained DiT or MMDiT jointly processes the text condition and image latent. Image token representations from selected layers are pooled, concatenated, and mapped to a scalar reward by a lightweight reward head. (b) For preferred and rejected images under the same prompt, DiT-Reward with shared parameters predicts two rewards and learns their ordering through the Bradley–Terry loss. (c) During reinforcement learning, a conventional pixel interface requires VAE decoding followed by visual encoding. When the policy and DiT-Reward share a latent space, the red path sends the generated latent directly to the reward model, avoiding additional decoding and encoding while reducing inference cost and peak memory.

## 3.3 PREFERENCE LEARNING OBJECTIVE

We train DiT-Reward with the standard Bradley-Terry preference objective. Given a prompt p and a preferred/rejected image pair $( x ^ { + } , x ^ { - } )$ , the two images are scored by the same reward model. We then optimize

$$
\mathcal {L} (\theta) = - \log \sigma \left(r _ {\theta} (x ^ {+}, p) - r _ {\theta} (x ^ {-}, p)\right),\tag{6}
$$

where $\sigma ( \cdot )$ is the sigmoid function. This loss increases the reward margin between human-preferred and rejected images. In our main setting, the text encoders and VAE remain frozen, while the DiT backbone and reward head can be fine-tuned for preference prediction; we also study a frozenbackbone variant in the ablation section to probe how much reward-relevant information already exists in the pretrained generator.

## 3.4 REINFORCEMENT LEARNING IN THE LATENT SPACE

As a general reward model, DiT-Reward can be used in the same way as conventional image reward models: a policy model first generates a complete image, and the reward model scores the resulting prompt-image pair. This makes DiT-Reward directly applicable to reranking and reward-based policy optimization.

Its generator-native design also enables a more efficient pathway when the policy model and reward model share the same latent space. Conventional VLM-based reward models require the policy latent to be decoded into pixels and then processed again by an external vision encoder. In contrast, DiT-Reward can score candidates from the shared latent representation after generation, avoiding unnecessary decoding and re-encoding steps. This latent-space reward evaluation reduces redundant computation and memory usage, which is especially useful for reinforcement learning with large diffusion or flow policies.

## 4 EXPERIMENT

## 4.1 EXPERIMENTAL SETUP

We instantiate DiT-Reward from Stable Diffusion 3.5 Large by reusing its MMDiT backbone as the reward feature extractor. This backbone has approximately 8B parameters and consists of 38 joint transformer layers that process VAE image-latent tokens together with text tokens formed from the concatenated embeddings of CLIP-L, OpenCLIP bigG, and T5-XXL. In our default implementation, image-token hidden states are extracted from layers 15, 23, 28, and 37, where the layer index starts from 0, and the pooled multi-layer representation is fed into a lightweight MLP reward head. Additional backbone configuration details are provided in Appendix A.

To focus the comparison on reward-model architecture rather than supervision, we adopt the public HPSv3 training recipe. The training mixture includes HPDv3 pairwise data, a filtered golden subset from HPDv2, sampled Pick-A-Pic data, sampled ImageReward data, and Midjourney user-choice data, without introducing additional data sources.

During training, images are encoded into the SD3.5 latent space and perturbed with Gaussian noise using $\tau = 0 . 0 0 5$ . We train for 2–3 epochs with a warmup ratio of 0.05. The VAE and text encoders remain frozen. In the main setting, the DiT backbone and reward head are trainable, with learning rates $5 \times 1 0 ^ { - 6 }$ and $2 \times 1 0 ^ { - 5 }$ , respectively. The reward head has ∼4M–16M parameters depending on layer count; Appendix A reports training and benchmark trajectories.

Table 1: Main reward-model benchmark results. We report pairwise preference prediction accuracy (%) on ImageReward, PickScore, HPDv2, and HPDv3. Best results are shown in bold and secondbest results are underlined. For the HPSv3 PickScore entry marked by <sup>∗</sup>, we evaluate the official released HPSv3 checkpoint with the official PickScore evaluation protocol.

<table><tr><td>Model</td><td>ImageReward</td><td>PickScore</td><td>HPDv2</td><td>HPDv3</td></tr><tr><td>CLIP ViT-H/14</td><td>57.1</td><td>60.8</td><td>65.1</td><td>48.6</td></tr><tr><td>Aesthetic Score Predictor</td><td>57.4</td><td>56.8</td><td>76.8</td><td>59.9</td></tr><tr><td>ImageReward</td><td>65.1</td><td>61.1</td><td>74.0</td><td>58.6</td></tr><tr><td>PickScore</td><td>61.6</td><td>70.5</td><td>79.8</td><td>65.6</td></tr><tr><td>HPS</td><td>61.2</td><td>66.7</td><td>77.6</td><td>63.8</td></tr><tr><td>HPSv2</td><td>65.7</td><td>63.8</td><td>83.3</td><td>65.3</td></tr><tr><td>MPS</td><td>67.5</td><td>63.1</td><td>83.5</td><td>64.3</td></tr><tr><td>HPSv3</td><td>66.8</td><td>65.1*</td><td>85.4</td><td>76.9</td></tr><tr><td>DiT-Reward (Ours)</td><td>67.0</td><td>66.7</td><td>85.6</td><td>77.6</td></tr></table>

## 4.2 MAIN RESULTS ON PREFERENCE BENCHMARKS

Table 1 compares DiT-Reward with representative reward models on pairwise preference prediction benchmarks. DiT-Reward achieves the best performance on HPDv3, reaching 77.6%. We regard

HPDv3 as the most important benchmark in this comparison because it contains images generated by more recent text-to-image models and is therefore more indicative of current reward-model performance. DiT-Reward also achieves the best result on HPDv2 with 85.6%, ranks second on ImageReward, and ties for second on PickScore. Notably, DiT-Reward outperforms HPSv3 on all four benchmarks while using the same training data mixture.

## 4.3 REINFORCEMENT LEARNING WITH DIT-REWARD

We evaluate DiT-Reward as a reward signal for text-to-image reinforcement learning using Flow-GRPO (Liu et al., 2025). We use Stable Diffusion 3.5 Large as the policy and follow the official Flow-GRPO PickScore prompt configuration for both RL training and evaluation. Training images are generated at 512×512 resolution with 10 sampling steps and guidance scale 4.5, while evaluation uses 40 sampling steps. We use a group size of 16 and, through distributed gradient accumulation across 16 GPUs, an effective optimization batch size of 384 samples per update. The policy is optimized with LoRA, with $\beta \stackrel { - } { = } 0 . 0 1$ controlling the strength of KL regularization. The preceding policy, sampling, and optimization settings are matched between the DiT-Reward and HPSv3 runs, while each reward model retains its native scoring pathway; in particular, DiT-Reward scores samples in latent space.

For external evaluation, we use the 2,048 PickScore prompts from the official Flow-GRPO protocol. GPT-5 and Gemini-3-Flash evaluate each generated image independently at 60-step intervals along four dimensions: prompt alignment, visual quality, realism, and detail richness. The complete prompts and evaluation protocol are provided in Appendix C. For qualitative analysis, we select the best-performing checkpoint from each training run and use the same selected checkpoint for all examples from that policy in Figure 3.

(a) DiT-Reward eval  
![](images/0a6baac1e16bcdd00793017a4f6b5f0043f5c1c44b160374d1b59fd6d84781de.jpg)

(b) GPT judge  
![](images/dd52bf001de862544a549f4501e8c23e8e007269cfdbbf8f26eb38fabde13776.jpg)

(c) Gemini judge  
![](images/f07f6e73c8d69c4aff0b4c090511d327e229ad2f9ccd65ebfcd39444436585e0.jpg)  
Figure 2: Internal and external evaluation during Flow-GRPO training. Panel (a) shows mean DiT-Reward on evaluation samples for the DiT-Reward-trained policy. Panels (b) and (c) compare overall scores from GPT-5 and Gemini-3-Flash for policies trained with DiT-Reward (blue) and HPSv3 (orange). Gray dashed lines denote base-policy scores of 7.29 and 6.53, respectively. Red dashed lines mark step 1020, used as a common reference because visible reward-hacking artifacts emerge at a similar stage in both RL runs.

As shown in Figure 2, Flow-GRPO steadily increases the DiT-Reward evaluation score from −0.94 to 5.06, confirming that the policy can optimize the learned reward. Across the 18 matched evaluation checkpoints from steps 60 to 1080, replacing HPSv3 with DiT-Reward increases the average overall score from 7.18 to 7.41 under GPT-5 and from 6.61 to 7.02 under Gemini-3-Flash. DiT-Reward achieves a higher overall score at every matched checkpoint under both judges. The gains are concentrated in visual quality, realism, and detail richness, while HPSv3 retains a modest advantage in prompt alignment, particularly during later training. For the DiT-Reward run, the internal reward continues to rise after the external scores peak. Qualitative inspection indicates that visible reward hacking emerges at a similar stage in both reward-model runs, so step 1020 is shown as a common reference threshold. Appendix C reports complete trajectories for all four dimensions.

![](images/34f86eca89cf6809530a67a17e723eccd3ab9ea29fee13eaba06dd3f6e51a504.jpg)  
Figure 3: Qualitative comparison using the best-performing checkpoint from each training run. All examples from a given policy are sampled from the same selected checkpoint, and each row uses the same prompt to compare the base policy with policies optimized using HPSv3 or DiT-Reward.

Figure 3 compares matched-prompt generations from the base policy and policies optimized with HPSv3 and DiT-Reward. Before pronounced reward overoptimization, the policy optimized with

DiT-Reward produces richer local details and more realistic portraits, animals, and objects. It also exhibits less drift toward excessive brightness, saturation, and contrast than the policy optimized with HPSv3, avoiding some unnatural colors and distorted shapes.

Both reward models eventually exhibit reward hacking: faces and objects become frontal and rigid, hallucinated details accumulate, and outputs converge toward reward-favored styles at the expense of prompt fidelity. Their failure modes nevertheless differ. HPSv3 tends to amplify saturation and contrast, sometimes producing implausible colors and geometry, whereas DiT-Reward better preserves global structure but may yield muted colors, spurious faces, and weaker prompt alignment. These differences indicate that reward models trained on the same preference data can induce distinct optimization biases because of their pretraining objectives and backbone architectures. A plausible explanation is that the diffusion backbone provides stronger structural priors, while the VLM-based reward retains an advantage in text-instruction following. One possible direction for future work is to incorporate text-token representations from the joint transformer and explicitly model text–image interactions to improve the prompt-alignment sensitivity of DiT-Reward.

## 4.4 EFFICIENT REWARD INFERENCE

A key advantage of DiT-Reward is that it can directly evaluate generated latents when the reward model and policy share the same latent space. This pathway avoids VAE decoding and pixel-space re-encoding by an external vision backbone. We compare its inference efficiency with HPSv3 and the pixel-interface DiT-Reward pathway in Table 2.

Table 2: Reward inference efficiency on $5 1 2 \times 5 1 2$ images measured on one NVIDIA RTX PRO 6000 Blackwell GPU with batch size 1. Each mode uses one warmup iteration excluded from timing. The reward interface indicates whether the model receives decoded pixels or policy latents. Lower values are better.

<table><tr><td>Reward Model</td><td>Reward Interface</td><td>Per-Image Latency</td><td>Peak Memory</td></tr><tr><td>DiT-Reward</td><td>Pixel</td><td>154 ms</td><td>27.2 GB</td></tr><tr><td>HPSv3</td><td>Pixel</td><td>89 ms</td><td>16.0 GB</td></tr><tr><td>DiT-Reward</td><td>Latent</td><td>54 ms</td><td>16.1 GB</td></tr></table>

Although the complete DiT-Reward inference stack contains an 8B DiT backbone and approximately 5B additional parameters in its frozen text encoders, the latent-interface pathway achieves the lowest inference latency: 54 ms per image compared with 89 ms for HPSv3, corresponding to a 1.65× speedup. Its peak memory consumption is 16.1 GB, nearly identical to the 16.0 GB required by HPSv3. DiT-Reward with the pixel interface instead requires 154 ms per image and 27.2 GB of memory. Together with the improved reward-model and downstream RL performance reported above, these results indicate that aligning the policy and reward model in a shared latent representation space can improve performance without sacrificing inference efficiency. In particular, the aligned latent interface enables a large generative reward model to achieve faster inference with a memory footprint comparable to HPSv3, highlighting representation-space alignment as both a modeling and systems advantage for text-to-image reinforcement learning.

## 5 ABLATION AND INTERPRETABILITY

We organize our analysis around two objectives. First, we examine whether representations from a pretrained generative DiT can support reward modeling with a frozen backbone and identify which layers are most informative for this task. Second, we investigate whether reward modeling performance exhibits scaling behavior as the parameter count of the generative backbone increases.

## 5.1 REWARD-RELEVANT REPRESENTATIONS IN GENERATIVE DITS

We first examine whether representations learned for text-to-image generation can directly support reward modeling. We freeze the pretrained SD3.5-Large DiT and train only a lightweight reward head with approximately 10M parameters on top of its intermediate image-token representations. Because these image tokens have been jointly conditioned on the prompt, this setting tests whether the frozen generative backbone already provides image–text features that are sufficiently discriminative for downstream preference prediction, rather than acquiring such features only through full reward-model fine-tuning.

(a) Layer similarity at Step 39  
![](images/5e9562b802617c67061b2e38bbbdf8f101fcaa3b54f8fb0d0865edbe57b63ee6.jpg)  
(c) Single-layer probing

(b) Similarity across denoising steps  
![](images/69f0e52dd34d96aad0ef4c508f98ca73481569836eb5989db0265b609f823625.jpg)  
(d) Two-layer block pairs

![](images/33e1811d51823568eda90cd1de2ae0ce4acb86ea6562a8991e8e5658999dc349.jpg)

![](images/4a7c742220d3c139bdd43a3d05a051ff3e01727b2b6c54f7c5446114ca025da7.jpg)  
Figure 4: Layer organization and reward-relevant representations in the pretrained DiT. (a) Promptaveraged layer similarity at denoising step 39, with B1–B4 denoting the layer ranges used for blockwise probing. (b) Mean similarity within and across blocks over denoising steps; shaded regions show prompt-wise variability. (c) Single-layer probing on HPDv3. (d) Accuracy of two-layer block pairs, with the redundant upper triangle omitted.

Table 3: Effects of backbone adaptation and capacity under identical training data, reward objective, and evaluation protocol. “Frozen” means training only the reward head.

<table><tr><td>Backbone</td><td>Backbone Training</td><td>ImageReward</td><td>PickScore</td><td>HPDv2</td><td>HPDv3</td></tr><tr><td>SD3.5-Medium (2.5B)</td><td>Full adaptation</td><td>57.9</td><td>60.9</td><td>79.7</td><td>71.5</td></tr><tr><td>SD3.5-Large (8.1B)</td><td>Frozen</td><td>57.5</td><td>60.9</td><td>78.0</td><td>72.3</td></tr><tr><td>SD3.5-Large (8.1B)</td><td>Full adaptation</td><td>67.0</td><td>66.7</td><td>85.6</td><td>77.6</td></tr></table>

As shown in Table 3, the head-only model reaches 57.5%, 60.9%, 78.0%, and 72.3% on ImageReward, PickScore, HPDv2, and HPDv3, respectively. A lightweight head therefore extracts substantial preference information from fixed generative representations. Full backbone adaptation adds 9.5, 5.8, 7.6, and 5.3 percentage points, showing that preference fine-tuning further strengthens the pretrained features.

We next investigate how these reward-relevant representations are organized across depth and whether different representation stages provide complementary information. Figure 4 summarizes both the layer geometry and task-level probing results. Panel (a) shows the prompt-averaged layersimilarity matrix at denoising step 39. Its block structure indicates high redundancy within local layer groups and motivates the four probing regions B1–B4. Panel (b) shows that within-block similarity remains near 0.94 throughout denoising, whereas similarity across blocks remains substantially lower at 0.71–0.75. The narrow uncertainty bands are computed from prompt-wise variation. Their mean pairwise standard deviation ranges only from 0.006 to 0.016, indicating that the global layer organization is highly consistent across prompts, although variability gradually increases at later denoising steps. Complete matrices for all steps appear in Appendix B.

To connect this representation geometry to the reward task, we train probes using individual layers and layer-block pairs. Panel (c) shows that reward accuracy is strongly depth-dependent: it rises from 58.71% at layer 0 to 71.98% at layer 24, before declining in the final layers. Panel (d) further shows that features from different stages are complementary. The cross-block B1+B2 pair reaches 72.65%, outperforming the corresponding within-block B1+B1 and B2+B2 pairs. Together, these results show that a pretrained generative DiT already contains image–text representations suitable for reward modeling. The most useful information is concentrated in the middle-to-late layers while remaining distributed across distinct representation stages.

## 5.2 SCALING WITH GENERATIVE BACKBONE CAPACITY

Finally, we study whether DiT-based reward modeling benefits from a larger generative backbone. Under the same training data, reward objective, and evaluation protocol, increasing the backbone from SD3.5-Medium (2.5B) to SD3.5-Large (8.1B) improves ImageReward from 57.9% to 67.0%, PickScore from 60.9% to 66.7%, HPDv2 from 79.7% to 85.6%, and HPDv3 from 71.5% to 77.6%. These consistent gains indicate that larger generative backbones provide stronger representations under fixed supervision and learning objectives.

## 6 CONCLUSION

This work investigated whether text-to-image models can provide effective representations for reward modeling in addition to generating images. We introduced DiT-Reward, which converts a pretrained text-to-image DiT into a reward model. Under the same training data, DiT-Reward outperforms HPSv3 on all four preference benchmarks and achieves the best results on HPDv2 and HPDv3. Experiments that freeze the generative backbone and train only a lightweight reward head show that image representations learned by a pretrained DiT can support downstream preference prediction without further backbone adaptation. Analysis across network layers further shows that representations from the middle and later layers provide stronger reward prediction, while distinct representation stages contain complementary information. Reward performance also improves consistently as the generative backbone becomes larger. In reinforcement learning, DiT-Reward and the policy are both derived from Stable Diffusion 3.5 Large. The resulting policy consistently outperforms the HPSv3 baseline in overall score, visual quality, realism, and detail richness, while direct evaluation of generated latents enables faster reward inference with comparable memory. Although later training still exhibits weaker prompt alignment and reward hacking, our results show that pretrained generative DiTs can serve not only as image generators, but also as transferable representation backbones for evaluating generated results and optimizing related generative policies.

## REFERENCES

Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. Qwen technical report. arXiv preprint arXiv:2309.16609, 2023.

Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine. Training diffusion models with reinforcement learning. In International Conference on Learning Representations, 2024.

Black Forest Labs. FLUX.1 [dev] model card. Technical report, Black Forest Labs, 2024.

Siyu Cao, Hangting Chen, Peng Chen, et al. HunyuanImage 3.0 technical report. arXiv preprint arXiv:2509.23951, 2025.

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Muller, Harry Saini, Yam¨ Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al. Scaling rectified flow transformers for high-resolution image synthesis. In International Conference on Machine Learning, 2024.

Ying Fan, Olivia Watkins, Yuqing Du, Hao Liu, Moonkyung Ryu, Craig Boutilier, Pieter Abbeel, Mohammad Ghavamzadeh, Kangwook Lee, and Kimin Lee. DPOK: Reinforcement learning for fine-tuning text-to-image diffusion models. In Advances in Neural Information Processing Systems, volume 36, 2023.

Xuan He, Dongfu Jiang, Ge Zhang, Max Ku, Achint Soni, Sherman Siu, Haonan Chen, Abhranil Chandra, Ziyan Jiang, Aaran Arulraj, Kai Wang, Quy Duc Do, Yuansheng Ni, Bohan Lyu, Yaswanth Narsupalli, Rongqi Fan, Zhiheng Lyu, Bill Yuchen Lin, and Wenhu Chen. VideoScore: Building automatic metrics to simulate fine-grained human feedback for video generation. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pp. 2105–2123, 2024.

Yuval Kirstain, Adam Polyak, Uriel Singer, Shahbuland Matiana, Joe Penna, and Omer Levy. Picka-pic: An open dataset of user preferences for text-to-image generation. In Advances in Neural Information Processing Systems, 2023.

Junnan Li, Dongxu Li, Caiming Xiong, and Steven Hoi. BLIP: Bootstrapping language-image pretraining for unified vision-language understanding and generation. In International Conference on Machine Learning, pp. 12888–12900. PMLR, 2022.

Jie Liu, Gongye Liu, Jiajun Liang, Yangguang Li, Jiaheng Liu, Xintao Wang, Pengfei Wan, Di Zhang, and Wanli Ouyang. Flow-GRPO: Training flow matching models via online RL. In Advances in Neural Information Processing Systems, 2025.

Grace Luo, Lisa Dunlap, Dong Huk Park, Aleksander Holynski, and Trevor Darrell. Diffusion hyperfeatures: Searching through time and space for semantic correspondence. In Advances in Neural Information Processing Systems, volume 36, 2023.

Yuhang Ma, Yunhao Shui, Xiaoshi Wu, Keqiang Sun, and Hongsheng Li. HPSv3: Towards widespectrum human preference score. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 15086–15095, 2025.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. In Advances in Neural Information Processing Systems, volume 35, pp. 27730–27744, 2022.

William Peebles and Saining Xie. Scalable diffusion models with transformers. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 4195–4205, 2023.

Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Muller, Joe¨ Penna, and Robin Rombach. SDXL: Improving latent diffusion models for high-resolution image synthesis. arXiv preprint arXiv:2307.01952, 2023.

Mihir Prabhudesai, Anirudh Goyal, Deepak Pathak, and Katerina Fragkiadaki. Aligning text-toimage diffusion models with reward backpropagation. arXiv preprint arXiv:2310.03739, 2023.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. Learning transferable visual models from natural language supervision. In International Conference on Machine Learning, pp. 8748–8763. PMLR, 2021.

Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Bjorn Ommer. High- ¨ resolution image synthesis with latent diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10684–10695, 2022.

Stability AI. Introducing stable diffusion 3.5. Technical report, 2024.

Nick Stracke, Stefan Andreas Baumann, Kolja Bauer, Frank Fundel, and Bjorn Ommer. CleanDIFT: Diffusion features without noise. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2025.

Luming Tang, Menglin Jia, Qianqian Wang, Cheng Perng Phoo, and Bharath Hariharan. Emergent correspondence from image diffusion. In Advances in Neural Information Processing Systems, volume 36, 2023.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.

Bram Wallace, Meihua Dang, Rafael Rafailov, Linqi Zhou, Aaron Lou, Senthil Purushwalkam, Stefano Ermon, Caiming Xiong, Shafiq Joty, and Nikhil Naik. Diffusion model alignment using direct preference optimization. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024.

Chenfei Wu, Jiahao Li, Jingren Zhou, Junyang Lin, Kaiyuan Gao, Kun Yan, Shengming Yin, Shuai Bai, Xiao Xu, Yilei Chen, et al. Qwen-Image technical report. arXiv preprint arXiv:2508.02324, 2025a.

Jie Wu, Yu Gao, Zilyu Ye, Ming Li, Liang Li, Hanzhong Guo, Jie Liu, Zeyue Xue, Xiaoxia Hou, Wei Liu, Yan Zeng, and Weilin Huang. RewardDance: Reward scaling in visual generation. arXiv preprint arXiv:2509.08826, 2025b.

Xiaoshi Wu, Yiming Hao, Keqiang Sun, Yixiong Chen, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score v2: A solid benchmark for evaluating human preferences of text-toimage synthesis. arXiv preprint arXiv:2306.09341, 2023a.

Xiaoshi Wu, Keqiang Sun, Feng Zhu, Rui Zhao, and Hongsheng Li. Human preference score: Better aligning text-to-image models with human preference. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 2096–2105, 2023b.

Jiarui Xu, Sifei Liu, Arash Vahdat, Wonmin Byeon, Xiaolong Wang, and Shalini De Mello. Openvocabulary panoptic segmentation with text-to-image diffusion models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 2955–2966, 2023a.

Jiazheng Xu, Xiao Liu, Yuchen Wu, Yuxuan Tong, Qinkai Li, Ming Ding, Jie Tang, and Yuxiao Dong. ImageReward: Learning and evaluating human preferences for text-to-image generation. In Advances in Neural Information Processing Systems, volume 36, pp. 15903–15935, 2023b.

Jiazheng Xu, Yu Huang, Jiale Cheng, Yuanming Yang, Jiajun Xu, Yuan Wang, Wenbo Duan, Shen Yang, Qunlin Jin, Shurun Li, Jiayan Teng, Zhuoyi Yang, Wendi Zheng, Xiao Liu, Dan Zhang, Ming Ding, Xiaohan Zhang, Shiyu Huang, Xiaotao Gu, Minlie Huang, Jie Tang, and Yuxiao Dong. VisionReward: Fine-grained multi-dimensional human preference learning for image and video generation. In Proceedings of the AAAI Conference on Artificial Intelligence, pp. 11269– 11277, 2026.

Aiyuan Yang, Bin Xiao, Bingning Wang, Borong Zhang, Ce Bian, Chao Yin, Chenxu Lv, Da Pan, Dian Wang, Dong Yan, et al. Baichuan 2: Open large-scale language models. arXiv preprint arXiv:2309.10305, 2023.

Wenliang Zhao, Yongming Rao, Zuyan Liu, Benlin Liu, Jie Zhou, and Jiwen Lu. Unleashing textto-image diffusion models for visual perception. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023.

Kaiwen Zheng, Huayu Chen, Haotian Ye, Haoxiang Wang, Qinsheng Zhang, Kai Jiang, Hang Su, Stefano Ermon, Jun Zhu, and Ming-Yu Liu. DiffusionNFT: Online diffusion reinforcement with forward process. In International Conference on Learning Representations, 2026.

## A DIT-REWARD TRAINING DETAILS

We instantiate DiT-Reward with Stable Diffusion 3.5 Large (SD3.5-L). In the official diffusers configuration, the MMDiT transformer contains 38 layers and uses 38 attention heads with head dimension 64, corresponding to a hidden dimension of 2432. The joint text-conditioning dimension is 4096, and the pooled projection dimension is 2048.

The text condition is produced by three frozen pretrained text encoders: CLIP-L/14, OpenCLIP bigG/14, and T5-XXL. The two CLIP token embeddings are concatenated along the channel dimension and padded to match the T5 embedding dimension; the resulting CLIP sequence is then concatenated with the T5 sequence along the sequence dimension and fed to the text side of the joint DiT. The pooled CLIP embeddings are also concatenated and used as global conditioning together with the timestep embedding. For image inputs, a frozen 16-channel VAE encoder maps the image into latent tokens, which are patchified and fed to the image side of the joint DiT. Under the rectified-flow formulation, the DiT takes the latent coordinate $z ,$ text condition $c ,$ and timestep t as inputs and predicts a velocity field in the latent space.

![](images/a6d4bccfc9c613f1bfbd4885c7f86150bd1c97bed8a8aaadec76c513e46b9eb9.jpg)

![](images/a0b5b1f76bb1a7edd797356de2db8c07fcb2c754d062f52df2ad21b0991cf100.jpg)

![](images/c23c2d5ab417dec1740d226cd51cb057c099fb94c424c88d2d5931d91745e68d.jpg)

![](images/a54d3ddb7be7b56d644184c409bbe0d32c51de61318e6c58f8ed75038f1e91d1.jpg)  
Figure 5: Training dynamics of DiT-Reward. Top-left: training loss. Top-right: ImageReward benchmark accuracy. Bottom-left: HPDv2 benchmark accuracy. Bottom-right: HPDv3 benchmark accuracy. Dashed lines indicate the corresponding baseline scores.

## B LAYER SIMILARITY ACROSS DENOISING STEPS

Figure 6 provides the complete layer-similarity statistics underlying the compact analysis in Figure 4. The prompt-averaged matrices retain a similar global block structure throughout denoising, demonstrating that the organization is not specific to a single step. Prompt-wise variability remains low relative to the similarity values, supporting the consistency of this structure across prompts. The variability nevertheless increases toward later denoising steps and is concentrated in interactions involving shallow and deep layers, suggesting increasing prompt-dependent specialization while the global layer organization remains stable.

![](images/cee9228e7b290b42c748667416b05e81c78486f7f5593c018ebade2a040915d0.jpg)  
Figure 6: Layer similarity throughout denoising. The top row shows prompt-averaged similarity among all 38 MMDiT layers, and the bottom row shows the corresponding standard deviation across prompts. Columns report denoising steps 0, 10, 20, 30, and 39.

## C FINE-GRAINED EXTERNAL VLM EVALUATION DURING RL TRAINING

We use GPT-5 and Gemini-3-Flash as external judges. At each evaluation checkpoint, every generated image is evaluated independently against its text prompt. For each policy and checkpoint, we average each numeric field across the 2,048 independently evaluated images. The overall score is returned directly by the judge rather than recomputed from the four dimensions; the rationale field in the JSON response is not used in the quantitative analysis. Figure 7 reports the resulting trajectories for visual quality, realism, detail richness, and prompt alignment.

![](images/08814bae288d8855588c0556f659334a325ddb46c9a750eaa2fb94ec0cfe39ad.jpg)

![](images/fe2948929ce79d40f3c4b4b2dcac13411b56e8f46d80618ce220b500ce80a99c.jpg)  
(a) Visual quality

![](images/9e6382220792a055475cbf20c01fe02965c764f42aa1ea11df6f4bdfd3988132.jpg)

(b) Realism  
![](images/fb740f4f50f6eef089993c21654d034c8714d65c94da4057dd3630c68fddb072.jpg)

![](images/7a4c1b6c1bf926367aab3243812b8fe609b98533056dd2ab48b36d45cc4f2435.jpg)

![](images/589d622c0e8a83fc5e33c0826f1f8d65f50afeffb6daf090c0265cdec2baeca8.jpg)  
(c) Detail richness

![](images/da5ed80849f91fb39df4dc8e1311b20ad2ec2cf483310e0ff4b4496bad0e2084.jpg)

![](images/c2df93c99e14a1101a166f8cac8ad90f96e9df6da8d430259110781a25419096.jpg)  
(d) Prompt alignment  
Figure 7: Fine-grained external evaluation throughout RL training. Each panel reports scores from the GPT-5 and Gemini-3-Flash judges.

Across the 18 matched checkpoints from steps 60 to 1080, DiT-Reward consistently outperforms the HPSv3 baseline in visual quality, realism, and detail richness. Under GPT-5, the average gains in these three dimensions are 0.39, 0.38, and 0.49 points, respectively; under Gemini-3-Flash, the corresponding gains are 0.69, 0.62, and 0.68 points. All three dimensions favor DiT-Reward at every matched checkpoint under both judges. Prompt alignment follows a different trajectory: the two policies are comparable during early training, but DiT-Reward falls behind HPSv3 at later checkpoints, yielding average differences of −0.09 under GPT-5 and −0.10 under Gemini-3-Flash. Together, these results show that the overall advantage of DiT-Reward is driven by stronger visual quality, realism, and detail richness, alongside a trade-off in prompt alignment.

Both judges receive the same system and user prompts. The system prompt is:

```txt
You are an expert evaluator for text-to-image generation models. Your task is to assess how well a generated image matches a given text prompt. You must return a JSON object and nothing else.
```

The user prompt is shown below, where {prompt} is replaced by the corresponding text prompt for each image:

```txt
Evaluate the generated image against the following text prompt:
TEXT PROMPT: " {prompt}"
Score the image on 4 dimensions (1--10, use the full range):
```

```snap
1. prompt_alignment --- Does the image depict what the prompt describes? (subject, count, attributes, scene)
2. visual_quality --- Is the image sharp, well-composed, free of artifacts?
3. realism --- Does it look like a plausible, coherent scene?
4. detail_richness --- Are relevant details present and well-rendered?
Also give an overall score (1--10, one decimal) that holistically reflects all four dimensions.
Return ONLY valid JSON (no markdown, no explanation):
{
    "prompt_alignment": <int 1-10>,
    "visual_quality": <int 1-10>,
    "realism": <int 1-10>,
    "detail_richness": <int 1-10>,
    "overall": <float 1-10, one decimal>,
    "rationale": "<one sentence>"
}
```