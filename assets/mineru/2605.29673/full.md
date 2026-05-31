# A Geometric View of SRC: Learning Representations for Stable Residual Inference

Vangelis P. Oikonomou

viknmu@gmail.com

# Abstract

Reconstruction-based inference assigns a class by comparing class-wise reconstruction residuals; Sparse Representation Classification (SRC) is a canonical instance whose reliability depends on the geometry of the learned representation. We adopt a strict training–inference separation: SRC is used only as a fixed test-time rule and is never differentiated, unrolled, or optimized during training. In a span-level idealization based on class-conditional spans and their associated projection residuals, we formalize residual-ordering stability through a residual margin and characterize geometric obstructions—span overlap, dominance, and near-overlap via small principal angles—that can collapse this margin in worst-case directions. This span-level theory is primary: it specifies when the idealized residual family is well-separated, and it provides a conditional solver-level interpretation for practical residual approximations (e.g., OMP) insofar as they remain close to the span-level residual ordering. Under explicit coverage and separation assumptions, we derive a quantitative lower bound on the (idealized) residual margin. Guided by these targets, we propose geometryshaping objectives that promote masked within-class self-expressiveness, discourage cross-class reconstruction pathways and inter-class span alignment, and prevent collapse—without invoking SRC residuals or predictions during training. Experiments on images (COIL-100), text (TREC), and EEG connectivity evaluate all representations under identical fixed SRC/OMP inference and report residual margins and geometric diagnostics; cross-entropy is included only as a reference geometry under the same evaluation protocol.

Keywords: Reconstruction-based inference, Sparse Representation Classification, geometry-shaped representations, residual margins, stability

# 1 Introduction

Reconstruction-based inference attributes an input to the class that can best reconstruct its representation from class-specific training examples. Unlike decision-boundary reasoning, this paradigm does not rely on calibrated probabilities or discriminative scores: class assignment is determined by comparing class-wise reconstruction residuals, and the reliability of the decision is governed by the residual ordering and its margin. Sparse Representation Classification (SRC) is a canonical reconstruction-based rule. Given a test embedding, SRC computes a global sparse code over the full training dictionary and then evaluates, for each class, a residual obtained by restricting coefficients to that class (Wright et al., 2009). This inference principle is closely related to sparse approximation and sparse recovery methods (Donoho, 2006; Tropp and Gilbert, 2007), but its decision mechanism is fundamentally residual-based rather than discriminative. A recurring difficulty is that SRC-style inference is only as meaningful as the geometry of the representation on which it operates. If multiple classes can reconstruct the same embedding nearly equally well, the residual margin becomes small and the rule becomes ambiguous (in worst-case directions), even if the implementation is numerically stable. This observation shifts emphasis from learning a boundary to learning a geometry in which residual comparisons are interpretable.

This paper adopts a strict methodological stance: SRC is treated as a fixed inference principle, not a module to be trained. SRC is never differentiated, optimized, approximated by a trainable surrogate, or otherwise coupled to training. Training and inference are separated both conceptually and algorithmically: representation learning shapes geometry; inference applies an unaltered residual rule on frozen embeddings. In the experiments, cross-entropy (CE) is included only as a reference way to shape embeddings under standard supervised training. Under this separation, the central question becomes: what geometric properties of learned embeddings make residual-based inference well-defined and stable?

Our analysis treats class-conditional structure through empirical spans/subspaces and their relative arrangement (e.g., principal angles), reflecting the broader view of data as approximately drawn from a union of low-dimensional subspaces (Elhamifar and Vidal, 2013; Soltanolkotabi and Candés, 2012; Liu et al., 2013). The relevant stability object is the residual margin, which directly certifies when the argmin over class residuals is robust to perturbations. To isolate intrinsic geometric structure from solver-side effects, we analyze a span-level idealization. Each class induces an empirical subspace (the span of its training embeddings), and class-wise residuals reduce to distances from a test embedding to these subspaces. This abstraction leverages standard geometric objects for subspace relations, including principal angles (Bjorck and Golub, 1973; Knyazev and Argentati, 2002), and it lets us express stability and ambiguity as properties of geometry rather than as properties of a particular sparse solver. In practice, SRC/OMP instantiates a practical residual family that approximates these span-level residuals; throughout, solver-level conclusions are therefore interpreted conditionally, insofar as the solver-induced residual ordering remains close to the span-level ordering.

A first set of results records structural obstructions under the span idealization. If two class spans intersect, there exist embeddings with zero residual for multiple classes, forcing the residual margin to vanish. Even without exact intersection, near-overlap—quantified by small principal angles—implies the existence of directions with small competing residuals, again collapsing margins (Bjorck and Golub, 1973; Knyazev and Argentati, 2002). A second obstruction concerns span inflation. If one class span contains (or effectively dominates) another, then points from the smaller span can be reconstructed perfectly by the larger one, eliminating any possibility of a positive margin. Together, these observations formalize that (under the span idealization) stability cannot be guaranteed in worst-case directions when the induced geometry contains overlap, containment, or near-overlap; they also identify geometric failure modes that any solver-side approximation must confront when it attempts to realize class-wise residual comparisons.

Complementing necessity statements, we provide a sufficient geometric condition yielding a quantitative residual-margin lower bound. Under explicit assumptions capturing (i) coverage (the test embedding decomposes into an in-class component plus bounded perturbation), (ii) separation (a lower bound on principal angles between the true class span and all others), and (iii) non-degenerate signal scale, we derive a lower bound on the subspace residual margin. This clarifies precisely what geometry shaping can guarantee for a fixed residual rule in an idealized subspace setting, and why such guarantees are inherently conditional.

Guided by these targets, we introduce geometry-shaping objectives that operate only at training time and never invoke SRC residuals, margins, predictions, or accuracy-driven losses. The objectives promote masked self-expressiveness within class, suppress cross-class reconstruction pathways, prevent collapse via a non-discriminative anchoring term, and discourage inter-class span alignment via a subspace repulsion penalty. The use of self-expressiveness as a geometric regularizer is aligned with established subspace modeling principles (Elhamifar and Vidal, 2013; Liu et al., 2013) and with classical perspectives on sparse representation and dictionary modeling (Rubinstein et al., 2010; Aharon et al., 2006).

We summarize the contributions as follows: (i) a strict training–inference separation in which SRC remains fixed and is applied only at test time on frozen embeddings (Wright et al., 2009); (ii) a span-level geometric framework for residual inference centered on class spans, distances-tospans, and residual margins as stability certificates; (iii) impossibility/necessity results showing that overlap, dominance, and near-overlap force margin collapse via principal-angle geometry (Bjorck and Golub, 1973; Knyazev and Argentati, 2002); (iv) a sufficient condition providing a quantitative residual-margin lower bound under explicit coverage and separation assumptions; and (v) geometryshaping training objectives aligned with these conditions without coupling learning to the inference rule, together with a conditional solver-level interpretation for practical SRC/OMP residuals as approximations of the span-level residual family.

Finally, we emphasize scope. Our statements characterize stability as a property of the learned geometry under residual-based inference; they do not imply Bayes-optimality, calibrated probabilities, or a complete solver-theoretic account of the practical global sparse-coding plus class-restriction pipeline (Donoho, 2006; Tropp and Gilbert, 2007). Rather, the span-level theory remains primary: it provides a principled account of when reconstruction-based inference is interpretable (via residual margins) and when it is intrinsically unstable in worst-case directions. Solver-level behavior is discussed only through a conditional, interpretive lens—as a stability-transfer statement that applies when the solver-side discrepancy between practical residuals (e.g., OMP-based) and the span-level residual ordering is controlled. This framing is also compatible with empirical and theoretical observations that standard cross-entropy training can induce highly structured terminal feature geometry (e.g., neural collapse) that may be favorable for residual-based rules in some regimes (Papyan et al., 2020; Lu and Steinerberger, 2022; Han et al., 2022).

The remainder of the paper is organized as follows. Section 2 reviews related work on reconstruction-based inference and subspace modeling. Section 3 develops the span-level stability framework for fixed SRC inference: it characterizes when residual decisions fail or are stable under geometric assumptions, derives conditional residual-margin bounds, and introduces geometry-shaping objectives together with an explicit training–inference separation and a conditional solver-level interpretation for OMP-based residual approximations. Section 4 reports experiments and response-surface analyses under fixed SRC/OMP inference. Section 5 provides conclusions and future directions of the current study.

# 2 Related Work

Reconstruction-based classification and SRC. Sparse Representation Classification (SRC) popularized residual-based decision rules in which a test point is assigned to the class achieving the smallest class-wise reconstruction residual under a global sparse code (Wright et al., 2009). A closely related line replaces strict sparsity with collaborative (typically ℓ2-regularized) representations while retaining the residual-based decision mechanism (Zhang et al., 2014). Furthermore, probabilistic variants such as ProCRC preserve the residual-comparison mechanism while providing alternative regularization and modeling interpretations (Cai et al., 2016). Recent SRC formulations keep the residual-comparison decision rule unchanged while altering the coding dictionary via augmentation (e.g., extended design matrices) (Oikonomou et al., 2024). These works motivate our focus on residual comparisons and, in particular, on the residual margin as the natural stability certificate for reconstruction-based inference. In contrast to approaches that treat SRC as a trainable component, our methodology keeps SRC fixed and confines learning to representation geometry.

Sparse coding, sparse recovery, and dictionary modeling. SRC inference relies on sparse approximation over a training dictionary, connecting it to the broader literature on sparse recovery and pursuit algorithms, including compressed sensing and orthogonal matching pursuit (Donoho, 2006; Tropp and Gilbert, 2007). A complementary thread studies how dictionaries are constructed or learned for sparse modeling, including overcomplete dictionary learning and synthesis models (Aharon et al., 2006; Rubinstein et al., 2010). In our paper, these results play a supporting role: they motivate solver-side implementations of residual computations and clarify the distinction between geometry (what the learned representation makes possible at the level of class-conditional spans) and algorithmics (how a particular sparse solver approximates the intended class-wise residual comparisons). Accordingly, our theoretical statements are span-level, with solver-level connections treated conditionally through approximation fidelity.

Union-of-subspaces perspective and self-expressiveness. Modeling data as (approximately) lying in a union of low-dimensional subspaces underlies both classical subspace clustering and the intuition behind reconstruction-based inference. Sparse Subspace Clustering (SSC) uses selfexpressiveness (representing each point as a sparse combination of others) to recover subspacepreserving affinities under suitable conditions (Elhamifar and Vidal, 2013), and Low-Rank Representation (LRR) uses low-rank self-representation to promote global subspace structure (Liu et al., 2013). Geometric analyses of subspace clustering characterize regimes in which self-expressiveness yields subspace-preserving behavior and how outliers or coherence can affect performance (Soltanolkotabi and Candés, 2012). Greedy neighbor-selection variants further connect self-expressiveness to pursuitstyle constructions, e.g., generalized OMP formulations for SSC (Wu et al., 2023). While our setting is supervised and our test-time inference is SRC (not clustering), these works supply the geometric vocabulary we adopt: within-class self-expressiveness and low effective rank as coverage surrogates, and cross-class separation via limited alignment/overlap as a prerequisite for stable residual comparisons.

Subspace geometry and principal-angle notions. Quantifying subspace overlap and nearoverlap naturally leads to principal angles and standard notions of distance between linear subspaces (Bjorck and Golub, 1973; Knyazev and Argentati, 2002). Recent theory in sequential/iterative projection settings also highlights principal angles as informative for residual-type quantities and stability, through connections to alternating projections and related methods (Evron et al., 2022). These notions align directly with our obstruction results: exact overlap (nontrivial intersection) and near-overlap (a small smallest principal angle) imply the existence of directions with vanishing or arbitrarily small residual margins, rendering residual-based inference intrinsically ambiguous in worst-case directions. Accordingly, we use a span-level idealization that expresses class-wise residuals as distances-to-subspaces and links stability to principal-angle separation. Our use of principal-angle separation as a proxy for residual ambiguity aligns with analyses of subspace classification that relate principal angles to classification behavior in idealized settings (Huang et al., 2016).

Deep subspace methods and learned self-expressiveness. Recent work embeds subspace modeling into deep representations, often by introducing a self-expressive mechanism in the latent space. Deep Subspace Clustering Networks (DSCN) place a self-expressive layer between an encoder and decoder to learn representations amenable to subspace clustering (Ji et al., 2017). In (Peng et al., 2020) develop a deep extension of subspace clustering (DSC) that combines nonlinear feature learning with a self-expressiveness objective. Scaling issues induced by the n × n self-representation matrix have motivated alternative formulations; for example, in (Cai et al., 2022) propose a more efficient embedded subspace clustering approach that avoids explicitly optimizing a full self-representation matrix. Surveys of subspace clustering increasingly cover these deep variants and systematize design choices in nonlinear subspace learning (Miao et al., 2025). Recent self-expressive representation learning methods further develop scalable objectives for enforcing self-reconstruction structure in deep embeddings (Zhao et al., 2024). These works are conceptually adjacent to our training-time geometry shaping through masked self-expressiveness. However, our focus differs in two key respects: (i) our objective is not clustering but stability of residual-based classification under fixed SRC inference; and (ii) we enforce a strict training–inference separation, using self-expressiveness only as a geometric regularizer rather than as a trainable surrogate for the inference rule.

Deep SRC and end-to-end sparse-representation classifiers. A separate line integrates sparse representation mechanisms into neural architectures for classification, effectively coupling representation learning to sparse coding or its approximation. For example, in (Abavisani and Patel, 2019) propose a deep formulation in which a network learns features and includes a component responsible for sparse representation-based classification. Related end-to-end pipelines explicitly couple reconstruction and classification objectives within a single trainable system (Cao et al., 2022). Such approaches are valuable points of comparison because they blur the boundary between representation learning and SRC-style inference. Our methodology explicitly departs from this direction: SRC is never embedded as a differentiable module, never optimized during training, and is applied only at test time as a fixed residual rule on frozen embeddings.

Differentiation through optimization, unrolling, and implicit gradients. A broad modern toolkit differentiates through iterative algorithms or optimization problems, either by unrolling a solver for a finite number of steps or by implicit differentiation of optimality conditions. Algorithmunrolling surveys emphasize how iterative methods can be turned into trainable networks while retaining some interpretability (Monga et al., 2020). Implicit differentiation has been developed for nonsmooth and structured problems, including Lasso-type models (Bertrand et al., 2020), and more generally for modular differentiation of optimization layers (Blondel et al., 2022). Related work also studies computationally efficient approximations such as one-step differentiation of iterative algorithms (Bolte et al., 2023). We cite these works to delineate scope: although such techniques enable end-to-end training with optimization layers, our paper intentionally avoids this coupling. We do not backpropagate through sparse coding, do not use solver outputs to define training losses, and do not reinterpret SRC as a trainable layer.

Positioning and methodological gap. Taken together, prior work provides (i) fixed residualbased inference rules (SRC/CRC), (ii) geometric modeling principles (self-expressiveness, lowrank/union-of-subspaces), and (iii) deep formulations that often couple inference mechanisms to training through unrolling or optimization layers. The gap addressed here is methodological and geometric: we characterize when reconstruction-based residual comparisons are well-posed and stable at a span level as a function of representation geometry, and we design training objectives that shape that geometry without invoking SRC during training. This positioning emphasizes inference reliability (via residual margins) rather than discriminative decision boundaries, consistent with the reconstruction-based principle underlying SRC. More broadly, our emphasis on geometry as an organizing principle aligns with analyses that isolate geometric properties of normalized representations (Wang and Isola, 2020) and with perspectives linking sparse and low-rank structure to representation mappings (Saul, 2022).

# 3 Methodology

# 3.1 Overview and methodological contract

We develop a representation learning methodology whose sole purpose is to shape embedding geometry so that reconstruction-based inference via a fixed Sparse Representation Classification (SRC) rule is well-defined and stable. Crucially, SRC is treated strictly as an inference principle: it is never differentiated, optimized, approximated by a trainable surrogate, or otherwise coupled to training. Training and inference are separated both conceptually and algorithmically.

What is fixed (test time). Given a frozen encoder $f _ { \theta }$ and a training embedding dictionary grouped by class, SRC assigns labels using only class-wise reconstruction residuals and their ordering (Algorithm 2). The relevant reliability object is the residual margin—the gap between the smallest and second-smallest class residuals—which certifies when residual-based decisions are robust to perturbations of the residual values (e.g., induced by embedding noise, finite-sample span estimation, or solver-side approximation). Accordingly, our aim is not to learn decision boundaries or calibrated scores, but to learn embeddings for which residual comparisons are interpretable.

What can be learned (training time). Since inference is fixed, training can only act by regularizing embedding geometry. We therefore design objectives that target three geometric requirements for stable residual inference:

R1 Within-class coverage/self-expressiveness: test embeddings from class y should lie close to the empirical class span.   
R2 Limited cross-class reconstructability (low leakage): other classes should not reconstruct a point from class y nearly as well.   
R3 Non-collapse (non-degeneracy): embeddings must avoid trivial collapsed solutions that invalidate residual comparisons.

At no point do we compute SRC residuals, margins, or predictions during training.

Analysis scope: span idealization. To isolate intrinsic geometric structure from solver-specific artifacts, we analyze residuals at the level of empirical class spans $S _ { k } = \operatorname { s p a n } ( Z _ { k } )$ and idealized residuals $r _ { k } ^ { \mathrm { s u b } } ( z ) = \mathrm { d i s t } ( z , S _ { k } )$ . This abstraction lets us (i) state geometry-only obstructions (overlap/near-overlap) that force small margins, and (ii) derive a quantitative margin lower bound under explicit regularity assumptions (coverage, separation, bounded perturbations). These statements are not universal guarantees for a particular sparse solver; they clarify which geometric properties representation learning should promote. The span-level abstraction, where decisions depend on distances to class subspaces, is also consistent with classical matched-subspace detection viewpoints based on residual energy (Scharf and Friedlander, 1994).

# 3.2 Notation and fixed inference principle

Let $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { N }$ be labeled training data with $y _ { i } \in \{ 1 , \ldots , K \}$ . We learn an encoder

$$
f _ {\theta}: \mathcal {X} \rightarrow \mathbb {R} ^ {p}, \quad z _ {i} = f _ {\theta} (x _ {i}), \quad \| z _ {i} \| _ {2} = 1. \tag {1}
$$

All embeddings are $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ . Let $Z _ { \mathrm { t r } } = [ z _ { 1 } , \dots , z _ { N } ] \ \in \ \mathbb { R } ^ { p \times N }$ be the training embedding dictionary (columns grouped by class), and let $Z _ { k } = \{ z _ { i } : \ y _ { i } = k \}$ denote the embeddings of class k. The empirical class span is $S _ { k } : = \operatorname { s p a n } ( Z _ { k } ) \subset \mathbb { R } ^ { p }$ , with orthogonal projector $P _ { S _ { k } }$ . For class-wise residuals $\{ r _ { k } ( z ) \} _ { k = 1 } ^ { K }$ , write $r _ { ( 1 ) } ( z ) \leq r _ { ( 2 ) } ( z ) \leq \cdot \cdot \cdot$ for ordered values and define the residual margin

$$
m (z) := r _ {(2)} (z) - r _ {(1)} (z). \tag {2}
$$

# 3.3 Span idealization and margin-based stability

Subspace residuals. Replacing sparse coding by unconstrained least squares on each class dictionary yields the span-level residual

$$
r _ {k} ^ {\mathrm{sub}} (z) = \operatorname{dist} \left(z, \operatorname{span} \left(Z _ {k}\right)\right) = \| \left(I - P _ {S _ {k}}\right) z \| _ {2}, \quad S _ {k} := \operatorname{span} \left(Z _ {k}\right). \tag {3}
$$

Definition 1 (Well-posed residual inference and margin-based stability) Under the span idealization, define $m ^ { \mathrm { s u b } } ( z ) : = r _ { ( 2 ) } ^ { \mathrm { s u b } } ( z ) - r _ { ( 1 ) } ^ { \mathrm { s u b } } ( z )$ . Residual-based inference is well-posed at $z \ i f$ $m ^ { \mathrm { s u b } } ( z ) > 0$ .

More generally, for $\eta \geq 0$ it is η-stable a $\mid z \ i f m ^ { \mathrm { s u b } } ( z ) > 2 \eta , \ i . e .$ , the argmin of $\{ r _ { k } ^ { \mathrm { s u b } } ( z ) \} _ { k = 1 } ^ { K }$ cannot change under perturbations $\{ \tilde { r } _ { k } \} _ { k = 1 } ^ { K }$ satisfying $\operatorname* { s u p } _ { k } | \tilde { r } _ { k } ( z ) - r _ { k } ^ { \mathrm { s u b } } ( z ) | \leq \eta$ .

Lemma 1 (Margin gap implies argmin stability) Let $a _ { 1 } , \dots , a _ { K } \in \mathbb { R }$ and let $\tilde { a } _ { 1 } , \dots , \tilde { a } _ { K } \in \mathbb { R }$ satisfy max ${ \mathrm { } _ { : } } \left| { \tilde { a } } _ { k } - a _ { k } \right| \le \eta . \mathrm { } \ I f { a } _ { ( 2 ) } - a _ { ( 1 ) } > 2 \eta$ , then arg min $\mathbf { \sigma } _ { \cdot k } a _ { k } = \arg \operatorname* { m i n } _ { k } \tilde { a } _ { k }$ .

Proof Let $i ^ { \star } \in \arg \operatorname* { m i n } _ { k } a _ { k }$ so that $a _ { i ^ { \star } } = a _ { ( 1 ) }$ . For any $j \neq i ^ { \star }$ , we have $a _ { j } \geq a _ { ( 2 ) }$ , hence

$$
\tilde {a} _ {j} - \tilde {a} _ {i ^ {\star}} \geq (a _ {j} - \eta) - (a _ {i ^ {\star}} + \eta) \geq (a _ {(2)} - a _ {(1)}) - 2 \eta > 0,
$$

which implies $\tilde { a } _ { i ^ { \star } } < \tilde { a } _ { j }$ for all $j \neq i ^ { \star }$ .

Lemma 2 (Residual perturbation under subspace (projector) error) Let $S , S ^ { \prime } \subset \mathbb { R } ^ { p }$ be subspaces with orthogonal projectors $P _ { S } , P _ { S ^ { \prime } }$ . Then for any $z \in \mathbb { R } ^ { p }$ ,

$$
\left| \operatorname{dist} (z, S) - \operatorname{dist} (z, S ^ {\prime}) \right| \leq \| (P _ {S} - P _ {S ^ {\prime}}) z \| _ {2} \leq \| P _ {S} - P _ {S ^ {\prime}} \| _ {2} \| z \| _ {2}. \tag {4}
$$

Moreover, $\| P _ { S } - P _ { S ^ { \prime } } \| _ { 2 }$ equals the sine of the largest principal angle between S and $S ^ { \prime }$ (Knyazev and Argentati, 2002).

Proof By triangle inequality,

$$
\| (I - P _ {S}) z \| _ {2} \leq \| (I - P _ {S ^ {\prime}}) z \| _ {2} + \| (P _ {S} - P _ {S ^ {\prime}}) z \| _ {2}.
$$

Swapping $S$ and $S ^ { \prime }$ yields the absolute-value bound. The second inequality is the definition of the operator norm.

Corollary 1 (Operational η-stability under residual and span perturbations) Fix a test point z and suppose the inference procedure uses residuals $\{ \tilde { r } _ { k } ( z ) \} _ { k = 1 } ^ { K }$ satisfying

$$
\sup _ {k} \left| \tilde {r} _ {k} (z) - r _ {k} ^ {\mathrm{sub}} (z) \right| \leq \eta_ {\mathrm{solver}} + \eta_ {\mathrm{subspace}}, \quad \eta_ {\mathrm{subspace}} := \max _ {k} \| P _ {S _ {k}} - P _ {\tilde {S} _ {k}} \| _ {2} \| z \| _ {2},
$$

where $\{ \tilde { S } _ { k } \}$ are perturbed/estimated class spans $( e . g .$ , due to finite-sample effects or dictionary changes). $I f m ^ { \mathrm { s u b } } ( z ) > 2 ( \eta _ { \mathrm { s o l v e r } } + \eta _ { \mathrm { s u b s p a c e } } )$ , then the argmin label under $\{ \tilde { r } _ { k } ( z ) \}$ matches the argmin label under $\{ r _ { k } ^ { \mathrm { s u b } } ( z ) \}$ .

Proof Apply Lemma 1 with $a _ { k } = r _ { k } ^ { \mathrm { s u b } } ( z )$ and $\tilde { a } _ { k } = \tilde { r } _ { k } ( z )$

Why margins (and not only labels). A fixed inference rule returns only a discrete label, but its reliability is governed by the robustness of the residual ordering. Two geometries can agree on the argmin on a finite set yet differ substantially in their margins and thus in sensitivity to perturbations (embedding noise, dictionary thinning, or solver variability). This is why margin diagnostics are load-bearing for reconstruction-based inference.

Solver–geometry separation. A fixed SRC solver (e.g., OMP with sparsity budget s) is an algorithmic mechanism that produces a practical set of residuals which may be viewed as a perturbation of the span-level residual family. Deviations may occur when sparse recovery is inconsistent or dictionaries are poorly conditioned. Our claims characterize stability as a property of the learned geometry (spans and margins), not universal guarantees for a particular sparse solver.

# 3.4 Intrinsic obstructions: when residual inference must be unstable

This subsection records three geometry-only failure modes that directly violate requirement R2 (low leakage) and hence force small margins. The statements are elementary linear-algebraic observations, but they are methodologically essential: they identify what geometry shaping must avoid.

Overlap obstruction (exact intersection). We first record the overlap failure mode: if two empirical class spans share a nontrivial direction, then worst-case residual margins must collapse.

Lemma 3 (Distance to a subspace is 1-Lipschitz) Let $S \subset \mathbb { R } ^ { p }$ be a subspace with orthogonal projector $P _ { S }$ . Then, for all $z , z ^ { \prime } \in \mathbb { R } ^ { p }$ ,

$$
\left| \operatorname{dist} (z, S) - \operatorname{dist} \left(z ^ {\prime}, S\right) \right| \leq \| z - z ^ {\prime} \| _ {2}.
$$

Proof Using dist $( z , S ) = \| ( I - P _ { S } ) z \| _ { 2 }$ and $\| I - P _ { S } \| _ { 2 } = 1$ ,

$$
\operatorname{dist} (z, S) \leq \operatorname{dist} (z ^ {\prime}, S) + \| (I - P _ {S}) (z - z ^ {\prime}) \| _ {2} \leq \operatorname{dist} (z ^ {\prime}, S) + \| z - z ^ {\prime} \| _ {2}.
$$

Swap $z , z ^ { \prime }$ to conclude.

Proposition 1 (Impossibility under span overlap) If there exist classes $y \ne j$ such that $S _ { y } \cap$ $S _ { j } \neq \{ 0 \}$ , then for any $\varepsilon > 0$ there exists a unit vector $z \in S _ { y }$ such that $m ^ { \mathrm { s u b } } ( z ) \leq \varepsilon$ .

Proof Choose any nonzero $u \in S _ { y } \cap S _ { j }$ and set $z = u / \| u \| _ { 2 }$ . Then $z \in S _ { y }$ and also $z \in S _ { j }$ , so $r _ { y } ^ { \mathrm { s u b } } ( z ) = r _ { j } ^ { \mathrm { s u b } } ( z ) = 0$ . Hence $m ^ { \mathrm { s u b } } ( z ) = 0 \leq \varepsilon$ .

Moreover, Lemma 3 implies a quantitative neighborhood version: for any $\delta > 0$ , any $z ^ { \prime }$ with $\| z ^ { \prime } - z \| _ { 2 } \leq \delta$ satisfies $r _ { y } ^ { \mathrm { s u b } } ( z ^ { \prime } ) \leq \delta$ and $r _ { j } ^ { \mathrm { s u b } } ( z ^ { \prime } ) \leq \delta$ , hence $m ^ { \mathrm { { s u b } } } ( z ^ { \prime } ) \leq 2 \delta$ .

Dominance obstruction (span containment). We next record dominance: if one class span contains another, then residual inference is degenerate on the contained span.

Lemma 4 (Degeneracy under span dominance) If for some $y \ne j , S _ { j } \subseteq S _ { y }$ , then for every unit $z \in S _ { j } , r _ { y } ^ { \mathrm { s u b } } ( z ) = r _ { j } ^ { \mathrm { s u b } } ( z ) = 0$ , hence $m ^ { \mathrm { s u b } } ( z ) = 0$ .

Proof If $z \in S _ { j } \subseteq S _ { y }$ , then dist $( z , S _ { j } ) = \mathrm { d i s t } ( z , S _ { y } ) = 0$ by definition.

Near-overlap obstruction (small principal angles). Finally, even without exact overlap, near-alignment quantified by small principal angles yields small worst-case competing residuals.

Let $\theta _ { \operatorname* { m i n } } ( S _ { y } , S _ { j } ) \in [ 0 , \pi / 2 ]$ denote the smallest principal angle between subspaces.

Lemma 5 (Small principal angle ⇒ small competing residual and margin) Let $S _ { y } , S _ { j }$ be two subspaces. Then there exists a unit vector $z \in S _ { y }$ such that

$$
r _ {y} ^ {\mathrm{sub}} (z) = 0, \qquad r _ {j} ^ {\mathrm{sub}} (z) = \sin \theta_ {\mathrm{min}} (S _ {y}, S _ {j}), \qquad m ^ {\mathrm{sub}} (z) \leq \sin \theta_ {\mathrm{min}} (S _ {y}, S _ {j}). \qquad (5)
$$

Proof Let $U \in \mathbb { R } ^ { p \times d _ { y } }$ and $V \in \mathbb { R } ^ { p \times d _ { j } }$ be orthonormal bases for $S _ { y }$ and $S _ { j }$ . By the classical principal-angle characterization, the singular values of $U ^ { \top } V$ are $\{ \cos ( \theta _ { i } ) \}$ , with $\theta _ { 1 } = \theta _ { \operatorname* { m i n } } ( S _ { y } , S _ { j } )$ (Bjorck and Golub, 1973; Knyazev and Argentati, 2002). Let $u \in S _ { y }$ be a corresponding principal vector with $\| u \| _ { 2 } = 1$ achieving $\| V ^ { \top } u \| _ { 2 } = \cos \theta _ { \mathrm { m i n } }$ . Then

$$
r _ {j} ^ {\mathrm{sub}} (u) = \mathrm{dist} (u, S _ {j}) = \| (I - P _ {S _ {j}}) u \| _ {2} = \sqrt {1 - \| V ^ {\top} u \| _ {2} ^ {2}} = \sin \theta_ {\mathrm{min}} (S _ {y}, S _ {j}),
$$

and since $u \in S _ { y }$ we have $r _ { y } ^ { \mathrm { s u b } } ( u ) = 0$ . Thus $m ^ { \mathrm { s u b } } ( u ) \leq r _ { j } ^ { \mathrm { s u b } } ( u )$ .

Taken together, these three obstructions isolate geometric regimes in which a fixed residual rule is intrinsically ambiguous in worst-case directions: shared span directions (overlap), span containment (dominance), and near-alignment (small principal angles) all force the span-level margin to collapse. Accordingly, geometry shaping should (i) suppress cross-class reconstruction pathways that realize leakage across class spans, and (ii) discourage inter-class span alignment to avoid small-angle nearoverlap. In Sections 3.6.1 and 3.6.3 we instantiate these requirements via masked self-expressiveness and Frobenius-based subspace repulsion, while Section 3.6.2 addresses the complementary noncollapse requirement R3.

# 3.5 A sufficient condition: residual margin lower bound under geometric regularity

The preceding results identify geometries for which small margins are unavoidable. We now state a complementary sufficient condition: under explicit regularity assumptions capturing in-class coverage (R1), cross-class separation (R2), and bounded perturbations, the subspace residual margin admits a quantitative lower bound.

Assumptions A1–A4 (geometry-aligned). Fix a test embedding $z \in \mathbb { R } ^ { p }$ with ground-truth class y.

A1 (In-class representability in the empirical span; coverage at test time). There exist $s \in S _ { y }$ and $e \in \mathbb { R } ^ { p }$ such that $z = s + e$ .

A2 (Bounded perturbation / embedding noise). $\| e \| _ { 2 } \leq \varepsilon$ .

A3 (Span separation; exclusion of overlap and near-overlap). For all $j \neq y , \theta _ { \mathrm { m i n } } ( S _ { y } , S _ { j } ) \geq \alpha > 0$

A4 (Non-degenerate signal scale). $\| s \| _ { 2 } \geq \gamma > 0$ .

Lemma 6 (Principal-angle separation implies a distance lower bound) Let $S _ { y } , S _ { j } \subset \mathbb { R } ^ { p }$ be subspaces and let $\theta _ { \operatorname* { m i n } } ( S _ { y } , S _ { j } )$ be their smallest principal angle. Then for any $s \in S _ { y }$ ,

$$
\operatorname{dist} (s, S _ {j}) \geq \| s \| _ {2} \sin \theta_ {\min} (S _ {y}, S _ {j}).
$$

Proof If $s = 0$ the claim is trivial. Otherwise write $s = \| s \| _ { 2 } u$ with $\| u \| _ { 2 } = 1$ and $u \in S _ { y }$ . By definition of $\theta _ { \mathrm { m i n } }$ , for any unit $u \in S _ { y }$ we have $\| P _ { S _ { j } } u \| _ { 2 } \le \cos \theta _ { \operatorname* { m i n } } ( S _ { y } , S _ { j } )$ . Therefore,

$$
\operatorname{dist} (u, S _ {j}) = \| \left(I - P _ {S _ {j}}\right) u \| _ {2} = \sqrt {1 - \| P _ {S _ {j}} u \| _ {2} ^ {2}} \geq \sqrt {1 - \cos^ {2} \theta_ {\min}} = \sin \theta_ {\min}.
$$

Scaling by ∥s∥2 gives the result.

Theorem 1 (Subspace residual margin lower bound) Let $r _ { k } ^ { \mathrm { s u b } } ( z ) = \mathrm { d i s t } ( z , S _ { k } )$ and $m ^ { \mathrm { s u b } } ( z ) =$ $r _ { ( 2 ) } ^ { \mathrm { s u b } } ( z ) - r _ { ( 1 ) } ^ { \mathrm { s u b } } ( z )$ . Under Assumptions $A 1 { - } A 4 ,$ ,

$$
m ^ {\text { sub }} (z) \geq \gamma \sin (\alpha) - 2 \varepsilon . \tag {6}
$$

In particular, $i f \gamma \sin ( \alpha ) > 2 \varepsilon$ , then $m ^ { \mathrm { s u b } } ( z ) > 0$ and the smallest subspace residual is uniquely attained at class y.

Proof By Lemma 3 (applied with $S = S _ { y } )$ and since $s \in S _ { y } .$

$$
r _ {y} ^ {\text { sub }} (z) = \operatorname{dist} (s + e, S _ {y}) \leq \operatorname{dist} (s, S _ {y}) + \| e \| _ {2} = \| e \| _ {2} \leq \varepsilon .
$$

For $j \neq y$ , again by Lemma 3 (applied with $S = S _ { j } )$ ,

$$
r _ {j} ^ {\text { sub }} (z) = \operatorname{dist} (s + e, S _ {j}) \geq \operatorname{dist} (s, S _ {j}) - \| e \| _ {2}.
$$

By Lemma 6 and A3–A4,

$$
\operatorname{dist} (s, S _ {j}) \geq \| s \| _ {2} \sin \theta_ {\min} (S _ {y}, S _ {j}) \geq \gamma \sin (\alpha).
$$

Thus $r _ { j } ^ { \mathrm { s u b } } ( z ) \geq \gamma \sin ( \alpha ) - \varepsilon$ for all $j \neq y$ , and therefore

$$
m ^ {\text { sub }} (z) = r _ {(2)} ^ {\text { sub }} (z) - r _ {(1)} ^ {\text { sub }} (z) \geq (\gamma \sin (\alpha) - \varepsilon) - \varepsilon = \gamma \sin (\alpha) - 2 \varepsilon .
$$

To interpret Theorem 1 in methodological terms, it is useful to read the bound as a direct translation from geometric regularity to residual-ordering stability. Theorem 1 turns the informal requirements into explicit geometric targets: coverage (A1) and non-degeneracy (A4) motivate within-class reconstructability and anti-collapse, while separation (A3) motivates coherence/leakage suppression. The guarantee is inherently conditional: it cannot hold in the overlap or near-overlap regimes above, and it degrades with perturbation scale ε. Guided by these targets, Section 3.6 introduces training-time surrogates that promote coverage, separation, and non-collapse without coupling learning to SRC inference.

# 3.6 Training objective for geometry shaping under fixed inference

Motivated by the obstruction results (Section 3.4) and the sufficient margin condition (Theorem 1), we shape embeddings to satisfy R1–R3 using implementation-independent surrogates, without invoking SRC during training. Each training term corresponds to a geometric requirement and a failure mode: masked self-expressiveness targets R1 while suppressing cross-class coefficients (part of R2); subspace repulsion targets R2 by discouraging near-alignment (Lemma 5); and a non-discriminative anchor targets R3 by preventing collapse. At evaluation time, we therefore report residual margins (stability) and leakage/coherence proxies (cross-class alignment) in addition to accuracy. Furthermore, given a mini-batch $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$ , we form normalized embeddings $z _ { i } = f _ { \theta } ( x _ { i } )$ with $\| z _ { i } \| _ { 2 } = 1$ and stack $Z = [ z _ { 1 } , \dots , z _ { n } ] \in \mathbb { R } ^ { p \times n }$ , and, we define the class mask $M \in \{ 0 , 1 \} ^ { n \times n }$ by $M _ { i j } = 1 [ y _ { i } = y _ { j } ]$ . Next, we describe each term in the training objective and how it operationalizes the geometric requirements R1–R3.

# 3.6.1 Masked ridge self-expressiveness (core)

The core training-time geometry term is a masked self-expressiveness objective. We define the coefficient matrix $C ^ { \star } ( Z ) \in \mathbb { R } ^ { n \times n }$ by

$$
C ^ {\star} (Z) \in \arg \min _ {C \in \mathbb {R} ^ {n \times n}} \| Z - Z C \| _ {F} ^ {2} + \lambda \| C \| _ {F} ^ {2} + \mu \| (1 - M) \odot C \| _ {F} ^ {2} \quad \text { s.t. } \operatorname{diag} (C) = 0. \tag {7}
$$

For $\lambda > 0$ , this constrained ridge problem is well-defined column-wise. The i-th column $c _ { i } ^ { \star }$ reconstructs $z _ { i }$ from the other embeddings in the mini-batch, while the diagonal constraint $c _ { i i } = 0$ prevents the trivial self-copying solution. The three terms in Eq. (7) have distinct roles. The reconstruction term $\| Z - Z C \| _ { F } ^ { 2 }$ encourages self-expressiveness: each embedding should be recoverable from other batch embeddings. The ridge term $\lambda \| C \| _ { F } ^ { 2 }$ stabilizes the coefficient matrix and discourages excessively large reconstruction weights. The masked penalty $\mu \| ( 1 - M ) \odot C \| _ { F } ^ { 2 }$ penalizes coefficients assigned to samples from different classes. Thus, within-class reconstruction pathways are allowed, whereas cross-class reconstruction pathways are discouraged but not hard forbidden.

The self-expressiveness loss is then

$$
L _ {\mathrm{SE}} (Z, y) := \| Z - Z C ^ {\star} (Z) \| _ {F} ^ {2}. \tag {8}
$$

The coefficient matrix $C ^ { \star } ( Z )$ is used only to define a training-time geometric regularizer. It is not the test-time SRC/OMP sparse code, it is not used to form training predictions, and no SRC residuals, residual margins, or SRC-based labels are computed during training. In implementation, the diagonal constraint is enforced as part of the constrained masked-ridge solve.

# 3.6.2 Variance anchor (anti-collapse)

To avoid degenerate collapsed representations, we use a non-discriminative scale-aware variance anchor. Since all embeddings are $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ , the natural coordinate-wise standard-deviation scale is $1 / { \sqrt { p } }$ , up to a constant factor, rather than a fixed value such as 1. For unit-normalized embeddings in $\mathbb { R } ^ { p }$ , the average coordinate energy is approximately $1 / p ,$ so a coordinate-wise standard-deviation floor should be scaled relative to $1 / { \sqrt { p } } .$

We therefore define

$$
L _ {\text { anch }} (Z) = \frac {1}{p} \sum_ {j = 1} ^ {p} \max \left\{0, \frac {c}{\sqrt {p}} - \operatorname{std} (Z _ {j,:}) \right\}, \tag {9}
$$

where $c > 0$ controls the strength of the variance floor. This term is not intended to impose class separation or to optimize SRC residuals. It is a batch-level anti-collapse regularizer that discourages all embeddings from collapsing to a single point by maintaining a minimum amount of coordinate-wise dispersion compatible with unit-normalized embeddings.

# 3.6.3 Inter-class subspace repulsion

To discourage near-alignment of class spans, we penalize coherence between estimated class subspace bases. For each class k represented in the batch, form centered class embeddings $Z _ { k }$ and extract an orthonormal basis $U _ { k } \ \in \ \mathbb { R } ^ { p \times d }$ from the top d principal directions. Let $P = \{ ( k , \ell ) : k <$ $\ell , \ U _ { k } , U _ { \ell }$ defined} and define

$$
\mathcal {L} _ {\text { rep }} (Z, y) := \frac {1}{| P |} \sum_ {(k, \ell) \in P} \| U _ {k} ^ {\top} U _ {\ell} \| _ {F} ^ {2}. \tag {10}
$$

Lemma 7 (Frobenius repulsion upper-bounds worst-case alignment) For any matrix A, $\| A \| _ { 2 } \leq \| A \| _ { F }$ . Consequently, for orthonormal class bases $U _ { k } , U _ { \ell }$ ,

$$
\sigma_ {\mathrm{max}} (U _ {k} ^ {\top} U _ {\ell}) = \| U _ {k} ^ {\top} U _ {\ell} \| _ {2} \leq \| U _ {k} ^ {\top} U _ {\ell} \| _ {F},
$$

and since $\sigma _ { \mathrm { m a x } } ( U _ { k } ^ { \top } U _ { \ell } ) = \cos ( \theta _ { \mathrm { m i n } } ( S _ { k } , S _ { \ell } ) )$ , the repulsion penalty controls (an upper bound on) the worst-case alignment mode tied to the smallest principal angle.

Proof Let A have singular values $\{ \sigma _ { i } \}$ . Then $\begin{array} { r } { \| A \| _ { 2 } = \operatorname* { m a x } _ { i } \sigma _ { i } \leq ( \sum _ { i } \sigma _ { i } ^ { 2 } ) ^ { 1 / 2 } = \| A \| _ { F } . } \end{array}$

Training objective and freezing: We train $f _ { \theta }$ using only geometry-shaping objectives:

$$
\min _ {\theta} \mathbb {E} _ {\text {batches}} \left[ \lambda_ {\mathrm{SE}} \mathcal {L} _ {\mathrm{SE}} (Z, y) + \beta_ {\text {anch}} \mathcal {L} _ {\text {anch}} (Z) + \lambda_ {\text {rep}} \mathcal {L} _ {\text {rep}} (Z, y) \right], \quad z _ {i} = \text {normalize} (f _ {\theta} (x _ {i})). \tag {11}
$$

No classifier head, cross-entropy, SRC residual term, or any accuracy-driven loss appears during training. After training, $f _ { \theta }$ is frozen.

# 3.7 Additional implications: leakage suppression and coverage limits

We next record two complementary implications that connect the training surrogates to the spanlevel stability framework while clarifying scope. First, the masked self-expressiveness objective suppresses cross-class reconstruction pathways at the batch level, linking the training signal to R2 without invoking SRC during training.

Proposition 2 (Approximate block self-expressiveness from masking) Let $Z = [ Z _ { 1 } , \dots , Z _ { K } ]$ be a batch embedding matrix grouped by class, M the corresponding mask, and $l e t C ^ { \star } ( Z )$ be the constrained masked-ridge solution defined in Eq. (7). $I f \| ( 1 { - } M ) \odot C ^ { * } ( Z ) \| _ { F }$ is small and $\| Z - Z C ^ { * } ( Z ) \| _ { F }$ is small, then the cross-class contribution to reconstructed embeddings is small:

$$
\| Z ((1 - M) \odot C ^ {*} (Z)) \| _ {F} \leq \| Z \| _ {2} \| (1 - M) \odot C ^ {*} (Z) \| _ {F}. \tag {12}
$$

Proof Apply $\| A B \| _ { F } \leq \| A \| _ { 2 } \| B \| _ { F }$ with $A = Z$ and $B = ( 1 - M ) \odot C ^ { * } ( Z )$ .

Second, residual-based inference is inherently conditional on empirical coverage: if a test embedding lies outside the in-class empirical span, then even the idealized in-class residual cannot vanish, irrespective of inter-class separation.

Proposition 3 (Coverage limitation: insufficiency of the empirical class span) Fix a class y. If a test embedding z lies outside the empirical span span $( Z _ { y } )$ , then

$$
r _ {y} ^ {\text { sub }} (z) = \text { dist } (z, \text { span } (Z _ {y})) > 0. \tag {13}
$$

Hence any guarantee requiring vanishing or uniformly small in-class residuals fails for such z, regardless of inter-class separation.

Proof Distance to a subspace is strictly positive for points outside it.

Together, these observations (i) justify masking as an explicit leakage-control mechanism during training and (ii) motivate stating coverage explicitly whenever residual-margin stability is invoked for fixed residual-based inference. We now make the training–inference separation operational by stating the two procedures explicitly in Section 3.8.

# 3.8 Algorithms (operational separation)

We now present the two procedures explicitly in order to make the training–inference separation operational and reproducible. Algorithm 1 specifies representation learning using only geometryshaping losses computed on mini-batches of embeddings. In particular, training never computes SRC/OMP sparse codes, class-wise residuals, residual margins, or predictions, and it does not use any accuracy-driven loss. Algorithm 2 specifies test-time inference on frozen embeddings via a fixed SRC rule: a global sparse code is computed once over the full training dictionary (implemented by OMP with a fixed sparsity budget), class-restricted reconstructions are formed, and prediction is obtained by the minimum class-wise residual. All empirical comparisons in Section 4 follow this protocol: the learned component is the encoder, while the inference routine is kept identical across representations.

Algorithm 1 Geometry-Shaping Representation Learning (Training)   
Require: training data $\{(x_i, y_i)\}_{i=1}^N$ ; encoder $f_\theta$ ; $\lambda_{\text{SE}}, \beta_{\text{anch}}, \lambda_{\text{rep}}, \lambda, \mu, d, c$ ; optimizer  
Ensure: trained encoder $f_\theta$

1: Initialize parameters θ   
2: repeat   
3: Sample a mini-batch $\{ ( x _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n }$   
4: Compute embeddings $z _ { i } = f _ { \theta } ( x _ { i } )$ and normalize $z _ { i } \gets z _ { i } / \| z _ { i } \| _ { 2 }$   
5: Form $Z = [ z _ { 1 } , \dots , z _ { n } ] \in \mathbb { R } ^ { p \times n }$   
6: Construct class mask $M \in \{ 0 , 1 \} ^ { n \times n }$ with $M _ { i j } = 1 [ y _ { i } = y _ { j } ]$   
7: Compute masked ridge self-expressive coefficients:

$$
C ^ {*} (Z) \in \underset {C \in \mathbb {R} ^ {n \times n}} {\arg \min} \| Z - Z C \| _ {F} ^ {2} + \lambda \| C \| _ {F} ^ {2} + \mu \| (1 - M) \odot C \| _ {F} ^ {2} \quad \text {s.t.} \quad \operatorname{diag} (C) = 0.
$$

8: Compute losses:

$$
\mathcal {L} _ {\mathrm{SE}} = \| Z - Z C ^ {*} (Z) \| _ {F} ^ {2}, \quad \mathcal {L} _ {\text {anch}} = \frac {1}{p} \sum_ {j = 1} ^ {p} \max \left\{0, \frac {c}{\sqrt {p}} - \operatorname{std} \left(Z _ {j,:}\right) \right\}
$$

9: For each class k present in the batch, center $Z _ { k } = \{ z _ { i } : \ y _ { i } = k \}$ and extract basis $U _ { k } \in \mathbb { R } ^ { p \times d }$ from top d principal directions

10: Let $P = \{ ( k , \ell ) : k < \ell , \ U _ { k } , U _ { \ell }$ defined} and compute

$$
\mathcal {L} _ {\mathrm{rep}} = \frac {1}{| P |} \sum_ {(k, \ell) \in P} \| U _ {k} ^ {\top} U _ {\ell} \| _ {F} ^ {2}
$$

11: Form total loss $\mathcal { L } = \lambda _ { \mathrm { S E } } \mathcal { L } _ { \mathrm { S E } } + \beta _ { \mathrm { a n c h } } \mathcal { L } _ { \mathrm { a n c h } } + \lambda _ { \mathrm { r e p } } \mathcal { L } _ { \mathrm { r e p } }$   
12: Update θ by gradient-based optimization of $\mathcal { L }$   
13: until convergence   
14: return trained encoder $f _ { \theta }$

Algorithm 2 Fixed SRC Inference on Frozen Embeddings (Test Time)   
Require: frozen encoder $f_{\theta}$ ; training dictionary $Z_{\mathrm{tr}} = [z_1, \ldots, z_N]$ with labels; test point $x$ ; sparsity level $s$   
Ensure: predicted label $\hat { y } ;$ residual margin m

1: Compute embedding $z = f _ { \boldsymbol { \theta } } ( x )$ and normalize $z  z / \| z \| _ { 2 }$   
2: Solve global sparse coding (implemented by OMP with budget $s )$

$$
\hat {c} \in \underset {c \in \mathbb {R} ^ {N}} {\arg \min} \| z - Z _ {\mathrm{tr}} c \| _ {2} ^ {2} \quad \text {s.t.} \quad \| c \| _ {0} \leq s
$$

3: for $k = 1 , \ldots , K$ do   
4: Compute class-wise residual $r _ { k } ( z ) = \| z - Z _ { \mathrm { t r } } \delta _ { k } ( \hat { c } ) \| _ { 2 }$   
5: end for   
6: Output $\hat { y } = \arg \operatorname* { m i n } _ { k } r _ { k } ( z )$ and margin $m = r _ { ( 2 ) } ( z ) - r _ { ( 1 ) } ( z )$

# 3.9 Solver-level interpretation (conditional stability-transfer)

From span-level margins to a pursuit routine. Our main stability statement is Theorem 1, which is formulated at the span/subspace level under explicit geometric assumptions $\left( \mathrm { A 1 - A 4 } \right)$ . In practice, test-time residuals are obtained by a pursuit routine (here OMP) applied to a finite dictionary. The goal of this section is therefore not to provide an end-to-end guarantee for a particular solver, but to give a sufficient-condition interpretive link showing how training-time geometric regularization can translate into span-level separation, and how span-level stability can be related to practical residual behavior under controlled solver-side discrepancy.

Notation. For each class k, let $S _ { k } \subset \mathbb { R } ^ { p }$ denote the empirical class span with orthogonal projector $P _ { S _ { k } }$ and orthonormal basis $U _ { k } \in \mathbb { R } ^ { p \times d _ { k } }$ . Principal angles between $S _ { k }$ and $S _ { \ell }$ are denoted $\theta _ { 1 } \leq \cdots \leq \theta _ { r }$ , where $r = \operatorname* { m i n } ( d _ { k } , d _ { \ell } )$ and

$$
\cos \theta_ {i} = \sigma_ {i} (U _ {k} ^ {\top} U _ {\ell}).
$$

We first connect the Frobenius repulsion penalty used at training time to principal-angle separation and the corresponding span-level residual-margin bound.

Recall the repulsion penalty

$$
\mathcal {L} _ {\mathrm{rep}} (k, \ell) = \| U _ {k} ^ {\top} U _ {\ell} \| _ {F} ^ {2} = \sum_ {i = 1} ^ {r} \cos^ {2} \theta_ {i} (S _ {k}, S _ {\ell}).
$$

Lemma 8 (Frobenius-to-spectral control) For any two subspaces $S _ { k } , S _ { \ell }$

$$
\cos^ {2} \theta_ {\mathrm{min}} (S _ {k}, S _ {\ell}) = \| U _ {k} ^ {\top} U _ {\ell} \| _ {2} ^ {2} \leq \| U _ {k} ^ {\top} U _ {\ell} \| _ {F} ^ {2}.
$$

Proof The singular values of $U _ { k } ^ { \top } U _ { \ell }$ are $\{ \cos \theta _ { i } \}$ . The spectral norm equals the largest singular value, while the Frobenius norm equals the sum of squared singular values. Since maxi $\begin{array} { r } { a _ { i } ^ { 2 } \leq \sum _ { i } a _ { i } ^ { 2 } } \end{array}$ , the claim follows.

Corollary 2 (Repulsion implies principal-angle separation) Suppose that for all k $\neq \ell ,$ ,

$$
\| U _ {k} ^ {\top} U _ {\ell} \| _ {F} ^ {2} \leq \eta .
$$

Then

$$
\theta_ {\mathrm{min}} (S _ {k}, S _ {\ell}) \geq \arccos (\sqrt {\eta}).
$$

Proof By Lemma 8,

$$
\cos^ {2} \theta_ {\mathrm{min}} (S _ {k}, S _ {\ell}) \leq \eta .
$$

Taking square roots and applying arccos yields the stated lower bound.

Corollary 3 (Repulsion implies subspace margin lower bound) Fix a test embedding $z =$ $s + e$ with ground-truth class y, where

$$
s \in S _ {y}, \qquad \| s \| _ {2} \geq \gamma > 0, \qquad \| e \| _ {2} \leq \varepsilon .
$$

Assume that for all $j \neq y$

$$
\| U _ {y} ^ {\top} U _ {j} \| _ {F} ^ {2} \leq \eta .
$$

Then the subspace residual margin satisfies

$$
m ^ {\mathrm{sub}} (z) \geq \gamma \sqrt {1 - \eta} - 2 \varepsilon .
$$

Proof [Proof sketch] By Corollary 2, $\theta _ { \operatorname* { m i n } } ( S _ { y } , S _ { j } ) \geq \operatorname { a r c c o s } ( \sqrt { \eta } )$ , so

$$
\sin \theta_ {\mathrm{min}} (S _ {y}, S _ {j}) \geq \sqrt {1 - \eta}.
$$

Hence for $j \neq y$

$$
\mathrm{dist} (s, S _ {j}) \geq \| s \| _ {2} \sin \theta_ {\mathrm{min}} \geq \gamma \sqrt {1 - \eta}.
$$

Using

$$
r _ {y} ^ {\mathrm{sub}} (z) \leq \varepsilon , \qquad r _ {j} ^ {\mathrm{sub}} (z) \geq \gamma \sqrt {1 - \eta} - \varepsilon ,
$$

the margin bound follows.

# 3.9.1 A Solver-Level Interpretation via Global Reconstruction Error and Cross-Class Leakage

The results above are stated at the span level: they characterize when the classwise subspace residuals

$$
r _ {k} ^ {s u b} (z) = \| z - P _ {S _ {k}} z \| _ {2}
$$

are stably ordered in favor of the correct class. This remains the primary theoretical level of the paper. At test time, however, inference is carried out on a finite training dictionary using the fixed SRC/OMP protocol of Algorithm 2: OMP is run on the full dictionary to produce a sparse code, and classwise residuals are then formed by restricting that global code to class-specific indices.

As discussed throughout, the purpose of this subsection is not to provide a standalone sparserecovery theorem for global OMP with class restriction. Rather, the goal is to make the solver-side transfer statement more concrete by expressing it in terms of quantities that are intrinsic to the actual inference pipeline. In particular, instead of postulating an abstract residual-discrepancy term over the entire classwise residual family, we isolate two directly interpretable quantities: (i) the global reconstruction error of the OMP code, and (ii) the amount of cross-class leakage carried by that code.

Let $z \in \mathbb { R } ^ { p }$ be a test embedding of ground-truth class y, and let

$$
\hat {c} (z) \in \mathbb {R} ^ {N}
$$

denote the global sparse code returned by OMP in Algorithm 2. Partition the dictionary and the coefficient vector according to the true class:

$$
Z = [ Z _ {y} Z _ {- y} ], \qquad \hat {c} (z) = \left[ \begin{array}{c} \hat {c} _ {y} (z) \\ \hat {c} _ {- y} (z) \end{array} \right].
$$

The practical SRC residuals are

$$
\tilde {r} _ {k} (z) = \| z - Z \delta_ {k} (\hat {c} (z)) \| _ {2}.
$$

We also define the global OMP reconstruction error

$$
\eta_ {\mathrm{rec}} (z) := \| z - Z \hat {c} (z) \| _ {2}
$$

and the cross-class leakage magnitude

$$
\eta_ {\mathrm{leak}} (z) := \| Z _ {- y} \hat {c} _ {- y} (z) \| _ {2}.
$$

We first record two elementary facts.

One-sided domination. For every class k,

$$
r _ {k} ^ {s u b} (z) \leq \tilde {r} _ {k} (z).
$$

Indeed, $Z \delta _ { k } ( \hat { c } ( z ) ) \in S _ { k } = \operatorname { s p a n } ( Z _ { k } )$ , whereas $r _ { k } ^ { s u b } ( z )$ is the minimum distance from z to $S _ { k }$

True-class residual bound from global error and leakage. For the true class y,

$$
\tilde {r} _ {y} (z) = \| z - Z _ {y} \hat {c} _ {y} (z) \| _ {2} = \| (z - Z \hat {c} (z)) + Z _ {- y} \hat {c} _ {- y} (z) \| _ {2} \leq \eta_ {\mathrm{rec}} (z) + \eta_ {\mathrm{leak}} (z).
$$

The next theorem turns these identities into an explicit margin-transfer statement for the actual global-OMP-plus-class-restriction protocol.

Theorem 2 (Leakage-aware transfer from span-level margin to practical SRC/OMP margin) Let

$$
m ^ {s u b} (z) := \min _ {j \neq y} \bigl (r _ {j} ^ {s u b} (z) - r _ {y} ^ {s u b} (z) \bigr)
$$

denote the span-level residual margin for a sample z of ground-truth class y, and let

$$
m ^ {p r} (z) := \min _ {j \neq y} \bigl (\tilde {r} _ {j} (z) - \tilde {r} _ {y} (z) \bigr)
$$

denote the practical residual margin induced by Algorithm 2.

Assume that

$$
m ^ {s u b} (z) \geq \Delta > 0.
$$

Then the practical residual margin satisfies

$$
m ^ {p r} (z) \geq \min _ {j \neq y} r _ {j} ^ {s u b} (z) - (\eta_ {\mathrm{rec}} (z) + \eta_ {\mathrm{leak}} (z)) = \Delta + r _ {y} ^ {s u b} (z) - \eta_ {\mathrm{rec}} (z) - \eta_ {\mathrm{leak}} (z).
$$

In particular, if

$$
\eta_ {\mathrm{rec}} (z) + \eta_ {\mathrm{leak}} (z) <   \min _ {j \neq y} r _ {j} ^ {s u b} (z),
$$

equivalently if

$$
\Delta > \eta_ {\mathrm{rec}} (z) + \eta_ {\mathrm{leak}} (z) - r _ {y} ^ {s u b} (z),
$$

then the practical residual ordering agrees with the span-level ordering, and the correct class remains the unique minimizer of the practical residual family.

Proof Fix any j ̸= y. By one-sided domination,

$$
\tilde {r} _ {j} (z) \geq r _ {j} ^ {s u b} (z).
$$

Also, by the true-class residual bound,

$$
\tilde {r} _ {y} (z) \leq \eta_ {\mathrm{rec}} (z) + \eta_ {\mathrm{leak}} (z).
$$

Therefore,

$$
\tilde {r} _ {j} (z) - \tilde {r} _ {y} (z) \geq r _ {j} ^ {s u b} (z) - \big (\eta_ {\mathrm{rec}} (z) + \eta_ {\mathrm{leak}} (z) \big).
$$

Taking the minimum over all $j \neq y$ yields

$$
m ^ {p r} (z) \geq \min _ {j \neq y} r _ {j} ^ {s u b} (z) - \big (\eta_ {\mathrm{rec}} (z) + \eta_ {\mathrm{leak}} (z) \big).
$$

Using

$$
\min _ {j \neq y} r _ {j} ^ {s u b} (z) = m ^ {s u b} (z) + r _ {y} ^ {s u b} (z) \geq \Delta + r _ {y} ^ {s u b} (z),
$$

we obtain

$$
m ^ {p r} (z) \geq \Delta + r _ {y} ^ {s u b} (z) - \eta_ {\mathrm{rec}} (z) - \eta_ {\mathrm{leak}} (z).
$$

If

$$
\eta_ {\mathrm{rec}} (z) + \eta_ {\mathrm{leak}} (z) <   \min _ {j \neq y} r _ {j} ^ {s u b} (z),
$$

then $\tilde { r } _ { j } ( z ) > \tilde { r } _ { y } ( z )$ for all $j \neq y$ , so the correct class remains the unique minimizer of the practical residual family. ■

The theorem should be read as an explicit stability-transfer statement for the actual fixed SRC/OMP protocol. Its content is more concrete than a generic solver discrepancy term: the degradation from span-level margin to practical margin is controlled by $\eta _ { \mathrm { r e c } } ( z )$ , which measures how well the global OMP code reconstructs the test embedding, and by $\eta _ { \mathrm { l e a k } } ( z )$ , which measures how much of that reconstruction is carried by atoms outside the true class. This makes the role of leakage precise. Wrong-class practical residuals are not dangerous because they are automatically lower bounded by their span-level counterparts. What can destroy the span-level ordering is that the global sparse code reconstructs z well only by using substantial mass on wrong-class atoms, thereby keeping the true-class practical residual $\tilde { r } _ { y } ( z )$ artificially large. In this sense, the solver-side issue is not uniform closeness of the entire practical residual family to the span-level family, but whether global pursuit achieves low reconstruction error with low cross-class leakage.

The theorem above isolates the part of the discrepancy that is intrinsic to the actual global OMP pipeline. One may further refine the interpretation by introducing the best class-y, s-sparse residual

$$
\rho_{y,s}(z):= \min_{\substack{\operatorname{supp}(c)\subseteq I_{y}\\ \| c\|_{0}\leq s}}\| z - Zc\|_{2},
$$

where $I _ { y }$ is the index set of atoms belonging to class y. Then

$$
r _ {y} ^ {s u b} (z) \leq \rho_ {y, s} (z) \leq \tilde {r} _ {y} (z),
$$

so the true-class practical residual may still be viewed as containing two conceptually distinct effects: (i) a sparse coverage gap $\rho _ { y , s } ( z ) - r _ { y } ^ { s u b } ( z )$ , and (ii) a protocol gap $\tilde { r } _ { y } ( z ) - \rho _ { y , s } ( z )$ . The leakage-aware bound above complements this decomposition by tying the protocol gap to explicit quantities produced by the actual global pursuit. This perspective is consistent with the overall methodological stance of the paper. The training objective shapes representation geometry so as to enlarge and stabilize span-level residual margins, while test-time inference remains fixed and reconstruction based. The solver-level connection therefore remains interpretive: the main theoretical contribution is still the span-level geometry, and practical stability follows insofar as the fixed global pursuit achieves low reconstruction error with limited cross-class leakage.

# 4 Experiments

Goal and evaluation principle. Our experiments evaluate whether geometry-shaped representation learning can improve the observed behavior of a fixed reconstruction-based inference rule. Throughout, training and inference are operationally separated: the encoder is trained only via geometric regularizers, while test-time prediction is performed exclusively by sparse reconstruction (SRC) using class-wise residuals. Accordingly, we emphasize diagnostics that reflect the stability of the residual ordering (margins) and the underlying embedding geometry (span utilization and cross-class alignment/overlap), in addition to classification performance. These results are intended as empirical support for the geometry-shaping hypothesis under fixed residual-based inference, not as a solver-theoretic validation of the global pursuit pipeline.

Reference representations (CE as reference, same inference rule). To contextualize the geometry-shaped family, we include a cross-entropy (CE) reference representation. The CE protocol is: (i) train an encoder with a linear classification head using CE on the training set; (ii) discard the head; (iii) extract frozen $\ell _ { 2 } { \mathrm { - n o r m a l i z e d } }$ embeddings and build the same class-partitioned dictionary; and (iv) evaluate only with the same fixed SRC/OMP inference rule and the same diagnostics. Thus, CE is not a target to “beat”: it provides a reference geometry under an identical residual-based evaluation rule.

Response-surface protocol. Geometry shaping is controlled by two hyperparameters $( \mu , \lambda )$ , which we sweep on a predefined grid to characterize the response surface of fixed SRC inference across regimes. We do not treat this sweep as test-time model selection. When we report a “peak” configuration, it is only a descriptive argmax used as a reference point for visualization; our conclusions are based on surface-level trends and sweep-level summaries (e.g., grid averages) rather than on any single selected setting. Unless otherwise stated, the variance-anchor scale is fixed to $c = 0 . 2 5$ in all experiments.

Fixed SRC inference (OMP-s). All reported predictions are produced by the fixed SRC inference procedure in Alg. 2: given a test embedding z and the training dictionary $Z = [ Z _ { 1 } \cdots Z _ { C } ]$ , we compute a sparse code via OMP with sparsity s and predict by the minimum class-wise reconstruction residual. In particular, for the inferred coefficients cˆ we form class-restricted reconstructions using $\delta _ { k } ( \hat { c } )$ and residuals $r _ { k } ( z ) = \| z - Z \delta _ { k } ( \hat { c } ) \| _ { 2 }$ , and assign $\hat { y } ( z ) = \arg \operatorname* { m i n } _ { k } r _ { k } ( z )$ .

# 4.1 Evaluation metrics

Predictive performance. We report standard accuracy, and balanced accuracy when class imbalance is present (EEG). For a dataset with class set $\{ 1 , \ldots , C \}$ and per-class recall $\mathrm { T P R } _ { k }$ balanced accuracy is

$$
\text { BalAcc } = \frac {1}{C} \sum_ {k = 1} ^ {C} \mathrm{TPR} _ {k}. \tag {14}
$$

Residual margins (stability of residual ordering). Let $r _ { ( 1 ) } ( z ) \leq r _ { ( 2 ) } ( z ) \leq \cdots \leq r _ { ( C ) } ( z )$ b e the sorted residuals for test point z. We use the following margin-based diagnostic:

$$
m _ {\mathrm{top2}} (z) = r _ {(2)} (z) - r _ {(1)} (z). \tag {15}
$$

Large positive margins indicate a well-separated residual ordering and, by the perturbation logic of Section 3.3, increased stability of the SRC decision under bounded residual perturbations (e.g., induced by embedding noise, finite-sample dictionary variation, or solver-side approximation variability).

Entropy effective rank (span utilization). To summarize how concentrated the class-conditional embedding spectra are, we compute the entropy effective rank per class (Roy and Vetterli, 2007). Let $Z _ { k } \in \mathbb { R } ^ { n _ { k } \times p }$ be the matrix of embeddings from class k after per-class centering (subtracting the class mean). Let $s ( Z _ { k } ) = ( s _ { 1 } , \ldots , s _ { r _ { k } } )$ denote its nonzero singular values. Define normalized weights

$$
p _ {i} = \frac {s _ {i}}{\sum_ {j} s _ {j}}, \quad i = 1, \dots , r _ {k}, \tag {16}
$$

and Shannon entropy $\begin{array} { r } { H ( p ) = - \sum _ { i } p _ { i } \log p _ { i } } \end{array}$ . The entropy effective rank is

$$
\operatorname{EffRank} (Z _ {k}) = \exp (H (p)). \tag {17}
$$

We report the mean over classes: EffRank $\textstyle = { \frac { 1 } { C } } \sum _ { k = 1 } ^ { C }$ EffRank $( Z _ { k } )$ . Higher values indicate more diffuse energy across directions, while lower values indicate concentration into fewer dominant directions.

Worst-case inter-class alignment. To quantify near-overlap between class subspaces, we report a worst-case alignment diagnostic based on the smallest principal angle. For each class $k ,$ let $U _ { k } \in \mathbb { R } ^ { p \times d }$ contain the top-d left singular vectors of the centered class matrix $Z _ { k }$ (with a fixed d across classes). For a pair of classes $( k , \ell )$ , define the max-correlation coherence

$$
\operatorname{coh} _ {\max} (k, \ell) = \sigma_ {\max} (U _ {k} ^ {\top} U _ {\ell}) = \| U _ {k} ^ {\top} U _ {\ell} \| _ {2} = \cos (\theta_ {\min} (S _ {k}, S _ {\ell})), \tag {18}
$$

where $\theta _ { \mathrm { m i n } } ( S _ { k } , S _ { \ell } )$ is the smallest principal angle between the subspaces $S _ { k } \ = \ \operatorname { s p a n } ( U _ { k } )$ and $S _ { \ell } = \operatorname { s p a n } ( U _ { \ell } )$ . We report the average over distinct class pairs,

$$
\text { Cohesion } _ {\max} = \frac {2}{C (C - 1)} \sum_ {1 \leq k <   \ell \leq C} \operatorname{coh} _ {\max} (k, \ell). \tag {19}
$$

Lower values indicate larger minimal principal angles (less near-overlap), which is favorable for residual-based inference. This is a worst-case diagnostic: it captures the most aligned direction between class subspaces and need not vary monotonically with overlap penalties based on Frobenius norms.

Aggregation. Unless stated otherwise, we aggregate metrics by averaging over random seeds (and over phases when the protocol is phase-conditioned), and report sweep-level summaries (grid averages and descriptive peak locations) to characterize regime-level behavior. All geometry diagnostics (effective rank and $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } )$ are computed with the same procedure and hyperparameters for all representations (pretrained/raw, geometry-shaped, and CE reference).

# 4.2 Image Experiments: COIL-100

Dataset and task. We evaluate our image pipeline on COIL-100 (Nene et al., 1996), a controlled object-recognition benchmark consisting of 100 object categories captured under a turntable setup. Each object is photographed from multiple viewpoints along a full rotation, producing a structured multi-view dataset where appearance changes smoothly with the viewing angle. We treat each image as a labeled instance for object classification. This setting is particularly useful for probing representation geometry: since intra-class variation is largely driven by viewpoint, the learned embedding is expected to organize samples of the same object along coherent low-dimensional manifolds, while preserving separability across objects.

Setup and protocol. Our evaluation compares three representation regimes under the same fixed SRC/OMP inference rule: (i) a frozen ImageNet-pretrained encoder (He et al., 2016; Deng et al., 2009) as a strong off-the-shelf reference, (ii) a CE-shaped reference representation, and (iii) a geometry-shaped representation trained under the proposed geometric regularization controlled by $( \mu , \lambda )$ . We run a full sweep over $\mu \in \{ 0 . 1 , 1 , 1 0 , 1 0 0 \}$ and $\lambda \in \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 \}$ , repeating across phases $p \in \{ 0 , 1 , 2 , 3 \}$ and seeds $\{ 0 , 1 \}$ for a total of $2 \times 4 \times 4 \times 3 = 9 6$ runs. Results in the sweep tables are aggregated across seeds and phases per configuration to characterize the response surface of fixed SRC inference under geometry shaping, while the frozen ImageNet baseline is evaluated under the same metric pipeline as a reference point.

Training settings (COIL-100). For all COIL-100 runs we use a ResNet-18 backbone (He et al., 2016) with a projection head that outputs ℓ2-normalized embeddings of dimension $p = 1 2 8 ;$ inputs are resized to $1 2 8 \times 1 2 8$ and ImageNet-normalized. Models are trained for 40 epochs with AdamW (Kingma and Ba, 2017; Loshchilov and Hutter, 2019) (learning rate $1 0 ^ { - 4 }$ , weight decay $1 0 ^ { - 4 } )$ using batch size 640, and this optimization setup is kept fixed across all runs; in the geometry-shaped condition, only the masked-ridge hyperparameters $( \mu , \lambda )$ are varied while the remaining loss weights are held constant, whereas the CE-shaped reference is trained with cross-entropy under the same optimizer settings (dropping the linear head at evaluation).

Inference and evaluation. At inference time, we evaluate each representation with a sparse representation-based classifier (SRC): given a test sample, we compute its sparse reconstruction over a dictionary of training features (OMP-s) and assign the label based on class-wise reconstruction residuals. We report SRC accuracy and the residual-based margin diagnostic, alongside geometry descriptors of the representation space (effective rank and worst-case inter-class alignment $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } )$ . These diagnostics are used to interpret residual behavior under fixed inference; they are not intended as solver-level guarantees.

# 4.2.1 COIL-100: Geometry sweep evaluation (q=4, phases 0–3)

We evaluate the geometry-shaped representation on COIL-100 under a sweep of $( \mu , \lambda )$ and compare against frozen ResNet embeddings pre-trained on ImageNet and CE-shaped reference embeddings. All representations are evaluated with identical fixed $\mathrm { S R C / O M P }$ inference.

Table 5 shows that COIL-100 remains an accuracy-ceiling regime under fixed SRC/OMP inference. Across the geometry sweep, SRC accuracy ranges only from 0.999 to 1.000, with several configurations attaining perfect accuracy. Consequently, raw accuracy is not sufficiently discriminative for interpreting the effect of geometry shaping in this experiment. We therefore interpret COIL-100 primarily through residual-ordering stability and geometric diagnostics: Top-2 residual margin, effective rank, and worst-case inter-class alignment, measured by $\mathrm { C o h e s i o n } _ { \operatorname* { m a x } }$ .

Because several $( \mu , \lambda )$ cells attain the same maximum SRC accuracy, the representative geometryshaped cell in Table 1 is not selected by accuracy alone. We use an accuracy-constrained stability rule: among all cells with maximum SRC accuracy, we select the cell with the largest Top-2 residual margin. Lower $\mathrm { C o h e s i o n } _ { \operatorname* { m a x } }$ and higher effective rank are used only as secondary tie-breakers if needed. Under this rule, the representative geometry-shaped cell is $( \mu , \lambda ) = ( 1 . 0 , 0 . 0 1 )$ , which achieves perfect SRC accuracy, Top-2 margin 0.947, effective rank 27.754, and $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 1 5 6$

Table 1 compares the frozen ImageNet baseline, the CE-shaped reference, and geometry-shaped embeddings. The frozen ImageNet baseline already achieves perfect SRC accuracy, with Top-2 margin 0.785, effective rank 31.862, and $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 1 7 0$ . The selected geometry-shaped cell improves substantially over the frozen baseline in residual separation (0.947 versus 0.785) and slightly reduces worst-case inter-class alignment (0.156 versus 0.170), while maintaining perfect accuracy. The sweep-averaged geometry-shaped representation also remains highly accurate, with mean SRC accuracy approximately 1.000, Top-2 margin 0.893, effective rank 28.058, and $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 1 9 4$ The CE-shaped reference is particularly strong on COIL-100. It achieves perfect SRC accuracy, Top-2 margin 0.9308, effective rank 35.81, and the lowest summary-level $\mathrm { C o h e s i o n } _ { \operatorname* { m a x } }$ value (0.08). Thus, COIL-100 should be viewed as a near-ceiling, CE-strong regime: geometry shaping can substantially improve residual separation relative to frozen ImageNet features, but it does not uniformly dominate CE-shaped embeddings on all geometric diagnostics.

Table 4 shows that the Top-2 residual margin depends strongly on $\mu .$ The weakest masking regime, $\mu = 0 . 1$ , yields noticeably lower margins, approximately 0.743–0.748, despite perfect SRC accuracy. In contrast, all configurations with $\mu \geq 1$ yield substantially larger margins, ranging from

Table 1: COIL-100 summary comparison between frozen ImageNet baseline, CE-shaped reference embeddings, and geometry-shaped representations. The “acc-tied, margin-selected” row denotes the cell selected by first maximizing SRC accuracy and then, among accuracy ties, maximizing the Top-2 residual margin. The sweep average reports the mean over all $( \mu , \lambda )$ configurations. 

<table><tr><td>Representation</td><td>SRC Acc</td><td>Top-2 Margin</td><td>Eff. Rank</td><td>Cohesionmax</td></tr><tr><td>ImageNet (frozen)</td><td>1.000</td><td>0.785</td><td>31.862</td><td>0.170</td></tr><tr><td>Geometry-shaped (acc-tied, margin-selected)</td><td>1.000</td><td>0.947</td><td>27.754</td><td>0.156</td></tr><tr><td>Geometry-shaped ( $\mu$ ,  $\lambda$ -avg)</td><td>1.000</td><td>0.893</td><td>28.058</td><td>0.194</td></tr><tr><td>CE-shaped (reference)</td><td>1.000</td><td>0.931</td><td>35.810</td><td>0.080</td></tr></table>

0.932 to 0.953. The largest observed margin occurs at $( \mu , \lambda ) = ( 1 0 . 0 , 0 . 1 )$ , where $m _ { \mathrm { t o p 2 } } = 0 . 9 5 3$ , closely followed by $( \mu , \lambda ) = ( 1 . 0 , 0 . 1 )$ with $m _ { \mathrm { t o p 2 } } = 0 . 9 5 2$ and $( \mu , \lambda ) = ( 1 . 0 , 0 . 0 1 )$ with $m _ { \mathrm { t o p 2 } } = 0 . 9 4 7$ . This indicates that increasing the off-class masking pressure beyond the weakest setting improves residual-ordering stability, even though classification accuracy is already saturated.

Table 2 shows that effective rank generally increases in the larger-µ region. At $\mu = 0 . 1$ , effective rank lies around 26.3–27.4, whereas at $\mu = 1 0 0$ it rises from 28.110 to 32.793 as λ increases. The maximum effective rank occurs at $( \mu , \lambda ) = ( 1 0 0 . 0 , 0 . 1 )$ , where it reaches 32.793. This slightly exceeds the frozen ImageNet baseline effective rank (31.862), although it remains below the CE-shaped reference (35.81). Thus, within the geometry-shaped family, large $\mu$ encourages broader span utilization, but CE training still yields the broadest representation according to this diagnostic.

Table 3 shows that worst-case inter-class alignment is not minimized by the same cells that maximize residual margin. The lowest geometry-shaped $\mathrm { C o h e s i o n } _ { \operatorname* { m a x } }$ value occurs at $( \mu , \lambda ) =$ $( 1 . 0 , 0 . 0 0 1 )$ , where it reaches 0.136. Large-µ settings such as (100.0, 0.01) and (100.0, 0.1) also yield low alignment, with $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 1 3 9$ . By contrast, the highest-margin cell $( 1 0 . 0 , 0 . 1 )$ has the largest observed $\mathrm { C o h e s i o n } _ { \operatorname* { m a x } }$ value (0.319). This confirms that residual separation, span utilization, and worst-case alignment control are distinct aspects of the learned geometry and are not simultaneously optimized by a single $( \mu , \lambda )$ setting.

Reading the COIL100 sweep tables together, the COIL-100 sweep supports a more nuanced interpretation than a simple accuracy comparison. Geometry-shaped embeddings preserve perfect or near-perfect SRC accuracy across the sweep and allow residual geometry to be modulated by $( \mu , \lambda )$ . However, the CE-shaped reference remains the strongest single representation in terms of effective rank and worst-case alignment. COIL-100 therefore functions primarily as a controlled near-ceiling setting for examining how masked-ridge geometry shaping affects residual ordering and class-geometry diagnostics under fixed SRC/OMP inference, rather than as evidence of uniform dominance of geometry-shaped training over CE on this dataset.

For concreteness, we reference two sweep locations that typify the trade-off without treating either as a selected model. The accuracy-tied, margin-selected representative cell is $( \mu , \lambda ) = ( 1 . 0 , 0 . 0 1 )$ , with perfect accuracy, Top-2 margin 0.947, effective rank 27.754, and $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 1 5 6$ . A contrasting span-utilizing and low-alignment region appears at $( \mu , \lambda ) = ( 1 0 0 . 0 , 0 . 1 )$ , where effective rank reaches 32.793 and $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 1 3 9$ , while the Top-2 margin remains strong at 0.937. Both points lie in the same near-ceiling accuracy regime; their role is descriptive, illustrating how $( \mu , \lambda )$ shifts which aspect of residual-inference conditioning is emphasized under fixed inference.

Table 2: COIL100- Effective rank (learned) over the geometry sweep. Rows correspond to $\mu$ and columns to $\lambda .$ . 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.01</td><td>0.1</td></tr><tr><td>0.1</td><td>26.429</td><td>26.322</td><td>27.398</td></tr><tr><td>1.0</td><td>25.825</td><td>27.754</td><td>28.868</td></tr><tr><td>10.0</td><td>26.057</td><td>27.854</td><td>28.886</td></tr><tr><td>100.0</td><td>28.110</td><td>30.403</td><td>32.793</td></tr></table>

Table 3: COIL-100 $\mathrm { C o h e s i o n } _ { \operatorname* { m a x } }$ over the geometry sweep. Lower values indicate weaker inter-class subspace alignment. Rows correspond to $\mu$ and columns to $\lambda .$ . 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.01</td><td>0.1</td></tr><tr><td>0.1</td><td>0.231</td><td>0.225</td><td>0.258</td></tr><tr><td>1.0</td><td>0.136</td><td>0.156</td><td>0.169</td></tr><tr><td>10.0</td><td>0.192</td><td>0.219</td><td>0.319</td></tr><tr><td>100.0</td><td>0.152</td><td>0.139</td><td>0.139</td></tr></table>

Table 4: COIL100- Top-2 residual margin (learned) over the geometry sweep. Rows correspond to $\mu$ and columns to $\lambda .$ 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.01</td><td>0.1</td></tr><tr><td>0.1</td><td>0.748</td><td>0.748</td><td>0.743</td></tr><tr><td>1.0</td><td>0.943</td><td>0.947</td><td>0.952</td></tr><tr><td>10.0</td><td>0.940</td><td>0.943</td><td>0.953</td></tr><tr><td>100.0</td><td>0.932</td><td>0.934</td><td>0.937</td></tr></table>

Table 5: COIL100- Accuracy (SRC) over the geometry sweep. Rows correspond to $\mu$ and columns to λ. 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.01</td><td>0.1</td></tr><tr><td>0.1</td><td>1.000</td><td>1.000</td><td>1.000</td></tr><tr><td>1.0</td><td>1.000</td><td>1.000</td><td>0.999</td></tr><tr><td>10.0</td><td>0.999</td><td>0.999</td><td>0.999</td></tr><tr><td>100.0</td><td>1.000</td><td>1.000</td><td>0.999</td></tr></table>

# 4.3 Text Experiments: TREC (Question Classification)

Dataset and task. We evaluate our text pipeline on the TREC question classification benchmark (coarse labels) (Li and Roth, 2002), a 6-way classification task where each example is a naturallanguage question and the goal is to predict its coarse category (ABBR, DESC, ENTY, HUM, LOC, NUM). Compared to COIL-100, intra-class variability is predominantly semantic (paraphrases, lexical variation) rather than geometric transformations. This setting probes whether geometry shaping can improve the residual behavior of class-wise reconstruction inference in a semantically structured embedding space, under a fixed residual-based decision rule.

Encoder and representations. All TREC experiments use a RoBERTa-base encoder Liu et al. (2019). Given a tokenized question, we compute mean pooling over the final-layer token embeddings (masked by the attention mask), then apply a lightweight projection head $\mathrm { ( L i n e a r  R e L U  }$ Linear) to obtain $p { = } 1 2 8 { \mathrm { - } } \mathrm { d i m e n s i o n a l }$ embeddings, followed by $\ell _ { 2 }$ normalization. We compare: (i) a pretrained representation (baseline) and (ii) a geometry-shaped representation obtained by fine-tuning under the geometry objectives controlled by $( \mu , \lambda )$ . For reference, we also report a CE-shaped representation evaluated under the same fixed SRC rule.

Setup and protocol. We perform a geometry sweep over $\mu \in \{ 0 . 1 , 1 , 5 , 1 0 , 1 0 0 \}$ and $\lambda \ \in$ $\{ 0 . 0 0 1 , 0 . 0 0 3 , 0 . 0 1 , 0 . 0 3 , 0 . 1 , 1 . 0 \}$ , repeating each configuration with seeds {0, 1}, for a total of $5 \times 6 \times 2 = 6 0$ runs. The pretrained baseline is evaluated under the same inference and metric pipeline to provide a consistent reference point. The sweep is used to characterize the response surface of the fixed SRC rule under geometry shaping, rather than to perform hyperparameter selection.

Training settings. Fine-tuning runs are trained for 20 epochs using AdamW with learning rate $2 \times 1 0 ^ { - 5 }$ and weight decay $1 0 ^ { - 2 }$ . Inputs are tokenized with maximum sequence length 32. To make masked self-expressiveness meaningful, training uses class-balanced batches with $m { = } 1 2$ examples per class (6 classes), resulting in batch size 72. Unless stated otherwise, the outer weights of the geometry losses are fixed to $\lambda _ { \mathrm { S E } } { = } 1 . 0 , \ \beta _ { \mathrm { a n c h o r } } { = } 1 . 0$ , and $\lambda _ { \mathrm { { r e p } } } { = } 1 . 0$ , with repulsion subspace dimension $d _ { \mathrm { r e p } } { = } 6$ (excluding the top singular direction). Only the inner hyperparameters $( \mu , \lambda )$ are swept.

Inference and evaluation. Evaluation is performed with a sparse representation classifier (SRC) on top of frozen embeddings: given a test embedding, we compute its sparse reconstruction over the training dictionary via Orthogonal Matching Pursuit (OMP-s) and classify using class-wise reconstruction residuals. We report SRC accuracy and Top-2 residual margins, which quantify separation in the observed residual ordering under the fixed residual decision rule (and, per Section 3.3, relate to stability under bounded residual perturbations). In parallel, we report geometry descriptors (effective rank and worst-case inter-class alignment $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } )$ . All results are aggregated across seeds for each $( \mu , \lambda )$ .

# 4.3.1 TREC: Geometry sweep evaluation (RoBERTa-base)

We evaluate geometry shaping on TREC via a sweep of $( \mu , \lambda )$ and compare against the pretrained RoBERTa baseline and a CE-shaped reference representation. All representations are evaluated under the same fixed $\mathrm { S R C / O M P }$ residual inference rule. Unlike COIL-100, TREC is not an accuracyceiling setting for the pretrained representation; hence, we jointly interpret SRC accuracy and Top-2 residual margin together with geometry diagnostics such as effective rank and $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } .$ .

Table 6 summarizes the pretrained baseline, the geometry-shaped family, and the CE-shaped reference. Since two geometry-shaped cells attain the same maximum SRC accuracy, the representative geometry-shaped cell is selected using the same accuracy-constrained stability rule used in the COIL-100 analysis: among cells with maximum SRC accuracy, we select the cell with the largest Top-2 residual margin. Under this rule, the representative geometry-shaped cell is $( \mu , \lambda ) = ( 1 0 . 0 , 1 . 0 )$ , which attains SRC accuracy 0.946, Top-2 margin 0.632, effective rank 53.608, and $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 9 1 3$ .

Relative to pretrained RoBERTa, geometry shaping substantially improves SRC/OMP performance under the fixed residual classifier. Accuracy increases from 0.867 to 0.946 at the representative geometry-shaped cell, while the Top-2 residual margin increases from 0.514 to 0.632. Averaged over the entire $( \mu , \lambda )$ grid, the geometry-shaped family remains above the pretrained baseline, with mean accuracy 0.907 and mean Top-2 margin 0.581. Thus, the gain is not confined to a single isolated configuration, although the strongest results occur in the large-λ region of the sweep.

The CE-shaped reference remains the strongest representation on TREC under the same fixed SRC/OMP evaluation, reaching SRC accuracy 0.9880 and Top-2 margin 0.9451. We interpret this row as a supervised reference point indicating the headroom available when labels are used through a standard discriminative cross-entropy objective. The geometry-shaped sweep is therefore not presented as a replacement for CE training, but as a way to study how masked-ridge geometry shaping modifies the residual family seen by a fixed SRC/OMP classifier.

Tables 9 and 10 show that both SRC accuracy and Top-2 residual margin are strongly affected by λ. At the smallest regularization value, $\lambda = 0 . 0 0 1$ , accuracy is lowest across the grid, ranging from 0.819 to 0.862, and margins are also comparatively small, ranging from 0.481 to 0.503. As λ increases, both accuracy and margins improve markedly. The highest SRC accuracy, 0.946, is attained at both $( \mu , \lambda ) = ( 1 0 . 0 , 1 . 0 )$ and (100.0, 1.0), while the largest Top-2 margin is obtained at (10.0, 1.0), with $m _ { \mathrm { t o p 2 } } = 0 . 6 3 2$ . This co-movement between accuracy and residual-margin amplification suggests that, on TREC, improvements in fixed SRC/OMP performance are closely associated with more stable residual ordering.

Table 7 shows that effective rank increases primarily with λ, with only milder dependence on $\mu .$ At $\lambda = 0 . 0 0 1$ , effective rank lies around 45.2–45.6, whereas at $\lambda = 1 . 0$ it rises to approximately 53.6– 55.2. The largest effective rank in the geometry-shaped sweep is 55.160, attained at $( \mu , \lambda ) = ( 5 . 0 , 1 . 0 )$ , which is comparable to the pretrained RoBERTa effective rank 55.011. The representative accuracytied, margin-selected cell has effective rank 53.608, slightly below the pretrained baseline but clearly higher than the CE-shaped reference effective rank 32.83. This indicates that effective rank is not a monotonic proxy for SRC accuracy: CE training produces the strongest residual classifier despite a lower effective-rank summary.

Table 8 shows a clearer trend in worst-case inter-class alignment. $\mathrm { C o h e s i o n } _ { \operatorname* { m a x } }$ is highest at very small $\lambda ,$ around 0.967–0.969, and decreases as λ increases. The lowest geometry-shaped value is 0.908, obtained at $( \mu , \lambda ) = ( 0 . 1 , 1 . 0 )$ , while the representative geometry-shaped cell (10.0, 1.0) achieves $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 9 1 3$ . This improves over the pretrained baseline (0.938), although it remains far above the CE-shaped reference (0.634). Thus, geometry shaping reduces worst-case alignment in the large-λ region, but CE training produces a much stronger separation according to this coarse subspace diagnostic.

Reading the TREC sweep tables together, the TREC sweep shows a coherent large-λ regime in which accuracy, residual margins, effective rank, and $\mathrm { C o h e s i o n } _ { \operatorname* { m a x } }$ all improve relative to the weakest regularization settings. However, these diagnostics are not perfectly aligned. For example, the maximum effective rank occurs at (5.0, 1.0), whereas the selected accuracy-tied, margin-maximizing cell is (10.0, 1.0), and the lowest Cohesionmax occurs at (0.1, 1.0). This indicates that $( \mu , \lambda )$ controls several distinct aspects of the residual geometry: residual separability, span utilization, and worstcase subspace alignment.

Comparing across representations in Table $6 ,$ geometry shaping improves substantially over pretrained RoBERTa in fixed $\mathrm { S R C / O M P }$ accuracy and margins, while also improving Cohesionmax at the selected cell. The sweep average, however, has slightly worse $\mathrm { C o h e s i o n } _ { \operatorname* { m a x } }$ than pretrained (0.944 versus 0.938), reflecting the poor alignment behavior of the small-λ region. The CE-shaped reference remains much stronger in accuracy, margin, and ${ \mathrm { C o h e s i o n } } _ { \operatorname* { m a x } }$ , but has lower effective rank. This reinforces the interpretation that effective rank and worst-case alignment are complementary diagnostics rather than standalone performance surrogates.

Table 6: TREC summary comparison between the pretrained RoBERTa baseline, geometry-shaped embeddings, and a CE-shaped reference, all evaluated with the same fixed SRC/OMP inference. The “acc-tied, margin-selected” row denotes the cell selected by first maximizing SRC accuracy and then, among accuracy ties, maximizing the Top-2 residual margin. The grid average reports the mean over all $( \mu , \lambda )$ configurations. 

<table><tr><td>Representation</td><td>SRC Acc</td><td>Top-2 Margin</td><td>Eff. Rank</td><td>Cohesionmax</td></tr><tr><td>Pretrained</td><td>0.867</td><td>0.514</td><td>55.011</td><td>0.938</td></tr><tr><td>Geometry-shaped (acc-tied, margin-selected)</td><td>0.946</td><td>0.632</td><td>53.608</td><td>0.913</td></tr><tr><td>Geometry-shaped (avg over grid)</td><td>0.907</td><td>0.581</td><td>50.257</td><td>0.944</td></tr><tr><td>CE-shaped (reference)</td><td>0.9880</td><td>0.9451</td><td>32.8300</td><td>0.6340</td></tr></table>

Table 7: Effective rank (geometry-shaped) over the sweep on TREC. Rows correspond to $\mu$ and columns to λ. 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.003</td><td>0.01</td><td>0.03</td><td>0.1</td><td>1.0</td></tr><tr><td>0.1</td><td>45.614</td><td>47.053</td><td>49.341</td><td>51.104</td><td>54.083</td><td>54.422</td></tr><tr><td>1.0</td><td>45.223</td><td>48.021</td><td>50.193</td><td>51.227</td><td>53.229</td><td>54.386</td></tr><tr><td>5.0</td><td>45.467</td><td>48.398</td><td>50.651</td><td>51.210</td><td>52.630</td><td>55.160</td></tr><tr><td>10.0</td><td>45.440</td><td>47.404</td><td>49.348</td><td>50.877</td><td>52.677</td><td>53.608</td></tr><tr><td>100.0</td><td>45.484</td><td>47.826</td><td>50.273</td><td>50.182</td><td>52.644</td><td>54.534</td></tr></table>

For concreteness, we reference two sweep locations that typify distinct regimes without treating either as a selected model. The accuracy-tied, margin-selected point is $( \mu , \lambda ) = ( 1 0 . 0 , 1 . 0 )$ , with accuracy 0.946, Top-2 margin 0.632, effective rank 53.608, and $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 9 1 3$ . A high-span point occurs at $( \mu , \lambda ) = ( 5 . 0 , 1 . 0 )$ , where effective rank reaches 55.160, while accuracy remains high at 0.933 and Top-2 margin remains strong at 0.605. A low-alignment point occurs at (0.1, 1.0), where $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 9 0 8$ , with accuracy 0.934 and margin 0.610. These points illustrate how the sweep shifts emphasis between residual separability, span utilization, and inter-class alignment under identical fixed SRC/OMP inference.

Table 8: TREC Cohesionmax over the geometry sweep. Lower values indicate weaker inter-class subspace alignment. Rows correspond to $\mu$ and columns to $\lambda .$ . 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.003</td><td>0.01</td><td>0.03</td><td>0.1</td><td>1.0</td></tr><tr><td>0.1</td><td>0.967</td><td>0.966</td><td>0.954</td><td>0.942</td><td>0.927</td><td>0.908</td></tr><tr><td>1.0</td><td>0.968</td><td>0.962</td><td>0.953</td><td>0.939</td><td>0.920</td><td>0.915</td></tr><tr><td>5.0</td><td>0.968</td><td>0.962</td><td>0.946</td><td>0.939</td><td>0.924</td><td>0.909</td></tr><tr><td>10.0</td><td>0.969</td><td>0.964</td><td>0.964</td><td>0.944</td><td>0.927</td><td>0.913</td></tr><tr><td>100.0</td><td>0.969</td><td>0.964</td><td>0.950</td><td>0.938</td><td>0.934</td><td>0.915</td></tr></table>

Table 9: Top-2 residual margin (geometry-shaped) over the sweep on TREC. Rows correspond to $\mu$ and columns to $\lambda .$ . 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.003</td><td>0.01</td><td>0.03</td><td>0.1</td><td>1.0</td></tr><tr><td>0.1</td><td>0.484</td><td>0.558</td><td>0.587</td><td>0.595</td><td>0.591</td><td>0.610</td></tr><tr><td>1.0</td><td>0.481</td><td>0.575</td><td>0.587</td><td>0.600</td><td>0.610</td><td>0.629</td></tr><tr><td>5.0</td><td>0.503</td><td>0.572</td><td>0.596</td><td>0.606</td><td>0.621</td><td>0.605</td></tr><tr><td>10.0</td><td>0.492</td><td>0.577</td><td>0.587</td><td>0.603</td><td>0.626</td><td>0.632</td></tr><tr><td>100.0</td><td>0.497</td><td>0.562</td><td>0.599</td><td>0.617</td><td>0.620</td><td>0.609</td></tr></table>

Table 10: SRC accuracy (geometry-shaped) over the sweep on TREC. Rows correspond to $\mu$ and columns to $\lambda .$ . 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.003</td><td>0.01</td><td>0.03</td><td>0.1</td><td>1.0</td></tr><tr><td>0.1</td><td>0.854</td><td>0.880</td><td>0.909</td><td>0.928</td><td>0.921</td><td>0.934</td></tr><tr><td>1.0</td><td>0.819</td><td>0.886</td><td>0.901</td><td>0.932</td><td>0.928</td><td>0.940</td></tr><tr><td>5.0</td><td>0.831</td><td>0.900</td><td>0.921</td><td>0.922</td><td>0.938</td><td>0.933</td></tr><tr><td>10.0</td><td>0.829</td><td>0.894</td><td>0.918</td><td>0.927</td><td>0.940</td><td>0.946</td></tr><tr><td>100.0</td><td>0.862</td><td>0.886</td><td>0.915</td><td>0.928</td><td>0.936</td><td>0.946</td></tr></table>

# 4.4 EEG Experiments: AD vs HC (Connectivity Features)

Task and data representation. For the purpose of this study we use the CAUEEG dataset (Kim et al., 2023) This dataset contains EEG recordings come with detailed clinical annotations and event histories, providing comprehensive data for training and evaluating data analysis. Patients were diagnosed with normal (HC) or dementia (AD). We consider binary classification AD and HC using resting-state EEG connectivity features (Oikonomou et al., 2025). Unlike the image (COIL-100) and text (TREC) pipelines, there is no pretrained encoder: the input is a fixed feature representation derived from EEG preprocessing and functional connectivity construction, as detailed in (Oikonomou et al., 2025). Furthermore, all train/validation/test partitions are subject-disjoint: no subject appears in more than one split, preventing identity leakage across examples/samples.

Geometry shaping with fixed SRC inference. We follow the same contract as in the previous sections: training shapes geometry only, while inference uses a fixed SRC rule. Specifically, we learn a lightweight embedding mapping trained with the geometry objectives and evaluate with SRC using Orthogonal Matching Pursuit (OMP-s) at a fixed sparsity level $s { = } 3 0$ . We sweep the inner geometry hyperparameters over $\mu \in \{ 0 . 1 , 1 , 5 , 1 0 \}$ and $\lambda \in \{ 1 0 ^ { - 3 } , 1 0 ^ { - 2 } , 1 0 ^ { - 1 } , 1 \}$ , keeping the remaining protocol fixed.

Training settings (EEG). All EEG runs operate on fixed connectivity feature vectors (no pretrained encoder). We train a lightweight MLP encoder to produce ℓ2-normalized embeddings (output dimension $p = 1 2 8 )$ . Training uses AdamW with learning rate $1 0 ^ { - 3 }$ and weight decay $1 0 ^ { - 4 } .$ for 60 epochs. Because the geometry objectives are batch-structured, optimization proceeds via class-balanced mini-batches: at each training step we sample m = 12 examples per class (binary task; batch size 24) and perform 50 balanced-batch steps per epoch. Unless stated otherwise, the outer loss weights are fixed to $\lambda _ { \mathrm { S E } } = 1 . 0 , \beta _ { \mathrm { a n c h o r } } = 1 . 0$ , and $\lambda _ { \mathrm { r e p } } = 1 . 0$ ; only the inner hyperparameters $( \mu , \lambda )$ are varied in the sweep.

Evaluation metrics. We report SRC balanced accuracy (to control for class imbalance), the Top-2 residual margin (median; stability diagnostic), and two geometry descriptors: effective rank and $\mathrm { C o h e s i o n } _ { \operatorname* { m a x } }$ . We compare (i) the raw EEG features used directly with SRC, against (ii) geometry-shaped embeddings evaluated with the same fixed SRC rule.

# 4.4.1 EEG: Geometry sweep evaluation (AD vs HC)

We evaluate geometry shaping on EEG (AD vs. HC) via a sweep of $( \mu , \lambda )$ and compare against raw connectivity features and a CE-shaped reference representation. All representations are evaluated under the same fixed SRC/OMP inference rule (OMP-s with s = 30). Since this setting starts from handcrafted connectivity features rather than a strong pretrained encoder, we interpret the results jointly in terms of predictive performance, residual ordering stability, and representation geometry. Predictive performance is measured by balanced accuracy, residual stability by the Top-2 residual margin, and geometry by effective rank and worst-case inter-class alignment Cohesionmax.

Table 11 summarizes the raw EEG baseline, the geometry-shaped family, and the CE-shaped reference. Because two geometry-shaped cells attain the same maximum balanced accuracy, the representative geometry-shaped cell is selected using an accuracy-constrained stability rule: among all cells with maximum balanced accuracy, we select the cell with the largest Top-2 residual margin. Under this rule, the representative geometry-shaped cell is $( \mu , \lambda ) = ( 0 . 1 , 0 . 1 )$ , which attains balanced accuracy 0.871, Top-2 margin 0.899, effective rank 2.154, and $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 9 9 6$ .

Relative to raw EEG connectivity features, geometry shaping substantially improves fixed-SRC performance. The raw baseline achieves balanced accuracy 0.821 with a small Top-2 margin 0.145, indicating weak residual separation under SRC/OMP inference. The geometry-shaped sweep average improves balanced accuracy to 0.851 and increases the Top-2 margin to 0.869, showing that the shaped representations produce much more stable residual ordering on average. The representative accuracy-tied, margin-selected geometry-shaped cell further improves balanced accuracy to 0.871, while maintaining a high margin of 0.899.

The CE-shaped reference provides a strong supervised comparison. It reaches balanced accuracy 0.859 and Top-2 margin 0.9236. Thus, the geometry-shaped peak slightly exceeds CE in balanced accuracy, whereas CE achieves the stronger residual margin. However, the CE-shaped reference also has high worst-case alignment $( \mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 9 8 0 )$ and relatively low effective rank (9.64) compared with the raw feature space. We therefore interpret CE primarily as a predictive and residual-margin reference rather than as a geometry-controlled solution.

Table 15 shows that the best balanced accuracy is concentrated in the small- $\cdot \mu ,$ , small-to-moderateλ region. The maximum balanced accuracy 0.871 occurs at both $( \mu , \lambda ) = ( 0 . 1 , 0 . 0 1 )$ and (0.1, 0.1). For $\lambda \in \{ 0 . 0 0 1 , 0 . 0 1 , 0 . 1 \}$ , balanced accuracy remains relatively stable across the grid, mostly between 0.853 and 0.871. In contrast, the $\lambda = 1 . 0$ column is consistently weaker, with balanced accuracy between 0.791 and 0.835. This indicates that overly strong ridge regularization degrades predictive performance in this EEG setting.

Table 14 shows that geometry-shaped embeddings produce high Top-2 residual margins throughout the sweep. Margins range from 0.829 to 0.937, far above the raw baseline margin of 0.145. The largest margin occurs at $( \mu , \lambda ) = ( 1 . 0 , 0 . 1 )$ ), where $m _ { \mathrm { t o p 2 } } = 0 . 9 3 7$ . The representative peak-accuracy cell (0.1, 0.1) does not maximize the margin, but its margin 0.899 remains within the high-stability regime. Thus, geometry shaping consistently improves residual-ordering stability relative to raw EEG features, even when balanced accuracy varies across operating points.

The geometry diagnostics reveal a more nuanced picture. Table 12 shows that effective rank is maximized in the intermediate-λ region, especially at $\lambda = 0 . 0 1$ . The largest effective rank occurs at $( \mu , \lambda ) = ( 5 . 0 , 0 . 0 1 )$ , where it reaches 14.757, closely followed by (10.0, 0.01) with 13.968 and $( 1 . 0 , 0 . 0 1 )$ with 13.886. By contrast, the representative peak-accuracy cell (0.1, 0.1) has much lower effective rank (2.154), indicating that the best predictive cell lies in a compact, low-rank regime rather than in the most span-utilizing region of the sweep.

Table 13 shows that worst-case inter-class alignment also depends strongly on $( \mu , \lambda )$ . The $\lambda = 1 . 0$ column yields $\mathrm { { C o h e s i o n } _ { \mathrm { { m a x } } } = 1 . 0 }$ for all values of $\mu ,$ indicating a near-degenerate worst-case alignment regime. The lowest alignment occurs at $( \mu , \lambda ) = ( 5 . 0 , 0 . 0 1 )$ , where $\mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 6 3 9$ followed by (10.0, 0.1) with 0.671 and (10.0, 0.01) with 0.693. The representative peak-accuracy cell has high alignment $( \mathrm { C o h e s i o n } _ { \mathrm { m a x } } = 0 . 9 9 6 )$ , showing that peak predictive performance does not coincide with strongest near-overlap control.

Taken jointly, the EEG sweep separates predictive performance, residual stability, and geometric conditioning. Geometry shaping improves balanced accuracy and residual margins relative to raw EEG features. However, the accuracy-optimal region is compact and highly aligned according to the coarse subspace diagnostics, whereas the best geometry-controlled region occurs at $( \mu , \lambda ) = ( 5 . 0 , 0 . 0 1 )$ with high effective rank, substantially reduced $\mathrm { C o h e s i o n } _ { \mathrm { m a x } }$ , and still competitive balanced accuracy (0.855). Thus, the main trade-off in EEG is not simply between accuracy and margin, since margins remain high across the shaped family, but between peak predictive performance and explicit geometric conditioning.

For concreteness, we reference two operating points without treating either as a universally selected model. The accuracy-tied, margin-selected point $( \mu , \lambda ) = ( 0 . 1 , 0 . 1 )$ achieves the highest balanced accuracy (0.871) and a high Top-2 margin (0.899), but has low effective rank (2.154) and high worst-case alignment (0.996). A more geometry-controlled point appears at $( \mu , \lambda ) = ( 5 . 0 , 0 . 0 1 )$ , where effective rank is highest (14.757) and $\mathrm { C o h e s i o n } _ { \operatorname* { m a x } }$ is lowest (0.639), while balanced accuracy remains competitive (0.855) and Top-2 margin remains high (0.850). These points illustrate how $( \mu , \lambda )$ moves the EEG representation along a predictive accuracy–geometric conditioning axis under fixed residual inference.

Table 11: EEG (AD vs HC) summary comparison between raw connectivity features, geometryshaped embeddings, and a CE-shaped reference, all evaluated with fixed SRC inference (OMP-s with $s = 3 0 )$ . The “acc-tied, margin-selected” row denotes the cell selected by first maximizing SRC balanced accuracy and then, among accuracy ties, maximizing the Top-2 residual margin. The grid average reports the mean over all $( \mu , \lambda )$ configurations. 

<table><tr><td>Representation</td><td>SRC bal. acc</td><td>Top-2 Margin</td><td>Eff. Rank</td><td>Cohesionmax</td></tr><tr><td>Raw EEG (connectivity features)</td><td>0.821</td><td>0.145</td><td>50.053</td><td>0.955</td></tr><tr><td>Geometry-shaped (avg over grid)</td><td>0.851</td><td>0.869</td><td>5.925</td><td>0.885</td></tr><tr><td>Geometry-shaped (acc-tied, margin-selected)</td><td>0.871</td><td>0.899</td><td>2.154</td><td>0.996</td></tr><tr><td>CE-shaped (reference)</td><td>0.859</td><td>0.9236</td><td>9.640</td><td>0.980</td></tr></table>

Table 12: EEG geometry sweep: effective rank of the geometry-shaped embeddings. Rows correspond to $\mu$ and columns to λ. 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td></tr><tr><td>0.1</td><td>3.194</td><td>5.782</td><td>2.154</td><td>2.627</td></tr><tr><td>1.0</td><td>3.498</td><td>13.886</td><td>2.736</td><td>2.549</td></tr><tr><td>5.0</td><td>5.024</td><td>14.757</td><td>4.622</td><td>2.016</td></tr><tr><td>10.0</td><td>6.771</td><td>13.968</td><td>9.417</td><td>1.798</td></tr></table>

Table 13: EEG Cohesionmax over the geometry sweep. Lower values indicate weaker inter-class subspace alignment. Rows correspond to $\mu$ and columns to $\lambda .$ . 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td></tr><tr><td>0.1</td><td>0.999</td><td>0.963</td><td>0.996</td><td>1.0</td></tr><tr><td>1.0</td><td>0.892</td><td>0.810</td><td>0.995</td><td>1.0</td></tr><tr><td>5.0</td><td>0.826</td><td>0.639</td><td>0.878</td><td>1.0</td></tr><tr><td>10.0</td><td>0.801</td><td>0.693</td><td>0.671</td><td>1.0</td></tr></table>

Table 14: EEG geometry sweep: Top-2 Margin (geometry-shaped embeddings). Rows correspond to $\mu$ and columns to λ. 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td></tr><tr><td>0.1</td><td>0.829</td><td>0.877</td><td>0.899</td><td>0.830</td></tr><tr><td>1.0</td><td>0.867</td><td>0.830</td><td>0.937</td><td>0.890</td></tr><tr><td>5.0</td><td>0.829</td><td>0.850</td><td>0.891</td><td>0.901</td></tr><tr><td>10.0</td><td>0.856</td><td>0.839</td><td>0.868</td><td>0.905</td></tr></table>

Table 15: EEG geometry sweep: SRC balanced accuracy (geometry-shaped embeddings). Rows correspond to $\mu$ and columns to λ. 

<table><tr><td>μ\λ</td><td>0.001</td><td>0.01</td><td>0.1</td><td>1.0</td></tr><tr><td>0.1</td><td>0.863</td><td>0.871</td><td>0.871</td><td>0.791</td></tr><tr><td>1.0</td><td>0.860</td><td>0.853</td><td>0.869</td><td>0.819</td></tr><tr><td>5.0</td><td>0.858</td><td>0.855</td><td>0.859</td><td>0.834</td></tr><tr><td>10.0</td><td>0.862</td><td>0.855</td><td>0.855</td><td>0.835</td></tr></table>

Across-dataset synthesis (conditioning regimes under fixed residual inference). Across COIL-100, TREC, and EEG, all representations are evaluated under the same fixed SRC/OMP residual rule, and margins are used as stability diagnostics alongside accuracy. COIL-100 is an accuracy-ceiling regime: all methods achieve near-perfect SRC accuracy, so the sweep mainly probes how geometry shaping modulates residual margins and class-geometry diagnostics under saturated performance; in this setting, CE-shaped embeddings remain a very strong geometry and margin reference. TREC is a strong-CE but geometry-responsive regime: CE shaping gives the best fixed-SRC performance, while geometry shaping still improves substantially over pretrained RoBERTa and exposes a clear large-λ region with stronger residual margins and improved alignment within the shaped family. EEG represents a low-pretraining regime based on handcrafted connectivity features: geometry shaping improves balanced accuracy and residual-ordering stability over raw features, while revealing a trade-off between peak predictive performance and explicit geometry control. Overall, the sweeps provide a controlled map from training-time geometric regularization to test-time residual-conditioning behavior under a fixed inference principle.

# 5 Conclusions

This work argues for a geometry-centered contract for reconstruction-based classification: inference is an unmodified residual rule, while learning is responsible for producing embeddings under which residual comparisons are well-posed and stable. Under a span-level idealization based on empirical class spans and their projection residuals, we define a residual margin that certifies stable residual ordering and show that several failure modes are intrinsic to geometry. In particular, span overlap, dominance, and near-overlap through small principal angles force the span-level margins to collapse in worst-case directions, independently of any particular numerical routine used to approximate residuals. Conversely, when class spans provide adequate coverage of the data and are sufficiently separated, we obtain a quantitative lower bound on the idealized residual margin, making explicit which geometric properties support stable reconstruction-based decisions.

We then design training objectives that target these properties without coupling training to SRC. Masked ridge self-expressiveness encourages within-class reconstructability while penalizing cross-class reconstruction pathways; a repulsion term discourages inter-class span alignment; and a scale-aware non-discriminative anchor mitigates collapse under normalized embeddings. Importantly, these objectives are not proposed as a universal replacement for discriminative training. Instead, they provide controlled geometric interventions and a diagnostic lens for residual inference under a fixed test-time rule. This perspective is essential because common geometric diagnostics do not necessarily co-vary: residual margins, worst-case span alignment, and effective rank can trade off, and improvements in one axis need not imply monotone gains in another. Our response-surface analyses make these trade-offs explicit and show how (µ, λ) shifts the representation among qualitatively different conditioning regimes under identical fixed inference.

The cross-dataset results support a boundary-regime view. COIL-100 behaves as an accuracyceiling and CE-strong regime: frozen ImageNet, CE-shaped, and geometry-shaped representations all achieve near-perfect SRC/OMP accuracy, so the informative signal lies primarily in residual margins and geometry diagnostics rather than in raw accuracy. In this setting, geometry shaping can increase residual separation relative to frozen ImageNet features, but it does not uniformly dominate CE-shaped embeddings in effective rank or worst-case alignment. TREC exhibits a strong-CE but geometry-responsive regime: CE-shaped embeddings provide the strongest downstream SRC accuracy and margins, while geometry shaping still improves substantially over pretrained RoBERTa and reveals a large-λ region in which residual ordering and inter-class alignment improve within the shaped family. EEG provides a complementary low-pretraining regime based on handcrafted connectivity features: geometry shaping improves balanced accuracy and residual-ordering stability relative to raw features, but the peak-accuracy operating point and the most explicitly geometrycontrolled operating point do not coincide. Thus, the empirical message is not that a single geometry objective uniformly dominates across modalities, but that explicit geometry shaping provides a controllable way to probe and modify the residual-conditioning regime induced by a representation.

Including cross-entropy as a reference clarifies the boundary of the approach. In some settings, especially under strong visual or textual supervision, CE training can implicitly produce representations that are highly favorable for fixed residual-based evaluation. In such regimes, geometry-only objectives should not be interpreted as a replacement for discriminative fine-tuning. Rather, they expose how residual margins, span utilization, and inter-class alignment respond to explicit geometric interventions. In settings where the initial feature geometry is not already SRC-aligned, as in the EEG connectivity experiments, geometry shaping can yield direct predictive and residual-stability benefits relative to raw features. These observations suggest that the value of geometry shaping is conditional: it is most informative when used to diagnose, stress-test, or regularize residual inference, and most practically beneficial when the baseline representation does not already provide stable class-wise residual comparisons.

The limitations are correspondingly geometric and conditional. Our theory operates at a span level with finite-sample subspace estimates; the proposed diagnostics are descriptive and should not be interpreted as universally predictive proxies; and stability depends on adequate class-wise coverage and on operating in a regime where the practical residual family, such as OMP-based SRC residuals, remains a controlled perturbation of the span-level residual ordering. The experiments also show that high predictive performance can occur in compact or prototype-like regimes where effective rank and worst-case alignment diagnostics are not optimized. This reinforces the need to interpret margins, rank, and alignment jointly rather than reducing representation quality to a single scalar geometry score. Within this scope, the paper provides a coherent geometric language—obstructions, sufficient conditions, and controllable surrogates—for reasoning about when residual-based classification is interpretable and when it is intrinsically ambiguous.

Future work can extend this geometry-first contract along three concrete axes. First, it would be valuable to further formalize the conditional solver-level interpretation by quantifying how finitesample subspace estimation, dictionary size or thinning, and sparsity-level or stopping-rule choices contribute to solver-side residual perturbations and hence to stability-transfer conditions. Such statements would remain inference-aware without collapsing the learning objective into a particular end-to-end SRC pipeline. Second, the diagnostic program could be broadened beyond principal-angle and effective-rank proxies toward measures that more directly capture class-conditional coverage, prototype concentration, and cross-class leakage in self-expressive coefficient structure. Third, because discriminative training can implicitly induce strong class-wise concentration and related geometric collapse phenomena, it is important to characterize when CE-induced geometry aligns with reconstruction-based inference and when it merely produces high accuracy through a different geometric mechanism. Response-surface analyses of the kind used here could then help identify regimes where explicit geometric interventions offer robustness, interpretability, or diagnostic benefits beyond standard fine-tuning.

# References

M. Abavisani and V. M. Patel. Deep sparse representation-based classification. IEEE Signal Processing Letters, 26(6):948–952, 2019. ISSN 1070-9908, 1558-2361. doi: 10.1109/LSP.2019. 2913022. URL http://arxiv.org/abs/1904.11093.

M. Aharon, M. Elad, and A. Bruckstein. k-SVD: An algorithm for designing overcomplete dictionaries for sparse representation. IEEE Transactions on Signal Processing, 54(11):4311–4322, 2006. ISSN 1053-587X. doi: 10.1109/TSP.2006.881199. URL http://ieeexplore.ieee.org/document/ 1710377/.   
Q. Bertrand, Q. Klopfenstein, M. Blondel, S. Vaiter, A. Gramfort, and J. Salmon. Implicit differentiation of lasso-type models for hyperparameter optimization. In H. D. III and A. Singh, editors, Proceedings of the 37th International Conference on Machine Learning, volume 119 of Proceedings of Machine Learning Research, pages 810–821. PMLR, 13–18 Jul 2020. URL https://proceedings.mlr.press/v119/bertrand20a.html.   
A. Bjorck and G. Golub. Numerical methods for computing angles between linear subspaces. Mathematics of Computation, 27:123, 07 1973. doi: 10.2307/2005662.   
M. Blondel, Q. Berthet, M. Cuturi, R. Frostig, S. Hoyer, F. Llinares-Lopez, F. Pedregosa, and J.-P. Vert. Efficient and modular implicit differentiation. In S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A. Oh, editors, Advances in Neural Information Processing Systems, volume 35, pages 5230–5242. Curran Associates, Inc., 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/file/ 228b9279ecf9bbafe582406850c57115-Paper-Conference.pdf.   
J. Bolte, E. Pauwels, and S. Vaiter. One-step differentiation of iterative algorithms. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, editors, Advances in Neural Information Processing Systems, volume 36, pages 77089–77103. Curran Associates, Inc., 2023. URL https://proceedings.neurips.cc/paper\_files/paper/2023/file/ f3716db40060004d0629d4051b2c57ab-Paper-Conference.pdf.   
J. Cai, J. Fan, W. Guo, S. Wang, Y. Zhang, and Z. Zhang. Efficient deep embedded subspace clustering. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 21–30, 2022. doi: 10.1109/CVPR52688.2022.00012.   
S. Cai, L. Zhang, W. Zuo, and X. Feng. A probabilistic collaborative representation based approach for pattern classification. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 2950–2959, 2016. doi: 10.1109/CVPR.2016.322.   
J. Cao, C. Ma, T. Yao, S. Chen, S. Ding, and X. Yang. End-to-end reconstruction-classification learning for face forgery detection. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 4103–4112, New Orleans, LA, USA, 2022. IEEE. ISBN 978-1-6654-6946-3. doi: 10.1109/CVPR52688.2022.00408. URL https://ieeexplore.ieee.org/ document/9878441/.   
J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, and L. Fei-Fei. Imagenet: A large-scale hierarchical image database. In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 248–255, 2009. doi: 10.1109/CVPR.2009.5206848.   
D. Donoho. Compressed sensing. IEEE Transactions on Information Theory, 52(4):1289–1306, 2006. ISSN 0018-9448. doi: 10.1109/TIT.2006.871582. URL http://ieeexplore.ieee.org/ document/1614066/.   
E. Elhamifar and R. Vidal. Sparse subspace clustering: Algorithm, theory, and applications. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(11):2765–2781, 2013. doi: 10.1109/TPAMI.2013.57.

I. Evron, E. Moroshko, R. Ward, N. Srebro, and D. Soudry. How catastrophic can catastrophic forgetting be in linear regression? In P.-L. Loh and M. Raginsky, editors, Proceedings of Thirty Fifth Conference on Learning Theory, volume 178 of Proceedings of Machine Learning Research, pages 4028–4079. PMLR, 02–05 Jul 2022. URL https://proceedings.mlr.press/ v178/evron22a.html.   
X. Y. Han, V. Papyan, and D. L. Donoho. Neural collapse under MSE loss: Proximity to and dynamics on the central path. In The Tenth International Conference on Learning Representations, ICLR 2022, Virtual Event, April 25-29, 2022, 2022.   
K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In 2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 770–778, 2016. doi: 10.1109/CVPR.2016.90.   
J. Huang, Q. Qiu, and R. Calderbank. The role of principal angles in subspace classification. IEEE Transactions on Signal Processing, 64(8):1933–1945, 2016. doi: 10.1109/TSP.2015.2500889.   
P. Ji, T. Zhang, H. Li, M. Salzmann, and I. Reid. Deep subspace clustering networks. In Proceedings of the 31st International Conference on Neural Information Processing Systems, NIPS’17, page 23–32, 2017. ISBN 9781510860964.   
M.-j. Kim, Y. C. Youn, and J. Paik. Deep learning-based EEG analysis to classify normal, mild cognitive impairment, and dementia: Algorithms and dataset. NeuroImage, 272, May 2023. ISSN 10538119. doi: 10.1016/j.neuroimage.2023.120054.   
D. P. Kingma and J. Ba. Adam: A method for stochastic optimization, 2017. URL https: //arxiv.org/abs/1412.6980.   
A. V. Knyazev and M. E. Argentati. Principal angles between subspaces in an a-based scalar product: Algorithms and perturbation estimates. SIAM Journal on Scientific Computing, 23(6): 2008–2040, 2002. doi: 10.1137/S1064827500377332.   
X. Li and D. Roth. Learning question classifiers. In COLING 2002: The 19th International Conference on Computational Linguistics, 2002. URL https://aclanthology.org/C02-1150/.   
G. Liu, Z. Lin, S. Yan, J. Sun, Y. Yu, and Y. Ma. Robust recovery of subspace structures by low-rank representation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 35(1):171–184, 2013. ISSN 0162-8828, 2160-9292. doi: 10.1109/TPAMI.2012.88. URL http: //ieeexplore.ieee.org/document/6180173/.   
Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov. Roberta: A robustly optimized bert pretraining approach, 2019. URL https: //arxiv.org/abs/1907.11692.   
I. Loshchilov and F. Hutter. Decoupled weight decay regularization. In International Conference on Learning Representations, 2019.   
J. Lu and S. Steinerberger. Neural collapse under cross-entropy loss. Applied and Computational Harmonic Analysis, 59:224–241, 2022. ISSN 1063-5203. doi: https://doi.org/10.1016/j.acha.2021. 12.011.

J. Miao, X. Zhang, T. Yang, C. Fan, Y. Tian, Y. Shi, and M. Xu. A comprehensive survey on subspace clustering: Methods and applications. Artificial Intelligence Review, 58(11):346, 2025. ISSN 1573-7462. doi: 10.1007/s10462-025-11349-w. URL https://link.springer.com/ 10.1007/s10462-025-11349-w.   
V. Monga, Y. Li, and Y. C. Eldar. Algorithm unrolling: Interpretable, efficient deep learning for signal and image processing, 2020. URL http://arxiv.org/abs/1912.10557.   
S. Nene, S. Nayar, and H. Murase. Columbia Object Image Library (COIL-100). In Technical Report, Department of Computer Science, Columbia University CUCS-006-96, Feb 1996.   
V. P. Oikonomou, K. Geordiadis, F. P. Kalaganis, S. Nikolopoulos, and I. Kompatsiaris. Prediction of successful memory formation during audiovisual advertising using eeg signals. In 2024 IEEE Conference on Artificial Intelligence (CAI), pages 1111–1116, 2024. doi: 10.1109/CAI59869.2024. 00200.   
V. P. Oikonomou, K. Georgiadis, I. Lazarou, S. Nikolopoulos, I. Kompatsiaris, and P. Consortium. Exploring functional brain networks in alzheimer’s disease using resting state eeg signals. Journal of Dementia and Alzheimer”s Disease, 2(2), 2025. ISSN 3042-4518. doi: 10.3390/jdad2020012. URL https://www.mdpi.com/3042-4518/2/2/12.   
V. Papyan, X. Y. Han, and D. L. Donoho. Prevalence of neural collapse during the terminal phase of deep learning training. Proceedings of the National Academy of Sciences, 117(40):24652–24663, 2020. doi: 10.1073/pnas.2015509117. URL https://www.pnas.org/doi/abs/10.1073/pnas. 2015509117.   
X. Peng, J. Feng, J. T. Zhou, Y. Lei, and S. Yan. Deep subspace clustering. IEEE Transactions on Neural Networks and Learning Systems, 31(12):5509–5521, 2020. ISSN 2162-237X, 2162-2388. doi: 10.1109/TNNLS.2020.2968848. URL https://ieeexplore.ieee.org/document/9000790/.   
O. Roy and M. Vetterli. The effective rank: A measure of effective dimensionality. In 2007 15th European Signal Processing Conference, pages 606–610, 2007.   
R. Rubinstein, A. M. Bruckstein, and M. Elad. Dictionaries for sparse representation modeling. Proceedings of the IEEE, 98(6):1045–1057, 2010. ISSN 0018-9219, 1558-2256. doi: 10.1109/ JPROC.2010.2040551. URL http://ieeexplore.ieee.org/document/5452966/.   
L. K. Saul. A geometrical connection between sparse and low-rank matrices and its application to manifold learning. Transactions on Machine Learning Research, 2022. ISSN 2835-8856.   
L. Scharf and B. Friedlander. Matched subspace detectors. IEEE Transactions on Signal Processing, 42(8):2146–2157, 1994. doi: 10.1109/78.301849.   
M. Soltanolkotabi and E. J. Candés. A geometric analysis of subspace clustering with outliers. The Annals of Statistics, 40(4), 2012. ISSN 0090-5364. doi: 10.1214/12-AOS1034. URL https://projecteuclid.org/journals/annals-of-statistics/volume-40/issue-4/ A-geometric-analysis-of-subspace-clustering-with-outliers/10.1214/12-AOS1034. full.   
J. A. Tropp and A. C. Gilbert. Signal recovery from random measurements via orthogonal matching pursuit. IEEE Transactions on Information Theory, 53(12):4655–4666, 2007. ISSN 0018-9448. doi: 10.1109/TIT.2007.909108. URL http://ieeexplore.ieee.org/document/4385788/.

T. Wang and P. Isola. Understanding contrastive representation learning through alignment and uniformity on the hypersphere. In Proceedings of the 37th International Conference on Machine Learning, ICML 2020, 13-18 July 2020, Virtual Event, volume 119 of Proceedings of Machine Learning Research, pages 9929–9939. PMLR, 2020. URL http://proceedings.mlr.press/v119/ wang20k.html.   
J. Wright, A. Yang, A. Ganesh, S. Sastry, and Y. Ma. Robust face recognition via sparse representation. IEEE Transactions on Pattern Analysis and Machine Intelligence, 31(2):210–227, 2009. ISSN 0162-8828. doi: 10.1109/TPAMI.2008.79. URL http://ieeexplore.ieee.org/document/ 4483511/.   
J.-Y. Wu, L.-C. Huang, W. H. Li, C.-H. Liu, and R.-H. Gau. Greedier is better: Selecting multiple neighbors per iteration for sparse subspace clustering. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=djD8IbSvgm.   
L. Zhang, M. Yang, X. Feng, Y. Ma, and D. Zhang. Collaborative representation based classification for face recognition, 2014. URL https://arxiv.org/abs/1204.2358.   
C. Zhao, C.-G. Li, W. He, and C. You. Deep self-expressive learning. In Y. Chi, G. K. Dziugaite, Q. Qu, A. W. Wang, and Z. Zhu, editors, Conference on Parsimony and Learning, volume 234 of Proceedings of Machine Learning Research, pages 228–247. PMLR, 03–06 Jan 2024.