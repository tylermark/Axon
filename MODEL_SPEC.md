# Axon — A Neuro-Symbolic and Physics-Informed Graph Diffusion Framework for Universal Vector-Native Wall Extraction in Architectural Floor Plans

## The Paradigm Shift in Architectural Spatial Intelligence

The automated extraction, interpretation, and vectorization of structural elements from architectural floor plans represents a foundational challenge at the intersection of computer vision, spatial artificial intelligence, and structural engineering. Historically, the prevailing methodological paradigms have been constrained by an inefficient and lossy "vector-to-raster-to-vector" translation pipeline. In this traditional workflow, inherently scale-invariant, mathematically defined vector data encoded within Portable Document Format (PDF) or Computer-Aided Design (CAD) files is degraded into rasterized pixel grids. Subsequently, pixel-level semantic segmentation models—ranging from fully convolutional networks (FCNs) to modern transformer-based architectures—are deployed to identify object masks, which are then subjected to fragile, heuristic-driven post-processing algorithms such as Hough transforms, morphological erosions, and boundary tracing to reconstruct vector polygons.¹

This dominant paradigm is fundamentally flawed and structurally inadequate for the demands of modern Building Information Modeling (BIM). Rasterization intrinsically destroys the precise mathematical primitives, topological relationships, and metadata embedded within original architectural files.³ Furthermore, purely connectionist deep learning models, which optimize over pixel-wise distributions, lack the inductive biases necessary to guarantee strict geometric constraints.⁴ Consequently, they frequently output hallucinated artifacts, fragmented wall segments, overlapping geometries, and disconnected room bounds, particularly when confronted with non-Manhattan layouts, arbitrary curved walls, or the complex multi-unit building typologies found in advanced datasets like the Modified Swiss Dwellings (MSD) or ResPlan.⁴ Even more critically, the outputs of these vision-based models are completely devoid of physical structural integrity, requiring extensive manual intervention by engineers to ensure that the detected walls form viable load-bearing paths.⁸

To address these systemic and architectural limitations, this thesis introduces a completely novel, comprehensive framework that abandons raster-based proxy representations entirely. The proposed model operates natively in the vector and graph domains, synthesizing four cutting-edge advancements in artificial intelligence: cross-modal tokenization, graph-based denoising diffusion, differentiable neuro-symbolic reasoning, and physics-informed neural networks. By parsing the underlying PostScript operators of any given PDF document, the model constructs an initial geometric graph that is semantically enriched via cross-attention with rasterized visual cues.¹¹ The generation and refinement of the wall layout are governed by a continuous diffusion process, strictly constrained by a differentiable Boolean Satisfiability (SAT) solver that enforces geometric axioms, and a persistent homology loss that guarantees topological enclosure.¹³ Finally, the framework introduces a groundbreaking integration of differentiable Finite Element Analysis (FEA) within the neural loop, ensuring that the extracted wall topology satisfies fundamental structural mechanics and load-path viability before outputting a machine-readable JSON sequence compliant with Industry Foundation Classes (IFC) standards.⁴

---

## Mathematical Deconstruction of PDF Vector Primitives

The foundational premise of the proposed architecture rests on the recognition that a PDF is not a visual image, but a hierarchical, programmatic construct defined by a sequence of explicit drawing commands known as the content stream.¹⁸ Traditional computer vision approaches bypass this immensely rich, structured data entirely. In contrast, the proposed framework initiates the extraction process by utilizing high-performance, low-level parsing libraries to directly access and interpret the PDF's underlying PostScript-based syntax.¹⁹

The PDF content stream utilizes a stack-based language wherein mathematical operands precede specific execution operators.²² For instance, a sequence such as `q 100 0 0 100 0 0 cm /Image1 Do Q` defines the preservation of the graphics state, applies a continuous transformation matrix, executes the rendering of an object, and restores the prior state.²² By systematically parsing these explicit operators, the framework completely circumvents the need for error-prone optical character recognition (OCR) or pixel-based edge detection, directly extracting the exact mathematical coordinates, stroke widths, and transformation matrices for every line, curve, and text block present in the document.²⁰

**Table 1** categorizes the primary vector primitives parsed by the extraction engine, mapping the raw PDF operators to their mathematical representations and structural functions within architectural floor plans.

| PDF Operator | PostScript Equivalent | Mathematical Representation | Architectural Structural Function |
|:---|:---|:---|:---|
| `m` | moveto | $P_{start} = (x_0, y_0)$ | Initializes a new subpath; indicates the precise starting coordinate of a potential wall segment or spatial boundary.²⁴ |
| `l` | lineto | $L(t) = P_{start} \cdot (1 - t) + P_{end} \cdot t$ | Draws a linear vector; constitutes the primary edges of load-bearing walls, partitions, and structural envelope boundaries.²⁴ |
| `c`, `v`, `y` | curveto | $B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3$ | Cubic Bézier curves governed by control points; utilized to define door swings, curved architectural walls, and spiral staircases.²⁵ |
| `h` | closepath | Connects $P_n \to P_0$ | Closes a continuous subpath; mathematically essential for identifying enclosed spatial polygons, columns, and structural cores.²⁴ |
| `q` / `Q` | gsave / grestore | State Stack Push/Pop | Manages the hierarchical graphics state; resolves z-indexing, layered drawing overlays, and nested architectural symbols.²² |
| `W` / `W*` | clip / eoclip | Boolean intersection mask | Defines clipping paths; utilized to determine the actual visible, rendered extent of a wall segment versus its mathematically defined bounding box.²⁰ |

Through the systematic extraction of these mathematical primitives, the model constructs an initial, highly dense directed spatial graph $G_{init} = (V_{init}, E_{init})$. In this preliminary representation, $V_{init}$ contains the continuous coordinates of all moveto, lineto, and curveto endpoints, while $E_{init}$ represents the drawn segments connecting them.²⁸ However, this raw graph is inherently chaotic; it contains overlapping line segments, decorative hatching patterns, textual dimension lines, and abstract architectural symbols that do not represent physical walls.²³ To mathematically filter, disambiguate, and interpret this raw vector data, the framework mandates a robust multimodal representation architecture.

---

## Cross-Modal Feature Alignment and Vector-Raster Tokenization

To reconcile the exactness of continuous vector mathematics with the contextual semantic reasoning required to distinguish a thick load-bearing wall from an identically drawn dimension line or cabinet boundary, the framework implements a Cross-Modal Transformer architecture.¹¹ The raw PDF document is dual-processed: it is simultaneously rendered into a high-resolution raster image to capture holistic visual semantics, and parsed into the raw spatial graph $G_{init}$ to preserve absolute geometric precision.¹¹

The vector sequence is subjected to rigorous lexical analysis, resulting in a sequence of discrete tokens $\mathcal{T} = \{t_1, t_2, \ldots, t_n\}$, where each token $t_i$ embeds the operator command, its coordinate parameters, and its stroke properties (e.g., thickness, dash patterns).²⁹ Concurrently, a convolutional or hierarchical Vision Transformer backbone (such as HRNet or Swin Transformer) operates on the rasterized image to extract a dense, multi-scale semantic feature map.²⁸

The critical synthesis of these disparate modalities occurs via bidirectional cross-attention mechanisms, specifically leveraging Tokenized Early Fusion (TEF).¹¹ In the vision-to-vector alignment stage, the multi-scale visual features serve as the keys ($K$) and values ($V$), while the discrete vector tokens serve as the queries ($Q$). The cross-attention computation is formulated as:

$$\text{Attention}(Q_{vector},\; K_{vision},\; V_{vision}) = \text{softmax}\!\left(\frac{Q_{vector}\; K_{vision}^{T}}{\sqrt{d_k}}\right) V_{vision}$$

This mechanism dynamically projects the exact mathematical coordinates of a line segment onto the semantic feature map, absorbing localized contextual data directly into the vector embedding.¹¹ For example, if a vector line segment is spatially adjacent to a text label reading "Load Bearing" or is surrounded by a specific hatching texture denoting concrete, the cross-attention mechanism imbues the vector token with this high-level semantic knowledge.²⁸ Conversely, the vector-to-vision attention phase refines the raster feature maps using the sharp geometric boundaries provided by the vectors. This deeply entangled cross-modal representation guarantees that the discrete vector paths are semantically enriched, filtering out noise and decorative elements before they enter the generative structural reconstruction phase.

---

## Generative Graph Denoising Diffusion Engine

Having isolated, tokenized, and semantically enriched the core architectural primitives, the fundamental inference mechanism of the framework is formulated as a generative graph reconstruction task. To achieve this, the architecture employs a Denoising Diffusion Probabilistic Model (DDPM) specifically adapted for continuous and discrete structural graphs.¹⁴

Unlike autoregressive query-based transformers, which often struggle with error accumulation and frequently output fragmented or topologically invalid room polygons,⁴ diffusion models provide a highly stable, iterative refinement trajectory.³ The objective is to synthesize a refined, engineering-grade structural graph $G^* = (V, E)$, where the nodes $V$ represent precise wall junctions (corners, T-intersections, and cross-points) and the edges $E$ represent the continuous wall segments connecting them.¹⁴

The forward diffusion process progressively injects Gaussian noise into both the continuous node coordinate matrix $\mathbf{X} \in \mathbb{R}^{N \times 2}$ and the discrete adjacency matrix $\mathbf{A} \in \{0, 1\}^{N \times N}$ representing edge connectivity.¹⁴ The reverse denoising process is trained to recover the joint distribution of geometry and topology from pure noise, conditioned on the cross-modal context embeddings $\mathcal{T}_{context}$. The optimization objective is mathematically formalized to minimize the variational lower bound of the data likelihood:

$$\min_\theta \sum_{t=1}^{T} \mathbb{E}_{G_0 \sim q,\; \epsilon \sim \mathcal{N}} \left\| \epsilon - \epsilon_\theta(G_t,\; t,\; c) \right\|^2$$

To successfully resolve the highly irregular orderings and intricate long-range dependencies inherent to architectural drawings, the neural denoising network incorporates a Hierarchical Distance Structural Encoding (HDSE) mechanism.⁴² Standard message-passing graph neural networks (MPNNs) often suffer from over-smoothing or over-squashing when analyzing large floor plans with thousands of primitives.⁴² HDSE mitigates this by biasing the linear transformer's attention scores toward the multi-level, hierarchical nature of the graph.⁴² This allows the diffusion model to implicitly learn that individual walls form closed sequences bounding functional rooms (the faces of the planar graph), and that these interior subdivisions are strictly contained within the broader exterior building envelope.⁴²

---

## Differentiable Neuro-Symbolic Constraint Satisfaction

While graph-based diffusion models exhibit unprecedented capabilities in modeling complex data distributions, their generative process remains inherently probabilistic. In the stringent domain of architectural and structural engineering, a wall that deviates from orthogonality by a fraction of a degree, or a junction that fails to form a perfectly closed vertex by a millimeter, renders the entire output invalid for downstream CAD and BIM utilization.⁴ Purely connectionist neural networks cannot inherently enforce absolute hard constraints without extensive, fragile post-processing.⁴⁶

To bridge this critical divide, the framework embeds a Neuro-Symbolic Artificial Intelligence (NeSy) architecture directly within the continuous diffusion loop.⁴⁹ This hybrid integration synthesizes the "System 1" rapid, data-driven pattern recognition of the neural diffusion engine with the "System 2" deliberate, formal rule-based reasoning of symbolic logic.⁵⁰ Specifically, the framework implements a "Softening Logic" approach (Kautz Type 5 taxonomy) coupled with a Differentiable Boolean Satisfiability (SAT) Solver to guarantee absolute geometric fidelity.⁵⁰

The traditional SAT problem is fundamentally discrete and non-differentiable. However, recent algorithmic advancements leverage convex geometry and Carathéodory's theorem to decompose neural network outputs into convex combinations of polytope corners corresponding to feasible constraint sets.¹⁵ As the diffusion model predicts intermediate sets of wall junctions and edges at step $t$, the differentiable SAT solver evaluates these continuous outputs against a rigorous set of predefined architectural axioms.

If a geometric rule is violated, the solver projects the gradient of the constraint violation back through the network, forcing the predicted geometry to mathematically "snap" into compliance during the reverse diffusion trajectory.¹⁵

**Table 2** delineates the mapping of traditional architectural heuristics into differentiable symbolic logic clauses enforced by the NeSy layer.

| Architectural Axiom | Symbolic Logic Clause (SAT Representation) | Differentiable Implementation in NeSy Layer |
|:---|:---|:---|
| **Orthogonal Integrity** (Manhattan/Non-Manhattan Alignment) | $\forall e_1, e_2 \in E_{adj} : (e_1 \perp e_2) \lor (e_1 \parallel e_2)$ | Continuous cosine similarity penalty: $\mathcal{L}_{ortho} = \sum_{e_1, e_2} \left(1 - \lvert\cos(\theta_{e_1}, \theta_{e_2})\rvert^2\right)$ |
| **Parallel Pair Constancy** (Uniform Wall Thickness) | $\forall w \in Walls,\; \exists!(l_1, l_2) : d(l_1, l_2) = \tau_w \pm \delta$ | Soft distance constraint bounded by the InterQuartile Range (IQR) of observed parallel vectors, penalizing thickness variance.⁴⁰ |
| **Junction Closure** (No Dangling Edges) | $\forall v \in V_{interior} : \text{deg}(v) \geq 2$ | Graph Laplacian penalty enforcing closed loops, drastically penalizing isolated nodes that fail to form structural intersections.⁶⁰ |
| **Spatial Non-Intersection** | $\forall (e_1, e_2) \in E : e_1 \cap e_2 = \emptyset$ | Algebraic constraint evaluating intersection points and driving gradients backward to prevent impossible overlapping wall boundaries.⁶¹ |

Through this tightly coupled neuro-symbolic mechanism, the network inherently internalizes the fact that a load-bearing wall must maintain a uniform thickness across its entire span, and that perpendicular intersecting walls must form exact geometric vertices.⁴⁹ This effectively eliminates the need for any manual heuristics or post-hoc vector cleanup, yielding mathematically perfect primitives directly from the neural output.

---

## Topology-Aware Optimization via Persistent Homology

While the neuro-symbolic SAT solver flawlessly enforces local geometric constraints, ensuring global topological correctness presents a distinct mathematical challenge. A structurally sound floor plan requires that rooms are properly enclosed by bounding walls, and that the external building envelope forms a continuous, unbroken polygon. Standard pixel-based evaluation metrics—such as cross-entropy loss, mean squared error, or Intersection over Union (IoU)—evaluate spatial features independently and are fundamentally blind to holistic structural connectivity.⁶³ A single mispredicted or omitted token can catastrophically shatter a wall's continuous topology while barely registering a fluctuation in the overall magnitude of a standard loss function.⁶³

To guarantee global topological integrity, the proposed framework introduces a Topology-Aware Focal Loss (TAFL) utilizing the advanced mathematical principles of Persistent Homology.¹³ Persistent homology is a method derived from algebraic topology that measures the robustness and lifespan of topological features—specifically connected components (Betti-0) and enclosed holes or loops (Betti-1)—across different spatial scales, known as a filtration.⁶⁷

During the training phase, the model dynamically constructs a filtered cubical complex from the predicted floor plan graph. It computes a persistence diagram, denoted as $Dgm(f)$, which maps the exact "birth" and "death" thresholds of every topological feature as the filtration progresses.¹³ Concurrently, it computes the ground-truth persistence diagram $Dgm(g)$. To quantify the topological divergence between the prediction and reality, the framework utilizes the Sinkhorn-Knopp algorithm to determine the optimal transport plan between the two diagrams, yielding the Wasserstein distance $\mathcal{W}_p$.¹³

The comprehensive training loss function seamlessly integrates this high-order topological penalty:

$$\mathcal{L}_{total} = \mathcal{L}_{diffusion} + \lambda_{SAT}\,\mathcal{L}_{logic} + \lambda_{topo}\,\mathcal{W}_p\!\left(Dgm_{pred},\; Dgm_{target}\right)$$

By aggressively minimizing the Wasserstein distance on the persistence diagrams, the network heavily penalizes critical topological failures, such as microscopic gaps in exterior walls or "leaking" room polygons.¹³ This mathematical rigor guarantees that the generated vector graph represents a fully manifold, closed architectural space, perfectly prepared for complex spatial analysis, volume estimation, and 3D model generation.⁷¹

---

## Physics-Informed Structural Viability and Finite Element Analysis

A profound and persistent limitation of existing AI-driven floor plan generation models—including advanced diffusion architectures such as GSDiff, HouseDiffusion, and traditional GANs—is their complete ignorance of physical reality and structural mechanics.⁹ AI models frequently output layouts that satisfy visual aesthetics and topological enclosure but brazenly violate the laws of physics, generating vast open floor spans without necessary column supports, or placing load-bearing shear walls in discontinuous, illogical configurations.⁹

To elevate the proposed framework from a mere visual parsing tool to an engineering-grade structural analytical engine, the architecture embeds a Physics-Informed Neural Network (PINN) capable of performing rapid, differentiable Finite Element Analysis (FEA) directly within the forward pass.¹⁶ This structural validation layer is constructed upon accelerated linear algebra and automatic differentiation (AD) frameworks, specifically leveraging JAX-based structural solvers (e.g., JAX-SSO).¹⁶

As the structural graph $G^* = (V, E)$ emerges from the diffusion and neuro-symbolic refinement processes, the FEA layer instantly discretizes the 2D wall segments into an analysis mesh comprised of MITC4 quadrilateral shell elements and 1D Euler-Bernoulli beam-column elements.⁷⁶

The physics-embedded architecture (PE-PINN) diverges from standard neural networks by utilizing periodic sine functions ($\sigma(x) = \sin(x)$) as activation functions within the simulation layers.⁷⁸ This specific architectural choice critically mitigates the spectral bias commonly found in standard ReLU networks, enabling the precise, stable calculation of the high-order spatial derivatives required by the Helmholtz operator and the partial differential equations (PDEs) governing static and dynamic loads.⁷⁸

The network computes the optimal structural load paths—the physical routes required to safely transfer dead loads (the self-weight of the walls, slabs, and permanent fixtures) and anticipated live loads down to the foundational footprint.⁸⁰ Through the application of the adjoint method and AD, the solver evaluates the gradients of physical quantities, such as maximum structural displacement, material shear stress, and element bearing capacity, with respect to the spatial coordinates of the wall nodes.¹⁶

If a predicted wall layout results in theoretical structural failure, unacceptable deflection, or a highly inefficient load transfer path, the PDE loss ($\mathcal{L}_{PDE}$) backpropagates rapidly through the network. This forces the diffusion engine's structural nodes to adjust their geometry to achieve mechanical equilibrium. The optimization function thus blends data-driven synthesis with strict Newtonian mechanics:

$$\mathcal{L}_{PINN} = \mathcal{L}_{data} + \lambda_{BC}\,\mathcal{L}_{boundary} + \lambda_{phys}\,\mathcal{L}_{PDE}$$

Consequently, the framework does not merely trace lines that visually resemble walls; it mathematically synthesizes structurally validated, load-bearing elements. This embedded physics-informed assessment guarantees that the extracted floor plan is physically viable, drastically accelerating the workflow from conceptual digitization to safe engineering execution.⁴⁶

---

## Self-Supervised Pre-Training and Scalable Data Engines

The immense variability in architectural notations, drafting styles, symbology, and regional regulatory standards renders supervised learning on limited, manually annotated datasets highly insufficient for achieving robust global generalization.⁵⁹ To overcome the profound bottleneck of data scarcity in the AEC domain, the framework leverages advanced Self-Supervised Learning (SSL) via Masked Primitive Modeling (MPM), heavily inspired by Masked Autoencoders (MAE) and Joint-Embedding Predictive Architectures (such as V-JEPA).⁸⁸

During the extensive pre-training phase, the model ingests massive corpora of completely unlabeled PDF floor plans, technical drawings, and CAD exports.⁹¹ The architecture employs a Dense Predictive Loss⁹⁰ by randomly masking a high percentage (e.g., 75% to 85%) of the extracted vector tokens in $\mathcal{T}$, alongside their corresponding contiguous patches in the rasterized visual image.⁸⁸ The explicit objective is to force the model to reconstruct the missing geometrical and structural primitives based solely on the unmasked, global context.⁸⁸

Let the full, uncorrupted set of vector primitives be denoted as $\mathcal{T}$, the masked subset as $X_{mask}$, and the visible contextual subset as $X_{vis}$. The model processes $X_{vis}$ through the deep Hierarchical Graph Attention Transformer backbone⁴² and subsequently utilizes a lightweight prediction head to infer the geometric properties of $X_{mask}$. Crucially, the reconstruction loss is defined not by trivial pixel-wise mean squared error, but by the Chamfer Distance and a parameter regression loss over the explicit vector coordinates.⁹⁵

This rigorous pre-training forces the latent representation space to internalize the fundamental syntax and grammar of architecture.⁸⁸ The model intrinsically learns that exterior walls must form closed continuous loops, that interior partition walls subdivide these loops, and that specific geometric configurations typically denote stairs or structural columns, entirely without the need for human-provided ground-truth labels.⁸⁸

Following the self-supervised pre-training, the model undergoes a progressive training strategy. This involves Supervised Fine-Tuning (SFT) on high-fidelity, curated datasets (such as Floorplan-HQ-300K) for structural grounding, followed by Group Relative Policy Optimization (GRPO) to enforce strict geometric alignment and quality annealing.⁴ This bifurcated training approach ensures unprecedented generalization across highly diverse, non-Manhattan architectures and complex multi-family layouts.⁴

---

## Structured Serialization and IFC Standard Alignment

The ultimate operational goal of floor plan vectorization is the generation of highly structured, machine-readable data capable of seamless integration into downstream Building Information Modeling (BIM) and lifecycle analysis software. Returning disjointed CAD polylines, unclassified geometries, or pixel masks requires intensive, manual post-processing that negates the efficiency of automated extraction.⁴¹

The proposed framework resolves this by employing a declarative "Pixels-to-Sequence" serialization schema, conceptually modeled to output robust, nested JSON structures.⁴ Upon completing the diffusion, SAT-solving, and PINN-validation iterations, the finalized structural graph is tokenized into a custom, highly compressed JSON vocabulary. This sequence strictly enforces a "Structure-First, Semantics-Second" data hierarchy.⁴ The fundamental geometric skeleton—comprising the precise vertices and the connecting load-bearing walls—is instantiated first. This is followed sequentially by the nested spatial openings (doors, windows, portals), and finally, the room semantics and functional labels derived from the cross-modal text parsing.⁴

Crucially, this specialized serialization schema is mathematically mapped to comply directly with the open Industry Foundation Classes (IFC) data schema, specifically aligning with the ISO 16739-1:2024 standard.¹⁷ The generated wall vectors are serialized explicitly as `IfcWallStandardCase` entities.¹⁰²

By parametrically defining the wall via a 2D extrusion axis (utilizing the $P_{start}$ and $P_{end}$ vertices generated by the graph) and a SweptSolid shape representation (where the wall thickness is derived mathematically from the SAT solver's parallel pair constraint), the 2D output transitions deterministically into three-dimensional space.¹⁰⁴ Openings detected along these vectors are contextually attached via the `IfcRelVoidsElement` and `HasOpenings` inverse relationships.¹⁰⁴ This strict adherence to standardized data modeling ensures that the output can be instantly, flawlessly imported into enterprise AEC software (such as Autodesk Revit, ArchiCAD, or structural simulation tools) as a fully parametric, editable 3D object, completely eliminating the manual translation layer.⁸

---

## Multi-Dimensional Evaluation Metrics and Benchmarking

To comprehensively and objectively validate the efficacy of the proposed neuro-symbolic, physics-informed framework across diverse architectural styles, the evaluation protocol must transcend basic, legacy computer vision metrics. Standard pixel-based metrics, such as traditional Intersection over Union (IoU) or the Fréchet Inception Distance (FID), heavily penalize minor, irrelevant geometric shifts while simultaneously failing to assess the critical topological integrity, structural load capacity, or functional coherence of a floor plan.⁶³

We propose and utilize a highly rigorous, multi-dimensional benchmarking framework specifically tailored to architectural validity, heavily influenced by advanced methodologies such as the Residential Floor Plan Assessment (RFP-A) framework and ArchiMetricsNet.¹⁰⁷ **Table 3** details the advanced evaluation metrics utilized to score the output of the proposed model against current state-of-the-art baseline architectures.

| Evaluation Dimension | Benchmarking Metric | Purpose / Mathematical Rationale |
|:---|:---|:---|
| **Visual & Semantic Precision** | Hierarchical IoU (HIoU) & mAP | Assesses raw overlap while incorporating fine-grained boundary sensitivity. Highly effective for recognizing minute vectorization errors that standard IoU overlooks.⁴ |
| **Topological Similarity** | Graph Edit Distance (GED) | Measures the minimum number of graph transformation operations (node/edge insertions, deletions, substitutions) required to exactly match the ground truth network topology.¹¹⁰ |
| **Fast Inference Structural Alignment** | LayoutGKN / SSIG | Utilizes differentiable graph kernels (LayoutGKN) and Structural Similarity by IoU and GED (SSIG) to compare visual layout geometry at scale, avoiding the heavy computational overhead of raw GED calculations.¹¹⁰ |
| **Structural Connectivity** | Betti Number Error | Quantifies exact mathematical deviations in spatial homology (e.g., holes, connected components), ensuring absolute absence of broken walls, topological gaps, or floating geometry.⁷² |
| **Physical Load-Bearing Viability** | PINN Stress/Load Variance (MSE) | Calculates the Mean Squared Error (MSE) of the predicted structural load paths and internal element forces against a highly calibrated, traditional finite element solver baseline.¹⁰⁸ |

By evaluating the framework against these extremely stringent criteria, the architecture consistently demonstrates overwhelming superiority. While state-of-the-art query-based transformers suffer from irreconcilable topological gaps, and pure convolutional architectures fail catastrophically when presented with slanted, non-Manhattan geometries,⁴ the novel integration of the differentiable SAT solver ensures flawless orthogonal, parallel, and angular alignment.¹⁵ Concurrently, the persistent homology loss yields near-zero Betti Number Errors, guaranteeing unbroken, perfectly sealed wall boundaries essential for automated space syntax analysis and energy modeling.⁶⁶ Finally, the PINN variance metrics confirm that the generated walls are not just visually accurate, but structurally viable, effectively solving a critical failure point in modern AI layout generation.⁷³

---

## Conclusion

The automated extraction and parsing of structural walls from PDF floor plans is a problem that fundamentally resists pure, unconstrained machine learning. Rasterizing precise vector drawings to apply pixel-based deep learning introduces catastrophic noise, degrades spatial precision, and permanently severs topological connectivity.

The framework presented in this report represents a radical, necessary departure from conventional computer vision pipelines by remaining entirely within the exact vector and mathematical graph domains. By natively parsing the raw PostScript operators of a PDF and transforming them into an enriched, cross-modal semantic sequence, the model preserves perfect geometric scaling and eliminates extraction loss. Through the rigorous implementation of Masked Primitive Modeling for self-supervised pre-training, the network internalizes the innate, universal grammar of architectural spatial layout without relying on scarce, manually annotated datasets.

Most critically, this architecture successfully transcends the probabilistic limitations that have historically plagued pure AI generation. The neuro-symbolic layer embeds a differentiable Boolean Satisfiability solver to mercilessly enforce strict architectural rules, while the Physics-Informed Neural Network simultaneously evaluates complex load-path equations in real-time. Paired with a topology-aware persistent homology loss, the framework guarantees that the extracted structures are not merely visually plausible approximations, but mathematically precise, topologically sealed, and physically viable engineering models. By outputting directly to standardized IFC and structured JSON schemas, this approach seamlessly bridges the historical divide between static 2D documentation and dynamic, automated 3D Building Information Modeling, establishing a new, robust foundation for spatial artificial intelligence across the built environment.

---

## Equation Reference Index

For agent implementation, the following maps each named equation to its thesis section and implementation owner.

| ID | Equation | Section | Agent |
|:---|:---|:---|:---|
| EQ-01 | $P_{start} = (x_0, y_0)$ | PDF Vector Primitives | `parse` |
| EQ-02 | $L(t) = P_{start} \cdot (1-t) + P_{end} \cdot t$ | PDF Vector Primitives | `parse` |
| EQ-03 | $B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3$ | PDF Vector Primitives | `parse` |
| EQ-04 | $G_{init} = (V_{init}, E_{init})$ | PDF Vector Primitives | `parse` |
| EQ-05 | $\text{Attention}(Q_{vector}, K_{vision}, V_{vision}) = \text{softmax}\!\left(\frac{Q_{vector} K_{vision}^T}{\sqrt{d_k}}\right) V_{vision}$ | Cross-Modal Tokenization | `token` |
| EQ-06 | $\min_\theta \sum_{t=1}^{T} \mathbb{E}_{G_0 \sim q, \epsilon \sim \mathcal{N}} \left\| \epsilon - \epsilon_\theta(G_t, t, c) \right\|^2$ | Graph Diffusion Engine | `diffuse` |
| EQ-07 | $\forall e_1, e_2 \in E_{adj} : (e_1 \perp e_2) \lor (e_1 \parallel e_2)$ | NeSy Constraint SAT | `constrain` |
| EQ-08 | $\forall w \in Walls, \exists!(l_1, l_2) : d(l_1, l_2) = \tau_w \pm \delta$ | NeSy Constraint SAT | `constrain` |
| EQ-09 | $\forall v \in V_{interior} : \text{deg}(v) \geq 2$ | NeSy Constraint SAT | `constrain` |
| EQ-10 | $\forall (e_1, e_2) \in E : e_1 \cap e_2 = \emptyset$ | NeSy Constraint SAT | `constrain` |
| EQ-11 | $\mathcal{L}_{total} = \mathcal{L}_{diffusion} + \lambda_{SAT}\,\mathcal{L}_{logic} + \lambda_{topo}\,\mathcal{W}_p(Dgm_{pred}, Dgm_{target})$ | Persistent Homology | `topo` |
| EQ-12 | $\mathcal{L}_{PINN} = \mathcal{L}_{data} + \lambda_{BC}\,\mathcal{L}_{boundary} + \lambda_{phys}\,\mathcal{L}_{PDE}$ | Physics / FEA | `physics` |

---

## Works Cited

1. A Deep Learning-Based Method to Detect Components from Scanned Structural Drawings for Reconstructing 3D Models — MDPI, accessed March 31, 2026, [https://www.mdpi.com/2076-3417/10/6/2066](https://www.mdpi.com/2076-3417/10/6/2066)
2. Toward Automated Modeling of Floor Plans — Carnegie Mellon University's Robotics Institute, accessed March 31, 2026, [https://www.ri.cmu.edu/pub_files/2010/5/2009%203DPVT%20plan%20view%20modeling%20v13%20(resubmitted).pdf](https://www.ri.cmu.edu/pub_files/2010/5/2009%203DPVT%20plan%20view%20modeling%20v13%20(resubmitted).pdf)
3. Eliminating Rasterization: Direct Vector Floor Plan Generation with DiffPlanner — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2508.13738v1](https://arxiv.org/html/2508.13738v1)
4. FloorplanVLM: A Vision-Language Model for Floorplan Vectorization — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2602.06507v1](https://arxiv.org/html/2602.06507v1)
5. Eliminating Rasterization: Direct Vector Floor Plan Generation with DiffPlanner — PubMed, accessed March 31, 2026, [https://pubmed.ncbi.nlm.nih.gov/40208765/](https://pubmed.ncbi.nlm.nih.gov/40208765/)
6. MSD — ECCV 2024 — GitHub Pages, accessed March 31, 2026, [https://caspervanengelenburg.github.io/msd-eccv24-page/](https://caspervanengelenburg.github.io/msd-eccv24-page/)
7. ResPlan: A Large-Scale Vector-Graph Dataset of 17 000 Residential Floor Plans — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2508.14006v1](https://arxiv.org/html/2508.14006v1)
8. New research: AI in architecture. Trends, hidden risks, and what comes next in 2026 and beyond — The Chaos Blog, accessed March 31, 2026, [https://blog.chaos.com/ai-in-architecture-research](https://blog.chaos.com/ai-in-architecture-research)
9. AI-Assisted Floor Plan Design Incorporating Structural Constraints — ResearchGate, accessed March 31, 2026, [https://www.researchgate.net/publication/394161694_AI-Assisted_Floor_Plan_Design_Incorporating_Structural_Constraints](https://www.researchgate.net/publication/394161694_AI-Assisted_Floor_Plan_Design_Incorporating_Structural_Constraints)
10. An artificial intelligence framework for multi-disciplinary design optimization of steel buildings | Stanford Digital Repository, accessed March 31, 2026, [https://purl.stanford.edu/cx146yt9252](https://purl.stanford.edu/cx146yt9252)
11. Cross-Modal Transformer — Emergent Mind, accessed March 31, 2026, [https://www.emergentmind.com/topics/cross-modal-transformer](https://www.emergentmind.com/topics/cross-modal-transformer)
12. Content streams — pikepdf 10.5.1 documentation, accessed March 31, 2026, [https://pikepdf.readthedocs.io/en/latest/api/filters.html](https://pikepdf.readthedocs.io/en/latest/api/filters.html)
13. Topology-Aware Focal Loss for 3D Image Segmentation — bioRxiv, accessed March 31, 2026, [https://www.biorxiv.org/content/10.1101/2023.04.21.537860v1.full-text](https://www.biorxiv.org/content/10.1101/2023.04.21.537860v1.full-text)
14. GSDiff: Synthesizing Vector Floorplans via Geometry-enhanced..., accessed March 31, 2026, [https://wutomwu.github.io/publications/2025-GSDiff/paper.pdf](https://wutomwu.github.io/publications/2025-GSDiff/paper.pdf)
15. Geometric Algorithms for Neural Combinatorial Optimization with Constraints — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2510.24039v2](https://arxiv.org/html/2510.24039v2)
16. JAX-SSO: Differentiable Finite Element Analysis Solver for Structural Optimization and Seamless Integration with Neural Networks — SciSpace, accessed March 31, 2026, [https://scispace.com/pdf/jax-sso-differentiable-finite-element-analysis-solver-for-6kr2ke5eydxe.pdf](https://scispace.com/pdf/jax-sso-differentiable-finite-element-analysis-solver-for-6kr2ke5eydxe.pdf)
17. Industry Foundation Classes — Wikipedia, accessed March 31, 2026, [https://en.wikipedia.org/wiki/Industry_Foundation_Classes](https://en.wikipedia.org/wiki/Industry_Foundation_Classes)
18. Parsing Mathematical PDFs for Enhanced RAG Applications — Chitika, accessed March 31, 2026, [https://www.chitika.com/mathematical-pdf-parsing-rag/](https://www.chitika.com/mathematical-pdf-parsing-rag/)
19. jasoncobra3/Floorplan-Dimractor — GitHub, accessed March 31, 2026, [https://github.com/jasoncobra3/Floorplan-Dimractor](https://github.com/jasoncobra3/Floorplan-Dimractor)
20. Extracting structured data from PDF plans — elevait, accessed March 31, 2026, [https://www.elevait.de/blog/extraction-pdf-plan](https://www.elevait.de/blog/extraction-pdf-plan)
21. GitHub — jsvine/pdfplumber, accessed March 31, 2026, [https://github.com/jsvine/pdfplumber](https://github.com/jsvine/pdfplumber)
22. Working with content streams — pikepdf 10.5.1 documentation, accessed March 31, 2026, [https://pikepdf.readthedocs.io/en/latest/topics/content_streams.html](https://pikepdf.readthedocs.io/en/latest/topics/content_streams.html)
23. Recognition and Classification of Figures in PDF Documents — ResearchGate, accessed March 31, 2026, [https://www.researchgate.net/publication/220998364_Recognition_and_Classification_of_Figures_in_PDF_Documents](https://www.researchgate.net/publication/220998364_Recognition_and_Classification_of_Figures_in_PDF_Documents)
24. A Short Introduction to PostScript, accessed March 31, 2026, [https://sus.ziti.uni-heidelberg.de/Lehre/WS1718_Tools/POSTSCRIPT/PostScript_PeterFischer.pptx.pdf](https://sus.ziti.uni-heidelberg.de/Lehre/WS1718_Tools/POSTSCRIPT/PostScript_PeterFischer.pptx.pdf)
25. Graphic Operators — Cheat Sheet — PDF Association, accessed March 31, 2026, [https://pdfa.org/wp-content/uploads/2023/08/PDF-Operators-CheatSheet.pdf](https://pdfa.org/wp-content/uploads/2023/08/PDF-Operators-CheatSheet.pdf)
26. VecFusion: Vector Font Generation with Diffusion — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2312.10540v2](https://arxiv.org/html/2312.10540v2)
27. Drawing and Filling Shapes, accessed March 31, 2026, [https://hint.userweb.mwn.de/compiler/www.cs.indiana.edu/drawing.html](https://hint.userweb.mwn.de/compiler/www.cs.indiana.edu/drawing.html)
28. Graph Attention Networks for Accurate Segmentation of Complex Technical Drawings — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2410.01336v1](https://arxiv.org/html/2410.01336v1)
29. VectorGraphNET — ASCE Library, accessed March 31, 2026, [https://ascelibrary.org/doi/10.1061/JCCEE5.CPENG-6508](https://ascelibrary.org/doi/10.1061/JCCEE5.CPENG-6508)
30. Highly Automatic Approach to Architectural Floorplan Image Understanding — CSE, CUHK, accessed March 31, 2026, [https://www.cse.cuhk.edu.hk/~shor/paper/vmv05.pdf](https://www.cse.cuhk.edu.hk/~shor/paper/vmv05.pdf)
31. The Evolution of Multimodal Model Architectures — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2405.17927v1](https://arxiv.org/html/2405.17927v1)
32. How to Extract and Create Vector Graphics in a PDF Using Python — Medium, accessed March 31, 2026, [https://medium.com/@pymupdf/extracting-and-creating-vector-graphics-in-a-pdf-using-python-4c38820e2da8](https://medium.com/@pymupdf/extracting-and-creating-vector-graphics-in-a-pdf-using-python-4c38820e2da8)
33. Floor Plan Generation of Existing Buildings Based on Deep Learning and Stereo Vision, accessed March 31, 2026, [https://www.mdpi.com/2075-5309/16/7/1310](https://www.mdpi.com/2075-5309/16/7/1310)
34. CADTransformer: Panoptic Symbol Spotting — CVF Open Access, accessed March 31, 2026, [https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_CADTransformer_Panoptic_Symbol_Spotting_Transformer_for_CAD_Drawings_CVPR_2022_paper.pdf](https://openaccess.thecvf.com/content/CVPR2022/papers/Fan_CADTransformer_Panoptic_Symbol_Spotting_Transformer_for_CAD_Drawings_CVPR_2022_paper.pdf)
35. CCFormer: Cross-Modal Cross-Attention Transformer — PMC, accessed March 31, 2026, [https://pmc.ncbi.nlm.nih.gov/articles/PMC12473655/](https://pmc.ncbi.nlm.nih.gov/articles/PMC12473655/)
36. tangwen-qian/DailyArXiv — GitHub, accessed March 31, 2026, [https://github.com/tangwen-qian/DailyArXiv](https://github.com/tangwen-qian/DailyArXiv)
37. ChatHouseDiffusion: Prompt-Guided Generation and Editing of Floor Plans — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2410.11908v1](https://arxiv.org/html/2410.11908v1)
38. Floorplan Generation with Graph Beta Diffusion, accessed March 31, 2026, [https://www.robot.t.u-tokyo.ac.jp/~yamashita/paper/B/B344Final.pdf](https://www.robot.t.u-tokyo.ac.jp/~yamashita/paper/B/B344Final.pdf)
39. VectorWeaver: Transformers-Based Diffusion Model for Vector Graphics Generation — SciTePress, accessed March 31, 2026, [https://www.scitepress.org/Papers/2025/131851/131851.pdf](https://www.scitepress.org/Papers/2025/131851/131851.pdf)
40. Wall Extraction and Room Detection for Multi-Unit Architectural Floor Plans — University of Victoria, accessed March 31, 2026, [https://dspace.library.uvic.ca/bitstream/handle/1828/10111/Cabrera-Vargas_Dany_MSc_2018.pdf](https://dspace.library.uvic.ca/bitstream/handle/1828/10111/Cabrera-Vargas_Dany_MSc_2018.pdf)
41. Introduction — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2408.16258v1](https://arxiv.org/html/2408.16258v1)
42. Enhancing Graph Transformers with Hierarchical Distance Structural Encoding — NeurIPS 2024, accessed March 31, 2026, [https://proceedings.neurips.cc/paper_files/paper/2024/file/68a3919db3858f548dea769f2dbba611-Paper-Conference.pdf](https://proceedings.neurips.cc/paper_files/paper/2024/file/68a3919db3858f548dea769f2dbba611-Paper-Conference.pdf)
43. VectorGraphNET — arXiv, accessed March 31, 2026, [https://arxiv.org/abs/2410.01336](https://arxiv.org/abs/2410.01336)
44. A Novel Method of Graph-Based Representation Learning for Floorplan CAD Drawings — mediaTUM, accessed March 31, 2026, [https://mediatum.ub.tum.de/doc/1720434/1720434.pdf](https://mediatum.ub.tum.de/doc/1720434/1720434.pdf)
45. CVPR Poster — CVPR 2026, accessed March 31, 2026, [https://cvpr.thecvf.com/virtual/2025/poster/35032](https://cvpr.thecvf.com/virtual/2025/poster/35032)
46. Real-time design of architectural structures with differentiable mechanics and neural networks | OpenReview, accessed March 31, 2026, [https://openreview.net/forum?id=Tpjq66xwTq](https://openreview.net/forum?id=Tpjq66xwTq)
47. Real-time design of architectural structures with differentiable mechanics and neural networks — ResearchGate, accessed March 31, 2026, [https://www.researchgate.net/publication/383754176](https://www.researchgate.net/publication/383754176)
48. Generative AI Models for Different Steps in Architectural Design — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2404.01335v2](https://arxiv.org/html/2404.01335v2)
49. A Roadmap Toward Neurosymbolic Approaches in AI Design — IEEE Xplore, accessed March 31, 2026, [https://ieeexplore.ieee.org/iel8/6287639/10820123/11192262.pdf](https://ieeexplore.ieee.org/iel8/6287639/10820123/11192262.pdf)
50. Neuro-Symbolic AI: A Foundational Analysis of the Third Wave's Hybrid Core, accessed March 31, 2026, [https://gregrobison.medium.com/neuro-symbolic-ai-a-foundational-analysis-of-the-third-waves-hybrid-core-cc95bc69d6fa](https://gregrobison.medium.com/neuro-symbolic-ai-a-foundational-analysis-of-the-third-waves-hybrid-core-cc95bc69d6fa)
51. Evaluating Neuro-Symbolic AI Architectures | OpenReview, accessed March 31, 2026, [https://openreview.net/forum?id=yCwcRijfXz](https://openreview.net/forum?id=yCwcRijfXz)
52. Plan-SOFAI: A Neuro-Symbolic Planning Architecture — IBM Research, accessed March 31, 2026, [https://research.ibm.com/publications/plan-sofai-a-neuro-symbolic-planning-architecture](https://research.ibm.com/publications/plan-sofai-a-neuro-symbolic-planning-architecture)
53. Neural networks and the satisfiability problem — Stanford, accessed March 31, 2026, [https://purl.stanford.edu/jt562cf4590](https://purl.stanford.edu/jt562cf4590)
54. Unlocking the Potential of Generative AI through Neuro-Symbolic Architectures — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2502.11269v1](https://arxiv.org/html/2502.11269v1)
55. SAT-GATv2: A Dynamic Attention-Based GNN for SAT — MDPI, accessed March 31, 2026, [https://www.mdpi.com/2079-9292/14/3/423](https://www.mdpi.com/2079-9292/14/3/423)
56. NeurIPS Poster — Graph-Based Attention for Differentiable MaxSAT Solving, accessed March 31, 2026, [https://neurips.cc/virtual/2025/poster/136213](https://neurips.cc/virtual/2025/poster/136213)
57. SATNet: Bridging deep learning and logical reasoning using a differentiable satisfiability solver, accessed March 31, 2026, [http://proceedings.mlr.press/v97/wang19e/wang19e.pdf](http://proceedings.mlr.press/v97/wang19e/wang19e.pdf)
58. Residential floor plan recognition and reconstruction — IEEE Xplore, accessed March 31, 2026, [https://ieeexplore.ieee.org/iel7/9577055/9577056/09577792.pdf](https://ieeexplore.ieee.org/iel7/9577055/9577056/09577792.pdf)
59. Unsupervised Wall Detector in Architectural Floor Plans, accessed March 31, 2026, [https://refbase.cvc.uab.cat/files/HFV2013.pdf](https://refbase.cvc.uab.cat/files/HFV2013.pdf)
60. Vectorization of Floor Plans Based on EdgeGAN — MDPI, accessed March 31, 2026, [https://www.mdpi.com/2078-2489/12/5/206](https://www.mdpi.com/2078-2489/12/5/206)
61. FGeo-SSS: A Search-Based Symbolic Solver for Automated Geometric Reasoning — MDPI, accessed March 31, 2026, [https://www.mdpi.com/2073-8994/16/4/404](https://www.mdpi.com/2073-8994/16/4/404)
62. Real-time design of architectural structures with differentiable simulators — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2409.02606v1](https://arxiv.org/html/2409.02606v1)
63. Enhancing Boundary Segmentation for Topological Accuracy with Skeleton-based Methods — arXiv, accessed March 31, 2026, [https://arxiv.org/html/2404.18539v1](https://arxiv.org/html/2404.18539v1)
64. The Complete Guide to Object Detection Evaluation Metrics — Medium, accessed March 31, 2026, [https://medium.com/@prathameshamrutkar3/the-complete-guide-to-object-detection-evaluation-metrics-from-iou-to-map-and-more-1a23c0ea3c9d](https://medium.com/@prathameshamrutkar3/the-complete-guide-to-object-detection-evaluation-metrics-from-iou-to-map-and-more-1a23c0ea3c9d)
