# Axon — A Neuro-Symbolic Graph Diffusion Framework for Universal Vector-Native Wall Extraction and Automated Prefabricated Component Placement

## The Paradigm Shift in Architectural Spatial Intelligence and Modular Construction

The automated extraction, interpretation, and vectorization of structural elements from architectural floor plans represents a foundational challenge. Historically, the prevailing methodological paradigms have been constrained by an inefficient "vector-to-raster-to-vector" translation pipeline, which destroys the precise mathematical primitives embedded within original architectural files. Furthermore, purely connectionist deep learning models lack the inductive biases necessary to guarantee strict geometric constraints, frequently outputting fragmented wall segments or overlapping geometries.

More importantly, existing floor plan extraction models fail to address the ultimate goal of modern industrialized construction: prefabrication and product placement. A simple vector line representing a wall is useless to a modular manufacturer unless it can be translated into a sequence of specific, buildable, prefabricated panels and proprietary products.

To address these systemic limitations, this thesis introduces a completely novel framework that abandons raster-based proxy representations entirely. The proposed model operates natively in the vector and graph domains to parse the underlying PostScript operators of any given PDF document. Once the pure geometric skeleton is extracted, the framework utilizes a Knowledge Graph to ingest a manufacturer's specific product catalog. Finally, a Deep Reinforcement Learning (DRL) engine automatically panelizes the walls and places existing prefabricated products into the layout, ensuring the extracted floor plan is instantly ready for factory production.

---

## Mathematical Deconstruction of PDF Vector Primitives

The foundational premise of the proposed architecture rests on the recognition that a PDF is a hierarchical, programmatic construct defined by a sequence of explicit drawing commands. The proposed framework initiates the extraction process by utilizing high-performance parsing libraries to directly access and interpret the PDF's underlying PostScript-based syntax.

The PDF content stream utilizes a stack-based language wherein mathematical operands precede specific execution operators. By systematically parsing these explicit operators, the framework completely circumvents the need for error-prone optical character recognition (OCR) or pixel-based edge detection, directly extracting the exact mathematical coordinates, stroke widths, and transformation matrices for every line, curve, and text block present in the document.

**Table 1** categorizes the primary vector primitives parsed by the extraction engine, mapping the raw PDF operators to their mathematical representations.

| PDF Operator | PostScript Equivalent | Mathematical Representation | Architectural Function |
|:---|:---|:---|:---|
| `m` | moveto | $P_{start} = (x, y)$ | Initializes a new subpath; indicates the precise starting coordinate of a potential wall segment. |
| `l` | lineto | $L(t) = P_{start} \cdot (1-t) + P_{end} \cdot t$ | Draws a linear vector; constitutes the primary edges of boundaries and partitions. |
| `c`, `v`, `y` | curveto | $B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3$ | Cubic Bézier curves governed by control points; utilized to define door swings and curved walls. |
| `h` | closepath | Connects $P_n \to P_0$ | Closes a continuous subpath; mathematically essential for identifying enclosed spatial polygons. |

Through the systematic extraction of these mathematical primitives, the model constructs an initial, highly dense directed spatial graph $G_{init} = (V_{init}, E_{init})$.

---

## Cross-Modal Feature Alignment and Vector-Raster Tokenization

To reconcile the exactness of continuous vector mathematics with the contextual semantic reasoning required to distinguish a load-bearing wall from an identically drawn cabinet boundary, the framework implements a Cross-Modal Transformer architecture. The raw PDF document is dual-processed: it is simultaneously rendered into a high-resolution raster image to capture holistic visual semantics, and parsed into the raw spatial graph $G_{init}$ to preserve absolute geometric precision.

The vector sequence is subjected to rigorous lexical analysis, resulting in a sequence of discrete tokens $\mathcal{T} = \{t_1, t_2, \ldots, t_n\}$, where each token $t_i$ embeds the operator command, its coordinate parameters, and its stroke properties. Concurrently, a Vision Transformer backbone operates on the rasterized image to extract a dense, multi-scale semantic feature map.

The critical synthesis of these disparate modalities occurs via bidirectional cross-attention mechanisms, specifically leveraging Tokenized Early Fusion (TEF):

$$\text{Attention}(Q_{vector},\; K_{vision},\; V_{vision}) = \text{softmax}\!\left(\frac{Q_{vector}\; K_{vision}^{T}}{\sqrt{d_k}}\right) V_{vision}$$

This deeply entangled cross-modal representation guarantees that the discrete vector paths are semantically enriched, filtering out noise and decorative elements before they enter the generative reconstruction phase.

---

## Generative Graph Denoising Diffusion Engine

Having isolated, tokenized, and semantically enriched the core architectural primitives, the fundamental inference mechanism of the framework is formulated as a generative graph reconstruction task. To achieve this, the architecture employs a Denoising Diffusion Probabilistic Model (DDPM) specifically adapted for continuous and discrete structural graphs.

The objective is to synthesize a refined, clean structural graph $G^* = (V, E)$, where the nodes $V$ represent precise wall junctions and the edges $E$ represent the continuous wall segments connecting them. The forward diffusion process progressively injects Gaussian noise into both the continuous node coordinate matrix $\mathbf{X} \in \mathbb{R}^{N \times 2}$ and the discrete adjacency matrix $\mathbf{A} \in \{0, 1\}^{N \times N}$. The reverse denoising process is trained to recover the joint distribution of geometry and topology from pure noise, conditioned on the cross-modal context embeddings. The optimization objective is mathematically formalized to minimize the variational lower bound of the data likelihood:

$$\min_\theta \sum_{t=1}^{T} \mathbb{E}_{G_0 \sim q,\; \epsilon \sim \mathcal{N}} \left\| \epsilon - \epsilon_\theta(G_t,\; t,\; c) \right\|^2$$

To resolve the highly irregular orderings and intricate long-range dependencies inherent to architectural drawings, the neural denoising network incorporates a Hierarchical Distance Structural Encoding (HDSE) mechanism, which biases the linear transformer's attention scores toward the multi-level, hierarchical nature of the graph. This allows the diffusion model to implicitly learn that individual walls form closed sequences bounding functional rooms, and that these interior subdivisions are strictly contained within the broader exterior building envelope.

---

## Differentiable Neuro-Symbolic Geometric Constraints

While graph-based diffusion models exhibit unprecedented capabilities in modeling complex data distributions, their generative process remains inherently probabilistic. In the strict domain of modular construction, a wall that deviates from orthogonality prevents standard panels from fitting.

To bridge this divide, the framework embeds a Neuro-Symbolic Artificial Intelligence (NeSy) architecture directly within the continuous diffusion loop. Specifically, the framework implements a Differentiable Boolean Satisfiability (SAT) Solver to guarantee that the extracted geometry conforms to strict architectural axioms.

**Table 2** delineates the geometric axioms enforced by the NeSy layer during diffusion.

| Architectural Axiom | Symbolic Logic Clause | Differentiable Implementation |
|:---|:---|:---|
| **Orthogonal Integrity** | $\forall e_1, e_2 \in E_{adj} : (e_1 \perp e_2) \lor (e_1 \parallel e_2)$ | Continuous cosine similarity penalty: $\mathcal{L}_{ortho}$ |
| **Parallel Pair Constancy** (Uniform Wall Thickness) | $\forall w \in Walls,\; \exists!(l_1, l_2) : d(l_1, l_2) = \tau_w \pm \delta$ | Soft distance constraint bounded by IQR of observed parallel vectors |
| **Junction Closure** (No Dangling Edges) | $\forall v \in V_{interior} : \text{deg}(v) \geq 2$ | Graph Laplacian penalty enforcing closed loops |
| **Spatial Non-Intersection** | $\forall (e_1, e_2) \in E : e_1 \cap e_2 = \emptyset$ | Algebraic intersection constraint with backward gradients |

If a geometric rule is violated, the solver projects the gradient of the constraint violation back through the network, forcing the predicted geometry to mathematically "snap" into compliance during the reverse diffusion trajectory. This guarantees clean, orthogonal geometry suitable for downstream panelization.

The composite training loss integrates the diffusion objective with the constraint penalty:

$$\mathcal{L}_{total} = \mathcal{L}_{diffusion} + \lambda_{SAT}\,\mathcal{L}_{constraints}$$

A lightweight topological regularization term is additionally applied during training to ensure room enclosure (walls form closed polygons), using a simplified Betti number check on the predicted graph connectivity.

---

## Knowledge Graph-Driven Product Catalog Integration

For an AI to successfully populate a floor plan with proprietary prefabricated components, it must possess a deep, deterministic understanding of the manufacturer's specific product catalog and assembly rules. Standard neural networks struggle to memorize explicit dimensional constraints and inventory lists.

To resolve this, the framework integrates a multi-dimensional Knowledge Graph (KG) that acts as a structured, queryable database containing the exact specifications of every prefabricated product available:

- **CFS Panel Types** — gauge, stud depth, stud spacing, max fabrication length, fire rating, load capacity
- **Pod Assemblies** — bathroom pods, kitchen pods, MEP pods with dimensions, included trades, connection types
- **Machine Capabilities** — Howick 2.5, Howick 3.5, Zund capacity limits, tolerances, speed
- **Connection Details** — track splices, clip angles, bridging, blocking, fastener schedules
- **Regulatory Compliance** — AISI S100/S240, fire separation requirements, accessibility standards

When the AI extracts the geometric skeleton of the floor plan, it queries this Knowledge Graph to filter and retrieve only the valid structural panels, door frames, and modular units that physically fit the extracted dimensions. This ensures that every design decision and product placement is strictly grounded in real-world manufacturing capabilities and current inventory.

---

## Reinforcement Learning for Automated Panelization and Placement

Once the valid prefabricated components are retrieved from the Knowledge Graph, the model must solve a complex combinatorial optimization problem: how to best tile, panelize, and populate the extracted floor plan using discrete, standardized parts. This is achieved using a Deep Reinforcement Learning (DRL) agent.

### Wall Panelization

The automated panelization process divides the continuous wall vectors into discrete prefabricated panels. For each classified wall segment, the DRL agent:

1. Queries the KG for compatible panel types (matching wall classification, gauge, fire rating)
2. Explores segmentation strategies that maximize standard-length panel usage
3. Handles constraints: panels cannot obstruct extracted openings (doors, windows), must accommodate angled joints, and must respect machine fabrication limits

### Product Placement

For room-level placement, the DRL agent:

1. Analyzes room geometry, function labels, and dimensional data from the structural graph
2. Queries the KG for compatible pod assemblies and products
3. Optimizes placement: maximize coverage while respecting clearances, code setbacks, and MEP alignment

### Reward Function

The DRL agent receives:
- **Positive rewards** for maximizing standard-sized component usage (reduces custom manufacturing cost), achieving high catalog match rates, and minimizing material waste
- **Negative penalties** for spatial overlaps, structural gaps, opening obstructions, code violations, and logic errors

By simulating the production planning process across thousands of iterations, this reinforcement learning approach derives the optimal configuration of prefabricated components, drastically reducing material waste and streamlining the manufacturing timeline.

---

## BIM Library Transplant and IFC Serialization

The final output of the framework is not merely a 2D vector drawing, but a fully realized, manufacture-ready 3D model composed of the company's actual products. To bridge the gap between the optimized 2D layout and the company's existing 3D assets, the framework employs a "BIM Library Transplant" methodology.

Upon completing panelization and placement, each designated location in the DRL-optimized layout is mapped to the corresponding high-Level of Development (LOD) object from the company's donor BIM library via deterministic KG lookup. The placed 2D panel slots are matched to their exact 3D prefabricated models—including panel seams, hardware, and product SKUs—by querying the Knowledge Graph for the BIM family that matches the panel type, gauge, length, and fire rating.

The fully populated model is serialized into an Industry Foundation Classes (IFC) compliant schema, explicitly mapping placed products to `IfcWallStandardCase`, `IfcProduct`, and `IfcRelVoidsElement` entities per ISO 16739-1:2024. The output can be instantly imported into Autodesk Revit or ArchiCAD for immediate factory production planning.

---

## Multi-Dimensional Evaluation Metrics

**Table 3** details the evaluation metrics utilized to score the output of the proposed model.

| Evaluation Dimension | Benchmarking Metric | Purpose |
|:---|:---|:---|
| **Visual & Semantic Precision** | Hierarchical IoU (HIoU) & mAP | Assesses raw extraction overlap with fine-grained boundary sensitivity. |
| **Topological Correctness** | Graph Edit Distance (GED) & Betti Number Error | Measures structural graph accuracy and room enclosure integrity. |
| **Panelization Efficiency** | Standard Part Utilization Ratio (SPUR) | Percentage of standard catalog components used vs custom/cut components. |
| **Catalog Match Accuracy** | KG Lookup Precision & BIM Transplant Success Rate | Accuracy of matching extracted geometry to correct KG products and 3D BIM families. |
| **Spatial Optimization** | DRL Reward Convergence & Waste Ratio | Evaluates layout optimization, gap minimization, and material waste reduction. |
| **Prefab Feasibility** | Prefab Coverage Percentage | Fraction of building (by wall length, area, cost) achievable with catalog components. |

---

## Conclusion

The automated extraction and parsing of floor plans is only the first step in the modern construction pipeline; the true value lies in translating those plans into buildable, prefabricated reality.

The framework presented in this report achieves this by natively parsing the raw PostScript operators of a PDF to preserve perfect geometric scaling. It then transcends basic extraction by utilizing a Neuro-Symbolic layer and Knowledge Graphs to enforce geometric constraints and ground every decision in real manufacturing capabilities. Deep Reinforcement Learning automatically panelizes the walls and populates the space with discrete components from a proprietary product catalog. By automatically transplanting high-LOD BIM models into the extracted skeleton, this approach seamlessly bridges the gap between static 2D client PDFs and dynamic, automated factory prefabrication.

---

## Equation Reference Index

| ID | Equation | Section | Agent |
|:---|:---|:---|:---|
| EQ-01 | $P_{start} = (x, y)$ | PDF Vector Primitives | `parse` |
| EQ-02 | $L(t) = P_{start} \cdot (1-t) + P_{end} \cdot t$ | PDF Vector Primitives | `parse` |
| EQ-03 | $B(t) = (1-t)^3 P_0 + 3(1-t)^2 t P_1 + 3(1-t) t^2 P_2 + t^3 P_3$ | PDF Vector Primitives | `parse` |
| EQ-04 | $G_{init} = (V_{init}, E_{init})$ | PDF Vector Primitives | `parse` |
| EQ-05 | $\text{Attention}(Q_{vector}, K_{vision}, V_{vision})$ | Cross-Modal Tokenization | `token` |
| EQ-06 | $\min_\theta \sum_{t=1}^{T} \mathbb{E} \left\| \epsilon - \epsilon_\theta(G_t, t, c) \right\|^2$ | Graph Diffusion Engine | `diffuse` |
| EQ-07 | $\mathcal{L}_{total} = \mathcal{L}_{diffusion} + \lambda_{SAT}\,\mathcal{L}_{constraints}$ | NeSy Constraints | `constrain` |
