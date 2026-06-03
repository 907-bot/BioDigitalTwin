# Conversation handoff from Claude (claude.ai)

**Title:** Building a phased document structure  ·  **Started:** 1 June 2026  ·  **Turns:** 9

## Briefing

You are continuing a working session that began in Claude. Read the briefing, the artefacts, and the recent turns below. Pick up where the previous assistant left off — don't re-introduce yourself, match the user's working style.

**Original ask:**
> in this document that i have attached i will also share the phases or layers that i have divided , help build this step by step

**Where the previous assistant left off:**
> The user wants all 6 domain areas covered at an advanced/technical/mathematical level. This is a comprehensive domain knowledge session. Let me create detailed visual explanations for each domain, covering: 1. Human Biology & Physiology (hierarchical multi-scale from genome to whole body) 2. Graph …

---

## Recent conversation (last 6 of 9 turns; 3 earlier omitted)

**Assistant:**
The user wants a full visual roadmap first and needs setup guidance too. Let me create a comprehensive visual roadmap of the entire Bio-Digital Twin Platform. I'll use the visualizer to create an interactive/visual roadmap that covers both the PRD phases and the architecture layers.

Let me read the frontend-design skill first since I'll be building a visual.
```
This block is not supported on your current device yet.
```

Now I have the design system. Let me create a comprehensive interactive visual roadmap for the Bio-Digital Twin Platform. I'll build an HTML interactive widget that shows:

1. The full architecture layers (from the second document)
2. The implementation phases (from the PRD)
3. Tech stack per layer
4. Setup guidance overview

Let me design this as an interactive roadmap with tabs - one for the Architecture Layers view and one for the Implementation Phases view. Users can click on each layer/phase to see details.

I'll create a clean, interactive HTML widget that shows:
- Two views: "Architecture Layers" and "Build Phases"
- Clickable items that reveal tech stack and details
- Color coding by category
- Setup requirements highlighted

Let me think about the layout:
- Header with tabs (Architecture | Phases | Setup)
- Architecture tab: 17 layers in a vertical stack, color coded by category
- Phases tab: 6 phases (PRD) + 5 phases (extended arch doc)
- Setup tab: what you need to get started

I'll keep it clean and interactive. Let me code this up properly.

For architecture layers, I'll group them by category:
- Biological layers (1-7): teal
- Intelligence layers (8-10): purple  
- Memory/Simulation (11-14): amber
- Agent/Output (15-17): coral

For phases:
- Phase 1: Synthetic Patient Generator
- Phase 2: Graph-Based Digital Twin
- Phase 3: Disease Dynamics
- Phase 4: Counterfactual Simulation
- Phase 5: LLM Agent Layer
- Phase 6: Full Dashboard

Let me build a nice interactive HTML widget.
```
This block is not supported on your current device yet.
```

Here's your full interactive roadmap. Here's how to use it:

**Architecture layers tab** — all 17 layers from genomic graph to 3D anatomical twin, color-coded by category (biological → intelligence → memory/simulation → agent/output). Click any layer to expand its tech stack and what it does.

**Build phases tab** — the 6 PRD phases mapped week-by-week. Click each phase to see specific tasks, the deliverable you'll ship, and every library involved.

**Setup guide tab** — the exact commands to get your machine ready, from Docker to Ollama to the frontend.

---

**The recommended starting sequence for your setup:**

1. Install Docker Desktop and run the compose file (Neo4j + PostgreSQL + Redis + Qdrant will all start together)
2. Create a Python 3.11 venv and install the Phase 1 libraries only — no need to install everything upfront
3. Pull Llama 3.1 and Qwen2.5-Coder via Ollama (these are large downloads — do it early)
4. Don't touch Next.js until Phase 6 — it'll just be noise before then

When you're ready, say **"start Phase 1"** and I'll scaffold the full synthetic patient generator — folder structure, FastAPI app, PostgreSQL schema, and synthetic data pipeline, all running locally.

**User:**
before building this provide me th domain knowledge

**Assistant:**
The user wants domain knowledge before building. This is a broad request covering the Bio-Digital Twin platform. Let me ask what specific domain knowledge they want - since the project spans many fields.Great call — this project spans several deep fields. Let me understand which areas to cover so I give you exactly what's useful.
```
This block is not supported on your current device yet.
```

**User:**
Q: Which domain areas do you want covered? (Select all that apply)
A: Human biology & physiology, Graph Neural Networks (GNNs), Digital Twin concepts, Causal AI & counterfactuals, Spiking Neural Networks (SNNs), Synthetic data generation

Q: How deep should the explanations go?
A: Advanced — go technical & mathematical

**Assistant:**
The user wants all 6 domain areas covered at an advanced/technical/mathematical level. This is a comprehensive domain knowledge session. Let me create detailed visual explanations for each domain, covering:

1. Human Biology & Physiology (hierarchical multi-scale from genome to whole body)
2. Graph Neural Networks (GNNs) - message passing, mathematical formulation
3. Digital Twin concepts - what they are, how they work
4. Causal AI & counterfactuals - SCMs, do-calculus, Pearl's framework
5. Spiking Neural Networks (SNNs) - LIF neurons, STDP, temporal coding
6. Synthetic data generation - GANs, VAEs, CTGAN, TimeGAN

I'll cover each one systematically with visuals. Let me build interactive/visual explanations for each. I'll do them one by one with prose explanations and visuals.

Let me plan the approach:
- For each domain, write technical prose + create a visualization
- Start with Human Biology (since it's the foundation), then GNNs, Digital Twins, Causal AI, SNNs, Synthetic Data

Actually, given the breadth and depth, let me create a comprehensive interactive widget that covers all 6 domains with tabs and rich technical content + diagrams. Or I could do them sequentially with individual visualizations.

Let me do them one by one - each with a good visual and solid technical prose. This will be a long response but very valuable.

Let me structure this:

1. **Human Biology & Physiology** - the hierarchical multi-scale model
2. **Graph Neural Networks** - message passing equations, applications to biology
3. **Digital Twin concepts** - state space representation, synchronization
4. **Causal AI & Counterfactuals** - SCMs, do-calculus, Pearl's ladder of causation
5. **Spiking Neural Networks** - LIF model, spike trains, STDP
6. **Synthetic Data Generation** - GAN/VAE math, CTGAN, TimeGAN

For each, I'll write technical content and create a visualization. Let me be selective with visuals - use them where spatial/structural understanding helps most.

Let me create one big interactive reference widget that covers all 6 domains with rich technical content, then supplement with individual diagrams for the most visual concepts (GNNs, SNNs, Causal graphs).

Actually, let me do a mix:
- Create a comprehensive multi-tab interactive knowledge base widget
- Then add specific diagrams for concepts that benefit most from visualization

Let me start with the widget covering all 6 domains with mathematical depth, then add 2-3 focused diagrams.

Given the complexity, I'll build one large interactive knowledge base widget with all 6 domains, deep technical content, equations (using simple text math since we can't render LaTeX easily), and diagrams embedded.

Let me plan what to include per domain:

**1. Human Biology & Physiology**
- Hierarchical scales: genome → protein → cell → tissue → organ → system → body
- Key physiological signals: HR, HRV, SpO2, cortisol, glucose
- Homeostasis: feedback loops
- Biomarkers and their clinical meaning
- Relevant to digital twin: which signals to model

**2. Graph Neural Networks**
- Graph definition: G = (V, E, X)
- Message passing: h_v^(l+1) = UPDATE(h_v^l, AGGREGATE({m_uv : u ∈ N(v)}))
- Key architectures: GCN, GAT, GraphSAGE
- Application to biology: PPI networks, disease graphs
- Mathematical formulation of each

**3. Digital Twin**
- Definition: virtual replica synchronized with real entity
- State space model: x_{t+1} = f(x_t, u_t, w_t), y_t = g(x_t, v_t)
- Synchronization loop
- Difference from simulation
- Hierarchical twins

**4. Causal AI & Counterfactuals**
- Pearl's Ladder of Causation: Association, Intervention, Counterfactual
- SCM: X := f_X(PA_X, U_X)
- do-calculus: P(Y|do(X=x))
- Backdoor criterion
- Counterfactual: Y_{X=x}(u) 
- DoWhy framework

**5. Spiking Neural Networks**
- LIF neuron model: τ_m dV/dt = -(V - V_rest) + R·I(t)
- Spike generation: V ≥ V_th → spike, reset
- Temporal coding vs rate coding
- STDP learning rule
- Brian2 / SNNTorch implementation concepts

**6. Synthetic Data Generation**
- GAN objective: min_G max_D E[log D(x)] + E[log(1-D(G(z)))]
- CTGAN for tabular: mode-specific normalization, conditional vector
- TimeGAN: embedding + supervisor + generator
- SDV/Gaussian Copula for structured data
- Privacy-utility tradeoff

Let me build a comprehensive interactive widget. I'll use tabs for each domain, with subsections, math, and good explanation.

This is going to be a large widget. Let me make it clean and navigable.

For the math, I'll use Unicode characters and plain text math notation since we can't render LaTeX: subscripts with Unicode subscript chars, or just write it descriptively.

Let me write the widget code. I'll make it a tabbed knowledge base with rich content per domain.

Actually, I think the best approach here is to write substantial prose for each domain (outside the tool) and use the visualizer for the most impactful visuals - specifically:
1. A diagram showing the human biology hierarchy
2. A GNN message passing diagram (interactive)
3. A causal DAG / ladder of causation
4. An SNN LIF neuron diagram

And write the mathematical/technical content as prose.

But that would be a very long response. Let me instead build ONE big interactive knowledge base widget that is navigable, with all 6 domains, with mathematical depth. Then I write brief intro prose outside the widget.

Let me code this up carefully.

```html
<style>
/* Knowledge base styles */
.kb-nav { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 20px; }
.kb-btn { ... }
/* etc */
</style>
```

I'll make each domain section scroll-friendly with:
- Overview paragraph
- Key equations (in monospace/code blocks)
- Key concepts as expandable items
- Visual diagram (SVG inline) for some

Actually for a really good technical deep dive, let me do this as a series of sections - write the prose and create targeted visualizations. Let me do maybe 3-4 visual widgets and write the rest as detailed prose.

Here's my plan:
1. Write intro prose
2. **Domain 1: Human Biology** → visual (hierarchy diagram)
3. Write GNN prose + **Domain 2: GNN** → visual (message passing diagram)  
4. Write **Domain 3: Digital Twin** prose + visual (state-space sync loop)
5. Write **Domain 4: Causal AI** prose + visual (ladder of causation + DAG)
6. Write **Domain 5: SNN** prose + visual (LIF neuron + spike train)
7. Write **Domain 6: Synthetic Data** prose (no visual needed, equations in prose)

Actually, that's a LOT of tool calls and prose. Let me create one comprehensive interactive knowledge base widget that covers all 6, with mathematical depth, using inline SVG diagrams for the visual ones, and mathematical notation throughout. This keeps everything in one place and navigable.

Let me write this large but well-structured widget.

For math notation I'll use:
- Subscripts: h_v, x_t etc (plain text)
- Superscripts: h^(l) for layer l
- Greek: τ, μ, σ, θ etc (unicode directly)
- Arrows: →, ←, ↔
- Summation: Σ

Let me structure each domain section in the widget with:
- One-line definition
- Core mathematical framework (in a code/math block)
- 3-5 key concepts
- Relevance to Bio-Digital Twin
- A mini SVG diagram where applicable

Let me write the full HTML widget. This will be substantial but I'll keep it clean.

For the widget, I'll have:
- A sidebar navigation (domain names)
- Main content area
- Each domain: header, math block, key concepts, bio-twin application

Let me code this now, keeping it under reasonable size but technically deep.

Actually I realize this widget is going to be HUGE. Let me be strategic:

Option A: One big 6-tab widget with all technical content
Option B: 6 separate, focused visualizations with prose between them

I'll go with Option A - one large navigable widget. Let me write it efficiently.

Key design decisions:
- Left nav pills for domain switching
- Each domain has: definition box, math section, concepts, application
- Use code-style blocks for equations
- Include mini SVG diagrams inline for GNN message passing, SNN LIF, Causal DAG

Let me write the widget now. I need to be careful about:
1. No position:fixed
2. CSS variables for colors
3. Dark mode support
4. No hardcoded hex colors (use CSS vars where possible, or the ramp hex)
5. Proper font sizes (11px minimum)

Here goes:

```html
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
.kb { display: flex; gap: 16px; min-height: 500px; }
.kb-nav { width: 160px; flex-shrink: 0; }
.kb-btn { display: block; width: 100%; text-align: left; padding: 8px 10px; font-size: 12px; border: none; background: transparent; border-radius: 6px; cursor: pointer; color: var(--color-text-secondary); margin-bottom: 2px; }
.kb-btn.active { background: var(--color-background-secondary); color: var(--color-text-primary); font-weight: 500; }
.kb-btn:hover:not(.active) { background: var(--color-background-secondary); }
.kb-content { flex: 1; min-width: 0; }
.domain { display: none; }
.domain.active { display: block; }
.domain-header { margin-bottom: 16px; }
.domain-title { font-size: 18px; font-weight: 500; color: var(--color-text-primary); margin-bottom: 4px; }
.domain-def { font-size: 13px; color: var(--color-text-secondary); line-height: 1.5; }
.section-label { font-size: 11px; font-weight: 500; text-transform: uppercase; letter-spacing: 0.06em; color: var(--color-text-secondary); margin: 16px 0 8px; }
.math-block { font-family: var(--font-mono); font-size: 12px; background: var(--color-background-secondary); border: 0.5px solid var(--color-border-tertiary); border-radius: 8px; padding: 12px 14px; line-height: 1.8; color: var(--color-text-primary); margin-bottom: 12px; }
.concept-row { border: 0.5px solid var(--color-border-tertiary); border-radius: 8px; margin-bottom: 6px; overflow: hidden; }
.concept-head { display: flex; justify-content: space-between; align-items: center; padding: 9px 12px; cursor: pointer; }
.concept-head:hover { background: var(--color-background-secondary); }
.concept-title { font-size: 13px; font-weight: 500; color: var(--color-text-primary); }
.concept-body { display: none; padding: 10px 12px; border-top: 0.5px solid var(--color-border-tertiary); font-size: 13px; color: var(--color-text-secondary); line-height: 1.6; }
.concept-body.open { display: block; }
.app-box { background: var(--color-background-secondary); border-radius: 8px; padding: 12px 14px; margin-top: 16px; border-left: 3px solid; }
.app-label { font-size: 11px; font-weight: 500; margin-bottom: 6px; }
.app-text { font-size: 13px; color: var(--color-text-secondary); line-height: 1.5; }
</style>

<h2 style="font-size:16px;font-weight:500;color:var(--color-text-primary);margin-bottom:16px">Bio-Digital Twin — Domain knowledge</h2>

<div class="kb">
  <div class="kb-nav">
    <!-- nav buttons -->
  </div>
  <div class="kb-content">
    <!-- domain panels -->
  </div>
</div>
```

OK let me just write the full thing. I'll be thorough but efficient.

For the GNN section, I'll include an inline SVG of message passing.
For SNN, I'll include an inline SVG of LIF neuron + spike train.
For Causal AI, I'll include an inline SVG of the Pearl ladder and a DAG.

Let me write this properly. Given the complexity, I'll focus on making the content excellent and the code clean.

One concern - this widget will be quite long. Let me estimate: 6 domains × ~300 chars of HTML per concept × 5 concepts each = ~9000 chars just for concepts. Plus SVGs... this might be too large.

Let me be more concise - use shorter explanations, fewer concepts per domain, but keep the math precise and the application to Bio-Digital Twin clear.

Structure per domain:
- Title + 2-sentence definition
- Math framework (1 code block, 3-5 equations)
- 3 key concepts (expandable)
- Application box

Let me write this. I'll try to keep total widget under 15KB.

Here's the full widget:

For inline SVGs I'll keep them simple - the GNN message passing can be a small 300x150 SVG showing nodes with arrows and aggregation. The SNN can show a neuron with voltage trace. The Causal can show a DAG.

Let me write the full thing now. I'll be efficient with the code.

Actually, you know what - let me create a really clean, comprehensive single-page knowledge base. I'll write the HTML widget now:

For the domains, here's the content I'll include:

**Domain 1: Human Biology & Physiology**
Definition: The human body operates as a hierarchical, multi-scale biological system from genome (~20,500 genes) to 30-37 trillion cells to 78 organs organized into 11 physiological systems.

Math framework:
- Homeostasis: dx/dt = f(x, u) where x is physiological state, u is control input
- HRV: SDNN = sqrt(1/N Σ(RR_i - RR_mean)²)
- Signal hierarchy: y_organ = g(Σ w_i · y_tissue_i)

Concepts:
1. Hierarchical scale - genome → protein → cell → tissue → organ → system
2. Biomarker dynamics - HR, HRV, SpO2, cortisol, glucose and their inter-relationships
3. Feedback loops - negative feedback (homeostasis), positive feedback (cascades)
4. Multi-scale coupling - how cellular dysfunction propagates to organ failure
5. Disease as perturbation - disease = sustained deviation from homeostatic equilibrium

Application: Each layer in our Bio-Digital Twin corresponds directly to one biological scale. GNNs model the graph structure, SNNs model the temporal dynamics.

**Domain 2: GNNs**
Definition: GNNs are a class of neural networks operating on graph-structured data G = (V, E, X) where V are nodes, E edges, and X ∈ ℝ^{|V|×d} node features.

Math:
- Message passing: m_uv^(l) = MSG(h_u^(l), h_v^(l), e_uv)
- Aggregation: a_v^(l) = AGG({m_uv^(l) : u ∈ N(v)})
- Update: h_v^(l+1) = UPDATE(h_v^(l), a_v^(l))
- GCN: h_v^(l+1) = σ(Σ_{u∈N(v)∪{v}} (1/√(d_u·d_v)) · W^(l) · h_u^(l))
- GAT: h_v^(l+1) = σ(Σ_{u∈N(v)} α_vu · W · h_u) where α_vu = softmax(e_vu)

Concepts:
1. Message passing neural networks (MPNN) framework
2. GCN - spectral graph convolution via Laplacian normalization
3. GAT - attention-weighted neighborhood aggregation
4. GraphSAGE - inductive learning via neighborhood sampling
5. Biological graph applications - PPI, gene regulatory networks, disease propagation

Application: Each organ/biomarker is a node. Disease relationships are edges. GNN propagates disease signals across the biological graph.

**Domain 3: Digital Twin**
Definition: A digital twin is a synchronized virtual replica of a physical system, described by a state-space model continuously updated by real-world observations.

Math:
- State equation: x_{t+1} = f(x_t, u_t) + w_t, w_t ~ N(0, Q)
- Observation equation: y_t = h(x_t) + v_t, v_t ~ N(0, R)
- Kalman update: x̂_t|t = x̂_t|t-1 + K_t(y_t - h(x̂_t|t-1))
- Kalman gain: K_t = P_t|t-1 H_t^T (H_t P_t|t-1 H_t^T + R)^{-1}
- Twin divergence: D(t) = ||x_physical(t) - x_digital(t)||₂

Concepts:
1. State-space representation - hidden state vs observable measurements
2. Synchronization - how the digital state tracks the physical reality
3. Kalman filtering - optimal state estimation under Gaussian noise
4. Prediction vs nowcasting - using twin to forecast future states
5. Fidelity vs tractability - hierarchical approximation strategies

Application: Patient vitals (HR, HRV, SpO2) are y_t. The biological graph state is x_t. The GNN+SNN is f(). The twin stays synchronized with wearable data.

**Domain 4: Causal AI & Counterfactuals**
Definition: Causal AI goes beyond correlation to model interventional and counterfactual distributions using Pearl's Structural Causal Models (SCMs).

Math:
- SCM: X_i := f_i(PA_i, U_i) for each variable X_i
- Observational: P(Y|X=x) — correlation
- Interventional: P(Y|do(X=x)) — causal effect
- Counterfactual: P(Y_{X=x}|X=x', Y=y') — what would have happened
- ATE: τ = E[Y|do(X=1)] - E[Y|do(X=0)]
- Backdoor adjustment: P(Y|do(X)) = Σ_z P(Y|X,Z=z)P(Z=z)

Concepts:
1. Pearl's Ladder of Causation - See (association) → Do (intervention) → Imagine (counterfactual)
2. DAGs and d-separation - encoding conditional independence
3. do-calculus - 3 rules for computing interventional distributions
4. Identifiability - when can P(Y|do(X)) be estimated from observational data?
5. Counterfactual inference - abduction → action → prediction

Application: "What if this patient took metformin?" → do(Medication=metformin). Counterfactual: "Would this patient have developed diabetes if they had exercised more?"

**Domain 5: SNNs**
Definition: SNNs are bio-inspired neural networks where neurons communicate via discrete spikes over continuous time, governed by differential equations modeling membrane potential dynamics.

Math:
- LIF: τ_m dV/dt = -(V - V_rest) + R_m I(t)
- Spike: if V(t) ≥ V_th → emit spike, reset V → V_reset
- Synaptic current: I(t) = Σ_j w_j Σ_k δ(t - t_j^k) * α(t)
- STDP: Δw = A_+ exp(-Δt/τ_+) if Δt>0, -A_- exp(Δt/τ_-) if Δt<0
- Temporal coding: information in spike timing, not just rate

Concepts:
1. LIF neuron - leaky integrate-and-fire, biologically grounded
2. Spike trains - sparse binary temporal representations
3. Rate coding vs temporal coding - mean firing rate vs precise spike timing
4. STDP - spike-timing dependent plasticity, Hebbian learning in time
5. Biological time constants - neural signaling (ms) to circadian rhythms (24h)

Application: Disease progression unfolds over time. SNNs model temporal dynamics: hormonal oscillations, neural signaling cascades, circadian rhythm disruption in metabolic disease.

**Domain 6: Synthetic Data**
Definition: Synthetic data generation creates statistically representative artificial datasets that preserve the joint distribution of real data while protecting privacy and enabling augmentation.

Math:
- GAN: min_G max_D E_x[log D(x)] + E_z[log(1 - D(G(z)))]
- VAE: L = E_q[log p(x|z)] - KL(q(z|x) || p(z))
- CTGAN: mode-specific normalization y = (y - μ_k)/σ_k, conditional vector c ~ Cat(K)
- Wasserstein distance: W(P,Q) = inf_{γ∈Π} E_(x,y)~γ [||x-y||]
- Privacy: (ε,δ)-differential privacy: P(M(D)∈S) ≤ e^ε P(M(D')∈S) + δ

Concepts:
1. GAN framework - adversarial training for realistic data generation
2. CTGAN - handles mixed tabular data (continuous + categorical)
3. TimeGAN - temporal GAN with embedding + supervisor networks
4. Copula models - SDV's approach via Gaussian Copula for dependency structure
5. Differential privacy - formal privacy guarantees on synthetic data

Application: No real patient data is available at MVP stage. CTGAN generates synthetic EHR tables. TimeGAN generates realistic physiological time series. This lets the twin train on statistically valid data.

OK I think I have enough content to write a good widget. Let me write it now. I'll keep the code clean and efficient.

Let me think about the layout more carefully. Since the container is 680px wide and we're on desktop:
- Left nav: 150px
- Main content: ~520px

Actually for mobile friendliness let me do top tabs instead of side nav, or make the side nav collapsible.

Actually the system says desktop, so let me go with side nav layout. 150px nav + 16px gap + main content.

Let me write the widget now. I'll include:
- Side navigation with domain icons
- Content area per domain
- Inline mini SVGs for GNN, SNN, Causal DAG
- Math blocks in monospace
- Expandable concept sections

Here's my attempt at writing this large but clean widget. I'll try to keep it readable and avoid any forbidden patterns.

Let me write it section by section in my head then output:

Navigation buttons: Biology, GNN, Digital Twin, Causal AI, SNN, Synthetic Data - with icons from Tabler

For the inline SVGs:
- GNN message passing: small graph showing node aggregation
- SNN: LIF voltage trace + spikes
- Causal: Pearl's ladder as 3 steps

Let me code this now. I'll use a function-based approach for toggling concepts.

Here's the complete widget HTML (I'll write it out):

Actually, I realize I should just write this and not overthink it. Here goes:

I'll create a clean two-column layout:
- Left: sticky domain navigation (150px)
- Right: domain content that scrolls

Each domain content:
1. Title + subtitle (what it is in one line)
2. Section: "Core mathematics" - a code block with key equations
3. Section: "Key concepts" - 3-4 expandable items
4. Section: "In your Bio-Digital Twin" - colored application box

For the SVGs, I'll keep them very simple and focused:
- Biology: hierarchical scale pyramid (simple boxes stacked)
- GNN: 5 nodes with aggregation arrows (clear message passing illustration)
- Causal: 3-rung ladder with DAG example
- SNN: voltage trace + spike raster

Let me write the final widget. I'll be careful about:
- Width constraints (680px total, nav 150px, content ~514px)
- No position:fixed
- CSS variables
- Min font 11px
- Working dark mode

Here's the code:

```html
<style>
*{box-sizing:border-box;margin:0;padding:0}
.kb-wrap{display:flex;gap:16px}
.kb-nav{width:148px;flex-shrink:0}
.nav-btn{display:flex;align-items:center;gap:7px;width:100%;padding:8px 10px;border:none;background:transparent;border-radius:8px;cursor:pointer;font-size:12px;color:var(--color-text-secondary);text-align:left;margin-bottom:3px;transition:background .1s}
.nav-btn i{font-size:15px;flex-shrink:0}
.nav-btn.active{background:var(--color-background-secondary);color:var(--color-text-primary);font-weight:500}
.nav-btn:hover:not(.active){background:var(--color-background-secondary)}
.kb-main{flex:1;min-width:0}
.domain{display:none}
.domain.active{display:block}
.dtitle{font-size:16px;font-weight:500;color:var(--color-text-primary);margin-bottom:4px}
.dsub{font-size:12px;color:var(--color-text-secondary);line-height:1.5;margin-bottom:16px}
.slabel{font-size:10px;font-weight:500;letter-spacing:.07em;color:var(--color-text-secondary);text-transform:uppercase;margin:14px 0 7px}
.mblock{font-family:var(--font-mono);font-size:11.5px;background:var(--color-background-secondary);border:0.5px solid var(--color-border-tertiary);border-radius:8px;padding:11px 13px;line-height:1.9;color:var(--color-text-primary)}
.citem{border:0.5px solid var(--color-border-tertiary);border-radius:8px;margin-bottom:5px}
.chead{display:flex;justify-content:space-between;align-items:center;padding:8px 11px;cursor:pointer;border-radius:8px}
.chead:hover{background:var(--color-background-secondary)}
.ctitle{font-size:12.5px;font-weight:500;color:var(--color-text-primary)}
.cbody{display:none;padding:9px 11px;font-size:12px;color:var(--color-text-secondary);line-height:1.6;border-top:0.5px solid var(--color-border-tertiary)}
.cbody.open{display:block}
.appbox{border-radius:8px;padding:11px 13px;margin-top:14px;border-left:3px solid}
.applabel{font-size:10px;font-weight:500;letter-spacing:.06em;margin-bottom:5px}
.apptext{font-size:12px;line-height:1.55}
.arr{font-size:11px;transition:transform .15s;color:var(--color-text-secondary)}
.arr.open{transform:rotate(90deg)}
</style>

<h2 style="font-size:16px;font-weight:500;color:var(--color-text-primary);margin-bottom:4px">Domain knowledge</h2>
<p style="font-size:12px;color:var(--color-text-secondary);margin-bottom:16px">Six technical domains underpinning the Bio-Digital Twin — click any concept to expand</p>

<div class="kb-wrap">
<div class="kb-nav">
  <button class="nav-btn active" onclick="show('bio',this)"><i class="ti ti-dna" aria-hidden="true"></i>Biology</button>
  <button class="nav-btn" onclick="show('gnn',this)"><i class="ti ti-topology-star" aria-hidden="true"></i>GNNs</button>
  <button class="nav-btn" onclick="show('dt',this)"><i class="ti ti-refresh" aria-hidden="true"></i>Digital Twin</button>
  <button class="nav-btn" onclick="show('causal',this)"><i class="ti ti-git-branch" aria-hidden="true"></i>Causal AI</button>
  <button class="nav-btn" onclick="show('snn',this)"><i class="ti ti-activity" aria-hidden="true"></i>SNNs</button>
  <button class="nav-btn" onclick="show('synth',this)"><i class="ti ti-table" aria-hidden="true"></i>Synthetic data</button>
</div>

<div class="kb-main">
  <!-- BIOLOGY -->
  <div class="domain active" id="bio">
    <div class="dtitle">Human biology & physiology</div>
    <div class="dsub">The body is a hierarchical multi-scale system: ~20,500 protein-coding genes drive ~30–37 trillion cells organized into tissues, organs, and 11 physiological systems. Digital twin targets: model each scale as a distinct computational layer.</div>
    
    <div class="slabel">Governing mathematics</div>
    <div class="mblock">
Homeostasis (state equation):
  dx/dt = f(x, u) − λ·x + w(t)
  where x = physiological state, u = control input, λ = decay rate

HRV — SDNN:
  SDNN = √( (1/N) Σᵢ (RRᵢ − R̄R)² )

Multi-scale signal coupling:
  y_organ = g( Σᵢ wᵢ · y_tissue_i + b )

Disease as perturbation:
  x_disease(t) = x_healthy + δ(t),  ||δ(t)||₂ → 0 iff recovery
    </div>

    <!-- inline bio hierarchy SVG -->
    <svg width="100%" viewBox="0 0 480 220" role="img"><title>Biological hierarchy</title><desc>Seven-level stack from genome to whole body</desc>
      <defs><marker id="ba" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
      <!-- stacked hierarchy centered -->
      <!-- widths decrease: 300, 270, 240, 210, 180, 150, 120 -->
      <!-- starts at x = (480-w)/2 -->
      <g class="c-teal">
        <rect x="90" y="10" width="300" height="22" rx="4" stroke-width="0.5"/>
        <text class="th" x="240" y="21" text-anchor="middle" dominant-baseline="central" style="font-size:11px">Genome — 20,500 genes</text>
      </g>
      <line x1="240" y1="32" x2="240" y2="40" stroke="#1D9E75" stroke-width="1" marker-end="url(#ba)"/>
      <g class="c-teal">
        <rect x="105" y="42" width="270" height="22" rx="4" stroke-width="0.5"/>
        <text class="th" x="240" y="53" text-anchor="middle" dominant-baseline="central" style="font-size:11px">Proteome — proteins & pathways</text>
      </g>
      <line x1="240" y1="64" x2="240" y2="72" stroke="#1D9E75" stroke-width="1" marker-end="url(#ba)"/>
      <g class="c-teal" style="opacity:.9">
        <rect x="120" y="74" width="240" height="22" rx="4" stroke-width="0.5"/>
        <text class="th" x="240" y="85" text-anchor="middle" dominant-baseline="central" style="font-size:11px">Cell populations — clusters</text>
      </g>
      <line x1="240" y1="96" x2="240" y2="104" stroke="#1D9E75" stroke-width="1" marker-end="url(#ba)"/>
      <g class="c-teal" style="opacity:.8">
        <rect x="135" y="106" width="210" height="22" rx="4" stroke-width="0.5"/>
        <text class="th" x="240" y="117" text-anchor="middle" dominant-baseline="central" style="font-size:11px">Tissues — cardiac, neural, hepatic</text>
      </g>
      <line x1="240" y1="128" x2="240" y2="136" stroke="#1D9E75" stroke-width="1" marker-end="url(#ba)"/>
      <g class="c-teal" style="opacity:.7">
        <rect x="150" y="138" width="180" height="22" rx="4" stroke-width="0.5"/>
        <text class="th" x="240" y="149" text-anchor="middle" dominant-baseline="central" style="font-size:11px">Organs — heart, liver, brain</text>
      </g>
      <line x1="240" y1="160" x2="240" y2="168" stroke="#1D9E75" stroke-width="1" marker-end="url(#ba)"/>
      <g class="c-teal" style="opacity:.6">
        <rect x="165" y="170" width="150" height="22" rx="4" stroke-width="0.5"/>
        <text class="th" x="240" y="181" text-anchor="middle" dominant-baseline="central" style="font-size:11px">Physiological systems</text>
      </g>
      <line x1="240" y1="192" x2="240" y2="200" stroke="#1D9E75" stroke-width="1" marker-end="url(#ba)"/>
      <g class="c-teal" style="opacity:.5">
        <rect x="180" y="202" width="120" height="22" rx="4" stroke-width="0.5"/>
        <text class="th" x="240" y="213" text-anchor="middle" dominant-baseline="central" style="font-size:11px">Whole human</text>
      </g>
    </svg>

    <div class="slabel">Key concepts</div>
    <div id="bio-concepts"></div>
    <div class="appbox" style="background:#E1F5EE30;border-color:#1D9E75;color:#085041">
      <div class="applabel" style="color:#0F6E56">IN YOUR BIO-DIGITAL TWIN</div>
      <div class="apptext" style="color:var(--color-text-secondary)">Each layer maps to one computational layer in the architecture. Layers 1–7 in your PRD correspond 1:1 with genome → whole human. GNNs model the graph topology at each scale. SNNs handle temporal dynamics. The homeostasis equation is literally your state update function.</div>
    </div>
  </div>

  <!-- GNN -->
  <div class="domain" id="gnn">
    <div class="dtitle">Graph Neural Networks</div>
    <div class="dsub">GNNs operate on G = (V, E, X) — a graph with node set V, edge set E, and node features X ∈ ℝ^{|V|×d}. They learn representations by iteratively aggregating information from graph neighborhoods via message passing.</div>

    <div class="slabel">Governing mathematics</div>
    <div class="mblock">
General MPNN framework (Gilmer et al., 2017):
  mᵥᵤ⁽ˡ⁾  = MSG(hᵥ⁽ˡ⁾, hᵤ⁽ˡ⁾, eᵥᵤ)          ← message
  aᵥ⁽ˡ⁾   = AGG({ mᵥᵤ⁽ˡ⁾ : u ∈ N(v) })        ← aggregate
  hᵥ⁽ˡ⁺¹⁾ = UPDATE(hᵥ⁽ˡ⁾, aᵥ⁽ˡ⁾)             ← update

GCN (Kipf & Welling, 2017):
  H⁽ˡ⁺¹⁾ = σ( D̃⁻¹/² Ã D̃⁻¹/² H⁽ˡ⁾ W⁽ˡ⁾ )
  where Ã = A + I,  D̃ᵢᵢ = Σⱼ Ãᵢⱼ

GAT (Veličković et al., 2018):
  αᵤᵥ = softmax( LeakyReLU(aᵀ[W·hᵤ || W·hᵥ]) )
  hᵥ⁽ˡ⁺¹⁾ = σ( Σᵤ∈N(v) αᵤᵥ · W · hᵤ⁽ˡ⁾ )
    </div>

    <!-- GNN message passing mini SVG -->
    <svg width="100%" viewBox="0 0 480 160" role="img"><title>GNN message passing</title><desc>Central node aggregates messages from neighbors</desc>
      <defs><marker id="ga" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
      <!-- neighbor nodes -->
      <g class="c-purple"><circle cx="90" cy="50" r="22" stroke-width="0.5"/><text class="ts" x="90" y="50" text-anchor="middle" dominant-baseline="central">Gene</text></g>
      <g class="c-purple"><circle cx="90" cy="120" r="22" stroke-width="0.5"/><text class="ts" x="90" y="120" text-anchor="middle" dominant-baseline="central">Organ</text></g>
      <g class="c-purple"><circle cx="200" cy="20" r="22" stroke-width="0.5"/><text class="ts" x="200" y="20" text-anchor="middle" dominant-baseline="central">Cell</text></g>
      <!-- center node -->
      <g class="c-coral"><circle cx="240" cy="85" r="28" stroke-width="1"/><text class="th" x="240" y="85" text-anchor="middle" dominant-baseline="central" style="font-size:11px">Disease</text></g>
      <!-- edges with messages -->
      <line x1="110" y1="57" x2="214" y2="79" stroke="#7F77DD" stroke-width="1.5" marker-end="url(#ga)"/>
      <line x1="110" y1="114" x2="214" y2="90" stroke="#7F77DD" stroke-width="1.5" marker-end="url(#ga)"/>
      <line x1="218" y1="35" x2="236" y2="58" stroke="#7F77DD" stroke-width="1.5" marker-end="url(#ga)"/>
      <!-- message labels -->
      <text class="ts" x="155" y="63" text-anchor="middle" style="font-size:10px;fill:#7F77DD">m_uv</text>
      <text class="ts" x="155" y="108" text-anchor="middle" style="font-size:10px;fill:#7F77DD">m_uv</text>
      <!-- aggregation box -->
      <g class="c-gray"><rect x="300" y="62" width="120" height="46" rx="6" stroke-width="0.5"/></g>
      <text class="th" x="360" y="81" text-anchor="middle" dominant-baseline="central" style="font-size:11px">AGGREGATE</text>
      <text class="ts" x="360" y="97" text-anchor="middle" dominant-baseline="central">+ UPDATE</text>
      <line x1="268" y1="85" x2="298" y2="85" stroke="#D85A30" stroke-width="1.5" marker-end="url(#ga)"/>
      <!-- right neighbor -->
      <g class="c-purple"><circle cx="430" cy="50" r="22" stroke-width="0.5"/><text class="ts" x="430" y="50" text-anchor="middle" dominant-baseline="central">Tissue</text></g>
      <line x1="408" y1="60" x2="268" y2="79" stroke="#7F77DD" stroke-width="1.5" marker-end="url(#ga)"/>
      <text class="ts" x="352" y="58" text-anchor="middle" style="font-size:10px;fill:#7F77DD">m_uv</text>
    </svg>

    <div class="slabel">Key concepts</div>
    <div id="gnn-concepts"></div>
    <div class="appbox" style="background:#EEEDFE30;border-color:#7F77DD;color:#26215C">
      <div class="applabel" style="color:#3C3489">IN YOUR BIO-DIGITAL TWIN</div>
      <div class="apptext" style="color:var(--color-text-secondary)">Biological entities (genes, proteins, organs, biomarkers) are nodes. Relationships (AFFECTS, CAUSES, ASSOCIATED_WITH) are edges. PyTorch Geometric runs GCN/GAT to propagate disease signals — e.g. insulin resistance in pancreas propagates to liver and cardiovascular nodes.</div>
    </div>
  </div>

  <!-- Digital Twin -->
  <div class="domain" id="dt">
    <div class="dtitle">Digital Twin</div>
    <div class="dsub">A digital twin is a synchronized virtual replica of a physical system, continuously updated by observations. Distinguished from pure simulation by its real-time data coupling and state estimation loop.</div>

    <div class="slabel">Governing mathematics</div>
    <div class="mblock">
State-space model:
  x_{t+1} = f(x_t, u_t) + w_t,   w_t ~ N(0, Q)   ← transition
  y_t     = h(x_t) + v_t,         v_t ~ N(0, R)   ← observation

Kalman filter (linear Gaussian case):
  Predict:  x̂_{t|t-1} = F·x̂_{t-1|t-1} + B·u_t
            P_{t|t-1}  = F·P_{t-1|t-1}·Fᵀ + Q

  Update:   K_t = P_{t|t-1}·Hᵀ·(H·P_{t|t-1}·Hᵀ + R)⁻¹
            x̂_{t|t} = x̂_{t|t-1} + K_t·(y_t − h(x̂_{t|t-1}))

Twin divergence metric:
  D(t) = || x_physical(t) − x_digital(t) ||₂
    </div>

    <div class="slabel">Key concepts</div>
    <div id="dt-concepts"></div>
    <div class="appbox" style="background:#FAEEDA30;border-color:#BA7517">
      <div class="applabel" style="color:#633806">IN YOUR BIO-DIGITAL TWIN</div>
      <div class="apptext" style="color:var(--color-text-secondary)">x_t is the biological graph state (organ health, biomarker levels). y_t is real-time wearable data (HR, HRV, SpO2, glucose). f() is the GNN+SNN model. The Kalman update step keeps the twin synchronized. D(t) detects drift — when your twin diverges from the patient, it triggers re-calibration.</div>
    </div>
  </div>

  <!-- Causal AI -->
  <div class="domain" id="causal">
    <div class="dtitle">Causal AI & counterfactuals</div>
    <div class="dsub">Causal AI reasons about interventions and hypotheticals — not just "what co-occurs" but "what causes what" and "what would have happened." Based on Pearl's Structural Causal Model (SCM) framework and the do-calculus.</div>

    <div class="slabel">Governing mathematics</div>
    <div class="mblock">
Structural Causal Model (Pearl, 2000):
  Xᵢ := fᵢ(PAᵢ, Uᵢ)    for each variable Xᵢ
  where PAᵢ = causal parents, Uᵢ = exogenous noise

Three rungs of causation:
  Rung 1 — Association:     P(Y | X=x)           observe
  Rung 2 — Intervention:    P(Y | do(X=x))        act
  Rung 3 — Counterfactual:  P(Yₓ=ₓ | X=x', Y=y') imagine

Backdoor adjustment (identification):
  P(Y|do(X)) = Σ_z P(Y|X,Z=z)·P(Z=z)
  valid when Z blocks all backdoor paths X ← … → Y

Average Treatment Effect:
  ATE = E[Y|do(X=1)] − E[Y|do(X=0)]

Counterfactual (3 steps):
  1. Abduction:  infer U from observed (X=x', Y=y')
  2. Action:     set X := x in the SCM
  3. Prediction: compute Yₓ₌ₓ under modified model
    </div>

    <!-- Causal DAG SVG -->
    <svg width="100%" viewBox="0 0 480 150" role="img"><title>Causal DAG example</title><desc>Sleep causes inflammation which causes insulin resistance which causes diabetes risk</desc>
      <defs><marker id="ca" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
      <!-- nodes left to right -->
      <g class="c-teal"><rect x="10" y="58" width="80" height="34" rx="6" stroke-width="0.5"/><text class="th" x="50" y="72" text-anchor="middle" dominant-baseline="central" style="font-size:11px">Sleep</text><text class="ts" x="50" y="85" text-anchor="middle" dominant-baseline="central">Lifestyle</text></g>
      <g class="c-amber"><rect x="120" y="58" width="90" height="34" rx="6" stroke-width="0.5"/><text class="th" x="165" y="72" text-anchor="middle" dominant-baseline="central" style="font-size:11px">Inflam.</text><text class="ts" x="165" y="85" text-anchor="middle" dominant-baseline="central">Biomarker</text></g>
      <g class="c-coral"><rect x="240" y="58" width="90" height="34" rx="6" stroke-width="0.5"/><text class="th" x="285" y="72" text-anchor="middle" dominant-baseline="central" style="font-size:10px">Insulin resist.</text><text class="ts" x="285" y="85" text-anchor="middle" dominant-baseline="central">Metabolic</text></g>
      <g class="c-red"><rect x="360" y="58" width="100" height="34" rx="6" stroke-width="0.5"/><text class="th" x="410" y="72" text-anchor="middle" dominant-baseline="central" style="font-size:11px">Diabetes</text><text class="ts" x="410" y="85" text-anchor="middle" dominant-baseline="central">Disease risk</text></g>
      <!-- causal edges -->
      <line x1="90" y1="75" x2="118" y2="75" stroke="#BA7517" stroke-width="1.5" marker-end="url(#ca)"/>
      <line x1="210" y1="75" x2="238" y2="75" stroke="#D85A30" stroke-width="1.5" marker-end="url(#ca)"/>
      <line x1="330" y1="75" x2="358" y2="75" stroke="#E24B4A" stroke-width="1.5" marker-end="url(#ca)"/>
      <!-- do intervention arc -->
      <path d="M50 58 Q50 20 285 58" fill="none" stroke="#7F77DD" stroke-width="1" stroke-dasharray="4 3" marker-end="url(#ca)"/>
      <text class="ts" x="170" y="18" text-anchor="middle" style="font-size:10px;fill:#7F77DD">do(Sleep=8h) ?</text>
    </svg>

    <div class="slabel">Key concepts</div>
    <div id="causal-concepts"></div>
    <div class="appbox" style="background:#EEEDFE30;border-color:#7F77DD">
      <div class="applabel" style="color:#3C3489">IN YOUR BIO-DIGITAL TWIN</div>
      <div class="apptext" style="color:var(--color-text-secondary)">DoWhy models the patient's biological SCM. Counterfactual engine answers: "What would this patient's HbA1c be if they had slept 8h instead of 5h for the past 6 months?" — using abduction on their observed biomarkers, then simulating the modified SCM. EconML estimates heterogeneous treatment effects across patient subgroups.</div>
    </div>
  </div>

  <!-- SNN -->
  <div class="domain" id="snn">
    <div class="dtitle">Spiking Neural Networks</div>
    <div class="dsub">SNNs are the third generation of neural networks. Neurons communicate via discrete spikes over continuous time, modeled by differential equations. They encode information in spike timing rather than firing rates, matching biological neural computation.</div>

    <div class="slabel">Governing mathematics</div>
    <div class="mblock">
Leaky Integrate-and-Fire (LIF) neuron:
  τ_m dV/dt = −(V − V_rest) + R_m · I(t)

Spike & reset rule:
  if V(t) ≥ V_th:  emit spike s(t) = δ(t − t_spike)
                   reset V → V_reset

Synaptic input current:
  I(t) = Σⱼ wⱼ · Σₖ α(t − tⱼᵏ)
  where α(t) = (t/τ_s)·exp(1 − t/τ_s)·Θ(t)  [α-function]

STDP learning rule (Bi & Poo, 1998):
  Δw = { A₊·exp(−Δt/τ₊)   if Δt > 0  (pre before post → LTP)
        { −A₋·exp(Δt/τ₋)  if Δt < 0  (post before pre → LTD)
  where Δt = t_post − t_pre

Biological time constants:
  τ_m ~ 10–30ms (membrane), τ_s ~ 5ms (synapse)
  τ_circadian ~ 24h, τ_hormonal ~ minutes–hours
    </div>

    <!-- SNN voltage trace mini SVG -->
    <svg width="100%" viewBox="0 0 480 120" role="img"><title>LIF voltage trace</title><desc>Membrane potential building up and spiking three times then resetting</desc>
      <defs><marker id="sa" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M2 1L8 5L2 9" fill="none" stroke="context-stroke" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></marker></defs>
      <!-- axes -->
      <line x1="40" y1="10" x2="40" y2="100" stroke="var(--color-border-secondary)" stroke-width="0.5"/>
      <line x1="40" y1="100" x2="460" y2="100" stroke="var(--color-border-secondary)" stroke-width="0.5"/>
      <!-- threshold line -->
      <line x1="40" y1="35" x2="460" y2="35" stroke="#E24B4A" stroke-width="0.5" stroke-dasharray="3 3"/>
      <text class="ts" x="36" y="38" text-anchor="end" style="font-size:10px;fill:#E24B4A">V_th</text>
      <!-- rest line -->
      <line x1="40" y1="85" x2="460" y2="85" stroke="var(--color-border-tertiary)" stroke-width="0.5" stroke-dasharray="3 3"/>
      <text class="ts" x="36" y="88" text-anchor="end" style="font-size:10px">V_rest</text>
      <!-- voltage trace: integrate up, spike, reset, repeat x3 -->
      <path d="M40 85 Q80 85 110 35 L110 10 L112 85 Q150 85 185 35 L185 10 L187 85 Q230 85 270 35 L270 10 L272 85 Q340 85 440 70" fill="none" stroke="#7F77DD" stroke-width="1.5"/>
      <!-- spike labels -->
      <text class="ts" x="110" y="8" text-anchor="middle" style="font-size:10px;fill:#7F77DD">spike</text>
      <text class="ts" x="185" y="8" text-anchor="middle" style="font-size:10px;fill:#7F77DD">spike</text>
      <text class="ts" x="270" y="8" text-anchor="middle" style="font-size:10px;fill:#7F77DD">spike</text>
      <!-- axis labels -->
      <text class="ts" x="250" y="115" text-anchor="middle" style="font-size:10px">time →</text>
      <text class="ts" x="18" y="55" text-anchor="middle" style="font-size:10px">V(t)</text>
    </svg>

    <div class="slabel">Key concepts</div>
    <div id="snn-concepts"></div>
    <div class="appbox" style="background:#E1F5EE30;border-color:#1D9E75">
      <div class="applabel" style="color:#0F6E56">IN YOUR BIO-DIGITAL TWIN</div>
      <div class="apptext" style="color:var(--color-text-secondary)">SNNs model biological temporal dynamics at multiple timescales: neural signaling (ms), hormonal rhythms (minutes), disease progression (days-years). SNNTorch's surrogate gradient allows backpropagation through spikes. Circadian disruption is encoded as phase-shifted spike patterns. Disease trajectory = long-horizon spike sequence prediction.</div>
    </div>
  </div>

  <!-- Synthetic Data -->
  <div class="domain" id="synth">
    <div class="dtitle">Synthetic data generation</div>
    <div class="dsub">Synthetic data preserves the statistical joint distribution P(X₁, X₂, …, Xₙ) of real data without exposing actual patients. Critical for medical AI where real EHR data is inaccessible, regulated, or insufficient.</div>

    <div class="slabel">Governing mathematics</div>
    <div class="mblock">
GAN objective (Goodfellow et al., 2014):
  min_G max_D E_x[log D(x)] + E_z[log(1 − D(G(z)))]
  Nash equilibrium: D*(x) = p_data(x) / (p_data(x) + p_g(x))

Wasserstein GAN (more stable training):
  min_G max_{||f||_L≤1} E_x[f(x)] − E_z[f(G(z))]

VAE:
  L(θ,φ) = E_q[log p_θ(x|z)] − KL(q_φ(z|x) || p(z))

CTGAN (tabular, mixed types):
  Mode-specific normalization: ỹ = (y − μₖ) / σₖ
  Conditional vector: c ~ Cat(K=column modes)
  Trains G(z, c) to match empirical column distributions

TimeGAN (time series):
  4 networks: Embedder E, Recovery R, Generator G, Discriminator D
  L = L_supervised + L_unsupervised + λ·L_moments

(ε,δ)-Differential Privacy:
  P(M(D) ∈ S) ≤ eᵉ · P(M(D') ∈ S) + δ
  for all adjacent datasets D, D' differing by 1 record
    </div>

    <div class="slabel">Key concepts</div>
    <div id="synth-concepts"></div>
    <div class="appbox" style="background:#FAECE730;border-color:#D85A30">
      <div class="applabel" style="color:#993C1D">IN YOUR BIO-DIGITAL TWIN</div>
      <div class="apptext" style="color:var(--color-text-secondary)">Phase 1 uses SDV + CTGAN to generate synthetic patient profiles (age, gender, BMI, comorbidities) and TimeGAN for physiological time series (HR, HRV, SpO2 over 30 days). The synthetic data must pass statistical fidelity tests before training the GNN. SDV's Gaussian Copula preserves inter-variable correlations (e.g. BMI ↔ glucose ↔ insulin).</div>
    </div>
  </div>

</div><!-- kb-main -->
</div><!-- kb-wrap -->
```

OK that's the bulk of the widget. Now I need to add the JavaScript to populate the concepts and handle toggling.

The concepts data:

Bio concepts:
1. Hierarchical coupling - How dysfunction propagates across scales: gene mutation → misfolded protein → cell death → tissue necrosis → organ failure
2. Biomarker dynamics - HR (autonomic nervous system), HRV (vagal tone, stress), SpO2 (respiratory/cardiovascular), Cortisol (HPA axis), Glucose (endocrine/metabolic)
3. Homeostasis & allostasis - Negative feedback loops maintain equilibrium. Allostatic load = cumulative wear from repeated stress responses
4. Disease as graph perturbation - In the biological graph, disease = subgraph activation pattern. Cascade propagation follows edge weights

GNN concepts:
1. Expressiveness & WL test - GNNs are at most as expressive as the 1-WL graph isomorphism test. GIN (Graph Isomorphism Network) achieves this upper bound
2. Spectral vs spatial methods - Spectral (GCN): operates on graph Laplacian eigenvectors. Spatial (GAT, GraphSAGE): operates directly on node neighborhoods
3. Oversmoothing - Deep GNNs collapse all node features toward the same value. Solution: residual connections, PairNorm, DropEdge
4. Heterogeneous graphs - Biological graphs have multiple node/edge types. HAN (Heterogeneous Attention Network) handles this via metapath-based aggregation

Digital Twin concepts:
1. State estimation vs prediction - Kalman/particle filters synchronize current state from noisy observations. Forward simulation predicts future states
2. Digital thread - The audit trail of all states, inputs, and model updates over time. Essential for clinical explainability
3. Model uncertainty quantification - Bayesian neural networks or ensembles propagate parameter uncertainty into state uncertainty: P(x_t | y_{1:t})
4. Twin fidelity levels - L0: data mirror, L1: descriptive, L2: predictive, L3: prescriptive, L4: autonomous. Your MVP targets L2-L3

Causal concepts:
1. D-separation - A node Z d-separates X and Y in a DAG if all paths are blocked. Encodes conditional independence: X ⊥ Y | Z
2. Front-door criterion - Alternative to backdoor when confounders are unobserved: P(Y|do(X)) = Σ_m P(M=m|X)·Σ_x P(Y|M,X=x)·P(X=x)
3. Identifiability - Not all causal effects are identifiable from observational data. Non-parametric identifiability requires the do-calculus rules
4. Heterogeneous treatment effects - CATE = E[Y(1)-Y(0)|X=x]. EconML's CausalForest/DML estimates this per patient subgroup

SNN concepts:
1. Surrogate gradient learning - Spikes are non-differentiable. Surrogate gradients (SuperSpike, FastSigmoid) enable backpropagation through spike times
2. Temporal credit assignment - Eligibility traces e(t) bridge the gap between spike times and reward signals in multi-step credit assignment
3. Encoding schemes - Rate coding: frequency = value. Temporal coding: latency from stimulus = value. Population coding: ensemble of neurons represents distribution
4. Energy efficiency - SNNs are sparse by nature (~5% neurons active per timestep). Neuromorphic hardware (Intel Loihi) runs SNNs at 1000× less energy than GPUs

Synth concepts:
1. Fidelity-privacy tradeoff - Higher ε in differential privacy = more noise = lower fidelity. SDMetrics evaluates via: Column Shapes, Column Pair Trends, Boundary Adherence
2. SMOTE for tabular imbalance - Synthetic Minority Over-sampling: x_new = xᵢ + λ·(xⱼ − xᵢ), λ~U(0,1). Used for rare disease class upsampling
3. Downstream task validity - Synthetic data quality = how well models trained on synthetic data perform on real data (TSTR: Train on Synthetic, Test on Real)
4. Temporal coherence in health data - Time series must preserve autocorrelation structure. TimeGAN's supervisor network enforces stepwise temporal consistency

OK now let me also add the JS for toggling and rendering concepts.

Let me write the final complete widget. I'll be careful about the SVG text colors and dark mode.

Actually wait - I have a concern about the SVG text colors. I used `fill="#7F77DD"` for some text which won't adapt to dark mode. Let me use the `ts` class instead and avoid inline fill colors on text. The `ts` class is `var(--color-text-secondary)` which works in both modes.

Let me revise and write the final clean version. I'll remove inline fill colors from SVG text elements where possible, or use the appropriate hex from the ramp documentation noting that ramp colors (like #7F77DD) work in both modes as they're just used for colored accents, not as backgrounds.

Actually the issue is that I'm using ramp colors as fill for text labels like "m_uv" in the GNN diagram. These should be fine as colored accent text since they're colored fills (not black/white) that remain visible in both modes.

Let me write the final widget now. I'll keep the JS clean:

```javascript
const concepts = {
  bio: [...],
  gnn: [...],
  dt: [...],
  causal: [...],
  snn: [...],
  synth: [...]
};

// Render concepts
Object.keys(concepts).forEach(domain => {
  const container = document.getElementById(domain+'-concepts');
  concepts[domain].forEach((c, i) => {
    const id = domain+'-c-'+i;
    const el = document.createElement('div');
    el.className = 'citem';
    el.innerHTML = `<div class="chead" onclick="toggleC('${id}')"><span class="ctitle">${c.title}</span><span class="arr" id="${id}-arr">›</span></div><div class="cbody" id="${id}">${c.body}</div>`;
    container.appendChild(el);
  });
});

function toggleC(id){
  const body = document.getElementById(id);
  const arr = document.getElementById(id+'-arr');
  const open = body.classList.contains('open');
  // Close all in same domain
  const prefix = id.split('-c-')[0];
  document.querySelectorAll('[id^="'+prefix+'-c-"]').forEach(el => {
    el.classList.remove('open');
    const a = document.getElementById(el.id+'-arr');
    if(a) a.classList.remove('open');
  });
  if(!open){ body.classList.add('open'); arr.classList.add('open'); }
}

function show(domain, btn){
  document.querySelectorAll('.domain').forEach(d => d.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(domain).classList.add('active');
  btn.classList.add('active');
}
```

OK let me now write the complete final widget. I think I have everything planned out. Let me write it cleanly.

One issue I realize: I should double-check the SVG viewBox and element positions. Let me plan them carefully:

**Biology SVG** (viewBox 480x220):
- 7 stacked boxes, centered at x=240
- Widths: 300, 270, 240, 210, 180, 150, 120
- x positions: 90, 105, 120, 135, 150, 165, 180
- Heights: each 22px tall
- y positions: 10, 42, 74, 106, 138, 170, 202
- Lines between: y1 = prev_y+22, y2 = next_y (with arrow at y2-2)
- Total height: 202+22+8 = 232... so viewBox 480x235 ✓

**GNN SVG** (viewBox 480x160):
- Center node (Disease) at 240, 85, r=28
- Left nodes: Gene (90,50,r=22), Organ (90,120,r=22), Cell (200,20,r=22)
- Right node: Tissue (430,50,r=22)
- Aggregation box: 300-420, 62-108
- Edges: from node edges to center node (approx)
  - Gene (90,50): edge from (110,57) to (212,79) (center node at 240-28=212 edge)
  - Organ (90,120): edge from (110,114) to (212,90)
  - Cell (200,20): edge from (218,35) to (236,58)
  - Tissue (430,50): edge from (408,60) to (268,79)
  - Center to AGG box: (268,85) to (298,85)
- Check overlaps: Cell node (200,20,r=22) extends to x=222,y=42. Center node at 240,85 extends to 212,57 upward. Distance between Cell(200,20) and center(240,85): sqrt(40²+65²)=76 > 22+28=50. OK no overlap.
- Gene(90,50,r=22) extends to (112,72). Cell(200,20,r=22) extends to (178,42). No overlap.
- Aggregation box at (300

**User:**
continue

---

_Continue from here. The user's next message will follow._