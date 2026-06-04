# Interactive Block Diagram Reducer — Implementation Plan

## Context

Week 5 of Control Engineering 3 covers block diagram reduction — a core visual skill where students simplify complex interconnections of transfer functions into a single equivalent transfer function. Students struggle most with knowing *which rule to apply next* and *tracking the algebra through multiple steps*. This tool addresses both by letting students step through reductions one rule at a time, with highlights showing what's being combined and the resulting algebra displayed alongside.

## Architecture

Self-contained web app following the Bode Plot Studio pattern: vanilla JS + SVG, no frameworks, no build step.

```
block-diagram-reducer/
  index.html       # HTML shell (same structure as bode-plotter/index.html)
  app.js           # All logic: data model, reduction engine, SVG renderer, UI
  styles.css       # Standalone styling (same CSS variable palette as bode-plotter)

cs-notes/
  BlockDiagramReducer.html   # Thin iframe embed page (8 lines, same as BodePlotStudio.html)

control-systems.html         # Add sidebar sub-tab + chapters array entry after ch05
```

## Core Design

### Data Model

Block diagrams are directed graphs with typed nodes and edges:
- **Node types**: input, output, block (transfer function), sumjunc (summing junction), pickoff (branch point)
- **Edge properties**: from, to, sign (+/-), waypoints for feedback path routing
- **Transfer functions**: Symbolic expression trees (sym, mul, add, div, neg) — students see G₁, G₂, H₁ not numeric polynomials

### SVG Rendering

- Blocks: rounded rectangles with centered label text
- Summing junctions: circles with +/- signs at compass positions
- Pick-off points: small filled circles
- Signal lines: Manhattan-routed paths (horizontal/vertical only) with arrowheads
- Highlights: teal glow via SVG drop-shadow filter on active elements, fade (opacity 0.25) on others

### Reduction Engine

Detects and applies rules in priority order:
1. Merge adjacent summing junctions
2. Series: two adjacent blocks with no fanout between them → product
3. Parallel: two blocks from same source to same sumjunc → sum
4. Feedback: forward path + backward path returning to a sumjunc → G/(1±GH)

Produces a sequence of `ReductionStep` objects, each containing before/after diagram snapshots, the rule applied, highlighted elements, and the resulting symbolic expression.

### Preset Examples (from Chapter 5)

| # | Name | Key concept |
|---|---|---|
| 1 | Simple feedback loop | Basic G/(1+GH) |
| 2 | Multiple feedback paths | Three feedback blocks, sign handling |
| 3 | Non-touching loops | Mason's formula motivation |
| 4 | Two nested loops | Inner-loop-first strategy |
| 5 | Multiple forward paths | Cofactors in Mason's formula |
| 6 | DC Motor | Practical real-world application |

Node positions are manually specified to match textbook figures exactly.

### UI Layout

```
+----------------------------------------------------------+
| HEADER: "Block Diagram Reducer"          [Status Badge]  |
+----------------------------------------------------------+
| EXAMPLE TABS: [Simple] [Multi-FB] [Nested] [DC Motor].. |
+----------+-----------------------------------------------+
|          |                                                |
| CONTROLS |          SVG DIAGRAM CANVAS                    |
|          |          (main visualization)                  |
| [Reduce  |                                                |
|  Step]   |                                                |
| [Show    +------------------------------------------------+
|  All]    |                                                |
| [Reset]  |     ALGEBRA / STEP HISTORY PANEL               |
|          |  Step 1: Series — G₂G₃                         |
|          |  Step 2: Feedback — G/(1+GH)                    |
|          |  Step 3: ...                                    |
+----------+------------------------------------------------+
```

## Phased Build Order

### Phase 1: Static Rendering
- HTML/CSS shell following bode-plotter structure
- Data model (nodes, edges, diagrams)
- SVG renderer: draw all element types with Manhattan-routed edges
- One hardcoded preset (simple feedback G/(1+GH))
- No interactivity yet — just a correctly drawn diagram

### Phase 2: Reduction Engine + Step Controls
- Symbolic expression tree and formatter
- Series, parallel, feedback detection algorithms
- computeReductionSteps() function
- "Reduce Step" / "Reset" buttons
- Highlight system (glow active elements, fade others)
- Algebra panel showing step history

### Phase 3: All Presets + Animation
- Encode all 6 preset diagrams with correct positions
- Example selector tabs/cards
- CSS transitions for smooth step animations
- "Show Full Solution" button

### Phase 4: Integration + Polish
- Create cs-notes/BlockDiagramReducer.html embed page
- Add sidebar sub-tab after Chapter 5 in control-systems.html
- Responsive layout for mobile
- Keyboard shortcuts (→ next step, R reset)

### Phase 5 (future): Custom Builder Mode
- Element palette, click-to-place, click-to-connect
- Grid snap layout
- Property editor for labels and signs
- Apply reduction engine to custom diagrams

## Files to Modify

- `control-systems.html` — add `toc-sub` entry after ch05, add chapters array entry
- Create `block-diagram-reducer/index.html`, `app.js`, `styles.css`
- Create `cs-notes/BlockDiagramReducer.html`

## Verification

1. Open `block-diagram-reducer/index.html` directly in browser — all presets should render correctly
2. Click "Reduce Step" through each preset — verify algebra matches Chapter 5 solutions
3. Verify the embed page loads correctly via `control-systems.html#bd-reducer`
4. Test embed mode (`?embed=1`) for Blackboard iframe integration
5. Test on mobile viewport (375px width)
