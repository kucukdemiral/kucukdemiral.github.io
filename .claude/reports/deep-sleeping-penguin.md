# Plan: Control Systems Teaching Page

## Context

The MPC teaching page (`model-predictive-control.html`) provides a two-column interactive textbook (sidebar TOC + dynamic chapter loading) with MathJax equations, Pyodide-powered Python code execution, and custom styled environments. The current `control-systems.html` page only embeds two iframe tools (PID simulator, Bode plotter).

The goal is to build an equivalent interactive textbook page for the Control Systems module, converting 12 LaTeX chapters (~796KB total) from the CE3 lecture notes into HTML. MATLAB code (85 listings) must be converted to runnable Python, with MATLAB kept in display-only mode. The PID simulator and Bode plotter tools should be embedded under their relevant chapters (Ch8: PID, Ch9: Frequency Domain) rather than on a standalone page. No coursework content should be included.

## Source Material

- **LaTeX chapters**: `/Users/ibrahim_kucukdemiral/Library/CloudStorage/OneDrive-GLASGOWCALEDONIANUNIVERSITY/Glasgow/GCU/Modules/CE3/Lecture notes_new/Chapters/Chapter01.tex` through `Chapter12.tex`
- **Graphics**: 121 PDF + 21 PNG files in `graphics/` folder, plus 112 inline TikZ/circuitikz diagrams
- **Compiled PDF**: `Main.pdf` (23.4 MB) with all rendered content
- **Custom environments**: summarybox (29), examplebox (51), example (83), remark (57), definition (32), notebox (11), alertbox (10), theorem (8)

## Chapter Map (lecture notes → web)

| # | LaTeX File | Web Title | Key Content |
|---|-----------|-----------|-------------|
| 1 | Chapter01.tex (62KB) | Introduction to Feedback Control | Open/closed-loop, block diagrams, feedback structure |
| 2 | Chapter02.tex (61KB) | Laplace Transform | Transform pairs, PFE, IVT/FVT, transfer functions |
| 3 | Chapter03.tex (100KB) | Modelling Dynamical Systems | Electrical/mechanical/thermal systems, transfer functions |
| 4 | Chapter04.tex (70KB) | State-Space Modelling | State equations, A/B/C/D matrices, canonical forms |
| 5 | Chapter05.tex (44KB) | Block Diagrams | Reduction rules, Mason's formula, Simulink |
| 6 | Chapter06.tex (101KB) | Time Response | 1st/2nd order response, overshoot, settling time, steady-state error |
| 7 | Chapter07.tex (54KB) | Stability | Poles, Routh-Hurwitz, eigenvalues |
| 8 | Chapter08.tex (52KB) | PID Control | P/I/D actions, Ziegler-Nichols, **embed PID simulator** |
| 9 | Chapter09.tex (130KB) | Frequency-Domain Analysis | Bode plots, margins, lead/lag compensation, **embed Bode plotter** |
| 10 | Chapter10.tex (36KB) | Controllability & Observability | Rank tests, canonical forms |
| 11 | Chapter11.tex (46KB) | State Feedback Control | Pole placement, Ackermann, integral action |
| 12 | Chapter12.tex (40KB) | Observer Design | Luenberger observer, separation principle |

## Architecture (mirroring MPC page)

### Files to create/modify

- **`control-systems.html`** — Rewrite as two-column layout (sidebar + chapter body), matching `model-predictive-control.html` pattern
- **`cs-notes/`** — New directory for chapter HTML files (`Chapter01.html` through `Chapter12.html`)
- **`cs-notes/figures/`** — Rendered figures (PNG from TikZ compilation + converted PDF graphics)
- **`css/control-systems.css`** — Rewrite with MPC-style chapter styling (reuse environment box classes, code block styles, responsive layout from `model-predictive-control.html` inline styles)

### Existing files to reuse

- **`js/pyodide-runner.js`** — Python code execution (already built for MPC page)
- **`css/styles.css`** — Base styles, CSS variables, header/footer
- **MathJax CDN** — Same configuration as MPC page
- **`bode-plotter/`** — Embed in Chapter 9 via iframe or inline
- **PID simulator** — Embed in Chapter 8 via iframe

## Implementation Steps

### Step 1: Infrastructure

1. Create `cs-notes/` and `cs-notes/figures/` directories
2. Rewrite `control-systems.html` with the MPC two-column layout:
   - Sidebar with 12 chapter links + PDF download link
   - Chapter body with dynamic loading via `fetch()`
   - Chapter navigation (prev/next buttons)
   - MathJax configuration (same as MPC)
   - Load `js/pyodide-runner.js` for Python execution
3. Update `css/control-systems.css` with MPC-style environment boxes (defnbox, examplebox, notebox, alertbox, summarybox, remark), code block styling, responsive sidebar

### Step 2: Figure Preparation

1. **PDF graphics → PNG**: Use `pdftoppm` to convert the 121 PDF files in `graphics/` to PNG, save to `cs-notes/figures/`
2. **Inline TikZ → PNG**: For each chapter's inline TikZ diagrams (112 total):
   - Extract TikZ code from LaTeX source
   - Create standalone LaTeX documents with required packages (tikz, circuitikz, pgfplots)
   - Compile with `pdflatex` → convert to PNG with `pdftoppm`/`sips`
   - Save with descriptive names to `cs-notes/figures/`
3. **Copy existing PNGs**: Copy the 21 PNG files directly

### Step 3: Chapter Conversion (per chapter, pandoc + post-processing)

For each chapter:

1. **Pandoc convert**: `pandoc ChapterXX.tex -f latex -t html --mathjax --wrap=none`
2. **Post-process HTML**:
   - Fix figure paths: `src="figures/..."` → relative to `cs-notes/`
   - Replace empty TikZ `<figure>` elements with `<img>` tags pointing to pre-rendered PNGs
   - Style custom environment divs (pandoc preserves class names: `summarybox`, `examplebox`, `definition`, `remark`, `alertbox`, `notebox`)
   - Fix glossary references (`\gls{...}` → plain text or tooltips)
   - Fix cross-references between chapters
3. **Convert MATLAB to Python**:
   - Each MATLAB `lstlisting` → dual code block:
     - **Python** (active): `<div class="pyodide-editor">` with `python-control`, `numpy`, `scipy`, `matplotlib`
     - **MATLAB** (passive): `<pre class="matlab">` display-only block
   - Key library mappings:
     - `tf()` → `control.tf()`
     - `step()` → `control.step_response()`
     - `bode()` → `control.bode_plot()`
     - `place()` → `control.place()`
     - `ctrb()/obsv()` → `control.ctrb()/control.obsv()`
     - `roots()` → `np.roots()`
     - `eig()` → `np.linalg.eig()`
4. **Save** as `cs-notes/ChapterXX.html`

### Step 4: Tool Integration

1. **Chapter 8 (PID)**: Add an interactive section at the end titled "Interactive PID Simulator" with the PID simulator embedded via iframe (`https://kucukdemiral.github.io/pid-simulation/`)
2. **Chapter 9 (Frequency Domain)**: Add an interactive section titled "Interactive Bode Sketch Studio" embedding `bode-plotter/index.html` via iframe
3. Both tools should appear as collapsible/expandable sections within the chapter content

### Step 5: Testing & Polish

1. Serve locally and test all 12 chapters load correctly
2. Verify MathJax renders all equations (inline and display)
3. Test Python code execution via Pyodide for representative examples
4. Test PID and Bode tools embedded correctly
5. Test responsive design on mobile viewport
6. Verify no coursework content is present

## Scope & Sequencing

This is a large project (~796KB of LaTeX, 112 TikZ diagrams, 85 MATLAB listings). The recommended approach is to:

1. Build infrastructure first (Step 1) — this establishes the framework
2. Process figures in batch (Step 2) — parallelizable
3. Convert chapters sequentially, starting with smaller ones (Ch10 → Ch5 → Ch7 → Ch11 → Ch12 → Ch8 → Ch1 → Ch2 → Ch4 → Ch3 → Ch6 → Ch9)
4. Integrate tools (Step 4) — quick, depends on Ch8 and Ch9 being done
5. Final testing pass (Step 5)

## Verification

- Open `control-systems.html` in browser, navigate through all 12 chapters
- Equations should render properly (both inline and display math)
- Python code blocks should be editable and runnable with output/plots
- MATLAB code should appear in read-only dark-themed blocks
- PID simulator accessible within Chapter 8
- Bode plotter accessible within Chapter 9
- Figures should display correctly (no broken images)
- Sidebar navigation highlights active chapter
- Mobile responsive layout works
