/**
 * Pyodide Interactive Code Runner
 * Browser-based Python execution with matplotlib plotting support.
 * Supports hidden setup/plot sections and locked preambles.
 *
 * HTML structure:
 *   <div class="pyodide-runner">
 *     <script type="pyodide-setup">  ...hidden imports/setup...  </script>
 *     <pre><code>  ...editable problem code...  </code></pre>
 *     <script type="pyodide-plot">   ...hidden plotting code...  </script>
 *   </div>
 */
(function () {
    'use strict';

    var PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/';
    var pyodideReady = null;
    var pyodideInstance = null;

    // ── Pyodide loader (lazy) ──────────────────────────────────────────
    function ensurePyodide() {
        if (pyodideReady) return pyodideReady;

        pyodideReady = new Promise(function (resolve, reject) {
            showGlobalStatus('Loading Python + NumPy + SciPy + Matplotlib...');

            var script = document.createElement('script');
            script.src = PYODIDE_CDN + 'pyodide.js';
            script.onload = function () {
                loadPyodide({ indexURL: PYODIDE_CDN })
                    .then(function (py) {
                        pyodideInstance = py;
                        showGlobalStatus('Installing packages (numpy, scipy, matplotlib)...');
                        return py.loadPackage(['numpy', 'scipy', 'matplotlib']);
                    })
                    .then(function () {
                        // Global preamble: imports + matplotlib capture setup
                        pyodideInstance.runPython(GLOBAL_PREAMBLE);
                        hideGlobalStatus();
                        resolve(pyodideInstance);
                    })
                    .catch(function (err) {
                        hideGlobalStatus();
                        reject(err);
                    });
            };
            script.onerror = function () {
                hideGlobalStatus();
                reject(new Error('Failed to load Pyodide script'));
            };
            document.head.appendChild(script);
        });

        return pyodideReady;
    }

    // Python code injected once on first load
    var GLOBAL_PREAMBLE = [
        'import numpy as np',
        'from scipy.optimize import minimize, minimize_scalar, linprog',
        'import io, sys, base64',
        '',
        '# Matplotlib with non-interactive backend',
        'import matplotlib',
        "matplotlib.use('Agg')",
        'import matplotlib.pyplot as plt',
        'from matplotlib.patches import Polygon as MplPolygon',
        'from matplotlib.collections import PatchCollection',
        '',
        '# Academic-style defaults',
        'plt.rcParams.update({',
        "    'figure.figsize': (7, 4.5),",
        "    'figure.dpi': 150,",
        "    'font.size': 10,",
        "    'axes.grid': True,",
        "    'axes.axisbelow': True,",
        "    'grid.alpha': 0.25,",
        "    'grid.linestyle': '--',",
        "    'axes.spines.top': False,",
        "    'axes.spines.right': False,",
        "    'figure.facecolor': 'white',",
        "    'axes.facecolor': 'white',",
        "    'axes.edgecolor': '#333333',",
        "    'text.color': '#222222',",
        "    'axes.labelcolor': '#222222',",
        "    'xtick.color': '#444444',",
        "    'ytick.color': '#444444',",
        '})',
        '',
        '# Plot capture list (read by JS after execution)',
        '_pyodide_plots = []',
        '',
        'def _capture_current_figures():',
        '    """Save all open figures as base64 PNG and close them."""',
        '    global _pyodide_plots',
        '    for num in plt.get_fignums():',
        '        fig = plt.figure(num)',
        '        buf = io.BytesIO()',
        '        fig.savefig(buf, format="png", dpi=150,',
        '                    bbox_inches="tight", facecolor="white", edgecolor="none")',
        '        buf.seek(0)',
        '        _pyodide_plots.append(base64.b64encode(buf.read()).decode("utf-8"))',
        '        buf.close()',
        '    plt.close("all")',
    ].join('\n');

    // ── Global loading banner ──────────────────────────────────────────
    var statusBanner = null;

    function showGlobalStatus(msg) {
        if (!statusBanner) {
            statusBanner = document.createElement('div');
            statusBanner.className = 'pyodide-global-status';
            document.body.appendChild(statusBanner);
        }
        statusBanner.textContent = msg;
        statusBanner.style.display = 'block';
    }

    function hideGlobalStatus() {
        if (statusBanner) statusBanner.style.display = 'none';
    }

    // ── Run code: setup + user + plot, capture stdout + figures ─────────
    function runCode(setup, userCode, plotCode, outputEl, plotEl, btnEl) {
        btnEl.disabled = true;
        btnEl.textContent = 'Running...';
        outputEl.textContent = '';
        outputEl.style.display = 'none';
        plotEl.innerHTML = '';
        plotEl.style.display = 'none';

        ensurePyodide()
            .then(function (py) {
                // Reset state
                py.runPython([
                    '_pyodide_plots = []',
                    'plt.close("all")',
                    '_pyodide_stdout = io.StringIO()',
                    '_pyodide_stderr = io.StringIO()',
                    'sys.stdout = _pyodide_stdout',
                    'sys.stderr = _pyodide_stderr'
                ].join('\n'));

                // Combine: setup + user code + plot code
                var fullCode = '';
                if (setup) fullCode += setup + '\n';
                fullCode += userCode;
                if (plotCode) fullCode += '\n' + plotCode + '\n_capture_current_figures()';

                var hasError = false;
                try {
                    py.runPython(fullCode);
                } catch (e) {
                    hasError = true;
                    var stderr = py.runPython('_pyodide_stderr.getvalue()');
                    outputEl.textContent = stderr || e.message;
                    outputEl.classList.add('pyodide-error');
                    outputEl.style.display = 'block';
                }

                if (!hasError) {
                    var stdout = py.runPython('_pyodide_stdout.getvalue()');
                    var stderr = py.runPython('_pyodide_stderr.getvalue()');

                    var text = '';
                    if (stdout) text += stdout;
                    if (stderr) text += (text ? '\n' : '') + stderr;
                    if (text) {
                        outputEl.textContent = text;
                        outputEl.classList.remove('pyodide-error');
                        outputEl.style.display = 'block';
                    }
                }

                // Render plots
                var plotsJson = py.runPython('import json; json.dumps(_pyodide_plots)');
                var plots = JSON.parse(plotsJson);
                if (plots.length > 0) {
                    plotEl.style.display = 'block';
                    for (var i = 0; i < plots.length; i++) {
                        var img = document.createElement('img');
                        img.src = 'data:image/png;base64,' + plots[i];
                        img.className = 'pyodide-plot-img';
                        img.alt = 'Plot output';
                        plotEl.appendChild(img);
                    }
                }

                // Restore streams
                py.runPython('sys.stdout = sys.__stdout__\nsys.stderr = sys.__stderr__');
                btnEl.disabled = false;
                btnEl.textContent = '\u25B6 Run';
            })
            .catch(function (err) {
                outputEl.textContent = 'Error: ' + err.message;
                outputEl.classList.add('pyodide-error');
                outputEl.style.display = 'block';
                btnEl.disabled = false;
                btnEl.textContent = '\u25B6 Run';
            });
    }

    // ── Build interactive widget ───────────────────────────────────────
    function initRunner(container) {
        if (container.dataset.initialized) return;
        container.dataset.initialized = '1';

        // Extract hidden setup code
        var setupEl = container.querySelector('script[type="pyodide-setup"]');
        var setupCode = setupEl ? setupEl.textContent.replace(/^\n+|\n+$/g, '') : '';

        // Extract visible/editable code
        var codeEl = container.querySelector('pre code') || container.querySelector('pre');
        var editableCode = (codeEl ? codeEl.textContent : '').replace(/^\n+|\n+$/g, '');

        // Extract hidden plot code
        var plotScriptEl = container.querySelector('script[type="pyodide-plot"]');
        var plotCode = plotScriptEl ? plotScriptEl.textContent.replace(/^\n+|\n+$/g, '') : '';

        var hasPlot = plotCode.length > 0;

        // Rebuild container
        container.innerHTML = '';

        // Label
        var label = document.createElement('div');
        label.className = 'pyodide-label';
        label.innerHTML = '<span class="pyodide-lang">\u25B6 Python</span> Interactive' +
            (hasPlot ? ' &middot; with plot' : '');
        container.appendChild(label);

        // Editable code area
        var textarea = document.createElement('textarea');
        textarea.className = 'pyodide-editor';
        textarea.spellcheck = false;
        textarea.value = editableCode;
        var lines = editableCode.split('\n').length;
        textarea.rows = Math.max(lines + 1, 4);
        container.appendChild(textarea);

        // Button bar
        var bar = document.createElement('div');
        bar.className = 'pyodide-btn-bar';

        var runBtn = document.createElement('button');
        runBtn.className = 'pyodide-run-btn';
        runBtn.textContent = '\u25B6 Run';
        runBtn.title = 'Execute code (Shift+Enter)';

        var resetBtn = document.createElement('button');
        resetBtn.className = 'pyodide-reset-btn';
        resetBtn.textContent = 'Reset';
        resetBtn.title = 'Reset to original code';

        bar.appendChild(runBtn);
        bar.appendChild(resetBtn);
        container.appendChild(bar);

        // Output area (text)
        var output = document.createElement('pre');
        output.className = 'pyodide-output';
        output.style.display = 'none';
        container.appendChild(output);

        // Plot area (images)
        var plotArea = document.createElement('div');
        plotArea.className = 'pyodide-plot-area';
        plotArea.style.display = 'none';
        container.appendChild(plotArea);

        // Events
        runBtn.addEventListener('click', function () {
            runCode(setupCode, textarea.value, plotCode, output, plotArea, runBtn);
        });

        resetBtn.addEventListener('click', function () {
            textarea.value = editableCode;
            textarea.rows = Math.max(editableCode.split('\n').length + 1, 4);
            output.textContent = '';
            output.style.display = 'none';
            plotArea.innerHTML = '';
            plotArea.style.display = 'none';
        });

        // Shift+Enter shortcut
        textarea.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && e.shiftKey) {
                e.preventDefault();
                runBtn.click();
            }
            if (e.key === 'Tab') {
                e.preventDefault();
                var start = textarea.selectionStart;
                var end = textarea.selectionEnd;
                textarea.value = textarea.value.substring(0, start) +
                    '    ' + textarea.value.substring(end);
                textarea.selectionStart = textarea.selectionEnd = start + 4;
            }
        });

        // Auto-resize
        textarea.addEventListener('input', function () {
            textarea.rows = Math.max(textarea.value.split('\n').length + 1, 4);
        });
    }

    // ── Scan & init ────────────────────────────────────────────────────
    function initAllRunners(root) {
        var runners = (root || document).querySelectorAll('.pyodide-runner');
        for (var i = 0; i < runners.length; i++) {
            initRunner(runners[i]);
        }
    }

    window.PyodideRunner = {
        initAll: initAllRunners,
        init: initRunner
    };
})();
