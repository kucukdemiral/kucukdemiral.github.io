/**
 * Pyodide Interactive Code Runner — with matplotlib support
 * Browser-based Python execution for educational content.
 *
 * HTML structure:
 *   <div class="pyodide-runner">
 *     <script type="pyodide-setup">  ...hidden imports...  </script>
 *     <pre><code>  ...editable problem code...  </code></pre>
 *   </div>
 */
(function () {
    'use strict';

    var PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/';
    var pyodideReady = null;
    var matplotlibReady = false;

    var MPL_THEME =
        'import matplotlib\n' +
        'matplotlib.use("Agg")\n' +
        'import matplotlib.pyplot as plt\n' +
        'plt.rcParams.update({\n' +
        '    "figure.facecolor": "#1a2332",\n' +
        '    "axes.facecolor":   "#1a2332",\n' +
        '    "text.color":       "#D4E0ED",\n' +
        '    "axes.labelcolor":  "#D4E0ED",\n' +
        '    "xtick.color":      "#8899AA",\n' +
        '    "ytick.color":      "#8899AA",\n' +
        '    "axes.edgecolor":   "#3a4a5a",\n' +
        '    "grid.color":       "#2a3a4a",\n' +
        '    "grid.alpha":       0.5,\n' +
        '    "legend.facecolor": "#1a2332",\n' +
        '    "legend.edgecolor": "#3a4a5a",\n' +
        '    "legend.labelcolor":"#D4E0ED",\n' +
        '    "figure.dpi":       100,\n' +
        '})\n';

    /* ── Try to install matplotlib, never reject ───────────────── */
    function tryInstallMatplotlib(py) {
        /* Method 1: loadPackage (works in Chrome) */
        console.log('[pyodide] Attempting matplotlib via loadPackage...');
        return py.loadPackage(['matplotlib'])
            .then(function () {
                /* Verify it actually imported */
                try {
                    py.runPython('import matplotlib');
                    console.log('[pyodide] matplotlib loaded via loadPackage ✓');
                    return true;
                } catch (e) {
                    console.warn('[pyodide] loadPackage resolved but import failed:', e.message);
                    return false;
                }
            })
            .catch(function (err) {
                console.warn('[pyodide] loadPackage rejected:', err);
                return false;
            })
            .then(function (ok) {
                if (ok) return py;
                /* Method 2: micropip (fallback for Safari) */
                console.log('[pyodide] Trying micropip fallback...');
                return py.loadPackage(['micropip'])
                    .then(function () {
                        console.log('[pyodide] micropip package loaded, installing matplotlib...');
                        return py.runPythonAsync(
                            'import micropip\n' +
                            'await micropip.install("matplotlib")\n'
                        );
                    })
                    .then(function () {
                        try {
                            py.runPython('import matplotlib');
                            console.log('[pyodide] matplotlib loaded via micropip ✓');
                            return py;
                        } catch (e) {
                            console.warn('[pyodide] micropip install succeeded but import failed:', e.message);
                            return null;
                        }
                    })
                    .catch(function (err) {
                        console.warn('[pyodide] micropip fallback failed:', err);
                        return null;
                    });
            })
            .then(function (result) {
                if (result) {
                    py.runPython(MPL_THEME);
                    matplotlibReady = true;
                    console.log('[pyodide] matplotlib ready with dark theme');
                } else {
                    console.warn('[pyodide] matplotlib unavailable — plots will not render');
                }
                return py;
            });
    }

    /* ── Load Pyodide core ─────────────────────────────────────── */
    function ensurePyodide() {
        if (pyodideReady) return pyodideReady;

        pyodideReady = new Promise(function (resolve, reject) {
            showGlobalStatus('Loading Python environment...');

            var script = document.createElement('script');
            script.src = PYODIDE_CDN + 'pyodide.js';
            script.onload = function () {
                loadPyodide({ indexURL: PYODIDE_CDN })
                    .then(function (py) {
                        showGlobalStatus('Installing numpy + scipy...');
                        return py.loadPackage(['numpy', 'scipy']).then(function () { return py; });
                    })
                    .then(function (py) {
                        py.runPython(
                            'import numpy as np\n' +
                            'from scipy.optimize import minimize, minimize_scalar, linprog\n' +
                            'import io, sys, base64'
                        );
                        showGlobalStatus('Installing matplotlib...');
                        return tryInstallMatplotlib(py);
                    })
                    .then(function (py) {
                        hideGlobalStatus();
                        resolve(py);
                    })
                    .catch(function (err) { hideGlobalStatus(); reject(err); });
            };
            script.onerror = function () { hideGlobalStatus(); reject(new Error('Failed to load Pyodide')); };
            document.head.appendChild(script);
        });

        return pyodideReady;
    }

    /* ── Detect whether code uses matplotlib ───────────────────── */
    function codeNeedsPlots(code) {
        return /\bmatplotlib\b|\bplt\.|\bfig\s*,|\bsubplot|\.plot\s*\(|\.scatter\s*\(|\.hist\s*\(|\.bar\s*\(|\.stem\s*\(|\.contour|\.imshow|\.fill_between|\.errorbar/.test(code);
    }

    /* ── Flush matplotlib figures into stdout with markers ─────── */
    var FIG_MARKER = '__PYOFIG__';

    function flushFiguresToStdout(py) {
        if (!matplotlibReady) return;
        try {
            py.runPython(
                'for _fn in plt.get_fignums():\n' +
                '    _fig = plt.figure(_fn)\n' +
                '    _buf = io.BytesIO()\n' +
                '    _fig.savefig(_buf, format="png", dpi=100, bbox_inches="tight")\n' +
                '    _buf.seek(0)\n' +
                '    print("' + FIG_MARKER + '" + base64.b64encode(_buf.read()).decode())\n' +
                'plt.close("all")\n'
            );
        } catch (e) { /* ignore */ }
    }

    /* ── Parse stdout: split text lines from figure data URIs ──── */
    function parseOutput(stdout) {
        var textLines = [];
        var figs = [];
        if (!stdout) return { text: '', figs: figs };
        var lines = stdout.split('\n');
        for (var i = 0; i < lines.length; i++) {
            if (lines[i].indexOf(FIG_MARKER) === 0) {
                figs.push('data:image/png;base64,' + lines[i].substring(FIG_MARKER.length));
            } else {
                textLines.push(lines[i]);
            }
        }
        while (textLines.length > 0 && textLines[textLines.length - 1] === '') {
            textLines.pop();
        }
        return { text: textLines.join('\n'), figs: figs };
    }

    /* ── Global loading banner ─────────────────────────────────── */
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

    /* ── Run code and capture stdout + figures ─────────────────── */
    function finishRun(btnEl) {
        btnEl.disabled = false;
        btnEl.textContent = '\u25B6 Run';
    }

    function runCode(setup, userCode, textEl, figsEl, btnEl) {
        btnEl.disabled = true;
        btnEl.textContent = 'Running...';
        textEl.textContent = '';
        textEl.classList.remove('pyodide-error');
        figsEl.innerHTML = '';
        textEl.parentElement.style.display = 'block';

        var fullCode = (setup ? setup + '\n' : '') + userCode;
        var wantsPlots = codeNeedsPlots(fullCode);

        ensurePyodide()
            .then(function (py) {
                /* Redirect stdout/stderr */
                py.runPython(
                    '_pyodide_stdout = io.StringIO()\n' +
                    '_pyodide_stderr = io.StringIO()\n' +
                    'sys.stdout = _pyodide_stdout\n' +
                    'sys.stderr = _pyodide_stderr'
                );

                var hasError = false;

                try {
                    py.runPython(fullCode);
                } catch (e) {
                    hasError = true;
                    try {
                        var stderr = py.runPython('_pyodide_stderr.getvalue()');
                        textEl.textContent = stderr || e.message;
                    } catch (_) {
                        textEl.textContent = e.message;
                    }
                    textEl.classList.add('pyodide-error');
                }

                if (!hasError) {
                    /* Flush any open matplotlib figures into stdout */
                    if (wantsPlots) {
                        flushFiguresToStdout(py);
                    }

                    /* Collect all output */
                    var stdout = py.runPython('_pyodide_stdout.getvalue()');
                    var stderr = py.runPython('_pyodide_stderr.getvalue()');
                    if (stderr) stdout = (stdout || '') + '\n' + stderr;

                    /* Parse: separate figure data URIs from text */
                    var parsed = parseOutput(stdout);

                    textEl.textContent = parsed.text || (parsed.figs.length ? '' : '(no output)');

                    for (var i = 0; i < parsed.figs.length; i++) {
                        var img = document.createElement('img');
                        img.src = parsed.figs[i];
                        img.className = 'pyodide-fig';
                        figsEl.appendChild(img);
                    }
                }

                /* Restore streams */
                try {
                    py.runPython('sys.stdout = sys.__stdout__\nsys.stderr = sys.__stderr__');
                } catch (_) {}

                finishRun(btnEl);
            })
            .catch(function (err) {
                textEl.textContent = 'Error: ' + err.message;
                textEl.classList.add('pyodide-error');
                textEl.parentElement.style.display = 'block';
                finishRun(btnEl);
            });
    }

    /* ── Build interactive widget ──────────────────────────────── */
    function initRunner(container) {
        if (container.dataset.initialized) return;
        container.dataset.initialized = '1';

        var setupEl = container.querySelector('script[type="pyodide-setup"]');
        var setupCode = setupEl ? setupEl.textContent.replace(/^\n+|\n+$/g, '') : '';

        var codeEl = container.querySelector('pre code') || container.querySelector('pre');
        var editableCode = (codeEl ? codeEl.textContent : '').replace(/^\n+|\n+$/g, '');

        container.innerHTML = '';

        var label = document.createElement('div');
        label.className = 'pyodide-label';
        label.innerHTML = '<span class="pyodide-lang">\u25B6 Python</span> Interactive';
        container.appendChild(label);

        var textarea = document.createElement('textarea');
        textarea.className = 'pyodide-editor';
        textarea.spellcheck = false;
        textarea.value = editableCode;
        textarea.rows = Math.max(editableCode.split('\n').length + 1, 4);
        container.appendChild(textarea);

        var bar = document.createElement('div');
        bar.className = 'pyodide-btn-bar';

        var runBtn = document.createElement('button');
        runBtn.className = 'pyodide-run-btn';
        runBtn.textContent = '\u25B6 Run';
        runBtn.title = 'Execute code (Shift+Enter)';

        var resetBtn = document.createElement('button');
        resetBtn.className = 'pyodide-reset-btn';
        resetBtn.textContent = 'Reset';
        resetBtn.title = 'Reset to original code (MATLAB defaults)';

        bar.appendChild(runBtn);
        bar.appendChild(resetBtn);
        container.appendChild(bar);

        var outputWrap = document.createElement('div');
        outputWrap.className = 'pyodide-output-wrap';
        outputWrap.style.display = 'none';

        var figsEl = document.createElement('div');
        figsEl.className = 'pyodide-figs';
        outputWrap.appendChild(figsEl);

        var textEl = document.createElement('pre');
        textEl.className = 'pyodide-output';
        outputWrap.appendChild(textEl);

        container.appendChild(outputWrap);

        runBtn.addEventListener('click', function () {
            runCode(setupCode, textarea.value, textEl, figsEl, runBtn);
        });

        resetBtn.addEventListener('click', function () {
            textarea.value = editableCode;
            textarea.rows = Math.max(editableCode.split('\n').length + 1, 4);
            textEl.textContent = '';
            textEl.classList.remove('pyodide-error');
            figsEl.innerHTML = '';
            outputWrap.style.display = 'none';
        });

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

        textarea.addEventListener('input', function () {
            textarea.rows = Math.max(textarea.value.split('\n').length + 1, 4);
        });
    }

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
