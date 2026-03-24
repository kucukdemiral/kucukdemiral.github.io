/**
 * Pyodide Interactive Code Runner
 * Browser-based Python execution for educational content.
 * Supports hidden setup preambles (locked, not shown to user).
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
                            'import io, sys'
                        );
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

    // ── Run code and capture stdout / stderr ───────────────────────────
    function finishRun(btnEl) {
        btnEl.disabled = false;
        btnEl.textContent = '\u25B6 Run';
    }

    function runCode(setup, userCode, outputEl, btnEl) {
        btnEl.disabled = true;
        btnEl.textContent = 'Running...';
        outputEl.textContent = '';
        outputEl.classList.remove('pyodide-error');
        outputEl.style.display = 'block';

        ensurePyodide()
            .then(function (py) {
                py.runPython(
                    '_pyodide_stdout = io.StringIO()\n' +
                    '_pyodide_stderr = io.StringIO()\n' +
                    'sys.stdout = _pyodide_stdout\n' +
                    'sys.stderr = _pyodide_stderr'
                );

                var fullCode = (setup ? setup + '\n' : '') + userCode;
                var hasError = false;

                try {
                    py.runPython(fullCode);
                } catch (e) {
                    hasError = true;
                    try {
                        var stderr = py.runPython('_pyodide_stderr.getvalue()');
                        outputEl.textContent = stderr || e.message;
                    } catch (_) {
                        outputEl.textContent = e.message;
                    }
                    outputEl.classList.add('pyodide-error');
                }

                if (!hasError) {
                    var stdout = py.runPython('_pyodide_stdout.getvalue()');
                    var stderr = py.runPython('_pyodide_stderr.getvalue()');
                    var text = '';
                    if (stdout) text += stdout;
                    if (stderr) text += (text ? '\n' : '') + stderr;
                    outputEl.textContent = text || '(no output)';
                }

                // Always restore streams
                try {
                    py.runPython('sys.stdout = sys.__stdout__\nsys.stderr = sys.__stderr__');
                } catch (_) {}
                finishRun(btnEl);
            })
            .catch(function (err) {
                outputEl.textContent = 'Error: ' + err.message;
                outputEl.classList.add('pyodide-error');
                outputEl.style.display = 'block';
                finishRun(btnEl);
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

        // Rebuild container
        container.innerHTML = '';

        // Label
        var label = document.createElement('div');
        label.className = 'pyodide-label';
        label.innerHTML = '<span class="pyodide-lang">\u25B6 Python</span> Interactive';
        container.appendChild(label);

        // Editable code area
        var textarea = document.createElement('textarea');
        textarea.className = 'pyodide-editor';
        textarea.spellcheck = false;
        textarea.value = editableCode;
        textarea.rows = Math.max(editableCode.split('\n').length + 1, 4);
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

        // Output area
        var output = document.createElement('pre');
        output.className = 'pyodide-output';
        output.style.display = 'none';
        container.appendChild(output);

        // Events
        runBtn.addEventListener('click', function () {
            runCode(setupCode, textarea.value, output, runBtn);
        });

        resetBtn.addEventListener('click', function () {
            textarea.value = editableCode;
            textarea.rows = Math.max(editableCode.split('\n').length + 1, 4);
            output.textContent = '';
            output.classList.remove('pyodide-error');
            output.style.display = 'none';
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
