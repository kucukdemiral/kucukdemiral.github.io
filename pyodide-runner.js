/**
 * Pyodide Interactive Code Runner
 * Provides browser-based Python execution for educational content.
 * Uses Pyodide (CPython compiled to WebAssembly) with NumPy & SciPy.
 */
(function () {
    'use strict';

    var PYODIDE_CDN = 'https://cdn.jsdelivr.net/pyodide/v0.27.5/full/';
    var pyodideReady = null;   // Promise that resolves to the pyodide instance
    var pyodideInstance = null;

    // ── Pyodide loader (lazy – only triggered on first "Run") ──────────
    function ensurePyodide() {
        if (pyodideReady) return pyodideReady;

        pyodideReady = new Promise(function (resolve, reject) {
            // Show global loading banner
            showGlobalStatus('Loading Python environment (first run may take a few seconds)...');

            var script = document.createElement('script');
            script.src = PYODIDE_CDN + 'pyodide.js';
            script.onload = function () {
                loadPyodide({ indexURL: PYODIDE_CDN })
                    .then(function (py) {
                        pyodideInstance = py;
                        return py.loadPackage(['numpy', 'scipy']);
                    })
                    .then(function () {
                        // Pre-import commonly used modules
                        pyodideInstance.runPython([
                            'import numpy as np',
                            'from scipy.optimize import minimize, minimize_scalar, linprog',
                            'import io, sys'
                        ].join('\n'));
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
    function runCode(code, outputEl, btnEl) {
        btnEl.disabled = true;
        btnEl.textContent = 'Running...';
        outputEl.textContent = '';
        outputEl.style.display = 'block';

        ensurePyodide()
            .then(function (py) {
                // Redirect stdout/stderr
                py.runPython([
                    '_pyodide_stdout = io.StringIO()',
                    '_pyodide_stderr = io.StringIO()',
                    'sys.stdout = _pyodide_stdout',
                    'sys.stderr = _pyodide_stderr'
                ].join('\n'));

                try {
                    py.runPython(code);
                } catch (e) {
                    // Python exception – show it
                    var stderr = py.runPython('_pyodide_stderr.getvalue()');
                    outputEl.textContent = stderr || e.message;
                    outputEl.classList.add('pyodide-error');
                    btnEl.disabled = false;
                    btnEl.textContent = 'Run';
                    restoreStreams(py);
                    return;
                }

                var stdout = py.runPython('_pyodide_stdout.getvalue()');
                var stderr = py.runPython('_pyodide_stderr.getvalue()');
                restoreStreams(py);

                var result = '';
                if (stdout) result += stdout;
                if (stderr) result += (result ? '\n' : '') + stderr;
                outputEl.textContent = result || '(no output)';
                outputEl.classList.remove('pyodide-error');
                btnEl.disabled = false;
                btnEl.textContent = 'Run';
            })
            .catch(function (err) {
                outputEl.textContent = 'Error: ' + err.message;
                outputEl.classList.add('pyodide-error');
                btnEl.disabled = false;
                btnEl.textContent = 'Run';
            });
    }

    function restoreStreams(py) {
        py.runPython([
            'sys.stdout = sys.__stdout__',
            'sys.stderr = sys.__stderr__'
        ].join('\n'));
    }

    // ── Build interactive widget from a <div class="pyodide-runner"> ──
    function initRunner(container) {
        if (container.dataset.initialized) return;
        container.dataset.initialized = '1';

        var codeEl = container.querySelector('code') || container.querySelector('pre');
        var initialCode = (codeEl ? codeEl.textContent : '').replace(/^\n+|\n+$/g, '');

        // Clear container and rebuild
        container.innerHTML = '';

        // Label
        var label = document.createElement('div');
        label.className = 'pyodide-label';
        label.innerHTML = '<span class="pyodide-lang">Python</span> Interactive';
        container.appendChild(label);

        // Editable code area
        var textarea = document.createElement('textarea');
        textarea.className = 'pyodide-editor';
        textarea.spellcheck = false;
        textarea.value = initialCode;
        // Auto-size rows
        var lines = initialCode.split('\n').length;
        textarea.rows = Math.max(lines + 1, 4);
        container.appendChild(textarea);

        // Button bar
        var bar = document.createElement('div');
        bar.className = 'pyodide-btn-bar';

        var runBtn = document.createElement('button');
        runBtn.className = 'pyodide-run-btn';
        runBtn.textContent = 'Run';
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
            runCode(textarea.value, output, runBtn);
        });

        resetBtn.addEventListener('click', function () {
            textarea.value = initialCode;
            textarea.rows = Math.max(initialCode.split('\n').length + 1, 4);
            output.textContent = '';
            output.style.display = 'none';
        });

        // Shift+Enter shortcut
        textarea.addEventListener('keydown', function (e) {
            if (e.key === 'Enter' && e.shiftKey) {
                e.preventDefault();
                runBtn.click();
            }
            // Tab inserts spaces
            if (e.key === 'Tab') {
                e.preventDefault();
                var start = textarea.selectionStart;
                var end = textarea.selectionEnd;
                textarea.value = textarea.value.substring(0, start) + '    ' + textarea.value.substring(end);
                textarea.selectionStart = textarea.selectionEnd = start + 4;
            }
        });

        // Auto-resize on input
        textarea.addEventListener('input', function () {
            var lc = textarea.value.split('\n').length;
            textarea.rows = Math.max(lc + 1, 4);
        });
    }

    // ── Scan for runner blocks and initialise them ─────────────────────
    function initAllRunners(root) {
        var runners = (root || document).querySelectorAll('.pyodide-runner');
        for (var i = 0; i < runners.length; i++) {
            initRunner(runners[i]);
        }
    }

    // Expose globally
    window.PyodideRunner = {
        initAll: initAllRunners,
        init: initRunner
    };
})();
