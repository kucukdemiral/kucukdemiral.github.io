/**
 * Interactive Scalar Kalman Filter Demo
 * Pure JS/Canvas GUI — no code shown, no Pyodide needed.
 *
 * HTML hook:  <div class="kalman-demo"></div>
 * Init call:  KalmanDemo.initAll(root)
 */
(function () {
    'use strict';

    /* ── Gaussian random (Box-Muller) ─────────────────────── */
    function randn() {
        var u = 0, v = 0;
        while (u === 0) u = Math.random();
        while (v === 0) v = Math.random();
        return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }

    function fmt(x) { return x.toFixed(4); }

    /* ── Fresh state ──────────────────────────────────────── */
    function freshState() {
        return {
            k: 0, x_true: 0, x_hat: 0, P: 1.0,
            ks: [0], xs_true: [0], ys: [], xs_hat: [0],
            band_hi: [2], band_lo: [-2],
            cur: null
        };
    }

    /* ── One Kalman step (scalar: A=1, C=1) ───────────────── */
    function stepKF(st, W, V) {
        st.k++;
        var w = Math.sqrt(W) * randn();
        var vn = Math.sqrt(V) * randn();
        st.x_true = st.x_true + w;
        var y = st.x_true + vn;

        var x_pred = st.x_hat;
        var P_pred = st.P + W;
        var S = P_pred + V;
        var L = P_pred / S;
        var innov = y - x_pred;
        st.x_hat = x_pred + L * innov;
        st.P = (1 - L) * P_pred;

        st.cur = {
            k: st.k, y: y,
            x_pred: x_pred, P_pred: P_pred,
            S: S, L: L, innov: innov,
            x_hat: st.x_hat, P: st.P
        };

        st.ks.push(st.k);
        st.xs_true.push(st.x_true);
        st.ys.push(y);
        st.xs_hat.push(st.x_hat);
        var s2 = 2 * Math.sqrt(st.P);
        st.band_hi.push(st.x_hat + s2);
        st.band_lo.push(st.x_hat - s2);
        return st.cur;
    }

    /* ── Nice grid step ───────────────────────────────────── */
    function niceStep(lo, hi, target) {
        var range = hi - lo;
        if (range <= 0) return 1;
        var rough = range / target;
        var mag = Math.pow(10, Math.floor(Math.log10(rough)));
        var norm = rough / mag;
        if (norm < 1.5) return mag;
        if (norm < 3.5) return 2 * mag;
        if (norm < 7.5) return 5 * mag;
        return 10 * mag;
    }

    /* ── Canvas plot ──────────────────────────────────────── */
    function drawPlot(canvas, st) {
        var ctx = canvas.getContext('2d');
        var dpr = window.devicePixelRatio || 1;
        var CW = canvas.clientWidth, CH = canvas.clientHeight;
        canvas.width = CW * dpr;
        canvas.height = CH * dpr;
        ctx.scale(dpr, dpr);

        ctx.fillStyle = '#0A1219';
        ctx.fillRect(0, 0, CW, CH);

        var m = { t: 20, r: 18, b: 36, l: 50 };
        var pw = CW - m.l - m.r;
        var ph = CH - m.t - m.b;

        if (st.ks.length < 2) {
            ctx.fillStyle = '#556677';
            ctx.font = '13px "Source Sans 3", sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Click \u25B6 Step or \u25B6\u25B6 Auto to begin', CW / 2, CH / 2);
            return;
        }

        /* ranges */
        var xMin = 0, xMax = Math.max(st.k, 10);
        var all = st.xs_true.concat(st.ys, st.band_hi, st.band_lo);
        var yMin = Infinity, yMax = -Infinity;
        for (var i = 0; i < all.length; i++) {
            if (all[i] != null && isFinite(all[i])) {
                if (all[i] < yMin) yMin = all[i];
                if (all[i] > yMax) yMax = all[i];
            }
        }
        var yPad = Math.max((yMax - yMin) * 0.12, 0.5);
        yMin -= yPad; yMax += yPad;

        function sx(v) { return m.l + (v - xMin) / (xMax - xMin) * pw; }
        function sy(v) { return m.t + (1 - (v - yMin) / (yMax - yMin)) * ph; }

        /* grid */
        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        var ys = niceStep(yMin, yMax, 5);
        ctx.font = '10px "SF Mono", "Fira Code", monospace';
        ctx.fillStyle = '#556677';
        ctx.textAlign = 'right';
        for (var v = Math.ceil(yMin / ys) * ys; v <= yMax; v += ys) {
            var yy = sy(v);
            ctx.beginPath(); ctx.moveTo(m.l, yy); ctx.lineTo(m.l + pw, yy); ctx.stroke();
            ctx.fillText(v.toFixed(1), m.l - 6, yy + 3);
        }

        var xs = Math.max(1, Math.ceil(xMax / 10));
        ctx.textAlign = 'center';
        for (var t = 0; t <= xMax; t += xs) {
            ctx.fillText(t, sx(t), m.t + ph + 16);
        }

        /* axes */
        ctx.strokeStyle = 'rgba(255,255,255,0.18)';
        ctx.beginPath();
        ctx.moveTo(m.l, m.t); ctx.lineTo(m.l, m.t + ph); ctx.lineTo(m.l + pw, m.t + ph);
        ctx.stroke();

        /* axis labels */
        ctx.fillStyle = '#778899';
        ctx.font = '11px "Source Sans 3", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Time step k', m.l + pw / 2, CH - 3);
        ctx.save();
        ctx.translate(13, m.t + ph / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('State value', 0, 0);
        ctx.restore();

        /* ±2σ band */
        ctx.fillStyle = 'rgba(231,76,60,0.12)';
        ctx.beginPath();
        ctx.moveTo(sx(0), sy(st.band_hi[0]));
        for (var i = 1; i < st.ks.length; i++) ctx.lineTo(sx(st.ks[i]), sy(st.band_hi[i]));
        for (var i = st.ks.length - 1; i >= 0; i--) ctx.lineTo(sx(st.ks[i]), sy(st.band_lo[i]));
        ctx.closePath();
        ctx.fill();

        /* true state (white/cream line) */
        ctx.strokeStyle = '#D4E0ED';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (var i = 0; i < st.ks.length; i++) {
            if (i === 0) ctx.moveTo(sx(st.ks[i]), sy(st.xs_true[i]));
            else ctx.lineTo(sx(st.ks[i]), sy(st.xs_true[i]));
        }
        ctx.stroke();

        /* measurements (blue dots) */
        ctx.fillStyle = 'rgba(52,152,219,0.8)';
        for (var i = 0; i < st.ys.length; i++) {
            ctx.beginPath();
            ctx.arc(sx(st.ks[i + 1]), sy(st.ys[i]), 3.5, 0, 2 * Math.PI);
            ctx.fill();
        }

        /* KF estimate (red line + dots) */
        ctx.strokeStyle = '#e74c3c';
        ctx.lineWidth = 2;
        ctx.beginPath();
        for (var i = 0; i < st.ks.length; i++) {
            if (i === 0) ctx.moveTo(sx(st.ks[i]), sy(st.xs_hat[i]));
            else ctx.lineTo(sx(st.ks[i]), sy(st.xs_hat[i]));
        }
        ctx.stroke();
        ctx.fillStyle = '#e74c3c';
        for (var i = 0; i < st.ks.length; i++) {
            ctx.beginPath();
            ctx.arc(sx(st.ks[i]), sy(st.xs_hat[i]), 3, 0, 2 * Math.PI);
            ctx.fill();
        }

        /* legend */
        var lx = m.l + 8, ly = m.t + 10;
        ctx.font = '10px "Source Sans 3", sans-serif';

        ctx.strokeStyle = '#D4E0ED'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 14, ly); ctx.stroke();
        ctx.fillStyle = '#99AABB'; ctx.textAlign = 'left';
        ctx.fillText('True state', lx + 20, ly + 3);

        ly += 16;
        ctx.fillStyle = 'rgba(52,152,219,0.8)';
        ctx.beginPath(); ctx.arc(lx + 7, ly, 3.5, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = '#99AABB';
        ctx.fillText('Measurement', lx + 20, ly + 3);

        ly += 16;
        ctx.strokeStyle = '#e74c3c'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 14, ly); ctx.stroke();
        ctx.fillStyle = '#99AABB';
        ctx.fillText('KF estimate', lx + 20, ly + 3);

        ly += 16;
        ctx.fillStyle = 'rgba(231,76,60,0.18)';
        ctx.fillRect(lx, ly - 6, 14, 12);
        ctx.fillStyle = '#99AABB';
        ctx.fillText('\u00B12\u03C3 band', lx + 20, ly + 3);
    }

    /* ── Build GUI ────────────────────────────────────────── */
    function initDemo(container) {
        if (container.dataset.initialized) return;
        container.dataset.initialized = '1';

        container.innerHTML =
            '<div class="kf-header">' +
                '<span class="kf-title">\u25B6 Interactive Scalar Kalman Filter</span>' +
            '</div>' +

            '<div class="kf-system-info">' +
                'System: &nbsp;<em>x</em><sub><em>k</em>+1</sub> = <em>x</em><sub><em>k</em></sub> + <em>w</em><sub><em>k</em></sub> , &nbsp;&nbsp;' +
                '<em>y</em><sub><em>k</em></sub> = <em>x</em><sub><em>k</em></sub> + <em>v</em><sub><em>k</em></sub>' +
                '<span class="kf-fixed">&nbsp;&nbsp;(A = 1, &nbsp;C = 1 &mdash; fixed) &nbsp; | &nbsp; Initial: x\u0302\u2080 = 0, &nbsp;P\u2080 = 1</span>' +
            '</div>' +

            /* ── sliders + buttons ── */
            '<div class="kf-controls-row">' +
                '<div class="kf-slider-group">' +
                    '<label class="kf-slider-label">Process noise <em>W</em>' +
                        '<input type="range" class="kf-slider kf-w-slider" min="0.01" max="5" step="0.01" value="0.50">' +
                        '<span class="kf-slider-val kf-w-val">0.50</span>' +
                    '</label>' +
                    '<label class="kf-slider-label">Measurement noise <em>V</em>' +
                        '<input type="range" class="kf-slider kf-v-slider" min="0.01" max="10" step="0.01" value="2.00">' +
                        '<span class="kf-slider-val kf-v-val">2.00</span>' +
                    '</label>' +
                '</div>' +
                '<div class="kf-btn-group">' +
                    '<button class="kf-btn kf-step-btn">\u25B6 Step</button>' +
                    '<button class="kf-btn kf-auto-btn">\u25B6\u25B6 Auto</button>' +
                    '<button class="kf-btn kf-reset-btn">\u21BA Reset</button>' +
                    '<span class="kf-step-counter">k = 0</span>' +
                '</div>' +
            '</div>' +

            /* ── algorithm + plot side-by-side ── */
            '<div class="kf-body">' +

                /* algorithm panel */
                '<div class="kf-algo">' +

                    '<div class="kf-phase kf-ph-meas">' +
                        '<div class="kf-phase-title">Measurement</div>' +
                        '<div class="kf-eq">' +
                            '<span class="kf-formula"><em>y</em><sub><em>k</em></sub></span>' +
                            '<span class="kf-computed">= <span class="kv-y">&mdash;</span></span>' +
                        '</div>' +
                    '</div>' +

                    '<div class="kf-phase kf-ph-pred">' +
                        '<div class="kf-phase-title">Predict</div>' +
                        '<div class="kf-eq">' +
                            '<span class="kf-formula">x\u0302<sup>\u2212</sup><sub>k</sub> = A \u00B7 x\u0302<sub>k\u22121</sub></span>' +
                            '<span class="kf-computed">= <span class="kv-xpred">&mdash;</span></span>' +
                        '</div>' +
                        '<div class="kf-eq">' +
                            '<span class="kf-formula">P<sup>\u2212</sup><sub>k</sub> = A\u00B7P<sub>k\u22121</sub>\u00B7A + W</span>' +
                            '<span class="kf-computed">= <span class="kv-Ppred">&mdash;</span></span>' +
                        '</div>' +
                    '</div>' +

                    '<div class="kf-phase kf-ph-gain">' +
                        '<div class="kf-phase-title">Compute Gain</div>' +
                        '<div class="kf-eq">' +
                            '<span class="kf-formula">S<sub>k</sub> = C\u00B7P<sup>\u2212</sup><sub>k</sub>\u00B7C + V</span>' +
                            '<span class="kf-computed">= <span class="kv-S">&mdash;</span></span>' +
                        '</div>' +
                        '<div class="kf-eq">' +
                            '<span class="kf-formula">L<sub>k</sub> = P<sup>\u2212</sup><sub>k</sub>\u00B7C / S<sub>k</sub></span>' +
                            '<span class="kf-computed">= <span class="kv-L">&mdash;</span></span>' +
                        '</div>' +
                    '</div>' +

                    '<div class="kf-phase kf-ph-corr">' +
                        '<div class="kf-phase-title">Correct</div>' +
                        '<div class="kf-eq">' +
                            '<span class="kf-formula">y\u0303<sub>k</sub> = y<sub>k</sub> \u2212 C\u00B7x\u0302<sup>\u2212</sup><sub>k</sub></span>' +
                            '<span class="kf-computed">= <span class="kv-innov">&mdash;</span></span>' +
                        '</div>' +
                        '<div class="kf-eq">' +
                            '<span class="kf-formula">x\u0302<sub>k</sub> = x\u0302<sup>\u2212</sup><sub>k</sub> + L<sub>k</sub>\u00B7y\u0303<sub>k</sub></span>' +
                            '<span class="kf-computed">= <span class="kv-xhat">&mdash;</span></span>' +
                        '</div>' +
                        '<div class="kf-eq">' +
                            '<span class="kf-formula">P<sub>k</sub> = (1 \u2212 L<sub>k</sub>\u00B7C)\u00B7P<sup>\u2212</sup><sub>k</sub></span>' +
                            '<span class="kf-computed">= <span class="kv-P">&mdash;</span></span>' +
                        '</div>' +
                    '</div>' +

                '</div>' +

                /* plot panel */
                '<div class="kf-plot-wrap">' +
                    '<canvas class="kf-canvas"></canvas>' +
                '</div>' +

            '</div>';

        /* ── element refs (scoped to container) ── */
        var wSlider   = container.querySelector('.kf-w-slider');
        var vSlider   = container.querySelector('.kf-v-slider');
        var wValEl    = container.querySelector('.kf-w-val');
        var vValEl    = container.querySelector('.kf-v-val');
        var stepBtn   = container.querySelector('.kf-step-btn');
        var autoBtn   = container.querySelector('.kf-auto-btn');
        var resetBtn  = container.querySelector('.kf-reset-btn');
        var counterEl = container.querySelector('.kf-step-counter');
        var canvas    = container.querySelector('.kf-canvas');

        var valEls = {
            y:     container.querySelector('.kv-y'),
            xpred: container.querySelector('.kv-xpred'),
            Ppred: container.querySelector('.kv-Ppred'),
            S:     container.querySelector('.kv-S'),
            L:     container.querySelector('.kv-L'),
            innov: container.querySelector('.kv-innov'),
            xhat:  container.querySelector('.kv-xhat'),
            P:     container.querySelector('.kv-P')
        };

        var phaseEls = {
            meas: container.querySelector('.kf-ph-meas'),
            pred: container.querySelector('.kf-ph-pred'),
            gain: container.querySelector('.kf-ph-gain'),
            corr: container.querySelector('.kf-ph-corr')
        };

        var state = freshState();
        var autoTimer = null;
        var animTimers = [];

        function getW() { return parseFloat(wSlider.value); }
        function getV() { return parseFloat(vSlider.value); }

        function clearHL() {
            phaseEls.meas.classList.remove('kf-active', 'kf-done');
            phaseEls.pred.classList.remove('kf-active', 'kf-done');
            phaseEls.gain.classList.remove('kf-active', 'kf-done');
            phaseEls.corr.classList.remove('kf-active', 'kf-done');
        }

        function clearAnims() {
            for (var i = 0; i < animTimers.length; i++) clearTimeout(animTimers[i]);
            animTimers = [];
        }

        function resetVals() {
            for (var k in valEls) valEls[k].textContent = '\u2014';
            clearHL();
            clearAnims();
        }

        function doStep() {
            clearAnims();
            var c = stepKF(state, getW(), getV());
            counterEl.textContent = 'k = ' + state.k;

            /* phase 1: measurement */
            clearHL();
            phaseEls.meas.classList.add('kf-active');
            valEls.y.textContent = fmt(c.y);

            /* phase 2: predict */
            animTimers.push(setTimeout(function () {
                clearHL();
                phaseEls.meas.classList.add('kf-done');
                phaseEls.pred.classList.add('kf-active');
                valEls.xpred.textContent = fmt(c.x_pred);
                valEls.Ppred.textContent = fmt(c.P_pred);
            }, 700));

            /* phase 3: gain */
            animTimers.push(setTimeout(function () {
                clearHL();
                phaseEls.meas.classList.add('kf-done');
                phaseEls.pred.classList.add('kf-done');
                phaseEls.gain.classList.add('kf-active');
                valEls.S.textContent = fmt(c.S);
                valEls.L.textContent = fmt(c.L);
            }, 1400));

            /* phase 4: correct + update plot */
            animTimers.push(setTimeout(function () {
                clearHL();
                phaseEls.meas.classList.add('kf-done');
                phaseEls.pred.classList.add('kf-done');
                phaseEls.gain.classList.add('kf-done');
                phaseEls.corr.classList.add('kf-active');
                valEls.innov.textContent = fmt(c.innov);
                valEls.xhat.textContent = fmt(c.x_hat);
                valEls.P.textContent = fmt(c.P);
                drawPlot(canvas, state);
            }, 2100));

            /* after animation: keep all phases visible with values */
            animTimers.push(setTimeout(function () {
                clearHL();
                phaseEls.meas.classList.add('kf-done');
                phaseEls.pred.classList.add('kf-done');
                phaseEls.gain.classList.add('kf-done');
                phaseEls.corr.classList.add('kf-done');
            }, 2700));
        }

        function doReset() {
            stopAuto();
            state = freshState();
            counterEl.textContent = 'k = 0';
            resetVals();
            drawPlot(canvas, state);
        }

        function toggleAuto() {
            if (autoTimer) { stopAuto(); return; }
            autoBtn.textContent = '\u23F8 Stop';
            autoBtn.classList.add('kf-auto-active');
            doStep();
            autoTimer = setInterval(doStep, 3500);
        }

        function stopAuto() {
            if (autoTimer) { clearInterval(autoTimer); autoTimer = null; }
            autoBtn.textContent = '\u25B6\u25B6 Auto';
            autoBtn.classList.remove('kf-auto-active');
        }

        /* ── events ── */
        stepBtn.addEventListener('click', function () { stopAuto(); doStep(); });
        autoBtn.addEventListener('click', toggleAuto);
        resetBtn.addEventListener('click', doReset);

        wSlider.addEventListener('input', function () {
            wValEl.textContent = parseFloat(this.value).toFixed(2);
            doReset();
        });
        vSlider.addEventListener('input', function () {
            vValEl.textContent = parseFloat(this.value).toFixed(2);
            doReset();
        });

        /* initial draw */
        drawPlot(canvas, state);

        /* resize handling */
        if (window.ResizeObserver) {
            var ro = new ResizeObserver(function () { drawPlot(canvas, state); });
            ro.observe(canvas);
        }
    }

    /* ── Public API ───────────────────────────────────────── */
    function initAllDemos(root) {
        var els = (root || document).querySelectorAll('.kalman-demo');
        for (var i = 0; i < els.length; i++) initDemo(els[i]);
    }

    window.KalmanDemo = { initAll: initAllDemos, init: initDemo };
})();
