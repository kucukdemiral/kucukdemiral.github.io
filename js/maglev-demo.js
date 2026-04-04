/**
 * Magnetic Levitation Animation — LQR Setpoint Tracking
 * Pure JS/Canvas: solves DARE, simulates nonlinear plant with RK4, animates.
 * HTML hook: <div class="maglev-demo"></div>
 */
(function () {
    'use strict';

    /* ── Physics ─────────────────────────────────────────── */
    var m = 0.05, g = 9.81, Cm = 0.001;
    var y0_eq = 0.01, Ts = 0.005, T_final = 1.0;

    function equilCurrent(y) { return y * Math.sqrt(m * g / Cm); }

    /* ── Nonlinear dynamics ──────────────────────────────── */
    function f_dyn(x0, x1, u) {
        return [x1, g - (Cm / m) * (u / x0) * (u / x0)];
    }
    function rk4(x, u) {
        var k1 = f_dyn(x[0], x[1], u);
        var k2 = f_dyn(x[0]+0.5*Ts*k1[0], x[1]+0.5*Ts*k1[1], u);
        var k3 = f_dyn(x[0]+0.5*Ts*k2[0], x[1]+0.5*Ts*k2[1], u);
        var k4 = f_dyn(x[0]+Ts*k3[0], x[1]+Ts*k3[1], u);
        var xn = [x[0]+(Ts/6)*(k1[0]+2*k2[0]+2*k3[0]+k4[0]),
                  x[1]+(Ts/6)*(k1[1]+2*k2[1]+2*k3[1]+k4[1])];
        if (xn[0] <= 0) { xn[0] = 1e-4; xn[1] = 0; }
        return xn;
    }

    /* ── ZOH discretisation (analytical for [[0,1],[ω²,0]]) */
    function discretise() {
        var w2 = 2 * g / y0_eq, w = Math.sqrt(w2);
        var ch = Math.cosh(w * Ts), sh = Math.sinh(w * Ts);
        var u0 = equilCurrent(y0_eq);
        var bc = -2 * g / u0;
        var Ad = [[ch, sh/w], [w*sh, ch]];
        var Bd = [(ch-1)/w2 * bc, sh/w * bc];
        return { Ad: Ad, Bd: Bd };
    }

    /* ── DARE iteration (2×2) ────────────────────────────── */
    function solveDARE(A, B, Q, R) {
        var P = [[Q[0], 0], [0, Q[1]]];
        for (var it = 0; it < 500; it++) {
            var pa00=P[0][0]*A[0][0]+P[0][1]*A[1][0], pa01=P[0][0]*A[0][1]+P[0][1]*A[1][1];
            var pa10=P[1][0]*A[0][0]+P[1][1]*A[1][0], pa11=P[1][0]*A[0][1]+P[1][1]*A[1][1];
            var atpa00=A[0][0]*pa00+A[1][0]*pa10, atpa01=A[0][0]*pa01+A[1][0]*pa11;
            var atpa10=A[0][1]*pa00+A[1][1]*pa10, atpa11=A[0][1]*pa01+A[1][1]*pa11;
            var pb0=P[0][0]*B[0]+P[0][1]*B[1], pb1=P[1][0]*B[0]+P[1][1]*B[1];
            var atpb0=A[0][0]*pb0+A[1][0]*pb1, atpb1=A[0][1]*pb0+A[1][1]*pb1;
            var s=R+B[0]*pb0+B[1]*pb1, inv=1/s;
            P=[[Q[0]+atpa00-atpb0*atpb0*inv, atpa01-atpb0*atpb1*inv],
               [atpa10-atpb1*atpb0*inv, Q[1]+atpa11-atpb1*atpb1*inv]];
        }
        return P;
    }

    function computeK(A, B, P, R) {
        var pb0=P[0][0]*B[0]+P[0][1]*B[1], pb1=P[1][0]*B[0]+P[1][1]*B[1];
        var s=R+B[0]*pb0+B[1]*pb1, inv=1/s;
        var bpa0=B[0]*(P[0][0]*A[0][0]+P[0][1]*A[1][0])+B[1]*(P[1][0]*A[0][0]+P[1][1]*A[1][0]);
        var bpa1=B[0]*(P[0][0]*A[0][1]+P[0][1]*A[1][1])+B[1]*(P[1][0]*A[0][1]+P[1][1]*A[1][1]);
        return [inv*bpa0, inv*bpa1];
    }

    /* ── Full trajectory ─────────────────────────────────── */
    function simulate(y_target) {
        var sys = discretise();
        var P = solveDARE(sys.Ad, sys.Bd, [5000, 10], 1);
        var K = computeK(sys.Ad, sys.Bd, P, 1);
        var u_t = equilCurrent(y_target), x_t = [y_target, 0];
        var N = Math.floor(T_final / Ts);
        var ys = [], vs = [], us = [], ts = [];
        var x = [y0_eq, 0];
        for (var k = 0; k < N; k++) {
            var u_val = u_t - K[0]*(x[0]-x_t[0]) - K[1]*(x[1]-x_t[1]);
            ts.push(k * Ts); ys.push(x[0]); vs.push(x[1]); us.push(u_val);
            x = rk4(x, u_val);
        }
        return { ts:ts, ys:ys, vs:vs, us:us, N:N, K:K, y_target:y_target, u_target:u_t };
    }

    /* ── Drawing helpers ─────────────────────────────────── */
    function setupCanvas(canvas) {
        var dpr = window.devicePixelRatio || 1;
        var w = canvas.clientWidth, h = canvas.clientHeight;
        canvas.width = w * dpr; canvas.height = h * dpr;
        var ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        return { ctx:ctx, w:w, h:h };
    }

    /* physical scene */
    function drawScene(canvas, yPos, yTarget) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0A1219'; ctx.fillRect(0, 0, W, H);

        var mgTop = 15, mgH = 38, mgW = 80, mgX = (W - mgW) / 2;
        var yMin = 0.005, yMax = 0.020;
        var trackTop = mgTop + mgH + 8, trackBot = H - 30;
        function yToPx(y) { return trackTop + (y - yMin) / (yMax - yMin) * (trackBot - trackTop); }

        /* electromagnet */
        ctx.fillStyle = '#3a3a50';
        ctx.fillRect(mgX, mgTop, mgW, mgH);
        ctx.strokeStyle = '#0D7C66'; ctx.lineWidth = 2;
        ctx.strokeRect(mgX, mgTop, mgW, mgH);
        /* coil windings */
        ctx.strokeStyle = '#D4924A'; ctx.lineWidth = 1.5;
        for (var i = 0; i < 6; i++) {
            var cx = mgX + 12 + i * 10, cy = mgTop + mgH / 2;
            ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.stroke();
        }
        ctx.fillStyle = '#99AABB'; ctx.font = '10px "Source Sans 3",sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('ELECTROMAGNET', W / 2, mgTop + mgH + 6);

        /* scale bar */
        ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(W - 22, trackTop); ctx.lineTo(W - 22, trackBot); ctx.stroke();
        ctx.fillStyle = '#556677'; ctx.font = '9px "SF Mono",monospace'; ctx.textAlign = 'right';
        for (var mm = 6; mm <= 18; mm += 2) {
            var py = yToPx(mm / 1000);
            ctx.beginPath(); ctx.moveTo(W - 26, py); ctx.lineTo(W - 18, py); ctx.stroke();
            ctx.fillText(mm + '', W - 28, py + 3);
        }
        ctx.fillText('mm', W - 28, trackBot + 14);

        /* reference line */
        var refY = yToPx(yTarget);
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = '#0D7C66'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(20, refY); ctx.lineTo(W - 35, refY); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#0D7C66'; ctx.font = '10px "Source Sans 3",sans-serif'; ctx.textAlign = 'left';
        ctx.fillText('ref', 8, refY - 4);

        /* ball */
        var ballY = yToPx(yPos);
        var ballR = 14;
        var grad = ctx.createRadialGradient(W/2 - 3, ballY - 3, 2, W/2, ballY, ballR);
        grad.addColorStop(0, '#c0c8d0');
        grad.addColorStop(0.7, '#6a7280');
        grad.addColorStop(1, '#3a4050');
        ctx.fillStyle = grad;
        ctx.beginPath(); ctx.arc(W / 2, ballY, ballR, 0, Math.PI * 2); ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
        ctx.stroke();

        /* force arrows */
        var arrX = W / 2 + 30;
        /* gravity arrow (down) */
        ctx.strokeStyle = '#e74c3c'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(arrX, ballY); ctx.lineTo(arrX, ballY + 22); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(arrX-4,ballY+17); ctx.lineTo(arrX,ballY+22); ctx.lineTo(arrX+4,ballY+17); ctx.stroke();
        ctx.fillStyle = '#e74c3c'; ctx.font = '9px "Source Sans 3",sans-serif'; ctx.textAlign = 'left';
        ctx.fillText('mg', arrX + 5, ballY + 20);
        /* magnetic force arrow (up) */
        ctx.strokeStyle = '#3498db';
        ctx.beginPath(); ctx.moveTo(arrX, ballY); ctx.lineTo(arrX, ballY - 22); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(arrX-4,ballY-17); ctx.lineTo(arrX,ballY-22); ctx.lineTo(arrX+4,ballY-17); ctx.stroke();
        ctx.fillStyle = '#3498db';
        ctx.fillText('F_m', arrX + 5, ballY - 16);

        /* ground */
        ctx.strokeStyle = 'rgba(255,255,255,0.25)'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(10, trackBot + 4); ctx.lineTo(W - 10, trackBot + 4); ctx.stroke();
        for (var hx = 15; hx < W - 10; hx += 8) {
            ctx.beginPath(); ctx.moveTo(hx, trackBot + 4); ctx.lineTo(hx - 5, trackBot + 12); ctx.stroke();
        }
    }

    /* plots (position + current) */
    function drawPlots(canvas, traj, idx) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0A1219'; ctx.fillRect(0, 0, W, H);

        var ml = 42, mr = 14, gap = 20;
        var pH = (H - gap) / 2;

        /* ── top plot: position ── */
        plotRegion(ctx, ml, 0, W - ml - mr, pH, traj.ts, traj.ys, idx,
            1000, 'Position (mm)', traj.y_target * 1000, '#e74c3c', '#0D7C66');

        /* ── bottom plot: current ── */
        plotRegion(ctx, ml, pH + gap, W - ml - mr, pH, traj.ts, traj.us, idx,
            1, 'Current (A)', traj.u_target, '#f39c12', '#0D7C66');
    }

    function plotRegion(ctx, ox, oy, pw, ph, xs, ys, idx, scale, label, refVal, color, refColor) {
        var mt = 18, mb = 22, mg = 4;
        var plotW = pw - mg, plotH = ph - mt - mb;

        /* y range */
        var yMin = Infinity, yMax = -Infinity;
        for (var i = 0; i < ys.length; i++) {
            var v = ys[i] * scale;
            if (v < yMin) yMin = v; if (v > yMax) yMax = v;
        }
        var rv = refVal * scale;
        if (rv < yMin) yMin = rv; if (rv > yMax) yMax = rv;
        var pad = Math.max((yMax - yMin) * 0.15, 0.001);
        yMin -= pad; yMax += pad;

        var xMin = 0, xMax = xs[xs.length - 1] || T_final;
        function sx(v) { return ox + mg + (v - xMin) / (xMax - xMin) * plotW; }
        function sy(v) { return oy + mt + (1 - (v - yMin) / (yMax - yMin)) * plotH; }

        /* grid */
        ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 1;
        ctx.fillStyle = '#556677'; ctx.font = '9px "SF Mono",monospace'; ctx.textAlign = 'right';
        var nyt = 4, ystep = (yMax - yMin) / nyt;
        for (var i = 0; i <= nyt; i++) {
            var v = yMin + i * ystep, py = sy(v);
            ctx.beginPath(); ctx.moveTo(ox + mg, py); ctx.lineTo(ox + mg + plotW, py); ctx.stroke();
            ctx.fillText(v.toFixed(scale > 100 ? 1 : 3), ox + mg - 4, py + 3);
        }
        ctx.textAlign = 'center';
        for (var t = 0; t <= xMax; t += 0.2) {
            ctx.fillText(t.toFixed(1), sx(t), oy + mt + plotH + 13);
        }

        /* axes */
        ctx.strokeStyle = 'rgba(255,255,255,0.18)'; ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(ox+mg, oy+mt); ctx.lineTo(ox+mg, oy+mt+plotH); ctx.lineTo(ox+mg+plotW, oy+mt+plotH);
        ctx.stroke();

        /* reference line */
        ctx.setLineDash([5, 3]);
        ctx.strokeStyle = refColor; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(ox+mg, sy(rv)); ctx.lineTo(ox+mg+plotW, sy(rv)); ctx.stroke();
        ctx.setLineDash([]);

        /* data line */
        var n = Math.min(idx + 1, ys.length);
        if (n > 1) {
            ctx.strokeStyle = color; ctx.lineWidth = 2;
            ctx.beginPath();
            for (var i = 0; i < n; i++) {
                var px = sx(xs[i]), py = sy(ys[i] * scale);
                if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
            }
            ctx.stroke();
        }

        /* current point */
        if (idx >= 0 && idx < ys.length) {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(sx(xs[idx]), sy(ys[idx] * scale), 4, 0, Math.PI * 2);
            ctx.fill();
        }

        /* label */
        ctx.fillStyle = '#99AABB'; ctx.font = '11px "Source Sans 3",sans-serif'; ctx.textAlign = 'left';
        ctx.fillText(label, ox + mg + 4, oy + mt - 5);
        ctx.textAlign = 'center';
        ctx.fillText('Time (s)', ox + mg + plotW / 2, oy + mt + plotH + mb - 2);
    }

    /* ── Build widget ────────────────────────────────────── */
    function initDemo(container) {
        if (container.dataset.initialized) return;
        container.dataset.initialized = '1';

        container.innerHTML =
            '<div class="mlv-header">' +
                '<span class="mlv-title">\u25B6 Magnetic Levitation \u2014 LQR Animation</span>' +
            '</div>' +
            '<div class="mlv-controls">' +
                '<label class="mlv-slider-label">Target gap' +
                    '<input type="range" class="mlv-slider mlv-target-slider" min="8" max="16" step="0.1" value="12">' +
                    '<span class="mlv-slider-val mlv-target-val">12.0 mm</span>' +
                '</label>' +
                '<label class="mlv-slider-label">Speed' +
                    '<input type="range" class="mlv-slider mlv-speed-slider" min="0.1" max="2" step="0.1" value="0.5">' +
                    '<span class="mlv-slider-val mlv-speed-val">0.5\u00D7</span>' +
                '</label>' +
                '<div class="mlv-btn-group">' +
                    '<button class="mlv-btn mlv-run-btn">\u25B6 Run</button>' +
                    '<button class="mlv-btn mlv-reset-btn">\u21BA Reset</button>' +
                '</div>' +
            '</div>' +
            '<div class="mlv-body">' +
                '<div class="mlv-scene-wrap"><canvas class="mlv-scene-canvas"></canvas></div>' +
                '<div class="mlv-plot-wrap"><canvas class="mlv-plot-canvas"></canvas></div>' +
            '</div>' +
            '<div class="mlv-readout">' +
                '<span class="mlv-ro-item">t = <span class="mlv-ro-t">0.000</span> s</span>' +
                '<span class="mlv-ro-item">y = <span class="mlv-ro-y">10.00</span> mm</span>' +
                '<span class="mlv-ro-item">i = <span class="mlv-ro-u">0.000</span> A</span>' +
                '<span class="mlv-ro-item">K = [<span class="mlv-ro-k">--, --</span>]</span>' +
            '</div>';

        var sceneCvs = container.querySelector('.mlv-scene-canvas');
        var plotCvs  = container.querySelector('.mlv-plot-canvas');
        var targetSlider = container.querySelector('.mlv-target-slider');
        var speedSlider  = container.querySelector('.mlv-speed-slider');
        var targetVal = container.querySelector('.mlv-target-val');
        var speedVal  = container.querySelector('.mlv-speed-val');
        var runBtn   = container.querySelector('.mlv-run-btn');
        var resetBtn = container.querySelector('.mlv-reset-btn');
        var roT = container.querySelector('.mlv-ro-t');
        var roY = container.querySelector('.mlv-ro-y');
        var roU = container.querySelector('.mlv-ro-u');
        var roK = container.querySelector('.mlv-ro-k');

        var traj = null, animId = null, running = false;
        var startTime = 0, pausedAt = 0;

        function getTarget() { return parseFloat(targetSlider.value) / 1000; }
        function getSpeed()  { return parseFloat(speedSlider.value); }

        function recompute() {
            traj = simulate(getTarget());
            roK.textContent = traj.K[0].toFixed(2) + ', ' + traj.K[1].toFixed(4);
        }

        function drawFrame(idx) {
            if (!traj) return;
            idx = Math.max(0, Math.min(idx, traj.N - 1));
            drawScene(sceneCvs, traj.ys[idx], traj.y_target);
            drawPlots(plotCvs, traj, idx);
            roT.textContent = (traj.ts[idx]).toFixed(3);
            roY.textContent = (traj.ys[idx] * 1000).toFixed(2);
            roU.textContent = traj.us[idx].toFixed(4);
        }

        function animLoop(timestamp) {
            if (!running) return;
            var elapsed = (timestamp - startTime) / 1000 * getSpeed();
            var idx = Math.floor(elapsed / Ts);
            if (idx >= traj.N) {
                idx = traj.N - 1;
                running = false;
                runBtn.textContent = '\u25B6 Run';
            }
            drawFrame(idx);
            if (running) animId = requestAnimationFrame(animLoop);
        }

        function doRun() {
            if (running) {
                running = false;
                pausedAt = performance.now() - startTime;
                runBtn.textContent = '\u25B6 Run';
                if (animId) cancelAnimationFrame(animId);
                return;
            }
            if (!traj) recompute();
            running = true;
            runBtn.textContent = '\u23F8 Pause';
            startTime = performance.now() - pausedAt;
            animId = requestAnimationFrame(animLoop);
        }

        function doReset() {
            running = false;
            if (animId) cancelAnimationFrame(animId);
            runBtn.textContent = '\u25B6 Run';
            pausedAt = 0;
            recompute();
            drawFrame(0);
        }

        runBtn.addEventListener('click', doRun);
        resetBtn.addEventListener('click', doReset);

        targetSlider.addEventListener('input', function () {
            targetVal.textContent = parseFloat(this.value).toFixed(1) + ' mm';
            doReset();
        });
        speedSlider.addEventListener('input', function () {
            speedVal.textContent = parseFloat(this.value).toFixed(1) + '\u00D7';
        });

        /* initial render */
        recompute();
        drawFrame(0);

        /* resize */
        if (window.ResizeObserver) {
            var ro = new ResizeObserver(function () { if (traj) drawFrame(0); });
            ro.observe(plotCvs);
        }
    }

    function initAllDemos(root) {
        var els = (root || document).querySelectorAll('.maglev-demo');
        for (var i = 0; i < els.length; i++) initDemo(els[i]);
    }

    window.MaglevDemo = { initAll: initAllDemos, init: initDemo };
})();
