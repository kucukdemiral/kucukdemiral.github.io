/**
 * Nonholonomic Robot — Reference Tracking MPC Simulation Environment
 * Interactive real-time linearised MPC with:
 *   - Click/drag to set initial position & heading
 *   - Real-time strip charts (v, omega, tracking error)
 *   - Gradient trail, predicted trajectory line
 *   - Full parameter tuning panel
 *
 * HTML: <div class="robot-demo-container" id="robot-demo-1"></div>
 */
(function () {
    'use strict';

    var NX = 3, NU = 2;

    var COL = {
        bg: '#1a2332', grid: '#2a3a4a', axis: '#3a4a5a',
        ref: '#4ecdc4', trail: '#ff6b6b', robot: '#ffffff',
        pred: '#ffb464', accent: '#ffd93d', text: '#D4E0ED',
        dim: '#8899AA', panel: '#0f1923', chartBg: '#111b27',
        err: '#ffd93d', vCol: '#4ecdc4', wCol: '#ff6b6b'
    };

    /* ─── Helpers ─────────────────────────────────────────── */
    function wrap(a) { return Math.atan2(Math.sin(a), Math.cos(a)); }
    function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
    function m33(A, B) {
        var C = new Array(9);
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 3; j++)
                C[i * 3 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[3 + j] + A[i * 3 + 2] * B[6 + j];
        return C;
    }
    function m32(A, B) {
        var C = new Array(6);
        for (var i = 0; i < 3; i++)
            for (var j = 0; j < 2; j++)
                C[i * 2 + j] = A[i * 3] * B[j] + A[i * 3 + 1] * B[2 + j] + A[i * 3 + 2] * B[4 + j];
        return C;
    }

    /* ─── Unicycle Model ──────────────────────────────────── */
    function unicycle(s, u, dt) {
        return [s[0] + dt * u[0] * Math.cos(s[2]), s[1] + dt * u[0] * Math.sin(s[2]), s[2] + dt * u[1]];
    }
    function jacobians(s, u, dt) {
        var c = Math.cos(s[2]), si = Math.sin(s[2]), v = u[0];
        return { A: [1, 0, -dt * v * si, 0, 1, dt * v * c, 0, 0, 1], B: [dt * c, 0, dt * si, 0, 0, dt] };
    }

    /* ─── Reference Trajectories ──────────────────────────── */
    function fig8(t, T, sc) {
        var w = 2 * Math.PI / T;
        var x = sc * Math.sin(w * t), y = sc * Math.sin(2 * w * t) / 2;
        var dx = sc * w * Math.cos(w * t), dy = sc * w * Math.cos(2 * w * t);
        var sp = Math.sqrt(dx * dx + dy * dy);
        var th = Math.atan2(dy, dx);
        var ddx = -sc * w * w * Math.sin(w * t), ddy = -2 * sc * w * w * Math.sin(2 * w * t);
        var om = sp > 0.01 ? (ddx * dy - dx * ddy) / (dx * dx + dy * dy) : 0;
        return { s: [x, y, th], u: [Math.max(sp, 0.01), om] };
    }
    function circ(t, T, r) {
        var w = 2 * Math.PI / T;
        return {
            s: [r * Math.cos(w * t), r * Math.sin(w * t), wrap(w * t + Math.PI / 2)],
            u: [r * w, w]
        };
    }

    /* ─── Condensed Linearised MPC ────────────────────────── */
    function solveMPC(x0, t0, P) {
        var N = P.N, dt = P.dt;
        var refFn = P.rType === 'circle' ? circ : fig8;
        var rA = P.rType === 'circle' ? P.rR : P.rS;
        var rS = [], rU = [];
        for (var i = 0; i <= N; i++) { var r = refFn(t0 + i * dt, P.rT, rA); rS.push(r.s); rU.push(r.u); }

        var e0 = [x0[0] - rS[0][0], x0[1] - rS[0][1], wrap(x0[2] - rS[0][2])];
        var Phi = [1, 0, 0, 0, 1, 0, 0, 0, 1], PsiB = [], ThB = [];
        for (var k = 0; k < N; k++) ThB.push(new Array(k + 1));

        for (var k = 0; k < N; k++) {
            var jac = jacobians(rS[k], rU[k], dt);
            Phi = m33(jac.A, Phi); PsiB.push(Phi.slice());
            if (k > 0) for (var j = 0; j < k; j++) ThB[k][j] = m32(jac.A, ThB[k - 1][j]);
            ThB[k][k] = jac.B.slice();
        }

        var pe = new Array(N * NX);
        for (var k = 0; k < N; k++) { var b = PsiB[k]; for (var ii = 0; ii < NX; ii++) pe[k * NX + ii] = b[ii * NX] * e0[0] + b[ii * NX + 1] * e0[1] + b[ii * NX + 2] * e0[2]; }

        var q = new Array(N * NX);
        for (var k = 0; k < N; k++) { var mult = (k === N - 1) ? 10 : 1; q[k * NX] = P.Qp * mult; q[k * NX + 1] = P.Qp * mult; q[k * NX + 2] = P.Qt * mult; }

        var Nm = N * NU, H = new Array(Nm * Nm), g = new Array(Nm);
        for (var i = 0; i < Nm * Nm; i++) H[i] = 0;
        for (var i = 0; i < Nm; i++) g[i] = 0;

        for (var k = 0; k < N; k++) {
            var qk0 = q[k * NX], qk1 = q[k * NX + 1], qk2 = q[k * NX + 2];
            var pe0 = pe[k * NX], pe1 = pe[k * NX + 1], pe2 = pe[k * NX + 2];
            for (var j1 = 0; j1 <= k; j1++) {
                var T1 = ThB[k][j1];
                for (var a = 0; a < NU; a++) g[j1 * NU + a] += T1[a] * qk0 * pe0 + T1[2 + a] * qk1 * pe1 + T1[4 + a] * qk2 * pe2;
                for (var j2 = j1; j2 <= k; j2++) {
                    var T2 = ThB[k][j2];
                    for (var a = 0; a < NU; a++) for (var bb = 0; bb < NU; bb++) {
                        var val = T1[a] * qk0 * T2[bb] + T1[2 + a] * qk1 * T2[2 + bb] + T1[4 + a] * qk2 * T2[4 + bb];
                        H[(j1 * NU + a) * Nm + (j2 * NU + bb)] += val;
                        if (j1 !== j2) H[(j2 * NU + bb) * Nm + (j1 * NU + a)] += val;
                    }
                }
            }
        }
        for (var k = 0; k < N; k++) { H[(k * NU) * Nm + (k * NU)] += P.Rv; H[(k * NU + 1) * Nm + (k * NU + 1)] += P.Rw; }

        var lb = new Array(Nm), ub = new Array(Nm);
        for (var k = 0; k < N; k++) {
            lb[k * NU] = -P.vM - rU[k][0]; ub[k * NU] = P.vM - rU[k][0];
            lb[k * NU + 1] = -P.wM - rU[k][1]; ub[k * NU + 1] = P.wM - rU[k][1];
        }

        var du = P.warm ? P.warm.slice() : new Array(Nm);
        if (!P.warm) for (var i = 0; i < Nm; i++) du[i] = 0;
        for (var iter = 0; iter < 35; iter++)
            for (var i = 0; i < Nm; i++) { var hii = H[i * Nm + i]; if (hii < 1e-12) continue; var s = g[i]; for (var jj = 0; jj < Nm; jj++) if (jj !== i) s += H[i * Nm + jj] * du[jj]; du[i] = clamp(-s / hii, lb[i], ub[i]); }

        var u0 = [clamp(rU[0][0] + du[0], -P.vM, P.vM), clamp(rU[0][1] + du[1], -P.wM, P.wM)];
        var pred = [x0.slice()], xp = x0.slice();
        for (var k = 0; k < N; k++) { xp = unicycle(xp, [clamp(rU[k][0] + du[k * NU], -P.vM, P.vM), clamp(rU[k][1] + du[k * NU + 1], -P.wM, P.wM)], dt); pred.push(xp.slice()); }

        var w = new Array(Nm);
        for (var k = 0; k < N - 1; k++) { w[k * NU] = du[(k + 1) * NU]; w[k * NU + 1] = du[(k + 1) * NU + 1]; }
        w[(N - 1) * NU] = 0; w[(N - 1) * NU + 1] = 0;

        return { u: u0, pred: pred, warm: w, ref: rS[0] };
    }

    /* ─── Strip Chart Renderer ────────────────────────────── */
    function drawChart(ctx, data, W, H, color, yMin, yMax, label, unit, limitVal) {
        ctx.fillStyle = COL.chartBg;
        ctx.fillRect(0, 0, W, H);

        /* Grid lines */
        ctx.strokeStyle = '#1e2e3e';
        ctx.lineWidth = 0.5;
        var nGrid = 4;
        for (var i = 1; i < nGrid; i++) {
            var gy = (i / nGrid) * H;
            ctx.beginPath(); ctx.moveTo(30, gy); ctx.lineTo(W, gy); ctx.stroke();
        }
        /* Zero line */
        if (yMin < 0 && yMax > 0) {
            var zy = H - ((-yMin) / (yMax - yMin)) * H;
            ctx.strokeStyle = '#3a4a5a'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(30, zy); ctx.lineTo(W, zy); ctx.stroke();
        }

        /* Limit lines */
        if (limitVal !== undefined) {
            ctx.strokeStyle = 'rgba(255,107,107,0.4)'; ctx.lineWidth = 1; ctx.setLineDash([4, 3]);
            var ly1 = H - ((limitVal - yMin) / (yMax - yMin)) * H;
            var ly2 = H - ((-limitVal - yMin) / (yMax - yMin)) * H;
            ctx.beginPath(); ctx.moveTo(30, ly1); ctx.lineTo(W, ly1); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(30, ly2); ctx.lineTo(W, ly2); ctx.stroke();
            ctx.setLineDash([]);
        }

        /* Data */
        var maxLen = 300;
        if (data.length > 1) {
            ctx.strokeStyle = color; ctx.lineWidth = 1.5; ctx.lineCap = 'round';
            ctx.beginPath();
            var plotW = W - 32;
            for (var i = 0; i < data.length; i++) {
                var x = 32 + (i / (maxLen - 1)) * plotW;
                var y = clamp(H - 4 - ((data[i] - yMin) / (yMax - yMin)) * (H - 8), 2, H - 2);
                i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            }
            ctx.stroke();
        }

        /* Y-axis labels */
        ctx.fillStyle = COL.dim; ctx.font = '9px system-ui'; ctx.textAlign = 'right';
        ctx.fillText(yMax.toFixed(1), 28, 11);
        ctx.fillText(yMin.toFixed(1), 28, H - 2);

        /* Title + current value */
        ctx.textAlign = 'left'; ctx.fillStyle = COL.dim; ctx.font = 'bold 10px system-ui';
        ctx.fillText(label, 34, 12);
        if (data.length > 0) {
            ctx.fillStyle = color; ctx.font = '10px system-ui'; ctx.textAlign = 'right';
            ctx.fillText(data[data.length - 1].toFixed(2) + ' ' + unit, W - 4, 12);
        }

        /* Border */
        ctx.strokeStyle = '#2a3a4a'; ctx.lineWidth = 1;
        ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
    }

    /* ─── Demo Instance ───────────────────────────────────── */
    function initDemo(container) {
        if (container.dataset.init) return;
        container.dataset.init = '1';

        var P = {
            dt: 0.05, N: 15,
            Qp: 50, Qt: 10, Rv: 1, Rw: 1,
            vM: 3, wM: 3,
            rType: 'figure8', rT: 12, rS: 3, rR: 3,
            speed: 1, warm: null
        };

        var simT = 0, state = [0, 0, 0], trail = [], running = false, raf = null, mpcR = null;
        var hist = { v: [], w: [], e: [], maxLen: 300 };
        var dragging = false, dragOrigin = null;

        /* ── Build DOM ── */
        container.innerHTML = '';
        container.style.cssText = 'background:#0f1923;border-radius:14px;padding:18px;max-width:960px;margin:2rem auto;font-family:system-ui,-apple-system,sans-serif;color:#D4E0ED;';

        /* Header */
        var hdr = document.createElement('div');
        hdr.style.cssText = 'text-align:center;margin-bottom:14px;';
        hdr.innerHTML = '<div style="font-size:1.15rem;font-weight:700;color:#ffd93d;margin-bottom:4px;">Reference Tracking MPC \u2014 Nonholonomic Robot</div>' +
            '<div style="font-size:0.75rem;color:#8899AA;">Unicycle model: \u1E8B = v cos\u03B8, \u1E8F = v sin\u03B8, \u03B8\u0307 = \u03C9 &nbsp;\u2502&nbsp; Linearised MPC with QP solver &nbsp;\u2502&nbsp; Click canvas to set start position</div>';
        container.appendChild(hdr);

        /* Main row */
        var mainRow = document.createElement('div');
        mainRow.style.cssText = 'display:flex;gap:14px;flex-wrap:wrap;';
        container.appendChild(mainRow);

        /* Left column: canvas + charts */
        var leftCol = document.createElement('div');
        leftCol.style.cssText = 'flex:1;min-width:380px;display:flex;flex-direction:column;gap:8px;';
        mainRow.appendChild(leftCol);

        /* Main canvas */
        var cvs = document.createElement('canvas');
        cvs.width = 620; cvs.height = 440;
        cvs.style.cssText = 'width:100%;border-radius:8px;background:#1a2332;display:block;cursor:crosshair;';
        leftCol.appendChild(cvs);
        var ctx = cvs.getContext('2d');

        /* Instruction text */
        var instrEl = document.createElement('div');
        instrEl.style.cssText = 'font-size:0.72rem;color:#8899AA;text-align:center;margin-top:-4px;';
        instrEl.textContent = '\uD83D\uDDB1\uFE0F Click to set start position \u2022 Drag to set heading direction';
        leftCol.appendChild(instrEl);

        /* Strip charts row */
        var chartRow = document.createElement('div');
        chartRow.style.cssText = 'display:flex;gap:6px;';
        leftCol.appendChild(chartRow);

        function mkChart() {
            var c = document.createElement('canvas');
            c.width = 200; c.height = 85;
            c.style.cssText = 'flex:1;border-radius:6px;display:block;height:85px;';
            chartRow.appendChild(c);
            return c;
        }
        var chartV = mkChart(), chartW = mkChart(), chartE = mkChart();

        /* Right column: controls */
        var pnl = document.createElement('div');
        pnl.style.cssText = 'width:255px;display:flex;flex-direction:column;gap:5px;font-size:0.8rem;';
        mainRow.appendChild(pnl);

        function sectionTitle(text) {
            var d = document.createElement('div');
            d.style.cssText = 'font-size:0.72rem;font-weight:600;color:#ffd93d;text-transform:uppercase;letter-spacing:0.5px;margin-top:4px;margin-bottom:1px;border-bottom:1px solid #2a3a4a;padding-bottom:3px;';
            d.textContent = text;
            pnl.appendChild(d);
        }

        function mkSlider(name, lo, hi, stp, val, cb) {
            var w = document.createElement('div');
            var lb = document.createElement('label');
            lb.style.cssText = 'display:flex;justify-content:space-between;color:#8899AA;margin-bottom:1px;font-size:0.78rem;';
            var nm = document.createElement('span'); nm.textContent = name;
            var vl = document.createElement('span'); vl.textContent = val; vl.style.color = '#D4E0ED';
            lb.appendChild(nm); lb.appendChild(vl); w.appendChild(lb);
            var inp = document.createElement('input');
            inp.type = 'range'; inp.min = lo; inp.max = hi; inp.step = stp; inp.value = val;
            inp.style.cssText = 'width:100%;accent-color:#4ecdc4;height:16px;';
            inp.oninput = function () { vl.textContent = inp.value; cb(+inp.value); };
            w.appendChild(inp); pnl.appendChild(w);
            return { inp: inp, valEl: vl };
        }

        /* ─ Reference ─ */
        sectionTitle('Reference Path');
        var rSel = document.createElement('div');
        rSel.style.cssText = 'display:flex;gap:5px;margin-bottom:2px;';
        ['figure8', 'circle'].forEach(function (t) {
            var b = document.createElement('button');
            b.textContent = t === 'figure8' ? '\u221E Figure-8' : '\u25CB Circle';
            b.dataset.t = t;
            b.style.cssText = 'flex:1;padding:5px;border:1px solid #3a4a5a;border-radius:6px;cursor:pointer;font-size:0.78rem;transition:all .15s;' +
                'background:' + (t === P.rType ? '#4ecdc4' : '#1a2332') + ';color:' + (t === P.rType ? '#0f1923' : '#D4E0ED') + ';';
            b.onclick = function () {
                P.rType = t;
                rSel.querySelectorAll('button').forEach(function (bb) {
                    var act = bb.dataset.t === t;
                    bb.style.background = act ? '#4ecdc4' : '#1a2332';
                    bb.style.color = act ? '#0f1923' : '#D4E0ED';
                });
                doReset();
            };
            rSel.appendChild(b);
        });
        pnl.appendChild(rSel);
        var scaleS = mkSlider('Scale', 1, 5, 0.5, P.rS, function (v) { P.rS = v; P.rR = v; if (!running) render(); });
        mkSlider('Period (s)', 6, 24, 1, P.rT, function (v) { P.rT = v; if (!running) render(); });

        /* ─ MPC Tuning ─ */
        sectionTitle('MPC Weights');
        mkSlider('Q position', 1, 200, 1, P.Qp, function (v) { P.Qp = v; });
        mkSlider('Q heading', 1, 100, 1, P.Qt, function (v) { P.Qt = v; });
        mkSlider('R velocity', 0.1, 20, 0.1, P.Rv, function (v) { P.Rv = v; });
        mkSlider('R angular', 0.1, 20, 0.1, P.Rw, function (v) { P.Rw = v; });
        mkSlider('Horizon N', 5, 30, 1, P.N, function (v) { P.N = v; P.warm = null; });

        /* ─ Constraints ─ */
        sectionTitle('Input Constraints');
        var vmS = mkSlider('v_max (m/s)', 0.5, 6, 0.5, P.vM, function (v) { P.vM = v; });
        var wmS = mkSlider('\u03C9_max (rad/s)', 0.5, 6, 0.5, P.wM, function (v) { P.wM = v; });

        /* ─ Initial State ─ */
        sectionTitle('Initial State');
        var x0S = mkSlider('x\u2080', -4, 4, 0.1, 1.0, function (v) { if (!running) { state[0] = v; render(); } });
        var y0S = mkSlider('y\u2080', -3, 3, 0.1, -0.5, function (v) { if (!running) { state[1] = v; render(); } });
        var th0S = mkSlider('\u03B8\u2080 (\u00B0)', -180, 180, 5, 17, function (v) { if (!running) { state[2] = v * Math.PI / 180; render(); } });

        /* ─ Simulation ─ */
        sectionTitle('Simulation');
        mkSlider('Speed', 0.5, 4, 0.5, P.speed, function (v) { P.speed = v; });

        /* Buttons */
        var bRow = document.createElement('div');
        bRow.style.cssText = 'display:flex;gap:6px;margin-top:4px;';
        var startB = document.createElement('button');
        startB.textContent = '\u25B6 Start';
        startB.style.cssText = 'flex:1;padding:8px;border:none;border-radius:6px;background:#4ecdc4;color:#0f1923;font-weight:700;cursor:pointer;font-size:0.85rem;transition:all .15s;';
        startB.onclick = function () {
            if (running) { running = false; startB.textContent = '\u25B6 Resume'; startB.style.background = '#4ecdc4'; cvs.style.cursor = 'crosshair'; }
            else { running = true; startB.textContent = '\u23F8 Pause'; startB.style.background = '#ff6b6b'; cvs.style.cursor = 'default'; lastTS = 0; raf = requestAnimationFrame(tick); }
        };
        bRow.appendChild(startB);
        var resetB = document.createElement('button');
        resetB.textContent = '\u21BA Reset';
        resetB.style.cssText = 'flex:1;padding:8px;border:1px solid #3a4a5a;border-radius:6px;background:#1a2332;color:#D4E0ED;cursor:pointer;font-size:0.85rem;transition:all .15s;';
        resetB.onclick = doReset;
        bRow.appendChild(resetB);
        pnl.appendChild(bRow);

        /* Info */
        var info = document.createElement('div');
        info.style.cssText = 'background:#1a2332;border-radius:6px;padding:8px 10px;font-size:0.76rem;color:#8899AA;line-height:1.65;margin-top:4px;';
        pnl.appendChild(info);

        /* Legend */
        var leg = document.createElement('div');
        leg.style.cssText = 'font-size:0.7rem;color:#8899AA;margin-top:2px;line-height:1.4;';
        leg.innerHTML =
            '<span style="color:#4ecdc4">\u2500 \u2500</span> Reference &nbsp; ' +
            '<span style="color:#ff6b6b">\u2501</span> Trail &nbsp; ' +
            '<span style="color:#ffb464">\u2501 \u25CF</span> MPC plan &nbsp; ' +
            '<span style="color:#fff">\u25B7</span> Robot';
        pnl.appendChild(leg);

        /* ── Canvas transform ── */
        var SC = 55, CX = cvs.width / 2, CY = cvs.height / 2;
        function w2c(wx, wy) { return [CX + wx * SC, CY - wy * SC]; }
        function c2w(cx, cy) { return [(cx - CX) / SC, -(cy - CY) / SC]; }

        /* ── Sync sliders ↔ state ── */
        function syncSlidersFromState() {
            x0S.inp.value = state[0].toFixed(1); x0S.valEl.textContent = state[0].toFixed(1);
            y0S.inp.value = state[1].toFixed(1); y0S.valEl.textContent = state[1].toFixed(1);
            var deg = Math.round(state[2] * 180 / Math.PI);
            th0S.inp.value = deg; th0S.valEl.textContent = deg;
        }

        /* ── Canvas mouse interaction ── */
        function getCanvasCoords(e) {
            var rect = cvs.getBoundingClientRect();
            var mx = (e.clientX - rect.left) * (cvs.width / rect.width);
            var my = (e.clientY - rect.top) * (cvs.height / rect.height);
            return [mx, my];
        }

        cvs.addEventListener('mousedown', function (e) {
            if (running) return;
            var mc = getCanvasCoords(e);
            var wc = c2w(mc[0], mc[1]);
            state[0] = clamp(wc[0], -5, 5);
            state[1] = clamp(wc[1], -4, 4);
            dragging = true;
            dragOrigin = mc;
            syncSlidersFromState();
            render();
        });

        cvs.addEventListener('mousemove', function (e) {
            if (!dragging) return;
            var mc = getCanvasCoords(e);
            var dx = mc[0] - dragOrigin[0], dy = -(mc[1] - dragOrigin[1]);
            if (Math.sqrt(dx * dx + dy * dy) > 8) {
                state[2] = Math.atan2(dy, dx);
                syncSlidersFromState();
                render();
            }
        });

        cvs.addEventListener('mouseup', function () { dragging = false; });
        cvs.addEventListener('mouseleave', function () { dragging = false; });

        /* Touch support */
        cvs.addEventListener('touchstart', function (e) {
            if (running) return;
            e.preventDefault();
            var t = e.touches[0];
            var fake = { clientX: t.clientX, clientY: t.clientY };
            var mc = getCanvasCoords(fake);
            var wc = c2w(mc[0], mc[1]);
            state[0] = clamp(wc[0], -5, 5);
            state[1] = clamp(wc[1], -4, 4);
            dragging = true; dragOrigin = mc;
            syncSlidersFromState(); render();
        }, { passive: false });

        cvs.addEventListener('touchmove', function (e) {
            if (!dragging) return;
            e.preventDefault();
            var t = e.touches[0];
            var mc = getCanvasCoords({ clientX: t.clientX, clientY: t.clientY });
            var dx = mc[0] - dragOrigin[0], dy = -(mc[1] - dragOrigin[1]);
            if (Math.sqrt(dx * dx + dy * dy) > 8) {
                state[2] = Math.atan2(dy, dx);
                syncSlidersFromState(); render();
            }
        }, { passive: false });

        cvs.addEventListener('touchend', function () { dragging = false; });

        /* ── Main canvas render ── */
        function render() {
            var W = cvs.width, H = cvs.height;
            ctx.fillStyle = COL.bg; ctx.fillRect(0, 0, W, H);

            /* Grid */
            ctx.strokeStyle = COL.grid; ctx.lineWidth = 0.5;
            for (var gx = CX % SC; gx < W; gx += SC) { ctx.beginPath(); ctx.moveTo(gx, 0); ctx.lineTo(gx, H); ctx.stroke(); }
            for (var gy = CY % SC; gy < H; gy += SC) { ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(W, gy); ctx.stroke(); }
            ctx.strokeStyle = COL.axis; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(0, CY); ctx.lineTo(W, CY); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(CX, 0); ctx.lineTo(CX, H); ctx.stroke();

            /* Ticks */
            ctx.fillStyle = COL.dim; ctx.font = '9px system-ui'; ctx.textAlign = 'center';
            for (var m = -5; m <= 5; m++) {
                if (m === 0) continue;
                var px = CX + m * SC;
                if (px > 15 && px < W - 15) ctx.fillText(m, px, CY + 13);
                var py = CY - m * SC;
                if (py > 10 && py < H - 5) { ctx.textAlign = 'right'; ctx.fillText(m, CX - 5, py + 3); ctx.textAlign = 'center'; }
            }
            ctx.fillText('x', W - 10, CY - 5);
            ctx.textAlign = 'left'; ctx.fillText('y', CX + 5, 12);

            /* Reference curve */
            ctx.strokeStyle = COL.ref; ctx.lineWidth = 1.5; ctx.setLineDash([6, 4]);
            ctx.beginPath();
            var rFn = P.rType === 'circle' ? circ : fig8;
            var rA = P.rType === 'circle' ? P.rR : P.rS;
            for (var t = 0; t <= P.rT + 0.01; t += 0.04) {
                var rr = rFn(t, P.rT, rA);
                var p = w2c(rr.s[0], rr.s[1]);
                t < 0.01 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]);
            }
            ctx.stroke(); ctx.setLineDash([]);

            /* Trail (gradient opacity) */
            if (trail.length > 1) {
                var segSize = Math.max(1, Math.floor(trail.length / 25));
                ctx.lineCap = 'round'; ctx.lineWidth = 2.5;
                for (var g = 0; g < trail.length - 1; g += segSize) {
                    ctx.globalAlpha = 0.1 + 0.9 * (g / trail.length);
                    ctx.strokeStyle = COL.trail;
                    ctx.beginPath();
                    var tp = w2c(trail[g][0], trail[g][1]);
                    ctx.moveTo(tp[0], tp[1]);
                    for (var i = g + 1; i < Math.min(g + segSize + 1, trail.length); i++) {
                        tp = w2c(trail[i][0], trail[i][1]);
                        ctx.lineTo(tp[0], tp[1]);
                    }
                    ctx.stroke();
                }
                ctx.globalAlpha = 1;
            }

            /* MPC predicted trajectory (line + dots) */
            if (mpcR && mpcR.pred && mpcR.pred.length > 1) {
                /* Line */
                ctx.strokeStyle = COL.pred; ctx.lineWidth = 2; ctx.globalAlpha = 0.6;
                ctx.beginPath();
                for (var i = 0; i < mpcR.pred.length; i++) {
                    var pp = w2c(mpcR.pred[i][0], mpcR.pred[i][1]);
                    i === 0 ? ctx.moveTo(pp[0], pp[1]) : ctx.lineTo(pp[0], pp[1]);
                }
                ctx.stroke();
                ctx.globalAlpha = 1;
                /* Dots */
                ctx.fillStyle = COL.pred;
                for (var i = 1; i < mpcR.pred.length; i++) {
                    var pp = w2c(mpcR.pred[i][0], mpcR.pred[i][1]);
                    ctx.beginPath(); ctx.arc(pp[0], pp[1], 3.5, 0, 6.283); ctx.fill();
                }
                /* Terminal marker */
                var last = mpcR.pred[mpcR.pred.length - 1];
                var lp = w2c(last[0], last[1]);
                ctx.strokeStyle = COL.pred; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.arc(lp[0], lp[1], 6, 0, 6.283); ctx.stroke();
            }

            /* Reference crosshair */
            if (mpcR) {
                var rp = w2c(mpcR.ref[0], mpcR.ref[1]);
                ctx.strokeStyle = COL.ref; ctx.lineWidth = 2;
                ctx.beginPath(); ctx.arc(rp[0], rp[1], 8, 0, 6.283); ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(rp[0] - 11, rp[1]); ctx.lineTo(rp[0] + 11, rp[1]);
                ctx.moveTo(rp[0], rp[1] - 11); ctx.lineTo(rp[0], rp[1] + 11);
                ctx.stroke();
            }

            /* Robot body */
            var rPt = w2c(state[0], state[1]);
            ctx.save();
            ctx.translate(rPt[0], rPt[1]);
            ctx.rotate(-state[2]);

            /* Wheels */
            ctx.fillStyle = '#6a7a8a';
            ctx.fillRect(-6, -11, 8, 4);
            ctx.fillRect(-6, 7, 8, 4);

            /* Body (arrow) */
            var L = 15;
            ctx.beginPath();
            ctx.moveTo(L + 2, 0);
            ctx.lineTo(-L * 0.5, -L * 0.55);
            ctx.quadraticCurveTo(-L * 0.3, 0, -L * 0.5, L * 0.55);
            ctx.closePath();
            ctx.fillStyle = COL.robot; ctx.fill();
            ctx.strokeStyle = COL.accent; ctx.lineWidth = 1.5; ctx.stroke();

            /* Heading indicator dot */
            ctx.fillStyle = COL.accent;
            ctx.beginPath(); ctx.arc(L + 5, 0, 3, 0, 6.283); ctx.fill();
            ctx.restore();

            /* Heading direction line (when setting up, not running) */
            if (!running && !mpcR) {
                var endX = state[0] + 0.8 * Math.cos(state[2]);
                var endY = state[1] + 0.8 * Math.sin(state[2]);
                var ep = w2c(endX, endY);
                ctx.strokeStyle = 'rgba(255,217,61,0.4)'; ctx.lineWidth = 2; ctx.setLineDash([4, 4]);
                ctx.beginPath(); ctx.moveTo(rPt[0], rPt[1]); ctx.lineTo(ep[0], ep[1]); ctx.stroke();
                ctx.setLineDash([]);
            }

            /* Info panel */
            var posE = mpcR ? Math.sqrt(Math.pow(state[0] - mpcR.ref[0], 2) + Math.pow(state[1] - mpcR.ref[1], 2)) : 0;
            var hdE = mpcR ? Math.abs(wrap(state[2] - mpcR.ref[2])) * 180 / Math.PI : 0;
            var cv = mpcR ? mpcR.u[0] : 0, cw = mpcR ? mpcR.u[1] : 0;
            info.innerHTML =
                '<b style="color:#D4E0ED">State</b>&nbsp; x=' + state[0].toFixed(2) + ' y=' + state[1].toFixed(2) + ' \u03B8=' + (state[2] * 180 / Math.PI).toFixed(1) + '\u00B0<br>' +
                '<b style="color:#D4E0ED">Error</b>&nbsp; pos <span style="color:#ffd93d">' + posE.toFixed(3) + '</span> m &middot; head <span style="color:#ffd93d">' + hdE.toFixed(1) + '\u00B0</span><br>' +
                '<b style="color:#D4E0ED">Input</b>&nbsp; v=<span style="color:#4ecdc4">' + cv.toFixed(2) + '</span> &middot; \u03C9=<span style="color:#ff6b6b">' + cw.toFixed(2) + '</span><br>' +
                '<b style="color:#D4E0ED">Time</b>&nbsp; ' + simT.toFixed(1) + 's &nbsp; <span style="color:#4ecdc4">N=' + P.N + '</span> &nbsp; <span style="color:#8899AA">Ts=' + P.dt + 's</span>';
        }

        /* ── Render strip charts ── */
        function renderCharts() {
            var cW = chartV.width, cH = chartV.height;
            drawChart(chartV.getContext('2d'), hist.v, cW, cH, COL.vCol, -P.vM - 0.5, P.vM + 0.5, 'v(t)', 'm/s', P.vM);
            drawChart(chartW.getContext('2d'), hist.w, cW, cH, COL.wCol, -P.wM - 0.5, P.wM + 0.5, '\u03C9(t)', 'rad/s', P.wM);
            var eMax = Math.max(1, hist.e.length > 0 ? Math.max.apply(null, hist.e) * 1.3 : 1);
            drawChart(chartE.getContext('2d'), hist.e, cW, cH, COL.err, 0, eMax, 'error(t)', 'm');
        }

        /* ── Simulation step ── */
        function simStep() {
            mpcR = solveMPC(state, simT, P);
            P.warm = mpcR.warm;

            /* Log history */
            hist.v.push(mpcR.u[0]); hist.w.push(mpcR.u[1]);
            hist.e.push(Math.sqrt(Math.pow(state[0] - mpcR.ref[0], 2) + Math.pow(state[1] - mpcR.ref[1], 2)));
            if (hist.v.length > hist.maxLen) { hist.v.shift(); hist.w.shift(); hist.e.shift(); }

            state = unicycle(state, mpcR.u, P.dt);
            state[2] = wrap(state[2]);
            simT += P.dt;
            trail.push(state.slice());
            if (trail.length > 500) trail.shift();
        }

        /* ── Animation loop ── */
        var lastTS = 0;
        function tick(ts) {
            if (!running) return;
            if (!lastTS) lastTS = ts;
            var dt = Math.min((ts - lastTS) / 1000, 0.05);
            lastTS = ts;
            var steps = Math.max(1, Math.min(Math.round(P.speed * dt / P.dt), 6));
            for (var i = 0; i < steps; i++) simStep();
            render();
            renderCharts();
            raf = requestAnimationFrame(tick);
        }

        /* ── Reset ── */
        function doReset() {
            running = false;
            startB.textContent = '\u25B6 Start'; startB.style.background = '#4ecdc4';
            cvs.style.cursor = 'crosshair';
            if (raf) cancelAnimationFrame(raf);
            simT = 0; P.warm = null;
            var rFn = P.rType === 'circle' ? circ : fig8;
            var rA = P.rType === 'circle' ? P.rR : P.rS;
            var r0 = rFn(0, P.rT, rA);
            state = [r0.s[0] + 1.0, r0.s[1] - 0.5, r0.s[2] + 0.3];
            trail = []; mpcR = null;
            hist.v = []; hist.w = []; hist.e = [];
            syncSlidersFromState();
            render();
            renderCharts();
        }

        /* Resize chart canvases to fit */
        function resizeCharts() {
            var totalW = chartRow.clientWidth;
            if (totalW < 100) return;
            var each = Math.floor((totalW - 12) / 3);
            chartV.width = each; chartW.width = each; chartE.width = each;
            renderCharts();
        }
        window.addEventListener('resize', resizeCharts);

        doReset();
        setTimeout(resizeCharts, 100);
    }

    /* ─── Public API ──────────────────────────────────────── */
    function initAll(root) {
        var els = (root || document).querySelectorAll('.robot-demo-container');
        for (var i = 0; i < els.length; i++) initDemo(els[i]);
    }
    window.RobotDemo = { initAll: initAll, init: initDemo };
})();
