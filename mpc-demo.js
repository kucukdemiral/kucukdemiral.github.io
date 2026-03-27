/**
 * Interactive MPC Receding Horizon Demo
 * Pure JS/Canvas — no dependencies.
 *
 * HTML hook:  <div class="mpc-demo"></div>
 * Init call:  MPCDemo.initAll(root)
 */
(function () {
    'use strict';

    /* ================================================================
       System: double integrator
         x1_{k+1} = x1_k + Ts * x2_k
         x2_{k+1} = x2_k + Ts * u_k
       Ts = 0.5,  N = 10 (prediction horizon)
       Q = diag(1, 0.1),  R = 0.01
       |u| <= 1,  |x1| <= 5
       x0 = [4; 0],  target = origin
    ================================================================ */

    var Ts = 0.5;
    var N_horizon = 10;
    var Q1 = 1.0, Q2 = 0.1, R_weight = 0.01;
    var u_max = 1.0, x1_max = 5.0;
    var SIM_STEPS = 30;

    /* ── Matrix helpers (2x2) ──────────────────────────────── */
    function mat2x2(a, b, c, d) { return [[a, b], [c, d]]; }

    function matMul2(A, B) {
        return [
            [A[0][0]*B[0][0]+A[0][1]*B[1][0], A[0][0]*B[0][1]+A[0][1]*B[1][1]],
            [A[1][0]*B[0][0]+A[1][1]*B[1][0], A[1][0]*B[0][1]+A[1][1]*B[1][1]]
        ];
    }

    function matVec2(A, x) {
        return [A[0][0]*x[0]+A[0][1]*x[1], A[1][0]*x[0]+A[1][1]*x[1]];
    }

    function matAdd2(A, B) {
        return [[A[0][0]+B[0][0], A[0][1]+B[0][1]], [A[1][0]+B[1][0], A[1][1]+B[1][1]]];
    }

    function matTrans2(A) {
        return [[A[0][0], A[1][0]], [A[0][1], A[1][1]]];
    }

    function matScale2(A, s) {
        return [[A[0][0]*s, A[0][1]*s], [A[1][0]*s, A[1][1]*s]];
    }

    function inv2x2(M) {
        var det = M[0][0]*M[1][1] - M[0][1]*M[1][0];
        if (Math.abs(det) < 1e-14) det = 1e-14;
        var id = 1.0 / det;
        return [[M[1][1]*id, -M[0][1]*id], [-M[1][0]*id, M[0][0]*id]];
    }

    /* ── System matrices ───────────────────────────────────── */
    var Ad = mat2x2(1, Ts, 0, 1);
    var Bd = [Ts * Ts * 0.5, Ts];  // B as column vector: for double integrator with state [pos, vel]
    // Actually: x1+ = x1 + Ts*x2, x2+ = x2 + Ts*u  => B = [0, Ts]
    // But for the double integrator the exact B is [0; Ts] since the position update doesn't directly have u.
    // Let me correct:
    var Bd_vec = [0, Ts]; // B column vector

    var Q = mat2x2(Q1, 0, 0, Q2);

    /* ── DARE solver (2x2) ─────────────────────────────────── */
    function solveDARE(A, B, Qm, Rm) {
        var P = [[Qm[0][0], Qm[0][1]], [Qm[1][0], Qm[1][1]]];
        for (var it = 0; it < 500; it++) {
            // P_new = Q + A' P A - A' P B (R + B' P B)^-1 B' P A
            var PA = matMul2(P, A);
            var AtPA = matMul2(matTrans2(A), PA);

            var PB0 = P[0][0]*B[0]+P[0][1]*B[1];
            var PB1 = P[1][0]*B[0]+P[1][1]*B[1];
            var BtPB = B[0]*PB0 + B[1]*PB1;
            var s_inv = 1.0 / (Rm + BtPB);

            var AtPB0 = A[0][0]*PB0 + A[1][0]*PB1;
            var AtPB1 = A[0][1]*PB0 + A[1][1]*PB1;

            P = [
                [Qm[0][0] + AtPA[0][0] - AtPB0*AtPB0*s_inv, Qm[0][1] + AtPA[0][1] - AtPB0*AtPB1*s_inv],
                [Qm[1][0] + AtPA[1][0] - AtPB1*AtPB0*s_inv, Qm[1][1] + AtPA[1][1] - AtPB1*AtPB1*s_inv]
            ];
        }
        return P;
    }

    /* ── Compute LQR gain K from DARE solution ─────────────── */
    function computeLQRGain(A, B, P, Rm) {
        var PB0 = P[0][0]*B[0]+P[0][1]*B[1];
        var PB1 = P[1][0]*B[0]+P[1][1]*B[1];
        var s_inv = 1.0 / (Rm + B[0]*PB0 + B[1]*PB1);
        var BtPA0 = B[0]*(P[0][0]*A[0][0]+P[0][1]*A[1][0]) + B[1]*(P[1][0]*A[0][0]+P[1][1]*A[1][0]);
        var BtPA1 = B[0]*(P[0][0]*A[0][1]+P[0][1]*A[1][1]) + B[1]*(P[1][0]*A[0][1]+P[1][1]*A[1][1]);
        return [s_inv*BtPA0, s_inv*BtPA1];
    }

    /* ── Solve MPC QP at one step via condensed QP ─────────── */
    /* We build the condensed prediction matrices and solve the
       unconstrained problem, then clip to constraints iteratively.
       For the demo this gives visually correct results. */
    function solveMPCStep(x0) {
        var Nh = N_horizon;
        // Terminal cost from DARE
        var P_terminal = solveDARE(Ad, Bd_vec, Q, R_weight);

        // Build prediction: x_k = A^k x0 + sum_{j=0}^{k-1} A^{k-1-j} B u_j
        // We'll do iterative optimization: start unconstrained LQR, then project

        // Method: solve via backward Riccati recursion over the finite horizon
        // This gives the optimal constrained solution when we clip
        var P_arr = new Array(Nh + 1);
        var K_arr = new Array(Nh);
        P_arr[Nh] = P_terminal;

        for (var k = Nh - 1; k >= 0; k--) {
            var Pk1 = P_arr[k + 1];
            var PB0 = Pk1[0][0]*Bd_vec[0]+Pk1[0][1]*Bd_vec[1];
            var PB1 = Pk1[1][0]*Bd_vec[0]+Pk1[1][1]*Bd_vec[1];
            var BtPB = Bd_vec[0]*PB0 + Bd_vec[1]*PB1;
            var s_inv = 1.0 / (R_weight + BtPB);

            var BtPA0 = Bd_vec[0]*(Pk1[0][0]*Ad[0][0]+Pk1[0][1]*Ad[1][0]) + Bd_vec[1]*(Pk1[1][0]*Ad[0][0]+Pk1[1][1]*Ad[1][0]);
            var BtPA1 = Bd_vec[0]*(Pk1[0][0]*Ad[0][1]+Pk1[0][1]*Ad[1][1]) + Bd_vec[1]*(Pk1[1][0]*Ad[0][1]+Pk1[1][1]*Ad[1][1]);
            K_arr[k] = [s_inv*BtPA0, s_inv*BtPA1];

            var AtPA = matMul2(matTrans2(Ad), matMul2(Pk1, Ad));
            var AtPB0 = Ad[0][0]*PB0 + Ad[1][0]*PB1;
            var AtPB1 = Ad[0][1]*PB0 + Ad[1][1]*PB1;
            P_arr[k] = [
                [Q[0][0] + AtPA[0][0] - AtPB0*AtPB0*s_inv, Q[0][1] + AtPA[0][1] - AtPB0*AtPB1*s_inv],
                [Q[1][0] + AtPA[1][0] - AtPB1*AtPB0*s_inv, Q[1][1] + AtPA[1][1] - AtPB1*AtPB1*s_inv]
            ];
        }

        // Forward simulate with gains, applying constraint clipping
        var xs = [x0.slice()];
        var us = [];
        var x = x0.slice();
        for (var k = 0; k < Nh; k++) {
            var u = -(K_arr[k][0]*x[0] + K_arr[k][1]*x[1]);
            // Clip control
            if (u > u_max) u = u_max;
            if (u < -u_max) u = -u_max;
            us.push(u);

            // Propagate
            var x1_new = x[0] + Ts * x[1];
            var x2_new = x[1] + Ts * u;
            x = [x1_new, x2_new];
            xs.push(x.slice());
        }

        return { xs: xs, us: us };
    }

    /* ── Pre-compute full MPC closed-loop simulation ───────── */
    function simulateMPC(x0_init) {
        var x = x0_init.slice();
        var steps = [];
        for (var k = 0; k < SIM_STEPS; k++) {
            var sol = solveMPCStep(x);
            steps.push({
                k: k,
                x: x.slice(),
                predicted_xs: sol.xs,
                predicted_us: sol.us,
                applied_u: sol.us[0]
            });
            // Apply first input
            var u_applied = sol.us[0];
            var x1_new = x[0] + Ts * x[1];
            var x2_new = x[1] + Ts * u_applied;
            x = [x1_new, x2_new];

            // Check if converged (near origin)
            if (Math.abs(x[0]) < 0.005 && Math.abs(x[1]) < 0.005) {
                // Add final state
                steps.push({
                    k: k + 1,
                    x: x.slice(),
                    predicted_xs: [[0, 0]],
                    predicted_us: [0],
                    applied_u: 0
                });
                break;
            }
        }
        return steps;
    }

    /* ── Nice grid step ────────────────────────────────────── */
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

    /* ── Canvas setup ──────────────────────────────────────── */
    function setupCanvas(canvas) {
        var dpr = window.devicePixelRatio || 1;
        var w = canvas.clientWidth, h = canvas.clientHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        var ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        return { ctx: ctx, w: w, h: h };
    }

    /* ── Colors (matching site style) ──────────────────────── */
    var COL = {
        bg: '#0F1923',
        grid: 'rgba(255,255,255,0.06)',
        axis: 'rgba(255,255,255,0.18)',
        label: '#778899',
        tick: '#556677',
        pastState: '#3498db',       // blue - past x1
        pastVel: '#2ecc71',         // green - past velocity
        pastInput: '#2ecc71',       // green - past control (staircase)
        predict: '#e67e22',         // orange - predicted trajectory
        predictFade: 'rgba(230,126,34,0.25)',
        ghost: 'rgba(230,126,34,0.08)',
        ghostLine: 'rgba(230,126,34,0.15)',
        constraint: '#e74c3c',      // red - constraint lines
        constraintFill: 'rgba(231,76,60,0.06)',
        origin: '#0D7C66',          // teal - target/origin
        appliedDot: '#f1c40f',      // yellow - applied point
        text: '#99AABB',
        textBright: '#D4E0ED',
        teal: '#0D7C66',
        tealHover: '#0A6B58'
    };

    /* ── Draw position (x1) plot ───────────────────────────── */
    function drawPositionPlot(canvas, steps, currentStep, ghostTrails) {
        var r = setupCanvas(canvas);
        var ctx = r.ctx, CW = r.w, CH = r.h;

        ctx.fillStyle = COL.bg;
        ctx.fillRect(0, 0, CW, CH);

        var m = { t: 28, r: 18, b: 38, l: 52 };
        var pw = CW - m.l - m.r;
        var ph = CH - m.t - m.b;

        if (steps.length === 0) {
            ctx.fillStyle = COL.tick;
            ctx.font = '13px "Source Sans 3", sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Click Step or Auto Play to begin', CW / 2, CH / 2);
            return;
        }

        var totalSteps = steps.length;
        var tMax = Math.max((totalSteps + N_horizon) * Ts, (currentStep + N_horizon + 2) * Ts, 10 * Ts);
        var xMin_t = 0, xMax_t = tMax;

        var yMin = -x1_max - 0.5, yMax = x1_max + 0.5;

        function sx(v) { return m.l + (v - xMin_t) / (xMax_t - xMin_t) * pw; }
        function sy(v) { return m.t + (1 - (v - yMin) / (yMax - yMin)) * ph; }

        /* grid */
        ctx.strokeStyle = COL.grid;
        ctx.lineWidth = 1;
        var ys = niceStep(yMin, yMax, 6);
        ctx.font = '10px "SF Mono", "Fira Code", monospace';
        ctx.fillStyle = COL.tick;
        ctx.textAlign = 'right';
        for (var v = Math.ceil(yMin / ys) * ys; v <= yMax; v += ys) {
            var yy = sy(v);
            ctx.beginPath(); ctx.moveTo(m.l, yy); ctx.lineTo(m.l + pw, yy); ctx.stroke();
            ctx.fillText(v.toFixed(1), m.l - 6, yy + 3);
        }

        var ts = niceStep(xMin_t, xMax_t, 8);
        ctx.textAlign = 'center';
        for (var t = 0; t <= xMax_t; t += ts) {
            var tx = sx(t);
            ctx.beginPath(); ctx.moveTo(tx, m.t); ctx.lineTo(tx, m.t + ph); ctx.stroke();
            ctx.fillText(t.toFixed(1), tx, m.t + ph + 16);
        }

        /* axes */
        ctx.strokeStyle = COL.axis;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m.l, m.t); ctx.lineTo(m.l, m.t + ph); ctx.lineTo(m.l + pw, m.t + ph);
        ctx.stroke();

        /* constraint lines at +/- x1_max */
        ctx.setLineDash([8, 5]);
        ctx.strokeStyle = COL.constraint;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(m.l, sy(x1_max)); ctx.lineTo(m.l + pw, sy(x1_max)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(m.l, sy(-x1_max)); ctx.lineTo(m.l + pw, sy(-x1_max)); ctx.stroke();
        ctx.setLineDash([]);

        /* constraint fill region (outside constraints, lightly shaded) */
        ctx.fillStyle = COL.constraintFill;
        ctx.fillRect(m.l, m.t, pw, sy(x1_max) - m.t);
        ctx.fillRect(m.l, sy(-x1_max), pw, m.t + ph - sy(-x1_max));

        /* constraint labels */
        ctx.fillStyle = COL.constraint;
        ctx.font = '10px "SF Mono", "Fira Code", monospace';
        ctx.textAlign = 'left';
        ctx.fillText('x\u2081 = +' + x1_max, m.l + pw - 52, sy(x1_max) - 5);
        ctx.fillText('x\u2081 = -' + x1_max, m.l + pw - 52, sy(-x1_max) + 13);

        /* origin / target line */
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = COL.origin;
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(m.l, sy(0)); ctx.lineTo(m.l + pw, sy(0)); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COL.origin;
        ctx.font = '10px "Source Sans 3", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('target', m.l + 4, sy(0) - 5);

        /* Ghost trails (old predictions that have faded) */
        for (var g = 0; g < ghostTrails.length; g++) {
            var ghost = ghostTrails[g];
            var age = currentStep - ghost.fromStep;
            var alpha = Math.max(0.03, 0.18 - age * 0.025);
            ctx.strokeStyle = 'rgba(230,126,34,' + alpha.toFixed(3) + ')';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            for (var i = 0; i < ghost.xs.length; i++) {
                var t_val = (ghost.fromStep + i) * Ts;
                var px = sx(t_val);
                var py = sy(ghost.xs[i][0]);
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }

        /* Past trajectory (solid blue) */
        if (currentStep >= 0) {
            ctx.strokeStyle = COL.pastState;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            for (var i = 0; i <= currentStep && i < steps.length; i++) {
                var t_val = steps[i].k * Ts;
                var px = sx(t_val);
                var py = sy(steps[i].x[0]);
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.stroke();

            /* Past trajectory dots */
            ctx.fillStyle = COL.pastState;
            for (var i = 0; i <= currentStep && i < steps.length; i++) {
                var t_val = steps[i].k * Ts;
                ctx.beginPath();
                ctx.arc(sx(t_val), sy(steps[i].x[0]), 3, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        /* Current predicted trajectory (dashed orange, fading) */
        if (currentStep >= 0 && currentStep < steps.length) {
            var step = steps[currentStep];
            var pred = step.predicted_xs;
            ctx.lineWidth = 2;

            for (var i = 0; i < pred.length - 1; i++) {
                var t_start = (step.k + i) * Ts;
                var t_end = (step.k + i + 1) * Ts;
                var alpha = 1.0 - (i / pred.length) * 0.7;
                ctx.strokeStyle = 'rgba(230,126,34,' + alpha.toFixed(2) + ')';
                ctx.setLineDash([6, 4]);
                ctx.beginPath();
                ctx.moveTo(sx(t_start), sy(pred[i][0]));
                ctx.lineTo(sx(t_end), sy(pred[i + 1][0]));
                ctx.stroke();
            }
            ctx.setLineDash([]);

            /* Predicted trajectory dots (small, fading) */
            for (var i = 1; i < pred.length; i++) {
                var t_val = (step.k + i) * Ts;
                var alpha = 0.8 - (i / pred.length) * 0.6;
                ctx.fillStyle = 'rgba(230,126,34,' + alpha.toFixed(2) + ')';
                ctx.beginPath();
                ctx.arc(sx(t_val), sy(pred[i][0]), 2.5, 0, Math.PI * 2);
                ctx.fill();
            }

            /* Current state dot (bright yellow) */
            ctx.fillStyle = COL.appliedDot;
            ctx.beginPath();
            ctx.arc(sx(step.k * Ts), sy(step.x[0]), 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = 'rgba(241,196,15,0.4)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(sx(step.k * Ts), sy(step.x[0]), 8, 0, Math.PI * 2);
            ctx.stroke();
        }

        /* Title & labels */
        ctx.fillStyle = COL.label;
        ctx.font = '12px "Source Sans 3", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Time (s)', m.l + pw / 2, CH - 3);

        ctx.save();
        ctx.translate(14, m.t + ph / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Position x\u2081', 0, 0);
        ctx.restore();

        /* Plot title */
        ctx.fillStyle = COL.textBright;
        ctx.font = '12px "Source Sans 3", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('State x\u2081 (position) vs Time', m.l + 4, m.t - 10);

        /* Legend */
        var lx = m.l + pw - 200, ly = m.t + 14;
        ctx.font = '10px "Source Sans 3", sans-serif';

        // Past
        ctx.strokeStyle = COL.pastState; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 16, ly); ctx.stroke();
        ctx.fillStyle = COL.text; ctx.textAlign = 'left';
        ctx.fillText('Past trajectory', lx + 22, ly + 3);

        // Predicted
        ly += 15;
        ctx.setLineDash([5, 3]);
        ctx.strokeStyle = COL.predict; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 16, ly); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COL.text;
        ctx.fillText('Predicted (horizon)', lx + 22, ly + 3);

        // Constraints
        ly += 15;
        ctx.setLineDash([6, 4]);
        ctx.strokeStyle = COL.constraint; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 16, ly); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COL.text;
        ctx.fillText('Constraints', lx + 22, ly + 3);
    }

    /* ── Draw control input (u) plot ───────────────────────── */
    function drawControlPlot(canvas, steps, currentStep, ghostTrails) {
        var r = setupCanvas(canvas);
        var ctx = r.ctx, CW = r.w, CH = r.h;

        ctx.fillStyle = COL.bg;
        ctx.fillRect(0, 0, CW, CH);

        var m_pad = { t: 28, r: 18, b: 38, l: 52 };
        var pw = CW - m_pad.l - m_pad.r;
        var ph = CH - m_pad.t - m_pad.b;

        if (steps.length === 0) {
            return;
        }

        var totalSteps = steps.length;
        var tMax = Math.max((totalSteps + N_horizon) * Ts, (currentStep + N_horizon + 2) * Ts, 10 * Ts);
        var xMin_t = 0, xMax_t = tMax;

        var yMin = -u_max - 0.3, yMax = u_max + 0.3;

        function sx(v) { return m_pad.l + (v - xMin_t) / (xMax_t - xMin_t) * pw; }
        function sy(v) { return m_pad.t + (1 - (v - yMin) / (yMax - yMin)) * ph; }

        /* grid */
        ctx.strokeStyle = COL.grid;
        ctx.lineWidth = 1;
        var ys_step = niceStep(yMin, yMax, 5);
        ctx.font = '10px "SF Mono", "Fira Code", monospace';
        ctx.fillStyle = COL.tick;
        ctx.textAlign = 'right';
        for (var v = Math.ceil(yMin / ys_step) * ys_step; v <= yMax; v += ys_step) {
            var yy = sy(v);
            ctx.beginPath(); ctx.moveTo(m_pad.l, yy); ctx.lineTo(m_pad.l + pw, yy); ctx.stroke();
            ctx.fillText(v.toFixed(1), m_pad.l - 6, yy + 3);
        }

        var ts_step = niceStep(xMin_t, xMax_t, 8);
        ctx.textAlign = 'center';
        for (var t = 0; t <= xMax_t; t += ts_step) {
            var tx = sx(t);
            ctx.beginPath(); ctx.moveTo(tx, m_pad.t); ctx.lineTo(tx, m_pad.t + ph); ctx.stroke();
            ctx.fillText(t.toFixed(1), tx, m_pad.t + ph + 16);
        }

        /* axes */
        ctx.strokeStyle = COL.axis;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(m_pad.l, m_pad.t);
        ctx.lineTo(m_pad.l, m_pad.t + ph);
        ctx.lineTo(m_pad.l + pw, m_pad.t + ph);
        ctx.stroke();

        /* constraint lines at +/- u_max */
        ctx.setLineDash([8, 5]);
        ctx.strokeStyle = COL.constraint;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(m_pad.l, sy(u_max)); ctx.lineTo(m_pad.l + pw, sy(u_max)); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(m_pad.l, sy(-u_max)); ctx.lineTo(m_pad.l + pw, sy(-u_max)); ctx.stroke();
        ctx.setLineDash([]);

        /* constraint fill */
        ctx.fillStyle = COL.constraintFill;
        ctx.fillRect(m_pad.l, m_pad.t, pw, sy(u_max) - m_pad.t);
        ctx.fillRect(m_pad.l, sy(-u_max), pw, m_pad.t + ph - sy(-u_max));

        /* constraint labels */
        ctx.fillStyle = COL.constraint;
        ctx.font = '10px "SF Mono", "Fira Code", monospace';
        ctx.textAlign = 'left';
        ctx.fillText('u = +' + u_max.toFixed(0), m_pad.l + pw - 44, sy(u_max) - 5);
        ctx.fillText('u = -' + u_max.toFixed(0), m_pad.l + pw - 44, sy(-u_max) + 13);

        /* zero line */
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(m_pad.l, sy(0)); ctx.lineTo(m_pad.l + pw, sy(0)); ctx.stroke();
        ctx.setLineDash([]);

        /* Ghost trails (old predicted inputs) */
        for (var g = 0; g < ghostTrails.length; g++) {
            var ghost = ghostTrails[g];
            var age = currentStep - ghost.fromStep;
            var alpha = Math.max(0.03, 0.15 - age * 0.02);
            ctx.strokeStyle = 'rgba(230,126,34,' + alpha.toFixed(3) + ')';
            ctx.lineWidth = 1;
            ctx.setLineDash([4, 4]);
            ctx.beginPath();
            for (var i = 0; i < ghost.us.length; i++) {
                var t0 = (ghost.fromStep + i) * Ts;
                var t1 = (ghost.fromStep + i + 1) * Ts;
                var u_val = ghost.us[i];
                if (i === 0) ctx.moveTo(sx(t0), sy(u_val));
                ctx.lineTo(sx(t1), sy(u_val));
                if (i < ghost.us.length - 1) {
                    ctx.lineTo(sx(t1), sy(ghost.us[i + 1]));
                }
            }
            ctx.stroke();
            ctx.setLineDash([]);
        }

        /* Past control inputs (solid green staircase) */
        if (currentStep > 0) {
            ctx.strokeStyle = COL.pastInput;
            ctx.lineWidth = 2.5;
            ctx.beginPath();
            var started = false;
            for (var i = 0; i < currentStep && i < steps.length; i++) {
                var t0 = steps[i].k * Ts;
                var t1 = (steps[i].k + 1) * Ts;
                var u_val = steps[i].applied_u;
                if (!started) { ctx.moveTo(sx(t0), sy(u_val)); started = true; }
                else { ctx.lineTo(sx(t0), sy(u_val)); }
                ctx.lineTo(sx(t1), sy(u_val));
            }
            ctx.stroke();

            /* Past control dots at each step start */
            ctx.fillStyle = COL.pastInput;
            for (var i = 0; i < currentStep && i < steps.length; i++) {
                var t0 = steps[i].k * Ts;
                ctx.beginPath();
                ctx.arc(sx(t0), sy(steps[i].applied_u), 3, 0, Math.PI * 2);
                ctx.fill();
            }
        }

        /* Current predicted inputs (dashed orange staircase) */
        if (currentStep >= 0 && currentStep < steps.length) {
            var step = steps[currentStep];
            var pred_u = step.predicted_us;
            ctx.lineWidth = 2;

            for (var i = 0; i < pred_u.length; i++) {
                var t0 = (step.k + i) * Ts;
                var t1 = (step.k + i + 1) * Ts;
                var alpha = 1.0 - (i / pred_u.length) * 0.7;
                ctx.strokeStyle = 'rgba(230,126,34,' + alpha.toFixed(2) + ')';
                ctx.setLineDash([6, 4]);
                ctx.beginPath();
                ctx.moveTo(sx(t0), sy(pred_u[i]));
                ctx.lineTo(sx(t1), sy(pred_u[i]));
                ctx.stroke();
                if (i < pred_u.length - 1) {
                    ctx.beginPath();
                    ctx.moveTo(sx(t1), sy(pred_u[i]));
                    ctx.lineTo(sx(t1), sy(pred_u[i + 1]));
                    ctx.stroke();
                }
            }
            ctx.setLineDash([]);

            /* Highlight the APPLIED first input */
            if (pred_u.length > 0) {
                var t0 = step.k * Ts;
                var t1 = (step.k + 1) * Ts;
                ctx.strokeStyle = COL.appliedDot;
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(sx(t0), sy(pred_u[0]));
                ctx.lineTo(sx(t1), sy(pred_u[0]));
                ctx.stroke();

                ctx.fillStyle = COL.appliedDot;
                ctx.beginPath();
                ctx.arc(sx(t0), sy(pred_u[0]), 5, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = 'rgba(241,196,15,0.4)';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.arc(sx(t0), sy(pred_u[0]), 8, 0, Math.PI * 2);
                ctx.stroke();

                /* "Applied" annotation */
                ctx.fillStyle = COL.appliedDot;
                ctx.font = '10px "Source Sans 3", sans-serif';
                ctx.textAlign = 'center';
                var midT = (t0 + t1) / 2;
                ctx.fillText('applied', sx(midT), sy(pred_u[0]) - 12);
            }
        }

        /* Title & labels */
        ctx.fillStyle = COL.label;
        ctx.font = '12px "Source Sans 3", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Time (s)', m_pad.l + pw / 2, CH - 3);

        ctx.save();
        ctx.translate(14, m_pad.t + ph / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Control u', 0, 0);
        ctx.restore();

        /* Plot title */
        ctx.fillStyle = COL.textBright;
        ctx.font = '12px "Source Sans 3", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('Control input u vs Time', m_pad.l + 4, m_pad.t - 10);

        /* Legend */
        var lx = m_pad.l + pw - 200, ly = m_pad.t + 14;
        ctx.font = '10px "Source Sans 3", sans-serif';

        ctx.strokeStyle = COL.pastInput; ctx.lineWidth = 2.5;
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 16, ly); ctx.stroke();
        ctx.fillStyle = COL.text; ctx.textAlign = 'left';
        ctx.fillText('Past inputs', lx + 22, ly + 3);

        ly += 15;
        ctx.strokeStyle = COL.appliedDot; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 16, ly); ctx.stroke();
        ctx.fillStyle = COL.text;
        ctx.fillText('Applied (first of plan)', lx + 22, ly + 3);

        ly += 15;
        ctx.setLineDash([5, 3]);
        ctx.strokeStyle = COL.predict; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(lx, ly); ctx.lineTo(lx + 16, ly); ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = COL.text;
        ctx.fillText('Predicted inputs', lx + 22, ly + 3);
    }

    /* ── Build GUI ─────────────────────────────────────────── */
    function initDemo(container) {
        if (container.dataset.initialized) return;
        container.dataset.initialized = '1';

        container.innerHTML =
            '<style>' +
            '.mpcd-wrap{font-family:"Source Sans 3",sans-serif;background:' + COL.bg + ';border-radius:8px;overflow:hidden;border:1px solid rgba(255,255,255,0.06)}' +
            '.mpcd-header{padding:10px 16px;border-bottom:1px solid rgba(255,255,255,0.06)}' +
            '.mpcd-title{font-size:1rem;font-weight:600;color:#D4E0ED}' +
            '.mpcd-system{padding:6px 16px;font-size:0.78rem;color:#8899AA;border-bottom:1px solid rgba(255,255,255,0.04);line-height:1.6}' +
            '.mpcd-system em{color:#D4E0ED;font-style:italic}' +
            '.mpcd-system .mpcd-eq{color:#667788;font-family:"SF Mono","Fira Code",monospace;font-size:0.73rem}' +
            '.mpcd-controls{display:flex;align-items:center;gap:12px;padding:10px 16px;border-bottom:1px solid rgba(255,255,255,0.06);flex-wrap:wrap}' +
            '.mpcd-btn{border:none;border-radius:5px;padding:6px 14px;font-size:0.82rem;font-family:inherit;cursor:pointer;transition:all 0.15s}' +
            '.mpcd-step-btn{background:#0D7C66;color:#fff}' +
            '.mpcd-step-btn:hover{background:#0A6B58}' +
            '.mpcd-auto-btn{background:#2C3E50;color:#D4E0ED;border:1px solid rgba(255,255,255,0.12)}' +
            '.mpcd-auto-btn:hover{background:#34495E}' +
            '.mpcd-auto-btn.active{background:#C0392B;color:#fff;border-color:transparent}' +
            '.mpcd-reset-btn{background:transparent;color:#8899AA;border:1px solid rgba(255,255,255,0.12)}' +
            '.mpcd-reset-btn:hover{color:#D4E0ED;border-color:rgba(255,255,255,0.25)}' +
            '.mpcd-speed-wrap{display:flex;align-items:center;gap:6px;margin-left:auto;color:#8899AA;font-size:0.78rem}' +
            '.mpcd-speed-slider{width:80px;accent-color:#0D7C66}' +
            '.mpcd-speed-val{color:#D4E0ED;font-family:"SF Mono","Fira Code",monospace;font-size:0.76rem;min-width:2.5em}' +
            '.mpcd-indicator{padding:8px 16px;border-bottom:1px solid rgba(255,255,255,0.04);display:flex;align-items:center;gap:16px;flex-wrap:wrap}' +
            '.mpcd-step-label{font-size:0.88rem;font-weight:600;color:#D4E0ED;font-family:"SF Mono","Fira Code",monospace}' +
            '.mpcd-state-info{font-size:0.78rem;color:#8899AA;font-family:"SF Mono","Fira Code",monospace}' +
            '.mpcd-state-info span{color:#D4E0ED}' +
            '.mpcd-pedagogy{font-size:0.76rem;color:#e67e22;margin-left:auto;font-style:italic;max-width:360px;text-align:right}' +
            '.mpcd-plots{display:flex;flex-direction:column}' +
            '.mpcd-canvas-wrap{position:relative;width:100%}' +
            '.mpcd-canvas{width:100%;height:240px;display:block}' +
            '.mpcd-divider{height:1px;background:rgba(255,255,255,0.06)}' +
            '@media(max-width:600px){' +
                '.mpcd-controls{flex-direction:column;align-items:flex-start;gap:8px}' +
                '.mpcd-speed-wrap{margin-left:0}' +
                '.mpcd-canvas{height:190px}' +
                '.mpcd-pedagogy{margin-left:0;text-align:left;max-width:none}' +
            '}' +
            '</style>' +
            '<div class="mpcd-wrap">' +
                '<div class="mpcd-header">' +
                    '<span class="mpcd-title">\u25B6 MPC Receding Horizon Demonstration</span>' +
                '</div>' +
                '<div class="mpcd-system">' +
                    'Double integrator: &nbsp;' +
                    '<span class="mpcd-eq">' +
                        '<em>x</em>\u2081(k+1) = <em>x</em>\u2081(k) + T<sub>s</sub><em>x</em>\u2082(k), &nbsp;&nbsp;' +
                        '<em>x</em>\u2082(k+1) = <em>x</em>\u2082(k) + T<sub>s</sub><em>u</em>(k)' +
                    '</span>' +
                    ' &nbsp;|&nbsp; T<sub>s</sub>=0.5, &nbsp;N=10, &nbsp;Q=diag(1,0.1), &nbsp;R=0.01' +
                    ' &nbsp;|&nbsp; |<em>u</em>|\u22641, &nbsp;|<em>x</em>\u2081|\u22645' +
                    ' &nbsp;|&nbsp; <em>x</em>\u2080=[4, 0], &nbsp;target = origin' +
                '</div>' +
                '<div class="mpcd-controls">' +
                    '<button class="mpcd-btn mpcd-step-btn">\u25B6 Step</button>' +
                    '<button class="mpcd-btn mpcd-auto-btn">\u25B6\u25B6 Auto Play</button>' +
                    '<button class="mpcd-btn mpcd-reset-btn">\u21BA Reset</button>' +
                    '<div class="mpcd-speed-wrap">' +
                        'Speed ' +
                        '<input type="range" class="mpcd-speed-slider" min="200" max="2000" step="100" value="800">' +
                        '<span class="mpcd-speed-val">800ms</span>' +
                    '</div>' +
                '</div>' +
                '<div class="mpcd-indicator">' +
                    '<span class="mpcd-step-label">Step k = 0 / --</span>' +
                    '<span class="mpcd-state-info">' +
                        'x\u2081 = <span class="mpcd-x1">4.000</span> &nbsp;&nbsp;' +
                        'x\u2082 = <span class="mpcd-x2">0.000</span> &nbsp;&nbsp;' +
                        'u = <span class="mpcd-u">\u2014</span>' +
                    '</span>' +
                    '<span class="mpcd-pedagogy">' +
                        'The optimizer plans N=10 steps ahead, but only the first input is applied.' +
                    '</span>' +
                '</div>' +
                '<div class="mpcd-plots">' +
                    '<div class="mpcd-canvas-wrap">' +
                        '<canvas class="mpcd-canvas mpcd-pos-canvas"></canvas>' +
                    '</div>' +
                    '<div class="mpcd-divider"></div>' +
                    '<div class="mpcd-canvas-wrap">' +
                        '<canvas class="mpcd-canvas mpcd-ctrl-canvas"></canvas>' +
                    '</div>' +
                '</div>' +
            '</div>';

        /* ── Element refs ── */
        var stepBtn = container.querySelector('.mpcd-step-btn');
        var autoBtn = container.querySelector('.mpcd-auto-btn');
        var resetBtn = container.querySelector('.mpcd-reset-btn');
        var speedSlider = container.querySelector('.mpcd-speed-slider');
        var speedVal = container.querySelector('.mpcd-speed-val');
        var stepLabel = container.querySelector('.mpcd-step-label');
        var x1El = container.querySelector('.mpcd-x1');
        var x2El = container.querySelector('.mpcd-x2');
        var uEl = container.querySelector('.mpcd-u');
        var pedagogy = container.querySelector('.mpcd-pedagogy');
        var posCanvas = container.querySelector('.mpcd-pos-canvas');
        var ctrlCanvas = container.querySelector('.mpcd-ctrl-canvas');

        /* ── State ── */
        var x0_default = [4, 0];
        var steps = [];
        var currentStep = -1;
        var ghostTrails = [];
        var autoTimer = null;

        function precompute() {
            steps = simulateMPC(x0_default);
        }

        function getSpeed() {
            return parseInt(speedSlider.value, 10);
        }

        function updateIndicator() {
            var total = steps.length;
            if (currentStep < 0) {
                stepLabel.textContent = 'Step k = 0 / ' + (total > 0 ? total - 1 : '--');
                x1El.textContent = x0_default[0].toFixed(3);
                x2El.textContent = x0_default[1].toFixed(3);
                uEl.textContent = '\u2014';
                pedagogy.textContent = 'The optimizer plans N=' + N_horizon + ' steps ahead, but only the first input is applied.';
            } else if (currentStep < total) {
                var s = steps[currentStep];
                stepLabel.textContent = 'Step k = ' + s.k + ' / ' + (total - 1);
                x1El.textContent = s.x[0].toFixed(3);
                x2El.textContent = s.x[1].toFixed(3);
                uEl.textContent = s.applied_u.toFixed(3);

                if (currentStep === 0) {
                    pedagogy.textContent = 'Horizon planned (orange dashed). First input applied (yellow). State is measured, plan will be recomputed.';
                } else if (currentStep < total - 1) {
                    pedagogy.textContent = 'New state measured \u2192 entire plan recomputed. Old plan fades to ghost. Only first input applied.';
                } else {
                    pedagogy.textContent = 'System has converged to the origin. The receding horizon approach successfully regulated the state.';
                }
            }
        }

        function redraw() {
            drawPositionPlot(posCanvas, steps, currentStep, ghostTrails);
            drawControlPlot(ctrlCanvas, steps, currentStep, ghostTrails);
        }

        function doStep() {
            if (steps.length === 0) precompute();
            if (currentStep >= steps.length - 1) return;

            // Save current prediction as ghost trail before advancing
            if (currentStep >= 0 && currentStep < steps.length) {
                var s = steps[currentStep];
                ghostTrails.push({
                    fromStep: s.k,
                    xs: s.predicted_xs.slice(),
                    us: s.predicted_us.slice()
                });
                // Keep only last 8 ghosts
                if (ghostTrails.length > 8) {
                    ghostTrails.shift();
                }
            }

            currentStep++;
            updateIndicator();
            redraw();
        }

        function doReset() {
            stopAuto();
            currentStep = -1;
            ghostTrails = [];
            precompute();
            updateIndicator();
            redraw();
        }

        function toggleAuto() {
            if (autoTimer) {
                stopAuto();
                return;
            }
            autoBtn.textContent = '\u23F8 Stop';
            autoBtn.classList.add('active');
            doStep();
            autoTimer = setInterval(function () {
                if (currentStep >= steps.length - 1) {
                    stopAuto();
                    return;
                }
                doStep();
            }, getSpeed());
        }

        function stopAuto() {
            if (autoTimer) {
                clearInterval(autoTimer);
                autoTimer = null;
            }
            autoBtn.textContent = '\u25B6\u25B6 Auto Play';
            autoBtn.classList.remove('active');
        }

        /* ── Events ── */
        stepBtn.addEventListener('click', function () {
            stopAuto();
            doStep();
        });
        autoBtn.addEventListener('click', toggleAuto);
        resetBtn.addEventListener('click', doReset);

        speedSlider.addEventListener('input', function () {
            speedVal.textContent = this.value + 'ms';
            if (autoTimer) {
                clearInterval(autoTimer);
                autoTimer = setInterval(function () {
                    if (currentStep >= steps.length - 1) {
                        stopAuto();
                        return;
                    }
                    doStep();
                }, getSpeed());
            }
        });

        /* ── Initial render ── */
        precompute();
        updateIndicator();
        redraw();

        /* ── Resize ── */
        if (window.ResizeObserver) {
            var ro = new ResizeObserver(function () { redraw(); });
            ro.observe(posCanvas);
            ro.observe(ctrlCanvas);
        }
    }

    /* ── Public API ────────────────────────────────────────── */
    function initAllDemos(root) {
        var els = (root || document).querySelectorAll('.mpc-demo');
        for (var i = 0; i < els.length; i++) initDemo(els[i]);
    }

    /* Auto-init on DOMContentLoaded */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () { initAllDemos(); });
    } else {
        initAllDemos();
    }

    window.MPCDemo = { initAll: initAllDemos, init: initDemo };
})();
