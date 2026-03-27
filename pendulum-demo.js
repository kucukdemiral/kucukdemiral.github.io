/**
 * Interactive Pendulum Swing-Up Demo — Nonlinear Model Predictive Control (NMPC)
 * Pure JS/Canvas GUI — no external dependencies.
 *
 * HTML hook:  <div class="pendulum-demo"></div>
 * Init call:  PendulumDemo.initAll(root)
 */
(function () {
    'use strict';

    /* ── Physics constants ───────────────────────────────── */
    var g = 9.81, L = 1.0, m = 1.0;
    var Ts = 0.05;
    var u_max = 5.0;
    var T_final = 10.0;

    /* ── NMPC parameters ─────────────────────────────────── */
    var N_mpc = 50;           /* prediction horizon (2.5 s) */
    var maxILQR = 40;         /* max iLQR iterations */
    var q_theta = 5.0;        /* stage cost: angle */
    var q_omega = 0.1;        /* stage cost: angular velocity */
    var r_u = 0.01;           /* stage cost: control effort */
    var q_theta_f = 200.0;    /* terminal cost: angle */
    var q_omega_f = 50.0;     /* terminal cost: angular velocity */
    var conv_tol = 1e-4;      /* convergence tolerance */

    /* ── Angle normalisation to [-pi, pi] ────────────────── */
    function wrapAngle(a) {
        while (a > Math.PI) a -= 2 * Math.PI;
        while (a < -Math.PI) a += 2 * Math.PI;
        return a;
    }

    /* ── Energy of pendulum ──────────────────────────────── */
    function energy(theta, omega) {
        return 0.5 * m * L * L * omega * omega - m * g * L * Math.cos(theta);
    }

    /* ── Forward Euler step (array form) ─────────────────── */
    function dynamics(x, u) {
        var theta_next = x[0] + Ts * x[1];
        var omega_next = x[1] + Ts * ((g / L) * Math.sin(x[0]) + u);
        return [theta_next, omega_next];
    }

    /* ── Saturate control ────────────────────────────────── */
    function saturate(u, lim) {
        if (u > lim) return lim;
        if (u < -lim) return -lim;
        return u;
    }

    /* ── Dynamics Jacobians ──────────────────────────────── */
    function dynJacobians(x) {
        var ct = Math.cos(x[0]);
        var A = [
            [1.0, Ts],
            [Ts * (g / L) * ct, 1.0]
        ];
        var B = [0.0, Ts];
        return { A: A, B: B };
    }

    /* ── Stage cost ──────────────────────────────────────── */
    function stageCost(x, u) {
        return q_theta * (1.0 - Math.cos(x[0])) + q_omega * x[1] * x[1] + r_u * u * u;
    }

    /* ── Terminal cost ───────────────────────────────────── */
    function termCost(x) {
        return q_theta_f * (1.0 - Math.cos(x[0])) + q_omega_f * x[1] * x[1];
    }

    /* ── Stage cost derivatives ──────────────────────────── */
    function stageCostDeriv(x, u) {
        return {
            lx: [q_theta * Math.sin(x[0]), 2.0 * q_omega * x[1]],
            lu: 2.0 * r_u * u,
            lxx: [
                [q_theta * Math.cos(x[0]), 0.0],
                [0.0, 2.0 * q_omega]
            ],
            luu: 2.0 * r_u,
            lux: [0.0, 0.0]
        };
    }

    /* ── Terminal cost derivatives ────────────────────────── */
    function termCostDeriv(x) {
        return {
            lx: [q_theta_f * Math.sin(x[0]), 2.0 * q_omega_f * x[1]],
            lxx: [
                [q_theta_f * Math.cos(x[0]), 0.0],
                [0.0, 2.0 * q_omega_f]
            ]
        };
    }

    /* ── Total trajectory cost ───────────────────────────── */
    function totalCost(xs, us, N) {
        var c = 0.0;
        for (var k = 0; k < N; k++) {
            c += stageCost(xs[k], us[k]);
        }
        c += termCost(xs[N]);
        return c;
    }

    /* ── Forward rollout ─────────────────────────────────── */
    function rollout(x0, us, N) {
        var xs = [x0];
        for (var k = 0; k < N; k++) {
            xs.push(dynamics(xs[k], us[k]));
        }
        return xs;
    }

    /* ── iLQR solver ─────────────────────────────────────── */
    function solveILQR(x0, uWarm, N) {
        var us = [];
        for (var i = 0; i < N; i++) {
            us.push(uWarm[i]);
        }

        /* Forward rollout with current controls */
        var xs = rollout(x0, us, N);
        var cost = totalCost(xs, us, N);

        var mu = 1e-6;
        var muMin = 1e-8;
        var muMax = 1e6;

        /* Storage for feedback gains and feedforward */
        var Ks = [];  /* each entry is [k0, k1] (1x2) */
        var ds = [];  /* each entry is scalar */

        for (var iter = 0; iter < maxILQR; iter++) {
            /* ── Backward pass ── */
            var backwardOk = false;
            while (!backwardOk) {
                backwardOk = true;
                Ks = [];
                ds = [];

                /* Initialise value function from terminal cost */
                var tcd = termCostDeriv(xs[N]);
                var Vx = [tcd.lx[0], tcd.lx[1]];
                var Vxx = [
                    [tcd.lxx[0][0], tcd.lxx[0][1]],
                    [tcd.lxx[1][0], tcd.lxx[1][1]]
                ];

                for (var k = N - 1; k >= 0; k--) {
                    var jac = dynJacobians(xs[k]);
                    var A = jac.A;
                    var B = jac.B;
                    var cd = stageCostDeriv(xs[k], us[k]);

                    /* Qx = lx + A'*Vx */
                    var Qx = [
                        cd.lx[0] + A[0][0] * Vx[0] + A[1][0] * Vx[1],
                        cd.lx[1] + A[0][1] * Vx[0] + A[1][1] * Vx[1]
                    ];

                    /* Qu = lu + B'*Vx */
                    var Qu = cd.lu + B[0] * Vx[0] + B[1] * Vx[1];

                    /* Qxx = lxx + A'*Vxx*A */
                    /* First compute Vxx*A */
                    var VA = [
                        [Vxx[0][0] * A[0][0] + Vxx[0][1] * A[1][0], Vxx[0][0] * A[0][1] + Vxx[0][1] * A[1][1]],
                        [Vxx[1][0] * A[0][0] + Vxx[1][1] * A[1][0], Vxx[1][0] * A[0][1] + Vxx[1][1] * A[1][1]]
                    ];
                    /* Then A'*VA */
                    var Qxx = [
                        [cd.lxx[0][0] + A[0][0] * VA[0][0] + A[1][0] * VA[1][0],
                         cd.lxx[0][1] + A[0][0] * VA[0][1] + A[1][0] * VA[1][1]],
                        [cd.lxx[1][0] + A[0][1] * VA[0][0] + A[1][1] * VA[1][0],
                         cd.lxx[1][1] + A[0][1] * VA[0][1] + A[1][1] * VA[1][1]]
                    ];

                    /* Quu = luu + B'*Vxx*B + mu */
                    /* B'*Vxx*B = B[0]*(Vxx[0][0]*B[0]+Vxx[0][1]*B[1]) + B[1]*(Vxx[1][0]*B[0]+Vxx[1][1]*B[1]) */
                    var Quu = cd.luu
                        + B[0] * (Vxx[0][0] * B[0] + Vxx[0][1] * B[1])
                        + B[1] * (Vxx[1][0] * B[0] + Vxx[1][1] * B[1])
                        + mu;

                    /* Qux = lux + B'*Vxx*A  (1x2 vector) */
                    /* B'*Vxx*A row j: B[0]*VA[0][j] + B[1]*VA[1][j] */
                    var Qux = [
                        cd.lux[0] + B[0] * VA[0][0] + B[1] * VA[1][0],
                        cd.lux[1] + B[0] * VA[0][1] + B[1] * VA[1][1]
                    ];

                    if (Quu <= 0) {
                        /* Regularisation needed */
                        mu = Math.max(mu * 10.0, 1e-6);
                        if (mu > muMax) {
                            /* Give up on this iteration */
                            backwardOk = true;
                            break;
                        }
                        backwardOk = false;
                        break;
                    }

                    /* Gains */
                    var K_k = [-Qux[0] / Quu, -Qux[1] / Quu];
                    var d_k = -Qu / Quu;

                    /* Store (we build them backwards, so unshift) */
                    Ks.unshift(K_k);
                    ds.unshift(d_k);

                    /* Update value function */
                    /* Vx = Qx - Qux * Qu / Quu  =>  Vx[i] = Qx[i] - Qux[i]*Qu/Quu */
                    Vx = [
                        Qx[0] - Qux[0] * Qu / Quu,
                        Qx[1] - Qux[1] * Qu / Quu
                    ];

                    /* Vxx = Qxx - Qux' * Qux / Quu  =>  Vxx[i][j] = Qxx[i][j] - Qux[i]*Qux[j]/Quu */
                    Vxx = [
                        [Qxx[0][0] - Qux[0] * Qux[0] / Quu, Qxx[0][1] - Qux[0] * Qux[1] / Quu],
                        [Qxx[1][0] - Qux[1] * Qux[0] / Quu, Qxx[1][1] - Qux[1] * Qux[1] / Quu]
                    ];
                }
            }

            /* If backward pass failed completely, return current solution */
            if (Ks.length < N) {
                break;
            }

            /* ── Forward pass with line search ── */
            var improved = false;
            var alpha = 1.0;
            for (var ls = 0; ls < 10; ls++) {
                var usNew = [];
                var xsNew = [x0];
                for (var k = 0; k < N; k++) {
                    var dx = [xsNew[k][0] - xs[k][0], xsNew[k][1] - xs[k][1]];
                    var uNew = us[k] + alpha * ds[k] + Ks[k][0] * dx[0] + Ks[k][1] * dx[1];
                    uNew = saturate(uNew, u_max);
                    usNew.push(uNew);
                    xsNew.push(dynamics(xsNew[k], uNew));
                }
                var newCost = totalCost(xsNew, usNew, N);
                if (newCost < cost) {
                    xs = xsNew;
                    us = usNew;
                    cost = newCost;
                    improved = true;
                    break;
                }
                alpha *= 0.5;
            }

            if (improved) {
                mu = Math.max(mu * 0.5, muMin);
            } else {
                mu = Math.min(mu * 10.0, muMax);
            }

            /* Check convergence */
            if (ds.length === N) {
                var maxD = 0.0;
                for (var k = 0; k < N; k++) {
                    var absD = Math.abs(ds[k]);
                    if (absD > maxD) maxD = absD;
                }
                if (maxD < conv_tol) {
                    break;
                }
            }
        }

        return { us: us, xs: xs, cost: cost };
    }

    /* ── Initial warm-start: bang-bang seed to break symmetry ── */
    function seedWarmStart() {
        var us = [];
        /* Apply a positive torque burst, then negative, to pump energy */
        for (var i = 0; i < N_mpc; i++) {
            if (i < 15) us.push(u_max);
            else if (i < 30) us.push(-u_max);
            else us.push(0);
        }
        return us;
    }

    /* ── Pre-compute the swing-up trajectory via NMPC ────── */
    function computeTrajectory() {
        var Nsim = Math.floor(T_final / Ts);
        var theta = Math.PI + 0.01, omega = 0.0;  /* tiny perturbation to break symmetry */
        var uWarm = seedWarmStart();

        var ts = [], thetas = [], omegas = [], us = [], Es = [];

        for (var k = 0; k < Nsim; k++) {
            var x = [theta, omega];

            /* Solve NMPC via iLQR */
            var result = solveILQR(x, uWarm, N_mpc);
            var u = saturate(result.us[0], u_max);

            /* Warm start: shift solution */
            uWarm = result.us.slice(1);
            uWarm.push(0);

            /* Log (with wrapped angle for display) */
            ts.push(k * Ts);
            thetas.push(wrapAngle(theta));
            omegas.push(omega);
            us.push(u);
            Es.push(energy(wrapAngle(theta), omega));

            /* Apply dynamics */
            var next = dynamics(x, u);
            theta = next[0];
            omega = next[1];
        }

        return {
            ts: ts, thetas: thetas, omegas: omegas, us: us, Es: Es, N: Nsim
        };
    }

    /* ── Canvas setup helper ─────────────────────────────── */
    function setupCanvas(canvas) {
        var dpr = window.devicePixelRatio || 1;
        var w = canvas.clientWidth, h = canvas.clientHeight;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        var ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        return { ctx: ctx, w: w, h: h };
    }

    /* ── Color interpolation helper ──────────────────────── */
    function lerpColor(c1, c2, t) {
        if (t < 0) t = 0; if (t > 1) t = 1;
        return [
            Math.round(c1[0] + (c2[0] - c1[0]) * t),
            Math.round(c1[1] + (c2[1] - c1[1]) * t),
            Math.round(c1[2] + (c2[2] - c1[2]) * t)
        ];
    }

    function rgbStr(c) {
        return 'rgb(' + c[0] + ',' + c[1] + ',' + c[2] + ')';
    }

    /* ── Bob color based on energy / stabilisation ───────── */
    function bobColor(E, theta) {
        var E_up = m * g * L;
        var th_w = Math.abs(wrapAngle(theta));

        /* Stabilised at top */
        if (th_w < 0.15 && Math.abs(E - E_up) < 0.5) {
            return [46, 204, 113]; /* green */
        }

        /* Ratio of current energy to target */
        var ratio = E / (2 * E_up);
        if (ratio < 0) ratio = 0;
        if (ratio > 1) ratio = 1;

        var cold = [52, 152, 219];   /* blue */
        var warm = [231, 76, 60];    /* red */
        return lerpColor(cold, warm, ratio);
    }

    /* ── Draw pendulum scene ─────────────────────────────── */
    function drawPendulum(canvas, traj, idx, trail) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0F1923';
        ctx.fillRect(0, 0, W, H);

        var theta = traj.thetas[idx];
        var E = traj.Es[idx];

        /* Pivot position */
        var pivotX = W / 2;
        var pivotY = H * 0.42;
        var rodLen = Math.min(W, H) * 0.32;

        /* Bob position (theta=0 is upright, positive CW from top) */
        /* In our convention: upright means bob is above pivot visually,
           but we draw it more naturally with the "upright" being straight up.
           theta=0 => bob directly above pivot (inverted pendulum upright)
           theta=pi => bob directly below pivot (hanging) */
        var bobX = pivotX + rodLen * Math.sin(theta);
        var bobY = pivotY - rodLen * Math.cos(theta);

        /* Draw trail */
        if (trail.length > 1) {
            ctx.strokeStyle = 'rgba(100, 180, 255, 0.08)';
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(trail[0][0], trail[0][1]);
            for (var i = 1; i < trail.length; i++) {
                ctx.lineTo(trail[i][0], trail[i][1]);
            }
            ctx.stroke();

            /* Fading trail dots */
            for (var i = 0; i < trail.length; i++) {
                var alpha = 0.03 + 0.15 * (i / trail.length);
                ctx.fillStyle = 'rgba(100, 180, 255,' + alpha.toFixed(3) + ')';
                ctx.beginPath();
                ctx.arc(trail[i][0], trail[i][1], 1.5, 0, 2 * Math.PI);
                ctx.fill();
            }
        }

        /* Draw support structure */
        ctx.strokeStyle = 'rgba(255,255,255,0.25)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(pivotX - 30, pivotY);
        ctx.lineTo(pivotX + 30, pivotY);
        ctx.stroke();

        /* Hatching above support */
        ctx.strokeStyle = 'rgba(255,255,255,0.12)';
        ctx.lineWidth = 1;
        for (var hx = pivotX - 28; hx <= pivotX + 28; hx += 6) {
            ctx.beginPath();
            ctx.moveTo(hx, pivotY);
            ctx.lineTo(hx - 4, pivotY - 8);
            ctx.stroke();
        }

        /* Draw rod */
        ctx.strokeStyle = 'rgba(200,210,220,0.9)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(pivotX, pivotY);
        ctx.lineTo(bobX, bobY);
        ctx.stroke();

        /* Draw pivot */
        ctx.fillStyle = '#556677';
        ctx.beginPath();
        ctx.arc(pivotX, pivotY, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.3)';
        ctx.lineWidth = 1;
        ctx.stroke();

        /* Draw bob */
        var bColor = bobColor(E, theta);
        var bobR = 12;
        var grad = ctx.createRadialGradient(bobX - 2, bobY - 2, 2, bobX, bobY, bobR);
        grad.addColorStop(0, 'rgba(' + Math.min(bColor[0]+60,255) + ',' + Math.min(bColor[1]+60,255) + ',' + Math.min(bColor[2]+60,255) + ',1)');
        grad.addColorStop(0.7, rgbStr(bColor));
        grad.addColorStop(1, 'rgba(' + Math.max(bColor[0]-40,0) + ',' + Math.max(bColor[1]-40,0) + ',' + Math.max(bColor[2]-40,0) + ',1)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.arc(bobX, bobY, bobR, 0, 2 * Math.PI);
        ctx.fill();
        ctx.strokeStyle = 'rgba(255,255,255,0.2)';
        ctx.lineWidth = 1;
        ctx.stroke();

        /* Draw torque arrow */
        var u = traj.us[idx];
        if (Math.abs(u) > 0.1) {
            var arrowLen = Math.abs(u) / u_max * 25;
            var arrowDir = u > 0 ? 1 : -1;
            ctx.strokeStyle = '#f39c12';
            ctx.lineWidth = 2;
            ctx.beginPath();
            var aRadius = 18;
            var startAngle = -Math.PI / 2 + theta;
            var endAngle = startAngle + arrowDir * (arrowLen / aRadius);
            ctx.arc(pivotX, pivotY, aRadius, startAngle, endAngle, arrowDir < 0);
            ctx.stroke();
            /* arrowhead */
            var ax = pivotX + aRadius * Math.cos(endAngle);
            var ay = pivotY + aRadius * Math.sin(endAngle);
            ctx.fillStyle = '#f39c12';
            ctx.beginPath();
            ctx.arc(ax, ay, 3, 0, 2 * Math.PI);
            ctx.fill();
        }

        /* Labels */
        ctx.fillStyle = '#99AABB';
        ctx.font = '10px "Source Sans 3", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('\u03B8 = ' + theta.toFixed(2) + ' rad', 10, H - 28);
        ctx.fillText('E = ' + E.toFixed(2) + ' J', 10, H - 14);

        /* Reference: upright ghost */
        ctx.setLineDash([3, 4]);
        ctx.strokeStyle = 'rgba(46,204,113,0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pivotX, pivotY);
        ctx.lineTo(pivotX, pivotY - rodLen);
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = 'rgba(46,204,113,0.35)';
        ctx.beginPath();
        ctx.arc(pivotX, pivotY - rodLen, 4, 0, 2 * Math.PI);
        ctx.fill();
        ctx.fillStyle = 'rgba(46,204,113,0.5)';
        ctx.font = '9px "Source Sans 3", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('target', pivotX, pivotY - rodLen - 8);

        return [bobX, bobY];
    }

    /* ── Generic mini-plot ───────────────────────────────── */
    function drawMiniPlot(ctx, ox, oy, pw, ph, xs, ys, idx, label, color, yRef) {
        var mt = 14, mb = 4, ml = 38, mr = 6;
        var plotW = pw - ml - mr;
        var plotH = ph - mt - mb;

        /* y range */
        var yMin = Infinity, yMax = -Infinity;
        for (var i = 0; i < ys.length; i++) {
            if (ys[i] < yMin) yMin = ys[i];
            if (ys[i] > yMax) yMax = ys[i];
        }
        if (yRef !== undefined && yRef !== null) {
            if (yRef < yMin) yMin = yRef;
            if (yRef > yMax) yMax = yRef;
        }
        var pad = Math.max((yMax - yMin) * 0.12, 0.5);
        yMin -= pad; yMax += pad;

        var xMin = 0, xMax = xs[xs.length - 1] || T_final;

        function sx(v) { return ox + ml + (v - xMin) / (xMax - xMin) * plotW; }
        function sy(v) { return oy + mt + (1 - (v - yMin) / (yMax - yMin)) * plotH; }

        /* background */
        ctx.fillStyle = '#0A1219';
        ctx.fillRect(ox, oy, pw, ph);

        /* grid */
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth = 1;
        var nyt = 3;
        var ystep = (yMax - yMin) / nyt;
        ctx.fillStyle = '#445566';
        ctx.font = '8px "SF Mono","Fira Code",monospace';
        ctx.textAlign = 'right';
        for (var i = 0; i <= nyt; i++) {
            var v = yMin + i * ystep;
            var py = sy(v);
            ctx.beginPath();
            ctx.moveTo(ox + ml, py);
            ctx.lineTo(ox + ml + plotW, py);
            ctx.stroke();
            ctx.fillText(v.toFixed(1), ox + ml - 4, py + 3);
        }

        /* axes */
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.beginPath();
        ctx.moveTo(ox + ml, oy + mt);
        ctx.lineTo(ox + ml, oy + mt + plotH);
        ctx.lineTo(ox + ml + plotW, oy + mt + plotH);
        ctx.stroke();

        /* reference line */
        if (yRef !== undefined && yRef !== null) {
            ctx.setLineDash([4, 3]);
            ctx.strokeStyle = 'rgba(46,204,113,0.4)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(ox + ml, sy(yRef));
            ctx.lineTo(ox + ml + plotW, sy(yRef));
            ctx.stroke();
            ctx.setLineDash([]);
        }

        /* data line */
        var n = Math.min(idx + 1, ys.length);
        if (n > 1) {
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (var i = 0; i < n; i++) {
                var px = sx(xs[i]), py = sy(ys[i]);
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }
            ctx.stroke();
        }

        /* current point */
        if (idx >= 0 && idx < ys.length) {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(sx(xs[idx]), sy(ys[idx]), 3, 0, 2 * Math.PI);
            ctx.fill();
        }

        /* label */
        ctx.fillStyle = '#99AABB';
        ctx.font = '10px "Source Sans 3",sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(label, ox + ml + 4, oy + mt - 3);
    }

    /* ── Draw the three stacked plots ────────────────────── */
    function drawPlots(canvas, traj, idx) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0F1923';
        ctx.fillRect(0, 0, W, H);

        var gap = 4;
        var pH = (H - 2 * gap) / 3;

        /* theta(t) */
        drawMiniPlot(ctx, 0, 0, W, pH, traj.ts, traj.thetas, idx,
            '\u03B8(t) [rad]', '#e74c3c', 0);

        /* omega(t) */
        drawMiniPlot(ctx, 0, pH + gap, W, pH, traj.ts, traj.omegas, idx,
            '\u03C9(t) [rad/s]', '#3498db', 0);

        /* u(t) */
        drawMiniPlot(ctx, 0, 2 * (pH + gap), W, pH, traj.ts, traj.us, idx,
            'u(t) [N\u00B7m]', '#f39c12', 0);

        /* time axis label */
        ctx.fillStyle = '#667788';
        ctx.font = '9px "Source Sans 3",sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Time (s)', W / 2, H - 2);
    }

    /* ── Build widget ────────────────────────────────────── */
    function initDemo(container) {
        if (container.dataset.initialized) return;
        container.dataset.initialized = '1';

        /* ── Inject scoped CSS ── */
        var styleId = 'pendulum-demo-styles';
        if (!document.getElementById(styleId)) {
            var style = document.createElement('style');
            style.id = styleId;
            style.textContent =
                '.pendulum-demo{' +
                    'background:#0F1923;' +
                    'border-radius:8px;' +
                    'overflow:hidden;' +
                    'font-family:"Source Sans 3","Segoe UI",-apple-system,BlinkMacSystemFont,sans-serif;' +
                    'color:#C8D6E5;' +
                    'margin:1.5rem 0;' +
                    'border:1px solid rgba(255,255,255,0.08);' +
                '}' +
                '.pnd-header{' +
                    'padding:12px 18px;' +
                    'background:rgba(255,255,255,0.03);' +
                    'border-bottom:1px solid rgba(255,255,255,0.06);' +
                '}' +
                '.pnd-title{' +
                    'font-family:"Crimson Text",Georgia,serif;' +
                    'font-size:1.1rem;' +
                    'font-weight:600;' +
                    'color:#D4E0ED;' +
                '}' +
                '.pnd-system-info{' +
                    'padding:8px 18px;' +
                    'font-size:0.78rem;' +
                    'color:#778899;' +
                    'background:rgba(0,0,0,0.15);' +
                    'border-bottom:1px solid rgba(255,255,255,0.04);' +
                '}' +
                '.pnd-controls{' +
                    'padding:10px 18px;' +
                    'display:flex;' +
                    'align-items:center;' +
                    'flex-wrap:wrap;' +
                    'gap:12px;' +
                    'background:rgba(255,255,255,0.02);' +
                    'border-bottom:1px solid rgba(255,255,255,0.06);' +
                '}' +
                '.pnd-btn-group{' +
                    'display:flex;' +
                    'gap:6px;' +
                    'align-items:center;' +
                '}' +
                '.pnd-btn{' +
                    'padding:5px 14px;' +
                    'font-family:"Source Sans 3",sans-serif;' +
                    'font-size:0.8rem;' +
                    'font-weight:500;' +
                    'color:#C8D6E5;' +
                    'background:rgba(255,255,255,0.06);' +
                    'border:1px solid rgba(255,255,255,0.1);' +
                    'border-radius:4px;' +
                    'cursor:pointer;' +
                    'transition:all 0.15s ease;' +
                '}' +
                '.pnd-btn:hover{' +
                    'background:rgba(255,255,255,0.12);' +
                    'color:#fff;' +
                '}' +
                '.pnd-btn.pnd-primary{' +
                    'background:rgba(46,80,144,0.5);' +
                    'border-color:rgba(46,80,144,0.7);' +
                '}' +
                '.pnd-btn.pnd-primary:hover{' +
                    'background:rgba(46,80,144,0.7);' +
                '}' +
                '.pnd-btn.pnd-active{' +
                    'background:rgba(231,76,60,0.3);' +
                    'border-color:rgba(231,76,60,0.5);' +
                '}' +
                '.pnd-btn.pnd-disturb{' +
                    'background:rgba(243,156,18,0.25);' +
                    'border-color:rgba(243,156,18,0.45);' +
                '}' +
                '.pnd-btn.pnd-disturb:hover{' +
                    'background:rgba(243,156,18,0.45);' +
                '}' +
                '.pnd-slider-label{' +
                    'display:flex;' +
                    'align-items:center;' +
                    'gap:8px;' +
                    'font-size:0.78rem;' +
                    'color:#99AABB;' +
                '}' +
                '.pnd-slider{' +
                    '-webkit-appearance:none;' +
                    'appearance:none;' +
                    'width:80px;' +
                    'height:4px;' +
                    'background:rgba(255,255,255,0.12);' +
                    'border-radius:2px;' +
                    'outline:none;' +
                '}' +
                '.pnd-slider::-webkit-slider-thumb{' +
                    '-webkit-appearance:none;' +
                    'appearance:none;' +
                    'width:14px;' +
                    'height:14px;' +
                    'border-radius:50%;' +
                    'background:#3498db;' +
                    'cursor:pointer;' +
                '}' +
                '.pnd-slider::-moz-range-thumb{' +
                    'width:14px;' +
                    'height:14px;' +
                    'border-radius:50%;' +
                    'background:#3498db;' +
                    'cursor:pointer;' +
                    'border:none;' +
                '}' +
                '.pnd-slider-val{' +
                    'font-family:"SF Mono","Fira Code",monospace;' +
                    'font-size:0.75rem;' +
                    'color:#778899;' +
                    'min-width:30px;' +
                '}' +
                '.pnd-body{' +
                    'display:flex;' +
                    'gap:0;' +
                '}' +
                '.pnd-scene-wrap{' +
                    'flex:1;' +
                    'min-width:0;' +
                '}' +
                '.pnd-scene-canvas{' +
                    'display:block;' +
                    'width:100%;' +
                    'height:320px;' +
                '}' +
                '.pnd-plot-wrap{' +
                    'flex:1;' +
                    'min-width:0;' +
                    'border-left:1px solid rgba(255,255,255,0.06);' +
                '}' +
                '.pnd-plot-canvas{' +
                    'display:block;' +
                    'width:100%;' +
                    'height:320px;' +
                '}' +
                '.pnd-readout{' +
                    'padding:8px 18px;' +
                    'display:flex;' +
                    'gap:18px;' +
                    'flex-wrap:wrap;' +
                    'font-size:0.78rem;' +
                    'color:#778899;' +
                    'background:rgba(0,0,0,0.15);' +
                    'border-top:1px solid rgba(255,255,255,0.04);' +
                '}' +
                '.pnd-ro-item{font-family:"SF Mono","Fira Code",monospace;font-size:0.72rem;}' +
                '.pnd-ro-item span{color:#C8D6E5;}' +
                '.pnd-status{' +
                    'margin-left:auto;' +
                    'font-family:"Source Sans 3",sans-serif;' +
                    'font-weight:500;' +
                '}' +
                '.pnd-status.pnd-swinging{color:#f39c12;}' +
                '.pnd-status.pnd-stabilised{color:#2ecc71;}' +
                '@media (max-width:640px){' +
                    '.pnd-body{flex-direction:column;}' +
                    '.pnd-plot-wrap{border-left:none;border-top:1px solid rgba(255,255,255,0.06);}' +
                    '.pnd-controls{flex-direction:column;align-items:flex-start;}' +
                '}';
            document.head.appendChild(style);
        }

        /* ── Build DOM ── */
        container.innerHTML =
            '<div class="pnd-header">' +
                '<span class="pnd-title">\u25B6 Pendulum Swing-Up \u2014 Nonlinear MPC (iLQR Solver)</span>' +
            '</div>' +
            '<div class="pnd-system-info">' +
                'Dynamics: &nbsp;\u03B8\u0307 = \u03C9, &nbsp;\u03C9\u0307 = (g/L)sin\u03B8 + u' +
                '&nbsp;&nbsp;|&nbsp;&nbsp;g = 9.81, L = 1.0, |u| \u2264 5' +
                '&nbsp;&nbsp;|&nbsp;&nbsp;Forward Euler, T<sub>s</sub> = 0.05 s' +
            '</div>' +
            '<div class="pnd-controls">' +
                '<div class="pnd-btn-group">' +
                    '<button class="pnd-btn pnd-primary pnd-swing-btn">\u25B6 Swing Up</button>' +
                    '<button class="pnd-btn pnd-reset-btn">\u21BA Reset</button>' +
                    '<button class="pnd-btn pnd-disturb pnd-disturb-btn">\u26A1 Disturb</button>' +
                '</div>' +
                '<label class="pnd-slider-label">Speed' +
                    '<input type="range" class="pnd-slider pnd-speed-slider" min="0.1" max="3.0" step="0.1" value="1.0">' +
                    '<span class="pnd-slider-val pnd-speed-val">1.0\u00D7</span>' +
                '</label>' +
            '</div>' +
            '<div class="pnd-body">' +
                '<div class="pnd-scene-wrap"><canvas class="pnd-scene-canvas"></canvas></div>' +
                '<div class="pnd-plot-wrap"><canvas class="pnd-plot-canvas"></canvas></div>' +
            '</div>' +
            '<div class="pnd-readout">' +
                '<span class="pnd-ro-item">t = <span class="pnd-ro-t">0.000</span> s</span>' +
                '<span class="pnd-ro-item">\u03B8 = <span class="pnd-ro-th">3.142</span> rad</span>' +
                '<span class="pnd-ro-item">\u03C9 = <span class="pnd-ro-om">0.000</span> rad/s</span>' +
                '<span class="pnd-ro-item">u = <span class="pnd-ro-u">0.000</span> N\u00B7m</span>' +
                '<span class="pnd-ro-item">E = <span class="pnd-ro-e">0.000</span> J</span>' +
                '<span class="pnd-status pnd-status-txt">Ready</span>' +
            '</div>';

        /* ── Element refs ── */
        var sceneCvs = container.querySelector('.pnd-scene-canvas');
        var plotCvs = container.querySelector('.pnd-plot-canvas');
        var swingBtn = container.querySelector('.pnd-swing-btn');
        var resetBtn = container.querySelector('.pnd-reset-btn');
        var disturbBtn = container.querySelector('.pnd-disturb-btn');
        var speedSlider = container.querySelector('.pnd-speed-slider');
        var speedVal = container.querySelector('.pnd-speed-val');
        var roT = container.querySelector('.pnd-ro-t');
        var roTh = container.querySelector('.pnd-ro-th');
        var roOm = container.querySelector('.pnd-ro-om');
        var roU = container.querySelector('.pnd-ro-u');
        var roE = container.querySelector('.pnd-ro-e');
        var statusEl = container.querySelector('.pnd-status-txt');

        /* ── State ── */
        var traj = null;
        var animId = null;
        var running = false;
        var startTime = 0;
        var pausedAt = 0;
        var trail = [];

        /* Live simulation state (for disturbance support) */
        var liveMode = false;
        var liveTheta = Math.PI;
        var liveOmega = 0.0;
        var liveTs = [];
        var liveThetas = [];
        var liveOmegas = [];
        var liveUs = [];
        var liveEs = [];
        var liveStep = 0;
        var liveTrail = [];

        /* Warm start controls for live NMPC */
        var liveUWarm = [];
        for (var wi = 0; wi < N_mpc; wi++) liveUWarm.push(0);

        function getSpeed() { return parseFloat(speedSlider.value); }

        function computeLiveStep() {
            var x = [liveTheta, liveOmega];

            /* Solve NMPC via iLQR */
            var result = solveILQR(x, liveUWarm, N_mpc);
            var u = saturate(result.us[0], u_max);

            /* Warm start: shift solution */
            liveUWarm = result.us.slice(1);
            liveUWarm.push(0);

            var th_wrapped = wrapAngle(liveTheta);
            var E = energy(th_wrapped, liveOmega);

            var t = liveStep * Ts;
            liveTs.push(t);
            liveThetas.push(th_wrapped);
            liveOmegas.push(liveOmega);
            liveUs.push(u);
            liveEs.push(E);

            var next = dynamics(x, u);
            liveTheta = next[0];
            liveOmega = next[1];
            liveStep++;
        }

        function getLiveTraj() {
            return {
                ts: liveTs,
                thetas: liveThetas,
                omegas: liveOmegas,
                us: liveUs,
                Es: liveEs,
                N: liveTs.length
            };
        }

        function drawFrame(idx) {
            var t, trj;
            if (liveMode) {
                /* Ensure we have enough steps computed */
                while (liveStep <= idx) {
                    computeLiveStep();
                }
                trj = getLiveTraj();
                if (idx >= trj.N) idx = trj.N - 1;
            } else {
                trj = traj;
                if (!trj) return;
                if (idx >= trj.N) idx = trj.N - 1;
            }

            /* Compute bob position for trail */
            var theta_cur = trj.thetas[idx];
            var W = sceneCvs.clientWidth, H = sceneCvs.clientHeight;
            var pivotX = W / 2, pivotY = H * 0.42;
            var rodLen = Math.min(W, H) * 0.32;
            var bobX = pivotX + rodLen * Math.sin(theta_cur);
            var bobY = pivotY - rodLen * Math.cos(theta_cur);

            if (liveMode) {
                liveTrail.push([bobX, bobY]);
                if (liveTrail.length > 300) liveTrail.shift();
                drawPendulum(sceneCvs, trj, idx, liveTrail);
            } else {
                trail.push([bobX, bobY]);
                if (trail.length > 300) trail.shift();
                drawPendulum(sceneCvs, trj, idx, trail);
            }
            drawPlots(plotCvs, trj, idx);

            /* Update readout */
            roT.textContent = trj.ts[idx].toFixed(3);
            roTh.textContent = trj.thetas[idx].toFixed(3);
            roOm.textContent = trj.omegas[idx].toFixed(3);
            roU.textContent = trj.us[idx].toFixed(3);
            roE.textContent = trj.Es[idx].toFixed(3);

            /* Update status */
            var th_abs = Math.abs(trj.thetas[idx]);
            if (th_abs < 0.15 && Math.abs(trj.omegas[idx]) < 0.5) {
                statusEl.textContent = 'Stabilised';
                statusEl.className = 'pnd-status pnd-status-txt pnd-stabilised';
            } else {
                statusEl.textContent = 'Swinging';
                statusEl.className = 'pnd-status pnd-status-txt pnd-swinging';
            }
        }

        function animLoop(timestamp) {
            if (!running) return;
            var elapsed = (timestamp - startTime) / 1000 * getSpeed();
            var idx = Math.floor(elapsed / Ts);

            if (!liveMode && idx >= traj.N) {
                idx = traj.N - 1;
                drawFrame(idx);
                running = false;
                swingBtn.textContent = '\u25B6 Swing Up';
                swingBtn.classList.remove('pnd-active');
                return;
            }

            /* In live mode, run indefinitely (up to a practical limit) */
            if (liveMode && idx > 10000) {
                idx = 10000;
                running = false;
                swingBtn.textContent = '\u25B6 Swing Up';
                swingBtn.classList.remove('pnd-active');
                drawFrame(idx);
                return;
            }

            drawFrame(idx);
            animId = requestAnimationFrame(animLoop);
        }

        function doSwing() {
            if (running) {
                /* Pause */
                running = false;
                pausedAt = performance.now() - startTime;
                swingBtn.textContent = '\u25B6 Resume';
                swingBtn.classList.remove('pnd-active');
                if (animId) cancelAnimationFrame(animId);
                return;
            }

            if (!traj && !liveMode) {
                traj = computeTrajectory();
            }

            running = true;
            swingBtn.textContent = '\u23F8 Pause';
            swingBtn.classList.add('pnd-active');
            startTime = performance.now() - pausedAt;
            animId = requestAnimationFrame(animLoop);
        }

        function doReset() {
            running = false;
            liveMode = false;
            if (animId) cancelAnimationFrame(animId);
            swingBtn.textContent = '\u25B6 Swing Up';
            swingBtn.classList.remove('pnd-active');
            pausedAt = 0;
            trail = [];
            liveTrail = [];

            /* Reset live state */
            liveTheta = Math.PI;
            liveOmega = 0.0;
            liveTs = [];
            liveThetas = [];
            liveOmegas = [];
            liveUs = [];
            liveEs = [];
            liveStep = 0;
            liveUWarm = [];
            for (var wi = 0; wi < N_mpc; wi++) liveUWarm.push(0);

            traj = computeTrajectory();
            drawFrame(0);

            statusEl.textContent = 'Ready';
            statusEl.className = 'pnd-status pnd-status-txt';
        }

        function doDisturb() {
            if (!running) return;

            /* Switch to live mode if not already */
            if (!liveMode) {
                /* Figure out current playback index */
                var elapsed = (performance.now() - startTime) / 1000 * getSpeed();
                var idx = Math.floor(elapsed / Ts);
                if (idx >= traj.N) idx = traj.N - 1;

                /* Copy trajectory up to current point into live arrays */
                liveTs = traj.ts.slice(0, idx + 1);
                liveThetas = traj.thetas.slice(0, idx + 1);
                liveOmegas = traj.omegas.slice(0, idx + 1);
                liveUs = traj.us.slice(0, idx + 1);
                liveEs = traj.Es.slice(0, idx + 1);

                liveTheta = traj.thetas[idx];
                liveOmega = traj.omegas[idx];
                /* Unwrap for forward simulation */
                liveStep = idx + 1;
                liveTrail = trail.slice();
                liveMode = true;

                /* Reset warm start for NMPC from current state */
                liveUWarm = [];
                for (var wi = 0; wi < N_mpc; wi++) liveUWarm.push(0);
            }

            /* Apply a random impulse disturbance */
            var disturbMag = 3.0 + Math.random() * 4.0;
            var disturbSign = (Math.random() > 0.5) ? 1 : -1;
            liveOmega += disturbSign * disturbMag;

            /* Reset warm start after disturbance since old plan is invalid */
            liveUWarm = [];
            for (var wi = 0; wi < N_mpc; wi++) liveUWarm.push(0);
        }

        /* ── Events ── */
        swingBtn.addEventListener('click', doSwing);
        resetBtn.addEventListener('click', doReset);
        disturbBtn.addEventListener('click', doDisturb);

        speedSlider.addEventListener('input', function () {
            speedVal.textContent = parseFloat(this.value).toFixed(1) + '\u00D7';
            /* Adjust startTime to keep current position when speed changes */
            if (running) {
                var now = performance.now();
                var elapsed = (now - startTime) / 1000;
                /* Keep elapsed the same ratio */
                pausedAt = now - startTime;
                startTime = now - pausedAt;
            }
        });

        /* ── Initial render ── */
        traj = computeTrajectory();
        drawFrame(0);
        statusEl.textContent = 'Ready';
        statusEl.className = 'pnd-status pnd-status-txt';

        /* ── Resize handling ── */
        if (window.ResizeObserver) {
            var ro = new ResizeObserver(function () {
                trail = [];
                liveTrail = [];
                if (traj) drawFrame(0);
            });
            ro.observe(sceneCvs);
        }
    }

    /* ── Public API ──────────────────────────────────────── */
    function initAllDemos(root) {
        var els = (root || document).querySelectorAll('.pendulum-demo');
        for (var i = 0; i < els.length; i++) initDemo(els[i]);
    }

    /* Auto-initialize on DOMContentLoaded */
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () { initAllDemos(); });
    } else {
        initAllDemos();
    }

    window.PendulumDemo = { initAll: initAllDemos, init: initDemo };
})();
