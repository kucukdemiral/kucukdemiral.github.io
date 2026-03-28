/**
 * Autonomous Racing LMPC Demo — Learning Model Predictive Control
 * Dynamic bicycle model with Pacejka tires (based on alexliniger/MPCC).
 * Car learns to drive faster lap by lap using safe sets and cost-to-go.
 *
 * HTML hook:  <div class="racing-demo"></div>
 */
(function () {
    'use strict';

    var NX = 6, NU = 2;
    // State: [X, Y, phi, vx, vy, r]  (global pos, heading, body-frame velocities, yaw rate)
    // Input: [delta, D]  (steering angle, throttle/brake duty cycle)

    /* ═══ Vehicle Parameters (full-size lightweight racer) ═══════ */
    var VP = {
        m: 800, Iz: 1200,
        lf: 1.2, lr: 1.3,
        Bf: 10, Cf: 1.3, Df: 5000,
        Br: 11, Cr: 1.3, Dr: 7000,
        Cm1: 5000, Cm2: 100,
        Cr0: 50, Cr2: 0.3,
        carL: 4.0, carW: 1.8,
        deltaMax: 0.45, DMax: 1.0, DMin: -1.0,
        vxMin: 0.5, vxMax: 20
    };

    /* ═══ MPC / LMPC constants ══════════════════════════════════ */
    var DT_SIM = 0.05;
    var DT_MPC = 0.1;
    var MPC_EVERY = 2;
    var N_DEFAULT = 20;
    var SCP_ITERS = 3;
    var QP_ITERS = 12;
    var TR_WEIGHT = 2.0;
    var Rd = [0.01, 0.005];
    var TRACK_HW = 5.0;
    var W_TRACK = 500;
    var W_SS = 1.0;
    var KNN_K = 5;
    var TERM_HESS = [0.5, 0.5, 1.0, 0.3, 0.1, 0.1];
    var Q_LAT = 0.3;
    var Q_HEAD = 0.8;
    var TERM_SCALE = 0.05;
    var PP_SPEED = 5.0;

    function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }
    function wrapAngle(a) { a = a % (2 * Math.PI); if (a > Math.PI) a -= 2 * Math.PI; if (a < -Math.PI) a += 2 * Math.PI; return a; }
    function angleDiff(a, b) { return wrapAngle(a - b); }

    /* ═══ Catmull-Rom spline (scalar) ═══════════════════════════ */
    function crs(a, b, c, d, t) {
        return 0.5 * ((2 * b) + (-a + c) * t + (2 * a - 5 * b + 4 * c - d) * t * t + (-a + 3 * b - 3 * c + d) * t * t * t);
    }
    function crsd(a, b, c, d, t) {
        return 0.5 * ((-a + c) + (4 * a - 10 * b + 8 * c - 2 * d) * t + (-3 * a + 9 * b - 9 * c + 3 * d) * t * t);
    }

    /* ═══ Track ═════════════════════════════════════════════════ */
    var TRACK_WP = [
        [0, 0], [22, -1], [45, 1], [58, 10], [63, 25],
        [55, 40], [40, 50], [20, 54], [2, 48],
        [-10, 38], [-14, 25], [-10, 12], [-4, 2]
    ];

    function buildTrack(wp) {
        var n = wp.length, res = 30, total = n * res;
        var px = new Array(total), py = new Array(total);
        var txA = new Array(total), tyA = new Array(total);
        var nxA = new Array(total), nyA = new Array(total);
        var sArr = new Array(total);

        for (var seg = 0; seg < n; seg++) {
            var i0 = (seg - 1 + n) % n, i1 = seg, i2 = (seg + 1) % n, i3 = (seg + 2) % n;
            for (var j = 0; j < res; j++) {
                var t = j / res, idx = seg * res + j;
                px[idx] = crs(wp[i0][0], wp[i1][0], wp[i2][0], wp[i3][0], t);
                py[idx] = crs(wp[i0][1], wp[i1][1], wp[i2][1], wp[i3][1], t);
                var ddx = crsd(wp[i0][0], wp[i1][0], wp[i2][0], wp[i3][0], t);
                var ddy = crsd(wp[i0][1], wp[i1][1], wp[i2][1], wp[i3][1], t);
                var len = Math.sqrt(ddx * ddx + ddy * ddy) || 1e-10;
                txA[idx] = ddx / len;
                tyA[idx] = ddy / len;
                nxA[idx] = -tyA[idx];
                nyA[idx] = txA[idx];
            }
        }
        sArr[0] = 0;
        for (var i = 1; i < total; i++) {
            var dx = px[i] - px[i - 1], dy = py[i] - py[i - 1];
            sArr[i] = sArr[i - 1] + Math.sqrt(dx * dx + dy * dy);
        }
        var dx = px[0] - px[total - 1], dy = py[0] - py[total - 1];
        var totalLen = sArr[total - 1] + Math.sqrt(dx * dx + dy * dy);

        return { px: px, py: py, tx: txA, ty: tyA, nx: nxA, ny: nyA, s: sArr, n: total, totalLength: totalLen };
    }

    function projectOnTrack(X, Y, track) {
        var n = track.n, bestD2 = Infinity, bestIdx = 0;
        for (var i = 0; i < n; i += 4) {
            var dx = X - track.px[i], dy = Y - track.py[i];
            var d2 = dx * dx + dy * dy;
            if (d2 < bestD2) { bestD2 = d2; bestIdx = i; }
        }
        var start = (bestIdx - 8 + n) % n;
        bestD2 = Infinity;
        for (var j = 0; j < 16; j++) {
            var i = (start + j) % n;
            var dx = X - track.px[i], dy = Y - track.py[i];
            var d2 = dx * dx + dy * dy;
            if (d2 < bestD2) { bestD2 = d2; bestIdx = i; }
        }
        var p = track;
        var signedDist = (X - p.px[bestIdx]) * p.nx[bestIdx] + (Y - p.py[bestIdx]) * p.ny[bestIdx];
        return { s: p.s[bestIdx], d: signedDist, idx: bestIdx, cx: p.px[bestIdx], cy: p.py[bestIdx], nx: p.nx[bestIdx], ny: p.ny[bestIdx] };
    }

    function evalTrackAtIdx(idx, track) {
        var i = ((idx % track.n) + track.n) % track.n;
        return { x: track.px[i], y: track.py[i], tx: track.tx[i], ty: track.ty[i], nx: track.nx[i], ny: track.ny[i], s: track.s[i] };
    }

    function findTrackIdxForS(s, track) {
        s = ((s % track.totalLength) + track.totalLength) % track.totalLength;
        var lo = 0, hi = track.n - 1;
        while (lo < hi) {
            var mid = (lo + hi + 1) >> 1;
            if (track.s[mid] <= s) lo = mid; else hi = mid - 1;
        }
        return lo;
    }

    /* ═══ Vehicle Dynamics ══════════════════════════════════════ */
    function fCont(x, u) {
        var phi = x[2], vx = x[3], vy = x[4], r = x[5];
        var delta = u[0], D = u[1];
        var vxS = Math.max(vx, VP.vxMin);
        var alphaF = -Math.atan2(vy + r * VP.lf, vxS) + delta;
        var alphaR = -Math.atan2(vy - r * VP.lr, vxS);
        var Ffy = VP.Df * Math.sin(VP.Cf * Math.atan(VP.Bf * alphaF));
        var Fry = VP.Dr * Math.sin(VP.Cr * Math.atan(VP.Br * alphaR));
        var Frx = VP.Cm1 * D - VP.Cm2 * D * vx;
        var Fdrag = VP.Cr0 + VP.Cr2 * vx * vx;
        var cp = Math.cos(phi), sp = Math.sin(phi);
        var cd = Math.cos(delta), sd = Math.sin(delta);
        return [
            vx * cp - vy * sp,
            vx * sp + vy * cp,
            r,
            (Frx - Ffy * sd + VP.m * vy * r - Fdrag) / VP.m,
            (Fry + Ffy * cd - VP.m * vx * r) / VP.m,
            (Ffy * VP.lf * cd - Fry * VP.lr) / VP.Iz
        ];
    }

    function rk4Step(x, u, dt) {
        var k1 = fCont(x, u);
        var xm = new Array(NX);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + 0.5 * dt * k1[i];
        var k2 = fCont(xm, u);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + 0.5 * dt * k2[i];
        var k3 = fCont(xm, u);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + dt * k3[i];
        var k4 = fCont(xm, u);
        var xn = new Array(NX);
        for (var i = 0; i < NX; i++)
            xn[i] = x[i] + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        return xn;
    }

    function rk4Jac(x, u, dt) {
        var eps = 1e-6, f0 = rk4Step(x, u, dt);
        var A = [], B = [];
        for (var i = 0; i < NX; i++) { A.push(new Array(NX)); B.push(new Array(NU)); }
        for (var j = 0; j < NX; j++) {
            var xp = x.slice(); xp[j] += eps;
            var fp = rk4Step(xp, u, dt);
            for (var i = 0; i < NX; i++) A[i][j] = (fp[i] - f0[i]) / eps;
        }
        for (var j = 0; j < NU; j++) {
            var up = u.slice(); up[j] += eps;
            var fp = rk4Step(x, up, dt);
            for (var i = 0; i < NX; i++) B[i][j] = (fp[i] - f0[i]) / eps;
        }
        return { A: A, B: B, f0: f0 };
    }

    /* ═══ Pure Pursuit Controller ══════════════════════════════ */
    function purePursuitControl(x, sNow, track, targetSpeed) {
        var lookahead = Math.max(4.0, x[3] * 0.9);
        var sLook = sNow + lookahead;
        var idxL = findTrackIdxForS(sLook, track);
        var lp = evalTrackAtIdx(idxL, track);
        var dx = lp.x - x[0], dy = lp.y - x[1];
        var localAngle = Math.atan2(dy, dx) - x[2];
        localAngle = wrapAngle(localAngle);
        var L = VP.lf + VP.lr;
        var dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 0.1) dist = 0.1;
        var curvature = 2 * Math.sin(localAngle) / dist;
        var delta = clamp(Math.atan(curvature * L), -VP.deltaMax, VP.deltaMax);
        var speedErr = targetSpeed - x[3];
        var D = clamp(speedErr * 0.6, VP.DMin, VP.DMax);
        return [delta, D];
    }

    /* ═══ K-NN Terminal Cost ═══════════════════════════════════ */
    function findKNN(xN, safeSet, K) {
        var topK = [];
        for (var i = 0; i < safeSet.length; i++) {
            var dx = xN[0] - safeSet[i].x[0];
            var dy = xN[1] - safeSet[i].x[1];
            var dphi = angleDiff(xN[2], safeSet[i].x[2]);
            var dvx = xN[3] - safeSet[i].x[3];
            var d2 = dx * dx + dy * dy + 2 * dphi * dphi + 0.3 * dvx * dvx;
            if (topK.length < K) {
                topK.push({ d2: d2, J: safeSet[i].J });
                if (topK.length === K) topK.sort(function (a, b) { return a.d2 - b.d2; });
            } else if (d2 < topK[K - 1].d2) {
                topK[K - 1] = { d2: d2, J: safeSet[i].J };
                topK.sort(function (a, b) { return a.d2 - b.d2; });
            }
        }
        for (var i = 0; i < topK.length; i++) topK[i].d = Math.sqrt(topK[i].d2);
        return topK;
    }

    function knnCost(xN, safeSet) {
        if (safeSet.length === 0) return 0;
        var K = Math.min(KNN_K, safeSet.length);
        var topK = findKNN(xN, safeSet, K);
        var wSum = 0, JSum = 0;
        for (var i = 0; i < topK.length; i++) {
            var w = 1.0 / (topK[i].d + 0.05);
            wSum += w;
            JSum += w * topK[i].J;
        }
        var Vf = JSum / wSum;
        Vf = TERM_SCALE * Vf;
        Vf += W_SS * topK[0].d * topK[0].d;
        return Vf;
    }

    function knnGrad(xN, safeSet) {
        var eps = 1e-4, f0 = knnCost(xN, safeSet);
        var grad = new Array(NX);
        for (var i = 0; i < NX; i++) {
            var xp = xN.slice(); xp[i] += eps;
            grad[i] = (knnCost(xp, safeSet) - f0) / eps;
        }
        return grad;
    }

    /* ═══ Track + Centreline + Heading Cost ═════════════════════ */
    function getTrackInfo(x, track) {
        var proj = projectOnTrack(x[0], x[1], track);
        var absD = Math.abs(proj.d);
        var viol = Math.max(0, absD - TRACK_HW);
        var sgn = proj.d >= 0 ? 1 : -1;

        // Boundary penalty gradient/Hessian (active only outside track)
        var bGradX = viol > 0 ? W_TRACK * 2 * viol * sgn * proj.nx : 0;
        var bGradY = viol > 0 ? W_TRACK * 2 * viol * sgn * proj.ny : 0;
        var bHessXX = viol > 0 ? W_TRACK * 2 * proj.nx * proj.nx : 0;
        var bHessYY = viol > 0 ? W_TRACK * 2 * proj.ny * proj.ny : 0;

        // Centreline tracking cost: Q_LAT * d^2
        // d = (X - cx)*nx + (Y - cy)*ny  =>  dd/dX = nx,  dd/dY = ny
        var cGradX = Q_LAT * 2 * proj.d * proj.nx;
        var cGradY = Q_LAT * 2 * proj.d * proj.ny;
        var cHessXX = Q_LAT * 2 * proj.nx * proj.nx;
        var cHessYY = Q_LAT * 2 * proj.ny * proj.ny;

        // Heading cost: Q_HEAD * (phi - phi_ref)^2
        var phiRef = Math.atan2(track.ty[proj.idx], track.tx[proj.idx]);
        var ePhi = angleDiff(x[2], phiRef);
        var gradPhi = Q_HEAD * 2 * ePhi;
        var hessPhi = Q_HEAD * 2;

        return {
            gradX: bGradX + cGradX,
            gradY: bGradY + cGradY,
            hessXX: bHessXX + cHessXX,
            hessYY: bHessYY + cHessYY,
            gradPhi: gradPhi,
            hessPhi: hessPhi
        };
    }

    /* ═══ SCvx Solver for LMPC ═════════════════════════════════ */
    function scvxSolve(x0, usInit, track, safeSet, N) {
        var us = [];
        for (var k = 0; k < N; k++) us.push(usInit[k].slice());

        for (var scp = 0; scp < SCP_ITERS; scp++) {
            var xsBar = [x0.slice()];
            for (var k = 0; k < N; k++) xsBar.push(rk4Step(xsBar[k], us[k], DT_MPC));

            var Ab = [], Bb = [], cb = [];
            for (var k = 0; k < N; k++) {
                var jac = rk4Jac(xsBar[k], us[k], DT_MPC);
                Ab.push(jac.A); Bb.push(jac.B);
                var ck = new Array(NX);
                for (var i = 0; i < NX; i++) {
                    ck[i] = jac.f0[i];
                    for (var j = 0; j < NX; j++) ck[i] -= jac.A[i][j] * xsBar[k][j];
                    for (var j = 0; j < NU; j++) ck[i] -= jac.B[i][j] * us[k][j];
                }
                cb.push(ck);
            }

            var termGrad = knnGrad(xsBar[N], safeSet);
            var trkInfo = [];
            for (var k = 0; k <= N; k++) trkInfo.push(getTrackInfo(xsBar[k], track));

            var usBar = [];
            for (var k = 0; k < N; k++) usBar.push(us[k].slice());

            // Diagonal preconditioner
            var diagP = new Array(NX);
            for (var i = 0; i < NX; i++) {
                diagP[i] = TERM_HESS[i];
                if (i === 0) diagP[i] += trkInfo[N].hessXX;
                if (i === 1) diagP[i] += trkInfo[N].hessYY;
                if (i === 2) diagP[i] += trkInfo[N].hessPhi;
            }
            var prec = new Array(N);
            for (var k = N - 1; k >= 0; k--) {
                prec[k] = new Array(NU);
                for (var a = 0; a < NU; a++) {
                    prec[k][a] = Rd[a] + TR_WEIGHT;
                    for (var i = 0; i < NX; i++) prec[k][a] += Bb[k][i][a] * Bb[k][i][a] * diagP[i];
                    if (prec[k][a] < 1e-6) prec[k][a] = 1e-6;
                }
                var newDP = new Array(NX);
                for (var i = 0; i < NX; i++) {
                    newDP[i] = 0;
                    if (i === 0) newDP[i] += trkInfo[k].hessXX;
                    if (i === 1) newDP[i] += trkInfo[k].hessYY;
                    if (i === 2) newDP[i] += trkInfo[k].hessPhi;
                    for (var j = 0; j < NX; j++) newDP[i] += Ab[k][j][i] * Ab[k][j][i] * diagP[j];
                }
                diagP = newDP;
            }

            for (var qi = 0; qi < QP_ITERS; qi++) {
                // Forward pass (linearised dynamics)
                var xs = [x0.slice()];
                for (var k = 0; k < N; k++) {
                    var xn = new Array(NX);
                    for (var i = 0; i < NX; i++) {
                        var s = cb[k][i];
                        for (var j = 0; j < NX; j++) s += Ab[k][i][j] * xs[k][j];
                        for (var j = 0; j < NU; j++) s += Bb[k][i][j] * us[k][j];
                        xn[i] = s;
                    }
                    xs.push(xn);
                }

                // Adjoint backward pass
                var lam = new Array(NX);
                for (var i = 0; i < NX; i++) lam[i] = termGrad[i];

                var grad = new Array(N);
                for (var k = N - 1; k >= 0; k--) {
                    var gk = new Array(NU);
                    for (var a = 0; a < NU; a++) {
                        gk[a] = Rd[a] * us[k][a] + TR_WEIGHT * (us[k][a] - usBar[k][a]);
                        for (var i = 0; i < NX; i++) gk[a] += Bb[k][i][a] * lam[i];
                    }
                    grad[k] = gk;
                    var newLam = new Array(NX);
                    for (var i = 0; i < NX; i++) {
                        newLam[i] = 0;
                        if (i === 0) newLam[i] += trkInfo[k].gradX;
                        if (i === 1) newLam[i] += trkInfo[k].gradY;
                        if (i === 2) newLam[i] += trkInfo[k].gradPhi;
                        for (var j = 0; j < NX; j++) newLam[i] += Ab[k][j][i] * lam[j];
                    }
                    lam = newLam;
                }

                // Clamp gradient at bounds
                for (var k = 0; k < N; k++) {
                    if (us[k][0] <= -VP.deltaMax + 1e-4 && grad[k][0] > 0) grad[k][0] = 0;
                    if (us[k][0] >= VP.deltaMax - 1e-4 && grad[k][0] < 0) grad[k][0] = 0;
                    if (us[k][1] <= VP.DMin + 1e-4 && grad[k][1] > 0) grad[k][1] = 0;
                    if (us[k][1] >= VP.DMax - 1e-4 && grad[k][1] < 0) grad[k][1] = 0;
                }

                // Preconditioned direction
                var pdir = new Array(N);
                for (var k = 0; k < N; k++)
                    pdir[k] = [grad[k][0] / prec[k][0], grad[k][1] / prec[k][1]];

                // HVP for step size
                var dxH = [new Array(NX)];
                for (var i = 0; i < NX; i++) dxH[0][i] = 0;
                for (var k = 0; k < N; k++) {
                    var dxn = new Array(NX);
                    for (var i = 0; i < NX; i++) {
                        var s = 0;
                        for (var j = 0; j < NX; j++) s += Ab[k][i][j] * dxH[k][j];
                        for (var j = 0; j < NU; j++) s += Bb[k][i][j] * pdir[k][j];
                        dxn[i] = s;
                    }
                    dxH.push(dxn);
                }
                var dLam = new Array(NX);
                for (var i = 0; i < NX; i++) dLam[i] = TERM_HESS[i] * dxH[N][i];
                var Hpd = new Array(N);
                for (var k = N - 1; k >= 0; k--) {
                    var hk = new Array(NU);
                    for (var a = 0; a < NU; a++) {
                        hk[a] = (Rd[a] + TR_WEIGHT) * pdir[k][a];
                        for (var i = 0; i < NX; i++) hk[a] += Bb[k][i][a] * dLam[i];
                    }
                    Hpd[k] = hk;
                    var newDLam = new Array(NX);
                    for (var i = 0; i < NX; i++) {
                        newDLam[i] = 0;
                        if (i === 0) newDLam[i] += trkInfo[k].hessXX * dxH[k][i];
                        if (i === 1) newDLam[i] += trkInfo[k].hessYY * dxH[k][i];
                        if (i === 2) newDLam[i] += trkInfo[k].hessPhi * dxH[k][i];
                        for (var j = 0; j < NX; j++) newDLam[i] += Ab[k][j][i] * dLam[j];
                    }
                    dLam = newDLam;
                }

                var gd = 0, dHd = 0;
                for (var k = 0; k < N; k++) {
                    for (var a = 0; a < NU; a++) { gd += grad[k][a] * pdir[k][a]; dHd += pdir[k][a] * Hpd[k][a]; }
                }
                if (gd < 1e-8 || dHd < 1e-12) break;
                var alpha = gd / dHd;
                for (var k = 0; k < N; k++) {
                    us[k][0] = clamp(us[k][0] - alpha * pdir[k][0], -VP.deltaMax, VP.deltaMax);
                    us[k][1] = clamp(us[k][1] - alpha * pdir[k][1], VP.DMin, VP.DMax);
                }
            }
        }
        var xsFinal = [x0.slice()];
        for (var k = 0; k < N; k++) xsFinal.push(rk4Step(xsFinal[k], us[k], DT_MPC));
        return { xs: xsFinal, us: us };
    }

    /* ═══ Canvas helpers ═══════════════════════════════════════ */
    function setupCanvas(canvas) {
        var dpr = window.devicePixelRatio || 1;
        var w = canvas.clientWidth, h = canvas.clientHeight;
        canvas.width = w * dpr; canvas.height = h * dpr;
        var ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        return { ctx: ctx, w: w, h: h };
    }

    function w2s(wx, wy, cam) {
        return {
            x: cam.cx + (wx - cam.fx) * cam.sc,
            y: cam.cy - (wy - cam.fy) * cam.sc
        };
    }

    /* ═══ Drawing — Track View ═════════════════════════════════ */
    var ITER_COLORS = ['#888888', '#4488FF', '#44BB44', '#FF8800', '#FF4444', '#CC44FF', '#44DDDD', '#FFDD44'];

    function drawTrackView(canvas, sim) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0F1923'; ctx.fillRect(0, 0, W, H);

        var track = sim.track;
        var cam = { cx: W * 0.5, cy: H * 0.5, fx: sim.camFx, fy: sim.camFy, sc: sim.camSc * sim.camZoom };

        // Draw grass background
        ctx.fillStyle = '#1a2a1a'; ctx.fillRect(0, 0, W, H);

        // Track surface
        ctx.beginPath();
        for (var i = 0; i <= track.n; i++) {
            var ii = i % track.n;
            var p = w2s(track.px[ii] + TRACK_HW * track.nx[ii], track.py[ii] + TRACK_HW * track.ny[ii], cam);
            if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        ctx.moveTo(0, 0); // break path for evenodd
        ctx.beginPath();
        // outer
        for (var i = 0; i <= track.n; i++) {
            var ii = i % track.n;
            var p = w2s(track.px[ii] + TRACK_HW * track.nx[ii], track.py[ii] + TRACK_HW * track.ny[ii], cam);
            if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        // inner
        for (var i = track.n; i >= 0; i--) {
            var ii = ((i % track.n) + track.n) % track.n;
            var p = w2s(track.px[ii] - TRACK_HW * track.nx[ii], track.py[ii] - TRACK_HW * track.ny[ii], cam);
            if (i === track.n) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath();
        ctx.fillStyle = '#3a3a3a';
        ctx.fill('evenodd');

        // Edge lines
        ctx.lineWidth = 1.5; ctx.strokeStyle = '#ffffff';
        ctx.beginPath();
        for (var i = 0; i <= track.n; i++) {
            var ii = i % track.n;
            var p = w2s(track.px[ii] + TRACK_HW * track.nx[ii], track.py[ii] + TRACK_HW * track.ny[ii], cam);
            if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath(); ctx.stroke();
        ctx.beginPath();
        for (var i = 0; i <= track.n; i++) {
            var ii = i % track.n;
            var p = w2s(track.px[ii] - TRACK_HW * track.nx[ii], track.py[ii] - TRACK_HW * track.ny[ii], cam);
            if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath(); ctx.stroke();

        // Centreline dashed
        ctx.setLineDash([6, 5]); ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1;
        ctx.beginPath();
        for (var i = 0; i <= track.n; i++) {
            var ii = i % track.n;
            var p = w2s(track.px[ii], track.py[ii], cam);
            if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
        }
        ctx.closePath(); ctx.stroke(); ctx.setLineDash([]);

        // Start/finish line
        var sf0 = evalTrackAtIdx(0, track);
        var p1 = w2s(sf0.x + TRACK_HW * sf0.nx, sf0.y + TRACK_HW * sf0.ny, cam);
        var p2 = w2s(sf0.x - TRACK_HW * sf0.nx, sf0.y - TRACK_HW * sf0.ny, cam);
        ctx.strokeStyle = '#fff'; ctx.lineWidth = 3;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
        ctx.strokeStyle = '#D32F2F'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();

        // Previous lap trails
        if (sim.showPrevLaps) {
            for (var li = 0; li < sim.laps.length; li++) {
                var lap = sim.laps[li];
                var col = ITER_COLORS[li % ITER_COLORS.length];
                var alpha = 0.25 + 0.35 * (li / Math.max(1, sim.laps.length - 1));
                ctx.strokeStyle = col; ctx.globalAlpha = alpha; ctx.lineWidth = 1.5;
                ctx.beginPath();
                for (var i = 0; i < lap.pts.length; i++) {
                    var p = w2s(lap.pts[i][0], lap.pts[i][1], cam);
                    if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
                }
                ctx.stroke();
            }
            ctx.globalAlpha = 1;
        }

        // Safe set dots
        if (sim.showSafeSet && sim.safeSet.length > 0) {
            var step = Math.max(1, Math.floor(sim.safeSet.length / 600));
            for (var i = 0; i < sim.safeSet.length; i += step) {
                var ss = sim.safeSet[i];
                var p = w2s(ss.x[0], ss.x[1], cam);
                var col = ITER_COLORS[ss.iter % ITER_COLORS.length];
                ctx.fillStyle = col; ctx.globalAlpha = 0.35;
                ctx.beginPath(); ctx.arc(p.x, p.y, 2, 0, 2 * Math.PI); ctx.fill();
            }
            ctx.globalAlpha = 1;
        }

        // Current lap trail
        if (sim.curTraj.length > 1) {
            ctx.strokeStyle = '#D32F2F'; ctx.lineWidth = 2;
            ctx.beginPath();
            for (var i = 0; i < sim.curTraj.length; i++) {
                var p = w2s(sim.curTraj[i].x[0], sim.curTraj[i].x[1], cam);
                if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
            }
            ctx.stroke();
        }

        // MPC prediction
        if (sim.showPred && sim.pred && sim.pred.length > 1) {
            ctx.strokeStyle = '#00BCD4'; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
            ctx.beginPath();
            for (var i = 0; i < sim.pred.length; i++) {
                var p = w2s(sim.pred[i][0], sim.pred[i][1], cam);
                if (i === 0) ctx.moveTo(p.x, p.y); else ctx.lineTo(p.x, p.y);
            }
            ctx.stroke(); ctx.setLineDash([]);
        }

        // Car
        if (sim.x) {
            var cp = w2s(sim.x[0], sim.x[1], cam);
            ctx.save();
            ctx.translate(cp.x, cp.y);
            ctx.rotate(-sim.x[2]);
            var cL = VP.carL * cam.sc, cW = VP.carW * cam.sc;
            ctx.fillStyle = '#D32F2F';
            ctx.fillRect(-cL / 2, -cW / 2, cL, cW);
            ctx.beginPath();
            ctx.moveTo(cL / 2 + 3, 0);
            ctx.lineTo(cL / 2 - 4, -cW / 3);
            ctx.lineTo(cL / 2 - 4, cW / 3);
            ctx.closePath();
            ctx.fillStyle = '#FFD600'; ctx.fill();
            ctx.restore();
        }

        // HUD
        ctx.fillStyle = 'rgba(255,255,255,0.55)'; ctx.font = '11px monospace';
        var iter = sim.iteration;
        var mode = iter === 0 ? 'Pure Pursuit' : 'LMPC';
        ctx.fillText('Iter ' + iter + ' / ' + mode, 10, 18);
        if (sim.x) {
            ctx.fillText('v: ' + sim.x[3].toFixed(1) + ' m/s (' + (sim.x[3] * 3.6).toFixed(0) + ' km/h)', 10, 33);
            ctx.fillText('\u03B4: ' + (sim.lastU[0] * 180 / Math.PI).toFixed(1) + '\u00B0  D: ' + sim.lastU[1].toFixed(2), 10, 48);
        }
        var prog = sim.lapDist / sim.track.totalLength * 100;
        ctx.fillText('Progress: ' + prog.toFixed(0) + '%  t=' + sim.time.toFixed(1) + 's', 10, 63);
        ctx.fillStyle = 'rgba(255,255,255,0.2)'; ctx.font = '10px sans-serif';
        ctx.fillText('Scroll: zoom | Drag: pan', 10, H - 8);
    }

    /* ═══ Drawing — Plots ══════════════════════════════════════ */
    function drawPlots(canvas, sim) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0F1923'; ctx.fillRect(0, 0, W, H);

        var nPlots = 3, pH = Math.floor(H / nPlots);

        // 1. Lap time bar chart
        drawLapBars(ctx, 0, 0, W, pH, sim);
        // 2. Speed profile
        drawProfile(ctx, 0, pH, W, pH, sim, 'speed');
        // 3. Steering profile
        drawProfile(ctx, 0, 2 * pH, W, H - 2 * pH, sim, 'steer');
    }

    function drawLapBars(ctx, ox, oy, W, H, sim) {
        var pad = { l: 38, r: 8, t: 16, b: 18 };
        var pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
        ctx.save(); ctx.translate(ox, oy);

        ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 0.5;
        ctx.strokeRect(pad.l, pad.t, pw, ph);
        ctx.fillStyle = 'rgba(255,255,255,0.5)'; ctx.font = '10px sans-serif';
        ctx.fillText('Lap Times', pad.l + 4, pad.t + 12);

        if (sim.laps.length > 0) {
            var maxT = sim.laps[0].time;
            var barH = Math.min(16, (ph - 4) / sim.laps.length - 2);
            for (var i = 0; i < sim.laps.length; i++) {
                var bw = (sim.laps[i].time / maxT) * pw;
                var by = pad.t + 2 + i * (barH + 2);
                var frac = 1 - sim.laps[i].time / maxT;
                var rr = Math.floor(50 + 180 * frac);
                var gg = Math.floor(80 + 80 * (1 - Math.abs(frac - 0.5) * 2));
                var bb = Math.floor(200 * (1 - frac));
                ctx.fillStyle = 'rgb(' + rr + ',' + gg + ',' + bb + ')';
                ctx.fillRect(pad.l, by, bw, barH);
                ctx.fillStyle = '#ccc'; ctx.font = '9px monospace';
                ctx.fillText(sim.laps[i].time.toFixed(1) + 's', pad.l + bw + 3, by + barH - 2);
                ctx.fillText('#' + i, pad.l - 18, by + barH - 2);
            }
        }
        ctx.restore();
    }

    function drawProfile(ctx, ox, oy, W, H, sim, type) {
        var pad = { l: 38, r: 8, t: 16, b: 18 };
        var pw = W - pad.l - pad.r, ph = H - pad.t - pad.b;
        ctx.save(); ctx.translate(ox, oy);

        ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 0.5;
        ctx.strokeRect(pad.l, pad.t, pw, ph);

        var title = type === 'speed' ? 'Speed (m/s)' : 'Steering (\u00B0)';
        ctx.fillStyle = 'rgba(255,255,255,0.5)'; ctx.font = '10px sans-serif';
        ctx.fillText(title, pad.l + 4, pad.t + 12);

        // Best previous lap (grey dashed)
        if (sim.laps.length > 0) {
            var best = sim.laps[sim.laps.length - 1];
            var data = type === 'speed' ? best.speed : best.steer;
            if (data && data.length > 1) {
                var yMin = Infinity, yMax = -Infinity;
                for (var i = 0; i < data.length; i++) { if (data[i] < yMin) yMin = data[i]; if (data[i] > yMax) yMax = data[i]; }
                var margin = (yMax - yMin) * 0.15 + 0.5;
                yMin -= margin; yMax += margin;
                ctx.strokeStyle = 'rgba(150,150,150,0.5)'; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
                ctx.beginPath();
                for (var i = 0; i < data.length; i++) {
                    var sx = pad.l + (i / (data.length - 1)) * pw;
                    var sy = pad.t + (1 - (data[i] - yMin) / (yMax - yMin)) * ph;
                    if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
                }
                ctx.stroke(); ctx.setLineDash([]);
            }
        }

        // Current lap data
        var curData = [];
        for (var i = 0; i < sim.curTraj.length; i++) {
            if (type === 'speed') curData.push(sim.curTraj[i].x[3]);
            else curData.push(sim.curTraj[i].u[0] * 180 / Math.PI);
        }
        if (curData.length > 1) {
            var yMin = Infinity, yMax = -Infinity;
            for (var i = 0; i < curData.length; i++) { if (curData[i] < yMin) yMin = curData[i]; if (curData[i] > yMax) yMax = curData[i]; }
            // Include previous best range if available
            if (sim.laps.length > 0) {
                var bdata = type === 'speed' ? sim.laps[sim.laps.length - 1].speed : sim.laps[sim.laps.length - 1].steer;
                if (bdata) for (var i = 0; i < bdata.length; i++) { if (bdata[i] < yMin) yMin = bdata[i]; if (bdata[i] > yMax) yMax = bdata[i]; }
            }
            var margin = (yMax - yMin) * 0.15 + 0.5;
            yMin -= margin; yMax += margin;
            ctx.strokeStyle = '#D32F2F'; ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (var i = 0; i < curData.length; i++) {
                var sx = pad.l + (i / Math.max(1, curData.length - 1)) * pw;
                var sy = pad.t + (1 - (curData[i] - yMin) / (yMax - yMin)) * ph;
                if (i === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
            }
            ctx.stroke();

            ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.font = '9px monospace';
            ctx.fillText(yMax.toFixed(1), pad.l - 34, pad.t + 9);
            ctx.fillText(yMin.toFixed(1), pad.l - 34, pad.t + ph);
        }
        ctx.restore();
    }

    /* ═══ CSS Injection ════════════════════════════════════════ */
    var cssInjected = false;
    function injectCSS() {
        if (cssInjected) return; cssInjected = true;
        var s = document.createElement('style');
        s.textContent = [
            '.racing-demo{margin:1.5rem 0;border:2px solid #D32F2F;border-radius:6px;background:#0F1923;overflow:hidden;font-family:"Source Sans 3",system-ui,sans-serif;color:#ccc}',
            '.rc-header{background:#D32F2F;color:#fff;padding:0.45rem 1rem;font-weight:700;font-size:0.95rem}',
            '.rc-ctrls{display:flex;flex-wrap:wrap;align-items:center;gap:0.5rem 0.9rem;padding:0.45rem 1rem;border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.82rem}',
            '.rc-ctrls label{display:flex;align-items:center;gap:0.3rem;white-space:nowrap}',
            '.rc-slider{width:65px;accent-color:#D32F2F}',
            '.rc-val{min-width:2.4em;text-align:right;font-variant-numeric:tabular-nums;font-family:"SF Mono","Fira Code",monospace;font-size:0.78rem;color:#f88}',
            '.rc-btn{padding:0.25rem 0.85rem;border:1px solid rgba(255,255,255,0.2);border-radius:4px;background:rgba(211,47,47,0.18);color:#f88;font-size:0.82rem;cursor:pointer;transition:background 0.15s}',
            '.rc-btn:hover{background:rgba(211,47,47,0.35)}',
            '.rc-btn:disabled{opacity:0.4;cursor:default}',
            '.rc-check{accent-color:#D32F2F;margin-right:2px}',
            '.rc-sep{width:1px;height:18px;background:rgba(255,255,255,0.12);margin:0 0.2rem}',
            '.rc-body{display:flex;min-height:400px}',
            '.rc-scene-wrap{flex:1;min-width:0;position:relative}',
            '.rc-scene{width:100%;height:100%;display:block;cursor:grab}',
            '.rc-scene:active{cursor:grabbing}',
            '.rc-plot-wrap{flex:0 0 220px;border-left:1px solid rgba(255,255,255,0.06)}',
            '.rc-plots{width:100%;height:100%;display:block}',
            '.rc-info{padding:0.3rem 1rem;border-top:1px solid rgba(255,255,255,0.06);font-size:0.78rem;font-family:"SF Mono","Fira Code",monospace;color:#aaa;display:flex;gap:1.2rem;flex-wrap:wrap}',
            '.rc-info span{white-space:nowrap}',
            '@media(max-width:720px){.rc-body{flex-direction:column;min-height:auto}.rc-scene-wrap{height:320px}.rc-plot-wrap{flex:none;height:220px;border-left:none;border-top:1px solid rgba(255,255,255,0.06)}}'
        ].join('\n');
        document.head.appendChild(s);
    }

    /* ═══ Build HTML ═══════════════════════════════════════════ */
    function buildHTML(el) {
        el.innerHTML =
        '<div class="rc-header">Autonomous Racing — Learning MPC (LMPC)</div>' +
        '<div class="rc-ctrls">' +
          '<button class="rc-btn rc-run-btn">\u25B6 Run Lap</button>' +
          '<button class="rc-btn rc-auto-btn">\u23E9 Auto (5 laps)</button>' +
          '<button class="rc-btn rc-reset-btn">\u21BB Reset</button>' +
          '<span class="rc-sep"></span>' +
          '<label>N <input type="range" class="rc-slider" data-id="hor" min="8" max="30" value="20" step="1"><span class="rc-val">20</span></label>' +
          '<label>Speed <input type="range" class="rc-slider" data-id="spd" min="1" max="16" value="2" step="1"><span class="rc-val">2\u00D7</span></label>' +
          '<span class="rc-sep"></span>' +
          '<label><input type="checkbox" class="rc-check" data-id="showSS" checked> Safe Set</label>' +
          '<label><input type="checkbox" class="rc-check" data-id="showPred" checked> Prediction</label>' +
          '<label><input type="checkbox" class="rc-check" data-id="showLaps" checked> Prev Laps</label>' +
        '</div>' +
        '<div class="rc-body">' +
          '<div class="rc-scene-wrap"><canvas class="rc-scene"></canvas></div>' +
          '<div class="rc-plot-wrap"><canvas class="rc-plots"></canvas></div>' +
        '</div>' +
        '<div class="rc-info">' +
          '<span class="rc-info-iter">Iteration: 0 / Ready</span>' +
          '<span class="rc-info-time">Lap: --</span>' +
          '<span class="rc-info-best">Best: --</span>' +
          '<span class="rc-info-ss">SS: 0 pts</span>' +
        '</div>';
    }

    /* ═══ Init Demo ═══════════════════════════════════════════ */
    function initDemo(container) {
        injectCSS();
        buildHTML(container);

        var sceneCvs = container.querySelector('.rc-scene');
        var plotCvs = container.querySelector('.rc-plots');
        var runBtn = container.querySelector('.rc-run-btn');
        var autoBtn = container.querySelector('.rc-auto-btn');
        var resetBtn = container.querySelector('.rc-reset-btn');
        var infoIter = container.querySelector('.rc-info-iter');
        var infoTime = container.querySelector('.rc-info-time');
        var infoBest = container.querySelector('.rc-info-best');
        var infoSS = container.querySelector('.rc-info-ss');

        function sliderVal(id) {
            var s = container.querySelector('[data-id="' + id + '"]');
            return s ? parseFloat(s.value) : 0;
        }
        function checkVal(id) {
            var c = container.querySelector('[data-id="' + id + '"]');
            return c ? c.checked : false;
        }

        var track = buildTrack(TRACK_WP);

        // Compute track bounding box for camera
        var minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        for (var i = 0; i < track.n; i++) {
            if (track.px[i] - TRACK_HW < minX) minX = track.px[i] - TRACK_HW;
            if (track.px[i] + TRACK_HW > maxX) maxX = track.px[i] + TRACK_HW;
            if (track.py[i] - TRACK_HW < minY) minY = track.py[i] - TRACK_HW;
            if (track.py[i] + TRACK_HW > maxY) maxY = track.py[i] + TRACK_HW;
        }
        var trackCx = (minX + maxX) / 2, trackCy = (minY + maxY) / 2;
        var trackW = maxX - minX + 10, trackH = maxY - minY + 10;

        var startPhi = Math.atan2(track.ty[0], track.tx[0]);

        var sim = {
            state: 'idle',
            iteration: 0,
            x: [track.px[0], track.py[0], startPhi, 1.0, 0, 0],
            lapDist: 0, prevS: 0, time: 0, step: 0,
            curTraj: [],
            safeSet: [],
            laps: [],
            usWarm: null,
            pred: null,
            lastU: [0, 0],
            camFx: trackCx, camFy: trackCy,
            camSc: 6, camZoom: 1.0,
            showSafeSet: true, showPred: true, showPrevLaps: true,
            horizonN: N_DEFAULT,
            speedMult: 2,
            autoMode: false, autoLapsLeft: 0,
            track: track
        };

        var running = false, animId = null;

        function updateInfo() {
            var mode = sim.state === 'idle' ? 'Ready' : (sim.iteration === 0 ? 'Pure Pursuit' : 'LMPC');
            infoIter.textContent = 'Iteration: ' + sim.iteration + ' / ' + mode;
            infoTime.textContent = 'Lap: ' + sim.time.toFixed(1) + 's';
            var bestT = Infinity;
            for (var i = 0; i < sim.laps.length; i++) if (sim.laps[i].time < bestT) bestT = sim.laps[i].time;
            infoBest.textContent = 'Best: ' + (bestT < Infinity ? bestT.toFixed(1) + 's' : '--');
            infoSS.textContent = 'SS: ' + sim.safeSet.length + ' pts';
        }

        function resetCar() {
            sim.x = [track.px[0], track.py[0], startPhi, 1.0, 0, 0];
            sim.lapDist = 0;
            sim.prevS = track.s[0];
            sim.time = 0;
            sim.step = 0;
            sim.curTraj = [];
            sim.usWarm = null;
            sim.pred = null;
            sim.lastU = [0, 0];
        }

        function fullReset() {
            running = false;
            if (animId) { cancelAnimationFrame(animId); animId = null; }
            sim.state = 'idle';
            sim.iteration = 0;
            sim.safeSet = [];
            sim.laps = [];
            sim.autoMode = false;
            sim.autoLapsLeft = 0;
            sim.camZoom = 1.0;
            resetCar();
            runBtn.disabled = false;
            autoBtn.disabled = false;
            runBtn.textContent = '\u25B6 Run Lap';
            updateInfo();
            render();
        }

        function generateInitGuess(x0, sNow, N) {
            var us = [];
            var xSim = x0.slice();
            var s = sNow;
            for (var k = 0; k < N; k++) {
                var u = purePursuitControl(xSim, s, track, Math.max(PP_SPEED, x0[3]));
                us.push(u);
                xSim = rk4Step(xSim, u, DT_MPC);
                var proj = projectOnTrack(xSim[0], xSim[1], track);
                s = proj.s;
            }
            return us;
        }

        function physicsStep() {
            var proj = projectOnTrack(sim.x[0], sim.x[1], track);
            var sNow = proj.s;

            if (sim.step % MPC_EVERY === 0) {
                var N = sim.horizonN;
                if (sim.iteration === 0) {
                    sim.lastU = purePursuitControl(sim.x, sNow, track, PP_SPEED);
                    sim.pred = null;
                } else {
                    // LMPC
                    var usInit;
                    if (sim.usWarm && sim.usWarm.length >= N) {
                        usInit = [];
                        for (var k = 1; k < sim.usWarm.length; k++) usInit.push(sim.usWarm[k].slice());
                        while (usInit.length < N) usInit.push(sim.usWarm[sim.usWarm.length - 1].slice());
                        usInit = usInit.slice(0, N);
                    } else {
                        usInit = generateInitGuess(sim.x, sNow, N);
                    }
                    var sol = scvxSolve(sim.x, usInit, track, sim.safeSet, N);
                    sim.lastU = sol.us[0].slice();
                    sim.usWarm = sol.us;
                    sim.pred = sol.xs;
                }
            }

            // Clamp inputs
            sim.lastU[0] = clamp(sim.lastU[0], -VP.deltaMax, VP.deltaMax);
            sim.lastU[1] = clamp(sim.lastU[1], VP.DMin, VP.DMax);

            // Step true dynamics
            sim.x = rk4Step(sim.x, sim.lastU, DT_SIM);
            sim.x[3] = clamp(sim.x[3], VP.vxMin, VP.vxMax);
            sim.x[4] = clamp(sim.x[4], -5, 5);
            sim.x[5] = clamp(sim.x[5], -3, 3);

            // Track progress
            var projNew = projectOnTrack(sim.x[0], sim.x[1], track);
            var ds = projNew.s - sim.prevS;
            if (ds < -track.totalLength * 0.5) ds += track.totalLength;
            if (ds > track.totalLength * 0.5) ds -= track.totalLength;
            if (ds > 0) sim.lapDist += ds;
            sim.prevS = projNew.s;

            sim.time += DT_SIM;
            sim.step++;

            // Log trajectory
            sim.curTraj.push({ x: sim.x.slice(), u: sim.lastU.slice(), s: projNew.s });

            // Check lap completion
            if (sim.lapDist >= track.totalLength) {
                completeLap();
            }

            // Safety: if car is way off track, abort WITHOUT storing data
            if (Math.abs(projNew.d) > TRACK_HW * 3) {
                abortLap();
            }
        }

        function abortLap() {
            // Off-track: discard trajectory, don't add to safe set
            console.warn('LMPC: lap aborted (off track), discarding data');
            sim.state = 'idle';
            running = false;
            if (animId) { cancelAnimationFrame(animId); animId = null; }
            resetCar();
            updateInfo();
            render();
            // Auto mode: retry same iteration
            if (sim.autoMode && sim.autoLapsLeft > 0) {
                sim.autoLapsLeft--;
                setTimeout(function () { startLap(); }, 300);
            } else {
                sim.autoMode = false;
                runBtn.disabled = false;
                autoBtn.disabled = false;
                runBtn.textContent = '\u25B6 Run Lap';
            }
        }

        function completeLap() {
            var lapTime = sim.time;
            var lapSteps = sim.curTraj.length;

            // Build safe set entries
            for (var k = 0; k < lapSteps; k++) {
                sim.safeSet.push({
                    x: sim.curTraj[k].x.slice(),
                    J: lapSteps - k,
                    iter: sim.iteration
                });
            }

            // Store lap data
            var pts = [], speeds = [], steers = [];
            var sampleStep = Math.max(1, Math.floor(lapSteps / 200));
            for (var k = 0; k < lapSteps; k += sampleStep) {
                pts.push([sim.curTraj[k].x[0], sim.curTraj[k].x[1]]);
                speeds.push(sim.curTraj[k].x[3]);
                steers.push(sim.curTraj[k].u[0] * 180 / Math.PI);
            }
            sim.laps.push({ time: lapTime, pts: pts, speed: speeds, steer: steers });

            sim.iteration++;
            sim.state = 'idle';
            running = false;
            if (animId) { cancelAnimationFrame(animId); animId = null; }

            resetCar();
            updateInfo();
            render();

            // Auto mode: continue
            if (sim.autoMode && sim.autoLapsLeft > 0) {
                sim.autoLapsLeft--;
                setTimeout(function () { startLap(); }, 300);
            } else {
                sim.autoMode = false;
                runBtn.disabled = false;
                autoBtn.disabled = false;
                runBtn.textContent = '\u25B6 Run Lap';
            }
        }

        function render() {
            var cW = sceneCvs.clientWidth, cH = sceneCvs.clientHeight;
            sim.camSc = Math.min(cW / trackW, cH / trackH) * 0.85;
            drawTrackView(sceneCvs, sim);
            drawPlots(plotCvs, sim);
        }

        var lastFrame = 0, accum = 0;
        function animLoop(ts) {
            if (!lastFrame) lastFrame = ts;
            var frameDt = (ts - lastFrame) / 1000;
            lastFrame = ts;
            if (frameDt > 0.2) frameDt = 0.2;
            accum += frameDt * sim.speedMult;
            var steps = 0;
            while (accum >= DT_SIM && steps < 8) {
                physicsStep();
                accum -= DT_SIM;
                steps++;
                if (sim.state === 'idle') break;
            }
            updateInfo();
            render();
            if (running && sim.state !== 'idle') animId = requestAnimationFrame(animLoop);
        }

        function startLap() {
            if (sim.state !== 'idle') return;
            sim.state = 'running';
            running = true;
            lastFrame = 0; accum = 0;
            resetCar();
            runBtn.disabled = true;
            autoBtn.disabled = true;
            runBtn.textContent = '\u23F8 Running...';
            animId = requestAnimationFrame(animLoop);
        }

        // Event handlers
        container.addEventListener('input', function (e) {
            var s = e.target;
            if (s.classList.contains('rc-slider')) {
                var v = s.nextElementSibling;
                if (v) {
                    var id = s.getAttribute('data-id');
                    if (id === 'spd') v.textContent = parseFloat(s.value).toFixed(0) + '\u00D7';
                    else v.textContent = parseFloat(s.value).toFixed(s.step < 1 ? 1 : 0);
                }
                sim.horizonN = Math.round(sliderVal('hor'));
                sim.speedMult = sliderVal('spd');
            }
            if (s.classList.contains('rc-check')) {
                sim.showSafeSet = checkVal('showSS');
                sim.showPred = checkVal('showPred');
                sim.showPrevLaps = checkVal('showLaps');
            }
            if (!running) render();
        });

        runBtn.addEventListener('click', function () {
            if (sim.state === 'idle') startLap();
        });
        autoBtn.addEventListener('click', function () {
            if (sim.state === 'idle') {
                sim.autoMode = true;
                sim.autoLapsLeft = 4;
                startLap();
            }
        });
        resetBtn.addEventListener('click', fullReset);

        // Camera: drag to pan, scroll to zoom
        var drag = { on: false, sx: 0, sy: 0, fx0: 0, fy0: 0 };
        sceneCvs.addEventListener('mousedown', function (e) {
            e.preventDefault();
            drag.on = true; drag.sx = e.clientX; drag.sy = e.clientY;
            drag.fx0 = sim.camFx; drag.fy0 = sim.camFy;
        });
        window.addEventListener('mousemove', function (e) {
            if (!drag.on) return;
            var sc = sim.camSc * sim.camZoom;
            sim.camFx = drag.fx0 - (e.clientX - drag.sx) / sc;
            sim.camFy = drag.fy0 + (e.clientY - drag.sy) / sc;
            if (!running) render();
        });
        window.addEventListener('mouseup', function () { drag.on = false; });
        sceneCvs.addEventListener('contextmenu', function (e) { e.preventDefault(); });
        sceneCvs.addEventListener('wheel', function (e) {
            e.preventDefault();
            var delta = e.deltaY > 0 ? -0.08 : 0.08;
            sim.camZoom = clamp(sim.camZoom + delta * sim.camZoom, 0.3, 5.0);
            if (!running) render();
        }, { passive: false });

        // Touch
        var touch0 = null, touchDist0 = 0;
        sceneCvs.addEventListener('touchstart', function (e) {
            if (e.touches.length === 1) {
                touch0 = { x: e.touches[0].clientX, y: e.touches[0].clientY, fx: sim.camFx, fy: sim.camFy };
            } else if (e.touches.length === 2) {
                var dx = e.touches[0].clientX - e.touches[1].clientX;
                var dy = e.touches[0].clientY - e.touches[1].clientY;
                touchDist0 = Math.sqrt(dx * dx + dy * dy);
                touch0 = { zoom: sim.camZoom };
            }
        }, { passive: true });
        sceneCvs.addEventListener('touchmove', function (e) {
            e.preventDefault();
            if (e.touches.length === 1 && touch0 && touch0.fx !== undefined) {
                var sc = sim.camSc * sim.camZoom;
                sim.camFx = touch0.fx - (e.touches[0].clientX - touch0.x) / sc;
                sim.camFy = touch0.fy + (e.touches[0].clientY - touch0.y) / sc;
                if (!running) render();
            } else if (e.touches.length === 2 && touch0 && touch0.zoom !== undefined) {
                var dx = e.touches[0].clientX - e.touches[1].clientX;
                var dy = e.touches[0].clientY - e.touches[1].clientY;
                var d = Math.sqrt(dx * dx + dy * dy);
                if (touchDist0 > 0) {
                    sim.camZoom = clamp(touch0.zoom * d / touchDist0, 0.3, 5.0);
                    if (!running) render();
                }
            }
        }, { passive: false });
        sceneCvs.addEventListener('touchend', function () { touch0 = null; }, { passive: true });

        fullReset();
    }

    /* ═══ Auto-init ════════════════════════════════════════════ */
    function initAll(root) {
        var els = (root || document).querySelectorAll('.racing-demo');
        for (var i = 0; i < els.length; i++) initDemo(els[i]);
    }
    if (document.readyState === 'loading')
        document.addEventListener('DOMContentLoaded', function () { initAll(); });
    else initAll();

    if (typeof window !== 'undefined') window.RacingDemo = { initAll: initAll };
})();
