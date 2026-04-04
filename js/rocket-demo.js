/**
 * Interactive Rocket Landing Demo — SCvx (Successive Convexification) MPC
 * Pure JS/Canvas: at every time step the controller linearises the
 * nonlinear RK4 dynamics, solves a bounded QP via projected gradient
 * descent (adjoint gradient + Hessian-vector product for exact step
 * sizes), then applies the first control in a receding-horizon loop.
 *
 * HTML hook:  <div class="rocket-demo"></div>
 * Init call:  RocketDemo.initAll(root)
 */
(function () {
    'use strict';

    var NX = 6, NU = 2;

    function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

    /* ═══════════════════════════════════════════════════════
       Continuous-time rocket dynamics  f(x,u)
       State  x = [px, py, vx, vy, theta, omega]
       Input  u = [T, delta]   (thrust, gimbal angle)
       ═══════════════════════════════════════════════════════ */
    function fCont(x, u, p) {
        var sn = Math.sin(x[4] + u[1]);
        var cs = Math.cos(x[4] + u[1]);
        var Tm = u[0] / p.m;
        return [
            x[2],
            x[3],
            -Tm * sn,
            Tm * cs - p.g,
            x[5],
            -(u[0] * p.l / p.J) * Math.sin(u[1])
        ];
    }

    /* ═══ RK4 integrator ═══════════════════════════════════ */
    function rk4(x, u, p) {
        var h = p.dt;
        var k1 = fCont(x, u, p);
        var xm = new Array(NX);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + 0.5 * h * k1[i];
        var k2 = fCont(xm, u, p);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + 0.5 * h * k2[i];
        var k3 = fCont(xm, u, p);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + h * k3[i];
        var k4 = fCont(xm, u, p);
        var xn = new Array(NX);
        for (var i = 0; i < NX; i++)
            xn[i] = x[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        return xn;
    }

    /* ═══ Numerical Jacobians of the RK4 map ══════════════ */
    function rk4Jac(x, u, p) {
        var eps = 1e-6;
        var f0 = rk4(x, u, p);
        var A = [], B = [];
        for (var i = 0; i < NX; i++) { A.push(new Array(NX)); B.push(new Array(NU)); }

        for (var j = 0; j < NX; j++) {
            var xp = x.slice(); xp[j] += eps;
            var fp = rk4(xp, u, p);
            for (var i = 0; i < NX; i++) A[i][j] = (fp[i] - f0[i]) / eps;
        }
        for (var j = 0; j < NU; j++) {
            var up = u.slice(); up[j] += eps;
            var fp = rk4(x, up, p);
            for (var i = 0; i < NX; i++) B[i][j] = (fp[i] - f0[i]) / eps;
        }
        return { A: A, B: B };
    }

    /* ═══ Small-matrix utilities (NX×NX, NX×NU, NU×NU) ═══ */
    function matMul(A, B) {
        var m = A.length, pp = A[0].length, n = B[0].length;
        var C = [];
        for (var i = 0; i < m; i++) {
            C.push(new Array(n));
            for (var j = 0; j < n; j++) {
                var s = 0;
                for (var k = 0; k < pp; k++) s += A[i][k] * B[k][j];
                C[i][j] = s;
            }
        }
        return C;
    }
    function matTrans(A) {
        var m = A.length, n = A[0].length, T = [];
        for (var j = 0; j < n; j++) {
            T.push(new Array(m));
            for (var i = 0; i < m; i++) T[j][i] = A[i][j];
        }
        return T;
    }
    function matVec(A, v) {
        var m = A.length, n = A[0].length, r = new Array(m);
        for (var i = 0; i < m; i++) {
            var s = 0;
            for (var j = 0; j < n; j++) s += A[i][j] * v[j];
            r[i] = s;
        }
        return r;
    }
    function inv2x2(M) {
        var a = M[0][0], b = M[0][1], c = M[1][0], d = M[1][1];
        var det = a * d - b * c;
        if (Math.abs(det) < 1e-14) det = 1e-14;
        var id = 1 / det;
        return [[d * id, -b * id], [-c * id, a * id]];
    }
    function matAdd(A, B) {
        var m = A.length, n = A[0].length, C = [];
        for (var i = 0; i < m; i++) {
            C.push(new Array(n));
            for (var j = 0; j < n; j++) C[i][j] = A[i][j] + B[i][j];
        }
        return C;
    }
    function matSub(A, B) {
        var m = A.length, n = A[0].length, C = [];
        for (var i = 0; i < m; i++) {
            C.push(new Array(n));
            for (var j = 0; j < n; j++) C[i][j] = A[i][j] - B[i][j];
        }
        return C;
    }
    function diagMat(d) {
        var n = d.length, M = [];
        for (var i = 0; i < n; i++) {
            M.push(new Array(n));
            for (var j = 0; j < n; j++) M[i][j] = (i === j) ? d[i] : 0;
        }
        return M;
    }

    /* ═══ Clamp control ════════════════════════════════════ */
    function clampU(u, p) {
        return [clamp(u[0], p.Tmin, p.Tmax), clamp(u[1], -p.dmax, p.dmax)];
    }

    /* ═══ Cost helpers ═════════════════════════════════════ */
    var GROUND_PEN = 5000;

    function totalCost(xs, us, xRefs, uRef, Qd, Rd, Qfd) {
        var N = us.length, cost = 0;
        for (var k = 0; k < N; k++) {
            for (var i = 0; i < NX; i++) {
                var dx = xs[k][i] - xRefs[k][i];
                cost += 0.5 * Qd[i] * dx * dx;
            }
            for (var i = 0; i < NU; i++) {
                var du = us[k][i] - uRef[i];
                cost += 0.5 * Rd[i] * du * du;
            }
            if (xs[k][1] < 0) cost += 0.5 * GROUND_PEN * xs[k][1] * xs[k][1];
        }
        for (var i = 0; i < NX; i++) {
            var dx = xs[N][i] - xRefs[N][i];
            cost += 0.5 * Qfd[i] * dx * dx;
        }
        if (xs[N][1] < 0) cost += 0.5 * GROUND_PEN * xs[N][1] * xs[N][1];
        return cost;
    }

    /* ═══ Reference trajectory (cubic Hermite) ════════════ */
    function genReference(x0, xf, N, dt) {
        var T = N * dt, refs = [];
        for (var k = 0; k <= N; k++) {
            var s = k / N;
            var h00 = 1 - 3 * s * s + 2 * s * s * s;
            var h10 = s - 2 * s * s + s * s * s;
            var h01 = 3 * s * s - 2 * s * s * s;
            var dh00 = (-6 * s + 6 * s * s) / T;
            var dh10 = (1 - 4 * s + 3 * s * s) / T;
            var dh01 = (6 * s - 6 * s * s) / T;
            var ref = [
                x0[0] * h00 + xf[0] * h01 + x0[2] * T * h10,
                Math.max(x0[1] * h00 + xf[1] * h01 + x0[3] * T * h10, 0),
                x0[0] * dh00 + xf[0] * dh01 + x0[2] * T * dh10,
                x0[1] * dh00 + xf[1] * dh01 + x0[3] * T * dh10,
                x0[4] * (1 - s),
                0
            ];
            refs.push(ref);
        }
        return refs;
    }

    /* ═══ SCvx solver (Successive Convexification) ═════════
       Outer loop: SCP iterations (linearise → QP → update).
       Inner QP: projected steepest descent with exact step
       size via adjoint gradient + Hessian-vector product.
       Returns { xs, us, cost }                              */

    var SCVX_QP_ITERS = 20;
    var SCVX_SCP_ITERS = 5;
    var SCVX_TR = 1.5;

    /* QP gradient via adjoint (forward-backward on linearised dynamics) */
    function qpGrad(us, Ab, Bb, cb, x0, xRefs, Qd, Rd, Qfd, uRef, wTR, usBar) {
        var N = us.length;
        /* Forward: predict states via linearised dynamics */
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
        /* Backward: adjoint costate propagation */
        var lam = new Array(NX);
        for (var i = 0; i < NX; i++) lam[i] = Qfd[i] * (xs[N][i] - xRefs[N][i]);
        if (xs[N][1] < 0) lam[1] += GROUND_PEN * xs[N][1];
        var grad = new Array(N);
        for (var k = N - 1; k >= 0; k--) {
            var gk = new Array(NU);
            for (var a = 0; a < NU; a++) {
                gk[a] = Rd[a] * (us[k][a] - uRef[a]) + wTR * (us[k][a] - usBar[k][a]);
                for (var i = 0; i < NX; i++) gk[a] += Bb[k][i][a] * lam[i];
            }
            grad[k] = gk;
            var newLam = new Array(NX);
            for (var i = 0; i < NX; i++) {
                newLam[i] = Qd[i] * (xs[k][i] - xRefs[k][i]);
                if (i === 1 && xs[k][1] < 0) newLam[i] += GROUND_PEN * xs[k][1];
                for (var j = 0; j < NX; j++) newLam[i] += Ab[k][j][i] * lam[j];
            }
            lam = newLam;
        }
        return grad;
    }

    /* Hessian-vector product (for exact step size in steepest descent) */
    function qpHvp(d, Ab, Bb, Qd, Rd, Qfd, wTR, xsBar) {
        var N = d.length;
        /* Forward: propagate perturbation δx through linearised dynamics */
        var dx = [new Array(NX)];
        for (var i = 0; i < NX; i++) dx[0][i] = 0;
        for (var k = 0; k < N; k++) {
            var dxn = new Array(NX);
            for (var i = 0; i < NX; i++) {
                var s = 0;
                for (var j = 0; j < NX; j++) s += Ab[k][i][j] * dx[k][j];
                for (var j = 0; j < NU; j++) s += Bb[k][i][j] * d[k][j];
                dxn[i] = s;
            }
            dx.push(dxn);
        }
        /* Backward: adjoint for H·d */
        var lam = new Array(NX);
        for (var i = 0; i < NX; i++) {
            lam[i] = Qfd[i] * dx[N][i];
            if (i === 1 && xsBar[N][1] < 0) lam[i] += GROUND_PEN * dx[N][i];
        }
        var Hd = new Array(N);
        for (var k = N - 1; k >= 0; k--) {
            var hk = new Array(NU);
            for (var a = 0; a < NU; a++) {
                hk[a] = (Rd[a] + wTR) * d[k][a];
                for (var i = 0; i < NX; i++) hk[a] += Bb[k][i][a] * lam[i];
            }
            Hd[k] = hk;
            var newLam = new Array(NX);
            for (var i = 0; i < NX; i++) {
                newLam[i] = Qd[i] * dx[k][i];
                if (i === 1 && xsBar[k][1] < 0) newLam[i] += GROUND_PEN * dx[k][i];
                for (var j = 0; j < NX; j++) newLam[i] += Ab[k][j][i] * lam[j];
            }
            lam = newLam;
        }
        return Hd;
    }

    function scvxSolve(x0, usInit, xRefs, uRef, Qd, Rd, Qfd, p, nSCP, wTR) {
        var N = usInit.length;
        var us = [];
        for (var k = 0; k < N; k++) us.push(usInit[k].slice());

        for (var scp = 0; scp < nSCP; scp++) {
            /* 1. Nonlinear rollout */
            var xsBar = [x0.slice()];
            for (var k = 0; k < N; k++) xsBar.push(rk4(xsBar[k], us[k], p));

            /* 2. Linearise RK4 dynamics:  x_{k+1} ≈ A_k x_k + B_k u_k + c_k */
            var Ab = [], Bb = [], cb = [];
            for (var k = 0; k < N; k++) {
                var jac = rk4Jac(xsBar[k], us[k], p);
                Ab.push(jac.A); Bb.push(jac.B);
                var ck = new Array(NX);
                for (var i = 0; i < NX; i++) {
                    ck[i] = xsBar[k + 1][i];
                    for (var j = 0; j < NX; j++) ck[i] -= jac.A[i][j] * xsBar[k][j];
                    for (var j = 0; j < NU; j++) ck[i] -= jac.B[i][j] * us[k][j];
                }
                cb.push(ck);
            }

            var usBar = [];
            for (var k = 0; k < N; k++) usBar.push(us[k].slice());

            /* Diagonal preconditioner (backward-propagated Hessian approx)
               Captures cascading effect of u_k through the dynamics chain */
            var diagP = new Array(NX);
            for (var i = 0; i < NX; i++) diagP[i] = Qfd[i];
            if (xsBar[N][1] < 0) diagP[1] += GROUND_PEN;
            var prec = new Array(N);
            for (var k = N - 1; k >= 0; k--) {
                prec[k] = new Array(NU);
                for (var a = 0; a < NU; a++) {
                    prec[k][a] = Rd[a] + wTR;
                    for (var i = 0; i < NX; i++)
                        prec[k][a] += Bb[k][i][a] * Bb[k][i][a] * diagP[i];
                    if (prec[k][a] < 1e-6) prec[k][a] = 1e-6;
                }
                var newDP = new Array(NX);
                for (var i = 0; i < NX; i++) {
                    newDP[i] = Qd[i];
                    if (i === 1 && xsBar[k][1] < 0) newDP[i] += GROUND_PEN;
                    for (var j = 0; j < NX; j++)
                        newDP[i] += Ab[k][j][i] * Ab[k][j][i] * diagP[j];
                }
                diagP = newDP;
            }

            /* 3. Solve bounded QP via preconditioned projected gradient */
            for (var qi = 0; qi < SCVX_QP_ITERS; qi++) {
                var grad = qpGrad(us, Ab, Bb, cb, x0, xRefs, Qd, Rd, Qfd, uRef, wTR, usBar);

                /* Project gradient: zero components at active bounds */
                for (var k = 0; k < N; k++) {
                    if (us[k][0] <= p.Tmin + 0.01 && grad[k][0] > 0) grad[k][0] = 0;
                    if (us[k][0] >= p.Tmax - 0.01 && grad[k][0] < 0) grad[k][0] = 0;
                    if (us[k][1] <= -p.dmax + 1e-4 && grad[k][1] > 0) grad[k][1] = 0;
                    if (us[k][1] >= p.dmax - 1e-4 && grad[k][1] < 0) grad[k][1] = 0;
                }

                /* Preconditioned search direction: d = M^{-1} g */
                var pdir = new Array(N);
                for (var k = 0; k < N; k++)
                    pdir[k] = [grad[k][0] / prec[k][0], grad[k][1] / prec[k][1]];

                var Hpd = qpHvp(pdir, Ab, Bb, Qd, Rd, Qfd, wTR, xsBar);

                /* Exact step size: α = (g·d) / (d·Hd) */
                var gd = 0, dHd = 0;
                for (var k = 0; k < N; k++) {
                    for (var a = 0; a < NU; a++) {
                        gd += grad[k][a] * pdir[k][a];
                        dHd += pdir[k][a] * Hpd[k][a];
                    }
                }
                if (gd < 1e-8 || dHd < 1e-12) break;
                var alpha = gd / dHd;

                /* Update with projection to actuator bounds */
                for (var k = 0; k < N; k++) {
                    us[k][0] = clamp(us[k][0] - alpha * pdir[k][0], p.Tmin, p.Tmax);
                    us[k][1] = clamp(us[k][1] - alpha * pdir[k][1], -p.dmax, p.dmax);
                }
            }
        }

        /* Final nonlinear rollout */
        var xs = [x0.slice()];
        for (var k = 0; k < N; k++) xs.push(rk4(xs[k], us[k], p));
        return { xs: xs, us: us, cost: totalCost(xs, us, xRefs, uRef, Qd, Rd, Qfd) };
    }

    /* ═══ Canvas helpers ═══════════════════════════════════ */
    function setupCanvas(canvas) {
        var dpr = window.devicePixelRatio || 1;
        var w = canvas.clientWidth, h = canvas.clientHeight;
        canvas.width = w * dpr; canvas.height = h * dpr;
        var ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        return { ctx: ctx, w: w, h: h };
    }
    function niceStep(lo, hi, tgt) {
        var r = hi - lo; if (r <= 0) return 1;
        var rough = r / tgt;
        var mag = Math.pow(10, Math.floor(Math.log10(rough)));
        var n = rough / mag;
        if (n < 1.5) return mag; if (n < 3.5) return 2 * mag;
        if (n < 7.5) return 5 * mag; return 10 * mag;
    }

    /* ═══ Draw main rocket scene ═══════════════════════════ */
    function drawScene(canvas, sim) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0F1923'; ctx.fillRect(0, 0, W, H);

        var x = sim.x, padX = sim.padX;
        var px = x[0], py = x[1], vx = x[2], vy = x[3], th = x[4];

        /* Scene bounds */
        var xLo = Math.min(px, padX, 0) - 15;
        var xHi = Math.max(px, padX, 0) + 15;
        if (sim.predX) {
            for (var i = 0; i < sim.predX.length; i++) {
                if (sim.predX[i][0] - 5 < xLo) xLo = sim.predX[i][0] - 5;
                if (sim.predX[i][0] + 5 > xHi) xHi = sim.predX[i][0] + 5;
            }
        }
        for (var i = 0; i < sim.trail.length; i++) {
            if (sim.trail[i][0] - 5 < xLo) xLo = sim.trail[i][0] - 5;
            if (sim.trail[i][0] + 5 > xHi) xHi = sim.trail[i][0] + 5;
        }
        var xRange = Math.max(xHi - xLo, 30);
        var xC = (xLo + xHi) / 2; xLo = xC - xRange / 2; xHi = xC + xRange / 2;
        var pyMax = Math.max(py * 1.25, 20);
        for (var i = 0; i < sim.trail.length; i++)
            if (sim.trail[i][1] * 1.15 > pyMax) pyMax = sim.trail[i][1] * 1.15;

        var mt = 20, mr = 20, mb = 40, ml = 20;
        var sW = W - ml - mr, sH = H - mt - mb;
        function sx(v) { return ml + (v - xLo) / (xHi - xLo) * sW; }
        function sy(v) { return mt + (1 - v / pyMax) * sH; }

        /* Stars */
        ctx.fillStyle = 'rgba(255,255,255,0.3)';
        var seed = 12345;
        for (var i = 0; i < 60; i++) {
            seed = (seed * 1103515245 + 12345) & 0x7fffffff; var starX = seed % W;
            seed = (seed * 1103515245 + 12345) & 0x7fffffff; var starY = seed % (H * 0.6);
            seed = (seed * 1103515245 + 12345) & 0x7fffffff;
            ctx.beginPath(); ctx.arc(starX, starY, 0.5 + (seed % 100) / 100, 0, 2 * Math.PI); ctx.fill();
        }

        /* Ground */
        var groundY = sy(0);
        ctx.fillStyle = '#1a2633'; ctx.fillRect(0, groundY, W, H - groundY);
        ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 1;
        for (var gx = 0; gx < W; gx += 12) {
            ctx.beginPath(); ctx.moveTo(gx, groundY); ctx.lineTo(gx - 5, groundY + 8); ctx.stroke();
        }
        ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(0, groundY); ctx.lineTo(W, groundY); ctx.stroke();

        /* Landing pad */
        var pcx = sx(padX), pw = 40;
        ctx.fillStyle = '#2C3E50'; ctx.fillRect(pcx - pw / 2, groundY - 4, pw, 8);
        ctx.strokeStyle = '#e74c3c'; ctx.lineWidth = 2;
        ctx.strokeRect(pcx - pw / 2, groundY - 4, pw, 8);
        ctx.beginPath();
        ctx.moveTo(pcx - 6, groundY - 3); ctx.lineTo(pcx - 6, groundY + 3);
        ctx.moveTo(pcx - 6, groundY); ctx.lineTo(pcx + 6, groundY);
        ctx.moveTo(pcx + 6, groundY - 3); ctx.lineTo(pcx + 6, groundY + 3);
        ctx.stroke();

        /* Altitude scale */
        ctx.fillStyle = '#556677'; ctx.font = '9px "SF Mono","Fira Code",monospace'; ctx.textAlign = 'right';
        var altStep = niceStep(0, pyMax, 5);
        ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 1;
        for (var alt = altStep; alt < pyMax; alt += altStep) {
            var ay = sy(alt);
            ctx.beginPath(); ctx.moveTo(ml, ay); ctx.lineTo(W - mr, ay); ctx.stroke();
            ctx.fillText(alt.toFixed(0) + ' m', W - mr + 18, ay + 3);
        }

        /* Past trail */
        var trail = sim.trail;
        if (trail.length > 1) {
            for (var i = 1; i < trail.length; i++) {
                var alpha = 0.15 + 0.5 * (i / trail.length);
                ctx.strokeStyle = 'rgba(52,152,219,' + alpha.toFixed(2) + ')'; ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(sx(trail[i - 1][0]), sy(trail[i - 1][1]));
                ctx.lineTo(sx(trail[i][0]), sy(trail[i][1]));
                ctx.stroke();
            }
        }

        /* MPC predicted trajectory */
        if (sim.predX && sim.predX.length > 1 && sim.status === 'descending') {
            ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1.5;
            ctx.setLineDash([4, 3]);
            ctx.beginPath();
            ctx.moveTo(sx(sim.predX[0][0]), sy(Math.max(sim.predX[0][1], 0)));
            for (var i = 1; i < sim.predX.length; i++)
                ctx.lineTo(sx(sim.predX[i][0]), sy(Math.max(sim.predX[i][1], 0)));
            ctx.stroke(); ctx.setLineDash([]);
            ctx.fillStyle = 'rgba(255,255,255,0.15)';
            for (var i = 2; i < sim.predX.length; i += 3) {
                ctx.beginPath();
                ctx.arc(sx(sim.predX[i][0]), sy(Math.max(sim.predX[i][1], 0)), 2, 0, 2 * Math.PI);
                ctx.fill();
            }
            var last = sim.predX[sim.predX.length - 1];
            var lx = sx(last[0]), ly = sy(Math.max(last[1], 0));
            ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(lx - 4, ly - 4); ctx.lineTo(lx + 4, ly + 4);
            ctx.moveTo(lx + 4, ly - 4); ctx.lineTo(lx - 4, ly + 4);
            ctx.stroke();
        }

        /* ── Rocket ── */
        var rX = sx(px), rY = sy(Math.max(py, 0));
        var rH2 = 30, rW2 = 10;
        ctx.save(); ctx.translate(rX, rY); ctx.rotate(th);

        /* Flame */
        if (sim.u && sim.status === 'descending') {
            var Tcur = sim.u[0], dcur = sim.u[1];
            var fLen = (Tcur - sim.p.Tmin) / (sim.p.Tmax - sim.p.Tmin) * 25 + 5;
            var fW = rW2 * 0.6;
            ctx.save(); ctx.rotate(dcur);
            var fg = ctx.createLinearGradient(0, rH2 / 2, 0, rH2 / 2 + fLen);
            fg.addColorStop(0, 'rgba(255,165,0,0.9)');
            fg.addColorStop(0.4, 'rgba(255,100,0,0.7)');
            fg.addColorStop(1, 'rgba(255,50,0,0)');
            ctx.fillStyle = fg; ctx.beginPath();
            ctx.moveTo(-fW, rH2 / 2);
            ctx.lineTo(0, rH2 / 2 + fLen + Math.random() * 5);
            ctx.lineTo(fW, rH2 / 2); ctx.closePath(); ctx.fill();
            var ig = ctx.createLinearGradient(0, rH2 / 2, 0, rH2 / 2 + fLen * 0.5);
            ig.addColorStop(0, 'rgba(255,255,200,0.95)');
            ig.addColorStop(1, 'rgba(255,200,50,0)');
            ctx.fillStyle = ig; ctx.beginPath();
            ctx.moveTo(-fW * 0.4, rH2 / 2);
            ctx.lineTo(0, rH2 / 2 + fLen * 0.5 + Math.random() * 3);
            ctx.lineTo(fW * 0.4, rH2 / 2); ctx.closePath(); ctx.fill();
            var gr = fLen * 0.6;
            var glow = ctx.createRadialGradient(0, rH2 / 2 + 5, 0, 0, rH2 / 2 + 5, gr);
            glow.addColorStop(0, 'rgba(255,150,50,0.15)');
            glow.addColorStop(1, 'rgba(255,100,0,0)');
            ctx.fillStyle = glow;
            ctx.beginPath(); ctx.arc(0, rH2 / 2 + 5, gr, 0, 2 * Math.PI); ctx.fill();
            ctx.restore();
        }

        /* Body */
        ctx.fillStyle = '#D4E0ED';
        ctx.fillRect(-rW2 / 2, -rH2 / 2, rW2, rH2);
        ctx.fillStyle = '#2C3E50';
        ctx.fillRect(-rW2 / 2, -rH2 / 2 + rH2 * 0.3, rW2, 3);
        ctx.fillRect(-rW2 / 2, -rH2 / 2 + rH2 * 0.6, rW2, 2);
        ctx.fillStyle = '#e74c3c'; ctx.beginPath();
        ctx.moveTo(-rW2 / 2, -rH2 / 2); ctx.lineTo(0, -rH2 / 2 - 10);
        ctx.lineTo(rW2 / 2, -rH2 / 2); ctx.closePath(); ctx.fill();
        ctx.fillStyle = '#2C3E50';
        ctx.beginPath();
        ctx.moveTo(-rW2 / 2, rH2 / 2 - 5); ctx.lineTo(-rW2 / 2 - 6, rH2 / 2 + 3);
        ctx.lineTo(-rW2 / 2, rH2 / 2); ctx.closePath(); ctx.fill();
        ctx.beginPath();
        ctx.moveTo(rW2 / 2, rH2 / 2 - 5); ctx.lineTo(rW2 / 2 + 6, rH2 / 2 + 3);
        ctx.lineTo(rW2 / 2, rH2 / 2); ctx.closePath(); ctx.fill();
        ctx.fillStyle = '#555'; ctx.fillRect(-3, rH2 / 2, 6, 3);
        ctx.restore();

        /* Velocity vector */
        var vScale = 1.5, vLen = Math.sqrt(vx * vx + vy * vy);
        if (vLen > 0.5) {
            var vxP = vx * vScale, vyP = -vy * vScale;
            ctx.strokeStyle = 'rgba(46,204,113,0.7)'; ctx.lineWidth = 1.5;
            ctx.beginPath(); ctx.moveTo(rX, rY); ctx.lineTo(rX + vxP, rY + vyP); ctx.stroke();
            var vAng = Math.atan2(vyP, vxP);
            ctx.beginPath();
            ctx.moveTo(rX + vxP, rY + vyP);
            ctx.lineTo(rX + vxP - 5 * Math.cos(vAng - 0.4), rY + vyP - 5 * Math.sin(vAng - 0.4));
            ctx.moveTo(rX + vxP, rY + vyP);
            ctx.lineTo(rX + vxP - 5 * Math.cos(vAng + 0.4), rY + vyP - 5 * Math.sin(vAng + 0.4));
            ctx.stroke();
        }

        /* Telemetry overlay */
        ctx.fillStyle = 'rgba(15,25,35,0.8)'; ctx.fillRect(6, 6, 135, 82);
        ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 1; ctx.strokeRect(6, 6, 135, 82);
        ctx.font = '9px "SF Mono","Fira Code",monospace'; ctx.textAlign = 'left';
        var tY = 18, tX = 12, tD = 13;
        function tRow(lbl, val) {
            ctx.fillStyle = '#556677'; ctx.fillText(lbl, tX, tY);
            ctx.fillStyle = '#D4E0ED'; ctx.fillText(val, tX + 42, tY); tY += tD;
        }
        tRow('ALT', Math.max(py, 0).toFixed(1) + ' m');
        tRow('VEL', vLen.toFixed(1) + ' m/s');
        tRow('V_x', vx.toFixed(2) + ' m/s');
        tRow('V_y', vy.toFixed(2) + ' m/s');
        tRow('ATT', (th * 180 / Math.PI).toFixed(1) + '\u00B0');
        if (sim.u) {
            ctx.fillStyle = '#556677'; ctx.fillText('THR', tX, tY);
            ctx.fillStyle = '#f39c12'; ctx.fillText(sim.u[0].toFixed(0) + ' N', tX + 42, tY);
        }
        ctx.fillStyle = '#99AABB'; ctx.font = '10px "Source Sans 3",sans-serif'; ctx.textAlign = 'center';
        ctx.fillText('t = ' + sim.time.toFixed(1) + ' s', W / 2, H - 8);

        var lgX = 10, lgY = H - 30;
        ctx.font = '9px "Source Sans 3",sans-serif'; ctx.textAlign = 'left';
        ctx.strokeStyle = 'rgba(52,152,219,0.5)'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(lgX, lgY); ctx.lineTo(lgX + 14, lgY); ctx.stroke();
        ctx.fillStyle = '#778899'; ctx.fillText('Trail', lgX + 18, lgY + 3);
        ctx.strokeStyle = 'rgba(255,255,255,0.2)'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
        ctx.beginPath(); ctx.moveTo(lgX + 56, lgY); ctx.lineTo(lgX + 70, lgY); ctx.stroke(); ctx.setLineDash([]);
        ctx.fillText('MPC Plan', lgX + 74, lgY + 3);
        ctx.strokeStyle = 'rgba(46,204,113,0.7)'; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(lgX + 140, lgY); ctx.lineTo(lgX + 154, lgY); ctx.stroke();
        ctx.fillText('Velocity', lgX + 158, lgY + 3);
    }

    /* ═══ Strip chart plots ════════════════════════════════ */
    function drawPlots(canvas, sim) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0A1219'; ctx.fillRect(0, 0, W, H);
        if (sim.log.t.length < 2) {
            ctx.fillStyle = '#556677'; ctx.font = '12px "Source Sans 3",sans-serif';
            ctx.textAlign = 'center'; ctx.fillText('Press Start to begin', W / 2, H / 2);
            return;
        }
        var mlp = 46, mrp = 10, gap = 8, mtp = 4, mbp = 4;
        var pC = 3, pH = (H - (pC - 1) * gap - mtp - mbp) / pC;
        drawSub(ctx, mlp, mtp, W - mlp - mrp, pH, sim.log.t, sim.log.alt, 'Altitude (m)', '#3498db', null);
        drawSub(ctx, mlp, mtp + pH + gap, W - mlp - mrp, pH, sim.log.t, sim.log.thr, 'Thrust (N)', '#f39c12', [sim.p.Tmin, sim.p.Tmax]);
        drawSub(ctx, mlp, mtp + 2 * (pH + gap), W - mlp - mrp, pH, sim.log.t, sim.log.tilt, 'Tilt (\u00B0)', '#e74c3c', null);
    }
    function drawSub(ctx, ox, oy, pw, ph, xs, ys, label, color, bounds) {
        var pt = 16, pb = 18, plotH = ph - pt - pb;
        var yMin = Infinity, yMax = -Infinity;
        for (var i = 0; i < ys.length; i++) { if (ys[i] < yMin) yMin = ys[i]; if (ys[i] > yMax) yMax = ys[i]; }
        if (bounds) { if (bounds[0] < yMin) yMin = bounds[0]; if (bounds[1] > yMax) yMax = bounds[1]; }
        var yPad = Math.max((yMax - yMin) * 0.12, 0.5); yMin -= yPad; yMax += yPad;
        var xMin = 0, xMax = Math.max(xs[xs.length - 1], 1);
        function sx(v) { return ox + (v - xMin) / (xMax - xMin) * pw; }
        function sy(v) { return oy + pt + (1 - (v - yMin) / (yMax - yMin)) * plotH; }
        ctx.strokeStyle = 'rgba(255,255,255,0.05)'; ctx.lineWidth = 1;
        var ystep = niceStep(yMin, yMax, 3);
        ctx.fillStyle = '#556677'; ctx.font = '8px "SF Mono","Fira Code",monospace'; ctx.textAlign = 'right';
        for (var v = Math.ceil(yMin / ystep) * ystep; v <= yMax; v += ystep) {
            var py2 = sy(v);
            ctx.beginPath(); ctx.moveTo(ox, py2); ctx.lineTo(ox + pw, py2); ctx.stroke();
            ctx.fillText(v.toFixed(v >= 100 ? 0 : 1), ox - 4, py2 + 3);
        }
        ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(ox, oy + pt); ctx.lineTo(ox, oy + pt + plotH); ctx.lineTo(ox + pw, oy + pt + plotH); ctx.stroke();
        if (bounds) {
            ctx.setLineDash([4, 3]); ctx.strokeStyle = 'rgba(231,76,60,0.4)'; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(ox, sy(bounds[0])); ctx.lineTo(ox + pw, sy(bounds[0])); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(ox, sy(bounds[1])); ctx.lineTo(ox + pw, sy(bounds[1])); ctx.stroke();
            ctx.setLineDash([]);
        }
        if (ys.length > 1) {
            ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.beginPath();
            for (var i = 0; i < ys.length; i++) { if (i === 0) ctx.moveTo(sx(xs[i]), sy(ys[i])); else ctx.lineTo(sx(xs[i]), sy(ys[i])); }
            ctx.stroke();
            ctx.fillStyle = color; ctx.beginPath();
            ctx.arc(sx(xs[ys.length - 1]), sy(ys[ys.length - 1]), 3.5, 0, 2 * Math.PI); ctx.fill();
        }
        ctx.fillStyle = '#99AABB'; ctx.font = '10px "Source Sans 3",sans-serif'; ctx.textAlign = 'left';
        ctx.fillText(label, ox + 4, oy + pt - 4);
    }

    /* ═══ CSS ══════════════════════════════════════════════ */
    function injectCSS() {
        if (document.getElementById('rocket-demo-styles')) return;
        var s = document.createElement('style'); s.id = 'rocket-demo-styles';
        s.textContent =
            '.rocket-demo{margin:1.5rem 0;border:2px solid #0D7C66;border-radius:6px;background:#0F1923;overflow:hidden;font-family:"Source Sans 3",sans-serif}' +
            '.rkt-header{background:#0D7C66;padding:0.45rem 1rem}' +
            '.rkt-title{color:#fff;font-size:0.78rem;font-weight:700;text-transform:uppercase;letter-spacing:0.6px}' +
            '.rkt-ctrls{display:flex;flex-wrap:wrap;align-items:center;gap:0.6rem 1rem;padding:0.45rem 1rem;border-bottom:1px solid rgba(255,255,255,0.06)}' +
            '.rkt-group-label{color:#556677;font-size:0.68rem;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;margin-right:0.2rem}' +
            '.rkt-sl{display:flex;align-items:center;gap:0.3rem;color:#8899AA;font-size:0.75rem;white-space:nowrap}' +
            '.rkt-slider{width:70px;accent-color:#0D7C66;cursor:pointer}' +
            '.rkt-sl-val{color:#D4E0ED;font-family:"SF Mono","Fira Code",monospace;font-size:0.75rem;font-weight:600;min-width:2rem;text-align:right}' +
            '.rkt-btn{padding:0.3rem 0.8rem;border:1px solid #0D7C66;border-radius:4px;background:transparent;color:#0D7C66;font-family:"Source Sans 3",sans-serif;font-size:0.72rem;font-weight:700;text-transform:uppercase;letter-spacing:0.4px;cursor:pointer;transition:background 0.15s,color 0.15s}' +
            '.rkt-btn:hover{background:#0D7C66;color:#fff}' +
            '.rkt-start-btn{background:#0D7C66;color:#fff}.rkt-start-btn:hover{background:#0A6B58}' +
            '.rkt-reset-btn{background:transparent;color:#8899AA;border:1px solid rgba(255,255,255,0.12)}.rkt-reset-btn:hover{color:#D4E0ED;border-color:rgba(255,255,255,0.25)}' +
            '.rkt-status{color:#556677;font-size:0.75rem;font-family:"SF Mono","Fira Code",monospace;margin-left:0.3rem}' +
            '.rkt-status.rkt-landed{color:#2ecc71}.rkt-status.rkt-crashed{color:#e74c3c}' +
            '.rkt-body{display:flex;min-height:380px}' +
            '.rkt-scene-wrap{flex:1;min-width:0}.rkt-scene-wrap canvas{width:100%;height:100%;min-height:380px;display:block}' +
            '.rkt-plot-wrap{flex:0 0 280px;border-left:1px solid rgba(255,255,255,0.06)}.rkt-plot-wrap canvas{width:100%;height:100%;min-height:380px;display:block}' +
            '.rkt-readout{padding:0.5rem 1rem;border-top:1px solid rgba(255,255,255,0.06);color:#8899AA;font-size:0.75rem;font-family:"SF Mono","Fira Code",monospace;display:flex;flex-wrap:wrap;gap:1.2rem}' +
            '.rkt-readout span span{color:#D4E0ED}' +
            '@media(max-width:720px){.rkt-body{flex-direction:column}.rkt-plot-wrap{flex:none;border-left:none;border-top:1px solid rgba(255,255,255,0.06);min-height:240px}}';
        document.head.appendChild(s);
    }

    function slH(id, label, min, max, step, val, unit) {
        return '<label class="rkt-sl">' + label +
            '<input type="range" class="rkt-slider" data-id="' + id +
            '" min="' + min + '" max="' + max + '" step="' + step + '" value="' + val + '">' +
            '<span class="rkt-sl-val" data-for="' + id + '">' + val + (unit || '') + '</span></label>';
    }

    /* ═══ Widget builder ═══════════════════════════════════ */
    function initDemo(container) {
        if (container.dataset.initialized) return;
        container.dataset.initialized = '1';
        injectCSS();

        var P = { m: 30, g: 9.81, J: 10, l: 1.5, dt: 0.1,
                  Tmin: 40, Tmax: 400, dmax: 0.35 };

        container.innerHTML =
            '<div class="rkt-header"><span class="rkt-title">\u25B6 Rocket Powered Descent \u2014 SCvx MPC</span></div>' +
            '<div class="rkt-ctrls"><span class="rkt-group-label">Initial</span>' +
                slH('y0', 'Alt', 15, 80, 1, 50, 'm') +
                slH('x0', 'x\u2080', -25, 25, 1, 10, 'm') +
                slH('vx0', 'V\u2093', -10, 5, 0.5, -2, '') +
                slH('vy0', 'V\u1D67', -15, 0, 0.5, -8, '') +
                slH('th0', '\u03B8\u2080', -25, 25, 1, 14, '\u00B0') +
                slH('padx', 'Pad', -15, 15, 1, 0, 'm') +
            '</div>' +
            '<div class="rkt-ctrls"><span class="rkt-group-label">MPC</span>' +
                slH('qpos', 'Q_pos', 1, 50, 1, 15, '') +
                slH('qatt', 'Q_\u03B8', 5, 200, 5, 50, '') +
                slH('rcost', 'R', 1, 100, 1, 15, '') +
                slH('qf', 'Q_f', 20, 300, 10, 200, '\u00D7') +
                slH('hor', 'N', 10, 40, 1, 36, '') +
                slH('spd', 'Speed', 0.2, 3, 0.1, 1.0, '\u00D7') +
            '</div>' +
            '<div class="rkt-ctrls">' +
                '<button class="rkt-btn rkt-start-btn">\u25B6 Start</button>' +
                '<button class="rkt-btn rkt-reset-btn">\u21BA Reset</button>' +
                '<span class="rkt-status">READY</span>' +
            '</div>' +
            '<div class="rkt-body">' +
                '<div class="rkt-scene-wrap"><canvas class="rkt-scene"></canvas></div>' +
                '<div class="rkt-plot-wrap"><canvas class="rkt-plots"></canvas></div>' +
            '</div>' +
            '<div class="rkt-readout">' +
                '<span>t = <span data-ro="t">0.0</span> s</span>' +
                '<span>alt = <span data-ro="alt">50.0</span> m</span>' +
                '<span>|v| = <span data-ro="vel">8.2</span> m/s</span>' +
                '<span>T = <span data-ro="thr">\u2014</span> N</span>' +
                '<span>\u03B8 = <span data-ro="att">14.0</span>\u00B0</span>' +
            '</div>';

        var sceneCvs = container.querySelector('.rkt-scene');
        var plotCvs = container.querySelector('.rkt-plots');
        var startBtn = container.querySelector('.rkt-start-btn');
        var resetBtn = container.querySelector('.rkt-reset-btn');
        var statusEl = container.querySelector('.rkt-status');

        function getVal(id) { return parseFloat(container.querySelector('[data-id="' + id + '"]').value); }
        function setRO(id, v) { var e = container.querySelector('[data-ro="' + id + '"]'); if (e) e.textContent = v; }

        container.addEventListener('input', function (e) {
            var el = e.target; if (!el.classList.contains('rkt-slider')) return;
            var id = el.dataset.id, val = parseFloat(el.value);
            var span = container.querySelector('[data-for="' + id + '"]'); if (!span) return;
            var u = '';
            if (id === 'y0' || id === 'x0' || id === 'padx') u = 'm';
            else if (id === 'th0') u = '\u00B0';
            else if (id === 'qf') u = '\u00D7';
            else if (id === 'spd') u = '\u00D7';
            span.textContent = ((id === 'spd' || id === 'vx0' || id === 'vy0') ? val.toFixed(1) : '' + val) + u;
        });

        /* Simulation state */
        var sim = {
            x: null, u: null, time: 0, status: 'ready',
            trail: [], predX: null, padX: 0,
            log: { t: [], alt: [], thr: [], tilt: [] },
            p: P, nomUs: null
        };
        var running = false, animId = null, lastFrameTime = null, accum = 0;
        var simN = 25;

        function readInitState() {
            return [getVal('x0'), getVal('y0'), getVal('vx0'), getVal('vy0'),
                    getVal('th0') * Math.PI / 180, 0];
        }

        function readMPCWeights() {
            var qp = getVal('qpos'), qa = getVal('qatt'), rv = getVal('rcost'), qfm = getVal('qf');
            return {
                Qd: [qp, qp, qp * 0.5, qp * 0.5, qa, qa * 0.2],
                Rd: [rv * 0.0001, rv * 0.1],
                Qfd: [qp * qfm, qp * qfm, qp * 0.5 * qfm, qp * 0.5 * qfm, qa * qfm, qa * 0.2 * qfm]
            };
        }

        function updateReadout() {
            var x = sim.x, vLen = Math.sqrt(x[2] * x[2] + x[3] * x[3]);
            setRO('t', sim.time.toFixed(1));
            setRO('alt', Math.max(x[1], 0).toFixed(1));
            setRO('vel', vLen.toFixed(1));
            setRO('att', (x[4] * 180 / Math.PI).toFixed(1));
            setRO('thr', sim.u ? sim.u[0].toFixed(0) : '\u2014');
        }

        function render() {
            drawScene(sceneCvs, sim);
            drawPlots(plotCvs, sim);
            updateReadout();
        }

        /* ── Reset ── */
        function doReset() {
            running = false;
            if (animId) { cancelAnimationFrame(animId); animId = null; }
            lastFrameTime = null; accum = 0;
            simN = Math.round(getVal('hor'));
            var x0 = readInitState();
            sim.x = x0.slice(); sim.u = null; sim.time = 0; sim.status = 'ready';
            sim.trail = [x0.slice()]; sim.predX = null; sim.padX = getVal('padx');
            sim.log = { t: [0], alt: [x0[1]], thr: [], tilt: [x0[4] * 180 / Math.PI] };
            sim.nomUs = null;
            startBtn.textContent = '\u25B6 Start';
            statusEl.textContent = 'READY'; statusEl.className = 'rkt-status';
            render();
        }

        /* ── Physics step ── */
        function physicsStep() {
            var padX = getVal('padx');
            sim.padX = padX;
            var xf = [padX, 0, 0, 0, 0, 0];
            var uRef = [P.m * P.g, 0];
            var w = readMPCWeights();
            var N = simN;

            /* Reference trajectory from current state to target */
            var xRefs = genReference(sim.x, xf, N, P.dt);

            /* Initial control guess: shift previous or hover */
            var usInit = [];
            if (sim.nomUs && sim.nomUs.length === N) {
                for (var k = 0; k < N - 1; k++) usInit.push(sim.nomUs[k + 1].slice());
                usInit.push(sim.nomUs[N - 1].slice());
            } else {
                for (var k = 0; k < N; k++) usInit.push([P.m * P.g, 0]);
            }

            /* Run SCvx (3 SCP iterations, each solving a bounded QP) */
            var result = scvxSolve(sim.x, usInit, xRefs, uRef, w.Qd, w.Rd, w.Qfd, P, SCVX_SCP_ITERS, SCVX_TR);

            sim.nomUs = result.us;
            sim.predX = result.xs;

            var u0 = result.us[0].slice();
            if (!isFinite(u0[0]) || !isFinite(u0[1])) { u0 = [P.m * P.g, 0]; }
            sim.u = u0;

            /* Advance true plant with RK4 */
            sim.x = rk4(sim.x, u0, P);
            sim.time += P.dt;

            sim.trail.push(sim.x.slice());
            sim.log.t.push(sim.time);
            sim.log.alt.push(Math.max(sim.x[1], 0));
            sim.log.thr.push(u0[0]);
            sim.log.tilt.push(sim.x[4] * 180 / Math.PI);

            /* Landing / crash check */
            if (sim.x[1] <= 0.3) {
                if (Math.abs(sim.x[3]) < 3 && Math.abs(sim.x[2]) < 3 && Math.abs(sim.x[4]) < 0.4) {
                    sim.status = 'landed'; sim.x[1] = 0; sim.x[3] = 0; sim.x[5] = 0;
                    statusEl.textContent = 'LANDED'; statusEl.className = 'rkt-status rkt-landed';
                } else {
                    sim.status = 'crashed'; sim.x[1] = 0;
                    statusEl.textContent = 'CRASHED'; statusEl.className = 'rkt-status rkt-crashed';
                }
                running = false; startBtn.textContent = '\u25B6 Start'; return;
            }
            if (sim.x[1] < -1 || Math.abs(sim.x[0]) > 100 || sim.x[1] > 200 || sim.time > 20) {
                sim.status = 'crashed';
                statusEl.textContent = 'CRASHED'; statusEl.className = 'rkt-status rkt-crashed';
                running = false; startBtn.textContent = '\u25B6 Start';
            }
        }

        /* ── Animation loop ── */
        function animLoop(ts) {
            if (!running) return;
            if (!lastFrameTime) lastFrameTime = ts;
            var frameDt = (ts - lastFrameTime) / 1000 * getVal('spd');
            lastFrameTime = ts;
            accum += frameDt;
            var steps = 0;
            while (accum >= P.dt && steps < 3 && running) {
                physicsStep();
                accum -= P.dt;
                steps++;
            }
            render();
            if (running) animId = requestAnimationFrame(animLoop);
        }

        function doStart() {
            if (sim.status === 'landed' || sim.status === 'crashed') { doReset(); return; }
            if (running) {
                running = false; if (animId) cancelAnimationFrame(animId);
                startBtn.textContent = '\u25B6 Start'; statusEl.textContent = 'PAUSED'; return;
            }
            running = true; sim.status = 'descending';
            startBtn.textContent = '\u23F8 Pause'; statusEl.textContent = 'DESCENDING'; statusEl.className = 'rkt-status';
            lastFrameTime = null; accum = 0;
            animId = requestAnimationFrame(animLoop);
        }

        /* Canvas click */
        sceneCvs.addEventListener('click', function (e) {
            if (running) return;
            var rect = sceneCvs.getBoundingClientRect();
            var cx = e.clientX - rect.left, cy = e.clientY - rect.top;
            var w = rect.width, h = rect.height;
            var x = sim.x, padXv = getVal('padx');
            var xLo = Math.min(x[0], padXv, 0) - 15, xHi = Math.max(x[0], padXv, 0) + 15;
            var xRange = Math.max(xHi - xLo, 30), xC = (xLo + xHi) / 2;
            xLo = xC - xRange / 2; xHi = xC + xRange / 2;
            var pyMax = Math.max(x[1] * 1.25, 20);
            var mt2 = 20, mr2 = 20, mb2 = 40, ml2 = 20;
            var sW2 = w - ml2 - mr2, sH2 = h - mt2 - mb2;
            var wx = xLo + (cx - ml2) / sW2 * (xHi - xLo);
            var wy = Math.max((1 - (cy - mt2) / sH2) * pyMax, 5);
            if (e.shiftKey) {
                var ps = container.querySelector('[data-id="padx"]');
                ps.value = clamp(Math.round(wx), -15, 15);
                ps.dispatchEvent(new Event('input', { bubbles: true }));
            } else {
                var xs2 = container.querySelector('[data-id="x0"]');
                var ys2 = container.querySelector('[data-id="y0"]');
                xs2.value = clamp(Math.round(wx), -25, 25);
                ys2.value = clamp(Math.round(wy), 15, 80);
                xs2.dispatchEvent(new Event('input', { bubbles: true }));
                ys2.dispatchEvent(new Event('input', { bubbles: true }));
            }
            doReset();
        });

        startBtn.addEventListener('click', doStart);
        resetBtn.addEventListener('click', doReset);
        doReset();

        if (window.ResizeObserver) {
            var ro = new ResizeObserver(function () { render(); });
            ro.observe(sceneCvs); ro.observe(plotCvs);
        }
    }

    /* ═══ Public API ═══════════════════════════════════════ */
    function initAllDemos(root) {
        var els = (root || document).querySelectorAll('.rocket-demo');
        for (var i = 0; i < els.length; i++) initDemo(els[i]);
    }
    if (document.readyState === 'loading')
        document.addEventListener('DOMContentLoaded', function () { initAllDemos(); });
    else initAllDemos();
    window.RocketDemo = { initAll: initAllDemos, init: initDemo };
})();
