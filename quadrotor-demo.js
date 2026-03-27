/**
 * Interactive 3D Quadrotor NMPC Demo — SCvx (Successive Convexification)
 * Nonlinear quadrotor dynamics with RK4 integration.
 * Adjoint gradient + HVP for exact QP step sizes.
 * Features: zoom, realistic quadrotor, wind disturbance, EKF estimation.
 *
 * HTML hook:  <div class="quadrotor-demo"></div>
 */
(function () {
    'use strict';

    var NX = 6, NU = 3, NXA = 9, NZ = 6;
    // State: [px, py, pz, vx, vy, vz]
    // Input: [roll(phi), pitch(theta), thrust(T)]
    // Augmented: [px, py, pz, vx, vy, vz, dx, dy, dz]  (dx,dy,dz = disturbance forces)

    function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

    /* ═══ Continuous-time quadrotor dynamics ═══════════════════
       vdot_x =  (T/m) cos(phi) sin(theta) + dx/m
       vdot_y = -(T/m) sin(phi)            + dy/m
       vdot_z =  (T/m) cos(phi) cos(theta) - g + dz/m       */
    function fCont(x, u, p, dist) {
        var phi = u[0], theta = u[1], T = u[2];
        var cp = Math.cos(phi), sp = Math.sin(phi);
        var ct = Math.cos(theta), st = Math.sin(theta);
        var Tm = T / p.m;
        var ddx = dist ? dist[0] / p.m : 0;
        var ddy = dist ? dist[1] / p.m : 0;
        var ddz = dist ? dist[2] / p.m : 0;
        return [
            x[3], x[4], x[5],
            Tm * cp * st + ddx,
            -Tm * sp + ddy,
            Tm * cp * ct - p.g + ddz
        ];
    }

    /* ═══ RK4 integrator ═══════════════════════════════════════ */
    function rk4(x, u, p, dist) {
        var h = p.dt;
        var k1 = fCont(x, u, p, dist);
        var xm = new Array(NX);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + 0.5 * h * k1[i];
        var k2 = fCont(xm, u, p, dist);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + 0.5 * h * k2[i];
        var k3 = fCont(xm, u, p, dist);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + h * k3[i];
        var k4 = fCont(xm, u, p, dist);
        var xn = new Array(NX);
        for (var i = 0; i < NX; i++)
            xn[i] = x[i] + (h / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
        return xn;
    }

    /* ═══ Numerical Jacobians of the RK4 map ══════════════════ */
    function rk4Jac(x, u, p, dist) {
        var eps = 1e-6, f0 = rk4(x, u, p, dist || null);
        var A = [], B = [];
        for (var i = 0; i < NX; i++) { A.push(new Array(NX)); B.push(new Array(NU)); }
        for (var j = 0; j < NX; j++) {
            var xp = x.slice(); xp[j] += eps;
            var fp = rk4(xp, u, p, dist || null);
            for (var i = 0; i < NX; i++) A[i][j] = (fp[i] - f0[i]) / eps;
        }
        for (var j = 0; j < NU; j++) {
            var up = u.slice(); up[j] += eps;
            var fp = rk4(x, up, p, dist || null);
            for (var i = 0; i < NX; i++) B[i][j] = (fp[i] - f0[i]) / eps;
        }
        return { A: A, B: B };
    }

    /* ═══ EKF: Augmented Jacobian (9x9) ══════════════════════ */
    function augJacobian(x, u, p, dist) {
        var eps = 1e-6;
        var f0 = rk4(x, u, p, dist);
        var Fa = [];
        for (var i = 0; i < NXA; i++) { Fa.push(new Array(NXA)); for (var j = 0; j < NXA; j++) Fa[i][j] = 0; }
        // df/dx (top-left 6x6)
        for (var j = 0; j < NX; j++) {
            var xp = x.slice(); xp[j] += eps;
            var fp = rk4(xp, u, p, dist);
            for (var i = 0; i < NX; i++) Fa[i][j] = (fp[i] - f0[i]) / eps;
        }
        // df/dd (top-right 6x3)
        for (var j = 0; j < 3; j++) {
            var dp = dist ? dist.slice() : [0,0,0]; dp[j] += eps;
            var fp = rk4(x, u, p, dp);
            for (var i = 0; i < NX; i++) Fa[i][NX + j] = (fp[i] - f0[i]) / eps;
        }
        // Bottom-right 3x3: identity (random walk)
        for (var i = 0; i < 3; i++) Fa[NX + i][NX + i] = 1.0;
        return Fa;
    }

    /* ═══ EKF: Matrix utilities for 9x9 ═════════════════════ */
    function matMul(A, B, n) {
        var C = [];
        for (var i = 0; i < n; i++) {
            C.push(new Array(n));
            for (var j = 0; j < n; j++) {
                var s = 0;
                for (var k = 0; k < n; k++) s += A[i][k] * B[k][j];
                C[i][j] = s;
            }
        }
        return C;
    }
    function matTrans(A, n) {
        var T = [];
        for (var i = 0; i < n; i++) {
            T.push(new Array(n));
            for (var j = 0; j < n; j++) T[i][j] = A[j][i];
        }
        return T;
    }
    function matAdd(A, B, n) {
        var C = [];
        for (var i = 0; i < n; i++) {
            C.push(new Array(n));
            for (var j = 0; j < n; j++) C[i][j] = A[i][j] + B[i][j];
        }
        return C;
    }
    function matSub(A, B, n) {
        var C = [];
        for (var i = 0; i < n; i++) {
            C.push(new Array(n));
            for (var j = 0; j < n; j++) C[i][j] = A[i][j] - B[i][j];
        }
        return C;
    }
    function diagMat(diag, n) {
        var M = [];
        for (var i = 0; i < n; i++) { M.push(new Array(n)); for (var j = 0; j < n; j++) M[i][j] = (i===j) ? diag[i] : 0; }
        return M;
    }
    function eyeMat(n) {
        var M = [];
        for (var i = 0; i < n; i++) { M.push(new Array(n)); for (var j = 0; j < n; j++) M[i][j] = (i===j) ? 1 : 0; }
        return M;
    }
    /* Gauss-Jordan inversion for small matrices */
    function matInv(M, n) {
        var A = [];
        for (var i = 0; i < n; i++) { A.push([]); for (var j = 0; j < 2*n; j++) A[i].push(j < n ? M[i][j] : (j-n===i ? 1 : 0)); }
        for (var c = 0; c < n; c++) {
            var best = c;
            for (var r = c+1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[best][c])) best = r;
            var tmp = A[c]; A[c] = A[best]; A[best] = tmp;
            var piv = A[c][c];
            if (Math.abs(piv) < 1e-14) piv = 1e-14;
            for (var j = 0; j < 2*n; j++) A[c][j] /= piv;
            for (var r = 0; r < n; r++) {
                if (r === c) continue;
                var f = A[r][c];
                for (var j = 0; j < 2*n; j++) A[r][j] -= f * A[c][j];
            }
        }
        var R = [];
        for (var i = 0; i < n; i++) { R.push([]); for (var j = 0; j < n; j++) R[i].push(A[i][j+n]); }
        return R;
    }
    function symmetrise(M, n) {
        for (var i = 0; i < n; i++) for (var j = i+1; j < n; j++) {
            var avg = 0.5*(M[i][j]+M[j][i]); M[i][j] = avg; M[j][i] = avg;
        }
        return M;
    }

    /* ═══ EKF Predict & Update ══════════════════════════════ */
    function ekfPredict(xa, Pa, u, p) {
        var x = xa.slice(0, NX);
        var dist = xa.slice(NX, NXA);
        var xNext = rk4(x, u, p, dist);
        var xaPred = xNext.concat(dist);
        var Fa = augJacobian(x, u, p, dist);
        var FaPaFt = matMul(matMul(Fa, Pa, NXA), matTrans(Fa, NXA), NXA);
        var PaPred = symmetrise(matAdd(FaPaFt, p.Qa, NXA), NXA);
        return { xa: xaPred, Pa: PaPred };
    }

    function ekfUpdate(xaPred, PaPred, z, p) {
        // H = [I_6 | 0_{6x3}], innovation y = z - xaPred[0:6]
        var y = new Array(NZ);
        for (var i = 0; i < NZ; i++) y[i] = z[i] - xaPred[i];
        // S = H*P*H^T + R  →  P[0:6,0:6] + R
        var S = [];
        for (var i = 0; i < NZ; i++) {
            S.push(new Array(NZ));
            for (var j = 0; j < NZ; j++) S[i][j] = PaPred[i][j] + (i===j ? p.Ra[i] : 0);
        }
        var Sinv = matInv(S, NZ);
        // K = P*H^T * S^{-1}   →  P[:,0:6] * Sinv   (9x6 * 6x6 = 9x6)
        var K = [];
        for (var i = 0; i < NXA; i++) {
            K.push(new Array(NZ));
            for (var j = 0; j < NZ; j++) {
                var s = 0;
                for (var l = 0; l < NZ; l++) s += PaPred[i][l] * Sinv[l][j];
                K[i][j] = s;
            }
        }
        // xa = xaPred + K*y
        var xaUpd = xaPred.slice();
        for (var i = 0; i < NXA; i++) {
            for (var j = 0; j < NZ; j++) xaUpd[i] += K[i][j] * y[j];
        }
        // P = (I - K*H)*P  (Joseph form for stability: (I-KH)*P*(I-KH)^T + K*R*K^T)
        // KH is 9x9: K[:,j] for j<6 forms columns 0-5, columns 6-8 are zero
        var IKH = eyeMat(NXA);
        for (var i = 0; i < NXA; i++) for (var j = 0; j < NZ; j++) IKH[i][j] -= K[i][j];
        var PaUpd = matMul(matMul(IKH, PaPred, NXA), matTrans(IKH, NXA), NXA);
        // + K*R*K^T
        for (var i = 0; i < NXA; i++) for (var j = 0; j < NXA; j++) {
            for (var l = 0; l < NZ; l++) PaUpd[i][j] += K[i][l] * p.Ra[l] * K[j][l];
        }
        symmetrise(PaUpd, NXA);
        return { xa: xaUpd, Pa: PaUpd };
    }

    /* ═══ Cost and solver constants ════════════════════════════ */
    var GROUND_PEN = 3000;
    var SCVX_QP_ITERS = 20;
    var SCVX_SCP_ITERS = 5;
    var SCVX_TR = 1.5;
    var PROP_SPIN = 20; // rad/s visual spin

    function totalCost(xs, us, xRefs, uRef, Qd, Rd, Qfd) {
        var N = us.length, cost = 0;
        for (var k = 0; k < N; k++) {
            for (var i = 0; i < NX; i++) { var dx = xs[k][i] - xRefs[k][i]; cost += 0.5 * Qd[i] * dx * dx; }
            for (var i = 0; i < NU; i++) { var du = us[k][i] - uRef[i]; cost += 0.5 * Rd[i] * du * du; }
            if (xs[k][2] < 0) cost += 0.5 * GROUND_PEN * xs[k][2] * xs[k][2];
        }
        for (var i = 0; i < NX; i++) { var dx = xs[N][i] - xRefs[N][i]; cost += 0.5 * Qfd[i] * dx * dx; }
        if (xs[N][2] < 0) cost += 0.5 * GROUND_PEN * xs[N][2] * xs[N][2];
        return cost;
    }

    /* ═══ Reference trajectory (cubic Hermite) ═════════════════ */
    function genReference(x0, target, N, dt) {
        var T = N * dt, refs = [];
        for (var k = 0; k <= N; k++) {
            var s = k / N;
            var h00 = 1 - 3*s*s + 2*s*s*s, h10 = s - 2*s*s + s*s*s, h01 = 3*s*s - 2*s*s*s;
            var dh00 = (-6*s + 6*s*s)/T, dh10 = (1 - 4*s + 3*s*s)/T, dh01 = (6*s - 6*s*s)/T;
            refs.push([
                x0[0]*h00 + target[0]*h01 + x0[3]*T*h10,
                x0[1]*h00 + target[1]*h01 + x0[4]*T*h10,
                Math.max(x0[2]*h00 + target[2]*h01 + x0[5]*T*h10, 0.05),
                x0[0]*dh00 + target[0]*dh01 + x0[3]*T*dh10,
                x0[1]*dh00 + target[1]*dh01 + x0[4]*T*dh10,
                x0[2]*dh00 + target[2]*dh01 + x0[5]*T*dh10
            ]);
        }
        return refs;
    }

    /* ═══ QP gradient via adjoint ══════════════════════════════ */
    function qpGrad(us, Ab, Bb, cb, x0, xRefs, Qd, Rd, Qfd, uRef, wTR, usBar) {
        var N = us.length;
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
        var lam = new Array(NX);
        for (var i = 0; i < NX; i++) lam[i] = Qfd[i] * (xs[N][i] - xRefs[N][i]);
        if (xs[N][2] < 0) lam[2] += GROUND_PEN * xs[N][2];
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
                if (i === 2 && xs[k][2] < 0) newLam[i] += GROUND_PEN * xs[k][2];
                for (var j = 0; j < NX; j++) newLam[i] += Ab[k][j][i] * lam[j];
            }
            lam = newLam;
        }
        return grad;
    }

    /* ═══ Hessian-vector product ═══════════════════════════════ */
    function qpHvp(d, Ab, Bb, Qd, Rd, Qfd, wTR, xsBar) {
        var N = d.length;
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
        var lam = new Array(NX);
        for (var i = 0; i < NX; i++) {
            lam[i] = Qfd[i] * dx[N][i];
            if (i === 2 && xsBar[N][2] < 0) lam[i] += GROUND_PEN * dx[N][i];
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
                if (i === 2 && xsBar[k][2] < 0) newLam[i] += GROUND_PEN * dx[k][i];
                for (var j = 0; j < NX; j++) newLam[i] += Ab[k][j][i] * lam[j];
            }
            lam = newLam;
        }
        return Hd;
    }

    /* ═══ SCvx Solver ══════════════════════════════════════════ */
    function scvxSolve(x0, usInit, xRefs, uRef, Qd, Rd, Qfd, p, nSCP, wTR, distEst) {
        var N = usInit.length;
        var us = [];
        for (var k = 0; k < N; k++) us.push(usInit[k].slice());
        var dEst = distEst || null; // estimated disturbance feedforward

        for (var scp = 0; scp < nSCP; scp++) {
            var xsBar = [x0.slice()];
            for (var k = 0; k < N; k++) xsBar.push(rk4(xsBar[k], us[k], p, dEst));

            var Ab = [], Bb = [], cb = [];
            for (var k = 0; k < N; k++) {
                var jac = rk4Jac(xsBar[k], us[k], p, dEst);
                Ab.push(jac.A); Bb.push(jac.B);
                var ck = new Array(NX);
                for (var i = 0; i < NX; i++) {
                    ck[i] = xsBar[k+1][i];
                    for (var j = 0; j < NX; j++) ck[i] -= jac.A[i][j] * xsBar[k][j];
                    for (var j = 0; j < NU; j++) ck[i] -= jac.B[i][j] * us[k][j];
                }
                cb.push(ck);
            }

            var usBar = [];
            for (var k = 0; k < N; k++) usBar.push(us[k].slice());

            var diagP = new Array(NX);
            for (var i = 0; i < NX; i++) diagP[i] = Qfd[i];
            if (xsBar[N][2] < 0) diagP[2] += GROUND_PEN;
            var prec = new Array(N);
            for (var k = N-1; k >= 0; k--) {
                prec[k] = new Array(NU);
                for (var a = 0; a < NU; a++) {
                    prec[k][a] = Rd[a] + wTR;
                    for (var i = 0; i < NX; i++) prec[k][a] += Bb[k][i][a] * Bb[k][i][a] * diagP[i];
                    if (prec[k][a] < 1e-6) prec[k][a] = 1e-6;
                }
                var newDP = new Array(NX);
                for (var i = 0; i < NX; i++) {
                    newDP[i] = Qd[i];
                    if (i === 2 && xsBar[k][2] < 0) newDP[i] += GROUND_PEN;
                    for (var j = 0; j < NX; j++) newDP[i] += Ab[k][j][i] * Ab[k][j][i] * diagP[j];
                }
                diagP = newDP;
            }

            for (var qi = 0; qi < SCVX_QP_ITERS; qi++) {
                var grad = qpGrad(us, Ab, Bb, cb, x0, xRefs, Qd, Rd, Qfd, uRef, wTR, usBar);
                for (var k = 0; k < N; k++) {
                    if (us[k][0] <= p.phiMin + 1e-4 && grad[k][0] > 0) grad[k][0] = 0;
                    if (us[k][0] >= p.phiMax - 1e-4 && grad[k][0] < 0) grad[k][0] = 0;
                    if (us[k][1] <= p.thMin + 1e-4 && grad[k][1] > 0) grad[k][1] = 0;
                    if (us[k][1] >= p.thMax - 1e-4 && grad[k][1] < 0) grad[k][1] = 0;
                    if (us[k][2] <= p.Tmin + 0.01 && grad[k][2] > 0) grad[k][2] = 0;
                    if (us[k][2] >= p.Tmax - 0.01 && grad[k][2] < 0) grad[k][2] = 0;
                }
                var pdir = new Array(N);
                for (var k = 0; k < N; k++)
                    pdir[k] = [grad[k][0]/prec[k][0], grad[k][1]/prec[k][1], grad[k][2]/prec[k][2]];
                var Hpd = qpHvp(pdir, Ab, Bb, Qd, Rd, Qfd, wTR, xsBar);
                var gd = 0, dHd = 0;
                for (var k = 0; k < N; k++) {
                    for (var a = 0; a < NU; a++) { gd += grad[k][a]*pdir[k][a]; dHd += pdir[k][a]*Hpd[k][a]; }
                }
                if (gd < 1e-8 || dHd < 1e-12) break;
                var alpha = gd / dHd;
                for (var k = 0; k < N; k++) {
                    us[k][0] = clamp(us[k][0] - alpha*pdir[k][0], p.phiMin, p.phiMax);
                    us[k][1] = clamp(us[k][1] - alpha*pdir[k][1], p.thMin, p.thMax);
                    us[k][2] = clamp(us[k][2] - alpha*pdir[k][2], p.Tmin, p.Tmax);
                }
            }
        }
        var xs = [x0.slice()];
        for (var k = 0; k < N; k++) xs.push(rk4(xs[k], us[k], p, dEst));
        return { xs: xs, us: us, cost: totalCost(xs, us, xRefs, uRef, Qd, Rd, Qfd) };
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

    function proj(wx, wy, wz, cam) {
        var ca = Math.cos(cam.az), sa = Math.sin(cam.az);
        var x1 = wx * ca - wy * sa;
        var y1 = wx * sa + wy * ca;
        var se = Math.sin(cam.el), ce = Math.cos(cam.el);
        return {
            x: cam.cx + x1 * cam.sc,
            y: cam.cy - (-y1 * se + wz * ce) * cam.sc,
            d: y1 * ce + wz * se
        };
    }

    function unproject(sx, sy, cam) {
        var px = (sx - cam.cx) / cam.sc;
        var py = -(sy - cam.cy) / cam.sc;
        var se = Math.sin(cam.el), ce = Math.cos(cam.el);
        if (Math.abs(se) < 0.01) return null;
        var y1 = py / se;
        var ca = Math.cos(cam.az), sa = Math.sin(cam.az);
        return { x: px * ca + y1 * sa, y: -px * sa + y1 * ca };
    }

    function rotBody(bx, by, bz, phi, theta) {
        var cp = Math.cos(phi), sp = Math.sin(phi);
        var y1 = by * cp - bz * sp, z1 = by * sp + bz * cp, x1 = bx;
        var ct = Math.cos(theta), st = Math.sin(theta);
        return [x1*ct + z1*st, y1, -x1*st + z1*ct];
    }

    /* ═══ Drawing functions ════════════════════════════════════ */
    function drawGround(ctx, w, h, cam) {
        var step = 1, range = 8;
        ctx.lineWidth = 0.5;
        for (var gx = -range; gx <= range; gx += step) {
            var a = proj(gx, -range, 0, cam), b = proj(gx, range, 0, cam);
            ctx.strokeStyle = (gx === 0) ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.06)';
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
        for (var gy = -range; gy <= range; gy += step) {
            var a = proj(-range, gy, 0, cam), b = proj(range, gy, 0, cam);
            ctx.strokeStyle = (gy === 0) ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.06)';
            ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
    }

    function drawTarget(ctx, target, cam) {
        var gs = proj(target[0], target[1], 0, cam);
        ctx.beginPath(); ctx.arc(gs.x, gs.y, 6, 0, 2*Math.PI);
        ctx.fillStyle = 'rgba(255,165,0,0.3)'; ctx.fill();
        var ts = proj(target[0], target[1], target[2], cam);
        ctx.strokeStyle = 'rgba(255,165,0,0.4)'; ctx.lineWidth = 1;
        ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(gs.x, gs.y); ctx.lineTo(ts.x, ts.y); ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(ts.x, ts.y, 8, 0, 2*Math.PI);
        ctx.fillStyle = 'rgba(255,165,0,0.7)'; ctx.fill();
        ctx.strokeStyle = '#FFA500'; ctx.lineWidth = 2; ctx.stroke();
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(ts.x-6,ts.y); ctx.lineTo(ts.x+6,ts.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ts.x,ts.y-6); ctx.lineTo(ts.x,ts.y+6); ctx.stroke();
    }

    /* ═══ Realistic Quadrotor ══════════════════════════════════ */
    function drawQuadrotor(ctx, pos, phi, theta, cam, armLen, time) {
        var L = armLen || 0.35;
        var arms = [[L,0,0],[0,L,0],[-L,0,0],[0,-L,0]];
        var armColors = ['#FF4444','#4488FF','#888888','#4488FF'];
        var bodyW = 0.09, bodyH = 0.06;
        var gearDrop = -0.07, gearSpan = 0.18, gearLen = 0.25;

        // Compute all geometry in world coordinates
        var rotArms = [], scrArms = [], depthArms = [];
        for (var i = 0; i < 4; i++) {
            var r = rotBody(arms[i][0], arms[i][1], arms[i][2], phi, theta);
            rotArms.push(r);
            var p = proj(pos[0]+r[0], pos[1]+r[1], pos[2]+r[2], cam);
            scrArms.push(p);
            depthArms.push({ idx: i, d: p.d });
        }
        var center = proj(pos[0], pos[1], pos[2], cam);

        // Sort by depth (far first)
        depthArms.sort(function(a,b) { return a.d - b.d; });

        // Shadow on ground
        ctx.fillStyle = 'rgba(0,0,0,0.3)';
        var gs = proj(pos[0], pos[1], 0, cam);
        ctx.beginPath(); ctx.arc(gs.x, gs.y, 6, 0, 2*Math.PI); ctx.fill();

        // Landing gear (skids) — draw first (below body)
        var gearPts = [
            [-gearLen, -gearSpan, gearDrop], [gearLen, -gearSpan, gearDrop],
            [-gearLen,  gearSpan, gearDrop], [gearLen,  gearSpan, gearDrop]
        ];
        var strutPts = [
            [[-bodyW*0.8, -gearSpan*0.6, 0], [-gearLen*0.5, -gearSpan, gearDrop]],
            [[ bodyW*0.8, -gearSpan*0.6, 0], [ gearLen*0.5, -gearSpan, gearDrop]],
            [[-bodyW*0.8,  gearSpan*0.6, 0], [-gearLen*0.5,  gearSpan, gearDrop]],
            [[ bodyW*0.8,  gearSpan*0.6, 0], [ gearLen*0.5,  gearSpan, gearDrop]]
        ];
        ctx.strokeStyle = '#556677'; ctx.lineWidth = 1.5;
        // Skid bars
        for (var s = 0; s < 2; s++) {
            var a = gearPts[s*2], b = gearPts[s*2+1];
            var ra = rotBody(a[0],a[1],a[2],phi,theta), rb = rotBody(b[0],b[1],b[2],phi,theta);
            var pa = proj(pos[0]+ra[0],pos[1]+ra[1],pos[2]+ra[2],cam);
            var pb = proj(pos[0]+rb[0],pos[1]+rb[1],pos[2]+rb[2],cam);
            ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
        }
        // Struts
        ctx.lineWidth = 1;
        for (var s = 0; s < 4; s++) {
            var a = strutPts[s][0], b = strutPts[s][1];
            var ra = rotBody(a[0],a[1],a[2],phi,theta), rb = rotBody(b[0],b[1],b[2],phi,theta);
            var pa = proj(pos[0]+ra[0],pos[1]+ra[1],pos[2]+ra[2],cam);
            var pb = proj(pos[0]+rb[0],pos[1]+rb[1],pos[2]+rb[2],cam);
            ctx.beginPath(); ctx.moveTo(pa.x,pa.y); ctx.lineTo(pb.x,pb.y); ctx.stroke();
        }

        // Draw arms + rotors sorted by depth (far first)
        for (var di = 0; di < 4; di++) {
            var idx = depthArms[di].idx;
            var col = armColors[idx];
            var tip = scrArms[idx];

            // Arm beam (thick line)
            ctx.strokeStyle = col; ctx.lineWidth = 3.5;
            ctx.beginPath(); ctx.moveTo(center.x, center.y); ctx.lineTo(tip.x, tip.y); ctx.stroke();

            // Motor housing
            ctx.beginPath(); ctx.arc(tip.x, tip.y, 5, 0, 2*Math.PI);
            ctx.fillStyle = '#333'; ctx.fill();
            ctx.strokeStyle = '#555'; ctx.lineWidth = 1; ctx.stroke();

            // Propeller disc (semi-transparent)
            var rotorR = 12;
            ctx.beginPath(); ctx.arc(tip.x, tip.y, rotorR, 0, 2*Math.PI);
            ctx.fillStyle = col.substring(0,7) + '22'; ctx.fill();
            ctx.strokeStyle = col.substring(0,7) + '66'; ctx.lineWidth = 1; ctx.stroke();

            // Spinning blades (2 pairs at 90 degrees)
            var bladeAngle = (time || 0) * PROP_SPIN + idx * Math.PI * 0.25;
            ctx.lineWidth = 2; ctx.strokeStyle = col.substring(0,7) + 'AA';
            for (var b = 0; b < 2; b++) {
                var ang = bladeAngle + b * Math.PI * 0.5;
                var bx1 = tip.x + rotorR * Math.cos(ang);
                var by1 = tip.y + rotorR * Math.sin(ang);
                var bx2 = tip.x - rotorR * Math.cos(ang);
                var by2 = tip.y - rotorR * Math.sin(ang);
                ctx.beginPath(); ctx.moveTo(bx1, by1); ctx.lineTo(bx2, by2); ctx.stroke();
            }
        }

        // Body frame (central fuselage — octagonal)
        var bodyPts = [
            [bodyW, bodyH*0.5, 0.01], [bodyW*0.5, bodyH, 0.01],
            [-bodyW*0.5, bodyH, 0.01], [-bodyW, bodyH*0.5, 0.01],
            [-bodyW, -bodyH*0.5, 0.01], [-bodyW*0.5, -bodyH, 0.01],
            [bodyW*0.5, -bodyH, 0.01], [bodyW, -bodyH*0.5, 0.01]
        ];
        ctx.beginPath();
        for (var i = 0; i < bodyPts.length; i++) {
            var rb = rotBody(bodyPts[i][0], bodyPts[i][1], bodyPts[i][2], phi, theta);
            var pb = proj(pos[0]+rb[0], pos[1]+rb[1], pos[2]+rb[2], cam);
            if (i === 0) ctx.moveTo(pb.x, pb.y); else ctx.lineTo(pb.x, pb.y);
        }
        ctx.closePath();
        ctx.fillStyle = '#2a3a4a'; ctx.fill();
        ctx.strokeStyle = '#4a6a8a'; ctx.lineWidth = 1.5; ctx.stroke();

        // Direction indicator (small triangle on front of body)
        var fwd = rotBody(bodyW*1.2, 0, 0.02, phi, theta);
        var fl = rotBody(bodyW*0.6, bodyH*0.4, 0.02, phi, theta);
        var fr = rotBody(bodyW*0.6, -bodyH*0.4, 0.02, phi, theta);
        var pf = proj(pos[0]+fwd[0],pos[1]+fwd[1],pos[2]+fwd[2],cam);
        var pl = proj(pos[0]+fl[0],pos[1]+fl[1],pos[2]+fl[2],cam);
        var pr = proj(pos[0]+fr[0],pos[1]+fr[1],pos[2]+fr[2],cam);
        ctx.beginPath(); ctx.moveTo(pf.x,pf.y); ctx.lineTo(pl.x,pl.y); ctx.lineTo(pr.x,pr.y); ctx.closePath();
        ctx.fillStyle = '#FF4444'; ctx.fill();
    }

    /* ═══ Wind particles ═══════════════════════════════════════ */
    function updateWindParticles(sim) {
        if (!sim.windOn) { sim.windParticles.length = 0; return; }
        var wDir = sim.windDir * Math.PI / 180;
        var wStr = sim.windStrength;
        var particles = sim.windParticles;
        var maxParts = Math.min(60, Math.floor(wStr * 12) + 5);
        var spawnRate = Math.floor(wStr * 2) + 1;

        // Spawn at upwind edge
        for (var i = 0; i < spawnRate && particles.length < maxParts; i++) {
            var upwind = 7;
            var sx = -upwind * Math.cos(wDir) + (Math.random() - 0.5) * 14;
            var sy = -upwind * Math.sin(wDir) + (Math.random() - 0.5) * 14;
            var sz = Math.random() * 5 + 0.2;
            particles.push({ x: sx, y: sy, z: sz, age: 0, maxAge: 35 + Math.random() * 25 });
        }

        // Advance
        var speed = wStr * 0.08;
        for (var i = particles.length - 1; i >= 0; i--) {
            var p = particles[i];
            p.x += speed * Math.cos(wDir) + (Math.random()-0.5)*0.02;
            p.y += speed * Math.sin(wDir) + (Math.random()-0.5)*0.02;
            p.z += (Math.random()-0.5)*0.01;
            p.age++;
            if (p.age > p.maxAge || Math.abs(p.x) > 10 || Math.abs(p.y) > 10) {
                particles.splice(i, 1);
            }
        }
    }

    function drawWind(ctx, sim, cam, W, H) {
        if (!sim.windOn || sim.windStrength < 0.1) return;
        var wDir = sim.windDir * Math.PI / 180;
        var wStr = sim.windStrength;
        var particles = sim.windParticles;
        var streakLen = wStr * 0.18;

        ctx.lineWidth = 1.2;
        for (var i = 0; i < particles.length; i++) {
            var p = particles[i];
            var alpha = (1 - p.age / p.maxAge) * 0.45;
            var p0 = proj(p.x, p.y, p.z, cam);
            var p1 = proj(p.x - streakLen * Math.cos(wDir), p.y - streakLen * Math.sin(wDir), p.z, cam);
            ctx.strokeStyle = 'rgba(160,210,255,' + alpha.toFixed(2) + ')';
            ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
        }

        // Blurred wind zone overlay near the quadrotor
        if (sim.x && wStr > 0.5) {
            var qp = proj(sim.x[0], sim.x[1], sim.x[2], cam);
            var grad = ctx.createRadialGradient(qp.x - 30*Math.cos(wDir), qp.y + 30*Math.sin(wDir), 0,
                                                qp.x - 30*Math.cos(wDir), qp.y + 30*Math.sin(wDir), 60 + wStr*8);
            grad.addColorStop(0, 'rgba(140,190,240,' + (0.06*wStr).toFixed(2) + ')');
            grad.addColorStop(1, 'rgba(140,190,240,0)');
            ctx.fillStyle = grad;
            ctx.fillRect(0, 0, W, H);
        }

        // Wind arrow in corner
        ctx.save();
        var ax = W - 50, ay = H - 40;
        ctx.strokeStyle = 'rgba(160,210,255,0.6)'; ctx.lineWidth = 2;
        var arrLen = 12 + wStr * 3;
        var ex = ax + arrLen * Math.cos(wDir), ey = ay - arrLen * Math.sin(wDir);
        ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(ex, ey); ctx.stroke();
        // Arrowhead
        var ha = 0.4;
        ctx.beginPath();
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - 6*Math.cos(wDir-ha), ey + 6*Math.sin(wDir-ha));
        ctx.moveTo(ex, ey);
        ctx.lineTo(ex - 6*Math.cos(wDir+ha), ey + 6*Math.sin(wDir+ha));
        ctx.stroke();
        ctx.fillStyle = 'rgba(160,210,255,0.5)'; ctx.font = '9px sans-serif';
        ctx.fillText('Wind ' + wStr.toFixed(1) + 'N', ax - 25, ay + 14);
        ctx.restore();
    }

    function drawTrail(ctx, trail, cam) {
        if (trail.length < 2) return;
        ctx.lineWidth = 1.5;
        for (var i = 1; i < trail.length; i++) {
            var a = 0.15 + 0.6 * i / trail.length;
            ctx.strokeStyle = 'rgba(100,200,255,' + a.toFixed(2) + ')';
            var p0 = proj(trail[i-1][0], trail[i-1][1], trail[i-1][2], cam);
            var p1 = proj(trail[i][0], trail[i][1], trail[i][2], cam);
            ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.stroke();
        }
    }

    function drawPrediction(ctx, pred, cam) {
        if (!pred || pred.length < 2) return;
        ctx.strokeStyle = 'rgba(255,165,0,0.5)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([5, 4]);
        ctx.beginPath();
        var p0 = proj(pred[0][0], pred[0][1], pred[0][2], cam);
        ctx.moveTo(p0.x, p0.y);
        for (var i = 1; i < pred.length; i += 2) {
            var pi = proj(pred[i][0], pred[i][1], pred[i][2], cam);
            ctx.lineTo(pi.x, pi.y);
        }
        ctx.stroke();
        ctx.setLineDash([]);
    }

    function drawAxes(ctx, cam) {
        var o = proj(0,0,0,cam), len = 1.0;
        var ax = proj(len,0,0,cam), ay = proj(0,len,0,cam), az = proj(0,0,len,cam);
        ctx.lineWidth = 1.5; ctx.font = '11px sans-serif';
        ctx.strokeStyle = '#FF6666'; ctx.beginPath(); ctx.moveTo(o.x,o.y); ctx.lineTo(ax.x,ax.y); ctx.stroke();
        ctx.fillStyle = '#FF6666'; ctx.fillText('X', ax.x+3, ax.y-3);
        ctx.strokeStyle = '#66FF66'; ctx.beginPath(); ctx.moveTo(o.x,o.y); ctx.lineTo(ay.x,ay.y); ctx.stroke();
        ctx.fillStyle = '#66FF66'; ctx.fillText('Y', ay.x+3, ay.y-3);
        ctx.strokeStyle = '#6688FF'; ctx.beginPath(); ctx.moveTo(o.x,o.y); ctx.lineTo(az.x,az.y); ctx.stroke();
        ctx.fillStyle = '#6688FF'; ctx.fillText('Z', az.x+3, az.y-3);
    }

    function drawScene(canvas, sim) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0F1923'; ctx.fillRect(0, 0, W, H);

        var cam = { az: sim.camAz, el: sim.camEl, cx: W*0.5, cy: H*0.55,
                    sc: Math.min(W,H) * 0.07 * sim.camZoom };

        drawGround(ctx, W, H, cam);
        drawAxes(ctx, cam);
        drawWind(ctx, sim, cam, W, H);
        drawTarget(ctx, sim.target, cam);
        drawTrail(ctx, sim.trail, cam);
        if (sim.pred) drawPrediction(ctx, sim.pred, cam);

        var phi = sim.lastU ? sim.lastU[0] : 0;
        var theta = sim.lastU ? sim.lastU[1] : 0;
        drawQuadrotor(ctx, sim.x, phi, theta, cam, 0.35, sim.time);

        /* HUD */
        ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.font = '11px monospace';
        ctx.fillText('pos: ('+sim.x[0].toFixed(2)+', '+sim.x[1].toFixed(2)+', '+sim.x[2].toFixed(2)+')', 10, 18);
        ctx.fillText('vel: ('+sim.x[3].toFixed(2)+', '+sim.x[4].toFixed(2)+', '+sim.x[5].toFixed(2)+')', 10, 32);
        if (sim.lastU) {
            ctx.fillText('\u03C6: '+(sim.lastU[0]*180/Math.PI).toFixed(1)+'\u00B0  \u03B8: '+(sim.lastU[1]*180/Math.PI).toFixed(1)+'\u00B0  T: '+sim.lastU[2].toFixed(1)+'N', 10, 46);
        }
        ctx.fillText('t = '+sim.time.toFixed(1)+'s  step '+sim.step, 10, 60);

        // EKF disturbance estimate
        if (sim.ekfOn && sim.xa) {
            ctx.fillStyle = 'rgba(160,210,255,0.7)';
            ctx.fillText('d\u0302: ('+sim.xa[6].toFixed(2)+', '+sim.xa[7].toFixed(2)+', '+sim.xa[8].toFixed(2)+') N', 10, 74);
        }
        // True disturbance (if wind on)
        if (sim.windOn && sim.windStrength > 0) {
            var wd = sim.windDir * Math.PI / 180;
            ctx.fillStyle = 'rgba(255,180,100,0.5)';
            ctx.fillText('d: ('+(sim.windStrength*Math.cos(wd)).toFixed(2)+', '+(sim.windStrength*Math.sin(wd)).toFixed(2)+', 0.00) N', 10, sim.ekfOn ? 88 : 74);
        }

        // Zoom level
        if (Math.abs(sim.camZoom - 1.0) > 0.05) {
            ctx.fillStyle = 'rgba(255,255,255,0.35)';
            ctx.fillText('zoom: ' + sim.camZoom.toFixed(1) + 'x', W - 80, 18);
        }

        /* Reached indicator */
        var dx = sim.x[0]-sim.target[0], dy = sim.x[1]-sim.target[1], dz = sim.x[2]-sim.target[2];
        var dist = Math.sqrt(dx*dx+dy*dy+dz*dz);
        var spd = Math.sqrt(sim.x[3]*sim.x[3]+sim.x[4]*sim.x[4]+sim.x[5]*sim.x[5]);
        if (dist < 0.3 && spd < 0.5) {
            ctx.fillStyle = '#00FF88'; ctx.font = 'bold 14px sans-serif';
            ctx.fillText('\u2714 Target reached!', W-150, 22);
        }
    }

    /* ═══ Plot panel ═══════════════════════════════════════════ */
    function drawPlots(canvas, sim) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0F1923'; ctx.fillRect(0, 0, W, H);

        var nPlots = (sim.ekfOn && sim.windOn) ? 4 : 3;
        var pH = Math.floor(H / nPlots);
        drawOnePlot(ctx, 0, 0, W, pH, sim.log.t, [sim.log.pz], ['#6688FF'], ['z'], 'Altitude (m)', sim);
        drawOnePlot(ctx, 0, pH, W, pH, sim.log.t, [sim.log.px, sim.log.py], ['#FF6666','#66FF66'], ['x','y'], 'Position (m)', sim);
        drawOnePlot(ctx, 0, 2*pH, W, pH, sim.log.t, [sim.log.T], ['#FFaa44'], ['T'], 'Thrust (N)', sim);
        if (nPlots === 4) {
            drawOnePlot(ctx, 0, 3*pH, W, H-3*pH, sim.log.t,
                [sim.log.dxTrue, sim.log.dxEst, sim.log.dyTrue, sim.log.dyEst],
                ['rgba(255,180,100,0.7)','#A0D2FF','rgba(255,140,80,0.5)','#70B0E0'],
                ['dx','d\u0302x','dy','d\u0302y'], 'Disturbance (N)', sim);
        }
    }

    function drawOnePlot(ctx, ox, oy, W, H, ts, series, colors, labels, title, sim) {
        var pad = {l:38, r:8, t:18, b:20};
        var pw = W-pad.l-pad.r, ph = H-pad.t-pad.b;
        ctx.save(); ctx.translate(ox, oy);

        ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 0.5;
        ctx.strokeRect(pad.l, pad.t, pw, ph);

        if (ts.length < 2) { ctx.restore(); return; }
        var tMin = ts[0], tMax = Math.max(ts[ts.length-1], tMin+1);
        var yMin = Infinity, yMax = -Infinity;
        for (var s = 0; s < series.length; s++) {
            for (var i = 0; i < series[s].length; i++) {
                if (series[s][i] < yMin) yMin = series[s][i];
                if (series[s][i] > yMax) yMax = series[s][i];
            }
        }
        var yMargin = (yMax - yMin) * 0.15 + 0.1;
        yMin -= yMargin; yMax += yMargin;

        for (var s = 0; s < series.length; s++) {
            ctx.strokeStyle = colors[s]; ctx.lineWidth = 1.5;
            ctx.beginPath();
            for (var i = 0; i < series[s].length; i++) {
                var sx = pad.l + (ts[i]-tMin)/(tMax-tMin)*pw;
                var sy = pad.t + (1 - (series[s][i]-yMin)/(yMax-yMin))*ph;
                if (i===0) ctx.moveTo(sx,sy); else ctx.lineTo(sx,sy);
            }
            ctx.stroke();
        }

        if (title === 'Altitude (m)' && sim) {
            var tz = sim.target[2];
            var ty = pad.t + (1 - (tz-yMin)/(yMax-yMin))*ph;
            ctx.strokeStyle = 'rgba(255,165,0,0.5)'; ctx.lineWidth = 1; ctx.setLineDash([4,3]);
            ctx.beginPath(); ctx.moveTo(pad.l,ty); ctx.lineTo(pad.l+pw,ty); ctx.stroke(); ctx.setLineDash([]);
        }

        ctx.fillStyle = 'rgba(255,255,255,0.5)'; ctx.font = '10px sans-serif';
        ctx.fillText(title, pad.l+4, pad.t+12);
        for (var s = 0; s < labels.length; s++) {
            ctx.fillStyle = colors[s];
            ctx.fillText(labels[s], pad.l+pw-14*(labels.length-s), pad.t+12);
        }
        ctx.fillStyle = 'rgba(255,255,255,0.35)'; ctx.font = '9px monospace';
        ctx.fillText(yMax.toFixed(1), pad.l-34, pad.t+9);
        ctx.fillText(yMin.toFixed(1), pad.l-34, pad.t+ph);
        ctx.restore();
    }

    /* ═══ CSS Injection ════════════════════════════════════════ */
    var cssInjected = false;
    function injectCSS() {
        if (cssInjected) return; cssInjected = true;
        var s = document.createElement('style');
        s.textContent = [
            '.quadrotor-demo{margin:1.5rem 0;border:2px solid #2196F3;border-radius:6px;background:#0F1923;overflow:hidden;font-family:"Source Sans 3",system-ui,sans-serif;color:#ccc}',
            '.qd-header{background:#2196F3;color:#fff;padding:0.45rem 1rem;font-weight:700;font-size:0.95rem}',
            '.qd-ctrls{display:flex;flex-wrap:wrap;align-items:center;gap:0.5rem 0.9rem;padding:0.45rem 1rem;border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.82rem}',
            '.qd-ctrls label{display:flex;align-items:center;gap:0.3rem;white-space:nowrap}',
            '.qd-slider{width:65px;accent-color:#2196F3}',
            '.qd-val{min-width:2.4em;text-align:right;font-variant-numeric:tabular-nums;font-family:"SF Mono","Fira Code",monospace;font-size:0.78rem;color:#8cf}',
            '.qd-btn{padding:0.25rem 0.85rem;border:1px solid rgba(255,255,255,0.2);border-radius:4px;background:rgba(33,150,243,0.18);color:#8cf;font-size:0.82rem;cursor:pointer;transition:background 0.15s}',
            '.qd-btn:hover{background:rgba(33,150,243,0.35)}',
            '.qd-check{accent-color:#2196F3;margin-right:2px}',
            '.qd-sep{width:1px;height:18px;background:rgba(255,255,255,0.12);margin:0 0.2rem}',
            '.qd-body{display:flex;min-height:380px}',
            '.qd-scene-wrap{flex:1;min-width:0;position:relative}',
            '.qd-scene{width:100%;height:100%;display:block}',
            '.qd-plot-wrap{flex:0 0 240px;border-left:1px solid rgba(255,255,255,0.06)}',
            '.qd-plots{width:100%;height:100%;display:block}',
            '@media(max-width:720px){.qd-body{flex-direction:column;min-height:auto}.qd-scene-wrap{height:320px}.qd-plot-wrap{flex:none;height:240px;border-left:none;border-top:1px solid rgba(255,255,255,0.06)}}'
        ].join('\n');
        document.head.appendChild(s);
    }

    /* ═══ Build HTML ═══════════════════════════════════════════ */
    function buildHTML(el) {
        el.innerHTML =
        '<div class="qd-header">3D Quadrotor NMPC — Successive Convexification</div>' +
        '<div class="qd-ctrls">' +
          '<label>Target X <input type="range" class="qd-slider" data-id="tx" min="-6" max="6" value="3" step="0.5"><span class="qd-val">3.0</span></label>' +
          '<label>Target Y <input type="range" class="qd-slider" data-id="ty" min="-6" max="6" value="3" step="0.5"><span class="qd-val">3.0</span></label>' +
          '<label>Target Z <input type="range" class="qd-slider" data-id="tz" min="0.5" max="8" value="3" step="0.5"><span class="qd-val">3.0</span></label>' +
          '<label>Q<sub>pos</sub> <input type="range" class="qd-slider" data-id="qp" min="1" max="50" value="20" step="1"><span class="qd-val">20</span></label>' +
          '<label>R<sub>ctrl</sub> <input type="range" class="qd-slider" data-id="rc" min="1" max="20" value="3" step="1"><span class="qd-val">3</span></label>' +
          '<label>N <input type="range" class="qd-slider" data-id="hor" min="10" max="50" value="30" step="5"><span class="qd-val">30</span></label>' +
          '<span class="qd-sep"></span>' +
          '<label>Wind <input type="checkbox" class="qd-check" data-id="windOn"></label>' +
          '<label>W<sub>str</sub> <input type="range" class="qd-slider" data-id="wstr" min="0" max="5" value="2" step="0.5"><span class="qd-val">2.0</span></label>' +
          '<label>W<sub>dir</sub> <input type="range" class="qd-slider" data-id="wdir" min="0" max="360" value="45" step="15"><span class="qd-val">45</span></label>' +
          '<span class="qd-sep"></span>' +
          '<label>EKF <input type="checkbox" class="qd-check" data-id="ekfOn"></label>' +
          '<span class="qd-sep"></span>' +
          '<button class="qd-btn qd-start-btn">\u25B6 Start</button>' +
          '<button class="qd-btn qd-reset-btn">\u21BB Reset</button>' +
        '</div>' +
        '<div class="qd-body">' +
          '<div class="qd-scene-wrap"><canvas class="qd-scene"></canvas></div>' +
          '<div class="qd-plot-wrap"><canvas class="qd-plots"></canvas></div>' +
        '</div>';
    }

    /* ═══ Init Demo ════════════════════════════════════════════ */
    function initDemo(container) {
        injectCSS();
        buildHTML(container);

        var sceneCvs = container.querySelector('.qd-scene');
        var plotCvs  = container.querySelector('.qd-plots');
        var startBtn = container.querySelector('.qd-start-btn');
        var resetBtn = container.querySelector('.qd-reset-btn');

        function sliderVal(id) {
            var s = container.querySelector('[data-id="'+id+'"]');
            return s ? parseFloat(s.value) : 0;
        }
        function checkVal(id) {
            var c = container.querySelector('[data-id="'+id+'"]');
            return c ? c.checked : false;
        }

        container.addEventListener('input', function(e) {
            var s = e.target;
            if (s.classList.contains('qd-slider')) {
                var v = s.nextElementSibling;
                if (v) v.textContent = parseFloat(s.value).toFixed(s.step < 1 ? 1 : 0);
                sim.target = [sliderVal('tx'), sliderVal('ty'), sliderVal('tz')];
                sim.windStrength = sliderVal('wstr');
                sim.windDir = sliderVal('wdir');
            }
            if (s.classList.contains('qd-check')) {
                sim.windOn = checkVal('windOn');
                sim.ekfOn = checkVal('ekfOn');
            }
            if (!running) render();
        });

        var pp = { m: 1.0, g: 9.81, dt: 0.1,
                   phiMin: -0.5, phiMax: 0.5,
                   thMin: -0.5, thMax: 0.5,
                   Tmin: 0.0, Tmax: 20.0,
                   // EKF tuning
                   Qa: diagMat([1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-3, 0.15, 0.15, 0.15], NXA),
                   Ra: [1e-3, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2] };

        var sim = {
            x: [0, 0, 2, 0, 0, 0],
            target: [3, 3, 3],
            trail: [],
            pred: null,
            lastU: [0, 0, pp.m * pp.g],
            time: 0,
            step: 0,
            camAz: -0.65,
            camEl: 0.5,
            camZoom: 1.0,
            log: { t:[], px:[], py:[], pz:[], T:[], dxTrue:[], dxEst:[], dyTrue:[], dyEst:[] },
            usWarm: null,
            // Wind
            windOn: false,
            windStrength: 2.0,
            windDir: 45,
            windParticles: [],
            // EKF
            ekfOn: false,
            xa: null,
            Pa: null
        };

        var running = false, animId = null;

        function initEKF() {
            sim.xa = sim.x.slice().concat([0, 0, 0]);
            sim.Pa = diagMat([0.01, 0.01, 0.01, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0], NXA);
        }

        function getWeights() {
            var qp = sliderVal('qp'), rc = sliderVal('rc');
            return {
                Qd:  [qp, qp, qp*1.5, qp*0.3, qp*0.3, qp*0.3],
                Rd:  [rc, rc, rc*0.005],
                Qfd: [qp*5, qp*5, qp*7, qp*1.5, qp*1.5, qp*1.5],
                N: Math.round(sliderVal('hor'))
            };
        }

        function computeWindDist(sim) {
            if (!sim.windOn || sim.windStrength < 0.01) return null;
            var wDir = sim.windDir * Math.PI / 180;
            var wStr = sim.windStrength;
            return [
                wStr * Math.cos(wDir) + (Math.random()-0.5) * wStr * 0.25,
                wStr * Math.sin(wDir) + (Math.random()-0.5) * wStr * 0.25,
                (Math.random()-0.5) * wStr * 0.08
            ];
        }

        function reset() {
            running = false;
            if (animId) { cancelAnimationFrame(animId); animId = null; }
            sim.x = [0, 0, 2, 0, 0, 0];
            sim.target = [sliderVal('tx'), sliderVal('ty'), sliderVal('tz')];
            sim.trail = [[0, 0, 2]];
            sim.pred = null;
            sim.lastU = [0, 0, pp.m * pp.g];
            sim.time = 0; sim.step = 0;
            sim.camZoom = 1.0;
            sim.log = { t:[0], px:[0], py:[0], pz:[2], T:[pp.m*pp.g], dxTrue:[0], dxEst:[0], dyTrue:[0], dyEst:[0] };
            sim.usWarm = null;
            sim.windOn = checkVal('windOn');
            sim.windStrength = sliderVal('wstr');
            sim.windDir = sliderVal('wdir');
            sim.ekfOn = checkVal('ekfOn');
            sim.windParticles = [];
            initEKF();
            startBtn.textContent = '\u25B6 Start';
            render();
        }

        function physicsStep() {
            var w = getWeights();
            var N = w.N;
            var uHover = [0, 0, pp.m * pp.g];

            // Use EKF-estimated state if enabled, otherwise true state
            var xForMPC = (sim.ekfOn && sim.xa) ? sim.xa.slice(0, NX) : sim.x;
            var refs = genReference(xForMPC, sim.target, N, pp.dt);

            /* Warm start */
            var usInit;
            if (sim.usWarm && sim.usWarm.length >= N) {
                usInit = [];
                for (var k = 1; k < sim.usWarm.length; k++) usInit.push(sim.usWarm[k].slice());
                while (usInit.length < N) usInit.push(uHover.slice());
                usInit = usInit.slice(0, N);
            } else {
                usInit = [];
                for (var k = 0; k < N; k++) usInit.push(uHover.slice());
            }

            // Pass EKF-estimated disturbance as feedforward to the MPC model
            var distEst = (sim.ekfOn && sim.xa) ? sim.xa.slice(NX, NXA) : null;
            var sol = scvxSolve(xForMPC, usInit, refs, uHover, w.Qd, w.Rd, w.Qfd, pp, SCVX_SCP_ITERS, SCVX_TR, distEst);

            sim.lastU = sol.us[0].slice();
            sim.pred = sol.xs;
            sim.usWarm = sol.us;

            /* Compute actual wind disturbance */
            var windDist = computeWindDist(sim);

            /* Apply first control via nonlinear RK4 dynamics (true plant, with wind) */
            sim.x = rk4(sim.x, sim.lastU, pp, windDist);
            sim.x[2] = Math.max(sim.x[2], 0);

            /* EKF predict + update */
            if (sim.ekfOn) {
                if (!sim.xa || !sim.Pa) initEKF();
                var pred = ekfPredict(sim.xa, sim.Pa, sim.lastU, pp);
                // Measurement: true state (+ small sensor noise)
                var meas = sim.x.slice(0, NX);
                for (var i = 0; i < NZ; i++) meas[i] += (Math.random()-0.5)*0.002;
                var upd = ekfUpdate(pred.xa, pred.Pa, meas, pp);
                sim.xa = upd.xa;
                sim.Pa = upd.Pa;
            }

            sim.time += pp.dt;
            sim.step++;

            sim.trail.push([sim.x[0], sim.x[1], sim.x[2]]);
            if (sim.trail.length > 300) sim.trail.shift();

            // Logging
            var trueDistX = windDist ? windDist[0] : 0;
            var trueDistY = windDist ? windDist[1] : 0;
            sim.log.t.push(sim.time);
            sim.log.px.push(sim.x[0]);
            sim.log.py.push(sim.x[1]);
            sim.log.pz.push(sim.x[2]);
            sim.log.T.push(sim.lastU[2]);
            sim.log.dxTrue.push(trueDistX);
            sim.log.dyTrue.push(trueDistY);
            sim.log.dxEst.push(sim.xa ? sim.xa[6] : 0);
            sim.log.dyEst.push(sim.xa ? sim.xa[7] : 0);
            if (sim.log.t.length > 500) {
                sim.log.t.shift(); sim.log.px.shift(); sim.log.py.shift();
                sim.log.pz.shift(); sim.log.T.shift();
                sim.log.dxTrue.shift(); sim.log.dxEst.shift();
                sim.log.dyTrue.shift(); sim.log.dyEst.shift();
            }

            // Update wind particles
            updateWindParticles(sim);
        }

        function render() {
            updateWindParticles(sim);
            drawScene(sceneCvs, sim);
            drawPlots(plotCvs, sim);
        }

        var lastFrame = 0, accum = 0;
        function animLoop(ts) {
            if (!lastFrame) lastFrame = ts;
            var frameDt = (ts - lastFrame) / 1000;
            lastFrame = ts;
            if (frameDt > 0.2) frameDt = 0.2;
            accum += frameDt;
            var steps = 0;
            while (accum >= pp.dt && steps < 3) {
                physicsStep();
                accum -= pp.dt;
                steps++;
            }
            render();
            if (running) animId = requestAnimationFrame(animLoop);
        }

        function doStart() {
            if (running) {
                running = false;
                if (animId) { cancelAnimationFrame(animId); animId = null; }
                startBtn.textContent = '\u25B6 Start';
            } else {
                running = true;
                lastFrame = 0; accum = 0;
                if (sim.ekfOn && !sim.xa) initEKF();
                startBtn.textContent = '\u23F8 Pause';
                animId = requestAnimationFrame(animLoop);
            }
        }

        startBtn.addEventListener('click', doStart);
        resetBtn.addEventListener('click', reset);

        /* Click on scene to set target (x,y on ground) */
        sceneCvs.addEventListener('click', function(e) {
            var rect = sceneCvs.getBoundingClientRect();
            var sx = e.clientX - rect.left, sy = e.clientY - rect.top;
            var W = sceneCvs.clientWidth, H = sceneCvs.clientHeight;
            var cam = { az: sim.camAz, el: sim.camEl, cx: W*0.5, cy: H*0.55,
                        sc: Math.min(W,H) * 0.07 * sim.camZoom };
            var gp = unproject(sx, sy, cam);
            if (gp && Math.abs(gp.x) <= 8 && Math.abs(gp.y) <= 8) {
                sim.target[0] = Math.round(gp.x * 2) / 2;
                sim.target[1] = Math.round(gp.y * 2) / 2;
                var stx = container.querySelector('[data-id="tx"]');
                var sty = container.querySelector('[data-id="ty"]');
                if (stx) { stx.value = sim.target[0]; stx.nextElementSibling.textContent = sim.target[0].toFixed(1); }
                if (sty) { sty.value = sim.target[1]; sty.nextElementSibling.textContent = sim.target[1].toFixed(1); }
                if (!running) render();
            }
        });

        /* Zoom with mouse wheel */
        sceneCvs.addEventListener('wheel', function(e) {
            e.preventDefault();
            var delta = e.deltaY > 0 ? -0.08 : 0.08;
            sim.camZoom = clamp(sim.camZoom + delta * sim.camZoom, 0.3, 5.0);
            if (!running) render();
        }, { passive: false });

        reset();
    }

    /* ═══ Auto-init ════════════════════════════════════════════ */
    function initAll(root) {
        var els = (root || document).querySelectorAll('.quadrotor-demo');
        for (var i = 0; i < els.length; i++) initDemo(els[i]);
    }
    if (document.readyState === 'loading')
        document.addEventListener('DOMContentLoaded', function () { initAll(); });
    else initAll();

    if (typeof window !== 'undefined') window.QuadrotorDemo = { initAll: initAll };
})();
