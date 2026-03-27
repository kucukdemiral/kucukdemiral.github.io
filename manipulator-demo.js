/**
 * Interactive 3D 6-DOF Industrial Manipulator NMPC Demo
 * SCvx (Successive Convexification) with RK4 integration.
 * Full nonlinear Euler-Lagrange dynamics: M(q)q̈ + g(q) + Bq̇ = τ + d
 * EKF for joint state + disturbance torque estimation.
 *
 * HTML hook:  <div class="manipulator-demo"></div>
 */
(function () {
    'use strict';

    var NQ = 6, NX = 12, NU = 6, NXA = 18, NZ = 12;
    // State: [q1..q6, dq1..dq6]   Input: [tau1..tau6]
    // Augmented: [...state, d1..d6] (disturbance torques)

    function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

    /* ═══ DH Parameters (Standard convention, KUKA-like) ═══════ */
    var DH_A     = [0,    0.34, 0.02, 0,    0,    0   ];
    var DH_ALPHA = [Math.PI/2, 0, Math.PI/2, -Math.PI/2, Math.PI/2, 0];
    var DH_D     = [0.34, 0,    0,    0.34, 0,    0.06];

    var LINK_MASS   = [4.0, 3.5, 1.0, 2.5, 0.8, 0.5];
    var MOTOR_INERTIA = [0.08, 0.06, 0.04, 0.02, 0.01, 0.005];
    var B_DAMP      = [5.0, 5.0, 3.0, 2.0, 1.0, 0.5];
    var TAU_MAX     = [80, 80, 40, 20, 10, 5];
    var Q_LO = [-2.97, -2.09, -2.97, -2.09, -2.09, -6.28];
    var Q_HI = [ 2.97,  2.09,  2.97,  2.09,  2.09,  6.28];
    var GRAV = 9.81;

    var SCVX_QP_ITERS = 15;
    var SCVX_SCP_ITERS = 3;
    var SCVX_TR = 2.0;

    /* ═══ 4x4 Matrix ops (row-major flat arrays) ═══════════════ */
    function mat4Id() {
        return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1];
    }
    function mat4Mul(A, B) {
        var C = new Array(16);
        for (var r = 0; r < 4; r++) {
            for (var c = 0; c < 4; c++) {
                var s = 0;
                for (var k = 0; k < 4; k++) s += A[r*4+k] * B[k*4+c];
                C[r*4+c] = s;
            }
        }
        return C;
    }

    /* ═══ DH homogeneous transform ═════════════════════════════ */
    function dhTransform(a, alpha, d, theta) {
        var ct = Math.cos(theta), st = Math.sin(theta);
        var ca = Math.cos(alpha), sa = Math.sin(alpha);
        return [
            ct, -st*ca,  st*sa, a*ct,
            st,  ct*ca, -ct*sa, a*st,
            0,   sa,     ca,    d,
            0,   0,      0,     1
        ];
    }

    /* ═══ Forward Kinematics ═══════════════════════════════════ */
    // Returns 7 frames (base + 6 joints), each a flat 4x4
    function forwardKinematics(q) {
        var frames = [mat4Id()];
        var T = mat4Id();
        for (var i = 0; i < NQ; i++) {
            var Ti = dhTransform(DH_A[i], DH_ALPHA[i], DH_D[i], q[i]);
            T = mat4Mul(T, Ti);
            frames.push(T.slice());
        }
        return frames;
    }

    function framePos(T) { return [T[3], T[7], T[11]]; }
    function frameZ(T)   { return [T[2], T[6], T[10]]; }

    function endEffectorPos(q) {
        var T = mat4Id();
        for (var i = 0; i < NQ; i++)
            T = mat4Mul(T, dhTransform(DH_A[i], DH_ALPHA[i], DH_D[i], q[i]));
        return framePos(T);
    }

    /* ═══ Cross product ════════════════════════════════════════ */
    function cross(a, b) {
        return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]];
    }

    /* ═══ Geometric Jacobian (3×6 linear velocity for EE) ═════ */
    function computeJacob(frames) {
        var pee = framePos(frames[NQ]);
        var J = new Array(3 * NQ);  // row-major 3x6
        for (var i = 0; i < NQ; i++) {
            var zi = frameZ(frames[i]);
            var oi = framePos(frames[i]);
            var r = [pee[0]-oi[0], pee[1]-oi[1], pee[2]-oi[2]];
            var c = cross(zi, r);
            J[0*NQ+i] = c[0];
            J[1*NQ+i] = c[1];
            J[2*NQ+i] = c[2];
        }
        return J;
    }

    /* ═══ Jacobian for link CoM (3×NQ, only cols 0..link) ═════ */
    function linkCoM(frames, link) {
        var p0 = framePos(frames[link]);
        var p1 = framePos(frames[link+1]);
        return [0.5*(p0[0]+p1[0]), 0.5*(p0[1]+p1[1]), 0.5*(p0[2]+p1[2])];
    }

    function computeJacobLink(frames, link) {
        var pc = linkCoM(frames, link);
        var J = new Array(3 * NQ).fill(0);
        for (var i = 0; i <= link; i++) {
            var zi = frameZ(frames[i]);
            var oi = framePos(frames[i]);
            var r = [pc[0]-oi[0], pc[1]-oi[1], pc[2]-oi[2]];
            var c = cross(zi, r);
            J[0*NQ+i] = c[0];
            J[1*NQ+i] = c[1];
            J[2*NQ+i] = c[2];
        }
        return J;
    }

    /* ═══ Mass Matrix M(q) — 6×6 flat array ═══════════════════ */
    function computeM(frames) {
        var M = new Array(NQ*NQ).fill(0);
        for (var link = 0; link < NQ; link++) {
            var Jl = computeJacobLink(frames, link);
            var ml = LINK_MASS[link];
            // M += ml * Jl^T * Jl
            for (var i = 0; i <= link; i++) {
                for (var j = 0; j <= link; j++) {
                    var s = 0;
                    for (var k = 0; k < 3; k++) s += Jl[k*NQ+i] * Jl[k*NQ+j];
                    M[i*NQ+j] += ml * s;
                }
            }
        }
        // Add motor inertia (diagonal)
        for (var i = 0; i < NQ; i++) M[i*NQ+i] += MOTOR_INERTIA[i];
        return M;
    }

    /* ═══ Gravity vector g(q) — 6-vector ══════════════════════ */
    function computeGravVec(frames) {
        var g = new Array(NQ).fill(0);
        for (var link = 0; link < NQ; link++) {
            var Jl = computeJacobLink(frames, link);
            // g_i = m_link * GRAV * Jl[2, i]  (z-row, potential energy gradient)
            for (var i = 0; i <= link; i++) {
                g[i] += LINK_MASS[link] * GRAV * Jl[2*NQ+i];
            }
        }
        return g;
    }

    /* ═══ 6×6 Matrix solve (Gauss-Jordan) ═════════════════════ */
    function solve6(Mflat, rhs) {
        var n = NQ;
        var A = new Array(n);
        for (var i = 0; i < n; i++) {
            A[i] = new Array(n+1);
            for (var j = 0; j < n; j++) A[i][j] = Mflat[i*n+j];
            A[i][n] = rhs[i];
        }
        for (var c = 0; c < n; c++) {
            var best = c;
            for (var r = c+1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[best][c])) best = r;
            var tmp = A[c]; A[c] = A[best]; A[best] = tmp;
            var piv = A[c][c];
            if (Math.abs(piv) < 1e-14) piv = 1e-14;
            for (var j = c; j <= n; j++) A[c][j] /= piv;
            for (var r = 0; r < n; r++) {
                if (r === c) continue;
                var f = A[r][c];
                for (var j = c; j <= n; j++) A[r][j] -= f * A[c][j];
            }
        }
        var x = new Array(n);
        for (var i = 0; i < n; i++) x[i] = A[i][n];
        return x;
    }

    /* ═══ Continuous-time dynamics ═════════════════════════════ */
    function fCont(x, u, p, dist) {
        var q = x.slice(0, NQ), qd = x.slice(NQ, NX);
        var frames = forwardKinematics(q);
        var M = computeM(frames);
        var gvec = computeGravVec(frames);
        var rhs = new Array(NQ);
        for (var i = 0; i < NQ; i++) {
            rhs[i] = u[i] - gvec[i] - B_DAMP[i] * qd[i];
            if (dist) rhs[i] += dist[i];
        }
        var qdd = solve6(M, rhs);
        var xdot = new Array(NX);
        for (var i = 0; i < NQ; i++) xdot[i] = qd[i];
        for (var i = 0; i < NQ; i++) xdot[NQ+i] = qdd[i];
        return xdot;
    }

    /* ═══ RK4 integrator ═══════════════════════════════════════ */
    function rk4(x, u, p, dist) {
        var h = p.dt;
        var k1 = fCont(x, u, p, dist);
        var xm = new Array(NX);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + 0.5*h*k1[i];
        var k2 = fCont(xm, u, p, dist);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + 0.5*h*k2[i];
        var k3 = fCont(xm, u, p, dist);
        for (var i = 0; i < NX; i++) xm[i] = x[i] + h*k3[i];
        var k4 = fCont(xm, u, p, dist);
        var xn = new Array(NX);
        for (var i = 0; i < NX; i++)
            xn[i] = x[i] + h/6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
        return xn;
    }

    /* ═══ Numerical Jacobians ══════════════════════════════════ */
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

    /* ═══ EKF: Generic NxN matrix utilities ═══════════════════ */
    function diagMat(vals, n) {
        var M = [];
        for (var i = 0; i < n; i++) { M.push(new Array(n).fill(0)); M[i][i] = vals[i]; }
        return M;
    }
    function eyeMat(n) {
        var M = [];
        for (var i = 0; i < n; i++) { M.push(new Array(n).fill(0)); M[i][i] = 1; }
        return M;
    }
    function matMulNN(A, B, n) {
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
    function matTransNN(A, n) {
        var T = [];
        for (var i = 0; i < n; i++) {
            T.push(new Array(n));
            for (var j = 0; j < n; j++) T[i][j] = A[j][i];
        }
        return T;
    }
    function matAddNN(A, B, n) {
        var C = [];
        for (var i = 0; i < n; i++) {
            C.push(new Array(n));
            for (var j = 0; j < n; j++) C[i][j] = A[i][j] + B[i][j];
        }
        return C;
    }
    function symmetrise(M, n) {
        for (var i = 0; i < n; i++) for (var j = i+1; j < n; j++) {
            var avg = 0.5*(M[i][j]+M[j][i]); M[i][j] = avg; M[j][i] = avg;
        }
        return M;
    }
    function matInvNN(M, n) {
        var A = [];
        for (var i = 0; i < n; i++) {
            A.push([]);
            for (var j = 0; j < 2*n; j++) A[i].push(j < n ? M[i][j] : (j-n===i ? 1 : 0));
        }
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

    /* ═══ EKF Augmented Jacobian (18×18) ══════════════════════ */
    function augJacobian(x, u, p, dist) {
        var eps = 1e-6;
        var f0 = rk4(x, u, p, dist);
        var Fa = [];
        for (var i = 0; i < NXA; i++) { Fa.push(new Array(NXA).fill(0)); }
        // df/dx (top-left 12×12)
        for (var j = 0; j < NX; j++) {
            var xp = x.slice(); xp[j] += eps;
            var fp = rk4(xp, u, p, dist);
            for (var i = 0; i < NX; i++) Fa[i][j] = (fp[i] - f0[i]) / eps;
        }
        // df/dd (top-right 12×6)
        for (var j = 0; j < NQ; j++) {
            var dp = dist ? dist.slice() : new Array(NQ).fill(0);
            dp[j] += eps;
            var fp = rk4(x, u, p, dp);
            for (var i = 0; i < NX; i++) Fa[i][NX+j] = (fp[i] - f0[i]) / eps;
        }
        // Bottom-right 6×6: identity (random walk)
        for (var i = 0; i < NQ; i++) Fa[NX+i][NX+i] = 1.0;
        return Fa;
    }

    /* ═══ EKF Predict & Update ════════════════════════════════ */
    function ekfPredict(xa, Pa, u, p) {
        var x = xa.slice(0, NX);
        var dist = xa.slice(NX, NXA);
        var xNext = rk4(x, u, p, dist);
        var xaPred = xNext.concat(dist);
        var Fa = augJacobian(x, u, p, dist);
        var FPFt = matMulNN(matMulNN(Fa, Pa, NXA), matTransNN(Fa, NXA), NXA);
        var PaPred = symmetrise(matAddNN(FPFt, p.Qa, NXA), NXA);
        return { xa: xaPred, Pa: PaPred };
    }

    function ekfUpdate(xaPred, PaPred, z, p) {
        var y = new Array(NZ);
        for (var i = 0; i < NZ; i++) y[i] = z[i] - xaPred[i];
        // S = H*P*H^T + R → P[0:12,0:12] + R
        var S = [];
        for (var i = 0; i < NZ; i++) {
            S.push(new Array(NZ));
            for (var j = 0; j < NZ; j++) S[i][j] = PaPred[i][j] + (i===j ? p.Ra[i] : 0);
        }
        var Sinv = matInvNN(S, NZ);
        // K = P*H^T * S^{-1} → P[:,0:12] * Sinv  (18×12 * 12×12 = 18×12)
        var K = [];
        for (var i = 0; i < NXA; i++) {
            K.push(new Array(NZ));
            for (var j = 0; j < NZ; j++) {
                var s = 0;
                for (var l = 0; l < NZ; l++) s += PaPred[i][l] * Sinv[l][j];
                K[i][j] = s;
            }
        }
        var xaUpd = xaPred.slice();
        for (var i = 0; i < NXA; i++)
            for (var j = 0; j < NZ; j++) xaUpd[i] += K[i][j] * y[j];
        // Joseph form: P = (I-KH)*P*(I-KH)^T + K*R*K^T
        var IKH = eyeMat(NXA);
        for (var i = 0; i < NXA; i++) for (var j = 0; j < NZ; j++) IKH[i][j] -= K[i][j];
        var PaUpd = matMulNN(matMulNN(IKH, PaPred, NXA), matTransNN(IKH, NXA), NXA);
        for (var i = 0; i < NXA; i++) for (var j = 0; j < NXA; j++)
            for (var l = 0; l < NZ; l++) PaUpd[i][j] += K[i][l] * p.Ra[l] * K[j][l];
        symmetrise(PaUpd, NXA);
        return { xa: xaUpd, Pa: PaUpd };
    }

    /* ═══ IK solver (damped least-squares, position only) ═════ */
    function invert3(M) {
        var a=M[0],b=M[1],c=M[2],d=M[3],e=M[4],f=M[5],g=M[6],h=M[7],k=M[8];
        var det = a*(e*k-f*h)-b*(d*k-f*g)+c*(d*h-e*g);
        if (Math.abs(det) < 1e-12) det = 1e-12;
        var inv = 1/det;
        return [(e*k-f*h)*inv,(c*h-b*k)*inv,(b*f-c*e)*inv,
                (f*g-d*k)*inv,(a*k-c*g)*inv,(c*d-a*f)*inv,
                (d*h-e*g)*inv,(b*g-a*h)*inv,(a*e-b*d)*inv];
    }

    function solveIK(targetPos, qInit, maxIter) {
        var q = qInit.slice();
        for (var iter = 0; iter < maxIter; iter++) {
            var frames = forwardKinematics(q);
            var ee = framePos(frames[NQ]);
            var err = [targetPos[0]-ee[0], targetPos[1]-ee[1], targetPos[2]-ee[2]];
            if (err[0]*err[0]+err[1]*err[1]+err[2]*err[2] < 1e-10) break;
            var J = computeJacob(frames); // 3×6
            // JJT = J * J^T (3×3) + damping
            var JJT = new Array(9).fill(0);
            for (var i = 0; i < 3; i++) {
                for (var j = 0; j < 3; j++) {
                    for (var k = 0; k < NQ; k++) JJT[i*3+j] += J[i*NQ+k]*J[j*NQ+k];
                }
                JJT[i*3+i] += 0.005; // damping
            }
            var JJTinv = invert3(JJT);
            // tmp = JJTinv * err
            var tmp = [0, 0, 0];
            for (var i = 0; i < 3; i++)
                for (var j = 0; j < 3; j++) tmp[i] += JJTinv[i*3+j] * err[j];
            // dq = J^T * tmp
            for (var i = 0; i < NQ; i++) {
                var dq = 0;
                for (var k = 0; k < 3; k++) dq += J[k*NQ+i] * tmp[k];
                q[i] = clamp(q[i] + dq, Q_LO[i], Q_HI[i]);
            }
        }
        return q;
    }

    /* ═══ Reference trajectory generation ═════════════════════ */
    function genReference(x0, targetCart, N, dt, qPrev) {
        var q0 = x0.slice(0, NQ);
        var ee0 = endEffectorPos(q0);
        var refs = [];
        var qWarm = qPrev ? qPrev.slice() : q0.slice();
        for (var k = 0; k <= N; k++) {
            var s = k / N;
            var h01 = 3*s*s - 2*s*s*s; // smooth step
            var cartK = [
                ee0[0] + h01*(targetCart[0]-ee0[0]),
                ee0[1] + h01*(targetCart[1]-ee0[1]),
                ee0[2] + h01*(targetCart[2]-ee0[2])
            ];
            var qRef = solveIK(cartK, qWarm, 15);
            qWarm = qRef.slice();
            var xRef = qRef.concat([0,0,0,0,0,0]);
            refs.push(xRef);
        }
        return refs;
    }

    /* ═══ Cost function ════════════════════════════════════════ */
    function totalCost(xs, us, xRefs, uRef, Qd, Rd, Qfd) {
        var N = us.length, cost = 0;
        for (var k = 0; k < N; k++) {
            for (var i = 0; i < NX; i++) { var dx = xs[k][i]-xRefs[k][i]; cost += 0.5*Qd[i]*dx*dx; }
            for (var i = 0; i < NU; i++) { var du = us[k][i]-uRef[i]; cost += 0.5*Rd[i]*du*du; }
        }
        for (var i = 0; i < NX; i++) { var dx = xs[N][i]-xRefs[N][i]; cost += 0.5*Qfd[i]*dx*dx; }
        return cost;
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
        var grad = new Array(N);
        for (var k = N-1; k >= 0; k--) {
            var gk = new Array(NU);
            for (var a = 0; a < NU; a++) {
                gk[a] = Rd[a]*(us[k][a]-uRef[a]) + wTR*(us[k][a]-usBar[k][a]);
                for (var i = 0; i < NX; i++) gk[a] += Bb[k][i][a] * lam[i];
            }
            grad[k] = gk;
            var newLam = new Array(NX);
            for (var i = 0; i < NX; i++) {
                newLam[i] = Qd[i] * (xs[k][i] - xRefs[k][i]);
                for (var j = 0; j < NX; j++) newLam[i] += Ab[k][j][i] * lam[j];
            }
            lam = newLam;
        }
        return grad;
    }

    /* ═══ Hessian-vector product ══════════════════════════════ */
    function qpHvp(d, Ab, Bb, Qd, Rd, Qfd, wTR) {
        var N = d.length;
        var dx = [new Array(NX).fill(0)];
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
        for (var i = 0; i < NX; i++) lam[i] = Qfd[i] * dx[N][i];
        var Hd = new Array(N);
        for (var k = N-1; k >= 0; k--) {
            var hk = new Array(NU);
            for (var a = 0; a < NU; a++) {
                hk[a] = (Rd[a]+wTR) * d[k][a];
                for (var i = 0; i < NX; i++) hk[a] += Bb[k][i][a] * lam[i];
            }
            Hd[k] = hk;
            var newLam = new Array(NX);
            for (var i = 0; i < NX; i++) {
                newLam[i] = Qd[i] * dx[k][i];
                for (var j = 0; j < NX; j++) newLam[i] += Ab[k][j][i] * lam[j];
            }
            lam = newLam;
        }
        return Hd;
    }

    /* ═══ SCvx Solver ═════════════════════════════════════════ */
    function scvxSolve(x0, usInit, xRefs, uRef, Qd, Rd, Qfd, p, nSCP, wTR, distEst) {
        var N = usInit.length;
        var us = [];
        for (var k = 0; k < N; k++) us.push(usInit[k].slice());
        var dEst = distEst || null;

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
                    for (var j = 0; j < NX; j++) ck[i] -= jac.A[i][j]*xsBar[k][j];
                    for (var j = 0; j < NU; j++) ck[i] -= jac.B[i][j]*us[k][j];
                }
                cb.push(ck);
            }
            var usBar = [];
            for (var k = 0; k < N; k++) usBar.push(us[k].slice());
            // Preconditioner (diagonal approximation)
            var diagP = new Array(NX);
            for (var i = 0; i < NX; i++) diagP[i] = Qfd[i];
            var prec = new Array(N);
            for (var k = N-1; k >= 0; k--) {
                prec[k] = new Array(NU);
                for (var a = 0; a < NU; a++) {
                    prec[k][a] = Rd[a] + wTR;
                    for (var i = 0; i < NX; i++) prec[k][a] += Bb[k][i][a]*Bb[k][i][a]*diagP[i];
                    if (prec[k][a] < 1e-6) prec[k][a] = 1e-6;
                }
                var newDP = new Array(NX);
                for (var i = 0; i < NX; i++) {
                    newDP[i] = Qd[i];
                    for (var j = 0; j < NX; j++) newDP[i] += Ab[k][j][i]*Ab[k][j][i]*diagP[j];
                }
                diagP = newDP;
            }
            // QP iterations
            for (var qi = 0; qi < SCVX_QP_ITERS; qi++) {
                var grad = qpGrad(us, Ab, Bb, cb, x0, xRefs, Qd, Rd, Qfd, uRef, wTR, usBar);
                // Project gradient at bounds
                for (var k = 0; k < N; k++) {
                    for (var a = 0; a < NU; a++) {
                        if (us[k][a] <= -TAU_MAX[a]+0.01 && grad[k][a] > 0) grad[k][a] = 0;
                        if (us[k][a] >= TAU_MAX[a]-0.01 && grad[k][a] < 0) grad[k][a] = 0;
                    }
                }
                var pdir = new Array(N);
                for (var k = 0; k < N; k++) {
                    pdir[k] = new Array(NU);
                    for (var a = 0; a < NU; a++) pdir[k][a] = grad[k][a]/prec[k][a];
                }
                var Hpd = qpHvp(pdir, Ab, Bb, Qd, Rd, Qfd, wTR);
                var gd = 0, dHd = 0;
                for (var k = 0; k < N; k++)
                    for (var a = 0; a < NU; a++) { gd += grad[k][a]*pdir[k][a]; dHd += pdir[k][a]*Hpd[k][a]; }
                if (gd < 1e-8 || dHd < 1e-12) break;
                var alpha = gd / dHd;
                for (var k = 0; k < N; k++)
                    for (var a = 0; a < NU; a++)
                        us[k][a] = clamp(us[k][a] - alpha*pdir[k][a], -TAU_MAX[a], TAU_MAX[a]);
            }
        }
        var xs = [x0.slice()];
        for (var k = 0; k < N; k++) xs.push(rk4(xs[k], us[k], p, dEst));
        return { xs: xs, us: us };
    }

    /* ═══ Canvas helpers ═══════════════════════════════════════ */
    function setupCanvas(canvas) {
        var dpr = window.devicePixelRatio || 1;
        var w = canvas.clientWidth, h = canvas.clientHeight;
        canvas.width = w*dpr; canvas.height = h*dpr;
        var ctx = canvas.getContext('2d');
        ctx.scale(dpr, dpr);
        return { ctx: ctx, w: w, h: h };
    }

    function proj(wx, wy, wz, cam) {
        var ca = Math.cos(cam.az), sa = Math.sin(cam.az);
        var x1 = wx*ca - wy*sa;
        var y1 = wx*sa + wy*ca;
        var se = Math.sin(cam.el), ce = Math.cos(cam.el);
        return {
            x: cam.cx + x1*cam.sc,
            y: cam.cy - (-y1*se + wz*ce)*cam.sc,
            d: y1*ce + wz*se
        };
    }

    /* ═══ Drawing functions ═══════════════════════════════════ */
    function drawGround(ctx, W, H, cam) {
        var step = 0.2, range = 1.5;
        ctx.lineWidth = 0.5;
        for (var gx = -range; gx <= range+0.001; gx += step) {
            var a = proj(gx, -range, 0, cam), b = proj(gx, range, 0, cam);
            ctx.strokeStyle = Math.abs(gx) < 0.01 ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.05)';
            ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
        }
        for (var gy = -range; gy <= range+0.001; gy += step) {
            var a = proj(-range, gy, 0, cam), b = proj(range, gy, 0, cam);
            ctx.strokeStyle = Math.abs(gy) < 0.01 ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.05)';
            ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
        }
    }

    function drawAxes(ctx, cam) {
        var o = proj(0,0,0,cam), len = 0.2;
        var ax = proj(len,0,0,cam), ay = proj(0,len,0,cam), az = proj(0,0,len,cam);
        ctx.lineWidth = 1.5; ctx.font = '10px sans-serif';
        ctx.strokeStyle = '#FF6666'; ctx.beginPath(); ctx.moveTo(o.x,o.y); ctx.lineTo(ax.x,ax.y); ctx.stroke();
        ctx.fillStyle = '#FF6666'; ctx.fillText('X',ax.x+3,ax.y-3);
        ctx.strokeStyle = '#66FF66'; ctx.beginPath(); ctx.moveTo(o.x,o.y); ctx.lineTo(ay.x,ay.y); ctx.stroke();
        ctx.fillStyle = '#66FF66'; ctx.fillText('Y',ay.x+3,ay.y-3);
        ctx.strokeStyle = '#6688FF'; ctx.beginPath(); ctx.moveTo(o.x,o.y); ctx.lineTo(az.x,az.y); ctx.stroke();
        ctx.fillStyle = '#6688FF'; ctx.fillText('Z',az.x+3,az.y-3);
    }

    function drawTarget(ctx, target, cam) {
        var gs = proj(target[0], target[1], 0, cam);
        ctx.beginPath(); ctx.arc(gs.x,gs.y,5,0,2*Math.PI);
        ctx.fillStyle = 'rgba(255,165,0,0.25)'; ctx.fill();
        var ts = proj(target[0], target[1], target[2], cam);
        ctx.strokeStyle = 'rgba(255,165,0,0.4)'; ctx.lineWidth = 1;
        ctx.setLineDash([3,3]); ctx.beginPath(); ctx.moveTo(gs.x,gs.y); ctx.lineTo(ts.x,ts.y); ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(ts.x,ts.y,7,0,2*Math.PI);
        ctx.fillStyle = 'rgba(255,165,0,0.6)'; ctx.fill();
        ctx.strokeStyle = '#FFA500'; ctx.lineWidth = 2; ctx.stroke();
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(ts.x-6,ts.y); ctx.lineTo(ts.x+6,ts.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ts.x,ts.y-6); ctx.lineTo(ts.x,ts.y+6); ctx.stroke();
    }

    var LINK_COLORS = ['#5588DD','#5588DD','#44AA66','#44AA66','#DD7733','#DD7733'];
    var LINK_WIDTHS = [10, 9, 7, 7, 5, 4];

    function drawManipulator(ctx, q, cam) {
        var frames = forwardKinematics(q);
        var pts = [], scr = [];
        for (var i = 0; i <= NQ; i++) {
            var p = framePos(frames[i]);
            pts.push(p);
            scr.push(proj(p[0], p[1], p[2], cam));
        }
        // Shadow
        ctx.strokeStyle = 'rgba(0,0,0,0.15)'; ctx.lineWidth = 3; ctx.lineCap = 'round';
        for (var i = 0; i < NQ; i++) {
            var a = proj(pts[i][0],pts[i][1],0,cam), b = proj(pts[i+1][0],pts[i+1][1],0,cam);
            ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
        }
        // Base pedestal
        var base0 = proj(-0.06,-0.06,0,cam), base1 = proj(0.06,-0.06,0,cam);
        var base2 = proj(0.06,0.06,0,cam), base3 = proj(-0.06,0.06,0,cam);
        ctx.fillStyle = '#2a2a3a';
        ctx.beginPath(); ctx.moveTo(base0.x,base0.y); ctx.lineTo(base1.x,base1.y);
        ctx.lineTo(base2.x,base2.y); ctx.lineTo(base3.x,base3.y); ctx.closePath(); ctx.fill();
        ctx.strokeStyle = '#444'; ctx.lineWidth = 1; ctx.stroke();

        // Depth-sorted links
        var linkDepths = [];
        for (var i = 0; i < NQ; i++)
            linkDepths.push({ idx: i, d: (scr[i].d+scr[i+1].d)*0.5 });
        linkDepths.sort(function(a,b) { return a.d - b.d; });

        for (var di = 0; di < NQ; di++) {
            var idx = linkDepths[di].idx;
            var a = scr[idx], b = scr[idx+1];
            var w = LINK_WIDTHS[idx];
            // Link body
            var dx = b.x-a.x, dy = b.y-a.y;
            var len = Math.sqrt(dx*dx+dy*dy);
            if (len > 0.5) {
                var nx = -dy/len*w*0.5, ny = dx/len*w*0.5;
                ctx.beginPath();
                ctx.moveTo(a.x+nx,a.y+ny); ctx.lineTo(b.x+nx,b.y+ny);
                ctx.lineTo(b.x-nx,b.y-ny); ctx.lineTo(a.x-nx,a.y-ny);
                ctx.closePath();
                ctx.fillStyle = LINK_COLORS[idx]; ctx.fill();
                ctx.strokeStyle = 'rgba(255,255,255,0.15)'; ctx.lineWidth = 0.5; ctx.stroke();
            }
            // Joint circle
            ctx.beginPath(); ctx.arc(a.x,a.y,w*0.55,0,2*Math.PI);
            ctx.fillStyle = '#1a1a2e'; ctx.fill();
            ctx.strokeStyle = 'rgba(255,255,255,0.3)'; ctx.lineWidth = 1; ctx.stroke();
        }
        // End-effector last joint
        var ee = scr[NQ];
        ctx.beginPath(); ctx.arc(ee.x,ee.y,3,0,2*Math.PI);
        ctx.fillStyle = '#1a1a2e'; ctx.fill();
        // EE crosshair
        ctx.strokeStyle = '#FF4444'; ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(ee.x-7,ee.y); ctx.lineTo(ee.x+7,ee.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ee.x,ee.y-7); ctx.lineTo(ee.x,ee.y+7); ctx.stroke();
        ctx.beginPath(); ctx.arc(ee.x,ee.y,5,0,2*Math.PI); ctx.stroke();
    }

    function drawTrail(ctx, trail, cam) {
        if (trail.length < 2) return;
        ctx.strokeStyle = 'rgba(0,200,255,0.3)'; ctx.lineWidth = 1.5; ctx.lineCap = 'round';
        ctx.beginPath();
        var p0 = proj(trail[0][0],trail[0][1],trail[0][2],cam);
        ctx.moveTo(p0.x, p0.y);
        for (var i = 1; i < trail.length; i++) {
            var pi = proj(trail[i][0],trail[i][1],trail[i][2],cam);
            ctx.lineTo(pi.x, pi.y);
        }
        ctx.stroke();
    }

    function drawPrediction(ctx, predEE, cam) {
        if (!predEE || predEE.length < 2) return;
        ctx.strokeStyle = 'rgba(255,100,255,0.5)'; ctx.lineWidth = 1;
        ctx.setLineDash([4,3]);
        ctx.beginPath();
        var p0 = proj(predEE[0][0],predEE[0][1],predEE[0][2],cam);
        ctx.moveTo(p0.x,p0.y);
        for (var i = 1; i < predEE.length; i += 2) {
            var pi = proj(predEE[i][0],predEE[i][1],predEE[i][2],cam);
            ctx.lineTo(pi.x,pi.y);
        }
        ctx.stroke(); ctx.setLineDash([]);
    }

    /* ═══ Scene composite ═════════════════════════════════════ */
    function drawScene(canvas, sim) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0F1923'; ctx.fillRect(0,0,W,H);
        var cam = { az: sim.camAz, el: sim.camEl,
                    cx: W*0.5+(sim.camPanX||0), cy: H*0.6+(sim.camPanY||0),
                    sc: Math.min(W,H)*0.35*sim.camZoom };
        drawGround(ctx, W, H, cam);
        drawAxes(ctx, cam);
        drawTarget(ctx, sim.target, cam);
        drawTrail(ctx, sim.trail, cam);
        if (sim.predEE) drawPrediction(ctx, sim.predEE, cam);
        drawManipulator(ctx, sim.x.slice(0,NQ), cam);

        /* HUD */
        ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.font = '11px monospace';
        var ee = endEffectorPos(sim.x.slice(0,NQ));
        ctx.fillText('ee: ('+ee[0].toFixed(3)+', '+ee[1].toFixed(3)+', '+ee[2].toFixed(3)+')', 10, 18);
        var qDeg = '';
        for (var i = 0; i < NQ; i++) qDeg += (sim.x[i]*180/Math.PI).toFixed(1) + (i<NQ-1?', ':'');
        ctx.fillText('q\u00B0: ['+qDeg+']', 10, 32);
        if (sim.lastU) {
            var tStr = '';
            for (var i = 0; i < 3; i++) tStr += sim.lastU[i].toFixed(1) + (i<2?', ':'');
            ctx.fillText('\u03C4: ['+tStr+', ...]', 10, 46);
        }
        ctx.fillText('t = '+sim.time.toFixed(1)+'s', 10, 60);

        var hudY = 74;
        if (sim.ekfOn && sim.xa) {
            ctx.fillStyle = 'rgba(160,210,255,0.7)';
            var dStr = '';
            for (var i = 0; i < 3; i++) dStr += sim.xa[NX+i].toFixed(1) + (i<2?', ':'');
            ctx.fillText('d\u0302: ['+dStr+', ...]', 10, hudY);
            hudY += 14;
        }

        // Tracking error
        var eP = Math.sqrt(Math.pow(ee[0]-sim.target[0],2)+Math.pow(ee[1]-sim.target[1],2)+Math.pow(ee[2]-sim.target[2],2));
        ctx.fillStyle = 'rgba(255,255,255,0.5)';
        ctx.fillText('|e|: '+eP.toFixed(4)+' m', W-130, 18);
        if (eP < 0.02) {
            ctx.fillStyle = '#00FF88'; ctx.font = 'bold 13px sans-serif';
            ctx.fillText('\u2714 Target reached', W-140, 36);
        }
        ctx.fillStyle = 'rgba(255,255,255,0.18)'; ctx.font = '10px sans-serif';
        ctx.fillText('Drag: orbit | Shift+drag: pan | Scroll: zoom', 10, H-8);
    }

    /* ═══ Plot panel ══════════════════════════════════════════ */
    function drawOnePlot(ctx, x0, y0, pw, ph, tData, series, colors, labels, title, sim) {
        ctx.save();
        ctx.beginPath(); ctx.rect(x0,y0,pw,ph); ctx.clip();
        var pad = { l: x0+40, t: y0+4, r: x0+pw-6 };
        var plotW = pad.r - pad.l, plotH = ph - 22;
        // Auto-range
        var yMin = Infinity, yMax = -Infinity;
        for (var s = 0; s < series.length; s++) {
            for (var i = 0; i < series[s].length; i++) {
                if (series[s][i] < yMin) yMin = series[s][i];
                if (series[s][i] > yMax) yMax = series[s][i];
            }
        }
        var margin = (yMax-yMin)*0.15 + 0.01;
        yMin -= margin; yMax += margin;
        if (yMax - yMin < 0.02) { yMax += 0.01; yMin -= 0.01; }
        // Draw
        var n = tData.length;
        for (var s = 0; s < series.length; s++) {
            ctx.strokeStyle = colors[s]; ctx.lineWidth = 1.2;
            ctx.beginPath();
            for (var i = 0; i < n; i++) {
                var px = pad.l + (i/(n-1||1))*plotW;
                var py = pad.t + (1-(series[s][i]-yMin)/(yMax-yMin))*plotH;
                i === 0 ? ctx.moveTo(px,py) : ctx.lineTo(px,py);
            }
            ctx.stroke();
        }
        // Target line for EE plots
        ctx.strokeStyle = 'rgba(255,255,255,0.12)'; ctx.lineWidth = 0.5;
        ctx.beginPath(); ctx.moveTo(pad.l, pad.t+plotH); ctx.lineTo(pad.r, pad.t+plotH); ctx.stroke();

        ctx.fillStyle = 'rgba(255,255,255,0.5)'; ctx.font = '10px sans-serif';
        ctx.fillText(title, pad.l+4, pad.t+12);
        for (var s = 0; s < labels.length; s++) {
            ctx.fillStyle = colors[s];
            ctx.fillText(labels[s], pad.l+plotW-14*(labels.length-s), pad.t+12);
        }
        ctx.fillStyle = 'rgba(255,255,255,0.3)'; ctx.font = '9px monospace';
        ctx.fillText(yMax.toFixed(1), pad.l-34, pad.t+9);
        ctx.fillText(yMin.toFixed(1), pad.l-34, pad.t+plotH);
        ctx.restore();
    }

    function drawPlots(canvas, sim) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0F1923'; ctx.fillRect(0,0,W,H);
        var nPlots = (sim.ekfOn && (sim.payloadOn||sim.frictionOn)) ? 4 : 3;
        var pH = Math.floor(H / nPlots);
        drawOnePlot(ctx, 0, 0, W, pH, sim.log.t,
            [sim.log.eex, sim.log.eey, sim.log.eez],
            ['#FF6666','#66FF66','#6688FF'], ['x','y','z'], 'End-Effector (m)', sim);
        drawOnePlot(ctx, 0, pH, W, pH, sim.log.t,
            [sim.log.errNorm], ['#FFaa44'], ['|e|'], 'Track Error (m)', sim);
        drawOnePlot(ctx, 0, 2*pH, W, pH, sim.log.t,
            [sim.log.tau1, sim.log.tau2, sim.log.tau3],
            ['#FF6666','#66FF66','#6688FF'], ['\u03C4\u2081','\u03C4\u2082','\u03C4\u2083'], 'Torques (Nm)', sim);
        if (nPlots === 4) {
            drawOnePlot(ctx, 0, 3*pH, W, H-3*pH, sim.log.t,
                [sim.log.d1True, sim.log.d1Est],
                ['rgba(255,180,100,0.8)','#A0D2FF'], ['d\u2081','d\u0302\u2081'], 'Dist Torque J1 (Nm)', sim);
        }
    }

    /* ═══ CSS Injection ═══════════════════════════════════════ */
    var cssInjected = false;
    function injectCSS() {
        if (cssInjected) return; cssInjected = true;
        var s = document.createElement('style');
        s.textContent = [
            '.manipulator-demo{margin:1.5rem 0;border:2px solid #FF6B35;border-radius:6px;background:#0F1923;overflow:hidden;font-family:"Source Sans 3",system-ui,sans-serif;color:#ccc}',
            '.md-header{background:#FF6B35;color:#fff;padding:0.45rem 1rem;font-weight:700;font-size:0.95rem}',
            '.md-ctrls{display:flex;flex-wrap:wrap;align-items:center;gap:0.5rem 0.9rem;padding:0.45rem 1rem;border-bottom:1px solid rgba(255,255,255,0.06);font-size:0.82rem}',
            '.md-ctrls label{display:flex;align-items:center;gap:0.3rem;white-space:nowrap}',
            '.md-slider{width:60px;accent-color:#FF6B35}',
            '.md-val{min-width:2.2em;text-align:right;font-variant-numeric:tabular-nums;font-family:"SF Mono","Fira Code",monospace;font-size:0.78rem;color:#fc9}',
            '.md-btn{padding:0.25rem 0.85rem;border:1px solid rgba(255,255,255,0.2);border-radius:4px;background:rgba(255,107,53,0.18);color:#fc9;font-size:0.82rem;cursor:pointer;transition:background 0.15s}',
            '.md-btn:hover{background:rgba(255,107,53,0.35)}',
            '.md-check{accent-color:#FF6B35;margin-right:2px}',
            '.md-sep{width:1px;height:18px;background:rgba(255,255,255,0.12);margin:0 0.2rem}',
            '.md-body{display:flex;min-height:400px}',
            '.md-scene-wrap{flex:1;min-width:0;position:relative}',
            '.md-scene{width:100%;height:100%;display:block;cursor:grab}',
            '.md-scene:active{cursor:grabbing}',
            '.md-plot-wrap{flex:0 0 240px;border-left:1px solid rgba(255,255,255,0.06)}',
            '.md-plots{width:100%;height:100%;display:block}',
            '@media(max-width:720px){.md-body{flex-direction:column;min-height:auto}.md-scene-wrap{height:340px}.md-plot-wrap{flex:none;height:260px;border-left:none;border-top:1px solid rgba(255,255,255,0.06)}}'
        ].join('\n');
        document.head.appendChild(s);
    }

    /* ═══ Build HTML ══════════════════════════════════════════ */
    function buildHTML(el) {
        el.innerHTML =
        '<div class="md-header">6-DOF Industrial Manipulator NMPC \u2014 SCvx + EKF</div>' +
        '<div class="md-ctrls">' +
          '<label>X <input type="range" class="md-slider" data-id="tx" min="-0.7" max="0.7" value="0.4" step="0.05"><span class="md-val">0.40</span></label>' +
          '<label>Y <input type="range" class="md-slider" data-id="ty" min="-0.7" max="0.7" value="0.2" step="0.05"><span class="md-val">0.20</span></label>' +
          '<label>Z <input type="range" class="md-slider" data-id="tz" min="0.05" max="0.9" value="0.50" step="0.05"><span class="md-val">0.50</span></label>' +
          '<span class="md-sep"></span>' +
          '<label>Q <input type="range" class="md-slider" data-id="qp" min="5" max="200" value="60" step="5"><span class="md-val">60</span></label>' +
          '<label>R <input type="range" class="md-slider" data-id="rc" min="0.01" max="2" value="0.1" step="0.01"><span class="md-val">0.10</span></label>' +
          '<label>N <input type="range" class="md-slider" data-id="hor" min="5" max="15" value="10" step="1"><span class="md-val">10</span></label>' +
          '<span class="md-sep"></span>' +
          '<label>Payload <input type="checkbox" class="md-check" data-id="payloadOn"></label>' +
          '<label>kg <input type="range" class="md-slider" data-id="pmass" min="0.5" max="5" value="2" step="0.5"><span class="md-val">2.0</span></label>' +
          '<label>Friction <input type="checkbox" class="md-check" data-id="frictionOn"></label>' +
          '<span class="md-sep"></span>' +
          '<label>EKF <input type="checkbox" class="md-check" data-id="ekfOn"></label>' +
          '<span class="md-sep"></span>' +
          '<button class="md-btn md-start-btn">\u25B6 Start</button>' +
          '<button class="md-btn md-reset-btn">\u21BB Reset</button>' +
        '</div>' +
        '<div class="md-body">' +
          '<div class="md-scene-wrap"><canvas class="md-scene"></canvas></div>' +
          '<div class="md-plot-wrap"><canvas class="md-plots"></canvas></div>' +
        '</div>';
    }

    /* ═══ Init Demo ═══════════════════════════════════════════ */
    function initDemo(container) {
        injectCSS();
        buildHTML(container);

        var sceneCvs = container.querySelector('.md-scene');
        var plotCvs  = container.querySelector('.md-plots');
        var startBtn = container.querySelector('.md-start-btn');
        var resetBtn = container.querySelector('.md-reset-btn');

        function sliderVal(id) {
            var s = container.querySelector('[data-id="'+id+'"]');
            return s ? parseFloat(s.value) : 0;
        }
        function checkVal(id) {
            var c = container.querySelector('[data-id="'+id+'"]');
            return c ? c.checked : false;
        }

        var pp = {
            dt: 0.05,
            Qa: diagMat([
                1e-4,1e-4,1e-4,1e-4,1e-4,1e-4,
                1e-3,1e-3,1e-3,1e-3,1e-3,1e-3,
                0.5, 0.5, 0.5, 0.3, 0.2, 0.1
            ], NXA),
            Ra: [1e-3,1e-3,1e-3,1e-3,1e-3,1e-3, 1e-2,1e-2,1e-2,1e-2,1e-2,1e-2]
        };

        var Q0 = [0, -0.3, 0.8, 0, 0.5, 0]; // initial joint angles

        var sim = {
            x: Q0.concat([0,0,0,0,0,0]),
            target: [0.4, 0.2, 0.50],
            trail: [],
            predEE: null,
            lastU: new Array(NU).fill(0),
            time: 0, step: 0,
            camAz: -0.7, camEl: 0.45, camZoom: 1.8,
            camPanX: 0, camPanY: 0,
            log: { t:[], eex:[], eey:[], eez:[], errNorm:[],
                   tau1:[], tau2:[], tau3:[],
                   d1True:[], d1Est:[] },
            usWarm: null,
            qRefPrev: null,
            payloadOn: false, payloadMass: 2.0,
            frictionOn: false,
            ekfOn: false, xa: null, Pa: null,
            distFF: null
        };

        var running = false, animId = null;

        function initEKF() {
            sim.xa = sim.x.slice().concat(new Array(NQ).fill(0));
            var diag = [];
            for (var i = 0; i < NQ; i++) diag.push(0.01);
            for (var i = 0; i < NQ; i++) diag.push(0.1);
            for (var i = 0; i < NQ; i++) diag.push(1.0);
            sim.Pa = diagMat(diag, NXA);
        }

        function getWeights() {
            var qp = sliderVal('qp'), rc = sliderVal('rc');
            var Qd = [], Qfd = [];
            for (var i = 0; i < NQ; i++) { Qd.push(qp); Qfd.push(qp*5); }
            for (var i = 0; i < NQ; i++) { Qd.push(qp*0.1); Qfd.push(qp*0.5); }
            var Rd = [];
            for (var i = 0; i < NU; i++) Rd.push(rc);
            return { Qd: Qd, Rd: Rd, Qfd: Qfd, N: Math.round(sliderVal('hor')) };
        }

        function computeDisturbance(q, qd) {
            var dist = new Array(NQ).fill(0);
            if (sim.payloadOn) {
                var frames = forwardKinematics(q);
                var J = computeJacob(frames); // 3×6
                var fg = [0, 0, -sim.payloadMass * GRAV];
                for (var i = 0; i < NQ; i++)
                    dist[i] += J[0*NQ+i]*fg[0] + J[1*NQ+i]*fg[1] + J[2*NQ+i]*fg[2];
            }
            if (sim.frictionOn) {
                for (var i = 0; i < NQ; i++)
                    dist[i] -= 3.0 * Math.sign(qd[i]+1e-8) * (1.0 + (Math.random()-0.5)*0.2);
            }
            return dist;
        }

        function reset() {
            running = false;
            if (animId) { cancelAnimationFrame(animId); animId = null; }
            sim.x = Q0.slice().concat([0,0,0,0,0,0]);
            sim.target = [sliderVal('tx'), sliderVal('ty'), sliderVal('tz')];
            var ee0 = endEffectorPos(Q0);
            sim.trail = [ee0];
            sim.predEE = null;
            sim.lastU = new Array(NU).fill(0);
            sim.time = 0; sim.step = 0;
            sim.camZoom = 1.8; sim.camPanX = 0; sim.camPanY = 0;
            sim.log = { t:[0], eex:[ee0[0]], eey:[ee0[1]], eez:[ee0[2]], errNorm:[0],
                        tau1:[0], tau2:[0], tau3:[0], d1True:[0], d1Est:[0] };
            sim.usWarm = null;
            sim.qRefPrev = null;
            sim.payloadOn = checkVal('payloadOn');
            sim.payloadMass = sliderVal('pmass');
            sim.frictionOn = checkVal('frictionOn');
            sim.ekfOn = checkVal('ekfOn');
            sim.distFF = null;
            initEKF();
            startBtn.textContent = '\u25B6 Start';
            render();
        }

        function physicsStep() {
            var w = getWeights();
            var N = w.N;

            var xForMPC = (sim.ekfOn && sim.xa) ? sim.xa.slice(0, NX) : sim.x;
            var refs = genReference(xForMPC, sim.target, N, pp.dt, sim.qRefPrev);
            sim.qRefPrev = refs[N].slice(0, NQ);

            // Gravity compensation as reference input
            var q0 = xForMPC.slice(0, NQ);
            var frames0 = forwardKinematics(q0);
            var uRef = computeGravVec(frames0);

            // Warm start
            var usInit;
            if (sim.usWarm && sim.usWarm.length >= N) {
                usInit = [];
                for (var k = 1; k < sim.usWarm.length; k++) usInit.push(sim.usWarm[k].slice());
                while (usInit.length < N) usInit.push(uRef.slice());
                usInit = usInit.slice(0, N);
            } else {
                usInit = [];
                for (var k = 0; k < N; k++) usInit.push(uRef.slice());
            }

            var distEst = (sim.ekfOn && sim.xa) ? sim.xa.slice(NX, NXA) : null;
            sim.distFF = distEst ? distEst.slice() : null;
            var sol = scvxSolve(xForMPC, usInit, refs, uRef, w.Qd, w.Rd, w.Qfd, pp, SCVX_SCP_ITERS, SCVX_TR, distEst);

            sim.lastU = sol.us[0].slice();
            sim.usWarm = sol.us;
            // Predicted EE trajectory
            sim.predEE = [];
            for (var k = 0; k <= N; k++)
                sim.predEE.push(endEffectorPos(sol.xs[k].slice(0, NQ)));

            // True plant disturbance
            var trueDist = computeDisturbance(sim.x.slice(0,NQ), sim.x.slice(NQ,NX));
            // RK4 step on true plant
            sim.x = rk4(sim.x, sim.lastU, pp, trueDist);
            // Clamp joints
            for (var i = 0; i < NQ; i++) {
                sim.x[i] = clamp(sim.x[i], Q_LO[i], Q_HI[i]);
                if (sim.x[i] === Q_LO[i] || sim.x[i] === Q_HI[i]) sim.x[NQ+i] = 0;
            }

            // EKF predict + update
            if (sim.ekfOn) {
                if (!sim.xa || !sim.Pa) initEKF();
                var pred = ekfPredict(sim.xa, sim.Pa, sim.lastU, pp);
                var meas = sim.x.slice(0, NX);
                for (var i = 0; i < NZ; i++) meas[i] += (Math.random()-0.5)*0.001;
                var upd = ekfUpdate(pred.xa, pred.Pa, meas, pp);
                sim.xa = upd.xa;
                sim.Pa = upd.Pa;
            }

            sim.time += pp.dt;
            sim.step++;

            // Logging
            var ee = endEffectorPos(sim.x.slice(0,NQ));
            sim.trail.push(ee.slice());
            if (sim.trail.length > 400) sim.trail.shift();
            var eNorm = Math.sqrt(Math.pow(ee[0]-sim.target[0],2)+Math.pow(ee[1]-sim.target[1],2)+Math.pow(ee[2]-sim.target[2],2));
            sim.log.t.push(sim.time);
            sim.log.eex.push(ee[0]); sim.log.eey.push(ee[1]); sim.log.eez.push(ee[2]);
            sim.log.errNorm.push(eNorm);
            sim.log.tau1.push(sim.lastU[0]); sim.log.tau2.push(sim.lastU[1]); sim.log.tau3.push(sim.lastU[2]);
            sim.log.d1True.push(trueDist ? trueDist[0] : 0);
            sim.log.d1Est.push(sim.xa ? sim.xa[NX] : 0);
            if (sim.log.t.length > 500) {
                sim.log.t.shift(); sim.log.eex.shift(); sim.log.eey.shift(); sim.log.eez.shift();
                sim.log.errNorm.shift();
                sim.log.tau1.shift(); sim.log.tau2.shift(); sim.log.tau3.shift();
                sim.log.d1True.shift(); sim.log.d1Est.shift();
            }
        }

        function render() {
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
            while (accum >= pp.dt && steps < 2) {
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

        // Event listeners
        container.addEventListener('input', function(e) {
            var s = e.target;
            if (s.classList.contains('md-slider')) {
                var v = s.nextElementSibling;
                if (v) v.textContent = parseFloat(s.value).toFixed(s.step < 1 ? 2 : 0);
                sim.target = [sliderVal('tx'), sliderVal('ty'), sliderVal('tz')];
                sim.payloadMass = sliderVal('pmass');
            }
            if (s.classList.contains('md-check')) {
                sim.payloadOn = checkVal('payloadOn');
                sim.frictionOn = checkVal('frictionOn');
                var wasEkf = sim.ekfOn;
                sim.ekfOn = checkVal('ekfOn');
                if (sim.ekfOn && !wasEkf) initEKF();
                if (!sim.ekfOn) sim.distFF = null;
            }
            if (!running) render();
        });

        startBtn.addEventListener('click', doStart);
        resetBtn.addEventListener('click', reset);

        /* Camera interaction */
        var drag = { on:false, btn:0, shift:false, sx:0, sy:0, az0:0, el0:0, px0:0, py0:0 };
        sceneCvs.addEventListener('mousedown', function(e) {
            e.preventDefault();
            drag.on = true; drag.btn = e.button; drag.shift = e.shiftKey;
            drag.sx = e.clientX; drag.sy = e.clientY;
            drag.az0 = sim.camAz; drag.el0 = sim.camEl;
            drag.px0 = sim.camPanX; drag.py0 = sim.camPanY;
        });
        window.addEventListener('mousemove', function(e) {
            if (!drag.on) return;
            var dx = e.clientX - drag.sx, dy = e.clientY - drag.sy;
            if (drag.btn === 0 && !drag.shift) {
                sim.camAz = drag.az0 - dx*0.005;
                sim.camEl = clamp(drag.el0 + dy*0.005, 0.05, 1.5);
            } else {
                sim.camPanX = drag.px0 + dx;
                sim.camPanY = drag.py0 + dy;
            }
            if (!running) render();
        });
        window.addEventListener('mouseup', function() { drag.on = false; });
        sceneCvs.addEventListener('contextmenu', function(e) { e.preventDefault(); });

        /* Touch */
        var touch0 = null, touchDist0 = 0;
        sceneCvs.addEventListener('touchstart', function(e) {
            if (e.touches.length === 1) {
                touch0 = { x:e.touches[0].clientX, y:e.touches[0].clientY, az:sim.camAz, el:sim.camEl };
            } else if (e.touches.length === 2) {
                var dx = e.touches[0].clientX-e.touches[1].clientX;
                var dy = e.touches[0].clientY-e.touches[1].clientY;
                touchDist0 = Math.sqrt(dx*dx+dy*dy);
                touch0 = { zoom: sim.camZoom };
            }
        }, { passive: true });
        sceneCvs.addEventListener('touchmove', function(e) {
            e.preventDefault();
            if (e.touches.length===1 && touch0 && touch0.az!==undefined) {
                sim.camAz = touch0.az - (e.touches[0].clientX-touch0.x)*0.005;
                sim.camEl = clamp(touch0.el + (e.touches[0].clientY-touch0.y)*0.005, 0.05, 1.5);
                if (!running) render();
            } else if (e.touches.length===2 && touch0 && touch0.zoom!==undefined) {
                var dx = e.touches[0].clientX-e.touches[1].clientX;
                var dy = e.touches[0].clientY-e.touches[1].clientY;
                var d = Math.sqrt(dx*dx+dy*dy);
                if (touchDist0>0) { sim.camZoom = clamp(touch0.zoom*d/touchDist0,0.3,5); if (!running) render(); }
            }
        }, { passive: false });
        sceneCvs.addEventListener('touchend', function() { touch0 = null; }, { passive: true });

        sceneCvs.addEventListener('wheel', function(e) {
            e.preventDefault();
            sim.camZoom = clamp(sim.camZoom + (e.deltaY>0?-0.08:0.08)*sim.camZoom, 0.3, 5);
            if (!running) render();
        }, { passive: false });

        reset();
    }

    /* ═══ Auto-init ═══════════════════════════════════════════ */
    function initAll(root) {
        var els = (root || document).querySelectorAll('.manipulator-demo');
        for (var i = 0; i < els.length; i++) initDemo(els[i]);
    }
    if (document.readyState === 'loading')
        document.addEventListener('DOMContentLoaded', function () { initAll(); });
    else initAll();

    if (typeof window !== 'undefined') window.ManipulatorDemo = { initAll: initAll };
})();
