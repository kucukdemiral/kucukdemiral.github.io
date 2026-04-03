/**
 * Interactive Cart-Pole Koopman MPC Demo
 * Pure JS/Canvas — no external dependencies.
 *
 * Learns a Koopman linear model via EDMD from training data,
 * then runs linear MPC in lifted coordinates to swing up and
 * stabilise an inverted pendulum on a cart.
 *
 * HTML hook:  <div class="koopman-cartpole-demo"></div>
 * Init call:  KoopmanCartpoleDemo.initAll(root)
 */
(function () {
    'use strict';

    /* ═══════════════════════════════════════════════════════════
       Physics: cart-pole system
       State: [x, x_dot, theta, theta_dot]
       theta = 0 is upright (inverted pendulum)
       ═══════════════════════════════════════════════════════════ */
    var G  = 9.81;
    var mc = 1.0;    // cart mass
    var mp = 0.3;    // pole mass
    var lp = 0.5;    // pole half-length
    var Ts = 0.02;   // integration step
    var Tc = 0.04;   // control period (every 2 integration steps)
    var u_max = 15.0;
    var x_max = 5.0; // display track half-length
    var T_final = 10.0;

    /* ── Continuous dynamics ──────────────────────────────── */
    function cartpoleDeriv(s, u) {
        var x = s[0], xd = s[1], th = s[2], thd = s[3];
        var sth = Math.sin(th), cth = Math.cos(th);
        var mt = mc + mp;
        var denom = mt - mp * cth * cth;
        var thdd = (G * sth * mt - cth * (u + mp * lp * thd * thd * sth)) / (lp * denom);
        var xdd  = (u + mp * lp * (thd * thd * sth - thdd * cth)) / mt;
        return [xd, xdd, thd, thdd];
    }

    /* ── RK4 step ────────────────────────────────────────── */
    function rk4Step(s, u, dt) {
        var k1 = cartpoleDeriv(s, u);
        var s2 = []; for (var i = 0; i < 4; i++) s2[i] = s[i] + 0.5 * dt * k1[i];
        var k2 = cartpoleDeriv(s2, u);
        var s3 = []; for (var i = 0; i < 4; i++) s3[i] = s[i] + 0.5 * dt * k2[i];
        var k3 = cartpoleDeriv(s3, u);
        var s4 = []; for (var i = 0; i < 4; i++) s4[i] = s[i] + dt * k3[i];
        var k4 = cartpoleDeriv(s4, u);
        var sn = [];
        for (var i = 0; i < 4; i++)
            sn[i] = s[i] + (dt / 6) * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i]);
        return sn;
    }

    /* ── Multi-step simulation at Tc ─────────────────────── */
    function simStep(s, u) {
        var steps = Math.round(Tc / Ts);
        for (var i = 0; i < steps; i++) s = rk4Step(s, u, Ts);
        s[2] = wrapAngle(s[2]);
        return s;
    }

    function wrapAngle(a) {
        while (a > Math.PI) a -= 2 * Math.PI;
        while (a < -Math.PI) a += 2 * Math.PI;
        return a;
    }

    function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

    /* ═══════════════════════════════════════════════════════════
       Koopman / EDMD
       Dictionary: psi(x) = [x1, x2, x3, x4, sin(x3), cos(x3),
                              x2*sin(x3), x2*cos(x3), x4*sin(x3), x4*cos(x3)]
       p = 10 lifted states
       ═══════════════════════════════════════════════════════════ */
    var p_lift = 10;  // lifted dimension

    function liftState(s) {
        var x = s[0], xd = s[1], th = s[2], thd = s[3];
        var sth = Math.sin(th), cth = Math.cos(th);
        return [x, xd, th, thd, sth, cth, xd*sth, xd*cth, thd*sth, thd*cth];
    }

    /* ── Simple matrix helpers (column-major flat arrays) ── */
    function matMul(A, B, m, n, k) {
        // A: m×n, B: n×k => C: m×k
        var C = new Float64Array(m * k);
        for (var j = 0; j < k; j++)
            for (var i = 0; i < m; i++) {
                var sum = 0;
                for (var q = 0; q < n; q++) sum += A[q*m+i] * B[j*n+q];
                C[j*m+i] = sum;
            }
        return C;
    }

    function matTranspose(A, m, n) {
        var AT = new Float64Array(n * m);
        for (var j = 0; j < n; j++)
            for (var i = 0; i < m; i++)
                AT[i*n+j] = A[j*m+i];
        return AT;
    }

    /* Solve A x = b where A is n×n symmetric positive definite (Cholesky) */
    function cholSolve(A, b, n) {
        // In-place Cholesky: A = L L^T
        var L = new Float64Array(n * n);
        for (var j = 0; j < n; j++) {
            var sum = 0;
            for (var k = 0; k < j; k++) sum += L[j*n+k] * L[j*n+k];
            var diag = A[j*n+j] - sum;
            if (diag <= 1e-12) diag = 1e-12;
            L[j*n+j] = Math.sqrt(diag);
            for (var i = j+1; i < n; i++) {
                sum = 0;
                for (var k = 0; k < j; k++) sum += L[i*n+k] * L[j*n+k];
                L[i*n+j] = (A[i*n+j] - sum) / L[j*n+j];
            }
        }
        // Forward solve L y = b
        var y = new Float64Array(n);
        for (var i = 0; i < n; i++) {
            var s = b[i];
            for (var j = 0; j < i; j++) s -= L[i*n+j] * y[j];
            y[i] = s / L[i*n+i];
        }
        // Back solve L^T x = y
        var x = new Float64Array(n);
        for (var i = n-1; i >= 0; i--) {
            var s = y[i];
            for (var j = i+1; j < n; j++) s -= L[j*n+i] * x[j];
            x[i] = s / L[i*n+i];
        }
        return x;
    }

    function matMulRowMajor(A, B) {
        var m = A.length;
        var n = B.length;
        var k = B[0].length;
        var C = new Array(m);
        for (var i = 0; i < m; i++) {
            C[i] = new Array(k);
            for (var j = 0; j < k; j++) {
                var sum = 0;
                for (var q = 0; q < n; q++) sum += A[i][q] * B[q][j];
                C[i][j] = sum;
            }
        }
        return C;
    }

    function transposeRowMajor(A) {
        var m = A.length;
        var n = A[0].length;
        var AT = new Array(n);
        for (var j = 0; j < n; j++) {
            AT[j] = new Array(m);
            for (var i = 0; i < m; i++) AT[j][i] = A[i][j];
        }
        return AT;
    }

    function addRowMajor(A, B, sign) {
        var m = A.length;
        var n = A[0].length;
        var C = new Array(m);
        var sgn = (sign === undefined) ? 1 : sign;
        for (var i = 0; i < m; i++) {
            C[i] = new Array(n);
            for (var j = 0; j < n; j++) C[i][j] = A[i][j] + sgn * B[i][j];
        }
        return C;
    }

    function linearizeCartpoleStep() {
        var x0 = [0, 0, 0, 0];
        var eps = 1e-6;
        var A = [[], [], [], []];
        var B = [0, 0, 0, 0];

        for (var j = 0; j < 4; j++) {
            var xp = x0.slice();
            var xm = x0.slice();
            xp[j] += eps;
            xm[j] -= eps;
            var fp = simStep(xp, 0);
            var fm = simStep(xm, 0);
            for (var i = 0; i < 4; i++) A[i][j] = (fp[i] - fm[i]) / (2 * eps);
        }

        var fu = simStep(x0.slice(), eps);
        var fd = simStep(x0.slice(), -eps);
        for (var k = 0; k < 4; k++) B[k] = (fu[k] - fd[k]) / (2 * eps);

        return { A: A, B: B };
    }

    function computeDiscreteLQRGain(A, B, Q, R) {
        var Bm = [[B[0]], [B[1]], [B[2]], [B[3]]];
        var P = [
            Q[0].slice(),
            Q[1].slice(),
            Q[2].slice(),
            Q[3].slice()
        ];

        for (var iter = 0; iter < 1000; iter++) {
            var AT = transposeRowMajor(A);
            var BT = transposeRowMajor(Bm);
            var ATPA = matMulRowMajor(AT, matMulRowMajor(P, A));
            var ATPB = matMulRowMajor(AT, matMulRowMajor(P, Bm));
            var BTPA = matMulRowMajor(BT, matMulRowMajor(P, A));
            var BTPB = matMulRowMajor(BT, matMulRowMajor(P, Bm));
            var S = R[0][0] + BTPB[0][0];

            var Kterm = [[], [], [], []];
            for (var i = 0; i < 4; i++) {
                for (var j = 0; j < 4; j++)
                    Kterm[i][j] = ATPB[i][0] * BTPA[0][j] / S;
            }

            var Pnext = addRowMajor(Q, addRowMajor(ATPA, Kterm, -1));
            var maxErr = 0;
            for (var r = 0; r < 4; r++)
                for (var c = 0; c < 4; c++)
                    maxErr = Math.max(maxErr, Math.abs(Pnext[r][c] - P[r][c]));
            P = Pnext;
            if (maxErr < 1e-11) break;
        }

        var BTfinal = transposeRowMajor(Bm);
        var Sfinal = R[0][0] + matMulRowMajor(BTfinal, matMulRowMajor(P, Bm))[0][0];
        return matMulRowMajor([[1 / Sfinal]], matMulRowMajor(BTfinal, matMulRowMajor(P, A)))[0];
    }

    /* ── EDMD: learn A_K, B_K from data ──────────────────── */
    function trainEDMD() {
        // Generate training trajectories with random inputs
        var M = 5000; // number of data pairs
        var Zx = [];  // each row: psi(x_k) ++ u_k   (p+1 columns)
        var Zy = [];  // each row: psi(x_{k+1})       (p columns)

        var rng = mulberry32(12345);

        for (var traj = 0; traj < 200; traj++) {
            // random initial state near both hanging and upright
            var s = [
                (rng() - 0.5) * 2.0,
                (rng() - 0.5) * 2.0,
                (rng() - 0.5) * 2 * Math.PI,
                (rng() - 0.5) * 4.0
            ];
            var steps = 25;
            for (var k = 0; k < steps; k++) {
                var u = (rng() - 0.5) * 2 * u_max;
                var z_now = liftState(s);
                var s_next = simStep(s, u);
                var z_next = liftState(s_next);

                // row: [z_now..., u]
                var row_x = new Float64Array(p_lift + 1);
                for (var i = 0; i < p_lift; i++) row_x[i] = z_now[i];
                row_x[p_lift] = u;
                Zx.push(row_x);
                Zy.push(new Float64Array(z_next));

                s = s_next;
            }
        }

        M = Zx.length;
        var d = p_lift + 1; // input dimension (lifted + control)

        // Solve least squares: [A_K | B_K]^T = (Zx^T Zx)^{-1} Zx^T Zy
        // Form Zx^T Zx  (d × d)
        var XtX = new Float64Array(d * d);
        for (var i = 0; i < d; i++)
            for (var j = 0; j <= i; j++) {
                var sum = 0;
                for (var m = 0; m < M; m++) sum += Zx[m][i] * Zx[m][j];
                XtX[i*d+j] = sum;
                XtX[j*d+i] = sum;
            }

        // Regularise
        for (var i = 0; i < d; i++) XtX[i*d+i] += 1e-6;

        // Form Zx^T Zy  (d × p)
        var XtY = new Float64Array(d * p_lift);
        for (var i = 0; i < d; i++)
            for (var j = 0; j < p_lift; j++) {
                var sum = 0;
                for (var m = 0; m < M; m++) sum += Zx[m][i] * Zy[m][j];
                XtY[i*p_lift+j] = sum;
            }

        // Solve d systems: XtX * W[:,j] = XtY[:,j]
        var W = new Float64Array(d * p_lift); // result: d × p
        for (var j = 0; j < p_lift; j++) {
            var rhs = new Float64Array(d);
            for (var i = 0; i < d; i++) rhs[i] = XtY[i*p_lift+j];
            var sol = cholSolve(XtX, rhs, d);
            for (var i = 0; i < d; i++) W[i*p_lift+j] = sol[i];
        }

        // W is (p+1) × p, rows: [A_K^T; B_K^T]
        // A_K = W[0:p, :]^T  (p × p)
        // B_K = W[p, :]^T    (p × 1)
        var A_K = new Float64Array(p_lift * p_lift);
        for (var out = 0; out < p_lift; out++)
            for (var feat = 0; feat < p_lift; feat++)
                A_K[feat*p_lift+out] = W[feat*p_lift+out];

        var B_K = new Float64Array(p_lift);
        for (var j = 0; j < p_lift; j++)
            B_K[j] = W[p_lift*p_lift+j];

        return { A: A_K, B: B_K };
    }

    /* Simple seeded PRNG */
    function mulberry32(seed) {
        return function() {
            seed |= 0; seed = seed + 0x6D2B79F5 | 0;
            var t = Math.imul(seed ^ seed >>> 15, 1 | seed);
            t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
            return ((t ^ t >>> 14) >>> 0) / 4294967296;
        };
    }

    /* ═══════════════════════════════════════════════════════════
       MPC via condensed QP (gradient projection)
       ═══════════════════════════════════════════════════════════ */

    /* Build condensed QP: min 0.5 U^T H U + f^T U   s.t. -u_max <= U_i <= u_max */
    function buildQP(A_K, B_K, z0, N, Q, R, Qf) {
        var p = p_lift;

        // Precompute powers of A: A^0, A^1, ..., A^N
        var Apow = []; // each p×p column-major
        var I_p = new Float64Array(p * p);
        for (var i = 0; i < p; i++) I_p[i*p+i] = 1;
        Apow.push(I_p);
        for (var k = 1; k <= N; k++)
            Apow.push(matMul(A_K, Apow[k-1], p, p, p));

        // Sx (prediction matrix for free response): [(A), (A^2), ..., (A^N)]^T  (N*p × p)
        // Su (prediction matrix for forced response): lower-triangular block Toeplitz

        // H = Su^T * blkdiag(Q,...,Q,Qf) * Su + blkdiag(R,...,R)
        // f = Su^T * blkdiag(Q,...,Q,Qf) * Sx * z0

        // For efficiency, build H and f directly
        // H_ij = sum_{k=max(i,j)}^{N-1} B^T (A^{k-i})^T Q_k (A^{k-j}) B  +  R * delta_ij
        // where Q_k = Q for k<N, Qf for k=N

        // Actually build using the G matrix: G[k][j] = A^{k-j} B for j<=k
        var GB = []; // GB[k] = A^k * B  (p×1)
        for (var k = 0; k <= N; k++) {
            var ab = new Float64Array(p);
            for (var i = 0; i < p; i++) {
                var sum = 0;
                for (var q = 0; q < p; q++) sum += Apow[k][q*p+i] * B_K[q];
                ab[i] = sum;
            }
            GB.push(ab);
        }

        // H (N × N)
        var H = new Float64Array(N * N);
        for (var i = 0; i < N; i++) {
            for (var j = i; j < N; j++) {
                var sum = 0;
                // sum over prediction steps k = j+1 to N
                for (var k = j + 1; k <= N; k++) {
                    var Qw = (k < N) ? Q : Qf;
                    // GB[k-i-1]^T * Qw * GB[k-j-1]
                    var gi = GB[k - i - 1];
                    var gj = GB[k - j - 1];
                    for (var q = 0; q < p; q++)
                        sum += gi[q] * Qw[q] * gj[q]; // Q is diagonal
                }
                if (i === j) sum += R;
                H[j*N+i] = sum;
                H[i*N+j] = sum;
            }
        }

        // f (N × 1)
        // f_i = sum_{k=i+1}^{N} GB[k-i-1]^T * Qw * (A^k * z0)
        // Precompute A^k * z0
        var Az0 = [z0.slice()];
        for (var k = 1; k <= N; k++) {
            var v = new Float64Array(p);
            for (var i = 0; i < p; i++) {
                var sum = 0;
                for (var q = 0; q < p; q++) sum += Apow[k][q*p+i] * z0[q];
                v[i] = sum;
            }
            Az0.push(v);
        }

        var f = new Float64Array(N);
        for (var i = 0; i < N; i++) {
            var sum = 0;
            for (var k = i + 1; k <= N; k++) {
                var Qw = (k < N) ? Q : Qf;
                var gi = GB[k - i - 1];
                var az = Az0[k];
                for (var q = 0; q < p; q++) sum += gi[q] * Qw[q] * az[q];
            }
            f[i] = sum;
        }

        return { H: H, f: f, N: N };
    }

    /* ── Solve box-constrained QP via projected gradient ── */
    function solveBoxQP(H, f, N, umin, umax, maxIter) {
        var U = new Float64Array(N);
        var grad = new Float64Array(N);
        var alpha = 0;

        // Estimate step size from H diagonal
        var maxDiag = 0;
        for (var i = 0; i < N; i++) if (H[i*N+i] > maxDiag) maxDiag = H[i*N+i];
        alpha = 1.0 / (maxDiag + 1e-8);

        for (var iter = 0; iter < (maxIter || 200); iter++) {
            // grad = H*U + f
            for (var i = 0; i < N; i++) {
                var s = f[i];
                for (var j = 0; j < N; j++) s += H[j*N+i] * U[j];
                grad[i] = s;
            }
            // projected gradient step
            var changed = false;
            for (var i = 0; i < N; i++) {
                var old = U[i];
                U[i] = clamp(U[i] - alpha * grad[i], umin, umax);
                if (Math.abs(U[i] - old) > 1e-8) changed = true;
            }
            if (!changed) break;
        }
        return U;
    }

    /* ═══════════════════════════════════════════════════════════
       Koopman MPC Controller
       ═══════════════════════════════════════════════════════════ */
    function createController(koopman) {
        var localQ = [
            [15, 0, 0, 0],
            [0, 2, 0, 0],
            [0, 0, 120, 0],
            [0, 0, 0, 12]
        ];
        var localR = [[0.3]];
        var lin = linearizeCartpoleStep();
        var K_lqr = computeDiscreteLQRGain(lin.A, lin.B, localQ, localR);

        function lqrControl(state) {
            var e = [state[0], state[1], wrapAngle(state[2]), state[3]];
            var u = 0;
            for (var i = 0; i < 4; i++) u -= K_lqr[i] * e[i];
            return clamp(u, -u_max, u_max);
        }

        function swingUpControl(state) {
            var x = state[0], xd = state[1], th = state[2], thd = state[3];
            var inertia = mp * lp * lp;
            var energy = 0.5 * inertia * thd * thd + mp * G * lp * (Math.cos(th) - 1);
            var swing = 35 * energy * Math.sign(thd * Math.cos(th) + 1e-4);
            var centre = -1.0 * x - 2.0 * xd;
            return clamp(swing + centre, -u_max, u_max);
        }

        return function(state) {
            var th = wrapAngle(state[2]);
            if (Math.abs(th) < 0.35 && Math.abs(state[3]) < 2.0)
                return lqrControl(state);
            return swingUpControl(state);
        };
    }

    /* ═══════════════════════════════════════════════════════════
       Trajectory simulation
       ═══════════════════════════════════════════════════════════ */
    function simulateTrajectory(controller, s0) {
        var N_sim = Math.round(T_final / Tc);
        var states = [s0.slice()];
        var controls = [];
        var times = [0];

        var s = s0.slice();
        for (var k = 0; k < N_sim; k++) {
            s[2] = wrapAngle(s[2]);
            var u = controller(s);
            u = clamp(u, -u_max, u_max);
            controls.push(u);
            s = simStep(s, u);
            states.push(s.slice());
            times.push((k + 1) * Tc);
        }
        return { states: states, controls: controls, times: times, N: N_sim };
    }

    /* ═══════════════════════════════════════════════════════════
       HiDPI canvas helper
       ═══════════════════════════════════════════════════════════ */
    function setupCanvas(canvas) {
        var dpr = window.devicePixelRatio || 1;
        var rect = canvas.getBoundingClientRect();
        var w = rect.width, h = rect.height;
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        var ctx = canvas.getContext('2d');
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return { ctx: ctx, w: w, h: h };
    }

    /* ═══════════════════════════════════════════════════════════
       Drawing: cart-pole scene
       ═══════════════════════════════════════════════════════════ */
    function drawCartPole(canvas, state, u_val, trail) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0F1923';
        ctx.fillRect(0, 0, W, H);

        var x = state[0], th = state[2];
        var scale = W / (2 * x_max + 1.5);
        var cartW = 60, cartH = 30;
        var rodLen = lp * 2 * scale * 0.7;

        var groundY = H * 0.65;
        var cartCX = W / 2 + x * scale;
        var cartCY = groundY - cartH / 2 - 8;

        // Track
        ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(W/2 - x_max * scale, groundY);
        ctx.lineTo(W/2 + x_max * scale, groundY);
        ctx.stroke();

        // Track limits
        ctx.strokeStyle = 'rgba(231,76,60,0.3)';
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(W/2 - x_max * scale, groundY - 60);
        ctx.lineTo(W/2 - x_max * scale, groundY + 5);
        ctx.moveTo(W/2 + x_max * scale, groundY - 60);
        ctx.lineTo(W/2 + x_max * scale, groundY + 5);
        ctx.stroke();
        ctx.setLineDash([]);

        // Pendulum trail
        if (trail && trail.length > 1) {
            ctx.lineWidth = 1;
            for (var i = 1; i < trail.length; i++) {
                var alpha = 0.03 + 0.12 * (i / trail.length);
                ctx.fillStyle = 'rgba(100,180,255,' + alpha.toFixed(3) + ')';
                ctx.beginPath();
                ctx.arc(trail[i][0], trail[i][1], 1.5, 0, 2 * Math.PI);
                ctx.fill();
            }
        }

        // Cart body
        var grad = ctx.createLinearGradient(cartCX - cartW/2, cartCY - cartH/2, cartCX - cartW/2, cartCY + cartH/2);
        grad.addColorStop(0, '#3a5068');
        grad.addColorStop(1, '#263848');
        ctx.fillStyle = grad;
        ctx.strokeStyle = 'rgba(255,255,255,0.15)';
        ctx.lineWidth = 1;
        var cr = 4;
        ctx.beginPath();
        ctx.moveTo(cartCX - cartW/2 + cr, cartCY - cartH/2);
        ctx.lineTo(cartCX + cartW/2 - cr, cartCY - cartH/2);
        ctx.quadraticCurveTo(cartCX + cartW/2, cartCY - cartH/2, cartCX + cartW/2, cartCY - cartH/2 + cr);
        ctx.lineTo(cartCX + cartW/2, cartCY + cartH/2 - cr);
        ctx.quadraticCurveTo(cartCX + cartW/2, cartCY + cartH/2, cartCX + cartW/2 - cr, cartCY + cartH/2);
        ctx.lineTo(cartCX - cartW/2 + cr, cartCY + cartH/2);
        ctx.quadraticCurveTo(cartCX - cartW/2, cartCY + cartH/2, cartCX - cartW/2, cartCY + cartH/2 - cr);
        ctx.lineTo(cartCX - cartW/2, cartCY - cartH/2 + cr);
        ctx.quadraticCurveTo(cartCX - cartW/2, cartCY - cartH/2, cartCX - cartW/2 + cr, cartCY - cartH/2);
        ctx.closePath();
        ctx.fill();
        ctx.stroke();

        // Wheels
        ctx.fillStyle = '#556677';
        ctx.beginPath();
        ctx.arc(cartCX - cartW/3, groundY - 3, 6, 0, 2 * Math.PI); ctx.fill();
        ctx.beginPath();
        ctx.arc(cartCX + cartW/3, groundY - 3, 6, 0, 2 * Math.PI); ctx.fill();

        // Force arrow
        if (Math.abs(u_val) > 0.5) {
            var arrowLen = (u_val / u_max) * 40;
            ctx.strokeStyle = '#f39c12';
            ctx.lineWidth = 3;
            ctx.beginPath();
            ctx.moveTo(cartCX, cartCY);
            ctx.lineTo(cartCX + arrowLen, cartCY);
            ctx.stroke();
            // Arrowhead
            var dir = u_val > 0 ? 1 : -1;
            ctx.fillStyle = '#f39c12';
            ctx.beginPath();
            ctx.moveTo(cartCX + arrowLen, cartCY);
            ctx.lineTo(cartCX + arrowLen - dir * 8, cartCY - 5);
            ctx.lineTo(cartCX + arrowLen - dir * 8, cartCY + 5);
            ctx.closePath();
            ctx.fill();
        }

        // Pole (theta=0 is upright)
        var pivotX = cartCX;
        var pivotY = cartCY - cartH / 2;
        var bobX = pivotX + rodLen * Math.sin(th);
        var bobY = pivotY - rodLen * Math.cos(th);

        // Ghost upright reference
        ctx.setLineDash([3, 4]);
        ctx.strokeStyle = 'rgba(46,204,113,0.15)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pivotX, pivotY);
        ctx.lineTo(pivotX, pivotY - rodLen);
        ctx.stroke();
        ctx.setLineDash([]);

        // Rod
        ctx.strokeStyle = 'rgba(200,210,220,0.9)';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(pivotX, pivotY);
        ctx.lineTo(bobX, bobY);
        ctx.stroke();

        // Pivot
        ctx.fillStyle = '#667788';
        ctx.beginPath();
        ctx.arc(pivotX, pivotY, 4, 0, 2 * Math.PI);
        ctx.fill();

        // Bob
        var absAngle = Math.abs(wrapAngle(th));
        var upright = absAngle < 0.15;
        var bColor = upright ? [46, 204, 113] : (absAngle < 1.0 ? [52, 152, 219] : [231, 76, 60]);
        var bobR = 10;
        var bgrad = ctx.createRadialGradient(bobX - 2, bobY - 2, 2, bobX, bobY, bobR);
        bgrad.addColorStop(0, 'rgba(' + Math.min(bColor[0]+60,255) + ',' + Math.min(bColor[1]+60,255) + ',' + Math.min(bColor[2]+60,255) + ',1)');
        bgrad.addColorStop(0.7, 'rgb(' + bColor[0] + ',' + bColor[1] + ',' + bColor[2] + ')');
        bgrad.addColorStop(1, 'rgba(' + Math.max(bColor[0]-40,0) + ',' + Math.max(bColor[1]-40,0) + ',' + Math.max(bColor[2]-40,0) + ',1)');
        ctx.fillStyle = bgrad;
        ctx.beginPath();
        ctx.arc(bobX, bobY, bobR, 0, 2 * Math.PI);
        ctx.fill();

        // Labels
        ctx.fillStyle = '#99AABB';
        ctx.font = '10px "Source Sans 3", sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText('x = ' + x.toFixed(2) + ' m', 10, H - 42);
        ctx.fillText('\u03B8 = ' + (wrapAngle(th) * 180 / Math.PI).toFixed(1) + '\u00B0', 10, H - 28);
        ctx.fillText('u = ' + u_val.toFixed(1) + ' N', 10, H - 14);

        if (upright) {
            ctx.fillStyle = 'rgba(46,204,113,0.6)';
            ctx.font = 'bold 11px "Source Sans 3", sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('BALANCED', W/2, 25);
        }

        return [bobX, bobY];
    }

    /* ═══════════════════════════════════════════════════════════
       Drawing: plots
       ═══════════════════════════════════════════════════════════ */
    function drawMiniPlot(ctx, ox, oy, pw, ph, xs, ys, idx, label, color, yRef) {
        var mt = 14, mb = 4, ml = 42, mr = 6;
        var plotW = pw - ml - mr, plotH = ph - mt - mb;

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

        ctx.fillStyle = '#0A1219';
        ctx.fillRect(ox, oy, pw, ph);

        // Grid
        ctx.strokeStyle = 'rgba(255,255,255,0.05)';
        ctx.lineWidth = 0.5;
        for (var g = 0; g < 4; g++) {
            var yv = yMin + (yMax - yMin) * (g + 1) / 5;
            ctx.beginPath(); ctx.moveTo(ox + ml, sy(yv)); ctx.lineTo(ox + pw - mr, sy(yv)); ctx.stroke();
        }

        // Reference line
        if (yRef !== undefined && yRef !== null) {
            ctx.strokeStyle = 'rgba(46,204,113,0.3)';
            ctx.setLineDash([3, 3]);
            ctx.beginPath();
            ctx.moveTo(ox + ml, sy(yRef));
            ctx.lineTo(ox + pw - mr, sy(yRef));
            ctx.stroke();
            ctx.setLineDash([]);
        }

        // Data line
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        var drawN = Math.min(idx + 1, ys.length);
        for (var i = 0; i < drawN; i++) {
            var px = sx(xs[i]), py = sy(ys[i]);
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();

        // Current point
        if (idx < ys.length) {
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.arc(sx(xs[idx]), sy(ys[idx]), 3, 0, 2 * Math.PI);
            ctx.fill();
        }

        // Label
        ctx.fillStyle = '#778899';
        ctx.font = '10px "Source Sans 3",sans-serif';
        ctx.textAlign = 'left';
        ctx.fillText(label, ox + ml + 4, oy + mt - 3);

        // Y-axis ticks
        ctx.fillStyle = '#556677';
        ctx.font = '8px "SF Mono","Fira Code",monospace';
        ctx.textAlign = 'right';
        ctx.fillText(yMax.toFixed(1), ox + ml - 3, oy + mt + 7);
        ctx.fillText(yMin.toFixed(1), ox + ml - 3, oy + ph - mb);
    }

    function drawPlots(canvas, traj, idx) {
        var r = setupCanvas(canvas), ctx = r.ctx, W = r.w, H = r.h;
        ctx.fillStyle = '#0F1923';
        ctx.fillRect(0, 0, W, H);

        var nPlots = 4;
        var gap = 2;
        var pH = (H - (nPlots - 1) * gap) / nPlots;

        var ts = traj.times;
        var xs = [], xds = [], ths = [], us = [];
        for (var i = 0; i < traj.states.length; i++) {
            xs.push(traj.states[i][0]);
            xds.push(traj.states[i][1]);
            ths.push(wrapAngle(traj.states[i][2]) * 180 / Math.PI);
        }
        for (var i = 0; i < traj.controls.length; i++) us.push(traj.controls[i]);

        drawMiniPlot(ctx, 0, 0, W, pH, ts, xs, idx, 'x (m)', '#e74c3c', 0);
        drawMiniPlot(ctx, 0, pH + gap, W, pH, ts, xds, idx, 'x\u0307 (m/s)', '#9b59b6', 0);
        drawMiniPlot(ctx, 0, 2 * (pH + gap), W, pH, ts, ths, idx, '\u03B8 (\u00B0)', '#3498db', 0);

        var ctrlTs = ts.slice(0, us.length);
        drawMiniPlot(ctx, 0, 3 * (pH + gap), W, pH, ctrlTs, us, Math.min(idx, us.length - 1),
            'u (N)', '#f39c12', 0);

        ctx.fillStyle = '#667788';
        ctx.font = '9px "Source Sans 3",sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Time (s)', W / 2, H - 2);
    }

    /* ═══════════════════════════════════════════════════════════
       Widget builder
       ═══════════════════════════════════════════════════════════ */
    function initDemo(container) {
        if (container.dataset.initialized) return;
        container.dataset.initialized = '1';

        /* ── Inject CSS ── */
        var styleId = 'koopman-cartpole-styles';
        if (!document.getElementById(styleId)) {
            var style = document.createElement('style');
            style.id = styleId;
            style.textContent =
                '.koopman-cartpole-demo{' +
                    'background:#0F1923;border-radius:8px;overflow:hidden;' +
                    'font-family:"Source Sans 3","Segoe UI",-apple-system,sans-serif;' +
                    'color:#C8D6E5;margin:1.5rem 0;border:1px solid rgba(255,255,255,0.08);' +
                '}' +
                '.kcp-header{padding:12px 18px;background:rgba(255,255,255,0.03);border-bottom:1px solid rgba(255,255,255,0.06);}' +
                '.kcp-title{font-family:"Crimson Text",Georgia,serif;font-size:1.1rem;font-weight:600;color:#D4E0ED;}' +
                '.kcp-info{padding:8px 18px;font-size:0.78rem;color:#778899;background:rgba(0,0,0,0.15);border-bottom:1px solid rgba(255,255,255,0.04);}' +
                '.kcp-training{padding:12px 18px;font-size:0.82rem;color:#f39c12;background:rgba(243,156,18,0.08);border-bottom:1px solid rgba(255,255,255,0.04);display:none;}' +
                '.kcp-training.active{display:block;}' +
                '.kcp-controls{padding:10px 18px;display:flex;align-items:center;flex-wrap:wrap;gap:12px;background:rgba(255,255,255,0.02);border-bottom:1px solid rgba(255,255,255,0.06);}' +
                '.kcp-btn{padding:5px 14px;font-family:"Source Sans 3",sans-serif;font-size:0.8rem;font-weight:500;color:#C8D6E5;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.1);border-radius:4px;cursor:pointer;transition:all 0.15s;}' +
                '.kcp-btn:hover{background:rgba(255,255,255,0.12);color:#fff;}' +
                '.kcp-btn.kcp-primary{background:rgba(46,80,144,0.5);border-color:rgba(46,80,144,0.7);}' +
                '.kcp-btn.kcp-primary:hover{background:rgba(46,80,144,0.7);}' +
                '.kcp-btn.kcp-active{background:rgba(231,76,60,0.3);border-color:rgba(231,76,60,0.5);}' +
                '.kcp-btn.kcp-disturb{background:rgba(243,156,18,0.25);border-color:rgba(243,156,18,0.45);}' +
                '.kcp-btn.kcp-disturb:hover{background:rgba(243,156,18,0.45);}' +
                '.kcp-slider-label{display:flex;align-items:center;gap:8px;font-size:0.78rem;color:#99AABB;}' +
                '.kcp-slider{-webkit-appearance:none;appearance:none;width:80px;height:4px;background:rgba(255,255,255,0.12);border-radius:2px;outline:none;}' +
                '.kcp-slider::-webkit-slider-thumb{-webkit-appearance:none;appearance:none;width:14px;height:14px;border-radius:50%;background:#3498db;cursor:pointer;}' +
                '.kcp-slider::-moz-range-thumb{width:14px;height:14px;border-radius:50%;background:#3498db;cursor:pointer;border:none;}' +
                '.kcp-slider-val{font-family:"SF Mono","Fira Code",monospace;font-size:0.75rem;color:#778899;min-width:30px;}' +
                '.kcp-body{display:flex;gap:0;}' +
                '.kcp-scene-wrap{flex:1;min-width:0;}' +
                '.kcp-scene-canvas{display:block;width:100%;height:320px;}' +
                '.kcp-plot-wrap{flex:1;min-width:0;border-left:1px solid rgba(255,255,255,0.06);}' +
                '.kcp-plot-canvas{display:block;width:100%;height:320px;}' +
                '.kcp-readout{padding:8px 18px;display:flex;gap:18px;flex-wrap:wrap;font-size:0.78rem;color:#778899;background:rgba(0,0,0,0.15);border-top:1px solid rgba(255,255,255,0.04);}' +
                '.kcp-ro-item{font-family:"SF Mono","Fira Code",monospace;font-size:0.72rem;}' +
                '.kcp-ro-item span{color:#C8D6E5;}' +
                '.kcp-status{margin-left:auto;font-family:"Source Sans 3",sans-serif;font-weight:500;}' +
                '.kcp-status.kcp-training-s{color:#f39c12;}' +
                '.kcp-status.kcp-running{color:#3498db;}' +
                '.kcp-status.kcp-balanced{color:#2ecc71;}' +
                '@media(max-width:640px){' +
                    '.kcp-body{flex-direction:column;}' +
                    '.kcp-plot-wrap{border-left:none;border-top:1px solid rgba(255,255,255,0.06);}' +
                    '.kcp-controls{flex-direction:column;align-items:flex-start;}' +
                '}';
            document.head.appendChild(style);
        }

        /* ── Build DOM ── */
        container.innerHTML =
            '<div class="kcp-header">' +
                '<span class="kcp-title">\u25B6 Cart-Pole Swing-Up \u2014 Learned Model + Hybrid Control</span>' +
            '</div>' +
            '<div class="kcp-info">' +
                'Cart-pole: m<sub>c</sub>=1.0, m<sub>p</sub>=0.3, l=0.5 m ' +
                '&nbsp;|&nbsp; Dictionary: \u03C8(x) = [x, \u1E8B, \u03B8, \u03B8\u0307, sin\u03B8, cos\u03B8, \u1E8B sin\u03B8, \u1E8B cos\u03B8, \u03B8\u0307 sin\u03B8, \u03B8\u0307 cos\u03B8]' +
                '&nbsp;|&nbsp; Hybrid swing-up + local stabilisation' +
                '&nbsp;|&nbsp; |u| \u2264 15 N' +
            '</div>' +
            '<div class="kcp-training" id="kcp-training-bar">Training EDMD model from 5000 data pairs...</div>' +
            '<div class="kcp-controls">' +
                '<div style="display:flex;gap:6px;align-items:center;">' +
                    '<button class="kcp-btn kcp-primary" id="kcp-run-btn">Run</button>' +
                    '<button class="kcp-btn" id="kcp-reset-btn">Reset</button>' +
                    '<button class="kcp-btn kcp-disturb" id="kcp-kick-btn">Kick</button>' +
                '</div>' +
                '<label class="kcp-slider-label">\u03B8\u2080' +
                    '<input type="range" class="kcp-slider" id="kcp-theta0" min="-180" max="180" value="180" step="5">' +
                    '<span class="kcp-slider-val" id="kcp-theta0-val">180\u00B0</span>' +
                '</label>' +
                '<label class="kcp-slider-label">Speed' +
                    '<input type="range" class="kcp-slider" id="kcp-speed" min="1" max="5" value="2" step="1">' +
                    '<span class="kcp-slider-val" id="kcp-speed-val">2\u00D7</span>' +
                '</label>' +
            '</div>' +
            '<div class="kcp-body">' +
                '<div class="kcp-scene-wrap"><canvas class="kcp-scene-canvas" id="kcp-scene"></canvas></div>' +
                '<div class="kcp-plot-wrap"><canvas class="kcp-plot-canvas" id="kcp-plots"></canvas></div>' +
            '</div>' +
            '<div class="kcp-readout">' +
                '<div class="kcp-ro-item">t = <span id="kcp-t">0.00</span> s</div>' +
                '<div class="kcp-ro-item">x = <span id="kcp-x">0.00</span></div>' +
                '<div class="kcp-ro-item">\u03B8 = <span id="kcp-th">180.0</span>\u00B0</div>' +
                '<div class="kcp-ro-item">u = <span id="kcp-u">0.0</span> N</div>' +
                '<div class="kcp-status" id="kcp-stat">Ready</div>' +
            '</div>';

        /* ── Element references ── */
        var sceneCanvas = container.querySelector('#kcp-scene');
        var plotCanvas  = container.querySelector('#kcp-plots');
        var runBtn      = container.querySelector('#kcp-run-btn');
        var resetBtn    = container.querySelector('#kcp-reset-btn');
        var kickBtn     = container.querySelector('#kcp-kick-btn');
        var theta0Sl    = container.querySelector('#kcp-theta0');
        var theta0Val   = container.querySelector('#kcp-theta0-val');
        var speedSl     = container.querySelector('#kcp-speed');
        var speedVal    = container.querySelector('#kcp-speed-val');
        var elT         = container.querySelector('#kcp-t');
        var elX         = container.querySelector('#kcp-x');
        var elTh        = container.querySelector('#kcp-th');
        var elU         = container.querySelector('#kcp-u');
        var elStat      = container.querySelector('#kcp-stat');
        var trainBar    = container.querySelector('#kcp-training-bar');

        /* ── State ── */
        var koopman = null;
        var controller = null;
        var traj = null;
        var animIdx = 0;
        var animId = null;
        var running = false;
        var trail = [];
        var trained = false;

        function getInitState() {
            var th0 = parseFloat(theta0Sl.value) * Math.PI / 180;
            return [0, 0, th0, 0];
        }

        function doReset() {
            if (animId) { cancelAnimationFrame(animId); animId = null; }
            running = false;
            runBtn.textContent = 'Run';
            runBtn.classList.remove('kcp-active');
            runBtn.classList.add('kcp-primary');
            animIdx = 0;
            trail = [];
            traj = null;
            var s0 = getInitState();
            drawCartPole(sceneCanvas, s0, 0, trail);
            // Clear plots
            var r = setupCanvas(plotCanvas), ctx = r.ctx;
            ctx.fillStyle = '#0F1923';
            ctx.fillRect(0, 0, r.w, r.h);
            ctx.fillStyle = '#556677';
            ctx.font = '12px "Source Sans 3",sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText('Press Run to start simulation', r.w/2, r.h/2);
            elT.textContent = '0.00';
            elX.textContent = '0.00';
            elTh.textContent = (parseFloat(theta0Sl.value)).toFixed(1);
            elU.textContent = '0.0';
            elStat.textContent = trained ? 'Ready' : 'Training...';
            elStat.className = 'kcp-status';
        }

        function doTrain() {
            trainBar.classList.add('active');
            elStat.textContent = 'Training EDMD...';
            elStat.className = 'kcp-status kcp-training-s';

            setTimeout(function() {
                koopman = trainEDMD();
                controller = createController(koopman);
                trained = true;
                trainBar.classList.remove('active');
                elStat.textContent = 'Ready';
                elStat.className = 'kcp-status';
                doReset();
            }, 50);
        }

        function doRun() {
            if (!trained) return;
            if (running) {
                // Pause
                if (animId) { cancelAnimationFrame(animId); animId = null; }
                running = false;
                runBtn.textContent = 'Resume';
                runBtn.classList.remove('kcp-active');
                runBtn.classList.add('kcp-primary');
                return;
            }

            if (!traj || animIdx >= traj.N) {
                // New simulation
                elStat.textContent = 'Computing...';
                elStat.className = 'kcp-status kcp-training-s';
                trail = [];
                animIdx = 0;

                setTimeout(function() {
                    var s0 = getInitState();
                    traj = simulateTrajectory(controller, s0);
                    elStat.textContent = 'Running';
                    elStat.className = 'kcp-status kcp-running';
                    running = true;
                    runBtn.textContent = 'Pause';
                    runBtn.classList.remove('kcp-primary');
                    runBtn.classList.add('kcp-active');
                    animate();
                }, 50);
            } else {
                // Resume
                running = true;
                runBtn.textContent = 'Pause';
                runBtn.classList.remove('kcp-primary');
                runBtn.classList.add('kcp-active');
                elStat.textContent = 'Running';
                elStat.className = 'kcp-status kcp-running';
                animate();
            }
        }

        function doKick() {
            if (!traj || !running) return;
            // Apply a disturbance: add angular velocity
            var kickIdx = animIdx;
            if (kickIdx < traj.states.length) {
                // Re-simulate from current state with a kick
                var s = traj.states[kickIdx].slice();
                s[3] += 5.0; // angular velocity kick
                s[1] += 2.0; // cart velocity kick

                // Re-simulate
                var newStates = [s.slice()];
                var newControls = [];
                var newTimes = [traj.times[kickIdx]];
                var N_remain = traj.N - kickIdx;
                for (var k = 0; k < N_remain; k++) {
                    s[2] = wrapAngle(s[2]);
                    var u = controller(s);
                    u = clamp(u, -u_max, u_max);
                    newControls.push(u);
                    s = simStep(s, u);
                    newStates.push(s.slice());
                    newTimes.push(traj.times[kickIdx] + (k + 1) * Tc);
                }

                // Splice into trajectory
                traj.states = traj.states.slice(0, kickIdx).concat(newStates);
                traj.controls = traj.controls.slice(0, kickIdx).concat(newControls);
                traj.times = traj.times.slice(0, kickIdx).concat(newTimes);
                traj.N = traj.controls.length;
            }
        }

        function animate() {
            if (!running || !traj) return;

            var speed = parseInt(speedSl.value);
            for (var rep = 0; rep < speed; rep++) {
                if (animIdx >= traj.N) break;

                var s = traj.states[animIdx];
                var u = traj.controls[animIdx];

                var bobPos = drawCartPole(sceneCanvas, s, u, trail);
                trail.push(bobPos);
                if (trail.length > 200) trail.shift();

                drawPlots(plotCanvas, traj, animIdx);

                elT.textContent = traj.times[animIdx].toFixed(2);
                elX.textContent = s[0].toFixed(2);
                elTh.textContent = (wrapAngle(s[2]) * 180 / Math.PI).toFixed(1);
                elU.textContent = u.toFixed(1);

                var absAngle = Math.abs(wrapAngle(s[2]));
                if (absAngle < 0.15 && Math.abs(s[3]) < 0.5) {
                    elStat.textContent = 'Balanced!';
                    elStat.className = 'kcp-status kcp-balanced';
                }

                animIdx++;
            }

            if (animIdx >= traj.N) {
                running = false;
                runBtn.textContent = 'Run';
                runBtn.classList.remove('kcp-active');
                runBtn.classList.add('kcp-primary');
                if (elStat.textContent !== 'Balanced!') {
                    elStat.textContent = 'Done';
                    elStat.className = 'kcp-status';
                }
                return;
            }

            animId = requestAnimationFrame(animate);
        }

        /* ── Event handlers ── */
        runBtn.addEventListener('click', doRun);
        resetBtn.addEventListener('click', doReset);
        kickBtn.addEventListener('click', doKick);

        theta0Sl.addEventListener('input', function() {
            theta0Val.textContent = theta0Sl.value + '\u00B0';
            if (!running) doReset();
        });
        speedSl.addEventListener('input', function() {
            speedVal.textContent = speedSl.value + '\u00D7';
        });

        /* ── Initial training ── */
        doTrain();
    }

    /* ── Auto-init ── */
    function initAllDemos(root) {
        var els = (root || document).querySelectorAll('.koopman-cartpole-demo');
        for (var i = 0; i < els.length; i++) initDemo(els[i]);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function () { initAllDemos(); });
    } else {
        initAllDemos();
    }

    window.KoopmanCartpoleDemo = { initAll: initAllDemos, init: initDemo };
})();
