/**
 * Interactive 3D Quadrotor NMPC Demo — SCvx (Successive Convexification)
 * Nonlinear quadrotor dynamics with RK4 integration.
 * Adjoint gradient + HVP for exact QP step sizes.
 *
 * HTML hook:  <div class="quadrotor-demo"></div>
 */
(function () {
    'use strict';

    var NX = 6, NU = 3;
    // State: [px, py, pz, vx, vy, vz]
    // Input: [roll(phi), pitch(theta), thrust(T)]

    function clamp(v, lo, hi) { return v < lo ? lo : v > hi ? hi : v; }

    /* ═══ Continuous-time quadrotor dynamics ═══════════════════
       vdot_x =  (T/m) cos(phi) sin(theta)
       vdot_y = -(T/m) sin(phi)
       vdot_z =  (T/m) cos(phi) cos(theta) - g               */
    function fCont(x, u, p) {
        var phi = u[0], theta = u[1], T = u[2];
        var cp = Math.cos(phi), sp = Math.sin(phi);
        var ct = Math.cos(theta), st = Math.sin(theta);
        var Tm = T / p.m;
        return [
            x[3], x[4], x[5],
            Tm * cp * st,
            -Tm * sp,
            Tm * cp * ct - p.g
        ];
    }

    /* ═══ RK4 integrator ═══════════════════════════════════════ */
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

    /* ═══ Numerical Jacobians of the RK4 map ══════════════════ */
    function rk4Jac(x, u, p) {
        var eps = 1e-6, f0 = rk4(x, u, p);
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

    /* ═══ Cost and solver constants ════════════════════════════ */
    var GROUND_PEN = 3000;
    var SCVX_QP_ITERS = 20;
    var SCVX_SCP_ITERS = 5;
    var SCVX_TR = 1.5;

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
    function scvxSolve(x0, usInit, xRefs, uRef, Qd, Rd, Qfd, p, nSCP, wTR) {
        var N = usInit.length;
        var us = [];
        for (var k = 0; k < N; k++) us.push(usInit[k].slice());

        for (var scp = 0; scp < nSCP; scp++) {
            var xsBar = [x0.slice()];
            for (var k = 0; k < N; k++) xsBar.push(rk4(xsBar[k], us[k], p));

            var Ab = [], Bb = [], cb = [];
            for (var k = 0; k < N; k++) {
                var jac = rk4Jac(xsBar[k], us[k], p);
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

            /* Diagonal preconditioner */
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
                /* Project gradient at active bounds */
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
        for (var k = 0; k < N; k++) xs.push(rk4(xs[k], us[k], p));
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

    /* ═══ 3D Projection (orthographic with azimuth + elevation) */
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

    /* Unproject screen click to ground plane (z=0) */
    function unproject(sx, sy, cam) {
        var px = (sx - cam.cx) / cam.sc;
        var py = -(sy - cam.cy) / cam.sc;
        var se = Math.sin(cam.el), ce = Math.cos(cam.el);
        if (Math.abs(se) < 0.01) return null;
        var y1 = py / se;
        var ca = Math.cos(cam.az), sa = Math.sin(cam.az);
        return { x: px * ca + y1 * sa, y: -px * sa + y1 * ca };
    }

    /* Rotate body-frame point by roll(phi) and pitch(theta) */
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
        /* Shadow on ground */
        var gs = proj(target[0], target[1], 0, cam);
        ctx.beginPath(); ctx.arc(gs.x, gs.y, 6, 0, 2*Math.PI);
        ctx.fillStyle = 'rgba(255,165,0,0.3)'; ctx.fill();
        /* Vertical line */
        var ts = proj(target[0], target[1], target[2], cam);
        ctx.strokeStyle = 'rgba(255,165,0,0.4)'; ctx.lineWidth = 1;
        ctx.setLineDash([4,4]); ctx.beginPath(); ctx.moveTo(gs.x, gs.y); ctx.lineTo(ts.x, ts.y); ctx.stroke(); ctx.setLineDash([]);
        /* Target marker */
        ctx.beginPath(); ctx.arc(ts.x, ts.y, 8, 0, 2*Math.PI);
        ctx.fillStyle = 'rgba(255,165,0,0.7)'; ctx.fill();
        ctx.strokeStyle = '#FFA500'; ctx.lineWidth = 2; ctx.stroke();
        /* Cross */
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(ts.x-6,ts.y); ctx.lineTo(ts.x+6,ts.y); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(ts.x,ts.y-6); ctx.lineTo(ts.x,ts.y+6); ctx.stroke();
    }

    function drawQuadrotor(ctx, pos, phi, theta, cam, armLen) {
        var L = armLen || 0.35;
        /* 4 arm tips in body frame (+ config) */
        var arms = [[L,0,0],[0,L,0],[-L,0,0],[0,-L,0]];
        var colors = ['#FF4444','#4488FF','#AAAAAA','#4488FF']; // front=red, others=blue/grey
        var rotArms = [], scrArms = [];
        for (var i = 0; i < 4; i++) {
            var r = rotBody(arms[i][0], arms[i][1], arms[i][2], phi, theta);
            rotArms.push(r);
            scrArms.push(proj(pos[0]+r[0], pos[1]+r[1], pos[2]+r[2], cam));
        }
        var center = proj(pos[0], pos[1], pos[2], cam);

        /* Shadow on ground */
        ctx.fillStyle = 'rgba(0,0,0,0.25)';
        var gs = proj(pos[0], pos[1], 0, cam);
        ctx.beginPath(); ctx.arc(gs.x, gs.y, 5, 0, 2*Math.PI); ctx.fill();

        /* Draw arms */
        ctx.lineWidth = 3;
        for (var i = 0; i < 4; i++) {
            ctx.strokeStyle = colors[i];
            ctx.beginPath(); ctx.moveTo(center.x, center.y); ctx.lineTo(scrArms[i].x, scrArms[i].y); ctx.stroke();
        }
        /* Rotor discs */
        var rotorR = 10;
        for (var i = 0; i < 4; i++) {
            ctx.beginPath(); ctx.arc(scrArms[i].x, scrArms[i].y, rotorR, 0, 2*Math.PI);
            ctx.fillStyle = colors[i] + '44'; ctx.fill();
            ctx.strokeStyle = colors[i]; ctx.lineWidth = 1.5; ctx.stroke();
        }
        /* Center body */
        ctx.beginPath(); ctx.arc(center.x, center.y, 4, 0, 2*Math.PI);
        ctx.fillStyle = '#FFFFFF'; ctx.fill();
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

        var cam = { az: sim.camAz, el: sim.camEl, cx: W*0.5, cy: H*0.55, sc: Math.min(W,H)*0.07 };

        drawGround(ctx, W, H, cam);
        drawAxes(ctx, cam);
        drawTarget(ctx, sim.target, cam);
        drawTrail(ctx, sim.trail, cam);
        if (sim.pred) drawPrediction(ctx, sim.pred, cam);

        var phi = sim.lastU ? sim.lastU[0] : 0;
        var theta = sim.lastU ? sim.lastU[1] : 0;
        drawQuadrotor(ctx, sim.x, phi, theta, cam, 0.35);

        /* HUD */
        ctx.fillStyle = 'rgba(255,255,255,0.6)'; ctx.font = '11px monospace';
        ctx.fillText('pos: ('+sim.x[0].toFixed(2)+', '+sim.x[1].toFixed(2)+', '+sim.x[2].toFixed(2)+')', 10, 18);
        ctx.fillText('vel: ('+sim.x[3].toFixed(2)+', '+sim.x[4].toFixed(2)+', '+sim.x[5].toFixed(2)+')', 10, 32);
        if (sim.lastU) {
            ctx.fillText('\u03C6: '+(sim.lastU[0]*180/Math.PI).toFixed(1)+'\u00B0  \u03B8: '+(sim.lastU[1]*180/Math.PI).toFixed(1)+'\u00B0  T: '+sim.lastU[2].toFixed(1)+'N', 10, 46);
        }
        ctx.fillText('t = '+sim.time.toFixed(1)+'s  step '+sim.step, 10, 60);

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

        var pH = Math.floor(H / 3);
        drawOnePlot(ctx, 0, 0, W, pH, sim.log.t, [sim.log.pz], ['#6688FF'], ['z'], 'Altitude (m)', sim);
        drawOnePlot(ctx, 0, pH, W, pH, sim.log.t, [sim.log.px, sim.log.py], ['#FF6666','#66FF66'], ['x','y'], 'Position (m)', sim);
        drawOnePlot(ctx, 0, 2*pH, W, H-2*pH, sim.log.t, [sim.log.T], ['#FFaa44'], ['T'], 'Thrust (N)', sim);
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

        /* Target line for altitude */
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
            ctx.fillText(labels[s], pad.l+pw-12*(labels.length-s), pad.t+12);
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

        /* Slider state */
        function sliderVal(id) {
            var s = container.querySelector('[data-id="'+id+'"]');
            return s ? parseFloat(s.value) : 0;
        }

        /* Update displayed values */
        container.addEventListener('input', function(e) {
            var s = e.target;
            if (!s.classList.contains('qd-slider')) return;
            var v = s.nextElementSibling;
            if (v) v.textContent = parseFloat(s.value).toFixed(s.step < 1 ? 1 : 0);
            sim.target = [sliderVal('tx'), sliderVal('ty'), sliderVal('tz')];
            if (!running) render();
        });

        /* Physics parameters */
        var pp = { m: 1.0, g: 9.81, dt: 0.1,
                   phiMin: -0.5, phiMax: 0.5,
                   thMin: -0.5, thMax: 0.5,
                   Tmin: 0.0, Tmax: 20.0 };

        /* Simulation state */
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
            log: { t:[], px:[], py:[], pz:[], T:[] },
            usWarm: null
        };

        var running = false, animId = null;

        function getWeights() {
            var qp = sliderVal('qp'), rc = sliderVal('rc');
            return {
                Qd:  [qp, qp, qp*1.5, qp*0.3, qp*0.3, qp*0.3],
                Rd:  [rc, rc, rc*0.005],
                Qfd: [qp*5, qp*5, qp*7, qp*1.5, qp*1.5, qp*1.5],
                N: Math.round(sliderVal('hor'))
            };
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
            sim.log = { t:[0], px:[0], py:[0], pz:[2], T:[pp.m*pp.g] };
            sim.usWarm = null;
            startBtn.textContent = '\u25B6 Start';
            render();
        }

        function physicsStep() {
            var w = getWeights();
            var N = w.N;
            var uHover = [0, 0, pp.m * pp.g];
            var refs = genReference(sim.x, sim.target, N, pp.dt);

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

            var sol = scvxSolve(sim.x, usInit, refs, uHover, w.Qd, w.Rd, w.Qfd, pp, SCVX_SCP_ITERS, SCVX_TR);

            sim.lastU = sol.us[0].slice();
            sim.pred = sol.xs;
            sim.usWarm = sol.us;

            /* Apply first control via nonlinear RK4 dynamics */
            sim.x = rk4(sim.x, sim.lastU, pp);
            sim.x[2] = Math.max(sim.x[2], 0); // ground constraint
            sim.time += pp.dt;
            sim.step++;

            sim.trail.push([sim.x[0], sim.x[1], sim.x[2]]);
            if (sim.trail.length > 300) sim.trail.shift();

            sim.log.t.push(sim.time);
            sim.log.px.push(sim.x[0]);
            sim.log.py.push(sim.x[1]);
            sim.log.pz.push(sim.x[2]);
            sim.log.T.push(sim.lastU[2]);
            if (sim.log.t.length > 500) {
                sim.log.t.shift(); sim.log.px.shift(); sim.log.py.shift();
                sim.log.pz.shift(); sim.log.T.shift();
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
            var cam = { az: sim.camAz, el: sim.camEl, cx: W*0.5, cy: H*0.55, sc: Math.min(W,H)*0.07 };
            var gp = unproject(sx, sy, cam);
            if (gp && Math.abs(gp.x) <= 8 && Math.abs(gp.y) <= 8) {
                sim.target[0] = Math.round(gp.x * 2) / 2;
                sim.target[1] = Math.round(gp.y * 2) / 2;
                /* Update sliders */
                var stx = container.querySelector('[data-id="tx"]');
                var sty = container.querySelector('[data-id="ty"]');
                if (stx) { stx.value = sim.target[0]; stx.nextElementSibling.textContent = sim.target[0].toFixed(1); }
                if (sty) { sty.value = sim.target[1]; sty.nextElementSibling.textContent = sim.target[1].toFixed(1); }
                if (!running) render();
            }
        });

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
