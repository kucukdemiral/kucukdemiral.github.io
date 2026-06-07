(function () {
  "use strict";

  // =====================================================================
  //  CONSTANTS
  // =====================================================================

  const SVG_NS = "http://www.w3.org/2000/svg";
  const DEG = 180 / Math.PI;
  const RAD = Math.PI / 180;
  const SAMPLES = 800;
  const TIME_SAMPLES = 600;

  const COLORS = {
    uncomp:     "#94a3b8",
    compMag:    "#0f766e",
    compPhase:  "#1d4ed8",
    ctrl:       "#7c3aed",
    ctrlPhase:  "#c026d3",
    uncompTime: "#94a3b8",
    compTime:   "#0f766e",
    target:     "#dc2626",
    pm:         "#b45309",
    grid:       "rgba(148,163,184,0.32)",
    gridMinor:  "rgba(148,163,184,0.16)",
    plot:       "rgba(255,255,255,0.65)",
    ref:        "rgba(220,38,38,0.4)",
  };

  // =====================================================================
  //  COMPLEX ARITHMETIC
  // =====================================================================

  function C(re, im) { return { re, im: im || 0 }; }
  function Cadd(a, b) { return C(a.re + b.re, a.im + b.im); }
  function Csub(a, b) { return C(a.re - b.re, a.im - b.im); }
  function Cmul(a, b) { return C(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
  function Cdiv(a, b) {
    const d = b.re * b.re + b.im * b.im;
    return C((a.re * b.re + a.im * b.im) / d, (a.im * b.re - a.re * b.im) / d);
  }
  function Cabs(a) { return Math.hypot(a.re, a.im); }
  function Carg(a) { return Math.atan2(a.im, a.re); }

  // =====================================================================
  //  POLYNOMIAL UTILITIES
  // =====================================================================

  function parseCoeffs(text) {
    const cleaned = text.trim().replace(/^[[(]+|[\])]+$/g, "");
    if (!cleaned) throw new Error("Enter at least one coefficient.");
    const tokens = cleaned.split(/[\s,;]+/).filter(Boolean);
    const coeffs = tokens.map((t) => {
      const v = Number(t);
      if (!Number.isFinite(v)) throw new Error(`Invalid coefficient "${t}".`);
      return v;
    });
    let i = 0;
    while (i < coeffs.length && Math.abs(coeffs[i]) < 1e-14) i++;
    if (i >= coeffs.length) throw new Error("Polynomial cannot be identically zero.");
    return coeffs.slice(i);
  }

  function polyEval(coeffs, z) {
    let r = C(coeffs[0], 0);
    for (let i = 1; i < coeffs.length; i++) r = Cadd(Cmul(r, z), C(coeffs[i], 0));
    return r;
  }

  function polyMul(a, b) {
    const r = new Array(a.length + b.length - 1).fill(0);
    for (let i = 0; i < a.length; i++)
      for (let j = 0; j < b.length; j++) r[i + j] += a[i] * b[j];
    return r;
  }

  function polyAdd(a, b) {
    const n = Math.max(a.length, b.length);
    const r = new Array(n).fill(0);
    for (let i = 0; i < a.length; i++) r[n - a.length + i] += a[i];
    for (let i = 0; i < b.length; i++) r[n - b.length + i] += b[i];
    return r;
  }

  function polyScale(a, k) { return a.map((c) => c * k); }

  function systemType(den) {
    let t = 0;
    for (let i = den.length - 1; i >= 0; i--) {
      if (Math.abs(den[i]) < 1e-12) t++;
      else break;
    }
    return t;
  }

  function formatPoly(coeffs) {
    if (!coeffs.length) return "0";
    const deg = coeffs.length - 1;
    const parts = [];
    coeffs.forEach((c, i) => {
      if (Math.abs(c) < 1e-12) return;
      const p = deg - i;
      const ac = Math.abs(c);
      const first = parts.length === 0;
      const sign = c < 0 ? "-" : first ? "" : "+";
      let term;
      if (p === 0) term = fmtNum(ac);
      else {
        const ct = Math.abs(ac - 1) < 1e-9 ? "" : fmtNum(ac);
        term = p === 1 ? ct + "s" : ct + "s<sup>" + p + "</sup>";
      }
      parts.push(first && sign === "" ? term : sign + " " + term);
    });
    return parts.join(" ") || "0";
  }

  function fmtNum(v) {
    if (v === 0) return "0";
    if (Math.abs(v) >= 1e4 || Math.abs(v) < 0.001) return v.toExponential(3);
    const s = Number(v.toPrecision(4)).toString();
    return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
  }

  function fmtDeg(v) { return v.toFixed(1) + "°"; }
  function fmtDb(v) { return v.toFixed(1) + " dB"; }
  function fmtFreq(v) {
    if (v >= 1e3 || v < 0.01) return v.toExponential(2) + " rad/s";
    return Number(v.toPrecision(4)).toString() + " rad/s";
  }

  function latexPoly(coeffs) {
    if (!coeffs.length) return "0";
    var deg = coeffs.length - 1;
    var parts = [];
    coeffs.forEach(function (c, i) {
      if (Math.abs(c) < 1e-12) return;
      var p = deg - i;
      var ac = Math.abs(c);
      var first = parts.length === 0;
      var sign = c < 0 ? " - " : first ? "" : " + ";
      var term;
      if (p === 0) {
        term = fmtNum(ac);
      } else {
        var ct = Math.abs(ac - 1) < 1e-9 ? "" : fmtNum(ac);
        term = p === 1 ? ct + "s" : ct + "s^{" + p + "}";
      }
      parts.push(sign + term);
    });
    return parts.join("") || "0";
  }

  function latexErrConst(et) {
    if (et === "Kp") return "K_p";
    if (et === "Kv") return "K_v";
    return "K_a";
  }

  function typesetMath(el) {
    if (window.MathJax && MathJax.typesetPromise) {
      if (MathJax.typesetClear) MathJax.typesetClear(el ? [el] : undefined);
      MathJax.typesetPromise(el ? [el] : undefined).catch(function () {});
    }
  }

  // =====================================================================
  //  BODE COMPUTATION
  // =====================================================================

  function logspace(a, b, n) {
    const r = [];
    for (let i = 0; i < n; i++) r.push(10 ** (a + (i / (n - 1)) * (b - a)));
    return r;
  }

  function freqRange(num, den) {
    const allCorners = [];
    function addRoots(p) {
      const roots = polyRoots(p);
      roots.forEach((r) => { const m = Cabs(r); if (m > 1e-6) allCorners.push(m); });
    }
    addRoots(num);
    addRoots(den);
    if (!allCorners.length) return [1e-2, 1e3];
    const lo = Math.min(...allCorners);
    const hi = Math.max(...allCorners);
    return [10 ** Math.floor(Math.log10(lo) - 1.5), 10 ** Math.ceil(Math.log10(hi) + 1.5)];
  }

  function computeBode(num, den, wArr) {
    const magDb = [], phaseDeg = [];
    for (let i = 0; i < wArr.length; i++) {
      const s = C(0, wArr[i]);
      const h = Cdiv(polyEval(num, s), polyEval(den, s));
      magDb.push(20 * Math.log10(Math.max(Cabs(h), 1e-300)));
      phaseDeg.push(Carg(h) * DEG);
    }
    return { magDb, phaseDeg: unwrapPhase(phaseDeg) };
  }

  function unwrapPhase(ph) {
    if (!ph.length) return [];
    const u = [ph[0]];
    for (let i = 1; i < ph.length; i++) {
      let d = ph[i] - ph[i - 1];
      if (d > 180) d -= 360;
      else if (d < -180) d += 360;
      u.push(u[i - 1] + d);
    }
    return u;
  }

  // =====================================================================
  //  POLYNOMIAL ROOTS (Laguerre)
  // =====================================================================

  function polyRoots(coeffs) {
    const deg = coeffs.length - 1;
    if (deg <= 0) return [];
    if (deg === 1) return [C(-coeffs[1] / coeffs[0])];
    if (deg === 2) return quadRoots(coeffs);
    const lead = coeffs[0];
    let defl = coeffs.map((c) => c / lead);
    const roots = [];
    for (let i = 0; i < deg; i++) {
      const cd = defl.length - 1;
      if (cd === 1) { roots.push(C(-defl[1] / defl[0])); break; }
      if (cd === 2) { quadRoots(defl).forEach((r) => roots.push(r)); break; }
      let z = C(0.4 + 0.9 * i, 0.7 + 0.3 * i);
      for (let it = 0; it < 200; it++) {
        const { p, dp, d2p } = polyDeriv(defl, z);
        if (Cabs(p) < 1e-14) break;
        const g = Cdiv(dp, p), g2 = Cmul(g, g);
        const h = Csub(g2, Cdiv(d2p, p));
        const disc = C((cd - 1) * (cd * h.re - g2.re), (cd - 1) * (cd * h.im - g2.im));
        const sqa = Math.sqrt(Cabs(disc)), ang = Carg(disc) / 2;
        const sq = C(sqa * Math.cos(ang), sqa * Math.sin(ang));
        const d1 = Cadd(g, sq), d2 = Csub(g, sq);
        const den = Cabs(d1) >= Cabs(d2) ? d1 : d2;
        if (Cabs(den) < 1e-30) break;
        const a = Cdiv(C(cd), den);
        const zn = Csub(z, a);
        if (Cabs(Csub(zn, z)) < 1e-13 * Math.max(1, Cabs(z))) { z = zn; break; }
        z = zn;
      }
      if (Math.abs(z.im) < 1e-9 * Math.max(1, Math.abs(z.re))) {
        z = C(z.re);
        roots.push(z);
        const nd = [defl[0]];
        for (let k = 1; k < defl.length - 1; k++) nd.push(defl[k] + nd[k - 1] * z.re);
        defl = nd;
      } else {
        roots.push(C(z.re, z.im));
        roots.push(C(z.re, -z.im));
        const q = [1, -2 * z.re, z.re * z.re + z.im * z.im];
        const nd = [defl[0]];
        for (let k = 1; k < defl.length - 2; k++)
          nd.push(defl[k] - q[1] * nd[k - 1] - (k >= 2 ? q[2] * nd[k - 2] : 0));
        defl = nd;
        i++;
      }
    }
    return roots.slice(0, deg);
  }

  function quadRoots(c) {
    const [a, b, cc] = c;
    const d = b * b - 4 * a * cc;
    if (d >= 0) {
      const sq = Math.sqrt(d);
      return [C((-b + sq) / (2 * a)), C((-b - sq) / (2 * a))];
    }
    const sq = Math.sqrt(-d);
    return [C(-b / (2 * a), sq / (2 * a)), C(-b / (2 * a), -sq / (2 * a))];
  }

  function polyDeriv(coeffs, z) {
    const n = coeffs.length - 1;
    let p = C(coeffs[0]), dp = C(0), d2p = C(0);
    for (let i = 1; i <= n; i++) {
      d2p = Cadd(Cmul(d2p, z), C(2 * dp.re, 2 * dp.im));
      dp = Cadd(Cmul(dp, z), p);
      p = Cadd(Cmul(p, z), C(coeffs[i]));
    }
    return { p, dp, d2p };
  }

  // =====================================================================
  //  FREQUENCY-DOMAIN HELPERS
  // =====================================================================

  function findCrossing(freq, vals, target) {
    for (let i = 0; i < vals.length - 1; i++) {
      const a = vals[i] - target, b = vals[i + 1] - target;
      if (a * b <= 0 && Math.abs(a - b) > 1e-15) {
        const r = a / (a - b);
        const logF = Math.log10(freq[i]) + r * (Math.log10(freq[i + 1]) - Math.log10(freq[i]));
        return { freq: 10 ** logF, idx: i, ratio: r };
      }
    }
    return null;
  }

  function interpVal(vals, idx, ratio) {
    return vals[idx] + ratio * (vals[idx + 1] - vals[idx]);
  }

  function phaseMargin(freq, magDb, phaseDeg) {
    const gc = findCrossing(freq, magDb, 0);
    if (!gc) return { pm: null, wgc: null };
    const ph = interpVal(phaseDeg, gc.idx, gc.ratio);
    return { pm: 180 + ph, wgc: gc.freq };
  }

  function gainMargin(freq, magDb, phaseDeg) {
    const pc = findCrossing(freq, phaseDeg, -180);
    if (!pc) return { gm: Infinity, wpc: null };
    const mg = interpVal(magDb, pc.idx, pc.ratio);
    return { gm: -mg, wpc: pc.freq };
  }

  // =====================================================================
  //  TIME-DOMAIN SPECS ↔ FREQUENCY-DOMAIN SPECS
  // =====================================================================

  function specsToFreqDomain(mpPercent, ts) {
    const mp = mpPercent / 100;
    const lnMp = Math.log(mp);
    const zeta = -lnMp / Math.sqrt(Math.PI * Math.PI + lnMp * lnMp);
    const wn = 4 / (zeta * ts);
    const z2 = zeta * zeta;
    const pmExact = Math.atan2(2 * zeta, Math.sqrt(-2 * z2 + Math.sqrt(1 + 4 * z2 * z2))) * DEG;
    const wBW = wn * Math.sqrt((1 - 2 * z2) + Math.sqrt(4 * z2 * z2 - 4 * z2 + 2));
    return { zeta, wn, pmRequired: pmExact, wBW };
  }

  // =====================================================================
  //  GAIN FROM ERROR CONSTANT
  // =====================================================================

  function computeGainK(num, den, errorType, errorValue) {
    const sysType = systemType(den);
    let reqType;
    if (errorType === "Kp") reqType = 0;
    else if (errorType === "Kv") reqType = 1;
    else reqType = 2;

    if (sysType < reqType)
      throw new Error(
        `System is type ${sysType} but ${errorType} requires type ≥ ${reqType}. Add integrator(s) to the plant.`
      );

    const numOrigin = systemType(num);
    const denReduced = den.slice(0, den.length - sysType);
    const numReduced = num.slice(0, num.length - numOrigin);

    const n0 = numReduced[numReduced.length - 1];
    const d0 = denReduced[denReduced.length - 1];

    if (Math.abs(n0) < 1e-14) throw new Error("Numerator constant term is zero.");

    const Kplant = n0 / d0;
    const effectiveType = sysType - numOrigin;

    if (effectiveType < reqType)
      throw new Error(`Effective system type ${effectiveType} is less than required for ${errorType}.`);

    if (effectiveType > reqType) return 1;

    return errorValue / Math.abs(Kplant);
  }

  // =====================================================================
  //  COMPENSATOR DESIGN ALGORITHMS (iterative)
  // =====================================================================

  function compBodeAtFreq(cNum, cDen, freq, magDb, phaseDeg) {
    const cm = [], cp = [];
    for (let i = 0; i < freq.length; i++) {
      const s = C(0, freq[i]);
      const h = Cdiv(polyEval(cNum, s), polyEval(cDen, s));
      cm.push(magDb[i] + 20 * Math.log10(Math.max(Cabs(h), 1e-300)));
      cp.push(phaseDeg[i] + Carg(h) * DEG);
    }
    return { magDb: cm, phaseDeg: unwrapPhase(cp) };
  }

  function designLead(freq, magDb, phaseDeg, pmReq, safetyDeg) {
    const { pm: pmCur, wgc } = phaseMargin(freq, magDb, phaseDeg);
    if (pmCur === null) return { error: "No gain crossover found for uncompensated system." };
    if (pmCur >= pmReq) return { error: "No lead compensation needed — specs already met.", noComp: true };

    let safety = safetyDeg;
    let bestResult = null;

    for (let iter = 0; iter < 8; iter++) {
      const phiMax = pmReq - pmCur + safety;
      if (phiMax <= 0) return { error: "No lead compensation needed.", noComp: true };
      if (phiMax >= 85) return { error: "Lead cannot provide " + fmtDeg(phiMax) + " phase (max ~70°). Try Lag-Lead." };

      const sinPhi = Math.sin(phiMax * RAD);
      const alpha = (1 - sinPhi) / (1 + sinPhi);
      const targetDb = 10 * Math.log10(alpha);

      const cross = findCrossing(freq, magDb, targetDb);
      if (!cross) return { error: "Cannot find new gain crossover. Check system gain or try Lag-Lead." };

      const wm = cross.freq;
      const T = 1 / (wm * Math.sqrt(alpha));
      const cNum = [T, 1], cDen = [alpha * T, 1];

      const compBode = compBodeAtFreq(cNum, cDen, freq, magDb, phaseDeg);
      const { pm: pmAchieved } = phaseMargin(freq, compBode.magDb, compBode.phaseDeg);

      bestResult = {
        type: "lead", alpha, T, wz: 1 / T, wp: 1 / (alpha * T),
        wm, phiMax, pmCur, targetDb, num: cNum, den: cDen, pmAchieved,
      };

      if (pmAchieved !== null && pmAchieved >= pmReq - 1) break;

      const deficit = pmReq - (pmAchieved || pmCur);
      safety += Math.max(deficit * 0.6, 3);
      if (safety > 55) break;
    }

    return bestResult;
  }

  function designLag(freq, magDb, phaseDeg, pmReq, safetyDeg) {
    const { pm: pmCur, wgc } = phaseMargin(freq, magDb, phaseDeg);

    const targetPhase = -180 + pmReq + safetyDeg;
    const pc = findCrossing(freq, phaseDeg, targetPhase);
    if (!pc) return { error: "Cannot find frequency where phase = " + fmtDeg(targetPhase) + ". Try Lead." };

    const wgcNew = pc.freq;
    const magAtNew = interpVal(magDb, pc.idx, pc.ratio);

    if (magAtNew <= 0) return { error: "Gain is already below 0 dB at desired crossover. Try Lead instead." };

    const beta = Math.pow(10, magAtNew / 20);
    const T = 10 / wgcNew;

    const cNum = [T, 1], cDen = [beta * T, 1];
    const compBode = compBodeAtFreq(cNum, cDen, freq, magDb, phaseDeg);
    const { pm: pmAchieved } = phaseMargin(freq, compBode.magDb, compBode.phaseDeg);

    return {
      type: "lag", beta, T, wz: 1 / T, wp: 1 / (beta * T),
      wgcNew, magAtNew, pmCur: pmCur || 0, pmAchieved,
      num: cNum, den: cDen,
    };
  }

  function designLagLead(freq, magDb, phaseDeg, pmReq, safetyDeg) {
    const { pm: pmCur } = phaseMargin(freq, magDb, phaseDeg);

    const targetPhase = -180 + pmReq + safetyDeg;
    const pc = findCrossing(freq, phaseDeg, targetPhase);
    let wgcDesired;
    if (pc) {
      wgcDesired = pc.freq;
    } else {
      const gc = findCrossing(freq, magDb, 0);
      wgcDesired = gc ? gc.freq * 0.3 : freq[Math.floor(freq.length * 0.4)];
    }

    const nIdx = nearestIdx(freq, wgcDesired);
    const magAtDesired = magDb[nIdx];
    const phaseAtDesired = phaseDeg[nIdx];
    const phaseNeeded = pmReq - (180 + phaseAtDesired);

    let leadPhiMax = Math.max(phaseNeeded + safetyDeg, safetyDeg);
    if (leadPhiMax > 65) leadPhiMax = 60;
    if (leadPhiMax < 10) leadPhiMax = 15;

    const sinPhi = Math.sin(leadPhiMax * RAD);
    const alphaLead = (1 - sinPhi) / (1 + sinPhi);
    const leadGainDb = -10 * Math.log10(alphaLead);
    const lagAttenNeeded = Math.max(magAtDesired + leadGainDb, 0);

    let betaLag = lagAttenNeeded > 0.5 ? Math.pow(10, lagAttenNeeded / 20) : 2;
    if (betaLag > 100) betaLag = 100;

    const Tlead = 1 / (wgcDesired * Math.sqrt(alphaLead));
    const Tlag = 10 / wgcDesired;

    const leadNum = [Tlead, 1], leadDen = [alphaLead * Tlead, 1];
    const lagNum = [Tlag, 1], lagDen = [betaLag * Tlag, 1];
    const cNum = polyMul(leadNum, lagNum);
    const cDen = polyMul(leadDen, lagDen);

    const compBode = compBodeAtFreq(cNum, cDen, freq, magDb, phaseDeg);
    const { pm: pmAchieved } = phaseMargin(freq, compBode.magDb, compBode.phaseDeg);

    return {
      type: "lag-lead",
      lead: {
        alpha: alphaLead, T: Tlead, wz: 1 / Tlead, wp: 1 / (alphaLead * Tlead),
        phiMax: leadPhiMax, num: leadNum, den: leadDen,
      },
      lag: {
        beta: betaLag, T: Tlag, wz: 1 / Tlag, wp: 1 / (betaLag * Tlag),
        num: lagNum, den: lagDen,
      },
      wgcDesired, pmCur: pmCur || 0, pmAchieved,
      num: cNum, den: cDen,
    };
  }

  function designDoubleLead(freq, magDb, phaseDeg, pmReq, safetyDeg) {
    const { pm: pmCur, wgc } = phaseMargin(freq, magDb, phaseDeg);
    if (pmCur === null) return { error: "No gain crossover found for uncompensated system." };
    if (pmCur >= pmReq) return { error: "No compensation needed — specs already met.", noComp: true };

    var safety = safetyDeg;
    var bestResult = null;

    for (var iter = 0; iter < 8; iter++) {
      var phiTotal = pmReq - pmCur + safety;
      var phiMax = phiTotal / 2;
      if (phiMax <= 0) return { error: "No compensation needed.", noComp: true };
      if (phiMax >= 85) return { error: "Even double lead cannot provide " + fmtDeg(phiTotal) + ". Try Lag-Lead." };

      var sinPhi = Math.sin(phiMax * RAD);
      var alpha = (1 - sinPhi) / (1 + sinPhi);
      var targetDb = 20 * Math.log10(alpha);

      var cross = findCrossing(freq, magDb, targetDb);
      if (!cross) return { error: "Cannot find gain crossover for double lead design. Try Lag-Lead." };

      var wm = cross.freq;
      var T = 1 / (wm * Math.sqrt(alpha));

      var singleNum = [T, 1], singleDen = [alpha * T, 1];
      var cNum = polyMul(singleNum, singleNum);
      var cDen = polyMul(singleDen, singleDen);

      var compBode = compBodeAtFreq(cNum, cDen, freq, magDb, phaseDeg);
      var pmAch = phaseMargin(freq, compBode.magDb, compBode.phaseDeg).pm;

      bestResult = {
        type: "double-lead", alpha: alpha, T: T,
        wz: 1 / T, wp: 1 / (alpha * T), wm: wm,
        phiMax: phiMax, phiTotal: phiTotal, pmCur: pmCur,
        targetDb: targetDb,
        singleNum: singleNum, singleDen: singleDen,
        num: cNum, den: cDen, pmAchieved: pmAch,
      };

      if (pmAch !== null && pmAch >= pmReq - 1) break;
      var deficit = pmReq - (pmAch || pmCur);
      safety += Math.max(deficit * 0.5, 3);
      if (safety > 80) break;
    }
    return bestResult;
  }

  function nearestIdx(arr, val) {
    let best = 0, bestD = Infinity;
    for (let i = 0; i < arr.length; i++) {
      const d = Math.abs(arr[i] - val);
      if (d < bestD) { bestD = d; best = i; }
    }
    return best;
  }

  // =====================================================================
  //  STEP RESPONSE (RK4)
  // =====================================================================

  function computeStepResponse(num, den, tEnd, nPts) {
    const a0 = den[0];
    const dm = den.map((c) => c / a0);
    const nm = num.map((c) => c / a0);
    const n = dm.length - 1;
    if (n <= 0) {
      const times = [], outputs = [];
      for (let i = 0; i <= nPts; i++) { times.push(i * tEnd / nPts); outputs.push(nm[0] / dm[0]); }
      return { times, outputs };
    }

    const bPad = new Array(n + 1).fill(0);
    const off = n + 1 - nm.length;
    for (let i = 0; i < nm.length; i++) bPad[off + i] = nm[i];

    const D = bPad[0];
    const Cvec = new Array(n);
    for (let i = 0; i < n; i++) Cvec[i] = bPad[n - i] - dm[n - i] * D;

    const dt = tEnd / nPts;
    const x = new Array(n).fill(0);
    const times = [], outputs = [];

    function f(st) {
      const dx = new Array(n).fill(0);
      for (let i = 0; i < n - 1; i++) dx[i] = st[i + 1];
      let sum = 1;
      for (let j = 0; j < n; j++) sum -= dm[n - j] * st[j];
      dx[n - 1] = sum;
      return dx;
    }

    function output(st) {
      let y = D;
      for (let i = 0; i < n; i++) y += Cvec[i] * st[i];
      return y;
    }

    for (let step = 0; step <= nPts; step++) {
      times.push(step * dt);
      outputs.push(output(x));
      if (step < nPts) {
        const k1 = f(x);
        const x2 = x.map((v, i) => v + 0.5 * dt * k1[i]);
        const k2 = f(x2);
        const x3 = x.map((v, i) => v + 0.5 * dt * k2[i]);
        const k3 = f(x3);
        const x4 = x.map((v, i) => v + dt * k3[i]);
        const k4 = f(x4);
        for (let i = 0; i < n; i++)
          x[i] += (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
      }
    }

    return { times, outputs };
  }

  function measureStepResponse(times, outputs) {
    const finalVal = outputs[outputs.length - 1];
    if (Math.abs(finalVal) < 1e-12) return { mp: null, ts: null, tr: null, finalVal: 0 };

    let peakVal = outputs[0], peakIdx = 0;
    for (let i = 1; i < outputs.length; i++) {
      if (outputs[i] > peakVal) { peakVal = outputs[i]; peakIdx = i; }
    }
    const mp = Math.max(0, ((peakVal - finalVal) / Math.abs(finalVal)) * 100);

    let ts = times[times.length - 1];
    const band = 0.02 * Math.abs(finalVal);
    for (let i = outputs.length - 1; i >= 0; i--) {
      if (Math.abs(outputs[i] - finalVal) > band) { ts = times[Math.min(i + 1, times.length - 1)]; break; }
    }

    let tr = null;
    const lo = 0.1 * finalVal, hi = 0.9 * finalVal;
    let tLo = null, tHi = null;
    for (let i = 1; i < outputs.length; i++) {
      if (tLo === null && outputs[i] >= lo) tLo = times[i];
      if (tHi === null && outputs[i] >= hi) { tHi = times[i]; break; }
    }
    if (tLo !== null && tHi !== null) tr = tHi - tLo;

    return { mp, ts, tr, finalVal, peakTime: times[peakIdx] };
  }

  // =====================================================================
  //  SVG UTILITIES
  // =====================================================================

  function svgEl(tag, attrs) {
    const el = document.createElementNS(SVG_NS, tag);
    if (attrs) Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
    return el;
  }

  function clearEl(el) { while (el.firstChild) el.removeChild(el.firstChild); }

  function pathStr(pts) {
    return pts.map((p, i) => `${i ? "L" : "M"}${p.x.toFixed(1)},${p.y.toFixed(1)}`).join("");
  }

  function niceStep(span, targetTicks) {
    if (!(span > 0)) return 1;
    const rough = span / (targetTicks || 6);
    const mag = 10 ** Math.floor(Math.log10(rough));
    const norm = rough / mag;
    if (norm <= 1) return mag;
    if (norm <= 2) return 2 * mag;
    if (norm <= 5) return 5 * mag;
    return 10 * mag;
  }

  function linTicks(lo, hi, n) {
    const step = niceStep(hi - lo, n || 6);
    const start = Math.floor(lo / step) * step;
    const end = Math.ceil(hi / step) * step;
    const ticks = [];
    for (let v = start; v <= end + step * 0.5; v += step) ticks.push(Number(v.toFixed(10)));
    return { ticks, step };
  }

  function decTicks(lo, hi) {
    const s = Math.floor(Math.log10(lo)), e = Math.ceil(Math.log10(hi));
    const maj = [], min = [];
    for (let exp = s; exp <= e; exp++) {
      const base = 10 ** exp;
      maj.push(base);
      for (let f = 2; f < 10; f++) {
        const t = f * base;
        if (t >= lo && t <= hi) min.push(t);
      }
    }
    return { maj, min };
  }

  function clip(v, lo, hi) { return Math.min(Math.max(v, lo), hi); }

  function placeholder(container, msg) {
    if (!container) return;
    container.innerHTML = '<div class="plot-placeholder">' + escHtml(msg) + "</div>";
  }

  function escHtml(t) {
    return String(t).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  // =====================================================================
  //  GENERAL SVG PLOT RENDERER
  // =====================================================================

  function renderPlot(containerId, opts) {
    const container = document.getElementById(containerId);
    if (!container) return null;
    clearEl(container);

    const W = Math.max(container.clientWidth || 700, 300);
    const H = Math.max(container.clientHeight || 400, 240);
    const M = { top: 48, right: 24, bottom: 52, left: 66 };
    const pW = W - M.left - M.right, pH = H - M.top - M.bottom;

    const logX = opts.logX !== false;
    const xMin = opts.xMin, xMax = opts.xMax, yMin = opts.yMin, yMax = opts.yMax;
    const xLogMin = logX ? Math.log10(xMin) : xMin;
    const xLogMax = logX ? Math.log10(xMax) : xMax;

    const toX = (v) => M.left + ((logX ? Math.log10(v) : v) - xLogMin) / (xLogMax - xLogMin || 1) * pW;
    const toY = (v) => M.top + (1 - (v - yMin) / (yMax - yMin || 1)) * pH;

    const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.title || "" });

    const defs = svgEl("defs");
    const cp = svgEl("clipPath", { id: containerId + "-clip" });
    cp.appendChild(svgEl("rect", { x: M.left, y: M.top, width: pW, height: pH, rx: 10 }));
    defs.appendChild(cp);
    svg.appendChild(defs);

    svg.appendChild(svgEl("rect", {
      x: M.left, y: M.top, width: pW, height: pH, rx: 12,
      fill: COLORS.plot, stroke: "rgba(30,36,48,0.05)",
    }));

    if (logX) {
      const dt = decTicks(xMin, xMax);
      dt.min.forEach((t) => {
        svg.appendChild(svgEl("line", {
          x1: toX(t), y1: M.top, x2: toX(t), y2: M.top + pH,
          stroke: COLORS.gridMinor, "stroke-width": 1,
        }));
      });
      dt.maj.forEach((t) => {
        if (t < xMin || t > xMax) return;
        svg.appendChild(svgEl("line", {
          x1: toX(t), y1: M.top, x2: toX(t), y2: M.top + pH,
          stroke: COLORS.grid, "stroke-width": 1.2,
        }));
        const lbl = svgEl("text", {
          x: toX(t), y: M.top + pH + 20, "text-anchor": "middle", "font-size": 11, fill: "#334155",
        });
        lbl.textContent = t >= 1e3 || t < 0.01 ? t.toExponential(0) : Number(t.toPrecision(3)).toString();
        svg.appendChild(lbl);
      });
    } else {
      const xt = linTicks(xMin, xMax, 8);
      xt.ticks.forEach((t) => {
        if (t < xMin - 1e-9 || t > xMax + 1e-9) return;
        svg.appendChild(svgEl("line", {
          x1: toX(t), y1: M.top, x2: toX(t), y2: M.top + pH,
          stroke: COLORS.grid, "stroke-width": 1,
        }));
        const lbl = svgEl("text", {
          x: toX(t), y: M.top + pH + 20, "text-anchor": "middle", "font-size": 11, fill: "#334155",
        });
        lbl.textContent = Number(t.toPrecision(4)).toString();
        svg.appendChild(lbl);
      });
    }

    const yt = linTicks(yMin, yMax);
    yt.ticks.forEach((t) => {
      if (t < yMin - 1e-9 || t > yMax + 1e-9) return;
      const y = toY(t);
      svg.appendChild(svgEl("line", {
        x1: M.left, y1: y, x2: M.left + pW, y2: y,
        stroke: COLORS.grid, "stroke-width": 1,
      }));
      const lbl = svgEl("text", {
        x: M.left - 8, y: y + 4, "text-anchor": "end", "font-size": 11, fill: "#334155",
      });
      lbl.textContent = Math.abs(t) < 1e-9 ? "0" : (Math.abs(t) >= 1e4 || Math.abs(t) < 0.01)
        ? t.toExponential(1) : Number(t.toPrecision(4)).toString();
      svg.appendChild(lbl);
    });

    if (opts.refLines) {
      opts.refLines.forEach((rl) => {
        const isH = rl.axis === "y";
        const x1 = isH ? M.left : toX(rl.value);
        const y1 = isH ? toY(rl.value) : M.top;
        const x2 = isH ? M.left + pW : toX(rl.value);
        const y2 = isH ? toY(rl.value) : M.top + pH;
        if (isH && (rl.value < yMin || rl.value > yMax)) return;
        svg.appendChild(svgEl("line", {
          x1, y1, x2, y2,
          stroke: rl.color || COLORS.ref,
          "stroke-width": rl.width || 1.5,
          "stroke-dasharray": rl.dash || "6 4",
        }));
        if (rl.label) {
          const lb = svgEl("text", {
            x: x2 - 4, y: isH ? y1 - 5 : y2 + 14,
            "text-anchor": "end", "font-size": 10, fill: rl.color || COLORS.ref,
          });
          lb.textContent = rl.label;
          svg.appendChild(lb);
        }
      });
    }

    const title = svgEl("text", {
      x: M.left, y: 26,
      "font-size": 18, "font-family": "'Iowan Old Style','Palatino Linotype',serif",
      "font-weight": 700, fill: "#1e2430",
    });
    title.textContent = opts.title || "";
    svg.appendChild(title);

    const yLbl = svgEl("text", {
      x: 16, y: M.top + pH / 2,
      transform: `rotate(-90 16 ${M.top + pH / 2})`, "font-size": 12, fill: "#334155",
    });
    yLbl.textContent = opts.yLabel || "";
    svg.appendChild(yLbl);

    const xLbl = svgEl("text", {
      x: M.left + pW / 2, y: H - 12, "text-anchor": "middle", "font-size": 12, fill: "#334155",
    });
    xLbl.textContent = opts.xLabel || "";
    svg.appendChild(xLbl);

    const cg = svgEl("g", { "clip-path": `url(#${containerId}-clip)` });

    (opts.traces || []).forEach((tr) => {
      const pts = tr.x.map((xv, i) => ({
        x: toX(xv),
        y: clip(toY(tr.y[i]), M.top - 20, M.top + pH + 20),
      }));
      const p = svgEl("path", {
        d: pathStr(pts), fill: "none",
        stroke: tr.color, "stroke-width": tr.width || 2.5,
        "stroke-dasharray": tr.dash || "",
        "stroke-opacity": tr.opacity != null ? tr.opacity : 1,
        "stroke-linejoin": "round", "stroke-linecap": "round",
      });
      if (tr.animate) p.classList.add("plot-trace-draw");
      if (tr.animateFast) p.classList.add("plot-trace-draw-fast");
      cg.appendChild(p);
    });

    svg.appendChild(cg);

    if (opts.annotations) {
      opts.annotations.forEach((an) => {
        if (an.type === "vline") {
          const x = toX(an.x);
          cg.appendChild(svgEl("line", {
            x1: x, y1: M.top, x2: x, y2: M.top + pH,
            stroke: an.color || COLORS.pm, "stroke-width": 1.5, "stroke-dasharray": "5 4",
          }));
          if (an.label) {
            const lb = svgEl("text", {
              x: x + 4, y: M.top + 16, "font-size": 10, fill: an.color || COLORS.pm, "font-weight": 600,
            });
            lb.textContent = an.label;
            cg.appendChild(lb);
          }
        } else if (an.type === "hline") {
          const y = toY(an.y);
          cg.appendChild(svgEl("line", {
            x1: M.left, y1: y, x2: M.left + pW, y2: y,
            stroke: an.color || COLORS.pm, "stroke-width": 1.5, "stroke-dasharray": "5 4",
          }));
        } else if (an.type === "point") {
          const px = toX(an.x), py = clip(toY(an.y), M.top, M.top + pH);
          cg.appendChild(svgEl("circle", {
            cx: px, cy: py, r: 7, fill: "rgba(255,255,255,0.85)",
            stroke: an.color || COLORS.compMag, "stroke-width": 2,
          }));
          cg.appendChild(svgEl("circle", {
            cx: px, cy: py, r: 3.5, fill: an.color || COLORS.compMag,
          }));
          if (an.label) {
            const lb = svgEl("text", {
              x: px + 10, y: py + 4, "font-size": 10, fill: an.color || COLORS.compMag, "font-weight": 600,
            });
            lb.textContent = an.label;
            cg.appendChild(lb);
          }
        } else if (an.type === "pmArc") {
          const px = toX(an.x);
          const y180 = toY(-180);
          const yPh = clip(toY(an.phase), M.top, M.top + pH);
          cg.appendChild(svgEl("line", {
            x1: px, y1: y180, x2: px, y2: yPh,
            stroke: an.color || COLORS.pm, "stroke-width": 3, "stroke-linecap": "round",
          }));
          const midY = (y180 + yPh) / 2;
          const lb = svgEl("text", {
            x: px + 8, y: midY + 4, "font-size": 11, fill: an.color || COLORS.pm, "font-weight": 700,
          });
          lb.textContent = "PM=" + fmtDeg(an.pm);
          cg.appendChild(lb);
        }
      });
    }

    if (opts.legend) {
      const lW = 180, lH = opts.legend.length * 18 + 12;
      const lX = W - M.right - lW + 4, lY = M.top + 10;
      svg.appendChild(svgEl("rect", {
        x: lX - 4, y: lY - 10, width: lW, height: lH, rx: 8,
        fill: "rgba(255,255,255,0.6)", stroke: "rgba(30,36,48,0.06)",
      }));
      opts.legend.forEach((le, i) => {
        const y = lY + i * 18;
        svg.appendChild(svgEl("line", {
          x1: lX, y1: y, x2: lX + 22, y2: y,
          stroke: le.color, "stroke-width": 2.5,
          "stroke-dasharray": le.dash || "", "stroke-linecap": "round",
        }));
        const lb = svgEl("text", { x: lX + 30, y: y + 3.5, "font-size": 10.5, fill: "#64748b" });
        lb.textContent = le.label;
        svg.appendChild(lb);
      });
    }

    container.appendChild(svg);
    return { toX, toY };
  }

  // =====================================================================
  //  FLOWCHART RENDERER
  // =====================================================================

  const FLOW_STEPS = [
    { id: "specs",     label: "Step 1", title: "Define Specifications",          decision: false },
    { id: "freqdom",   label: "Step 2", title: "Convert to Frequency Domain",   decision: false },
    { id: "gainK",     label: "Step 3", title: "Determine System Gain K",       decision: false },
    { id: "uncomp",    label: "Step 4", title: "Uncompensated Bode Analysis",   decision: false },
    { id: "check",     label: "Step 5", title: "Evaluate Phase Deficiency",     decision: true  },
    { id: "design",    label: "Step 6", title: "Design Compensator",            decision: false },
    { id: "build",     label: "Step 7", title: "Build Compensator C(s)",        decision: false },
    { id: "compbode",  label: "Step 8", title: "Compensated Bode Analysis",     decision: false },
    { id: "verify",    label: "Step 9", title: "Verify Design",                 decision: false },
    { id: "response",  label: "Step 10", title: "Closed-Loop Step Response",     decision: false },
  ];

  function renderFlowchart(container, activeIdx, briefs) {
    clearEl(container);
    FLOW_STEPS.forEach((step, i) => {
      const div = document.createElement("div");
      div.className = "fc-step" + (step.decision ? " fc-decision" : "");
      if (i < activeIdx) div.classList.add("fc-done");
      else if (i === activeIdx) div.classList.add("fc-active");

      const ind = document.createElement("div");
      ind.className = "fc-indicator";

      const dot = document.createElement("div");
      dot.className = "fc-dot" + (step.decision ? " fc-dot-diamond" : "");
      const inner = document.createElement("span");
      inner.className = "fc-dot-inner";
      inner.textContent = i < activeIdx ? "✓" : String(i + 1);
      dot.appendChild(inner);
      ind.appendChild(dot);

      if (i < FLOW_STEPS.length - 1) {
        const conn = document.createElement("div");
        conn.className = "fc-connector";
        ind.appendChild(conn);
      }

      const body = document.createElement("div");
      body.className = "fc-body";
      const lbl = document.createElement("div");
      lbl.className = "fc-label";
      lbl.textContent = step.label;
      const ttl = document.createElement("div");
      ttl.className = "fc-title";
      ttl.textContent = step.title;
      body.appendChild(lbl);
      body.appendChild(ttl);

      if (briefs && briefs[i]) {
        const brief = document.createElement("div");
        brief.className = "fc-brief";
        brief.innerHTML = briefs[i];
        body.appendChild(brief);
      }

      div.appendChild(ind);
      div.appendChild(body);
      container.appendChild(div);
    });
  }

  // =====================================================================
  //  READOUT PILLS
  // =====================================================================

  function renderReadout(node, entries) {
    if (!node) return;
    node.innerHTML = entries.map((e) =>
      '<div class="readout-pill"><span class="readout-label">' + escHtml(e.label) +
      '</span><span>' + escHtml(e.value) + "</span></div>"
    ).join("");
  }

  // =====================================================================
  //  DESIGN FLOW ENGINE
  // =====================================================================

  function runDesign(inputs) {
    const { num, den, mpPercent, ts, errorType, errorValue, compChoice, safetyDeg } = inputs;

    const steps = [];
    const fSpec = specsToFreqDomain(mpPercent, ts);
    const K = computeGainK(num, den, errorType, errorValue);
    const numK = polyScale(num, K);

    const [wLo, wHi] = freqRange(numK, den);
    const wArr = logspace(Math.log10(wLo), Math.log10(wHi), SAMPLES);
    const bodeUncomp = computeBode(numK, den, wArr);
    const { pm: pmUncomp, wgc: wgcUncomp } = phaseMargin(wArr, bodeUncomp.magDb, bodeUncomp.phaseDeg);
    const { gm: gmUncomp } = gainMargin(wArr, bodeUncomp.magDb, bodeUncomp.phaseDeg);

    steps.push({
      id: "specs",
      data: { num, den, mpPercent, ts, errorType, errorValue },
    });

    steps.push({
      id: "freqdom",
      data: { zeta: fSpec.zeta, wn: fSpec.wn, pmRequired: fSpec.pmRequired, wBW: fSpec.wBW },
    });

    steps.push({
      id: "gainK",
      data: { K, errorType, errorValue, numK, den },
    });

    steps.push({
      id: "uncomp",
      data: { wArr, bode: bodeUncomp, pm: pmUncomp, wgc: wgcUncomp, gm: gmUncomp },
    });

    const phaseDef = pmUncomp !== null ? fSpec.pmRequired - pmUncomp : 90;
    const totalNeeded = phaseDef + safetyDeg;

    let compType = compChoice;
    let compReason = "";

    if (compChoice === "auto") {
      var lagTarget = -180 + fSpec.pmRequired + safetyDeg;
      var lagCross = findCrossing(wArr, bodeUncomp.phaseDeg, lagTarget);
      var lagViable = lagCross && interpVal(bodeUncomp.magDb, lagCross.idx, lagCross.ratio) > 0;
      var lagFreq = lagCross ? lagCross.freq : null;

      if (totalNeeded <= 0) {
        compType = "none";
        compReason = "Current phase margin already meets or exceeds the requirement — no compensation needed.";
      } else if (totalNeeded <= 55) {
        compType = "lead";
        compReason = "Total phase boost of " + fmtDeg(totalNeeded) + " is within the practical range of a single lead compensator (up to ~55°). Lead adds phase near the gain crossover while accepting a moderate increase in crossover frequency.";
      } else if (lagViable && totalNeeded <= 110) {
        compType = "lag";
        compReason = "Phase boost of " + fmtDeg(totalNeeded) + " exceeds what a single lead can provide. The uncompensated system already has adequate phase (" + fmtDeg(lagTarget + 180) + ") at \\( \\omega = " + fmtNum(lagFreq) + " \\) rad/s where the gain is " + fmtDb(interpVal(bodeUncomp.magDb, lagCross.idx, lagCross.ratio)) + ". A lag compensator shifts the gain crossover to this lower frequency without amplifying high-frequency noise.";
      } else if (totalNeeded <= 110) {
        compType = "double-lead";
        compReason = "Phase boost of " + fmtDeg(totalNeeded) + " exceeds the practical limit of a single lead (~55°), and no suitable lower-frequency crossover exists for lag design. Two identical lead compensators in cascade, each contributing " + fmtDeg(totalNeeded / 2) + " of phase, can achieve the required boost.";
      } else if (lagViable) {
        compType = "lag";
        compReason = "Phase boost of " + fmtDeg(totalNeeded) + " is very large. The system has adequate phase at \\( \\omega = " + fmtNum(lagFreq) + " \\) rad/s. A lag compensator attenuates gain to shift the crossover there.";
      } else {
        compType = "lag-lead";
        compReason = "Phase boost of " + fmtDeg(totalNeeded) + " is too large for lead compensation alone (even double lead), and no suitable lower crossover exists for lag. A lag-lead compensator combines low-frequency gain reduction (lag section) with targeted phase boost near crossover (lead section).";
      }
    } else {
      var manualReasons = {
        "none": "No compensation — user override.",
        "lead": "Lead compensator (manual). Lead adds phase near the gain crossover frequency. Effective when the required total phase boost is up to ~55°. Total here: " + fmtDeg(totalNeeded) + ".",
        "double-lead": "Double lead compensator (manual). Two identical lead sections in cascade, each providing half the phase boost. Suitable when ~55°–110° of additional phase is needed and maintaining bandwidth is important. Total here: " + fmtDeg(totalNeeded) + "; each section provides ~" + fmtDeg(totalNeeded / 2) + ".",
        "lag": "Lag compensator (manual). Lag attenuates gain to shift the crossover to a lower frequency where existing phase is already adequate. Does not amplify high-frequency noise. Total phase boost needed: " + fmtDeg(totalNeeded) + ".",
        "lag-lead": "Lag-lead compensator (manual). Combines low-frequency gain reduction (lag) with targeted phase boost near crossover (lead). Suitable for large phase deficiencies. Total: " + fmtDeg(totalNeeded) + ".",
      };
      compReason = manualReasons[compType] || "Compensator type selected manually.";
    }

    steps.push({
      id: "check",
      data: { pmRequired: fSpec.pmRequired, pmUncomp, phaseDef, compType, safetyDeg, compReason, totalNeeded },
    });

    let comp;
    if (compType === "none") {
      comp = { type: "none", num: [1], den: [1], noComp: true };
    } else if (compType === "lead") {
      comp = designLead(wArr, bodeUncomp.magDb, bodeUncomp.phaseDeg, fSpec.pmRequired, safetyDeg);
      if (comp.error && !comp.noComp) {
        comp = designDoubleLead(wArr, bodeUncomp.magDb, bodeUncomp.phaseDeg, fSpec.pmRequired, safetyDeg);
      }
      if (comp.error && !comp.noComp) {
        comp = designLagLead(wArr, bodeUncomp.magDb, bodeUncomp.phaseDeg, fSpec.pmRequired, safetyDeg);
      }
    } else if (compType === "double-lead") {
      comp = designDoubleLead(wArr, bodeUncomp.magDb, bodeUncomp.phaseDeg, fSpec.pmRequired, safetyDeg);
      if (comp.error && !comp.noComp) {
        comp = designLagLead(wArr, bodeUncomp.magDb, bodeUncomp.phaseDeg, fSpec.pmRequired, safetyDeg);
      }
    } else if (compType === "lag") {
      comp = designLag(wArr, bodeUncomp.magDb, bodeUncomp.phaseDeg, fSpec.pmRequired, safetyDeg);
      if (comp.error) {
        comp = designLead(wArr, bodeUncomp.magDb, bodeUncomp.phaseDeg, fSpec.pmRequired, safetyDeg);
      }
    } else {
      comp = designLagLead(wArr, bodeUncomp.magDb, bodeUncomp.phaseDeg, fSpec.pmRequired, safetyDeg);
    }

    if (comp.error && !comp.noComp) throw new Error(comp.error);
    if (comp.noComp) comp = { type: "none", num: [1], den: [1] };

    steps.push({ id: "design", data: { comp, pmRequired: fSpec.pmRequired, safetyDeg } });

    steps.push({
      id: "build",
      data: { comp },
    });

    const numOL = polyMul(numK, comp.num);
    const denOL = polyMul(den, comp.den);

    const [wLo2, wHi2] = freqRange(numOL, denOL);
    const wLoBoth = Math.min(wLo, wLo2), wHiBoth = Math.max(wHi, wHi2);
    const wArrFull = logspace(Math.log10(wLoBoth), Math.log10(wHiBoth), SAMPLES);

    const bodeComp = computeBode(numOL, denOL, wArrFull);
    const bodeUncompFull = computeBode(numK, den, wArrFull);
    const bodeCtrl = computeBode(comp.num, comp.den, wArrFull);

    const { pm: pmComp, wgc: wgcComp } = phaseMargin(wArrFull, bodeComp.magDb, bodeComp.phaseDeg);
    const { gm: gmComp } = gainMargin(wArrFull, bodeComp.magDb, bodeComp.phaseDeg);

    steps.push({
      id: "compbode",
      data: {
        wArr: wArrFull,
        bodeUncomp: bodeUncompFull,
        bodeComp,
        bodeCtrl,
        pm: pmComp, wgc: wgcComp, gm: gmComp,
        comp,
      },
    });

    steps.push({
      id: "verify",
      data: {
        pmRequired: fSpec.pmRequired, pmAchieved: pmComp,
        wBWRequired: fSpec.wBW,
        gmAchieved: gmComp,
        comp,
      },
    });

    const numCL = numOL;
    const denCL = polyAdd(denOL, numOL);

    const tEnd = Math.max(ts * 2.5, 10);
    const stepResp = computeStepResponse(numCL, denCL, tEnd, TIME_SAMPLES);
    const metrics = measureStepResponse(stepResp.times, stepResp.outputs);

    let uncompCL_num = numK;
    let uncompCL_den = polyAdd(den, numK);
    const stepRespUncomp = computeStepResponse(uncompCL_num, uncompCL_den, tEnd, TIME_SAMPLES);
    const metricsUncomp = measureStepResponse(stepRespUncomp.times, stepRespUncomp.outputs);

    steps.push({
      id: "response",
      data: {
        stepResp, metrics, stepRespUncomp, metricsUncomp,
        mpRequired: mpPercent, tsRequired: ts, tEnd,
        numCL, denCL, comp,
      },
    });

    return { steps, fSpec, K, comp, numOL, denOL, numK, den: den };
  }

  // =====================================================================
  //  STEP DETAIL CONTENT GENERATORS
  // =====================================================================

  function stepHtml_specs(data) {
    return `
      <div class="calc-grid">
        <div class="calc-card">
          <div class="calc-label">Plant Transfer Function</div>
          <div class="calc-formula">
            \\[ G(s) = \\frac{${latexPoly(data.num)}}{${latexPoly(data.den)}} \\]
          </div>
        </div>
        <div class="calc-card">
          <div class="calc-label">Time-Domain Specs</div>
          <div class="calc-result">
            \\( M_p \\leq ${data.mpPercent}\\% \\)<br>
            \\( t_s \\leq ${data.ts} \\text{ s} \\)
          </div>
        </div>
        <div class="calc-card">
          <div class="calc-label">Error Requirement</div>
          <div class="calc-result">\\( ${latexErrConst(data.errorType)} \\geq ${data.errorValue} \\)</div>
        </div>
      </div>`;
  }

  function stepHtml_freqdom(data) {
    return `
      <div class="calc-card">
        <div class="calc-label">Damping Ratio from Overshoot</div>
        <div class="calc-formula">
          \\[ \\zeta = \\frac{-\\ln(M_p/100)}{\\sqrt{\\pi^2 + \\ln^2(M_p/100)}} = ${fmtNum(data.zeta)} \\]
        </div>
      </div>
      <div class="calc-card">
        <div class="calc-label">Natural Frequency from Settling Time</div>
        <div class="calc-formula">
          \\[ \\omega_n = \\frac{4}{\\zeta \\cdot t_s} = ${fmtNum(data.wn)} \\text{ rad/s} \\]
        </div>
      </div>
      <div class="calc-card">
        <div class="calc-label">Required Phase Margin</div>
        <div class="calc-formula">
          \\[ \\text{PM}_{\\text{req}} = \\tan^{-1}\\!\\left(\\frac{2\\zeta}{\\sqrt{-2\\zeta^2 + \\sqrt{1+4\\zeta^4}}}\\right) = ${fmtDeg(data.pmRequired)} \\]
        </div>
      </div>
      <div class="calc-grid">
        <div class="calc-metric">
          <div class="calc-label">Required PM</div>
          <div class="calc-metric-value">${fmtDeg(data.pmRequired)}</div>
        </div>
        <div class="calc-metric">
          <div class="calc-label">Desired Bandwidth</div>
          <div class="calc-metric-value">${fmtNum(data.wBW)} rad/s</div>
        </div>
        <div class="calc-metric">
          <div class="calc-label">Damping &zeta;</div>
          <div class="calc-metric-value">${fmtNum(data.zeta)}</div>
        </div>
        <div class="calc-metric">
          <div class="calc-label">Natural Freq &omega;<sub>n</sub></div>
          <div class="calc-metric-value">${fmtNum(data.wn)} rad/s</div>
        </div>
      </div>`;
  }

  function stepHtml_gainK(data) {
    return `
      <div class="calc-card">
        <div class="calc-label">Gain Calculation</div>
        <div class="calc-formula">
          From \\( ${latexErrConst(data.errorType)} \\geq ${data.errorValue} \\) requirement:
          \\[ K = ${fmtNum(data.K)} \\]
        </div>
      </div>
      <div class="calc-card">
        <div class="calc-label">Open-Loop with Gain</div>
        <div class="calc-formula">
          \\[ KG(s) = \\frac{${latexPoly(data.numK)}}{${latexPoly(data.den)}} \\]
        </div>
      </div>`;
  }

  function stepHtml_uncomp(data) {
    var pmText = data.pm !== null ? fmtDeg(data.pm) : "N/A";
    var wgcText = data.wgc !== null ? fmtFreq(data.wgc) : "N/A";
    var gmText = data.gm === Infinity ? "∞" : data.gm !== null ? fmtDb(data.gm) : "N/A";
    var pmCurStr = data.pm !== null ? fmtNum(data.pm) + "^\\circ" : "\\text{N/A}";
    var wgcNum = data.wgc !== null ? fmtNum(data.wgc) : null;
    return `
      <div class="calc-card">
        <div class="calc-label">Uncompensated Bode Analysis</div>
        <div class="calc-formula">
          \\[ \\text{PM} = 180^\\circ + \\angle KG(j\\omega_{gc}) = ${pmCurStr} \\]
          ${data.wgc !== null ? `\\[ \\omega_{gc} = ${wgcNum} \\text{ rad/s} \\]` : ""}
        </div>
      </div>
      <div class="calc-grid">
        <div class="calc-metric">
          <div class="calc-label">Phase Margin</div>
          <div class="calc-metric-value ${data.pm !== null && data.pm > 0 ? "val-warn" : "val-bad"}">${pmText}</div>
        </div>
        <div class="calc-metric">
          <div class="calc-label">&omega;<sub>gc</sub></div>
          <div class="calc-metric-value">${wgcText}</div>
        </div>
        <div class="calc-metric">
          <div class="calc-label">Gain Margin</div>
          <div class="calc-metric-value">${gmText}</div>
        </div>
      </div>`;
  }

  function stepHtml_check(data) {
    var totalStr = fmtDeg(data.totalNeeded);
    var typeLabel = {
      lead: "Lead Compensator",
      "double-lead": "Double Lead Compensator",
      lag: "Lag Compensator",
      "lag-lead": "Lag-Lead Compensator",
      none: "No Compensation Needed",
    }[data.compType] || data.compType;
    var pmCurStr = data.pmUncomp !== null ? fmtNum(data.pmUncomp) + "^\\circ" : "\\text{N/A}";
    return `
      <div class="calc-card">
        <div class="calc-label">Phase Deficiency Analysis</div>
        <div class="calc-formula">
          \\[ \\text{PM}_{\\text{req}} - \\text{PM}_{\\text{cur}} = ${fmtNum(data.pmRequired)}^\\circ - ${pmCurStr} = ${fmtNum(data.phaseDef)}^\\circ \\]
          With \\( ${fmtNum(data.safetyDeg)}^\\circ \\) safety margin \\( \\Rightarrow \\) total phase boost needed: <span class="calc-hl-orange">${totalStr}</span>
        </div>
      </div>
      <div class="calc-card">
        <div class="calc-label">Design Decision — Why ${typeLabel}?</div>
        <div class="calc-formula">${data.compReason}</div>
      </div>
      <div class="calc-card">
        <div class="calc-label">Selected Compensator</div>
        <div class="calc-formula">
          <span class="calc-hl" style="font-size:1.15em">${typeLabel}</span>
        </div>
      </div>`;
  }

  function stepHtml_design(data) {
    var c = data.comp;
    if (c.type === "none") {
      return '<div class="calc-card"><div class="calc-formula">No compensation needed.</div></div>';
    }
    if (c.type === "lead") {
      return `
        <div class="calc-card">
          <div class="calc-label">Lead Compensator Design</div>
          <div class="calc-formula">
            Maximum phase lead needed:
            \\[ \\phi_{\\max} = ${fmtNum(c.phiMax)}^\\circ \\]
            \\[ \\sin(\\phi_{\\max}) = ${fmtNum(Math.sin(c.phiMax * RAD))} \\]
            \\[ \\alpha = \\frac{1 - \\sin\\phi_{\\max}}{1 + \\sin\\phi_{\\max}} = ${fmtNum(c.alpha)} \\]
            Find new \\( \\omega_{gc} \\) where \\( |KG(j\\omega)| = ${fmtDb(c.targetDb)} \\):
            \\[ \\omega_m = ${fmtNum(c.wm)} \\text{ rad/s} \\]
            \\[ T = \\frac{1}{\\omega_m\\sqrt{\\alpha}} = ${fmtNum(c.T)} \\text{ s} \\]
          </div>
        </div>
        <div class="calc-grid">
          <div class="calc-metric">
            <div class="calc-label">Zero &omega;<sub>z</sub></div>
            <div class="calc-metric-value">${fmtNum(c.wz)} rad/s</div>
          </div>
          <div class="calc-metric">
            <div class="calc-label">Pole &omega;<sub>p</sub></div>
            <div class="calc-metric-value">${fmtNum(c.wp)} rad/s</div>
          </div>
          <div class="calc-metric">
            <div class="calc-label">&alpha;</div>
            <div class="calc-metric-value">${fmtNum(c.alpha)}</div>
          </div>
        </div>`;
    }
    if (c.type === "double-lead") {
      return `
        <div class="calc-card">
          <div class="calc-label">Double Lead Compensator Design</div>
          <div class="calc-formula">
            Total phase boost needed: \\( \\phi_{\\text{total}} = ${fmtNum(c.phiTotal)}^\\circ \\). Each lead section provides half:
            \\[ \\phi_{\\max} = \\frac{\\phi_{\\text{total}}}{2} = ${fmtNum(c.phiMax)}^\\circ \\]
            \\[ \\alpha = \\frac{1 - \\sin\\phi_{\\max}}{1 + \\sin\\phi_{\\max}} = ${fmtNum(c.alpha)} \\]
            Combined gain at \\( \\omega_m \\): two sections \\( \\Rightarrow \\) target \\( |KG| = 20\\log_{10}\\alpha = ${fmtDb(c.targetDb)} \\):
            \\[ \\omega_m = ${fmtNum(c.wm)} \\text{ rad/s} \\]
            \\[ T = \\frac{1}{\\omega_m\\sqrt{\\alpha}} = ${fmtNum(c.T)} \\text{ s} \\]
          </div>
        </div>
        <div class="calc-grid">
          <div class="calc-metric">
            <div class="calc-label">Zero &omega;<sub>z</sub> (each)</div>
            <div class="calc-metric-value">${fmtNum(c.wz)} rad/s</div>
          </div>
          <div class="calc-metric">
            <div class="calc-label">Pole &omega;<sub>p</sub> (each)</div>
            <div class="calc-metric-value">${fmtNum(c.wp)} rad/s</div>
          </div>
          <div class="calc-metric">
            <div class="calc-label">&alpha;</div>
            <div class="calc-metric-value">${fmtNum(c.alpha)}</div>
          </div>
        </div>`;
    }
    if (c.type === "lag") {
      return `
        <div class="calc-card">
          <div class="calc-label">Lag Compensator Design</div>
          <div class="calc-formula">
            Find new \\( \\omega_{gc} \\) where \\( \\angle G = -180^\\circ + \\text{PM}_{\\text{req}} + \\varepsilon \\):
            \\[ \\omega_{gc,\\text{new}} = ${fmtNum(c.wgcNew)} \\text{ rad/s} \\]
            Gain at new \\( \\omega_{gc} \\): \\( ${fmtNum(c.magAtNew)} \\text{ dB} \\) (must attenuate to 0 dB)
            \\[ \\beta = 10^{\\,\\text{gain}/20} = ${fmtNum(c.beta)} \\]
            \\[ T = \\frac{10}{\\omega_{gc,\\text{new}}} = ${fmtNum(c.T)} \\text{ s} \\]
          </div>
        </div>
        <div class="calc-grid">
          <div class="calc-metric">
            <div class="calc-label">Zero \\(\\omega_z = 1/T\\)</div>
            <div class="calc-metric-value">${fmtNum(c.wz)} rad/s</div>
          </div>
          <div class="calc-metric">
            <div class="calc-label">Pole \\(\\omega_p = 1/\\beta T\\)</div>
            <div class="calc-metric-value">${fmtNum(c.wp)} rad/s</div>
          </div>
          <div class="calc-metric">
            <div class="calc-label">&beta;</div>
            <div class="calc-metric-value">${fmtNum(c.beta)}</div>
          </div>
        </div>`;
    }
    if (c.type === "lag-lead") {
      var ld = c.lead, lg = c.lag;
      return `
        <div class="calc-card">
          <div class="calc-label">Lead Section</div>
          <div class="calc-formula">
            \\[ \\phi_{\\max} = ${fmtNum(ld.phiMax)}^\\circ, \\quad
            \\alpha = ${fmtNum(ld.alpha)}, \\quad
            T_{\\text{lead}} = ${fmtNum(ld.T)} \\text{ s} \\]
            Zero: \\( ${fmtNum(ld.wz)} \\) rad/s, &ensp; Pole: \\( ${fmtNum(ld.wp)} \\) rad/s
          </div>
        </div>
        <div class="calc-card">
          <div class="calc-label">Lag Section</div>
          <div class="calc-formula">
            \\[ \\beta = ${fmtNum(lg.beta)}, \\quad
            T_{\\text{lag}} = ${fmtNum(lg.T)} \\text{ s} \\]
            Zero: \\( ${fmtNum(lg.wz)} \\) rad/s, &ensp; Pole: \\( ${fmtNum(lg.wp)} \\) rad/s
          </div>
        </div>`;
    }
    return "";
  }

  function stepHtml_build(data) {
    var c = data.comp;
    if (c.type === "none") {
      return '<div class="calc-card"><div class="calc-formula">\\( C(s) = 1 \\)</div></div>';
    }
    var genForm = "";
    if (c.type === "lead") {
      genForm = `\\[ C(s) = \\frac{Ts + 1}{\\alpha Ts + 1} \\]`;
    } else if (c.type === "double-lead") {
      genForm = `\\[ C(s) = \\left(\\frac{Ts + 1}{\\alpha Ts + 1}\\right)^{\\!2} \\]`;
    } else if (c.type === "lag") {
      genForm = `\\[ C(s) = \\frac{Ts + 1}{\\beta Ts + 1} \\]`;
    } else if (c.type === "lag-lead") {
      genForm = `\\[ C(s) = \\underbrace{\\frac{T_1 s + 1}{\\alpha T_1 s + 1}}_{\\text{Lead}} \\;\\cdot\\; \\underbrace{\\frac{T_2 s + 1}{\\beta T_2 s + 1}}_{\\text{Lag}} \\]`;
    }
    var extra = "";
    if (c.type === "double-lead") {
      extra = `
        <div class="calc-card">
          <div class="calc-label">Each Lead Section (identical)</div>
          <div class="calc-formula">
            \\[ C_1(s) = C_2(s) = \\frac{${latexPoly(c.singleNum)}}{${latexPoly(c.singleDen)}} \\]
          </div>
        </div>`;
    } else if (c.type === "lag-lead") {
      extra = `
        <div class="calc-card">
          <div class="calc-label">Lead Part</div>
          <div class="calc-formula">
            \\[ C_{\\text{lead}}(s) = \\frac{${latexPoly(c.lead.num)}}{${latexPoly(c.lead.den)}} \\]
          </div>
        </div>
        <div class="calc-card">
          <div class="calc-label">Lag Part</div>
          <div class="calc-formula">
            \\[ C_{\\text{lag}}(s) = \\frac{${latexPoly(c.lag.num)}}{${latexPoly(c.lag.den)}} \\]
          </div>
        </div>`;
    }
    return `
      <div class="calc-card">
        <div class="calc-label">Compensator Transfer Function</div>
        <div class="calc-formula">
          ${genForm}
          \\[ C(s) = \\frac{${latexPoly(c.num)}}{${latexPoly(c.den)}} \\]
        </div>
      </div>${extra}`;
  }

  function stepHtml_compbode(data) {
    var pmText = data.pm !== null ? fmtDeg(data.pm) : "N/A";
    var wgcText = data.wgc !== null ? fmtFreq(data.wgc) : "N/A";
    var gmText = data.gm === Infinity ? "∞" : data.gm !== null ? fmtDb(data.gm) : "N/A";
    var pmStr = data.pm !== null ? fmtNum(data.pm) + "^\\circ" : "\\text{N/A}";
    return `
      <div class="calc-card">
        <div class="calc-label">Compensated Open-Loop Analysis</div>
        <div class="calc-formula">
          \\[ \\text{PM}_{\\text{comp}} = 180^\\circ + \\angle C(j\\omega_{gc})\\,KG(j\\omega_{gc}) = ${pmStr} \\]
        </div>
      </div>
      <div class="calc-grid">
        <div class="calc-metric">
          <div class="calc-label">Compensated PM</div>
          <div class="calc-metric-value val-good">${pmText}</div>
        </div>
        <div class="calc-metric">
          <div class="calc-label">New &omega;<sub>gc</sub></div>
          <div class="calc-metric-value">${wgcText}</div>
        </div>
        <div class="calc-metric">
          <div class="calc-label">Gain Margin</div>
          <div class="calc-metric-value">${gmText}</div>
        </div>
      </div>`;
  }

  function stepHtml_verify(data) {
    var pmOk = data.pmAchieved !== null && data.pmAchieved >= data.pmRequired - 2;
    var gmOk = data.gmAchieved > 0;
    return `
      <div class="calc-card">
        <table class="summary-table">
          <tr><th>Specification</th><th>Required</th><th>Achieved</th><th>Status</th></tr>
          <tr>
            <td>Phase Margin</td>
            <td>\\( \\geq ${fmtNum(data.pmRequired)}^\\circ \\)</td>
            <td>${data.pmAchieved !== null ? fmtDeg(data.pmAchieved) : "N/A"}</td>
            <td class="${pmOk ? "val-pass" : "val-fail"}">${pmOk ? "PASS" : "FAIL"}</td>
          </tr>
          <tr>
            <td>Gain Margin</td>
            <td>\\( > 0 \\text{ dB} \\)</td>
            <td>${data.gmAchieved === Infinity ? "∞" : fmtDb(data.gmAchieved)}</td>
            <td class="${gmOk ? "val-pass" : "val-fail"}">${gmOk ? "PASS" : "FAIL"}</td>
          </tr>
        </table>
      </div>`;
  }

  function stepHtml_response(data) {
    var m = data.metrics;
    var mu = data.metricsUncomp;
    var mpOk = m.mp !== null && m.mp <= data.mpRequired + 1;
    var tsOk = m.ts !== null && m.ts <= data.tsRequired * 1.1;
    return `
      <div class="calc-card">
        <table class="summary-table">
          <tr><th>Metric</th><th>Required</th><th>Uncompensated</th><th>Compensated</th><th>Status</th></tr>
          <tr>
            <td>\\( M_p \\)</td>
            <td>\\( \\leq ${data.mpRequired}\\% \\)</td>
            <td>${mu.mp !== null ? fmtNum(mu.mp) + "%" : "N/A"}</td>
            <td>${m.mp !== null ? fmtNum(m.mp) + "%" : "N/A"}</td>
            <td class="${mpOk ? "val-pass" : "val-fail"}">${mpOk ? "PASS" : "REVIEW"}</td>
          </tr>
          <tr>
            <td>\\( t_s \\)</td>
            <td>\\( \\leq ${data.tsRequired} \\text{ s} \\)</td>
            <td>${mu.ts !== null ? fmtNum(mu.ts) + " s" : "N/A"}</td>
            <td>${m.ts !== null ? fmtNum(m.ts) + " s" : "N/A"}</td>
            <td class="${tsOk ? "val-pass" : "val-fail"}">${tsOk ? "PASS" : "REVIEW"}</td>
          </tr>
          <tr>
            <td>\\( t_r \\)</td>
            <td>&mdash;</td>
            <td>${mu.tr !== null ? fmtNum(mu.tr) + " s" : "N/A"}</td>
            <td>${m.tr !== null ? fmtNum(m.tr) + " s" : "N/A"}</td>
            <td>&mdash;</td>
          </tr>
          <tr>
            <td>Final Value</td>
            <td>1.0</td>
            <td>${fmtNum(mu.finalVal)}</td>
            <td>${fmtNum(m.finalVal)}</td>
            <td>&mdash;</td>
          </tr>
        </table>
      </div>`;
  }

  const stepRenderers = {
    specs: stepHtml_specs,
    freqdom: stepHtml_freqdom,
    gainK: stepHtml_gainK,
    uncomp: stepHtml_uncomp,
    check: stepHtml_check,
    design: stepHtml_design,
    build: stepHtml_build,
    compbode: stepHtml_compbode,
    verify: stepHtml_verify,
    response: stepHtml_response,
  };

  // =====================================================================
  //  PLOT UPDATE FUNCTIONS (per step)
  // =====================================================================

  function renderUncompBode(allSteps, animate) {
    const d = allSteps.find((s) => s.id === "uncomp").data;
    const w = d.wArr, b = d.bode, pad = 5;

    const magMin = Math.min(...b.magDb) - pad, magMax = Math.max(...b.magDb) + pad;
    const phMin = Math.min(...b.phaseDeg) - 15, phMax = Math.max(...b.phaseDeg) + 15;

    const ann = [];
    if (d.wgc) ann.push({ type: "vline", x: d.wgc, color: COLORS.pm, label: "ωgc=" + fmtNum(d.wgc) });

    renderPlot("mag-plot", {
      title: "Magnitude (Uncompensated)", yLabel: "Gain (dB)", xLabel: "Frequency (rad/s)",
      xMin: w[0], xMax: w[w.length - 1], yMin: magMin, yMax: magMax,
      refLines: [{ axis: "y", value: 0, color: COLORS.ref, label: "0 dB" }],
      traces: [{ x: w, y: b.magDb, color: COLORS.uncomp, width: 3, animate }],
      annotations: ann,
      legend: [{ label: "KG(jω)", color: COLORS.uncomp }],
    });

    const phAnn = [...ann];
    if (d.pm !== null && d.wgc) {
      phAnn.push({ type: "pmArc", x: d.wgc, phase: interpVal(b.phaseDeg, nearestIdx(w, d.wgc), 0), pm: d.pm, color: COLORS.pm });
    }

    renderPlot("phase-plot", {
      title: "Phase (Uncompensated)", yLabel: "Phase (deg)", xLabel: "Frequency (rad/s)",
      xMin: w[0], xMax: w[w.length - 1], yMin: phMin, yMax: phMax,
      refLines: [{ axis: "y", value: -180, color: COLORS.ref, label: "-180°" }],
      traces: [{ x: w, y: b.phaseDeg, color: COLORS.uncomp, width: 3, animate }],
      annotations: phAnn,
      legend: [{ label: "∠KG(jω)", color: COLORS.uncomp }],
    });

    renderReadout(document.getElementById("mag-readout"), [
      { label: "PM", value: d.pm !== null ? fmtDeg(d.pm) : "N/A" },
      { label: "ωgc", value: d.wgc ? fmtFreq(d.wgc) : "N/A" },
    ]);
    renderReadout(document.getElementById("phase-readout"), [
      { label: "GM", value: d.gm === Infinity ? "∞" : d.gm !== null ? fmtDb(d.gm) : "N/A" },
    ]);
  }

  function renderCtrlBode(allSteps, animate) {
    const compStep = allSteps.find((s) => s.id === "build") || allSteps.find((s) => s.id === "design");
    if (!compStep || !compStep.data.comp || compStep.data.comp.type === "none") {
      placeholder(document.getElementById("ctrl-plot"), "No compensator (specs already met).");
      return;
    }
    const c = compStep.data.comp;
    const uncompStep = allSteps.find((s) => s.id === "uncomp");
    const w = uncompStep.data.wArr;
    const bc = computeBode(c.num, c.den, w);
    const pad = 3;

    renderPlot("ctrl-plot", {
      title: "Compensator C(jω)", yLabel: "Gain (dB) / Phase (deg)", xLabel: "Frequency (rad/s)",
      xMin: w[0], xMax: w[w.length - 1],
      yMin: Math.min(Math.min(...bc.magDb) - pad, Math.min(...bc.phaseDeg) - 5),
      yMax: Math.max(Math.max(...bc.magDb) + pad, Math.max(...bc.phaseDeg) + 5),
      refLines: [{ axis: "y", value: 0, color: "rgba(148,163,184,0.35)", dash: "4 4" }],
      traces: [
        { x: w, y: bc.magDb, color: COLORS.ctrl, width: 2.5, animate },
        { x: w, y: bc.phaseDeg, color: COLORS.ctrlPhase, width: 2.5, dash: "6 4", animate },
      ],
      legend: [
        { label: "|C(jω)| dB", color: COLORS.ctrl },
        { label: "∠C(jω) deg", color: COLORS.ctrlPhase, dash: "6 4" },
      ],
    });
  }

  function renderCompBode(allSteps, animate) {
    const d = allSteps.find((s) => s.id === "compbode").data;
    const w = d.wArr, pad = 5;

    const allMag = [...d.bodeUncomp.magDb, ...d.bodeComp.magDb];
    const magMin = Math.min(...allMag) - pad, magMax = Math.max(...allMag) + pad;
    const allPh = [...d.bodeUncomp.phaseDeg, ...d.bodeComp.phaseDeg];
    const phMin = Math.min(...allPh) - 15, phMax = Math.max(...allPh) + 15;

    const ann = [];
    if (d.wgc) ann.push({ type: "vline", x: d.wgc, color: COLORS.compMag, label: "ωgc=" + fmtNum(d.wgc) });

    renderPlot("mag-plot", {
      title: "Magnitude (Compensated vs Uncompensated)", yLabel: "Gain (dB)", xLabel: "Frequency (rad/s)",
      xMin: w[0], xMax: w[w.length - 1], yMin: magMin, yMax: magMax,
      refLines: [{ axis: "y", value: 0, color: COLORS.ref, label: "0 dB" }],
      traces: [
        { x: w, y: d.bodeUncomp.magDb, color: COLORS.uncomp, width: 2, dash: "8 5", opacity: 0.6 },
        { x: w, y: d.bodeComp.magDb, color: COLORS.compMag, width: 3, animate },
      ],
      annotations: ann,
      legend: [
        { label: "KG(jω)", color: COLORS.uncomp, dash: "6 4" },
        { label: "C·KG(jω)", color: COLORS.compMag },
      ],
    });

    const phAnn = [...ann];
    if (d.pm !== null && d.wgc) {
      const idx = nearestIdx(w, d.wgc);
      phAnn.push({ type: "pmArc", x: d.wgc, phase: d.bodeComp.phaseDeg[idx], pm: d.pm, color: COLORS.compMag });
    }

    renderPlot("phase-plot", {
      title: "Phase (Compensated vs Uncompensated)", yLabel: "Phase (deg)", xLabel: "Frequency (rad/s)",
      xMin: w[0], xMax: w[w.length - 1], yMin: phMin, yMax: phMax,
      refLines: [{ axis: "y", value: -180, color: COLORS.ref, label: "-180°" }],
      traces: [
        { x: w, y: d.bodeUncomp.phaseDeg, color: COLORS.uncomp, width: 2, dash: "8 5", opacity: 0.6 },
        { x: w, y: d.bodeComp.phaseDeg, color: COLORS.compPhase, width: 3, animate },
      ],
      annotations: phAnn,
      legend: [
        { label: "∠KG", color: COLORS.uncomp, dash: "6 4" },
        { label: "∠C·KG", color: COLORS.compPhase },
      ],
    });

    renderReadout(document.getElementById("mag-readout"), [
      { label: "PM", value: d.pm !== null ? fmtDeg(d.pm) : "N/A" },
      { label: "ωgc", value: d.wgc ? fmtFreq(d.wgc) : "N/A" },
    ]);
    renderReadout(document.getElementById("phase-readout"), [
      { label: "GM", value: d.gm === Infinity ? "∞" : d.gm !== null ? fmtDb(d.gm) : "N/A" },
    ]);
  }

  function renderTimeResp(allSteps, animate) {
    const d = allSteps.find((s) => s.id === "response").data;
    const sr = d.stepResp, sru = d.stepRespUncomp;
    const allY = [...sr.outputs, ...sru.outputs];
    const yMax = Math.max(...allY.map(Math.abs), 1.5) * 1.1;
    const yMin = Math.min(0, Math.min(...allY)) - 0.1;

    const ann = [];
    if (d.metrics.ts !== null) {
      ann.push({ type: "vline", x: d.metrics.ts, color: COLORS.compMag, label: "ts=" + fmtNum(d.metrics.ts) + "s" });
    }

    renderPlot("time-plot", {
      title: "Closed-Loop Step Response", yLabel: "Output y(t)", xLabel: "Time (s)",
      logX: false, xMin: 0, xMax: d.tEnd, yMin, yMax,
      refLines: [
        { axis: "y", value: 1, color: "rgba(22,163,74,0.4)", label: "Reference", dash: "6 4" },
        { axis: "y", value: 1.02, color: "rgba(220,38,38,0.2)", dash: "3 3" },
        { axis: "y", value: 0.98, color: "rgba(220,38,38,0.2)", dash: "3 3" },
      ],
      traces: [
        { x: sru.times, y: sru.outputs, color: COLORS.uncompTime, width: 2, dash: "8 5", opacity: 0.6 },
        { x: sr.times, y: sr.outputs, color: COLORS.compTime, width: 3, animate },
      ],
      annotations: ann,
      legend: [
        { label: "Uncompensated", color: COLORS.uncompTime, dash: "6 4" },
        { label: "Compensated", color: COLORS.compTime },
      ],
    });
  }

  function plotsForStep(stepIdx, allSteps) {
    const step = allSteps[stepIdx];
    if (!step) return;

    if (stepIdx <= 2) {
      placeholder(document.getElementById("mag-plot"), "Bode magnitude plot appears at Step 4.");
      placeholder(document.getElementById("phase-plot"), "Bode phase plot appears at Step 4.");
      placeholder(document.getElementById("ctrl-plot"), "Compensator Bode appears at Step 7.");
      placeholder(document.getElementById("time-plot"), "Step response appears at Step 10.");
      renderReadout(document.getElementById("mag-readout"), []);
      renderReadout(document.getElementById("phase-readout"), []);
      return;
    }

    if (stepIdx <= 4) {
      renderUncompBode(allSteps, stepIdx === 3);
      placeholder(document.getElementById("ctrl-plot"), "Compensator Bode appears at Step 7.");
      placeholder(document.getElementById("time-plot"), "Step response appears at Step 10.");
      return;
    }

    if (stepIdx <= 6) {
      renderUncompBode(allSteps, false);
      renderCtrlBode(allSteps, stepIdx === 5 || stepIdx === 6);
      placeholder(document.getElementById("time-plot"), "Step response appears at Step 10.");
      return;
    }

    if (stepIdx <= 8) {
      renderCompBode(allSteps, stepIdx === 7);
      renderCtrlBode(allSteps, false);
      placeholder(document.getElementById("time-plot"), "Step response appears at Step 10.");
      return;
    }

    renderCompBode(allSteps, false);
    renderCtrlBode(allSteps, false);
    renderTimeResp(allSteps, true);
  }

  // =====================================================================
  //  MAIN UI CONTROLLER
  // =====================================================================

  function init() {
    const els = {
      numInput:     document.getElementById("num-input"),
      denInput:     document.getElementById("den-input"),
      mpInput:      document.getElementById("mp-input"),
      tsInput:      document.getElementById("ts-input"),
      errorType:    document.getElementById("error-type"),
      errorValue:   document.getElementById("error-value"),
      compType:     document.getElementById("comp-type"),
      safetyMargin: document.getElementById("safety-margin"),
      designBtn:    document.getElementById("design-btn"),
      resetBtn:     document.getElementById("reset-btn"),
      prevBtn:      document.getElementById("prev-btn"),
      playBtn:      document.getElementById("play-btn"),
      nextBtn:      document.getElementById("next-btn"),
      speedSlider:  document.getElementById("speed-slider"),
      statusBadge:  document.getElementById("status-badge"),
      messageBox:   document.getElementById("message-box"),
      flowchart:    document.getElementById("flowchart"),
      stepTitle:    document.getElementById("step-title"),
      stepSubtitle: document.getElementById("step-subtitle"),
      stepContent:  document.getElementById("step-content"),
    };

    let designResult = null;
    let currentStep = -1;
    let playTimer = null;
    let isPlaying = false;

    function setStatus(text, cls) {
      els.statusBadge.textContent = text;
      els.statusBadge.className = "status-badge " + cls;
    }

    function setMessage(text, cls) {
      els.messageBox.textContent = text;
      els.messageBox.className = "message-box" + (cls ? " " + cls : "");
    }

    function clearPlots() {
      placeholder(document.getElementById("mag-plot"), "Bode magnitude will appear after analysis.");
      placeholder(document.getElementById("phase-plot"), "Bode phase will appear after analysis.");
      placeholder(document.getElementById("ctrl-plot"), "Compensator Bode will appear after design.");
      placeholder(document.getElementById("time-plot"), "Step response will appear at the final step.");
      renderReadout(document.getElementById("mag-readout"), []);
      renderReadout(document.getElementById("phase-readout"), []);
    }

    function updatePlaybackButtons() {
      const hasResult = designResult !== null;
      const totalSteps = hasResult ? designResult.steps.length : 0;
      els.prevBtn.disabled = !hasResult || currentStep <= 0;
      els.nextBtn.disabled = !hasResult || currentStep >= totalSteps - 1;
      els.playBtn.disabled = !hasResult;
      els.playBtn.textContent = isPlaying ? "⏸" : "▶";
    }

    function goToStep(idx) {
      if (!designResult) return;
      const steps = designResult.steps;
      if (idx < 0 || idx >= steps.length) return;
      currentStep = idx;
      const step = steps[idx];

      const briefs = steps.map((s, i) => {
        if (i > idx) return "";
        if (s.id === "freqdom") return "PM<sub>req</sub>=" + fmtDeg(s.data.pmRequired);
        if (s.id === "gainK") return "K=" + fmtNum(s.data.K);
        if (s.id === "uncomp") return "PM=" + (s.data.pm !== null ? fmtDeg(s.data.pm) : "N/A");
        if (s.id === "check") return s.data.compType;
        if (s.id === "compbode") return "PM=" + (s.data.pm !== null ? fmtDeg(s.data.pm) : "N/A");
        return "";
      });

      renderFlowchart(els.flowchart, idx, briefs);

      const flowStep = FLOW_STEPS[idx];
      els.stepTitle.textContent = flowStep ? flowStep.label + ": " + flowStep.title : "Design Steps";
      els.stepSubtitle.textContent = "";

      const renderer = stepRenderers[step.id];
      els.stepContent.innerHTML = renderer ? renderer(step.data) : "";
      typesetMath(els.stepContent);

      plotsForStep(idx, steps);
      updatePlaybackButtons();
    }

    function getSpeed() {
      return parseFloat(els.speedSlider.value) || 1;
    }

    function playStep() {
      if (!designResult) return stopPlaying();
      if (currentStep >= designResult.steps.length - 1) return stopPlaying();
      goToStep(currentStep + 1);
      if (currentStep >= designResult.steps.length - 1) {
        stopPlaying();
        setStatus("Design Complete", "status-done");
        setMessage("Design procedure completed. Review the results above.", "msg-success");
      }
    }

    function startPlaying() {
      if (!designResult) return;
      isPlaying = true;
      updatePlaybackButtons();
      setStatus("Designing...", "status-running");
      const interval = 2000 / getSpeed();
      playTimer = setInterval(playStep, interval);
    }

    function stopPlaying() {
      isPlaying = false;
      if (playTimer) { clearInterval(playTimer); playTimer = null; }
      updatePlaybackButtons();
    }

    function doDesign() {
      stopPlaying();
      try {
        const num = parseCoeffs(els.numInput.value);
        const den = parseCoeffs(els.denInput.value);
        const mpPercent = parseFloat(els.mpInput.value);
        const ts = parseFloat(els.tsInput.value);
        const errorType = els.errorType.value;
        const errorValue = parseFloat(els.errorValue.value);
        const compChoice = els.compType.value;
        const safetyDeg = parseFloat(els.safetyMargin.value);

        if (!Number.isFinite(mpPercent) || mpPercent <= 0 || mpPercent >= 100)
          throw new Error("Overshoot must be between 0% and 100%.");
        if (!Number.isFinite(ts) || ts <= 0)
          throw new Error("Settling time must be positive.");
        if (!Number.isFinite(errorValue) || errorValue <= 0)
          throw new Error("Error constant must be positive.");
        if (!Number.isFinite(safetyDeg) || safetyDeg < 0)
          throw new Error("Safety margin must be non-negative.");

        designResult = runDesign({ num, den, mpPercent, ts, errorType, errorValue, compChoice, safetyDeg });
        currentStep = -1;
        clearPlots();
        setMessage("Design computed successfully. Press Play or step through.", "msg-success");
        setStatus("Ready", "status-running");

        goToStep(0);
        setTimeout(() => startPlaying(), 400);
      } catch (err) {
        designResult = null;
        currentStep = -1;
        clearPlots();
        renderFlowchart(els.flowchart, -1, []);
        els.stepContent.innerHTML = "";
        els.stepTitle.textContent = "Design Steps";
        els.stepSubtitle.textContent = "Fix the error and try again.";
        setStatus("Error", "status-error");
        setMessage(err.message, "msg-error");
        updatePlaybackButtons();
      }
    }

    function doReset() {
      stopPlaying();
      designResult = null;
      currentStep = -1;
      clearPlots();
      renderFlowchart(els.flowchart, -1, []);
      els.stepContent.innerHTML = "";
      els.stepTitle.textContent = "Design Steps";
      els.stepSubtitle.textContent = 'Click "Design" to begin the compensator design procedure.';
      setStatus("Ready", "status-idle");
      setMessage("");
      updatePlaybackButtons();
    }

    const PRESETS = [
      { num: "1", den: "1 4 0", mp: 20, ts: 2, et: "Kv", ev: 20, ct: "lead", sm: 10 },
      { num: "1", den: "1 5 4 0", mp: 15, ts: 4, et: "Kv", ev: 10, ct: "double-lead", sm: 10 },
      { num: "1", den: "1 6 5 0", mp: 20, ts: 12, et: "Kv", ev: 5, ct: "lag", sm: 10 },
      { num: "10", den: "1 11 10 0", mp: 20, ts: 8, et: "Kv", ev: 10, ct: "lag-lead", sm: 10 },
    ];

    function loadPreset(idx) {
      const p = PRESETS[idx];
      if (!p) return;
      els.numInput.value = p.num;
      els.denInput.value = p.den;
      els.mpInput.value = p.mp;
      els.tsInput.value = p.ts;
      els.errorType.value = p.et;
      els.errorValue.value = p.ev;
      els.compType.value = p.ct;
      els.safetyMargin.value = p.sm;
      doDesign();
    }

    els.designBtn.addEventListener("click", doDesign);
    els.resetBtn.addEventListener("click", doReset);
    els.prevBtn.addEventListener("click", () => { stopPlaying(); goToStep(currentStep - 1); });
    els.nextBtn.addEventListener("click", () => { stopPlaying(); goToStep(currentStep + 1); });
    els.playBtn.addEventListener("click", () => {
      if (isPlaying) {
        stopPlaying();
      } else {
        if (designResult && currentStep >= designResult.steps.length - 1) {
          currentStep = -1;
          clearPlots();
          goToStep(0);
        }
        startPlaying();
      }
    });
    els.speedSlider.addEventListener("input", () => {
      if (isPlaying) { stopPlaying(); startPlaying(); }
    });

    document.querySelectorAll(".preset-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = parseInt(btn.getAttribute("data-preset"), 10) - 1;
        loadPreset(idx);
      });
    });

    renderFlowchart(els.flowchart, -1, []);
    clearPlots();
    updatePlaybackButtons();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
