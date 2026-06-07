(function () {
  'use strict';

  var SPEED_MS = [1200, 800, 500, 300, 150];
  var LABELS = 'ABCDEFGH'.split('');

  var PRESETS = [
    { name: 'Simple poles',  num: [2],    den: [1,8,17,10], roots: [{re:-1,im:0},{re:-2,im:0},{re:-5,im:0}] },
    { name: 'Complex poles', num: [1],    den: [1,2,5,0],   roots: [{re:0,im:0},{re:-1,im:2},{re:-1,im:-2}] },
    { name: 'Repeated',      num: [1,-1], den: [1,2,1],     roots: [{re:-1,im:0},{re:-1,im:0}] },
    { name: 'Mixed',         num: [1,3],  den: [1,5,8,4],   roots: [{re:-1,im:0},{re:-2,im:0},{re:-2,im:0}] },
    { name: 'Step response', num: [6],    den: [1,4,3,0],   roots: [{re:0,im:0},{re:-1,im:0},{re:-3,im:0}] },
    { name: 'Two poles',     num: [2,7],  den: [1,5,4],     roots: [{re:-1,im:0},{re:-4,im:0}] }
  ];

  var state = { steps: [], currentStep: -1, isPlaying: false, playTimer: null, speed: 2 };

  // ─── Complex Arithmetic ─────────────────────────────────
  function cadd(a, b) { return { re: a.re + b.re, im: a.im + b.im }; }
  function csub(a, b) { return { re: a.re - b.re, im: a.im - b.im }; }
  function cmul(a, b) { return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }; }
  function cdiv(a, b) { var d = b.re * b.re + b.im * b.im; return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d }; }
  function cabs(a) { return Math.sqrt(a.re * a.re + a.im * a.im); }
  function csqrt(z) { var r = Math.sqrt(cabs(z)), th = Math.atan2(z.im, z.re) / 2; return { re: r * Math.cos(th), im: r * Math.sin(th) }; }

  // ─── Polynomial Operations ──────────────────────────────
  function polyEval(c, z) {
    var r = { re: c[0], im: 0 };
    for (var i = 1; i < c.length; i++) r = cadd(cmul(r, z), { re: c[i], im: 0 });
    return r;
  }
  function polyDeriv(c) {
    var n = c.length - 1, d = [];
    for (var i = 0; i < n; i++) d.push(c[i] * (n - i));
    return d.length ? d : [0];
  }

  // ─── Root Finding ───────────────────────────────────────
  function findRoots(coeffs) {
    var n = coeffs.length - 1;
    if (n <= 0) return [];
    if (n === 1) return [{ re: -coeffs[1] / coeffs[0], im: 0 }];
    if (n === 2) return quadRoots(coeffs[0], coeffs[1], coeffs[2]);
    var roots = [], def = coeffs.slice();
    while (def.length > 3) {
      var r = laguerre(def, { re: 0.4 + roots.length * 0.1, im: 0.9 });
      r = laguerre(coeffs, r);
      if (Math.abs(r.im) < 1e-6 * (1 + Math.abs(r.re))) {
        r.im = 0;
        def = synDiv(def, r.re);
        roots.push(r);
      } else {
        def = synDivQuad(def, r.re, r.im);
        roots.push({ re: r.re, im: Math.abs(r.im) });
        roots.push({ re: r.re, im: -Math.abs(r.im) });
      }
    }
    if (def.length === 3) {
      var qr = quadRoots(def[0], def[1], def[2]);
      roots.push(qr[0]); roots.push(qr[1]);
    } else if (def.length === 2) {
      roots.push({ re: -def[1] / def[0], im: 0 });
    }
    return roots;
  }

  function quadRoots(a, b, c) {
    var disc = b * b - 4 * a * c;
    if (disc >= 0) {
      var sq = Math.sqrt(disc);
      return [{ re: (-b + sq) / (2 * a), im: 0 }, { re: (-b - sq) / (2 * a), im: 0 }];
    }
    var sq = Math.sqrt(-disc);
    return [{ re: -b / (2 * a), im: sq / (2 * a) }, { re: -b / (2 * a), im: -sq / (2 * a) }];
  }

  function laguerre(c, z) {
    var n = c.length - 1;
    for (var iter = 0; iter < 200; iter++) {
      var pz = polyEval(c, z);
      if (cabs(pz) < 1e-14) break;
      var dp = polyEval(polyDeriv(c), z);
      var d2 = polyEval(polyDeriv(polyDeriv(c)), z);
      var G = cdiv(dp, pz), H = csub(cmul(G, G), cdiv(d2, pz));
      var disc = cmul({ re: n - 1, im: 0 }, csub(cmul({ re: n, im: 0 }, H), cmul(G, G)));
      var sq = csqrt(disc);
      var d1 = cadd(G, sq), d2b = csub(G, sq);
      var den = cabs(d1) > cabs(d2b) ? d1 : d2b;
      if (cabs(den) < 1e-15) break;
      var a = cdiv({ re: n, im: 0 }, den);
      z = csub(z, a);
      if (cabs(a) < 1e-12 * (1 + cabs(z))) break;
    }
    return z;
  }

  function synDiv(c, r) {
    var q = [c[0]];
    for (var i = 1; i < c.length - 1; i++) q.push(c[i] + r * q[i - 1]);
    return q;
  }

  function synDivQuad(c, re, im) {
    var a = -2 * re, b = re * re + im * im, n = c.length - 1;
    if (n < 2) return [1];
    var q = [c[0], c[1] - a * c[0]];
    for (var i = 2; i < n - 1; i++) q.push(c[i] - a * q[i - 1] - b * q[i - 2]);
    return q;
  }

  // ─── Root Grouping ──────────────────────────────────────
  function groupRoots(roots) {
    var used = [], groups = [];
    for (var i = 0; i < roots.length; i++) {
      if (used[i]) continue;
      var r = roots[i], mult = 1;
      used[i] = true;
      if (Math.abs(r.im) > 1e-6) {
        for (var j = i + 1; j < roots.length; j++) {
          if (!used[j] && Math.abs(roots[j].re - r.re) < 0.01 && Math.abs(roots[j].im + r.im) < 0.01) {
            used[j] = true; break;
          }
        }
        groups.push({ re: r.re, im: Math.abs(r.im), mult: 1, type: 'complex' });
      } else {
        for (var j = i + 1; j < roots.length; j++) {
          if (!used[j] && Math.abs(roots[j].im) < 1e-6 && Math.abs(roots[j].re - r.re) < 0.01) {
            mult++; used[j] = true;
          }
        }
        groups.push({ re: r.re, im: 0, mult: mult, type: mult > 1 ? 'repeated' : 'simple' });
      }
    }
    return groups;
  }

  // ─── Number & LaTeX Formatting ──────────────────────────
  function toFrac(x, maxD) {
    if (Math.abs(x) < 1e-12) return { n: 0, d: 1 };
    var sign = x < 0 ? -1 : 1; x = Math.abs(x);
    var bn = Math.round(x), bd = 1, be = Math.abs(x - bn);
    for (var d = 2; d <= maxD; d++) { var n = Math.round(x * d), e = Math.abs(x - n / d); if (e < be - 1e-12) { be = e; bn = n; bd = d; } }
    return be < 1e-9 ? { n: sign * bn, d: bd } : null;
  }

  function nL(v) {
    if (Math.abs(v) < 1e-12) return '0';
    var f = toFrac(v, 100);
    if (f) { if (f.d === 1) return '' + f.n; return (f.n < 0 ? '-' : '') + '\\frac{' + Math.abs(f.n) + '}{' + f.d + '}'; }
    return Number(v.toPrecision(4)).toString();
  }

  function nLi(v) {
    if (Math.abs(v) < 1e-12) return '0';
    var f = toFrac(v, 100);
    if (f) { if (f.d === 1) return '' + f.n; return Math.abs(f.n) + '/' + f.d; }
    return Number(v.toPrecision(4)).toString();
  }

  function polyL(c) {
    var n = c.length - 1, parts = [];
    for (var i = 0; i <= n; i++) {
      var v = c[i], p = n - i;
      if (Math.abs(v) < 1e-12) continue;
      var sign = v > 0 ? (parts.length ? '+' : '') : '-';
      var av = Math.abs(v);
      var cs = (Math.abs(av - 1) < 1e-9 && p > 0) ? '' : nL(av);
      var vs = p === 0 ? '' : p === 1 ? 's' : 's^{' + p + '}';
      parts.push(sign + cs + vs);
    }
    return parts.length ? parts.join('') : '0';
  }

  function factorL(re) {
    if (Math.abs(re) < 1e-9) return 's';
    return re < 0 ? '(s-' + nL(Math.abs(re)) + ')' : '(s+' + nL(re) + ')';
  }

  function quadFactorL(re, im) {
    var b = -2 * re, c = re * re + im * im;
    var parts = 's^{2}';
    if (Math.abs(b) > 1e-9) parts += (b > 0 ? '+' : '-') + (Math.abs(Math.abs(b) - 1) < 1e-9 ? '' : nL(Math.abs(b))) + 's';
    if (Math.abs(c) > 1e-9) parts += '+' + nL(c);
    return '(' + parts + ')';
  }

  // ─── Residue Computation ────────────────────────────────
  function coverUpResidue(numC, roots, skipIdx) {
    var s = roots[skipIdx];
    var num = polyEval(numC, s);
    var den = { re: 1, im: 0 };
    for (var j = 0; j < roots.length; j++) {
      if (j === skipIdx) continue;
      den = cmul(den, csub(s, roots[j]));
    }
    return cdiv(num, den);
  }

  function repeatedResidues(numC, allRoots, groupIndices, pole, mult) {
    var excl = {};
    for (var i = 0; i < groupIndices.length; i++) excl[groupIndices[i]] = true;
    function evalF(s) {
      var num = polyEval(numC, s);
      var den = { re: 1, im: 0 };
      for (var j = 0; j < allRoots.length; j++) {
        if (excl[j]) continue;
        den = cmul(den, csub(s, allRoots[j]));
      }
      if (cabs(den) < 1e-15) return { re: 0, im: 0 };
      return cdiv(num, den);
    }
    var h = 1e-4, res = [];
    for (var k = mult; k >= 1; k--) {
      var order = mult - k;
      var fv;
      if (order === 0) fv = evalF(pole);
      else if (order === 1) { var fp = evalF(cadd(pole, { re: h, im: 0 })), fm = evalF(csub(pole, { re: h, im: 0 })); fv = { re: (fp.re - fm.re) / (2 * h), im: (fp.im - fm.im) / (2 * h) }; }
      else if (order === 2) { var fp = evalF(cadd(pole, { re: h, im: 0 })), f0 = evalF(pole), fm = evalF(csub(pole, { re: h, im: 0 })); fv = { re: (fp.re - 2 * f0.re + fm.re) / (h * h), im: (fp.im - 2 * f0.im + fm.im) / (h * h) }; }
      else { fv = evalF(pole); }
      var fact = 1; for (var f = 2; f <= order; f++) fact *= f;
      res[k - 1] = { re: fv.re / fact, im: fv.im / fact };
    }
    return res;
  }

  function complexCoeffs(numC, allRoots, poleRe, poleIm) {
    var pole = { re: poleRe, im: poleIm };
    var idx = -1;
    for (var i = 0; i < allRoots.length; i++) {
      if (Math.abs(allRoots[i].re - poleRe) < 0.01 && Math.abs(allRoots[i].im - poleIm) < 0.01) { idx = i; break; }
    }
    if (idx < 0) return { B: 0, C: 0 };
    var R = coverUpResidue(numC, allRoots, idx);
    var B = 2 * R.re;
    var sigma = -poleRe, omega = poleIm;
    var C = -2 * omega * R.im + B * (-poleRe);
    return { B: B, C: C };
  }

  // ─── Step Generation ───────────────────────────────────
  function buildSteps(numC, denC, roots) {
    var groups = groupRoots(roots);
    var steps = [], lines = [];

    var facParts = [];
    for (var g = 0; g < groups.length; g++) {
      var gr = groups[g];
      if (gr.type === 'complex') facParts.push(quadFactorL(gr.re, gr.im));
      else { var f = factorL(-gr.re); facParts.push(gr.mult > 1 ? f + '^{' + gr.mult + '}' : f); }
    }
    var facDenL = facParts.join('');

    lines = [{ latex: 'F(s) = \\frac{' + polyL(numC) + '}{' + polyL(denC) + '}' }];
    steps.push({ formula: 'Starting partial fraction decomposition', lines: dl(lines) });

    lines[0] = { latex: 'F(s) = \\frac{' + polyL(numC) + '}{' + facDenL + '}' };
    steps.push({ formula: 'Factor the denominator to identify poles', lines: dl(lines) });

    var poleTexts = [];
    for (var g = 0; g < groups.length; g++) {
      var gr = groups[g];
      if (gr.type === 'complex') poleTexts.push('\\(s=' + nL(gr.re) + '\\pm' + nL(gr.im) + 'j\\) (complex pair)');
      else if (gr.mult > 1) poleTexts.push('\\(s=' + nL(gr.re) + '\\) (repeated, mult ' + gr.mult + ')');
      else poleTexts.push('\\(s=' + nL(gr.re) + '\\) (simple)');
    }
    lines.push({ html: '<b>Poles:</b> ' + poleTexts.join(', &nbsp;'), cls: 'poles-info' });
    steps.push({ formula: 'Identify and classify poles', lines: dl(lines) });

    var labelIdx = 0, terms = [];
    var pfeTerms = [];
    for (var g = 0; g < groups.length; g++) {
      var gr = groups[g];
      if (gr.type === 'complex') {
        var l1 = LABELS[labelIdx++], l2 = LABELS[labelIdx++];
        pfeTerms.push('\\frac{' + l1 + 's+' + l2 + '}{' + quadFactorL(gr.re, gr.im) + '}');
        terms.push({ group: g, type: 'complex', labels: [l1, l2] });
      } else {
        for (var k = gr.mult; k >= 1; k--) {
          var l = LABELS[labelIdx++];
          var den = factorL(-gr.re) + (k > 1 ? '^{' + k + '}' : '');
          pfeTerms.push('\\frac{' + l + '}{' + den + '}');
          terms.push({ group: g, type: gr.type, label: l, power: k });
        }
      }
    }
    lines.push({ latex: '= ' + pfeTerms.join(' + ') });
    steps.push({ formula: 'Set up the partial fraction template', lines: dl(lines) });

    var allRoots = roots;
    var rootIndices = buildRootIndices(roots, groups);
    var computedValues = {};

    for (var t = 0; t < terms.length; t++) {
      var term = terms[t], gr = groups[term.group];

      if (term.type === 'simple') {
        var rIdx = rootIndices[term.group][0];
        var pole = allRoots[rIdx];
        var otherParts = [];
        for (var j = 0; j < allRoots.length; j++) {
          if (j === rIdx) continue;
          otherParts.push(factorL(-allRoots[j].re));
        }
        var coveredDen = otherParts.join('');

        lines.push({ latex: term.label + ' = \\left.\\frac{' + polyL(numC) + '}{' + coveredDen + '}\\right|_{s=' + nL(pole.re) + '}', cls: 'step-highlight' });
        steps.push({ formula: 'Find ' + term.label + ': cover up ' + factorL(-pole.re) + ', substitute s = ' + nLi(pole.re), lines: dl(lines) });

        var val = coverUpResidue(numC, allRoots, rIdx);
        lines[lines.length - 1] = { latex: term.label + ' = ' + nL(val.re), cls: 'step-highlight' };
        steps.push({ formula: term.label + ' = ' + nLi(val.re), lines: dl(lines) });
        computedValues[term.label] = val.re;
      }

      if (term.type === 'repeated') {
        var gIdx = rootIndices[term.group];
        var pole = { re: gr.re, im: 0 };

        if (term.power === gr.mult) {
          var covParts = [];
          for (var j = 0; j < allRoots.length; j++) {
            var inGroup = false;
            for (var gi = 0; gi < gIdx.length; gi++) { if (gIdx[gi] === j) inGroup = true; }
            if (inGroup) continue;
            covParts.push(factorL(-allRoots[j].re));
          }
          lines.push({ latex: term.label + ' = \\left.' + (covParts.length ? '\\frac{' + polyL(numC) + '}{' + covParts.join('') + '}' : polyL(numC)) + '\\right|_{s=' + nL(pole.re) + '}', cls: 'step-highlight' });
          steps.push({ formula: 'Find ' + term.label + ' (highest power): cover up ' + factorL(-pole.re) + '^{' + gr.mult + '}', lines: dl(lines) });

          var residues = repeatedResidues(numC, allRoots, gIdx, pole, gr.mult);
          var val = residues[gr.mult - 1].re;
          lines[lines.length - 1] = { latex: term.label + ' = ' + nL(val), cls: 'step-highlight' };
          steps.push({ formula: term.label + ' = ' + nLi(val), lines: dl(lines) });
          computedValues[term.label] = val;

          for (var k = gr.mult - 1; k >= 1; k--) {
            var tIdx = -1;
            for (var tt = t + 1; tt < terms.length; tt++) { if (terms[tt].group === term.group && terms[tt].power === k) { tIdx = tt; break; } }
            if (tIdx < 0) continue;
            var dOrder = gr.mult - k;
            var dval = residues[k - 1].re;
            lines.push({ latex: terms[tIdx].label + ' = \\frac{1}{' + dOrder + '!}\\frac{d' + (dOrder > 1 ? '^{' + dOrder + '}' : '') + '}{ds' + (dOrder > 1 ? '^{' + dOrder + '}' : '') + '}\\left[' + factorL(-pole.re) + '^{' + gr.mult + '}F(s)\\right]_{s=' + nL(pole.re) + '} = ' + nL(dval), cls: 'step-highlight' });
            steps.push({ formula: terms[tIdx].label + ' = ' + nLi(dval) + ' (via ' + ordStr(dOrder) + ' derivative)', lines: dl(lines) });
            computedValues[terms[tIdx].label] = dval;
            terms[tIdx]._done = true;
          }
        }
        if (term._done) continue;
      }

      if (term.type === 'complex') {
        var cc = complexCoeffs(numC, allRoots, gr.re, gr.im);
        lines.push({ latex: '\\text{Complex residue at } s=' + nL(gr.re) + '+' + nL(gr.im) + 'j: \\quad ' + term.labels[0] + '=' + nL(cc.B) + ',\\;' + term.labels[1] + '=' + nL(cc.C), cls: 'step-highlight' });
        steps.push({ formula: 'Find ' + term.labels[0] + ', ' + term.labels[1] + ' for the complex pole pair', lines: dl(lines) });
        computedValues[term.labels[0]] = cc.B;
        computedValues[term.labels[1]] = cc.C;

        var sigma = -gr.re, omega = gr.im;
        var alpha = cc.B;
        var beta = (cc.C - cc.B * sigma) / omega;
        lines.push({ latex: '\\frac{' + nL(cc.B) + 's+' + nL(cc.C) + '}{' + quadFactorL(gr.re, gr.im) + '} = ' + nL(alpha) + '\\frac{(s+' + nL(sigma) + ')}{(s+' + nL(sigma) + ')^2+' + nL(omega * omega) + '} + ' + nL(beta) + '\\frac{' + nL(omega) + '}{(s+' + nL(sigma) + ')^2+' + nL(omega * omega) + '}', cls: 'step-highlight' });
        steps.push({ formula: 'Complete the square for inverse Laplace', lines: dl(lines) });
        term._alpha = alpha; term._beta = beta; term._sigma = sigma; term._omega = omega;
      }
    }

    var filledTerms = [];
    for (var i = 0; i < pfeTerms.length; i++) {
      var s = pfeTerms[i];
      for (var lbl in computedValues) {
        if (computedValues.hasOwnProperty(lbl)) {
          var re = new RegExp('\\{' + lbl + '\\}', 'g');
          var re2 = new RegExp('^' + lbl + 's', 'g');
          s = s.replace(re, '{' + nL(computedValues[lbl]) + '}');
          s = s.replace(new RegExp('\\{' + lbl + 's\\+' + lbl.charCodeAt(0) === lbl.charCodeAt(0) ? '' : ''), s);
        }
      }
      filledTerms.push(s);
    }
    var assembledL = buildAssembledPFE(terms, groups, computedValues);
    lines.push({ latex: 'F(s) = ' + assembledL, cls: 'final-result' });
    steps.push({ formula: 'Complete partial fraction expansion', lines: dl(lines) });

    lines.push({ latex: '\\text{Inverse Laplace Transform:}' });
    steps.push({ formula: 'Apply inverse Laplace to each term', lines: dl(lines) });

    var ftParts = [];
    for (var t = 0; t < terms.length; t++) {
      var term = terms[t], gr = groups[term.group];
      if (term._done) continue;
      if (term.type === 'simple' || (term.type === 'repeated')) {
        var coeff = computedValues[term.label];
        var sigma = -gr.re;
        var k = term.power;
        var timeTerm = buildTimeTerm(coeff, sigma, k);
        ftParts.push(timeTerm);
        lines.push({ latex: '\\frac{' + nL(coeff) + '}{' + factorL(-gr.re) + (k > 1 ? '^{' + k + '}' : '') + '} \\;\\longrightarrow\\; ' + timeTerm });
        steps.push({ formula: 'Inverse Laplace of term ' + term.label, lines: dl(lines) });
      }
      if (term.type === 'complex') {
        var alpha = term._alpha, beta = term._beta, sigma = term._sigma, omega = term._omega;
        var timeTerm = buildComplexTimeTerm(alpha, beta, sigma, omega);
        ftParts.push(timeTerm);
        lines.push({ latex: '\\longrightarrow\\; ' + timeTerm });
        steps.push({ formula: 'Inverse Laplace of complex pole pair', lines: dl(lines) });
      }
    }

    lines.push({ latex: '\\boxed{f(t) = ' + ftParts.join(' ') + ', \\quad t \\geq 0}', cls: 'final-result' });
    steps.push({ formula: 'Final result', lines: dl(lines) });

    return { steps: steps, groups: groups };
  }

  function buildRootIndices(roots, groups) {
    var map = {}, used = [];
    for (var g = 0; g < groups.length; g++) {
      map[g] = [];
      var gr = groups[g];
      for (var i = 0; i < roots.length; i++) {
        if (used[i]) continue;
        if (gr.type === 'complex') {
          if (Math.abs(roots[i].re - gr.re) < 0.01 && Math.abs(Math.abs(roots[i].im) - gr.im) < 0.01) {
            map[g].push(i); used[i] = true;
          }
        } else {
          if (Math.abs(roots[i].im) < 1e-6 && Math.abs(roots[i].re - gr.re) < 0.01) {
            map[g].push(i); used[i] = true;
            if (map[g].length >= gr.mult) break;
          }
        }
      }
    }
    return map;
  }

  function buildAssembledPFE(terms, groups, vals) {
    var parts = [];
    for (var t = 0; t < terms.length; t++) {
      var term = terms[t], gr = groups[term.group];
      if (term._done) continue;
      if (term.type === 'complex') {
        var B = vals[term.labels[0]], C = vals[term.labels[1]];
        var numL = nL(B) + 's';
        if (Math.abs(C) > 1e-9) numL += (C > 0 ? '+' : '') + nL(C);
        parts.push('\\frac{' + numL + '}{' + quadFactorL(gr.re, gr.im) + '}');
      } else {
        var v = vals[term.label];
        var den = factorL(-gr.re) + (term.power > 1 ? '^{' + term.power + '}' : '');
        parts.push('\\frac{' + nL(v) + '}{' + den + '}');
      }
    }
    return parts.join('+');
  }

  function buildTimeTerm(coeff, sigma, power) {
    if (Math.abs(coeff) < 1e-12) return '0';
    var cL = nL(coeff);
    var tPow = power - 1;
    var tL = tPow === 0 ? '' : tPow === 1 ? 't\\,' : 't^{' + tPow + '}\\,';
    if (tPow > 0) {
      var fact = 1; for (var i = 2; i <= tPow; i++) fact *= i;
      if (Math.abs(Math.abs(coeff) - fact) > 1e-9) { /* keep coeff */ }
    }
    var eL = Math.abs(sigma) < 1e-9 ? '' : 'e^{' + (sigma > 0 ? '-' + nL(sigma) : nL(-sigma)) + 't}';
    if (!tL && !eL) return cL;
    return cL + tL + eL;
  }

  function buildComplexTimeTerm(alpha, beta, sigma, omega) {
    var parts = [];
    var eL = 'e^{-' + nL(sigma) + 't}';
    if (Math.abs(alpha) > 1e-9) parts.push(nL(alpha) + eL + '\\cos(' + nL(omega) + 't)');
    if (Math.abs(beta) > 1e-9) parts.push((beta > 0 && parts.length ? '+' : '') + nL(beta) + eL + '\\sin(' + nL(omega) + 't)');
    return parts.length ? parts.join('') : '0';
  }

  function ordStr(n) { return n === 1 ? '1st' : n === 2 ? '2nd' : n === 3 ? '3rd' : n + 'th'; }
  function dl(lines) { return lines.map(function (l) { return { latex: l.latex, html: l.html, text: l.text, cls: l.cls }; }); }

  // ─── Rendering ──────────────────────────────────────────
  function renderStep(idx) {
    if (idx < 0 || idx >= state.steps.length) return;
    var step = state.steps[idx];
    document.getElementById('formula-bar').innerHTML = step.formula;
    var display = document.getElementById('main-display');
    display.innerHTML = '';
    for (var i = 0; i < step.lines.length; i++) {
      var line = step.lines[i];
      var div = document.createElement('div');
      div.className = 'display-line ' + (line.cls || '');
      if (line.latex) div.innerHTML = '\\[' + line.latex + '\\]';
      else if (line.html) div.innerHTML = line.html;
      else if (line.text) div.textContent = line.text;
      display.appendChild(div);
    }
    if (window.MathJax && MathJax.typesetPromise) {
      MathJax.typesetClear([display, document.getElementById('formula-bar')]);
      MathJax.typesetPromise([display, document.getElementById('formula-bar')]).catch(function (e) { console.warn('MathJax:', e); });
    }
    display.scrollTop = display.scrollHeight;
  }

  // ─── Animation Engine ──────────────────────────────────
  function stepForward() {
    if (!state.steps.length || state.currentStep >= state.steps.length - 1) { stopPlay(); return; }
    state.currentStep++;
    renderStep(state.currentStep);
    updateControls();
    if (state.currentStep >= state.steps.length - 1) showResults();
  }

  function stepBack() {
    if (state.currentStep <= 0) { state.currentStep = -1; clearDisplay(); updateControls(); return; }
    state.currentStep--;
    renderStep(state.currentStep);
    clearResults();
    updateControls();
  }

  function togglePlay() {
    if (state.isPlaying) { stopPlay(); return; }
    state.isPlaying = true; updateControls(); tick();
  }

  function tick() {
    if (!state.isPlaying) return;
    stepForward();
    if (state.currentStep < state.steps.length - 1) state.playTimer = setTimeout(tick, SPEED_MS[state.speed]);
    else stopPlay();
  }

  function stopPlay() { state.isPlaying = false; clearTimeout(state.playTimer); updateControls(); }

  function resetAnimation() {
    stopPlay(); state.currentStep = -1; clearDisplay(); clearResults(); updateControls();
  }

  function showAll() {
    stopPlay();
    state.currentStep = state.steps.length - 1;
    renderStep(state.currentStep);
    showResults(); updateControls();
  }

  function updateControls() {
    var has = state.steps.length > 0;
    var atStart = state.currentStep < 0;
    var atEnd = has && state.currentStep >= state.steps.length - 1;
    document.getElementById('btn-reset').disabled = !has || atStart;
    document.getElementById('btn-step-back').disabled = !has || atStart;
    document.getElementById('btn-step-fwd').disabled = !has || atEnd;
    document.getElementById('btn-skip').disabled = !has || atEnd;
    var pb = document.getElementById('btn-play');
    pb.disabled = !has || atEnd;
    pb.classList.toggle('playing', state.isPlaying);
    pb.innerHTML = state.isPlaying ? '&#9646;&#9646;' : '&#9654;';
  }

  function clearDisplay() {
    document.getElementById('main-display').innerHTML = '';
    document.getElementById('formula-bar').innerHTML = 'Press <b>Play</b> or <b>Step</b> to animate.';
  }

  // ─── Results ───────────────────────────────────────────
  function showResults() {
    if (!state.groups) return;
    var g = state.groups, n = 0, types = [];
    for (var i = 0; i < g.length; i++) {
      n += g[i].type === 'complex' ? 2 : g[i].mult;
      if (types.indexOf(g[i].type) < 0) types.push(g[i].type);
    }
    var allStable = true;
    for (var i = 0; i < g.length; i++) { if (g[i].re > 1e-6 || (Math.abs(g[i].re) < 1e-6 && g[i].type !== 'complex')) allStable = false; }
    document.getElementById('r-poles').textContent = n;
    document.getElementById('r-type').textContent = types.length > 1 ? 'Mixed' : types[0] === 'simple' ? 'Distinct' : types[0] === 'repeated' ? 'Repeated' : 'Complex';
    var stEl = document.getElementById('r-stable');
    stEl.textContent = allStable ? 'Yes' : 'No';
    stEl.className = 'result-value ' + (allStable ? 'stable' : 'unstable');
    document.getElementById('r-degree').textContent = n;
  }

  function clearResults() {
    ['r-poles', 'r-type', 'r-stable', 'r-degree'].forEach(function (id) {
      var el = document.getElementById(id); el.textContent = '—'; el.className = 'result-value';
    });
  }

  // ─── Input Handling ────────────────────────────────────
  function parseCoeffs(text) {
    var parts = text.trim().split(/[\s,;]+/), c = [];
    for (var i = 0; i < parts.length; i++) { var v = parseFloat(parts[i]); if (isNaN(v)) return null; c.push(v); }
    return c.length ? c : null;
  }

  function buildFromInput() {
    var nc = parseCoeffs(document.getElementById('num-input').value);
    var dc = parseCoeffs(document.getElementById('den-input').value);
    if (!nc || !dc || dc.length < 2) {
      document.getElementById('formula-bar').innerHTML = '<span style="color:var(--red)">Invalid input.</span>';
      return;
    }
    decompose(nc, dc, null);
  }

  function decompose(nc, dc, presetRoots) {
    stopPlay(); state.currentStep = -1;
    var roots = presetRoots || findRoots(dc);
    var result = buildSteps(nc, dc, roots);
    state.steps = result.steps;
    state.groups = result.groups;
    document.getElementById('fs-display').innerHTML = '\\(F(s)=\\frac{' + polyL(nc) + '}{' + polyL(dc) + '}\\)';
    if (window.MathJax && MathJax.typesetPromise) MathJax.typesetPromise([document.getElementById('fs-display')]);
    clearDisplay(); clearResults(); updateControls();
  }

  function loadPreset(p) {
    document.getElementById('num-input').value = p.num.join(' ');
    document.getElementById('den-input').value = p.den.join(' ');
    decompose(p.num, p.den, p.roots);
  }

  // ─── Init ──────────────────────────────────────────────
  function init() {
    document.getElementById('btn-build').addEventListener('click', buildFromInput);
    document.getElementById('btn-reset').addEventListener('click', resetAnimation);
    document.getElementById('btn-step-back').addEventListener('click', stepBack);
    document.getElementById('btn-play').addEventListener('click', togglePlay);
    document.getElementById('btn-step-fwd').addEventListener('click', stepForward);
    document.getElementById('btn-skip').addEventListener('click', showAll);
    document.getElementById('speed-slider').addEventListener('input', function () { state.speed = parseInt(this.value); });
    document.getElementById('num-input').addEventListener('keydown', function (e) { if (e.key === 'Enter') buildFromInput(); });
    document.getElementById('den-input').addEventListener('keydown', function (e) { if (e.key === 'Enter') buildFromInput(); });

    var container = document.getElementById('preset-btns');
    for (var i = 0; i < PRESETS.length; i++) {
      (function (p) {
        var btn = document.createElement('button');
        btn.className = 'preset-btn';
        btn.textContent = p.name;
        btn.addEventListener('click', function () { loadPreset(p); });
        container.appendChild(btn);
      })(PRESETS[i]);
    }

    loadPreset(PRESETS[5]);
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
  else init();
})();
