(function () {
  'use strict';

  // ─── State ──────────────────────────────────────────────────
  var poles = [];   // [{re, im}]
  var zeros = [];   // [{re, im}]
  var gain = 1;
  var tool = 'pole'; // 'pole' | 'zero' | 'delete'
  var conjugate = true;
  var dragging = null; // {type:'pole'|'zero', index, startRe, startIm}

  // ─── S-plane view bounds ────────────────────────────────────
  var VIEW = { xMin: -8, xMax: 3, yMin: -6, yMax: 6 };

  // ─── Canvas refs ────────────────────────────────────────────
  var splaneCanvas, stepCanvas, impulseCanvas;
  var splaneCtx, stepCtx, impulseCtx;

  // ─── Presets ────────────────────────────────────────────────
  var PRESETS = {
    'first-order':   { poles: [{re:-2, im:0}], zeros: [] },
    'underdamped':   { poles: [{re:-1, im:3}, {re:-1, im:-3}], zeros: [] },
    'overdamped':    { poles: [{re:-1, im:0}, {re:-5, im:0}], zeros: [] },
    'critically':    { poles: [{re:-3, im:0}, {re:-3, im:0}], zeros: [] },
    'unstable':      { poles: [{re:0.5, im:2}, {re:0.5, im:-2}], zeros: [] },
    'resonant':      { poles: [{re:-0.2, im:4}, {re:-0.2, im:-4}], zeros: [] },
    'notch':         { poles: [{re:-1, im:3}, {re:-1, im:-3}], zeros: [{re:-0.1, im:3}, {re:-0.1, im:-3}] },
    'third-order':   { poles: [{re:-1, im:2}, {re:-1, im:-2}, {re:-4, im:0}], zeros: [] },
    'non-min-phase': { poles: [{re:-2, im:3}, {re:-2, im:-3}], zeros: [{re:0.5, im:0}] }
  };

  // ─── Init ───────────────────────────────────────────────────
  function init() {
    splaneCanvas = document.getElementById('splane-canvas');
    stepCanvas = document.getElementById('step-canvas');
    impulseCanvas = document.getElementById('impulse-canvas');
    splaneCtx = splaneCanvas.getContext('2d');
    stepCtx = stepCanvas.getContext('2d');
    impulseCtx = impulseCanvas.getContext('2d');

    resizeAllCanvases();
    window.addEventListener('resize', debounce(function () { resizeAllCanvases(); update(); }, 150));

    // S-plane mouse events
    splaneCanvas.addEventListener('mousedown', onSplaneDown);
    splaneCanvas.addEventListener('mousemove', onSplaneMove);
    splaneCanvas.addEventListener('mouseup', onSplaneUp);
    splaneCanvas.addEventListener('mouseleave', onSplaneUp);

    // Touch events for mobile
    splaneCanvas.addEventListener('touchstart', function (e) { e.preventDefault(); onSplaneDown(touchToMouse(e)); }, { passive: false });
    splaneCanvas.addEventListener('touchmove', function (e) { e.preventDefault(); onSplaneMove(touchToMouse(e)); }, { passive: false });
    splaneCanvas.addEventListener('touchend', function (e) { onSplaneUp(); }, { passive: false });

    // Toolbar
    document.getElementById('btn-add-pole').addEventListener('click', function () { setTool('pole'); });
    document.getElementById('btn-add-zero').addEventListener('click', function () { setTool('zero'); });
    document.getElementById('btn-delete').addEventListener('click', function () { setTool('delete'); });
    document.getElementById('btn-add-conj').addEventListener('click', function () {
      conjugate = !conjugate;
      this.classList.toggle('active', conjugate);
    });
    document.getElementById('btn-clear').addEventListener('click', function () {
      poles = []; zeros = []; update();
    });
    document.getElementById('btn-add-conj').classList.add('active');

    // Presets
    document.querySelectorAll('.preset-btn').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var p = PRESETS[btn.dataset.preset];
        if (p) { poles = p.poles.map(clonePt); zeros = p.zeros.map(clonePt); update(); }
      });
    });

    // Load default
    poles = [{re:-2, im:3}, {re:-2, im:-3}];
    zeros = [];
    update();
  }

  function clonePt(p) { return {re: p.re, im: p.im}; }

  function debounce(fn, ms) {
    var timer;
    return function () { clearTimeout(timer); timer = setTimeout(fn, ms); };
  }

  function touchToMouse(e) {
    var t = e.touches[0];
    var rect = splaneCanvas.getBoundingClientRect();
    return { offsetX: t.clientX - rect.left, offsetY: t.clientY - rect.top };
  }

  // ─── Canvas sizing ──────────────────────────────────────────
  function resizeAllCanvases() {
    [splaneCanvas, stepCanvas, impulseCanvas].forEach(function (c) {
      var rect = c.parentElement.getBoundingClientRect();
      var dpr = window.devicePixelRatio || 1;
      c.width = rect.width * dpr;
      c.height = rect.height * dpr;
      c.style.width = rect.width + 'px';
      c.style.height = rect.height + 'px';
      c.getContext('2d').setTransform(dpr, 0, 0, dpr, 0, 0);
    });
  }

  // ─── Coordinate transforms ─────────────────────────────────
  function sToPixel(re, im) {
    var w = splaneCanvas.width / (window.devicePixelRatio || 1);
    var h = splaneCanvas.height / (window.devicePixelRatio || 1);
    var x = (re - VIEW.xMin) / (VIEW.xMax - VIEW.xMin) * w;
    var y = (1 - (im - VIEW.yMin) / (VIEW.yMax - VIEW.yMin)) * h;
    return { x: x, y: y };
  }

  function pixelToS(px, py) {
    var w = splaneCanvas.width / (window.devicePixelRatio || 1);
    var h = splaneCanvas.height / (window.devicePixelRatio || 1);
    var re = px / w * (VIEW.xMax - VIEW.xMin) + VIEW.xMin;
    var im = (1 - py / h) * (VIEW.yMax - VIEW.yMin) + VIEW.yMin;
    return { re: re, im: im };
  }

  // ─── S-plane interaction ────────────────────────────────────
  function findNearest(px, py, threshold) {
    var best = null, bestDist = threshold || 15;
    function check(arr, type) {
      for (var i = 0; i < arr.length; i++) {
        var p = sToPixel(arr[i].re, arr[i].im);
        var d = Math.hypot(p.x - px, p.y - py);
        if (d < bestDist) { bestDist = d; best = { type: type, index: i }; }
      }
    }
    check(poles, 'pole');
    check(zeros, 'zero');
    return best;
  }

  function onSplaneDown(e) {
    var px = e.offsetX, py = e.offsetY;
    var s = pixelToS(px, py);

    if (tool === 'delete') {
      var hit = findNearest(px, py, 18);
      if (hit) {
        var arr = hit.type === 'pole' ? poles : zeros;
        var removed = arr[hit.index];
        arr.splice(hit.index, 1);
        if (conjugate && Math.abs(removed.im) > 0.05) {
          for (var i = arr.length - 1; i >= 0; i--) {
            if (Math.abs(arr[i].re - removed.re) < 0.05 && Math.abs(arr[i].im + removed.im) < 0.05) {
              arr.splice(i, 1); break;
            }
          }
        }
        update();
      }
      return;
    }

    var hit2 = findNearest(px, py, 18);
    if (hit2) {
      var arr2 = hit2.type === 'pole' ? poles : zeros;
      dragging = { type: hit2.type, index: hit2.index, startRe: arr2[hit2.index].re, startIm: arr2[hit2.index].im };
      splaneCanvas.style.cursor = 'grabbing';
      return;
    }

    // Place new
    var arr3 = tool === 'pole' ? poles : zeros;
    var snapRe = Math.round(s.re * 4) / 4;
    var snap = conjugate ? Math.round(s.im * 4) / 4 : 0;
    var adding = (conjugate && Math.abs(snap) > 0.05) ? 2 : 1;
    if (tool === 'zero' && zeros.length + adding > poles.length) return;
    arr3.push({ re: snapRe, im: snap });
    if (conjugate && Math.abs(snap) > 0.05) {
      arr3.push({ re: snapRe, im: -snap });
    }
    update();
  }

  function onSplaneMove(e) {
    if (!dragging) {
      var hit = findNearest(e.offsetX, e.offsetY, 18);
      splaneCanvas.style.cursor = hit ? 'grab' : 'crosshair';
      return;
    }
    var s = pixelToS(e.offsetX, e.offsetY);
    var snapRe = Math.round(s.re * 4) / 4;
    var snapIm = Math.round(s.im * 4) / 4;
    var arr = dragging.type === 'pole' ? poles : zeros;
    var item = arr[dragging.index];

    var isRealPoleZero = Math.abs(dragging.startIm) < 0.05;

    if (isRealPoleZero) {
      // Real pole/zero: always constrained to real axis
      item.re = snapRe;
      item.im = 0;
    } else if (conjugate) {
      // Complex pole/zero with conjugate on: move pair together
      for (var i = 0; i < arr.length; i++) {
        if (i !== dragging.index &&
            Math.abs(arr[i].re - item.re) < 0.05 &&
            Math.abs(arr[i].im + item.im) < 0.05) {
          arr[i].re = snapRe;
          arr[i].im = -snapIm;
          break;
        }
      }
      item.re = snapRe;
      item.im = snapIm;
    } else {
      // Complex pole/zero with conjugate off: constrain to real axis
      item.re = snapRe;
      item.im = 0;
    }
    update();
  }

  function onSplaneUp() {
    dragging = null;
    splaneCanvas.style.cursor = 'crosshair';
  }

  function setTool(t) {
    tool = t;
    document.getElementById('btn-add-pole').classList.toggle('active', t === 'pole');
    document.getElementById('btn-add-zero').classList.toggle('active', t === 'zero');
    document.getElementById('btn-delete').classList.toggle('active', t === 'delete');
  }

  // ─── Master update ──────────────────────────────────────────
  function update() {
    drawSplane();
    var resp = computeResponse();
    drawTimePlot(stepCtx, stepCanvas, resp.t, resp.step, 'Step Response', '#2563eb');
    drawTimePlot(impulseCtx, impulseCanvas, resp.t, resp.impulse, 'Impulse Response', '#d97706');
    updateMetrics(resp);
    updateTFDisplay();
  }

  // ─── Draw S-plane ───────────────────────────────────────────
  function drawSplane() {
    var ctx = splaneCtx;
    var w = splaneCanvas.width / (window.devicePixelRatio || 1);
    var h = splaneCanvas.height / (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, w, h);

    // RHP shading
    var rhpLeft = sToPixel(0, 0).x;
    ctx.fillStyle = 'rgba(220, 38, 38, 0.04)';
    ctx.fillRect(rhpLeft, 0, w - rhpLeft, h);

    // Grid
    ctx.strokeStyle = 'rgba(0,0,0,0.06)';
    ctx.lineWidth = 0.5;
    for (var re = Math.ceil(VIEW.xMin); re <= VIEW.xMax; re++) {
      var p = sToPixel(re, 0);
      ctx.beginPath(); ctx.moveTo(p.x, 0); ctx.lineTo(p.x, h); ctx.stroke();
    }
    for (var im = Math.ceil(VIEW.yMin); im <= VIEW.yMax; im++) {
      var p2 = sToPixel(0, im);
      ctx.beginPath(); ctx.moveTo(0, p2.y); ctx.lineTo(w, p2.y); ctx.stroke();
    }

    // Axes
    var origin = sToPixel(0, 0);
    ctx.strokeStyle = 'rgba(0,0,0,0.25)';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(origin.x, 0); ctx.lineTo(origin.x, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, origin.y); ctx.lineTo(w, origin.y); ctx.stroke();

    // Axis labels
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.font = '11px Source Sans 3, sans-serif';
    ctx.textAlign = 'center';
    for (var re2 = Math.ceil(VIEW.xMin); re2 <= VIEW.xMax; re2++) {
      if (re2 === 0) continue;
      var lp = sToPixel(re2, 0);
      ctx.fillText(re2, lp.x, origin.y + 14);
    }
    ctx.textAlign = 'right';
    for (var im2 = Math.ceil(VIEW.yMin); im2 <= VIEW.yMax; im2++) {
      if (im2 === 0) continue;
      var lp2 = sToPixel(0, im2);
      ctx.fillText(im2 + 'j', origin.x - 6, lp2.y + 4);
    }

    // Labels
    ctx.fillStyle = 'rgba(0,0,0,0.2)';
    ctx.font = '12px Source Sans 3, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('σ (Real)', w / 2, h - 5);
    ctx.save();
    ctx.translate(12, h / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('jω (Imaginary)', 0, 0);
    ctx.restore();

    // LHP / RHP labels
    ctx.font = '10px Source Sans 3, sans-serif';
    ctx.fillStyle = 'rgba(22,163,74,0.3)';
    ctx.textAlign = 'left';
    ctx.fillText('STABLE', 8, 16);
    ctx.fillStyle = 'rgba(220,38,38,0.3)';
    ctx.textAlign = 'right';
    ctx.fillText('UNSTABLE', w - 8, 16);

    // Draw constant ζ lines
    ctx.strokeStyle = 'rgba(0,0,0,0.06)';
    ctx.lineWidth = 0.5;
    ctx.setLineDash([4, 4]);
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9].forEach(function (z) {
      var angle = Math.acos(z);
      var r = 10;
      var endRe = -r * Math.cos(angle);
      var endIm = r * Math.sin(angle);
      var p1 = sToPixel(0, 0);
      var p2u = sToPixel(endRe, endIm);
      var p2l = sToPixel(endRe, -endIm);
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2u.x, p2u.y); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2l.x, p2l.y); ctx.stroke();
    });
    ctx.setLineDash([]);

    // Count multiplicities
    function countAt(arr) {
      var map = {};
      arr.forEach(function (p) {
        var key = p.re.toFixed(3) + ',' + p.im.toFixed(3);
        map[key] = (map[key] || 0) + 1;
      });
      return map;
    }
    function uniquePoints(arr) {
      var seen = {};
      var result = [];
      arr.forEach(function (p) {
        var key = p.re.toFixed(3) + ',' + p.im.toFixed(3);
        if (!seen[key]) { seen[key] = true; result.push(p); }
      });
      return result;
    }

    var zeroCounts = countAt(zeros);
    var poleCounts = countAt(poles);

    // Draw zeros (○)
    uniquePoints(zeros).forEach(function (z) {
      var p = sToPixel(z.re, z.im);
      ctx.beginPath();
      ctx.arc(p.x, p.y, 8, 0, Math.PI * 2);
      ctx.strokeStyle = '#2563eb';
      ctx.lineWidth = 2.5;
      ctx.stroke();
      var key = z.re.toFixed(3) + ',' + z.im.toFixed(3);
      var count = zeroCounts[key];
      if (count > 1) {
        ctx.font = 'bold 10px Source Sans 3, sans-serif';
        ctx.fillStyle = '#2563eb';
        ctx.textAlign = 'left';
        ctx.fillText(count, p.x + 10, p.y - 6);
      }
    });

    // Draw poles (×)
    uniquePoints(poles).forEach(function (p) {
      var pt = sToPixel(p.re, p.im);
      var s = 7;
      ctx.strokeStyle = '#dc2626';
      ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(pt.x - s, pt.y - s); ctx.lineTo(pt.x + s, pt.y + s); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(pt.x + s, pt.y - s); ctx.lineTo(pt.x - s, pt.y + s); ctx.stroke();
      var key = p.re.toFixed(3) + ',' + p.im.toFixed(3);
      var count = poleCounts[key];
      if (count > 1) {
        ctx.font = 'bold 10px Source Sans 3, sans-serif';
        ctx.fillStyle = '#dc2626';
        ctx.textAlign = 'left';
        ctx.fillText(count, pt.x + 10, pt.y - 6);
      }
    });
  }

  // ─── Compute time response ──────────────────────────────────
  function computeResponse() {
    var N = 500;
    var tMax = estimateTmax();
    var dt = tMax / N;
    var t = [], step = [], impulse = [];

    if (poles.length === 0 && zeros.length === 0) {
      for (var i = 0; i <= N; i++) { t.push(i * dt); step.push(0); impulse.push(0); }
      return { t: t, step: step, impulse: impulse };
    }

    for (var i = 0; i <= N; i++) {
      var ti = i * dt;
      t.push(ti);

      var hVal = evalImpulse(ti);
      impulse.push(hVal);

      var sVal = 0;
      for (var j = 0; j <= i; j++) {
        sVal += evalImpulse(j * dt) * dt;
      }
      step.push(sVal);
    }
    return { t: t, step: step, impulse: impulse };
  }

  function factorial(n) {
    var r = 1;
    for (var i = 2; i <= n; i++) r *= i;
    return r;
  }

  function evalTransferAt(s, excludeIndices) {
    var excl = {};
    for (var i = 0; i < excludeIndices.length; i++) excl[excludeIndices[i]] = true;
    var num = { re: 1, im: 0 };
    for (var i = 0; i < zeros.length; i++) {
      num = cmul(num, { re: s.re - zeros[i].re, im: s.im - zeros[i].im });
    }
    var den = { re: 1, im: 0 };
    for (var j = 0; j < poles.length; j++) {
      if (excl[j]) continue;
      den = cmul(den, { re: s.re - poles[j].re, im: s.im - poles[j].im });
    }
    if (Math.abs(den.re) < 1e-15 && Math.abs(den.im) < 1e-15) return { re: 0, im: 0 };
    var r = cdiv(num, den);
    return { re: r.re * gain, im: r.im * gain };
  }

  function nthDerivative(s, excludeIndices, n) {
    var h = 1e-4;
    function f(ds) { return evalTransferAt({ re: s.re + ds, im: s.im }, excludeIndices); }
    if (n === 0) return f(0);
    if (n === 1) {
      var fp = f(h), fm = f(-h);
      return { re: (fp.re - fm.re) / (2 * h), im: (fp.im - fm.im) / (2 * h) };
    }
    if (n === 2) {
      var fp = f(h), f0 = f(0), fm = f(-h);
      return { re: (fp.re - 2 * f0.re + fm.re) / (h * h), im: (fp.im - 2 * f0.im + fm.im) / (h * h) };
    }
    if (n === 3) {
      var a = f(2 * h), b = f(h), c = f(-h), d = f(-2 * h);
      var dd = 2 * h * h * h;
      return { re: (a.re - 2 * b.re + 2 * c.re - d.re) / dd, im: (a.im - 2 * b.im + 2 * c.im - d.im) / dd };
    }
    var a = f(2 * h), b = f(h), c = f(0), d = f(-h), e = f(-2 * h);
    var dd = h * h * h * h;
    return { re: (a.re - 4 * b.re + 6 * c.re - 4 * d.re + e.re) / dd, im: (a.im - 4 * b.im + 6 * c.im - 4 * d.im + e.im) / dd };
  }

  function evalImpulse(t) {
    if (t < 0) return 0;
    var val = 0;
    var assigned = [];
    var groups = [];
    for (var i = 0; i < poles.length; i++) {
      if (assigned[i]) continue;
      var grp = { re: poles[i].re, im: poles[i].im, indices: [i] };
      for (var j = i + 1; j < poles.length; j++) {
        if (!assigned[j] && Math.abs(poles[j].re - poles[i].re) < 0.05 && Math.abs(poles[j].im - poles[i].im) < 0.05) {
          grp.indices.push(j);
          assigned[j] = true;
        }
      }
      assigned[i] = true;
      groups.push(grp);
    }

    var processed = [];
    for (var g = 0; g < groups.length; g++) {
      if (processed[g]) continue;
      processed[g] = true;
      var grp = groups[g];
      var p = { re: grp.re, im: grp.im };
      var m = grp.indices.length;

      if (Math.abs(p.im) < 0.001) {
        var contribution = 0;
        for (var k = 1; k <= m; k++) {
          var d = nthDerivative(p, grp.indices, m - k);
          contribution += (d.re / factorial(m - k)) * Math.pow(t, k - 1) / factorial(k - 1);
        }
        val += contribution * Math.exp(p.re * t);
      } else {
        for (var cg = g + 1; cg < groups.length; cg++) {
          if (!processed[cg] && Math.abs(groups[cg].re - p.re) < 0.05 && Math.abs(groups[cg].im + p.im) < 0.05) {
            processed[cg] = true;
            break;
          }
        }
        var sumRe = 0, sumIm = 0;
        for (var k = 1; k <= m; k++) {
          var d = nthDerivative(p, grp.indices, m - k);
          var fact = factorial(m - k);
          var tc = Math.pow(t, k - 1) / factorial(k - 1);
          sumRe += (d.re / fact) * tc;
          sumIm += (d.im / fact) * tc;
        }
        val += 2 * Math.exp(p.re * t) * (sumRe * Math.cos(p.im * t) - sumIm * Math.sin(p.im * t));
      }
    }
    return val;
  }

  function cmul(a, b) { return { re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re }; }
  function cdiv(a, b) {
    var d = b.re * b.re + b.im * b.im;
    return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
  }

  function estimateTmax() {
    if (poles.length === 0) return 10;
    var minRe = 0;
    poles.forEach(function (p) { if (p.re < minRe) minRe = p.re; });
    var maxIm = 0;
    poles.forEach(function (p) { if (Math.abs(p.im) > maxIm) maxIm = Math.abs(p.im); });
    if (minRe >= 0) return 10;
    var tau = -1 / minRe;
    return Math.min(Math.max(5 * tau, 2 * Math.PI / Math.max(maxIm, 0.5)), 30);
  }

  // ─── Draw time plot ─────────────────────────────────────────
  function drawTimePlot(ctx, canvas, tArr, yArr, title, color) {
    var w = canvas.width / (window.devicePixelRatio || 1);
    var h = canvas.height / (window.devicePixelRatio || 1);
    var pad = { left: 45, right: 15, top: 10, bottom: 25 };
    var pw = w - pad.left - pad.right;
    var ph = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    if (tArr.length === 0) return;

    var tMin = 0, tMax = tArr[tArr.length - 1];
    var yMin = Infinity, yMax = -Infinity;
    for (var i = 0; i < yArr.length; i++) {
      if (isFinite(yArr[i])) {
        if (yArr[i] < yMin) yMin = yArr[i];
        if (yArr[i] > yMax) yMax = yArr[i];
      }
    }
    if (yMin === yMax) { yMin -= 1; yMax += 1; }
    var yPad = (yMax - yMin) * 0.1;
    yMin -= yPad; yMax += yPad;

    function mapX(t) { return pad.left + (t - tMin) / (tMax - tMin) * pw; }
    function mapY(y) { return pad.top + (1 - (y - yMin) / (yMax - yMin)) * ph; }

    // Grid
    ctx.strokeStyle = 'rgba(0,0,0,0.06)';
    ctx.lineWidth = 0.5;
    var nyt = 5;
    for (var j = 0; j <= nyt; j++) {
      var yv = yMin + j * (yMax - yMin) / nyt;
      var py = mapY(yv);
      ctx.beginPath(); ctx.moveTo(pad.left, py); ctx.lineTo(w - pad.right, py); ctx.stroke();
      ctx.fillStyle = 'rgba(0,0,0,0.3)';
      ctx.font = '10px Source Sans 3, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(yv.toFixed(2), pad.left - 4, py + 3);
    }

    // Zero line
    if (yMin < 0 && yMax > 0) {
      ctx.strokeStyle = 'rgba(0,0,0,0.15)';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.left, mapY(0)); ctx.lineTo(w - pad.right, mapY(0)); ctx.stroke();
    }

    // Time axis labels
    ctx.fillStyle = 'rgba(0,0,0,0.3)';
    ctx.font = '10px Source Sans 3, sans-serif';
    ctx.textAlign = 'center';
    var ntx = 5;
    for (var k = 0; k <= ntx; k++) {
      var tv = tMin + k * (tMax - tMin) / ntx;
      ctx.fillText(tv.toFixed(1) + 's', mapX(tv), h - 4);
    }

    // Plot line
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    var started = false;
    for (var i = 0; i < tArr.length; i++) {
      if (!isFinite(yArr[i])) continue;
      var clamped = Math.max(yMin - 1, Math.min(yMax + 1, yArr[i]));
      var px = mapX(tArr[i]);
      var py2 = mapY(clamped);
      if (!started) { ctx.moveTo(px, py2); started = true; }
      else ctx.lineTo(px, py2);
    }
    ctx.stroke();

    // Steady state line for step response
    if (title === 'Step Response' && poles.length > 0) {
      var dcGain = computeDCGain();
      if (isFinite(dcGain) && isStable()) {
        ctx.setLineDash([4, 4]);
        ctx.strokeStyle = 'rgba(0,0,0,0.2)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(pad.left, mapY(dcGain));
        ctx.lineTo(w - pad.right, mapY(dcGain));
        ctx.stroke();
        ctx.setLineDash([]);
      }
    }
  }

  // ─── Metrics ────────────────────────────────────────────────
  function computeDCGain() {
    if (poles.length === 0) return 0;
    var hasOriginPole = poles.some(function (p) { return Math.abs(p.re) < 0.01 && Math.abs(p.im) < 0.01; });
    if (hasOriginPole) return Infinity;

    var numDC = { re: 1, im: 0 };
    zeros.forEach(function (z) { numDC = cmul(numDC, { re: -z.re, im: -z.im }); });
    var denDC = { re: 1, im: 0 };
    poles.forEach(function (p) { denDC = cmul(denDC, { re: -p.re, im: -p.im }); });
    var dc = cdiv(numDC, denDC);
    return dc.re * gain;
  }

  function isStable() {
    return poles.every(function (p) { return p.re < -0.001; });
  }

  function getDominantPoles() {
    if (poles.length === 0) return null;
    var sorted = poles.slice().sort(function (a, b) { return b.re - a.re; });
    return sorted[0];
  }

  function updateMetrics(resp) {
    var stable = isStable();
    var dom = getDominantPoles();

    document.getElementById('m-stable').textContent = poles.length === 0 ? '—' : (stable ? 'Yes' : 'No');
    document.getElementById('m-stable').className = 'metric-value ' + (poles.length === 0 ? '' : (stable ? 'stable' : 'unstable'));
    document.getElementById('m-order').textContent = poles.length || '—';

    if (!dom || poles.length === 0) {
      ['m-dcgain', 'm-zeta', 'm-wn', 'm-overshoot', 'm-rise', 'm-settle'].forEach(function (id) {
        document.getElementById(id).textContent = '—';
      });
      return;
    }

    var dc = computeDCGain();
    document.getElementById('m-dcgain').textContent = isFinite(dc) ? dc.toFixed(3) : '∞';

    var wn = Math.sqrt(dom.re * dom.re + dom.im * dom.im);
    var zeta = wn > 0.001 ? -dom.re / wn : 1;
    document.getElementById('m-wn').textContent = wn.toFixed(2) + ' rad/s';
    document.getElementById('m-zeta').textContent = zeta.toFixed(3);

    if (stable && zeta > 0 && zeta < 1) {
      var os = 100 * Math.exp(-Math.PI * zeta / Math.sqrt(1 - zeta * zeta));
      document.getElementById('m-overshoot').textContent = os.toFixed(1) + '%';
      var tr = (Math.PI - Math.acos(zeta)) / (wn * Math.sqrt(1 - zeta * zeta));
      document.getElementById('m-rise').textContent = tr.toFixed(2) + ' s';
      var ts = 4 / (zeta * wn);
      document.getElementById('m-settle').textContent = ts.toFixed(2) + ' s';
    } else if (stable && zeta >= 1) {
      document.getElementById('m-overshoot').textContent = '0%';
      document.getElementById('m-rise').textContent = '—';
      document.getElementById('m-settle').textContent = (4 / Math.abs(dom.re)).toFixed(2) + ' s';
    } else {
      document.getElementById('m-overshoot').textContent = '—';
      document.getElementById('m-rise').textContent = '—';
      document.getElementById('m-settle').textContent = '—';
    }
  }

  function updateTFDisplay() {
    var el = document.getElementById('tf-display');
    if (poles.length === 0 && zeros.length === 0) {
      el.textContent = 'G(s) = ?';
      return;
    }

    var num = zeros.length === 0 ? '1' : zeros.map(function (z) {
      if (Math.abs(z.im) < 0.01) return '(s + ' + (-z.re).toFixed(1) + ')';
      return '(s² + ' + (-2 * z.re).toFixed(1) + 's + ' + (z.re * z.re + z.im * z.im).toFixed(1) + ')';
    }).filter(function (v, i, a) { return a.indexOf(v) === i; }).join('');

    var den = poles.map(function (p) {
      if (Math.abs(p.im) < 0.01) return '(s + ' + (-p.re).toFixed(1) + ')';
      return '(s² + ' + (-2 * p.re).toFixed(1) + 's + ' + (p.re * p.re + p.im * p.im).toFixed(1) + ')';
    }).filter(function (v, i, a) { return a.indexOf(v) === i; }).join('');

    el.textContent = 'G(s) = ' + num + ' / ' + den;
  }

  // ─── Boot ───────────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
