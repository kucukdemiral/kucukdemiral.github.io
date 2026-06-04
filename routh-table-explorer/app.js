(function () {
  'use strict';

  var EPSILON = 1e-10;
  var SPEED_MS = [1200, 800, 500, 300, 150];
  var SUP = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹'];

  var PRESETS = [
    { name: 'Stable',      coeffs: [1,6,11,6],     desc: 'poles at -1,-2,-3' },
    { name: 'Unstable',    coeffs: [1,2,3,4,5],    desc: '2 sign changes' },
    { name: 'Zero 1st col',coeffs: [1,1,2,2,3],    desc: 'ε replacement' },
    { name: 'Row of zeros',coeffs: [1,1,3,2,2],    desc: 'jω axis poles' },
    { name: 'Marginal',    coeffs: [1,2,1,2],      desc: 'poles at ±j, -2' },
    { name: '5th Order',   coeffs: [1,2,3,6,5,3],  desc: 'higher order' }
  ];

  var state = {
    coeffs: [], degree: 0, inputMode: 'coeff',
    rows: null, steps: null, specialCases: null, auxPolys: null,
    currentStep: -1, isPlaying: false, playTimer: null, speed: 2
  };

  function init() {
    document.getElementById('btn-mode-coeff').addEventListener('click', function () { setInputMode('coeff'); });
    document.getElementById('btn-mode-poly').addEventListener('click', function () { setInputMode('poly'); });
    document.getElementById('btn-build').addEventListener('click', buildFromInput);
    document.getElementById('btn-reset').addEventListener('click', resetAnimation);
    document.getElementById('btn-step-back').addEventListener('click', stepBack);
    document.getElementById('btn-play').addEventListener('click', togglePlay);
    document.getElementById('btn-step-fwd').addEventListener('click', stepForward);
    document.getElementById('btn-skip').addEventListener('click', showAll);
    document.getElementById('speed-slider').addEventListener('input', function () { state.speed = parseInt(this.value); });
    document.getElementById('coeff-input').addEventListener('keydown', function (e) { if (e.key === 'Enter') buildFromInput(); });
    document.getElementById('poly-input').addEventListener('keydown', function (e) { if (e.key === 'Enter') buildFromInput(); });

    var container = document.getElementById('preset-btns');
    for (var i = 0; i < PRESETS.length; i++) {
      (function (p) {
        var btn = document.createElement('button');
        btn.className = 'preset-btn';
        btn.textContent = p.name;
        btn.title = p.desc;
        btn.addEventListener('click', function () { loadPreset(p); });
        container.appendChild(btn);
      })(PRESETS[i]);
    }

    loadPreset(PRESETS[0]);
  }

  function setInputMode(mode) {
    state.inputMode = mode;
    document.getElementById('btn-mode-coeff').classList.toggle('active', mode === 'coeff');
    document.getElementById('btn-mode-poly').classList.toggle('active', mode === 'poly');
    document.getElementById('coeff-field').classList.toggle('hidden', mode !== 'coeff');
    document.getElementById('poly-field').classList.toggle('hidden', mode !== 'poly');
  }

  function loadPreset(p) {
    document.getElementById('coeff-input').value = p.coeffs.join(' ');
    document.getElementById('poly-input').value = formatPoly(p.coeffs);
    buildTable(p.coeffs);
  }

  function buildFromInput() {
    var coeffs;
    if (state.inputMode === 'coeff') {
      coeffs = parseCoeffs(document.getElementById('coeff-input').value);
    } else {
      coeffs = parsePoly(document.getElementById('poly-input').value);
    }
    if (!coeffs || coeffs.length < 2) {
      setFormula('<span class="special-msg">Invalid input. Enter at least 2 coefficients.</span>');
      return;
    }
    if (coeffs[0] === 0) {
      setFormula('<span class="special-msg">Leading coefficient cannot be zero.</span>');
      return;
    }
    buildTable(coeffs);
  }

  function parseCoeffs(text) {
    var parts = text.trim().split(/[\s,;]+/);
    var coeffs = [];
    for (var i = 0; i < parts.length; i++) {
      var v = parseFloat(parts[i]);
      if (isNaN(v)) return null;
      coeffs.push(v);
    }
    return coeffs.length > 0 ? coeffs : null;
  }

  function parsePoly(text) {
    text = text.replace(/\s+/g, '').toLowerCase().replace(/−/g, '-');
    if (!text) return null;
    if (text[0] !== '+' && text[0] !== '-') text = '+' + text;
    var terms = text.match(/[+-][^+-]+/g);
    if (!terms) return null;
    var maxPow = 0, pmap = {};
    for (var i = 0; i < terms.length; i++) {
      var t = terms[i], pow, coeff;
      if (t.indexOf('s') === -1) {
        pow = 0; coeff = parseFloat(t);
      } else if (t.indexOf('^') !== -1) {
        var sp = t.split('s^');
        coeff = sp[0] === '+' || sp[0] === '' ? 1 : sp[0] === '-' ? -1 : parseFloat(sp[0]);
        pow = parseInt(sp[1]);
      } else {
        var sp2 = t.split('s');
        coeff = sp2[0] === '+' || sp2[0] === '' ? 1 : sp2[0] === '-' ? -1 : parseFloat(sp2[0]);
        pow = 1;
      }
      if (isNaN(coeff) || isNaN(pow)) return null;
      pmap[pow] = (pmap[pow] || 0) + coeff;
      if (pow > maxPow) maxPow = pow;
    }
    var c = [];
    for (var p = maxPow; p >= 0; p--) c.push(pmap[p] || 0);
    return c;
  }

  function formatPoly(coeffs) {
    var n = coeffs.length - 1, parts = [];
    for (var i = 0; i <= n; i++) {
      var c = coeffs[i], pow = n - i;
      if (c === 0) continue;
      var sign = c > 0 ? (parts.length ? ' + ' : '') : (parts.length ? ' − ' : '−');
      var ac = Math.abs(c);
      var cs = (ac === 1 && pow > 0) ? '' : '' + ac;
      var vs = pow === 0 ? '' : pow === 1 ? 's' : 's' + supNum(pow);
      parts.push(sign + cs + vs);
    }
    return parts.length ? parts.join('') : '0';
  }

  function supNum(n) {
    var s = '' + n, r = '';
    for (var i = 0; i < s.length; i++) r += SUP[parseInt(s[i])];
    return r;
  }

  // ─── Routh Table Computation ────────────────────────────
  function buildTable(coeffs) {
    stopPlay();
    state.coeffs = coeffs;
    state.degree = coeffs.length - 1;
    state.currentStep = -1;
    state.auxPolys = [];

    document.getElementById('poly-display').textContent = formatPoly(coeffs);

    var n = state.degree;
    var numCols = Math.ceil((n + 1) / 2);
    var rows = [];
    var steps = [];

    var row0 = [], row1 = [];
    for (var j = 0; j < numCols; j++) {
      row0.push(j * 2 < coeffs.length ? coeffs[j * 2] : 0);
      row1.push(j * 2 + 1 < coeffs.length ? coeffs[j * 2 + 1] : 0);
    }
    rows.push({ power: n, values: row0, type: 'normal' });
    rows.push({ power: n - 1, values: row1, type: 'normal' });

    for (var j = 0; j < numCols; j++) {
      steps.push({ type: 'place', row: 0, col: j, value: row0[j], display: fmtVal(row0[j]) });
    }
    for (var j = 0; j < numCols; j++) {
      steps.push({ type: 'place', row: 1, col: j, value: row1[j], display: fmtVal(row1[j]) });
    }

    for (var i = 2; i <= n; i++) {
      var prev = rows[i - 1], pprev = rows[i - 2];
      var pivot = prev.values[0];

      if (Math.abs(pivot) < 1e-12) {
        if (isAllZero(prev.values)) {
          var auxCoeffs = formAuxPoly(pprev);
          var deriv = diffPoly(auxCoeffs);
          state.auxPolys.push({ poly: auxCoeffs, row: i - 1 });
          var newVals = auxDerivToRow(deriv, pprev.power - 1, numCols);
          steps.push({ type: 'zero-row', row: i - 1, message: 'Row of zeros detected. Auxiliary polynomial: ' + formatPolyFromFull(auxCoeffs) + '. Replacing with derivative: ' + formatPolyFromFull(deriv) });
          prev.values = newVals;
          prev.type = 'auxiliary';
          for (var j = 0; j < numCols; j++) {
            steps.push({ type: 'aux-place', row: i - 1, col: j, value: newVals[j], display: fmtVal(newVals[j]) });
          }
          pivot = prev.values[0];
        } else {
          steps.push({ type: 'epsilon', row: i - 1, col: 0, message: 'Zero in first column. Replacing with ε (small positive value).' });
          prev.values[0] = EPSILON;
          prev.type = 'epsilon';
          pivot = EPSILON;
        }
      }

      var newRow = [];
      for (var j = 0; j < numCols; j++) {
        var a = pprev.values[0] || 0;
        var b = (j + 1 < numCols) ? (pprev.values[j + 1] || 0) : 0;
        var c = prev.values[0] || 0;
        var d = (j + 1 < numCols) ? (prev.values[j + 1] || 0) : 0;
        var val = (c * b - a * d) / c;
        if (!isFinite(val)) val = 0;
        newRow.push(val);

        var src = [
          { row: i - 2, col: 0 }, { row: i - 2, col: j + 1 },
          { row: i - 1, col: 0 }, { row: i - 1, col: j + 1 }
        ];
        var fStr = '(' + fmtSrc(c) + ' × ' + fmtSrc(b) + ' − ' + fmtSrc(a) + ' × ' + fmtSrc(d) + ') ÷ ' + fmtSrc(c) + ' = ' + fmtVal(val);
        steps.push({ type: 'compute', row: i, col: j, value: val, display: fmtVal(val), sources: src, formula: fStr });
      }
      rows.push({ power: n - i, values: newRow, type: 'normal' });
    }

    state.rows = rows;
    state.steps = steps;
    renderTable();
    clearResults();
    setFormula('Press <b>Play</b> or <b>Step</b> to animate the Routh array construction.');
    updateControls();
  }

  function isAllZero(vals) {
    for (var i = 0; i < vals.length; i++) { if (Math.abs(vals[i]) > 1e-12) return false; }
    return true;
  }

  function formAuxPoly(row) {
    var p = row.power;
    var full = [];
    for (var i = 0; i <= p; i++) full.push(0);
    for (var j = 0; j < row.values.length; j++) {
      var pow = p - 2 * j;
      if (pow < 0) break;
      full[p - pow] = row.values[j];
    }
    return full;
  }

  function diffPoly(coeffs) {
    var n = coeffs.length - 1;
    if (n <= 0) return [0];
    var d = [];
    for (var i = 0; i < n; i++) d.push(coeffs[i] * (n - i));
    return d;
  }

  function auxDerivToRow(deriv, power, numCols) {
    var vals = [];
    for (var j = 0; j < numCols; j++) {
      var p = power - 2 * j;
      if (p < 0) { vals.push(0); continue; }
      var idx = (deriv.length - 1) - p;
      vals.push(idx >= 0 && idx < deriv.length ? deriv[idx] : 0);
    }
    return vals;
  }

  function formatPolyFromFull(coeffs) {
    return formatPoly(coeffs);
  }

  // ─── Number Formatting ──────────────────────────────────
  function fmtVal(v) {
    if (Math.abs(v) < 1e-12) return '0';
    if (Math.abs(v - EPSILON) < 1e-15 || Math.abs(v + EPSILON) < 1e-15) return 'ε';
    var frac = toFraction(v, 100);
    if (frac) {
      if (frac.den === 1) return '' + frac.num;
      return frac.num + '/' + frac.den;
    }
    return Number(v.toPrecision(4)).toString();
  }

  function fmtSrc(v) {
    if (Math.abs(v - EPSILON) < 1e-15) return 'ε';
    return fmtVal(v);
  }

  function toFraction(x, maxDen) {
    if (Math.abs(x) < 1e-12) return { num: 0, den: 1 };
    var sign = x < 0 ? -1 : 1;
    x = Math.abs(x);
    var best_n = Math.round(x), best_d = 1, best_err = Math.abs(x - best_n);
    for (var d = 2; d <= maxDen; d++) {
      var n = Math.round(x * d);
      var err = Math.abs(x - n / d);
      if (err < best_err - 1e-12) { best_err = err; best_n = n; best_d = d; }
    }
    if (best_err < 1e-9) return { num: sign * best_n, den: best_d };
    return null;
  }

  // ─── Table DOM ──────────────────────────────────────────
  function renderTable() {
    var tbl = document.getElementById('routh-table');
    tbl.innerHTML = '';
    if (!state.rows) return;
    var tbody = document.createElement('tbody');
    for (var i = 0; i < state.rows.length; i++) {
      var r = state.rows[i];
      var tr = document.createElement('tr');
      if (r.type === 'auxiliary') tr.className = 'row-auxiliary';
      tr.setAttribute('data-row', i);

      var lbl = document.createElement('td');
      lbl.className = 'row-label';
      lbl.innerHTML = r.power === 0 ? 's<sup>0</sup>' : r.power === 1 ? 's' : 's<sup>' + r.power + '</sup>';
      tr.appendChild(lbl);

      for (var j = 0; j < r.values.length; j++) {
        var td = document.createElement('td');
        td.className = 'cell cell-hidden';
        td.setAttribute('data-row', i);
        td.setAttribute('data-col', j);
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
    tbl.appendChild(tbody);
  }

  function getCell(row, col) {
    return document.querySelector('.routh-table td[data-row="' + row + '"][data-col="' + col + '"]');
  }

  // ─── Animation ──────────────────────────────────────────
  function revealStep(idx) {
    clearHighlights();
    var step = state.steps[idx];

    if (step.type === 'zero-row' || step.type === 'epsilon') {
      setFormula('<span class="special-msg">' + step.message + '</span>');
      if (step.type === 'epsilon') {
        var cell = getCell(step.row, 0);
        if (cell) {
          cell.textContent = 'ε';
          cell.classList.remove('cell-hidden');
          cell.classList.add('cell-visible', 'cell-epsilon');
        }
      }
      if (step.type === 'zero-row') {
        var tr = document.querySelector('tr[data-row="' + step.row + '"]');
        if (tr) tr.className = 'row-auxiliary';
      }
      return;
    }

    if (step.type === 'aux-place') {
      var cell = getCell(step.row, step.col);
      if (cell) {
        cell.textContent = step.display;
        cell.classList.remove('cell-hidden');
        cell.classList.add('cell-visible', 'cell-active');
        if (step.col === 0) applyFirstColColor(cell, step.value);
      }
      setFormula('Placing auxiliary derivative coefficient: <span class="hl">' + step.display + '</span>');
      return;
    }

    var cell = getCell(step.row, step.col);
    if (!cell) return;
    cell.textContent = step.display;
    cell.classList.remove('cell-hidden');
    cell.classList.add('cell-visible', 'cell-active');

    if (step.col === 0) applyFirstColColor(cell, step.value);

    if (step.type === 'place') {
      var power = state.degree - step.row;
      var coefIdx = step.row === 0 ? step.col * 2 : step.col * 2 + 1;
      setFormula('Placing coefficient a<sub>' + (state.degree - coefIdx) + '</sub> = <span class="hl">' + step.display + '</span>');
    }

    if (step.type === 'compute') {
      for (var s = 0; s < step.sources.length; s++) {
        var sc = getCell(step.sources[s].row, step.sources[s].col);
        if (sc) sc.classList.add('cell-source');
      }
      setFormula('Row s' + (state.rows[step.row].power > 1 ? '<sup>' + state.rows[step.row].power + '</sup>' : state.rows[step.row].power === 1 ? '' : '<sup>0</sup>') + ', col ' + (step.col + 1) + ': ' + step.formula);
    }

    addSignChangeMarkers(step.row);
  }

  function hideStep(idx) {
    var step = state.steps[idx];
    clearHighlights();
    removeAllSignChangeMarkers();

    if (step.type === 'zero-row') {
      var tr = document.querySelector('tr[data-row="' + step.row + '"]');
      if (tr) tr.className = '';
      return;
    }
    if (step.type === 'epsilon') {
      var cell = getCell(step.row, 0);
      if (cell) { cell.textContent = '0'; cell.classList.remove('cell-epsilon'); }
      return;
    }
    if (step.type === 'aux-place') {
      var cell = getCell(step.row, step.col);
      if (cell) { cell.textContent = '0'; cell.classList.remove('cell-active', 'cell-positive', 'cell-negative'); }
      return;
    }

    var cell = getCell(step.row, step.col);
    if (cell) {
      cell.textContent = '';
      cell.classList.remove('cell-visible', 'cell-active', 'cell-positive', 'cell-negative');
      cell.classList.add('cell-hidden');
    }
  }

  function clearHighlights() {
    var active = document.querySelectorAll('.cell-active, .cell-source');
    for (var i = 0; i < active.length; i++) {
      active[i].classList.remove('cell-active', 'cell-source');
    }
  }

  function applyFirstColColor(cell, value) {
    cell.classList.remove('cell-positive', 'cell-negative');
    if (value > 1e-12) cell.classList.add('cell-positive');
    else if (value < -1e-12) cell.classList.add('cell-negative');
  }

  function addSignChangeMarkers(row) {
    removeAllSignChangeMarkers();
    for (var i = 1; i <= row && i < state.rows.length; i++) {
      var prevCell = getCell(i - 1, 0);
      var curCell = getCell(i, 0);
      if (!prevCell || !curCell) continue;
      if (!curCell.classList.contains('cell-visible') || !prevCell.classList.contains('cell-visible')) continue;
      var prevVal = state.rows[i - 1].values[0];
      var curVal = state.rows[i].values[0];
      if ((prevVal > 1e-12 && curVal < -1e-12) || (prevVal < -1e-12 && curVal > 1e-12)) {
        var marker = document.createElement('span');
        marker.className = 'sign-change';
        marker.textContent = '↓';
        marker.title = 'Sign change';
        curCell.appendChild(marker);
      }
    }
  }

  function removeSignChangeMarkers(row) { removeAllSignChangeMarkers(); }

  function removeAllSignChangeMarkers() {
    var markers = document.querySelectorAll('.sign-change');
    for (var i = 0; i < markers.length; i++) markers[i].parentNode.removeChild(markers[i]);
  }

  function stepForward() {
    if (!state.steps || state.currentStep >= state.steps.length - 1) {
      stopPlay();
      if (state.steps && state.currentStep >= state.steps.length - 1) showResults();
      return;
    }
    state.currentStep++;
    revealStep(state.currentStep);
    updateControls();
    if (state.currentStep >= state.steps.length - 1) showResults();
  }

  function stepBack() {
    if (!state.steps || state.currentStep < 0) return;
    hideStep(state.currentStep);
    state.currentStep--;
    if (state.currentStep >= 0) revealStep(state.currentStep);
    else setFormula('');
    clearResults();
    updateControls();
  }

  function togglePlay() {
    if (state.isPlaying) { stopPlay(); return; }
    state.isPlaying = true;
    updateControls();
    tick();
  }

  function tick() {
    if (!state.isPlaying) return;
    stepForward();
    if (state.currentStep < state.steps.length - 1) {
      state.playTimer = setTimeout(tick, SPEED_MS[state.speed]);
    } else {
      stopPlay();
    }
  }

  function stopPlay() {
    state.isPlaying = false;
    clearTimeout(state.playTimer);
    updateControls();
  }

  function resetAnimation() {
    stopPlay();
    if (!state.steps) return;
    for (var i = state.currentStep; i >= 0; i--) hideStep(i);
    state.currentStep = -1;
    setFormula('Press <b>Play</b> or <b>Step</b> to animate.');
    clearResults();
    updateControls();
  }

  function showAll() {
    stopPlay();
    if (!state.steps) return;
    for (var i = state.currentStep + 1; i < state.steps.length; i++) {
      state.currentStep = i;
      var step = state.steps[i];
      if (step.type === 'zero-row') {
        var tr = document.querySelector('tr[data-row="' + step.row + '"]');
        if (tr) tr.className = 'row-auxiliary';
        continue;
      }
      if (step.type === 'epsilon') {
        var cell = getCell(step.row, 0);
        if (cell) { cell.textContent = 'ε'; cell.classList.remove('cell-hidden'); cell.classList.add('cell-visible', 'cell-epsilon'); }
        continue;
      }
      var cell = getCell(step.row, step.col);
      if (cell) {
        cell.textContent = step.display;
        cell.classList.remove('cell-hidden');
        cell.classList.add('cell-visible');
        if (step.col === 0) applyFirstColColor(cell, step.value);
      }
    }
    clearHighlights();
    addSignChangeMarkers(state.rows.length - 1);
    setFormula('Table complete.');
    showResults();
    updateControls();
  }

  function updateControls() {
    var hasSteps = state.steps && state.steps.length > 0;
    var atStart = state.currentStep < 0;
    var atEnd = hasSteps && state.currentStep >= state.steps.length - 1;
    document.getElementById('btn-reset').disabled = !hasSteps || atStart;
    document.getElementById('btn-step-back').disabled = !hasSteps || atStart;
    document.getElementById('btn-step-fwd').disabled = !hasSteps || atEnd;
    document.getElementById('btn-skip').disabled = !hasSteps || atEnd;
    var playBtn = document.getElementById('btn-play');
    playBtn.disabled = !hasSteps || atEnd;
    playBtn.classList.toggle('playing', state.isPlaying);
    playBtn.innerHTML = state.isPlaying ? '&#9646;&#9646;' : '&#9654;';
    playBtn.title = state.isPlaying ? 'Pause' : 'Play';
  }

  function setFormula(html) {
    document.getElementById('formula-bar').innerHTML = html;
  }

  // ─── Results Analysis ───────────────────────────────────
  function showResults() {
    if (!state.rows) return;
    var n = state.degree;
    var firstCol = [];
    for (var i = 0; i < state.rows.length; i++) firstCol.push(state.rows[i].values[0]);

    var signChanges = 0;
    for (var i = 1; i < firstCol.length; i++) {
      if ((firstCol[i - 1] > 1e-12 && firstCol[i] < -1e-12) || (firstCol[i - 1] < -1e-12 && firstCol[i] > 1e-12)) {
        signChanges++;
      }
    }

    var jwPoles = 0;
    for (var a = 0; a < state.auxPolys.length; a++) {
      jwPoles += countJwPoles(state.auxPolys[a].poly);
    }

    var rhp = signChanges;
    var lhp = n - rhp - jwPoles;

    var verdict, cls;
    if (rhp === 0 && jwPoles === 0) { verdict = 'STABLE'; cls = 'stable'; }
    else if (rhp === 0 && jwPoles > 0) { verdict = 'MARGINALLY STABLE'; cls = 'marginal'; }
    else { verdict = 'UNSTABLE'; cls = 'unstable'; }

    setResult('r-verdict', verdict, cls);
    setResult('r-sign', '' + signChanges);
    setResult('r-rhp', '' + rhp);
    setResult('r-lhp', '' + lhp);
    setResult('r-jw', '' + jwPoles);
    setResult('r-deg', '' + n);
  }

  function countJwPoles(auxCoeffs) {
    var deg = auxCoeffs.length - 1;
    if (deg <= 0) return 0;
    if (deg === 2) {
      var a = auxCoeffs[0], c = auxCoeffs[2];
      if (a !== 0 && c / a > 0) return 2;
      return 0;
    }
    if (deg === 4) {
      var a = auxCoeffs[0], b = auxCoeffs[2], c = auxCoeffs[4];
      if (a === 0) return 0;
      var disc = b * b - 4 * a * c;
      var count = 0;
      if (disc >= 0) {
        var u1 = (-b + Math.sqrt(disc)) / (2 * a);
        var u2 = (-b - Math.sqrt(disc)) / (2 * a);
        if (u1 < -1e-12) count += 2;
        if (u2 < -1e-12) count += 2;
      }
      return count;
    }
    return deg;
  }

  function setResult(id, text, cls) {
    var el = document.getElementById(id);
    el.textContent = text;
    el.className = 'result-value' + (cls ? ' ' + cls : '');
  }

  function clearResults() {
    ['r-verdict','r-sign','r-rhp','r-lhp','r-jw','r-deg'].forEach(function (id) {
      setResult(id, '—');
    });
  }

  // ─── Boot ───────────────────────────────────────────────
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
