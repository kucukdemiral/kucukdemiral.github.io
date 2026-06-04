(function () {
  'use strict';

  var SVG_NS = 'http://www.w3.org/2000/svg';
  var BLOCK_W = 100, BLOCK_H = 50;
  var SUM_R = 16;
  var PICK_R = 4;
  var ARROW_SIZE = 8;

  // ─── Symbolic Expressions ───────────────────────────────────

  function sym(name) { return { t: 'sym', n: name }; }
  function num(val) { return { t: 'num', v: val }; }
  function mul() { return { t: 'mul', ops: [].slice.call(arguments) }; }
  function add() { return { t: 'add', ops: [].slice.call(arguments) }; }
  function neg(e) { return { t: 'neg', op: e }; }
  function div(n, d) { return { t: 'div', n: n, d: d }; }

  function exprStr(e) {
    if (!e) return '?';
    switch (e.t) {
      case 'sym': return e.n;
      case 'num': return String(e.v);
      case 'neg': return '−' + wrapIfComplex(e.op);
      case 'mul':
        return e.ops.map(function (o) { return wrapIfComplex(o); }).join(' · ');
      case 'add':
        return e.ops.map(function (o, i) {
          if (i === 0) return exprStr(o);
          if (o.t === 'neg') return ' − ' + exprStr(o.op);
          return ' + ' + exprStr(o);
        }).join('');
      case 'div':
        return wrapIfComplex(e.n) + ' / (' + exprStr(e.d) + ')';
    }
    return '?';
  }

  function wrapIfComplex(e) {
    if (e.t === 'add' || e.t === 'div') return '(' + exprStr(e) + ')';
    return exprStr(e);
  }

  function exprHtml(e) {
    if (!e) return '?';
    switch (e.t) {
      case 'sym': return '<i>' + e.n.replace(/_(\w+)/g, '<sub>$1</sub>') + '</i>';
      case 'num': return String(e.v);
      case 'neg': return '&minus;' + wrapHtmlIfComplex(e.op);
      case 'mul':
        return e.ops.map(function (o) { return wrapHtmlIfComplex(o); }).join('');
      case 'add':
        return e.ops.map(function (o, i) {
          if (i === 0) return exprHtml(o);
          if (o.t === 'neg') return ' &minus; ' + exprHtml(o.op);
          return ' + ' + exprHtml(o);
        }).join('');
      case 'div':
        return '<span class="frac">' + wrapHtmlIfComplex(e.n) +
          ' / (' + exprHtml(e.d) + ')</span>';
    }
    return '?';
  }

  function wrapHtmlIfComplex(e) {
    if (e.t === 'add' || e.t === 'div') return '(' + exprHtml(e) + ')';
    return exprHtml(e);
  }

  // ─── SVG Helpers ────────────────────────────────────────────

  function svgEl(tag, attrs) {
    var el = document.createElementNS(SVG_NS, tag);
    if (attrs) Object.keys(attrs).forEach(function (k) { el.setAttribute(k, attrs[k]); });
    return el;
  }

  function buildDefs(svg) {
    var defs = svgEl('defs');
    var marker = svgEl('marker', {
      id: 'arrowhead', viewBox: '0 0 10 7', refX: '10', refY: '3.5',
      markerWidth: String(ARROW_SIZE), markerHeight: String(ARROW_SIZE * 0.7),
      orient: 'auto-start-reverse', markerUnits: 'userSpaceOnUse'
    });
    var poly = svgEl('polygon', { points: '0 0, 10 3.5, 0 7', fill: 'var(--ink)' });
    marker.appendChild(poly);
    defs.appendChild(marker);

    var markerHL = svgEl('marker', {
      id: 'arrowhead-hl', viewBox: '0 0 10 7', refX: '10', refY: '3.5',
      markerWidth: String(ARROW_SIZE), markerHeight: String(ARROW_SIZE * 0.7),
      orient: 'auto-start-reverse', markerUnits: 'userSpaceOnUse'
    });
    var polyHL = svgEl('polygon', { points: '0 0, 10 3.5, 0 7', fill: 'var(--teal)' });
    markerHL.appendChild(polyHL);
    defs.appendChild(markerHL);

    svg.appendChild(defs);
  }

  // ─── SVG Renderer ───────────────────────────────────────────

  function renderDiagram(diagram, container, highlights) {
    highlights = highlights || {};
    var hlNodes = highlights.nodes || {};
    var hlEdges = highlights.edges || {};
    var fadeAll = highlights.fadeOthers || false;

    container.innerHTML = '';
    var svg = svgEl('svg', { viewBox: '0 0 900 420', preserveAspectRatio: 'xMidYMid meet' });
    buildDefs(svg);

    diagram.edges.forEach(function (edge) {
      var fromNode = nodeById(diagram, edge.from);
      var toNode = nodeById(diagram, edge.to);
      if (!fromNode || !toNode) return;

      var pts = edgePoints(fromNode, toNode, edge);
      var d = pointsToPath(pts);
      var g = svgEl('g', { 'class': 'bd-edge' + hlClass(edge.id, hlEdges, fadeAll) });
      var isHL = hlEdges[edge.id];
      g.appendChild(svgEl('path', {
        d: d,
        'marker-end': isHL ? 'url(#arrowhead-hl)' : 'url(#arrowhead)'
      }));

      if (edge.sign === '-' && toNode.type === 'sumjunc') {
        var lp = pts[pts.length - 2] || pts[0];
        var tp = pts[pts.length - 1];
        var mx = (lp.x + tp.x) / 2, my = (lp.y + tp.y) / 2;
        var signX = tp.x < lp.x ? mx + 10 : mx - 10;
        var signY = tp.y < lp.y ? my + 12 : my - 8;
        if (Math.abs(tp.y - lp.y) < 2) signY = my - 12;
        var st = svgEl('text', { x: signX, y: signY, 'font-size': '12', 'font-weight': '700', fill: 'var(--red)', 'text-anchor': 'middle' });
        st.textContent = '−';
        g.appendChild(st);
      }

      if (edge.label) {
        var mid = pts[Math.floor(pts.length / 2)];
        var lt = svgEl('text', { x: mid.x, y: mid.y - 10, 'font-size': '13', 'font-style': 'italic', fill: 'var(--ink-light)', 'text-anchor': 'middle' });
        lt.textContent = edge.label;
        g.appendChild(lt);
      }

      svg.appendChild(g);
    });

    diagram.nodes.forEach(function (node) {
      var g = svgEl('g', { 'class': nodeClass(node) + hlClass(node.id, hlNodes, fadeAll), transform: 'translate(' + node.x + ',' + node.y + ')' });

      switch (node.type) {
        case 'block':
          g.appendChild(svgEl('rect', { x: -BLOCK_W / 2, y: -BLOCK_H / 2, width: BLOCK_W, height: BLOCK_H, rx: 4 }));
          var txt = svgEl('text', { x: 0, y: 1 });
          txt.textContent = labelDisplay(node.label);
          g.appendChild(txt);
          break;
        case 'sumjunc':
          g.appendChild(svgEl('circle', { cx: 0, cy: 0, r: SUM_R }));
          var cross1 = svgEl('line', { x1: -6, y1: 0, x2: 6, y2: 0, stroke: 'var(--ink)', 'stroke-width': 1 });
          var cross2 = svgEl('line', { x1: 0, y1: -6, x2: 0, y2: 6, stroke: 'var(--ink)', 'stroke-width': 1 });
          g.appendChild(cross1);
          g.appendChild(cross2);
          break;
        case 'pickoff':
          g.appendChild(svgEl('circle', { cx: 0, cy: 0, r: PICK_R }));
          break;
        case 'input':
        case 'output':
          var lbl = svgEl('text', {
            x: 0, y: 1,
            'font-size': '14', 'font-style': 'italic',
            fill: 'var(--ink)', 'text-anchor': node.type === 'input' ? 'end' : 'start'
          });
          lbl.textContent = labelDisplay(node.label);
          g.appendChild(lbl);
          break;
      }

      svg.appendChild(g);
    });

    container.appendChild(svg);
  }

  function nodeById(diagram, id) {
    for (var i = 0; i < diagram.nodes.length; i++) {
      if (diagram.nodes[i].id === id) return diagram.nodes[i];
    }
    return null;
  }

  function nodeClass(node) {
    switch (node.type) {
      case 'block': return 'bd-block';
      case 'sumjunc': return 'bd-sumjunc';
      case 'pickoff': return 'bd-pickoff';
      default: return 'bd-label';
    }
  }

  function hlClass(id, hlSet, fadeOthers) {
    if (hlSet[id]) return ' bd-highlight';
    if (fadeOthers) return ' bd-fade';
    return '';
  }

  function labelDisplay(label) {
    return label.replace(/_(\w+)/g, ' $1').replace(/G_/g, 'G').replace(/H_/g, 'H');
  }

  function portOut(node) {
    switch (node.type) {
      case 'block': return { x: node.x + BLOCK_W / 2, y: node.y };
      case 'sumjunc': return { x: node.x + SUM_R, y: node.y };
      case 'pickoff': return { x: node.x, y: node.y };
      case 'input': return { x: node.x + 10, y: node.y };
      case 'output': return { x: node.x, y: node.y };
      default: return { x: node.x, y: node.y };
    }
  }

  function portIn(node) {
    switch (node.type) {
      case 'block': return { x: node.x - BLOCK_W / 2, y: node.y };
      case 'sumjunc': return { x: node.x - SUM_R, y: node.y };
      case 'pickoff': return { x: node.x, y: node.y };
      case 'input': return { x: node.x, y: node.y };
      case 'output': return { x: node.x - 10, y: node.y };
      default: return { x: node.x, y: node.y };
    }
  }

  function edgePoints(fromNode, toNode, edge) {
    var start = portOut(fromNode);
    var end = portIn(toNode);

    if (edge.waypoints && edge.waypoints.length > 0) {
      return [start].concat(edge.waypoints).concat([end]);
    }

    if (Math.abs(start.y - end.y) < 2) {
      return [start, end];
    }

    var midX = (start.x + end.x) / 2;
    return [start, { x: midX, y: start.y }, { x: midX, y: end.y }, end];
  }

  function pointsToPath(pts) {
    if (pts.length === 0) return '';
    var d = 'M ' + pts[0].x + ' ' + pts[0].y;
    for (var i = 1; i < pts.length; i++) {
      d += ' L ' + pts[i].x + ' ' + pts[i].y;
    }
    return d;
  }

  // ─── Reduction Engine ───────────────────────────────────────

  function cloneDiagram(d) {
    return JSON.parse(JSON.stringify(d));
  }

  function outEdges(diagram, nodeId) {
    return diagram.edges.filter(function (e) { return e.from === nodeId; });
  }

  function inEdges(diagram, nodeId) {
    return diagram.edges.filter(function (e) { return e.to === nodeId; });
  }

  function removeNode(diagram, nodeId) {
    diagram.nodes = diagram.nodes.filter(function (n) { return n.id !== nodeId; });
    diagram.edges = diagram.edges.filter(function (e) { return e.from !== nodeId && e.to !== nodeId; });
  }

  function findSeriesPair(diagram) {
    for (var i = 0; i < diagram.nodes.length; i++) {
      var a = diagram.nodes[i];
      if (a.type !== 'block') continue;
      var outs = outEdges(diagram, a.id);
      if (outs.length !== 1) continue;
      var b = nodeById(diagram, outs[0].to);
      if (!b || b.type !== 'block') continue;
      var insB = inEdges(diagram, b.id);
      if (insB.length !== 1) continue;
      var outsFromA = outEdges(diagram, a.id);
      var pickoffCheck = diagram.nodes.filter(function (n) {
        return n.type === 'pickoff' && inEdges(diagram, n.id).some(function (e) { return e.from === a.id; });
      });
      if (pickoffCheck.length > 0) continue;
      return { a: a, b: b, edge: outs[0] };
    }
    return null;
  }

  function applySeries(diagram, pair) {
    var a = nodeById(diagram, pair.a.id);
    var b = nodeById(diagram, pair.b.id);
    a.label = pair.a.label + pair.b.label;
    a.expr = mul(pair.a.expr, pair.b.expr);
    a.x = (pair.a.x + pair.b.x) / 2;
    a.y = pair.a.y;

    var bOuts = outEdges(diagram, b.id);
    bOuts.forEach(function (e) { e.from = a.id; });
    diagram.edges = diagram.edges.filter(function (e) { return e.id !== pair.edge.id; });
    diagram.nodes = diagram.nodes.filter(function (n) { return n.id !== b.id; });
    return diagram;
  }

  function findFeedbackLoop(diagram) {
    for (var i = 0; i < diagram.nodes.length; i++) {
      var s = diagram.nodes[i];
      if (s.type !== 'sumjunc') continue;

      var sOuts = outEdges(diagram, s.id);
      if (sOuts.length !== 1) continue;

      var forwardStart = nodeById(diagram, sOuts[0].to);
      if (!forwardStart) continue;

      var forwardBlocks = [];
      var current = forwardStart;
      var visited = {};
      while (current && current.type === 'block' && !visited[current.id]) {
        visited[current.id] = true;
        forwardBlocks.push(current);
        var cOuts = outEdges(diagram, current.id);
        if (cOuts.length !== 1) break;
        var next = nodeById(diagram, cOuts[0].to);
        if (next && next.type === 'block') {
          current = next;
        } else {
          current = next;
          break;
        }
      }

      if (forwardBlocks.length === 0) continue;

      var lastForward = forwardBlocks[forwardBlocks.length - 1];
      var lastOuts = outEdges(diagram, lastForward.id);
      var outputNode = null;
      for (var j = 0; j < lastOuts.length; j++) {
        var candidate = nodeById(diagram, lastOuts[j].to);
        if (candidate && (candidate.type === 'pickoff' || candidate.type === 'output')) {
          outputNode = candidate;
          break;
        }
      }
      if (!outputNode) {
        if (lastOuts.length === 1) {
          outputNode = lastForward;
        } else continue;
      }

      var feedbackSourceId = outputNode.type === 'pickoff' ? outputNode.id : lastForward.id;
      var fbEdges = outEdges(diagram, feedbackSourceId);

      for (var k = 0; k < fbEdges.length; k++) {
        var fbTarget = nodeById(diagram, fbEdges[k].to);
        if (!fbTarget) continue;

        var feedbackBlocks = [];
        var fc = fbTarget;
        var fbVisited = {};
        while (fc && fc.type === 'block' && !fbVisited[fc.id]) {
          fbVisited[fc.id] = true;
          feedbackBlocks.push(fc);
          var fcOuts = outEdges(diagram, fc.id);
          if (fcOuts.length !== 1) break;
          var fcNext = nodeById(diagram, fcOuts[0].to);
          if (fcNext && fcNext.type === 'block') fc = fcNext;
          else { fc = fcNext; break; }
        }

        var feedbackEnd = feedbackBlocks.length > 0
          ? outEdges(diagram, feedbackBlocks[feedbackBlocks.length - 1].id)
          : [fbEdges[k]];

        var reachesSum = feedbackEnd.some(function (e) { return e.to === s.id; });
        if (!reachesSum && feedbackBlocks.length > 0) continue;
        if (!reachesSum && feedbackBlocks.length === 0) {
          if (fbTarget.id !== s.id) continue;
        }

        var fbSign = '-';
        var sumInEdges = inEdges(diagram, s.id);
        for (var m = 0; m < sumInEdges.length; m++) {
          var src = feedbackBlocks.length > 0
            ? feedbackBlocks[feedbackBlocks.length - 1].id
            : feedbackSourceId;
          if (sumInEdges[m].from === src) {
            fbSign = sumInEdges[m].sign || '+';
            break;
          }
        }

        return {
          sumjunc: s,
          forwardBlocks: forwardBlocks,
          feedbackBlocks: feedbackBlocks,
          outputNode: outputNode,
          feedbackSourceId: feedbackSourceId,
          fbSign: fbSign,
          feedbackEdge: fbEdges[k]
        };
      }
    }
    return null;
  }

  function applyFeedback(diagram, loop) {
    var gExpr = loop.forwardBlocks.length === 1
      ? loop.forwardBlocks[0].expr
      : mul.apply(null, loop.forwardBlocks.map(function (b) { return b.expr; }));

    var hExpr;
    if (loop.feedbackBlocks.length === 0) {
      hExpr = num(1);
    } else if (loop.feedbackBlocks.length === 1) {
      hExpr = loop.feedbackBlocks[0].expr;
    } else {
      hExpr = mul.apply(null, loop.feedbackBlocks.map(function (b) { return b.expr; }));
    }

    var denomTerm = mul(gExpr, hExpr);
    var denom = loop.fbSign === '-' ? add(num(1), denomTerm) : add(num(1), neg(denomTerm));
    var closedLoop = div(gExpr, denom);

    var first = loop.forwardBlocks[0];
    first.expr = closedLoop;
    first.label = 'T';
    first.x = (loop.forwardBlocks[0].x + loop.forwardBlocks[loop.forwardBlocks.length - 1].x) / 2;

    for (var i = 1; i < loop.forwardBlocks.length; i++) {
      removeNode(diagram, loop.forwardBlocks[i].id);
    }
    loop.feedbackBlocks.forEach(function (b) { removeNode(diagram, b); });

    diagram.edges = diagram.edges.filter(function (e) { return e.id !== loop.feedbackEdge.id; });
    diagram.edges = diagram.edges.filter(function (e) {
      return !loop.feedbackBlocks.some(function (b) { return e.from === b.id || e.to === b.id; });
    });

    var sOuts = outEdges(diagram, loop.sumjunc.id);
    if (sOuts.length === 1 && sOuts[0].to === first.id) {
      var sIns = inEdges(diagram, loop.sumjunc.id);
      var forwardIn = sIns.filter(function (e) {
        return !loop.feedbackBlocks.some(function (b) { return e.from === b.id; }) &&
          e.from !== loop.feedbackSourceId;
      });
      if (forwardIn.length === 1) {
        forwardIn[0].to = first.id;
        removeNode(diagram, loop.sumjunc.id);
      }
    }

    if (loop.outputNode && loop.outputNode.type === 'pickoff') {
      var poOuts = outEdges(diagram, loop.outputNode.id);
      var poIns = inEdges(diagram, loop.outputNode.id);
      if (poOuts.length <= 1 && poIns.length <= 1) {
        if (poOuts.length === 1) poOuts[0].from = first.id;
        if (poIns.length === 1) poIns[0].to = poOuts.length > 0 ? poOuts[0].to : first.id;
        removeNode(diagram, loop.outputNode.id);
      }
    }

    return diagram;
  }

  function computeSteps(preset) {
    if (preset.steps) return preset.steps;

    var steps = [];
    var current = cloneDiagram(preset);
    var maxIter = 20;

    while (maxIter-- > 0) {
      var blocks = current.nodes.filter(function (n) { return n.type === 'block'; });
      if (blocks.length <= 1) break;

      var series = findSeriesPair(current);
      if (series) {
        var before = cloneDiagram(current);
        var hl = {};
        hl[series.a.id] = true;
        hl[series.b.id] = true;
        applySeries(current, series);
        var merged = nodeById(current, series.a.id);
        steps.push({
          rule: 'Series',
          desc: 'Combine ' + exprStr(series.a.expr) + ' and ' + exprStr(series.b.expr) + ' in series',
          exprHtml: exprHtml(merged.expr),
          before: before,
          after: cloneDiagram(current),
          hlNodes: hl
        });
        continue;
      }

      var fb = findFeedbackLoop(current);
      if (fb) {
        var beforeFb = cloneDiagram(current);
        var hlFb = {};
        fb.forwardBlocks.forEach(function (b) { hlFb[b.id] = true; });
        fb.feedbackBlocks.forEach(function (b) { hlFb[b.id] = true; });
        hlFb[fb.sumjunc.id] = true;
        applyFeedback(current, fb);
        var closedNode = current.nodes.filter(function (n) { return n.type === 'block'; })[0];
        steps.push({
          rule: 'Feedback',
          desc: (fb.fbSign === '-' ? 'Negative' : 'Positive') + ' feedback loop',
          exprHtml: closedNode ? exprHtml(closedNode.expr) : '?',
          before: beforeFb,
          after: cloneDiagram(current),
          hlNodes: hlFb
        });
        continue;
      }

      break;
    }

    return steps;
  }

  // ─── Preset Examples ────────────────────────────────────────

  var PRESETS = [
    {
      id: 'simple_fb',
      title: 'Simple Feedback',
      nodes: [
        { id: 'in', type: 'input', label: 'R(s)', x: 60, y: 200 },
        { id: 's1', type: 'sumjunc', label: '', x: 140, y: 200 },
        { id: 'g', type: 'block', label: 'G', x: 300, y: 200, expr: sym('G') },
        { id: 'po', type: 'pickoff', label: '', x: 460, y: 200 },
        { id: 'out', type: 'output', label: 'Y(s)', x: 540, y: 200 },
        { id: 'h', type: 'block', label: 'H', x: 300, y: 340, expr: sym('H') }
      ],
      edges: [
        { id: 'e0', from: 'in', to: 's1', sign: '+', label: '' },
        { id: 'e1', from: 's1', to: 'g', sign: '+', label: 'E(s)' },
        { id: 'e2', from: 'g', to: 'po', sign: '+' },
        { id: 'e3', from: 'po', to: 'out', sign: '+' },
        { id: 'e4', from: 'po', to: 'h', sign: '+', waypoints: [{ x: 460, y: 340 }] },
        { id: 'e5', from: 'h', to: 's1', sign: '-', waypoints: [{ x: 140, y: 340 }] }
      ]
    },
    {
      id: 'series_fb',
      title: 'Series + Feedback',
      nodes: [
        { id: 'in', type: 'input', label: 'R(s)', x: 40, y: 200 },
        { id: 's1', type: 'sumjunc', label: '', x: 120, y: 200 },
        { id: 'g1', type: 'block', label: 'G_1', x: 250, y: 200, expr: sym('G_1') },
        { id: 'g2', type: 'block', label: 'G_2', x: 420, y: 200, expr: sym('G_2') },
        { id: 'po', type: 'pickoff', label: '', x: 560, y: 200 },
        { id: 'out', type: 'output', label: 'Y(s)', x: 640, y: 200 },
        { id: 'h', type: 'block', label: 'H', x: 340, y: 330, expr: sym('H') }
      ],
      edges: [
        { id: 'e0', from: 'in', to: 's1', sign: '+' },
        { id: 'e1', from: 's1', to: 'g1', sign: '+' },
        { id: 'e2', from: 'g1', to: 'g2', sign: '+' },
        { id: 'e3', from: 'g2', to: 'po', sign: '+' },
        { id: 'e4', from: 'po', to: 'out', sign: '+' },
        { id: 'e5', from: 'po', to: 'h', sign: '+', waypoints: [{ x: 560, y: 330 }] },
        { id: 'e6', from: 'h', to: 's1', sign: '-', waypoints: [{ x: 120, y: 330 }] }
      ]
    },
    {
      id: 'nested_loops',
      title: 'Nested Loops',
      nodes: [
        { id: 'in', type: 'input', label: 'R(s)', x: 30, y: 200 },
        { id: 's1', type: 'sumjunc', label: '', x: 100, y: 200 },
        { id: 'g1', type: 'block', label: 'G_1', x: 220, y: 200, expr: sym('G_1') },
        { id: 's2', type: 'sumjunc', label: '', x: 340, y: 200 },
        { id: 'g2', type: 'block', label: 'G_2', x: 460, y: 200, expr: sym('G_2') },
        { id: 'po2', type: 'pickoff', label: '', x: 580, y: 200 },
        { id: 'po1', type: 'pickoff', label: '', x: 660, y: 200 },
        { id: 'out', type: 'output', label: 'Y(s)', x: 740, y: 200 },
        { id: 'h1', type: 'block', label: 'H_1', x: 460, y: 320, expr: sym('H_1') },
        { id: 'h2', type: 'block', label: 'H_2', x: 300, y: 390, expr: sym('H_2') }
      ],
      edges: [
        { id: 'e0', from: 'in', to: 's1', sign: '+' },
        { id: 'e1', from: 's1', to: 'g1', sign: '+' },
        { id: 'e2', from: 'g1', to: 's2', sign: '+' },
        { id: 'e3', from: 's2', to: 'g2', sign: '+' },
        { id: 'e4', from: 'g2', to: 'po2', sign: '+' },
        { id: 'e5', from: 'po2', to: 'po1', sign: '+' },
        { id: 'e6', from: 'po1', to: 'out', sign: '+' },
        { id: 'e7', from: 'po2', to: 'h1', sign: '+', waypoints: [{ x: 580, y: 320 }] },
        { id: 'e8', from: 'h1', to: 's2', sign: '-', waypoints: [{ x: 340, y: 320 }] },
        { id: 'e9', from: 'po1', to: 'h2', sign: '+', waypoints: [{ x: 660, y: 390 }] },
        { id: 'e10', from: 'h2', to: 's1', sign: '-', waypoints: [{ x: 100, y: 390 }] }
      ]
    },
    {
      id: 'dc_motor',
      title: 'DC Motor',
      nodes: [
        { id: 'in', type: 'input', label: 'V_in(s)', x: 30, y: 200 },
        { id: 's1', type: 'sumjunc', label: '', x: 110, y: 200 },
        { id: 'ge', type: 'block', label: '1/(Ls+R)', x: 260, y: 200, expr: div(num(1), sym('Ls+R')) },
        { id: 'gk', type: 'block', label: 'K_m', x: 430, y: 200, expr: sym('K_m') },
        { id: 'gm', type: 'block', label: '1/(Js+b)', x: 600, y: 200, expr: div(num(1), sym('Js+b')) },
        { id: 'po', type: 'pickoff', label: '', x: 740, y: 200 },
        { id: 'out', type: 'output', label: 'Ω(s)', x: 820, y: 200 },
        { id: 'kb', type: 'block', label: 'K_b', x: 430, y: 340, expr: sym('K_b') }
      ],
      edges: [
        { id: 'e0', from: 'in', to: 's1', sign: '+' },
        { id: 'e1', from: 's1', to: 'ge', sign: '+' },
        { id: 'e2', from: 'ge', to: 'gk', sign: '+' },
        { id: 'e3', from: 'gk', to: 'gm', sign: '+' },
        { id: 'e4', from: 'gm', to: 'po', sign: '+' },
        { id: 'e5', from: 'po', to: 'out', sign: '+' },
        { id: 'e6', from: 'po', to: 'kb', sign: '+', waypoints: [{ x: 740, y: 340 }] },
        { id: 'e7', from: 'kb', to: 's1', sign: '-', waypoints: [{ x: 110, y: 340 }] }
      ]
    }
  ];

  // ─── App State ──────────────────────────────────────────────

  var state = {
    presetIndex: 0,
    steps: [],
    currentStep: -1,
    originalDiagram: null
  };

  // ─── UI Controller ──────────────────────────────────────────

  function init() {
    var tabsContainer = document.getElementById('example-tabs');
    PRESETS.forEach(function (p, i) {
      var btn = document.createElement('button');
      btn.className = 'example-tab' + (i === 0 ? ' active' : '');
      btn.textContent = p.title;
      btn.dataset.index = i;
      btn.addEventListener('click', function () { loadPreset(i); });
      tabsContainer.appendChild(btn);
    });

    document.getElementById('btn-step').addEventListener('click', stepForward);
    document.getElementById('btn-all').addEventListener('click', showAll);
    document.getElementById('btn-reset').addEventListener('click', resetCurrent);

    document.addEventListener('keydown', function (e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (e.key === 'ArrowRight') stepForward();
      if (e.key === 'r' || e.key === 'R') resetCurrent();
    });

    loadPreset(0);
  }

  function loadPreset(index) {
    state.presetIndex = index;
    state.originalDiagram = cloneDiagram(PRESETS[index]);
    state.steps = computeSteps(PRESETS[index]);
    state.currentStep = -1;

    document.querySelectorAll('.example-tab').forEach(function (t, i) {
      t.classList.toggle('active', i === index);
    });

    updateUI();
  }

  function stepForward() {
    if (state.currentStep >= state.steps.length - 1) return;
    state.currentStep++;
    updateUI();
  }

  function showAll() {
    state.currentStep = state.steps.length - 1;
    updateUI();
  }

  function resetCurrent() {
    state.currentStep = -1;
    updateUI();
  }

  function updateUI() {
    var container = document.getElementById('diagram-plot');
    var stepList = document.getElementById('step-list');
    var badge = document.getElementById('rule-badge');
    var counter = document.getElementById('step-counter');
    var statusBadge = document.getElementById('status-badge');
    var btnStep = document.getElementById('btn-step');

    var step = state.currentStep >= 0 ? state.steps[state.currentStep] : null;
    var diagram;

    if (state.currentStep < 0) {
      diagram = state.originalDiagram;
      renderDiagram(diagram, container);
      badge.classList.remove('visible');
      badge.textContent = '';
      statusBadge.textContent = 'Ready';
      statusBadge.className = 'status-badge status-ready';
    } else {
      diagram = step.after;
      var hl = { nodes: step.hlNodes || {}, fadeOthers: false };
      renderDiagram(step.before, container, hl);

      setTimeout(function () {
        renderDiagram(step.after, container);
      }, 800);

      badge.textContent = step.rule;
      badge.classList.add('visible');

      if (state.currentStep >= state.steps.length - 1) {
        statusBadge.textContent = 'Complete';
        statusBadge.className = 'status-badge status-done';
      } else {
        statusBadge.textContent = 'Step ' + (state.currentStep + 1);
        statusBadge.className = 'status-badge status-reducing';
      }
    }

    btnStep.disabled = state.currentStep >= state.steps.length - 1;
    counter.textContent = 'Step ' + (state.currentStep + 1) + ' / ' + state.steps.length;

    stepList.innerHTML = '';
    state.steps.forEach(function (s, i) {
      var li = document.createElement('li');
      li.className = 'step-item';
      if (i < state.currentStep) li.classList.add('completed');
      if (i === state.currentStep) li.classList.add('active');

      if (i <= state.currentStep) {
        li.innerHTML =
          '<span class="step-rule">' + s.rule + '</span>' +
          '<span class="step-desc">' + s.desc + '</span>' +
          '<div class="step-expr">' + s.exprHtml + '</div>';
      } else {
        li.innerHTML = '<span class="step-rule">Step ' + (i + 1) + '</span><span class="step-desc">...</span>';
      }

      li.addEventListener('click', function () {
        if (i <= state.currentStep) return;
        state.currentStep = i;
        updateUI();
      });

      stepList.appendChild(li);
    });
  }

  // ─── Boot ───────────────────────────────────────────────────

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
