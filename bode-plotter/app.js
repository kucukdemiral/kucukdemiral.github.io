(function () {
  "use strict";

  const COEFFICIENT_SPLIT_RE = /[\s,;]+/;
  const DEG_PER_RAD = 180 / Math.PI;
  const SAMPLE_COUNT = 900;
  const SVG_NS = "http://www.w3.org/2000/svg";

  const COLORS = {
    magnitude: "#0f766e",
    asymMagnitude: "#b45309",
    phase: "#1d4ed8",
    asymPhase: "#dc2626",
    pole: "#b91c1c",
    zero: "#0f766e",
    guide: "#94a3b8",
    paper: "rgba(255,252,247,0)",
    plot: "rgba(255,255,255,0.65)",
    grid: "rgba(148,163,184,0.32)",
  };
  const COMPONENT_COLORS = [
    "#64748b",
    "#7c3aed",
    "#0891b2",
    "#059669",
    "#d97706",
    "#dc2626",
    "#4f46e5",
    "#0f766e",
  ];

  function complex(re, im = 0) {
    return { re, im };
  }

  function add(a, b) {
    return complex(a.re + b.re, a.im + b.im);
  }

  function sub(a, b) {
    return complex(a.re - b.re, a.im - b.im);
  }

  function mul(a, b) {
    return complex(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
  }

  function div(a, b) {
    const denominator = b.re * b.re + b.im * b.im;
    return complex(
      (a.re * b.re + a.im * b.im) / denominator,
      (a.im * b.re - a.re * b.im) / denominator
    );
  }

  function scale(a, scalar) {
    return complex(a.re * scalar, a.im * scalar);
  }

  function absComplex(a) {
    return Math.hypot(a.re, a.im);
  }

  function angleComplex(a) {
    return Math.atan2(a.im, a.re);
  }

  function fromPolar(radius, angle) {
    return complex(radius * Math.cos(angle), radius * Math.sin(angle));
  }

  function expImag(theta) {
    return complex(Math.cos(theta), Math.sin(theta));
  }

  function parseCoefficients(text) {
    const cleaned = text.trim().replace(/^[[(]+|[\])]+$/g, "");
    if (!cleaned) {
      throw new Error("Enter at least one numerator and denominator coefficient.");
    }

    const tokens = cleaned.split(COEFFICIENT_SPLIT_RE).filter(Boolean);
    if (!tokens.length) {
      throw new Error("Could not parse the coefficient list.");
    }

    const coeffs = tokens.map((token) => {
      const value = Number(token);
      if (!Number.isFinite(value)) {
        throw new Error(`Invalid coefficient "${token}".`);
      }
      return value;
    });

    let firstNonZero = 0;
    while (firstNonZero < coeffs.length && Math.abs(coeffs[firstNonZero]) < 1e-12) {
      firstNonZero += 1;
    }

    if (firstNonZero >= coeffs.length) {
      throw new Error("A polynomial cannot be identically zero.");
    }

    return coeffs.slice(firstNonZero);
  }

  function parseDelay(text) {
    const cleaned = text.trim();
    if (cleaned === "") {
      return 0;
    }

    const value = Number(cleaned);
    if (!Number.isFinite(value)) {
      throw new Error("Delay must be a numeric value in seconds.");
    }
    if (value < 0) {
      throw new Error("Delay must be zero or positive.");
    }
    return value;
  }

  function evaluatePolynomial(coefficients, z) {
    let result = complex(coefficients[0], 0);
    for (let index = 1; index < coefficients.length; index += 1) {
      result = add(mul(result, z), complex(coefficients[index], 0));
    }
    return result;
  }

  function polyEvalWithDeriv(coefficients, z) {
    const n = coefficients.length - 1;
    let p = complex(coefficients[0], 0);
    let dp = complex(0, 0);
    let d2p = complex(0, 0);
    for (let i = 1; i <= n; i++) {
      d2p = add(mul(d2p, z), scale(dp, 2));
      dp = add(mul(dp, z), p);
      p = add(mul(p, z), complex(coefficients[i], 0));
    }
    return { p, dp, d2p };
  }

  function laguerreStep(coefficients, z) {
    const n = coefficients.length - 1;
    const { p, dp, d2p } = polyEvalWithDeriv(coefficients, z);
    if (absComplex(p) < 1e-15) return { root: z, converged: true };
    const g = div(dp, p);
    const g2 = mul(g, g);
    const h = sub(g2, div(d2p, p));
    const disc = scale(sub(scale(h, n), g2), n - 1);
    const sq = complex(Math.sqrt(absComplex(disc)), 0);
    const ang = angleComplex(disc);
    const sqrtDisc = complex(Math.sqrt(absComplex(disc)) * Math.cos(ang / 2), Math.sqrt(absComplex(disc)) * Math.sin(ang / 2));
    const denom1 = add(g, sqrtDisc);
    const denom2 = sub(g, sqrtDisc);
    const denom = absComplex(denom1) >= absComplex(denom2) ? denom1 : denom2;
    if (absComplex(denom) < 1e-30) return { root: z, converged: false };
    const a = div(complex(n, 0), denom);
    return { root: sub(z, a), converged: false };
  }

  function laguerreRoot(coefficients, initial) {
    let z = initial;
    for (let iter = 0; iter < 200; iter++) {
      const { root, converged } = laguerreStep(coefficients, z);
      if (converged || absComplex(sub(root, z)) < 1e-13 * Math.max(1, absComplex(z))) {
        return root;
      }
      z = root;
    }
    return z;
  }

  function syntheticDivide(coefficients, root) {
    const result = [coefficients[0]];
    for (let i = 1; i < coefficients.length - 1; i++) {
      result.push(coefficients[i] + result[i - 1] * root);
    }
    return result;
  }

  function syntheticDivideComplex(coefficients, root) {
    const result = [complex(coefficients[0], 0)];
    for (let i = 1; i < coefficients.length - 1; i++) {
      result.push(add(complex(coefficients[i], 0), mul(root, result[i - 1])));
    }
    return result.map((c) => c.re);
  }

  function polynomialRoots(coefficients) {
    const degree = coefficients.length - 1;
    if (degree <= 0) {
      return [];
    }
    if (degree === 1) {
      return [complex(-coefficients[1] / coefficients[0], 0)];
    }
    if (degree === 2) {
      return quadraticRoots(coefficients);
    }

    const leading = coefficients[0];
    const monic = coefficients.map((value) => value / leading);

    const rawRoots = [];
    let deflated = monic.slice();
    for (let i = 0; i < degree; i++) {
      const currentDegree = deflated.length - 1;
      if (currentDegree === 1) {
        rawRoots.push(complex(-deflated[1] / deflated[0], 0));
        break;
      }
      if (currentDegree === 2) {
        quadraticRoots(deflated).forEach((r) => rawRoots.push(r));
        break;
      }
      const start = fromPolar(1 + Math.abs(deflated[deflated.length - 1]), 0.73 + i * 1.17);
      let root = laguerreRoot(deflated, start);
      if (Math.abs(root.im) < 1e-10 * Math.max(1, Math.abs(root.re))) {
        root = complex(root.re, 0);
        deflated = syntheticDivide(deflated, root.re);
      } else {
        const re = root.re;
        const im = Math.abs(root.im);
        rawRoots.push(complex(re, im));
        rawRoots.push(complex(re, -im));
        const quadFactor = [1, -2 * re, re * re + im * im];
        const newDeflated = [deflated[0]];
        for (let k = 1; k < deflated.length - 2; k++) {
          newDeflated.push(deflated[k] - quadFactor[1] * newDeflated[k - 1] - (k >= 2 ? quadFactor[2] * newDeflated[k - 2] : 0));
        }
        deflated = newDeflated;
        i++;
      }
      if (root.im === 0) {
        rawRoots.push(root);
      }
    }

    const polished = rawRoots.map((r) => {
      let z = r;
      for (let iter = 0; iter < 20; iter++) {
        const { p, dp } = polyEvalWithDeriv(monic, z);
        if (absComplex(p) < 1e-14) break;
        if (absComplex(dp) < 1e-30) break;
        z = sub(z, div(p, dp));
      }
      return z;
    });

    return polished
      .slice(0, degree)
      .map((r) => cleanComplex(r))
      .sort((a, b) => absComplex(a) - absComplex(b) || a.re - b.re || a.im - b.im);
  }

  function quadraticRoots(coefficients) {
    const [a, b, c] = coefficients;
    const discriminant = b * b - 4 * a * c;
    if (discriminant >= 0) {
      const rootDiscriminant = Math.sqrt(discriminant);
      return [
        complex((-b + rootDiscriminant) / (2 * a), 0),
        complex((-b - rootDiscriminant) / (2 * a), 0),
      ];
    }

    const rootDiscriminant = Math.sqrt(-discriminant);
    return [
      complex(-b / (2 * a), rootDiscriminant / (2 * a)),
      complex(-b / (2 * a), -rootDiscriminant / (2 * a)),
    ];
  }

  function cleanComplex(value, tolerance = 1e-9) {
    const re = Math.abs(value.re) < tolerance ? 0 : value.re;
    const im = Math.abs(value.im) < tolerance ? 0 : value.im;
    return complex(re, im);
  }

  function classifyRoots(roots, tolerance = 1e-7) {
    const cleanedRoots = roots.map((root) => cleanComplex(root));
    const result = {
      originCount: 0,
      realRoots: [],
      complexPairs: [],
    };

    const complexRoots = [];

    cleanedRoots.forEach((root) => {
      if (absComplex(root) < tolerance) {
        result.originCount += 1;
        return;
      }

      if (Math.abs(root.im) < tolerance * Math.max(1, Math.abs(root.re))) {
        result.realRoots.push(root.re);
        return;
      }

      complexRoots.push(root);
    });

    const used = new Array(complexRoots.length).fill(false);
    for (let index = 0; index < complexRoots.length; index += 1) {
      if (used[index]) {
        continue;
      }

      const root = complexRoots[index];
      const conjugate = complex(root.re, -root.im);
      let matchIndex = -1;

      for (let candidateIndex = index + 1; candidateIndex < complexRoots.length; candidateIndex += 1) {
        if (used[candidateIndex]) {
          continue;
        }
        const candidate = complexRoots[candidateIndex];
        if (absComplex(sub(candidate, conjugate)) < tolerance * Math.max(1, absComplex(root))) {
          matchIndex = candidateIndex;
          break;
        }
      }

      if (matchIndex === -1) {
        if (root.im > 0) {
          result.complexPairs.push(root);
        }
        used[index] = true;
        continue;
      }

      used[index] = true;
      used[matchIndex] = true;
      const representative = root.im > 0 ? root : complexRoots[matchIndex];
      result.complexPairs.push(representative.im < 0 ? complex(representative.re, -representative.im) : representative);
    }

    result.realRoots.sort((a, b) => Math.abs(a) - Math.abs(b) || a - b);
    result.complexPairs.sort(
      (a, b) => absComplex(a) - absComplex(b) || a.re - b.re || a.im - b.im
    );
    return result;
  }

  function normalisedGain(numerator, denominator) {
    let numConstant = 0;
    for (let i = numerator.length - 1; i >= 0; i--) {
      if (Math.abs(numerator[i]) > 1e-12) {
        numConstant = numerator[i];
        break;
      }
    }
    let denConstant = 0;
    for (let i = denominator.length - 1; i >= 0; i--) {
      if (Math.abs(denominator[i]) > 1e-12) {
        denConstant = denominator[i];
        break;
      }
    }
    return complex(numConstant / denConstant, 0);
  }

  function firstOrderPhaseApproximation(frequency, cornerFrequency, finalPhaseDeg) {
    const lower = cornerFrequency / 10;
    const upper = cornerFrequency * 10;
    return frequency.map((value) => {
      if (value >= upper) {
        return finalPhaseDeg;
      }
      if (value <= lower) {
        return 0;
      }
      return (finalPhaseDeg * Math.log10(value / lower)) / 2;
    });
  }

  function pairPhaseSign(root) {
    return root.re <= 0 ? 1 : -1;
  }

  function accumulateMarker(markerGroups, frequency, isZero, order) {
    const existing = markerGroups.find(
      (marker) => Math.abs(marker.frequency - frequency) <= Math.max(1e-12, Math.abs(frequency) * 1e-6)
    );

    if (existing) {
      if (isZero) {
        existing.zeroOrder += order;
      } else {
        existing.poleOrder += order;
      }
      return;
    }

    markerGroups.push({
      frequency,
      zeroOrder: isZero ? order : 0,
      poleOrder: isZero ? 0 : order,
    });
  }

  function determineFrequencyRange(zeroInfo, poleInfo, delay) {
    const finiteCorners = [
      ...zeroInfo.realRoots.map(Math.abs),
      ...poleInfo.realRoots.map(Math.abs),
      ...zeroInfo.complexPairs.map(absComplex),
      ...poleInfo.complexPairs.map(absComplex),
    ].filter((value) => value > 0);

    if (!finiteCorners.length) {
      const delayUpper = delay > 0 ? Math.max(1e2, 40 / delay) : 1e2;
      return [1e-2, delayUpper];
    }

    const minimum = Math.min(...finiteCorners);
    const maximum = Math.max(...finiteCorners);
    const lower = 10 ** Math.floor(Math.log10(minimum) - 2);
    const delayAdjustedMax = delay > 0 ? Math.max(maximum, 20 / delay) : maximum;
    const upper = 10 ** Math.ceil(Math.log10(delayAdjustedMax) + 2);
    return [lower, upper];
  }

  function logspace(startExponent, endExponent, samples) {
    const values = [];
    for (let index = 0; index < samples; index += 1) {
      const blend = index / (samples - 1);
      values.push(10 ** (startExponent + blend * (endExponent - startExponent)));
    }
    return values;
  }

  function unwrapDegrees(anglesDeg) {
    if (!anglesDeg.length) {
      return [];
    }
    const unwrapped = [anglesDeg[0]];
    for (let index = 1; index < anglesDeg.length; index += 1) {
      let delta = anglesDeg[index] - anglesDeg[index - 1];
      if (delta > 180) {
        delta -= 360;
      } else if (delta < -180) {
        delta += 360;
      }
      unwrapped.push(unwrapped[index - 1] + delta);
    }
    return unwrapped;
  }

  function alignPhaseTrace(candidate, reference) {
    if (!candidate.length || !reference.length) {
      return candidate.slice();
    }
    const phaseShift = 360 * Math.round((reference[0] - candidate[0]) / 360);
    return candidate.map((value) => value + phaseShift);
  }

  function formatFrequency(value) {
    if (value === 0) {
      return "0";
    }
    if (value >= 1000 || value < 0.01) {
      return value.toExponential(3);
    }
    return Number(value.toPrecision(4)).toString();
  }

  function formatScalarNumber(value) {
    if (value === 0) {
      return "0";
    }
    if (Math.abs(value) >= 1000 || Math.abs(value) < 0.001) {
      return value.toExponential(2);
    }
    return Number(value.toPrecision(4)).toString();
  }

  function isNearlyOne(value, tolerance = 1e-9) {
    return Math.abs(value - 1) < tolerance;
  }

  function isNearlyZero(value, tolerance = 1e-9) {
    return Math.abs(value) < tolerance;
  }

  function trimDecimalZeros(text) {
    return text.includes(".") ? text.replace(/\.?0+$/, "") : text;
  }

  function formatPolynomial(coefficients) {
    if (!coefficients.length) {
      return "0";
    }

    const terms = [];
    const degree = coefficients.length - 1;

    coefficients.forEach((coefficient, index) => {
      if (Math.abs(coefficient) < 1e-12) {
        return;
      }

      const power = degree - index;
      const absValue = Math.abs(coefficient);
      const isFirst = terms.length === 0;
      const sign = coefficient < 0 ? "-" : isFirst ? "" : "+";
      let term = "";

      if (power === 0) {
        term = trimDecimalZeros(absValue.toPrecision(4));
      } else {
        const coeffText = Math.abs(absValue - 1) < 1e-12 ? "" : trimDecimalZeros(absValue.toPrecision(4));
        const variable = power === 1 ? "s" : `s^${power}`;
        term = `${coeffText}${variable}`;
      }

      terms.push(isFirst && sign === "" ? term : `${sign} ${term}`);
    });

    return terms.join(" ") || "0";
  }

  function formatPolynomialHTML(coefficients) {
    if (!coefficients.length) {
      return "0";
    }

    const terms = [];
    const degree = coefficients.length - 1;

    coefficients.forEach((coefficient, index) => {
      if (Math.abs(coefficient) < 1e-12) {
        return;
      }

      const power = degree - index;
      const absValue = Math.abs(coefficient);
      const isFirst = terms.length === 0;
      const sign = coefficient < 0 ? "-" : isFirst ? "" : "+";
      let term = "";

      if (power === 0) {
        term = formatScalarNumber(absValue);
      } else {
        const coeffText = Math.abs(absValue - 1) < 1e-12 ? "" : formatScalarNumber(absValue);
        const variable = power === 1 ? "s" : `s<sup>${power}</sup>`;
        term = `${coeffText}${variable}`;
      }

      terms.push(isFirst && sign === "" ? term : `<span class="mono">${sign}</span> ${term}`);
    });

    return terms.join(" ") || "0";
  }

  function formatFactorLinearText(tau) {
    if (isNearlyOne(tau)) {
      return "s + 1";
    }
    if (isNearlyOne(-tau)) {
      return "-s + 1";
    }
    const omega = 1 / Math.abs(tau);
    const sign = tau < 0 ? "-" : "";
    if (isNearlyOne(omega)) {
      return `${sign}s + 1`;
    }
    return `${sign}<sup>s</sup>/<sub>${formatScalarNumber(omega)}</sub> + 1`;
  }

  function formatSecondOrderFactorText(root) {
    const wn = absComplex(root);
    const coefficientS2 = 1 / (wn * wn);
    const coefficientS1 = (-2 * root.re) / (wn * wn);
    const parts = [];

    if (isNearlyOne(coefficientS2)) {
      parts.push("s<sup>2</sup>");
    } else {
      parts.push(`${formatScalarNumber(coefficientS2)}s<sup>2</sup>`);
    }

    if (!isNearlyZero(coefficientS1)) {
      const sign = coefficientS1 >= 0 ? "+" : "-";
      const absCoeff = Math.abs(coefficientS1);
      const coeffText = isNearlyOne(absCoeff) ? "s" : `${formatScalarNumber(absCoeff)}s`;
      parts.push(`${sign} ${coeffText}`);
    }

    parts.push("+ 1");
    return parts.join(" ");
  }

  function factorChipHtml(text, color) {
    return `<span class="factor-chip" style="--factor-color: ${escapeHtml(color)}">${text}</span>`;
  }

  function productRowHtml(items) {
    if (!items.length) {
      return `<span class="factor-chip factor-neutral">1</span>`;
    }
    return items
      .map((item, index) => `${index ? '<span class="factor-dot">·</span>' : ""}${item}`)
      .join("");
  }

  function formatGainText(gain) {
    if (Math.abs(gain.im) < 1e-8) {
      return formatScalarNumber(gain.re);
    }
    const realText = formatScalarNumber(gain.re);
    const imagText = formatScalarNumber(Math.abs(gain.im));
    return `${realText} ${gain.im >= 0 ? "+" : "-"} ${imagText}j`;
  }

  function stripHtmlTags(text) {
    return String(text).replace(/<[^>]*>/g, "");
  }

  function equationDensityClass(parts, factorCount = 0) {
    const plainLength = parts
      .map((part) => stripHtmlTags(part))
      .join(" ")
      .replace(/\s+/g, " ")
      .trim().length;
    const score = plainLength + factorCount * 10;

    if (score >= 90) {
      return "equation-compact equation-tight";
    }
    if (score >= 58) {
      return "equation-compact";
    }
    return "";
  }

  function computeBodeData(numerator, denominator, delay) {
    const zeros = polynomialRoots(numerator);
    const poles = polynomialRoots(denominator);
    const zeroInfo = classifyRoots(zeros);
    const poleInfo = classifyRoots(poles);
    const [wMin, wMax] = determineFrequencyRange(zeroInfo, poleInfo, delay);
    const frequency = logspace(Math.log10(wMin), Math.log10(wMax), SAMPLE_COUNT);

    const magnitudeDb = [];
    const wrappedRationalPhaseDeg = [];

    frequency.forEach((omega) => {
      const s = complex(0, omega);
      const ratio = div(evaluatePolynomial(numerator, s), evaluatePolynomial(denominator, s));
      magnitudeDb.push(20 * Math.log10(Math.max(absComplex(ratio), Number.MIN_VALUE)));
      wrappedRationalPhaseDeg.push(angleComplex(ratio) * DEG_PER_RAD);
    });

    const phaseDeg = unwrapDegrees(wrappedRationalPhaseDeg).map(
      (deg, i) => deg - frequency[i] * delay * DEG_PER_RAD
    );
    const gain = normalisedGain(numerator, denominator);
    let asymptoticMagnitudeDb = frequency.map(
      () => 20 * Math.log10(Math.max(absComplex(gain), Number.MIN_VALUE))
    );
    let asymptoticPhaseDeg = frequency.map(() => angleComplex(gain) * DEG_PER_RAD);

    const markerGroups = [];
    const factorComponents = [];
    let componentColorIndex = 0;
    const nextComponentColor = () => {
      const color = COMPONENT_COLORS[componentColorIndex % COMPONENT_COLORS.length];
      componentColorIndex += 1;
      return color;
    };

    function addFactorComponent(component) {
      factorComponents.push(component);
      asymptoticMagnitudeDb = asymptoticMagnitudeDb.map(
        (value, index) => value + component.magnitudeDb[index]
      );
      asymptoticPhaseDeg = asymptoticPhaseDeg.map(
        (value, index) => value + component.phaseDeg[index]
      );
    }

    if (delay > 0) {
      addFactorComponent({
        kind: "delay",
        location: "prefix",
        displayText: `e<sup>-${formatScalarNumber(delay)}s</sup>`,
        color: nextComponentColor(),
        magnitudeDb: frequency.map(() => 0),
        phaseDeg: frequency.map((omega) => -omega * delay * DEG_PER_RAD),
        showMagnitude: false,
        showPhase: true,
      });
    }

    if (zeroInfo.originCount) {
      addFactorComponent({
        kind: "origin",
        location: "numerator",
        displayText: `s${zeroInfo.originCount > 1 ? `<sup>${zeroInfo.originCount}</sup>` : ""}`,
        color: nextComponentColor(),
        magnitudeDb: frequency.map((omega) => 20 * zeroInfo.originCount * Math.log10(omega)),
        phaseDeg: frequency.map(() => 90 * zeroInfo.originCount),
        showMagnitude: true,
        showPhase: true,
      });
    }

    if (poleInfo.originCount) {
      addFactorComponent({
        kind: "origin",
        location: "denominator",
        displayText: `s${poleInfo.originCount > 1 ? `<sup>${poleInfo.originCount}</sup>` : ""}`,
        color: nextComponentColor(),
        magnitudeDb: frequency.map((omega) => -20 * poleInfo.originCount * Math.log10(omega)),
        phaseDeg: frequency.map(() => -90 * poleInfo.originCount),
        showMagnitude: true,
        showPhase: true,
      });
    }

    zeroInfo.realRoots.forEach((root) => {
      const corner = Math.abs(root);
      const phaseContribution = firstOrderPhaseApproximation(
        frequency,
        corner,
        (root !== 0 ? -Math.sign(root) : 1) * 90
      );
      const tau = -1 / root;
      addFactorComponent({
        kind: "first-order",
        location: "numerator",
        displayText: formatFactorLinearText(tau),
        color: nextComponentColor(),
        magnitudeDb: frequency.map((omega) => 20 * Math.max(Math.log10(omega / corner), 0)),
        phaseDeg: phaseContribution,
        showMagnitude: true,
        showPhase: true,
      });
      accumulateMarker(markerGroups, corner, true, 1);
    });

    poleInfo.realRoots.forEach((root) => {
      const corner = Math.abs(root);
      const phaseContribution = firstOrderPhaseApproximation(
        frequency,
        corner,
        (root !== 0 ? Math.sign(root) : -1) * 90
      );
      const tau = -1 / root;
      addFactorComponent({
        kind: "first-order",
        location: "denominator",
        displayText: formatFactorLinearText(tau),
        color: nextComponentColor(),
        magnitudeDb: frequency.map((omega) => -20 * Math.max(Math.log10(omega / corner), 0)),
        phaseDeg: phaseContribution,
        showMagnitude: true,
        showPhase: true,
      });
      accumulateMarker(markerGroups, corner, false, 1);
    });

    zeroInfo.complexPairs.forEach((root) => {
      const corner = absComplex(root);
      const phaseContribution = firstOrderPhaseApproximation(
        frequency,
        corner,
        pairPhaseSign(root) * 180
      );
      addFactorComponent({
        kind: "second-order",
        location: "numerator",
        displayText: formatSecondOrderFactorText(root),
        color: nextComponentColor(),
        magnitudeDb: frequency.map((omega) => 40 * Math.max(Math.log10(omega / corner), 0)),
        phaseDeg: phaseContribution,
        showMagnitude: true,
        showPhase: true,
      });
      accumulateMarker(markerGroups, corner, true, 2);
    });

    poleInfo.complexPairs.forEach((root) => {
      const corner = absComplex(root);
      const phaseContribution = firstOrderPhaseApproximation(
        frequency,
        corner,
        -pairPhaseSign(root) * 180
      );
      addFactorComponent({
        kind: "second-order",
        location: "denominator",
        displayText: formatSecondOrderFactorText(root),
        color: nextComponentColor(),
        magnitudeDb: frequency.map((omega) => -40 * Math.max(Math.log10(omega / corner), 0)),
        phaseDeg: phaseContribution,
        showMagnitude: true,
        showPhase: true,
      });
      accumulateMarker(markerGroups, corner, false, 2);
    });

    markerGroups.sort((a, b) => a.frequency - b.frequency);
    asymptoticPhaseDeg = alignPhaseTrace(asymptoticPhaseDeg, phaseDeg);

    return {
      numerator,
      denominator,
      delay,
      gain,
      zeroInfo,
      poleInfo,
      frequency,
      magnitudeDb,
      phaseDeg,
      asymptoticMagnitudeDb,
      asymptoticPhaseDeg,
      factorComponents,
      markerGroups,
    };
  }

  function buildEquationHtml({ subtitle, prefixItems, numeratorHtml, denominatorHtml, densityClass }) {
    const prefixHtml = prefixItems.length
      ? `<div class="equation-prefix">${prefixItems
          .map((item, index) => `${index ? '<span class="factor-dot">·</span>' : ""}${item}`)
          .join("")}</div>`
      : "";

    return `
      <div class="equation-block">
        <div class="formula-subtitle">${subtitle}</div>
        <div class="equation ${densityClass || ""}">
          <span class="equation-symbol">G(s) =</span>
          <div class="equation-main">
            ${prefixHtml}
            <div class="fraction" aria-label="${escapeHtml(subtitle)}">
              <div class="fraction-num">${numeratorHtml}</div>
              <div class="fraction-bar"></div>
              <div class="fraction-den">${denominatorHtml}</div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function buildTransferFunctionHTML(data) {
    const exactPrefix =
      data.delay > 0 ? [`<span class="delay-term">e<sup>-${formatScalarNumber(data.delay)}s</sup></span>`] : [];
    const exactPrefixText = data.delay > 0 ? [`e^(-${formatScalarNumber(data.delay)}s)`] : [];
    const numeratorFactors = data.factorComponents
      .filter((component) => component.location === "numerator")
      .map((component) => factorChipHtml(component.displayText, component.color));
    const numeratorFactorText = data.factorComponents
      .filter((component) => component.location === "numerator")
      .map((component) => component.displayText);
    const denominatorFactors = data.factorComponents
      .filter((component) => component.location === "denominator")
      .map((component) => factorChipHtml(component.displayText, component.color));
    const denominatorFactorText = data.factorComponents
      .filter((component) => component.location === "denominator")
      .map((component) => component.displayText);
    const bodePrefix = [];
    const bodePrefixText = [];

    if (Math.abs(data.gain.re - 1) > 1e-9 || Math.abs(data.gain.im) > 1e-9) {
      const gainText = formatGainText(data.gain);
      bodePrefix.push(
        `<span class="factor-chip gain-chip">${escapeHtml(gainText)}</span>`
      );
      bodePrefixText.push(gainText);
    }

    data.factorComponents
      .filter((component) => component.location === "prefix")
      .forEach((component) => {
        bodePrefix.push(factorChipHtml(component.displayText, component.color));
        bodePrefixText.push(component.displayText);
      });

    const numeratorPolynomialHtml = `<span class="poly-formula">${formatPolynomialHTML(data.numerator)}</span>`;
    const denominatorPolynomialHtml = `<span class="poly-formula">${formatPolynomialHTML(data.denominator)}</span>`;
    const numeratorFactorHtml = `<div class="factor-row">${productRowHtml(numeratorFactors)}</div>`;
    const denominatorFactorHtml = `<div class="factor-row">${productRowHtml(denominatorFactors)}</div>`;

    return `
      <div class="formula-stack">
        ${buildEquationHtml({
          subtitle: "Polynomial form",
          prefixItems: exactPrefix,
          numeratorHtml: numeratorPolynomialHtml,
          denominatorHtml: denominatorPolynomialHtml,
          densityClass: equationDensityClass(
            [...exactPrefixText, formatPolynomial(data.numerator), formatPolynomial(data.denominator)],
            2
          ),
        })}
        ${buildEquationHtml({
          subtitle: "Bode factor form",
          prefixItems: bodePrefix,
          numeratorHtml: numeratorFactorHtml,
          denominatorHtml: denominatorFactorHtml,
          densityClass: equationDensityClass(
            [...bodePrefixText, ...numeratorFactorText, ...denominatorFactorText],
            bodePrefixText.length + numeratorFactorText.length + denominatorFactorText.length
          ),
        })}
      </div>
    `;
  }

  function formatComplexPair(root) {
    const realText = formatScalarNumber(root.re);
    const imagText = formatScalarNumber(Math.abs(root.im));
    return `${realText} ± j${imagText}`;
  }

  function buildRootLocationChips(rootInfo) {
    const chips = [];

    if (rootInfo.originCount) {
      chips.push(`<span class="chip">0${rootInfo.originCount > 1 ? ` ×${rootInfo.originCount}` : ""}</span>`);
    }
    rootInfo.realRoots.forEach((root) => {
      chips.push(`<span class="chip">${escapeHtml(formatScalarNumber(root))}</span>`);
    });
    rootInfo.complexPairs.forEach((root) => {
      chips.push(`<span class="chip">${escapeHtml(formatComplexPair(root))}</span>`);
    });

    return chips.length ? chips.join("") : `<span class="chip">None</span>`;
  }

  function computeDcGain(numerator, denominator) {
    const numeratorAtZero = numerator[numerator.length - 1];
    const denominatorAtZero = denominator[denominator.length - 1];

    if (Math.abs(denominatorAtZero) > 1e-12) {
      return { type: "finite", value: numeratorAtZero / denominatorAtZero };
    }
    if (Math.abs(numeratorAtZero) > 1e-12) {
      return { type: "infinite" };
    }
    return { type: "undefined" };
  }

  function interpolateLogFrequency(x1, x2, ratio) {
    const logX = Math.log10(x1) + ratio * (Math.log10(x2) - Math.log10(x1));
    return 10 ** logX;
  }

  function interpolateSeriesAtRatio(values, index, ratio) {
    return values[index] + ratio * (values[index + 1] - values[index]);
  }

  function findFirstCrossing(frequency, values, target) {
    for (let index = 0; index < values.length - 1; index += 1) {
      const first = values[index] - target;
      const second = values[index + 1] - target;

      if (Math.abs(first) < 1e-12) {
        return { index, ratio: 0, frequency: frequency[index] };
      }
      if (first * second < 0 || Math.abs(second) < 1e-12) {
        const denominator = values[index + 1] - values[index];
        const ratio = Math.abs(denominator) < 1e-12 ? 0 : (target - values[index]) / denominator;
        return {
          index,
          ratio,
          frequency: interpolateLogFrequency(frequency[index], frequency[index + 1], ratio),
        };
      }
    }
    return null;
  }

  function computeMargins(data) {
    const gainCrossing = findFirstCrossing(data.frequency, data.magnitudeDb, 0);
    let phaseMargin = { value: null, frequency: null, meta: "no 0 dB cross" };

    if (gainCrossing) {
      const phaseAtCrossing = interpolateSeriesAtRatio(data.phaseDeg, gainCrossing.index, gainCrossing.ratio);
      phaseMargin = {
        value: 180 + phaseAtCrossing,
        frequency: gainCrossing.frequency,
        meta: `@ ${formatFrequency(gainCrossing.frequency)} rad/s`,
      };
    }

    const phaseCrossing = findFirstCrossing(data.frequency, data.phaseDeg, -180);
    let gainMargin = { value: null, frequency: null, meta: "no -180° cross" };

    if (phaseCrossing) {
      const magnitudeAtCrossing = interpolateSeriesAtRatio(
        data.magnitudeDb,
        phaseCrossing.index,
        phaseCrossing.ratio
      );
      gainMargin = {
        value: -magnitudeAtCrossing,
        frequency: phaseCrossing.frequency,
        meta: `@ ${formatFrequency(phaseCrossing.frequency)} rad/s`,
      };
    } else if (Math.min(...data.phaseDeg) > -180) {
      gainMargin = { value: Infinity, frequency: null, meta: "no -180° cross" };
    }

    return { phaseMargin, gainMargin };
  }

  function formatDcGainCard(dcGain) {
    if (dcGain.type === "finite") {
      const magnitude = Math.abs(dcGain.value);
      const gainDb = magnitude > 0 ? `${(20 * Math.log10(magnitude)).toFixed(2)} dB` : "-∞ dB";
      return {
        value: formatScalarNumber(dcGain.value),
        meta: gainDb,
      };
    }
    if (dcGain.type === "infinite") {
      return { value: "∞", meta: "non-finite at s = 0" };
    }
    return { value: "undefined", meta: "zero over zero at s = 0" };
  }

  function formatMarginCard(margin, unit) {
    if (margin.value === Infinity) {
      return { value: "∞", meta: margin.meta };
    }
    if (margin.value == null || !Number.isFinite(margin.value)) {
      return { value: "N/A", meta: margin.meta };
    }
    return {
      value: `${trimDecimalZeros(margin.value.toFixed(2))} ${unit}`,
      meta: margin.meta,
    };
  }

  function computeSystemType(poleInfo) {
    return poleInfo.originCount;
  }

  function computeErrorConstants(numerator, denominator) {
    const numAtZero = numerator[numerator.length - 1];
    const denAtZero = denominator[denominator.length - 1];

    let numOriginCount = 0;
    for (let i = numerator.length - 1; i >= 0; i--) {
      if (Math.abs(numerator[i]) < 1e-12) numOriginCount++;
      else break;
    }
    let denOriginCount = 0;
    for (let i = denominator.length - 1; i >= 0; i--) {
      if (Math.abs(denominator[i]) < 1e-12) denOriginCount++;
      else break;
    }
    const systemType = denOriginCount - numOriginCount;

    function evalReducedRatio() {
      const numIdx = numerator.length - 1 - numOriginCount;
      const denIdx = denominator.length - 1 - denOriginCount;
      if (numIdx < 0 || denIdx < 0) return 0;
      return numerator[numIdx] / denominator[denIdx];
    }

    const Kbode = Math.abs(evalReducedRatio());

    const Kp = systemType <= 0 ? Kbode : Infinity;
    const Kv = systemType < 1 ? 0 : systemType === 1 ? Kbode : Infinity;
    const Ka = systemType < 2 ? 0 : systemType === 2 ? Kbode : Infinity;

    return { Kp, Kv, Ka };
  }

  function computeStability(poleInfo) {
    const hasRHP = poleInfo.realRoots.some((r) => r > 1e-9) ||
      poleInfo.complexPairs.some((r) => r.re > 1e-9);
    const hasImagAxis = poleInfo.originCount > 0 ||
      poleInfo.complexPairs.some((r) => Math.abs(r.re) < 1e-9);

    if (hasRHP) return { label: "Unstable", css: "stability-unstable" };
    if (hasImagAxis) return { label: "Marginally stable", css: "stability-marginal" };
    return { label: "Stable", css: "stability-stable" };
  }

  function computeBandwidth(frequency, magnitudeDb) {
    const dcDb = magnitudeDb[0];
    const target = dcDb - 3;
    const crossing = findFirstCrossing(frequency, magnitudeDb, target);
    if (!crossing) return null;
    return crossing.frequency;
  }

  function getCornerFrequencies(data) {
    const corners = [];
    const addCorners = (roots, location, kind) => {
      roots.forEach((root) => {
        const omega = typeof root === "number" ? Math.abs(root) : absComplex(root);
        corners.push({ frequency: omega, location, kind });
      });
    };
    addCorners(data.zeroInfo.realRoots, "zero", "1st");
    addCorners(data.zeroInfo.complexPairs, "zero", "2nd");
    addCorners(data.poleInfo.realRoots, "pole", "1st");
    addCorners(data.poleInfo.complexPairs, "pole", "2nd");
    corners.sort((a, b) => a.frequency - b.frequency);
    return corners;
  }

  function getDominantPairInfo(poleInfo) {
    if (!poleInfo.complexPairs.length) return null;
    const dominant = poleInfo.complexPairs[0];
    const wn = absComplex(dominant);
    const zeta = -dominant.re / wn;
    return { wn, zeta };
  }

  function formatErrorConstant(value) {
    if (value === Infinity) return "∞";
    if (value === 0) return "0";
    return formatScalarNumber(value);
  }

  function buildMetricsHTML(data) {
    const dcGain = formatDcGainCard(computeDcGain(data.numerator, data.denominator));
    const margins = computeMargins(data);
    const phaseMargin = formatMarginCard(margins.phaseMargin, "deg");
    const gainMargin = formatMarginCard(margins.gainMargin, "dB");

    const systemType = computeSystemType(data.poleInfo);
    const errorK = computeErrorConstants(data.numerator, data.denominator);
    const stability = computeStability(data.poleInfo);
    const bandwidth = computeBandwidth(data.frequency, data.magnitudeDb);
    const corners = getCornerFrequencies(data);
    const dominant = getDominantPairInfo(data.poleInfo);

    const bandwidthText = bandwidth != null
      ? `${formatScalarNumber(bandwidth)} rad/s`
      : "N/A";

    const cornerChips = corners.length
      ? corners.map((c) => {
          const icon = c.location === "pole" ? "×" : "○";
          const order = c.kind === "2nd" ? "²" : "";
          return `<span class="chip corner-chip"><span class="corner-icon corner-${c.location}">${icon}${order}</span> ${escapeHtml(formatFrequency(c.frequency))}</span>`;
        }).join("")
      : `<span class="chip">None</span>`;

    let dominantHtml = "";
    if (dominant) {
      dominantHtml = `
        <div class="metrics-row">
          <div class="metric-card">
            <div class="metric-label">Natural Freq. ω<sub>n</sub></div>
            <div class="metric-value">${escapeHtml(formatScalarNumber(dominant.wn))} rad/s</div>
          </div>
          <div class="metric-card">
            <div class="metric-label">Damping ζ</div>
            <div class="metric-value">${escapeHtml(formatScalarNumber(dominant.zeta))}</div>
            <div class="metric-meta">${dominant.zeta < 0 ? "unstable" : dominant.zeta < 1 ? "underdamped" : dominant.zeta === 1 ? "critically damped" : "overdamped"}</div>
          </div>
        </div>`;
    }

    return `
      <div class="metrics-top">
        <div class="metric-card">
          <div class="metric-label">DC Gain</div>
          <div class="metric-value">${escapeHtml(dcGain.value)}</div>
          <div class="metric-meta">${escapeHtml(dcGain.meta)}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Phase Margin</div>
          <div class="metric-value">${escapeHtml(phaseMargin.value)}</div>
          <div class="metric-meta">${escapeHtml(phaseMargin.meta)}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Gain Margin</div>
          <div class="metric-value">${escapeHtml(gainMargin.value)}</div>
          <div class="metric-meta">${escapeHtml(gainMargin.meta)}</div>
        </div>
      </div>
      <div class="metrics-row">
        <div class="metric-card">
          <div class="metric-label">Stability</div>
          <div class="metric-value ${stability.css}">${stability.label}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">System Type</div>
          <div class="metric-value">${systemType}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Bandwidth</div>
          <div class="metric-value">${escapeHtml(bandwidthText)}</div>
          <div class="metric-meta">-3 dB frequency</div>
        </div>
      </div>
      <div class="metrics-row">
        <div class="metric-card">
          <div class="metric-label">K<sub>p</sub></div>
          <div class="metric-value">${formatErrorConstant(errorK.Kp)}</div>
          <div class="metric-meta">position</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">K<sub>v</sub></div>
          <div class="metric-value">${formatErrorConstant(errorK.Kv)}</div>
          <div class="metric-meta">velocity</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">K<sub>a</sub></div>
          <div class="metric-value">${formatErrorConstant(errorK.Ka)}</div>
          <div class="metric-meta">acceleration</div>
        </div>
      </div>
      ${dominantHtml}
      <div class="location-block">
        <span class="metric-label">Pole Locations</span>
        <div class="location-list">${buildRootLocationChips(data.poleInfo)}</div>
      </div>
      <div class="location-block">
        <span class="metric-label">Zero Locations</span>
        <div class="location-list">${buildRootLocationChips(data.zeroInfo)}</div>
      </div>
      <div class="location-block">
        <span class="metric-label">Corner Frequencies (rad/s)</span>
        <div class="location-list">${cornerChips}</div>
      </div>
    `;
  }

  function escapeHtml(text) {
    return String(text)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  function clearNode(node) {
    while (node.firstChild) {
      node.removeChild(node.firstChild);
    }
  }

  function createSvgNode(tagName, attributes = {}) {
    const node = document.createElementNS(SVG_NS, tagName);
    Object.entries(attributes).forEach(([key, value]) => {
      node.setAttribute(key, String(value));
    });
    return node;
  }

  function setSvgText(node, text) {
    node.textContent = text;
  }

  function niceStep(span, targetTicks = 6) {
    if (!(span > 0)) {
      return 1;
    }

    const rough = span / targetTicks;
    const magnitude = 10 ** Math.floor(Math.log10(rough));
    const normalized = rough / magnitude;

    if (normalized <= 1) {
      return magnitude;
    }
    if (normalized <= 2) {
      return 2 * magnitude;
    }
    if (normalized <= 5) {
      return 5 * magnitude;
    }
    return 10 * magnitude;
  }

  function linearTicks(minimum, maximum, targetTicks = 6) {
    const step = niceStep(maximum - minimum, targetTicks);
    const start = Math.floor(minimum / step) * step;
    const end = Math.ceil(maximum / step) * step;
    const ticks = [];

    for (let value = start; value <= end + step * 0.5; value += step) {
      ticks.push(Number(value.toFixed(10)));
    }

    return { ticks, step, min: start, max: end };
  }

  function formatLinearTick(value, step) {
    if (value === 0) {
      return "0";
    }
    if (Math.abs(value) >= 1000 || Math.abs(value) < 0.01) {
      return value.toExponential(2);
    }

    const decimals = Math.max(0, Math.min(4, -Math.floor(Math.log10(step)) + 1));
    return trimDecimalZeros(value.toFixed(decimals));
  }

  function formatProbeValue(value, unit) {
    return `${trimDecimalZeros(value.toFixed(2))} ${unit}`;
  }

  function formatProbeFrequency(value) {
    return `${formatFrequency(value)} rad/s`;
  }

  function clip(value, lower, upper) {
    return Math.min(Math.max(value, lower), upper);
  }

  function nearestIndex(sortedValues, target) {
    if (!sortedValues.length) {
      return 0;
    }

    let low = 0;
    let high = sortedValues.length - 1;

    while (low < high) {
      const middle = Math.floor((low + high) / 2);
      if (sortedValues[middle] < target) {
        low = middle + 1;
      } else {
        high = middle;
      }
    }

    if (low === 0) {
      return 0;
    }

    const previous = low - 1;
    return Math.abs(sortedValues[low] - target) < Math.abs(sortedValues[previous] - target)
      ? low
      : previous;
  }

  function pathFromPoints(points) {
    if (!points.length) {
      return "";
    }
    return points
      .map((point, index) => `${index === 0 ? "M" : "L"} ${point.x.toFixed(2)} ${point.y.toFixed(2)}`)
      .join(" ");
  }

  function decadeTicks(minimum, maximum) {
    const startExponent = Math.floor(Math.log10(minimum));
    const endExponent = Math.ceil(Math.log10(maximum));
    const majors = [];
    const minors = [];

    for (let exponent = startExponent; exponent <= endExponent; exponent += 1) {
      const decadeStart = 10 ** exponent;
      majors.push(decadeStart);
      for (let factor = 2; factor < 10; factor += 1) {
        const tick = factor * decadeStart;
        if (tick >= minimum && tick <= maximum) {
          minors.push(tick);
        }
      }
    }

    return { majors, minors };
  }

  function renderPlaceholder(container, message) {
    if (!container) {
      return;
    }
    container.innerHTML = `<div class="plot-placeholder">${escapeHtml(message)}</div>`;
  }

  function renderReadout(node, entries) {
    if (!node) {
      return;
    }

    node.innerHTML = entries
      .map(
        (entry) =>
          `<div class="readout-pill"><span class="readout-label">${escapeHtml(entry.label)}</span><span>${escapeHtml(
            entry.value
          )}</span></div>`
      )
      .join("");
  }

  function preferredCursorIndex(data, previousFrequency) {
    if (previousFrequency && previousFrequency > 0) {
      return nearestIndex(data.frequency, previousFrequency);
    }
    if (data.markerGroups.length) {
      return nearestIndex(data.frequency, data.markerGroups[0].frequency);
    }
    return Math.floor(data.frequency.length / 2);
  }

  function updateCursorReadouts(state, data, index) {
    const frequency = data.frequency[index];
    renderReadout(state.magnitudeReadout, [
      { label: "Frequency", value: formatProbeFrequency(frequency) },
      { label: "Gain", value: formatProbeValue(data.magnitudeDb[index], "dB") },
    ]);
    renderReadout(state.phaseReadout, [
      { label: "Frequency", value: formatProbeFrequency(frequency) },
      { label: "Phase", value: formatProbeValue(data.phaseDeg[index], "deg") },
    ]);
  }

  function renderSvgPlot(containerId, options) {
    const container = document.getElementById(containerId);
    if (!container) {
      return null;
    }
    if (typeof document.createElementNS !== "function") {
      renderPlaceholder(container, "SVG rendering is not available in this environment.");
      return null;
    }

    clearNode(container);

    const width = Math.max(container.clientWidth || 880, 320);
    const height = Math.max(container.clientHeight || 430, 280);
    const margin = { top: 52, right: 28, bottom: 58, left: 72 };
    const plotWidth = width - margin.left - margin.right;
    const plotHeight = height - margin.top - margin.bottom;
    const xLogMin = Math.log10(options.xMin);
    const xLogMax = Math.log10(options.xMax);

    const svg = createSvgNode("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": options.title,
    });

    const defs = createSvgNode("defs");
    const clipPath = createSvgNode("clipPath", { id: `${containerId}-clip` });
    clipPath.appendChild(
      createSvgNode("rect", {
        x: margin.left,
        y: margin.top,
        width: plotWidth,
        height: plotHeight,
        rx: 12,
      })
    );
    defs.appendChild(clipPath);
    svg.appendChild(defs);

    svg.appendChild(
      createSvgNode("rect", {
        x: margin.left,
        y: margin.top,
        width: plotWidth,
        height: plotHeight,
        rx: 16,
        fill: COLORS.plot,
        stroke: "rgba(30,36,48,0.05)",
      })
    );

    const xToPixel = (value) =>
      margin.left + ((Math.log10(value) - xLogMin) / (xLogMax - xLogMin || 1)) * plotWidth;
    const yToPixel = (value) =>
      margin.top + (1 - (value - options.yMin) / (options.yMax - options.yMin || 1)) * plotHeight;

    const xTicks = decadeTicks(options.xMin, options.xMax);
    xTicks.minors.forEach((tick) => {
      const x = xToPixel(tick);
      svg.appendChild(
        createSvgNode("line", {
          x1: x,
          y1: margin.top,
          x2: x,
          y2: margin.top + plotHeight,
          stroke: "rgba(148,163,184,0.16)",
          "stroke-width": 1,
        })
      );
    });

    xTicks.majors.forEach((tick) => {
      if (tick < options.xMin || tick > options.xMax) {
        return;
      }
      const x = xToPixel(tick);
      svg.appendChild(
        createSvgNode("line", {
          x1: x,
          y1: margin.top,
          x2: x,
          y2: margin.top + plotHeight,
          stroke: COLORS.grid,
          "stroke-width": 1.2,
        })
      );
    });

    const yTickData = linearTicks(options.yMin, options.yMax);
    yTickData.ticks.forEach((tick) => {
      if (tick < options.yMin - 1e-9 || tick > options.yMax + 1e-9) {
        return;
      }
      const y = yToPixel(tick);
      svg.appendChild(
        createSvgNode("line", {
          x1: margin.left,
          y1: y,
          x2: margin.left + plotWidth,
          y2: y,
          stroke: COLORS.grid,
          "stroke-width": 1,
        })
      );

      const label = createSvgNode("text", {
        x: margin.left - 10,
        y: y + 4,
        "text-anchor": "end",
        "font-size": 12,
        fill: "#334155",
      });
      setSvgText(label, formatLinearTick(tick, yTickData.step));
      svg.appendChild(label);
    });

    options.markerGroups.forEach((marker) => {
      const x = xToPixel(marker.frequency);
      svg.appendChild(
        createSvgNode("line", {
          x1: x,
          y1: margin.top,
          x2: x,
          y2: margin.top + plotHeight,
          stroke: COLORS.guide,
          "stroke-width": 1,
          "stroke-dasharray": "3 5",
        })
      );
    });

    const title = createSvgNode("text", {
      x: margin.left,
      y: 28,
      "font-size": 22,
      "font-family": "'Iowan Old Style', 'Palatino Linotype', serif",
      "font-weight": 700,
      fill: "#1e2430",
    });
    setSvgText(title, options.title);
    svg.appendChild(title);

    const yLabel = createSvgNode("text", {
      x: 20,
      y: margin.top + plotHeight / 2,
      transform: `rotate(-90 20 ${margin.top + plotHeight / 2})`,
      "font-size": 13,
      fill: "#334155",
    });
    setSvgText(yLabel, options.yTitle);
    svg.appendChild(yLabel);

    const xLabel = createSvgNode("text", {
      x: margin.left + plotWidth / 2,
      y: height - 16,
      "text-anchor": "middle",
      "font-size": 13,
      fill: "#334155",
    });
    setSvgText(xLabel, "Frequency (rad/s)");
    svg.appendChild(xLabel);

    xTicks.majors.forEach((tick) => {
      if (tick < options.xMin || tick > options.xMax) {
        return;
      }
      const x = xToPixel(tick);
      const tickLabel = createSvgNode("text", {
        x,
        y: margin.top + plotHeight + 22,
        "text-anchor": "middle",
        "font-size": 12,
        fill: "#334155",
      });
      setSvgText(tickLabel, formatFrequency(tick));
      svg.appendChild(tickLabel);
    });

    const clippedGroup = createSvgNode("g", { "clip-path": `url(#${containerId}-clip)` });

    options.traces.forEach((trace) => {
      const points = trace.x.map((value, index) => ({
        x: xToPixel(value),
        y: clip(yToPixel(trace.y[index]), margin.top - 20, margin.top + plotHeight + 20),
      }));

      clippedGroup.appendChild(
        createSvgNode("path", {
          d: pathFromPoints(points),
          fill: "none",
          stroke: trace.color,
          "stroke-width": trace.width,
          "stroke-dasharray": trace.dashPattern || (trace.dashed ? "8 6" : ""),
          "stroke-opacity": trace.opacity == null ? 1 : trace.opacity,
          "stroke-linejoin": "round",
          "stroke-linecap": "round",
        })
      );
    });

    svg.appendChild(clippedGroup);

    const markerBaseY = margin.top + plotHeight - 16;
    const markerStep = 14;
    options.markerGroups.forEach((marker) => {
      const x = xToPixel(marker.frequency);
      for (let index = 0; index < marker.poleOrder; index += 1) {
        const y = markerBaseY - index * markerStep;
        clippedGroup.appendChild(
          createSvgNode("line", {
            x1: x - 5,
            y1: y - 5,
            x2: x + 5,
            y2: y + 5,
            stroke: COLORS.pole,
            "stroke-width": 2,
          })
        );
        clippedGroup.appendChild(
          createSvgNode("line", {
            x1: x - 5,
            y1: y + 5,
            x2: x + 5,
            y2: y - 5,
            stroke: COLORS.pole,
            "stroke-width": 2,
          })
        );
      }
      for (let index = 0; index < marker.zeroOrder; index += 1) {
        const y = markerBaseY - (marker.poleOrder + index + 1) * markerStep;
        clippedGroup.appendChild(
          createSvgNode("circle", {
            cx: x,
            cy: y,
            r: 5,
            fill: "rgba(255,255,255,0.92)",
            stroke: COLORS.zero,
            "stroke-width": 2,
          })
        );
      }
    });

    const legend = createSvgNode("g");
    const legendW = 190;
    const legendH = 44;
    const legendX = width - margin.right - legendW + 4;
    const legendY = margin.top + 14;
    svg.appendChild(legend);

    const legendBox = createSvgNode("rect", {
      x: legendX - 4,
      y: legendY - 14,
      width: legendW,
      height: legendH,
      rx: 8,
      fill: "rgba(255,255,255,0.55)",
      stroke: "rgba(30,36,48,0.06)",
    });
    legend.appendChild(legendBox);

    [
      { label: options.legend.actual, color: options.legend.actualColor, dashed: false, y: legendY },
      { label: options.legend.asymptotic, color: options.legend.asymColor, dashed: true, y: legendY + 18 },
    ].forEach((entry) => {
      legend.appendChild(
        createSvgNode("line", {
          x1: legendX,
          y1: entry.y,
          x2: legendX + 26,
          y2: entry.y,
          stroke: entry.color,
          "stroke-width": 2,
          "stroke-dasharray": entry.dashed ? "6 5" : "",
          "stroke-linecap": "round",
        })
      );
      const label = createSvgNode("text", {
        x: legendX + 34,
        y: entry.y + 3.5,
        "font-size": 10.5,
        fill: "#64748b",
      });
      setSvgText(label, entry.label);
      legend.appendChild(label);
    });

    const cursorLine = createSvgNode("line", {
      x1: margin.left,
      y1: margin.top,
      x2: margin.left,
      y2: margin.top + plotHeight,
      stroke: "rgba(30,36,48,0.55)",
      "stroke-width": 1.35,
      "stroke-dasharray": "5 5",
    });
    svg.appendChild(cursorLine);

    const cursorHalo = createSvgNode("circle", {
      cx: margin.left,
      cy: margin.top + plotHeight / 2,
      r: 8.5,
      fill: "rgba(255,255,255,0.82)",
      stroke: options.cursorColor,
      "stroke-width": 1.5,
    });
    clippedGroup.appendChild(cursorHalo);

    const cursorPoint = createSvgNode("circle", {
      cx: margin.left,
      cy: margin.top + plotHeight / 2,
      r: 4.6,
      fill: options.cursorColor,
      stroke: "rgba(255,255,255,0.96)",
      "stroke-width": 1.6,
    });
    clippedGroup.appendChild(cursorPoint);

    const hitBox = createSvgNode("rect", {
      x: margin.left,
      y: margin.top,
      width: plotWidth,
      height: plotHeight,
      fill: "transparent",
      "pointer-events": "all",
      style: "cursor: crosshair;",
    });
    svg.appendChild(hitBox);

    container.appendChild(svg);

    const updateCursorIndex = (index) => {
      const safeIndex = clip(index, 0, options.cursorTrace.x.length - 1);
      const x = xToPixel(options.cursorTrace.x[safeIndex]);
      const y = clip(yToPixel(options.cursorTrace.y[safeIndex]), margin.top - 12, margin.top + plotHeight + 12);

      cursorLine.setAttribute("x1", x);
      cursorLine.setAttribute("x2", x);
      cursorHalo.setAttribute("cx", x);
      cursorHalo.setAttribute("cy", y);
      cursorPoint.setAttribute("cx", x);
      cursorPoint.setAttribute("cy", y);
    };

    const pickIndexFromClientX = (clientX) => {
      if (typeof svg.getBoundingClientRect !== "function") {
        return 0;
      }

      const bounds = svg.getBoundingClientRect();
      const svgX = ((clientX - bounds.left) / (bounds.width || 1)) * width;
      const clampedX = clip(svgX, margin.left, margin.left + plotWidth);
      const blend = (clampedX - margin.left) / (plotWidth || 1);
      const targetFrequency = 10 ** (xLogMin + blend * (xLogMax - xLogMin));
      return nearestIndex(options.cursorTrace.x, targetFrequency);
    };

    const bindCursor = (onMove) => {
      if (typeof hitBox.addEventListener !== "function") {
        return;
      }
      const handlePointer = (event) => {
        onMove(pickIndexFromClientX(event.clientX));
      };
      hitBox.addEventListener("pointermove", handlePointer);
      hitBox.addEventListener("pointerdown", handlePointer);
    };

    return {
      bindCursor,
      setCursorIndex: updateCursorIndex,
    };
  }

  function renderPlots(data) {
    const flattenSeries = (seriesList) => seriesList.reduce((all, values) => all.concat(values), []);
    const magnitudeComponentTraces = data.factorComponents
      .filter((component) => component.showMagnitude)
      .map((component) => ({
        x: data.frequency,
        y: component.magnitudeDb,
        color: component.color,
        width: 1.6,
        dashPattern: "5 4",
        opacity: 0.55,
      }));
    const phaseComponentTraces = data.factorComponents
      .filter((component) => component.showPhase)
      .map((component) => ({
        x: data.frequency,
        y: component.phaseDeg,
        color: component.color,
        width: 1.6,
        dashPattern: "5 4",
        opacity: 0.55,
      }));

    const magnitudeSeries = [data.magnitudeDb, data.asymptoticMagnitudeDb, ...magnitudeComponentTraces.map((trace) => trace.y)];
    const phaseSeries = [data.phaseDeg, data.asymptoticPhaseDeg, ...phaseComponentTraces.map((trace) => trace.y)];
    const flatMagnitudeSeries = flattenSeries(magnitudeSeries);
    const flatPhaseSeries = flattenSeries(phaseSeries);
    const magnitudeMin = Math.min(...flatMagnitudeSeries);
    const magnitudeMax = Math.max(...flatMagnitudeSeries);
    const magnitudePadding = Math.max((magnitudeMax - magnitudeMin) * 0.08, 1);

    const phaseMin = Math.min(...flatPhaseSeries);
    const phaseMax = Math.max(...flatPhaseSeries);
    const phasePadding = Math.max((phaseMax - phaseMin) * 0.08, 10);

    const magnitudeRenderer = renderSvgPlot("magnitude-plot", {
      title: "Gain Plot",
      yTitle: "Gain (dB)",
      xMin: data.frequency[0],
      xMax: data.frequency[data.frequency.length - 1],
      yMin: magnitudeMin - magnitudePadding,
      yMax: magnitudeMax + magnitudePadding,
      markerGroups: data.markerGroups,
      traces: [
        ...magnitudeComponentTraces,
        {
          x: data.frequency,
          y: data.magnitudeDb,
          color: COLORS.magnitude,
          width: 3,
          dashed: false,
        },
        {
          x: data.frequency,
          y: data.asymptoticMagnitudeDb,
          color: COLORS.asymMagnitude,
          width: 3,
          dashed: true,
        },
      ],
      legend: {
        actual: "Actual magnitude",
        asymptotic: "Asymptotic magnitude",
        actualColor: COLORS.magnitude,
        asymColor: COLORS.asymMagnitude,
      },
      cursorTrace: {
        x: data.frequency,
        y: data.magnitudeDb,
      },
      cursorColor: COLORS.magnitude,
    });

    const phaseRenderer = renderSvgPlot("phase-plot", {
      title: "Phase Plot",
      yTitle: "Phase (deg)",
      xMin: data.frequency[0],
      xMax: data.frequency[data.frequency.length - 1],
      yMin: phaseMin - phasePadding,
      yMax: phaseMax + phasePadding,
      markerGroups: data.markerGroups,
      traces: [
        ...phaseComponentTraces,
        {
          x: data.frequency,
          y: data.phaseDeg,
          color: COLORS.phase,
          width: 3,
          dashed: false,
        },
        {
          x: data.frequency,
          y: data.asymptoticPhaseDeg,
          color: COLORS.asymPhase,
          width: 3,
          dashed: true,
        },
      ],
      legend: {
        actual: "Actual phase",
        asymptotic: "Asymptotic phase",
        actualColor: COLORS.phase,
        asymColor: COLORS.asymPhase,
      },
      cursorTrace: {
        x: data.frequency,
        y: data.phaseDeg,
      },
      cursorColor: COLORS.phase,
    });

    return {
      magnitude: magnitudeRenderer,
      phase: phaseRenderer,
    };
  }

  function buildModelFromInputs(state) {
    const numerator = parseCoefficients(state.numInput.value);
    const denominator = parseCoefficients(state.denInput.value);
    const delay = parseDelay(state.delayInput.value);
    return { numerator, denominator, delay };
  }

  function setStatus(state, message, isError) {
    state.messageBox.textContent = message;
    state.messageBox.classList.toggle("error", Boolean(isError));
    state.statusBadge.textContent = isError ? "Needs valid input" : "Live";
    state.statusBadge.className = `status-badge ${isError ? "status-error" : "status-valid"}`;
  }

  function updateAll(state) {
    try {
      const model = buildModelFromInputs(state);
      const data = computeBodeData(model.numerator, model.denominator, model.delay);
      state.tfDisplay.innerHTML = buildTransferFunctionHTML(data);
      state.metricsContent.innerHTML = buildMetricsHTML(data);
      const plots = renderPlots(data);
      const setSharedCursorIndex = (index) => {
        const safeIndex = clip(index, 0, data.frequency.length - 1);
        state.cursorFrequency = data.frequency[safeIndex];
        if (plots.magnitude) {
          plots.magnitude.setCursorIndex(safeIndex);
        }
        if (plots.phase) {
          plots.phase.setCursorIndex(safeIndex);
        }
        updateCursorReadouts(state, data, safeIndex);
      };

      if (plots.magnitude) {
        plots.magnitude.bindCursor(setSharedCursorIndex);
      }
      if (plots.phase) {
        plots.phase.bindCursor(setSharedCursorIndex);
      }

      setSharedCursorIndex(preferredCursorIndex(data, state.cursorFrequency));
      setStatus(state, "Plots updated automatically from the current transfer function.", false);
    } catch (error) {
      state.tfDisplay.innerHTML = `<span class="mono">Waiting for valid coefficients…</span>`;
      if (state.metricsContent) {
        state.metricsContent.innerHTML = "";
      }
      renderPlaceholder(state.magnitudePlot, "Magnitude plot will appear when the transfer function is valid.");
      renderPlaceholder(state.phasePlot, "Phase plot will appear when the transfer function is valid.");
      renderReadout(state.magnitudeReadout, [
        { label: "Frequency", value: "-" },
        { label: "Gain", value: "-" },
      ]);
      renderReadout(state.phaseReadout, [
        { label: "Frequency", value: "-" },
        { label: "Phase", value: "-" },
      ]);
      setStatus(state, error.message, true);
    }
  }

  function debounce(callback, waitMs) {
    let timeoutId = null;
    return function debounced() {
      window.clearTimeout(timeoutId);
      timeoutId = window.setTimeout(() => callback(), waitMs);
    };
  }

  function applyExample(state, numerator, denominator, delay) {
    state.numInput.value = numerator;
    state.denInput.value = denominator;
    state.delayInput.value = delay;
    updateAll(state);
  }

  function init() {
    const state = {
      numInput: document.getElementById("num-input"),
      denInput: document.getElementById("den-input"),
      delayInput: document.getElementById("delay-input"),
      tfDisplay: document.getElementById("tf-display"),
      statusBadge: document.getElementById("status-badge"),
      messageBox: document.getElementById("message-box"),
      metricsContent: document.getElementById("metrics-content"),
      magnitudePlot: document.getElementById("magnitude-plot"),
      phasePlot: document.getElementById("phase-plot"),
      magnitudeReadout: document.getElementById("magnitude-readout"),
      phaseReadout: document.getElementById("phase-readout"),
      exampleButton: document.getElementById("example-btn"),
      clearButton: document.getElementById("clear-btn"),
      cursorFrequency: null,
    };

    const liveUpdate = debounce(() => updateAll(state), 120);
    [state.numInput, state.denInput, state.delayInput].forEach((element) => {
      element.addEventListener("input", liveUpdate);
    });

    state.exampleButton.addEventListener("click", () => applyExample(state, "1 4 5", "2 3 6", "0"));
    state.clearButton.addEventListener("click", () => applyExample(state, "1", "1", "0"));

    window.addEventListener("resize", debounce(() => updateAll(state), 100));
    updateAll(state);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
