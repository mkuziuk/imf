"use strict";

/* ---------- palette (validated, see dataviz notes in the PR) ---------- */

const C = {
  ink: "#16181c",
  secondary: "#52514e",
  muted: "#898781",
  faint: "#c9c7c0",
  grid: "#e1e0d9",
  line: "#dcdad2",
  linear: "#2a78d6",
  gd: "#eb6834",
  contam: "#e34948",
  // blue ordinal ramp, light -> dark, for stage-indexed series
  ramp: ["#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281"],
};

const FONT = { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', size: 12, color: C.secondary };
const PLOT_CONFIG = { displayModeBar: false, responsive: true };

function baseLayout(extra) {
  return Object.assign(
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: FONT,
      margin: { l: 54, r: 16, t: 12, b: 38 },
      hoverlabel: { bgcolor: "#fcfcfb", bordercolor: C.line, font: { ...FONT, color: C.ink } },
      showlegend: true,
      legend: { orientation: "h", x: 0, y: 1.08, yanchor: "bottom", font: { size: 11.5 } },
    },
    extra
  );
}

function axisDefaults(extra) {
  return Object.assign(
    {
      gridcolor: C.grid,
      zerolinecolor: C.grid,
      linecolor: C.line,
      tickfont: { size: 11, color: C.muted },
    },
    extra
  );
}

/* ---------- window schedule (mirror of imf_core.make_window_schedule) ---------- */

function oddCeiling(v) {
  let s = Math.ceil(v);
  if (s % 2 === 0) s += 1;
  return Math.max(1, s);
}

function nearestOdd(v) {
  const r = Math.round(v);
  if (r % 2 === 1) return Math.max(1, r);
  const lower = Math.max(1, r - 1);
  const upper = r + 1;
  return Math.abs(v - lower) <= Math.abs(upper - v) ? lower : upper;
}

function makeSchedule(n, factor, minWindow) {
  let first = oddCeiling(n / 2);
  if (first > n) first = n % 2 === 1 ? n : n - 1;
  const minSize = nearestOdd(minWindow);
  if (minSize > first) return [first];
  const sizes = [first];
  let current = first;
  while (current > minSize) {
    let cand = Math.min(nearestOdd(current / factor), current - 2);
    if (cand % 2 === 0) cand -= 1;
    if (cand < minSize) cand = minSize;
    sizes.push(cand);
    current = cand;
  }
  return sizes;
}

/* ---------- controls & state ---------- */

const $ = (id) => document.getElementById(id);
const controls = [
  "signal", "n", "sigma", "p", "cscale", "model", "contrast", "h_ratio", "kernel", "factor", "min_window",
  "slow_amp", "fast_amp", "fast_cycles", "bump_amp", "bump_pos", "bump_width", "trend_slope",
  "signal_variant", "signal_scale",
];
let seed = 777;
let stageMode = "components";
let errorMode = "raw";
let lastPayload = null;

function signalParams() {
  if ($("signal").value === "simple") {
    return {
      slow_amp: Number($("slow_amp").value),
      fast_amp: Number($("fast_amp").value),
      fast_cycles: Number($("fast_cycles").value),
      bump_amp: Number($("bump_amp").value),
      bump_pos: Number($("bump_pos").value),
      bump_width: Number($("bump_width").value),
      trend_slope: Number($("trend_slope").value),
    };
  }
  return {
    signal_seed: Number($("signal_variant").value),
    signal_scale: Number($("signal_scale").value),
  };
}

function currentParams() {
  return {
    signal: $("signal").value,
    signal_params: signalParams(),
    n: Number($("n").value),
    sigma: Number($("sigma").value),
    p: Number($("p").value),
    scale: Number($("cscale").value),
    model: $("model").value,
    contrast: $("contrast").value,
    h_ratio: Number($("h_ratio").value),
    kernel: $("kernel").value,
    factor: Number($("factor").value),
    min_window: Number($("min_window").value),
    seed,
  };
}

function refreshControlChrome() {
  const simple = $("signal").value === "simple";
  $("simple-params").hidden = !simple;
  $("complex-params").hidden = simple;
  $("slow_amp-val").textContent = Number($("slow_amp").value).toFixed(2);
  $("fast_amp-val").textContent = Number($("fast_amp").value).toFixed(2);
  $("fast_cycles-val").textContent = $("fast_cycles").value;
  $("bump_amp-val").textContent = Number($("bump_amp").value).toFixed(2);
  $("bump_pos-val").textContent = Number($("bump_pos").value).toFixed(2);
  $("bump_width-val").textContent = Number($("bump_width").value).toFixed(3);
  $("trend_slope-val").textContent = Number($("trend_slope").value).toFixed(2);
  $("signal_scale-val").textContent = Number($("signal_scale").value).toFixed(2);
  $("sigma-val").textContent = Number($("sigma").value).toFixed(2);
  $("p-val").textContent = Number($("p").value).toFixed(2);
  $("cscale-val").textContent = Number($("cscale").value).toFixed(2);
  $("h_ratio-val").textContent = Number($("h_ratio").value).toFixed(2);
  $("field-h").classList.toggle("disabled", $("contrast").value === "quadratic");
  const p = currentParams();
  const schedule = makeSchedule(p.n, p.factor, p.min_window);
  $("schedule-badges").innerHTML = schedule.map((w) => `<span>${w}</span>`).join("");
  if (p.contrast === "quadratic") {
    $("gd-step").textContent = "η = 0.95";
  } else {
    const h = Math.max(p.h_ratio * p.sigma, 1e-3);
    $("gd-step").textContent = `η = 0.95·H·√(π∕2) = ${(0.95 * h * Math.sqrt(Math.PI / 2)).toFixed(3)}`;
  }
}

/* ---------- worker ---------- */

const worker = new Worker("imf_worker.js");
let workerReady = false;
let inFlight = false;
let queuedRun = false;
let runCounter = 0;
let runStarted = 0;

function setStatus(state, text) {
  $("status-dot").className = `status-dot ${state}`;
  $("status-text").textContent = text;
}

function setStale(on) {
  document.querySelectorAll(".plot-wrap").forEach((el) => el.classList.toggle("stale", on));
}

function launchRun() {
  if (!workerReady) return;
  if (inFlight) {
    queuedRun = true;
    return;
  }
  inFlight = true;
  runCounter += 1;
  runStarted = performance.now();
  setStatus("busy", "Computing decomposition…");
  setStale(true);
  worker.postMessage({ type: "run", runId: runCounter, params: currentParams() });
}

worker.onmessage = (event) => {
  const msg = event.data;
  if (msg.type === "boot") {
    setStatus("busy", msg.stage);
  } else if (msg.type === "boot-error") {
    setStatus("error", `Failed to start Python runtime: ${msg.message}`);
  } else if (msg.type === "ready") {
    workerReady = true;
    setStatus("busy", "Runtime ready — first run…");
    launchRun();
  } else if (msg.type === "result") {
    inFlight = false;
    if (queuedRun) {
      queuedRun = false;
      launchRun();
      return; // skip rendering the outdated result
    }
    lastPayload = msg.payload;
    const total = ((performance.now() - runStarted) / 1000).toFixed(2);
    setStatus("ready", `Ready — NumPy time ${msg.payload.elapsed_seconds.toFixed(2)} s (round trip ${total} s)`);
    setStale(false);
    renderAll(msg.payload);
  } else if (msg.type === "error") {
    inFlight = false;
    queuedRun = false;
    setStale(false);
    setStatus("error", `Computation failed: ${msg.message}`);
  }
};

let debounceTimer = null;
function scheduleRun() {
  refreshControlChrome();
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(launchRun, 320);
}

controls.forEach((id) => {
  $(id).addEventListener("input", scheduleRun);
  $(id).addEventListener("change", scheduleRun);
});
$("redraw").addEventListener("click", () => {
  seed += 1;
  scheduleRun();
});

document.querySelectorAll("#stage-mode button").forEach((btn) =>
  btn.addEventListener("click", () => {
    stageMode = btn.dataset.mode;
    document.querySelectorAll("#stage-mode button").forEach((b) => b.classList.toggle("active", b === btn));
    if (lastPayload) renderStages(lastPayload);
  })
);
document.querySelectorAll("#error-mode button").forEach((btn) =>
  btn.addEventListener("click", () => {
    errorMode = btn.dataset.mode;
    document.querySelectorAll("#error-mode button").forEach((b) => b.classList.toggle("active", b === btn));
    if (lastPayload) renderError(lastPayload);
  })
);

/* ---------- rendering ---------- */

function fmtSci(v) {
  if (v === 0) return "0";
  const exp = Math.floor(Math.log10(Math.abs(v)));
  if (exp >= -3 && exp < 4) return v.toPrecision(3);
  return `${(v / 10 ** exp).toFixed(1)}·10${superscript(exp)}`;
}

function superscript(num) {
  const map = { "-": "⁻", 0: "⁰", 1: "¹", 2: "²", 3: "³", 4: "⁴", 5: "⁵", 6: "⁶", 7: "⁷", 8: "⁸", 9: "⁹" };
  return String(num).split("").map((ch) => map[ch] ?? ch).join("");
}

function sub(k) {
  const map = { 0: "₀", 1: "₁", 2: "₂", 3: "₃", 4: "₄", 5: "₅", 6: "₆", 7: "₇", 8: "₈", 9: "₉" };
  return String(k).split("").map((ch) => map[ch] ?? ch).join("");
}

function gdLabel(payload) {
  return payload.params_contrast === "quadratic" ? "GD · quadratic" : "GD · robust";
}

function renderAll(payload) {
  payload.params_contrast = $("contrast").value;
  $("obs-meta").textContent =
    `realized p̂ = ${(payload.metrics.contam_fraction * 100).toFixed(1)}% · seed ${seed}`;
  renderObservation(payload);
  renderStages(payload);
  renderRecon(payload);
  renderGd(payload);
  renderError(payload);
}

function renderObservation(p) {
  const maskT = p.mask_idx.map((i) => p.t[i]);
  const maskY = p.mask_idx.map((i) => p.observed[i]);
  const traces = [
    {
      x: p.t, y: p.observed, name: "observed Y", mode: "lines",
      line: { color: C.muted, width: 1 }, hovertemplate: "Y = %{y:.3f}<extra>observed</extra>",
    },
    {
      x: p.t, y: p.clean, name: "clean signal X", mode: "lines",
      line: { color: C.ink, width: 1.7 }, hovertemplate: "X = %{y:.3f}<extra>clean</extra>",
    },
    {
      x: maskT, y: maskY, name: "contaminated points", mode: "markers",
      marker: { color: C.contam, size: 5.5, opacity: 0.85 },
      hovertemplate: "Y = %{y:.3f}<extra>contaminated</extra>",
    },
  ];
  Plotly.react(
    "plot-observation",
    traces,
    baseLayout({
      height: 320,
      hovermode: "x unified",
      xaxis: axisDefaults({ title: { text: "t", font: { size: 11 } } }),
      yaxis: axisDefaults({}),
    }),
    PLOT_CONFIG
  );
}

function renderStages(p) {
  const K = p.windows.length;
  const rows = K + 1; // final row: residual after the last stage
  const sumRows = (mat, upto) => p.t.map((_, i) => {
    let s = 0;
    for (let k = 0; k < upto; k++) s += mat[k][i];
    return s;
  });

  const cleanRefResidual = (refComponents) =>
    p.t.map((_, i) => p.clean[i] - refComponents.reduce((acc, comp) => acc + comp[i], 0));

  const columns = [
    { key: "linear", ref: p.linear_ref, est: p.linear, color: C.linear, label: "linear" },
    { key: "gd", ref: p.gd_ref, est: p.gd, color: C.gd, label: gdLabel(p) },
  ];

  const traces = [];
  const plotHeight = 92 * rows + 104;
  const layout = baseLayout({
    height: plotHeight,
    grid: { rows, columns: 2, pattern: "independent" },
    margin: { l: 64, r: 16, t: 80, b: 34 },
    legend: { orientation: "h", x: 0, y: 1 + 56 / plotHeight, yanchor: "bottom", font: { size: 11.5 } },
    annotations: [
      { text: "Linear (closed form)", x: 0.22, y: 1.0, xref: "paper", yref: "paper", yshift: 22, showarrow: false, font: { size: 12, color: C.ink } },
      { text: gdLabel(p) === "GD · quadratic" ? "Gradient descent (quadratic ρ)" : "Gradient descent (robust ρ_H)", x: 0.78, y: 1.0, xref: "paper", yref: "paper", yshift: 22, showarrow: false, font: { size: 12, color: C.ink } },
    ],
  });

  const rowSeries = (r, col) =>
    r === K
      ? [col.est.residual, cleanRefResidual(col.ref.components)]
      : stageMode === "components"
        ? [col.est.components[r], col.ref.components[r]]
        : [sumRows(col.est.components, r).map((s, i) => p.observed[i] - s), col.est.components[r]];

  // one shared y-scale across all component rows; the residual row keeps its own
  const rangeOf = (rowIndices) => {
    let lo = Infinity, hi = -Infinity;
    for (const r of rowIndices)
      for (const col of columns)
        for (const arr of rowSeries(r, col))
          for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
    return [lo, hi];
  };
  const stageRange = rangeOf(Array.from({ length: K }, (_, i) => i));
  const residualRange = rangeOf([K]);

  for (let r = 0; r < rows; r++) {
    const isResidualRow = r === K;
    const [lo, hi] = isResidualRow ? residualRange : stageRange;
    const pad = (hi - lo) * 0.08 || 0.1;

    for (let c = 0; c < 2; c++) {
      const col = columns[c];
      const idx = r * 2 + c + 1;
      const xa = idx === 1 ? "x" : `x${idx}`;
      const ya = idx === 1 ? "y" : `y${idx}`;
      const axSuffix = idx === 1 ? "" : idx;

      layout[`xaxis${axSuffix}`] = axisDefaults({
        showticklabels: r === rows - 1,
        matches: idx === 1 ? undefined : "x",
      });
      layout[`yaxis${axSuffix}`] = axisDefaults({
        range: [lo - pad, hi + pad],
        title: c === 0
          ? { text: isResidualRow ? `R${sub(K + 1)}` : `S${sub(r + 1)} · ${p.windows[r]}`, font: { size: 10.5, color: C.muted } }
          : undefined,
        showticklabels: c === 0,
      });

      if (isResidualRow) {
        traces.push({
          x: p.t, y: cleanRefResidual(col.ref.components), xaxis: xa, yaxis: ya,
          mode: "lines", line: { color: C.muted, width: 1, dash: "dot" },
          name: "reference (clean signal)", legendgroup: "ref", showlegend: r === 0 && c === 0,
          hovertemplate: "%{y:.3f}<extra>reference residual</extra>",
        });
        traces.push({
          x: p.t, y: col.est.residual, xaxis: xa, yaxis: ya,
          mode: "lines", line: { color: col.color, width: 1.2 },
          name: `${col.label} estimate`, legendgroup: col.key, showlegend: false,
          hovertemplate: "%{y:.3f}<extra>" + col.label + " residual</extra>",
        });
      } else if (stageMode === "components") {
        traces.push({
          x: p.t, y: col.ref.components[r], xaxis: xa, yaxis: ya,
          mode: "lines", line: { color: C.muted, width: 1, dash: "dot" },
          name: "reference (clean signal)", legendgroup: "ref", showlegend: r === 0 && c === 0,
          hovertemplate: "%{y:.3f}<extra>reference S" + (r + 1) + "</extra>",
        });
        traces.push({
          x: p.t, y: col.est.components[r], xaxis: xa, yaxis: ya,
          mode: "lines", line: { color: col.color, width: 1.4 },
          name: `${col.label} estimate`, legendgroup: col.key, showlegend: r === 0,
          hovertemplate: "%{y:.3f}<extra>" + col.label + " S" + (r + 1) + "</extra>",
        });
      } else {
        const before = sumRows(col.est.components, r).map((s, i) => p.observed[i] - s);
        traces.push({
          x: p.t, y: before, xaxis: xa, yaxis: ya,
          mode: "lines", line: { color: C.faint, width: 1 },
          name: "residual before stage", legendgroup: "before", showlegend: r === 0 && c === 0,
          hovertemplate: "%{y:.3f}<extra>before S" + (r + 1) + "</extra>",
        });
        traces.push({
          x: p.t, y: col.est.components[r], xaxis: xa, yaxis: ya,
          mode: "lines", line: { color: col.color, width: 1.4 },
          name: `${col.label} component`, legendgroup: col.key, showlegend: r === 0,
          hovertemplate: "%{y:.3f}<extra>" + col.label + " S" + (r + 1) + "</extra>",
        });
      }
    }
  }

  Plotly.react("plot-stages", traces, layout, PLOT_CONFIG);
}

function renderRecon(p) {
  const m = p.metrics;
  const improvement = (name, value) =>
    m.rmse_observed > 0 && value > 0 ? `${(m.rmse_observed / value).toFixed(1)}× closer than Y` : "";
  $("recon-tiles").innerHTML = `
    <div class="tile">
      <div class="tile-label">RMSE, observed</div>
      <div class="tile-value">${m.rmse_observed.toFixed(3)}</div>
      <div class="tile-sub">‖Y − X‖ₙ</div>
    </div>
    <div class="tile linear">
      <div class="tile-label">RMSE, linear</div>
      <div class="tile-value">${m.rmse_linear.toFixed(3)}</div>
      <div class="tile-sub">${improvement("linear", m.rmse_linear)}</div>
    </div>
    <div class="tile gd">
      <div class="tile-label">RMSE, ${gdLabel(p)}</div>
      <div class="tile-value">${m.rmse_gd.toFixed(3)}</div>
      <div class="tile-sub">${improvement("gd", m.rmse_gd)}</div>
    </div>
    <div class="tile">
      <div class="tile-label">Reconstruction, linear</div>
      <div class="tile-value">${fmtSci(m.recon_linear)}</div>
      <div class="tile-sub">max |Y − ΣS − R|</div>
    </div>
    <div class="tile">
      <div class="tile-label">Reconstruction, GD</div>
      <div class="tile-value">${fmtSci(m.recon_gd)}</div>
      <div class="tile-sub">max |Y − ΣS − R|</div>
    </div>`;

  const traces = [
    { x: p.t, y: p.clean, name: "clean signal X", mode: "lines", line: { color: C.ink, width: 1.7 } },
    { x: p.t, y: p.linear.denoised, name: "linear ΣS", mode: "lines", line: { color: C.linear, width: 1.3 } },
    { x: p.t, y: p.gd.denoised, name: `${gdLabel(p)} ΣS`, mode: "lines", line: { color: C.gd, width: 1.3 } },
  ];
  Plotly.react(
    "plot-recon",
    traces,
    baseLayout({
      height: 320,
      hovermode: "x unified",
      xaxis: axisDefaults({ title: { text: "t", font: { size: 11 } } }),
      yaxis: axisDefaults({}),
    }),
    PLOT_CONFIG
  );
}

function renderGd(p) {
  const K = p.windows.length;
  $("gd-meta").textContent = `iterations per stage: ${p.gd.iters.join(", ")}`;
  const traces = p.gd.traces.map((trace, k) => {
    const rampIdx = K === 1 ? C.ramp.length - 1 : Math.round((k * (C.ramp.length - 1)) / (K - 1));
    return {
      x: trace.map((_, i) => i + 1),
      y: trace,
      name: `stage ${k + 1} · w = ${p.windows[k]}`,
      mode: "lines+markers",
      line: { color: C.ramp[rampIdx], width: 1.6 },
      marker: { size: 4.5, color: C.ramp[rampIdx] },
      hovertemplate: "iter %{x}: %{y:.2e}<extra>stage " + (k + 1) + "</extra>",
    };
  });
  Plotly.react(
    "plot-gd",
    traces,
    baseLayout({
      height: 340,
      xaxis: axisDefaults({ title: { text: "iteration m", font: { size: 11 } }, dtick: 1 }),
      yaxis: axisDefaults({ type: "log", title: { text: "max |Δx|", font: { size: 11 } }, exponentformat: "power" }),
    }),
    PLOT_CONFIG
  );
}

function renderError(p) {
  const m = p.metrics;
  const K = p.windows.length;
  const stages = Array.from({ length: K }, (_, i) => i + 1);
  const lin = errorMode === "raw" ? m.stage_rmse_linear : m.stage_rmse_linear_scaled;
  const gd = errorMode === "raw" ? m.stage_rmse_gd : m.stage_rmse_gd_scaled;
  const traces = [
    {
      x: stages, y: lin, name: "linear", mode: "lines+markers",
      line: { color: C.linear, width: 1.8 }, marker: { size: 7 },
      hovertemplate: "stage %{x}: %{y:.4f}<extra>linear</extra>",
    },
    {
      x: stages, y: gd, name: gdLabel(p), mode: "lines+markers",
      line: { color: C.gd, width: 1.8 }, marker: { size: 7, symbol: "square" },
      hovertemplate: "stage %{x}: %{y:.4f}<extra>" + gdLabel(p) + "</extra>",
    },
  ];
  Plotly.react(
    "plot-error",
    traces,
    baseLayout({
      height: 320,
      margin: { l: 54, r: 16, t: 12, b: 48 },
      xaxis: axisDefaults({
        tickmode: "array", tickvals: stages,
        ticktext: stages.map((k) => `k = ${k}<br><span style="font-size:9px">w = ${p.windows[k - 1]}</span>`),
      }),
      yaxis: axisDefaults({
        rangemode: "tozero",
        title: { text: errorMode === "raw" ? "RMSE" : "RMSE ∕ a^(k∕2)", font: { size: 11 } },
      }),
    }),
    PLOT_CONFIG
  );

  $("stage-table").innerHTML = `
    <tr><th>stage</th><th>window</th><th>RMSE linear</th><th>RMSE GD</th>
        <th>scaled linear</th><th>scaled GD</th><th>GD iters</th></tr>
    ${stages
      .map(
        (k) => `<tr><td>S${sub(k)}</td><td>${p.windows[k - 1]}</td>
          <td>${m.stage_rmse_linear[k - 1].toFixed(4)}</td>
          <td>${m.stage_rmse_gd[k - 1].toFixed(4)}</td>
          <td>${m.stage_rmse_linear_scaled[k - 1].toFixed(4)}</td>
          <td>${m.stage_rmse_gd_scaled[k - 1].toFixed(4)}</td>
          <td>${p.gd.iters[k - 1]}</td></tr>`
      )
      .join("")}`;
}

/* ---------- boot ---------- */

if (window.renderMathInElement) {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "\\(", right: "\\)", display: false },
      { left: "$$", right: "$$", display: true },
    ],
  });
}
refreshControlChrome();
