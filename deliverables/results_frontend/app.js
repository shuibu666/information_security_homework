const points = [
  { method: "KGW", param: "δ = 0.5", tpr: 30.2, simcse: 0.7248, ppl: 22.06 },
  { method: "KGW", param: "δ = 1.0", tpr: 92.27, simcse: 0.6901, ppl: 23.97 },
  { method: "KGW", param: "δ = 1.5", tpr: 98.73, simcse: 0.6761, ppl: 26.08 },
  { method: "KGW", param: "δ = 1.7", tpr: 98.93, simcse: 0.6729, ppl: 27.1 },
  { method: "KGW", param: "δ = 2.0", tpr: 99.33, simcse: 0.6636, ppl: 29.67 },
  { method: "KGW", param: "δ = 2.5", tpr: 99.53, simcse: 0.661, ppl: 34.91 },
  { method: "KGW", param: "δ = 3.0", tpr: 99.73, simcse: 0.6515, ppl: 41.37 },
  { method: "CA-KL", param: "ε = 0.02", tpr: 29.67, simcse: 0.7218, ppl: 22.97 },
  { method: "CA-KL", param: "ε = 0.05", tpr: 70.47, simcse: 0.7012, ppl: 24.22 },
  { method: "CA-KL", param: "ε = 0.10", tpr: 94.8, simcse: 0.6875, ppl: 25.25 },
  { method: "CA-KL", param: "ε = 0.20", tpr: 98.8, simcse: 0.6752, ppl: 28.25 },
  { method: "CA-KL", param: "ε = 0.30", tpr: 99.33, simcse: 0.6692, ppl: 30.1 },
  { method: "CA-KL", param: "ε = 0.40", tpr: 99.6, simcse: 0.6678, ppl: 32.23 },
  { method: "CA-KL", param: "ε = 0.50", tpr: 99.53, simcse: 0.6566, ppl: 34.14 },
];

const palette = {
  KGW: "#2f6fb0",
  "CA-KL": "#c84b47",
};

function setupCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const scale = window.devicePixelRatio || 1;
  canvas.width = Math.max(1, Math.floor(rect.width * scale));
  canvas.height = Math.max(1, Math.floor(rect.height * scale));
  const ctx = canvas.getContext("2d");
  ctx.setTransform(scale, 0, 0, scale, 0, 0);
  return { ctx, width: rect.width, height: rect.height };
}

function catmullRomPath(ctx, coords) {
  if (coords.length < 2) return;
  ctx.moveTo(coords[0].x, coords[0].y);
  for (let i = 0; i < coords.length - 1; i += 1) {
    const p0 = coords[Math.max(0, i - 1)];
    const p1 = coords[i];
    const p2 = coords[i + 1];
    const p3 = coords[Math.min(coords.length - 1, i + 2)];
    const cp1x = p1.x + (p2.x - p0.x) / 6;
    const cp1y = p1.y + (p2.y - p0.y) / 6;
    const cp2x = p2.x - (p3.x - p1.x) / 6;
    const cp2y = p2.y - (p3.y - p1.y) / 6;
    ctx.bezierCurveTo(cp1x, cp1y, cp2x, cp2y, p2.x, p2.y);
  }
}

function drawChart(canvasId, options) {
  const canvas = document.getElementById(canvasId);
  const { ctx, width, height } = setupCanvas(canvas);
  const margin = { top: 28, right: 24, bottom: 54, left: 64 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const sx = (value) =>
    margin.left + ((value - options.xMin) / (options.xMax - options.xMin)) * plotW;
  const sy = (value) =>
    margin.top + (1 - (value - options.yMin) / (options.yMax - options.yMin)) * plotH;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#d9e0ea";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#667085";
  ctx.font = "12px Inter, sans-serif";

  for (let i = 0; i <= 5; i += 1) {
    const x = margin.left + (plotW * i) / 5;
    const y = margin.top + (plotH * i) / 5;
    ctx.beginPath();
    ctx.moveTo(x, margin.top);
    ctx.lineTo(x, margin.top + plotH);
    ctx.moveTo(margin.left, y);
    ctx.lineTo(margin.left + plotW, y);
    ctx.stroke();
  }

  ctx.strokeStyle = "#172033";
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(margin.left, margin.top);
  ctx.lineTo(margin.left, margin.top + plotH);
  ctx.lineTo(margin.left + plotW, margin.top + plotH);
  ctx.stroke();

  ctx.textAlign = "center";
  ctx.fillText(options.xLabel, margin.left + plotW / 2, height - 16);
  ctx.save();
  ctx.translate(18, margin.top + plotH / 2);
  ctx.rotate(-Math.PI / 2);
  ctx.fillText("TPR@1%FPR", 0, 0);
  ctx.restore();

  for (const method of ["KGW", "CA-KL"]) {
    const group = points
      .filter((p) => p.method === method)
      .filter((p) => options.include(p))
      .sort((a, b) => a[options.xField] - b[options.xField]);
    const coords = group.map((p) => ({ x: sx(p[options.xField]), y: sy(p.tpr / 100), p }));

    ctx.strokeStyle = palette[method];
    ctx.lineWidth = 3;
    ctx.beginPath();
    catmullRomPath(ctx, coords);
    ctx.stroke();

    for (const item of coords) {
      ctx.fillStyle = palette[method];
      ctx.beginPath();
      if (method === "KGW") {
        ctx.arc(item.x, item.y, 4.5, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillRect(item.x - 4.5, item.y - 4.5, 9, 9);
      }
      ctx.fillStyle = palette[method];
      ctx.textAlign = "left";
      ctx.font = "11px Inter, sans-serif";
      ctx.fillText(item.p.param.replace(" = ", "="), item.x + 6, item.y - 6);
    }
  }

  ctx.font = "12px Inter, sans-serif";
  const legendX = margin.left + 10;
  const legendY = margin.top + 10;
  [
    ["KGW", "KGW fixed-delta"],
    ["CA-KL", "CA-KL epsilon"],
  ].forEach(([method, label], index) => {
    const y = legendY + index * 22;
    ctx.strokeStyle = palette[method];
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(legendX, y);
    ctx.lineTo(legendX + 26, y);
    ctx.stroke();
    ctx.fillStyle = "#344054";
    ctx.textAlign = "left";
    ctx.fillText(label, legendX + 36, y + 4);
  });
}

function renderCharts() {
  drawChart("simChart", {
    xField: "simcse",
    xMin: 0.648,
    xMax: 0.69,
    yMin: 0.94,
    yMax: 1.0,
    xLabel: "SimCSE cosine (higher is better)",
    include: (p) => p.simcse <= 0.69 && p.tpr >= 94,
  });
  drawChart("pplChart", {
    xField: "ppl",
    xMin: 21.2,
    xMax: 42.2,
    yMin: 0.25,
    yMax: 1.01,
    xLabel: "Corpus PPL (lower is better)",
    include: () => true,
  });
}

function renderRows(filter = "all") {
  const body = document.getElementById("resultRows");
  body.innerHTML = "";
  points
    .filter((point) => filter === "all" || point.method === filter)
    .forEach((point) => {
      const row = document.createElement("tr");
      row.innerHTML = `
        <td><span class="method-pill ${point.method === "KGW" ? "method-kgw" : "method-cakl"}">${point.method}</span></td>
        <td>${point.param}</td>
        <td>${point.tpr.toFixed(2)}%</td>
        <td>${point.simcse.toFixed(4)}</td>
        <td>${point.ppl.toFixed(2)}</td>
      `;
      body.appendChild(row);
    });
}

document.querySelectorAll(".filter-button").forEach((button) => {
  button.addEventListener("click", () => {
    document.querySelectorAll(".filter-button").forEach((item) => item.classList.remove("active"));
    button.classList.add("active");
    renderRows(button.dataset.filter);
  });
});

let resizeTimer;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(renderCharts, 120);
});

renderRows();
renderCharts();
