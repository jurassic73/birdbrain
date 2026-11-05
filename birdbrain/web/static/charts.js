function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function hourLabels() {
  return Array.from({ length: 24 }, (_, h) => (h % 12 || 12) + (h < 12 ? "a" : "p"));
}

const birdbrainCharts = [];

function chartColors() {
  return { bar: cssVar("--accent"), grid: cssVar("--line"), text: cssVar("--muted") };
}

function styleChart(chart) {
  const c = chartColors();
  chart.data.datasets[0].backgroundColor = c.bar;
  chart.options.scales.x.ticks.color = c.text;
  chart.options.scales.y.ticks.color = c.text;
  chart.options.scales.y.grid.color = c.grid;
}

function barChart(id, labels, data) {
  const el = document.getElementById(id);
  if (!el || typeof Chart === "undefined") return;
  const chart = new Chart(el, {
    type: "bar",
    data: { labels, datasets: [{ data, borderRadius: 3 }] },
    options: {
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true } },
        y: { beginAtZero: true, grid: {}, ticks: { precision: 0 } },
      },
    },
  });
  styleChart(chart);
  chart.update();
  birdbrainCharts.push(chart);
}

// 100% stacked bars, one per day: BioCLIP 2 (accent) under Claude (terracotta, --claude). days: [{day, bioclip, claude}]; a day with
// no IDs is an empty slot. Used by Stats.
function styleShare(chart) {
  const c = chartColors();
  const [bio, claude] = chart.data.datasets;
  bio.backgroundColor = c.bar;
  claude.backgroundColor = `color-mix(in srgb, ${cssVar("--claude")} 60%, ${cssVar("--surface")})`;
  bio.borderColor = claude.borderColor = cssVar("--surface");
  chart.options.scales.x.ticks.color = c.text;
  chart.options.scales.y.ticks.color = c.text;
  chart.options.scales.y.grid.color = c.grid;
}

function shareChart(id, days) {
  const el = document.getElementById(id);
  if (!el || typeof Chart === "undefined") return;
  const pct = (n, d) => (d ? (100 * n) / d : null);
  const dataset = (label, key, share) => ({
    label, stack: "s", borderRadius: 3, borderWidth: { top: 2 },
    data: days.map(d => { const t = d.bioclip + d.claude; return t ? share(d, t) : null; }),
  });
  const chart = new Chart(el, {
    type: "bar",
    data: {
      labels: days.map(d => shortDay(d.day)),
      datasets: [dataset("BioCLIP 2", "bioclip", (d, t) => pct(d.bioclip, t)), dataset("Claude", "claude", (d, t) => pct(d.claude, t))],
    },
    options: {
      maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: items => shortDay(days[items[0].dataIndex].day),
            label: item => {
              const d = days[item.dataIndex];
              return item.datasetIndex ? `Claude: ${d.claude} (${Math.round(item.raw)}%)` : `BioCLIP 2: ${d.bioclip} (${Math.round(item.raw)}%)`;
            },
          },
        },
      },
      scales: {
        x: { stacked: true, grid: { display: false }, ticks: { maxRotation: 0, autoSkip: true } },
        y: { stacked: true, min: 0, max: 100, grid: {}, ticks: { stepSize: 25, callback: v => v + "%" } },
      },
    },
  });
  chart.$restyle = styleShare;
  styleShare(chart);
  chart.update();
  birdbrainCharts.push(chart);
}

// One dot per item, each labelled beside it where there's room (crowded dots keep their date for the hover tip). points: [{rain, rate,
// visitors, daylight, day 'YYYY-MM-DD', partial}]; partial dots are hollow. Used by the Weather & visits page.
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function shortDay(day) {
  const [, m, d] = day.split("-").map(Number);
  return MONTHS[m - 1] + " " + d;
}

const dotLabels = {
  id: "dotLabels",
  afterDatasetsDraw(chart) {
    const { ctx, chartArea } = chart;
    const meta = chart.getDatasetMeta(0);
    const points = chart.data.datasets[0].data;
    const taken = meta.data.map(el => ({ x0: el.x - 7, x1: el.x + 7, y0: el.y - 7, y1: el.y + 7 }));
    ctx.save();
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = cssVar("--muted");
    ctx.textBaseline = "middle";
    meta.data.forEach((el, i) => {
      const text = shortDay(points[i].day);
      const w = ctx.measureText(text).width, h = 12;
      const spots = [[el.x + 9, el.y], [el.x - 9 - w, el.y], [el.x + 6, el.y - 12], [el.x + 6, el.y + 12],
        [el.x - 6 - w, el.y - 12], [el.x - 6 - w, el.y + 12], [el.x - w / 2, el.y - 15], [el.x - w / 2, el.y + 15]];
      const pick = spots.find(([x, y]) => x >= chartArea.left - 4 && x + w <= chartArea.right + 34 && y >= chartArea.top - 4 && y <= chartArea.bottom + 4 &&
        !taken.some(r => x < r.x1 && x + w > r.x0 && y - h / 2 < r.y1 && y + h / 2 > r.y0));
      if (!pick) return;  // too crowded: this dot's date shows on hover instead
      taken.push({ x0: pick[0], x1: pick[0] + w, y0: pick[1] - h / 2, y1: pick[1] + h / 2 });
      ctx.fillText(text, pick[0], pick[1]);
    });
    ctx.restore();
  },
};

function styleScatter(chart) {
  const c = chartColors();
  const ds = chart.data.datasets[0];
  ds.borderColor = c.bar;
  ds.backgroundColor = ds.data.map(p => (p.partial ? cssVar("--surface") : c.bar));
  for (const axis of ["x", "y"]) {
    chart.options.scales[axis].ticks.color = c.text;
    chart.options.scales[axis].grid.color = c.grid;
    chart.options.scales[axis].title.color = c.text;
  }
}

function scatterChart(id, points, titles) {
  const el = document.getElementById(id);
  if (!el || typeof Chart === "undefined" || !points.length) return;
  const data = points.map(p => ({ ...p, x: p.rain, y: p.rate }));
  const chart = new Chart(el, {
    type: "scatter",
    data: { datasets: [{ data, pointRadius: 5, pointHoverRadius: 7, pointHitRadius: 12, borderWidth: 2 }] },
    options: {
      maintainAspectRatio: false,
      layout: { padding: { right: 36, top: 8 } },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            title: items => shortDay(items[0].raw.day) + (items[0].raw.partial ? " (partly watched)" : ""),
            label: item => [
              `${item.raw.rain.toFixed(1)} mm of rain`,
              `${item.raw.rate.toFixed(1)} visitors per daylight hour`,
              `${item.raw.visitors} visitors in ${item.raw.daylight} h`,
            ],
          },
        },
      },
      scales: {
        x: { beginAtZero: true, grid: {}, ticks: {}, title: { display: true, text: titles.x } },
        y: { beginAtZero: true, grid: {}, ticks: {}, title: { display: true, text: titles.y } },
      },
    },
    plugins: [dotLabels],
  });
  chart.$restyle = styleScatter;
  styleScatter(chart);
  chart.update();
  birdbrainCharts.push(chart);
}

// Repaint charts in the new theme's colors when the toggle (or the device setting) changes it.
document.addEventListener("themechange", () => {
  birdbrainCharts.forEach(chart => { (chart.$restyle || styleChart)(chart); chart.update("none"); });
});
