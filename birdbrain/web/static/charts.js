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

// Repaint charts in the new theme's colors when the toggle (or the device setting) changes it.
document.addEventListener("themechange", () => {
  birdbrainCharts.forEach(chart => { styleChart(chart); chart.update("none"); });
});
