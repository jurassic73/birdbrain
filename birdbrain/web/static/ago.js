// Keeps every "5 min ago" on the site current (home page, species page and its pop-up). The server renders the label and
// the age in seconds (data-age); from there the browser's own clock counts on, so the site's time zone never
// matters. Wording matches ago() in web/app.py. A swapped-in hero is picked up on the next tick or its cards event.
(() => {
  const plural = (n, unit) => `${n} ${unit}${n === 1 ? "" : "s"} ago`;
  function daysAgo(d) {
    if (d < 7) return plural(d, "day");
    if (d < 30) return plural(Math.floor(d / 7), "week");
    if (d < 365) return plural(Math.floor(d / 30), "month");
    return plural(Math.floor(d / 365), "year");
  }
  function text(s) {
    if (s < 60) return "just now";
    if (s < 3600) return `${Math.floor(s / 60)} min ago`;
    if (s < 86400) return `~${plural(Math.floor(s / 3600), "hour")}`;
    return daysAgo(Math.floor(s / 86400));
  }
  // Bracketed after a clock time; data-bare drops the brackets.
  const label = (s, bare) => bare ? text(s) : `(${text(s)})`;
  function tick() {
    document.querySelectorAll(".ago[data-age]").forEach(el => {
      if (!el.dataset.t0) el.dataset.t0 = Date.now();
      const s = Number(el.dataset.age) + Math.floor((Date.now() - Number(el.dataset.t0)) / 1000);
      el.textContent = label(s, "bare" in el.dataset);
    });
  }
  tick();
  setInterval(tick, 30000);
  document.addEventListener("birdbrain:cards", tick);
  document.addEventListener("htmx:load", tick);  // the species pop-up and the species page's card arrive over htmx
})();
