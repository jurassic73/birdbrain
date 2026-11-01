// Everything drawn over photos (Claude's circles, the zones) and the switches that hide it, one place for all pages.
// Loaded in <head> without defer, so a hidden overlay never flashes before the page paints.
(() => {
  // How big a circle is drawn over a photo, so every view agrees (thumbnails, the home photo, the visit page).
  // Display only: stored circles keep their true radius. r is a fraction of the photo's width. The ring sits a
  // little outside the animal (x1.15) and is never smaller than 6% of the width, so a small bird stays visible.
  window.markRadius = r => Math.max(r * 1.15, 0.06);

  // Circles and zones are each on or off for this browser (localStorage), shown as classes on <html> so the
  // card thumbnails, the visit photo and the full-size viewer all follow at once.
  const KEYS = {circles: "bb-circles", zones: "bb-visit-zones"};
  const root = document.documentElement;
  const read = kind => { try { return localStorage.getItem(KEYS[kind]) !== "off"; } catch { return true; } };
  const overlays = window.bbOverlays = {
    on: kind => !root.classList.contains(`${kind}-off`),
    set(kind, on) {
      try { localStorage.setItem(KEYS[kind], on ? "on" : "off"); } catch {}
      root.classList.toggle(`${kind}-off`, !on);
      document.dispatchEvent(new CustomEvent("birdbrain:overlays"));
    },
  };
  for (const kind of Object.keys(KEYS)) root.classList.toggle(`${kind}-off`, !read(kind));

  // The header's eye: one press hides every overlay (or shows them all again). Only on pages that have any.
  const sync = () => {
    const btn = document.getElementById("overlays-toggle");
    if (!btn) return;
    btn.hidden = !document.querySelector("svg.marks, svg.zones-overlay");
    const shown = overlays.on("circles") || overlays.on("zones");
    btn.setAttribute("aria-pressed", shown);
    btn.setAttribute("aria-label", shown ? "Hide circles and zones" : "Show circles and zones");
    btn.title = btn.getAttribute("aria-label");
  };
  document.addEventListener("DOMContentLoaded", () => {
    sync();
    document.getElementById("overlays-toggle")?.addEventListener("click", () => {
      const show = !(overlays.on("circles") || overlays.on("zones"));
      overlays.set("circles", show); overlays.set("zones", show);
    });
  });
  for (const ev of ["birdbrain:overlays", "birdbrain:cards", "htmx:load"]) document.addEventListener(ev, sync);
})();
