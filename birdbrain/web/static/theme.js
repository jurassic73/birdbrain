// Light/dark toggle and the color palette button. With no saved choice the page follows the device setting; clicking saves an
// explicit theme in this browser. Pages that draw with JS (charts) listen for "themechange".
(() => {
  const root = document.documentElement;
  const button = document.getElementById("theme-toggle");
  const deviceDark = window.matchMedia("(prefers-color-scheme: dark)");
  const palette = document.getElementById("palette-toggle");
  const PALETTES = ["forest", "ocean", "sunset", "plum"];
  const NAMES = { forest: "Forest", ocean: "Ocean", sunset: "Sunset", plum: "Plum" };
  const currentPalette = () => root.dataset.palette || "forest";

  const saved = () => {
    try { return localStorage.getItem("theme"); } catch (e) { return null; }
  };
  const current = () => root.dataset.theme || (deviceDark.matches ? "dark" : "light");

  let shown = null;  // theme and palette last announced with "themechange"

  // read: take the saved choice again first (false right after a click, which may not have been saved).
  function sync(read = true) {
    window.bbTheme.apply(read);  // bar color, from base.html's early script
    const theme = current();
    const label = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
    if (button) {
      button.setAttribute("aria-label", label);
      button.title = label;
    }
    if (palette) {
      const now = currentPalette(), next = PALETTES[(PALETTES.indexOf(now) + 1) % PALETTES.length];
      palette.title = `Colors: ${NAMES[now]}. Click for ${NAMES[next]}`;
      palette.setAttribute("aria-label", palette.title);
    }
    if (shown === theme + currentPalette()) return;
    shown = theme + currentPalette();
    document.dispatchEvent(new CustomEvent("themechange", { detail: { theme } }));
  }

  if (button) {
    button.addEventListener("click", () => {
      const next = current() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (e) { /* private mode: still switches */ }
      sync(false);
    });
  }
  if (palette) {
    palette.addEventListener("click", () => {
      const next = PALETTES[(PALETTES.indexOf(currentPalette()) + 1) % PALETTES.length];
      if (next === "forest") delete root.dataset.palette; else root.dataset.palette = next;
      try { localStorage.setItem("palette", next); } catch (e) { /* private mode: still switches */ }
      sync(false);
    });
  }
  deviceDark.addEventListener("change", () => { if (!saved()) sync(); });
  // Keep the theme and the bar color right however a page comes to be shown (the Android navigation bar kept
  // the light color after opening a visit): a normal load, a prerendered visit page (home's speculation rules)
  // being shown, a page restored from the back/forward cache, the tab coming back, full screen ending, the
  // theme changed in another tab, and HTML swapped in by htmx.
  const refresh = () => { sync(); window.bbTheme.nudge(); };
  addEventListener("pageshow", refresh);
  document.addEventListener("prerenderingchange", refresh);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") refresh(); });
  document.addEventListener("fullscreenchange", () => { if (!document.fullscreenElement) refresh(); });
  addEventListener("storage", e => { if (e.key === null || e.key === "theme" || e.key === "palette") sync(); });
  document.addEventListener("htmx:afterSettle", () => window.bbTheme.apply(false));
  sync();
})();
