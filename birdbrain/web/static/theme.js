// Light/dark toggle and the color palette button. With no saved choice the page follows the device setting; clicking saves an
// explicit theme in this browser. Pages that draw with JS (charts) listen for "themechange".
(() => {
  const root = document.documentElement;
  const button = document.getElementById("theme-toggle");
  const meta = document.getElementById("theme-color");
  const deviceDark = window.matchMedia("(prefers-color-scheme: dark)");
  const palette = document.getElementById("palette-toggle");
  // Page backgrounds per palette, for the browser's toolbar color; defined in base.html's early script.
  const BG = window.bbBg;
  const PALETTES = ["forest", "ocean", "sunset", "plum"];
  const NAMES = { forest: "Forest", ocean: "Ocean", sunset: "Sunset", plum: "Plum" };
  const currentPalette = () => root.dataset.palette || "forest";

  const saved = () => {
    try { return localStorage.getItem("theme"); } catch (e) { return null; }
  };
  const current = () => root.dataset.theme || (deviceDark.matches ? "dark" : "light");

  function sync() {
    const theme = current();
    const label = theme === "dark" ? "Switch to light mode" : "Switch to dark mode";
    if (button) {
      button.setAttribute("aria-label", label);
      button.title = label;
    }
    if (meta) meta.content = BG[currentPalette()][theme];
    if (palette) {
      const now = currentPalette(), next = PALETTES[(PALETTES.indexOf(now) + 1) % PALETTES.length];
      palette.title = `Colors: ${NAMES[now]}. Click for ${NAMES[next]}`;
      palette.setAttribute("aria-label", palette.title);
    }
    document.dispatchEvent(new CustomEvent("themechange", { detail: { theme } }));
  }

  if (button) {
    button.addEventListener("click", () => {
      const next = current() === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      try { localStorage.setItem("theme", next); } catch (e) { /* private mode: still switches */ }
      sync();
    });
  }
  if (palette) {
    palette.addEventListener("click", () => {
      const next = PALETTES[(PALETTES.indexOf(currentPalette()) + 1) % PALETTES.length];
      if (next === "forest") delete root.dataset.palette; else root.dataset.palette = next;
      try { localStorage.setItem("palette", next); } catch (e) { /* private mode: still switches */ }
      sync();
    });
  }
  deviceDark.addEventListener("change", () => { if (!saved()) sync(); });
  sync();
})();
