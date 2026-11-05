// Site-wide keyboard shortcuts (desktop keyboards; keys work at any width, the keyboard button in the top bar is
// desktop only) and the "Keyboard shortcuts" pop-up (#keys-dialog in base.html, looks: the Pop-ups section of
// style.css). A page lists its own keys in its {% block keys_page %}; the site-wide keys are listed in base.html.
//   ?  opens the pop-up     1 2 3 4  Visits, Species, Live, Stats     T  light/dark     P  next palette
// (A visit page with a burst takes P for play/stop instead of the palette.)
// Pages with their own keys (home, visit, species-nav.js) keep them; window.bbShortcuts.free(e) is the shared
// "is this key press ours to use" check, so every handler skips the same situations. Not loaded in embeds.
(() => {
  const root = document.documentElement;

  // Skip: key repeat, a key somebody already handled, Ctrl/Meta/Alt (Shift too, unless the key needs it, like ?),
  // the tour, typing in a field, any open pop-up, the full-screen photo viewer and the scene editor.
  const free = (e, shift = false) => {
    if (e.repeat || e.defaultPrevented || e.metaKey || e.altKey || e.ctrlKey || (e.shiftKey && !shift)) return false;
    if (root.dataset.tour) return false;
    if (e.target.closest?.("input, textarea, select, [contenteditable]")) return false;
    if (document.querySelector("dialog[open]")) return false;
    const lb = document.querySelector(".lb");
    if (lb && !lb.hidden) return false;
    if (root.classList.contains("se-open") || document.querySelector(".se-big")) return false;
    return true;
  };
  window.bbShortcuts = { free };

  const dlg = document.getElementById("keys-dialog"), btn = document.getElementById("keys-btn");
  const openKeys = () => { if (dlg && !dlg.open) { dlg.showModal(); document.getElementById("keys-ok").focus(); } };
  if (dlg) {
    const close = () => { if (dlg.open) dlg.close(); };
    document.getElementById("keys-ok").addEventListener("click", close);
    dlg.addEventListener("click", (e) => { if (e.target === dlg) close(); });
  }
  if (btn) btn.addEventListener("click", openKeys);

  // The top nav's own links (Visits, Species, Live, Stats come first), so the public mirror's /birdbrain prefix is
  // already in them. A visit page with unsaved changes asks first (bbUnsaved, visit.html).
  const go = (href) => {
    if (window.bbUnsaved) window.bbUnsaved.check().then((ok) => { if (ok) location.href = href; });
    else location.href = href;
  };
  const click = (id) => document.getElementById(id)?.click();  // exactly what a click on the button does (theme.js)

  document.addEventListener("keydown", (e) => {
    if (!free(e, e.key === "?")) return;
    if (e.key === "?") { e.preventDefault(); openKeys(); return; }
    const k = e.key.toLowerCase();
    if (k === "t") { e.preventDefault(); click("theme-toggle"); return; }
    if (k === "p") { e.preventDefault(); click("palette-toggle"); return; }
    const to = "1234".indexOf(k);
    if (to < 0 || k.length !== 1) return;
    const link = document.querySelectorAll("header.topbar nav a")[to];
    if (!link) return;
    e.preventDefault();
    go(link.href);
  });
})();
