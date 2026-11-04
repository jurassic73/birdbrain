// Species page, desktop: the left/right keys go to the previous/next species (the same links as the arrows in the
// top bar, .sp-nav, which keep the list's sort), and the keyboard button opens the "Keyboard shortcuts" pop-up
// (#keys-dialog, looks: the Pop-ups section of style.css). Add a shortcut by adding a row to .keys-list and a case here.
(() => {
  const dlg = document.getElementById("keys-dialog"), btn = document.getElementById("keys-btn");
  if (btn && dlg) {
    const close = () => { if (dlg.open) dlg.close(); };
    btn.addEventListener("click", () => { if (!dlg.open) { dlg.showModal(); document.getElementById("keys-ok").focus(); } });
    document.getElementById("keys-ok").addEventListener("click", close);
    dlg.addEventListener("click", (e) => { if (e.target === dlg) close(); });
  }
  document.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    if (e.repeat || e.defaultPrevented || e.metaKey || e.altKey || e.ctrlKey || e.shiftKey) return;
    if (document.documentElement.dataset.tour) return;
    const t = e.target;
    if (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable) return;
    if (document.querySelector("dialog[open]")) return;
    const lb = document.querySelector(".lb");
    if (lb && !lb.hidden) return;
    const link = document.getElementById(e.key === "ArrowLeft" ? "sp-prev" : "sp-next");
    if (link) location.href = link.href;
  });
})();
