// The rotate hint: a header button (phones held upright only, see .rotate-btn in style.css) opens a "Try landscape"
// pop-up (#rotate-dialog, looks: the Pop-ups section of style.css). Escape, Got it or a click on the dimmed page close it,
// and so does turning the phone sideways while it is open.
(() => {
  const btn = document.getElementById("rotate-btn"), dlg = document.getElementById("rotate-dialog");
  if (!btn || !dlg) return;
  const close = () => { if (dlg.open) dlg.close(); };
  btn.addEventListener("click", () => { if (!dlg.open) { dlg.showModal(); document.getElementById("rotate-ok").focus(); } });
  document.getElementById("rotate-ok").addEventListener("click", close);
  dlg.addEventListener("click", (e) => { if (e.target === dlg) close(); });
  matchMedia("(orientation: landscape)").addEventListener("change", (e) => { if (e.matches) close(); });
})();
