// Flag new visitors in the browser tab: green favicon plus "(n) New visitor: ..." in the title.
// Pages can't color the tab itself; this is the same trick mail and chat apps use.
// Each poll that finds something new also fires "birdbrain:visits" on the document, so pages can pull in
// new cards, numbers and arrows (detail.real: a new real visitor; detail.any: any new visit, false triggers too).
(() => {
  const POLL_MS = 20000;
  const SHOW_WHILE_VISIBLE_MS = 15000;
  const icon = document.querySelector('link[rel="icon"][type="image/png"]');
  const normalIcon = icon ? icon.href : null;
  const greenIcon = "/static/favicon-new-64.png";
  const baseTitle = document.title;
  let lastSeen = null, lastAny = null;
  let unseen = 0;
  let clearTimer = null;
  let busy = false;

  function clear() {
    unseen = 0;
    document.title = baseTitle;
    if (icon && normalIcon) icon.href = normalIcon;
  }

  function flag(label) {
    unseen += 1;
    document.title = `(${unseen}) New visitor: ${label} | Birdbrain`;
    if (icon) icon.href = greenIcon;
    // If you're already looking at the tab, show it briefly and then settle back.
    clearTimeout(clearTimer);
    if (!document.hidden) clearTimer = setTimeout(clear, SHOW_WHILE_VISIBLE_MS);
  }

  async function check() {
    if (busy) return;
    busy = true;
    try {
      const r = await fetch("/api/latest-visit", { cache: "no-store" });
      const v = await r.json();
      if (lastSeen === null) { lastSeen = v.id; lastAny = v.any_id; return; }
      const real = v.id > lastSeen, any = v.any_id > lastAny;
      if (real) { lastSeen = v.id; flag(v.label); }
      if (any) lastAny = v.any_id;
      if (real || any) document.dispatchEvent(new CustomEvent("birdbrain:visits", { detail: { real, any, id: v.id } }));
    } catch (e) { /* atmos unreachable; try again next poll */ }
    finally { busy = false; }
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) return;
    // Phones pause timers in background tabs: catch up the moment the page is back in view.
    check();
    if (unseen) {
      clearTimeout(clearTimer);
      clearTimer = setTimeout(clear, 1500);
    }
  });
  addEventListener("pageshow", e => { if (e.persisted) check(); });  // back to a page the browser kept in memory
  check();
  setInterval(check, POLL_MS);
})();
