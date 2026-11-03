// Settings > Camera: the tabs, and the live view beside them so a saved change can be seen.
// The panel is swapped by htmx after every save, so everything here looks the elements up afresh.
(() => {
  const EVERY = 3000;
  let busy = false;

  function showTab(name) {
    const panel = document.getElementById("camera-panel");
    if (!panel) return;
    panel.querySelectorAll("[data-cam-tab]").forEach((b) => b.setAttribute("aria-selected", b.dataset.camTab === name));
    panel.querySelectorAll("[data-cam-pane]").forEach((p) => { p.hidden = p.dataset.camPane !== name; });
  }

  document.addEventListener("click", (e) => {
    const b = e.target.closest("[data-cam-tab]");
    if (b) showTab(b.dataset.camTab);
  });

  // A fresh frame, decoded off-screen before it replaces the one showing, so it never flashes.
  async function refresh() {
    const img = document.getElementById("cam-live-img");
    const card = img?.closest("details");
    if (!img || busy || document.hidden || (card && !card.open)) return;
    busy = true;
    try {
      const r = await fetch("/live.jpg?t=" + Date.now(), { cache: "no-store" });
      if (!r.ok) throw new Error("down");
      const takenAt = Date.now() / 1000 - parseFloat(r.headers.get("X-Frame-Age") || "0");
      const url = URL.createObjectURL(await r.blob());
      const next = new Image();
      next.src = url;
      await next.decode();
      const old = img.src;
      img.src = url;
      if (old.startsWith("blob:")) URL.revokeObjectURL(old);
      const saved = Number(img.closest(".cam-layout")?.dataset.saved || 0);
      const note = document.getElementById("cam-live-note");
      if (note) {
        const at = new Date(takenAt * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit", second: "2-digit" });
        note.textContent = saved && takenAt > saved + 1 ? `Taken ${at}, after your change.`
          : saved ? `Taken ${at}, before your change; a newer one is coming.` : `Live view, taken ${at}.`;
      }
    } catch {
      const note = document.getElementById("cam-live-note");
      if (note) note.textContent = "Couldn't get a fresh frame; trying again.";
    } finally {
      busy = false;
    }
  }

  setInterval(refresh, EVERY);
  // Right after a save the panel comes back: look again soon rather than waiting a full turn.
  document.addEventListener("htmx:afterSwap", (e) => { if (e.target.id === "camera-panel" || e.detail?.elt?.id === "camera-panel") setTimeout(refresh, 800); });
  document.addEventListener("htmx:load", (e) => { if (e.target.id === "camera-panel") refresh(); });
})();
