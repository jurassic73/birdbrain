// The public mirror's anonymous counts (birdbrain/audience.py; base.html loads this only where counting is on).
// A click on any element with data-track="name" sends that name to /api/track; a script can count one itself with
// window.bbTrack("name"). Only names on the server's list count. A page the browser prerendered (and the server
// therefore didn't count) is counted here once it is really shown. Nothing here reads or stores anything about you.
(() => {
  const base = window.BB_BASE || "";
  const send = (query) => {
    const url = `${base}/api/track?${query}`;
    try {
      if (navigator.sendBeacon && navigator.sendBeacon(url)) return;
    } catch (e) {}
    fetch(url, { method: "POST", keepalive: true, credentials: "same-origin" }).catch(() => {});
  };
  window.bbTrack = (name) => { if (name) send("to=" + encodeURIComponent(name)); };

  // Capture phase, so a control whose own handler stops the click is still counted.
  document.addEventListener("click", (e) => {
    const el = e.target instanceof Element && e.target.closest("[data-track]");
    if (el) window.bbTrack(el.dataset.track);
  }, true);
  // The species cards' recordings are <audio> players: count a play started from one.
  document.addEventListener("play", (e) => {
    if (e.target instanceof HTMLMediaElement && e.target.closest(".sp-sounds")) window.bbTrack("bird_song");
  }, true);

  const page = document.querySelector('meta[name="bb-page"]')?.content;
  if (!page) return;
  if (document.prerendering) {
    document.addEventListener("prerenderingchange", () => send("view=" + encodeURIComponent(page)), { once: true });
  } else {
    const nav = performance.getEntriesByType?.("navigation")[0];
    if (nav && nav.deliveryType === "navigational-prefetch") send("view=" + encodeURIComponent(page));
  }
})();
