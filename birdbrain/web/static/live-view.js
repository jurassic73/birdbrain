// The Live tab (/live): the camera frame, refreshed in place, and the slim panel beside it.
//
// Stale-while-revalidate: the saved frame paints instantly (the img src in the page), then fresh frames are
// downloaded and decoded off-screen and swapped in only when complete, so the box never blanks. Clicking the frame
// opens it big (it keeps refreshing there); Escape, the x or a click closes it. The badge says Live, Last frame or
// Offline with the frame's age; the panel repeats the age and the light (X-Frame-Light, judged from the picture).
// A frame already on screen is never downloaded again: the page sends its ETag back and the server answers 304
// (the public mirror's frame only changes every few minutes, and a full-size frame is about half a megabyte).
// Nothing runs in a background tab, and the page catches up the moment it comes back. When a new visitor arrives
// (tab-alert.js fires birdbrain:visits) the panel's Today and Last visitor parts are fetched again and swapped in.
(() => {
  const box = document.getElementById("live-box");
  if (!box) return;
  const REFRESH_MS = Number(box.dataset.refreshMs) || 5000;
  const STALE_S = Number(box.dataset.stale) || 30;
  // The picture is faded only when the frame is older than it should be: the public mirror's frame is minutes old
  // by design (it arrives by push), so there the label says Last frame and its time, but the picture stays bright.
  const DIM_S = Number(box.dataset.dim) || STALE_S;
  const img = document.getElementById("live");
  const status = document.getElementById("live-status");
  const age = document.getElementById("live-age");
  const sideAge = document.getElementById("live-side-age");
  const viewer = document.getElementById("live-viewer");
  const big = document.getElementById("live-big");

  box.querySelectorAll(".live-zoom, .live-zoom-btn").forEach(b => b.addEventListener("click", e => {
    e.stopPropagation();  // keeps tips.js from showing its touch tooltip behind the viewer
    big.src = img.src;
    viewer.showModal();
    refresh();
  }));
  viewer.addEventListener("click", () => viewer.close());

  // Say Offline only after a while with no good frame and repeated failures: one slow camera grab isn't an outage.
  const OFFLINE_AFTER_MS = 15000;
  let frameAge = null, receivedAt = 0, failures = 0, lastOkAt = Date.now(), busy = false;
  let etag = null;  // of the frame on screen
  const isOffline = () => failures >= 2 && Date.now() - lastOkAt > OFFLINE_AFTER_MS;

  const ageNow = () => frameAge === null ? null : frameAge + (Date.now() - receivedAt) / 1000;
  const clock = s => new Date(Date.now() - s * 1000).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });

  function showLight(light) {  // the panel carries a day and a night pill; one shows
    if (light !== "day" && light !== "night") return;
    document.querySelectorAll("[data-light]").forEach(p => { p.hidden = p.dataset.light !== light; });
  }

  function render() {
    const a = ageNow();
    const offline = isOffline();
    const stale = a !== null && a > STALE_S;
    box.classList.toggle("stale", a !== null && a > DIM_S && !offline);
    box.classList.toggle("offline", offline && a === null);
    status.textContent = offline ? "Offline" : stale ? "Last frame" : "Live";
    age.textContent = a === null ? "" : stale ? clock(a) : `updated ${Math.round(a)} s ago`;
    if (sideAge) sideAge.textContent = offline ? "offline" : a === null ? "" : stale ? clock(a) : `${Math.round(a)} s ago`;
    box.querySelector(".dot").classList.toggle("off", offline || stale);
    document.getElementById("live-big-status").textContent = status.textContent;
    document.getElementById("live-big-age").textContent = age.textContent;
    viewer.querySelector(".dot").classList.toggle("off", offline || stale);
  }

  async function refresh() {
    if (busy) return;
    busy = true;
    try {
      const r = await fetch("/live.jpg", { cache: "no-store", signal: AbortSignal.timeout(10000),
                                           headers: etag ? { "If-None-Match": etag } : {} });
      if (r.status === 304) {  // the frame on screen is still the newest: only its age moves on
        const a = parseFloat(r.headers.get("X-Frame-Age"));
        if (!Number.isNaN(a)) { frameAge = a; receivedAt = Date.now(); }
        showLight(r.headers.get("X-Frame-Light"));
        lastOkAt = Date.now(); failures = 0;
        return;
      }
      if (!r.ok) throw new Error(r.status);
      const serverAge = parseFloat(r.headers.get("X-Frame-Age") || "0");
      const url = URL.createObjectURL(await r.blob());
      const next = new Image();
      next.src = url;
      await next.decode();  // fully decoded before it touches the page
      const old = img.src;
      img.src = url;
      big.src = url;
      if (old.startsWith("blob:")) URL.revokeObjectURL(old);
      showLight(r.headers.get("X-Frame-Light"));
      etag = r.headers.get("ETag");
      frameAge = serverAge; receivedAt = lastOkAt = Date.now(); failures = 0;
    } catch (e) {
      failures++;  // keep showing the last frame; labeled Offline only once failures add up
    } finally {
      busy = false;
      render();
    }
  }

  // Age of the instantly painted saved frame, so it is labeled honestly until the fresh one lands, and its ETag,
  // so the first refresh doesn't download the same frame a second time: the first refresh waits for both the
  // answer and the saved frame itself (8 s at most), and trusts the ETag only if that frame really is on screen.
  let savedEtag = null;
  const head = fetch("/live/latest.jpg", { method: "HEAD", cache: "no-store", signal: AbortSignal.timeout(5000) }).then(r => {
    if (r.ok && frameAge === null) {
      frameAge = parseFloat(r.headers.get("X-Frame-Age") || "0"); receivedAt = Date.now();
      savedEtag = r.headers.get("ETag");
      showLight(r.headers.get("X-Frame-Light"));
      render();
    }
  }).catch(() => {});
  const painted = new Promise(done => {
    if (img.complete) return done();
    img.addEventListener("load", done, { once: true });
    img.addEventListener("error", done, { once: true });
    setTimeout(done, 8000);
  });
  img.addEventListener("error", () => { etag = null; if (frameAge === null) { failures++; render(); } });
  Promise.all([head, painted]).then(() => {
    if (img.complete && img.naturalWidth && !img.src.startsWith("blob:")) etag = savedEtag;
    refresh();
  });
  // Not in a background tab: each fetch past the frame's max age makes the Pi grab a new one. The handler below catches up on return.
  setInterval(() => { if (!document.hidden) refresh(); }, REFRESH_MS);
  setInterval(render, 1000);
  document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });

  // A new visitor: fetch the page again and swap the Today and Last visitor parts of the panel.
  let updating = false, again = false;
  async function update() {
    if (updating) { again = true; return; }
    updating = true;
    try {
      const r = await fetch(location.href, { cache: "no-store", signal: AbortSignal.timeout(10000) });
      if (!r.ok) throw new Error(r.status);
      const doc = new DOMParser().parseFromString(await r.text(), "text/html");
      const old = document.getElementById("live-visits"), fresh = doc.getElementById("live-visits");
      if (old && fresh) {
        old.replaceWith(fresh);
        fresh.dispatchEvent(new CustomEvent("birdbrain:cards", { bubbles: true }));  // the thumbnail's circle
      }
    } catch (e) {
      document.dispatchEvent(new CustomEvent("birdbrain:update-failed"));  // tab-alert.js says so again on its next poll
    } finally {
      updating = false;
      if (again) { again = false; update(); }
    }
  }
  document.addEventListener("birdbrain:visits", update);
})();
