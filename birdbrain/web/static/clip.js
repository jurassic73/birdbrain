// The visit page's Play clip button: the visit's short video plays in a loop in place of the photo; the magnifier
// opens it big in its own pop-up.
// The file is only fetched on the first press (preload="none", src set here), so opening a visit costs nothing.
(() => {
  const btn = document.getElementById("clip-play"), dlg = document.getElementById("clip-viewer");
  if (!btn || !dlg) return;
  const video = dlg.querySelector("video");
  const play = () => {
    if (!video.getAttribute("src")) video.src = video.dataset.src;
    dlg.showModal();
    video.currentTime = 0;
    video.play().catch(() => {});  // autoplay refused: the controls are there
  };
  // "Auto-play clips": a per-browser choice (localStorage), so every visit with a clip opens it on arrival.
  const KEY = "birdbrain.clipAutoplay", auto = document.getElementById("clip-autoplay");
  let on = false;
  try { on = localStorage.getItem(KEY) === "1"; } catch (e) {}
  // Auto-play shows the clip in place of the photo, at its normal size; the magnifier opens the big pop-up.
  // Ticking the box starts it on the visit already open; unticking puts the photo back.
  const box = document.getElementById("clip-inline");
  const small = box && box.querySelector("video");
  const inline = show => {
    if (!box) return;
    if (show && !small.getAttribute("src")) small.src = video.dataset.src;
    box.hidden = !show;
    show ? small.play().catch(() => {}) : small.pause();
  };
  // Play clip plays it in place too (from the start); without the in-place player it opens big.
  btn.addEventListener("click", () => {
    if (!box) return play();
    inline(true);
    small.currentTime = 0;
  });
  // The in-place player starts without the browser's play bar (it flashed up on every visit stepped to); the first tap brings it.
  if (small) small.addEventListener("click", () => { small.controls = true; }, {once: true});
  if (box) {
    document.getElementById("clip-inline-zoom").addEventListener("click", () => { small.pause(); play(); });
    dlg.addEventListener("close", () => { if (!box.hidden) small.play().catch(() => {}); });
  }
  if (auto) {
    auto.checked = on;
    auto.addEventListener("change", async () => {
      try { auto.checked ? localStorage.setItem(KEY, "1") : localStorage.removeItem(KEY); } catch (e) {}
      inline(auto.checked);
      // The home page's latest visit has its own box (hero-clip.js, on unless unticked): offer it too.
      let homeOn = true;
      try { homeOn = localStorage.getItem("birdbrain.heroAutoplay") !== "0"; } catch (e) {}
      if (auto.checked && !homeOn && window.birdbrain && birdbrain.ask && await birdbrain.ask({
        title: "Auto-play the latest visit too?",
        message: "You're turning on auto-play for visit clips. Would you like the latest visit's clip on the home page to play by itself as well?",
        ok: "Yes, home too", cancel: "Just visits" })) {
        try { localStorage.setItem("birdbrain.heroAutoplay", "1"); } catch (e) {}
      }
    });
  }
  if (on) inline(true);
  // Came here from the big clip's newer/older arrows (?zoom=clip; visit.html cleans it from the address and sets
  // window.bbClipZoom): open this visit's clip big again.
  if (window.bbClipZoom) play();
  dlg.addEventListener("close", () => video.pause());
  // A click beside the video or on the round button closes it; the video's own controls keep working.
  dlg.addEventListener("click", e => { if (e.target !== video && !e.target.closest(".photo-viewer-arrow")) dlg.close(); });
  // The page's arrow keys step to other visits: not while the clip is open (left and right seek in the video).
  dlg.addEventListener("keydown", e => e.stopPropagation());
})();
