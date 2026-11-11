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
  if (box) {
    document.getElementById("clip-inline-zoom").addEventListener("click", () => { small.pause(); play(); });
    dlg.addEventListener("close", () => { if (!box.hidden) small.play().catch(() => {}); });
  }
  if (auto) {
    auto.checked = on;
    auto.addEventListener("change", () => {
      try { auto.checked ? localStorage.setItem(KEY, "1") : localStorage.removeItem(KEY); } catch (e) {}
      inline(auto.checked);
    });
  }
  if (on) inline(true);
  dlg.addEventListener("close", () => video.pause());
  // A click beside the video or on the round button closes it; the video's own controls keep working.
  dlg.addEventListener("click", e => { if (e.target !== video) dlg.close(); });
  // The page's arrow keys step to other visits: not while the clip is open (left and right seek in the video).
  dlg.addEventListener("keydown", e => e.stopPropagation());
})();
