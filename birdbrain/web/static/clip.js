// The visit page's Play clip button: the visit's short video opens big in its own pop-up and plays in a loop.
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
  btn.addEventListener("click", play);
  // "Auto-play clips": a per-browser choice (localStorage), so every visit with a clip opens it on arrival.
  const KEY = "birdbrain.clipAutoplay", auto = document.getElementById("clip-autoplay");
  let on = false;
  try { on = localStorage.getItem(KEY) === "1"; } catch (e) {}
  if (auto) {
    auto.checked = on;
    auto.addEventListener("change", () => {
      try { auto.checked ? localStorage.setItem(KEY, "1") : localStorage.removeItem(KEY); } catch (e) {}
    });
  }
  // Auto-play shows the clip in place of the photo, at its normal size; the magnifier opens the big pop-up.
  const box = document.getElementById("clip-inline");
  if (on && box) {
    const small = box.querySelector("video");
    small.src = video.dataset.src;
    box.hidden = false;
    small.play().catch(() => {});
    document.getElementById("clip-inline-zoom").addEventListener("click", () => { small.pause(); play(); });
    dlg.addEventListener("close", () => small.play().catch(() => {}));
  }
  dlg.addEventListener("close", () => video.pause());
  // A click beside the video or on the round button closes it; the video's own controls keep working.
  dlg.addEventListener("click", e => { if (e.target !== video) dlg.close(); });
  // The page's arrow keys step to other visits: not while the clip is open (left and right seek in the video).
  dlg.addEventListener("keydown", e => e.stopPropagation());
})();
