// The visit page's Play clip button: the visit's short video opens big in its own pop-up and plays in a loop.
// The file is only fetched on the first press (preload="none", src set here), so opening a visit costs nothing.
(() => {
  const btn = document.getElementById("clip-play"), dlg = document.getElementById("clip-viewer");
  if (!btn || !dlg) return;
  const video = dlg.querySelector("video");
  btn.addEventListener("click", () => {
    if (!video.getAttribute("src")) video.src = video.dataset.src;
    dlg.showModal();
    video.currentTime = 0;
    video.play().catch(() => {});  // autoplay refused: the controls are there
  });
  dlg.addEventListener("close", () => video.pause());
  // A click beside the video or on the round button closes it; the video's own controls keep working.
  dlg.addEventListener("click", e => { if (e.target !== video) dlg.close(); });
  // The page's arrow keys step to other visits: not while the clip is open (left and right seek in the video).
  dlg.addEventListener("keydown", e => e.stopPropagation());
})();
