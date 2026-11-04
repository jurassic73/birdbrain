// Species page: the left/right keys go to the previous/next species (the same links as the arrows in the top bar,
// .sp-nav, which keep the list's sort) and Up goes back to the Species list (<link rel="up">, same sort).
// The keyboard pop-up and the other keys are in shortcuts.js, which this needs (window.bbShortcuts). Add a key
// here and a row to {% block keys_page %} in species.html.
(() => {
  const free = window.bbShortcuts?.free;
  if (!free) return;
  document.addEventListener("keydown", (e) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== "ArrowUp") return;
    if (!free(e)) return;
    if (e.key === "ArrowUp") {
      // Up goes back to the Species list, but only at the top of the page, where Up has nothing left to scroll;
      // held down, it scrolls up and stops there.
      const up = document.querySelector('link[rel="up"]');
      if (!up || scrollY > 0) return;
      e.preventDefault();
      location.href = up.href;
      return;
    }
    const link = document.getElementById(e.key === "ArrowLeft" ? "sp-prev" : "sp-next");
    if (link) location.href = link.href;
  });
})();
