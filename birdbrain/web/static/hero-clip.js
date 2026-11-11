// Home page: "Auto-play latest visit". The latest visit's clip loops over its photo unless this browser unticked
// the box (localStorage). The hero is swapped when a new visit arrives, so this re-runs on birdbrain:cards.
(() => {
  const KEY = "birdbrain.heroAutoplay";
  const isOn = () => { try { return localStorage.getItem(KEY) !== "0"; } catch (e) { return true; } };
  const show = (video, on) => {
    if (on && !video.getAttribute("src")) video.src = video.dataset.src;
    video.hidden = !on;
    on ? video.play().catch(() => {}) : video.pause();
  };
  const setup = () => {
    const hero = document.querySelector(".hero-latest");
    const video = hero && hero.querySelector(".hero-clip"), box = hero && hero.querySelector("#hero-autoplay");
    if (!video || !box || box.dataset.ready) return;
    box.dataset.ready = "1";
    box.checked = isOn();
    box.addEventListener("change", async () => {
      try { localStorage.setItem(KEY, box.checked ? "1" : "0"); } catch (e) {}
      show(video, box.checked);
      // The visit pages have their own box (clip.js, off unless ticked): offer it too, never set it silently.
      let visitOn = true;
      try { visitOn = localStorage.getItem("birdbrain.clipAutoplay") === "1"; } catch (e) {}
      if (box.checked && !visitOn && window.birdbrain && birdbrain.ask && await birdbrain.ask({
        title: "Auto-play visit clips too?",
        message: "You're turning on auto-play for the latest visit here. Would you like each visit's clip to play by itself when you open its page as well?",
        ok: "Yes, visits too", cancel: "Just here" })) {
        try { localStorage.setItem("birdbrain.clipAutoplay", "1"); } catch (e) {}
      }
    });
    show(video, box.checked);
  };
  setup();
  document.addEventListener("birdbrain:cards", setup);
})();
