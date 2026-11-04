// Home page: the facts box at the bottom of the Today panel (.today-facts) shows only when it fits beside the latest-visitor
// photo. The panel is exactly as tall as the photo (style.css, contain: size) and the visitors list takes what is left and
// scrolls, so "fits" means: with the box showing, neither the panel nor the list overflows. Facts go from the last one up
// (3, then 2, then 1) before the box gives up; if even one doesn't fit the whole box is hidden (never cut off, the panel
// never grows, the visitors list always wins the room). Re-checked on resize and whenever the page swaps the panel, the
// photo or the scoreboard (new visitor, quick filters, period buttons).
(() => {
  const hero = document.querySelector(".hero");
  if (!hero) return;
  const SKIP = matchMedia("(max-width: 800px), (orientation: landscape) and (max-height: 500px)");  // as in style.css
  function fit() {
    const box = hero.querySelector(".today-facts");
    if (!box) return;
    const panel = box.closest(".today");
    if (!panel || SKIP.matches) return;  // a phone: CSS leaves it out
    const facts = [...box.querySelectorAll("li")];
    const list = panel.querySelector(".critters ul");
    const crowded = () => panel.scrollHeight > panel.clientHeight + 1 || (list && list.scrollHeight > list.clientHeight + 1);
    box.hidden = false;  // measure with it showing; nothing paints in between
    facts.forEach(li => { li.hidden = false; });
    for (let n = facts.length; n > 1 && crowded(); n--) facts[n - 1].hidden = true;
    box.hidden = crowded();
  }
  new MutationObserver(fit).observe(hero, { childList: true, subtree: true });
  new ResizeObserver(fit).observe(hero);
  addEventListener("load", fit);
  document.fonts?.ready.then(fit);
  fit();
})();
