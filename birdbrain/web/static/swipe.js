// Phone swipe between pages. A horizontal swipe on a top-level page steps along the top nav bar:
// swipe left = next tab, swipe right = previous tab (no wrap-around at the ends).
//
// Where it runs: base.html puts data-swipe on <body> only on the top-level pages (/, /species, /stats, /review,
// /false-triggers, /notes) and never in embeds. The value picks the targets:
//   data-swipe="nav"   (default) the links of <header.topbar> <nav>, current one found by aria-current="page"
//   data-swipe="links" the page's own <link rel="prev"> and <link rel="next"> in <head> (for the species detail page)
//   no attribute / "off" nothing happens
// A page can opt out in Jinja with {% set swipe = "off" %} (or "links") before extending base.html, or at runtime
// with document.body.dataset.swipe = "off". Any element can carry data-no-swipe to be left alone.
// Touch events only (a mouse or trackpad never swipes) and only on phone-ish widths.
(() => {
  const MIN_DX = 70, RATIO = 2, MAX_MS = 600, NARROW = "(max-width: 820px)";
  let start = null;

  function blocked(el) {
    if (document.querySelector("dialog[open], .lb:not([hidden]), .se-big, html.se-open")) return true;
    if (document.body.style.overflow === "hidden") return true;
    if (window.visualViewport && window.visualViewport.scale > 1.01) return true; // pinch-zoomed in
    for (let n = el; n && n !== document.documentElement; n = n.parentElement) {
      if (n.nodeType !== 1) continue;
      if (n.matches("input, textarea, select, canvas, video, audio, [contenteditable], [data-no-swipe], .leaflet-container, .se-big, .lb")) return true;
      if (n.scrollWidth > n.clientWidth + 1 && /(auto|scroll)/.test(getComputedStyle(n).overflowX)) return true;
    }
    return false;
  }

  function targets() {
    const mode = document.body.dataset.swipe;
    if (mode === "links") {
      return { prev: document.querySelector('link[rel="prev"]')?.href, next: document.querySelector('link[rel="next"]')?.href };
    }
    if (mode !== "nav") return {};
    const links = [...document.querySelectorAll("header.topbar nav a")].filter((a) => a.offsetParent !== null);
    const at = links.findIndex((a) => a.getAttribute("aria-current") === "page");
    if (at < 0) return {};
    return { prev: links[at - 1]?.href, next: links[at + 1]?.href };
  }

  function go(url, dir) {
    const main = document.querySelector("main");
    const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!main || calm || !main.animate) { location.href = url; return; }
    // a short nudge and fade in the swipe direction, then load the next page
    const a = main.animate([{ transform: "none", opacity: 1 }, { transform: `translateX(${dir * -28}px)`, opacity: 0.35 }],
      { duration: 130, easing: "ease-in", fill: "forwards" });
    let done = false;
    const leave = () => { if (!done) { done = true; location.href = url; } };
    a.onfinish = leave;
    setTimeout(leave, 400);
  }

  document.addEventListener("touchstart", (e) => {
    start = null;
    if (e.touches.length !== 1 || !matchMedia(NARROW).matches) return;
    if (!document.body.dataset.swipe || document.body.dataset.swipe === "off") return;
    const t = e.touches[0];
    start = { x: t.clientX, y: t.clientY, at: e.timeStamp, el: e.target };
  }, { passive: true });

  document.addEventListener("touchmove", (e) => { if (e.touches.length > 1) start = null; }, { passive: true });
  document.addEventListener("touchcancel", () => { start = null; }, { passive: true });

  document.addEventListener("touchend", (e) => {
    const s = start;
    start = null;
    if (!s || e.touches.length) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - s.x, dy = t.clientY - s.y;
    if (Math.abs(dx) < MIN_DX || Math.abs(dx) < RATIO * Math.abs(dy) || e.timeStamp - s.at > MAX_MS) return;
    if (String(getSelection() || "")) return;
    if (blocked(s.el)) return;
    const to = targets();
    const url = dx < 0 ? to.next : to.prev;
    if (url) go(url, dx < 0 ? 1 : -1);
  }, { passive: true });
})();
