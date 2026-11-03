// Tooltips for any element with a data-tip attribute (every status pill gets one from the badge macro).
// Mouse: hover shows it. Touch: tapping a pill that isn't inside a link shows it; tapping elsewhere hides it.
// The tip lives on <body> so cards with overflow: hidden can't clip it.
// data-tip-anchor="<selector>" places the tip outside that ancestor (below it, else above), e.g. off a photo.
// Pressing anywhere else hides the tip, so it never stays over something being used.
(() => {
  const tip = document.createElement("div");
  tip.className = "tip";
  tip.setAttribute("role", "tooltip");
  tip.hidden = true;
  document.body.append(tip);

  let timer = null, current = null;
  function hide() {
    clearTimeout(timer);
    current = null;
    tip.hidden = true;
  }
  function show(el) {
    current = el;
    tip.textContent = el.dataset.tip;
    tip.hidden = false;
    const anchor = (el.dataset.tipAnchor && el.closest(el.dataset.tipAnchor)) || el;
    const r = anchor.getBoundingClientRect();
    const w = tip.offsetWidth, h = tip.offsetHeight;
    const x = Math.min(Math.max(8, r.left + r.width / 2 - w / 2), innerWidth - w - 8);
    const above = r.top - h - 6;
    let top = above < 8 ? r.bottom + 6 : above;
    if (anchor !== el) top = r.bottom + h + 14 <= innerHeight || above < 8 ? r.bottom + 6 : above;
    tip.style.left = `${x}px`;
    tip.style.top = `${top}px`;
  }

  if (matchMedia("(hover: hover) and (pointer: fine)").matches) {
    document.addEventListener("mouseover", (e) => {
      const el = e.target.closest("[data-tip]");
      if (!el || el === current) return;
      hide();
      current = el;
      timer = setTimeout(() => current === el && show(el), 200);
    });
    document.addEventListener("mouseout", (e) => {
      const el = e.target.closest("[data-tip]");
      if (el && !el.contains(e.relatedTarget)) hide();
    });
  } else {
    document.addEventListener("click", (e) => {
      const el = e.target.closest("[data-tip]");
      if (el && !el.closest("a") && el !== current) show(el);
      else hide();
    });
  }
  document.addEventListener("pointerdown", (e) => {
    if (current && !current.contains(e.target)) hide();
  }, true);
  addEventListener("scroll", hide, { passive: true });
})();
