// The visit page's circles on thumbnails (the photo_marks macro): fractions of the photo (r of its width).
// The svg uses the photo's own size as its viewBox, and "slice" matches the img's object-fit: cover.
(() => {
  const NS = "http://www.w3.org/2000/svg";

  function draw(svg) {
    const img = svg.previousElementSibling, w = img.naturalWidth, h = img.naturalHeight;
    if (!w) return;
    svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
    svg.replaceChildren(...JSON.parse(svg.dataset.marks).map(m => {
      const g = document.createElementNS(NS, "g");
      if (m.claude) g.setAttribute("class", "claude");
      for (const cls of ["ring-shadow", "ring"]) {
        const c = document.createElementNS(NS, "circle");
        c.setAttribute("cx", m.x * w); c.setAttribute("cy", m.y * h); c.setAttribute("r", m.r * w); c.setAttribute("class", cls);
        g.append(c);
      }
      return g;
    }));
  }

  function init(root) {
    root.querySelectorAll("svg.thumb-marks:not([viewBox])").forEach(svg => {
      const img = svg.previousElementSibling;
      img.complete ? draw(svg) : img.addEventListener("load", () => draw(svg), {once: true});
    });
  }

  document.addEventListener("DOMContentLoaded", () => init(document));
  document.addEventListener("htmx:load", e => init(e.target));  // cards added by "load more"
  document.addEventListener("birdbrain:cards", e => init(e.target));  // new visitors slid in by the home page
})();
