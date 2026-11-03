// Full-size photo browser for the species page: the expand button (.lb-open) opens every photo in the grid
// (a.card[data-full] inside #visits, read when opened, so "Load more" tiles count) in a 95% box over a dimmed page.
// Left/right (or the arrows, or a swipe) step and wrap around; Escape, ArrowUp, x or a click outside the photo close it.
(() => {
  let box, img, count, when, tiles = [], i = 0, touchX = null, opener = null;

  function build() {
    box = document.createElement("div");
    box.className = "lb";
    box.hidden = true;
    box.setAttribute("role", "dialog");
    box.setAttribute("aria-modal", "true");
    box.setAttribute("aria-label", "Photos full size");
    box.innerHTML = '<div class="lb-stage"><img class="lb-img" alt="">' +
      '<div class="lb-cap"><span class="lb-count"></span> <span class="lb-when"></span></div></div>' +
      '<button type="button" class="lb-btn lb-prev" aria-label="Previous photo">&#8249;</button>' +
      '<button type="button" class="lb-btn lb-next" aria-label="Next photo">&#8250;</button>' +
      '<button type="button" class="lb-btn lb-close" aria-label="Close">&times;</button>';
    document.body.appendChild(box);
    img = box.querySelector(".lb-img");
    count = box.querySelector(".lb-count");
    when = box.querySelector(".lb-when");
    box.querySelector(".lb-prev").onclick = () => step(-1);
    box.querySelector(".lb-next").onclick = () => step(1);
    box.querySelector(".lb-close").onclick = close;
    // a click anywhere but the photo or a button is the backdrop
    box.addEventListener("click", (e) => { if (e.target === box || e.target.classList.contains("lb-stage")) close(); });
    box.addEventListener("touchstart", (e) => { touchX = e.touches[0].clientX; }, { passive: true });
    box.addEventListener("touchend", (e) => {
      if (touchX === null) return;
      const dx = e.changedTouches[0].clientX - touchX;
      touchX = null;
      if (Math.abs(dx) > 50) step(dx < 0 ? 1 : -1);
    });
  }

  function show() {
    const t = tiles[i];
    img.src = t.dataset.full;
    img.alt = t.querySelector("img")?.alt || "";
    count.textContent = (i + 1) + " / " + tiles.length;
    when.textContent = t.dataset.when || "";
    // warm the neighbours so stepping feels instant
    [1, -1].forEach((d) => { new Image().src = tiles[(i + d + tiles.length) % tiles.length].dataset.full; });
  }

  function step(d) {
    i = (i + d + tiles.length) % tiles.length;
    show();
  }

  function open(at) {
    tiles = [...document.querySelectorAll("#visits a.card[data-full]")];
    if (!tiles.length) return;
    if (!box) build();
    i = at || 0;
    box.querySelectorAll(".lb-prev, .lb-next").forEach((b) => { b.hidden = tiles.length < 2; });
    show();
    box.hidden = false;
    document.body.style.overflow = "hidden";
    box.querySelector(".lb-close").focus();
  }

  function close() {
    if (!box || box.hidden) return;
    box.hidden = true;
    img.removeAttribute("src");
    document.body.style.overflow = "";
    if (opener) opener.focus();
  }

  document.addEventListener("click", (e) => {
    const b = e.target.closest(".lb-open");
    if (!b) return;
    opener = b;
    open(0);
  });

  document.addEventListener("keydown", (e) => {
    if (!box || box.hidden) return;
    if (e.key === "Escape" || e.key === "ArrowUp") close();
    else if (e.key === "ArrowLeft") step(-1);
    else if (e.key === "ArrowRight") step(1);
    else return;
    e.preventDefault();
    e.stopPropagation();
  }, true);
})();
