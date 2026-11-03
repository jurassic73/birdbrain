// The star on each photo card (admins): unset -> species cover (yellow) -> favorite (blue, kept) -> unset.
// The server decides the new state and says which visit is the species' cover now, so only one star per species
// is yellow; blue follows each visit's keep flag. A small pill beside the star says what it just became.
// Delegated and in the capture phase, so cards slid in later by the home page work too and the click never
// reaches the card's link or the lightbox.
(() => {
  const PILL = {unset: "unset", cover: "species cover", favorite: "favorite"};

  function paint(star, cover) {
    const st = Number(star.dataset.id) === cover ? "cover" : star.dataset.keep === "1" ? "favorite" : "unset";
    star.dataset.state = st;
    star.setAttribute("aria-pressed", st === "unset" ? "false" : "true");
    star.dataset.tip = star.dataset["tip" + st[0].toUpperCase() + st.slice(1)];
    star.setAttribute("aria-label", star.dataset.tip);
    // the bookmark on the same card follows keep
    const mark = star.closest(".card").querySelector(".keep-mark");
    if (mark) {
      const on = star.dataset.keep === "1";
      mark.setAttribute("aria-pressed", on ? "true" : "false");
      mark.setAttribute("hx-vals", JSON.stringify({on: on ? 0 : 1}));
    }
  }

  function coverOf(species) {
    for (const s of document.querySelectorAll(".fav-star[data-state=cover]")) if (s.dataset.species === species) return Number(s.dataset.id);
    return null;
  }

  function flash(star, state) {
    const pill = star.closest(".card").querySelector(".fav-pill");
    if (!pill) return;
    pill.textContent = PILL[state];
    pill.dataset.state = state;
    pill.hidden = false;
    pill.classList.add("show");
    clearTimeout(pill._t);
    pill._t = setTimeout(() => {
      pill.classList.remove("show");
      pill._t = setTimeout(() => { pill.hidden = true; }, 400);
    }, 2500);
  }

  document.addEventListener("click", async (e) => {
    const star = e.target.closest && e.target.closest(".fav-star");
    if (!star) return;
    e.preventDefault();
    e.stopPropagation();
    if (star.disabled) return;
    star.disabled = true;
    try {
      const body = new FormData();
      body.append("visit_id", star.dataset.id);
      const res = await fetch("/species/" + encodeURIComponent(star.dataset.species) + "/photo", {method: "POST", body});
      if (!res.ok) return;
      const r = await res.json();
      star.dataset.keep = r.state === "unset" ? "0" : "1";
      document.querySelectorAll(".fav-star").forEach((b) => {
        if (b.dataset.species === r.species) paint(b, r.cover_id);
      });
      flash(star, r.state);
    } finally {
      star.disabled = false;
    }
  }, true);

  // The bookmark was clicked: its old copy still holds the value it sent, so the star on that card follows it.
  document.addEventListener("htmx:afterRequest", (e) => {
    const btn = e.detail.elt;
    if (!btn.classList || !btn.classList.contains("keep-mark") || !e.detail.successful) return;
    const id = (btn.getAttribute("hx-post").match(/visit\/(\d+)/) || [])[1];
    const star = document.querySelector('.fav-star[data-id="' + id + '"]');
    if (!star) return;
    star.dataset.keep = JSON.parse(btn.getAttribute("hx-vals")).on ? "1" : "0";
    paint(star, coverOf(star.dataset.species));
  });

  // Cards slid in by the home page aren't known to htmx yet.
  document.addEventListener("birdbrain:cards", (e) => { if (window.htmx) htmx.process(e.target); });
})();
