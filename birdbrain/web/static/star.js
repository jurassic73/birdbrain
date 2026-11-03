// The star on each photo card, the home page's big latest-visitor photo and the visit page's big photo (admins):
// unset -> favorite (blue, kept) -> species cover (yellow) -> unset.
// Yellow is held for HOLD ms first, so tapping on through to unset never touches the cover. Once it settles, a
// species that already has another cover asks first, with the two photos side by side, each cropped around the
// animal's circle; "Keep current" puts the star back to blue. The server sets the state it's sent and says which
// visit is the species' cover now, so only one star per species is yellow; blue follows each visit's keep flag.
// A visit with no species (no data-star-species) has no cover: its star only toggles favorite, via /visit/<id>/keep.
// A small pill beside the star says what it just became.
// Delegated and in the capture phase, so cards slid in later by the home page work too and the click never
// reaches the card's link or the lightbox.
(() => {
  const PILL = {unset: "unset", cover: "species cover", favorite: "favorite"};
  const BOX = ".card, .hero-latest, .visit-photo";  // what a star sits in
  const HOLD = 2000;
  const holds = new Map();  // visit id -> timer while its star sits on a not-yet-saved yellow

  const starsOf = (id) => document.querySelectorAll('.fav-star[data-id="' + id + '"]');

  function paint(star, cover) {
    const st = Number(star.dataset.id) === cover ? "cover" : star.dataset.keep === "1" ? "favorite" : "unset";
    delete star.dataset.pending;
    star.dataset.state = st;
    star.setAttribute("aria-pressed", st === "unset" ? "false" : "true");
    star.dataset.tip = star.dataset["tip" + st[0].toUpperCase() + st.slice(1)];
    star.setAttribute("aria-label", star.dataset.tip);
    // the bookmark on the same card follows keep
    const mark = star.closest(BOX).querySelector(".keep-mark");
    if (mark) {
      const on = star.dataset.keep === "1";
      mark.setAttribute("aria-pressed", on ? "true" : "false");
      mark.setAttribute("hx-vals", JSON.stringify({on: on ? 0 : 1}));
    }
  }

  function coverOf(species) {
    for (const s of document.querySelectorAll(".fav-star[data-state=cover]:not([data-pending])"))
      if (s.dataset.starSpecies === species) return Number(s.dataset.id);
    return null;
  }

  function flash(star, state) {
    const pill = star.closest(BOX).querySelector(".fav-pill");
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

  async function save(star, state) {
    const id = star.dataset.id;
    starsOf(id).forEach((b) => { b.disabled = true; });
    try {
      if (!star.dataset.starSpecies) {  // no species: favorite on/off only
        const body = new FormData();
        body.append("on", state === "favorite" ? "1" : "0");
        const res = await fetch("/visit/" + id + "/keep", {method: "POST", body});
        if (res.ok) starsOf(id).forEach((b) => { b.dataset.keep = state === "favorite" ? "1" : "0"; paint(b, null); flash(star, state); });
        return;
      }
      const body = new FormData();
      body.append("visit_id", id);
      body.append("state", state);
      const res = await fetch("/species/" + encodeURIComponent(star.dataset.starSpecies) + "/photo", {method: "POST", body});
      if (!res.ok) { starsOf(id).forEach((b) => paint(b, coverOf(b.dataset.starSpecies))); return; }
      const r = await res.json();
      // The latest visit has two stars (big photo and first card): both follow.
      const keep = r.state === "unset" ? "0" : "1";
      starsOf(id).forEach((b) => { b.dataset.keep = keep; });
      document.querySelectorAll(".fav-star").forEach((b) => {
        // another photo of the species still waiting on its own yellow keeps waiting
        if (b.dataset.starSpecies === r.species && !(b.dataset.pending && b.dataset.id !== id)) paint(b, r.cover_id);
      });
      flash(star, r.state);
    } finally {
      starsOf(id).forEach((b) => { b.disabled = false; });
    }
  }

  // Yellow, not saved yet: a third tap within HOLD goes straight to unset.
  function hold(star) {
    const id = star.dataset.id;
    starsOf(id).forEach((b) => {
      b.dataset.pending = "1";
      b.dataset.state = "cover";
      b.setAttribute("aria-pressed", "true");
    });
    flash(star, "cover");
    holds.set(id, setTimeout(() => settle(star), HOLD));
  }

  function release(id) {
    clearTimeout(holds.get(id));
    holds.delete(id);
  }

  async function settle(star) {
    const id = star.dataset.id, species = star.dataset.starSpecies;
    holds.delete(id);
    let r = null;
    try {
      const res = await fetch("/species/" + encodeURIComponent(species) + "/cover-compare?visit_id=" + id, {cache: "no-store"});
      if (res.ok) r = await res.json();
    } catch (e) { /* r stays null: put it back */ }
    if (r && (!r.cover || await askReplace(r))) return save(star, "cover");
    starsOf(id).forEach((b) => paint(b, coverOf(species)));  // Keep current: back to blue
    flash(star, "favorite");
  }

  // ---- "Replace the cover?" with the two photos side by side (looks: .popup-cover in style.css) ----
  let dlg = null, answer = null;

  function dialog() {
    if (dlg) return dlg;
    dlg = document.createElement("dialog");
    dlg.className = "popup popup-cover";
    dlg.setAttribute("aria-labelledby", "cover-ask-title");
    dlg.innerHTML = '<h2 class="popup-title" id="cover-ask-title"></h2>'
      + '<div class="cover-compare">'
      + '<figure class="old"><div class="cover-crop" role="img"></div><figcaption><strong>Current cover</strong><span class="muted"></span></figcaption></figure>'
      + '<figure class="new"><div class="cover-crop" role="img"></div><figcaption><strong>This photo</strong><span class="muted"></span></figcaption></figure>'
      + '</div>'
      + '<div class="popup-msg"><p class="muted">Either way, both photos stay favorites (never deleted when space is cleared).</p></div>'
      + '<div class="popup-actions"><button type="button" class="popup-cancel">Keep current</button>'
      + '<button type="button" class="popup-ok">Use this photo</button></div>';
    const done = (yes) => { const a = answer; answer = null; if (dlg.open) dlg.close(); a?.(yes); };
    dlg.querySelector(".popup-ok").addEventListener("click", () => done(true));
    dlg.querySelector(".popup-cancel").addEventListener("click", () => done(false));
    dlg.addEventListener("close", () => done(false));  // Escape
    dlg.addEventListener("click", (e) => { if (e.target === dlg) done(false); });  // the dimmed page
    document.body.append(dlg);
    return dlg;
  }

  // Show the part of the photo around the animal's circle (a 4:3 box about 5 circle radii wide); no circle: all of it.
  async function crop(box, photo) {
    box.style.backgroundImage = 'url("' + photo.src + '")';
    box.setAttribute("aria-label", photo.when);
    const img = new Image();
    img.src = photo.src;
    await img.decode().catch(() => {});
    const W = img.naturalWidth, H = img.naturalHeight, m = photo.mark;
    if (!W || !H || !m) {
      box.style.backgroundSize = "contain";
      box.style.backgroundPosition = "center";
      return;
    }
    const R = (window.markRadius ? window.markRadius(m.r) : Math.max(m.r * 1.15, 0.06)) * W;
    let cw = Math.min(W, Math.max(R * 5, W * 0.2)), ch = cw * 3 / 4;
    if (ch > H) { ch = H; cw = Math.min(W, H * 4 / 3); }
    const x0 = Math.min(Math.max(m.x * W - cw / 2, 0), W - cw), y0 = Math.min(Math.max(m.y * H - ch / 2, 0), H - ch);
    box.style.backgroundSize = (W / cw * 100) + "% auto";
    box.style.backgroundPosition = (W > cw ? x0 / (W - cw) * 100 : 50) + "% " + (H > ch ? y0 / (H - ch) * 100 : 50) + "%";
  }

  async function askReplace(r) {
    const d = dialog();
    if (answer) { answer(false); answer = null; }
    d.querySelector(".popup-title").textContent = "Replace the " + r.species + " cover?";
    const [oldFig, newFig] = d.querySelectorAll("figure");
    oldFig.querySelector(".muted").textContent = r.cover.when;
    newFig.querySelector(".muted").textContent = r.candidate.when;
    await Promise.all([crop(oldFig.querySelector(".cover-crop"), r.cover), crop(newFig.querySelector(".cover-crop"), r.candidate)]);
    return new Promise((resolve) => {
      answer = resolve;
      d.showModal();
      d.querySelector(".popup-ok").focus();
    });
  }

  document.addEventListener("click", (e) => {
    const star = e.target.closest && e.target.closest(".fav-star");
    if (!star) return;
    e.preventDefault();
    e.stopPropagation();
    if (star.disabled) return;
    const id = star.dataset.id;
    if (star.dataset.pending) {
      if (!holds.has(id)) return;  // already settling (asking the server or the owner)
      release(id);
      save(star, "unset");
    } else if (star.dataset.state === "unset") {
      save(star, "favorite");
    } else if (star.dataset.state === "favorite" && star.dataset.starSpecies) {
      hold(star);
    } else {
      save(star, "unset");
    }
  }, true);

  // The bookmark was clicked: its old copy still holds the value it sent, so the star on that card follows it.
  document.addEventListener("htmx:afterRequest", (e) => {
    const btn = e.detail.elt;
    if (!btn.classList || !btn.classList.contains("keep-mark") || !e.detail.successful) return;
    const id = (btn.getAttribute("hx-post").match(/visit\/(\d+)/) || [])[1];
    const keep = JSON.parse(btn.getAttribute("hx-vals")).on ? "1" : "0";
    release(id);
    starsOf(id).forEach((star) => {
      star.dataset.keep = keep;
      paint(star, coverOf(star.dataset.starSpecies));
    });
  });

  // Cards slid in by the home page aren't known to htmx yet.
  document.addEventListener("birdbrain:cards", (e) => { if (window.htmx) htmx.process(e.target); });
})();
