// Species info cards: hover a species name for a preview (mouse only), click or tap it for the full card.
(() => {
  // Our own copy (static/vendor), under the site's base path when it has one (the shim in base.html sets it).
  const LEAFLET = (window.BB_BASE || "") + "/static/vendor/leaflet-1.9.4/";
  const cache = new Map();
  const load = (name, mini) => {
    const url = `/species-info/${encodeURIComponent(name)}${mini ? "?mini=1" : ""}`;
    if (!cache.has(url)) {
      cache.set(url, fetch(url).then((r) => (r.ok ? r.text() : Promise.reject(r.status)))
        .catch((e) => { cache.delete(url); throw e; }));
    }
    return cache.get(url);
  };

  const dialog = document.createElement("dialog");
  dialog.className = "popup sp-dialog";
  dialog.innerHTML = '<button type="button" class="popup-close sp-close" aria-label="Close"><svg viewBox="0 0 16 16" width="14" height="14" aria-hidden="true" focusable="false"><path d="M3 3 13 13M13 3 3 13" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg></button><div class="sp-body"></div>';
  const body = dialog.querySelector(".sp-body");
  dialog.querySelector(".sp-close").addEventListener("click", () => dialog.close());
  dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });  // backdrop

  const tip = document.createElement("div");
  tip.className = "sp-tip";
  tip.hidden = true;
  document.body.append(dialog, tip);

  let hold = null;
  dialog.addEventListener("close", () => { if (hold) { const h = hold; hold = null; h.release(); } });

  async function open(name) {
    hideTip();
    const p = document.createElement("p");
    p.className = "muted sp-loading";
    p.textContent = `Loading ${name}…`;
    body.replaceChildren(p);
    dialog.showModal();
    hold = window.bbBack ? bbBack.hold(() => dialog.close()) : null;  // Back closes the card first
    try {
      body.innerHTML = await load(name, false);
      initMaps(body);
    } catch {
      p.textContent = "Couldn't load species info. Try again in a bit.";
    }
  }

  // Capture phase, so a name inside a visit card opens the species card instead of following the card's link.
  document.addEventListener("click", (e) => {
    const el = e.target.closest("[data-species]");
    if (!el || e.metaKey || e.ctrlKey || e.shiftKey) return;
    e.preventDefault();
    e.stopPropagation();
    open(el.dataset.species);
  }, true);
  document.addEventListener("keydown", (e) => {
    const el = e.target.closest?.("[data-species][role=button]");
    if (el && (e.key === "Enter" || e.key === " ")) {
      e.preventDefault();
      e.stopPropagation();
      open(el.dataset.species);
    }
  }, true);

  let timer = null, current = null;
  function hideTip() {
    clearTimeout(timer);
    current = null;
    tip.hidden = true;
  }
  function place(el) {
    const r = el.getBoundingClientRect();
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight, m = 8;
    const clampX = (x) => Math.min(Math.max(m, x), innerWidth - w - m);
    const clampY = (y) => Math.min(Math.max(m, y), innerHeight - h - m);
    const below = r.bottom + m;
    let x = clampX(r.left), y = below;
    if (below + h > innerHeight - m) {
      // No room below: sit beside the card so its own photo stays visible, else flip above the name.
      const box = el.closest(".card, .hero-latest, .visit-photo");
      const c = (box || el).getBoundingClientRect();
      if (box && c.right + m + w <= innerWidth - m) { x = c.right + m; y = clampY(r.top); }
      else if (box && c.left - m - w >= m) { x = c.left - m - w; y = clampY(r.top); }
      else y = r.top - h - m;
    }
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }
  if (matchMedia("(hover: hover) and (pointer: fine)").matches) {
    document.addEventListener("mouseover", (e) => {
      const el = e.target.closest("[data-species]");
      if (!el || el === current) return;
      hideTip();
      current = el;
      timer = setTimeout(async () => {
        try {
          const html = await load(el.dataset.species, true);
          if (current !== el || dialog.open) return;
          tip.innerHTML = html;
          place(el);
        } catch {}
      }, 350);
    });
    document.addEventListener("mouseout", (e) => {
      const el = e.target.closest("[data-species]");
      if (el && !el.contains(e.relatedTarget)) hideTip();
    });
    addEventListener("scroll", hideTip, { passive: true });
  }

  // The title's song/call buttons: one shared Audio so only one thing plays; the card's own <audio> players pause it (and it pauses them).
  const song = new Audio();
  song.preload = "none";
  let songBtn = null;
  const songState = () => document.querySelectorAll(".sp-play").forEach((b) => {
    const on = b === songBtn && !song.paused;
    b.setAttribute("aria-pressed", on ? "true" : "false");
    const name = b.getAttribute("aria-label").replace(/^(Play|Pause)/, on ? "Pause" : "Play");
    b.setAttribute("aria-label", name);
  });
  for (const ev of ["play", "pause", "ended", "error"]) song.addEventListener(ev, songState);
  document.addEventListener("click", (e) => {
    const b = e.target.closest(".sp-play");
    if (!b) return;
    if (songBtn === b && !song.paused) { song.pause(); return; }
    document.querySelectorAll(".sp-sounds audio").forEach((a) => a.pause());
    if (songBtn !== b || song.ended) { songBtn = b; song.src = b.dataset.src; }
    song.play().catch(songState);
  });
  document.addEventListener("play", (e) => {  // media events don't bubble, so listen in the capture phase
    if (e.target.closest && e.target.closest(".sp-sounds")) song.pause();
  }, true);

  let leaflet = null;
  function loadLeaflet() {
    if (window.L) return Promise.resolve();
    leaflet ??= new Promise((resolve, reject) => {
      const css = Object.assign(document.createElement("link"), { rel: "stylesheet", href: LEAFLET + "leaflet.css" });
      const js = Object.assign(document.createElement("script"), { src: LEAFLET + "leaflet.js", onload: resolve, onerror: reject });
      document.head.append(css, js);
    }).catch((e) => { leaflet = null; throw e; });
    return leaflet;
  }
  async function initMaps(root) {
    for (const el of root.querySelectorAll(".sp-map:not(.ready)")) {
      el.classList.add("ready");
      try {
        await loadLeaflet();
      } catch {
        el.textContent = "Map unavailable.";
        continue;
      }
      const home = JSON.parse(el.dataset.center || "null");
      const map = L.map(el, { scrollWheelZoom: false }).setView(home || [40, -100], home ? 4 : 3);
      L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 12, attribution: "&copy; OpenStreetMap" }).addTo(map);
      L.tileLayer(`https://api.inaturalist.org/v1/grid/{z}/{x}/{y}.png?taxon_id=${el.dataset.taxon}&color=%23e0592a`,
        { maxZoom: 12, opacity: 0.8 }).addTo(map);
      if (home) L.circleMarker(home, { radius: 6, color: "#1f2a24", weight: 2, fillColor: "#ffffff", fillOpacity: 1 }).addTo(map);
      el._map = map;
      setTimeout(() => map.invalidateSize(), 60);
    }
  }
  document.addEventListener("htmx:afterSwap", (e) => { initMaps(e.target); initMaps(document); });
  initMaps(document);  // cards rendered with the page
})();
