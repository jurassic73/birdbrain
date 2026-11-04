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

  // Jump to visits inside the pop-up: when the page is already that species page, close the pop-up so the scroll shows.
  dialog.addEventListener("click", (e) => {
    const a = e.target.closest("a.sp-jump");
    if (a && new URL(a.href).pathname === location.pathname) {
      // scroll by hand: following the link would race with the history entry the card gives back
      const target = a.hash && document.getElementById(a.hash.slice(1));
      if (target) { e.preventDefault(); dialog.close(); target.scrollIntoView(); } else dialog.close();
    }
  });

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
    const w = tip.offsetWidth, h = tip.offsetHeight;
    const x = Math.min(Math.max(8, r.left), innerWidth - w - 8);
    const below = r.bottom + 8;
    tip.style.left = `${x}px`;
    tip.style.top = `${below + h > innerHeight - 8 ? r.top - h - 8 : below}px`;
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
  // A map inside a closed fold has no size; fix it when the fold opens.
  document.addEventListener("toggle", (e) => {
    if (!e.target.classList?.contains("sp-more") || !e.target.open) return;
    for (const el of e.target.querySelectorAll(".sp-map")) el._map?.invalidateSize();
  }, true);
  document.addEventListener("htmx:afterSwap", (e) => initMaps(e.target));
  initMaps(document);  // cards rendered with the page
})();
