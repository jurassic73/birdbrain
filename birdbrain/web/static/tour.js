// Guided tour: a spotlight over a real element plus a small card (title, a sentence or two, Back / Next / Skip).
// Runs on its own the first time a browser sees the home page; an admin gets three extra steps once. "Run the tour
// again" (Settings, About) goes to /?tour=1. Seen flags live in localStorage; if storage fails it never starts by itself.
(() => {
  const SEEN = "bb_tour_seen", ADMIN_SEEN = "bb_tour_admin_seen", LOGIN = "bb_tour_login";
  const root = document.documentElement;
  const get = (k) => { try { return localStorage.getItem(k); } catch { return null; } };
  const set = (k) => { try { localStorage.setItem(k, "1"); } catch { /* nothing to do */ } };
  const storageOk = () => { try { localStorage.setItem("bb_tour_probe", "1"); localStorage.removeItem("bb_tour_probe"); return true; } catch { return false; } };

  // Rerun links clear the flags (the tour starts on the home page because of ?tour=1).
  document.addEventListener("click", (e) => {
    if (!e.target.closest("[data-tour-restart]")) return;
    try { localStorage.removeItem(SEEN); localStorage.removeItem(ADMIN_SEEN); } catch { /* ignore */ }
  });
  // The admin login form marks this tab, so the page it lands on can offer the admin part.
  document.addEventListener("submit", (e) => {
    if (e.target.matches("form[action$='/login']")) { try { sessionStorage.setItem(LOGIN, "1"); } catch { /* ignore */ } }
  });

  // part: "public" or "admin". sel: selectors tried in order (first one that shows wins), or with all:true every
  // visible match is highlighted together.
  const STEPS = [
    {part: "public", sel: [".hero-latest", ".hero #live-box"], title: "The latest visitor",
     text: "Click the photo to open that visit. The magnifier opens it big, where you can zoom, drag and press Reset view. \"Go to visit\" in the caption does the same as a click. So does the Right arrow key."},
    {part: "public", sel: ["#overlays-toggle"], title: "See the bird without the circles",
     text: "The eye hides Claude's circles and the zones drawn on photos so you can see the bird itself. Press it again to bring them back."},
    {part: "public", sel: [".today #live-box", ".today"], title: "Today and the live view",
     text: "The live view shows what the camera sees right now, with today's visitors and numbers under it."},
    {part: "public", sel: [".topbar nav"], title: "Find your way around",
     text: "Latest is this page. Visits lists every visit, Species shows each kind of animal seen, and Stats has the charts."},
    {part: "public", sel: [".grid .sp-name", ".sp-name"], title: "Tap a species name",
     text: "Any species name opens an info card with a photo, sounds, a range map and when it is usually seen here."},
    {part: "public", sel: ["#palette-toggle", "#theme-toggle"], all: true, title: "Colors and dark mode",
     text: "The half-filled circle changes the color palette. The sun or moon switches between light and dark. Both are remembered in this browser."},
    {part: "public", sel: [".topbar a[href='/about']"], title: "How it works",
     text: "The i button explains how Birdbrain spots, photographs and names visitors. You can run this tour again from there."},
    {part: "admin", sel: [".topbar a[href='/review']", ".topbar a[href='/false-triggers']"], all: true, title: "Review and false triggers",
     text: "Review holds visits Claude was unsure about. False triggers are the times nothing was there. On a visit page, \"Looks right\" confirms a label and the form corrects a wrong one."},
    {part: "admin", sel: [".topbar a[href='/notes']"], title: "Claude's Notes",
     text: "Everything Claude is told about your yard lives here: what the camera sees, what to look for and ignore, and the lessons from your corrections."},
    {part: "admin", sel: [".topbar a[href='/settings']"], title: "Settings",
     text: "Zones, camera, local checks, storage and alerts. The Run the tour again button is here too."},
  ];

  let steps = [], idx = 0, parts = [], els = {}, savedFocus = null;

  const visible = (el) => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0 && getComputedStyle(el).visibility !== "hidden"; };
  function targetOf(step) {
    const found = step.sel.map((s) => document.querySelector(s)).filter((el) => el && visible(el));
    return found.length ? (step.all ? found : [found[0]]) : null;
  }
  function rectOf(list) {
    let l = 1e9, t = 1e9, r = -1e9, b = -1e9;
    list.forEach((el) => { const x = el.getBoundingClientRect(); l = Math.min(l, x.left); t = Math.min(t, x.top); r = Math.max(r, x.right); b = Math.max(b, x.bottom); });
    return {left: l, top: t, right: r, bottom: b, width: r - l, height: b - t};
  }

  function build() {
    const wrap = document.createElement("div");
    wrap.className = "tour";
    wrap.innerHTML = '<div class="tour-block"></div><div class="tour-spot"></div>' +
      '<div class="tour-card" role="dialog" aria-modal="true" aria-labelledby="tour-title" tabindex="-1">' +
      '<div class="tour-count" id="tour-count"></div><h2 class="tour-title" id="tour-title"></h2><p class="tour-text" id="tour-text"></p>' +
      '<div class="tour-actions"><button type="button" class="link-btn tour-skip">Skip tour</button><span class="tour-gap"></span>' +
      '<button type="button" class="link-btn tour-back">Back</button><button type="button" class="tour-next">Next</button></div></div>';
    document.body.append(wrap);
    els = {wrap, spot: wrap.querySelector(".tour-spot"), card: wrap.querySelector(".tour-card"), count: wrap.querySelector("#tour-count"),
      title: wrap.querySelector("#tour-title"), text: wrap.querySelector("#tour-text"), back: wrap.querySelector(".tour-back"),
      next: wrap.querySelector(".tour-next"), skip: wrap.querySelector(".tour-skip")};
    els.skip.addEventListener("click", end);
    els.back.addEventListener("click", () => go(-1));
    els.next.addEventListener("click", () => go(1));
  }

  // Spotlight on the target; the card goes below it, else above, else beside it (wide screens), never on top of it
  // unless the window is too small for anything else.
  function place() {
    const t = targetOf(steps[idx]);
    if (!t) return;
    const pad = 6, r = rectOf(t), vw = innerWidth, vh = innerHeight;
    const s = els.spot.style;
    s.left = r.left - pad + "px"; s.top = r.top - pad + "px"; s.width = r.width + pad * 2 + "px"; s.height = r.height + pad * 2 + "px";
    const c = els.card, cw = Math.min(320, vw - 24);
    c.style.width = cw + "px";
    const ch = c.offsetHeight, gap = 14, m = 12;
    const room = {below: vh - r.bottom - pad - gap - m, above: r.top - pad - gap - m, right: vw - r.right - pad - gap - m, left: r.left - pad - gap - m};
    const need = {below: ch, above: ch, right: cw, left: cw};
    const order = vw < 640 ? ["below", "above"] : ["below", "above", "right", "left"];
    let side = order.find((k) => room[k] >= need[k]);
    const fits = !!side;
    if (!side) side = order.reduce((a, k) => (room[k] > room[a] ? k : a), order[0]);
    let left, top;
    if (side === "below") { top = r.bottom + pad + gap; left = r.left + r.width / 2 - cw / 2; }
    else if (side === "above") { top = r.top - pad - gap - ch; left = r.left + r.width / 2 - cw / 2; }
    else if (side === "right") { left = r.right + pad + gap; top = r.top + r.height / 2 - ch / 2; }
    else { left = r.left - pad - gap - cw; top = r.top + r.height / 2 - ch / 2; }
    c.dataset.covers = fits ? "no" : "yes";
    c.style.left = Math.max(m, Math.min(left, vw - cw - m)) + "px";
    c.style.top = Math.max(m, Math.min(top, vh - ch - m)) + "px";
  }

  function show() {
    const step = steps[idx];
    els.count.textContent = (idx + 1) + " of " + steps.length;
    els.title.textContent = step.title;
    els.text.textContent = step.text;
    els.back.hidden = !steps.slice(0, idx).some((s) => targetOf(s));
    els.next.textContent = steps.slice(idx + 1).some((s) => targetOf(s)) ? "Next" : "Done";
    const t = targetOf(step), r = rectOf(t);
    if (r.left < 0 || r.right > innerWidth) t[0].scrollIntoView({block: "nearest", inline: "center", behavior: "auto"});  // phone nav scrolls sideways
    if (r.top < 60 || r.bottom > innerHeight - 20) {
      // Tall elements line up under the header so the card has the room below; others are centered.
      const tall = r.height > innerHeight * 0.4;
      t[0].scrollIntoView({block: tall ? "start" : "center", behavior: "auto"});
      if (tall) scrollBy(0, -64);
    }
    place();
    if (els.card.dataset.covers === "yes") {
      // The card would sit on the target: bring the target up under the header so the card fits below it.
      const top = rectOf(t).top;
      if (top > 70) { scrollBy(0, top - 70); place(); }
    }
    requestAnimationFrame(place);
    els.next.focus({preventScroll: true});
  }

  function go(d) {
    let n = idx + d;
    while (n >= 0 && n < steps.length && !targetOf(steps[n])) n += d;
    if (n < 0) return;
    if (n >= steps.length) { end(); return; }
    idx = n; show();
  }

  function end() {
    if (parts.includes("public")) set(SEEN);
    if (parts.includes("admin")) set(ADMIN_SEEN);
    delete root.dataset.tour;
    removeEventListener("resize", place); removeEventListener("scroll", place, true);
    removeEventListener("keydown", onKey, true);
    els.wrap.remove();
    if (savedFocus && savedFocus.focus) { try { savedFocus.focus({preventScroll: true}); } catch { /* gone */ } }
  }

  function onKey(e) {
    if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); end(); }
    else if (e.key === "ArrowRight") { e.preventDefault(); go(1); }
    else if (e.key === "ArrowLeft") { e.preventDefault(); go(-1); }
    else if (e.key === "Tab") {
      const f = [els.skip, els.back, els.next].filter((b) => !b.hidden);
      const i = f.indexOf(document.activeElement);
      e.preventDefault();
      f[(i + (e.shiftKey ? f.length - 1 : 1)) % f.length].focus();
    }
  }

  function start(which) {
    parts = which;
    steps = STEPS.filter((s) => which.includes(s.part));
    savedFocus = document.activeElement;
    idx = steps.findIndex((s) => targetOf(s));
    if (idx < 0) return;
    root.dataset.tour = "running";
    build();
    addEventListener("resize", place); addEventListener("scroll", place, true);
    addEventListener("keydown", onKey, true);
    show();
  }

  // Wait out an open dialog (the weekly photo reminder, a card), up to 8 s; give up quietly for this load.
  function whenClear(fn) {
    let tries = 0;
    const tick = () => {
      if (document.querySelector("dialog[open]")) { if (++tries <= 16) setTimeout(tick, 500); return; }
      fn();
    };
    tick();
  }

  addEventListener("DOMContentLoaded", () => {
    if (document.body.classList.contains("embed")) return;
    const admin = document.body.dataset.admin === "1";
    const all = admin ? ["public", "admin"] : ["public"];
    const home = !!document.querySelector(".hero");
    if (home && new URLSearchParams(location.search).get("tour") === "1") {
      history.replaceState(null, "", location.pathname + location.hash);
      setTimeout(() => whenClear(() => start(all)), 500);
      return;
    }
    if (!storageOk()) return;  // can't remember it, so never start by itself
    const need = [];
    if (home && !get(SEEN)) need.push("public");
    let fromLogin = false;
    try { fromLogin = sessionStorage.getItem(LOGIN) === "1"; if (admin) sessionStorage.removeItem(LOGIN); } catch { /* ignore */ }
    if (admin && !get(ADMIN_SEEN) && (home || fromLogin)) need.push("admin");
    if (need.length) setTimeout(() => whenClear(() => start(need)), 700);
  });
})();
