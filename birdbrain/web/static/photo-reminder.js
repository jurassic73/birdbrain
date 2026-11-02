// Weekly reminder (owner only): species with 2+ photos and no favourite photo yet. Closing it any way means
// "remind me next week". Skipped while the first-visit tour runs (html[data-tour="running"] or body.tour-running).
(() => {
  const KEY = "bb_photo_reminder", WEEK = 7 * 24 * 3600 * 1000;
  let shownThisPage = false;
  const last = () => { try { return Number(localStorage.getItem(KEY)) || 0; } catch { return null; } };
  const remember = () => { try { localStorage.setItem(KEY, String(Date.now())); } catch { /* once per page then */ } };
  const touring = () => document.documentElement.dataset.tour === "running" || document.body.classList.contains("tour-running");
  const href = (name) => "/species/" + encodeURIComponent(name) + "#visits";  // the cards with the stars
  const link = (name) => {
    const a = document.createElement("a");
    a.href = href(name);
    a.textContent = name;
    return a;
  };

  function show(names) {
    const dlg = document.createElement("dialog");
    dlg.className = "popup popup-ask";
    dlg.setAttribute("aria-labelledby", "photo-reminder-title");
    const n = names.length, first = names[0], rest = names.slice(1, 6);
    const title = document.createElement("h2");
    title.className = "popup-title";
    title.id = "photo-reminder-title";
    title.textContent = n === 1 ? "1 species has no favourite photo yet" : n + " species have no favourite photo yet";
    const msg = document.createElement("div");
    msg.className = "popup-msg";
    const p = document.createElement("p");
    p.append("Open ", link(first), " and tap the star on the photo you like best.");
    msg.append(p);
    if (rest.length) {
      const q = document.createElement("p");
      q.append("Also waiting: ");
      rest.forEach((r, i) => q.append(i ? ", " : "", link(r)));
      if (n - 1 > rest.length) q.append(" and " + (n - 1 - rest.length) + " more");
      msg.append(q);
    }
    const actions = document.createElement("div");
    actions.className = "popup-actions";
    const later = document.createElement("button");
    later.type = "button";
    later.className = "popup-cancel";
    later.textContent = "Remind me next week";
    const go = document.createElement("a");
    go.className = "popup-go";
    go.href = href(first);
    go.textContent = "Go to " + first + " page";
    actions.append(later, go);
    dlg.append(title, msg, actions);
    document.body.append(dlg);
    later.addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
    dlg.addEventListener("close", () => { remember(); dlg.remove(); });
    dlg.showModal();
    go.focus();
  }

  async function run() {
    const t = last();
    if (shownThisPage || (t !== null && Date.now() - t < WEEK) || touring()) return;
    try {
      const r = await fetch("/api/photo-reminder", { cache: "no-store" });
      if (!r.ok) return;
      const names = (await r.json()).species || [];
      if (!names.length || shownThisPage || touring() || document.querySelector("dialog[open]")) return;
      shownThisPage = true;
      show(names);
    } catch { /* offline: try next visit */ }
  }
  setTimeout(run, 1500);
})();
