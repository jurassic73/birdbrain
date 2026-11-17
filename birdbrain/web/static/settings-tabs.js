// Settings: one section at a time, as tabs (Zones, Camera, Pi checks, Storage, Notifications, Public mirror). The tab
// picked lives in the URL hash (#storage), so /settings#zones links and the redirect after a Save land on the right
// tab; with no hash, the last tab used on this browser, else the first. Which panel shows is plain CSS on
// <html data-settings-show> (settings.html), set here before the panels are drawn, so nothing flashes. Charts and the
// zone editor measure their box: one resize after a panel shows makes them redraw at full size (as species-tabs.js).
(() => {
  const tabs = [...document.querySelectorAll("[data-settings-tab]")];
  if (!tabs.length) return;
  const ids = tabs.map((t) => t.dataset.settingsTab);
  const KEY = "settings-tab";

  function show(id, focus) {
    if (!ids.includes(id)) return false;
    document.documentElement.dataset.settingsShow = id;
    for (const t of tabs) {
      const on = t.dataset.settingsTab === id;
      t.setAttribute("aria-selected", on);
      t.tabIndex = on ? 0 : -1;
      if (on && focus) t.focus();
    }
    try { localStorage.setItem(KEY, id); } catch {}
    if (typeof Chart !== "undefined") Object.values(Chart.instances).forEach((c) => c.resize());
    window.dispatchEvent(new Event("resize"));
    return true;
  }

  // The hash names a section, or something inside one (#presence is in Pi checks): that section's id, else "".
  function fromHash() {
    let id = "";
    try { id = decodeURIComponent(location.hash.slice(1)); } catch {}
    if (!id || ids.includes(id)) return id;
    return document.getElementById(id)?.closest(".settings-panel")?.id || "";
  }

  let last = "";
  try { last = localStorage.getItem(KEY) || ""; } catch {}
  show([location.hash.slice(1), last, ids[0]].find((id) => ids.includes(id)));

  tabs.forEach((t) => {
    t.addEventListener("click", () => {
      show(t.dataset.settingsTab);
      history.replaceState(null, "", "#" + t.dataset.settingsTab);
    });
    t.addEventListener("keydown", (e) => {
      const to = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
      if (to === undefined) return;
      e.preventDefault();
      const next = tabs[(tabs.indexOf(t) + to + tabs.length) % tabs.length];
      show(next.dataset.settingsTab, true);
      history.replaceState(null, "", "#" + next.dataset.settingsTab);
    });
  });

  window.addEventListener("hashchange", () => show(fromHash()));
  // A link to something inside a section (#presence) can only be followed once the panels exist.
  document.addEventListener("DOMContentLoaded", () => {
    const id = fromHash();
    if (!id || id === location.hash.slice(1)) return;
    show(id);
    document.getElementById(location.hash.slice(1))?.scrollIntoView();
  });
})();
