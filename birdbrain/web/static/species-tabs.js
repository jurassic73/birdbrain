// Species page: the Info / Visits tabs. The choice lives in the URL hash (#info, #visits) so a reload and old
// #visits links land on the right tab. Charts and maps drawn while their panel was hidden are resized on show.
(() => {
  const tabs = [...document.querySelectorAll("[data-sp-tab]")];
  if (!tabs.length) return;

  function show(name, focus) {
    if (!tabs.some((t) => t.dataset.spTab === name)) name = "info";
    for (const t of tabs) {
      const on = t.dataset.spTab === name;
      t.setAttribute("aria-selected", on);
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
      if (on && focus) t.focus();
    }
    // The expand button (phones, end of the tab row) opens the visit photos: dimmed and inert on Info.
    const expand = document.querySelector(".sp-tabs .lb-tab");
    if (expand) {
      const off = name !== "visits";
      expand.setAttribute("aria-disabled", off);
      expand.tabIndex = off ? -1 : 0;
    }
    // Charts and Leaflet maps measure their box; one resize after the panel shows makes them redraw at full size.
    if (typeof Chart !== "undefined") Object.values(Chart.instances).forEach((c) => c.resize());
    window.dispatchEvent(new Event("resize"));
  }

  const fromHash = () => location.hash.slice(1);

  tabs.forEach((t) => {
    t.addEventListener("click", () => {
      show(t.dataset.spTab);
      history.replaceState(null, "", "#" + t.dataset.spTab);
    });
    t.addEventListener("keydown", (e) => {
      const to = { ArrowLeft: -1, ArrowRight: 1 }[e.key];
      if (to === undefined) return;
      e.preventDefault();
      const next = tabs[(tabs.indexOf(t) + to + tabs.length) % tabs.length];
      show(next.dataset.spTab, true);
      history.replaceState(null, "", "#" + next.dataset.spTab);
    });
  });

  window.addEventListener("hashchange", () => show(fromHash()));
  if (fromHash()) show(fromHash());
})();
