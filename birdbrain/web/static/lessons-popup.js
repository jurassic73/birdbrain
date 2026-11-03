// Review pop-up for a Lessons summary Claude drafted on its own. base.html renders the <dialog> only for an admin
// with such a draft waiting (never for visitors, the public mirror, or on Claude's Notes itself). "Later" (or
// "Review it", or Escape) hides it in this browser until tomorrow; a newer draft shows again at once, and an
// accepted or discarded draft isn't rendered at all. Waits out the tour and any other open dialog.
(() => {
  const dlg = document.getElementById("lessons-popup");
  if (!dlg) return;
  const KEY = "bb_lessons_later", DAY = 24 * 3600 * 1000, at = dlg.dataset.at;
  const later = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch { return {}; } };
  const remember = () => { try { localStorage.setItem(KEY, JSON.stringify({ at, until: Date.now() + DAY })); } catch { /* once per page then */ } };
  const touring = () => document.documentElement.dataset.tour === "running" || document.body.classList.contains("tour-running");
  const l = later();
  if (l.at === at && Date.now() < Number(l.until)) return;

  dlg.querySelector(".popup-cancel").addEventListener("click", () => dlg.close());
  dlg.querySelector(".popup-go").addEventListener("click", remember);
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener("close", remember);

  let tries = 0;
  const show = () => {
    if (touring() || document.querySelector("dialog[open]")) { if (++tries <= 16) setTimeout(show, 1000); return; }
    dlg.showModal();
    dlg.querySelector(".popup-go").focus();
  };
  setTimeout(show, 1200);
})();
