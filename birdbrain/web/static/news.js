// One-time "what's new" pop-up. base.html renders the <dialog> only while the notice is current (birdbrain/web/news.py:
// never on the public mirror, never after its last day). Closing it any way (Got it, the link, Escape, a click
// beside it) is remembered in this browser (bb_news_<id>), and it never shows again there. The tour's step about the
// same news sets the same flag, so a first-time visitor gets the tour and not this too. If storage is blocked it
// never shows: it could not be dismissed for good. Stays out of the way of the tour and any other open dialog.
(() => {
  const dlg = document.getElementById("news-popup");
  if (!dlg) return;
  const KEY = "bb_news_" + dlg.dataset.news;
  const seen = () => { try { return localStorage.getItem(KEY) !== null; } catch { return true; } };
  const remember = () => { try { localStorage.setItem(KEY, "1"); } catch { /* nothing to do */ } };
  const storageOk = () => { try { localStorage.setItem("bb_news_probe", "1"); localStorage.removeItem("bb_news_probe"); return true; } catch { return false; } };
  // A page kept open or cached past the last day: the date is checked here too.
  const expired = () => Date.now() > new Date(dlg.dataset.until + "T23:59:59").getTime();
  const touring = () => document.documentElement.dataset.tour === "running";
  if (seen() || expired() || !storageOk()) return;

  dlg.querySelector(".popup-cancel").addEventListener("click", () => dlg.close());
  dlg.querySelector(".popup-go").addEventListener("click", remember);
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener("close", remember);

  let tries = 0;
  const show = () => {
    if (seen() || touring()) return;  // the tour tells the same news; if it was skipped, the next page shows this
    if (document.querySelector("dialog[open]")) { if (++tries <= 16) setTimeout(show, 1000); return; }
    dlg.showModal();
    dlg.querySelector(".popup-cancel").focus();
  };
  setTimeout(show, 1200);
})();
