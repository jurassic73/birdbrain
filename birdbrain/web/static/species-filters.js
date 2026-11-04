// Species page, Visits tab: the quick filter chips and the More filters form (Apply, Clear) swap the filtered parts in
// place (no page load, so no jump) and keep the URL in step (#visits keeps the tab), so Back/Forward and a shared link
// land on the same filters. Same idea as the home page's script; anything that goes wrong falls back to a page load.
(() => {
  const panel = document.getElementById("panel-visits"), form = document.getElementById("more-filters");
  const btn = panel?.querySelector(".filters-toggle");
  if (!panel || !form) return;

  // More filters: the button after the chips shows or hides the filter boxes under the row.
  btn?.addEventListener("click", () => {
    const open = form.hidden;
    form.hidden = !open;
    btn.setAttribute("aria-expanded", String(open));
    if (open) form.querySelector("select, input")?.focus({preventScroll: true});
  });

  let token = 0;
  const chipsTop = () => panel.querySelector(".chips")?.getBoundingClientRect().top;
  const keepTab = (url) => url.replace(/#.*$/, "") + "#visits";

  async function show(url, push, chip, busy = chip) {
    const mine = ++token, y = scrollY, top = chipsTop();
    busy?.classList.add("sending");
    if (chip) { panel.querySelectorAll(".chips a").forEach(a => a.removeAttribute("aria-current")); chip.setAttribute("aria-current", "page"); }
    try {
      const r = await fetch(url.replace(/#.*$/, ""), { cache: "no-store", signal: AbortSignal.timeout(10000) });
      if (!r.ok) throw new Error(r.status);
      const doc = new DOMParser().parseFromString(await r.text(), "text/html");
      if (mine !== token) return;  // a later tap won
      const fresh = doc.getElementById("panel-visits"), freshForm = doc.getElementById("more-filters");
      if (!fresh || !freshForm) throw new Error("page shape");
      // The chips, the count, the tab's count and the grid (or empty note) are replaced; the open or closed More
      // filters form keeps its state and takes the new values.
      panel.querySelector(".visits-head .chips").replaceWith(fresh.querySelector(".visits-head .chips"));
      panel.querySelector(".grid-count").replaceWith(fresh.querySelector(".grid-count"));
      const tabCount = document.querySelector("#tab-visits .tab-count"), freshCount = doc.querySelector("#tab-visits .tab-count");
      if (tabCount && freshCount) tabCount.replaceWith(freshCount);
      form.innerHTML = freshForm.innerHTML;
      while (form.nextElementSibling) form.nextElementSibling.remove();
      for (let n = freshForm.nextElementSibling; n; n = n.nextElementSibling) panel.append(n.cloneNode(true));
      if (fresh.hasAttribute("data-filtered")) panel.setAttribute("data-filtered", ""); else panel.removeAttribute("data-filtered");
      const g = panel.querySelector(".grid");
      if (g) { window.htmx?.process(g); g.dispatchEvent(new CustomEvent("birdbrain:cards", { bubbles: true })); }
      if (push) history.pushState(null, "", keepTab(url));
      // Keep the chips row where it was on screen. The page scrolls smoothly (style.css), so say "instant".
      const now = chipsTop();
      if (top === undefined || now === undefined) scrollTo({ top: y, behavior: "instant" });
      else if (Math.abs(now - top) >= 0.5) scrollBy({ top: now - top, behavior: "instant" });
    } catch (e) {
      if (mine === token) location.href = keepTab(url);
    } finally { busy?.classList.remove("sending"); }
  }

  const plain = e => !e.defaultPrevented && !e.button && !e.metaKey && !e.ctrlKey && !e.shiftKey && !e.altKey;

  document.addEventListener("submit", e => {
    const f = e.target;
    if (f.id !== "more-filters" || e.defaultPrevented) return;
    e.preventDefault();
    const q = new URLSearchParams();
    for (const [k, v] of new FormData(f)) if (v) q.append(k, v);
    show(`${f.getAttribute("action")}${q.size ? "?" + q : ""}`, true, null, f.querySelector("button[type=submit]"));
  });
  document.addEventListener("click", e => {
    const link = e.target.closest?.("#more-filters a.clear, #panel-visits .empty a");
    if (link && plain(e)) { e.preventDefault(); show(link.getAttribute("href"), true, null, null); return; }
    const a = e.target.closest?.("#panel-visits .chips a");
    if (!a || !plain(e) || a.target) return;
    e.preventDefault();
    if (a.getAttribute("aria-current") === "page") return;
    show(a.getAttribute("href"), true, a);
  });
  addEventListener("popstate", () => { if (location.hash === "#visits") show(location.pathname + location.search, false, null); });
})();
