// Save buttons confirm right where you clicked: the button flashes and a green check fades in beside it.
// Settings forms save in the background so the page keeps its scroll position instead of jumping to a notice.
(() => {
  const isSave = (b) => b && b.textContent.trim() === "Save";

  function confirmSaved(btn) {
    if (!btn) return;
    btn.classList.remove("save-flash");
    void btn.offsetWidth; // restart the animation on repeat saves
    btn.classList.add("save-flash");
    btn.parentElement.querySelector(":scope > .save-check")?.remove();
    const check = document.createElement("span");
    check.className = "save-check";
    check.setAttribute("role", "status");
    check.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>Saved';
    btn.after(check);
    setTimeout(() => check.classList.add("gone"), 2600);
    setTimeout(() => check.remove(), 3300);
  }

  // Refresh one settings card from the server's page, minus the "Saved." notice the check replaces.
  function refreshCard(card, html) {
    const fresh = new DOMParser().parseFromString(html, "text/html").getElementById(card.id);
    if (!fresh) return;
    fresh.querySelectorAll(".notice.saved").forEach((n) => { if (/^Saved/.test(n.textContent.trim())) n.remove(); });
    const hint = card.querySelector("summary .summary-hint"), newHint = fresh.querySelector("summary .summary-hint");
    if (hint && newHint) hint.textContent = newHint.textContent;
    const body = card.querySelector(".settings-body"), newBody = fresh.querySelector(".settings-body");
    // Bodies with inline scripts keep their live DOM (the fields already show what was saved).
    if (body && newBody && !newBody.querySelector("script")) {
      body.innerHTML = newBody.innerHTML;
      window.htmx?.process(body);
    }
  }

  document.addEventListener("submit", async (e) => {
    const form = e.target;
    if (!form.matches("form.settings, form.settings-form") || form.matches("[hx-post]")) return;  // htmx sends those
    const btn = e.submitter;
    const card = form.closest("details.settings-card") || form.querySelector("details.settings-card");
    if (!card) return;
    btn?.classList.add("sending"); // the slow gradient until the server answers
    e.preventDefault();
    const save = isSave(btn);
    const saveIndex = [...card.querySelectorAll("button")].indexOf(btn);
    try {
      const r = await fetch(btn?.getAttribute("formaction") || form.action,
                            { method: "POST", body: new URLSearchParams(new FormData(form, btn)) });
      if (!r.ok) throw new Error(r.status);
      refreshCard(card, await r.text());
      // The card may have been rebuilt; find the same button again by position.
      if (save) confirmSaved(btn.isConnected ? btn : card.querySelectorAll("button")[saveIndex]);
    } catch {
      form.submit(); // fall back to a normal page load
    } finally {
      btn?.classList.remove("sending");
    }
  });

  // Plain forms that load a new page: keep the button busy until it does.
  document.addEventListener("submit", (e) => {
    if (!e.defaultPrevented) e.submitter?.classList.add("sending");
  });
  window.addEventListener("pageshow", () => document.querySelectorAll("button.sending").forEach((b) => b.classList.remove("sending")));

  // htmx forms (visit page): show the check, then do the page refresh the server asked for.
  document.addEventListener("htmx:beforeOnLoad", (e) => {
    const { elt, xhr } = e.detail;
    const btn = elt.matches?.("form") && elt.querySelector('button[type="submit"]');
    if (!isSave(btn) || xhr.status >= 300 || xhr.getResponseHeader("HX-Refresh") !== "true") return;
    e.preventDefault();
    confirmSaved(btn);
    setTimeout(() => location.reload(), 1400);
  });
})();
