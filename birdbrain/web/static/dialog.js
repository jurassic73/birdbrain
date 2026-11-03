// Site-styled questions in place of the browser's own confirm box: the one way any page asks "are you sure?".
// Looks: the Pop-ups section of style.css (.popup, .popup-ask, .popup-actions).
//
//   birdbrain.ask({ title, message, ok: "Delete", cancel: "Cancel", danger: true }) -> Promise<boolean>
//     message: blank lines start a new paragraph. Escape, Cancel or a click on the dimmed page answer no.
//   hx-confirm="..."      on any htmx element goes through it (data-confirm-title / data-confirm-ok optional;
//                         a button.danger gets a red OK).
//   <form data-confirm="..."> the same for plain forms (title/ok attributes as above).
//   birdbrain.confirmSubmit(event, opts) from a form's own submit handler when the question is built at submit time;
//     returns true when the submit is going ahead (the second, confirmed pass).
(() => {
  const dlg = document.createElement("dialog");
  dlg.className = "popup popup-ask";
  dlg.setAttribute("aria-labelledby", "popup-ask-title");
  dlg.innerHTML = '<h2 class="popup-title" id="popup-ask-title"></h2><div class="popup-msg"></div>'
    + '<div class="popup-actions"><button type="button" class="popup-cancel"></button>'
    + '<button type="button" class="popup-ok"></button></div>';
  const title = dlg.querySelector(".popup-title"), msg = dlg.querySelector(".popup-msg");
  const okBtn = dlg.querySelector(".popup-ok"), cancelBtn = dlg.querySelector(".popup-cancel");
  let answer = null;  // the open question's resolve

  const done = (yes) => {
    const r = answer;
    answer = null;
    if (dlg.open) dlg.close();
    r?.(yes);
  };
  okBtn.addEventListener("click", () => done(true));
  cancelBtn.addEventListener("click", () => done(false));
  dlg.addEventListener("close", () => { if (!dlg.open) done(false); });  // Escape (not a newer question reopening it)
  dlg.addEventListener("click", (e) => { if (e.target === dlg) done(false); });  // the dimmed page

  function ask({ title: t = "", message = "", ok = "OK", cancel = "Cancel", danger = false } = {}) {
    if (answer) done(false);  // one question at a time
    if (!dlg.isConnected) document.body.append(dlg);
    title.textContent = t;
    title.hidden = !t;
    msg.replaceChildren(...String(message).split(/\n\s*\n/).filter((s) => s.trim()).map((s) => {
      const p = document.createElement("p");
      p.textContent = s.trim();
      return p;
    }));
    if (t) dlg.setAttribute("aria-labelledby", "popup-ask-title");
    else dlg.removeAttribute("aria-labelledby");
    dlg.setAttribute("aria-label", t || msg.textContent.slice(0, 80));
    okBtn.textContent = ok;
    okBtn.classList.toggle("danger", danger);
    cancelBtn.textContent = cancel;
    return new Promise((resolve) => {
      answer = resolve;
      dlg.showModal();
      (danger ? cancelBtn : okBtn).focus();  // Enter never deletes by accident
    });
  }

  const optsFrom = (el, question) => ({
    title: el.dataset.confirmTitle || "",
    message: question,
    ok: el.dataset.confirmOk || "OK",
    danger: el.classList.contains("danger") || el.querySelector?.("button.danger[type=submit], button.danger:not([type])") != null,
  });

  function confirmSubmit(e, opts) {
    const form = e.target;
    if (form.dataset.confirmed) {
      delete form.dataset.confirmed;
      return true;
    }
    e.preventDefault();
    const submitter = e.submitter;
    ask(opts).then((yes) => {
      if (!yes) return;
      form.dataset.confirmed = "1";
      form.requestSubmit(submitter || null);
    });
    return false;
  }

  // htmx asks the page before sending; answer it here instead of with window.confirm.
  document.addEventListener("htmx:confirm", (e) => {
    const question = e.detail.question;
    if (!question) return;
    e.preventDefault();
    ask(optsFrom(e.detail.elt, question)).then((yes) => { if (yes) e.detail.issueRequest(true); });
  });

  // Plain forms with data-confirm. Loaded before save-feedback.js, so its busy gradient waits for the answer.
  document.addEventListener("submit", (e) => {
    const form = e.target;
    if (e.defaultPrevented || !form.matches?.("form[data-confirm]")) return;
    confirmSubmit(e, optsFrom(form, form.dataset.confirm));
  });

  window.birdbrain = Object.assign(window.birdbrain || {}, { ask, confirmSubmit });
})();
