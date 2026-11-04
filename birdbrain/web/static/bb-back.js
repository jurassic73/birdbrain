// Back closes an overlay first. An overlay (the photo viewer, the species card) calls
//   const h = bbBack.hold(closeFn)   when it opens: pushes one history entry (same URL), and
//   h.release()                      when it closes by its own controls: pops that entry again.
// The browser's Back (a phone's Back button or gesture) then runs closeFn and stays on the page.
// Stepping inside an overlay must not call hold() again. Not used inside the embedded pop-up (?embed=1).
(() => {
  const embedded = new URLSearchParams(location.search).get("embed") === "1";
  const open = [];       // holds not yet closed, newest last
  let skip = 0;          // popstates that come from our own history.back()
  let seq = 0;

  addEventListener("popstate", () => {
    if (skip) { skip--; return; }
    const h = open.pop();
    if (h) { h.live = false; h.close(); }
  });

  window.bbBack = {
    hold(close) {
      const h = {
        id: ++seq, live: true, close,
        release() {
          if (!h.live) return;                 // Back already closed it
          h.live = false;
          const at = open.indexOf(h);
          if (at >= 0) open.splice(at, 1);
          // pop our entry, but only when it is still the current one (no dead entries left behind)
          if (!embedded && history.state && history.state.bbBack === h.id) { skip++; history.back(); }
        },
      };
      if (embedded) return h;
      history.pushState({ bbBack: h.id }, "");
      open.push(h);
      return h;
    },
  };
})();
