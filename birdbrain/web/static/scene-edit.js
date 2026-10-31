// Adjust Claude's scene outline by hand. Click anywhere on the photo to open it big over the page; click a shape (or
// Adjust in the list) to pick it. Ellipses get a handle on each side (drag to resize); a birdbath ring also gets handles
// for the water inside; polygons get a handle on every corner. Drag the middle icon, or the picked shape itself, to
// move it. The Shape switch turns it into points or an ellipse; the chips hide or show shapes. On points, the faded
// dot midway along each edge adds a corner there (drag or click it); double-click a corner, or click it and press
// Delete, to remove it (3 corners at least).
// Undo / Redo (or Cmd/Ctrl+Z, Shift+Cmd+Z, Ctrl+Y) step back and forward through the changes made since the last Save.
// Changes stay on the page until Save; closing (Done, Escape, a click outside the photo) asks to save them.
// Shapes are frame fractions, drawn in the SVG's viewBox (W x H), the same as the server's scene_shape.
// Settings > Zones uses it too: its svg has data-save (post every shape there at once as {objects}), data-panel and
// data-panel-url (what to reload after), and data-fixed (shapes keep their form: no Ellipse / Points switch).
(() => {
  const NS = "http://www.w3.org/2000/svg";
  const KEYS = ["ellipse", "inner", "points", "pieces"];
  const CORNERS = 8; // an ellipse switched to points; matches scene.CORNERS
  let lastDown = { kind: null, t: 0 }; // the last handle pressed, for a double press that removes a corner
  let sel = null; // the open big view: { svg, which, i, obj, pending: Set of changed shape indexes }

  const box = (svg) => svg.viewBox.baseVal;
  const objs = (svg) => JSON.parse(svg.dataset.objs || "[]");
  const geometry = (o) => Object.fromEntries(KEYS.filter((k) => o[k]).map((k) => [k, o[k]]));

  function ellipsePath(svg, [cx, cy, rx, ry]) {
    const { width: W, height: H } = box(svg);
    const x = cx * W, y = cy * H, a = rx * W, b = ry * H;
    return `M${x - a},${y} a${a},${b} 0 1,0 ${2 * a},0 a${a},${b} 0 1,0 ${-2 * a},0 Z`;
  }
  function polyPath(svg, pts) {
    const { width: W, height: H } = box(svg);
    return "M" + pts.map(([x, y]) => `${x * W},${y * H}`).join(" L") + " Z";
  }
  function shapePath(svg, o) {
    if (o.ellipse) return ellipsePath(svg, o.ellipse) + (o.inner ? " " + ellipsePath(svg, o.inner) : "");
    return [o.points, ...(o.pieces || [])].map((p) => polyPath(svg, p)).join(" ");
  }
  function labelAt(o) {
    if (o.ellipse) return [o.ellipse[0], Math.max(o.ellipse[1] - o.ellipse[3], 0.04)];
    const top = o.points.reduce((a, b) => (b[1] < a[1] ? b : a));
    return [top[0], Math.max(top[1], 0.04)];
  }

  // A pointer position as frame fractions.
  function frac(svg, e) {
    const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(svg.getScreenCTM().inverse());
    const { width: W, height: H } = box(svg);
    return [Math.min(Math.max(p.x / W, 0), 1), Math.min(Math.max(p.y / H, 0), 1)];
  }

  // Handles: [kind, x, y] in fractions. Kinds: move, o-x / o-y (outer radius), i-x / i-y (water), p-k-j (corner j of
  // outline k: 0 is points, 1.. are pieces), m-k-j (the add-a-corner dot between corners j and j+1).
  function handles(o) {
    const out = [];
    if (o.ellipse) {
      const [cx, cy, rx, ry] = o.ellipse;
      out.push(["move", cx, cy], ["o-x", cx - rx, cy], ["o-x", cx + rx, cy], ["o-y", cx, cy - ry], ["o-y", cx, cy + ry]);
      if (o.inner) {
        const [ix, iy, irx, iry] = o.inner;
        out.push(["i-x", ix - irx, iy], ["i-x", ix + irx, iy], ["i-y", ix, iy - iry], ["i-y", ix, iy + iry]);
      }
    } else {
      const all = [o.points, ...(o.pieces || [])];
      let n = 0, sx = 0, sy = 0;
      all.forEach((pts, k) => pts.forEach(([x, y], j) => {
        const [nx, ny] = pts[(j + 1) % pts.length];
        out.push([`m-${k}-${j}`, (x + nx) / 2, (y + ny) / 2]);
      }));
      all.forEach((pts, k) => pts.forEach(([x, y], j) => { out.push([`p-${k}-${j}`, x, y]); sx += x; sy += y; n++; }));
      out.unshift(["move", sx / n, sy / n]);
    }
    return out;
  }

  // Redraw shape i's outline, ring click target, water line and label from o (no handles).
  function paint(svg, i, o) {
    const { width: W, height: H } = box(svg);
    svg.querySelector(`path[data-i="${i}"]:not(.scene-hit):not(.scene-water)`)?.setAttribute("d", shapePath(svg, o));
    svg.querySelector(`path.scene-hit[data-i="${i}"]`)?.setAttribute("d", o.inner ? ellipsePath(svg, o.ellipse) : "");
    svg.querySelector(`path.scene-water[data-i="${i}"]`)?.setAttribute("d", o.inner ? ellipsePath(svg, o.inner) : "");
    const label = svg.querySelector(`text[data-i="${i}"]`);
    if (label) { const [lx, ly] = labelAt(o); label.setAttribute("x", lx * W); label.setAttribute("y", ly * H - 1); }
  }

  // Redraw the picked shape (and its ring's click target and water line) and its handles.
  function draw() {
    if (!sel?.obj) return;
    const { svg, i, obj } = sel;
    const { width: W, height: H } = box(svg);
    paint(svg, i, obj);
    let g = svg.querySelector("g.se-handles");
    if (!g) { g = document.createElementNS(NS, "g"); g.setAttribute("class", "se-handles"); svg.appendChild(g); }
    g.replaceChildren();
    // handles stay the same size on screen: bigger for fingers
    const px = svg.getScreenCTM().a || 1, r = (matchMedia("(pointer: coarse)").matches ? 13 : 8) / px;
    for (const [kind, x, y] of handles(obj)) {
      const h = document.createElementNS(NS, kind === "move" ? "g" : "circle");
      h.dataset.kind = kind;
      if (kind === "move") {
        h.setAttribute("class", "se-move");
        h.setAttribute("transform", `translate(${x * W},${y * H}) scale(${r / 10})`);
        // the big invisible circle is the grab area; the small one is the icon
        h.innerHTML = '<circle class="se-grab" r="28"/><circle r="10"/><path d="M0-7v14M-7 0h14M0-7l-2.5 2.5M0-7l2.5 2.5M0 7l-2.5-2.5M0 7l2.5-2.5M-7 0l2.5-2.5M-7 0l2.5 2.5M7 0l-2.5-2.5M7 0l-2.5 2.5"/>';
        g.prepend(h);  // under the side handles, so they stay grabbable on a small shape
        continue;
      }
      const cls = kind.startsWith("i-") ? "se-handle se-inner" : kind.startsWith("m-") ? "se-handle se-mid" : "se-handle";
      h.setAttribute("class", cls + (kind === sel.corner ? " se-active" : ""));
      h.setAttribute("cx", x * W); h.setAttribute("cy", y * H);
      h.setAttribute("r", r * (kind.startsWith("m-") ? 0.5 : kind.startsWith("p-") ? 0.7 : kind.startsWith("i-") ? 0.8 : 1));
      if (kind.startsWith("p-")) h.innerHTML = "<title>Double-click (or press Delete) to remove this corner</title>";
      if (kind.startsWith("m-")) h.innerHTML = "<title>Drag or click to add a corner here</title>";
      g.appendChild(h);
    }
  }

  // Shapes hidden to get at the ones under them: per outline, in this browser only.
  const hiddenKey = (svg) => `scene-hidden:${svg.dataset.which}:${svg.dataset.at}`;
  function hidden(svg) {
    try { return new Set(JSON.parse(localStorage.getItem(hiddenKey(svg)) || "[]")); } catch { return new Set(); }
  }
  function setHidden(svg, i, hide) {
    const h = hidden(svg);
    hide ? h.add(i) : h.delete(i);
    try { localStorage.setItem(hiddenKey(svg), JSON.stringify([...h])); } catch { /* private mode: still hides */ }
    applyHidden(svg);
  }
  // All shapes shown (none) or all hidden.
  function setAllHidden(svg, hide) {
    const all = hide ? objs(svg).map((_, i) => i) : [];
    try { localStorage.setItem(hiddenKey(svg), JSON.stringify(all)); } catch { /* private mode */ }
    applyHidden(svg);
  }

  function applyHidden(svg) {
    const h = hidden(svg);
    svg.querySelectorAll(":scope > [data-i]").forEach((el) => el.classList.toggle("se-hidden", h.has(Number(el.dataset.i))));
    const list = svg.closest(".scene-body")?.querySelector(".scene-list");
    list?.querySelectorAll("li[data-i]").forEach((li) => {
      const off = h.has(Number(li.dataset.i));
      li.classList.toggle("se-off", off);
      const eye = li.querySelector(".scene-eye");
      if (eye) eye.textContent = off ? "Show" : "Hide";
    });
    document.querySelectorAll(".se-chip").forEach((c) => c.classList.toggle("off", h.has(Number(c.dataset.i))));
  }
  const applyAll = () => document.querySelectorAll(".scene-frame svg[data-which]").forEach(applyHidden);

  // --- the big view ---

  function enlarge(svg) {
    const frame = svg.closest(".scene-frame");
    const back = document.createElement("div");
    back.className = "se-backdrop";
    const bar = document.createElement("div");
    bar.className = "se-bar";
    bar.innerHTML = '<p class="se-tip">Click a shape to adjust it: drag a side to resize, the middle or the shape to move. '
      + "On points: drag a faded dot to add a corner, double-click a corner (or press Delete) to remove it. "
      + "Click a name to hide or show it.</p>";
    const chips = document.createElement("div");
    chips.className = "se-chips";
    chips.innerHTML = '<button type="button" class="se-all" data-show="1">All</button>'
      + '<button type="button" class="se-all" data-show="0">None</button>';
    svg.closest(".scene-body")?.querySelectorAll(".scene-list li[data-i]").forEach((li) => {
      const c = document.createElement("button");
      c.type = "button";
      c.className = "se-chip";
      c.dataset.i = li.dataset.i;
      c.innerHTML = `<span style="background:${li.dataset.color}"></span>${Number(li.dataset.i) + 1} ${li.dataset.name}`;
      chips.append(c);
    });
    const tools = document.createElement("div");
    tools.className = "se-tools";
    tools.innerHTML = '<span>Shape:</span><button type="button" class="se-tool" data-to="ellipse">Ellipse</button>'
      + '<button type="button" class="se-tool" data-to="points">Points</button>';
    if (svg.dataset.fixed) tools.style.display = "none";
    bar.append(chips, tools);
    const end = document.createElement("div");
    end.className = "se-end";
    end.innerHTML = '<span class="se-status"></span>'
      + '<button type="button" class="link-btn se-undo" title="Undo (Cmd/Ctrl+Z)" disabled>Undo</button>'
      + '<button type="button" class="link-btn se-redo" title="Redo (Shift+Cmd+Z or Ctrl+Y)" disabled>Redo</button><button type="button" class="se-save" hidden>Save</button>'
      + '<button type="button" class="se-done">Done</button>';
    back.append(bar, end);
    document.body.append(back);
    document.documentElement.style.setProperty("--se-bar", `${bar.offsetHeight + 16}px`);
    frame.classList.add("se-big");
    document.documentElement.classList.add("se-open");
    setAllHidden(svg, false);  // the big view always opens with every shape showing
    updateTools();
  }

  function open(svg) {
    if (sel && sel.svg === svg) return clearPick();
    closeView();
    enlarge(svg);
    sel = { svg, which: svg.dataset.which, i: null, obj: null, pending: new Set() };
    resetHistory();
    svg.classList.add("se-editing");
    if (svg.dataset.which === "zones") window.zonesFreshPhoto?.();  // the live photo, not one from minutes ago
  }

  // Drop the picked shape but keep the big view open.
  function clearPick() {
    if (!sel) return;
    sel.svg.querySelector("g.se-handles")?.remove();
    document.querySelectorAll(".se-selected").forEach((el) => el.classList.remove("se-selected"));
    document.querySelectorAll(".se-chip.current").forEach((c) => c.classList.remove("current"));
    sel.i = null;
    sel.obj = null;
    sel.corner = null;
    updateTools();
  }

  function select(svg, i) {
    const o = objs(svg)[i];
    if (!o) return;
    if (sel && sel.svg === svg) clearPick();  // switching shapes in the big view: keep it open
    else open(svg);
    if (hidden(svg).has(i)) setHidden(svg, i, false);  // picking a hidden shape shows it
    sel.i = i;
    sel.obj = structuredClone(o);
    sel.corner = null;
    document.querySelectorAll(".se-chip").forEach((c) => c.classList.toggle("current", Number(c.dataset.i) === i));
    updateTools();
    svg.querySelector(`path[data-i="${i}"]:not(.scene-hit):not(.scene-water)`)?.classList.add("se-selected");
    svg.closest(".scene-body")?.querySelector(`.scene-list li[data-i="${i}"]`)?.classList.add("se-selected");
    draw();
  }

  // Close the big view as it is (any unsaved changes stay drawn until the panel reloads).
  function closeView() {
    if (!sel) return;
    sel.svg.querySelector("g.se-handles")?.remove();
    sel.svg.classList.remove("se-editing");
    document.querySelectorAll(".se-selected").forEach((el) => el.classList.remove("se-selected"));
    sel = null;
    document.querySelector(".se-backdrop")?.remove();
    document.querySelector(".se-ask")?.remove();
    document.querySelectorAll(".scene-frame.se-big").forEach((f) => f.classList.remove("se-big"));
    document.documentElement.classList.remove("se-open");
  }

  // Done, Escape or a click outside: ask first if anything changed.
  function requestClose() {
    if (!sel) return;
    if (document.querySelector(".se-ask")) return;
    if (!sel.pending.size) {  // nothing unsaved; if something was saved, the list needs its "adjusted" marks
      const { saved, svg } = sel;
      closeView();
      if (saved) reloadPanel(svg);
      return;
    }
    const n = sel.pending.size;
    const ask = document.createElement("div");
    ask.className = "se-ask";
    ask.setAttribute("role", "dialog");
    ask.innerHTML = `<p>You changed ${n} shape${n === 1 ? "" : "s"}. Save before closing?</p>`
      + '<div class="actions"><button type="button" class="se-ask-save">Save</button>'
      + '<button type="button" class="danger se-ask-discard">Discard</button>'
      + '<button type="button" class="link-btn se-ask-keep">Keep editing</button></div>';
    document.body.append(ask);  // on top of the big photo (the backdrop sits under it)
    ask.querySelector(".se-ask-save").focus();
  }

  // The Ellipse / Points switch is always there, dimmed until a shape is picked; the picked shape's form is marked.
  function updateTools() {
    const tools = document.querySelector(".se-tools");
    if (!tools) return;
    tools.classList.toggle("se-dim", !sel?.obj);
    tools.querySelectorAll(".se-tool").forEach((b) => {
      b.disabled = !sel?.obj;
      b.title = sel?.obj ? "" : "Click a shape first";
      b.classList.toggle("current", !!sel?.obj && (b.dataset.to === "ellipse") === !!sel.obj.ellipse);
    });
  }
  function updateStatus() {
    const n = sel?.pending.size || 0;
    const status = document.querySelector(".se-status"), save = document.querySelector(".se-save");
    if (status) status.textContent = n ? `${n} unsaved change${n === 1 ? "" : "s"}` : "";
    if (save) save.hidden = !n;
    const undo = document.querySelector(".se-undo"), redo = document.querySelector(".se-redo");
    if (undo) undo.disabled = !sel || sel.pos <= 0;
    if (redo) redo.disabled = !sel || sel.pos >= sel.hist.length - 1;
  }

  // Change history since the open (or the last Save): snapshots of every shape and which ones are unsaved.
  const snap = () => ({ objs: sel.svg.dataset.objs || "[]", pending: [...sel.pending] });
  function resetHistory() {
    sel.hist = [snap()];
    sel.pos = 0;
    updateStatus();
  }
  function record() {
    sel.hist.length = sel.pos + 1;  // a new change drops the steps that were undone
    sel.hist.push(snap());
    sel.pos++;
  }
  function stepTo(pos) {
    if (!sel || document.querySelector(".se-ask") || pos < 0 || pos >= sel.hist.length) return;
    sel.pos = pos;
    const { objs: json, pending } = sel.hist[pos], svg = sel.svg;
    svg.dataset.objs = json;
    sel.pending = new Set(pending);
    JSON.parse(json).forEach((o, i) => paint(svg, i, o));
    if (sel.i !== null) {
      sel.obj = structuredClone(objs(svg)[sel.i]);
      sel.corner = null;
      draw();
    }
    updateTools();
    updateStatus();
  }

  // Keep the picked shape's new geometry on the page until Save.
  function stage() {
    const { svg, i, obj } = sel;
    const all = objs(svg);
    const o = { ...all[i] };
    for (const k of KEYS) delete o[k];
    all[i] = { ...o, ...geometry(obj) };
    svg.dataset.objs = JSON.stringify(all);
    sel.pending.add(i);
    record();
    updateStatus();
  }

  async function saveAll() {
    const { svg, which, pending } = sel;
    const all = objs(svg);
    if (svg.dataset.save) {  // all at once (Settings > Zones)
      const r = await fetch(svg.dataset.save, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ objects: all }),
      });
      if (!r.ok) {
        const status = document.querySelector(".se-status");
        if (status) status.textContent = `Couldn't save (${r.status}). Try again.`;
        return false;
      }
      pending.clear();
      sel.saved = true;
      resetHistory();
      return true;
    }
    for (const i of [...pending]) {
      const r = await fetch(`/notes/scene/shape/${which}/${i}`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(geometry(all[i])),
      });
      if (!r.ok) {
        const status = document.querySelector(".se-status");
        if (status) status.textContent = `Couldn't save shape ${i + 1} (${r.status}). Try again.`;
        return false;
      }
      pending.delete(i);
      sel.saved = true;
    }
    resetHistory();  // what's saved is the new starting point
    return true;
  }

  // Reload the panel from the server: shows saved marks, or drops discarded changes.
  function reloadPanel(svg) {
    const panel = document.getElementById(svg?.dataset.panel || "scene-panel");
    const url = svg?.dataset.panelUrl || "/notes/scene/panel";
    if (panel && window.htmx) htmx.ajax("GET", url, { target: panel, swap: "outerHTML" });
  }

  async function finish(save) {
    document.querySelector(".se-ask")?.remove();
    if (save && !(await saveAll())) return;  // a failed save keeps the view open
    const svg = sel?.svg;
    if (sel) sel.pending.clear();
    closeView();
    reloadPanel(svg);
  }

  function drag(kind, start, startObj, [x, y]) {
    const o = sel.obj, dx = x - start[0], dy = y - start[1];
    const shift = (e) => [e[0] + dx, e[1] + dy, e[2], e[3]];
    const min = 0.005;
    if (kind === "move") {
      if (startObj.ellipse) {
        o.ellipse = shift(startObj.ellipse);
        if (startObj.inner) o.inner = shift(startObj.inner);
      } else {
        const mv = (pts) => pts.map(([px, py]) => [px + dx, py + dy]);
        o.points = mv(startObj.points);
        if (startObj.pieces) o.pieces = startObj.pieces.map(mv);
      }
    } else if (kind === "o-x") o.ellipse[2] = Math.max(Math.abs(x - o.ellipse[0]), min);
    else if (kind === "o-y") o.ellipse[3] = Math.max(Math.abs(y - o.ellipse[1]), min);
    else if (kind === "i-x") o.inner[2] = Math.max(Math.abs(x - o.inner[0]), min);
    else if (kind === "i-y") o.inner[3] = Math.max(Math.abs(y - o.inner[1]), min);
    else if (kind.startsWith("p-")) {
      const [, k, j] = kind.split("-").map(Number);
      (k === 0 ? o.points : o.pieces[k - 1])[j] = [x, y];
    }
    draw();
  }

  // Switch the picked shape between an ellipse and points (same math as scene.convert_shape).
  function convert(to) {
    const { svg, i, obj } = sel;
    if ((to === "ellipse") === !!obj.ellipse) return;
    let geom;
    if (to === "points") {
      const [cx, cy, rx, ry] = obj.ellipse;
      geom = { points: Array.from({ length: CORNERS }, (_, k) => {
        const a = (2 * Math.PI * k) / CORNERS;
        return [cx + rx * Math.cos(a), cy + ry * Math.sin(a)];
      }) };
    } else {
      const pts = [obj.points, ...(obj.pieces || [])].flat();
      const xs = pts.map((p) => p[0]), ys = pts.map((p) => p[1]);
      const x1 = Math.min(...xs), x2 = Math.max(...xs), y1 = Math.min(...ys), y2 = Math.max(...ys);
      geom = { ellipse: [(x1 + x2) / 2, (y1 + y2) / 2, Math.max((x2 - x1) / 2, 0.005), Math.max((y2 - y1) / 2, 0.005)] };
    }
    for (const k of KEYS) delete obj[k];
    Object.assign(obj, geom);
    // a ring's click target and water line go with the ellipse
    // (emptied, not removed, so Undo can bring the ring back)
    svg.querySelectorAll(`path.scene-hit[data-i="${i}"], path.scene-water[data-i="${i}"]`).forEach((p) => p.setAttribute("d", ""));
    draw();
    updateTools();
    stage();
  }

  // The polygon (points, or one of the pieces) a corner kind refers to.
  const poly = (o, k) => (k === 0 ? o.points : o.pieces[k - 1]);

  // Remove a corner (an outline keeps 3 at least).
  function removeCorner(kind) {
    if (!sel?.obj || !kind?.startsWith("p-")) return;
    const [, k, j] = kind.split("-").map(Number);
    const pts = poly(sel.obj, k);
    if (!pts || pts.length <= 3) return;
    pts.splice(j, 1);
    sel.corner = null;
    draw();
    stage();
  }

  // Start dragging: a handle, or (kind "move") the picked shape itself. A midpoint dot becomes a new corner first.
  function startDrag(e, kind, target) {
    e.preventDefault();
    const before = structuredClone(sel.obj);
    if (kind.startsWith("m-")) {
      const [, k, j] = kind.split("-").map(Number);
      const pts = poly(sel.obj, k), [x, y] = pts[j], [nx, ny] = pts[(j + 1) % pts.length];
      pts.splice(j + 1, 0, [(x + nx) / 2, (y + ny) / 2]);
      kind = `p-${k}-${j + 1}`;
      draw();
    }
    if (kind.startsWith("p-")) { sel.corner = kind; draw(); }  // the corner Delete would remove
    const svg = sel.svg, start = frac(svg, e), startObj = structuredClone(sel.obj);
    target.setPointerCapture?.(e.pointerId);
    const move = (ev) => drag(kind, start, startObj, frac(svg, ev));
    const up = () => {
      document.removeEventListener("pointermove", move);
      document.removeEventListener("pointerup", up);
      if (sel?.obj && JSON.stringify(before) !== JSON.stringify(sel.obj)) stage();
    };
    document.addEventListener("pointermove", move);
    document.addEventListener("pointerup", up);
  }

  document.addEventListener("pointerdown", (e) => {
    if (e.target.closest?.(".se-ask")) return;
    const handle = e.target.closest?.(".se-handles [data-kind]");
    if (handle && sel?.obj) {
      // two presses on the same corner: remove it (the handles are redrawn on each press, so the browser's own
      // dblclick never sees the same element twice)
      const kind = handle.dataset.kind, now = Date.now();
      if (kind.startsWith("p-") && lastDown.kind === kind && now - lastDown.t < 400) {
        lastDown = { kind: null, t: 0 };
        return removeCorner(kind);
      }
      lastDown = { kind, t: now };
      return startDrag(e, kind, sel.svg);
    }
    const shape = e.target.closest?.(".scene-frame svg[data-which] [data-i]");
    if (shape) {
      const svg = shape.ownerSVGElement, i = Number(shape.dataset.i);
      if (sel?.obj && sel.svg === svg && sel.i === i) return startDrag(e, "move", shape);  // drag the picked shape
      return select(svg, i);
    }
    // anywhere else on the photo: open the big view (or, already open, let go of the picked shape)
    const frameSvg = e.target.closest?.(".scene-frame")?.querySelector("svg[data-which]");
    if (frameSvg) return open(frameSvg);
    if (sel && !e.target.closest?.(".scene-adjust, .scene-eye, .se-bar, .se-end")) requestClose();
  });

  document.addEventListener("click", (e) => {
    if (e.target.closest?.(".se-ask-save")) return finish(true);
    if (e.target.closest?.(".se-ask-discard")) return finish(false);
    if (e.target.closest?.(".se-ask-keep")) return document.querySelector(".se-ask")?.remove();
    if (e.target.closest?.(".se-save")) return sel && saveAll().then(updateStatus);
    if (e.target.closest?.(".se-done")) return requestClose();
    if (e.target.closest?.(".se-undo")) return sel && stepTo(sel.pos - 1);
    if (e.target.closest?.(".se-redo")) return sel && stepTo(sel.pos + 1);
    const tool = e.target.closest?.(".se-tool");
    if (tool && sel?.obj) return convert(tool.dataset.to);
    const all = e.target.closest?.(".se-all");
    if (all && sel) {
      if (all.dataset.show === "0") clearPick();  // hiding everything lets go of the picked shape
      return setAllHidden(sel.svg, all.dataset.show === "0");
    }
    const chip = e.target.closest?.(".se-chip");
    if (chip && sel) {
      const i = Number(chip.dataset.i), hide = !hidden(sel.svg).has(i);
      if (hide && sel.i === i) clearPick();  // hiding the picked shape lets go of it
      return setHidden(sel.svg, i, hide);
    }
    const eye = e.target.closest?.(".scene-eye");
    if (eye) {
      const svg = eye.closest(".scene-body")?.querySelector(`.scene-frame svg[data-which="${eye.dataset.which}"]`);
      if (svg) setHidden(svg, Number(eye.dataset.i), !hidden(svg).has(Number(eye.dataset.i)));
      return;
    }
    const b = e.target.closest?.(".scene-adjust");
    if (!b) return;
    const svg = b.closest(".scene-body")?.querySelector(`.scene-frame svg[data-which="${b.dataset.which}"]`);
    if (svg) select(svg, Number(b.dataset.i));
  });

  document.addEventListener("keydown", (e) => {
    const k = e.key.toLowerCase();
    if (sel && (e.metaKey || e.ctrlKey) && (k === "z" || k === "y") && !e.target.closest?.("input, textarea")) {
      e.preventDefault();
      return stepTo(sel.pos + (k === "y" || e.shiftKey ? 1 : -1));
    }
    if ((e.key === "Delete" || e.key === "Backspace") && sel?.corner && !document.querySelector(".se-ask")) {
      e.preventDefault();
      return removeCorner(sel.corner);
    }
    if (e.key !== "Escape" || !sel) return;
    const ask = document.querySelector(".se-ask");
    ask ? ask.remove() : requestClose();
  });
  // leaving the page with unsaved changes: the browser asks
  window.addEventListener("beforeunload", (e) => { if (sel?.pending.size) e.preventDefault(); });
  // the big view changes size with the window; handles keep their on-screen size
  window.addEventListener("resize", () => { if (sel?.obj) draw(); });
  document.addEventListener("DOMContentLoaded", applyAll);
  document.addEventListener("htmx:afterSettle", applyAll);
  applyAll();
  // the panel was swapped (outline, accept, reset): the old selection is gone
  document.addEventListener("htmx:beforeSwap", (e) => { if (["scene-panel", "zones-status"].includes(e.detail.target?.id)) closeView(); });
})();
