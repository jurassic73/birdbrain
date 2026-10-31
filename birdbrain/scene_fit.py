"""Tidy Claude's scene outline against the photo it was drawn on.

1. Round things (a birdbath, a landing spot, a birdhouse hole) are always ellipses: a polygon becomes the ellipse
   around its points.
2. A birdbath is one ring: its outer rim and the water inside (`inner`). Rim or water objects that duplicate it are
   folded into it.
3. The ellipses snap onto the photo: the outer rim first (concrete is brighter than the ground around it, day and
   night), then the water's edge inside it (by day the water mirrors the sky and can be brighter than the rim, at
   night darker, so either way), by sliding each ellipse to where the image's edges line up with it best. By day, a landing spot inside the water
   moves onto the nearest blob that stands out from the water, darker or lighter (a gray rock on a black liner).
   At night (IR) only the bath snaps; rocks barely show.
4. Ground is cut around the bath, so its outline never crosses the bowl.

Objects are dicts as stored by scene.clean: ellipse [cx, cy, rx, ry] or points [[x, y], ...] as frame fractions.
OpenCV is imported inside the functions that need it so the web app doesn't load it at startup.
"""

import numpy as np

ROUND = {"birdbath", "landing_spot", "birdhouse"}
FIT_W = 640          # the photo is fitted at this width (the substream's), plenty for an outline
SPAN = 0.3           # how far (as a share of the radius) a snapped ellipse may move or grow
SAMPLES = 160        # points around an ellipse when scoring it
SNAP_GAIN = 1.15     # a snapped ellipse must line up with edges this much better than Claude's to replace it
ROCK_DIFF = 30       # how far from the water's brightness a pixel must be to count as rock
MERGE_IOU = 0.55     # boxes overlapping this much describe the same bath


# --- geometry ---

def box(o: dict) -> tuple[float, float, float, float]:
    if "ellipse" in o:
        cx, cy, rx, ry = o["ellipse"]
        return cx - rx, cy - ry, cx + rx, cy + ry
    xs, ys = [p[0] for p in o["points"]], [p[1] for p in o["points"]]
    return min(xs), min(ys), max(xs), max(ys)


def iou(a, b) -> float:
    ix, iy = max(0.0, min(a[2], b[2]) - max(a[0], b[0])), max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def round4(e) -> list[float]:
    return [round(min(max(float(v), 0.0), 1.0), 4) for v in e]


def as_ellipse(o: dict) -> dict:
    """Round objects drawn as polygons become the ellipse around their points."""
    if o["kind"] not in ROUND or "ellipse" in o:
        return o
    x1, y1, x2, y2 = box(o)
    o = {k: v for k, v in o.items() if k != "points"}
    o["ellipse"] = round4([(x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 2, (y2 - y1) / 2])
    return o


def merge_bath(objects: list[dict]) -> list[dict]:
    """Fold rim and water objects that duplicate a birdbath into it; a smaller one inside becomes its water."""
    baths = [o for o in objects if o["kind"] == "birdbath" and "ellipse" in o]
    if not baths:
        return objects
    bath = max(baths, key=lambda o: o["ellipse"][2] * o["ellipse"][3])
    out = []
    for o in objects:
        if o is bath or o["kind"] not in ("birdbath", "perch") or iou(box(o), box(bath)) < MERGE_IOU:
            out.append(o)
            continue
        e = as_ellipse({**o, "kind": "birdbath"}).get("ellipse")
        if e and "inner" not in bath and e[2] * e[3] < 0.9 * bath["ellipse"][2] * bath["ellipse"][3]:
            bath["inner"] = e
        if o.get("note") and o["note"] not in bath["note"]:
            bath["note"] = f"{bath['note']} {o['note']}".strip()
    return out


# --- snapping onto the photo ---

def ellipse_points(cx, cy, rx, ry, n=SAMPLES):
    t = np.linspace(0, 2 * np.pi, n, endpoint=False)
    ct, st = np.cos(t), np.sin(t)
    nx, ny = ct / max(rx, 1e-6), st / max(ry, 1e-6)
    norm = np.hypot(nx, ny)
    return cx + rx * ct, cy + ry * st, nx / norm, ny / norm


def edge_score(gx, gy, e, sign: int = 0) -> float:
    """How well an ellipse (pixels) lies along the image's edges: the brightness change across it, averaged.
    sign=1 counts only edges that get brighter going outward, -1 only darker going outward (a light rim on dark
    ground), 0 either."""
    H, W = gx.shape
    xs, ys, nx, ny = ellipse_points(*e)
    inside = (xs >= 0) & (xs < W - 1) & (ys >= 0) & (ys < H - 1)
    if inside.mean() < 0.7:
        return 0.0
    xi, yi = xs[inside].astype(int), ys[inside].astype(int)
    d = gx[yi, xi] * nx[inside] + gy[yi, xi] * ny[inside]
    return float(np.mean(np.maximum(d * sign, 0) if sign else np.abs(d)))


def snap(gx, gy, e, sign: int = 0, ok=lambda c: True, span: float = SPAN) -> tuple[list[float], float, float]:
    """Search around ellipse e (pixels) for the one that best follows the edges, among those ok() allows; coarse to
    fine. Returns the best ellipse, its score and the starting ellipse's score."""
    start = edge_score(gx, gy, e, sign)
    best, best_s = list(e), start
    for _ in range(4):
        cx, cy, rx, ry = best
        steps = np.linspace(-1, 1, 5)
        for dx in steps * span * rx:
            for dy in steps * span * ry:
                for dr in steps * span:
                    for dq in steps * span / 2:
                        c = [cx + dx, cy + dy, rx * (1 + dr), ry * (1 + dr + dq)]
                        if not ok(c):
                            continue
                        s = edge_score(gx, gy, c, sign)
                        if s > best_s:
                            best, best_s = c, s
        span /= 2.5
    return best, best_s, start


def contour_seeds(gray, box_px, n: int = 4) -> list[list[float]]:
    """Ellipses fitted to the closed edges near Claude's box, best first: a starting point that doesn't depend on
    where Claude put the bath (its guesses can be off by half the bowl)."""
    import cv2
    x1, y1, x2, y2 = box_px
    w, h = x2 - x1, y2 - y1
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120)
    contours, _ = cv2.findContours(cv2.dilate(edges, np.ones((3, 3), np.uint8)), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in contours:
        if len(c) < 40:
            continue
        (cx, cy), (a, b), _ = cv2.fitEllipse(c)
        x, y, bw, bh = cv2.boundingRect(c)  # the box is axis-aligned, like our ellipses
        if not (x1 - 0.25 * w < cx < x2 + 0.25 * w and y1 - 0.25 * h < cy < y2 + 0.25 * h):
            continue
        if not (0.25 * w < bw < 1.3 * w and 0.25 * h < bh < 1.3 * h and max(a, b) < 2.5 * min(a, b) * 2):
            continue
        out.append([x + bw / 2, y + bh / 2, bw / 2, bh / 2])
    return out


def is_night(bgr) -> bool:
    import cv2
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].mean()) < 12


def ring_around(gx, gy, e, scales) -> tuple[list[float], float]:
    """The best ellipse around (scale > 1) or inside (scale < 1) ellipse e: about the same shape, scaled, and a
    little lower or higher, since the camera looks down at the bowl at an angle."""
    cx, cy, rx, ry = e
    best, best_s = None, -1.0
    for k in scales:
        for dy in np.linspace(-0.1, 0.3, 9) * ry * abs(k - 1) / 0.2:
            for dx in np.linspace(-0.04, 0.04, 3) * rx:
                for a in (0.92, 0.96, 1.0, 1.04, 1.08, 1.12):
                    c = [cx + dx, cy + (dy if k > 1 else -dy), rx * k, ry * k * a]
                    sc = edge_score(gx, gy, c)
                    if sc > best_s:
                        best, best_s = c, sc
    return best, best_s


def fit_bath(gray, bath: dict) -> dict:
    """Find the bowl as two rings: the water's edge (the strongest ring: by day bright water inside a black liner,
    at night dark water) and the outer rim just around it. The ring found first might be either one, so both readings
    are tried and the pair that lines up best wins; it replaces Claude's only if clearly better."""
    import cv2
    H, W = gray.shape
    blur = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
    gx, gy = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    px = lambda e: [e[0] * W, e[1] * H, e[2] * W, e[3] * H]  # noqa: E731
    fr = lambda e: round4([e[0] / W, e[1] / H, e[2] / W, e[3] / H])  # noqa: E731
    outer = px(bath["ellipse"])
    inner = px(bath["inner"]) if "inner" in bath else [outer[0], outer[1], outer[2] * 0.85, outer[3] * 0.85]
    claude = edge_score(gx, gy, outer) + edge_score(gx, gy, inner)

    # Starting points: Claude's water ellipse (also smaller, since Claude's are often too big) and ellipses fitted to
    # closed edges near Claude's box (in case Claude's is off to one side). The strongest few rings get a rim.
    seeds = [[inner[0], inner[1], inner[2] * k, inner[3] * k] for k in (1.0, 0.8, 0.65)]
    seeds += contour_seeds(gray, (outer[0] - outer[2], outer[1] - outer[3], outer[0] + outer[2], outer[1] + outer[3]))
    rings = sorted((snap(gx, gy, sd)[:2] for sd in seeds), key=lambda r: r[1], reverse=True)[:3]
    best, best_s = None, 0.0
    for ring, ring_s in rings:
        rim, rim_s = ring_around(gx, gy, ring, np.arange(1.12, 1.46, 0.03))  # the ring is the water (the rim is wider than the liner)
        water, water_s = ring_around(gx, gy, ring, np.arange(0.66, 0.9, 0.03))  # the ring is the rim
        # usually the strongest ring is the water's edge; only call it the rim if the water inside it is clearly better
        pair, sc = ((water, ring), water_s) if water_s > 1.3 * rim_s else ((ring, rim), rim_s)
        if ring_s + sc > best_s:
            best, best_s = pair, ring_s + sc
    if best and best_s > SNAP_GAIN * claude:
        bath["inner"], bath["ellipse"] = fr(best[0]), fr(best[1])
        bath["fitted"] = ["water", "rim"]
    return bath


def fit_spot(gray, spot: dict, water: list[float]) -> dict:
    """Move a landing spot in the water onto the nearest blob that stands out from the water."""
    import cv2
    H, W = gray.shape
    wcx, wcy, wrx, wry = water[0] * W, water[1] * H, water[2] * W, water[3] * H
    mask = np.zeros_like(gray)
    cv2.ellipse(mask, (int(wcx), int(wcy)), (max(int(wrx * 0.95), 1), max(int(wry * 0.95), 1)), 0, 0, 360, 255, -1)
    if not cv2.countNonZero(mask):
        return spot
    level = float(np.median(gray[mask > 0]))
    diff = cv2.absdiff(gray, np.full_like(gray, int(level)))
    m = cv2.bitwise_and(cv2.threshold(diff, ROCK_DIFF, 255, cv2.THRESH_BINARY)[1], mask)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(m)
    cx, cy, rx, ry = spot["ellipse"][0] * W, spot["ellipse"][1] * H, spot["ellipse"][2] * W, spot["ellipse"][3] * H
    guess_area = np.pi * rx * ry
    best = None
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not 0.15 * guess_area <= area <= 2.5 * guess_area:
            continue
        # a reflection of the sky or trees usually reaches the water's edge; a rock sits inside it
        edge = mask[max(y - 2, 0):y + h + 2, max(x - 2, 0):x + w + 2]
        if edge.size and edge.min() == 0:
            continue
        dist = np.hypot((cents[i][0] - cx) / max(rx, 1), (cents[i][1] - cy) / max(ry, 1))
        if dist <= 3 and (best is None or dist < best[0]):
            best = (dist, x, y, w, h)
    if best:
        _, x, y, w, h = best
        spot["ellipse"] = round4([(x + w / 2) / W, (y + h / 2) / H, w * 0.55 / W, h * 0.55 / H])
        spot["fitted"] = ["photo"]
    return spot


def fit(objects: list[dict], jpeg: bytes) -> list[dict]:
    """Snap round objects onto the photo they were outlined on."""
    import cv2
    bgr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        return objects
    scale = FIT_W / bgr.shape[1]
    bgr = cv2.resize(bgr, (FIT_W, max(int(bgr.shape[0] * scale), 1)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    night = is_night(bgr)
    baths = [o for o in objects if o["kind"] == "birdbath" and "ellipse" in o]
    for b in baths:
        fit_bath(gray, b)
    if not night:
        for b in baths:
            water = b.get("inner") or [b["ellipse"][0], b["ellipse"][1], b["ellipse"][2] * 0.85, b["ellipse"][3] * 0.85]
            for o in objects:
                if o["kind"] != "birdbath" and "ellipse" in o and o["kind"] in ("landing_spot", "distraction") \
                        and ((o["ellipse"][0] - water[0]) / water[2]) ** 2 + ((o["ellipse"][1] - water[1]) / water[3]) ** 2 < 1:
                    fit_spot(gray, o, water)
    return objects


# --- ground around the bath ---

def cut_ground(objects: list[dict], grid: int = 400) -> list[dict]:
    """Ground polygons minus every birdbath's outer ellipse (a little larger), redrawn as outlines."""
    import cv2
    baths = [o["ellipse"] for o in objects if o["kind"] == "birdbath" and "ellipse" in o]
    if not baths:
        return objects
    W, H = grid, grid  # fractions, so a square grid is fine
    out = []
    for o in objects:
        if o["kind"] != "ground" or "points" not in o:
            out.append(o)
            continue
        m = np.zeros((H, W), np.uint8)
        cv2.fillPoly(m, [np.array([[x * W, y * H] for x, y in o["points"]], np.int32)], 255)
        for cx, cy, rx, ry in baths:
            cv2.ellipse(m, (int(cx * W), int(cy * H)), (int(rx * W * 1.04) + 1, int(ry * H * 1.04) + 1), 0, 0, 360, 0, -1)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        if not contours:
            continue
        keep = [c for c in contours if cv2.contourArea(c) >= 0.08 * cv2.contourArea(contours[0])]
        polys = []
        for c in keep:
            a = cv2.approxPolyDP(c, 1.5, True)[:, 0, :]
            if len(a) >= 3:
                polys.append([[round(float(x) / W, 4), round(float(y) / H, 4)] for x, y in a])
        if polys:
            out.append({**o, "points": polys[0], **({"pieces": polys[1:]} if polys[1:] else {})})
    return out


def tidy(objects: list[dict], jpeg: bytes | None = None) -> list[dict]:
    """All four steps; the photo steps are skipped (and Claude's shapes kept) if anything goes wrong there."""
    objects = merge_bath([as_ellipse(o) for o in objects])
    if jpeg:
        try:
            objects = fit(objects, jpeg)
        except Exception:  # an odd frame shouldn't lose the outline
            pass
    return cut_ground(objects)
