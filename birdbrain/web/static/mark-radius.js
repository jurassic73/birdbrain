// One rule for how big a circle is drawn over a photo, so every view agrees (thumbnails, the home
// photo, the visit page). Display only: stored circles keep their true radius.
// r is a fraction of the photo's width. The ring sits a little outside the animal (x1.15) and is
// never smaller than 4% of the width, so a small bird (a chickadee) stays visible inside it.
window.markRadius = (r) => Math.max(r * 1.15, 0.04);
