# Third-party scripts, served from here instead of a CDN

Exact pinned releases, copied unchanged from each project's npm package (`dist/`), so a page never runs
whatever a CDN happens to serve. No build step: to upgrade, replace the file with the new release and
change the version in its name and in the templates (`base.html`, `stats.html`, `species.html`,
`weather.html`, `false_triggers.html`) or `species-card.js`.

| File | Project | Version | License |
| --- | --- | --- | --- |
| `htmx-2.0.4.min.js` | https://htmx.org | 2.0.4 | BSD Zero Clause (`htmx-2.0.4.LICENSE`) |
| `chart-4.4.1.umd.js` | https://www.chartjs.org | 4.4.1 | MIT (`chart-4.4.1.LICENSE.md`) |
| `leaflet-1.9.4/` | https://leafletjs.com | 1.9.4 | BSD 2-Clause (`leaflet-1.9.4/LICENSE`) |

Leaflet is loaded only when a species card with a range map opens (`species-card.js`); `images/` holds
the icons its stylesheet refers to.
