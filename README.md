# Birdbrain

A backyard birdbath camera that spots visitors and identifies them with Claude, then shows them on a website on your home network: latest visitors, species, stats, and a live view.

![The home page: the latest visitor with the spot where it was found circled, a live view of the bath, today's visitors and the newest visits](docs/screenshots/home.jpg)

<table>
<tr>
<td width="50%"><img src="docs/screenshots/visit.jpg" alt="A visit page: a Steller's Jay circled on the rim, Claude's answer and notes, and the steps that decided it"><br><b>Every visit</b> shows the photo, where the animal was spotted, and each step that decided it: motion zone, local detector, Claude Haiku, the local BioCLIP 2 guess.</td>
<td width="50%"><img src="docs/screenshots/species.jpg" alt="A species card for Steller's Jay with a reference photo, description, behavior notes, three local recordings and a monthly chart"><br><b>Species cards</b> with a reference photo, behavior notes, songs and calls recorded nearby, and when the species is usually around in your county.</td>
</tr>
<tr>
<td width="50%"><img src="docs/screenshots/stats.jpg" alt="The stats page: visits per day and by hour, top visitors, new species and scorecards for the local models"><br><b>Stats</b>: visits by day and hour, top visitors, new arrivals, and scorecards that show how well the local models would do without Claude.</td>
<td width="50%"><img src="docs/screenshots/about.jpg" alt="The About page: live totals, including Claude spend, and a diagram of the whole pipeline from camera to the owner"><br><b>How it works</b>, built into the site: live totals (2,700 false alarms caught, $0.54 spent on Claude) and the trip from splash to photo.</td>
</tr>
<tr>
<td width="50%"><img src="docs/screenshots/ntfy.jpg" alt="Phone push alerts from the ntfy app: New species Steller's Jay and New species American Crow, each with the photo"></td>
<td width="50%"><b>Phone alerts</b> through the free <a href="https://ntfy.sh">ntfy</a> app, with the photo attached: a species seen for the first time, any raccoon or possum, and birds showing up outside their usual season. Quiet hours and each kind of alert are set on the Settings page.</td>
</tr>
</table>

- A cheap Wi-Fi camera watches the birdbath.
- Motion on the rim (not ripples in the water) triggers a burst of photos.
- Claude Haiku identifies the visitor from the sharpest photo. If it isn't sure, Claude Sonnet takes a second look at three photos. If it's still unsure, the visit waits for you on the Review page.
- When you correct a mistake, your correction is shown to Claude on future photos, so it learns your birdbath's quirks.
- Phone alerts come through [ntfy.sh](https://ntfy.sh), a notification service that takes minutes to set up: install the free ntfy app, subscribe to a topic name you make up, and put the same name on the Settings page. No account or server needed, and the free tier allows 250 notifications a day, far more than a birdbath sends.

> **A note on Claude tokens:** a big part of this project is spending as few Claude API tokens as possible. Claude is very good at naming birds, but most motion at a birdbath is rain, leaves and ripples, and on busy days nearly every call was an empty frame. So the work is steadily moving onto a stronger local model running on the Pi itself: [BioCLIP 2](#local-species-classifier-shadow-mode) already decides whether anything is there at all (the [local animal check](#local-animal-check), which keeps empty frames off Claude) and makes its own species guess on every visit, scored against Claude's answers (25 of the first 28 birds right on its first guess). The goal is for the Pi to name the everyday visitors by itself and to save Claude for the hard cases: new species, odd angles, night visitors and anything the local model isn't sure about.

## What you need

| Part | Notes |
|---|---|
| Camera with RTSP | Tested with a Wyze Cam v3 (~$30) flashed with [Thingino](https://thingino.com) open firmware (no soldering: SD-card installer from [wltechblog/thingino-installers](https://github.com/wltechblog/thingino-installers)). Any camera with an RTSP main stream and a smaller substream should work. |
| Always-on computer | A Raspberry Pi 4 (2 GB+) on your network runs the watcher and the website. A Mac or Linux PC works for trying it out. |
| Anthropic API key | Pay-as-you-go. A typical backyard costs a few dollars a month (see below). |
| Software | Python 3.11+, [uv](https://docs.astral.sh/uv/), ffmpeg. |

## Set up the Anthropic API key safely

1. Create an account at [console.anthropic.com](https://console.anthropic.com) and add **prepaid credits**. About **$20** is plenty to start.
2. **Turn auto-reload off.** Then a leaked key can never spend more than what's left on the balance.
3. Create an API key for Birdbrain only. A key that doesn't expire is fine here, because the small prepaid balance is what limits your exposure.
4. Put the key in `.env` (below), never in code, chat, or a commit.

Cost per photo is tiny: Haiku on a cropped photo is about $0.00015; a Sonnet second look with three photos is under a cent. `MAX_CALLS_PER_DAY` (default 500) is a hard daily cap.

## Install

```bash
git clone <this repo> birdbrain && cd birdbrain
uv sync
cp .env.example .env      # then edit .env: camera address and password, API key, Pi address
```

Check the camera stream:

```bash
scripts/snap.sh main      # saves a full-resolution frame to snapshots/
```

Try it locally:

```bash
uv run uvicorn birdbrain.web.app:app --port 8000      # the website: http://localhost:8000
uv run python -m birdbrain.watch                       # the watcher (Ctrl-C to stop)
```

## Aim and calibrate

1. Mount the camera 2 to 3 feet from the bath, above it and angled down 30 to 45 degrees. That shows the whole rim, makes the water reflect sky instead of your house, and keeps moving background out of view.
2. Draw the areas to watch and to send to Claude, and check them on a real frame:

   ```bash
   uv run python -m birdbrain.calibrate preview.jpg --ring cx,cy,orx,ory,irx,iry --crop x,y,w,h --spots cx,cy,rx,ry
   ```

   Green is watched for motion (the rim, plus spots like rocks birds land on). Orange is what Claude sees. Adjust until they fit, then copy the values into `.env`.
3. Optional: watch the ground around the bath too, for critters passing by that never climb onto it. Add `--ground "x,y;x,y;x,y;..."` (the corners of the area, as fractions of the frame) to the command above. Blue is the ground zone (the bath itself is left out) and yellow is what Claude sees for ground visits (the box around the zone, or `--ground-crop x,y,w,h`). Leave out shrubs and plants that sway and the camera's clock overlay. Save the values as `GROUND_ZONE` (and `GROUND_CROP` if you set one) in `.env`. See [The ground zone](#the-ground-zone) below for how it avoids false alarms.
4. After that the zones look after themselves. Twice a day (10:30 and 14:30 by default; change or turn off under **Settings > Zones**) the watcher spends about a minute studying the bath and takes the middle value of each pixel, so ripples, drifting leaves and birds drop out. If the camera has shifted a few pixels, every zone moves with it; a big move is only reported, since that needs a fresh hand calibration. Then each rock spot moves back onto its rocks, which by day are dark against the sky reflected in the water. **Recalibrate now** runs it right away, every run is listed with a before/after picture, and **Undo** takes a change back, and you can drag any zone by hand. It only runs by daylight. The adjusted zones live in the database, not `.env`, so redeploying never undoes them; editing the zones in `.env` by hand makes those the starting point again.
5. Open **Claude's Notes** on the website and describe your setup in three boxes: what the camera sees, what to look for (the visitors you expect and their sizes), and what to ignore (a branch that sways, rocks poking out of the water, a cord). **Refine with Claude** tidies them into short lines, and **Save and outline** has Claude outline each thing on a fresh frame. Claude reads this with every photo. You can see exactly what it's told at the bottom of that page, along with the lessons from your corrections.

## Optional: a local detector to save Claude calls

Rain and wind can trigger motion every few seconds. Two things keep that cheap:

- **Backoff (always on):** each false trigger in a row doubles the wait before the next try (20 s, 40 s, ... up to 5 minutes), and a real visitor resets it. With the local detector installed, motion during a backoff still gets a free look on the Pi, and only goes to Claude if it finds an animal.
- **Birds in the water:** besides the rim, the open water is watched with a stricter rule: one solid, bird-sized blob that stays put for over half a second. Ripples and raindrops never qualify, and anything that does is checked by the local detector before Claude. This zone sleeps at night (nothing flies in to land after dark).
- **Local detector:** a small open model ([YOLOX-tiny](https://github.com/Megvii-BaseDetection/YOLOX), Apache-2.0) runs on the Pi's CPU in well under a second and answers "is there an animal here?" before Claude is asked.

```bash
scripts/get_model.sh      # downloads and verifies the ~19 MB model into models/
```

It starts in **shadow mode**: it runs and is recorded on every visit, but Claude still sees everything. The Stats page shows how many calls it would have saved and whether it ever missed an animal Claude found. When you trust it, set `DETECTOR=gate` in `.env` and redeploy: daytime frames where it sees nothing then skip Claude. It never skips at night, because it doesn't know raccoons, possums or squirrels (COCO classes); those often show up as "cat", "dog" or "bear", and anything like that still goes to Claude.

## Local species classifier (shadow mode)

A second small model guesses each bird's species on the Pi, so you can see whether birds could one day be named without Claude. It's [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2) (Imageomics Institute, MIT), with its image encoder converted to an 8-bit ONNX file (300 MB) and its names precomputed for every species in the eBird list; the files are on this repo's [`bioclip2-v1` release](https://github.com/jurassic73/birdbrain/releases/tag/bioclip2-v1). It downloads itself into the data folder the first time it's needed (checked by SHA-256), takes about 6 seconds per photo on a Pi 4 and uses about 400 MB of RAM. Guesses are limited to birds reported to iNaturalist in your `INAT_PLACE_ID` (the list is cached and refreshed monthly).

It only ever records its top 3 guesses: Claude (or your correction) is still the label. Each visit page shows the guesses under "How it was decided", and the Stats page scores them against the final labels, overall and species by species. To score the visits you already have:

```bash
uv run python -m birdbrain.classifier --backfill            # every real visit without a guess yet
uv run python -m birdbrain.classifier --backfill --limit 50 # just the newest 50
```

It knows mammals too (every iNaturalist mammal, limited to your place's), so squirrels get a species.

### Local animal check

The same model also answers "is anything there?" for every trigger, using a small check trained on your own labelled frames (Claude's answers, your corrections, and frames you mark "Looks right" or confirm on the False triggers page). It retrains itself whenever labels change. In **gate** mode (Settings > Local animal check) a frame it judges empty skips Claude and is kept as a false trigger marked "pi only" for you to check; correct any with an animal and it learns from that. It needs at least 10 visitors and 30 empty frames with embeddings before it judges anything. To embed frames you already have (about 6 s each on a Pi 4):

```bash
uv run python -m birdbrain.presence --backfill
```

Set `CLASSIFIER=off` in `.env` to skip it. It hasn't been checked on infrared night photos yet.

### The ground zone

With `GROUND_ZONE` set, the ground around the bath is watched for animals walking by. Blowing leaves, rain splashes and changing light are the hard part, so it only counts one sizable blob that stays in view for about half a second and moves at walking pace; a change across the whole zone at once (clouds, headlights, the camera switching to night vision) is ignored.

- **By day** a ground-bird-sized blob is enough, and the local detector looks first: if it sees no animal, the photo is dropped and nothing is stored or sent. Without the detector, daytime ground motion goes to Claude.
- **At night** (infrared) the blob must be about cat-sized, since night visitors are raccoons, possums, cats and the like, and moths at the infrared lights are the usual false alarm. Night ground motion goes to Claude, because the detector hasn't proven it recognizes these animals in grayscale.
- It has its own budget, `GROUND_MAX_CALLS_PER_DAY` (default 100), and its own backoff, so a windy day on the ground never uses up the birdbath's calls or makes the watcher miss a bird at the bath.

Tuning (all optional, fractions of the frame): `GROUND_MIN_BLOB` (day, 0.003), `GROUND_MIN_BLOB_NIGHT` (0.006), `GROUND_DIFF` (20), `GROUND_PERSIST_FRAMES` (8). The visit page says which zone saw the motion.

## Run it on a Raspberry Pi

Set `DEPLOY_HOST` and `DEPLOY_USER` in `.env`, make sure you can `ssh` to the Pi with a key, then:

```bash
scripts/deploy.sh
```

This copies the code to `/opt/birdbrain`, installs Python packages, copies `.env` (readable only by that user), and starts two services that also start at boot: `birdbrain-web` (the site on port 8000) and `birdbrain-watch`. Photos and the database live in `/var/lib/birdbrain`, separate from the code, so redeploying never touches them. Logs: `journalctl -u birdbrain-watch -f`.

## Optional: read-only mode for guests

Set `ADMIN_PASSWORD` in `.env` and the site becomes read-only for everyone else: they can browse visits, species, stats and the About page, but can't correct, delete, change settings or ask Claude anything. The Admin button at the top logs you in on that device for 30 days. Leave it blank to keep the site fully open (fine on a home LAN). Before putting the site on the internet, put HTTPS in front of it so the password isn't sent in the clear.

## What stays out of git

Everything specific to your installation stays on your machines (see `.gitignore`):

| Path | What it is |
|---|---|
| `.env` | Camera password, API key, addresses, calibration. Start from `.env.example`. |
| `data/` | The database (visits, your Settings, corrections) and photos. On the Pi: `/var/lib/birdbrain`. |
| `snapshots/` | Test frames from `scripts/snap.sh`. |
| `models/` | The local detector model, downloaded by `scripts/get_model.sh`. (The species classifier downloads itself into `data/models/`.) |
| `firmware-backup/` | Your camera's factory firmware backup, if you flashed Thingino. It's unique to that camera; keep a copy somewhere safe. |

The code itself has no addresses, passwords or locations in it.

## Develop

```bash
uv run pytest                                   # ~90 tests, a few seconds; never calls the camera or the API
uv run python -m birdbrain.devseed --reset      # fill the local database with fake visits for UI work
```

The code is small: `birdbrain/config.py` (settings from `.env`), `db.py` (all database queries), `identify.py` (Claude calls), `watch.py` (motion and capture), `web/` (FastAPI site with Jinja templates and htmx, no build step).

If you use [Claude Code](https://claude.com/claude-code) to change it: `CLAUDE.md` holds the original author's working notes and decisions, which are specific to their setup. Read it for context, and keep your own setup details in `.env` and Claude's Notes and Settings pages.

## Credits

The light/dark toggle uses the sun and crescent moon from [Twemoji](https://github.com/jdecked/twemoji) (© Twitter and contributors, [CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/)).

## License

MIT, see [LICENSE](LICENSE). The BioCLIP 2 model files are MIT (Imageomics Institute) and YOLOX-tiny is Apache-2.0; each keeps its own license.
