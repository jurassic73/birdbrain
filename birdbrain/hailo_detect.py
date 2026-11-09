"""Hailo-8 shadow detector, command line (birdbrain/hailo/ holds the pieces).

  python -m birdbrain.hailo_detect                 run the always-on detector (the systemd service does this)
  python -m birdbrain.hailo_detect --status        fps, last frame age, chip temperature, event counts
  python -m birdbrain.hailo_detect --evaluate      offline scorecard over the stored visit photos (read-only)
  python -m birdbrain.hailo_detect --report        events joined to visits by time, --since 24h

Shadow only: nothing here creates visits, calls Claude or changes the watcher.
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

from . import config
from .hailo import store


def status() -> str:
    hb = store.read_heartbeat()
    lines = []
    if hb is None:
        lines.append(f"no heartbeat at {config.HAILO_HEARTBEAT} (service not running yet?)")
    else:
        age = (datetime.now() - datetime.fromisoformat(hb["at"])).total_seconds()
        lines.append(f"heartbeat {age:.0f} s old (pid {hb.get('pid')}, up {hb.get('uptime_s', 0) // 60} min), "
                     f"model {hb.get('model')}")
        lines.append(f"fps {hb.get('fps')}, inference {hb.get('infer_ms')} ms, cpu {hb.get('cpu_pct')}% of a core, "
                     f"chip {hb.get('chip_temp_c')} C, {'night' if hb.get('night') else 'day'}")
        lines.append(f"last frame {hb.get('last_frame_age_s')} s ago ({hb.get('last_frame_at')}), "
                     f"{hb.get('frames')} frames this run, {hb.get('open_events')} events in view")
        if hb.get("other_classes"):
            lines.append("non-animal classes seen this run: "
                         + ", ".join(f"{k} {n}" for k, n in hb["other_classes"].items()))
    now = datetime.now()
    with store.session() as conn:
        today = store.event_counts(conn, now.strftime("%Y-%m-%dT00:00:00"))
        day = store.event_counts(conn, (now - timedelta(hours=24)).isoformat(timespec="seconds"))
        total = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    for name, c in (("today", today), ("last 24 h", day)):
        lines.append(f"events {name}: {c['events']} ({c['animals']} animal-like, {c['in_zone']} of those in the "
                     f"bath or crop zone, {c['open']} open)")
    lines.append(f"events in all: {total} in {config.HAILO_SHADOW_DB}; snapshots "
                 f"{store.folder_size(config.HAILO_SNAPSHOT_DIR) / 1e6:.0f} MB in {config.HAILO_SNAPSHOT_DIR}")
    return "\n".join(lines)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--status", action="store_true", help="print the heartbeat and event counts")
    parser.add_argument("--evaluate", action="store_true", help="score the stored visit photos (read-only)")
    parser.add_argument("--hef", action="append", type=Path, help="HEF(s) to evaluate (default: HAILO_HEF)")
    parser.add_argument("--limit", type=int, help="evaluate only the first N photos")
    parser.add_argument("--out", type=Path, default=Path("docs/hailo-shadow-scorecard.md"),
                        help="where --evaluate writes its markdown report")
    parser.add_argument("--report", action="store_true", help="join detector events to visits by time")
    parser.add_argument("--since", default="24h", help="window for --report: e.g. 90m, 24h, 3d")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.status:
        print(status())
    elif args.evaluate:
        from .hailo import evaluate
        print(evaluate.run(args.hef or [config.HAILO_HEF], args.out, args.limit))
    elif args.report:
        from .hailo import report
        print(report.run(args.since))
    else:
        from .hailo import service
        service.main()


if __name__ == "__main__":
    main()
