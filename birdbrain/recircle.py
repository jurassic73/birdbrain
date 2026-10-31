"""Add Claude's circle to visits stored before it drew one. Labels are never changed.

uv run python -m birdbrain.recircle 2026-10-26 [--dry-run]

Each real visit of that day without a circle goes back to the model that answered it, on the saved full frame.
The circle is kept only when the new answer names the same kind of visitor as the visit's label.
"""
import json
import sys
from datetime import datetime

from . import config, db, identify


def pending(conn, day: str) -> list:
    return conn.execute(
        "SELECT * FROM visits WHERE substr(captured_at, 1, 10) = ? AND COALESCE(corrected_visitor, visitor) != 'none' "
        "AND json_extract(raw_json, '$.calls.circle') IS NULL ORDER BY id",
        (day,),
    ).fetchall()


def save_circle(conn, visit_id: int, circle: dict) -> None:
    conn.execute(
        "UPDATE visits SET raw_json = json_set(COALESCE(raw_json, '{}'), '$.calls.circle', json(?)) WHERE id = ?",
        (json.dumps(circle), visit_id),
    )
    conn.commit()


def recircle(conn, visit, site: dict | None = None) -> dict | None:
    jpeg = (config.IMAGES_DIR / visit["image_path"]).read_bytes()
    model = identify.SONNET if visit["model"] == identify.SONNET else identify.HAIKU
    sighting, _ = identify.ask(model, [identify.prepare(jpeg)], datetime.fromisoformat(visit["captured_at"]), site=site)
    if not sighting or sighting.visitor != (visit["corrected_visitor"] or visit["visitor"]):
        return None
    circle = identify.frame_circle(sighting)
    if circle:
        save_circle(conn, visit["id"], circle)
    return circle


def main() -> None:
    day, dry = sys.argv[1], "--dry-run" in sys.argv
    with db.session() as conn:
        db.init_db(conn)
        site = db.prompt_site(conn)
        for v in pending(conn, day):
            label = v["corrected_species"] or v["species"] or v["corrected_visitor"] or v["visitor"]
            circle = None if dry else recircle(conn, v, site)
            print(v["id"], v["captured_at"], label, "(dry run)" if dry else circle or "no matching circle")


if __name__ == "__main__":
    main()
