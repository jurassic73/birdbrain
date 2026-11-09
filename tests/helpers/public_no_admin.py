"""Shared helpers for the public no admin tests (used by more than one test module)."""

# What the mirror serves: (method, route template). The two mounts are the static files and the photos.
PUBLIC = {
    ("GET", "/"), ("GET", "/live"), ("GET", "/scoreboard"), ("GET", "/visits"), ("GET", "/visits/archive-more"),
    ("GET", "/visits/more"), ("GET", "/visit/{visit_id}"), ("GET", "/species"), ("GET", "/species/{name}"),
    ("GET", "/species/{name}/more"), ("GET", "/species-info/{name}"), ("GET", "/stats"), ("GET", "/weather"),
    ("GET", "/about"), ("GET", "/api/latest-visit"), ("GET", "/live/latest.jpg"), ("GET", "/live.jpg"),
    ("GET", "/manifest.webmanifest"), ("GET", "/favicon.ico"),
    ("GET", "/guest-login"), ("POST", "/guest-login"),
    # the Guests counter's beacon (static/track.js): counts a click or a prerendered page, answers 204
    ("POST", "/api/track"),
    ("GET", "/static"), ("GET", "/images"),
}


# Everything else the app has: the admin's, so a 404 on the mirror. A new route goes in one list or the other.
ADMIN = {
    # admin login
    ("GET", "/login"), ("POST", "/login"), ("POST", "/logout"),
    # the Hailo detector card on Stats and its snapshot thumbnails (auth.ADMIN_PATHS "/hailo")
    ("GET", "/hailo/card"), ("GET", "/hailo/snapshot/{rel:path}"),
    # the visit page's admin actions: corrections, circles and their checks, Ask Claude, stars, privacy calls
    ("POST", "/visit/{visit_id}/photo-frame"), ("POST", "/visit/{visit_id}/correct"), ("GET", "/species-search"),
    ("POST", "/visit/{visit_id}/ask-species"), ("POST", "/visit/{visit_id}/check-circle/{source}"),
    ("GET", "/species-photo"), ("POST", "/visit/{visit_id}/clear-correction"), ("POST", "/visit/{visit_id}/reviewed"),
    ("POST", "/visit/{visit_id}/unconfirm"), ("POST", "/visit/{visit_id}/keep"), ("POST", "/visit/{visit_id}/public"),
    ("POST", "/visit/{visit_id}/private"), ("POST", "/visit/{visit_id}/delete"),
    ("GET", "/api/photo-reminder"), ("POST", "/species/{name}/photo"), ("GET", "/species/{name}/cover-compare"),
    ("POST", "/species-info/{name}/sound-broken"),
    # Review, False triggers, Claude's Notes and Lessons
    ("GET", "/review"), ("GET", "/false-triggers"), ("GET", "/false-triggers/cards"), ("GET", "/false-triggers/more"),
    ("POST", "/false-triggers/confirm"), ("POST", "/false-triggers/clean"),
    ("GET", "/notes"), ("POST", "/notes"), ("POST", "/notes/refine"), ("POST", "/notes/summary/distill"),
    ("POST", "/notes/summary/accept"), ("POST", "/notes/summary/discard"),
    ("GET", "/lessons"), ("POST", "/lessons/{visit_id}/edit"), ("POST", "/lessons/{visit_id}/{state}"),
    # Settings: Pi checks, alerts, the mirror, storage, camera, zones, calibration pictures
    ("GET", "/settings"), ("POST", "/settings/presence"), ("POST", "/settings/alerts/test"), ("POST", "/settings/alerts"),
    ("POST", "/settings/mirror"), ("POST", "/settings/storage"), ("POST", "/settings/storage/limits"),
    ("GET", "/settings/storage/preview"), ("POST", "/settings/storage/run"), ("GET", "/settings/storage/run"),
    ("GET", "/settings/camera"), ("POST", "/settings/camera/picture"), ("POST", "/settings/camera/daynight"),
    ("POST", "/settings/camera/undo/{change_id}"), ("POST", "/settings/camera/reboot"),
    ("GET", "/settings/zones"), ("GET", "/settings/zones/status"), ("POST", "/settings/zones/now"),
    ("POST", "/settings/zones"), ("POST", "/settings/zones/undo/{cal_id}"), ("POST", "/settings/zones/manual"),
    ("GET", "/calibration/{name}"),
    # Guests: the admin's page, and the mirror's counts feed, which opens only to its secret token (a 404 without)
    ("GET", "/guests"), ("GET", "/api/audience-counts"),
}
