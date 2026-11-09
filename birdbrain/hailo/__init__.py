"""Hailo-8 shadow detector: an always-on COCO object detector on the AI HAT+ that watches the substream beside the
Motion detector and keeps its own notes (birdbrain/hailo/), so the two can be compared before anything is gated.

Shadow only: it never creates visits, never calls Claude, never sends alerts and never changes what the watcher does.
It opens its own RTSP connection, writes to its own SQLite file (config.HAILO_SHADOW_DB) and snapshot folder, and
reads the production database read-only. CLI: python -m birdbrain.hailo_detect.
"""
