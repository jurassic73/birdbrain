"""Shared helpers for the species photo tests (used by more than one test module)."""
from birdbrain import db


def photo(conn, label):
    return db.species_row(conn, label)["thumb_path"]
