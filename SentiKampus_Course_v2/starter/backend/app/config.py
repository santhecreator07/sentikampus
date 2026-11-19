"""Konfigurasi terpusat SentiKampus Backend.

Semua path dan environment variable dibaca di sini agar tidak tersebar
di berbagai modul. Gunakan get_settings() untuk mendapatkan instance Settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


# Direktori root backend (satu level di atas folder app/).
BACKEND_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    """Konfigurasi runtime yang dibaca dari environment variable."""

    model_path: Path
    metadata_path: Path
    allowed_origins: list[str]
    log_level: str


def _resolve_path(value: str | None, fallback: Path) -> Path:
    """Kembalikan Path absolut; jika relatif, resolve dari parent BACKEND_DIR."""
    if not value:
        return fallback
    path = Path(value)
    return path if path.is_absolute() else (BACKEND_DIR.parent / path).resolve()


def get_settings() -> Settings:
    """Baca environment variable dan kembalikan objek Settings."""
    origins_raw = os.getenv("ALLOWED_ORIGINS", "*")
    return Settings(
        model_path=_resolve_path(
            os.getenv("MODEL_PATH"),
            BACKEND_DIR / "models" / "sentiment.pkl",
        ),
        metadata_path=_resolve_path(
            os.getenv("METADATA_PATH"),
            BACKEND_DIR / "models" / "metadata.json",
        ),
        allowed_origins=[
            item.strip() for item in origins_raw.split(",") if item.strip()
        ],
        log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
    )
