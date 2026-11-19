"""Konfigurasi pytest untuk SentiKampus Backend tests.

Menyediakan fixture 'backend_client' yang dapat digunakan langsung
di test tanpa perlu membuat client secara manual.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture(scope="function")
def backend_client(tmp_path: Path) -> TestClient:
    """Fixture: TestClient dengan model yang di-train ke direktori sementara.

    Setiap test function mendapatkan direktori tmp_path yang bersih.
    """
    settings = Settings(
        model_path=tmp_path / "sentiment.pkl",
        metadata_path=tmp_path / "metadata.json",
        allowed_origins=["*"],
        log_level="WARNING",
    )
    with TestClient(create_app(settings)) as client:
        yield client
