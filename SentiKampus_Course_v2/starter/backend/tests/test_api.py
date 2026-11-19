"""Integration tests untuk SentiKampus Backend.

Pengujian ini berjalan menggunakan FastAPI TestClient (synchronous)
tanpa perlu menjalankan server secara nyata. Model di-train ke tmp_path
agar setiap test run bersih dan tidak bergantung pada artifact produksi.

Jalankan:
    pytest tests/test_api.py -v
    pytest tests/ -v --tb=short
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def build_client(tmp_path: Path) -> TestClient:
    """Buat TestClient dengan model yang dilatih ke direktori sementara."""
    settings = Settings(
        model_path=tmp_path / "sentiment.pkl",
        metadata_path=tmp_path / "metadata.json",
        allowed_origins=["*"],
        log_level="WARNING",
    )
    return TestClient(create_app(settings))


# ---------------------------------------------------------------------------
# Tests: health dan prediksi dasar
# ---------------------------------------------------------------------------


def test_health_endpoint(tmp_path: Path) -> None:
    """GET /health harus mengembalikan status ok dan model_loaded=True."""
    with build_client(tmp_path) as client:
        response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_version"] != ""


def test_prediction_positive(tmp_path: Path) -> None:
    """Prediksi kalimat positif harus mengembalikan label yang valid."""
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/predict",
            json={"text": "Pelayanan akademik sangat membantu", "week": 4},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"positif", "negatif", "netral"}
    assert 0.0 <= body["score"] <= 1.0
    assert set(body["probabilities"]) == {"positif", "negatif", "netral"}
    # Jumlah probabilitas harus mendekati 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.001


def test_prediction_with_xai(tmp_path: Path) -> None:
    """Prediksi dengan include_explanation=True harus menyertakan explanation."""
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/predict",
            json={
                "text": "Dosen menjelaskan materi dengan jelas",
                "week": 7,
                "include_explanation": True,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["label"] in {"positif", "negatif", "netral"}
    assert isinstance(body["explanation"], list)
    assert len(body["explanation"]) > 0
    for item in body["explanation"]:
        assert "term" in item
        assert "contribution" in item


def test_xai_auto_enabled_from_week_7(tmp_path: Path) -> None:
    """Mulai pertemuan 7, XAI aktif otomatis meski include_explanation=False."""
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/predict",
            json={"text": "Fasilitas laboratorium sangat baik", "week": 7},
        )

    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["explanation"], list)
    assert len(body["explanation"]) > 0


def test_trace_id_in_response_header(tmp_path: Path) -> None:
    """Setiap response harus menyertakan header X-Trace-Id."""
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/predict",
            json={"text": "Internet kampus sering terputus", "week": 4},
        )

    assert response.status_code == 200
    assert "X-Trace-Id" in response.headers
    assert response.json()["trace_id"] != ""


# ---------------------------------------------------------------------------
# Tests: validasi input (422)
# ---------------------------------------------------------------------------


def test_text_too_short_returns_422(tmp_path: Path) -> None:
    """Teks dengan kurang dari 3 karakter bermakna harus ditolak (422)."""
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/predict",
            json={"text": " x ", "week": 4},
        )
    assert response.status_code == 422


def test_empty_text_returns_422(tmp_path: Path) -> None:
    """Teks kosong harus ditolak (422)."""
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/predict",
            json={"text": "", "week": 4},
        )
    assert response.status_code == 422


def test_week_out_of_range_returns_422(tmp_path: Path) -> None:
    """Week di luar 1–16 harus ditolak (422)."""
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/predict",
            json={"text": "Pelayanan baik", "week": 17},
        )
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Tests: monitoring
# ---------------------------------------------------------------------------


def test_metrics_initial_state(tmp_path: Path) -> None:
    """Sebelum ada prediksi, total_requests harus 0."""
    with build_client(tmp_path) as client:
        response = client.get("/api/v1/metrics")

    assert response.status_code == 200
    body = response.json()
    assert body["total_requests"] == 0
    assert body["error_rate"] == 0.0


def test_metrics_records_prediction(tmp_path: Path) -> None:
    """Setelah satu prediksi, total_requests harus 1."""
    with build_client(tmp_path) as client:
        client.post(
            "/api/v1/predict",
            json={"text": "Jadwal kuliah tersusun dengan baik", "week": 4},
        )
        response = client.get("/api/v1/metrics")

    assert response.status_code == 200
    assert response.json()["total_requests"] == 1


def test_metrics_reset(tmp_path: Path) -> None:
    """Setelah reset, total_requests kembali ke 0."""
    with build_client(tmp_path) as client:
        client.post(
            "/api/v1/predict",
            json={"text": "Perpustakaan bersih dan nyaman", "week": 4},
        )
        client.post("/api/v1/metrics/reset")
        response = client.get("/api/v1/metrics")

    assert response.json()["total_requests"] == 0


# ---------------------------------------------------------------------------
# Tests: course weeks
# ---------------------------------------------------------------------------


def test_course_weeks_returns_16_items(tmp_path: Path) -> None:
    """GET /api/v1/course/weeks harus mengembalikan tepat 16 pertemuan."""
    with build_client(tmp_path) as client:
        response = client.get("/api/v1/course/weeks")

    assert response.status_code == 200
    weeks = response.json()
    assert len(weeks) == 16
    # Pastikan nomor pertemuan 1–16 ada.
    week_numbers = {w["week"] for w in weeks}
    assert week_numbers == set(range(1, 17))


def test_course_week_detail_found(tmp_path: Path) -> None:
    """GET /api/v1/course/weeks/7 harus mengembalikan data pertemuan 7."""
    with build_client(tmp_path) as client:
        response = client.get("/api/v1/course/weeks/7")

    assert response.status_code == 200
    body = response.json()
    assert body["week"] == 7
    assert "monitoring" in body["capabilities"]
    assert "xai" in body["capabilities"]


def test_course_week_not_found(tmp_path: Path) -> None:
    """GET /api/v1/course/weeks/99 harus mengembalikan 404."""
    with build_client(tmp_path) as client:
        response = client.get("/api/v1/course/weeks/99")

    assert response.status_code == 404


# ---------------------------------------------------------------------------
# Tests: tiga kelas sentimen
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "Pelayanan akademik sangat membantu",    # cenderung positif
        "Internet kampus sering terputus",        # cenderung negatif
        "Kuliah dimulai pukul delapan",           # cenderung netral
    ],
)
def test_all_three_classes_predicted(tmp_path: Path, text: str) -> None:
    """Setiap prediksi harus menghasilkan label dari salah satu tiga kelas."""
    with build_client(tmp_path) as client:
        response = client.post(
            "/api/v1/predict",
            json={"text": text, "week": 4},
        )

    assert response.status_code == 200
    assert response.json()["label"] in {"positif", "negatif", "netral"}
