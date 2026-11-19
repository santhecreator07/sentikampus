"""Schema Pydantic untuk request dan response API SentiKampus.

Mendefinisikan kontrak yang digunakan bersama oleh FastAPI, Streamlit,
dan Flutter agar seluruh klien menerima struktur JSON yang konsisten.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """Payload yang dikirim klien ke POST /api/v1/predict."""

    text: str = Field(
        min_length=3,
        max_length=500,
        examples=["Pelayanan akademik sangat membantu"],
        description="Teks masukan mahasiswa yang akan dianalisis sentimennya.",
    )
    week: int = Field(
        default=4,
        ge=1,
        le=16,
        description="Nomor pertemuan (1–16) yang menentukan fitur yang diaktifkan.",
    )
    include_explanation: bool = Field(
        default=False,
        description="Jika True, sertakan kontribusi fitur model (XAI) dalam respons.",
    )

    @field_validator("text")
    @classmethod
    def normalize_text(cls, value: str) -> str:
        """Normalisasi whitespace; pastikan teks bermakna minimal 3 karakter."""
        cleaned = " ".join(value.split())
        if len(cleaned) < 3:
            raise ValueError("Teks harus berisi sedikitnya 3 karakter bermakna.")
        return cleaned


class ExplanationItem(BaseModel):
    """Satu entri kontribusi fitur dari model linear (XAI sederhana)."""

    term: str = Field(description="Kata atau n-gram yang berkontribusi terhadap prediksi.")
    contribution: float = Field(description="Nilai kontribusi (positif = mendukung kelas terpilih).")


class PredictionResponse(BaseModel):
    """Respons lengkap dari POST /api/v1/predict."""

    label: str = Field(description="Kelas sentimen terprediksi: positif, negatif, atau netral.")
    score: float = Field(description="Probabilitas kelas terpilih (0–1).")
    probabilities: dict[str, float] = Field(
        description="Distribusi probabilitas semua kelas untuk visualisasi."
    )
    model_version: str = Field(description="Versi model yang digunakan untuk prediksi ini.")
    latency_ms: float = Field(description="Waktu pemrosesan dalam milidetik.")
    explanation: list[ExplanationItem] = Field(
        description="Kontribusi fitur model; kosong jika include_explanation=False."
    )
    capabilities: list[str] = Field(
        description="Daftar kapabilitas yang aktif pada pertemuan yang dipilih."
    )
    trace_id: str = Field(description="Identifier unik untuk tracing request ini.")
    source: str = Field(default="api", description="Sumber prediksi: 'api' atau 'offline_demo'.")


class HealthResponse(BaseModel):
    """Respons dari GET /health."""

    status: str = Field(description="'ok' jika model dimuat, 'degraded' jika belum.")
    model_loaded: bool = Field(description="True jika pipeline model tersedia dan siap prediksi.")
    model_version: str = Field(description="Versi model yang dimuat.")
