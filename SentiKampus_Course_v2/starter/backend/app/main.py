"""FastAPI application factory untuk SentiKampus.

Pola factory (create_app) memungkinkan pengujian dengan settings berbeda
tanpa memulai ulang proses uvicorn. Semua dependensi (ModelService,
MetricsStore) dibuat di dalam factory agar mudah di-mock dalam test.

Endpoint yang tersedia:
  GET  /                            → info umum
  GET  /health                      → status model
  POST /api/v1/predict              → prediksi sentimen
  GET  /api/v1/metrics              → snapshot monitoring
  POST /api/v1/metrics/reset        → reset metrik (demo)
  GET  /api/v1/course/weeks         → daftar 16 pertemuan
  GET  /api/v1/course/weeks/{week}  → detail satu pertemuan
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import BACKEND_DIR, Settings, get_settings
from app.course_weeks import COURSE_WEEKS, get_week
from app.model_service import ModelService
from app.monitoring import MetricsStore
from app.schemas import HealthResponse, PredictionRequest, PredictionResponse


def create_app(settings: Settings | None = None) -> FastAPI:
    """Buat instance FastAPI dengan semua komponen yang terhubung.

    Args:
        settings: Override settings untuk pengujian. Jika None, baca dari env.

    Returns:
        Instance FastAPI yang siap dijalankan.
    """
    active_settings = settings or get_settings()
    logging.basicConfig(level=active_settings.log_level)

    metrics = MetricsStore()
    model_service = ModelService(
        model_path=active_settings.model_path,
        metadata_path=active_settings.metadata_path,
        data_path=BACKEND_DIR / "data" / "training_samples.csv",
    )

    # ------------------------------------------------------------------
    # Lifespan: startup + shutdown
    # ------------------------------------------------------------------
    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        """Muat model saat startup; bersihkan sumber daya saat shutdown."""
        model_service.load_or_train()
        application.state.model_service = model_service
        application.state.metrics = metrics
        yield
        # Tidak ada resource yang perlu dibersihkan saat ini.

    # ------------------------------------------------------------------
    # Buat aplikasi
    # ------------------------------------------------------------------
    application = FastAPI(
        title="SentiKampus API",
        version="1.0.0",
        description=(
            "API demonstrasi analisis sentimen masukan mahasiswa. "
            "Model bersifat edukatif dan tidak untuk keputusan administratif. "
            "Gunakan input anonim."
        ),
        lifespan=lifespan,
    )

    # CORS — terbuka untuk pengembangan lokal dan emulator Android.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    # Middleware observabilitas (trace_id + X-Process-Time-Ms)
    # ------------------------------------------------------------------
    @application.middleware("http")
    async def attach_observability(request: Request, call_next):
        """Tambahkan trace_id dan catat waktu pemrosesan setiap request."""
        trace_id = request.headers.get("X-Trace-Id", str(uuid.uuid4()))
        request.state.trace_id = trace_id
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        response.headers["X-Trace-Id"] = trace_id
        response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
        return response

    # ------------------------------------------------------------------
    # Endpoint: sistem
    # ------------------------------------------------------------------
    @application.get("/", tags=["system"])
    def root() -> dict[str, str]:
        """Halaman root — arahkan pengguna ke /docs."""
        return {
            "name": "SentiKampus API",
            "message": "Buka /docs untuk mencoba API secara interaktif.",
        }

    @application.get("/health", response_model=HealthResponse, tags=["system"])
    def health() -> HealthResponse:
        """Periksa kesehatan layanan dan status model."""
        return HealthResponse(
            status="ok" if model_service.is_loaded else "degraded",
            model_loaded=model_service.is_loaded,
            model_version=model_service.version,
        )

    # ------------------------------------------------------------------
    # Endpoint: pertemuan
    # ------------------------------------------------------------------
    @application.get("/api/v1/course/weeks", tags=["course"])
    def list_course_weeks() -> list[dict[str, object]]:
        """Kembalikan metadata semua 16 pertemuan."""
        return COURSE_WEEKS

    @application.get("/api/v1/course/weeks/{week}", tags=["course"])
    def course_week_detail(week: int) -> dict[str, object]:
        """Kembalikan metadata satu pertemuan berdasarkan nomornya (1–16)."""
        result = get_week(week)
        if result is None:
            raise HTTPException(
                status_code=404,
                detail=f"Pertemuan {week} tidak ditemukan. Nomor valid: 1–16.",
            )
        return result

    # ------------------------------------------------------------------
    # Endpoint: prediksi sentimen
    # ------------------------------------------------------------------
    @application.post(
        "/api/v1/predict",
        response_model=PredictionResponse,
        tags=["sentiment"],
        summary="Analisis sentimen teks masukan mahasiswa",
    )
    def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
        """Prediksi sentimen menggunakan pipeline TF-IDF + Logistic Regression.

        - Validasi dilakukan oleh Pydantic (schema PredictionRequest).
        - Model tidak dilatih pada setiap request; artifact dimuat saat startup.
        - Penjelasan XAI disertakan jika include_explanation=True atau week >= 7.
        """
        started = time.perf_counter()
        trace_id = request.state.trace_id

        try:
            week_data = get_week(payload.week)
            if week_data is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Pertemuan {payload.week} tidak ditemukan.",
                )

            # Aktifkan XAI jika diminta eksplisit atau mulai pertemuan 7.
            use_xai = payload.include_explanation or payload.week >= 7

            result = model_service.predict(payload.text, include_explanation=use_xai)

            latency_ms = (time.perf_counter() - started) * 1000
            metrics.record(
                label=str(result["label"]),
                latency_ms=latency_ms,
                success=True,
                trace_id=trace_id,
            )

            return PredictionResponse(
                label=str(result["label"]),
                score=float(result["score"]),
                probabilities={
                    str(k): float(v) for k, v in result["probabilities"].items()
                },
                model_version=model_service.version,
                latency_ms=round(latency_ms, 2),
                explanation=result["explanation"],
                capabilities=list(week_data["capabilities"]),
                trace_id=trace_id,
            )

        except HTTPException:
            raise  # Teruskan HTTPException tanpa dibungkus.
        except Exception as exc:
            latency_ms = (time.perf_counter() - started) * 1000
            metrics.record("error", latency_ms, False, trace_id)
            logging.exception("Prediksi gagal untuk trace_id=%s", trace_id)
            raise HTTPException(
                status_code=500,
                detail="Prediksi gagal diproses. Periksa log server.",
            ) from exc

    # ------------------------------------------------------------------
    # Endpoint: monitoring
    # ------------------------------------------------------------------
    @application.get("/api/v1/metrics", tags=["monitoring"])
    def current_metrics() -> dict[str, object]:
        """Kembalikan snapshot metrik: latensi, error rate, distribusi label."""
        return metrics.snapshot()

    @application.post("/api/v1/metrics/reset", tags=["monitoring"])
    def reset_metrics() -> dict[str, str]:
        """Reset semua metrik demonstrasi (tidak memengaruhi model)."""
        metrics.reset()
        return {"message": "Metrik demonstrasi telah direset."}

    return application


# Buat instance aplikasi untuk uvicorn.
app = create_app()
