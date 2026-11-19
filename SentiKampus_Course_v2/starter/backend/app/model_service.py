"""ModelService — pemuatan dan inferensi model SentiKampus.

Bertanggung jawab atas:
  - Memuat artifact .pkl (atau melatih jika belum ada).
  - Menyediakan metode predict() yang aman dan teruji.
  - Menghasilkan penjelasan fitur sederhana (XAI linear).

ModelService TIDAK melakukan training ulang pada setiap request API.
Training hanya terjadi sekali saat load_or_train() dipanggil dan
file model belum ada.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from app.training import persist_demo_model


LOGGER = logging.getLogger(__name__)

# Jumlah fitur teratas yang ditampilkan dalam penjelasan XAI.
_MAX_EXPLANATION_TERMS = 5


class ModelService:
    """Layanan inferensi model sentimen.

    Memuat artifact yang dibuat secara lokal oleh proyek ini sendiri.
    Jangan pernah memuat file pickle dari sumber yang tidak dipercaya.
    """

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        data_path: Path,
    ) -> None:
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.data_path = data_path
        self.pipeline: Pipeline | None = None
        self.metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load_or_train(self) -> None:
        """Muat artifact jika sudah ada; latih dan simpan jika belum.

        Dipanggil sekali saat startup FastAPI (lifespan context).
        """
        if not self.model_path.exists():
            LOGGER.info(
                "Model belum tersedia di %s. Melatih model demonstrasi lokal …",
                self.model_path,
            )
            persist_demo_model(
                csv_path=self.data_path,
                model_path=self.model_path,
                metadata_path=self.metadata_path,
            )

        LOGGER.info("Memuat artifact model dari %s …", self.model_path)
        # Artifact dibuat lokal oleh proyek ini — aman untuk dimuat.
        with self.model_path.open("rb") as handle:
            artifact = pickle.load(handle)  # noqa: S301

        self.pipeline = artifact["pipeline"]
        self.metadata = artifact.get("metadata", {})

        # Gabungkan metadata dari disk jika ada (lebih lengkap).
        if self.metadata_path.exists():
            disk_meta = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            self.metadata.update(disk_meta)

        LOGGER.info(
            "Model '%s' berhasil dimuat. Kelas: %s",
            self.version,
            self.metadata.get("labels", []),
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        """True jika pipeline model sudah dimuat dan siap melakukan prediksi."""
        return self.pipeline is not None

    @property
    def version(self) -> str:
        """Versi model yang dimuat, atau 'unknown' jika metadata kosong."""
        return str(self.metadata.get("model_version", "unknown"))

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        text: str,
        include_explanation: bool = False,
    ) -> dict[str, Any]:
        """Prediksi sentimen satu teks.

        Args:
            text: Teks input yang sudah dinormalisasi (dari schema validator).
            include_explanation: Jika True, sertakan kontribusi fitur model.

        Returns:
            dict berisi: label, score, probabilities, explanation.

        Raises:
            RuntimeError: jika model belum dimuat (load_or_train belum dipanggil).
        """
        if self.pipeline is None:
            raise RuntimeError(
                "ModelService.predict() dipanggil sebelum load_or_train(). "
                "Pastikan ModelService.load_or_train() dipanggil saat startup."
            )

        # predict_proba mengembalikan array [n_samples, n_classes].
        probabilities: np.ndarray = self.pipeline.predict_proba([text])[0]
        classifier = self.pipeline.named_steps["classifier"]

        best_index = int(np.argmax(probabilities))
        label = str(classifier.classes_[best_index])

        probability_map = {
            str(cls): round(float(prob), 6)
            for cls, prob in zip(classifier.classes_, probabilities)
        }

        explanation = (
            self._explain(text, best_index) if include_explanation else []
        )

        return {
            "label": label,
            "score": float(probabilities[best_index]),
            "probabilities": probability_map,
            "explanation": explanation,
        }

    # ------------------------------------------------------------------
    # XAI (Explainable AI sederhana — kontribusi fitur model linear)
    # ------------------------------------------------------------------

    def _explain(
        self, text: str, class_index: int
    ) -> list[dict[str, float | str]]:
        """Hitung kontribusi fitur untuk kelas terpilih.

        Kontribusi = nilai TF-IDF * koefisien Logistic Regression.
        Hanya token yang muncul dalam teks (TF-IDF ≠ 0) yang dipertimbangkan.

        Catatan pedagogi: ini adalah penjelasan model linear, bukan
        interpretasi psikologis penulis teks.
        """
        if self.pipeline is None:
            return []

        vectorizer = self.pipeline.named_steps["vectorizer"]
        classifier = self.pipeline.named_steps["classifier"]

        # Transform teks ke ruang fitur TF-IDF.
        vector = vectorizer.transform([text])
        feature_names = vectorizer.get_feature_names_out()

        # Koefisien untuk kelas terpilih.
        coefficients = classifier.coef_[class_index]
        dense = vector.toarray()[0]

        # Kontribusi = TF-IDF weight × koefisien.
        contributions = dense * coefficients

        # Hanya fitur yang aktif (nilai TF-IDF ≠ 0) yang relevan.
        active_indices = np.flatnonzero(dense)
        ranked = sorted(
            active_indices,
            key=lambda idx: abs(contributions[idx]),
            reverse=True,
        )[:_MAX_EXPLANATION_TERMS]

        return [
            {
                "term": str(feature_names[idx]),
                "contribution": round(float(contributions[idx]), 4),
            }
            for idx in ranked
        ]
