"""Modul pelatihan model SentiKampus.

Menyediakan tiga fungsi utama:
  - read_samples()         → membaca CSV dan memvalidasi dataset
  - build_demo_pipeline()  → membangun pipeline TF-IDF + Logistic Regression
  - persist_demo_model()   → melatih, mengevaluasi, dan menyimpan artifact

Pemisahan training/inference penting: modul ini hanya dipanggil saat model
belum ada atau saat skrip train_model.py dijalankan secara eksplisit.
ModelService (model_service.py) bertanggung jawab atas inference.
"""
from __future__ import annotations

import csv
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

UTC = timezone.utc


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def read_samples(csv_path: Path) -> tuple[list[str], list[str]]:
    """Baca CSV, kembalikan (texts, labels).

    Kolom yang diharapkan: text, label
    Peringatan jika dataset terlalu kecil untuk evaluasi yang valid.

    Raises:
        ValueError: jika data tidak mencukupi atau kolom tidak sesuai.
    """
    texts: list[str] = []
    labels: list[str] = []

    with csv_path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            text = row.get("text", "").strip()
            label = row.get("label", "").strip()
            if text and label:
                texts.append(text)
                labels.append(label)

    n_samples = len(texts)
    n_classes = len(set(labels))

    if n_samples < 9:
        raise ValueError(
            f"Dataset terlalu kecil ({n_samples} baris). "
            "Butuh minimal 9 contoh agar cross-validation dapat berjalan."
        )
    if n_classes < 3:
        raise ValueError(
            f"Dataset hanya memiliki {n_classes} kelas unik. "
            "SentiKampus memerlukan minimal 3 kelas: positif, negatif, netral."
        )

    if n_samples < 100:
        import warnings

        warnings.warn(
            f"Dataset kecil ({n_samples} sampel). "
            "Evaluasi bersifat demonstrasi; jangan klaim sebagai performa produksi.",
            UserWarning,
            stacklevel=2,
        )

    return texts, labels


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def build_demo_pipeline() -> Pipeline:
    """Bangun pipeline TF-IDF + Logistic Regression yang belum di-fit.

    Fungsi ini dipisahkan dari training agar evaluasi (cross-validation)
    dapat memakai definisi yang sama tanpa data leakage.

    Hiperparameter:
      - TfidfVectorizer: unigram + bigram, sublinear TF, lowercase
      - LogisticRegression: balanced class weight, max_iter=1000, solver='lbfgs'
    """
    return Pipeline(
        steps=[
            (
                "vectorizer",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=1,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                    solver="lbfgs",
                ),
            ),
        ]
    )


def train_demo_pipeline(csv_path: Path) -> Pipeline:
    """Baca dataset, fit pipeline, dan kembalikan pipeline yang sudah dilatih."""
    texts, labels = read_samples(csv_path)
    pipeline = build_demo_pipeline()
    pipeline.fit(texts, labels)
    return pipeline


# ---------------------------------------------------------------------------
# Serialisasi artifact
# ---------------------------------------------------------------------------


def persist_demo_model(
    csv_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> dict[str, Any]:
    """Latih model, simpan artifact .pkl, dan tulis metadata.json.

    Artifact berisi dict {"pipeline": Pipeline, "metadata": dict}.
    Model tidak pernah dimuat dari sumber eksternal yang tidak dipercaya.

    Returns:
        dict metadata yang juga ditulis ke metadata_path.
    """
    pipeline = train_demo_pipeline(csv_path)
    texts, labels = read_samples(csv_path)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    metadata: dict[str, Any] = {
        "model_name": "sentikampus-tfidf-logistic-regression",
        "model_version": "demo-2.0.0",
        "trained_at": datetime.now(UTC).isoformat(),
        "training_samples": len(texts),
        "labels": sorted(set(labels)),
        "scikit_learn_version": sklearn.__version__,
        "algorithm": "TF-IDF (unigram+bigram) + Logistic Regression (balanced)",
        "purpose": "Demonstrasi pembelajaran; bukan untuk keputusan administratif.",
    }

    artifact = {"pipeline": pipeline, "metadata": metadata}

    with model_path.open("wb") as handle:
        # noqa: S301 — artifact dibuat secara lokal oleh proyek ini, bukan dari sumber eksternal.
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)

    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return metadata
