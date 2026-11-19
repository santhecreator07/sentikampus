"""Modul evaluasi model SentiKampus menggunakan stratified cross-validation.

Digunakan oleh scripts/evaluate_model.py. Output berupa laporan JSON
yang menyertakan peringatan bahwa dataset bersifat demonstrasi.

Fungsi evaluate_demo_pipeline() menggunakan StratifiedKFold agar
distribusi kelas terjaga di setiap fold — penting untuk dataset kecil.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.training import build_demo_pipeline, read_samples


def evaluate_demo_pipeline(csv_path: Path, folds: int = 3) -> dict[str, Any]:
    """Evaluasi pipeline dengan stratified cross-validation.

    Dengan hanya 36 sampel yang ditulis tangan, setiap metrik memiliki
    ketidakpastian yang tinggi. Fungsi ini ditujukan untuk pembelajaran
    cara kerja evaluasi, bukan klaim performa produksi.

    Args:
        csv_path: Path ke file training_samples.csv.
        folds:    Jumlah fold CV (default 3; minimum agar tiap kelas muncul di train).

    Returns:
        dict berisi: accuracy, macro_f1, per_class, confusion_matrix, warning.
    """
    texts, labels = read_samples(csv_path)
    label_order = sorted(set(labels))

    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    pipeline = build_demo_pipeline()

    # cross_val_predict menggunakan pipeline yang di-fit K kali pada K-1 fold.
    # Tidak ada data leakage karena fit hanya terjadi pada training fold.
    predictions = cross_val_predict(pipeline, texts, labels, cv=cv)

    accuracy = accuracy_score(labels, predictions)

    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_order,
        average="macro",
        zero_division=0,
    )
    per_p, per_r, per_f, per_s = precision_recall_fscore_support(
        labels,
        predictions,
        labels=label_order,
        average=None,
        zero_division=0,
    )

    per_class = {
        lbl: {
            "precision": round(float(per_p[i]), 4),
            "recall": round(float(per_r[i]), 4),
            "f1": round(float(per_f[i]), 4),
            "support": int(per_s[i]),
        }
        for i, lbl in enumerate(label_order)
    }

    return {
        "evaluation_type": f"{folds}-fold stratified cross-validation",
        "samples": len(texts),
        "labels": label_order,
        "accuracy": round(float(accuracy), 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "confusion_matrix": confusion_matrix(
            labels, predictions, labels=label_order
        ).tolist(),
        "per_class": per_class,
        "warning": (
            "Dataset ini sangat kecil (36 sampel) dan dibuat untuk demonstrasi. "
            "Metrik tidak boleh dipakai sebagai klaim performa produksi."
        ),
    }
