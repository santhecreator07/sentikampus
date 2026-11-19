"""Script CLI untuk mengevaluasi model SentiKampus dengan cross-validation.

Penggunaan:
    python scripts/evaluate_model.py

Output:
    models/evaluation.json — laporan evaluasi (accuracy, F1, confusion matrix)
    Stdout                 — ringkasan hasil

Evaluasi menggunakan StratifiedKFold (3-fold) agar distribusi kelas
tetap proporsional di setiap fold. Hasilnya bersifat demonstrasi;
lihat peringatan dalam output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.evaluation import evaluate_demo_pipeline  # noqa: E402


def main() -> None:
    csv_path = BACKEND_DIR / "data" / "training_samples.csv"
    if not csv_path.exists():
        print(f"ERROR: Dataset tidak ditemukan di {csv_path}", file=sys.stderr)
        sys.exit(1)

    print("Mengevaluasi pipeline dengan 3-fold stratified cross-validation …\n")
    report = evaluate_demo_pipeline(csv_path)

    output_path = BACKEND_DIR / "models" / "evaluation.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nLaporan evaluasi disimpan di: {output_path}")
    print(f"\nPERINGATAN: {report['warning']}")


if __name__ == "__main__":
    main()
