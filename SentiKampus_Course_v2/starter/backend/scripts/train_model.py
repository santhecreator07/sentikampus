"""Script CLI untuk melatih model SentiKampus dan menyimpan artifact.

Penggunaan:
    python scripts/train_model.py
    python scripts/train_model.py --output-dir custom/path

Artifact yang dihasilkan:
    models/sentiment.pkl   — pipeline TF-IDF + Logistic Regression
    models/metadata.json   — informasi model (versi, waktu pelatihan, dll.)

Script ini harus dijalankan dari direktori starter/backend/ atau
menggunakan path absolut.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Tambahkan direktori backend ke sys.path agar import app.* berfungsi.
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.training import persist_demo_model  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Melatih model demonstrasi SentiKampus (TF-IDF + Logistic Regression).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Contoh:\n"
            "  python scripts/train_model.py\n"
            "  python scripts/train_model.py --output-dir custom/models\n"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=BACKEND_DIR / "models",
        help="Folder penyimpanan sentiment.pkl dan metadata.json (default: models/).",
    )
    args = parser.parse_args()

    csv_path = BACKEND_DIR / "data" / "training_samples.csv"
    if not csv_path.exists():
        print(f"ERROR: Dataset tidak ditemukan di {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Dataset   : {csv_path}")
    print(f"Output    : {args.output_dir.resolve()}")
    print("Melatih model …")

    metadata = persist_demo_model(
        csv_path=csv_path,
        model_path=args.output_dir / "sentiment.pkl",
        metadata_path=args.output_dir / "metadata.json",
    )

    print("\nModel demonstrasi berhasil dibuat.")
    print(f"   Versi       : {metadata['model_version']}")
    print(f"   Pelatihan   : {metadata['training_samples']} sampel")
    print(f"   Kelas       : {metadata['labels']}")
    print(f"   Scikit-learn: {metadata['scikit_learn_version']}")
    print(f"   Lokasi      : {args.output_dir.resolve()}")
    print(
        "\nPERINGATAN: Dataset ini bersifat demonstrasi (36 sampel). "
        "Jangan gunakan untuk keputusan administratif."
    )


if __name__ == "__main__":
    main()
