"""Metadata 16 pertemuan Mata Kuliah Kecerdasan Buatan (Modul 788).

Setiap entri mendefinisikan:
  - week          : nomor pertemuan (1–16)
  - title         : judul pertemuan
  - level         : level kesulitan
  - focus         : tujuan pembelajaran utama
  - demo_action   : langkah demonstrasi untuk dosen/mahasiswa
  - evidence      : bukti ketercapaian
  - capabilities  : daftar fitur API yang diaktifkan pada pertemuan ini

Field 'capabilities' digunakan oleh endpoint predict untuk menentukan
fitur apa yang dikembalikan dalam response (mis. xai dimulai di P7).
"""
from __future__ import annotations

from typing import Any


COURSE_WEEKS: list[dict[str, Any]] = [
    {
        "week": 1,
        "title": "Pengantar AI Software Engineering dan Arsitektur Sistem",
        "level": "Dasar",
        "focus": "Mengenali masalah, pengguna, input, proses AI, dan output sebelum menulis kode.",
        "demo_action": "Buka tab Arsitektur dan telusuri aliran teks dari frontend sampai hasil prediksi.",
        "evidence": "Mahasiswa mampu menjelaskan fungsi setiap komponen dengan kalimat sendiri.",
        "capabilities": ["architecture", "offline_demo"],
    },
    {
        "week": 2,
        "title": "REST API dengan FastAPI",
        "level": "Dasar",
        "focus": "Memahami endpoint, metode HTTP, request, response, dan status code.",
        "demo_action": "Buka /docs, kirim POST /api/v1/predict, lalu baca JSON.",
        "evidence": "Mahasiswa dapat membedakan GET dan POST serta menjelaskan struktur JSON.",
        "capabilities": ["architecture", "api"],
    },
    {
        "week": 3,
        "title": "Model Inference API dan Serialisasi",
        "level": "Dasar ke Menengah",
        "focus": "Melatih model sederhana, menyimpan pipeline, memuatnya kembali, lalu melakukan inference.",
        "demo_action": "Jalankan train_model.py, periksa metadata.json, lalu bandingkan prediksi sebelum dan sesudah restart.",
        "evidence": "Mahasiswa memahami perbedaan proses pelatihan dan proses prediksi.",
        "capabilities": ["api", "model", "serialization"],
    },
    {
        "week": 4,
        "title": "Frontend UI untuk Aplikasi AI",
        "level": "Menengah",
        "focus": "Mengubah kontrak API menjadi antarmuka yang mudah digunakan.",
        "demo_action": "Jalankan Streamlit, masukkan masukan mahasiswa, tekan Analisis Sentimen.",
        "evidence": "Mahasiswa mampu menghubungkan input UI, pemanggilan API, dan hasil prediksi.",
        "capabilities": ["api", "model", "streamlit_ui"],
    },
    {
        "week": 5,
        "title": "Containerization dengan Docker",
        "level": "Menengah",
        "focus": "Membungkus kode, dependency, dan perintah mulai agar aplikasi konsisten.",
        "demo_action": "Jalankan docker compose up --build, akses /health, lakukan prediksi.",
        "evidence": "Mahasiswa dapat menjelaskan image, container, port, dan environment variable.",
        "capabilities": ["api", "model", "docker"],
    },
    {
        "week": 6,
        "title": "Cloud Deployment",
        "level": "Menengah",
        "focus": "Menempatkan backend pada layanan cloud dan menghubungkan frontend melalui URL publik.",
        "demo_action": "Jalankan frontend dengan API_BASE_URL dan uji endpoint cloud.",
        "evidence": "Mahasiswa dapat menunjukkan URL health dan prediksi dari dua perangkat berbeda.",
        "capabilities": ["api", "model", "docker", "cloud"],
    },
    {
        "week": 7,
        "title": "Monitoring, Optimasi Latensi, dan XAI",
        "level": "Menengah ke Lanjut",
        "focus": "Mengamati kesehatan layanan, waktu respons, kesalahan, dan alasan sederhana di balik prediksi.",
        "demo_action": "Lakukan beberapa prediksi, buka /api/v1/metrics, tampilkan kata paling berpengaruh.",
        "evidence": "Mahasiswa mampu membaca p50/p95 dan membedakan prediksi dengan penjelasannya.",
        "capabilities": ["api", "model", "monitoring", "xai"],
    },
    {
        "week": 8,
        "title": "UTS — Demonstrasi Prototipe",
        "level": "Evaluasi",
        "focus": "Menunjukkan alur aplikasi dari input, API, model, sampai output secara utuh.",
        "demo_action": "Demonstrasikan tiga kelas sentimen, satu skenario gagal, arsitektur, dan pembagian tugas.",
        "evidence": "Prototipe berjalan dan keputusan desain dapat dijelaskan.",
        "capabilities": ["api", "model", "streamlit_ui", "monitoring", "assessment"],
    },
    {
        "week": 9,
        "title": "Integration Testing",
        "level": "Menengah ke Lanjut",
        "focus": "Memastikan komponen yang terhubung menghasilkan perilaku yang diharapkan.",
        "demo_action": "Jalankan pytest, uji endpoint terintegrasi, ubah satu input invalid.",
        "evidence": "Mahasiswa dapat menjelaskan unit test dan integration test beserta satu contohnya.",
        "capabilities": ["api", "model", "streamlit_ui", "testing"],
    },
    {
        "week": 10,
        "title": "Penyempurnaan UI/UX dan Visualisasi Prediksi",
        "level": "Lanjut",
        "focus": "Menyempurnakan web UI, visualisasi probabilitas, loading state, error state, aksesibilitas.",
        "demo_action": "Perbaiki Streamlit agar probabilitas, confidence, error, dan status API terbaca jelas.",
        "evidence": "Mahasiswa mampu menjelaskan loading state, error state, aksesibilitas, dan visualisasi probabilitas.",
        "capabilities": ["api", "model", "streamlit_ui", "visualization", "accessibility"],
    },
    {
        "week": 11,
        "title": "Laporan Kemajuan: Arsitektur dan API",
        "level": "Evaluasi Proyek",
        "focus": "Mengevaluasi konsistensi arsitektur, kontrak API, dan batas tanggung jawab komponen.",
        "demo_action": "Cocokkan diagram arsitektur dengan endpoint dan file yang ada di repository.",
        "evidence": "Laporan kemajuan menyertakan bukti endpoint berjalan dan keputusan arsitektur.",
        "capabilities": ["architecture", "api", "documentation", "progress_review"],
    },
    {
        "week": 12,
        "title": "Laporan Kemajuan: UI, System Flow, dan Multi-Platform",
        "level": "Evaluasi Proyek",
        "focus": "Mengevaluasi aliran pengguna dan memperluas API yang sama ke Android Flutter.",
        "demo_action": "Jalankan web UI dan Flutter Android; buktikan keduanya memakai kontrak API yang sama.",
        "evidence": "Mahasiswa menunjukkan system flow web dan Android serta perbedaan alamat API emulator.",
        "capabilities": ["streamlit_ui", "flutter_ui", "multi_platform", "system_flow", "progress_review"],
    },
    {
        "week": 13,
        "title": "Stress Test dan Analisis Latensi",
        "level": "Lanjut",
        "focus": "Mengukur perilaku layanan saat menerima banyak request dan mencari bottleneck.",
        "demo_action": "Jalankan Locust, naikkan pengguna simulasi, bandingkan p50, p95, throughput, error rate.",
        "evidence": "Mahasiswa menyajikan hasil ukur sebelum dan sesudah satu optimasi.",
        "capabilities": ["monitoring", "stress_test", "optimization"],
    },
    {
        "week": 14,
        "title": "Dokumentasi Proyek",
        "level": "Lanjut",
        "focus": "Membuat proyek dapat dipasang, dijalankan, diuji, dan dipahami oleh orang lain.",
        "demo_action": "Ikuti README dari komputer bersih tanpa penjelasan lisan dari pembuat.",
        "evidence": "Dokumentasi memuat arsitektur, API, instalasi, pengujian, batas model, dan troubleshooting.",
        "capabilities": ["documentation", "reproducibility"],
    },
    {
        "week": 15,
        "title": "Demonstrasi End-to-End",
        "level": "Integrasi Akhir",
        "focus": "Mengintegrasikan aplikasi, API, model, monitoring, pengujian, dan dokumentasi.",
        "demo_action": "Mulai dari aplikasi belum berjalan, nyalakan semua komponen, prediksi, observasi.",
        "evidence": "Seluruh alur utama berhasil tanpa perbaikan kode mendadak.",
        "capabilities": ["end_to_end", "cloud", "monitoring", "testing"],
    },
    {
        "week": 16,
        "title": "UAS — Presentasi dan Penyerahan Proyek",
        "level": "Evaluasi Akhir",
        "focus": "Mempertanggungjawabkan solusi, bukti kualitas, keterbatasan, etika, dan peluang pengembangan.",
        "demo_action": "Presentasikan masalah, demo live, arsitektur, hasil tes, metrik, keterbatasan, URL repository.",
        "evidence": "Produk, laporan, repository, dan demonstrasi akhir konsisten satu sama lain.",
        "capabilities": ["final_assessment", "ethics", "end_to_end"],
    },
]


def get_week(week: int) -> dict[str, Any] | None:
    """Kembalikan metadata pertemuan berdasarkan nomornya, atau None jika tidak ditemukan."""
    return next((item for item in COURSE_WEEKS if item["week"] == week), None)
