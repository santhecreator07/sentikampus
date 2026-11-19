"""Locust stress test untuk SentiKampus API (Pertemuan 13).

Penggunaan:
    locust -f load_test/locustfile.py --host=http://127.0.0.1:8000

Buka http://localhost:8089 untuk menjalankan test dari browser.
"""
from locust import HttpUser, between, task


SAMPLES = [
    "Pelayanan akademik sangat membantu",
    "Internet kampus sering terputus",
    "Kuliah dimulai pukul delapan",
    "Dosen menjelaskan materi dengan jelas",
    "Ruang kelas panas dan tidak nyaman",
]


class SentiKampusUser(HttpUser):
    """Simulasi pengguna yang mengirim prediksi dan memeriksa health."""

    wait_time = between(0.5, 1.5)

    def on_start(self) -> None:
        self.index = 0

    @task(4)
    def predict(self) -> None:
        """Kirim POST /api/v1/predict dengan teks bergantian."""
        text = SAMPLES[self.index % len(SAMPLES)]
        self.index += 1
        self.client.post(
            "/api/v1/predict",
            json={"text": text, "week": 13, "include_explanation": False},
            name="POST /api/v1/predict",
        )

    @task(1)
    def health(self) -> None:
        """Kirim GET /health untuk memeriksa ketersediaan layanan."""
        self.client.get("/health", name="GET /health")
