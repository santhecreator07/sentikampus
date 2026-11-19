"""MetricsStore — monitoring latensi dan distribusi label.

Komponen ini mencatat setiap request prediksi (berhasil atau gagal) dan
menyediakan snapshot ringkasan yang dikembalikan oleh GET /api/v1/metrics.

Implementasi thread-safe menggunakan threading.Lock agar aman dipakai
bersama oleh banyak request uvicorn yang berjalan secara bersamaan.
"""
from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from threading import Lock

UTC = timezone.utc


@dataclass(frozen=True)
class _RequestRecord:
    """Snapshot satu request yang dicatat."""

    timestamp: str
    label: str
    latency_ms: float
    success: bool
    trace_id: str


def _percentile(values: list[float], p: float) -> float:
    """Hitung persentil ke-p dari daftar nilai latensi.

    Menggunakan interpolasi linear agar stabil untuk daftar kecil.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * p
    lower = int(pos)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = pos - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


class MetricsStore:
    """Penyimpanan metrik in-memory dengan batas ukuran (deque berkapasitas tetap).

    Metrik hilang saat proses API dimulai ulang. Untuk persistensi,
    ganti deque dengan backend eksternal (mis. Redis atau Prometheus).
    """

    def __init__(self, max_items: int = 500) -> None:
        self._items: deque[_RequestRecord] = deque(maxlen=max_items)
        self._lock = Lock()

    def record(
        self,
        label: str,
        latency_ms: float,
        success: bool,
        trace_id: str,
    ) -> None:
        """Catat satu request ke dalam store."""
        record = _RequestRecord(
            timestamp=datetime.now(UTC).isoformat(),
            label=label,
            latency_ms=round(latency_ms, 2),
            success=success,
            trace_id=trace_id,
        )
        with self._lock:
            self._items.append(record)

    def snapshot(self) -> dict[str, object]:
        """Kembalikan ringkasan metrik saat ini.

        Mencakup:
          - total_requests, successful_requests, failed_requests
          - error_rate
          - latency_ms (average, p50, p95, maximum)
          - label_distribution
          - recent (10 request terakhir, urutan terbaru dulu)
        """
        with self._lock:
            items = list(self._items)

        successful = [item for item in items if item.success]
        latencies = [item.latency_ms for item in successful]
        label_counts = Counter(item.label for item in successful)
        total = len(items)
        errors = total - len(successful)

        return {
            "total_requests": total,
            "successful_requests": len(successful),
            "failed_requests": errors,
            "error_rate": round(errors / total, 4) if total else 0.0,
            "latency_ms": {
                "average": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
                "p50": round(_percentile(latencies, 0.50), 2),
                "p95": round(_percentile(latencies, 0.95), 2),
                "maximum": round(max(latencies), 2) if latencies else 0.0,
            },
            "label_distribution": dict(label_counts),
            "recent": [asdict(item) for item in list(items)[-10:]][::-1],
        }

    def reset(self) -> None:
        """Hapus semua record (untuk demo/reset di kelas)."""
        with self._lock:
            self._items.clear()
