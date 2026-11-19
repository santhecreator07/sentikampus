
import json
import os
import urllib.error
import urllib.request
import streamlit as st
# ==============================
# KONFIGURASI API
# ==============================
API_BASE_URL = os.getenv(
    "API_BASE_URL",
    "http://127.0.0.1:8000"
).rstrip("/")
# ==============================
# FUNGSI UNTUK MENGIRIM REQUEST
# ==============================
def post_json(path: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")

    request = urllib.request.Request(
        url=f"{API_BASE_URL}{path}",
        data=body,
        headers={
            "Content-Type": "application/json"
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=8) as response:
        return json.loads(
            response.read().decode("utf-8")
        )
# ==============================
# KONFIGURASI HALAMAN STREAMLIT
# ==============================
st.set_page_config(
    page_title="SentiKampus Starter",
    page_icon="🎓",
    layout="centered",
)
st.title("SentiKampus — Starter Web")
st.write("Aplikasi analisis sentimen masukan mahasiswa.")
# ==============================
# INPUT PENGGUNA
# ==============================
text = st.text_area(
    "Masukan mahasiswa",
    value="Pelayanan akademik sangat membantu",
)
week = st.selectbox(
    "Konteks pertemuan",
    options=list(range(1, 17)),
    index=3,
)

# ==============================
# ANALISIS SENTIMEN
# ==============================
if st.button("Analisis Sentimen", type="primary"):
    if len(text.strip()) < 3:
        st.warning("Teks minimal tiga karakter.")

    else:
        try:
            result = post_json(
                "/api/v1/predict",
                {
                    "text": text.strip(),
                    "week": int(week),
                    "include_explanation": False,
                },
            )
            st.success(
                f"Hasil: {result['label'].upper()}"
            )
            st.write(
                f"Confidence: {result['score'] * 100:.1f}%"
            )
            with st.expander("Lihat detail hasil API"):
                st.json(result)
        except urllib.error.HTTPError as exc:
            st.error(
                f"API mengembalikan HTTP {exc.code}"
            )
        except urllib.error.URLError:
            st.error(
                "Tidak dapat terhubung ke API. "
                "Pastikan server API sudah berjalan."
            )
        except (KeyError, TypeError, ValueError) as exc:
            st.error(
                f"Format respons API tidak sesuai: {exc}"
            )
        except Exception as exc:
            st.error(
                f"Prediksi gagal: {exc}"
            )
