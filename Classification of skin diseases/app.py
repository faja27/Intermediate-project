# =========================================================
# Streamlit App - Skin Disease Detection (MobileNetV2)
# =========================================================

# Import library utama untuk membuat web app
import streamlit as st

# Import library untuk operasi file/path
import os
import json

# Import numpy untuk manipulasi array numerik
import numpy as np

# Import PIL untuk membuka gambar upload user
from PIL import Image

# Import TensorFlow/Keras untuk load model dan inferensi
import tensorflow as tf
from tensorflow import keras

# Import preprocess_input khusus MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


# ---------------------------------------------------------
# Konfigurasi dasar halaman Streamlit
# ---------------------------------------------------------

# Mengatur judul tab browser + layout halaman
st.set_page_config(
    page_title="Skin Disease Detection",
    page_icon="🩺",
    layout="centered"
)

# Menampilkan judul utama di halaman app
st.title("🩺 Skin Disease Detection from Images")

# Menampilkan deskripsi singkat app
st.write(
    "Upload a skin image, then the model will predict one of these classes: "
    "**acne, eksim, herpes, panu, rosacea**."
)

# Menampilkan catatan penting (disclaimer)
st.warning(
    "Disclaimer: This tool is for educational/portfolio purposes only and "
    "is NOT a medical diagnosis tool."
)


# ---------------------------------------------------------
# Konstanta/path model & class file
# ---------------------------------------------------------

# Nama file model (asumsikan ada di folder yang sama dengan streamlit_app.py)
MODEL_PATH = "skin_disease_mobilenetv2.keras"

# Nama file class mapping (opsional tapi direkomendasikan)
CLASS_JSON_PATH = "class_names.json"

# Ukuran input gambar sesuai model MobileNetV2
IMG_SIZE = (224, 224)


# ---------------------------------------------------------
# Fungsi utilitas: load class names
# ---------------------------------------------------------

def load_class_names():
    """
    Memuat daftar nama kelas dari class_names.json jika tersedia.
    Jika file tidak ada / rusak, pakai default list.
    """
    # Cek apakah file class_names.json ada
    if os.path.exists(CLASS_JSON_PATH):
        try:
            # Buka file JSON mode read
            with open(CLASS_JSON_PATH, "r") as f:
                # Parse JSON jadi python list
                class_names = json.load(f)

            # Validasi sederhana: pastikan hasilnya list dan tidak kosong
            if isinstance(class_names, list) and len(class_names) > 0:
                return class_names
        except Exception:
            # Jika ada error parsing, lanjut fallback default
            pass

    # Fallback default jika JSON tidak tersedia
    return ["acne", "eksim", "herpes", "panu", "rosacea"]


# ---------------------------------------------------------
# Fungsi utilitas: load model (dengan cache)
# ---------------------------------------------------------

@st.cache_resource
def load_model():
    """
    Memuat model Keras sekali saja (cache),
    agar app lebih cepat saat prediksi berulang.
    """
    # Cek file model ada atau tidak
    if not os.path.exists(MODEL_PATH):
        # Kalau tidak ada, raise error agar user tahu file belum tersedia
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Please place your .keras model in the same folder as streamlit_app.py"
        )

    # Load model Keras dari file .keras
    model = keras.models.load_model(MODEL_PATH)

    # Kembalikan model yang sudah diload
    return model


# ---------------------------------------------------------
# Fungsi preprocessing gambar
# ---------------------------------------------------------

def preprocess_pil_image(pil_img: Image.Image):
    """
    Mengubah PIL image menjadi tensor batch siap prediksi:
    1) convert RGB
    2) resize 224x224
    3) image -> array float32
    4) tambah batch dimension
    5) preprocess_input MobileNetV2
    """
    # Pastikan gambar dalam mode RGB (3 channel)
    img = pil_img.convert("RGB")

    # Resize gambar sesuai input model
    img = img.resize(IMG_SIZE)

    # Ubah PIL image jadi numpy array float32
    arr = np.array(img).astype("float32")

    # Tambah dimensi batch: (224,224,3) -> (1,224,224,3)
    arr = np.expand_dims(arr, axis=0)

    # Terapkan preprocessing khusus MobileNetV2
    arr = preprocess_input(arr)

    # Return array final siap inferensi
    return arr


# ---------------------------------------------------------
# Load resource utama (model + class names)
# ---------------------------------------------------------

# Muat daftar kelas
class_names = load_class_names()

# Coba load model, tampilkan error user-friendly jika gagal
try:
    model = load_model()
    st.success("Model loaded successfully ✅")
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()  # hentikan app kalau model gagal load


# ---------------------------------------------------------
# Komponen upload image
# ---------------------------------------------------------

# Buat uploader yang menerima file gambar
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp", "bmp"]
)

# Jika user sudah upload file
if uploaded_file is not None:
    # Buka file upload sebagai PIL image
    image = Image.open(uploaded_file)

    # Tampilkan preview gambar upload
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Tombol untuk trigger prediksi
    if st.button("Predict"):
        # Tampilkan spinner saat proses prediksi
        with st.spinner("Running inference..."):
            # Preprocess gambar jadi format input model
            x = preprocess_pil_image(image)

            # Jalankan prediksi model (probabilitas per kelas)
            probs = model.predict(x, verbose=0)[0]

            # Ambil index kelas dengan probabilitas tertinggi
            pred_idx = int(np.argmax(probs))

            # Ambil label kelas prediksi
            pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

            # Ambil confidence kelas prediksi (0-100%)
            confidence = float(probs[pred_idx] * 100.0)

        # Tampilkan hasil utama prediksi
        st.subheader("Prediction Result")
        st.write(f"**Predicted class:** {pred_label}")
        st.write(f"**Confidence:** {confidence:.2f}%")

        # Tampilkan confidence sebagai progress bar (0..1)
        st.progress(min(max(confidence / 100.0, 0.0), 1.0))

        # Tampilkan probabilitas semua kelas agar transparan
        st.subheader("Class Probabilities")
        for i, cls in enumerate(class_names):
            # Ambil probabilitas kelas ke-i, aman kalau index melebihi output model
            p = float(probs[i] * 100.0) if i < len(probs) else 0.0
            st.write(f"- {cls}: {p:.2f}%")

        # Menampilkan info tambahan
        st.info(
            "Tip: Use clear, close-up, well-lit images for more stable predictions."
        )
else:
    # Pesan instruksi jika belum ada gambar diupload
    st.write("Please upload an image to start prediction.")
