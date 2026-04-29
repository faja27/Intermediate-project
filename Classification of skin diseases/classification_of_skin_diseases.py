# =========================================================
# #opening 1 - Mount Drive + Import Library + Konfigurasi Awal
# =========================================================

from google.colab import drive                        # Dipakai agar Colab bisa membaca dataset/model yang kamu simpan di Google Drive
drive.mount('/content/drive')                         # Wajib mount dulu, kalau tidak path /content/drive tidak bisa diakses

import os                                             # Untuk cek path, baca isi folder, dan operasi file dasar lintas OS
import json                                           # Untuk simpan/load metadata (misalnya class_names) agar inferensi konsisten
import random                                         # Untuk kontrol proses acak (shuffle/split) supaya hasil bisa diulang
import shutil                                         # Untuk copy/hapus folder saat menyiapkan struktur data eksperimen
import numpy as np                                    # Fondasi komputasi numerik (array/tensor sebelum masuk model)
import matplotlib.pyplot as plt                       # Untuk visualisasi data dan hasil evaluasi (contoh confusion matrix)

import tensorflow as tf                               # Framework utama deep learning yang dipakai untuk training/inference
from tensorflow import keras                          # API high-level dari TensorFlow agar build model lebih ringkas
from tensorflow.keras import layers                   # Kumpulan layer jaringan saraf (Dense, Dropout, pooling, dll.)

from tensorflow.keras.applications import MobileNetV2 # Backbone pretrained: training lebih cepat & akurat di data terbatas
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # Preprocess wajib agar input sesuai standar MobileNetV2

from sklearn.metrics import classification_report     # Memberi metrik detail per kelas (precision/recall/F1), bukan hanya accuracy
from sklearn.metrics import confusion_matrix          # Menunjukkan pola salah prediksi antar kelas secara jelas

SEED = 42                                             # Satu angka acuan agar eksperimen reproducible (hasil lebih konsisten)
random.seed(SEED)                                     # Mengunci random di Python (mis. urutan shuffle)
np.random.seed(SEED)                                  # Mengunci random di NumPy
tf.random.set_seed(SEED)                              # Mengunci random di TensorFlow

DATASET_DIR = "/content/drive/MyDrive/G Colab File/Classification of skin diseases/train"  # Sumber data utama (folder per kelas)

print("Dataset path exists:", os.path.exists(DATASET_DIR))   # Quick check: pastikan path benar sebelum lanjut proses berikutnya
print("Class folders:", os.listdir(DATASET_DIR))             # Validasi awal: pastikan kelas terdeteksi sesuai ekspektasi

# =========================================================
# #opening 2 - Validasi Dataset (jumlah gambar + file rusak)
# =========================================================

from PIL import Image                                    # Import PIL Image untuk validasi file gambar

VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp") # Daftar ekstensi file yang dianggap gambar valid

classes = sorted([d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))])  # Ambil nama kelas dari subfolder

print("Detected classes:", classes)                      # Cetak nama kelas yang terdeteksi

total_images = 0                                         # Inisialisasi total gambar

for c in classes:                                        # Loop setiap kelas
    class_path = os.path.join(DATASET_DIR, c)            # Bentuk path folder kelas
    files = os.listdir(class_path)                       # Ambil semua nama file dalam folder kelas
    img_files = [f for f in files if f.lower().endswith(VALID_EXTS)]  # Saring hanya file gambar valid
    count = len(img_files)                               # Hitung jumlah file gambar kelas ini
    total_images += count                                # Tambahkan ke total keseluruhan
    print(f"{c}: {count} images")                        # Cetak jumlah gambar per kelas

print("Total images:", total_images)                     # Cetak total seluruh gambar

bad_files = []                                           # List untuk menampung file rusak/corrupt

for c in classes:                                        # Loop setiap kelas untuk cek corrupt
    class_path = os.path.join(DATASET_DIR, c)            # Path folder kelas
    for fname in os.listdir(class_path):                 # Loop setiap file di folder kelas
        fpath = os.path.join(class_path, fname)          # Bentuk path lengkap file
        if not fname.lower().endswith(VALID_EXTS):       # Jika bukan file gambar valid
            continue                                      # Lewati file ini
        try:                                              # Coba buka/validasi gambar
            with Image.open(fpath) as img:               # Buka gambar
                img.verify()                              # Verifikasi integritas file gambar
        except Exception:                                 # Jika gagal dibuka/diverifikasi
            bad_files.append(fpath)                       # Tambahkan ke daftar file rusak

print("Broken files found:", len(bad_files))             # Cetak jumlah file rusak

for bf in bad_files[:10]:                                # Tampilkan maksimal 10 file rusak pertama
    print("Broken:", bf)                                 # Cetak path file rusak

# Jika kamu ingin hapus file rusak, buka komentar blok ini:
# for bf in bad_files:                                   # Loop semua file rusak
#     os.remove(bf)                                      # Hapus file rusak dari disk
# print("Broken files removed:", len(bad_files))         # Konfirmasi jumlah file yang dihapus

# =========================================================
# #opening 3 - Gunakan dataset langsung dari lokasi awal (tanpa copy)
# =========================================================

# Tetapkan folder kerja project untuk output/model (bukan untuk menyalin dataset)
WORK_DIR = "/content/skin_project"                                  # Folder kerja lokal untuk artifacts, model, dsb

# Tetapkan dataset lokal ke lokasi awal di Google Drive (tanpa dipindah/copy)
LOCAL_DATA_DIR = DATASET_DIR                                        # Dataset tetap dipakai langsung dari path Drive

# Jika folder kerja lokal sudah ada dari run sebelumnya, hapus agar bersih
if os.path.exists(WORK_DIR):                                        # Cek keberadaan folder kerja lokal
    shutil.rmtree(WORK_DIR)                                         # Hapus folder kerja lama

# Buat ulang folder kerja lokal
os.makedirs(WORK_DIR, exist_ok=True)                                # Buat folder kerja baru

# Verifikasi bahwa dataset path benar-benar ada
print("Using dataset directly from Drive:", LOCAL_DATA_DIR)         # Tampilkan path dataset yang dipakai
print("Dataset exists:", os.path.exists(LOCAL_DATA_DIR))            # Pastikan path valid (True)
print("Classes:", os.listdir(LOCAL_DATA_DIR))                       # Tampilkan folder kelas

# =========================================================
# #opening 4 - Membuat train/validation/test split manual (70/15/15)
# =========================================================

SPLIT_ROOT = os.path.join(WORK_DIR, "split_data")        # Root folder untuk data hasil split
TRAIN_DIR = os.path.join(SPLIT_ROOT, "train")            # Folder train split
VAL_DIR = os.path.join(SPLIT_ROOT, "val")                # Folder validation split
TEST_DIR = os.path.join(SPLIT_ROOT, "test")              # Folder test split

if os.path.exists(SPLIT_ROOT):                           # Jika folder split lama sudah ada
    shutil.rmtree(SPLIT_ROOT)                            # Hapus agar tidak tercampur dengan split lama

for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:         # Loop untuk ketiga folder split
    for c in classes:                                    # Loop untuk tiap kelas
        os.makedirs(os.path.join(split_dir, c), exist_ok=True)  # Buat subfolder kelas di tiap split

train_ratio = 0.70                                       # Rasio data train 70%
val_ratio = 0.15                                         # Rasio data val 15%
# test_ratio implicitly = 1 - train_ratio - val_ratio = 15%

for c in classes:                                        # Loop setiap kelas
    src = os.path.join(LOCAL_DATA_DIR, c)                # Path sumber folder kelas
    files = [f for f in os.listdir(src) if f.lower().endswith(VALID_EXTS)]  # Ambil daftar file gambar
    random.shuffle(files)                                # Acak urutan file agar split random

    n = len(files)                                       # Jumlah total file pada kelas ini
    n_train = int(n * train_ratio)                       # Hitung jumlah file train
    n_val = int(n * val_ratio)                           # Hitung jumlah file val
    n_test = n - n_train - n_val                         # Sisanya jadi test

    train_files = files[:n_train]                        # Slice file untuk train
    val_files = files[n_train:n_train + n_val]           # Slice file untuk val
    test_files = files[n_train + n_val:]                 # Slice file untuk test

    for f in train_files:                                # Copy file train
        shutil.copy2(os.path.join(src, f), os.path.join(TRAIN_DIR, c, f))  # Copy file + metadata
    for f in val_files:                                  # Copy file val
        shutil.copy2(os.path.join(src, f), os.path.join(VAL_DIR, c, f))    # Copy file + metadata
    for f in test_files:                                 # Copy file test
        shutil.copy2(os.path.join(src, f), os.path.join(TEST_DIR, c, f))   # Copy file + metadata

print("Split finished.")                                 # Konfirmasi split selesai

for split_name, split_path in [("TRAIN", TRAIN_DIR), ("VAL", VAL_DIR), ("TEST", TEST_DIR)]:  # Loop tiap split
    print(f"\n{split_name}")                             # Cetak nama split
    for c in classes:                                    # Loop tiap kelas
        count = len([f for f in os.listdir(os.path.join(split_path, c)) if f.lower().endswith(VALID_EXTS)])  # Hitung jumlah file
        print(f"{c}: {count}")                           # Cetak jumlah file per kelas di split ini

# =========================================================
# #opening 5 - Membuat tf.data Dataset dari folder split
# =========================================================

IMG_SIZE = (224, 224)                                    # Ukuran target gambar sesuai input MobileNetV2
BATCH_SIZE = 32                                          # Jumlah gambar per batch
AUTOTUNE = tf.data.AUTOTUNE                              # Biarkan TensorFlow optimasi pipeline data otomatis

train_ds = tf.keras.utils.image_dataset_from_directory(  # Buat dataset train dari folder
    TRAIN_DIR,                                           # Folder sumber train
    labels="inferred",                                   # Label diambil dari nama folder kelas
    label_mode="int",                                    # Label dalam bentuk integer
    image_size=IMG_SIZE,                                 # Resize gambar ke 224x224
    batch_size=BATCH_SIZE,                               # Batch size
    shuffle=True,                                        # Acak urutan data untuk training
    seed=SEED                                            # Seed agar urutan acak reproducible
)

val_ds = tf.keras.utils.image_dataset_from_directory(    # Buat dataset validation
    VAL_DIR,                                             # Folder sumber validation
    labels="inferred",                                   # Label dari nama folder
    label_mode="int",                                    # Label integer
    image_size=IMG_SIZE,                                 # Resize sama seperti train
    batch_size=BATCH_SIZE,                               # Batch size
    shuffle=False                                        # Tidak perlu shuffle validation
)

test_ds = tf.keras.utils.image_dataset_from_directory(   # Buat dataset test
    TEST_DIR,                                            # Folder sumber test
    labels="inferred",                                   # Label dari nama folder
    label_mode="int",                                    # Label integer
    image_size=IMG_SIZE,                                 # Resize sama
    batch_size=BATCH_SIZE,                               # Batch size
    shuffle=False                                        # Tidak perlu shuffle test
)

class_names = train_ds.class_names                       # Ambil nama kelas berdasarkan urutan internal dataset
num_classes = len(class_names)                           # Hitung jumlah kelas

print("Class names:", class_names)                       # Cetak nama kelas final
print("Number of classes:", num_classes)                 # Cetak jumlah kelas

# =========================================================
# #opening 6 - Visualisasi contoh gambar train
# =========================================================

for images, labels in train_ds.take(1):                  # Ambil satu batch pertama dari train dataset
    plt.figure(figsize=(12, 8))                          # Buat figure ukuran 12x8
    for i in range(min(12, len(images))):                # Tampilkan maksimal 12 gambar
        ax = plt.subplot(3, 4, i + 1)                    # Buat grid subplot 3 baris x 4 kolom
        plt.imshow(images[i].numpy().astype("uint8"))    # Tampilkan gambar i (konversi ke uint8 untuk visual)
        plt.title(class_names[int(labels[i].numpy())])   # Tampilkan label kelas gambar
        plt.axis("off")                                  # Sembunyikan sumbu agar lebih bersih
    plt.tight_layout()                                   # Rapikan layout
    plt.show()                                           # Render plot

# =========================================================
# #opening 7 - Data augmentation + preprocessing pipeline
# =========================================================

data_augmentation = keras.Sequential([                   # Buat pipeline augmentasi data
    layers.RandomFlip("horizontal"),                     # Flip horizontal acak
    layers.RandomRotation(0.05),                         # Rotasi kecil (±5%)
    layers.RandomZoom(0.10),                             # Zoom kecil (±10%)
], name="data_augmentation")

def prepare_dataset(ds, training=False):                 # Definisikan fungsi utilitas untuk siapkan dataset
    if training:                                         # Jika dataset untuk training
        ds = ds.map(                                     # Terapkan transform per batch
            lambda x, y: (data_augmentation(x, training=True), y),  # Augment gambar, label tetap
            num_parallel_calls=AUTOTUNE                 # Paralelisasi proses map
        )
    ds = ds.map(                                         # Terapkan preprocess MobileNetV2
        lambda x, y: (preprocess_input(x), y),          # Skala/transform pixel sesuai backbone pretrained
        num_parallel_calls=AUTOTUNE                     # Paralelisasi proses map
    )
    ds = ds.prefetch(AUTOTUNE)                           # Prefetch batch berikutnya saat GPU sedang training
    return ds                                            # Kembalikan dataset siap pakai

train_ds_prepared = prepare_dataset(train_ds, training=True)   # Siapkan dataset train (dengan augmentasi)
val_ds_prepared = prepare_dataset(val_ds, training=False)      # Siapkan dataset val (tanpa augmentasi)
test_ds_prepared = prepare_dataset(test_ds, training=False)    # Siapkan dataset test (tanpa augmentasi)

# =========================================================
# #opening 8 - Build model MobileNetV2 (transfer learning)
# =========================================================

base_model = MobileNetV2(                                # Inisialisasi backbone MobileNetV2
    input_shape=(224, 224, 3),                           # Shape input gambar (H, W, C)
    include_top=False,                                   # Jangan pakai classifier default ImageNet
    weights="imagenet"                                   # Muat bobot pretrained dari ImageNet
)

base_model.trainable = False                             # Freeze semua layer backbone pada tahap awal

inputs = keras.Input(shape=(224, 224, 3), name="input_image")  # Definisikan layer input model

x = base_model(inputs, training=False)                   # Forward input ke backbone dalam mode inference (bn stabil)
x = layers.GlobalAveragePooling2D(name="gap")(x)         # Ubah feature map 2D jadi vektor 1D
x = layers.Dropout(0.2, name="dropout")(x)               # Tambahkan dropout untuk kurangi overfitting
outputs = layers.Dense(num_classes, activation="softmax", name="classifier")(x)  # Layer klasifikasi final

model = keras.Model(inputs, outputs, name="skin_mobilenetv2_classifier")  # Gabungkan jadi model end-to-end

model.compile(                                           # Compile model sebelum training
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),# Optimizer Adam dengan LR awal 0.001
    loss="sparse_categorical_crossentropy",              # Loss untuk multiclass label integer
    metrics=["accuracy"]                                 # Pantau metrik akurasi
)

model.summary()                                          # Tampilkan ringkasan arsitektur model

# =========================================================
# #opening 9 - Training tahap 1 (feature extraction)
# =========================================================

callbacks_stage1 = [                                     # Definisikan callback untuk training tahap 1
    keras.callbacks.EarlyStopping(                       # Callback early stopping
        monitor="val_loss",                              # Pantau validation loss
        patience=4,                                      # Stop jika 4 epoch tidak membaik
        restore_best_weights=True                        # Kembalikan bobot terbaik otomatis
    ),
    keras.callbacks.ReduceLROnPlateau(                   # Callback turunkan LR saat stagnan
        monitor="val_loss",                              # Pantau validation loss
        factor=0.2,                                      # LR baru = LR lama * 0.2
        patience=2,                                      # Tunggu 2 epoch stagnan sebelum turun
        verbose=1                                        # Tampilkan log perubahan LR
    ),
    keras.callbacks.ModelCheckpoint(                     # Callback simpan model terbaik
        filepath="/content/best_stage1.keras",           # Path file model tahap 1
        monitor="val_accuracy",                          # Kriteria terbaik berdasarkan val_accuracy
        save_best_only=True,                             # Simpan hanya model terbaik
        verbose=1                                        # Tampilkan log saat save
    )
]

history_stage1 = model.fit(                              # Jalankan training tahap 1
    train_ds_prepared,                                   # Dataset train yang sudah di-augment + preprocess
    validation_data=val_ds_prepared,                     # Dataset validation
    epochs=12,                                           # Maksimal 12 epoch
    callbacks=callbacks_stage1                           # Pakai callback yang sudah didefinisikan
)

# =========================================================
# #opening 10 - Fine-tuning tahap 2 (unfreeze layer atas backbone)
# =========================================================

base_model.trainable = True                              # Aktifkan trainable seluruh backbone dulu

for layer in base_model.layers[:-30]:                    # Loop semua layer kecuali 30 layer terakhir
    layer.trainable = False                              # Freeze layer bawah (general features tetap stabil)

model.compile(                                           # Compile ulang WAJIB setelah ubah trainable
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),# Gunakan LR kecil saat fine-tuning
    loss="sparse_categorical_crossentropy",              # Loss tetap sama
    metrics=["accuracy"]                                 # Metrik tetap sama
)

callbacks_stage2 = [                                     # Definisikan callback untuk tahap 2
    keras.callbacks.EarlyStopping(                       # Early stopping lagi untuk mencegah overfit
        monitor="val_loss",                              # Pantau validation loss
        patience=4,                                      # Stop jika stagnan 4 epoch
        restore_best_weights=True                        # Restore bobot terbaik
    ),
    keras.callbacks.ModelCheckpoint(                     # Simpan model final terbaik
        filepath="/content/best_model_final.keras",      # Path model final
        monitor="val_accuracy",                          # Berdasarkan val_accuracy
        save_best_only=True,                             # Simpan yang terbaik saja
        verbose=1                                        # Log save model
    )
]

history_stage2 = model.fit(                              # Jalankan training tahap 2 (fine-tuning)
    train_ds_prepared,                                   # Dataset train
    validation_data=val_ds_prepared,                     # Dataset validation
    epochs=10,                                           # Maksimal 10 epoch fine-tuning
    callbacks=callbacks_stage2                           # Callback tahap 2
)

# =========================================================
# #opening 11 - Evaluasi model final di test set
# =========================================================

best_model = keras.models.load_model("/content/best_model_final.keras")  # Muat model final terbaik dari checkpoint

test_loss, test_acc = best_model.evaluate(test_ds_prepared, verbose=0)    # Evaluasi model pada test set

print(f"Test Loss    : {test_loss:.4f}")               # Cetak nilai test loss
print(f"Test Accuracy: {test_acc:.4f}")                # Cetak nilai test accuracy

y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)  # Kumpulkan seluruh label asli dari test_ds

y_prob = best_model.predict(test_ds_prepared, verbose=0)           # Prediksi probabilitas tiap kelas di test set
y_pred = np.argmax(y_prob, axis=1)                                 # Ambil indeks kelas dengan probabilitas terbesar

print("\nClassification Report:")                                   # Header report
print(classification_report(y_true, y_pred, target_names=class_names, digits=4))  # Cetak precision/recall/f1 per kelas

cm = confusion_matrix(y_true, y_pred)                               # Hitung confusion matrix

plt.figure(figsize=(7, 6))                                          # Buat figure
plt.imshow(cm, interpolation="nearest")                             # Tampilkan confusion matrix sebagai image
plt.title("Confusion Matrix (Test Set)")                            # Judul plot
plt.colorbar()                                                      # Tampilkan colorbar
ticks = np.arange(len(class_names))                                 # Posisi tick sesuai jumlah kelas
plt.xticks(ticks, class_names, rotation=45)                         # Label sumbu X (predicted classes)
plt.yticks(ticks, class_names)                                      # Label sumbu Y (true classes)
plt.xlabel("Predicted Label")                                       # Nama sumbu X
plt.ylabel("True Label")                                            # Nama sumbu Y
plt.tight_layout()                                                  # Rapikan layout
plt.savefig("/content/confusion_matrix_test.png", dpi=200)          # Simpan confusion matrix ke file PNG
plt.show()                                                          # Tampilkan plot di output

# =========================================================
# #opening 12 - Simpan model + metadata + history ke Drive
# =========================================================

ARTIFACT_DIR = "/content/artifacts"                                 # Folder artifacts lokal
os.makedirs(ARTIFACT_DIR, exist_ok=True)                            # Buat folder jika belum ada

final_model_local = os.path.join(ARTIFACT_DIR, "skin_disease_mobilenetv2.keras")  # Path model lokal
best_model.save(final_model_local)                                  # Simpan model final ke format .keras

class_json_local = os.path.join(ARTIFACT_DIR, "class_names.json")   # Path file class names lokal
with open(class_json_local, "w") as f:                              # Buka file JSON mode tulis
    json.dump(class_names, f)                                       # Simpan list class_names ke JSON

history_dict = {                                                    # Gabungkan history stage 1 dan stage 2
    "stage1": history_stage1.history,                               # Simpan metric/loss per epoch stage 1
    "stage2": history_stage2.history                                # Simpan metric/loss per epoch stage 2
}
history_json_local = os.path.join(ARTIFACT_DIR, "training_history.json")  # Path history lokal
with open(history_json_local, "w") as f:                            # Buka file JSON mode tulis
    json.dump(history_dict, f)                                      # Simpan history ke JSON

cm_local = "/content/confusion_matrix_test.png"                     # Path confusion matrix PNG lokal

DRIVE_SAVE_DIR = "/content/drive/MyDrive/G Colab File/Classification of skin diseases/artifacts"  # Folder tujuan di Drive
os.makedirs(DRIVE_SAVE_DIR, exist_ok=True)                          # Buat folder tujuan jika belum ada

shutil.copy2(final_model_local, DRIVE_SAVE_DIR)                     # Copy model final ke Drive
shutil.copy2(class_json_local, DRIVE_SAVE_DIR)                      # Copy class_names.json ke Drive
shutil.copy2(history_json_local, DRIVE_SAVE_DIR)                    # Copy training_history.json ke Drive
shutil.copy2(cm_local, DRIVE_SAVE_DIR)                              # Copy confusion matrix PNG ke Drive

print("Artifacts saved to Drive:", DRIVE_SAVE_DIR)                  # Konfirmasi lokasi penyimpanan artifact
print("Saved files:")                                                # Header daftar file
print("- skin_disease_mobilenetv2.keras")                            # Nama file model
print("- class_names.json")                                          # Nama file label mapping
print("- training_history.json")                                     # Nama file riwayat training
print("- confusion_matrix_test.png")                                 # Nama file confusion matrix