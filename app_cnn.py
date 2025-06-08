import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, UnidentifiedImageError
import io
import base64
import json # Tambahkan ini

st.set_page_config(layout="wide")
st.title("🧠 Klasifikasi Sampah - CNN + Visualisasi")

# Load CNN model
@st.cache_resource # Cache model agar tidak load ulang terus
def load_model_and_classes():
    model = tf.keras.models.load_model("klasifikasi_sampah_simple_CNN.h5") # Pastikan path ini benar
    # Load class names dari file yang disimpan saat training
    try:
        with open('class_names.json', 'r') as f: # Pastikan file ini ada di direktori yang sama dengan app.py
            class_names = json.load(f)
    except FileNotFoundError:
        st.error("File class_names.json tidak ditemukan. Pastikan file ada dan berisi urutan kelas yang benar.")
        # Fallback jika file tidak ada, tapi ini BERISIKO jika urutannya salah
        class_names = ['battery', 'cardboard', 'clothes', 'glass',
         'metal', 'paper', 'plastic', 'shoes',
         'trash'] # SESUAIKAN DENGAN URUTAN TRAINING!
        st.warning(f"Menggunakan class_names default (fallback): {class_names}. Ini mungkin tidak akurat.")
    return model, class_names

model, class_names = load_model_and_classes()

# Ambil semua layer dari Sequential
base_layers = model.layers # Model .h5 biasanya langsung Sequential, bukan bersarang
# Filter visualizable layer
visual_types = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.MaxPooling2D,
    tf.keras.layers.AveragePooling2D,
    tf.keras.layers.Dense,
    tf.keras.layers.Flatten,
    tf.keras.layers.Activation,
    tf.keras.layers.BatchNormalization # Tambahkan BN jika ingin divisualisasikan
)
visual_layers = [(i, layer.name) for i, layer in enumerate(base_layers) if isinstance(layer, visual_types)]

if not visual_layers:
    st.error("❌ Tidak ada layer visual yang ditemukan.")
    st.stop()

layer_idx_tuple = st.sidebar.selectbox("🔍 Pilih Layer CNN", visual_layers, format_func=lambda x: f"{x[1]} (index {x[0]})")
layer_idx = layer_idx_tuple[0] # Ambil hanya indeks integer

# Upload gambar hanya JPG dan JPEG
uploaded_file = st.file_uploader("📤 Upload Gambar Sampah (.jpg, .jpeg)", type=["jpg", "jpeg"]) # Izinkan format lain

if uploaded_file:
    # if not uploaded_file.name.lower().endswith((".jpg", ".jpeg")):
    #     st.warning("⚠️ File yang diunggah bukan format gambar yang didukung.")
    #     st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("❌ Gagal membaca file gambar. Pastikan file berformat JPG, JPEG, atau PNG.")
        st.stop()

    st.image(image, caption="🖼️ Gambar Diupload", use_container_width=True)

    # Preprocessing - HARUS SAMA DENGAN TRAINING
    resized = image.resize((224, 224))
    img_array_input = np.array(resized)
    # Jika model Anda punya layer Rescaling(1./255), maka tidak perlu / 255.0 di sini
    # Jika tidak, maka lakukan normalisasi:
    img_array_normalized = img_array_input / 255.0
    img_array_expanded = np.expand_dims(img_array_normalized, axis=0)

    # Prediksi klasifikasi
    pred = model.predict(img_array_expanded)
    pred_class_idx = np.argmax(pred)
    pred_class = class_names[pred_class_idx]
    conf = np.max(pred)
    st.success(f"✅ Prediksi: **{pred_class}** ({conf * 100:.2f}%)")

    # ---- Model untuk visualisasi feature map ----
    # Dapatkan output dari layer yang dipilih
    try:
        selected_layer_output = model.layers[layer_idx].output
        feature_model = tf.keras.Model(inputs=model.inputs, outputs=selected_layer_output)
        feature_map_batch = feature_model.predict(img_array_expanded)
        feature_map = feature_map_batch[0] # Ambil feature map dari batch pertama (dan satu-satunya)
    except Exception as e:
        st.error(f"Gagal membuat model fitur untuk layer '{model.layers[layer_idx].name}': {e}")
        st.stop()


    # ---- Visualisasi Grid ----
    if len(feature_map.shape) == 3: # Conv2D, Pooling, BN setelah Conv
        num_channels = feature_map.shape[-1]
        st.subheader(f"🧱 Semua Channel - {visual_layers[layer_idx_tuple[0]][1]}") # Gunakan nama dari tuple
        cols = st.sidebar.slider("Jumlah kolom grid", 2, 16, 6)
        rows = (num_channels + cols - 1) // cols
        fig_grid, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.5))
        axes = axes.flatten() if num_channels > 1 else [axes] # Handle single channel case
        for i in range(num_channels):
            axes[i].imshow(feature_map[:, :, i], cmap='viridis')
            axes[i].axis("off")
            axes[i].set_title(f"Channel {i}")
        for j in range(num_channels, len(axes)):
            axes[j].axis("off")
        st.pyplot(fig_grid)
        # ... (kode download Anda) ...

        # ---- 3D scatter ----
        if num_channels > 1: # Hanya jika ada multiple channels
            st.subheader("🌐 3D Scatter Channel")
            ch_idx = st.sidebar.slider("Channel untuk Visualisasi 3D", 0, num_channels - 1, 0)
            fmap_3d = feature_map[:, :, ch_idx]
            x_pos, y_pos = np.meshgrid(range(fmap_3d.shape[1]), range(fmap_3d.shape[0]))
            fig_3d = plt.figure(figsize=(10, 6))
            ax = fig_3d.add_subplot(111, projection='3d')
            sc = ax.scatter(x_pos.flatten(), y_pos.flatten(), fmap_3d.flatten(), c=fmap_3d.flatten(), cmap="plasma", s=10)
            ax.set_title(f"3D Feature Map - Channel {ch_idx}")
            fig_3d.colorbar(sc, shrink=0.5)
            st.pyplot(fig_3d)
        # ... (kode visualisasi lainnya seperti heatmap overlay) ...
        # ---- Heatmap Overlay ----
        st.subheader("🔥 Heatmap Overlay di Gambar Asli (Mean Activation)")
        heatmap = np.mean(feature_map, axis=-1)
        heatmap = np.maximum(heatmap, 0) # ReLU-like
        if np.max(heatmap) > 1e-6: # Hindari division by zero
            heatmap /= np.max(heatmap)
        
        heatmap_img = np.uint8(255 * heatmap)
        heatmap_img = Image.fromarray(heatmap_img).resize(resized.size).convert("L") # L untuk grayscale
        
        heatmap_array_color = plt.cm.jet(np.array(heatmap_img))[:, :, :3] # Ambil RGB, buang alpha
        
        overlay = 0.6 * (np.array(resized) / 255.0) + 0.4 * heatmap_array_color
        overlay = np.clip(overlay, 0, 1)

        fig_overlay = plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        st.pyplot(fig_overlay)
        # ... (kode download Anda) ...


    elif len(feature_map.shape) == 1: # Dense, Flatten
        st.subheader(f"📊 Aktivasi - {visual_layers[layer_idx_tuple[0]][1]}")
        st.bar_chart(feature_map.flatten()[:min(200, len(feature_map.flatten()))]) # Batasi jumlah bar
    else:
        st.warning(f"❗ Layer {visual_layers[layer_idx_tuple[0]][1]} dengan shape {feature_map.shape} tidak bisa divisualisasikan dengan metode ini.")
