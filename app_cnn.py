import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, UnidentifiedImageError
import json

st.set_page_config(layout="wide")
st.title("🧠 Klasifikasi Sampah - CNN + Visualisasi")

# Load CNN model
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("klasifikasi_sampah_simple_CNN.h5")
    try:
        with open('class_names.json', 'r') as f:
            class_names = json.load(f)
    except FileNotFoundError:
        st.error("File class_names.json tidak ditemukan. Menggunakan fallback.")
        class_names = ['battery', 'cardboard', 'clothes', 'glass',
                       'metal', 'paper', 'plastic', 'shoes', 'trash']
    return model, class_names

model, class_names = load_model_and_classes()

# Layer visualisasi
visual_types = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.MaxPooling2D,
    tf.keras.layers.AveragePooling2D,
    tf.keras.layers.Dense,
    tf.keras.layers.Flatten,
    tf.keras.layers.Activation,
    tf.keras.layers.BatchNormalization
)
visual_layers = [(i, layer.name) for i, layer in enumerate(model.layers) if isinstance(layer, visual_types)]
if not visual_layers:
    st.error("❌ Tidak ada layer visual yang ditemukan.")
    st.stop()

layer_idx_tuple = st.sidebar.selectbox(
    "🔍 Pilih Layer CNN",
    visual_layers,
    format_func=lambda x: f"{x[1]} (index {x[0]})"
)
layer_idx = layer_idx_tuple[0]

# Upload gambar
uploaded_file = st.file_uploader("📤 Upload Gambar Sampah (.jpg, .jpeg)", type=["jpg", "jpeg"])
if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("❌ Gagal membaca file gambar.")
        st.stop()

    # ✅ Gambar statis: selalu ditampilkan dalam ukuran 300x300
    image_display = image.resize((300, 300))
    st.image(image_display, caption="🖼️ Gambar Diupload (ukuran tetap)", width=300)

    # Preprocessing
    resized = image.resize((224, 224))
    img_array_input = np.array(resized)
    img_array_normalized = img_array_input / 255.0
    img_array_expanded = np.expand_dims(img_array_normalized, axis=0)

    # Prediksi
    pred = model.predict(img_array_expanded)
    pred_class_idx = np.argmax(pred)
    pred_class = class_names[pred_class_idx]
    conf = np.max(pred)
    st.success(f"✅ Prediksi: **{pred_class}** ({conf * 100:.2f}%)")

    # Visualisasi feature map
    try:
        selected_layer_output = model.layers[layer_idx].output
        feature_model = tf.keras.Model(inputs=model.inputs, outputs=selected_layer_output)
        feature_map_batch = feature_model.predict(img_array_expanded)
        feature_map = feature_map_batch[0]
    except Exception as e:
        st.error(f"Gagal memproses feature map: {e}")
        st.stop()

    if len(feature_map.shape) == 3:
        num_channels = feature_map.shape[-1]
        st.subheader(f"🧱 Semua Channel - {model.layers[layer_idx].name}")
        cols = st.sidebar.slider("Jumlah kolom grid", 2, 16, 6)
        rows = (num_channels + cols - 1) // cols
        fig_grid, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.5))
        axes = axes.flatten()
        for i in range(min(num_channels, len(axes))):
            axes[i].imshow(feature_map[:, :, i], cmap='viridis')
            axes[i].axis("off")
            axes[i].set_title(f"Channel {i}")
        for j in range(num_channels, len(axes)):
            axes[j].axis("off")
        plt.tight_layout()
        st.pyplot(fig_grid)

        # 3D Scatter
        if num_channels > 1:
            st.subheader("🌐 3D Scatter Channel")
            ch_idx = st.sidebar.slider("Channel untuk Visualisasi 3D", 0, num_channels - 1, 0)
            fmap_3d = feature_map[:, :, ch_idx]
            x_pos, y_pos = np.meshgrid(range(fmap_3d.shape[1]), range(fmap_3d.shape[0]))
            fig_3d = plt.figure(figsize=(10, 6))
            ax = fig_3d.add_subplot(111, projection='3d')
            sc = ax.scatter(x_pos.flatten(), y_pos.flatten(), fmap_3d.flatten(),
                            c=fmap_3d.flatten(), cmap="plasma", s=10)
            ax.set_title(f"3D Feature Map - Channel {ch_idx}")
            fig_3d.colorbar(sc, shrink=0.5)
            st.pyplot(fig_3d)

        # Heatmap overlay
        st.subheader("🔥 Heatmap Overlay di Gambar Asli (Mean Activation)")
        heatmap = np.mean(feature_map, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        if np.max(heatmap) > 1e-6:
            heatmap /= np.max(heatmap)
        heatmap_img = np.uint8(255 * heatmap)
        heatmap_img = Image.fromarray(heatmap_img).resize(resized.size).convert("L")
        heatmap_array_color = plt.cm.jet(np.array(heatmap_img))[:, :, :3]
        overlay = 0.6 * (np.array(resized) / 255.0) + 0.4 * heatmap_array_color
        overlay = np.clip(overlay, 0, 1)
        fig_overlay = plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        st.pyplot(fig_overlay)

    elif len(feature_map.shape) == 1:
        st.subheader(f"📊 Aktivasi - {model.layers[layer_idx].name}")
        st.bar_chart(feature_map[:min(200, len(feature_map))])
    else:
        st.warning(f"❗ Layer {model.layers[layer_idx].name} tidak bisa divisualisasikan.")
