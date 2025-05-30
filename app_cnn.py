import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, UnidentifiedImageError
import io
import base64

st.set_page_config(layout="wide")
st.title("🧠 Klasifikasi Sampah - CNN + Visualisasi")

# Load CNN model
model = tf.keras.models.load_model("klasifikasi_sampah_simple_CNN.h5")
class_names = ['clothes', 'paper', 'plastic', 'battery', 'cardboard', 'shoes', 'glass', 'metal', 'trash']

# Ambil semua layer dari Sequential
base_layers = model.layers[0].layers if hasattr(model.layers[0], 'layers') else model.layers

# Filter visualizable layer
visual_types = (
    tf.keras.layers.Conv2D,
    tf.keras.layers.MaxPooling2D,
    tf.keras.layers.AveragePooling2D,
    tf.keras.layers.Dense,
    tf.keras.layers.Flatten,
    tf.keras.layers.Activation
)
visual_layers = [(i, layer.name) for i, layer in enumerate(base_layers) if isinstance(layer, visual_types)]

if not visual_layers:
    st.error("❌ Tidak ada layer visual yang ditemukan.")
    st.stop()

layer_idx = st.sidebar.selectbox("🔍 Pilih Layer CNN", visual_layers, format_func=lambda x: f"{x[1]} (index {x[0]})")

# Upload gambar hanya JPG
uploaded_file = st.file_uploader("📤 Upload Gambar Sampah (.jpg saja)", type=["jpg"])

if uploaded_file:
    if not uploaded_file.name.lower().endswith(".jpg"):
        st.warning("⚠️ File yang diunggah bukan JPG.")
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except UnidentifiedImageError:
        st.error("❌ Gagal membaca file gambar. Pastikan file berformat JPG.")
        st.stop()

    st.image(image, caption="🖼️ Gambar Diupload", use_container_width=True)

    # Preprocessing
    resized = image.resize((224, 224))
    img_array = np.array(resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi klasifikasi
    pred = model.predict(img_array)
    pred_class = class_names[np.argmax(pred)]
    conf = np.max(pred)
    st.success(f"✅ Prediksi: **{pred_class}** ({conf * 100:.2f}%)")

    # ---- Build feature model by manual pass ----
    input_tensor = tf.keras.Input(shape=(224, 224, 3))
    x = input_tensor
    for i in range(layer_idx[0] + 1):
        try:
            x = base_layers[i](x)
        except Exception as e:
            st.warning(f"⚠️ Gagal pada layer {base_layers[i].name}: {e}")
            break

    feature_model = tf.keras.Model(inputs=input_tensor, outputs=x)
    feature_map = feature_model.predict(img_array)[0]

    # ---- Visualisasi Grid ----
    if len(feature_map.shape) == 3:
        num_channels = feature_map.shape[-1]
        st.subheader(f"🧱 Semua Channel - {layer_idx[1]}")
        cols = 6
        rows = (num_channels + cols - 1) // cols
        fig_grid, axes = plt.subplots(rows, cols, figsize=(16, rows * 2.5))
        axes = axes.flatten()
        for i in range(num_channels):
            axes[i].imshow(feature_map[:, :, i], cmap='viridis')
            axes[i].axis("off")
            axes[i].set_title(f"Channel {i}")
        for j in range(num_channels, len(axes)):
            axes[j].axis("off")
        st.pyplot(fig_grid)

        # Download button
        buf = io.BytesIO()
        fig_grid.savefig(buf, format="png")
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode()
        st.markdown(f'<a href="data:image/png;base64,{b64}" download="cnn_feature_map_grid.png">📥 Download Grid</a>', unsafe_allow_html=True)

        # ---- 3D scatter ----
        st.subheader("🌐 3D Scatter Channel")
        ch_idx = st.sidebar.slider("Channel untuk Visualisasi 3D", 0, num_channels - 1, 0)
        fmap = feature_map[:, :, ch_idx]
        x_pos, y_pos = np.meshgrid(range(fmap.shape[1]), range(fmap.shape[0]))
        fig_3d = plt.figure(figsize=(10, 6))
        ax = fig_3d.add_subplot(111, projection='3d')
        sc = ax.scatter(x_pos.flatten(), y_pos.flatten(), fmap.flatten(), c=fmap.flatten(), cmap="plasma", s=10)
        ax.set_title(f"3D Feature Map - Channel {ch_idx}")
        fig_3d.colorbar(sc, shrink=0.5)
        st.pyplot(fig_3d)

        # ---- Heatmap Overlay ----
        st.subheader("🔥 Heatmap Overlay di Gambar Asli")
        heatmap = np.mean(feature_map, axis=-1)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap) + 1e-6
        heatmap_img = np.uint8(255 * heatmap)
        heatmap_img = Image.fromarray(heatmap_img).resize(resized.size).convert("L")
        heatmap_array = np.array(heatmap_img)
        heatmap_color = plt.cm.jet(heatmap_array)[:, :, :3]
        overlay = 0.6 * np.array(resized)/255.0 + 0.4 * heatmap_color
        overlay = np.clip(overlay, 0, 1)

        fig_overlay = plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis("off")
        st.pyplot(fig_overlay)

        buf_overlay = io.BytesIO()
        fig_overlay.savefig(buf_overlay, format="png")
        buf_overlay.seek(0)
        b64_overlay = base64.b64encode(buf_overlay.read()).decode()
        st.markdown(f'<a href="data:image/png;base64,{b64_overlay}" download="cnn_overlay.png">📥 Download Overlay</a>', unsafe_allow_html=True)

    elif len(feature_map.shape) in [1, 2]:
        st.subheader(f"📊 Aktivasi - {layer_idx[1]}")
        st.bar_chart(feature_map.flatten()[:200])
    else:
        st.warning("❗ Layer ini tidak bisa divisualisasikan.")
