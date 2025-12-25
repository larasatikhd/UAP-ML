def run():
    import streamlit as st
    import tensorflow as tf
    import os
    import numpy as np
    from PIL import Image
    import pandas as pd

    # ================================
    # HEADER
    # ================================
    st.header("🧘 Klasifikasi Citra Pose Yoga")

    # ================================
    # BASE DIR & MODEL PATH
    # ================================
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATHS = {
        "CNN": os.path.join(BASE_DIR, "models", "yoga_cnn_model.h5"),
        "MobileNetV2": os.path.join(BASE_DIR, "models", "model_mobilenetv2_yoga.h5"),
        "ResNet101": os.path.join(BASE_DIR, "models", "resnet101_model.keras"),
    }

    CLASS_NAMES = [
        'adho mukha svanasana', 'adho mukha vriksasana', 'agnistambhasana',
        'ananda balasana', 'anantasana', 'anjaneyasana', 'ardha bhekasana',
        'ardha chandrasana', 'ardha matsyendrasana', 'ardha pincha mayurasana',
        'ardha uttanasana', 'ashtanga namaskara', 'astavakrasana',
        'baddha konasana', 'bakasana', 'balasana', 'bhairavasana',
        'bharadvajasana i', 'bhekasana', 'bhujangasana', 'bhujapidasana',
        'bitilasana', 'camatkarasana', 'chakravakasana',
        'chaturanga dandasana', 'dandasana', 'dhanurasana', 'durvasasana',
        'dwi pada viparita dandasana', 'eka pada koundinyanasana i',
        'eka pada koundinyanasana ii', 'eka pada rajakapotasana',
        'eka pada rajakapotasana ii', 'ganda bherundasana',
        'garbha pindasana', 'garudasana', 'gomukhasana', 'halasana',
        'hanumanasana', 'janu sirsasana', 'kapotasana', 'krounchasana',
        'kurmasana', 'lolasana', 'makara adho mukha svanasana',
        'makarasana', 'malasana', 'marichyasana i', 'marichyasana iii',
        'marjaryasana', 'matsyasana', 'mayurasana', 'natarajasana',
        'padangusthasana', 'padmasana', 'parighasana',
        'paripurna navasana', 'parivrtta janu sirsasana',
        'parivrtta parsvakonasana', 'parivrtta trikonasana',
        'parsva bakasana', 'parsvottanasana', 'pasasana',
        'paschimottanasana', 'phalakasana', 'pincha mayurasana',
        'prasarita padottanasana', 'purvottanasana', 'salabhasana',
        'salamba bhujangasana', 'salamba sarvangasana',
        'salamba sirsasana', 'savasana', 'setu bandha sarvangasana',
        'simhasana', 'sukhasana', 'supta baddha konasana',
        'supta matsyendrasana', 'supta padangusthasana',
        'supta virasana', 'tadasana', 'tittibhasana', 'tolasana',
        'tulasana', 'upavistha konasana', 'urdhva dhanurasana',
        'urdhva hastasana', 'urdhva mukha svanasana',
        'urdhva prasarita eka padasana', 'ustrasana', 'utkatasana',
        'uttana shishosana', 'uttanasana',
        'utthita ashwa sanchalanasana',
        'utthita hasta padangustasana',
        'utthita parsvakonasana', 'utthita trikonasana',
        'vajrasana', 'vasisthasana', 'viparita karani',
        'virabhadrasana i', 'virabhadrasana ii',
        'virabhadrasana iii', 'virasana', 'vriksasana',
        'vrischikasana', 'yoganidrasana'
    ]

    # ================================
    # SIDEBAR
    # ================================
    st.sidebar.title("⚙️ Pengaturan")
    selected_model = st.sidebar.selectbox("Pilih Model", list(MODEL_PATHS.keys()))

    @st.cache_resource
    def load_model(model_path):
        tf.keras.backend.clear_session()
        return tf.keras.models.load_model(model_path)

    model = load_model(MODEL_PATHS[selected_model])

    # ================================
    # FILE UPLOADER
    # ================================
    uploaded_files = st.file_uploader(
        "Upload maksimal 3 gambar pose yoga (jpg/png)",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > 3:
            st.warning("Hanya 3 gambar pertama yang diproses.")
            uploaded_files = uploaded_files[:3]

        st.subheader("Preview Gambar")
        cols = st.columns(len(uploaded_files))
        for i, file in enumerate(uploaded_files):
            with cols[i]:
                img = Image.open(file)
                st.image(img, use_column_width=True)

        results = []

        for file in uploaded_files:
            try:
                img = Image.open(file).convert("RGB").resize((224, 224))
                arr = np.array(img)
                arr = np.expand_dims(arr, axis=0)

                if selected_model == "CNN":
                    arr = arr / 255.0
                elif selected_model == "MobileNetV2":
                    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
                else:
                    arr = tf.keras.applications.resnet.preprocess_input(arr)

                preds = model.predict(arr, verbose=0)[0]
                idx = np.argmax(preds)

                results.append({
                    "Nama File": file.name,
                    "Model": selected_model,
                    "Prediksi": CLASS_NAMES[idx],
                    "Confidence (%)": f"{preds[idx] * 100:.2f}"
                })

            except Exception as e:
                st.error(f"Gagal memproses {file.name}: {e}")

        if results:
            st.subheader("📊 Hasil Prediksi")
            df = pd.DataFrame(results)
            st.dataframe(df, use_container_width=True)

            st.subheader("📈 Grafik Confidence")
            chart_df = df[["Prediksi", "Confidence (%)"]]
            chart_df = chart_df.set_index("Prediksi")
            st.bar_chart(chart_df)

            st.success("✅ Prediksi selesai")

    st.markdown("---")
    st.markdown("**UAP Pembelajaran Mesin – Yoga Pose Classification**")

# ⬇️ INI WAJIB ADA
if __name__ == "__main__":
    run()