import streamlit as st
import cnn_app
import mobile_app
import resnet_app
import os
import base64

# ============================= 
# PAGE CONFIG
# =============================
st.set_page_config(
    page_title="Dashboard Tugas Praktikum ML",
    layout="wide"
)

# =============================
# LOAD BACKGROUND IMAGE
# =============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BG_PATH = os.path.join(BASE_DIR, "background3.jpg")

def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg_image = get_base64_image(BG_PATH)

# =============================
# GLOBAL CSS
# =============================
st.markdown(
    f"""
    <style>

    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700;800&family=Poppins:wght@400;500;600&display=swap');

    /* ===== BACKGROUND ===== */
    .stApp {{
        background:
            linear-gradient(rgba(0,0,0,0.55), rgba(0,0,0,0.55)),
            url("data:image/png;base64,{bg_image}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }}

    /* ===== SAMAKAN JUDUL SUB PAGE ===== */
    div[data-testid="stTitle"] h1,
    div[data-testid="stHeader"] h2,
    h1, h2 {{
        font-family: 'Playfair Display', serif !important;
        font-size: 28px !important;
        font-weight: 800 !important;
        color: #FFC0D9 !important;
        letter-spacing: 1px;
    }}

    /* ===== HEADER HOME ===== */
    .header {{
        padding-top: 26px;
        padding-bottom: 18px;
        text-align: center;
    }}

    .header h1 {{
        font-size: 58px !important;
        margin: 0;
    }}

    .header p {{
        margin-top: 4px;
        font-size: 15.5px;
        color: #FFD1DC;
        font-family: 'Poppins', sans-serif;
    }}

    /* ===== HOME TITLE ===== */
    .home-title {{
        margin-top: 22px;
        margin-bottom: 32px;
        color: #FFD1DC;
        text-align: center;
        font-family: 'Playfair Display', serif;
        font-weight: 600;
    }}

    /* ===== PARAGRAPH ===== */
    p {{
        color: #F8D7E3;
        font-family: 'Poppins', sans-serif;
    }}

    /* ===== BOX ===== */
    .pink-box {{
        padding: 32px;
        border-radius: 22px;
        text-align: center;
        background: rgba(255, 255, 255, 0.18);
        box-shadow: 0 10px 25px rgba(0,0,0,0.35);
        transition: 0.35s ease;
    }}

    .pink-box:hover {{
        transform: translateY(-8px);
        box-shadow: 0 18px 45px rgba(155, 58, 90, 0.9);
    }}

    /* ===== BUTTON ===== */
    .stButton > button {{
        background-color: #DB7093;
        color: white;
        border-radius: 10px;
        padding: 6px 20px;
        border: none;
        font-weight: 600;
        font-size: 14px;
        margin-top: 12px;
        transition: 0.3s;
        font-family: 'Poppins', sans-serif;
    }}

    .stButton > button:hover {{
        background-color: #9B3A5A;
        transform: scale(1.04);
    }}

    footer {{
        visibility: hidden;
    }}

    </style>
    """,
    unsafe_allow_html=True
)

# =============================
# SESSION STATE
# =============================
if "page" not in st.session_state:
    st.session_state.page = "home"

# =============================
# HEADER
# =============================
st.markdown(
    """
    <div class="header">
        <h1>KLASIFIKASI MODEL CITRA</h1>
        <p>Dashboard Praktikum Machine Learning</p>
    </div>
    """,
    unsafe_allow_html=True
)

# =============================
# HOME PAGE
# =============================
if st.session_state.page == "home":

    st.markdown(
        "<h3 class='home-title'>Implementasi Model Citra</h3>",
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
            <div class="pink-box">
                <h4>Model Non-Pretrained (CNN)</h4>
                <p>Analisis gambar menggunakan CNN</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Klik", key="cnn"): 
            st.session_state.page = "cnn" 
            st.rerun()

    with col2:
        st.markdown("""
            <div class="pink-box">
                <h4>Model Pretrained 1 - MobileNetV2</h4>
                <p>Analisis gambar menggunakan MobileNetV2</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Klik", key="mobile"): 
            st.session_state.page = "mobile" 
            st.rerun()

    with col3:
        st.markdown("""
            <div class="pink-box">
                <h4>Model Pretrained 2 - ResNet101</h4>
                <p>Analisis gambar menggunakan ResNet101</p>
            </div>
        """, unsafe_allow_html=True)
        if st.button("Klik", key="resnet"): 
            st.session_state.page = "resnet" 
            st.rerun()

# =============================
# SUB PAGES
# =============================
elif st.session_state.page == "cnn": 
    cnn_app.run()
    if st.button("Kembali"):
        st.session_state.page = "home"
        st.rerun()

elif st.session_state.page == "mobile": 
    mobile_app.run()
    if st.button("Kembali"):
        st.session_state.page = "home"
        st.rerun()

elif st.session_state.page == "resnet": 
    resnet_app.run()
    if st.button("Kembali"):
        st.session_state.page = "home"
        st.rerun()

# =============================
# FOOTER
# =============================
st.markdown(
    """
    <div style="text-align:center; font-size:14px; color:#FFD1DC; padding:46px 0 28px 0;">
        🌸 Praktikum UAP Machine Learning 🌸<br>
        <b>Larasati Khadijah Kalimantari Karnain</b> | <b>202210370311410</b>
    </div>
    """,
    unsafe_allow_html=True
)
