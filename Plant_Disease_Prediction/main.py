



import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="PlantHealth AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================= GLOBAL STYLES =================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800&family=Lora:wght@400;500;600;700&display=swap');

:root {
    --green-50:  #f0faf2;
    --green-100: #d6f5dc;
    --green-200: #aeeabb;
    --green-300: #76d48a;
    --green-400: #42b85c;
    --green-500: #2a9944;
    --green-600: #1e7a34;
    --green-700: #165c27;
    --mint:      #e8f7eb;
    --cream:     #fafdf8;
    --white:     #ffffff;
    --amber:     #f59e0b;
    --amber-lt:  #fef3c7;
    --blue:      #3b82f6;
    --blue-lt:   #eff6ff;
    --text-dark: #1a2e1c;
    --text-mid:  #3d5e42;
    --text-mute: #6b8f72;
    --border:    #c8e6cc;
    --shadow-sm: 0 1px 4px rgba(30,122,52,0.08);
    --shadow-md: 0 4px 16px rgba(30,122,52,0.12);
}

html, body, [class*="css"] {
    font-family: 'Nunito', sans-serif !important;
    color: var(--text-dark) !important;
}

.stApp {
    background: linear-gradient(155deg, #f0faf2 0%, #fafdf8 40%, #f5fbf6 100%) !important;
    min-height: 100vh;
}

/* ===== WIDER SIDEBAR ===== */
[data-testid="stSidebar"] {
    width: 320px !important;
    min-width: 320px !important;
    background: linear-gradient(180deg, #1e7a34 0%, #165c27 50%, #0f4520 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(30,122,52,0.2) !important;
}

[data-testid="stSidebar"] > div:first-child { padding: 0 !important; }

[data-testid="stSidebar"] .stSelectbox label {
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.72rem !important;
    font-weight: 600 !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.12) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 12px !important;
    color: white !important;
    font-weight: 600 !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div:hover {
    background: rgba(255,255,255,0.18) !important;
    border-color: rgba(255,255,255,0.35) !important;
}

/* ===== MAIN BLOCK ===== */
.main .block-container {
    padding: 2rem 2.5rem 3rem !important;
    max-width: 1100px !important;
}

h1, h2, h3 {
    font-family: 'Lora', serif !important;
    color: var(--text-dark) !important;
}

/* ===== BUTTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #2a9944, #1e7a34) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.65rem 1.8rem !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 3px 12px rgba(30,122,52,0.3) !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(30,122,52,0.4) !important;
    background: linear-gradient(135deg, #42b85c, #2a9944) !important;
}

/* ===== FILE UPLOADER ===== */
[data-testid="stFileUploader"] {
    background: var(--mint) !important;
    border: 2px dashed var(--green-300) !important;
    border-radius: 16px !important;
    padding: 2rem !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--green-500) !important;
    background: var(--green-100) !important;
}

/* ===== SUCCESS / WARNING ===== */
.stSuccess {
    background: var(--green-50) !important;
    border: 1px solid var(--green-300) !important;
    border-radius: 12px !important;
    color: var(--green-700) !important;
}

.stWarning {
    background: var(--amber-lt) !important;
    border: 1px solid #fcd34d !important;
    border-radius: 12px !important;
}

/* ===== EXPANDER ===== */
.streamlit-expanderHeader {
    background: var(--white) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-dark) !important;
    font-weight: 600 !important;
    font-family: 'Nunito', sans-serif !important;
}

.streamlit-expanderContent {
    background: var(--mint) !important;
    border: 1px solid var(--border) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
    color: var(--text-mid) !important;
}

/* ===== SELECTBOX ===== */
.stSelectbox > div > div {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text-dark) !important;
}

.stSelectbox > div > div:hover { border-color: var(--green-400) !important; }

/* ===== CHAT INPUT ===== */
.stChatInput > div {
    background: white !important;
    border: 2px solid var(--border) !important;
    border-radius: 20px !important;
    box-shadow: var(--shadow-sm) !important;
}

.stChatInput > div:focus-within {
    border-color: var(--green-400) !important;
    box-shadow: 0 0 0 3px rgba(66,184,92,0.15) !important;
}

/* ===== IMAGE ===== */
[data-testid="stImage"] img {
    border-radius: 20px !important;
    box-shadow: var(--shadow-md) !important;
    border: 3px solid var(--green-100) !important;
}

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #f0faf2; }
::-webkit-scrollbar-thumb { background: var(--green-300); border-radius: 3px; }

.stSpinner > div { border-top-color: var(--green-500) !important; }
</style>
""", unsafe_allow_html=True)


# ================= MODEL =================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('trained_model.h5')

model = load_model()

def model_prediction(test_image):
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    return np.argmax(prediction)

# ================= DISEASE LIST =================
class_name = [
    'Apple___Apple_scab','Apple___Black_rot','Apple___Cedar_apple_rust','Apple___healthy',
    'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy',
    'Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy',
    'Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy',
    'Potato___Early_blight','Potato___Late_blight','Potato___healthy',
    'Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch','Strawberry___healthy',
    'Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight',
    'Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy'
]

# ================= TREATMENT DATA =================
treatment_data = {
    "Apple___Apple_scab": ["Cause: Fungus due to wet conditions","Remove infected leaves","Apply sulfur fungicide","Avoid overhead watering"],
    "Apple___Black_rot": ["Cause: Fungal infection in warm weather","Remove dead fruit and branches","Use fungicide spray","Maintain tree hygiene"],
    "Apple___Cedar_apple_rust": ["Cause: Fungus from cedar trees","Remove nearby cedar plants","Use fungicide","Prune infected areas"],
    "Corn_(maize)___Common_rust_": ["Cause: Fungal spores in air","Use resistant seeds","Apply fungicide early","Maintain spacing"],
    "Corn_(maize)___Northern_Leaf_Blight": ["Cause: Moist environment fungus","Crop rotation","Use fungicide","Remove infected debris"],
    "Grape___Black_rot": ["Cause: Warm humid fungus","Remove infected grapes","Use fungicide spray","Improve airflow"],
    "Grape___Esca_(Black_Measles)": ["Cause: Fungal trunk disease","Prune infected wood","Avoid overwatering","Disinfect tools"],
    "Orange___Haunglongbing_(Citrus_greening)": ["Cause: Bacterial infection spread by insects","Control insects (psyllids)","Remove infected trees","Use certified plants"],
    "Peach___Bacterial_spot": ["Cause: Bacteria in wet weather","Use copper sprays","Avoid overhead watering","Remove infected leaves"],
    "Pepper,_bell___Bacterial_spot": ["Cause: Bacteria spread by water","Use disease-free seeds","Apply copper fungicide","Avoid touching wet plants"],
    "Potato___Early_blight": ["Cause: Fungus in soil","Crop rotation","Remove infected leaves","Apply fungicide"],
    "Potato___Late_blight": ["Cause: High humidity fungus","Remove infected plants","Use copper fungicide","Avoid waterlogging"],
    "Strawberry___Leaf_scorch": ["Cause: Fungal infection","Remove affected leaves","Improve airflow","Use fungicide spray"],
    "Tomato___Early_blight": ["Cause: Soil fungus","Remove infected leaves","Use mulch","Apply fungicide"],
    "Tomato___Late_blight": ["Cause: Moist fungal infection","Avoid wet leaves","Use fungicide","Remove infected plants"],
    "Tomato___Leaf_Mold": ["Cause: High humidity","Improve ventilation","Avoid excess watering","Use fungicide"],
    "Tomato___Septoria_leaf_spot": ["Cause: Fungal spores","Remove infected leaves","Avoid splashing water","Use fungicide"],
    "Tomato___Bacterial_spot": ["Cause: Bacteria spread by water","Avoid overhead watering","Use copper spray","Remove infected leaves"],
    "Tomato___Spider_mites Two-spotted_spider_mite": ["Cause: Tiny pests (mites)","Spray neem oil","Increase humidity","Wash leaves regularly"],
    "Tomato___Target_Spot": ["Cause: Fungal infection","Remove infected leaves","Use fungicide","Improve airflow"],
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": ["Cause: Virus spread by whiteflies","Control whiteflies","Remove infected plants","Use resistant seeds"],
    "Tomato___Tomato_mosaic_virus": ["Cause: Virus from contaminated tools","Disinfect tools","Remove infected plants","Avoid touching healthy plants"]
}

def get_treatment(disease):
    if disease in treatment_data:
        return treatment_data[disease]
    elif "healthy" in disease.lower():
        return ["Your plant is perfectly healthy","Keep your watering schedule consistent","Ensure 6-8 hours of sunlight daily","Use organic compost for nutrients"]
    else:
        return ["Disease not fully identified","Remove infected leaves immediately","Avoid overwatering","Consult a local agriculture expert"]

# ================= CHATBOT DATA =================
@st.cache_data
def load_chatbot():
    return pd.read_csv("chatbot_data.csv")

df = load_chatbot()

def get_greeting():
    hour = datetime.now().hour
    if hour < 12:
        return "Good Morning"
    elif hour < 18:
        return "Good Afternoon"
    else:
        return "Good Evening"

# ================= SIDEBAR =================
st.sidebar.markdown("""
<div style="background:linear-gradient(135deg,rgba(255,255,255,0.15),rgba(255,255,255,0.05));
            padding:2rem 1.8rem 1.5rem;border-bottom:1px solid rgba(255,255,255,0.15);">
    <div style="display:flex;align-items:center;gap:12px;margin-bottom:0.5rem;">
        <div style="width:46px;height:46px;background:rgba(255,255,255,0.2);border-radius:14px;
                    display:flex;align-items:center;justify-content:center;font-size:1.5rem;
                    box-shadow:0 2px 10px rgba(0,0,0,0.2);">🌿</div>
        <div>
            <div style="font-family:'Lora',serif;font-size:1.3rem;font-weight:700;color:white;">PlantHealth</div>
            <div style="font-size:0.68rem;color:rgba(255,255,255,0.6);letter-spacing:1.5px;
                        text-transform:uppercase;font-weight:600;">AI System</div>
        </div>
    </div>
</div>
<div style="padding:1.5rem 1.8rem 0.5rem;">
    <div style="font-size:0.68rem;color:rgba(255,255,255,0.5);letter-spacing:2px;
                text-transform:uppercase;font-weight:700;margin-bottom:0.8rem;">Navigation</div>
</div>
""", unsafe_allow_html=True)

nav_options = ["Home", "About", "Disease Recognition", "Chatbot", "Seasonal Care"]
nav_icons   = ["🏠", "ℹ️", "🔬", "🤖", "🌦️"]
nav_display = [f"{i}  {n}" for i, n in zip(nav_icons, nav_options)]

app_mode = st.sidebar.selectbox("", nav_display, label_visibility="collapsed")

st.sidebar.markdown("""
<div style="padding:1.2rem 1.8rem;">
    <div style="background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);
                border-radius:14px;padding:1.2rem;">
        <div style="font-size:0.68rem;color:rgba(255,255,255,0.55);letter-spacing:1px;
                    text-transform:uppercase;font-weight:700;margin-bottom:0.8rem;">Quick Stats</div>
        <div style="display:flex;flex-direction:column;gap:0.6rem;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:rgba(255,255,255,0.75);font-size:0.82rem;">Plant Classes</span>
                <span style="background:rgba(255,255,255,0.15);color:white;font-weight:700;
                             font-size:0.8rem;padding:2px 10px;border-radius:20px;">38</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:rgba(255,255,255,0.75);font-size:0.82rem;">Training Images</span>
                <span style="background:rgba(255,255,255,0.15);color:white;font-weight:700;
                             font-size:0.8rem;padding:2px 10px;border-radius:20px;">87K+</span>
            </div>
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <span style="color:rgba(255,255,255,0.75);font-size:0.82rem;">Model Type</span>
                <span style="background:rgba(255,255,255,0.15);color:white;font-weight:700;
                             font-size:0.8rem;padding:2px 10px;border-radius:20px;">CNN</span>
            </div>
        </div>
    </div>
</div>
<div style="padding:0 1.8rem 1rem;">
    <div style="background:rgba(66,184,92,0.2);border:1px solid rgba(66,184,92,0.35);
                border-radius:12px;padding:0.9rem 1rem;">
        <div style="color:rgba(255,255,255,0.9);font-size:0.8rem;line-height:1.5;font-weight:500;">
            💡 <b style="color:white;">Tip:</b> Upload a clear, well-lit photo of the leaf for best AI accuracy.
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# HOME
# =====================================================================
if "Home" in app_mode:

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e7a34 0%,#2a9944 55%,#42b85c 100%);
                border-radius:24px;padding:3rem 3.5rem;margin-bottom:2rem;
                box-shadow:0 8px 32px rgba(30,122,52,0.25);position:relative;overflow:hidden;">
        <div style="position:absolute;top:-30px;right:-30px;width:180px;height:180px;
                    background:rgba(255,255,255,0.07);border-radius:50%;"></div>
        <div style="font-family:'Lora',serif;font-size:2.6rem;font-weight:700;color:white;
                    line-height:1.2;margin-bottom:0.75rem;">🌿 Plant Health AI</div>
        <div style="color:rgba(255,255,255,0.88);font-size:1.05rem;max-width:500px;
                    line-height:1.7;margin-bottom:1.5rem;">
            Detect plant diseases instantly using AI. Upload a leaf photo and get diagnosis,
            treatment tips, and seasonal care guidance — all in one place.
        </div>
        <div style="display:flex;gap:0.7rem;flex-wrap:wrap;">
            <span style="background:rgba(255,255,255,0.2);color:white;border-radius:20px;
                         padding:0.3rem 1rem;font-size:0.82rem;font-weight:600;">🧬 38 Disease Classes</span>
            <span style="background:rgba(255,255,255,0.2);color:white;border-radius:20px;
                         padding:0.3rem 1rem;font-size:0.82rem;font-weight:600;">📊 87K+ Images Trained</span>
            <span style="background:rgba(255,255,255,0.2);color:white;border-radius:20px;
                         padding:0.3rem 1rem;font-size:0.82rem;font-weight:600;">⚡ Instant AI Results</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_img, col_text = st.columns([1, 1], gap="large")
    with col_img:
        st.image("home_page.jpeg", use_column_width=True)
    with col_text:
        st.markdown("""
        <div style="padding:0.5rem 0;">
            <div style="font-family:'Lora',serif;font-size:1.5rem;font-weight:600;
                        color:#1a2e1c;margin-bottom:0.8rem;">Why Use PlantHealth AI?</div>
            <p style="color:#3d5e42;line-height:1.8;font-size:0.93rem;">
                Every year, plant diseases cause billions in crop losses worldwide.
                Early detection is the key to saving your harvest. Our AI model, trained on
                over 87,000 annotated leaf images, can identify 38 disease categories
                across 14 common crops with high accuracy.
            </p>
            <div style="display:flex;flex-direction:column;gap:0.7rem;margin-top:1.2rem;">
                <div style="display:flex;align-items:center;gap:0.8rem;">
                    <div style="width:36px;height:36px;background:#d6f5dc;border-radius:10px;
                                display:flex;align-items:center;justify-content:center;flex-shrink:0;">📸</div>
                    <div style="color:#3d5e42;font-size:0.9rem;font-weight:600;">Upload any clear leaf photo</div>
                </div>
                <div style="display:flex;align-items:center;gap:0.8rem;">
                    <div style="width:36px;height:36px;background:#d6f5dc;border-radius:10px;
                                display:flex;align-items:center;justify-content:center;flex-shrink:0;">🧠</div>
                    <div style="color:#3d5e42;font-size:0.9rem;font-weight:600;">AI analyzes and identifies the disease</div>
                </div>
                <div style="display:flex;align-items:center;gap:0.8rem;">
                    <div style="width:36px;height:36px;background:#d6f5dc;border-radius:10px;
                                display:flex;align-items:center;justify-content:center;flex-shrink:0;">💊</div>
                    <div style="color:#3d5e42;font-size:0.9rem;font-weight:600;">Get instant treatment steps</div>
                </div>
                <div style="display:flex;align-items:center;gap:0.8rem;">
                    <div style="width:36px;height:36px;background:#d6f5dc;border-radius:10px;
                                display:flex;align-items:center;justify-content:center;flex-shrink:0;">🌱</div>
                    <div style="color:#3d5e42;font-size:0.9rem;font-weight:600;">Learn seasonal care best practices</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin:2rem 0 1rem;'></div>", unsafe_allow_html=True)

    st.markdown("""<div style="font-family:'Lora',serif;font-size:1.4rem;font-weight:600;
                color:#1a2e1c;margin-bottom:1.2rem;text-align:center;">How It Works</div>""", unsafe_allow_html=True)

    steps = [
        ("01","📷","Upload Photo","Take a clear photo of the infected leaf and upload it to the system."),
        ("02","🔍","AI Analysis","Our CNN model processes the image and identifies the disease pattern."),
        ("03","📋","Get Results","Receive the disease name, cause, and a step-by-step treatment plan."),
        ("04","🌿","Recover Plant","Follow the care guide and monitor your plant back to full health."),
    ]
    for col, (num, icon, title, desc) in zip(st.columns(4), steps):
        with col:
            st.markdown(f"""
            <div style="background:white;border:1.5px solid #c8e6cc;border-radius:18px;
                        padding:1.5rem 1.2rem;text-align:center;
                        box-shadow:0 2px 12px rgba(30,122,52,0.08);">
                <div style="font-size:0.65rem;font-weight:800;color:#42b85c;letter-spacing:2px;
                            text-transform:uppercase;margin-bottom:0.6rem;">Step {num}</div>
                <div style="font-size:2rem;margin-bottom:0.6rem;">{icon}</div>
                <div style="font-family:'Lora',serif;font-size:0.95rem;font-weight:600;
                            color:#1a2e1c;margin-bottom:0.5rem;">{title}</div>
                <div style="color:#6b8f72;font-size:0.8rem;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:2rem 0 1rem;'></div>", unsafe_allow_html=True)

    st.markdown("""<div style="font-family:'Lora',serif;font-size:1.4rem;font-weight:600;
                color:#1a2e1c;margin-bottom:1.2rem;">Explore Features</div>""", unsafe_allow_html=True)

    features = [
        ("🔬","Disease Detection","Upload a leaf image and get instant AI-powered disease diagnosis with treatment steps.","#e8f7eb","#2a9944"),
        ("🤖","AI Chatbot","Ask plant care questions and get expert-level answers powered by our knowledge base.","#eff6ff","#3b82f6"),
        ("🌦️","Seasonal Care","Tailored plant care guides for Summer, Rainy, and Winter seasons for 14 crops.","#fef3c7","#f59e0b"),
    ]
    for col, (icon, title, desc, bg, accent) in zip(st.columns(3), features):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border:1.5px solid {accent}30;border-radius:18px;
                        padding:1.8rem 1.5rem;box-shadow:0 2px 12px rgba(0,0,0,0.06);">
                <div style="font-size:2rem;margin-bottom:0.8rem;">{icon}</div>
                <div style="font-family:'Lora',serif;font-size:1rem;font-weight:600;
                            color:#1a2e1c;margin-bottom:0.5rem;">{title}</div>
                <div style="color:#3d5e42;font-size:0.85rem;line-height:1.6;">{desc}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown("<div style='margin:0.5rem 0;'></div>", unsafe_allow_html=True)
            st.button(f"Open {title}", key=f"feat_{title}")

    st.markdown("<div style='margin:1.5rem 0 0.5rem;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:white;border:1.5px solid #c8e6cc;border-radius:18px;
                padding:1.5rem 2rem;box-shadow:0 2px 12px rgba(30,122,52,0.08);">
        <div style="font-family:'Lora',serif;font-size:1.05rem;font-weight:600;
                    color:#1a2e1c;margin-bottom:0.8rem;">🌾 Supported Crops</div>
        <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🍎 Apple</span>
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🍅 Tomato</span>
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🥔 Potato</span>
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🌽 Corn</span>
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🍇 Grape</span>
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🍓 Strawberry</span>
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🍑 Peach</span>
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🍊 Orange</span>
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🫑 Bell Pepper</span>
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🍒 Cherry</span>
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🫐 Blueberry</span>
            <span style="background:#d6f5dc;color:#165c27;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.82rem;font-weight:600;">🍋 Raspberry</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='margin:1.5rem 0 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown("""<div style="font-family:'Lora',serif;font-size:1.4rem;font-weight:600;
                color:#1a2e1c;margin-bottom:1rem;">Frequently Asked Questions</div>""", unsafe_allow_html=True)

    faqs = [
        ("How does disease detection work?","Upload a plant leaf image. Our CNN model processes it through 128x128 pixel analysis and predicts from 38 disease categories trained on 87,000+ images."),
        ("Is this system accurate?","Our model achieves strong validation accuracy on the PlantVillage dataset. For critical decisions, cross-verify with a local agronomist."),
        ("Can I use this for all plants?","Currently 38 disease categories across 14 crops are supported. We are continuously expanding the dataset and model coverage."),
        ("What image quality do I need?","Use a clear, well-lit photo with the leaf filling most of the frame. Avoid blurry or heavily shadowed images for best results."),
    ]
    for q, a in faqs:
        with st.expander(q):
            st.write(a)

# =====================================================================
# ABOUT
# =====================================================================
elif "About" in app_mode:

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1e7a34,#2a9944);border-radius:24px;
                padding:3rem 3.5rem;margin-bottom:2.5rem;
                box-shadow:0 8px 32px rgba(30,122,52,0.25);position:relative;overflow:hidden;">
        <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
                    background:rgba(255,255,255,0.06);border-radius:50%;"></div>
        <div style="font-family:'Lora',serif;font-size:2.4rem;font-weight:700;
                    color:white;margin-bottom:0.6rem;">About PlantHealth AI</div>
        <p style="color:rgba(255,255,255,0.85);font-size:1rem;max-width:560px;line-height:1.8;margin:0;">
            Built to bridge the gap between modern AI and everyday farming.
            Our platform empowers farmers, gardeners, and researchers to detect
            crop diseases early and take decisive action.
        </p>
    </div>
    """, unsafe_allow_html=True)

    stats = [("87,000+","Training Images","#e8f7eb","#1e7a34"),("38","Disease Classes","#eff6ff","#3b82f6"),
             ("14","Crop Types","#fef3c7","#f59e0b"),("CNN","Model Architecture","#fef2f2","#f87171")]
    for col, (val, label, bg, color) in zip(st.columns(4), stats):
        with col:
            st.markdown(f"""
            <div style="background:{bg};border-radius:16px;padding:1.5rem;
                        text-align:center;border:1.5px solid {color}25;">
                <div style="font-family:'Lora',serif;font-size:1.8rem;font-weight:700;
                            color:{color};margin-bottom:0.2rem;">{val}</div>
                <div style="color:#6b8f72;font-size:0.82rem;font-weight:600;">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:2rem 0;'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns(2, gap="large")
    with col_l:
        st.markdown("""
        <div style="background:white;border:1.5px solid #c8e6cc;border-radius:18px;
                    padding:2rem;box-shadow:0 2px 12px rgba(30,122,52,0.08);margin-bottom:1.5rem;">
            <div style="font-size:1.8rem;margin-bottom:0.6rem;">📊</div>
            <div style="font-family:'Lora',serif;font-size:1.1rem;font-weight:600;
                        color:#1a2e1c;margin-bottom:1rem;border-bottom:1.5px solid #d6f5dc;
                        padding-bottom:0.6rem;">Dataset and Model</div>
            <div style="display:flex;flex-direction:column;gap:0;">
                <div style="padding:0.5rem 0;color:#3d5e42;font-size:0.88rem;border-bottom:1px solid #f0faf2;display:flex;gap:0.7rem;align-items:flex-start;">
                    <span style="color:#42b85c;font-weight:800;flex-shrink:0;">✓</span>87,000+ labeled plant images
                </div>
                <div style="padding:0.5rem 0;color:#3d5e42;font-size:0.88rem;border-bottom:1px solid #f0faf2;display:flex;gap:0.7rem;align-items:flex-start;">
                    <span style="color:#42b85c;font-weight:800;flex-shrink:0;">✓</span>38 distinct disease categories
                </div>
                <div style="padding:0.5rem 0;color:#3d5e42;font-size:0.88rem;border-bottom:1px solid #f0faf2;display:flex;gap:0.7rem;align-items:flex-start;">
                    <span style="color:#42b85c;font-weight:800;flex-shrink:0;">✓</span>Deep CNN architecture (128x128)
                </div>
                <div style="padding:0.5rem 0;color:#3d5e42;font-size:0.88rem;display:flex;gap:0.7rem;align-items:flex-start;">
                    <span style="color:#42b85c;font-weight:800;flex-shrink:0;">✓</span>Validated on real field samples
                </div>
            </div>
        </div>
        <div style="background:white;border:1.5px solid #c8e6cc;border-radius:18px;
                    padding:2rem;box-shadow:0 2px 12px rgba(30,122,52,0.08);">
            <div style="font-size:1.8rem;margin-bottom:0.6rem;">🎯</div>
            <div style="font-family:'Lora',serif;font-size:1.1rem;font-weight:600;
                        color:#1a2e1c;margin-bottom:0.8rem;border-bottom:1.5px solid #d6f5dc;padding-bottom:0.6rem;">Our Mission</div>
            <p style="color:#3d5e42;font-size:0.88rem;line-height:1.8;margin:0;">
                Reduce crop loss and improve plant health worldwide through accessible,
                intelligent technology that empowers farmers to make informed decisions
                in real-time. We believe every farmer deserves expert-level crop diagnostics.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div style="background:white;border:1.5px solid #c8e6cc;border-radius:18px;
                    padding:2rem;box-shadow:0 2px 12px rgba(30,122,52,0.08);margin-bottom:1.5rem;">
            <div style="font-size:1.8rem;margin-bottom:0.6rem;">✨</div>
            <div style="font-family:'Lora',serif;font-size:1.1rem;font-weight:600;
                        color:#1a2e1c;margin-bottom:1rem;border-bottom:1.5px solid #d6f5dc;padding-bottom:0.6rem;">Core Features</div>
            <div style="display:flex;flex-direction:column;gap:0;">
                <div style="padding:0.5rem 0;color:#3d5e42;font-size:0.88rem;border-bottom:1px solid #f0faf2;display:flex;gap:0.7rem;">
                    <span>🌿</span>Image-based disease detection
                </div>
                <div style="padding:0.5rem 0;color:#3d5e42;font-size:0.88rem;border-bottom:1px solid #f0faf2;display:flex;gap:0.7rem;">
                    <span>💊</span>Customized treatment guidance
                </div>
                <div style="padding:0.5rem 0;color:#3d5e42;font-size:0.88rem;border-bottom:1px solid #f0faf2;display:flex;gap:0.7rem;">
                    <span>🤖</span>PlantDoc AI chatbot
                </div>
                <div style="padding:0.5rem 0;color:#3d5e42;font-size:0.88rem;display:flex;gap:0.7rem;">
                    <span>🌦️</span>Seasonal care recommendations
                </div>
            </div>
        </div>
        <div style="background:linear-gradient(135deg,#f0faf2,#d6f5dc);border:1.5px solid #aeeabb;
                    border-radius:18px;padding:2rem;box-shadow:0 2px 12px rgba(30,122,52,0.08);">
            <div style="font-size:1.8rem;margin-bottom:0.6rem;">🛠️</div>
            <div style="font-family:'Lora',serif;font-size:1.1rem;font-weight:600;
                        color:#1a2e1c;margin-bottom:1rem;border-bottom:1.5px solid #aeeabb;padding-bottom:0.6rem;">Tech Stack</div>
            <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
                <span style="background:white;color:#1e7a34;border:1px solid #aeeabb;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.8rem;font-weight:700;">Python</span>
                <span style="background:white;color:#1e7a34;border:1px solid #aeeabb;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.8rem;font-weight:700;">TensorFlow</span>
                <span style="background:white;color:#1e7a34;border:1px solid #aeeabb;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.8rem;font-weight:700;">Streamlit</span>
                <span style="background:white;color:#1e7a34;border:1px solid #aeeabb;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.8rem;font-weight:700;">Keras</span>
                <span style="background:white;color:#1e7a34;border:1px solid #aeeabb;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.8rem;font-weight:700;">NumPy</span>
                <span style="background:white;color:#1e7a34;border:1px solid #aeeabb;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.8rem;font-weight:700;">Pandas</span>
                <span style="background:white;color:#1e7a34;border:1px solid #aeeabb;border-radius:20px;padding:0.3rem 0.9rem;font-size:0.8rem;font-weight:700;">PlantVillage</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================
# DISEASE RECOGNITION
# =====================================================================
elif "Disease" in app_mode:

    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="font-family:'Lora',serif;font-size:2rem;font-weight:700;color:#1a2e1c;">
            🔬 Disease Recognition
        </div>
        <div style="color:#6b8f72;font-size:0.92rem;margin-top:0.2rem;">
            Upload a clear photo of the affected leaf to get an instant AI diagnosis
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.markdown("""<div style="background:white;border:1.5px solid #c8e6cc;border-radius:20px;
                    padding:1.5rem;box-shadow:0 2px 12px rgba(30,122,52,0.08);margin-bottom:1rem;">
            <div style="font-size:0.7rem;font-weight:800;color:#42b85c;letter-spacing:2px;
                        text-transform:uppercase;margin-bottom:1rem;">📁 Upload Plant Image</div>""", unsafe_allow_html=True)
        test_image = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        if test_image:
            st.image(test_image, use_column_width=True)

        st.markdown("""
        <div style="background:#fef3c7;border:1px solid #fcd34d;border-radius:14px;
                    padding:1rem 1.2rem;margin-top:1rem;">
            <div style="font-weight:700;color:#92400e;font-size:0.85rem;margin-bottom:0.4rem;">
                📌 Tips for Best Results
            </div>
            <ul style="margin:0;padding-left:1.1rem;color:#78350f;font-size:0.82rem;line-height:1.8;">
                <li>Use natural daylight, avoid flash</li>
                <li>Focus on one leaf, fill the frame</li>
                <li>Capture both sides if possible</li>
                <li>Avoid blurry or shadowed photos</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""<div style="background:white;border:1.5px solid #c8e6cc;border-radius:20px;
                    padding:1.5rem;box-shadow:0 2px 12px rgba(30,122,52,0.08);">
            <div style="font-size:0.7rem;font-weight:800;color:#42b85c;letter-spacing:2px;
                        text-transform:uppercase;margin-bottom:1rem;">🧠 Analysis Results</div>""", unsafe_allow_html=True)

        if st.button("🔍 Analyze Plant Disease"):
            if test_image:
                with st.spinner("Analyzing your plant..."):
                    result_index = model_prediction(test_image)
                    disease = class_name[result_index]
                    is_healthy = "healthy" in disease.lower()
                    if is_healthy:
                        color, bg, border_c, label = "#166534","#f0fdf4","#bbf7d0","✅ Plant is Healthy"
                    else:
                        color, bg, border_c, label = "#92400e","#fffbeb","#fde68a","⚠️ Disease Detected"

                    disease_display = disease.replace("___"," → ").replace("_"," ")
                    st.markdown(f"""
                    <div style="background:{bg};border:2px solid {border_c};border-radius:14px;
                                padding:1.2rem 1.5rem;margin-bottom:1.2rem;">
                        <div style="font-size:0.75rem;font-weight:800;color:{color};letter-spacing:1px;margin-bottom:0.3rem;">{label}</div>
                        <div style="color:#1a2e1c;font-size:1rem;font-weight:700;font-family:'Lora',serif;">{disease_display}</div>
                    </div>""", unsafe_allow_html=True)

                    treatment = get_treatment(disease)
                    st.markdown("""<div style="font-size:0.7rem;font-weight:800;color:#42b85c;letter-spacing:2px;
                                text-transform:uppercase;margin-bottom:0.8rem;">💊 Treatment and Care Plan</div>""", unsafe_allow_html=True)
                    for i, step in enumerate(treatment):
                        ic = "🔎" if i == 0 else "✔"
                        st.markdown(f"""
                        <div style="display:flex;align-items:flex-start;gap:0.8rem;
                                    padding:0.65rem 0;border-bottom:1px solid #f0faf2;">
                            <span style="color:#42b85c;font-size:1rem;flex-shrink:0;">{ic}</span>
                            <span style="color:#3d5e42;font-size:0.88rem;line-height:1.5;">{step}</span>
                        </div>""", unsafe_allow_html=True)
            else:
                st.warning("Please upload a plant image first.")
        st.markdown("</div>", unsafe_allow_html=True)

# =====================================================================
# CHATBOT
# =====================================================================
elif "Chatbot" in app_mode:

    st.markdown("""
    <div style="display:flex;align-items:center;gap:1rem;margin-bottom:1.2rem;">
        <div style="width:52px;height:52px;background:linear-gradient(135deg,#2a9944,#1e7a34);
                    border-radius:16px;display:flex;align-items:center;justify-content:center;
                    font-size:1.6rem;box-shadow:0 4px 14px rgba(30,122,52,0.3);">🤖</div>
        <div>
            <div style="font-family:'Lora',serif;font-size:1.8rem;font-weight:700;color:#1a2e1c;line-height:1.1;">PlantDoc AI</div>
            <div style="color:#6b8f72;font-size:0.85rem;">Your intelligent plant health assistant</div>
        </div>
        <div style="margin-left:auto;background:#d6f5dc;border:1px solid #aeeabb;border-radius:20px;
                    padding:0.3rem 0.9rem;font-size:0.78rem;font-weight:700;color:#1e7a34;">
            🟢 Online
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-bottom:1rem;">
        <div style="font-size:0.7rem;font-weight:800;color:#6b8f72;letter-spacing:1.5px;
                    text-transform:uppercase;margin-bottom:0.6rem;">Quick Questions</div>
        <div style="display:flex;flex-wrap:wrap;gap:0.5rem;">
            <span style="background:#f0faf2;border:1.5px solid #c8e6cc;color:#1e7a34;border-radius:20px;
                         padding:0.35rem 0.9rem;font-size:0.82rem;font-weight:600;">🌿 How to water tomatoes?</span>
            <span style="background:#f0faf2;border:1.5px solid #c8e6cc;color:#1e7a34;border-radius:20px;
                         padding:0.35rem 0.9rem;font-size:0.82rem;font-weight:600;">🍃 Signs of overwatering</span>
            <span style="background:#f0faf2;border:1.5px solid #c8e6cc;color:#1e7a34;border-radius:20px;
                         padding:0.35rem 0.9rem;font-size:0.82rem;font-weight:600;">🌱 Best organic fertilizer</span>
            <span style="background:#f0faf2;border:1.5px solid #c8e6cc;color:#1e7a34;border-radius:20px;
                         padding:0.35rem 0.9rem;font-size:0.82rem;font-weight:600;">🦠 What causes leaf spots?</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if len(st.session_state.messages) == 0:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"{get_greeting()}! I'm PlantDoc AI — your personal plant health assistant. Ask me anything about plant diseases, care tips, watering schedules, or treatments."
        })

    st.markdown("""<div style="background:white;border:1.5px solid #c8e6cc;border-radius:20px;
                padding:1.5rem;min-height:380px;max-height:480px;overflow-y:auto;
                box-shadow:0 2px 16px rgba(30,122,52,0.08);">""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        is_user = msg["role"] == "user"
        if is_user:
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-end;margin-bottom:0.9rem;align-items:flex-end;gap:0.6rem;">
                <div style="background:linear-gradient(135deg,#2a9944,#1e7a34);color:white;
                            border-radius:18px 18px 4px 18px;padding:0.8rem 1.1rem;
                            max-width:72%;font-size:0.9rem;line-height:1.6;
                            box-shadow:0 2px 10px rgba(30,122,52,0.2);">{msg['content']}</div>
                <div style="width:32px;height:32px;background:#d6f5dc;border-radius:50%;
                            display:flex;align-items:center;justify-content:center;font-size:0.9rem;flex-shrink:0;">👤</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="display:flex;justify-content:flex-start;margin-bottom:0.9rem;align-items:flex-end;gap:0.6rem;">
                <div style="width:32px;height:32px;background:linear-gradient(135deg,#2a9944,#1e7a34);
                            border-radius:50%;display:flex;align-items:center;justify-content:center;
                            font-size:0.9rem;flex-shrink:0;">🤖</div>
                <div style="background:#f0faf2;border:1px solid #c8e6cc;color:#1a2e1c;
                            border-radius:18px 18px 18px 4px;padding:0.8rem 1.1rem;
                            max-width:72%;font-size:0.9rem;line-height:1.6;
                            box-shadow:0 1px 6px rgba(30,122,52,0.08);">{msg['content']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    st.markdown("<div style='margin:0.5rem 0;'></div>", unsafe_allow_html=True)

    user_input = st.chat_input("Ask PlantDoc AI about plant health, diseases, or care tips...")

    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        reply = None
        text = user_input.lower()

        if text in ["hi", "hello", "hey"]:
            reply = f"{get_greeting()}! Great to have you here. What plant health question can I help you with today?"
        elif text in ["bye", "goodbye"]:
            reply = "Goodbye! Happy gardening and healthy harvests to you!"
        else:
            for _, row in df.iterrows():
                if text in str(row["Question"]).lower():
                    reply = row["Answer"]
                    break

        if reply is None:
            reply = "I don't have a specific answer for that yet, but I'm learning! Try asking about specific diseases, watering, fertilization, or seasonal care."

        st.session_state.messages.append({"role": "assistant", "content": reply})
        st.rerun()

# =====================================================================
# SEASONAL CARE
# =====================================================================
elif "Seasonal" in app_mode:

    st.markdown("""
    <div style="margin-bottom:2rem;">
        <div style="font-family:'Lora',serif;font-size:2rem;font-weight:700;color:#1a2e1c;">
            🌦️ Seasonal Plant Care Guide
        </div>
        <div style="color:#6b8f72;font-size:0.92rem;margin-top:0.2rem;">
            Tailored care tips for every crop and season — keep your plants thriving all year round
        </div>
    </div>
    """, unsafe_allow_html=True)

    plant = st.selectbox("🌱 Select Your Crop", [
        "🍅 Tomato","🥔 Potato","🍎 Apple","🍇 Grape","🌽 Corn",
        "🍓 Strawberry","🌶️ Chilli","🧅 Onion","🧄 Garlic",
        "🥬 Spinach","🥦 Broccoli","🌿 Coriander","🫑 Bell Pepper","🍆 Brinjal"
    ])

    plant_key = plant.split(" ", 1)[1]

    seasonal_data = {
        "Tomato": {
            "Summer": ["Water twice daily — morning and evening","Provide partial shade during peak afternoon heat","Mulch the base to lock soil moisture","Stake plants to support heavy fruit load","Avoid midday watering to prevent leaf scorch"],
            "Rainy":  ["Reduce watering — let rain do the work","Ensure raised beds or proper drainage","Spray copper fungicide to prevent blight","Remove yellowing or water-damaged leaves","Avoid stagnant water around roots"],
            "Winter": ["Water every 2–3 days (soil retains moisture)","Ensure 6–8 hours of direct sunlight","Cover with frost cloth if temp drops below 10C","Prune suckers to focus energy on fruit","Use balanced NPK fertilizer monthly"]
        },
        "Potato": {
            "Summer": ["Keep soil consistently moist but not waterlogged","Mulch heavily to keep roots cool","Water at soil level — never on foliage","Harvest early if heat is extreme","Use shade nets in peak heat zones"],
            "Rainy":  ["Use raised beds to prevent waterlogging","Space plants 30cm apart for airflow","Apply systemic fungicide for late blight","Monitor for slugs and pests after rain","Avoid fertilizing right before heavy rain"],
            "Winter": ["Ideal growing season in most regions","Plant seed potatoes after first frost risk passes","Earth up soil around stems as they grow","Water moderately — check soil before watering","Harvest before ground freezes"]
        },
        "Apple": {
            "Summer": ["Deep water once a week at the base","Apply 4–6 inch mulch ring around trunk","Thin fruit clusters to 1 per cluster","Protect from apple scab with fungicide","Prune crossing branches for sunlight access"],
            "Rainy":  ["Spray preventive fungicide every 10–14 days","Ensure good soil drainage — avoid puddles","Remove fallen fruits to reduce disease spread","Stake young trees against wind","Prune dead or diseased wood promptly"],
            "Winter": ["Tree enters dormancy — ideal for pruning","Apply dormant oil spray for pest control","Protect young trees from frost cracks","Minimal watering — only if soil is bone dry","Plan next season's fertilization schedule"]
        },
        "Grape": {
            "Summer": ["Water deeply 1–2 times per week","Ensure full sun — at least 8 hours daily","Use strong trellis support for vines","Remove excess foliage for airflow","Apply potassium-rich fertilizer for fruit quality"],
            "Rainy":  ["Spray systemic fungicide every 7–10 days","Train vines upward to keep fruit off ground","Ensure row orientation allows wind flow","Avoid overhead irrigation","Remove damaged or moldy clusters promptly"],
            "Winter": ["Prune canes back to 2–3 buds per shoot","Remove all old leaves and debris","Protect trunk with burlap in harsh winters","Minimal watering until spring growth starts","Apply dormant fungicide spray in late winter"]
        },
        "Corn": {
            "Summer": ["Water 1–1.5 inches per week consistently","Full sun is essential — no shading","Side-dress with nitrogen fertilizer at knee-high","Maintain 30–40 cm plant spacing","Watch for corn borer — apply neem oil"],
            "Rainy":  ["Ensure well-drained furrows between rows","Avoid over-fertilizing — rain leaches nutrients","Monitor for gray leaf spot in humid weather","Stake tall plants if winds are strong","Harvest cobs before heavy rains if near maturity"],
            "Winter": ["Not suitable — corn needs warm soil (18C+)","Use greenhouse or polytunnel if needed","Prepare soil with compost for spring planting","Clear crop debris to reduce pest overwintering","Plan crop rotation: avoid same field next season"]
        },
        "Strawberry": {
            "Summer": ["Water daily in the morning at soil level","Provide shade cloth in extreme heat","Use straw mulch to retain moisture and cool roots","Harvest berries frequently to avoid rot","Pinch runners to focus plant energy on fruiting"],
            "Rainy":  ["Elevate with raised beds for drainage","Remove infected or rotten berries immediately","Apply copper fungicide for leaf diseases","Ensure good spacing — 30cm between plants","Cover with polytunnel during heavy downpours"],
            "Winter": ["Cover crowns with straw mulch for frost protection","Reduce watering to once a week","Ensure at least 4 hours of winter sunlight","Remove old leaves to prevent fungal buildup","Feed with balanced fertilizer in late winter"]
        },
        "Chilli": {
            "Summer": ["Water every day — chilli loves warmth but not drought","Mulch base to prevent rapid soil drying","Apply potassium-rich fertilizer for better fruiting","Stake tall chilli plants to prevent breaking","Watch for spider mites — spray neem oil weekly"],
            "Rainy":  ["Reduce watering — rain-fed is usually enough","Ensure excellent drainage — chilli hates wet feet","Spray copper fungicide for anthracnose prevention","Remove fruit that shows dark water-soaked spots","Avoid overhead irrigation during flowering"],
            "Winter": ["Protect from frost — move pots indoors if needed","Reduce watering to every 3–4 days","Ensure maximum sun exposure","Apply phosphorus fertilizer to boost root health","Harvest all remaining fruits before a hard frost"]
        },
        "Onion": {
            "Summer": ["Water every 4–5 days — onions are drought-tolerant","Stop watering 2 weeks before harvest","Ensure full sun — at least 8 hours","Watch for thrips — use reflective mulch","Bend tops when 50% have fallen for curing"],
            "Rainy":  ["Avoid waterlogging — use raised rows","Space 10–15cm apart for airflow","Apply fungicide for purple blotch disease","Harvest before continuous heavy rains","Store in dry ventilated space after harvest"],
            "Winter": ["Best planting season in many regions","Plant sets 2–3cm deep in loose fertile soil","Water lightly — overwintered onions need little water","Weed regularly — onions compete poorly","Mulch lightly to protect from frost"]
        },
        "Garlic": {
            "Summer": ["Harvest when 50% of leaves have turned brown","Cure in warm dry ventilated shade for 3–4 weeks","Do not water after mid-summer","Store dried bulbs in mesh bags in cool place","Plant new cloves in autumn for next year"],
            "Rainy":  ["Avoid planting in waterlogged soil","Use sandy loam for fast drainage","Reduce irrigation — rain provides enough moisture","Watch for white rot fungal disease","Harvest promptly before prolonged wet spells"],
            "Winter": ["Ideal planting season — plant cloves pointy-side up","Mulch with straw for root insulation","Water lightly every 2 weeks","Apply balanced fertilizer in early growth","Expect harvest in 8–9 months"]
        },
        "Spinach": {
            "Summer": ["Spinach bolts in heat — use shade cloth","Sow bolt-resistant varieties","Water frequently — every 1–2 days","Harvest young leaves before flowering","Grow in containers that can be moved to shade"],
            "Rainy":  ["Excellent season for spinach growth","Ensure good drainage to avoid root rot","Space plants 15cm apart","Harvest regularly to encourage new growth","Watch for slugs and snails in wet conditions"],
            "Winter": ["Best season — spinach loves cool weather","Water every 3–4 days","Sow in succession every 2 weeks for continuous harvest","Apply nitrogen-rich fertilizer for lush leaves","Can survive light frost — ideal for autumn sowing"]
        },
        "Broccoli": {
            "Summer": ["Broccoli is a cool-weather crop — avoid peak summer","Use shade nets if growing in warm weather","Water deeply 2–3 times per week","Watch for cabbage worms — apply BT spray","Harvest heads before they start to flower"],
            "Rainy":  ["Excellent growing conditions for broccoli","Ensure drainage — don't let heads sit in water","Apply balanced fertilizer after each rainfall","Watch for clubroot in acidic soils","Harvest promptly after heads form"],
            "Winter": ["Ideal planting season for most regions","Space plants 45cm apart for large heads","Water every 3–4 days","Protect from hard frost with row covers","Apply boron foliar spray to prevent hollow stem"]
        },
        "Coriander": {
            "Summer": ["Coriander bolts quickly — sow every 3 weeks","Keep soil moist — water daily","Grow in partial shade to slow bolting","Harvest outer leaves only to extend plant life","Use for seeds once plant bolts"],
            "Rainy":  ["Excellent growth season — coriander loves coolness","Ensure soil drainage — avoid soggy roots","Thin seedlings to 5–8cm spacing","Harvest heavily to prevent early bolting","Protect from heavy downpours with shade cloth"],
            "Winter": ["Best season for coriander in tropical climates","Sow in succession every 2 weeks","Water every 2–3 days","Full sun preferred — 5–6 hours minimum","Harvest before flowering for best flavor"]
        },
        "Bell Pepper": {
            "Summer": ["Water every day — peppers love warmth","Stake plants when 30cm tall","Apply potassium fertilizer for fruit quality","Use row covers to protect from extreme heat","Harvest frequently to encourage more production"],
            "Rainy":  ["Reduce watering significantly","Provide support against wind and rain","Spray fungicide for bacterial spot","Ensure good air circulation between plants","Remove any diseased or damaged fruits immediately"],
            "Winter": ["Bring potted peppers indoors if frost expected","Reduce watering to every 4–5 days","Ensure bright light — use grow light if needed","Fertilize lightly every 4 weeks","Prune to 2–3 main stems for compact growth"]
        },
        "Brinjal": {
            "Summer": ["Water daily — brinjal thrives in heat","Full sun is essential for fruit development","Stake tall varieties to prevent toppling","Apply balanced NPK fertilizer every 3 weeks","Pinch off first flower to encourage bushy growth"],
            "Rainy":  ["Reduce watering — monitor soil moisture","Improve drainage with raised mounds","Watch for phomopsis blight in humid conditions","Apply copper-based fungicide preventively","Harvest when skin is glossy — don't let overripen"],
            "Winter": ["Water every 3–4 days","Brinjal slows down in cool weather — protect from frost","Mulch roots for warmth retention","Apply phosphorus fertilizer for root strength","Harvest all fruits before first hard frost"]
        }
    }

    tips = seasonal_data.get(plant_key, {})

    st.markdown("<div style='margin:1rem 0;'></div>", unsafe_allow_html=True)

    season_configs = [
        ("Summer","☀️","#fffbeb","#f59e0b","#fef3c7","#92400e"),
        ("Rainy", "🌧️","#eff6ff","#3b82f6","#dbeafe","#1e40af"),
        ("Winter","❄️","#f0f9ff","#0ea5e9","#e0f2fe","#0c4a6e"),
    ]

    for col, (season, icon, bg, accent, header_bg, text_c) in zip(st.columns(3, gap="medium"), season_configs):
        with col:
            season_tips = tips.get(season, [])
            tips_html = "".join([
                f"<div style='display:flex;align-items:flex-start;gap:0.6rem;"
                f"padding:0.55rem 0;border-bottom:1px solid {bg};'>"
                f"<span style='color:{accent};flex-shrink:0;font-weight:800;font-size:0.85rem;'>✓</span>"
                f"<span style='color:#3d5e42;font-size:0.82rem;line-height:1.55;'>{tip}</span></div>"
                for tip in season_tips
            ])
            st.markdown(f"""
            <div style="background:white;border:1.5px solid {accent}40;border-radius:20px;
                        overflow:hidden;box-shadow:0 3px 14px rgba(0,0,0,0.06);">
                <div style="background:{header_bg};padding:1.2rem 1.5rem;border-bottom:1.5px solid {accent}30;">
                    <div style="font-size:1.6rem;margin-bottom:0.3rem;">{icon}</div>
                    <div style="font-family:'Lora',serif;font-size:1.05rem;font-weight:700;color:{text_c};">{season}</div>
                </div>
                <div style="padding:1rem 1.4rem;">{tips_html}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='margin:2rem 0 0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#f0faf2,#d6f5dc);border:1.5px solid #aeeabb;
                border-radius:18px;padding:1.5rem 2rem;">
        <div style="font-family:'Lora',serif;font-size:1rem;font-weight:600;
                    color:#1a2e1c;margin-bottom:0.8rem;">
            🌟 Universal Care Tips for {plant}
        </div>
        <div style="display:flex;flex-wrap:wrap;gap:0.7rem;">
            <span style="background:white;border:1px solid #aeeabb;color:#165c27;border-radius:12px;
                         padding:0.4rem 1rem;font-size:0.82rem;font-weight:600;">
                💧 Check soil moisture before every watering
            </span>
            <span style="background:white;border:1px solid #aeeabb;color:#165c27;border-radius:12px;
                         padding:0.4rem 1rem;font-size:0.82rem;font-weight:600;">
                🧪 Test soil pH every 3 months
            </span>
            <span style="background:white;border:1px solid #aeeabb;color:#165c27;border-radius:12px;
                         padding:0.4rem 1rem;font-size:0.82rem;font-weight:600;">
                🔍 Inspect leaves weekly for early disease signs
            </span>
            <span style="background:white;border:1px solid #aeeabb;color:#165c27;border-radius:12px;
                         padding:0.4rem 1rem;font-size:0.82rem;font-weight:600;">
                🌱 Rotate crops each season to prevent soil diseases
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)